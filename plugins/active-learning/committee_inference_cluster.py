#!/usr/bin/env python3
"""Run approved committee inference against site-owned immutable models."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import traceback
import types
from pathlib import Path, PurePosixPath
from typing import Any


PREDICTION_CONTRACT = "mlipflow/active-learning-committee-predictions"
DATASET_CONTRACT = "mlipflow/active-learning-evaluation-dataset"
MODEL_INDEX_CONTRACT = "mlipflow/active-learning-committee-model-index"


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        value = yaml.safe_load(text)
    else:
        value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("relative_path must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("relative_path must stay below the site model root")
    return value


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load staged module: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _science_modules(input_dir: Path):
    mlipflow = types.ModuleType("mlipflow")
    science = types.ModuleType("mlipflow.science")
    mlipflow.__path__ = []
    science.__path__ = []
    sys.modules["mlipflow"] = mlipflow
    sys.modules["mlipflow.science"] = science
    runtime = _module("mlipflow.science.model_runtime", input_dir / "model_runtime.py")
    active = _module(
        "mlipflow.science.active_learning", input_dir / "active_learning_science.py"
    )
    science.model_runtime = runtime
    science.active_learning = active
    mlipflow.science = science
    return runtime, active


def _project_node(project: Path, node_id: str) -> dict[str, Any]:
    raw = _load_mapping(project)
    workflow = raw.get("workflow")
    nodes = workflow.get("nodes") if isinstance(workflow, dict) else None
    matches = [
        node
        for node in nodes or []
        if isinstance(node, dict) and node.get("id") == node_id
    ]
    if len(matches) != 1:
        raise ValueError(f"project must contain exactly one node {node_id!r}")
    return matches[0]


def _resolve_model(root: Path, member: dict[str, Any]) -> Path:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("site model root does not exist")
    relative = _safe_relative(member.get("relative_path"))
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("model reference escapes the site model root") from exc
    kind = member.get("kind")
    if kind == "file" and not candidate.is_file():
        raise ValueError(f"referenced model file does not exist: {relative}")
    if kind == "directory" and not candidate.is_dir():
        raise ValueError(f"referenced model directory does not exist: {relative}")
    if kind not in {"file", "directory"}:
        raise ValueError("model reference kind must be file or directory")
    return candidate


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _matrix(value: Any, rows: int, columns: int, field: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{field} must contain {rows} rows")
    result = []
    for index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != columns:
            raise ValueError(f"{field}[{index}] must contain {columns} values")
        result.append([_finite(item, f"{field}[{index}]") for item in row])
    return result


def _sample(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("evaluation samples must be objects")
    sample_id = raw.get("sample_id")
    split = raw.get("split")
    species = raw.get("species")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("evaluation sample requires sample_id")
    if split not in {"calibration", "candidate"}:
        raise ValueError(f"{sample_id}.split must be calibration or candidate")
    if not isinstance(species, list) or not species or any(
        not isinstance(symbol, str) or not symbol for symbol in species
    ):
        raise ValueError(f"{sample_id}.species must be a non-empty string array")
    natoms = len(species)
    pbc = raw.get("pbc")
    if not isinstance(pbc, bool) and not (
        isinstance(pbc, list)
        and len(pbc) == 3
        and all(isinstance(item, bool) for item in pbc)
    ):
        raise ValueError(f"{sample_id}.pbc must be boolean or length three")
    sample = {
        "id": sample_id,
        "sample_id": sample_id,
        "split": split,
        "species": list(species),
        "natoms": natoms,
        "positions": _matrix(raw.get("positions"), natoms, 3, f"{sample_id}.positions"),
        "cell": _matrix(raw.get("cell"), 3, 3, f"{sample_id}.cell"),
        "pbc": pbc,
    }
    if split == "calibration":
        sample["reference_forces"] = _matrix(
            raw.get("reference_forces"), natoms, 3, f"{sample_id}.reference_forces"
        )
    else:
        for key in (
            "condition_id",
            "condition",
            "replica",
            "frame_index",
            "structure_path",
            "near_duplicate_group",
            "physical_validity",
        ):
            sample[key] = raw.get(key)
    return sample


def _prediction_record(sample: dict[str, Any], prediction: Any) -> dict[str, Any]:
    if not isinstance(prediction, dict):
        raise ValueError(f"model returned no prediction for {sample['id']}")
    record = {
        "sample_id": sample["id"],
        "split": sample["split"],
        "atom_count": sample["natoms"],
        "forces": _matrix(
            prediction.get("force"), sample["natoms"], 3, f"{sample['id']}.force"
        ),
        "energy": _finite(prediction.get("energy"), f"{sample['id']}.energy"),
    }
    if sample["split"] == "candidate":
        record.update(
            {
                key: sample[key]
                for key in (
                    "condition_id",
                    "condition",
                    "replica",
                    "frame_index",
                    "structure_path",
                    "near_duplicate_group",
                    "physical_validity",
                )
            }
        )
    return record


def run(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "FAIL",
        "node_id": args.node_id,
    }
    try:
        runtime, active = _science_modules(input_dir)
        node = _project_node(Path(args.project).expanduser().resolve(), args.node_id)
        parameters = node.get("parameters")
        if not isinstance(parameters, dict) or parameters.get("operation") != "committee-evaluate":
            raise ValueError("active-learning node must declare operation=committee-evaluate")
        policy_path = input_dir / "policy.json"
        index_path = input_dir / "committee-model-index.json"
        dataset_path = input_dir / "evaluation-dataset.json"
        policy = _load_mapping(policy_path)
        policy_info = active.validate_policy(policy)
        index = _load_mapping(index_path)
        dataset = _load_mapping(dataset_path)
        if index.get("schema_version") != 1 or index.get("contract") != MODEL_INDEX_CONTRACT:
            raise ValueError("committee model index uses an unsupported contract")
        if dataset.get("schema_version") != 1 or dataset.get("contract") != DATASET_CONTRACT:
            raise ValueError("evaluation dataset uses an unsupported contract")
        if index.get("strategy") != policy.get("strategy"):
            raise ValueError("committee model index strategy differs from policy")
        if dataset.get("dataset_split") is None:
            raise ValueError("evaluation dataset requires dataset_split")
        units = dataset.get("units")
        if not isinstance(units, dict) or units.get("force") != policy_info["force_unit"]:
            raise ValueError("evaluation dataset force unit differs from policy")
        if units.get("energy") != "eV" or units.get("force") != "eV/angstrom":
            raise ValueError("committee runtime requires eV and eV/angstrom evaluation units")

        samples = [_sample(value) for value in dataset.get("samples", [])]
        sample_ids = [sample["id"] for sample in samples]
        if not samples or len(sample_ids) != len(set(sample_ids)):
            raise ValueError("evaluation dataset requires unique non-empty samples")
        split = dataset["dataset_split"]
        calibration_ids = set(split.get("calibration_ids", []))
        observed_calibration = {
            sample["id"] for sample in samples if sample["split"] == "calibration"
        }
        if calibration_ids != observed_calibration:
            raise ValueError("evaluation calibration samples differ from dataset_split")

        models = index.get("models")
        if not isinstance(models, list):
            raise ValueError("committee model index requires models")
        if {model.get("model_id") for model in models if isinstance(model, dict)} != set(
            policy_info["models"]
        ):
            raise ValueError("committee model index models differ from policy")

        prediction_models = []
        observed_models = []
        for model in models:
            if not isinstance(model, dict):
                raise ValueError("committee model records must be objects")
            model_id = model.get("model_id")
            family = model.get("model_family")
            framework = runtime.MODEL_FAMILY_FRAMEWORKS.get(family)
            if framework is None or model.get("framework") != framework:
                raise ValueError(f"{model_id} exact family/framework is unsupported")
            members = model.get("members")
            if not isinstance(members, list):
                raise ValueError(f"{model_id} requires committee members")
            seeds = [member.get("seed") for member in members if isinstance(member, dict)]
            if set(seeds) != set(policy_info["seeds"][model_id]):
                raise ValueError(f"{model_id} member seeds differ from policy")
            prediction_members = []
            for member in members:
                if not isinstance(member, dict):
                    raise ValueError(f"{model_id} member must be an object")
                model_path = _resolve_model(Path(args.model_root), member)
                predictor = runtime.load_inference_predictor(
                    model_path, str(family), str(parameters.get("device", "cpu"))
                )
                if predictor.prediction_units.get("energy") != "eV" or predictor.prediction_units.get(
                    "force"
                ) != "eV/angstrom":
                    raise ValueError("committee predictor units differ from the active-learning contract")
                predictions = [
                    _prediction_record(sample, predictor.predict(sample)) for sample in samples
                ]
                prediction_members.append(
                    {
                        "member_id": member.get("member_id"),
                        "seed": member.get("seed"),
                        "predictions": predictions,
                    }
                )
                observed_models.append(
                    {
                        "model_id": model_id,
                        "member_id": member.get("member_id"),
                        "seed": member.get("seed"),
                        "relative_path": _safe_relative(member.get("relative_path")),
                        "kind": member.get("kind"),
                    }
                )
            prediction_models.append(
                {
                    "model_id": model_id,
                    "framework": framework,
                    "supports_stress": bool(model.get("supports_stress", False)),
                    "members": prediction_members,
                }
            )

        predictions = {
            "schema_version": 1,
            "contract": PREDICTION_CONTRACT,
            "strategy": policy["strategy"],
            "units": {"energy": "eV", "force": "eV/angstrom"},
            "dataset_split": dataset["dataset_split"],
            "calibration_labels": [
                {"sample_id": sample["id"], "forces": sample["reference_forces"]}
                for sample in samples
                if sample["split"] == "calibration"
            ],
            "models": prediction_models,
        }
        predictions_path = output_dir / "committee-predictions.json"
        evaluation_path = output_dir / "committee-evaluation.json"
        _write_json(predictions_path, predictions)
        evaluation = active.evaluate_committee(predictions, policy)
        evaluation["validation_claim"] = "FRESH_ACTIVE_LEARNING_ROUND"
        evaluation["input_paths"] = {
            "committee_predictions": str(predictions_path),
            "policy": str(policy_path),
        }
        _write_json(evaluation_path, evaluation)
        report.update(
            {
                "status": "OK",
                "return_code": 0,
                "policy_id": policy_info["policy_id"],
                "models": observed_models,
                "sample_count": len(samples),
                "calibration_count": len(calibration_ids),
                "candidate_count": len(samples) - len(calibration_ids),
                "outputs": [str(predictions_path), str(evaluation_path)],
            }
        )
        _write_json(report_path, report)
        return 0
    except Exception as exc:
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc(limit=8)
        _write_json(report_path, report)
        return 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--project", required=True)
    value.add_argument("--node-id", required=True)
    value.add_argument("--input-dir", required=True)
    value.add_argument("--output-dir", required=True)
    value.add_argument("--report", required=True)
    value.add_argument("--model-root", required=True)
    return value


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
