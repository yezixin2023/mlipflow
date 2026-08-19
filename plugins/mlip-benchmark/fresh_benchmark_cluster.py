#!/usr/bin/env python3
"""Run one approved fresh benchmark against site-owned model/data roots."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
import types
from pathlib import Path, PurePosixPath
from typing import Any


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


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("relative_path must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("relative_path must stay below its site root")
    return value


def _reference(path: Path, id_key: str) -> dict[str, str]:
    raw = _load_mapping(path)
    artifact_id = raw.get(id_key)
    fingerprint = raw.get("fingerprint")
    if raw.get("schema_version") != 1 or not isinstance(artifact_id, str) or not artifact_id:
        raise ValueError(f"{path.name} requires schema_version=1 and {id_key}")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 71
        or not fingerprint.startswith("sha256:")
        or any(char not in "0123456789abcdef" for char in fingerprint[7:])
    ):
        raise ValueError(f"{path.name} requires a full SHA-256 fingerprint")
    kind = raw.get("kind")
    if kind not in {"file", "directory"}:
        raise ValueError(f"{path.name} kind must be file or directory")
    result = {
        "id": artifact_id,
        "relative_path": _safe_relative(raw.get("relative_path")),
        "kind": kind,
        "fingerprint": fingerprint,
    }
    if isinstance(raw.get("framework"), str):
        result["framework"] = raw["framework"]
    return result


def _resolve(root: Path, reference: dict[str, str]) -> Path:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"site artifact root does not exist: {root}")
    candidate = (root / reference["relative_path"]).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("artifact reference escapes its site root") from exc
    if reference["kind"] == "file" and not candidate.is_file():
        raise ValueError(f"referenced file does not exist: {candidate}")
    if reference["kind"] == "directory" and not candidate.is_dir():
        raise ValueError(f"referenced directory does not exist: {candidate}")
    if candidate.is_symlink() or (
        candidate.is_dir() and any(item.is_symlink() for item in candidate.rglob("*"))
    ):
        raise ValueError("referenced artifact contains a symbolic link")
    return candidate


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
    artifact = _module(
        "mlipflow.science.artifact_identity", input_dir / "artifact_identity.py"
    )
    runtime = _module("mlipflow.science.model_runtime", input_dir / "model_runtime.py")
    science.artifact_identity = artifact
    science.model_runtime = runtime
    mlipflow.science = science
    return artifact, runtime


def _project_node(project: Path, node_id: str) -> dict[str, Any]:
    raw = _load_mapping(project)
    workflow = raw.get("workflow")
    nodes = workflow.get("nodes") if isinstance(workflow, dict) else None
    matches = [
        item
        for item in nodes or []
        if isinstance(item, dict) and item.get("id") == node_id
    ]
    if len(matches) != 1:
        raise ValueError(f"project must contain exactly one node {node_id!r}")
    return matches[0]


def _write_report(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    report: dict[str, Any] = {"schema_version": 1, "status": "FAIL", "node_id": args.node_id}
    try:
        artifact, _ = _science_modules(input_dir)
        fresh = _module("mlipflow_fresh_benchmark", input_dir / "fresh_benchmark.py")
        node = _project_node(Path(args.project).expanduser().resolve(), args.node_id)
        parameters = node.get("parameters")
        if not isinstance(parameters, dict) or parameters.get("operation") != "evaluate-fresh":
            raise ValueError("benchmark node must declare operation=evaluate-fresh")
        family = parameters.get("model_family")
        framework = fresh.MODEL_FRAMEWORKS.get(family)
        if framework is None:
            raise ValueError("unsupported exact model family")
        model_reference = _reference(input_dir / "model-reference.json", "model_id")
        dataset_reference = _reference(
            input_dir / "benchmark-dataset-reference.json", "dataset_id"
        )
        if model_reference.get("framework", framework) != framework:
            raise ValueError("model reference framework differs from the exact family")
        declared_model_fingerprint = parameters.get("model_fingerprint")
        declared_dataset_fingerprint = parameters.get("dataset_fingerprint")
        if (
            declared_model_fingerprint is not None
            and model_reference["fingerprint"] != declared_model_fingerprint
        ):
            raise ValueError("model reference fingerprint differs from the approved request")
        if (
            declared_dataset_fingerprint is not None
            and dataset_reference["fingerprint"] != declared_dataset_fingerprint
        ):
            raise ValueError("dataset reference fingerprint differs from the approved request")
        model_path = _resolve(Path(args.model_root), model_reference)
        dataset_path = _resolve(Path(args.data_root), dataset_reference)
        observed_model = artifact.fingerprint_path(model_path)
        observed_dataset = artifact.fingerprint_path(dataset_path)
        if observed_model != model_reference["fingerprint"]:
            raise ValueError("site model content differs from its reference")
        if observed_dataset != dataset_reference["fingerprint"]:
            raise ValueError("site benchmark dataset differs from its reference")
        targets = parameters.get("targets")
        units = parameters.get("units")
        if not isinstance(targets, list) or not isinstance(units, dict):
            raise ValueError("benchmark targets and units are required")
        outputs = fresh.evaluate_fresh(
            model=model_path,
            dataset=dataset_path,
            output_dir=output_dir,
            model_family=str(family),
            model_fingerprint=observed_model,
            dataset_fingerprint=observed_dataset,
            task=str(parameters.get("task")),
            scenario=str(parameters.get("scenario")),
            split=str(parameters.get("split")),
            targets=targets,
            units=units,
            energy_normalization=str(parameters.get("energy_normalization")),
            stress_convention=parameters.get("stress_convention"),
            device=str(parameters.get("device", "cpu")),
        )
        report.update(
            {
                "status": "OK",
                "return_code": 0,
                "exact_model_family": family,
                "framework": framework,
                "model": {**model_reference, "observed_fingerprint": observed_model},
                "dataset": {**dataset_reference, "observed_fingerprint": observed_dataset},
                "outputs": {
                    name: artifact.fingerprint_path(path) for name, path in outputs.items()
                },
            }
        )
        _write_report(report_path, report)
        return 0
    except Exception as exc:
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc(limit=8)
        _write_report(report_path, report)
        return 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--project", required=True)
    value.add_argument("--node-id", required=True)
    value.add_argument("--input-dir", required=True)
    value.add_argument("--output-dir", required=True)
    value.add_argument("--report", required=True)
    value.add_argument("--model-root", required=True)
    value.add_argument("--data-root", required=True)
    return value


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
