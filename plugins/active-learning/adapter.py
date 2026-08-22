"""MLIPFlow adapter for deterministic offline active-learning decisions."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from mlipflow.science import active_learning as active_science
from mlipflow.science import model_runtime


PLUGIN_ID = "active-learning"
RUNNER = Path(globals().get("__file__", "adapter.py")).absolute().with_name(
    "active_learning.py"
)
CLUSTER_RUNNER = RUNNER.with_name("committee_inference_cluster.py")
SHARED_MODEL_RUNTIME = Path(model_runtime.__file__).resolve()
SHARED_ACTIVE_SCIENCE = Path(active_science.__file__).resolve()
OPERATIONS = ("committee-evaluate", "select-candidates", "assess-round")
EXECUTION_BACKENDS = frozenset({"local", "ssh-slurm"})
HPC_RESOURCES = {"cpus", "gpus", "memory", "walltime"}
MAX_JSON_BYTES = 128 * 1024 * 1024
MODEL_INDEX_CONTRACT = "mlipflow/active-learning-committee-model-index"
EVALUATION_DATASET_CONTRACT = "mlipflow/active-learning-evaluation-dataset"
REQUIRED_INPUTS = {
    "committee-evaluate": ("committee_predictions", "policy"),
    "select-candidates": ("committee_evaluation", "candidate_manifest", "policy"),
    "assess-round": (
        "committee_evaluation",
        "selection_result",
        "audit_benchmark",
        "spot_checks",
        "labeling_result",
        "dataset_split",
        "round_history",
        "campaign",
        "policy",
    ),
}
OPTIONAL_INPUTS = {
    "committee-evaluate": (),
    "select-candidates": ("direct_selection",),
    "assess-round": ("transport_evidence",),
}
DEFAULT_RESULTS = {
    "committee-evaluate": "committee-evaluation.json",
    "select-candidates": "selection-result.json",
    "assess-round": "round-assessment.json",
}


def _diagnostic(level: str, code: str, message: str) -> dict[str, str]:
    return {"level": level, "code": code, "message": message}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _plain(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and "\x00" not in value
        and "\n" not in value
        and "\r" not in value
    )


def _reference(value: Any) -> str | None:
    if _plain(value):
        return str(value)
    if isinstance(value, Mapping) and _plain(value.get("path")):
        return str(value["path"])
    return None


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _input_path(context: Mapping[str, Any], value: Any) -> Path:
    reference = _reference(value)
    if reference is None:
        raise ValueError("input requires a path")
    root = Path(str(context["project_root"])).expanduser().absolute().resolve()
    candidate = Path(reference).expanduser()
    candidate = candidate if candidate.is_absolute() else root / candidate
    candidate = candidate.absolute()
    if candidate.is_symlink() or not candidate.is_file() or not _within(candidate.resolve(), root):
        raise ValueError(f"input must be an ordinary file below project_root: {reference}")
    return candidate.resolve()


def _result_path(context: Mapping[str, Any], operation: str) -> Path:
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute().resolve()
    parameters = _mapping(context.get("parameters"))
    value = parameters.get("result_manifest", DEFAULT_RESULTS[operation])
    if not _plain(value):
        raise ValueError("parameters.result_manifest must be a safe relative path")
    relative = Path(str(value))
    if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
        raise ValueError("parameters.result_manifest must stay below attempt_dir")
    result = (attempt / relative).absolute()
    if not _within(result, attempt):
        raise ValueError("parameters.result_manifest escapes attempt_dir")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _project_file(root: Path, selected: Any = None) -> Path:
    if isinstance(selected, str) and selected:
        path = Path(selected).expanduser().absolute()
        if path.is_file() and not path.is_symlink() and path.resolve().parent == root.resolve():
            return path.resolve()
        raise ValueError("selected project file is invalid")
    found = [
        root / name
        for name in ("project.yaml", "project.yml", "project.json")
        if (root / name).is_file() and not (root / name).is_symlink()
    ]
    if len(found) != 1:
        raise ValueError("scheduled committee inference requires exactly one project file")
    return found[0].resolve()


def _safe_relative(value: Any, field: str) -> str:
    if not _plain(value):
        raise ValueError(f"{field} must be a non-empty relative POSIX path")
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts or "\\" in str(value):
        raise ValueError(f"{field} must stay below the site model root")
    return str(value)


def _staged(path: Path, remote_name: str) -> dict[str, Any]:
    return {
        "source": str(path),
        "remote_name": remote_name,
        "sensitive": False,
        "fetch_allowed": False,
    }


def _fetch(
    remote_name: str, remote_path: str, local_name: str, role: str
) -> dict[str, Any]:
    return {
        "remote_name": remote_name,
        "remote_path": remote_path,
        "local_name": local_name,
        "required": True,
        "max_bytes": MAX_JSON_BYTES,
        "role": role,
    }


def _scheduled_inputs(
    context: Mapping[str, Any],
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    root = Path(str(context["project_root"])).expanduser().absolute().resolve()
    inputs = _mapping(context.get("inputs"))
    allowed = {"policy", "committee_model_index", "evaluation_dataset"}
    unknown = sorted(set(inputs) - allowed)
    if unknown:
        raise ValueError("scheduled committee inference does not accept: " + ", ".join(unknown))
    policy_path = _input_path(context, inputs.get("policy"))
    index_path = _input_path(context, inputs.get("committee_model_index"))
    dataset_path = _input_path(context, inputs.get("evaluation_dataset"))
    project = _project_file(root, context.get("project_path"))
    policy = _runner_module().load_mapping(policy_path)
    policy_info = active_science.validate_policy(policy)
    index = _read_json(index_path)
    dataset = _read_json(dataset_path)
    if index.get("schema_version") != 1 or index.get("contract") != MODEL_INDEX_CONTRACT:
        raise ValueError("committee model index uses an unsupported contract")
    if dataset.get("schema_version") != 1 or dataset.get("contract") != EVALUATION_DATASET_CONTRACT:
        raise ValueError("evaluation dataset uses an unsupported contract")
    if index.get("strategy") != policy.get("strategy"):
        raise ValueError("committee model index strategy differs from policy")
    models = index.get("models")
    if not isinstance(models, list) or {
        model.get("model_id") for model in models if isinstance(model, dict)
    } != set(policy_info["models"]):
        raise ValueError("committee model index models differ from policy")
    approved_models = []
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("committee model records must be objects")
        model_id = model.get("model_id")
        family = model.get("model_family")
        framework = model_runtime.MODEL_FAMILY_FRAMEWORKS.get(family)
        if framework is None or model.get("framework") != framework:
            raise ValueError(f"{model_id} exact family/framework is unsupported")
        expected_kind = "directory" if framework == "m3gnet" else "file"
        members = model.get("members")
        if not isinstance(members, list) or len(members) < 2:
            raise ValueError(f"{model_id} requires at least two committee members")
        seeds = []
        approved_members = []
        for member in members:
            if not isinstance(member, dict):
                raise ValueError(f"{model_id} member must be an object")
            seed = member.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError(f"{model_id} member seed must be an integer")
            if member.get("kind") != expected_kind:
                raise ValueError(f"{model_id} member model kind must be {expected_kind}")
            seeds.append(seed)
            approved_members.append(
                {
                    "member_id": member.get("member_id"),
                    "seed": seed,
                    "relative_path": _safe_relative(
                        member.get("relative_path"), f"{model_id}.relative_path"
                    ),
                    "kind": member["kind"],
                }
            )
        if set(seeds) != set(policy_info["seeds"][model_id]):
            raise ValueError(f"{model_id} member seeds differ from policy")
        approved_models.append(
            {
                "model_id": model_id,
                "model_family": family,
                "framework": framework,
                "members": approved_members,
            }
        )
    samples = dataset.get("samples")
    split = dataset.get("dataset_split")
    if not isinstance(samples, list) or not samples or not isinstance(split, dict):
        raise ValueError("evaluation dataset requires samples and dataset_split")
    sample_ids = [sample.get("sample_id") for sample in samples if isinstance(sample, dict)]
    if len(sample_ids) != len(samples) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("evaluation dataset sample ids must be unique")
    calibration_ids = split.get("calibration_ids")
    if not isinstance(calibration_ids, list) or not calibration_ids:
        raise ValueError("evaluation dataset requires calibration_ids")
    if set(calibration_ids) != {
        sample.get("sample_id")
        for sample in samples
        if isinstance(sample, dict) and sample.get("split") == "calibration"
    }:
        raise ValueError("evaluation calibration samples differ from dataset_split")
    resources = context.get("resources")
    if not isinstance(resources, Mapping) or set(resources) != HPC_RESOURCES:
        raise ValueError("scheduled committee resources must be cpus, gpus, memory, walltime")
    for path in (
        CLUSTER_RUNNER,
        SHARED_MODEL_RUNTIME,
        SHARED_ACTIVE_SCIENCE,
    ):
        if not path.is_file():
            raise ValueError(f"scheduled committee bundled file is missing: {path.name}")
    calculation = {
        "policy_id": policy_info["policy_id"],
        "strategy": policy["strategy"],
        "models": approved_models,
        "sample_count": len(samples),
        "calibration_count": len(calibration_ids),
        "candidate_count": len(samples) - len(calibration_ids),
    }
    return policy_path, index_path, dataset_path, project, calculation


def _plan_scheduled(context: Mapping[str, Any]) -> dict[str, Any]:
    policy, index, dataset, project, calculation = _scheduled_inputs(context)
    result = _result_path(context, "committee-evaluate")
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute().resolve()
    result_relative = result.relative_to(attempt).as_posix()
    staged = [
        _staged(project, "project.yaml"),
        _staged(policy, "policy.json"),
        _staged(index, "committee-model-index.json"),
        _staged(dataset, "evaluation-dataset.json"),
        _staged(CLUSTER_RUNNER, "committee_inference_cluster.py"),
        _staged(SHARED_MODEL_RUNTIME, "model_runtime.py"),
        _staged(SHARED_ACTIVE_SCIENCE, "active_learning_science.py"),
    ]
    fetch_outputs = [
        _fetch(
            "committee-predictions.json",
            "output/active-learning/committee-predictions.json",
            "committee-predictions.json",
            "committee-predictions",
        ),
        _fetch(
            "committee-evaluation.json",
            "output/active-learning/committee-evaluation.json",
            result_relative,
            "committee-evaluation",
        ),
        _fetch(
            "cluster-active-learning-report.json",
            "output/cluster-active-learning-report.json",
            "cluster-active-learning-report.json",
            "active-learning-cluster-report",
        ),
    ]
    member_count = sum(len(model["members"]) for model in calculation["models"])
    return {
        "plugin_id": PLUGIN_ID,
        "status": "READY",
        "executable": True,
        "scheduler_contract": "bundled-active-learning-committee-v1",
        "argv": ["template-family:active-learning-committee-canonical"],
        "cwd": "remote-attempt-workspace",
        "shell": False,
        "expected_outputs": [item["remote_name"] for item in fetch_outputs],
        "committee": calculation,
        "input_paths": {
            "policy": str(policy),
            "committee_model_index": str(index),
            "evaluation_dataset": str(dataset),
        },
        "approval_summary": {
            "expensive": True,
            "submits_jobs": True,
            "execution_model": "single-python",
            "committee_count": len(calculation["models"]),
            "member_model_count": member_count,
            "sample_count": calculation["sample_count"],
            "calibration_count": calculation["calibration_count"],
            "candidate_count": calculation["candidate_count"],
            "model_execution": True,
        },
        "scheduled_execution": {
            "schema_version": 3,
            "execution_model": "single-python",
            "template_family": "active-learning-committee-canonical",
            "staged_files": staged,
            "fetch_outputs": fetch_outputs,
        },
        "diagnostics": [],
    }


def _verify_cluster_report(context: Mapping[str, Any], result_path: Path) -> None:
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute().resolve()
    predictions_path = attempt / "committee-predictions.json"
    report_path = attempt / "cluster-active-learning-report.json"
    if not predictions_path.is_file() or not report_path.is_file():
        raise ValueError("scheduled committee prediction/report artifacts are incomplete")
    report = _read_json(report_path)
    plan = _mapping(_mapping(context.get("execution")).get("plan"))
    calculation = _mapping(plan.get("committee"))
    if (
        report.get("schema_version") != 1
        or report.get("status") != "OK"
        or report.get("return_code") != 0
        or report.get("policy_id") != calculation.get("policy_id")
        or report.get("sample_count") != calculation.get("sample_count")
        or report.get("calibration_count") != calculation.get("calibration_count")
        or report.get("candidate_count") != calculation.get("candidate_count")
    ):
        raise ValueError("cluster committee report did not record approved success")
    approved_members = {
        (model["model_id"], member["member_id"]): member
        for model in calculation.get("models", [])
        for member in model.get("members", [])
    }
    observed_members = {
        (item.get("model_id"), item.get("member_id")): item
        for item in report.get("models", [])
        if isinstance(item, dict)
    }
    if set(observed_members) != set(approved_members):
        raise ValueError("cluster committee member set drift")
    for key, approved in approved_members.items():
        observed = observed_members[key]
        if (
            observed.get("seed") != approved.get("seed")
            or observed.get("relative_path") != approved.get("relative_path")
            or observed.get("kind") != approved.get("kind")
        ):
            raise ValueError("cluster committee model record differs from plan")


def _runner_module():
    spec = importlib.util.spec_from_file_location("mlipflow_active_learning_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load bundled active-learning runner: {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _operation(context: Any) -> str:
    return str(_mapping(_mapping(context).get("parameters")).get("operation", ""))


class Adapter:
    """Plan the local runner and independently reproduce every result in check."""

    def validate(self, context: Any) -> list[dict[str, str]]:
        diagnostics: list[dict[str, str]] = []
        if not isinstance(context, Mapping):
            return [_diagnostic("error", "context.type", "context must be an object")]
        for key in ("project_root", "attempt_dir"):
            if not _plain(context.get(key)):
                diagnostics.append(
                    _diagnostic("error", f"context.{key}", f"{key} must be a path string")
                )
        if not isinstance(context.get("inputs"), Mapping):
            diagnostics.append(_diagnostic("error", "context.inputs", "inputs must be an object"))
        if not isinstance(context.get("parameters"), Mapping):
            diagnostics.append(
                _diagnostic("error", "context.parameters", "parameters must be an object")
            )
        if not isinstance(context.get("resources"), Mapping):
            diagnostics.append(
                _diagnostic("error", "context.resources", "resources must be an object")
            )
        if context.get("backend") not in EXECUTION_BACKENDS:
            diagnostics.append(
                _diagnostic("error", "backend.unsupported", "unsupported active-learning backend")
            )
        operation = _operation(context)
        if operation not in OPERATIONS:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "parameters.operation",
                    "operation must be committee-evaluate, select-candidates, or assess-round",
                )
            )
            return diagnostics
        if diagnostics:
            return diagnostics
        if context.get("backend") == "ssh-slurm":
            if operation != "committee-evaluate":
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "backend.operation",
                        "scheduled active-learning supports committee-evaluate only",
                    )
                )
                return diagnostics
            try:
                _scheduled_inputs(context)
            except (KeyError, OSError, TypeError, ValueError) as exc:
                diagnostics.append(
                    _diagnostic("error", "committee.scheduled_contract", str(exc))
                )
            try:
                result = _result_path(context, operation)
                if result.exists() or result.is_symlink():
                    raise ValueError(
                        "scientific result already exists and will not be overwritten"
                    )
            except (KeyError, OSError, ValueError) as exc:
                diagnostics.append(
                    _diagnostic("error", "parameters.result_manifest", str(exc))
                )
            return diagnostics
        inputs = _mapping(context.get("inputs"))
        allowed = set(REQUIRED_INPUTS[operation]) | set(OPTIONAL_INPUTS[operation])
        unknown = sorted(set(inputs) - allowed)
        if unknown:
            diagnostics.append(
                _diagnostic("error", "inputs.unknown", "unsupported inputs: " + ", ".join(unknown))
            )
        for name in REQUIRED_INPUTS[operation]:
            try:
                _input_path(context, inputs.get(name))
            except (KeyError, OSError, ValueError) as exc:
                diagnostics.append(_diagnostic("error", f"inputs.{name}", str(exc)))
        for name in OPTIONAL_INPUTS[operation]:
            if inputs.get(name) is not None:
                try:
                    _input_path(context, inputs[name])
                except (KeyError, OSError, ValueError) as exc:
                    diagnostics.append(_diagnostic("error", f"inputs.{name}", str(exc)))
        try:
            result = _result_path(context, operation)
            if result.exists() or result.is_symlink():
                raise ValueError("scientific result already exists and will not be overwritten")
        except (KeyError, OSError, ValueError) as exc:
            diagnostics.append(_diagnostic("error", "parameters.result_manifest", str(exc)))
        if not RUNNER.is_file():
            diagnostics.append(
                _diagnostic("error", "implementation.missing", "bundled runner is missing")
            )
        return diagnostics

    def _paths(self, context: Mapping[str, Any], operation: str) -> dict[str, Path]:
        if context.get("backend") == "ssh-slurm":
            attempt = Path(str(context["attempt_dir"])).expanduser().absolute().resolve()
            return {
                "policy": _input_path(context, _mapping(context.get("inputs")).get("policy")),
                "committee_predictions": attempt / "committee-predictions.json",
            }
        inputs = _mapping(context.get("inputs"))
        names = list(REQUIRED_INPUTS[operation]) + [
            name for name in OPTIONAL_INPUTS[operation] if inputs.get(name) is not None
        ]
        return {name: _input_path(context, inputs[name]) for name in names}

    def plan(self, context: Any) -> dict[str, Any]:
        diagnostics = self.validate(context)
        if diagnostics:
            return {
                "plugin_id": PLUGIN_ID,
                "status": "BLOCKED",
                "executable": False,
                "diagnostics": diagnostics,
            }
        assert isinstance(context, Mapping)
        operation = _operation(context)
        if context.get("backend") == "ssh-slurm":
            return _plan_scheduled(context)
        paths = self._paths(context, operation)
        result_path = _result_path(context, operation)
        argv = [sys.executable, str(RUNNER), operation]
        for name, path in paths.items():
            argv.extend(["--" + name.replace("_", "-"), str(path)])
        argv.extend(["--result-manifest", str(result_path)])
        return {
            "plugin_id": PLUGIN_ID,
            "status": "READY",
            "executable": True,
            "shell": False,
            "argv": argv,
            "cwd": str(Path(str(context["attempt_dir"])).expanduser().absolute()),
            "expected_outputs": [str(result_path)],
            "operation": operation,
            "calculation_claim": "deterministic-offline-active-learning-decision",
            "implementation_path": str(RUNNER),
            "input_paths": {name: str(path) for name, path in sorted(paths.items())},
            "diagnostics": [],
        }


    def check(self, context: Any) -> dict[str, Any]:
        if not isinstance(context, Mapping):
            return {
                "plugin_id": PLUGIN_ID,
                "status": "FAIL",
                "diagnostics": [_diagnostic("error", "context.type", "context must be an object")],
            }
        operation = _operation(context)
        diagnostics = self.validate(context)
        diagnostics = [
            item
            for item in diagnostics
            if not (
                item["code"] == "parameters.result_manifest"
                and "already exists" in item["message"]
            )
        ]
        if diagnostics:
            return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
        execution = _mapping(context.get("execution"))
        if execution.get("returncode") not in {None, 0}:
            return {
                "plugin_id": PLUGIN_ID,
                "status": "FAIL",
                "diagnostics": [
                    _diagnostic(
                        "error",
                        "execution.nonzero",
                        f"active-learning runner returned {execution.get('returncode')}",
                    )
                ],
            }
        result_path = _result_path(context, operation)
        if not result_path.is_file() or result_path.is_symlink():
            return {
                "plugin_id": PLUGIN_ID,
                "status": "WAIT",
                "diagnostics": [
                    _diagnostic("info", "result.missing", "result manifest does not exist yet")
                ],
            }
        try:
            actual = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(actual, dict):
                raise ValueError("result root must be an object")
            if context.get("backend") == "ssh-slurm":
                _verify_cluster_report(context, result_path)
            expected = _runner_module().build_result(
                operation, self._paths(context, operation)
            )
            if actual != expected:
                raise ValueError("result differs from deterministic recomputation")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            return {
                "plugin_id": PLUGIN_ID,
                "status": "FAIL",
                "diagnostics": [_diagnostic("error", "result.invalid", str(exc))],
            }
        scientific_status = actual.get(
            "decision",
            actual.get("selection_status", actual.get("evaluation_status")),
        )
        return {
            "plugin_id": PLUGIN_ID,
            "status": "OK",
            "operation": operation,
            "scientific_status": scientific_status,
            "validation_claim": actual.get("validation_claim"),
            "result_manifest": str(result_path),
            "diagnostics": [],
        }

    def collect(self, context: Any) -> dict[str, Any]:
        checked = self.check(context)
        if checked.get("status") != "OK":
            return checked
        assert isinstance(context, Mapping)
        operation = _operation(context)
        result_path = _result_path(context, operation)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        role = {
            "committee-evaluate": "committee-evaluation",
            "select-candidates": "candidate-selection",
            "assess-round": "round-assessment",
        }[operation]
        metrics = {}
        if operation == "committee-evaluate":
            metrics = {
                "candidate_count": len(result["candidates"]),
                "model_count": len(result["committees"]),
                "calibration_ready": result["evaluation_status"] == "READY",
            }
        elif operation == "select-candidates":
            metrics = dict(result["counts"])
        else:
            metrics = {
                "decision": result["decision"],
                "new_dft_labels": result["dft_labels"]["new_this_round"],
                "cumulative_dft_labels": result["dft_labels"]["cumulative"],
                "passed_consecutive_rounds": result["passed_consecutive_rounds"],
            }
        artifacts = [
            {
                "role": role,
                "path": str(result_path),
                "media_type": "application/json",
            }
        ]
        if context.get("backend") == "ssh-slurm":
            attempt = Path(str(context["attempt_dir"])).expanduser().absolute().resolve()
            artifacts.extend(
                [
                    {
                        "role": "committee-predictions",
                        "path": str(attempt / "committee-predictions.json"),
                        "media_type": "application/json",
                    },
                    {
                        "role": "active-learning-cluster-report",
                        "path": str(attempt / "cluster-active-learning-report.json"),
                        "media_type": "application/json",
                    },
                ]
            )
        return {
            "plugin_id": PLUGIN_ID,
            "status": "OK",
            "artifacts": artifacts,
            "metrics": metrics,
            "validation_claim": result.get("validation_claim"),
        }
