"""MLIPFlow lifecycle and strict result validation for benchmark artifacts.

The adapter deliberately does not import any of the six reviewed model
configurations and never executes a program itself.  A user-owned prediction
CLI is required to accept the argv contract emitted by :meth:`Adapter.plan`
and to write the versioned result manifest consumed by :meth:`Adapter.check`
and :meth:`Adapter.collect`.  The bundled ``benchmark_wrapper.py`` is exposed
through separate normalize operations that recompute metrics only from
explicit pairs or replay existing evidence.  ``evaluate-fresh`` invokes the
bundled first-party runner and is the only operation that may claim model
execution.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from mlipflow.science import model_runtime


PLUGIN_ID = "mlip-benchmark"
MODEL_FAMILIES = frozenset(model_runtime.MODEL_FAMILIES)
LEGACY_OPERATIONS = frozenset({"evaluate-static", "collect-existing"})
NORMALIZE_OPERATIONS = frozenset({"normalize-replay", "normalize-execute"})
FRESH_OPERATION = "evaluate-fresh"
OPERATIONS = LEGACY_OPERATIONS | NORMALIZE_OPERATIONS | {FRESH_OPERATION}
EXECUTION_BACKENDS = frozenset({"local", "ssh-slurm"})
SHELL_EXECUTABLES = frozenset(
    {"bash", "csh", "cmd", "dash", "fish", "ksh", "powershell", "pwsh", "sh", "tcsh", "zsh"}
)
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_CSV_BYTES = 32 * 1024 * 1024
NORMALIZED_OUTPUT_NAMES = (
    "metrics.json",
    "benchmark_summary.csv",
    "model_ranking.json",
    "provenance.json",
)
FRESH_OUTPUT_NAMES = ("prediction_evidence.json",) + NORMALIZED_OUTPUT_NAMES
SUMMARY_COLUMNS = (
    "model",
    "task",
    "scenario",
    "split",
    "metric",
    "value",
    "unit",
    "direction",
    "sample_count",
    "mode",
    "source_path",
    "source_format",
    "evidence_locator",
    "unit_provenance",
    "dimensions_json",
)
BUNDLED_WRAPPER = (
    Path(globals().get("__file__", "adapter.py")).absolute().with_name("benchmark_wrapper.py")
)
BUNDLED_FRESH_RUNNER = BUNDLED_WRAPPER.with_name("fresh_benchmark.py")
BUNDLED_NORMALIZATION = BUNDLED_WRAPPER.with_name("benchmark_normalization.py")
SHARED_MODEL_RUNTIME = Path(model_runtime.__file__).resolve()
CLUSTER_FRESH_RUNNER = BUNDLED_FRESH_RUNNER.with_name("fresh_benchmark_cluster.py")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
HPC_RESOURCES = {"cpus", "gpus", "memory", "walltime"}
UNAVAILABLE_PEARSON_REASONS = frozenset(
    {"insufficient-scalar-pairs", "constant-reference-or-prediction"}
)


def _normalization_module():
    spec = importlib.util.spec_from_file_location(
        "mlipflow_benchmark_check_normalization", BUNDLED_NORMALIZATION
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot load bundled benchmark normalization")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _diagnostic(level: str, code: str, message: str) -> dict[str, str]:
    return {"level": level, "code": code, "message": message}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _metric_records_match(
    stored: list[dict[str, Any]], recomputed: list[dict[str, Any]]
) -> bool:
    if len(stored) != len(recomputed):
        return False
    for left, right in zip(stored, recomputed, strict=True):
        if (
            {key: value for key, value in left.items() if key != "value"}
            != {key: value for key, value in right.items() if key != "value"}
            or not math.isclose(
                float(left.get("value")),
                float(right.get("value")),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            return False
    return True


def _reference(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict) and isinstance(value.get("path"), str) and value["path"]:
        return value["path"]
    return None


def _plain_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and "\x00" not in value and "\n" not in value


def _project_path(context: dict[str, Any], value: Any) -> str:
    reference = _reference(value)
    if reference is None:
        raise ValueError("missing path reference")
    path = Path(reference)
    if path.is_absolute():
        return str(path)
    return str(Path(str(context["project_root"])) / path)


def _safe_attempt_output(context: dict[str, Any], value: Any) -> str:
    reference = _reference(value)
    if reference is None:
        raise ValueError("missing output path reference")
    path = Path(reference)
    if path.is_absolute() or path in {Path("."), Path("")} or ".." in path.parts:
        raise ValueError("output must be a relative path inside attempt_dir")
    return str(Path(str(context["attempt_dir"])) / path)


def _operation(context: dict[str, Any]) -> str:
    return str(_mapping(context.get("parameters")).get("operation", "evaluate-static"))


def _normalize_mode(operation: str) -> str:
    if operation == "normalize-replay":
        return "replay"
    if operation == "normalize-execute":
        return "execute"
    raise ValueError("operation is not a normalization operation")


def _portable_locator(value: Any, fallback: str, field: str) -> str:
    if value in (None, ""):
        return fallback
    if not _plain_string(value):
        raise ValueError(f"{field} must be a non-empty single-line string")
    locator = str(value).replace("\\", "/")
    lowered = locator.lower()
    if locator.startswith("/") or re.match(r"^[a-zA-Z]:/", locator) or lowered.startswith("file:"):
        raise ValueError(f"{field} must not expose an absolute local path")
    if "://" not in locator and ".." in Path(locator).parts:
        raise ValueError(f"{field} must not traverse parent directories")
    return locator


def _evidence_specs(context: dict[str, Any]) -> list[dict[str, str]]:
    inputs = _mapping(context.get("inputs"))
    raw = inputs.get("evidence_inputs")
    if not isinstance(raw, list) or not raw:
        raise ValueError("inputs.evidence_inputs must be a non-empty list")
    result: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        reference = _reference(item)
        if reference is None:
            raise ValueError(f"inputs.evidence_inputs[{index}] requires a path")
        resolved = _project_path(context, item)
        locator_value = item.get("evidence_locator") if isinstance(item, dict) else None
        locator = _portable_locator(
            locator_value, Path(reference).name, f"inputs.evidence_inputs[{index}].evidence_locator"
        )
        result.append({"path": resolved, "locator": locator})
    locators = [item["locator"] for item in result]
    if len(locators) != len(set(locators)):
        raise ValueError(
            "multiple evidence inputs require unique basenames or explicit unique evidence_locator values"
        )
    return result


def _project_file(root: Path, selected: Any = None) -> Path | None:
    if isinstance(selected, str) and selected:
        path = Path(selected).expanduser().absolute()
        if path.is_file() and not path.is_symlink() and path.resolve().parent == root.resolve():
            return path.resolve()
        return None
    found = [
        root / name
        for name in ("project.yaml", "project.yml", "project.json")
        if (root / name).is_file() and not (root / name).is_symlink()
    ]
    return found[0].resolve() if len(found) == 1 else None


def _project_input(root: Path, value: Any) -> Path | None:
    reference = _reference(value)
    if reference is None:
        return None
    candidate = Path(reference)
    candidate = candidate if candidate.is_absolute() else root / candidate
    if candidate.is_symlink() or not candidate.is_file():
        return None
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    return candidate.resolve()


def _artifact_reference(path: Path, id_key: str) -> dict[str, Any]:
    raw = _read_json(path)
    artifact_id = raw.get(id_key)
    relative = raw.get("relative_path")
    kind = raw.get("kind")
    if raw.get("schema_version") != 1 or not _plain_string(artifact_id):
        raise ValueError(f"{path.name} requires schema_version=1 and {id_key}")
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ValueError(f"{path.name} relative_path must stay below its site root")
    if kind not in {"file", "directory"}:
        raise ValueError(f"{path.name} kind must be file or directory")
    return {
        "id": artifact_id,
        "relative_path": relative,
        "kind": kind,
        "framework": raw.get("framework"),
    }


def _staged(path: Path, remote_name: str) -> dict[str, Any]:
    return {
        "source": str(path),
        "remote_name": remote_name,
        "sensitive": False,
        "fetch_allowed": False,
    }


def _fetch(
    remote_name: str,
    remote_path: str,
    local_name: str,
    maximum: int,
    role: str,
) -> dict[str, Any]:
    return {
        "remote_name": remote_name,
        "remote_path": remote_path,
        "local_name": local_name,
        "required": True,
        "max_bytes": maximum,
        "role": role,
    }


def _plan_scheduled_fresh(context: dict[str, Any]) -> dict[str, Any]:
    model_path, dataset_path, model, dataset, project = _scheduled_fresh_inputs(context)
    parameters = _mapping(context["parameters"])
    family = str(parameters["model_family"])
    framework = str(model_runtime.MODEL_FAMILY_FRAMEWORKS[family])
    output_reference = _reference(_mapping(context["inputs"]).get("output_dir"))
    if output_reference is None:
        raise ValueError("scheduled fresh benchmark requires output_dir")
    output_subdir = Path(output_reference).as_posix()
    staged = [
        _staged(project, "project.yaml"),
        _staged(model_path, "model-reference.json"),
        _staged(dataset_path, "benchmark-dataset-reference.json"),
        _staged(CLUSTER_FRESH_RUNNER, "fresh_benchmark_cluster.py"),
        _staged(BUNDLED_FRESH_RUNNER, "fresh_benchmark.py"),
        _staged(BUNDLED_WRAPPER, "benchmark_wrapper.py"),
        _staged(BUNDLED_NORMALIZATION, "benchmark_normalization.py"),
        _staged(SHARED_MODEL_RUNTIME, "model_runtime.py"),
    ]
    fetch_outputs = [
        _fetch(
            name,
            f"output/benchmark/{name}",
            f"{output_subdir}/{name}",
            MAX_CSV_BYTES if name.endswith(".csv") else MAX_JSON_BYTES,
            "prediction-evidence" if name == "prediction_evidence.json" else "benchmark-output",
        )
        for name in FRESH_OUTPUT_NAMES
    ]
    fetch_outputs.append(
        _fetch(
            "cluster-benchmark-report.json",
            "output/cluster-benchmark-report.json",
            "cluster-benchmark-report.json",
            MAX_JSON_BYTES,
            "benchmark-cluster-report",
        )
    )
    return {
        "plugin_id": PLUGIN_ID,
        "status": "READY",
        "executable": True,
        "scheduler_contract": "bundled-fresh-benchmark-v1",
        "argv": [f"template-family:benchmark-{framework}-canonical"],
        "cwd": "remote-attempt-workspace",
        "shell": False,
        "expected_outputs": [item["remote_name"] for item in fetch_outputs],
        "fresh_calculation": {
            "exact_model_family": family,
            "framework": framework,
            "model": model,
            "dataset": dataset,
            "task": parameters["task"],
            "scenario": parameters["scenario"],
            "split": parameters["split"],
            "targets": list(_fresh_targets(parameters)),
            "units": _units(parameters),
            "energy_normalization": parameters["energy_normalization"],
            "stress_convention": parameters.get("stress_convention"),
        },
        "input_paths": {
            "model_reference": str(model_path),
            "benchmark_dataset_reference": str(dataset_path),
        },
        "approval_summary": {
            "expensive": True,
            "submits_jobs": True,
            "execution_model": "single-python",
            "cpus_meaning": "threads-per-process",
            "exact_model_family": family,
            "framework": framework,
            "model_id": model["id"],
            "dataset_id": dataset["id"],
            "model_path": model["relative_path"],
            "dataset_path": dataset["relative_path"],
            "split": parameters["split"],
            "targets": list(_fresh_targets(parameters)),
            "model_execution": True,
        },
        "scheduled_execution": {
            "schema_version": 3,
            "execution_model": "single-python",
            "template_family": f"benchmark-{framework}-canonical",
            "staged_files": staged,
            "fetch_outputs": fetch_outputs,
        },
        "diagnostics": [],
    }


def _normalized_output_dir(context: dict[str, Any]) -> Path:
    inputs = _mapping(context.get("inputs"))
    return Path(_safe_attempt_output(context, inputs.get("output_dir")))


def _normalized_output_paths(context: dict[str, Any]) -> dict[str, Path]:
    output_dir = _normalized_output_dir(context)
    names = FRESH_OUTPUT_NAMES if _operation(context) == FRESH_OPERATION else NORMALIZED_OUTPUT_NAMES
    return {name: output_dir / name for name in names}


def _fresh_inputs(context: dict[str, Any]) -> tuple[Path, Path]:
    inputs = _mapping(context.get("inputs"))
    return (
        Path(_project_path(context, inputs.get("model"))),
        Path(_project_path(context, inputs.get("benchmark_dataset"))),
    )


def _fresh_targets(parameters: dict[str, Any]) -> tuple[str, ...]:
    raw = parameters.get("targets")
    if not isinstance(raw, list) or not raw:
        raise ValueError("parameters.targets must be a non-empty list")
    if any(item not in {"energy", "force", "stress"} for item in raw):
        raise ValueError("parameters.targets entries must be energy, force, or stress")
    if len(set(raw)) != len(raw):
        raise ValueError("parameters.targets entries must be unique")
    return tuple(raw)


def _scheduled_fresh_inputs(
    context: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any], dict[str, Any], Path]:
    root = Path(str(context["project_root"])).expanduser().absolute().resolve()
    inputs = _mapping(context.get("inputs"))
    allowed = {"model_reference", "benchmark_dataset_reference", "output_dir"}
    unknown = sorted(set(inputs) - allowed)
    if unknown:
        raise ValueError("scheduled fresh benchmark does not accept: " + ", ".join(unknown))
    model_path = _project_input(root, inputs.get("model_reference"))
    dataset_path = _project_input(root, inputs.get("benchmark_dataset_reference"))
    project = _project_file(root, context.get("project_path"))
    if model_path is None or dataset_path is None or project is None:
        raise ValueError(
            "scheduled fresh benchmark requires project-scoped model/dataset references and one project file"
        )
    model = _artifact_reference(model_path, "model_id")
    dataset = _artifact_reference(dataset_path, "dataset_id")
    parameters = _mapping(context.get("parameters"))
    family = str(parameters.get("model_family"))
    framework = model_runtime.MODEL_FAMILY_FRAMEWORKS.get(family)
    if framework is None:
        raise ValueError("scheduled fresh benchmark requires an exact supported model family")
    expected_kind = "directory" if framework == "m3gnet" else "file"
    if model["kind"] != expected_kind:
        raise ValueError(f"{framework} benchmark model reference kind must be {expected_kind}")
    if model.get("framework") not in {None, framework}:
        raise ValueError("model reference framework differs from the exact model family")
    if dataset["kind"] != "file":
        raise ValueError("benchmark dataset reference kind must be file")
    resources = context.get("resources")
    if not isinstance(resources, dict) or set(resources) != HPC_RESOURCES:
        raise ValueError("scheduled benchmark resources must be cpus, gpus, memory, walltime")
    output_dir = _normalized_output_dir(context)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("fresh output directory must be absent or empty")
    for path in (
        CLUSTER_FRESH_RUNNER,
        BUNDLED_FRESH_RUNNER,
        BUNDLED_WRAPPER,
        BUNDLED_NORMALIZATION,
        SHARED_MODEL_RUNTIME,
    ):
        if not path.is_file():
            raise ValueError(f"scheduled benchmark bundled file is missing: {path.name}")
    return model_path, dataset_path, model, dataset, project


def _source_script_spec(context: dict[str, Any]) -> dict[str, str] | None:
    inputs = _mapping(context.get("inputs"))
    value = inputs.get("source_script")
    if value in (None, ""):
        return None
    reference = _reference(value)
    if reference is None:
        raise ValueError("inputs.source_script requires a path")
    locator_value = value.get("source_script_locator") if isinstance(value, dict) else None
    if locator_value in (None, ""):
        locator_value = _mapping(context.get("parameters")).get("source_script_locator")
    return {
        "path": _project_path(context, value),
        "locator": _portable_locator(
            locator_value, Path(reference).name, "inputs.source_script.source_script_locator"
        ),
    }


def _expected_values(
    parameters: dict[str, Any], singular: str, plural: str, *, models: bool = False
) -> set[str]:
    singular_value = parameters.get(singular)
    plural_value = parameters.get(plural)
    if singular_value not in (None, "") and plural_value not in (None, []):
        raise ValueError(f"parameters.{singular} and parameters.{plural} are mutually exclusive")
    if plural_value not in (None, []):
        if not isinstance(plural_value, list) or not plural_value:
            raise ValueError(f"parameters.{plural} must be a non-empty list")
        values = plural_value
    elif singular_value not in (None, ""):
        values = [singular_value]
    else:
        raise ValueError(f"parameters.{singular} or parameters.{plural} is required")
    normalized: set[str] = set()
    for value in values:
        if not _plain_string(value):
            raise ValueError(f"parameters.{plural} entries must be non-empty strings")
        token = str(value)
        if models:
            if token not in MODEL_FAMILIES:
                raise ValueError(f"unsupported exact model family: {token}")
        elif not _IDENTIFIER.fullmatch(token):
            raise ValueError(f"parameters.{plural} entries must be normalized identifiers")
        normalized.add(token)
    if len(normalized) != len(values):
        raise ValueError(f"parameters.{plural} entries must be unique")
    return normalized


def _expected_identities(parameters: dict[str, Any]) -> dict[str, set[str]]:
    return {
        "model": _expected_values(
            parameters, "model_family", "expected_models", models=True
        ),
        "task": _expected_values(parameters, "task", "expected_tasks"),
        "scenario": _expected_values(parameters, "scenario", "expected_scenarios"),
    }


def _units(parameters: dict[str, Any]) -> dict[str, str]:
    value = parameters.get("units", {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("parameters.units must be an object")
    result: dict[str, str] = {}
    for key, unit in value.items():
        if key not in {"energy", "force", "stress"}:
            raise ValueError("parameters.units keys must be energy, force, or stress")
        if not _plain_string(unit):
            raise ValueError(f"parameters.units.{key} must be a non-empty unit string")
        result[key] = str(unit)
    return result


def _result_path(context: dict[str, Any]) -> Path:
    inputs = _mapping(context.get("inputs"))
    operation = _mapping(context.get("parameters")).get("operation", "evaluate-static")
    if operation == "collect-existing":
        return Path(_project_path(context, inputs.get("result_manifest")))
    return Path(_safe_attempt_output(context, inputs.get("result_manifest")))


def _read_json(path: Path) -> Any:
    size = path.stat().st_size
    if size > MAX_JSON_BYTES:
        raise ValueError(f"JSON artifact exceeds {MAX_JSON_BYTES} bytes: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _portable_child(base: Path, value: Any, field: str) -> Path:
    reference = _reference(value)
    if reference is None:
        raise ValueError(f"{field} must contain a non-empty path")
    path = Path(reference)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be relative to the result manifest")
    return base / path


def _validate_metric_records(metrics: Any) -> tuple[dict[str, float], list[dict[str, str]]]:
    diagnostics: list[dict[str, str]] = []
    values: dict[str, float] = {}
    if not isinstance(metrics, dict) or not metrics:
        return {}, [_diagnostic("error", "benchmark.metrics_required", "metrics must be a non-empty object")]
    for name, record in sorted(metrics.items()):
        prefix = f"metric {name!r}"
        if not isinstance(name, str) or not name:
            diagnostics.append(_diagnostic("error", "benchmark.metric_name", "metric names must be non-empty strings"))
            continue
        if not isinstance(record, dict):
            diagnostics.append(_diagnostic("error", "benchmark.metric_record", f"{prefix} must be an object"))
            continue
        value = record.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            diagnostics.append(_diagnostic("error", "benchmark.metric_value", f"{prefix} value must be finite"))
        else:
            values[name] = float(value)
        if not _plain_string(record.get("unit")):
            diagnostics.append(_diagnostic("error", "benchmark.metric_unit", f"{prefix} requires an explicit unit"))
        if record.get("direction") not in {"minimize", "maximize"}:
            diagnostics.append(
                _diagnostic("error", "benchmark.metric_direction", f"{prefix} direction must be minimize or maximize")
            )
        count = record.get("sample_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            diagnostics.append(
                _diagnostic("error", "benchmark.metric_sample_count", f"{prefix} sample_count must be a positive integer")
            )
    return values, diagnostics


def _load_result(
    path: Path, context: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, float], list[dict[str, Any]], list[dict[str, str]]]:
    diagnostics: list[dict[str, str]] = []
    artifacts: list[dict[str, Any]] = []
    try:
        raw = _read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return None, {}, [], [_diagnostic("error", "benchmark.result_unreadable", str(exc))]
    if not isinstance(raw, dict):
        return None, {}, [], [_diagnostic("error", "benchmark.result_type", "result manifest must be a JSON object")]
    if raw.get("schema_version") != 1:
        diagnostics.append(_diagnostic("error", "benchmark.result_schema", "schema_version must equal 1"))
    if raw.get("plugin_id") != PLUGIN_ID:
        diagnostics.append(_diagnostic("error", "benchmark.result_plugin", f"plugin_id must equal {PLUGIN_ID}"))
    if raw.get("status") not in {"OK", "FAIL"}:
        diagnostics.append(_diagnostic("error", "benchmark.result_status", "status must be OK or FAIL"))
    parameters = _mapping(context.get("parameters"))
    if raw.get("model_family") not in MODEL_FAMILIES:
        supported = ", ".join(sorted(MODEL_FAMILIES))
        diagnostics.append(
            _diagnostic(
                "error",
                "benchmark.result_family",
                f"model_family must be one of the exact reviewed names: {supported}",
            )
        )
    elif raw.get("model_family") != parameters.get("model_family"):
        diagnostics.append(
            _diagnostic("error", "benchmark.family_mismatch", "result model_family does not match the approved plan")
        )
    for key in ("task", "scenario"):
        if not _plain_string(raw.get(key)):
            diagnostics.append(_diagnostic("error", f"benchmark.result_{key}", f"{key} must be a non-empty string"))
        elif raw.get(key) != parameters.get(key):
            diagnostics.append(
                _diagnostic("error", f"benchmark.{key}_mismatch", f"result {key} does not match the approved plan")
            )
    conventions = raw.get("conventions")
    if not isinstance(conventions, dict):
        diagnostics.append(_diagnostic("error", "benchmark.result_conventions", "conventions must be an object"))
    else:
        for key in ("energy_normalization", "stress_convention"):
            if not _plain_string(conventions.get(key)):
                diagnostics.append(_diagnostic("error", f"benchmark.result_{key}", f"conventions.{key} is required"))
            elif conventions.get(key) != parameters.get(key):
                diagnostics.append(
                    _diagnostic("error", f"benchmark.{key}_mismatch", f"result conventions.{key} does not match the approved plan")
                )

    metrics_raw = raw.get("metrics")
    metrics_path: Path | None = None
    if metrics_raw is None and raw.get("metrics_file") is not None:
        try:
            metrics_path = _portable_child(path.parent, raw["metrics_file"], "metrics_file")
            loaded_metrics = _read_json(metrics_path)
            metrics_raw = loaded_metrics.get("metrics") if isinstance(loaded_metrics, dict) and "metrics" in loaded_metrics else loaded_metrics
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            diagnostics.append(_diagnostic("error", "benchmark.metrics_unreadable", str(exc)))
    values, metric_diagnostics = _validate_metric_records(metrics_raw)
    diagnostics.extend(metric_diagnostics)

    raw_artifacts = raw.get("artifacts", [])
    if not isinstance(raw_artifacts, list):
        diagnostics.append(_diagnostic("error", "benchmark.artifacts_type", "artifacts must be a list"))
    else:
        for index, item in enumerate(raw_artifacts):
            if not isinstance(item, dict):
                diagnostics.append(_diagnostic("error", "benchmark.artifact_record", f"artifacts[{index}] must be an object"))
                continue
            try:
                artifact_path = _portable_child(path.parent, item.get("path"), f"artifacts[{index}].path")
            except ValueError as exc:
                diagnostics.append(_diagnostic("error", "benchmark.artifact_path", str(exc)))
                continue
            if not artifact_path.is_file():
                diagnostics.append(_diagnostic("error", "benchmark.artifact_missing", f"artifact is not a regular file: {artifact_path}"))
            artifacts.append(
                {
                    "path": str(artifact_path),
                    "role": item.get("role", "benchmark-output"),
                    "media_type": item.get("media_type"),
                }
            )
    artifacts.insert(0, {"path": str(path), "role": "result-manifest", "media_type": "application/json"})
    if metrics_path is not None:
        artifacts.append({"path": str(metrics_path), "role": "metrics", "media_type": "application/json"})
    return raw, values, artifacts, diagnostics


def _read_summary(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if path.stat().st_size > MAX_CSV_BYTES:
        raise ValueError(f"CSV artifact exceeds {MAX_CSV_BYTES} bytes: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("benchmark_summary.csv has no header")
        return list(reader.fieldnames), list(reader)


def _expected_sources(context: dict[str, Any]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for spec in _evidence_specs(context):
        path = Path(spec["path"])
        if not path.is_file():
            raise ValueError(f"evidence input is not a regular file: {path}")
        if spec["locator"] in result:
            raise ValueError(f"duplicate portable evidence locator: {spec['locator']}")
        result[spec["locator"]] = path
    return result


def _normalized_metric_key(record: dict[str, Any]) -> str:
    dimensions = json.dumps(record.get("dimensions", {}), sort_keys=True, separators=(",", ":"))
    return "/".join(
        str(record.get(name, ""))
        for name in ("model", "task", "scenario", "split", "metric", "unit")
    ) + "/" + dimensions


def _validate_unavailable_metrics(
    raw: Any,
    *,
    mode: str,
    expected_identities: dict[str, set[str]],
    expected_split: str,
    expected_source_paths: set[str],
) -> dict[str, dict[str, Any]]:
    if raw is None and mode == "replay":
        return {}
    if not isinstance(raw, list):
        raise ValueError("unavailable_metrics must be a list")
    validated: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(raw):
        if not isinstance(record, dict) or record.get("mode") != mode:
            raise ValueError(f"unavailable_metrics[{index}] mode/schema drift")
        for identity, expected in expected_identities.items():
            if record.get(identity) not in expected:
                raise ValueError(f"unavailable_metrics[{index}] {identity} drift")
        dimensions = _mapping(record.get("dimensions"))
        target = dimensions.get("target")
        if (
            target not in {"energy", "force", "stress"}
            or record.get("split") != expected_split
            or record.get("metric") != f"{target}_pearson_r"
            or record.get("unit") != "dimensionless"
            or record.get("direction") != "maximize"
            or record.get("unit_provenance") != "defined-by-metric"
            or record.get("reason") not in UNAVAILABLE_PEARSON_REASONS
            or "value" in record
        ):
            raise ValueError(f"unavailable_metrics[{index}] scientific identity drift")
        sample_count = record.get("sample_count")
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
            raise ValueError(f"unavailable_metrics[{index}] sample_count is invalid")
        if record.get("reason") == "insufficient-scalar-pairs" and sample_count >= 2:
            raise ValueError(f"unavailable_metrics[{index}] reason/count mismatch")
        if record.get("reason") == "constant-reference-or-prediction" and sample_count < 2:
            raise ValueError(f"unavailable_metrics[{index}] reason/count mismatch")
        if record.get("source_path") not in expected_source_paths:
            raise ValueError(f"unavailable_metrics[{index}] source path drift")
        if not _plain_string(record.get("source_format")) or not _plain_string(
            record.get("evidence_locator")
        ):
            raise ValueError(f"unavailable_metrics[{index}] source provenance is incomplete")
        key = _normalized_metric_key(record)
        if key in validated:
            raise ValueError(f"duplicate unavailable metric identity: {key}")
        validated[key] = record
    return validated


def _load_normalized_outputs(
    context: dict[str, Any], paths: dict[str, Path]
) -> tuple[dict[str, Any], dict[str, float], list[dict[str, Any]]]:
    parameters = _mapping(context.get("parameters"))
    mode = _normalize_mode(_operation(context))
    expected_identities = _expected_identities(parameters)
    expected_sources = _expected_sources(context)

    metrics = _read_json(paths["metrics.json"])
    ranking = _read_json(paths["model_ranking.json"])
    provenance = _read_json(paths["provenance.json"])
    summary_fields, summary_rows = _read_summary(paths["benchmark_summary.csv"])
    for name, payload in (
        ("metrics.json", metrics),
        ("model_ranking.json", ranking),
        ("provenance.json", provenance),
    ):
        if not isinstance(payload, dict):
            raise ValueError(f"{name} must contain a JSON object")
        if payload.get("schema_version") != 1:
            raise ValueError(f"{name} schema_version must equal 1")
        if payload.get("plugin_id") != PLUGIN_ID:
            raise ValueError(f"{name} plugin_id must equal {PLUGIN_ID}")

    if metrics.get("mode") != mode or provenance.get("mode") != mode:
        raise ValueError("normalized metrics/provenance mode does not match the approved operation")
    if provenance.get("model_execution") is not False:
        raise ValueError("normalized provenance must state model_execution=false")
    if provenance.get("network_access") is not False:
        raise ValueError("normalized provenance must state network_access=false")
    if not _plain_string(provenance.get("wrapper_version")):
        raise ValueError("normalized provenance requires wrapper_version")

    records = metrics.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("metrics.json records must be a non-empty list")
    if metrics.get("record_count") != len(records):
        raise ValueError("metrics.json record_count does not match records")
    supported = metrics.get("supported_models")
    if not isinstance(supported, list) or set(supported) != set(MODEL_FAMILIES):
        raise ValueError("metrics.json supported_models does not name the six exact models")

    observed_identities = {name: set() for name in expected_identities}
    record_values: dict[str, float] = {}
    expected_source_paths = set(expected_sources)
    unavailable_metrics = _validate_unavailable_metrics(
        metrics.get("unavailable_metrics"),
        mode=mode,
        expected_identities=expected_identities,
        expected_split=str(parameters.get("split", "all")),
        expected_source_paths=expected_source_paths,
    )
    if provenance.get("unavailable_metrics") != metrics.get("unavailable_metrics"):
        raise ValueError("metrics/provenance unavailable metric drift")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"metrics.json records[{index}] must be an object")
        if record.get("mode") != mode:
            raise ValueError(f"metrics.json records[{index}] mode mismatch")
        for identity in observed_identities:
            value = record.get(identity)
            if not _plain_string(value):
                raise ValueError(f"metrics.json records[{index}] requires {identity}")
            observed_identities[identity].add(str(value))
        if record.get("model") not in MODEL_FAMILIES:
            raise ValueError(f"metrics.json records[{index}] uses an unsupported model")
        value = record.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"metrics.json records[{index}] value must be finite")
        if not _plain_string(record.get("metric")) or not _plain_string(record.get("unit")):
            raise ValueError(f"metrics.json records[{index}] requires metric and unit")
        if record.get("direction") not in {"minimize", "maximize"}:
            raise ValueError(f"metrics.json records[{index}] direction is invalid")
        sample_count = record.get("sample_count")
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
            raise ValueError(f"metrics.json records[{index}] sample_count must be positive")
        if record.get("source_path") not in expected_source_paths:
            raise ValueError(f"metrics.json records[{index}] source path does not match input evidence")
        if not _plain_string(record.get("source_format")) or not _plain_string(
            record.get("evidence_locator")
        ):
            raise ValueError(f"metrics.json records[{index}] source provenance is incomplete")
        key = _normalized_metric_key(record)
        if key in record_values:
            raise ValueError(f"metrics.json contains duplicate metric identity: {key}")
        record_values[key] = float(value)
    for identity, expected in expected_identities.items():
        if observed_identities[identity] != expected:
            raise ValueError(
                f"normalized {identity} identities {sorted(observed_identities[identity])} "
                f"do not match approved {sorted(expected)}"
            )

    source_evidence = provenance.get("source_evidence")
    if not isinstance(source_evidence, list) or len(source_evidence) != len(expected_sources):
        raise ValueError("provenance source_evidence count does not match approved inputs")
    actual_sources: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(source_evidence):
        if not isinstance(item, dict) or not _plain_string(item.get("path")):
            raise ValueError(f"provenance source_evidence[{index}] is invalid")
        locator = str(item["path"])
        if locator in actual_sources:
            raise ValueError(f"provenance contains duplicate source locator: {locator}")
        actual_sources[locator] = item
    if set(actual_sources) != set(expected_sources):
        raise ValueError("provenance source locators do not match approved evidence inputs")
    source_script = _source_script_spec(context)
    for locator in expected_sources:
        item = actual_sources[locator]
        if item.get("read_only_import") is not True:
            raise ValueError(f"provenance import mode drift for evidence source {locator}")
        if not _plain_string(item.get("parser")) or not isinstance(
            item.get("normalized_record_count"), int
        ):
            raise ValueError(f"provenance parser/count missing for evidence source {locator}")
        if item["normalized_record_count"] != sum(
            record.get("source_path") == locator for record in records
        ):
            raise ValueError(f"provenance normalized record count drift for {locator}")
        source_unavailable = item.get("unavailable_metrics", [])
        if mode == "execute" and item.get("unavailable_metric_count") != len(
            source_unavailable
        ):
            raise ValueError(f"provenance unavailable metric count drift for {locator}")
        original = item.get("original_implementation")
        if source_script is None:
            if original is not None:
                raise ValueError("unexpected source-script provenance in normalized outputs")
        else:
            script_path = Path(source_script["path"])
            if not script_path.is_file():
                raise ValueError(f"source script is not a regular file: {script_path}")
            if not isinstance(original, dict) or (
                original.get("path") != source_script["locator"]
                or original.get("read_only_reference") is not True
            ):
                raise ValueError("source-script provenance does not match the approved source")

    flattened_unavailable = [
        record
        for source in source_evidence
        for record in source.get("unavailable_metrics", [])
    ]
    if sorted(
        (json.dumps(record, sort_keys=True) for record in flattened_unavailable)
    ) != sorted(json.dumps(record, sort_keys=True) for record in unavailable_metrics.values()):
        raise ValueError("source/normalized unavailable metric drift")

    output_artifacts = provenance.get("output_artifacts")
    if not isinstance(output_artifacts, list):
        raise ValueError("provenance output_artifacts must be a list")
    actual_output_records = {
        item.get("path"): item for item in output_artifacts if isinstance(item, dict)
    }
    expected_provenance_outputs = set(NORMALIZED_OUTPUT_NAMES) - {"provenance.json"}
    if set(actual_output_records) != expected_provenance_outputs:
        raise ValueError("provenance output paths must name metrics, summary, and ranking")

    if tuple(summary_fields) != SUMMARY_COLUMNS:
        raise ValueError("benchmark_summary.csv columns do not match the fixed contract")
    if len(summary_rows) != len(records):
        raise ValueError("benchmark_summary.csv row count does not match metrics.json")
    for index, (row, record) in enumerate(zip(summary_rows, records)):
        for field in (
            "model",
            "task",
            "scenario",
            "split",
            "metric",
            "unit",
            "direction",
            "mode",
            "source_path",
            "source_format",
            "evidence_locator",
            "unit_provenance",
        ):
            if row.get(field) != str(record.get(field)):
                raise ValueError(f"benchmark_summary.csv row {index + 2} field {field} drift")
        try:
            row_value = float(row["value"])
            row_count = int(row["sample_count"])
            row_dimensions = json.loads(row["dimensions_json"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"benchmark_summary.csv row {index + 2} is malformed") from exc
        if row_value != float(record["value"]) or row_count != record["sample_count"]:
            raise ValueError(f"benchmark_summary.csv row {index + 2} value/count drift")
        if row_dimensions != record.get("dimensions"):
            raise ValueError(f"benchmark_summary.csv row {index + 2} dimensions drift")

    rankings = ranking.get("rankings")
    if not isinstance(rankings, list):
        raise ValueError("model_ranking.json rankings must be a list")
    for index, item in enumerate(rankings):
        if not isinstance(item, dict):
            raise ValueError(f"model_ranking.json rankings[{index}] must be an object")
        if item.get("task") not in expected_identities["task"] or item.get(
            "scenario"
        ) not in expected_identities["scenario"]:
            raise ValueError(f"model_ranking.json rankings[{index}] identity mismatch")
        for field in ("split", "metric", "unit"):
            if not _plain_string(item.get(field)):
                raise ValueError(f"model_ranking.json rankings[{index}] requires {field}")
        if item.get("direction") not in {"minimize", "maximize"}:
            raise ValueError(f"model_ranking.json rankings[{index}] direction is invalid")
        ranking_dimensions = item.get("dimensions")
        if not isinstance(ranking_dimensions, dict):
            raise ValueError(f"model_ranking.json rankings[{index}] dimensions must be an object")
        matching_records = [
            record
            for record in records
            if record.get("task") == item.get("task")
            and record.get("scenario") == item.get("scenario")
            and record.get("split") == item.get("split")
            and record.get("metric") == item.get("metric")
            and record.get("unit") == item.get("unit")
            and record.get("direction") == item.get("direction")
            and {
                key: value
                for key, value in record.get("dimensions", {}).items()
                if key not in {"sheet", "num_structures", "count_in_bin"}
            }
            == ranking_dimensions
        ]
        candidates = item.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"model_ranking.json rankings[{index}] needs candidates")
        if len(matching_records) != len(candidates):
            raise ValueError(f"model_ranking.json rankings[{index}] candidate count drift")
        expected_candidates = sorted(
            matching_records,
            key=lambda record: (
                record["value"]
                if item["direction"] == "minimize"
                else -record["value"],
                record["model"],
            ),
        )
        candidate_models = set()
        previous_value: float | None = None
        expected_rank = 0
        for position, (candidate, record) in enumerate(
            zip(candidates, expected_candidates), start=1
        ):
            if not isinstance(candidate, dict) or candidate.get("model") not in expected_identities[
                "model"
            ]:
                raise ValueError(f"model_ranking.json rankings[{index}] model mismatch")
            value = candidate.get("value")
            rank = candidate.get("rank")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or isinstance(rank, bool)
                or not isinstance(rank, int)
                or rank < 1
            ):
                raise ValueError(f"model_ranking.json rankings[{index}] candidate is malformed")
            if previous_value is None or float(record["value"]) != previous_value:
                expected_rank = position
            previous_value = float(record["value"])
            if (
                candidate.get("model") != record.get("model")
                or float(candidate["value"]) != float(record["value"])
                or candidate.get("sample_count") != record.get("sample_count")
                or candidate.get("rank") != expected_rank
            ):
                raise ValueError(
                    f"model_ranking.json rankings[{index}] does not match metrics.json"
                )
            candidate_models.add(candidate["model"])
        if item.get("comparable_model_count") != len(candidate_models):
            raise ValueError(f"model_ranking.json rankings[{index}] model count drift")

    artifact_roles = {
        "metrics.json": ("normalized-metrics", "application/json"),
        "benchmark_summary.csv": ("benchmark-summary", "text/csv"),
        "model_ranking.json": ("model-ranking", "application/json"),
        "provenance.json": ("provenance", "application/json"),
    }
    artifacts = [
        {
            "path": str(paths[name]),
            "role": artifact_roles[name][0],
            "media_type": artifact_roles[name][1],
        }
        for name in NORMALIZED_OUTPUT_NAMES
    ]
    return metrics, record_values, artifacts


def _load_fresh_outputs(
    context: dict[str, Any], paths: dict[str, Path]
) -> tuple[dict[str, Any], dict[str, float], list[dict[str, Any]]]:
    """Validate the scientific contents of the five fresh benchmark outputs."""

    parameters = _mapping(context.get("parameters"))

    evidence = _read_json(paths["prediction_evidence.json"])
    metrics = _read_json(paths["metrics.json"])
    ranking = _read_json(paths["model_ranking.json"])
    provenance = _read_json(paths["provenance.json"])
    for name, payload in (
        ("prediction_evidence.json", evidence),
        ("metrics.json", metrics),
        ("model_ranking.json", ranking),
        ("provenance.json", provenance),
    ):
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError(f"{name} must be a schema_version=1 JSON object")
        if payload.get("plugin_id") != PLUGIN_ID:
            raise ValueError(f"{name} plugin identity drift")
    if evidence.get("contract") != "mlip-benchmark/prediction-evidence":
        raise ValueError("prediction evidence contract identity drift")
    if any(payload.get("mode") != "fresh" for payload in (evidence, metrics, provenance)):
        raise ValueError("fresh output mode drift")
    if evidence.get("model_execution") is not True or provenance.get("model_execution") is not True:
        raise ValueError("fresh output must state model_execution=true")
    if provenance.get("network_access") is not False:
        raise ValueError("fresh provenance must state network_access=false")

    family = parameters.get("model_family")
    framework = "deepmd" if str(family).startswith("deepmd-") else family
    model_identity = evidence.get("model")
    dataset_identity = evidence.get("dataset")
    runtime = evidence.get("runtime")
    if not isinstance(model_identity, dict) or (
        model_identity.get("family") != family
        or model_identity.get("framework") != framework
        or not _plain_string(model_identity.get("path"))
    ):
        raise ValueError("prediction evidence model record drift")
    if not isinstance(dataset_identity, dict) or not _plain_string(dataset_identity.get("path")):
        raise ValueError("prediction evidence dataset record drift")
    if not isinstance(runtime, dict) or (
        runtime.get("framework") != framework
        or not _plain_string(runtime.get("framework_version"))
        or not _plain_string(runtime.get("backend"))
    ):
        raise ValueError("prediction evidence runtime record is incomplete")
    if runtime.get("exact_model_family") not in {None, family}:
        raise ValueError("prediction evidence runtime model-family drift")
    source_paths = evidence.get("source_paths")
    if not isinstance(source_paths, dict) or any(
        not _plain_string(source_paths.get(name))
        for name in ("fresh_runner", "metric_normalization", "shared_model_runtime")
    ):
        raise ValueError("fresh implementation source paths are incomplete")
    for key in ("task", "scenario", "split"):
        if evidence.get(key) != parameters.get(key) or provenance.get(key) != parameters.get(key):
            raise ValueError(f"fresh {key} identity drift")
    targets = _fresh_targets(parameters)
    units = _units(parameters)
    if evidence.get("targets") != list(targets) or evidence.get("units") != {
        target: units[target] for target in targets
    }:
        raise ValueError("fresh targets or units drift")
    conventions = evidence.get("conventions")
    if not isinstance(conventions, dict) or conventions.get(
        "energy_normalization"
    ) != parameters.get("energy_normalization"):
        raise ValueError("fresh energy convention drift")
    if conventions.get("stress_convention") != parameters.get("stress_convention"):
        raise ValueError("fresh stress convention drift")

    evidence_records = evidence.get("records")
    structure_count = evidence.get("structure_count")
    scalar_counts = evidence.get("scalar_sample_counts")
    if not isinstance(evidence_records, list) or not evidence_records:
        raise ValueError("prediction evidence records must be non-empty")
    if not isinstance(structure_count, int) or structure_count < 1:
        raise ValueError("prediction evidence structure_count must be positive")
    if dataset_identity.get("structure_count") != structure_count:
        raise ValueError("prediction evidence structure count drift")
    if not isinstance(scalar_counts, dict) or set(scalar_counts) != set(targets):
        raise ValueError("prediction evidence scalar sample counts are incomplete")
    observed_ids: dict[str, set[str]] = {target: set() for target in targets}
    observed_counts = {target: 0 for target in targets}
    force_atom_count = 0
    for index, record in enumerate(evidence_records):
        if not isinstance(record, dict) or record.get("target") not in targets:
            raise ValueError(f"prediction evidence record {index} has an invalid target")
        target = record["target"]
        if record.get("unit") != units[target] or not _plain_string(record.get("sample_id")):
            raise ValueError(f"prediction evidence record {index} identity/unit drift")
        count = record.get("scalar_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"prediction evidence record {index} scalar_count is invalid")
        observed_counts[target] += count
        observed_ids[target].add(record["sample_id"])
        if "reference" not in record or "prediction" not in record:
            raise ValueError(f"prediction evidence record {index} is partial")
        if target == "force":
            natoms = record.get("natoms")
            if (
                isinstance(natoms, bool)
                or not isinstance(natoms, int)
                or natoms < 1
                or count != 3 * natoms
            ):
                raise ValueError(
                    f"prediction evidence force record {index} atom/scalar count drift"
                )
            force_atom_count += natoms
    if observed_counts != scalar_counts:
        raise ValueError("prediction evidence scalar sample count drift")
    if any(len(ids) != structure_count for ids in observed_ids.values()):
        raise ValueError("prediction evidence target/structure coverage is partial")

    unavailable_metrics = _validate_unavailable_metrics(
        metrics.get("unavailable_metrics"),
        mode="fresh",
        expected_identities={
            "model": {str(family)},
            "task": {str(parameters.get("task"))},
            "scenario": {str(parameters.get("scenario"))},
        },
        expected_split=str(parameters.get("split")),
        expected_source_paths={"prediction_evidence.json"},
    )
    if provenance.get("unavailable_metrics") != metrics.get("unavailable_metrics"):
        raise ValueError("fresh metrics/provenance unavailable metric drift")
    if (
        provenance.get("exact_model_family") != family
        or provenance.get("model_path") != model_identity.get("path")
        or provenance.get("dataset_path") != dataset_identity.get("path")
        or provenance.get("runtime") != runtime
        or provenance.get("source_paths") != source_paths
        or provenance.get("units") != evidence.get("units")
        or provenance.get("conventions") != conventions
        or provenance.get("structure_count") != structure_count
        or provenance.get("scalar_sample_counts") != scalar_counts
    ):
        raise ValueError("fresh provenance record drift")

    output_artifacts = provenance.get("output_artifacts")
    if not isinstance(output_artifacts, list):
        raise ValueError("fresh provenance output_artifacts must be a list")
    artifact_map = {item.get("path"): item for item in output_artifacts if isinstance(item, dict)}
    expected_recorded = set(FRESH_OUTPUT_NAMES) - {"provenance.json"}
    if set(artifact_map) != expected_recorded:
        raise ValueError("fresh provenance output artifact set drift")

    records = metrics.get("records")
    if not isinstance(records, list) or not records or metrics.get("record_count") != len(records):
        raise ValueError("fresh metrics records are missing or partial")
    supported = metrics.get("supported_models")
    if not isinstance(supported, list) or set(supported) != set(MODEL_FAMILIES):
        raise ValueError("fresh metrics exact model-family set drift")
    metric_values: dict[str, float] = {}
    metric_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("mode") != "fresh":
            raise ValueError(f"fresh metric record {index} mode drift")
        if (
            record.get("model") != family
            or record.get("task") != parameters.get("task")
            or record.get("scenario") != parameters.get("scenario")
            or record.get("split") != parameters.get("split")
            or record.get("source_path") != "prediction_evidence.json"
        ):
            raise ValueError(f"fresh metric record {index} identity drift")
        target = _mapping(record.get("dimensions")).get("target")
        if target not in targets:
            raise ValueError(f"fresh metric record {index} target drift")
        metric_name = str(record.get("metric", ""))
        if metric_name == "maximum_atomic_force_error":
            if target != "force" or _mapping(record.get("dimensions")) != {
                "scope": "overall",
                "target": "force",
                "error_norm": "atomic-l2-vector",
                "aggregation": "maximum-over-atoms",
            }:
                raise ValueError(
                    f"fresh metric record {index} maximum-force dimensions drift"
                )
            statistic = metric_name
            expected_count = force_atom_count
        else:
            statistic = metric_name.removeprefix(f"{target}_")
            expected_count = scalar_counts[target]
        expected_unit = "dimensionless" if statistic == "pearson_r" else units[target]
        expected_direction = "maximize" if statistic == "pearson_r" else "minimize"
        value = record.get("value")
        if (
            statistic not in {"mae", "rmse", "pearson_r", "maximum_atomic_force_error"}
            or record.get("unit") != expected_unit
            or record.get("direction") != expected_direction
            or record.get("sample_count") != expected_count
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"fresh metric record {index} metric/count/unit drift")
        key = (target, statistic, expected_unit)
        if key in metric_by_key:
            raise ValueError("fresh metrics contain a duplicate identity")
        metric_by_key[key] = record
        metric_values[_normalized_metric_key(record)] = float(value)
    unavailable_targets: set[str] = set()
    for record in unavailable_metrics.values():
        target = str(_mapping(record.get("dimensions")).get("target"))
        if target not in targets or record.get("sample_count") != scalar_counts[target]:
            raise ValueError("fresh unavailable Pearson target/count drift")
        if target in unavailable_targets:
            raise ValueError("fresh unavailable Pearson target is duplicated")
        unavailable_targets.add(target)
    expected_metric_keys = {
        (target, statistic, units[target])
        for target in targets
        for statistic in ("mae", "rmse")
    }
    expected_metric_keys.update(
        (target, "pearson_r", "dimensionless")
        for target in targets
        if target not in unavailable_targets
    )
    if "force" in targets:
        expected_metric_keys.add(
            ("force", "maximum_atomic_force_error", units["force"])
        )
    if set(metric_by_key) != expected_metric_keys:
        raise ValueError("fresh normalized metric set is incomplete")

    summary_fields, summary_rows = _read_summary(paths["benchmark_summary.csv"])
    if tuple(summary_fields) != SUMMARY_COLUMNS or len(summary_rows) != len(records):
        raise ValueError("fresh benchmark summary schema/count drift")
    for row, record in zip(summary_rows, records):
        if row.get("metric") != record.get("metric") or row.get("mode") != "fresh":
            raise ValueError("fresh benchmark summary content drift")
        if float(row["value"]) != float(record["value"]) or int(row["sample_count"]) != record[
            "sample_count"
        ]:
            raise ValueError("fresh benchmark summary value/count drift")

    rankings = ranking.get("rankings")
    if not isinstance(rankings, list) or len(rankings) != len(records):
        raise ValueError("fresh ranking set is incomplete")
    for item in rankings:
        candidates = item.get("candidates") if isinstance(item, dict) else None
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise ValueError("fresh per-model ranking candidates are malformed")
        candidate = candidates[0]
        if candidate.get("rank") != 1 or candidate.get("model") != family:
            raise ValueError("fresh ranking is not derived from metric direction")
        matching = [
            record
            for record in records
            if record.get("task") == item.get("task")
            and record.get("scenario") == item.get("scenario")
            and record.get("split") == item.get("split")
            and record.get("metric") == item.get("metric")
            and record.get("unit") == item.get("unit")
            and record.get("direction") == item.get("direction")
            and _mapping(record.get("dimensions")) == item.get("dimensions")
        ]
        if len(matching) != 1:
            raise ValueError("fresh ranking identity does not match one metric")
        record = matching[0]
        if (
            float(candidate.get("value")) != float(record["value"])
            or candidate.get("sample_count") != record.get("sample_count")
            or item.get("comparable_model_count") != 1
        ):
            raise ValueError("fresh ranking value/count/evidence drift")

    normalizer = _normalization_module()
    recomputed_records, recomputed_details = normalizer._execute_source(
        {
            "path": paths["prediction_evidence.json"],
            "evidence_locator": "prediction_evidence.json",
            "model": family,
            "task": parameters["task"],
            "scenario": parameters["scenario"],
            "split": parameters["split"],
            "units": units,
        }
    )
    for record in recomputed_records:
        record["mode"] = "fresh"
    recomputed_unavailable = recomputed_details["unavailable_metrics"]
    for record in recomputed_unavailable:
        record["mode"] = "fresh"
    recomputed_records = normalizer._validate_records(recomputed_records)
    if not _metric_records_match(records, recomputed_records) or metrics.get(
        "unavailable_metrics"
    ) != recomputed_unavailable:
        raise ValueError("fresh metrics differ from prediction evidence")

    roles = {
        "prediction_evidence.json": ("prediction-evidence", "application/json"),
        "metrics.json": ("normalized-metrics", "application/json"),
        "benchmark_summary.csv": ("benchmark-summary", "text/csv"),
        "model_ranking.json": ("model-ranking", "application/json"),
        "provenance.json": ("provenance", "application/json"),
    }
    artifacts = [
        {"path": str(paths[name]), "role": roles[name][0], "media_type": roles[name][1]}
        for name in FRESH_OUTPUT_NAMES
    ]
    return metrics, metric_values, artifacts


def _verify_cluster_fresh_report(
    context: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any]:
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    report = _read_json(attempt / "cluster-benchmark-report.json")
    execution = _mapping(context.get("execution"))
    plan = _mapping(execution.get("plan"))
    calculation = _mapping(plan.get("fresh_calculation"))
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != 1
        or report.get("status") != "OK"
        or report.get("return_code") != 0
        or report.get("exact_model_family") != calculation.get("exact_model_family")
        or report.get("framework") != calculation.get("framework")
    ):
        raise ValueError("cluster fresh benchmark report did not record approved success")
    model = _mapping(report.get("model"))
    dataset = _mapping(report.get("dataset"))
    if (
        model.get("id") != _mapping(calculation.get("model")).get("id")
        or model.get("relative_path")
        != _mapping(calculation.get("model")).get("relative_path")
        or dataset.get("id") != _mapping(calculation.get("dataset")).get("id")
        or dataset.get("relative_path")
        != _mapping(calculation.get("dataset")).get("relative_path")
    ):
        raise ValueError("cluster fresh benchmark model/dataset record drift")
    outputs = report.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(paths):
        raise ValueError("cluster fresh benchmark output path set drift")
    return report


class Adapter:
    """Plan reviewed external CLIs and parse only their explicit artifacts."""

    def validate(self, context: dict[str, Any]) -> list[dict[str, str]]:
        diagnostics: list[dict[str, str]] = []
        if not isinstance(context, dict):
            return [_diagnostic("error", "context.type", "context must be an object")]
        for key in ("project_root", "attempt_dir"):
            if not _plain_string(context.get(key)):
                diagnostics.append(_diagnostic("error", f"context.{key}", f"{key} must be a non-empty path string"))
        inputs = context.get("inputs")
        parameters = context.get("parameters")
        resources = context.get("resources")
        if not isinstance(inputs, dict):
            diagnostics.append(_diagnostic("error", "context.inputs", "inputs must be an object"))
            inputs = {}
        if not isinstance(parameters, dict):
            diagnostics.append(_diagnostic("error", "context.parameters", "parameters must be an object"))
            parameters = {}
        if not isinstance(resources, dict):
            diagnostics.append(_diagnostic("error", "context.resources", "resources must be an object"))
        if context.get("backend") not in EXECUTION_BACKENDS:
            diagnostics.append(_diagnostic("error", "context.backend", "unsupported benchmark backend"))

        operation = parameters.get("operation", "evaluate-static")
        if operation not in OPERATIONS:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "benchmark.operation",
                    "operation must be evaluate-fresh, evaluate-static, collect-existing, normalize-replay, or normalize-execute",
                )
            )
            return diagnostics

        if operation == FRESH_OPERATION:
            try:
                if parameters.get("model_family") not in MODEL_FAMILIES:
                    raise ValueError("parameters.model_family must name one exact supported family")
                for key in ("task", "scenario", "split"):
                    if not _plain_string(parameters.get(key)) or not _IDENTIFIER.fullmatch(
                        str(parameters[key])
                    ):
                        raise ValueError(f"parameters.{key} must be a normalized identifier")
                targets = _fresh_targets(parameters)
                units = _units(parameters)
                missing_units = sorted(set(targets) - set(units))
                if missing_units:
                    raise ValueError("parameters.units missing targets: " + ", ".join(missing_units))
                if parameters.get("energy_normalization") not in {"total", "per-atom"}:
                    raise ValueError("parameters.energy_normalization must be total or per-atom")
                if "stress" in targets and not _plain_string(parameters.get("stress_convention")):
                    raise ValueError("stress evaluation requires parameters.stress_convention")
                if context.get("backend") == "ssh-slurm":
                    _scheduled_fresh_inputs(context)
                else:
                    model_path, dataset_path = _fresh_inputs(context)
                    if not model_path.exists():
                        raise ValueError(f"model path does not exist: {model_path}")
                    if not dataset_path.is_file():
                        raise ValueError(f"benchmark dataset is not a regular file: {dataset_path}")
                    output_dir = _normalized_output_dir(context)
                    if output_dir.exists() and not output_dir.is_dir():
                        raise ValueError(f"fresh output path is not a directory: {output_dir}")
                    if output_dir.is_dir() and any(output_dir.iterdir()):
                        raise ValueError(
                            "fresh output directory must be empty; benchmark artifacts are never overwritten"
                        )
                    if not BUNDLED_FRESH_RUNNER.is_file():
                        raise ValueError(f"bundled fresh benchmark runner is missing: {BUNDLED_FRESH_RUNNER}")
                    if not SHARED_MODEL_RUNTIME.is_file():
                        raise ValueError(f"shared model runtime is missing: {SHARED_MODEL_RUNTIME}")
            except (KeyError, OSError, TypeError, ValueError) as exc:
                diagnostics.append(_diagnostic("error", "benchmark.fresh_contract", str(exc)))
            return diagnostics

        if operation in NORMALIZE_OPERATIONS:
            try:
                if context.get("backend") != "local":
                    raise ValueError("benchmark normalization operations require the local backend")
                evidence = _evidence_specs(context)
                for item in evidence:
                    path = Path(item["path"])
                    if not path.is_file():
                        raise ValueError(f"evidence input is not a regular file: {path}")
                output_dir = _normalized_output_dir(context)
                if output_dir.exists() and not output_dir.is_dir():
                    raise ValueError(f"normalized output path is not a directory: {output_dir}")
                if output_dir.is_dir() and any(output_dir.iterdir()):
                    raise ValueError(
                        "normalized output directory must be fresh and empty; existing artifacts are never overwritten"
                    )
                _expected_identities(parameters)
                _units(parameters)
                split = parameters.get("split")
                if split not in (None, "") and (
                    not _plain_string(split) or not _IDENTIFIER.fullmatch(str(split))
                ):
                    raise ValueError("parameters.split must be a normalized identifier")
                source_script = _source_script_spec(context)
                if source_script is not None:
                    script_path = Path(source_script["path"])
                    if not script_path.is_file():
                        raise ValueError(f"source script is not a regular file: {script_path}")
                    if operation != "normalize-replay":
                        raise ValueError(
                            "source_script is supported only for normalize-replay provenance"
                        )
                if not BUNDLED_WRAPPER.is_file():
                    raise ValueError(f"bundled benchmark wrapper is missing: {BUNDLED_WRAPPER}")
            except (KeyError, OSError, TypeError, ValueError) as exc:
                diagnostics.append(
                    _diagnostic("error", "benchmark.normalize_contract", str(exc))
                )
            return diagnostics

        if context.get("backend") != "local":
            diagnostics.append(
                _diagnostic("error", "context.backend", "legacy benchmark operations require local")
            )
        if parameters.get("model_family") not in MODEL_FAMILIES:
            supported = ", ".join(sorted(MODEL_FAMILIES))
            diagnostics.append(
                _diagnostic(
                    "error",
                    "benchmark.model_family",
                    f"model_family must be one of the exact reviewed names: {supported}",
                )
            )
        for key in (
            "task",
            "scenario",
            "energy_normalization",
            "stress_convention",
        ):
            if not _plain_string(parameters.get(key)):
                diagnostics.append(_diagnostic("error", f"benchmark.{key}", f"parameters.{key} is required"))

        required_inputs = ["result_manifest"]
        if operation == "evaluate-static":
            required_inputs.extend(["executable", "script", "model", "benchmark_dataset", "benchmark_config"])
        for key in required_inputs:
            if _reference(inputs.get(key)) is None:
                diagnostics.append(_diagnostic("error", f"benchmark.input_{key}", f"inputs.{key} is required"))
        executable = _reference(inputs.get("executable"))
        if executable is not None and Path(executable).name.lower() in SHELL_EXECUTABLES:
            diagnostics.append(_diagnostic("error", "benchmark.shell_forbidden", "a shell executable is not permitted"))
        if operation == "evaluate-static" and _reference(inputs.get("result_manifest")) is not None:
            try:
                _safe_attempt_output(context, inputs["result_manifest"])
            except (KeyError, ValueError) as exc:
                diagnostics.append(_diagnostic("error", "benchmark.output_scope", str(exc)))
        return diagnostics

    def plan(self, context: dict[str, Any]) -> dict[str, Any]:
        diagnostics = self.validate(context)
        parameters = _mapping(context.get("parameters")) if isinstance(context, dict) else {}
        if parameters.get("operation", "evaluate-static") == "collect-existing" and not diagnostics:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "benchmark.collect_is_read_only",
                    "collect-existing is read-only; call check/collect or replay instead of executing a plan",
                )
            )
        if diagnostics:
            return {
                "plugin_id": PLUGIN_ID,
                "status": "BLOCKED",
                "executable": False,
                "diagnostics": diagnostics,
            }
        operation = str(parameters.get("operation", "evaluate-static"))
        if operation == FRESH_OPERATION:
            if context.get("backend") == "ssh-slurm":
                try:
                    return _plan_scheduled_fresh(context)
                except (KeyError, OSError, TypeError, ValueError) as exc:
                    return {
                        "plugin_id": PLUGIN_ID,
                        "status": "BLOCKED",
                        "executable": False,
                        "diagnostics": [
                            _diagnostic("error", "benchmark.scheduled_plan", str(exc))
                        ],
                    }
            model_path, dataset_path = _fresh_inputs(context)
            output_paths = _normalized_output_paths(context)
            argv = [
                sys.executable,
                str(BUNDLED_FRESH_RUNNER),
                "--model",
                str(model_path),
                "--dataset",
                str(dataset_path),
                "--output-dir",
                str(_normalized_output_dir(context)),
                "--model-family",
                str(parameters["model_family"]),
                "--task",
                str(parameters["task"]),
                "--scenario",
                str(parameters["scenario"]),
                "--split",
                str(parameters["split"]),
                "--energy-normalization",
                str(parameters["energy_normalization"]),
                "--device",
                str(parameters.get("device", "cpu")),
            ]
            for target in _fresh_targets(parameters):
                argv.extend(["--target", target])
                argv.extend([f"--{target}-unit", _units(parameters)[target]])
            if parameters.get("stress_convention") not in (None, ""):
                argv.extend(["--stress-convention", str(parameters["stress_convention"])])
            return {
                "plugin_id": PLUGIN_ID,
                "status": "READY",
                "executable": True,
                "argv": argv,
                "shell": False,
                "cwd": str(context["attempt_dir"]),
                "expected_outputs": [str(output_paths[name]) for name in FRESH_OUTPUT_NAMES],
                "input_paths": {
                    "model": str(model_path),
                    "dataset": str(dataset_path),
                },
                "calculation_claim": "fresh-model-inference",
                "resources": dict(_mapping(context.get("resources"))),
                "diagnostics": [],
            }
        if operation in NORMALIZE_OPERATIONS:
            evidence = _evidence_specs(context)
            output_paths = _normalized_output_paths(context)
            argv = [sys.executable, str(BUNDLED_WRAPPER), _normalize_mode(operation)]
            for item in evidence:
                argv.extend(["--input", item["path"]])
            argv.extend(["--output-dir", str(_normalized_output_dir(context))])
            if len(evidence) == 1:
                argv.extend(["--evidence-locator", evidence[0]["locator"]])
            else:
                for item in evidence:
                    argv.extend(["--input-locator", item["locator"]])
            for parameter, flag in (
                ("model_family", "--model"),
                ("task", "--task"),
                ("scenario", "--scenario"),
                ("split", "--split"),
            ):
                value = parameters.get(parameter)
                if value not in (None, ""):
                    argv.extend([flag, str(value)])
            for target, unit in sorted(_units(parameters).items()):
                argv.extend(["--{}-unit".format(target), unit])
            source_script = _source_script_spec(context)
            if source_script is not None:
                argv.extend(["--source-script", source_script["path"]])
                argv.extend(["--source-script-locator", source_script["locator"]])
            return {
                "plugin_id": PLUGIN_ID,
                "status": "READY",
                "executable": True,
                "argv": argv,
                "shell": False,
                "cwd": str(context["attempt_dir"]),
                "expected_outputs": [str(output_paths[name]) for name in NORMALIZED_OUTPUT_NAMES],
                "input_paths": [
                    {"path": item["locator"], "source": item["path"]}
                    for item in evidence
                ],
                "calculation_claim": (
                    "imported-existing-evidence"
                    if operation == "normalize-replay"
                    else "metrics-recomputed-from-supplied-reference-prediction-pairs; no model execution"
                ),
                "resources": dict(_mapping(context.get("resources"))),
                "diagnostics": [],
            }
        inputs = _mapping(context["inputs"])
        result_manifest = _safe_attempt_output(context, inputs["result_manifest"])
        argv = [
            _reference(inputs["executable"]),
            _project_path(context, inputs["script"]),
            "--model-family",
            str(parameters["model_family"]),
            "--model",
            _project_path(context, inputs["model"]),
            "--dataset",
            _project_path(context, inputs["benchmark_dataset"]),
            "--config",
            _project_path(context, inputs["benchmark_config"]),
            "--task",
            str(parameters["task"]),
            "--scenario",
            str(parameters["scenario"]),
            "--energy-normalization",
            str(parameters["energy_normalization"]),
            "--stress-convention",
            str(parameters["stress_convention"]),
            "--result-manifest",
            result_manifest,
        ]
        return {
            "plugin_id": PLUGIN_ID,
            "status": "READY",
            "executable": True,
            "argv": argv,
            "shell": False,
            "cwd": str(context["attempt_dir"]),
            "expected_outputs": [result_manifest],
            "resources": dict(_mapping(context.get("resources"))),
            "diagnostics": [],
        }

    def prepare(self, context: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(plan, dict) or plan.get("status") != "READY":
            return {
                "plugin_id": PLUGIN_ID,
                "status": "BLOCKED",
                "executable": False,
                "diagnostics": [_diagnostic("error", "benchmark.plan_required", "a READY plan is required")],
            }
        return {
            "plugin_id": PLUGIN_ID,
            "status": "READY",
            "executable": False,
            "prepared": False,
            "message": (
                "No files were written; the bundled fresh runner consumes pinned model/dataset inputs."
                if _operation(context) == FRESH_OPERATION
                else "No files were written; the wrapper consumes existing inputs."
            ),
        }

    def check(self, context: dict[str, Any]) -> dict[str, Any]:
        if _operation(context) in NORMALIZE_OPERATIONS or _operation(context) == FRESH_OPERATION:
            try:
                paths = _normalized_output_paths(context)
            except (KeyError, TypeError, ValueError) as exc:
                return {
                    "plugin_id": PLUGIN_ID,
                    "status": "FAIL",
                    "diagnostics": [
                        _diagnostic("error", "benchmark.normalized_output_path", str(exc))
                    ],
                }
            existing = {name for name, path in paths.items() if path.is_file()}
            if not existing:
                return {
                    "plugin_id": PLUGIN_ID,
                    "status": "WAIT",
                    "diagnostics": [
                        _diagnostic(
                            "info",
                            "benchmark.normalized_outputs_pending",
                            "normalized benchmark four-artifact suite is not present",
                        )
                    ],
                }
            expected_names = (
                set(FRESH_OUTPUT_NAMES)
                if _operation(context) == FRESH_OPERATION
                else set(NORMALIZED_OUTPUT_NAMES)
            )
            if existing != expected_names:
                missing = sorted(expected_names - existing)
                return {
                    "plugin_id": PLUGIN_ID,
                    "status": "FAIL",
                    "diagnostics": [
                        _diagnostic(
                            "error",
                            "benchmark.normalized_outputs_partial",
                            "normalized benchmark output is partial; missing: " + ", ".join(missing),
                        )
                    ],
                }
            output_dir = _normalized_output_dir(context)
            try:
                children = {item.name for item in output_dir.iterdir()}
            except OSError as exc:
                return {
                    "plugin_id": PLUGIN_ID,
                    "status": "FAIL",
                    "diagnostics": [
                        _diagnostic("error", "benchmark.normalized_outputs_unreadable", str(exc))
                    ],
                }
            if children != expected_names:
                unexpected = sorted(children - expected_names)
                return {
                    "plugin_id": PLUGIN_ID,
                    "status": "FAIL",
                    "diagnostics": [
                        _diagnostic(
                            "error",
                            "benchmark.normalized_outputs_not_fixed",
                            "normalized output directory must contain exactly four artifacts; unexpected: "
                            + ", ".join(unexpected),
                        )
                    ],
                }
            try:
                if _operation(context) == FRESH_OPERATION:
                    payload, metric_values, artifacts = _load_fresh_outputs(context, paths)
                    if context.get("backend") == "ssh-slurm":
                        _verify_cluster_fresh_report(context, paths)
                        artifacts.append(
                            {
                                "path": str(
                                    Path(str(context["attempt_dir"]))
                                    / "cluster-benchmark-report.json"
                                ),
                                "role": "benchmark-cluster-report",
                                "media_type": "application/json",
                            }
                        )
                else:
                    payload, metric_values, artifacts = _load_normalized_outputs(context, paths)
            except (
                OSError,
                UnicodeError,
                csv.Error,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                return {
                    "plugin_id": PLUGIN_ID,
                    "status": "FAIL",
                    "diagnostics": [
                        _diagnostic("error", "benchmark.normalized_contract", str(exc))
                    ],
                }
            return {
                "plugin_id": PLUGIN_ID,
                "status": "OK",
                "mode": payload["mode"],
                "records": payload["records"],
                "metrics": metric_values,
                "artifacts": artifacts,
                "diagnostics": [],
            }
        try:
            path = _result_path(context)
        except (KeyError, TypeError, ValueError) as exc:
            return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": [_diagnostic("error", "benchmark.result_path", str(exc))]}
        if not path.is_file():
            return {
                "plugin_id": PLUGIN_ID,
                "status": "WAIT",
                "diagnostics": [_diagnostic("info", "benchmark.result_pending", f"result manifest not found: {path}")],
            }
        raw, metrics, artifacts, diagnostics = _load_result(path, context)
        status = "FAIL" if diagnostics or raw is None or raw.get("status") == "FAIL" else "OK"
        if raw is not None and raw.get("status") == "FAIL" and not diagnostics:
            diagnostics.append(_diagnostic("error", "benchmark.reported_failure", "benchmark result reports FAIL"))
        return {
            "plugin_id": PLUGIN_ID,
            "status": status,
            "diagnostics": diagnostics,
            "metrics": metrics,
            "artifacts": artifacts,
        }

    def collect(self, context: dict[str, Any]) -> dict[str, Any]:
        checked = self.check(context)
        if checked["status"] != "OK":
            return checked
        result = {
            "plugin_id": PLUGIN_ID,
            "status": "OK",
            "artifacts": checked["artifacts"],
            "metrics": checked["metrics"],
            "diagnostics": [],
        }
        if "mode" in checked:
            result["mode"] = checked["mode"]
            result["records"] = checked["records"]
        return result

    def replay(self, context: dict[str, Any]) -> dict[str, Any]:
        """Replay is collection only and therefore cannot execute numerical code."""

        return self.collect(context)
