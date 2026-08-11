"""Safe external-wrapper planner for MLIP training frameworks.

This module is intentionally framework-free.  It translates reviewed,
explicit context fields into an argv list for a user-owned wrapper and parses
only a versioned result manifest.  It never invokes a shell or a trainer.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


PLUGIN_ID = "mlip-training"
FRAMEWORK_OPERATIONS = {
    "deepmd": frozenset({"train"}),
    "m3gnet": frozenset({"train", "finetune"}),
    "chgnet": frozenset({"train", "finetune"}),
    "mace": frozenset({"train", "finetune"}),
}
EXECUTION_BACKENDS = frozenset({"local"})
SHELL_EXECUTABLES = frozenset(
    {"bash", "csh", "cmd", "dash", "fish", "ksh", "powershell", "pwsh", "sh", "tcsh", "zsh"}
)
MAX_JSON_BYTES = 8 * 1024 * 1024


def _diagnostic(level: str, code: str, message: str) -> dict[str, str]:
    return {"level": level, "code": code, "message": message}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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


def _validate_metrics(value: Any) -> tuple[dict[str, float], list[dict[str, str]]]:
    if value is None:
        return {}, []
    if not isinstance(value, dict):
        return {}, [_diagnostic("error", "training.metrics_type", "metrics must be an object")]
    diagnostics: list[dict[str, str]] = []
    metrics: dict[str, float] = {}
    for name, raw in sorted(value.items()):
        if not isinstance(name, str) or not name:
            diagnostics.append(_diagnostic("error", "training.metric_name", "metric names must be non-empty strings"))
        elif isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            diagnostics.append(_diagnostic("error", "training.metric_value", f"metric {name!r} must be finite"))
        else:
            metrics[name] = float(raw)
    return metrics, diagnostics


def _load_result(path: Path, context: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, float], list[dict[str, Any]], list[dict[str, str]]]:
    diagnostics: list[dict[str, str]] = []
    try:
        raw = _read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return None, {}, [], [_diagnostic("error", "training.result_unreadable", str(exc))]
    if not isinstance(raw, dict):
        return None, {}, [], [_diagnostic("error", "training.result_type", "result manifest must be a JSON object")]
    parameters = _mapping(context.get("parameters"))
    if raw.get("schema_version") != 1:
        diagnostics.append(_diagnostic("error", "training.result_schema", "schema_version must equal 1"))
    if raw.get("plugin_id") != PLUGIN_ID:
        diagnostics.append(_diagnostic("error", "training.result_plugin", f"plugin_id must equal {PLUGIN_ID}"))
    if raw.get("status") not in {"OK", "FAIL"}:
        diagnostics.append(_diagnostic("error", "training.result_status", "status must be OK or FAIL"))
    if raw.get("framework") not in FRAMEWORK_OPERATIONS:
        diagnostics.append(_diagnostic("error", "training.result_framework", "framework is unsupported"))
    elif parameters.get("framework") in FRAMEWORK_OPERATIONS and raw.get("framework") != parameters.get("framework"):
        diagnostics.append(_diagnostic("error", "training.framework_mismatch", "result framework does not match the plan"))
    seed = raw.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        diagnostics.append(_diagnostic("error", "training.result_seed", "seed must be a non-negative integer"))
    elif isinstance(parameters.get("seed"), int) and seed != parameters.get("seed"):
        diagnostics.append(_diagnostic("error", "training.seed_mismatch", "result seed does not match the plan"))
    if not _plain_string(raw.get("framework_version")):
        diagnostics.append(
            _diagnostic("error", "training.result_framework_version", "framework_version must be a non-empty string")
        )
    for key in ("dataset_fingerprint", "config_fingerprint"):
        value = raw.get(key)
        if value is None or value == "":
            diagnostics.append(_diagnostic("error", f"training.result_{key}", f"{key} is required"))
        elif value != parameters.get(key):
            diagnostics.append(
                _diagnostic("error", f"training.{key}_mismatch", f"result {key} does not match the approved plan")
            )
    for key in ("operation", "device", "precision"):
        if not _plain_string(raw.get(key)):
            diagnostics.append(
                _diagnostic("error", f"training.result_{key}", f"result {key} is required")
            )
        elif raw.get(key) != parameters.get(key):
            diagnostics.append(
                _diagnostic("error", f"training.{key}_mismatch", f"result {key} does not match the approved plan")
            )
    if parameters.get("operation") == "finetune":
        value = raw.get("foundation_model_fingerprint")
        if value != parameters.get("foundation_model_fingerprint"):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "training.foundation_model_fingerprint_mismatch",
                    "result foundation model fingerprint does not match the approved plan",
                )
            )

    model_path: Path | None = None
    model = raw.get("model_artifact")
    if not isinstance(model, dict):
        diagnostics.append(_diagnostic("error", "training.model_artifact", "model_artifact must be an object"))
    else:
        try:
            model_path = _portable_child(path.parent, model.get("path"), "model_artifact.path")
        except ValueError as exc:
            diagnostics.append(_diagnostic("error", "training.model_path", str(exc)))
        if model_path is not None:
            if not model_path.is_file():
                diagnostics.append(_diagnostic("error", "training.model_missing", f"model artifact is not a regular file: {model_path}"))
            try:
                expected = Path(_safe_attempt_output(context, _mapping(context.get("inputs")).get("output")))
                if model_path != expected:
                    diagnostics.append(
                        _diagnostic("error", "training.model_output_mismatch", "model artifact does not match inputs.output")
                    )
            except (KeyError, ValueError):
                pass

    metrics, metric_diagnostics = _validate_metrics(raw.get("metrics"))
    diagnostics.extend(metric_diagnostics)
    artifacts = [{"path": str(path), "role": "result-manifest", "media_type": "application/json"}]
    if model_path is not None:
        artifacts.append(
            {
                "path": str(model_path),
                "role": "model",
                "media_type": model.get("media_type", "application/octet-stream") if isinstance(model, dict) else None,
            }
        )
    return raw, metrics, artifacts, diagnostics


class Adapter:
    """Build deterministic wrapper argv for four declared MLIP frameworks."""

    def validate(self, context: dict[str, Any]) -> list[dict[str, str]]:
        diagnostics: list[dict[str, str]] = []
        if not isinstance(context, dict):
            return [_diagnostic("error", "context.type", "context must be an object")]
        for key in ("project_root", "attempt_dir"):
            if not _plain_string(context.get(key)):
                diagnostics.append(_diagnostic("error", f"context.{key}", f"{key} must be a non-empty path string"))
        inputs = context.get("inputs")
        parameters = context.get("parameters")
        if not isinstance(inputs, dict):
            diagnostics.append(_diagnostic("error", "context.inputs", "inputs must be an object"))
            inputs = {}
        if not isinstance(parameters, dict):
            diagnostics.append(_diagnostic("error", "context.parameters", "parameters must be an object"))
            parameters = {}
        if not isinstance(context.get("resources"), dict):
            diagnostics.append(_diagnostic("error", "context.resources", "resources must be an object"))
        if context.get("backend") not in EXECUTION_BACKENDS:
            diagnostics.append(_diagnostic("error", "context.backend", "this adapter currently supports only the local backend"))

        framework = parameters.get("framework")
        operation = parameters.get("operation", "train")
        if framework not in FRAMEWORK_OPERATIONS:
            diagnostics.append(_diagnostic("error", "training.framework", "framework must be deepmd, m3gnet, chgnet, or mace"))
        elif operation not in FRAMEWORK_OPERATIONS[framework]:
            allowed = ", ".join(sorted(FRAMEWORK_OPERATIONS[framework]))
            diagnostics.append(_diagnostic("error", "training.operation", f"{framework} operation must be one of: {allowed}"))
        seed = parameters.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            diagnostics.append(_diagnostic("error", "training.seed", "parameters.seed must be a non-negative integer"))
        for key in ("device", "precision", "dataset_fingerprint", "config_fingerprint"):
            if not _plain_string(parameters.get(key)):
                diagnostics.append(_diagnostic("error", f"training.{key}", f"parameters.{key} is required"))

        for key in ("executable", "script", "config", "data", "output", "result_manifest"):
            if _reference(inputs.get(key)) is None:
                diagnostics.append(_diagnostic("error", f"training.input_{key}", f"inputs.{key} is required"))
        executable = _reference(inputs.get("executable"))
        if executable is not None and Path(executable).name.lower() in SHELL_EXECUTABLES:
            diagnostics.append(_diagnostic("error", "training.shell_forbidden", "a shell executable is not permitted"))
        if operation == "finetune" and _reference(inputs.get("foundation_model")) is None:
            diagnostics.append(
                _diagnostic("error", "training.foundation_model", "inputs.foundation_model is required for finetune")
            )
        if operation == "finetune" and not _plain_string(
            parameters.get("foundation_model_fingerprint")
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "training.foundation_model_fingerprint",
                    "parameters.foundation_model_fingerprint is required for finetune",
                )
            )
        for key in ("output", "result_manifest"):
            if _reference(inputs.get(key)) is not None:
                try:
                    _safe_attempt_output(context, inputs[key])
                except (KeyError, ValueError) as exc:
                    diagnostics.append(_diagnostic("error", f"training.{key}_scope", str(exc)))
        return diagnostics

    def plan(self, context: dict[str, Any]) -> dict[str, Any]:
        diagnostics = self.validate(context)
        if diagnostics:
            return {
                "plugin_id": PLUGIN_ID,
                "status": "BLOCKED",
                "executable": False,
                "diagnostics": diagnostics,
            }
        inputs = _mapping(context["inputs"])
        parameters = _mapping(context["parameters"])
        model_output = _safe_attempt_output(context, inputs["output"])
        result_manifest = _safe_attempt_output(context, inputs["result_manifest"])
        argv = [
            _reference(inputs["executable"]),
            _project_path(context, inputs["script"]),
            "--framework",
            str(parameters["framework"]),
            "--operation",
            str(parameters.get("operation", "train")),
            "--config",
            _project_path(context, inputs["config"]),
            "--data",
            _project_path(context, inputs["data"]),
            "--output",
            model_output,
            "--result-manifest",
            result_manifest,
            "--seed",
            str(parameters["seed"]),
            "--device",
            str(parameters["device"]),
            "--precision",
            str(parameters["precision"]),
            "--dataset-fingerprint",
            str(parameters["dataset_fingerprint"]),
            "--config-fingerprint",
            str(parameters["config_fingerprint"]),
        ]
        if parameters.get("operation", "train") == "finetune":
            argv.extend(["--foundation-model", _project_path(context, inputs["foundation_model"])])
            argv.extend(
                [
                    "--foundation-model-fingerprint",
                    str(parameters["foundation_model_fingerprint"]),
                ]
            )
        return {
            "plugin_id": PLUGIN_ID,
            "status": "READY",
            "executable": True,
            "argv": argv,
            "cwd": str(context["attempt_dir"]),
            "expected_outputs": [model_output, result_manifest],
            "resources": dict(_mapping(context.get("resources"))),
            "diagnostics": [],
        }

    def prepare(self, context: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(plan, dict) or plan.get("status") != "READY":
            return {
                "plugin_id": PLUGIN_ID,
                "status": "BLOCKED",
                "executable": False,
                "diagnostics": [_diagnostic("error", "training.plan_required", "a READY plan is required")],
            }
        return {
            "plugin_id": PLUGIN_ID,
            "status": "READY",
            "executable": False,
            "prepared": False,
            "message": "No files were written; the external wrapper consumes existing inputs.",
        }

    def check(self, context: dict[str, Any]) -> dict[str, Any]:
        try:
            path = Path(_safe_attempt_output(context, _mapping(context.get("inputs")).get("result_manifest")))
        except (KeyError, TypeError, ValueError) as exc:
            return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": [_diagnostic("error", "training.result_path", str(exc))]}
        if not path.is_file():
            return {
                "plugin_id": PLUGIN_ID,
                "status": "WAIT",
                "diagnostics": [_diagnostic("info", "training.result_pending", f"result manifest not found: {path}")],
            }
        raw, metrics, artifacts, diagnostics = _load_result(path, context)
        status = "FAIL" if diagnostics or raw is None or raw.get("status") == "FAIL" else "OK"
        if raw is not None and raw.get("status") == "FAIL" and not diagnostics:
            diagnostics.append(_diagnostic("error", "training.reported_failure", "training result reports FAIL"))
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
        return {
            "plugin_id": PLUGIN_ID,
            "status": "OK",
            "artifacts": checked["artifacts"],
            "metrics": checked["metrics"],
            "diagnostics": [],
        }

    def replay(self, context: dict[str, Any]) -> dict[str, Any]:
        """Replay only parses an already existing manifest and model reference."""

        return self.collect(context)
