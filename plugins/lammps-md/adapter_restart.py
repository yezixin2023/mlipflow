"""LAMMPS facade adding periodic binary restart + approved retry.

The v0.2 prepare/execute adapter remains the scientific execution contract.  This
facade adds an explicit checkpoint cadence, failure salvage of two alternating
LAMMPS restart files plus a runtime-compatibility record, and fresh-attempt
resume after scheduler interruption.  It never treats a normal scientific FAIL
as restart-eligible.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

PLUGIN_ID = "lammps-md"
EXECUTE = "execute"
RESTART_DISABLED = "disabled"
RESTART_AUTO = "auto-from-previous-attempt"
RESTART_POLICIES = {RESTART_DISABLED, RESTART_AUTO}
RUNTIME_CONTRACT = "lammps-restart-runtime-v1"
RECOVERABLE_SCHEDULER_STATES = {
    "FAILED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "DEADLINE",
    "CANCELLED",
    "REVOKED",
}
CHECKPOINT_FILES = ("checkpoint.1.restart", "checkpoint.2.restart")


def _load_base():
    path = Path(__file__).resolve().with_name("adapter_execute.py")
    spec = importlib.util.spec_from_file_location("_mlipflow_lammps_execute_restart_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load bundled adapter_execute.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base()


def _diagnostic(level: str, code: str, message: str) -> dict[str, str]:
    return {"level": level, "code": code, "message": message}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _attempt_number(context: dict[str, Any]) -> int:
    try:
        name = Path(str(context["attempt_dir"])).name
        prefix, raw = name.rsplit("-", 1)
        if prefix != "attempt":
            raise ValueError
        attempt = int(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("attempt_dir must end in attempt-<positive integer>") from exc
    if attempt < 1:
        raise ValueError("attempt number must be positive")
    return attempt


def _proxy(context: dict[str, Any]) -> dict[str, Any]:
    if base._operation(context) != EXECUTE:
        return context
    copied = dict(context)
    parameters = _mapping(context.get("parameters"))
    copied["parameters"] = {
        key: parameters[key]
        for key in ("operation", "target")
        if key in parameters
    }
    return copied


def _previous_attempt_path(context: dict[str, Any], attempt: int, name: str) -> Path:
    current = Path(str(context["attempt_dir"])).expanduser().absolute()
    return current.parent / f"attempt-{attempt}" / name


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing or empty JSON file: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return raw


def _pin_facade_helpers(plan: dict[str, Any]) -> dict[str, Any]:
    """Record the facade helper paths in the plan."""

    if plan.get("status") != "READY":
        return plan
    paths = plan.setdefault("input_paths", {})
    if not isinstance(paths, dict):
        return {
            "plugin_id": PLUGIN_ID,
            "status": "BLOCKED",
            "executable": False,
            "diagnostics": [_diagnostic("error", "lammps.facade_paths", "plan input_paths must be an object")],
        }
    for filename, key in (
        ("adapter_execute.py", "adapter_execute"),
        ("adapter.py", "adapter_legacy"),
    ):
        path = Path(__file__).resolve().with_name(filename)
        if not base._ordinary_file(path):
            return {
                "plugin_id": PLUGIN_ID,
                "status": "BLOCKED",
                "executable": False,
                "diagnostics": [_diagnostic("error", "lammps.facade_helper", f"missing bundled helper: {filename}")],
            }
        paths[key] = str(path)
    return plan


def _validate(context: dict[str, Any]) -> list[dict[str, str]]:
    diagnostics = list(base.Adapter().validate(_proxy(context)))
    if base._operation(context) != EXECUTE:
        return diagnostics
    parameters = _mapping(context.get("parameters"))
    allowed = {
        "operation",
        "target",
        "restart_policy",
        "checkpoint_interval",
    }
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        diagnostics.append(
            _diagnostic("error", "lammps.restart_parameters", "unsupported execute parameter(s): " + ", ".join(unknown))
        )
    policy = parameters.get("restart_policy", RESTART_DISABLED)
    if policy not in RESTART_POLICIES:
        diagnostics.append(
            _diagnostic(
                "error",
                "lammps.restart_policy",
                "restart_policy must be disabled or auto-from-previous-attempt",
            )
        )
    interval = parameters.get("checkpoint_interval")
    if interval is not None and (
        isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0
    ):
        diagnostics.append(
            _diagnostic("error", "lammps.checkpoint_interval", "checkpoint_interval must be positive")
        )
    if policy == RESTART_AUTO and interval is None:
        diagnostics.append(
            _diagnostic(
                "error",
                "lammps.restart_checkpoint_interval",
                "auto restart requires an explicit checkpoint_interval",
            )
        )
    try:
        _attempt_number(context)
    except ValueError as exc:
        diagnostics.append(_diagnostic("error", "lammps.attempt", str(exc)))
    return diagnostics


def _runtime_matches(
    runtime: dict[str, Any], calculation: dict[str, Any], resources: dict[str, Any], previous: int
) -> None:
    expected = {
        "schema_version": 1,
        "runtime_contract": RUNTIME_CONTRACT,
        "attempt": previous,
        "framework": calculation.get("framework"),
        "target": calculation.get("target"),
        "input_manifest_path": calculation.get("input_manifest_path"),
        "model_path": calculation.get("model_path"),
        "model_kind": calculation.get("model_kind"),
        "lammps_interface": calculation.get("lammps_interface"),
        "checkpoint_interval": calculation.get("checkpoint_interval"),
        "resources": resources,
    }
    for key, value in expected.items():
        if runtime.get(key) != value:
            raise ValueError(f"previous restart runtime setting mismatch for {key}")
    if not base._plain(runtime.get("lammps_executable")):
        raise ValueError("previous restart runtime lacks the LAMMPS executable path")
    if not isinstance(runtime.get("launcher_prefix"), list) or not isinstance(
        runtime.get("prepared_launcher"), list
    ):
        raise ValueError("previous restart runtime lacks launcher arguments")
    platform = runtime.get("platform")
    if not isinstance(platform, dict) or set(platform) != {"system", "machine", "byteorder"}:
        raise ValueError("previous restart runtime platform record is invalid")


def _previous_restart(
    context: dict[str, Any], calculation: dict[str, Any], attempt: int
) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    previous = attempt - 1
    final_manifest = _read_json(
        _previous_attempt_path(context, previous, "run-manifest.final.json")
    )
    if final_manifest.get("state") not in {"FAIL", "STOPPED"}:
        raise ValueError("previous attempt is not failed/stopped")
    job = _mapping(final_manifest.get("job"))
    scheduler_state = str(job.get("scheduler_state", "")).upper()
    if scheduler_state not in RECOVERABLE_SCHEDULER_STATES:
        raise ValueError("previous attempt did not end in a restart-eligible scheduler terminal state")

    runtime_path = _previous_attempt_path(context, previous, "restart-runtime.json")
    runtime = _read_json(runtime_path)
    resources = _mapping(context.get("resources"))
    _runtime_matches(runtime, calculation, resources, previous)

    candidates: dict[str, Path] = {}
    for name in CHECKPOINT_FILES:
        path = _previous_attempt_path(context, previous, name)
        if base._ordinary_file(path):
            candidates[name] = path
    if not candidates:
        raise ValueError("previous scheduler interruption has no salvaged periodic restart file")
    return runtime_path, runtime, candidates


def _plan_execute(context: dict[str, Any]) -> dict[str, Any]:
    diagnostics = _validate(context)
    if diagnostics:
        return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": diagnostics}
    plan = _pin_facade_helpers(base.Adapter().plan(_proxy(context)))
    if plan.get("status") != "READY":
        return plan

    parameters = _mapping(context["parameters"])
    interval = parameters.get("checkpoint_interval")
    policy = str(parameters.get("restart_policy", RESTART_DISABLED))
    calculation = _mapping(plan.get("lammps_calculation"))
    total_steps = calculation.get("steps")
    if interval is not None and (
        not isinstance(total_steps, int) or isinstance(total_steps, bool) or interval > total_steps
    ):
        return {
            "plugin_id": PLUGIN_ID,
            "status": "BLOCKED",
            "executable": False,
            "diagnostics": [_diagnostic("error", "lammps.checkpoint_interval", "checkpoint_interval cannot exceed approved total steps")],
        }

    scheduled = _mapping(plan.get("scheduled_execution"))
    staged = scheduled.get("staged_files")
    fetch_outputs = scheduled.get("fetch_outputs")
    if not isinstance(staged, list) or not isinstance(fetch_outputs, list):
        return {
            "plugin_id": PLUGIN_ID,
            "status": "BLOCKED",
            "executable": False,
            "diagnostics": [_diagnostic("error", "lammps.restart_contract", "base scheduled contract is incomplete")],
        }

    restart_runner = Path(__file__).resolve().with_name("lammps_cluster_restart.py")
    restart_helper = Path(__file__).resolve().with_name("lammps_restart.py")
    for path, remote_name in (
        (restart_runner, "lammps_cluster_restart.py"),
        (restart_helper, "lammps_restart.py"),
    ):
        if not base._ordinary_file(path):
            return {
                "plugin_id": PLUGIN_ID,
                "status": "BLOCKED",
                "executable": False,
                "diagnostics": [_diagnostic("error", "lammps.restart_helper", f"missing bundled helper: {path.name}")],
            }
        staged.append(base._staged(path, remote_name))

    attempt = _attempt_number(context)
    calculation.update(
        {
            "restart_policy": policy,
            "checkpoint_interval": interval,
            "restart_from_attempt": None,
            "restart_runtime_path": None,
            "restart_candidates": {},
        }
    )

    if interval is not None:
        for name in CHECKPOINT_FILES:
            fetch_outputs.append(
                {
                    "remote_name": name,
                    "remote_path": f"output/{name}",
                    "local_name": name,
                    "required": False,
                    "role": "lammps-periodic-restart",
                }
            )
        fetch_outputs.append(
            {
                "remote_name": "restart-runtime.json",
                "remote_path": "output/restart-runtime.json",
                "local_name": "restart-runtime.json",
                "required": False,
                "role": "lammps-restart-runtime",
            }
        )
        salvage = _mapping(plan.get("failure_salvage"))
        names = salvage.get("fetch_remote_names")
        if not isinstance(names, list):
            names = []
        salvage["schema_version"] = 1
        salvage["fetch_remote_names"] = list(dict.fromkeys([*names, *CHECKPOINT_FILES, "restart-runtime.json"]))
        plan["failure_salvage"] = salvage

    if policy == RESTART_AUTO and attempt > 1:
        try:
            runtime_path, runtime, candidates = _previous_restart(context, calculation, attempt)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            return {
                "plugin_id": PLUGIN_ID,
                "status": "BLOCKED",
                "executable": False,
                "diagnostics": [_diagnostic("error", "lammps.restart_previous", str(exc))],
            }
        staged.append(base._staged(runtime_path, "restart/restart-runtime.json"))
        candidate_records: dict[str, dict[str, Any]] = {}
        for name, path in sorted(candidates.items()):
            staged.append(base._staged(path, f"restart/{name}"))
            candidate_records[name] = {"path": str(path), "source_attempt": attempt - 1}
        calculation.update(
            {
                "restart_from_attempt": attempt - 1,
                "restart_runtime_path": str(runtime_path),
                "restart_candidates": candidate_records,
                "previous_runtime_executable": runtime["lammps_executable"],
                "previous_launcher_prefix": runtime["launcher_prefix"],
                "previous_prepared_launcher": runtime["prepared_launcher"],
                "previous_runtime_platform": runtime["platform"],
            }
        )

    plan["lammps_calculation"] = calculation
    paths = plan.setdefault("input_paths", {})
    if isinstance(paths, dict):
        paths["restart_runner"] = str(restart_runner)
        paths["restart_helper"] = str(restart_helper)
        if calculation.get("restart_runtime_path"):
            paths["restart_runtime"] = calculation["restart_runtime_path"]
        for name, record in _mapping(calculation.get("restart_candidates")).items():
            paths[f"restart_{name}"] = record["path"]

    summary = _mapping(plan.get("approval_summary"))
    summary.update(
        {
            "checkpoint_interval": interval,
            "restart_policy": policy,
            "restart_from_attempt": calculation.get("restart_from_attempt"),
            "restart_candidates": calculation.get("restart_candidates"),
            "restart_start_step_resolved_on_compute_node": bool(calculation.get("restart_from_attempt")),
            "binary_restart_runtime_bound": True,
            "bitwise_exact_restart_guaranteed": False,
            "failure_salvage": plan.get("failure_salvage"),
        }
    )
    plan["approval_summary"] = summary
    return plan


def _check_restart(context: dict[str, Any]) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    calculation = _mapping(base._scheduled_plan(context).get("lammps_calculation"))
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    try:
        result = base._read_json(attempt / "lammps-execution-result.json")
    except Exception as exc:
        return [_diagnostic("error", "lammps.restart_result", str(exc))]
    restart = _mapping(result.get("restart"))
    resumed = calculation.get("restart_from_attempt") is not None
    expected = {
        "policy": calculation.get("restart_policy"),
        "checkpoint_interval": calculation.get("checkpoint_interval"),
        "resumed": resumed,
        "from_attempt": calculation.get("restart_from_attempt"),
        "runtime_compatibility_checked": resumed,
        "bitwise_exact_guaranteed": False,
    }
    for key, value in expected.items():
        if restart.get(key) != value:
            diagnostics.append(_diagnostic("error", f"lammps.restart_{key}", f"restart field {key} differs from approved plan"))
    start = result.get("segment_start_step")
    if resumed:
        interval = calculation.get("checkpoint_interval")
        total = calculation.get("steps")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or not isinstance(interval, int)
            or not isinstance(total, int)
            or not 0 < start < total
            or start % interval != 0
        ):
            diagnostics.append(_diagnostic("error", "lammps.restart_start_step", "selected restart timestep is not a valid periodic checkpoint"))
        selected = restart.get("selected_checkpoint")
        approved = set(_mapping(calculation.get("restart_candidates")))
        if selected not in approved:
            diagnostics.append(_diagnostic("error", "lammps.restart_checkpoint", "selected restart path was not a staged salvaged candidate"))
        if restart.get("lammps_executable") != calculation.get("previous_runtime_executable"):
            diagnostics.append(_diagnostic("error", "lammps.restart_executable", "resume used a different LAMMPS executable path"))
        if restart.get("launcher_prefix") != calculation.get("previous_launcher_prefix"):
            diagnostics.append(_diagnostic("error", "lammps.restart_launcher", "resume used different site launcher arguments"))
        if restart.get("prepared_launcher") != calculation.get("previous_prepared_launcher"):
            diagnostics.append(_diagnostic("error", "lammps.restart_prepared_launcher", "resume used different prepared launcher arguments"))
        if restart.get("platform") != calculation.get("previous_runtime_platform"):
            diagnostics.append(_diagnostic("error", "lammps.restart_platform", "resume platform differs from the salvaged runtime record"))
    else:
        if start != 0 or restart.get("selected_checkpoint") is not None:
            diagnostics.append(_diagnostic("error", "lammps.restart_fresh", "fresh execution reported unexpected restart state"))
    try:
        report = base._read_json(attempt / "cluster-run-report.json")
    except Exception as exc:
        diagnostics.append(_diagnostic("error", "lammps.restart_report", str(exc)))
        report = {}
    if (
        report.get("segment_start_step") != start
        or report.get("restart_from_attempt") != calculation.get("restart_from_attempt")
        or report.get("selected_checkpoint") != restart.get("selected_checkpoint")
    ):
        diagnostics.append(_diagnostic("error", "lammps.restart_report", "cluster restart report differs from execution result"))
    return diagnostics


class Adapter:
    def validate(self, context: Any) -> list[dict[str, str]]:
        if not isinstance(context, dict):
            return base.Adapter().validate(context)
        return _validate(context)

    def plan(self, context: Any) -> dict[str, Any]:
        if not isinstance(context, dict):
            return base.Adapter().plan(context)
        if base._operation(context) != EXECUTE:
            return _pin_facade_helpers(base.Adapter().plan(context))
        return _plan_execute(context)


    def check(self, context: Any) -> dict[str, Any]:
        if not isinstance(context, dict) or base._operation(context) != EXECUTE:
            return base.Adapter().check(context)
        checked = base.Adapter().check(_proxy(context))
        if checked.get("status") != "OK":
            return checked
        diagnostics = _check_restart(context)
        if diagnostics:
            return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
        calculation = _mapping(base._scheduled_plan(context).get("lammps_calculation"))
        result = base._read_json(Path(str(context["attempt_dir"])) / "lammps-execution-result.json")
        metrics = dict(_mapping(checked.get("metrics")))
        metrics["segment_start_step"] = float(result.get("segment_start_step", 0))
        metrics["resumed"] = 1.0 if calculation.get("restart_from_attempt") is not None else 0.0
        checked["metrics"] = metrics
        return checked

    def collect(self, context: Any) -> dict[str, Any]:
        if not isinstance(context, dict) or base._operation(context) != EXECUTE:
            return base.Adapter().collect(context)
        checked = self.check(context)
        if checked.get("status") != "OK":
            return checked
        collected = base.Adapter().collect(_proxy(context))
        if collected.get("status") == "OK":
            collected["metrics"] = checked.get("metrics", {})
        return collected
