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
        for key in ("operation", "target", "input_manifest_fingerprint")
        if key in parameters
    }
    return copied


def _previous_attempt_path(context: dict[str, Any], attempt: int, name: str) -> Path:
    current = Path(str(context["attempt_dir"])).expanduser().absolute()
    return current.parent / f"attempt-{attempt}" / name


def _read_json(path: Path, max_bytes: int) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= max_bytes:
        raise ValueError(f"missing, unsafe or oversized JSON file: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return raw


def _pin_facade_helpers(plan: dict[str, Any]) -> dict[str, Any]:
    """Bind dynamic control-plane dependencies into the approval digest."""

    if plan.get("status") != "READY":
        return plan
    fingerprints = plan.setdefault("input_fingerprints", {})
    if not isinstance(fingerprints, dict):
        return {
            "plugin_id": PLUGIN_ID,
            "status": "BLOCKED",
            "executable": False,
            "diagnostics": [_diagnostic("error", "lammps.facade_fingerprints", "plan input_fingerprints must be an object")],
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
        fingerprints[key] = base._sha256(path)
    return plan


def _validate(context: dict[str, Any]) -> list[dict[str, str]]:
    diagnostics = list(base.Adapter().validate(_proxy(context)))
    if base._operation(context) != EXECUTE:
        return diagnostics
    parameters = _mapping(context.get("parameters"))
    allowed = {
        "operation",
        "target",
        "input_manifest_fingerprint",
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


def _runtime_identity_matches(
    runtime: dict[str, Any], identity: dict[str, Any], resources: dict[str, Any], previous: int
) -> None:
    expected = {
        "schema_version": 1,
        "runtime_contract": RUNTIME_CONTRACT,
        "attempt": previous,
        "framework": identity.get("framework"),
        "target": identity.get("target"),
        "input_manifest_fingerprint": identity.get("input_manifest_fingerprint"),
        "model_fingerprint": identity.get("model_fingerprint"),
        "model_kind": identity.get("model_kind"),
        "lammps_interface": identity.get("lammps_interface"),
        "checkpoint_interval": identity.get("checkpoint_interval"),
        "resources": resources,
    }
    for key, value in expected.items():
        if runtime.get(key) != value:
            raise ValueError(f"previous restart runtime identity mismatch for {key}")
    for key in (
        "lammps_executable_sha256",
        "launcher_prefix_sha256",
        "prepared_launcher_sha256",
    ):
        if not base._fingerprint(runtime.get(key)):
            raise ValueError(f"previous restart runtime lacks valid {key}")
    platform = runtime.get("platform")
    if not isinstance(platform, dict) or set(platform) != {"system", "machine", "byteorder"}:
        raise ValueError("previous restart runtime platform identity is invalid")


def _previous_restart(
    context: dict[str, Any], identity: dict[str, Any], attempt: int
) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    previous = attempt - 1
    final_manifest = _read_json(
        _previous_attempt_path(context, previous, "run-manifest.final.json"), base.MAX_JSON_BYTES
    )
    if final_manifest.get("state") not in {"FAIL", "STOPPED"}:
        raise ValueError("previous attempt is not failed/stopped")
    job = _mapping(final_manifest.get("job"))
    scheduler_state = str(job.get("scheduler_state", "")).upper()
    if scheduler_state not in RECOVERABLE_SCHEDULER_STATES:
        raise ValueError("previous attempt did not end in a restart-eligible scheduler terminal state")

    runtime_path = _previous_attempt_path(context, previous, "restart-runtime.json")
    runtime = _read_json(runtime_path, base.MAX_JSON_BYTES)
    resources = _mapping(context.get("resources"))
    _runtime_identity_matches(runtime, identity, resources, previous)

    candidates: dict[str, Path] = {}
    for name in CHECKPOINT_FILES:
        path = _previous_attempt_path(context, previous, name)
        if base._ordinary_file(path, base.MAX_RESTART_BYTES):
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
    identity = _mapping(plan.get("lammps_execution_identity"))
    total_steps = identity.get("steps")
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
    identity.update(
        {
            "restart_policy": policy,
            "checkpoint_interval": interval,
            "restart_from_attempt": None,
            "restart_runtime_sha256": None,
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
                    "max_bytes": base.MAX_RESTART_BYTES,
                    "role": "lammps-periodic-restart",
                }
            )
        fetch_outputs.append(
            {
                "remote_name": "restart-runtime.json",
                "remote_path": "output/restart-runtime.json",
                "local_name": "restart-runtime.json",
                "required": False,
                "max_bytes": base.MAX_JSON_BYTES,
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
            runtime_path, runtime, candidates = _previous_restart(context, identity, attempt)
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
            candidate_records[name] = {
                "sha256": base._sha256(path),
                "size_bytes": path.stat().st_size,
            }
        identity.update(
            {
                "restart_from_attempt": attempt - 1,
                "restart_runtime_sha256": base._sha256(runtime_path),
                "restart_candidates": candidate_records,
                "previous_runtime_executable_sha256": runtime["lammps_executable_sha256"],
                "previous_runtime_platform": runtime["platform"],
            }
        )

    plan["lammps_execution_identity"] = identity
    fingerprints = plan.setdefault("input_fingerprints", {})
    if isinstance(fingerprints, dict):
        fingerprints["restart_runner"] = base._sha256(restart_runner)
        fingerprints["restart_helper"] = base._sha256(restart_helper)
        if identity.get("restart_runtime_sha256"):
            fingerprints["restart_runtime"] = identity["restart_runtime_sha256"]
        for name, record in _mapping(identity.get("restart_candidates")).items():
            fingerprints[f"restart_{name}"] = record["sha256"]

    summary = _mapping(plan.get("approval_summary"))
    summary.update(
        {
            "checkpoint_interval": interval,
            "restart_policy": policy,
            "restart_from_attempt": identity.get("restart_from_attempt"),
            "restart_candidates": identity.get("restart_candidates"),
            "restart_start_step_resolved_on_compute_node": bool(identity.get("restart_from_attempt")),
            "binary_restart_runtime_bound": True,
            "bitwise_exact_restart_guaranteed": False,
            "failure_salvage": plan.get("failure_salvage"),
        }
    )
    plan["approval_summary"] = summary
    return plan


def _check_restart(context: dict[str, Any]) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    identity = _mapping(base._scheduled_plan(context).get("lammps_execution_identity"))
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    try:
        result = base._read_json(attempt / "lammps-execution-result.json")
    except Exception as exc:
        return [_diagnostic("error", "lammps.restart_result", str(exc))]
    restart = _mapping(result.get("restart"))
    resumed = identity.get("restart_from_attempt") is not None
    expected = {
        "policy": identity.get("restart_policy"),
        "checkpoint_interval": identity.get("checkpoint_interval"),
        "resumed": resumed,
        "from_attempt": identity.get("restart_from_attempt"),
        "runtime_compatibility_checked": resumed,
        "bitwise_exact_guaranteed": False,
    }
    for key, value in expected.items():
        if restart.get(key) != value:
            diagnostics.append(_diagnostic("error", f"lammps.restart_{key}", f"restart field {key} differs from approved plan"))
    start = result.get("segment_start_step")
    if resumed:
        interval = identity.get("checkpoint_interval")
        total = identity.get("steps")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or not isinstance(interval, int)
            or not isinstance(total, int)
            or not 0 < start < total
            or start % interval != 0
        ):
            diagnostics.append(_diagnostic("error", "lammps.restart_start_step", "selected restart timestep is not a valid periodic checkpoint"))
        selected = restart.get("selected_checkpoint_sha256")
        approved = {
            record.get("sha256")
            for record in _mapping(identity.get("restart_candidates")).values()
            if isinstance(record, dict)
        }
        if selected not in approved:
            diagnostics.append(_diagnostic("error", "lammps.restart_checkpoint", "selected restart SHA was not an approved salvaged candidate"))
        if restart.get("lammps_executable_sha256") != identity.get("previous_runtime_executable_sha256"):
            diagnostics.append(_diagnostic("error", "lammps.restart_executable", "resume did not use the previously bound LAMMPS executable identity"))
        if restart.get("platform") != identity.get("previous_runtime_platform"):
            diagnostics.append(_diagnostic("error", "lammps.restart_platform", "resume platform differs from the salvaged runtime identity"))
    else:
        if start != 0 or restart.get("selected_checkpoint_sha256") is not None:
            diagnostics.append(_diagnostic("error", "lammps.restart_fresh", "fresh execution reported unexpected restart state"))
    try:
        report = base._read_json(attempt / "cluster-run-report.json")
    except Exception as exc:
        diagnostics.append(_diagnostic("error", "lammps.restart_report", str(exc)))
        report = {}
    if (
        report.get("segment_start_step") != start
        or report.get("restart_from_attempt") != identity.get("restart_from_attempt")
        or report.get("selected_checkpoint_sha256") != restart.get("selected_checkpoint_sha256")
    ):
        diagnostics.append(_diagnostic("error", "lammps.restart_report_identity", "cluster restart report differs from execution result"))
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

    def prepare(self, context: Any, plan: Any) -> dict[str, Any]:
        return base.Adapter().prepare(_proxy(context) if isinstance(context, dict) else context, plan)

    def check(self, context: Any) -> dict[str, Any]:
        if not isinstance(context, dict) or base._operation(context) != EXECUTE:
            return base.Adapter().check(context)
        checked = base.Adapter().check(_proxy(context))
        if checked.get("status") != "OK":
            return checked
        diagnostics = _check_restart(context)
        if diagnostics:
            return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
        identity = _mapping(base._scheduled_plan(context).get("lammps_execution_identity"))
        result = base._read_json(Path(str(context["attempt_dir"])) / "lammps-execution-result.json")
        metrics = dict(_mapping(checked.get("metrics")))
        metrics["segment_start_step"] = float(result.get("segment_start_step", 0))
        metrics["resumed"] = 1.0 if identity.get("restart_from_attempt") is not None else 0.0
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

    def replay(self, context: Any) -> dict[str, Any]:
        if not isinstance(context, dict) or base._operation(context) != EXECUTE:
            return base.Adapter().replay(context)
        return self.collect(context)
