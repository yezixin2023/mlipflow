"""ASE MD facade adding approved checkpoint/restart across scheduler attempts.

The NVT/NPT scientific contracts remain in adapter_npt.py. This facade adds a
periodic checkpoint output, terminal-failure salvage allowlist, and an explicit
``auto-from-previous-attempt`` retry policy. A retry stages only a checkpoint that
was already salvaged into the previous local attempt and binds its SHA-256 and
completed step into the new approval plan.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

PLUGIN_ID = "ase-md"
RESTART_DISABLED = "disabled"
RESTART_AUTO = "auto-from-previous-attempt"
RESTART_POLICIES = {RESTART_DISABLED, RESTART_AUTO}
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
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


def _load_npt():
    path = Path(__file__).resolve().with_name("adapter_npt.py")
    spec = importlib.util.spec_from_file_location("_mlipflow_ase_md_npt_restart_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load bundled adapter_npt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_npt()
legacy = base.legacy


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


def _positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _checkpoint_path(context: dict[str, Any], attempt: int) -> Path:
    current = Path(str(context["attempt_dir"])).expanduser().absolute()
    return current.parent / f"attempt-{attempt}" / "md-checkpoint.json"


def _final_manifest_path(context: dict[str, Any], attempt: int) -> Path:
    current = Path(str(context["attempt_dir"])).expanduser().absolute()
    return current.parent / f"attempt-{attempt}" / "run-manifest.final.json"


def _read_json(path: Path, max_bytes: int) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= max_bytes:
        raise ValueError(f"missing, unsafe or oversized JSON file: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return raw


def _segment_steps(start: int, total: int, interval: int) -> list[int]:
    values = [start]
    next_step = ((start // interval) + 1) * interval
    values.extend(range(next_step, total + 1, interval))
    if values[-1] != total:
        values.append(total)
    return values


def _checkpoint_matches_identity(checkpoint: dict[str, Any], identity: dict[str, Any]) -> int:
    expected = {
        "schema_version": 1,
        "plugin_id": PLUGIN_ID,
        "checkpoint_state_version": "ase-md-checkpoint-v1",
        "calculator": identity.get("calculator"),
        "ensemble": identity.get("ensemble"),
        "structure_fingerprint": identity.get("structure_fingerprint"),
        "temperature_K": identity.get("temperature_k"),
        "timestep_fs": identity.get("timestep_fs"),
        "steps_requested": identity.get("steps"),
        "seed": identity.get("seed"),
        "device": identity.get("device"),
        "default_dtype": identity.get("default_dtype"),
        "fix_com": identity.get("fix_com"),
    }
    if "supercell_repeat" in identity:
        expected.update(
            {
                "supercell_repeat": identity.get("supercell_repeat"),
                "minimum_initial_cell_length_A": identity.get(
                    "minimum_initial_cell_length_angstrom"
                ),
            }
        )
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"previous checkpoint identity mismatch for {key}")
    model = _mapping(checkpoint.get("model"))
    if (
        model.get("id") != identity.get("model_id")
        or model.get("fingerprint") != identity.get("model_fingerprint")
    ):
        raise ValueError("previous checkpoint model identity differs from the approved model")
    if identity.get("ensemble") == base.NVT:
        if checkpoint.get("friction_per_fs") != identity.get("friction_per_fs"):
            raise ValueError("previous checkpoint Langevin friction differs")
        if checkpoint.get("rng_algorithm") != "PCG64":
            raise ValueError("previous checkpoint RNG algorithm is not PCG64")
    else:
        for key, identity_key in (
            ("pressure_GPa", "pressure_gpa"),
            ("thermostat_damping_fs", "thermostat_damping_fs"),
            ("barostat_damping_fs", "barostat_damping_fs"),
            ("thermostat_chain_length", "thermostat_chain_length"),
            ("barostat_chain_length", "barostat_chain_length"),
            ("thermostat_substeps", "thermostat_substeps"),
            ("barostat_substeps", "barostat_substeps"),
        ):
            if checkpoint.get(key) != identity.get(identity_key):
                raise ValueError(f"previous checkpoint NPT identity mismatch for {key}")
    completed = checkpoint.get("completed_steps")
    total = identity.get("steps")
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or not isinstance(total, int)
        or not 0 <= completed < total
    ):
        raise ValueError("previous checkpoint completed_steps cannot be resumed")
    if not isinstance(checkpoint.get("ase_version"), str) or not checkpoint["ase_version"]:
        raise ValueError("previous checkpoint does not record ASE version")
    return completed


def _previous_checkpoint(
    context: dict[str, Any], identity: dict[str, Any], attempt: int
) -> tuple[Path, dict[str, Any], int] | None:
    if attempt == 1:
        return None
    previous = attempt - 1
    manifest_path = _final_manifest_path(context, previous)
    manifest = _read_json(manifest_path, legacy.MAX_JSON_BYTES)
    if manifest.get("state") not in {"FAIL", "STOPPED"}:
        raise ValueError("previous attempt is not a failed/stopped scheduler attempt")
    job = _mapping(manifest.get("job"))
    scheduler_state = str(job.get("scheduler_state", "")).upper()
    if scheduler_state not in RECOVERABLE_SCHEDULER_STATES:
        raise ValueError(
            "previous attempt did not end in a restart-eligible scheduler terminal state"
        )
    checkpoint_path = _checkpoint_path(context, previous)
    checkpoint = _read_json(checkpoint_path, MAX_CHECKPOINT_BYTES)
    completed = _checkpoint_matches_identity(checkpoint, identity)
    return checkpoint_path, checkpoint, completed


def _validate_restart(context: dict[str, Any]) -> list[dict[str, str]]:
    diagnostics = list(base.Adapter().validate(context))
    parameters = _mapping(context.get("parameters"))
    policy = parameters.get("restart_policy", RESTART_DISABLED)
    if policy not in RESTART_POLICIES:
        diagnostics.append(
            _diagnostic(
                "error",
                "ase_md.restart_policy",
                "restart_policy must be disabled or auto-from-previous-attempt",
            )
        )
    interval = parameters.get("checkpoint_interval")
    if interval is not None:
        if not _positive_int(interval):
            diagnostics.append(
                _diagnostic(
                    "error", "ase_md.checkpoint_interval", "checkpoint_interval must be positive"
                )
            )
        elif _positive_int(parameters.get("steps")) and interval > parameters["steps"]:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "ase_md.checkpoint_interval",
                    "checkpoint_interval cannot exceed total steps",
                )
            )
    if policy == RESTART_AUTO and interval is None:
        diagnostics.append(
            _diagnostic(
                "error",
                "ase_md.restart_checkpoint_interval",
                "auto restart requires an explicit checkpoint_interval",
            )
        )
    try:
        _attempt_number(context)
    except ValueError as exc:
        diagnostics.append(_diagnostic("error", "ase_md.attempt", str(exc)))
    return diagnostics


def _pin_restart_helpers(plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("status") != "READY":
        return plan
    scheduled = _mapping(plan.get("scheduled_execution"))
    staged = scheduled.get("staged_files")
    if not isinstance(staged, list):
        return plan
    for filename, remote_name in (
        ("adapter_npt.py", "adapter-npt.py"),
        ("adapter.py", "adapter-legacy.py"),
    ):
        path = Path(__file__).resolve().with_name(filename)
        if not legacy._ordinary_file(path):
            return legacy._blocked(
                [_diagnostic("error", "ase_md.restart_helper", f"missing bundled {filename}")]
            )
        if not any(
            isinstance(item, dict) and item.get("remote_name") == remote_name for item in staged
        ):
            staged.append(legacy._staged_record(path, remote_name))
    fingerprints = plan.setdefault("input_fingerprints", {})
    if isinstance(fingerprints, dict):
        fingerprints["adapter_npt"] = legacy._sha256(
            Path(__file__).resolve().with_name("adapter_npt.py")
        )
    return plan


def _plan_restart(context: dict[str, Any]) -> dict[str, Any]:
    diagnostics = _validate_restart(context)
    if diagnostics:
        return legacy._blocked(diagnostics)
    plan = base.Adapter().plan(context)
    if plan.get("status") != "READY":
        return plan
    plan = _pin_restart_helpers(plan)
    if plan.get("status") != "READY":
        return plan
    parameters = _mapping(context["parameters"])
    identity = _mapping(plan.get("md_identity"))
    scheduled = _mapping(plan.get("scheduled_execution"))
    fetch_outputs = scheduled.get("fetch_outputs")
    staged = scheduled.get("staged_files")
    if not isinstance(fetch_outputs, list) or not isinstance(staged, list):
        return legacy._blocked(
            [_diagnostic("error", "ase_md.restart_contract", "base scheduled contract is incomplete")]
        )

    interval = parameters.get("checkpoint_interval")
    policy = str(parameters.get("restart_policy", RESTART_DISABLED))
    attempt = _attempt_number(context)
    start_step = 0
    restart_sha: str | None = None
    restart_attempt: int | None = None
    if interval is not None:
        fetch_outputs.append(
            {
                "remote_name": "md-checkpoint.json",
                "remote_path": "output/md-checkpoint.json",
                "local_name": "md-checkpoint.json",
                "required": True,
                "max_bytes": MAX_CHECKPOINT_BYTES,
                "role": "md-checkpoint",
            }
        )
        plan["failure_salvage"] = {
            "schema_version": 1,
            "fetch_remote_names": [
                "md-checkpoint.json",
                "trajectory.traj",
                "trajectory-index.json",
                "thermo.csv",
            ],
        }

    if policy == RESTART_AUTO and attempt > 1:
        try:
            previous = _previous_checkpoint(context, identity, attempt)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            return legacy._blocked([_diagnostic("error", "ase_md.restart_previous", str(exc))])
        if previous is None:
            return legacy._blocked(
                [_diagnostic("error", "ase_md.restart_previous", "previous checkpoint is missing")]
            )
        checkpoint_path, _checkpoint, start_step = previous
        restart_sha = legacy._sha256(checkpoint_path)
        restart_attempt = attempt - 1
        staged.append(legacy._staged_record(checkpoint_path, "restart/md-checkpoint.json"))
        fingerprints = plan.setdefault("input_fingerprints", {})
        if isinstance(fingerprints, dict):
            fingerprints["restart_checkpoint"] = restart_sha

    total = int(identity["steps"])
    trajectory_interval = int(identity["trajectory_interval"])
    thermo_interval = int(identity["thermo_interval"])
    identity.update(
        {
            "checkpoint_interval": interval,
            "restart_policy": policy,
            "segment_start_step": start_step,
            "restart_checkpoint_sha256": restart_sha,
            "restart_from_attempt": restart_attempt,
            "trajectory_steps": _segment_steps(start_step, total, trajectory_interval),
            "thermo_steps": _segment_steps(start_step, total, thermo_interval),
        }
    )
    plan["md_identity"] = identity
    summary = _mapping(plan.get("approval_summary"))
    summary.update(
        {
            "checkpoint_interval": interval,
            "restart_policy": policy,
            "segment_start_step": start_step,
            "remaining_steps": total - start_step,
            "restart_from_attempt": restart_attempt,
            "restart_checkpoint_sha256": restart_sha,
            "trajectory_frames_this_attempt": len(identity["trajectory_steps"]),
            "thermo_records_this_attempt": len(identity["thermo_steps"]),
            "failure_salvage": plan.get("failure_salvage"),
        }
    )
    plan["approval_summary"] = summary
    return plan


def _check_restart_metadata(context: dict[str, Any]) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    identity = _mapping(legacy._scheduled_plan(context).get("md_identity"))
    interval = identity.get("checkpoint_interval")
    try:
        result = legacy._read_bounded_json(attempt / "md-result.json")
    except Exception as exc:
        return [_diagnostic("error", "ase_md.restart_result", str(exc))]
    if result.get("segment_start_step") != identity.get("segment_start_step"):
        diagnostics.append(
            _diagnostic("error", "ase_md.segment_start", "md-result segment start differs")
        )
    restart = _mapping(result.get("restart"))
    resumed = int(identity.get("segment_start_step", 0)) > 0
    expected_restart = {
        "resumed": resumed,
        "checkpoint_sha256": identity.get("restart_checkpoint_sha256"),
        "checkpoint_start_step": identity.get("segment_start_step") if resumed else None,
        "checkpoint_interval": interval,
    }
    for key, value in expected_restart.items():
        if restart.get(key) != value:
            diagnostics.append(
                _diagnostic("error", f"ase_md.restart_{key}", f"restart field {key} differs")
            )
    if interval is not None:
        checkpoint_path = attempt / "md-checkpoint.json"
        if not legacy._ordinary_file(checkpoint_path, MAX_CHECKPOINT_BYTES):
            diagnostics.append(
                _diagnostic("error", "ase_md.checkpoint_missing", "final checkpoint is missing")
            )
        else:
            checkpoint_record = _mapping(result.get("checkpoint"))
            if (
                checkpoint_record.get("path") != "md-checkpoint.json"
                or checkpoint_record.get("sha256") != legacy._sha256(checkpoint_path)
                or checkpoint_record.get("size_bytes") != checkpoint_path.stat().st_size
                or checkpoint_record.get("completed_steps") != identity.get("steps")
            ):
                diagnostics.append(
                    _diagnostic(
                        "error", "ase_md.checkpoint_identity", "final checkpoint record differs"
                    )
                )
            try:
                checkpoint = _read_json(checkpoint_path, MAX_CHECKPOINT_BYTES)
                completed = _checkpoint_matches_identity(checkpoint, identity)
                if completed != identity.get("steps"):
                    diagnostics.append(
                        _diagnostic(
                            "error",
                            "ase_md.checkpoint_completed",
                            "final checkpoint is not at the requested total step",
                        )
                    )
            except ValueError as exc:
                # _checkpoint_matches_identity intentionally rejects completed==total
                # for an input restart. Validate final identity separately here.
                checkpoint = _read_json(checkpoint_path, MAX_CHECKPOINT_BYTES)
                expected_total = identity.get("steps")
                if checkpoint.get("completed_steps") != expected_total:
                    diagnostics.append(_diagnostic("error", "ase_md.checkpoint", str(exc)))
                else:
                    probe = dict(checkpoint)
                    probe["completed_steps"] = max(0, int(expected_total) - 1)
                    try:
                        _checkpoint_matches_identity(probe, identity)
                    except ValueError as identity_exc:
                        diagnostics.append(
                            _diagnostic("error", "ase_md.checkpoint", str(identity_exc))
                        )
    try:
        report = legacy._read_bounded_json(attempt / "cluster-run-report.json")
    except Exception as exc:
        diagnostics.append(_diagnostic("error", "ase_md.restart_cluster", str(exc)))
        report = {}
    if report.get("segment_start_step") != identity.get("segment_start_step"):
        diagnostics.append(
            _diagnostic("error", "ase_md.cluster_segment_start", "cluster segment start differs")
        )
    if _mapping(report.get("restart")) != _mapping(result.get("restart")):
        diagnostics.append(
            _diagnostic("error", "ase_md.cluster_restart", "cluster restart record differs")
        )
    return diagnostics


def _normalize_result_for_base_checker(context: dict[str, Any]) -> None:
    """No-op marker retained for clarity.

    The checkpoint is fetched as a scheduler artifact but deliberately not listed
    in md-result.artifacts, so the stable NVT/NPT artifact-set checkers remain valid.
    """


class Adapter:
    def validate(self, context: dict[str, Any]) -> list[dict[str, str]]:
        return _validate_restart(context)

    def plan(self, context: dict[str, Any]) -> dict[str, Any]:
        return _plan_restart(context)

    def prepare(self, context: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
        return base.Adapter().prepare(context, plan)

    def check(self, context: dict[str, Any]) -> dict[str, Any]:
        checked = base.Adapter().check(context)
        if checked.get("status") != "OK":
            return checked
        diagnostics = _check_restart_metadata(context)
        if diagnostics:
            return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
        return checked

    def collect(self, context: dict[str, Any]) -> dict[str, Any]:
        checked = self.check(context)
        if checked.get("status") != "OK":
            return checked
        collected = base.Adapter().collect(context)
        if collected.get("status") != "OK":
            return collected
        attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
        if legacy._ordinary_file(attempt / "md-checkpoint.json", MAX_CHECKPOINT_BYTES):
            artifacts = collected.setdefault("artifacts", [])
            if isinstance(artifacts, list):
                artifacts.append(
                    {
                        "path": str(attempt / "md-checkpoint.json"),
                        "role": "md-checkpoint",
                        "media_type": "application/json",
                    }
                )
        return collected

    def replay(self, context: dict[str, Any]) -> dict[str, Any]:
        return self.collect(context)
