"""ase md: restart."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import npt, nvt

PLUGIN_ID = "ase-md"

RESTART_DISABLED = "disabled"

RESTART_AUTO = "auto-from-previous-attempt"

RESTART_POLICIES = {RESTART_DISABLED, RESTART_AUTO}

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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing JSON file: {path}")
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


def _checkpoint_matches_parameters(checkpoint: dict[str, Any], settings: dict[str, Any]) -> int:
    expected = {
        "schema_version": 1,
        "plugin_id": PLUGIN_ID,
        "checkpoint_state_version": "ase-md-checkpoint-v1",
        "calculator": settings.get("calculator"),
        "ensemble": settings.get("ensemble"),
        "structure_path": settings.get("structure_path"),
        "temperature_K": settings.get("temperature_k"),
        "timestep_fs": settings.get("timestep_fs"),
        "steps_requested": settings.get("steps"),
        "seed": settings.get("seed"),
        "device": settings.get("device"),
        "default_dtype": settings.get("default_dtype"),
        "fix_com": settings.get("fix_com"),
    }
    if "supercell_repeat" in settings:
        expected.update(
            {
                "supercell_repeat": settings.get("supercell_repeat"),
                "minimum_initial_cell_length_A": settings.get(
                    "minimum_initial_cell_length_angstrom"
                ),
            }
        )
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"previous checkpoint parameter mismatch for {key}")
    model = nvt._mapping(checkpoint.get("model"))
    if model.get("id") != settings.get("model_id") or model.get("path") != settings.get(
        "model_path"
    ):
        raise ValueError("previous checkpoint model record differs from the plan")
    if settings.get("ensemble") == npt.NVT:
        if checkpoint.get("friction_per_fs") != settings.get("friction_per_fs"):
            raise ValueError("previous checkpoint Langevin friction differs")
        if checkpoint.get("rng_algorithm") != "PCG64":
            raise ValueError("previous checkpoint RNG algorithm is not PCG64")
    else:
        for key, setting_key in (
            ("pressure_GPa", "pressure_gpa"),
            ("thermostat_damping_fs", "thermostat_damping_fs"),
            ("barostat_damping_fs", "barostat_damping_fs"),
            ("thermostat_chain_length", "thermostat_chain_length"),
            ("barostat_chain_length", "barostat_chain_length"),
            ("thermostat_substeps", "thermostat_substeps"),
            ("barostat_substeps", "barostat_substeps"),
        ):
            if checkpoint.get(key) != settings.get(setting_key):
                raise ValueError(f"previous checkpoint NPT parameter mismatch for {key}")
    completed = checkpoint.get("completed_steps")
    total = settings.get("steps")
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
    context: dict[str, Any], settings: dict[str, Any], attempt: int
) -> tuple[Path, dict[str, Any], int] | None:
    if attempt == 1:
        return None
    previous = attempt - 1
    manifest_path = _final_manifest_path(context, previous)
    manifest = _read_json(manifest_path)
    if manifest.get("state") not in {"FAIL", "STOPPED"}:
        raise ValueError("previous attempt is not a failed/stopped scheduler attempt")
    job = nvt._mapping(manifest.get("job"))
    scheduler_state = str(job.get("scheduler_state", "")).upper()
    if scheduler_state not in RECOVERABLE_SCHEDULER_STATES:
        raise ValueError(
            "previous attempt did not end in a restart-eligible scheduler terminal state"
        )
    checkpoint_path = _checkpoint_path(context, previous)
    checkpoint = _read_json(checkpoint_path)
    completed = _checkpoint_matches_parameters(checkpoint, settings)
    return checkpoint_path, checkpoint, completed


def _validate_restart(context: dict[str, Any]) -> list[dict[str, str]]:
    diagnostics = list(npt.validate(context))
    parameters = nvt._mapping(context.get("parameters"))
    policy = parameters.get("restart_policy", RESTART_DISABLED)
    if policy not in RESTART_POLICIES:
        diagnostics.append(
            nvt._diagnostic(
                "error",
                "ase_md.restart_policy",
                "restart_policy must be disabled or auto-from-previous-attempt",
            )
        )
    interval = parameters.get("checkpoint_interval")
    if interval is not None:
        if not _positive_int(interval):
            diagnostics.append(
                nvt._diagnostic(
                    "error", "ase_md.checkpoint_interval", "checkpoint_interval must be positive"
                )
            )
        elif _positive_int(parameters.get("steps")) and interval > parameters["steps"]:
            diagnostics.append(
                nvt._diagnostic(
                    "error",
                    "ase_md.checkpoint_interval",
                    "checkpoint_interval cannot exceed total steps",
                )
            )
    if policy == RESTART_AUTO and interval is None:
        diagnostics.append(
            nvt._diagnostic(
                "error",
                "ase_md.restart_checkpoint_interval",
                "auto restart requires an explicit checkpoint_interval",
            )
        )
    try:
        _attempt_number(context)
    except ValueError as exc:
        diagnostics.append(nvt._diagnostic("error", "ase_md.attempt", str(exc)))
    return diagnostics


def _plan_restart(context: dict[str, Any]) -> dict[str, Any]:
    diagnostics = _validate_restart(context)
    if diagnostics:
        return nvt._blocked(diagnostics)
    plan = npt.plan(context)
    if plan.get("status") != "READY":
        return plan
    parameters = nvt._mapping(context["parameters"])
    settings = nvt._mapping(plan.get("md_parameters"))
    scheduled = nvt._mapping(plan.get("scheduled_execution"))
    fetch_outputs = scheduled.get("fetch_outputs")
    staged = scheduled.get("staged_files")
    if not isinstance(fetch_outputs, list) or not isinstance(staged, list):
        return nvt._blocked(
            [
                nvt._diagnostic(
                    "error", "ase_md.restart_contract", "base scheduled contract is incomplete"
                )
            ]
        )

    interval = parameters.get("checkpoint_interval")
    policy = str(parameters.get("restart_policy", RESTART_DISABLED))
    attempt = _attempt_number(context)
    start_step = 0
    restart_path: str | None = None
    restart_attempt: int | None = None
    if interval is not None:
        fetch_outputs.append(
            {
                "remote_name": "md-checkpoint.json",
                "remote_path": "output/md-checkpoint.json",
                "local_name": "md-checkpoint.json",
                "required": True,
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
            previous = _previous_checkpoint(context, settings, attempt)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            return nvt._blocked([nvt._diagnostic("error", "ase_md.restart_previous", str(exc))])
        if previous is None:
            return nvt._blocked(
                [nvt._diagnostic("error", "ase_md.restart_previous", "previous checkpoint is missing")]
            )
        checkpoint_path, _checkpoint, start_step = previous
        restart_path = "restart/md-checkpoint.json"
        restart_attempt = attempt - 1
        staged.append(nvt._staged_record(checkpoint_path, "restart/md-checkpoint.json"))
        input_paths = plan.setdefault("input_paths", {})
        if isinstance(input_paths, dict):
            input_paths["restart_checkpoint"] = str(checkpoint_path)

    total = int(settings["steps"])
    trajectory_interval = int(settings["trajectory_interval"])
    thermo_interval = int(settings["thermo_interval"])
    settings.update(
        {
            "checkpoint_interval": interval,
            "restart_policy": policy,
            "segment_start_step": start_step,
            "restart_checkpoint_path": restart_path,
            "restart_from_attempt": restart_attempt,
            "trajectory_steps": _segment_steps(start_step, total, trajectory_interval),
            "thermo_steps": _segment_steps(start_step, total, thermo_interval),
        }
    )
    plan["md_parameters"] = settings
    summary = nvt._mapping(plan.get("approval_summary"))
    summary.update(
        {
            "checkpoint_interval": interval,
            "restart_policy": policy,
            "segment_start_step": start_step,
            "remaining_steps": total - start_step,
            "restart_from_attempt": restart_attempt,
            "restart_checkpoint_path": restart_path,
            "trajectory_frames_this_attempt": len(settings["trajectory_steps"]),
            "thermo_records_this_attempt": len(settings["thermo_steps"]),
            "failure_salvage": plan.get("failure_salvage"),
        }
    )
    plan["approval_summary"] = summary
    return plan


def _check_restart_metadata(context: dict[str, Any]) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    settings = nvt._mapping(nvt._scheduled_plan(context).get("md_parameters"))
    interval = settings.get("checkpoint_interval")
    try:
        result = nvt._read_json(attempt / "md-result.json")
    except Exception as exc:
        return [nvt._diagnostic("error", "ase_md.restart_result", str(exc))]
    if result.get("segment_start_step") != settings.get("segment_start_step"):
        diagnostics.append(
            nvt._diagnostic("error", "ase_md.segment_start", "md-result segment start differs")
        )
    restart = nvt._mapping(result.get("restart"))
    resumed = int(settings.get("segment_start_step", 0)) > 0
    expected_restart = {
        "resumed": resumed,
        "checkpoint_path": settings.get("restart_checkpoint_path"),
        "checkpoint_start_step": settings.get("segment_start_step") if resumed else None,
        "checkpoint_interval": interval,
    }
    for key, value in expected_restart.items():
        if restart.get(key) != value:
            diagnostics.append(
                nvt._diagnostic("error", f"ase_md.restart_{key}", f"restart field {key} differs")
            )
    if interval is not None:
        checkpoint_path = attempt / "md-checkpoint.json"
        if not nvt._ordinary_file(checkpoint_path):
            diagnostics.append(
                nvt._diagnostic("error", "ase_md.checkpoint_missing", "final checkpoint is missing")
            )
        else:
            checkpoint_record = nvt._mapping(result.get("checkpoint"))
            if checkpoint_record.get("path") != "md-checkpoint.json" or checkpoint_record.get(
                "completed_steps"
            ) != settings.get("steps"):
                diagnostics.append(
                    nvt._diagnostic(
                        "error", "ase_md.checkpoint_record", "final checkpoint record differs"
                    )
                )
            try:
                checkpoint = _read_json(checkpoint_path)
                completed = _checkpoint_matches_parameters(checkpoint, settings)
                if completed != settings.get("steps"):
                    diagnostics.append(
                        nvt._diagnostic(
                            "error",
                            "ase_md.checkpoint_completed",
                            "final checkpoint is not at the requested total step",
                        )
                    )
            except ValueError as exc:
                # Input restart validation intentionally rejects completed==total.
                # Validate the final checkpoint parameters separately here.
                checkpoint = _read_json(checkpoint_path)
                expected_total = settings.get("steps")
                if checkpoint.get("completed_steps") != expected_total:
                    diagnostics.append(nvt._diagnostic("error", "ase_md.checkpoint", str(exc)))
                else:
                    probe = dict(checkpoint)
                    probe["completed_steps"] = max(0, int(expected_total) - 1)
                    try:
                        _checkpoint_matches_parameters(probe, settings)
                    except ValueError as parameter_exc:
                        diagnostics.append(
                            nvt._diagnostic("error", "ase_md.checkpoint", str(parameter_exc))
                        )
    try:
        report = nvt._read_json(attempt / "cluster-run-report.json")
    except Exception as exc:
        diagnostics.append(nvt._diagnostic("error", "ase_md.restart_cluster", str(exc)))
        report = {}
    if report.get("segment_start_step") != settings.get("segment_start_step"):
        diagnostics.append(
            nvt._diagnostic("error", "ase_md.cluster_segment_start", "cluster segment start differs")
        )
    if nvt._mapping(report.get("restart")) != nvt._mapping(result.get("restart")):
        diagnostics.append(
            nvt._diagnostic("error", "ase_md.cluster_restart", "cluster restart record differs")
        )
    return diagnostics
