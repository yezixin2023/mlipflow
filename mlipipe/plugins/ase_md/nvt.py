"""NVT Langevin validation, planning, and completion checks."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from . import common

NVT = "nvt-langevin"


def validate(context: dict[str, Any]) -> list[dict[str, str]]:
    diagnostics = common.validate(context)
    parameters = common._mapping(common._mapping(context).get("parameters"))
    if parameters.get("ensemble", NVT) != NVT:
        diagnostics.append(common._diagnostic("error", "ase_md.ensemble", "ensemble must be nvt-langevin"))
    if not common._finite_positive(parameters.get("friction_per_fs")):
        diagnostics.append(common._diagnostic("error", "ase_md.friction", "friction_per_fs must be finite and positive"))
    return diagnostics


def plan(context: dict[str, Any]) -> dict[str, Any]:
    diagnostics = validate(context)
    if diagnostics:
        return common._blocked(diagnostics)
    result = common.build_plan(context, NVT)
    if result["status"] == "READY":
        result["md_parameters"]["friction_per_fs"] = float(context["parameters"]["friction_per_fs"])
    return result



def _check_thermo(path: Path, settings: dict[str, Any]) -> tuple[str | None, dict[str, float] | None]:
    if not common._ordinary_file(path):
        return "thermo.csv is missing or empty", None
    expected_header = ["step", "time_fs", "temperature_K", "potential_energy_eV", "kinetic_energy_eV", "total_energy_eV", "volume_A3"]
    expected_steps = settings.get("thermo_steps")
    if not isinstance(expected_steps, list):
        return "approved plan lacks thermo step schedule", None
    seen: list[int] = []
    last: dict[str, float] | None = None
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != expected_header:
                return "thermo.csv header differs from the contract", None
            for index, row in enumerate(reader):
                if index >= common.MAX_RECORDS:
                    return "thermo.csv exceeds the bounded record count", None
                try:
                    step = int(row["step"])
                    numeric = {key: float(row[key]) for key in expected_header[1:]}
                except (TypeError, ValueError, KeyError) as exc:
                    return f"thermo.csv contains an invalid numeric row: {exc}", None
                if any(not math.isfinite(value) for value in numeric.values()):
                    return "thermo.csv contains a non-finite value", None
                if not math.isclose(numeric["time_fs"], step * float(settings["timestep_fs"]), rel_tol=1e-12, abs_tol=1e-9):
                    return "thermo.csv time_fs is inconsistent with timestep_fs", None
                seen.append(step)
                last = numeric
    except (OSError, UnicodeError) as exc:
        return f"thermo.csv is unreadable: {exc}", None
    if seen != expected_steps:
        return "thermo.csv step schedule differs from the approved plan", None
    return None, last


def _check(context: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    diagnostics: list[dict[str, str]] = []
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    settings = common._mapping(common._scheduled_plan(context).get("md_parameters"))
    if not settings:
        return [common._diagnostic("error", "ase_md.plan_parameters", "plan lacks md_parameters")], None
    try:
        result = common._read_json(attempt / "md-result.json")
    except Exception as exc:
        return [common._diagnostic("error", "ase_md.result", f"md-result.json is unreadable: {exc}")], None
    expected_pairs = {
        "schema_version": 1,
        "plugin_id": common.PLUGIN_ID,
        "status": "OK",
        "calculator": settings.get("calculator"),
        "ensemble": "nvt-langevin",
        "structure_path": settings.get("structure_path"),
        "temperature_K": settings.get("temperature_k"),
        "timestep_fs": settings.get("timestep_fs"),
        "steps_requested": settings.get("steps"),
        "steps_completed": settings.get("steps"),
        "trajectory_interval": settings.get("trajectory_interval"),
        "thermo_interval": settings.get("thermo_interval"),
        "trajectory_frames": len(settings.get("trajectory_steps", [])),
        "thermo_records": len(settings.get("thermo_steps", [])),
        "seed": settings.get("seed"),
        "device": settings.get("device"),
        "default_dtype": settings.get("default_dtype"),
        "friction_per_fs": settings.get("friction_per_fs"),
        "fix_com": settings.get("fix_com"),
    }
    for key, expected in expected_pairs.items():
        if result.get(key) != expected:
            diagnostics.append(common._diagnostic("error", f"ase_md.result_{key}", f"md-result.json field {key} differs from the approved plan"))
    diagnostics.extend(common._check_structure_summary(result, settings))
    model = common._mapping(result.get("model"))
    if model.get("id") != settings.get("model_id") or model.get("path") != settings.get("model_path"):
        diagnostics.append(common._diagnostic("error", "ase_md.result_model", "md-result model record differs from the plan"))
    if not common._plain_string(result.get("calculator_version")) or not common._plain_string(result.get("ase_version")):
        diagnostics.append(common._diagnostic("error", "ase_md.versions", "md-result must record calculator and ASE versions"))
    try:
        artifacts = common._artifact_records(result.get("artifacts"))
    except ValueError as exc:
        diagnostics.append(common._diagnostic("error", "ase_md.artifacts", str(exc)))
        artifacts = {}
    expected_artifacts = {
        "trajectory": "trajectory.traj",
        "trajectory-index": "trajectory-index.json",
        "thermo": "thermo.csv",
        "final-structure": "final.extxyz",
    }
    if set(artifacts) != set(expected_artifacts):
        diagnostics.append(common._diagnostic("error", "ase_md.artifact_set", "md-result artifact set is incomplete or unexpected"))
    for role, name in expected_artifacts.items():
        if role in artifacts:
            error = common._check_artifact(attempt / name, artifacts[role])
            if error:
                diagnostics.append(common._diagnostic("error", f"ase_md.artifact_{role}", error))
    index_error = common._check_trajectory_index(attempt / "trajectory-index.json", settings)
    if index_error:
        diagnostics.append(common._diagnostic("error", "ase_md.trajectory_index", index_error))
    thermo_error, final_thermo = _check_thermo(attempt / "thermo.csv", settings)
    if thermo_error:
        diagnostics.append(common._diagnostic("error", "ase_md.thermo", thermo_error))
    try:
        report = common._read_json(attempt / "cluster-run-report.json")
    except Exception as exc:
        diagnostics.append(common._diagnostic("error", "ase_md.cluster_report", f"cluster-run-report.json is unreadable: {exc}"))
        report = {}
    if report.get("status") != "OK" or report.get("calculator") != settings.get("calculator") or report.get("ensemble") != "nvt-langevin":
        diagnostics.append(common._diagnostic("error", "ase_md.cluster_run", "cluster report does not describe the planned successful ASE MD run"))
    report_model = common._mapping(report.get("model"))
    if report_model.get("id") != settings.get("model_id") or report_model.get("path") != settings.get("model_path"):
        diagnostics.append(common._diagnostic("error", "ase_md.cluster_model", "cluster report model record differs from the plan"))
    if report.get("structure_path") != settings.get("structure_path"):
        diagnostics.append(common._diagnostic("error", "ase_md.cluster_structure", "cluster report structure path differs from the plan"))
    if "supercell_repeat" in settings:
        for key in (
            "supercell_repeat",
            "source_atom_count",
            "atom_count",
            "initial_cell_lengths_A",
            "minimum_initial_cell_length_A",
            "observed_stability",
        ):
            if report.get(key) != result.get(key):
                diagnostics.append(
                    common._diagnostic(
                        "error",
                        f"ase_md.cluster_{key}",
                        f"cluster report {key} differs",
                    )
                )
    if diagnostics:
        return diagnostics, None
    return diagnostics, {"result": result, "final_thermo": final_thermo or {}}


def check(context: dict[str, Any]) -> dict[str, Any]:
    diagnostics, analysis = _check(context)
    if diagnostics or analysis is None:
        return {"plugin_id": common.PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
    final = common._mapping(analysis.get("final_thermo"))
    return {
        "plugin_id": common.PLUGIN_ID,
        "status": "OK",
        "diagnostics": [],
        "metrics": {
            "steps_completed": float(common._mapping(analysis["result"]).get("steps_completed", 0)),
            "final_temperature_K": float(final.get("temperature_K", 0.0)),
            "final_total_energy_eV": float(final.get("total_energy_eV", 0.0)),
        },
        "artifacts": [],
    }
