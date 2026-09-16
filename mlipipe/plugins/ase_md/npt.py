"""Isotropic MTK NPT validation, planning, and completion checks."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from . import common

PLUGIN_ID = "ase-md"
NPT = "npt-isotropic-mtk"
NPT_TCHAIN = 3
NPT_PCHAIN = 3
NPT_TLOOP = 1
NPT_PLOOP = 1


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _finite_positive(value: Any) -> bool:
    return _finite(value) and float(value) > 0.0


def validate(context: dict[str, Any]) -> list[dict[str, str]]:
    diagnostics = common.validate(context)
    parameters = common._mapping(common._mapping(context).get("parameters"))
    if parameters.get("ensemble") != NPT:
        diagnostics.append(
            common._diagnostic("error", "ase_md.ensemble", "ensemble must be npt-isotropic-mtk")
        )
    if not _finite(parameters.get("pressure_gpa")):
        diagnostics.append(
            common._diagnostic("error", "ase_md.pressure", "pressure_gpa must be a finite number")
        )
    for key in ("thermostat_damping_fs", "barostat_damping_fs"):
        if not _finite_positive(parameters.get(key)):
            diagnostics.append(
                common._diagnostic("error", f"ase_md.{key}", f"{key} must be finite and positive")
            )
    if parameters.get("friction_per_fs") is not None:
        diagnostics.append(
            common._diagnostic(
                "error",
                "ase_md.npt_friction",
                "friction_per_fs is NVT-only and must be omitted for npt-isotropic-mtk",
            )
        )
    if parameters.get("fix_com") is not False:
        diagnostics.append(
            common._diagnostic(
                "error",
                "ase_md.npt_constraints",
                "npt-isotropic-mtk requires fix_com=false because ASE IsotropicMTKNPT "
                "does not support constraints",
            )
        )
    return diagnostics


def plan(context: dict[str, Any]) -> dict[str, Any]:
    diagnostics = validate(context)
    if diagnostics:
        return common._blocked(diagnostics)
    base = common.build_plan(context, NPT)
    if base.get("status") != "READY":
        return base
    parameters = common._mapping(context["parameters"])
    settings = common._mapping(base.get("md_parameters"))
    settings.update(
        {
            "pressure_gpa": float(parameters["pressure_gpa"]),
            "thermostat_damping_fs": float(parameters["thermostat_damping_fs"]),
            "barostat_damping_fs": float(parameters["barostat_damping_fs"]),
            "fix_com": False,
            "stress_required": True,
            "thermostat_chain_length": NPT_TCHAIN,
            "barostat_chain_length": NPT_PCHAIN,
            "thermostat_substeps": NPT_TLOOP,
            "barostat_substeps": NPT_PLOOP,
        }
    )
    summary = common._mapping(base.get("approval_summary"))
    summary.update(
        {
            "ensemble": NPT,
            "pressure_GPa": settings["pressure_gpa"],
            "thermostat_damping_fs": settings["thermostat_damping_fs"],
            "barostat_damping_fs": settings["barostat_damping_fs"],
            "stress_required": True,
            "cell_mode": "isotropic-volume",
            "constraints_allowed": False,
            "mtk_chain_configuration": {
                "thermostat_chain_length": NPT_TCHAIN,
                "barostat_chain_length": NPT_PCHAIN,
                "thermostat_substeps": NPT_TLOOP,
                "barostat_substeps": NPT_PLOOP,
            },
        }
    )
    base["md_parameters"] = settings
    base["approval_summary"] = summary
    return base


def _npt_thermo_header() -> list[str]:
    return [
        "step",
        "time_fs",
        "temperature_K",
        "potential_energy_eV",
        "kinetic_energy_eV",
        "total_energy_eV",
        "volume_A3",
        "pressure_GPa",
        "cell_a_A",
        "cell_b_A",
        "cell_c_A",
    ]


def _check_npt_thermo(
    path: Path, settings: dict[str, Any]
) -> tuple[str | None, dict[str, float] | None]:
    if not common._ordinary_file(path):
        return "thermo.csv is missing or empty", None
    expected_steps = settings.get("thermo_steps")
    if not isinstance(expected_steps, list):
        return "approved plan lacks thermo step schedule", None
    header = _npt_thermo_header()
    seen: list[int] = []
    last: dict[str, float] | None = None
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != header:
                return "NPT thermo.csv header differs from the contract", None
            for index, row in enumerate(reader):
                if index >= common.MAX_RECORDS:
                    return "thermo.csv exceeds the bounded record count", None
                try:
                    step = int(row["step"])
                    numeric = {key: float(row[key]) for key in header[1:]}
                except (TypeError, ValueError, KeyError) as exc:
                    return f"thermo.csv contains an invalid numeric row: {exc}", None
                if any(not math.isfinite(value) for value in numeric.values()):
                    return "thermo.csv contains a non-finite value", None
                if not math.isclose(
                    numeric["time_fs"],
                    step * float(settings["timestep_fs"]),
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                ):
                    return "thermo.csv time_fs is inconsistent with timestep_fs", None
                if numeric["volume_A3"] <= 0 or any(
                    numeric[key] <= 0 for key in ("cell_a_A", "cell_b_A", "cell_c_A")
                ):
                    return "NPT thermo.csv contains a non-positive cell metric", None
                seen.append(step)
                last = numeric
    except (OSError, UnicodeError) as exc:
        return f"thermo.csv is unreadable: {exc}", None
    if seen != expected_steps:
        return "thermo.csv step schedule differs from the approved plan", None
    return None, last


def _check_npt(
    context: dict[str, Any]
) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    diagnostics: list[dict[str, str]] = []
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    settings = common._mapping(common._scheduled_plan(context).get("md_parameters"))
    if settings.get("ensemble") != NPT:
        return [
            common._diagnostic("error", "ase_md.plan_parameters", "plan is not isotropic MTK NPT")
        ], None
    try:
        result = common._read_json(attempt / "md-result.json")
    except Exception as exc:
        return [
            common._diagnostic("error", "ase_md.result", f"md-result.json is unreadable: {exc}")
        ], None

    expected_pairs = {
        "schema_version": 1,
        "plugin_id": PLUGIN_ID,
        "status": "OK",
        "calculator": settings.get("calculator"),
        "ensemble": NPT,
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
        "stochastic_scope": "velocity-initialization-only",
        "device": settings.get("device"),
        "default_dtype": settings.get("default_dtype"),
        "fix_com": False,
        "pressure_GPa": settings.get("pressure_gpa"),
        "thermostat_damping_fs": settings.get("thermostat_damping_fs"),
        "barostat_damping_fs": settings.get("barostat_damping_fs"),
        "thermostat_chain_length": NPT_TCHAIN,
        "barostat_chain_length": NPT_PCHAIN,
        "thermostat_substeps": NPT_TLOOP,
        "barostat_substeps": NPT_PLOOP,
    }
    for key, expected in expected_pairs.items():
        if result.get(key) != expected:
            diagnostics.append(
                common._diagnostic(
                    "error",
                    f"ase_md.result_{key}",
                    f"md-result.json field {key} differs from the approved NPT plan",
                )
            )
    diagnostics.extend(common._check_structure_summary(result, settings))
    initial_pressure = result.get("initial_pressure_GPa")
    if not _finite(initial_pressure):
        diagnostics.append(
            common._diagnostic(
                "error",
                "ase_md.initial_pressure",
                "md-result must record the finite pressure from the pre-run stress probe",
            )
        )
    model = common._mapping(result.get("model"))
    if (
        model.get("id") != settings.get("model_id")
        or model.get("path") != settings.get("model_path")
    ):
        diagnostics.append(
            common._diagnostic("error", "ase_md.result_model", "md-result model record differs")
        )
    if not common._plain_string(result.get("calculator_version")) or not common._plain_string(
        result.get("ase_version")
    ):
        diagnostics.append(
            common._diagnostic("error", "ase_md.versions", "md-result must record calculator and ASE versions")
        )

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
        diagnostics.append(
            common._diagnostic("error", "ase_md.artifact_set", "md-result artifact set is incomplete")
        )
    for role, name in expected_artifacts.items():
        if role in artifacts:
            error = common._check_artifact(attempt / name, artifacts[role])
            if error:
                diagnostics.append(common._diagnostic("error", f"ase_md.artifact_{role}", error))

    index_error = common._check_trajectory_index(attempt / "trajectory-index.json", settings)
    if index_error:
        diagnostics.append(common._diagnostic("error", "ase_md.trajectory_index", index_error))
    thermo_error, final_thermo = _check_npt_thermo(attempt / "thermo.csv", settings)
    if thermo_error:
        diagnostics.append(common._diagnostic("error", "ase_md.thermo", thermo_error))

    try:
        report = common._read_json(attempt / "cluster-run-report.json")
    except Exception as exc:
        diagnostics.append(
            common._diagnostic(
                "error", "ase_md.cluster_report", f"cluster-run-report.json is unreadable: {exc}"
            )
        )
        report = {}
    if (
        report.get("status") != "OK"
        or report.get("calculator") != settings.get("calculator")
        or report.get("ensemble") != NPT
        or report.get("steps_completed") != settings.get("steps")
    ):
        diagnostics.append(
            common._diagnostic(
                "error",
                "ase_md.cluster_run",
                "cluster report does not describe the planned completed NPT run",
            )
        )
    for key, setting_key in (
        ("pressure_GPa", "pressure_gpa"),
        ("thermostat_damping_fs", "thermostat_damping_fs"),
        ("barostat_damping_fs", "barostat_damping_fs"),
    ):
        if report.get(key) != settings.get(setting_key):
            diagnostics.append(
                common._diagnostic("error", f"ase_md.cluster_{key}", f"cluster report {key} differs")
            )
    report_model = common._mapping(report.get("model"))
    if (
        report_model.get("id") != settings.get("model_id")
        or report_model.get("path") != settings.get("model_path")
    ):
        diagnostics.append(
            common._diagnostic("error", "ase_md.cluster_model", "cluster report model record differs")
        )
    if report.get("structure_path") != settings.get("structure_path"):
        diagnostics.append(
            common._diagnostic("error", "ase_md.cluster_structure", "cluster report structure differs")
        )
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
    diagnostics, analysis = _check_npt(context)
    if diagnostics or analysis is None:
        return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
    result = common._mapping(analysis.get("result"))
    final = common._mapping(analysis.get("final_thermo"))
    return {
        "plugin_id": PLUGIN_ID,
        "status": "OK",
        "diagnostics": [],
        "metrics": {
            "steps_completed": float(result.get("steps_completed", 0)),
            "final_temperature_K": float(final.get("temperature_K", 0.0)),
            "final_total_energy_eV": float(final.get("total_energy_eV", 0.0)),
            "final_pressure_GPa": float(final.get("pressure_GPa", 0.0)),
            "final_volume_A3": float(final.get("volume_A3", 0.0)),
        },
        "artifacts": [],
    }
