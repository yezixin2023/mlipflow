"""ASE MD adapter facade adding isotropic MTK NPT to the stable NVT contract.

NVT remains delegated to adapter.py. NPT reuses the same staging, model-reference
and fetch lifecycle, but has its own pressure/barostat parameters and
scientific checker. The delegated adapter source is staged with the calculation.
"""
from __future__ import annotations

import csv
import importlib.util
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

PLUGIN_ID = "ase-md"
NVT = "nvt-langevin"
NPT = "npt-isotropic-mtk"
NPT_TCHAIN = 3
NPT_PCHAIN = 3
NPT_TLOOP = 1
NPT_PLOOP = 1


def _load_legacy():
    path = Path(__file__).resolve().with_name("adapter.py")
    spec = importlib.util.spec_from_file_location("_mlipflow_ase_md_nvt_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load bundled adapter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = _load_legacy()


def _diagnostic(level: str, code: str, message: str) -> dict[str, str]:
    return {"level": level, "code": code, "message": message}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _finite_positive(value: Any) -> bool:
    return _finite(value) and float(value) > 0.0


def _nvt_proxy(context: dict[str, Any]) -> dict[str, Any]:
    proxy = deepcopy(context)
    parameters = dict(_mapping(proxy.get("parameters")))
    parameters["ensemble"] = NVT
    parameters["friction_per_fs"] = 0.01
    proxy["parameters"] = parameters
    return proxy


def _validate_npt(context: dict[str, Any]) -> list[dict[str, str]]:
    diagnostics = list(legacy.Adapter().validate(_nvt_proxy(context)))
    parameters = _mapping(context.get("parameters"))
    if parameters.get("ensemble") != NPT:
        diagnostics.append(
            _diagnostic("error", "ase_md.ensemble", "ensemble must be npt-isotropic-mtk")
        )
    if not _finite(parameters.get("pressure_gpa")):
        diagnostics.append(
            _diagnostic("error", "ase_md.pressure", "pressure_gpa must be a finite number")
        )
    for key in ("thermostat_damping_fs", "barostat_damping_fs"):
        if not _finite_positive(parameters.get(key)):
            diagnostics.append(
                _diagnostic("error", f"ase_md.{key}", f"{key} must be finite and positive")
            )
    if parameters.get("friction_per_fs") is not None:
        diagnostics.append(
            _diagnostic(
                "error",
                "ase_md.npt_friction",
                "friction_per_fs is NVT-only and must be omitted for npt-isotropic-mtk",
            )
        )
    if parameters.get("fix_com") is not False:
        diagnostics.append(
            _diagnostic(
                "error",
                "ase_md.npt_constraints",
                "npt-isotropic-mtk requires fix_com=false because ASE IsotropicMTKNPT "
                "does not support constraints",
            )
        )
    return diagnostics


def _pin_legacy_helper(plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("status") != "READY":
        return plan
    scheduled = _mapping(plan.get("scheduled_execution"))
    staged = scheduled.get("staged_files")
    if not isinstance(staged, list):
        return plan
    path = Path(__file__).resolve().with_name("adapter.py")
    if not legacy._ordinary_file(path):
        return legacy._blocked(
            [_diagnostic("error", "ase_md.legacy_adapter", "bundled adapter.py is missing")]
        )
    if not any(item.get("remote_name") == "adapter-legacy.py" for item in staged if isinstance(item, dict)):
        staged.append(legacy._staged_record(path, "adapter-legacy.py"))
    input_paths = plan.setdefault("input_paths", {})
    if isinstance(input_paths, dict):
        input_paths["legacy_adapter"] = str(path)
    return plan


def _plan_npt(context: dict[str, Any]) -> dict[str, Any]:
    diagnostics = _validate_npt(context)
    if diagnostics:
        return legacy._blocked(diagnostics)
    base = legacy._plan(_nvt_proxy(context))
    if base.get("status") != "READY":
        return base
    parameters = _mapping(context["parameters"])
    settings = _mapping(base.get("md_parameters"))
    settings["ensemble"] = NPT
    settings.pop("friction_per_fs", None)
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
    summary = _mapping(base.get("approval_summary"))
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
    return _pin_legacy_helper(base)


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
    if not legacy._ordinary_file(path):
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
                if index >= legacy.MAX_RECORDS:
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
    settings = _mapping(legacy._scheduled_plan(context).get("md_parameters"))
    if settings.get("ensemble") != NPT:
        return [
            _diagnostic("error", "ase_md.plan_parameters", "plan is not isotropic MTK NPT")
        ], None
    try:
        result = legacy._read_json(attempt / "md-result.json")
    except Exception as exc:
        return [
            _diagnostic("error", "ase_md.result", f"md-result.json is unreadable: {exc}")
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
                _diagnostic(
                    "error",
                    f"ase_md.result_{key}",
                    f"md-result.json field {key} differs from the approved NPT plan",
                )
            )
    diagnostics.extend(legacy._check_structure_summary(result, settings))
    initial_pressure = result.get("initial_pressure_GPa")
    if not _finite(initial_pressure):
        diagnostics.append(
            _diagnostic(
                "error",
                "ase_md.initial_pressure",
                "md-result must record the finite pressure from the pre-run stress probe",
            )
        )
    model = _mapping(result.get("model"))
    if (
        model.get("id") != settings.get("model_id")
        or model.get("path") != settings.get("model_path")
    ):
        diagnostics.append(
            _diagnostic("error", "ase_md.result_model", "md-result model record differs")
        )
    if not legacy._plain_string(result.get("calculator_version")) or not legacy._plain_string(
        result.get("ase_version")
    ):
        diagnostics.append(
            _diagnostic("error", "ase_md.versions", "md-result must record calculator and ASE versions")
        )

    try:
        artifacts = legacy._artifact_records(result.get("artifacts"))
    except ValueError as exc:
        diagnostics.append(_diagnostic("error", "ase_md.artifacts", str(exc)))
        artifacts = {}
    expected_artifacts = {
        "trajectory": "trajectory.traj",
        "trajectory-index": "trajectory-index.json",
        "thermo": "thermo.csv",
        "final-structure": "final.extxyz",
    }
    if set(artifacts) != set(expected_artifacts):
        diagnostics.append(
            _diagnostic("error", "ase_md.artifact_set", "md-result artifact set is incomplete")
        )
    for role, name in expected_artifacts.items():
        if role in artifacts:
            error = legacy._check_artifact(attempt / name, artifacts[role])
            if error:
                diagnostics.append(_diagnostic("error", f"ase_md.artifact_{role}", error))

    index_error = legacy._check_trajectory_index(attempt / "trajectory-index.json", settings)
    if index_error:
        diagnostics.append(_diagnostic("error", "ase_md.trajectory_index", index_error))
    thermo_error, final_thermo = _check_npt_thermo(attempt / "thermo.csv", settings)
    if thermo_error:
        diagnostics.append(_diagnostic("error", "ase_md.thermo", thermo_error))

    try:
        report = legacy._read_json(attempt / "cluster-run-report.json")
    except Exception as exc:
        diagnostics.append(
            _diagnostic(
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
            _diagnostic(
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
                _diagnostic("error", f"ase_md.cluster_{key}", f"cluster report {key} differs")
            )
    report_model = _mapping(report.get("model"))
    if (
        report_model.get("id") != settings.get("model_id")
        or report_model.get("path") != settings.get("model_path")
    ):
        diagnostics.append(
            _diagnostic("error", "ase_md.cluster_model", "cluster report model record differs")
        )
    if report.get("structure_path") != settings.get("structure_path"):
        diagnostics.append(
            _diagnostic("error", "ase_md.cluster_structure", "cluster report structure differs")
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
                    _diagnostic(
                        "error",
                        f"ase_md.cluster_{key}",
                        f"cluster report {key} differs",
                    )
                )
    if diagnostics:
        return diagnostics, None
    return diagnostics, {"result": result, "final_thermo": final_thermo or {}}


def _check_ensemble(context: dict[str, Any]) -> str | None:
    plan_parameters = _mapping(legacy._scheduled_plan(context).get("md_parameters"))
    if plan_parameters:
        return plan_parameters.get("ensemble")
    return _mapping(context.get("parameters")).get("ensemble")


class Adapter:
    def validate(self, context: dict[str, Any]) -> list[dict[str, str]]:
        if _mapping(context.get("parameters")).get("ensemble", NVT) == NPT:
            return _validate_npt(context)
        return legacy.Adapter().validate(context)

    def plan(self, context: dict[str, Any]) -> dict[str, Any]:
        if _mapping(context.get("parameters")).get("ensemble", NVT) == NPT:
            return _plan_npt(context)
        return _pin_legacy_helper(legacy.Adapter().plan(context))


    def check(self, context: dict[str, Any]) -> dict[str, Any]:
        if _check_ensemble(context) != NPT:
            return legacy.Adapter().check(context)
        diagnostics, analysis = _check_npt(context)
        if diagnostics or analysis is None:
            return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
        result = _mapping(analysis.get("result"))
        final = _mapping(analysis.get("final_thermo"))
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

    def collect(self, context: dict[str, Any]) -> dict[str, Any]:
        if _check_ensemble(context) != NPT:
            return legacy.Adapter().collect(context)
        checked = self.check(context)
        if checked.get("status") != "OK":
            return checked
        attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
        roles = (
            ("md-result.json", "md-result", "application/json"),
            ("trajectory.traj", "trajectory", "application/octet-stream"),
            ("trajectory-index.json", "trajectory-index", "application/json"),
            ("thermo.csv", "thermodynamics", "text/csv"),
            ("final.extxyz", "final-structure", "chemical/x-xyz"),
            ("cluster-run-report.json", "cluster-run-report", "application/json"),
        )
        artifacts = [
            {"path": str(attempt / name), "role": role, "media_type": media}
            for name, role, media in roles
            if legacy._ordinary_file(attempt / name)
        ]
        return {
            "plugin_id": PLUGIN_ID,
            "status": "OK",
            "diagnostics": [],
            "metrics": checked.get("metrics", {}),
            "artifacts": artifacts,
        }
