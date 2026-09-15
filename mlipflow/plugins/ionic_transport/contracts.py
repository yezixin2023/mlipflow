"""ionic transport: contracts."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import math
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

PLUGIN_ID = "ionic-transport"

_MODULE_PATH = Path(__file__).resolve()

_REQUIRED_RESULT_NAMES = (
    "diffusion_results_by_temperature.csv",
    "arrhenius_summary.json",
    "postprocess_failures.json",
    "analysis_manifest.json",
)

_OPTIONAL_RESULT_NAMES = ("msd_by_temperature.html", "arrhenius_fit.html")

_RDF_CURVES_NAME = "rdf_curves.csv"

_AIMD_MLIP_COMPARISON_NAME = "aimd_mlip_comparison.json"

_INTEGRATION_MANIFEST_NAME = "md-integration-manifest.json"

_SMOKE_OPERATION = "md-smoke-and-analyze"

_ANALYZE_OPERATION = "analyze-existing"

_SMOKE_CALCULATORS = {"mace", "chgnet", "m3gnet", "matgl", "emt", "lj"}

_SMOKE_DEVICES = {"cpu", "cuda", "mps"}

_SMOKE_DTYPES = {"float32", "float64"}

_SMOKE_MAX_TEMPERATURES = 4

_SMOKE_MAX_EQUILIBRATION_PS = 0.1

_SMOKE_MAX_PRODUCTION_PS = 0.2

_SMOKE_MAX_TOTAL_STEPS = 5000

_RESULT_COLUMNS = {
    "temperature_K",
    "diffusivity_cm2_s",
    "conductivity_NE_mS_cm",
    "diffusion_method",
    "conductivity_method",
    "arrhenius_method",
    "analysis_start_ps",
    "analysis_end_ps",
    "haven_ratio",
    "msd_curve_csv",
    "msd_fit_html",
}


def _bundled_analysis_script() -> Path:
    return _MODULE_PATH.with_name("ionic_conductivity.py")


_ANALYZE_PARAMETERS = {
    "operation",
    "output_subdir",
    "source",
    "specie",
    "seed",
    "fit_start_ps",
    "fit_end_ps",
    "trajectory_start_ps",
    "trajectory_end_ps",
    "trajectory_msd_engine",
    "diffusion_analyzer_smoothed",
    "diffusion_analyzer_min_obs",
    "diffusion_analyzer_avg_nsteps",
    "diffusion_analyzer_step_skip",
    "min_msd_fit_points",
    "msd_smooth_window_points",
    "msd_smooth_window_ps",
    "fit_smoothed_msd",
    "ase_frame_step_fs",
    "lammps_timestep_ps",
    "lammps_data_name",
    "vasp_file_name",
    "vasp_step_fs",
    "temperature_k",
    "aimd_temperature_k",
    "mobile_type",
    "msd_file_name",
    "msd_time_column",
    "msd_column",
    "msd_time_unit",
    "msd_step_ps",
    "msd_unit",
    "msd_temperature_k",
    "n_mobile_ions",
    "volume_a3",
    "fit_temperatures",
    "exclude_temperatures",
    "fit_scope",
    "target_temperature_k",
    "piecewise",
    "min_segment_points",
    "piecewise_slope_change",
    "piecewise_bic_delta",
    "rdf_pair",
    "rdf_r_min_angstrom",
    "rdf_r_max_angstrom",
    "rdf_bins",
    "rdf_temperature_k",
    "comparison_model",
    "comparison_scenario",
    "comparison_split",
    "allow_partial_results",
}

_SMOKE_PARAMETERS = {
    "operation",
    "output_subdir",
    "md_output_subdir",
    "integration_manifest",
    "calculator",
    "temperatures_k",
    "seed",
    "timestep_fs",
    "npt_time_ps",
    "nvt_equil_time_ps",
    "production_time_ps",
    "trajectory_interval_steps",
    "device",
    "default_dtype",
    "input_format",
    "input_index",
    "source",
    "specie",
    "trajectory_start_ps",
    "trajectory_end_ps",
    "trajectory_msd_engine",
    "diffusion_analyzer_smoothed",
    "diffusion_analyzer_min_obs",
    "diffusion_analyzer_avg_nsteps",
    "diffusion_analyzer_step_skip",
    "min_msd_fit_points",
    "msd_smooth_window_points",
    "msd_smooth_window_ps",
    "fit_smoothed_msd",
    "ase_frame_step_fs",
    "fit_temperatures",
    "exclude_temperatures",
    "fit_scope",
    "target_temperature_k",
    "piecewise",
    "min_segment_points",
    "piecewise_slope_change",
    "piecewise_bic_delta",
    "allow_partial_results",
}


def _diagnostic(level: str, code: str, message: str) -> Dict[str, str]:
    return {"level": level, "code": code, "message": message}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _resolve(value: Any, base: Path) -> Optional[Path]:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_subdirectory(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    path = Path(value)
    return not path.is_absolute() and path != Path(".") and ".." not in path.parts


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _positive_number(value: Any) -> bool:
    return _finite_number(value) and float(value) > 0


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _optional_positive(
    parameters: Mapping[str, Any], name: str, diagnostics: List[Dict[str, str]]
) -> None:
    value = parameters.get(name)
    if value is not None and not _positive_number(value):
        diagnostics.append(
            _diagnostic(
                "ERROR", "parameter.%s" % name, "%s must be a finite positive number" % name
            )
        )


def _validate_interval(
    parameters: Mapping[str, Any],
    start_name: str,
    end_name: str,
    diagnostics: List[Dict[str, str]],
    required: bool,
) -> None:
    start = parameters.get(start_name)
    end = parameters.get(end_name)
    if required and (start is None or end is None):
        diagnostics.append(
            _diagnostic(
                "ERROR",
                "parameter.%s_%s" % (start_name, end_name),
                "%s and %s must both be explicit" % (start_name, end_name),
            )
        )
        return
    if (start is None) != (end is None):
        diagnostics.append(
            _diagnostic(
                "ERROR",
                "parameter.%s_%s" % (start_name, end_name),
                "%s and %s must be provided together" % (start_name, end_name),
            )
        )
        return
    if start is None:
        return
    if (
        not _finite_number(start)
        or not _finite_number(end)
        or float(start) < 0
        or float(start) >= float(end)
    ):
        diagnostics.append(
            _diagnostic(
                "ERROR",
                "parameter.%s_%s" % (start_name, end_name),
                "%s must be non-negative and smaller than %s" % (start_name, end_name),
            )
        )


def _has_errors(diagnostics: List[Dict[str, str]]) -> bool:
    return any(item["level"] == "ERROR" for item in diagnostics)


def _numbers_close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1.0e-10, abs_tol=1.0e-14)


def _verify_file_record(
    record: Any,
    *,
    role: str,
    confined_to: Optional[Path] = None,
) -> Tuple[Optional[Path], List[Dict[str, str]]]:
    diagnostics: List[Dict[str, str]] = []
    if not isinstance(record, Mapping):
        return None, [_diagnostic("ERROR", "result.manifest_record", f"{role} is not an object")]
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return None, [_diagnostic("ERROR", "result.manifest_path", f"{role} has no path")]
    path = Path(raw_path).expanduser().resolve()
    if confined_to is not None and not _is_within(path, confined_to):
        diagnostics.append(
            _diagnostic("ERROR", "result.manifest_escape", f"{role} escapes the output directory")
        )
    if not path.is_file():
        diagnostics.append(
            _diagnostic("ERROR", "result.manifest_missing", f"{role} is missing: {path}")
        )
        return path, diagnostics
    return path, diagnostics


def _blocked(diagnostics: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "plugin_id": PLUGIN_ID,
        "status": "BLOCKED",
        "executable": False,
        "diagnostics": diagnostics,
    }


def _stringify(value: Any) -> str:
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)


def _operation_name(parameters: Mapping[str, Any]) -> str:
    raw = parameters.get("operation", _ANALYZE_OPERATION)
    if raw == "analyze-existing-transport":
        return _ANALYZE_OPERATION
    return str(raw)


def _installed_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _formal_runtime_probe() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        from pymatgen.analysis.diffusion.analyzer import (
            DiffusionAnalyzer,
            fit_arrhenius,
            get_conversion_factor,
            get_diffusivity_from_msd,
        )

        _ = (
            DiffusionAnalyzer,
            fit_arrhenius,
            get_conversion_factor,
            get_diffusivity_from_msd,
        )
    except (ImportError, ModuleNotFoundError):
        return None, "Formal ionic transport requires pymatgen-analysis-diffusion"
    return (
        {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "mlipflow_version": _installed_version("mlipflow"),
            "pymatgen_version": _installed_version("pymatgen"),
            "pymatgen_analysis_diffusion_version": _installed_version(
                "pymatgen-analysis-diffusion"
            ),
            "numpy_version": _installed_version("numpy"),
            "ase_version": _installed_version("ase"),
        },
        None,
    )


def _formal_dependency_diagnostics() -> List[Dict[str, str]]:
    _, error = _formal_runtime_probe()
    if error is not None:
        return [
            _diagnostic(
                "ERROR",
                "dependency.pymatgen_analysis_diffusion",
                f"{error}. Install with: pip install 'mlipflow[transport]'",
            )
        ]
    return []


def _is_current_python(python_executable: str) -> bool:
    candidate = Path(python_executable).expanduser()
    return candidate.is_absolute() and candidate.resolve() == Path(sys.executable).resolve()


def _analysis_arguments(
    parameters: Mapping[str, Any], *, ase_frame_step_fs: Optional[float] = None
) -> List[str]:
    """Build only the reviewed ionic_conductivity.py analysis flags."""

    always = (
        ("specie", "--specie"),
        ("source", "--source"),
        ("trajectory_msd_engine", "--trajectory-msd-engine"),
        ("diffusion_analyzer_smoothed", "--diffusion-analyzer-smoothed"),
        ("diffusion_analyzer_min_obs", "--diffusion-analyzer-min-obs"),
        ("diffusion_analyzer_avg_nsteps", "--diffusion-analyzer-avg-nsteps"),
        ("diffusion_analyzer_step_skip", "--diffusion-analyzer-step-skip"),
        ("min_msd_fit_points", "--min-msd-fit-points"),
        ("fit_scope", "--fit-scope"),
        ("target_temperature_k", "--target-temperature-K"),
        ("piecewise", "--piecewise"),
        ("min_segment_points", "--min-segment-points"),
        ("piecewise_slope_change", "--piecewise-slope-change"),
        ("piecewise_bic_delta", "--piecewise-bic-delta"),
    )
    optional = (
        ("fit_start_ps", "--fit-start-ps"),
        ("fit_end_ps", "--fit-end-ps"),
        ("trajectory_start_ps", "--trajectory-start-ps"),
        ("trajectory_end_ps", "--trajectory-end-ps"),
        ("msd_smooth_window_points", "--msd-smooth-window-points"),
        ("msd_smooth_window_ps", "--msd-smooth-window-ps"),
        ("lammps_timestep_ps", "--lammps-timestep-ps"),
        ("lammps_data_name", "--lammps-data-name"),
        ("vasp_file_name", "--vasp-file-name"),
        ("vasp_step_fs", "--vasp-step-fs"),
        ("temperature_k", "--temperature-K"),
        ("aimd_temperature_k", "--aimd-temperature-K"),
        ("mobile_type", "--mobile-type"),
        ("msd_file_name", "--msd-file-name"),
        ("msd_time_column", "--msd-time-column"),
        ("msd_column", "--msd-column"),
        ("msd_step_ps", "--msd-step-ps"),
        ("msd_temperature_k", "--msd-temperature-K"),
        ("n_mobile_ions", "--n-mobile-ions"),
        ("volume_a3", "--volume-A3"),
        ("fit_temperatures", "--fit-temperatures"),
        ("exclude_temperatures", "--exclude-temperatures"),
        ("rdf_r_min_angstrom", "--rdf-r-min-angstrom"),
        ("rdf_r_max_angstrom", "--rdf-r-max-angstrom"),
        ("rdf_bins", "--rdf-bins"),
        ("rdf_temperature_k", "--rdf-temperature-K"),
        ("comparison_model", "--comparison-model"),
        ("comparison_scenario", "--comparison-scenario"),
        ("comparison_split", "--comparison-split"),
    )
    argv: List[str] = []
    for name, flag in always:
        if parameters.get(name) is not None:
            argv.extend([flag, _stringify(parameters[name])])
    for name in ("msd_time_unit", "msd_unit"):
        if parameters.get(name) is not None:
            argv.extend(
                [
                    "--msd-time-unit" if name == "msd_time_unit" else "--msd-unit",
                    str(parameters[name]),
                ]
            )
    for name, flag in optional:
        if parameters.get(name) is not None:
            argv.extend([flag, _stringify(parameters[name])])
    effective_frame_step = (
        ase_frame_step_fs if ase_frame_step_fs is not None else parameters.get("ase_frame_step_fs")
    )
    if effective_frame_step is not None:
        argv.extend(["--ase-frame-step-fs", _stringify(effective_frame_step)])
    if parameters.get("fit_smoothed_msd"):
        argv.append("--fit-smoothed-msd")
    if parameters.get("rdf_pair") is not None:
        argv.extend(["--rdf-pair", *[str(value) for value in parameters["rdf_pair"]]])
    return argv
