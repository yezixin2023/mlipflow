"""Safe adapter for formal ionic transport and a bounded local MD smoke handoff.

Planning is side-effect free. The optional smoke delegates only the tiny MD stage
to a reviewed historical source, then uses the pinned pymatgen formal runner. Every
output stays in a fresh attempt and is never production or scientific-parity evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import platform
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mlipflow.science.transport import (
    linear_fit,
    nernst_einstein_conductivity,
)


PLUGIN_ID = "ionic-transport"
_MODULE_PATH = Path(str(globals().get("__file__", "adapter.py"))).resolve()
MANUSCRIPT_PARITY_VERSION = "1.0"
LEGACY_SCRIPT_CONVENTION = "legacy_script"
COMPOSITION_CORRECTED_CONVENTION = "composition_corrected"
EXACT_ARRHENIUS_CONVENTION = "exact_unrounded_ols_v1"
LEGACY_ARRHENIUS_CONVENTION = "legacy_get_sigma_v1"
_LEGACY_LI10_N_MOBILE_IONS = 7
_HISTORICAL_ELEMENTARY_CHARGE_C = 1.6e-19
_HISTORICAL_BOLTZMANN_J_K = 1.38e-23
_MAX_PARITY_SOURCE_BYTES = 64 * 1024 * 1024
_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_SAMPLING_PROFILES: Dict[str, Dict[str, Any]] = {
    "mace_li10_legacy_v1": {
        "start_line_index": 3,
        "line_modulus": 8,
        "line_offset": 2,
        "timestep_fs": 1.0,
        "description": (
            "Exact line selection used by the historical Li10 MACE get_MSD_from_* scripts: "
            "range(3, len(lines)) and (i - 2) % 8 == 0."
        ),
    },
    "deepmd_li10_legacy_v1": {
        "start_line_index": 990,
        "line_modulus": 12,
        "line_offset": 2,
        "timestep_fs": 1.0,
        "description": (
            "Exact line selection used by the historical Li10 DeepMD screening scripts: "
            "range(990, len(lines)) and (i - 2) % 12 == 0."
        ),
    },
}
_REQUIRED_RESULT_NAMES = (
    "diffusion_results_by_temperature.csv",
    "arrhenius_summary.json",
    "postprocess_failures.json",
    "analysis_manifest.json",
)
_OPTIONAL_RESULT_NAMES = ("msd_by_temperature.html", "arrhenius_fit.html")
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


def _optional_nonnegative(
    parameters: Mapping[str, Any], name: str, diagnostics: List[Dict[str, str]]
) -> None:
    value = parameters.get(name)
    if value is not None and (not _finite_number(value) or float(value) < 0):
        diagnostics.append(
            _diagnostic("ERROR", "parameter.%s" % name, "%s must be finite and non-negative" % name)
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
    if record.get("sha256") != _sha256_file(path):
        diagnostics.append(
            _diagnostic("ERROR", "result.manifest_sha256", f"{role} SHA-256 does not match")
        )
    if record.get("size_bytes") != path.stat().st_size:
        diagnostics.append(
            _diagnostic("ERROR", "result.manifest_size", f"{role} size does not match")
        )
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


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
    )
    argv: List[str] = []
    for name, flag in always:
        if parameters.get(name) is not None:
            argv.extend([flag, _stringify(parameters[name])])
    for name in ("msd_time_unit", "msd_unit"):
        if parameters.get(name) is not None:
            argv.extend(
                ["--msd-time-unit" if name == "msd_time_unit" else "--msd-unit", str(parameters[name])]
            )
    for name, flag in optional:
        if parameters.get(name) is not None:
            argv.extend([flag, _stringify(parameters[name])])
    effective_frame_step = (
        ase_frame_step_fs
        if ase_frame_step_fs is not None
        else parameters.get("ase_frame_step_fs")
    )
    if effective_frame_step is not None:
        argv.extend(["--ase-frame-step-fs", _stringify(effective_frame_step)])
    if parameters.get("fit_smoothed_msd"):
        argv.append("--fit-smoothed-msd")
    return argv


def _validate_md_smoke(
    *,
    project_root: Path,
    attempt_dir: Path,
    inputs: Mapping[str, Any],
    parameters: Mapping[str, Any],
    resources: Mapping[str, Any],
) -> List[Dict[str, str]]:
    diagnostics: List[Dict[str, str]] = []
    if "analysis_script" in inputs:
        diagnostics.append(
            _diagnostic(
                "ERROR",
                "input.analysis_script_unsupported",
                "analysis_script is packaged by MLIPFlow and cannot be overridden",
            )
        )
    unknown = sorted(set(parameters) - _SMOKE_PARAMETERS)
    if unknown:
        diagnostics.append(
            _diagnostic(
                "ERROR",
                "parameter.unknown",
                "unsupported md-smoke parameter(s): %s" % ", ".join(unknown),
            )
        )

    resolved_inputs: Dict[str, Path] = {}
    for name, label in (
        ("md_script", "historical ASE-MD source"),
        ("structure", "input structure"),
    ):
        path = _resolve(inputs.get(name), project_root)
        if path is None or not path.is_file():
            diagnostics.append(
                _diagnostic("ERROR", "path.%s" % name, "inputs.%s must name %s" % (name, label))
            )
        else:
            resolved_inputs[name] = path
            if name.endswith("script") and path.suffix != ".py":
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "path.%s_suffix" % name, "inputs.%s must be a Python file" % name
                    )
                )

    analysis_script = _bundled_analysis_script()
    if not analysis_script.is_file():
        diagnostics.append(
            _diagnostic(
                "ERROR",
                "path.bundled_analysis_missing",
                "the packaged ionic_conductivity.py runner is missing",
            )
        )

    calculator = parameters.get("calculator")
    if calculator not in _SMOKE_CALCULATORS:
        diagnostics.append(
            _diagnostic(
                "ERROR",
                "parameter.calculator",
                "calculator must be one of %s" % sorted(_SMOKE_CALCULATORS),
            )
        )
    raw_model = inputs.get("model", "default")
    if raw_model is None or str(raw_model).strip().lower() in {"", "none", "default"}:
        model = "default"
    elif isinstance(raw_model, (str, Path)):
        model_path = _resolve(raw_model, project_root)
        if model_path is None or not model_path.is_file():
            diagnostics.append(
                _diagnostic(
                    "ERROR", "path.model", "an explicit inputs.model must name a local model file"
                )
            )
        else:
            resolved_inputs["model"] = model_path
        model = str(raw_model)
    else:
        model = ""
        diagnostics.append(
            _diagnostic("ERROR", "path.model", "inputs.model must be 'default' or a file path")
        )
    if calculator in {"mace", "chgnet", "m3gnet", "matgl"} and model == "default":
        diagnostics.append(
            _diagnostic(
                "ERROR",
                "parameter.default_model_network",
                "%s requires an explicit local model; default loading may use packaged or network state"
                % calculator,
            )
        )
    if calculator in {"emt", "lj"} and model != "default":
        diagnostics.append(
            _diagnostic(
                "ERROR",
                "parameter.model_unused",
                "%s ignores model paths; use inputs.model='default'" % calculator,
            )
        )

    output_values = {
        "analysis": parameters.get("output_subdir", "ionic-transport-postprocess"),
        "md": parameters.get("md_output_subdir", "ionic-md-smoke"),
        "manifest": parameters.get("integration_manifest", _INTEGRATION_MANIFEST_NAME),
    }
    output_paths: Dict[str, Path] = {}
    for name, value in output_values.items():
        if not _safe_subdirectory(value):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "path.%s_output" % name,
                    "%s output must be a non-empty attempt-relative path without '..'" % name,
                )
            )
            continue
        path = (attempt_dir / str(value)).resolve()
        output_paths[name] = path
        if not _is_within(path, attempt_dir):
            diagnostics.append(
                _diagnostic("ERROR", "path.%s_escape" % name, "%s output escapes attempt_dir" % name)
            )
        elif path.exists():
            diagnostics.append(
                _diagnostic(
                    "ERROR", "path.%s_exists" % name, "%s output already exists; use a fresh attempt" % name
                )
            )
    if len(set(output_paths.values())) != len(output_paths):
        diagnostics.append(
            _diagnostic("ERROR", "path.smoke_output_collision", "MD, analysis, and manifest paths must differ")
        )
    output_items = list(output_paths.items())
    for index, (left_name, left_path) in enumerate(output_items):
        for right_name, right_path in output_items[index + 1 :]:
            if _is_within(left_path, right_path) or _is_within(right_path, left_path):
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "path.smoke_output_overlap",
                        "%s and %s outputs must not contain one another"
                        % (left_name, right_name),
                    )
                )
    for output in output_paths.values():
        for source_path in resolved_inputs.values():
            if _is_within(output, source_path) or _is_within(source_path, output):
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "path.input_output_overlap", "source/input and smoke output paths overlap"
                    )
                )
                break

    temperatures = parameters.get("temperatures_k")
    normalized_temperatures: List[float] = []
    if not isinstance(temperatures, list) or not (
        2 <= len(temperatures) <= _SMOKE_MAX_TEMPERATURES
    ):
        diagnostics.append(
            _diagnostic(
                "ERROR",
                "parameter.temperatures_k",
                "temperatures_k must contain 2-%d temperatures for an Arrhenius smoke"
                % _SMOKE_MAX_TEMPERATURES,
            )
        )
    else:
        for value in temperatures:
            if not _positive_number(value) or not float(value).is_integer():
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "parameter.temperatures_k",
                        "historical T<int> directories require positive integer-K temperatures",
                    )
                )
                break
            normalized_temperatures.append(float(value))
        if len(set(normalized_temperatures)) != len(normalized_temperatures):
            diagnostics.append(
                _diagnostic("ERROR", "parameter.temperature_duplicate", "temperatures_k must be unique")
            )

    seed = parameters.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        diagnostics.append(
            _diagnostic("ERROR", "parameter.seed", "seed must be an explicit non-negative integer")
        )
    timestep = parameters.get("timestep_fs")
    if not _positive_number(timestep) or float(timestep) > 2.0:
        diagnostics.append(
            _diagnostic(
                "ERROR", "parameter.timestep_fs", "timestep_fs must be positive and no greater than 2 fs"
            )
        )
    durations = (
        ("npt_time_ps", 0.0, _SMOKE_MAX_EQUILIBRATION_PS),
        ("nvt_equil_time_ps", 0.0, _SMOKE_MAX_EQUILIBRATION_PS),
        ("production_time_ps", 0.0, _SMOKE_MAX_PRODUCTION_PS),
    )
    for name, minimum, maximum in durations:
        value = parameters.get(name)
        valid = _finite_number(value) and float(value) >= minimum and float(value) <= maximum
        if name == "production_time_ps":
            valid = valid and float(value) > 0 if _finite_number(value) else False
        if not valid:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "parameter.%s" % name,
                    "%s exceeds the integration-smoke bound [%.3g, %.3g] ps"
                    % (name, minimum, maximum),
                )
            )
    trajectory_interval = parameters.get("trajectory_interval_steps")
    if not _positive_int(trajectory_interval) or int(trajectory_interval) > 100:
        diagnostics.append(
            _diagnostic(
                "ERROR",
                "parameter.trajectory_interval_steps",
                "trajectory_interval_steps must be an integer in [1, 100]",
            )
        )

    if (
        _positive_number(timestep)
        and _finite_number(parameters.get("npt_time_ps"))
        and _finite_number(parameters.get("nvt_equil_time_ps"))
        and _positive_number(parameters.get("production_time_ps"))
        and _positive_int(trajectory_interval)
        and normalized_temperatures
    ):
        dt = float(timestep)
        npt_steps = int(round(float(parameters["npt_time_ps"]) * 1000.0 / dt))
        nvt_steps = int(round(float(parameters["nvt_equil_time_ps"]) * 1000.0 / dt))
        production_steps = int(round(float(parameters["production_time_ps"]) * 1000.0 / dt))
        frames = int(math.ceil(production_steps / int(trajectory_interval)))
        min_points = parameters.get("min_msd_fit_points")
        if production_steps < 1 or not _positive_int(min_points) or frames < int(min_points):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "parameter.smoke_frames",
                    "production/trajectory settings must yield at least min_msd_fit_points frames",
                )
            )
        total_steps = len(normalized_temperatures) * (npt_steps + nvt_steps + production_steps)
        if total_steps > _SMOKE_MAX_TOTAL_STEPS:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "parameter.smoke_step_cap",
                    "bounded smoke permits at most %d total MD steps" % _SMOKE_MAX_TOTAL_STEPS,
                )
            )

    device = parameters.get("device")
    if device not in _SMOKE_DEVICES:
        diagnostics.append(
            _diagnostic("ERROR", "parameter.device", "device must be one of %s" % sorted(_SMOKE_DEVICES))
        )
    elif device != "cpu":
        diagnostics.append(
            _diagnostic(
                "WARNING",
                "randomness.gpu_not_bitwise",
                "GPU/MPS execution is not guaranteed bitwise deterministic by the historical source",
            )
        )
    if parameters.get("default_dtype") not in _SMOKE_DTYPES:
        diagnostics.append(
            _diagnostic(
                "ERROR", "parameter.default_dtype", "default_dtype must be float32 or float64"
            )
        )
    for name in ("input_format", "input_index"):
        value = parameters.get(name)
        if name == "input_format" and value is None:
            continue
        if not isinstance(value, (str, int)) or not str(value).strip() or any(
            token in str(value) for token in ("\x00", "\n", "\r")
        ):
            diagnostics.append(
                _diagnostic("ERROR", "parameter.%s" % name, "%s must be a plain scalar" % name)
            )

    if parameters.get("source") != "trajectory":
        diagnostics.append(
            _diagnostic(
                "ERROR", "parameter.source", "md-smoke-and-analyze requires source='trajectory'"
            )
        )
    specie = parameters.get("specie")
    if not isinstance(specie, str) or re.fullmatch(r"[A-Z][a-z]?", specie) is None:
        diagnostics.append(
            _diagnostic("ERROR", "parameter.specie", "specie must be one element symbol")
        )
    _validate_interval(
        parameters, "trajectory_start_ps", "trajectory_end_ps", diagnostics, required=False
    )
    if _finite_number(parameters.get("production_time_ps")):
        production_time = float(parameters["production_time_ps"])
        for name in ("trajectory_end_ps",):
            value = parameters.get(name)
            if value is not None and _finite_number(value) and float(value) > production_time:
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "parameter.%s" % name, "%s cannot exceed production_time_ps" % name
                    )
                )
    choices = {
        "trajectory_msd_engine": {"diffusion-analyzer"},
        "diffusion_analyzer_smoothed": {"max", "constant", "none"},
        "piecewise": {"auto", "always", "never"},
    }
    for name, allowed in choices.items():
        if parameters.get(name) not in allowed:
            diagnostics.append(
                _diagnostic("ERROR", "parameter.%s" % name, "%s must be one of %s" % (name, sorted(allowed)))
            )
    if parameters.get("piecewise") != "never":
        diagnostics.append(
            _diagnostic(
                "ERROR",
                "parameter.piecewise_unsupported",
                "the smoke reuses only the formal single-line Arrhenius analysis",
            )
        )
    if parameters.get("fit_scope") != "all":
        diagnostics.append(
            _diagnostic(
                "ERROR", "parameter.fit_scope", "multi-temperature smoke requires fit_scope='all'"
            )
        )
    for name in (
        "diffusion_analyzer_min_obs",
        "diffusion_analyzer_avg_nsteps",
        "diffusion_analyzer_step_skip",
        "min_msd_fit_points",
    ):
        if not _positive_int(parameters.get(name)):
            diagnostics.append(
                _diagnostic("ERROR", "parameter.%s" % name, "%s must be a positive integer" % name)
            )
    if _positive_int(parameters.get("min_msd_fit_points")) and int(
        parameters["min_msd_fit_points"]
    ) < 3:
        diagnostics.append(
            _diagnostic("ERROR", "parameter.min_msd_fit_points", "min_msd_fit_points must be >= 3")
        )
    for name in ("piecewise_slope_change", "piecewise_bic_delta"):
        value = parameters.get(name)
        if value is not None and (not _finite_number(value) or float(value) < 0):
            diagnostics.append(
                _diagnostic("ERROR", "parameter.%s" % name, "%s must be finite and non-negative" % name)
            )
    if not _positive_number(parameters.get("target_temperature_k")):
        diagnostics.append(
            _diagnostic(
                "ERROR", "parameter.target_temperature_k", "target_temperature_k must be positive"
            )
        )
    for name in ("msd_smooth_window_points",):
        value = parameters.get(name)
        if value is not None and not _positive_int(value):
            diagnostics.append(
                _diagnostic("ERROR", "parameter.%s" % name, "%s must be a positive integer" % name)
            )
    _optional_positive(parameters, "msd_smooth_window_ps", diagnostics)
    if not isinstance(parameters.get("fit_smoothed_msd"), bool):
        diagnostics.append(
            _diagnostic("ERROR", "parameter.fit_smoothed_msd", "fit_smoothed_msd must be boolean")
        )
    if parameters.get("allow_partial_results") is not False:
        diagnostics.append(
            _diagnostic(
                "ERROR",
                "parameter.allow_partial_results",
                "integration smoke requires allow_partial_results=false",
            )
        )
    if _positive_number(timestep) and _positive_int(trajectory_interval):
        derived_frame_step = float(timestep) * int(trajectory_interval)
        supplied_frame_step = parameters.get("ase_frame_step_fs")
        if supplied_frame_step is not None and (
            not _positive_number(supplied_frame_step)
            or not math.isclose(
                float(supplied_frame_step), derived_frame_step, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "parameter.ase_frame_step_fs",
                    "ase_frame_step_fs must equal timestep_fs * trajectory_interval_steps",
                )
            )
    for name in ("fit_temperatures", "exclude_temperatures"):
        value = parameters.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            diagnostics.append(
                _diagnostic("ERROR", "parameter.%s" % name, "%s must be a non-empty string" % name)
            )

    python_executable = resources.get("python_executable", sys.executable)
    if not isinstance(python_executable, str) or not python_executable.strip():
        diagnostics.append(
            _diagnostic(
                "ERROR", "resource.python_executable", "python_executable must be a non-empty string"
            )
        )
    elif not _is_current_python(python_executable):
        diagnostics.append(
            _diagnostic(
                "ERROR",
                "resource.python_executable",
                "formal transport must use the same local Python interpreter that runs MLIPFlow",
            )
        )
    else:
        diagnostics.extend(_formal_dependency_diagnostics())
    return diagnostics


class ManuscriptTransportError(ValueError):
    """Raised when historical evidence is incomplete or scientifically ambiguous."""


def _read_parity_source(path_value: Any, role: str) -> Tuple[str, Dict[str, Any]]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ManuscriptTransportError("%s must name an existing file: %s" % (role, path))
    before = path.stat()
    if before.st_size > _MAX_PARITY_SOURCE_BYTES:
        raise ManuscriptTransportError(
            "%s exceeds the %d-byte local parity read limit: %s"
            % (role, _MAX_PARITY_SOURCE_BYTES, path)
        )
    payload = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ManuscriptTransportError("%s changed while it was being read: %s" % (role, path))
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManuscriptTransportError("%s is not UTF-8 text: %s" % (role, path)) from exc
    provenance = {
        "role": role,
        "path": str(path),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "read_only": True,
    }
    return text, provenance


def _transport_value(value: float, unit_kind: str) -> Dict[str, float]:
    if unit_kind == "diffusivity":
        return {
            "diffusivity_m2_s": float(value),
            "diffusivity_cm2_s": float(value) * 1.0e4,
        }
    if unit_kind == "conductivity":
        return {
            "conductivity_S_m": float(value),
            "conductivity_NE_mS_cm": float(value) * 10.0,
        }
    raise AssertionError("unknown transport unit kind: %s" % unit_kind)


def _history_method_record(diffusivity: float, conductivity: float) -> Dict[str, Any]:
    record: Dict[str, Any] = {}
    record.update(_transport_value(diffusivity, "diffusivity"))
    record.update(_transport_value(conductivity, "conductivity"))
    return record


def parse_historical_transport_output(path_value: Any) -> Dict[str, Any]:
    """Parse Li10_test.out/Li10_MACE.out without executing or modifying them.

    Both historical files use the same four-line record: a linear-fit diffusivity,
    a mean(MSD/t) diffusivity, then the corresponding conductivity (1)/(2).
    Labels such as ``11221`` start named blocks; unlabeled blocks receive stable
    ``block-N`` identifiers. Unrelated scheduler diagnostics are counted but not
    interpreted as scientific records.
    """

    text, provenance = _read_parity_source(path_value, "historical_transport_output")
    diffusion_linear_re = re.compile(
        r"^\s*Diffusion\s+Coefficients\s+of\s+Li\+?\s*:\s*(%s)\s*m\^?2/s\s*$"
        % _FLOAT_PATTERN,
        re.IGNORECASE,
    )
    diffusion_ratio_re = re.compile(r"^\s*扩散系数\s*=\s*(%s)\s*$" % _FLOAT_PATTERN)
    conductivity_re = re.compile(
        r"^\s*(%s)\s*(?:电导率|conductivity)\s*[\(（]\s*([12])\s*[\)）]"
        r"\s*[:：]\s*(%s)\s*S/m\s*$" % (_FLOAT_PATTERN, _FLOAT_PATTERN),
        re.IGNORECASE,
    )
    label_re = re.compile(r"^[A-Za-z0-9_.+-]+$")

    datasets: List[Dict[str, Any]] = []
    current_label: Optional[str] = None
    current_records: List[Dict[str, Any]] = []
    pending: Dict[str, Any] = {}
    ignored_line_count = 0

    def finish_dataset() -> None:
        nonlocal current_label, current_records, pending
        if pending:
            raise ManuscriptTransportError(
                "historical output ended a block with an incomplete four-line record"
            )
        if not current_records:
            current_label = None
            return
        temperatures = [item["temperature_K"] for item in current_records]
        if len(temperatures) != len(set(temperatures)):
            raise ManuscriptTransportError("historical output repeats a temperature in one block")
        block_index = len(datasets) + 1
        datasets.append(
            {
                "dataset_id": current_label or "block-%d" % block_index,
                "source_label": current_label,
                "block_index": block_index,
                "by_temperature": current_records,
            }
        )
        current_label = None
        current_records = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            if current_records:
                finish_dataset()
            continue
        matched = diffusion_linear_re.match(line)
        if matched:
            if pending:
                raise ManuscriptTransportError(
                    "line %d starts a new record before the previous record completed" % line_number
                )
            pending["linear_diffusivity_m2_s"] = float(matched.group(1))
            continue
        matched = diffusion_ratio_re.match(line)
        if matched:
            if "linear_diffusivity_m2_s" not in pending or len(pending) != 1:
                raise ManuscriptTransportError(
                    "line %d has mean(MSD/t) diffusivity out of order" % line_number
                )
            pending["ratio_diffusivity_m2_s"] = float(matched.group(1))
            continue
        matched = conductivity_re.match(line)
        if matched:
            temperature = float(matched.group(1))
            method_number = int(matched.group(2))
            conductivity = float(matched.group(3))
            if method_number == 1:
                expected = {"linear_diffusivity_m2_s", "ratio_diffusivity_m2_s"}
                if set(pending) != expected:
                    raise ManuscriptTransportError(
                        "line %d has conductivity (1) before both diffusivities" % line_number
                    )
                pending["temperature_K"] = temperature
                pending["linear_conductivity_S_m"] = conductivity
                continue
            expected = {
                "linear_diffusivity_m2_s",
                "ratio_diffusivity_m2_s",
                "temperature_K",
                "linear_conductivity_S_m",
            }
            if set(pending) != expected:
                raise ManuscriptTransportError(
                    "line %d has conductivity (2) before conductivity (1)" % line_number
                )
            if not math.isclose(
                temperature, float(pending["temperature_K"]), rel_tol=0.0, abs_tol=1e-12
            ):
                raise ManuscriptTransportError(
                    "line %d changes temperature inside one transport record" % line_number
                )
            current_records.append(
                {
                    "temperature_K": temperature,
                    "methods": {
                        "linear_msd_fit": _history_method_record(
                            float(pending["linear_diffusivity_m2_s"]),
                            float(pending["linear_conductivity_S_m"]),
                        ),
                        "mean_msd_over_time": _history_method_record(
                            float(pending["ratio_diffusivity_m2_s"]), conductivity
                        ),
                    },
                }
            )
            pending = {}
            continue

        if any(
            token in line
            for token in ("Diffusion Coefficients", "扩散系数", "电导率", "conductivity")
        ):
            raise ManuscriptTransportError(
                "line %d resembles a transport result but has an unknown format" % line_number
            )
        if label_re.fullmatch(line) and not pending:
            if current_records:
                finish_dataset()
            current_label = line
        else:
            ignored_line_count += 1

    finish_dataset()
    if not datasets:
        raise ManuscriptTransportError("historical output contains no complete transport records")
    return {
        "schema_version": 1,
        "artifact_type": "mlipflow.historical_transport_output",
        "status": "OK",
        "source_format": "li10_dual_method_stdout_v1",
        "datasets": datasets,
        "units": {
            "temperature_K": "K",
            "diffusivity_m2_s": "m^2/s",
            "diffusivity_cm2_s": "cm^2/s",
            "conductivity_S_m": "S/m",
            "conductivity_NE_mS_cm": "mS/cm",
        },
        "fit": {
            "linear_msd_fit": "historical stdout method (1); fit window is not encoded",
            "mean_msd_over_time": "historical stdout method (2); arithmetic mean of MSD/t",
            "arrhenius": "not encoded in Li10_test.out or Li10_MACE.out",
        },
        "convention": {
            "carrier_count": "not inferable from stdout; inspect the producing script",
            "conductivity_model": "uncorrected Nernst-Einstein in producing scripts",
            "methods_are_not_interchangeable": True,
        },
        "provenance": {
            "parser": "ionic-transport.adapter.parse_historical_transport_output",
            "parser_version": MANUSCRIPT_PARITY_VERSION,
            "ignored_non_scientific_line_count": ignored_line_count,
            "sources": [provenance],
        },
    }


def _sampling_profile(name: str) -> Dict[str, Any]:
    try:
        return dict(_SAMPLING_PROFILES[name])
    except KeyError as exc:
        raise ManuscriptTransportError(
            "sampling_profile must be one of %s" % sorted(_SAMPLING_PROFILES)
        ) from exc


def analyze_historical_target_msd(path_value: Any, sampling_profile: str) -> Dict[str, Any]:
    """Reproduce both diffusivity calculations in the historical MSD scripts."""

    text, provenance = _read_parity_source(path_value, "target_msd")
    lines = text.splitlines()
    profile = _sampling_profile(sampling_profile)
    header_index: Optional[int] = None
    msd_column_index: Optional[int] = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") and "TimeStep" in stripped and "c_msd1[4]" in stripped:
            header = stripped.lstrip("#").split()
            header_index = index
            msd_column_index = header.index("c_msd1[4]")
            break
    if header_index is None or msd_column_index is None:
        raise ManuscriptTransportError(
            "target.msd must declare '# TimeStep ... c_msd1[4] ...'"
        )

    timesteps: List[float] = []
    msd_values: List[float] = []
    sampled_line_indices: List[int] = []
    start = int(profile["start_line_index"])
    modulus = int(profile["line_modulus"])
    offset = int(profile["line_offset"])
    timestep_fs = float(profile["timestep_fs"])
    for line_index in range(start, len(lines)):
        if (line_index - offset) % modulus != 0:
            continue
        fields = lines[line_index].split()
        if len(fields) <= msd_column_index:
            raise ManuscriptTransportError(
                "target.msd line index %d is not a complete numeric row" % line_index
            )
        try:
            timestep = float(fields[0])
            msd = float(fields[msd_column_index])
        except ValueError as exc:
            raise ManuscriptTransportError(
                "target.msd line index %d contains non-numeric data" % line_index
            ) from exc
        if not math.isfinite(timestep) or timestep <= 0:
            raise ManuscriptTransportError(
                "sampled target.msd timestep must be finite and positive at line index %d"
                % line_index
            )
        if not math.isfinite(msd):
            raise ManuscriptTransportError(
                "sampled target.msd MSD must be finite at line index %d" % line_index
            )
        timesteps.append(timestep)
        msd_values.append(msd)
        sampled_line_indices.append(line_index)
    if len(timesteps) < 3:
        raise ManuscriptTransportError(
            "sampling profile %s selected fewer than three target.msd points" % sampling_profile
        )
    if any(right <= left for left, right in zip(timesteps, timesteps[1:])):
        raise ManuscriptTransportError("sampled target.msd timesteps must increase strictly")

    try:
        fit = linear_fit(timesteps, msd_values)
    except ValueError as exc:
        raise ManuscriptTransportError(str(exc)) from exc
    slope_a2_fs = float(fit["slope"]) / timestep_fs
    mean_ratio_a2_fs = sum(
        msd / (timestep * timestep_fs)
        for timestep, msd in zip(timesteps, msd_values)
    ) / len(timesteps)
    diffusion_linear = slope_a2_fs / 6.0 * 1.0e-5
    diffusion_ratio = mean_ratio_a2_fs / 6.0 * 1.0e-5
    if diffusion_linear <= 0 or diffusion_ratio <= 0:
        raise ManuscriptTransportError(
            "historical MSD analysis produced non-positive diffusivity; Arrhenius log is undefined"
        )

    sampled_digest = hashlib.sha256(
        "\n".join(
            "%d %.17g %.17g" % (line_index, timestep, msd)
            for line_index, timestep, msd in zip(
                sampled_line_indices, timesteps, msd_values
            )
        ).encode("ascii")
    ).hexdigest()
    linear_result: Dict[str, Any] = {
        "method": "linear_msd_fit",
        "slope_A2_fs": slope_a2_fs,
        "intercept_A2": float(fit["intercept"]),
        "r2": float(fit["r_squared"]),
    }
    linear_result.update(_transport_value(diffusion_linear, "diffusivity"))
    ratio_result: Dict[str, Any] = {
        "method": "mean_msd_over_time",
        "mean_MSD_over_t_A2_fs": mean_ratio_a2_fs,
    }
    ratio_result.update(_transport_value(diffusion_ratio, "diffusivity"))
    return {
        "status": "OK",
        "source_format": "lammps_fix_ave_time_target_msd",
        "mobile_msd_column": "c_msd1[4]",
        "sampling": {
            "profile": sampling_profile,
            **profile,
            "selected_point_count": len(timesteps),
            "first_selected_source_line_index": sampled_line_indices[0],
            "last_selected_source_line_index": sampled_line_indices[-1],
            "first_timestep": timesteps[0],
            "last_timestep": timesteps[-1],
            "first_time_ps": timesteps[0] * timestep_fs / 1000.0,
            "last_time_ps": timesteps[-1] * timestep_fs / 1000.0,
            "sampled_points_sha256": sampled_digest,
        },
        "methods": {
            "linear_msd_fit": linear_result,
            "mean_msd_over_time": ratio_result,
        },
        "units": {
            "raw_timestep": "LAMMPS step",
            "interpreted_timestep": "fs",
            "msd": "angstrom^2",
            "diffusivity_m2_s": "m^2/s",
            "diffusivity_cm2_s": "cm^2/s",
        },
        "fit": {
            "linear_msd_fit": "numpy.polyfit degree=1 on the exact historical line subset",
            "mean_msd_over_time": "numpy.mean(MSD/t)/6 on the same line subset",
            "dimensions": 3,
        },
        "provenance": {
            "parser": "ionic-transport.adapter.analyze_historical_target_msd",
            "parser_version": MANUSCRIPT_PARITY_VERSION,
            "sources": [provenance],
        },
    }


def _determinant_3x3(matrix: Sequence[Sequence[float]]) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def parse_poscar_mobile_ions(path_value: Any, mobile_species: str = "Li") -> Dict[str, Any]:
    """Read a VASP 5 POSCAR carrier count and volume without changing the structure."""

    if re.fullmatch(r"[A-Z][a-z]?", mobile_species) is None:
        raise ManuscriptTransportError("mobile_species must be one element symbol")
    text, provenance = _read_parity_source(path_value, "carrier_structure_poscar")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        raise ManuscriptTransportError("POSCAR is too short to contain a VASP 5 structure")
    try:
        scale_values = [float(value) for value in lines[1].split()]
        lattice = [[float(value) for value in lines[index].split()] for index in range(2, 5)]
    except ValueError as exc:
        raise ManuscriptTransportError("POSCAR scale/lattice contains non-numeric data") from exc
    if len(scale_values) not in {1, 3} or any(len(vector) != 3 for vector in lattice):
        raise ManuscriptTransportError("POSCAR must contain one/three scale values and 3x3 lattice")
    raw_volume = abs(_determinant_3x3(lattice))
    if raw_volume <= 0:
        raise ManuscriptTransportError("POSCAR lattice has zero volume")
    if len(scale_values) == 1:
        scale = scale_values[0]
        if scale == 0:
            raise ManuscriptTransportError("POSCAR scale must be non-zero")
        volume_a3 = abs(scale) if scale < 0 else raw_volume * scale**3
    else:
        if any(value <= 0 for value in scale_values):
            raise ManuscriptTransportError("three-component POSCAR scale must be positive")
        volume_a3 = raw_volume * math.prod(scale_values)

    species = lines[5].split()
    if not species or all(re.fullmatch(r"\d+", token) for token in species):
        raise ManuscriptTransportError(
            "VASP 4 POSCAR lacks species names; corrected carrier count cannot be inferred"
        )
    try:
        counts = [int(value) for value in lines[6].split()]
    except ValueError as exc:
        raise ManuscriptTransportError("POSCAR atom counts must be integers") from exc
    if len(species) != len(counts) or any(value <= 0 for value in counts):
        raise ManuscriptTransportError("POSCAR species/count rows are inconsistent")
    if mobile_species not in species:
        raise ManuscriptTransportError("POSCAR contains no %s species" % mobile_species)
    n_mobile_ions = sum(count for name, count in zip(species, counts) if name == mobile_species)
    return {
        "n_mobile_ions": n_mobile_ions,
        "mobile_species": mobile_species,
        "volume_A3": volume_a3,
        "species_counts": dict(zip(species, counts)),
        "provenance": provenance,
    }


def parse_legacy_transport_script(path_value: Any) -> Dict[str, Any]:
    """Extract literal N/V/kB/q constants from a historical script; never execute it."""

    text, provenance = _read_parity_source(path_value, "legacy_transport_script")
    assignments: Dict[str, float] = {}
    for name in ("N", "V", "kB", "q", "T"):
        matched = re.search(
            r"(?m)^\s*%s\s*=\s*(%s)(?:\s*#.*)?$" % (re.escape(name), _FLOAT_PATTERN),
            text,
        )
        if matched:
            assignments[name] = float(matched.group(1))
    required = {"N", "V", "kB", "q"}
    if not required.issubset(assignments):
        raise ManuscriptTransportError(
            "legacy transport script lacks literal assignments for %s"
            % sorted(required - set(assignments))
        )
    n_value = assignments["N"]
    if not n_value.is_integer() or n_value <= 0:
        raise ManuscriptTransportError("legacy script N must be a positive integer literal")
    if any(assignments[name] <= 0 for name in ("V", "kB", "q")):
        raise ManuscriptTransportError("legacy script V/kB/q constants must be positive")
    return {
        "n_mobile_ions": int(n_value),
        "volume_m3": assignments["V"],
        "volume_A3": assignments["V"] * 1.0e30,
        "boltzmann_J_K": assignments["kB"],
        "elementary_charge_C": assignments["q"],
        "temperature_K": assignments.get("T"),
        "provenance": provenance,
    }


def _positive_explicit_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ManuscriptTransportError("%s must be an explicit positive integer" % name)
    return value


def _resolve_carrier_convention(
    *,
    convention: str,
    n_mobile_ions: Optional[int],
    volume_a3: Optional[float],
    structure_path: Optional[Any],
    mobile_species: str,
    legacy_script_path: Optional[Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if convention not in {LEGACY_SCRIPT_CONVENTION, COMPOSITION_CORRECTED_CONVENTION}:
        raise ManuscriptTransportError(
            "carrier_convention must be legacy_script or composition_corrected"
        )
    sources: List[Dict[str, Any]] = []
    structure: Optional[Dict[str, Any]] = None
    legacy_script: Optional[Dict[str, Any]] = None
    if structure_path is not None:
        structure = parse_poscar_mobile_ions(structure_path, mobile_species)
        sources.append(structure["provenance"])
    if legacy_script_path is not None:
        legacy_script = parse_legacy_transport_script(legacy_script_path)
        sources.append(legacy_script["provenance"])

    volume_candidates: List[Tuple[str, float]] = []
    if volume_a3 is not None:
        if not _positive_number(volume_a3):
            raise ManuscriptTransportError("volume_a3 must be a finite positive number")
        volume_candidates.append(("explicit", float(volume_a3)))
    if structure is not None:
        volume_candidates.append(("structure", float(structure["volume_A3"])))
    if legacy_script is not None:
        volume_candidates.append(("legacy_script", float(legacy_script["volume_A3"])))
    if not volume_candidates:
        raise ManuscriptTransportError(
            "volume must come from volume_a3, a POSCAR, or a reviewed legacy script"
        )
    resolved_volume = volume_candidates[0][1]
    for source, candidate in volume_candidates[1:]:
        if not math.isclose(candidate, resolved_volume, rel_tol=1e-9, abs_tol=1e-8):
            raise ManuscriptTransportError(
                "volume sources disagree: %.17g versus %.17g A^3 from %s"
                % (resolved_volume, candidate, source)
            )

    if legacy_script is not None:
        if not math.isclose(
            float(legacy_script["boltzmann_J_K"]),
            _HISTORICAL_BOLTZMANN_J_K,
            rel_tol=0.0,
            abs_tol=1e-35,
        ) or not math.isclose(
            float(legacy_script["elementary_charge_C"]),
            _HISTORICAL_ELEMENTARY_CHARGE_C,
            rel_tol=0.0,
            abs_tol=1e-30,
        ):
            raise ManuscriptTransportError(
                "legacy script constants differ from kB=1.38e-23 and q=1.6e-19"
            )

    if convention == LEGACY_SCRIPT_CONVENTION:
        if n_mobile_ions is not None or structure_path is not None:
            raise ManuscriptTransportError(
                "legacy_script fixes N=7; do not provide n_mobile_ions or a structure"
            )
        if legacy_script is not None and legacy_script["n_mobile_ions"] != 7:
            raise ManuscriptTransportError(
                "legacy_script convention requires source-script N=7 exactly"
            )
        count = _LEGACY_LI10_N_MOBILE_IONS
        count_source = "historical Li10 MACE script literal N=7"
        legacy_preserved = True
    else:
        explicit_count = (
            None
            if n_mobile_ions is None
            else _positive_explicit_int(n_mobile_ions, "n_mobile_ions")
        )
        structure_count = None if structure is None else int(structure["n_mobile_ions"])
        if explicit_count is None and structure_count is None:
            raise ManuscriptTransportError(
                "composition_corrected requires explicit n_mobile_ions or a VASP 5 POSCAR"
            )
        if (
            explicit_count is not None
            and structure_count is not None
            and explicit_count != structure_count
        ):
            raise ManuscriptTransportError(
                "explicit n_mobile_ions=%d disagrees with structure count=%d"
                % (explicit_count, structure_count)
            )
        count = explicit_count if explicit_count is not None else int(structure_count)
        count_source = (
            "explicit+structure-verified"
            if explicit_count is not None and structure_count is not None
            else ("explicit" if explicit_count is not None else "structure")
        )
        legacy_preserved = False

    return (
        {
            "name": convention,
            "n_mobile_ions": count,
            "n_mobile_ions_source": count_source,
            "mobile_species": mobile_species,
            "volume_A3": resolved_volume,
            "volume_m3": resolved_volume * 1.0e-30,
            "volume_sources": [name for name, _ in volume_candidates],
            "historical_N7_error_preserved": legacy_preserved,
            "legacy_script_n_mobile_ions": (
                None if legacy_script is None else legacy_script["n_mobile_ions"]
            ),
            "conductivity_model": "uncorrected Nernst-Einstein",
            "dimensions": 3,
            "haven_ratio": 1.0,
            "charge_number": 1.0,
            "constants_convention": "historical rounded constants retained to isolate N/V",
        },
        sources,
    )


def _conductivity_s_m(
    diffusivity_m2_s: float, temperature_k: float, carrier: Mapping[str, Any]
) -> float:
    if diffusivity_m2_s <= 0 or temperature_k <= 0:
        raise ManuscriptTransportError("diffusivity and temperature must be positive")
    try:
        return nernst_einstein_conductivity(
            diffusivity_m2_s,
            temperature_k,
            int(carrier["n_mobile_ions"]),
            float(carrier["charge_number"]),
            float(carrier["volume_A3"]),
            elementary_charge_c=_HISTORICAL_ELEMENTARY_CHARGE_C,
            boltzmann_j_per_k=_HISTORICAL_BOLTZMANN_J_K,
        )["conductivity_s_m"]
    except ValueError as exc:
        raise ManuscriptTransportError(str(exc)) from exc


def _ols(x_values: Sequence[float], y_values: Sequence[float]) -> Dict[str, float]:
    try:
        fit = linear_fit(x_values, y_values)
    except ValueError as exc:
        raise ManuscriptTransportError(str(exc)) from exc
    return {
        "slope": float(fit["slope"]),
        "intercept": float(fit["intercept"]),
        "r2": float(fit["r_squared"]),
    }


def _arrhenius_fit(
    rows: Sequence[Mapping[str, Any]],
    method: str,
    convention: str,
    target_temperature_k: float,
    carrier: Mapping[str, Any],
) -> Dict[str, Any]:
    if not _positive_number(target_temperature_k):
        raise ManuscriptTransportError("target_temperature_k must be positive")
    temperatures = [float(row["temperature_K"]) for row in rows]
    diffusivities = [float(row["methods"][method]["diffusivity_m2_s"]) for row in rows]
    if convention == EXACT_ARRHENIUS_CONVENTION:
        x_values = [1000.0 / value for value in temperatures]
        y_values = [math.log10(value) for value in diffusivities]
        target_x = 1000.0 / float(target_temperature_k)
        rounding = "none"
    elif convention == LEGACY_ARRHENIUS_CONVENTION:
        if set(temperatures) != {400.0, 600.0, 800.0} or len(temperatures) != 3:
            raise ManuscriptTransportError(
                "legacy_get_sigma_v1 requires exactly 400, 600, and 800 K"
            )
        historical_x = {400.0: 2.5, 600.0: 1.667, 800.0: 1.25}
        x_values = [historical_x[value] for value in temperatures]
        y_values = [round(math.log10(value), 2) for value in diffusivities]
        if not math.isclose(float(target_temperature_k), 300.0, rel_tol=0.0, abs_tol=1e-12):
            raise ManuscriptTransportError("legacy_get_sigma_v1 fixes target temperature at 300 K")
        target_x = 3.33
        rounding = "log10(D) rounded to 2 decimals; x600=1.667; x300=3.33"
    else:
        raise ManuscriptTransportError(
            "arrhenius_convention must be exact_unrounded_ols_v1 or legacy_get_sigma_v1"
        )
    fit = _ols(x_values, y_values)
    target_log10_d = fit["slope"] * target_x + fit["intercept"]
    target_d = 10.0**target_log10_d
    target_sigma = _conductivity_s_m(
        target_d, float(target_temperature_k), carrier
    )
    historical_kb_ev_k = (
        _HISTORICAL_BOLTZMANN_J_K / _HISTORICAL_ELEMENTARY_CHARGE_C
    )
    activation_energy_ev = (
        -fit["slope"] * historical_kb_ev_k * math.log(10.0) * 1000.0
    )
    result: Dict[str, Any] = {
        "method": method,
        "convention": convention,
        "regression": "unweighted OLS of log10(D_m2_s) against 1000/T_K",
        "historical_script_usage": (
            "direct get_MSD_Li10.py -> get_sigma.py parity path"
            if convention == LEGACY_ARRHENIUS_CONVENTION
            and method == "mean_msd_over_time"
            else (
                "same legacy transform applied as a secondary comparison; not the original shell path"
                if convention == LEGACY_ARRHENIUS_CONVENTION
                else "wrapper-normalized unrounded Arrhenius fit"
            )
        ),
        "slope_log10_D_per_1000_over_K": fit["slope"],
        "intercept_log10_D": fit["intercept"],
        "r2": fit["r2"],
        "activation_energy_eV": activation_energy_ev,
        "activation_energy_status": "derived by this wrapper from the declared fitted slope",
        "prefactor_m2_s": 10.0 ** fit["intercept"],
        "target_temperature_K": float(target_temperature_k),
        "target_log10_diffusivity_m2_s": target_log10_d,
        "fit_inputs": [
            {
                "temperature_K": temperature,
                "x_1000_over_T_K": x_value,
                "log10_diffusivity_m2_s": y_value,
            }
            for temperature, x_value, y_value in zip(temperatures, x_values, y_values)
        ],
        "rounding": rounding,
    }
    result.update(_transport_value(target_d, "diffusivity"))
    result.update(_transport_value(target_sigma, "conductivity"))
    return result


def _select_historical_dataset(
    parsed: Mapping[str, Any], selector: Optional[str]
) -> Mapping[str, Any]:
    datasets = parsed.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ManuscriptTransportError("parsed historical output has no datasets")
    if selector is None:
        if len(datasets) != 1:
            raise ManuscriptTransportError(
                "historical_dataset is required when stdout contains multiple blocks"
            )
        return datasets[0]
    matches = [item for item in datasets if str(item.get("dataset_id")) == str(selector)]
    if len(matches) != 1:
        raise ManuscriptTransportError(
            "historical_dataset=%s does not select exactly one output block" % selector
        )
    return matches[0]


def _relative_close(actual: float, expected: float, rel_tol: float, abs_tol: float) -> bool:
    return math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol)


def _compare_historical_dataset(
    *,
    reproduced_rows: Sequence[Mapping[str, Any]],
    parsed: Mapping[str, Any],
    selector: Optional[str],
    reproduction_carrier: Mapping[str, Any],
    historical_carrier_convention: str,
    rel_tol: float = 1e-12,
    abs_tol: float = 1e-20,
) -> Dict[str, Any]:
    if historical_carrier_convention not in {
        LEGACY_SCRIPT_CONVENTION,
        "same_as_reproduction",
        "unknown",
    }:
        raise ManuscriptTransportError(
            "historical_carrier_convention must be legacy_script, "
            "same_as_reproduction, or unknown"
        )
    dataset = _select_historical_dataset(parsed, selector)
    historical_by_temperature = {
        float(row["temperature_K"]): row for row in dataset["by_temperature"]
    }
    reproduced_by_temperature = {
        float(row["temperature_K"]): row for row in reproduced_rows
    }
    if set(historical_by_temperature) != set(reproduced_by_temperature):
        raise ManuscriptTransportError(
            "historical and reproduced temperature sets differ: %s versus %s"
            % (
                sorted(historical_by_temperature),
                sorted(reproduced_by_temperature),
            )
        )
    if historical_carrier_convention == LEGACY_SCRIPT_CONVENTION:
        historical_n: Optional[int] = _LEGACY_LI10_N_MOBILE_IONS
    elif historical_carrier_convention == "same_as_reproduction":
        historical_n = int(reproduction_carrier["n_mobile_ions"])
    else:
        historical_n = None
    reproduced_n = int(reproduction_carrier["n_mobile_ions"])
    expected_sigma_ratio = (
        None if historical_n is None else float(reproduced_n) / float(historical_n)
    )

    comparisons: List[Dict[str, Any]] = []
    diffusion_close: List[bool] = []
    conductivity_close: List[bool] = []
    conductivity_ratio_close: List[bool] = []
    for temperature in sorted(reproduced_by_temperature):
        historical_row = historical_by_temperature[temperature]
        reproduced_row = reproduced_by_temperature[temperature]
        for method in ("linear_msd_fit", "mean_msd_over_time"):
            historical_method = historical_row["methods"][method]
            reproduced_method = reproduced_row["methods"][method]
            historical_d = float(historical_method["diffusivity_m2_s"])
            reproduced_d = float(reproduced_method["diffusivity_m2_s"])
            d_close = _relative_close(reproduced_d, historical_d, rel_tol, abs_tol)
            diffusion_close.append(d_close)
            historical_sigma = float(historical_method["conductivity_S_m"])
            reproduced_sigma = float(reproduced_method["conductivity_S_m"])
            sigma_close = _relative_close(
                reproduced_sigma, historical_sigma, rel_tol, abs_tol
            )
            conductivity_close.append(sigma_close)
            actual_ratio = reproduced_sigma / historical_sigma
            ratio_close: Optional[bool] = None
            if expected_sigma_ratio is not None:
                ratio_close = _relative_close(
                    actual_ratio, expected_sigma_ratio, rel_tol, abs_tol
                )
                conductivity_ratio_close.append(ratio_close)
            comparisons.append(
                {
                    "temperature_K": temperature,
                    "method": method,
                    "historical_diffusivity_m2_s": historical_d,
                    "reproduced_diffusivity_m2_s": reproduced_d,
                    "diffusivity_close": d_close,
                    "historical_conductivity_S_m": historical_sigma,
                    "reproduced_conductivity_S_m": reproduced_sigma,
                    "conductivity_close": sigma_close,
                    "conductivity_ratio_reproduced_over_historical": actual_ratio,
                    "expected_conductivity_ratio": expected_sigma_ratio,
                    "conductivity_ratio_close": ratio_close,
                }
            )
    all_d = all(diffusion_close)
    all_sigma = all(conductivity_close)
    all_ratio = bool(conductivity_ratio_close) and all(conductivity_ratio_close)
    if all_d and all_sigma:
        status = "EXACT_NUMERICAL_PARITY"
    elif all_d and expected_sigma_ratio is not None and all_ratio:
        status = "EXPECTED_CARRIER_CONVENTION_DIVERGENCE"
    elif all_d and historical_n is None:
        status = "DIFFUSIVITY_PARITY_CONDUCTIVITY_UNASSESSED"
    else:
        status = "PARITY_MISMATCH"
    return {
        "status": status,
        "historical_dataset_id": dataset["dataset_id"],
        "historical_carrier_convention": historical_carrier_convention,
        "historical_n_mobile_ions": historical_n,
        "reproduced_n_mobile_ions": reproduced_n,
        "diffusivity_all_close": all_d,
        "conductivity_all_close": all_sigma,
        "conductivity_ratio_all_close": all_ratio if historical_n is not None else None,
        "relative_tolerance": rel_tol,
        "absolute_tolerance": abs_tol,
        "comparisons": comparisons,
    }


def reproduce_manuscript_transport(
    target_msd_by_temperature: Mapping[Any, Any],
    *,
    sampling_profile: str,
    carrier_convention: str,
    arrhenius_convention: str,
    target_temperature_k: float = 300.0,
    n_mobile_ions: Optional[int] = None,
    volume_a3: Optional[float] = None,
    structure_path: Optional[Any] = None,
    mobile_species: str = "Li",
    legacy_script_path: Optional[Any] = None,
    historical_output_path: Optional[Any] = None,
    historical_dataset: Optional[str] = None,
    historical_carrier_convention: Optional[str] = None,
) -> Dict[str, Any]:
    """Reproduce a local three-temperature transport series from existing MSD files.

    This is a post-processing wrapper only. It never launches MD, imports a calculator,
    submits a job, or changes any evidence file.
    """

    if not isinstance(target_msd_by_temperature, Mapping) or len(target_msd_by_temperature) < 2:
        raise ManuscriptTransportError(
            "target_msd_by_temperature must map at least two temperatures to target.msd files"
        )
    normalized_inputs: List[Tuple[float, Any]] = []
    for raw_temperature, path in target_msd_by_temperature.items():
        try:
            temperature = float(raw_temperature)
        except (TypeError, ValueError) as exc:
            raise ManuscriptTransportError("temperature keys must be numeric") from exc
        if not math.isfinite(temperature) or temperature <= 0:
            raise ManuscriptTransportError("temperature keys must be finite and positive")
        normalized_inputs.append((temperature, path))
    if len({item[0] for item in normalized_inputs}) != len(normalized_inputs):
        raise ManuscriptTransportError("target MSD mapping contains duplicate temperatures")
    normalized_inputs.sort(key=lambda item: item[0])

    carrier, carrier_sources = _resolve_carrier_convention(
        convention=carrier_convention,
        n_mobile_ions=n_mobile_ions,
        volume_a3=volume_a3,
        structure_path=structure_path,
        mobile_species=mobile_species,
        legacy_script_path=legacy_script_path,
    )
    rows: List[Dict[str, Any]] = []
    source_provenance: List[Dict[str, Any]] = list(carrier_sources)
    for temperature, target_path in normalized_inputs:
        analyzed = analyze_historical_target_msd(target_path, sampling_profile)
        methods: Dict[str, Any] = {}
        for method_name, method_result in analyzed["methods"].items():
            normalized_method = dict(method_result)
            sigma = _conductivity_s_m(
                float(method_result["diffusivity_m2_s"]), temperature, carrier
            )
            normalized_method.update(_transport_value(sigma, "conductivity"))
            methods[method_name] = normalized_method
        rows.append(
            {
                "temperature_K": temperature,
                "methods": methods,
                "msd_fit": analyzed["fit"],
                "sampling": analyzed["sampling"],
            }
        )
        source_provenance.extend(analyzed["provenance"]["sources"])

    arrhenius = {
        method: _arrhenius_fit(
            rows,
            method,
            arrhenius_convention,
            float(target_temperature_k),
            carrier,
        )
        for method in ("linear_msd_fit", "mean_msd_over_time")
    }
    parity: Optional[Dict[str, Any]] = None
    if historical_output_path is not None:
        if historical_carrier_convention is None:
            raise ManuscriptTransportError(
                "historical_carrier_convention is required when comparing historical stdout"
            )
        parsed_history = parse_historical_transport_output(historical_output_path)
        parity = _compare_historical_dataset(
            reproduced_rows=rows,
            parsed=parsed_history,
            selector=historical_dataset,
            reproduction_carrier=carrier,
            historical_carrier_convention=historical_carrier_convention,
        )
        source_provenance.extend(parsed_history["provenance"]["sources"])
    elif historical_dataset is not None or historical_carrier_convention is not None:
        raise ManuscriptTransportError(
            "historical dataset/convention options require historical_output_path"
        )

    wrapper_path = _MODULE_PATH
    wrapper_digest = hashlib.sha256(wrapper_path.read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "artifact_type": "mlipflow.manuscript_transport_parity",
        "status": "OK",
        "execution": {
            "mode": "read-only-postprocess-existing-msd",
            "md_executed": False,
            "scheduler_used": False,
            "network_used": False,
        },
        "carrier_convention": carrier,
        "sampling_convention": {
            "profile": sampling_profile,
            **_sampling_profile(sampling_profile),
            "same_sampling_used_for_all_carrier_conventions": True,
        },
        "by_temperature": rows,
        "arrhenius": {
            "convention": arrhenius_convention,
            "methods": arrhenius,
        },
        "parity": parity,
        "units": {
            "temperature_K": "K",
            "raw_timestep": "LAMMPS step",
            "interpreted_timestep": "fs",
            "msd": "angstrom^2",
            "volume_A3": "angstrom^3",
            "diffusivity_m2_s": "m^2/s",
            "diffusivity_cm2_s": "cm^2/s",
            "conductivity_S_m": "S/m",
            "conductivity_NE_mS_cm": "mS/cm",
            "activation_energy_eV": "eV",
        },
        "constants": {
            "elementary_charge_C": _HISTORICAL_ELEMENTARY_CHARGE_C,
            "boltzmann_J_K": _HISTORICAL_BOLTZMANN_J_K,
            "reason": (
                "historical rounded q/kB retained in both modes so the corrected result "
                "changes only the explicit carrier convention"
            ),
        },
        "provenance": {
            "wrapper": "ionic-transport.adapter.reproduce_manuscript_transport",
            "wrapper_version": MANUSCRIPT_PARITY_VERSION,
            "wrapper_sha256": wrapper_digest,
            "sources": source_provenance,
        },
    }


def _integration_manifest_path(context: Mapping[str, Any]) -> Optional[Path]:
    project_root = _resolve(context.get("project_root"), Path.cwd()) or Path.cwd().resolve()
    execution = _mapping(context.get("execution"))
    plan = _mapping(execution.get("plan"))
    planned = _resolve(plan.get("integration_manifest"), project_root)
    if planned is not None:
        return planned
    attempt_dir = _resolve(context.get("attempt_dir"), project_root)
    if attempt_dir is None:
        return None
    parameters = _mapping(context.get("parameters"))
    relative = parameters.get("integration_manifest", _INTEGRATION_MANIFEST_NAME)
    if not _safe_subdirectory(relative):
        return None
    return (attempt_dir / str(relative)).resolve()


def _verify_smoke_manifest(
    context: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Verify the bounded handoff manifest and every recorded SHA-256."""

    diagnostics: List[Dict[str, str]] = []
    artifacts: List[Dict[str, str]] = []
    project_root = _resolve(context.get("project_root"), Path.cwd()) or Path.cwd().resolve()
    attempt_dir = _resolve(context.get("attempt_dir"), project_root)
    manifest_path = _integration_manifest_path(context)
    if attempt_dir is None or manifest_path is None or not _is_within(manifest_path, attempt_dir):
        diagnostics.append(
            _diagnostic(
                "ERROR",
                "integration.manifest_path",
                "md-smoke integration manifest must be explicit and inside attempt_dir",
            )
        )
        return None, artifacts, diagnostics
    if not manifest_path.is_file():
        diagnostics.append(
            _diagnostic("INFO", "integration.manifest_missing", "integration manifest is pending")
        )
        return None, artifacts, diagnostics
    try:
        if manifest_path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("integration manifest exceeds 8 MiB")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        diagnostics.append(
            _diagnostic("ERROR", "integration.manifest_unreadable", str(exc))
        )
        return None, artifacts, diagnostics
    if not isinstance(manifest, dict):
        diagnostics.append(
            _diagnostic("ERROR", "integration.manifest_type", "integration manifest must be an object")
        )
        return None, artifacts, diagnostics
    expected_scalars = {
        "schema_version": 1,
        "plugin_id": PLUGIN_ID,
        "operation": _SMOKE_OPERATION,
        "status": "OK",
        "scientific_use": "integration-smoke-only",
    }
    for name, expected in expected_scalars.items():
        if manifest.get(name) != expected:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "integration.%s" % name,
                    "integration manifest %s must equal %r" % (name, expected),
                )
            )
    boundaries = _mapping(manifest.get("boundaries"))
    for name in (
        "production_parameters_accepted",
        "scientific_parity_claimed",
        "model_quality_validated",
        "network_access_authorized_by_handoff",
    ):
        if boundaries.get(name) is not False:
            diagnostics.append(
                _diagnostic(
                    "ERROR", "integration.boundary_%s" % name, "integration boundary %s must be false" % name
                )
            )
    execution_record = _mapping(manifest.get("execution"))
    if execution_record.get("analysis_shell") is not False or execution_record.get(
        "analysis_returncode"
    ) != 0:
        diagnostics.append(
            _diagnostic(
                "ERROR", "integration.analysis_execution", "analysis must finish with shell=false and returncode=0"
            )
        )
    if execution_record.get("absolute_paths_recorded") is not False:
        diagnostics.append(
            _diagnostic(
                "ERROR",
                "integration.path_disclosure",
                "integration manifest must redact machine-local absolute paths",
            )
        )

    parameters = _mapping(context.get("parameters"))
    configuration = _mapping(manifest.get("configuration"))
    comparisons = {
        "calculator": parameters.get("calculator"),
        "temperatures_K": [float(value) for value in parameters.get("temperatures_k", [])],
        "base_seed": parameters.get("seed"),
        "timestep_fs": float(parameters.get("timestep_fs", float("nan"))),
        "npt_time_ps": float(parameters.get("npt_time_ps", float("nan"))),
        "nvt_equil_time_ps": float(parameters.get("nvt_equil_time_ps", float("nan"))),
        "production_time_ps": float(parameters.get("production_time_ps", float("nan"))),
        "trajectory_interval_steps": parameters.get("trajectory_interval_steps"),
        "device": parameters.get("device"),
        "default_dtype": parameters.get("default_dtype"),
        "specie": parameters.get("specie"),
    }
    for name, expected in comparisons.items():
        if configuration.get(name) != expected:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "integration.configuration_%s" % name,
                    "integration configuration %s does not match the approved plan" % name,
                )
            )

    project_root = _resolve(context.get("project_root"), Path.cwd()) or Path.cwd().resolve()
    inputs = _mapping(context.get("inputs"))
    expected_source_paths = {
        "historical-ase-md-source": _resolve(inputs.get("md_script"), project_root),
        "mlipflow-transport-analysis-source": _bundled_analysis_script(),
        "mlipflow-handoff-wrapper": _MODULE_PATH.with_name("md_smoke_handoff.py").resolve(),
    }
    expected_input_paths = {
        "structure": _resolve(inputs.get("structure"), project_root),
    }
    raw_model = inputs.get("model", "default")
    if raw_model is not None and str(raw_model).strip().lower() not in {
        "",
        "none",
        "default",
    }:
        expected_input_paths["model"] = _resolve(raw_model, project_root)
    expected_external_by_group = {
        "source_artifacts": expected_source_paths,
        "input_artifacts": expected_input_paths,
    }

    artifact_groups = (
        ("source_artifacts", False),
        ("input_artifacts", False),
        ("trajectory_artifacts", True),
        ("md_artifacts", True),
        ("result_artifacts", True),
    )
    verified_by_group: Dict[str, List[Path]] = {}
    for group_name, confined in artifact_groups:
        records = manifest.get(group_name)
        if not isinstance(records, list) or not records:
            diagnostics.append(
                _diagnostic(
                    "ERROR", "integration.%s" % group_name, "%s must be a non-empty list" % group_name
                )
            )
            continue
        verified: List[Path] = []
        seen_roles: set[str] = set()
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "integration.artifact_record", "%s[%d] must be an object" % (group_name, index)
                    )
                )
                continue
            role = record.get("role")
            if not isinstance(role, str) or not role or role in seen_roles:
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "integration.artifact_role",
                        "%s[%d] must have a unique non-empty role" % (group_name, index),
                    )
                )
                continue
            seen_roles.add(role)
            locator = record.get("path")
            is_relative = record.get("path_is_attempt_relative") is True
            if not isinstance(locator, str) or not locator:
                diagnostics.append(
                    _diagnostic("ERROR", "integration.artifact_path", "artifact path must be non-empty")
                )
                continue
            expected_external = expected_external_by_group.get(group_name)
            if expected_external is not None:
                path = expected_external.get(role)
                if (
                    path is None
                    or locator != path.name
                    or record.get("locator_kind") != "local-basename"
                    or record.get("portable") is not False
                    or record.get("absolute_path_recorded") is not False
                    or is_relative
                ):
                    diagnostics.append(
                        _diagnostic(
                            "ERROR",
                            "integration.external_locator",
                            "%s[%d] must be the approved non-portable local basename"
                            % (group_name, index),
                        )
                    )
                    continue
                path = path.resolve()
            else:
                path = (attempt_dir / locator).resolve() if is_relative else Path(locator).resolve()
            if confined and (not is_relative or not _is_within(path, attempt_dir)):
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "integration.artifact_escape",
                        "%s[%d] must be attempt-relative and confined" % (group_name, index),
                    )
                )
                continue
            if not path.is_file():
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "integration.artifact_missing", "recorded artifact is missing: %s" % path
                    )
                )
                continue
            expected_sha = record.get("sha256")
            if expected_sha != _sha256_file(path):
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "integration.artifact_sha256", "recorded SHA-256 does not match: %s" % path
                    )
                )
                continue
            if record.get("size_bytes") != path.stat().st_size:
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "integration.artifact_size", "recorded size does not match: %s" % path
                    )
                )
                continue
            verified.append(path)
        expected_external = expected_external_by_group.get(group_name)
        if expected_external is not None and seen_roles != set(expected_external):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "integration.artifact_roles",
                    "%s roles do not match the approved inputs" % group_name,
                )
            )
        verified_by_group[group_name] = verified

    temperatures = parameters.get("temperatures_k", [])
    trajectories = verified_by_group.get("trajectory_artifacts", [])
    if len(trajectories) != len(temperatures):
        diagnostics.append(
            _diagnostic(
                "ERROR",
                "integration.trajectory_count",
                "one hashed production trajectory is required per temperature",
            )
        )
    result_names = {path.name for path in verified_by_group.get("result_artifacts", [])}
    missing_results = set(_REQUIRED_RESULT_NAMES) - result_names
    if missing_results:
        diagnostics.append(
            _diagnostic(
                "ERROR",
                "integration.result_artifacts",
                "integration manifest omits required result hashes: %s" % sorted(missing_results),
            )
        )
    randomness = _mapping(manifest.get("randomness"))
    if randomness.get("framework_rngs_fully_controlled") is not False or randomness.get(
        "gpu_bitwise_determinism_guaranteed"
    ) is not False:
        diagnostics.append(
            _diagnostic(
                "ERROR",
                "integration.randomness_boundary",
                "manifest must not claim full framework/GPU determinism",
            )
        )
    artifacts.append(
        {"role": "md-integration-manifest", "path": str(manifest_path), "media_type": "application/json"}
    )
    for path in trajectories:
        artifacts.append(
            {"role": "production-trajectory", "path": str(path), "media_type": "application/octet-stream"}
        )
    return manifest, artifacts, diagnostics


def _verify_analysis_manifest(
    context: Mapping[str, Any], output_dir: Path, manifest_path: Path
) -> List[Dict[str, str]]:
    diagnostics: List[Dict[str, str]] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [_diagnostic("ERROR", "result.analysis_manifest", f"cannot parse manifest: {exc}")]
    if not isinstance(manifest, Mapping):
        return [_diagnostic("ERROR", "result.analysis_manifest", "manifest must be an object")]
    if manifest.get("schema_version") != 1 or manifest.get("plugin_id") != PLUGIN_ID:
        diagnostics.append(
            _diagnostic("ERROR", "result.analysis_manifest_identity", "manifest identity is invalid")
        )
    contract = _mapping(manifest.get("scientific_contract"))
    if (
        contract.get("formal_implementation")
        != "pymatgen-analysis-diffusion public API"
        or contract.get("trajectory_diffusion") != "DiffusionAnalyzer"
        or contract.get("msd_only_diffusion") != "get_diffusivity_from_msd"
        or contract.get("arrhenius") != "fit_arrhenius(mode='linear')"
        or contract.get("historical_implementation")
        != "separate adapter-only legacy reproduction"
    ):
        diagnostics.append(
            _diagnostic(
                "ERROR", "result.analysis_contract", "manifest scientific contract is invalid"
            )
        )

    runtime = manifest.get("runtime_provenance")
    required_runtime = (
        "python_executable",
        "python_version",
        "mlipflow_version",
        "pymatgen_version",
        "pymatgen_analysis_diffusion_version",
        "numpy_version",
    )
    if not isinstance(runtime, Mapping):
        diagnostics.append(
            _diagnostic(
                "ERROR",
                "result.runtime_provenance",
                "manifest runtime_provenance must be an object",
            )
        )
    else:
        expected_runtime, probe_error = _formal_runtime_probe()
        if probe_error is not None or expected_runtime is None:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "result.runtime_probe",
                    f"cannot recheck the formal runtime: {probe_error}",
                )
            )
        else:
            for name in required_runtime:
                actual = runtime.get(name)
                expected = expected_runtime.get(name)
                if not isinstance(actual, str) or not actual or actual != expected:
                    diagnostics.append(
                        _diagnostic(
                            "ERROR",
                            "result.runtime_identity",
                            f"manifest runtime {name} differs from the selected local interpreter",
                        )
                    )
            source_records = manifest.get("source_artifacts")
            uses_ase = isinstance(source_records, list) and any(
                isinstance(record, Mapping)
                and Path(str(record.get("path", ""))).name == "production.traj"
                for record in source_records
            )
            if uses_ase:
                ase_version = runtime.get("ase_version")
                if (
                    not isinstance(ase_version, str)
                    or not ase_version
                    or ase_version != expected_runtime.get("ase_version")
                ):
                    diagnostics.append(
                        _diagnostic(
                            "ERROR",
                            "result.runtime_identity",
                            "manifest ASE version differs from the selected local interpreter",
                        )
                    )

    recorded_parameters = _mapping(manifest.get("parameters"))
    parameters = _mapping(context.get("parameters"))
    operation = _operation_name(parameters)
    adapter_only = {"operation", "output_subdir", "seed", "allow_partial_results"}
    runner_destinations = {
        "temperature_k": "temperature_K",
        "aimd_temperature_k": "aimd_temperature_K",
        "msd_temperature_k": "msd_temperature_K",
        "target_temperature_k": "target_temperature_K",
        "volume_a3": "volume_A3",
    }
    for name, expected in parameters.items():
        if name in adapter_only or name not in _ANALYZE_PARAMETERS:
            continue
        if operation == _SMOKE_OPERATION and name == "ase_frame_step_fs":
            continue
        actual = recorded_parameters.get(runner_destinations.get(name, name))
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            if not _finite_number(actual) or not _numbers_close(float(actual), float(expected)):
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "result.parameter_mismatch", f"manifest parameter {name} differs"
                    )
                )
        elif actual != expected:
            diagnostics.append(
                _diagnostic(
                    "ERROR", "result.parameter_mismatch", f"manifest parameter {name} differs"
                )
            )

    project_root = _resolve(context.get("project_root"), Path.cwd()) or Path.cwd().resolve()
    inputs = _mapping(context.get("inputs"))
    if operation == _SMOKE_OPERATION:
        execution = _mapping(context.get("execution"))
        plan = _mapping(execution.get("plan"))
        smoke_input = _resolve(plan.get("md_output_dir"), project_root)
        expected_inputs = [] if smoke_input is None else [str(smoke_input)]
    else:
        expected_inputs = [
            str(path)
            for path in (_resolve(value, project_root) for value in inputs.get("input_paths", []))
            if path is not None
        ]
    if recorded_parameters.get("input") != expected_inputs:
        diagnostics.append(
            _diagnostic(
                "ERROR", "result.input_paths_mismatch", "manifest inputs differ from approved inputs"
            )
        )
    if recorded_parameters.get("output") != str(output_dir):
        diagnostics.append(
            _diagnostic(
                "ERROR", "result.output_path_mismatch", "manifest output differs from execution plan"
            )
        )

    implementation = manifest.get("implementation_artifacts")
    if not isinstance(implementation, list) or len(implementation) != 1:
        diagnostics.append(
            _diagnostic(
                "ERROR", "result.implementation_artifacts", "manifest implementation is incomplete"
            )
        )
    else:
        expected_implementation = {
            "packaged-analysis-runner": _bundled_analysis_script(),
        }
        for index, record in enumerate(implementation):
            if isinstance(record, Mapping):
                expected_path = expected_implementation.get(str(record.get("role")))
                actual_path = _resolve(record.get("path"), project_root)
                if expected_path is None or actual_path != expected_path:
                    diagnostics.append(
                        _diagnostic(
                            "ERROR",
                            "result.implementation_identity",
                            f"implementation_artifacts[{index}] is not the pinned implementation",
                        )
                    )
            _, record_diagnostics = _verify_file_record(
                record, role=f"implementation_artifacts[{index}]"
            )
            diagnostics.extend(record_diagnostics)

    sources = manifest.get("source_artifacts")
    if not isinstance(sources, list) or not sources:
        diagnostics.append(
            _diagnostic("ERROR", "result.source_artifacts", "manifest has no source artifacts")
        )
    else:
        approved_roots = [Path(value).resolve() for value in expected_inputs]
        structure_input = _resolve(inputs.get("structure"), project_root)
        if structure_input is not None:
            approved_roots.append(structure_input)
        for index, record in enumerate(sources):
            path, record_diagnostics = _verify_file_record(
                record, role=f"source_artifacts[{index}]"
            )
            diagnostics.extend(record_diagnostics)
            if path is not None and not any(_is_within(path, root) for root in approved_roots):
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "result.source_outside_inputs",
                        f"source_artifacts[{index}] is outside the approved inputs",
                    )
                )
    results = manifest.get("result_artifacts")
    if not isinstance(results, list) or not results:
        diagnostics.append(
            _diagnostic("ERROR", "result.result_artifacts", "manifest has no result artifacts")
        )
    else:
        for index, record in enumerate(results):
            _, record_diagnostics = _verify_file_record(
                record, role=f"result_artifacts[{index}]", confined_to=output_dir
            )
            diagnostics.extend(record_diagnostics)
    return diagnostics


def _load_formal_runner():
    name = "mlipflow_ionic_transport_formal_runner"
    spec = importlib.util.spec_from_file_location(name, _bundled_analysis_script())
    if spec is None or spec.loader is None:
        raise ImportError("cannot load bundled formal ionic-transport runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _optional_csv_number(value: Any) -> Optional[float]:
    if value is None or str(value).strip().lower() in {"", "nan", "none", "null"}:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _verify_transport_rows(
    rows: Sequence[Mapping[str, str]], output_dir: Path
) -> List[Dict[str, str]]:
    diagnostics: List[Dict[str, str]] = []
    try:
        manifest = json.loads(
            (output_dir / "analysis_manifest.json").read_text(encoding="utf-8")
        )
        runner = _load_formal_runner()
        args = argparse.Namespace(**_mapping(manifest.get("parameters")))
    except (OSError, UnicodeError, json.JSONDecodeError, ImportError, TypeError) as exc:
        return [
            _diagnostic(
                "ERROR",
                "result.formal_dependency",
                f"formal checker cannot load pymatgen analysis: {exc}",
            )
        ]

    for index, row in enumerate(rows, start=1):
        curve_path = _resolve(row.get("msd_curve_csv"), output_dir)
        if curve_path is None or not curve_path.is_file() or not _is_within(curve_path, output_dir):
            continue
        try:
            run = runner.load_run_data(
                str(row["dataset"]), Path(str(row["run_dir"])).resolve(), args
            )
            expected = runner.formal_result_values(run, args)
            curve_reader = csv.DictReader(curve_path.read_text(encoding="utf-8").splitlines())
            curve_rows = list(curve_reader)
            required_curve_columns = {"time_ps", "msd_A2", "used_for_analysis"}
            if curve_reader.fieldnames is None or not required_curve_columns.issubset(
                curve_reader.fieldnames
            ):
                raise ValueError("curve columns are incomplete")
            times = [float(item["time_ps"]) for item in curve_rows]
            msd = [float(item["msd_A2"]) for item in curve_rows]
            used = [
                str(item["used_for_analysis"]).strip().lower() in {"true", "1"}
                for item in curve_rows
            ]
            expected_times = [float(value) for value in expected["time_ps"]]
            expected_msd = [float(value) for value in expected["msd_A2"]]
            expected_used = [bool(value) for value in expected["used"]]
            if len(times) != len(expected_times) or any(
                not _numbers_close(actual, wanted)
                for actual, wanted in zip(times, expected_times)
            ):
                raise ValueError("time_ps differs from the rerun pymatgen analysis")
            if len(msd) != len(expected_msd) or any(
                not _numbers_close(actual, wanted)
                for actual, wanted in zip(msd, expected_msd)
            ):
                raise ValueError("MSD differs from the rerun pymatgen analysis")
            if used != expected_used:
                raise ValueError("analysis point mask differs from the approved window")
        except (OSError, UnicodeError, csv.Error, KeyError, TypeError, ValueError) as exc:
            diagnostics.append(
                _diagnostic(
                    "ERROR", "result.pymatgen_recheck", f"row {index} cannot be verified: {exc}"
                )
            )
            continue

        comparisons = {
            "diffusivity_cm2_s": expected["diffusivity_cm2_s"],
            "diffusivity_std_dev_cm2_s": expected["diffusivity_std_dev_cm2_s"],
            "conductivity_NE_mS_cm": expected["conductivity_mS_cm"],
            "conductivity_std_dev_mS_cm": expected["conductivity_std_dev_mS_cm"],
            "chg_diffusivity_cm2_s": expected["chg_diffusivity_cm2_s"],
            "chg_conductivity_mS_cm": expected["chg_conductivity_mS_cm"],
            "haven_ratio": expected["haven_ratio"],
        }
        for name, wanted in comparisons.items():
            try:
                actual = _optional_csv_number(row.get(name))
            except (TypeError, ValueError):
                actual = None
            if wanted is None:
                matches = actual is None
            else:
                matches = actual is not None and _numbers_close(actual, float(wanted))
            if not matches:
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "result.pymatgen_value",
                        f"row {index} {name} differs from the rerun pymatgen API result",
                    )
                )
        methods = {
            "diffusion_method": expected["diffusion_method"],
            "conductivity_method": expected["conductivity_method"],
            "arrhenius_method": "pymatgen-fit-arrhenius-linear",
        }
        for name, wanted in methods.items():
            if row.get(name) != wanted:
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "result.method_identity",
                        f"row {index} {name} is not {wanted}",
                    )
                )
    return diagnostics


def _verify_arrhenius_summary(summary: Mapping[str, Any]) -> List[Dict[str, str]]:
    diagnostics: List[Dict[str, str]] = []
    candidates: List[Tuple[str, Mapping[str, Any]]] = []
    if isinstance(summary.get("single"), Mapping):
        candidates.append(("all", summary))
    datasets = summary.get("datasets")
    if isinstance(datasets, Mapping):
        candidates.extend(
            (str(name), value) for name, value in datasets.items() if isinstance(value, Mapping)
        )
    for label, candidate in candidates:
        try:
            from pymatgen.analysis.diffusion.analyzer import (
                fit_arrhenius,
                get_extrapolated_diffusivity,
            )

            temperatures = [float(value) for value in candidate["fit_temperatures_K"]]
            diffusivities = [float(value) for value in candidate["fit_diffusivities_cm2_s"]]
            target = float(candidate["target_temperature_K"])
            ea_eV, prefactor, ea_std = fit_arrhenius(
                temperatures, diffusivities, mode="linear"
            )
            target_diffusivity = get_extrapolated_diffusivity(
                temperatures, diffusivities, target, mode="linear"
            )
            single = _mapping(candidate["single"])
            expected = {
                "Ea_eV": ea_eV,
                "D0_cm2_s": prefactor,
                f"D_{target:g}K_cm2_s": target_diffusivity,
            }
            for name, value in expected.items():
                if not _finite_number(single.get(name)) or not _numbers_close(
                    float(single[name]), float(value)
                ):
                    raise ValueError(f"{name} differs from the temperature rows")
            actual_std = single.get("Ea_stderr_eV")
            if ea_std is None:
                if actual_std is not None:
                    raise ValueError("Ea_stderr_eV must be null")
            elif not _finite_number(actual_std) or not _numbers_close(
                float(actual_std), float(ea_std)
            ):
                raise ValueError("Ea_stderr_eV differs from pymatgen")
            if candidate.get("arrhenius_method") != "pymatgen-fit-arrhenius-linear":
                raise ValueError("Arrhenius method identity is invalid")
        except (ImportError, KeyError, TypeError, ValueError) as exc:
            diagnostics.append(
                _diagnostic(
                    "ERROR", "result.arrhenius_relation", f"Arrhenius summary {label} failed: {exc}"
                )
            )
    return diagnostics


class Adapter:
    """Plan and collect reviewed analysis or its bounded local MD smoke handoff."""

    def validate(self, context: Any) -> List[Dict[str, str]]:
        diagnostics: List[Dict[str, str]] = []
        if not isinstance(context, Mapping):
            return [_diagnostic("ERROR", "context.mapping_required", "context must be a mapping")]

        project_root = _resolve(context.get("project_root"), Path.cwd())
        if project_root is None or not project_root.is_dir():
            diagnostics.append(
                _diagnostic(
                    "ERROR", "path.project_root", "project_root must name an existing directory"
                )
            )
            project_root = Path.cwd().resolve()
        attempt_dir = _resolve(context.get("attempt_dir"), project_root)
        if attempt_dir is None:
            diagnostics.append(
                _diagnostic("ERROR", "path.attempt_dir", "attempt_dir must be an explicit path")
            )
            attempt_dir = project_root / ".mlipflow-invalid-attempt"
        elif not _is_within(attempt_dir, project_root):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "path.attempt_outside_project",
                    "attempt_dir must stay inside project_root",
                )
            )
        if context.get("backend", "local") != "local":
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "backend.local_only",
                    "analysis is exposed as local argv only; scheduler wrapping belongs to the core",
                )
            )

        inputs = _mapping(context.get("inputs"))
        parameters = _mapping(context.get("parameters"))
        resources = _mapping(context.get("resources"))
        if not isinstance(context.get("inputs", {}), Mapping):
            diagnostics.append(
                _diagnostic("ERROR", "inputs.mapping_required", "inputs must be a mapping")
            )
        if not isinstance(context.get("parameters", {}), Mapping):
            diagnostics.append(
                _diagnostic("ERROR", "parameters.mapping_required", "parameters must be a mapping")
            )
        if not isinstance(context.get("resources", {}), Mapping):
            diagnostics.append(
                _diagnostic("ERROR", "resources.mapping_required", "resources must be a mapping")
            )

        operation = _operation_name(parameters)
        if operation not in {_ANALYZE_OPERATION, _SMOKE_OPERATION}:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "parameter.operation",
                    "operation must be analyze-existing or md-smoke-and-analyze",
                )
            )
            return diagnostics
        if operation == _SMOKE_OPERATION:
            diagnostics.extend(
                _validate_md_smoke(
                    project_root=project_root,
                    attempt_dir=attempt_dir,
                    inputs=inputs,
                    parameters=parameters,
                    resources=resources,
                )
            )
            return diagnostics

        unknown_parameters = sorted(set(parameters) - _ANALYZE_PARAMETERS)
        if unknown_parameters:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "parameter.unknown",
                    "unsupported analyze-existing parameter(s): %s"
                    % ", ".join(unknown_parameters),
                )
            )

        if "analysis_script" in inputs:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "input.analysis_script_unsupported",
                    "analysis_script is packaged by MLIPFlow and cannot be overridden",
                )
            )

        script = _bundled_analysis_script()
        if not script.is_file():
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "path.bundled_analysis_missing",
                    "the packaged ionic_conductivity.py runner is missing",
                )
            )

        raw_paths = inputs.get("input_paths")
        input_paths: List[Path] = []
        if not isinstance(raw_paths, list) or not raw_paths:
            diagnostics.append(
                _diagnostic(
                    "ERROR", "path.input_paths", "inputs.input_paths must be a non-empty list"
                )
            )
        else:
            for index, raw_path in enumerate(raw_paths):
                resolved = _resolve(raw_path, project_root)
                if resolved is None or not resolved.exists():
                    diagnostics.append(
                        _diagnostic(
                            "ERROR",
                            "path.input",
                            "input_paths[%d] must name an existing file or directory" % index,
                        )
                    )
                else:
                    input_paths.append(resolved)
        structure_input = _resolve(inputs.get("structure"), project_root)
        if inputs.get("structure") is not None and (
            structure_input is None or not structure_input.is_file()
        ):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "path.structure",
                    "inputs.structure must name a real structure file for MSD conductivity",
                )
            )

        output_subdir = parameters.get("output_subdir", "ionic-transport-postprocess")
        if not _safe_subdirectory(output_subdir):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "path.output_subdir",
                    "output_subdir must be a non-empty relative path without '..'",
                )
            )
            output_dir = attempt_dir / "ionic-transport-postprocess"
        else:
            output_dir = (attempt_dir / str(output_subdir)).resolve()
            if not _is_within(output_dir, attempt_dir):
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "path.output_escape",
                        "analysis output must stay inside attempt_dir",
                    )
                )
        if output_dir.exists():
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "path.output_exists",
                    "source CLI overwrites result files; use a fresh attempt/output_subdir",
                )
            )
        for input_path in input_paths:
            if _is_within(output_dir, input_path) or _is_within(input_path, output_dir):
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "path.input_output_overlap",
                        "input and output paths must not overlap",
                    )
                )
                break
        if structure_input is not None and (
            _is_within(output_dir, structure_input) or _is_within(structure_input, output_dir)
        ):
            diagnostics.append(
                _diagnostic(
                    "ERROR", "path.input_output_overlap", "structure and output paths must not overlap"
                )
            )

        python_executable = resources.get("python_executable", sys.executable)
        if not isinstance(python_executable, str) or not python_executable.strip():
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "resource.python_executable",
                    "python_executable must be a non-empty string",
                )
            )
        elif not _is_current_python(python_executable):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "resource.python_executable",
                    "formal transport must use the same local Python interpreter that runs MLIPFlow",
                )
            )
        else:
            diagnostics.extend(_formal_dependency_diagnostics())

        source = parameters.get("source")
        if source not in {"auto", "trajectory", "vasp", "msd"}:
            diagnostics.append(
                _diagnostic(
                    "ERROR", "parameter.source", "source must be auto, trajectory, vasp, or msd"
                )
            )
        specie = parameters.get("specie")
        if not isinstance(specie, str) or re.fullmatch(r"[A-Z][a-z]?", specie) is None:
            diagnostics.append(
                _diagnostic("ERROR", "parameter.specie", "specie must be one element symbol")
            )
        if parameters.get("seed") is not None:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "parameter.seed_unsupported",
                    "post-processing has no stochastic stage or seed CLI; omit seed instead of recording an unused value",
                )
            )

        if source in {"trajectory", "vasp"} and (
            parameters.get("fit_start_ps") is not None
            or parameters.get("fit_end_ps") is not None
        ):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "parameter.msd_fit_window_trajectory",
                    "fit_start_ps/fit_end_ps apply only to MSD-table input",
                )
            )
        else:
            _validate_interval(
                parameters, "fit_start_ps", "fit_end_ps", diagnostics, required=False
            )
        _validate_interval(
            parameters, "trajectory_start_ps", "trajectory_end_ps", diagnostics, required=False
        )
        if (
            not _positive_int(parameters.get("min_msd_fit_points"))
            or int(parameters.get("min_msd_fit_points", 0)) < 3
        ):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "parameter.min_msd_fit_points",
                    "min_msd_fit_points must be an integer >= 3",
                )
            )

        choices = {
            "trajectory_msd_engine": {"diffusion-analyzer"},
            "diffusion_analyzer_smoothed": {"max", "constant", "none"},
            "msd_time_unit": {"auto", "ps", "fs", "ns", "step"},
            "msd_unit": {"auto", "A2", "nm2"},
            "fit_scope": {"dataset", "all"},
            "piecewise": {"auto", "always", "never"},
        }
        for name, allowed in choices.items():
            if parameters.get(name) not in allowed:
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "parameter.%s" % name,
                        "%s must be one of %s" % (name, sorted(allowed)),
                    )
                )
        if parameters.get("piecewise") != "never":
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "parameter.piecewise_unsupported",
                    "the formal workflow supports only the declared single-line Arrhenius fit",
                )
            )

        if source == "msd":
            if parameters.get("msd_time_unit") == "auto":
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "unit.msd_time_explicit",
                        "MSD input requires an explicit time unit",
                    )
                )
            if parameters.get("msd_unit") == "auto":
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "unit.msd_explicit", "MSD input requires an explicit MSD unit"
                    )
                )
            if parameters.get("n_mobile_ions") is not None or parameters.get("volume_a3") is not None:
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "assumption.legacy_ne_inputs_unsupported",
                        "formal MSD conductivity does not accept n_mobile_ions/volume_a3; "
                        "provide inputs.structure or leave conductivity unavailable",
                    )
                )
        elif parameters.get("msd_time_unit") == "auto" or parameters.get("msd_unit") == "auto":
            diagnostics.append(
                _diagnostic(
                    "WARNING",
                    "unit.auto_detection",
                    "auto source/unit detection is delegated to the reviewed CLI and recorded in its outputs",
                )
            )
        if parameters.get("msd_time_unit") == "step" and not _positive_number(
            parameters.get("msd_step_ps")
        ):
            diagnostics.append(
                _diagnostic(
                    "ERROR", "unit.msd_step_ps", "step time units require positive msd_step_ps"
                )
            )

        for name in (
            "ase_frame_step_fs",
            "lammps_timestep_ps",
            "vasp_step_fs",
            "temperature_k",
            "aimd_temperature_k",
            "msd_step_ps",
            "msd_temperature_k",
            "volume_a3",
            "msd_smooth_window_ps",
        ):
            _optional_positive(parameters, name, diagnostics)
        if not _positive_number(parameters.get("target_temperature_k")):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "parameter.target_temperature_k",
                    "target_temperature_k must be a finite positive number",
                )
            )
        for name in ("piecewise_slope_change", "piecewise_bic_delta"):
            value = parameters.get(name)
            if value is not None and (not _finite_number(value) or float(value) < 0):
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "parameter.%s" % name, "%s must be finite and non-negative" % name
                    )
                )
        for name in (
            "diffusion_analyzer_min_obs",
            "diffusion_analyzer_avg_nsteps",
            "diffusion_analyzer_step_skip",
        ):
            value = parameters.get(name)
            if not _positive_int(value):
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "parameter.%s" % name, "%s must be a positive integer" % name
                    )
                )
        for name in ("n_mobile_ions", "msd_smooth_window_points", "mobile_type"):
            value = parameters.get(name)
            if value is not None and not _positive_int(value):
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "parameter.%s" % name, "%s must be a positive integer" % name
                    )
                )
        for name in ("fit_smoothed_msd", "allow_partial_results"):
            if not isinstance(parameters.get(name), bool):
                diagnostics.append(
                    _diagnostic("ERROR", "parameter.%s" % name, "%s must be boolean" % name)
                )
        if parameters.get("fit_smoothed_msd") is True or parameters.get(
            "msd_smooth_window_points"
        ) is not None or parameters.get("msd_smooth_window_ps") is not None:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "parameter.custom_msd_smoothing_unsupported",
                    "formal transport passes the declared smoothed mode directly to pymatgen and "
                    "does not apply MLIPFlow moving-average smoothing",
                )
            )
        for name in (
            "lammps_data_name",
            "vasp_file_name",
            "msd_file_name",
            "msd_time_column",
            "msd_column",
            "fit_temperatures",
            "exclude_temperatures",
        ):
            value = parameters.get(name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "parameter.%s" % name, "%s must be a non-empty string" % name
                    )
                )
        return diagnostics

    def plan(self, context: Any) -> Dict[str, Any]:
        diagnostics = self.validate(context)
        if _has_errors(diagnostics):
            return _blocked(diagnostics)

        project_root = _resolve(context["project_root"], Path.cwd())
        attempt_dir = _resolve(context["attempt_dir"], project_root)
        inputs = _mapping(context["inputs"])
        parameters = _mapping(context["parameters"])
        resources = _mapping(context.get("resources"))
        assert project_root is not None and attempt_dir is not None
        if _operation_name(parameters) == _SMOKE_OPERATION:
            python_executable = str(resources.get("python_executable", sys.executable))
            md_script = _resolve(inputs["md_script"], project_root)
            analysis_script = _bundled_analysis_script()
            structure = _resolve(inputs["structure"], project_root)
            assert md_script is not None and analysis_script is not None and structure is not None
            raw_model = inputs.get("model", "default")
            if raw_model is None or str(raw_model).strip().lower() in {"", "none", "default"}:
                model = "default"
            else:
                resolved_model = _resolve(raw_model, project_root)
                assert resolved_model is not None
                model = str(resolved_model)
            md_output = (
                attempt_dir / str(parameters.get("md_output_subdir", "ionic-md-smoke"))
            ).resolve()
            output_dir = (
                attempt_dir / str(parameters.get("output_subdir", "ionic-transport-postprocess"))
            ).resolve()
            manifest_path = (
                attempt_dir
                / str(parameters.get("integration_manifest", _INTEGRATION_MANIFEST_NAME))
            ).resolve()
            frame_step_fs = float(parameters["timestep_fs"]) * int(
                parameters["trajectory_interval_steps"]
            )
            analysis_arguments = _analysis_arguments(
                parameters, ase_frame_step_fs=frame_step_fs
            )
            argv = [
                python_executable,
                str(_MODULE_PATH.with_name("md_smoke_handoff.py").resolve()),
                "--attempt-dir",
                str(attempt_dir),
                "--md-script",
                str(md_script),
                "--analysis-script",
                str(analysis_script),
                "--structure",
                str(structure),
                "--model",
                model,
                "--calculator",
                str(parameters["calculator"]),
                "--temperatures-json",
                json.dumps(parameters["temperatures_k"], separators=(",", ":")),
                "--seed",
                str(parameters["seed"]),
                "--timestep-fs",
                _stringify(parameters["timestep_fs"]),
                "--npt-time-ps",
                _stringify(parameters["npt_time_ps"]),
                "--nvt-equil-time-ps",
                _stringify(parameters["nvt_equil_time_ps"]),
                "--production-time-ps",
                _stringify(parameters["production_time_ps"]),
                "--trajectory-interval-steps",
                str(parameters["trajectory_interval_steps"]),
                "--device",
                str(parameters["device"]),
                "--default-dtype",
                str(parameters["default_dtype"]),
                "--input-index",
                str(parameters.get("input_index", "-1")),
                "--specie",
                str(parameters["specie"]),
                "--md-output",
                str(md_output),
                "--analysis-output",
                str(output_dir),
                "--manifest",
                str(manifest_path),
                "--analysis-arguments-json",
                json.dumps(analysis_arguments, separators=(",", ":")),
                "--python-executable",
                python_executable,
            ]
            if parameters.get("input_format") is not None:
                argv.extend(["--input-format", str(parameters["input_format"])])
            trajectory_paths = [
                md_output / ("T%d" % int(float(temperature))) / "production.traj"
                for temperature in parameters["temperatures_k"]
            ]
            metadata_paths = [path.parent / "metadata.json" for path in trajectory_paths]
            expected_outputs = [
                *[str(output_dir / name) for name in _REQUIRED_RESULT_NAMES],
                str(manifest_path),
                *[str(path) for path in trajectory_paths],
                *[str(path) for path in metadata_paths],
            ]
            return {
                "plugin_id": PLUGIN_ID,
                "operation": _SMOKE_OPERATION,
                "status": "READY",
                "executable": True,
                "argv": argv,
                "cwd": str(attempt_dir),
                "shell": False,
                "expected_outputs": expected_outputs,
                "output_dir": str(output_dir),
                "md_output_dir": str(md_output),
                "integration_manifest": str(manifest_path),
                "trajectory_paths": [str(path) for path in trajectory_paths],
                "assumptions": {
                    "scientific_use": "integration-smoke-only",
                    "transport_values": "pymatgen-diffusion-analyzer",
                    "velocity_seed": int(parameters["seed"]),
                    "framework_rngs_fully_controlled": False,
                    "gpu_bitwise_determinism_guaranteed": False,
                    "production_scale_parameters_accepted": False,
                },
                "provenance": {
                    "md_source_sha256": _sha256_file(md_script),
                    "analysis_source_sha256": _sha256_file(analysis_script),
                    "structure_sha256": _sha256_file(structure),
                    "model_sha256": None if model == "default" else _sha256_file(Path(model)),
                },
                "diagnostics": diagnostics,
            }
        script = _bundled_analysis_script()
        input_paths = [_resolve(value, project_root) for value in inputs["input_paths"]]
        output_dir = (
            attempt_dir / str(parameters.get("output_subdir", "ionic-transport-postprocess"))
        ).resolve()

        argv = [str(resources.get("python_executable", sys.executable)), str(script)]
        for path in input_paths:
            argv.extend(["--input", str(path)])
        argv.extend(["--output", str(output_dir)])

        argv.extend(_analysis_arguments(parameters))
        if structure_input := _resolve(inputs.get("structure"), project_root):
            argv.extend(["--msd-structure", str(structure_input)])

        return {
            "plugin_id": PLUGIN_ID,
            "operation": _ANALYZE_OPERATION,
            "status": "READY",
            "executable": True,
            "argv": argv,
            "cwd": str(attempt_dir),
            "shell": False,
            "expected_outputs": [str(output_dir / name) for name in _REQUIRED_RESULT_NAMES],
            "output_dir": str(output_dir),
            "assumptions": {
                "formal_scientific_implementation": "pymatgen-analysis-diffusion",
                "trajectory_haven_ratio": "reported-directly-by-DiffusionAnalyzer",
                "conductivity_method": "pymatgen-public-api-or-unavailable",
                "seed": None,
            },
            "provenance": {
                "analysis_source_sha256": _sha256_file(script),
            },
            "diagnostics": diagnostics,
        }

    def prepare(self, context: Any, plan: Any) -> Dict[str, Any]:
        diagnostics = self.validate(context)
        if (
            _has_errors(diagnostics)
            or not isinstance(plan, Mapping)
            or plan.get("status") != "READY"
        ):
            if not isinstance(plan, Mapping) or plan.get("status") != "READY":
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "plan.ready_required", "prepare requires a READY execution plan"
                    )
                )
            return _blocked(diagnostics)
        return {
            "plugin_id": PLUGIN_ID,
            "status": "READY",
            "executable": True,
            "prepared": False,
            "writes_files": False,
            "plan": dict(plan),
            "diagnostics": diagnostics,
        }

    def _result_paths(self, context: Any) -> Tuple[Optional[Path], Dict[str, Path]]:
        if not isinstance(context, Mapping):
            return None, {}
        project_root = _resolve(context.get("project_root"), Path.cwd()) or Path.cwd().resolve()
        inputs = _mapping(context.get("inputs"))
        explicit = inputs.get("result_files")
        paths: Dict[str, Path] = {}
        if isinstance(explicit, Mapping):
            for name in _REQUIRED_RESULT_NAMES:
                path = _resolve(explicit.get(name), project_root)
                if path is not None:
                    paths[name] = path
            if len(paths) == len(_REQUIRED_RESULT_NAMES):
                parents = {path.parent for path in paths.values()}
                return (next(iter(parents)) if len(parents) == 1 else None), paths

        execution = _mapping(context.get("execution"))
        plan = _mapping(execution.get("plan"))
        expected = plan.get("expected_outputs")
        if isinstance(expected, list) and len(expected) >= len(_REQUIRED_RESULT_NAMES):
            for name, value in zip(_REQUIRED_RESULT_NAMES, expected[: len(_REQUIRED_RESULT_NAMES)]):
                path = _resolve(value, project_root)
                if path is not None:
                    paths[name] = path
            output_dir = _resolve(plan.get("output_dir"), project_root)
            if output_dir is not None and len(paths) == len(_REQUIRED_RESULT_NAMES):
                return output_dir, paths
        return None, {}

    def _read_results(
        self, context: Any
    ) -> Tuple[
        Optional[Path],
        List[Dict[str, str]],
        Dict[str, Any],
        List[Dict[str, Any]],
        List[Dict[str, str]],
    ]:
        output_dir, paths = self._result_paths(context)
        diagnostics: List[Dict[str, str]] = []
        if output_dir is None or len(paths) != len(_REQUIRED_RESULT_NAMES):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "result.files_explicit_required",
                    "result files must come from execution.plan.expected_outputs or inputs.result_files",
                )
            )
            return None, [], {}, [], diagnostics
        for name, path in paths.items():
            if not _is_within(path, output_dir):
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "result.path_escape",
                        "%s is outside the explicit output directory" % name,
                    )
                )
            elif not path.is_file():
                diagnostics.append(
                    _diagnostic("INFO", "result.missing", "%s does not exist yet" % name)
                )
        if any(item["code"] == "result.missing" for item in diagnostics):
            return output_dir, [], {}, [], diagnostics
        if _has_errors(diagnostics):
            return output_dir, [], {}, [], diagnostics

        try:
            reader = csv.DictReader(
                io.StringIO(
                    paths["diffusion_results_by_temperature.csv"].read_text(encoding="utf-8")
                )
            )
            rows = [dict(row) for row in reader]
            summary = json.loads(paths["arrhenius_summary.json"].read_text(encoding="utf-8"))
            failures = json.loads(paths["postprocess_failures.json"].read_text(encoding="utf-8"))
        except (OSError, UnicodeError, csv.Error, json.JSONDecodeError) as exc:
            diagnostics.append(
                _diagnostic(
                    "ERROR", "result.unreadable", "cannot parse transport results: %s" % exc
                )
            )
            return output_dir, [], {}, [], diagnostics
        if reader.fieldnames is None or not _RESULT_COLUMNS.issubset(set(reader.fieldnames)):
            diagnostics.append(
                _diagnostic(
                    "ERROR", "result.columns", "diffusion results CSV has an unexpected header"
                )
            )
        if not rows:
            diagnostics.append(
                _diagnostic(
                    "ERROR", "result.no_rows", "diffusion results CSV contains no successful runs"
                )
            )
        if not isinstance(summary, dict):
            diagnostics.append(
                _diagnostic(
                    "ERROR", "result.summary_mapping", "arrhenius_summary.json must be an object"
                )
            )
            summary = {}
        if not isinstance(failures, list) or not all(isinstance(item, dict) for item in failures):
            diagnostics.append(
                _diagnostic(
                    "ERROR", "result.failures_list", "postprocess_failures.json must be a list"
                )
            )
            failures = []
        for index, row in enumerate(rows):
            for name in (
                "temperature_K",
                "diffusivity_cm2_s",
                "analysis_start_ps",
                "analysis_end_ps",
            ):
                try:
                    number = float(row.get(name, ""))
                except (TypeError, ValueError):
                    number = float("nan")
                if not math.isfinite(number):
                    diagnostics.append(
                        _diagnostic(
                            "ERROR",
                            "result.nonfinite",
                            "row %d has non-finite %s" % (index + 1, name),
                        )
                    )
                    break
            else:
                if float(row["temperature_K"]) <= 0 or float(row["diffusivity_cm2_s"]) <= 0:
                    diagnostics.append(
                        _diagnostic(
                            "ERROR",
                            "result.nonpositive_transport",
                            "temperature and diffusivity must be positive in row %d" % (index + 1),
                        )
                    )
                if float(row["analysis_start_ps"]) >= float(row["analysis_end_ps"]):
                    diagnostics.append(
                        _diagnostic(
                            "ERROR",
                            "result.fit_window",
                            "result analysis window is invalid in row %d" % (index + 1),
                        )
                    )
            conductivity_method = row.get("conductivity_method")
            conductivity = _optional_csv_number(row.get("conductivity_NE_mS_cm"))
            if conductivity_method == "unavailable":
                if conductivity is not None or not row.get("conductivity_unavailable_reason"):
                    diagnostics.append(
                        _diagnostic(
                            "ERROR",
                            "result.conductivity_unavailable",
                            "row %d must explain unavailable conductivity" % (index + 1),
                        )
                    )
            elif conductivity is None:
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "result.conductivity_missing",
                        "row %d must contain pymatgen conductivity" % (index + 1),
                    )
                )
            for name in ("msd_curve_csv", "msd_fit_html"):
                artifact = _resolve(row.get(name), output_dir)
                if (
                    artifact is None
                    or not _is_within(artifact, output_dir)
                    or not artifact.is_file()
                ):
                    diagnostics.append(
                        _diagnostic(
                            "ERROR",
                            "result.artifact_missing",
                            "row %d references a missing/out-of-tree %s" % (index + 1, name),
                        )
                    )
        return output_dir, rows, summary, failures, diagnostics

    def check(self, context: Any) -> Dict[str, Any]:
        execution = _mapping(context.get("execution")) if isinstance(context, Mapping) else {}
        returncode = execution.get("returncode")
        if returncode is not None and returncode != 0:
            return {
                "plugin_id": PLUGIN_ID,
                "status": "FAIL",
                "diagnostics": [
                    _diagnostic(
                        "ERROR", "execution.nonzero", "analysis process returned %s" % returncode
                    )
                ],
            }
        output_dir, rows, summary, failures, diagnostics = self._read_results(context)
        if output_dir is not None and any(item["code"] == "result.missing" for item in diagnostics):
            return {"plugin_id": PLUGIN_ID, "status": "WAIT", "diagnostics": diagnostics}
        parameters = _mapping(context.get("parameters")) if isinstance(context, Mapping) else {}
        operation = _operation_name(parameters)
        if output_dir is not None and isinstance(context, Mapping):
            diagnostics.extend(
                _verify_analysis_manifest(
                    context, output_dir, output_dir / "analysis_manifest.json"
                )
            )
            diagnostics.extend(_verify_transport_rows(rows, output_dir))
            diagnostics.extend(_verify_arrhenius_summary(summary))
        integration_manifest: Optional[Dict[str, Any]] = None
        if operation == _SMOKE_OPERATION and isinstance(context, Mapping):
            integration_manifest, _, integration_diagnostics = _verify_smoke_manifest(context)
            diagnostics.extend(integration_diagnostics)
            if any(item["code"] == "integration.manifest_missing" for item in diagnostics):
                return {"plugin_id": PLUGIN_ID, "status": "WAIT", "diagnostics": diagnostics}
            if not isinstance(summary.get("single"), Mapping) or summary.get(
                "arrhenius_fit_skipped"
            ) is True:
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "integration.arrhenius_incomplete",
                        "md smoke must complete the trajectory-to-Arrhenius handoff",
                    )
                )
            else:
                activation = summary["single"].get("Ea_eV")
                if not _finite_number(activation):
                    diagnostics.append(
                        _diagnostic(
                            "ERROR",
                            "integration.activation_energy",
                            "smoke Arrhenius result requires finite single.Ea_eV",
                        )
                    )
            expected_temperatures = sorted(float(value) for value in parameters["temperatures_k"])
            actual_temperatures = sorted(float(row["temperature_K"]) for row in rows)
            if actual_temperatures != expected_temperatures:
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "integration.temperature_coverage",
                        "transport rows must cover every approved smoke temperature exactly once",
                    )
                )
        if failures and parameters.get("allow_partial_results") is not True:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "result.partial_disallowed",
                    "%d input run(s) failed; set allow_partial_results=true only after review"
                    % len(failures),
                )
            )
        elif failures:
            diagnostics.append(
                _diagnostic(
                    "WARNING",
                    "result.partial_allowed",
                    "%d input run(s) were skipped" % len(failures),
                )
            )
        if _has_errors(diagnostics):
            return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
        return {
            "plugin_id": PLUGIN_ID,
            "operation": operation,
            "status": "OK",
            "run_count": len(rows),
            "failed_run_count": len(failures),
            "scientific_use": (
                integration_manifest.get("scientific_use")
                if integration_manifest is not None
                else "post-processing"
            ),
            "diagnostics": diagnostics,
        }

    def collect(self, context: Any) -> Dict[str, Any]:
        checked = self.check(context)
        if checked.get("status") != "OK":
            return {
                "plugin_id": PLUGIN_ID,
                "status": checked.get("status", "FAIL"),
                "artifacts": [],
                "metrics": {},
                "diagnostics": checked.get("diagnostics", []),
            }
        output_dir, rows, summary, failures, diagnostics = self._read_results(context)
        assert output_dir is not None
        _, paths = self._result_paths(context)
        artifacts: List[Dict[str, str]] = [
            {
                "role": "transport-results",
                "path": str(paths["diffusion_results_by_temperature.csv"]),
                "media_type": "text/csv",
            },
            {
                "role": "arrhenius-summary",
                "path": str(paths["arrhenius_summary.json"]),
                "media_type": "application/json",
            },
            {
                "role": "postprocess-failures",
                "path": str(paths["postprocess_failures.json"]),
                "media_type": "application/json",
            },
            {
                "role": "analysis-manifest",
                "path": str(paths["analysis_manifest.json"]),
                "media_type": "application/json",
            },
        ]
        for name in _OPTIONAL_RESULT_NAMES:
            path = output_dir / name
            if path.is_file():
                artifacts.append(
                    {"role": "transport-diagnostic", "path": str(path), "media_type": "text/html"}
                )
        seen = {item["path"] for item in artifacts}
        for row in rows:
            for field, role, media_type in (
                ("msd_curve_csv", "msd-curve", "text/csv"),
                ("msd_fit_html", "msd-fit", "text/html"),
            ):
                path = _resolve(row[field], output_dir)
                assert path is not None
                if str(path) not in seen:
                    artifacts.append({"role": role, "path": str(path), "media_type": media_type})
                    seen.add(str(path))

        parameters = _mapping(context.get("parameters")) if isinstance(context, Mapping) else {}
        operation = _operation_name(parameters)
        integration_manifest: Optional[Dict[str, Any]] = None
        if operation == _SMOKE_OPERATION and isinstance(context, Mapping):
            integration_manifest, integration_artifacts, integration_diagnostics = (
                _verify_smoke_manifest(context)
            )
            diagnostics.extend(integration_diagnostics)
            for artifact in integration_artifacts:
                if artifact["path"] not in seen:
                    artifacts.append(artifact)
                    seen.add(artifact["path"])

        by_temperature = []
        for row in rows:
            by_temperature.append(
                {
                    "temperature_K": float(row["temperature_K"]),
                    "diffusivity_cm2_s": float(row["diffusivity_cm2_s"]),
                    "diffusivity_std_dev_cm2_s": _optional_csv_number(
                        row.get("diffusivity_std_dev_cm2_s")
                    ),
                    "conductivity_NE_mS_cm": _optional_csv_number(
                        row.get("conductivity_NE_mS_cm")
                    ),
                    "conductivity_std_dev_mS_cm": _optional_csv_number(
                        row.get("conductivity_std_dev_mS_cm")
                    ),
                    "chg_diffusivity_cm2_s": _optional_csv_number(
                        row.get("chg_diffusivity_cm2_s")
                    ),
                    "chg_conductivity_mS_cm": _optional_csv_number(
                        row.get("chg_conductivity_mS_cm")
                    ),
                    "haven_ratio": _optional_csv_number(row.get("haven_ratio")),
                    "analysis_start_ps": float(row["analysis_start_ps"]),
                    "analysis_end_ps": float(row["analysis_end_ps"]),
                    "diffusion_method": row["diffusion_method"],
                    "conductivity_method": row["conductivity_method"],
                }
            )
        return {
            "plugin_id": PLUGIN_ID,
            "operation": operation,
            "status": "OK",
            "artifacts": artifacts,
            "metrics": {
                "run_count": len(rows),
                "failed_run_count": len(failures),
                "temperature_count": len({item["temperature_K"] for item in by_temperature}),
                "by_temperature": by_temperature,
                "arrhenius": summary,
                "assumptions": {
                    "formal_scientific_implementation": "pymatgen-analysis-diffusion",
                    "scientific_use": (
                        integration_manifest.get("scientific_use")
                        if integration_manifest is not None
                        else "post-processing"
                    ),
                    "scientific_parity_claimed": False if integration_manifest else None,
                },
            },
            "diagnostics": diagnostics,
        }

    def replay(self, context: Any) -> Dict[str, Any]:
        if not isinstance(context, Mapping) or not isinstance(
            context.get("result_manifest"), Mapping
        ):
            return {
                "plugin_id": PLUGIN_ID,
                "operation": "replay",
                "status": "BLOCKED",
                "executable": False,
                "code": "replay.result_manifest_required",
                "diagnostics": [
                    _diagnostic(
                        "ERROR",
                        "replay.result_manifest_required",
                        "replay requires an in-memory result manifest",
                    )
                ],
            }
        return {
            "plugin_id": PLUGIN_ID,
            "operation": "replay",
            "status": "OK",
            "executable": False,
            "code": "replay.reference_only",
            "result_manifest": context["result_manifest"],
        }


def _parse_temperature_paths(values: Sequence[str]) -> Dict[float, str]:
    result: Dict[float, str] = {}
    for value in values:
        if "=" not in value:
            raise ManuscriptTransportError("--run must use TEMPERATURE=PATH")
        raw_temperature, raw_path = value.split("=", 1)
        try:
            temperature = float(raw_temperature)
        except ValueError as exc:
            raise ManuscriptTransportError("--run temperature must be numeric") from exc
        if temperature in result:
            raise ManuscriptTransportError("--run repeats temperature %.17g" % temperature)
        if not raw_path.strip():
            raise ManuscriptTransportError("--run path must be non-empty")
        result[temperature] = raw_path
    return result


def _emit_parity_json(result: Mapping[str, Any], output: Optional[str]) -> None:
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output is None:
        sys.stdout.write(rendered)
        return
    path = Path(output).expanduser().resolve()
    if not path.parent.is_dir():
        raise ManuscriptTransportError("output parent directory does not exist: %s" % path.parent)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    except FileExistsError as exc:
        raise ManuscriptTransportError(
            "output already exists; parity wrapper never overwrites: %s" % path
        ) from exc


def manuscript_transport_main(argv: Optional[Sequence[str]] = None) -> int:
    """Plugin-local read-only evidence wrapper; this is not a top-level MLIPFlow CLI."""

    parser = argparse.ArgumentParser(
        description="Read-only parser/parity wrapper for historical Li10 transport evidence."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    history_parser = subparsers.add_parser(
        "parse-historical-output", help="normalize Li10_test.out or Li10_MACE.out"
    )
    history_parser.add_argument("source")
    history_parser.add_argument("--output")

    reproduce_parser = subparsers.add_parser(
        "reproduce-msd", help="post-process existing target.msd files; never run MD"
    )
    reproduce_parser.add_argument(
        "--run", action="append", required=True, help="repeat TEMPERATURE=PATH"
    )
    reproduce_parser.add_argument(
        "--sampling-profile", choices=sorted(_SAMPLING_PROFILES), required=True
    )
    reproduce_parser.add_argument(
        "--carrier-convention",
        choices=[LEGACY_SCRIPT_CONVENTION, COMPOSITION_CORRECTED_CONVENTION],
        required=True,
    )
    reproduce_parser.add_argument(
        "--arrhenius-convention",
        choices=[EXACT_ARRHENIUS_CONVENTION, LEGACY_ARRHENIUS_CONVENTION],
        required=True,
    )
    reproduce_parser.add_argument("--target-temperature-K", type=float, default=300.0)
    reproduce_parser.add_argument("--n-mobile-ions", type=int)
    reproduce_parser.add_argument("--volume-A3", type=float)
    reproduce_parser.add_argument("--structure")
    reproduce_parser.add_argument("--mobile-species", default="Li")
    reproduce_parser.add_argument("--legacy-script")
    reproduce_parser.add_argument("--historical-output")
    reproduce_parser.add_argument("--historical-dataset")
    reproduce_parser.add_argument(
        "--historical-carrier-convention",
        choices=[LEGACY_SCRIPT_CONVENTION, "same_as_reproduction", "unknown"],
    )
    reproduce_parser.add_argument("--output")

    arguments = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if arguments.command == "parse-historical-output":
            result = parse_historical_transport_output(arguments.source)
            _emit_parity_json(result, arguments.output)
            return 0
        target_paths = _parse_temperature_paths(arguments.run)
        result = reproduce_manuscript_transport(
            target_paths,
            sampling_profile=arguments.sampling_profile,
            carrier_convention=arguments.carrier_convention,
            arrhenius_convention=arguments.arrhenius_convention,
            target_temperature_k=arguments.target_temperature_K,
            n_mobile_ions=arguments.n_mobile_ions,
            volume_a3=arguments.volume_A3,
            structure_path=arguments.structure,
            mobile_species=arguments.mobile_species,
            legacy_script_path=arguments.legacy_script,
            historical_output_path=arguments.historical_output,
            historical_dataset=arguments.historical_dataset,
            historical_carrier_convention=arguments.historical_carrier_convention,
        )
        _emit_parity_json(result, arguments.output)
        return 0
    except ManuscriptTransportError as exc:
        parser.error(str(exc))
    return 2  # pragma: no cover - argparse.error raises SystemExit


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess smoke tests
    raise SystemExit(manuscript_transport_main())
