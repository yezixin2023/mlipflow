"""ionic transport: md handoff."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from . import contracts


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
            contracts._diagnostic(
                "ERROR",
                "input.analysis_script_unsupported",
                "analysis_script is packaged by MLIPFlow and cannot be overridden",
            )
        )
    unknown = sorted(set(parameters) - contracts._SMOKE_PARAMETERS)
    if unknown:
        diagnostics.append(
            contracts._diagnostic(
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
        path = contracts._resolve(inputs.get(name), project_root)
        if path is None or not path.is_file():
            diagnostics.append(
                contracts._diagnostic("ERROR", "path.%s" % name, "inputs.%s must name %s" % (name, label))
            )
        else:
            resolved_inputs[name] = path
            if name.endswith("script") and path.suffix != ".py":
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR", "path.%s_suffix" % name, "inputs.%s must be a Python file" % name
                    )
                )

    analysis_script = contracts._bundled_analysis_script()
    if not analysis_script.is_file():
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "path.bundled_analysis_missing",
                "the packaged ionic_conductivity.py runner is missing",
            )
        )

    calculator = parameters.get("calculator")
    if calculator not in contracts._SMOKE_CALCULATORS:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "parameter.calculator",
                "calculator must be one of %s" % sorted(contracts._SMOKE_CALCULATORS),
            )
        )
    raw_model = inputs.get("model", "default")
    if raw_model is None or str(raw_model).strip().lower() in {"", "none", "default"}:
        model = "default"
    elif isinstance(raw_model, (str, Path)):
        model_path = contracts._resolve(raw_model, project_root)
        if model_path is None or not model_path.is_file():
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "path.model", "an explicit inputs.model must name a local model file"
                )
            )
        else:
            resolved_inputs["model"] = model_path
        model = str(raw_model)
    else:
        model = ""
        diagnostics.append(
            contracts._diagnostic("ERROR", "path.model", "inputs.model must be 'default' or a file path")
        )
    if calculator in {"mace", "chgnet", "m3gnet", "matgl"} and model == "default":
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "parameter.default_model_network",
                "%s requires an explicit local model; default loading may use packaged or network state"
                % calculator,
            )
        )
    if calculator in {"emt", "lj"} and model != "default":
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "parameter.model_unused",
                "%s ignores model paths; use inputs.model='default'" % calculator,
            )
        )

    output_values = {
        "analysis": parameters.get("output_subdir", "ionic-transport-postprocess"),
        "md": parameters.get("md_output_subdir", "ionic-md-smoke"),
        "manifest": parameters.get("integration_manifest", contracts._INTEGRATION_MANIFEST_NAME),
    }
    output_paths: Dict[str, Path] = {}
    for name, value in output_values.items():
        if not contracts._safe_subdirectory(value):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "path.%s_output" % name,
                    "%s output must be a non-empty attempt-relative path without '..'" % name,
                )
            )
            continue
        path = (attempt_dir / str(value)).resolve()
        output_paths[name] = path
        if not contracts._is_within(path, attempt_dir):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "path.%s_escape" % name, "%s output escapes attempt_dir" % name
                )
            )
        elif path.exists():
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "path.%s_exists" % name,
                    "%s output already exists; use a fresh attempt" % name,
                )
            )
    if len(set(output_paths.values())) != len(output_paths):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "path.smoke_output_collision",
                "MD, analysis, and manifest paths must differ",
            )
        )
    output_items = list(output_paths.items())
    for index, (left_name, left_path) in enumerate(output_items):
        for right_name, right_path in output_items[index + 1 :]:
            if contracts._is_within(left_path, right_path) or contracts._is_within(right_path, left_path):
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        "path.smoke_output_overlap",
                        "%s and %s outputs must not contain one another" % (left_name, right_name),
                    )
                )
    for output in output_paths.values():
        for source_path in resolved_inputs.values():
            if contracts._is_within(output, source_path) or contracts._is_within(source_path, output):
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        "path.input_output_overlap",
                        "source/input and smoke output paths overlap",
                    )
                )
                break

    temperatures = parameters.get("temperatures_k")
    normalized_temperatures: List[float] = []
    if not isinstance(temperatures, list) or not (
        2 <= len(temperatures) <= contracts._SMOKE_MAX_TEMPERATURES
    ):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "parameter.temperatures_k",
                "temperatures_k must contain 2-%d temperatures for an Arrhenius smoke"
                % contracts._SMOKE_MAX_TEMPERATURES,
            )
        )
    else:
        for value in temperatures:
            if not contracts._positive_number(value) or not float(value).is_integer():
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        "parameter.temperatures_k",
                        "historical T<int> directories require positive integer-K temperatures",
                    )
                )
                break
            normalized_temperatures.append(float(value))
        if len(set(normalized_temperatures)) != len(normalized_temperatures):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "parameter.temperature_duplicate", "temperatures_k must be unique"
                )
            )

    seed = parameters.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        diagnostics.append(
            contracts._diagnostic("ERROR", "parameter.seed", "seed must be an explicit non-negative integer")
        )
    timestep = parameters.get("timestep_fs")
    if not contracts._positive_number(timestep) or float(timestep) > 2.0:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "parameter.timestep_fs",
                "timestep_fs must be positive and no greater than 2 fs",
            )
        )
    durations = (
        ("npt_time_ps", 0.0, contracts._SMOKE_MAX_EQUILIBRATION_PS),
        ("nvt_equil_time_ps", 0.0, contracts._SMOKE_MAX_EQUILIBRATION_PS),
        ("production_time_ps", 0.0, contracts._SMOKE_MAX_PRODUCTION_PS),
    )
    for name, minimum, maximum in durations:
        value = parameters.get(name)
        valid = contracts._finite_number(value) and float(value) >= minimum and float(value) <= maximum
        if name == "production_time_ps":
            valid = valid and float(value) > 0 if contracts._finite_number(value) else False
        if not valid:
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "parameter.%s" % name,
                    "%s exceeds the integration-smoke bound [%.3g, %.3g] ps"
                    % (name, minimum, maximum),
                )
            )
    trajectory_interval = parameters.get("trajectory_interval_steps")
    if not contracts._positive_int(trajectory_interval) or int(trajectory_interval) > 100:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "parameter.trajectory_interval_steps",
                "trajectory_interval_steps must be an integer in [1, 100]",
            )
        )

    if (
        contracts._positive_number(timestep)
        and contracts._finite_number(parameters.get("npt_time_ps"))
        and contracts._finite_number(parameters.get("nvt_equil_time_ps"))
        and contracts._positive_number(parameters.get("production_time_ps"))
        and contracts._positive_int(trajectory_interval)
        and normalized_temperatures
    ):
        dt = float(timestep)
        npt_steps = int(round(float(parameters["npt_time_ps"]) * 1000.0 / dt))
        nvt_steps = int(round(float(parameters["nvt_equil_time_ps"]) * 1000.0 / dt))
        production_steps = int(round(float(parameters["production_time_ps"]) * 1000.0 / dt))
        frames = int(math.ceil(production_steps / int(trajectory_interval)))
        min_points = parameters.get("min_msd_fit_points")
        if production_steps < 1 or not contracts._positive_int(min_points) or frames < int(min_points):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "parameter.smoke_frames",
                    "production/trajectory settings must yield at least min_msd_fit_points frames",
                )
            )
        total_steps = len(normalized_temperatures) * (npt_steps + nvt_steps + production_steps)
        if total_steps > contracts._SMOKE_MAX_TOTAL_STEPS:
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "parameter.smoke_step_cap",
                    "bounded smoke permits at most %d total MD steps" % contracts._SMOKE_MAX_TOTAL_STEPS,
                )
            )

    device = parameters.get("device")
    if device not in contracts._SMOKE_DEVICES:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "parameter.device", "device must be one of %s" % sorted(contracts._SMOKE_DEVICES)
            )
        )
    elif device != "cpu":
        diagnostics.append(
            contracts._diagnostic(
                "WARNING",
                "randomness.gpu_not_bitwise",
                "GPU/MPS execution is not guaranteed bitwise deterministic by the historical source",
            )
        )
    if parameters.get("default_dtype") not in contracts._SMOKE_DTYPES:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "parameter.default_dtype", "default_dtype must be float32 or float64"
            )
        )
    for name in ("input_format", "input_index"):
        value = parameters.get(name)
        if name == "input_format" and value is None:
            continue
        if (
            not isinstance(value, (str, int))
            or not str(value).strip()
            or any(token in str(value) for token in ("\x00", "\n", "\r"))
        ):
            diagnostics.append(
                contracts._diagnostic("ERROR", "parameter.%s" % name, "%s must be a plain scalar" % name)
            )

    if parameters.get("source") != "trajectory":
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "parameter.source", "md-smoke-and-analyze requires source='trajectory'"
            )
        )
    specie = parameters.get("specie")
    if not isinstance(specie, str) or re.fullmatch(r"[A-Z][a-z]?", specie) is None:
        diagnostics.append(
            contracts._diagnostic("ERROR", "parameter.specie", "specie must be one element symbol")
        )
    contracts._validate_interval(
        parameters, "trajectory_start_ps", "trajectory_end_ps", diagnostics, required=False
    )
    if contracts._finite_number(parameters.get("production_time_ps")):
        production_time = float(parameters["production_time_ps"])
        for name in ("trajectory_end_ps",):
            value = parameters.get(name)
            if value is not None and contracts._finite_number(value) and float(value) > production_time:
                diagnostics.append(
                    contracts._diagnostic(
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
                contracts._diagnostic(
                    "ERROR", "parameter.%s" % name, "%s must be one of %s" % (name, sorted(allowed))
                )
            )
    if parameters.get("fit_scope") != "all":
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "parameter.fit_scope", "multi-temperature smoke requires fit_scope='all'"
            )
        )
    for name in (
        "diffusion_analyzer_min_obs",
        "diffusion_analyzer_avg_nsteps",
        "diffusion_analyzer_step_skip",
        "min_msd_fit_points",
    ):
        if not contracts._positive_int(parameters.get(name)):
            diagnostics.append(
                contracts._diagnostic("ERROR", "parameter.%s" % name, "%s must be a positive integer" % name)
            )
    if (
        contracts._positive_int(parameters.get("min_msd_fit_points"))
        and int(parameters["min_msd_fit_points"]) < 3
    ):
        diagnostics.append(
            contracts._diagnostic("ERROR", "parameter.min_msd_fit_points", "min_msd_fit_points must be >= 3")
        )
    if (
        not contracts._positive_int(parameters.get("min_segment_points"))
        or int(parameters.get("min_segment_points", 0)) < 3
    ):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "parameter.min_segment_points",
                "min_segment_points must be an integer >= 3",
            )
        )
    for name in ("piecewise_slope_change", "piecewise_bic_delta"):
        value = parameters.get(name)
        if value is not None and (not contracts._finite_number(value) or float(value) < 0):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "parameter.%s" % name, "%s must be finite and non-negative" % name
                )
            )
    if not contracts._positive_number(parameters.get("target_temperature_k")):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "parameter.target_temperature_k", "target_temperature_k must be positive"
            )
        )
    for name in ("msd_smooth_window_points",):
        value = parameters.get(name)
        if value is not None and not contracts._positive_int(value):
            diagnostics.append(
                contracts._diagnostic("ERROR", "parameter.%s" % name, "%s must be a positive integer" % name)
            )
    contracts._optional_positive(parameters, "msd_smooth_window_ps", diagnostics)
    if not isinstance(parameters.get("fit_smoothed_msd"), bool):
        diagnostics.append(
            contracts._diagnostic("ERROR", "parameter.fit_smoothed_msd", "fit_smoothed_msd must be boolean")
        )
    if parameters.get("allow_partial_results") is not False:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "parameter.allow_partial_results",
                "integration smoke requires allow_partial_results=false",
            )
        )
    if contracts._positive_number(timestep) and contracts._positive_int(trajectory_interval):
        derived_frame_step = float(timestep) * int(trajectory_interval)
        supplied_frame_step = parameters.get("ase_frame_step_fs")
        if supplied_frame_step is not None and (
            not contracts._positive_number(supplied_frame_step)
            or not math.isclose(
                float(supplied_frame_step), derived_frame_step, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "parameter.ase_frame_step_fs",
                    "ase_frame_step_fs must equal timestep_fs * trajectory_interval_steps",
                )
            )
    for name in ("fit_temperatures", "exclude_temperatures"):
        value = parameters.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            diagnostics.append(
                contracts._diagnostic("ERROR", "parameter.%s" % name, "%s must be a non-empty string" % name)
            )

    python_executable = resources.get("python_executable", sys.executable)
    if not isinstance(python_executable, str) or not python_executable.strip():
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "resource.python_executable",
                "python_executable must be a non-empty string",
            )
        )
    elif not contracts._is_current_python(python_executable):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "resource.python_executable",
                "formal transport must use the same local Python interpreter that runs MLIPFlow",
            )
        )
    else:
        diagnostics.extend(contracts._formal_dependency_diagnostics())
    return diagnostics


def _integration_manifest_path(context: Mapping[str, Any]) -> Optional[Path]:
    project_root = contracts._resolve(context.get("project_root"), Path.cwd()) or Path.cwd().resolve()
    execution = contracts._mapping(context.get("execution"))
    plan = contracts._mapping(execution.get("plan"))
    planned = contracts._resolve(plan.get("integration_manifest"), project_root)
    if planned is not None:
        return planned
    attempt_dir = contracts._resolve(context.get("attempt_dir"), project_root)
    if attempt_dir is None:
        return None
    parameters = contracts._mapping(context.get("parameters"))
    relative = parameters.get("integration_manifest", contracts._INTEGRATION_MANIFEST_NAME)
    if not contracts._safe_subdirectory(relative):
        return None
    return (attempt_dir / str(relative)).resolve()


def _verify_smoke_manifest(
    context: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Verify the bounded handoff manifest and its declared files."""

    diagnostics: List[Dict[str, str]] = []
    artifacts: List[Dict[str, str]] = []
    project_root = contracts._resolve(context.get("project_root"), Path.cwd()) or Path.cwd().resolve()
    attempt_dir = contracts._resolve(context.get("attempt_dir"), project_root)
    manifest_path = _integration_manifest_path(context)
    if attempt_dir is None or manifest_path is None or not contracts._is_within(manifest_path, attempt_dir):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "integration.manifest_path",
                "md-smoke integration manifest must be explicit and inside attempt_dir",
            )
        )
        return None, artifacts, diagnostics
    if not manifest_path.is_file():
        diagnostics.append(
            contracts._diagnostic("INFO", "integration.manifest_missing", "integration manifest is pending")
        )
        return None, artifacts, diagnostics
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        diagnostics.append(contracts._diagnostic("ERROR", "integration.manifest_unreadable", str(exc)))
        return None, artifacts, diagnostics
    if not isinstance(manifest, dict):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "integration.manifest_type", "integration manifest must be an object"
            )
        )
        return None, artifacts, diagnostics
    expected_scalars = {
        "schema_version": 1,
        "plugin_id": contracts.PLUGIN_ID,
        "operation": contracts._SMOKE_OPERATION,
        "status": "OK",
        "scientific_use": "integration-smoke-only",
    }
    for name, expected in expected_scalars.items():
        if manifest.get(name) != expected:
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "integration.%s" % name,
                    "integration manifest %s must equal %r" % (name, expected),
                )
            )
    boundaries = contracts._mapping(manifest.get("boundaries"))
    for name in (
        "production_parameters_accepted",
        "scientific_parity_claimed",
        "model_quality_validated",
        "network_access_authorized_by_handoff",
    ):
        if boundaries.get(name) is not False:
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "integration.boundary_%s" % name,
                    "integration boundary %s must be false" % name,
                )
            )
    execution_record = contracts._mapping(manifest.get("execution"))
    if (
        execution_record.get("analysis_shell") is not False
        or execution_record.get("analysis_returncode") != 0
    ):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "integration.analysis_execution",
                "analysis must finish with shell=false and returncode=0",
            )
        )
    if execution_record.get("absolute_paths_recorded") is not False:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "integration.path_disclosure",
                "integration manifest must redact machine-local absolute paths",
            )
        )

    parameters = contracts._mapping(context.get("parameters"))
    configuration = contracts._mapping(manifest.get("configuration"))
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
                contracts._diagnostic(
                    "ERROR",
                    "integration.configuration_%s" % name,
                    "integration configuration %s does not match the approved plan" % name,
                )
            )

    project_root = contracts._resolve(context.get("project_root"), Path.cwd()) or Path.cwd().resolve()
    inputs = contracts._mapping(context.get("inputs"))
    expected_source_paths = {
        "historical-ase-md-source": contracts._resolve(inputs.get("md_script"), project_root),
        "mlipflow-transport-analysis-source": contracts._bundled_analysis_script(),
        "mlipflow-handoff-wrapper": contracts._MODULE_PATH.with_name("md_smoke_handoff.py").resolve(),
    }
    expected_input_paths = {
        "structure": contracts._resolve(inputs.get("structure"), project_root),
    }
    raw_model = inputs.get("model", "default")
    if raw_model is not None and str(raw_model).strip().lower() not in {
        "",
        "none",
        "default",
    }:
        expected_input_paths["model"] = contracts._resolve(raw_model, project_root)
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
                contracts._diagnostic(
                    "ERROR",
                    "integration.%s" % group_name,
                    "%s must be a non-empty list" % group_name,
                )
            )
            continue
        verified: List[Path] = []
        seen_roles: set[str] = set()
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        "integration.artifact_record",
                        "%s[%d] must be an object" % (group_name, index),
                    )
                )
                continue
            role = record.get("role")
            if not isinstance(role, str) or not role or role in seen_roles:
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        "integration.artifact_role",
                        "%s[%d] must have a unique non-empty role" % (group_name, index),
                    )
                )
                continue
            seen_roles.add(role)
            locator = record.get("path")
            if not isinstance(locator, str) or not locator:
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR", "integration.artifact_path", "artifact path must be non-empty"
                    )
                )
                continue
            expected_external = expected_external_by_group.get(group_name)
            if expected_external is not None:
                path = expected_external.get(role)
                if path is None or Path(locator).expanduser().resolve() != path.resolve():
                    diagnostics.append(
                        contracts._diagnostic(
                            "ERROR",
                            "integration.external_path",
                            "%s[%d] differs from the selected input path" % (group_name, index),
                        )
                    )
                    continue
                path = path.resolve()
            else:
                candidate = Path(locator).expanduser()
                path = (
                    (attempt_dir / candidate).resolve()
                    if not candidate.is_absolute()
                    else candidate.resolve()
                )
            if confined and not contracts._is_within(path, attempt_dir):
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        "integration.artifact_escape",
                        "%s[%d] must be attempt-relative and confined" % (group_name, index),
                    )
                )
                continue
            if not path.is_file():
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        "integration.artifact_missing",
                        "recorded artifact is missing: %s" % path,
                    )
                )
                continue
            verified.append(path)
        expected_external = expected_external_by_group.get(group_name)
        if expected_external is not None and seen_roles != set(expected_external):
            diagnostics.append(
                contracts._diagnostic(
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
            contracts._diagnostic(
                "ERROR",
                "integration.trajectory_count",
                "one production trajectory is required per temperature",
            )
        )
    result_names = {path.name for path in verified_by_group.get("result_artifacts", [])}
    missing_results = set(contracts._REQUIRED_RESULT_NAMES) - result_names
    if missing_results:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "integration.result_artifacts",
                "integration manifest omits required result files: %s" % sorted(missing_results),
            )
        )
    randomness = contracts._mapping(manifest.get("randomness"))
    if (
        randomness.get("framework_rngs_fully_controlled") is not False
        or randomness.get("gpu_bitwise_determinism_guaranteed") is not False
    ):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "integration.randomness_boundary",
                "manifest must not claim full framework/GPU determinism",
            )
        )
    artifacts.append(
        {
            "role": "md-integration-manifest",
            "path": str(manifest_path),
            "media_type": "application/json",
        }
    )
    for path in trajectories:
        artifacts.append(
            {
                "role": "production-trajectory",
                "path": str(path),
                "media_type": "application/octet-stream",
            }
        )
    return manifest, artifacts, diagnostics


def plan_md_handoff(context, diagnostics, project_root, attempt_dir, inputs, parameters, resources):
    python_executable = str(resources.get("python_executable", sys.executable))
    md_script = contracts._resolve(inputs["md_script"], project_root)
    analysis_script = contracts._bundled_analysis_script()
    structure = contracts._resolve(inputs["structure"], project_root)
    assert md_script is not None and analysis_script is not None and structure is not None
    raw_model = inputs.get("model", "default")
    if raw_model is None or str(raw_model).strip().lower() in {"", "none", "default"}:
        model = "default"
    else:
        resolved_model = contracts._resolve(raw_model, project_root)
        assert resolved_model is not None
        model = str(resolved_model)
    md_output = (attempt_dir / str(parameters.get("md_output_subdir", "ionic-md-smoke"))).resolve()
    output_dir = (
        attempt_dir / str(parameters.get("output_subdir", "ionic-transport-postprocess"))
    ).resolve()
    manifest_path = (
        attempt_dir / str(parameters.get("integration_manifest", contracts._INTEGRATION_MANIFEST_NAME))
    ).resolve()
    frame_step_fs = float(parameters["timestep_fs"]) * int(parameters["trajectory_interval_steps"])
    analysis_arguments = contracts._analysis_arguments(parameters, ase_frame_step_fs=frame_step_fs)
    argv = [
        python_executable,
        str(contracts._MODULE_PATH.with_name("md_smoke_handoff.py").resolve()),
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
        contracts._stringify(parameters["timestep_fs"]),
        "--npt-time-ps",
        contracts._stringify(parameters["npt_time_ps"]),
        "--nvt-equil-time-ps",
        contracts._stringify(parameters["nvt_equil_time_ps"]),
        "--production-time-ps",
        contracts._stringify(parameters["production_time_ps"]),
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
        *[str(output_dir / name) for name in contracts._REQUIRED_RESULT_NAMES],
        str(manifest_path),
        *[str(path) for path in trajectory_paths],
        *[str(path) for path in metadata_paths],
    ]
    return {
        "plugin_id": contracts.PLUGIN_ID,
        "operation": contracts._SMOKE_OPERATION,
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
            "md_source_path": str(md_script),
            "analysis_source_path": str(analysis_script),
            "structure_path": str(structure),
            "model_path": None if model == "default" else str(model),
        },
        "diagnostics": diagnostics,
    }
