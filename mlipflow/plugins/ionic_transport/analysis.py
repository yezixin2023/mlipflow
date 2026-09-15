"""ionic transport: analysis."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import contracts, md_handoff


def validate(context: Any) -> List[Dict[str, str]]:
    diagnostics: List[Dict[str, str]] = []
    if not isinstance(context, Mapping):
        return [contracts._diagnostic("ERROR", "context.mapping_required", "context must be a mapping")]

    project_root = contracts._resolve(context.get("project_root"), Path.cwd())
    if project_root is None or not project_root.is_dir():
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "path.project_root", "project_root must name an existing directory"
            )
        )
        project_root = Path.cwd().resolve()
    attempt_dir = contracts._resolve(context.get("attempt_dir"), project_root)
    if attempt_dir is None:
        diagnostics.append(
            contracts._diagnostic("ERROR", "path.attempt_dir", "attempt_dir must be an explicit path")
        )
        attempt_dir = project_root / ".mlipflow-invalid-attempt"
    elif not contracts._is_within(attempt_dir, project_root):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "path.attempt_outside_project",
                "attempt_dir must stay inside project_root",
            )
        )
    if context.get("backend", "local") != "local":
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "backend.local_only",
                "analysis is exposed as local argv only; scheduler wrapping belongs to the core",
            )
        )

    inputs = contracts._mapping(context.get("inputs"))
    parameters = contracts._mapping(context.get("parameters"))
    resources = contracts._mapping(context.get("resources"))
    if not isinstance(context.get("inputs", {}), Mapping):
        diagnostics.append(
            contracts._diagnostic("ERROR", "inputs.mapping_required", "inputs must be a mapping")
        )
    if not isinstance(context.get("parameters", {}), Mapping):
        diagnostics.append(
            contracts._diagnostic("ERROR", "parameters.mapping_required", "parameters must be a mapping")
        )
    if not isinstance(context.get("resources", {}), Mapping):
        diagnostics.append(
            contracts._diagnostic("ERROR", "resources.mapping_required", "resources must be a mapping")
        )

    operation = contracts._operation_name(parameters)
    if operation not in {contracts._ANALYZE_OPERATION, contracts._SMOKE_OPERATION}:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "parameter.operation",
                "operation must be analyze-existing or md-smoke-and-analyze",
            )
        )
        return diagnostics
    if operation == contracts._SMOKE_OPERATION:
        diagnostics.extend(
            md_handoff._validate_md_smoke(
                project_root=project_root,
                attempt_dir=attempt_dir,
                inputs=inputs,
                parameters=parameters,
                resources=resources,
            )
        )
        return diagnostics

    unknown_parameters = sorted(set(parameters) - contracts._ANALYZE_PARAMETERS)
    if unknown_parameters:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "parameter.unknown",
                "unsupported analyze-existing parameter(s): %s" % ", ".join(unknown_parameters),
            )
        )

    if "analysis_script" in inputs:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "input.analysis_script_unsupported",
                "analysis_script is packaged by MLIPFlow and cannot be overridden",
            )
        )

    script = contracts._bundled_analysis_script()
    if not script.is_file():
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "path.bundled_analysis_missing",
                "the packaged ionic_conductivity.py runner is missing",
            )
        )

    raw_paths = inputs.get("input_paths")
    input_paths: List[Path] = []
    if not isinstance(raw_paths, list) or not raw_paths:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "path.input_paths", "inputs.input_paths must be a non-empty list"
            )
        )
    else:
        for index, raw_path in enumerate(raw_paths):
            resolved = contracts._resolve(raw_path, project_root)
            if resolved is None or not resolved.exists():
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        "path.input",
                        "input_paths[%d] must name an existing file or directory" % index,
                    )
                )
            else:
                input_paths.append(resolved)
    structure_input = contracts._resolve(inputs.get("structure"), project_root)
    if inputs.get("structure") is not None and (
        structure_input is None or not structure_input.is_file()
    ):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "path.structure",
                "inputs.structure must name a real structure file for MSD conductivity",
            )
        )

    output_subdir = parameters.get("output_subdir", "ionic-transport-postprocess")
    if not contracts._safe_subdirectory(output_subdir):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "path.output_subdir",
                "output_subdir must be a non-empty relative path without '..'",
            )
        )
        output_dir = attempt_dir / "ionic-transport-postprocess"
    else:
        output_dir = (attempt_dir / str(output_subdir)).resolve()
        if not contracts._is_within(output_dir, attempt_dir):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "path.output_escape",
                    "analysis output must stay inside attempt_dir",
                )
            )
    if output_dir.exists():
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "path.output_exists",
                "source CLI overwrites result files; use a fresh attempt/output_subdir",
            )
        )
    for input_path in input_paths:
        if contracts._is_within(output_dir, input_path) or contracts._is_within(input_path, output_dir):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "path.input_output_overlap",
                    "input and output paths must not overlap",
                )
            )
            break
    if structure_input is not None and (
        contracts._is_within(output_dir, structure_input) or contracts._is_within(structure_input, output_dir)
    ):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "path.input_output_overlap",
                "structure and output paths must not overlap",
            )
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

    source = parameters.get("source")
    if source not in {"auto", "trajectory", "vasp", "msd"}:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "parameter.source", "source must be auto, trajectory, vasp, or msd"
            )
        )
    specie = parameters.get("specie")
    if not isinstance(specie, str) or re.fullmatch(r"[A-Z][a-z]?", specie) is None:
        diagnostics.append(
            contracts._diagnostic("ERROR", "parameter.specie", "specie must be one element symbol")
        )
    if parameters.get("seed") is not None:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "parameter.seed_unsupported",
                "post-processing has no stochastic stage or seed CLI; omit seed instead of recording an unused value",
            )
        )

    if source in {"trajectory", "vasp"} and (
        parameters.get("fit_start_ps") is not None or parameters.get("fit_end_ps") is not None
    ):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "parameter.msd_fit_window_trajectory",
                "fit_start_ps/fit_end_ps apply only to MSD-table input",
            )
        )
    else:
        contracts._validate_interval(
            parameters, "fit_start_ps", "fit_end_ps", diagnostics, required=False
        )
    contracts._validate_interval(
        parameters, "trajectory_start_ps", "trajectory_end_ps", diagnostics, required=False
    )
    if (
        not contracts._positive_int(parameters.get("min_msd_fit_points"))
        or int(parameters.get("min_msd_fit_points", 0)) < 3
    ):
        diagnostics.append(
            contracts._diagnostic(
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
                contracts._diagnostic(
                    "ERROR",
                    "parameter.%s" % name,
                    "%s must be one of %s" % (name, sorted(allowed)),
                )
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

    if source == "msd":
        if parameters.get("msd_time_unit") == "auto":
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "unit.msd_time_explicit",
                    "MSD input requires an explicit time unit",
                )
            )
        if parameters.get("msd_unit") == "auto":
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "unit.msd_explicit", "MSD input requires an explicit MSD unit"
                )
            )
        if (
            parameters.get("n_mobile_ions") is not None
            or parameters.get("volume_a3") is not None
        ):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "assumption.legacy_ne_inputs_unsupported",
                    "formal MSD conductivity does not accept n_mobile_ions/volume_a3; "
                    "provide inputs.structure or leave conductivity unavailable",
                )
            )
    elif parameters.get("msd_time_unit") == "auto" or parameters.get("msd_unit") == "auto":
        diagnostics.append(
            contracts._diagnostic(
                "WARNING",
                "unit.auto_detection",
                "auto source/unit detection is delegated to the reviewed CLI and recorded in its outputs",
            )
        )
    if parameters.get("msd_time_unit") == "step" and not contracts._positive_number(
        parameters.get("msd_step_ps")
    ):
        diagnostics.append(
            contracts._diagnostic(
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
        contracts._optional_positive(parameters, name, diagnostics)
    if not contracts._positive_number(parameters.get("target_temperature_k")):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "parameter.target_temperature_k",
                "target_temperature_k must be a finite positive number",
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
    for name in (
        "diffusion_analyzer_min_obs",
        "diffusion_analyzer_avg_nsteps",
        "diffusion_analyzer_step_skip",
    ):
        value = parameters.get(name)
        if not contracts._positive_int(value):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "parameter.%s" % name, "%s must be a positive integer" % name
                )
            )
    for name in ("n_mobile_ions", "msd_smooth_window_points", "mobile_type"):
        value = parameters.get(name)
        if value is not None and not contracts._positive_int(value):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "parameter.%s" % name, "%s must be a positive integer" % name
                )
            )
    for name in ("fit_smoothed_msd", "allow_partial_results"):
        if not isinstance(parameters.get(name), bool):
            diagnostics.append(
                contracts._diagnostic("ERROR", "parameter.%s" % name, "%s must be boolean" % name)
            )
    if (
        parameters.get("fit_smoothed_msd") is True
        or parameters.get("msd_smooth_window_points") is not None
        or parameters.get("msd_smooth_window_ps") is not None
    ):
        diagnostics.append(
            contracts._diagnostic(
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
                contracts._diagnostic(
                    "ERROR", "parameter.%s" % name, "%s must be a non-empty string" % name
                )
            )

    rdf_names = (
        "rdf_pair",
        "rdf_r_min_angstrom",
        "rdf_r_max_angstrom",
        "rdf_bins",
        "rdf_temperature_k",
        "comparison_model",
        "comparison_scenario",
        "comparison_split",
    )
    rdf_requested = parameters.get("rdf_pair") is not None
    if any(parameters.get(name) is not None for name in rdf_names) and not rdf_requested:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "parameter.rdf_pair",
                "rdf_pair is required when AIMD/MLIP comparison parameters are supplied",
            )
        )
    if rdf_requested:
        pair = parameters.get("rdf_pair")
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(
                not isinstance(value, str) or re.fullmatch(r"[A-Z][a-z]?", value) is None
                for value in pair
            )
        ):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "parameter.rdf_pair",
                    "rdf_pair must contain exactly two element symbols",
                )
            )
        r_min = parameters.get("rdf_r_min_angstrom")
        r_max = parameters.get("rdf_r_max_angstrom")
        if (
            not contracts._finite_number(r_min)
            or float(r_min) < 0.0
            or not contracts._positive_number(r_max)
            or float(r_max) <= float(r_min)
        ):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "parameter.rdf_range",
                    "rdf range must satisfy 0 <= rdf_r_min_angstrom < rdf_r_max_angstrom",
                )
            )
        if not contracts._positive_int(parameters.get("rdf_bins")):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "parameter.rdf_bins", "rdf_bins must be a positive integer"
                )
            )
        if parameters.get("rdf_temperature_k") is not None and not contracts._positive_number(
            parameters.get("rdf_temperature_k")
        ):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "parameter.rdf_temperature_k",
                    "rdf_temperature_k must be positive when supplied",
                )
            )
        for name in (
            "comparison_model",
            "comparison_scenario",
            "comparison_split",
        ):
            value = parameters.get(name)
            if (
                not isinstance(value, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None
            ):
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        f"parameter.{name}",
                        f"{name} must be an explicit portable identifier",
                    )
                )
        if source not in {"auto", "trajectory"}:
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "parameter.rdf_source",
                    "AIMD/MLIP RDF comparison requires source=auto or trajectory",
                )
            )
        if parameters.get("fit_scope") != "dataset":
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "parameter.rdf_fit_scope",
                    "AIMD and MLIP Arrhenius evidence must remain separated with fit_scope=dataset",
                )
            )
    return diagnostics


def plan(context: Any) -> Dict[str, Any]:
    diagnostics = validate(context)
    if contracts._has_errors(diagnostics):
        return contracts._blocked(diagnostics)

    project_root = contracts._resolve(context["project_root"], Path.cwd())
    attempt_dir = contracts._resolve(context["attempt_dir"], project_root)
    inputs = contracts._mapping(context["inputs"])
    parameters = contracts._mapping(context["parameters"])
    resources = contracts._mapping(context.get("resources"))
    assert project_root is not None and attempt_dir is not None
    if contracts._operation_name(parameters) == contracts._SMOKE_OPERATION:
        return md_handoff.plan_md_handoff(
            context, diagnostics, project_root, attempt_dir, inputs, parameters, resources
        )
    script = contracts._bundled_analysis_script()
    input_paths = [contracts._resolve(value, project_root) for value in inputs["input_paths"]]
    output_dir = (
        attempt_dir / str(parameters.get("output_subdir", "ionic-transport-postprocess"))
    ).resolve()

    argv = [str(resources.get("python_executable", sys.executable)), str(script)]
    for path in input_paths:
        argv.extend(["--input", str(path)])
    argv.extend(["--output", str(output_dir)])

    argv.extend(contracts._analysis_arguments(parameters))
    if structure_input := contracts._resolve(inputs.get("structure"), project_root):
        argv.extend(["--msd-structure", str(structure_input)])

    expected_outputs = [str(output_dir / name) for name in contracts._REQUIRED_RESULT_NAMES]
    if parameters.get("rdf_pair") is not None:
        expected_outputs.extend(
            [
                str(output_dir / contracts._RDF_CURVES_NAME),
                str(output_dir / contracts._AIMD_MLIP_COMPARISON_NAME),
            ]
        )
    return {
        "plugin_id": contracts.PLUGIN_ID,
        "operation": contracts._ANALYZE_OPERATION,
        "status": "READY",
        "executable": True,
        "argv": argv,
        "cwd": str(attempt_dir),
        "shell": False,
        "expected_outputs": expected_outputs,
        "output_dir": str(output_dir),
        "assumptions": {
            "formal_scientific_implementation": "pymatgen-analysis-diffusion",
            "trajectory_haven_ratio": "reported-directly-by-DiffusionAnalyzer",
            "conductivity_method": "pymatgen-public-api-or-unavailable",
            "seed": None,
        },
        "provenance": {
            "analysis_source_path": str(script),
        },
        "diagnostics": diagnostics,
    }


def _verify_analysis_manifest(
    context: Mapping[str, Any], output_dir: Path, manifest_path: Path
) -> List[Dict[str, str]]:
    diagnostics: List[Dict[str, str]] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [contracts._diagnostic("ERROR", "result.analysis_manifest", f"cannot parse manifest: {exc}")]
    if not isinstance(manifest, Mapping):
        return [contracts._diagnostic("ERROR", "result.analysis_manifest", "manifest must be an object")]
    if manifest.get("schema_version") != 1 or manifest.get("plugin_id") != contracts.PLUGIN_ID:
        diagnostics.append(
            contracts._diagnostic("ERROR", "result.analysis_manifest", "manifest header is invalid")
        )
    contract = contracts._mapping(manifest.get("scientific_contract"))
    if (
        contract.get("formal_implementation") != "pymatgen-analysis-diffusion public API"
        or contract.get("trajectory_diffusion") != "DiffusionAnalyzer"
        or contract.get("msd_only_diffusion") != "get_diffusivity_from_msd"
        or contract.get("arrhenius") != "fit_arrhenius(mode='linear')"
        or contract.get("arrhenius_model_selection")
        != (
            "single baseline plus at most one adjacent-temperature breakpoint by BIC "
            "and relative activation-energy change"
        )
        or contract.get("historical_implementation") != "separate adapter-only legacy reproduction"
    ):
        diagnostics.append(
            contracts._diagnostic(
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
            contracts._diagnostic(
                "ERROR",
                "result.runtime_provenance",
                "manifest runtime_provenance must be an object",
            )
        )
    else:
        expected_runtime, probe_error = contracts._formal_runtime_probe()
        if probe_error is not None or expected_runtime is None:
            diagnostics.append(
                contracts._diagnostic(
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
                        contracts._diagnostic(
                            "ERROR",
                            "result.runtime_version",
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
                        contracts._diagnostic(
                            "ERROR",
                            "result.runtime_version",
                            "manifest ASE version differs from the selected local interpreter",
                        )
                    )

    recorded_parameters = contracts._mapping(manifest.get("parameters"))
    parameters = contracts._mapping(context.get("parameters"))
    operation = contracts._operation_name(parameters)
    adapter_only = {"operation", "output_subdir", "seed", "allow_partial_results"}
    runner_destinations = {
        "temperature_k": "temperature_K",
        "aimd_temperature_k": "aimd_temperature_K",
        "msd_temperature_k": "msd_temperature_K",
        "target_temperature_k": "target_temperature_K",
        "rdf_temperature_k": "rdf_temperature_K",
        "volume_a3": "volume_A3",
    }
    for name, expected in parameters.items():
        if name in adapter_only or name not in contracts._ANALYZE_PARAMETERS:
            continue
        if operation == contracts._SMOKE_OPERATION and name == "ase_frame_step_fs":
            continue
        actual = recorded_parameters.get(runner_destinations.get(name, name))
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            if not contracts._finite_number(actual) or not contracts._numbers_close(float(actual), float(expected)):
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR", "result.parameter_mismatch", f"manifest parameter {name} differs"
                    )
                )
        elif actual != expected:
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "result.parameter_mismatch", f"manifest parameter {name} differs"
                )
            )

    project_root = contracts._resolve(context.get("project_root"), Path.cwd()) or Path.cwd().resolve()
    inputs = contracts._mapping(context.get("inputs"))
    if operation == contracts._SMOKE_OPERATION:
        execution = contracts._mapping(context.get("execution"))
        plan = contracts._mapping(execution.get("plan"))
        smoke_input = contracts._resolve(plan.get("md_output_dir"), project_root)
        expected_inputs = [] if smoke_input is None else [str(smoke_input)]
    else:
        expected_inputs = [
            str(path)
            for path in (contracts._resolve(value, project_root) for value in inputs.get("input_paths", []))
            if path is not None
        ]
    if recorded_parameters.get("input") != expected_inputs:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "result.input_paths_mismatch",
                "manifest inputs differ from approved inputs",
            )
        )
    if recorded_parameters.get("output") != str(output_dir):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "result.output_path_mismatch",
                "manifest output differs from execution plan",
            )
        )

    implementation = manifest.get("implementation_artifacts")
    if not isinstance(implementation, list) or len(implementation) != 1:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "result.implementation_artifacts", "manifest implementation is incomplete"
            )
        )
    else:
        expected_implementation = {
            "packaged-analysis-runner": contracts._bundled_analysis_script(),
        }
        for index, record in enumerate(implementation):
            if isinstance(record, Mapping):
                expected_path = expected_implementation.get(str(record.get("role")))
                actual_path = contracts._resolve(record.get("path"), project_root)
                if expected_path is None or actual_path != expected_path:
                    diagnostics.append(
                        contracts._diagnostic(
                            "ERROR",
                            "result.implementation_path",
                            f"implementation_artifacts[{index}] is not the selected implementation",
                        )
                    )
            _, record_diagnostics = contracts._verify_file_record(
                record, role=f"implementation_artifacts[{index}]"
            )
            diagnostics.extend(record_diagnostics)

    sources = manifest.get("source_artifacts")
    if not isinstance(sources, list) or not sources:
        diagnostics.append(
            contracts._diagnostic("ERROR", "result.source_artifacts", "manifest has no source artifacts")
        )
    else:
        approved_roots = [Path(value).resolve() for value in expected_inputs]
        structure_input = contracts._resolve(inputs.get("structure"), project_root)
        if structure_input is not None:
            approved_roots.append(structure_input)
        for index, record in enumerate(sources):
            path, record_diagnostics = contracts._verify_file_record(
                record, role=f"source_artifacts[{index}]"
            )
            diagnostics.extend(record_diagnostics)
            if path is not None and not any(contracts._is_within(path, root) for root in approved_roots):
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        "result.source_outside_inputs",
                        f"source_artifacts[{index}] is outside the approved inputs",
                    )
                )
    results = manifest.get("result_artifacts")
    if not isinstance(results, list) or not results:
        diagnostics.append(
            contracts._diagnostic("ERROR", "result.result_artifacts", "manifest has no result artifacts")
        )
    else:
        for index, record in enumerate(results):
            _, record_diagnostics = contracts._verify_file_record(
                record, role=f"result_artifacts[{index}]", confined_to=output_dir
            )
            diagnostics.extend(record_diagnostics)
    return diagnostics


def _load_formal_runner():
    from . import ionic_conductivity

    return ionic_conductivity


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
        manifest = json.loads((output_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
        runner = _load_formal_runner()
        args = argparse.Namespace(**contracts._mapping(manifest.get("parameters")))
    except (OSError, UnicodeError, json.JSONDecodeError, ImportError, TypeError) as exc:
        return [
            contracts._diagnostic(
                "ERROR",
                "result.formal_dependency",
                f"formal checker cannot load pymatgen analysis: {exc}",
            )
        ]

    for index, row in enumerate(rows, start=1):
        curve_path = contracts._resolve(row.get("msd_curve_csv"), output_dir)
        if curve_path is None or not curve_path.is_file() or not contracts._is_within(curve_path, output_dir):
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
                not contracts._numbers_close(actual, wanted) for actual, wanted in zip(times, expected_times)
            ):
                raise ValueError("time_ps differs from the rerun pymatgen analysis")
            if len(msd) != len(expected_msd) or any(
                not contracts._numbers_close(actual, wanted) for actual, wanted in zip(msd, expected_msd)
            ):
                raise ValueError("MSD differs from the rerun pymatgen analysis")
            if used != expected_used:
                raise ValueError("analysis point mask differs from the approved window")
        except (OSError, UnicodeError, csv.Error, KeyError, TypeError, ValueError) as exc:
            diagnostics.append(
                contracts._diagnostic(
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
                matches = actual is not None and contracts._numbers_close(actual, float(wanted))
            if not matches:
                diagnostics.append(
                    contracts._diagnostic(
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
                    contracts._diagnostic(
                        "ERROR",
                        "result.method",
                        f"row {index} {name} is not {wanted}",
                    )
                )
    return diagnostics


def _same_scientific_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return actual == expected
    if isinstance(expected, (int, float)):
        return contracts._finite_number(actual) and contracts._numbers_close(float(actual), float(expected))
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_same_scientific_value(left, right) for left, right in zip(actual, expected))
        )
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_same_scientific_value(actual[key], value) for key, value in expected.items())
        )
    return actual == expected


def _verify_aimd_mlip_comparison(
    context: Mapping[str, Any],
    output_dir: Path,
    rows: Sequence[Mapping[str, str]],
    summary: Mapping[str, Any],
) -> List[Dict[str, str]]:
    parameters = contracts._mapping(context.get("parameters"))
    if parameters.get("rdf_pair") is None:
        return []
    curve_path = output_dir / contracts._RDF_CURVES_NAME
    comparison_path = output_dir / contracts._AIMD_MLIP_COMPARISON_NAME
    if not curve_path.is_file() or not comparison_path.is_file():
        return [
            contracts._diagnostic(
                "ERROR",
                "result.rdf_missing",
                "requested AIMD/MLIP RDF comparison outputs are missing",
            )
        ]
    try:
        manifest = json.loads((output_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
        runner = _load_formal_runner()
        args = argparse.Namespace(**contracts._mapping(manifest.get("parameters")))
        runs = [
            runner.load_run_data(str(row["dataset"]), Path(str(row["run_dir"])).resolve(), args)
            for row in rows
        ]
        expected, expected_curve = runner.build_aimd_mlip_comparison(
            runs, [dict(row) for row in rows], dict(summary), args
        )
        expected["rdf"]["curve_csv"] = str(curve_path.resolve())
        actual = json.loads(comparison_path.read_text(encoding="utf-8"))
        if not _same_scientific_value(actual, runner.json_ready(expected)):
            raise ValueError("comparison JSON differs from the trajectory rerun")

        reader = csv.DictReader(curve_path.read_text(encoding="utf-8").splitlines())
        curve_rows = list(reader)
        if reader.fieldnames is None or set(reader.fieldnames) != set(expected_curve):
            raise ValueError("RDF curve columns differ from the approved contract")
        for name, expected_values in expected_curve.items():
            actual_values = [float(row[name]) for row in curve_rows]
            if len(actual_values) != len(expected_values) or any(
                not contracts._numbers_close(actual_value, float(expected_value))
                for actual_value, expected_value in zip(actual_values, expected_values)
            ):
                raise ValueError(f"RDF curve column {name} differs from the trajectory rerun")
    except (
        OSError,
        UnicodeError,
        csv.Error,
        json.JSONDecodeError,
        ImportError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        return [
            contracts._diagnostic(
                "ERROR",
                "result.rdf_recheck",
                f"AIMD/MLIP RDF comparison cannot be verified: {exc}",
            )
        ]
    return []


def _verify_arrhenius_summary(
    output_dir: Path,
    rows: Sequence[Mapping[str, str]],
    summary: Mapping[str, Any],
) -> List[Dict[str, str]]:
    try:
        manifest = json.loads((output_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
        runner = _load_formal_runner()
        args = argparse.Namespace(**contracts._mapping(manifest.get("parameters")))
        results_df = runner.pd.DataFrame([dict(row) for row in rows])
        for name in (
            "temperature_K",
            "diffusivity_cm2_s",
            "conductivity_NE_mS_cm",
        ):
            if name in results_df:
                results_df[name] = runner.pd.to_numeric(results_df[name], errors="coerce")
        expected = runner.build_arrhenius_summary(results_df, args)
        if not _same_scientific_value(summary, runner.json_ready(expected)):
            raise ValueError(
                "saved single/piecewise Arrhenius evidence differs from the declared rerun"
            )
    except (
        AttributeError,
        ImportError,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        return [
            contracts._diagnostic(
                "ERROR",
                "result.arrhenius_relation",
                f"Arrhenius summary rerun failed: {exc}",
            )
        ]
    return []


def _result_paths(context: Any) -> Tuple[Optional[Path], Dict[str, Path]]:
    if not isinstance(context, Mapping):
        return None, {}
    project_root = contracts._resolve(context.get("project_root"), Path.cwd()) or Path.cwd().resolve()
    inputs = contracts._mapping(context.get("inputs"))
    explicit = inputs.get("result_files")
    paths: Dict[str, Path] = {}
    if isinstance(explicit, Mapping):
        for name in contracts._REQUIRED_RESULT_NAMES:
            path = contracts._resolve(explicit.get(name), project_root)
            if path is not None:
                paths[name] = path
        if len(paths) == len(contracts._REQUIRED_RESULT_NAMES):
            parents = {path.parent for path in paths.values()}
            return (next(iter(parents)) if len(parents) == 1 else None), paths

    execution = contracts._mapping(context.get("execution"))
    plan = contracts._mapping(execution.get("plan"))
    expected = plan.get("expected_outputs")
    if isinstance(expected, list) and len(expected) >= len(contracts._REQUIRED_RESULT_NAMES):
        for name, value in zip(contracts._REQUIRED_RESULT_NAMES, expected[: len(contracts._REQUIRED_RESULT_NAMES)]):
            path = contracts._resolve(value, project_root)
            if path is not None:
                paths[name] = path
        output_dir = contracts._resolve(plan.get("output_dir"), project_root)
        if output_dir is not None and len(paths) == len(contracts._REQUIRED_RESULT_NAMES):
            return output_dir, paths
    return None, {}


def _read_results(
    context: Any
) -> Tuple[
    Optional[Path],
    List[Dict[str, str]],
    Dict[str, Any],
    List[Dict[str, Any]],
    List[Dict[str, str]],
]:
    output_dir, paths = _result_paths(context)
    diagnostics: List[Dict[str, str]] = []
    if output_dir is None or len(paths) != len(contracts._REQUIRED_RESULT_NAMES):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "result.files_explicit_required",
                "result files must come from execution.plan.expected_outputs or inputs.result_files",
            )
        )
        return None, [], {}, [], diagnostics
    for name, path in paths.items():
        if not contracts._is_within(path, output_dir):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "result.path_escape",
                    "%s is outside the explicit output directory" % name,
                )
            )
        elif not path.is_file():
            diagnostics.append(
                contracts._diagnostic("INFO", "result.missing", "%s does not exist yet" % name)
            )
    if any(item["code"] == "result.missing" for item in diagnostics):
        return output_dir, [], {}, [], diagnostics
    if contracts._has_errors(diagnostics):
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
            contracts._diagnostic(
                "ERROR", "result.unreadable", "cannot parse transport results: %s" % exc
            )
        )
        return output_dir, [], {}, [], diagnostics
    if reader.fieldnames is None or not contracts._RESULT_COLUMNS.issubset(set(reader.fieldnames)):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "result.columns", "diffusion results CSV has an unexpected header"
            )
        )
    if not rows:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "result.no_rows", "diffusion results CSV contains no successful runs"
            )
        )
    if not isinstance(summary, dict):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "result.summary_mapping", "arrhenius_summary.json must be an object"
            )
        )
        summary = {}
    if not isinstance(failures, list) or not all(isinstance(item, dict) for item in failures):
        diagnostics.append(
            contracts._diagnostic(
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
                    contracts._diagnostic(
                        "ERROR",
                        "result.nonfinite",
                        "row %d has non-finite %s" % (index + 1, name),
                    )
                )
                break
        else:
            if float(row["temperature_K"]) <= 0 or float(row["diffusivity_cm2_s"]) <= 0:
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        "result.nonpositive_transport",
                        "temperature and diffusivity must be positive in row %d" % (index + 1),
                    )
                )
            if float(row["analysis_start_ps"]) >= float(row["analysis_end_ps"]):
                diagnostics.append(
                    contracts._diagnostic(
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
                    contracts._diagnostic(
                        "ERROR",
                        "result.conductivity_unavailable",
                        "row %d must explain unavailable conductivity" % (index + 1),
                    )
                )
        elif conductivity is None:
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "result.conductivity_missing",
                    "row %d must contain pymatgen conductivity" % (index + 1),
                )
            )
        for name in ("msd_curve_csv", "msd_fit_html"):
            artifact = contracts._resolve(row.get(name), output_dir)
            if (
                artifact is None
                or not contracts._is_within(artifact, output_dir)
                or not artifact.is_file()
            ):
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        "result.artifact_missing",
                        "row %d references a missing/out-of-tree %s" % (index + 1, name),
                    )
                )
    return output_dir, rows, summary, failures, diagnostics


def check(context: Any) -> Dict[str, Any]:
    execution = contracts._mapping(context.get("execution")) if isinstance(context, Mapping) else {}
    returncode = execution.get("returncode")
    if returncode is not None and returncode != 0:
        return {
            "plugin_id": contracts.PLUGIN_ID,
            "status": "FAIL",
            "diagnostics": [
                contracts._diagnostic(
                    "ERROR", "execution.nonzero", "analysis process returned %s" % returncode
                )
            ],
        }
    output_dir, rows, summary, failures, diagnostics = _read_results(context)
    if output_dir is not None and any(item["code"] == "result.missing" for item in diagnostics):
        return {"plugin_id": contracts.PLUGIN_ID, "status": "WAIT", "diagnostics": diagnostics}
    parameters = contracts._mapping(context.get("parameters")) if isinstance(context, Mapping) else {}
    operation = contracts._operation_name(parameters)
    if output_dir is not None and isinstance(context, Mapping):
        diagnostics.extend(
            _verify_analysis_manifest(
                context, output_dir, output_dir / "analysis_manifest.json"
            )
        )
        diagnostics.extend(_verify_transport_rows(rows, output_dir))
        diagnostics.extend(_verify_arrhenius_summary(output_dir, rows, summary))
        diagnostics.extend(_verify_aimd_mlip_comparison(context, output_dir, rows, summary))
    integration_manifest: Optional[Dict[str, Any]] = None
    if operation == contracts._SMOKE_OPERATION and isinstance(context, Mapping):
        integration_manifest, _, integration_diagnostics = md_handoff._verify_smoke_manifest(context)
        diagnostics.extend(integration_diagnostics)
        if any(item["code"] == "integration.manifest_missing" for item in diagnostics):
            return {"plugin_id": contracts.PLUGIN_ID, "status": "WAIT", "diagnostics": diagnostics}
        if (
            not isinstance(summary.get("single"), Mapping)
            or summary.get("arrhenius_fit_skipped") is True
        ):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "integration.arrhenius_incomplete",
                    "md smoke must complete the trajectory-to-Arrhenius handoff",
                )
            )
        else:
            activation = summary["single"].get("Ea_eV")
            if not contracts._finite_number(activation):
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        "integration.activation_energy",
                        "smoke Arrhenius result requires finite single.Ea_eV",
                    )
                )
        expected_temperatures = sorted(float(value) for value in parameters["temperatures_k"])
        actual_temperatures = sorted(float(row["temperature_K"]) for row in rows)
        if actual_temperatures != expected_temperatures:
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "integration.temperature_coverage",
                    "transport rows must cover every approved smoke temperature exactly once",
                )
            )
    if failures and parameters.get("allow_partial_results") is not True:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "result.partial_disallowed",
                "%d input run(s) failed; set allow_partial_results=true only after review"
                % len(failures),
            )
        )
    elif failures:
        diagnostics.append(
            contracts._diagnostic(
                "WARNING",
                "result.partial_allowed",
                "%d input run(s) were skipped" % len(failures),
            )
        )
    if contracts._has_errors(diagnostics):
        return {"plugin_id": contracts.PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
    return {
        "plugin_id": contracts.PLUGIN_ID,
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


def collect(context: Any) -> Dict[str, Any]:
    output_dir, paths = _result_paths(context)
    assert output_dir is not None
    with paths["diffusion_results_by_temperature.csv"].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads(paths["arrhenius_summary.json"].read_text(encoding="utf-8"))
    failures = json.loads(paths["postprocess_failures.json"].read_text(encoding="utf-8"))
    diagnostics: List[Dict[str, str]] = []
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
    for name in contracts._OPTIONAL_RESULT_NAMES:
        path = output_dir / name
        if path.is_file():
            artifacts.append(
                {"role": "transport-diagnostic", "path": str(path), "media_type": "text/html"}
            )
    comparison: Optional[Dict[str, Any]] = None
    if contracts._mapping(context.get("parameters")).get("rdf_pair") is not None:
        curve_path = output_dir / contracts._RDF_CURVES_NAME
        comparison_path = output_dir / contracts._AIMD_MLIP_COMPARISON_NAME
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        artifacts.extend(
            [
                {
                    "role": "rdf-curves",
                    "path": str(curve_path),
                    "media_type": "text/csv",
                },
                {
                    "role": "aimd-mlip-comparison",
                    "path": str(comparison_path),
                    "media_type": "application/json",
                },
            ]
        )
    seen = {item["path"] for item in artifacts}
    for row in rows:
        for field, role, media_type in (
            ("msd_curve_csv", "msd-curve", "text/csv"),
            ("msd_fit_html", "msd-fit", "text/html"),
        ):
            path = contracts._resolve(row[field], output_dir)
            assert path is not None
            if str(path) not in seen:
                artifacts.append({"role": role, "path": str(path), "media_type": media_type})
                seen.add(str(path))

    parameters = contracts._mapping(context.get("parameters")) if isinstance(context, Mapping) else {}
    operation = contracts._operation_name(parameters)
    integration_manifest: Optional[Dict[str, Any]] = None
    if operation == contracts._SMOKE_OPERATION and isinstance(context, Mapping):
        manifest_path = md_handoff._integration_manifest_path(context)
        assert manifest_path is not None
        project_root = contracts._resolve(context.get("project_root"), Path.cwd()) or Path.cwd().resolve()
        attempt_dir = contracts._resolve(context.get("attempt_dir"), project_root)
        assert attempt_dir is not None
        integration_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        integration_artifacts = [
            {"role": "md-integration-manifest", "path": str(manifest_path),
             "media_type": "application/json"},
            *({"role": "production-trajectory",
               "path": str(contracts._resolve(record["path"], attempt_dir)),
               "media_type": "application/octet-stream"}
              for record in integration_manifest["trajectory_artifacts"]),
        ]
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
                "conductivity_NE_mS_cm": _optional_csv_number(row.get("conductivity_NE_mS_cm")),
                "conductivity_std_dev_mS_cm": _optional_csv_number(
                    row.get("conductivity_std_dev_mS_cm")
                ),
                "chg_diffusivity_cm2_s": _optional_csv_number(row.get("chg_diffusivity_cm2_s")),
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
        "plugin_id": contracts.PLUGIN_ID,
        "operation": operation,
        "status": "OK",
        "artifacts": artifacts,
        "metrics": {
            "run_count": len(rows),
            "failed_run_count": len(failures),
            "temperature_count": len({item["temperature_K"] for item in by_temperature}),
            "by_temperature": by_temperature,
            "arrhenius": summary,
            "aimd_mlip_comparison": comparison,
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
