"""pes sampling: ssw."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from . import contracts


def validate(context: Any) -> List[Dict[str, str]]:
    diagnostics: List[Dict[str, str]] = []
    if not isinstance(context, Mapping):
        return [contracts._diagnostic("ERROR", "context.mapping_required", "context must be a mapping")]

    operation = contracts._operation(context)
    if operation not in contracts.LASP_OPERATIONS:
        return [contracts._diagnostic("ERROR", "operation.unsupported", "unsupported LASP/SSW operation")]
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
                "LASP/SSW adapter execution is local-only; it never submits scheduler jobs",
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
    unknown_resources = sorted(set(resources) - {"python_executable", "mpi_launcher"})
    if unknown_resources:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "resource.unknown",
                "unsupported LASP resource(s): %s" % ", ".join(unknown_resources),
            )
        )
    unknown_parameters = sorted(set(parameters) - contracts._LASP_PARAMETERS)
    if unknown_parameters:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "parameter.unknown",
                "unsupported parameter(s): %s" % ", ".join(unknown_parameters),
            )
        )

    allowed_inputs = (
        {"historical_run_dir", "lasp_input", "result_manifest"}
        if operation == "lasp-ssw-normalize-replay"
        else {
            "lasp_executable",
            "input_structure",
            "lasp_input",
            "lasp_auxiliary_files",
            "result_manifest",
        }
    )
    unknown_inputs = sorted(set(inputs) - allowed_inputs)
    if unknown_inputs:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "input.unknown",
                "unsupported LASP input(s): %s" % ", ".join(unknown_inputs),
            )
        )

    if not contracts._ordinary_file(contracts.BUNDLED_LASP_WRAPPER):
        diagnostics.append(
            contracts._diagnostic("ERROR", "path.lasp_wrapper", "bundled lasp_ssw.py wrapper is missing")
        )
    lasp_input = contracts._resolve(inputs.get("lasp_input"), project_root)
    if not contracts._ordinary_file(lasp_input):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "path.lasp_input",
                "inputs.lasp_input must name a lasp.in file",
            )
        )
    else:
        assert lasp_input is not None
        try:
            contracts._validate_ssw_input(contracts._parse_lasp_input(lasp_input))
        except (OSError, UnicodeError, ValueError) as exc:
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "input.lasp_ssw_contract", f"invalid LASP/SSW input: {exc}"
                )
            )

    output_subdir = parameters.get("output_subdir", "lasp-ssw")
    if not contracts._safe_subdirectory(output_subdir):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "path.output_subdir",
                "output_subdir must be a non-empty relative path without '..'",
            )
        )
        output_dir = attempt_dir / "lasp-ssw"
    else:
        output_dir = (attempt_dir / str(output_subdir)).resolve()
        if not contracts._is_within(output_dir, attempt_dir):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "path.output_escape", "LASP output must stay inside attempt_dir"
                )
            )
    if output_dir.exists() or output_dir.is_symlink():
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "path.output_exists",
                "LASP/SSW requires a fresh attempt/output_subdir",
            )
        )

    if operation == "lasp-ssw-normalize-replay":
        source_dir = contracts._resolve(inputs.get("historical_run_dir"), project_root)
        if source_dir is None or not source_dir.is_dir():
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "path.historical_run_dir",
                    "inputs.historical_run_dir must be a directory",
                )
            )
        else:
            allstr = source_dir / "allstr.arc"
            if not contracts._ordinary_file(allstr):
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        "path.allstr_arc",
                        "historical_run_dir must contain ordinary allstr.arc",
                    )
                )
            if contracts._is_within(output_dir, source_dir) or contracts._is_within(source_dir, output_dir):
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        "path.input_output_overlap",
                        "historical input and output directories must not overlap",
                    )
                )
            for flag, name in (
                (parameters.get("include_best_arc", False), "best.arc"),
                (parameters.get("include_md_arc", False), "md.arc"),
            ):
                if flag is True and not contracts._ordinary_file(source_dir / name):
                    diagnostics.append(
                        contracts._diagnostic(
                            "ERROR", "path.%s" % name.replace(".", "_"), f"missing {name}"
                        )
                    )
        if parameters.get("lasp_version") not in (None, ""):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "parameter.lasp_version_replay",
                    "lasp_version is execute-only; replay records the historical input exactly",
                )
            )
        if (
            parameters.get("mpi_processes") is not None
            or resources.get("mpi_launcher") is not None
        ):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "resource.mpi_replay", "MPI settings are not used by replay"
                )
            )
    else:
        executable = contracts._resolve(inputs.get("lasp_executable"), project_root)
        if not contracts._ordinary_file(executable):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "path.lasp_executable",
                    "inputs.lasp_executable must name a user-supplied ordinary LASP binary",
                )
            )
        elif executable.stat().st_mode & 0o111 == 0:
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "path.lasp_not_executable", "LASP binary is not executable"
                )
            )
        input_structure = contracts._resolve(inputs.get("input_structure"), project_root)
        if not contracts._ordinary_file(input_structure):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "path.input_structure",
                    "inputs.input_structure must name an ordinary ARC structure file",
                )
            )
        auxiliary = inputs.get("lasp_auxiliary_files", {})
        if not isinstance(auxiliary, Mapping):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "input.lasp_auxiliary_files",
                    "lasp_auxiliary_files must map safe destination basenames to files",
                )
            )
        else:
            seen = set()
            reserved = {
                "input.arc",
                "lasp.in",
                "all.arc",
                "allstr.arc",
                "allstr.native.arc",
                "best.arc",
                "md.arc",
                "sampling-result.json",
            }
            for name, raw_path in auxiliary.items():
                if (
                    not isinstance(name, str)
                    or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name)
                    or name in reserved
                    or name in seen
                ):
                    diagnostics.append(
                        contracts._diagnostic(
                            "ERROR",
                            "input.auxiliary_name",
                            "auxiliary destinations must be unique safe non-reserved basenames",
                        )
                    )
                    break
                seen.add(name)
                if not contracts._ordinary_file(contracts._resolve(raw_path, project_root)):
                    diagnostics.append(
                        contracts._diagnostic(
                            "ERROR",
                            "path.auxiliary_file",
                            f"auxiliary input {name} is missing",
                        )
                    )
                    break
        if not contracts._plain_string(parameters.get("lasp_version")):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "parameter.lasp_version",
                    "execute requires an explicit single-line lasp_version",
                )
            )
        launcher = resources.get("mpi_launcher")
        processes = parameters.get("mpi_processes")
        if launcher is not None:
            launcher_path = contracts._resolve(launcher, project_root)
            if (
                not contracts._plain_string(launcher)
                or launcher_path is None
                or not contracts._ordinary_file(launcher_path)
                or launcher_path.name not in {"mpirun", "mpiexec"}
                or launcher_path.stat().st_mode & 0o111 == 0
            ):
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        "resource.mpi_launcher",
                        "mpi_launcher must be an explicit ordinary executable path named mpirun or mpiexec",
                    )
                )
            if not contracts._positive_int(processes):
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        "parameter.mpi_processes",
                        "mpi_processes must be positive when mpi_launcher is set",
                    )
                )
        elif processes is not None:
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "parameter.mpi_without_launcher",
                    "mpi_processes requires resources.mpi_launcher",
                )
            )

    source_id = parameters.get("historical_source_id")
    if not contracts._portable_source_id(source_id):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "parameter.historical_source_id",
                "historical_source_id must be portable and must not expose an absolute path",
            )
        )
    if not contracts._positive_int(parameters.get("selection_stride")):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "parameter.selection_stride", "selection_stride must be positive"
            )
        )
    max_frames = parameters.get("max_frames")
    if not contracts._positive_int(max_frames) or (
        isinstance(max_frames, int) and max_frames > contracts.MAX_LASP_FRAMES
    ):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "parameter.max_frames",
                f"max_frames must be a positive integer no greater than {contracts.MAX_LASP_FRAMES}",
            )
        )
    energy_max = parameters.get("energy_max_ev")
    if energy_max is not None and (
        not isinstance(energy_max, (int, float))
        or isinstance(energy_max, bool)
        or not math.isfinite(float(energy_max))
    ):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "parameter.energy_max_ev", "energy_max_ev must be finite or null"
            )
        )
    for name in ("include_best_arc", "include_md_arc"):
        if not isinstance(parameters.get(name, False), bool):
            diagnostics.append(
                contracts._diagnostic("ERROR", f"parameter.{name}", f"{name} must be boolean")
            )
    if parameters.get("preserve_historical_order") is not True:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "parameter.preserve_historical_order",
                "preserve_historical_order must be true",
            )
        )
    if parameters.get("seed_status") != contracts.UNKNOWN:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "seed.historical_unknown",
                f"seed_status must be {contracts.UNKNOWN}; the reviewed source did not record a seed",
            )
        )
    if parameters.get("acknowledge_uncontrolled_seed") is not True:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "seed.acknowledgement_required",
                "acknowledge_uncontrolled_seed must be true",
            )
        )
    else:
        diagnostics.append(
            contracts._diagnostic(
                "WARNING",
                "seed.not_reconstructed",
                "the historical LASP seed is unknown and is not reconstructed",
            )
        )
    python_executable = resources.get("python_executable")
    if contracts._explicit_executable(python_executable, project_root) is None:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "resource.python_executable",
                "python_executable must be an explicit absolute executable file",
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
    operation = contracts._operation(context)
    output_dir = (attempt_dir / str(parameters.get("output_subdir", "lasp-ssw"))).resolve()
    input_paths: Dict[str, str] = {}

    def remember_input(name: str, path: Path) -> None:
        input_paths[name] = str(path)

    python_executable = contracts._explicit_executable(
        resources.get("python_executable"),
        project_root,
    )
    assert python_executable is not None
    wrapper = contracts.BUNDLED_LASP_WRAPPER.resolve()
    remember_input("python_executable", python_executable)
    remember_input("lasp_wrapper", wrapper)
    argv = [
        str(python_executable),
        str(wrapper),
        "normalize-replay" if operation == "lasp-ssw-normalize-replay" else "execute",
    ]
    if operation == "lasp-ssw-normalize-replay":
        source_dir = contracts._resolve(inputs["historical_run_dir"], project_root)
        assert source_dir is not None
        argv.extend(["--historical-run-dir", str(source_dir)])
        remember_input("allstr.arc", source_dir / "allstr.arc")
        if parameters.get("include_best_arc", False):
            remember_input("best.arc", source_dir / "best.arc")
        if parameters.get("include_md_arc", False):
            remember_input("md.arc", source_dir / "md.arc")
    else:
        executable = contracts._resolve(inputs["lasp_executable"], project_root)
        structure = contracts._resolve(inputs["input_structure"], project_root)
        assert executable is not None and structure is not None
        argv.extend(
            [
                "--lasp-executable",
                str(executable),
                "--input-structure",
                str(structure),
                "--lasp-version",
                str(parameters["lasp_version"]),
            ]
        )
        remember_input("lasp_executable", executable)
        remember_input("input_structure", structure)
        launcher = resources.get("mpi_launcher")
        if launcher is not None:
            launcher_path = contracts._resolve(launcher, project_root)
            assert launcher_path is not None
            argv.extend(
                [
                    "--mpi-launcher",
                    str(launcher_path),
                    "--mpi-processes",
                    str(parameters["mpi_processes"]),
                ]
            )
            remember_input("mpi_launcher", launcher_path)
        auxiliary = contracts._mapping(inputs.get("lasp_auxiliary_files"))
        for name in sorted(auxiliary):
            path = contracts._resolve(auxiliary[name], project_root)
            assert path is not None
            argv.extend(["--auxiliary", name, str(path)])
            remember_input(f"auxiliary:{name}", path)
    lasp_input = contracts._resolve(inputs["lasp_input"], project_root)
    assert lasp_input is not None
    remember_input("lasp_input", lasp_input)
    argv.extend(
        [
            "--lasp-input",
            str(lasp_input),
            "--output-dir",
            str(output_dir),
            "--historical-source-id",
            str(parameters["historical_source_id"]),
            "--selection-stride",
            str(parameters["selection_stride"]),
            "--max-frames",
            str(parameters["max_frames"]),
            "--seed-status",
            contracts.UNKNOWN,
        ]
    )
    if parameters.get("energy_max_ev") is not None:
        argv.extend(["--energy-max-ev", str(parameters["energy_max_ev"])])
    if parameters.get("include_best_arc", False):
        argv.append("--include-best-arc")
    if parameters.get("include_md_arc", False):
        argv.append("--include-md-arc")
    return {
        "plugin_id": contracts.PLUGIN_ID,
        "operation": operation,
        "status": "READY",
        "executable": True,
        "argv": argv,
        "cwd": str(attempt_dir),
        "shell": False,
        "expected_outputs": [str(output_dir / "sampling-result.json")],
        "output_dir": str(output_dir),
        "input_paths": input_paths,
        "assumptions": {
            "scientific_mode": (
                "historical-replay" if operation == "lasp-ssw-normalize-replay" else "execute"
            ),
            "seed_status": contracts.UNKNOWN,
            "selection_stride_applies_after_energy_filter": True,
            "preserve_historical_order": True,
            "fresh_output_directory_required": True,
            "scheduler_submission": False,
        },
        "diagnostics": diagnostics,
    }


def _read_lasp_result(
    context: Any
) -> Tuple[
    Optional[Path],
    Dict[str, Any],
    List[Dict[str, str]],
    List[Dict[str, Any]],
]:
    diagnostics: List[Dict[str, str]] = []
    manifest = contracts._manifest_path(context)
    if isinstance(context, Mapping):
        inputs = contracts._mapping(context.get("inputs"))
        execution = contracts._mapping(context.get("execution"))
        plan = contracts._mapping(execution.get("plan"))
        expected = plan.get("expected_outputs")
        explicit = inputs.get("result_manifest")
        if explicit is not None and isinstance(expected, list) and len(expected) == 1:
            project_root = (
                contracts._resolve(context.get("project_root"), Path.cwd()) or Path.cwd().resolve()
            )
            explicit_path = contracts._resolve(explicit, project_root)
            expected_path = contracts._resolve(expected[0], project_root)
            if explicit_path != expected_path:
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        "result.manifest_plan_mismatch",
                        "inputs.result_manifest must equal execution.plan.expected_outputs[0]",
                    )
                )
    if manifest is None:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "result.manifest_explicit_required",
                "sampling-result.json must come from the execution plan or inputs.result_manifest",
            )
        )
        return None, {}, diagnostics, []
    if not manifest.is_file():
        diagnostics.append(
            contracts._diagnostic(
                "INFO", "result.manifest_missing", "sampling-result.json does not exist yet"
            )
        )
        return manifest, {}, diagnostics, []
    result = contracts._read_json_object(manifest, "sampling_result", diagnostics)
    if result is None:
        return manifest, {}, diagnostics, []
    operation = contracts._operation(context)
    expected_mode = (
        "historical-replay" if operation == "lasp-ssw-normalize-replay" else "execute"
    )
    for key, expected in (
        ("schema_version", 1),
        ("plugin_id", contracts.PLUGIN_ID),
        ("operation", operation),
        ("status", "OK"),
        ("mode", expected_mode),
    ):
        if result.get(key) != expected:
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "result.schema",
                    f"sampling result {key} must be {expected!r}",
                )
            )
    context_parameters = (
        contracts._mapping(context.get("parameters")) if isinstance(context, Mapping) else {}
    )
    if result.get("source_id") != context_parameters.get("historical_source_id"):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "result.source_id", "sampling result source_id does not match the plan"
            )
        )
    result_parameters = result.get("parameters")
    if not isinstance(result_parameters, Mapping):
        diagnostics.append(
            contracts._diagnostic("ERROR", "result.parameters", "result parameters must be an object")
        )
        result_parameters = {}
    expected_energy = context_parameters.get("energy_max_ev")
    for key, expected in (
        ("seed_status", contracts.UNKNOWN),
        ("selection_stride", context_parameters.get("selection_stride")),
        ("energy_max_ev", expected_energy),
        ("stride_applies_after_energy_filter", True),
        ("preserve_historical_order", True),
    ):
        if result_parameters.get(key) != expected:
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "result.parameter",
                    f"result parameter {key} does not match the approved plan",
                )
            )

    counts = result.get("counts")
    required_count_keys = {
        "generated_structure_count",
        "energy_accepted_count",
        "selected_structure_count",
        "aimd_seed_candidate_count",
        "md_structure_count",
    }
    if not isinstance(counts, Mapping) or required_count_keys - set(counts):
        diagnostics.append(
            contracts._diagnostic("ERROR", "result.counts", "result counts are incomplete")
        )
        counts = {}
    else:
        for name in required_count_keys:
            value = counts.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR", "result.count", f"result count {name} must be non-negative"
                    )
                )
        if counts.get("selected_structure_count", 0) < 1:
            diagnostics.append(
                contracts._diagnostic("ERROR", "result.no_samples", "LASP result selected no structures")
            )
        max_frames = context_parameters.get("max_frames")
        if (
            isinstance(max_frames, int)
            and counts.get("generated_structure_count", 0) > max_frames
        ):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "result.frame_limit", "LASP result exceeds approved max_frames"
                )
            )

    completion = result.get("completion_evidence")
    if (
        not isinstance(completion, Mapping)
        or any(
            completion.get(key) is not True
            for key in (
                "allstr_arc_parsed",
                "structure_manifest_validated_on_write",
                "selected_structures_nonempty",
                "incomplete_marker_removed",
            )
        )
        or completion.get("stdout_success_phrase_used") is not False
    ):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "result.completion_evidence",
                "completion must come from parsed files, never a stdout success phrase",
            )
        )
    if (manifest.parent / "INCOMPLETE.json").exists() or (
        manifest.parent / "INCOMPLETE.json"
    ).is_symlink():
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "result.incomplete_marker",
                "result directory is explicitly marked INCOMPLETE",
            )
        )

    raw_artifacts = result.get("artifacts")
    artifacts: List[Dict[str, Any]] = []
    role_paths: Dict[str, List[Path]] = {}
    seen_paths = set()
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "result.artifacts", "result artifacts must be a non-empty list"
            )
        )
        raw_artifacts = []
    for index, item in enumerate(raw_artifacts):
        if not isinstance(item, Mapping) or not contracts._plain_string(item.get("role")):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "result.artifact_record", f"artifact {index + 1} is invalid"
                )
            )
            continue
        if item.get("role") not in contracts.LASP_ARTIFACT_ROLES:
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "result.artifact_role_unknown",
                    f"artifact {index + 1} has an undeclared role",
                )
            )
            continue
        path = contracts._artifact_path(
            manifest, item.get("path"), f"artifacts[{index}]", diagnostics
        )
        if path is None:
            continue
        relative = path.relative_to(manifest.parent.resolve()).as_posix()
        if relative in seen_paths:
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "result.artifact_duplicate", f"duplicate artifact path: {relative}"
                )
            )
        seen_paths.add(relative)
        role = str(item["role"])
        role_paths.setdefault(role, []).append(path)
        artifacts.append(
            {
                "role": role,
                "path": str(path),
                "media_type": str(item.get("media_type", "application/octet-stream")),
            }
        )
    for role in (
        "ssw-structure-manifest",
        "selected-structure-manifest",
        "lasp-run-metadata",
    ):
        if len(role_paths.get(role, [])) != 1:
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "result.artifact_role", f"exactly one {role} artifact is required"
                )
            )

    source_frames: List[Dict[str, Any]] = []
    best_source_frames: List[Dict[str, Any]] = []
    md_source_frames: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    context_inputs = contracts._mapping(context.get("inputs")) if isinstance(context, Mapping) else {}
    context_resources = (
        contracts._mapping(context.get("resources")) if isinstance(context, Mapping) else {}
    )
    max_frames = context_parameters.get("max_frames")
    if not contracts._positive_int(max_frames) or int(max_frames) > contracts.MAX_LASP_FRAMES:
        max_frames = contracts.MAX_LASP_FRAMES
    source_paths: Dict[str, Path] = {}
    lasp_input_path: Optional[Path] = None
    if operation == "lasp-ssw-normalize-replay":
        project_root = (
            contracts._resolve(context.get("project_root"), Path.cwd())
            if isinstance(context, Mapping)
            else Path.cwd().resolve()
        ) or Path.cwd().resolve()
        source_dir = contracts._resolve(context_inputs.get("historical_run_dir"), project_root)
        lasp_input_path = contracts._resolve(context_inputs.get("lasp_input"), project_root)
        if source_dir is None or not source_dir.is_dir() or lasp_input_path is None:
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "result.source_inputs_required",
                    "historical_run_dir and lasp_input are required to bind replay results",
                )
            )
        else:
            source_paths = {
                "lasp-input": lasp_input_path,
                "ssw-archive": source_dir / "allstr.arc",
            }
            if context_parameters.get("include_best_arc", False):
                source_paths["best-archive"] = source_dir / "best.arc"
            if context_parameters.get("include_md_arc", False):
                source_paths["md-archive"] = source_dir / "md.arc"
    else:
        raw_run = manifest.parent / "raw-run"
        if not raw_run.is_dir():
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "result.raw_run_required",
                    "execute result must retain an ordinary raw-run directory",
                )
            )
        else:
            lasp_input_path = raw_run / "lasp.in"
            source_paths = {
                "lasp-input": lasp_input_path,
                "ssw-archive": raw_run / "allstr.arc",
            }
            if context_parameters.get("include_best_arc", False):
                source_paths["best-archive"] = raw_run / "best.arc"
            if context_parameters.get("include_md_arc", False):
                source_paths["md-archive"] = raw_run / "md.arc"

    for role, path in source_paths.items():
        if not contracts._ordinary_file(path):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "result.source_file", f"bound source file is missing: {role}"
                )
            )
    if contracts._ordinary_file(source_paths.get("ssw-archive")):
        try:
            source_frames = contracts._read_arc_frames(source_paths["ssw-archive"], int(max_frames))
        except (OSError, UnicodeError, ValueError) as exc:
            diagnostics.append(
                contracts._diagnostic("ERROR", "result.source_arc", f"cannot parse allstr.arc: {exc}")
            )
    if contracts._ordinary_file(source_paths.get("best-archive")):
        try:
            best_source_frames = contracts._read_arc_frames(source_paths["best-archive"], int(max_frames))
        except (OSError, UnicodeError, ValueError) as exc:
            diagnostics.append(
                contracts._diagnostic("ERROR", "result.best_arc", f"cannot parse best.arc: {exc}")
            )
    if contracts._ordinary_file(source_paths.get("md-archive")):
        try:
            md_source_frames = contracts._read_arc_frames(source_paths["md-archive"], int(max_frames))
        except (OSError, UnicodeError, ValueError) as exc:
            diagnostics.append(
                contracts._diagnostic("ERROR", "result.md_arc", f"cannot parse md.arc: {exc}")
            )
    if contracts._ordinary_file(lasp_input_path):
        try:
            parsed_lasp_input = contracts._parse_lasp_input(lasp_input_path)
            contracts._validate_ssw_input(parsed_lasp_input)
            if result_parameters.get("lasp_input") != parsed_lasp_input:
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        "result.lasp_input_parameters",
                        "parsed lasp.in parameters differ from the bound input",
                    )
                )
        except (OSError, UnicodeError, ValueError) as exc:
            diagnostics.append(
                contracts._diagnostic("ERROR", "result.lasp_input", f"cannot parse lasp.in: {exc}")
            )

    if len(role_paths.get("lasp-run-metadata", [])) == 1:
        value = contracts._read_json_object(
            role_paths["lasp-run-metadata"][0], "lasp_run_metadata", diagnostics
        )
        if value is not None:
            metadata = value
    for key, expected in (
        ("schema_version", 1),
        ("plugin_id", contracts.PLUGIN_ID),
        ("operation", operation),
        ("mode", expected_mode),
    ):
        if metadata.get(key) != expected:
            diagnostics.append(
                contracts._diagnostic("ERROR", "result.metadata_schema", f"metadata {key} is not bound")
            )
    if metadata.get("parameters") != dict(result_parameters):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "result.metadata_parameters",
                "metadata parameters differ from sampling-result parameters",
            )
        )
    source_metadata = metadata.get("source")
    raw_source_records = (
        source_metadata.get("files") if isinstance(source_metadata, Mapping) else None
    )
    if (
        not isinstance(source_metadata, Mapping)
        or source_metadata.get("id") != result.get("source_id")
        or not isinstance(raw_source_records, list)
    ):
        diagnostics.append(
            contracts._diagnostic("ERROR", "result.metadata_source", "metadata source record is invalid")
        )
        raw_source_records = []
    source_records: Dict[str, Mapping[str, Any]] = {}
    for item in raw_source_records:
        if not isinstance(item, Mapping) or not contracts._plain_string(item.get("role")):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "result.metadata_source_record", "source records are invalid"
                )
            )
            continue
        role = str(item["role"])
        if role in source_records:
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "result.metadata_source_duplicate", "source roles must be unique"
                )
            )
        source_records[role] = item
    if set(source_records) != set(source_paths):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "result.metadata_source_set",
                "metadata source files differ from approved source roles",
            )
        )
    for role, path in source_paths.items():
        record = source_records.get(role, {})
        if contracts._ordinary_file(path) and (
            record.get("name") != path.name or record.get("path") != str(path)
        ):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "result.metadata_source_path",
                    f"metadata does not match bound source file: {role}",
                )
            )

    execution_metadata = metadata.get("execution")
    if not isinstance(execution_metadata, Mapping):
        diagnostics.append(
            contracts._diagnostic("ERROR", "result.metadata_execution", "execution metadata is missing")
        )
        execution_metadata = {}
    if operation == "lasp-ssw-normalize-replay":
        if execution_metadata.get("performed") is not False:
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "result.metadata_replay", "replay must record performed=false"
                )
            )
    else:
        project_root = (
            contracts._resolve(context.get("project_root"), Path.cwd())
            if isinstance(context, Mapping)
            else Path.cwd().resolve()
        ) or Path.cwd().resolve()
        executable = contracts._resolve(context_inputs.get("lasp_executable"), project_root)
        input_structure = contracts._resolve(context_inputs.get("input_structure"), project_root)
        approved_lasp_input = contracts._resolve(context_inputs.get("lasp_input"), project_root)
        if not all(
            contracts._ordinary_file(path) for path in (executable, input_structure, approved_lasp_input)
        ):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "result.execute_inputs_required",
                    "execute checker requires the approved executable and staged inputs",
                )
            )
        else:
            assert executable is not None
            if (
                execution_metadata.get("performed") is not True
                or execution_metadata.get("version") != context_parameters.get("lasp_version")
                or execution_metadata.get("executable_path") != str(executable)
                or execution_metadata.get("returncode") != 0
                or execution_metadata.get("command_uses_shell") is not False
            ):
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        "result.executable",
                        "execution metadata does not match the approved LASP executable",
                    )
                )
        expected_staged: List[Dict[str, Any]] = []
        if contracts._ordinary_file(input_structure):
            assert input_structure is not None
            expected_staged.append(
                {
                    "role": "input-structure",
                    "destination": "input.arc",
                    "source": str(input_structure),
                }
            )
        if contracts._ordinary_file(approved_lasp_input):
            assert approved_lasp_input is not None
            expected_staged.append(
                {
                    "role": "lasp-input",
                    "destination": "lasp.in",
                    "source": str(approved_lasp_input),
                }
            )
        auxiliary = contracts._mapping(context_inputs.get("lasp_auxiliary_files"))
        for name in sorted(auxiliary):
            path = contracts._resolve(auxiliary[name], project_root)
            if not contracts._ordinary_file(path):
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        "result.auxiliary_input",
                        f"approved auxiliary {name} is missing",
                    )
                )
                continue
            assert path is not None
            expected_staged.append(
                {
                    "role": "auxiliary-input",
                    "destination": name,
                    "source": str(path),
                }
            )
        raw_staged = execution_metadata.get("staged_inputs")
        staged_valid = isinstance(raw_staged, list) and all(
            isinstance(item, Mapping) for item in raw_staged
        )
        normalized_staged = [dict(item) for item in raw_staged] if staged_valid else []
        if not staged_valid or sorted(
            normalized_staged, key=lambda item: str(item.get("destination"))
        ) != sorted(expected_staged, key=lambda item: item["destination"]):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "result.staged_inputs",
                    "staged input paths differ from the approved inputs",
                )
            )
        if staged_valid:
            for item in normalized_staged:
                destination = item.get("destination")
                if not isinstance(destination, str) or Path(destination).name != destination:
                    diagnostics.append(
                        contracts._diagnostic(
                            "ERROR",
                            "result.staged_input_path",
                            "staged input destinations must be safe basenames",
                        )
                    )
                    continue
                staged_path = raw_run / destination
                if not contracts._ordinary_file(staged_path):
                    diagnostics.append(
                        contracts._diagnostic(
                            "ERROR",
                            "result.staged_input",
                            f"staged input is missing: {destination}",
                        )
                    )
        launcher_value = context_resources.get("mpi_launcher")
        if launcher_value is None:
            launcher_path_value = None
        else:
            launcher_path = contracts._resolve(launcher_value, project_root)
            launcher_path_value = str(launcher_path) if contracts._ordinary_file(launcher_path) else None
        if execution_metadata.get(
            "mpi_launcher"
        ) != launcher_path_value or execution_metadata.get(
            "mpi_processes"
        ) != context_parameters.get("mpi_processes"):
            diagnostics.append(
                contracts._diagnostic("ERROR", "result.mpi", "MPI launcher settings differ from the plan")
            )

    structure_records: List[Dict[str, Any]] = []
    selected_records: List[Dict[str, Any]] = []
    if len(role_paths.get("ssw-structure-manifest", [])) == 1:
        value = contracts._read_json_object(
            role_paths["ssw-structure-manifest"][0], "ssw_structure_manifest", diagnostics
        )
        if value is not None:
            raw = value.get("structures")
            if (
                value.get("schema_version") != 1
                or value.get("source_id") != result.get("source_id")
                or not isinstance(raw, list)
            ):
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        "result.structure_manifest",
                        "SSW structure manifest schema or records are invalid",
                    )
                )
            else:
                structure_records = [item for item in raw if isinstance(item, dict)]
                if len(structure_records) != len(raw):
                    diagnostics.append(
                        contracts._diagnostic(
                            "ERROR", "result.structure_record", "SSW records must be objects"
                        )
                    )
    if len(role_paths.get("selected-structure-manifest", [])) == 1:
        value = contracts._read_json_object(
            role_paths["selected-structure-manifest"][0],
            "selected_structure_manifest",
            diagnostics,
        )
        if value is not None:
            raw = value.get("structures")
            if (
                value.get("schema_version") != 1
                or value.get("source_id") != result.get("source_id")
                or not isinstance(raw, list)
            ):
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        "result.selected_manifest",
                        "selected structure manifest schema or records are invalid",
                    )
                )
            else:
                selected_records = [item for item in raw if isinstance(item, dict)]
                if len(selected_records) != len(raw):
                    diagnostics.append(
                        contracts._diagnostic(
                            "ERROR",
                            "result.selected_record",
                            "selected records must be objects",
                        )
                    )

    if counts:
        if len(structure_records) != counts.get("generated_structure_count"):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "result.generated_count", "generated structure count mismatch"
                )
            )
        if len(source_frames) != counts.get("generated_structure_count"):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "result.source_generated_count",
                    "bound allstr.arc frame count differs from the result",
                )
            )
        if len(selected_records) != counts.get("selected_structure_count"):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "result.selected_count", "selected structure count mismatch"
                )
            )
        accepted = sum(
            1 for record in structure_records if record.get("energy_filter_pass") is True
        )
        if accepted != counts.get("energy_accepted_count"):
            diagnostics.append(
                contracts._diagnostic("ERROR", "result.accepted_count", "energy accepted count mismatch")
            )

    identifiers: List[str] = []
    selected_from_all: List[Dict[str, Any]] = []
    accepted_order = 0
    recomputed_selected_order = 0
    approved_stride = context_parameters.get("selection_stride")
    if not contracts._positive_int(approved_stride):
        approved_stride = 1
    for index, record in enumerate(structure_records):
        identifier = record.get("structure_id")
        frame_index = record.get("frame_index")
        energy = record.get("energy_ev")
        if not contracts._plain_string(identifier):
            diagnostics.append(
                contracts._diagnostic("ERROR", "result.structure_id", "structure_id is required")
            )
        else:
            identifiers.append(str(identifier))
        if frame_index != index + 1 or record.get("historical_order") != index + 1:
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "result.structure_order", "historical frame order is invalid"
                )
            )
        energy_valid = not (
            not isinstance(energy, (int, float))
            or isinstance(energy, bool)
            or not math.isfinite(float(energy))
        )
        if not energy_valid:
            diagnostics.append(
                contracts._diagnostic("ERROR", "result.energy", f"frame {index + 1} energy is invalid")
            )
        if (
            isinstance(frame_index, int)
            and not isinstance(frame_index, bool)
            and identifier != contracts._lasp_structure_id("ssw", frame_index)
        ):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "result.structure_id_order",
                    "structure_id does not match the frame order",
                )
            )
        if record.get("source_role") != "allstr.arc":
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "result.source_role", "SSW records must come from allstr.arc"
                )
            )
        if index >= len(source_frames) or any(
            record.get(key) != source_frames[index].get(key)
            for key in ("frame_index", "energy_ev")
        ):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "result.source_frame",
                    "SSW record differs from the corresponding bound allstr.arc frame",
                )
            )
        expected_accepted = bool(
            energy_valid
            and (expected_energy is None or float(energy) <= float(expected_energy))
        )
        if expected_accepted:
            accepted_order += 1
        expected_selected = (
            expected_accepted and (accepted_order - 1) % int(approved_stride) == 0
        )
        if expected_selected:
            recomputed_selected_order += 1
        if (
            record.get("energy_filter_pass") is not expected_accepted
            or record.get("energy_filter_order")
            != (accepted_order if expected_accepted else None)
            or record.get("selected") is not expected_selected
            or record.get("selected_order")
            != (recomputed_selected_order if expected_selected else None)
        ):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "result.selection_recomputed",
                    "energy filter, accepted order, or stride selection differs from the plan",
                )
            )
        if expected_selected:
            selected_from_all.append(record)
    if len(set(identifiers)) != len(identifiers):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "result.structure_id_duplicate", "structure IDs must be unique"
            )
        )

    generated_artifact_paths = {
        path.relative_to(manifest.parent.resolve()).as_posix(): path
        for path in role_paths.get("ssw-generated-structure", [])
    }
    if len(generated_artifact_paths) != len(structure_records):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "result.generated_artifacts",
                "generated structure artifact count does not match SSW records",
            )
        )
    for record in structure_records:
        path = generated_artifact_paths.get(record.get("structure_file"))
        if path is None:
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "result.generated_path",
                    "SSW structure_file is not a generated artifact",
                )
            )
        else:
            try:
                parsed = contracts._read_arc_frames(path, 1)
                frame_index = int(record.get("frame_index", 0))
                if (
                    len(parsed) != 1
                    or frame_index < 1
                    or frame_index > len(source_frames)
                    or parsed[0].get("energy_ev") != record.get("energy_ev")
                    or parsed[0].get("payload") != source_frames[frame_index - 1].get("payload")
                ):
                    raise ValueError("ARC structure does not match its record")
            except (OSError, UnicodeError, ValueError) as exc:
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        "result.generated_arc",
                        f"generated structure is not a bound LASP ARC frame: {exc}",
                    )
                )

    selected_ids = [record.get("structure_id") for record in selected_records]
    expected_ids = [
        record.get("structure_id")
        for record in sorted(selected_from_all, key=lambda item: item.get("selected_order", 0))
    ]
    if selected_ids != expected_ids:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "result.selection_records",
                "selected manifest does not match selected SSW records in historical order",
            )
        )
    selected_artifact_paths = {
        path.relative_to(manifest.parent.resolve()).as_posix(): path
        for path in role_paths.get("selected-structure", [])
    }
    if len(selected_artifact_paths) != len(selected_records):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "result.selected_artifacts",
                "selected structure artifact count does not match selected records",
            )
        )
    for order, record in enumerate(selected_records, 1):
        if record.get("selected_order") != order:
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "result.selected_order", "selected_order must be contiguous"
                )
            )
        path = selected_artifact_paths.get(record.get("output_file"))
        if path is None:
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "result.selected_path", "selected output_file is not an artifact"
                )
            )
        else:
            try:
                parsed = contracts._read_arc_frames(path, 1)
                frame_index = int(record.get("frame_index", 0))
                if (
                    len(parsed) != 1
                    or frame_index < 1
                    or frame_index > len(source_frames)
                    or parsed[0].get("energy_ev") != record.get("energy_ev")
                    or parsed[0].get("payload") != source_frames[frame_index - 1].get("payload")
                ):
                    raise ValueError("selected ARC does not match its record")
            except (OSError, UnicodeError, ValueError) as exc:
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        "result.selected_arc",
                        f"selected structure is not a bound LASP ARC frame: {exc}",
                    )
                )

    optional_manifest_counts = (
        (
            "aimd-seed-manifest",
            "aimd_seed_candidate_count",
            "aimd-seed-candidate",
            "aimd-seeds",
            "best.arc",
            best_source_frames,
        ),
        (
            "md-structure-manifest",
            "md_structure_count",
            "md-sampled-structure",
            "md-structures",
            "md.arc",
            md_source_frames,
        ),
    )
    for (
        role,
        count_key,
        structure_role,
        record_role,
        source_role,
        bound_frames,
    ) in optional_manifest_counts:
        expected_count = counts.get(count_key, 0) if counts else 0
        paths = role_paths.get(role, [])
        if expected_count == 0 and paths:
            diagnostics.append(
                contracts._diagnostic("ERROR", "result.optional_manifest", f"unexpected {role} artifact")
            )
        elif expected_count > 0:
            if len(paths) != 1:
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR", "result.optional_manifest", f"exactly one {role} is required"
                    )
                )
            else:
                value = contracts._read_json_object(paths[0], role.replace("-", "_"), diagnostics)
                records = value.get("structures") if isinstance(value, dict) else None
                if (
                    not isinstance(value, dict)
                    or value.get("schema_version") != 1
                    or value.get("source_id") != result.get("source_id")
                    or not isinstance(records, list)
                    or len(records) != expected_count
                ):
                    diagnostics.append(
                        contracts._diagnostic("ERROR", "result.optional_count", f"{role} count mismatch")
                    )
                else:
                    if len(bound_frames) != expected_count:
                        diagnostics.append(
                            contracts._diagnostic(
                                "ERROR",
                                "result.optional_source_count",
                                f"bound {source_role} frame count differs from result",
                            )
                        )
                    artifact_paths = {
                        path.relative_to(manifest.parent.resolve()).as_posix(): path
                        for path in role_paths.get(structure_role, [])
                    }
                    if len(artifact_paths) != expected_count:
                        diagnostics.append(
                            contracts._diagnostic(
                                "ERROR",
                                "result.optional_artifacts",
                                f"{structure_role} artifact count mismatch",
                            )
                        )
                    seen_optional_ids = set()
                    for order, record in enumerate(records, 1):
                        if not isinstance(record, Mapping):
                            diagnostics.append(
                                contracts._diagnostic(
                                    "ERROR",
                                    "result.optional_record",
                                    f"{role} records must be objects",
                                )
                            )
                            continue
                        identifier = record.get("structure_id")
                        path = artifact_paths.get(record.get("output_file"))
                        energy = record.get("energy_ev")
                        if not contracts._plain_string(identifier) or identifier in seen_optional_ids:
                            diagnostics.append(
                                contracts._diagnostic(
                                    "ERROR",
                                    "result.optional_id",
                                    f"{role} structure IDs must be unique",
                                )
                            )
                        elif isinstance(identifier, str):
                            seen_optional_ids.add(identifier)
                        expected_identifier = contracts._lasp_structure_id(record_role, order)
                        if (
                            identifier != expected_identifier
                            or record.get("source_role") != source_role
                        ):
                            diagnostics.append(
                                contracts._diagnostic(
                                    "ERROR",
                                    "result.optional_source",
                                    f"{role} ID or source role differs from its record order",
                                )
                            )
                        if (
                            record.get("frame_index") != order
                            or record.get("historical_order") != order
                            or not isinstance(energy, (int, float))
                            or isinstance(energy, bool)
                            or not math.isfinite(float(energy))
                        ):
                            diagnostics.append(
                                contracts._diagnostic(
                                    "ERROR",
                                    "result.optional_order",
                                    f"{role} order or energy is invalid",
                                )
                            )
                        if path is None:
                            diagnostics.append(
                                contracts._diagnostic(
                                    "ERROR",
                                    "result.optional_path",
                                    f"{role} output path is invalid",
                                )
                            )
                        else:
                            try:
                                parsed = contracts._read_arc_frames(path, 1)
                                if (
                                    len(parsed) != 1
                                    or order > len(bound_frames)
                                    or any(
                                        record.get(key) != bound_frames[order - 1].get(key)
                                        for key in ("frame_index", "energy_ev")
                                    )
                                    or parsed[0].get("energy_ev") != record.get("energy_ev")
                                    or parsed[0].get("payload")
                                    != bound_frames[order - 1].get("payload")
                                ):
                                    raise ValueError("ARC differs from bound source frame")
                            except (OSError, UnicodeError, ValueError) as exc:
                                diagnostics.append(
                                    contracts._diagnostic(
                                        "ERROR",
                                        "result.optional_arc",
                                        f"{role} structure is not bound to its source: {exc}",
                                    )
                                )
    return manifest, result, diagnostics, artifacts


def check(context: Any) -> Dict[str, Any]:
    operation = contracts._operation(context)
    execution = contracts._mapping(context.get("execution")) if isinstance(context, Mapping) else {}
    returncode = execution.get("returncode")
    if returncode is not None and returncode != 0:
        return {
            "plugin_id": contracts.PLUGIN_ID,
            "operation": operation,
            "status": "FAIL",
            "diagnostics": [
                contracts._diagnostic(
                    "ERROR", "execution.nonzero", f"LASP/SSW wrapper returned {returncode}"
                )
            ],
        }
    manifest, result, diagnostics, _ = _read_lasp_result(context)
    if manifest is not None and not manifest.exists() and not contracts._has_errors(diagnostics):
        return {
            "plugin_id": contracts.PLUGIN_ID,
            "operation": operation,
            "status": "WAIT",
            "diagnostics": diagnostics,
        }
    if contracts._has_errors(diagnostics):
        return {
            "plugin_id": contracts.PLUGIN_ID,
            "operation": operation,
            "status": "FAIL",
            "diagnostics": diagnostics,
        }
    return {
        "plugin_id": contracts.PLUGIN_ID,
        "operation": operation,
        "status": "OK",
        "result_file": str(manifest),
        "counts": result.get("counts", {}),
        "diagnostics": diagnostics,
    }


def collect(context: Any) -> Dict[str, Any]:
    operation = contracts._operation(context)
    manifest = contracts._manifest_path(context)
    assert manifest is not None
    result = json.loads(manifest.read_text(encoding="utf-8"))
    artifacts = [
        {"role": str(item["role"]), "path": str((manifest.parent / item["path"]).resolve()),
         "media_type": str(item.get("media_type", "application/octet-stream"))}
        for item in result["artifacts"]
    ]
    counts = contracts._mapping(result.get("counts"))
    return {
        "plugin_id": contracts.PLUGIN_ID,
        "operation": operation,
        "status": "OK",
        "artifacts": [
            {
                "role": "sampling-result",
                "path": str(manifest),
                "media_type": "application/json",
            },
            *artifacts,
        ],
        "metrics": {
            "generated_structure_count": counts.get("generated_structure_count", 0),
            "energy_accepted_count": counts.get("energy_accepted_count", 0),
            "selected_structure_count": counts.get("selected_structure_count", 0),
            "aimd_seed_candidate_count": counts.get("aimd_seed_candidate_count", 0),
            "md_structure_count": counts.get("md_structure_count", 0),
            "lasp_execution": operation == "lasp-ssw-execute",
            "lasp_replay": operation == "lasp-ssw-normalize-replay",
        },
        "diagnostics": [],
    }
