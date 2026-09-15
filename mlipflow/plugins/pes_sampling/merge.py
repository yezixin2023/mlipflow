"""pes sampling: merge."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from . import contracts


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
    if attempt_dir is None or not contracts._is_within(attempt_dir, project_root):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "path.attempt_dir", "attempt_dir must stay inside project_root"
            )
        )
        attempt_dir = project_root / ".mlipflow-invalid-attempt"
    if context.get("backend", "local") != "local":
        diagnostics.append(
            contracts._diagnostic("ERROR", "backend.local_only", "merge-structures is local-only")
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
    unknown_inputs = sorted(
        set(inputs)
        - {
            "direct_manifest",
            "lasp_selected_manifest",
            "lasp_selected_archive",
            "result_manifest",
        }
    )
    if unknown_inputs:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "input.unknown",
                "unsupported merge input(s): " + ", ".join(unknown_inputs),
            )
        )
    for name in ("direct_manifest", "lasp_selected_manifest", "lasp_selected_archive"):
        path = contracts._resolve(inputs.get(name), project_root)
        if not contracts._ordinary_file(path) or path is None or not contracts._is_within(path, project_root):
            diagnostics.append(
                contracts._diagnostic("ERROR", f"path.{name}", f"{name} must be an ordinary project file")
            )
    unknown_parameters = sorted(set(parameters) - contracts._MERGE_PARAMETERS)
    if unknown_parameters:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "parameter.unknown",
                "unsupported merge parameter(s): " + ", ".join(unknown_parameters),
            )
        )
    output_subdir = parameters.get("output_subdir", "merged-structures")
    if not contracts._safe_subdirectory(output_subdir):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "path.output_subdir", "output_subdir must be a safe relative path"
            )
        )
    else:
        output_dir = (attempt_dir / str(output_subdir)).resolve()
        if not contracts._is_within(output_dir, attempt_dir):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "path.output_escape", "merge output must stay inside attempt_dir"
                )
            )
        elif output_dir.exists():
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "path.output_exists", "merge-structures requires fresh output"
                )
            )
    for name in ("direct_source_group_id", "lasp_source_group_id"):
        value = parameters.get(name)
        if not isinstance(value, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._+-]*", value
        ):
            diagnostics.append(
                contracts._diagnostic("ERROR", f"parameter.{name}", f"{name} must be a portable ID")
            )
    numeric_ranges = {
        "matcher_ltol": (0.0, 1.0),
        "matcher_stol": (0.0, 1.0),
        "matcher_angle_tol_deg": (0.0, 30.0),
        "minimum_distance_angstrom": (0.0, 5.0),
    }
    for name, (lower, upper) in numeric_ranges.items():
        value = parameters.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not lower < float(value) <= upper
        ):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", f"parameter.{name}", f"{name} must be in ({lower}, {upper}]"
                )
            )
    max_structures = parameters.get("max_structures")
    if not contracts._positive_int(max_structures) or int(max_structures) > 10000:
        diagnostics.append(
            contracts._diagnostic("ERROR", "parameter.max_structures", "max_structures must be 1..10000")
        )
    if set(resources) != {"python_executable"}:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "resource.python_executable",
                "merge resources must contain only an explicit python_executable",
            )
        )
    elif contracts._explicit_executable(resources.get("python_executable"), project_root) is None:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "resource.python_executable",
                "python_executable must be an absolute executable file",
            )
        )
    if not contracts._ordinary_file(contracts.BUNDLED_MERGE_WRAPPER):
        diagnostics.append(
            contracts._diagnostic("ERROR", "path.bundled_merge_wrapper", "structure_merge.py is missing")
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
    resources = contracts._mapping(context["resources"])
    assert project_root is not None and attempt_dir is not None
    direct = contracts._resolve(inputs["direct_manifest"], project_root)
    lasp_manifest = contracts._resolve(inputs["lasp_selected_manifest"], project_root)
    lasp_archive = contracts._resolve(inputs["lasp_selected_archive"], project_root)
    python_executable = contracts._explicit_executable(resources["python_executable"], project_root)
    assert direct is not None and lasp_manifest is not None and lasp_archive is not None
    assert python_executable is not None
    output_dir = (
        attempt_dir / str(parameters.get("output_subdir", "merged-structures"))
    ).resolve()
    script = contracts.BUNDLED_MERGE_WRAPPER.resolve()
    argv = [
        str(python_executable),
        str(script),
        "--direct-manifest",
        str(direct),
        "--lasp-selected-manifest",
        str(lasp_manifest),
        "--lasp-selected-archive",
        str(lasp_archive),
        "--output-dir",
        str(output_dir),
        "--direct-source-group-id",
        str(parameters["direct_source_group_id"]),
        "--lasp-source-group-id",
        str(parameters["lasp_source_group_id"]),
        "--matcher-ltol",
        str(parameters["matcher_ltol"]),
        "--matcher-stol",
        str(parameters["matcher_stol"]),
        "--matcher-angle-tol-deg",
        str(parameters["matcher_angle_tol_deg"]),
        "--minimum-distance-angstrom",
        str(parameters["minimum_distance_angstrom"]),
        "--max-structures",
        str(parameters["max_structures"]),
    ]
    return {
        "plugin_id": contracts.PLUGIN_ID,
        "operation": contracts.MERGE_OPERATION,
        "status": "READY",
        "executable": True,
        "argv": argv,
        "cwd": str(attempt_dir),
        "shell": False,
        "expected_outputs": [str(output_dir / "structures.json")],
        "output_dir": str(output_dir),
        "approval_summary": {
            "expensive": False,
            "submits_jobs": False,
            "input_structure_bound": parameters["max_structures"],
            "near_duplicate_matcher": {
                "ltol": parameters["matcher_ltol"],
                "stol": parameters["matcher_stol"],
                "angle_tol_deg": parameters["matcher_angle_tol_deg"],
                "scale": False,
            },
            "minimum_distance_angstrom": parameters["minimum_distance_angstrom"],
        },
        "input_paths": {
            "direct_manifest": str(direct),
            "lasp_selected_manifest": str(lasp_manifest),
            "lasp_selected_archive": str(lasp_archive),
            "merge_wrapper": str(script),
            "python_executable": str(python_executable),
        },
        "diagnostics": diagnostics,
    }


def _read_merge_manifest(
    context: Any
) -> Tuple[Optional[Path], Dict[str, Any], List[Dict[str, str]], List[Dict[str, Any]]]:
    path = contracts._manifest_path(context)
    diagnostics: List[Dict[str, str]] = []
    if path is None:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "result.manifest_explicit_required",
                "merge result manifest must come from the execution plan or result_manifest",
            )
        )
        return None, {}, diagnostics, []
    if not path.is_file():
        diagnostics.append(
            contracts._diagnostic(
                "INFO", "result.manifest_missing", "merge structures.json does not exist yet"
            )
        )
        return path, {}, diagnostics, []
    manifest = contracts._read_json_object(path, "merge_manifest", diagnostics)
    if manifest is None:
        return path, {}, diagnostics, []
    if (
        manifest.get("schema_version") != 1
        or manifest.get("operation") != contracts.MERGE_OPERATION
        or manifest.get("status") != "OK"
    ):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "result.merge_contract", "merge manifest schema/status is invalid"
            )
        )
    records = manifest.get("structures")
    if not isinstance(records, list) or not records:
        diagnostics.append(
            contracts._diagnostic("ERROR", "result.structures", "merge manifest must contain structures")
        )
        records = []
    counts = contracts._mapping(manifest.get("counts"))
    duplicates = manifest.get("duplicates")
    rejected = manifest.get("rejected")
    if not isinstance(duplicates, list) or not isinstance(rejected, list):
        diagnostics.append(
            contracts._diagnostic("ERROR", "result.lineage_lists", "duplicates/rejected must be lists")
        )
        duplicates = []
        rejected = []
    expected_input = len(records) + len(duplicates) + len(rejected)
    if (
        counts.get("input") != expected_input
        or counts.get("unique") != len(records)
        or counts.get("rejected_bad") != len(rejected)
        or counts.get("exact_duplicates")
        != sum(
            isinstance(item, Mapping) and item.get("duplicate_kind") == "exact"
            for item in duplicates
        )
        or counts.get("near_duplicates")
        != sum(
            isinstance(item, Mapping) and item.get("duplicate_kind") == "near"
            for item in duplicates
        )
    ):
        diagnostics.append(
            contracts._diagnostic("ERROR", "result.counts", "merge counts do not match manifest records")
        )
    input_artifacts = contracts._mapping(manifest.get("input_artifacts"))
    inputs = contracts._mapping(context.get("inputs")) if isinstance(context, Mapping) else {}
    project_root = (
        contracts._resolve(context.get("project_root"), Path.cwd())
        if isinstance(context, Mapping)
        else None
    )
    if project_root is not None:
        for key in ("direct_manifest", "lasp_selected_manifest", "lasp_selected_archive"):
            source = contracts._resolve(inputs.get(key), project_root)
            if (
                source is None
                or not contracts._ordinary_file(source)
                or input_artifacts.get(key) != str(source)
            ):
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR", f"result.input_{key}", f"merge input {key} path differs"
                    )
                )
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for index, record in enumerate(records, 1):
        prefix = f"structure {index}"
        if not isinstance(record, Mapping):
            diagnostics.append(
                contracts._diagnostic("ERROR", "result.structure_record", f"{prefix} is invalid")
            )
            continue
        structure_id = record.get("id")
        relative = record.get("path")
        methods = record.get("source_sampling_methods")
        sources = record.get("source_records")
        if structure_id != f"structure-{index:06d}" or structure_id in seen_ids:
            diagnostics.append(
                contracts._diagnostic("ERROR", "result.structure_id", f"{prefix} id is invalid")
            )
        else:
            seen_ids.add(structure_id)
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in seen_paths
        ):
            diagnostics.append(
                contracts._diagnostic("ERROR", "result.structure_path", f"{prefix} path is invalid")
            )
            continue
        seen_paths.add(relative)
        output = (path.parent / relative).absolute()
        if (
            not contracts._is_within(output, path.parent.resolve())
            or not output.is_file()
            or output.stat().st_size < 1
        ):
            diagnostics.append(
                contracts._diagnostic("ERROR", "result.structure_file", f"{prefix} file is invalid")
            )
        if (
            not isinstance(methods, list)
            or not methods
            or any(method not in {"DIRECT", "LASP_SSW"} for method in methods)
            or not isinstance(sources, list)
            or not sources
            or sorted(
                {item.get("sampling_method") for item in sources if isinstance(item, Mapping)}
            )
            != methods
        ):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "result.source_provenance",
                    f"{prefix} source provenance is invalid",
                )
            )
    if (path.parent / "INCOMPLETE.json").exists():
        diagnostics.append(
            contracts._diagnostic("ERROR", "result.incomplete_marker", "merge output remains incomplete")
        )
    return (
        path,
        manifest,
        diagnostics,
        [dict(item) for item in records if isinstance(item, Mapping)],
    )


def check(context: Any) -> Dict[str, Any]:
    execution = contracts._mapping(context.get("execution")) if isinstance(context, Mapping) else {}
    returncode = execution.get("returncode")
    if returncode is not None and returncode != 0:
        return {
            "plugin_id": contracts.PLUGIN_ID,
            "operation": contracts.MERGE_OPERATION,
            "status": "FAIL",
            "diagnostics": [
                contracts._diagnostic(
                    "ERROR", "execution.nonzero", f"merge-structures returned {returncode}"
                )
            ],
        }
    path, manifest, diagnostics, records = _read_merge_manifest(context)
    if path is not None and not path.exists() and not contracts._has_errors(diagnostics):
        return {
            "plugin_id": contracts.PLUGIN_ID,
            "operation": contracts.MERGE_OPERATION,
            "status": "WAIT",
            "diagnostics": diagnostics,
        }
    if contracts._has_errors(diagnostics):
        return {
            "plugin_id": contracts.PLUGIN_ID,
            "operation": contracts.MERGE_OPERATION,
            "status": "FAIL",
            "diagnostics": diagnostics,
        }
    return {
        "plugin_id": contracts.PLUGIN_ID,
        "operation": contracts.MERGE_OPERATION,
        "status": "OK",
        "result_file": str(path),
        "counts": manifest.get("counts", {}),
        "structure_count": len(records),
        "diagnostics": diagnostics,
    }


def collect(context: Any) -> Dict[str, Any]:
    checked = check(context)
    if checked.get("status") != "OK":
        return {
            "plugin_id": contracts.PLUGIN_ID,
            "operation": contracts.MERGE_OPERATION,
            "status": checked.get("status", "FAIL"),
            "artifacts": [],
            "metrics": {},
            "diagnostics": checked.get("diagnostics", []),
        }
    manifest, value, diagnostics, records = _read_merge_manifest(context)
    assert manifest is not None
    artifacts: List[Dict[str, str]] = [
        {"role": "structures-manifest", "path": str(manifest), "media_type": "application/json"}
    ]
    for record in records:
        artifacts.append(
            {
                "role": "merged-structure",
                "path": str((manifest.parent / record["path"]).resolve()),
                "media_type": "chemical/x-vasp-poscar",
            }
        )
    counts = contracts._mapping(value.get("counts"))
    return {
        "plugin_id": contracts.PLUGIN_ID,
        "operation": contracts.MERGE_OPERATION,
        "status": "OK",
        "artifacts": artifacts,
        "metrics": {
            "input_structure_count": counts.get("input", 0),
            "unique_structure_count": counts.get("unique", 0),
            "exact_duplicate_count": counts.get("exact_duplicates", 0),
            "near_duplicate_count": counts.get("near_duplicates", 0),
            "rejected_bad_structure_count": counts.get("rejected_bad", 0),
        },
        "diagnostics": diagnostics,
    }
