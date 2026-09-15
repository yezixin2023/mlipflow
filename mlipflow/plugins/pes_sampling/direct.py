"""pes sampling: direct."""

from __future__ import annotations

import csv
import io
import json
import re
import sys
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
                "DIRECT is exposed as a local argv only; scheduler wrapping belongs to the core",
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
    unknown_parameters = sorted(set(parameters) - contracts._DIRECT_PARAMETERS)
    if unknown_parameters:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "parameter.unknown",
                "unsupported parameter(s): %s" % ", ".join(unknown_parameters),
            )
        )
    if not isinstance(context.get("resources", {}), Mapping):
        diagnostics.append(
            contracts._diagnostic("ERROR", "resources.mapping_required", "resources must be a mapping")
        )
    unknown_resources = sorted(set(resources) - {"python_executable"})
    if unknown_resources:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "resource.unknown",
                "unsupported DIRECT resource(s): %s" % ", ".join(unknown_resources),
            )
        )

    unknown_inputs = sorted(set(inputs) - {"input_dirs", "result_manifest"})
    if unknown_inputs:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "input.unknown",
                "unsupported DIRECT input(s): %s" % ", ".join(unknown_inputs),
            )
        )
    if not contracts.BUNDLED_DIRECT_WRAPPER.is_file():
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "path.bundled_direct_wrapper",
                "the bundled DIRECT runner is missing or is not an ordinary file",
            )
        )

    raw_input_dirs = inputs.get("input_dirs")
    input_dirs: List[Path] = []
    if not isinstance(raw_input_dirs, list) or not raw_input_dirs:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "path.input_dirs", "inputs.input_dirs must be a non-empty list"
            )
        )
    else:
        for index, raw_path in enumerate(raw_input_dirs):
            resolved = contracts._resolve(raw_path, project_root)
            if resolved is None or not resolved.is_dir():
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        "path.input_dir",
                        "input_dirs[%d] must name an existing directory" % index,
                    )
                )
            else:
                input_dirs.append(resolved)

    output_subdir = parameters.get("output_subdir", "direct-selected")
    if not contracts._safe_subdirectory(output_subdir):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "path.output_subdir",
                "output_subdir must be a non-empty relative path without '..'",
            )
        )
        output_dir = attempt_dir / "direct-selected"
    else:
        output_dir = (attempt_dir / str(output_subdir)).resolve()
        if not contracts._is_within(output_dir, attempt_dir):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "path.output_escape", "DIRECT output must stay inside attempt_dir"
                )
            )
    if output_dir.exists():
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "path.output_exists",
                "the bundled DIRECT runner requires a fresh attempt/output_subdir",
            )
        )
    for input_dir in input_dirs:
        if contracts._is_within(output_dir, input_dir) or contracts._is_within(input_dir, output_dir):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "path.input_output_overlap",
                    "input and output directories must not overlap",
                )
            )
            break

    globs = parameters.get("globs")
    if (
        not isinstance(globs, list)
        or not globs
        or not all(isinstance(item, str) and item.strip() for item in globs)
    ):
        diagnostics.append(
            contracts._diagnostic("ERROR", "parameter.globs", "globs must be a non-empty string list")
        )
    elif any(Path(item).is_absolute() or ".." in Path(item).parts for item in globs):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "parameter.glob_escape", "globs must be relative patterns without '..'"
            )
        )

    if not isinstance(parameters.get("recursive", True), bool):
        diagnostics.append(
            contracts._diagnostic("ERROR", "parameter.recursive", "recursive must be boolean")
        )
    for name in ("stride", "n_clusters", "k_per_cluster"):
        if not contracts._positive_int(parameters.get(name)):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "parameter.%s" % name, "%s must be a positive integer" % name
                )
            )
    for name in ("max_frames_per_file", "max_input_structures"):
        value = parameters.get(name)
        if value is not None and not contracts._positive_int(value):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "parameter.%s" % name,
                    "%s must be null or a positive integer" % name,
                )
            )
    if not contracts._positive_number(parameters.get("threshold_init")):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "parameter.threshold_init",
                "threshold_init must be a finite positive number",
            )
        )

    seed = parameters.get("seed")
    if not contracts._nonnegative_int(seed):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "parameter.seed", "seed must be an explicit non-negative integer"
            )
        )
    if parameters.get("acknowledge_uncontrolled_seed") is not True:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "seed.no_cli_control",
                "MAML DIRECT exposes no seed control in this selection path; set acknowledge_uncontrolled_seed=true to accept this limitation",
            )
        )
    else:
        diagnostics.append(
            contracts._diagnostic(
                "WARNING",
                "seed.provenance_only",
                "seed is recorded in the plan but cannot be passed to the bundled MAML DIRECT selection path",
            )
        )

    lammps_map = parameters.get("lammps_type_map")
    if lammps_map is not None:
        if not isinstance(lammps_map, Mapping) or not lammps_map:
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "parameter.lammps_type_map",
                    "lammps_type_map must be a non-empty mapping",
                )
            )
        else:
            for atom_type, symbol in lammps_map.items():
                try:
                    numeric_type = int(atom_type)
                except (TypeError, ValueError):
                    numeric_type = 0
                if (
                    numeric_type < 1
                    or not isinstance(symbol, str)
                    or not re.fullmatch(r"[A-Z][a-z]?", symbol)
                ):
                    diagnostics.append(
                        contracts._diagnostic(
                            "ERROR",
                            "parameter.lammps_type_map_entry",
                            "LAMMPS types must be positive integers mapped to element symbols",
                        )
                    )
                    break

    python_executable = resources.get("python_executable", sys.executable)
    if not isinstance(python_executable, str) or not python_executable.strip():
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "resource.python_executable",
                "python_executable must be a non-empty string",
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
    script = contracts.BUNDLED_DIRECT_WRAPPER.resolve()
    input_dirs = [contracts._resolve(value, project_root) for value in inputs["input_dirs"]]
    output_dir = (
        attempt_dir / str(parameters.get("output_subdir", "direct-selected"))
    ).resolve()

    argv = [
        str(resources.get("python_executable", sys.executable)),
        str(script),
        "--input-dirs",
    ]
    argv.extend(str(path) for path in input_dirs)
    argv.extend(["--globs"])
    argv.extend(str(value) for value in parameters["globs"])
    if parameters.get("recursive", True) is False:
        argv.append("--no-recursive")
    argv.extend(["--stride", str(parameters["stride"])])
    if parameters.get("max_frames_per_file") is not None:
        argv.extend(["--max-frames-per-file", str(parameters["max_frames_per_file"])])
    if parameters.get("max_input_structures") is not None:
        argv.extend(["--max-input-structures", str(parameters["max_input_structures"])])
    argv.extend(["--n-clusters", str(parameters["n_clusters"])])
    argv.extend(["--threshold-init", str(parameters["threshold_init"])])
    argv.extend(["--k-per-cluster", str(parameters["k_per_cluster"])])
    argv.extend(["--out-dir", str(output_dir)])
    if parameters.get("lammps_type_map") is not None:
        encoded = json.dumps(
            {str(key): value for key, value in parameters["lammps_type_map"].items()},
            sort_keys=True,
            separators=(",", ":"),
        )
        argv.extend(["--lammps-type-map", encoded])

    return {
        "plugin_id": contracts.PLUGIN_ID,
        "operation": "direct-select",
        "status": "READY",
        "executable": True,
        "argv": argv,
        "cwd": str(attempt_dir),
        "shell": False,
        "expected_outputs": [str(output_dir / "manifest.csv")],
        "output_dir": str(output_dir),
        "input_paths": {
            "direct_wrapper": str(script),
            "input_directories": [str(path) for path in input_dirs],
        },
        "assumptions": {
            "seed": parameters["seed"],
            "seed_control": "not-exposed-by-bundled-maml-interface",
            "selection_implementation": "maml.sampling.direct.DIRECTSampler",
            "fresh_output_directory_required": True,
        },
        "diagnostics": diagnostics,
    }


def _read_direct_manifest(
    context: Any
) -> Tuple[Optional[Path], List[Dict[str, str]], List[Dict[str, str]]]:
    path = contracts._manifest_path(context)
    diagnostics: List[Dict[str, str]] = []
    if path is None:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "result.manifest_explicit_required",
                "result manifest must come from execution.plan.expected_outputs or inputs.result_manifest",
            )
        )
        return None, [], diagnostics
    if not path.is_file():
        diagnostics.append(
            contracts._diagnostic("INFO", "result.manifest_missing", "DIRECT manifest does not exist yet")
        )
        return path, [], diagnostics
    try:
        reader = csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")))
        rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "result.manifest_unreadable", "cannot read manifest.csv: %s" % exc
            )
        )
        return path, [], diagnostics
    if reader.fieldnames is None or not contracts._DIRECT_COLUMNS.issubset(set(reader.fieldnames)):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "result.manifest_columns", "manifest.csv has an unexpected header"
            )
        )
    if not rows:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "result.no_samples", "manifest.csv contains no selected structures"
            )
        )
    manifest_dir = path.parent.resolve()
    seen_orders = set()
    for index, row in enumerate(rows):
        try:
            order = int(row.get("selected_order", ""))
            int(row.get("input_index", ""))
            int(row.get("frame_index", ""))
        except (TypeError, ValueError):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "result.manifest_integer",
                    "manifest row %d has invalid integer fields" % (index + 1),
                )
            )
            continue
        if order < 1 or order in seen_orders:
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "result.selected_order",
                    "selected_order values must be unique and positive",
                )
            )
        seen_orders.add(order)
        output_file = contracts._resolve(row.get("output_file"), manifest_dir)
        if output_file is None or not contracts._is_within(output_file, manifest_dir):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "result.output_escape",
                    "selected output must stay beside manifest.csv",
                )
            )
        elif not output_file.is_file() or output_file.stat().st_size == 0:
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "result.output_missing",
                    "selected structure is missing or empty: %s" % output_file,
                )
            )
    return path, rows, diagnostics


def check(context: Any) -> Dict[str, Any]:
    execution = contracts._mapping(context.get("execution")) if isinstance(context, Mapping) else {}
    returncode = execution.get("returncode")
    if returncode is not None and returncode != 0:
        return {
            "plugin_id": contracts.PLUGIN_ID,
            "status": "FAIL",
            "diagnostics": [
                contracts._diagnostic(
                    "ERROR", "execution.nonzero", "DIRECT process returned %s" % returncode
                )
            ],
        }
    path, rows, diagnostics = _read_direct_manifest(context)
    if path is not None and not path.exists() and not contracts._has_errors(diagnostics):
        return {"plugin_id": contracts.PLUGIN_ID, "status": "WAIT", "diagnostics": diagnostics}
    if contracts._has_errors(diagnostics):
        return {"plugin_id": contracts.PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
    return {
        "plugin_id": contracts.PLUGIN_ID,
        "status": "OK",
        "result_file": str(path),
        "selected_structure_count": len(rows),
        "diagnostics": diagnostics,
    }


def collect(context: Any) -> Dict[str, Any]:
    checked = check(context)
    if checked.get("status") != "OK":
        return {
            "plugin_id": contracts.PLUGIN_ID,
            "status": checked.get("status", "FAIL"),
            "artifacts": [],
            "metrics": {},
            "diagnostics": checked.get("diagnostics", []),
        }
    manifest, rows, diagnostics = _read_direct_manifest(context)
    assert manifest is not None
    artifacts: List[Dict[str, str]] = [
        {"role": "sample-manifest", "path": str(manifest), "media_type": "text/csv"}
    ]
    formulas = set()
    sources = set()
    manifest_dir = manifest.parent.resolve()
    for row in rows:
        output_file = contracts._resolve(row["output_file"], manifest_dir)
        assert output_file is not None
        artifacts.append(
            {
                "role": "selected-structure",
                "path": str(output_file),
                "media_type": "chemical/x-vasp-poscar",
            }
        )
        formulas.add(row.get("formula", ""))
        sources.add(row.get("source_file", ""))
    for name in contracts._OPTIONAL_PLOTS:
        path = manifest_dir / name
        if path.is_file():
            artifacts.append(
                {
                    "role": "sampling-diagnostic",
                    "path": str(path),
                    "media_type": "text/html" if path.suffix == ".html" else "image/png",
                }
            )
    return {
        "plugin_id": contracts.PLUGIN_ID,
        "status": "OK",
        "artifacts": artifacts,
        "metrics": {
            "selected_structure_count": len(rows),
            "unique_formula_count": len(formulas),
            "source_file_count": len(sources),
        },
        "diagnostics": diagnostics,
    }
