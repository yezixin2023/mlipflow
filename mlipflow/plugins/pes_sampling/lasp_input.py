"""pes sampling: lasp input."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

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
            contracts._diagnostic("ERROR", "backend.local_only", "lasp-input-prepare is local-only")
        )

    inputs = contracts._mapping(context.get("inputs"))
    parameters = contracts._mapping(context.get("parameters"))
    resources = contracts._mapping(context.get("resources"))
    unknown_inputs = sorted(
        set(inputs)
        - {
            "input_structure",
            "pseudopotential_reference",
            "result_manifest",
        }
    )
    unknown_parameters = sorted(set(parameters) - contracts._LASP_INPUT_PARAMETERS)
    unknown_resources = sorted(set(resources) - {"python_executable"})
    for code, values in (
        ("input.unknown", unknown_inputs),
        ("parameter.unknown", unknown_parameters),
        ("resource.unknown", unknown_resources),
    ):
        if values:
            diagnostics.append(
                contracts._diagnostic("ERROR", code, "unsupported value(s): %s" % ", ".join(values))
            )

    source = contracts._resolve(inputs.get("input_structure"), project_root)
    if source is None:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "path.input_structure",
                "input_structure must be a non-empty path",
            )
        )
    elif not source.exists():
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "path.input_structure",
                f"input_structure file does not exist: {source}",
            )
        )
    elif not contracts._ordinary_file(source):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "path.input_structure",
                f"input_structure must be an ordinary file: {source}",
            )
        )
    else:
        try:
            with source.open("rb") as stream:
                stream.read(1)
        except OSError:
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "path.input_structure",
                    f"input_structure must be readable: {source}",
                )
            )
    pseudopotential = inputs.get("pseudopotential_reference")
    if pseudopotential is not None:
        reference_path = contracts._resolve(pseudopotential, project_root)
        if (
            not contracts._ordinary_file(reference_path)
            or reference_path is None
            or not contracts._is_within(reference_path, project_root)
        ):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "path.pseudopotential_reference",
                    "pseudopotential_reference must be an ordinary project file",
                )
            )
        else:
            try:
                contracts._pseudopotential_reference(reference_path)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        "input.pseudopotential_reference",
                        str(exc),
                    )
                )
    if not contracts._ordinary_file(contracts.BUNDLED_LASP_INPUT_WRAPPER):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "path.lasp_input_wrapper",
                "bundled LASP input converter is missing",
            )
        )
    python_executable = contracts._explicit_executable(resources.get("python_executable"), project_root)
    if python_executable is None:
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "resource.python_executable",
                "python_executable must be an explicit ordinary executable",
            )
        )
    output_subdir = parameters.get("output_subdir", "lasp-input")
    if not contracts._safe_subdirectory(output_subdir):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "path.output_subdir", "output_subdir must be a safe relative path"
            )
        )
        output_dir = attempt_dir / "lasp-input"
    else:
        output_dir = (attempt_dir / str(output_subdir)).resolve()
    if (
        not contracts._is_within(output_dir, attempt_dir)
        or output_dir.exists()
        or output_dir.is_symlink()
    ):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR", "path.output_exists", "LASP input conversion requires a fresh output"
            )
        )
    for name in ("input_format", "input_index"):
        value = parameters.get(name)
        if value is not None and not contracts._plain_string(str(value)):
            diagnostics.append(
                contracts._diagnostic("ERROR", f"parameter.{name}", f"{name} must be a plain string")
            )
    minimum = parameters.get("minimum_cell_length_angstrom")
    if minimum is not None and not contracts._positive_number(minimum):
        diagnostics.append(
            contracts._diagnostic(
                "ERROR",
                "parameter.minimum_cell_length_angstrom",
                "minimum_cell_length_angstrom must be finite and positive",
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
    resources = contracts._mapping(context["resources"])
    assert project_root is not None and attempt_dir is not None
    source = contracts._resolve(inputs["input_structure"], project_root)
    python_executable = contracts._explicit_executable(resources["python_executable"], project_root)
    assert source is not None and python_executable is not None
    output_dir = (attempt_dir / str(parameters.get("output_subdir", "lasp-input"))).resolve()
    wrapper = contracts.BUNDLED_LASP_INPUT_WRAPPER.resolve()
    argv = [
        str(python_executable),
        str(wrapper),
        "--input-structure",
        str(source),
        "--output-dir",
        str(output_dir),
        "--input-index",
        str(parameters.get("input_index", "-1")),
    ]
    if parameters.get("input_format") is not None:
        argv.extend(["--input-format", str(parameters["input_format"])])
    if parameters.get("minimum_cell_length_angstrom") is not None:
        argv.extend(
            [
                "--minimum-cell-length-angstrom",
                str(parameters["minimum_cell_length_angstrom"]),
            ]
        )
    reference_path = contracts._resolve(inputs.get("pseudopotential_reference"), project_root)
    reference = None
    if reference_path is not None:
        reference = contracts._pseudopotential_reference(reference_path)
        argv.extend(["--pseudopotential-reference", str(reference_path)])
    conversion = {
        "source_path": str(source),
        "input_format": parameters.get("input_format"),
        "input_index": str(parameters.get("input_index", "-1")),
        "minimum_cell_length_angstrom": parameters.get("minimum_cell_length_angstrom"),
        "pseudopotential_reference_path": (
            str(reference_path) if reference_path is not None else None
        ),
        "pseudopotential": reference,
    }
    return {
        "plugin_id": contracts.PLUGIN_ID,
        "operation": contracts.LASP_INPUT_OPERATION,
        "status": "READY",
        "executable": True,
        "argv": argv,
        "cwd": str(attempt_dir),
        "shell": False,
        "expected_outputs": [str(output_dir / "lasp-input-manifest.json")],
        "output_dir": str(output_dir),
        "conversion": conversion,
        "approval_summary": {
            "expensive": False,
            "submits_jobs": False,
            "materializes_licensed_potcar": reference is not None,
            "potcar_collectable": False if reference is not None else None,
            **conversion,
        },
        "input_paths": {
            "source_structure": str(source),
            "conversion_wrapper": str(wrapper),
            "python_executable": str(python_executable),
            **(
                {"pseudopotential_reference": str(reference_path)}
                if reference_path is not None
                else {}
            ),
        },
        "diagnostics": diagnostics,
    }


def check(context: Any) -> Dict[str, Any]:
    execution = contracts._mapping(context.get("execution")) if isinstance(context, Mapping) else {}
    if execution.get("returncode") not in (None, 0):
        return {
            "plugin_id": contracts.PLUGIN_ID,
            "operation": contracts.LASP_INPUT_OPERATION,
            "status": "FAIL",
            "diagnostics": [
                contracts._diagnostic(
                    "ERROR", "execution.nonzero", "LASP input conversion returned nonzero"
                )
            ],
        }
    project_root = contracts._resolve(context.get("project_root"), Path.cwd()) or Path.cwd()
    plan = contracts._mapping(execution.get("plan"))
    expected = plan.get("expected_outputs")
    path = (
        contracts._unresolved(expected[0], project_root)
        if isinstance(expected, list) and len(expected) == 1
        else contracts._unresolved(contracts._mapping(context.get("inputs")).get("result_manifest"), project_root)
    )
    diagnostics: List[Dict[str, str]] = []
    if not contracts._ordinary_file(path):
        diagnostics.append(
            contracts._diagnostic("ERROR", "result.manifest", "LASP input manifest is missing")
        )
        value: Dict[str, Any] = {}
    else:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            value = dict(loaded) if isinstance(loaded, Mapping) else {}
        except (OSError, UnicodeError, json.JSONDecodeError):
            value = {}
        if not value:
            diagnostics.append(
                contracts._diagnostic("ERROR", "result.manifest", "LASP input manifest is invalid")
            )
    conversion = contracts._mapping(plan.get("conversion"))
    if value:
        source = contracts._mapping(value.get("source"))
        for key, expected_value in (
            ("path", conversion.get("source_path")),
            ("input_format", conversion.get("input_format")),
            ("input_index", conversion.get("input_index")),
        ):
            if source.get(key) != expected_value:
                diagnostics.append(
                    contracts._diagnostic("ERROR", f"result.source_{key}", f"source {key} differs")
                )
        structure = contracts._mapping(value.get("structure"))
        lengths = structure.get("cell_lengths_A")
        minimum = conversion.get("minimum_cell_length_angstrom")
        if (
            value.get("schema_version") != 1
            or value.get("plugin_id") != contracts.PLUGIN_ID
            or value.get("operation") != contracts.LASP_INPUT_OPERATION
            or not contracts._positive_int(structure.get("atom_count"))
            or not isinstance(lengths, list)
            or len(lengths) != 3
            or any(not contracts._positive_number(item) for item in lengths)
            or (minimum is not None and any(float(item) <= float(minimum) for item in lengths))
        ):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR", "result.structure", "converted structure metadata is invalid"
                )
            )
        output = contracts._mapping(value.get("output"))
        arc = path.parent / str(output.get("path", "")) if path is not None else None
        if output.get("path") != "input.arc" or not contracts._ordinary_file(arc):
            diagnostics.append(
                contracts._diagnostic("ERROR", "result.arc", "converted ARC output is invalid")
            )
        reference = contracts._mapping(conversion.get("pseudopotential"))
        if reference:
            potcar = contracts._mapping(value.get("potcar"))
            potcar_output = contracts._mapping(potcar.get("output"))
            potcar_path = (
                path.parent / str(potcar_output.get("path", "")) if path is not None else None
            )
            elements = potcar.get("elements")
            symbols = potcar.get("symbols")
            expected_symbols = reference.get("symbols")
            components = potcar.get("components")
            valid_components = (
                isinstance(symbols, list)
                and isinstance(components, list)
                and len(components) == len(symbols)
                and all(
                    isinstance(component, Mapping) and component.get("symbol") == symbol
                    for component, symbol in zip(components, symbols)
                )
            )
            valid_symbols = (
                isinstance(elements, list)
                and bool(elements)
                and isinstance(symbols, list)
                and isinstance(expected_symbols, Mapping)
                and symbols == [expected_symbols.get(element) for element in elements]
            )
            if (
                value.get("pseudopotential_reference_path")
                != conversion.get("pseudopotential_reference_path")
                or potcar.get("reference_id") != reference.get("reference_id")
                or potcar.get("source_env") != "PMG_VASP_PSP_DIR"
                or potcar.get("configuration_source")
                not in {"environment", "pymatgen-settings"}
                or potcar.get("functional") != reference.get("functional")
                or potcar.get("portable_artifact") is not False
                or not valid_symbols
                or not valid_components
                or potcar_output.get("path") != "POTCAR"
                or potcar_output.get("collectable") is not False
                or not contracts._ordinary_file(potcar_path)
                or potcar_path is None
            ):
                diagnostics.append(
                    contracts._diagnostic(
                        "ERROR",
                        "result.potcar",
                        "runtime-only POTCAR settings differ from the approved reference",
                    )
                )
        elif (
            value.get("pseudopotential_reference_path") is not None
            or value.get("potcar") is not None
        ):
            diagnostics.append(
                contracts._diagnostic(
                    "ERROR",
                    "result.potcar_unapproved",
                    "manifest contains an unapproved POTCAR record",
                )
            )
    return {
        "plugin_id": contracts.PLUGIN_ID,
        "operation": contracts.LASP_INPUT_OPERATION,
        "status": "FAIL" if diagnostics else "OK",
        "result_file": str(path) if path is not None else None,
        "diagnostics": diagnostics,
    }


def collect(context: Any) -> Dict[str, Any]:
    checked = check(context)
    if checked["status"] != "OK":
        return {**checked, "artifacts": [], "metrics": {}}
    manifest = Path(str(checked["result_file"]))
    value = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        **checked,
        "artifacts": [
            {
                "role": "lasp-input-manifest",
                "path": str(manifest),
                "media_type": "application/json",
            },
            {
                "role": "lasp-input-structure",
                "path": str(manifest.parent / "input.arc"),
                "media_type": "chemical/x-biosym-archive",
            },
        ],
        "metrics": {"atom_count": contracts._mapping(value.get("structure")).get("atom_count", 0)},
    }
