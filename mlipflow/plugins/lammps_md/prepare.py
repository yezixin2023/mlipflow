"""MLIPFlow adapter for deterministic LAMMPS input preparation.

Preparation is local-only. It never launches LAMMPS, resolves a cluster
model path, converts a model artifact, or submits a scheduler job. Validation reuses the bundled generator's pure contract helpers
so a READY plan cannot drift from the wrapper that will materialize the files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from . import lammps_prepare as generator

PLUGIN_ID = "lammps-md"
OPERATION = "lammps-prepare"
BUNDLED_PREPARE = Path(__file__).absolute().with_name(
    "lammps_prepare.py"
)


def _diagnostic(level: str, code: str, message: str) -> dict[str, str]:
    return {"level": level, "code": code, "message": message}


def _errors(items: list[dict[str, str]]) -> list[dict[str, str]]:
    return [item for item in items if item.get("level") == "error"]


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_relative(value: Any) -> bool:
    if not isinstance(value, str) or not value or any(c in value for c in "\x00\r\n"):
        return False
    path = Path(value)
    return path != Path(".") and not path.is_absolute() and ".." not in path.parts


def _path(root: Any, relative: Any) -> Path:
    return Path(str(root)).expanduser().absolute() / str(relative)


def _structure_input(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip() or any(c in value for c in "\x00\r\n"):
        raise ValueError("structure must be a non-empty path")
    raw = Path(value).expanduser()
    return (raw if raw.is_absolute() else root / raw).resolve()


def _readable_file(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        with path.open("rb") as stream:
            stream.read(1)
        return True
    except OSError:
        return False


def _ordinary_project_file(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return path.is_file() and path.stat().st_size > 0


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file() or path.stat().st_size <= 0:
        return None, "missing or empty JSON file"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"invalid UTF-8 JSON: {exc}"
    return (value, None) if isinstance(value, dict) else (None, "JSON root must be an object")


def _project_inputs(context: dict[str, Any]) -> tuple[Path, Path, Path]:
    root = Path(str(context["project_root"])).expanduser().absolute()
    inputs = _mapping(context["inputs"])
    return (
        _structure_input(root, inputs["structure"]),
        _path(root, inputs["model_reference"]),
        _path(root, inputs["lammps_config"]),
    )


def _source_contract(model_ref: Path, config: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        model = generator._model_reference(model_ref)
        cfg = generator._config(config, model)
    except generator.ContractError as exc:
        raise ValueError(str(exc)) from exc
    return model, cfg


def _validate(context: Any, *, require_fresh_output: bool) -> list[dict[str, str]]:
    if not isinstance(context, dict):
        return [_diagnostic("error", "context.type", "context must be an object")]
    diagnostics: list[dict[str, str]] = []
    for key in ("project_root", "attempt_dir", "inputs", "parameters", "backend", "resources"):
        if key not in context:
            diagnostics.append(_diagnostic("error", "context.keys", f"missing context field: {key}"))
    if context.get("backend") != "local":
        diagnostics.append(
            _diagnostic("error", "backend.unsupported", "lammps-prepare is local-only and never submits LAMMPS")
        )
    inputs = context.get("inputs")
    parameters = context.get("parameters")
    if not isinstance(inputs, dict) or not isinstance(parameters, dict):
        diagnostics.append(_diagnostic("error", "context.mapping", "inputs and parameters must be objects"))
        return diagnostics
    if parameters.get("operation", OPERATION) != OPERATION:
        diagnostics.append(_diagnostic("error", "parameters.operation", "operation must be lammps-prepare"))
    unknown_parameters = set(parameters) - {"operation", "output_dir", "structure_format"}
    if unknown_parameters:
        diagnostics.append(
            _diagnostic(
                "error",
                "parameters.unknown",
                "unsupported parameters: " + ", ".join(sorted(str(item) for item in unknown_parameters)),
            )
        )
    unknown_inputs = set(inputs) - {"structure", "model_reference", "lammps_config"}
    if unknown_inputs:
        diagnostics.append(
            _diagnostic(
                "error",
                "inputs.unknown",
                "unsupported inputs: " + ", ".join(sorted(str(item) for item in unknown_inputs)),
            )
        )
    structure_value = inputs.get("structure")
    if (
        not isinstance(structure_value, str)
        or not structure_value.strip()
        or any(c in structure_value for c in "\x00\r\n")
    ):
        diagnostics.append(
            _diagnostic("error", "inputs.structure", "structure must be a non-empty path")
        )
    for name in ("model_reference", "lammps_config"):
        if not _safe_relative(inputs.get(name)):
            diagnostics.append(
                _diagnostic("error", f"inputs.{name}", f"{name} must be a safe project-relative path")
            )
    output_dir = parameters.get("output_dir", "lammps-inputs")
    if not _safe_relative(output_dir):
        diagnostics.append(_diagnostic("error", "parameters.output_dir", "output_dir must be attempt-relative"))
    try:
        generator._structure_format(parameters.get("structure_format"))
    except generator.ContractError as exc:
        diagnostics.append(_diagnostic("error", "parameters.structure_format", str(exc)))
    root = Path(str(context.get("project_root", ""))).expanduser().absolute()
    attempt = Path(str(context.get("attempt_dir", ""))).expanduser().absolute()
    if not root.is_dir():
        diagnostics.append(_diagnostic("error", "project_root", "project_root must exist"))
        return diagnostics
    try:
        structure, model_ref, config = _project_inputs(context)
    except (KeyError, ValueError):
        return diagnostics
    if not structure.exists():
        diagnostics.append(
            _diagnostic(
                "error",
                "inputs.structure_file",
                f"structure file does not exist: {structure}",
            )
        )
    elif not _readable_file(structure):
        diagnostics.append(
            _diagnostic(
                "error",
                "inputs.structure_file",
                f"structure must be a readable file: {structure}",
            )
        )
    for name, path in (("model_reference", model_ref), ("lammps_config", config)):
        if not _ordinary_project_file(path, root):
            diagnostics.append(
                _diagnostic("error", f"inputs.{name}_file", f"{name} must be a project file")
            )
    if _ordinary_project_file(model_ref, root) and _ordinary_project_file(config, root):
        try:
            _source_contract(model_ref, config)
        except ValueError as exc:
            diagnostics.append(_diagnostic("error", "source.contract", str(exc)))
    destination = attempt / str(output_dir)
    if require_fresh_output and destination.exists():
        diagnostics.append(_diagnostic("error", "output.exists", "fresh prepare output_dir must not already exist"))
    return diagnostics


def validate(context: Any) -> list[dict[str, str]]:
    return _validate(context, require_fresh_output=True)


def plan(context: Any) -> dict[str, Any]:
    diagnostics = validate(context)
    if _errors(diagnostics):
        return {
            "plugin_id": PLUGIN_ID,
            "status": "BLOCKED",
            "executable": False,
            "diagnostics": diagnostics,
        }
    assert isinstance(context, dict)
    structure, model_ref, config = _project_inputs(context)
    model, cfg = _source_contract(model_ref, config)
    output_dir = Path(str(context["attempt_dir"])).expanduser().absolute() / str(
        _mapping(context["parameters"]).get("output_dir", "lammps-inputs")
    )
    targets = list(cfg["targets"])
    expected = [
        str(output_dir / "structure.data"),
        str(output_dir / "lammps-input-manifest.json"),
        *[str(output_dir / f"in.{target}.lammps") for target in targets],
    ]
    argv = [
        sys.executable,
        str(BUNDLED_PREPARE),
        "--structure",
        str(structure),
        "--model-reference",
        str(model_ref),
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
    ]
    structure_format = generator._structure_format(
        _mapping(context["parameters"]).get("structure_format")
    )
    if structure_format is not None:
        argv.extend(["--structure-format", structure_format])
    return {
        "plugin_id": PLUGIN_ID,
        "status": "READY",
        "executable": True,
        "argv": argv,
        "cwd": str(Path(context["attempt_dir"]).expanduser().absolute()),
        "expected_outputs": expected,
        "diagnostics": diagnostics,
        "approval_summary": {"preparation_contract": generator.PREPARATION_CONTRACT,
            "operation": OPERATION,
            "framework": model["framework"],
            "model_id": model["model_id"],
            "model_path": model["relative_path"],
            "model_kind": model["kind"],
            "artifact_format": model["artifact_format"],
            "lammps_interface": model.get("lammps_interface"),
            "ensemble": cfg["ensemble"],
            "targets": targets,
            "structure_format": structure_format or "auto",
            "type_map": cfg["type_map"],
            "temperature_K": cfg["temperature_k"],
            "timestep_fs": cfg["timestep_fs"],
            "steps": cfg["steps"],
            "simulated_time_ps": float(cfg["timestep_fs"]) * int(cfg["steps"]) / 1000.0,
            "executes_lammps": False,
        },
        "input_paths": {
            "structure": str(structure),
            "model_reference": str(model_ref),
            "lammps_config": str(config),
            "prepare_wrapper": str(BUNDLED_PREPARE),
        },
    }


def _output_dir(context: dict[str, Any]) -> Path:
    return Path(str(context["attempt_dir"])).expanduser().absolute() / str(
        _mapping(context["parameters"]).get("output_dir", "lammps-inputs")
    )


def _verify(
    context: dict[str, Any], manifest: dict[str, Any], verify_files: bool
) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    if manifest.get("schema_version") != 1 or manifest.get("plugin_id") != PLUGIN_ID:
        diagnostics.append(
            _diagnostic("error", "result.schema", "LAMMPS input manifest schema is invalid")
        )
    if manifest.get("operation") != OPERATION or manifest.get("status") != "OK":
        diagnostics.append(
            _diagnostic("error", "result.status", "prepare manifest must report lammps-prepare status=OK")
        )
    structure, model_ref, config = _project_inputs(context)
    expected_inputs = {
        "structure": str(structure),
        "model_reference": str(model_ref),
        "lammps_config": str(config),
    }
    if manifest.get("input_paths") != expected_inputs:
        diagnostics.append(
            _diagnostic("error", "result.inputs", "input paths differ from current project inputs")
        )
    try:
        model, cfg = _source_contract(model_ref, config)
    except ValueError as exc:
        diagnostics.append(_diagnostic("error", "result.source_contract", str(exc)))
        return diagnostics
    if manifest.get("model") != model:
        diagnostics.append(_diagnostic("error", "result.model", "manifest model record differs"))
    if manifest.get("md") != cfg:
        diagnostics.append(_diagnostic("error", "result.md", "manifest MD configuration differs"))
    expected_structure_format = generator._structure_format(
        _mapping(context["parameters"]).get("structure_format")
    ) or "auto"
    if manifest.get("source_structure_format") != expected_structure_format:
        diagnostics.append(
            _diagnostic(
                "error",
                "result.structure_format",
                "manifest source structure format differs from the approved parameter",
            )
        )
    if manifest.get("runtime_model_variable") != "MODEL_FILE":
        diagnostics.append(
            _diagnostic("error", "result.model_variable", "runtime model variable must be MODEL_FILE")
        )
    files = manifest.get("generated_files")
    if not isinstance(files, list):
        diagnostics.append(_diagnostic("error", "result.files", "generated_files must be a list"))
        return diagnostics
    expected_names = {"structure.data", *{f"in.{target}.lammps" for target in cfg["targets"]}}
    records = {
        item.get("name"): item
        for item in files
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if set(records) != expected_names:
        diagnostics.append(
            _diagnostic("error", "result.file_set", "generated file set differs from approved targets")
        )
    if verify_files:
        output_dir = _output_dir(context)
        for name in expected_names:
            record = records.get(name)
            path = output_dir / name
            if (
                not isinstance(record, dict)
                or not path.is_file()
                or path.stat().st_size <= 0
            ):
                diagnostics.append(
                    _diagnostic(
                        "error",
                        f"result.file.{name}",
                        f"generated file {name} is missing or invalid",
                    )
                )
                continue
            if name.startswith("in."):
                try:
                    text = path.read_text(encoding="utf-8")
                except (OSError, UnicodeError) as exc:
                    diagnostics.append(
                        _diagnostic("error", f"result.file.{name}.text", f"input deck is unreadable: {exc}")
                    )
                    continue
                if "${MODEL_FILE}" not in text:
                    diagnostics.append(
                        _diagnostic("error", f"result.file.{name}.model", "input deck must use ${MODEL_FILE}")
                    )
                if str(model["relative_path"]) in text:
                    diagnostics.append(
                        _diagnostic(
                            "error",
                            f"result.file.{name}.path",
                            "input deck must not embed the site model relative path",
                        )
                    )
    launchers = manifest.get("launchers")
    launcher_targets = {
        item.get("target")
        for item in launchers
        if isinstance(item, dict) and isinstance(item.get("target"), str)
    } if isinstance(launchers, list) else set()
    if launcher_targets != set(cfg["targets"]):
        diagnostics.append(
            _diagnostic("error", "result.launchers", "launcher metadata must cover exactly the approved targets")
        )
    return diagnostics


def check(context: Any) -> dict[str, Any]:
    diagnostics = _validate(context, require_fresh_output=False)
    if _errors(diagnostics) or not isinstance(context, dict):
        return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
    manifest_path = _output_dir(context) / "lammps-input-manifest.json"
    manifest, error = _read_json(manifest_path)
    if manifest is None:
        return {
            "plugin_id": PLUGIN_ID,
            "status": "WAIT" if not manifest_path.exists() else "FAIL",
            "diagnostics": [_diagnostic("error", "result.manifest", str(error))],
        }
    diagnostics.extend(_verify(context, manifest, True))
    return {
        "plugin_id": PLUGIN_ID,
        "status": "FAIL" if _errors(diagnostics) else "OK",
        "diagnostics": diagnostics,
        "result_manifest": str(manifest_path),
    }


def collect(context: Any) -> dict[str, Any]:
    checked = check(context)
    if checked.get("status") != "OK" or not isinstance(context, dict):
        return checked
    output_dir = _output_dir(context)
    manifest, _ = _read_json(output_dir / "lammps-input-manifest.json")
    assert manifest is not None
    artifacts = [
        {
            "path": str(output_dir / "lammps-input-manifest.json"),
            "role": "lammps-input-manifest",
            "media_type": "application/json",
        }
    ]
    for item in manifest["generated_files"]:
        name = str(item["name"])
        artifacts.append(
            {
                "path": str(output_dir / name),
                "role": "lammps-structure" if name == "structure.data" else "lammps-input",
                "media_type": "text/plain",
            }
        )
    return {
        "plugin_id": PLUGIN_ID,
        "status": "OK",
        "diagnostics": [],
        "artifacts": artifacts,
        "metrics": {
            "target_count": float(len(manifest["md"]["targets"])),
            "steps": float(manifest["md"]["steps"]),
            "simulated_time_ps": float(manifest["md"]["steps"])
            * float(manifest["md"]["timestep_fs"])
            / 1000.0,
        },
    }
