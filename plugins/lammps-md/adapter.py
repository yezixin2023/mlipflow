"""MLIPFlow adapter for deterministic LAMMPS input preparation.

Version 0.1 is deliberately prepare-only and local-only.  It never launches
LAMMPS, resolves a cluster model path, converts a model artifact, or submits a
scheduler job.  The bundled generator writes reviewable input decks plus an
immutable manifest for a later execution contract.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PLUGIN_ID = "lammps-md"
OPERATION = "lammps-prepare"
BUNDLED_PREPARE = Path(globals().get("__file__", "adapter.py")).absolute().with_name(
    "lammps_prepare.py"
)
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_STRUCTURE_BYTES = 128 * 1024 * 1024
MAX_OUTPUT_BYTES = 256 * 1024 * 1024
SUPPORTED_FRAMEWORKS = {"deepmd", "mace", "m3gnet"}
ALL_FRAMEWORKS = SUPPORTED_FRAMEWORKS | {"chgnet"}
EXPECTED_FORMATS = {
    "deepmd": "deepmd-lammps-model",
    "mace": "mace-lammps-torchscript",
    "m3gnet": "matgl-lammps-torchscript",
}


def _diagnostic(level: str, code: str, message: str) -> dict[str, str]:
    return {"level": level, "code": code, "message": message}


def _errors(items: list[dict[str, str]]) -> list[dict[str, str]]:
    return [item for item in items if item.get("level") == "error"]


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_relative(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\x00" in value or "\n" in value or "\r" in value:
        return False
    path = Path(value)
    return path != Path(".") and not path.is_absolute() and ".." not in path.parts


def _path(root: Any, relative: Any) -> Path:
    return Path(str(root)).expanduser().absolute() / str(relative)


def _ordinary_project_file(path: Path, root: Path, max_bytes: int) -> bool:
    base = root.expanduser().absolute()
    candidate = path.expanduser().absolute()
    try:
        relative = candidate.relative_to(base)
    except ValueError:
        return False
    current = base
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return False
    if candidate.is_symlink() or not candidate.is_file():
        return False
    size = candidate.stat().st_size
    return 0 < size <= max_bytes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path, max_bytes: int = MAX_JSON_BYTES) -> tuple[dict[str, Any] | None, str | None]:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= max_bytes:
        return None, "missing, unsafe or oversized JSON file"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"invalid UTF-8 JSON: {exc}"
    return (value, None) if isinstance(value, dict) else (None, "JSON root must be an object")


def _project_inputs(context: dict[str, Any]) -> tuple[Path, Path, Path]:
    root = Path(str(context["project_root"])).expanduser().absolute()
    inputs = _mapping(context["inputs"])
    return (
        _path(root, inputs["structure"]),
        _path(root, inputs["model_reference"]),
        _path(root, inputs["lammps_config"]),
    )


def _parse_model_reference(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    value, error = _read_json(path)
    if value is None:
        return None, error
    framework = value.get("framework")
    if framework not in ALL_FRAMEWORKS:
        return None, "model_reference.framework must be deepmd, mace, m3gnet, or chgnet"
    if framework == "chgnet":
        return None, (
            "CHGNet LAMMPS is intentionally blocked in lammps-md@0.1 because no pinned native "
            "CHGNet export/pair-style contract is recorded; use ase-md until a bridge is reviewed"
        )
    if value.get("schema_version") != 1:
        return None, "model_reference.schema_version must be 1"
    if value.get("kind") != "file":
        return None, "LAMMPS model_reference.kind must be file"
    if value.get("artifact_format") != EXPECTED_FORMATS[str(framework)]:
        return None, f"{framework} artifact_format must be {EXPECTED_FORMATS[str(framework)]}"
    fingerprint = value.get("fingerprint")
    if not (
        isinstance(fingerprint, str)
        and fingerprint.startswith("sha256:")
        and len(fingerprint) == 71
        and all(c in "0123456789abcdef" for c in fingerprint[7:])
    ):
        return None, "model_reference.fingerprint must be sha256:<64 lowercase hex>"
    elements = value.get("elements")
    if not isinstance(elements, list) or not elements or len(elements) != len(set(elements)):
        return None, "model_reference.elements must be a non-empty unique list"
    relative_path = value.get("relative_path")
    if not _safe_relative(relative_path):
        return None, "model_reference.relative_path must be a safe site-root-relative path"
    return value, None


def _parse_config(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    value, error = _read_json(path)
    if value is None:
        return None, error
    if value.get("schema_version") != 1 or value.get("engine") != "lammps":
        return None, "lammps_config must declare schema_version=1 and engine=lammps"
    targets = value.get("targets")
    if not isinstance(targets, list) or not targets or any(item not in {"cpu", "gpu"} for item in targets):
        return None, "lammps_config.targets must contain cpu and/or gpu"
    if len(targets) != len(set(targets)):
        return None, "lammps_config.targets must be unique"
    if value.get("ensemble") not in {"nvt", "npt-isotropic"}:
        return None, "lammps_config.ensemble must be nvt or npt-isotropic"
    type_map = value.get("type_map")
    if not isinstance(type_map, list) or not type_map or len(type_map) != len(set(type_map)):
        return None, "lammps_config.type_map must be a non-empty unique list"
    for key in ("steps", "thermo_interval", "dump_interval", "seed"):
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            return None, f"lammps_config.{key} must be a positive integer"
    if value["thermo_interval"] > value["steps"] or value["dump_interval"] > value["steps"]:
        return None, "thermo_interval/dump_interval cannot exceed steps"
    for key in ("temperature_k", "timestep_fs", "thermostat_damping_fs"):
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, (int, float)) or float(item) <= 0:
            return None, f"lammps_config.{key} must be positive"
    if float(value["timestep_fs"]) > 10:
        return None, "lammps_config.timestep_fs must not exceed 10 fs"
    if value["ensemble"] == "npt-isotropic":
        pressure = value.get("pressure_gpa")
        pdamp = value.get("barostat_damping_fs")
        if isinstance(pressure, bool) or not isinstance(pressure, (int, float)):
            return None, "NPT requires finite pressure_gpa"
        if isinstance(pdamp, bool) or not isinstance(pdamp, (int, float)) or float(pdamp) <= 0:
            return None, "NPT requires positive barostat_damping_fs"
    elif value.get("pressure_gpa") is not None or value.get("barostat_damping_fs") is not None:
        return None, "pressure_gpa/barostat_damping_fs are NPT-only"
    return value, None


class Adapter:
    def validate(self, context: Any) -> list[dict[str, str]]:
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
        if parameters.get("extra_args") is not None:
            diagnostics.append(_diagnostic("error", "parameters.extra_args", "arbitrary extra_args are not accepted"))
        for name in ("structure", "model_reference", "lammps_config"):
            if not _safe_relative(inputs.get(name)):
                diagnostics.append(_diagnostic("error", f"inputs.{name}", f"{name} must be a safe project-relative path"))
        output_dir = parameters.get("output_dir", "lammps-inputs")
        if not _safe_relative(output_dir):
            diagnostics.append(_diagnostic("error", "parameters.output_dir", "output_dir must be attempt-relative"))
        root = Path(str(context.get("project_root", ""))).expanduser().absolute()
        attempt = Path(str(context.get("attempt_dir", ""))).expanduser().absolute()
        if not root.is_dir():
            diagnostics.append(_diagnostic("error", "project_root", "project_root must exist"))
            return diagnostics
        try:
            structure, model_ref, config = _project_inputs(context)
        except KeyError:
            return diagnostics
        for name, path, limit in (
            ("structure", structure, MAX_STRUCTURE_BYTES),
            ("model_reference", model_ref, MAX_JSON_BYTES),
            ("lammps_config", config, MAX_JSON_BYTES),
        ):
            if not _ordinary_project_file(path, root, limit):
                diagnostics.append(_diagnostic("error", f"inputs.{name}_file", f"{name} must be an ordinary bounded project file"))
        if _ordinary_project_file(model_ref, root, MAX_JSON_BYTES):
            model, error = _parse_model_reference(model_ref)
            if model is None:
                diagnostics.append(_diagnostic("error", "model_reference", str(error)))
        else:
            model = None
        if _ordinary_project_file(config, root, MAX_JSON_BYTES):
            cfg, error = _parse_config(config)
            if cfg is None:
                diagnostics.append(_diagnostic("error", "lammps_config", str(error)))
        else:
            cfg = None
        if model is not None and cfg is not None:
            model_elements = model.get("elements")
            type_map = cfg.get("type_map")
            if isinstance(model_elements, list) and isinstance(type_map, list):
                unsupported = sorted(set(type_map) - set(model_elements))
                if unsupported:
                    diagnostics.append(
                        _diagnostic("error", "type_map.model", "type_map elements absent from model: " + ", ".join(unsupported))
                    )
        destination = attempt / str(output_dir)
        if destination.exists():
            diagnostics.append(_diagnostic("error", "output.exists", "fresh prepare output_dir must not already exist"))
        return diagnostics

    def plan(self, context: Any) -> dict[str, Any]:
        diagnostics = self.validate(context)
        if _errors(diagnostics):
            return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": diagnostics}
        assert isinstance(context, dict)
        structure, model_ref, config = _project_inputs(context)
        cfg, _ = _parse_config(config)
        model, _ = _parse_model_reference(model_ref)
        assert cfg is not None and model is not None
        output_dir = Path(str(context["attempt_dir"])).expanduser().absolute() / str(
            _mapping(context["parameters"]).get("output_dir", "lammps-inputs")
        )
        targets = list(cfg["targets"])
        expected = [str(output_dir / "structure.data"), str(output_dir / "lammps-input-manifest.json")]
        expected.extend(str(output_dir / f"in.{target}.lammps") for target in targets)
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
        return {
            "plugin_id": PLUGIN_ID,
            "status": "READY",
            "executable": True,
            "argv": argv,
            "cwd": str(Path(context["attempt_dir"]).expanduser().absolute()),
            "expected_outputs": expected,
            "diagnostics": diagnostics,
            "approval_summary": {
                "operation": OPERATION,
                "framework": model["framework"],
                "model_id": model.get("model_id"),
                "model_fingerprint": model["fingerprint"],
                "artifact_format": model["artifact_format"],
                "ensemble": cfg["ensemble"],
                "targets": targets,
                "type_map": cfg["type_map"],
                "temperature_K": cfg["temperature_k"],
                "timestep_fs": cfg["timestep_fs"],
                "steps": cfg["steps"],
                "simulated_time_ps": float(cfg["timestep_fs"]) * int(cfg["steps"]) / 1000.0,
                "executes_lammps": False,
            },
            "input_fingerprints": {
                "structure": _sha256(structure),
                "model_reference": _sha256(model_ref),
                "lammps_config": _sha256(config),
                "prepare_wrapper": _sha256(BUNDLED_PREPARE),
            },
        }

    def prepare(self, context: Any, plan: Any) -> dict[str, Any]:
        if not isinstance(plan, dict) or plan.get("status") != "READY":
            return {
                "plugin_id": PLUGIN_ID,
                "status": "BLOCKED",
                "executable": False,
                "diagnostics": [_diagnostic("error", "plan.not_ready", "prepare requires a READY plan")],
            }
        return dict(plan)

    def _output_dir(self, context: dict[str, Any]) -> Path:
        return Path(str(context["attempt_dir"])).expanduser().absolute() / str(
            _mapping(context["parameters"]).get("output_dir", "lammps-inputs")
        )

    def _verify(self, context: dict[str, Any], manifest: dict[str, Any], verify_files: bool) -> list[dict[str, str]]:
        diagnostics: list[dict[str, str]] = []
        if manifest.get("schema_version") != 1 or manifest.get("plugin_id") != PLUGIN_ID:
            diagnostics.append(_diagnostic("error", "result.identity", "LAMMPS input manifest identity is invalid"))
        if manifest.get("operation") != OPERATION or manifest.get("status") != "OK":
            diagnostics.append(_diagnostic("error", "result.status", "prepare manifest must report lammps-prepare status=OK"))
        structure, model_ref, config = _project_inputs(context)
        expected_inputs = {
            "structure": _sha256(structure),
            "model_reference": _sha256(model_ref),
            "lammps_config": _sha256(config),
        }
        if manifest.get("input_fingerprints") != expected_inputs:
            diagnostics.append(_diagnostic("error", "result.inputs", "input fingerprints differ from current project inputs"))
        model, _ = _parse_model_reference(model_ref)
        cfg, _ = _parse_config(config)
        if model is None or cfg is None:
            diagnostics.append(_diagnostic("error", "result.source_contract", "source model/config contract is no longer valid"))
            return diagnostics
        result_model = _mapping(manifest.get("model"))
        for key in ("model_id", "framework", "relative_path", "kind", "fingerprint", "artifact_format", "elements"):
            if result_model.get(key) != model.get(key):
                diagnostics.append(_diagnostic("error", f"result.model.{key}", f"manifest model field {key} differs"))
        if manifest.get("md") != cfg:
            diagnostics.append(_diagnostic("error", "result.md", "manifest MD configuration differs from approved config"))
        if manifest.get("runtime_model_variable") != "MODEL_FILE":
            diagnostics.append(_diagnostic("error", "result.model_variable", "runtime model variable must be MODEL_FILE"))
        files = manifest.get("generated_files")
        if not isinstance(files, list):
            diagnostics.append(_diagnostic("error", "result.files", "generated_files must be a list"))
            return diagnostics
        expected_names = {"structure.data", *{f"in.{target}.lammps" for target in cfg["targets"]}}
        records = {item.get("name"): item for item in files if isinstance(item, dict)}
        if set(records) != expected_names:
            diagnostics.append(_diagnostic("error", "result.file_set", "generated file set differs from approved targets"))
        if verify_files:
            output_dir = self._output_dir(context)
            for name in expected_names:
                record = records.get(name)
                path = output_dir / name
                if (
                    not isinstance(record, dict)
                    or path.is_symlink()
                    or not path.is_file()
                    or not 0 < path.stat().st_size <= MAX_OUTPUT_BYTES
                    or record.get("size_bytes") != path.stat().st_size
                    or record.get("sha256") != _sha256(path)
                ):
                    diagnostics.append(_diagnostic("error", f"result.file.{name}", f"generated file {name} is missing or fingerprint-mismatched"))
                    continue
                if name.startswith("in."):
                    text = path.read_text(encoding="utf-8")
                    if "${MODEL_FILE}" not in text:
                        diagnostics.append(_diagnostic("error", f"result.file.{name}.model", "input deck must use ${MODEL_FILE}"))
                    if str(model.get("relative_path")) in text:
                        diagnostics.append(_diagnostic("error", f"result.file.{name}.path", "input deck must not embed site model relative paths"))
        launchers = manifest.get("launchers")
        if not isinstance(launchers, list) or {item.get("target") for item in launchers if isinstance(item, dict)} != set(cfg["targets"]):
            diagnostics.append(_diagnostic("error", "result.launchers", "launcher metadata must cover exactly the approved targets"))
        return diagnostics

    def check(self, context: Any) -> dict[str, Any]:
        diagnostics = self.validate({**context, "parameters": {**_mapping(context.get("parameters")), "output_dir": "__check__"}}) if isinstance(context, dict) else self.validate(context)
        # Ignore the synthetic fresh-output diagnostic and validate current source inputs independently.
        diagnostics = [item for item in diagnostics if item.get("code") != "output.exists"]
        if _errors(diagnostics) or not isinstance(context, dict):
            return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
        manifest_path = self._output_dir(context) / "lammps-input-manifest.json"
        manifest, error = _read_json(manifest_path)
        if manifest is None:
            return {"plugin_id": PLUGIN_ID, "status": "WAIT" if not manifest_path.exists() else "FAIL", "diagnostics": [_diagnostic("error", "result.manifest", str(error))]}
        diagnostics.extend(self._verify(context, manifest, True))
        return {"plugin_id": PLUGIN_ID, "status": "FAIL" if _errors(diagnostics) else "OK", "diagnostics": diagnostics, "result_manifest": str(manifest_path)}

    def collect(self, context: Any) -> dict[str, Any]:
        checked = self.check(context)
        if checked.get("status") != "OK" or not isinstance(context, dict):
            return checked
        output_dir = self._output_dir(context)
        manifest, _ = _read_json(output_dir / "lammps-input-manifest.json")
        assert manifest is not None
        artifacts = [
            {"path": str(output_dir / "lammps-input-manifest.json"), "role": "lammps-input-manifest", "media_type": "application/json"}
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
                "simulated_time_ps": float(manifest["md"]["steps"]) * float(manifest["md"]["timestep_fs"]) / 1000.0,
            },
        }

    def replay(self, context: Any) -> dict[str, Any]:
        collected = self.collect(context)
        collected["executable"] = False
        return collected
