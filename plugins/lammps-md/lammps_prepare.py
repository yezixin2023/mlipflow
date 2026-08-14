"""Deterministic LAMMPS input preparation for explicit MLIP artifacts.

This module never launches LAMMPS or a scheduler.  It converts one ASE-readable
structure into a LAMMPS data file and writes CPU/GPU input decks whose model path
is supplied later through the LAMMPS command-line variable ``MODEL_FILE``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
from pathlib import Path
from typing import Any

PLUGIN_ID = "lammps-md"
OPERATION = "lammps-prepare"
SCHEMA_VERSION = 1
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_STRUCTURE_BYTES = 128 * 1024 * 1024
MAX_OUTPUT_BYTES = 256 * 1024 * 1024
SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
SAFE_ELEMENT = re.compile(r"[A-Z][a-z]?")
TARGETS = {"cpu", "gpu"}
ENSEMBLES = {"nvt", "npt-isotropic"}

INTERFACES: dict[str, dict[str, Any]] = {
    "deepmd": {
        "artifact_format": "deepmd-lammps-model",
        "cpu_pair_style": "deepmd",
        "gpu_pair_style": "deepmd",
        "cpu_packages": ["USER-DEEPMD or DeePMD LAMMPS plugin"],
        "gpu_packages": ["USER-DEEPMD or DeePMD LAMMPS plugin", "DeePMD CUDA build"],
        "gpu_launcher": [],
        "gpu_limitations": ["GPU selection is owned by the DeePMD/LAMMPS runtime; no Kokkos suffix is added."],
    },
    "mace": {
        "artifact_format": "mace-lammps-torchscript",
        "cpu_pair_style": "mace",
        "gpu_pair_style": "mace no_domain_decomposition",
        "cpu_packages": ["ML-MACE"],
        "gpu_packages": ["ML-MACE", "KOKKOS", "CUDA-enabled LibTorch"],
        "gpu_launcher": ["-k", "on", "g", "1", "-sf", "kk"],
        "gpu_limitations": ["Version 0.1 prepares the original ML-MACE single-GPU Kokkos interface; ML-IAP is a later interface contract."],
    },
    "m3gnet": {
        "artifact_format": "matgl-lammps-torchscript",
        "cpu_pair_style": "matgl",
        "gpu_pair_style": "matgl/kk",
        "cpu_packages": ["ML-MATGL", "LibTorch"],
        "gpu_packages": ["ML-MATGL", "KOKKOS", "CUDA-enabled LibTorch"],
        "gpu_launcher": ["-k", "on", "g", "1", "-sf", "kk"],
        "gpu_limitations": ["MatGL Kokkos inference is treated as single-GPU/single-rank in version 0.1."],
    },
}


class ContractError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _ordinary_file(path: Path, label: str, max_bytes: int) -> Path:
    unresolved = path.expanduser().absolute()
    if unresolved.is_symlink():
        raise ContractError(f"{label} must not be a symlink")
    resolved = unresolved.resolve()
    if not resolved.is_file():
        raise ContractError(f"{label} must be an ordinary file")
    size = resolved.stat().st_size
    if not 0 < size <= max_bytes:
        raise ContractError(f"{label} must be between 1 and {max_bytes} bytes")
    return resolved


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = _ordinary_file(path, label, MAX_JSON_BYTES)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must contain a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{label} must be a finite number")
    return result


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result <= 0:
        raise ContractError(f"{label} must be positive")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{label} must be a positive integer")
    return value


def _elements(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{label} must be a non-empty element list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not SAFE_ELEMENT.fullmatch(item):
            raise ContractError(f"{label} contains an invalid element symbol")
        if item in result:
            raise ContractError(f"{label} must not contain duplicate elements")
        result.append(item)
    return result


def _model_reference(path: Path) -> dict[str, Any]:
    value = _read_json(path, "model_reference")
    allowed = {
        "schema_version", "model_id", "framework", "relative_path", "kind",
        "fingerprint", "artifact_format", "elements",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ContractError("model_reference contains unsupported fields: " + ", ".join(unknown))
    if value.get("schema_version") != 1:
        raise ContractError("model_reference.schema_version must be 1")
    model_id = value.get("model_id")
    if not isinstance(model_id, str) or not SAFE_ID.fullmatch(model_id):
        raise ContractError("model_reference.model_id is invalid")
    framework = value.get("framework")
    if framework == "chgnet":
        raise ContractError(
            "CHGNet has no pinned native LAMMPS export/pair-style contract in lammps-md@0.1; use ase-md or provide a separately reviewed bridge"
        )
    if framework not in INTERFACES:
        raise ContractError("framework must be deepmd, mace, m3gnet, or chgnet")
    relative = value.get("relative_path")
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise ContractError("model_reference.relative_path must be a non-empty POSIX relative path")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ContractError("model_reference.relative_path must remain below the site model root")
    if value.get("kind") != "file":
        raise ContractError("LAMMPS model_reference.kind must be file")
    fingerprint = value.get("fingerprint")
    if not isinstance(fingerprint, str) or not SHA256.fullmatch(fingerprint):
        raise ContractError("model_reference.fingerprint must be sha256:<64 lowercase hex>")
    expected_format = INTERFACES[str(framework)]["artifact_format"]
    if value.get("artifact_format") != expected_format:
        raise ContractError(f"{framework} LAMMPS artifact_format must be {expected_format}")
    return {
        "model_id": model_id,
        "framework": framework,
        "relative_path": relative,
        "kind": "file",
        "fingerprint": fingerprint,
        "artifact_format": expected_format,
        "elements": _elements(value.get("elements"), "model_reference.elements"),
    }


def _config(path: Path, model: dict[str, Any]) -> dict[str, Any]:
    value = _read_json(path, "lammps_config")
    allowed = {
        "schema_version", "engine", "ensemble", "targets", "type_map", "temperature_k",
        "timestep_fs", "steps", "thermo_interval", "dump_interval", "seed",
        "thermostat_damping_fs", "pressure_gpa", "barostat_damping_fs",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ContractError("lammps_config contains unsupported fields: " + ", ".join(unknown))
    if value.get("schema_version") != 1 or value.get("engine") != "lammps":
        raise ContractError("lammps_config must declare schema_version=1 and engine=lammps")
    ensemble = value.get("ensemble")
    if ensemble not in ENSEMBLES:
        raise ContractError("ensemble must be nvt or npt-isotropic")
    targets = value.get("targets")
    if not isinstance(targets, list) or not targets or any(item not in TARGETS for item in targets):
        raise ContractError("targets must be a non-empty list containing only cpu/gpu")
    if len(set(targets)) != len(targets):
        raise ContractError("targets must not contain duplicates")
    type_map = _elements(value.get("type_map"), "lammps_config.type_map")
    unsupported = [item for item in type_map if item not in model["elements"]]
    if unsupported:
        raise ContractError("type_map contains elements absent from model_reference.elements: " + ", ".join(unsupported))
    temperature = _positive(value.get("temperature_k"), "temperature_k")
    timestep_fs = _positive(value.get("timestep_fs"), "timestep_fs")
    if timestep_fs > 10:
        raise ContractError("timestep_fs must not exceed 10 fs")
    steps = _positive_int(value.get("steps"), "steps")
    thermo_interval = _positive_int(value.get("thermo_interval"), "thermo_interval")
    dump_interval = _positive_int(value.get("dump_interval"), "dump_interval")
    if thermo_interval > steps or dump_interval > steps:
        raise ContractError("thermo_interval and dump_interval cannot exceed steps")
    seed = value.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0 or seed > 2_147_483_647:
        raise ContractError("seed must be an integer in [1, 2147483647]")
    tdamp = _positive(value.get("thermostat_damping_fs"), "thermostat_damping_fs")
    result = {
        "ensemble": ensemble,
        "targets": list(targets),
        "type_map": type_map,
        "temperature_k": temperature,
        "timestep_fs": timestep_fs,
        "steps": steps,
        "thermo_interval": thermo_interval,
        "dump_interval": dump_interval,
        "seed": seed,
        "thermostat_damping_fs": tdamp,
    }
    if ensemble == "npt-isotropic":
        result["pressure_gpa"] = _finite(value.get("pressure_gpa"), "pressure_gpa")
        result["barostat_damping_fs"] = _positive(value.get("barostat_damping_fs"), "barostat_damping_fs")
    elif value.get("pressure_gpa") is not None or value.get("barostat_damping_fs") is not None:
        raise ContractError("pressure_gpa/barostat_damping_fs are NPT-only")
    return result


def _pair_block(framework: str, target: str, type_map: list[str]) -> list[str]:
    interface = INTERFACES[framework]
    pair_style = interface[f"{target}_pair_style"]
    elements = " ".join(type_map)
    if framework == "deepmd":
        return [f"pair_style      {pair_style} ${{MODEL_FILE}}", f"pair_coeff      * * {elements}"]
    return [f"pair_style      {pair_style}", f"pair_coeff      * * ${{MODEL_FILE}} {elements}"]


def _deck(framework: str, target: str, config: dict[str, Any]) -> str:
    type_map = list(config["type_map"])
    timestep_ps = float(config["timestep_fs"]) / 1000.0
    tdamp_ps = float(config["thermostat_damping_fs"]) / 1000.0
    lines = [
        "# Generated by MLIPFlow lammps-md@0.1; MODEL_FILE is supplied at runtime.",
        "units           metal",
        "atom_style      atomic",
        "atom_modify     map yes",
        "newton          on",
        "boundary        p p p",
        "read_data       structure.data",
        "",
        *_pair_block(framework, target, type_map),
        "",
        f"timestep        {timestep_ps:.16g}",
        f"thermo          {config['thermo_interval']}",
        "thermo_style    custom step time temp pe ke etotal press vol lx ly lz",
        f"velocity        all create {config['temperature_k']:.16g} {config['seed']} mom yes rot yes dist gaussian",
        f"dump            mlipflow all custom {config['dump_interval']} trajectory.lammpstrj id type element x y z vx vy vz",
        "dump_modify     mlipflow element " + " ".join(type_map) + " sort id",
    ]
    if config["ensemble"] == "nvt":
        lines.append(
            f"fix             mlipflow all nvt temp {config['temperature_k']:.16g} {config['temperature_k']:.16g} {tdamp_ps:.16g}"
        )
    else:
        pdamp_ps = float(config["barostat_damping_fs"]) / 1000.0
        pressure_bar = float(config["pressure_gpa"]) * 10000.0
        lines.append(
            f"fix             mlipflow all npt temp {config['temperature_k']:.16g} {config['temperature_k']:.16g} {tdamp_ps:.16g} iso {pressure_bar:.16g} {pressure_bar:.16g} {pdamp_ps:.16g}"
        )
    lines.extend(
        [
            f"run             {config['steps']}",
            "write_data      final.data",
            "write_restart   final.restart",
            "",
        ]
    )
    return "\n".join(lines)


def _launcher(framework: str, target: str, input_name: str) -> dict[str, Any]:
    interface = INTERFACES[framework]
    prefix = list(interface["gpu_launcher"]) if target == "gpu" else []
    return {
        "target": target,
        "executable": "site-owned-lammps",
        "argv_after_executable": [*prefix, "-in", input_name, "-var", "MODEL_FILE", "<site-resolved-model-file>"],
        "required_packages": list(interface[f"{target}_packages"]),
        "limitations": list(interface["gpu_limitations"]) if target == "gpu" else [],
    }


def prepare(structure: Path, model_reference: Path, config_path: Path, output_dir: Path) -> dict[str, Any]:
    structure = _ordinary_file(structure, "structure", MAX_STRUCTURE_BYTES)
    model_reference = _ordinary_file(model_reference, "model_reference", MAX_JSON_BYTES)
    config_path = _ordinary_file(config_path, "lammps_config", MAX_JSON_BYTES)
    output_dir = output_dir.expanduser().absolute()
    if output_dir.exists():
        raise ContractError("output_dir must not already exist")
    output_dir.mkdir(parents=True)

    model = _model_reference(model_reference)
    config = _config(config_path, model)

    from ase.io import read, write

    atoms = read(str(structure), index=-1)
    if not hasattr(atoms, "get_chemical_symbols") or len(atoms) == 0:
        raise ContractError("structure must select exactly one non-empty ASE Atoms object")
    present = set(atoms.get_chemical_symbols())
    missing = sorted(present - set(config["type_map"]))
    unused = sorted(set(config["type_map"]) - present)
    if missing:
        raise ContractError("structure contains species absent from type_map: " + ", ".join(missing))
    if unused:
        raise ContractError("type_map contains species absent from structure: " + ", ".join(unused))
    if atoms.cell.rank != 3 or not all(bool(value) for value in atoms.get_pbc()):
        raise ContractError("version 0.1 LAMMPS MD preparation requires a full-rank 3D periodic structure")

    structure_data = output_dir / "structure.data"
    write(
        str(structure_data),
        atoms,
        format="lammps-data",
        atom_style="atomic",
        specorder=config["type_map"],
        masses=True,
    )
    if not structure_data.is_file() or not 0 < structure_data.stat().st_size <= MAX_OUTPUT_BYTES:
        raise ContractError("ASE did not produce a bounded structure.data")

    generated: list[dict[str, Any]] = []
    launchers: list[dict[str, Any]] = []
    for target in config["targets"]:
        name = f"in.{target}.lammps"
        path = output_dir / name
        path.write_text(_deck(str(model["framework"]), target, config), encoding="utf-8")
        generated.append({"name": name, "sha256": _sha256(path), "size_bytes": path.stat().st_size})
        launchers.append(_launcher(str(model["framework"]), target, name))

    generated.insert(
        0,
        {"name": "structure.data", "sha256": _sha256(structure_data), "size_bytes": structure_data.stat().st_size},
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "plugin_id": PLUGIN_ID,
        "operation": OPERATION,
        "status": "OK",
        "ase_version": importlib.metadata.version("ase"),
        "input_fingerprints": {
            "structure": _sha256(structure),
            "model_reference": _sha256(model_reference),
            "lammps_config": _sha256(config_path),
        },
        "model": model,
        "md": config,
        "runtime_model_variable": "MODEL_FILE",
        "generated_files": generated,
        "launchers": launchers,
        "notes": [
            "Input decks contain no absolute model or cluster path.",
            "The execution layer must resolve the approved model fingerprint and pass -var MODEL_FILE <path>.",
            "MACE and MatGL inputs require LAMMPS-exported model artifacts, not raw training checkpoints.",
        ],
    }
    manifest_path = output_dir / "lammps-input-manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare deterministic MLIP LAMMPS inputs")
    parser.add_argument("--structure", required=True)
    parser.add_argument("--model-reference", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepare(Path(args.structure), Path(args.model_reference), Path(args.config), Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
