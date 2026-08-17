"""Deterministic pymatgen VASP-input preparation for ``dft-labeling``.

This executable never launches VASP or a scheduler.  It materializes inputs in a
fresh MLIPFlow attempt, records exact provenance, and deliberately excludes
POTCAR content from the portable result manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple


PLUGIN_ID = "dft-labeling"
OPERATION = "vasp-prepare"
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_STRUCTURE_BYTES = 64 * 1024 * 1024
MAX_STRUCTURES = 10000
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
SAFE_POTCAR_SYMBOL = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
SHA256 = re.compile(r"sha256:[0-9a-f]{64}")

MANUSCRIPT_STATIC_PRESET = "manuscript-static-v1"
MANUSCRIPT_STATIC_INCAR: dict[str, Any] = {
    "ISTART": 0,
    "ICHARG": 2,
    "LCHARG": False,
    "LWAVE": False,
    "LREAL": "Auto",
    "IALGO": 38,
    "EDIFF": 5e-6,
    "ISMEAR": 0,
    "SIGMA": 0.1,
    "PREC": "Normal",
    "NELM": 700,
    "NELMIN": 4,
    "ENCUT": 450,
    "IVDW": 12,
    "NSW": 0,
    "IBRION": -1,
    "ISIF": 2,
    "ISPIN": 2,
}
MANUSCRIPT_STATIC_KPOINTS = {
    "mode": "monkhorst",
    "grid": [1, 1, 1],
    "shift": [0, 0, 0],
}
MANUSCRIPT_STATIC_PROVENANCE = {
    "paper": {
        "locator": "manuscript-supplement://SI_0510zdl.docx#page=2&figure=S3",
        "sha256": "sha256:0e421d4236d8c9389eddd4e61f9503ada6ff0fd07f70f5bd1c32eea35214e732",
        "declared_parameters": {"ENCUT": 450, "EDIFF": 5e-6, "IALGO": 38},
        "note": (
            "Figure S3 labels EDIFF as eV/atom; VASP INCAR EDIFF is an absolute "
            "electronic stopping threshold, so the numerical value is preserved without "
            "silently converting units."
        ),
    },
    "historical_template": {
        "incar_locator": "remote-mlp://4-Element/Li8/scf/single/INCAR",
        "incar_sha256": "sha256:69fbece3f536be6aa275d83e39a29bedda4d0ddc0a704d47ba7200642864b01c",
        "kpoints_locator": "remote-mlp://4-Element/Li8/scf/single/KPOINTS",
        "kpoints_sha256": "sha256:3eda09df03e3fa250fd362b3f1f97b8a89cd9612eaebbaea5dbac1e2cf7a73a6",
        "excluded_runtime_parameter": {
            "NPAR": 4,
            "reason": "hardware- and VASP-version-dependent parallelization setting",
        },
    },
}


class ContractError(ValueError):
    """Raised when input preparation would violate the reviewed contract."""


class PymatgenApi(NamedTuple):
    Structure: Any
    Incar: Any
    Kpoints: Any
    Poscar: Any
    Potcar: Any
    version: str
    configured_psp_root: str | None


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    return any(candidate.is_symlink() for candidate in (absolute, *absolute.parents))


def _has_symlink_below(path: Path, root: Path) -> bool:
    candidate = path.expanduser().absolute()
    base = root.expanduser().absolute()
    try:
        relative = candidate.relative_to(base)
    except ValueError:
        return True
    current = base
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _ordinary_file(path: Path, label: str, max_bytes: int) -> Path:
    unresolved = path.expanduser().absolute()
    if unresolved.is_symlink():
        raise ContractError(f"{label} must not contain symlink path components")
    resolved = unresolved.resolve()
    if not resolved.is_file():
        raise ContractError(f"{label} must be an ordinary file")
    size = resolved.stat().st_size
    if size < 1 or size > max_bytes:
        raise ContractError(f"{label} size must be between 1 and {max_bytes} bytes")
    return resolved


def _read_json(path: Path, label: str, max_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    path = _ordinary_file(path, label, max_bytes)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_text(payload, encoding="utf-8")


def _safe_relative(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip() or any(c in value for c in "\x00\r\n"):
        raise ContractError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise ContractError(f"{label} must be a confined relative path")
    return path


def _portable_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise ContractError(f"{label} must match {SAFE_ID.pattern}")
    return value


def _fingerprint(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ContractError(f"{label} must be a full sha256 fingerprint")
    return value


def _json_incar_value(value: Any, key: str) -> Any:
    if value is None or isinstance(value, Mapping):
        raise ContractError(f"INCAR {key} must be a JSON scalar or flat list")
    if isinstance(value, list):
        if not value or any(item is None or isinstance(item, (list, Mapping)) for item in value):
            raise ContractError(f"INCAR {key} must be a non-empty flat list")
        return [_json_incar_value(item, key) for item in value]
    if not isinstance(value, (str, int, float, bool)):
        raise ContractError(f"INCAR {key} has an unsupported value type")
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(f"INCAR {key} must be finite")
    return value


def _incar_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be an object")
    result: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", raw_key):
            raise ContractError(f"{label} contains an invalid INCAR key")
        key = raw_key.upper()
        if key in result:
            raise ContractError(f"{label} contains a duplicate normalized key: {key}")
        result[key] = _json_incar_value(raw_value, key)
    return result


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ContractError(f"{label} must be a positive integer")
    return value


def _validate_calculation_incar(calculation_type: str, incar: Mapping[str, Any]) -> None:
    if calculation_type == "static":
        if incar.get("NSW") != 0 or incar.get("IBRION") != -1:
            raise ContractError("static inputs require NSW=0 and IBRION=-1")
        return
    if calculation_type == "relax":
        if not isinstance(incar.get("NSW"), int) or int(incar["NSW"]) < 1:
            raise ContractError("relax inputs require positive NSW")
        if incar.get("IBRION") not in {1, 2, 3}:
            raise ContractError("relax inputs require IBRION=1, 2, or 3")
        for key in ("EDIFFG", "ISIF"):
            if key not in incar:
                raise ContractError(f"relax inputs require explicit {key}")
        return
    if calculation_type == "aimd":
        if incar.get("IBRION") != 0:
            raise ContractError("aimd inputs require IBRION=0")
        for key in ("NSW", "POTIM"):
            value = incar.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0:
                raise ContractError(f"aimd inputs require positive {key}")
        for key in ("TEBEG", "TEEND", "MDALGO"):
            if key not in incar:
                raise ContractError(f"aimd inputs require explicit {key}")
        return
    raise ContractError("calculation_type must be static, relax, or aimd")


def _kpoints_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError("kpoints must be an object with an explicit grid")
    unknown = set(value) - {"mode", "grid", "shift"}
    if unknown:
        raise ContractError("unsupported kpoints field(s): " + ", ".join(sorted(unknown)))
    mode = value.get("mode")
    if mode not in {"gamma", "monkhorst"}:
        raise ContractError("kpoints.mode must be gamma or monkhorst")
    grid = value.get("grid")
    if (
        not isinstance(grid, list)
        or len(grid) != 3
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in grid)
    ):
        raise ContractError("kpoints.grid must contain three positive integers")
    shift = value.get("shift", [0, 0, 0])
    if (
        not isinstance(shift, list)
        or len(shift) != 3
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            for item in shift
        )
    ):
        raise ContractError("kpoints.shift must contain three finite numbers")
    return {"mode": mode, "grid": list(grid), "shift": list(shift)}


def _labeling_config(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "schema_version",
        "engine",
        "calculation_type",
        "preset",
        "incar",
        "kpoints",
        "sort_structure",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ContractError("unsupported labeling_config field(s): " + ", ".join(sorted(unknown)))
    if value.get("schema_version") != 1 or value.get("engine") != "vasp":
        raise ContractError("labeling_config requires schema_version=1 and engine=vasp")
    calculation_type = value.get("calculation_type")
    if calculation_type not in {"static", "relax", "aimd"}:
        raise ContractError("calculation_type must be static, relax, or aimd")
    preset = value.get("preset")
    if preset is not None and preset != MANUSCRIPT_STATIC_PRESET:
        raise ContractError(f"unsupported preset: {preset}")
    if preset == MANUSCRIPT_STATIC_PRESET and calculation_type != "static":
        raise ContractError(f"{MANUSCRIPT_STATIC_PRESET} is valid only for static calculations")
    provided_incar = _incar_mapping(value.get("incar", {}), "incar")
    if preset == MANUSCRIPT_STATIC_PRESET:
        effective_incar = dict(MANUSCRIPT_STATIC_INCAR)
        effective_incar.update(provided_incar)
        kpoints = _kpoints_config(value.get("kpoints", MANUSCRIPT_STATIC_KPOINTS))
    else:
        if not provided_incar:
            raise ContractError("an explicit non-empty incar object is required without a preset")
        effective_incar = provided_incar
        kpoints = _kpoints_config(value.get("kpoints"))
    _validate_calculation_incar(str(calculation_type), effective_incar)
    sort_structure = value.get("sort_structure", True)
    if not isinstance(sort_structure, bool):
        raise ContractError("sort_structure must be boolean")
    return {
        "calculation_type": calculation_type,
        "preset": preset,
        "incar_overrides": provided_incar if preset else {},
        "effective_incar": effective_incar,
        "kpoints": kpoints,
        "sort_structure": sort_structure,
    }


def _pseudopotential_reference(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "schema_version",
        "reference_id",
        "source_env",
        "license_acknowledged",
        "functional",
        "symbols",
        "expected_component_sha256",
        "expected_combined_sha256",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ContractError(
            "unsupported pseudopotential_reference field(s): " + ", ".join(sorted(unknown))
        )
    if value.get("schema_version") != 1:
        raise ContractError("pseudopotential_reference schema_version must be 1")
    reference_id = value.get("reference_id")
    if (
        not isinstance(reference_id, str)
        or not reference_id.strip()
        or reference_id.startswith("/")
        or ".." in Path(reference_id.replace("://", "/")).parts
    ):
        raise ContractError("reference_id must be portable and must not contain an absolute path")
    if value.get("source_env") != "PMG_VASP_PSP_DIR":
        raise ContractError("source_env must be PMG_VASP_PSP_DIR")
    if value.get("license_acknowledged") is not True:
        raise ContractError("license_acknowledged must be true")
    functional = value.get("functional")
    if not isinstance(functional, str) or not functional.strip():
        raise ContractError("functional must be an explicit pymatgen POTCAR functional")
    raw_symbols = value.get("symbols")
    if not isinstance(raw_symbols, Mapping) or not raw_symbols:
        raise ContractError("symbols must explicitly map every element to a POTCAR symbol")
    symbols: dict[str, str] = {}
    for element, symbol in raw_symbols.items():
        if (
            not isinstance(element, str)
            or not re.fullmatch(r"[A-Z][a-z]?", element)
            or not isinstance(symbol, str)
            or not SAFE_POTCAR_SYMBOL.fullmatch(symbol)
        ):
            raise ContractError("symbols contains an invalid element or POTCAR symbol")
        symbols[element] = symbol
    expected_components = value.get("expected_component_sha256", {})
    if not isinstance(expected_components, Mapping):
        raise ContractError("expected_component_sha256 must be an object")
    normalized_expected: dict[str, str] = {}
    for symbol, digest in expected_components.items():
        if not isinstance(symbol, str) or not SAFE_POTCAR_SYMBOL.fullmatch(symbol):
            raise ContractError("expected_component_sha256 contains an invalid symbol")
        normalized_expected[symbol] = _fingerprint(digest, f"expected POTCAR hash for {symbol}")
    expected_combined = value.get("expected_combined_sha256")
    if expected_combined is not None:
        expected_combined = _fingerprint(expected_combined, "expected_combined_sha256")
    return {
        "reference_id": reference_id,
        "functional": functional,
        "symbols": symbols,
        "expected_component_sha256": normalized_expected,
        "expected_combined_sha256": expected_combined,
    }


def _structure_records(
    manifest: Mapping[str, Any], manifest_path: Path, project_root: Path, max_structures: int
) -> list[dict[str, Any]]:
    raw = manifest.get("structures")
    if manifest.get("schema_version") != 1 or not isinstance(raw, list):
        raise ContractError("structures_manifest requires schema_version=1 and a structures list")
    if not 1 <= len(raw) <= max_structures:
        raise ContractError(f"structures count must be 1..{max_structures}")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw, 1):
        if not isinstance(item, Mapping):
            raise ContractError(f"structure record {index} must be an object")
        structure_id = item.get("id", item.get("structure_id"))
        structure_id = _portable_id(structure_id, f"structure record {index} id")
        if structure_id in seen:
            raise ContractError(f"duplicate structure id: {structure_id}")
        seen.add(structure_id)
        relative_value = item.get("path", item.get("output_file"))
        relative = _safe_relative(relative_value, f"structure {structure_id} path")
        source = (manifest_path.parent / relative).absolute()
        if _has_symlink_below(source, project_root) or not _within(source.resolve(), project_root):
            raise ContractError(f"structure {structure_id} must remain inside project_root")
        source = _ordinary_file(source, f"structure {structure_id}", MAX_STRUCTURE_BYTES)
        declared = item.get("fingerprint", item.get("frame_sha256", item.get("sha256")))
        declared = _fingerprint(declared, f"structure {structure_id} fingerprint")
        actual = _sha256(source)
        if actual != declared:
            raise ContractError(f"structure {structure_id} fingerprint does not match its source file")
        records.append(
            {
                "id": structure_id,
                "relative_path": relative.as_posix(),
                "source_path": source,
                "fingerprint": actual,
            }
        )
    return records


def _load_pymatgen() -> PymatgenApi:
    try:
        from pymatgen.core import SETTINGS, Structure
        from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar
    except (ImportError, ModuleNotFoundError) as exc:
        raise ContractError("pymatgen with VASP input support is required") from exc
    try:
        version = importlib.metadata.version("pymatgen")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ContractError("installed pymatgen distribution version is required") from exc
    configured_root = SETTINGS.get("PMG_VASP_PSP_DIR")
    return PymatgenApi(Structure, Incar, Kpoints, Poscar, Potcar, version, configured_root)


def _runtime_psp_configuration(api: PymatgenApi) -> str:
    configured_root = os.environ.get("PMG_VASP_PSP_DIR")
    source = "environment"
    if not isinstance(configured_root, str) or not configured_root.strip():
        configured_root = api.configured_psp_root
        source = "pymatgen-settings"
    if not isinstance(configured_root, str) or not configured_root.strip():
        raise ContractError(
            "PMG_VASP_PSP_DIR must be configured by environment or pymatgen settings"
        )
    root = Path(configured_root).expanduser()
    if not root.is_absolute() or not root.resolve().is_dir():
        raise ContractError("configured PMG_VASP_PSP_DIR must resolve to an existing directory")
    return source


def _potcar_component_records(potcar: Iterable[Any], symbols: list[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for symbol, single in zip(symbols, potcar):
        records.append(
            {
                "symbol": symbol,
                "sha256": _sha256_bytes(str(single).encode("utf-8")),
            }
        )
    if len(records) != len(symbols):
        raise ContractError("pymatgen POTCAR component count differs from POSCAR species count")
    return records


def _verify_expected_potcar(
    reference: Mapping[str, Any], components: list[dict[str, str]], combined_sha256: str
) -> None:
    expected = reference["expected_component_sha256"]
    for component in components:
        expected_digest = expected.get(component["symbol"])
        if expected_digest is not None and component["sha256"] != expected_digest:
            raise ContractError(f"POTCAR component hash mismatch for {component['symbol']}")
    expected_combined = reference.get("expected_combined_sha256")
    if expected_combined is not None and combined_sha256 != expected_combined:
        raise ContractError("combined POTCAR hash differs from the approved reference")


def _file_record(
    actual_path: Path, portable_path: Path, media_type: str, collectable: bool
) -> dict[str, Any]:
    return {
        "path": portable_path.as_posix(),
        "sha256": _sha256(actual_path),
        "size_bytes": actual_path.stat().st_size,
        "media_type": media_type,
        "collectable": collectable,
    }


def prepare_inputs(args: argparse.Namespace, api: PymatgenApi | None = None) -> dict[str, Any]:
    project_root = Path(args.project_root).expanduser().absolute().resolve()
    attempt_dir = Path(args.attempt_dir).expanduser().absolute()
    if attempt_dir.is_symlink() or not attempt_dir.is_dir():
        raise ContractError("attempt_dir must be an existing ordinary directory")
    attempt_dir = attempt_dir.resolve()
    if not _within(attempt_dir, project_root):
        raise ContractError("attempt_dir must remain inside project_root")
    output_relative = _safe_relative(args.output_subdir, "output_subdir")
    output_dir = (attempt_dir / output_relative).absolute()
    if _has_symlink_below(output_dir, attempt_dir) or not _within(
        output_dir.parent.resolve(), attempt_dir
    ):
        raise ContractError("output_subdir must remain inside attempt_dir without symlinks")
    result_path = Path(args.result_manifest).expanduser().absolute()
    if (
        _has_symlink_below(result_path, attempt_dir)
        or not _within(result_path.parent.resolve(), attempt_dir)
        or result_path.parent.resolve() != attempt_dir
    ):
        raise ContractError("result_manifest must be a direct child of attempt_dir")
    if result_path == output_dir or result_path in output_dir.parents:
        raise ContractError("output_subdir must not overlap result_manifest")
    if output_dir.exists() or result_path.exists():
        raise ContractError("vasp-prepare requires fresh output and result paths")
    max_structures = _positive_int(args.max_structures, "max_structures")
    if max_structures > MAX_STRUCTURES:
        raise ContractError(f"max_structures must not exceed {MAX_STRUCTURES}")

    structures_path = _ordinary_file(
        Path(args.structures_manifest), "structures_manifest", MAX_JSON_BYTES
    )
    labeling_path = _ordinary_file(Path(args.labeling_config), "labeling_config", MAX_JSON_BYTES)
    pseudopotential_path = _ordinary_file(
        Path(args.pseudopotential_reference), "pseudopotential_reference", MAX_JSON_BYTES
    )
    for path, label in (
        (structures_path, "structures_manifest"),
        (labeling_path, "labeling_config"),
        (pseudopotential_path, "pseudopotential_reference"),
    ):
        if _has_symlink_below(path, project_root) or not _within(path, project_root):
            raise ContractError(f"{label} must remain inside project_root")

    structures_manifest = _read_json(structures_path, "structures_manifest")
    labeling_value = _read_json(labeling_path, "labeling_config")
    pseudopotential_value = _read_json(
        pseudopotential_path, "pseudopotential_reference"
    )
    config = _labeling_config(labeling_value)
    reference = _pseudopotential_reference(pseudopotential_value)
    records = _structure_records(
        structures_manifest, structures_path, project_root, max_structures
    )
    api = api or _load_pymatgen()
    psp_configuration_source = _runtime_psp_configuration(api)

    output_dir.mkdir(parents=True)
    incomplete = output_dir / "INCOMPLETE.json"
    _write_json(
        incomplete,
        {
            "schema_version": 1,
            "plugin_id": PLUGIN_ID,
            "operation": OPERATION,
            "status": "INCOMPLETE",
        },
    )
    calculations: list[dict[str, Any]] = []
    for order, record in enumerate(records, 1):
        structure_id = record["id"]
        directory_name = f"{order:06d}-{structure_id}"
        calculation_dir = output_dir / directory_name
        calculation_dir.mkdir()
        try:
            structure = api.Structure.from_file(str(record["source_path"]))
            poscar = api.Poscar(structure, sort_structure=config["sort_structure"])
            poscar_path = calculation_dir / "POSCAR"
            incar_path = calculation_dir / "INCAR"
            kpoints_path = calculation_dir / "KPOINTS"
            potcar_path = calculation_dir / "POTCAR"
            poscar.write_file(str(poscar_path))
            api.Incar(config["effective_incar"]).write_file(str(incar_path))
            kpoints_config = config["kpoints"]
            if kpoints_config["mode"] == "gamma":
                kpoints = api.Kpoints.gamma_automatic(
                    kpts=tuple(kpoints_config["grid"]),
                    shift=tuple(kpoints_config["shift"]),
                )
            else:
                kpoints = api.Kpoints.monkhorst_automatic(
                    kpts=tuple(kpoints_config["grid"]),
                    shift=tuple(kpoints_config["shift"]),
                )
            kpoints.write_file(str(kpoints_path))
            elements = list(poscar.site_symbols)
            missing = [element for element in elements if element not in reference["symbols"]]
            if missing:
                raise ContractError(
                    "pseudopotential_reference lacks element mapping(s): "
                    + ", ".join(sorted(set(missing)))
                )
            potcar_symbols = [reference["symbols"][element] for element in elements]
            try:
                potcar = api.Potcar(potcar_symbols, functional=reference["functional"])
                potcar.write_file(str(potcar_path))
            except ContractError:
                raise
            except Exception as exc:
                raise ContractError(
                    "pymatgen could not assemble POTCAR from the configured PMG_VASP_PSP_DIR"
                ) from exc
            component_records = _potcar_component_records(potcar, potcar_symbols)
            combined_sha256 = _sha256(potcar_path)
            _verify_expected_potcar(reference, component_records, combined_sha256)
        except ContractError:
            raise
        except Exception as exc:
            raise ContractError(f"pymatgen failed to prepare structure {structure_id}") from exc

        relative_directory = (output_relative / directory_name).as_posix()
        files = {
            "POSCAR": _file_record(
                poscar_path,
                Path(relative_directory) / "POSCAR",
                "chemical/x-vasp-poscar",
                True,
            ),
            "INCAR": _file_record(
                incar_path, Path(relative_directory) / "INCAR", "text/plain", True
            ),
            "KPOINTS": _file_record(
                kpoints_path, Path(relative_directory) / "KPOINTS", "text/plain", True
            ),
            "POTCAR": _file_record(
                potcar_path,
                Path(relative_directory) / "POTCAR",
                "application/x-vasp-potcar",
                False,
            ),
        }
        calculations.append(
            {
                "order": order,
                "structure_id": structure_id,
                "source": {
                    "path": record["relative_path"],
                    "sha256": record["fingerprint"],
                },
                "directory": relative_directory,
                "formula": str(structure.composition.reduced_formula),
                "atom_count": len(structure),
                "files": files,
                "potcar": {
                    "reference_id": reference["reference_id"],
                    "source_env": "PMG_VASP_PSP_DIR",
                    "functional": reference["functional"],
                    "elements": elements,
                    "symbols": potcar_symbols,
                    "components": component_records,
                    "combined_sha256": combined_sha256,
                    "portable_artifact": False,
                },
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "plugin_id": PLUGIN_ID,
        "operation": OPERATION,
        "status": "OK",
        "engine": "vasp",
        "generator": {"name": "pymatgen", "version": api.version},
        "runtime": {
            "python_executable": str(Path(sys.executable).expanduser().absolute().resolve()),
            "python_version": platform.python_version(),
            "pymatgen_version": api.version,
            "prepare_wrapper_sha256": _sha256(Path(__file__).resolve()),
        },
        "calculation_type": config["calculation_type"],
        "preset": config["preset"],
        "parameter_provenance": (
            MANUSCRIPT_STATIC_PROVENANCE if config["preset"] == MANUSCRIPT_STATIC_PRESET else {}
        ),
        "incar": {
            "effective": config["effective_incar"],
            "overrides": config["incar_overrides"],
        },
        "kpoints": config["kpoints"],
        "sort_structure": config["sort_structure"],
        "input_fingerprints": {
            "structures_manifest": _sha256(structures_path),
            "labeling_config": _sha256(labeling_path),
            "pseudopotential_reference": _sha256(pseudopotential_path),
        },
        "structure_count": len(calculations),
        "execution_ready": True,
        "potcar_policy": {
            "source_env": "PMG_VASP_PSP_DIR",
            "configuration_source": psp_configuration_source,
            "materialized_in_attempt": True,
            "portable_or_collectable": False,
        },
        "calculations": calculations,
    }
    _write_json(result_path, manifest)
    incomplete.unlink()
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare deterministic VASP inputs with pymatgen")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--attempt-dir", required=True)
    parser.add_argument("--structures-manifest", required=True)
    parser.add_argument("--labeling-config", required=True)
    parser.add_argument("--pseudopotential-reference", required=True)
    parser.add_argument("--result-manifest", required=True)
    parser.add_argument("--output-subdir", required=True)
    parser.add_argument("--max-structures", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        prepare_inputs(args)
    except ContractError as exc:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "plugin_id": PLUGIN_ID,
                    "operation": OPERATION,
                    "status": "FAIL",
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
