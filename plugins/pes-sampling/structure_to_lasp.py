"""Convert one explicit ASE-readable structure into a LASP input ARC."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple


SAFE_POTCAR_SYMBOL = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


class ConversionError(RuntimeError):
    pass


sys.path.insert(0, str(Path(__file__).resolve().parent))
from lasp_input_arc import canonical_from_atoms  # noqa: E402


class PymatgenApi(NamedTuple):
    Potcar: Any
    configured_psp_root: str | None


def _fresh_directory(path: Path) -> Path:
    path = path.expanduser().absolute()
    if path.is_symlink() or path.exists():
        raise ConversionError("output directory must be a fresh non-symlink path")
    path.mkdir(parents=True)
    return path


def _minimum_distance(atoms: Any) -> float | None:
    if len(atoms) < 2:
        return None
    import numpy as np

    distances = atoms.get_all_distances(mic=bool(any(atoms.get_pbc())))
    distances[np.diag_indices(len(atoms))] = np.inf
    value = float(np.min(distances))
    if not math.isfinite(value) or value <= 0:
        raise ConversionError("input structure has a non-finite or non-positive pair distance")
    return value


def _pseudopotential_reference(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConversionError("pseudopotential reference must be a file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConversionError("pseudopotential reference must be valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ConversionError("pseudopotential reference must contain an object")
    allowed = {
        "schema_version",
        "reference_id",
        "source_env",
        "license_acknowledged",
        "functional",
        "symbols",
    }
    if set(value) - allowed:
        raise ConversionError("pseudopotential reference contains unsupported fields")
    if value.get("schema_version") != 1 or value.get("source_env") != "PMG_VASP_PSP_DIR":
        raise ConversionError(
            "pseudopotential reference requires schema_version=1 and PMG_VASP_PSP_DIR"
        )
    if value.get("license_acknowledged") is not True:
        raise ConversionError("pseudopotential license acknowledgement is required")
    reference_id = value.get("reference_id")
    if (
        not isinstance(reference_id, str)
        or not reference_id.strip()
        or reference_id.startswith("/")
        or ".." in Path(reference_id.replace("://", "/")).parts
    ):
        raise ConversionError("pseudopotential reference_id must be portable")
    functional = value.get("functional")
    if not isinstance(functional, str) or not functional.strip():
        raise ConversionError("pseudopotential functional must be explicit")
    raw_symbols = value.get("symbols")
    if not isinstance(raw_symbols, Mapping) or not raw_symbols:
        raise ConversionError("pseudopotential symbols must be an explicit mapping")
    symbols: dict[str, str] = {}
    for element, symbol in raw_symbols.items():
        if (
            not isinstance(element, str)
            or re.fullmatch(r"[A-Z][a-z]?", element) is None
            or not isinstance(symbol, str)
            or SAFE_POTCAR_SYMBOL.fullmatch(symbol) is None
        ):
            raise ConversionError("pseudopotential symbols mapping is invalid")
        symbols[element] = symbol
    return {
        "reference_id": reference_id,
        "functional": functional,
        "symbols": symbols,
    }


def _load_pymatgen() -> PymatgenApi:
    try:
        from pymatgen.core import SETTINGS
        from pymatgen.io.vasp.inputs import Potcar
    except (ImportError, ModuleNotFoundError) as exc:
        raise ConversionError("pymatgen with VASP POTCAR support is required") from exc
    return PymatgenApi(Potcar=Potcar, configured_psp_root=SETTINGS.get("PMG_VASP_PSP_DIR"))


def _runtime_psp_configuration(api: PymatgenApi) -> str:
    configured_root = os.environ.get("PMG_VASP_PSP_DIR")
    source = "environment"
    if not isinstance(configured_root, str) or not configured_root.strip():
        configured_root = api.configured_psp_root
        source = "pymatgen-settings"
    if not isinstance(configured_root, str) or not configured_root.strip():
        raise ConversionError("PMG_VASP_PSP_DIR must be configured")
    root = Path(configured_root).expanduser()
    if not root.is_absolute() or not root.resolve().is_dir():
        raise ConversionError("configured PMG_VASP_PSP_DIR must be an existing directory")
    return source


def _component_records(
    potcar: Iterable[Any], symbols: list[str]
) -> list[dict[str, str]]:
    singles = list(potcar)
    if len(singles) != len(symbols):
        raise ConversionError("POTCAR component count differs from the element order")
    return [{"symbol": symbol} for symbol in symbols]


def _prepare_potcar(
    atoms: Any,
    reference_path: Path,
    output_dir: Path,
    api: PymatgenApi | None = None,
) -> dict[str, Any]:
    reference = _pseudopotential_reference(reference_path)
    element_order = list(dict.fromkeys(atoms.get_chemical_symbols()))
    if set(reference["symbols"]) != set(element_order):
        raise ConversionError(
            "pseudopotential element mapping must exactly match the selected structure"
        )
    potcar_symbols = [reference["symbols"][element] for element in element_order]
    api = api or _load_pymatgen()
    configuration_source = _runtime_psp_configuration(api)
    try:
        potcar = api.Potcar(potcar_symbols, functional=reference["functional"])
        path = output_dir / "POTCAR"
        potcar.write_file(str(path))
    except ConversionError:
        raise
    except Exception as exc:
        raise ConversionError(
            "pymatgen could not assemble the approved POTCAR"
        ) from exc
    components = _component_records(potcar, potcar_symbols)
    return {
        "reference_id": reference["reference_id"],
        "source_env": "PMG_VASP_PSP_DIR",
        "configuration_source": configuration_source,
        "functional": reference["functional"],
        "elements": element_order,
        "symbols": potcar_symbols,
        "components": components,
        "portable_artifact": False,
        "output": {
            "path": "POTCAR",
            "collectable": False,
        },
    }


def run(args: argparse.Namespace, api: PymatgenApi | None = None) -> dict[str, Any]:
    source = Path(args.input_structure).expanduser().absolute()
    if not source.is_file():
        raise ConversionError("input structure must be a file")
    source = source.resolve()
    output_dir = _fresh_directory(Path(args.output_dir))

    from ase.io import read

    atoms = read(
        str(source),
        format=args.input_format,
        index=str(args.input_index),
    )
    if not hasattr(atoms, "get_positions") or len(atoms) < 1:
        raise ConversionError("input selection must produce exactly one non-empty structure")
    if atoms.cell.rank != 3 or not all(bool(value) for value in atoms.get_pbc()):
        raise ConversionError("LASP input preparation requires a full-rank 3D periodic cell")
    lengths = [float(value) for value in atoms.cell.lengths()]
    if any(not math.isfinite(value) or value <= 0 for value in lengths):
        raise ConversionError("input structure has an invalid cell length")
    minimum_cell = args.minimum_cell_length_angstrom
    if minimum_cell is not None:
        if not math.isfinite(minimum_cell) or minimum_cell <= 0:
            raise ConversionError("minimum cell length must be finite and positive")
        if any(value <= minimum_cell for value in lengths):
            raise ConversionError("input structure does not satisfy the strict cell-length bound")

    output = output_dir / "input.arc"
    with output.open("xb") as stream:
        stream.write(canonical_from_atoms(atoms))
    if not output.is_file() or output.stat().st_size == 0:
        raise ConversionError("ASE did not produce a non-empty ARC structure")

    symbols: dict[str, int] = {}
    for symbol in atoms.get_chemical_symbols():
        symbols[symbol] = symbols.get(symbol, 0) + 1
    potcar = None
    if args.pseudopotential_reference is not None:
        reference_path = Path(args.pseudopotential_reference).expanduser().absolute()
        reference_path = reference_path.resolve()
        potcar = _prepare_potcar(atoms, reference_path, output_dir, api=api)
    manifest = {
        "schema_version": 1,
        "plugin_id": "pes-sampling",
        "operation": "lasp-input-prepare",
        "source": {
            "basename": source.name,
            "path": str(source),
            "input_format": args.input_format,
            "input_index": str(args.input_index),
        },
        "structure": {
            "atom_count": len(atoms),
            "composition": symbols,
            "cell_lengths_A": lengths,
            "minimum_pair_distance_A": _minimum_distance(atoms),
            "minimum_cell_length_requirement_A": minimum_cell,
        },
        "output": {
            "path": output.name,
        },
        "arc_contract": {
            "format": "biosym-archive-2",
            "coordinates_wrapped": True,
            "energy_placeholder_ev": 0.0,
            "energy_semantics": "input-format-placeholder-not-a-label",
        },
    }
    if potcar is not None:
        manifest["pseudopotential_reference_path"] = str(reference_path)
        manifest["potcar"] = potcar
    manifest_path = output_dir / "lasp-input-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare one LASP ARC input structure")
    parser.add_argument("--input-structure", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-format")
    parser.add_argument("--input-index", default="-1")
    parser.add_argument("--minimum-cell-length-angstrom", type=float)
    parser.add_argument("--pseudopotential-reference")
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
