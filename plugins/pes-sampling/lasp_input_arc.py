"""Deterministic LASP-compatible single-structure ARC formatting."""
from __future__ import annotations

import math
import re
from typing import Any, Iterable, Sequence


CANONICAL_HEADER = b"!BIOSYM archive 2\nPBC=ON\n"
ASE_HEADER = b"!BIOSYM archive 3\nPBC=ON\n"
ELEMENT = re.compile(r"[A-Z][a-z]?")


class LaspInputArcError(ValueError):
    pass


def _finite(values: Iterable[float], field: str) -> list[float]:
    result = [float(value) for value in values]
    if not result or any(not math.isfinite(value) for value in result):
        raise LaspInputArcError(f"{field} must contain finite values")
    return result


def _render(
    cell_parameters: Sequence[float],
    records: Sequence[tuple[str, Sequence[float]]],
) -> bytes:
    cell = _finite(cell_parameters, "cell parameters")
    if len(cell) != 6 or any(value <= 0 for value in cell[:3]):
        raise LaspInputArcError("cell parameters must contain positive a, b, c and three angles")
    if not records:
        raise LaspInputArcError("LASP input ARC requires at least one atom")
    lines = [
        "!BIOSYM archive 2\n",
        "PBC=ON\n",
        "Energy 1 0.000000 0.000000\n",
        "!DATE\n",
        "PBC " + " ".join(f"{value:.9f}" for value in cell) + "\n",
    ]
    for index, (symbol, raw_position) in enumerate(records, 1):
        if ELEMENT.fullmatch(symbol) is None:
            raise LaspInputArcError(f"invalid element symbol: {symbol!r}")
        position = _finite(raw_position, f"position {index}")
        if len(position) != 3:
            raise LaspInputArcError("each atom position must contain three coordinates")
        lines.append(
            f"{symbol:<4s} {position[0]:15.9f} {position[1]:15.9f} "
            f"{position[2]:15.9f} CORE {index:4d} {symbol:<2s} {symbol:<2s} "
            f"0.0000 {index:4d}\n"
        )
    lines.extend(["end\n", "end\n"])
    return "".join(lines).encode("ascii")


def canonical_from_atoms(atoms: Any) -> bytes:
    """Render one periodic ASE Atoms object in the historical LASP input form."""

    wrapped = atoms.copy()
    wrapped.wrap()
    cell = [float(value) for value in wrapped.cell.cellpar()]
    records = list(zip(wrapped.get_chemical_symbols(), wrapped.get_positions()))
    return _render(cell, records)


def _validate_canonical(payload: bytes) -> None:
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise LaspInputArcError("LASP input ARC must be ASCII") from exc
    if lines[:2] != ["!BIOSYM archive 2", "PBC=ON"]:
        raise LaspInputArcError("LASP input ARC header is invalid")
    if sum(line.lstrip().startswith("Energy") for line in lines) != 1:
        raise LaspInputArcError("LASP input ARC must contain exactly one Energy record")
    if sum(line.strip() == "end" for line in lines) != 2:
        raise LaspInputArcError("LASP input ARC must contain exactly one complete frame")


def canonicalize_prepared_arc(payload: bytes) -> tuple[bytes, dict[str, Any]]:
    """Accept canonical v2 or normalize ASE's deterministic single-frame v3 ARC."""

    if payload.startswith(CANONICAL_HEADER):
        _validate_canonical(payload)
        return payload, {
            "source_format": "biosym-archive-2",
            "canonical_format": "biosym-archive-2",
            "canonicalization_applied": False,
            "coordinates_wrapped": False,
            "energy_placeholder_ev": 0.0,
            "energy_semantics": "input-format-placeholder-not-a-label",
        }
    if not payload.startswith(ASE_HEADER):
        raise LaspInputArcError("unsupported prepared ARC header")
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise LaspInputArcError("prepared ARC must be ASCII") from exc
    pbc_indexes = [index for index, line in enumerate(lines) if line.startswith("PBC ")]
    if len(pbc_indexes) != 1 or sum(line.strip() == "end" for line in lines) != 2:
        raise LaspInputArcError("ASE prepared ARC must contain exactly one complete frame")
    pbc_index = pbc_indexes[0]
    pbc_tokens = lines[pbc_index].split()
    if len(pbc_tokens) != 7:
        raise LaspInputArcError("ASE prepared ARC cell record is invalid")
    cell = _finite((float(value) for value in pbc_tokens[1:]), "cell parameters")
    records: list[tuple[str, Sequence[float]]] = []
    for line in lines[pbc_index + 1 :]:
        if line.strip() == "end":
            break
        tokens = line.split()
        if len(tokens) < 9:
            raise LaspInputArcError("ASE prepared ARC atom record is invalid")
        symbol = tokens[-2]
        if ELEMENT.fullmatch(symbol) is None:
            raise LaspInputArcError("ASE prepared ARC element record is invalid")
        records.append((symbol, [float(value) for value in tokens[1:4]]))
    canonical = _render(cell, records)
    _validate_canonical(canonical)
    return canonical, {
        "source_format": "ase-dmol-archive-3",
        "canonical_format": "biosym-archive-2",
        "canonicalization_applied": True,
        "coordinates_wrapped": False,
        "energy_placeholder_ev": 0.0,
        "energy_semantics": "input-format-placeholder-not-a-label",
    }
