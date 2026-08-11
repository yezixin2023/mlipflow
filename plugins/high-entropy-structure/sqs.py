"""Seeded icet SQS generator for the high-entropy sulfide workflows.

The historical scripts hard-coded prototype paths, metal counts, supercells,
cutoffs and an unseeded Monte Carlo search. This wrapper exposes those values
in a versioned composition manifest and calls icet's public
``generate_sqs_from_supercells`` API with an explicit per-candidate seed.

ASE and icet are imported only during execution, so validation also works on
hosts without the optional scientific dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
from pathlib import Path
from typing import Any


PLUGIN_ID = "high-entropy-structure"
IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
FORMATS = {
    "vasp": (".vasp", "chemical/x-vasp-poscar"),
    "cif": (".cif", "chemical/x-cif"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read composition manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("composition manifest root must be an object")
    return value


def validate_manifest(
    value: dict[str, Any], *, prototype_symbols: list[str] | None = None
) -> dict[str, Any]:
    """Return a normalized manifest or raise before any file is written."""

    if value.get("schema_version") != 1:
        raise ValueError("composition manifest schema_version must equal 1")
    sublattice = value.get("alloy_sublattice")
    if not isinstance(sublattice, dict):
        raise ValueError("alloy_sublattice must be an object")
    prototype_species = sublattice.get("prototype_species")
    allowed_species = sublattice.get("allowed_species")
    if not isinstance(prototype_species, str) or not prototype_species:
        raise ValueError("alloy_sublattice.prototype_species is required")
    if not (
        isinstance(allowed_species, list)
        and len(allowed_species) >= 2
        and all(isinstance(item, str) and item for item in allowed_species)
        and len(allowed_species) == len(set(allowed_species))
        and prototype_species in allowed_species
    ):
        raise ValueError(
            "alloy_sublattice.allowed_species must be unique and include prototype_species"
        )
    cutoffs = value.get("cluster_cutoffs_angstrom")
    if not (
        isinstance(cutoffs, list)
        and cutoffs
        and all(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
            and float(item) > 0
            for item in cutoffs
        )
    ):
        raise ValueError("cluster_cutoffs_angstrom must contain positive finite values")
    repeat = value.get("supercell_repeat")
    if not (
        isinstance(repeat, list)
        and len(repeat) == 3
        and all(
            not isinstance(item, bool) and isinstance(item, int) and item > 0 for item in repeat
        )
    ):
        raise ValueError("supercell_repeat must contain three positive integers")
    n_steps = value.get("n_steps")
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    output_format = value.get("output_format", "vasp")
    if output_format not in FORMATS:
        raise ValueError("output_format must be vasp or cif")
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a non-empty array")

    expected_sites: int | None = None
    if prototype_symbols is not None:
        primitive_sites = sum(symbol == prototype_species for symbol in prototype_symbols)
        if primitive_sites < 1:
            raise ValueError(
                f"prototype contains no sites labelled {prototype_species!r} for the alloy sublattice"
            )
        expected_sites = primitive_sites * repeat[0] * repeat[1] * repeat[2]

    normalized_candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"candidate {index} must be an object")
        candidate_id = candidate.get("id")
        if not isinstance(candidate_id, str) or not IDENTIFIER.fullmatch(candidate_id):
            raise ValueError(f"candidate {index} has an invalid id")
        if candidate_id in seen:
            raise ValueError(f"duplicate candidate id: {candidate_id}")
        seen.add(candidate_id)
        counts = candidate.get("counts")
        if not isinstance(counts, dict) or set(counts) != set(allowed_species):
            raise ValueError(
                f"candidate {candidate_id} counts must contain exactly the allowed species"
            )
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in counts.values()
        ):
            raise ValueError(f"candidate {candidate_id} counts must be non-negative integers")
        site_count = sum(counts.values())
        if site_count < 1:
            raise ValueError(f"candidate {candidate_id} has no alloy atoms")
        if expected_sites is not None and site_count != expected_sites:
            raise ValueError(
                f"candidate {candidate_id} has {site_count} alloy atoms; prototype/repeat require "
                f"{expected_sites}"
            )
        normalized_candidates.append(
            {
                "id": candidate_id,
                "counts": {species: counts[species] for species in allowed_species},
                "alloy_site_count": site_count,
            }
        )
    return {
        "schema_version": 1,
        "alloy_sublattice": {
            "prototype_species": prototype_species,
            "allowed_species": list(allowed_species),
        },
        "cluster_cutoffs_angstrom": [float(item) for item in cutoffs],
        "supercell_repeat": list(repeat),
        "n_steps": n_steps,
        "output_format": output_format,
        "candidates": normalized_candidates,
        "alloy_site_count_from_prototype": expected_sites,
    }


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def generate(
    *,
    prototype_path: Path,
    composition_path: Path,
    seed: int,
    max_candidates: int,
    output_dir: Path,
    result_path: Path,
) -> dict[str, Any]:
    """Generate SQS structures through the public icet API."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (
        isinstance(max_candidates, bool)
        or not isinstance(max_candidates, int)
        or max_candidates < 1
    ):
        raise ValueError("max_candidates must be a positive integer")
    if not prototype_path.is_file() or not composition_path.is_file():
        raise ValueError("prototype and composition manifest must be regular files")
    if output_dir.exists() or result_path.exists():
        raise ValueError("output directory and result manifest must not already exist")
    attempt_dir = Path.cwd().resolve()
    output_dir = output_dir.resolve(strict=False)
    result_path = result_path.resolve(strict=False)
    try:
        output_dir.relative_to(attempt_dir)
        result_path.relative_to(attempt_dir)
    except ValueError as exc:
        raise ValueError(
            "output directory and result manifest must be inside the working attempt"
        ) from exc

    try:
        from ase.io import read, write
        from icet import ClusterSpace
        from icet.tools.structure_generation import generate_sqs_from_supercells
    except ImportError as exc:
        raise ValueError("SQS execution requires the optional ase and icet packages") from exc

    prototype = read(str(prototype_path))
    prototype_symbols = list(prototype.get_chemical_symbols())
    spec = validate_manifest(_load_object(composition_path), prototype_symbols=prototype_symbols)
    if len(spec["candidates"]) > max_candidates:
        raise ValueError(
            f"composition manifest contains {len(spec['candidates'])} candidates, exceeding "
            f"max_candidates={max_candidates}"
        )
    prototype_species = spec["alloy_sublattice"]["prototype_species"]
    allowed_species = spec["alloy_sublattice"]["allowed_species"]
    chemical_symbols = [
        allowed_species if symbol == prototype_species else [symbol] for symbol in prototype_symbols
    ]
    cluster_space = ClusterSpace(
        prototype,
        spec["cluster_cutoffs_angstrom"],
        chemical_symbols,
    )
    supercell = prototype.repeat(tuple(spec["supercell_repeat"]))
    suffix, media_type = FORMATS[spec["output_format"]]
    output_dir.mkdir(parents=True, exist_ok=False)
    structures: list[dict[str, Any]] = []
    for index, candidate in enumerate(spec["candidates"]):
        candidate_seed = seed + index
        total = candidate["alloy_site_count"]
        target_concentrations = {
            species: candidate["counts"][species] / total for species in allowed_species
        }
        structure = generate_sqs_from_supercells(
            cluster_space=cluster_space,
            supercells=[supercell],
            target_concentrations=target_concentrations,
            n_steps=spec["n_steps"],
            random_seed=candidate_seed,
        )
        output_path = output_dir / f"{candidate['id']}{suffix}"
        if output_path.exists():
            raise ValueError(f"refusing to replace existing structure: {output_path}")
        if spec["output_format"] == "vasp":
            write(str(output_path), structure, format="vasp", direct=True, vasp5=True, sort=True)
        else:
            write(str(output_path), structure, format="cif")
        relative = output_path.absolute().relative_to(attempt_dir)
        structures.append(
            {
                "id": candidate["id"],
                "path": relative.as_posix(),
                "fingerprint": _sha256(output_path),
                "composition": dict(candidate["counts"]),
                "media_type": media_type,
                "random_seed": candidate_seed,
                "cluster_vector": [
                    float(item) for item in cluster_space.get_cluster_vector(structure)
                ],
            }
        )

    result = {
        "schema_version": 1,
        "plugin_id": PLUGIN_ID,
        "status": "OK",
        "seed": seed,
        "prototype_fingerprint": _sha256(prototype_path),
        "composition_manifest_fingerprint": _sha256(composition_path),
        "structures": structures,
        "method": {
            "library": "icet",
            "library_version": importlib.metadata.version("icet"),
            "api": "generate_sqs_from_supercells",
            "ase_version": importlib.metadata.version("ase"),
            "cluster_cutoffs_angstrom": spec["cluster_cutoffs_angstrom"],
            "supercell_repeat": spec["supercell_repeat"],
            "n_steps": spec["n_steps"],
            "random_seed_policy": "base-seed-plus-candidate-index",
        },
    }
    _write_new_json(result_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototype", type=Path, required=True)
    parser.add_argument("--composition-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-candidates", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        generate(
            prototype_path=args.prototype,
            composition_path=args.composition_manifest,
            seed=args.seed,
            max_candidates=args.max_candidates,
            output_dir=args.output_dir,
            result_path=args.result_manifest,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
