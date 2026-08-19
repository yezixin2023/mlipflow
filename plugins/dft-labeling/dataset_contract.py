"""Canonical DFT records, one shared split, and framework record views."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

ENERGY_CONVENTION = "total-energy-per-configuration"
COORDINATE_CONVENTION = "fractional-row-vectors"
STRESS_CONVENTION = {
    "source": "VASP vasprun.xml varray[name=stress]",
    "unit": "kbar-vasp-3x3",
    "sign": "VASP-native",
    "order": "3x3-row-major",
}
VASP_KBAR_PER_EV_PER_ANGSTROM3 = 1602.176621
FRAMEWORKS = ("deepmd", "m3gnet", "chgnet", "mace")
SPLIT_NAMES = ("train", "validation", "test")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


class DatasetContractError(ValueError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def is_fingerprint(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256.fullmatch(value))


def safe_relative(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _matrix(value: Any, rows: int, columns: int) -> bool:
    return isinstance(value, list) and len(value) == rows and all(
        isinstance(row, list) and len(row) == columns and all(_finite(x) for x in row)
        for row in value
    )


def _attempt(value: Mapping[str, Any]) -> dict[str, Any]:
    project_id, node_id, attempt = value.get("project_id"), value.get("node_id"), value.get("attempt")
    if (
        not isinstance(project_id, str) or not SAFE_ID.fullmatch(project_id)
        or not isinstance(node_id, str) or not SAFE_ID.fullmatch(node_id)
        or isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1
    ):
        raise DatasetContractError("source DFT attempt identity is invalid")
    return {"project_id": project_id, "node_id": node_id, "attempt": attempt}


def _sources(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    items = manifest.get("structures")
    if not isinstance(items, list) or not items:
        raise DatasetContractError("structures manifest must contain structures")
    result = {}
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise DatasetContractError(f"structures[{index}] must be an object")
        structure_id = item.get("id", item.get("structure_id"))
        path = item.get("path", item.get("output_file"))
        fingerprint = item.get("fingerprint", item.get("frame_sha256", item.get("sha256")))
        if (
            not isinstance(structure_id, str) or not SAFE_ID.fullmatch(structure_id)
            or structure_id in result or not safe_relative(path) or not is_fingerprint(fingerprint)
        ):
            raise DatasetContractError(f"structures[{index}] has invalid identity")
        source = {"structure_id": structure_id, "path": str(path), "fingerprint": str(fingerprint)}
        group = item.get("source_group_id")
        if group is not None:
            if not isinstance(group, str) or not SAFE_ID.fullmatch(group):
                raise DatasetContractError(f"structures[{index}].source_group_id is invalid")
            source["source_group_id"] = group
        method = item.get("source_sampling_method")
        methods = item.get("source_sampling_methods")
        source_records = item.get("source_records")
        if method is not None:
            if not isinstance(method, str) or not SAFE_ID.fullmatch(method):
                raise DatasetContractError(
                    f"structures[{index}].source_sampling_method is invalid"
                )
            source["source_sampling_method"] = method
        if methods is not None:
            if (
                not isinstance(methods, list)
                or not methods
                or len(methods) != len(set(methods))
                or any(not isinstance(value, str) or not SAFE_ID.fullmatch(value) for value in methods)
                or (method is not None and method not in methods)
            ):
                raise DatasetContractError(
                    f"structures[{index}].source_sampling_methods is invalid"
                )
            source["source_sampling_methods"] = list(methods)
        if source_records is not None:
            normalized_records = []
            if not isinstance(source_records, list) or not source_records:
                raise DatasetContractError(f"structures[{index}].source_records is invalid")
            for record in source_records:
                if (
                    not isinstance(record, Mapping)
                    or not isinstance(record.get("sampling_method"), str)
                    or not SAFE_ID.fullmatch(record["sampling_method"])
                    or not isinstance(record.get("source_group_id"), str)
                    or not SAFE_ID.fullmatch(record["source_group_id"])
                    or not isinstance(record.get("source_id"), str)
                    or not SAFE_ID.fullmatch(record["source_id"])
                    or isinstance(record.get("source_order"), bool)
                    or not isinstance(record.get("source_order"), int)
                    or record["source_order"] < 1
                    or not safe_relative(record.get("source_path"))
                    or not is_fingerprint(record.get("source_sha256"))
                ):
                    raise DatasetContractError(f"structures[{index}].source_records is invalid")
                normalized_records.append(dict(record))
            source["source_records"] = normalized_records
        result[structure_id] = source
    return result


def _record_id(attempt: Mapping[str, Any], source: Mapping[str, Any], calculation_id: str, ionic_step: int) -> str:
    identity = {
        "source_dft_attempt": dict(attempt),
        "source_structure_id": source["structure_id"],
        "source_structure_fingerprint": source["fingerprint"],
        "calculation_id": calculation_id,
        "ionic_step": ionic_step,
    }
    return "dft-record-" + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def build_canonical_dataset(
    *, label_records: Sequence[Mapping[str, Any]], units: Mapping[str, Any],
    calculation_type: str, source_attempt_identity: Mapping[str, Any],
    structures_manifest: Mapping[str, Any], raw_outputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Build canonical records without changing atom order or numeric labels."""
    expected_units = {"energy": "eV", "length": "angstrom", "force": "eV/angstrom", "stress": "kbar-vasp-3x3"}
    if not label_records or dict(units) != expected_units:
        raise DatasetContractError("canonical records or units are invalid")
    if calculation_type not in {"static", "relax", "aimd"}:
        raise DatasetContractError("unsupported DFT calculation type")
    attempt, sources = _attempt(source_attempt_identity), _sources(structures_manifest)
    records, stress_presence, seen = [], set(), set()
    for index, raw in enumerate(label_records):
        structure_id, calculation_id, ionic_step = raw.get("structure_id"), raw.get("calculation_id"), raw.get("ionic_step")
        species = raw.get("species")
        if structure_id not in sources:
            raise DatasetContractError(f"label record {index} references an unknown structure")
        if not isinstance(calculation_id, str) or not SAFE_ID.fullmatch(calculation_id):
            raise DatasetContractError(f"label record {index} has an invalid calculation id")
        if isinstance(ionic_step, bool) or not isinstance(ionic_step, int) or ionic_step < 1:
            raise DatasetContractError(f"label record {index} has an invalid ionic step")
        if not isinstance(species, list) or not species or any(not isinstance(x, str) for x in species):
            raise DatasetContractError(f"label record {index} has invalid species")
        n = len(species)
        lattice, coordinates = raw.get("lattice_angstrom"), raw.get("fractional_coordinates")
        energy, forces, stress = raw.get("energy_ev"), raw.get("forces_ev_per_angstrom"), raw.get("stress_kbar_vasp_3x3")
        if not _matrix(lattice, 3, 3) or not _matrix(coordinates, n, 3):
            raise DatasetContractError(f"label record {index} has invalid geometry")
        if not _finite(energy) or not _matrix(forces, n, 3):
            raise DatasetContractError(f"label record {index} has invalid energy or forces")
        if stress is not None and not _matrix(stress, 3, 3):
            raise DatasetContractError(f"label record {index} has invalid stress")
        stress_presence.add(stress is not None)
        source = sources[str(structure_id)]
        record_attempt = (
            _attempt(raw["source_dft_attempt"])
            if isinstance(raw.get("source_dft_attempt"), Mapping)
            else attempt
        )
        record_id = _record_id(record_attempt, source, calculation_id, ionic_step)
        if record_id in seen:
            raise DatasetContractError("canonical record identity collision")
        seen.add(record_id)
        record = {
            "record_id": record_id, "source_structure_id": structure_id,
            "calculation_id": calculation_id, "ionic_step": ionic_step,
            "calculation_type": calculation_type, "species": list(species),
            "atom_order": list(range(n)), "lattice": [list(row) for row in lattice],
            "coordinates": {"kind": "fractional", "values": [list(row) for row in coordinates]},
            "total_energy": energy, "atomic_forces": [list(row) for row in forces],
            "stress": None if stress is None else [list(row) for row in stress],
            "source_structure": dict(source), "source_dft_attempt": dict(record_attempt),
            "source_dft_outputs": {
                name: dict(value) for name, value in sorted(raw_outputs.items())
                if isinstance(name, str) and name.startswith(calculation_id + "/") and isinstance(value, Mapping)
            },
        }
        group = raw.get("source_group_id", source.get("source_group_id"))
        if group is None and calculation_type == "aimd":
            group = calculation_id
        if group is not None:
            if not isinstance(group, str) or not SAFE_ID.fullmatch(group):
                raise DatasetContractError(f"label record {index} has invalid source_group_id")
            record["source_group_id"] = group
        records.append(record)
    if len(stress_presence) != 1:
        raise DatasetContractError("stress must be present for all records or absent for all")
    dataset_id = "dft-" + hashlib.sha256(canonical_json_bytes([r["record_id"] for r in records])).hexdigest()[:32]
    targets = ["total_energy", "atomic_forces"] + (["stress"] if True in stress_presence else [])
    return {
        "schema_version": 1, "dataset_id": dataset_id, "source_attempt_identity": dict(attempt),
        "record_count": len(records), "targets": targets, "units": expected_units,
        "energy_convention": ENERGY_CONVENTION, "coordinate_convention": COORDINATE_CONVENTION,
        "stress_convention": dict(STRESS_CONVENTION) if "stress" in targets else None,
        "records": records,
    }


def validate_canonical_dataset(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return ["canonical dataset must be an object"]
    errors, records = [], value.get("records")
    if value.get("schema_version") != 1:
        errors.append("canonical schema_version must be 1")
    try:
        _attempt(value.get("source_attempt_identity", {}))
    except DatasetContractError:
        errors.append("canonical source attempt identity is invalid")
    if not isinstance(records, list) or not records:
        return errors + ["canonical records must be a non-empty list"]
    if value.get("record_count") != len(records):
        errors.append("canonical record_count mismatch")
    targets = ["total_energy", "atomic_forces"] + (["stress"] if records[0].get("stress") is not None else [])
    if value.get("targets") != targets:
        errors.append("canonical targets mismatch")
    if value.get("energy_convention") != ENERGY_CONVENTION or value.get("coordinate_convention") != COORDINATE_CONVENTION:
        errors.append("canonical energy/coordinate convention mismatch")
    if value.get("stress_convention") != (STRESS_CONVENTION if "stress" in targets else None):
        errors.append("canonical stress convention mismatch")
    seen = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            errors.append(f"canonical record {index} must be an object")
            continue
        species, source = record.get("species"), record.get("source_structure")
        n = len(species) if isinstance(species, list) else 0
        record_id = record.get("record_id")
        record_attempt = record.get("source_dft_attempt")
        try:
            normalized_record_attempt = (
                _attempt(record_attempt) if isinstance(record_attempt, Mapping) else {}
            )
            if not normalized_record_attempt:
                raise DatasetContractError("missing record attempt")
        except DatasetContractError:
            normalized_record_attempt = {}
            errors.append(f"canonical record {index} source DFT attempt is invalid")
        if not isinstance(source, Mapping) or not is_fingerprint(source.get("fingerprint")):
            errors.append(f"canonical record {index} source structure is invalid")
        else:
            try:
                _sources({"structures": [source]})
            except DatasetContractError:
                errors.append(f"canonical record {index} source provenance is invalid")
            if normalized_record_attempt and record_id != _record_id(
                normalized_record_attempt,
                source,
                record.get("calculation_id"),
                record.get("ionic_step"),
            ):
                errors.append(f"canonical record {index} stable record_id mismatch")
        if not isinstance(record_id, str) or not SAFE_ID.fullmatch(record_id) or record_id in seen:
            errors.append(f"canonical record {index} identity is invalid")
        else:
            seen.add(record_id)
        group = record.get("source_group_id")
        if group is not None and (not isinstance(group, str) or not SAFE_ID.fullmatch(group)):
            errors.append(f"canonical record {index} source_group_id is invalid")
        if not isinstance(species, list) or not species or any(not re.fullmatch(r"[A-Z][a-z]?", x) for x in species):
            errors.append(f"canonical record {index} species are invalid")
        if record.get("atom_order") != list(range(n)):
            errors.append(f"canonical record {index} atom order changed")
        coord = record.get("coordinates")
        if not _matrix(record.get("lattice"), 3, 3) or not isinstance(coord, Mapping) or coord.get("kind") != "fractional" or not _matrix(coord.get("values"), n, 3):
            errors.append(f"canonical record {index} geometry is invalid")
        if not _finite(record.get("total_energy")) or not _matrix(record.get("atomic_forces"), n, 3):
            errors.append(f"canonical record {index} labels are invalid")
        stress = record.get("stress")
        if ("stress" in targets and not _matrix(stress, 3, 3)) or ("stress" not in targets and stress is not None):
            errors.append(f"canonical record {index} stress is invalid")
    expected_id = "dft-" + hashlib.sha256(canonical_json_bytes([r.get("record_id") for r in records])).hexdigest()[:32]
    if value.get("dataset_id") != expected_id:
        errors.append("canonical dataset_id mismatch")
    return errors


def _split_counts(total: int, fractions: Mapping[str, Any]) -> dict[str, int]:
    values = [fractions.get(name) for name in SPLIT_NAMES]
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or x <= 0 for x in values):
        raise DatasetContractError("split fractions must be positive numbers")
    if not math.isclose(sum(float(x) for x in values), 1.0, rel_tol=0, abs_tol=1e-12):
        raise DatasetContractError("split fractions must sum to one")
    if total < 3:
        raise DatasetContractError("train/validation/test split requires at least three records")
    raw, counts = [total * float(x) for x in values], [max(1, int(math.floor(total * float(x)))) for x in values]
    while sum(counts) > total:
        candidates = [i for i, count in enumerate(counts) if count > 1]
        counts[max(candidates, key=lambda i: counts[i] - raw[i])] -= 1
    while sum(counts) < total:
        counts[max(range(3), key=lambda i: raw[i] - counts[i])] += 1
    return dict(zip(SPLIT_NAMES, counts))


def _rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def build_split_manifest(
    canonical: Mapping[str, Any], *, strategy: str = "deterministic", seed: int = 0,
    fractions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    errors = validate_canonical_dataset(canonical)
    if errors:
        raise DatasetContractError("; ".join(errors))
    if strategy not in {"deterministic", "group-aware"}:
        raise DatasetContractError("split strategy must be deterministic or group-aware")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DatasetContractError("split seed must be a non-negative integer")
    targets = _split_counts(len(canonical["records"]), fractions or {"train": .8, "validation": .1, "test": .1})
    assigned = {name: [] for name in SPLIT_NAMES}
    if strategy == "deterministic":
        ordered = sorted((r["record_id"] for r in canonical["records"]), key=lambda x: (_rank(seed, x), x))
        start = 0
        for name in SPLIT_NAMES:
            assigned[name] = ordered[start:start + targets[name]]
            start += targets[name]
    else:
        groups: dict[str, list[str]] = {}
        for record in canonical["records"]:
            group = record.get("source_group_id")
            if not isinstance(group, str):
                raise DatasetContractError("group-aware split requires source_group_id on every record")
            groups.setdefault(group, []).append(record["record_id"])
        if len(groups) < 3:
            raise DatasetContractError("group-aware split requires at least three source groups")
        for group in sorted(groups, key=lambda x: (-len(groups[x]), _rank(seed, x), x)):
            name = max(SPLIT_NAMES, key=lambda x: (targets[x] - len(assigned[x]), -SPLIT_NAMES.index(x)))
            assigned[name].extend(sorted(groups[group], key=lambda x: (_rank(seed, x), x)))
        if any(not assigned[name] for name in SPLIT_NAMES):
            raise DatasetContractError("group-aware split leaves an empty partition")
    identity = {"dataset_id": canonical["dataset_id"], "strategy": strategy, "seed": seed, **{f"{name}_record_ids": assigned[name] for name in SPLIT_NAMES}}
    return {
        "dataset_id": canonical["dataset_id"],
        "split_id": "split-" + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()[:32],
        "strategy": strategy, "seed": seed,
        "train_record_ids": assigned["train"], "validation_record_ids": assigned["validation"],
        "test_record_ids": assigned["test"],
        "counts": {"train": len(assigned["train"]), "validation": len(assigned["validation"]), "test": len(assigned["test"]), "total": len(canonical["records"])},
    }


def validate_split_manifest(canonical: Mapping[str, Any], split: Any) -> list[str]:
    if not isinstance(split, Mapping):
        return ["split manifest must be an object"]
    allowed = {"dataset_id", "split_id", "strategy", "seed", "train_record_ids", "validation_record_ids", "test_record_ids", "counts"}
    errors = []
    if set(split) != allowed:
        errors.append("split manifest fields are not minimal contract fields")
    if split.get("dataset_id") != canonical.get("dataset_id"):
        errors.append("split dataset_id mismatch")
    all_ids = []
    for name in SPLIT_NAMES:
        values = split.get(f"{name}_record_ids")
        if not isinstance(values, list) or not values:
            errors.append(f"split {name} record IDs are invalid")
        else:
            all_ids.extend(values)
    canonical_ids = [r["record_id"] for r in canonical.get("records", [])]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(canonical_ids):
        errors.append("split record IDs are not a disjoint cover of canonical records")
    expected_counts = {name: len(split.get(f"{name}_record_ids", [])) if isinstance(split.get(f"{name}_record_ids"), list) else -1 for name in SPLIT_NAMES}
    expected_counts["total"] = len(canonical_ids)
    if split.get("counts") != expected_counts:
        errors.append("split counts mismatch")
    if split.get("strategy") == "group-aware":
        membership = {rid: name for name in SPLIT_NAMES for rid in split.get(f"{name}_record_ids", [])}
        groups: dict[str, set[str]] = {}
        for record in canonical.get("records", []):
            group = record.get("source_group_id")
            if not isinstance(group, str):
                errors.append("group-aware split record lacks source_group_id")
                break
            groups.setdefault(group, set()).add(membership.get(record["record_id"], "missing"))
        if any(len(names) != 1 for names in groups.values()):
            errors.append("source group leaks across split partitions")
    return errors


def records_for_split(canonical: Mapping[str, Any], split: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    if name not in SPLIT_NAMES:
        raise DatasetContractError("unknown split name")
    errors = validate_split_manifest(canonical, split)
    if errors:
        raise DatasetContractError("; ".join(errors))
    by_id = {record["record_id"]: record for record in canonical["records"]}
    return [dict(by_id[record_id]) for record_id in split[f"{name}_record_ids"]]


def cartesian_coordinates(record: Mapping[str, Any]) -> list[list[float]]:
    lattice = record["lattice"]
    return [[sum(float(frac[k]) * float(lattice[k][j]) for k in range(3)) for j in range(3)] for frac in record["coordinates"]["values"]]


def benchmark_dataset(
    canonical: Mapping[str, Any], split: Mapping[str, Any]
) -> dict[str, Any]:
    samples = []
    include_stress = "stress" in canonical["targets"]
    for record in records_for_split(canonical, split, "test"):
        references: dict[str, Any] = {
            "energy": record["total_energy"],
            "force": record["atomic_forces"],
        }
        if include_stress:
            stress = record["stress"]
            references["stress"] = [
                -float(stress[0][0]) / VASP_KBAR_PER_EV_PER_ANGSTROM3,
                -float(stress[1][1]) / VASP_KBAR_PER_EV_PER_ANGSTROM3,
                -float(stress[2][2]) / VASP_KBAR_PER_EV_PER_ANGSTROM3,
                -float(stress[1][2]) / VASP_KBAR_PER_EV_PER_ANGSTROM3,
                -float(stress[0][2]) / VASP_KBAR_PER_EV_PER_ANGSTROM3,
                -float(stress[0][1]) / VASP_KBAR_PER_EV_PER_ANGSTROM3,
            ]
        samples.append(
            {
                "id": record["record_id"],
                "species": record["species"],
                "positions": cartesian_coordinates(record),
                "cell": record["lattice"],
                "pbc": True,
                "references": references,
            }
        )
    units = {"energy": "eV", "force": "eV/angstrom"}
    if include_stress:
        units["stress"] = "eV/angstrom^3"
    return {
        "schema_version": 1,
        "dataset_id": canonical["dataset_id"],
        "split_id": split["split_id"],
        "split": "test",
        "units": units,
        "energy_convention": "total",
        "stress_convention": (
            "ase-voigt-xx-yy-zz-yz-xz-xy" if include_stress else None
        ),
        "samples": samples,
    }


def pymatgen_structure(record: Mapping[str, Any]) -> dict[str, Any]:
    cartesian = cartesian_coordinates(record)
    return {
        "@module": "pymatgen.core.structure", "@class": "Structure", "charge": 0,
        "lattice": {"matrix": [list(row) for row in record["lattice"]], "pbc": [True, True, True]}, "properties": {},
        "sites": [{"species": [{"element": species, "occu": 1}], "abc": list(frac), "xyz": list(xyz), "properties": {}, "label": species}
                  for species, frac, xyz in zip(record["species"], record["coordinates"]["values"], cartesian)],
    }


def framework_json_dataset(dataset_id: str, split_id: str, framework: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if framework not in {"m3gnet", "chgnet"}:
        raise DatasetContractError("JSON view is only defined for M3GNet and CHGNet")
    converted = []
    for record in records:
        item = {"record_id": record["record_id"], "source_structure_id": record["source_structure_id"], "structure": pymatgen_structure(record), "energy": record["total_energy"], "forces": record["atomic_forces"]}
        if record["stress"] is not None:
            item["stresses" if framework == "m3gnet" else "stress"] = record["stress"]
        converted.append(item)
    return {"dataset_id": dataset_id, "split_id": split_id, "records": converted}


def training_tree_fingerprint(path: Path) -> str:
    """Match the directory identity required by mlip-training references."""
    if path.is_symlink() or not path.is_dir():
        raise DatasetContractError(f"not an ordinary dataset directory: {path}")
    digest, found = hashlib.sha256(), False
    for item in sorted(path.rglob("*"), key=lambda p: p.relative_to(path).as_posix()):
        if item.is_symlink():
            raise DatasetContractError(f"dataset tree contains a symlink: {item}")
        if item.is_file():
            found = True
            content, relative = item.read_bytes(), item.relative_to(path).as_posix()
            digest.update(f"{relative}\0{len(content)}\0{sha256_bytes(content)}\n".encode())
    if not found:
        raise DatasetContractError("dataset directory is empty")
    return "sha256:" + digest.hexdigest()
