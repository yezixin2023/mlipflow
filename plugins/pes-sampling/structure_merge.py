"""Merge verified DIRECT and LASP selections into one DFT-ready structure set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_STRUCTURE_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_STRUCTURES = 10000
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


class MergeError(ValueError):
    pass


@dataclass
class Candidate:
    method: str
    source_group_id: str
    source_id: str
    source_order: int
    source_path: str
    source_sha256: str
    structure: Any


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _ordinary_file(path: Path, label: str, maximum: int) -> Path:
    path = path.expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise MergeError(f"{label} must be an ordinary file")
    size = path.stat().st_size
    if not 0 < size <= maximum:
        raise MergeError(f"{label} size must be between 1 and {maximum} bytes")
    return path.resolve()


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise MergeError(f"{label} must match {SAFE_ID.pattern}")
    return value


def _safe_relative(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise MergeError(f"{label} must be a safe relative path")
    path = Path(value)
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise MergeError(f"{label} must be a safe relative path")
    return path


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = _ordinary_file(path, label, MAX_JSON_BYTES)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MergeError(f"{label} must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise MergeError(f"{label} must be a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_pymatgen() -> tuple[Any, Any, Any, str]:
    try:
        import importlib.metadata
        from pymatgen.analysis.structure_matcher import StructureMatcher
        from pymatgen.core import Structure
        from pymatgen.io.ase import AseAtomsAdaptor
    except (ImportError, ModuleNotFoundError) as exc:
        raise MergeError("pymatgen with ASE support is required") from exc
    try:
        version = importlib.metadata.version("pymatgen")
    except importlib.metadata.PackageNotFoundError as exc:
        raise MergeError("the pymatgen distribution version is required") from exc
    return Structure, StructureMatcher, AseAtomsAdaptor, version


def _read_arc(payload: bytes, adaptor: Any) -> Any:
    try:
        from ase.io.dmol import read_dmol_arc
    except (ImportError, ModuleNotFoundError) as exc:
        raise MergeError("ASE is required to read LASP ARC structures") from exc
    try:
        images = read_dmol_arc(
            io.StringIO(payload.decode("utf-8")), index=slice(None)
        )
    except Exception as exc:
        raise MergeError("ASE could not parse a selected LASP ARC structure") from exc
    if not isinstance(images, list) or len(images) != 1:
        raise MergeError("each selected LASP ARC member must contain one structure")
    return adaptor.get_structure(images[0])


def _direct_candidates(
    manifest_path: Path, source_group_id: str, Structure: Any
) -> list[Candidate]:
    manifest_path = _ordinary_file(manifest_path, "direct_manifest", MAX_JSON_BYTES)
    try:
        reader = csv.DictReader(io.StringIO(manifest_path.read_text(encoding="utf-8")))
        rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise MergeError("direct_manifest must be readable CSV") from exc
    required = {"selected_order", "output_file"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames) or not rows:
        raise MergeError("direct_manifest has no selected structures or required columns")
    candidates: list[Candidate] = []
    seen_orders: set[int] = set()
    for index, row in enumerate(rows, 1):
        try:
            order = int(row["selected_order"])
        except (TypeError, ValueError) as exc:
            raise MergeError(f"direct_manifest row {index} has invalid selected_order") from exc
        if order < 1 or order in seen_orders:
            raise MergeError("DIRECT selected_order values must be unique and positive")
        seen_orders.add(order)
        relative = _safe_relative(row.get("output_file"), f"DIRECT row {index} output_file")
        path = (manifest_path.parent / relative).absolute()
        if not _within(path, manifest_path.parent):
            raise MergeError("DIRECT selected structure escapes the manifest directory")
        path = _ordinary_file(path, f"DIRECT selected structure {order}", MAX_STRUCTURE_BYTES)
        try:
            structure = Structure.from_file(str(path))
        except Exception as exc:
            raise MergeError(f"pymatgen could not parse DIRECT selected structure {order}") from exc
        candidates.append(
            Candidate(
                method="DIRECT",
                source_group_id=source_group_id,
                source_id=f"direct-{order:06d}",
                source_order=order,
                source_path=relative.as_posix(),
                source_sha256=_sha256(path),
                structure=structure,
            )
        )
    return sorted(candidates, key=lambda item: item.source_order)


def _safe_archive_members(archive_path: Path) -> dict[str, bytes]:
    archive_path = _ordinary_file(archive_path, "lasp_selected_archive", MAX_ARCHIVE_BYTES)
    result: dict[str, bytes] = {}
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                relative = _safe_relative(member.name, "LASP archive member")
                if not member.isfile() or member.size < 1 or member.size > MAX_STRUCTURE_BYTES:
                    raise MergeError("LASP archive members must be bounded ordinary files")
                if relative.as_posix() in result:
                    raise MergeError("LASP archive contains duplicate member names")
                stream = archive.extractfile(member)
                if stream is None:
                    raise MergeError("LASP archive member could not be read")
                payload = stream.read(MAX_STRUCTURE_BYTES + 1)
                if len(payload) != member.size:
                    raise MergeError("LASP archive member size changed while reading")
                result[relative.as_posix()] = payload
    except (OSError, tarfile.TarError) as exc:
        raise MergeError("lasp_selected_archive must be a valid gzip tar archive") from exc
    if not result:
        raise MergeError("lasp_selected_archive contains no files")
    return result


def _lasp_candidates(
    manifest_path: Path, archive_path: Path, source_group_id: str, adaptor: Any
) -> list[Candidate]:
    manifest = _read_json(manifest_path, "lasp_selected_manifest")
    records = manifest.get("structures")
    if manifest.get("schema_version") != 1 or not isinstance(records, list) or not records:
        raise MergeError("lasp_selected_manifest must contain selected structures")
    members = _safe_archive_members(archive_path)
    candidates: list[Candidate] = []
    seen_orders: set[int] = set()
    for index, record in enumerate(records, 1):
        if not isinstance(record, Mapping):
            raise MergeError(f"LASP selected record {index} must be an object")
        source_id = _safe_id(record.get("structure_id"), f"LASP selected record {index} id")
        order = record.get("selected_order")
        if isinstance(order, bool) or not isinstance(order, int) or order < 1 or order in seen_orders:
            raise MergeError("LASP selected_order values must be unique positive integers")
        seen_orders.add(order)
        relative = _safe_relative(record.get("output_file"), f"LASP record {index} output_file")
        payload = members.get(relative.as_posix())
        if payload is None:
            raise MergeError(f"LASP selected archive lacks {relative.as_posix()}")
        declared = record.get("frame_sha256")
        if not isinstance(declared, str) or not SHA256.fullmatch(declared):
            raise MergeError(f"LASP selected record {index} lacks a full frame_sha256")
        actual = _sha256_bytes(payload)
        if actual != declared:
            raise MergeError(f"LASP selected record {index} differs from its archive member")
        candidates.append(
            Candidate(
                method="LASP_SSW",
                source_group_id=source_group_id,
                source_id=source_id,
                source_order=order,
                source_path=relative.as_posix(),
                source_sha256=actual,
                structure=_read_arc(payload, adaptor),
            )
        )
    if set(members) != {candidate.source_path for candidate in candidates}:
        raise MergeError("LASP selected archive members do not exactly match the selected manifest")
    return sorted(candidates, key=lambda item: item.source_order)


def _source_record(candidate: Candidate) -> dict[str, Any]:
    return {
        "sampling_method": candidate.method,
        "source_group_id": candidate.source_group_id,
        "source_id": candidate.source_id,
        "source_order": candidate.source_order,
        "source_path": candidate.source_path,
        "source_sha256": candidate.source_sha256,
    }


def _structure_payload(structure: Any) -> bytes:
    try:
        from pymatgen.io.vasp.inputs import Poscar
    except (ImportError, ModuleNotFoundError) as exc:
        raise MergeError("pymatgen VASP support is required") from exc
    return Poscar(structure, sort_structure=True).get_str(significant_figures=16).encode("utf-8")


def _finite_structure(structure: Any, minimum_distance: float) -> tuple[bool, str | None]:
    if not getattr(structure, "is_ordered", False):
        return False, "disordered-structure"
    try:
        lattice = structure.lattice.matrix
        values = [float(value) for row in lattice for value in row]
        values.extend(float(value) for row in structure.frac_coords for value in row)
        volume = float(structure.volume)
    except Exception:
        return False, "invalid-geometry"
    if not values or any(not math.isfinite(value) for value in values) or not math.isfinite(volume) or volume <= 0:
        return False, "non-finite-or-nonpositive-cell"
    if not structure.is_valid(tol=minimum_distance):
        return False, "interatomic-distance-below-threshold"
    return True, None


def merge(args: argparse.Namespace) -> dict[str, Any]:
    if not 0 < args.matcher_ltol <= 1:
        raise MergeError("matcher_ltol must be in (0, 1]")
    if not 0 < args.matcher_stol <= 1:
        raise MergeError("matcher_stol must be in (0, 1]")
    if not 0 < args.matcher_angle_tol_deg <= 30:
        raise MergeError("matcher_angle_tol_deg must be in (0, 30]")
    if not 0 < args.minimum_distance_angstrom <= 5:
        raise MergeError("minimum_distance_angstrom must be in (0, 5]")
    if not 1 <= args.max_structures <= MAX_STRUCTURES:
        raise MergeError(f"max_structures must be in 1..{MAX_STRUCTURES}")
    direct_group = _safe_id(args.direct_source_group_id, "direct_source_group_id")
    lasp_group = _safe_id(args.lasp_source_group_id, "lasp_source_group_id")
    output_dir = Path(args.output_dir).expanduser().absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise MergeError("merge-structures requires a fresh output directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    marker = output_dir / "INCOMPLETE.json"
    _write_json(marker, {"schema_version": 1, "operation": "merge-structures", "status": "INCOMPLETE"})

    Structure, StructureMatcher, adaptor, pymatgen_version = _load_pymatgen()
    direct_manifest = _ordinary_file(Path(args.direct_manifest), "direct_manifest", MAX_JSON_BYTES)
    lasp_manifest = _ordinary_file(
        Path(args.lasp_selected_manifest), "lasp_selected_manifest", MAX_JSON_BYTES
    )
    lasp_archive = _ordinary_file(
        Path(args.lasp_selected_archive), "lasp_selected_archive", MAX_ARCHIVE_BYTES
    )
    candidates = _direct_candidates(direct_manifest, direct_group, Structure)
    candidates.extend(_lasp_candidates(lasp_manifest, lasp_archive, lasp_group, adaptor))
    if len(candidates) > args.max_structures:
        raise MergeError("input structure count exceeds max_structures")

    exact_matcher = StructureMatcher(
        ltol=1e-8,
        stol=1e-8,
        angle_tol=1e-6,
        primitive_cell=False,
        scale=False,
        attempt_supercell=False,
        allow_subset=False,
    )
    near_matcher = StructureMatcher(
        ltol=args.matcher_ltol,
        stol=args.matcher_stol,
        angle_tol=args.matcher_angle_tol_deg,
        primitive_cell=False,
        scale=False,
        attempt_supercell=False,
        allow_subset=False,
    )
    structures_dir = output_dir / "structures"
    structures_dir.mkdir()
    kept: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        valid, reason = _finite_structure(candidate.structure, args.minimum_distance_angstrom)
        if not valid:
            rejected.append({**_source_record(candidate), "reason": reason})
            continue
        duplicate: dict[str, Any] | None = None
        for record in kept:
            if exact_matcher.fit(candidate.structure, record["_structure"]):
                duplicate = record
                duplicate_kind = "exact"
                break
            if near_matcher.fit(candidate.structure, record["_structure"]):
                duplicate = record
                duplicate_kind = "near"
                break
        if duplicate is not None:
            source = _source_record(candidate)
            duplicate["source_records"].append(source)
            duplicate["source_sampling_methods"] = sorted(
                {item["sampling_method"] for item in duplicate["source_records"]}
            )
            duplicates.append(
                {
                    **source,
                    "duplicate_kind": duplicate_kind,
                    "kept_structure_id": duplicate["id"],
                }
            )
            continue
        payload = _structure_payload(candidate.structure)
        fingerprint = _sha256_bytes(payload)
        structure_id = "structure-" + fingerprint.removeprefix("sha256:")[:24]
        path = structures_dir / f"{structure_id}.vasp"
        if path.exists():
            raise MergeError("normalized structure identity collision")
        path.write_bytes(payload)
        source = _source_record(candidate)
        kept.append(
            {
                "id": structure_id,
                "path": path.relative_to(output_dir).as_posix(),
                "fingerprint": fingerprint,
                "source_group_id": candidate.source_group_id,
                "source_sampling_method": candidate.method,
                "source_sampling_methods": [candidate.method],
                "source_records": [source],
                "formula": candidate.structure.composition.reduced_formula,
                "atom_count": len(candidate.structure),
                "_structure": candidate.structure,
            }
        )
    if not kept:
        raise MergeError("no valid unique structures remain after quality checks and deduplication")

    public_records = [{key: value for key, value in record.items() if key != "_structure"} for record in kept]
    matching = {
        "implementation": "pymatgen.analysis.structure_matcher.StructureMatcher",
        "pymatgen_version": pymatgen_version,
        "exact": {"ltol": 1e-8, "stol": 1e-8, "angle_tol_deg": 1e-6, "scale": False},
        "near": {
            "ltol": args.matcher_ltol,
            "stol": args.matcher_stol,
            "angle_tol_deg": args.matcher_angle_tol_deg,
            "scale": False,
        },
        "minimum_distance_angstrom": args.minimum_distance_angstrom,
    }
    identity = {
        "input_sha256": {
            "direct_manifest": _sha256(direct_manifest),
            "lasp_selected_manifest": _sha256(lasp_manifest),
            "lasp_selected_archive": _sha256(lasp_archive),
        },
        "matching": matching,
        "structure_ids": [record["id"] for record in public_records],
        "duplicates": duplicates,
        "rejected": rejected,
    }
    structure_set_id = "structure-set-" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:32]
    per_source: dict[str, dict[str, int]] = {}
    for method in ("DIRECT", "LASP_SSW"):
        inputs = sum(candidate.method == method for candidate in candidates)
        exact = sum(item["sampling_method"] == method and item["duplicate_kind"] == "exact" for item in duplicates)
        near = sum(item["sampling_method"] == method and item["duplicate_kind"] == "near" for item in duplicates)
        bad = sum(item["sampling_method"] == method for item in rejected)
        per_source[method] = {"input": inputs, "exact_duplicates": exact, "near_duplicates": near, "rejected_bad": bad}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "operation": "merge-structures",
        "status": "OK",
        "structure_set_id": structure_set_id,
        "input_artifacts": identity["input_sha256"],
        "matching": matching,
        "counts": {
            "input": len(candidates),
            "unique": len(public_records),
            "exact_duplicates": sum(item["duplicate_kind"] == "exact" for item in duplicates),
            "near_duplicates": sum(item["duplicate_kind"] == "near" for item in duplicates),
            "rejected_bad": len(rejected),
        },
        "source_statistics": per_source,
        "structures": public_records,
        "duplicates": duplicates,
        "rejected": rejected,
    }
    _write_json(output_dir / "structures.json", manifest)
    marker.unlink()
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge DIRECT and LASP selected structures")
    parser.add_argument("--direct-manifest", required=True)
    parser.add_argument("--lasp-selected-manifest", required=True)
    parser.add_argument("--lasp-selected-archive", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--direct-source-group-id", required=True)
    parser.add_argument("--lasp-source-group-id", required=True)
    parser.add_argument("--matcher-ltol", type=float, required=True)
    parser.add_argument("--matcher-stol", type=float, required=True)
    parser.add_argument("--matcher-angle-tol-deg", type=float, required=True)
    parser.add_argument("--minimum-distance-angstrom", type=float, required=True)
    parser.add_argument("--max-structures", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        merge(build_parser().parse_args(argv))
    except Exception as exc:
        print(f"structure merge error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
