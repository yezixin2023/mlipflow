#!/usr/bin/env python3
"""Create four split-preserving training views on the compute site."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
import tempfile
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__:
    from .dataset_contract import (
        FRAMEWORKS,
        benchmark_dataset,
        build_split_manifest,
        canonical_json_bytes,
        cartesian_coordinates,
        framework_json_dataset,
        merge_canonical_datasets,
        records_for_split,
        validate_canonical_dataset,
    )
else:
    from dataset_contract import (
        FRAMEWORKS,
        benchmark_dataset,
        build_split_manifest,
        canonical_json_bytes,
        cartesian_coordinates,
        framework_json_dataset,
        merge_canonical_datasets,
        records_for_split,
        validate_canonical_dataset,
    )

VASP_KBAR_PER_EV_PER_ANGSTROM3 = 1602.176621
DEEPMD_SET_SIZE = 2000
PARTITIONS = (("train", "train"), ("validation", "valid"), ("test", "test"))


class ConversionError(RuntimeError):
    pass


class DependencyBlocked(ConversionError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConversionError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConversionError(f"{path} must contain a JSON object")
    return value


def _canonical_input(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("contract") != "mlipflow/canonical-dataset-merge":
        return value
    if value.get("schema_version") != 1 or not isinstance(value.get("sources"), list):
        raise ConversionError("canonical merge manifest is invalid")
    sources = []
    for index, item in enumerate(value["sources"]):
        if not isinstance(item, dict):
            raise ConversionError(f"canonical merge source {index} must be an object")
        relative = item.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ConversionError(f"canonical merge source {index} path is unsafe")
        source = (path.parent / relative).resolve()
        try:
            source.relative_to(path.parent.resolve())
        except ValueError as exc:
            raise ConversionError("canonical merge source escapes the input directory") from exc
        if not source.is_file() or source.is_symlink():
            raise ConversionError(f"canonical merge source does not exist: {relative}")
        sources.append(_read_json(source))
    return merge_canonical_datasets(sources)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ConversionError(f"refusing to overwrite {path}")
    path.write_bytes(canonical_json_bytes(value))


def _frameworks(value: str) -> list[str]:
    requested = value.split(",") if value else []
    if not requested or len(requested) != len(set(requested)) or any(x not in FRAMEWORKS for x in requested):
        raise ConversionError("frameworks must be a unique comma-separated subset of deepmd,m3gnet,chgnet,mace")
    return [name for name in FRAMEWORKS if name in requested]


def _type_map(canonical: Mapping[str, Any]) -> list[str]:
    result = []
    for record in canonical["records"]:
        for species in record["species"]:
            if species not in result:
                result.append(species)
    return result


def _deepmd(root: Path, canonical: Mapping[str, Any], split: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import dpdata
        import numpy as np
    except (ImportError, ModuleNotFoundError) as exc:
        raise DependencyBlocked("DeepMD conversion requires dpdata and NumPy") from exc
    target, type_map = root / "deepmd", _type_map(canonical)
    type_index = {name: index for index, name in enumerate(type_map)}
    index: dict[str, Any] = {"dataset_id": canonical["dataset_id"], "split_id": split["split_id"], "type_map": type_map, "partitions": {}}
    for split_name, directory_name in PARTITIONS:
        records = records_for_split(canonical, split, split_name)
        grouped: OrderedDict[tuple[str, ...], list[dict[str, Any]]] = OrderedDict()
        for record in records:
            grouped.setdefault(tuple(record["species"]), []).append(record)
        partition_index = []
        for system_index, (species, frames) in enumerate(grouped.items(), start=1):
            system_name = f"system-{system_index:06d}"
            system_path = target / directory_name / system_name
            cells = np.asarray([r["lattice"] for r in frames], dtype=np.float64)
            data: dict[str, Any] = {
                "atom_names": type_map,
                "atom_numbs": [sum(name == element for name in species) for element in type_map],
                "atom_types": np.asarray([type_index[name] for name in species], dtype=np.int32),
                "orig": np.zeros(3, dtype=np.float64),
                "cells": cells,
                "coords": np.asarray([cartesian_coordinates(r) for r in frames], dtype=np.float64),
                "energies": np.asarray([r["total_energy"] for r in frames], dtype=np.float64),
                "forces": np.asarray([r["atomic_forces"] for r in frames], dtype=np.float64),
            }
            if "stress" in canonical["targets"]:
                data["virials"] = np.asarray([
                    np.asarray(r["stress"], dtype=np.float64) * float(np.linalg.det(cell)) / VASP_KBAR_PER_EV_PER_ANGSTROM3
                    for r, cell in zip(frames, cells)
                ])
            labeled = dpdata.LabeledSystem(data=data)
            labeled.to("deepmd/npy", str(system_path), set_size=DEEPMD_SET_SIZE)
            reread = dpdata.LabeledSystem(str(system_path), fmt="deepmd/npy")
            if reread.get_nframes() != len(frames) or reread.get_natoms() != len(species):
                raise ConversionError("dpdata could not reread generated DeepMD data")
            partition_index.extend({"record_id": r["record_id"], "system": system_name, "frame_index": i} for i, r in enumerate(frames))
        order = {record_id: i for i, record_id in enumerate(split[f"{split_name}_record_ids"])}
        partition_index.sort(key=lambda item: order[item["record_id"]])
        index["partitions"][split_name] = partition_index
    _write_json(target / "record-index.json", index)
    return {"format": "dpdata deepmd/npy", "dpdata_version": importlib.metadata.version("dpdata")}


def _json_view(root: Path, canonical: Mapping[str, Any], split: Mapping[str, Any], framework: str) -> dict[str, Any]:
    target = root / framework
    for split_name, file_stem in PARTITIONS:
        records = records_for_split(canonical, split, split_name)
        _write_json(target / f"{file_stem}.json", framework_json_dataset(canonical["dataset_id"], split["split_id"], framework, records))
    return {"format": "JSON records", "stress_field": "stresses" if framework == "m3gnet" else "stress"}


def _mace(root: Path, canonical: Mapping[str, Any], split: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import numpy as np
        from ase import Atoms
        from ase.calculators.singlepoint import SinglePointCalculator
        from ase.io import write
        from ase.stress import full_3x3_to_voigt_6_stress
    except (ImportError, ModuleNotFoundError) as exc:
        raise DependencyBlocked("MACE conversion requires ASE and NumPy") from exc
    target = root / "mace"
    target.mkdir(parents=True)
    for split_name, file_stem in PARTITIONS:
        frames = []
        for record in records_for_split(canonical, split, split_name):
            atoms = Atoms(symbols=record["species"], cell=record["lattice"], scaled_positions=record["coordinates"]["values"], pbc=True)
            results = {"energy": float(record["total_energy"]), "forces": record["atomic_forces"]}
            if "stress" in canonical["targets"]:
                ase_stress = -np.asarray(record["stress"], dtype=float) / VASP_KBAR_PER_EV_PER_ANGSTROM3
                results["stress"] = full_3x3_to_voigt_6_stress(ase_stress)
            atoms.calc = SinglePointCalculator(atoms, **results)
            atoms.info.update({
                "mlipflow_record_id": record["record_id"],
                "mlipflow_source_structure_id": record["source_structure_id"],
                "mlipflow_calculation_id": record["calculation_id"],
                "mlipflow_ionic_step": int(record["ionic_step"]),
            })
            frames.append(atoms)
        write(str(target / f"{file_stem}.extxyz"), frames, format="extxyz", write_results=True)
    return {"format": "ASE extxyz", "ase_version": importlib.metadata.version("ase")}


def _reference(
    dataset_id: str, split_id: str, framework: str, relative_path: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset_id": f"{dataset_id}-{framework}-{split_id}",
        "relative_path": f"{relative_path}/{framework}",
        "kind": "directory",
        "split_id": split_id,
    }


def _same_numeric_tree(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and abs(float(actual) - float(expected))
            <= 1e-10 * max(1.0, abs(float(expected)))
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _same_numeric_tree(a, e) for a, e in zip(actual, expected)
        )
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(
                _same_numeric_tree(actual[key], value)
                for key, value in expected.items()
            )
        )
    return actual == expected


def _write_collected_references(
    *,
    output: Path,
    target: Path,
    canonical: Mapping[str, Any],
    split: Mapping[str, Any],
    frameworks: list[str],
    relative_path: str,
    result_name: str,
    result: Mapping[str, Any],
    benchmark: Mapping[str, Any],
) -> None:
    _write_json(output / "split.json", split)
    _write_json(output / result_name, result)
    _write_json(output / "benchmark-test.json", benchmark)
    _write_json(
        output / "benchmark-dataset-reference.json",
        {
            "schema_version": 1,
            "dataset_id": (
                f"{canonical['dataset_id']}-benchmark-{split['split_id']}"
            ),
            "relative_path": f"{relative_path}/benchmark/test.json",
            "kind": "file",
            "split_id": split["split_id"],
            "split": "test",
        },
    )
    for framework in frameworks:
        _write_json(
            output / f"{framework}-dataset-reference.json",
            _reference(
                str(canonical["dataset_id"]),
                str(split["split_id"]),
                framework,
                relative_path,
            ),
        )


def _reuse_existing(
    *,
    target: Path,
    output: Path,
    canonical: Mapping[str, Any],
    split: Mapping[str, Any],
    frameworks: list[str],
    relative_path: str,
    result_name: str,
    reference_dir: Path,
) -> dict[str, Any]:
    if target.is_symlink() or not target.is_dir():
        raise ConversionError("reused dataset path must be an ordinary directory")
    if _read_json(target / "canonical.json") != dict(canonical):
        raise ConversionError("reused dataset canonical records differ")
    if _read_json(target / "split.json") != dict(split):
        raise ConversionError("reused dataset split differs")
    result = _read_json(target / "assembly-result.json")
    expected_paths = {
        framework: f"{relative_path}/{framework}" for framework in frameworks
    }
    if (
        result.get("dataset_id") != canonical["dataset_id"]
        or result.get("split_id") != split["split_id"]
        or result.get("counts") != split["counts"]
        or result.get("framework_output_paths") != expected_paths
        or set(result.get("formats", {})) != set(frameworks)
        or result.get("benchmark_record_ids") != split["test_record_ids"]
    ):
        raise ConversionError("reused dataset assembly result differs")
    benchmark = _read_json(target / "benchmark" / "test.json")
    if not _same_numeric_tree(benchmark, benchmark_dataset(canonical, split)):
        raise ConversionError("reused benchmark differs from the approved test split")
    for framework in frameworks:
        expected_reference = _read_json(
            reference_dir / f"{framework}-dataset-reference.json"
        )
        actual_reference = _reference(
            str(canonical["dataset_id"]),
            str(split["split_id"]),
            framework,
            relative_path,
        )
        if expected_reference != actual_reference:
            raise ConversionError(f"reused {framework} dataset reference differs")
    expected_benchmark_reference = _read_json(
        reference_dir / "benchmark-dataset-reference.json"
    )
    actual_benchmark_reference = {
        "schema_version": 1,
        "dataset_id": f"{canonical['dataset_id']}-benchmark-{split['split_id']}",
        "relative_path": f"{relative_path}/benchmark/test.json",
        "kind": "file",
        "split_id": split["split_id"],
        "split": "test",
    }
    if expected_benchmark_reference != actual_benchmark_reference:
        raise ConversionError("reused benchmark reference differs")
    output.mkdir(parents=True, exist_ok=True)
    _write_collected_references(
        output=output,
        target=target,
        canonical=canonical,
        split=split,
        frameworks=frameworks,
        relative_path=relative_path,
        result_name=result_name,
        result=result,
        benchmark=benchmark,
    )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    canonical = _canonical_input(Path(args.canonical).resolve())
    errors = validate_canonical_dataset(canonical)
    if errors:
        raise ConversionError("invalid canonical dataset: " + "; ".join(errors))
    frameworks = _frameworks(args.frameworks)
    split = build_split_manifest(
        canonical, strategy=args.split_strategy, seed=args.split_seed,
        fractions={"train": args.train_fraction, "validation": args.validation_fraction, "test": args.test_fraction},
    )
    data_root, output = Path(args.data_root).resolve(), Path(args.output_dir).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    relative_path = args.dataset_relative_path or canonical["dataset_id"]
    if relative_path != canonical["dataset_id"]:
        raise ConversionError("dataset_relative_path must equal dataset_id")
    target = data_root / relative_path
    if target.exists() or target.is_symlink():
        if not getattr(args, "reuse_existing", False):
            raise ConversionError(
                f"dataset path already exists; reuse its collected references: {target}"
            )
        reference_value = getattr(args, "reuse_reference_dir", None)
        if not isinstance(reference_value, str) or not reference_value:
            raise ConversionError("reuse-existing requires approved reference files")
        return _reuse_existing(
            target=target,
            output=output,
            canonical=canonical,
            split=split,
            frameworks=frameworks,
            relative_path=relative_path,
            result_name=args.result_name,
            reference_dir=Path(reference_value).resolve(),
        )
    if getattr(args, "reuse_existing", False):
        raise ConversionError("approved reused dataset path does not exist")
    output.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{canonical['dataset_id']}.", dir=data_root))
    try:
        _write_json(temporary / "canonical.json", canonical)
        _write_json(temporary / "split.json", split)
        format_info = {}
        for framework in frameworks:
            if framework == "deepmd":
                format_info[framework] = _deepmd(temporary, canonical, split)
            elif framework in {"m3gnet", "chgnet"}:
                format_info[framework] = _json_view(temporary, canonical, split, framework)
            else:
                format_info[framework] = _mace(temporary, canonical, split)
        benchmark = benchmark_dataset(canonical, split)
        _write_json(temporary / "benchmark" / "test.json", benchmark)
        conventions = {
            "atom_order": "canonical order unchanged in every view",
            "species_mapping": "element symbols preserved; DeepMD type_map records first canonical occurrence",
            "energy": "total eV per configuration, unchanged",
            "forces": "eV/angstrom, unchanged",
            "stress": {
                "deepmd": "virial_eV = VASP_kbar * volume_A3 / 1602.176621; no sign inversion",
                "m3gnet": "VASP-native kbar 3x3; trainer stress_unit=kbar",
                "chgnet": "VASP-native kbar 3x3; CHGNet StructureData applies its -0.1 conversion",
                "mace": "ASE stress eV/angstrom^3 = -VASP_kbar / 1602.176621, ASE Voigt order",
            },
        }
        paths = {framework: f"{relative_path}/{framework}" for framework in frameworks}
        result = {
            "dataset_id": canonical["dataset_id"], "split_id": split["split_id"],
            "counts": dict(split["counts"]), "framework_output_paths": paths,
            "benchmark_output_path": f"{relative_path}/benchmark/test.json",
            "benchmark_record_ids": list(split["test_record_ids"]),
            "formats": format_info, "units_and_conventions": conventions,
        }
        _write_json(temporary / "assembly-result.json", result)
        temporary.rename(target)
        _write_collected_references(
            output=output,
            target=target,
            canonical=canonical,
            split=split,
            frameworks=frameworks,
            relative_path=relative_path,
            result_name=args.result_name,
            result=result,
            benchmark=benchmark,
        )
        return result
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frameworks", required=True)
    parser.add_argument("--dataset-relative-path")
    parser.add_argument("--split-strategy", choices=("deterministic", "group-aware"), default="deterministic")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=.8)
    parser.add_argument("--validation-fraction", type=float, default=.1)
    parser.add_argument("--test-fraction", type=float, default=.1)
    parser.add_argument("--result-name", default="dataset-assembly-result.json")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--reuse-reference-dir")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(_parser().parse_args(argv))
        return 0
    except DependencyBlocked as exc:
        print(json.dumps({"status": "BLOCKED", "message": str(exc)}), file=sys.stderr)
        return 3
    except (ConversionError, OSError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "message": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
