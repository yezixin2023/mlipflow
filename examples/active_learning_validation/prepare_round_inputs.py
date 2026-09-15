#!/usr/bin/env python3
"""Prepare exact, small data handoffs for the fresh active-learning validation.

This utility does no model inference, DFT, training, MD, or DIRECT selection.  It
only converts already collected ASE/canonical artifacts into the explicit JSON and
structure inputs consumed by the corresponding MLIPFlow plugins.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

ACTIVE_CANDIDATE_CONTRACT = "mlipflow/active-learning-candidates"


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_bytes(_json_bytes(value))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _canonical_split_ids(split: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    names = ("train_record_ids", "validation_record_ids", "test_record_ids")
    values: dict[str, list[str]] = {}
    ownership: dict[str, str] = {}
    for name in names:
        records = split.get(name)
        if not isinstance(records, list) or any(
            not isinstance(record_id, str) or not record_id for record_id in records
        ):
            raise ValueError(f"{name} must be an array of non-empty record IDs")
        if len(records) != len(set(records)):
            raise ValueError(f"{name} contains duplicate record IDs")
        for record_id in records:
            if record_id in ownership:
                raise ValueError(
                    f"record {record_id} appears in both {ownership[record_id]} and {name}"
                )
            ownership[record_id] = name
        values[name] = list(records)
    return (
        values["train_record_ids"],
        values["validation_record_ids"],
        values["test_record_ids"],
    )


def _active_learning_dataset_split(
    canonical_dataset_id: str,
    bootstrap_split: dict[str, Any],
    training_split: dict[str, Any],
) -> dict[str, Any]:
    calibration_ids = [
        *bootstrap_split["train_record_ids"],
        *bootstrap_split["validation_record_ids"],
    ]
    audit_ids = list(bootstrap_split["test_record_ids"])
    train_ids, validation_ids, excluded_test_ids = _canonical_split_ids(training_split)
    groups = (train_ids, validation_ids, excluded_test_ids, calibration_ids, audit_ids)
    flattened = [record_id for group in groups for record_id in group]
    if len(flattened) != len(set(flattened)):
        raise ValueError(
            "training, excluded test, calibration, and immutable audit IDs must be disjoint"
        )
    return {
        "dataset_id": f"{training_split['dataset_id']}-active-evaluation",
        "split_id": f"{training_split['split_id']}-active-evaluation",
        "bootstrap_dataset_id": canonical_dataset_id,
        "bootstrap_split_id": bootstrap_split["split_id"],
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "calibration_ids": calibration_ids,
        "audit_ids": audit_ids,
    }


def _fresh_directory(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to reuse {path}")
    path.mkdir(parents=True)


def _write_vasp(path: Path, atoms: Any) -> None:
    from ase.io import write

    write(path, atoms, format="vasp", direct=True, sort=False, vasp5=True)


def _parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise ValueError("expected NAME=/path")
    return name, Path(raw_path).expanduser().resolve()


def bootstrap(args: argparse.Namespace) -> None:
    from ase.io import read

    output = Path(args.output_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    _fresh_directory(output)
    frame_indexes = [int(value) for value in args.frame_indexes.split(",")]
    records = []
    source_trajectories = []
    for condition_id, trajectory in map(_parse_named_path, args.trajectory):
        frames = read(trajectory, index=":")
        source_trajectories.append(
            {
                "condition_id": condition_id,
                "path": str(trajectory),
                "frame_count": len(frames),
            }
        )
        for frame_index in frame_indexes:
            atoms = frames[frame_index]
            structure_id = f"bootstrap-{condition_id}-frame-{frame_index:04d}"
            filename = f"{structure_id}.vasp"
            structure_path = output / filename
            _write_vasp(structure_path, atoms)
            relative = structure_path.relative_to(manifest_path.parent).as_posix()
            records.append(
                {
                    "id": structure_id,
                    "path": relative,
                    "source_group_id": f"bootstrap-{condition_id}-frame-{frame_index:04d}",
                    "source_sampling_method": "ASE_MD",
                    "source_sampling_methods": ["ASE_MD"],
                    "source_records": [
                        {
                            "sampling_method": "ASE_MD",
                            "source_group_id": condition_id,
                            "source_id": structure_id,
                            "source_order": frame_index + 1,
                            "source_path": relative,
                        }
                    ],
                    "source_trajectory_path": str(trajectory),
                    "source_frame_index": frame_index,
                    "condition_id": condition_id,
                }
            )
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "structures": records,
            "generation": {
                "mode": "fixed-existing-trajectory-frames",
                "frame_indexes": frame_indexes,
                "source_trajectories": source_trajectories,
            },
        },
    )


def _parse_composition(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in value.split(","):
        symbol, separator, raw_count = item.partition("=")
        if not separator or not symbol or symbol in result:
            raise ValueError("composition must be ELEMENT=COUNT pairs")
        count = int(raw_count)
        if count < 1:
            raise ValueError("composition counts must be positive")
        result[symbol] = count
    if not result:
        raise ValueError("composition cannot be empty")
    return result


def _site_symbol(site: dict[str, Any]) -> str:
    species = site.get("species")
    if not isinstance(species, list) or len(species) != 1:
        raise ValueError("historical structure sites must have one species")
    symbol = species[0].get("element")
    if not isinstance(symbol, str) or not symbol:
        raise ValueError("historical structure site is missing its element")
    return symbol


def historical_chgnet_bootstrap(args: argparse.Namespace) -> None:
    """Capture a bounded, deterministic subset of an existing CHGNet VASP export."""

    source = Path(args.source_json).expanduser().resolve()
    generator = Path(args.generator_script).expanduser().resolve()
    output = Path(args.output_dir).resolve()
    _fresh_directory(output)
    raw = _read_json(source)
    required = ("structure", "uncorrected_total_energy", "force")
    if any(not isinstance(raw.get(name), list) for name in required):
        raise ValueError("historical CHGNet export is missing structure, energy, or force arrays")
    total = len(raw["structure"])
    if any(len(raw[name]) != total for name in required) or total == 0:
        raise ValueError("historical CHGNet arrays must have one shared non-zero length")

    expected_composition = _parse_composition(args.composition)
    eligible: list[tuple[int, dict[str, Any], list[str]]] = []
    for index, structure in enumerate(raw["structure"]):
        if not isinstance(structure, dict) or not isinstance(structure.get("sites"), list):
            raise ValueError(f"historical structure {index} is invalid")
        species = [_site_symbol(site) for site in structure["sites"]]
        counts = {symbol: species.count(symbol) for symbol in set(species)}
        if counts == expected_composition:
            eligible.append((index, structure, species))
    requested = args.calibration_count + args.audit_count
    if args.calibration_count < 2 or args.audit_count < 1 or len(eligible) < requested:
        raise ValueError("historical capture lacks the requested calibration/audit records")
    random.Random(args.seed).shuffle(eligible)

    records = []
    benchmark_samples = []
    for index, structure, species in eligible[:requested]:
        lattice = structure["lattice"]["matrix"]
        fractional = [site["abc"] for site in structure["sites"]]
        cartesian = [site["xyz"] for site in structure["sites"]]
        forces = raw["force"][index]
        energy = raw["uncorrected_total_energy"][index]
        record_id = f"historical-dft-record-{index:06d}"
        records.append(
            {
                "record_id": record_id,
                "source_structure_id": f"historical-aimd-record-{index:06d}",
                "source_group_id": "historical-aimd-condition-unresolved",
                "species": species,
                "lattice": lattice,
                "coordinates": {"kind": "fractional", "values": fractional},
                "total_energy": energy,
                "atomic_forces": forces,
                "source_record_index": index,
            }
        )
        benchmark_samples.append(
            {
                "id": record_id,
                "species": species,
                "positions": cartesian,
                "cell": lattice,
                "pbc": True,
                "references": {"energy": energy, "force": forces},
            }
        )

    calibration_ids = [record["record_id"] for record in records[: args.calibration_count]]
    audit_ids = [record["record_id"] for record in records[args.calibration_count :]]
    dataset_id = "historical-chgnet-aimd-capture"
    split_id = f"historical-calibration-audit-seed-{args.seed}"
    midpoint = len(calibration_ids) // 2
    capture = {
        "schema_version": 1,
        "contract": "mlipflow/historical-labeled-capture",
        "dataset_id": dataset_id,
        "record_count": len(records),
        "units": {"energy": "eV", "force": "eV/angstrom"},
        "condition_metadata": "HISTORICAL_PARAMETER_UNKNOWN",
        "records": records,
    }
    split = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "split_id": split_id,
        "train_record_ids": calibration_ids[:midpoint],
        "validation_record_ids": calibration_ids[midpoint:],
        "test_record_ids": audit_ids,
    }
    benchmark = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "split_id": split_id,
        "split": "test",
        "units": {"energy": "eV", "force": "eV/angstrom"},
        "energy_convention": "total",
        "stress_convention": None,
        "samples": benchmark_samples[args.calibration_count :],
    }
    provenance = {
        "schema_version": 1,
        "mode": "existing-result-structured-capture",
        "source_locator": args.source_locator,
        "source_path": str(source),
        "source_record_count": total,
        "selected_record_indexes": [record["source_record_index"] for record in records],
        "selection_seed": args.seed,
        "composition": expected_composition,
        "generator_script_locator": args.generator_locator,
        "generator_script_path": str(generator),
        "fresh_numerical_execution": False,
    }
    _write_json(output / "historical-capture.json", capture)
    _write_json(output / "split.json", split)
    _write_json(output / "audit-benchmark.json", benchmark)
    _write_json(output / "capture-provenance.json", provenance)


def _atoms_from_record(record: dict[str, Any]) -> Any:
    from ase import Atoms

    return Atoms(
        symbols=record["species"],
        scaled_positions=record["coordinates"]["values"],
        cell=record["lattice"],
        pbc=True,
    )


def _validity(atoms: Any, minimum_distance: float, severe_distance: float) -> dict[str, Any]:
    import numpy as np

    volume = float(atoms.get_volume())
    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    minimum = float(np.min(distances))
    finite = math.isfinite(volume) and math.isfinite(minimum)
    severe = not finite or volume <= 0 or minimum < severe_distance
    valid = finite and volume > 0 and minimum >= minimum_distance
    reasons = []
    if not finite:
        reasons.append("non-finite-cell-or-distance")
    elif volume <= 0:
        reasons.append("non-positive-cell-volume")
    elif minimum < severe_distance:
        reasons.append("severe-short-contact")
    elif minimum < minimum_distance:
        reasons.append("short-contact")
    return {
        "valid": valid,
        "severe": severe,
        "reasons": reasons,
        "minimum_distance_angstrom": minimum,
        "cell_volume_angstrom3": volume,
    }


def _near_duplicate_groups(items: list[tuple[str, Any]], ltol: float, stol: float, angle: float) -> dict[str, str]:
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.io.ase import AseAtomsAdaptor

    matcher = StructureMatcher(
        ltol=ltol,
        stol=stol,
        angle_tol=angle,
        primitive_cell=False,
        scale=False,
        attempt_supercell=False,
    )
    representatives: list[tuple[str, Any]] = []
    groups: dict[str, str] = {}
    for sample_id, atoms in items:
        structure = AseAtomsAdaptor.get_structure(atoms)
        match = next(
            (group for group, representative in representatives if matcher.fit(representative, structure)),
            None,
        )
        if match is None:
            match = f"near-{len(representatives) + 1:04d}"
            representatives.append((match, structure))
        groups[sample_id] = match
    return groups


def evaluation(args: argparse.Namespace) -> None:
    from ase.io import read

    canonical = _read_json(Path(args.bootstrap_canonical).resolve())
    bootstrap_split = _read_json(Path(args.bootstrap_split).resolve())
    training_split = _read_json(Path(args.training_split).resolve())
    output_dir = Path(args.output_dir).resolve()
    evaluation_path = Path(args.evaluation_dataset).resolve()
    candidate_manifest_path = Path(args.candidate_manifest).resolve()
    _fresh_directory(output_dir)

    by_id = {record["record_id"]: record for record in canonical["records"]}
    dataset_split = _active_learning_dataset_split(
        canonical["dataset_id"], bootstrap_split, training_split
    )
    calibration_ids = dataset_split["calibration_ids"]
    samples = []
    for record_id in calibration_ids:
        record = by_id[record_id]
        atoms = _atoms_from_record(record)
        samples.append(
            {
                "sample_id": record_id,
                "split": "calibration",
                "species": atoms.get_chemical_symbols(),
                "positions": atoms.get_positions().tolist(),
                "cell": atoms.cell.array.tolist(),
                "pbc": True,
                "reference_forces": record["atomic_forces"],
            }
        )

    candidate_rows: list[tuple[str, Any, dict[str, Any], Path]] = []
    for value in args.candidate_trajectory:
        fields = value.split("=", 2)
        if len(fields) != 3:
            raise ValueError("candidate trajectory must be CONDITION=REPLICA=/path")
        condition_id, replica, raw_path = fields
        trajectory = Path(raw_path).expanduser().resolve()
        frames = read(trajectory, index=":")
        kept = 0
        for frame_index in range(args.start_frame, len(frames), args.source_stride):
            if kept >= args.max_frames_per_trajectory:
                break
            atoms = frames[frame_index]
            sample_id = f"candidate-{condition_id}-{replica}-frame-{frame_index:04d}"
            path = output_dir / f"{sample_id}.vasp"
            _write_vasp(path, atoms)
            structure_path = path.relative_to(candidate_manifest_path.parent).as_posix()
            metadata = {
                "condition_id": condition_id,
                "condition": {
                    "temperature_k": float(condition_id.removesuffix("k")),
                    "ensemble": "NVT",
                },
                "replica": replica,
                "frame_index": frame_index,
                "structure_path": structure_path,
                "physical_validity": _validity(
                    atoms, args.minimum_distance, args.severe_distance
                ),
                "source_trajectory_path": str(trajectory),
            }
            candidate_rows.append((sample_id, atoms, metadata, path))
            kept += 1
    groups = _near_duplicate_groups(
        [(sample_id, atoms) for sample_id, atoms, _, _ in candidate_rows],
        args.matcher_ltol,
        args.matcher_stol,
        args.matcher_angle_tol,
    )
    candidates = []
    for sample_id, atoms, metadata, path in candidate_rows:
        metadata["near_duplicate_group"] = groups[sample_id]
        samples.append(
            {
                "sample_id": sample_id,
                "split": "candidate",
                "species": atoms.get_chemical_symbols(),
                "positions": atoms.get_positions().tolist(),
                "cell": atoms.cell.array.tolist(),
                "pbc": True,
                **metadata,
            }
        )
        candidates.append(
            {
                "id": sample_id,
                "structure_path": path.relative_to(candidate_manifest_path.parent).as_posix(),
            }
        )
    _write_json(
        evaluation_path,
        {
            "schema_version": 1,
            "contract": "mlipflow/active-learning-evaluation-dataset",
            "units": {"energy": "eV", "force": "eV/angstrom"},
            "dataset_split": dataset_split,
            "samples": samples,
        },
    )
    _write_json(
        candidate_manifest_path,
        {
            "schema_version": 1,
            "contract": ACTIVE_CANDIDATE_CONTRACT,
            "candidates": candidates,
        },
    )


def model_index(args: argparse.Namespace) -> None:
    policy = _read_json(Path(args.policy).resolve())
    contract = _load_dataset_contract(Path(args.dataset_contract).expanduser().resolve())
    canonical_sources = [
        _read_json(Path(value).expanduser().resolve()) for value in args.canonical_source
    ]
    canonical = contract.merge_canonical_datasets(canonical_sources)
    model_id = policy["strategy"]["primary_model"]
    expected_seeds = {
        item["model_id"]: item["member_seeds"] for item in policy["committee"]["models"]
    }[model_id]
    members = []
    shared_training_record = None
    for value in args.member_reference:
        raw_seed, separator, raw_path = value.partition("=")
        if not separator:
            raise ValueError("member reference must be SEED=/path")
        seed = int(raw_seed)
        reference_path = Path(raw_path).expanduser().resolve()
        reference = _read_json(reference_path)
        report_path = reference_path.with_name("cluster-run-report.json")
        result_path = reference_path.with_name("training-result.json")
        report = _read_json(report_path)
        result = _read_json(result_path)
        if report.get("status") != "OK" or report.get("return_code") != 0:
            raise ValueError(f"seed {seed} training cluster report is not OK")
        if result.get("status") != "OK" or result.get("seed") != seed:
            raise ValueError(f"seed {seed} training result is not OK or has seed drift")
        published = report.get("published_model")
        reference_fields = ("schema_version", "model_id", "framework", "relative_path", "kind")
        if not isinstance(published, dict) or any(
            published.get(name) != reference.get(name) for name in reference_fields
        ):
            raise ValueError(f"seed {seed} published model differs from model reference")
        model_artifact = result.get("model_artifact")
        if not isinstance(model_artifact, dict) or not isinstance(
            model_artifact.get("path"), str
        ):
            raise ValueError(f"seed {seed} training result differs from model reference")
        split = result.get("provenance", {}).get("split")
        if not isinstance(split, dict) or split.get("source") != "predefined":
            raise ValueError(f"seed {seed} lacks a predefined train/validation/test split")
        dataset = report.get("dataset")
        foundation = report.get("foundation_model")
        if not isinstance(dataset, dict):
            raise ValueError(f"seed {seed} lacks a training dataset record")
        if (
            report.get("framework") != result.get("framework")
            or report.get("operation") != result.get("operation")
        ):
            raise ValueError(f"seed {seed} training result differs from cluster report")
        expected_dataset_id = (
            f"{canonical['dataset_id']}-{result.get('framework')}-{split.get('split_id')}"
        )
        if dataset.get("id") != expected_dataset_id:
            raise ValueError(f"seed {seed} training export differs from canonical dataset/split")
        training_record = {
            "framework": result.get("framework"),
            "operation": result.get("operation"),
            "dataset": {
                "id": dataset.get("id"),
                "relative_path": dataset.get("relative_path"),
            },
            "canonical_dataset_id": canonical["dataset_id"],
            "split": {
                name: split.get(name)
                for name in (
                    "split_id",
                    "train_record_ids",
                    "validation_record_ids",
                    "test_record_ids",
                )
            },
            "precision": result.get("precision"),
            "foundation_model": (
                {
                    "id": foundation.get("id"),
                    "relative_path": foundation.get("relative_path"),
                }
                if isinstance(foundation, dict)
                else None
            ),
            "framework_version": result.get("framework_version"),
        }
        if shared_training_record is None:
            shared_training_record = training_record
        elif training_record != shared_training_record:
            raise ValueError(
                "committee members must share the training dataset, split, and parameters"
            )
        members.append(
            {
                "member_id": f"{model_id}-seed-{seed}",
                "seed": seed,
                "relative_path": reference["relative_path"],
                "kind": reference["kind"],
            }
        )
    if sorted(member["seed"] for member in members) != sorted(expected_seeds):
        raise ValueError("member reference seeds differ from policy")
    _write_json(
        Path(args.output).resolve(),
        {
            "schema_version": 1,
            "contract": "mlipflow/active-learning-committee-model-index",
            "strategy": policy["strategy"],
            "models": [
                {
                    "model_id": model_id,
                    "model_family": "chgnet",
                    "framework": "chgnet",
                    "supports_stress": True,
                    "training": shared_training_record,
                    "members": members,
                }
            ],
        },
    )


def direct_input(args: argparse.Namespace) -> None:
    from ase.io import read, write
    from mlipflow.plugins.active_learning import science as active_learning

    evaluation_value = _read_json(Path(args.committee_evaluation).resolve())
    policy = _read_json(Path(args.policy).resolve())
    manifest_path = Path(args.candidate_manifest).resolve()
    manifest = _read_json(manifest_path)
    output_dir = Path(args.output_dir).resolve()
    _fresh_directory(output_dir)
    selection_policy = active_learning._selection_policy(policy)
    prefiltered, _ = active_learning._prefilter_query_candidates(
        evaluation_value["candidates"], selection_policy
    )
    expected_ids = [item["sample_id"] for item in prefiltered]
    by_id = {item["id"]: item for item in manifest["candidates"]}
    frames = [read(manifest_path.parent / by_id[sample_id]["structure_path"]) for sample_id in expected_ids]
    if not frames:
        raise ValueError("no QUERY candidates remain for DIRECT")
    write(output_dir / "query.extxyz", frames, format="extxyz")
    _write_json(
        output_dir / "input-order.json",
        {"schema_version": 1, "input_candidate_ids": expected_ids},
    )


def selected_manifests(args: argparse.Namespace) -> None:
    selection = _read_json(Path(args.selection_result).resolve())
    candidates = _read_json(Path(args.candidate_manifest).resolve())
    by_id = {item["id"]: item for item in candidates["candidates"]}

    def build(key: str) -> dict[str, Any]:
        structures = []
        for item in selection[key]:
            candidate = by_id[item["sample_id"]]
            structures.append(
                {
                    "id": item["sample_id"],
                    "path": candidate["structure_path"],
                    "source_group_id": (
                        f"{item['condition_id']}-{item['replica']}-frame-{item['frame_index']:04d}"
                    ),
                    "active_learning_selection_kind": item["selection_kind"],
                }
            )
        if not structures:
            raise ValueError(f"{key} is empty")
        return {"schema_version": 1, "structures": structures}

    _write_json(Path(args.query_manifest).resolve(), build("selected_query_candidates"))
    _write_json(Path(args.spot_manifest).resolve(), build("selected_safe_spot_checks"))


def cumulative_merge(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    source_dir = output.parent / "canonical-sources"
    _fresh_directory(source_dir)
    sources = []
    for index, raw_path in enumerate(args.canonical_source, start=1):
        source = Path(raw_path).expanduser().resolve()
        target = source_dir / f"source-{index:04d}.json"
        shutil.copy2(source, target)
        sources.append(
            {
                "path": target.relative_to(output.parent).as_posix(),
            }
        )
    _write_json(
        output,
        {
            "schema_version": 1,
            "contract": "mlipflow/canonical-dataset-merge",
            "sources": sources,
        },
    )


def _load_dataset_contract(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "mlipflow_validation_dataset_contract", path
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load dataset contract from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_legacy_canonical(args: argparse.Namespace) -> None:
    """Copy an older canonical dataset into the current natural-ID contract."""

    contract = _load_dataset_contract(Path(args.dataset_contract).expanduser().resolve())
    source = _read_json(Path(args.source_canonical).expanduser().resolve())
    source_split = _read_json(Path(args.source_split).expanduser().resolve())
    records = source.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("source canonical dataset has no records")
    calculation_types = {record.get("calculation_type") for record in records}
    if len(calculation_types) != 1:
        raise ValueError("source canonical records must share one calculation type")

    structures = []
    labels = []
    raw_outputs: dict[str, Any] = {}
    for record_index, record in enumerate(records, start=1):
        structure = dict(record["source_structure"])
        structure.pop("fingerprint", None)
        structure_id = f"initial-label-{record_index:04d}"
        structure["structure_id"] = structure_id
        if "source_records" in structure:
            structure["source_records"] = [
                {
                    **{
                        key: value
                        for key, value in source_record.items()
                        if key not in {"source_id", "source_sha256"}
                    },
                    "source_id": f"source-record-{record_index:04d}-{source_index:04d}",
                }
                for source_index, source_record in enumerate(
                    structure["source_records"], start=1
                )
            ]
        structures.append(structure)
        labels.append(
            {
                "structure_id": structure_id,
                "calculation_id": record["calculation_id"],
                "ionic_step": record["ionic_step"],
                "species": record["species"],
                "lattice_angstrom": record["lattice"],
                "fractional_coordinates": record["coordinates"]["values"],
                "energy_ev": record["total_energy"],
                "forces_ev_per_angstrom": record["atomic_forces"],
                "stress_kbar_vasp_3x3": record["stress"],
                "source_dft_attempt": record["source_dft_attempt"],
                **(
                    {"source_group_id": record["source_group_id"]}
                    if "source_group_id" in record
                    else {}
                ),
            }
        )
        for name, output in record.get("source_dft_outputs", {}).items():
            raw_outputs[name] = {
                key: output[key]
                for key in ("path", "source_dft_attempt")
                if key in output
            }

    normalized = contract.build_canonical_dataset(
        label_records=labels,
        units=source["units"],
        calculation_type=calculation_types.pop(),
        source_attempt_identity=source["source_attempt_identity"],
        structures_manifest={"structures": structures},
        raw_outputs=raw_outputs,
    )
    errors = contract.validate_canonical_dataset(normalized)
    if errors:
        raise ValueError("normalized canonical dataset is invalid: " + "; ".join(errors))

    old_to_new = {
        old["record_id"]: new["record_id"]
        for old, new in zip(records, normalized["records"], strict=True)
    }
    train_ids, validation_ids, test_ids = _canonical_split_ids(source_split)
    normalized_split = {
        "dataset_id": normalized["dataset_id"],
        "split_id": (
            f"{normalized['dataset_id']}-{source_split['strategy']}-"
            f"seed-{source_split['seed']}-preserved-membership"
        ),
        "strategy": source_split["strategy"],
        "seed": source_split["seed"],
        "train_record_ids": [old_to_new[value] for value in train_ids],
        "validation_record_ids": [old_to_new[value] for value in validation_ids],
        "test_record_ids": [old_to_new[value] for value in test_ids],
        "counts": {
            "train": len(train_ids),
            "validation": len(validation_ids),
            "test": len(test_ids),
            "total": len(records),
        },
    }
    errors = contract.validate_split_manifest(normalized, normalized_split)
    if errors:
        raise ValueError("normalized split is invalid: " + "; ".join(errors))
    _write_json(Path(args.canonical_output).resolve(), normalized)
    _write_json(Path(args.split_output).resolve(), normalized_split)


def split_seed_review(args: argparse.Namespace) -> None:
    """Choose the first split seed that preserves active-learning membership."""

    contract_path = Path(args.dataset_contract).expanduser().resolve()
    contract = _load_dataset_contract(contract_path)
    source_paths = [Path(value).expanduser().resolve() for value in args.canonical_source]
    sources = [_read_json(path) for path in source_paths]
    canonical = contract.merge_canonical_datasets(sources)
    previous_split = _read_json(Path(args.previous_training_split).expanduser().resolve())
    previous_train, previous_validation, _ = _canonical_split_ids(previous_split)

    query_source_ids = list(args.query_source_structure_id)
    if len(query_source_ids) != len(set(query_source_ids)):
        raise ValueError("QUERY source structure IDs must be unique")
    query_records = {}
    for source_id in query_source_ids:
        matches = [
            record["record_id"]
            for record in canonical["records"]
            if record.get("source_structure_id") == source_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"QUERY source structure {source_id} must resolve to one canonical record"
            )
        query_records[source_id] = matches[0]

    canonical_ids = {record["record_id"] for record in canonical["records"]}
    required_ids = [*previous_train, *previous_validation, *query_records.values()]
    if len(required_ids) != len(set(required_ids)):
        raise ValueError("prior training/validation and new QUERY records must be disjoint")
    if not set(required_ids) <= canonical_ids:
        raise ValueError("required training/validation records are absent from canonical merge")
    if args.maximum_seed < 0:
        raise ValueError("maximum seed must be non-negative")

    fractions = {
        "train": args.train_fraction,
        "validation": args.validation_fraction,
        "test": args.test_fraction,
    }
    selected = None
    for seed in range(args.maximum_seed + 1):
        candidate = contract.build_split_manifest(
            canonical,
            strategy="deterministic",
            seed=seed,
            fractions=fractions,
        )
        training_ids = set(candidate["train_record_ids"]) | set(
            candidate["validation_record_ids"]
        )
        if set(required_ids) <= training_ids:
            selected = candidate
            break
    if selected is None:
        raise ValueError(
            "no reviewed split seed preserves prior train/validation and new QUERY records"
        )

    _write_json(
        Path(args.output).resolve(),
        {
            "schema_version": 1,
            "contract": "mlipflow/active-learning-split-seed-review",
            "selection_rule": (
                "lowest-nonnegative-seed-preserving-prior-train-validation-and-new-query"
            ),
            "canonical_sources": [
                {
                    "index": index,
                    "dataset_id": source["dataset_id"],
                }
                for index, source in enumerate(sources, start=1)
            ],
            "merged_dataset_id": canonical["dataset_id"],
            "previous_training_split_id": previous_split["split_id"],
            "query_source_structure_records": query_records,
            "required_training_validation_record_ids": required_ids,
            "maximum_seed_reviewed": args.maximum_seed,
            "selected_seed": selected["seed"],
            "split_fractions": fractions,
            "split_manifest": selected,
        },
    )


def prediction_split_rebind(args: argparse.Namespace) -> None:
    """Replace only a known-bad split claim while preserving fresh predictions."""

    source_path = Path(args.committee_predictions).resolve()
    evaluation_path = Path(args.evaluation_dataset).resolve()
    source = _read_json(source_path)
    evaluation_value = _read_json(evaluation_path)
    source_split = source.get("dataset_split")
    corrected_split = evaluation_value.get("dataset_split")
    if not isinstance(source_split, dict) or not isinstance(corrected_split, dict):
        raise ValueError("prediction and evaluation manifests require dataset_split")
    for name in ("calibration_ids", "audit_ids"):
        if source_split.get(name) != corrected_split.get(name):
            raise ValueError(f"split rebind cannot change {name}")
    models = source.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("committee predictions require non-empty model evidence")
    source["dataset_split"] = corrected_split
    source["split_rebind"] = {
        "mode": "existing-result-structured-correction",
        "source_prediction_path": str(source_path),
        "corrected_evaluation_path": str(evaluation_path),
        "reason": "exclude canonical test records that were not used for training or validation",
        "numerical_predictions_changed": False,
    }
    _write_json(Path(args.output).resolve(), source)


def direct_selection_replay(args: argparse.Namespace) -> None:
    """Normalize one collected active-learning/DIRECT binding for exact replay."""

    source_path = Path(args.selection_result).resolve()
    source = _read_json(source_path)
    diversity = source.get("diversity_selection")
    input_ids = source.get("direct_input_candidate_ids")
    selected_ids = source.get("direct_selected_candidate_ids")
    if not isinstance(diversity, dict) or diversity.get("method") != "DIRECT":
        raise ValueError("selection result lacks reviewed DIRECT evidence")
    if not isinstance(input_ids, list) or not isinstance(selected_ids, list):
        raise ValueError("selection result lacks exact DIRECT candidate IDs")
    if len(input_ids) != len(set(input_ids)) or not set(selected_ids) <= set(input_ids):
        raise ValueError("selection result has invalid DIRECT candidate IDs")
    _write_json(
        Path(args.output).resolve(),
        {
            "schema_version": 1,
            "method": "DIRECT",
            "input_candidate_ids": input_ids,
            "selected_candidate_ids": selected_ids,
            "parameters": {
                **diversity.get("parameters", {}),
                "source_selection_path": str(source_path),
                "replay_mode": "existing-result-structured-capture",
            },
        },
    )


def explicit_spot_training_policy(args: argparse.Namespace) -> None:
    """Make the reviewed SAFE spot-training choice explicit in continuation inputs."""

    policy = _read_json(Path(args.policy).resolve())
    selection = _read_json(Path(args.selection_result).resolve())
    policy.setdefault("selection", {})[
        "include_safe_spot_checks_in_training"
    ] = args.include
    selection["include_safe_spot_checks_in_training"] = args.include
    _write_json(Path(args.policy_output).resolve(), policy)
    _write_json(Path(args.selection_output).resolve(), selection)


def _audit_handoff(
    metric_values: list[str], evidence_values: list[str], audit_ids: list[str]
) -> dict[str, Any]:
    metric_paths = dict(map(_parse_named_path, metric_values))
    evidence_paths = dict(map(_parse_named_path, evidence_values))
    if not metric_paths or len(metric_paths) != len(metric_values):
        raise ValueError("audit metric member names must be unique and non-empty")
    if set(metric_paths) != set(evidence_paths) or len(evidence_paths) != len(
        evidence_values
    ):
        raise ValueError("audit metrics and prediction evidence require identical member names")
    required_metrics = {
        "energy_mae",
        "energy_rmse",
        "force_mae",
        "force_rmse",
        "maximum_atomic_force_error",
    }
    member_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    member_evidence = []
    for member in sorted(metric_paths):
        metrics = _read_json(metric_paths[member])
        evidence = _read_json(evidence_paths[member])
        evidence_ids = {
            record.get("sample_id")
            for record in evidence.get("records", [])
            if isinstance(record, dict)
        }
        if None in evidence_ids or evidence_ids != set(audit_ids):
            raise ValueError(
                f"{member} fresh benchmark evidence differs from immutable audit IDs"
            )
        if evidence.get("split") not in {"audit", "test"}:
            raise ValueError(f"{member} fresh benchmark must use the audit/test split")
        rows = {
            record["metric"]: record
            for record in metrics.get("records", [])
            if isinstance(record, dict) and record.get("metric") in required_metrics
        }
        if set(rows) != required_metrics:
            raise ValueError(f"{member} fresh benchmark lacks a required audit metric")
        member_metrics[member] = rows
        member_evidence.append(
            {
                "member": member,
                "prediction_evidence_path": str(evidence_paths[member]),
                "metrics_path": str(metric_paths[member]),
            }
        )

    aggregate_records = []
    for metric in sorted(required_metrics):
        rows = [member_metrics[member][metric] for member in sorted(member_metrics)]
        units = {row.get("unit") for row in rows}
        if len(units) != 1 or None in units:
            raise ValueError(f"audit metric unit drift for {metric}")
        values = {
            member: float(member_metrics[member][metric]["value"])
            for member in sorted(member_metrics)
        }
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError(f"audit metric {metric} contains a non-finite value")
        aggregate_records.append(
            {
                "model": "chgnet-primary",
                "metric": metric,
                "value": max(values.values()),
                "unit": units.pop(),
                "split": "audit",
                "aggregation": "maximum-over-committee-members",
                "member_values": values,
            }
        )
    return {
        "schema_version": 1,
        "audit_ids": audit_ids,
        "aggregation": "maximum-over-committee-members",
        "member_evidence": member_evidence,
        "records": aggregate_records,
    }


def assessment_inputs(args: argparse.Namespace) -> None:
    from mlipflow.plugins.active_learning.science import committee_mean_force_error

    evaluation_value = _read_json(Path(args.committee_evaluation).resolve())
    predictions = _read_json(Path(args.committee_predictions).resolve())
    selection = _read_json(Path(args.selection_result).resolve())
    query = _read_json(Path(args.query_canonical).resolve())
    spot = _read_json(Path(args.spot_canonical).resolve())
    cumulative_split = _read_json(Path(args.cumulative_split).resolve())
    policy = _read_json(Path(args.policy).resolve())
    campaign = _read_json(Path(args.campaign).resolve())
    include_spot = policy.get("selection", {}).get(
        "include_safe_spot_checks_in_training"
    )
    if not isinstance(include_spot, bool):
        raise ValueError("policy must declare SAFE spot-check training inclusion")

    audit_ids = list(evaluation_value["dataset_split"]["audit_ids"])
    audit_handoff = _audit_handoff(
        list(args.audit_metrics), list(args.audit_evidence), audit_ids
    )

    selected_query = {item["sample_id"] for item in selection["selected_query_candidates"]}
    selected_spot = {
        item["sample_id"] for item in selection["selected_safe_spot_checks"]
    }
    query_records = {
        record["source_structure_id"]: record["record_id"] for record in query["records"]
    }
    query_ids = set(query_records)
    spot_by_id = {record["source_structure_id"]: record for record in spot["records"]}
    spot_records = {
        source_id: record["record_id"] for source_id, record in spot_by_id.items()
    }
    if not query_ids <= selected_query or set(spot_by_id) != selected_spot:
        raise ValueError("collected DFT structures differ from active-learning selection")
    failed_query_ids = selected_query - query_ids

    training_ids, validation_ids, test_ids = _canonical_split_ids(cumulative_split)
    cumulative_record_ids = set(training_ids) | set(validation_ids)
    if not set(query_records.values()) <= cumulative_record_ids:
        raise ValueError(
            "cumulative train/validation split does not contain every QUERY DFT record"
        )
    spot_training_ids = set(spot_records.values()) & cumulative_record_ids
    if (include_spot and spot_training_ids != set(spot_records.values())) or (
        not include_spot and spot_training_ids
    ):
        raise ValueError("cumulative split differs from SAFE spot-check training policy")
    assessment_split = {
        "dataset_id": cumulative_split["dataset_id"],
        "split_id": cumulative_split["split_id"],
        "train_ids": training_ids,
        "validation_ids": validation_ids,
        "calibration_ids": evaluation_value["dataset_split"]["calibration_ids"],
        "audit_ids": evaluation_value["dataset_split"]["audit_ids"],
    }

    model = next(
        value for value in predictions["models"] if value["model_id"] == "chgnet-primary"
    )
    member_predictions = [
        {item["sample_id"]: item for item in member["predictions"]}
        for member in model["members"]
    ]
    evaluation_by_id = {
        item["sample_id"]: item for item in evaluation_value["candidates"]
    }
    spot_samples = []
    for sample_id in sorted(selected_spot):
        force_predictions = [member[sample_id]["forces"] for member in member_predictions]
        actual_error = committee_mean_force_error(
            force_predictions, spot_by_id[sample_id]["atomic_forces"]
        )
        predicted_error = evaluation_by_id[sample_id]["models"]["chgnet-primary"][
            "estimated_force_error"
        ]
        spot_samples.append(
            {
                "sample_id": sample_id,
                "model": "chgnet-primary",
                "predicted_force_error": predicted_error,
                "actual_force_error": actual_error,
                "unit": "eV/angstrom",
            }
        )

    _write_json(Path(args.audit_output).resolve(), audit_handoff)
    _write_json(Path(args.spot_output).resolve(), {"schema_version": 1, "samples": spot_samples})
    _write_json(
        Path(args.labeling_output).resolve(),
        {
            "schema_version": 1,
            "status": "OK" if not failed_query_ids else "INCOMPLETE",
            "successful_query_records": [
                {"sample_id": sample_id, "record_id": query_records[sample_id]}
                for sample_id in sorted(query_ids)
            ],
            "failed_query_records": [
                {
                    "sample_id": sample_id,
                    "status": "DFT_FAIL",
                    "reason": "not-present-in-verified-canonical-labels",
                }
                for sample_id in sorted(failed_query_ids)
            ],
            "successful_spot_check_records": [
                {"sample_id": sample_id, "record_id": spot_records[sample_id]}
                for sample_id in sorted(selected_spot)
            ],
            "cumulative_dft_label_count": len(
                set(training_ids)
                | set(validation_ids)
                | set(test_ids)
                | set(spot_records.values())
            ),
            "canonical_training_record_count": len(training_ids)
            + len(validation_ids)
            + len(test_ids),
            "cumulative_dataset_id": cumulative_split["dataset_id"],
            "cumulative_split_id": cumulative_split["split_id"],
        },
    )
    _write_json(Path(args.dataset_split_output).resolve(), assessment_split)
    campaign["policy_id"] = policy["policy_id"]
    campaign["current_cumulative_dataset"] = {
        "dataset_id": cumulative_split["dataset_id"],
        "split_id": cumulative_split["split_id"],
        "reference": str(Path(args.cumulative_split).resolve()),
    }
    campaign["current_decision"] = "PENDING"
    _write_json(Path(args.campaign_output).resolve(), campaign)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="operation", required=True)

    command = subparsers.add_parser("bootstrap")
    command.add_argument("--trajectory", action="append", required=True)
    command.add_argument("--frame-indexes", required=True)
    command.add_argument("--output-dir", required=True)
    command.add_argument("--manifest", required=True)
    command.set_defaults(function=bootstrap)

    command = subparsers.add_parser("historical-chgnet-bootstrap")
    command.add_argument("--source-json", required=True)
    command.add_argument("--source-locator", required=True)
    command.add_argument("--generator-script", required=True)
    command.add_argument("--generator-locator", required=True)
    command.add_argument("--composition", required=True)
    command.add_argument("--calibration-count", type=int, default=6)
    command.add_argument("--audit-count", type=int, default=6)
    command.add_argument("--seed", type=int, default=47)
    command.add_argument("--output-dir", required=True)
    command.set_defaults(function=historical_chgnet_bootstrap)

    command = subparsers.add_parser("evaluation")
    command.add_argument("--bootstrap-canonical", required=True)
    command.add_argument("--bootstrap-split", required=True)
    command.add_argument("--training-split", required=True)
    command.add_argument("--candidate-trajectory", action="append", required=True)
    command.add_argument("--output-dir", required=True)
    command.add_argument("--evaluation-dataset", required=True)
    command.add_argument("--candidate-manifest", required=True)
    command.add_argument("--start-frame", type=int, default=1)
    command.add_argument("--source-stride", type=int, default=1)
    command.add_argument("--max-frames-per-trajectory", type=int, default=10)
    command.add_argument("--minimum-distance", type=float, default=0.5)
    command.add_argument("--severe-distance", type=float, default=0.4)
    command.add_argument("--matcher-ltol", type=float, default=0.01)
    command.add_argument("--matcher-stol", type=float, default=0.05)
    command.add_argument("--matcher-angle-tol", type=float, default=1.0)
    command.set_defaults(function=evaluation)

    command = subparsers.add_parser("model-index")
    command.add_argument("--policy", required=True)
    command.add_argument("--dataset-contract", required=True)
    command.add_argument("--canonical-source", action="append", required=True)
    command.add_argument("--member-reference", action="append", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(function=model_index)

    command = subparsers.add_parser("prediction-split-rebind")
    command.add_argument("--committee-predictions", required=True)
    command.add_argument("--evaluation-dataset", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(function=prediction_split_rebind)

    command = subparsers.add_parser("direct-selection-replay")
    command.add_argument("--selection-result", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(function=direct_selection_replay)

    command = subparsers.add_parser("explicit-spot-training-policy")
    command.add_argument("--policy", required=True)
    command.add_argument("--selection-result", required=True)
    command.add_argument("--include", action="store_true")
    command.add_argument("--policy-output", required=True)
    command.add_argument("--selection-output", required=True)
    command.set_defaults(function=explicit_spot_training_policy)

    command = subparsers.add_parser("direct-input")
    command.add_argument("--committee-evaluation", required=True)
    command.add_argument("--candidate-manifest", required=True)
    command.add_argument("--policy", required=True)
    command.add_argument("--output-dir", required=True)
    command.set_defaults(function=direct_input)

    command = subparsers.add_parser("selected-manifests")
    command.add_argument("--selection-result", required=True)
    command.add_argument("--candidate-manifest", required=True)
    command.add_argument("--query-manifest", required=True)
    command.add_argument("--spot-manifest", required=True)
    command.set_defaults(function=selected_manifests)

    command = subparsers.add_parser("cumulative-merge")
    command.add_argument("--canonical-source", action="append", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(function=cumulative_merge)

    command = subparsers.add_parser("normalize-legacy-canonical")
    command.add_argument("--dataset-contract", required=True)
    command.add_argument("--source-canonical", required=True)
    command.add_argument("--source-split", required=True)
    command.add_argument("--canonical-output", required=True)
    command.add_argument("--split-output", required=True)
    command.set_defaults(function=normalize_legacy_canonical)

    command = subparsers.add_parser("split-seed-review")
    command.add_argument("--dataset-contract", required=True)
    command.add_argument("--canonical-source", action="append", required=True)
    command.add_argument("--previous-training-split", required=True)
    command.add_argument("--query-source-structure-id", action="append", required=True)
    command.add_argument("--train-fraction", type=float, default=0.6)
    command.add_argument("--validation-fraction", type=float, default=0.2)
    command.add_argument("--test-fraction", type=float, default=0.2)
    command.add_argument("--maximum-seed", type=int, default=10000)
    command.add_argument("--output", required=True)
    command.set_defaults(function=split_seed_review)

    command = subparsers.add_parser("assessment-inputs")
    command.add_argument("--policy", required=True)
    command.add_argument("--committee-evaluation", required=True)
    command.add_argument("--committee-predictions", required=True)
    command.add_argument("--selection-result", required=True)
    command.add_argument("--query-canonical", required=True)
    command.add_argument("--spot-canonical", required=True)
    command.add_argument("--audit-metrics", action="append", required=True)
    command.add_argument("--audit-evidence", action="append", required=True)
    command.add_argument("--cumulative-split", required=True)
    command.add_argument("--campaign", required=True)
    command.add_argument("--audit-output", required=True)
    command.add_argument("--spot-output", required=True)
    command.add_argument("--labeling-output", required=True)
    command.add_argument("--dataset-split-output", required=True)
    command.add_argument("--campaign-output", required=True)
    command.set_defaults(function=assessment_inputs)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
