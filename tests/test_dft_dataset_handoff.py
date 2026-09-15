"""Canonical DFT -> one split -> four training view tests."""

from __future__ import annotations

import builtins
from tests.helpers import load_module
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlipflow.hpc import resolve_hpc_execution_plan
from mlipflow.services.contracts import _safe_remote_relative
from mlipflow.site import ClusterProfile
from tests.helpers import write_json
from tests.test_scheduled_multi_calculation import ScheduledLifecycle

ROOT = Path(__file__).resolve().parents[1]
DFT_PLUGIN = ROOT / "mlipflow" / "plugins" / "dft_labeling"


def _load(name: str, path: Path):
    module = load_module(path, name)
    return module


def _contract():
    return _load("dft_dataset_contract_test", DFT_PLUGIN / "dataset_contract.py")


def _canonical(
    count: int = 10,
    *,
    groups: bool = False,
    sampling: bool = False,
    project_id: str = "dataset",
    start: int = 0,
) -> dict:
    contract = _contract()
    labels, structures = [], []
    for index in range(start, start + count):
        structure_id = f"structure-{index:03d}"
        item = {"id": structure_id, "path": f"structures/{structure_id}.vasp"}
        if groups:
            item["source_group_id"] = f"trajectory-{index // 3:03d}"
        if sampling:
            item.update(
                {
                    "source_sampling_method": "DIRECT",
                    "source_sampling_methods": ["DIRECT", "LASP_SSW"],
                    "source_records": [
                        {
                            "sampling_method": "DIRECT",
                            "source_group_id": f"trajectory-{index:03d}",
                            "source_id": f"direct-{index:06d}",
                            "source_order": index + 1,
                            "source_path": f"selected/{structure_id}.vasp",
                        }
                    ],
                }
            )
        structures.append(item)
        labels.append({
            "structure_id": structure_id, "calculation_id": f"calc-{index:04d}",
            "ionic_step": 1, "species": ["Li", "S"],
            "lattice_angstrom": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
            "fractional_coordinates": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            "energy_ev": -2.0 - index / 10,
            "forces_ev_per_angstrom": [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
            "stress_kbar_vasp_3x3": [[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]],
        })
    return contract.build_canonical_dataset(
        label_records=labels,
        units={"energy": "eV", "length": "angstrom", "force": "eV/angstrom", "stress": "kbar-vasp-3x3"},
        calculation_type="static",
        source_attempt_identity={"project_id": project_id, "node_id": "label", "attempt": 1},
        structures_manifest={"schema_version": 1, "structures": structures},
        raw_outputs={},
    )


def test_canonical_source_accepts_external_structure_input_paths(tmp_path: Path) -> None:
    contract = _contract()
    absolute = str((tmp_path / "shared" / "A.vasp").resolve())
    for source_path in ("../../shared/A.vasp", absolute):
        sources = contract._sources(
            {
                "schema_version": 1,
                "structures": [{"id": "A", "path": source_path}],
            }
        )
        assert sources["A"]["path"] == source_path


def _convert(tmp_path: Path, canonical: dict, frameworks: str = "deepmd,m3gnet,chgnet,mace"):
    sys.path.insert(0, str(DFT_PLUGIN))
    try:
        converter = _load("dft_dataset_converter_test", DFT_PLUGIN / "dataset_convert.py")
    finally:
        sys.path.pop(0)
    canonical_path = tmp_path / "canonical.json"
    write_json(canonical_path, canonical)
    data_root, output = tmp_path / "datasets", tmp_path / "output"
    args = SimpleNamespace(
        canonical=str(canonical_path), data_root=str(data_root), output_dir=str(output),
        frameworks=frameworks, dataset_relative_path=None,
        split_strategy="deterministic", split_seed=17,
        train_fraction=.6, validation_fraction=.2, test_fraction=.2,
        result_name="dataset-assembly-result.json",
    )
    return converter, args, converter.run(args), data_root / canonical["dataset_id"], output


def test_scheduled_label_emits_canonical_without_manifest_stack(tmp_path: Path) -> None:
    lifecycle = ScheduledLifecycle(tmp_path, "static", 2)
    assert lifecycle.run()["changed"][0]["state"] == "OK"
    attempt = lifecycle.attempt
    result = json.loads((attempt / "dft-labeling-result.json").read_text())
    canonical = json.loads((attempt / "canonical-labeled-dataset.json").read_text())
    assert result["dataset_id"] == canonical["dataset_id"]
    assert len({record["record_id"] for record in canonical["records"]}) == 2
    assert not (attempt / "dataset-manifest.json").exists()
    assert not (attempt / "dft-source-result.json").exists()
    assert not (attempt / "split-manifest.json").exists()
    assert {item["name"] for item in result["artifacts"]} == {"labels-json", "canonical-labeled-dataset"}


def test_deterministic_split_is_minimal_stable_and_complete() -> None:
    contract, canonical = _contract(), _canonical(10)
    first = contract.build_split_manifest(canonical, strategy="deterministic", seed=8)
    second = contract.build_split_manifest(canonical, strategy="deterministic", seed=8)
    changed = contract.build_split_manifest(canonical, strategy="deterministic", seed=9)
    assert first == second and first["split_id"] != changed["split_id"]
    assert set(first) == {
        "dataset_id", "split_id", "strategy", "seed", "train_record_ids",
        "validation_record_ids", "test_record_ids", "counts",
    }
    ids = first["train_record_ids"] + first["validation_record_ids"] + first["test_record_ids"]
    assert len(ids) == len(set(ids)) == canonical["record_count"]
    assert not contract.validate_split_manifest(canonical, first)


def test_canonical_merge_preserves_source_order_and_per_record_dft_lineage() -> None:
    contract = _contract()
    old = _canonical(3, project_id="old-labels")
    query = _canonical(2, project_id="round-001", start=10)
    merged = contract.merge_canonical_datasets([old, query])

    assert merged["source_dataset_ids"] == [old["dataset_id"], query["dataset_id"]]
    assert merged["record_count"] == 5
    assert [record["record_id"] for record in merged["records"]] == [
        *(record["record_id"] for record in old["records"]),
        *(record["record_id"] for record in query["records"]),
    ]
    assert [record["source_dft_attempt"]["project_id"] for record in merged["records"]] == [
        "old-labels", "old-labels", "old-labels", "round-001", "round-001"
    ]
    assert not contract.validate_canonical_dataset(merged)

    with pytest.raises(contract.DatasetContractError, match="duplicate record_id"):
        contract.merge_canonical_datasets([old, old])


def test_group_aware_split_keeps_trajectory_frames_together() -> None:
    contract, canonical = _contract(), _canonical(12, groups=True)
    split = contract.build_split_manifest(
        canonical, strategy="group-aware", seed=23,
        fractions={"train": .5, "validation": .25, "test": .25},
    )
    membership = {record_id: name for name in ("train", "validation", "test") for record_id in split[f"{name}_record_ids"]}
    grouped = {}
    for record in canonical["records"]:
        grouped.setdefault(record["source_group_id"], set()).add(membership[record["record_id"]])
    assert all(len(partitions) == 1 for partitions in grouped.values())
    assert not contract.validate_split_manifest(canonical, split)


def test_sampling_provenance_survives_canonical_dataset() -> None:
    contract, canonical = _contract(), _canonical(3, sampling=True)
    source = canonical["records"][0]["source_structure"]
    assert source["source_sampling_method"] == "DIRECT"
    assert source["source_sampling_methods"] == ["DIRECT", "LASP_SSW"]
    assert source["source_records"][0]["source_id"] == "direct-000000"
    assert not contract.validate_canonical_dataset(canonical)

    source["source_records"][0]["source_path"] = "../escaped.vasp"
    assert "canonical record 0 source provenance is invalid" in contract.validate_canonical_dataset(
        canonical
    )


def test_four_views_share_exact_record_ids_and_scientific_conventions(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    dpdata = pytest.importorskip("dpdata")
    ase_io = pytest.importorskip("ase.io")
    _, _, result, root, output = _convert(tmp_path, _canonical(10))
    split = json.loads((root / "split.json").read_text())
    assert result["counts"] == {"train": 6, "validation": 2, "test": 2, "total": 10}
    assert (root / "canonical.json").is_file() and (root / "assembly-result.json").is_file()
    aliases = {"train": "train", "validation": "valid", "test": "test"}
    deep_index = json.loads((root / "deepmd" / "record-index.json").read_text())
    for split_name, stem in aliases.items():
        expected = split[f"{split_name}_record_ids"]
        for framework in ("m3gnet", "chgnet"):
            view = json.loads((root / framework / f"{stem}.json").read_text())
            assert [record["record_id"] for record in view["records"]] == expected
        mace = ase_io.read(root / "mace" / f"{stem}.extxyz", index=":")
        assert [atoms.info["mlipflow_record_id"] for atoms in mace] == expected
        assert [item["record_id"] for item in deep_index["partitions"][split_name]] == expected
        frame_count = 0
        for system in sorted((root / "deepmd" / stem).glob("system-*")):
            loaded = dpdata.LabeledSystem(str(system), fmt="deepmd/npy")
            frame_count += loaded.get_nframes()
            assert "virials" in loaded.data
        assert frame_count == len(expected)
    convention = result["units_and_conventions"]
    assert convention["energy"] == "total eV per configuration, unchanged"
    assert convention["forces"] == "eV/angstrom, unchanged"
    assert "no sign inversion" in convention["stress"]["deepmd"]
    assert "-VASP_kbar" in convention["stress"]["mace"]
    for framework in ("deepmd", "m3gnet", "chgnet", "mace"):
        reference = json.loads((output / f"{framework}-dataset-reference.json").read_text())
        assert reference["kind"] == "directory"
        assert reference["split_id"] == split["split_id"]
        assert reference["relative_path"] == f"{root.name}/{framework}"
    benchmark_path = output / "benchmark-test.json"
    benchmark = json.loads(benchmark_path.read_text())
    assert [sample["id"] for sample in benchmark["samples"]] == split["test_record_ids"]
    assert benchmark["energy_convention"] == "total"
    assert benchmark["stress_convention"] == "ase-voigt-xx-yy-zz-yz-xz-xy"
    assert benchmark["samples"][0]["references"]["stress"][0] == pytest.approx(
        -1.0 / 1602.176621
    )
    benchmark_reference = json.loads(
        (output / "benchmark-dataset-reference.json").read_text()
    )
    assert benchmark_reference["split_id"] == split["split_id"]
    assert benchmark_reference["relative_path"] == f"{root.name}/benchmark/test.json"
    fresh = _load(
        "dft_benchmark_dataset_reader",
        ROOT / "mlipflow" / "plugins" / "mlip_benchmark" / "fresh_benchmark.py",
    )
    _, loaded_samples = fresh._load_dataset(benchmark_path)
    assert [sample["id"] for sample in loaded_samples] == split["test_record_ids"]
    assert not list(tmp_path.rglob("*.tar"))


def test_converter_refuses_existing_dataset_when_canonical_records_differ(tmp_path: Path) -> None:
    converter, args, _, root, first_output = _convert(tmp_path, _canonical(6), "m3gnet,chgnet,mace")
    (root / "canonical.json").write_text("{}")
    second_output = tmp_path / "second-output"
    args.output_dir = str(second_output)
    with pytest.raises(converter.ConversionError, match="already exists"):
        converter.run(args)
    assert not second_output.exists()
    args.reuse_existing = True
    args.reuse_reference_dir = str(first_output)
    with pytest.raises(converter.ConversionError, match="canonical records differ"):
        converter.run(args)


def test_converter_recollects_verified_existing_dataset_without_overwrite(
    tmp_path: Path,
) -> None:
    converter, args, first, root, first_output = _convert(
        tmp_path, _canonical(6), "m3gnet,chgnet"
    )
    second_output = tmp_path / "second-output"
    args.output_dir = str(second_output)
    args.reuse_existing = True
    args.reuse_reference_dir = str(first_output)
    assert converter.run(args) == first
    assert root.is_dir()
    for name in (
        "m3gnet-dataset-reference.json",
        "chgnet-dataset-reference.json",
        "benchmark-dataset-reference.json",
    ):
        assert (second_output / name).read_bytes() == (first_output / name).read_bytes()


def test_adapter_plan_passes_explicit_split_parameters_and_fetches_no_bundle(tmp_path: Path) -> None:
    canonical = _canonical(10)
    write_json(tmp_path / "canonical.json", canonical)
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    context = {
        "project_root": str(tmp_path), "attempt_dir": str(attempt), "backend": "ssh-slurm",
        "inputs": {"canonical_dataset": "canonical.json"},
        "parameters": {
            "operation": "dataset-assemble", "frameworks": ["deepmd", "m3gnet", "chgnet", "mace"],
            "split_strategy": "group-aware" if all("source_group_id" in r for r in canonical["records"]) else "deterministic",
            "split_seed": 11, "split_fractions": {"train": .6, "validation": .2, "test": .2},
        },
        "resources": {"cpus": 2, "gpus": 0, "memory": "4G", "walltime": "00:10:00"},
    }
    adapter = _load("dft_dataset_adapter_plan", DFT_PLUGIN / "adapter.py").Adapter()
    plan = adapter.plan(context)
    assert plan["status"] == "READY", plan.get("diagnostics")
    staged = {item["remote_name"] for item in plan["scheduled_execution"]["staged_files"]}
    fetched = {item["remote_name"] for item in plan["scheduled_execution"]["fetch_outputs"]}
    assert staged == {"canonical.json", "dataset_convert.py", "dataset_contract.py"}
    assert "project.yaml" not in staged and not any(name.endswith(".tar") for name in fetched)
    variables = plan["scheduled_execution"]["template_variables"]
    assert variables["PLUGIN_SPLIT_SEED"] == "11"
    assert variables["PLUGIN_FRAMEWORKS"] == "deepmd,m3gnet,chgnet,mace"


def test_dataset_adapter_stages_and_converter_resolves_canonical_merge_manifest(
    tmp_path: Path,
) -> None:
    old = _canonical(3, project_id="old-labels")
    query = _canonical(3, project_id="round-001", start=10)
    source_dir = tmp_path / ".mlipflow" / "runs" / "labels" / "attempt-1"
    write_json(source_dir / "old.json", old)
    write_json(source_dir / "query.json", query)
    merge_manifest = {
        "schema_version": 1,
        "contract": "mlipflow/canonical-dataset-merge",
        "sources": [
            {"path": ".mlipflow/runs/labels/attempt-1/old.json"},
            {"path": ".mlipflow/runs/labels/attempt-1/query.json"},
        ],
    }
    merge_path = tmp_path / "merge.json"
    write_json(merge_path, merge_manifest)
    merged = _contract().merge_canonical_datasets([old, query])

    sys.path.insert(0, str(DFT_PLUGIN))
    try:
        converter = _load("dft_dataset_converter_merge", DFT_PLUGIN / "dataset_convert.py")
    finally:
        sys.path.pop(0)
    assert converter._canonical_input(merge_path) == merged

    attempt = tmp_path / "attempt"
    attempt.mkdir()
    context = {
        "project_root": str(tmp_path),
        "attempt_dir": str(attempt),
        "backend": "ssh-slurm",
        "inputs": {"canonical_dataset": "merge.json"},
        "parameters": {
            "operation": "dataset-assemble",
            "frameworks": ["chgnet"],
            "split_strategy": "deterministic",
            "split_seed": 23,
            "split_fractions": {"train": 2 / 3, "validation": 1 / 6, "test": 1 / 6},
        },
        "resources": {"cpus": 2, "gpus": 0, "memory": "4G", "walltime": "00:10:00"},
    }
    adapter = _load("dft_dataset_adapter_merge", DFT_PLUGIN / "adapter.py").Adapter()
    assert adapter.validate(context) == []
    plan = adapter.plan(context)
    assert plan["status"] == "READY", plan.get("diagnostics")
    assert plan["approval_summary"]["dataset_id"] == merged["dataset_id"]
    assert plan["approval_summary"]["canonical_source_count"] == 2
    staged_names = {
        item["remote_name"] for item in plan["scheduled_execution"]["staged_files"]
    }
    assert staged_names == {
        "canonical.json",
        ".mlipflow/runs/labels/attempt-1/old.json",
        ".mlipflow/runs/labels/attempt-1/query.json",
        "dataset_convert.py",
        "dataset_contract.py",
    }
    assert all(_safe_remote_relative(name) for name in staged_names)
    assert not _safe_remote_relative(".ssh/id_ed25519")


def test_dataset_adapter_accepts_absolute_project_scoped_canonical_path(tmp_path: Path) -> None:
    canonical_path = tmp_path / "canonical.json"
    write_json(canonical_path, _canonical(4))
    context = {
        "project_root": str(tmp_path),
        "attempt_dir": str(tmp_path / "attempt"),
        "backend": "ssh-slurm",
        "inputs": {"canonical_dataset": str(canonical_path.resolve())},
        "parameters": {
            "operation": "dataset-assemble",
            "frameworks": ["m3gnet", "chgnet"],
            "split_strategy": "deterministic",
            "split_seed": 23,
            "split_fractions": {"train": .8, "validation": .1, "test": .1},
        },
        "resources": {"cpus": 2, "gpus": 0, "memory": "4G", "walltime": "00:10:00"},
    }
    adapter = _load("dft_dataset_adapter_absolute", DFT_PLUGIN / "adapter.py").Adapter()
    assert adapter.validate(context) == []
    assert adapter.plan(context)["status"] == "READY"


def test_adapter_checks_and_collects_only_split_result_and_references(tmp_path: Path) -> None:
    canonical = _canonical(10)
    converter, args, _, _, output = _convert(tmp_path, canonical)
    del converter, args
    context = {
        "project_root": str(tmp_path), "attempt_dir": str(output), "backend": "ssh-slurm",
        "inputs": {"canonical_dataset": "canonical.json"},
        "parameters": {
            "operation": "dataset-assemble", "frameworks": ["deepmd", "m3gnet", "chgnet", "mace"],
            "split_strategy": "deterministic", "split_seed": 17,
            "split_fractions": {"train": .6, "validation": .2, "test": .2},
        },
        "resources": {"cpus": 2, "gpus": 0, "memory": "4G", "walltime": "00:10:00"},
    }
    # _convert wrote the source canonical before publishing the remote view.
    adapter = _load("dft_dataset_adapter_collect", DFT_PLUGIN / "adapter.py").Adapter()
    assert adapter.check(context)["status"] == "OK"
    collected = adapter.collect(context)
    assert collected["status"] == "OK"
    roles = {item["role"] for item in collected["artifacts"]}
    assert "split-manifest" in roles and "dataset-assembly-result" in roles
    assert all("bundle" not in role and "manifest" not in role.replace("split-manifest", "") for role in roles if "dataset-reference" not in role)
    assert collected["metrics"]["test"] == 2


def test_dataset_template_consumes_only_explicit_converter_arguments() -> None:
    run_template = (ROOT / "examples/training_all_models/cluster/dft-dataset.run.sh.example").read_text()
    assert "project.yaml" not in run_template and "--canonical " in run_template
    submit = """#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={{CPUS}}
#SBATCH --gpus={{GPUS}}
#SBATCH --mem={{MEMORY}}
#SBATCH --time={{WALLTIME}}
#SBATCH --output={{LOG_DIR}}/out
cd {{RUN_DIR}}
bash {{RUN_DIR}}/run.sh
"""
    class Library:
        def read_template(self, _root, relative):
            content = run_template if relative == "dft-dataset/run.sh" else submit
            return {"content": content}
    variables = {
        "PLUGIN_FRAMEWORKS": "deepmd,m3gnet,chgnet,mace", "PLUGIN_DATASET_ID": "dft-abc",
        "PLUGIN_SPLIT_STRATEGY": "deterministic", "PLUGIN_SPLIT_SEED": "4",
        "PLUGIN_TRAIN_FRACTION": "0.8", "PLUGIN_VALIDATION_FRACTION": "0.1",
        "PLUGIN_TEST_FRACTION": "0.1", "PLUGIN_RESULT_NAME": "dataset-assembly-result.json",
    }
    plan = resolve_hpc_execution_plan(
        profile=ClusterProfile(name="a", backend="ssh-slurm", ssh_profile="a", remote_template_root="/templates", work_root="/work"),
        project_id="p", node_id="assemble", attempt=1,
        resources_value={"cpus": 2, "gpus": 0, "memory": "4G", "walltime": "00:10:00"},
        scheduled_execution={"template_family": "dft-dataset", "execution_model": "single-python", "template_variables": variables},
        library=Library(),
    )
    assert '--split-seed "4"' in plan["rendered_scripts"]["run.sh"]


def test_missing_dpdata_blocks_only_deepmd_conversion(tmp_path: Path, monkeypatch) -> None:
    sys.path.insert(0, str(DFT_PLUGIN))
    try:
        converter = _load("dft_dataset_converter_missing", DFT_PLUGIN / "dataset_convert.py")
    finally:
        sys.path.pop(0)
    real_import = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name == "dpdata":
            raise ModuleNotFoundError("blocked")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked)
    canonical = _canonical(6)
    split = _contract().build_split_manifest(canonical, seed=1)
    with pytest.raises(converter.DependencyBlocked, match="dpdata and NumPy"):
        converter._deepmd(tmp_path, canonical, split)


def test_training_runners_consume_predefined_split_directories(tmp_path: Path) -> None:
    _, _, _, root, _ = _convert(tmp_path, _canonical(10), "m3gnet,chgnet,mace")
    sys.path.insert(0, str(ROOT / "mlipflow" / "plugins" / "mlip_training"))
    try:
        m3gnet = _load("m3gnet_predefined_test", ROOT / "mlipflow/plugins/mlip_training/mlip_m3gnet.py")
        chgnet = _load("chgnet_predefined_test", ROOT / "mlipflow/plugins/mlip_training/mlip_chgnet.py")
        mace = _load("mace_predefined_test", ROOT / "mlipflow/plugins/mlip_training/mlip_mace.py")
    finally:
        sys.path.pop(0)
    m3_contract = m3gnet._dataset_contract({})
    records, train, valid, test, evidence = m3gnet._records_and_split(
        root / "m3gnet", m3_contract, True, {}, 999
    )
    assert (len(records), len(train), len(valid), len(test)) == (10, 6, 2, 2)
    assert evidence["source"] == "predefined"
    chg_contract = chgnet._dataset_contract({})
    records, train, valid, test, chg_evidence = chgnet._records_and_split(
        root / "chgnet", chg_contract, "efs", {}, 999
    )
    assert (len(records), len(train), len(valid), len(test)) == (10, 6, 2, 2)
    assert chg_evidence["split_id"] == evidence["split_id"]
    args = SimpleNamespace(
        output="model.model", seed=1, device="cpu", precision="float32",
        operation="train", foundation_model=None,
    )
    _, argv, auxiliary = mace._argv(args, {"mace": {"options": {"max_num_epochs": 1}}}, root / "mace", tmp_path / "work")
    assert auxiliary is None
    assert argv[argv.index("--train_file") + 1].endswith("train.extxyz")
    assert argv[argv.index("--valid_file") + 1].endswith("valid.extxyz")
    assert argv[argv.index("--test_file") + 1].endswith("test.extxyz")
