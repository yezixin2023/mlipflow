from __future__ import annotations

from tests.helpers import load_module
import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "mlipflow" / "plugins" / "mlip_training"


def _load(name: str, path: Path):
    module = load_module(path, name)
    return module


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _context(tmp_path: Path, framework: str, operation: str) -> dict:
    _write_json(
        tmp_path / "project.yaml",
        {"schema_version": 1, "project": {"id": "matrix"}, "workflow": {"nodes": []}},
    )
    config = tmp_path / "inputs" / f"{framework}.json"
    _write_json(config, {"framework": framework})
    dataset_kind = "directory" if framework == "deepmd" else "file"
    dataset = tmp_path / "inputs" / f"{framework}-dataset.json"
    _write_json(
        dataset,
        {
            "schema_version": 1,
            "dataset_id": f"{framework}-dataset-v1",
            "relative_path": f"{framework}/dataset-v1",
            "kind": dataset_kind,
        },
    )
    inputs = {
        "training_config": str(config.relative_to(tmp_path)),
        "dataset_reference": str(dataset.relative_to(tmp_path)),
    }
    parameters = {
        "framework": framework,
        "operation": operation,
        "seed": 7,
        "device": "cpu",
        "precision": "float32" if framework == "chgnet" else "float64",
        "result_manifest": f"{framework}-{operation}-result.json",
    }
    if operation == "finetune":
        foundation_kind = "directory" if framework == "m3gnet" else "file"
        foundation = tmp_path / "inputs" / f"{framework}-foundation.json"
        _write_json(
            foundation,
            {
                "schema_version": 1,
                "model_id": f"{framework}-foundation-v1",
                "relative_path": f"{framework}/foundation-v1",
                "kind": foundation_kind,
            },
        )
        inputs["foundation_model_reference"] = str(foundation.relative_to(tmp_path))
    return {
        "project_root": str(tmp_path),
        "attempt_dir": str(tmp_path / ".mlipflow" / "runs" / "node" / "attempt-1"),
        "backend": "ssh-slurm",
        "inputs": inputs,
        "parameters": parameters,
        "resources": {"cpus": 8, "gpus": 0, "memory": "32G", "walltime": "02:00:00"},
    }


@pytest.mark.parametrize("framework", ["deepmd", "m3gnet", "chgnet", "mace"])
@pytest.mark.parametrize("operation", ["train", "finetune"])
def test_generic_scheduler_matrix_is_ready(
    tmp_path: Path, framework: str, operation: str
) -> None:
    module = _load("mlip_training_cluster_adapter", PLUGIN / "adapter.py")
    plan = module.Adapter().plan(_context(tmp_path, framework, operation))
    assert plan["status"] == "READY", plan.get("diagnostics")
    assert plan["scheduler_contract"] == "bundled-mlip-v1"
    calculation = plan["training_calculation"]
    assert calculation["framework"] == framework
    assert calculation["operation"] == operation
    assert calculation["seed"] == 7
    assert calculation["dataset"]["relative_path"] == f"{framework}/dataset-v1"
    scheduled = plan["scheduled_execution"]
    assert scheduled["schema_version"] == 3
    assert scheduled["execution_model"] == "single-python"
    assert scheduled["template_family"] == f"mlip-{framework}"
    staged = {item["remote_name"] for item in scheduled["staged_files"]}
    assert {"project.yaml", "training-config.json", "dataset-reference.json"} <= staged
    assert ("foundation-model-reference.json" in staged) is (operation == "finetune")
    required = {
        item["remote_name"]
        for item in scheduled["fetch_outputs"]
        if item["required"]
    }
    assert {"training-result.json", "model-artifact", "cluster-run-report.json"} <= required


def test_generic_scheduler_stages_the_selected_project_file(tmp_path: Path) -> None:
    module = _load("mlip_training_selected_project", PLUGIN / "adapter.py")
    context = _context(tmp_path, "chgnet", "finetune")
    selected = tmp_path / "continuation.project.json"
    _write_json(
        selected,
        {
            "schema_version": 1,
            "project": {"id": "continuation"},
            "workflow": {"nodes": []},
        },
    )
    context["project_path"] = str(selected)

    plan = module.Adapter().plan(context)

    staged = {
        item["remote_name"]: item["source"]
        for item in plan["scheduled_execution"]["staged_files"]
    }
    assert staged["project.yaml"] == str(selected)


def test_cluster_runner_records_resolved_dataset_path(tmp_path: Path, monkeypatch) -> None:
    cluster = _load("mlip_training_cluster_handoff", PLUGIN / "training_cluster.py")
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    data_root = tmp_path / "data"
    input_dir.mkdir()
    dataset = data_root / "chgnet" / "dataset.json"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"dataset")
    _write_json(input_dir / "training-config.json", {"framework": "chgnet"})
    _write_json(
        input_dir / "dataset-reference.json",
        {
            "schema_version": 1,
            "dataset_id": "chgnet-upstream-v1",
            "relative_path": "chgnet/dataset.json",
            "kind": "file",
        },
    )
    project = tmp_path / "project.json"
    _write_json(
        project,
        {
            "workflow": {
                "nodes": [
                    {
                        "id": "train",
                        "parameters": {
                            "framework": "chgnet",
                            "operation": "train",
                            "seed": 23,
                            "device": "cpu",
                            "precision": "float32",
                        },
                    }
                ]
            }
        },
    )

    def fake_training(argv: list[str]) -> int:
        model = Path(argv[argv.index("--output") + 1])
        result = Path(argv[argv.index("--result-manifest") + 1])
        model.write_bytes(b"model")
        _write_json(result, {"framework_version": "test"})
        return 0

    monkeypatch.setitem(sys.modules, "training_wrapper", SimpleNamespace(main=fake_training))
    args = cluster.build_parser().parse_args(
        [
            "run",
            "--project",
            str(project),
            "--node-id",
            "train",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--data-root",
            str(data_root),
        ]
    )
    assert cluster.run(args) == 0
    report = json.loads((output_dir / "cluster-run-report.json").read_text())
    assert report["dataset"] == {
        "id": "chgnet-upstream-v1",
        "relative_path": "chgnet/dataset.json",
        "kind": "file",
        "resolved_path": str(dataset),
    }


def test_chgnet_scheduled_float64_is_rejected(tmp_path: Path) -> None:
    module = _load("mlip_training_cluster_adapter_bad", PLUGIN / "adapter.py")
    context = _context(tmp_path, "chgnet", "train")
    context["parameters"]["precision"] = "float64"
    plan = module.Adapter().plan(context)
    assert plan["status"] == "BLOCKED"
    assert any(item["code"] == "training.chgnet_precision" for item in plan["diagnostics"])


def test_cluster_model_publication_records_destination_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    cluster = _load("mlip_training_cluster_publish", PLUGIN / "training_cluster.py")
    model_root = tmp_path / "models"
    model_root.mkdir()
    checkpoint = tmp_path / "chgnet.pt"
    checkpoint.write_bytes(b"checkpoint")
    chgnet = cluster._publish_model(
        checkpoint, model_root, "chgnet", "chgnet-v1", "chgnet/chgnet-v1.pt"
    )
    assert chgnet == {
        "schema_version": 1,
        "model_id": "chgnet-v1",
        "framework": "chgnet",
        "relative_path": "chgnet/chgnet-v1.pt",
        "kind": "file",
    }
    with pytest.raises(ValueError, match="already exists"):
        cluster._publish_model(
            checkpoint, model_root, "chgnet", "chgnet-v1", "chgnet/chgnet-v1.pt"
        )

    archive = tmp_path / "m3gnet.tar.gz"
    payload = b"model-data"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo("model/model.json")
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))
    m3gnet = cluster._publish_model(
        archive, model_root, "m3gnet", "m3gnet-v1", "m3gnet/m3gnet-v1"
    )
    assert m3gnet["kind"] == "directory"
    assert (model_root / "m3gnet/m3gnet-v1/model.json").read_bytes() == payload


def test_cluster_runner_accepts_separate_foundation_and_publication_roots() -> None:
    cluster = _load("mlip_training_cluster_roots", PLUGIN / "training_cluster.py")
    args = cluster.build_parser().parse_args(
        [
            "run",
            "--project",
            "project.yaml",
            "--node-id",
            "finetune",
            "--input-dir",
            "input",
            "--output-dir",
            "output",
            "--data-root",
            "/site/data",
            "--model-root",
            "/site/models",
            "--foundation-model-root",
            "/site/foundations",
        ]
    )
    assert args.model_root == "/site/models"
    assert args.foundation_model_root == "/site/foundations"

@pytest.mark.parametrize('content', [None, '{invalid json', '[]'])
def test_bad_deepmd_reference_is_reported_without_legacy_fallback(tmp_path, content):
    from mlipflow.plugins.mlip_training.adapter import Adapter
    context = _context(tmp_path, 'deepmd', 'train')
    reference = tmp_path / context['inputs']['dataset_reference']
    if content is None:
        reference.unlink()
    else:
        reference.write_text(content)
    diagnostics = Adapter().validate(context)
    assert any(item['level'] == 'error' and 'dataset' in item['code']
               for item in diagnostics), diagnostics
    plan = Adapter().plan(context)
    assert plan['status'] == 'BLOCKED'
    assert any(str(reference) in item['message'] for item in plan['diagnostics'])


def test_id_only_dataset_uses_the_same_bundled_training_contract(tmp_path):
    from mlipflow.plugins.mlip_training.adapter import Adapter
    context = _context(tmp_path, 'deepmd', 'train')
    _write_json(tmp_path / context['inputs']['dataset_reference'],
                {'schema_version': 1, 'dataset_id': 'existing-data'})
    plan = Adapter().plan(context)
    assert plan['status'] == 'READY', plan['diagnostics']
    assert plan['scheduler_contract'] == 'bundled-mlip-v1'
    assert plan['training_calculation']['dataset'] == {
        'id': 'existing-data', 'relative_path': 'existing-data', 'kind': 'directory'
    }
