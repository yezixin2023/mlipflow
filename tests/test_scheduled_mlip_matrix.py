from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "mlip-training"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _context(tmp_path: Path, framework: str, operation: str) -> dict:
    _write_json(
        tmp_path / "project.yaml",
        {
            "schema_version": 1,
            "project": {"id": "matrix"},
            "workflow": {"nodes": []},
        },
    )
    config = tmp_path / "inputs" / f"{framework}.json"
    _write_json(config, {"framework": framework})
    dataset_fp = "sha256:" + "1" * 64
    dataset_kind = "directory" if framework == "deepmd" else "file"
    dataset = tmp_path / "inputs" / f"{framework}-dataset.json"
    _write_json(
        dataset,
        {
            "schema_version": 1,
            "dataset_id": f"{framework}-dataset-v1",
            "relative_path": f"{framework}/dataset-v1",
            "kind": dataset_kind,
            "fingerprint": dataset_fp,
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
        "dataset_fingerprint": dataset_fp,
        "config_fingerprint": _sha(config),
        "result_manifest": f"{framework}-{operation}-result.json",
    }
    if operation == "finetune":
        foundation_fp = "sha256:" + "2" * 64
        foundation_kind = "directory" if framework == "m3gnet" else "file"
        foundation = tmp_path / "inputs" / f"{framework}-foundation.json"
        _write_json(
            foundation,
            {
                "schema_version": 1,
                "model_id": f"{framework}-foundation-v1",
                "relative_path": f"{framework}/foundation-v1",
                "kind": foundation_kind,
                "fingerprint": foundation_fp,
            },
        )
        inputs["foundation_model_reference"] = str(foundation.relative_to(tmp_path))
        parameters["foundation_model_fingerprint"] = foundation_fp
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
def test_generic_scheduler_matrix_is_ready(tmp_path: Path, framework: str, operation: str) -> None:
    module = _load("mlip_training_cluster_adapter", PLUGIN / "adapter_cluster.py")
    plan = module.Adapter().plan(_context(tmp_path, framework, operation))
    assert plan["status"] == "READY", plan.get("diagnostics")
    assert plan["scheduler_contract"] == "bundled-mlip-v1"
    assert plan["framework"] == framework
    assert plan["operation"] == operation
    scheduled = plan["scheduled_execution"]
    assert scheduled["schema_version"] == 2
    assert scheduled["template_family"] == f"mlip-{framework}"
    staged = {item["remote_name"] for item in scheduled["staged_files"]}
    assert {
        "project.yaml",
        "training-config.json",
        "dataset-reference.json",
        "training_cluster.py",
        "training_wrapper.py",
        "mlip_common.py",
        "mlip_deepmd.py",
        "mlip_m3gnet.py",
        "mlip_chgnet.py",
        "mlip_mace.py",
    } <= staged
    assert ("foundation-model-reference.json" in staged) is (operation == "finetune")
    outputs = {item["remote_name"]: item for item in scheduled["fetch_outputs"]}
    assert outputs["training-result.json"]["required"] is True
    assert outputs["model-artifact"]["required"] is True
    assert outputs["cluster-run-report.json"]["required"] is True


def test_chgnet_scheduled_float64_is_rejected(tmp_path: Path) -> None:
    module = _load("mlip_training_cluster_adapter_bad", PLUGIN / "adapter_cluster.py")
    context = _context(tmp_path, "chgnet", "train")
    context["parameters"]["precision"] = "float64"
    plan = module.Adapter().plan(context)
    assert plan["status"] == "BLOCKED"
    assert any(item["code"] == "training.chgnet_precision" for item in plan["diagnostics"])


def test_cluster_fingerprint_is_stable_for_files_and_trees(tmp_path: Path) -> None:
    runner = _load("mlip_training_cluster_runner", PLUGIN / "training_cluster.py")
    file_path = tmp_path / "data.extxyz"
    file_path.write_bytes(b"frame-1\n")
    assert runner.fingerprint(file_path) == _sha(file_path)
    tree = tmp_path / "deepmd-data"
    (tree / "set.000").mkdir(parents=True)
    (tree / "type.raw").write_text("0\n", encoding="utf-8")
    (tree / "set.000" / "coord.npy").write_bytes(b"coords")
    first = runner.fingerprint(tree)
    second = runner.fingerprint(tree)
    assert first == second
    assert first.startswith("sha256:") and len(first) == 71
