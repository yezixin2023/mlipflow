from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlipflow.science.artifact_identity import fingerprint_path

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


def _completed_finetune_context(
    tmp_path: Path, module, *, publish: bool = False
) -> tuple[dict, Path]:
    context = _context(tmp_path, "chgnet", "finetune")
    config = tmp_path / context["inputs"]["training_config"]
    _write_json(
        config,
        {
            "framework": "chgnet",
            "chgnet": {"trainer": {"epochs": 1}},
        },
    )
    context["parameters"]["config_fingerprint"] = _sha(config)
    if publish:
        context["parameters"].update(
            {
                "publish_model_id": "chgnet-finetuned-v1",
                "publish_model_relative_path": "chgnet/chgnet-finetuned-v1.pt",
            }
        )
    plan = module.Adapter().plan(context)
    assert plan["status"] == "READY"
    identity = plan["training_identity"]
    attempt = Path(context["attempt_dir"])
    attempt.mkdir(parents=True)
    model = attempt / "model-artifact"
    model.write_bytes(b"trained-model")
    result_name = context["parameters"]["result_manifest"]
    _write_json(
        attempt / result_name,
        {
            "schema_version": 1,
            "plugin_id": "mlip-training",
            "status": "OK",
            "framework": identity["framework"],
            "framework_version": "0.4.0",
            "operation": identity["operation"],
            "seed": identity["seed"],
            "device": identity["device"],
            "precision": identity["precision"],
            "dataset_fingerprint": identity["dataset"]["fingerprint"],
            "config_fingerprint": identity["config_fingerprint"],
            "foundation_model_fingerprint": identity["foundation_model"]["fingerprint"],
            "metrics": {"loss": 0.125},
            "provenance": {
                "completion": {
                    "requested_epochs": 1,
                    "completed_epochs": 1,
                    "normal_completion": True,
                    "all_recorded_metrics_finite": True,
                    "native_model_reload": "OK",
                },
                "environment": {
                    "compute_node": "cpu1",
                    "python_version": "3.11.15",
                    "chgnet_version": "0.4.2",
                    "torch_version": "2.13.0",
                    "torch_cuda_version": "13.0",
                    "gpu_model": "unavailable",
                },
                "freeze_modules": [],
                "split": {
                    "seed": identity["seed"],
                    "train_indices_sha256": "sha256:" + "3" * 64,
                    "val_indices_sha256": "sha256:" + "4" * 64,
                    "test_indices_sha256": None,
                },
            },
            "model_artifact": {
                "path": "model-artifact",
                "media_type": "application/x-pytorch",
                "sha256": _sha(model),
                "size_bytes": model.stat().st_size,
            },
        },
    )
    report_path = attempt / "cluster-run-report.json"
    report = {
            "schema_version": 1,
            "status": "OK",
            "return_code": 0,
            "framework": identity["framework"],
            "operation": identity["operation"],
            "config_fingerprint": identity["config_fingerprint"],
            "dataset": {
                **identity["dataset"],
                "observed_fingerprint": identity["dataset"]["fingerprint"],
            },
            "foundation_model": {
                **identity["foundation_model"],
                "observed_fingerprint": identity["foundation_model"]["fingerprint"],
            },
        }
    if publish:
        published = {
            "schema_version": 1,
            **identity["publish_model"],
            "fingerprint": "sha256:" + "8" * 64,
        }
        report["published_model"] = published
        _write_json(attempt / "model-reference.json", published)
    _write_json(report_path, report)
    context["execution"] = {"plan": plan}
    return context, report_path


def _completed_mace_finetune_context(tmp_path: Path, module) -> tuple[dict, Path]:
    context = _context(tmp_path, "mace", "finetune")
    config = tmp_path / context["inputs"]["training_config"]
    _write_json(
        config,
        {
            "framework": "mace",
            "mace": {"options": {"max_num_epochs": 1}},
        },
    )
    context["parameters"]["config_fingerprint"] = _sha(config)
    plan = module.Adapter().plan(context)
    assert plan["status"] == "READY"
    identity = plan["training_identity"]
    attempt = Path(context["attempt_dir"])
    attempt.mkdir(parents=True)
    model = attempt / "model-artifact"
    model.write_bytes(b"mace-model")
    result_path = attempt / context["parameters"]["result_manifest"]
    _write_json(
        result_path,
        {
            "schema_version": 1,
            "plugin_id": "mlip-training",
            "status": "OK",
            "framework": "mace",
            "framework_version": "0.3.12",
            "operation": "finetune",
            "seed": identity["seed"],
            "device": identity["device"],
            "precision": identity["precision"],
            "dataset_fingerprint": identity["dataset"]["fingerprint"],
            "config_fingerprint": identity["config_fingerprint"],
            "foundation_model_fingerprint": identity["foundation_model"]["fingerprint"],
            "metrics": {"requested_epochs": 1.0, "completed_epochs": 1.0},
            "provenance": {
                "completion": {
                    "requested_epochs": 1,
                    "completed_epochs": 1,
                    "normal_completion": True,
                    "all_recorded_metrics_finite": True,
                    "native_model_reload": "OK",
                },
                "environment": {
                    "compute_node": "gpu3",
                    "python_version": "3.12.2",
                    "mace_source_version": "0.3.12",
                    "mace_dist_version": "0.3.12",
                    "torch_version": "2.5.1",
                    "torch_cuda_version": "11.8",
                    "gpu_model": "NVIDIA A800 80GB PCIe",
                },
            },
            "model_artifact": {
                "path": "model-artifact",
                "media_type": "application/x-pytorch",
                "sha256": _sha(model),
                "size_bytes": model.stat().st_size,
            },
        },
    )
    _write_json(
        attempt / "cluster-run-report.json",
        {
            "schema_version": 1,
            "status": "OK",
            "return_code": 0,
            "framework": "mace",
            "operation": "finetune",
            "config_fingerprint": identity["config_fingerprint"],
            "dataset": {
                **identity["dataset"],
                "observed_fingerprint": identity["dataset"]["fingerprint"],
            },
            "foundation_model": {
                **identity["foundation_model"],
                "observed_fingerprint": identity["foundation_model"]["fingerprint"],
            },
        },
    )
    context["execution"] = {"plan": plan}
    return context, result_path


def _completed_m3gnet_context(tmp_path: Path, module, operation: str) -> tuple[dict, Path]:
    context = _context(tmp_path, "m3gnet", operation)
    config = tmp_path / context["inputs"]["training_config"]
    _write_json(
        config,
        {
            "framework": "m3gnet",
            "m3gnet": {"trainer": {"max_epochs": 1}},
        },
    )
    context["parameters"]["config_fingerprint"] = _sha(config)
    plan = module.Adapter().plan(context)
    assert plan["status"] == "READY"
    identity = plan["training_identity"]
    attempt = Path(context["attempt_dir"])
    attempt.mkdir(parents=True)
    model = attempt / "model-artifact"
    model.write_bytes(b"m3gnet-model")
    result_path = attempt / context["parameters"]["result_manifest"]
    result = {
        "schema_version": 1,
        "plugin_id": "mlip-training",
        "status": "OK",
        "framework": "m3gnet",
        "framework_version": "1.1.3",
        "operation": operation,
        "seed": identity["seed"],
        "device": identity["device"],
        "precision": identity["precision"],
        "dataset_fingerprint": identity["dataset"]["fingerprint"],
        "config_fingerprint": identity["config_fingerprint"],
        "metrics": {"completed_epochs": 1.0},
        "provenance": {
            "completion": {
                "requested_epochs": 1,
                "completed_epochs": 1,
                "normal_completion": True,
                "all_recorded_metrics_finite": True,
                "native_model_reload": "OK",
            },
            "environment": {
                "compute_node": "cpu1",
                "python_version": "3.11.8",
                "matgl_version": "1.1.3",
                "lightning_version": "2.3.0",
                "graph_backend": "dgl",
                "graph_library_version": "1.1.3",
                "torch_version": "2.5.0",
                "torch_cuda_version": "12.4",
                "gpu_model": "unavailable",
            },
            "foundation_element_refs": operation == "finetune",
            "split": {
                "seed": identity["seed"],
                "include_test": True,
                "train_indices_sha256": "sha256:" + "3" * 64,
                "val_indices_sha256": "sha256:" + "4" * 64,
                "test_indices_sha256": "sha256:" + "5" * 64,
            },
        },
        "model_artifact": {
            "path": "model-artifact",
            "media_type": "application/gzip",
            "sha256": _sha(model),
            "size_bytes": model.stat().st_size,
        },
    }
    if operation == "finetune":
        result["foundation_model_fingerprint"] = identity["foundation_model"]["fingerprint"]
    _write_json(result_path, result)
    report = {
        "schema_version": 1,
        "status": "OK",
        "return_code": 0,
        "framework": "m3gnet",
        "operation": operation,
        "config_fingerprint": identity["config_fingerprint"],
        "dataset": {
            **identity["dataset"],
            "observed_fingerprint": identity["dataset"]["fingerprint"],
        },
    }
    if operation == "finetune":
        report["foundation_model"] = {
            **identity["foundation_model"],
            "observed_fingerprint": identity["foundation_model"]["fingerprint"],
        }
    _write_json(attempt / "cluster-run-report.json", report)
    context["execution"] = {"plan": plan}
    return context, result_path


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
    assert scheduled["schema_version"] == 3
    assert scheduled["execution_model"] == "single-python"
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
        "model_runtime.py",
    } <= staged
    assert ("foundation-model-reference.json" in staged) is (operation == "finetune")
    outputs = {item["remote_name"]: item for item in scheduled["fetch_outputs"]}
    assert outputs["training-result.json"]["required"] is True
    assert outputs["model-artifact"]["required"] is True
    assert outputs["cluster-run-report.json"]["required"] is True


def test_upstream_dataset_reference_supplies_undeclared_fingerprint(tmp_path: Path) -> None:
    module = _load("mlip_training_cluster_adapter_handoff", PLUGIN / "adapter_cluster.py")
    context = _context(tmp_path, "m3gnet", "finetune")
    expected = context["parameters"].pop("dataset_fingerprint")

    plan = module.Adapter().plan(context)

    assert plan["status"] == "READY", plan.get("diagnostics")
    assert plan["training_identity"]["dataset"]["fingerprint"] == expected
    assert plan["approval_summary"]["dataset_fingerprint"] == expected


def test_cluster_runner_uses_bound_dataset_fingerprint_when_parameter_is_absent(
    tmp_path: Path, monkeypatch
) -> None:
    cluster = _load("mlip_training_cluster_handoff", PLUGIN / "training_cluster.py")
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    data_root = tmp_path / "data"
    input_dir.mkdir()
    dataset = data_root / "chgnet" / "dataset.json"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"approved-dataset")
    config = input_dir / "training-config.json"
    _write_json(config, {"framework": "chgnet"})
    _write_json(
        input_dir / "dataset-reference.json",
        {
            "schema_version": 1,
            "dataset_id": "chgnet-upstream-v1",
            "relative_path": "chgnet/dataset.json",
            "kind": "file",
            "fingerprint": cluster.fingerprint(dataset),
        },
    )
    project = tmp_path / "project.json"
    _write_json(
        project,
        {
            "workflow": {
                "nodes": [
                    {
                        "id": "finetune",
                        "parameters": {
                            "framework": "chgnet",
                            "operation": "train",
                            "seed": 23,
                            "device": "cpu",
                            "precision": "float32",
                            "config_fingerprint": _sha(config),
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
            "finetune",
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
    assert report["dataset"]["observed_fingerprint"] == cluster.fingerprint(dataset)


def test_chgnet_scheduled_float64_is_rejected(tmp_path: Path) -> None:
    module = _load("mlip_training_cluster_adapter_bad", PLUGIN / "adapter_cluster.py")
    context = _context(tmp_path, "chgnet", "train")
    context["parameters"]["precision"] = "float64"
    plan = module.Adapter().plan(context)
    assert plan["status"] == "BLOCKED"
    assert any(item["code"] == "training.chgnet_precision" for item in plan["diagnostics"])


def test_finetune_cluster_foundation_identity_is_accepted(tmp_path: Path) -> None:
    module = _load("mlip_training_cluster_adapter_foundation_ok", PLUGIN / "adapter_cluster.py")
    context, _ = _completed_finetune_context(tmp_path, module)

    result = module.Adapter().check(context)

    assert result["status"] == "OK", result.get("diagnostics")


def test_published_model_reference_is_fetched_checked_and_collected(tmp_path: Path) -> None:
    module = _load("mlip_training_cluster_adapter_publish", PLUGIN / "adapter_cluster.py")
    context, _ = _completed_finetune_context(tmp_path, module, publish=True)
    plan = context["execution"]["plan"]
    outputs = {
        item["remote_name"]: item for item in plan["scheduled_execution"]["fetch_outputs"]
    }
    assert plan["scheduled_execution"]["template_family"] == "mlip-chgnet-publish"
    assert outputs["model-reference.json"]["required"] is True

    collected = module.Adapter().collect(context)

    assert collected["status"] == "OK", collected.get("diagnostics")
    assert "model-reference" in {item["role"] for item in collected["artifacts"]}


def test_cluster_model_publication_handles_file_and_m3gnet_archive(tmp_path: Path) -> None:
    cluster = _load("mlip_training_cluster_publish", PLUGIN / "training_cluster.py")
    model_root = tmp_path / "models"
    model_root.mkdir()
    checkpoint = tmp_path / "chgnet.pt"
    checkpoint.write_bytes(b"checkpoint")
    chgnet = cluster._publish_model(
        checkpoint,
        model_root,
        "chgnet",
        "chgnet-v1",
        "chgnet/chgnet-v1.pt",
    )
    assert chgnet["kind"] == "file"
    assert chgnet["fingerprint"] == _sha(model_root / "chgnet/chgnet-v1.pt")
    with pytest.raises(ValueError, match="already exists"):
        cluster._publish_model(
            checkpoint,
            model_root,
            "chgnet",
            "chgnet-v1",
            "chgnet/chgnet-v1.pt",
        )

    archive = tmp_path / "m3gnet.tar.gz"
    payload = b"model-data"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo("model/model.json")
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))
    m3gnet = cluster._publish_model(
        archive,
        model_root,
        "m3gnet",
        "m3gnet-v1",
        "m3gnet/m3gnet-v1",
    )
    assert m3gnet["kind"] == "directory"
    assert (model_root / "m3gnet/m3gnet-v1/model.json").read_bytes() == payload
    assert m3gnet["fingerprint"] == cluster.fingerprint(
        model_root / "m3gnet/m3gnet-v1"
    )


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "other-foundation"),
        ("observed_fingerprint", "sha256:" + "3" * 64),
    ],
)
def test_finetune_cluster_foundation_identity_mismatch_fails(
    tmp_path: Path, field: str, value: str
) -> None:
    module = _load("mlip_training_cluster_adapter_foundation_bad", PLUGIN / "adapter_cluster.py")
    context, report_path = _completed_finetune_context(tmp_path, module)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["foundation_model"][field] = value
    _write_json(report_path, report)

    result = module.Adapter().check(context)

    assert result["status"] == "FAIL"
    assert any(
        item["code"] == "training.cluster_foundation_identity" for item in result["diagnostics"]
    )


def test_mace_scheduled_completion_evidence_is_accepted(tmp_path: Path) -> None:
    module = _load("mlip_training_cluster_adapter_mace_ok", PLUGIN / "adapter_cluster.py")
    context, _ = _completed_mace_finetune_context(tmp_path, module)

    result = module.Adapter().check(context)

    assert result["status"] == "OK", result.get("diagnostics")


@pytest.mark.parametrize("operation", ["train", "finetune"])
def test_m3gnet_scheduled_completion_evidence_is_accepted(tmp_path: Path, operation: str) -> None:
    module = _load(
        f"mlip_training_cluster_adapter_m3gnet_{operation}", PLUGIN / "adapter_cluster.py"
    )
    context, _ = _completed_m3gnet_context(tmp_path, module, operation)

    result = module.Adapter().check(context)

    assert result["status"] == "OK", result.get("diagnostics")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("completed_epochs", 0),
        ("normal_completion", False),
        ("all_recorded_metrics_finite", False),
        ("native_model_reload", "FAIL"),
    ],
)
def test_m3gnet_scheduled_completion_mismatch_fails(tmp_path: Path, field: str, value) -> None:
    module = _load("mlip_training_cluster_adapter_m3gnet_bad", PLUGIN / "adapter_cluster.py")
    context, result_path = _completed_m3gnet_context(tmp_path, module, "finetune")
    manifest = json.loads(result_path.read_text(encoding="utf-8"))
    manifest["provenance"]["completion"][field] = value
    _write_json(result_path, manifest)

    result = module.Adapter().check(context)

    assert result["status"] == "FAIL"
    assert any(item["code"] == f"training.m3gnet_{field}" for item in result["diagnostics"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("completed_epochs", 0),
        ("normal_completion", False),
        ("all_recorded_metrics_finite", False),
        ("native_model_reload", "FAIL"),
    ],
)
def test_chgnet_scheduled_completion_mismatch_fails(tmp_path: Path, field: str, value) -> None:
    module = _load("mlip_training_cluster_adapter_chgnet_bad", PLUGIN / "adapter_cluster.py")
    context, _ = _completed_finetune_context(tmp_path, module)
    result_path = Path(context["attempt_dir"]) / context["parameters"]["result_manifest"]
    manifest = json.loads(result_path.read_text(encoding="utf-8"))
    manifest["provenance"]["completion"][field] = value
    _write_json(result_path, manifest)

    result = module.Adapter().check(context)

    assert result["status"] == "FAIL"
    assert any(item["code"] == f"training.chgnet_{field}" for item in result["diagnostics"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("completed_epochs", 0),
        ("normal_completion", False),
        ("all_recorded_metrics_finite", False),
        ("native_model_reload", "FAIL"),
    ],
)
def test_mace_scheduled_completion_mismatch_fails(tmp_path: Path, field: str, value) -> None:
    module = _load("mlip_training_cluster_adapter_mace_bad", PLUGIN / "adapter_cluster.py")
    context, result_path = _completed_mace_finetune_context(tmp_path, module)
    manifest = json.loads(result_path.read_text(encoding="utf-8"))
    manifest["provenance"]["completion"][field] = value
    _write_json(result_path, manifest)

    result = module.Adapter().check(context)

    assert result["status"] == "FAIL"
    assert any(item["code"] == f"training.mace_{field}" for item in result["diagnostics"])


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
    assert fingerprint_path(tree) == first
    assert first.startswith("sha256:") and len(first) == 71
