import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

PLUGIN = Path(__file__).resolve().parents[1] / "plugins" / "mlip-training"
sys.path.insert(0, str(PLUGIN))
import mlip_chgnet  # noqa: E402
import mlip_deepmd  # noqa: E402
import mlip_m3gnet  # noqa: E402
import mlip_mace  # noqa: E402
from mlip_common import TrainingError, write_result  # noqa: E402


def ns(framework, operation="train", precision="float64"):
    return argparse.Namespace(
        framework=framework,
        operation=operation,
        output="model.bin",
        result_manifest="result.json",
        seed=23,
        device="cpu",
        precision=precision,
        foundation_model="foundation.model",
        dataset_fingerprint="sha256:" + "1" * 64,
        config_fingerprint="sha256:" + "2" * 64,
    )


def test_deepmd_plan():
    cfg = {
        "_mlipflow": {"backend": "pt"},
        "model": {"descriptor": {"type": "dpa2"}},
        "training": {"seed": 23},
    }
    assert "--finetune" in mlip_deepmd.plan(ns("deepmd", "finetune"), cfg)["train_args"]


def test_chgnet_guard(tmp_path):
    with pytest.raises(Exception):
        mlip_chgnet.plan(
            ns("chgnet"), {"framework": "chgnet", "chgnet": {}}, tmp_path / "data.jsonl"
        )


def test_m3gnet_plan(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text("{}\n")
    assert (
        mlip_m3gnet.plan(ns("m3gnet"), {"framework": "m3gnet", "m3gnet": {}}, data)["framework"]
        == "m3gnet"
    )


def test_mace_plan(tmp_path):
    data = tmp_path / "train.xyz"
    data.write_text("0\ncomment\n")
    cfg = {"framework": "mace", "mace": {"name": "demo", "options": {"max_num_epochs": 1}}}
    assert mlip_mace.plan(ns("mace"), cfg, tmp_path / "config.json", data)["framework"] == "mace"


@pytest.mark.parametrize("metric", [True, "1.0", None, float("nan"), float("inf"), float("-inf")])
def test_write_result_rejects_invalid_metrics_without_manifest(tmp_path, metric):
    args = ns("chgnet", precision="float32")
    args.output = str(tmp_path / "model.pth.tar")
    args.result_manifest = str(tmp_path / "result.json")
    Path(args.output).write_bytes(b"model")

    with pytest.raises(TrainingError, match="metric 'loss'"):
        write_result(args, "OK", "test", {"loss": metric}, "application/octet-stream", {})

    assert not Path(args.result_manifest).exists()


def test_write_result_preserves_finite_numeric_metrics(tmp_path):
    args = ns("chgnet", precision="float32")
    args.output = str(tmp_path / "model.pth.tar")
    args.result_manifest = str(tmp_path / "result.json")
    Path(args.output).write_bytes(b"model")

    write_result(
        args,
        "OK",
        "test",
        {"Validation Loss": 0.125, "epochs": 2},
        "application/octet-stream",
        {},
    )

    result = json.loads(Path(args.result_manifest).read_text(encoding="utf-8"))
    assert result["status"] == "OK"
    assert result["metrics"] == {"epochs": 2.0, "validation_loss": 0.125}


def test_chgnet_training_completion_accepts_requested_finite_epochs():
    trainer = SimpleNamespace(
        epochs=2,
        starting_epoch=0,
        targets="ef",
        training_history={
            "e": {"train": [0.4, 0.2], "val": [0.5, 0.3], "test": []},
            "f": {"train": [0.8, 0.6], "val": [0.9, 0.7], "test": []},
        },
    )

    mlip_chgnet._validate_training_completion(trainer)


def test_chgnet_training_completion_rejects_early_exit():
    trainer = SimpleNamespace(
        epochs=2,
        starting_epoch=0,
        targets="e",
        training_history={"e": {"train": [0.4], "val": [0.5], "test": []}},
    )

    with pytest.raises(TrainingError, match="completed 1 of 2 requested epochs"):
        mlip_chgnet._validate_training_completion(trainer)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_chgnet_training_completion_rejects_non_finite_history(value):
    trainer = SimpleNamespace(
        epochs=1,
        starting_epoch=0,
        targets="e",
        training_history={"e": {"train": [0.4], "val": [value], "test": []}},
    )

    with pytest.raises(TrainingError, match="non-finite"):
        mlip_chgnet._validate_training_completion(trainer)
