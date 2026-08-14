import argparse
import sys
from pathlib import Path
import pytest

PLUGIN = Path(__file__).resolve().parents[1] / "plugins" / "mlip-training"
sys.path.insert(0, str(PLUGIN))
import mlip_chgnet  # noqa: E402
import mlip_deepmd  # noqa: E402
import mlip_m3gnet  # noqa: E402
import mlip_mace  # noqa: E402


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
