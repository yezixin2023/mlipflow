"""Tests for the read-only historical training source auditor."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "plugins" / "mlip-training" / "audit_sources.py"


def _module():
    spec = importlib.util.spec_from_file_location("training_source_auditor", AUDITOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrainingSourceAuditTests(unittest.TestCase):
    def test_detects_real_mace_missing_continuation_pattern(self) -> None:
        auditor = _module()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "run.slurm"
            source.write_text(
                """#!/bin/bash
#SBATCH -N 3
python /home/user/mace/cli/run_train.py \\
  --name=model \\
  --E0s=average
  --energy_key=energy \\
  --seed=15
""",
                encoding="utf-8",
            )
            record = auditor.audit_source("mace", source)
            codes = {issue["code"] for issue in record["issues"]}
            self.assertIn("shell.missing_continuation", codes)
            self.assertIn("shell.absolute_paths", codes)
            # The seed is after the broken command and therefore correctly not captured.
            self.assertIn("shell.seed_missing", codes)

    def test_detects_unseeded_chgnet_and_mixed_m3gnet_matgl(self) -> None:
        auditor = _module()
        with tempfile.TemporaryDirectory() as directory:
            chgnet = Path(directory) / "chgnet.py"
            chgnet.write_text(
                """from chgnet.model import CHGNet
from chgnet.trainer import Trainer
model = CHGNet.load()
trainer = Trainer(model=model, epochs=5, use_device='cpu')
trainer.train(train_loader, val_loader, test_loader)
""",
                encoding="utf-8",
            )
            m3gnet = Path(directory) / "m3gnet.py"
            m3gnet.write_text(
                """import matgl
from m3gnet.models import M3GNet
from dgl.data.utils import split_dataset
parts = split_dataset(dataset, frac_list=[0.8, 0.05, 0.15], shuffle=False, random_state=42)
""",
                encoding="utf-8",
            )
            chg_record = auditor.audit_source("chgnet", chgnet)
            self.assertIn("python.seed_missing", {issue["code"] for issue in chg_record["issues"]})
            m3_record = auditor.audit_source("m3gnet", m3gnet)
            self.assertIn(
                "python.mixed_m3gnet_matgl",
                {issue["code"] for issue in m3_record["issues"]},
            )
            self.assertEqual(42, m3_record["seeds"][0]["value"])
            self.assertEqual(
                [0.8, 0.05, 0.15], m3_record["split_settings"][0]["keywords"]["frac_list"]
            )

    def test_deepmd_config_summary_is_path_redacted(self) -> None:
        auditor = _module()
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "input.json"
            config.write_text(
                json.dumps(
                    {
                        "model": {
                            "type_map": ["Li", "P", "S"],
                            "descriptor": {"type": "se_atten_v2", "seed": 1},
                            "fitting_net": {"seed": 2, "precision": "float64"},
                        },
                        "training": {
                            "seed": 10,
                            "numb_steps": 300000,
                            "training_data": {
                                "systems": ["/public/home/example/train/system-a/"]
                            },
                            "validation_data": {
                                "systems": ["/public/home/example/test/system-a/"]
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            report = auditor.build_report([("deepmd", config)])
            source = report["sources"][0]
            self.assertEqual("se_atten_v2", source["configuration"]["descriptor"]["type"])
            self.assertEqual(1, source["data"]["training"]["system_count"])
            self.assertEqual(2, source["absolute_path_count"])
            rendered = json.dumps(report)
            self.assertNotIn("/public/home/", rendered)
            self.assertTrue(report["redaction"]["private_absolute_path_values_removed"])


if __name__ == "__main__":
    unittest.main()
