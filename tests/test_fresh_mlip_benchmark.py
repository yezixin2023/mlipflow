"""Contract tests for first-party fresh MLIP benchmark evaluation."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
FRESH_PATH = ROOT / "plugins" / "mlip-benchmark" / "fresh_benchmark.py"
ADAPTER_PATH = ROOT / "plugins" / "mlip-benchmark" / "adapter.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakePredictor:
    prediction_units = {
        "energy": "eV",
        "force": "eV/angstrom",
        "stress": "GPa",
    }
    runtime_identity = {
        "framework": "deepmd",
        "framework_version": "fake-contract-1",
        "backend": "injected-test-predictor",
        "device": "cpu",
        "exact_model_family": "deepmd-dpa2",
        "energy_convention": "total",
        "stress_convention": "ase-voigt-xx-yy-zz-yz-xz-xy",
    }

    def predict(self, sample):
        if sample["id"] == "s1":
            return {
                "energy": 4.2,
                "force": [[0.1, 1.1, 2.1], [3.1, 4.1, 5.1]],
                "stress": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            }
        return {
            "energy": 2.8,
            "force": [[0.2, 1.2, 2.2]],
            "stress": [2.2, 3.2, 4.2, 5.2, 6.2, 7.2],
        }


class FreshBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.fresh = _load(FRESH_PATH, "test_fresh_benchmark_runner")
        self.adapter = _load(ADAPTER_PATH, "test_fresh_benchmark_adapter").Adapter()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = self.root / "model.pb"
        self.model.write_bytes(b"test-only-model-artifact")
        self.dataset = self.root / "dataset.json"
        self.dataset.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "units": {
                        "energy": "eV",
                        "force": "eV/angstrom",
                        "stress": "GPa",
                    },
                    "energy_convention": "total",
                    "stress_convention": "ase-voigt-xx-yy-zz-yz-xz-xy",
                    "samples": [
                        {
                            "id": "s1",
                            "species": ["Li", "S"],
                            "positions": [[0, 0, 0], [1, 1, 1]],
                            "cell": [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
                            "pbc": True,
                            "references": {
                                "energy": 4.0,
                                "force": [[0, 1, 2], [3, 4, 5]],
                                "stress": [1, 2, 3, 4, 5, 6],
                            },
                        },
                        {
                            "id": "s2",
                            "species": ["Li"],
                            "positions": [[0, 0, 0]],
                            "cell": [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
                            "pbc": [True, True, True],
                            "references": {
                                "energy": 3.0,
                                "force": [[0, 1, 2]],
                                "stress": [2, 3, 4, 5, 6, 7],
                            },
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.model_sha = self.fresh.fingerprint_path(self.model)
        self.dataset_sha = self.fresh.fingerprint_path(self.dataset)
        self.output = self.root / "attempt" / "fresh"

    def tearDown(self):
        self.temporary.cleanup()

    def evaluate(self, output=None, energy_normalization="per-atom", targets=None):
        return self.fresh.evaluate_fresh(
            model=self.model,
            dataset=self.dataset,
            output_dir=output or self.output,
            model_family="deepmd-dpa2",
            model_fingerprint=self.model_sha,
            dataset_fingerprint=self.dataset_sha,
            task="static-pes",
            scenario="fresh-contract-v1",
            split="test",
            targets=targets or ["energy", "force", "stress"],
            units={"energy": "eV/atom", "force": "eV/angstrom", "stress": "GPa"},
            energy_normalization=energy_normalization,
            stress_convention="ase-voigt-xx-yy-zz-yz-xz-xy",
            predictor=FakePredictor(),
        )

    def context(self):
        return {
            "project_root": str(self.root),
            "attempt_dir": str(self.root / "attempt"),
            "inputs": {
                "model": "model.pb",
                "benchmark_dataset": "dataset.json",
                "output_dir": "fresh",
            },
            "parameters": {
                "operation": "evaluate-fresh",
                "model_family": "deepmd-dpa2",
                "model_fingerprint": self.model_sha,
                "dataset_fingerprint": self.dataset_sha,
                "task": "static-pes",
                "scenario": "fresh-contract-v1",
                "split": "test",
                "targets": ["energy", "force", "stress"],
                "units": {
                    "energy": "eV/atom",
                    "force": "eV/angstrom",
                    "stress": "GPa",
                },
                "energy_normalization": "per-atom",
                "stress_convention": "ase-voigt-xx-yy-zz-yz-xz-xy",
                "device": "cpu",
            },
            "backend": "local",
            "resources": {"cpus": 1},
        }

    def test_fresh_energy_force_stress_evidence_metrics_counts_and_provenance(self):
        paths = self.evaluate()
        self.assertEqual(set(self.fresh.OUTPUT_NAMES), set(paths))
        evidence = json.loads(paths["prediction_evidence.json"].read_text())
        provenance = json.loads(paths["provenance.json"].read_text())
        metrics = json.loads(paths["metrics.json"].read_text())
        self.assertTrue(evidence["model_execution"])
        self.assertTrue(provenance["model_execution"])
        self.assertEqual("fresh", provenance["mode"])
        self.assertEqual(2, evidence["structure_count"])
        self.assertEqual({"energy": 2, "force": 9, "stress": 12}, evidence["scalar_sample_counts"])
        energy_rows = [row for row in evidence["records"] if row["target"] == "energy"]
        self.assertEqual([2.0, 3.0], [row["reference"] for row in energy_rows])
        self.assertEqual([2.1, 2.8], [row["prediction"] for row in energy_rows])
        counts = {row["metric"]: row["sample_count"] for row in metrics["records"]}
        self.assertEqual(2, counts["energy_mae"])
        self.assertEqual(9, counts["force_rmse"])
        self.assertEqual(12, counts["stress_pearson_r"])
        self.assertEqual(self.model_sha, provenance["model_fingerprint"])
        self.assertEqual(self.dataset_sha, provenance["dataset_fingerprint"])
        self.assertTrue(provenance["prediction_evidence_sha256"].startswith("sha256:"))
        self.assertEqual(3, len(provenance["normalized_artifact_sha256"]))
        checked = self.adapter.check(self.context())
        self.assertEqual("OK", checked["status"], checked["diagnostics"])
        self.assertEqual("fresh", checked["mode"])

    def test_total_and_per_atom_energy_conventions_are_distinct(self):
        self.evaluate()
        per_atom = json.loads((self.output / "prediction_evidence.json").read_text())
        total_output = self.root / "total-attempt" / "fresh"
        self.evaluate(output=total_output, energy_normalization="total")
        total = json.loads((total_output / "prediction_evidence.json").read_text())
        per_values = [row["prediction"] for row in per_atom["records"] if row["target"] == "energy"]
        total_values = [row["prediction"] for row in total["records"] if row["target"] == "energy"]
        self.assertEqual([2.1, 2.8], per_values)
        self.assertEqual([4.2, 2.8], total_values)
        self.assertEqual("per-atom", per_atom["conventions"]["energy_normalization"])
        self.assertEqual("total", total["conventions"]["energy_normalization"])

    def test_fresh_plan_is_argv_based_and_rejects_overwrite(self):
        plan = self.adapter.plan(self.context())
        self.assertEqual("READY", plan["status"])
        self.assertFalse(plan["shell"])
        self.assertEqual("fresh-model-inference", plan["calculation_claim"])
        self.assertEqual(5, len(plan["expected_outputs"]))
        self.assertIn("--model-family", plan["argv"])
        self.evaluate()
        self.assertEqual("BLOCKED", self.adapter.plan(self.context())["status"])
        with self.assertRaisesRegex(self.fresh.FreshBenchmarkError, "must not already contain"):
            self.evaluate()

    def test_all_six_exact_families_plan_and_four_deepmd_families_share_loader(self):
        context = self.context()
        for family in self.fresh.MODEL_FRAMEWORKS:
            with self.subTest(family=family):
                context["parameters"]["model_family"] = family
                plan = self.adapter.plan(context)
                self.assertEqual("READY", plan["status"], plan["diagnostics"])
                self.assertIn(family, plan["argv"])

        runtime = self.fresh._shared_runtime_module()
        sentinel = object()
        with mock.patch.object(runtime, "DeepMDPredictor", return_value=sentinel) as loader:
            for family in (
                "deepmd-se_e2_a",
                "deepmd-se_e2_r",
                "deepmd-se_atten_v2",
                "deepmd-dpa2",
            ):
                self.assertIs(sentinel, runtime.load_inference_predictor(self.model, family, "cpu"))
        self.assertEqual(4, loader.call_count)

    def test_model_dataset_and_prediction_evidence_drift_fail_collection(self):
        self.evaluate()
        self.model.write_bytes(b"changed-model")
        self.assertEqual("FAIL", self.adapter.check(self.context())["status"])
        self.model.write_bytes(b"test-only-model-artifact")
        original_dataset = self.dataset.read_text()
        self.dataset.write_text(original_dataset + "\n")
        self.assertEqual("FAIL", self.adapter.check(self.context())["status"])
        self.dataset.write_text(original_dataset)
        evidence_path = self.output / "prediction_evidence.json"
        evidence = json.loads(evidence_path.read_text())
        evidence["records"][0]["prediction"] += 1
        evidence_path.write_text(json.dumps(evidence))
        self.assertEqual("FAIL", self.adapter.check(self.context())["status"])

    def test_partial_output_and_model_load_failure_leave_no_success(self):
        self.evaluate()
        (self.output / "metrics.json").unlink()
        checked = self.adapter.check(self.context())
        self.assertEqual("FAIL", checked["status"])
        self.assertIn("benchmark.normalized_outputs_partial", {item["code"] for item in checked["diagnostics"]})

        failed_output = self.root / "failed" / "fresh"
        with mock.patch.object(
            self.fresh,
            "load_predictor",
            side_effect=self.fresh.FreshBenchmarkError("injected model-load failure"),
        ):
            with self.assertRaisesRegex(self.fresh.FreshBenchmarkError, "model-load failure"):
                self.fresh.evaluate_fresh(
                    model=self.model, dataset=self.dataset, output_dir=failed_output,
                    model_family="deepmd-dpa2", model_fingerprint=self.model_sha,
                    dataset_fingerprint=self.dataset_sha, task="static-pes",
                    scenario="fresh-contract-v1", split="test", targets=["energy"],
                    units={"energy": "eV/atom"}, energy_normalization="per-atom",
                    stress_convention=None,
                )
        self.assertFalse(failed_output.exists())

    def test_replay_and_fresh_mode_are_isolated(self):
        self.evaluate()
        provenance = json.loads((self.output / "provenance.json").read_text())
        provenance["mode"] = "replay"
        provenance["model_execution"] = False
        (self.output / "provenance.json").write_text(json.dumps(provenance))
        self.assertEqual("FAIL", self.adapter.check(self.context())["status"])


if __name__ == "__main__":
    unittest.main()
