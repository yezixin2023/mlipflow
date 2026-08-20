"""Contract tests for first-party fresh MLIP benchmark evaluation."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
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


def test_metric_recomputation_accepts_only_floating_roundoff() -> None:
    adapter = _load(ADAPTER_PATH, "test_benchmark_metric_tolerance")
    stored = [{"metric": "force_mae", "sample_count": 3, "value": 0.3}]
    recomputed = [
        {
            "metric": "force_mae",
            "sample_count": 3,
            "value": math.nextafter(0.3, math.inf),
        }
    ]
    assert adapter._metric_records_match(stored, recomputed)

    recomputed[0]["value"] += 1e-6
    assert not adapter._metric_records_match(stored, recomputed)


class FakePredictor:
    prediction_units = {
        "energy": "eV",
        "force": "eV/angstrom",
        "stress": "GPa",
    }
    runtime_details = {
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
        self.output = self.root / "attempt" / "fresh"

    def tearDown(self):
        self.temporary.cleanup()

    def evaluate(self, output=None, energy_normalization="per-atom", targets=None):
        return self.fresh.evaluate_fresh(
            model=self.model,
            dataset=self.dataset,
            output_dir=output or self.output,
            model_family="deepmd-dpa2",
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
        values = {row["metric"]: row["value"] for row in metrics["records"]}
        self.assertEqual(2, counts["energy_mae"])
        self.assertEqual(9, counts["force_rmse"])
        self.assertEqual(3, counts["maximum_atomic_force_error"])
        self.assertAlmostEqual((3 * 0.2**2) ** 0.5, values["maximum_atomic_force_error"])
        self.assertEqual(12, counts["stress_pearson_r"])
        maximum_force = next(
            row
            for row in metrics["records"]
            if row["metric"] == "maximum_atomic_force_error"
        )
        self.assertEqual(
            {
                "scope": "overall",
                "target": "force",
                "error_norm": "atomic-l2-vector",
                "aggregation": "maximum-over-atoms",
            },
            maximum_force["dimensions"],
        )
        self.assertEqual(str(self.model.resolve()), provenance["model_path"])
        self.assertEqual(str(self.dataset.resolve()), provenance["dataset_path"])
        self.assertEqual(
            {"prediction_evidence.json", "metrics.json", "benchmark_summary.csv", "model_ranking.json"},
            {item["path"] for item in provenance["output_artifacts"]},
        )
        checked = self.adapter.check(self.context())
        self.assertEqual("OK", checked["status"], checked["diagnostics"])
        self.assertEqual("fresh", checked["mode"])

    def test_maximum_atomic_force_error_requires_explicit_n_by_three_vectors(self):
        normalizer = self.fresh._normalizer_module()
        defaults = {
            "model": "chgnet",
            "task": "static-pes",
            "scenario": "force-vector-contract",
            "split": "audit",
            "units": {"force": "eV/angstrom"},
        }
        explicit = self.root / "explicit-force-vectors.json"
        explicit.write_text(
            json.dumps(
                [
                    {
                        "target": "force",
                        "reference": [[0, 0, 0], [1, 1, 1]],
                        "prediction": [[0.1, 0.2, 0.2], [1.3, 1.4, 1.0]],
                    }
                ]
            ),
            encoding="utf-8",
        )
        records, _ = normalizer._execute_source({"path": explicit, **defaults})
        maximum = next(
            record
            for record in records
            if record["metric"] == "maximum_atomic_force_error"
        )
        self.assertEqual(2, maximum["sample_count"])
        self.assertAlmostEqual(0.5, maximum["value"])

        flattened = self.root / "flattened-force-components.json"
        flattened.write_text(
            json.dumps(
                [
                    {
                        "target": "force",
                        "reference": [0, 0, 0, 1, 1, 1],
                        "prediction": [0.1, 0.2, 0.2, 1.3, 1.4, 1.0],
                    }
                ]
            ),
            encoding="utf-8",
        )
        flat_records, _ = normalizer._execute_source({"path": flattened, **defaults})
        self.assertNotIn(
            "maximum_atomic_force_error",
            {record["metric"] for record in flat_records},
        )

    def test_single_structure_keeps_errors_and_records_energy_pearson_unavailable(self):
        dataset = json.loads(self.dataset.read_text(encoding="utf-8"))
        dataset["samples"] = dataset["samples"][:1]
        self.dataset.write_text(json.dumps(dataset), encoding="utf-8")

        paths = self.evaluate()
        metrics = json.loads(paths["metrics.json"].read_text())
        provenance = json.loads(paths["provenance.json"].read_text())
        metric_names = {record["metric"] for record in metrics["records"]}
        self.assertIn("energy_mae", metric_names)
        self.assertIn("energy_rmse", metric_names)
        self.assertNotIn("energy_pearson_r", metric_names)
        self.assertEqual(1, len(metrics["unavailable_metrics"]))
        unavailable = metrics["unavailable_metrics"][0]
        self.assertEqual("energy_pearson_r", unavailable["metric"])
        self.assertEqual("insufficient-scalar-pairs", unavailable["reason"])
        self.assertEqual(1, unavailable["sample_count"])
        self.assertEqual(metrics["unavailable_metrics"], provenance["unavailable_metrics"])

        checked = self.adapter.check(self.context())
        self.assertEqual("OK", checked["status"], checked["diagnostics"])

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

    def test_changed_prediction_evidence_fails_collection(self):
        self.evaluate()
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
                    model_family="deepmd-dpa2", task="static-pes",
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

    def test_model_only_stress_is_not_a_fair_two_model_selection_axis(self):
        self.evaluate()
        first = self.root / "model-a" / "prediction_evidence.json"
        second = self.root / "model-b" / "prediction_evidence.json"
        first.parent.mkdir()
        second.parent.mkdir()
        payload = json.loads((self.output / "prediction_evidence.json").read_text())
        first.write_text(json.dumps(payload), encoding="utf-8")
        payload["model"]["family"] = "chgnet"
        payload["model"]["framework"] = "chgnet"
        for record in payload["records"]:
            record["model"] = "chgnet"
        payload["records"] = [
            record for record in payload["records"] if record["target"] != "stress"
        ]
        second.write_text(json.dumps(payload), encoding="utf-8")
        context = {
            "project_root": str(self.root),
            "attempt_dir": str(self.root / "joint-attempt"),
            "inputs": {
                "evidence_inputs": [
                    {
                        "path": "model-a/prediction_evidence.json",
                        "evidence_locator": "fresh/deepmd-dpa2/prediction_evidence.json",
                    },
                    {
                        "path": "model-b/prediction_evidence.json",
                        "evidence_locator": "fresh/chgnet/prediction_evidence.json",
                    },
                ],
                "output_dir": "joint",
            },
            "parameters": {
                "operation": "normalize-execute",
                "expected_models": ["deepmd-dpa2", "chgnet"],
                "task": "static-pes",
                "scenario": "fresh-contract-v1",
                "split": "test",
                "units": {
                    "energy": "eV/atom",
                    "force": "eV/angstrom",
                    "stress": "GPa",
                },
            },
            "backend": "local",
            "resources": {"cpus": 1},
        }
        plan = self.adapter.plan(context)
        self.assertEqual("READY", plan["status"], plan["diagnostics"])
        self.assertEqual(2, plan["argv"].count("--input-locator"))
        result = subprocess.run(plan["argv"], check=False, capture_output=True, text=True)
        self.assertEqual(0, result.returncode, result.stderr)
        checked = self.adapter.check(context)
        self.assertEqual("OK", checked["status"], checked["diagnostics"])
        ranking = json.loads(
            (Path(context["attempt_dir"]) / "joint" / "model_ranking.json").read_text()
        )
        fair_axes = [
            item
            for item in ranking["rankings"]
            if item["comparable_model_count"] == 2
        ]
        diagnostic_only = [
            item
            for item in ranking["rankings"]
            if item["comparable_model_count"] < 2
        ]
        self.assertTrue(fair_axes)
        self.assertTrue(
            all(
                item["metric"].startswith(("energy_", "force_"))
                or item["metric"] == "maximum_atomic_force_error"
                for item in fair_axes
            )
        )
        self.assertTrue(diagnostic_only)
        self.assertTrue(
            all(item["metric"].startswith("stress_") for item in diagnostic_only)
        )

    def test_scheduled_fresh_plan_and_fetched_report_roundtrip(self):
        model_reference = self.root / "model-reference.json"
        dataset_reference = self.root / "benchmark-dataset-reference.json"
        model_reference.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "model_id": "deepmd-test-model",
                    "framework": "deepmd",
                    "relative_path": "deepmd/deepmd-test-model.pb",
                    "kind": "file",
                }
            ),
            encoding="utf-8",
        )
        dataset_reference.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "dataset_id": "test-dataset",
                    "relative_path": "test-dataset/benchmark/test.json",
                    "kind": "file",
                }
            ),
            encoding="utf-8",
        )
        (self.root / "project.yaml").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "project": {"id": "scheduled-benchmark-test"},
                    "workflow": {"nodes": [{"id": "benchmark-deepmd"}]},
                }
            ),
            encoding="utf-8",
        )
        context = self.context()
        context.update(
            {
                "inputs": {
                    "model_reference": "model-reference.json",
                    "benchmark_dataset_reference": "benchmark-dataset-reference.json",
                    "output_dir": "fresh",
                },
                "backend": "ssh-slurm",
                "resources": {
                    "cpus": 1,
                    "gpus": 0,
                    "memory": "2G",
                    "walltime": "00:05:00",
                },
            }
        )
        plan = self.adapter.plan(context)
        self.assertEqual("READY", plan["status"], plan["diagnostics"])
        self.assertEqual(
            "benchmark-deepmd-canonical",
            plan["scheduled_execution"]["template_family"],
        )
        self.assertEqual(6, len(plan["scheduled_execution"]["fetch_outputs"]))
        self.evaluate()
        report = {
            "schema_version": 1,
            "status": "OK",
            "return_code": 0,
            "exact_model_family": "deepmd-dpa2",
            "framework": "deepmd",
            "model": {
                "id": "deepmd-test-model",
                "relative_path": "deepmd/deepmd-test-model.pb",
            },
            "dataset": {
                "id": "test-dataset",
                "relative_path": "test-dataset/benchmark/test.json",
            },
            "outputs": {
                name: str(self.output / name) for name in self.fresh.OUTPUT_NAMES
            },
        }
        (self.root / "attempt" / "cluster-benchmark-report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        context["execution"] = {"plan": plan}
        checked = self.adapter.check(context)
        self.assertEqual("OK", checked["status"], checked["diagnostics"])


if __name__ == "__main__":
    unittest.main()
