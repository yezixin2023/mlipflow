"""Scientific regressions from the read-only historical benchmark audit."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "tests" / "fixtures" / "mlip_benchmark_historical_audit.json"
WRAPPER = ROOT / "plugins" / "mlip-benchmark" / "benchmark_wrapper.py"


def _wrapper():
    spec = importlib.util.spec_from_file_location("historical_benchmark_wrapper", WRAPPER)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {WRAPPER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HistoricalBenchmarkAuditTests(unittest.TestCase):
    def test_real_source_and_output_extractions_preserve_their_claim_boundary(self) -> None:
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        families = audit["families"]
        self.assertFalse(audit["model_execution"])
        self.assertIn("not fresh inference parity", audit["claim"])
        self.assertEqual(set(_wrapper().MODEL_NAMES), set(families))

        for family in ("deepmd-se_e2_a", "deepmd-se_e2_r", "deepmd-se_atten_v2"):
            record = families[family]
            self.assertEqual("REPLAY_VERIFIED", record["status"])
            self.assertEqual("deepmd-metrics-xlsx", record["parser"])
            self.assertEqual("per-atom", record["energy_normalization"])
            self.assertEqual("source-unit-unspecified", record["energy_unit"])
            self.assertEqual("source-unit-unspecified", record["force_unit"])
            self.assertEqual(13024, record["structure_count"])
            self.assertEqual(2673726, record["force_component_count"])

        chgnet = families["chgnet"]
        self.assertEqual("chgnet-efs-metrics-xlsx", chgnet["parser"])
        self.assertEqual(9753, chgnet["structure_count"])
        self.assertEqual(1992795, chgnet["force_component_count"])
        self.assertEqual(9 * chgnet["structure_count"], chgnet["stress_component_count"])
        self.assertEqual("source-unit-unspecified", chgnet["stress_unit"])
        self.assertEqual("source-order-and-sign-unspecified", chgnet["stress_convention"])
        self.assertAlmostEqual(0.0080861044340676, chgnet["energy_rmse"])
        self.assertAlmostEqual(0.1889819035175336, chgnet["force_rmse"])
        self.assertAlmostEqual(12.46909130171059, chgnet["stress_rmse"])

        self.assertEqual("MISSING_SOURCE", families["m3gnet"]["status"])
        self.assertEqual("MISSING_SOURCE", families["deepmd-dpa2"]["status"])

    def test_ranking_direction_and_ties_are_metric_only(self) -> None:
        wrapper = _wrapper()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "tie.json"
            evidence.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "model": model,
                                "task": "static-pes",
                                "scenario": "tie-regression",
                                "split": "test",
                                "metric": "energy_rmse",
                                "value": 0.1,
                                "unit": "eV/atom",
                                "direction": "minimize",
                                "sample_count": 4,
                            }
                            for model in ("m3gnet", "chgnet")
                        ]
                    }
                ),
                encoding="utf-8",
            )
            outputs = wrapper.normalize_benchmark(
                [{"path": evidence, "evidence_locator": "audit://tie.json"}],
                root / "normalized",
                mode="replay",
            )
            ranking = json.loads(outputs["model_ranking.json"].read_text(encoding="utf-8"))
            candidates = ranking["rankings"][0]["candidates"]
            self.assertEqual(["chgnet", "m3gnet"], [item["model"] for item in candidates])
            self.assertEqual([1, 1], [item["rank"] for item in candidates])
            metrics = json.loads(outputs["metrics.json"].read_text(encoding="utf-8"))
            self.assertEqual("replay", metrics["mode"])
            self.assertNotIn("model_execution", metrics)


if __name__ == "__main__":
    unittest.main()
