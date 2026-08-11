"""Contract tests for the real, seeded icet SQS wrapper."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "high-entropy-structure"


def _module(filename: str):
    path = PLUGIN / filename
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ManuscriptSQSTests(unittest.TestCase):
    def test_committed_local_integration_smoke_is_code_bound_and_bounded(self) -> None:
        report_path = ROOT / "reports" / "sqs_local_integration_smoke.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        entrypoint = ROOT / report["implementation"]["entrypoint_locator"]
        digest = hashlib.sha256(entrypoint.read_bytes()).hexdigest()

        self.assertEqual("LOCAL_INTEGRATION_SMOKE_PASS", report["status"])
        self.assertEqual(digest, report["implementation"]["entrypoint_sha256"])
        self.assertEqual(2, report["result"]["run_count"])
        self.assertTrue(report["result"]["byte_identical_between_runs"])
        self.assertEqual(57, sum(report["result"]["output_species_counts"].values()))
        self.assertEqual(7, sum(report["bounded_parameters"]["metal_counts"].values()))
        self.assertLessEqual(report["bounded_parameters"]["n_steps"], 100)
        self.assertFalse(report["scientific_claim"]["production_sqs_executed"])
        self.assertFalse(report["scientific_claim"]["historical_structure_parity_established"])
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn("/Users/", serialized)
        self.assertNotIn("/public/home/", serialized)

    def test_li10_manifest_matches_228_atom_historical_setup(self) -> None:
        sqs = _module("sqs.py")
        manifest = {
            "schema_version": 1,
            "alloy_sublattice": {
                "prototype_species": "Zn",
                "allowed_species": ["Zn", "Fe", "Cu", "Ni", "Mn"],
            },
            "cluster_cutoffs_angstrom": [7, 5],
            "supercell_repeat": [2, 2, 1],
            "n_steps": 20000,
            "output_format": "vasp",
            "candidates": [
                {
                    "id": "Mn6_Fe3_Ni8_Cu4_Zn7",
                    "counts": {"Zn": 7, "Fe": 3, "Cu": 4, "Ni": 8, "Mn": 6},
                }
            ],
        }
        # Li10M7P8S32: 57-atom prototype, seven metal sites, 2x2x1 repeat.
        symbols = ["Li"] * 10 + ["Zn"] * 7 + ["P"] * 8 + ["S"] * 32
        normalized = sqs.validate_manifest(manifest, prototype_symbols=symbols)
        self.assertEqual(28, normalized["alloy_site_count_from_prototype"])
        self.assertEqual(20000, normalized["n_steps"])
        self.assertEqual([7.0, 5.0], normalized["cluster_cutoffs_angstrom"])

        manifest["candidates"][0]["counts"]["Mn"] = 5
        with self.assertRaisesRegex(ValueError, "require 28"):
            sqs.validate_manifest(manifest, prototype_symbols=symbols)

    def test_adapter_defaults_to_fingerprinted_bundled_generator(self) -> None:
        adapter = _module("adapter.py").Adapter()
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            attempt = project / "attempt"
            attempt.mkdir()
            context = {
                "project_root": str(project),
                "attempt_dir": str(attempt),
                "inputs": {
                    "prototype_structure": "prototype.cif",
                    "composition_manifest": "composition.json",
                },
                "parameters": {"seed": 17, "max_candidates": 1},
                "backend": "local",
                "resources": {"cpus": 1},
            }
            plan = adapter.plan(context)
            self.assertEqual("READY", plan["status"])
            self.assertFalse(plan["provenance"]["generator_is_user_supplied"])
            self.assertTrue(plan["provenance"]["generator_fingerprint"].startswith("sha256:"))
            self.assertIn(str(PLUGIN / "sqs.py"), plan["argv"])


if __name__ == "__main__":
    unittest.main()
