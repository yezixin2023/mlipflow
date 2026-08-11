"""Numerical and format parity tests for the manuscript-era Li10 screen."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "composition-screening"


def _adapter():
    path = PLUGIN / "adapter.py"
    spec = importlib.util.spec_from_file_location("manuscript_screen_adapter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Adapter()


class ManuscriptScreeningTests(unittest.TestCase):
    def test_legacy_element_orders_and_real_top_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "output.txt"
            candidates.write_text(
                "\n".join(
                    [
                        "7,3,4,8,6 -> inventory://POSCAR_Li10_Zn7Fe3Cu4Ni8Mn6",
                        "6,2,9,9,2 -> inventory://POSCAR_Li10_Zn6Fe2Cu9Ni9Mn2",
                        "3,8,6,7,4 -> inventory://POSCAR_Li10_Zn3Fe8Cu6Ni7Mn4",
                        "2,5,8,6,7 -> inventory://POSCAR_Li10_Zn2Fe5Cu8Ni6Mn7",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            metrics = root / "sigma.txt"
            metrics.write_text(
                "\n".join(
                    [
                        "Mn6_Fe3_Ni8_Cu4_Zn7 18.420176685825222",
                        "Mn2_Fe2_Ni9_Cu9_Zn6 17.626045819654934",
                        "Mn4_Fe8_Ni7_Cu6_Zn3 14.282118728392",
                        "Mn7_Fe5_Ni6_Cu8_Zn2 14.034032196953927",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            candidate_manifest = root / "candidates.json"
            transport_manifest = root / "transport.json"
            subprocess.run(
                [
                    sys.executable,
                    str(PLUGIN / "normalize_legacy.py"),
                    "--candidate-list",
                    str(candidates),
                    "--metric-file",
                    str(metrics),
                    "--metric-name",
                    "ionic_conductivity_300k_s_per_m",
                    "--metric-unit",
                    "S/m",
                    "--candidate-manifest",
                    str(candidate_manifest),
                    "--transport-results-manifest",
                    str(transport_manifest),
                ],
                check=True,
            )
            normalized = json.loads(candidate_manifest.read_text(encoding="utf-8"))
            self.assertEqual("Mn6_Fe3_Ni8_Cu4_Zn7", normalized["candidates"][2]["id"])
            transport = json.loads(transport_manifest.read_text(encoding="utf-8"))
            rendered = candidate_manifest.read_text(encoding="utf-8") + transport_manifest.read_text(
                encoding="utf-8"
            )
            self.assertNotIn(str(root), rendered)
            self.assertEqual(
                "output.txt", normalized["provenance"]["candidate_sources"][0]["locator"]
            )
            self.assertEqual(
                "sigma.txt", transport["results"][0]["source_records"][0]["locator"]
            )
            values = {
                item["candidate_id"]: item["metrics"]["ionic_conductivity_300k_s_per_m"]
                for item in transport["results"]
            }
            self.assertEqual(18.420176685825222, values["Mn6_Fe3_Ni8_Cu4_Zn7"])

    def test_bundled_screen_executes_and_adapter_rechecks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            attempt = project / "attempt"
            attempt.mkdir()
            (project / "candidates.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "candidates": [
                            {"id": "Mn2_Fe2_Ni9_Cu9_Zn6"},
                            {"id": "Mn6_Fe3_Ni8_Cu4_Zn7"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (project / "transport.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "results": [
                            {
                                "candidate_id": "Mn2_Fe2_Ni9_Cu9_Zn6",
                                "metrics": {"ionic_conductivity_300k_s_per_m": 17.626045819654934},
                            },
                            {
                                "candidate_id": "Mn6_Fe3_Ni8_Cu4_Zn7",
                                "metrics": {"ionic_conductivity_300k_s_per_m": 18.420176685825222},
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            context = {
                "project_root": str(project),
                "attempt_dir": str(attempt),
                "inputs": {
                    "candidate_manifest": "candidates.json",
                    "transport_results_manifest": "transport.json",
                },
                "parameters": {
                    "metric": "ionic_conductivity_300k_s_per_m",
                    "direction": "maximize",
                    "top_k": 1,
                    "missing_metric_policy": "reject",
                },
                "backend": "local",
                "resources": {"cpus": 1},
            }
            adapter = _adapter()
            plan = adapter.plan(context)
            self.assertEqual("READY", plan["status"])
            self.assertTrue(plan["implementation"]["bundled"])
            subprocess.run(plan["argv"], cwd=plan["cwd"], check=True)
            self.assertEqual("OK", adapter.check(context)["status"])
            result = json.loads((attempt / "screening-result.json").read_text(encoding="utf-8"))
            self.assertEqual("Mn6_Fe3_Ni8_Cu4_Zn7", result["ranked_candidates"][0]["candidate_id"])


if __name__ == "__main__":
    unittest.main()
