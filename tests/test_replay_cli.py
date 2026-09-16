from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .helpers import project_config, run_cli, write_json


class ReplayCliTests(unittest.TestCase):
    def test_replay_references_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "evidence.csv"
            artifact.write_text("metric,value\nforce_rmse,0.1\n", encoding="utf-8")
            write_json(
                root / "result.json",
                {
                    "schema_version": 1,
                    "status": "OK",
                    "metrics": {"force_rmse": {"value": 0.1, "unit": "eV/angstrom"}},
                    "artifacts": [{"role": "metrics-table", "path": "evidence.csv"}],
                },
            )
            node = {
                "id": "benchmark",
                "uses": "candidate-ranking",
                "mode": "replay",
                "needs": [],
                "inputs": {"result_manifest": "result.json"},
                "parameters": {},
                "backend": "local",
                "resources": {},
            }
            write_json(root / "project.yaml", project_config([node]))
            code, _, stderr = run_cli(["--project", str(root), "init"])
            self.assertEqual(code, 0, stderr)
            base = ["--project", str(root), "--format", "json"]
            code, stdout, stderr = run_cli([*base, "run", "benchmark", "--dry-run"])
            self.assertEqual(code, 0, stderr)
            self.assertFalse(json.loads(stdout)["data"]["approval_required"])
            code, stdout, stderr = run_cli([*base, "run", "benchmark"])
            self.assertEqual(code, 0, stderr)
            result = json.loads(stdout)["data"]
            self.assertEqual(result["state"], "OK")
            self.assertNotIn("step", result)
            manifest = root / ".mlipipe/runs/benchmark/attempt-1/run-manifest.json"
            self.assertTrue(manifest.is_file())
            manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertTrue(
                {"provenance", "retry", "dependencies", "resources"}.isdisjoint(
                    manifest_data
                )
            )
            self.assertEqual(artifact.read_text(encoding="utf-8"), "metric,value\nforce_rmse,0.1\n")
            self.assertFalse((manifest.parent / artifact.name).exists())
            code, status, stderr = run_cli([*base, "json", "benchmark"])
            self.assertEqual(code, 0, stderr)
            self.assertEqual(json.loads(status)["data"]["nodes"][0]["state"], "OK")

    def test_replay_refuses_explicit_failed_scientific_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(
                root / "result.json",
                {"schema_version": 1, "status": "FAIL", "metrics": {}, "artifacts": []},
            )
            write_json(
                root / "project.yaml",
                project_config(
                    [
                        {
                            "id": "failed-result",
                            "uses": "candidate-ranking",
                            "mode": "replay",
                            "inputs": {"result_manifest": "result.json"},
                        }
                    ]
                ),
            )
            base = ["--project", str(root), "--format", "json"]
            code, _, stderr = run_cli(["--project", str(root), "init"])
            self.assertEqual(code, 0, stderr)
            code, stdout, stderr = run_cli([*base, "run", "failed-result", "--dry-run"])
            self.assertEqual(code, 0, stderr)
            self.assertFalse(json.loads(stdout)["data"]["approval_required"])
            code, _, stderr = run_cli([*base, "run", "failed-result"])
            self.assertEqual(code, 2)
            self.assertIn("not scientifically successful", stderr)

if __name__ == "__main__":
    unittest.main()
