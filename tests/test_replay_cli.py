from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .helpers import plugin_manifest, project_config, run_cli, write_json


class ReplayCliTests(unittest.TestCase):
    def test_replay_references_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugins = root / "plugins"
            write_json(plugins / "demo" / "plugin.yaml", plugin_manifest())
            (plugins / "demo" / "adapter.py").write_text(
                "raise AssertionError('generic replay imported the Adapter')\n",
                encoding="utf-8",
            )
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
                "uses": "demo@1",
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
            base = ["--project", str(root), "--plugins", str(plugins), "--format", "json"]
            code, stdout, stderr = run_cli([*base, "run", "benchmark", "--dry-run"])
            self.assertEqual(code, 0, stderr)
            self.assertFalse(json.loads(stdout)["data"]["approval_required"])
            code, stdout, stderr = run_cli([*base, "run", "benchmark"])
            self.assertEqual(code, 0, stderr)
            result = json.loads(stdout)["data"]
            self.assertEqual(result["state"], "OK")
            self.assertNotIn("step", result)
            code, status, stderr = run_cli([*base, "json", "benchmark", "--audit"])
            self.assertEqual(code, 0, stderr)
            manifest = Path(json.loads(status)["data"]["steps"][0]["manifest_path"])
            self.assertTrue(manifest.is_file())
            try:
                import jsonschema
            except ImportError:
                jsonschema = None
            if jsonschema is not None:
                schema_path = Path(__file__).resolve().parents[1] / "schemas" / "run-manifest.schema.json"
                jsonschema.Draft202012Validator(
                    json.loads(schema_path.read_text(encoding="utf-8")),
                    format_checker=jsonschema.FormatChecker(),
                ).validate(json.loads(manifest.read_text(encoding="utf-8")))
            self.assertEqual(artifact.read_text(encoding="utf-8"), "metric,value\nforce_rmse,0.1\n")
            self.assertFalse((manifest.parent / artifact.name).exists())
            code, status, stderr = run_cli([*base, "json", "benchmark"])
            self.assertEqual(code, 0, stderr)
            self.assertEqual(json.loads(status)["data"]["nodes"][0]["state"], "OK")

    def test_replay_refuses_explicit_failed_scientific_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugins = root / "plugins"
            write_json(plugins / "demo" / "plugin.yaml", plugin_manifest())
            (plugins / "demo" / "adapter.py").write_text(
                "raise AssertionError('generic replay imported the Adapter')\n",
                encoding="utf-8",
            )
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
                            "uses": "demo@1",
                            "mode": "replay",
                            "inputs": {"result_manifest": "result.json"},
                        }
                    ]
                ),
            )
            base = ["--project", str(root), "--plugins", str(plugins), "--format", "json"]
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
