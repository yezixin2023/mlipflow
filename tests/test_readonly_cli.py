from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlipflow.services import initialize

from .helpers import plugin_manifest, project_config, run_cli, snapshot, write_json


class ReadOnlyCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.plugins = self.root / "plugins"
        write_json(self.plugins / "demo" / "plugin.yaml", plugin_manifest())
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
        write_json(self.root / "project.yaml", project_config([node]))
        evidence = self.root / "benchmark.csv"
        evidence.write_text("model,diffusion_mae\ndeepmd,0.1\n", encoding="utf-8")
        write_json(
            self.root / "model_registry.yaml",
            {
                "schema_version": 1,
                "models": [
                    {
                        "id": "deepmd",
                        "elements": ["Li", "P", "S"],
                        "tasks": ["ionic-transport"],
                        "benchmarks": [
                            {
                                "task": "ionic-transport",
                                "scenario": "fixture",
                                "run_id": "r1",
                                "validation_samples": 5,
                                "artifact": {
                                    "uri": "benchmark.csv",
                                },
                                "metrics": {"diffusion_mae": 0.1},
                            }
                        ],
                    }
                ],
            },
        )
        write_json(self.root / "result.json", {"schema_version": 1, "metrics": {}, "artifacts": []})

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_uninitialized_queries_create_nothing(self) -> None:
        before = snapshot(self.root)
        commands = [
            ["list"],
            ["status"],
            ["json"],
            ["inspect", "benchmark"],
            ["route", "--task", "ionic-transport", "--elements", "Li", "P", "S", "--scenario", "fixture"],
            ["doctor"],
        ]
        with patch("mlipflow.backends.subprocess.run", side_effect=AssertionError("backend invoked")):
            for command in commands:
                code, _, stderr = run_cli(
                    [
                        "--project",
                        str(self.root),
                        "--plugins",
                        str(self.plugins),
                        "--format",
                        "json",
                        *command,
                    ]
                )
                self.assertIn(code, {0, 1}, stderr)
        self.assertEqual(snapshot(self.root), before)
        self.assertFalse((self.root / ".mlipflow").exists())

    def test_initialized_queries_and_logs_are_zero_write(self) -> None:
        initialize(self.root)
        log_dir = self.root / ".mlipflow" / "runs" / "benchmark" / "attempt-1"
        log_dir.mkdir(parents=True)
        (log_dir / "stdout.log").write_text("one\ntwo\n", encoding="utf-8")
        before = snapshot(self.root)
        commands = [["list"], ["status"], ["json"], ["inspect", "benchmark"], ["logs", "benchmark"], ["doctor"]]
        for command in commands:
            code, _, stderr = run_cli(
                [
                    "--project",
                    str(self.root),
                    "--plugins",
                    str(self.plugins),
                    "--format",
                    "json",
                    *command,
                ]
            )
            self.assertIn(code, {0, 1}, stderr)
        self.assertEqual(snapshot(self.root), before)

    def test_run_dry_run_is_zero_write(self) -> None:
        before = snapshot(self.root)
        code, stdout, stderr = run_cli(
            [
                "--project",
                str(self.root),
                "--plugins",
                str(self.plugins),
                "--format",
                "json",
                "run",
                "benchmark",
                "--dry-run",
            ]
        )
        self.assertEqual(code, 0, stderr)
        compact = json.loads(stdout)["data"]
        self.assertFalse(compact["approval_required"])
        self.assertNotIn("adapter_plan", compact)

        code, stdout, stderr = run_cli(
            [
                "--project",
                str(self.root),
                "--plugins",
                str(self.plugins),
                "--format",
                "json",
                "run",
                "benchmark",
                "--dry-run",
                "--audit",
            ]
        )
        self.assertEqual(code, 0, stderr)
        audit = json.loads(stdout)["data"]
        self.assertEqual({"result_manifest": "result.json"}, audit["inputs"])
        self.assertEqual(snapshot(self.root), before)


if __name__ == "__main__":
    unittest.main()
