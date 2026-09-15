from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mlipflow.services.commands import initialize

from .helpers import project_config, run_cli, write_json


class CliPresentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        write_json(
            self.root / "project.yaml",
            project_config(
                [
                    {
                        "id": "benchmark",
                        "uses": "candidate-ranking",
                        "mode": "replay",
                        "backend": "local",
                        "inputs": {"result_manifest": "result.json"},
                    }
                ]
            ),
        )
        write_json(
            self.root / "result.json",
            {
                "schema_version": 1,
                "status": "OK",
                "artifacts": [{"role": "metrics", "path": "metrics.csv"}],
            },
        )
        (self.root / "metrics.csv").write_text("mae\n0.1\n", encoding="utf-8")
        (self.root / "benchmark.csv").write_text(
            "model,diffusion_mae\ndeepmd,0.1\n", encoding="utf-8"
        )
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
                                "artifact": {"uri": "benchmark.csv"},
                                "metrics": {"diffusion_mae": 0.1},
                            }
                        ],
                    },
                    {
                        "id": "missing-elements",
                        "elements": ["Li"],
                        "tasks": ["ionic-transport"],
                        "benchmarks": [],
                    },
                ],
            },
        )
        initialize(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @property
    def base(self) -> list[str]:
        return ["--project", str(self.root)]

    def _json(self, command: list[str]) -> dict:
        code, stdout, stderr = run_cli(
            [*self.base, "--format", "json", *command]
        )
        self.assertEqual(code, 0, stderr)
        return json.loads(stdout)

    def test_json_status_and_inspect_keep_actionable_details(self) -> None:
        run_code, _, run_error = run_cli([*self.base, "run", "benchmark"])
        self.assertEqual(run_code, 0, run_error)

        status = self._json(["status", "benchmark"])["data"]
        self.assertIn("nodes", status)
        serialized = json.dumps(status)
        for removed in ("run_id", "provenance", "events"):
            self.assertNotIn(removed, serialized)
        self.assertEqual("metrics", status["nodes"][0]["artifacts"][0]["role"])
        self.assertTrue(Path(status["nodes"][0]["artifacts"][0]["path"]).is_file())
        self.assertTrue(Path(status["nodes"][0]["manifest_path"]).is_file())

        inspected = self._json(["inspect", "benchmark"])["data"]
        self.assertEqual("candidate-ranking", inspected["capability"]["id"])
        self.assertFalse(inspected["approval_required"])
        self.assertEqual([], inspected["capability"]["approval_operations"])
        self.assertNotIn("approval_required", inspected["capability"])
        self.assertEqual(
            {"result_manifest": "result.json"}, inspected["inputs"]
        )

    def test_route_default_keeps_selection_and_reasons_only(self) -> None:
        compact = self._json(
            [
                "route",
                "--task",
                "ionic-transport",
                "--elements",
                "Li",
                "P",
                "S",
                "--scenario",
                "fixture",
            ]
        )["data"]
        self.assertEqual("deepmd", compact["selected_model"])
        self.assertIn("missing elements", compact["rejected"][0]["reason"])

    def test_replay_dry_run_shows_no_approval_requirement(self) -> None:
        replay = self._json(["run", "benchmark", "--dry-run"])["data"]
        self.assertFalse(replay["approval_required"])
        self.assertEqual(
            {"kind": "replay", "summary": "read existing results; no numerical program"},
            replay["will_run"],
        )

    def test_default_text_is_command_specific_not_json(self) -> None:
        commands = [
            (["status", "benchmark"], "READY"),
            (["inspect", "benchmark"], "action: replay"),
            (["run", "benchmark", "--dry-run"], "approval: not required"),
        ]
        for command, expected in commands:
            code, stdout, stderr = run_cli([*self.base, *command])
            self.assertEqual(code, 0, stderr)
            self.assertIn(expected, stdout)
            self.assertFalse(stdout.lstrip().startswith("{"), stdout)


if __name__ == "__main__":
    unittest.main()
