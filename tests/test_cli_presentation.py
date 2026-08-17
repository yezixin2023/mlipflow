from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from mlipflow.services import initialize

from .helpers import plugin_manifest, project_config, run_cli, write_json


class CliPresentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.plugins = self.root / "plugins"
        write_json(self.plugins / "demo" / "plugin.yaml", plugin_manifest())
        (self.plugins / "demo" / "adapter.py").write_text("class Adapter: pass\n", encoding="utf-8")

        expensive = plugin_manifest("expensive")
        expensive["implementation"].update(
            {"kind": "external-command", "status": "adapter-ready"}
        )
        expensive["execution"].update({"backends": ["local"], "cost_class": "expensive"})
        expensive["safety"] = {
            "expensive": True,
            "requires_approval_before_execution": True,
        }
        write_json(self.plugins / "expensive" / "plugin.yaml", expensive)
        (self.plugins / "expensive" / "adapter.py").write_text(
            "class Adapter:\n"
            "    def validate(self, context): return []\n"
            "    def plan(self, context):\n"
            "        return {'status': 'READY', 'executable': True, "
            "'argv': ['python', 'worker.py', '--steps', '10'], "
            "'cwd': context['attempt_dir'], 'diagnostics': []}\n",
            encoding="utf-8",
        )

        nodes = [
            {
                "id": "benchmark",
                "uses": "demo@1",
                "mode": "replay",
                "backend": "local",
                "inputs": {"result_manifest": "result.json"},
                "parameters": {"operation": "replay"},
                "resources": {},
            },
            {
                "id": "expensive-run",
                "uses": "expensive@1",
                "mode": "execute",
                "backend": "local",
                "inputs": {"dataset": "dataset.json"},
                "parameters": {"operation": "train", "steps": 10},
                "resources": {"cpus": 2, "memory": "4G"},
            },
        ]
        write_json(self.root / "project.yaml", project_config(nodes))
        write_json(
            self.root / "result.json",
            {
                "schema_version": 1,
                "status": "OK",
                "metrics": {"mae": 0.1},
                "artifacts": [{"role": "metrics", "path": "metrics.csv"}],
            },
        )
        (self.root / "metrics.csv").write_text("mae\n0.1\n", encoding="utf-8")
        write_json(self.root / "dataset.json", {"systems": 2})
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
                                    "fingerprint": "sha256:"
                                    + hashlib.sha256(evidence.read_bytes()).hexdigest(),
                                    "size_bytes": evidence.stat().st_size,
                                },
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
        return ["--project", str(self.root), "--plugins", str(self.plugins)]

    def _json(self, command: list[str]) -> dict:
        code, stdout, stderr = run_cli([*self.base, "--format", "json", *command])
        self.assertEqual(code, 0, stderr)
        return json.loads(stdout)

    def test_status_and_inspect_are_compact_but_audit_preserves_details(self) -> None:
        run_code, _, run_error = run_cli([*self.base, "run", "benchmark"])
        self.assertEqual(run_code, 0, run_error)

        status = self._json(["status", "benchmark"])
        self.assertFalse(status["audit"])
        self.assertIn("nodes", status["data"])
        serialized = json.dumps(status["data"])
        for forbidden in ("run_id", "manifest_path", "fingerprint", "sha256:", "file://"):
            self.assertNotIn(forbidden, serialized)

        status_audit = self._json(["status", "benchmark", "--audit"])
        self.assertTrue(status_audit["audit"])
        self.assertIn("run_id", status_audit["data"]["steps"][0])
        self.assertIn("fingerprint", status_audit["data"]["steps"][0]["artifacts"][0])

        inspected = self._json(["inspect", "benchmark"])["data"]
        self.assertNotIn("manifest", inspected["plugin"])
        self.assertEqual(inspected["inputs"], {"result_manifest": "result.json"})
        inspected_audit = self._json(["inspect", "benchmark", "--audit"])["data"]
        self.assertIn("manifest", inspected_audit["plugin"])

    def test_route_default_keeps_selection_and_reasons_only(self) -> None:
        args = [
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
        compact = self._json(args)["data"]
        self.assertEqual(compact["selected_model"], "deepmd")
        self.assertIn("missing elements", compact["rejected"][0]["reason"])
        serialized = json.dumps(compact)
        for forbidden in ("evidence_verification", "contributions", "evidence_status", "sha256:"):
            self.assertNotIn(forbidden, serialized)

        audit = self._json([*args, "--audit"])["data"]
        self.assertIn("evidence_verification", audit)
        self.assertIn("contributions", audit["ranking"][0])

    def test_dry_run_shows_token_only_when_required(self) -> None:
        replay = self._json(["run", "benchmark", "--dry-run"])["data"]
        self.assertFalse(replay["approval_required"])
        self.assertNotIn("approval_token", replay)
        self.assertNotIn("plan_digest", replay)
        self.assertNotIn("sha256:", json.dumps(replay))

        expensive = self._json(["run", "expensive-run", "--dry-run"])["data"]
        self.assertTrue(expensive["approval_required"])
        self.assertTrue(expensive["approval_token"].startswith("sha256:"))
        self.assertNotIn("adapter_plan", expensive)
        self.assertEqual(json.dumps(expensive).count("sha256:"), 1)

        audit = self._json(["run", "expensive-run", "--dry-run", "--audit"])["data"]
        self.assertIn("adapter_plan", audit)
        self.assertIn("input_identities", audit)

    def test_default_text_is_command_specific_not_json(self) -> None:
        commands = [
            (["status", "benchmark"], "READY"),
            (["inspect", "benchmark"], "action: replay"),
            (
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
                ],
                "selected model: deepmd",
            ),
            (["run", "benchmark", "--dry-run"], "approval: not required"),
        ]
        for command, expected in commands:
            code, stdout, stderr = run_cli([*self.base, *command])
            self.assertEqual(code, 0, stderr)
            self.assertIn(expected, stdout)
            self.assertFalse(stdout.lstrip().startswith("{"), stdout)
            self.assertNotIn('"schema_version"', stdout)


if __name__ == "__main__":
    unittest.main()
