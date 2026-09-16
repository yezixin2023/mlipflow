from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlipipe.services.commands import initialize
from mlipipe.config import load_project
from mlipipe.services.paths import state_path
from mlipipe.services import queries
from mlipipe.state import StateStore, RunState
from .helpers import project_config, run_cli, snapshot, write_json


class ReadOnlyCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        node = {
            "id": "benchmark",
            "uses": "mlip-benchmark",
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
        with patch("mlipipe.backends.subprocess.run", side_effect=AssertionError("backend invoked")):
            for command in commands:
                code, _, stderr = run_cli(
                    [
                        "--project",
                        str(self.root),
                        "--format",
                        "json",
                        *command,
                    ]
                )
                self.assertIn(code, {0, 1}, stderr)
        self.assertEqual(snapshot(self.root), before)
        self.assertFalse((self.root / ".mlipipe").exists())

    def test_single_node_queries_read_only_its_attempt(self) -> None:
        node = load_project(self.root).nodes[0]
        write_json(self.root / "project.yaml", project_config([
            {**node, "id": node_id} for node_id in ("first", "benchmark", "last")
        ]))
        initialize(self.root)
        project = load_project(self.root)
        self.assertEqual(3, len(queries.query_workflow(project)["steps"]))
        for query in (queries.query_workflow, queries.query_inspect):
            with self.subTest(query=query.__name__), patch.object(
                queries, "attempt_details", wraps=queries.attempt_details
            ) as details, patch.object(
                StateStore, "latest_step", autospec=True, side_effect=StateStore.latest_step
            ) as latest:
                query(project, "benchmark")
                details.assert_called_once_with(project, "benchmark", 1)
                self.assertEqual(["benchmark"], [call.args[2] for call in latest.call_args_list])

    def test_initialized_queries_and_logs_are_zero_write(self) -> None:
        initialize(self.root)
        log_dir = self.root / ".mlipipe" / "runs" / "benchmark" / "attempt-1"
        log_dir.mkdir(parents=True)
        (log_dir / "stdout.log").write_text("one\ntwo\n", encoding="utf-8")
        before = snapshot(self.root)
        commands = [["list"], ["status"], ["json"], ["inspect", "benchmark"], ["logs", "benchmark"], ["doctor"]]
        for command in commands:
            code, _, stderr = run_cli(
                [
                    "--project",
                    str(self.root),
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

        self.assertEqual(snapshot(self.root), before)

    def test_queries_never_advance_pending_or_failed_scheduler_jobs(self) -> None:
        config = project_config([{
            "id": "benchmark", "uses": "dft-labeling", "backend": "ssh-slurm",
            "backend_profile": "configured", "parameters": {"operation": "label"},
            "resources": {"cpus": 1, "gpus": 0, "memory": "1G", "walltime": "00:01:00"},
        }])
        write_json(self.root / "project.yaml", config)
        site = self.root / "site.yaml"
        write_json(site, {"schema_version": 1, "clusters": {"configured": {
            "backend": "ssh-slurm", "ssh_profile": "test-login",
            "remote_template_root": "/templates", "work_root": "/work",
        }}})
        initialize(self.root)
        project = load_project(self.root)
        with StateStore(state_path(project), readonly=False) as store:
            step = store.latest_step(project.project_id, "benchmark")
            store.bind_attempt(step.run_id, "ssh-slurm")
            store.transition(step.run_id, RunState.SUBMITTED, job_id="123")
            store.transition(step.run_id, RunState.PENDING)
        commands = [["list"], ["status"], ["json"], ["inspect", "benchmark"],
                    ["logs", "benchmark"], ["doctor"],
                    ["route", "--task", "ionic-transport", "--elements", "Li", "P", "S",
                     "--scenario", "fixture"]]
        for state in (RunState.PENDING, RunState.FAIL):
            if state == RunState.FAIL:
                with StateStore(state_path(project), readonly=False) as store:
                    store.transition(step.run_id, RunState.FAIL, diagnostic="fixture failure")
            before = snapshot(self.root)
            mtimes = {p: p.stat().st_mtime_ns for p in self.root.rglob("*")}
            with patch("mlipipe.backends.subprocess.run", side_effect=AssertionError("backend invoked")), patch.object(
                StateStore, "transition", side_effect=AssertionError("state mutation")
            ), patch.object(StateStore, "initialize_project", side_effect=AssertionError("initialization")):
                for command in commands:
                    code, out, err = run_cli(["--project", str(self.root), "--site", str(site),
                                               "--format", "json", *command])
                    self.assertEqual(code, 0, err)
                    if command[0] in {"status", "json"}:
                        self.assertEqual(state.value, json.loads(out)["data"]["nodes"][0]["state"])
            self.assertEqual(snapshot(self.root), before)
            self.assertEqual({p: p.stat().st_mtime_ns for p in self.root.rglob("*")}, mtimes)


if __name__ == "__main__":
    unittest.main()
