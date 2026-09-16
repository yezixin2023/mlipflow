from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlipipe.backends import ExecutionResult
from mlipipe.config import load_project
from mlipipe.services.commands import initialize, make_retry_plan, make_run_plan, retry, run_node
from mlipipe.services.queries import query_workflow

from .helpers import project_config, snapshot, write_json


class FixtureAdapter:
    def __init__(self, argv: list[str]):
        self.argv = argv

    def operation(self, context):
        return "rank-candidates"

    def validate(self, context):
        return []

    def plan(self, context):
        return {
            "status": "READY",
            "diagnostics": self.validate(context),
            "executable": True,
            "argv": self.argv,
            "cwd": context["attempt_dir"],
        }

    def check(self, context):
        return {"status": "OK"}

    def collect(self, context):
        result = Path(context["attempt_dir"]) / "result.json"
        artifacts = (
            [{"role": "metric", "path": str(result)}] if result.is_file() else []
        )
        return {"status": "OK", "artifacts": artifacts, "metrics": {}}


class AdapterExecutionTests(unittest.TestCase):
    def test_retry_creates_fresh_attempts_and_keeps_old_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            node = {
                "id": "fails",
                "uses": "candidate-ranking",
                "backend": "local",
            }
            write_json(root / "project.yaml", project_config([node]))
            initialize(root)
            project = load_project(root)
            adapter = FixtureAdapter(
                [sys.executable, "-c", "raise SystemExit(7)"]
            )
            with patch(
                "mlipipe.services.commands.load_adapter", return_value=adapter
            ), patch(
                "mlipipe.services.execution.load_adapter", return_value=adapter
            ):
                first = run_node(project, "fails")
                first_run_id = first["step"]["run_id"]
                retry_plan = make_retry_plan(project, "fails")
                self.assertEqual(1, retry_plan["details"]["previous_attempt"])
                retry(project, "fails")
                second = run_node(project, "fails")
                self.assertEqual(2, second["step"]["attempt"])
                manifest = json.loads(
                    (
                        root
                        / ".mlipipe/runs/fails/attempt-2/run-manifest.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertNotEqual(first_run_id, second["step"]["run_id"])
                self.assertNotIn("retry", manifest)
                self.assertEqual("process exited 7", manifest["state_reason"])
                retry(project, "fails")
                third = run_node(project, "fails")
            self.assertEqual(3, third["step"]["attempt"])
            for attempt in (1, 2, 3):
                self.assertTrue(
                    (root / f".mlipipe/runs/fails/attempt-{attempt}").is_dir()
                )

    def test_local_four_method_lifecycle_and_small_run_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(
                root / "project.yaml",
                project_config(
                    [{"id": "ordered", "uses": "candidate-ranking"}]
                ),
            )
            project = load_project(root)
            before = snapshot(root)
            calls: list[str] = []

            class OrderedAdapter(FixtureAdapter):
                def validate(self, context):
                    calls.append("validate")
                    return []

                def plan(self, context):
                    calls.append("plan")
                    return super().plan(context)

                def check(self, context):
                    calls.append("check")
                    return {"status": "OK"}

                def collect(self, context):
                    calls.append("collect")
                    return super().collect(context)

            adapter = OrderedAdapter(["fixture-tool"])
            with patch(
                "mlipipe.services.commands.load_adapter", return_value=adapter
            ):
                plan = make_run_plan(project, "ordered")
            self.assertEqual("READY", plan["adapter_plan"]["status"])
            self.assertEqual(["plan", "validate"], calls)
            self.assertEqual(plan["adapter_plan"]["diagnostics"], plan["adapter_diagnostics"])
            self.assertEqual(before, snapshot(root))
            initialize(root)
            calls.clear()

            def execute(_argv, cwd, _environment=None):
                calls.append("process")
                write_json(cwd / "result.json", {"metrics": {"mae": 0.1}})
                return ExecutionResult(0, "fixture stdout\n", "")

            with patch(
                "mlipipe.services.commands.load_adapter", return_value=adapter
            ), patch(
                "mlipipe.services.execution.load_adapter", return_value=adapter
            ), patch("mlipipe.backends.LocalBackend.run", side_effect=execute):
                result = run_node(project, "ordered")

            self.assertEqual("OK", result["step"]["state"])
            self.assertEqual(
                ["plan", "validate", "process", "check", "collect"], calls
            )
            state = query_workflow(project)["steps"][0]
            self.assertEqual(
                {"stdout", "stderr", "metric"},
                {artifact["role"] for artifact in state["artifacts"]},
            )
            manifest = json.loads(
                Path(state["manifest_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(
                {
                    "project_id",
                    "node_id",
                    "run_id",
                    "attempt",
                    "capability",
                    "state",
                    "state_reason",
                    "mode",
                    "backend",
                    "backend_profile",
                    "job",
                    "remote_dir",
                    "timestamps",
                    "command",
                    "artifacts",
                    "result",
                },
                set(manifest),
            )


if __name__ == "__main__":
    unittest.main()
