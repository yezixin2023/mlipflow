from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlipflow.backends import ExecutionResult
from mlipflow.config import load_project
from mlipflow.errors import ApprovalError
from mlipflow.services import initialize, make_run_plan, retry, run_node, state_path
from mlipflow.state import RunState, StateStore

from .helpers import project_config, write_json


class FixtureAdapter:
    def validate(self, context):
        return []

    def plan(self, context):
        return {
            "status": "READY",
            "executable": True,
            "argv": ["true"],
            "cwd": context["attempt_dir"],
        }

    def check(self, context):
        return {"status": "OK"}

    def collect(self, context):
        return {"status": "OK", "artifacts": [], "metrics": {}}


class ApprovalTests(unittest.TestCase):
    def test_expensive_run_requires_explicit_boolean_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(
                root / "project.yaml",
                project_config([{"id": "train", "uses": "mlip-training"}]),
            )
            initialize(root)
            project = load_project(root)
            adapter = FixtureAdapter()
            with patch(
                "mlipflow.services.commands.load_adapter", return_value=adapter
            ):
                plan = make_run_plan(project, "train")
                self.assertTrue(plan["approval_required"])
                with self.assertRaises(ApprovalError):
                    run_node(project, "train")
            self.assertFalse((root / ".mlipflow/runs/train/attempt-1").exists())

            with patch(
                "mlipflow.services.commands.load_adapter", return_value=adapter
            ), patch(
                "mlipflow.services.execution.load_adapter", return_value=adapter
            ), patch(
                "mlipflow.services.LocalBackend.run",
                return_value=ExecutionResult(0, "", ""),
            ):
                result = run_node(project, "train", approval=True)
            self.assertEqual("OK", result["step"]["state"])

    def test_retry_uses_attempt_number_as_its_only_counter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(
                root / "project.yaml",
                project_config(
                    [
                        {"id": "x", "uses": "candidate-ranking", "mode": "replay"},
                        {"id": "y", "uses": "candidate-ranking", "mode": "replay"},
                    ]
                ),
            )
            initialize(root)
            project = load_project(root)
            first = make_run_plan(project, "x")
            other = make_run_plan(project, "y")
            with StateStore(state_path(project), readonly=False) as store:
                step = store.latest_step(project.project_id, "x")
                store.transition(step.run_id, RunState.RUNNING)
                store.transition(step.run_id, RunState.FAIL)
            retry(project, "x")
            second = make_run_plan(project, "x")
            self.assertEqual(("x", 1), (first["node_id"], first["attempt"]))
            self.assertEqual(("y", 1), (other["node_id"], other["attempt"]))
            self.assertEqual(("x", 2), (second["node_id"], second["attempt"]))


if __name__ == "__main__":
    unittest.main()
