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

from .helpers import plugin_manifest, project_config, write_json


def _approved_project(root: Path) -> tuple[Path, object]:
    plugins = root / "plugins"
    manifest = plugin_manifest()
    manifest["implementation"].update(
        {"kind": "external-command", "status": "adapter-ready"}
    )
    manifest["safety"] = {
        "expensive": True,
        "requires_approval_before_execution": True,
    }
    write_json(plugins / "demo" / "plugin.yaml", manifest)
    (plugins / "demo" / "adapter.py").write_text(
        "class Adapter:\n"
        "    def validate(self, context): return []\n"
        "    def plan(self, context):\n"
        "        return {'status': 'READY', 'executable': True, "
        "'argv': ['true'], 'cwd': context['attempt_dir']}\n"
        "    def check(self, context):\n"
        "        return {'status': 'OK', 'diagnostics': []}\n"
        "    def collect(self, context):\n"
        "        return {'status': 'OK', 'artifacts': [], 'metrics': {}, "
        "'diagnostics': []}\n",
        encoding="utf-8",
    )
    write_json(
        root / "project.yaml",
        project_config(
            [
                {"id": "x", "uses": "demo@1"},
                {"id": "y", "uses": "demo@1"},
            ]
        ),
    )
    initialize(root)
    return plugins, load_project(root)


class ApprovalTests(unittest.TestCase):
    def test_expensive_run_requires_explicit_boolean_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugins, project = _approved_project(root)
            plan = make_run_plan(project, "x", plugins)
            self.assertTrue(plan["approval_required"])
            with self.assertRaises(ApprovalError):
                run_node(project, "x", plugins)
            self.assertFalse((root / ".mlipflow/runs/x/attempt-1").exists())

            with patch(
                "mlipflow.services.LocalBackend.run",
                return_value=ExecutionResult(0, "", ""),
            ):
                result = run_node(project, "x", plugins, approval=True)
            self.assertEqual("OK", result["step"]["state"])

    def test_attempt_and_node_are_plain_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugins, project = _approved_project(root)
            first = make_run_plan(project, "x", plugins)
            other = make_run_plan(project, "y", plugins)
            with StateStore(state_path(project), readonly=False) as store:
                step = store.latest_step(project.project_id, "x")
                store.transition(step.run_id, RunState.RUNNING)
                store.transition(step.run_id, RunState.FAIL)
            retry(project, "x")
            second = make_run_plan(project, "x", plugins)
            self.assertEqual(("x", 1), (first["node_id"], first["attempt"]))
            self.assertEqual(("y", 1), (other["node_id"], other["attempt"]))
            self.assertEqual(("x", 2), (second["node_id"], second["attempt"]))


if __name__ == "__main__":
    unittest.main()
