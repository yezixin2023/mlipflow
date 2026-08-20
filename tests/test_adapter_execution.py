from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlipflow.backends import ExecutionResult
from mlipflow.config import load_project
from mlipflow.errors import PluginError, StateError
from mlipflow.services import (
    initialize,
    make_retry_plan,
    make_run_plan,
    query_workflow,
    retry,
    run_node,
    _normalize_adapter_artifacts,
)

from .helpers import plugin_manifest, project_config, snapshot, write_json


ADAPTER_SOURCE = '''
import json
from pathlib import Path

class Adapter:
    def validate(self, context):
        return []

    def plan(self, context):
        return {
            "status": "READY",
            "executable": True,
            "argv": ["fixture-tool", "--output", "result.json"],
            "expected_outputs": ["result.json"],
        }

    def prepare(self, context, plan):
        return {"status": "READY"}

    def check(self, context):
        path = Path(context["attempt_dir"]) / "result.json"
        return {"status": "OK" if path.is_file() else "FAIL"}

    def collect(self, context):
        path = Path(context["attempt_dir"]) / "result.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        return {
            "status": "OK",
            "artifacts": [{"role": "metric", "path": str(path)}],
            "metrics": value["metrics"],
        }

    def replay(self, context):
        return {"status": "OK", "executable": False, "result_manifest": context.get("result_manifest")}
'''


class AdapterExecutionTests(unittest.TestCase):
    def test_adapter_file_artifacts_cannot_escape_or_use_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            attempt = root / "attempt"
            attempt.mkdir()
            outside = root / "outside.dat"
            outside.write_text("outside\n", encoding="utf-8")
            project = type(
                "FixtureProject",
                (),
                {"raw": {}},
            )()
            with self.assertRaisesRegex(PluginError, "escapes"):
                _normalize_adapter_artifacts(
                    project, attempt, [{"role": "bad", "path": str(outside)}]
                )
            link = attempt / "link.dat"
            link.symlink_to(outside)
            with self.assertRaisesRegex(PluginError, "symlink"):
                _normalize_adapter_artifacts(
                    project, attempt, [{"role": "bad", "path": str(link)}]
                )

    def test_adapter_and_command_paths_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugins = root / "plugins"
            manifest = plugin_manifest()
            manifest["implementation"]["status"] = "adapter-ready"
            manifest["implementation"]["kind"] = "external-command"
            write_json(plugins / "demo" / "plugin.yaml", manifest)
            adapter_path = plugins / "demo" / "adapter.py"
            adapter_path.write_text(
                """\
class Adapter:
    def validate(self, context): return []
    def plan(self, context):
        return {"status": "READY", "executable": True,
                "argv": [context["inputs"]["script"]], "cwd": context["attempt_dir"]}
""",
                encoding="utf-8",
            )
            worker = root / "worker.py"
            worker.write_text("print('one')\n", encoding="utf-8")
            write_json(
                root / "project.yaml",
                project_config(
                    [
                        {
                            "id": "bound",
                            "uses": "demo@1",
                            "inputs": {"script": str(worker)},
                        }
                    ]
                ),
            )
            project = load_project(root)
            first = make_run_plan(project, "bound", plugins)
            worker.write_text("print('two')\n", encoding="utf-8")
            second = make_run_plan(project, "bound", plugins)
            self.assertEqual(first, second)
            adapter_path.write_text(
                adapter_path.read_text(encoding="utf-8") + "\n# reviewed change\n",
                encoding="utf-8",
            )
            third = make_run_plan(project, "bound", plugins)
            self.assertEqual(second, third)
            self.assertEqual(str(worker), third["adapter_plan"]["argv"][0])

    def test_retry_limit_and_manifest_lineage_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugins = root / "plugins"
            manifest = plugin_manifest()
            manifest["implementation"]["kind"] = "external-command"
            manifest["implementation"]["status"] = "adapter-ready"
            manifest["execution"]["backends"] = ["local"]
            manifest["retry"]["max_attempts"] = 2
            write_json(plugins / "demo" / "plugin.yaml", manifest)
            (plugins / "demo" / "adapter.py").write_text(
                """\
class Adapter:
    def validate(self, context): return []
    def plan(self, context):
        return {"status": "READY", "executable": True,
                "argv": context["parameters"]["argv"], "cwd": context["attempt_dir"]}
""",
                encoding="utf-8",
            )
            node = {
                "id": "fails",
                "uses": "demo@1",
                "backend": "local",
                "parameters": {"argv": [sys.executable, "-c", "raise SystemExit(7)"]},
            }
            write_json(root / "project.yaml", project_config([node]))
            initialize(root)
            project = load_project(root)
            first = run_node(project, "fails", plugins)
            first_run_id = first["step"]["run_id"]
            retry_plan = make_retry_plan(project, "fails", plugins)
            self.assertEqual(1, retry_plan["details"]["previous_attempt"])
            retry(project, "fails", plugins)
            second = run_node(project, "fails", plugins)
            self.assertEqual(2, second["step"]["attempt"])
            second_manifest = json.loads(
                Path(second["step"]["manifest_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(second_manifest["retry"]["previous_run_id"], first_run_id)
            with self.assertRaisesRegex(StateError, "retry limit"):
                make_retry_plan(project, "fails", plugins)

    def test_adapter_plan_is_zero_write_and_approved_execution_collects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugins = root / "plugins"
            manifest = plugin_manifest()
            manifest["implementation"]["status"] = "adapter-ready"
            manifest["implementation"]["kind"] = "external-command"
            manifest["execution"]["backends"] = ["local"]
            write_json(plugins / "demo" / "plugin.yaml", manifest)
            (plugins / "demo" / "adapter.py").write_text(ADAPTER_SOURCE, encoding="utf-8")
            node = {
                "id": "adapter",
                "uses": "demo@1",
                "mode": "execute",
                "backend": "local",
                "inputs": {},
                "parameters": {},
            }
            write_json(root / "project.yaml", project_config([node]))
            project = load_project(root)
            before = snapshot(root)
            plan = make_run_plan(project, "adapter", plugins)
            self.assertEqual(plan["adapter_plan"]["status"], "READY")
            self.assertEqual(before, snapshot(root))
            initialize(root)

            def execute(_argv, cwd, _environment=None):
                write_json(cwd / "result.json", {"metrics": {"mae": 0.1}})
                return ExecutionResult(0, "fixture stdout\n", "")

            with patch("mlipflow.services.LocalBackend.run", side_effect=execute):
                result = run_node(project, "adapter", plugins)
            self.assertEqual(result["step"]["state"], "OK")
            state = query_workflow(project)["steps"][0]
            self.assertEqual({"stdout", "stderr", "metric"}, {a["role"] for a in state["artifacts"]})
            manifest_data = json.loads(Path(state["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest_data["command"][0], "fixture-tool")
            self.assertEqual(manifest_data["metrics"]["mae"], 0.1)


if __name__ == "__main__":
    unittest.main()
