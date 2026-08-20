"""Plans store portable paths and ordinary calculation records."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlipflow.backends import ExecutionResult
from mlipflow.config import load_project
from mlipflow.portable import (
    ATTEMPT_DIR_TOKEN as ATTEMPT_TOKEN,
    TOKENS,
    PortableRoots,
    contains_absolute_root,
    to_runtime,
)
from mlipflow.services import make_run_plan, run_node

from .helpers import project_config, write_json


ROOT = Path(__file__).resolve().parents[1]
PLUGINS = ROOT / "plugins"
EXAMPLE = ROOT / "examples" / "high_entropy_sulfide"
def build_training_project(root: Path) -> None:
    shutil.copytree(PLUGINS, root / "plugins")
    (root / "wrappers").mkdir()
    (root / "configs").mkdir()
    (root / "data" / "labeled").mkdir(parents=True)
    (root / "wrappers" / "mace_train.py").write_text(
        "# test wrapper\n", encoding="utf-8"
    )
    write_json(root / "configs" / "mace.json", {})
    node = {
        "id": "n",
        "uses": "mlip-training@0",
        "mode": "execute",
        "inputs": {
            "executable": "/usr/bin/python3",
            "script": "wrappers/mace_train.py",
            "config": "configs/mace.json",
            "data": "data/labeled",
            "output": "model.bin",
            "result_manifest": "training-result.json",
        },
        "parameters": {
            "framework": "mace",
            "operation": "train",
            "seed": 20260810,
            "device": "cpu",
            "precision": "float64",
        },
    }
    write_json(root / "project.yaml", project_config([node]))


class PlanPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name).resolve() / "project"
        self.root.mkdir(parents=True)
        build_training_project(self.root)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_ready_plan_uses_portable_paths(self) -> None:
        plan = make_run_plan(load_project(self.root), "n", self.root / "plugins")
        self.assertEqual("READY", plan["adapter_plan"]["status"])
        self.assertEqual(ATTEMPT_TOKEN, plan["adapter_plan"]["cwd"])
        self.assertTrue(any(token in json.dumps(plan) for token in TOKENS))
        roots = PortableRoots(
            project_root=self.root,
            attempt_dir=self.root / ".mlipflow" / "runs" / "n" / "attempt-1",
            plugin_dir=self.root / "plugins" / "mlip-training",
        )
        self.assertEqual([], contains_absolute_root(plan, roots))

    def test_runtime_resolution_restores_absolute_paths(self) -> None:
        plan = make_run_plan(load_project(self.root), "n", self.root / "plugins")
        roots = PortableRoots(
            project_root=self.root,
            attempt_dir=self.root / ".mlipflow" / "runs" / "n" / "attempt-1",
            plugin_dir=self.root / "plugins" / "mlip-training",
        )
        runtime = to_runtime(plan["adapter_plan"], roots)
        self.assertEqual(str(roots.attempt_dir), runtime["cwd"])
        self.assertNotEqual([], contains_absolute_root(runtime, roots))


class RuntimeResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name).resolve() / "project"
        self.root.mkdir(parents=True)
        build_training_project(self.root)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_executed_argv_and_cwd_are_resolved_absolute_paths(self) -> None:
        attempt = self.root / ".mlipflow" / "runs" / "n" / "attempt-1"
        captured: dict[str, object] = {}

        def fake_run(_self, argv, cwd, environment=None):
            del environment
            captured["argv"] = list(argv)
            captured["cwd"] = str(cwd)
            Path(cwd).mkdir(parents=True, exist_ok=True)
            (Path(cwd) / "model.bin").write_bytes(b"small-test-model")
            (Path(cwd) / "training-result.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "plugin_id": "mlip-training",
                        "status": "OK",
                        "framework": "mace",
                        "framework_version": "test-only",
                        "operation": "train",
                        "seed": 20260810,
                        "device": "cpu",
                        "precision": "float64",
                        "dataset_path": str(self.root / "data" / "labeled"),
                        "config_path": str(self.root / "configs" / "mace.json"),
                        "model_artifact": {
                            "path": "model.bin",
                            "media_type": "application/octet-stream",
                        },
                        "metrics": {"best_validation_loss": 0.012},
                    }
                ),
                encoding="utf-8",
            )
            return ExecutionResult(0, "", "")

        with patch("mlipflow.services.LocalBackend.run", autospec=True, side_effect=fake_run):
            result = run_node(
                load_project(self.root), "n", self.root / "plugins", approval=True
            )

        self.assertEqual("OK", result["step"]["state"])
        self.assertEqual(str(attempt), captured["cwd"])
        rendered = " ".join(captured["argv"])
        for token in TOKENS:
            self.assertNotIn(token, rendered)
        self.assertIn(str(attempt), rendered)
        self.assertEqual("/usr/bin/python3", captured["argv"][0])


class ExamplePlanTests(unittest.TestCase):
    def test_example_plans_have_no_machine_specific_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            project_dir = root / "project"
            plugins_dir = root / "plugins"
            shutil.copytree(EXAMPLE, project_dir)
            shutil.copytree(PLUGINS, plugins_dir)
            project = load_project(project_dir)
            for node in project.nodes:
                node_id = str(node["id"])
                with self.subTest(node=node_id):
                    plan = make_run_plan(project, node_id, plugins_dir)
                    roots = PortableRoots(
                        project_root=project_dir,
                        attempt_dir=project_dir
                        / ".mlipflow"
                        / "runs"
                        / node_id
                        / "attempt-1",
                        plugin_dir=plugins_dir / str(node["uses"]).split("@", 1)[0],
                    )
                    self.assertEqual([], contains_absolute_root(plan, roots))


if __name__ == "__main__":
    unittest.main()
