from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from mlipflow.config import load_project
from mlipflow.errors import ApprovalError
from mlipflow.planning import execution_identity, with_digest
from mlipflow.services import initialize, make_run_plan, run_node

from .helpers import plugin_manifest, project_config, write_json


def _semantic_plan() -> dict:
    return {
        "schema_version": 3,
        "action": "run",
        "project_id": "display-project",
        "node_id": "display-node",
        "plugin": {
            "id": "demo",
            "version": "1.0.0",
            "implementation_identity": {"content": "sha256:" + "1" * 64},
        },
        "mode": "execute",
        "backend": "local",
        "backend_profile": None,
        "inputs": {"data": "data.json"},
        "input_identities": {"data": {"content": "sha256:" + "2" * 64}},
        "parameters": {"model_fingerprint": "sha256:" + "3" * 64},
        "resources": {"cpus": 2},
        "cost_class": "expensive",
        "approval_required": True,
        "warnings": ["human warning"],
        "adapter_diagnostics": [{"message": "human diagnostic"}],
        "adapter_command_identities": {"0": {"content": "sha256:" + "4" * 64}},
        "adapter_plan": {
            "status": "READY",
            "executable": True,
            "argv": ["python", "worker.py", "--steps", "10"],
            "cwd": "{ATTEMPT_DIR}",
            "diagnostics": [],
            "approval_summary": {"description": "display only"},
            "input_fingerprints": {"data": "sha256:" + "2" * 64},
        },
    }


class ApprovalIdentityTests(unittest.TestCase):
    def test_cosmetic_and_duplicate_fields_do_not_change_digest(self) -> None:
        first = with_digest(_semantic_plan())
        changed = _semantic_plan()
        changed["project_id"] = "renamed-project"
        changed["node_id"] = "renamed-node"
        changed["plugin"]["version"] = "9.9.9"
        changed["warnings"] = ["different warning"]
        changed["adapter_diagnostics"] = [{"message": "different diagnostic"}]
        changed["adapter_plan"]["diagnostics"] = ["different"]
        changed["adapter_plan"]["approval_summary"] = {"different": True}
        changed["adapter_plan"]["input_fingerprints"] = {"duplicate": "changed"}
        changed["parameters"]["model_fingerprint"] = "sha256:" + "f" * 64
        changed["cost_class"] = "standard"
        changed["approval_required"] = False
        second = with_digest(changed)
        self.assertEqual(first["plan_digest"], second["plan_digest"])
        identity_text = repr(execution_identity(second))
        for excluded in (
            "warning",
            "diagnostic",
            "approval_summary",
            "model_fingerprint",
            "declared",
        ):
            self.assertNotIn(excluded, identity_text)

    def test_execution_semantics_change_digest(self) -> None:
        original = with_digest(_semantic_plan())["plan_digest"]
        changes = []
        for path, value in (
            (("backend",), "ssh-slurm"),
            (("resources", "cpus"), 8),
            (("adapter_plan", "argv"), ["python", "worker.py", "--steps", "20"]),
            (("input_identities", "data", "content"), "sha256:" + "a" * 64),
            (("adapter_command_identities", "0", "content"), "sha256:" + "b" * 64),
        ):
            changed = copy.deepcopy(_semantic_plan())
            target = changed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            changes.append(with_digest(changed)["plan_digest"])
        self.assertTrue(all(item != original for item in changes))

    def test_approval_required_run_rejects_missing_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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
                "'argv': ['true'], 'cwd': context['attempt_dir']}\n",
                encoding="utf-8",
            )
            write_json(
                root / "project.yaml",
                project_config([{"id": "expensive", "uses": "demo@1"}]),
            )
            initialize(root)
            project = load_project(root)
            plan = make_run_plan(project, "expensive", plugins)
            self.assertTrue(plan["approval_required"])
            with self.assertRaises(ApprovalError):
                run_node(project, "expensive", plugins)
            self.assertFalse((root / ".mlipflow/runs/expensive/attempt-1").exists())


if __name__ == "__main__":
    unittest.main()
