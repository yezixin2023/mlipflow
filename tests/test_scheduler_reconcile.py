from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlipflow.backends import ExecutionResult
from mlipflow.config import load_project
from mlipflow.services import (
    advance,
    initialize,
    make_advance_plan,
    make_run_plan,
    query_workflow,
    run_node,
    _materialize_slurm_adapter,
)

from .helpers import plugin_manifest, project_config, write_json


class SchedulerReconcileTests(unittest.TestCase):
    def test_fixed_slurm_wrapper_never_interpolates_adapter_argv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt = root / "attempt-1"
            attempt.mkdir()
            script, command = _materialize_slurm_adapter(
                attempt,
                {
                    "argv": ["python3", "worker.py", "literal;not-shell"],
                    "cwd": str(attempt),
                },
                {"partition": "compute", "cpus_per_task": 4, "time": "00:10:00"},
                "safe-adapter",
                {"MLIPFLOW_COMPLETION_CONTEXT": str(attempt / "completion-context.json")},
            )
            script_text = script.read_text(encoding="utf-8")
            self.assertIn("#SBATCH --partition=compute", script_text)
            self.assertIn("#SBATCH --cpus-per-task=4", script_text)
            self.assertNotIn("literal;not-shell", script_text)
            command_file = json.loads((attempt / "command.json").read_text(encoding="utf-8"))
            self.assertEqual(command[-1], "literal;not-shell")
            self.assertEqual(command_file["argv"], command)

    def test_completed_scheduler_needs_scientific_manifest_before_ok(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugins = root / "plugins"
            manifest = plugin_manifest()
            manifest["implementation"]["kind"] = "external-command"
            manifest["execution"]["backends"] = ["slurm"]
            write_json(plugins / "demo" / "plugin.yaml", manifest)
            (root / "run.slurm").write_text("#!/bin/sh\n", encoding="utf-8")
            output = root / "metrics.csv"
            output.write_text("metric,value\nmae,0.1\n", encoding="utf-8")
            write_json(
                root / "completion.json",
                {
                    "schema_version": 1,
                    "status": "OK",
                    "metrics": {"mae": 0.1},
                    "artifacts": [{"role": "metric", "path": "metrics.csv"}],
                },
            )
            node = {
                "id": "hpc",
                "uses": "demo@1",
                "backend": "slurm",
                "parameters": {
                    "submit_script": "run.slurm",
                    "completion_manifest": "completion.json",
                },
            }
            write_json(root / "project.yaml", project_config([node]))
            initialize(root)
            project = load_project(root)
            plan = make_run_plan(project, "hpc", plugins)
            with patch(
                "mlipflow.services.SlurmBackend.submit",
                return_value=ExecutionResult(0, "Submitted batch job 42\n", "", "42"),
            ):
                submitted = run_node(project, "hpc", plugins, plan["plan_digest"])
            completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
            completion.update(
                {
                    "project_id": project.project_id,
                    "node_id": "hpc",
                    "run_id": submitted["step"]["run_id"],
                    "attempt": 1,
                    "plugin_id": "demo",
                    "plan_digest": plan["plan_digest"],
                }
            )
            write_json(root / "completion.json", completion)
            self.assertEqual(query_workflow(project)["steps"][0]["state"], "PENDING")
            with patch(
                "mlipflow.services.SlurmBackend.status",
                return_value={"state": "COMPLETED", "detail": None, "source": "sacct"},
            ):
                advance_plan = make_advance_plan(project)
                result = advance(project, advance_plan["plan_digest"])
            self.assertEqual(result["changed"][0]["state"], "OK")
            state = query_workflow(project)["steps"][0]
            self.assertEqual(state["artifacts"][0]["uri"], output.resolve().as_uri())
            final_manifest = Path(state["manifest_path"])
            self.assertEqual(final_manifest.name, "run-manifest.final.json")
            final = json.loads(final_manifest.read_text(encoding="utf-8"))
            self.assertEqual(final["state"], "OK")
            self.assertEqual(final["job"]["scheduler_state"], "COMPLETED")

    def test_scheduler_completed_without_checker_withholds_ok(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            node = {"id": "hpc", "uses": "demo@1", "backend": "slurm"}
            write_json(root / "project.yaml", project_config([node]))
            initialize(root)
            project = load_project(root)
            from mlipflow.state import RunState, StateStore
            from mlipflow.services import state_path

            with StateStore(state_path(project)) as store:
                step = store.latest_step(project.project_id, "hpc")
                store.transition(step.run_id, RunState.SUBMITTED, job_id="7")
                store.transition(step.run_id, RunState.PENDING, job_id="7")
            with patch(
                "mlipflow.services.SlurmBackend.status",
                return_value={"state": "COMPLETED", "detail": None, "source": "sacct"},
            ):
                plan = make_advance_plan(project)
            self.assertEqual([], plan["details"]["transitions"])
            self.assertIn("scientific OK is withheld", plan["details"]["observations"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
