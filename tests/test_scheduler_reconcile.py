from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlipflow.backends import ExecutionResult
from mlipflow.config import load_project
from mlipflow.services import initialize, make_advance_plan, query_workflow, state_path
from mlipflow.state import RunState, StateStore

from .helpers import project_config, write_json


class RecordingScheduler:
    """A scheduler that answers from a script instead of shelling out."""

    name = "fake"

    def __init__(self, state: str) -> None:
        self.state = state
        self.queried: list[str] = []

    def submit(self, script, cwd) -> ExecutionResult:  # pragma: no cover - unused here
        raise AssertionError("submit must not be called while reconciling")

    def cancel(self, job_id: str) -> ExecutionResult:  # pragma: no cover - unused here
        raise AssertionError("cancel must not be called while reconciling")

    def status(self, job_id: str) -> dict[str, str | None]:
        self.queried.append(job_id)
        return {"state": self.state, "detail": None, "source": "fake"}


class SchedulerReconcileTests(unittest.TestCase):
    def test_injected_scheduler_replaces_subprocess_patching(self) -> None:
        """The factory seam lets a test supply a scheduler outright.

        Without it, the only way to exercise reconciliation was to patch
        ``mlipflow.backends.subprocess.run``, which pins the test to the
        assumption that a backend is implemented with subprocess at all.
        """

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            node = {"id": "hpc", "uses": "demo@1", "backend": "slurm"}
            write_json(root / "project.yaml", project_config([node]))
            initialize(root)
            project = load_project(root)
            with StateStore(state_path(project)) as store:
                step = store.latest_step(project.project_id, "hpc")
                store.transition(step.run_id, RunState.SUBMITTED, job_id="7")
                store.transition(step.run_id, RunState.PENDING, job_id="7")

            scheduler = RecordingScheduler("RUNNING")
            with patch(
                "mlipflow.backends.subprocess.run",
                side_effect=AssertionError("no subprocess may be spawned"),
            ):
                plan = make_advance_plan(
                    project, factory=lambda backend, ssh_profile: scheduler
                )

            self.assertEqual(["7"], scheduler.queried)
            self.assertEqual(
                [{"node_id": "hpc", "to": "RUNNING", "reason": "scheduler reports RUNNING"}],
                [
                    {key: item[key] for key in ("node_id", "to", "reason")}
                    for item in plan["details"]["transitions"]
                ],
            )

    def test_scheduler_completed_without_scientific_contract_withholds_ok(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            node = {"id": "hpc", "uses": "demo@1", "backend": "slurm"}
            write_json(root / "project.yaml", project_config([node]))
            initialize(root)
            project = load_project(root)
            with StateStore(state_path(project)) as store:
                step = store.latest_step(project.project_id, "hpc")
                store.transition(step.run_id, RunState.SUBMITTED, job_id="7")
                store.transition(step.run_id, RunState.PENDING, job_id="7")
            with patch(
                "mlipflow.services.SlurmBackend.status",
                return_value={"state": "COMPLETED", "detail": None, "source": "fake"},
            ):
                plan = make_advance_plan(project)
            self.assertEqual([], plan["details"]["transitions"])
            self.assertIn(
                "scientific OK is withheld", plan["details"]["observations"][0]["reason"]
            )
            self.assertEqual("PENDING", query_workflow(project)["steps"][0]["state"])


if __name__ == "__main__":
    unittest.main()
