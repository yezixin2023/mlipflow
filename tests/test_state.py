from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from mlipflow.errors import StateError
from mlipflow.state import ALLOWED_TRANSITIONS, RunState, StateStore, validate_transition


class TransitionTests(unittest.TestCase):
    def test_database_contains_only_minimal_step_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "state.sqlite3"
            with StateStore(database):
                pass
            connection = sqlite3.connect(database)
            tables = [
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
            ]
            columns = [
                row[1] for row in connection.execute("PRAGMA table_info(step_runs)")
            ]
            connection.close()
            self.assertEqual(["step_runs"], tables)
            self.assertEqual(
                [
                    "run_id", "project_id", "node_id", "attempt", "state",
                    "backend", "job_id", "remote_dir", "created_at", "updated_at",
                    "submitted_at", "started_at", "ended_at", "diagnostic",
                ],
                columns,
            )

    def test_every_declared_transition_is_accepted(self) -> None:
        for before, afters in ALLOWED_TRANSITIONS.items():
            validate_transition(before, before)
            for after in afters:
                validate_transition(before, after)

    def test_terminal_ok_cannot_transition(self) -> None:
        with self.assertRaises(StateError):
            validate_transition(RunState.OK, RunState.RUNNING)

    def test_every_undeclared_transition_is_rejected(self) -> None:
        for before in RunState:
            for after in RunState:
                if after == before or after in ALLOWED_TRANSITIONS[before]:
                    continue
                with self.subTest(before=before, after=after):
                    with self.assertRaises(StateError):
                        validate_transition(before, after)

    def test_retry_creates_new_attempt_and_preserves_previous(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "state.sqlite3"
            nodes = [{"id": "train", "uses": "mlip-training", "needs": [], "backend": "local"}]
            with StateStore(database) as store:
                store.initialize_project("p", nodes)
                first = store.latest_step("p", "train")
                store.transition(first.run_id, RunState.RUNNING)
                store.transition(first.run_id, RunState.FAIL, diagnostic="fixture failure")
                second = store.create_retry("p", "train", "local")
                self.assertEqual(second.attempt, 2)
                self.assertEqual(second.state, RunState.READY.value)
                preserved = store.step_by_run_id(first.run_id)
                self.assertEqual(preserved.state, RunState.FAIL.value)

    def test_readonly_store_rejects_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "state.sqlite3"
            with StateStore(database):
                pass
            with StateStore(database, readonly=True) as store:
                with self.assertRaises(StateError):
                    store.initialize_project("p", [])

    def test_state_survives_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "state.sqlite3"
            nodes = [{"id": "x", "uses": "candidate-ranking", "needs": []}]
            with StateStore(database) as store:
                store.initialize_project("p", nodes)
                step = store.latest_step("p", "x")
                store.transition(step.run_id, RunState.RUNNING)
            with StateStore(database, readonly=True) as reopened:
                self.assertEqual(reopened.latest_step("p", "x").state, RunState.RUNNING.value)


if __name__ == "__main__":
    unittest.main()
