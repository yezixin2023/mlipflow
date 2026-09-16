"""Minimal persistent state for workflow attempts and scheduler jobs."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Optional

from .errors import StateError


class RunState(str, Enum):
    READY = "READY"
    SUBMITTED = "SUBMITTED"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    OK = "OK"
    FAIL = "FAIL"
    WAIT = "WAIT"
    BLOCKED = "BLOCKED"
    STOPPED = "STOPPED"


ALLOWED_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.READY: frozenset(
        {RunState.SUBMITTED, RunState.RUNNING, RunState.OK, RunState.FAIL, RunState.STOPPED}
    ),
    RunState.SUBMITTED: frozenset(
        {RunState.PENDING, RunState.RUNNING, RunState.FAIL, RunState.STOPPED}
    ),
    RunState.PENDING: frozenset({RunState.RUNNING, RunState.FAIL, RunState.STOPPED}),
    RunState.RUNNING: frozenset({RunState.OK, RunState.FAIL, RunState.STOPPED}),
    RunState.OK: frozenset(),
    RunState.FAIL: frozenset(),
    RunState.WAIT: frozenset({RunState.READY, RunState.BLOCKED, RunState.STOPPED}),
    RunState.BLOCKED: frozenset({RunState.WAIT, RunState.READY, RunState.STOPPED}),
    RunState.STOPPED: frozenset(),
}


def validate_transition(before: RunState, after: RunState) -> None:
    if before != after and after not in ALLOWED_TRANSITIONS[before]:
        raise StateError(f"illegal state transition: {before.value} -> {after.value}")


@dataclass(frozen=True)
class StepRun:
    run_id: str
    project_id: str
    node_id: str
    attempt: int
    state: str
    backend: str
    job_id: Optional[str]
    remote_dir: Optional[str]
    created_at: str
    updated_at: str
    submitted_at: Optional[str]
    started_at: Optional[str]
    ended_at: Optional[str]
    diagnostic: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


SCHEMA = """
CREATE TABLE IF NOT EXISTS step_runs (
  run_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  node_id TEXT NOT NULL,
  attempt INTEGER NOT NULL,
  state TEXT NOT NULL,
  backend TEXT NOT NULL,
  job_id TEXT,
  remote_dir TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  submitted_at TEXT,
  started_at TEXT,
  ended_at TEXT,
  diagnostic TEXT,
  UNIQUE(project_id, node_id, attempt)
);
"""


class StateStore:
    def __init__(self, path: Path, readonly: bool = False):
        self.path = path
        self.readonly = readonly
        if readonly:
            if not path.is_file():
                raise StateError(f"state database does not exist: {path}")
            uri = f"file:{path.resolve().as_posix()}?mode=ro&immutable=1"
            self.connection = sqlite3.connect(uri, uri=True)
            self.connection.execute("PRAGMA query_only = ON")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(str(path))
            self.connection.executescript(SCHEMA)
            self.connection.commit()
        self.connection.row_factory = sqlite3.Row

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "StateStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        if self.readonly:
            raise StateError("cannot start a write transaction in read-only mode")
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            yield self.connection
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def initialize_project(self, project_id: str, nodes: list[dict[str, Any]]) -> None:
        self.assert_project_id(project_id)
        now = utc_now()
        with self.transaction() as connection:
            for node in nodes:
                initial = RunState.WAIT if node.get("needs") else RunState.READY
                connection.execute(
                    """INSERT OR IGNORE INTO step_runs(
                         run_id, project_id, node_id, attempt, state, backend,
                         created_at, updated_at
                       ) VALUES (?, ?, ?, 1, ?, ?, ?, ?)""",
                    (
                        f"{project_id}:{node['id']}:attempt-1",
                        project_id,
                        node["id"],
                        initial.value,
                        node.get("backend", "local"),
                        now,
                        now,
                    ),
                )

    def assert_project_id(self, project_id: str) -> None:
        owners = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT DISTINCT project_id FROM step_runs"
            ).fetchall()
        }
        if owners and owners != {project_id}:
            raise StateError(
                f"state database belongs to project(s) {sorted(owners)!r}, "
                f"not {project_id!r}"
            )

    def bind_attempt(self, run_id: str, backend: str) -> StepRun:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT state FROM step_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise StateError(f"unknown run id: {run_id}")
            if row["state"] != RunState.READY.value:
                raise StateError("only an unstarted READY attempt may bind execution config")
            connection.execute(
                "UPDATE step_runs SET backend=?, updated_at=? WHERE run_id=?",
                (backend, utc_now(), run_id),
            )
        return self.step_by_run_id(run_id)

    def latest_steps(self, project_id: str) -> list[StepRun]:
        rows = self.connection.execute(
            """SELECT s.* FROM step_runs s
               JOIN (
                 SELECT node_id, MAX(attempt) attempt
                 FROM step_runs WHERE project_id = ? GROUP BY node_id
               ) latest ON latest.node_id = s.node_id AND latest.attempt = s.attempt
               WHERE s.project_id = ? ORDER BY s.node_id""",
            (project_id, project_id),
        ).fetchall()
        return [_row_to_step(row) for row in rows]

    def latest_step(self, project_id: str, node_id: str) -> StepRun:
        row = self.connection.execute(
            """SELECT * FROM step_runs WHERE project_id = ? AND node_id = ?
               ORDER BY attempt DESC LIMIT 1""",
            (project_id, node_id),
        ).fetchone()
        if row is None:
            raise StateError(f"no state for workflow node: {node_id}")
        return _row_to_step(row)

    def steps_for_node(self, project_id: str, node_id: str) -> list[StepRun]:
        rows = self.connection.execute(
            """SELECT * FROM step_runs WHERE project_id = ? AND node_id = ?
               ORDER BY attempt""",
            (project_id, node_id),
        ).fetchall()
        return [_row_to_step(row) for row in rows]

    def transition(
        self,
        run_id: str,
        after: RunState,
        *,
        diagnostic: Optional[str] = None,
        job_id: Optional[str] = None,
        remote_dir: Optional[str] = None,
    ) -> StepRun:
        now = utc_now()
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM step_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise StateError(f"unknown run id: {run_id}")
            before = RunState(row["state"])
            validate_transition(before, after)
            submitted_at = now if after == RunState.SUBMITTED else row["submitted_at"]
            started_at = now if after == RunState.RUNNING else row["started_at"]
            ended_at = (
                now
                if after in {RunState.OK, RunState.FAIL, RunState.STOPPED}
                else row["ended_at"]
            )
            connection.execute(
                """UPDATE step_runs SET state=?, updated_at=?, submitted_at=?, started_at=?,
                   ended_at=?, diagnostic=?, job_id=COALESCE(?, job_id),
                   remote_dir=COALESCE(?, remote_dir) WHERE run_id=?""",
                (
                    after.value,
                    now,
                    submitted_at,
                    started_at,
                    ended_at,
                    diagnostic,
                    job_id,
                    remote_dir,
                    run_id,
                ),
            )
        return self.step_by_run_id(run_id)

    def step_by_run_id(self, run_id: str) -> StepRun:
        row = self.connection.execute(
            "SELECT * FROM step_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise StateError(f"unknown run id: {run_id}")
        return _row_to_step(row)

    def create_retry(self, project_id: str, node_id: str, backend: str) -> StepRun:
        previous = self.latest_step(project_id, node_id)
        if RunState(previous.state) not in {RunState.FAIL, RunState.STOPPED}:
            raise StateError(f"retry requires FAIL or STOPPED, got {previous.state}")
        now = utc_now()
        attempt = previous.attempt + 1
        run_id = f"{project_id}:{node_id}:attempt-{attempt}"
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO step_runs(
                     run_id, project_id, node_id, attempt, state, backend,
                     created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    project_id,
                    node_id,
                    attempt,
                    RunState.READY.value,
                    backend,
                    now,
                    now,
                ),
            )
        return self.step_by_run_id(run_id)


def _row_to_step(row: sqlite3.Row) -> StepRun:
    return StepRun(
        **{name: row[name] for name in StepRun.__dataclass_fields__}
    )
