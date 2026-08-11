"""Persistent workflow state and legal transitions."""

from __future__ import annotations

import json
import hashlib
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Optional

from .errors import StateError


class RunState(str, Enum):
    PREP = "PREP"
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
    RunState.PREP: frozenset({RunState.READY, RunState.FAIL, RunState.STOPPED}),
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
    if before == after:
        return
    if after not in ALLOWED_TRANSITIONS[before]:
        raise StateError(f"illegal state transition: {before.value} -> {after.value}")


@dataclass(frozen=True)
class StepRun:
    run_id: str
    project_id: str
    node_id: str
    plugin_id: str
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
    retry_count: int
    manifest_path: Optional[str]
    diagnostic: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS step_runs (
  run_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  node_id TEXT NOT NULL,
  plugin_id TEXT NOT NULL,
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
  retry_count INTEGER NOT NULL DEFAULT 0,
  manifest_path TEXT,
  diagnostic TEXT,
  node_json TEXT NOT NULL,
  UNIQUE(project_id, node_id, attempt)
);
CREATE TABLE IF NOT EXISTS dependencies (
  project_id TEXT NOT NULL,
  node_id TEXT NOT NULL,
  needs_node_id TEXT NOT NULL,
  PRIMARY KEY(project_id, node_id, needs_node_id)
);
CREATE TABLE IF NOT EXISTS artifacts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL REFERENCES step_runs(run_id),
  role TEXT NOT NULL,
  uri TEXT NOT NULL,
  fingerprint TEXT,
  size_bytes INTEGER,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL REFERENCES step_runs(run_id),
  event_type TEXT NOT NULL,
  at TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS submission_intents (
  digest TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  node_id TEXT NOT NULL,
  plan_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  consumed_at TEXT
);
"""


class StateStore:
    """SQLite store.

    Callers must choose ``readonly=True`` for every query path. In that mode
    SQLite is opened with ``mode=ro`` and schema creation is impossible.
    """

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
            self.connection.execute(
                "INSERT OR IGNORE INTO metadata(key, value) VALUES('schema_version', '1')"
            )
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
        now = utc_now()
        with self.transaction() as connection:
            for node in nodes:
                needs = list(node.get("needs", []))
                initial = RunState.WAIT if needs else RunState.READY
                run_id = str(uuid.uuid4())
                connection.execute(
                    """INSERT OR IGNORE INTO step_runs(
                         run_id, project_id, node_id, plugin_id, attempt, state, backend,
                         created_at, updated_at, retry_count, node_json
                       ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, 0, ?)""",
                    (
                        run_id,
                        project_id,
                        node["id"],
                        node["uses"],
                        initial.value,
                        node.get("backend", "local"),
                        now,
                        now,
                        json.dumps(node, sort_keys=True),
                    ),
                )
                for dependency in needs:
                    connection.execute(
                        "INSERT OR IGNORE INTO dependencies VALUES (?, ?, ?)",
                        (project_id, node["id"], dependency),
                    )

    def record_project_config(self, project_id: str, config: dict[str, Any]) -> None:
        if self.readonly:
            raise StateError("cannot record project config in read-only mode")
        digest = _config_digest(config)
        key = f"project_config:{project_id}"
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT value FROM metadata WHERE key = ?", (key,)
            ).fetchone()
            if existing is not None and existing[0] != digest:
                raise StateError(
                    "project configuration differs from initialized state; use an explicit "
                    "migration or a new project state database"
                )
            connection.execute(
                "INSERT OR IGNORE INTO metadata(key, value) VALUES(?, ?)", (key, digest)
            )

    def assert_project_config(self, project_id: str, config: dict[str, Any]) -> None:
        key = f"project_config:{project_id}"
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            raise StateError(
                "state database has no project configuration snapshot; explicit migration is required"
            )
        if row[0] != _config_digest(config):
            raise StateError(
                "project configuration differs from initialized state; use an explicit "
                "migration or a new project state database"
            )

    def assert_project_nodes(self, project_id: str, nodes: list[dict[str, Any]]) -> None:
        """Reject silent state/config drift before any mutation is planned."""

        rows = self.latest_steps(project_id)
        persisted = {row.node_id: self.node_snapshot(row.run_id) for row in rows}
        configured = {str(node["id"]): node for node in nodes}
        if set(persisted) != set(configured):
            raise StateError(
                "initialized workflow nodes differ from project.yaml; create an explicit "
                "migration or a new project state database"
            )
        for node_id in sorted(configured):
            if json.dumps(persisted[node_id], sort_keys=True) != json.dumps(
                configured[node_id], sort_keys=True
            ):
                raise StateError(
                    f"node {node_id} differs from its initialized snapshot; retries and "
                    "submissions require an explicit migration or new project state"
                )

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

    def previous_step(
        self, project_id: str, node_id: str, before_attempt: int
    ) -> StepRun | None:
        row = self.connection.execute(
            """SELECT * FROM step_runs
               WHERE project_id = ? AND node_id = ? AND attempt < ?
               ORDER BY attempt DESC LIMIT 1""",
            (project_id, node_id, before_attempt),
        ).fetchone()
        return _row_to_step(row) if row is not None else None

    def node_snapshot(self, run_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT node_json FROM step_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise StateError(f"unknown run id: {run_id}")
        return json.loads(row[0])

    def transition(
        self,
        run_id: str,
        after: RunState,
        *,
        diagnostic: Optional[str] = None,
        job_id: Optional[str] = None,
        remote_dir: Optional[str] = None,
        manifest_path: Optional[str] = None,
    ) -> StepRun:
        if self.readonly:
            raise StateError("cannot transition state in read-only mode")
        now = utc_now()
        with self.transaction() as connection:
            row = connection.execute("SELECT * FROM step_runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                raise StateError(f"unknown run id: {run_id}")
            before = RunState(row["state"])
            validate_transition(before, after)
            submitted_at = now if after == RunState.SUBMITTED else row["submitted_at"]
            started_at = now if after == RunState.RUNNING else row["started_at"]
            ended_at = now if after in {RunState.OK, RunState.FAIL, RunState.STOPPED} else row["ended_at"]
            connection.execute(
                """UPDATE step_runs SET state=?, updated_at=?, submitted_at=?, started_at=?,
                   ended_at=?, diagnostic=?, job_id=COALESCE(?, job_id),
                   remote_dir=COALESCE(?, remote_dir),
                   manifest_path=COALESCE(?, manifest_path) WHERE run_id=?""",
                (
                    after.value,
                    now,
                    submitted_at,
                    started_at,
                    ended_at,
                    diagnostic,
                    job_id,
                    remote_dir,
                    manifest_path,
                    run_id,
                ),
            )
            connection.execute(
                "INSERT INTO events(run_id, event_type, at, payload_json) VALUES(?, ?, ?, ?)",
                (run_id, "state_transition", now, json.dumps({"from": before, "to": after})),
            )
        return self.step_by_run_id(run_id)

    def step_by_run_id(self, run_id: str) -> StepRun:
        row = self.connection.execute("SELECT * FROM step_runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise StateError(f"unknown run id: {run_id}")
        return _row_to_step(row)

    def create_retry(self, project_id: str, node_id: str) -> StepRun:
        previous = self.latest_step(project_id, node_id)
        if RunState(previous.state) not in {RunState.FAIL, RunState.STOPPED}:
            raise StateError(f"retry requires FAIL or STOPPED, got {previous.state}")
        node = self.node_snapshot(previous.run_id)
        now = utc_now()
        run_id = str(uuid.uuid4())
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO step_runs(
                     run_id, project_id, node_id, plugin_id, attempt, state, backend,
                     created_at, updated_at, retry_count, node_json
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    project_id,
                    node_id,
                    previous.plugin_id,
                    previous.attempt + 1,
                    RunState.READY.value,
                    previous.backend,
                    now,
                    now,
                    previous.retry_count + 1,
                    json.dumps(node, sort_keys=True),
                ),
            )
            connection.execute(
                "INSERT INTO events(run_id, event_type, at, payload_json) VALUES(?, ?, ?, ?)",
                (run_id, "retry_created", now, json.dumps({"previous_run_id": previous.run_id})),
            )
        return self.step_by_run_id(run_id)

    def dependencies_satisfied(self, project_id: str, node_id: str) -> bool:
        needs = self.connection.execute(
            "SELECT needs_node_id FROM dependencies WHERE project_id=? AND node_id=?",
            (project_id, node_id),
        ).fetchall()
        for need in needs:
            dependency = self.latest_step(project_id, need[0])
            if dependency.state != RunState.OK.value:
                return False
        return True

    def add_artifact(
        self,
        run_id: str,
        role: str,
        uri: str,
        fingerprint: Optional[str] = None,
        size_bytes: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if self.readonly:
            raise StateError("cannot add artifact in read-only mode")
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO artifacts(run_id, role, uri, fingerprint, size_bytes, metadata_json)
                   VALUES(?, ?, ?, ?, ?, ?)""",
                (run_id, role, uri, fingerprint, size_bytes, json.dumps(metadata or {}, sort_keys=True)),
            )

    def record_intent(self, digest: str, project_id: str, node_id: str, plan: dict[str, Any]) -> None:
        if self.readonly:
            raise StateError("cannot record an intent in read-only mode")
        with self.transaction() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO submission_intents(
                     digest, project_id, node_id, plan_json, created_at, consumed_at
                   ) VALUES(?, ?, ?, ?, ?, NULL)""",
                (digest, project_id, node_id, json.dumps(plan, sort_keys=True), utc_now()),
            )

    def consume_intent(self, digest: str) -> None:
        if self.readonly:
            raise StateError("cannot consume an intent in read-only mode")
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE submission_intents SET consumed_at=? WHERE digest=? AND consumed_at IS NULL",
                (utc_now(), digest),
            )
            if cursor.rowcount != 1:
                raise StateError(f"submission intent is missing or already consumed: {digest}")

    def artifacts(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT role, uri, fingerprint, size_bytes, metadata_json FROM artifacts WHERE run_id=?",
            (run_id,),
        ).fetchall()
        return [
            {
                "role": row["role"],
                "uri": row["uri"],
                "fingerprint": row["fingerprint"],
                "size_bytes": row["size_bytes"],
                "metadata": json.loads(row["metadata_json"]),
            }
            for row in rows
        ]


def _row_to_step(row: sqlite3.Row) -> StepRun:
    fields = StepRun.__dataclass_fields__
    return StepRun(**{name: row[name] for name in fields})
