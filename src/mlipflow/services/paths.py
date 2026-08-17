"""Project-scoped path resolution.

This module is imported by both the query and the command side, so it must not
import execution backends.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Project
from ..errors import ConfigError


STATE_RELATIVE = Path(".mlipflow/state.sqlite3")


def state_path(project: Project) -> Path:
    configuration = project.raw.get("state", {})
    reference = (
        configuration.get("database_path", str(STATE_RELATIVE))
        if isinstance(configuration, dict)
        else str(STATE_RELATIVE)
    )
    if not isinstance(reference, str) or not reference or Path(reference).is_absolute():
        raise ConfigError("state.database_path must be a non-empty project-relative path")
    candidate = (project.root / reference).resolve()
    try:
        candidate.relative_to(project.root.resolve())
    except ValueError as exc:
        raise ConfigError("state.database_path must remain inside the project root") from exc
    return candidate


def attempt_directory(project: Project, node_id: str, attempt: int) -> Path:
    """Return the fixed local workspace for one node attempt."""

    return project.root / ".mlipflow" / "runs" / str(node_id) / f"attempt-{attempt}"
