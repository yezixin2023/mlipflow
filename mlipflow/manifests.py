"""Small records of what happened in one workflow attempt."""

from __future__ import annotations

from typing import Any, Iterable


def run_manifest(
    *,
    project_id: str,
    node_id: str,
    run_id: str,
    attempt: int,
    capability_id: str,
    mode: str,
    backend: str,
    timestamps: dict[str, str | None],
    artifacts: Iterable[dict[str, Any]] = (),
    state: str = "OK",
    state_reason: str | None = None,
    backend_profile: str | None = None,
    job: dict[str, Any] | None = None,
    remote_dir: str | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "project_id": project_id,
        "node_id": node_id,
        "run_id": run_id,
        "attempt": attempt,
        "capability": capability_id,
        "state": state,
        "state_reason": state_reason,
        "mode": mode,
        "backend": backend,
        "backend_profile": backend_profile,
        "job": job,
        "remote_dir": remote_dir,
        "timestamps": timestamps,
        "command": command,
        "artifacts": list(artifacts),
    }
