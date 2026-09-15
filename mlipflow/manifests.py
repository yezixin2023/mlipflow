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
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
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
    if result is not None:
        record["result"] = result_summary(result)
    return record


def result_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Keep reported metrics and completion evidence, without logs or data arrays.

    Local execution nests metrics under check/collection; replay can report them
    directly. Collection values take precedence when the same metric is repeated.
    Units and metric names are preserved exactly as reported by the capability.
    """
    summary = {key: result[key] for key in ("returncode", "status", "diagnostics", "summary") if key in result}
    metrics = dict(result["metrics"]) if isinstance(result.get("metrics"), dict) else {}
    for phase in ("check", "collection"):
        value = result.get(phase)
        if isinstance(value, dict):
            summary[phase] = {
                key: value[key] for key in ("status", "diagnostics", "metrics") if key in value
            }
            if isinstance(value.get("metrics"), dict):
                metrics.update(value["metrics"])
            if isinstance(value.get("summary"), dict):
                summary["summary"] = value["summary"]
        elif phase in result:
            summary[phase] = None
    if metrics:
        summary["metrics"] = metrics
    return summary
