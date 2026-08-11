"""Run/result manifest construction matching ``run-manifest.schema.json``."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from . import __version__
from .io import canonical_json
from .provenance import capture, redact


def run_manifest(
    *,
    project_id: str,
    node_id: str,
    run_id: str,
    attempt: int,
    plugin_id: str,
    plugin_version: str,
    mode: str,
    backend: str,
    plan_digest: str,
    inputs: dict[str, Any],
    parameters: dict[str, Any],
    artifacts: Iterable[dict[str, Any]] = (),
    metrics: dict[str, Any] | None = None,
    state: str = "OK",
    state_reason: str | None = None,
    backend_profile: str | None = None,
    job: dict[str, Any] | None = None,
    remote_dir: str | None = None,
    timestamps: dict[str, str | None] | None = None,
    retry_count: int = 0,
    max_retries: int | None = None,
    previous_run_id: str | None = None,
    dependencies: Iterable[dict[str, Any]] = (),
    command: list[str] | None = None,
    resources: dict[str, Any] | None = None,
    source_run_id: str | None = None,
    source_root: str | None = None,
) -> dict[str, Any]:
    captured = capture(source_root=source_root)
    config_value = {
        "plugin": {"id": plugin_id, "version": plugin_version},
        "mode": mode,
        "backend": backend,
        "inputs": inputs,
        "parameters": parameters,
        "command": command,
        "resources": resources or {},
    }
    config_digest = hashlib.sha256(canonical_json(config_value).encode("utf-8")).hexdigest()
    time_values = timestamps or {}
    return redact(
        {
            "$schema": "urn:mlipflow:schema:run-manifest:1",
            "schema_version": 1,
            "project_id": project_id,
            "node_id": node_id,
            "run_id": run_id,
            "attempt": attempt,
            "plugin": {"id": plugin_id, "version": plugin_version},
            "state": state,
            "state_reason": state_reason,
            "mode": mode,
            "backend": {"type": backend, "profile": backend_profile},
            "job": job,
            "remote_dir": remote_dir,
            "timestamps": {
                "created_at": time_values.get("created_at") or captured["captured_at"],
                "updated_at": time_values.get("updated_at") or captured["captured_at"],
                "submitted_at": time_values.get("submitted_at"),
                "started_at": time_values.get("started_at"),
                "finished_at": time_values.get("finished_at"),
            },
            "retry": {
                "retry_count": retry_count,
                "max_retries": max_retries,
                "previous_run_id": previous_run_id,
            },
            "dependencies": list(dependencies),
            "command": command,
            "resources": resources or {},
            "inputs": inputs,
            "parameters": parameters,
            "artifacts": list(artifacts),
            "metrics": metrics or {},
            "provenance": {
                "plan_digest": plan_digest,
                "config_digest": f"sha256:{config_digest}",
                "created_by": "mlipflow",
                "software": [{"name": "mlipflow", "version": __version__}],
                "environment": {
                    "python": captured["python"],
                    "platform": captured["platform"],
                    "variables": captured["environment"],
                },
                "source_git_commit": captured["source_git_commit"],
                "captured_at": captured["captured_at"],
                "source_run_id": source_run_id,
            },
        }
    )
