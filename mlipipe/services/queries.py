"""Read-only query services.

These functions open SQLite read-only and read saved manifests and local logs.
They never observe a scheduler, fetch outputs, run checks, or advance state.

``query_doctor`` is the single documented exception to the "queries never touch
site knowledge" rule: diagnosing an ``ssh-slurm`` project requires validating the
local site profile.  It still only *reads* that configuration.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

from ..config import Project, load_project
from ..errors import CapabilityError, ConfigError, StateError
from ..io import load_mapping
from ..manifests import result_summary
from ..planning import approval_required, resolve_reference
from ..plugins import (
    BUILTIN_CAPABILITIES,
    capability,
    load_adapter,
    resolve_operation,
)
from ..routing import load_registry, route_models
from ..site import load_site_config
from ..state import RunState, StateStore
from .paths import attempt_directory, state_path


def query_workflow(project: Project, node_id: str | None = None) -> dict[str, Any]:
    """Return saved steps for one node or the project, including result locations.

    An uninitialized project returns a configuration preview. Metrics retain the
    producing capability's names and units; this query never recomputes them.
    """
    nodes = project.nodes if node_id is None else [project.node(node_id)]
    database = state_path(project)
    if database.is_file():
        with StateStore(database, readonly=True) as store:
            store.assert_project_id(project.project_id)
            steps = []
            for node in nodes:
                record = store.latest_step(project.project_id, str(node["id"]))
                step = record.to_dict()
                step["capability"] = node["uses"]
                step.update(attempt_details(project, record.node_id, record.attempt))
                step["persisted"] = True
                steps.append(step)
    else:
        steps = [
            {
                "run_id": None,
                "project_id": project.project_id,
                "node_id": node["id"],
                "capability": node["uses"],
                "attempt": 0,
                "state": RunState.WAIT.value if node.get("needs") else RunState.READY.value,
                "backend": node.get("backend", "local"),
                "job_id": None,
                "remote_dir": None,
                "created_at": None,
                "updated_at": None,
                "submitted_at": None,
                "started_at": None,
                "ended_at": None,
                "manifest_path": None,
                "diagnostic": "preview: project has not been initialized",
                "artifacts": [],
                "persisted": False,
            }
            for node in nodes
        ]
    counts: dict[str, int] = {}
    for step in steps:
        counts[step["state"]] = counts.get(step["state"], 0) + 1
    return {
        "project": {"id": project.project_id, "path": str(project.path)},
        "state_database": str(database),
        "initialized": database.is_file(),
        "counts": dict(sorted(counts.items())),
        "steps": steps,
    }


def attempt_details(project: Project, node_id: str, attempt: int) -> dict[str, Any]:
    """Read one attempt's saved outputs and checks; never collect or run a check."""
    directory = attempt_directory(project, node_id, attempt)
    final = directory / "run-manifest.final.json"
    initial = directory / "run-manifest.json"
    path = final if final.is_file() else initial
    manifest = load_mapping(path) if path.is_file() else {}
    raw_artifacts = manifest.get("artifacts", [])
    if not isinstance(raw_artifacts, list):
        raise CapabilityError(f"attempt artifact list is invalid: {path}")
    artifacts = []
    for item in raw_artifacts:
        if not isinstance(item, dict):
            continue
        record = dict(item)
        uri = urlparse(str(record.get("uri", "")))
        if uri.scheme == "file":
            record["path"] = unquote(uri.path)
        artifacts.append(record)
    result = manifest.get("result", {})
    return {
        "manifest_path": str(path) if path.is_file() else None,
        "artifacts": artifacts,
        "logs": {
            name: str(directory / f"{name}.log")
            for name in ("stdout", "stderr") if (directory / f"{name}.log").is_file()
        },
        **(result_summary(result) if isinstance(result, dict) else {}),
    }


def query_inspect(project: Project, node_id: str) -> dict[str, Any]:
    """Return the node's inputs, operation, capability contract, and saved state."""
    node = project.node(node_id)
    capability_id = str(node["uses"])
    spec = capability(capability_id)
    operation = None
    if node.get("mode", "execute") == "execute":
        adapter = load_adapter(capability_id)
        operation = resolve_operation(adapter, capability_id, node)
    workflow = query_workflow(project, node_id)
    return {
        "project_id": project.project_id,
        "node": node,
        "state": workflow["steps"][0],
        "operation": operation,
        "approval_required": approval_required(node, spec, operation),
        "capability": {
            "id": capability_id,
            "description": spec["description"],
            "operations": list(spec["operations"]),
            "backends": list(spec["backends"]),
            "approval_operations": list(spec["approval_operations"]),
        },
    }


def query_logs(project: Project, node_id: str, tail: int = 80) -> dict[str, Any]:
    """Return local log paths and up to ``tail`` lines for the latest attempt."""
    if tail < 1 or tail > 10000:
        raise ConfigError("--tail must be between 1 and 10000")
    database = state_path(project)
    if not database.is_file():
        raise StateError("project has not been initialized")
    with StateStore(database, readonly=True) as store:
        store.assert_project_id(project.project_id)
        step = store.latest_step(project.project_id, node_id)
    directory = attempt_directory(project, node_id, step.attempt)
    logs: dict[str, Any] = {}
    for name in ("stdout.log", "stderr.log"):
        path = directory / name
        logs[name] = {
            "path": str(path),
            "exists": path.is_file(),
            "lines": _tail_lines(path, tail) if path.is_file() else [],
        }
    return {"node_id": node_id, "attempt": step.attempt, "logs": logs}


def query_route(
    project: Project,
    *,
    task: str,
    elements: Iterable[str],
    scenario: str,
) -> dict[str, Any]:
    registry_ref = project.raw.get("model_registry")
    if not isinstance(registry_ref, str):
        raise ConfigError("project.model_registry must be a path")
    registry_path = resolve_reference(registry_ref, project.root)
    policies = project.raw.get("routing", {}).get("policies", {})
    policy = policies.get(task) if isinstance(policies, dict) else None
    if not isinstance(policy, dict):
        raise ConfigError(f"no routing policy for task {task!r}")
    return route_models(
        load_registry(registry_path),
        task=task,
        elements=set(elements),
        scenario=scenario,
        policy=policy,
    )


def query_doctor(project_path: Path, site_path: Path | None = None) -> dict[str, Any]:
    diagnostics: list[dict[str, Any]] = []
    try:
        project = load_project(project_path)
        diagnostics.append({"check": "project", "ok": True, "detail": str(project.path)})
    except Exception as exc:
        return {"ok": False, "diagnostics": [{"check": "project", "ok": False, "detail": str(exc)}]}
    try:
        for node in project.nodes:
            capability(str(node["uses"]))
        diagnostics.append(
            {
                "check": "capabilities",
                "ok": True,
                "detail": f"{len(BUILTIN_CAPABILITIES)} built in",
            }
        )
    except Exception as exc:
        diagnostics.append(
            {"check": "capabilities", "ok": False, "detail": str(exc)}
        )
    database = state_path(project)
    if database.is_file():
        try:
            with StateStore(database, readonly=True) as store:
                store.assert_project_id(project.project_id)
                store.latest_steps(project.project_id)
            diagnostics.append({"check": "state", "ok": True, "detail": str(database)})
        except Exception as exc:
            diagnostics.append({"check": "state", "ok": False, "detail": str(exc)})
    else:
        diagnostics.append(
            {"check": "state", "ok": True, "detail": "not initialized (read-only check)"}
        )
    backends = {str(node.get("backend", "local")) for node in project.nodes}
    if "ssh-slurm" in backends:
        try:
            site = load_site_config(site_path)
            selected = sorted(
                {
                    site.cluster(node.get("backend_profile")).name
                    for node in project.nodes
                    if node.get("backend") == "ssh-slurm"
                    and node.get("backend_profile") != "auto"
                }
            )
            automatic = any(
                node.get("backend") == "ssh-slurm"
                and node.get("backend_profile") == "auto"
                for node in project.nodes
            )
            detail = f"{site.path}; selected profiles: {', '.join(selected) or 'none'}"
            if automatic:
                candidates = [
                    profile.name
                    for profile in site.clusters.values()
                    if profile.scheduler is not None
                ]
                detail += "; automatic candidates: " + ", ".join(candidates)
            diagnostics.append(
                {
                    "check": "site-config",
                    "ok": True,
                    "detail": detail,
                }
            )
        except Exception as exc:
            diagnostics.append(
                {"check": "site-config", "ok": False, "detail": str(exc)}
            )
    required = set()
    if "ssh-slurm" in backends:
        required.add("ssh")
    for executable in sorted(required):
        path = shutil.which(executable)
        diagnostics.append(
            {"check": f"executable:{executable}", "ok": path is not None, "detail": path}
        )
    return {"ok": all(item["ok"] for item in diagnostics), "diagnostics": diagnostics}


def _tail_lines(path: Path, count: int) -> list[str]:
    block_size = 64 * 1024
    maximum = 8 * 1024 * 1024
    with path.open("rb") as stream:
        stream.seek(0, 2)
        position = stream.tell()
        chunks: list[bytes] = []
        collected = 0
        newlines = 0
        while position > 0 and collected < maximum and newlines <= count:
            size = min(block_size, position, maximum - collected)
            position -= size
            stream.seek(position)
            chunk = stream.read(size)
            chunks.append(chunk)
            collected += len(chunk)
            newlines += chunk.count(b"\n")
    payload = b"".join(reversed(chunks))
    lines = payload.decode("utf-8", errors="replace").splitlines(keepends=True)
    if position > 0 and lines:
        lines = lines[1:]
    return lines[-count:]
