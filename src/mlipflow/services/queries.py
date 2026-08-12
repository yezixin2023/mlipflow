"""Read-only query services.

These functions only open SQLite in read-only mode and never call execution
backends.  The absence of a ``..backends`` import in this module is the
machine-checkable form of that guarantee; ``tests/test_module_boundaries.py``
asserts it statically.

``query_doctor`` is the single documented exception to the "queries never touch
site knowledge" rule: diagnosing an ``ssh-slurm`` project requires validating the
local site profile.  It still only *reads* that configuration.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any, Iterable

from ..config import Project, load_project
from ..errors import ConfigError, StateError
from ..planning import resolve_reference
from ..plugins import discover_plugins, select_plugin
from ..routing import load_registry, route_models
from ..site import load_site_config
from ..state import RunState, StateStore
from .paths import attempt_directory, state_path


def query_workflow(project: Project, node_id: str | None = None) -> dict[str, Any]:
    database = state_path(project)
    if database.is_file():
        with StateStore(database, readonly=True) as store:
            steps = [step.to_dict() for step in store.latest_steps(project.project_id)]
            for step in steps:
                step["artifacts"] = store.artifacts(step["run_id"])
                step["persisted"] = True
    else:
        steps = [
            {
                "run_id": None,
                "project_id": project.project_id,
                "node_id": node["id"],
                "plugin_id": node["uses"],
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
                "retry_count": 0,
                "manifest_path": None,
                "diagnostic": "preview: project has not been initialized",
                "artifacts": [],
                "persisted": False,
            }
            for node in project.nodes
        ]
    if node_id is not None:
        steps = [step for step in steps if step["node_id"] == node_id]
        if not steps:
            raise ConfigError(f"unknown workflow node: {node_id}")
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


def query_inspect(project: Project, node_id: str, plugin_root: Path) -> dict[str, Any]:
    node = project.node(node_id)
    plugins = discover_plugins(plugin_root)
    plugin = select_plugin(plugins, str(node["uses"]))
    workflow = query_workflow(project, node_id)
    return {
        "project_id": project.project_id,
        "node": node,
        "state": workflow["steps"][0],
        "plugin": {"path": str(plugin.path), "manifest": plugin.raw},
    }


def query_logs(project: Project, node_id: str, tail: int = 80) -> dict[str, Any]:
    if tail < 1 or tail > 10000:
        raise ConfigError("--tail must be between 1 and 10000")
    database = state_path(project)
    if not database.is_file():
        raise StateError("project has not been initialized")
    with StateStore(database, readonly=True) as store:
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
        load_registry(
            registry_path,
            full_hash_max_bytes=int(
                project.raw.get("fingerprints", {}).get("full_hash_max_bytes", 67108864)
            ),
        ),
        task=task,
        elements=set(elements),
        scenario=scenario,
        policy=policy,
    )


def query_doctor(
    project_path: Path, plugin_root: Path, site_path: Path | None = None
) -> dict[str, Any]:
    diagnostics: list[dict[str, Any]] = []
    try:
        project = load_project(project_path)
        diagnostics.append({"check": "project", "ok": True, "detail": str(project.path)})
    except Exception as exc:
        return {"ok": False, "diagnostics": [{"check": "project", "ok": False, "detail": str(exc)}]}
    try:
        plugins = discover_plugins(plugin_root)
        diagnostics.append(
            {"check": "plugins", "ok": bool(plugins), "detail": f"{len(plugins)} discovered"}
        )
        for node in project.nodes:
            plugin = select_plugin(plugins, str(node["uses"]))
            if node.get("mode", "execute") == "replay":
                continue
            dependencies = plugin.raw.get("dependencies", {})
            for dependency in dependencies.get("python_packages", []):
                if not dependency.get("required"):
                    continue
                package = (
                    str(dependency.get("name", "")).replace("-", "_").split(".", 1)[0]
                )
                available = bool(package) and importlib.util.find_spec(package) is not None
                diagnostics.append(
                    {
                        "check": f"python:{package}",
                        "ok": available,
                        "detail": f"required by {plugin.plugin_id}",
                    }
                )
            for dependency in dependencies.get("external_programs", []):
                if not dependency.get("required"):
                    continue
                executable = str(dependency.get("name", ""))
                path = shutil.which(executable) if executable and ":" not in executable else None
                diagnostics.append(
                    {
                        "check": f"executable:{executable}",
                        "ok": path is not None,
                        "detail": path or f"required by {plugin.plugin_id}",
                    }
                )
    except Exception as exc:
        diagnostics.append({"check": "plugins", "ok": False, "detail": str(exc)})
    database = state_path(project)
    if database.is_file():
        try:
            with StateStore(database, readonly=True) as store:
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
                }
            )
            diagnostics.append(
                {
                    "check": "site-config",
                    "ok": True,
                    "detail": f"{site.path}; selected profiles: {', '.join(selected)}",
                }
            )
        except Exception as exc:
            diagnostics.append(
                {"check": "site-config", "ok": False, "detail": str(exc)}
            )
    required = set()
    if "slurm" in backends:
        required.update({"sbatch", "scancel"})
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
