"""Execution planning for supervised MLIPFlow runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Project
from .errors import PluginError
from .plugins import PluginSpec


PLAN_SCHEMA_VERSION = 4


def node_plan(
    project: Project, node: dict[str, Any], plugin: PluginSpec, *, attempt: int
) -> dict[str, Any]:
    mode = node.get("mode", "execute")
    backend = node.get("backend", "local")
    if mode == "replay" and not plugin.raw.get("replay", {}).get("supported", False):
        raise PluginError(f"plugin {plugin.plugin_id} does not declare replay support")
    if mode == "execute" and backend not in plugin.raw.get("execution", {}).get("backends", []):
        raise PluginError(
            f"plugin {plugin.plugin_id} does not declare execute support on backend {backend}"
        )
    warnings: list[str] = []
    safety = plugin.raw.get("safety", {})
    cost_class = plugin.raw.get("execution", {}).get("cost_class")
    if cost_class is None:
        cost_class = "expensive" if safety.get("expensive") else "standard"
    if cost_class in {"expensive", "very-expensive"} or safety.get("expensive"):
        warnings.append("This plan may consume substantial compute allocation.")
    if safety.get("destructive"):
        warnings.append("The plugin declares destructive behavior; inspect overwrite/delete targets.")
    if safety.get("network_access"):
        warnings.append("The plugin declares network access.")
    if backend in {"slurm", "ssh-slurm"}:
        warnings.append("Approval will submit a scheduler job.")
    if mode == "replay":
        warnings.append("Replay only parses and references existing artifacts; it must not run numerics.")
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "action": "run",
        "project_id": project.project_id,
        "node_id": node["id"],
        "attempt": attempt,
        "plugin": {"id": plugin.plugin_id, "version": plugin.raw["version"]},
        "mode": mode,
        "backend": backend,
        "backend_profile": node.get("backend_profile"),
        "inputs": node.get("inputs", {}),
        "parameters": node.get("parameters", {}),
        "resources": node.get("resources", {}),
        "cost_class": cost_class,
        "approval_required": _approval_required(node, plugin),
        "warnings": warnings,
    }


def _approval_required(node: dict[str, Any], plugin: PluginSpec) -> bool:
    if node.get("mode", "execute") != "execute":
        return False
    safety = plugin.raw.get("safety", {})
    explicit = safety.get("requires_approval_before_execution")
    if type(explicit) is bool:
        return explicit
    execution = plugin.raw.get("execution", {})
    cost_class = execution.get("cost_class")
    return bool(
        safety.get("expensive")
        or safety.get("destructive")
        or safety.get("network_access")
        or execution.get("submits_jobs")
        or cost_class in {"expensive", "very-expensive"}
        or node.get("backend", "local") in {"slurm", "ssh-slurm"}
    )


def action_plan(action: str, project: Project, node_id: str | None, details: Any) -> dict[str, Any]:
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "action": action,
        "project_id": project.project_id,
        "node_id": node_id,
        "details": details,
    }


def resolve_reference(value: Any, project_root: Path) -> Path:
    if isinstance(value, str):
        path = Path(value)
    elif isinstance(value, dict) and isinstance(value.get("path"), str):
        path = Path(value["path"])
    else:
        raise ValueError(f"expected a path reference, got {value!r}")
    return path if path.is_absolute() else project_root / path
