"""Execution planning for supervised MLIPFlow runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Project
from .errors import CapabilityError
from .plugins import capability


def node_plan(
    project: Project,
    node: dict[str, Any],
    capability_id: str,
    *,
    attempt: int,
    operation: str | None,
) -> dict[str, Any]:
    spec = capability(capability_id)
    mode = node.get("mode", "execute")
    backend = node.get("backend", "local")
    if mode == "execute" and backend not in spec["backends"]:
        raise CapabilityError(
            f"capability {capability_id} does not support execution on backend {backend}"
        )
    requires_approval = approval_required(node, spec, operation)
    warnings: list[str] = []
    if requires_approval and backend != "ssh-slurm":
        warnings.append("This plan may consume substantial compute allocation.")
    if mode == "execute" and backend == "ssh-slurm":
        warnings.append("Approval will submit a scheduler job.")
    if mode == "replay":
        warnings.append("Replay only parses and references existing artifacts; it must not run numerics.")
    return {
        "action": "run",
        "project_id": project.project_id,
        "node_id": node["id"],
        "attempt": attempt,
        "capability": capability_id,
        "operation": operation,
        "mode": mode,
        "backend": backend,
        "backend_profile": node.get("backend_profile"),
        "inputs": node.get("inputs", {}),
        "parameters": node.get("parameters", {}),
        "resources": node.get("resources", {}),
        "approval_required": requires_approval,
        "warnings": warnings,
    }


def approval_required(
    node: dict[str, Any], spec: dict[str, Any], operation: str | None
) -> bool:
    if node.get("mode", "execute") != "execute":
        return False
    if node.get("backend", "local") == "ssh-slurm":
        return True
    return operation in spec["approval_operations"]


def action_plan(action: str, project: Project, node_id: str | None, details: Any) -> dict[str, Any]:
    return {
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
