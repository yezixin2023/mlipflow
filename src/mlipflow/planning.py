"""Execution planning and narrowly scoped approval identity.

A plan contains both executable facts and explanatory material. Only the
former belongs in an approval digest. In particular, warnings, diagnostics,
plugin descriptions, and raw project/site/manifest configuration are useful to
display but are not authority to execute anything.

``plan_digest`` is therefore the digest of :func:`execution_identity`, not of
the plan dictionary. This keeps approvals stable when human-readable material
changes while still binding commands, content inputs, resources, backends, and
the exact scripts or staged files used by a scheduled run.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .artifacts import content_identity
from .config import Project
from .errors import PluginError
from .io import canonical_json
from .plugins import PluginSpec


PLAN_SCHEMA_VERSION = 3
APPROVAL_SCHEMA_VERSION = 1


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
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "action": "run",
        "project_id": project.project_id,
        "node_id": node["id"],
        "attempt": attempt,
        "plugin": {
            "id": plugin.plugin_id,
            "version": plugin.raw["version"],
            "implementation_identity": _plugin_implementation_identity(plugin),
        },
        "mode": mode,
        "backend": backend,
        "backend_profile": node.get("backend_profile"),
        "inputs": node.get("inputs", {}),
        "input_identities": _input_identities(project, node.get("inputs", {})),
        "parameters": node.get("parameters", {}),
        "resources": node.get("resources", {}),
        "cost_class": cost_class,
        "approval_required": _approval_required(node, plugin),
        "warnings": warnings,
    }
    return with_digest(plan)


def _approval_required(node: dict[str, Any], plugin: PluginSpec) -> bool:
    """Return whether executing this node needs explicit approval."""

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


def _input_identities(project: Project, inputs: Any) -> dict[str, Any]:
    if not isinstance(inputs, dict):
        return {}
    values: dict[str, Any] = {}
    for key, value in sorted(inputs.items()):
        captured = _input_value_identity(value, key, project.root)
        if captured is not None:
            values[key] = captured
    return values


def _plugin_implementation_identity(plugin: PluginSpec) -> dict[str, Any] | None:
    entrypoint = plugin.raw.get("implementation", {}).get("entrypoint")
    if not isinstance(entrypoint, str) or ":" not in entrypoint:
        return None
    filename = entrypoint.split(":", 1)[0]
    plugin_root = plugin.path.parent
    candidate = (plugin_root / filename).resolve()
    if candidate.parent != plugin_root.resolve() or not candidate.is_file():
        return {"locator": filename, "exists": False}
    return content_identity(candidate, root=plugin_root)


def _input_value_identity(value: Any, key: str, project_root: Path) -> Any:
    explicit_path = isinstance(value, dict) and isinstance(value.get("path"), str)
    path_value = value.get("path") if explicit_path else value if isinstance(value, str) else None
    if isinstance(path_value, str):
        path = resolve_reference(path_value, project_root)
        if path.exists():
            return content_identity(path, root=project_root)
        if explicit_path or key.endswith(("_manifest", "_file", "_path")):
            return {"locator": _project_locator(path, project_root), "exists": False}
        return None
    if isinstance(value, list):
        captured = {
            str(index): item
            for index, raw in enumerate(value)
            if (item := _input_value_identity(raw, key, project_root)) is not None
        }
        return captured or None
    return None


def _project_locator(path: Path, project_root: Path) -> str | None:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except (ValueError, OSError):
        return None


def action_plan(action: str, project: Project, node_id: str | None, details: Any) -> dict[str, Any]:
    """Describe a lifecycle action without manufacturing an approval token."""

    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "action": action,
        "project_id": project.project_id,
        "node_id": node_id,
        "details": details,
    }


def execution_identity(plan: dict[str, Any]) -> dict[str, Any]:
    """Return the execution-relevant semantic projection of a run plan."""

    if plan.get("action") != "run":
        raise ValueError("only run plans have an execution approval identity")
    adapter_plan = plan.get("adapter_plan")
    scheduled = (
        adapter_plan.get("scheduled_execution")
        if isinstance(adapter_plan, dict)
        else None
    )
    identity: dict[str, Any] = {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "action": "run",
        "project_id": plan.get("project_id"),
        "node_id": plan.get("node_id"),
        "attempt": plan.get("attempt"),
        "mode": plan.get("mode"),
        "backend": plan.get("backend"),
        "backend_profile": plan.get("backend_profile"),
        "implementation": (
            plan.get("plugin", {}).get("implementation_identity")
            if isinstance(plan.get("plugin"), dict)
            else None
        ),
        "resources": plan.get("resources", {}),
    }
    if isinstance(adapter_plan, dict):
        if isinstance(scheduled, dict):
            identity["execution"] = {
                "staged_files": [
                    {
                        "remote_name": item.get("remote_name"),
                        "sha256": item.get("sha256"),
                        "size_bytes": item.get("size_bytes"),
                    }
                    for item in scheduled.get("staged_files", [])
                    if isinstance(item, dict)
                ],
                "fetch_outputs": [
                    {
                        key: item.get(key)
                        for key in (
                            "remote_name",
                            "remote_path",
                            "local_name",
                            "required",
                            "max_bytes",
                        )
                        if key in item
                    }
                    for item in scheduled.get("fetch_outputs", [])
                    if isinstance(item, dict)
                ],
            }
            if isinstance(adapter_plan.get("failure_salvage"), dict):
                identity["execution"]["failure_salvage"] = adapter_plan[
                    "failure_salvage"
                ].get("fetch_remote_names")
        else:
            identity["execution"] = {
                key: adapter_plan.get(key)
                for key in ("argv", "cwd", "environment")
                if key in adapter_plan
            }
            identity["command_identities"] = plan.get("adapter_command_identities", {})
            identity["inputs"] = _semantic_inputs(
                plan.get("inputs", {}), plan.get("input_identities", {})
            )
    else:
        identity["inputs"] = _semantic_inputs(
            plan.get("inputs", {}), plan.get("input_identities", {})
        )
        identity["parameters"] = plan.get("parameters", {})
    hpc = plan.get("hpc_execution")
    if isinstance(hpc, dict):
        cluster = hpc.get("cluster_profile")
        if not isinstance(cluster, dict):
            cluster = {}
        # Resources and profile selection are already represented by their
        # resolved HPC forms; carrying the raw node values as well adds no
        # integrity property.
        identity.pop("resources", None)
        identity.pop("backend_profile", None)
        identity["hpc"] = {
            "cluster": {
                key: cluster.get(key)
                for key in ("name", "backend", "ssh_profile", "work_root", "scheduler")
                if key in cluster
            },
            "execution_model": hpc.get("execution_model"),
            "resources": hpc.get("resources"),
            "rendered_scripts": hpc.get("rendered_scripts"),
            "workspace": hpc.get("workspace"),
            "fresh_workspace_required": hpc.get("fresh_workspace_required"),
            "overwrite": hpc.get("overwrite"),
        }
    return identity


def _semantic_inputs(declared: Any, captured: Any) -> Any:
    """Replace path references with their content identity exactly once."""

    if isinstance(captured, dict) and any(
        key in captured for key in ("content", "exists", "locator")
    ):
        return captured
    if isinstance(declared, dict):
        captured_values = captured if isinstance(captured, dict) else {}
        return {
            key: _semantic_inputs(value, captured_values.get(key))
            for key, value in sorted(declared.items())
        }
    if isinstance(declared, list):
        captured_values = captured if isinstance(captured, dict) else {}
        return [
            _semantic_inputs(value, captured_values.get(str(index)))
            for index, value in enumerate(declared)
        ]
    return captured if captured is not None else declared


def with_digest(plan: dict[str, Any]) -> dict[str, Any]:
    unsigned = {key: value for key, value in plan.items() if key != "plan_digest"}
    payload = canonical_json(execution_identity(unsigned)).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return {**unsigned, "plan_digest": f"sha256:{digest}"}


def resolve_reference(value: Any, project_root: Path) -> Path:
    if isinstance(value, str):
        path = Path(value)
    elif isinstance(value, dict) and isinstance(value.get("path"), str):
        path = Path(value["path"])
    else:
        raise ValueError(f"expected a path reference, got {value!r}")
    return path if path.is_absolute() else project_root / path
