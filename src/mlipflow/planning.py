"""Pure execution planning and approval digests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .config import Project
from .artifacts import fingerprint
from .errors import PluginError
from .io import canonical_json
from .plugins import PluginSpec


def node_plan(project: Project, node: dict[str, Any], plugin: PluginSpec) -> dict[str, Any]:
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
    profile_name = node.get("backend_profile")
    profiles = project.raw.get("backend_profiles", {})
    profile_configuration = (
        profiles.get(profile_name)
        if isinstance(profile_name, str) and isinstance(profiles, dict)
        else None
    )
    plan = {
        "schema_version": 1,
        "action": "run",
        "project_id": project.project_id,
        "node_id": node["id"],
        "plugin": {
            "id": plugin.plugin_id,
            "version": plugin.raw["version"],
            "manifest_digest": "sha256:"
            + hashlib.sha256(canonical_json(plugin.raw).encode("utf-8")).hexdigest(),
            "implementation_fingerprint": _plugin_implementation_fingerprint(plugin),
        },
        "mode": mode,
        "backend": backend,
        "backend_profile": {
            "name": profile_name,
            "configuration": profile_configuration,
        },
        "project_config_digest": _project_config_digest(project),
        "inputs": node.get("inputs", {}),
        "input_fingerprints": _input_fingerprints(project, node.get("inputs", {})),
        "parameters": node.get("parameters", {}),
        "resources": node.get("resources", {}),
        "cost_class": cost_class,
        "warnings": warnings,
    }
    return with_digest(plan)


def _input_fingerprints(project: Project, inputs: Any) -> dict[str, Any]:
    if not isinstance(inputs, dict):
        return {}
    threshold = int(project.raw.get("fingerprints", {}).get("full_hash_max_bytes", 67108864))
    values: dict[str, Any] = {}
    for key, value in sorted(inputs.items()):
        captured = _fingerprint_input_value(value, key, project.root, threshold)
        if captured is not None:
            values[key] = captured
    return values


def _plugin_implementation_fingerprint(plugin: PluginSpec) -> dict[str, Any] | None:
    entrypoint = plugin.raw.get("implementation", {}).get("entrypoint")
    if not isinstance(entrypoint, str) or ":" not in entrypoint:
        return None
    filename = entrypoint.split(":", 1)[0]
    candidate = (plugin.path.parent / filename).resolve()
    if candidate.parent != plugin.path.parent.resolve() or not candidate.is_file():
        return {"path": str(candidate), "exists": False}
    return fingerprint(candidate)


def _fingerprint_input_value(
    value: Any, key: str, project_root: Path, threshold: int
) -> Any:
    explicit_path = isinstance(value, dict) and isinstance(value.get("path"), str)
    path_value = value.get("path") if explicit_path else value if isinstance(value, str) else None
    if isinstance(path_value, str):
        path = resolve_reference(path_value, project_root)
        if path.exists():
            return fingerprint(path, full_hash_max_bytes=threshold)
        if explicit_path or key.endswith(("_manifest", "_file", "_path")):
            return {"path": str(path), "exists": False}
        return None
    if isinstance(value, list):
        captured = {
            str(index): item
            for index, raw in enumerate(value)
            if (item := _fingerprint_input_value(raw, key, project_root, threshold)) is not None
        }
        return captured or None
    return None


def action_plan(action: str, project: Project, node_id: str | None, details: Any) -> dict[str, Any]:
    return with_digest(
        {
            "schema_version": 1,
            "action": action,
            "project_id": project.project_id,
            "project_config_digest": _project_config_digest(project),
            "node_id": node_id,
            "details": details,
        }
    )


def _project_config_digest(project: Project) -> str:
    digest = hashlib.sha256(canonical_json(project.raw).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def with_digest(plan: dict[str, Any]) -> dict[str, Any]:
    unsigned = {key: value for key, value in plan.items() if key != "plan_digest"}
    digest = hashlib.sha256(canonical_json(unsigned).encode("utf-8")).hexdigest()
    return {**unsigned, "plan_digest": f"sha256:{digest}"}


def resolve_reference(value: Any, project_root: Path) -> Path:
    if isinstance(value, str):
        path = Path(value)
    elif isinstance(value, dict) and isinstance(value.get("path"), str):
        path = Path(value["path"])
    else:
        raise ValueError(f"expected a path reference, got {value!r}")
    return path if path.is_absolute() else project_root / path
