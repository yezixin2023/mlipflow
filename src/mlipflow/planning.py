"""Pure execution planning and approval digests.

Approval identity answers "what will run and on what content", so every value
that reaches :func:`with_digest` must be derived from declared configuration or
from file *contents* — never from an inode timestamp or an absolute local path.
See :mod:`mlipflow.artifacts` for the identity/observation split that enforces
this on the filesystem side.

Plan ``schema_version`` is 2.  Version 1 embedded ``mtime_ns`` and absolute
``file://`` URIs in ``implementation_fingerprint`` and ``input_fingerprints``,
which made an approved digest depend on when a file was last touched and on
where the checkout happened to live.  Those fields are replaced by
``implementation_identity``, ``input_identities`` and
``adapter_command_identities``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .config import Project
from .artifacts import content_identity
from .errors import PluginError
from .io import canonical_json
from .plugins import PluginSpec


PLAN_SCHEMA_VERSION = 2


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
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "action": "run",
        "project_id": project.project_id,
        "node_id": node["id"],
        "plugin": {
            "id": plugin.plugin_id,
            "version": plugin.raw["version"],
            "manifest_digest": "sha256:"
            + hashlib.sha256(canonical_json(plugin.raw).encode("utf-8")).hexdigest(),
            "implementation_identity": _plugin_implementation_identity(plugin),
        },
        "mode": mode,
        "backend": backend,
        "backend_profile": profile_name,
        "project_config_digest": _project_config_digest(project),
        "inputs": node.get("inputs", {}),
        "input_identities": _input_identities(project, node.get("inputs", {})),
        "parameters": node.get("parameters", {}),
        "resources": node.get("resources", {}),
        "cost_class": cost_class,
        "warnings": warnings,
    }
    return with_digest(plan)


def _input_identities(project: Project, inputs: Any) -> dict[str, Any]:
    if not isinstance(inputs, dict):
        return {}
    threshold = int(project.raw.get("fingerprints", {}).get("full_hash_max_bytes", 67108864))
    values: dict[str, Any] = {}
    for key, value in sorted(inputs.items()):
        captured = _input_value_identity(value, key, project.root, threshold)
        if captured is not None:
            values[key] = captured
    return values


def _plugin_implementation_identity(plugin: PluginSpec) -> dict[str, Any] | None:
    """Bind the adapter source by content, located relative to its plugin.

    The locator is plugin-relative rather than absolute so that installing the
    same plugin under a different prefix — a wheel's ``share/mlipflow/plugins``
    versus a source checkout — yields the same approval identity.
    """

    entrypoint = plugin.raw.get("implementation", {}).get("entrypoint")
    if not isinstance(entrypoint, str) or ":" not in entrypoint:
        return None
    filename = entrypoint.split(":", 1)[0]
    plugin_root = plugin.path.parent
    candidate = (plugin_root / filename).resolve()
    if candidate.parent != plugin_root.resolve() or not candidate.is_file():
        return {"locator": filename, "exists": False}
    return content_identity(candidate, root=plugin_root)


def _input_value_identity(
    value: Any, key: str, project_root: Path, threshold: int
) -> Any:
    explicit_path = isinstance(value, dict) and isinstance(value.get("path"), str)
    path_value = value.get("path") if explicit_path else value if isinstance(value, str) else None
    if isinstance(path_value, str):
        path = resolve_reference(path_value, project_root)
        if path.exists():
            return content_identity(
                path, root=project_root, full_hash_max_bytes=threshold
            )
        if explicit_path or key.endswith(("_manifest", "_file", "_path")):
            # The declared reference is already carried verbatim in the plan's
            # ``inputs``; recording the absolute resolution here would only make
            # the digest depend on where the checkout lives.
            return {"locator": _project_locator(path, project_root), "exists": False}
        return None
    if isinstance(value, list):
        captured = {
            str(index): item
            for index, raw in enumerate(value)
            if (item := _input_value_identity(raw, key, project_root, threshold)) is not None
        }
        return captured or None
    return None


def _project_locator(path: Path, project_root: Path) -> str | None:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except (ValueError, OSError):
        return None


def action_plan(action: str, project: Project, node_id: str | None, details: Any) -> dict[str, Any]:
    return with_digest(
        {
            "schema_version": PLAN_SCHEMA_VERSION,
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
