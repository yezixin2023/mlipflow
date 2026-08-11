"""Static plugin discovery and runtime adapter loading."""

from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

from .errors import PluginError
from .io import load_mapping


PLUGIN_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
REQUIRED_FIELDS = {
    "schema_version",
    "id",
    "version",
    "api_version",
    "description",
    "implementation",
    "execution",
    "replay",
}


@dataclass(frozen=True)
class PluginSpec:
    path: Path
    raw: dict[str, Any]

    @property
    def plugin_id(self) -> str:
        return str(self.raw["id"])

    @property
    def kind(self) -> str:
        return str(self.raw["implementation"]["kind"])


class Adapter(Protocol):
    def validate(self, context: dict[str, Any]) -> list[dict[str, Any]]: ...

    def plan(self, context: dict[str, Any]) -> dict[str, Any]: ...

    def prepare(self, context: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]: ...

    def check(self, context: dict[str, Any]) -> dict[str, Any]: ...

    def collect(self, context: dict[str, Any]) -> dict[str, Any]: ...

    def replay(self, context: dict[str, Any]) -> dict[str, Any]: ...


def discover_plugins(root: Path) -> dict[str, PluginSpec]:
    """Parse manifests only; never import plugin Python code."""

    if not root.is_dir():
        return {}
    found: dict[str, PluginSpec] = {}
    for manifest in sorted(root.glob("*/plugin.yaml")) + sorted(root.glob("*/plugin.json")):
        raw = load_mapping(manifest)
        validate_plugin(raw, manifest)
        plugin_id = str(raw["id"])
        if plugin_id in found:
            raise PluginError(f"duplicate plugin id {plugin_id}: {found[plugin_id].path}, {manifest}")
        found[plugin_id] = PluginSpec(manifest, raw)
    return found


def validate_plugin(raw: dict[str, Any], source: Path | str = "plugin") -> None:
    missing = sorted(REQUIRED_FIELDS - set(raw))
    if missing:
        raise PluginError(f"{source}: missing fields {missing}")
    if raw.get("schema_version") != 1 or raw.get("api_version") != 1:
        raise PluginError(f"{source}: unsupported schema/api version")
    plugin_id = raw.get("id")
    if not isinstance(plugin_id, str) or not PLUGIN_ID.fullmatch(plugin_id):
        raise PluginError(f"{source}: invalid plugin id {plugin_id!r}")
    implementation = raw.get("implementation")
    if not isinstance(implementation, dict):
        raise PluginError(f"{source}: implementation must be a mapping")
    if implementation.get("kind") not in {"python-adapter", "external-command", "replay-only"}:
        raise PluginError(f"{source}: unsupported implementation kind")
    execution = raw.get("execution")
    if not isinstance(execution, dict) or not isinstance(execution.get("backends"), list):
        raise PluginError(f"{source}: execution.backends must be a list")
    invalid_backends = set(execution["backends"]) - {"local", "slurm", "ssh-slurm"}
    if invalid_backends:
        raise PluginError(f"{source}: invalid backends {sorted(invalid_backends)}")
    replay = raw.get("replay")
    if not isinstance(replay, dict) or not isinstance(replay.get("supported"), bool):
        raise PluginError(f"{source}: replay.supported must be boolean")


def select_plugin(plugins: dict[str, PluginSpec], uses: str) -> PluginSpec:
    requested, _, requested_major = uses.partition("@")
    if requested in plugins:
        spec = plugins[requested]
    else:
        candidates = [spec for key, spec in plugins.items() if key.split(".", 1)[0] == requested]
        if len(candidates) != 1:
            raise PluginError(f"cannot resolve plugin {uses!r}")
        spec = candidates[0]
    if requested_major and not str(spec.raw["version"]).startswith(requested_major + "."):
        raise PluginError(
            f"plugin {spec.plugin_id} version {spec.raw['version']} does not satisfy @{requested_major}"
        )
    return spec


def load_adapter(spec: PluginSpec) -> Adapter:
    """Load adapter only from an explicitly executing command path."""

    entrypoint = spec.raw["implementation"].get("entrypoint")
    if not isinstance(entrypoint, str) or ":" not in entrypoint:
        raise PluginError(f"plugin {spec.plugin_id} has no Python entrypoint")
    filename, object_name = entrypoint.split(":", 1)
    adapter_path = (spec.path.parent / filename).resolve()
    if adapter_path.parent != spec.path.parent.resolve() or not adapter_path.is_file():
        raise PluginError(f"invalid adapter path for {spec.plugin_id}: {filename}")
    module = _load_module(adapter_path, f"mlipflow_adapter_{spec.plugin_id.replace('.', '_')}")
    try:
        factory = getattr(module, object_name)
    except AttributeError as exc:
        raise PluginError(f"adapter object {object_name} not found in {adapter_path}") from exc
    return factory()


def _load_module(path: Path, name: str) -> ModuleType:
    module_spec = importlib.util.spec_from_file_location(name, path)
    if module_spec is None or module_spec.loader is None:
        raise PluginError(f"cannot load adapter module {path}")
    module = importlib.util.module_from_spec(module_spec)
    previous = sys.dont_write_bytecode
    try:
        # Planning must not create ``__pycache__`` inside a plugin directory.
        # SourceFileLoader honours this process flag while executing the module.
        sys.dont_write_bytecode = True
        module_spec.loader.exec_module(module)
    except Exception as exc:
        raise PluginError(f"adapter import failed for {path}: {exc}") from exc
    finally:
        sys.dont_write_bytecode = previous
    return module
