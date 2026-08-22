"""Project configuration loading without implicit filesystem discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from .errors import ConfigError
from .hpc import validate_hpc_resources
from .io import load_mapping


PROJECT_FILENAMES = ("project.yaml", "project.yml", "project.json")
SENSITIVE_KEY = re.compile(
    r"(password|passwd|token|secret|credential|api[_-]?key|private[_-]?key|identityfile)", re.I
)
IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
PLUGIN_REFERENCE = re.compile(r"^[a-z0-9][a-z0-9._-]*(?:@[0-9]+)?$")


@dataclass(frozen=True)
class Project:
    root: Path
    path: Path
    raw: dict[str, Any]

    @property
    def project_id(self) -> str:
        return str(self.raw["project"]["id"])

    @property
    def nodes(self) -> list[dict[str, Any]]:
        return list(self.raw["workflow"]["nodes"])

    def node(self, node_id: str) -> dict[str, Any]:
        matches = [node for node in self.nodes if node.get("id") == node_id]
        if not matches:
            raise ConfigError(f"unknown workflow node: {node_id}")
        return matches[0]


def resolve_project_file(path: Path) -> Path:
    candidate = path.resolve()
    if candidate.is_file():
        return candidate
    for filename in PROJECT_FILENAMES:
        direct = candidate / filename
        if direct.is_file():
            return direct
    raise ConfigError(f"no project file in {candidate}")


def load_project(path: Path) -> Project:
    project_file = resolve_project_file(path)
    raw = load_mapping(project_file)
    validate_project(raw, project_file)
    return Project(root=project_file.parent, path=project_file, raw=raw)


def validate_project(raw: dict[str, Any], source: Path | str = "project") -> None:
    _reject_embedded_credentials(raw, source)
    if "backend_profiles" in raw:
        raise ConfigError(
            f"{source}: project.backend_profiles is no longer supported; configure named "
            "clusters in ~/.mlipflow/site.yaml"
        )
    if raw.get("schema_version") != 1:
        raise ConfigError(f"{source}: schema_version must be 1")
    project = raw.get("project")
    if (
        not isinstance(project, dict)
        or not isinstance(project.get("id"), str)
        or not IDENTIFIER.fullmatch(project["id"])
    ):
        raise ConfigError(f"{source}: project.id is required")
    workflow = raw.get("workflow")
    nodes = workflow.get("nodes") if isinstance(workflow, dict) else None
    if not isinstance(nodes, list):
        raise ConfigError(f"{source}: workflow.nodes must be a list")
    ids: set[str] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ConfigError(f"{source}: node {index} must be a mapping")
        node_id = node.get("id")
        uses = node.get("uses")
        if not isinstance(node_id, str) or not IDENTIFIER.fullmatch(node_id):
            raise ConfigError(
                f"{source}: node {index} id must match {IDENTIFIER.pattern!r}"
            )
        if node_id in ids:
            raise ConfigError(f"{source}: duplicate node id {node_id}")
        if not isinstance(uses, str) or not PLUGIN_REFERENCE.fullmatch(uses):
            raise ConfigError(
                f"{source}: node {node_id} uses must match {PLUGIN_REFERENCE.pattern!r}"
            )
        ids.add(node_id)
        backend = node.get("backend", "local")
        if backend not in {"local", "ssh-slurm"}:
            raise ConfigError(
                f"{source}: node {node_id} backend must be local or ssh-slurm"
            )
        if backend == "ssh-slurm":
            embedded_scheduler = sorted(
                {"partition", "partition_candidates", "scheduler", "ssh_profile"}
                & set(node)
            )
            if embedded_scheduler:
                raise ConfigError(
                    f"{source}: ssh-slurm node {node_id} must not provide "
                    f"{', '.join(embedded_scheduler)}; scheduler routing is site-owned"
                )
            profile = node.get("backend_profile")
            if profile is not None and (
                not isinstance(profile, str) or not IDENTIFIER.fullmatch(profile)
            ):
                raise ConfigError(
                    f"{source}: ssh-slurm node {node_id} backend_profile must be a safe identifier"
                )
            validate_hpc_resources(node.get("resources"))
        parameters = node.get("parameters", {})
        if backend == "ssh-slurm" and isinstance(parameters, dict):
            forbidden = sorted({"submit_script", "remote_cwd"} & set(parameters))
            if forbidden:
                raise ConfigError(
                    f"{source}: node {node_id} must not provide {', '.join(forbidden)}; "
                    "HPC scripts and workspaces come from the selected site profile and "
                    "remote template library"
                )
    for node in nodes:
        needs = node.get("needs", [])
        if not isinstance(needs, list) or not all(isinstance(item, str) for item in needs):
            raise ConfigError(f"{source}: node {node['id']} needs must be a string list")
        missing = sorted(set(needs) - ids)
        if missing:
            raise ConfigError(f"{source}: node {node['id']} needs unknown nodes {missing}")
        if node["id"] in needs:
            raise ConfigError(f"{source}: node {node['id']} depends on itself")
    _assert_acyclic(nodes, source)


def _reject_embedded_credentials(value: Any, source: Path | str, trail: str = "") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            location = f"{trail}.{key}" if trail else str(key)
            if SENSITIVE_KEY.search(str(key)):
                raise ConfigError(
                    f"{source}: embedded credential field {location!r} is forbidden; use an SSH profile"
                )
            _reject_embedded_credentials(item, source, location)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_embedded_credentials(item, source, f"{trail}[{index}]")
    elif isinstance(value, str):
        if "BEGIN " in value and "PRIVATE KEY" in value:
            raise ConfigError(f"{source}: embedded private-key material is forbidden at {trail}")
        if re.search(r"\bBearer\s+\S+", value, re.I) or re.search(
            r"://[^/\s:@]+:[^/@\s]+@", value
        ):
            raise ConfigError(
                f"{source}: embedded credential-like value is forbidden at {trail}"
            )


def _assert_acyclic(nodes: list[dict[str, Any]], source: Path | str) -> None:
    graph = {str(node["id"]): list(node.get("needs", [])) for node in nodes}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ConfigError(f"{source}: dependency cycle includes {node_id}")
        if node_id in visited:
            return
        visiting.add(node_id)
        for dependency in graph[node_id]:
            visit(dependency)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in graph:
        visit(node_id)
