"""Deterministic HPC template selection, validation, rendering, and planning."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Protocol

from .errors import ConfigError
from .site import ClusterProfile


TEMPLATE_VARIABLES = frozenset(
    {
        "PROJECT_ID",
        "NODE_ID",
        "ATTEMPT",
        "RUN_DIR",
        "INPUT_DIR",
        "OUTPUT_DIR",
        "LOG_DIR",
        "CPUS",
        "GPUS",
        "MEMORY",
        "WALLTIME",
    }
)
SUBMIT_REQUIRED = frozenset(
    {"RUN_DIR", "LOG_DIR", "CPUS", "GPUS", "MEMORY", "WALLTIME"}
)
RUN_REQUIRED = frozenset({"RUN_DIR", "INPUT_DIR", "OUTPUT_DIR"})
PLACEHOLDER = re.compile(r"{{([A-Z][A-Z0-9_]*)}}")
TEMPLATE_FAMILY = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
MEMORY = re.compile(r"^[1-9][0-9]*(?:[KMGTP](?:i?B)?)?$")
WALLTIME = re.compile(r"^[0-9]{2,3}:[0-5][0-9]:[0-5][0-9]$")
MAX_TEMPLATE_BYTES = 1024 * 1024


class TemplateLibrary(Protocol):
    """Read-only provider for a cluster's persistent remote template library."""

    def read_template(self, template_root: str, relative_path: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class HpcResources:
    cpus: int
    gpus: int
    memory: str
    walltime: str

    def to_plan_dict(self) -> dict[str, Any]:
        return {
            "cpus": self.cpus,
            "gpus": self.gpus,
            "memory": self.memory,
            "walltime": self.walltime,
        }


def validate_hpc_resources(value: Any) -> HpcResources:
    if not isinstance(value, dict):
        raise ConfigError("ssh-slurm resources must be a mapping")
    expected = {"cpus", "gpus", "memory", "walltime"}
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise ConfigError("ssh-slurm resources lack: " + ", ".join(missing))
    if unknown:
        raise ConfigError(
            "ssh-slurm resources contain unsupported/site-specific fields: "
            + ", ".join(unknown)
        )
    cpus = value.get("cpus")
    gpus = value.get("gpus")
    memory = value.get("memory")
    walltime = value.get("walltime")
    if isinstance(cpus, bool) or not isinstance(cpus, int) or cpus < 1:
        raise ConfigError("ssh-slurm resources.cpus must be a positive integer")
    if isinstance(gpus, bool) or not isinstance(gpus, int) or gpus < 0:
        raise ConfigError("ssh-slurm resources.gpus must be a non-negative integer")
    if not isinstance(memory, str) or not MEMORY.fullmatch(memory):
        raise ConfigError("ssh-slurm resources.memory must look like 64G or 65536M")
    if not isinstance(walltime, str) or not WALLTIME.fullmatch(walltime):
        raise ConfigError("ssh-slurm resources.walltime must use HH:MM:SS")
    return HpcResources(cpus=cpus, gpus=gpus, memory=memory, walltime=walltime)


def remote_attempt_workspace(
    profile: ClusterProfile, project_id: str, node_id: str, attempt: int
) -> dict[str, str]:
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ConfigError("HPC attempt must be a positive integer")
    run_dir = PurePosixPath(profile.work_root) / project_id / node_id / f"attempt-{attempt:04d}"
    return {
        "run_dir": str(run_dir),
        "input_dir": str(run_dir / "input"),
        "output_dir": str(run_dir / "output"),
        "log_dir": str(run_dir / "logs"),
        "completion_path": str(run_dir / "completion.json"),
    }


def template_variables(
    profile: ClusterProfile,
    project_id: str,
    node_id: str,
    attempt: int,
    resources: HpcResources,
) -> dict[str, str]:
    workspace = remote_attempt_workspace(profile, project_id, node_id, attempt)
    return {
        "PROJECT_ID": project_id,
        "NODE_ID": node_id,
        "ATTEMPT": f"{attempt:04d}",
        "RUN_DIR": workspace["run_dir"],
        "INPUT_DIR": workspace["input_dir"],
        "OUTPUT_DIR": workspace["output_dir"],
        "LOG_DIR": workspace["log_dir"],
        "CPUS": str(resources.cpus),
        "GPUS": str(resources.gpus),
        "MEMORY": resources.memory,
        "WALLTIME": resources.walltime,
    }


def render_template(
    text: str,
    variables: dict[str, str],
    *,
    template_name: str,
    required: frozenset[str],
) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_TEMPLATE_BYTES:
        raise ConfigError(f"remote template is too large: {template_name}")
    if "\x00" in text:
        raise ConfigError(f"remote template contains NUL: {template_name}")
    used = set(PLACEHOLDER.findall(text))
    unknown = sorted(used - TEMPLATE_VARIABLES)
    if unknown:
        raise ConfigError(
            f"remote template {template_name} uses unknown variables: {', '.join(unknown)}"
        )
    missing = sorted(required - used)
    if missing:
        raise ConfigError(
            f"remote template {template_name} lacks required variables: {', '.join(missing)}"
        )
    unavailable = sorted(used - set(variables))
    if unavailable:
        raise ConfigError(
            f"remote template {template_name} has unresolved variables: {', '.join(unavailable)}"
        )
    rendered = PLACEHOLDER.sub(lambda match: variables[match.group(1)], text)
    if "{{" in rendered or "}}" in rendered:
        raise ConfigError(f"remote template {template_name} contains unsupported template syntax")
    normalized = rendered.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n"
    return normalized


def resolve_hpc_execution_plan(
    *,
    profile: ClusterProfile,
    project_id: str,
    node_id: str,
    attempt: int,
    resources_value: Any,
    scheduled_execution: dict[str, Any],
    library: TemplateLibrary,
) -> dict[str, Any]:
    resources = validate_hpc_resources(resources_value)
    family = scheduled_execution.get("template_family")
    if not isinstance(family, str) or not TEMPLATE_FAMILY.fullmatch(family):
        raise ConfigError("scheduled_execution.template_family is required and must be safe")
    submit_template = "slurm/gpu.sbatch" if resources.gpus > 0 else "slurm/cpu.sbatch"
    run_template = f"{family}/run.sh"
    selected = {
        "submit.sbatch": (submit_template, SUBMIT_REQUIRED),
        "run.sh": (run_template, RUN_REQUIRED),
    }
    variables = template_variables(profile, project_id, node_id, attempt, resources)
    template_records: dict[str, dict[str, Any]] = {}
    rendered_records: dict[str, dict[str, Any]] = {}
    used_variables: set[str] = set()
    for destination, (relative, required) in selected.items():
        raw = library.read_template(profile.remote_template_root, relative)
        if isinstance(raw, dict) and raw.get("root_exists") is False:
            raise ConfigError(
                "remote template root is missing or unsafe: "
                + profile.remote_template_root
            )
        content = raw.get("content") if isinstance(raw, dict) else None
        if not isinstance(content, str):
            raise ConfigError(f"required remote template is missing: {relative}")
        raw_bytes = content.encode("utf-8")
        digest = "sha256:" + hashlib.sha256(raw_bytes).hexdigest()
        declared_digest = raw.get("sha256")
        declared_size = raw.get("size_bytes")
        if declared_digest != digest or declared_size != len(raw_bytes):
            raise ConfigError(f"remote template identity is inconsistent: {relative}")
        rendered = render_template(
            content, variables, template_name=relative, required=required
        )
        used_variables.update(PLACEHOLDER.findall(content))
        rendered_bytes = rendered.encode("utf-8")
        template_records[destination] = {
            "relative_path": relative,
            "sha256": digest,
            "size_bytes": len(raw_bytes),
        }
        rendered_records[destination] = {
            "content": rendered,
            "sha256": "sha256:" + hashlib.sha256(rendered_bytes).hexdigest(),
            "size_bytes": len(rendered_bytes),
        }
    missing_contract = sorted(TEMPLATE_VARIABLES - used_variables)
    if missing_contract:
        raise ConfigError(
            "selected remote templates do not cover the parameter contract: "
            + ", ".join(missing_contract)
        )
    return {
        "schema_version": 1,
        "cluster_profile": profile.to_plan_dict(),
        "resources": resources.to_plan_dict(),
        "template_variables": variables,
        "templates": template_records,
        "rendered_scripts": rendered_records,
        "workspace": remote_attempt_workspace(profile, project_id, node_id, attempt),
        "fresh_workspace_required": True,
        "overwrite": False,
    }
