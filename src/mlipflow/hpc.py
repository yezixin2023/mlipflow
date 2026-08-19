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
# After substitution, only a *residual placeholder* is an error: something that
# tried to be MLIPFlow template syntax and was not recognised, such as
# ``{{lowercase}}``, ``{{ NAME }}`` or a Jinja-style ``{{NAME|filter}}``.
#
# The previous check rejected any surviving ``{{`` or ``}}`` anywhere in the
# rendered text, which made ordinary shell and JSON impossible to write: a JSON
# object closing after a nested one (``{"a":{"b":1}}``) or a brace expansion
# directly before a literal brace (``"${value}}"``) both end in ``}}`` without
# any templating involved.  Requiring a matching ``{{ ... }}`` pair with no
# braces between keeps the guard against unsupported templating while letting
# those through.
RESIDUAL_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
TEMPLATE_FAMILY = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
EXECUTION_MODELS = frozenset({"single-python", "mpi"})
MEMORY = re.compile(r"^[1-9][0-9]*(?:[KMGTP](?:i?B)?)?$")
WALLTIME = re.compile(r"^[0-9]{2,3}:[0-5][0-9]:[0-5][0-9]$")
MAX_TEMPLATE_BYTES = 1024 * 1024

_SLURM_CPU_SEMANTICS = {
    "single-python": {
        "cpus_meaning": "threads-per-process",
        "slurm_ntasks": 1,
        "slurm_cpus_per_task": "CPUS",
    },
    "mpi": {
        "cpus_meaning": "mpi-task-count",
        "slurm_ntasks": "CPUS",
        "slurm_cpus_per_task": "site-owned-literal",
    },
}


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
    profile: ClusterProfile,
    project_id: str,
    node_id: str,
    attempt: int,
    submission_id: str | None = None,
) -> dict[str, str]:
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ConfigError("HPC attempt must be a positive integer")
    run_dir = PurePosixPath(profile.work_root) / project_id / node_id / f"attempt-{attempt:04d}"
    if submission_id is not None:
        if not TEMPLATE_FAMILY.fullmatch(submission_id):
            raise ConfigError("HPC submission id must be a safe identifier")
        run_dir = run_dir / "submissions" / submission_id
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
    submission_id: str | None = None,
) -> dict[str, str]:
    workspace = remote_attempt_workspace(
        profile, project_id, node_id, attempt, submission_id
    )
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
    allowed_variables: frozenset[str] = TEMPLATE_VARIABLES,
) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_TEMPLATE_BYTES:
        raise ConfigError(f"remote template is too large: {template_name}")
    if "\x00" in text:
        raise ConfigError(f"remote template contains NUL: {template_name}")
    used = set(PLACEHOLDER.findall(text))
    unknown = sorted(used - allowed_variables)
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
    residual = RESIDUAL_TEMPLATE.search(rendered)
    if residual is not None:
        raise ConfigError(
            f"remote template {template_name} contains unsupported template syntax: "
            f"{residual.group(0)}"
        )
    normalized = rendered.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n"
    return normalized


def _slurm_directive_values(text: str, option: str) -> list[str]:
    pattern = re.compile(
        rf"^[ \t]*#SBATCH[ \t]+--{re.escape(option)}(?:=|[ \t]+)([^ \t#\r\n]+)",
        re.MULTILINE,
    )
    return pattern.findall(text)


def validate_slurm_cpu_semantics(
    text: str, execution_model: str, *, template_name: str
) -> None:
    """Fail closed when a v3 submit template maps ``CPUS`` incorrectly.

    A single Python process must receive the abstract CPU budget as threads on
    its one Slurm task. MPI contracts instead expose that budget as the task/rank
    count. Keeping these layouts distinct prevents Lightning from interpreting
    an accidental ``--ntasks={{CPUS}}`` as a distributed launch while preserving
    the rank semantics required by VASP, LAMMPS, and LASP.
    """

    if execution_model not in EXECUTION_MODELS:
        raise ConfigError(f"unsupported scheduled execution model: {execution_model!r}")
    ntasks = _slurm_directive_values(text, "ntasks")
    cpus_per_task = _slurm_directive_values(text, "cpus-per-task")
    if execution_model == "single-python":
        if ntasks != ["1"] or cpus_per_task != ["{{CPUS}}"]:
            raise ConfigError(
                f"remote template {template_name} must map single-python resources "
                "as --ntasks=1 and --cpus-per-task={{CPUS}}"
            )
        return
    if ntasks != ["{{CPUS}}"] or "{{CPUS}}" in cpus_per_task:
        raise ConfigError(
            f"remote template {template_name} must map mpi resources as "
            "--ntasks={{CPUS}} and must not also use CPUS for --cpus-per-task"
        )


def resolve_hpc_execution_plan(
    *,
    profile: ClusterProfile,
    project_id: str,
    node_id: str,
    attempt: int,
    resources_value: Any,
    scheduled_execution: dict[str, Any],
    library: TemplateLibrary,
    submission_id: str | None = None,
) -> dict[str, Any]:
    resources = validate_hpc_resources(resources_value)
    family = scheduled_execution.get("template_family")
    if not isinstance(family, str) or not TEMPLATE_FAMILY.fullmatch(family):
        raise ConfigError("scheduled_execution.template_family is required and must be safe")
    execution_model = scheduled_execution.get("execution_model")
    if not isinstance(execution_model, str) or execution_model not in EXECUTION_MODELS:
        raise ConfigError(
            "scheduled_execution.execution_model must be single-python or mpi"
        )
    device = "gpu" if resources.gpus > 0 else "cpu"
    submit_template = f"slurm/{execution_model}/{device}.sbatch"
    cpu_semantics = dict(_SLURM_CPU_SEMANTICS[execution_model])
    run_template = f"{family}/run.sh"
    selected = {
        "submit.sbatch": (submit_template, SUBMIT_REQUIRED),
        "run.sh": (run_template, RUN_REQUIRED),
    }
    variables = template_variables(
        profile, project_id, node_id, attempt, resources, submission_id
    )
    plugin_variables = scheduled_execution.get("template_variables", {})
    if not isinstance(plugin_variables, dict):
        raise ConfigError("scheduled_execution.template_variables must be a mapping")
    variables.update({str(name): str(value) for name, value in plugin_variables.items()})
    allowed_variables = frozenset({*TEMPLATE_VARIABLES, *plugin_variables})
    template_paths: dict[str, str] = {}
    rendered_scripts: dict[str, str] = {}
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
        if destination == "submit.sbatch":
            validate_slurm_cpu_semantics(
                content, execution_model, template_name=relative
            )
        rendered = render_template(
            content,
            variables,
            template_name=relative,
            required=required,
            allowed_variables=allowed_variables,
        )
        used_variables.update(PLACEHOLDER.findall(content))
        template_paths[destination] = relative
        rendered_scripts[destination] = rendered
    missing_contract = sorted(TEMPLATE_VARIABLES - used_variables)
    if missing_contract:
        raise ConfigError(
            "selected remote templates do not cover the parameter contract: "
            + ", ".join(missing_contract)
        )
    return {
        "schema_version": 1,
        "cluster_profile": profile.to_plan_dict(),
        "execution_model": execution_model,
        "cpu_resource_semantics": cpu_semantics,
        "resources": resources.to_plan_dict(),
        "template_variables": variables,
        "template_paths": template_paths,
        "rendered_scripts": rendered_scripts,
        "workspace": remote_attempt_workspace(
            profile, project_id, node_id, attempt, submission_id
        ),
        "fresh_workspace_required": True,
        "overwrite": False,
    }
