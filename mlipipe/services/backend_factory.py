"""The single place where scheduler backends are constructed.

Before this module, ``SshSlurmBackend`` was instantiated at five separate call
sites, each re-deriving and re-validating the SSH profile out of a plan mapping.
Centralising that has two effects: adding a scheduler means implementing
``SchedulerBackend`` and extending one dispatch, and tests can inject a fake
scheduler instead of patching ``mlipipe.backends.subprocess.run``.

The ``factory`` keyword mirrors the ``template_library`` injection seam that
``make_run_plan`` already uses for remote template reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from ..backends import SchedulerBackend, SshSlurmBackend
from ..errors import BackendError, ConfigError
from .contracts import _cluster_profile


SchedulerFactory = Callable[[str, Optional[str]], SchedulerBackend]


def default_scheduler_factory(backend: str, ssh_profile: str | None) -> SchedulerBackend:
    """Build the concrete scheduler for one backend name."""

    if backend == "ssh-slurm":
        if not isinstance(ssh_profile, str) or not ssh_profile:
            raise ConfigError("ssh-slurm scheduler requires an SSH profile")
        return SshSlurmBackend(ssh_profile)
    raise BackendError(f"stored job id is incompatible with backend {backend}")


def scheduler_for_node(
    backend: str,
    node: dict[str, Any],
    site_path: Path | None,
    *,
    factory: SchedulerFactory | None = None,
) -> SchedulerBackend:
    """Resolve a scheduler from live project/site configuration.

    Used by ``stop``, which must reach the cluster without a pinned plan.
    """

    if backend != "ssh-slurm":
        raise BackendError(f"stored job id is incompatible with backend {backend}")
    build = factory or default_scheduler_factory
    return build(backend, _cluster_profile(node, site_path).ssh_profile)


def scheduler_from_cluster_record(
    record: Any,
    *,
    factory: SchedulerFactory | None = None,
) -> SchedulerBackend:
    """Resolve a scheduler from the ``cluster_profile`` of an approved plan.

    The approved plan, not the current site file, is the authority once a job has
    been submitted; the two are compared separately by
    ``_load_pinned_scheduled_plan``.
    """

    if not isinstance(record, dict) or not isinstance(record.get("ssh_profile"), str):
        raise ConfigError("pinned HPC plan lacks an SSH profile")
    build = factory or default_scheduler_factory
    return build("ssh-slurm", str(record["ssh_profile"]))
