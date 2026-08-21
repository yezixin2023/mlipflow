"""User-local cluster control-plane configuration.

Project manifests name a backend profile but never contain SSH targets, remote
roots, modules, partitions, or executable paths.  Those site-owned details live
in ``~/.mlipflow/site.yaml`` (or an explicitly supplied test/config path).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .errors import ConfigError
from .io import load_mapping


DEFAULT_SITE_PATH = Path("~/.mlipflow/site.yaml")
PROFILE_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SSH_PROFILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
REMOTE_PATH = re.compile(r"^/[A-Za-z0-9_./+\-]+$")
PARTITION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+\-]*$")
CLUSTER_FIELDS = frozenset(
    {"backend", "ssh_profile", "remote_template_root", "work_root", "scheduler"}
)
REQUIRED_CLUSTER_FIELDS = frozenset(
    {"backend", "ssh_profile", "remote_template_root", "work_root"}
)
SCHEDULER_FIELDS = frozenset({"partition_candidates", "memory_constraint"})
REQUIRED_SCHEDULER_FIELDS = frozenset({"partition_candidates"})
MEMORY_CONSTRAINTS = frozenset({"reported", "unreported"})


@dataclass(frozen=True)
class SchedulerConfig:
    """Site-owned scheduler routing policy for one cluster."""

    partition_candidates: tuple[str, ...]
    memory_constraint: str = "reported"

    def to_plan_dict(self) -> dict[str, Any]:
        return {
            "partition_candidates": list(self.partition_candidates),
            "memory_constraint": self.memory_constraint,
        }


@dataclass(frozen=True)
class ClusterProfile:
    """Resolved, credential-free configuration for one named cluster."""

    name: str
    backend: str
    ssh_profile: str
    remote_template_root: str
    work_root: str
    scheduler: SchedulerConfig | None = None

    def to_plan_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "backend": self.backend,
            "ssh_profile": self.ssh_profile,
            "remote_template_root": self.remote_template_root,
            "work_root": self.work_root,
        }
        if self.scheduler is not None:
            result["scheduler"] = self.scheduler.to_plan_dict()
        return result


@dataclass(frozen=True)
class SiteConfig:
    """Validated collection of named cluster profiles."""

    path: Path
    raw: dict[str, Any]
    clusters: dict[str, ClusterProfile]

    def cluster(self, name: str | None) -> ClusterProfile:
        if not isinstance(name, str) or not name:
            raise ConfigError("ssh-slurm node requires backend_profile")
        try:
            return self.clusters[name]
        except KeyError as exc:
            raise ConfigError(f"site config has no cluster profile {name!r}") from exc


def default_site_path() -> Path:
    """Return the conventional path without reading it."""

    return DEFAULT_SITE_PATH.expanduser()


def load_site_config(path: Path | None = None) -> SiteConfig:
    """Load one explicit or conventional user-local site configuration."""

    candidate = (path if path is not None else default_site_path()).expanduser().absolute()
    if candidate.is_symlink() or not candidate.is_file():
        raise ConfigError(
            f"site config is missing or unsafe: {candidate}; create ~/.mlipflow/site.yaml "
            "during a separate site bootstrap step or pass --site"
        )
    candidate = candidate.resolve()
    raw = load_mapping(candidate)
    clusters = validate_site_config(raw, candidate)
    return SiteConfig(path=candidate, raw=raw, clusters=clusters)


def validate_site_config(
    raw: dict[str, Any], source: Path | str = "site config"
) -> dict[str, ClusterProfile]:
    unknown_root = sorted(set(raw) - {"schema_version", "clusters"})
    if unknown_root:
        raise ConfigError(f"{source}: unsupported top-level fields: {', '.join(unknown_root)}")
    if raw.get("schema_version") != 1:
        raise ConfigError(f"{source}: schema_version must be 1")
    values = raw.get("clusters")
    if not isinstance(values, dict) or not values:
        raise ConfigError(f"{source}: clusters must be a non-empty mapping")
    clusters: dict[str, ClusterProfile] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not PROFILE_NAME.fullmatch(name):
            raise ConfigError(f"{source}: invalid cluster profile name {name!r}")
        if not isinstance(value, dict):
            raise ConfigError(f"{source}: cluster {name!r} must be a mapping")
        missing = sorted(REQUIRED_CLUSTER_FIELDS - set(value))
        unknown = sorted(set(value) - CLUSTER_FIELDS)
        if missing:
            raise ConfigError(f"{source}: cluster {name!r} lacks {', '.join(missing)}")
        if unknown:
            raise ConfigError(
                f"{source}: cluster {name!r} has unsupported fields: {', '.join(unknown)}"
            )
        backend = value.get("backend")
        ssh_profile = value.get("ssh_profile")
        template_root = value.get("remote_template_root")
        work_root = value.get("work_root")
        scheduler_value = value.get("scheduler")
        if backend != "ssh-slurm":
            raise ConfigError(f"{source}: cluster {name!r} backend must be ssh-slurm")
        if not isinstance(ssh_profile, str) or not SSH_PROFILE.fullmatch(ssh_profile):
            raise ConfigError(f"{source}: cluster {name!r} ssh_profile is unsafe")
        for field, path_value in (
            ("remote_template_root", template_root),
            ("work_root", work_root),
        ):
            if (
                not isinstance(path_value, str)
                or not REMOTE_PATH.fullmatch(path_value)
                or not PurePosixPath(path_value).is_absolute()
                or ".." in PurePosixPath(path_value).parts
                or path_value == "/"
            ):
                raise ConfigError(
                    f"{source}: cluster {name!r} {field} must be a safe absolute remote path"
                )
        assert isinstance(template_root, str) and isinstance(work_root, str)
        template_path = PurePosixPath(template_root)
        work_path = PurePosixPath(work_root)
        if template_path == work_path or template_path in work_path.parents or work_path in template_path.parents:
            raise ConfigError(
                f"{source}: cluster {name!r} remote_template_root and work_root "
                "must be disjoint"
            )
        scheduler: SchedulerConfig | None = None
        if scheduler_value is not None:
            if not isinstance(scheduler_value, dict):
                raise ConfigError(
                    f"{source}: cluster {name!r} scheduler must be a mapping"
                )
            missing_scheduler = sorted(
                REQUIRED_SCHEDULER_FIELDS - set(scheduler_value)
            )
            unknown_scheduler = sorted(set(scheduler_value) - SCHEDULER_FIELDS)
            if missing_scheduler:
                raise ConfigError(
                    f"{source}: cluster {name!r} scheduler lacks "
                    + ", ".join(missing_scheduler)
                )
            if unknown_scheduler:
                raise ConfigError(
                    f"{source}: cluster {name!r} scheduler has unsupported fields: "
                    + ", ".join(unknown_scheduler)
                )
            candidates = scheduler_value.get("partition_candidates")
            if (
                not isinstance(candidates, list)
                or not candidates
                or any(
                    not isinstance(candidate, str)
                    or not PARTITION_NAME.fullmatch(candidate)
                    for candidate in candidates
                )
                or len(set(candidates)) != len(candidates)
            ):
                raise ConfigError(
                    f"{source}: cluster {name!r} scheduler.partition_candidates "
                    "must be a non-empty unique list of safe partition names"
                )
            memory_constraint = scheduler_value.get("memory_constraint", "reported")
            if memory_constraint not in MEMORY_CONSTRAINTS:
                raise ConfigError(
                    f"{source}: cluster {name!r} scheduler.memory_constraint must be "
                    "reported or unreported"
                )
            scheduler = SchedulerConfig(tuple(candidates), str(memory_constraint))
        clusters[name] = ClusterProfile(
            name=name,
            backend=backend,
            ssh_profile=ssh_profile,
            remote_template_root=str(PurePosixPath(template_root)),
            work_root=str(PurePosixPath(work_root)),
            scheduler=scheduler,
        )
    return clusters
