"""Execution backends.

Only command services instantiate these classes. Query services never import or
call them, which is part of the read-side-effect boundary.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol, Sequence

from .errors import BackendError


JOB_ID = re.compile(r"Submitted batch job\s+(\d+)")
SAFE_REMOTE_PATH = re.compile(r"[A-Za-z0-9_./+\-]+")
SAFE_REMOTE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+\-]*")
SAFE_REMOTE_RELATIVE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./+\-]*")
SAFE_PARTITION_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+\-]*")
_SCONTROL_NODE_MARKER = "__MLIPFLOW_SCONTROL_NODES__"
SAFE_ENVIRONMENT_KEYS = frozenset(
    {
        "HOME",
        "LANG",
        "LANGUAGE",
        "PATH",
        "PYTHONHOME",
        "PYTHONPATH",
        "TMP",
        "TEMP",
        "TMPDIR",
        "TZ",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
    }
)
SAFE_ENVIRONMENT_PREFIXES = ("LC_", "SLURM_", "CUDA_", "ROCR_", "OMP_", "MKL_")


@dataclass(frozen=True)
class ExecutionResult:
    returncode: int
    stdout: str
    stderr: str
    job_id: str | None = None
    submission_provenance: dict[str, Any] | None = None


class SchedulerBackend(Protocol):
    """The asynchronous half of the execution contract.

    ``SshSlurmBackend`` satisfies this structurally. What every scheduler must
    share is a persistable job id, a cancellation path and a state query, because
    those three are what the lifecycle state machine drives.

    ``LocalBackend`` deliberately does *not* implement this: it is synchronous and
    has no job id, so it belongs to a different lifecycle shape.
    """

    name: str

    def submit(self, script: Any, cwd: Any) -> ExecutionResult: ...

    def cancel(self, job_id: str) -> ExecutionResult: ...

    def status(self, job_id: str) -> dict[str, str | None]: ...


class LocalBackend:
    name = "local"

    def run(
        self,
        argv: Sequence[str],
        cwd: Path,
        environment: Mapping[str, str] | None = None,
    ) -> ExecutionResult:
        validate_argv(argv)
        merged = sanitized_environment(os.environ)
        merged.update(validate_environment(environment or {}))
        completed = subprocess.run(
            list(argv),
            cwd=str(cwd),
            env=merged,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
        )
        return ExecutionResult(completed.returncode, completed.stdout, completed.stderr)

    def status(self, _: str | None = None) -> dict[str, str]:
        return {"state": "synchronous", "detail": "local runs return only after completion"}

    def fetch(self, source: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination

    def logs(self, path: Path, tail: int = 80) -> list[str]:
        if tail < 1:
            raise BackendError("tail must be positive")
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            return stream.readlines()[-tail:]


class SshSlurmBackend:
    """SSH+SLURM backend using an SSH config alias, never an embedded key path."""

    name = "ssh-slurm"

    def __init__(self, profile: str):
        if not re.fullmatch(r"[A-Za-z0-9._-]+", profile):
            raise BackendError("SSH profile must be a simple ~/.ssh/config alias")
        self.profile = profile

    def submit(
        self,
        remote_script: str,
        remote_run_dir: str,
        *,
        partition_candidates: Sequence[str] | None = None,
        resources: Mapping[str, Any] | None = None,
        memory_constraint: str = "reported",
    ) -> ExecutionResult:
        _validate_remote_directory(remote_run_dir, allow_dot=True)
        _validate_remote_name(remote_script)
        routing: dict[str, Any] | None = None
        selected_partition: str | None = None
        if partition_candidates is not None:
            if resources is None:
                raise BackendError("partition selection requires requested resources")
            routing = self.select_partition(
                partition_candidates,
                resources,
                memory_constraint=memory_constraint,
            )
            selected_partition = str(routing["selected_partition"])
        # Arguments are passed as separate argv fields. The remote helper should
        # eventually replace OpenSSH command composition with a fixed RPC.
        partition_argument = (
            f" --partition={_quote_remote(selected_partition)}"
            if selected_partition is not None
            else ""
        )
        command = (
            f"cd -- {_quote_remote(remote_run_dir)} && "
            f"sbatch{partition_argument} -- {_quote_remote(remote_script)}"
        )
        completed = subprocess.run(
            ["ssh", "--", self.profile, command],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
        )
        match = JOB_ID.search(completed.stdout)
        return ExecutionResult(
            completed.returncode,
            completed.stdout,
            completed.stderr,
            match.group(1) if match else None,
            routing,
        )

    def select_partition(
        self,
        candidates: Sequence[str],
        resources: Mapping[str, Any],
        *,
        memory_constraint: str = "reported",
    ) -> dict[str, Any]:
        """Select a site-owned candidate from one submission-time Slurm snapshot.

        No node is selected or reserved here.  A currently idle capable node wins
        by candidate order; when every healthy capable partition is busy, the
        preferred healthy candidate is returned so Slurm can queue normally.
        """

        validated = _validate_partition_candidates(candidates)
        if memory_constraint not in {"reported", "unreported"}:
            raise BackendError("memory_constraint must be reported or unreported")
        requested = _requested_node_resources(resources)
        command = (
            "scontrol show partition -o && "
            f"printf '\\n{_SCONTROL_NODE_MARKER}\\n' && "
            "scontrol show node -o"
        )
        completed = subprocess.run(
            ["ssh", "--", self.profile, command],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
        )
        if completed.returncode != 0:
            raise BackendError(
                completed.stderr
                or completed.stdout
                or "remote Slurm partition snapshot failed"
            )
        before, separator, after = completed.stdout.partition(
            f"{_SCONTROL_NODE_MARKER}\n"
        )
        if not separator:
            raise BackendError("remote Slurm partition snapshot is malformed")
        partitions = {
            record["PartitionName"]: record
            for record in _parse_scontrol_records(before, "PartitionName")
        }
        nodes = _parse_scontrol_records(after, "NodeName")
        observations = [
            _partition_observation(
                candidate,
                partitions.get(candidate),
                nodes,
                requested,
                enforce_memory=memory_constraint == "reported",
            )
            for candidate in validated
        ]
        available = [item for item in observations if item["currently_available"]]
        schedulable = [item for item in observations if item["schedulable"]]
        if available:
            selected = str(available[0]["partition"])
            selection_mode = "available-now"
        elif schedulable:
            selected = str(schedulable[0]["partition"])
            selection_mode = "healthy-queue"
        else:
            reasons = "; ".join(
                f"{item['partition']}: {item['reason']}" for item in observations
            )
            raise BackendError(
                "no candidate Slurm partition is currently schedulable for the "
                f"requested per-node resources ({reasons})"
            )
        return {
            "candidate_partitions": list(validated),
            "observed_partition_availability": observations,
            "selected_partition": selected,
            "selection_mode": selection_mode,
            "snapshot_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "snapshot_scope": (
                "submission-time observation only; no node or resource was reserved"
            ),
        }

    def read_template(
        self, template_root: str, relative_path: str, max_bytes: int = 1024 * 1024
    ) -> dict[str, object]:
        """Read one bounded ordinary template from the persistent site library."""

        _validate_absolute_remote_directory(template_root)
        _validate_remote_relative(relative_path)
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise BackendError("template size bound must be a positive integer")
        command = (
            f"if [ ! -d {_quote_remote(template_root)} ] || "
            f"[ -L {_quote_remote(template_root)} ]; then printf 'ROOT_MISSING\\n'; "
            f"else cd -- {_quote_remote(template_root)} && "
            f"if [ -f {_quote_remote(relative_path)} ] && "
            f"[ ! -L {_quote_remote(relative_path)} ]; then "
            f"size=$(stat -c '%s' -- {_quote_remote(relative_path)}); "
            f"if [ \"$size\" -le {max_bytes} ]; then "
            "printf 'OK\\n'; "
            f"cat -- {_quote_remote(relative_path)}; "
            "else printf 'TOO_LARGE\\n'; fi; "
            "else printf 'MISSING\\n'; fi; fi"
        )
        completed = subprocess.run(
            ["ssh", "--", self.profile, command],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
        )
        if completed.returncode != 0:
            raise BackendError(
                completed.stderr or completed.stdout or "remote template read failed"
            )
        if completed.stdout in {"ROOT_MISSING\n", "ROOT_MISSING"}:
            return {
                "relative_path": relative_path,
                "root_exists": False,
                "exists": False,
            }
        if completed.stdout in {"MISSING\n", "MISSING"}:
            return {
                "relative_path": relative_path,
                "root_exists": True,
                "exists": False,
            }
        if completed.stdout in {"TOO_LARGE\n", "TOO_LARGE"}:
            raise BackendError(f"remote template exceeds size bound: {relative_path}")
        first, separator, content = completed.stdout.partition("\n")
        if not separator or first != "OK":
            raise BackendError(f"unexpected remote template response: {relative_path}")
        return {
            "relative_path": relative_path,
            "root_exists": True,
            "exists": True,
            "content": content,
        }

    def stage_workspace(
        self,
        remote_run_dir: str,
        files: Sequence[tuple[Path, str]],
    ) -> str:
        """Create one deterministic attempt workspace and stage an allowlist.

        The attempt directory itself is the no-overwrite boundary: mkdir must
        fail when that attempt already exists. Inside a fresh attempt the
        backend owns the common workspace skeleton and pre-creates empty
        input/, output/ and logs/ directories. Scheduled runners therefore
        receive an existing but empty output root and own only its contents.
        """

        _validate_absolute_remote_directory(remote_run_dir)
        run_path = PurePosixPath(remote_run_dir)
        attempt_workspace = bool(
            re.fullmatch(r"attempt-[0-9]{4,}", run_path.name)
        )
        independent_workspace = bool(
            SAFE_REMOTE_NAME.fullmatch(run_path.name)
            and run_path.parent.name == "submissions"
            and re.fullmatch(r"attempt-[0-9]{4,}", run_path.parent.parent.name)
        )
        if not attempt_workspace and not independent_workspace:
            raise BackendError(
                "remote run directory must be attempt-XXXX or its submissions/<id> child"
            )
        if not files:
            raise BackendError("remote workspace staging requires at least one file")
        parent = str(run_path.parent)
        # A node may now stage several calculations, each in its own
        # subdirectory.  Every parent directory is derived from the validated
        # relative paths and created up front, in sorted order, so the adapter
        # never has to issue its own ssh/mkdir and staging stays one deterministic
        # sequence of operations.
        nested: set[str] = set()
        for _, remote_relative in files:
            _validate_remote_relative(remote_relative)
            for ancestor in PurePosixPath(remote_relative).parents:
                if str(ancestor) not in {".", "/"}:
                    nested.add(str(run_path / ancestor))
        directories = [
            str(run_path / "input"),
            str(run_path / "output"),
            str(run_path / "logs"),
        ]
        directories.extend(sorted(nested - set(directories)))
        create = (
            f"mkdir -p -- {_quote_remote(parent)} && "
            f"mkdir -- {_quote_remote(remote_run_dir)} && "
            "mkdir -p -- " + " ".join(_quote_remote(item) for item in directories)
        )
        completed = subprocess.run(
            ["ssh", "--", self.profile, create],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
        )
        if completed.returncode != 0:
            raise BackendError(
                completed.stderr or completed.stdout or "fresh remote workspace creation failed"
            )
        seen: set[str] = set()
        for source, remote_relative in files:
            _validate_remote_relative(remote_relative)
            if remote_relative in seen:
                raise BackendError(f"duplicate remote staging path: {remote_relative}")
            seen.add(remote_relative)
            if source.is_symlink() or not source.is_file():
                raise BackendError(f"staging source must be an ordinary file: {source}")
            uploaded = subprocess.run(
                ["scp", "--", str(source), f"{self.profile}:{remote_run_dir}/{remote_relative}"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                check=False,
            )
            if uploaded.returncode != 0:
                raise BackendError(
                    uploaded.stderr
                    or uploaded.stdout
                    or f"remote staging failed: {remote_relative}"
                )
        return remote_run_dir

    def inspect_file(self, remote_run_dir: str, remote_name: str) -> dict[str, object]:
        """Read whether a remote file exists and its size for transfer bounds."""

        _validate_remote_directory(remote_run_dir)
        _validate_remote_relative(remote_name)
        command = (
            f"cd -- {_quote_remote(remote_run_dir)} && "
            f"if [ -f {_quote_remote(remote_name)} ] && "
            f"[ ! -L {_quote_remote(remote_name)} ]; then "
            f"stat -c '%s' -- {_quote_remote(remote_name)}; "
            "else printf 'MISSING\\n'; fi"
        )
        completed = subprocess.run(
            ["ssh", "--", self.profile, command],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
        )
        if completed.returncode != 0:
            raise BackendError(
                completed.stderr or completed.stdout or "remote file inspection failed"
            )
        lines = completed.stdout.strip().splitlines()
        if lines == ["MISSING"] or not lines:
            return {"path": remote_name, "exists": False}
        if len(lines) != 1 or not lines[0].isdigit():
            raise BackendError(f"unexpected remote file response for {remote_name}")
        return {
            "path": remote_name,
            "exists": True,
            "size_bytes": int(lines[0]),
        }

    def fetch_from(
        self, remote_run_dir: str, remote_name: str, destination: Path
    ) -> Path:
        """Fetch one approved basename into a fresh local destination."""

        _validate_remote_directory(remote_run_dir)
        _validate_remote_relative(remote_name)
        if destination.exists() or destination.is_symlink():
            raise BackendError(f"remote fetch destination must be fresh: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [
                "scp",
                "--",
                f"{self.profile}:{remote_run_dir}/{remote_name}",
                str(destination),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
        )
        if completed.returncode != 0:
            raise BackendError(completed.stderr or completed.stdout or "scp fetch failed")
        if destination.is_symlink() or not destination.is_file():
            raise BackendError(f"fetched output is not an ordinary file: {destination}")
        return destination

    def cancel(self, job_id: str) -> ExecutionResult:
        if not job_id.isdigit():
            raise BackendError(f"invalid SLURM job id: {job_id!r}")
        completed = subprocess.run(
            ["ssh", "--", self.profile, "scancel", "--", job_id],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
        )
        return ExecutionResult(completed.returncode, completed.stdout, completed.stderr, job_id)

    def status(self, job_id: str) -> dict[str, str | None]:
        if not job_id.isdigit():
            raise BackendError(f"invalid SLURM job id: {job_id!r}")
        command = (
            f"active=$(squeue -h -j {job_id} -o '%T|%R' 2>/dev/null || true); "
            "if [ -n \"$active\" ]; then printf '%s\\n' \"$active\"; "
            f"else history=$(sacct -n -X -j {job_id} -o State,Reason -P "
            "2>/dev/null || true); "
            "if [ -n \"$history\" ]; then printf '%s\\n' \"$history\"; "
            f"else control=$(scontrol show job -o {job_id} 2>/dev/null || true); "
            "state=''; reason=''; "
            "for field in $control; do case \"$field\" in "
            "JobState=*) state=${field#JobState=} ;; "
            "Reason=*) reason=${field#Reason=} ;; esac; done; "
            "if [ -n \"$state\" ]; then printf '%s|%s\\n' \"$state\" \"$reason\"; fi; "
            "fi; fi"
        )
        completed = subprocess.run(
            ["ssh", "--", self.profile, command],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
        )
        if completed.returncode != 0:
            raise BackendError(completed.stderr or completed.stdout or "remote scheduler query failed")
        entries = [item for item in completed.stdout.strip().splitlines() if item]
        if not entries:
            return {"state": "UNKNOWN", "detail": None, "source": "remote"}
        state, _, detail = entries[-1].partition("|")
        return {"state": state.split("+")[0].upper(), "detail": detail or None, "source": "remote"}

    def fetch(self, remote_path: str, destination: Path) -> Path:
        if (
            not remote_path.startswith("/")
            or not re.fullmatch(r"/[A-Za-z0-9_./+\-]+", remote_path)
            or ".." in Path(remote_path).parts
        ):
            raise BackendError("remote fetch path must be an absolute path using safe characters")
        destination.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            ["scp", "--", f"{self.profile}:{remote_path}", str(destination)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
        )
        if completed.returncode != 0:
            raise BackendError(completed.stderr or completed.stdout or "scp fetch failed")
        return destination

    def logs(self, remote_path: str, tail: int = 80) -> list[str]:
        if tail < 1 or tail > 10000:
            raise BackendError("tail must be between 1 and 10000")
        command = f"tail -n {tail} -- {_quote_remote(remote_path)}"
        completed = subprocess.run(
            ["ssh", "--", self.profile, command],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
        )
        if completed.returncode != 0:
            raise BackendError(completed.stderr or completed.stdout or "remote log read failed")
        return completed.stdout.splitlines(keepends=True)


def _validate_partition_candidates(candidates: Sequence[str]) -> tuple[str, ...]:
    if isinstance(candidates, (str, bytes)):
        raise BackendError("partition candidates must be a non-empty sequence")
    values = tuple(candidates)
    if (
        not values
        or any(
            not isinstance(candidate, str)
            or not SAFE_PARTITION_NAME.fullmatch(candidate)
            for candidate in values
        )
        or len(set(values)) != len(values)
    ):
        raise BackendError(
            "partition candidates must be a non-empty unique sequence of safe names"
        )
    return values


def _requested_node_resources(resources: Mapping[str, Any]) -> dict[str, int]:
    cpus = resources.get("cpus")
    gpus = resources.get("gpus")
    memory = resources.get("memory")
    if (
        isinstance(cpus, bool)
        or not isinstance(cpus, int)
        or cpus < 1
        or isinstance(gpus, bool)
        or not isinstance(gpus, int)
        or gpus < 0
        or not isinstance(memory, str)
    ):
        raise BackendError("partition selection received invalid requested resources")
    return {"cpus": cpus, "gpus": gpus, "memory_mib": _memory_to_mib(memory)}


def _memory_to_mib(value: str) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)([KMGTP]?)(?:i?B)?", value)
    if match is None:
        raise BackendError(f"unsupported Slurm memory request: {value!r}")
    amount = int(match.group(1))
    unit = match.group(2)
    if unit == "K":
        return max(1, (amount + 1023) // 1024)
    factors = {"": 1, "M": 1, "G": 1024, "T": 1024**2, "P": 1024**3}
    return amount * factors[unit]


def _parse_scontrol_records(text: str, identity_field: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    field_pattern = re.compile(r"(?:^|\s)([A-Za-z][A-Za-z0-9_/]*)=(\S*)")
    for line in text.splitlines():
        fields = {match.group(1): match.group(2) for match in field_pattern.finditer(line)}
        if fields.get(identity_field):
            records.append(fields)
    return records


def _integer_field(record: Mapping[str, str], name: str) -> int:
    value = record.get(name, "")
    return int(value) if value.isdigit() else 0


def _tres_quantity(value: str, resource: str) -> int:
    parsed: dict[str, int] = {}
    for item in value.split(","):
        key, separator, raw = item.partition("=")
        if separator and raw.isdigit():
            parsed[key] = int(raw)
    if resource in parsed:
        return parsed[resource]
    typed = [amount for key, amount in parsed.items() if key.startswith(resource + ":")]
    return sum(typed)


def _configured_gpus(record: Mapping[str, str]) -> int:
    configured = _tres_quantity(record.get("CfgTRES", ""), "gres/gpu")
    if configured:
        return configured
    gres = record.get("Gres", "")
    total = 0
    for item in gres.split(","):
        match = re.match(r"gpu(?::[^:,()]+)?:(\d+)(?:\(|$)", item)
        if match is not None:
            total += int(match.group(1))
    return total


def _node_is_healthy(state: str) -> bool:
    normalized = state.upper().replace("-", "_")
    unhealthy = (
        "DOWN",
        "DRAIN",
        "FAIL",
        "MAINT",
        "POWER_DOWN",
        "REBOOT",
        "FUTURE",
        "UNKNOWN",
        "INVALID_REG",
        "NO_RESPOND",
        "RESERVED",
    )
    return bool(normalized) and not any(marker in normalized for marker in unhealthy)


def _partition_observation(
    candidate: str,
    partition: Mapping[str, str] | None,
    nodes: Sequence[Mapping[str, str]],
    requested: Mapping[str, int],
    *,
    enforce_memory: bool = True,
) -> dict[str, Any]:
    if partition is None:
        return {
            "partition": candidate,
            "exists": False,
            "state": None,
            "schedulable": False,
            "currently_available": False,
            "observed_node_availability": {
                "total": 0,
                "healthy": 0,
                "capable_for_request": 0,
                "available_now": 0,
                "busy_capable": 0,
                "states": {},
            },
            "reason": "partition does not exist",
        }

    state = partition.get("State", "UNKNOWN").upper()
    members = [
        node
        for node in nodes
        if candidate in node.get("Partitions", "").split(",")
    ]
    states: dict[str, int] = {}
    healthy = 0
    capable = 0
    available = 0
    max_healthy = {"cpus": 0, "gpus": 0, "memory_mib": 0}
    for node in members:
        node_state = node.get("State", "UNKNOWN").upper()
        states[node_state] = states.get(node_state, 0) + 1
        if not _node_is_healthy(node_state):
            continue
        healthy += 1
        total_cpus = _integer_field(node, "CPUTot") or _tres_quantity(
            node.get("CfgTRES", ""), "cpu"
        )
        allocated_cpus = _integer_field(node, "CPUAlloc") or _tres_quantity(
            node.get("AllocTRES", ""), "cpu"
        )
        total_gpus = _configured_gpus(node)
        allocated_gpus = _tres_quantity(node.get("AllocTRES", ""), "gres/gpu")
        total_memory = _integer_field(node, "RealMemory")
        allocated_memory = _integer_field(node, "AllocMem")
        max_healthy = {
            "cpus": max(max_healthy["cpus"], total_cpus),
            "gpus": max(max_healthy["gpus"], total_gpus),
            "memory_mib": max(max_healthy["memory_mib"], total_memory),
        }
        node_capable = (
            total_cpus >= requested["cpus"]
            and total_gpus >= requested["gpus"]
            and (not enforce_memory or total_memory >= requested["memory_mib"])
        )
        if not node_capable:
            continue
        capable += 1
        if (
            total_cpus - allocated_cpus >= requested["cpus"]
            and total_gpus - allocated_gpus >= requested["gpus"]
            and (
                not enforce_memory
                or total_memory - allocated_memory >= requested["memory_mib"]
            )
        ):
            available += 1

    partition_healthy = state not in {"DOWN", "DRAIN", "INACTIVE"} and state == "UP"
    schedulable = partition_healthy and capable > 0
    currently_available = schedulable and available > 0
    if not partition_healthy:
        reason = f"partition state is {state}"
    elif healthy == 0:
        reason = "partition has no healthy observed nodes"
    elif capable == 0:
        reason = "no healthy node satisfies the requested per-node resources"
    elif available == 0:
        reason = "healthy capable nodes are currently busy; Slurm queueing is allowed"
    else:
        reason = "a healthy capable node has the requested resources available now"
    return {
        "partition": candidate,
        "exists": True,
        "state": state,
        "schedulable": schedulable,
        "currently_available": currently_available,
        "observed_node_availability": {
            "total": len(members),
            "healthy": healthy,
            "capable_for_request": capable,
            "available_now": available,
            "busy_capable": capable - available,
            "states": {key: states[key] for key in sorted(states)},
            "max_healthy_node_capacity": max_healthy,
        },
        "reason": reason,
    }


def validate_argv(argv: Sequence[str]) -> None:
    if not argv or not all(isinstance(value, str) and value for value in argv):
        raise BackendError("external command must be a non-empty argv string list")
    if any("\x00" in value for value in argv):
        raise BackendError("NUL is not allowed in argv")
    executable = Path(argv[0]).name
    if executable in {"ssh", "scp", "sftp", "sbatch", "scancel"}:
        raise BackendError(
            f"{executable} must be invoked through the configured transport/scheduler backend"
        )
    sensitive_option = re.compile(
        r"^--?(?:password|passwd|token|secret|api[-_]?key|identityfile|private[-_]?key)(?:=|$)",
        re.I,
    )
    if any(sensitive_option.search(value) for value in argv):
        raise BackendError("credentials must not be passed in workflow argv")
    if any(
        re.search(r"\bBearer\s+\S+", value, re.I)
        or re.search(r"://[^/\s:@]+:[^/@\s]+@", value)
        for value in argv
    ):
        raise BackendError("credential-like values must not be passed in workflow argv")


def sanitized_environment(source: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in source.items()
        if key in SAFE_ENVIRONMENT_KEYS or key.startswith(SAFE_ENVIRONMENT_PREFIXES)
    }


def validate_environment(environment: Mapping[str, str]) -> dict[str, str]:
    sensitive = re.compile(
        r"password|passwd|token|secret|credential|api[_-]?key|private[_-]?key|identity",
        re.I,
    )
    result: dict[str, str] = {}
    for key, value in environment.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise BackendError("execution environment must be a string mapping")
        if not key or "\x00" in key or "=" in key or "\x00" in value:
            raise BackendError("invalid execution environment entry")
        if sensitive.search(key):
            raise BackendError(f"sensitive environment variable is forbidden: {key}")
        result[key] = value
    return result


def _quote_remote(value: str) -> str:
    if "\x00" in value or "\n" in value:
        raise BackendError("invalid remote path")
    return "'" + value.replace("'", "'\\''") + "'"


def _validate_remote_name(value: str) -> None:
    if not isinstance(value, str) or not SAFE_REMOTE_NAME.fullmatch(value):
        raise BackendError(f"remote file name must be a safe basename: {value!r}")


def _validate_remote_directory(value: str, *, allow_dot: bool = False) -> None:
    if not isinstance(value, str) or not value or not SAFE_REMOTE_PATH.fullmatch(value):
        raise BackendError(f"remote directory uses unsafe characters: {value!r}")
    path = Path(value)
    if ".." in path.parts or (path == Path(".") and not allow_dot):
        raise BackendError(f"remote directory is not allowed: {value!r}")


def _validate_absolute_remote_directory(value: str) -> None:
    _validate_remote_directory(value)
    if not PurePosixPath(value).is_absolute() or value == "/":
        raise BackendError(f"remote directory must be a non-root absolute path: {value!r}")


def _validate_remote_relative(value: str) -> None:
    if not isinstance(value, str) or not SAFE_REMOTE_RELATIVE.fullmatch(value):
        raise BackendError(f"remote path must be a safe relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {".", ""}:
        raise BackendError(f"remote path is not allowed: {value!r}")
