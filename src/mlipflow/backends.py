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
from pathlib import Path
from typing import Mapping, Sequence

from .errors import BackendError


JOB_ID = re.compile(r"Submitted batch job\s+(\d+)")
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


class SlurmBackend:
    name = "slurm"

    def submit(self, script: Path, cwd: Path) -> ExecutionResult:
        if not script.is_file():
            raise BackendError(f"SLURM script not found: {script}")
        completed = subprocess.run(
            ["sbatch", str(script)],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
        )
        match = JOB_ID.search(completed.stdout)
        if completed.returncode == 0 and match is None:
            raise BackendError(f"sbatch returned no job id: {completed.stdout.strip()}")
        return ExecutionResult(
            completed.returncode,
            completed.stdout,
            completed.stderr,
            match.group(1) if match else None,
        )

    def cancel(self, job_id: str) -> ExecutionResult:
        if not job_id.isdigit():
            raise BackendError(f"invalid SLURM job id: {job_id!r}")
        completed = subprocess.run(
            ["scancel", job_id],
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
        active = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T|%R"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
        )
        if active.returncode != 0:
            raise BackendError(active.stderr or active.stdout or "squeue failed")
        line = active.stdout.strip().splitlines()
        if line:
            state, _, detail = line[0].partition("|")
            return {"state": state.upper(), "detail": detail or None, "source": "squeue"}
        history = subprocess.run(
            ["sacct", "-n", "-X", "-j", job_id, "-o", "State,Reason", "-P"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
        )
        if history.returncode != 0:
            raise BackendError(history.stderr or history.stdout or "sacct failed")
        entries = [item for item in history.stdout.strip().splitlines() if item]
        if not entries:
            return {"state": "UNKNOWN", "detail": None, "source": "sacct"}
        state, _, detail = entries[0].partition("|")
        return {"state": state.split("+")[0].upper(), "detail": detail or None, "source": "sacct"}

    def fetch(self, source: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination

    def logs(self, path: Path, tail: int = 80) -> list[str]:
        return LocalBackend().logs(path, tail)


class SshSlurmBackend:
    """SSH+SLURM backend using an SSH config alias, never an embedded key path."""

    name = "ssh-slurm"

    def __init__(self, profile: str):
        if not re.fullmatch(r"[A-Za-z0-9._-]+", profile):
            raise BackendError("SSH profile must be a simple ~/.ssh/config alias")
        self.profile = profile

    def submit(self, remote_script: str, remote_cwd: str) -> ExecutionResult:
        # Arguments are passed as separate argv fields. The remote helper should
        # eventually replace OpenSSH command composition with a fixed RPC.
        command = f"cd -- {_quote_remote(remote_cwd)} && sbatch -- {_quote_remote(remote_script)}"
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
        )

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
            f"squeue -h -j {job_id} -o '%T|%R' || exit $?; "
            f"if ! squeue -h -j {job_id} | grep -q .; then "
            f"sacct -n -X -j {job_id} -o State,Reason -P; fi"
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
