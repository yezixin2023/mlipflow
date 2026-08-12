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
SAFE_REMOTE_PATH = re.compile(r"[A-Za-z0-9_./+\-]+")
SAFE_REMOTE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+\-]*")
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
        _validate_remote_directory(remote_cwd, allow_dot=True)
        _validate_remote_name(remote_script)
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

    def stage_fresh(
        self,
        remote_root: str,
        run_name: str,
        files: Sequence[tuple[Path, str, str]],
    ) -> str:
        """Create one fresh remote directory, upload an allowlist, and verify SHA-256.

        ``files`` entries are ``(local_source, remote_basename, sha256:...)``.  A
        basename-only contract intentionally avoids recursive staging and makes
        every remote write visible in the approved run plan.
        """

        _validate_remote_directory(remote_root, allow_dot=True)
        _validate_remote_name(run_name)
        if not files:
            raise BackendError("remote staging requires at least one file")
        remote_cwd = run_name if remote_root == "." else f"{remote_root.rstrip('/')}/{run_name}"
        _validate_remote_directory(remote_cwd)
        create = (
            f"cd -- {_quote_remote(remote_root)} && "
            f"mkdir -- {_quote_remote(run_name)}"
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
                completed.stderr or completed.stdout or "fresh remote directory creation failed"
            )
        seen: set[str] = set()
        for source, remote_name, expected_sha256 in files:
            _validate_remote_name(remote_name)
            if remote_name in seen:
                raise BackendError(f"duplicate remote staging name: {remote_name}")
            seen.add(remote_name)
            if source.is_symlink() or not source.is_file():
                raise BackendError(f"staging source must be an ordinary file: {source}")
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_sha256):
                raise BackendError(f"invalid staging fingerprint for {remote_name}")
            uploaded = subprocess.run(
                ["scp", "--", str(source), f"{self.profile}:{remote_cwd}/{remote_name}"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                check=False,
            )
            if uploaded.returncode != 0:
                raise BackendError(
                    uploaded.stderr or uploaded.stdout or f"remote staging failed: {remote_name}"
                )
            observed = self.inspect_file(remote_cwd, remote_name)
            if observed.get("sha256") != expected_sha256:
                raise BackendError(f"remote staging fingerprint mismatch: {remote_name}")
        return remote_cwd

    def inspect_file(self, remote_cwd: str, remote_name: str) -> dict[str, object]:
        """Read one remote file's bounded identity without modifying remote state."""

        _validate_remote_directory(remote_cwd)
        _validate_remote_name(remote_name)
        command = (
            f"cd -- {_quote_remote(remote_cwd)} && "
            f"if [ -f {_quote_remote(remote_name)} ] && "
            f"[ ! -L {_quote_remote(remote_name)} ]; then "
            f"stat -c '%s' -- {_quote_remote(remote_name)} && "
            f"sha256sum -- {_quote_remote(remote_name)}; "
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
        if len(lines) != 2 or not lines[0].isdigit():
            raise BackendError(f"unexpected remote fingerprint response for {remote_name}")
        digest, separator, returned_name = lines[1].partition("  ")
        if not separator or returned_name != remote_name or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise BackendError(f"invalid remote fingerprint response for {remote_name}")
        return {
            "path": remote_name,
            "exists": True,
            "size_bytes": int(lines[0]),
            "sha256": f"sha256:{digest}",
        }

    def fetch_from(self, remote_cwd: str, remote_name: str, destination: Path) -> Path:
        """Fetch one approved basename into a fresh local destination."""

        _validate_remote_directory(remote_cwd)
        _validate_remote_name(remote_name)
        if destination.exists() or destination.is_symlink():
            raise BackendError(f"remote fetch destination must be fresh: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            ["scp", "--", f"{self.profile}:{remote_cwd}/{remote_name}", str(destination)],
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
            f"else sacct -n -X -j {job_id} -o State,Reason -P; fi"
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


def _validate_remote_name(value: str) -> None:
    if not isinstance(value, str) or not SAFE_REMOTE_NAME.fullmatch(value):
        raise BackendError(f"remote file name must be a safe basename: {value!r}")


def _validate_remote_directory(value: str, *, allow_dot: bool = False) -> None:
    if not isinstance(value, str) or not value or not SAFE_REMOTE_PATH.fullmatch(value):
        raise BackendError(f"remote directory uses unsafe characters: {value!r}")
    path = Path(value)
    if ".." in path.parts or (path == Path(".") and not allow_dot):
        raise BackendError(f"remote directory is not allowed: {value!r}")
