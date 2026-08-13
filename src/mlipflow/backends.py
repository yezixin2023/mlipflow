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
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol, Sequence

from .errors import BackendError


JOB_ID = re.compile(r"Submitted batch job\s+(\d+)")
SAFE_REMOTE_PATH = re.compile(r"[A-Za-z0-9_./+\-]+")
SAFE_REMOTE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+\-]*")
SAFE_REMOTE_RELATIVE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./+\-]*")
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


class SchedulerBackend(Protocol):
    """The asynchronous half of the execution contract.

    ``SlurmBackend`` and ``SshSlurmBackend`` satisfy this structurally.  The
    submit signature intentionally stays loose: a local scheduler takes a script
    path plus a working directory, a remote one takes two remote strings.  What
    every scheduler must share is a persistable job id, a cancellation path and a
    state query, because those three are what the lifecycle state machine drives.

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

    def submit(self, remote_script: str, remote_run_dir: str) -> ExecutionResult:
        _validate_remote_directory(remote_run_dir, allow_dot=True)
        _validate_remote_name(remote_script)
        # Arguments are passed as separate argv fields. The remote helper should
        # eventually replace OpenSSH command composition with a fixed RPC.
        command = (
            f"cd -- {_quote_remote(remote_run_dir)} && "
            f"sbatch -- {_quote_remote(remote_script)}"
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
        )

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
            "printf '%s\\n' \"$size\"; "
            f"sha256sum -- {_quote_remote(relative_path)}; "
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
        first, separator, remainder = completed.stdout.partition("\n")
        second, separator2, content = remainder.partition("\n")
        if not separator or not separator2 or not first.isdigit():
            raise BackendError(f"unexpected remote template response: {relative_path}")
        digest, digest_separator, returned_name = second.partition("  ")
        if (
            not digest_separator
            or returned_name != relative_path
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or len(content.encode("utf-8")) != int(first)
        ):
            raise BackendError(f"invalid remote template identity: {relative_path}")
        return {
            "relative_path": relative_path,
            "root_exists": True,
            "exists": True,
            "size_bytes": int(first),
            "sha256": f"sha256:{digest}",
            "content": content,
        }

    def stage_workspace(
        self,
        remote_run_dir: str,
        files: Sequence[tuple[Path, str, str]],
    ) -> str:
        """Create one deterministic attempt workspace and stage an allowlist."""

        _validate_absolute_remote_directory(remote_run_dir)
        run_path = PurePosixPath(remote_run_dir)
        if not re.fullmatch(r"attempt-[0-9]{4,}", run_path.name):
            raise BackendError("remote run directory must end in attempt-XXXX")
        if not files:
            raise BackendError("remote workspace staging requires at least one file")
        parent = str(run_path.parent)
        # A node may now stage several calculations, each in its own
        # subdirectory.  Every parent directory is derived from the validated
        # relative paths and created up front, in sorted order, so the adapter
        # never has to issue its own ssh/mkdir and staging stays one deterministic
        # sequence of operations.
        nested: set[str] = set()
        for _, remote_relative, _ in files:
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
        for source, remote_relative, expected_sha256 in files:
            _validate_remote_relative(remote_relative)
            if remote_relative in seen:
                raise BackendError(f"duplicate remote staging path: {remote_relative}")
            seen.add(remote_relative)
            if source.is_symlink() or not source.is_file():
                raise BackendError(f"staging source must be an ordinary file: {source}")
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_sha256):
                raise BackendError(f"invalid staging fingerprint for {remote_relative}")
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
            observed = self.inspect_file(remote_run_dir, remote_relative)
            if observed.get("sha256") != expected_sha256:
                raise BackendError(
                    f"remote staging fingerprint mismatch: {remote_relative}"
                )
        return remote_run_dir

    def inspect_file(self, remote_run_dir: str, remote_name: str) -> dict[str, object]:
        """Read one remote file's bounded identity without modifying remote state."""

        _validate_remote_directory(remote_run_dir)
        _validate_remote_relative(remote_name)
        command = (
            f"cd -- {_quote_remote(remote_run_dir)} && "
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
