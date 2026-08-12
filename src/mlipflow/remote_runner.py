"""Self-contained runner staged by MLIPFlow for one approved remote argv.

This file intentionally imports only the Python standard library so a remote
compute node does not need an MLIPFlow installation.  The scheduler script
executes this file directly; user argv remains JSON data and is never rendered
into a shell program.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence


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


def _validate_argv(argv: Sequence[str]) -> None:
    if not argv or not all(isinstance(value, str) and value for value in argv):
        raise ValueError("external command must be a non-empty argv string list")
    if any("\x00" in value for value in argv):
        raise ValueError("NUL is not allowed in argv")
    executable = Path(argv[0]).name
    if executable in {"ssh", "scp", "sftp", "sbatch", "scancel"}:
        raise ValueError(f"nested transport/scheduler command is forbidden: {executable}")
    sensitive = re.compile(
        r"^--?(?:password|passwd|token|secret|api[-_]?key|identityfile|private[-_]?key)(?:=|$)",
        re.I,
    )
    if any(sensitive.search(value) for value in argv):
        raise ValueError("credentials must not be passed in workflow argv")


def _environment(source: Mapping[str, str], overrides: object) -> dict[str, str]:
    if not isinstance(overrides, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in overrides.items()
    ):
        raise ValueError("environment must be a string mapping")
    sensitive = re.compile(
        r"password|passwd|token|secret|credential|api[_-]?key|private[_-]?key|identity",
        re.I,
    )
    if any(
        not key
        or "\x00" in key
        or "=" in key
        or "\x00" in value
        or sensitive.search(key)
        for key, value in overrides.items()
    ):
        raise ValueError("invalid or sensitive environment override")
    result = {
        key: value
        for key, value in source.items()
        if key in SAFE_ENVIRONMENT_KEYS or key.startswith(SAFE_ENVIRONMENT_PREFIXES)
    }
    result.update(overrides)
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        print("usage: remote_runner.py COMMAND.json", file=sys.stderr)
        return 2
    try:
        command_file = Path(arguments[0])
        value = json.loads(command_file.read_text(encoding="utf-8"))
        command = value["argv"]
        cwd = Path(value.get("cwd", "."))
        if not isinstance(command, list):
            raise ValueError("argv must be a list")
        _validate_argv(command)
        if not cwd.is_dir() or cwd.is_symlink():
            raise ValueError("cwd must be an existing ordinary directory")
        environment = _environment(os.environ, value.get("environment", {}))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"invalid MLIPFlow remote command file: {exc}", file=sys.stderr)
        return 2
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=environment,
        shell=False,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
