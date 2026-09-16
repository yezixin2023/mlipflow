"""Execute one approved argv file on a scheduler node without a user shell command."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .backends import sanitized_environment, validate_argv, validate_environment
from .errors import BackendError


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        print("usage: python -m mlipipe.runner COMMAND.json", file=sys.stderr)
        return 2
    path = Path(arguments[0])
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        command = value["argv"]
        cwd = Path(value["cwd"])
        environment = value.get("environment", {})
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise ValueError("argv must be a string list")
        if not isinstance(environment, dict):
            raise ValueError("environment must be a string mapping")
        environment = validate_environment(environment)
        validate_argv(command)
        if not cwd.is_dir():
            raise ValueError(f"working directory does not exist: {cwd}")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, BackendError) as exc:
        print(f"invalid MLIPipe command file: {exc}", file=sys.stderr)
        return 2
    process_environment = sanitized_environment(os.environ)
    process_environment.update(environment)
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=process_environment,
        shell=False,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
