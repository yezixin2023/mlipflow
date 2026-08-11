"""Provenance capture with credential redaction."""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


SENSITIVE = re.compile(r"(TOKEN|PASSWORD|PASSWD|SECRET|PRIVATE|API_KEY|CREDENTIAL)", re.I)


def capture(environment_allowlist: tuple[str, ...] = (), source_root: str | None = None) -> dict[str, Any]:
    environment: dict[str, str] = {}
    for key in environment_allowlist:
        if key in os.environ and not SENSITIVE.search(key):
            environment[key] = os.environ[key]
    provenance = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "environment": environment,
    }
    provenance["source_git_commit"] = _git_commit(source_root)
    return provenance


def _git_commit(source_root: str | None) -> str | None:
    if source_root is None:
        return None
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            shell=False,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and re.fullmatch(r"[0-9a-f]{40,64}", commit) else None


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<redacted>" if SENSITIVE.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value
