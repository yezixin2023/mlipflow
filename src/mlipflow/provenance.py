"""Provenance capture with credential redaction."""

from __future__ import annotations

import os
import platform
import re
import sys
from datetime import datetime, timezone
from typing import Any


SENSITIVE = re.compile(r"(TOKEN|PASSWORD|PASSWD|SECRET|PRIVATE|API_KEY|CREDENTIAL)", re.I)


def capture(environment_allowlist: tuple[str, ...] = ()) -> dict[str, Any]:
    environment: dict[str, str] = {}
    for key in environment_allowlist:
        if key in os.environ and not SENSITIVE.search(key):
            environment[key] = os.environ[key]
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "environment": environment,
    }


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<redacted>" if SENSITIVE.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value
