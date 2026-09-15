"""Small path records for produced artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def artifact(path: Path) -> dict[str, Any]:
    """Return the location of one produced file or directory."""

    return {"uri": path.resolve().as_uri()}
