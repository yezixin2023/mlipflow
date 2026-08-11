"""Small deterministic IO helpers.

Configuration files may be JSON (which is valid YAML) so that minimal Python
environments can inspect projects without importing a YAML package.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigError


def load_mapping(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ConfigError(
                f"{path} is YAML rather than JSON; install PyYAML to read it"
            ) from exc
        try:
            value = yaml.safe_load(text)
        except Exception as exc:  # PyYAML exposes several parser exception types
            raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Write JSON using replace(2); mutation services only."""

    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    write_text_atomic(path, payload)


def write_text_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
