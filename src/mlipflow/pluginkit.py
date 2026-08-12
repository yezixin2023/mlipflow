"""Stable helper API for plugin adapters.

Every bundled adapter had been re-deriving the same primitives: ``_diagnostic``
appears in all eight, ``_sha256`` / ``_read_json`` / ``_mapping`` in five,
``_safe_relative`` / ``_path`` / ``_plain_string`` / ``_operation`` in four.  Two
of those — ``_safe_relative`` and the path-containment check — are security
boundaries, so having four independently drifting copies means a fix to one is
not a fix to the others.

This module is that shared implementation, taking the strictest variant of each.
It is a public, versioned surface: adapters may import it, and its behaviour must
not change silently.

**It imports nothing but the standard library.**  That is deliberate and
enforced by ``tests/test_module_boundaries.py``: an adapter importing this must
not thereby pull execution backends, the state store or the service layer into
the plugin's import graph.

One deliberate widening: the bundled adapters disagreed on diagnostic severity
casing — ``dft-labeling`` filters ``level == "error"`` while ``pes-sampling``
checks ``level == "ERROR"``.  :func:`errors` and :func:`has_errors` compare
case-insensitively, which is a superset of both, so migrating an adapter to this
module cannot make it miss an error it used to catch.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


__all__ = [
    "SHA256_PATTERN",
    "diagnostic",
    "errors",
    "has_errors",
    "blocked",
    "mapping",
    "plain_string",
    "finite_number",
    "positive_number",
    "positive_int",
    "nonnegative_int",
    "is_fingerprint",
    "safe_relative",
    "is_within",
    "has_symlink_component",
    "ordinary_file",
    "under_root",
    "join_under",
    "sha256_file",
    "read_json",
    "operation",
]


SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


# --- diagnostics -----------------------------------------------------------


def diagnostic(level: str, code: str, message: str) -> dict[str, str]:
    """Build one diagnostic record in the shape every adapter already emits."""

    return {"level": level, "code": code, "message": message}


def errors(diagnostics: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Return only the error-level diagnostics, matching casing insensitively."""

    return [
        dict(item)
        for item in diagnostics
        if str(item.get("level", "")).lower() == "error"
    ]


def has_errors(diagnostics: Sequence[Mapping[str, Any]]) -> bool:
    return any(str(item.get("level", "")).lower() == "error" for item in diagnostics)


def blocked(
    plugin_id: str, diagnostics: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Build the standard non-executable plan an adapter returns when blocked."""

    return {
        "plugin_id": plugin_id,
        "status": "BLOCKED",
        "executable": False,
        "diagnostics": list(diagnostics),
    }


# --- value validation ------------------------------------------------------


def mapping(value: Any) -> Mapping[str, Any]:
    """Coerce to a mapping, treating anything else as empty.

    Accepts any ``Mapping`` rather than only ``dict``; that is the wider of the
    two variants the adapters carried.
    """

    return value if isinstance(value, Mapping) else {}


def plain_string(value: Any) -> bool:
    """True for a non-blank single-line string with no NUL."""

    return (
        isinstance(value, str)
        and bool(value.strip())
        and "\x00" not in value
        and "\n" not in value
        and "\r" not in value
    )


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def positive_number(value: Any) -> bool:
    return finite_number(value) and float(value) > 0


def positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def is_fingerprint(value: Any) -> bool:
    """True for a canonical ``sha256:<64 hex>`` fingerprint string."""

    return isinstance(value, str) and bool(SHA256_PATTERN.fullmatch(value))


# --- path safety -----------------------------------------------------------


def safe_relative(value: Any) -> bool:
    """True for a portable relative path: not absolute, no ``..``, not ``.``."""

    if not plain_string(value):
        return False
    path = Path(str(value))
    return path != Path(".") and not path.is_absolute() and ".." not in path.parts


def is_within(path: Path, root: Path) -> bool:
    """Pure lexical containment; callers resolve first when that matters."""

    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def has_symlink_component(path: Path) -> bool:
    """True if the path or any of its parents is a symlink."""

    absolute = path.absolute()
    return any(candidate.is_symlink() for candidate in (absolute, *absolute.parents))


def ordinary_file(path: Path, max_bytes: int | None = None) -> bool:
    """True for a real, non-symlink file, optionally within a size bound.

    With ``max_bytes`` set, an empty file is rejected too: adapters use this to
    require actual content, not merely an existing inode.
    """

    if path.is_symlink() or not path.is_file():
        return False
    if max_bytes is None:
        return True
    size = path.stat().st_size
    return 1 <= size <= max_bytes


def under_root(path: Path, root: Path, max_bytes: int | None = None) -> bool:
    """True for an ordinary file inside ``root`` reached without any symlink.

    Every component from ``root`` down is checked, so a symlinked intermediate
    directory cannot be used to escape and then resolve back inside.
    """

    resolved_root = root.expanduser().absolute()
    candidate = path.expanduser().absolute()
    try:
        relative = candidate.relative_to(resolved_root)
    except ValueError:
        return False
    current = resolved_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return False
    return is_within(candidate.resolve(), resolved_root.resolve()) and ordinary_file(
        candidate, max_bytes
    )


def join_under(root: Any, relative: Any) -> Path:
    """Join a relative reference onto an absolute root.

    Containment is *not* checked here — validate with :func:`safe_relative`
    before joining and :func:`under_root` after.
    """

    return Path(str(root)).expanduser().absolute() / str(relative)


# --- filesystem reads ------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Stream a file and return its canonical ``sha256:`` fingerprint."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json(
    path: Path, max_bytes: int | None = None
) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    """Read a bounded JSON object, returning ``(value, diagnostic)``.

    Never raises for ordinary bad input: a missing file, unreadable bytes, bad
    JSON and a non-object top level each come back as a diagnostic, which is how
    adapters report them to ``validate``.
    """

    if not ordinary_file(path, max_bytes):
        return None, diagnostic(
            "warning", "artifact.missing", f"not an ordinary readable file: {path}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, diagnostic("error", "artifact.invalid_json", f"cannot read JSON: {exc}")
    if not isinstance(value, dict):
        return None, diagnostic(
            "error", "artifact.not_object", "manifest must be a JSON object"
        )
    return value, None


# --- adapter context -------------------------------------------------------


def operation(context: Any, default: str = "") -> str:
    """Read ``parameters.operation`` from an adapter context.

    The core passes the node's parameters through untouched; which operation a
    node selects is therefore always a plain string here.
    """

    if not isinstance(context, Mapping):
        return default
    return str(mapping(context.get("parameters")).get("operation", default))
