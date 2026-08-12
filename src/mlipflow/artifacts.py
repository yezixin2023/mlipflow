"""Artifact references, content identity and bounded fingerprints.

Two different questions get asked about a file, and conflating them was a defect:

**"Is this the same content?"** — :func:`content_identity`.  This is what an
approval digest may depend on.  It is derived only from bytes, sizes and
repository-relative paths, so the same checkout answers identically after a
``touch``, a ``git clone`` into another directory, or an ``rsync`` to another
machine.

**"What did we observe on this filesystem?"** — :func:`fingerprint`.  This is the
provenance record stored with an artifact.  It keeps the absolute URI and
``mtime_ns`` because they are genuinely useful when auditing a run, and it must
never enter an approval digest.

Both share one content digest, so a directory tree's identity is mtime-free in
either view.  Before this split, ``_fingerprint_directory`` folded per-entry
mtimes into the *content* hash, which meant even the "full" tree mode changed
when nothing but a timestamp had.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


DEFAULT_FULL_HASH_MAX_BYTES = 64 * 1024 * 1024


# ---------------------------------------------------------------------------
# Content identity: safe to place in an approval digest.
# ---------------------------------------------------------------------------


def content_identity(
    path: Path,
    *,
    root: Path | None = None,
    full_hash_max_bytes: int = DEFAULT_FULL_HASH_MAX_BYTES,
) -> dict[str, Any]:
    """Describe *what* a path contains, never *when it was touched*.

    ``root`` supplies the frame of reference for the locator: a path inside it is
    recorded relative to it, so a project or plugin can be relocated or cloned
    without changing any digest.  A path outside every known root reports
    ``locator: None`` and is identified by content alone — the literal path is
    already bound elsewhere in the plan (in ``inputs`` or in the adapter's
    ``argv``), so repeating it here would only add machine dependence.

    Files larger than ``full_hash_max_bytes`` are deliberately not read.  They
    report ``content: None`` with ``content_mode: "size-only"`` rather than a
    hash of their metadata, because a metadata hash is not evidence of content
    and must not be mistaken for it.
    """

    locator = _locator(path, root)
    if not path.exists():
        return {"locator": locator, "exists": False}
    if path.is_dir():
        content, mode, size, count = _tree_content(path, full_hash_max_bytes)
        return {
            "locator": locator,
            "exists": True,
            "size_bytes": size,
            "file_count": count,
            "content": content,
            "content_mode": mode,
        }
    content, mode, size = _file_content(path, full_hash_max_bytes)
    return {
        "locator": locator,
        "exists": True,
        "size_bytes": size,
        "content": content,
        "content_mode": mode,
    }


def _locator(path: Path, root: Path | None) -> str | None:
    """Return a POSIX path relative to ``root``, or ``None`` when outside it."""

    if root is None:
        return None
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return None


def _file_content(path: Path, full_hash_max_bytes: int) -> tuple[str | None, str, int]:
    size = path.stat().st_size
    if size > full_hash_max_bytes:
        return None, "size-only", size
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}", "full", size


def _tree_content(
    path: Path, full_hash_max_bytes: int
) -> tuple[str, str, int, int]:
    """Hash a directory tree by structure and bytes, never by timestamps.

    Symlinks are recorded by their target text and never followed, so a tree
    cannot be made to hash as its target's contents.
    """

    entries: list[tuple[Path, str, int, str | None]] = []
    total_size = 0
    file_count = 0
    for item in sorted(path.rglob("*"), key=lambda value: value.relative_to(path).as_posix()):
        relative = item.relative_to(path).as_posix()
        if item.is_symlink():
            entries.append((item, relative, 0, item.readlink().as_posix()))
            continue
        if item.is_dir():
            entries.append((item, relative + "/", 0, None))
            continue
        size = item.stat().st_size
        total_size += size
        file_count += 1
        entries.append((item, relative, size, None))

    structure = hashlib.sha256()
    for _, relative, size, link_target in entries:
        kind = "L" if link_target is not None else ("D" if relative.endswith("/") else "F")
        structure.update(
            f"{kind}\0{relative}\0{size}\0{link_target or ''}\n".encode("utf-8")
        )
    if total_size > full_hash_max_bytes:
        return (
            f"tree-structure-sha256:{structure.hexdigest()}",
            "tree-size-only",
            total_size,
            file_count,
        )
    digest = hashlib.sha256()
    digest.update(structure.digest())
    for item, relative, _, link_target in entries:
        if link_target is not None or relative.endswith("/"):
            continue
        digest.update(relative.encode("utf-8") + b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return f"tree-sha256:{digest.hexdigest()}", "tree-full", total_size, file_count


# ---------------------------------------------------------------------------
# Observational record: provenance only, never an approval digest input.
# ---------------------------------------------------------------------------


def fingerprint(
    path: Path, full_hash_max_bytes: int = DEFAULT_FULL_HASH_MAX_BYTES
) -> dict[str, Any]:
    """Record one artifact as observed on this filesystem.

    Includes the absolute ``uri`` and ``mtime_ns`` on purpose: this is the record
    stored alongside a run so a human can find and audit the file later.  Because
    those two fields are machine- and time-specific, this result must not be
    placed in a plan — use :func:`content_identity` there.
    """

    stat = path.stat()
    if path.is_dir():
        content, mode, size, count = _tree_content(path, full_hash_max_bytes)
        return {
            "uri": path.resolve().as_uri(),
            "size_bytes": size,
            "mtime_ns": stat.st_mtime_ns,
            "file_count": count,
            "fingerprint": content,
            "fingerprint_mode": "tree-full" if mode == "tree-full" else "tree-metadata",
        }
    result: dict[str, Any] = {
        "uri": path.resolve().as_uri(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    content, mode, _ = _file_content(path, full_hash_max_bytes)
    if content is not None:
        result["fingerprint"] = content
        result["fingerprint_mode"] = "full"
    else:
        # Too large to hash.  The observational record still needs a stable
        # non-empty identifier, and here — unlike in content identity — encoding
        # the observed location and timestamp is the honest thing to report.
        digest = hashlib.sha256(
            f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
        ).hexdigest()
        result["fingerprint"] = f"metadata-sha256:{digest}"
        result["fingerprint_mode"] = "metadata"
    return result
