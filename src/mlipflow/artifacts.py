"""Artifact references and content identity.

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
either view. Files are hashed when their content identity matters; MLIPFlow no
longer manufactures ``metadata-sha256`` values for files it did not read.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Content identity: safe to place in an approval digest.
# ---------------------------------------------------------------------------


def content_identity(
    path: Path,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Describe *what* a path contains, never *when it was touched*.

    ``root`` supplies the frame of reference for the locator: a path inside it is
    recorded relative to it, so a project or plugin can be relocated or cloned
    without changing any digest.  A path outside every known root reports
    ``locator: None`` and is identified by content alone — the literal path is
    already bound elsewhere in the plan (in ``inputs`` or in the adapter's
    ``argv``), so repeating it here would only add machine dependence.

    Content identity is used only at real integrity boundaries, so regular files
    and trees are fully hashed instead of substituting metadata for bytes.
    """

    locator = _locator(path, root)
    if not path.exists():
        return {"locator": locator, "exists": False}
    if path.is_dir():
        content, size, count = _tree_content(path)
        return {
            "locator": locator,
            "exists": True,
            "size_bytes": size,
            "file_count": count,
            "content": content,
            "content_mode": "tree-full",
        }
    content, size = _file_content(path)
    return {
        "locator": locator,
        "exists": True,
        "size_bytes": size,
        "content": content,
        "content_mode": "full",
    }


def _locator(path: Path, root: Path | None) -> str | None:
    """Return a POSIX path relative to ``root``, or ``None`` when outside it."""

    if root is None:
        return None
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return None


def _file_content(path: Path) -> tuple[str, int]:
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}", size


def _tree_content(path: Path) -> tuple[str, int, int]:
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
    digest = hashlib.sha256()
    digest.update(structure.digest())
    for item, relative, _, link_target in entries:
        if link_target is not None or relative.endswith("/"):
            continue
        digest.update(relative.encode("utf-8") + b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return f"tree-sha256:{digest.hexdigest()}", total_size, file_count


# ---------------------------------------------------------------------------
# Observational record: provenance only, never an approval digest input.
# ---------------------------------------------------------------------------


def fingerprint(path: Path) -> dict[str, Any]:
    """Record one artifact as observed on this filesystem.

    Includes the absolute ``uri`` and ``mtime_ns`` on purpose: this is the record
    stored alongside a run so a human can find and audit the file later.  Because
    those two fields are machine- and time-specific, this result must not be
    placed in a plan — use :func:`content_identity` there.
    """

    stat = path.stat()
    if path.is_dir():
        content, size, count = _tree_content(path)
        return {
            "uri": path.resolve().as_uri(),
            "size_bytes": size,
            "mtime_ns": stat.st_mtime_ns,
            "file_count": count,
            "fingerprint": content,
            "fingerprint_mode": "tree-full",
        }
    result: dict[str, Any] = {
        "uri": path.resolve().as_uri(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    content, _ = _file_content(path)
    result["fingerprint"] = content
    result["fingerprint_mode"] = "full"
    return result
