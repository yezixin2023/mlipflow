"""Artifact references and bounded fingerprints."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


DEFAULT_FULL_HASH_MAX_BYTES = 64 * 1024 * 1024


def fingerprint(path: Path, full_hash_max_bytes: int = DEFAULT_FULL_HASH_MAX_BYTES) -> dict[str, Any]:
    stat = path.stat()
    if path.is_dir():
        return _fingerprint_directory(path, full_hash_max_bytes)
    result: dict[str, Any] = {
        "uri": path.resolve().as_uri(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if path.is_file() and stat.st_size <= full_hash_max_bytes:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result["fingerprint"] = f"sha256:{digest.hexdigest()}"
        result["fingerprint_mode"] = "full"
    else:
        digest = hashlib.sha256(
            f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
        ).hexdigest()
        result["fingerprint"] = f"metadata-sha256:{digest}"
        result["fingerprint_mode"] = "metadata"
    return result


def _fingerprint_directory(path: Path, full_hash_max_bytes: int) -> dict[str, Any]:
    """Fingerprint a directory tree without following symlink targets.

    Small trees include file bytes. Large trees use deterministic path/type/
    size/mtime metadata, matching the project's bounded-hash policy.
    """

    root_stat = path.stat()
    entries: list[tuple[Path, str, int, int, str | None]] = []
    total_size = 0
    file_count = 0
    for item in sorted(path.rglob("*"), key=lambda value: value.relative_to(path).as_posix()):
        relative = item.relative_to(path).as_posix()
        if item.is_symlink():
            entries.append((item, relative, 0, 0, item.readlink().as_posix()))
            continue
        item_stat = item.stat()
        if item.is_dir():
            entries.append((item, relative + "/", 0, item_stat.st_mtime_ns, None))
            continue
        size = item_stat.st_size
        total_size += size
        file_count += 1
        entries.append((item, relative, size, item_stat.st_mtime_ns, None))

    metadata = hashlib.sha256()
    for _, relative, size, mtime_ns, link_target in entries:
        kind = "L" if link_target is not None else ("D" if relative.endswith("/") else "F")
        metadata.update(
            f"{kind}\0{relative}\0{size}\0{mtime_ns}\0{link_target or ''}\n".encode("utf-8")
        )
    if total_size <= full_hash_max_bytes:
        digest = hashlib.sha256()
        digest.update(metadata.digest())
        for item, relative, _, _, link_target in entries:
            if link_target is not None or relative.endswith("/"):
                continue
            digest.update(relative.encode("utf-8") + b"\0")
            with item.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        value = f"tree-sha256:{digest.hexdigest()}"
        mode = "tree-full"
    else:
        value = f"tree-metadata-sha256:{metadata.hexdigest()}"
        mode = "tree-metadata"
    return {
        "uri": path.resolve().as_uri(),
        "size_bytes": total_size,
        "mtime_ns": root_stat.st_mtime_ns,
        "file_count": file_count,
        "fingerprint": value,
        "fingerprint_mode": mode,
    }
