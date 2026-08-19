"""Canonical SHA-256 identities for immutable scientific artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


class ArtifactIdentityError(ValueError):
    """An artifact cannot be assigned a stable file/tree identity."""


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    path = Path(path).resolve()
    if not path.is_file():
        raise ArtifactIdentityError(f"artifact is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def fingerprint_path(path: Path) -> str:
    """Hash a file or directory with the shared tree-sha256-v1 framing."""

    path = Path(path).resolve()
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise ArtifactIdentityError(f"artifact is not a regular file or directory: {path}")
    files = sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(path).as_posix(),
    )
    if not files:
        raise ArtifactIdentityError(f"artifact directory contains no files: {path}")
    digest = hashlib.sha256()
    for item in files:
        if item.is_symlink():
            raise ArtifactIdentityError(f"artifact tree contains a symlinked file: {item}")
        relative = item.relative_to(path).as_posix()
        record = f"{relative}\0{item.stat().st_size}\0{sha256_file(item)}\n"
        digest.update(record.encode("utf-8"))
    return "sha256:" + digest.hexdigest()
