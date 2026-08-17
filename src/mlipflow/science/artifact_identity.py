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
    """Hash a file or directory with stable relative-path and size framing."""

    path = Path(path).resolve()
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise ArtifactIdentityError(f"artifact is not a regular file or directory: {path}")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ArtifactIdentityError(f"artifact directory contains no files: {path}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(item.stat().st_size.to_bytes(8, "big"))
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return "sha256:" + digest.hexdigest()
