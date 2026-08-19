"""Portable rewriting of machine-specific paths in adapter-authored plan fields.

An adapter necessarily thinks in absolute paths: it emits the argv it wants run,
the working directory to run it in, and diagnostics naming the files it looked
at.  Those values are correct for the machine that produced them and meaningless
on any other, so placing them verbatim into ``plan_digest`` makes an approval
non-reproducible across a clone, a relocation or a second machine — the same
defect ``mtime_ns`` caused, arriving by a different route.

The core therefore rewrites adapter output against the roots it already knows
before the plan is signed::

    /home/u/proj/.mlipflow/runs/label/attempt-1/out   ->  {ATTEMPT_DIR}/out
    /home/u/proj/prepared/POSCAR                      ->  {PROJECT_ROOT}/prepared/POSCAR
    /opt/venv/share/mlipflow/plugins/dft/helper.py    ->  {PLUGIN_DIR}/helper.py

and resolves them back immediately before anything runs, so adapters keep
receiving the absolute paths they expect.  Approval sees a portable description
of what will happen; execution sees this machine's realisation of it.  Scientific
behaviour is untouched — the same argv runs in the same directory.

Paths outside every known root are left alone.  A declared interpreter such as
``resources.python_executable`` is site configuration that the project states
explicitly, so it is identical wherever that project file is used; discovering it
and rewriting it would hide a real part of the approval.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT_TOKEN = "{PROJECT_ROOT}"
ATTEMPT_DIR_TOKEN = "{ATTEMPT_DIR}"
PLUGIN_DIR_TOKEN = "{PLUGIN_DIR}"
PACKAGE_DIR_TOKEN = "{PACKAGE_DIR}"

TOKENS = (PROJECT_ROOT_TOKEN, ATTEMPT_DIR_TOKEN, PLUGIN_DIR_TOKEN, PACKAGE_DIR_TOKEN)


@dataclass(frozen=True)
class PortableRoots:
    """The roots an adapter's absolute paths may legitimately point into."""

    project_root: Path | None = None
    attempt_dir: Path | None = None
    plugin_dir: Path | None = None
    package_dir: Path | None = None

    def substitutions(self) -> list[tuple[str, str]]:
        """Return ``(absolute, token)`` pairs, longest absolute form first.

        Several spellings of one root are emitted because adapters variously call
        ``absolute()`` and ``resolve()``; on macOS those differ (``/var`` versus
        ``/private/var``), and a root that only matched one spelling would leave
        machine-specific text in the digest.

        Longest-first ordering matters: ``attempt_dir`` lives inside
        ``project_root``, and rewriting the parent first would strand the
        remainder as a relative fragment.
        """

        pairs: list[tuple[str, str]] = []
        for root, token in (
            (self.attempt_dir, ATTEMPT_DIR_TOKEN),
            (self.plugin_dir, PLUGIN_DIR_TOKEN),
            (self.package_dir, PACKAGE_DIR_TOKEN),
            (self.project_root, PROJECT_ROOT_TOKEN),
        ):
            for spelling in _spellings(root):
                pairs.append((spelling, token))
        pairs.sort(key=lambda item: len(item[0]), reverse=True)
        return pairs


def _spellings(root: Path | None) -> list[str]:
    if root is None:
        return []
    found: list[str] = []
    for candidate in (root, root.absolute(), _resolved(root)):
        if candidate is None:
            continue
        text = str(candidate).rstrip("/")
        if text and text != "/" and text not in found:
            found.append(text)
    return found


def _resolved(root: Path) -> Path | None:
    try:
        return root.resolve()
    except OSError:  # pragma: no cover - unreadable parent directory
        return None


def to_portable(value: Any, roots: PortableRoots) -> Any:
    """Replace this machine's roots with tokens throughout a JSON-able value."""

    return _rewrite(value, roots.substitutions())


def to_runtime(value: Any, roots: PortableRoots) -> Any:
    """Resolve tokens back to this machine's absolute paths."""

    pairs = [(token, absolute) for absolute, token in roots.substitutions()]
    # Later pairs would undo earlier ones only if two roots shared a token, which
    # ``substitutions`` never produces; the first spelling of each token wins so
    # that resolution is deterministic.
    seen: set[str] = set()
    unique = [
        (token, absolute)
        for token, absolute in pairs
        if not (token in seen or seen.add(token))
    ]
    return _rewrite(value, unique)


def _rewrite(value: Any, pairs: Iterable[tuple[str, str]]) -> Any:
    pairs = list(pairs)
    if isinstance(value, dict):
        return {key: _rewrite(item, pairs) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite(item, pairs) for item in value]
    if isinstance(value, str):
        for source, target in pairs:
            if source in value:
                value = value.replace(source, target)
        return value
    return value


def contains_absolute_root(value: Any, roots: PortableRoots) -> list[str]:
    """Return every string still naming one of this machine's roots.

    Used by tests and by ``doctor``-style checks to assert that nothing
    machine-specific survived into a signed plan.
    """

    spellings = [absolute for absolute, _ in roots.substitutions()]
    found: list[str] = []

    def walk(item: Any, trail: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                walk(child, f"{trail}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{trail}[{index}]")
        elif isinstance(item, str):
            if any(spelling in item for spelling in spellings) or item.startswith(
                "file:///"
            ):
                found.append(trail)

    walk(value, "")
    return found
