"""Approval identity must describe content, not filesystem incidentals.

Plan schema version 2 is an *intentional* digest semantic change.  Version 1 fed
``mtime_ns`` and absolute ``file://`` URIs into ``plan_digest`` through
``fingerprint()``, so a ``touch``, a fresh ``git clone`` into another directory,
or an ``rsync`` to another machine invalidated an already-approved plan even
though the bytes were identical.  Worse, a queued scheduled job could no longer
be advanced, because ``_load_pinned_scheduled_plan`` compares the pinned plan's
``plugin`` block against a freshly computed one.

Every test here pins one half of the contract:

* identical bytes must produce an identical approval digest, however the file
  got there;
* changed bytes must produce a different one.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from mlipflow.artifacts import content_identity, fingerprint
from mlipflow.config import load_project
from mlipflow.services import make_run_plan

from .helpers import plugin_manifest, project_config, write_json


OLD_MTIME = (1_600_000_000, 1_600_000_000)
NEW_MTIME = (1_700_000_000, 1_700_000_000)


def build_project(root: Path) -> Path:
    """Create a project whose plan binds a plugin adapter, a file and a tree."""

    plugins = root / "plugins"
    manifest = plugin_manifest()
    manifest["implementation"]["status"] = "contract-only"
    manifest["execution"]["backends"] = ["local"]
    write_json(plugins / "demo" / "plugin.yaml", manifest)
    (plugins / "demo" / "adapter.py").write_text(
        "class Adapter:\n    pass\n", encoding="utf-8"
    )

    (root / "structures.xyz").write_text("one\n", encoding="utf-8")
    dataset = root / "dataset"
    (dataset / "nested").mkdir(parents=True)
    (dataset / "part.dat").write_text("alpha\n", encoding="utf-8")
    (dataset / "nested" / "more.dat").write_text("beta\n", encoding="utf-8")

    node = {
        "id": "x",
        "uses": "demo@1",
        "mode": "execute",
        "inputs": {"structures": "structures.xyz", "data": "dataset"},
        "parameters": {"argv": ["true"]},
    }
    write_json(root / "project.yaml", project_config([node]))
    return plugins


def plan_digest(root: Path) -> str:
    return make_run_plan(load_project(root), "x", root / "plugins")["plan_digest"]


def touch_tree(root: Path, times: tuple[int, int]) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        os.utime(path, times, follow_symlinks=False)
    os.utime(root, times)


class ApprovalDigestStabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.base = Path(self._temporary.name).resolve()

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_same_bytes_different_mtime_gives_same_digest(self) -> None:
        root = self.base / "project"
        root.mkdir()
        build_project(root)
        touch_tree(root, OLD_MTIME)
        before = plan_digest(root)

        touch_tree(root, NEW_MTIME)
        self.assertEqual(before, plan_digest(root))

    def test_same_bytes_different_checkout_root_gives_same_digest(self) -> None:
        """A copy at another path — the clone/rsync case — must approve alike."""

        root = self.base / "checkout-a"
        root.mkdir()
        build_project(root)
        before = plan_digest(root)

        other = self.base / "some" / "deeper" / "checkout-b"
        other.parent.mkdir(parents=True)
        shutil.copytree(root, other)
        touch_tree(other, NEW_MTIME)
        self.assertEqual(before, plan_digest(other))

    def test_real_git_clone_gives_same_digest(self) -> None:
        """The end-to-end case that motivated this fix."""

        if shutil.which("git") is None:  # pragma: no cover - git is normally present
            self.skipTest("git is unavailable")
        root = self.base / "origin"
        root.mkdir()
        build_project(root)
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.invalid",
        }
        for argv in (
            ["git", "init", "-q", "-b", "main"],
            ["git", "add", "-A"],
            ["git", "commit", "-q", "-m", "fixture"],
        ):
            subprocess.run(argv, cwd=root, env=env, check=True)
        before = plan_digest(root)

        clone = self.base / "clone"
        subprocess.run(
            ["git", "clone", "-q", str(root), str(clone)], env=env, check=True
        )
        self.assertEqual(before, plan_digest(clone))

    def test_changed_file_bytes_change_the_digest(self) -> None:
        root = self.base / "project"
        root.mkdir()
        build_project(root)
        before = plan_digest(root)

        target = root / "structures.xyz"
        os.utime(target, OLD_MTIME)
        target.write_text("two\n", encoding="utf-8")
        os.utime(target, OLD_MTIME)
        self.assertNotEqual(before, plan_digest(root))

    def test_changed_adapter_source_changes_the_digest(self) -> None:
        """The guarantee the mtime dependence was accidentally providing."""

        root = self.base / "project"
        root.mkdir()
        plugins = build_project(root)
        touch_tree(root, OLD_MTIME)
        before = plan_digest(root)

        adapter = plugins / "demo" / "adapter.py"
        adapter.write_text("class Adapter:\n    changed = True\n", encoding="utf-8")
        touch_tree(root, OLD_MTIME)
        self.assertNotEqual(before, plan_digest(root))

    def test_directory_content_change_changes_the_digest(self) -> None:
        root = self.base / "project"
        root.mkdir()
        build_project(root)
        touch_tree(root, OLD_MTIME)
        before = plan_digest(root)

        (root / "dataset" / "part.dat").write_text("gamma\n", encoding="utf-8")
        touch_tree(root, OLD_MTIME)
        self.assertNotEqual(before, plan_digest(root))

    def test_directory_rename_changes_the_digest(self) -> None:
        """Structure is identity too: same bytes at a different path differ."""

        root = self.base / "project"
        root.mkdir()
        build_project(root)
        touch_tree(root, OLD_MTIME)
        before = plan_digest(root)

        (root / "dataset" / "part.dat").rename(root / "dataset" / "renamed.dat")
        touch_tree(root, OLD_MTIME)
        self.assertNotEqual(before, plan_digest(root))


class ContentIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name).resolve()

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_identity_never_carries_mtime_or_absolute_path(self) -> None:
        path = self.root / "file.txt"
        path.write_text("payload", encoding="utf-8")
        identity = content_identity(path, root=self.root)
        self.assertEqual(
            {"locator", "exists", "size_bytes", "content", "content_mode"},
            set(identity),
        )
        self.assertEqual("file.txt", identity["locator"])
        self.assertNotIn("mtime_ns", identity)
        self.assertNotIn("uri", identity)
        self.assertNotIn(str(self.root), repr(identity))

    def test_file_identity_is_the_plain_content_hash(self) -> None:
        import hashlib

        path = self.root / "file.txt"
        payload = b"payload"
        path.write_bytes(payload)
        self.assertEqual(
            "sha256:" + hashlib.sha256(payload).hexdigest(),
            content_identity(path, root=self.root)["content"],
        )

    def test_path_outside_root_reports_no_locator_but_keeps_content(self) -> None:
        inside = self.root / "project"
        inside.mkdir()
        outside = self.root / "elsewhere.bin"
        outside.write_bytes(b"payload")
        identity = content_identity(outside, root=inside)
        self.assertIsNone(identity["locator"])
        self.assertTrue(identity["content"].startswith("sha256:"))

    def test_missing_path_reports_absence_by_relative_locator(self) -> None:
        """A declared-but-absent input keeps a checkout-stable locator."""

        identity = content_identity(self.root / "absent.txt", root=self.root)
        self.assertEqual({"locator": "absent.txt", "exists": False}, identity)
        self.assertNotIn(str(self.root), repr(identity))

    def test_missing_path_outside_root_reports_no_locator(self) -> None:
        inside = self.root / "project"
        inside.mkdir()
        identity = content_identity(self.root / "absent.txt", root=inside)
        self.assertEqual({"locator": None, "exists": False}, identity)

    def test_large_file_uses_its_content_not_a_metadata_hash(self) -> None:

        path = self.root / "big.bin"
        path.write_bytes(b"x" * 4096)
        identity = content_identity(path, root=self.root)
        self.assertTrue(identity["content"].startswith("sha256:"))
        self.assertEqual("full", identity["content_mode"])
        self.assertEqual(4096, identity["size_bytes"])

    def test_oversized_file_identity_is_mtime_and_location_stable(self) -> None:
        first = self.root / "a"
        second = self.root / "b"
        first.mkdir()
        second.mkdir()
        (first / "big.bin").write_bytes(b"x" * 4096)
        (second / "big.bin").write_bytes(b"x" * 4096)
        os.utime(first / "big.bin", OLD_MTIME)
        os.utime(second / "big.bin", NEW_MTIME)
        self.assertEqual(
            content_identity(first / "big.bin", root=first),
            content_identity(second / "big.bin", root=second),
        )


class TreeIdentityTests(unittest.TestCase):
    """Directories had the same defect: mtimes were folded into 'tree-full'."""

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name).resolve()
        self.tree = self.root / "tree"
        (self.tree / "sub").mkdir(parents=True)
        (self.tree / "a.dat").write_text("alpha\n", encoding="utf-8")
        (self.tree / "sub" / "b.dat").write_text("beta\n", encoding="utf-8")

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def identity(self, **kwargs: object) -> dict:
        return content_identity(self.tree, root=self.root, **kwargs)  # type: ignore[arg-type]

    def test_tree_full_ignores_mtimes(self) -> None:
        touch_tree(self.tree, OLD_MTIME)
        before = self.identity()
        self.assertEqual("tree-full", before["content_mode"])

        touch_tree(self.tree, NEW_MTIME)
        self.assertEqual(before, self.identity())

    def test_tree_full_mode_ignores_mtimes(self) -> None:
        touch_tree(self.tree, OLD_MTIME)
        before = self.identity()
        self.assertEqual("tree-full", before["content_mode"])

        touch_tree(self.tree, NEW_MTIME)
        self.assertEqual(before, self.identity())

    def test_tree_content_change_is_detected(self) -> None:
        touch_tree(self.tree, OLD_MTIME)
        before = self.identity()
        (self.tree / "sub" / "b.dat").write_text("changed\n", encoding="utf-8")
        touch_tree(self.tree, OLD_MTIME)
        self.assertNotEqual(before, self.identity())

    def test_tree_path_change_is_detected(self) -> None:
        touch_tree(self.tree, OLD_MTIME)
        full_before = self.identity()

        (self.tree / "sub" / "b.dat").rename(self.tree / "sub" / "renamed.dat")
        touch_tree(self.tree, OLD_MTIME)
        self.assertNotEqual(full_before, self.identity())

    def test_tree_identity_survives_a_copy_to_another_root(self) -> None:
        touch_tree(self.tree, OLD_MTIME)
        before = self.identity()
        other_root = self.root / "other"
        other_root.mkdir()
        shutil.copytree(self.tree, other_root / "tree")
        touch_tree(other_root / "tree", NEW_MTIME)
        self.assertEqual(
            before, content_identity(other_root / "tree", root=other_root)
        )

    def test_symlink_target_text_is_part_of_identity_and_is_not_followed(self) -> None:
        secret = self.root / "secret.txt"
        secret.write_text("secret\n", encoding="utf-8")
        (self.tree / "link").symlink_to(secret)
        touch_tree(self.tree, OLD_MTIME)
        before = self.identity()

        (self.tree / "link").unlink()
        (self.tree / "link").symlink_to(self.root / "other.txt")
        touch_tree(self.tree, OLD_MTIME)
        self.assertNotEqual(before, self.identity())


class ObservationalFingerprintTests(unittest.TestCase):
    """The provenance record keeps what identity must not."""

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name).resolve()

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_fingerprint_still_records_uri_and_mtime(self) -> None:
        path = self.root / "file.txt"
        path.write_text("payload", encoding="utf-8")
        os.utime(path, OLD_MTIME)
        observed = fingerprint(path)
        self.assertTrue(observed["uri"].startswith("file://"))
        self.assertEqual(OLD_MTIME[1] * 1_000_000_000, observed["mtime_ns"])
        self.assertEqual("full", observed["fingerprint_mode"])

    def test_fingerprint_and_identity_agree_on_content(self) -> None:
        path = self.root / "file.txt"
        path.write_text("payload", encoding="utf-8")
        self.assertEqual(
            fingerprint(path)["fingerprint"],
            content_identity(path, root=self.root)["content"],
        )

    def test_directory_fingerprint_is_now_mtime_free_too(self) -> None:
        tree = self.root / "tree"
        tree.mkdir()
        (tree / "a.dat").write_text("alpha\n", encoding="utf-8")
        touch_tree(tree, OLD_MTIME)
        before = fingerprint(tree)["fingerprint"]
        touch_tree(tree, NEW_MTIME)
        self.assertEqual(before, fingerprint(tree)["fingerprint"])


if __name__ == "__main__":
    unittest.main()
