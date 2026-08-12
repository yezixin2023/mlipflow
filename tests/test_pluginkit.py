"""Unit tests for the shared adapter helper surface.

These primitives replace four to eight independently drifting copies each, and
two of them (``safe_relative``, ``under_root``) are security boundaries, so they
are tested directly rather than only through the adapters that will use them.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from mlipflow import pluginkit as kit


class DiagnosticTests(unittest.TestCase):
    def test_diagnostic_shape(self) -> None:
        self.assertEqual(
            {"level": "error", "code": "x.y", "message": "boom"},
            kit.diagnostic("error", "x.y", "boom"),
        )

    def test_error_filtering_accepts_both_casings(self) -> None:
        """dft-labeling used 'error', pes-sampling used 'ERROR'; both count."""

        diagnostics = [
            kit.diagnostic("error", "a", "lower"),
            kit.diagnostic("ERROR", "b", "upper"),
            kit.diagnostic("warning", "c", "ignored"),
        ]
        self.assertEqual(["a", "b"], [item["code"] for item in kit.errors(diagnostics)])
        self.assertTrue(kit.has_errors(diagnostics))

    def test_no_errors(self) -> None:
        diagnostics = [kit.diagnostic("warning", "c", "ignored")]
        self.assertEqual([], kit.errors(diagnostics))
        self.assertFalse(kit.has_errors(diagnostics))
        self.assertFalse(kit.has_errors([]))

    def test_missing_level_is_not_an_error(self) -> None:
        self.assertFalse(kit.has_errors([{"code": "a", "message": "b"}]))

    def test_errors_returns_copies(self) -> None:
        original = [kit.diagnostic("error", "a", "m")]
        extracted = kit.errors(original)
        extracted[0]["message"] = "mutated"
        self.assertEqual("m", original[0]["message"])

    def test_blocked_plan(self) -> None:
        diagnostics = [kit.diagnostic("error", "a", "m")]
        self.assertEqual(
            {
                "plugin_id": "demo",
                "status": "BLOCKED",
                "executable": False,
                "diagnostics": diagnostics,
            },
            kit.blocked("demo", diagnostics),
        )


class ValueValidationTests(unittest.TestCase):
    def test_mapping_accepts_any_mapping(self) -> None:
        import types

        self.assertEqual({"a": 1}, dict(kit.mapping({"a": 1})))
        self.assertEqual({"a": 1}, dict(kit.mapping(types.MappingProxyType({"a": 1}))))
        self.assertEqual({}, dict(kit.mapping(None)))
        self.assertEqual({}, dict(kit.mapping([("a", 1)])))

    def test_plain_string_rejects_blank_and_control_characters(self) -> None:
        self.assertTrue(kit.plain_string("ok"))
        for bad in ("", "   ", "a\nb", "a\rb", "a\x00b", 1, None, b"bytes"):
            with self.subTest(value=bad):
                self.assertFalse(kit.plain_string(bad))

    def test_numbers_reject_bool_and_non_finite(self) -> None:
        self.assertTrue(kit.finite_number(0))
        self.assertTrue(kit.finite_number(-1.5))
        for bad in (True, False, float("nan"), float("inf"), "1", None):
            with self.subTest(value=bad):
                self.assertFalse(kit.finite_number(bad))
        self.assertTrue(kit.positive_number(0.5))
        self.assertFalse(kit.positive_number(0))
        self.assertTrue(kit.positive_int(1))
        self.assertFalse(kit.positive_int(0))
        self.assertFalse(kit.positive_int(True))
        self.assertFalse(kit.positive_int(1.0))
        self.assertTrue(kit.nonnegative_int(0))
        self.assertFalse(kit.nonnegative_int(-1))
        self.assertFalse(kit.nonnegative_int(True))

    def test_fingerprint_pattern(self) -> None:
        self.assertTrue(kit.is_fingerprint("sha256:" + "a" * 64))
        self.assertFalse(kit.is_fingerprint("sha256:" + "A" * 64))
        self.assertFalse(kit.is_fingerprint("sha256:" + "a" * 63))
        self.assertFalse(kit.is_fingerprint("a" * 64))
        self.assertFalse(kit.is_fingerprint(None))


class PathSafetyTests(unittest.TestCase):
    def test_safe_relative_rejects_escapes(self) -> None:
        self.assertTrue(kit.safe_relative("input/POSCAR"))
        for bad in ("/etc/passwd", "../secret", "a/../../b", ".", "", "a\nb", None):
            with self.subTest(value=bad):
                self.assertFalse(kit.safe_relative(bad))

    def test_is_within(self) -> None:
        root = Path("/tmp/project")
        self.assertTrue(kit.is_within(root / "a" / "b", root))
        self.assertTrue(kit.is_within(root, root))
        self.assertFalse(kit.is_within(Path("/tmp/other"), root))


class FilesystemTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name).resolve()

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_sha256_matches_hashlib(self) -> None:
        import hashlib

        path = self.root / "payload.bin"
        payload = os.urandom(4096)
        path.write_bytes(payload)
        self.assertEqual(
            "sha256:" + hashlib.sha256(payload).hexdigest(), kit.sha256_file(path)
        )

    def test_ordinary_file_rejects_symlink_and_bounds(self) -> None:
        target = self.root / "real.txt"
        target.write_text("content", encoding="utf-8")
        link = self.root / "link.txt"
        link.symlink_to(target)
        self.assertTrue(kit.ordinary_file(target))
        self.assertFalse(kit.ordinary_file(link))
        self.assertFalse(kit.ordinary_file(self.root / "missing.txt"))
        self.assertTrue(kit.ordinary_file(target, 100))
        self.assertFalse(kit.ordinary_file(target, 3))
        empty = self.root / "empty.txt"
        empty.write_text("", encoding="utf-8")
        self.assertTrue(kit.ordinary_file(empty))
        self.assertFalse(kit.ordinary_file(empty, 100), "bounded reads require content")

    def test_under_root_rejects_symlinked_intermediate_directory(self) -> None:
        """A symlinked parent must not be usable to re-enter the root."""

        outside = self.root / "outside"
        outside.mkdir()
        (outside / "payload.txt").write_text("x", encoding="utf-8")
        project = self.root / "project"
        project.mkdir()
        (project / "sneaky").symlink_to(outside)
        self.assertFalse(kit.under_root(project / "sneaky" / "payload.txt", project))

        honest = project / "input"
        honest.mkdir()
        (honest / "payload.txt").write_text("x", encoding="utf-8")
        self.assertTrue(kit.under_root(honest / "payload.txt", project))

    def test_under_root_rejects_paths_outside(self) -> None:
        project = self.root / "project"
        project.mkdir()
        other = self.root / "other.txt"
        other.write_text("x", encoding="utf-8")
        self.assertFalse(kit.under_root(other, project))

    def test_join_under(self) -> None:
        joined = kit.join_under(self.root, "input/POSCAR")
        self.assertEqual(self.root / "input" / "POSCAR", joined)
        self.assertTrue(joined.is_absolute())

    def test_read_json_reports_problems_as_diagnostics(self) -> None:
        missing, note = kit.read_json(self.root / "absent.json", 1024)
        self.assertIsNone(missing)
        self.assertEqual("artifact.missing", note["code"])

        broken = self.root / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        value, note = kit.read_json(broken, 1024)
        self.assertIsNone(value)
        self.assertEqual("artifact.invalid_json", note["code"])

        listed = self.root / "list.json"
        listed.write_text("[1, 2]", encoding="utf-8")
        value, note = kit.read_json(listed, 1024)
        self.assertIsNone(value)
        self.assertEqual("artifact.not_object", note["code"])

        good = self.root / "good.json"
        good.write_text(json.dumps({"a": 1}), encoding="utf-8")
        value, note = kit.read_json(good, 1024)
        self.assertEqual({"a": 1}, value)
        self.assertIsNone(note)

    def test_read_json_honours_the_size_bound(self) -> None:
        path = self.root / "big.json"
        path.write_text(json.dumps({"a": "x" * 500}), encoding="utf-8")
        value, note = kit.read_json(path, 10)
        self.assertIsNone(value)
        self.assertEqual("artifact.missing", note["code"])


class OperationTests(unittest.TestCase):
    def test_operation_reads_parameters(self) -> None:
        self.assertEqual(
            "label", kit.operation({"parameters": {"operation": "label"}})
        )

    def test_operation_falls_back(self) -> None:
        self.assertEqual("", kit.operation({}))
        self.assertEqual("", kit.operation({"parameters": {}}))
        self.assertEqual("direct", kit.operation(None, "direct"))
        self.assertEqual("direct", kit.operation({"parameters": None}, "direct"))


if __name__ == "__main__":
    unittest.main()
