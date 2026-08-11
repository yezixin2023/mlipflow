from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryHygieneTests(unittest.TestCase):
    def test_release_data_declarations_cover_publishable_assets(self) -> None:
        """Keep wheel declarations in sync with the source assets shipped by the sdist."""

        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        section = pyproject.split("[tool.setuptools.data-files]", 1)[1]
        section = section.split("\n[", 1)[0]
        declared: set[str] = set()
        for line in section.splitlines():
            match = re.fullmatch(r'"[^"]+"\s*=\s*(\[.*\])', line.strip())
            if match is None:
                continue
            for pattern in ast.literal_eval(match.group(1)):
                declared.update(
                    path.relative_to(ROOT).as_posix() for path in ROOT.glob(pattern)
                )

        publishable: set[str] = set()
        for plugin_dir in (ROOT / "plugins").iterdir():
            if plugin_dir.is_dir():
                publishable.update(
                    path.relative_to(ROOT).as_posix()
                    for path in plugin_dir.iterdir()
                    if path.is_file() and path.suffix in {".py", ".yaml"}
                )
        for directory, suffixes in (
            (ROOT / "schemas", {".json"}),
            (ROOT / ".agents" / "skills", {".md", ".yaml"}),
            (ROOT / "docs", {".md"}),
            (
                ROOT / "examples" / "high_entropy_sulfide_reproduction",
                {".py", ".md", ".yaml", ".json", ".csv"},
            ),
            (ROOT / "reports", {".md", ".json"}),
        ):
            publishable.update(
                path.relative_to(ROOT).as_posix()
                for path in directory.rglob("*")
                if path.is_file() and path.suffix in suffixes
            )

        self.assertEqual(set(), publishable - declared)

    def test_no_generated_build_directories(self) -> None:
        # pytest and Python may create their own ignored caches before this
        # test module is collected.  Treating those runtime caches as a test
        # failure makes an ordinary ``python -m pytest`` self-defeating.  The
        # publishable-tree check therefore rejects build products here and the
        # release command separately runs from a cleaned tree.
        found = {
            path.relative_to(ROOT).as_posix()
            for path in ROOT.rglob("*")
            if path.is_dir()
            and (path.name in {"build", "dist"} or path.name.endswith(".egg-info"))
        }
        self.assertEqual(set(), found)

    def test_no_finder_metadata(self) -> None:
        found = {
            path.relative_to(ROOT).as_posix()
            for path in ROOT.rglob(".DS_Store")
            if path.is_file()
        }
        self.assertEqual(set(), found)

    def test_no_large_or_proprietary_artifacts(self) -> None:
        forbidden_names = {"POTCAR", "WAVECAR", "CHGCAR", "OUTCAR", "vasprun.xml"}
        forbidden_suffixes = {".model", ".pt", ".pth", ".pb", ".ckpt", ".traj", ".zip"}
        for path in ROOT.rglob("*"):
            if not path.is_file() or any(part.startswith(".") and part != ".agents" for part in path.parts):
                continue
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertNotIn(path.name, forbidden_names)
                self.assertNotIn(path.suffix, forbidden_suffixes)
                self.assertLess(path.stat().st_size, 1_000_000)

    def test_no_private_key_or_personal_absolute_path(self) -> None:
        forbidden = re.compile(
            r"BEGIN [A-Z ]*PRIVATE KEY"
            r"|/Users/[A-Za-z0-9._-]+/"
            r"|/public/home/(?!example(?:/|[\"']))[A-Za-z0-9._-]+"
            r"|IdentityFile\s+|hfeshell|lihr1008",
            re.I,
        )
        for path in ROOT.rglob("*"):
            if (
                not path.is_file()
                or path.resolve() == Path(__file__).resolve()
                or path.suffix
                not in {
                    ".cff",
                    ".csv",
                    ".in",
                    ".json",
                    ".md",
                    ".py",
                    ".toml",
                    ".txt",
                    ".yaml",
                }
                or any(part in {".pytest_cache", "__pycache__"} for part in path.parts)
            ):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            self.assertIsNone(forbidden.search(text), str(path.relative_to(ROOT)))


if __name__ == "__main__":
    unittest.main()
