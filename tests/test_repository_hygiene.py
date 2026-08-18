from __future__ import annotations

import ast
import re
import subprocess
import unittest
from pathlib import Path, PurePosixPath

import yaml


ROOT = Path(__file__).resolve().parents[1]


def declared_agent_skill_files() -> dict[str, set[Path]]:
    """Return every source file named by an agent-skill data-files declaration."""

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = pyproject.split("[tool.setuptools.data-files]", 1)[1]
    section = section.split("\n[", 1)[0]
    declarations: dict[str, set[Path]] = {}
    prefix = "share/mlipflow/agent-skills/"
    for line in section.splitlines():
        match = re.fullmatch(r'"([^"]+)"\s*=\s*(\[.*\])', line.strip())
        if match is None or not match.group(1).startswith(prefix):
            continue
        destination = match.group(1)[len(prefix) :]
        skill_name = destination.split("/", 1)[0]
        declarations.setdefault(skill_name, set()).update(
            ROOT / pattern for pattern in ast.literal_eval(match.group(2))
        )
    return declarations


class RepositoryHygieneTests(unittest.TestCase):
    def test_agent_skill_packaging_declarations_resolve_to_files(self) -> None:
        declarations = declared_agent_skill_files()
        self.assertTrue(declarations)
        missing = sorted(
            path.relative_to(ROOT).as_posix()
            for paths in declarations.values()
            for path in paths
            if not path.is_file()
        )
        if missing:
            self.fail(
                "agent skill packaging declarations reference missing files:\n"
                + "\n".join(missing)
            )

    def test_agent_skill_default_prompts_start_with_declared_skill_name(self) -> None:
        invalid: list[str] = []
        for skill_name, paths in declared_agent_skill_files().items():
            metadata_paths = [
                path
                for path in paths
                if path.as_posix().endswith("/agents/openai.yaml")
            ]
            if len(metadata_paths) != 1 or not metadata_paths[0].is_file():
                continue
            metadata = yaml.safe_load(metadata_paths[0].read_text(encoding="utf-8"))
            interface = metadata.get("interface") if isinstance(metadata, dict) else None
            prompt = interface.get("default_prompt") if isinstance(interface, dict) else None
            prefix = f"${skill_name}"
            if not isinstance(prompt, str) or not (
                prompt == prefix or prompt.startswith(prefix + " ")
            ):
                invalid.append(metadata_paths[0].relative_to(ROOT).as_posix())
        self.assertEqual([], sorted(invalid))

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

    def test_no_generated_build_artifacts_are_tracked(self) -> None:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
        )
        tracked_paths = (
            item.decode("utf-8", errors="surrogateescape")
            for item in result.stdout.split(b"\0")
            if item
        )
        violations = sorted(
            path
            for path in tracked_paths
            if any(
                part in {"build", "dist"} or part.endswith(".egg-info")
                for part in PurePosixPath(path).parts
            )
        )
        if violations:
            self.fail(
                "generated build artifacts must not be part of the repository source tree:\n"
                + "\n".join(violations)
            )

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
                or any(
                    part in {".mlipflow", ".pytest_cache", "__pycache__"}
                    for part in path.parts
                )
            ):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            self.assertIsNone(forbidden.search(text), str(path.relative_to(ROOT)))


if __name__ == "__main__":
    unittest.main()
