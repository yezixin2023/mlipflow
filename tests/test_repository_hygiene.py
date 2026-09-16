from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path, PurePosixPath

import yaml


ROOT = Path(__file__).resolve().parents[1]


def tracked_files() -> list[Path]:
    """Inspect public source files without scanning ignored research workspaces."""
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT, text=True)
    return [ROOT / name for name in output.split("\0") if name and (ROOT / name).is_file()]


class RepositoryHygieneTests(unittest.TestCase):
    def test_specialist_skills_are_discoverable_through_relative_directory_links(self) -> None:
        discovery = ROOT / ".agents" / "skills"
        canonical_skills = sorted((ROOT / "mlipipe" / "plugins").glob("*/skill"))
        self.assertTrue(canonical_skills)
        names = {"mlip-workflow"}
        for canonical in canonical_skills:
            frontmatter = (canonical / "SKILL.md").read_text(encoding="utf-8").split("---", 2)[1]
            name = yaml.safe_load(frontmatter)["name"]
            with self.subTest(skill=name):
                self.assertNotIn(name, names)
                names.add(name)
                entry = discovery / name
                self.assertTrue(entry.is_symlink())
                self.assertFalse(entry.readlink().is_absolute())
                self.assertEqual(canonical, entry.resolve())
                self.assertEqual(
                    (canonical / "SKILL.md").read_bytes(),
                    (entry / "SKILL.md").read_bytes(),
                )
        self.assertFalse((discovery / "mlip-workflow").is_symlink())
        self.assertEqual(
            names,
            {entry.name for entry in discovery.iterdir() if (entry / "SKILL.md").is_file()},
        )


    def test_agent_skill_default_prompts_start_with_declared_skill_name(self) -> None:
        invalid: list[str] = []
        for skill in sorted((ROOT / ".agents/skills").iterdir()):
            skill_name = skill.name
            metadata_path = skill / "agents/openai.yaml"
            self.assertTrue(metadata_path.is_file(), metadata_path)
            metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
            interface = metadata.get("interface") if isinstance(metadata, dict) else None
            prompt = interface.get("default_prompt") if isinstance(interface, dict) else None
            prefix = f"${skill_name}"
            if not isinstance(prompt, str) or not (
                prompt == prefix or prompt.startswith(prefix + " ")
            ):
                invalid.append(metadata_path.relative_to(ROOT).as_posix())
        self.assertEqual([], sorted(invalid))


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
        tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
        found = {path for path in tracked.splitlines() if Path(path).name == ".DS_Store"}
        self.assertEqual(set(), found)

    def test_no_large_or_proprietary_artifacts(self) -> None:
        forbidden_names = {"POTCAR", "WAVECAR", "CHGCAR", "OUTCAR", "vasprun.xml"}
        forbidden_suffixes = {".model", ".pt", ".pth", ".pb", ".ckpt", ".traj", ".zip"}
        for path in tracked_files():
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
        for path in tracked_files():
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
                    part in {".mlipipe", ".pytest_cache", "__pycache__"}
                    or part.endswith(".egg-info")
                    for part in path.parts
                )
            ):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            self.assertIsNone(forbidden.search(text), str(path.relative_to(ROOT)))


if __name__ == "__main__":
    unittest.main()
