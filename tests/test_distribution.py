"""Release artifacts must work without importing the developer checkout."""

from __future__ import annotations

import os
from email.parser import Parser
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import yaml
from packaging.specifiers import SpecifierSet


ROOT = Path(__file__).resolve().parents[1]


def run(argv, *, cwd):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    result = subprocess.run(
        [str(value) for value in argv],
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def assert_public_members(members):
    forbidden_names = {".DS_Store", "POTCAR", "WAVECAR", "CHGCAR", "OUTCAR", "vasprun.xml"}
    forbidden_suffixes = {".model", ".pt", ".pth", ".pb", ".ckpt", ".traj", ".zip", ".pyc"}
    for name, size in members:
        assert Path(name).name not in forbidden_names, name
        assert Path(name).suffix not in forbidden_suffixes, name
        assert size < 1_000_000, name


def agent_skill_sources():
    directories = sorted((ROOT / "mlipflow" / "plugins").glob("*/skill"))
    directories.append(ROOT / ".agents" / "skills" / "mlip-workflow")
    for directory in directories:
        frontmatter = (directory / "SKILL.md").read_text(encoding="utf-8").split("---", 2)[1]
        name = yaml.safe_load(frontmatter)["name"]
        files = sorted(path for path in directory.rglob("*") if path.is_file())
        yield directory, name, files


def test_wheel_and_sdist_install_outside_checkout(tmp_path):
    release = tmp_path / "release"
    run([sys.executable, "-m", "build", "--no-isolation", "--outdir", release], cwd=ROOT)
    wheel = next(release.glob("*.whl"))
    sdist = next(release.glob("*.tar.gz"))
    with zipfile.ZipFile(wheel) as archive:
        assert_public_members((item.filename, item.file_size) for item in archive.infolist())
        names = archive.namelist()
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
        supported_python = SpecifierSet(metadata["Requires-Python"])
        assert "3.12" in supported_python
        assert all(version not in supported_python for version in ("3.9", "3.10", "3.11"))
        assert "mlipflow/plugins/dft_labeling/adapter.py" in names
        sources = list(agent_skill_sources())
        assert sum(name.endswith("/SKILL.md") for name in names) == len(sources)
        expected_resources = set()
        for directory, skill_name, files in sources:
            for path in files:
                relative = path.relative_to(directory).as_posix()
                if directory.is_relative_to(ROOT / "mlipflow"):
                    packaged = path.relative_to(ROOT).as_posix()
                else:
                    export = f"/share/mlipflow/agent-skills/{skill_name}/{relative}"
                    matches = [name for name in names if name.endswith(export)]
                    assert len(matches) == 1, export
                    packaged = matches[0]
                expected_resources.add(packaged)
                assert archive.read(packaged) == path.read_bytes()
        # Only Python code, one copy of each Skill, and wheel metadata belong here.
        actual_resources = {
            name for name in names
            if not (name.startswith("mlipflow/") and name.endswith(".py"))
            and ".dist-info/" not in name
        }
        assert actual_resources == expected_resources
    with tarfile.open(sdist) as archive:
        assert_public_members((item.name, item.size) for item in archive.getmembers() if item.isfile())
        names = archive.getnames()
        assert any(name.endswith("mlipflow/plugins/dft_labeling/adapter.py") for name in names)
        assert not any(Path(name).name == ".DS_Store" for name in names)
        sdist_root = names[0].split("/", 1)[0]
        assert not any(
            Path(name).parts[1:2] in [(directory,) for directory in ("docs", "examples", "schemas", "tests")]
            for name in names
        )
        for _, _, files in agent_skill_sources():
            for path in files:
                member = archive.getmember(f"{sdist_root}/{path.relative_to(ROOT).as_posix()}")
                assert member.isfile(), member.name
                with archive.extractfile(member) as source:
                    assert source.read() == path.read_bytes()

    environment = tmp_path / "installed"
    # Reuse scientific dependencies without network access. The wheel itself must
    # resolve inside this environment, never through an existing editable install.
    run([sys.executable, "-m", "venv", "--system-site-packages", environment], cwd=tmp_path)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run(
        [python, "-m", "pip", "install", "--no-deps", "--no-index", "--ignore-installed", wheel],
        cwd=tmp_path,
    )
    # Example inputs are repository fixtures, not installation resources.
    shutil.copytree(ROOT / "examples/high_entropy_sulfide", tmp_path / "replay")
    shutil.copytree(ROOT / "examples/local_ranking", tmp_path / "local-ranking")
    script = r"""
import importlib.abc
import json
import sys
import sysconfig
from pathlib import Path

import yaml

class NoScientificImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"numpy", "torch", "ase", "pymatgen", "dpdata", "mace", "matgl", "chgnet", "deepmd"}:
            raise AssertionError("eager scientific import: " + fullname)

sys.meta_path.insert(0, NoScientificImports())
import mlipflow
from mlipflow.cli import main
from mlipflow.plugins import BUILTIN_CAPABILITIES, capability_directory, load_adapter

installed = Path(sys.prefix).resolve()
assert Path(mlipflow.__file__).resolve().is_relative_to(installed)
assert len(BUILTIN_CAPABILITIES) == 11
for name in BUILTIN_CAPABILITIES:
    assert capability_directory(name).is_relative_to(installed)
    adapter = load_adapter(name)
    assert all(callable(getattr(adapter, method)) for method in ("operation", "validate", "plan", "check", "collect"))
    adapter.operation({"parameters": {}})
data = Path(sysconfig.get_path("data")) / "share" / "mlipflow"
for name in BUILTIN_CAPABILITIES:
    skill = capability_directory(name) / "skill"
    if name == "electrochemical-voltage":
        assert not skill.exists()
        continue
    frontmatter = (skill / "SKILL.md").read_text(encoding="utf-8").split("---", 2)[1]
    skill_name = yaml.safe_load(frontmatter)["name"]
    assert (skill / "agents/openai.yaml").is_file()
    assert not (data / "agent-skills" / skill_name).exists()
assert sorted(path.name for path in data.iterdir()) == ["agent-skills"]
assert sorted(path.name for path in (data / "agent-skills").iterdir()) == ["mlip-workflow"]
workflow = data / "agent-skills/mlip-workflow"
assert (workflow / "SKILL.md").is_file()
assert (workflow / "agents/openai.yaml").is_file()
assert (workflow / "references/workflow-contract.md").is_file()
example = Path.cwd() / "replay"
for command in (["init"], ["list"], ["run", "structure-replay", "--dry-run"], ["run", "structure-replay"], ["status"]):
    assert main(["--project", str(example), *command]) == 0
records = list((example / ".mlipflow" / "runs").rglob("run-manifest.json"))
assert len(records) == 1
assert json.loads(records[0].read_text())["state"] == "OK"
ranking = Path.cwd() / "local-ranking"
for command in (["init"], ["run", "rank", "--dry-run"], ["run", "rank"], ["json", "rank"]):
    assert main(["--project", str(ranking), "--format", "json", *command]) == 0
result = json.loads((ranking / ".mlipflow/runs/rank/attempt-1/ranking-result.json").read_text())
assert [item["candidate_id"] for item in result["ranked_candidates"]] == ["b", "c"]
assert result["excluded_missing"] == ["missing"]
"""
    run([python, "-I", "-m", "mlipflow", "--version"], cwd=tmp_path)
    run([python, "-I", "-c", script], cwd=tmp_path)
