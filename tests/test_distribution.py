"""Release artifacts must work without importing the developer checkout."""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


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


def test_wheel_and_sdist_install_outside_checkout(tmp_path):
    release = tmp_path / "release"
    run([sys.executable, "-m", "build", "--no-isolation", "--outdir", release], cwd=ROOT)
    wheel = next(release.glob("*.whl"))
    sdist = next(release.glob("*.tar.gz"))
    with zipfile.ZipFile(wheel) as archive:
        assert_public_members((item.filename, item.file_size) for item in archive.infolist())
        names = archive.namelist()
        assert "mlipflow/plugins/dft_labeling/adapter.py" in names
        assert not any("share/mlipflow/plugins/" in name for name in names)
        assert not any(name.startswith(("tests/", "src/")) for name in names)
        assert not any(Path(name).name == ".DS_Store" for name in names)
        assert any(
            name.endswith("examples/high_entropy_sulfide/replay/structure-result.json")
            for name in names
        )
        assert any(name.endswith("agent-skills/mlip-workflow/SKILL.md") for name in names)
    with tarfile.open(sdist) as archive:
        assert_public_members((item.name, item.size) for item in archive.getmembers() if item.isfile())
        names = archive.getnames()
        assert any(name.endswith("mlipflow/plugins/dft_labeling/adapter.py") for name in names)
        assert not any(Path(name).name == ".DS_Store" for name in names)

    environment = tmp_path / "installed"
    # Reuse scientific dependencies without network access. The wheel itself must
    # resolve inside this environment, never through an existing editable install.
    run([sys.executable, "-m", "venv", "--system-site-packages", environment], cwd=tmp_path)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run(
        [python, "-m", "pip", "install", "--no-deps", "--no-index", "--ignore-installed", wheel],
        cwd=tmp_path,
    )
    script = r"""
import importlib.abc
import json
import shutil
import sys
import sysconfig
from pathlib import Path

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
example = Path.cwd() / "replay"
shutil.copytree(data / "examples" / "high_entropy_sulfide", example)
for command in (["init"], ["list"], ["run", "structure-replay", "--dry-run"], ["run", "structure-replay"], ["status"]):
    assert main(["--project", str(example), *command]) == 0
records = list((example / ".mlipflow" / "runs").rglob("run-manifest.json"))
assert len(records) == 1
assert json.loads(records[0].read_text())["state"] == "OK"
"""
    run([python, "-I", "-m", "mlipflow", "--version"], cwd=tmp_path)
    run([python, "-I", "-c", script], cwd=tmp_path)
