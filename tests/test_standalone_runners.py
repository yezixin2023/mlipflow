"""Scientific runner bundles must not require the controller package."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

from mlipflow.plugins import capability_directory


# These are the small Python dependencies distributed with the corresponding
# runners. Existing scheduled-plan tests verify staging and scientific execution.
BUNDLES = [
    (
        "mlip-benchmark",
        "benchmark_wrapper.py",
        ["benchmark_wrapper.py", "benchmark_normalization.py"],
        True,
    ),
    (
        "mlip-benchmark",
        "fresh_benchmark.py",
        ["fresh_benchmark.py", "benchmark_wrapper.py", "benchmark_normalization.py"],
        True,
    ),
    ("ase-md", "ase_md_cluster.py", ["ase_md_cluster.py", "ase_md.py"], False),
    ("lammps-md", "lammps_prepare.py", ["lammps_prepare.py"], False),
    (
        "lammps-md",
        "lammps_cluster_restart.py",
        ["lammps_cluster_restart.py", "lammps_cluster.py", "lammps_restart.py"],
        False,
    ),
    (
        "pes-sampling",
        "lasp_cluster.py",
        ["lasp_cluster.py", "lasp_ssw.py", "lasp_input_arc.py"],
        False,
    ),
    ("pes-sampling", "structure_to_lasp.py", ["structure_to_lasp.py", "lasp_input_arc.py"], False),
    ("dft-labeling", "dataset_convert.py", ["dataset_convert.py", "dataset_contract.py"], False),
    ("active-learning", "active_learning.py", ["active_learning.py", "science.py"], False),
    (
        "ionic-transport",
        "manuscript.py",
        ["manuscript.py"],
        False,
    ),
    (
        "electrochemical-voltage",
        "adapter.py",
        ["adapter.py", "science.py", "manuscript_replay.py"],
        False,
    ),
]


@pytest.mark.parametrize(
    "capability,entry,files,shared_runtime",
    BUNDLES,
    ids=[f"{cap}:{entry}" for cap, entry, _, _ in BUNDLES],
)
@pytest.mark.parametrize("staged", [True, False], ids=["uploaded-files", "source-script"])
def test_runner_imports_without_controller(
    tmp_path, capability, entry, files, shared_runtime, staged
):
    source = capability_directory(capability)
    if staged:
        directory = tmp_path / "input"
        directory.mkdir()
        for name in files:
            shutil.copy2(source / name, directory / name)
        if shared_runtime:
            shutil.copy2(source.parent / "model_runtime.py", directory / "model_runtime.py")
    else:
        directory = source
    script = r"""
import importlib.abc
import runpy
import sys
from pathlib import Path

class NoController(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlipflow" or fullname.startswith("mlipflow."):
            raise AssertionError("standalone runner imported controller: " + fullname)

sys.meta_path.insert(0, NoController())
entry = Path(sys.argv[1])
sys.path.insert(0, str(entry.parent))
sys.argv = [str(entry), "--help"]
runpy.run_path(str(entry), run_name="__main__")
"""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(directory / entry)],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()
