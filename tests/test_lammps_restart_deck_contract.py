from __future__ import annotations

from tests.helpers import load_module
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "mlipflow" / "plugins" / "lammps_md" / "lammps_restart.py"


def _load():
    module = load_module(HELPER, 'lammps_restart_deck_contract')
    return module


def _fresh(fix_line: str) -> str:
    return (
        "units           metal\n"
        "atom_style      atomic\n"
        "atom_modify     map yes\n"
        "newton          on\n"
        "boundary        p p p\n"
        "read_data       structure.data\n"
        "pair_style      mace\n"
        "pair_coeff      * * ${MODEL_FILE} Li\n"
        "timestep        0.001\n"
        "thermo          10\n"
        "thermo_style    custom step temp pe press vol\n"
        "velocity        all create 900 7 mom yes rot yes dist gaussian\n"
        "dump            mlipflow all custom 10 trajectory.lammpstrj id type x y z\n"
        "dump_modify     mlipflow sort id\n"
        f"{fix_line}\n"
        "run             1000\n"
        "write_data      final.data\n"
        "write_restart   final.restart\n"
        'print           "MLIPFLOW_LAMMPS_COMPLETED step=1000"\n'
    )


@pytest.mark.parametrize(
    "fix_line",
    [
        "fix             mlipflow all nvt temp 900 900 0.1",
        "fix             mlipflow all npt temp 900 900 0.1 iso 0 0 1",
    ],
)
def test_resume_preserves_nose_hoover_fix_without_fresh_initialization(fix_line: str) -> None:
    helper = _load()
    resume = helper.build_resume(_fresh(fix_line), 100, 1000)
    assert "read_restart    ${RESTART_FILE}" in resume
    assert fix_line in resume
    assert "velocity" not in resume
    assert "atom_modify" not in resume
    assert "run             1000 upto" in resume
    assert "restart         100 checkpoint.1.restart checkpoint.2.restart" in resume
