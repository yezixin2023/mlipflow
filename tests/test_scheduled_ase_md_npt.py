from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "ase-md"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _context(tmp_path: Path, calculator: str) -> dict:
    structure = tmp_path / "inputs" / "start.extxyz"
    structure.parent.mkdir(parents=True, exist_ok=True)
    structure.write_text(
        "1\nLattice=\"10 0 0 0 10 0 0 0 10\" "
        "Properties=species:S:1:pos:R:3 pbc=\"T T T\"\nLi 0 0 0\n",
        encoding="utf-8",
    )
    model_ref = tmp_path / "inputs" / f"{calculator}-model.json"
    _write_json(
        model_ref,
        {
            "schema_version": 1,
            "model_id": f"{calculator}-model-v1",
            "relative_path": f"{calculator}/model-v1"
            + ("" if calculator == "m3gnet" else ".model"),
            "kind": "directory" if calculator == "m3gnet" else "file",
        },
    )
    parameters = {
        "calculator": calculator,
        "ensemble": "npt-isotropic-mtk",
        "temperature_k": 900.0,
        "pressure_gpa": 0.0,
        "timestep_fs": 1.0,
        "thermostat_damping_fs": 100.0,
        "barostat_damping_fs": 1000.0,
        "steps": 1000,
        "trajectory_interval": 10,
        "thermo_interval": 20,
        "seed": 7,
        "device": "cpu",
        "default_dtype": "float32" if calculator == "chgnet" else "float64",
        "fix_com": False,
        "input_index": "-1",
    }
    project = {
        "schema_version": 1,
        "project": {"id": "ase-md-npt-matrix"},
        "workflow": {
            "nodes": [
                {
                    "id": "md",
                    "uses": "ase-md@0",
                    "mode": "execute",
                    "backend": "ssh-slurm",
                    "backend_profile": "cluster-a",
                    "inputs": {
                        "structure": "inputs/start.extxyz",
                        "model_reference": f"inputs/{calculator}-model.json",
                    },
                    "parameters": parameters,
                    "resources": {
                        "cpus": 8,
                        "gpus": 0,
                        "memory": "32G",
                        "walltime": "02:00:00",
                    },
                }
            ]
        },
    }
    _write_json(tmp_path / "project.yaml", project)
    return {
        "project_root": str(tmp_path),
        "attempt_dir": str(tmp_path / ".mlipflow" / "runs" / "md" / "attempt-1"),
        "backend": "ssh-slurm",
        "inputs": {
            "structure": "inputs/start.extxyz",
            "model_reference": f"inputs/{calculator}-model.json",
        },
        "parameters": parameters,
        "resources": {"cpus": 8, "gpus": 0, "memory": "32G", "walltime": "02:00:00"},
    }


@pytest.mark.parametrize("calculator", ["deepmd", "m3gnet", "chgnet", "mace"])
def test_npt_scheduler_matrix_is_ready(tmp_path: Path, calculator: str) -> None:
    module = _load("ase_md_npt_adapter", PLUGIN / "adapter_npt.py")
    plan = module.Adapter().plan(_context(tmp_path, calculator))
    assert plan["status"] == "READY", plan.get("diagnostics")
    settings = plan["md_parameters"]
    assert settings["calculator"] == calculator
    assert settings["ensemble"] == "npt-isotropic-mtk"
    assert settings["pressure_gpa"] == 0.0
    assert settings["thermostat_damping_fs"] == 100.0
    assert settings["barostat_damping_fs"] == 1000.0
    assert settings["stress_required"] is True
    assert "friction_per_fs" not in settings
    assert settings["fix_com"] is False
    assert plan["approval_summary"]["cell_mode"] == "isotropic-volume"
    staged = {
        item["remote_name"] for item in plan["scheduled_execution"]["staged_files"]
    }
    assert "adapter-legacy.py" in staged


def test_npt_rejects_langevin_friction(tmp_path: Path) -> None:
    module = _load("ase_md_npt_adapter_friction", PLUGIN / "adapter_npt.py")
    context = _context(tmp_path, "mace")
    context["parameters"]["friction_per_fs"] = 0.01
    plan = module.Adapter().plan(context)
    assert plan["status"] == "BLOCKED"
    assert any(item["code"] == "ase_md.npt_friction" for item in plan["diagnostics"])


def test_npt_rejects_fix_com_constraint(tmp_path: Path) -> None:
    module = _load("ase_md_npt_adapter_constraint", PLUGIN / "adapter_npt.py")
    context = _context(tmp_path, "deepmd")
    context["parameters"]["fix_com"] = True
    plan = module.Adapter().plan(context)
    assert plan["status"] == "BLOCKED"
    assert any(item["code"] == "ase_md.npt_constraints" for item in plan["diagnostics"])


def test_npt_requires_explicit_damping_times(tmp_path: Path) -> None:
    module = _load("ase_md_npt_adapter_damping", PLUGIN / "adapter_npt.py")
    context = _context(tmp_path, "m3gnet")
    context["parameters"].pop("barostat_damping_fs")
    plan = module.Adapter().plan(context)
    assert plan["status"] == "BLOCKED"
    assert any(item["code"] == "ase_md.barostat_damping_fs" for item in plan["diagnostics"])


def test_npt_checker_verifies_pressure_cell_schedule_and_outputs(tmp_path: Path) -> None:
    module = _load("ase_md_npt_checker", PLUGIN / "adapter_npt.py")
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    settings = {
        "calculator": "mace",
        "ensemble": "npt-isotropic-mtk",
        "model_id": "mace-v1",
        "model_path": "mace/mace-v1.model",
        "structure_path": "structure/start.extxyz",
        "temperature_k": 900.0,
        "pressure_gpa": 0.0,
        "timestep_fs": 1.0,
        "thermostat_damping_fs": 100.0,
        "barostat_damping_fs": 1000.0,
        "steps": 5,
        "trajectory_interval": 2,
        "thermo_interval": 2,
        "trajectory_steps": [0, 2, 4, 5],
        "thermo_steps": [0, 2, 4, 5],
        "seed": 7,
        "device": "cpu",
        "default_dtype": "float64",
        "fix_com": False,
        "stress_required": True,
        "thermostat_chain_length": 3,
        "barostat_chain_length": 3,
        "thermostat_substeps": 1,
        "barostat_substeps": 1,
        "input_format": None,
        "input_index": "-1",
    }
    trajectory = attempt / "trajectory.traj"
    trajectory.write_bytes(b"fake-ase-trajectory")
    final = attempt / "final.extxyz"
    final.write_text(
        "1\nLattice=\"10 0 0 0 10 0 0 0 10\" "
        "Properties=species:S:1:pos:R:3 pbc=\"T T T\"\nLi 0 0 0\n",
        encoding="utf-8",
    )
    _write_json(
        attempt / "trajectory-index.json",
        {
            "schema_version": 1,
            "steps": settings["trajectory_steps"],
            "time_fs": [0.0, 2.0, 4.0, 5.0],
        },
    )
    with (attempt / "thermo.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "step",
                "time_fs",
                "temperature_K",
                "potential_energy_eV",
                "kinetic_energy_eV",
                "total_energy_eV",
                "volume_A3",
                "pressure_GPa",
                "cell_a_A",
                "cell_b_A",
                "cell_c_A",
            ]
        )
        for step in settings["thermo_steps"]:
            writer.writerow(
                [step, float(step), 900.0, -10.0, 1.0, -9.0, 1000.0, 0.1, 10, 10, 10]
            )
    artifact_rows = []
    for role, path in (
        ("trajectory", trajectory),
        ("trajectory-index", attempt / "trajectory-index.json"),
        ("thermo", attempt / "thermo.csv"),
        ("final-structure", final),
    ):
        artifact_rows.append(
            {
                "name": role,
                "path": path.name,
            }
        )
    result = {
        "schema_version": 1,
        "plugin_id": "ase-md",
        "status": "OK",
        "calculator": "mace",
        "calculator_version": "test",
        "ase_version": "test",
        "ensemble": "npt-isotropic-mtk",
        "model": {"id": settings["model_id"], "path": settings["model_path"]},
        "structure_path": settings["structure_path"],
        "temperature_K": 900.0,
        "pressure_GPa": 0.0,
        "initial_pressure_GPa": 0.2,
        "timestep_fs": 1.0,
        "thermostat_damping_fs": 100.0,
        "barostat_damping_fs": 1000.0,
        "thermostat_chain_length": 3,
        "barostat_chain_length": 3,
        "thermostat_substeps": 1,
        "barostat_substeps": 1,
        "steps_requested": 5,
        "steps_completed": 5,
        "trajectory_interval": 2,
        "thermo_interval": 2,
        "trajectory_frames": 4,
        "thermo_records": 4,
        "seed": 7,
        "stochastic_scope": "velocity-initialization-only",
        "device": "cpu",
        "default_dtype": "float64",
        "fix_com": False,
        "artifacts": artifact_rows,
    }
    _write_json(attempt / "md-result.json", result)
    _write_json(
        attempt / "cluster-run-report.json",
        {
            "schema_version": 1,
            "status": "OK",
            "calculator": "mace",
            "ensemble": "npt-isotropic-mtk",
            "model": {
                "id": settings["model_id"],
                "path": settings["model_path"],
            },
            "structure_path": settings["structure_path"],
            "steps_completed": 5,
            "pressure_GPa": 0.0,
            "thermostat_damping_fs": 100.0,
            "barostat_damping_fs": 1000.0,
        },
    )
    context = {
        "attempt_dir": str(attempt),
        "execution": {"plan": {"md_parameters": settings}},
    }
    checked = module.Adapter().check(context)
    assert checked["status"] == "OK", checked.get("diagnostics")
    assert checked["metrics"]["final_pressure_GPa"] == 0.1
    assert checked["metrics"]["final_volume_A3"] == 1000.0
