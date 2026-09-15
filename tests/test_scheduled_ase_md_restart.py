from __future__ import annotations

from copy import deepcopy
from tests.helpers import load_module
import json
from pathlib import Path

import pytest
from unittest.mock import patch

from mlipflow.plugins.ase_md import common, npt, nvt
from mlipflow.plugins.ase_md.adapter import Adapter

from mlipflow.errors import ConfigError
from mlipflow.services.scheduled import _failure_salvage_outputs

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "mlipflow" / "plugins" / "ase_md"


@pytest.mark.parametrize("ensemble", ["nvt-langevin", "npt-isotropic-mtk"])
def test_planning_validates_only_the_selected_ensemble_once(tmp_path, ensemble):
    context = _context(tmp_path, ensemble=ensemble)
    before = deepcopy(context)
    selected, other = (npt, nvt) if ensemble == npt.NPT else (nvt, npt)
    with patch.object(selected, "validate", wraps=selected.validate) as validate, patch.object(
        common, "validate", wraps=common.validate
    ) as common_validate, patch.object(
        other, "validate", side_effect=AssertionError("wrong ensemble validation")
    ), patch.object(other, "plan", side_effect=AssertionError("wrong ensemble plan")), patch.object(
        common, "build_plan", wraps=common.build_plan
    ) as build:
        plan = Adapter().plan(context)
    assert plan["status"] == "READY", plan["diagnostics"]
    validate.assert_called_once_with(context)
    common_validate.assert_called_once_with(context)
    build.assert_called_once_with(context, ensemble)
    assert context == before
    assert ("friction_per_fs" in plan["md_parameters"]) is (ensemble == nvt.NVT)


def _load(name: str, path: Path):
    module = load_module(path, name)
    return module


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _context(tmp_path: Path, *, ensemble: str = "nvt-langevin", attempt: int = 1) -> dict:
    structure = tmp_path / "inputs" / "start.extxyz"
    structure.parent.mkdir(parents=True, exist_ok=True)
    structure.write_text(
        "1\n"
        'Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
        "Li 0 0 0\n",
        encoding="utf-8",
    )
    model_ref = tmp_path / "inputs" / "mace-model.json"
    _write_json(
        model_ref,
        {
            "schema_version": 1,
            "model_id": "mace-model-v1",
            "relative_path": "mace/model-v1.model",
            "kind": "file",
        },
    )
    parameters = {
        "calculator": "mace",
        "ensemble": ensemble,
        "temperature_k": 900.0,
        "timestep_fs": 1.0,
        "steps": 1000,
        "trajectory_interval": 100,
        "thermo_interval": 200,
        "checkpoint_interval": 50,
        "restart_policy": "auto-from-previous-attempt",
        "seed": 7,
        "device": "cpu",
        "default_dtype": "float64",
        "fix_com": ensemble == "nvt-langevin",
        "input_index": "-1",
    }
    if ensemble == "nvt-langevin":
        parameters["friction_per_fs"] = 0.01
    else:
        parameters.update(
            {
                "pressure_gpa": 0.0,
                "thermostat_damping_fs": 100.0,
                "barostat_damping_fs": 1000.0,
            }
        )
    node = {
        "id": "md",
        "uses": "ase-md",
        "mode": "execute",
        "backend": "ssh-slurm",
        "backend_profile": "cluster-a",
        "inputs": {
            "structure": "inputs/start.extxyz",
            "model_reference": "inputs/mace-model.json",
        },
        "parameters": parameters,
        "resources": {"cpus": 8, "gpus": 0, "memory": "32G", "walltime": "02:00:00"},
    }
    _write_json(
        tmp_path / "project.yaml",
        {
            "schema_version": 1,
            "project": {"id": "ase-md-restart"},
            "workflow": {"nodes": [node]},
        },
    )
    return {
        "project_root": str(tmp_path),
        "attempt_dir": str(
            tmp_path / ".mlipflow" / "runs" / "md" / f"attempt-{attempt}"
        ),
        "backend": "ssh-slurm",
        "inputs": node["inputs"],
        "parameters": parameters,
        "resources": node["resources"],
    }


def _checkpoint(settings: dict, *, completed_steps: int, ase_version: str = "test-ase") -> dict:
    value = {
        "schema_version": 1,
        "plugin_id": "ase-md",
        "checkpoint_state_version": "ase-md-checkpoint-v1",
        "calculator": settings["calculator"],
        "ensemble": settings["ensemble"],
        "model": {
            "id": settings["model_id"],
            "path": settings["model_path"],
        },
        "structure_path": settings["structure_path"],
        "temperature_K": settings["temperature_k"],
        "timestep_fs": settings["timestep_fs"],
        "steps_requested": settings["steps"],
        "seed": settings["seed"],
        "device": settings["device"],
        "default_dtype": settings["default_dtype"],
        "fix_com": settings["fix_com"],
        "ase_version": ase_version,
        "completed_steps": completed_steps,
    }
    if settings["ensemble"] == "nvt-langevin":
        value["friction_per_fs"] = settings["friction_per_fs"]
        value["rng_algorithm"] = "PCG64"
    else:
        value.update(
            {
                "pressure_GPa": settings["pressure_gpa"],
                "thermostat_damping_fs": settings["thermostat_damping_fs"],
                "barostat_damping_fs": settings["barostat_damping_fs"],
                "thermostat_chain_length": settings["thermostat_chain_length"],
                "barostat_chain_length": settings["barostat_chain_length"],
                "thermostat_substeps": settings["thermostat_substeps"],
                "barostat_substeps": settings["barostat_substeps"],
            }
        )
    return value


@pytest.mark.parametrize("ensemble", ["nvt-langevin", "npt-isotropic-mtk"])
def test_first_attempt_declares_checkpoint_and_failure_salvage(
    tmp_path: Path, ensemble: str
) -> None:
    module = _load("ase_md_restart_adapter_first", PLUGIN / "adapter.py")
    plan = module.Adapter().plan(_context(tmp_path, ensemble=ensemble, attempt=1))
    assert plan["status"] == "READY", plan.get("diagnostics")
    assert plan["md_parameters"]["segment_start_step"] == 0
    assert plan["md_parameters"]["checkpoint_interval"] == 50
    assert plan["md_parameters"]["restart_from_attempt"] is None
    assert plan["failure_salvage"] == {
        "schema_version": 1,
        "fetch_remote_names": [
            "md-checkpoint.json",
            "trajectory.traj",
            "trajectory-index.json",
            "thermo.csv",
        ],
    }
    outputs = {
        item["remote_name"]: item
        for item in plan["scheduled_execution"]["fetch_outputs"]
    }
    assert outputs["md-checkpoint.json"]["required"] is True
    staged = {
        item["remote_name"] for item in plan["scheduled_execution"]["staged_files"]
    }
    assert "restart/md-checkpoint.json" not in staged
    assert {"ase_md.py", "ase_md_cluster.py"} <= staged
    assert not any(name.startswith("adapter-") for name in staged)


@pytest.mark.parametrize("ensemble", ["nvt-langevin", "npt-isotropic-mtk"])
def test_retry_stages_immediately_previous_salvaged_checkpoint(
    tmp_path: Path, ensemble: str
) -> None:
    module = _load("ase_md_restart_adapter_retry", PLUGIN / "adapter.py")
    first_context = _context(tmp_path, ensemble=ensemble, attempt=1)
    first = module.Adapter().plan(first_context)
    assert first["status"] == "READY", first.get("diagnostics")
    settings = first["md_parameters"]
    previous_dir = tmp_path / ".mlipflow" / "runs" / "md" / "attempt-1"
    checkpoint_path = previous_dir / "md-checkpoint.json"
    _write_json(checkpoint_path, _checkpoint(settings, completed_steps=400))
    _write_json(
        previous_dir / "run-manifest.final.json",
        {
            "state": "FAIL",
            "job": {"scheduler_state": "TIMEOUT"},
        },
    )

    retry = module.Adapter().plan(_context(tmp_path, ensemble=ensemble, attempt=2))
    assert retry["status"] == "READY", retry.get("diagnostics")
    restart_settings = retry["md_parameters"]
    assert restart_settings["segment_start_step"] == 400
    assert restart_settings["restart_from_attempt"] == 1
    assert restart_settings["restart_checkpoint_path"] == "restart/md-checkpoint.json"
    assert restart_settings["trajectory_steps"] == [400, 500, 600, 700, 800, 900, 1000]
    assert restart_settings["thermo_steps"] == [400, 600, 800, 1000]
    assert retry["approval_summary"]["remaining_steps"] == 600
    assert retry["input_paths"]["restart_checkpoint"] == str(checkpoint_path)
    staged = {
        item["remote_name"]: item
        for item in retry["scheduled_execution"]["staged_files"]
    }
    assert "restart/md-checkpoint.json" in staged
    assert staged["restart/md-checkpoint.json"]["source"] == str(checkpoint_path)


def test_retry_does_not_resume_scientific_fail_after_completed_scheduler(tmp_path: Path) -> None:
    module = _load("ase_md_restart_adapter_completed_fail", PLUGIN / "adapter.py")
    first = module.Adapter().plan(_context(tmp_path, attempt=1))
    assert first["status"] == "READY"
    previous_dir = tmp_path / ".mlipflow" / "runs" / "md" / "attempt-1"
    _write_json(
        previous_dir / "md-checkpoint.json",
        _checkpoint(first["md_parameters"], completed_steps=400),
    )
    _write_json(
        previous_dir / "run-manifest.final.json",
        {"state": "FAIL", "job": {"scheduler_state": "COMPLETED"}},
    )
    retry = module.Adapter().plan(_context(tmp_path, attempt=2))
    assert retry["status"] == "BLOCKED"
    assert any(item["code"] == "ase_md.restart_previous" for item in retry["diagnostics"])


def test_retry_requires_locally_salvaged_checkpoint(tmp_path: Path) -> None:
    module = _load("ase_md_restart_adapter_no_checkpoint", PLUGIN / "adapter.py")
    _context(tmp_path, attempt=1)
    previous_dir = tmp_path / ".mlipflow" / "runs" / "md" / "attempt-1"
    _write_json(
        previous_dir / "run-manifest.final.json",
        {"state": "FAIL", "job": {"scheduler_state": "PREEMPTED"}},
    )
    retry = module.Adapter().plan(_context(tmp_path, attempt=2))
    assert retry["status"] == "BLOCKED"
    assert any(item["code"] == "ase_md.restart_previous" for item in retry["diagnostics"])


def test_failure_salvage_is_only_a_subset_of_normal_fetch_allowlist() -> None:
    scheduled = {
        "fetch_outputs": [
            {
                "remote_name": "md-checkpoint.json",
                "remote_path": "output/md-checkpoint.json",
                "local_name": "md-checkpoint.json",
                "required": True,
                "role": "md-checkpoint",
            },
            {
                "remote_name": "thermo.csv",
                "remote_path": "output/thermo.csv",
                "local_name": "thermo.csv",
                "required": True,
                "role": "thermodynamics",
            },
        ]
    }
    plan = {
        "adapter_plan": {
            "failure_salvage": {
                "schema_version": 1,
                "fetch_remote_names": ["md-checkpoint.json"],
            }
        }
    }
    outputs = _failure_salvage_outputs(plan, scheduled)
    assert len(outputs) == 1
    assert outputs[0]["remote_name"] == "md-checkpoint.json"
    assert outputs[0]["required"] is False

    plan["adapter_plan"]["failure_salvage"]["fetch_remote_names"] = ["../../secret"]
    with pytest.raises(ConfigError):
        _failure_salvage_outputs(plan, scheduled)
