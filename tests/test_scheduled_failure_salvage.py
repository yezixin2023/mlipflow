from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

from mlipflow.backends import ExecutionResult
from mlipflow.config import load_project
from mlipflow.services import advance, initialize, make_advance_plan, make_run_plan, run_node

from .helpers import project_config, write_json
from .test_scheduled_dft import FakeTemplateLibrary, write_site


ASE_MD_RUN_TEMPLATE = """#!/bin/bash
# ASE MD test template
# inputs={{INPUT_DIR}}
# outputs={{OUTPUT_DIR}}
cd {{RUN_DIR}}
"""


def _library() -> FakeTemplateLibrary:
    templates = dict(FakeTemplateLibrary().templates)
    templates["ase-md-mace/run.sh"] = ASE_MD_RUN_TEMPLATE
    return FakeTemplateLibrary(templates)


def _build_project(root: Path) -> Path:
    structure = root / "inputs" / "start.extxyz"
    structure.parent.mkdir(parents=True, exist_ok=True)
    structure.write_text(
        "1\n"
        'Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
        "Li 0 0 0\n",
        encoding="utf-8",
    )
    write_json(
        root / "inputs" / "mace-model.json",
        {
            "schema_version": 1,
            "model_id": "mace-model-v1",
            "relative_path": "mace/model-v1.model",
            "kind": "file",
        },
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
        "parameters": {
            "calculator": "mace",
            "ensemble": "nvt-langevin",
            "temperature_k": 900.0,
            "timestep_fs": 1.0,
            "steps": 1000,
            "trajectory_interval": 100,
            "thermo_interval": 100,
            "checkpoint_interval": 50,
            "restart_policy": "auto-from-previous-attempt",
            "seed": 7,
            "device": "cpu",
            "default_dtype": "float64",
            "friction_per_fs": 0.01,
            "fix_com": True,
            "input_index": "-1",
        },
        "resources": {"cpus": 8, "gpus": 0, "memory": "32G", "walltime": "00:05:00"},
    }
    write_json(root / "project.yaml", project_config([node]))
    return write_site(root)


def test_timeout_checkpoint_is_fetched_only_through_approved_failure_salvage(
    tmp_path: Path,
) -> None:
    site = _build_project(tmp_path)
    initialize(tmp_path)
    project = load_project(tmp_path)
    library = _library()
    remote = tmp_path / "fake-remote"

    run_plan = make_run_plan(project, "md", site, library)
    assert run_plan["adapter_plan"]["status"] == "READY"

    def stage(_remote_dir, _files):
        return _remote_dir

    with patch(
        "mlipflow.backends.SshSlurmBackend.stage_workspace", side_effect=stage
    ), patch(
        "mlipflow.backends.SshSlurmBackend.submit",
        return_value=ExecutionResult(0, "Submitted batch job 91\n", "", "91"),
    ):
        run_node(
            project,
            "md",
            True,
            site,
            library,
        )

    output = remote / "output"
    output.mkdir(parents=True)
    checkpoint = output / "md-checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "plugin_id": "ase-md",
                "checkpoint_state_version": "ase-md-checkpoint-v1",
                "completed_steps": 400,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    def inspect(_self, _cwd, remote_path):
        path = remote / remote_path
        if not path.is_file():
            return {"path": remote_path, "exists": False}
        return {
            "path": remote_path,
            "exists": True,
            "size_bytes": path.stat().st_size,
        }

    def fetch(_self, _cwd, remote_path, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote / remote_path, destination)
        return destination

    timeout = {"state": "TIMEOUT", "detail": "walltime", "source": "fake"}
    with patch(
        "mlipflow.backends.SshSlurmBackend.status", return_value=timeout
    ), patch(
        "mlipflow.backends.SshSlurmBackend.inspect_file",
        autospec=True,
        side_effect=inspect,
    ):
        salvage_plan = make_advance_plan(project)

    transitions = salvage_plan["details"]["transitions"]
    assert len(transitions) == 1
    change = transitions[0]
    assert change["action"] == "adapter-finalize"
    assert change["scheduler_state"] == "TIMEOUT"
    finalization = change["adapter_finalization"]
    assert finalization["failure_salvage"] is True
    assert finalization["terminal_target"] == "FAIL"
    inventory = {item["remote_name"]: item for item in finalization["outputs"]}
    assert inventory["md-checkpoint.json"]["exists"] is True
    assert not (
        tmp_path / ".mlipflow" / "runs" / "md" / "attempt-1" / "md-checkpoint.json"
    ).exists()

    with patch(
        "mlipflow.backends.SshSlurmBackend.status", return_value=timeout
    ), patch(
        "mlipflow.backends.SshSlurmBackend.inspect_file",
        autospec=True,
        side_effect=inspect,
    ), patch(
        "mlipflow.backends.SshSlurmBackend.fetch_from",
        autospec=True,
        side_effect=fetch,
    ):
        outcome = advance(project)

    assert outcome["changed"][0]["state"] == "FAIL"
    local_checkpoint = (
        tmp_path / ".mlipflow" / "runs" / "md" / "attempt-1" / "md-checkpoint.json"
    )
    assert local_checkpoint.is_file()
    assert local_checkpoint.read_bytes() == checkpoint.read_bytes()
