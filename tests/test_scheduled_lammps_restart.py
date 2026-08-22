from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "lammps-md"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prepared(root: Path, framework: str = "mace", target: str = "gpu") -> Path:
    prepared = root / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    structure = prepared / "structure.data"
    structure.write_text("LAMMPS data\n\n1 atoms\n1 atom types\n", encoding="utf-8")
    steps = 1000
    marker = f"MLIPFLOW_LAMMPS_COMPLETED step={steps}"
    pair_style = {
        "deepmd": "pair_style      deepmd ${MODEL_FILE}",
        "mace": "pair_style      mace no_domain_decomposition" if target == "gpu" else "pair_style      mace",
        "m3gnet": "pair_style      matgl/kk" if target == "gpu" else "pair_style      matgl",
    }[framework]
    pair_coeff = (
        "pair_coeff      * * Li"
        if framework == "deepmd"
        else "pair_coeff      * * ${MODEL_FILE} Li"
    )
    deck = prepared / f"in.{target}.lammps"
    deck.write_text(
        "# generated\n"
        "units           metal\n"
        "atom_style      atomic\n"
        "atom_modify     map yes\n"
        "newton          on\n"
        "boundary        p p p\n"
        "read_data       structure.data\n"
        f"{pair_style}\n"
        f"{pair_coeff}\n"
        "timestep        0.001\n"
        "thermo          10\n"
        "thermo_style    custom step time temp pe ke etotal press vol lx ly lz\n"
        "velocity        all create 900 7 mom yes rot yes dist gaussian\n"
        "dump            mlipflow all custom 10 trajectory.lammpstrj id type element x y z vx vy vz\n"
        "dump_modify     mlipflow element Li sort id\n"
        "fix             mlipflow all nvt temp 900 900 0.1\n"
        f"run             {steps}\n"
        "write_data      final.data\n"
        "write_restart   final.restart\n"
        f'print           "{marker}"\n',
        encoding="utf-8",
    )
    launcher_prefix = ["-k", "on", "g", "1", "-sf", "kk"] if target == "gpu" and framework in {"mace", "m3gnet"} else []
    artifact_format = {
        "deepmd": "deepmd-lammps-model",
        "mace": "mace-lammps-torchscript",
        "m3gnet": "matgl-lammps-torchscript",
    }[framework]
    manifest = {
        "schema_version": 1,
        "plugin_id": "lammps-md",
        "operation": "lammps-prepare",
        "status": "OK",
        "preparation_contract": "lammps-md-input-v2",
        "runtime_model_variable": "MODEL_FILE",
        "completion_marker": marker,
        "model": {
            "model_id": f"{framework}-v1",
            "framework": framework,
            "relative_path": f"{framework}/model.pt",
            "kind": "file",
            "artifact_format": artifact_format,
            "elements": ["Li"],
        },
        "md": {
            "ensemble": "nvt",
            "targets": [target],
            "type_map": ["Li"],
            "temperature_k": 900.0,
            "timestep_fs": 1.0,
            "steps": steps,
            "thermo_interval": 10,
            "dump_interval": 10,
            "seed": 7,
            "thermostat_damping_fs": 100.0,
        },
        "generated_files": [
            {"name": "structure.data"},
            {"name": deck.name},
        ],
        "launchers": [
            {
                "target": target,
                "executable": "site-owned-lammps",
                "argv_after_executable": [
                    *launcher_prefix,
                    "-in",
                    deck.name,
                    "-var",
                    "MODEL_FILE",
                    "<site-resolved-model-file>",
                ],
                "required_packages": [framework],
                "limitations": [],
            }
        ],
    }
    path = prepared / "lammps-input-manifest.json"
    _write_json(path, manifest)
    return path


def _context(root: Path, attempt: int = 1, policy: str = "auto-from-previous-attempt") -> dict:
    _prepared(root)
    parameters = {
        "operation": "execute",
        "target": "gpu",
        "restart_policy": policy,
        "checkpoint_interval": 100,
    }
    resources = {"cpus": 8, "gpus": 1, "memory": "32G", "walltime": "01:00:00"}
    node = {
        "id": "lammps-run",
        "uses": "lammps-md@0",
        "mode": "execute",
        "backend": "ssh-slurm",
        "backend_profile": "cluster-a",
        "inputs": {"lammps_input_manifest": "prepared/lammps-input-manifest.json"},
        "parameters": parameters,
        "resources": resources,
    }
    _write_json(
        root / "project.yaml",
        {"schema_version": 1, "project": {"id": "lammps-restart"}, "workflow": {"nodes": [node]}},
    )
    return {
        "project_root": str(root),
        "attempt_dir": str(root / ".mlipflow" / "runs" / "lammps-run" / f"attempt-{attempt}"),
        "backend": "ssh-slurm",
        "inputs": node["inputs"],
        "parameters": parameters,
        "resources": resources,
    }


def _previous_runtime(context: dict, scheduler_state: str = "TIMEOUT", checkpoint: bool = True) -> None:
    current = Path(context["attempt_dir"])
    previous = current.parent / "attempt-1"
    _write_json(
        previous / "run-manifest.final.json",
        {"state": "FAIL", "job": {"scheduler_state": scheduler_state}},
    )
    prepared = json.loads(
        (
            Path(context["project_root"])
            / "prepared"
            / "lammps-input-manifest.json"
        ).read_text()
    )
    _write_json(
        previous / "restart-runtime.json",
        {
            "schema_version": 1,
            "runtime_contract": "lammps-restart-runtime-v1",
            "attempt": 1,
            "framework": "mace",
            "target": "gpu",
            "input_manifest_path": "lammps-input-manifest.json",
            "model_path": prepared["model"]["relative_path"],
            "model_kind": prepared["model"]["kind"],
            "lammps_interface": None,
            "checkpoint_interval": 100,
            "resources": context["resources"],
            "lammps_executable": "/site/lmp",
            "launcher_prefix": ["srun"],
            "prepared_launcher": prepared["launchers"][0]["argv_after_executable"],
            "platform": {"system": "Linux", "machine": "x86_64", "byteorder": "little"},
        },
    )
    if checkpoint:
        (previous / "checkpoint.1.restart").write_bytes(b"fake-lammps-restart-100")
        (previous / "checkpoint.2.restart").write_bytes(b"fake-lammps-restart-200")


def test_attempt_one_declares_periodic_failure_salvage(tmp_path: Path) -> None:
    module = _load("lammps_restart_adapter_first", PLUGIN / "adapter_restart.py")
    plan = module.Adapter().plan(_context(tmp_path))
    assert plan["status"] == "READY", plan.get("diagnostics")
    calculation = plan["lammps_calculation"]
    assert calculation["restart_policy"] == "auto-from-previous-attempt"
    assert calculation["checkpoint_interval"] == 100
    assert calculation["restart_from_attempt"] is None
    salvage = set(plan["failure_salvage"]["fetch_remote_names"])
    assert {"checkpoint.1.restart", "checkpoint.2.restart", "restart-runtime.json"} <= salvage
    staged = {item["remote_name"] for item in plan["scheduled_execution"]["staged_files"]}
    assert {"lammps_cluster_restart.py", "lammps_restart.py", "lammps_cluster.py"} <= staged


def test_attempt_two_binds_immediately_previous_salvaged_candidates(tmp_path: Path) -> None:
    module = _load("lammps_restart_adapter_second", PLUGIN / "adapter_restart.py")
    context = _context(tmp_path, attempt=2)
    _previous_runtime(context, "TIMEOUT")
    plan = module.Adapter().plan(context)
    assert plan["status"] == "READY", plan.get("diagnostics")
    calculation = plan["lammps_calculation"]
    assert calculation["restart_from_attempt"] == 1
    assert set(calculation["restart_candidates"]) == {"checkpoint.1.restart", "checkpoint.2.restart"}
    staged = {item["remote_name"] for item in plan["scheduled_execution"]["staged_files"]}
    assert {
        "restart/restart-runtime.json",
        "restart/checkpoint.1.restart",
        "restart/checkpoint.2.restart",
    } <= staged
    assert plan["approval_summary"]["restart_start_step_resolved_on_compute_node"] is True


@pytest.mark.parametrize("scheduler_state", ["TIMEOUT", "PREEMPTED", "NODE_FAIL", "OUT_OF_MEMORY"])
def test_scheduler_terminal_states_are_restart_eligible(tmp_path: Path, scheduler_state: str) -> None:
    module = _load(f"lammps_restart_state_{scheduler_state}", PLUGIN / "adapter_restart.py")
    context = _context(tmp_path, attempt=2)
    _previous_runtime(context, scheduler_state)
    plan = module.Adapter().plan(context)
    assert plan["status"] == "READY", plan.get("diagnostics")


def test_scientific_fail_after_scheduler_completed_is_not_auto_resumed(tmp_path: Path) -> None:
    module = _load("lammps_restart_science_fail", PLUGIN / "adapter_restart.py")
    context = _context(tmp_path, attempt=2)
    _previous_runtime(context, "COMPLETED")
    plan = module.Adapter().plan(context)
    assert plan["status"] == "BLOCKED"
    assert any(item["code"] == "lammps.restart_previous" for item in plan["diagnostics"])


def test_missing_salvaged_checkpoint_blocks_retry(tmp_path: Path) -> None:
    module = _load("lammps_restart_missing", PLUGIN / "adapter_restart.py")
    context = _context(tmp_path, attempt=2)
    _previous_runtime(context, "TIMEOUT", checkpoint=False)
    plan = module.Adapter().plan(context)
    assert plan["status"] == "BLOCKED"
    assert any("no salvaged periodic restart" in item["message"] for item in plan["diagnostics"])


def test_auto_restart_requires_checkpoint_interval(tmp_path: Path) -> None:
    module = _load("lammps_restart_interval", PLUGIN / "adapter_restart.py")
    context = _context(tmp_path)
    del context["parameters"]["checkpoint_interval"]
    project = json.loads((tmp_path / "project.yaml").read_text())
    del project["workflow"]["nodes"][0]["parameters"]["checkpoint_interval"]
    _write_json(tmp_path / "project.yaml", project)
    plan = module.Adapter().plan(context)
    assert plan["status"] == "BLOCKED"
    assert any(item["code"] == "lammps.restart_checkpoint_interval" for item in plan["diagnostics"])


def test_restart_deck_preserves_fix_and_does_not_reinitialize_velocity() -> None:
    helper = _load("lammps_restart_helper", PLUGIN / "lammps_restart.py")
    fresh = (
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
        "thermo_style    custom step temp pe\n"
        "velocity        all create 900 7 mom yes rot yes dist gaussian\n"
        "dump            mlipflow all custom 10 trajectory.lammpstrj id type x y z\n"
        "dump_modify     mlipflow sort id\n"
        "fix             mlipflow all nvt temp 900 900 0.1\n"
        "run             1000\n"
        "write_data      final.data\n"
        "write_restart   final.restart\n"
        'print           "MLIPFLOW_LAMMPS_COMPLETED step=1000"\n'
    )
    instrumented = helper.instrument_fresh(fresh, 100, 1000)
    assert "restart         100 checkpoint.1.restart checkpoint.2.restart" in instrumented
    resume = helper.build_resume(fresh, 100, 1000)
    assert "read_restart    ${RESTART_FILE}" in resume
    assert "pair_coeff      * * ${MODEL_FILE} Li" in resume
    assert "fix             mlipflow all nvt temp 900 900 0.1" in resume
    assert "velocity" not in resume
    assert "run             1000 upto" in resume


def test_restart_step_uses_restart2info_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(PLUGIN))
    module = _load("lammps_restart_probe_current", PLUGIN / "lammps_cluster_restart.py")
    executable = tmp_path / "lmp"
    executable.write_text("binary", encoding="utf-8")
    candidate = tmp_path / "checkpoint.1.restart"
    candidate.write_bytes(b"restart")

    def run(argv, **kwargs):
        assert argv[-2:] == ["-restart2info", str(candidate)]
        assert "input" not in kwargs
        return SimpleNamespace(
            returncode=0,
            stdout=b"Current timestep number = 900\n",
            stderr=b"",
        )

    monkeypatch.setattr(module.subprocess, "run", run)
    assert module._restart_step(executable, [], candidate) == 900


def test_restart_step_falls_back_to_read_restart_for_older_lammps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(PLUGIN))
    module = _load("lammps_restart_probe_legacy", PLUGIN / "lammps_cluster_restart.py")
    executable = tmp_path / "lmp"
    executable.write_text("binary", encoding="utf-8")
    candidate = tmp_path / "checkpoint.2.restart"
    candidate.write_bytes(b"restart")
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if len(calls) == 1:
            return SimpleNamespace(
                returncode=1,
                stdout=b"",
                stderr=b"Invalid command-line argument: -restart2info\n",
            )
        return SimpleNamespace(
            returncode=0,
            stdout=b"MLIPFLOW_RESTART_STEP=800\n",
            stderr=b"",
        )

    monkeypatch.setattr(module.subprocess, "run", run)
    assert module._restart_step(executable, ["-sf", "kk"], candidate) == 800
    assert calls[1][0] == [str(executable), "-sf", "kk", "-log", "none"]
    assert f'read_restart "{candidate}"' in calls[1][1]["input"].decode("utf-8")
