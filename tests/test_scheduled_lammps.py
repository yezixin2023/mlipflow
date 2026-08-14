from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "lammps-md"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _launcher(framework: str, target: str) -> dict:
    prefix: list[str] = []
    if target == "gpu" and framework in {"mace", "m3gnet"}:
        prefix = ["-k", "on", "g", "1", "-sf", "kk"]
    return {
        "target": target,
        "executable": "site-owned-lammps",
        "argv_after_executable": [
            *prefix,
            "-in",
            f"in.{target}.lammps",
            "-var",
            "MODEL_FILE",
            "<site-resolved-model-file>",
        ],
        "required_packages": [framework],
        "limitations": [],
    }


def _prepared(tmp_path: Path, framework: str, targets: list[str] | None = None) -> Path:
    targets = targets or ["cpu", "gpu"]
    prepared = tmp_path / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    structure = prepared / "structure.data"
    structure.write_text("LAMMPS data file\n\n1 atoms\n1 atom types\n", encoding="utf-8")
    steps = 1000
    marker = f"MLIPFLOW_LAMMPS_COMPLETED step={steps}"
    generated = [
        {"name": "structure.data", "sha256": _sha(structure), "size_bytes": structure.stat().st_size}
    ]
    launchers = []
    for target in targets:
        deck = prepared / f"in.{target}.lammps"
        pair = {
            "deepmd": "pair_style deepmd ${MODEL_FILE}\npair_coeff * * Li\n",
            "mace": "pair_style mace\npair_coeff * * ${MODEL_FILE} Li\n",
            "m3gnet": "pair_style matgl\npair_coeff * * ${MODEL_FILE} Li\n",
        }[framework]
        if target == "gpu" and framework == "mace":
            pair = "pair_style mace no_domain_decomposition\npair_coeff * * ${MODEL_FILE} Li\n"
        if target == "gpu" and framework == "m3gnet":
            pair = "pair_style matgl/kk\npair_coeff * * ${MODEL_FILE} Li\n"
        deck.write_text(
            "units metal\nread_data structure.data\n"
            + pair
            + f"run {steps}\nwrite_data final.data\nwrite_restart final.restart\nprint \"{marker}\"\n",
            encoding="utf-8",
        )
        generated.append({"name": deck.name, "sha256": _sha(deck), "size_bytes": deck.stat().st_size})
        launchers.append(_launcher(framework, target))
    formats = {
        "deepmd": "deepmd-lammps-model",
        "mace": "mace-lammps-torchscript",
        "m3gnet": "matgl-lammps-torchscript",
    }
    manifest = {
        "schema_version": 1,
        "plugin_id": "lammps-md",
        "operation": "lammps-prepare",
        "status": "OK",
        "preparation_contract": "lammps-md-input-v2",
        "runtime_model_variable": "MODEL_FILE",
        "completion_marker": marker,
        "model": {
            "model_id": f"{framework}-lmp-v1",
            "framework": framework,
            "relative_path": f"{framework}/model.bin",
            "kind": "file",
            "fingerprint": "sha256:" + "2" * 64,
            "artifact_format": formats[framework],
            "elements": ["Li"],
        },
        "md": {
            "ensemble": "nvt",
            "targets": targets,
            "type_map": ["Li"],
            "temperature_k": 900.0,
            "timestep_fs": 1.0,
            "steps": steps,
            "thermo_interval": 10,
            "dump_interval": 10,
            "seed": 7,
            "thermostat_damping_fs": 100.0,
        },
        "generated_files": generated,
        "launchers": launchers,
    }
    manifest_path = prepared / "lammps-input-manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def _context(tmp_path: Path, framework: str, target: str, gpus: int | None = None) -> dict:
    manifest = _prepared(tmp_path, framework)
    if gpus is None:
        gpus = 0 if target == "cpu" else 1
    parameters = {
        "operation": "execute",
        "target": target,
        "input_manifest_fingerprint": _sha(manifest),
    }
    node = {
        "id": "lammps-run",
        "uses": "lammps-md@0",
        "mode": "execute",
        "backend": "ssh-slurm",
        "backend_profile": "cluster-a",
        "inputs": {"lammps_input_manifest": "prepared/lammps-input-manifest.json"},
        "parameters": parameters,
        "resources": {"cpus": 8, "gpus": gpus, "memory": "32G", "walltime": "02:00:00"},
    }
    _write_json(
        tmp_path / "project.yaml",
        {"schema_version": 1, "project": {"id": "lammps-test"}, "workflow": {"nodes": [node]}},
    )
    return {
        "project_root": str(tmp_path),
        "attempt_dir": str(tmp_path / ".mlipflow" / "runs" / "lammps-run" / "attempt-1"),
        "backend": "ssh-slurm",
        "inputs": node["inputs"],
        "parameters": parameters,
        "resources": node["resources"],
    }


@pytest.mark.parametrize(
    ("framework", "target"),
    [
        ("deepmd", "cpu"),
        ("deepmd", "gpu"),
        ("mace", "cpu"),
        ("mace", "gpu"),
        ("m3gnet", "cpu"),
        ("m3gnet", "gpu"),
    ],
)
def test_execute_plan_matrix(tmp_path: Path, framework: str, target: str) -> None:
    module = _load(f"lammps_execute_{framework}_{target}", PLUGIN / "adapter_execute.py")
    plan = module.Adapter().plan(_context(tmp_path, framework, target))
    assert plan["status"] == "READY", plan.get("diagnostics")
    assert plan["scheduled_execution"]["template_family"] == f"lammps-{framework}-{target}"
    identity = plan["lammps_execution_identity"]
    assert identity["framework"] == framework
    assert identity["target"] == target
    assert identity["steps"] == 1000
    staged = {item["remote_name"] for item in plan["scheduled_execution"]["staged_files"]}
    assert {
        "project.yaml",
        "lammps-input-manifest.json",
        "lammps/structure.data",
        f"lammps/in.{target}.lammps",
        "lammps_cluster.py",
    } <= staged
    salvage = set(plan["failure_salvage"]["fetch_remote_names"])
    assert "cluster-run-report.json" in salvage
    assert "trajectory.lammpstrj" not in salvage


def test_cpu_rejects_gpu_allocation(tmp_path: Path) -> None:
    module = _load("lammps_cpu_resource", PLUGIN / "adapter_execute.py")
    plan = module.Adapter().plan(_context(tmp_path, "deepmd", "cpu", gpus=1))
    assert plan["status"] == "BLOCKED"
    assert any(item["code"] == "lammps.cpu_gpus" for item in plan["diagnostics"])


@pytest.mark.parametrize("framework", ["mace", "m3gnet"])
def test_single_gpu_frameworks_reject_two_gpus(tmp_path: Path, framework: str) -> None:
    module = _load(f"lammps_gpu_bound_{framework}", PLUGIN / "adapter_execute.py")
    plan = module.Adapter().plan(_context(tmp_path, framework, "gpu", gpus=2))
    assert plan["status"] == "BLOCKED"
    assert any(item["code"] == "lammps.single_gpu" for item in plan["diagnostics"])


def test_changed_prepared_deck_blocks_execution(tmp_path: Path) -> None:
    module = _load("lammps_changed_deck", PLUGIN / "adapter_execute.py")
    context = _context(tmp_path, "mace", "gpu")
    (tmp_path / "prepared" / "in.gpu.lammps").write_text("tampered\n", encoding="utf-8")
    plan = module.Adapter().plan(context)
    assert plan["status"] == "BLOCKED"
    assert any(item["code"].startswith("lammps.prepared_") for item in plan["diagnostics"])


def test_prepare_plan_uses_execution_ready_wrapper(tmp_path: Path) -> None:
    module = _load("lammps_prepare_v2_adapter", PLUGIN / "adapter_execute.py")
    structure = tmp_path / "start.extxyz"
    structure.write_text(
        '1\nLattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3 pbc="T T T"\nLi 0 0 0\n',
        encoding="utf-8",
    )
    model = tmp_path / "model.json"
    _write_json(
        model,
        {
            "schema_version": 1,
            "model_id": "dp",
            "framework": "deepmd",
            "relative_path": "deepmd/model.pb",
            "kind": "file",
            "fingerprint": "sha256:" + "2" * 64,
            "artifact_format": "deepmd-lammps-model",
            "elements": ["Li"],
        },
    )
    config = tmp_path / "config.json"
    _write_json(
        config,
        {
            "schema_version": 1,
            "engine": "lammps",
            "ensemble": "nvt",
            "targets": ["cpu"],
            "type_map": ["Li"],
            "temperature_k": 900.0,
            "timestep_fs": 1.0,
            "steps": 100,
            "thermo_interval": 10,
            "dump_interval": 10,
            "seed": 7,
            "thermostat_damping_fs": 100.0,
        },
    )
    context = {
        "project_root": str(tmp_path),
        "attempt_dir": str(tmp_path / "attempt"),
        "backend": "local",
        "inputs": {"structure": "start.extxyz", "model_reference": "model.json", "lammps_config": "config.json"},
        "parameters": {"operation": "lammps-prepare", "output_dir": "lammps-inputs"},
        "resources": {},
    }
    plan = module.Adapter().plan(context)
    assert plan["status"] == "READY", plan.get("diagnostics")
    assert Path(plan["argv"][1]).name == "lammps_prepare_v2.py"
    assert plan["approval_summary"]["preparation_contract"] == "lammps-md-input-v2"


def test_execution_checker_rebinds_outputs(tmp_path: Path) -> None:
    module = _load("lammps_checker", PLUGIN / "adapter_execute.py")
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    marker = "MLIPFLOW_LAMMPS_COMPLETED step=1000"
    identity = {
        "framework": "mace",
        "target": "gpu",
        "input_manifest_fingerprint": "sha256:" + "1" * 64,
        "model_id": "mace-lmp-v1",
        "model_fingerprint": "sha256:" + "2" * 64,
        "artifact_format": "mace-lammps-torchscript",
        "ensemble": "nvt",
        "temperature_k": 900.0,
        "timestep_fs": 1.0,
        "steps": 1000,
        "type_map": ["Li"],
        "thermo_interval": 10,
        "dump_interval": 10,
        "seed": 7,
        "completion_marker": marker,
        "prepared_launcher": _launcher("mace", "gpu"),
    }
    files = {
        "trajectory.lammpstrj": b"ITEM: TIMESTEP\n1000\n",
        "final.data": b"final data\n",
        "final.restart": b"binary-restart",
        "lammps.log": ("LAMMPS (4 Jul 2026)\n" + marker + "\n").encode(),
        "lammps.screen.log": b"LAMMPS (4 Jul 2026)\n",
    }
    artifacts = []
    for name, payload in files.items():
        path = attempt / name
        path.write_bytes(payload)
        artifacts.append({"name": name, "path": name, "sha256": _sha(path), "size_bytes": path.stat().st_size})
    result = {
        "schema_version": 1,
        "plugin_id": "lammps-md",
        "status": "OK",
        "operation": "execute",
        "framework": "mace",
        "target": "gpu",
        "lammps_version": "4 Jul 2026",
        "input_manifest_fingerprint": identity["input_manifest_fingerprint"],
        "model": {"id": identity["model_id"], "fingerprint": identity["model_fingerprint"], "artifact_format": identity["artifact_format"]},
        "ensemble": "nvt",
        "steps_requested": 1000,
        "steps_completed": 1000,
        "completion_marker": marker,
        "launcher": {"site_launcher_used": True, "prepared_argv_after_executable": identity["prepared_launcher"]["argv_after_executable"], "required_packages": ["mace"]},
        "artifacts": artifacts,
    }
    _write_json(attempt / "lammps-execution-result.json", result)
    _write_json(
        attempt / "cluster-run-report.json",
        {
            "schema_version": 1,
            "status": "OK",
            "framework": "mace",
            "target": "gpu",
            "model_id": identity["model_id"],
            "model_fingerprint": identity["model_fingerprint"],
            "input_manifest_fingerprint": identity["input_manifest_fingerprint"],
            "steps_completed": 1000,
            "lammps_version": "4 Jul 2026",
            "result_sha256": _sha(attempt / "lammps-execution-result.json"),
        },
    )
    context = {
        "attempt_dir": str(attempt),
        "parameters": {"operation": "execute"},
        "execution": {"plan": {"lammps_execution_identity": identity}},
    }
    checked = module.Adapter().check(context)
    assert checked["status"] == "OK", checked.get("diagnostics")
    assert checked["metrics"]["steps_completed"] == 1000.0
