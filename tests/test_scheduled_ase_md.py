from __future__ import annotations

import csv
import hashlib
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


def _sha(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return "sha256:" + digest


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _context(tmp_path: Path, calculator: str) -> dict:
    structure = tmp_path / "inputs" / "start.extxyz"
    structure.parent.mkdir(parents=True, exist_ok=True)
    structure.write_text("1\nLattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3 pbc=\"T T T\"\nLi 0 0 0\n", encoding="utf-8")
    model_fp = "sha256:" + "2" * 64
    model_ref = tmp_path / "inputs" / f"{calculator}-model.json"
    _write_json(
        model_ref,
        {
            "schema_version": 1,
            "model_id": f"{calculator}-model-v1",
            "relative_path": f"{calculator}/model-v1" + ("" if calculator == "m3gnet" else ".model"),
            "kind": "directory" if calculator == "m3gnet" else "file",
            "fingerprint": model_fp,
        },
    )
    parameters = {
        "calculator": calculator,
        "ensemble": "nvt-langevin",
        "temperature_k": 900.0,
        "timestep_fs": 1.0,
        "steps": 1000,
        "trajectory_interval": 10,
        "thermo_interval": 20,
        "seed": 7,
        "device": "cpu",
        "default_dtype": "float32" if calculator == "chgnet" else "float64",
        "friction_per_fs": 0.01,
        "fix_com": True,
        "model_fingerprint": model_fp,
        "structure_fingerprint": _sha(structure),
        "input_index": "-1",
    }
    project = {
        "schema_version": 1,
        "project": {"id": "ase-md-matrix"},
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
                    "resources": {"cpus": 8, "gpus": 0, "memory": "32G", "walltime": "02:00:00"},
                }
            ]
        },
    }
    _write_json(tmp_path / "project.yaml", project)
    return {
        "project_root": str(tmp_path),
        "attempt_dir": str(tmp_path / ".mlipflow" / "runs" / "md" / "attempt-1"),
        "backend": "ssh-slurm",
        "inputs": {"structure": "inputs/start.extxyz", "model_reference": f"inputs/{calculator}-model.json"},
        "parameters": parameters,
        "resources": {"cpus": 8, "gpus": 0, "memory": "32G", "walltime": "02:00:00"},
    }


@pytest.mark.parametrize("calculator", ["deepmd", "m3gnet", "chgnet", "mace"])
def test_scheduler_matrix_is_ready(tmp_path: Path, calculator: str) -> None:
    module = _load("ase_md_adapter", PLUGIN / "adapter.py")
    plan = module.Adapter().plan(_context(tmp_path, calculator))
    assert plan["status"] == "READY", plan.get("diagnostics")
    assert plan["md_identity"]["calculator"] == calculator
    scheduled = plan["scheduled_execution"]
    assert scheduled["schema_version"] == 2
    assert scheduled["template_family"] == f"ase-md-{calculator}"
    staged = {item["remote_name"] for item in scheduled["staged_files"]}
    assert {"project.yaml", "model-reference.json", "ase_md.py", "ase_md_cluster.py", "structure/start.extxyz"} <= staged
    outputs = {item["remote_name"]: item for item in scheduled["fetch_outputs"]}
    for name in ("md-result.json", "trajectory.traj", "trajectory-index.json", "thermo.csv", "final.extxyz", "cluster-run-report.json"):
        assert outputs[name]["required"] is True


def test_cuda_requires_scheduled_gpu(tmp_path: Path) -> None:
    module = _load("ase_md_adapter_cuda", PLUGIN / "adapter.py")
    context = _context(tmp_path, "mace")
    context["parameters"]["device"] = "cuda"
    plan = module.Adapter().plan(context)
    assert plan["status"] == "BLOCKED"
    assert any(item["code"] == "ase_md.cuda_resource" for item in plan["diagnostics"])


def test_chgnet_requires_float32(tmp_path: Path) -> None:
    module = _load("ase_md_adapter_chgnet", PLUGIN / "adapter.py")
    context = _context(tmp_path, "chgnet")
    context["parameters"]["default_dtype"] = "float64"
    plan = module.Adapter().plan(context)
    assert plan["status"] == "BLOCKED"
    assert any(item["code"] == "ase_md.chgnet_dtype" for item in plan["diagnostics"])


def test_m3gnet_reference_must_be_directory(tmp_path: Path) -> None:
    module = _load("ase_md_adapter_kind", PLUGIN / "adapter.py")
    context = _context(tmp_path, "m3gnet")
    reference = tmp_path / "inputs" / "m3gnet-model.json"
    raw = json.loads(reference.read_text(encoding="utf-8"))
    raw["kind"] = "file"
    _write_json(reference, raw)
    plan = module.Adapter().plan(context)
    assert plan["status"] == "BLOCKED"
    assert any(item["code"] == "ase_md.model_reference" for item in plan["diagnostics"])


def test_cluster_fingerprint_is_stable_for_file_and_tree(tmp_path: Path) -> None:
    cluster = _load("ase_md_cluster", PLUGIN / "ase_md_cluster.py")
    model = tmp_path / "model.model"
    model.write_bytes(b"model")
    assert cluster.fingerprint(model) == _sha(model)
    tree = tmp_path / "matgl-model"
    tree.mkdir()
    (tree / "model.json").write_text("{}\n", encoding="utf-8")
    (tree / "model.pt").write_bytes(b"weights")
    first = cluster.fingerprint(tree)
    second = cluster.fingerprint(tree)
    assert first == second
    assert first.startswith("sha256:") and len(first) == 71


def test_checker_verifies_schedule_and_hashes(tmp_path: Path) -> None:
    module = _load("ase_md_checker", PLUGIN / "adapter.py")
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    identity = {
        "calculator": "mace",
        "ensemble": "nvt-langevin",
        "model_id": "mace-v1",
        "model_fingerprint": "sha256:" + "2" * 64,
        "structure_fingerprint": "sha256:" + "3" * 64,
        "temperature_k": 900.0,
        "timestep_fs": 1.0,
        "steps": 5,
        "trajectory_interval": 2,
        "thermo_interval": 2,
        "trajectory_steps": [0, 2, 4, 5],
        "thermo_steps": [0, 2, 4, 5],
        "seed": 7,
        "device": "cpu",
        "default_dtype": "float64",
        "friction_per_fs": 0.01,
        "fix_com": True,
        "input_format": None,
        "input_index": "-1",
    }
    trajectory = attempt / "trajectory.traj"
    trajectory.write_bytes(b"fake-ase-trajectory")
    final = attempt / "final.extxyz"
    final.write_text("1\nProperties=species:S:1:pos:R:3\nLi 0 0 0\n", encoding="utf-8")
    _write_json(
        attempt / "trajectory-index.json",
        {"schema_version": 1, "steps": identity["trajectory_steps"], "time_fs": [0.0, 2.0, 4.0, 5.0]},
    )
    with (attempt / "thermo.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "time_fs", "temperature_K", "potential_energy_eV", "kinetic_energy_eV", "total_energy_eV", "volume_A3"])
        for step in identity["thermo_steps"]:
            writer.writerow([step, float(step), 900.0, -10.0, 1.0, -9.0, 100.0])
    artifact_rows = []
    for role, path in (
        ("trajectory", trajectory),
        ("trajectory-index", attempt / "trajectory-index.json"),
        ("thermo", attempt / "thermo.csv"),
        ("final-structure", final),
    ):
        artifact_rows.append({"name": role, "path": path.name, "sha256": _sha(path), "size_bytes": path.stat().st_size})
    result = {
        "schema_version": 1,
        "plugin_id": "ase-md",
        "status": "OK",
        "calculator": "mace",
        "calculator_version": "test",
        "ase_version": "test",
        "ensemble": "nvt-langevin",
        "model": {"id": identity["model_id"], "fingerprint": identity["model_fingerprint"]},
        "structure_fingerprint": identity["structure_fingerprint"],
        "temperature_K": 900.0,
        "timestep_fs": 1.0,
        "steps_requested": 5,
        "steps_completed": 5,
        "trajectory_interval": 2,
        "thermo_interval": 2,
        "trajectory_frames": 4,
        "thermo_records": 4,
        "seed": 7,
        "device": "cpu",
        "default_dtype": "float64",
        "friction_per_fs": 0.01,
        "fix_com": True,
        "artifacts": artifact_rows,
    }
    _write_json(attempt / "md-result.json", result)
    _write_json(
        attempt / "cluster-run-report.json",
        {
            "schema_version": 1,
            "status": "OK",
            "calculator": "mace",
            "ensemble": "nvt-langevin",
            "model": {"id": identity["model_id"], "observed_fingerprint": identity["model_fingerprint"]},
            "structure_fingerprint": identity["structure_fingerprint"],
            "result_sha256": _sha(attempt / "md-result.json"),
        },
    )
    context = {"attempt_dir": str(attempt), "execution": {"plan": {"md_identity": identity}}}
    checked = module.Adapter().check(context)
    assert checked["status"] == "OK", checked.get("diagnostics")
    assert checked["metrics"]["steps_completed"] == 5.0
