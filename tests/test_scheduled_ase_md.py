from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path

import pytest
from ase import Atoms
from ase.constraints import FixCom
from ase.io import read, write

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
    structure.write_text("1\nLattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3 pbc=\"T T T\"\nLi 0 0 0\n", encoding="utf-8")
    model_ref = tmp_path / "inputs" / f"{calculator}-model.json"
    _write_json(
        model_ref,
        {
            "schema_version": 1,
            "model_id": f"{calculator}-model-v1",
            "relative_path": f"{calculator}/model-v1" + ("" if calculator == "m3gnet" else ".model"),
            "kind": "directory" if calculator == "m3gnet" else "file",
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
        "input_index": "-1",
    }
    project = {
        "schema_version": 1,
        "project": {"id": "ase-md-matrix"},
        "workflow": {
            "nodes": [
                {
                    "id": "md",
                    "uses": "ase-md",
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
    assert plan["md_parameters"]["calculator"] == calculator
    scheduled = plan["scheduled_execution"]
    assert scheduled["schema_version"] == 3
    assert scheduled["execution_model"] == "single-python"
    expected_family = (
        f"ase-md-{calculator}-canonical"
        if calculator in {"m3gnet", "chgnet"}
        else f"ase-md-{calculator}"
    )
    assert scheduled["template_family"] == expected_family
    staged = {item["remote_name"] for item in scheduled["staged_files"]}
    assert {"project.yaml", "model-reference.json", "ase_md.py", "ase_md_cluster.py", "structure/start.extxyz"} <= staged
    outputs = {item["remote_name"]: item for item in scheduled["fetch_outputs"]}
    for name in ("md-result.json", "trajectory.traj", "trajectory-index.json", "thermo.csv", "final.extxyz", "cluster-run-report.json"):
        assert outputs[name]["required"] is True


@pytest.mark.parametrize("path_kind", ["project-relative", "parent-relative", "absolute"])
def test_structure_path_runs_adapter_staging_and_cluster_chain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path_kind: str
) -> None:
    from mlipflow import backends as backend_module
    from mlipflow.backends import SshSlurmBackend
    from mlipflow.config import load_project
    from mlipflow.services.contracts import _scheduled_contract

    project_root = tmp_path / "project"
    project_root.mkdir()
    context = _context(project_root, "mace")
    local_source = project_root / "inputs" / "start.extxyz"
    if path_kind == "project-relative":
        source = local_source
        value = "inputs/start.extxyz"
    else:
        source = tmp_path / ("shared" if path_kind == "parent-relative" else "external") / "A.extxyz"
        source.parent.mkdir()
        source.write_text(local_source.read_text(encoding="utf-8"), encoding="utf-8")
        value = "../shared/A.extxyz" if path_kind == "parent-relative" else str(source)
    context["inputs"]["structure"] = value
    project_data = json.loads((project_root / "project.yaml").read_text(encoding="utf-8"))
    project_data["workflow"]["nodes"][0]["inputs"]["structure"] = value
    _write_json(project_root / "project.yaml", project_data)

    adapter = _load(f"ase_md_adapter_structure_{path_kind}", PLUGIN / "adapter.py").Adapter()
    plan = adapter.plan(context)

    assert plan["status"] == "READY", plan.get("diagnostics")
    assert plan["input_paths"]["structure"] == str(source.resolve())
    project = load_project(project_root)
    contract = _scheduled_contract(project, "ase-md", {"adapter_plan": plan})
    staged_structure = next(
        item for item in contract["staged_files"] if item["remote_name"].startswith("structure/")
    )
    assert staged_structure == {
        "source": str(source.resolve()),
        "remote_name": f"structure/{source.name}",
    }

    remote_run_dir = "/work/test/ase-md/attempt-0001"
    remote_workspace = tmp_path / "remote" / "attempt-0001"

    def fake_transfer(argv, **_kwargs):
        if argv[0] == "ssh":
            for name in ("input", "output", "logs"):
                (remote_workspace / name).mkdir(parents=True, exist_ok=True)
        elif argv[0] == "scp":
            remote_path = argv[-1].split(":", 1)[1]
            relative = remote_path.removeprefix(remote_run_dir + "/")
            destination = remote_workspace / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(argv[-2], destination)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(backend_module.subprocess, "run", fake_transfer)
    files = [
        (Path(item["source"]), f"input/{item['remote_name']}")
        for item in contract["staged_files"]
    ]
    SshSlurmBackend("cluster-a").stage_workspace(remote_run_dir, files)

    cluster = _load(f"ase_md_cluster_structure_{path_kind}", PLUGIN / "ase_md_cluster.py")
    model_root = tmp_path / "models"
    model = model_root / "mace" / "model-v1.model"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    observed: dict[str, object] = {}

    class FakeRunner:
        @staticmethod
        def run_md(**kwargs):
            atoms = read(kwargs["structure"], format="extxyz")
            observed["structure"] = Path(kwargs["structure"])
            observed["symbols"] = atoms.get_chemical_symbols()
            return {
                "schema_version": 1,
                "status": "OK",
                "model": {
                    "id": kwargs["model_id"],
                    "path": kwargs["model_record_path"],
                },
                "artifacts": [],
                "steps_completed": 0,
            }

    monkeypatch.setattr(cluster, "_load_runner", lambda _: FakeRunner)
    code = cluster.run(
        types.SimpleNamespace(
            input_dir=str(remote_workspace / "input"),
            output_dir=str(remote_workspace / "output"),
            project=str(remote_workspace / "input" / "project.yaml"),
            node_id="md",
            model_root=str(model_root),
        )
    )

    assert code == 0
    assert observed == {
        "structure": remote_workspace / "input" / "structure" / source.name,
        "symbols": ["Li"],
    }
    report = json.loads(
        (remote_workspace / "output" / "cluster-run-report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "OK"
    assert report["structure_path"] == f"structure/{source.name}"


@pytest.mark.parametrize("path_kind", ["parent-relative", "absolute"])
def test_missing_structure_path_is_blocked(tmp_path: Path, path_kind: str) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    context = _context(project_root, "mace")
    missing = tmp_path / "shared" / "missing.extxyz"
    context["inputs"]["structure"] = (
        "../shared/missing.extxyz" if path_kind == "parent-relative" else str(missing)
    )
    adapter = _load(f"ase_md_adapter_missing_{path_kind}", PLUGIN / "adapter.py").Adapter()

    plan = adapter.plan(context)

    assert plan["status"] == "BLOCKED"
    messages = [item["message"] for item in plan["diagnostics"]]
    assert any("structure file does not exist" in message for message in messages)
    assert str(missing.resolve()) in "\n".join(messages)


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


def test_upstream_model_reference_path_is_authoritative(tmp_path: Path) -> None:
    module = _load("ase_md_adapter_upstream_binding", PLUGIN / "adapter_restart.py")
    context = _context(tmp_path, "m3gnet")
    reference = tmp_path / "inputs" / "m3gnet-model.json"
    context["inputs"]["model_reference"] = str(reference)

    plan = module.Adapter().plan(context)

    assert plan["status"] == "READY", plan.get("diagnostics")
    assert plan["md_parameters"]["model_id"] == "m3gnet-model-v1"
    assert plan["md_parameters"]["model_path"] == "m3gnet/model-v1"
    assert plan["scheduled_execution"]["template_family"] == "ase-md-m3gnet-canonical"


def test_cluster_report_uses_stable_model_path_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load("ase_md_cluster_model_record", PLUGIN / "ase_md_cluster.py")
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    model_root = tmp_path / "models"
    (input_dir / "structure").mkdir(parents=True)
    output_dir.mkdir()
    model_root.mkdir()
    (input_dir / "structure" / "start.extxyz").write_text("structure\n", encoding="utf-8")
    (model_root / "mace.model").write_bytes(b"model")
    _write_json(
        input_dir / "model-reference.json",
        {
            "schema_version": 1,
            "model_id": "mace-v1",
            "relative_path": "mace.model",
            "kind": "file",
        },
    )
    _write_json(
        tmp_path / "project.yaml",
        {
            "schema_version": 1,
            "project": {"id": "ase-md-cluster-report"},
            "workflow": {
                "nodes": [
                    {
                        "id": "md",
                        "uses": "ase-md",
                        "backend": "ssh-slurm",
                        "inputs": {"structure": "inputs/start.extxyz"},
                        "parameters": {
                            "calculator": "mace",
                            "ensemble": "nvt-langevin",
                        },
                    }
                ]
            },
        },
    )

    class FakeRunner:
        @staticmethod
        def run_md(**kwargs):
            return {
                "schema_version": 1,
                "status": "OK",
                "model": {
                    "id": kwargs["model_id"],
                    "path": kwargs["model_record_path"],
                },
                "artifacts": [],
                "steps_completed": 0,
            }

    monkeypatch.setattr(module, "_load_runner", lambda _: FakeRunner)
    code = module.run(
        types.SimpleNamespace(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            project=str(tmp_path / "project.yaml"),
            node_id="md",
            model_root=str(model_root),
        )
    )

    assert code == 0
    report = json.loads((output_dir / "cluster-run-report.json").read_text())
    assert report["model"] == {"id": "mace-v1", "path": "mace.model"}


def test_runner_accepts_scheduler_precreated_empty_output_root(tmp_path: Path) -> None:
    runner = _load("ase_md_runner_output_contract", PLUGIN / "ase_md.py")
    output = tmp_path / "attempt-0001" / "output"
    output.mkdir(parents=True)

    claimed = runner._prepare_output_directory(output)

    assert claimed == output.resolve()
    assert claimed.is_dir()

    standalone_output = tmp_path / "standalone-output"
    standalone_claimed = runner._prepare_output_directory(standalone_output)
    assert standalone_claimed == standalone_output.resolve()
    assert standalone_claimed.is_dir()


def test_runner_output_root_still_has_no_overwrite_semantics(tmp_path: Path) -> None:
    runner = _load("ase_md_runner_no_overwrite", PLUGIN / "ase_md.py")
    output = tmp_path / "attempt-0001" / "output"
    output.mkdir(parents=True)
    (output / "thermo.csv").write_text("step\n0\n", encoding="utf-8")

    with pytest.raises(runner.AseMDError, match="not empty"):
        runner._prepare_output_directory(output)

    ordinary_file = tmp_path / "output-file"
    ordinary_file.write_text("occupied", encoding="utf-8")
    with pytest.raises(runner.AseMDError, match="not a directory"):
        runner._prepare_output_directory(ordinary_file)

    empty_target = tmp_path / "empty-target"
    empty_target.mkdir()
    symlink = tmp_path / "output-link"
    symlink.symlink_to(empty_target, target_is_directory=True)
    with pytest.raises(runner.AseMDError, match="must not be a symlink"):
        runner._prepare_output_directory(symlink)


def test_final_structure_copy_drops_runtime_attachments() -> None:
    runner = _load("ase_md_runner_final_structure", PLUGIN / "ase_md.py")
    calculator = object()
    input_constraints = [object()]
    runtime_constraints = [*input_constraints, object()]

    class FakeAtoms:
        def __init__(self, calc, constraints):
            self.calc = calc
            self.constraints = constraints

        def copy(self):
            return FakeAtoms(self.calc, self.constraints)

        def set_constraint(self, constraints):
            self.constraints = constraints

    atoms = FakeAtoms(calculator, runtime_constraints)
    final_atoms = runner._final_structure_copy(atoms, input_constraints)

    assert final_atoms is not atoms
    assert atoms.calc is calculator
    assert atoms.constraints == runtime_constraints
    assert final_atoms.calc is None
    assert final_atoms.constraints == input_constraints


def test_final_structure_copy_avoids_fixcom_extxyz_failure(tmp_path: Path) -> None:
    runner = _load("ase_md_runner_final_structure_extxyz", PLUGIN / "ase_md.py")
    atoms = Atoms("Li2", positions=[[0, 0, 0], [1, 0, 0]])
    atoms.set_constraint(FixCom())

    final_atoms = runner._final_structure_copy(atoms, [])
    output = tmp_path / "final.extxyz"
    write(
        output,
        final_atoms,
        format="extxyz",
        write_results=False,
    )

    assert final_atoms.constraints == []
    restored = read(output, format="extxyz")
    assert isinstance(restored, Atoms)
    assert restored.get_chemical_symbols() == ["Li", "Li"]


def test_callback_step_zero_is_recorded_once() -> None:
    runner = _load("ase_md_runner_callback_step", PLUGIN / "ase_md.py")

    assert runner._is_new_callback_step([], 0) is True
    assert runner._is_new_callback_step([0], 0) is False
    assert runner._is_new_callback_step([0], 1) is True


def test_explicit_sampling_supercell_enforces_strict_cell_bound(tmp_path: Path) -> None:
    runner = _load("ase_md_runner_supercell", PLUGIN / "ase_md.py")
    atoms = Atoms(
        "Li",
        positions=[[0.0, 0.0, 0.0]],
        cell=[8.94053, 11.3388017, 11.18310136],
        pbc=True,
    )

    expanded, repeat, source_count, lengths, minimum = runner._prepare_structure(
        atoms, [2, 1, 1], 10.0
    )

    assert repeat == (2, 1, 1)
    assert source_count == 1
    assert len(expanded) == 2
    assert lengths == pytest.approx([17.88106, 11.3388017, 11.18310136])
    assert minimum == 10.0

    with pytest.raises(runner.AseMDError, match="strict minimum"):
        runner._prepare_structure(atoms, [1, 1, 1], 10.0)


def test_plan_binds_explicit_supercell_and_cell_bound(tmp_path: Path) -> None:
    module = _load("ase_md_adapter_supercell", PLUGIN / "adapter.py")
    context = _context(tmp_path, "mace")
    context["parameters"].update(
        {
            "supercell_repeat": [2, 1, 1],
            "minimum_initial_cell_length_angstrom": 10.0,
        }
    )

    plan = module.Adapter().plan(context)

    assert plan["status"] == "READY", plan.get("diagnostics")
    assert plan["md_parameters"]["supercell_repeat"] == [2, 1, 1]
    assert plan["md_parameters"]["minimum_initial_cell_length_angstrom"] == 10.0
    assert plan["approval_summary"]["supercell_repeat"] == [2, 1, 1]
    assert plan["approval_summary"]["minimum_initial_cell_length_A"] == 10.0


def test_mace_calculator_uses_supported_model_paths_keyword(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load("ase_md_runner_mace_keyword", PLUGIN / "ase_md.py")
    observed = {}

    class FakeMACECalculator:
        def __init__(self, **kwargs):
            observed.update(kwargs)

    package = types.ModuleType("mace")
    calculators = types.ModuleType("mace.calculators")
    calculators.MACECalculator = FakeMACECalculator
    package.calculators = calculators
    monkeypatch.setitem(sys.modules, "mace", package)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)

    model = tmp_path / "MACE.model"
    runner._build_calculator("mace", model, "cpu", "float64")

    assert observed == {
        "model_paths": str(model),
        "device": "cpu",
        "default_dtype": "float64",
    }


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


def test_checker_verifies_schedule_and_scientific_outputs(tmp_path: Path) -> None:
    module = _load("ase_md_checker", PLUGIN / "adapter.py")
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    settings = {
        "calculator": "mace",
        "ensemble": "nvt-langevin",
        "model_id": "mace-v1",
        "model_path": "mace/mace-v1.model",
        "structure_path": "structure/start.extxyz",
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
        "supercell_repeat": [2, 1, 1],
        "minimum_initial_cell_length_angstrom": 10.0,
    }
    trajectory = attempt / "trajectory.traj"
    trajectory.write_bytes(b"fake-ase-trajectory")
    final = attempt / "final.extxyz"
    final.write_text("1\nProperties=species:S:1:pos:R:3\nLi 0 0 0\n", encoding="utf-8")
    _write_json(
        attempt / "trajectory-index.json",
        {"schema_version": 1, "steps": settings["trajectory_steps"], "time_fs": [0.0, 2.0, 4.0, 5.0]},
    )
    with (attempt / "thermo.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "time_fs", "temperature_K", "potential_energy_eV", "kinetic_energy_eV", "total_energy_eV", "volume_A3"])
        for step in settings["thermo_steps"]:
            writer.writerow([step, float(step), 900.0, -10.0, 1.0, -9.0, 100.0])
    artifact_rows = []
    for role, path in (
        ("trajectory", trajectory),
        ("trajectory-index", attempt / "trajectory-index.json"),
        ("thermo", attempt / "thermo.csv"),
        ("final-structure", final),
    ):
        artifact_rows.append({"name": role, "path": path.name})
    result = {
        "schema_version": 1,
        "plugin_id": "ase-md",
        "status": "OK",
        "calculator": "mace",
        "calculator_version": "test",
        "ase_version": "test",
        "ensemble": "nvt-langevin",
        "model": {"id": settings["model_id"], "path": settings["model_path"]},
        "structure_path": settings["structure_path"],
        "supercell_repeat": [2, 1, 1],
        "source_atom_count": 2,
        "atom_count": 4,
        "initial_cell_lengths_A": [20.0, 11.0, 12.0],
        "minimum_initial_cell_length_A": 10.0,
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
        "observed_stability": {
            "all_recorded_values_finite": True,
            "minimum_pair_distance_A": 1.8,
            "minimum_pair_distance_step": 2,
            "sampled_trajectory_frames": 4,
            "thermodynamics": {
                "temperature_K": {"minimum": 850.0, "maximum": 950.0, "mean": 900.0},
                "potential_energy_eV_per_atom": {"minimum": -2.6, "maximum": -2.4, "mean": -2.5},
                "total_energy_eV_per_atom": {"minimum": -2.4, "maximum": -2.1, "mean": -2.25},
                "volume_A3": {"minimum": 2640.0, "maximum": 2640.0, "mean": 2640.0},
            },
        },
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
            "model": {"id": settings["model_id"], "path": settings["model_path"]},
            "structure_path": settings["structure_path"],
            "supercell_repeat": result["supercell_repeat"],
            "source_atom_count": result["source_atom_count"],
            "atom_count": result["atom_count"],
            "initial_cell_lengths_A": result["initial_cell_lengths_A"],
            "minimum_initial_cell_length_A": result[
                "minimum_initial_cell_length_A"
            ],
            "observed_stability": result["observed_stability"],
        },
    )
    context = {
        "attempt_dir": str(attempt),
        "execution": {"plan": {"md_parameters": settings}},
    }
    checked = module.Adapter().check(context)
    assert checked["status"] == "OK", checked.get("diagnostics")
    assert checked["metrics"]["steps_completed"] == 5.0
