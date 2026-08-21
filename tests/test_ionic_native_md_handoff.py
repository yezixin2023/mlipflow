from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "plugins" / "ionic-transport" / "ionic_conductivity.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner():
    return _load(RUNNER_PATH, "ionic_native_handoff_runner")


def _args(**overrides):
    values = {
        "source": "trajectory",
        "specie": "Li",
        "trajectory_start_ps": None,
        "trajectory_end_ps": None,
        "diffusion_analyzer_step_skip": 1,
        "diffusion_analyzer_smoothed": "none",
        "diffusion_analyzer_min_obs": 1,
        "diffusion_analyzer_avg_nsteps": 2,
        "ase_frame_step_fs": None,
        "lammps_timestep_ps": 0.001,
        "lammps_data_name": "data.LYC",
        "mobile_type": None,
        "temperature_K": None,
        "vasp_file_name": "vasprun.xml",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _capture_analyzer(module):
    captured = {}

    def fake(structures, specie, temperature_K, time_step_fs, args, *, continuous_frac=None):
        structures = list(structures)
        captured.update(
            structures=structures,
            specie=specie,
            temperature_K=temperature_K,
            time_step_fs=time_step_fs,
            continuous_frac=np.asarray(continuous_frac, dtype=float),
        )
        analyzer = SimpleNamespace(
            msd=np.arange(len(structures), dtype=float),
            dt=np.arange(len(structures), dtype=float) * time_step_fs,
        )
        return analyzer, analyzer.dt / 1000.0, analyzer.msd, structures

    return captured, mock.patch.object(module, "diffusion_analyzer_from_structures", fake)


def _write_data(path: Path) -> None:
    path.write_text(
        "2 atoms\n2 atom types\n\n"
        "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n"
        "Masses\n\n1 6.94 # Li\n2 4.00 # He\n\n"
        "Atoms # atomic\n\n1 1 9.8 0 0\n2 2 5 5 5\n",
        encoding="utf-8",
    )


def _write_dump(
    path: Path,
    columns: str,
    li_rows: list[str],
    *,
    start_step: int = 0,
) -> None:
    blocks = []
    for offset, li_row in enumerate(li_rows):
        blocks.append(
            "ITEM: TIMESTEP\n"
            f"{start_step + offset}\n"
            "ITEM: NUMBER OF ATOMS\n2\n"
            "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
            f"ITEM: ATOMS id type {columns}\n"
            f"1 1 {li_row}\n"
            f"2 2 {_stationary_row(columns)}\n"
        )
    path.write_text("".join(blocks), encoding="utf-8")


def _stationary_row(columns: str) -> str:
    names = columns.split()
    values = {
        "element": "He",
        "x": "5",
        "y": "5",
        "z": "5",
        "xs": "0.5",
        "ys": "0.5",
        "zs": "0.5",
        "xu": "5",
        "yu": "5",
        "zu": "5",
        "xsu": "0.5",
        "ysu": "0.5",
        "zsu": "0.5",
        "ix": "0",
        "iy": "0",
        "iz": "0",
    }
    return " ".join(values[name] for name in names)


def _legacy_lammps_run(tmp_path: Path, columns: str, rows: list[str]) -> Path:
    run = tmp_path / "T500"
    run.mkdir()
    _write_data(run / "data.LYC")
    _write_dump(run / "traj.lammpstrj", columns, rows)
    return run


@pytest.mark.parametrize(
    ("columns", "rows", "expected", "mode"),
    [
        (
            "element xu yu zu",
            ["Li 9.8 0 0", "Li 10.2 0 0", "Li 10.6 0 0"],
            [0.98, 1.02, 1.06],
            "unwrapped_cartesian",
        ),
        (
            "element xsu ysu zsu",
            ["Li 0.98 0 0", "Li 1.02 0 0", "Li 1.06 0 0"],
            [0.98, 1.02, 1.06],
            "unwrapped_scaled",
        ),
        (
            "element x y z",
            ["Li 9.8 0 0", "Li 0.2 0 0", "Li 0.6 0 0"],
            [0.98, 1.02, 1.06],
            "wrapped_cartesian_continuity",
        ),
        (
            "element xs ys zs",
            ["Li 0.98 0 0", "Li 0.02 0 0", "Li 0.06 0 0"],
            [0.98, 1.02, 1.06],
            "wrapped_scaled_continuity",
        ),
    ],
)
def test_lammps_coordinate_modes_and_pbc_continuity(
    tmp_path: Path, runner, columns: str, rows: list[str], expected: list[float], mode: str
) -> None:
    run_dir = _legacy_lammps_run(tmp_path, columns, rows)
    captured, patcher = _capture_analyzer(runner)
    with patcher:
        result = runner.load_lammps_trajectory(run_dir, "legacy", _args())

    assert np.allclose(captured["continuous_frac"][:, 0, 0], expected)
    assert mode in result.handoff["coordinate_modes"]
    assert result.source_path.name == "traj.lammpstrj"


def test_lammps_image_flags_take_precedence_and_unwrap_exactly(tmp_path: Path, runner) -> None:
    run_dir = _legacy_lammps_run(
        tmp_path,
        "element x y z ix iy iz",
        ["Li 9.8 0 0 0 0 0", "Li 0.2 0 0 1 0 0", "Li 0.6 0 0 1 0 0"],
    )
    captured, patcher = _capture_analyzer(runner)
    with patcher:
        result = runner.load_lammps_trajectory(run_dir, "legacy", _args())

    assert np.allclose(captured["continuous_frac"][:, 0, 0], [0.98, 1.02, 1.06])
    assert result.handoff["coordinate_modes"] == ["cartesian_with_image_flags"]


def _write_native_lammps_attempt(
    attempt: Path,
    rows: list[str],
    *,
    start_step: int,
    temperature: float = 600.0,
) -> None:
    attempt.mkdir(parents=True)
    _write_dump(
        attempt / "trajectory.lammpstrj",
        "element x y z",
        rows,
        start_step=start_step,
    )
    (attempt / "lammps-execution-result.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "plugin_id": "lammps-md",
                "status": "OK",
                "temperature_K": temperature,
                "timestep_fs": 1.0,
                "dump_interval": 1,
                "type_map": ["Li", "He"],
                "ensemble": "nvt",
                "model": {"id": "model-a", "path": "mace/model-a.pt"},
                "structure_path": "prepared/structure.data",
            }
        ),
        encoding="utf-8",
    )


def test_native_lammps_metadata_filename_and_restart_stitch(tmp_path: Path, runner) -> None:
    node = tmp_path / ".mlipflow" / "runs" / "lammps-600"
    _write_native_lammps_attempt(
        node / "attempt-1",
        ["Li 9.8 0 0", "Li 0.2 0 0", "Li 0.6 0 0"],
        start_step=0,
    )
    (node / "attempt-1" / "lammps-execution-result.json").unlink()
    (node / "attempt-1" / "approved-plan.json").write_text(
        json.dumps(
            {
                "adapter_plan": {
                    "lammps_calculation": {
                        "temperature_k": 600.0,
                        "timestep_fs": 1.0,
                        "dump_interval": 1,
                        "type_map": ["Li", "He"],
                        "ensemble": "nvt",
                        "model_id": "model-a",
                        "model_path": "mace/model-a.pt",
                        "structure_path": "prepared/structure.data",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    _write_native_lammps_attempt(
        node / "attempt-2",
        ["Li 0.6 0 0", "Li 1.0 0 0", "Li 1.4 0 0"],
        start_step=2,
    )
    captured, patcher = _capture_analyzer(runner)
    with patcher:
        result = runner.load_lammps_trajectory(node, "lammps-600", _args(lammps_timestep_ps=None))

    assert result.source_kind == "lammps_md_artifact"
    assert result.temperature_K == 600.0
    assert result.time_step_fs == 1.0
    assert result.handoff["global_steps"] == [0, 1, 2, 3, 4]
    assert result.handoff["segments"] == 2
    assert result.handoff["type_map"] == ["Li", "He"]
    assert result.handoff["model"]["id"] == "model-a"
    assert result.handoff["structure_path"] == "prepared/structure.data"
    assert np.allclose(captured["continuous_frac"][:, 0, 0], [0.98, 1.02, 1.06, 1.10, 1.14])


def _write_ase_trajectory(path: Path, positions: list[float]) -> None:
    from ase import Atoms
    from ase.io.trajectory import Trajectory

    trajectory = Trajectory(str(path), "w")
    for x in positions:
        trajectory.write(
            Atoms("LiHe", positions=[[x, 0, 0], [5, 5, 5]], cell=[10, 10, 10], pbc=True)
        )
    trajectory.close()


def _write_native_ase_attempt(
    attempt: Path,
    steps: list[int],
    positions: list[float],
    *,
    temperature: float = 500.0,
) -> None:
    attempt.mkdir(parents=True)
    _write_ase_trajectory(attempt / "trajectory.traj", positions)
    (attempt / "trajectory-index.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "steps": steps,
                "time_fs": [float(step) for step in steps],
            }
        ),
        encoding="utf-8",
    )
    (attempt / "md-result.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "plugin_id": "ase-md",
                "status": "OK",
                "temperature_K": temperature,
                "timestep_fs": 1.0,
                "trajectory_interval": 1,
                "ensemble": "nvt-langevin",
                "model": {"id": "model-b", "path": "mace/model-b.model"},
                "structure_path": "structure/start.extxyz",
            }
        ),
        encoding="utf-8",
    )


def test_native_ase_metadata_filename_and_restart_stitch(tmp_path: Path, runner) -> None:
    node = tmp_path / ".mlipflow" / "runs" / "ase-500"
    _write_native_ase_attempt(node / "attempt-1", [0, 1, 2], [9.8, 10.2, 10.6])
    (node / "attempt-1" / "md-result.json").unlink()
    (node / "attempt-1" / "approved-plan.json").write_text(
        json.dumps(
            {
                "adapter_plan": {
                    "md_parameters": {
                        "temperature_k": 500.0,
                        "timestep_fs": 1.0,
                        "trajectory_interval": 1,
                        "ensemble": "nvt-langevin",
                        "model_id": "model-b",
                        "model_path": "mace/model-b.model",
                        "structure_path": "structure/start.extxyz",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    _write_native_ase_attempt(node / "attempt-2", [2, 3, 4], [10.6, 11.0, 11.4])
    captured, patcher = _capture_analyzer(runner)
    with patcher:
        result = runner.load_ase_trajectory(node, "ase-500", _args())

    assert result.source_kind == "ase_md_artifact"
    assert result.temperature_K == 500.0
    assert result.time_step_fs == 1.0
    assert result.handoff["global_steps"] == [0, 1, 2, 3, 4]
    assert result.handoff["physical_time_fs"] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert result.handoff["segments"] == 2
    assert result.handoff["model"]["id"] == "model-b"
    assert result.handoff["structure_path"] == "structure/start.extxyz"
    assert np.allclose(captured["continuous_frac"][:, 0, 0], [0.98, 1.02, 1.06, 1.10, 1.14])


def test_historical_production_traj_remains_supported(tmp_path: Path, runner) -> None:
    run_dir = tmp_path / "legacy-ase" / "T700"
    run_dir.mkdir(parents=True)
    _write_ase_trajectory(run_dir / "production.traj", [9.8, 0.2, 0.6])
    (run_dir / "metadata.json").write_text(
        json.dumps({"temperature_K": 700, "time_step_fs_between_frames": 2.0}),
        encoding="utf-8",
    )
    captured, patcher = _capture_analyzer(runner)
    with patcher:
        result = runner.load_ase_trajectory(run_dir, "legacy", _args())

    assert result.source_path.name == "production.traj"
    assert result.temperature_K == 700.0
    assert result.time_step_fs == 2.0
    assert np.allclose(captured["continuous_frac"][:, 0, 0], [0.98, 1.02, 1.06])


def test_core_hands_all_preserved_dependency_attempts_to_transport(tmp_path: Path) -> None:
    from mlipflow.config import Project
    from mlipflow.services.contracts import _adapter_context
    from mlipflow.services.paths import attempt_directory, state_path
    from mlipflow.state import RunState, StateStore

    nodes = [
        {"id": "md", "uses": "ase-md", "backend": "local"},
        {"id": "transport", "uses": "ionic-transport", "needs": ["md"], "backend": "local"},
    ]
    project = Project(
        root=tmp_path,
        path=tmp_path / "project.json",
        raw={"project": {"id": "native-handoff"}, "workflow": {"nodes": nodes}},
    )
    first_dir = attempt_directory(project, "md", 1)
    first_dir.mkdir(parents=True)
    first_trajectory = first_dir / "trajectory.traj"
    first_trajectory.write_bytes(b"segment-1")
    with StateStore(state_path(project), readonly=False) as store:
        store.initialize_project(project.project_id, nodes)
        first = store.latest_step(project.project_id, "md")
        store.add_artifact(first.run_id, "trajectory", str(first_trajectory))
        store.transition(first.run_id, RunState.RUNNING)
        store.transition(first.run_id, RunState.FAIL)
        second = store.create_retry(project.project_id, "md")
        second_dir = attempt_directory(project, "md", 2)
        second_dir.mkdir(parents=True)
        second_trajectory = second_dir / "trajectory.traj"
        second_trajectory.write_bytes(b"segment-2")
        store.add_artifact(second.run_id, "trajectory", str(second_trajectory))
        store.transition(second.run_id, RunState.RUNNING)
        store.transition(second.run_id, RunState.OK)

    context = _adapter_context(project, nodes[1], 1)
    assert [item["attempt"] for item in context["upstream_artifacts"]] == [1, 2]
    assert [item["state"] for item in context["upstream_artifacts"]] == ["FAIL", "OK"]
    assert context["inputs"]["input_paths"] == [str(first_dir), str(second_dir)]


def test_discovery_stitches_only_selected_collected_attempts(tmp_path: Path, runner) -> None:
    node = tmp_path / ".mlipflow" / "runs" / "md"
    for attempt_number in (1, 2, 3):
        attempt = node / f"attempt-{attempt_number}"
        attempt.mkdir(parents=True)
        (attempt / "trajectory.lammpstrj").write_text("segment\n", encoding="utf-8")

    discovered = runner.discover_run_dirs([node / "attempt-1", node / "attempt-2"])
    assert len(discovered) == 1
    assert discovered[0][1] == node.resolve()
    assert discovered[0][2] == (
        (node / "attempt-1").resolve(),
        (node / "attempt-2").resolve(),
    )


@pytest.mark.parametrize("producer", ["ase", "lammps"])
def test_four_temperature_native_artifacts_feed_one_arrhenius_fit(
    tmp_path: Path, runner, producer: str
) -> None:
    root = tmp_path / producer
    for temperature, displacement in ((500, 0.2), (600, 0.3), (700, 0.4), (800, 0.5)):
        node = root / f"{producer}-{temperature}" / "attempt-1"
        if producer == "ase":
            _write_native_ase_attempt(
                node,
                list(range(8)),
                [index * displacement for index in range(8)],
                temperature=float(temperature),
            )
        else:
            _write_native_lammps_attempt(
                node,
                [f"Li {index * displacement:g} 0 0" for index in range(8)],
                start_step=0,
                temperature=float(temperature),
            )

    output = tmp_path / f"{producer}-transport"
    argv = [
        "--input",
        str(root),
        "--output",
        str(output),
        "--source",
        "trajectory",
        "--specie",
        "Li",
        "--diffusion-analyzer-smoothed",
        "none",
        "--diffusion-analyzer-min-obs",
        "1",
        "--diffusion-analyzer-avg-nsteps",
        "2",
        "--fit-scope",
        "all",
        "--piecewise",
        "never",
    ]
    assert runner.main(argv) == 0
    summary = json.loads((output / "arrhenius_summary.json").read_text(encoding="utf-8"))
    assert summary["fit_temperatures_K"] == [500.0, 600.0, 700.0, 800.0]
    assert summary["single"]["Ea_eV"] > 0
