from __future__ import annotations

from pathlib import Path

import pytest

from mlipflow.config import Project
from mlipflow.errors import CapabilityError
from mlipflow.io import write_json_atomic
from mlipflow.services.contracts import _adapter_context
from mlipflow.services.paths import attempt_directory, state_path
from mlipflow.services.scheduled import _attempt_node
from mlipflow.state import RunState, StateStore


def _project(tmp_path: Path, binding: object) -> tuple[Project, list[dict[str, object]]]:
    nodes: list[dict[str, object]] = [
        {"id": "producer", "uses": "ase-md", "backend": "local"},
        {
            "id": "consumer",
            "uses": "pes-sampling",
            "needs": ["producer"],
            "backend": "local",
            "inputs": {"input_dirs": [binding]},
        },
    ]
    project = Project(
        root=tmp_path,
        path=tmp_path / "project.json",
        raw={"project": {"id": "artifact-handoff"}, "workflow": {"nodes": nodes}},
    )
    return project, nodes


def _record(attempt: Path, artifacts: list[dict[str, str]]) -> None:
    write_json_atomic(attempt / "run-manifest.final.json", {"artifacts": artifacts})


def test_explicit_artifact_binding_uses_final_ok_attempt_and_parent(tmp_path: Path) -> None:
    binding = {"from_node": "producer", "role": "trajectory", "resolve": "parent"}
    project, nodes = _project(tmp_path, binding)
    failed_dir = attempt_directory(project, "producer", 1)
    failed_dir.mkdir(parents=True)
    failed_trajectory = failed_dir / "trajectory.traj"
    failed_trajectory.write_bytes(b"failed")

    with StateStore(state_path(project), readonly=False) as store:
        store.initialize_project(project.project_id, nodes)
        failed = store.latest_step(project.project_id, "producer")
        _record(
            failed_dir,
            [{"role": "trajectory", "uri": failed_trajectory.resolve().as_uri()}],
        )
        store.transition(failed.run_id, RunState.RUNNING)
        store.transition(failed.run_id, RunState.FAIL)
        successful = store.create_retry(project.project_id, "producer", "local")
        successful_dir = attempt_directory(project, "producer", 2)
        successful_dir.mkdir(parents=True)
        successful_trajectory = successful_dir / "trajectory.traj"
        successful_trajectory.write_bytes(b"ok")
        _record(
            successful_dir,
            [
                {
                    "role": "trajectory",
                    "uri": successful_trajectory.resolve().as_uri(),
                }
            ],
        )
        store.transition(successful.run_id, RunState.RUNNING)
        store.transition(successful.run_id, RunState.OK)

    context = _adapter_context(project, nodes[1], 1)
    assert context["project_path"] == str(project.path)
    assert context["inputs"]["input_dirs"] == [str(successful_dir.resolve())]


def test_explicit_artifact_binding_accepts_collected_file_uri(tmp_path: Path) -> None:
    binding = {"from_node": "producer", "role": "trajectory"}
    project, nodes = _project(tmp_path, binding)
    attempt = attempt_directory(project, "producer", 1)
    attempt.mkdir(parents=True)
    trajectory = attempt / "trajectory with space.traj"
    trajectory.write_bytes(b"ok")

    with StateStore(state_path(project), readonly=False) as store:
        store.initialize_project(project.project_id, nodes)
        producer = store.latest_step(project.project_id, "producer")
        _record(
            attempt,
            [{"role": "trajectory", "uri": trajectory.resolve().as_uri()}],
        )
        store.transition(producer.run_id, RunState.RUNNING)
        store.transition(producer.run_id, RunState.OK)

    context = _adapter_context(project, nodes[1], 1)
    assert context["inputs"]["input_dirs"] == [str(trajectory.resolve())]


def test_scheduled_node_reads_dependency_edges_from_project(tmp_path: Path) -> None:
    binding = {"from_node": "producer", "role": "trajectory"}
    project, nodes = _project(tmp_path, binding)
    attempt = attempt_directory(project, "producer", 1)
    attempt.mkdir(parents=True)
    trajectory = attempt / "trajectory.traj"
    trajectory.write_bytes(b"ok")

    with StateStore(state_path(project), readonly=False) as store:
        store.initialize_project(project.project_id, nodes)
        producer = store.latest_step(project.project_id, "producer")
        _record(
            attempt,
            [{"role": "trajectory", "uri": trajectory.resolve().as_uri()}],
        )
        store.transition(producer.run_id, RunState.RUNNING)
        store.transition(producer.run_id, RunState.OK)
        consumer = store.latest_step(project.project_id, "consumer")

    pinned = _attempt_node(project, {}, consumer)
    context = _adapter_context(project, pinned, 1)
    assert pinned["needs"] == ["producer"]
    assert context["inputs"]["input_dirs"] == [str(trajectory.resolve())]


def test_artifact_binding_rejects_ambiguous_role(tmp_path: Path) -> None:
    binding = {"from_node": "producer", "role": "selected-structure"}
    project, nodes = _project(tmp_path, binding)
    attempt = attempt_directory(project, "producer", 1)
    attempt.mkdir(parents=True)
    first = attempt / "first.vasp"
    second = attempt / "second.vasp"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    with StateStore(state_path(project), readonly=False) as store:
        store.initialize_project(project.project_id, nodes)
        producer = store.latest_step(project.project_id, "producer")
        _record(
            attempt,
            [
                {"role": "selected-structure", "uri": first.resolve().as_uri()},
                {"role": "selected-structure", "uri": second.resolve().as_uri()},
            ],
        )
        store.transition(producer.run_id, RunState.RUNNING)
        store.transition(producer.run_id, RunState.OK)

    with pytest.raises(CapabilityError, match="exactly one unique artifact"):
        _adapter_context(project, nodes[1], 1)
