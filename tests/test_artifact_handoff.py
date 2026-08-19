from __future__ import annotations

from pathlib import Path

import pytest

from mlipflow.config import Project
from mlipflow.errors import PluginError
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
        store.add_artifact(failed.run_id, "trajectory", str(failed_trajectory))
        store.transition(failed.run_id, RunState.RUNNING)
        store.transition(failed.run_id, RunState.FAIL)
        successful = store.create_retry(project.project_id, "producer")
        successful_dir = attempt_directory(project, "producer", 2)
        successful_dir.mkdir(parents=True)
        successful_trajectory = successful_dir / "trajectory.traj"
        successful_trajectory.write_bytes(b"ok")
        store.add_artifact(successful.run_id, "trajectory", str(successful_trajectory))
        store.add_artifact(successful.run_id, "trajectory", str(successful_trajectory))
        store.transition(successful.run_id, RunState.RUNNING)
        store.transition(successful.run_id, RunState.OK)

    context = _adapter_context(project, nodes[1], 1)
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
        store.add_artifact(producer.run_id, "trajectory", trajectory.resolve().as_uri())
        store.transition(producer.run_id, RunState.RUNNING)
        store.transition(producer.run_id, RunState.OK)

    context = _adapter_context(project, nodes[1], 1)
    assert context["inputs"]["input_dirs"] == [str(trajectory.resolve())]


def test_pinned_scheduled_node_restores_snapshot_dependency_edges(tmp_path: Path) -> None:
    binding = {"from_node": "producer", "role": "trajectory"}
    project, nodes = _project(tmp_path, binding)
    attempt = attempt_directory(project, "producer", 1)
    attempt.mkdir(parents=True)
    trajectory = attempt / "trajectory.traj"
    trajectory.write_bytes(b"ok")

    with StateStore(state_path(project), readonly=False) as store:
        store.initialize_project(project.project_id, nodes)
        producer = store.latest_step(project.project_id, "producer")
        store.add_artifact(producer.run_id, "trajectory", trajectory.resolve().as_uri())
        store.transition(producer.run_id, RunState.RUNNING)
        store.transition(producer.run_id, RunState.OK)
        consumer = store.latest_step(project.project_id, "consumer")
        snapshot = store.node_snapshot(consumer.run_id)

    pinned = _attempt_node(
        {"inputs": {"input_dirs": [binding]}},
        consumer,
        needs=snapshot["needs"],
    )
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
        store.add_artifact(producer.run_id, "selected-structure", str(first))
        store.add_artifact(producer.run_id, "selected-structure", str(second))
        store.transition(producer.run_id, RunState.RUNNING)
        store.transition(producer.run_id, RunState.OK)

    with pytest.raises(PluginError, match="exactly one unique artifact"):
        _adapter_context(project, nodes[1], 1)
