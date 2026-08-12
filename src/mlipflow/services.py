"""Read and mutation services.

The top half of this module contains query functions. They only open SQLite in
read-only mode and never call execution backends. Mutation functions are kept
separate below the explicit boundary marker.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

from .artifacts import fingerprint
from .backends import LocalBackend, SlurmBackend, SshSlurmBackend, validate_argv
from .config import Project, load_project
from .errors import ApprovalError, BackendError, ConfigError, PluginError, StateError
from .io import load_mapping, write_json_atomic, write_text_atomic
from .manifests import run_manifest
from .planning import action_plan, node_plan, resolve_reference, with_digest
from .plugins import PluginSpec, discover_plugins, load_adapter, select_plugin
from .routing import load_registry, route_models
from .state import RunState, StateStore, utc_now


STATE_RELATIVE = Path(".mlipflow/state.sqlite3")


def state_path(project: Project) -> Path:
    configuration = project.raw.get("state", {})
    reference = (
        configuration.get("database_path", str(STATE_RELATIVE))
        if isinstance(configuration, dict)
        else str(STATE_RELATIVE)
    )
    if not isinstance(reference, str) or not reference or Path(reference).is_absolute():
        raise ConfigError("state.database_path must be a non-empty project-relative path")
    candidate = (project.root / reference).resolve()
    try:
        candidate.relative_to(project.root.resolve())
    except ValueError as exc:
        raise ConfigError("state.database_path must remain inside the project root") from exc
    return candidate


def _assert_state_matches_project(store: StateStore, project: Project) -> None:
    store.assert_project_config(project.project_id, project.raw)
    store.assert_project_nodes(project.project_id, project.nodes)


def query_workflow(project: Project, node_id: str | None = None) -> dict[str, Any]:
    database = state_path(project)
    if database.is_file():
        with StateStore(database, readonly=True) as store:
            steps = [step.to_dict() for step in store.latest_steps(project.project_id)]
            for step in steps:
                step["artifacts"] = store.artifacts(step["run_id"])
                step["persisted"] = True
    else:
        steps = [
            {
                "run_id": None,
                "project_id": project.project_id,
                "node_id": node["id"],
                "plugin_id": node["uses"],
                "attempt": 0,
                "state": RunState.WAIT.value if node.get("needs") else RunState.READY.value,
                "backend": node.get("backend", "local"),
                "job_id": None,
                "remote_dir": None,
                "created_at": None,
                "updated_at": None,
                "submitted_at": None,
                "started_at": None,
                "ended_at": None,
                "retry_count": 0,
                "manifest_path": None,
                "diagnostic": "preview: project has not been initialized",
                "artifacts": [],
                "persisted": False,
            }
            for node in project.nodes
        ]
    if node_id is not None:
        steps = [step for step in steps if step["node_id"] == node_id]
        if not steps:
            raise ConfigError(f"unknown workflow node: {node_id}")
    counts: dict[str, int] = {}
    for step in steps:
        counts[step["state"]] = counts.get(step["state"], 0) + 1
    return {
        "project": {"id": project.project_id, "path": str(project.path)},
        "state_database": str(database),
        "initialized": database.is_file(),
        "counts": dict(sorted(counts.items())),
        "steps": steps,
    }


def query_inspect(project: Project, node_id: str, plugin_root: Path) -> dict[str, Any]:
    node = project.node(node_id)
    plugins = discover_plugins(plugin_root)
    plugin = select_plugin(plugins, str(node["uses"]))
    workflow = query_workflow(project, node_id)
    return {
        "project_id": project.project_id,
        "node": node,
        "state": workflow["steps"][0],
        "plugin": {"path": str(plugin.path), "manifest": plugin.raw},
    }


def query_logs(project: Project, node_id: str, tail: int = 80) -> dict[str, Any]:
    if tail < 1 or tail > 10000:
        raise ConfigError("--tail must be between 1 and 10000")
    database = state_path(project)
    if not database.is_file():
        raise StateError("project has not been initialized")
    with StateStore(database, readonly=True) as store:
        step = store.latest_step(project.project_id, node_id)
    directory = project.root / ".mlipflow" / "runs" / node_id / f"attempt-{step.attempt}"
    logs: dict[str, Any] = {}
    for name in ("stdout.log", "stderr.log"):
        path = directory / name
        logs[name] = {
            "path": str(path),
            "exists": path.is_file(),
            "lines": _tail_lines(path, tail) if path.is_file() else [],
        }
    return {"node_id": node_id, "attempt": step.attempt, "logs": logs}


def query_route(
    project: Project,
    *,
    task: str,
    elements: Iterable[str],
    scenario: str,
) -> dict[str, Any]:
    registry_ref = project.raw.get("model_registry")
    if not isinstance(registry_ref, str):
        raise ConfigError("project.model_registry must be a path")
    registry_path = resolve_reference(registry_ref, project.root)
    policies = project.raw.get("routing", {}).get("policies", {})
    policy = policies.get(task) if isinstance(policies, dict) else None
    if not isinstance(policy, dict):
        raise ConfigError(f"no routing policy for task {task!r}")
    return route_models(
        load_registry(
            registry_path,
            full_hash_max_bytes=int(
                project.raw.get("fingerprints", {}).get("full_hash_max_bytes", 67108864)
            ),
        ),
        task=task,
        elements=set(elements),
        scenario=scenario,
        policy=policy,
    )


def query_doctor(project_path: Path, plugin_root: Path) -> dict[str, Any]:
    diagnostics: list[dict[str, Any]] = []
    try:
        project = load_project(project_path)
        diagnostics.append({"check": "project", "ok": True, "detail": str(project.path)})
    except Exception as exc:
        return {"ok": False, "diagnostics": [{"check": "project", "ok": False, "detail": str(exc)}]}
    try:
        plugins = discover_plugins(plugin_root)
        diagnostics.append(
            {"check": "plugins", "ok": bool(plugins), "detail": f"{len(plugins)} discovered"}
        )
        for node in project.nodes:
            plugin = select_plugin(plugins, str(node["uses"]))
            if node.get("mode", "execute") == "replay":
                continue
            dependencies = plugin.raw.get("dependencies", {})
            for dependency in dependencies.get("python_packages", []):
                if not dependency.get("required"):
                    continue
                package = (
                    str(dependency.get("name", "")).replace("-", "_").split(".", 1)[0]
                )
                available = bool(package) and importlib.util.find_spec(package) is not None
                diagnostics.append(
                    {
                        "check": f"python:{package}",
                        "ok": available,
                        "detail": f"required by {plugin.plugin_id}",
                    }
                )
            for dependency in dependencies.get("external_programs", []):
                if not dependency.get("required"):
                    continue
                executable = str(dependency.get("name", ""))
                path = shutil.which(executable) if executable and ":" not in executable else None
                diagnostics.append(
                    {
                        "check": f"executable:{executable}",
                        "ok": path is not None,
                        "detail": path or f"required by {plugin.plugin_id}",
                    }
                )
    except Exception as exc:
        diagnostics.append({"check": "plugins", "ok": False, "detail": str(exc)})
    database = state_path(project)
    if database.is_file():
        try:
            with StateStore(database, readonly=True) as store:
                store.latest_steps(project.project_id)
            diagnostics.append({"check": "state", "ok": True, "detail": str(database)})
        except Exception as exc:
            diagnostics.append({"check": "state", "ok": False, "detail": str(exc)})
    else:
        diagnostics.append(
            {"check": "state", "ok": True, "detail": "not initialized (read-only check)"}
        )
    backends = {str(node.get("backend", "local")) for node in project.nodes}
    required = set()
    if "slurm" in backends:
        required.update({"sbatch", "scancel"})
    if "ssh-slurm" in backends:
        required.add("ssh")
    for executable in sorted(required):
        path = shutil.which(executable)
        diagnostics.append(
            {"check": f"executable:{executable}", "ok": path is not None, "detail": path}
        )
    return {"ok": all(item["ok"] for item in diagnostics), "diagnostics": diagnostics}


def _tail_lines(path: Path, count: int) -> list[str]:
    block_size = 64 * 1024
    maximum = 8 * 1024 * 1024
    with path.open("rb") as stream:
        stream.seek(0, 2)
        position = stream.tell()
        chunks: list[bytes] = []
        collected = 0
        newlines = 0
        while position > 0 and collected < maximum and newlines <= count:
            size = min(block_size, position, maximum - collected)
            position -= size
            stream.seek(position)
            chunk = stream.read(size)
            chunks.append(chunk)
            collected += len(chunk)
            newlines += chunk.count(b"\n")
    payload = b"".join(reversed(chunks))
    lines = payload.decode("utf-8", errors="replace").splitlines(keepends=True)
    if position > 0 and lines:
        lines = lines[1:]
    return lines[-count:]


# ---------------------------------------------------------------------------
# Mutation boundary: functions below may write or invoke execution backends.
# ---------------------------------------------------------------------------


DEFAULT_PROJECT = {
    "schema_version": 1,
    "project": {
        "id": "my-mlip-project",
        "name": "My MLIP project",
        "description": "Edit this deterministic workflow before running it.",
    },
    "locations": {},
    "backend_profiles": {},
    "model_registry": "model_registry.yaml",
    "workflow": {"nodes": []},
    "routing": {"policies": {}},
    "fingerprints": {"full_hash_max_bytes": 67108864},
    "safety": {"auto_submit": False},
}


def initialize(target: Path) -> dict[str, Any]:
    target = target.resolve()
    if target.suffix in {".yaml", ".yml", ".json"}:
        project_file = target
        root = target.parent
    else:
        root = target
        project_file = root / "project.yaml"
    root.mkdir(parents=True, exist_ok=True)
    created_project = False
    if not project_file.exists():
        write_json_atomic(project_file, DEFAULT_PROJECT)
        created_project = True
    project = load_project(project_file)
    database = state_path(project)
    existed = database.is_file()
    with StateStore(database, readonly=False) as store:
        if existed:
            _assert_state_matches_project(store, project)
        else:
            store.initialize_project(project.project_id, project.nodes)
            store.record_project_config(project.project_id, project.raw)
    return {
        "project": str(project_file),
        "state_database": str(database),
        "created_project": created_project,
        "nodes": len(project.nodes),
    }


def make_run_plan(project: Project, node_id: str, plugin_root: Path) -> dict[str, Any]:
    database = state_path(project)
    if database.is_file():
        with StateStore(database, readonly=True) as store:
            _assert_state_matches_project(store, project)
    node = project.node(node_id)
    plugins = discover_plugins(plugin_root)
    plugin = select_plugin(plugins, str(node["uses"]))
    plan = node_plan(project, node, plugin)
    if node.get("mode", "execute") == "execute" and plugin.raw.get("implementation", {}).get(
        "status"
    ) in {"adapter-ready", "implemented"}:
        context = _adapter_context(project, node, _planned_attempt(project, node_id))
        adapter = load_adapter(plugin)
        diagnostics = adapter.validate(context)
        adapter_plan = adapter.plan(context)
        unsigned = {key: value for key, value in plan.items() if key != "plan_digest"}
        unsigned["adapter_diagnostics"] = diagnostics
        unsigned["adapter_plan"] = adapter_plan
        unsigned["adapter_command_fingerprints"] = _adapter_command_fingerprints(
            project, adapter_plan
        )
        if (
            str(node.get("backend", "local")) == "ssh-slurm"
            and isinstance(adapter_plan, dict)
            and isinstance(adapter_plan.get("scheduled_execution"), dict)
        ):
            unsigned["scheduled_transport"] = _scheduled_transport_plan(project, node)
        plan = with_digest(unsigned)
    return plan


def run_node(
    project: Project,
    node_id: str,
    plugin_root: Path,
    approval: str,
) -> dict[str, Any]:
    node = project.node(node_id)
    plugins = discover_plugins(plugin_root)
    plugin = select_plugin(plugins, str(node["uses"]))
    plan = make_run_plan(project, node_id, plugin_root)
    _require_approval(plan, approval)
    adapter_plan = plan.get("adapter_plan")
    if isinstance(adapter_plan, dict) and (
        adapter_plan.get("status") != RunState.READY.value
        or adapter_plan.get("executable") is not True
    ):
        raise PluginError(
            f"plugin {plugin.plugin_id} blocked execution: {adapter_plan.get('diagnostics', adapter_plan)}"
        )
    if node.get("mode", "execute") == "execute":
        backend = str(node.get("backend", "local"))
        if backend == "local" and not isinstance(adapter_plan, dict):
            raise PluginError(
                "local execution requires an adapter-ready plugin with scientific check/collect"
            )
        if backend == "slurm" and isinstance(adapter_plan, dict):
            raise BackendError(
                "local SLURM adapter execution is disabled; only the controlled ssh-slurm "
                "staging/fetch/check contract is currently implemented"
            )
        if backend == "ssh-slurm" and isinstance(adapter_plan, dict):
            if not isinstance(adapter_plan.get("scheduled_execution"), dict) or not isinstance(
                plan.get("scheduled_transport"), dict
            ):
                raise BackendError(
                    "ssh-slurm adapter execution requires an explicit staged execution contract"
                )
    database = state_path(project)
    existed = database.is_file()
    with StateStore(database, readonly=False) as store:
        if existed:
            _assert_state_matches_project(store, project)
        else:
            store.initialize_project(project.project_id, project.nodes)
            store.record_project_config(project.project_id, project.raw)
        step = store.latest_step(project.project_id, node_id)
        if step.state == RunState.WAIT.value:
            if not store.dependencies_satisfied(project.project_id, node_id):
                raise StateError(f"node {node_id} is waiting for dependencies")
            step = store.transition(step.run_id, RunState.READY)
        if step.state != RunState.READY.value:
            raise StateError(f"node {node_id} is {step.state}, expected READY")
        store.record_intent(plan["plan_digest"], project.project_id, node_id, plan)
        try:
            result = _execute_ready(project, node, plugin, plan, store, step.run_id, step.attempt)
        finally:
            store.consume_intent(plan["plan_digest"])
    return result


def make_advance_plan(project: Project, plugin_root: Path | None = None) -> dict[str, Any]:
    database = state_path(project)
    transitions: list[dict[str, str]] = []
    observations: list[dict[str, Any]] = []
    if database.is_file():
        with StateStore(database, readonly=True) as store:
            _assert_state_matches_project(store, project)
            for step in store.latest_steps(project.project_id):
                if step.state in {
                    RunState.SUBMITTED.value,
                    RunState.PENDING.value,
                    RunState.RUNNING.value,
                } and step.job_id:
                    observation = _observe_scheduled_step(project, step, plugin_root)
                    observations.append(observation)
                    finalization = observation.get("adapter_finalization")
                    if isinstance(finalization, dict):
                        transitions.append(
                            {
                                "node_id": step.node_id,
                                "action": "adapter-finalize",
                                "reason": str(
                                    observation.get(
                                        "reason", "scheduler adapter finalization"
                                    )
                                ),
                                "scheduler_state": str(
                                    observation.get("scheduler_state", "UNKNOWN")
                                ),
                                "adapter_finalization": finalization,
                            }
                        )
                        continue
                    target = observation.get("target_state")
                    if isinstance(target, str) and target != step.state:
                        transitions.append(
                            {
                                "node_id": step.node_id,
                                "to": target,
                                "reason": str(observation.get("reason", "scheduler reconciliation")),
                                "scheduler_state": str(observation.get("scheduler_state", "UNKNOWN")),
                                "completion_manifest": observation.get("completion_manifest"),
                                "completion_fingerprint": observation.get(
                                    "completion_fingerprint"
                                ),
                            }
                        )
                    continue
                if step.state not in {RunState.WAIT.value, RunState.BLOCKED.value}:
                    continue
                node = project.node(step.node_id)
                dependency_states = [
                    store.latest_step(project.project_id, dependency).state
                    for dependency in node.get("needs", [])
                ]
                if all(state == RunState.OK.value for state in dependency_states):
                    target = RunState.READY.value
                elif any(state in {RunState.FAIL.value, RunState.STOPPED.value} for state in dependency_states):
                    target = RunState.BLOCKED.value
                elif step.state == RunState.BLOCKED.value:
                    target = RunState.WAIT.value
                else:
                    target = step.state
                if target != step.state:
                    transitions.append(
                        {"node_id": step.node_id, "to": target, "reason": "dependency reconciliation"}
                    )
    return action_plan(
        "advance", project, None, {"transitions": transitions, "observations": observations}
    )


def advance(
    project: Project, approval: str, plugin_root: Path | None = None
) -> dict[str, Any]:
    plan = make_advance_plan(project, plugin_root)
    _require_approval(plan, approval)
    database = state_path(project)
    if not database.is_file():
        raise StateError("project has not been initialized")
    changed: list[dict[str, Any]] = []
    with StateStore(database, readonly=False) as store:
        _assert_state_matches_project(store, project)
        for change in plan["details"]["transitions"]:
            step = store.latest_step(project.project_id, change["node_id"])
            if change.get("action") == "adapter-finalize":
                if plugin_root is None:
                    raise PluginError("scheduled adapter finalization requires the plugin root")
                changed.append(
                    _finalize_scheduled_adapter(
                        project, step, change, plugin_root, store
                    ).to_dict()
                )
                continue
            target = RunState(change["to"])
            manifest_ref = change.get("completion_manifest")
            artifacts: list[dict[str, Any]] = []
            metrics: dict[str, Any] = {}
            if target == RunState.OK and isinstance(manifest_ref, str):
                result_path = _project_scoped_result_path(
                    project, resolve_reference(manifest_ref, project.root)
                )
                approved_fingerprint = change.get("completion_fingerprint")
                current_fingerprint = fingerprint(result_path)
                if not isinstance(approved_fingerprint, dict) or (
                    current_fingerprint != approved_fingerprint
                ):
                    raise StateError(
                        "completion manifest changed after the approved advance plan"
                    )
                result_data, artifacts = _load_result(
                    result_path, _scheduler_expected_identity(step)
                )
                raw_metrics = result_data.get("metrics", {})
                metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
            if target == RunState.OK and RunState(step.state) in {
                RunState.SUBMITTED,
                RunState.PENDING,
            }:
                step = store.transition(
                    step.run_id,
                    RunState.RUNNING,
                    diagnostic="scheduler completed between observations",
                )
            for artifact in artifacts:
                store.add_artifact(
                    step.run_id,
                    str(artifact.get("role", "output")),
                    str(artifact["uri"]),
                    artifact.get("fingerprint"),
                    artifact.get("size_bytes"),
                    artifact.get("metadata", {}),
                )
            final_manifest = _finalize_scheduler_manifest(
                project,
                step,
                target,
                str(change.get("reason", "scheduler reconciliation")),
                str(change.get("scheduler_state", "UNKNOWN")),
                artifacts,
                metrics,
            )
            updated = store.transition(
                step.run_id,
                target,
                diagnostic=str(change.get("reason", "scheduler reconciliation")),
                manifest_path=str(final_manifest) if final_manifest else None,
            )
            changed.append(updated.to_dict())
    return {"plan_digest": plan["plan_digest"], "changed": changed}


def make_retry_plan(
    project: Project, node_id: str, plugin_root: Path | None = None
) -> dict[str, Any]:
    database = state_path(project)
    if not database.is_file():
        raise StateError("project has not been initialized")
    with StateStore(database, readonly=True) as store:
        _assert_state_matches_project(store, project)
        step = store.latest_step(project.project_id, node_id)
    if step.state not in {RunState.FAIL.value, RunState.STOPPED.value}:
        raise StateError(f"retry requires FAIL or STOPPED, got {step.state}")
    max_attempts: int | None = None
    if plugin_root is not None:
        plugin = select_plugin(discover_plugins(plugin_root), str(project.node(node_id)["uses"]))
        configured = plugin.raw.get("retry", {}).get("max_attempts")
        if isinstance(configured, int) and not isinstance(configured, bool):
            max_attempts = configured
            if step.attempt >= max_attempts:
                raise StateError(
                    f"node {node_id} reached plugin retry limit of {max_attempts} attempts"
                )
    return action_plan(
        "retry",
        project,
        node_id,
        {
            "previous_run_id": step.run_id,
            "previous_attempt": step.attempt,
            "state": step.state,
            "max_attempts": max_attempts,
        },
    )


def retry(
    project: Project,
    node_id: str,
    approval: str,
    plugin_root: Path | None = None,
) -> dict[str, Any]:
    plan = make_retry_plan(project, node_id, plugin_root)
    _require_approval(plan, approval)
    with StateStore(state_path(project), readonly=False) as store:
        _assert_state_matches_project(store, project)
        retried = store.create_retry(project.project_id, node_id)
    return {"plan_digest": plan["plan_digest"], "step": retried.to_dict()}


def make_stop_plan(project: Project, node_id: str) -> dict[str, Any]:
    database = state_path(project)
    if not database.is_file():
        raise StateError("project has not been initialized")
    with StateStore(database, readonly=True) as store:
        _assert_state_matches_project(store, project)
        step = store.latest_step(project.project_id, node_id)
    scheduler_target = None
    if step.backend == "ssh-slurm":
        scheduler_target = _ssh_profile(project, project.node(node_id))
    return action_plan(
        "stop",
        project,
        node_id,
        {
            "run_id": step.run_id,
            "state": step.state,
            "backend": step.backend,
            "backend_profile": project.node(node_id).get("backend_profile"),
            "scheduler_target": scheduler_target,
            "job_id": step.job_id,
        },
    )


def stop(project: Project, node_id: str, approval: str) -> dict[str, Any]:
    plan = make_stop_plan(project, node_id)
    _require_approval(plan, approval)
    with StateStore(state_path(project), readonly=False) as store:
        _assert_state_matches_project(store, project)
        step = store.latest_step(project.project_id, node_id)
        state = RunState(step.state)
        if state in {RunState.OK, RunState.FAIL, RunState.STOPPED}:
            raise StateError(f"cannot stop node in terminal state {state.value}")
        if step.job_id:
            if step.backend == "slurm":
                result = SlurmBackend().cancel(step.job_id)
            elif step.backend == "ssh-slurm":
                node = project.node(node_id)
                profile = _ssh_profile(project, node)
                result = SshSlurmBackend(profile).cancel(step.job_id)
            else:
                raise BackendError(f"stored job id is incompatible with backend {step.backend}")
            if result.returncode != 0:
                raise BackendError(result.stderr or result.stdout or "scheduler cancellation failed")
        elif state == RunState.RUNNING:
            raise BackendError("cannot safely stop a local run without a persisted process handle")
        updated = store.transition(step.run_id, RunState.STOPPED)
    return {"plan_digest": plan["plan_digest"], "step": updated.to_dict()}


def _execute_ready(
    project: Project,
    node: dict[str, Any],
    plugin: PluginSpec,
    plan: dict[str, Any],
    store: StateStore,
    run_id: str,
    attempt: int,
) -> dict[str, Any]:
    node_id = str(node["id"])
    mode = str(node.get("mode", "execute"))
    backend = str(node.get("backend", "local"))
    directory = project.root / ".mlipflow" / "runs" / node_id / f"attempt-{attempt}"
    directory.mkdir(parents=True, exist_ok=False)
    manifest_path = directory / "run-manifest.json"
    try:
        if mode == "replay":
            store.transition(run_id, RunState.RUNNING)
            result_data, artifacts = _replay(project, node)
            manifest = run_manifest(
                project_id=project.project_id,
                node_id=node_id,
                run_id=run_id,
                attempt=attempt,
                plugin_id=plugin.plugin_id,
                plugin_version=str(plugin.raw["version"]),
                mode=mode,
                backend=backend,
                plan_digest=plan["plan_digest"],
                inputs=node.get("inputs", {}),
                parameters=node.get("parameters", {}),
                artifacts=artifacts,
                metrics=result_data.get("metrics", {}),
                command=None,
                resources=node.get("resources", {}),
                state=RunState.OK.value,
                state_reason="replay collected",
                **_manifest_context(project, node, plugin, store, run_id, finished=True),
            )
            write_json_atomic(manifest_path, manifest)
            for artifact in artifacts:
                store.add_artifact(
                    run_id,
                    str(artifact.get("role", "output")),
                    str(artifact["uri"]),
                    artifact.get("fingerprint"),
                    artifact.get("size_bytes"),
                    artifact.get("metadata", {}),
                )
            updated = store.transition(
                run_id, RunState.OK, manifest_path=str(manifest_path), diagnostic="replay collected"
            )
            return {
                "plan_digest": plan["plan_digest"],
                "step": updated.to_dict(),
                "result": result_data,
            }
        if plugin.kind == "replay-only":
            raise PluginError(f"plugin {plugin.plugin_id} only supports replay")
        if backend == "local":
            store.transition(run_id, RunState.RUNNING)
            adapter_plan = plan.get("adapter_plan")
            if not isinstance(adapter_plan, dict):
                raise PluginError(
                    "local execution requires an adapter-ready plugin with scientific check/collect"
                )
            if isinstance(adapter_plan, dict):
                argv = adapter_plan.get("argv")
            else:
                argv = node.get("parameters", {}).get("argv")
            if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
                raise ConfigError(f"node {node_id} execution plan must provide argv as a string list")
            result = LocalBackend().run(argv, directory)
            write_text_atomic(directory / "stdout.log", result.stdout)
            write_text_atomic(directory / "stderr.log", result.stderr)
            artifacts = [fingerprint(path) | {"role": path.stem} for path in directory.glob("*.log")]
            check_result: dict[str, Any] | None = None
            collected: dict[str, Any] | None = None
            metrics: dict[str, Any] = {}
            if result.returncode != 0:
                final = RunState.FAIL
                reason = f"process exited {result.returncode}"
            elif isinstance(adapter_plan, dict):
                adapter = load_adapter(plugin)
                context = _adapter_context(project, node, attempt)
                context["execution"] = {
                    "returncode": result.returncode,
                    "stdout_path": str(directory / "stdout.log"),
                    "stderr_path": str(directory / "stderr.log"),
                    "plan": adapter_plan,
                }
                check_result = adapter.check(context)
                if not isinstance(check_result, dict) or check_result.get("status") != RunState.OK.value:
                    final = RunState.FAIL
                    reason = f"plugin completion check did not return OK: {check_result}"
                else:
                    collected = adapter.collect(context)
                    if not isinstance(collected, dict) or collected.get("status") != RunState.OK.value:
                        final = RunState.FAIL
                        reason = f"plugin collection did not return OK: {collected}"
                    else:
                        artifacts.extend(
                            _normalize_adapter_artifacts(
                                project, directory, collected.get("artifacts", [])
                            )
                        )
                        raw_metrics = collected.get("metrics", {})
                        metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
                        final = RunState.OK
                        reason = "external command and plugin completion checks succeeded"
            result_data = {
                "returncode": result.returncode,
                "check": check_result,
                "collection": collected,
            }
            manifest = run_manifest(
                project_id=project.project_id,
                node_id=node_id,
                run_id=run_id,
                attempt=attempt,
                plugin_id=plugin.plugin_id,
                plugin_version=str(plugin.raw["version"]),
                mode=mode,
                backend=backend,
                plan_digest=plan["plan_digest"],
                inputs=node.get("inputs", {}),
                parameters=node.get("parameters", {}),
                artifacts=artifacts,
                metrics=metrics,
                command=argv,
                resources=node.get("resources", {}),
                state=final.value,
                state_reason=reason,
                **_manifest_context(project, node, plugin, store, run_id, finished=True),
            )
            write_json_atomic(manifest_path, manifest)
            for artifact in artifacts:
                store.add_artifact(
                    run_id,
                    str(artifact.get("role", "output")),
                    str(artifact["uri"]),
                    artifact.get("fingerprint"),
                    artifact.get("size_bytes"),
                    artifact.get("metadata", {}),
                )
            updated = store.transition(
                run_id,
                final,
                manifest_path=str(manifest_path),
                diagnostic=None if final == RunState.OK else reason,
            )
            return {"plan_digest": plan["plan_digest"], "step": updated.to_dict(), "result": result_data}
        completion_context_path = directory / "completion-context.json"
        write_json_atomic(
            completion_context_path,
            {
                "schema_version": 1,
                "project_id": project.project_id,
                "node_id": node_id,
                "run_id": run_id,
                "attempt": attempt,
                "plugin_id": plugin.plugin_id,
                "plan_digest": plan["plan_digest"],
            },
        )
        adapter_plan = plan.get("adapter_plan")
        command: list[str]
        if isinstance(adapter_plan, dict):
            if backend != "ssh-slurm":
                raise BackendError(
                    "scheduled adapters currently require the controlled ssh-slurm contract"
                )
            scheduled_transport = plan.get("scheduled_transport")
            if not isinstance(scheduled_transport, dict) or not isinstance(
                scheduled_transport.get("remote_cwd"), str
            ):
                raise BackendError("approved plan lacks an exact remote directory")
            remote_dir = str(scheduled_transport["remote_cwd"])
            result, observed_remote_dir, command = _stage_and_submit_scheduled_adapter(
                project, node, plugin, plan, directory
            )
            if observed_remote_dir != remote_dir:
                raise BackendError("staged remote directory differs from the approved target")
        else:
            script_value = node.get("parameters", {}).get("submit_script")
            if not isinstance(script_value, str):
                raise ConfigError(f"node {node_id} requires parameters.submit_script")
            script_ref = script_value
            command = [backend, script_ref]
            if backend == "slurm":
                script = resolve_reference(script_ref, project.root)
                result = SlurmBackend().submit(script, project.root)
                remote_dir = str(project.root)
            elif backend == "ssh-slurm":
                profile = _ssh_profile(project, node)
                remote_cwd = node.get("parameters", {}).get("remote_cwd")
                if not isinstance(remote_cwd, str):
                    raise ConfigError(f"node {node_id} requires parameters.remote_cwd")
                result = SshSlurmBackend(profile).submit(script_ref, remote_cwd)
                remote_dir = remote_cwd
            else:
                raise BackendError(f"unsupported backend: {backend}")
        if result.returncode != 0 or result.job_id is None:
            raise BackendError(result.stderr or result.stdout or "scheduler submission failed")
        store.transition(
            run_id, RunState.SUBMITTED, job_id=result.job_id, remote_dir=remote_dir
        )
        updated = store.transition(
            run_id, RunState.PENDING, job_id=result.job_id, remote_dir=remote_dir
        )
        control_artifacts: list[dict[str, Any]] = []
        if isinstance(adapter_plan, dict):
            for path, role in (
                (directory / "approved-plan.json", "approved-plan"),
                (directory / _REMOTE_COMMAND_NAME, "scheduler-command"),
                (directory / _REMOTE_SCRIPT_NAME, "scheduler-script"),
                (directory / _REMOTE_RUNNER_NAME, "scheduler-runner"),
            ):
                control_artifacts.append(fingerprint(path) | {"role": role})
        manifest = run_manifest(
            project_id=project.project_id,
            node_id=node_id,
            run_id=run_id,
            attempt=attempt,
            plugin_id=plugin.plugin_id,
            plugin_version=str(plugin.raw["version"]),
            mode=mode,
            backend=backend,
            plan_digest=plan["plan_digest"],
            inputs=node.get("inputs", {}),
            parameters=node.get("parameters", {}),
            artifacts=control_artifacts,
            state=RunState.PENDING.value,
            state_reason="submitted to scheduler",
            job={
                "job_id": result.job_id,
                "scheduler_state": RunState.PENDING.value,
                "scheduler_info": None,
            },
            remote_dir=remote_dir,
            command=command,
            resources=node.get("resources", {}),
            **_manifest_context(project, node, plugin, store, run_id, finished=False),
        )
        write_json_atomic(manifest_path, manifest)
        for artifact in control_artifacts:
            store.add_artifact(
                run_id,
                str(artifact["role"]),
                str(artifact["uri"]),
                artifact.get("fingerprint"),
                artifact.get("size_bytes"),
                artifact.get("metadata", {}),
            )
        updated = store.transition(
            run_id,
            RunState.PENDING,
            job_id=result.job_id,
            remote_dir=remote_dir,
            manifest_path=str(manifest_path),
        )
        return {"plan_digest": plan["plan_digest"], "step": updated.to_dict()}
    except Exception as exc:
        try:
            current = store.step_by_run_id(run_id)
            if RunState(current.state) in {
                RunState.READY,
                RunState.SUBMITTED,
                RunState.PENDING,
                RunState.RUNNING,
            }:
                remote = locals().get("remote_dir")
                store.transition(
                    run_id,
                    RunState.FAIL,
                    diagnostic=str(exc),
                    remote_dir=remote if isinstance(remote, str) else None,
                )
        except Exception:
            pass
        raise


def _replay(project: Project, node: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reference = node.get("inputs", {}).get("result_manifest")
    if reference is None:
        raise ConfigError(f"replay node {node['id']} requires inputs.result_manifest")
    result_path = _project_scoped_result_path(
        project, resolve_reference(reference, project.root)
    )
    return _load_result(result_path)


def _project_scoped_result_path(project: Project, path: Path) -> Path:
    if path.is_symlink():
        raise ConfigError(f"result manifest must not be a symlink: {path}")
    resolved = path.resolve()
    try:
        resolved.relative_to(project.root.resolve())
    except ValueError as exc:
        raise ConfigError(f"result manifest escapes the project root: {path}") from exc
    return resolved


def _load_result(
    result_path: Path, expected_identity: dict[str, Any] | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = load_mapping(result_path)
    if result.get("schema_version") != 1:
        raise ConfigError(f"unsupported replay result schema in {result_path}")
    scientific_state = result.get("state", result.get("status"))
    if not isinstance(scientific_state, str):
        raise ConfigError(f"result manifest has no explicit state/status in {result_path}")
    if scientific_state.upper() != RunState.OK.value:
        raise ConfigError(
            f"result manifest is not scientifically successful: {scientific_state!r}"
        )
    if expected_identity is not None:
        for key, expected in expected_identity.items():
            if result.get(key) != expected:
                raise ConfigError(
                    f"result identity mismatch for {key}: expected {expected!r}, "
                    f"got {result.get(key)!r}"
                )
    raw_artifacts = result.get("artifacts", [])
    if not isinstance(raw_artifacts, list):
        raise ConfigError(f"replay artifacts must be a list in {result_path}")
    artifacts: list[dict[str, Any]] = []
    for item in raw_artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ConfigError(f"invalid replay artifact in {result_path}: {item!r}")
        portable = Path(item["path"])
        if portable.is_absolute() or ".." in portable.parts:
            raise ConfigError(
                f"result artifact must be portable and relative to its manifest: {portable}"
            )
        path = result_path.parent / portable
        if path.is_symlink():
            raise ConfigError(f"result artifact must not be a symlink: {path}")
        path = path.resolve()
        try:
            path.relative_to(result_path.parent.resolve())
        except ValueError as exc:
            raise ConfigError(f"result artifact escapes its manifest directory: {path}") from exc
        if not path.is_file():
            raise ConfigError(f"replay artifact does not exist: {path}")
        artifacts.append({**fingerprint(path), "role": item.get("role", "output")})
    return result, artifacts


_SLURM_RESOURCE_FLAGS = {
    "partition": "partition",
    "account": "account",
    "qos": "qos",
    "nodes": "nodes",
    "ntasks": "ntasks",
    "cpus_per_task": "cpus-per-task",
    "gpus": "gpus",
    "gres": "gres",
    "time": "time",
    "memory": "mem",
    "mem": "mem",
    "constraint": "constraint",
}
_SAFE_SLURM_VALUE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+/@%=-]*")
_SAFE_REMOTE_ROOT = re.compile(r"[A-Za-z0-9_./+\-]+")
_SAFE_REMOTE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+\-]*")
_SAFE_REMOTE_EXECUTABLE = re.compile(r"(?:/[A-Za-z0-9_./+\-]+|[A-Za-z0-9][A-Za-z0-9._+\-]*)")
_SAFE_MODULE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+\-/]*")
_REMOTE_RUNNER_NAME = "mlipflow-remote-runner.py"
_REMOTE_COMMAND_NAME = "mlipflow-command.json"
_REMOTE_SCRIPT_NAME = "run.slurm"


def _materialize_slurm_adapter(
    attempt_dir: Path,
    adapter_plan: dict[str, Any],
    resources: Any,
    node_id: str,
    environment: dict[str, str] | None = None,
) -> tuple[Path, list[str]]:
    """Materialize a fixed scheduler wrapper around an already approved argv.

    User argv is stored in JSON and is never interpolated into the shell script.
    Only a small, newline-free SLURM resource allowlist is rendered as directives.
    """

    argv = adapter_plan.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise PluginError("adapter SLURM plan must provide argv as a string list")
    validate_argv(argv)
    planned_cwd = adapter_plan.get("cwd")
    if planned_cwd is not None:
        if not isinstance(planned_cwd, str):
            raise PluginError("adapter plan cwd must be a string")
        if Path(planned_cwd).resolve() != attempt_dir.resolve():
            raise PluginError("adapter SLURM cwd must be the fresh attempt directory")
    if not isinstance(resources, dict):
        raise ConfigError("SLURM resources must be a mapping")
    unknown = sorted(set(resources) - set(_SLURM_RESOURCE_FLAGS) - {"exclusive"})
    if unknown:
        raise ConfigError(f"unsupported SLURM resource keys: {', '.join(unknown)}")

    command_file = attempt_dir / "command.json"
    write_json_atomic(
        command_file,
        {
            "schema_version": 1,
            "argv": argv,
            "cwd": str(attempt_dir),
            "environment": environment or {},
        },
    )
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "-", node_id).strip("-.") or "mlipflow"
    directives = [
        f"#SBATCH --job-name={safe_name[:64]}",
        "#SBATCH --output=slurm-%j.out",
        "#SBATCH --error=slurm-%j.err",
    ]
    for key, flag in _SLURM_RESOURCE_FLAGS.items():
        if key not in resources:
            continue
        value = resources[key]
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ConfigError(f"SLURM resource {key} must be a string or integer")
        rendered = str(value)
        if not _SAFE_SLURM_VALUE.fullmatch(rendered):
            raise ConfigError(f"unsafe SLURM resource value for {key}: {rendered!r}")
        directives.append(f"#SBATCH --{flag}={rendered}")
    exclusive = resources.get("exclusive", False)
    if not isinstance(exclusive, bool):
        raise ConfigError("SLURM resource exclusive must be boolean")
    if exclusive:
        directives.append("#SBATCH --exclusive")
    runner = (
        "exec "
        + shlex.quote(sys.executable)
        + " -m mlipflow.runner "
        + shlex.quote(str(command_file))
    )
    script = attempt_dir / "run.slurm"
    write_text_atomic(
        script,
        "\n".join(["#!/bin/sh", *directives, "set -eu", "umask 077", runner, ""]),
    )
    return script, list(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _scheduled_transport_plan(project: Project, node: dict[str, Any]) -> dict[str, Any]:
    profile_name = node.get("backend_profile")
    profiles = project.raw.get("backend_profiles", {})
    configuration = profiles.get(profile_name) if isinstance(profiles, dict) else None
    if not isinstance(configuration, dict):
        raise ConfigError(
            f"node {node['id']} requires an object backend profile with ssh_profile and remote_root"
        )
    profile = configuration.get("ssh_profile")
    remote_root = configuration.get("remote_root")
    if not isinstance(profile, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", profile):
        raise ConfigError(f"node {node['id']} has an invalid SSH profile alias")
    if (
        not isinstance(remote_root, str)
        or not _SAFE_REMOTE_ROOT.fullmatch(remote_root)
        or ".." in Path(remote_root).parts
    ):
        raise ConfigError(
            f"node {node['id']} remote_root must be an existing safe path without '..'"
        )
    database = state_path(project)
    if not database.is_file():
        raise StateError("project must be initialized before planning remote staging")
    with StateStore(database, readonly=True) as store:
        step = store.latest_step(project.project_id, str(node["id"]))
    prefix = f"mlipflow-{project.project_id}-{node['id']}-a{step.attempt}"
    safe_prefix = re.sub(r"[^A-Za-z0-9._+-]", "-", prefix).strip("-.")[:96]
    run_name = f"{safe_prefix}-{step.run_id[:12]}"
    remote_cwd = run_name if remote_root == "." else f"{remote_root.rstrip('/')}/{run_name}"
    remote_runner = Path(__file__).with_name("remote_runner.py")
    if remote_runner.is_symlink() or not remote_runner.is_file():
        raise BackendError("self-contained remote runner is missing")
    return {
        "ssh_profile": profile,
        "remote_root": remote_root,
        "run_name": run_name,
        "remote_cwd": remote_cwd,
        "remote_runner_sha256": _sha256_file(remote_runner),
        "fresh_directory_required": True,
        "overwrite": False,
    }


def _scheduled_contract(
    project: Project,
    plugin: PluginSpec,
    plan: dict[str, Any],
) -> dict[str, Any]:
    adapter_plan = plan.get("adapter_plan")
    scheduled = adapter_plan.get("scheduled_execution") if isinstance(adapter_plan, dict) else None
    if not isinstance(scheduled, dict) or scheduled.get("schema_version") != 1:
        raise PluginError("scheduled adapter plan requires scheduled_execution schema_version=1")
    remote_argv = scheduled.get("remote_argv")
    if not isinstance(remote_argv, list) or not all(isinstance(item, str) for item in remote_argv):
        raise PluginError("scheduled_execution.remote_argv must be a string list")
    validate_argv(remote_argv)
    remote_python = scheduled.get("remote_python")
    if not isinstance(remote_python, str) or not _SAFE_REMOTE_EXECUTABLE.fullmatch(remote_python):
        raise PluginError("scheduled_execution.remote_python is unsafe")
    if ".." in Path(remote_python).parts:
        raise PluginError("scheduled_execution.remote_python must not contain '..'")
    modules = scheduled.get("modules", [])
    if not isinstance(modules, list) or not all(
        isinstance(item, str) and _SAFE_MODULE.fullmatch(item) for item in modules
    ):
        raise PluginError("scheduled_execution.modules contains an unsafe module name")
    if len(set(modules)) != len(modules):
        raise PluginError("scheduled_execution.modules must be unique")
    staged = scheduled.get("staged_files")
    if not isinstance(staged, list) or not staged:
        raise PluginError("scheduled_execution.staged_files must be a non-empty list")
    allowed_roots = (project.root.resolve(), plugin.path.parent.resolve())
    names: set[str] = {_REMOTE_RUNNER_NAME, _REMOTE_COMMAND_NAME, _REMOTE_SCRIPT_NAME}
    normalized_stage: list[dict[str, Any]] = []
    for index, item in enumerate(staged):
        if not isinstance(item, dict):
            raise PluginError(f"scheduled staged file {index} must be a mapping")
        source_value = item.get("source")
        remote_name = item.get("remote_name")
        declared = item.get("sha256")
        size_bytes = item.get("size_bytes")
        if not isinstance(source_value, str) or not isinstance(remote_name, str):
            raise PluginError(f"scheduled staged file {index} lacks source/remote_name")
        if not _SAFE_REMOTE_NAME.fullmatch(remote_name) or remote_name in names:
            raise PluginError(f"unsafe or duplicate remote staging name: {remote_name!r}")
        names.add(remote_name)
        source = Path(source_value).expanduser().absolute()
        if source.is_symlink() or not source.is_file():
            raise PluginError(f"staging source must be an ordinary file: {source}")
        resolved = source.resolve()
        if not any(_is_within(resolved, root) for root in allowed_roots):
            raise PluginError(f"staging source is outside the project/plugin roots: {source}")
        if not isinstance(declared, str) or declared != _sha256_file(source):
            raise PluginError(f"staging source fingerprint changed: {source}")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes != source.stat().st_size:
            raise PluginError(f"staging source size changed: {source}")
        normalized_stage.append({**item, "source": str(source), "remote_name": remote_name})
    outputs = scheduled.get("fetch_outputs")
    if not isinstance(outputs, list) or not outputs:
        raise PluginError("scheduled_execution.fetch_outputs must be a non-empty list")
    output_names: set[str] = set()
    normalized_outputs: list[dict[str, Any]] = []
    for index, item in enumerate(outputs):
        if not isinstance(item, dict):
            raise PluginError(f"scheduled fetch output {index} must be a mapping")
        remote_name = item.get("remote_name")
        local_name = item.get("local_name")
        required = item.get("required")
        maximum = item.get("max_bytes")
        if (
            not isinstance(remote_name, str)
            or not _SAFE_REMOTE_NAME.fullmatch(remote_name)
            or remote_name in output_names
            or not isinstance(local_name, str)
            or not _SAFE_REMOTE_NAME.fullmatch(local_name)
            or type(required) is not bool
            or isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum < 1
        ):
            raise PluginError(f"invalid scheduled fetch output {index}")
        output_names.add(remote_name)
        normalized_outputs.append(dict(item))
    return {
        **scheduled,
        "remote_argv": list(remote_argv),
        "remote_python": remote_python,
        "modules": list(modules),
        "staged_files": normalized_stage,
        "fetch_outputs": normalized_outputs,
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _materialize_remote_slurm_adapter(
    attempt_dir: Path,
    scheduled: dict[str, Any],
    resources: Any,
    node_id: str,
) -> tuple[Path, Path, Path]:
    """Create a self-contained scheduler control bundle for remote staging."""

    command_file = attempt_dir / _REMOTE_COMMAND_NAME
    write_json_atomic(
        command_file,
        {
            "schema_version": 1,
            "argv": scheduled["remote_argv"],
            "cwd": ".",
            "environment": {},
        },
    )
    runner_source = Path(__file__).with_name("remote_runner.py")
    if runner_source.is_symlink() or not runner_source.is_file():
        raise BackendError("self-contained remote runner is missing")
    runner = attempt_dir / _REMOTE_RUNNER_NAME
    shutil.copy2(runner_source, runner)
    if not isinstance(resources, dict):
        raise ConfigError("SLURM resources must be a mapping")
    unknown = sorted(set(resources) - set(_SLURM_RESOURCE_FLAGS) - {"exclusive"})
    if unknown:
        raise ConfigError(f"unsupported SLURM resource keys: {', '.join(unknown)}")
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "-", node_id).strip("-.") or "mlipflow"
    directives = [
        f"#SBATCH --job-name={safe_name[:64]}",
        "#SBATCH --output=slurm.out",
        "#SBATCH --error=slurm.err",
    ]
    for key, flag in _SLURM_RESOURCE_FLAGS.items():
        if key not in resources:
            continue
        value = resources[key]
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ConfigError(f"SLURM resource {key} must be a string or integer")
        rendered = str(value)
        if not _SAFE_SLURM_VALUE.fullmatch(rendered):
            raise ConfigError(f"unsafe SLURM resource value for {key}: {rendered!r}")
        directives.append(f"#SBATCH --{flag}={rendered}")
    exclusive = resources.get("exclusive", False)
    if not isinstance(exclusive, bool):
        raise ConfigError("SLURM resource exclusive must be boolean")
    if exclusive:
        directives.append("#SBATCH --exclusive")
    modules = [f"module load {shlex.quote(item)}" for item in scheduled["modules"]]
    runner_command = "exec " + " ".join(
        shlex.quote(item)
        for item in [
            scheduled["remote_python"],
            _REMOTE_RUNNER_NAME,
            _REMOTE_COMMAND_NAME,
        ]
    )
    script = attempt_dir / _REMOTE_SCRIPT_NAME
    write_text_atomic(
        script,
        "\n".join(
            [
                "#!/bin/bash -l",
                *directives,
                "set -euo pipefail",
                "umask 077",
                *modules,
                runner_command,
                "",
            ]
        ),
    )
    return script, command_file, runner


def _stage_and_submit_scheduled_adapter(
    project: Project,
    node: dict[str, Any],
    plugin: PluginSpec,
    plan: dict[str, Any],
    attempt_dir: Path,
) -> tuple[Any, str, list[str]]:
    scheduled = _scheduled_contract(project, plugin, plan)
    transport = plan.get("scheduled_transport")
    if not isinstance(transport, dict):
        raise BackendError("approved plan lacks scheduled transport identity")
    expected_transport = _scheduled_transport_plan(project, node)
    if transport != expected_transport:
        raise BackendError("scheduled transport target changed after approval")
    approved_plan = attempt_dir / "approved-plan.json"
    write_json_atomic(approved_plan, plan)
    script, command_file, runner = _materialize_remote_slurm_adapter(
        attempt_dir, scheduled, node.get("resources", {}), str(node["id"])
    )
    files: list[tuple[Path, str, str]] = [
        (
            Path(str(item["source"])),
            str(item["remote_name"]),
            str(item["sha256"]),
        )
        for item in scheduled["staged_files"]
    ]
    files.extend(
        (path, name, _sha256_file(path))
        for path, name in (
            (script, _REMOTE_SCRIPT_NAME),
            (command_file, _REMOTE_COMMAND_NAME),
            (runner, _REMOTE_RUNNER_NAME),
        )
    )
    backend = SshSlurmBackend(str(transport["ssh_profile"]))
    remote_cwd = backend.stage_fresh(
        str(transport["remote_root"]), str(transport["run_name"]), files
    )
    if remote_cwd != transport["remote_cwd"]:
        raise BackendError("backend returned an unexpected remote directory")
    result = backend.submit(_REMOTE_SCRIPT_NAME, remote_cwd)
    return result, remote_cwd, list(scheduled["remote_argv"])


def _scheduled_completion_path(
    project: Project, node: dict[str, Any], attempt: int
) -> Path | None:
    parameters = node.get("parameters", {})
    inputs = node.get("inputs", {})
    explicit = parameters.get("completion_manifest") if isinstance(parameters, dict) else None
    if explicit is not None:
        try:
            return resolve_reference(explicit, project.root)
        except ValueError:
            return None
    result_reference: Any = None
    if isinstance(parameters, dict):
        result_reference = parameters.get("result_manifest")
    if result_reference is None and isinstance(inputs, dict):
        result_reference = inputs.get("result_manifest")
    if result_reference is None:
        return None
    attempt_dir = project.root / ".mlipflow" / "runs" / str(node["id"]) / f"attempt-{attempt}"
    try:
        return resolve_reference(result_reference, attempt_dir)
    except ValueError:
        return None


def _scheduler_expected_identity(step: Any) -> dict[str, Any]:
    if not isinstance(step.manifest_path, str):
        raise ConfigError("scheduled run has no initial run manifest")
    initial_path = Path(step.manifest_path)
    if not initial_path.is_file():
        raise ConfigError(f"scheduled run manifest does not exist: {initial_path}")
    initial = load_mapping(initial_path)
    plugin = initial.get("plugin")
    provenance = initial.get("provenance")
    plugin_id = plugin.get("id") if isinstance(plugin, dict) else None
    plan_digest = provenance.get("plan_digest") if isinstance(provenance, dict) else None
    if not isinstance(plugin_id, str) or not isinstance(plan_digest, str):
        raise ConfigError("scheduled run manifest lacks plugin/plan identity")
    return {
        "project_id": step.project_id,
        "node_id": step.node_id,
        "run_id": step.run_id,
        "attempt": step.attempt,
        "plugin_id": plugin_id,
        "plan_digest": plan_digest,
    }


def _observe_scheduled_step(
    project: Project, step: Any, plugin_root: Path | None = None
) -> dict[str, Any]:
    node = project.node(step.node_id)
    try:
        if step.backend == "slurm":
            scheduler = SlurmBackend().status(str(step.job_id))
        elif step.backend == "ssh-slurm":
            scheduler = SshSlurmBackend(_ssh_profile(project, node)).status(str(step.job_id))
        else:
            return {
                "node_id": step.node_id,
                "job_id": step.job_id,
                "scheduler_state": "UNKNOWN",
                "reason": f"backend {step.backend} has no asynchronous scheduler",
            }
    except Exception as exc:
        return {
            "node_id": step.node_id,
            "job_id": step.job_id,
            "scheduler_state": "QUERY_ERROR",
            "reason": str(exc),
        }
    raw_state = str(scheduler.get("state", "UNKNOWN")).upper().split()[0]
    observation: dict[str, Any] = {
        "node_id": step.node_id,
        "job_id": step.job_id,
        "scheduler_state": raw_state,
        "scheduler_detail": scheduler.get("detail"),
    }
    if raw_state in {"PENDING", "CONFIGURING", "RESIZING", "SUSPENDED"}:
        if step.state == RunState.RUNNING.value:
            observation["reason"] = (
                f"scheduler regressed from persisted RUNNING to {raw_state}; no automatic rollback"
            )
        else:
            observation["target_state"] = RunState.PENDING.value
            observation["reason"] = f"scheduler reports {raw_state}"
    elif raw_state in {"RUNNING", "COMPLETING"}:
        observation["target_state"] = RunState.RUNNING.value
        observation["reason"] = f"scheduler reports {raw_state}"
    elif raw_state in {
        "FAILED",
        "TIMEOUT",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
    }:
        observation["target_state"] = RunState.FAIL.value
        observation["reason"] = f"scheduler terminal failure: {raw_state}"
    elif raw_state in {"CANCELLED", "REVOKED"}:
        observation["target_state"] = RunState.STOPPED.value
        observation["reason"] = f"scheduler reports cancellation: {raw_state}"
    elif raw_state == "COMPLETED":
        approved_plan_path = (
            project.root
            / ".mlipflow"
            / "runs"
            / step.node_id
            / f"attempt-{step.attempt}"
            / "approved-plan.json"
        )
        if approved_plan_path.is_file():
            if plugin_root is None:
                observation["reason"] = (
                    "scheduler completed; plugin root is required for pinned adapter finalization"
                )
                return observation
            try:
                approved_plan, plugin, scheduled = _load_pinned_scheduled_plan(
                    project, step, plugin_root
                )
                transport = approved_plan.get("scheduled_transport")
                if not isinstance(transport, dict) or step.remote_dir != transport.get("remote_cwd"):
                    raise BackendError("persisted remote directory differs from the approved plan")
                backend = SshSlurmBackend(str(transport["ssh_profile"]))
                inventory: list[dict[str, Any]] = []
                for item in scheduled["fetch_outputs"]:
                    observed = backend.inspect_file(
                        str(transport["remote_cwd"]), str(item["remote_name"])
                    )
                    record = {**item, **observed}
                    if observed.get("exists") and int(observed.get("size_bytes", 0)) > int(
                        item["max_bytes"]
                    ):
                        record["oversized"] = True
                    inventory.append(record)
                observation["adapter_finalization"] = {
                    "approved_plan_fingerprint": fingerprint(approved_plan_path),
                    "plugin_id": plugin.plugin_id,
                    "remote_cwd": transport["remote_cwd"],
                    "outputs": inventory,
                }
                missing = [
                    str(item["remote_name"])
                    for item in inventory
                    if item["required"] and not item.get("exists")
                ]
                oversized = [
                    str(item["remote_name"])
                    for item in inventory
                    if item.get("oversized")
                ]
                if missing or oversized:
                    details = []
                    if missing:
                        details.append("missing required outputs: " + ", ".join(missing))
                    if oversized:
                        details.append("oversized outputs: " + ", ".join(oversized))
                    observation["reason"] = (
                        "scheduler completed; approved fetch/check will fail safely ("
                        + "; ".join(details)
                        + ")"
                    )
                else:
                    observation["reason"] = (
                        "scheduler completed; approved outputs are ready for bounded fetch and "
                        "the pinned plugin scientific checker"
                    )
                return observation
            except Exception as exc:
                observation["target_state"] = RunState.FAIL.value
                observation["reason"] = f"pinned scheduled adapter contract is invalid: {exc}"
                return observation
        completion_path = _scheduled_completion_path(project, node, step.attempt)
        if completion_path is None:
            observation["reason"] = (
                "scheduler completed; no completion_manifest or result_manifest is configured, "
                "so scientific OK is withheld"
            )
            return observation
        if completion_path is not None:
            try:
                completion_path = _project_scoped_result_path(project, completion_path)
            except ConfigError as exc:
                observation["target_state"] = RunState.FAIL.value
                observation["reason"] = str(exc)
                return observation
        observation["completion_manifest"] = str(completion_path)
        if not completion_path.is_file():
            observation["reason"] = (
                f"scheduler completed; awaiting completion manifest {completion_path}"
            )
            return observation
        observation["completion_fingerprint"] = fingerprint(completion_path)
        try:
            result, _ = _load_result(
                completion_path, _scheduler_expected_identity(step)
            )
        except Exception as exc:
            observation["target_state"] = RunState.FAIL.value
            observation["reason"] = f"completion manifest is invalid: {exc}"
            return observation
        scientific_state = str(result.get("state", result.get("status", ""))).upper()
        if scientific_state == RunState.OK.value:
            observation["target_state"] = RunState.OK.value
            observation["reason"] = "scheduler completed and scientific completion manifest is OK"
        else:
            observation["target_state"] = RunState.FAIL.value
            observation["reason"] = (
                f"scheduler completed but scientific completion state is {scientific_state or 'missing'}"
            )
    else:
        observation["reason"] = f"unrecognized scheduler state {raw_state}; no state change"
    return observation


def _load_pinned_scheduled_plan(
    project: Project, step: Any, plugin_root: Path
) -> tuple[dict[str, Any], PluginSpec, dict[str, Any]]:
    attempt_dir = (
        project.root / ".mlipflow" / "runs" / step.node_id / f"attempt-{step.attempt}"
    )
    plan_path = attempt_dir / "approved-plan.json"
    if plan_path.is_symlink() or not plan_path.is_file():
        raise ConfigError("approved scheduled plan is missing or is a symlink")
    plan = load_mapping(plan_path)
    expected_digest = with_digest(
        {key: value for key, value in plan.items() if key != "plan_digest"}
    )["plan_digest"]
    if plan.get("plan_digest") != expected_digest:
        raise ApprovalError("approved scheduled plan digest is invalid")
    identity = _scheduler_expected_identity(step)
    if plan.get("plan_digest") != identity["plan_digest"]:
        raise ApprovalError("approved plan differs from the initial run manifest")
    node = project.node(step.node_id)
    plugin = select_plugin(discover_plugins(plugin_root), str(node["uses"]))
    current = node_plan(project, node, plugin)
    for key in (
        "project_id",
        "node_id",
        "plugin",
        "backend",
        "backend_profile",
        "project_config_digest",
        "inputs",
        "input_fingerprints",
        "parameters",
        "resources",
    ):
        if plan.get(key) != current.get(key):
            raise ApprovalError(f"pinned scheduled plan field changed: {key}")
    scheduled = _scheduled_contract(project, plugin, plan)
    return plan, plugin, scheduled


def _finalize_scheduled_adapter(
    project: Project,
    step: Any,
    change: dict[str, Any],
    plugin_root: Path,
    store: StateStore,
) -> Any:
    details = change.get("adapter_finalization")
    if not isinstance(details, dict):
        raise StateError("approved transition lacks adapter finalization details")
    plan, plugin, scheduled = _load_pinned_scheduled_plan(project, step, plugin_root)
    attempt_dir = (
        project.root / ".mlipflow" / "runs" / step.node_id / f"attempt-{step.attempt}"
    )
    plan_path = attempt_dir / "approved-plan.json"
    if fingerprint(plan_path) != details.get("approved_plan_fingerprint"):
        raise StateError("approved scheduled plan changed after advance dry-run")
    transport = plan.get("scheduled_transport")
    if not isinstance(transport, dict) or details.get("remote_cwd") != transport.get("remote_cwd"):
        raise StateError("approved remote directory changed before fetch")
    approved_inventory = details.get("outputs")
    if not isinstance(approved_inventory, list):
        raise StateError("advance plan lacks the approved remote output inventory")
    backend = SshSlurmBackend(str(transport["ssh_profile"]))
    current_inventory: list[dict[str, Any]] = []
    for item in scheduled["fetch_outputs"]:
        observed = backend.inspect_file(
            str(transport["remote_cwd"]), str(item["remote_name"])
        )
        record = {**item, **observed}
        if observed.get("exists") and int(observed.get("size_bytes", 0)) > int(
            item["max_bytes"]
        ):
            record["oversized"] = True
        current_inventory.append(record)
    if current_inventory != approved_inventory:
        raise StateError("remote outputs changed after the approved advance plan")
    fetched: list[dict[str, Any]] = []
    fetch_errors: list[str] = []
    for item in current_inventory:
        if not item.get("exists"):
            if item["required"]:
                fetch_errors.append(f"required output is missing: {item['remote_name']}")
            continue
        if item.get("oversized"):
            fetch_errors.append(f"output exceeds approved bound: {item['remote_name']}")
            continue
        destination = attempt_dir / str(item["local_name"])
        backend.fetch_from(
            str(transport["remote_cwd"]), str(item["remote_name"]), destination
        )
        if destination.stat().st_size != item["size_bytes"] or _sha256_file(destination) != item["sha256"]:
            raise StateError(f"fetched output fingerprint mismatch: {item['remote_name']}")
        fetched.append(fingerprint(destination) | {"role": "scheduler-output"})
    node = project.node(step.node_id)
    context = _adapter_context(project, node, step.attempt)
    context["execution"] = {
        "returncode": 0,
        "scheduler_state": "COMPLETED",
        "remote_output_inventory": current_inventory,
        "plan": plan["adapter_plan"],
    }
    adapter = load_adapter(plugin)
    checked = adapter.check(context)
    collected: dict[str, Any] | None = None
    metrics: dict[str, Any] = {}
    reason: str
    if fetch_errors:
        final = RunState.FAIL
        reason = "; ".join(fetch_errors)
    elif not isinstance(checked, dict) or checked.get("status") != RunState.OK.value:
        final = RunState.FAIL
        reason = f"pinned plugin completion check did not return OK: {checked}"
    else:
        collected = adapter.collect(context)
        if not isinstance(collected, dict) or collected.get("status") != RunState.OK.value:
            final = RunState.FAIL
            reason = f"pinned plugin collection did not return OK: {collected}"
        else:
            normalized = _normalize_adapter_artifacts(
                project, attempt_dir, collected.get("artifacts", [])
            )
            known = {str(item["uri"]) for item in fetched}
            fetched.extend(item for item in normalized if str(item["uri"]) not in known)
            raw_metrics = collected.get("metrics", {})
            metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
            final = RunState.OK
            reason = "scheduler completed; bounded fetch and pinned plugin checks succeeded"
    initial_artifacts = store.artifacts(step.run_id)
    artifacts = [
        {
            **item,
            "fingerprint_mode": (
                "full"
                if isinstance(item.get("fingerprint"), str)
                and str(item["fingerprint"]).startswith("sha256:")
                else "metadata"
            ),
            "mtime_ns": None,
        }
        for item in initial_artifacts
    ]
    known_uris = {str(item["uri"]) for item in artifacts}
    artifacts.extend(item for item in fetched if str(item["uri"]) not in known_uris)
    current_state = RunState(step.state)
    if current_state in {RunState.SUBMITTED, RunState.PENDING}:
        step = store.transition(
            step.run_id,
            RunState.RUNNING,
            diagnostic="scheduler completed between approved observations",
        )
    for artifact in fetched:
        if str(artifact["uri"]) in known_uris:
            continue
        known_uris.add(str(artifact["uri"]))
        store.add_artifact(
            step.run_id,
            str(artifact["role"]),
            str(artifact["uri"]),
            artifact.get("fingerprint"),
            artifact.get("size_bytes"),
            artifact.get("metadata", {}),
        )
    final_manifest = _finalize_scheduler_manifest(
        project,
        step,
        final,
        reason,
        "COMPLETED",
        artifacts,
        metrics,
    )
    return store.transition(
        step.run_id,
        final,
        diagnostic=None if final == RunState.OK else reason,
        manifest_path=str(final_manifest) if final_manifest else None,
    )


def _finalize_scheduler_manifest(
    project: Project,
    step: Any,
    target: RunState,
    reason: str,
    scheduler_state: str,
    artifacts: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> Path | None:
    if not step.manifest_path:
        return None
    initial = Path(step.manifest_path)
    if not initial.is_file():
        return None
    manifest = load_mapping(initial)
    manifest["state"] = target.value
    manifest["state_reason"] = reason
    manifest["artifacts"] = artifacts or manifest.get("artifacts", [])
    manifest["metrics"] = metrics or manifest.get("metrics", {})
    manifest["timestamps"]["updated_at"] = utc_now()
    if target in {RunState.OK, RunState.FAIL, RunState.STOPPED}:
        manifest["timestamps"]["finished_at"] = utc_now()
    if isinstance(manifest.get("job"), dict):
        manifest["job"]["scheduler_state"] = scheduler_state
    destination = initial.with_name("run-manifest.final.json")
    write_json_atomic(destination, manifest)
    return destination


def _ssh_profile(project: Project, node: dict[str, Any]) -> str:
    profile_name = node.get("backend_profile")
    profiles = project.raw.get("backend_profiles", {})
    profile = profiles.get(profile_name) if isinstance(profiles, dict) else None
    if isinstance(profile, str):
        return profile
    if isinstance(profile, dict) and isinstance(profile.get("ssh_profile"), str):
        return profile["ssh_profile"]
    raise ConfigError(f"node {node['id']} has no valid SSH backend profile")


def _planned_attempt(project: Project, node_id: str) -> int:
    database = state_path(project)
    if not database.is_file():
        return 1
    with StateStore(database, readonly=True) as store:
        return store.latest_step(project.project_id, node_id).attempt


def _adapter_context(project: Project, node: dict[str, Any], attempt: int) -> dict[str, Any]:
    return {
        "project_root": str(project.root),
        "attempt_dir": str(
            project.root / ".mlipflow" / "runs" / str(node["id"]) / f"attempt-{attempt}"
        ),
        "inputs": node.get("inputs", {}),
        "parameters": node.get("parameters", {}),
        "backend": node.get("backend", "local"),
        "resources": node.get("resources", {}),
    }


def _adapter_command_fingerprints(
    project: Project, adapter_plan: Any
) -> dict[str, dict[str, Any]]:
    if not isinstance(adapter_plan, dict):
        return {}
    if isinstance(adapter_plan.get("scheduled_execution"), dict):
        # Remote argv refers only to staged basenames and remote executables.
        # Their local sources are exhaustively fingerprinted by staged_files;
        # resolving values such as "." against the controller cwd is incorrect.
        return {}
    argv = adapter_plan.get("argv")
    if not isinstance(argv, list):
        return {}
    raw_cwd = adapter_plan.get("cwd")
    base = Path(raw_cwd) if isinstance(raw_cwd, str) else project.root
    threshold = int(project.raw.get("fingerprints", {}).get("full_hash_max_bytes", 67108864))
    captured: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(argv):
        if not isinstance(raw, str) or not raw or "\x00" in raw:
            continue
        candidate = Path(raw)
        candidate = candidate if candidate.is_absolute() else base / candidate
        if candidate.exists():
            captured[str(index)] = fingerprint(
                candidate, full_hash_max_bytes=threshold
            )
    return captured


def _normalize_adapter_artifacts(
    project: Project, attempt_dir: Path, raw_artifacts: Any
) -> list[dict[str, Any]]:
    if not isinstance(raw_artifacts, list):
        raise PluginError("adapter artifacts must be a list")
    threshold = int(project.raw.get("fingerprints", {}).get("full_hash_max_bytes", 67108864))
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw_artifacts):
        if not isinstance(item, dict):
            raise PluginError(f"adapter artifact {index} must be a mapping")
        role = str(item.get("role", "output"))
        path_value = item.get("path")
        if isinstance(path_value, str):
            path = Path(path_value)
            path = path if path.is_absolute() else attempt_dir / path
            if path.is_symlink():
                raise PluginError(f"adapter artifact must not be a symlink: {path}")
            resolved = path.resolve()
            try:
                resolved.relative_to(attempt_dir.resolve())
            except ValueError as exc:
                raise PluginError(
                    f"adapter artifact escapes the fresh attempt directory: {path}"
                ) from exc
            if not resolved.is_file():
                raise PluginError(f"adapter artifact does not exist: {path}")
            artifact = fingerprint(resolved, full_hash_max_bytes=threshold)
            artifact["role"] = role
            if isinstance(item.get("media_type"), str):
                artifact["media_type"] = item["media_type"]
            normalized.append(artifact)
            continue
        uri = item.get("uri")
        external_fingerprint = item.get("fingerprint")
        if not isinstance(uri, str) or not isinstance(external_fingerprint, str):
            raise PluginError(
                f"adapter artifact {index} needs either path or uri plus fingerprint"
            )
        normalized.append(
            {
                "role": role,
                "uri": uri,
                "media_type": item.get("media_type"),
                "fingerprint": external_fingerprint,
                "fingerprint_mode": "external",
                "size_bytes": item.get("size_bytes"),
                "mtime_ns": None,
                "metadata": item.get("metadata", {}),
            }
        )
    return normalized


def _manifest_context(
    project: Project,
    node: dict[str, Any],
    plugin: PluginSpec,
    store: StateStore,
    run_id: str,
    *,
    finished: bool,
) -> dict[str, Any]:
    step = store.step_by_run_id(run_id)
    dependencies = []
    for dependency_id in node.get("needs", []):
        dependency = store.latest_step(project.project_id, dependency_id)
        dependencies.append(
            {
                "node_id": dependency_id,
                "run_id": dependency.run_id,
                "required_states": [RunState.OK.value],
            }
        )
    max_attempts = plugin.raw.get("retry", {}).get("max_attempts")
    max_retries = max(0, int(max_attempts) - 1) if isinstance(max_attempts, int) else None
    previous = store.previous_step(project.project_id, str(node["id"]), step.attempt)
    return {
        "backend_profile": node.get("backend_profile"),
        "timestamps": {
            "created_at": step.created_at,
            "updated_at": utc_now(),
            "submitted_at": step.submitted_at,
            "started_at": step.started_at,
            "finished_at": utc_now() if finished else None,
        },
        "retry_count": step.retry_count,
        "max_retries": max_retries,
        "previous_run_id": previous.run_id if previous is not None else None,
        "dependencies": dependencies,
        "source_root": str(project.root),
    }


def _require_approval(plan: dict[str, Any], approval: str) -> None:
    expected = plan["plan_digest"]
    if approval != expected:
        raise ApprovalError(f"approval digest mismatch; run --dry-run and approve {expected}")
