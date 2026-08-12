"""Mutating lifecycle commands: init, run, advance, retry, stop.

Every function here either returns a plan (dry-run) or requires the exact digest
of one.  The plan dictionaries produced by ``planning.node_plan`` and
``planning.action_plan`` are the approval contract, so their key set, nesting and
values must never change for cosmetic reasons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..artifacts import content_identity
from ..backends import SshSlurmBackend
from ..config import Project, load_project
from ..errors import ApprovalError, BackendError, PluginError, StateError
from ..hpc import TemplateLibrary, resolve_hpc_execution_plan
from ..io import write_json_atomic
from ..planning import action_plan, node_plan, resolve_reference, with_digest
from ..portable import to_portable
from ..plugins import discover_plugins, load_adapter, select_plugin
from ..site import load_site_config
from ..state import RunState, StateStore
from .backend_factory import SchedulerFactory, scheduler_for_node
from .contracts import (
    _adapter_command_identities,
    _adapter_context,
    _portable_roots,
    _load_result,
    _planned_attempt,
    _project_scoped_result_path,
    _scheduled_contract,
)
from .execution import _execute_ready
from .paths import _assert_state_matches_project, state_path
from .scheduled import (
    _finalize_scheduled_adapter,
    _finalize_scheduler_manifest,
    _observe_scheduled_step,
    _scheduler_expected_identity,
)


DEFAULT_PROJECT = {
    "schema_version": 1,
    "project": {
        "id": "my-mlip-project",
        "name": "My MLIP project",
        "description": "Edit this deterministic workflow before running it.",
    },
    "locations": {},
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


def make_run_plan(
    project: Project,
    node_id: str,
    plugin_root: Path,
    site_path: Path | None = None,
    template_library: TemplateLibrary | None = None,
) -> dict[str, Any]:
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
        attempt = _planned_attempt(project, node_id)
        context = _adapter_context(project, node, attempt)
        adapter = load_adapter(plugin)
        diagnostics = adapter.validate(context)
        adapter_plan = adapter.plan(context)
        unsigned = {key: value for key, value in plan.items() if key != "plan_digest"}
        # Adapters emit this machine's absolute paths in argv, cwd, staged
        # sources and diagnostic messages.  Approval must describe the work, not
        # the filesystem it was planned on, so those roots become tokens here and
        # are resolved again in ``_execute_ready`` just before anything runs.
        roots = _portable_roots(project, plugin, node_id, attempt)
        unsigned["adapter_diagnostics"] = to_portable(diagnostics, roots)
        unsigned["adapter_plan"] = to_portable(adapter_plan, roots)
        unsigned["adapter_command_identities"] = _adapter_command_identities(
            project, adapter_plan, plugin=plugin, node_id=node_id, attempt=attempt
        )
        if (
            str(node.get("backend", "local")) == "ssh-slurm"
            and isinstance(adapter_plan, dict)
            and isinstance(adapter_plan.get("scheduled_execution"), dict)
        ):
            _scheduled_contract(
                project,
                plugin,
                {"adapter_plan": adapter_plan},
                node_id=node_id,
                attempt=attempt,
            )
            site = load_site_config(site_path)
            profile = site.cluster(node.get("backend_profile"))
            provider: TemplateLibrary = (
                template_library
                if template_library is not None
                else SshSlurmBackend(profile.ssh_profile)
            )
            unsigned["site_config_digest"] = site.digest
            unsigned["hpc_execution"] = resolve_hpc_execution_plan(
                profile=profile,
                project_id=project.project_id,
                node_id=str(node["id"]),
                attempt=attempt,
                resources_value=node.get("resources"),
                scheduled_execution=adapter_plan["scheduled_execution"],
                library=provider,
            )
        plan = with_digest(unsigned)
    return plan


def run_node(
    project: Project,
    node_id: str,
    plugin_root: Path,
    approval: str,
    site_path: Path | None = None,
    template_library: TemplateLibrary | None = None,
    *,
    factory: SchedulerFactory | None = None,
) -> dict[str, Any]:
    node = project.node(node_id)
    plugins = discover_plugins(plugin_root)
    plugin = select_plugin(plugins, str(node["uses"]))
    plan = make_run_plan(
        project, node_id, plugin_root, site_path, template_library
    )
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
                plan.get("hpc_execution"), dict
            ):
                raise BackendError(
                    "ssh-slurm adapter execution requires templates, a site profile, and an "
                    "explicit staged execution contract"
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
            result = _execute_ready(
                project,
                node,
                plugin,
                plan,
                store,
                step.run_id,
                step.attempt,
                factory=factory,
            )
        finally:
            store.consume_intent(plan["plan_digest"])
    return result


def make_advance_plan(
    project: Project,
    plugin_root: Path | None = None,
    site_path: Path | None = None,
    *,
    factory: SchedulerFactory | None = None,
) -> dict[str, Any]:
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
                    observation = _observe_scheduled_step(
                        project, step, plugin_root, site_path, factory=factory
                    )
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
                                "completion_identity": observation.get(
                                    "completion_identity"
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
    project: Project,
    approval: str,
    plugin_root: Path | None = None,
    site_path: Path | None = None,
    *,
    factory: SchedulerFactory | None = None,
) -> dict[str, Any]:
    plan = make_advance_plan(project, plugin_root, site_path, factory=factory)
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
                        project, step, change, plugin_root, store, site_path, factory=factory
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
                approved_identity = change.get("completion_identity")
                current_identity = content_identity(result_path, root=project.root)
                if not isinstance(approved_identity, dict) or (
                    current_identity != approved_identity
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


def make_stop_plan(
    project: Project, node_id: str, site_path: Path | None = None
) -> dict[str, Any]:
    database = state_path(project)
    if not database.is_file():
        raise StateError("project has not been initialized")
    with StateStore(database, readonly=True) as store:
        _assert_state_matches_project(store, project)
        step = store.latest_step(project.project_id, node_id)
    scheduler_target: dict[str, Any] | None = None
    site_config_digest: str | None = None
    if step.backend == "ssh-slurm":
        site = load_site_config(site_path)
        profile = site.cluster(project.node(node_id).get("backend_profile"))
        scheduler_target = profile.to_plan_dict()
        site_config_digest = site.digest
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
            "site_config_digest": site_config_digest,
            "job_id": step.job_id,
        },
    )


def stop(
    project: Project,
    node_id: str,
    approval: str,
    site_path: Path | None = None,
    *,
    factory: SchedulerFactory | None = None,
) -> dict[str, Any]:
    plan = make_stop_plan(project, node_id, site_path)
    _require_approval(plan, approval)
    with StateStore(state_path(project), readonly=False) as store:
        _assert_state_matches_project(store, project)
        step = store.latest_step(project.project_id, node_id)
        state = RunState(step.state)
        if state in {RunState.OK, RunState.FAIL, RunState.STOPPED}:
            raise StateError(f"cannot stop node in terminal state {state.value}")
        if step.job_id:
            scheduler = scheduler_for_node(
                str(step.backend), project.node(node_id), site_path, factory=factory
            )
            result = scheduler.cancel(step.job_id)
            if result.returncode != 0:
                raise BackendError(result.stderr or result.stdout or "scheduler cancellation failed")
        elif state == RunState.RUNNING:
            raise BackendError("cannot safely stop a local run without a persisted process handle")
        updated = store.transition(step.run_id, RunState.STOPPED)
    return {"plan_digest": plan["plan_digest"], "step": updated.to_dict()}


def _require_approval(plan: dict[str, Any], approval: str) -> None:
    expected = plan["plan_digest"]
    if approval != expected:
        raise ApprovalError(f"approval digest mismatch; run --dry-run and approve {expected}")
