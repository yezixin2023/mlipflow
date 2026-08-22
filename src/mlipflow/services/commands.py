"""Mutating lifecycle commands: init, run, advance, retry, stop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..backends import SshSlurmBackend
from ..config import Project, load_project
from ..errors import ApprovalError, BackendError, PluginError, StateError
from ..hpc import TemplateLibrary, resolve_hpc_execution_plan
from ..io import load_mapping, write_json_atomic
from ..planning import action_plan, node_plan, resolve_reference
from ..portable import to_portable
from ..plugins import discover_plugins, load_adapter, select_plugin
from ..site import ClusterProfile, SiteConfig, load_site_config
from ..state import RunState, StateStore
from .backend_factory import (
    SchedulerFactory,
    scheduler_for_node,
    scheduler_from_cluster_record,
)
from .contracts import (
    _adapter_context,
    _portable_roots,
    _load_result,
    _planned_attempt,
    _project_scoped_result_path,
    _scheduled_contract,
)
from .execution import _execute_ready
from .paths import attempt_directory, state_path
from .scheduled import (
    _finalize_scheduled_adapter,
    _finalize_scheduler_manifest,
    _observe_scheduled_step,
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
    "safety": {"auto_submit": False},
}


def _automatic_cluster(site: SiteConfig, resources: Any) -> ClusterProfile:
    if not isinstance(resources, dict):
        raise BackendError("automatic cluster selection requires HPC resources")
    best: tuple[tuple[int, int], ClusterProfile] | None = None
    failures: list[str] = []
    for profile in site.clusters.values():
        scheduler = profile.scheduler
        if scheduler is None:
            failures.append(f"{profile.name}: no partition candidates")
            continue
        try:
            backend = SshSlurmBackend(profile.ssh_profile)
            if scheduler.memory_constraint == "unreported":
                routing = backend.select_partition(
                    scheduler.partition_candidates,
                    resources,
                    memory_constraint="unreported",
                )
            else:
                routing = backend.select_partition(
                    scheduler.partition_candidates,
                    resources,
                )
            selected_partition = routing.get("selected_partition")
            observations = routing.get("observed_partition_availability")
            selected = next(
                (
                    item
                    for item in observations
                    if isinstance(item, dict)
                    and item.get("partition") == selected_partition
                ),
                None,
            ) if isinstance(observations, list) else None
            availability = (
                selected.get("observed_node_availability")
                if isinstance(selected, dict)
                else None
            )
            if not isinstance(availability, dict):
                raise BackendError("cluster availability observation is malformed")
            score = (
                int(availability.get("available_now", 0)),
                int(availability.get("capable_for_request", 0)),
            )
        except Exception as exc:
            failures.append(f"{profile.name}: {exc}")
            continue
        if best is None or score > best[0]:
            best = (score, profile)
    if best is None:
        raise BackendError(
            "automatic cluster selection found no schedulable profile ("
            + "; ".join(failures)
            + ")"
        )
    return best[1]


def _run_cluster(
    site: SiteConfig, node: dict[str, Any]
) -> tuple[ClusterProfile, dict[str, str] | None]:
    requested = node.get("backend_profile")
    if requested is not None:
        return site.cluster(requested), None
    profile = _automatic_cluster(site, node.get("resources"))
    return profile, {
        "mode": "automatic",
        "selected_profile": profile.name,
        "criterion": "most currently available capable nodes; site order breaks ties",
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
    with StateStore(database, readonly=False) as store:
        store.initialize_project(project.project_id, project.nodes)
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
            store.assert_project_topology(project.project_id, project.nodes)
    node = project.node(node_id)
    plugins = discover_plugins(plugin_root)
    plugin = select_plugin(plugins, str(node["uses"]))
    attempt = _planned_attempt(project, node_id)
    plan = node_plan(project, node, plugin, attempt=attempt)
    if node.get("mode", "execute") == "execute" and plugin.raw.get("implementation", {}).get(
        "status"
    ) in {"adapter-ready", "implemented"}:
        context = _adapter_context(project, node, attempt)
        adapter = load_adapter(plugin)
        diagnostics = adapter.validate(context)
        adapter_plan = adapter.plan(context)
        if not isinstance(adapter_plan, dict):
            raise PluginError(f"plugin {plugin.plugin_id} plan must return a mapping")
        # Keep the reviewed plan portable so it can be used on the selected site.
        roots = _portable_roots(project, plugin, node_id, attempt)
        plan["adapter_diagnostics"] = to_portable(diagnostics, roots)
        plan["adapter_plan"] = to_portable(adapter_plan, roots)
        if (
            str(node.get("backend", "local")) == "ssh-slurm"
            and isinstance(adapter_plan.get("scheduled_execution"), dict)
        ):
            scheduled = _scheduled_contract(
                project,
                plugin,
                {"adapter_plan": adapter_plan},
                node_id=node_id,
                attempt=attempt,
            )
            site = load_site_config(site_path)
            profile, cluster_selection = _run_cluster(site, node)
            plan["backend_profile"] = profile.name
            if cluster_selection is not None:
                plan["cluster_selection"] = cluster_selection
            provider: TemplateLibrary = (
                template_library
                if template_library is not None
                else SshSlurmBackend(profile.ssh_profile)
            )
            if scheduled["schema_version"] == 4:
                plan["hpc_executions"] = [
                    {
                        "submission_id": submission["id"],
                        "hpc_execution": resolve_hpc_execution_plan(
                            profile=profile,
                            project_id=project.project_id,
                            node_id=str(node["id"]),
                            attempt=attempt,
                            resources_value=node.get("resources"),
                            scheduled_execution={
                                "execution_model": scheduled["execution_model"],
                                "template_family": scheduled["template_family"],
                                "template_variables": submission[
                                    "template_variables"
                                ],
                            },
                            library=provider,
                            submission_id=str(submission["id"]),
                        ),
                    }
                    for submission in scheduled["submissions"]
                ]
            else:
                plan["hpc_execution"] = resolve_hpc_execution_plan(
                    profile=profile,
                    project_id=project.project_id,
                    node_id=str(node["id"]),
                    attempt=attempt,
                    resources_value=node.get("resources"),
                    scheduled_execution=scheduled,
                    library=provider,
                )
    return plan


def run_node(
    project: Project,
    node_id: str,
    plugin_root: Path,
    approval: bool = False,
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
    has_adapter = isinstance(adapter_plan, dict)
    if has_adapter and (
        adapter_plan.get("status") != RunState.READY.value
        or adapter_plan.get("executable") is not True
    ):
        raise PluginError(
            f"plugin {plugin.plugin_id} blocked execution: {adapter_plan.get('diagnostics', adapter_plan)}"
        )
    if node.get("mode", "execute") == "execute":
        backend = str(node.get("backend", "local"))
        if backend == "local" and not has_adapter:
            raise PluginError(
                "local execution requires an adapter-ready plugin with scientific check/collect"
            )
        if backend == "slurm" and has_adapter:
            raise BackendError(
                "local SLURM adapter execution is disabled; only the controlled ssh-slurm "
                "staging/fetch/check contract is currently implemented"
            )
        if backend == "ssh-slurm" and has_adapter:
            scheduled = adapter_plan.get("scheduled_execution")
            resolved = (
                plan.get("hpc_executions")
                if isinstance(scheduled, dict)
                and scheduled.get("schema_version") == 4
                else plan.get("hpc_execution")
            )
            if not isinstance(scheduled, dict) or not isinstance(
                resolved, (dict, list)
            ):
                raise BackendError(
                    "ssh-slurm adapter execution requires templates, a site profile, and an "
                    "explicit staged execution contract"
                )
    database = state_path(project)
    existed = database.is_file()
    with StateStore(database, readonly=False) as store:
        if existed:
            store.assert_project_topology(project.project_id, project.nodes)
        else:
            store.initialize_project(project.project_id, project.nodes)
        step = store.latest_step(project.project_id, node_id)
        if step.attempt != plan["attempt"]:
            raise StateError("planned attempt changed before execution; generate a fresh dry-run")
        if step.state == RunState.WAIT.value:
            if not store.dependencies_satisfied(project.project_id, node_id):
                raise StateError(f"node {node_id} is waiting for dependencies")
            step = store.transition(step.run_id, RunState.READY)
        if step.state != RunState.READY.value:
            raise StateError(f"node {node_id} is {step.state}, expected READY")
        step = store.bind_attempt_node(step.run_id, node)
        execution_node = node
        if (
            node.get("backend") == "ssh-slurm"
            and node.get("backend_profile") is None
            and isinstance(plan.get("backend_profile"), str)
        ):
            execution_node = {**node, "backend_profile": plan["backend_profile"]}
        result = _execute_ready(
            project,
            execution_node,
            plugin,
            plan,
            store,
            step.run_id,
            step.attempt,
            factory=factory,
        )
    return result


def make_advance_plan(
    project: Project,
    plugin_root: Path | None = None,
    *,
    factory: SchedulerFactory | None = None,
) -> dict[str, Any]:
    database = state_path(project)
    transitions: list[dict[str, str]] = []
    observations: list[dict[str, Any]] = []
    if database.is_file():
        with StateStore(database, readonly=True) as store:
            store.assert_project_topology(project.project_id, project.nodes)
            for step in store.latest_steps(project.project_id):
                if step.state in {
                    RunState.SUBMITTED.value,
                    RunState.PENDING.value,
                    RunState.RUNNING.value,
                } and step.job_id:
                    observation = _observe_scheduled_step(
                        project, step, plugin_root, factory=factory
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
                            }
                        )
                    continue
                if step.state not in {RunState.WAIT.value, RunState.BLOCKED.value}:
                    continue
                node = store.node_snapshot(step.run_id)
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
    plugin_root: Path | None = None,
    *,
    factory: SchedulerFactory | None = None,
) -> dict[str, Any]:
    plan = make_advance_plan(project, plugin_root, factory=factory)
    database = state_path(project)
    if not database.is_file():
        raise StateError("project has not been initialized")
    changed: list[dict[str, Any]] = []
    with StateStore(database, readonly=False) as store:
        store.assert_project_topology(project.project_id, project.nodes)
        for change in plan["details"]["transitions"]:
            step = store.latest_step(project.project_id, change["node_id"])
            if change.get("action") == "adapter-finalize":
                if plugin_root is None:
                    raise PluginError("scheduled adapter finalization requires the plugin root")
                changed.append(
                    _finalize_scheduled_adapter(
                        project, step, change, plugin_root, store, factory=factory
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
                result_data, artifacts = _load_result(result_path)
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
    return {"changed": changed}


def make_retry_plan(project: Project, node_id: str) -> dict[str, Any]:
    database = state_path(project)
    if not database.is_file():
        raise StateError("project has not been initialized")
    with StateStore(database, readonly=True) as store:
        store.assert_project_topology(project.project_id, project.nodes)
        step = store.latest_step(project.project_id, node_id)
    if step.state not in {RunState.FAIL.value, RunState.STOPPED.value}:
        raise StateError(f"retry requires FAIL or STOPPED, got {step.state}")
    return action_plan(
        "retry",
        project,
        node_id,
        {
            "previous_run_id": step.run_id,
            "previous_attempt": step.attempt,
            "state": step.state,
        },
    )


def retry(project: Project, node_id: str) -> dict[str, Any]:
    make_retry_plan(project, node_id)
    with StateStore(state_path(project), readonly=False) as store:
        store.assert_project_topology(project.project_id, project.nodes)
        retried = store.create_retry(project.project_id, node_id)
    return {"step": retried.to_dict()}


def make_stop_plan(
    project: Project, node_id: str, site_path: Path | None = None
) -> dict[str, Any]:
    database = state_path(project)
    if not database.is_file():
        raise StateError("project has not been initialized")
    with StateStore(database, readonly=True) as store:
        store.assert_project_topology(project.project_id, project.nodes)
        step = store.latest_step(project.project_id, node_id)
        node = store.node_snapshot(step.run_id)
    scheduler_target: dict[str, Any] | None = None
    backend_profile = node.get("backend_profile")
    if step.backend == "ssh-slurm":
        if backend_profile is None and step.job_id:
            scheduler_target = _approved_cluster_record(project, node_id, step.attempt)
            backend_profile = scheduler_target.get("name")
        elif backend_profile is not None:
            site = load_site_config(site_path)
            profile = site.cluster(backend_profile)
            scheduler_target = profile.to_plan_dict()
    return action_plan(
        "stop",
        project,
        node_id,
        {
            "run_id": step.run_id,
            "state": step.state,
            "backend": step.backend,
            "backend_profile": backend_profile,
            "scheduler_target": scheduler_target,
            "job_id": step.job_id,
            "job_ids": step.job_id.split(",") if step.job_id else [],
        },
    )


def stop(
    project: Project,
    node_id: str,
    site_path: Path | None = None,
    *,
    factory: SchedulerFactory | None = None,
) -> dict[str, Any]:
    database = state_path(project)
    if not database.is_file():
        raise StateError("project has not been initialized")
    with StateStore(database, readonly=False) as store:
        store.assert_project_topology(project.project_id, project.nodes)
        step = store.latest_step(project.project_id, node_id)
        node = store.node_snapshot(step.run_id)
        state = RunState(step.state)
        if state in {RunState.OK, RunState.FAIL, RunState.STOPPED}:
            raise StateError(f"cannot stop node in terminal state {state.value}")
        if step.job_id:
            if step.backend == "ssh-slurm" and node.get("backend_profile") is None:
                scheduler = scheduler_from_cluster_record(
                    _approved_cluster_record(project, node_id, step.attempt),
                    factory=factory,
                )
            else:
                scheduler = scheduler_for_node(
                    str(step.backend), node, site_path, factory=factory
                )
            for job_id in step.job_id.split(","):
                result = scheduler.cancel(job_id)
                if result.returncode != 0:
                    raise BackendError(
                        result.stderr
                        or result.stdout
                        or "scheduler cancellation failed"
                    )
        elif state == RunState.RUNNING:
            raise BackendError("cannot safely stop a local run without a persisted process handle")
        updated = store.transition(step.run_id, RunState.STOPPED)
    return {"step": updated.to_dict()}


def _approved_cluster_record(
    project: Project, node_id: str, attempt: int
) -> dict[str, Any]:
    path = attempt_directory(project, node_id, attempt) / "approved-plan.json"
    if path.is_symlink() or not path.is_file():
        raise StateError("approved scheduled plan is missing or is a symlink")
    plan = load_mapping(path)
    records: list[dict[str, Any]] = []
    execution = plan.get("hpc_execution")
    if isinstance(execution, dict) and isinstance(execution.get("cluster_profile"), dict):
        records.append(execution["cluster_profile"])
    executions = plan.get("hpc_executions")
    if isinstance(executions, list):
        for item in executions:
            resolved = item.get("hpc_execution") if isinstance(item, dict) else None
            cluster = resolved.get("cluster_profile") if isinstance(resolved, dict) else None
            if isinstance(cluster, dict):
                records.append(cluster)
    names = {record.get("name") for record in records}
    if not records or len(names) != 1:
        raise StateError("approved scheduled plan lacks one unambiguous cluster profile")
    return records[0]


def _require_approval(plan: dict[str, Any], approval: bool) -> None:
    if plan.get("approval_required") is True and approval is not True:
        raise ApprovalError("review the dry-run, then confirm with --approve")
