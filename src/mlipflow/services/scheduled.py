"""Scheduler-backed execution: stage, submit, observe, fetch, finalize.

Submission is the approval boundary. Later scheduler observation, bounded
transport, and scientific completion checks are continuations of that approved
run and do not require another approval digest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..artifacts import fingerprint
from ..config import Project
from ..errors import ApprovalError, BackendError, ConfigError, StateError
from ..io import load_mapping, write_json_atomic, write_text_atomic
from ..planning import node_plan, with_digest
from ..plugins import PluginSpec, discover_plugins, load_adapter
from ..portable import to_runtime
from ..state import RunState, StateStore, StepRun, utc_now
from .backend_factory import (
    SchedulerFactory,
    default_scheduler_factory,
    scheduler_from_cluster_record,
)
from .contracts import (
    _adapter_context,
    _portable_roots,
    _normalize_adapter_artifacts,
    _scheduled_contract,
    _sha256_file,
)
from .paths import attempt_directory


_HPC_SUBMIT_SCRIPT = "submit.sbatch"
_HPC_RUN_SCRIPT = "run.sh"


def _materialize_hpc_scripts(
    attempt_dir: Path, hpc_execution: dict[str, Any]
) -> tuple[Path, Path]:
    rendered = hpc_execution.get("rendered_scripts")
    if not isinstance(rendered, dict):
        raise BackendError("approved plan lacks rendered HPC scripts")
    paths: list[Path] = []
    for name in (_HPC_SUBMIT_SCRIPT, _HPC_RUN_SCRIPT):
        content = rendered.get(name)
        if not isinstance(content, str):
            raise BackendError(f"approved plan lacks rendered {name}")
        path = attempt_dir / name
        write_text_atomic(path, content)
        paths.append(path)
    return paths[0], paths[1]


def _stage_and_submit_scheduled_adapter(
    project: Project,
    node: dict[str, Any],
    plugin: PluginSpec,
    plan: dict[str, Any],
    attempt_dir: Path,
    *,
    factory: SchedulerFactory | None = None,
) -> tuple[Any, str, list[str]]:
    node_id = str(node["id"])
    attempt = int(str(attempt_dir.name).rsplit("-", 1)[1])
    scheduled = _scheduled_contract(
        project, plugin, plan, node_id=node_id, attempt=attempt
    )
    hpc_execution = plan.get("hpc_execution")
    if not isinstance(hpc_execution, dict):
        raise BackendError("approved plan lacks resolved HPC execution")
    cluster = hpc_execution.get("cluster_profile")
    workspace = hpc_execution.get("workspace")
    if not isinstance(cluster, dict) or not isinstance(workspace, dict):
        raise BackendError("approved plan lacks cluster/workspace identity")
    remote_run_dir = workspace.get("run_dir")
    ssh_profile = cluster.get("ssh_profile")
    if not isinstance(remote_run_dir, str) or not isinstance(ssh_profile, str):
        raise BackendError("approved plan has invalid cluster/workspace identity")
    approved_plan = attempt_dir / "approved-plan.json"
    write_json_atomic(approved_plan, plan)
    submit_script, run_script = _materialize_hpc_scripts(attempt_dir, hpc_execution)
    files: list[tuple[Path, str, str]] = [
        (
            Path(str(item["source"])),
            f"input/{item['remote_name']}",
            str(item["sha256"]),
        )
        for item in scheduled["staged_files"]
    ]
    files.extend(
        (path, name, _sha256_file(path))
        for path, name in (
            (submit_script, _HPC_SUBMIT_SCRIPT),
            (run_script, _HPC_RUN_SCRIPT),
        )
    )
    backend = scheduler_from_cluster_record(cluster, factory=factory)
    observed_run_dir = backend.stage_workspace(remote_run_dir, files)
    if observed_run_dir != remote_run_dir:
        raise BackendError("backend returned an unexpected remote attempt workspace")
    scheduler_config = cluster.get("scheduler")
    partition_candidates = (
        scheduler_config.get("partition_candidates")
        if isinstance(scheduler_config, dict)
        else None
    )
    if partition_candidates is None:
        result = backend.submit(_HPC_SUBMIT_SCRIPT, remote_run_dir)
    else:
        resources = hpc_execution.get("resources")
        if not isinstance(partition_candidates, list) or not isinstance(resources, dict):
            raise BackendError("approved plan has invalid scheduler routing configuration")
        result = backend.submit(
            _HPC_SUBMIT_SCRIPT,
            remote_run_dir,
            partition_candidates=partition_candidates,
            resources=resources,
        )
    return result, remote_run_dir, ["template", str(scheduled["template_family"])]


def _scheduler_expected_identity(step: StepRun) -> dict[str, Any]:
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


def _failure_salvage_outputs(
    plan: dict[str, Any], scheduled: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return the approved bounded output subset that may be fetched on failure.

    Plugins cannot introduce new paths here.  Every requested name must already
    exist in the validated scheduled fetch allowlist, and salvage treats missing
    files as optional because a hard failure may happen before the first checkpoint.
    """

    adapter_plan = plan.get("adapter_plan")
    salvage = adapter_plan.get("failure_salvage") if isinstance(adapter_plan, dict) else None
    if salvage is None:
        return []
    if (
        not isinstance(salvage, dict)
        or salvage.get("schema_version") != 1
        or set(salvage) != {"schema_version", "fetch_remote_names"}
    ):
        raise ConfigError("adapter failure_salvage must use schema_version=1 and fetch_remote_names")
    names = salvage.get("fetch_remote_names")
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ConfigError("failure_salvage.fetch_remote_names must be a non-empty unique string list")
    allowed = {
        str(item["remote_name"]): item
        for item in scheduled.get("fetch_outputs", [])
        if isinstance(item, dict) and isinstance(item.get("remote_name"), str)
    }
    missing = [name for name in names if name not in allowed]
    if missing:
        raise ConfigError(
            "failure salvage may reference only validated fetch outputs: " + ", ".join(missing)
        )
    return [{**allowed[name], "required": False} for name in names]


def _approved_workspace(
    approved_plan: dict[str, Any], step: StepRun
) -> tuple[dict[str, Any], dict[str, Any]]:
    hpc_execution = approved_plan.get("hpc_execution")
    cluster_record = (
        hpc_execution.get("cluster_profile") if isinstance(hpc_execution, dict) else None
    )
    workspace = hpc_execution.get("workspace") if isinstance(hpc_execution, dict) else None
    if (
        not isinstance(cluster_record, dict)
        or not isinstance(workspace, dict)
        or step.remote_dir != workspace.get("run_dir")
    ):
        raise BackendError("persisted remote workspace differs from the approved plan")
    return cluster_record, workspace


def _observe_scheduled_step(
    project: Project,
    step: StepRun,
    plugin_root: Path | None = None,
    *,
    factory: SchedulerFactory | None = None,
) -> dict[str, Any]:
    try:
        if step.backend == "slurm":
            build = factory or default_scheduler_factory
            scheduler = build("slurm", None).status(str(step.job_id))
        elif step.backend == "ssh-slurm":
            if plugin_root is None:
                raise ConfigError(
                    "plugin root is required to verify the pinned HPC execution plan"
                )
            pinned_plan, _, _ = _load_pinned_scheduled_plan(project, step, plugin_root)
            hpc_execution = pinned_plan.get("hpc_execution")
            cluster_record = (
                hpc_execution.get("cluster_profile")
                if isinstance(hpc_execution, dict)
                else None
            )
            scheduler = scheduler_from_cluster_record(
                cluster_record, factory=factory
            ).status(str(step.job_id))
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
        "CANCELLED",
        "REVOKED",
    }:
        target = (
            RunState.STOPPED
            if raw_state in {"CANCELLED", "REVOKED"}
            else RunState.FAIL
        )
        approved_plan_path = (
            attempt_directory(project, step.node_id, step.attempt) / "approved-plan.json"
        )
        if approved_plan_path.is_file() and plugin_root is not None:
            try:
                approved_plan, plugin, scheduled = _load_pinned_scheduled_plan(
                    project, step, plugin_root
                )
                salvage_outputs = _failure_salvage_outputs(approved_plan, scheduled)
                if salvage_outputs:
                    cluster_record, workspace = _approved_workspace(approved_plan, step)
                    backend = scheduler_from_cluster_record(cluster_record, factory=factory)
                    inventory = _remote_output_inventory_items(
                        backend, workspace, salvage_outputs
                    )
                    observation["adapter_finalization"] = {
                        "plugin_id": plugin.plugin_id,
                        "remote_run_dir": workspace["run_dir"],
                        "outputs": inventory,
                        "failure_salvage": True,
                        "terminal_target": target.value,
                    }
                    existing = [
                        str(item["remote_name"])
                        for item in inventory
                        if item.get("exists") and not item.get("oversized")
                    ]
                    oversized = [
                        str(item["remote_name"])
                        for item in inventory
                        if item.get("oversized")
                    ]
                    detail = (
                        "approved failure salvage is ready for: " + ", ".join(existing)
                        if existing
                        else "no approved salvage output exists yet"
                    )
                    if oversized:
                        detail += "; oversized: " + ", ".join(oversized)
                    observation["reason"] = (
                        f"scheduler terminal state {raw_state}; {detail}"
                    )
                    return observation
            except Exception as exc:
                observation["target_state"] = target.value
                observation["reason"] = (
                    f"scheduler terminal state {raw_state}; failure salvage unavailable: {exc}"
                )
                return observation
        observation["target_state"] = target.value
        observation["reason"] = (
            f"scheduler terminal failure: {raw_state}"
            if target == RunState.FAIL
            else f"scheduler reports cancellation: {raw_state}"
        )
    elif raw_state == "COMPLETED":
        approved_plan_path = (
            attempt_directory(project, step.node_id, step.attempt) / "approved-plan.json"
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
                cluster_record, workspace = _approved_workspace(approved_plan, step)
                backend = scheduler_from_cluster_record(cluster_record, factory=factory)
                inventory = _remote_output_inventory(backend, workspace, scheduled)
                observation["adapter_finalization"] = {
                    "plugin_id": plugin.plugin_id,
                    "remote_run_dir": workspace["run_dir"],
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
        observation["reason"] = (
            "scheduler completed, but no pinned adapter execution plan is available; "
            "scientific OK is withheld"
        )
    else:
        observation["reason"] = f"unrecognized scheduler state {raw_state}; no state change"
    return observation


def _remote_output_inventory_items(
    backend: Any, workspace: dict[str, Any], outputs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for item in outputs:
        observed = backend.inspect_file(
            str(workspace["run_dir"]), str(item["remote_path"])
        )
        record = {**item, **observed}
        if observed.get("exists") and int(observed.get("size_bytes", 0)) > int(
            item["max_bytes"]
        ):
            record["oversized"] = True
        inventory.append(record)
    return inventory


def _remote_output_inventory(
    backend: Any, workspace: dict[str, Any], scheduled: dict[str, Any]
) -> list[dict[str, Any]]:
    """Read the bounded identity of every approved output without mutating it.

    The inventory is re-read immediately before fetch so a remote file cannot
    change between observation and transport.
    """

    return _remote_output_inventory_items(
        backend, workspace, list(scheduled["fetch_outputs"])
    )


def _load_pinned_scheduled_plan(
    project: Project,
    step: StepRun,
    plugin_root: Path,
) -> tuple[dict[str, Any], PluginSpec, dict[str, Any]]:
    attempt_dir = attempt_directory(project, step.node_id, step.attempt)
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
    node = _attempt_node(plan, step)
    pinned_plugin = plan.get("plugin")
    if not isinstance(pinned_plugin, dict) or not isinstance(pinned_plugin.get("id"), str):
        raise ApprovalError("approved scheduled plan lacks a plugin identity")
    plugins = discover_plugins(plugin_root)
    plugin_id = str(pinned_plugin["id"])
    if plugin_id not in plugins:
        raise ApprovalError(f"pinned scheduled plugin is unavailable: {plugin_id}")
    plugin = plugins[plugin_id]
    current = node_plan(project, node, plugin, attempt=step.attempt)
    current_plugin = current.get("plugin")
    if (
        pinned_plugin.get("id") != current_plugin.get("id")
        or pinned_plugin.get("implementation_identity")
        != current_plugin.get("implementation_identity")
    ):
        raise ApprovalError("pinned scheduled checker implementation changed")
    scheduled = _scheduled_contract(
        project,
        plugin,
        plan,
        node_id=step.node_id,
        attempt=step.attempt,
        verify_staged_sources=False,
    )
    return plan, plugin, scheduled


def _attempt_node(plan: dict[str, Any], step: StepRun) -> dict[str, Any]:
    """Recover the node semantics bound into an approved attempt plan."""

    plugin = plan.get("plugin")
    plugin_id = plugin.get("id") if isinstance(plugin, dict) else step.plugin_id
    return {
        "id": step.node_id,
        "uses": plugin_id,
        "mode": plan.get("mode", "execute"),
        "backend": plan.get("backend", step.backend),
        "backend_profile": plan.get("backend_profile"),
        "inputs": plan.get("inputs", {}),
        "parameters": plan.get("parameters", {}),
        "resources": plan.get("resources", {}),
    }


def _stored_fingerprint_mode(value: Any) -> str:
    if isinstance(value, str) and value.startswith("sha256:"):
        return "full"
    if isinstance(value, str) and value.startswith("tree-sha256:"):
        return "tree-full"
    return "external"


def _finalize_scheduled_adapter(
    project: Project,
    step: StepRun,
    change: dict[str, Any],
    plugin_root: Path,
    store: StateStore,
    *,
    factory: SchedulerFactory | None = None,
) -> StepRun:
    details = change.get("adapter_finalization")
    if not isinstance(details, dict):
        raise StateError("approved transition lacks adapter finalization details")
    plan, plugin, scheduled = _load_pinned_scheduled_plan(project, step, plugin_root)
    attempt_dir = attempt_directory(project, step.node_id, step.attempt)
    hpc_execution = plan.get("hpc_execution")
    cluster_record = (
        hpc_execution.get("cluster_profile")
        if isinstance(hpc_execution, dict)
        else None
    )
    workspace = (
        hpc_execution.get("workspace") if isinstance(hpc_execution, dict) else None
    )
    if (
        not isinstance(cluster_record, dict)
        or not isinstance(workspace, dict)
        or details.get("remote_run_dir") != workspace.get("run_dir")
    ):
        raise StateError("approved remote attempt workspace changed before fetch")
    observed_inventory = details.get("outputs")
    if not isinstance(observed_inventory, list):
        raise StateError("advance observation lacks the remote output inventory")
    failure_salvage = details.get("failure_salvage") is True
    backend = scheduler_from_cluster_record(cluster_record, factory=factory)
    if failure_salvage:
        output_contract = _failure_salvage_outputs(plan, scheduled)
        current_inventory = _remote_output_inventory_items(
            backend, workspace, output_contract
        )
    else:
        current_inventory = _remote_output_inventory(backend, workspace, scheduled)
    if current_inventory != observed_inventory:
        raise StateError("remote outputs changed between observation and fetch")
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
            str(workspace["run_dir"]), str(item["remote_path"]), destination
        )
        if destination.stat().st_size != item["size_bytes"] or _sha256_file(destination) != item["sha256"]:
            raise StateError(f"fetched output fingerprint mismatch: {item['remote_name']}")
        fetched.append(
            fingerprint(destination)
            | {"role": str(item.get("role", "scheduler-output"))}
        )

    metrics: dict[str, Any] = {}
    reason: str
    if failure_salvage:
        raw_target = details.get("terminal_target")
        if raw_target not in {RunState.FAIL.value, RunState.STOPPED.value}:
            raise StateError("failure salvage lacks a valid terminal target")
        final = RunState(str(raw_target))
        if fetch_errors:
            reason = str(change.get("reason", "scheduler terminal failure")) + "; " + "; ".join(fetch_errors)
        else:
            reason = str(change.get("reason", "scheduler terminal failure"))
    else:
        node = _attempt_node(plan, step)
        context = _adapter_context(project, node, step.attempt)
        context["execution"] = {
            "returncode": 0,
            "scheduler_state": "COMPLETED",
            "remote_output_inventory": current_inventory,
            # Resolve portable tokens so the pinned checker reads the same absolute
            # paths its planning half emitted.
            "plan": to_runtime(
                plan["adapter_plan"],
                _portable_roots(project, plugin, step.node_id, step.attempt),
            ),
            "hpc_execution": hpc_execution,
        }
        adapter = load_adapter(plugin)
        collected: dict[str, Any] | None = None
        checked: dict[str, Any] | None = None
        completion_error = _validate_hpc_completion(attempt_dir / "completion.json", step)
        if fetch_errors:
            final = RunState.FAIL
            reason = "; ".join(fetch_errors)
        elif completion_error is not None:
            final = RunState.FAIL
            reason = completion_error
        else:
            checked = adapter.check(context)
            if checked.get("status") != RunState.OK.value:
                final = RunState.FAIL
                reason = f"pinned plugin completion check did not return OK: {checked}"
            else:
                collected = adapter.collect(context)
                if collected.get("status") != RunState.OK.value:
                    final = RunState.FAIL
                    reason = f"pinned plugin collection did not return OK: {collected}"
                else:
                    normalized = _normalize_adapter_artifacts(
                        project, attempt_dir, collected.get("artifacts", [])
                    )
                    known = {str(item["uri"]) for item in fetched}
                    fetched.extend(
                        item for item in normalized if str(item["uri"]) not in known
                    )
                    metrics = collected.get("metrics", {})
                    final = RunState.OK
                    reason = (
                        "scheduler completed; bounded fetch and pinned plugin checks succeeded"
                    )

    initial_artifacts = store.artifacts(step.run_id)
    artifacts = [
        {
            **item,
            "fingerprint_mode": _stored_fingerprint_mode(item.get("fingerprint")),
            "mtime_ns": None,
        }
        for item in initial_artifacts
    ]
    known_uris = {str(item["uri"]) for item in artifacts}
    artifacts.extend(item for item in fetched if str(item["uri"]) not in known_uris)
    current_state = RunState(step.state)
    if not failure_salvage and current_state in {RunState.SUBMITTED, RunState.PENDING}:
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
    scheduler_state = str(change.get("scheduler_state", "UNKNOWN"))
    final_manifest = _finalize_scheduler_manifest(
        project,
        step,
        final,
        reason,
        scheduler_state,
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
    step: StepRun,
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


def _validate_hpc_completion(path: Path, step: StepRun) -> str | None:
    if path.is_symlink() or not path.is_file():
        return "scheduler completion.json is missing or unsafe"
    try:
        completion = load_mapping(path)
    except Exception as exc:
        return f"scheduler completion.json is invalid: {exc}"
    expected = {
        "schema_version": 1,
        "status": "COMPLETED",
        "exit_code": 0,
        "project_id": step.project_id,
        "node_id": step.node_id,
        "attempt": step.attempt,
    }
    for key, value in expected.items():
        if completion.get(key) != value:
            return (
                f"scheduler completion.json field {key} differs from the approved "
                f"attempt: expected {value!r}, got {completion.get(key)!r}"
            )
    return None
