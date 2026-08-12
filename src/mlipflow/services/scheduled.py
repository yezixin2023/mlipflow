"""Scheduler-backed execution: stage, submit, observe, fetch, finalize.

This is the asynchronous two-approval lifecycle.  The first approval covers
resolution, rendering, staging and submission; the second binds the remote output
inventory before anything is fetched and handed to the pinned scientific checker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..artifacts import fingerprint
from ..config import Project
from ..errors import ApprovalError, BackendError, ConfigError, StateError
from ..io import load_mapping, write_json_atomic, write_text_atomic
from ..planning import node_plan, with_digest
from ..plugins import PluginSpec, discover_plugins, load_adapter, select_plugin
from ..site import load_site_config
from ..state import RunState, StateStore, StepRun, utc_now
from .backend_factory import (
    SchedulerFactory,
    default_scheduler_factory,
    scheduler_from_cluster_record,
)
from .contracts import (
    _adapter_context,
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
        record = rendered.get(name)
        content = record.get("content") if isinstance(record, dict) else None
        if not isinstance(content, str):
            raise BackendError(f"approved plan lacks rendered {name}")
        path = attempt_dir / name
        write_text_atomic(path, content)
        if (
            record.get("sha256") != _sha256_file(path)
            or record.get("size_bytes") != path.stat().st_size
        ):
            raise BackendError(f"rendered HPC script identity mismatch: {name}")
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
    scheduled = _scheduled_contract(project, plugin, plan)
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
    result = backend.submit(_HPC_SUBMIT_SCRIPT, remote_run_dir)
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


def _observe_scheduled_step(
    project: Project,
    step: StepRun,
    plugin_root: Path | None = None,
    site_path: Path | None = None,
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
            pinned_plan, _, _ = _load_pinned_scheduled_plan(
                project, step, plugin_root, site_path
            )
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
    }:
        observation["target_state"] = RunState.FAIL.value
        observation["reason"] = f"scheduler terminal failure: {raw_state}"
    elif raw_state in {"CANCELLED", "REVOKED"}:
        observation["target_state"] = RunState.STOPPED.value
        observation["reason"] = f"scheduler reports cancellation: {raw_state}"
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
                    project, step, plugin_root, site_path
                )
                hpc_execution = approved_plan.get("hpc_execution")
                cluster_record = (
                    hpc_execution.get("cluster_profile")
                    if isinstance(hpc_execution, dict)
                    else None
                )
                workspace = (
                    hpc_execution.get("workspace")
                    if isinstance(hpc_execution, dict)
                    else None
                )
                if (
                    not isinstance(cluster_record, dict)
                    or not isinstance(workspace, dict)
                    or step.remote_dir != workspace.get("run_dir")
                ):
                    raise BackendError(
                        "persisted remote workspace differs from the approved plan"
                    )
                backend = scheduler_from_cluster_record(cluster_record, factory=factory)
                inventory = _remote_output_inventory(backend, workspace, scheduled)
                observation["adapter_finalization"] = {
                    "approved_plan_fingerprint": fingerprint(approved_plan_path),
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


def _remote_output_inventory(
    backend: Any, workspace: dict[str, Any], scheduled: dict[str, Any]
) -> list[dict[str, Any]]:
    """Read the bounded identity of every approved output without mutating it.

    ``advance --dry-run`` and ``advance --approve`` must produce byte-identical
    inventories for the fetch to be allowed, so both go through this one function.
    """

    inventory: list[dict[str, Any]] = []
    for item in scheduled["fetch_outputs"]:
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


def _load_pinned_scheduled_plan(
    project: Project,
    step: StepRun,
    plugin_root: Path,
    site_path: Path | None,
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
    site = load_site_config(site_path)
    profile = site.cluster(node.get("backend_profile"))
    hpc_execution = plan.get("hpc_execution")
    cluster_record = (
        hpc_execution.get("cluster_profile")
        if isinstance(hpc_execution, dict)
        else None
    )
    if plan.get("site_config_digest") != site.digest:
        raise ApprovalError("site configuration changed after scheduler submission")
    if cluster_record != profile.to_plan_dict():
        raise ApprovalError("cluster profile changed after scheduler submission")
    scheduled = _scheduled_contract(project, plugin, plan)
    return plan, plugin, scheduled


def _finalize_scheduled_adapter(
    project: Project,
    step: StepRun,
    change: dict[str, Any],
    plugin_root: Path,
    store: StateStore,
    site_path: Path | None,
    *,
    factory: SchedulerFactory | None = None,
) -> StepRun:
    details = change.get("adapter_finalization")
    if not isinstance(details, dict):
        raise StateError("approved transition lacks adapter finalization details")
    plan, plugin, scheduled = _load_pinned_scheduled_plan(
        project, step, plugin_root, site_path
    )
    attempt_dir = attempt_directory(project, step.node_id, step.attempt)
    plan_path = attempt_dir / "approved-plan.json"
    if fingerprint(plan_path) != details.get("approved_plan_fingerprint"):
        raise StateError("approved scheduled plan changed after advance dry-run")
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
    approved_inventory = details.get("outputs")
    if not isinstance(approved_inventory, list):
        raise StateError("advance plan lacks the approved remote output inventory")
    backend = scheduler_from_cluster_record(cluster_record, factory=factory)
    current_inventory = _remote_output_inventory(backend, workspace, scheduled)
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
            str(workspace["run_dir"]), str(item["remote_path"]), destination
        )
        if destination.stat().st_size != item["size_bytes"] or _sha256_file(destination) != item["sha256"]:
            raise StateError(f"fetched output fingerprint mismatch: {item['remote_name']}")
        fetched.append(
            fingerprint(destination)
            | {"role": str(item.get("role", "scheduler-output"))}
        )
    node = project.node(step.node_id)
    context = _adapter_context(project, node, step.attempt)
    context["execution"] = {
        "returncode": 0,
        "scheduler_state": "COMPLETED",
        "remote_output_inventory": current_inventory,
        "plan": plan["adapter_plan"],
        "hpc_execution": hpc_execution,
    }
    adapter = load_adapter(plugin)
    collected: dict[str, Any] | None = None
    checked: dict[str, Any] | None = None
    metrics: dict[str, Any] = {}
    reason: str
    completion_error = _validate_hpc_completion(attempt_dir / "completion.json", step)
    if fetch_errors:
        final = RunState.FAIL
        reason = "; ".join(fetch_errors)
    elif completion_error is not None:
        final = RunState.FAIL
        reason = completion_error
    else:
        checked = adapter.check(context)
        if not isinstance(checked, dict) or checked.get("status") != RunState.OK.value:
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
                fetched.extend(
                    item for item in normalized if str(item["uri"]) not in known
                )
                raw_metrics = collected.get("metrics", {})
                metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
                final = RunState.OK
                reason = (
                    "scheduler completed; bounded fetch and pinned plugin checks succeeded"
                )
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
