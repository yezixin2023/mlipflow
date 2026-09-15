"""Scheduler-backed execution: stage, submit, observe, fetch, finalize.

Later scheduler observation, output fetching, and scientific completion
checks continue the reviewed submission without another approval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..artifacts import artifact as artifact_record
from ..config import Project
from ..errors import ApprovalError, BackendError, ConfigError, StateError
from ..manifests import result_summary
from ..io import load_mapping, write_json_atomic, write_text_atomic
from ..plugins import capability, load_adapter
from ..state import RunState, StateStore, StepRun, utc_now
from .backend_factory import (
    SchedulerFactory,
    scheduler_from_cluster_record,
)
from .contracts import (
    _adapter_context,
    _attempt_artifacts,
    _normalize_adapter_artifacts,
    _scheduled_contract,
)
from .paths import attempt_directory


_HPC_SUBMIT_SCRIPT = "submit.sbatch"
_HPC_RUN_SCRIPT = "run.sh"
_HPC_SUBMISSIONS = "scheduler-submissions.json"


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
    capability_id: str,
    plan: dict[str, Any],
    attempt_dir: Path,
    *,
    factory: SchedulerFactory | None = None,
) -> tuple[Any, str, list[str]]:
    node_id = str(node["id"])
    attempt = int(str(attempt_dir.name).rsplit("-", 1)[1])
    scheduled = _scheduled_contract(
        project, capability_id, plan, node_id=node_id, attempt=attempt
    )
    hpc_execution = plan.get("hpc_execution")
    if not isinstance(hpc_execution, dict):
        raise BackendError("approved plan lacks resolved HPC execution")
    cluster = hpc_execution.get("cluster_profile")
    workspace = hpc_execution.get("workspace")
    if not isinstance(cluster, dict) or not isinstance(workspace, dict):
        raise BackendError("approved plan lacks cluster/workspace settings")
    remote_run_dir = workspace.get("run_dir")
    ssh_profile = cluster.get("ssh_profile")
    if not isinstance(remote_run_dir, str) or not isinstance(ssh_profile, str):
        raise BackendError("approved plan has invalid cluster/workspace settings")
    approved_plan = attempt_dir / "approved-plan.json"
    write_json_atomic(
        approved_plan,
        {
            key: plan[key]
            for key in (
                "project_id",
                "node_id",
                "attempt",
                "capability",
                "backend_profile",
                "adapter_plan",
                "hpc_execution",
            )
            if key in plan
        },
    )
    submit_script, run_script = _materialize_hpc_scripts(attempt_dir, hpc_execution)
    files: list[tuple[Path, str]] = [
        (Path(str(item["source"])), f"input/{item['remote_name']}")
        for item in scheduled["staged_files"]
    ]
    files.extend(
        (path, name)
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
        submit_options = {
            "partition_candidates": partition_candidates,
            "resources": resources,
        }
        if scheduler_config.get("memory_constraint") == "unreported":
            submit_options["memory_constraint"] = "unreported"
        result = backend.submit(_HPC_SUBMIT_SCRIPT, remote_run_dir, **submit_options)
    return result, remote_run_dir, ["template", str(scheduled["template_family"])]


def _stage_and_submit_independent_jobs(
    project: Project,
    node: dict[str, Any],
    capability_id: str,
    plan: dict[str, Any],
    attempt_dir: Path,
    *,
    factory: SchedulerFactory | None = None,
) -> list[dict[str, Any]]:
    """Stage every approved calculation and submit each as its own Slurm job."""

    node_id = str(node["id"])
    attempt = int(str(attempt_dir.name).rsplit("-", 1)[1])
    scheduled = _scheduled_contract(
        project, capability_id, plan, node_id=node_id, attempt=attempt
    )
    if scheduled.get("schema_version") != 4:
        raise BackendError("independent submission requires scheduled schema_version 4")
    resolved = plan.get("hpc_executions")
    if not isinstance(resolved, list) or len(resolved) != len(
        scheduled["submissions"]
    ):
        raise BackendError("approved plan lacks resolved independent HPC executions")
    by_id: dict[str, dict[str, Any]] = {}
    for item in resolved:
        submission_id = item.get("submission_id") if isinstance(item, dict) else None
        execution = item.get("hpc_execution") if isinstance(item, dict) else None
        if (
            not isinstance(submission_id, str)
            or submission_id in by_id
            or not isinstance(execution, dict)
        ):
            raise BackendError("approved independent HPC execution is malformed")
        by_id[submission_id] = execution
    expected_ids = [str(item["id"]) for item in scheduled["submissions"]]
    if set(by_id) != set(expected_ids):
        raise BackendError("resolved independent HPC execution ids changed")

    write_json_atomic(
        attempt_dir / "approved-plan.json",
        {
            key: plan[key]
            for key in (
                "project_id",
                "node_id",
                "attempt",
                "capability",
                "backend_profile",
                "adapter_plan",
                "hpc_executions",
            )
            if key in plan
        },
    )
    staged: list[tuple[str, dict[str, Any], Any, str]] = []
    for submission in scheduled["submissions"]:
        submission_id = str(submission["id"])
        hpc_execution = by_id[submission_id]
        cluster = hpc_execution.get("cluster_profile")
        workspace = hpc_execution.get("workspace")
        if not isinstance(cluster, dict) or not isinstance(workspace, dict):
            raise BackendError("approved independent job lacks cluster/workspace settings")
        remote_run_dir = workspace.get("run_dir")
        if not isinstance(remote_run_dir, str):
            raise BackendError("approved independent job has an invalid workspace")
        control_dir = attempt_dir / "submissions" / submission_id
        control_dir.mkdir(parents=True, exist_ok=False)
        submit_script, run_script = _materialize_hpc_scripts(
            control_dir, hpc_execution
        )
        files: list[tuple[Path, str]] = [
            (Path(str(item["source"])), f"input/{item['remote_name']}")
            for item in submission["staged_files"]
        ]
        files.extend(
            (path, name)
            for path, name in (
                (submit_script, _HPC_SUBMIT_SCRIPT),
                (run_script, _HPC_RUN_SCRIPT),
            )
        )
        backend = scheduler_from_cluster_record(cluster, factory=factory)
        observed = backend.stage_workspace(remote_run_dir, files)
        if observed != remote_run_dir:
            raise BackendError(
                "backend returned an unexpected independent job workspace"
            )
        staged.append((submission_id, hpc_execution, backend, remote_run_dir))

    submitted: list[dict[str, Any]] = []
    try:
        for index, (submission_id, hpc_execution, backend, remote_run_dir) in enumerate(
            staged
        ):
            cluster = hpc_execution["cluster_profile"]
            scheduler_config = cluster.get("scheduler")
            candidates = (
                scheduler_config.get("partition_candidates")
                if isinstance(scheduler_config, dict)
                else None
            )
            if candidates is None:
                result = backend.submit(_HPC_SUBMIT_SCRIPT, remote_run_dir)
            else:
                resources = hpc_execution.get("resources")
                if not isinstance(candidates, list) or not isinstance(resources, dict):
                    raise BackendError(
                        "approved independent job has invalid scheduler routing"
                    )
                offset = index % len(candidates)
                rotated = [*candidates[offset:], *candidates[:offset]]
                submit_options = {
                    "partition_candidates": rotated,
                    "resources": resources,
                }
                if scheduler_config.get("memory_constraint") == "unreported":
                    submit_options["memory_constraint"] = "unreported"
                result = backend.submit(
                    _HPC_SUBMIT_SCRIPT, remote_run_dir, **submit_options
                )
            if result.returncode != 0 or result.job_id is None:
                raise BackendError(
                    result.stderr or result.stdout or "scheduler submission failed"
                )
            record = {
                "submission_id": submission_id,
                "job_id": result.job_id,
                "remote_run_dir": remote_run_dir,
            }
            routing = result.submission_routing
            if isinstance(routing, dict) and isinstance(
                routing.get("selected_partition"), str
            ):
                record["partition"] = routing["selected_partition"]
            submitted.append(record)
    except Exception:
        for record, (_, _, backend, _) in zip(submitted, staged):
            backend.cancel(str(record["job_id"]))
        raise
    return submitted


def _failure_salvage_outputs(
    plan: dict[str, Any], scheduled: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return the declared output subset that may be fetched on failure.

    Adapters cannot introduce new paths here. Every requested name must already
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


def _independent_submission_records(
    project: Project, step: StepRun, plan: dict[str, Any]
) -> list[dict[str, Any]]:
    path = attempt_directory(project, step.node_id, step.attempt) / _HPC_SUBMISSIONS
    if not path.is_file():
        raise BackendError("persisted independent scheduler submissions are missing")
    manifest = load_mapping(path)
    records = manifest.get("submissions")
    resolved = plan.get("hpc_executions")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("submission_strategy") != "independent-jobs"
        or not isinstance(records, list)
        or not isinstance(resolved, list)
        or len(records) != len(resolved)
    ):
        raise BackendError("independent scheduler submission manifest is invalid")
    expected: dict[str, dict[str, Any]] = {}
    for item in resolved:
        submission_id = item.get("submission_id") if isinstance(item, dict) else None
        execution = item.get("hpc_execution") if isinstance(item, dict) else None
        if not isinstance(submission_id, str) or not isinstance(execution, dict):
            raise BackendError("approved independent scheduler execution is invalid")
        expected[submission_id] = execution
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        submission_id = record.get("submission_id") if isinstance(record, dict) else None
        job_id = record.get("job_id") if isinstance(record, dict) else None
        remote_run_dir = record.get("remote_run_dir") if isinstance(record, dict) else None
        execution = expected.get(str(submission_id))
        workspace = execution.get("workspace") if isinstance(execution, dict) else None
        if (
            not isinstance(submission_id, str)
            or submission_id in seen
            or not isinstance(job_id, str)
            or not job_id.isdigit()
            or not isinstance(workspace, dict)
            or remote_run_dir != workspace.get("run_dir")
        ):
            raise BackendError("persisted independent scheduler job record changed")
        seen.add(submission_id)
        normalized.append({**record, "hpc_execution": execution})
    if seen != set(expected):
        raise BackendError("persisted independent scheduler job set changed")
    if step.job_id != ",".join(str(item["job_id"]) for item in normalized):
        raise BackendError("persisted aggregate scheduler job ids changed")
    return normalized


def _observe_independent_jobs(
    project: Project,
    step: StepRun,
    plan: dict[str, Any],
    scheduled: dict[str, Any],
    *,
    factory: SchedulerFactory | None = None,
) -> dict[str, Any]:
    records = _independent_submission_records(project, step, plan)
    contracts = {
        str(item["id"]): item for item in scheduled["submissions"]
    }
    observations: list[dict[str, Any]] = []
    for record in records:
        execution = record["hpc_execution"]
        backend = scheduler_from_cluster_record(
            execution.get("cluster_profile"), factory=factory
        )
        status = backend.status(str(record["job_id"]))
        raw_state = str(status.get("state", "UNKNOWN")).upper().split()[0]
        observations.append(
            {
                "submission_id": record["submission_id"],
                "job_id": record["job_id"],
                "scheduler_state": raw_state,
                "scheduler_detail": status.get("detail"),
            }
        )
    states = [str(item["scheduler_state"]) for item in observations]
    pending = {"PENDING", "CONFIGURING", "RESIZING", "SUSPENDED"}
    running = {"RUNNING", "COMPLETING"}
    successful = {"COMPLETED"}
    failures = {
        "FAILED",
        "TIMEOUT",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
        "CANCELLED",
        "REVOKED",
    }
    terminal = successful | failures
    result: dict[str, Any] = {
        "node_id": step.node_id,
        "job_id": step.job_id,
        "scheduler_state": states[0] if len(set(states)) == 1 else "MIXED",
        "scheduler_jobs": observations,
    }
    if any(state in running for state in states):
        result["target_state"] = RunState.RUNNING.value
        result["reason"] = "independent scheduler jobs are running or completing"
        return result
    if any(state not in terminal for state in states):
        if all(state in pending for state in states):
            if step.state != RunState.RUNNING.value:
                result["target_state"] = RunState.PENDING.value
            result["reason"] = "independent scheduler jobs are queued"
        else:
            result["reason"] = (
                "independent scheduler jobs have nonterminal or unknown mixed states"
            )
        return result

    all_completed = all(state == "COMPLETED" for state in states)
    finalizations: list[dict[str, Any]] = []
    for record in records:
        submission_id = str(record["submission_id"])
        execution = record["hpc_execution"]
        workspace = execution["workspace"]
        backend = scheduler_from_cluster_record(
            execution.get("cluster_profile"), factory=factory
        )
        outputs = list(contracts[submission_id]["fetch_outputs"])
        if not all_completed:
            outputs = [{**item, "required": False} for item in outputs]
        finalizations.append(
            {
                "submission_id": submission_id,
                "job_id": record["job_id"],
                "remote_run_dir": workspace["run_dir"],
                "outputs": _remote_output_inventory_items(
                    backend, workspace, outputs
                ),
            }
        )
    details: dict[str, Any] = {"submissions": finalizations}
    if all_completed:
        result["scheduler_state"] = "COMPLETED"
        result["reason"] = (
            "all independent scheduler jobs completed; approved outputs are ready "
            "for fetch and pinned scientific checks"
        )
    else:
        cancelled_only = all(state in successful | {"CANCELLED", "REVOKED"} for state in states)
        target = RunState.STOPPED if cancelled_only else RunState.FAIL
        details["failure_salvage"] = True
        details["terminal_target"] = target.value
        result["reason"] = (
            "all independent scheduler jobs are terminal; approved calculation-level "
            "failure salvage is ready"
        )
    result["adapter_finalization"] = details
    return result


def _observe_scheduled_step(
    project: Project,
    step: StepRun,
    *,
    factory: SchedulerFactory | None = None,
) -> dict[str, Any]:
    if step.backend == "ssh-slurm":
        try:
            pinned_plan, _, scheduled = _load_pinned_scheduled_plan(
                project, step
            )
            if scheduled.get("schema_version") == 4:
                return _observe_independent_jobs(
                    project,
                    step,
                    pinned_plan,
                    scheduled,
                    factory=factory,
                )
        except Exception as exc:
            return {
                "node_id": step.node_id,
                "job_id": step.job_id,
                "scheduler_state": "QUERY_ERROR",
                "reason": str(exc),
            }
    try:
        if step.backend == "ssh-slurm":
            pinned_plan, _, _ = _load_pinned_scheduled_plan(project, step)
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
        if approved_plan_path.is_file():
            try:
                approved_plan, capability_id, scheduled = _load_pinned_scheduled_plan(
                    project, step
                )
                salvage_outputs = _failure_salvage_outputs(approved_plan, scheduled)
                if salvage_outputs:
                    cluster_record, workspace = _approved_workspace(approved_plan, step)
                    backend = scheduler_from_cluster_record(cluster_record, factory=factory)
                    inventory = _remote_output_inventory_items(
                        backend, workspace, salvage_outputs
                    )
                    observation["adapter_finalization"] = {
                        "capability": capability_id,
                        "remote_run_dir": workspace["run_dir"],
                        "outputs": inventory,
                        "failure_salvage": True,
                        "terminal_target": target.value,
                    }
                    existing = [
                        str(item["remote_name"])
                        for item in inventory
                        if item.get("exists")
                    ]
                    detail = (
                        "approved failure salvage is ready for: " + ", ".join(existing)
                        if existing
                        else "no approved salvage output exists yet"
                    )
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
            try:
                approved_plan, capability_id, scheduled = _load_pinned_scheduled_plan(
                    project, step
                )
                cluster_record, workspace = _approved_workspace(approved_plan, step)
                backend = scheduler_from_cluster_record(cluster_record, factory=factory)
                inventory = _remote_output_inventory(backend, workspace, scheduled)
                observation["adapter_finalization"] = {
                    "capability": capability_id,
                    "remote_run_dir": workspace["run_dir"],
                    "outputs": inventory,
                }
                missing = [
                    str(item["remote_name"])
                    for item in inventory
                    if item["required"] and not item.get("exists")
                ]
                if missing:
                    observation["reason"] = (
                        "scheduler completed; required outputs are missing: "
                        + ", ".join(missing)
                    )
                else:
                    observation["reason"] = (
                        "scheduler completed; declared outputs are ready for fetch and "
                        "the built-in capability scientific checker"
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
        inventory.append({**item, **observed})
    return inventory


def _remote_output_inventory(
    backend: Any, workspace: dict[str, Any], scheduled: dict[str, Any]
) -> list[dict[str, Any]]:
    """Read whether each declared output exists."""

    return _remote_output_inventory_items(
        backend, workspace, list(scheduled["fetch_outputs"])
    )


def _load_pinned_scheduled_plan(
    project: Project,
    step: StepRun,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    attempt_dir = attempt_directory(project, step.node_id, step.attempt)
    plan_path = attempt_dir / "approved-plan.json"
    if not plan_path.is_file():
        raise ConfigError("approved scheduled plan is missing")
    plan = load_mapping(plan_path)
    if (
        plan.get("project_id") != step.project_id
        or plan.get("node_id") != step.node_id
        or plan.get("attempt") != step.attempt
    ):
        raise ApprovalError("stored run plan does not match this attempt")
    capability_id = plan.get("capability")
    if not isinstance(capability_id, str):
        raise ApprovalError("stored run plan lacks a capability id")
    capability(capability_id)
    scheduled = _scheduled_contract(
        project,
        capability_id,
        plan,
        node_id=step.node_id,
        attempt=step.attempt,
        verify_staged_sources=False,
    )
    return plan, capability_id, scheduled


def _attempt_node(project: Project, plan: dict[str, Any], step: StepRun) -> dict[str, Any]:
    node = dict(project.node(step.node_id))
    if node.get("backend_profile") is None and isinstance(
        plan.get("backend_profile"), str
    ):
        node["backend_profile"] = plan["backend_profile"]
    return node


def _write_independent_completion(
    attempt_dir: Path,
    step: StepRun,
    submission_ids: list[str],
) -> Path | None:
    calculations: list[dict[str, Any]] = []
    all_successful = True
    found = 0
    for submission_id in submission_ids:
        path = attempt_dir / "scheduler" / submission_id / "completion.json"
        if not path.is_file():
            all_successful = False
            continue
        completion = load_mapping(path)
        for key, expected in {
            "schema_version": 1,
            "project_id": step.project_id,
            "node_id": step.node_id,
            "attempt": step.attempt,
        }.items():
            if completion.get(key) != expected:
                raise StateError(
                    f"independent completion {submission_id} has invalid {key}"
                )
        found += 1
        if completion.get("status") != "COMPLETED" or completion.get("exit_code") != 0:
            all_successful = False
        records = completion.get("calculations")
        if not isinstance(records, list):
            raise StateError(
                f"independent completion {submission_id} lacks calculations"
            )
        calculations.extend(dict(item) for item in records if isinstance(item, dict))
    if found == 0:
        return None
    ids = [item.get("id") for item in calculations]
    if any(not isinstance(item, str) for item in ids) or len(set(ids)) != len(ids):
        raise StateError("independent completion calculation ids are invalid")
    aggregate = {
        "schema_version": 1,
        "status": "COMPLETED" if all_successful and found == len(submission_ids) else "FAILED",
        "exit_code": 0 if all_successful and found == len(submission_ids) else 1,
        "project_id": step.project_id,
        "node_id": step.node_id,
        "attempt": step.attempt,
        "calculations": calculations,
    }
    destination = attempt_dir / "completion.json"
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or load_mapping(destination) != aggregate:
            raise StateError("pre-existing aggregate completion differs")
    else:
        write_json_atomic(destination, aggregate)
    return destination


def _finalize_independent_jobs(
    project: Project,
    step: StepRun,
    change: dict[str, Any],
    plan: dict[str, Any],
    capability_id: str,
    scheduled: dict[str, Any],
    store: StateStore,
    *,
    factory: SchedulerFactory | None = None,
) -> StepRun:
    details = change.get("adapter_finalization")
    if not isinstance(details, dict) or not isinstance(details.get("submissions"), list):
        raise StateError("independent finalization details are invalid")
    attempt_dir = attempt_directory(project, step.node_id, step.attempt)
    records = _independent_submission_records(project, step, plan)
    contracts = {str(item["id"]): item for item in scheduled["submissions"]}
    observed = {
        str(item.get("submission_id")): item
        for item in details["submissions"]
        if isinstance(item, dict)
    }
    if set(observed) != set(contracts):
        raise StateError("independent finalization submission set changed")
    failure_salvage = details.get("failure_salvage") is True
    fetched: list[dict[str, Any]] = []
    fetch_errors: list[str] = []
    for record in records:
        submission_id = str(record["submission_id"])
        execution = record["hpc_execution"]
        workspace = execution["workspace"]
        observed_record = observed[submission_id]
        if observed_record.get("remote_run_dir") != workspace.get("run_dir"):
            raise StateError("approved independent remote workspace changed")
        outputs = list(contracts[submission_id]["fetch_outputs"])
        if failure_salvage:
            outputs = [{**item, "required": False} for item in outputs]
        backend = scheduler_from_cluster_record(
            execution.get("cluster_profile"), factory=factory
        )
        current = _remote_output_inventory_items(backend, workspace, outputs)
        for item in current:
            if not item.get("exists"):
                if item["required"]:
                    fetch_errors.append(
                        f"{submission_id}: required output is missing: {item['remote_name']}"
                    )
                continue
            destination = attempt_dir / str(item["local_name"])
            if destination.exists() or destination.is_symlink():
                if destination.is_symlink() or not destination.is_file():
                    raise StateError(
                        "pre-existing independent output is not an ordinary file: "
                        f"{submission_id}/{item['remote_name']}"
                    )
            else:
                backend.fetch_from(
                    str(workspace["run_dir"]),
                    str(item["remote_path"]),
                    destination,
                )
            fetched.append(
                artifact_record(destination)
                | {"role": str(item.get("role", "scheduler-output"))}
            )

    completion = _write_independent_completion(
        attempt_dir, step, [str(item["submission_id"]) for item in records]
    )
    if completion is not None:
        fetched.append(artifact_record(completion) | {"role": "scheduler-completion"})
    checked: dict[str, Any] | None = None
    collected: dict[str, Any] | None = None
    if failure_salvage:
        raw_target = details.get("terminal_target")
        if raw_target not in {RunState.FAIL.value, RunState.STOPPED.value}:
            raise StateError("independent failure salvage lacks a terminal target")
        final = RunState(str(raw_target))
        reason = str(change.get("reason", "scheduler terminal failure"))
        if fetch_errors:
            reason += "; " + "; ".join(fetch_errors)
    else:
        node = _attempt_node(project, plan, step)
        first_execution = records[0]["hpc_execution"]
        context = _adapter_context(project, node, step.attempt)
        context["execution"] = {
            "returncode": 0,
            "scheduler_state": "COMPLETED",
            "remote_output_inventory": [
                item for record in observed.values() for item in record["outputs"]
            ],
            "plan": plan["adapter_plan"],
            "hpc_execution": first_execution,
            "hpc_executions": [item["hpc_execution"] for item in records],
        }
        completion_error = (
            "aggregate scheduler completion is missing"
            if completion is None
            else _validate_hpc_completion(completion, step)
        )
        if fetch_errors:
            final = RunState.FAIL
            reason = "; ".join(fetch_errors)
        elif completion_error is not None:
            final = RunState.FAIL
            reason = completion_error
        else:
            adapter = load_adapter(capability_id)
            checked = adapter.check(context)
            if checked.get("status") != RunState.OK.value:
                final = RunState.FAIL
                reason = f"built-in capability completion check did not return OK: {checked}"
            else:
                collected = adapter.collect(context)
                if collected.get("status") != RunState.OK.value:
                    final = RunState.FAIL
                    reason = f"built-in capability collection did not return OK: {collected}"
                else:
                    normalized = _normalize_adapter_artifacts(
                        project, attempt_dir, collected.get("artifacts", [])
                    )
                    known = {str(item["uri"]) for item in fetched}
                    fetched.extend(
                        item for item in normalized if str(item["uri"]) not in known
                    )
                    final = RunState.OK
                    reason = (
                        "all scheduler jobs completed; output fetch and built-in capability "
                        "checks succeeded"
                    )

    initial_artifacts = _attempt_artifacts(project, step)
    artifacts = list(initial_artifacts)
    known_uris = {str(item["uri"]) for item in artifacts}
    artifacts.extend(item for item in fetched if str(item["uri"]) not in known_uris)
    if not failure_salvage and RunState(step.state) in {
        RunState.SUBMITTED,
        RunState.PENDING,
    }:
        step = store.transition(
            step.run_id,
            RunState.RUNNING,
            diagnostic="all scheduler jobs completed between observations",
        )
    _finalize_scheduler_manifest(
        project,
        step,
        final,
        reason,
        str(change.get("scheduler_state", "MIXED")),
        artifacts,
        result={"check": checked, "collection": collected},
    )
    return store.transition(
        step.run_id,
        final,
        diagnostic=None if final == RunState.OK else reason,
    )


def _finalize_scheduled_adapter(
    project: Project,
    step: StepRun,
    change: dict[str, Any],
    store: StateStore,
    *,
    factory: SchedulerFactory | None = None,
) -> StepRun:
    details = change.get("adapter_finalization")
    if not isinstance(details, dict):
        raise StateError("approved transition lacks adapter finalization details")
    plan, capability_id, scheduled = _load_pinned_scheduled_plan(project, step)
    if scheduled.get("schema_version") == 4:
        return _finalize_independent_jobs(
            project,
            step,
            change,
            plan,
            capability_id,
            scheduled,
            store,
            factory=factory,
        )
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
    fetched: list[dict[str, Any]] = []
    fetch_errors: list[str] = []
    for item in current_inventory:
        if not item.get("exists"):
            if item["required"]:
                fetch_errors.append(f"required output is missing: {item['remote_name']}")
            continue
        destination = attempt_dir / str(item["local_name"])
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink() or not destination.is_file():
                raise StateError(
                    "pre-existing fetched output is not an ordinary file: "
                    f"{item['remote_name']}"
                )
            fetched.append(
                artifact_record(destination)
                | {"role": str(item.get("role", "scheduler-output"))}
            )
            continue
        backend.fetch_from(
            str(workspace["run_dir"]), str(item["remote_path"]), destination
        )
        fetched.append(
            artifact_record(destination)
            | {"role": str(item.get("role", "scheduler-output"))}
        )

    reason: str
    checked: dict[str, Any] | None = None
    collected: dict[str, Any] | None = None
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
        node = _attempt_node(project, plan, step)
        context = _adapter_context(project, node, step.attempt)
        context["execution"] = {
            "returncode": 0,
            "scheduler_state": "COMPLETED",
            "remote_output_inventory": current_inventory,
            "plan": plan["adapter_plan"],
            "hpc_execution": hpc_execution,
        }
        adapter = load_adapter(capability_id)
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
                reason = f"built-in capability completion check did not return OK: {checked}"
            else:
                collected = adapter.collect(context)
                if collected.get("status") != RunState.OK.value:
                    final = RunState.FAIL
                    reason = f"built-in capability collection did not return OK: {collected}"
                else:
                    normalized = _normalize_adapter_artifacts(
                        project, attempt_dir, collected.get("artifacts", [])
                    )
                    known = {str(item["uri"]) for item in fetched}
                    fetched.extend(
                        item for item in normalized if str(item["uri"]) not in known
                    )
                    final = RunState.OK
                    reason = (
                        "scheduler completed; output fetch and built-in capability checks succeeded"
                    )

    initial_artifacts = _attempt_artifacts(project, step)
    artifacts = list(initial_artifacts)
    known_uris = {str(item["uri"]) for item in artifacts}
    artifacts.extend(item for item in fetched if str(item["uri"]) not in known_uris)
    current_state = RunState(step.state)
    if not failure_salvage and current_state in {RunState.SUBMITTED, RunState.PENDING}:
        step = store.transition(
            step.run_id,
            RunState.RUNNING,
            diagnostic="scheduler completed between approved observations",
        )
    scheduler_state = str(change.get("scheduler_state", "UNKNOWN"))
    _finalize_scheduler_manifest(
        project,
        step,
        final,
        reason,
        scheduler_state,
        artifacts,
        result={"check": checked, "collection": collected},
    )
    return store.transition(
        step.run_id,
        final,
        diagnostic=None if final == RunState.OK else reason,
    )


def _finalize_scheduler_manifest(
    project: Project,
    step: StepRun,
    target: RunState,
    reason: str,
    scheduler_state: str,
    artifacts: list[dict[str, Any]],
    *,
    result: dict[str, Any] | None = None,
) -> Path | None:
    initial = attempt_directory(project, step.node_id, step.attempt) / "run-manifest.json"
    if not initial.is_file():
        return None
    manifest = load_mapping(initial)
    if result is not None:
        manifest["result"] = result_summary(result)
    manifest["state"] = target.value
    manifest["state_reason"] = reason
    manifest["artifacts"] = artifacts or manifest.get("artifacts", [])
    manifest["timestamps"]["updated_at"] = utc_now()
    if target in {RunState.OK, RunState.FAIL, RunState.STOPPED}:
        manifest["timestamps"]["finished_at"] = utc_now()
    if isinstance(manifest.get("job"), dict):
        manifest["job"]["scheduler_state"] = scheduler_state
    destination = initial.with_name("run-manifest.final.json")
    write_json_atomic(destination, manifest)
    return destination


def _validate_hpc_completion(path: Path, step: StepRun) -> str | None:
    if not path.is_file():
        return "scheduler completion.json is missing"
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
