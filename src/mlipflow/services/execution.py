"""Attempt execution: the three lifecycle shapes a READY node can take.

``replay`` parses existing evidence and runs no numerics.  ``local`` is a single
synchronous approval.  The scheduled path is asynchronous and delegates to
``scheduled.py`` after submission; its second approval happens in ``advance``.

These are deliberately not collapsed into one polymorphic call: a synchronous
run and a two-approval remote run are different lifecycles, not two
implementations of one.
"""

from __future__ import annotations

from typing import Any

from ..artifacts import fingerprint
from ..backends import LocalBackend
from ..config import Project
from ..errors import BackendError, ConfigError, PluginError
from ..io import write_json_atomic, write_text_atomic
from ..manifests import run_manifest
from ..planning import resolve_reference
from ..portable import to_runtime
from ..plugins import PluginSpec, load_adapter
from ..state import RunState, StateStore
from .backend_factory import SchedulerFactory
from .contracts import (
    _adapter_context,
    _portable_roots,
    _load_result,
    _manifest_context,
    _normalize_adapter_artifacts,
    _project_scoped_result_path,
)
from .paths import attempt_directory
from .scheduled import (
    _HPC_RUN_SCRIPT,
    _HPC_SUBMIT_SCRIPT,
    _stage_and_submit_scheduled_adapter,
)


def _execute_ready(
    project: Project,
    node: dict[str, Any],
    plugin: PluginSpec,
    plan: dict[str, Any],
    store: StateStore,
    run_id: str,
    attempt: int,
    *,
    factory: SchedulerFactory | None = None,
) -> dict[str, Any]:
    node_id = str(node["id"])
    mode = str(node.get("mode", "execute"))
    backend = str(node.get("backend", "local"))
    directory = attempt_directory(project, node_id, attempt)
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
            # The signed plan holds portable tokens; runtime path resolution
            # happens here, so the adapter still sees this machine's absolute
            # paths in argv, cwd and its own echoed plan.
            adapter_plan = to_runtime(
                plan.get("adapter_plan"),
                _portable_roots(project, plugin, node_id, attempt),
            )
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
        adapter_plan = plan.get("adapter_plan")
        command: list[str]
        if isinstance(adapter_plan, dict):
            if backend != "ssh-slurm":
                raise BackendError(
                    "scheduled adapters currently require the site/template ssh-slurm contract"
                )
            hpc_execution = plan.get("hpc_execution")
            workspace = (
                hpc_execution.get("workspace")
                if isinstance(hpc_execution, dict)
                else None
            )
            if not isinstance(workspace, dict) or not isinstance(
                workspace.get("run_dir"), str
            ):
                raise BackendError("approved plan lacks an exact remote attempt workspace")
            remote_dir = str(workspace["run_dir"])
            result, observed_remote_dir, command = _stage_and_submit_scheduled_adapter(
                project, node, plugin, plan, directory, factory=factory
            )
            if observed_remote_dir != remote_dir:
                raise BackendError("staged remote directory differs from the approved target")
        else:
            raise BackendError(
                "scheduled execution requires an adapter scientific contract plus "
                "site-resolved remote templates; parameters.submit_script/remote_cwd "
                "are not supported"
            )
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
                (directory / _HPC_SUBMIT_SCRIPT, "scheduler-script"),
                (directory / _HPC_RUN_SCRIPT, "application-script"),
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
