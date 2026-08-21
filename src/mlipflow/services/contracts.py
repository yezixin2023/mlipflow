"""Pure validation and context assembly shared by both execution paths.

Nothing here talks to a backend or mutates workflow state; these are the checks
that must hold identically whether a node runs locally or through a scheduler.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from ..artifacts import artifact as artifact_record
from ..config import Project
from ..errors import ConfigError, PluginError
from ..hpc import EXECUTION_MODELS
from ..io import load_mapping
from ..plugins import PluginSpec
from ..portable import PortableRoots, to_runtime
from ..site import ClusterProfile, load_site_config
from ..state import RunState, StateStore, utc_now
from .paths import attempt_directory, state_path


_SAFE_REMOTE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+\-]*")


def _safe_remote_relative(value: Any) -> bool:
    """True for a bounded relative path such as ``calc-0001/POSCAR``.

    A scheduled node may now carry several calculations, so staged and fetched
    names are relative paths rather than basenames — otherwise every
    calculation's ``POSCAR`` would collide.  Each segment must still be an
    ordinary safe name, which keeps absolute paths, ``..``, empty segments and
    anything shell-significant out.
    """

    if not isinstance(value, str) or not value or value.endswith("/"):
        return False
    segments = value.split("/")
    return all(_SAFE_REMOTE_NAME.fullmatch(segment) for segment in segments)


def _portable_roots(
    project: Project,
    plugin: PluginSpec | None = None,
    node_id: str | None = None,
    attempt: int | None = None,
) -> PortableRoots:
    """Build the root set used to make adapter-authored plan fields portable."""

    return PortableRoots(
        project_root=project.root,
        attempt_dir=(
            attempt_directory(project, node_id, attempt)
            if node_id is not None and attempt is not None
            else None
        ),
        plugin_dir=plugin.path.parent if plugin is not None else None,
        package_dir=Path(__file__).resolve().parents[1],
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _project_scoped_result_path(project: Project, path: Path) -> Path:
    if path.is_symlink():
        raise ConfigError(f"result manifest must not be a symlink: {path}")
    resolved = path.resolve()
    try:
        resolved.relative_to(project.root.resolve())
    except ValueError as exc:
        raise ConfigError(f"result manifest escapes the project root: {path}") from exc
    return resolved


def _load_result(result_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
        artifacts.append({**artifact_record(path), "role": item.get("role", "output")})
    return result, artifacts


def _scheduled_contract(
    project: Project,
    plugin: PluginSpec,
    plan: dict[str, Any],
    *,
    node_id: str | None = None,
    attempt: int | None = None,
    verify_staged_sources: bool = True,
) -> dict[str, Any]:
    # Staging needs real paths, so resolve portable plan values before validation.
    adapter_plan = to_runtime(
        plan.get("adapter_plan"),
        _portable_roots(project, plugin, node_id, attempt),
    )
    scheduled = adapter_plan.get("scheduled_execution") if isinstance(adapter_plan, dict) else None
    if isinstance(scheduled, dict) and scheduled.get("schema_version") == 4:
        if scheduled.get("submission_strategy") != "independent-jobs":
            raise PluginError(
                "scheduled_execution schema_version 4 requires independent-jobs"
            )
        execution_model = scheduled.get("execution_model")
        template_family = scheduled.get("template_family")
        submissions = scheduled.get("submissions")
        if not isinstance(submissions, list) or not submissions:
            raise PluginError(
                "scheduled_execution.submissions must be a non-empty list"
            )
        normalized_submissions: list[dict[str, Any]] = []
        submission_ids: set[str] = set()
        local_names: set[str] = set()
        for index, submission in enumerate(submissions):
            if not isinstance(submission, dict):
                raise PluginError(f"scheduled submission {index} must be a mapping")
            submission_id = submission.get("id")
            if (
                not isinstance(submission_id, str)
                or not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", submission_id)
                or submission_id in submission_ids
            ):
                raise PluginError(
                    f"scheduled submission {index} has an unsafe or duplicate id"
                )
            submission_ids.add(submission_id)
            unit = {
                "schema_version": 3,
                "execution_model": execution_model,
                "template_family": template_family,
                "template_variables": submission.get("template_variables", {}),
                "staged_files": submission.get("staged_files"),
                "fetch_outputs": submission.get("fetch_outputs"),
            }
            normalized = _scheduled_contract(
                project,
                plugin,
                {"adapter_plan": {"scheduled_execution": unit}},
                node_id=node_id,
                attempt=attempt,
                verify_staged_sources=verify_staged_sources,
            )
            outputs: list[dict[str, Any]] = []
            for item in normalized["fetch_outputs"]:
                adjusted = dict(item)
                if item["remote_name"] in {
                    "completion.json",
                    "stdout.log",
                    "stderr.log",
                }:
                    adjusted["local_name"] = (
                        f"scheduler/{submission_id}/{item['local_name']}"
                    )
                local_name = str(adjusted["local_name"])
                if local_name in local_names:
                    raise PluginError(
                        "scheduled independent jobs have duplicate local output: "
                        + local_name
                    )
                local_names.add(local_name)
                outputs.append(adjusted)
            normalized_submissions.append(
                {
                    "id": submission_id,
                    "template_variables": normalized["template_variables"],
                    "staged_files": normalized["staged_files"],
                    "fetch_outputs": outputs,
                }
            )
        return {
            "schema_version": 4,
            "submission_strategy": "independent-jobs",
            "execution_model": execution_model,
            "template_family": template_family,
            "submissions": normalized_submissions,
        }
    if not isinstance(scheduled, dict) or scheduled.get("schema_version") != 3:
        raise PluginError(
            "scheduled adapter plan requires scheduled_execution schema_version 3 or 4"
        )
    schema_version = 3
    execution_model = scheduled.get("execution_model")
    if execution_model not in EXECUTION_MODELS:
        raise PluginError(
            "scheduled_execution.execution_model must be single-python or mpi"
        )
    template_family = scheduled.get("template_family")
    if not isinstance(template_family, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9._-]*", template_family
    ):
        raise PluginError("scheduled_execution.template_family is unsafe")
    template_variables = scheduled.get("template_variables", {})
    if not isinstance(template_variables, dict) or any(
        not isinstance(name, str)
        or not re.fullmatch(r"PLUGIN_[A-Z0-9_]+", name)
        or not isinstance(value, str)
        or not value
        or not re.fullmatch(r"[A-Za-z0-9._,+-]+", value)
        for name, value in template_variables.items()
    ):
        raise PluginError(
            "scheduled_execution.template_variables must be safe PLUGIN_* string values"
        )
    staged = scheduled.get("staged_files")
    if not isinstance(staged, list) or not staged:
        raise PluginError("scheduled_execution.staged_files must be a non-empty list")
    allowed_roots = (
        project.root.resolve(),
        plugin.path.parent.resolve(),
        Path(__file__).resolve().parents[1],
    )
    names: set[str] = set()
    normalized_stage: list[dict[str, Any]] = []
    for index, item in enumerate(staged):
        if not isinstance(item, dict):
            raise PluginError(f"scheduled staged file {index} must be a mapping")
        source_value = item.get("source")
        remote_name = item.get("remote_name")
        if not isinstance(source_value, str) or not isinstance(remote_name, str):
            raise PluginError(f"scheduled staged file {index} lacks source/remote_name")
        # Uniqueness is on the full relative path, so calc-0001/POSCAR and
        # calc-0002/POSCAR coexist while a genuine duplicate is still refused.
        if not _safe_remote_relative(remote_name) or remote_name in names:
            raise PluginError(f"unsafe or duplicate remote staging name: {remote_name!r}")
        names.add(remote_name)
        source = Path(source_value).expanduser().absolute()
        if verify_staged_sources:
            if source.is_symlink() or not source.is_file():
                raise PluginError(f"staging source must be an ordinary file: {source}")
            resolved = source.resolve()
            if not any(_is_within(resolved, root) for root in allowed_roots):
                raise PluginError(f"staging source is outside the project/plugin roots: {source}")
        normalized_stage.append({"source": str(source), "remote_name": remote_name})
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
        # ``remote_path`` defaults to output/<remote_name>; an adapter may state
        # it explicitly to reach a per-calculation log under logs/.  Either way
        # it stays a bounded relative path inside the attempt workspace.
        explicit_path = item.get("remote_path")
        if (
            not isinstance(remote_name, str)
            or not _safe_remote_relative(remote_name)
            or remote_name in output_names
            or not isinstance(local_name, str)
            or not _safe_remote_relative(local_name)
            or type(required) is not bool
            or isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum < 1
            or (explicit_path is not None and not _safe_remote_relative(explicit_path))
        ):
            raise PluginError(f"invalid scheduled fetch output {index}")
        output_names.add(remote_name)
        normalized_outputs.append(
            {
                **item,
                "remote_path": explicit_path
                if isinstance(explicit_path, str)
                else f"output/{remote_name}",
            }
        )
    normalized_outputs.extend(
        [
            {
                "remote_name": "completion.json",
                "remote_path": "completion.json",
                "local_name": "completion.json",
                "required": True,
                "max_bytes": 1024 * 1024,
                "role": "scheduler-completion",
            },
            {
                "remote_name": "stdout.log",
                "remote_path": "logs/stdout.log",
                "local_name": "stdout.log",
                "required": False,
                "max_bytes": 16 * 1024 * 1024,
                "role": "scheduler-log",
            },
            {
                "remote_name": "stderr.log",
                "remote_path": "logs/stderr.log",
                "local_name": "stderr.log",
                "required": False,
                "max_bytes": 16 * 1024 * 1024,
                "role": "scheduler-log",
            },
        ]
    )
    return {
        "schema_version": schema_version,
        "execution_model": execution_model,
        "template_family": template_family,
        "template_variables": dict(template_variables),
        "staged_files": normalized_stage,
        "fetch_outputs": normalized_outputs,
    }


def _cluster_profile(
    node: dict[str, Any], site_path: Path | None
) -> ClusterProfile:
    site = load_site_config(site_path)
    return site.cluster(node.get("backend_profile"))


def _planned_attempt(project: Project, node_id: str) -> int:
    database = state_path(project)
    if not database.is_file():
        return 1
    with StateStore(database, readonly=True) as store:
        return store.latest_step(project.project_id, node_id).attempt


def _resolve_collected_artifact_bindings(
    value: Any,
    *,
    project: Project,
    upstream_artifacts: list[dict[str, Any]],
) -> Any:
    """Resolve explicit ``from_node``/``role`` inputs from final OK artifacts."""

    if isinstance(value, list):
        return [
            _resolve_collected_artifact_bindings(
                item, project=project, upstream_artifacts=upstream_artifacts
            )
            for item in value
        ]
    if not isinstance(value, dict):
        return value
    if "from_node" not in value and "role" not in value:
        return {
            key: _resolve_collected_artifact_bindings(
                item, project=project, upstream_artifacts=upstream_artifacts
            )
            for key, item in value.items()
        }
    if set(value) - {"from_node", "role", "resolve"}:
        raise PluginError(
            "collected artifact bindings accept only from_node, role, and resolve"
        )
    node_id = value.get("from_node")
    role = value.get("role")
    resolution = value.get("resolve", "artifact")
    if not isinstance(node_id, str) or not node_id:
        raise PluginError("collected artifact binding requires a non-empty from_node")
    if not isinstance(role, str) or not role:
        raise PluginError("collected artifact binding requires a non-empty role")
    if resolution not in {"artifact", "parent"}:
        raise PluginError("collected artifact binding resolve must be artifact or parent")

    matches: set[str] = set()
    for record in upstream_artifacts:
        if record.get("node_id") != node_id or record.get("state") != RunState.OK.value:
            continue
        for artifact in record.get("artifacts", []):
            if isinstance(artifact, dict) and artifact.get("role") == role:
                raw_path = artifact.get("path")
                if isinstance(raw_path, str) and raw_path:
                    matches.add(raw_path)
    if len(matches) != 1:
        raise PluginError(
            f"artifact binding {node_id}:{role} requires exactly one unique artifact "
            f"from a final OK direct dependency; found {len(matches)}"
        )
    raw_path = matches.pop()
    if raw_path.startswith("file://"):
        parsed = urlparse(raw_path)
        if parsed.netloc not in {"", "localhost"}:
            raise PluginError(f"artifact binding {node_id}:{role} is not a local artifact")
        raw_path = unquote(parsed.path)
    elif "://" in raw_path:
        raise PluginError(f"artifact binding {node_id}:{role} is not a local artifact")
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = project.root / candidate
    if candidate.is_symlink() or not candidate.exists():
        raise PluginError(f"artifact binding {node_id}:{role} is missing or a symlink")
    resolved = candidate.resolve()
    if not _is_within(resolved, project.root.resolve()):
        raise PluginError(f"artifact binding {node_id}:{role} escapes the project root")
    if resolution == "parent":
        resolved = resolved.parent
    return str(resolved)


def _adapter_context(project: Project, node: dict[str, Any], attempt: int) -> dict[str, Any]:
    upstream_artifacts: list[dict[str, Any]] = []
    database = state_path(project)
    if database.is_file():
        with StateStore(database, readonly=True) as store:
            for dependency_id in node.get("needs", []):
                for dependency in store.steps_for_node(
                    project.project_id, str(dependency_id)
                ):
                    artifacts = [
                        {"role": item["role"], "path": item["uri"]}
                        for item in store.artifacts(dependency.run_id)
                    ]
                    if artifacts:
                        upstream_artifacts.append(
                            {
                                "node_id": dependency.node_id,
                                "plugin_id": dependency.plugin_id.split("@", 1)[0],
                                "attempt": dependency.attempt,
                                "state": dependency.state,
                                "attempt_dir": str(
                                    attempt_directory(
                                        project,
                                        dependency.node_id,
                                        dependency.attempt,
                                    )
                                ),
                                "artifacts": artifacts,
                            }
                        )
    raw_inputs = node.get("inputs", {})
    inputs = dict(raw_inputs) if isinstance(raw_inputs, dict) else raw_inputs
    if isinstance(inputs, dict):
        inputs = _resolve_collected_artifact_bindings(
            inputs,
            project=project,
            upstream_artifacts=upstream_artifacts,
        )
    if (
        isinstance(inputs, dict)
        and str(node.get("uses", "")).split("@", 1)[0] == "ionic-transport"
    ):
        explicit = inputs.get("input_paths", [])
        input_paths = list(explicit) if isinstance(explicit, list) else explicit
        if isinstance(input_paths, list):
            input_paths.extend(
                str(record["attempt_dir"])
                for record in upstream_artifacts
                if (
                    record["plugin_id"] in {"ase-md", "lammps-md"}
                    and any(
                        item.get("role") == "trajectory"
                        for item in record["artifacts"]
                        if isinstance(item, dict)
                    )
                )
                or (
                    record["plugin_id"] == "dft-labeling"
                    and record["state"] == RunState.OK.value
                    and any(
                        item.get("role") == "aimd-trajectory"
                        for item in record["artifacts"]
                        if isinstance(item, dict)
                    )
                )
            )
            inputs["input_paths"] = list(dict.fromkeys(input_paths))
    return {
        "project_root": str(project.root),
        "project_path": str(project.path),
        "attempt_dir": str(attempt_directory(project, str(node["id"]), attempt)),
        "inputs": inputs,
        "parameters": node.get("parameters", {}),
        "backend": node.get("backend", "local"),
        "resources": node.get("resources", {}),
        "upstream_artifacts": upstream_artifacts,
    }


def _normalize_adapter_artifacts(
    project: Project, attempt_dir: Path, raw_artifacts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in raw_artifacts:
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
            record = artifact_record(resolved)
            record["role"] = role
            if isinstance(item.get("media_type"), str):
                record["media_type"] = item["media_type"]
            if isinstance(item.get("metadata"), dict):
                record["metadata"] = item["metadata"]
            normalized.append(record)
            continue
        uri = item.get("uri")
        if not isinstance(uri, str):
            raise PluginError(f"adapter artifact {role!r} needs either path or uri")
        normalized.append(
            {
                "role": role,
                "uri": uri,
                "media_type": item.get("media_type"),
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
    }
