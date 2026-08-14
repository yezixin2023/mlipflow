"""LAMMPS prepare + reviewed ssh-slurm execute adapter.

Local preparation remains delegated to the v0.1 adapter, with its argv upgraded
to the execution-ready v2 preparation wrapper. Scheduled execute consumes only a
fingerprinted prepared bundle and a site-owned model artifact; cluster paths,
LAMMPS binaries, launchers and environment setup never enter the project.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

PLUGIN_ID = "lammps-md"
PREPARE = "lammps-prepare"
EXECUTE = "execute"
BACKEND = "ssh-slurm"
PREPARATION_CONTRACT = "lammps-md-input-v2"
FRAMEWORKS = {"deepmd", "mace", "m3gnet"}
TARGETS = {"cpu", "gpu"}
TEMPLATE_FAMILIES = {
    (framework, target): f"lammps-{framework}-{target}"
    for framework in FRAMEWORKS
    for target in TARGETS
}
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_INPUT_BYTES = 256 * 1024 * 1024
MAX_LOG_BYTES = 512 * 1024 * 1024
MAX_TRAJECTORY_BYTES = 8 * 1024 * 1024 * 1024
MAX_FINAL_DATA_BYTES = 2 * 1024 * 1024 * 1024
MAX_RESTART_BYTES = 8 * 1024 * 1024 * 1024


def _load_legacy():
    path = Path(__file__).resolve().with_name("adapter.py")
    spec = importlib.util.spec_from_file_location("_mlipflow_lammps_prepare_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load bundled lammps-md adapter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = _load_legacy()


def _diagnostic(level: str, code: str, message: str) -> dict[str, str]:
    return {"level": level, "code": code, "message": message}


def _errors(items: list[dict[str, str]]) -> list[dict[str, str]]:
    return [item for item in items if item.get("level") == "error"]


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _plain(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and "\x00" not in value
        and "\n" not in value
        and "\r" not in value
    )


def _fingerprint(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _ordinary_file(path: Path, max_bytes: int | None = None) -> bool:
    try:
        if path.is_symlink() or not path.is_file():
            return False
        size = path.stat().st_size
        return size > 0 and (max_bytes is None or size <= max_bytes)
    except OSError:
        return False


def _safe_relative(value: Any) -> bool:
    if not _plain(value) or "\\" in str(value):
        return False
    path = PurePosixPath(str(value))
    return not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)


def _project_file(root: Path, value: Any) -> Path:
    if not _safe_relative(value):
        raise ValueError("project file reference must be a safe relative path")
    candidate = (root / str(value)).absolute()
    try:
        candidate.resolve().relative_to(root.resolve())
    except (OSError, ValueError) as exc:
        raise ValueError("project file escapes project root") from exc
    return candidate


def _ordinary_project_file(path: Path, root: Path, max_bytes: int) -> bool:
    try:
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= max_bytes:
            return False
        path.resolve().relative_to(root.resolve())
        current = path
        while current != root:
            if current.is_symlink():
                return False
            current = current.parent
        return True
    except (OSError, ValueError):
        return False


def _project_config(root: Path) -> Path:
    candidates = [root / name for name in ("project.yaml", "project.yml", "project.json")]
    found = [path for path in candidates if path.is_file() and not path.is_symlink()]
    if len(found) != 1:
        raise ValueError("project root must contain exactly one project.yaml/project.yml/project.json")
    return found[0]


def _read_json(path: Path, max_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    if not _ordinary_file(path, max_bytes):
        raise ValueError(f"missing, unsafe or oversized JSON file: {path.name}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name} must contain an object")
    return raw


def _generated(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = manifest.get("generated_files")
    if not isinstance(raw, list):
        raise ValueError("prepared manifest generated_files must be a list")
    records: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict) or not _plain(item.get("name")):
            raise ValueError("generated file record is invalid")
        name = str(item["name"])
        if name in records:
            raise ValueError(f"duplicate generated file: {name}")
        if not _fingerprint(item.get("sha256")):
            raise ValueError(f"generated file fingerprint is invalid: {name}")
        if isinstance(item.get("size_bytes"), bool) or not isinstance(item.get("size_bytes"), int) or item["size_bytes"] <= 0:
            raise ValueError(f"generated file size is invalid: {name}")
        records[name] = item
    return records


def _launcher(manifest: dict[str, Any], target: str) -> dict[str, Any]:
    raw = manifest.get("launchers")
    if not isinstance(raw, list):
        raise ValueError("prepared manifest launchers must be a list")
    matches = [item for item in raw if isinstance(item, dict) and item.get("target") == target]
    if len(matches) != 1:
        raise ValueError(f"prepared manifest must contain exactly one {target} launcher")
    launcher = matches[0]
    argv = launcher.get("argv_after_executable")
    if launcher.get("executable") != "site-owned-lammps" or not isinstance(argv, list) or not argv:
        raise ValueError("prepared launcher contract is invalid")
    if any(not _plain(item) for item in argv):
        raise ValueError("prepared launcher argv contains an invalid token")
    if argv.count("<site-resolved-model-file>") != 1:
        raise ValueError("prepared launcher must contain exactly one site model placeholder")
    return launcher


def _staged(path: Path, remote_name: str) -> dict[str, Any]:
    return {
        "source": str(path.absolute()),
        "remote_name": remote_name,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "sensitive": False,
        "fetch_allowed": False,
    }


def _positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _nonnegative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _resources(context: dict[str, Any], target: str | None, framework: str | None = None) -> list[dict[str, str]]:
    resources = context.get("resources")
    if not isinstance(resources, dict):
        return [_diagnostic("error", "lammps.resources", "resources must be an object")]
    diagnostics: list[dict[str, str]] = []
    if set(resources) != {"cpus", "gpus", "memory", "walltime"}:
        return [_diagnostic("error", "lammps.resource_fields", "ssh-slurm resources must contain exactly cpus, gpus, memory, walltime")]
    if not _positive_int(resources.get("cpus")):
        diagnostics.append(_diagnostic("error", "lammps.cpus", "resources.cpus must be positive"))
    if not _nonnegative_int(resources.get("gpus")):
        diagnostics.append(_diagnostic("error", "lammps.gpus", "resources.gpus must be non-negative"))
    for key in ("memory", "walltime"):
        if not _plain(resources.get(key)):
            diagnostics.append(_diagnostic("error", f"lammps.{key}", f"resources.{key} must be a non-empty string"))
    if target == "cpu" and resources.get("gpus") != 0:
        diagnostics.append(_diagnostic("error", "lammps.cpu_gpus", "target=cpu requires resources.gpus=0"))
    if target == "gpu" and not _positive_int(resources.get("gpus")):
        diagnostics.append(_diagnostic("error", "lammps.gpu_resource", "target=gpu requires resources.gpus>=1"))
    if target == "gpu" and framework in {"mace", "m3gnet"} and resources.get("gpus") != 1:
        diagnostics.append(_diagnostic("error", "lammps.single_gpu", "MACE/MatGL GPU execution is pinned to one GPU in this contract"))
    return diagnostics


def _validate_execute(context: dict[str, Any]) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    if context.get("backend") != BACKEND:
        diagnostics.append(_diagnostic("error", "lammps.backend", "operation=execute requires backend=ssh-slurm"))
    inputs = context.get("inputs")
    parameters = context.get("parameters")
    if not isinstance(inputs, dict):
        diagnostics.append(_diagnostic("error", "lammps.inputs", "inputs must be an object"))
        inputs = {}
    if not isinstance(parameters, dict):
        diagnostics.append(_diagnostic("error", "lammps.parameters", "parameters must be an object"))
        parameters = {}
    if set(inputs) != {"lammps_input_manifest"}:
        diagnostics.append(_diagnostic("error", "lammps.execute_inputs", "execute requires exactly inputs.lammps_input_manifest"))
    if set(parameters) != {"operation", "target", "input_manifest_fingerprint"}:
        diagnostics.append(_diagnostic("error", "lammps.execute_parameters", "execute parameters must be exactly operation, target, input_manifest_fingerprint"))
    if parameters.get("operation") != EXECUTE:
        diagnostics.append(_diagnostic("error", "lammps.operation", "operation must be execute"))
    target = parameters.get("target")
    if target not in TARGETS:
        diagnostics.append(_diagnostic("error", "lammps.target", "target must be cpu or gpu"))
    if not _fingerprint(parameters.get("input_manifest_fingerprint")):
        diagnostics.append(_diagnostic("error", "lammps.input_manifest_fingerprint", "input_manifest_fingerprint must be sha256:<64 lowercase hex>"))
    diagnostics.extend(_resources(context, str(target) if target in TARGETS else None))
    return diagnostics


def _prepare_plan(context: dict[str, Any]) -> dict[str, Any]:
    plan = legacy.Adapter().plan(context)
    if plan.get("status") != "READY":
        return plan
    wrapper = Path(__file__).resolve().with_name("lammps_prepare_v2.py")
    old_wrapper = Path(__file__).resolve().with_name("lammps_prepare.py")
    if not _ordinary_file(wrapper) or not _ordinary_file(old_wrapper):
        return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": [_diagnostic("error", "lammps.prepare_wrapper", "execution-ready preparation helpers are missing")]}
    argv = plan.get("argv")
    if not isinstance(argv, list) or len(argv) < 2:
        return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": [_diagnostic("error", "lammps.prepare_argv", "legacy preparation plan has invalid argv")]}
    argv = list(argv)
    argv[1] = str(wrapper)
    plan["argv"] = argv
    fingerprints = plan.setdefault("input_fingerprints", {})
    if isinstance(fingerprints, dict):
        fingerprints["prepare_wrapper"] = _sha256(wrapper)
        fingerprints["legacy_prepare_wrapper"] = _sha256(old_wrapper)
    summary = plan.setdefault("approval_summary", {})
    if isinstance(summary, dict):
        summary["preparation_contract"] = PREPARATION_CONTRACT
    return plan


def _execute_plan(context: dict[str, Any]) -> dict[str, Any]:
    diagnostics = _validate_execute(context)
    if diagnostics:
        return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": diagnostics}
    root = Path(str(context["project_root"])).expanduser().absolute()
    inputs = _mapping(context["inputs"])
    parameters = _mapping(context["parameters"])
    try:
        project = _project_config(root)
        manifest_path = _project_file(root, inputs["lammps_input_manifest"])
    except ValueError as exc:
        return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": [_diagnostic("error", "lammps.project_inputs", str(exc))]}
    if not _ordinary_project_file(project, root, MAX_JSON_BYTES) or not _ordinary_project_file(manifest_path, root, MAX_JSON_BYTES):
        return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": [_diagnostic("error", "lammps.project_files", "project and prepared manifest must be ordinary bounded project files")]}
    manifest_sha = _sha256(manifest_path)
    if manifest_sha != parameters["input_manifest_fingerprint"]:
        return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": [_diagnostic("error", "lammps.manifest_identity", "input_manifest_fingerprint differs from the prepared manifest")]}
    try:
        manifest = _read_json(manifest_path)
        if (
            manifest.get("schema_version") != 1
            or manifest.get("plugin_id") != PLUGIN_ID
            or manifest.get("operation") != PREPARE
            or manifest.get("status") != "OK"
            or manifest.get("preparation_contract") != PREPARATION_CONTRACT
            or manifest.get("runtime_model_variable") != "MODEL_FILE"
        ):
            raise ValueError("prepared manifest is not execution-ready")
        model = _mapping(manifest.get("model"))
        md = _mapping(manifest.get("md"))
        framework = model.get("framework")
        target = str(parameters["target"])
        if framework not in FRAMEWORKS:
            raise ValueError("prepared model framework is unsupported")
        if target not in md.get("targets", []):
            raise ValueError("selected target was not prepared")
        if model.get("kind") != "file" or not _plain(model.get("model_id")) or not _safe_relative(model.get("relative_path")) or not _fingerprint(model.get("fingerprint")) or not _plain(model.get("artifact_format")):
            raise ValueError("prepared model identity is incomplete")
        steps = md.get("steps")
        timestep = md.get("timestep_fs")
        if not _positive_int(steps) or not isinstance(timestep, (int, float)) or isinstance(timestep, bool) or not math.isfinite(float(timestep)) or float(timestep) <= 0:
            raise ValueError("prepared MD step/timestep identity is invalid")
        marker = manifest.get("completion_marker")
        if marker != f"MLIPFLOW_LAMMPS_COMPLETED step={steps}":
            raise ValueError("prepared completion marker does not bind the step count")
        launcher = _launcher(manifest, target)
        generated = _generated(manifest)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": [_diagnostic("error", "lammps.prepared_manifest", str(exc))]}

    prepared_dir = manifest_path.parent
    selected_name = f"in.{target}.lammps"
    staged_inputs: list[tuple[str, Path]] = []
    for name in ("structure.data", selected_name):
        record = generated.get(name)
        path = prepared_dir / name
        if record is None or not _ordinary_project_file(path, root, MAX_INPUT_BYTES):
            diagnostics.append(_diagnostic("error", f"lammps.prepared_{name}", f"prepared file is missing or unsafe: {name}"))
            continue
        if path.stat().st_size != record["size_bytes"] or _sha256(path) != record["sha256"]:
            diagnostics.append(_diagnostic("error", f"lammps.prepared_{name}", f"prepared file fingerprint differs: {name}"))
            continue
        staged_inputs.append((name, path))
    if diagnostics:
        return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": diagnostics}
    deck = (prepared_dir / selected_name).read_text(encoding="utf-8")
    if manifest["completion_marker"] not in deck or "${MODEL_FILE}" not in deck or str(model["relative_path"]) in deck:
        return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": [_diagnostic("error", "lammps.deck_portability", "prepared deck does not satisfy the portable execution contract")]}

    diagnostics = _resources(context, target, str(framework))
    if diagnostics:
        return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": diagnostics}
    cluster = Path(__file__).resolve().with_name("lammps_cluster.py")
    if not _ordinary_file(cluster):
        return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": [_diagnostic("error", "lammps.cluster_runner", "bundled lammps_cluster.py is missing")]}
    staged_files = [
        _staged(project, "project.yaml"),
        _staged(manifest_path, "lammps-input-manifest.json"),
        *[_staged(path, f"lammps/{name}") for name, path in staged_inputs],
        _staged(cluster, "lammps_cluster.py"),
    ]
    fetch_outputs = [
        {"remote_name": "lammps-execution-result.json", "remote_path": "output/lammps-execution-result.json", "local_name": "lammps-execution-result.json", "required": True, "max_bytes": MAX_JSON_BYTES, "role": "lammps-execution-result"},
        {"remote_name": "trajectory.lammpstrj", "remote_path": "output/trajectory.lammpstrj", "local_name": "trajectory.lammpstrj", "required": True, "max_bytes": MAX_TRAJECTORY_BYTES, "role": "trajectory"},
        {"remote_name": "final.data", "remote_path": "output/final.data", "local_name": "final.data", "required": True, "max_bytes": MAX_FINAL_DATA_BYTES, "role": "final-structure"},
        {"remote_name": "final.restart", "remote_path": "output/final.restart", "local_name": "final.restart", "required": True, "max_bytes": MAX_RESTART_BYTES, "role": "lammps-restart"},
        {"remote_name": "lammps.log", "remote_path": "output/lammps.log", "local_name": "lammps.log", "required": True, "max_bytes": MAX_LOG_BYTES, "role": "lammps-log"},
        {"remote_name": "lammps.screen.log", "remote_path": "output/lammps.screen.log", "local_name": "lammps.screen.log", "required": True, "max_bytes": MAX_LOG_BYTES, "role": "lammps-log"},
        {"remote_name": "cluster-run-report.json", "remote_path": "output/cluster-run-report.json", "local_name": "cluster-run-report.json", "required": True, "max_bytes": MAX_JSON_BYTES, "role": "cluster-run-report"},
        {"remote_name": "lammps.stdout.log", "remote_path": "output/lammps.stdout.log", "local_name": "lammps.stdout.log", "required": False, "max_bytes": MAX_LOG_BYTES, "role": "lammps-log"},
        {"remote_name": "lammps.stderr.log", "remote_path": "output/lammps.stderr.log", "local_name": "lammps.stderr.log", "required": False, "max_bytes": MAX_LOG_BYTES, "role": "lammps-log"},
        {"remote_name": "runner.stdout.log", "remote_path": "logs/lammps-runner.stdout.log", "local_name": "runner.stdout.log", "required": False, "max_bytes": MAX_LOG_BYTES, "role": "runner-log"},
        {"remote_name": "runner.stderr.log", "remote_path": "logs/lammps-runner.stderr.log", "local_name": "runner.stderr.log", "required": False, "max_bytes": MAX_LOG_BYTES, "role": "runner-log"},
    ]
    identity = {
        "framework": framework,
        "target": target,
        "input_manifest_fingerprint": manifest_sha,
        "model_id": model["model_id"],
        "model_fingerprint": model["fingerprint"],
        "artifact_format": model["artifact_format"],
        "ensemble": md.get("ensemble"),
        "temperature_k": md.get("temperature_k"),
        "timestep_fs": float(timestep),
        "steps": int(steps),
        "type_map": md.get("type_map"),
        "thermo_interval": md.get("thermo_interval"),
        "dump_interval": md.get("dump_interval"),
        "seed": md.get("seed"),
        "completion_marker": manifest["completion_marker"],
        "prepared_launcher": launcher,
    }
    family = TEMPLATE_FAMILIES[(str(framework), target)]
    return {
        "plugin_id": PLUGIN_ID,
        "status": "READY",
        "executable": True,
        "argv": [f"template-family:{family}"],
        "cwd": "remote-attempt-workspace",
        "framework": framework,
        "operation": EXECUTE,
        "expected_outputs": [item["remote_name"] for item in fetch_outputs if item["required"]],
        "diagnostics": [],
        "lammps_execution_identity": identity,
        "input_fingerprints": {
            "project": _sha256(project),
            "prepared_manifest": manifest_sha,
            "structure_data": _sha256(prepared_dir / "structure.data"),
            "input_deck": _sha256(prepared_dir / selected_name),
            "cluster_runner": _sha256(cluster),
        },
        "approval_summary": {
            "expensive": True,
            "submits_jobs": True,
            "framework": framework,
            "target": target,
            "model_id": model["model_id"],
            "model_fingerprint": model["fingerprint"],
            "artifact_format": model["artifact_format"],
            "ensemble": md.get("ensemble"),
            "temperature_K": md.get("temperature_k"),
            "timestep_fs": float(timestep),
            "steps": int(steps),
            "simulated_time_ps": int(steps) * float(timestep) / 1000.0,
            "type_map": md.get("type_map"),
            "required_packages": launcher.get("required_packages", []),
            "resources": dict(_mapping(context.get("resources"))),
            "template_family": family,
            "model_path_site_owned": True,
        },
        "failure_salvage": {
            "schema_version": 1,
            "fetch_remote_names": [
                "cluster-run-report.json",
                "lammps.log",
                "lammps.screen.log",
                "lammps.stdout.log",
                "lammps.stderr.log",
                "runner.stdout.log",
                "runner.stderr.log",
            ],
        },
        "scheduled_execution": {
            "schema_version": 2,
            "template_family": family,
            "staged_files": staged_files,
            "fetch_outputs": fetch_outputs,
        },
    }


def _scheduled_plan(context: dict[str, Any]) -> dict[str, Any]:
    return _mapping(_mapping(context.get("execution")).get("plan"))


def _operation(context: Any) -> str:
    return str(_mapping(context.get("parameters") if isinstance(context, dict) else {}).get("operation", PREPARE))


def _artifact_records(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("execution result artifacts must be a list")
    records: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict) or not _plain(item.get("name")):
            raise ValueError("execution artifact record is invalid")
        name = str(item["name"])
        if name in records:
            raise ValueError(f"duplicate execution artifact: {name}")
        records[name] = item
    return records


def _check_artifact(path: Path, record: dict[str, Any], limit: int) -> str | None:
    if not _ordinary_file(path, limit):
        return f"artifact is missing, unsafe or oversized: {path.name}"
    if record.get("path") != path.name or record.get("sha256") != _sha256(path) or record.get("size_bytes") != path.stat().st_size:
        return f"artifact record differs for {path.name}"
    return None


def _log_has_marker(path: Path, marker: str) -> bool:
    if not _ordinary_file(path, MAX_LOG_BYTES):
        return False
    needle = marker.encode("utf-8")
    with path.open("rb") as stream:
        carry = b""
        while True:
            chunk = stream.read(1 << 20)
            if not chunk:
                return needle in carry
            data = carry + chunk
            if needle in data:
                return True
            carry = data[-max(len(needle) - 1, 0) :]


def _check_execute(context: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    diagnostics: list[dict[str, str]] = []
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    identity = _mapping(_scheduled_plan(context).get("lammps_execution_identity"))
    if not identity:
        return [_diagnostic("error", "lammps.plan_identity", "pinned plan lacks lammps_execution_identity")], None
    try:
        result = _read_json(attempt / "lammps-execution-result.json")
    except Exception as exc:
        return [_diagnostic("error", "lammps.result", f"lammps-execution-result.json is unreadable: {exc}")], None
    expected = {
        "schema_version": 1,
        "plugin_id": PLUGIN_ID,
        "status": "OK",
        "operation": EXECUTE,
        "framework": identity.get("framework"),
        "target": identity.get("target"),
        "input_manifest_fingerprint": identity.get("input_manifest_fingerprint"),
        "ensemble": identity.get("ensemble"),
        "steps_requested": identity.get("steps"),
        "steps_completed": identity.get("steps"),
        "completion_marker": identity.get("completion_marker"),
    }
    for key, value in expected.items():
        if result.get(key) != value:
            diagnostics.append(_diagnostic("error", f"lammps.result_{key}", f"execution result field {key} differs from approved plan"))
    model = _mapping(result.get("model"))
    if model != {
        "id": identity.get("model_id"),
        "fingerprint": identity.get("model_fingerprint"),
        "artifact_format": identity.get("artifact_format"),
    }:
        diagnostics.append(_diagnostic("error", "lammps.result_model", "execution result model identity differs"))
    if not _plain(result.get("lammps_version")):
        diagnostics.append(_diagnostic("error", "lammps.version", "execution result must record LAMMPS version"))
    try:
        artifacts = _artifact_records(result.get("artifacts"))
    except ValueError as exc:
        diagnostics.append(_diagnostic("error", "lammps.artifacts", str(exc)))
        artifacts = {}
    required = {
        "trajectory.lammpstrj": MAX_TRAJECTORY_BYTES,
        "final.data": MAX_FINAL_DATA_BYTES,
        "final.restart": MAX_RESTART_BYTES,
        "lammps.log": MAX_LOG_BYTES,
        "lammps.screen.log": MAX_LOG_BYTES,
    }
    optional = {"lammps.stdout.log": MAX_LOG_BYTES, "lammps.stderr.log": MAX_LOG_BYTES}
    if not set(required).issubset(artifacts) or not set(artifacts).issubset(set(required) | set(optional)):
        diagnostics.append(_diagnostic("error", "lammps.artifact_set", "execution result artifact set differs from contract"))
    for name, limit in {**required, **optional}.items():
        if name in artifacts:
            error = _check_artifact(attempt / name, artifacts[name], limit)
            if error:
                diagnostics.append(_diagnostic("error", f"lammps.artifact_{name}", error))
    marker = str(identity.get("completion_marker", ""))
    if not _log_has_marker(attempt / "lammps.log", marker):
        diagnostics.append(_diagnostic("error", "lammps.completion_marker", "fetched LAMMPS log lacks the approved completion marker"))
    try:
        report = _read_json(attempt / "cluster-run-report.json")
    except Exception as exc:
        diagnostics.append(_diagnostic("error", "lammps.cluster_report", f"cluster-run-report.json is unreadable: {exc}"))
        report = {}
    if (
        report.get("status") != "OK"
        or report.get("framework") != identity.get("framework")
        or report.get("target") != identity.get("target")
        or report.get("model_id") != identity.get("model_id")
        or report.get("model_fingerprint") != identity.get("model_fingerprint")
        or report.get("input_manifest_fingerprint") != identity.get("input_manifest_fingerprint")
        or report.get("steps_completed") != identity.get("steps")
        or report.get("lammps_version") != result.get("lammps_version")
    ):
        diagnostics.append(_diagnostic("error", "lammps.cluster_identity", "cluster report identity differs from the approved completed execution"))
    result_path = attempt / "lammps-execution-result.json"
    if _ordinary_file(result_path, MAX_JSON_BYTES) and report.get("result_sha256") != _sha256(result_path):
        diagnostics.append(_diagnostic("error", "lammps.cluster_result_hash", "cluster report result SHA differs"))
    return (diagnostics, None) if diagnostics else (diagnostics, result)


class Adapter:
    def validate(self, context: Any) -> list[dict[str, str]]:
        if _operation(context) == EXECUTE:
            return _validate_execute(context if isinstance(context, dict) else {})
        return legacy.Adapter().validate(context)

    def plan(self, context: Any) -> dict[str, Any]:
        if _operation(context) == EXECUTE:
            if not isinstance(context, dict):
                return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": [_diagnostic("error", "context.type", "context must be an object")]}
            return _execute_plan(context)
        if not isinstance(context, dict):
            return legacy.Adapter().plan(context)
        return _prepare_plan(context)

    def prepare(self, context: Any, plan: Any) -> dict[str, Any]:
        if _operation(context) == EXECUTE:
            if not isinstance(plan, dict) or plan.get("status") != "READY":
                return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": [_diagnostic("error", "plan.not_ready", "scheduled execute requires a READY plan")]}
            return dict(plan)
        return legacy.Adapter().prepare(context, plan)

    def check(self, context: Any) -> dict[str, Any]:
        if _operation(context) != EXECUTE:
            return legacy.Adapter().check(context)
        if not isinstance(context, dict):
            return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": [_diagnostic("error", "context.type", "context must be an object")]}
        diagnostics, result = _check_execute(context)
        if diagnostics or result is None:
            return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
        return {
            "plugin_id": PLUGIN_ID,
            "status": "OK",
            "diagnostics": [],
            "metrics": {
                "steps_completed": float(result["steps_completed"]),
                "simulated_time_ps": float(result["steps_completed"]) * float(_mapping(_scheduled_plan(context).get("lammps_execution_identity")).get("timestep_fs", 0.0)) / 1000.0,
            },
            "artifacts": [],
        }

    def collect(self, context: Any) -> dict[str, Any]:
        if _operation(context) != EXECUTE:
            return legacy.Adapter().collect(context)
        checked = self.check(context)
        if checked.get("status") != "OK":
            return checked
        assert isinstance(context, dict)
        attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
        specs = (
            ("lammps-execution-result.json", "lammps-execution-result", "application/json"),
            ("trajectory.lammpstrj", "trajectory", "text/plain"),
            ("final.data", "final-structure", "text/plain"),
            ("final.restart", "lammps-restart", "application/octet-stream"),
            ("lammps.log", "lammps-log", "text/plain"),
            ("lammps.screen.log", "lammps-log", "text/plain"),
            ("cluster-run-report.json", "cluster-run-report", "application/json"),
        )
        artifacts = [
            {"path": str(attempt / name), "role": role, "media_type": media}
            for name, role, media in specs
            if _ordinary_file(attempt / name)
        ]
        for name in ("lammps.stdout.log", "lammps.stderr.log", "runner.stdout.log", "runner.stderr.log"):
            if _ordinary_file(attempt / name):
                artifacts.append({"path": str(attempt / name), "role": "execution-log", "media_type": "text/plain"})
        return {"plugin_id": PLUGIN_ID, "status": "OK", "diagnostics": [], "metrics": checked.get("metrics", {}), "artifacts": artifacts}

    def replay(self, context: Any) -> dict[str, Any]:
        if _operation(context) != EXECUTE:
            return legacy.Adapter().replay(context)
        return self.collect(context)
