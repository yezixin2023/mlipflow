"""Scheduled LAMMPS execution from a reviewed input bundle and site-owned model.

Site templates own binaries, launchers and environment setup.
"""
from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

from . import prepare

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
M3GNET_TEMPLATE_FAMILIES = {
    ("matgl", "cpu"): "lammps-m3gnet-cpu",
    ("matgl", "gpu"): "lammps-m3gnet-gpu",
    ("gnnp", "cpu"): "lammps-m3gnet-gnnp-cpu",
    ("m3gnet", "cpu"): "lammps-m3gnet-legacy-cpu",
}
MODEL_CONTRACTS = {
    ("deepmd", None): ("file", "deepmd-lammps-model"),
    ("mace", None): ("file", "mace-lammps-torchscript"),
    ("m3gnet", "matgl"): ("file", "matgl-lammps-torchscript"),
    ("m3gnet", "gnnp"): ("directory", "matgl-model-directory"),
    ("m3gnet", "m3gnet"): ("directory", "matgl-model-directory"),
}


def _errors(items: list[dict[str, str]]) -> list[dict[str, str]]:
    return [item for item in items if item.get("level") == "error"]


def _plain(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and "\x00" not in value
        and "\n" not in value
        and "\r" not in value
    )


def _ordinary_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _safe_relative(value: Any) -> bool:
    if not _plain(value) or "\\" in str(value):
        return False
    path = PurePosixPath(str(value))
    return not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)


def _project_file(root: Path, value: Any) -> Path:
    if not _plain(value) or "\\" in str(value):
        raise ValueError("project file reference must be a path string")
    raw = Path(str(value)).expanduser()
    if raw.is_absolute():
        candidate = raw.absolute()
    elif _safe_relative(value):
        candidate = (root / raw).absolute()
    else:
        raise ValueError("project file reference must be a safe relative path")
    try:
        candidate.resolve().relative_to(root.resolve())
    except (OSError, ValueError) as exc:
        raise ValueError("project file escapes project root") from exc
    return candidate


def _ordinary_project_file(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return path.is_file() and path.stat().st_size > 0
    except (OSError, ValueError):
        return False


def _project_config(root: Path) -> Path:
    candidates = [root / name for name in ("project.yaml", "project.yml", "project.json")]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise ValueError("project root must contain exactly one project.yaml/project.yml/project.json")
    return found[0]


def _read_json(path: Path) -> dict[str, Any]:
    if not _ordinary_file(path):
        raise ValueError(f"missing or empty JSON file: {path.name}")
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
    model = prepare._mapping(manifest.get("model"))
    interface = model.get("lammps_interface") if model.get("framework") == "m3gnet" else None
    requires_interface_path = interface in {"gnnp", "m3gnet"}
    if bool(launcher.get("requires_interface_path", False)) != requires_interface_path:
        raise ValueError("prepared launcher interface-path contract differs from model interface")
    expected_interface_placeholders = 1 if requires_interface_path else 0
    if argv.count("<site-resolved-interface-path>") != expected_interface_placeholders:
        raise ValueError("prepared launcher has an invalid site interface-path placeholder count")
    if launcher.get("lammps_interface", interface or model.get("framework")) != (
        interface or model.get("framework")
    ):
        raise ValueError("prepared launcher interface differs from model interface")
    return launcher


def _staged(path: Path, remote_name: str) -> dict[str, Any]:
    return {
        "source": str(path.absolute()),
        "remote_name": remote_name,
    }


def _positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _nonnegative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _resources(context: dict[str, Any], target: str | None, framework: str | None = None) -> list[dict[str, str]]:
    resources = context.get("resources")
    if not isinstance(resources, dict):
        return [prepare._diagnostic("error", "lammps.resources", "resources must be an object")]
    diagnostics: list[dict[str, str]] = []
    if set(resources) != {"cpus", "gpus", "memory", "walltime"}:
        return [prepare._diagnostic("error", "lammps.resource_fields", "ssh-slurm resources must contain exactly cpus, gpus, memory, walltime")]
    if not _positive_int(resources.get("cpus")):
        diagnostics.append(prepare._diagnostic("error", "lammps.cpus", "resources.cpus must be positive"))
    if not _nonnegative_int(resources.get("gpus")):
        diagnostics.append(prepare._diagnostic("error", "lammps.gpus", "resources.gpus must be non-negative"))
    for key in ("memory", "walltime"):
        if not _plain(resources.get(key)):
            diagnostics.append(prepare._diagnostic("error", f"lammps.{key}", f"resources.{key} must be a non-empty string"))
    if target == "cpu" and resources.get("gpus") != 0:
        diagnostics.append(prepare._diagnostic("error", "lammps.cpu_gpus", "target=cpu requires resources.gpus=0"))
    if target == "gpu" and not _positive_int(resources.get("gpus")):
        diagnostics.append(prepare._diagnostic("error", "lammps.gpu_resource", "target=gpu requires resources.gpus>=1"))
    if target == "gpu" and framework in {"mace", "m3gnet"} and resources.get("gpus") != 1:
        diagnostics.append(prepare._diagnostic("error", "lammps.single_gpu", "MACE/MatGL GPU execution is pinned to one GPU in this contract"))
    return diagnostics


def validate(context: dict[str, Any]) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    if context.get("backend") != BACKEND:
        diagnostics.append(prepare._diagnostic("error", "lammps.backend", "operation=execute requires backend=ssh-slurm"))
    inputs = context.get("inputs")
    parameters = context.get("parameters")
    if not isinstance(inputs, dict):
        diagnostics.append(prepare._diagnostic("error", "lammps.inputs", "inputs must be an object"))
        inputs = {}
    if not isinstance(parameters, dict):
        diagnostics.append(prepare._diagnostic("error", "lammps.parameters", "parameters must be an object"))
        parameters = {}
    if set(inputs) != {"lammps_input_manifest"}:
        diagnostics.append(prepare._diagnostic("error", "lammps.execute_inputs", "execute requires exactly inputs.lammps_input_manifest"))
    if set(parameters) != {"operation", "target"}:
        diagnostics.append(prepare._diagnostic("error", "lammps.execute_parameters", "execute parameters must be exactly operation and target"))
    if parameters.get("operation") != EXECUTE:
        diagnostics.append(prepare._diagnostic("error", "lammps.operation", "operation must be execute"))
    target = parameters.get("target")
    if target not in TARGETS:
        diagnostics.append(prepare._diagnostic("error", "lammps.target", "target must be cpu or gpu"))
    diagnostics.extend(_resources(context, str(target) if target in TARGETS else None))
    return diagnostics


def plan(context: dict[str, Any]) -> dict[str, Any]:
    diagnostics = validate(context)
    if diagnostics:
        return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": diagnostics}
    root = Path(str(context["project_root"])).expanduser().absolute()
    inputs = prepare._mapping(context["inputs"])
    parameters = prepare._mapping(context["parameters"])
    try:
        project = _project_config(root)
        manifest_path = _project_file(root, inputs["lammps_input_manifest"])
    except ValueError as exc:
        return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": [prepare._diagnostic("error", "lammps.project_inputs", str(exc))]}
    if not _ordinary_project_file(project, root) or not _ordinary_project_file(manifest_path, root):
        return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": [prepare._diagnostic("error", "lammps.project_files", "project and prepared manifest must be project files")]}
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
        model = prepare._mapping(manifest.get("model"))
        md = prepare._mapping(manifest.get("md"))
        framework = model.get("framework")
        target = str(parameters["target"])
        if framework not in FRAMEWORKS:
            raise ValueError("prepared model framework is unsupported")
        if target not in md.get("targets", []):
            raise ValueError("selected target was not prepared")
        interface = model.get("lammps_interface") if framework == "m3gnet" else None
        if framework == "m3gnet" and interface is None:
            interface = "matgl"
        expected_model_contract = MODEL_CONTRACTS.get((str(framework), interface))
        if expected_model_contract is None:
            raise ValueError("prepared model LAMMPS interface is unsupported")
        if (
            (model.get("kind"), model.get("artifact_format")) != expected_model_contract
            or not _plain(model.get("model_id"))
            or not _safe_relative(model.get("relative_path"))
        ):
            raise ValueError("prepared model record is incomplete")
        if framework == "m3gnet" and (interface, target) not in M3GNET_TEMPLATE_FAMILIES:
            raise ValueError("prepared M3GNet LAMMPS interface does not support the selected target")
        steps = md.get("steps")
        timestep = md.get("timestep_fs")
        if not _positive_int(steps) or not isinstance(timestep, (int, float)) or isinstance(timestep, bool) or not math.isfinite(float(timestep)) or float(timestep) <= 0:
            raise ValueError("prepared MD step/timestep values are invalid")
        marker = manifest.get("completion_marker")
        if marker != f"MLIPFLOW_LAMMPS_COMPLETED step={steps}":
            raise ValueError("prepared completion marker does not bind the step count")
        launcher = _launcher(manifest, target)
        generated = _generated(manifest)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": [prepare._diagnostic("error", "lammps.prepared_manifest", str(exc))]}

    prepared_dir = manifest_path.parent
    selected_name = f"in.{target}.lammps"
    staged_inputs: list[tuple[str, Path]] = []
    for name in ("structure.data", selected_name):
        record = generated.get(name)
        path = prepared_dir / name
        if record is None or not _ordinary_project_file(path, root):
            diagnostics.append(prepare._diagnostic("error", f"lammps.prepared_{name}", f"prepared file is missing: {name}"))
            continue
        staged_inputs.append((name, path))
    if diagnostics:
        return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": diagnostics}
    deck = (prepared_dir / selected_name).read_text(encoding="utf-8")
    requires_interface_path = interface in {"gnnp", "m3gnet"}
    if (
        manifest["completion_marker"] not in deck
        or "${MODEL_FILE}" not in deck
        or str(model["relative_path"]) in deck
        or ("${INTERFACE_PATH}" in deck) != requires_interface_path
    ):
        return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": [prepare._diagnostic("error", "lammps.deck_portability", "prepared deck does not satisfy the portable execution contract")]}

    diagnostics = _resources(context, target, str(framework))
    if diagnostics:
        return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": diagnostics}
    cluster = Path(__file__).resolve().with_name("lammps_cluster.py")
    if not _ordinary_file(cluster):
        return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": [prepare._diagnostic("error", "lammps.cluster_runner", "bundled lammps_cluster.py is missing")]}
    staged_files = [
        _staged(project, "project.yaml"),
        _staged(manifest_path, "lammps-input-manifest.json"),
        *[_staged(path, f"lammps/{name}") for name, path in staged_inputs],
        _staged(cluster, "lammps_cluster.py"),
    ]
    fetch_outputs = [
        {"remote_name": "lammps-execution-result.json", "remote_path": "output/lammps-execution-result.json", "local_name": "lammps-execution-result.json", "required": True, "role": "lammps-execution-result"},
        {"remote_name": "trajectory.lammpstrj", "remote_path": "output/trajectory.lammpstrj", "local_name": "trajectory.lammpstrj", "required": True, "role": "trajectory"},
        {"remote_name": "final.data", "remote_path": "output/final.data", "local_name": "final.data", "required": True, "role": "final-structure"},
        {"remote_name": "final.restart", "remote_path": "output/final.restart", "local_name": "final.restart", "required": True, "role": "lammps-restart"},
        {"remote_name": "lammps.log", "remote_path": "output/lammps.log", "local_name": "lammps.log", "required": True, "role": "lammps-log"},
        {"remote_name": "lammps.screen.log", "remote_path": "output/lammps.screen.log", "local_name": "lammps.screen.log", "required": True, "role": "lammps-log"},
        {"remote_name": "cluster-run-report.json", "remote_path": "output/cluster-run-report.json", "local_name": "cluster-run-report.json", "required": True, "role": "cluster-run-report"},
        {"remote_name": "lammps.stdout.log", "remote_path": "output/lammps.stdout.log", "local_name": "lammps.stdout.log", "required": False, "role": "lammps-log"},
        {"remote_name": "lammps.stderr.log", "remote_path": "output/lammps.stderr.log", "local_name": "lammps.stderr.log", "required": False, "role": "lammps-log"},
        {"remote_name": "runner.stdout.log", "remote_path": "logs/lammps-runner.stdout.log", "local_name": "runner.stdout.log", "required": False, "role": "runner-log"},
        {"remote_name": "runner.stderr.log", "remote_path": "logs/lammps-runner.stderr.log", "local_name": "runner.stderr.log", "required": False, "role": "runner-log"},
    ]
    calculation = {
        "framework": framework,
        "target": target,
        "input_manifest_path": "lammps-input-manifest.json",
        "model_id": model["model_id"],
        "model_path": model["relative_path"],
        "model_kind": model["kind"],
        "artifact_format": model["artifact_format"],
        "lammps_interface": interface,
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
    family = (
        M3GNET_TEMPLATE_FAMILIES[(str(interface), target)]
        if framework == "m3gnet"
        else TEMPLATE_FAMILIES[(str(framework), target)]
    )
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
        "lammps_calculation": calculation,
        "input_paths": {
            "project": str(project),
            "prepared_manifest": str(manifest_path),
            "structure_data": str(prepared_dir / "structure.data"),
            "input_deck": str(prepared_dir / selected_name),
            "cluster_runner": str(cluster),
        },
        "approval_summary": {
            "expensive": True,
            "submits_jobs": True,
            "framework": framework,
            "target": target,
            "model_id": model["model_id"],
            "model_path": model["relative_path"],
            "model_kind": model["kind"],
            "artifact_format": model["artifact_format"],
            "lammps_interface": interface,
            "ensemble": md.get("ensemble"),
            "temperature_K": md.get("temperature_k"),
            "timestep_fs": float(timestep),
            "steps": int(steps),
            "simulated_time_ps": int(steps) * float(timestep) / 1000.0,
            "type_map": md.get("type_map"),
            "required_packages": launcher.get("required_packages", []),
            "resources": dict(prepare._mapping(context.get("resources"))),
            "template_family": family,
            "execution_model": "mpi",
            "cpus_meaning": "mpi-task-count",
            "model_path_site_owned": True,
        },
        "failure_salvage": {
            "schema_version": 1,
            "fetch_remote_names": [
                "trajectory.lammpstrj",
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
            "schema_version": 3,
            "execution_model": "mpi",
            "template_family": family,
            "staged_files": staged_files,
            "fetch_outputs": fetch_outputs,
        },
    }


def _scheduled_plan(context: dict[str, Any]) -> dict[str, Any]:
    return prepare._mapping(prepare._mapping(context.get("execution")).get("plan"))


def _operation(context: Any) -> str:
    return str(prepare._mapping(context.get("parameters") if isinstance(context, dict) else {}).get("operation", PREPARE))


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


def _check_artifact(path: Path, record: dict[str, Any]) -> str | None:
    if not _ordinary_file(path):
        return f"artifact is missing or empty: {path.name}"
    if record.get("path") != path.name:
        return f"artifact record differs for {path.name}"
    return None


def _log_has_marker(path: Path, marker: str) -> bool:
    if not _ordinary_file(path):
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
    calculation = prepare._mapping(_scheduled_plan(context).get("lammps_calculation"))
    if not calculation:
        return [prepare._diagnostic("error", "lammps.plan", "plan lacks lammps_calculation")], None
    try:
        result = _read_json(attempt / "lammps-execution-result.json")
    except Exception as exc:
        return [prepare._diagnostic("error", "lammps.result", f"lammps-execution-result.json is unreadable: {exc}")], None
    expected = {
        "schema_version": 1,
        "plugin_id": PLUGIN_ID,
        "status": "OK",
        "operation": EXECUTE,
        "framework": calculation.get("framework"),
        "target": calculation.get("target"),
        "input_manifest_path": calculation.get("input_manifest_path"),
        "ensemble": calculation.get("ensemble"),
        "steps_requested": calculation.get("steps"),
        "steps_completed": calculation.get("steps"),
        "completion_marker": calculation.get("completion_marker"),
    }
    for key, value in expected.items():
        if result.get(key) != value:
            diagnostics.append(prepare._diagnostic("error", f"lammps.result_{key}", f"execution result field {key} differs from approved plan"))
    model = prepare._mapping(result.get("model"))
    if model != {
        "id": calculation.get("model_id"),
        "path": calculation.get("model_path"),
        "kind": calculation.get("model_kind"),
        "artifact_format": calculation.get("artifact_format"),
        "lammps_interface": calculation.get("lammps_interface"),
    }:
        diagnostics.append(prepare._diagnostic("error", "lammps.result_model", "execution result model record differs"))
    if not _plain(result.get("lammps_version")):
        diagnostics.append(prepare._diagnostic("error", "lammps.version", "execution result must record LAMMPS version"))
    try:
        artifacts = _artifact_records(result.get("artifacts"))
    except ValueError as exc:
        diagnostics.append(prepare._diagnostic("error", "lammps.artifacts", str(exc)))
        artifacts = {}
    required = {
        "trajectory.lammpstrj",
        "final.data",
        "final.restart",
        "lammps.log",
        "lammps.screen.log",
    }
    optional = {"lammps.stdout.log", "lammps.stderr.log"}
    if not set(required).issubset(artifacts) or not set(artifacts).issubset(set(required) | set(optional)):
        diagnostics.append(prepare._diagnostic("error", "lammps.artifact_set", "execution result artifact set differs from contract"))
    for name in required | optional:
        if name in artifacts:
            error = _check_artifact(attempt / name, artifacts[name])
            if error:
                diagnostics.append(prepare._diagnostic("error", f"lammps.artifact_{name}", error))
    marker = str(calculation.get("completion_marker", ""))
    if not _log_has_marker(attempt / "lammps.log", marker):
        diagnostics.append(prepare._diagnostic("error", "lammps.completion_marker", "fetched LAMMPS log lacks the approved completion marker"))
    try:
        report = _read_json(attempt / "cluster-run-report.json")
    except Exception as exc:
        diagnostics.append(prepare._diagnostic("error", "lammps.cluster_report", f"cluster-run-report.json is unreadable: {exc}"))
        report = {}
    if (
        report.get("status") != "OK"
        or report.get("framework") != calculation.get("framework")
        or report.get("target") != calculation.get("target")
        or report.get("model_id") != calculation.get("model_id")
        or report.get("model_path") != calculation.get("model_path")
        or report.get("model_kind") != calculation.get("model_kind")
        or report.get("lammps_interface") != calculation.get("lammps_interface")
        or report.get("input_manifest_path") != calculation.get("input_manifest_path")
        or report.get("steps_completed") != calculation.get("steps")
        or report.get("lammps_version") != result.get("lammps_version")
    ):
        diagnostics.append(prepare._diagnostic("error", "lammps.cluster_report", "cluster report differs from the completed calculation"))
    return (diagnostics, None) if diagnostics else (diagnostics, result)


def check(context: Any) -> dict[str, Any]:
    if not isinstance(context, dict):
        return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": [prepare._diagnostic("error", "context.type", "context must be an object")]}
    diagnostics, result = _check_execute(context)
    if diagnostics or result is None:
        return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
    return {
        "plugin_id": PLUGIN_ID,
        "status": "OK",
        "diagnostics": [],
        "metrics": {
            "steps_completed": float(result["steps_completed"]),
            "simulated_time_ps": float(result["steps_completed"]) * float(prepare._mapping(_scheduled_plan(context).get("lammps_calculation")).get("timestep_fs", 0.0)) / 1000.0,
        },
        "artifacts": [],
    }


def collect(context: Any, *, checked: dict[str, Any] | None = None) -> dict[str, Any]:
    if checked is None:
        checked = check(context)
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
