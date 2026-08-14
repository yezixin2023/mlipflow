"""Cluster-aware facade for pes-sampling.

Local DIRECT/LASP behavior is delegated unchanged to ``adapter.py``.  LASP SSW
execution on ``ssh-slurm`` emits the generic scheduled_execution v2 contract so
MLIPFlow core owns staging, submission, polling, bounded fetch, and finalization.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import tarfile
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
LEGACY_PATH = HERE / "adapter.py"
WRAPPER_PATH = HERE / "lasp_ssw.py"
REMOTE_RUNNER_PATH = HERE / "lasp_cluster.py"
UNKNOWN = "HISTORICAL_PARAMETER_UNKNOWN"
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_ARC_BYTES = 512 * 1024 * 1024
MAX_FRAME_BYTES = 8 * 1024 * 1024
MAX_LOG_BYTES = 64 * 1024 * 1024
MAX_FRAMES = 10000
HPC_RESOURCES = {"cpus", "gpus", "memory", "walltime"}


def _load_legacy():
    spec = importlib.util.spec_from_file_location("mlipflow_pes_sampling_legacy", LEGACY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load legacy pes-sampling adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEGACY = _load_legacy()


def _diag(level: str, code: str, message: str) -> dict[str, str]:
    return {"level": level, "code": code, "message": message}


def _errors(items: list[dict[str, str]]) -> bool:
    return any(item.get("level") == "ERROR" for item in items)


def _blocked(items: list[dict[str, str]]) -> dict[str, Any]:
    return {"plugin_id": "pes-sampling", "status": "BLOCKED", "executable": False, "diagnostics": items}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _ordinary(path: Path | None, maximum: int | None = None) -> bool:
    if path is None or path.is_symlink() or not path.is_file():
        return False
    size = path.stat().st_size
    return size > 0 and (maximum is None or size <= maximum)


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _project_file(root: Path) -> Path | None:
    for name in ("project.yaml", "project.yml", "project.json"):
        candidate = root / name
        if _ordinary(candidate):
            return candidate
    return None


def _resolve_input(value: Any, root: Path) -> Path | None:
    if not isinstance(value, (str, Path)) or not str(value):
        return None
    path = Path(str(value)).expanduser()
    path = path if path.is_absolute() else root / path
    if path.is_symlink() or not path.is_file():
        return None
    return path.resolve()


def _stage(path: Path, remote_name: str) -> dict[str, Any]:
    return {"source": str(path), "remote_name": remote_name, "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _operation(context: Mapping[str, Any]) -> str:
    parameters = context.get("parameters", {})
    return str(parameters.get("operation", "direct-select")) if isinstance(parameters, Mapping) else "direct-select"


def _scheduled(context: Any) -> bool:
    return isinstance(context, Mapping) and context.get("backend") == "ssh-slurm" and _operation(context) == "lasp-ssw-execute"


def _validate_scheduled(context: Any) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    if not isinstance(context, Mapping):
        return [_diag("ERROR", "context.mapping_required", "context must be a mapping")]
    if context.get("backend") != "ssh-slurm":
        diagnostics.append(_diag("ERROR", "backend.ssh_slurm_required", "scheduled LASP requires backend ssh-slurm"))
    if _operation(context) != "lasp-ssw-execute":
        diagnostics.append(_diag("ERROR", "operation.scheduled_lasp", "ssh-slurm LASP supports lasp-ssw-execute only"))
    root = Path(str(context.get("project_root", ""))).expanduser().absolute()
    if not root.is_dir():
        diagnostics.append(_diag("ERROR", "path.project_root", "project_root must be an existing directory"))
        return diagnostics
    project_path = _project_file(root)
    if project_path is None:
        diagnostics.append(_diag("ERROR", "path.project_file", "scheduled LASP requires project.yaml/project.yml/project.json"))

    resources = context.get("resources", {})
    if not isinstance(resources, Mapping) or set(resources) != HPC_RESOURCES:
        diagnostics.append(_diag("ERROR", "resource.hpc_contract", "ssh-slurm LASP resources must be cpus, gpus, memory, walltime"))

    inputs = context.get("inputs", {})
    if not isinstance(inputs, Mapping):
        diagnostics.append(_diag("ERROR", "inputs.mapping_required", "inputs must be a mapping"))
        return diagnostics
    allowed_inputs = {"input_structure", "lasp_input", "lasp_auxiliary_files", "result_manifest"}
    unknown_inputs = sorted(set(inputs) - allowed_inputs)
    if unknown_inputs:
        diagnostics.append(_diag("ERROR", "input.unknown", "scheduled LASP does not accept: " + ", ".join(unknown_inputs)))
    structure = _resolve_input(inputs.get("input_structure"), root)
    lasp_input = _resolve_input(inputs.get("lasp_input"), root)
    if structure is None or not _within(structure, root):
        diagnostics.append(_diag("ERROR", "path.input_structure", "input_structure must be an ordinary project file"))
    else:
        try:
            if len(LEGACY._read_arc_frames(structure, 1)) != 1:
                raise ValueError("not exactly one frame")
        except Exception as exc:
            diagnostics.append(_diag("ERROR", "input.structure_arc", f"input_structure must contain one LASP ARC frame: {exc}"))
    if lasp_input is None or not _within(lasp_input, root):
        diagnostics.append(_diag("ERROR", "path.lasp_input", "lasp_input must be an ordinary project file"))
    else:
        try:
            LEGACY._validate_ssw_input(LEGACY._parse_lasp_input(lasp_input))
        except Exception as exc:
            diagnostics.append(_diag("ERROR", "input.lasp_ssw_contract", f"invalid LASP SSW input: {exc}"))

    auxiliary = inputs.get("lasp_auxiliary_files", {})
    if auxiliary is None:
        auxiliary = {}
    if not isinstance(auxiliary, Mapping):
        diagnostics.append(_diag("ERROR", "input.lasp_auxiliary_files", "lasp_auxiliary_files must be a mapping"))
    else:
        reserved = {"input.arc", "lasp.in", "project.yaml", "lasp_ssw.py", "lasp_cluster.py"}
        for name, source in auxiliary.items():
            if not isinstance(name, str) or not LEGACY.SAFE_NAME.fullmatch(name) or name in reserved:
                diagnostics.append(_diag("ERROR", "input.auxiliary_name", f"unsafe auxiliary destination: {name!r}"))
                break
            path = _resolve_input(source, root)
            if path is None or not _within(path, root):
                diagnostics.append(_diag("ERROR", "path.auxiliary_file", f"auxiliary {name} must be an ordinary project file"))
                break

    parameters = context.get("parameters", {})
    if not isinstance(parameters, Mapping):
        diagnostics.append(_diag("ERROR", "parameters.mapping_required", "parameters must be a mapping"))
        return diagnostics
    allowed_parameters = {
        "operation", "output_subdir", "historical_source_id", "selection_stride", "energy_max_ev",
        "max_frames", "include_best_arc", "include_md_arc", "seed_status",
        "acknowledge_uncontrolled_seed", "preserve_historical_order", "lasp_version", "mpi_processes",
    }
    unknown_parameters = sorted(set(parameters) - allowed_parameters)
    if unknown_parameters:
        diagnostics.append(_diag("ERROR", "parameter.unknown", "unsupported LASP parameter(s): " + ", ".join(unknown_parameters)))
    if parameters.get("output_subdir", "lasp-ssw") != "lasp-ssw":
        diagnostics.append(_diag("ERROR", "parameter.output_subdir", "scheduled LASP currently requires output_subdir=lasp-ssw"))
    if not LEGACY._portable_source_id(parameters.get("historical_source_id")):
        diagnostics.append(_diag("ERROR", "parameter.historical_source_id", "historical_source_id must be portable"))
    if not LEGACY._positive_int(parameters.get("selection_stride")):
        diagnostics.append(_diag("ERROR", "parameter.selection_stride", "selection_stride must be positive"))
    max_frames = parameters.get("max_frames")
    if not LEGACY._positive_int(max_frames) or int(max_frames) > MAX_FRAMES:
        diagnostics.append(_diag("ERROR", "parameter.max_frames", f"max_frames must be 1..{MAX_FRAMES}"))
    energy = parameters.get("energy_max_ev")
    if energy is not None and (isinstance(energy, bool) or not isinstance(energy, (int, float)) or not math.isfinite(float(energy))):
        diagnostics.append(_diag("ERROR", "parameter.energy_max_ev", "energy_max_ev must be finite or null"))
    for name in ("include_best_arc", "include_md_arc"):
        if not isinstance(parameters.get(name, False), bool):
            diagnostics.append(_diag("ERROR", f"parameter.{name}", f"{name} must be boolean"))
    if parameters.get("seed_status") != UNKNOWN:
        diagnostics.append(_diag("ERROR", "seed.historical_unknown", f"seed_status must be {UNKNOWN}"))
    if parameters.get("acknowledge_uncontrolled_seed") is not True:
        diagnostics.append(_diag("ERROR", "seed.acknowledgement_required", "acknowledge_uncontrolled_seed must be true"))
    if parameters.get("preserve_historical_order") is not True:
        diagnostics.append(_diag("ERROR", "parameter.preserve_historical_order", "preserve_historical_order must be true"))
    if not LEGACY._plain_string(parameters.get("lasp_version")):
        diagnostics.append(_diag("ERROR", "parameter.lasp_version", "lasp_version must be explicit"))
    if parameters.get("mpi_processes") is not None:
        diagnostics.append(_diag("ERROR", "parameter.mpi_site_owned", "scheduled LASP process count comes from resources.cpus/site template, not mpi_processes"))
    if not _ordinary(WRAPPER_PATH) or not _ordinary(REMOTE_RUNNER_PATH):
        diagnostics.append(_diag("ERROR", "path.bundled_runner", "bundled LASP cluster helpers are missing"))
    return diagnostics


def _fetch(remote_name: str, remote_path: str, local_name: str, required: bool, maximum: int, role: str) -> dict[str, Any]:
    return {"remote_name": remote_name, "remote_path": remote_path, "local_name": local_name, "required": required, "max_bytes": maximum, "role": role}


def _plan_scheduled(context: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = _validate_scheduled(context)
    if _errors(diagnostics):
        return _blocked(diagnostics)
    root = Path(str(context["project_root"])).expanduser().absolute().resolve()
    inputs = dict(context.get("inputs", {}))
    parameters = dict(context.get("parameters", {}))
    project_path = _project_file(root)
    structure = _resolve_input(inputs["input_structure"], root)
    lasp_input = _resolve_input(inputs["lasp_input"], root)
    assert project_path is not None and structure is not None and lasp_input is not None
    staged = [
        _stage(project_path, "project.yaml"),
        _stage(structure, "input.arc"),
        _stage(lasp_input, "lasp.in"),
        _stage(WRAPPER_PATH, "lasp_ssw.py"),
        _stage(REMOTE_RUNNER_PATH, "lasp_cluster.py"),
    ]
    auxiliary = inputs.get("lasp_auxiliary_files", {}) or {}
    for name in sorted(auxiliary):
        path = _resolve_input(auxiliary[name], root)
        assert path is not None
        staged.append(_stage(path, str(name)))

    include_best = bool(parameters.get("include_best_arc", False))
    include_md = bool(parameters.get("include_md_arc", False))
    fetch_outputs = [
        _fetch("cluster-run-report.json", "output/cluster-run-report.json", "lasp-ssw/cluster-run-report.json", True, MAX_JSON_BYTES, "lasp-cluster-report"),
        _fetch("sampling-result.json", "output/lasp-ssw/sampling-result.json", "lasp-ssw/sampling-result.json", True, MAX_JSON_BYTES, "sample-manifest"),
        _fetch("ssw-structures.json", "output/lasp-ssw/ssw-structures.json", "lasp-ssw/ssw-structures.json", True, MAX_JSON_BYTES, "ssw-structure-manifest"),
        _fetch("selected-structures.json", "output/lasp-ssw/selected-structures.json", "lasp-ssw/selected-structures.json", True, MAX_JSON_BYTES, "selected-structure-manifest"),
        _fetch("lasp-run-metadata.json", "output/lasp-ssw/lasp-run-metadata.json", "lasp-ssw/lasp-run-metadata.json", True, MAX_JSON_BYTES, "lasp-run-metadata"),
        _fetch("selected-structures.tar.gz", "output/selected-structures.tar.gz", "lasp-ssw/selected-structures.tar.gz", True, MAX_ARC_BYTES, "selected-structures-archive"),
        _fetch("allstr.arc", "output/lasp-ssw/raw-run/allstr.arc", "lasp-ssw/raw-run/allstr.arc", True, MAX_ARC_BYTES, "ssw-archive"),
        _fetch("best.arc", "output/lasp-ssw/raw-run/best.arc", "lasp-ssw/raw-run/best.arc", include_best, MAX_ARC_BYTES, "best-archive"),
        _fetch("md.arc", "output/lasp-ssw/raw-run/md.arc", "lasp-ssw/raw-run/md.arc", include_md, MAX_ARC_BYTES, "md-archive"),
        _fetch("aimd-seeds.json", "output/lasp-ssw/aimd-seeds.json", "lasp-ssw/aimd-seeds.json", include_best, MAX_JSON_BYTES, "aimd-seed-manifest"),
        _fetch("md-structures.json", "output/lasp-ssw/md-structures.json", "lasp-ssw/md-structures.json", include_md, MAX_JSON_BYTES, "md-structure-manifest"),
        _fetch("lasp.stdout.log", "output/lasp-ssw/raw-run/lasp.stdout.log", "lasp-ssw/lasp.stdout.log", False, MAX_LOG_BYTES, "lasp-stdout"),
        _fetch("lasp.stderr.log", "output/lasp-ssw/raw-run/lasp.stderr.log", "lasp-ssw/lasp.stderr.log", False, MAX_LOG_BYTES, "lasp-stderr"),
        _fetch("lasp.out", "output/lasp-ssw/raw-run/lasp.out", "lasp-ssw/lasp.out", False, MAX_LOG_BYTES, "lasp-native-log"),
    ]
    identity = {
        "historical_source_id": parameters["historical_source_id"],
        "selection_stride": parameters["selection_stride"],
        "energy_max_ev": parameters.get("energy_max_ev"),
        "max_frames": parameters["max_frames"],
        "include_best_arc": include_best,
        "include_md_arc": include_md,
        "seed_status": UNKNOWN,
        "preserve_historical_order": True,
        "lasp_version": parameters["lasp_version"],
        "input_structure_sha256": _sha256(structure),
        "lasp_input_sha256": _sha256(lasp_input),
    }
    return {
        "plugin_id": "pes-sampling",
        "operation": "lasp-ssw-execute",
        "status": "READY",
        "executable": True,
        "argv": ["template-family:lasp-ssw"],
        "cwd": "remote-attempt-workspace",
        "shell": False,
        "expected_outputs": [item["remote_name"] for item in fetch_outputs if item["required"]],
        "diagnostics": diagnostics,
        "lasp_scheduled_identity": identity,
        "approval_summary": {
            "expensive": True,
            "submits_jobs": True,
            "framework": "LASP",
            "sampling_method": "stochastic-surface-walking",
            "max_frames": parameters["max_frames"],
            "selection_stride": parameters["selection_stride"],
            "fetch_allowlist": sorted(item["remote_name"] for item in fetch_outputs),
            "staged_file_count": len(staged),
        },
        "input_fingerprints": {item["remote_name"]: item["sha256"] for item in staged},
        "scheduled_execution": {
            "schema_version": 2,
            "template_family": "lasp-ssw",
            "staged_files": staged,
            "fetch_outputs": fetch_outputs,
        },
        "assumptions": {
            "scheduler_submission": True,
            "cluster_lasp_executable_is_site_owned": True,
            "cluster_mpi_launcher_is_site_owned": True,
            "seed_status": UNKNOWN,
            "fresh_remote_workspace_required": True,
        },
    }


def _json(path: Path) -> dict[str, Any]:
    if not _ordinary(path, MAX_JSON_BYTES):
        raise ValueError(f"missing or oversized JSON artifact: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _scheduled_check(context: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics: list[dict[str, str]] = []
    execution = context.get("execution", {})
    plan = execution.get("plan", {}) if isinstance(execution, Mapping) else {}
    identity = plan.get("lasp_scheduled_identity", {}) if isinstance(plan, Mapping) else {}
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    root = attempt / "lasp-ssw"
    try:
        report = _json(root / "cluster-run-report.json")
        result = _json(root / "sampling-result.json")
        structures = _json(root / "ssw-structures.json")
        selected = _json(root / "selected-structures.json")
        metadata = _json(root / "lasp-run-metadata.json")
        allstr = root / "raw-run" / "allstr.arc"
        archive = root / "selected-structures.tar.gz"
        if not _ordinary(allstr, MAX_ARC_BYTES) or not _ordinary(archive, MAX_ARC_BYTES):
            raise ValueError("required LASP archive output is missing")
        if report.get("schema_version") != 1 or report.get("plugin_id") != "pes-sampling" or report.get("status") != "OK":
            raise ValueError("cluster-run-report identity/status is invalid")
        if report.get("operation") != "lasp-ssw-execute" or result.get("operation") != "lasp-ssw-execute" or result.get("status") != "OK":
            raise ValueError("remote result operation/status mismatch")
        if report.get("lasp_version") != identity.get("lasp_version"):
            raise ValueError("LASP version differs from approved plan")
        policy = report.get("selection_policy")
        expected_policy = {
            "selection_stride": identity.get("selection_stride"),
            "energy_max_ev": identity.get("energy_max_ev"),
            "max_frames": identity.get("max_frames"),
            "preserve_historical_order": True,
        }
        if policy != expected_policy:
            raise ValueError("cluster selection policy differs from approved plan")
        if report.get("source_outputs", {}).get("allstr_arc_sha256") != _sha256(allstr):
            raise ValueError("allstr.arc hash differs from cluster report")
        archive_record = report.get("selected_archive", {})
        if archive_record.get("sha256") != _sha256(archive) or archive_record.get("size_bytes") != archive.stat().st_size:
            raise ValueError("selected archive identity differs from cluster report")

        frames = LEGACY._read_arc_frames(allstr, int(identity["max_frames"]))
        records = structures.get("structures")
        selected_records = selected.get("structures")
        if not isinstance(records, list) or not isinstance(selected_records, list) or len(records) != len(frames):
            raise ValueError("structure manifests do not match allstr.arc frame count")
        source_id = str(identity["historical_source_id"])
        accepted_order = 0
        expected_selected: list[tuple[dict[str, Any], dict[str, Any], int]] = []
        threshold = identity.get("energy_max_ev")
        stride = int(identity["selection_stride"])
        for frame, record in zip(frames, records):
            if not isinstance(record, dict):
                raise ValueError("invalid SSW structure record")
            index = int(frame["frame_index"])
            expected_id = LEGACY._lasp_structure_id(source_id, "ssw", index, str(frame["frame_sha256"]))
            if record.get("frame_index") != index or record.get("historical_order") != index or record.get("frame_sha256") != frame["frame_sha256"] or record.get("structure_id") != expected_id:
                raise ValueError(f"SSW frame identity mismatch at frame {index}")
            if float(record.get("energy_ev")) != float(frame["energy_ev"]):
                raise ValueError(f"SSW energy mismatch at frame {index}")
            accepted = threshold is None or float(frame["energy_ev"]) <= float(threshold)
            if accepted:
                accepted_order += 1
            chosen = accepted and (accepted_order - 1) % stride == 0
            if bool(record.get("energy_filter_pass")) != accepted or bool(record.get("selected")) != chosen:
                raise ValueError(f"SSW selection flags are inconsistent at frame {index}")
            if chosen:
                expected_selected.append((frame, record, len(expected_selected) + 1))
        if len(expected_selected) != len(selected_records):
            raise ValueError("selected structure count does not match approved policy")
        expected_names: list[str] = []
        for (frame, record, order), selected_record in zip(expected_selected, selected_records):
            if not isinstance(selected_record, dict) or selected_record.get("structure_id") != record.get("structure_id") or selected_record.get("frame_sha256") != frame["frame_sha256"] or selected_record.get("selected_order") != order:
                raise ValueError("selected manifest identity/order mismatch")
            expected_names.append(f"selected/input-{order:06d}.arc")
        with tarfile.open(archive, "r:gz") as bundle:
            members = bundle.getmembers()
            names = [item.name for item in members]
            if names != expected_names:
                raise ValueError("selected archive members differ from selected manifest")
            for member, (frame, _, _) in zip(members, expected_selected):
                if not member.isfile() or member.issym() or member.islnk() or member.size > MAX_FRAME_BYTES:
                    raise ValueError("selected archive contains an unsafe member")
                stream = bundle.extractfile(member)
                if stream is None:
                    raise ValueError("selected archive member cannot be read")
                payload = stream.read(MAX_FRAME_BYTES + 1)
                if len(payload) > MAX_FRAME_BYTES or "sha256:" + hashlib.sha256(payload).hexdigest() != frame["frame_sha256"]:
                    raise ValueError("selected archive structure hash mismatch")

        counts = result.get("counts")
        expected_counts = {
            "generated_structure_count": len(frames),
            "energy_accepted_count": sum(1 for record in records if record.get("energy_filter_pass") is True),
            "selected_structure_count": len(selected_records),
        }
        if not isinstance(counts, dict) or any(counts.get(key) != value for key, value in expected_counts.items()):
            raise ValueError("sampling-result counts do not match fetched structures")
        if report.get("counts") != counts:
            raise ValueError("cluster report counts differ from sampling-result")
        meta_parameters = metadata.get("parameters", {})
        if meta_parameters.get("selection_stride") != identity.get("selection_stride") or meta_parameters.get("energy_max_ev") != identity.get("energy_max_ev") or meta_parameters.get("seed_status") != UNKNOWN:
            raise ValueError("LASP metadata differs from approved policy")
        if identity.get("include_best_arc"):
            best = root / "raw-run" / "best.arc"
            if not _ordinary(best, MAX_ARC_BYTES):
                raise ValueError("approved best.arc output is missing")
            if counts.get("aimd_seed_candidate_count") != len(
                LEGACY._read_arc_frames(best, int(identity["max_frames"]))
            ):
                raise ValueError("best.arc count differs from result")
        if identity.get("include_md_arc"):
            md = root / "raw-run" / "md.arc"
            if not _ordinary(md, MAX_ARC_BYTES):
                raise ValueError("approved md.arc output is missing")
            if counts.get("md_structure_count") != len(
                LEGACY._read_arc_frames(md, int(identity["max_frames"]))
            ):
                raise ValueError("md.arc count differs from result")
    except Exception as exc:
        diagnostics.append(_diag("ERROR", "lasp.scheduled_result", str(exc)))
        return {"plugin_id": "pes-sampling", "status": "FAIL", "diagnostics": diagnostics}
    return {
        "plugin_id": "pes-sampling",
        "status": "OK",
        "diagnostics": diagnostics,
        "metrics": {
            "generated_structure_count": len(frames),
            "selected_structure_count": len(selected_records),
            "energy_accepted_count": expected_counts["energy_accepted_count"],
        },
    }


def _scheduled_collect(context: Mapping[str, Any]) -> dict[str, Any]:
    checked = _scheduled_check(context)
    if checked.get("status") != "OK":
        return {"plugin_id": "pes-sampling", "status": "FAIL", "artifacts": [], "metrics": {}, "diagnostics": checked.get("diagnostics", [])}
    root = Path(str(context["attempt_dir"])).expanduser().absolute() / "lasp-ssw"
    roles = {
        "cluster-run-report.json": "lasp-cluster-report",
        "sampling-result.json": "sample-manifest",
        "ssw-structures.json": "ssw-structure-manifest",
        "selected-structures.json": "selected-structure-manifest",
        "lasp-run-metadata.json": "lasp-run-metadata",
        "selected-structures.tar.gz": "selected-structures-archive",
        "raw-run/allstr.arc": "ssw-archive",
        "raw-run/best.arc": "best-archive",
        "raw-run/md.arc": "md-archive",
        "aimd-seeds.json": "aimd-seed-manifest",
        "md-structures.json": "md-structure-manifest",
        "lasp.stdout.log": "lasp-stdout",
        "lasp.stderr.log": "lasp-stderr",
        "lasp.out": "lasp-native-log",
    }
    artifacts = []
    for relative, role in roles.items():
        path = root / relative
        if path.is_file() and not path.is_symlink():
            artifacts.append({"role": role, "path": str(path), "media_type": "application/json" if path.suffix == ".json" else "application/octet-stream"})
    return {"plugin_id": "pes-sampling", "status": "OK", "artifacts": artifacts, "metrics": checked.get("metrics", {}), "diagnostics": checked.get("diagnostics", [])}


class Adapter:
    def __init__(self) -> None:
        self._local = LEGACY.Adapter()

    def validate(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        if isinstance(context, Mapping) and context.get("backend") == "ssh-slurm":
            return _validate_scheduled(context)
        return self._local.validate(context)

    def plan(self, context: dict[str, Any]) -> dict[str, Any]:
        if isinstance(context, Mapping) and context.get("backend") == "ssh-slurm":
            return _plan_scheduled(context)
        return self._local.plan(context)

    def prepare(self, context: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
        if _scheduled(context):
            diagnostics = _validate_scheduled(context)
            if _errors(diagnostics) or not isinstance(plan, Mapping) or plan.get("status") != "READY":
                return _blocked(diagnostics + ([] if isinstance(plan, Mapping) and plan.get("status") == "READY" else [_diag("ERROR", "plan.ready_required", "prepare requires READY plan")]))
            return {"plugin_id": "pes-sampling", "status": "READY", "executable": True, "prepared": False, "writes_files": False, "plan": dict(plan), "diagnostics": diagnostics}
        return self._local.prepare(context, plan)

    def check(self, context: dict[str, Any]) -> dict[str, Any]:
        execution = context.get("execution", {}) if isinstance(context, Mapping) else {}
        plan = execution.get("plan", {}) if isinstance(execution, Mapping) else {}
        if isinstance(plan, Mapping) and isinstance(plan.get("scheduled_execution"), Mapping) and plan.get("operation") == "lasp-ssw-execute":
            return _scheduled_check(context)
        return self._local.check(context)

    def collect(self, context: dict[str, Any]) -> dict[str, Any]:
        execution = context.get("execution", {}) if isinstance(context, Mapping) else {}
        plan = execution.get("plan", {}) if isinstance(execution, Mapping) else {}
        if isinstance(plan, Mapping) and isinstance(plan.get("scheduled_execution"), Mapping) and plan.get("operation") == "lasp-ssw-execute":
            return _scheduled_collect(context)
        return self._local.collect(context)

    def replay(self, context: dict[str, Any]) -> dict[str, Any]:
        return self._local.replay(context)
