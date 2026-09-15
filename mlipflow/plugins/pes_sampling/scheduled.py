"""pes sampling: scheduled."""

from __future__ import annotations

import json
import math
import re
import tarfile
from pathlib import Path
from typing import Any, Mapping

from . import contracts
from . import lasp_input_arc as ARC_CONTRACT

HERE = Path(__file__).resolve().parent

WRAPPER_PATH = HERE / "lasp_ssw.py"

REMOTE_RUNNER_PATH = HERE / "lasp_cluster.py"

ARC_CONTRACT_PATH = HERE / "lasp_input_arc.py"

UNKNOWN = "HISTORICAL_PARAMETER_UNKNOWN"

MAX_FRAMES = 10000


SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _diag(level: str, code: str, message: str) -> dict[str, str]:
    return {"level": level, "code": code, "message": message}


def _errors(items: list[dict[str, str]]) -> bool:
    return any(item.get("level") == "ERROR" for item in items)


def _blocked(items: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "plugin_id": "pes-sampling",
        "status": "BLOCKED",
        "executable": False,
        "diagnostics": items,
    }


def _ordinary(path: Path | None) -> bool:
    return path is not None and path.is_file() and path.stat().st_size > 0


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _project_file(root: Path, selected: Any = None) -> Path | None:
    if isinstance(selected, str) and selected:
        path = Path(selected).expanduser().absolute()
        if path.is_file() and path.resolve().parent == root.resolve():
            return path.resolve()
        return None
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
    if not path.is_file():
        return None
    return path.resolve()


def _stage(path: Path, remote_name: str) -> dict[str, Any]:
    return {
        "source": str(path),
        "remote_name": remote_name,
    }


def _lasp_potential(path: Path) -> str:
    value = contracts._parse_lasp_input(path).get("potential")
    if not contracts._plain_string(value):
        raise ValueError("lasp.in must declare exactly one explicit potential")
    return str(value).lower()


def _potcar_bundle(
    manifest_path: Path, project_root: Path, structure: Path
) -> tuple[Path, dict[str, Any]]:
    manifest = _json(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("plugin_id") != "pes-sampling"
        or manifest.get("operation") != "lasp-input-prepare"
    ):
        raise ValueError("lasp_input_manifest schema is invalid")
    structure_record = manifest.get("output")
    if not isinstance(structure_record, Mapping) or structure_record.get("path") != "input.arc":
        raise ValueError("lasp_input_manifest does not bind input_structure")
    reference_path = manifest.get("pseudopotential_reference_path")
    potcar = manifest.get("potcar")
    if not isinstance(reference_path, str) or not reference_path:
        raise ValueError("lasp_input_manifest lacks a pseudopotential reference path")
    if not isinstance(potcar, Mapping):
        raise ValueError("lasp_input_manifest lacks a POTCAR record")
    reference_id = potcar.get("reference_id")
    functional = potcar.get("functional")
    elements = potcar.get("elements")
    symbols = potcar.get("symbols")
    components = potcar.get("components")
    output = potcar.get("output")
    if (
        not contracts._portable_source_id(reference_id)
        or potcar.get("source_env") != "PMG_VASP_PSP_DIR"
        or potcar.get("configuration_source") not in {"environment", "pymatgen-settings"}
        or not contracts._plain_string(functional)
        or potcar.get("portable_artifact") is not False
        or not isinstance(elements, list)
        or not elements
        or any(
            not isinstance(element, str) or re.fullmatch(r"[A-Z][a-z]?", element) is None
            for element in elements
        )
        or not isinstance(symbols, list)
        or len(symbols) != len(elements)
        or any(
            not isinstance(symbol, str) or SAFE_NAME.fullmatch(symbol) is None for symbol in symbols
        )
        or not isinstance(components, list)
        or len(components) != len(symbols)
        or any(
            not isinstance(component, Mapping) or component.get("symbol") != symbol
            for component, symbol in zip(components, symbols)
        )
        or not isinstance(output, Mapping)
        or output.get("path") != "POTCAR"
        or output.get("collectable") is not False
    ):
        raise ValueError("lasp_input_manifest POTCAR record is invalid")
    potcar_path = manifest_path.parent / "POTCAR"
    if not _ordinary(potcar_path) or not _within(potcar_path, project_root):
        raise ValueError("runtime-only POTCAR is missing or differs from its manifest")
    record = {
        "reference_id": reference_id,
        "reference_path": reference_path,
        "functional": functional,
        "elements": elements,
        "symbols": symbols,
        "components": [dict(component) for component in components],
        "configuration_source": potcar.get("configuration_source"),
        "manifest_path": str(manifest_path),
        "potcar_path": str(potcar_path),
        "portable_or_collectable": False,
    }
    return potcar_path, record


def _operation(context: Mapping[str, Any]) -> str:
    parameters = context.get("parameters", {})
    return (
        str(parameters.get("operation", "direct-select"))
        if isinstance(parameters, Mapping)
        else "direct-select"
    )


def validate(context: Any) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    if not isinstance(context, Mapping):
        return [_diag("ERROR", "context.mapping_required", "context must be a mapping")]
    if context.get("backend") != "ssh-slurm":
        diagnostics.append(
            _diag(
                "ERROR", "backend.ssh_slurm_required", "scheduled LASP requires backend ssh-slurm"
            )
        )
    if _operation(context) != "lasp-ssw-execute":
        diagnostics.append(
            _diag(
                "ERROR", "operation.scheduled_lasp", "ssh-slurm LASP supports lasp-ssw-execute only"
            )
        )
    root = Path(str(context.get("project_root", ""))).expanduser().absolute()
    if not root.is_dir():
        diagnostics.append(
            _diag("ERROR", "path.project_root", "project_root must be an existing directory")
        )
        return diagnostics
    project_path = _project_file(root, context.get("project_path"))
    if project_path is None:
        diagnostics.append(
            _diag(
                "ERROR",
                "path.project_file",
                "scheduled LASP requires project.yaml/project.yml/project.json",
            )
        )

    inputs = context.get("inputs", {})
    if not isinstance(inputs, Mapping):
        diagnostics.append(_diag("ERROR", "inputs.mapping_required", "inputs must be a mapping"))
        return diagnostics
    allowed_inputs = {
        "input_structure",
        "lasp_input",
        "lasp_input_manifest",
        "lasp_auxiliary_files",
        "result_manifest",
    }
    unknown_inputs = sorted(set(inputs) - allowed_inputs)
    if unknown_inputs:
        diagnostics.append(
            _diag(
                "ERROR",
                "input.unknown",
                "scheduled LASP does not accept: " + ", ".join(unknown_inputs),
            )
        )
    structure = _resolve_input(inputs.get("input_structure"), root)
    lasp_input = _resolve_input(inputs.get("lasp_input"), root)
    if structure is None or not _within(structure, root):
        diagnostics.append(
            _diag(
                "ERROR", "path.input_structure", "input_structure must be an ordinary project file"
            )
        )
    else:
        try:
            ARC_CONTRACT.canonicalize_prepared_arc(structure.read_bytes())
        except Exception as exc:
            diagnostics.append(
                _diag(
                    "ERROR",
                    "input.structure_arc",
                    f"input_structure must normalize to one LASP ARC frame: {exc}",
                )
            )
    potential: str | None = None
    if lasp_input is None or not _within(lasp_input, root):
        diagnostics.append(
            _diag("ERROR", "path.lasp_input", "lasp_input must be an ordinary project file")
        )
    else:
        try:
            contracts._validate_ssw_input(contracts._parse_lasp_input(lasp_input))
            potential = _lasp_potential(lasp_input)
        except Exception as exc:
            diagnostics.append(
                _diag("ERROR", "input.lasp_ssw_contract", f"invalid LASP SSW input: {exc}")
            )

    manifest_path = _resolve_input(inputs.get("lasp_input_manifest"), root)
    if potential == "vasp":
        if manifest_path is None or not _within(manifest_path, root) or structure is None:
            diagnostics.append(
                _diag(
                    "ERROR",
                    "path.lasp_input_manifest",
                    "potential vasp requires the verified lasp-input-prepare manifest",
                )
            )
        else:
            try:
                _potcar_bundle(manifest_path, root, structure)
            except Exception as exc:
                diagnostics.append(_diag("ERROR", "input.potcar_bundle", str(exc)))
    elif manifest_path is not None:
        diagnostics.append(
            _diag(
                "ERROR",
                "input.unused_lasp_input_manifest",
                "lasp_input_manifest is accepted only for potential vasp",
            )
        )

    auxiliary = inputs.get("lasp_auxiliary_files", {})
    if auxiliary is None:
        auxiliary = {}
    if not isinstance(auxiliary, Mapping):
        diagnostics.append(
            _diag("ERROR", "input.lasp_auxiliary_files", "lasp_auxiliary_files must be a mapping")
        )
    else:
        reserved = {
            "input.arc",
            "lasp.in",
            "all.arc",
            "allstr.arc",
            "allstr.native.arc",
            "project.yaml",
            "lasp_ssw.py",
            "lasp_cluster.py",
            "lasp_input_arc.py",
            "lasp-input-manifest.json",
            "POTCAR",
        }
        for name, source in auxiliary.items():
            if not isinstance(name, str) or not SAFE_NAME.fullmatch(name) or name in reserved:
                diagnostics.append(
                    _diag(
                        "ERROR", "input.auxiliary_name", f"unsafe auxiliary destination: {name!r}"
                    )
                )
                break
            path = _resolve_input(source, root)
            if path is None or not _within(path, root):
                diagnostics.append(
                    _diag(
                        "ERROR",
                        "path.auxiliary_file",
                        f"auxiliary {name} must be an ordinary project file",
                    )
                )
                break

    parameters = context.get("parameters", {})
    if not isinstance(parameters, Mapping):
        diagnostics.append(
            _diag("ERROR", "parameters.mapping_required", "parameters must be a mapping")
        )
        return diagnostics
    allowed_parameters = {
        "operation",
        "output_subdir",
        "historical_source_id",
        "selection_stride",
        "energy_max_ev",
        "max_frames",
        "include_best_arc",
        "include_md_arc",
        "seed_status",
        "acknowledge_uncontrolled_seed",
        "preserve_historical_order",
        "lasp_version",
        "mpi_processes",
    }
    unknown_parameters = sorted(set(parameters) - allowed_parameters)
    if unknown_parameters:
        diagnostics.append(
            _diag(
                "ERROR",
                "parameter.unknown",
                "unsupported LASP parameter(s): " + ", ".join(unknown_parameters),
            )
        )
    if parameters.get("output_subdir", "lasp-ssw") != "lasp-ssw":
        diagnostics.append(
            _diag(
                "ERROR",
                "parameter.output_subdir",
                "scheduled LASP currently requires output_subdir=lasp-ssw",
            )
        )
    if not contracts._portable_source_id(parameters.get("historical_source_id")):
        diagnostics.append(
            _diag(
                "ERROR", "parameter.historical_source_id", "historical_source_id must be portable"
            )
        )
    if not contracts._positive_int(parameters.get("selection_stride")):
        diagnostics.append(
            _diag("ERROR", "parameter.selection_stride", "selection_stride must be positive")
        )
    max_frames = parameters.get("max_frames")
    if not contracts._positive_int(max_frames) or int(max_frames) > MAX_FRAMES:
        diagnostics.append(
            _diag("ERROR", "parameter.max_frames", f"max_frames must be 1..{MAX_FRAMES}")
        )
    energy = parameters.get("energy_max_ev")
    if energy is not None and (
        isinstance(energy, bool)
        or not isinstance(energy, (int, float))
        or not math.isfinite(float(energy))
    ):
        diagnostics.append(
            _diag("ERROR", "parameter.energy_max_ev", "energy_max_ev must be finite or null")
        )
    for name in ("include_best_arc", "include_md_arc"):
        if not isinstance(parameters.get(name, False), bool):
            diagnostics.append(_diag("ERROR", f"parameter.{name}", f"{name} must be boolean"))
    if parameters.get("seed_status") != UNKNOWN:
        diagnostics.append(
            _diag("ERROR", "seed.historical_unknown", f"seed_status must be {UNKNOWN}")
        )
    if parameters.get("acknowledge_uncontrolled_seed") is not True:
        diagnostics.append(
            _diag(
                "ERROR",
                "seed.acknowledgement_required",
                "acknowledge_uncontrolled_seed must be true",
            )
        )
    if parameters.get("preserve_historical_order") is not True:
        diagnostics.append(
            _diag(
                "ERROR",
                "parameter.preserve_historical_order",
                "preserve_historical_order must be true",
            )
        )
    if not contracts._plain_string(parameters.get("lasp_version")):
        diagnostics.append(
            _diag("ERROR", "parameter.lasp_version", "lasp_version must be explicit")
        )
    if parameters.get("mpi_processes") is not None:
        diagnostics.append(
            _diag(
                "ERROR",
                "parameter.mpi_site_owned",
                "scheduled LASP process count comes from resources.cpus/site template, not mpi_processes",
            )
        )
    if (
        not _ordinary(WRAPPER_PATH)
        or not _ordinary(REMOTE_RUNNER_PATH)
        or not _ordinary(ARC_CONTRACT_PATH)
    ):
        diagnostics.append(
            _diag("ERROR", "path.bundled_runner", "bundled LASP cluster helpers are missing")
        )
    return diagnostics


def _fetch(
    remote_name: str, remote_path: str, local_name: str, required: bool, role: str
) -> dict[str, Any]:
    return {
        "remote_name": remote_name,
        "remote_path": remote_path,
        "local_name": local_name,
        "required": required,
        "role": role,
    }


def plan(context: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = validate(context)
    if _errors(diagnostics):
        return _blocked(diagnostics)
    root = Path(str(context["project_root"])).expanduser().absolute().resolve()
    inputs = dict(context.get("inputs", {}))
    parameters = dict(context.get("parameters", {}))
    project_path = _project_file(root, context.get("project_path"))
    structure = _resolve_input(inputs["input_structure"], root)
    lasp_input = _resolve_input(inputs["lasp_input"], root)
    assert project_path is not None and structure is not None and lasp_input is not None
    potential = _lasp_potential(lasp_input)
    _canonical_arc, arc_conversion = ARC_CONTRACT.canonicalize_prepared_arc(structure.read_bytes())
    staged = [
        _stage(project_path, "project.yaml"),
        _stage(structure, "input.arc"),
        _stage(lasp_input, "lasp.in"),
        _stage(WRAPPER_PATH, "lasp_ssw.py"),
        _stage(REMOTE_RUNNER_PATH, "lasp_cluster.py"),
        _stage(ARC_CONTRACT_PATH, "lasp_input_arc.py"),
    ]
    pseudopotential = None
    if potential == "vasp":
        lasp_input_manifest = _resolve_input(inputs["lasp_input_manifest"], root)
        assert lasp_input_manifest is not None
        potcar_path, pseudopotential = _potcar_bundle(lasp_input_manifest, root, structure)
        staged.extend(
            [
                _stage(lasp_input_manifest, "lasp-input-manifest.json"),
                _stage(potcar_path, "POTCAR"),
            ]
        )
    auxiliary = inputs.get("lasp_auxiliary_files", {}) or {}
    for name in sorted(auxiliary):
        path = _resolve_input(auxiliary[name], root)
        assert path is not None
        staged.append(_stage(path, str(name)))

    include_best = bool(parameters.get("include_best_arc", False))
    include_md = bool(parameters.get("include_md_arc", False))
    fetch_outputs = [
        _fetch(
            "cluster-run-report.json",
            "output/cluster-run-report.json",
            "lasp-ssw/cluster-run-report.json",
            True,
            "lasp-cluster-report",
        ),
        _fetch(
            "sampling-result.json",
            "output/lasp-ssw/sampling-result.json",
            "lasp-ssw/sampling-result.json",
            True,
            "sample-manifest",
        ),
        _fetch(
            "ssw-structures.json",
            "output/lasp-ssw/ssw-structures.json",
            "lasp-ssw/ssw-structures.json",
            True,
            "ssw-structure-manifest",
        ),
        _fetch(
            "selected-structures.json",
            "output/lasp-ssw/selected-structures.json",
            "lasp-ssw/selected-structures.json",
            True,
            "selected-structure-manifest",
        ),
        _fetch(
            "lasp-run-metadata.json",
            "output/lasp-ssw/lasp-run-metadata.json",
            "lasp-ssw/lasp-run-metadata.json",
            True,
            "lasp-run-metadata",
        ),
        _fetch(
            "selected-structures.tar.gz",
            "output/selected-structures.tar.gz",
            "lasp-ssw/selected-structures.tar.gz",
            True,
            "selected-structures-archive",
        ),
        _fetch(
            "allstr.arc",
            "output/lasp-ssw/raw-run/allstr.arc",
            "lasp-ssw/raw-run/allstr.arc",
            True,
            "ssw-archive",
        ),
        _fetch(
            "best.arc",
            "output/lasp-ssw/raw-run/best.arc",
            "lasp-ssw/raw-run/best.arc",
            include_best,
            "best-archive",
        ),
        _fetch(
            "md.arc",
            "output/lasp-ssw/raw-run/md.arc",
            "lasp-ssw/raw-run/md.arc",
            include_md,
            "md-archive",
        ),
        _fetch(
            "aimd-seeds.json",
            "output/lasp-ssw/aimd-seeds.json",
            "lasp-ssw/aimd-seeds.json",
            include_best,
            "aimd-seed-manifest",
        ),
        _fetch(
            "md-structures.json",
            "output/lasp-ssw/md-structures.json",
            "lasp-ssw/md-structures.json",
            include_md,
            "md-structure-manifest",
        ),
        _fetch(
            "lasp.stdout.log",
            "output/lasp-ssw/raw-run/lasp.stdout.log",
            "lasp-ssw/lasp.stdout.log",
            False,
            "lasp-stdout",
        ),
        _fetch(
            "lasp.stderr.log",
            "output/lasp-ssw/raw-run/lasp.stderr.log",
            "lasp-ssw/lasp.stderr.log",
            False,
            "lasp-stderr",
        ),
        _fetch(
            "lasp.out",
            "output/lasp-ssw/raw-run/lasp.out",
            "lasp-ssw/lasp.out",
            False,
            "lasp-native-log",
        ),
    ]
    calculation = {
        "historical_source_id": parameters["historical_source_id"],
        "selection_stride": parameters["selection_stride"],
        "energy_max_ev": parameters.get("energy_max_ev"),
        "max_frames": parameters["max_frames"],
        "include_best_arc": include_best,
        "include_md_arc": include_md,
        "seed_status": UNKNOWN,
        "preserve_historical_order": True,
        "lasp_version": parameters["lasp_version"],
        "potential": potential,
        "pseudopotential": pseudopotential,
        "input_structure_path": str(structure),
        "input_arc_conversion": arc_conversion,
        "lasp_input_path": str(lasp_input),
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
        "lasp_calculation": calculation,
        "approval_summary": {
            "expensive": True,
            "submits_jobs": True,
            "execution_model": "mpi",
            "cpus_meaning": "mpi-task-count",
            "framework": "LASP",
            "sampling_method": "stochastic-surface-walking",
            "potential": potential,
            "runs_vasp": potential == "vasp",
            "pseudopotential": pseudopotential,
            "potcar_staged_sensitive": potential == "vasp",
            "input_arc_conversion": arc_conversion,
            "input_structure_path": str(structure),
            "max_frames": parameters["max_frames"],
            "selection_stride": parameters["selection_stride"],
            "fetch_allowlist": sorted(item["remote_name"] for item in fetch_outputs),
            "staged_file_count": len(staged),
        },
        "input_paths": {item["remote_name"]: item["source"] for item in staged},
        "scheduled_execution": {
            "schema_version": 3,
            "execution_model": "mpi",
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
    if not _ordinary(path):
        raise ValueError(f"missing or empty JSON artifact: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _pseudopotential_settings(record: Any) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError("LASP pseudopotential record must be a mapping")
    path_names = {
        "reference_path": Path(str(record.get("reference_path", ""))).name,
        "manifest_path": Path(str(record.get("manifest_path", ""))).name,
        "potcar_path": Path(str(record.get("potcar_path", ""))).name,
    }
    if (
        not path_names["reference_path"]
        or path_names["manifest_path"] != "lasp-input-manifest.json"
        or path_names["potcar_path"] != "POTCAR"
    ):
        raise ValueError("LASP pseudopotential path record is invalid")
    return {
        "reference_id": record.get("reference_id"),
        "reference_name": path_names["reference_path"],
        "functional": record.get("functional"),
        "elements": record.get("elements"),
        "symbols": record.get("symbols"),
        "components": record.get("components"),
        "configuration_source": record.get("configuration_source"),
        "portable_or_collectable": record.get("portable_or_collectable"),
    }


def check(context: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics: list[dict[str, str]] = []
    execution = context.get("execution", {})
    plan = execution.get("plan", {}) if isinstance(execution, Mapping) else {}
    calculation = plan.get("lasp_calculation", {}) if isinstance(plan, Mapping) else {}
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
        if not _ordinary(allstr) or not _ordinary(archive):
            raise ValueError("required LASP archive output is missing")
        if (
            report.get("schema_version") != 1
            or report.get("plugin_id") != "pes-sampling"
            or report.get("status") != "OK"
        ):
            raise ValueError("cluster-run-report schema/status is invalid")
        if (
            report.get("operation") != "lasp-ssw-execute"
            or result.get("operation") != "lasp-ssw-execute"
            or result.get("status") != "OK"
        ):
            raise ValueError("remote result operation/status mismatch")
        if report.get("lasp_version") != calculation.get("lasp_version"):
            raise ValueError("LASP version differs from approved plan")
        if report.get("potential") != calculation.get("potential"):
            raise ValueError("LASP potential differs from approved plan")
        if _pseudopotential_settings(report.get("pseudopotential")) != (
            _pseudopotential_settings(calculation.get("pseudopotential"))
        ):
            raise ValueError("LASP pseudopotential settings differ from approved plan")
        input_structure = report.get("input_structure", {})
        if (
            not isinstance(input_structure.get("source_path"), str)
            or not input_structure.get("source_path")
            or not isinstance(input_structure.get("canonical_path"), str)
            or not input_structure.get("canonical_path")
            or input_structure.get("conversion") != calculation.get("input_arc_conversion")
        ):
            raise ValueError("LASP input ARC conversion differs from approved plan")
        policy = report.get("selection_policy")
        expected_policy = {
            "selection_stride": calculation.get("selection_stride"),
            "energy_max_ev": calculation.get("energy_max_ev"),
            "max_frames": calculation.get("max_frames"),
            "preserve_historical_order": True,
        }
        if policy != expected_policy:
            raise ValueError("cluster selection policy differs from approved plan")
        if not report.get("source_outputs", {}).get("allstr_arc_path"):
            raise ValueError("cluster report lacks the allstr.arc output path")
        archive_record = report.get("selected_archive", {})
        if archive_record.get("path") != "selected-structures.tar.gz":
            raise ValueError("cluster report has an unexpected selected archive path")

        frames = contracts._read_arc_frames(allstr, int(calculation["max_frames"]))
        records = structures.get("structures")
        selected_records = selected.get("structures")
        if (
            not isinstance(records, list)
            or not isinstance(selected_records, list)
            or len(records) != len(frames)
        ):
            raise ValueError("structure manifests do not match allstr.arc frame count")
        accepted_order = 0
        expected_selected: list[tuple[dict[str, Any], dict[str, Any], int]] = []
        threshold = calculation.get("energy_max_ev")
        stride = int(calculation["selection_stride"])
        for frame, record in zip(frames, records):
            if not isinstance(record, dict):
                raise ValueError("invalid SSW structure record")
            index = int(frame["frame_index"])
            expected_id = f"lasp-ssw-{index:06d}"
            if (
                record.get("frame_index") != index
                or record.get("historical_order") != index
                or record.get("structure_id") != expected_id
            ):
                raise ValueError(f"SSW frame record mismatch at frame {index}")
            if float(record.get("energy_ev")) != float(frame["energy_ev"]):
                raise ValueError(f"SSW energy mismatch at frame {index}")
            accepted = threshold is None or float(frame["energy_ev"]) <= float(threshold)
            if accepted:
                accepted_order += 1
            chosen = accepted and (accepted_order - 1) % stride == 0
            if (
                bool(record.get("energy_filter_pass")) != accepted
                or bool(record.get("selected")) != chosen
            ):
                raise ValueError(f"SSW selection flags are inconsistent at frame {index}")
            if chosen:
                expected_selected.append((frame, record, len(expected_selected) + 1))
        if len(expected_selected) != len(selected_records):
            raise ValueError("selected structure count does not match approved policy")
        expected_names: list[str] = []
        for (frame, record, order), selected_record in zip(expected_selected, selected_records):
            if (
                not isinstance(selected_record, dict)
                or selected_record.get("structure_id") != record.get("structure_id")
                or selected_record.get("selected_order") != order
            ):
                raise ValueError("selected manifest record/order mismatch")
            expected_names.append(f"selected/input-{order:06d}.arc")
        with tarfile.open(archive, "r:gz") as bundle:
            members = bundle.getmembers()
            names = [item.name for item in members]
            if names != expected_names:
                raise ValueError("selected archive members differ from selected manifest")
            for member, (frame, _, _) in zip(members, expected_selected):
                if not member.isfile():
                    raise ValueError("selected archive contains a non-file member")
                stream = bundle.extractfile(member)
                if stream is None:
                    raise ValueError("selected archive member cannot be read")
                payload = stream.read()
                if payload != frame["payload"]:
                    raise ValueError("selected archive structure differs from allstr.arc")

        counts = result.get("counts")
        expected_counts = {
            "generated_structure_count": len(frames),
            "energy_accepted_count": sum(
                1 for record in records if record.get("energy_filter_pass") is True
            ),
            "selected_structure_count": len(selected_records),
        }
        if not isinstance(counts, dict) or any(
            counts.get(key) != value for key, value in expected_counts.items()
        ):
            raise ValueError("sampling-result counts do not match fetched structures")
        if report.get("counts") != counts:
            raise ValueError("cluster report counts differ from sampling-result")
        meta_parameters = metadata.get("parameters", {})
        if (
            meta_parameters.get("selection_stride") != calculation.get("selection_stride")
            or meta_parameters.get("energy_max_ev") != calculation.get("energy_max_ev")
            or meta_parameters.get("seed_status") != UNKNOWN
        ):
            raise ValueError("LASP metadata differs from approved policy")
        if calculation.get("include_best_arc"):
            best = root / "raw-run" / "best.arc"
            if not _ordinary(best):
                raise ValueError("approved best.arc output is missing")
            if counts.get("aimd_seed_candidate_count") != len(
                contracts._read_arc_frames(best, int(calculation["max_frames"]))
            ):
                raise ValueError("best.arc count differs from result")
        if calculation.get("include_md_arc"):
            md = root / "raw-run" / "md.arc"
            if not _ordinary(md):
                raise ValueError("approved md.arc output is missing")
            if counts.get("md_structure_count") != len(
                contracts._read_arc_frames(md, int(calculation["max_frames"]))
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


def collect(context: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(context["attempt_dir"])).expanduser().absolute() / "lasp-ssw"
    result = json.loads((root / "sampling-result.json").read_text(encoding="utf-8"))
    counts = result["counts"]
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
        if path.is_file():
            artifacts.append(
                {
                    "role": role,
                    "path": str(path),
                    "media_type": "application/json"
                    if path.suffix == ".json"
                    else "application/octet-stream",
                }
            )
    return {
        "plugin_id": "pes-sampling",
        "status": "OK",
        "artifacts": artifacts,
        "metrics": {name: counts[name] for name in (
            "generated_structure_count", "selected_structure_count", "energy_accepted_count"
        )},
        "diagnostics": [],
    }
