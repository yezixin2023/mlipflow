"""Scheduled workflow adapter for explicit-model ASE molecular dynamics."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

PLUGIN_ID = "ase-md"
BACKEND = "ssh-slurm"
CALCULATORS = ("deepmd", "m3gnet", "chgnet", "mace")
MODEL_KINDS = {"deepmd": "file", "m3gnet": "directory", "chgnet": "file", "mace": "file"}
TEMPLATE_FAMILIES = {name: f"ase-md-{name}" for name in CALCULATORS}
TEMPLATE_FAMILIES.update(
    {name: f"ase-md-{name}-canonical" for name in ("m3gnet", "chgnet")}
)
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_STRUCTURE_BYTES = 64 * 1024 * 1024
MAX_TRAJECTORY_BYTES = 4 * 1024 * 1024 * 1024
MAX_THERMO_BYTES = 256 * 1024 * 1024
MAX_FINAL_BYTES = 64 * 1024 * 1024
MAX_STEPS = 100_000_000
MAX_RECORDS = 500_000


def _diagnostic(level: str, code: str, message: str) -> dict[str, str]:
    return {"level": level, "code": code, "message": message}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _reference(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict) and isinstance(value.get("path"), str) and value["path"]:
        return value["path"]
    return None


def _plain_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and "\x00" not in value and "\n" not in value and "\r" not in value


def _safe_relative(value: Any) -> bool:
    if not _plain_string(value) or "\\" in str(value):
        return False
    path = PurePosixPath(str(value))
    return not path.is_absolute() and bool(path.parts) and all(part not in {"", ".", ".."} for part in path.parts)


def _is_fingerprint(value: Any) -> bool:
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


def _project_file(root: Path, value: Any) -> Path:
    reference = _reference(value)
    if reference is None:
        raise ValueError("missing project file reference")
    path = Path(reference)
    candidate = path.absolute() if path.is_absolute() else (root / path).absolute()
    try:
        candidate.resolve().relative_to(root.resolve())
    except (OSError, ValueError) as exc:
        raise ValueError("project input escapes project root") from exc
    return candidate


def _ordinary_project_file(path: Path, root: Path, max_bytes: int) -> bool:
    try:
        if path.is_symlink() or not path.is_file():
            return False
        resolved = path.resolve()
        resolved.relative_to(root.resolve())
        if not 0 < resolved.stat().st_size <= max_bytes:
            return False
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


def _read_json(path: Path) -> dict[str, Any]:
    if not _ordinary_file(path, MAX_JSON_BYTES):
        raise ValueError(f"not an ordinary bounded JSON file: {path.name}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name} must contain an object")
    return raw


def _model_reference(path: Path, calculator: str) -> dict[str, str]:
    raw = _read_json(path)
    if raw.get("schema_version") != 1:
        raise ValueError("model reference schema_version must equal 1")
    model_id = raw.get("model_id")
    if not _plain_string(model_id):
        raise ValueError("model reference requires model_id")
    relative = raw.get("relative_path", model_id)
    if not _safe_relative(relative):
        raise ValueError("model relative_path must be a safe site-root-relative path")
    expected_kind = MODEL_KINDS[calculator]
    kind = raw.get("kind", expected_kind)
    if kind != expected_kind:
        raise ValueError(f"{calculator} model reference kind must be {expected_kind}")
    fingerprint = raw.get("fingerprint")
    if not _is_fingerprint(fingerprint):
        raise ValueError("model reference fingerprint must be sha256:<64 lowercase hex>")
    return {
        "model_id": str(model_id),
        "relative_path": str(relative),
        "kind": str(kind),
        "fingerprint": str(fingerprint),
    }


def _staged_record(source: Path, remote_name: str) -> dict[str, Any]:
    return {
        "source": str(source.absolute()),
        "remote_name": remote_name,
        "sha256": _sha256(source),
        "size_bytes": source.stat().st_size,
        "sensitive": False,
        "fetch_allowed": False,
    }


def _positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _nonnegative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0


def _valid_supercell_repeat(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(
            not isinstance(item, bool) and isinstance(item, int) and item > 0
            for item in value
        )
    )


def _expected_steps(total: int, interval: int) -> list[int]:
    values = [0, *range(interval, total + 1, interval)]
    if values[-1] != total:
        values.append(total)
    return values


def _validate_resources(context: dict[str, Any]) -> list[dict[str, str]]:
    resources = context.get("resources")
    if not isinstance(resources, dict):
        return [_diagnostic("error", "ase_md.resources", "resources must be an object")]
    diagnostics: list[dict[str, str]] = []
    allowed = {"cpus", "gpus", "memory", "walltime"}
    if set(resources) != allowed:
        diagnostics.append(_diagnostic("error", "ase_md.resource_fields", "ssh-slurm resources must contain exactly cpus, gpus, memory, walltime"))
        return diagnostics
    if not _positive_int(resources.get("cpus")):
        diagnostics.append(_diagnostic("error", "ase_md.cpus", "resources.cpus must be positive"))
    if not _nonnegative_int(resources.get("gpus")):
        diagnostics.append(_diagnostic("error", "ase_md.gpus", "resources.gpus must be non-negative"))
    for key in ("memory", "walltime"):
        if not _plain_string(resources.get(key)):
            diagnostics.append(_diagnostic("error", f"ase_md.{key}", f"resources.{key} must be a non-empty string"))
    return diagnostics


def _validate(context: dict[str, Any]) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    if not isinstance(context, dict):
        return [_diagnostic("error", "context.type", "context must be an object")]
    if context.get("backend") != BACKEND:
        diagnostics.append(_diagnostic("error", "ase_md.backend", "ase-md 0.1 supports ssh-slurm only"))
    if not _plain_string(context.get("project_root")) or not _plain_string(context.get("attempt_dir")):
        diagnostics.append(_diagnostic("error", "ase_md.context_paths", "project_root and attempt_dir are required"))
    inputs = context.get("inputs")
    parameters = context.get("parameters")
    if not isinstance(inputs, dict):
        diagnostics.append(_diagnostic("error", "ase_md.inputs", "inputs must be an object"))
        inputs = {}
    if not isinstance(parameters, dict):
        diagnostics.append(_diagnostic("error", "ase_md.parameters", "parameters must be an object"))
        parameters = {}
    for key in ("structure", "model_reference"):
        if _reference(inputs.get(key)) is None:
            diagnostics.append(_diagnostic("error", f"ase_md.input_{key}", f"inputs.{key} is required"))
    calculator = parameters.get("calculator")
    if calculator not in CALCULATORS:
        diagnostics.append(_diagnostic("error", "ase_md.calculator", "calculator must be deepmd, m3gnet, chgnet, or mace"))
    if parameters.get("ensemble", "nvt-langevin") != "nvt-langevin":
        diagnostics.append(_diagnostic("error", "ase_md.ensemble", "ase-md 0.1 implements nvt-langevin only"))
    if not _finite_positive(parameters.get("temperature_k")):
        diagnostics.append(_diagnostic("error", "ase_md.temperature", "temperature_k must be finite and positive"))
    if not _finite_positive(parameters.get("timestep_fs")) or float(parameters.get("timestep_fs", 0)) > 10.0:
        diagnostics.append(_diagnostic("error", "ase_md.timestep", "timestep_fs must be in (0, 10]"))
    steps = parameters.get("steps")
    if not _positive_int(steps) or int(steps or 0) > MAX_STEPS:
        diagnostics.append(_diagnostic("error", "ase_md.steps", f"steps must be a positive integer <= {MAX_STEPS}"))
    for key in ("trajectory_interval", "thermo_interval"):
        value = parameters.get(key)
        if not _positive_int(value) or (_positive_int(steps) and int(value) > int(steps)):
            diagnostics.append(_diagnostic("error", f"ase_md.{key}", f"{key} must be positive and no greater than steps"))
    if _positive_int(steps):
        for key in ("trajectory_interval", "thermo_interval"):
            value = parameters.get(key)
            if _positive_int(value) and len(_expected_steps(int(steps), int(value))) > MAX_RECORDS:
                diagnostics.append(_diagnostic("error", f"ase_md.{key}_records", f"{key} would exceed {MAX_RECORDS} bounded records"))
    seed = parameters.get("seed")
    if not _nonnegative_int(seed):
        diagnostics.append(_diagnostic("error", "ase_md.seed", "seed must be a non-negative integer"))
    device = parameters.get("device")
    if device not in {"cpu", "cuda"}:
        diagnostics.append(_diagnostic("error", "ase_md.device", "device must be cpu or cuda"))
    if device == "cuda" and isinstance(context.get("resources"), dict) and not _positive_int(context["resources"].get("gpus")):
        diagnostics.append(_diagnostic("error", "ase_md.cuda_resource", "device=cuda requires resources.gpus >= 1"))
    dtype = parameters.get("default_dtype")
    if dtype not in {"float32", "float64"}:
        diagnostics.append(_diagnostic("error", "ase_md.dtype", "default_dtype must be float32 or float64"))
    if calculator == "chgnet" and dtype != "float32":
        diagnostics.append(_diagnostic("error", "ase_md.chgnet_dtype", "CHGNet ASE MD requires default_dtype=float32"))
    if not _finite_positive(parameters.get("friction_per_fs")):
        diagnostics.append(_diagnostic("error", "ase_md.friction", "friction_per_fs must be finite and positive"))
    if not isinstance(parameters.get("fix_com"), bool):
        diagnostics.append(_diagnostic("error", "ase_md.fix_com", "fix_com must be boolean"))
    declared_model_fingerprint = parameters.get("model_fingerprint")
    if declared_model_fingerprint is not None and not _is_fingerprint(
        declared_model_fingerprint
    ):
        diagnostics.append(
            _diagnostic(
                "error",
                "ase_md.model_fingerprint",
                "model_fingerprint must be sha256:<64 lowercase hex> when supplied",
            )
        )
    if not _is_fingerprint(parameters.get("structure_fingerprint")):
        diagnostics.append(_diagnostic("error", "ase_md.structure_fingerprint", "structure_fingerprint must be sha256:<64 lowercase hex>"))
    repeat = parameters.get("supercell_repeat")
    if repeat is not None and not _valid_supercell_repeat(repeat):
        diagnostics.append(
            _diagnostic(
                "error",
                "ase_md.supercell_repeat",
                "supercell_repeat must contain exactly three positive integers",
            )
        )
    minimum_cell = parameters.get("minimum_initial_cell_length_angstrom")
    if minimum_cell is not None and not _finite_positive(minimum_cell):
        diagnostics.append(
            _diagnostic(
                "error",
                "ase_md.minimum_initial_cell_length",
                "minimum_initial_cell_length_angstrom must be finite and positive",
            )
        )
    for key in ("input_format", "input_index"):
        value = parameters.get(key)
        if value is not None and not _plain_string(str(value)):
            diagnostics.append(_diagnostic("error", f"ase_md.{key}", f"{key} must be a plain string when provided"))
    diagnostics.extend(_validate_resources(context))
    return diagnostics


def _blocked(diagnostics: list[dict[str, str]]) -> dict[str, Any]:
    return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": diagnostics}


def _plan(context: dict[str, Any]) -> dict[str, Any]:
    diagnostics = _validate(context)
    if diagnostics:
        return _blocked(diagnostics)
    root = Path(str(context["project_root"])).expanduser().absolute()
    inputs = _mapping(context["inputs"])
    parameters = _mapping(context["parameters"])
    try:
        project = _project_config(root)
        structure = _project_file(root, inputs["structure"])
        model_reference = _project_file(root, inputs["model_reference"])
    except ValueError as exc:
        return _blocked([_diagnostic("error", "ase_md.project_inputs", str(exc))])
    for name, path, limit in (
        ("project", project, MAX_JSON_BYTES),
        ("structure", structure, MAX_STRUCTURE_BYTES),
        ("model_reference", model_reference, MAX_JSON_BYTES),
    ):
        if not _ordinary_project_file(path, root, limit):
            diagnostics.append(_diagnostic("error", f"ase_md.{name}_file", f"{name} must be an ordinary bounded file inside project_root"))
    if diagnostics:
        return _blocked(diagnostics)
    calculator = str(parameters["calculator"])
    try:
        model = _model_reference(model_reference, calculator)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return _blocked([_diagnostic("error", "ase_md.model_reference", str(exc))])
    structure_fp = _sha256(structure)
    if parameters["structure_fingerprint"] != structure_fp:
        diagnostics.append(_diagnostic("error", "ase_md.structure_identity", "parameters.structure_fingerprint does not match the staged structure"))
    if (
        parameters.get("model_fingerprint") is not None
        and parameters["model_fingerprint"] != model["fingerprint"]
    ):
        diagnostics.append(_diagnostic("error", "ase_md.model_identity", "parameters.model_fingerprint does not match the model reference"))
    if diagnostics:
        return _blocked(diagnostics)

    plugin_root = Path(__file__).resolve().parent
    runner = plugin_root / "ase_md.py"
    cluster = plugin_root / "ase_md_cluster.py"
    for path in (runner, cluster):
        if not _ordinary_file(path):
            return _blocked([_diagnostic("error", "ase_md.bundled_runner", f"missing bundled runner: {path.name}")])
    structure_remote = f"structure/{structure.name}"
    staged = [
        _staged_record(project, "project.yaml"),
        _staged_record(structure, structure_remote),
        _staged_record(model_reference, "model-reference.json"),
        _staged_record(runner, "ase_md.py"),
        _staged_record(cluster, "ase_md_cluster.py"),
    ]
    fetch_outputs = [
        {"remote_name": "md-result.json", "remote_path": "output/md-result.json", "local_name": "md-result.json", "required": True, "max_bytes": MAX_JSON_BYTES, "role": "md-result"},
        {"remote_name": "trajectory.traj", "remote_path": "output/trajectory.traj", "local_name": "trajectory.traj", "required": True, "max_bytes": MAX_TRAJECTORY_BYTES, "role": "trajectory"},
        {"remote_name": "trajectory-index.json", "remote_path": "output/trajectory-index.json", "local_name": "trajectory-index.json", "required": True, "max_bytes": MAX_JSON_BYTES, "role": "trajectory-index"},
        {"remote_name": "thermo.csv", "remote_path": "output/thermo.csv", "local_name": "thermo.csv", "required": True, "max_bytes": MAX_THERMO_BYTES, "role": "thermodynamics"},
        {"remote_name": "final.extxyz", "remote_path": "output/final.extxyz", "local_name": "final.extxyz", "required": True, "max_bytes": MAX_FINAL_BYTES, "role": "final-structure"},
        {"remote_name": "cluster-run-report.json", "remote_path": "output/cluster-run-report.json", "local_name": "cluster-run-report.json", "required": True, "max_bytes": MAX_JSON_BYTES, "role": "cluster-run-report"},
        {"remote_name": "md.stdout.log", "remote_path": "logs/md.stdout.log", "local_name": "md.stdout.log", "required": False, "max_bytes": 256 * 1024 * 1024, "role": "md-log"},
        {"remote_name": "md.stderr.log", "remote_path": "logs/md.stderr.log", "local_name": "md.stderr.log", "required": False, "max_bytes": 256 * 1024 * 1024, "role": "md-log"},
    ]
    steps = int(parameters["steps"])
    trajectory_steps = _expected_steps(steps, int(parameters["trajectory_interval"]))
    thermo_steps = _expected_steps(steps, int(parameters["thermo_interval"]))
    identity = {
        "calculator": calculator,
        "ensemble": "nvt-langevin",
        "model_id": model["model_id"],
        "model_fingerprint": model["fingerprint"],
        "structure_fingerprint": structure_fp,
        "temperature_k": float(parameters["temperature_k"]),
        "timestep_fs": float(parameters["timestep_fs"]),
        "steps": steps,
        "trajectory_interval": int(parameters["trajectory_interval"]),
        "thermo_interval": int(parameters["thermo_interval"]),
        "trajectory_steps": trajectory_steps,
        "thermo_steps": thermo_steps,
        "seed": int(parameters["seed"]),
        "device": str(parameters["device"]),
        "default_dtype": str(parameters["default_dtype"]),
        "friction_per_fs": float(parameters["friction_per_fs"]),
        "fix_com": bool(parameters["fix_com"]),
        "input_format": parameters.get("input_format"),
        "input_index": str(parameters.get("input_index", "-1")),
    }
    if (
        "supercell_repeat" in parameters
        or "minimum_initial_cell_length_angstrom" in parameters
    ):
        identity.update(
            {
                "supercell_repeat": list(
                    parameters.get("supercell_repeat", [1, 1, 1])
                ),
                "minimum_initial_cell_length_angstrom": parameters.get(
                    "minimum_initial_cell_length_angstrom"
                ),
            }
        )
    return {
        "plugin_id": PLUGIN_ID,
        "status": "READY",
        "executable": True,
        "argv": [f"template-family:{TEMPLATE_FAMILIES[calculator]}"],
        "cwd": "remote-attempt-workspace",
        "framework": calculator,
        "operation": "run",
        "expected_outputs": [item["remote_name"] for item in fetch_outputs if item["required"]],
        "diagnostics": [],
        "md_identity": identity,
        "input_fingerprints": {
            "project": _sha256(project),
            "structure": structure_fp,
            "model_reference": _sha256(model_reference),
        },
        "approval_summary": {
            "expensive": True,
            "submits_jobs": True,
            "execution_model": "single-python",
            "cpus_meaning": "threads-per-process",
            "calculator": calculator,
            "ensemble": "nvt-langevin",
            "temperature_K": identity["temperature_k"],
            "timestep_fs": identity["timestep_fs"],
            "steps": steps,
            "simulated_time_ps": steps * identity["timestep_fs"] / 1000.0,
            "trajectory_frames": len(trajectory_steps),
            "thermo_records": len(thermo_steps),
            "model_id": model["model_id"],
            "model_fingerprint": model["fingerprint"],
            "structure_fingerprint": structure_fp,
            "supercell_repeat": identity.get("supercell_repeat", [1, 1, 1]),
            "minimum_initial_cell_length_A": identity.get(
                "minimum_initial_cell_length_angstrom"
            ),
            "device": identity["device"],
            "resources": dict(_mapping(context.get("resources"))),
            "template_family": TEMPLATE_FAMILIES[calculator],
        },
        "scheduled_execution": {
            "schema_version": 3,
            "execution_model": "single-python",
            "template_family": TEMPLATE_FAMILIES[calculator],
            "staged_files": staged,
            "fetch_outputs": fetch_outputs,
        },
    }


def _scheduled_plan(context: dict[str, Any]) -> dict[str, Any]:
    return _mapping(_mapping(context.get("execution")).get("plan"))


def _read_bounded_json(path: Path) -> dict[str, Any]:
    return _read_json(path)


def _artifact_records(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("md-result artifacts must be a list")
    records: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict) or not _plain_string(item.get("name")):
            raise ValueError("each md-result artifact must have a name")
        name = str(item["name"])
        if name in records:
            raise ValueError(f"duplicate md-result artifact: {name}")
        records[name] = item
    return records


def _check_artifact(path: Path, record: dict[str, Any], max_bytes: int) -> str | None:
    if not _ordinary_file(path, max_bytes):
        return f"missing, empty, unsafe or oversized artifact: {path.name}"
    if record.get("path") != path.name:
        return f"artifact record path mismatch for {path.name}"
    if record.get("sha256") != _sha256(path):
        return f"artifact SHA-256 mismatch for {path.name}"
    if record.get("size_bytes") != path.stat().st_size:
        return f"artifact size mismatch for {path.name}"
    return None


def _check_trajectory_index(path: Path, identity: dict[str, Any]) -> str | None:
    try:
        raw = _read_bounded_json(path)
    except Exception as exc:
        return f"trajectory-index.json is unreadable: {exc}"
    if raw.get("schema_version") != 1:
        return "trajectory-index.json schema_version must be 1"
    expected = identity.get("trajectory_steps")
    if raw.get("steps") != expected:
        return "trajectory-index.json step schedule differs from the approved plan"
    times = raw.get("time_fs")
    if not isinstance(times, list) or not isinstance(expected, list) or len(times) != len(expected):
        return "trajectory-index.json time vector has the wrong length"
    dt = float(identity["timestep_fs"])
    for step, value in zip(expected, times):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not math.isclose(float(value), float(step) * dt, rel_tol=1e-12, abs_tol=1e-9):
            return "trajectory-index.json contains a time inconsistent with timestep_fs"
    return None


def _check_thermo(path: Path, identity: dict[str, Any]) -> tuple[str | None, dict[str, float] | None]:
    if not _ordinary_file(path, MAX_THERMO_BYTES):
        return "thermo.csv is missing, empty, unsafe or oversized", None
    expected_header = ["step", "time_fs", "temperature_K", "potential_energy_eV", "kinetic_energy_eV", "total_energy_eV", "volume_A3"]
    expected_steps = identity.get("thermo_steps")
    if not isinstance(expected_steps, list):
        return "approved plan lacks thermo step schedule", None
    seen: list[int] = []
    last: dict[str, float] | None = None
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != expected_header:
                return "thermo.csv header differs from the contract", None
            for index, row in enumerate(reader):
                if index >= MAX_RECORDS:
                    return "thermo.csv exceeds the bounded record count", None
                try:
                    step = int(row["step"])
                    numeric = {key: float(row[key]) for key in expected_header[1:]}
                except (TypeError, ValueError, KeyError) as exc:
                    return f"thermo.csv contains an invalid numeric row: {exc}", None
                if any(not math.isfinite(value) for value in numeric.values()):
                    return "thermo.csv contains a non-finite value", None
                if not math.isclose(numeric["time_fs"], step * float(identity["timestep_fs"]), rel_tol=1e-12, abs_tol=1e-9):
                    return "thermo.csv time_fs is inconsistent with timestep_fs", None
                seen.append(step)
                last = numeric
    except (OSError, UnicodeError) as exc:
        return f"thermo.csv is unreadable: {exc}", None
    if seen != expected_steps:
        return "thermo.csv step schedule differs from the approved plan", None
    return None, last


def _check_structure_summary(
    result: dict[str, Any], identity: dict[str, Any]
) -> list[dict[str, str]]:
    if "supercell_repeat" not in identity:
        return []
    diagnostics: list[dict[str, str]] = []
    repeat = identity["supercell_repeat"]
    source_atoms = result.get("source_atom_count")
    atom_count = result.get("atom_count")
    lengths = result.get("initial_cell_lengths_A")
    minimum = identity.get("minimum_initial_cell_length_angstrom")
    expected = {
        "supercell_repeat": repeat,
        "minimum_initial_cell_length_A": minimum,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            diagnostics.append(
                _diagnostic(
                    "error",
                    f"ase_md.result_{key}",
                    f"md-result field {key} differs",
                )
            )
    if (
        isinstance(source_atoms, bool)
        or not isinstance(source_atoms, int)
        or source_atoms < 1
        or isinstance(atom_count, bool)
        or not isinstance(atom_count, int)
        or atom_count != source_atoms * math.prod(repeat)
    ):
        diagnostics.append(
            _diagnostic(
                "error",
                "ase_md.atom_count",
                "expanded atom count differs from the approved repeat",
            )
        )
    if (
        not isinstance(lengths, list)
        or len(lengths) != 3
        or any(not _finite_positive(value) for value in lengths)
        or (
            minimum is not None
            and any(float(value) <= float(minimum) for value in lengths)
        )
    ):
        diagnostics.append(
            _diagnostic(
                "error",
                "ase_md.initial_cell",
                "expanded initial cell does not satisfy the approved bound",
            )
        )
    stability = _mapping(result.get("observed_stability"))
    distance = stability.get("minimum_pair_distance_A")
    if stability.get("all_recorded_values_finite") is not True or (
        distance is not None and not _finite_positive(distance)
    ):
        diagnostics.append(
            _diagnostic(
                "error", "ase_md.stability_summary", "MD stability summary is invalid"
            )
        )
    thermodynamics = _mapping(stability.get("thermodynamics"))
    for name in (
        "temperature_K",
        "potential_energy_eV_per_atom",
        "total_energy_eV_per_atom",
        "volume_A3",
    ):
        stats = _mapping(thermodynamics.get(name))
        if any(
            not isinstance(stats.get(key), (int, float))
            or not math.isfinite(float(stats[key]))
            for key in ("minimum", "maximum", "mean")
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "ase_md.stability_summary",
                    f"stability statistics are invalid for {name}",
                )
            )
    return diagnostics


def _check(context: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    diagnostics: list[dict[str, str]] = []
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    identity = _mapping(_scheduled_plan(context).get("md_identity"))
    if not identity:
        return [_diagnostic("error", "ase_md.plan_identity", "pinned plan lacks md_identity")], None
    try:
        result = _read_bounded_json(attempt / "md-result.json")
    except Exception as exc:
        return [_diagnostic("error", "ase_md.result", f"md-result.json is unreadable: {exc}")], None
    expected_pairs = {
        "schema_version": 1,
        "plugin_id": PLUGIN_ID,
        "status": "OK",
        "calculator": identity.get("calculator"),
        "ensemble": "nvt-langevin",
        "structure_fingerprint": identity.get("structure_fingerprint"),
        "temperature_K": identity.get("temperature_k"),
        "timestep_fs": identity.get("timestep_fs"),
        "steps_requested": identity.get("steps"),
        "steps_completed": identity.get("steps"),
        "trajectory_interval": identity.get("trajectory_interval"),
        "thermo_interval": identity.get("thermo_interval"),
        "trajectory_frames": len(identity.get("trajectory_steps", [])),
        "thermo_records": len(identity.get("thermo_steps", [])),
        "seed": identity.get("seed"),
        "device": identity.get("device"),
        "default_dtype": identity.get("default_dtype"),
        "friction_per_fs": identity.get("friction_per_fs"),
        "fix_com": identity.get("fix_com"),
    }
    for key, expected in expected_pairs.items():
        if result.get(key) != expected:
            diagnostics.append(_diagnostic("error", f"ase_md.result_{key}", f"md-result.json field {key} differs from the approved plan"))
    diagnostics.extend(_check_structure_summary(result, identity))
    model = _mapping(result.get("model"))
    if model.get("id") != identity.get("model_id") or model.get("fingerprint") != identity.get("model_fingerprint"):
        diagnostics.append(_diagnostic("error", "ase_md.result_model", "md-result model identity differs from the approved plan"))
    if not _plain_string(result.get("calculator_version")) or not _plain_string(result.get("ase_version")):
        diagnostics.append(_diagnostic("error", "ase_md.versions", "md-result must record calculator and ASE versions"))
    try:
        artifacts = _artifact_records(result.get("artifacts"))
    except ValueError as exc:
        diagnostics.append(_diagnostic("error", "ase_md.artifacts", str(exc)))
        artifacts = {}
    expected_artifacts = {
        "trajectory": ("trajectory.traj", MAX_TRAJECTORY_BYTES),
        "trajectory-index": ("trajectory-index.json", MAX_JSON_BYTES),
        "thermo": ("thermo.csv", MAX_THERMO_BYTES),
        "final-structure": ("final.extxyz", MAX_FINAL_BYTES),
    }
    if set(artifacts) != set(expected_artifacts):
        diagnostics.append(_diagnostic("error", "ase_md.artifact_set", "md-result artifact set is incomplete or unexpected"))
    for role, (name, limit) in expected_artifacts.items():
        if role in artifacts:
            error = _check_artifact(attempt / name, artifacts[role], limit)
            if error:
                diagnostics.append(_diagnostic("error", f"ase_md.artifact_{role}", error))
    index_error = _check_trajectory_index(attempt / "trajectory-index.json", identity)
    if index_error:
        diagnostics.append(_diagnostic("error", "ase_md.trajectory_index", index_error))
    thermo_error, final_thermo = _check_thermo(attempt / "thermo.csv", identity)
    if thermo_error:
        diagnostics.append(_diagnostic("error", "ase_md.thermo", thermo_error))
    try:
        report = _read_bounded_json(attempt / "cluster-run-report.json")
    except Exception as exc:
        diagnostics.append(_diagnostic("error", "ase_md.cluster_report", f"cluster-run-report.json is unreadable: {exc}"))
        report = {}
    if report.get("status") != "OK" or report.get("calculator") != identity.get("calculator") or report.get("ensemble") != "nvt-langevin":
        diagnostics.append(_diagnostic("error", "ase_md.cluster_identity", "cluster report does not describe the approved successful ASE MD run"))
    report_model = _mapping(report.get("model"))
    if report_model.get("id") != identity.get("model_id") or report_model.get("observed_fingerprint") != identity.get("model_fingerprint"):
        diagnostics.append(_diagnostic("error", "ase_md.cluster_model", "cluster report model identity differs from the approved model"))
    if report.get("structure_fingerprint") != identity.get("structure_fingerprint"):
        diagnostics.append(_diagnostic("error", "ase_md.cluster_structure", "cluster report structure fingerprint differs from the approved structure"))
    if "supercell_repeat" in identity:
        for key in (
            "supercell_repeat",
            "source_atom_count",
            "atom_count",
            "initial_cell_lengths_A",
            "minimum_initial_cell_length_A",
            "observed_stability",
        ):
            if report.get(key) != result.get(key):
                diagnostics.append(
                    _diagnostic(
                        "error",
                        f"ase_md.cluster_{key}",
                        f"cluster report {key} differs",
                    )
                )
    if _ordinary_file(attempt / "md-result.json", MAX_JSON_BYTES) and report.get("result_sha256") != _sha256(attempt / "md-result.json"):
        diagnostics.append(_diagnostic("error", "ase_md.cluster_result_hash", "cluster report does not bind the fetched md-result.json"))
    if diagnostics:
        return diagnostics, None
    return diagnostics, {"result": result, "final_thermo": final_thermo or {}}


class Adapter:
    def validate(self, context: dict[str, Any]) -> list[dict[str, str]]:
        return _validate(context)

    def plan(self, context: dict[str, Any]) -> dict[str, Any]:
        return _plan(context)

    def prepare(self, context: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(plan, dict) or plan.get("status") != "READY":
            return _blocked([_diagnostic("error", "ase_md.plan_required", "a READY plan is required")])
        return {
            "plugin_id": PLUGIN_ID,
            "status": "READY",
            "executable": False,
            "prepared": False,
            "message": "No project files were mutated; the scheduler stages the pinned structure, model reference, and bundled ASE MD runners.",
        }

    def check(self, context: dict[str, Any]) -> dict[str, Any]:
        diagnostics, analysis = _check(context)
        if diagnostics or analysis is None:
            return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
        final = _mapping(analysis.get("final_thermo"))
        return {
            "plugin_id": PLUGIN_ID,
            "status": "OK",
            "diagnostics": [],
            "metrics": {
                "steps_completed": float(_mapping(analysis["result"]).get("steps_completed", 0)),
                "final_temperature_K": float(final.get("temperature_K", 0.0)),
                "final_total_energy_eV": float(final.get("total_energy_eV", 0.0)),
            },
            "artifacts": [],
        }

    def collect(self, context: dict[str, Any]) -> dict[str, Any]:
        checked = self.check(context)
        if checked.get("status") != "OK":
            return checked
        attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
        roles = (
            ("md-result.json", "md-result", "application/json"),
            ("trajectory.traj", "trajectory", "application/octet-stream"),
            ("trajectory-index.json", "trajectory-index", "application/json"),
            ("thermo.csv", "thermodynamics", "text/csv"),
            ("final.extxyz", "final-structure", "chemical/x-xyz"),
            ("cluster-run-report.json", "cluster-run-report", "application/json"),
        )
        artifacts = [
            {"path": str(attempt / name), "role": role, "media_type": media}
            for name, role, media in roles
            if _ordinary_file(attempt / name)
        ]
        return {
            "plugin_id": PLUGIN_ID,
            "status": "OK",
            "diagnostics": [],
            "metrics": checked.get("metrics", {}),
            "artifacts": artifacts,
        }

    def replay(self, context: dict[str, Any]) -> dict[str, Any]:
        return self.collect(context)
