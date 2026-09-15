"""pes sampling: contracts."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

PLUGIN_ID = "pes-sampling"

DIRECT_OPERATION = "direct-select"

MERGE_OPERATION = "merge-structures"

LASP_INPUT_OPERATION = "lasp-input-prepare"

LASP_OPERATIONS = frozenset({"lasp-ssw-execute", "lasp-ssw-normalize-replay"})

OPERATIONS = frozenset({DIRECT_OPERATION, MERGE_OPERATION, LASP_INPUT_OPERATION}) | LASP_OPERATIONS

UNKNOWN = "HISTORICAL_PARAMETER_UNKNOWN"

BUNDLED_DIRECT_WRAPPER = (
    Path(__file__).absolute().with_name("direct_select.py")
)

BUNDLED_LASP_WRAPPER = (
    Path(__file__).absolute().with_name("lasp_ssw.py")
)

BUNDLED_MERGE_WRAPPER = (
    Path(__file__).absolute().with_name("structure_merge.py")
)

BUNDLED_LASP_INPUT_WRAPPER = (
    Path(__file__).absolute().with_name("structure_to_lasp.py")
)

MAX_LASP_FRAMES = 10000

ARC_HEADER = b"!BIOSYM archive 2\nPBC=ON\n"

SAFE_POTCAR_SYMBOL = re.compile(r"[A-Za-z][A-Za-z0-9_]*")

LASP_ARTIFACT_ROLES = frozenset(
    {
        "ssw-generated-structure",
        "selected-structure",
        "ssw-structure-manifest",
        "selected-structure-manifest",
        "aimd-seed-candidate",
        "aimd-seed-manifest",
        "md-sampled-structure",
        "md-structure-manifest",
        "lasp-run-metadata",
        "lasp-stdout",
        "lasp-stderr",
        "lasp-native-log",
    }
)

_DIRECT_COLUMNS = {
    "selected_order",
    "input_index",
    "source_kind",
    "source_file",
    "frame_index",
    "output_file",
    "formula",
}

_OPTIONAL_PLOTS = (
    "direct_pca_coverage.html",
    "direct_pca_explained_variance.html",
    "direct_feature_coverage_score.html",
    "direct_pca_coverage.png",
    "direct_pca_explained_variance.png",
    "direct_feature_coverage_score.png",
)

_DIRECT_PARAMETERS = {
    "operation",
    "output_subdir",
    "globs",
    "recursive",
    "stride",
    "max_frames_per_file",
    "max_input_structures",
    "n_clusters",
    "threshold_init",
    "k_per_cluster",
    "lammps_type_map",
    "seed",
    "acknowledge_uncontrolled_seed",
}

_LASP_PARAMETERS = {
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

_MERGE_PARAMETERS = {
    "operation",
    "output_subdir",
    "direct_source_group_id",
    "lasp_source_group_id",
    "matcher_ltol",
    "matcher_stol",
    "matcher_angle_tol_deg",
    "minimum_distance_angstrom",
    "max_structures",
}

_LASP_INPUT_PARAMETERS = {
    "operation",
    "output_subdir",
    "input_format",
    "input_index",
    "minimum_cell_length_angstrom",
}


def _diagnostic(level: str, code: str, message: str) -> Dict[str, str]:
    return {"level": level, "code": code, "message": message}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _resolve(value: Any, base: Path) -> Optional[Path]:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _unresolved(value: Any, base: Path) -> Optional[Path]:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).absolute()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _safe_subdirectory(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    path = Path(value)
    return not path.is_absolute() and path != Path(".") and ".." not in path.parts


def _has_errors(diagnostics: List[Dict[str, str]]) -> bool:
    return any(item["level"] == "ERROR" for item in diagnostics)


def _blocked(diagnostics: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "plugin_id": PLUGIN_ID,
        "status": "BLOCKED",
        "executable": False,
        "diagnostics": diagnostics,
    }


def _operation(context: Any) -> str:
    if not isinstance(context, Mapping):
        return DIRECT_OPERATION
    return str(_mapping(context.get("parameters")).get("operation", DIRECT_OPERATION))


def _plain_string(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and "\x00" not in value
        and "\n" not in value
        and "\r" not in value
    )


def _portable_source_id(value: Any) -> bool:
    if not _plain_string(value):
        return False
    normalized = str(value).replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized):
        return False
    return "://" in normalized or ".." not in Path(normalized).parts


def _pseudopotential_reference(path: Path) -> Dict[str, Any]:
    if not _ordinary_file(path):
        raise ValueError("pseudopotential reference must be a JSON file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("pseudopotential reference must contain an object")
    allowed = {
        "schema_version",
        "reference_id",
        "source_env",
        "license_acknowledged",
        "functional",
        "symbols",
    }
    if set(value) - allowed:
        raise ValueError("pseudopotential reference contains unsupported fields")
    if value.get("schema_version") != 1 or value.get("source_env") != "PMG_VASP_PSP_DIR":
        raise ValueError("pseudopotential reference requires schema_version=1 and PMG_VASP_PSP_DIR")
    if value.get("license_acknowledged") is not True:
        raise ValueError("pseudopotential license acknowledgement is required")
    reference_id = value.get("reference_id")
    if not _portable_source_id(reference_id):
        raise ValueError("pseudopotential reference_id must be portable")
    functional = value.get("functional")
    if not _plain_string(functional):
        raise ValueError("pseudopotential functional must be explicit")
    raw_symbols = value.get("symbols")
    if not isinstance(raw_symbols, Mapping) or not raw_symbols:
        raise ValueError("pseudopotential symbols must be an explicit mapping")
    symbols: Dict[str, str] = {}
    for element, symbol in raw_symbols.items():
        if (
            not isinstance(element, str)
            or re.fullmatch(r"[A-Z][a-z]?", element) is None
            or not isinstance(symbol, str)
            or SAFE_POTCAR_SYMBOL.fullmatch(symbol) is None
        ):
            raise ValueError("pseudopotential symbols mapping is invalid")
        symbols[element] = symbol
    return {
        "reference_id": reference_id,
        "functional": functional,
        "symbols": symbols,
    }


def _lasp_structure_id(role: str, frame_index: int) -> str:
    return f"lasp-{role}-{frame_index:06d}"


def _ordinary_file(path: Optional[Path]) -> bool:
    return path is not None and path.is_file() and path.stat().st_size > 0


def _explicit_executable(value: Any, base: Path) -> Optional[Path]:
    """Resolve an absolute executable without consulting ``PATH``."""
    if not _plain_string(value):
        return None
    unresolved = Path(str(value)).expanduser()
    if not unresolved.is_absolute():
        return None
    path = _resolve(unresolved, base)
    if not _ordinary_file(path):
        return None
    assert path is not None
    return path if path.stat().st_mode & 0o111 else None


def _parse_energy_record(line: bytes, source: str, frame_index: int) -> float:
    try:
        tokens = line.decode("ascii").split()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source} frame {frame_index} Energy record is not ASCII") from exc
    if tokens[:1] != ["Energy"] or len(tokens) not in {3, 4, 5}:
        raise ValueError(f"{source} frame {frame_index} has an unsupported Energy record")
    try:
        int(tokens[1])
    except ValueError as exc:
        raise ValueError(f"{source} frame {frame_index} Energy index is invalid") from exc
    if len(tokens) == 5 and not re.fullmatch(r"C[0-9]+", tokens[4]):
        raise ValueError(f"{source} frame {frame_index} Energy suffix is invalid")
    energy_token = tokens[2] if len(tokens) == 3 else tokens[3]
    try:
        energy = float(energy_token)
    except ValueError as exc:
        raise ValueError(f"{source} frame {frame_index} energy is invalid") from exc
    if not math.isfinite(energy):
        raise ValueError(f"{source} frame {frame_index} energy is not finite")
    return energy


def _read_arc_frames(path: Path, max_frames: int) -> List[Dict[str, Any]]:
    if path.stat().st_size == 0:
        raise ValueError(f"{path.name} is empty")
    lines = path.read_bytes().splitlines(keepends=True)
    starts = [index for index, line in enumerate(lines) if line.lstrip().startswith(b"Energy")]
    if not starts or len(starts) > max_frames:
        raise ValueError(f"{path.name} frame count is zero or exceeds max_frames")
    frames: List[Dict[str, Any]] = []
    for position, start in enumerate(starts):
        stop = starts[position + 1] if position + 1 < len(starts) else len(lines)
        block = lines[start:stop]
        payload = ARC_HEADER + b"".join(block)
        if sum(1 for line in block if line.strip() == b"end") < 2:
            raise ValueError(f"{path.name} frame {position + 1} is truncated")
        frames.append(
            {
                "frame_index": position + 1,
                "energy_ev": _parse_energy_record(block[0], path.name, position + 1),
                "payload": payload,
            }
        )
    return frames


def _parse_lasp_input(path: Path) -> Dict[str, Any]:
    if path.stat().st_size == 0:
        raise ValueError("lasp.in is empty")
    parameters: Dict[str, Any] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw_line.split("#", 1)[0].strip()
        if not stripped or stripped.startswith(("#", "!")):
            continue
        tokens = stripped.split()
        if len(tokens) < 2:
            raise ValueError(f"lasp.in line {line_number} has no explicit value")
        key = tokens[0]
        value: Any = tokens[1] if len(tokens) == 2 else tokens[1:]
        if key in parameters:
            previous = parameters[key]
            parameters[key] = (
                previous + [value] if isinstance(previous, list) else [previous, value]
            )
        else:
            parameters[key] = value
    if not parameters:
        raise ValueError("lasp.in contains no parameters")
    return parameters


def _validate_ssw_input(parameters: Mapping[str, Any]) -> None:
    if str(parameters.get("explore_type", "")).lower() != "ssw":
        raise ValueError("lasp.in must explicitly declare explore_type ssw")
    steps = parameters.get("SSW.SSWsteps")
    try:
        numeric_steps = int(str(steps))
    except (TypeError, ValueError) as exc:
        raise ValueError("lasp.in must declare an integer SSW.SSWsteps") from exc
    if numeric_steps < 1:
        raise ValueError("lasp-ssw operations require SSW.SSWsteps >= 1")


def _manifest_path(context: Any) -> Optional[Path]:
    if not isinstance(context, Mapping):
        return None
    inputs = _mapping(context.get("inputs"))
    project_root = _resolve(context.get("project_root"), Path.cwd()) or Path.cwd().resolve()
    execution = _mapping(context.get("execution"))
    plan = _mapping(execution.get("plan"))
    expected = plan.get("expected_outputs")
    if isinstance(expected, list) and len(expected) == 1:
        return _unresolved(expected[0], project_root)
    explicit = inputs.get("result_manifest")
    if explicit is not None:
        return _unresolved(explicit, project_root)
    return None


def _read_json_object(
    path: Path, field: str, diagnostics: List[Dict[str, str]]
) -> Optional[Dict[str, Any]]:
    try:
        if path.stat().st_size == 0:
            raise ValueError("file is empty")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("top-level JSON value must be an object")
        return value
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        diagnostics.append(
            _diagnostic("ERROR", f"result.{field}_unreadable", f"cannot read {field}: {exc}")
        )
        return None


def _artifact_path(
    manifest: Path,
    value: Any,
    field: str,
    diagnostics: List[Dict[str, str]],
) -> Optional[Path]:
    if not isinstance(value, str) or not value:
        diagnostics.append(
            _diagnostic("ERROR", "result.artifact_path", f"{field} requires a path")
        )
        return None
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        diagnostics.append(
            _diagnostic("ERROR", "result.artifact_escape", f"{field} must be a relative path")
        )
        return None
    path = manifest.parent / relative
    resolved = path.resolve()
    if not _is_within(resolved, manifest.parent.resolve()):
        diagnostics.append(
            _diagnostic("ERROR", "result.artifact_escape", f"{field} escapes result directory")
        )
        return None
    if not resolved.is_file() or resolved.stat().st_size == 0:
        diagnostics.append(
            _diagnostic(
                "ERROR",
                "result.artifact_missing",
                f"{field} is missing or empty",
            )
        )
        return None
    return resolved
