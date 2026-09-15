"""dft labeling: contracts."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

PLUGIN_ID = "dft-labeling"

PREPARE_OPERATION = "vasp-prepare"

LABEL_OPERATION = "label"

DATASET_OPERATION = "dataset-assemble"

OPERATIONS = frozenset({PREPARE_OPERATION, LABEL_OPERATION, DATASET_OPERATION})

BUNDLED_PREPARE_WRAPPER = (
    Path(__file__).absolute().with_name("vasp_prepare.py")
)

DATASET_CONTRACT_PATH = (
    Path(__file__).absolute().with_name("dataset_contract.py")
)

DATASET_CONVERTER_PATH = (
    Path(__file__).absolute().with_name("dataset_convert.py")
)

SHELL_EXECUTABLES = frozenset(
    {"bash", "csh", "cmd", "dash", "fish", "ksh", "powershell", "pwsh", "sh", "tcsh", "zsh"}
)

SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

SAFE_POTCAR_SYMBOL = re.compile(r"[A-Za-z][A-Za-z0-9_]*")

MAX_STRUCTURES = 10000

DATASET_FRAMEWORKS = ("deepmd", "m3gnet", "chgnet", "mace")

MANUSCRIPT_STATIC_PRESET = "manuscript-static-v1"

MANUSCRIPT_STATIC_INCAR: dict[str, Any] = {
    "ISTART": 0,
    "ICHARG": 2,
    "LCHARG": False,
    "LWAVE": False,
    "LREAL": "Auto",
    "IALGO": 38,
    "EDIFF": 5e-6,
    "ISMEAR": 0,
    "SIGMA": 0.1,
    "PREC": "Normal",
    "NELM": 700,
    "NELMIN": 4,
    "ENCUT": 450,
    "IVDW": 12,
    "NSW": 0,
    "IBRION": -1,
    "ISIF": 2,
    "ISPIN": 2,
}

EXPECTED_FILE_NAMES = frozenset({"POSCAR", "INCAR", "KPOINTS", "POTCAR"})


def _diagnostic(level: str, code: str, message: str) -> dict[str, str]:
    return {"level": level, "code": code, "message": message}


def _errors(diagnostics: list[dict[str, str]]) -> list[dict[str, str]]:
    return [item for item in diagnostics if item["level"] == "error"]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _plain_string(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and "\x00" not in value
        and "\n" not in value
        and "\r" not in value
    )


def _safe_relative(value: Any) -> bool:
    if not _plain_string(value):
        return False
    path = Path(str(value))
    return path != Path(".") and not path.is_absolute() and ".." not in path.parts


def _path(root: Any, relative: Any) -> Path:
    return Path(str(root)).expanduser().absolute() / str(relative)


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _ordinary_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _ordinary_project_file(path: Path, project_root: Path) -> bool:
    try:
        path.resolve().relative_to(project_root.resolve())
    except (OSError, ValueError):
        return False
    return _ordinary_file(path)


def _project_input_path(project_root: Any, value: Any) -> Path | None:
    if not _plain_string(value):
        return None
    raw = Path(str(value)).expanduser()
    if raw.is_absolute():
        return raw.absolute()
    if not _safe_relative(value):
        return None
    return _path(project_root, value)


def _structure_input_path(manifest_path: Path, value: Any) -> Path | None:
    if not _plain_string(value):
        return None
    raw = Path(str(value)).expanduser()
    source = raw if raw.is_absolute() else manifest_path.parent / raw
    return source.resolve()


def _readable_structure_file(path: Path) -> bool:
    if not _ordinary_file(path):
        return False
    try:
        with path.open("rb") as handle:
            handle.read(1)
    except OSError:
        return False
    return True


def _explicit_executable(value: Any) -> Path | None:
    if not _plain_string(value):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute() or not _ordinary_file(path):
        return None
    resolved = path.resolve()
    return resolved if resolved.stat().st_mode & 0o111 else None


def _read_json(path: Path) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    if not _ordinary_file(path):
        return None, _diagnostic("warning", "artifact.missing", f"清单不存在或为空：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, _diagnostic("error", "artifact.invalid_json", f"无法读取 JSON：{exc}")
    if not isinstance(value, dict):
        return None, _diagnostic("error", "artifact.not_object", "清单必须是 JSON 对象。")
    return value, None


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _operation(context: Any) -> str:
    if not isinstance(context, Mapping):
        return ""
    return str(_mapping(context.get("parameters")).get("operation", ""))


def _base_diagnostics(
    context: Any, allowed_backends: frozenset[str] = frozenset({"local"})
) -> list[dict[str, str]]:
    if not isinstance(context, Mapping):
        return [_diagnostic("error", "context.type", "context 必须是对象。")]
    diagnostics: list[dict[str, str]] = []
    required = {"project_root", "attempt_dir", "inputs", "parameters", "backend", "resources"}
    missing = sorted(required - set(context))
    if missing:
        diagnostics.append(_diagnostic("error", "context.keys", "缺少字段：" + ", ".join(missing)))
    project_root = Path(str(context.get("project_root", ""))).expanduser().absolute()
    if not project_root.is_dir():
        diagnostics.append(_diagnostic("error", "path.project_root", "project_root 必须存在。"))
    attempt_dir = Path(str(context.get("attempt_dir", ""))).expanduser().absolute()
    if attempt_dir.exists() and not attempt_dir.is_dir():
        diagnostics.append(_diagnostic("error", "path.attempt_dir", "attempt_dir 必须是目录。"))
    elif project_root.is_dir() and not _within(attempt_dir.resolve(), project_root.resolve()):
        diagnostics.append(
            _diagnostic("error", "path.attempt_escape", "attempt_dir 必须位于 project_root 内。")
        )
    for key in ("inputs", "parameters", "resources"):
        if not isinstance(context.get(key), Mapping):
            diagnostics.append(_diagnostic("error", f"context.{key}", f"{key} 必须是对象。"))
    if context.get("backend") not in allowed_backends:
        diagnostics.append(
            _diagnostic(
                "error",
                "backend.unsupported",
                "该 operation 不支持所选 backend；调度提交只能由核心托管。",
            )
        )
    operation = _operation(context)
    if operation not in OPERATIONS:
        diagnostics.append(
            _diagnostic(
                "error",
                "parameters.operation",
                "operation 必须是 vasp-prepare、label 或 dataset-assemble。",
            )
        )
    return diagnostics


def _validate_input_path(
    diagnostics: list[dict[str, str]], context: Mapping[str, Any], name: str, required: bool = True
) -> Path | None:
    inputs = _mapping(context.get("inputs"))
    value = inputs.get(name)
    if value is None and not required:
        return None
    path = _project_input_path(context.get("project_root"), value)
    if path is None:
        diagnostics.append(
            _diagnostic(
                "error",
                f"inputs.{name}",
                f"{name} 必须是 project_root 下的安全相对或绝对路径。",
            )
        )
    return path


def _parse_labeling_contract(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    value, error = _read_json(path)
    if value is None:
        return None, error["message"] if error else "labeling_config 无效"
    allowed = {
        "schema_version",
        "engine",
        "calculation_type",
        "preset",
        "incar",
        "kpoints",
        "sort_structure",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        return None, "labeling_config 含不支持字段：" + ", ".join(unknown)
    if value.get("schema_version") != 1 or value.get("engine") != "vasp":
        return None, "labeling_config 必须声明 schema_version=1、engine=vasp"
    calculation_type = value.get("calculation_type")
    if calculation_type not in {"static", "relax", "aimd"}:
        return None, "calculation_type 必须是 static、relax 或 aimd"
    preset = value.get("preset")
    if preset not in {None, MANUSCRIPT_STATIC_PRESET}:
        return None, "preset 不受支持"
    if preset == MANUSCRIPT_STATIC_PRESET and calculation_type != "static":
        return None, "manuscript-static-v1 仅适用于 static"
    incar = value.get("incar", {})
    if not isinstance(incar, Mapping):
        return None, "incar 必须是对象"
    effective = dict(MANUSCRIPT_STATIC_INCAR) if preset else {}
    normalized_keys: set[str] = set()
    for key, item in incar.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key):
            return None, "incar 含无效键"
        normalized_key = key.upper()
        if normalized_key in normalized_keys:
            return None, f"incar 含重复的规范化键：{normalized_key}"
        normalized_keys.add(normalized_key)
        if not _valid_incar_value(item):
            return None, f"INCAR {normalized_key} 必须是有限 JSON scalar 或非空 flat list"
        effective[normalized_key] = item
    if not effective:
        return None, "未使用 preset 时必须显式给出 incar"
    if calculation_type == "static" and (
        effective.get("NSW") != 0 or effective.get("IBRION") != -1
    ):
        return None, "static 必须满足 NSW=0、IBRION=-1"
    if calculation_type == "relax":
        if not _positive_int(effective.get("NSW")) or effective.get("IBRION") not in {1, 2, 3}:
            return None, "relax 必须有正整数 NSW 且 IBRION 为 1、2 或 3"
        if "EDIFFG" not in effective or "ISIF" not in effective:
            return None, "relax 必须显式给出 EDIFFG 和 ISIF"
    if calculation_type == "aimd":
        if effective.get("IBRION") != 0 or not _positive_number(effective.get("NSW")):
            return None, "aimd 必须满足 IBRION=0 且 NSW>0"
        if not _positive_number(effective.get("POTIM")):
            return None, "aimd 必须显式给出 POTIM>0"
        if any(key not in effective for key in ("TEBEG", "TEEND", "MDALGO")):
            return None, "aimd 必须显式给出 TEBEG、TEEND、MDALGO"
    kpoints = value.get("kpoints")
    if kpoints is None and preset:
        kpoints = {"mode": "monkhorst", "grid": [1, 1, 1], "shift": [0, 0, 0]}
    if not isinstance(kpoints, Mapping):
        return None, "kpoints 必须显式声明"
    unknown_kpoints = sorted(set(kpoints) - {"mode", "grid", "shift"})
    if unknown_kpoints:
        return None, "kpoints 含不支持字段：" + ", ".join(unknown_kpoints)
    mode = kpoints.get("mode")
    grid = kpoints.get("grid")
    shift = kpoints.get("shift", [0, 0, 0])
    if mode not in {"gamma", "monkhorst"}:
        return None, "kpoints.mode 必须是 gamma 或 monkhorst"
    if not isinstance(grid, list) or len(grid) != 3 or any(not _positive_int(x) for x in grid):
        return None, "kpoints.grid 必须是三个正整数"
    if not isinstance(shift, list) or len(shift) != 3 or any(not _finite_number(x) for x in shift):
        return None, "kpoints.shift 必须是三个有限数"
    if not isinstance(value.get("sort_structure", True), bool):
        return None, "sort_structure 必须是布尔值"
    return {
        "calculation_type": calculation_type,
        "preset": preset,
        "effective_incar": effective,
        "kpoints": {"mode": mode, "grid": grid, "shift": shift},
        "sort_structure": value.get("sort_structure", True),
    }, None


def _positive_number(value: Any) -> bool:
    return _finite_number(value) and float(value) > 0


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _valid_incar_value(value: Any) -> bool:
    if value is None or isinstance(value, Mapping):
        return False
    if isinstance(value, list):
        return bool(value) and all(
            not isinstance(item, (list, Mapping)) and _valid_incar_value(item) for item in value
        )
    if isinstance(value, float):
        return math.isfinite(value)
    return isinstance(value, (str, int, bool))


def _parse_pseudopotential_reference(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    value, error = _read_json(path)
    if value is None:
        return None, error["message"] if error else "pseudopotential_reference 无效"
    allowed = {
        "schema_version",
        "reference_id",
        "source_env",
        "license_acknowledged",
        "functional",
        "symbols",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        return None, "赝势引用含不支持字段：" + ", ".join(unknown)
    if value.get("schema_version") != 1 or value.get("source_env") != "PMG_VASP_PSP_DIR":
        return None, "赝势引用必须声明 schema_version=1、source_env=PMG_VASP_PSP_DIR"
    if value.get("license_acknowledged") is not True:
        return None, "必须显式确认赝势许可"
    reference_id = value.get("reference_id")
    normalized_reference_id = str(reference_id).replace("://", "/")
    if (
        not _plain_string(reference_id)
        or str(reference_id).startswith("/")
        or ".." in Path(normalized_reference_id).parts
    ):
        return None, "reference_id 必须可移植且不得是绝对路径"
    if not _plain_string(value.get("functional")):
        return None, "functional 必须显式声明"
    symbols = value.get("symbols")
    if not isinstance(symbols, Mapping) or not symbols:
        return None, "symbols 必须显式映射元素到 POTCAR symbol"
    for element, symbol in symbols.items():
        if not isinstance(element, str) or not re.fullmatch(r"[A-Z][a-z]?", element):
            return None, "symbols 含无效元素"
        if not isinstance(symbol, str) or not SAFE_POTCAR_SYMBOL.fullmatch(symbol):
            return None, "symbols 含无效 POTCAR symbol"
    return value, None


def _file_paths(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: str(path) for name, path in paths.items()}


def _read_structure_count(path: Path) -> int | str:
    manifest, _ = _read_json(path)
    structures = manifest.get("structures") if isinstance(manifest, Mapping) else None
    return len(structures) if isinstance(structures, list) else "UNKNOWN"


def _staged_record(source: Path, remote_name: str, *, sensitive: bool = False) -> dict[str, Any]:
    return {"source": str(source.absolute()), "remote_name": remote_name}


def _attempt_index(path: Path) -> int | None:
    match = re.fullmatch(r"attempt-([1-9][0-9]*)", path.name)
    return int(match.group(1)) if match else None


STATIC_TYPE = "static"

RELAX_TYPE = "relax"

AIMD_TYPE = "aimd"

SCHEDULED_TYPES = frozenset({STATIC_TYPE, RELAX_TYPE, AIMD_TYPE})

MAX_SCHEDULED_CALCULATIONS = 64

MAX_AIMD_FRAMES = 5000

_STATIC_RELAX_LIMITS = {
    "OUTCAR": 128 * 1024 * 1024,
    "OSZICAR": 16 * 1024 * 1024,
    "vasprun.xml": 256 * 1024 * 1024,
    "CONTCAR": 4 * 1024 * 1024,
}

_AIMD_LIMITS = {
    "OUTCAR": 512 * 1024 * 1024,
    "OSZICAR": 64 * 1024 * 1024,
    "vasprun.xml": 1024 * 1024 * 1024,
    "XDATCAR": 512 * 1024 * 1024,
}

RELAX_CONVERGENCE_MARKER = "reached required accuracy"
