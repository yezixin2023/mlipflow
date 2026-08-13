"""Safe adapter for deterministic VASP preparation and user-owned DFT labeling.

The adapter is deliberately side-effect free: it builds reviewed local argv and
verifies explicit manifests.  The bundled preparation wrapper uses pymatgen but
never launches VASP or a scheduler.  DFT execution remains a separate ``label``
operation and therefore requires a separate approval.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping


PLUGIN_ID = "dft-labeling"
PREPARE_OPERATION = "vasp-prepare"
LABEL_OPERATION = "label"
OPERATIONS = frozenset({PREPARE_OPERATION, LABEL_OPERATION})
BUNDLED_PREPARE_WRAPPER = (
    Path(globals().get("__file__", "adapter.py")).absolute().with_name("vasp_prepare.py")
)
SHELL_EXECUTABLES = frozenset(
    {"bash", "csh", "cmd", "dash", "fish", "ksh", "powershell", "pwsh", "sh", "tcsh", "zsh"}
)
SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
SAFE_POTCAR_SYMBOL = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_STRUCTURES = 10000
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
MANUSCRIPT_SUPPLEMENT_SHA256 = (
    "sha256:0e421d4236d8c9389eddd4e61f9503ada6ff0fd07f70f5bd1c32eea35214e732"
)
HISTORICAL_INCAR_SHA256 = (
    "sha256:69fbece3f536be6aa275d83e39a29bedda4d0ddc0a704d47ba7200642864b01c"
)
HISTORICAL_KPOINTS_SHA256 = (
    "sha256:3eda09df03e3fa250fd362b3f1f97b8a89cd9612eaebbaea5dbac1e2cf7a73a6"
)
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


def _has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    return any(candidate.is_symlink() for candidate in (absolute, *absolute.parents))


def _ordinary_file(path: Path, max_bytes: int | None = None) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    if max_bytes is None:
        return True
    size = path.stat().st_size
    return 1 <= size <= max_bytes


def _ordinary_project_file(path: Path, project_root: Path, max_bytes: int | None = None) -> bool:
    root = project_root.expanduser().absolute()
    candidate = path.expanduser().absolute()
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return False
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return False
    return _within(candidate.resolve(), root.resolve()) and _ordinary_file(candidate, max_bytes)


def _explicit_executable(value: Any) -> Path | None:
    if not _plain_string(value):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute() or _has_symlink_component(path) or not _ordinary_file(path):
        return None
    resolved = path.resolve()
    return resolved if resolved.stat().st_mode & 0o111 else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    if not _ordinary_file(path, MAX_INPUT_BYTES):
        return None, _diagnostic("warning", "artifact.missing", f"清单不存在或不是普通文件：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, _diagnostic("error", "artifact.invalid_json", f"无法读取 JSON：{exc}")
    if not isinstance(value, dict):
        return None, _diagnostic("error", "artifact.not_object", "清单必须是 JSON 对象。")
    return value, None


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _fingerprint(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256.fullmatch(value))


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
    if attempt_dir.exists() and (not attempt_dir.is_dir() or attempt_dir.is_symlink()):
        diagnostics.append(_diagnostic("error", "path.attempt_dir", "attempt_dir 必须是目录且不得为符号链接。"))
    elif project_root.is_dir() and not _within(attempt_dir.resolve(), project_root.resolve()):
        diagnostics.append(_diagnostic("error", "path.attempt_escape", "attempt_dir 必须位于 project_root 内。"))
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
                "error", "parameters.operation", "operation 必须是 vasp-prepare 或 label。"
            )
        )
    return diagnostics


def _validate_input_path(
    diagnostics: list[dict[str, str]], context: Mapping[str, Any], name: str, required: bool = True
) -> None:
    inputs = _mapping(context.get("inputs"))
    value = inputs.get(name)
    if value is None and not required:
        return
    if not _safe_relative(value):
        diagnostics.append(
            _diagnostic("error", f"inputs.{name}", f"{name} 必须是 project_root 下的安全相对路径。")
        )


def _parse_labeling_contract(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    value, error = _read_json(path)
    if value is None:
        return None, error["message"] if error else "labeling_config 无效"
    allowed = {"schema_version", "engine", "calculation_type", "preset", "incar", "kpoints", "sort_structure"}
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
    if calculation_type == "static" and (effective.get("NSW") != 0 or effective.get("IBRION") != -1):
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
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _valid_incar_value(value: Any) -> bool:
    if value is None or isinstance(value, Mapping):
        return False
    if isinstance(value, list):
        return bool(value) and all(not isinstance(item, (list, Mapping)) and _valid_incar_value(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return isinstance(value, (str, int, bool))


def _parse_pseudopotential_reference(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    value, error = _read_json(path)
    if value is None:
        return None, error["message"] if error else "pseudopotential_reference 无效"
    allowed = {
        "schema_version", "reference_id", "source_env", "license_acknowledged",
        "functional", "symbols", "expected_component_sha256", "expected_combined_sha256",
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
    expected_components = value.get("expected_component_sha256", {})
    if not isinstance(expected_components, Mapping) or any(
        not isinstance(symbol, str)
        or not SAFE_POTCAR_SYMBOL.fullmatch(symbol)
        or not _fingerprint(digest)
        for symbol, digest in expected_components.items()
    ):
        return None, "expected_component_sha256 必须是 sha256 映射"
    expected_combined = value.get("expected_combined_sha256")
    if expected_combined is not None and not _fingerprint(expected_combined):
        return None, "expected_combined_sha256 必须是完整 sha256"
    return value, None


def _validate_prepare(
    context: Mapping[str, Any], require_fresh_outputs: bool = True
) -> list[dict[str, str]]:
    diagnostics = _base_diagnostics(context)
    inputs = _mapping(context.get("inputs"))
    parameters = _mapping(context.get("parameters"))
    for key in ("structures_manifest", "labeling_config", "pseudopotential_reference"):
        _validate_input_path(diagnostics, context, key)
        if _safe_relative(inputs.get(key)):
            path = _path(context.get("project_root"), inputs[key])
            project_root = Path(str(context.get("project_root"))).expanduser().absolute()
            if not _ordinary_project_file(path, project_root, MAX_INPUT_BYTES):
                diagnostics.append(_diagnostic("error", f"inputs.{key}.missing", f"{key} 必须是现有普通文件。"))
    allowed_parameters = {"operation", "interpreter_argv", "engine", "result_manifest", "output_subdir", "max_structures"}
    unknown = sorted(set(parameters) - allowed_parameters)
    if "extra_args" in parameters:
        diagnostics.append(
            _diagnostic(
                "error",
                "parameters.extra_args",
                "不接受 extra_args；固定输入、输出和引擎参数不得被覆盖。",
            )
        )
    if unknown:
        diagnostics.append(_diagnostic("error", "parameters.unknown", "不支持参数：" + ", ".join(unknown)))
    if parameters.get("engine") != "vasp":
        diagnostics.append(_diagnostic("error", "parameters.engine", "vasp-prepare 的 engine 必须是 vasp。"))
    interpreter = parameters.get("interpreter_argv")
    executable = None
    if isinstance(interpreter, list) and len(interpreter) == 1:
        executable = _explicit_executable(interpreter[0])
    if executable is None or executable.name.lower() in SHELL_EXECUTABLES:
        diagnostics.append(
            _diagnostic("error", "parameters.interpreter_argv", "vasp-prepare 需要一个显式绝对、可执行、无符号链接的 Python 文件。")
        )
    result_manifest = parameters.get("result_manifest", "dft-input-manifest.json")
    output_subdir = parameters.get("output_subdir", "vasp-inputs")
    if not _safe_relative(result_manifest) or len(Path(str(result_manifest)).parts) != 1:
        diagnostics.append(_diagnostic("error", "parameters.result_manifest", "result_manifest 必须是 attempt_dir 的直接子文件。"))
    if not _safe_relative(output_subdir):
        diagnostics.append(_diagnostic("error", "parameters.output_subdir", "output_subdir 必须是安全相对路径。"))
    elif _safe_relative(result_manifest) and Path(str(result_manifest)) in Path(str(output_subdir)).parents:
        diagnostics.append(_diagnostic("error", "path.output_result_overlap", "output_subdir 不得位于 result_manifest 同名路径下。"))
    elif _safe_relative(result_manifest) and str(output_subdir) == str(result_manifest):
        diagnostics.append(_diagnostic("error", "path.output_result_overlap", "output_subdir 不得与 result_manifest 相同。"))
    max_structures = parameters.get("max_structures", MAX_STRUCTURES)
    if not _positive_int(max_structures) or int(max_structures) > MAX_STRUCTURES:
        diagnostics.append(_diagnostic("error", "parameters.max_structures", f"max_structures 必须在 1..{MAX_STRUCTURES}。"))
    attempt = Path(str(context.get("attempt_dir", ""))).expanduser().absolute()
    if (
        require_fresh_outputs
        and _safe_relative(result_manifest)
        and (attempt / str(result_manifest)).exists()
    ):
        diagnostics.append(_diagnostic("error", "path.result_exists", "vasp-prepare 要求新的 result_manifest。"))
    if (
        require_fresh_outputs
        and _safe_relative(output_subdir)
        and (attempt / str(output_subdir)).exists()
    ):
        diagnostics.append(_diagnostic("error", "path.output_exists", "vasp-prepare 要求新的 output_subdir。"))
    if _safe_relative(inputs.get("labeling_config")):
        config, message = _parse_labeling_contract(_path(context.get("project_root"), inputs["labeling_config"]))
        if config is None:
            diagnostics.append(_diagnostic("error", "inputs.labeling_config.contract", str(message)))
    if _safe_relative(inputs.get("pseudopotential_reference")):
        reference, message = _parse_pseudopotential_reference(
            _path(context.get("project_root"), inputs["pseudopotential_reference"])
        )
        if reference is None:
            diagnostics.append(_diagnostic("error", "inputs.pseudopotential_reference.contract", str(message)))
    return diagnostics


def _validate_label(context: Mapping[str, Any]) -> list[dict[str, str]]:
    backend = context.get("backend")
    diagnostics = _base_diagnostics(
        context, frozenset({"local", "ssh-slurm"})
    )
    parameters = _mapping(context.get("parameters"))
    for key in ("structures_manifest", "labeling_config"):
        _validate_input_path(diagnostics, context, key)
    _validate_input_path(diagnostics, context, "dft_input_manifest", required=False)
    allowed_parameters = {
        "operation", "label_script", "interpreter_argv", "engine",
        "completion_policy", "units", "result_manifest",
    }
    unknown = sorted(set(parameters) - allowed_parameters)
    if "extra_args" in parameters:
        diagnostics.append(
            _diagnostic(
                "error",
                "parameters.extra_args",
                "不接受 extra_args；固定输入、输出和引擎参数不得被覆盖。",
            )
        )
    if unknown:
        diagnostics.append(_diagnostic("error", "parameters.unknown", "不支持参数：" + ", ".join(unknown)))
    if backend == "local":
        if not _safe_relative(parameters.get("label_script")):
            diagnostics.append(_diagnostic("error", "parameters.label_script", "local label_script 必须是 project_root 下的安全相对路径。"))
        interpreter = parameters.get("interpreter_argv")
        if not (
            isinstance(interpreter, list)
            and len(interpreter) == 1
            and _plain_string(interpreter[0])
        ):
            diagnostics.append(_diagnostic("error", "parameters.interpreter_argv", "local interpreter_argv 必须只含一个可执行文件。"))
        elif Path(interpreter[0]).name.lower() in SHELL_EXECUTABLES:
            diagnostics.append(_diagnostic("error", "parameters.shell_forbidden", "interpreter_argv 不得选择 shell。"))
    elif backend == "ssh-slurm":
        if parameters.get("label_script") is not None or parameters.get("interpreter_argv") is not None:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "parameters.local_runner",
                    "ssh-slurm 使用站点模板，不接受 local label_script/interpreter_argv。",
                )
            )
        if _mapping(context.get("inputs")).get("dft_input_manifest") is None:
            diagnostics.append(_diagnostic("error", "inputs.dft_input_manifest", "ssh-slurm label 必须绑定已审核的 dft_input_manifest。"))
    if not _plain_string(parameters.get("engine")):
        diagnostics.append(_diagnostic("error", "parameters.engine", "engine 必须显式声明。"))
    elif backend == "ssh-slurm" and parameters.get("engine") != "vasp":
        diagnostics.append(_diagnostic("error", "parameters.engine", "vasp template family 的 engine 必须是 vasp。"))
    if not _safe_relative(parameters.get("result_manifest", "dft-labeling-result.json")):
        diagnostics.append(_diagnostic("error", "parameters.result_manifest", "result_manifest 必须是安全相对路径。"))
    elif backend == "ssh-slurm" and not SAFE_ID.fullmatch(
        str(parameters.get("result_manifest", "dft-labeling-result.json"))
    ):
        diagnostics.append(_diagnostic("error", "parameters.result_manifest", "ssh-slurm result_manifest 必须是安全 basename。"))
    completion = parameters.get("completion_policy")
    if not isinstance(completion, Mapping) or type(completion.get("require_ionic_convergence")) is not bool:
        diagnostics.append(_diagnostic("error", "parameters.completion_policy", "completion_policy 必须显式声明 require_ionic_convergence。"))
    units = parameters.get("units")
    required_units = {"energy", "length", "force", "stress"}
    if not isinstance(units, Mapping) or any(not _plain_string(units.get(key)) for key in required_units):
        diagnostics.append(_diagnostic("error", "parameters.units", "units 必须声明 energy、length、force、stress。"))
    elif backend == "ssh-slurm" and units != {
        "energy": "eV",
        "length": "angstrom",
        "force": "eV/angstrom",
        "stress": "kbar-vasp-3x3",
    }:
        diagnostics.append(_diagnostic("error", "parameters.units", "bundled static runner 固定输出 eV/angstrom/kbar-vasp-3x3 单位。"))
    if backend == "ssh-slurm":
        config_path = _mapping(context.get("inputs")).get("labeling_config")
        config: Mapping[str, Any] | None = None
        if _safe_relative(config_path):
            config, _ = _parse_labeling_contract(
                _path(context.get("project_root"), config_path)
            )
        calculation_type = (
            config.get("calculation_type") if isinstance(config, Mapping) else None
        )
        if calculation_type is not None and calculation_type not in SCHEDULED_TYPES:
            diagnostics.append(
                _diagnostic("error", "inputs.labeling_config", "scheduler runner 支持 static、relax 与 aimd。")
            )
        # Ionic convergence is a completion criterion for relax and only for
        # relax: static has a single ionic step, and AIMD is a trajectory rather
        # than a structural optimisation.
        expected_ionic = calculation_type == RELAX_TYPE
        if isinstance(completion, Mapping) and completion.get(
            "require_ionic_convergence"
        ) is not expected_ionic:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "parameters.completion_policy",
                    f"{calculation_type or 'scheduled'} 计算要求 require_ionic_convergence="
                    f"{str(expected_ionic).lower()}。",
                )
            )
    return diagnostics


def _file_fingerprints(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
        for name, path in paths.items()
    }


def _read_structure_count(path: Path) -> int | str:
    manifest, _ = _read_json(path)
    structures = manifest.get("structures") if isinstance(manifest, Mapping) else None
    return len(structures) if isinstance(structures, list) else "UNKNOWN"


def _plan_prepare(context: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = _validate_prepare(context)
    if _errors(diagnostics):
        return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": diagnostics}
    inputs = _mapping(context["inputs"])
    parameters = _mapping(context["parameters"])
    project_root = Path(str(context["project_root"])).expanduser().absolute().resolve()
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute().resolve()
    executable = _explicit_executable(parameters["interpreter_argv"][0])
    assert executable is not None
    paths = {
        "structures_manifest": _path(project_root, inputs["structures_manifest"]),
        "labeling_config": _path(project_root, inputs["labeling_config"]),
        "pseudopotential_reference": _path(project_root, inputs["pseudopotential_reference"]),
        "python_executable": executable,
        "prepare_wrapper": BUNDLED_PREPARE_WRAPPER,
    }
    result_name = str(parameters.get("result_manifest", "dft-input-manifest.json"))
    output_subdir = str(parameters.get("output_subdir", "vasp-inputs"))
    result_path = attempt / result_name
    argv = [
        str(executable), str(BUNDLED_PREPARE_WRAPPER),
        "--project-root", str(project_root),
        "--attempt-dir", str(attempt),
        "--structures-manifest", str(paths["structures_manifest"]),
        "--labeling-config", str(paths["labeling_config"]),
        "--pseudopotential-reference", str(paths["pseudopotential_reference"]),
        "--result-manifest", str(result_path),
        "--output-subdir", output_subdir,
        "--max-structures", str(parameters.get("max_structures", MAX_STRUCTURES)),
    ]
    config, _ = _parse_labeling_contract(paths["labeling_config"])
    reference, _ = _parse_pseudopotential_reference(paths["pseudopotential_reference"])
    assert config is not None and reference is not None
    return {
        "plugin_id": PLUGIN_ID,
        "status": "READY",
        "executable": True,
        "argv": argv,
        "cwd": str(attempt),
        "expected_outputs": [str(result_path)],
        "diagnostics": diagnostics,
        "operation": PREPARE_OPERATION,
        "input_fingerprints": _file_fingerprints(paths),
        "approval_summary": {
            "expensive": False,
            "runs_vasp": False,
            "submits_jobs": False,
            "engine": "vasp",
            "calculation_type": config["calculation_type"],
            "preset": config["preset"],
            "effective_incar": config["effective_incar"],
            "kpoints": config["kpoints"],
            "structure_count": _read_structure_count(paths["structures_manifest"]),
            "pseudopotential_reference": reference["reference_id"],
            "pseudopotential_functional": reference["functional"],
            "pseudopotential_symbols": reference["symbols"],
            "potcar_from_environment": "PMG_VASP_PSP_DIR",
            "potcar_collectable": False,
            "python_executable": str(executable),
            "generator": "pymatgen (version captured at execution)",
            "output_subdir": output_subdir,
        },
    }


def _plan_label(context: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = _validate_label(context)
    if _errors(diagnostics):
        return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": diagnostics}
    if context.get("backend") == "ssh-slurm":
        return _plan_scheduled_label(context, diagnostics)
    inputs = _mapping(context["inputs"])
    parameters = _mapping(context["parameters"])
    result_path = _path(context["attempt_dir"], parameters.get("result_manifest", "dft-labeling-result.json"))
    argv = list(parameters["interpreter_argv"])
    argv.extend(
        [
            str(_path(context["project_root"], parameters["label_script"])),
            "--operation", LABEL_OPERATION,
            "--structures-manifest", str(_path(context["project_root"], inputs["structures_manifest"])),
            "--labeling-config", str(_path(context["project_root"], inputs["labeling_config"])),
            "--attempt-dir", str(Path(str(context["attempt_dir"])).expanduser().absolute()),
            "--result-manifest", str(result_path),
            "--engine", str(parameters["engine"]),
        ]
    )
    if inputs.get("dft_input_manifest") is not None:
        argv.extend(["--dft-input-manifest", str(_path(context["project_root"], inputs["dft_input_manifest"]))])
    return {
        "plugin_id": PLUGIN_ID,
        "status": "READY",
        "executable": True,
        "argv": argv,
        "cwd": str(Path(str(context["attempt_dir"])).expanduser().absolute()),
        "expected_outputs": [str(result_path)],
        "diagnostics": diagnostics,
        "operation": LABEL_OPERATION,
        "approval_summary": {
            "expensive": True,
            "runs_vasp": True,
            "engine": parameters["engine"],
            "backend": context["backend"],
            "resources": context["resources"],
            "completion_policy": parameters["completion_policy"],
            "prepared_input_manifest": inputs.get("dft_input_manifest"),
        },
    }


def _staged_record(source: Path, remote_name: str, *, sensitive: bool = False) -> dict[str, Any]:
    return {
        "source": str(source.absolute()),
        "remote_name": remote_name,
        "sha256": _sha256(source),
        "size_bytes": source.stat().st_size,
        "sensitive": sensitive,
        "fetch_allowed": False,
    }


def _plan_scheduled_label(
    context: Mapping[str, Any], diagnostics: list[dict[str, str]]
) -> dict[str, Any]:
    """Plan one scheduler job covering 1..N VASP calculations.

    Calculation order is taken from the prepare manifest and frozen into
    calc-0001..calc-N, so the remote layout is deterministic and never depends
    on directory iteration order.  The calculation type is shared by the whole
    batch; it selects which outputs are required, not how the job is launched.
    """

    inputs = _mapping(context["inputs"])
    project_root = Path(str(context["project_root"])).expanduser().absolute().resolve()
    paths = {
        "structures_manifest": _path(project_root, inputs["structures_manifest"]),
        "labeling_config": _path(project_root, inputs["labeling_config"]),
        "dft_input_manifest": _path(project_root, inputs["dft_input_manifest"]),
    }
    for name, path in paths.items():
        if not _ordinary_project_file(path, project_root, MAX_INPUT_BYTES):
            diagnostics.append(_diagnostic("error", f"inputs.{name}", f"{name} 必须是 project_root 内的普通小文件。"))
    prepared, read_error = _read_json(paths["dft_input_manifest"])
    calculations = prepared.get("calculations") if isinstance(prepared, Mapping) else None
    calculation_type = (
        prepared.get("calculation_type") if isinstance(prepared, Mapping) else None
    )
    if (
        not isinstance(prepared, Mapping)
        or prepared.get("status") != "OK"
        or prepared.get("operation") != PREPARE_OPERATION
        or prepared.get("engine") != "vasp"
        or calculation_type not in SCHEDULED_TYPES
        or not isinstance(calculations, list)
        or not 1 <= len(calculations) <= MAX_SCHEDULED_CALCULATIONS
        or any(not isinstance(item, Mapping) for item in calculations)
    ):
        diagnostics.append(
            _diagnostic(
                "error",
                "inputs.dft_input_manifest.contract",
                str(
                    read_error["message"]
                    if read_error
                    else "scheduler runner 需要一个 OK 的 static/relax/aimd prepare manifest，"
                    f"且 calculation 数量在 1..{MAX_SCHEDULED_CALCULATIONS}。"
                ),
            )
        )
    if _errors(diagnostics):
        return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": diagnostics}
    assert isinstance(prepared, Mapping) and isinstance(calculations, list)
    assert isinstance(calculation_type, str)

    config, config_error = _parse_labeling_contract(paths["labeling_config"])
    if config is None or config.get("calculation_type") != calculation_type:
        diagnostics.append(
            _diagnostic(
                "error",
                "inputs.labeling_config",
                str(config_error or "labeling_config 与 prepare manifest 的 calculation_type 不一致。"),
            )
        )
        return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": diagnostics}

    prepared_root = paths["dft_input_manifest"].parent
    staged: list[dict[str, Any]] = [
        _staged_record(paths["structures_manifest"], "structures.json"),
        _staged_record(paths["labeling_config"], "labeling.json"),
        _staged_record(paths["dft_input_manifest"], "dft-input-manifest.json"),
    ]
    planned: list[dict[str, Any]] = []
    for index, calculation in enumerate(calculations):
        calc_id = _calculation_id(index)
        files = calculation.get("files")
        if not isinstance(files, Mapping) or set(files) != EXPECTED_FILE_NAMES:
            diagnostics.append(
                _diagnostic("error", f"inputs.dft_input_manifest.{calc_id}.files", f"{calc_id} 必须且只能声明四个 VASP 输入。")
            )
            continue
        for name in ("POSCAR", "INCAR", "KPOINTS", "POTCAR"):
            record = files.get(name)
            relative = record.get("path") if isinstance(record, Mapping) else None
            declared = record.get("sha256") if isinstance(record, Mapping) else None
            if not _safe_relative(relative):
                diagnostics.append(_diagnostic("error", f"inputs.dft_input_manifest.{calc_id}.{name}", f"{calc_id} 的 {name} 路径不安全。"))
                continue
            source = prepared_root / str(relative)
            if (
                not _ordinary_project_file(source, project_root)
                or not _fingerprint(declared)
                or _sha256(source) != declared
            ):
                diagnostics.append(
                    _diagnostic("error", f"inputs.dft_input_manifest.{calc_id}.{name}", f"{calc_id} 的 {name} 缺失或指纹与 prepare manifest 不一致。")
                )
                continue
            # Nested remote name: calc-0001/POSCAR.  POTCAR is licensed, so it is
            # staged but marked non-fetchable here and never appears in outputs.
            staged.append(
                _staged_record(source, f"{calc_id}/{name}", sensitive=name == "POTCAR")
            )
        planned.append(
            {
                "id": calc_id,
                "structure_id": calculation.get("structure_id"),
                "atom_count": calculation.get("atom_count"),
            }
        )
    if _errors(diagnostics):
        return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": diagnostics}

    required, optional, limits = _scheduled_output_spec(calculation_type)
    fetch_outputs: list[dict[str, Any]] = []
    for item in planned:
        calc_id = str(item["id"])
        for name in required:
            fetch_outputs.append(
                {
                    "remote_name": f"{calc_id}/{name}",
                    "local_name": f"{calc_id}/{name}",
                    "required": True,
                    "max_bytes": limits[name],
                    "role": "vasp-output",
                }
            )
        for name in optional:
            fetch_outputs.append(
                {
                    "remote_name": f"{calc_id}/{name}",
                    "local_name": f"{calc_id}/{name}",
                    "required": False,
                    "max_bytes": limits[name],
                    "role": "vasp-output",
                }
            )
        for stream in ("stdout", "stderr"):
            fetch_outputs.append(
                {
                    "remote_name": f"{calc_id}.{stream}",
                    "remote_path": f"logs/{calc_id}.{stream}",
                    "local_name": f"{calc_id}/{stream}.log",
                    "required": False,
                    "max_bytes": 16 * 1024 * 1024,
                    "role": "scheduler-log",
                }
            )
    return {
        "plugin_id": PLUGIN_ID,
        "status": "READY",
        "executable": True,
        "argv": ["template-family:vasp"],
        "cwd": "remote-attempt-workspace",
        "expected_outputs": [item["remote_name"] for item in fetch_outputs if item["required"]],
        "diagnostics": diagnostics,
        "operation": LABEL_OPERATION,
        "calculation_type": calculation_type,
        "calculations": planned,
        "approval_summary": {
            "calculation_type": calculation_type,
            "calculation_count": len(planned),
            "runs_vasp": True,
            "submits_jobs": True,
            # POTCAR is staged but licensed, so it never appears here and never
            # becomes a fetched or collected artifact.
            "fetch_allowlist": sorted({*required, *optional}),
            "staged_file_count": len(staged),
        },
        "input_fingerprints": _file_fingerprints(paths),
        "scheduled_execution": {
            "schema_version": 2,
            "template_family": "vasp",
            "staged_files": staged,
            "fetch_outputs": fetch_outputs,
        },
    }


def _parse_incar_scalar(token: str) -> Any:
    stripped = token.strip()
    upper = stripped.upper().strip(".")
    if upper in {"TRUE", "T"}:
        return True
    if upper in {"FALSE", "F"}:
        return False
    try:
        return int(stripped)
    except ValueError:
        try:
            value = float(stripped.replace("D", "E").replace("d", "e"))
            return value if math.isfinite(value) else stripped
        except ValueError:
            return stripped


def _parse_incar_file(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].split("!", 1)[0]
        for statement in line.split(";"):
            if "=" not in statement:
                continue
            raw_key, raw_value = statement.split("=", 1)
            key = raw_key.strip().upper()
            tokens = raw_value.split()
            if key:
                result[key] = _parse_incar_scalar(tokens[0]) if len(tokens) == 1 else [_parse_incar_scalar(token) for token in tokens]
    return result


def _same_incar_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
        return math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-15)
    if isinstance(expected, list) and isinstance(actual, list) and len(actual) == len(expected):
        return all(_same_incar_value(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, str) and isinstance(actual, str):
        return actual.lower() == expected.lower()
    return actual == expected


def _verify_kpoints(path: Path, expected: Mapping[str, Any]) -> bool:
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        mode = lines[2].lower()
        grid = [int(value) for value in lines[3].split()]
        shift = [float(value) for value in lines[4].split()] if len(lines) > 4 else [0.0, 0.0, 0.0]
    except (OSError, UnicodeError, ValueError, IndexError):
        return False
    expected_mode = str(expected["mode"])
    mode_ok = mode.startswith("g") if expected_mode == "gamma" else mode.startswith("m")
    return mode_ok and grid == list(expected["grid"]) and all(
        math.isclose(a, float(b), rel_tol=0, abs_tol=1e-12) for a, b in zip(shift, expected["shift"])
    )


def _verify_prepare_result(
    context: Mapping[str, Any], manifest: Mapping[str, Any], verify_files: bool
) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    inputs = _mapping(context["inputs"])
    parameters = _mapping(context["parameters"])
    project_root = Path(str(context["project_root"])).expanduser().absolute().resolve()
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute().resolve()
    expected_identity = {
        "schema_version": 1, "plugin_id": PLUGIN_ID, "operation": PREPARE_OPERATION,
        "status": "OK", "engine": "vasp", "execution_ready": True,
    }
    for key, expected in expected_identity.items():
        if manifest.get(key) != expected:
            diagnostics.append(_diagnostic("error", f"result.{key}", f"结果 {key} 与已批准合同不一致。"))
    generator = manifest.get("generator")
    if not isinstance(generator, Mapping) or generator.get("name") != "pymatgen" or not _plain_string(generator.get("version")):
        diagnostics.append(_diagnostic("error", "result.generator", "必须记录 pymatgen 名称与版本。"))
    runtime = manifest.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("prepare_wrapper_sha256") != _sha256(BUNDLED_PREPARE_WRAPPER)
    ):
        diagnostics.append(_diagnostic("error", "result.runtime", "prepare wrapper 指纹与当前批准实现不一致。"))
    input_paths = {
        "structures_manifest": _path(project_root, inputs["structures_manifest"]),
        "labeling_config": _path(project_root, inputs["labeling_config"]),
        "pseudopotential_reference": _path(project_root, inputs["pseudopotential_reference"]),
    }
    fingerprints = manifest.get("input_fingerprints")
    if not isinstance(fingerprints, Mapping):
        diagnostics.append(_diagnostic("error", "result.input_fingerprints", "缺少三个输入指纹。"))
    else:
        for name, path in input_paths.items():
            declared = fingerprints.get(name)
            if not _fingerprint(declared):
                diagnostics.append(_diagnostic("error", f"result.input_fingerprints.{name}", "必须是完整 sha256。"))
            elif verify_files and (
                not _ordinary_project_file(path, project_root, MAX_INPUT_BYTES)
                or _sha256(path) != declared
            ):
                diagnostics.append(_diagnostic("error", f"result.input_fingerprints.{name}", "输入文件已变化或缺失。"))
    config, config_error = _parse_labeling_contract(input_paths["labeling_config"])
    reference, reference_error = _parse_pseudopotential_reference(input_paths["pseudopotential_reference"])
    if config is None:
        diagnostics.append(_diagnostic("error", "result.labeling_config", str(config_error)))
        return diagnostics
    if reference is None:
        diagnostics.append(_diagnostic("error", "result.pseudopotential_reference", str(reference_error)))
        return diagnostics
    if manifest.get("calculation_type") != config["calculation_type"] or manifest.get("preset") != config["preset"]:
        diagnostics.append(_diagnostic("error", "result.calculation_type", "calculation_type/preset 与输入配置不一致。"))
    incar_record = manifest.get("incar")
    if not isinstance(incar_record, Mapping) or incar_record.get("effective") != config["effective_incar"]:
        diagnostics.append(_diagnostic("error", "result.incar", "记录的有效 INCAR 参数与输入配置不一致。"))
    if manifest.get("kpoints") != config["kpoints"]:
        diagnostics.append(_diagnostic("error", "result.kpoints", "记录的 KPOINTS 规则与输入配置不一致。"))
    if manifest.get("sort_structure") is not config["sort_structure"]:
        diagnostics.append(_diagnostic("error", "result.sort_structure", "结构排序策略与输入配置不一致。"))
    provenance = manifest.get("parameter_provenance")
    if config["preset"] == MANUSCRIPT_STATIC_PRESET:
        paper = provenance.get("paper") if isinstance(provenance, Mapping) else None
        historical = provenance.get("historical_template") if isinstance(provenance, Mapping) else None
        if (
            not isinstance(paper, Mapping)
            or paper.get("sha256") != MANUSCRIPT_SUPPLEMENT_SHA256
            or paper.get("declared_parameters")
            != {"ENCUT": 450, "EDIFF": 5e-6, "IALGO": 38}
            or not isinstance(historical, Mapping)
            or historical.get("incar_sha256") != HISTORICAL_INCAR_SHA256
            or historical.get("kpoints_sha256") != HISTORICAL_KPOINTS_SHA256
        ):
            diagnostics.append(_diagnostic("error", "result.parameter_provenance", "论文/历史模板 provenance 不完整或已变化。"))
    elif provenance != {}:
        diagnostics.append(_diagnostic("error", "result.parameter_provenance", "非 preset 输入不得冒充论文参数 provenance。"))
    potcar_policy = manifest.get("potcar_policy")
    if (
        not isinstance(potcar_policy, Mapping)
        or potcar_policy.get("source_env") != "PMG_VASP_PSP_DIR"
        or potcar_policy.get("configuration_source")
        not in {"environment", "pymatgen-settings"}
        or potcar_policy.get("materialized_in_attempt") is not True
        or potcar_policy.get("portable_or_collectable") is not False
    ):
        diagnostics.append(_diagnostic("error", "result.potcar_policy", "POTCAR 策略不满足运行时生成且禁止收集。"))
    calculations = manifest.get("calculations")
    count = manifest.get("structure_count")
    if not isinstance(calculations, list) or not _positive_int(count) or len(calculations) != count:
        diagnostics.append(_diagnostic("error", "result.structure_count", "structure_count 与 calculations 不一致。"))
        return diagnostics
    structures_manifest, structure_error = _read_json(input_paths["structures_manifest"])
    raw_structures = structures_manifest.get("structures") if isinstance(structures_manifest, Mapping) else None
    if not isinstance(raw_structures, list) or len(raw_structures) != count:
        diagnostics.append(_diagnostic("error", "result.source_structures", str(structure_error or "源结构数量不一致。")))
        return diagnostics
    output_subdir = str(parameters.get("output_subdir", "vasp-inputs"))
    expected_reference_id = reference.get("reference_id")
    for index, (calculation, source) in enumerate(zip(calculations, raw_structures), 1):
        prefix = f"calculations[{index - 1}]"
        if not isinstance(calculation, Mapping) or not isinstance(source, Mapping):
            diagnostics.append(_diagnostic("error", prefix, "结构计算记录必须是对象。"))
            continue
        structure_id = source.get("id", source.get("structure_id"))
        source_path = source.get("path", source.get("output_file"))
        source_digest = source.get("fingerprint", source.get("frame_sha256", source.get("sha256")))
        if not isinstance(structure_id, str) or not SAFE_ID.fullmatch(structure_id):
            diagnostics.append(_diagnostic("error", f"{prefix}.structure_id", "源结构 ID 无效。"))
            continue
        if not _safe_relative(source_path) or not _fingerprint(source_digest):
            diagnostics.append(_diagnostic("error", f"{prefix}.source", "源结构必须有安全相对路径和完整 sha256。"))
            continue
        expected_dir = f"{output_subdir}/{index:06d}-{structure_id}"
        if calculation.get("order") != index or calculation.get("structure_id") != structure_id or calculation.get("directory") != expected_dir:
            diagnostics.append(_diagnostic("error", f"{prefix}.identity", "结构顺序、ID 或目录不一致。"))
        result_source = calculation.get("source")
        if not isinstance(result_source, Mapping) or result_source.get("path") != source_path or result_source.get("sha256") != source_digest:
            diagnostics.append(_diagnostic("error", f"{prefix}.source", "结构来源路径/指纹不一致。"))
        if verify_files and _safe_relative(source_path):
            actual_source = input_paths["structures_manifest"].parent / str(source_path)
            if (
                not _ordinary_project_file(actual_source, project_root)
                or _sha256(actual_source) != source_digest
            ):
                diagnostics.append(_diagnostic("error", f"{prefix}.source", "源结构已变化或缺失。"))
        files = calculation.get("files")
        if not isinstance(files, Mapping) or set(files) != EXPECTED_FILE_NAMES:
            diagnostics.append(_diagnostic("error", f"{prefix}.files", "必须且只能声明 POSCAR/INCAR/KPOINTS/POTCAR。"))
            continue
        for name in sorted(EXPECTED_FILE_NAMES):
            record = files.get(name)
            expected_path = f"{expected_dir}/{name}"
            if not isinstance(record, Mapping) or record.get("path") != expected_path or not _fingerprint(record.get("sha256")):
                diagnostics.append(_diagnostic("error", f"{prefix}.files.{name}", "文件路径或 sha256 无效。"))
                continue
            if record.get("collectable") is not (name != "POTCAR"):
                diagnostics.append(_diagnostic("error", f"{prefix}.files.{name}.collectable", "POTCAR 必须禁止收集，其余输入必须可收集。"))
            if verify_files:
                actual = attempt / expected_path
                if (
                    not _ordinary_project_file(actual, project_root)
                    or actual.stat().st_size != record.get("size_bytes")
                    or _sha256(actual) != record.get("sha256")
                ):
                    diagnostics.append(_diagnostic("error", f"{prefix}.files.{name}.integrity", f"{name} 缺失或指纹不匹配。"))
        potcar_file = files.get("POTCAR") if isinstance(files.get("POTCAR"), Mapping) else {}
        potcar = calculation.get("potcar")
        expected_symbols = reference.get("symbols")
        if not isinstance(potcar, Mapping) or potcar.get("reference_id") != expected_reference_id or potcar.get("source_env") != "PMG_VASP_PSP_DIR" or potcar.get("functional") != reference.get("functional") or potcar.get("portable_artifact") is not False:
            diagnostics.append(_diagnostic("error", f"{prefix}.potcar", "POTCAR 来源与批准引用不一致。"))
        elif isinstance(expected_symbols, Mapping):
            elements = potcar.get("elements")
            symbols = potcar.get("symbols")
            if not isinstance(elements, list) or not isinstance(symbols, list) or symbols != [expected_symbols.get(element) for element in elements]:
                diagnostics.append(_diagnostic("error", f"{prefix}.potcar.symbols", "POTCAR symbols 与显式元素映射不一致。"))
            if potcar.get("combined_sha256") != potcar_file.get("sha256"):
                diagnostics.append(_diagnostic("error", f"{prefix}.potcar.sha256", "POTCAR combined sha256 与文件不一致。"))
            components = potcar.get("components")
            expected_component_hashes = reference.get("expected_component_sha256", {})
            components_valid = isinstance(symbols, list) and isinstance(components, list)
            if components_valid:
                components_valid = len(components) == len(symbols) and all(
                    isinstance(component, Mapping)
                    and component.get("symbol") == symbol
                    and _fingerprint(component.get("sha256"))
                    and (
                        expected_component_hashes.get(symbol) is None
                        or component.get("sha256") == expected_component_hashes.get(symbol)
                    )
                    for component, symbol in zip(components, symbols)
                )
            if not components_valid:
                diagnostics.append(_diagnostic("error", f"{prefix}.potcar.components", "POTCAR component symbol/hash 与批准引用不一致。"))
            expected_combined = reference.get("expected_combined_sha256")
            if expected_combined is not None and potcar.get("combined_sha256") != expected_combined:
                diagnostics.append(_diagnostic("error", f"{prefix}.potcar.approved_sha256", "POTCAR 与批准 combined sha256 不一致。"))
        if verify_files and isinstance(files.get("INCAR"), Mapping):
            incar_path = attempt / str(files["INCAR"].get("path", ""))
            if _ordinary_project_file(incar_path, project_root):
                parsed_incar = _parse_incar_file(incar_path)
                if any(not _same_incar_value(parsed_incar.get(key), value) for key, value in config["effective_incar"].items()):
                    diagnostics.append(_diagnostic("error", f"{prefix}.incar.parameters", "INCAR 内容与批准参数不一致。"))
        if verify_files and isinstance(files.get("KPOINTS"), Mapping):
            kpoints_path = attempt / str(files["KPOINTS"].get("path", ""))
            if _ordinary_project_file(kpoints_path, project_root) and not _verify_kpoints(
                kpoints_path, config["kpoints"]
            ):
                diagnostics.append(_diagnostic("error", f"{prefix}.kpoints.rule", "KPOINTS 内容与批准规则不一致。"))
    return diagnostics


def _verify_label_result(
    context: Mapping[str, Any], manifest: Mapping[str, Any], verify_files: bool
) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    parameters = _mapping(context["parameters"])
    for key, expected in (("schema_version", 1), ("plugin_id", PLUGIN_ID), ("status", "OK"), ("engine", parameters.get("engine"))):
        if manifest.get(key) != expected:
            diagnostics.append(_diagnostic("error", f"result.{key}", f"结果 {key} 不匹配。"))
    input_paths = {
        "structures_manifest": _path(context["project_root"], _mapping(context["inputs"])["structures_manifest"]),
        "labeling_config": _path(context["project_root"], _mapping(context["inputs"])["labeling_config"]),
    }
    if _mapping(context["inputs"]).get("dft_input_manifest") is not None:
        input_paths["dft_input_manifest"] = _path(context["project_root"], _mapping(context["inputs"])["dft_input_manifest"])
    fingerprints = manifest.get("input_fingerprints")
    if not isinstance(fingerprints, Mapping):
        diagnostics.append(_diagnostic("error", "result.input_fingerprints", "缺少 input_fingerprints。"))
    else:
        for name, path in input_paths.items():
            declared = fingerprints.get(name)
            if not _fingerprint(declared):
                diagnostics.append(_diagnostic("error", f"result.input_fingerprints.{name}", "必须声明完整 sha256。"))
            elif verify_files and (not _ordinary_file(path) or _sha256(path) != declared):
                diagnostics.append(_diagnostic("error", f"result.input_fingerprints.{name}", "输入 sha256 不匹配。"))
    completion = manifest.get("completion")
    if not isinstance(completion, Mapping):
        diagnostics.append(_diagnostic("error", "result.completion", "缺少标准 completion。"))
    else:
        if completion.get("scheduler_success") is not True:
            diagnostics.append(_diagnostic("error", "completion.scheduler", "调度/进程未成功。"))
        if completion.get("electronic_converged") is not True:
            diagnostics.append(_diagnostic("error", "completion.electronic", "电子步未收敛。"))
        if completion.get("truncated") is not False:
            diagnostics.append(_diagnostic("error", "completion.truncated", "输出截断状态不安全。"))
        ionic_required = completion.get("ionic_convergence_required")
        ionic_converged = completion.get("ionic_converged")
        expected_required = _mapping(parameters.get("completion_policy")).get("require_ionic_convergence")
        if type(ionic_required) is not bool or ionic_required is not expected_required:
            diagnostics.append(_diagnostic("error", "completion.ionic_policy", "离子收敛策略与批准计划不一致。"))
        elif ionic_required and ionic_converged is not True:
            diagnostics.append(_diagnostic("error", "completion.ionic", "离子步未收敛。"))
        elif not ionic_required and ionic_converged not in {None, True}:
            diagnostics.append(_diagnostic("error", "completion.ionic", "静态计算 ionic_converged 只能为 null 或 true。"))
    if manifest.get("units") != parameters.get("units"):
        diagnostics.append(_diagnostic("error", "result.units", "结果单位与批准 units 不一致。"))
    source_count = manifest.get("source_structure_count")
    label_count = manifest.get("label_count")
    if not _positive_int(source_count) or label_count != source_count:
        diagnostics.append(_diagnostic("error", "result.label_count", "每个源结构必须恰有一个标签。"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        diagnostics.append(_diagnostic("error", "result.artifacts", "必须声明至少一个数据集产物。"))
        return diagnostics
    names: set[str] = set()
    for index, artifact in enumerate(artifacts):
        prefix = f"artifacts[{index}]"
        if not isinstance(artifact, Mapping):
            diagnostics.append(_diagnostic("error", prefix, "产物必须是对象。"))
            continue
        name = artifact.get("name")
        if not _plain_string(name) or name in names:
            diagnostics.append(_diagnostic("error", f"{prefix}.name", "产物 name 必须非空且唯一。"))
        else:
            names.add(str(name))
        relative = artifact.get("path")
        fingerprint = artifact.get("fingerprint")
        if not _safe_relative(relative) or not _fingerprint(fingerprint):
            diagnostics.append(_diagnostic("error", f"{prefix}.integrity", "产物必须有安全路径和完整 sha256。"))
            continue
        if not _plain_string(artifact.get("media_type")):
            diagnostics.append(_diagnostic("error", f"{prefix}.media_type", "产物必须声明 media_type。"))
        if verify_files:
            path = _path(context["attempt_dir"], relative)
            if not _ordinary_file(path) or path.stat().st_size == 0 or _sha256(path) != fingerprint:
                diagnostics.append(_diagnostic("error", f"{prefix}.integrity", "数据产物缺失或 sha256 不匹配。"))
    if context.get("backend") == "ssh-slurm":
        diagnostics.extend(_verify_scheduled_vasp_result(context, manifest, verify_files))
    return diagnostics


def _xml_rows(element: ET.Element | None) -> list[list[float]] | None:
    if element is None:
        return None
    try:
        rows = [[float(token) for token in (item.text or "").split()] for item in element.findall("v")]
    except ValueError:
        return None
    if not rows or any(not row or any(not math.isfinite(value) for value in row) for row in rows):
        return None
    return rows


def _same_numeric_tree(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-10)
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _same_numeric_tree(a, e) for a, e in zip(actual, expected)
        )
    return actual == expected


# ---------------------------------------------------------------------------
# Multi-calculation scheduled execution.
#
# One node -> one attempt -> one scheduler job -> 1..N VASP calculations run
# sequentially.  Calculation identity is positional and deterministic: the order
# of ``calculations`` in the prepare manifest defines calc-0001..calc-N, so the
# remote layout never depends on a glob.
# ---------------------------------------------------------------------------

STATIC_TYPE = "static"
RELAX_TYPE = "relax"
AIMD_TYPE = "aimd"
SCHEDULED_TYPES = frozenset({STATIC_TYPE, RELAX_TYPE, AIMD_TYPE})
MAX_SCHEDULED_CALCULATIONS = 64
MAX_AIMD_FRAMES = 5000

# static/relax keep the original conservative bounds.  AIMD gets its own larger
# but still bounded policy: a trajectory legitimately produces far more output,
# and pretending the static limits apply would either truncate silently or fail
# for the wrong reason.  Exceeding a bound is a hard failure, never a truncation.
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


def _calculation_id(index: int) -> str:
    return f"calc-{index + 1:04d}"


def _scheduled_output_spec(
    calculation_type: str,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, int]]:
    """Required outputs, optional outputs and byte bounds for one type."""

    if calculation_type == RELAX_TYPE:
        return (
            ("OUTCAR", "OSZICAR", "vasprun.xml", "CONTCAR"),
            (),
            dict(_STATIC_RELAX_LIMITS),
        )
    if calculation_type == AIMD_TYPE:
        return (
            ("OUTCAR", "OSZICAR", "vasprun.xml"),
            ("XDATCAR",),
            dict(_AIMD_LIMITS),
        )
    return ("OUTCAR", "OSZICAR", "vasprun.xml"), (), dict(_STATIC_RELAX_LIMITS)


def _frame_from_calculation(element: ET.Element) -> dict[str, Any] | None:
    """Build one ionic-step frame from a single ``<calculation>`` element."""

    scsteps = element.findall("./scstep")
    energy = math.nan
    if scsteps:
        node = scsteps[-1].find("./energy/i[@name='e_0_energy']")
        if node is not None:
            energy = float(str(node.text))
    if not math.isfinite(energy):
        fallback = element.find("./energy/i[@name='e_wo_entrp']")
        energy = float(str(fallback.text)) if fallback is not None else math.nan
    structure = element.find("./structure")
    lattice = (
        _xml_rows(structure.find("./crystal/varray[@name='basis']"))
        if structure is not None
        else None
    )
    positions = (
        _xml_rows(structure.find("./varray[@name='positions']"))
        if structure is not None
        else None
    )
    forces = _xml_rows(element.find("./varray[@name='forces']"))
    stress = _xml_rows(element.find("./varray[@name='stress']"))
    # lattice/positions may be absent when a writer emits only a document-level
    # finalpos structure; the caller fills the final frame from it.
    if not math.isfinite(energy) or forces is None or not scsteps:
        return None
    return {
        "energy_ev": energy,
        "lattice_angstrom": lattice,
        "fractional_coordinates": positions,
        "forces_ev_per_angstrom": forces,
        # AIMD steps do not always carry a stress tensor; that is legitimate and
        # is recorded as absent rather than invented.
        "stress_kbar_vasp_3x3": stress,
        "electronic_steps": len(scsteps),
    }


def _parse_vasprun_trajectory(
    path: Path, max_frames: int = MAX_AIMD_FRAMES
) -> dict[str, Any] | None:
    """Stream every ionic step out of one vasprun.xml.

    ``iterparse`` is used and each ``<calculation>`` is released as soon as its
    frame has been extracted, so a long AIMD trajectory is never materialised as
    one in-memory tree.  ``NELM`` and ``atominfo`` both precede the calculations
    in VASP's output, so a single forward pass suffices.
    """

    nelm: int | None = None
    species: list[str] = []
    frames: list[dict[str, Any]] = []
    final_lattice: Any = None
    final_positions: Any = None
    exceeded = False
    try:
        for _, element in ET.iterparse(str(path), events=("end",)):
            tag = element.tag
            if tag == "separator" and element.get("name") == "electronic convergence":
                node = element.find("./i[@name='NELM']")
                if node is not None and nelm is None:
                    nelm = int(float(str(node.text)))
                element.clear()
            elif tag == "atominfo":
                if not species:
                    species = [
                        str(row.findtext("c", default="")).strip()
                        for row in element.findall("./array[@name='atoms']/set/rc")
                    ]
                element.clear()
            elif tag == "structure" and element.get("name") == "finalpos":
                # Document-level final structure; only used to complete the last
                # frame when the writer omitted a per-calculation structure.
                final_lattice = _xml_rows(
                    element.find("./crystal/varray[@name='basis']")
                )
                final_positions = _xml_rows(element.find("./varray[@name='positions']"))
                element.clear()
            elif tag == "calculation":
                if len(frames) >= max_frames:
                    exceeded = True
                    break
                frame = _frame_from_calculation(element)
                if frame is None:
                    return None
                frames.append(frame)
                element.clear()
    except (ET.ParseError, OSError, ValueError, IndexError):
        return None
    if nelm is None or not species or not frames:
        return None
    if frames[-1]["lattice_angstrom"] is None:
        frames[-1]["lattice_angstrom"] = final_lattice
    if frames[-1]["fractional_coordinates"] is None:
        frames[-1]["fractional_coordinates"] = final_positions
    if any(
        frame["lattice_angstrom"] is None or frame["fractional_coordinates"] is None
        for frame in frames
    ):
        return None
    return {
        "nelm": nelm,
        "species": species,
        "frames": frames,
        "frame_limit_exceeded": exceeded,
        # The document-level finalpos is the geometry *after* the last ionic
        # update, so it matches CONTCAR but not the last force evaluation.  It is
        # reported separately rather than mixed into the frames.
        "final_lattice_angstrom": final_lattice,
        "final_fractional_coordinates": final_positions,
    }


def _parse_scheduled_vasprun(path: Path) -> dict[str, Any] | None:
    """Final-frame view of one vasprun.xml, used by the single-frame paths.

    Built on the streaming trajectory parser so static, relax and AIMD all read
    NELM and the sigma->0 energy through exactly one implementation.  Two real
    VASP 5.4.x layouts are handled there and must stay handled: NELM appears
    twice (electronic convergence, and 1 under response functions), and the
    final <calculation><energy> block reports the electronic entropy in
    e_0_energy while the real E0 lives in the last <scstep>.
    """

    parsed = _parse_vasprun_trajectory(path)
    if parsed is None or parsed["frame_limit_exceeded"]:
        return None
    final = parsed["frames"][-1]
    value = {
        "energy_ev": final["energy_ev"],
        "species": parsed["species"],
        "lattice_angstrom": final["lattice_angstrom"],
        "fractional_coordinates": final["fractional_coordinates"],
        "forces_ev_per_angstrom": final["forces_ev_per_angstrom"],
        "stress_kbar_vasp_3x3": final["stress_kbar_vasp_3x3"],
        "electronic_steps": final["electronic_steps"],
        "nelm": parsed["nelm"],
    }
    if any(
        value[key] is None
        for key in (
            "lattice_angstrom",
            "fractional_coordinates",
            "forces_ev_per_angstrom",
            "stress_kbar_vasp_3x3",
        )
    ):
        return None
    return value


def _verify_scheduled_vasp_result(
    context: Mapping[str, Any], manifest: Mapping[str, Any], verify_files: bool
) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    execution = manifest.get("execution")
    planned = _mapping(_mapping(context.get("execution")).get("plan"))
    scheduled = _mapping(planned.get("scheduled_execution"))
    hpc_execution = _mapping(_mapping(context.get("execution")).get("hpc_execution"))
    if (
        not isinstance(execution, Mapping)
        or execution.get("template_family") != scheduled.get("template_family")
        or execution.get("templates") != hpc_execution.get("templates")
    ):
        diagnostics.append(_diagnostic("error", "result.execution", "结果引用的远端模板与批准计划不一致。"))
    raw = manifest.get("raw_outputs")
    required = {"OUTCAR", "OSZICAR", "vasprun.xml"}
    if not isinstance(raw, Mapping) or set(raw) != required:
        diagnostics.append(_diagnostic("error", "result.raw_outputs", "必须且只能声明 OUTCAR、OSZICAR、vasprun.xml。"))
        return diagnostics
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    for name in sorted(required):
        record = raw.get(name)
        path = attempt / name
        if (
            not isinstance(record, Mapping)
            or record.get("path") != name
            or not _fingerprint(record.get("fingerprint"))
            or not _positive_int(record.get("size_bytes"))
        ):
            diagnostics.append(_diagnostic("error", f"result.raw_outputs.{name}", "原始 VASP 输出记录无效。"))
            continue
        if verify_files and (
            not _ordinary_file(path)
            or path.stat().st_size != record["size_bytes"]
            or _sha256(path) != record["fingerprint"]
        ):
            diagnostics.append(_diagnostic("error", f"result.raw_outputs.{name}", "拉回的原始 VASP 输出缺失或指纹不匹配。"))
    if not verify_files or _errors(diagnostics):
        return diagnostics
    outcar = (attempt / "OUTCAR").read_text(encoding="utf-8", errors="replace")
    if "General timing and accounting informations for this job:" not in outcar:
        diagnostics.append(_diagnostic("error", "completion.outcar", "OUTCAR 不含正常结束 footer。"))
    parsed = _parse_scheduled_vasprun(attempt / "vasprun.xml")
    if parsed is None:
        diagnostics.append(_diagnostic("error", "completion.vasprun", "vasprun.xml 无法完整解析。"))
        return diagnostics
    if not (0 < parsed["electronic_steps"] < parsed["nelm"]):
        diagnostics.append(_diagnostic("error", "completion.electronic", "独立解析显示电子步达到 NELM 或为空。"))
    if isinstance(execution, Mapping) and (
        execution.get("electronic_steps") != parsed["electronic_steps"]
        or execution.get("nelm") != parsed["nelm"]
    ):
        diagnostics.append(_diagnostic("error", "result.execution.steps", "电子步数/NELM 与独立解析不一致。"))
    artifacts = manifest.get("artifacts")
    labels_record = next(
        (
            item
            for item in artifacts
            if isinstance(item, Mapping) and item.get("name") == "labels-json"
        ),
        None,
    ) if isinstance(artifacts, list) else None
    labels_path = attempt / str(labels_record.get("path", "")) if isinstance(labels_record, Mapping) else attempt / ""
    labels, _ = _read_json(labels_path)
    records = labels.get("records") if isinstance(labels, Mapping) else None
    record = records[0] if isinstance(records, list) and len(records) == 1 and isinstance(records[0], Mapping) else None
    if record is None:
        diagnostics.append(_diagnostic("error", "result.labels", "labels.json 必须包含恰好一个标签记录。"))
        return diagnostics
    expected = {
        key: parsed[key]
        for key in (
            "energy_ev",
            "species",
            "lattice_angstrom",
            "fractional_coordinates",
            "forces_ev_per_angstrom",
            "stress_kbar_vasp_3x3",
        )
    }
    if any(not _same_numeric_tree(record.get(key), value) for key, value in expected.items()):
        diagnostics.append(_diagnostic("error", "result.labels.values", "标签能量/结构/力/应力与独立 vasprun.xml 解析不一致。"))
    return diagnostics


def _scheduled_calculation_type(context: Mapping[str, Any]) -> str:
    """Calculation type of the approved plan, defaulting to static."""

    planned = _mapping(_mapping(context.get("execution")).get("plan"))
    declared = planned.get("calculation_type")
    if declared in SCHEDULED_TYPES:
        return str(declared)
    prepared, _ = _read_json(
        _path(context["project_root"], _mapping(context["inputs"])["dft_input_manifest"])
    )
    value = prepared.get("calculation_type") if isinstance(prepared, Mapping) else None
    return str(value) if value in SCHEDULED_TYPES else STATIC_TYPE


def _requested_nsw(context: Mapping[str, Any]) -> int | None:
    config, _ = _parse_labeling_contract(
        _path(context["project_root"], _mapping(context["inputs"])["labeling_config"])
    )
    if config is None:
        return None
    value = _mapping(config.get("effective_incar")).get("NSW")
    return int(value) if _positive_int(value) else None


def _check_one_calculation(
    attempt: Path,
    calc_id: str,
    calculation_type: str,
    expected_atoms: Any,
    requested_nsw: int | None,
) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    """Scientific completion for a single calculation.

    Every diagnostic is prefixed with the calculation id, so a batch failure
    names the calculation and the reason instead of collapsing into
    "batch failed".
    """

    diagnostics: list[dict[str, str]] = []
    directory = attempt / calc_id
    required, _, limits = _scheduled_output_spec(calculation_type)
    for name in required:
        path = directory / name
        if not _ordinary_file(path) or path.stat().st_size == 0:
            diagnostics.append(
                _diagnostic("error", f"{calc_id}.raw.{name}", f"{calc_id} 缺少非空的 {name}。")
            )
        elif path.stat().st_size > limits.get(name, 0):
            diagnostics.append(
                _diagnostic("error", f"{calc_id}.raw.{name}", f"{calc_id} 的 {name} 超过已批准的大小上限。")
            )
    if _errors(diagnostics):
        return diagnostics, None

    outcar = (directory / "OUTCAR").read_text(encoding="utf-8", errors="replace")
    if "General timing and accounting informations for this job:" not in outcar:
        diagnostics.append(
            _diagnostic("error", f"{calc_id}.completion.outcar", f"{calc_id} 的 OUTCAR 不含正常结束 footer。")
        )
    max_frames = MAX_AIMD_FRAMES if calculation_type == AIMD_TYPE else 1024
    parsed = _parse_vasprun_trajectory(directory / "vasprun.xml", max_frames)
    if parsed is None:
        diagnostics.append(
            _diagnostic("error", f"{calc_id}.completion.vasprun", f"{calc_id} 的 vasprun.xml 无法完整解析。")
        )
        return diagnostics, None
    if parsed["frame_limit_exceeded"]:
        diagnostics.append(
            _diagnostic("error", f"{calc_id}.completion.frames", f"{calc_id} 的 ionic step 数超过已批准上限 {max_frames}。")
        )
        return diagnostics, None

    frames = parsed["frames"]
    nelm = parsed["nelm"]
    # Electronic convergence is required for every ionic step, not just the last
    # one: a single unconverged SCF step poisons that frame's forces.
    for position, frame in enumerate(frames, start=1):
        if not 0 < frame["electronic_steps"] < nelm:
            diagnostics.append(
                _diagnostic(
                    "error",
                    f"{calc_id}.completion.electronic",
                    f"{calc_id} 的第 {position} 个 ionic step 电子步达到 NELM({nelm}) 或为空。",
                )
            )
    species = parsed["species"]
    if _positive_int(expected_atoms) and len(species) != int(expected_atoms):
        diagnostics.append(
            _diagnostic("error", f"{calc_id}.result.atom_count", f"{calc_id} 的原子数与准备清单不一致。")
        )
    for position, frame in enumerate(frames, start=1):
        if len(frame["fractional_coordinates"]) != len(species) or len(
            frame["forces_ev_per_angstrom"]
        ) != len(species):
            diagnostics.append(
                _diagnostic(
                    "error",
                    f"{calc_id}.result.frame_shape",
                    f"{calc_id} 的第 {position} 帧原子数与 species 不一致。",
                )
            )

    ionic_converged: bool | None = None
    if calculation_type == STATIC_TYPE:
        if len(frames) != 1:
            diagnostics.append(
                _diagnostic("error", f"{calc_id}.completion.ionic", f"{calc_id} 是 static，必须恰好一个 ionic step。")
            )
    elif calculation_type == RELAX_TYPE:
        # VASP states ionic convergence explicitly; a zero exit code does not.
        ionic_converged = RELAX_CONVERGENCE_MARKER in outcar
        if not ionic_converged:
            diagnostics.append(
                _diagnostic(
                    "error",
                    f"{calc_id}.completion.ionic",
                    f"{calc_id} 的离子步未收敛：OUTCAR 未报告 reached required accuracy。",
                )
            )
        if requested_nsw is not None and len(frames) > requested_nsw:
            diagnostics.append(
                _diagnostic("error", f"{calc_id}.completion.ionic_steps", f"{calc_id} 的 ionic step 数超过 NSW。")
            )
        contcar = directory / "CONTCAR"
        if not _ordinary_file(contcar) or contcar.stat().st_size == 0:
            diagnostics.append(
                _diagnostic("error", f"{calc_id}.raw.CONTCAR", f"{calc_id} 缺少非空的 CONTCAR。")
            )
        else:
            contcar_lattice = _contcar_lattice(contcar)
            # CONTCAR holds the geometry *after* the final ionic update, which is
            # what vasprun reports as finalpos -- not the geometry the last
            # forces were evaluated at.  Comparing it against the last frame
            # would flag every converged relaxation as inconsistent.
            reference = parsed.get("final_lattice_angstrom") or frames[-1]["lattice_angstrom"]
            if contcar_lattice is None:
                diagnostics.append(
                    _diagnostic("error", f"{calc_id}.result.contcar", f"{calc_id} 的 CONTCAR 无法解析。")
                )
            elif not _same_lattice_across_formats(contcar_lattice, reference):
                diagnostics.append(
                    _diagnostic(
                        "error",
                        f"{calc_id}.result.contcar",
                        f"{calc_id} 的 CONTCAR 晶格与 vasprun finalpos 不一致。",
                    )
                )
    elif calculation_type == AIMD_TYPE:
        # AIMD is not a structural optimisation, so ionic convergence is not a
        # completion criterion; the trajectory being complete is.
        if requested_nsw is None:
            diagnostics.append(
                _diagnostic("error", f"{calc_id}.completion.nsw", f"{calc_id} 的 labeling_config 未声明可用的 NSW。")
            )
        elif len(frames) != requested_nsw:
            diagnostics.append(
                _diagnostic(
                    "error",
                    f"{calc_id}.completion.trajectory",
                    f"{calc_id} 完成 {len(frames)} 个 ionic step，与请求的 NSW={requested_nsw} 不一致。",
                )
            )
    return diagnostics, {
        "calc_id": calc_id,
        "calculation_type": calculation_type,
        "species": species,
        "nelm": nelm,
        "frames": frames,
        "ionic_converged": ionic_converged,
        "final_lattice_angstrom": parsed.get("final_lattice_angstrom"),
        "final_fractional_coordinates": parsed.get("final_fractional_coordinates"),
        "outcar_text_head": outcar[:4096],
    }


def _same_lattice_across_formats(actual: Any, expected: Any) -> bool:
    """Compare a CONTCAR lattice with a vasprun lattice.

    These are two different files at two different printed precisions: CONTCAR
    carries ~16 significant digits while vasprun.xml prints 8 decimals, so the
    same cell differs by ~1e-9.  The strict 1e-10 tolerance used elsewhere is
    correct when re-parsing one file twice, but here it would reject every real
    relaxation.  The bound below is tied to vasprun's printed precision.
    """

    if not isinstance(actual, list) or not isinstance(expected, list):
        return False
    if len(actual) != len(expected):
        return False
    for actual_row, expected_row in zip(actual, expected):
        if not isinstance(actual_row, list) or len(actual_row) != len(expected_row):
            return False
        for left, right in zip(actual_row, expected_row):
            if not _finite_number(left) or not _finite_number(right):
                return False
            if not math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-6):
                return False
    return True


def _contcar_lattice(path: Path) -> list[list[float]] | None:
    """Read the three lattice vectors from a CONTCAR, scale applied."""

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        scale = float(lines[1].split()[0])
        rows = [[float(x) * scale for x in lines[i].split()[:3]] for i in (2, 3, 4)]
    except (OSError, ValueError, IndexError):
        return None
    if any(len(row) != 3 for row in rows):
        return None
    return rows


def _scheduled_calculations_check(
    context: Mapping[str, Any]
) -> tuple[list[dict[str, str]], list[dict[str, Any]] | None]:
    """Check every calculation in the batch; all must pass for the node to pass."""

    diagnostics: list[dict[str, str]] = []
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    calculation_type = _scheduled_calculation_type(context)
    requested_nsw = _requested_nsw(context)
    prepared, _ = _read_json(
        _path(context["project_root"], _mapping(context["inputs"])["dft_input_manifest"])
    )
    calculations = prepared.get("calculations") if isinstance(prepared, Mapping) else None
    if not isinstance(calculations, list) or not calculations:
        diagnostics.append(_diagnostic("error", "result.calculations", "准备清单缺少 calculations。"))
        return diagnostics, None
    completion_diagnostics = _verify_completion_calculations(context, calculations)
    diagnostics.extend(completion_diagnostics)
    analyses: list[dict[str, Any]] = []
    for index, calculation in enumerate(calculations):
        calc_id = _calculation_id(index)
        expected_atoms = (
            calculation.get("atom_count") if isinstance(calculation, Mapping) else None
        )
        calc_diagnostics, analysis = _check_one_calculation(
            attempt, calc_id, calculation_type, expected_atoms, requested_nsw
        )
        diagnostics.extend(calc_diagnostics)
        if analysis is not None:
            analysis["structure_id"] = (
                calculation.get("structure_id") if isinstance(calculation, Mapping) else None
            )
            analyses.append(analysis)
    if _errors(diagnostics):
        return diagnostics, None
    return diagnostics, analyses


def _verify_completion_calculations(
    context: Mapping[str, Any], calculations: list[Any]
) -> list[dict[str, str]]:
    """Cross-check the remote completion record against the approved batch.

    The core only guarantees project/node/attempt identity and the overall exit
    status; the per-calculation list is a plugin-level contract, so it is
    verified here rather than in the scheduler lifecycle.
    """

    diagnostics: list[dict[str, str]] = []
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    completion, _ = _read_json(attempt / "completion.json")
    if not isinstance(completion, Mapping):
        return diagnostics
    reported = completion.get("calculations")
    if reported is None:
        return diagnostics
    expected_ids = [_calculation_id(index) for index in range(len(calculations))]
    if not isinstance(reported, list) or [
        item.get("id") if isinstance(item, Mapping) else None for item in reported
    ] != expected_ids:
        diagnostics.append(
            _diagnostic("error", "completion.calculations", "completion.json 的 calculation 列表与批准计划不一致。")
        )
        return diagnostics
    for item in reported:
        if item.get("exit_code") != 0:
            diagnostics.append(
                _diagnostic(
                    "error",
                    f"{item.get('id')}.completion.exit_code",
                    f"{item.get('id')} 的 VASP 进程退出码为 {item.get('exit_code')!r}。",
                )
            )
    return diagnostics


def _write_fresh_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"refusing to overwrite result: {path}")
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _collect_scheduled_result(context: Mapping[str, Any]) -> dict[str, Any]:
    """Build the labelled dataset for a whole batch.

    Label cardinality follows the calculation type: static and relax yield one
    label per source structure (the single, respectively final converged,
    configuration), while AIMD yields one label per ionic step.  Every label
    carries back-references so a row can be traced to its source structure,
    calculation and ionic step.
    """

    diagnostics, analyses = _scheduled_calculations_check(context)
    if _errors(diagnostics) or analyses is None:
        return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    parameters = _mapping(context["parameters"])
    inputs = _mapping(context["inputs"])
    calculation_type = _scheduled_calculation_type(context)

    records: list[dict[str, Any]] = []
    frame_count = 0
    for analysis in analyses:
        frames = analysis["frames"]
        selected = (
            list(enumerate(frames, start=1))
            if calculation_type == AIMD_TYPE
            # static has one frame; relax labels only the final converged
            # configuration.  Intermediate relaxation steps are deliberately not
            # treated as training labels in this change.
            else [(len(frames), frames[-1])]
        )
        frame_count += len(frames)
        for ionic_step, frame in selected:
            records.append(
                {
                    "structure_id": analysis.get("structure_id"),
                    "calculation_id": analysis["calc_id"],
                    "calculation_type": calculation_type,
                    "ionic_step": ionic_step,
                    "species": analysis["species"],
                    "energy_ev": frame["energy_ev"],
                    "lattice_angstrom": frame["lattice_angstrom"],
                    "fractional_coordinates": frame["fractional_coordinates"],
                    "forces_ev_per_angstrom": frame["forces_ev_per_angstrom"],
                    "stress_kbar_vasp_3x3": frame["stress_kbar_vasp_3x3"],
                }
            )
    labels_path = attempt / "labels.json"
    _write_fresh_json(
        labels_path,
        {"schema_version": 2, "records": records, "units": parameters["units"]},
    )

    raw_outputs: dict[str, Any] = {}
    required, optional, _ = _scheduled_output_spec(calculation_type)
    for analysis in analyses:
        calc_id = analysis["calc_id"]
        for name in (*required, *optional):
            path = attempt / calc_id / name
            if not _ordinary_file(path):
                continue
            raw_outputs[f"{calc_id}/{name}"] = {
                "path": f"{calc_id}/{name}",
                "fingerprint": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
    first = analyses[0]
    version = re.search(r"\bvasp\.([0-9][A-Za-z0-9._-]*)", first["outcar_text_head"], re.I)
    ionic_required = calculation_type == RELAX_TYPE
    ionic_converged = (
        all(item["ionic_converged"] is True for item in analyses) if ionic_required else None
    )
    execution_context = _mapping(context.get("execution"))
    hpc_execution = _mapping(execution_context.get("hpc_execution"))
    manifest = {
        "schema_version": 2,
        "plugin_id": PLUGIN_ID,
        "status": "OK",
        "engine": "vasp",
        "calculation_type": calculation_type,
        "input_fingerprints": {
            name: _sha256(_path(context["project_root"], inputs[name]))
            for name in ("structures_manifest", "labeling_config", "dft_input_manifest")
        },
        "completion": {
            "scheduler_success": True,
            "electronic_converged": True,
            "ionic_convergence_required": ionic_required,
            "ionic_converged": ionic_converged,
            "truncated": False,
        },
        "units": parameters["units"],
        "source_count": len(analyses),
        "calculation_count": len(analyses),
        "frame_count": frame_count,
        "label_count": len(records),
        "labels": "labels.json",
        "execution": {
            "template_family": "vasp",
            "templates": hpc_execution.get("templates"),
            "vasp_version": version.group(1) if version else "UNKNOWN",
            "calculations": [
                {
                    "id": item["calc_id"],
                    "structure_id": item.get("structure_id"),
                    "nelm": item["nelm"],
                    "ionic_steps": len(item["frames"]),
                    "ionic_converged": item["ionic_converged"],
                    # For relax, the labelled frame is the last force evaluation
                    # while the relaxed geometry is the subsequent finalpos; both
                    # identities are recorded so a row stays traceable.
                    "relaxed_structure": (
                        {
                            "lattice_angstrom": item["final_lattice_angstrom"],
                            "fractional_coordinates": item["final_fractional_coordinates"],
                        }
                        if calculation_type == RELAX_TYPE
                        else None
                    ),
                }
                for item in analyses
            ],
        },
        "raw_outputs": raw_outputs,
        "artifacts": [
            {
                "name": "labels-json",
                "path": "labels.json",
                "media_type": "application/json",
                "fingerprint": _sha256(labels_path),
            }
        ],
    }
    result_path = attempt / str(parameters.get("result_manifest", "dft-labeling-result.json"))
    _write_fresh_json(result_path, manifest)
    diagnostics.extend(_verify_scheduled_label_result(context, manifest))
    if _errors(diagnostics):
        return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
    return {
        "plugin_id": PLUGIN_ID,
        "status": "OK",
        "diagnostics": diagnostics,
        "result_manifest": str(result_path),
        "metrics": {
            "source_count": float(len(analyses)),
            "calculation_count": float(len(analyses)),
            "frame_count": float(frame_count),
            "label_count": float(len(records)),
        },
        "artifacts": [
            {"path": "labels.json", "role": "labels", "media_type": "application/json"},
            {
                "path": str(result_path.name),
                "role": "dft-label-result",
                "media_type": "application/json",
            },
        ],
    }


def _verify_scheduled_label_result(
    context: Mapping[str, Any], manifest: Mapping[str, Any]
) -> list[dict[str, str]]:
    """Schema-2 verification of a scheduled result manifest.

    Kept separate from the local wrapper contract (schema 1), whose
    "one label per source structure" rule is only correct for single static
    calculations and is wrong for AIMD.
    """

    diagnostics: list[dict[str, str]] = []
    parameters = _mapping(context["parameters"])
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    calculation_type = manifest.get("calculation_type")
    if calculation_type not in SCHEDULED_TYPES:
        diagnostics.append(_diagnostic("error", "result.calculation_type", "calculation_type 必须显式且受支持。"))
        return diagnostics
    if manifest.get("units") != parameters.get("units"):
        diagnostics.append(_diagnostic("error", "result.units", "结果单位与批准 units 不一致。"))
    completion = _mapping(manifest.get("completion"))
    if completion.get("scheduler_success") is not True or completion.get("electronic_converged") is not True:
        diagnostics.append(_diagnostic("error", "result.completion", "completion 状态不安全。"))
    if calculation_type == RELAX_TYPE and completion.get("ionic_converged") is not True:
        diagnostics.append(_diagnostic("error", "completion.ionic", "relax 结果必须报告 ionic_converged=true。"))
    if calculation_type != RELAX_TYPE and completion.get("ionic_convergence_required") is not False:
        diagnostics.append(_diagnostic("error", "completion.ionic_policy", "仅 relax 要求 ionic convergence。"))
    source_count = manifest.get("source_count")
    calculation_count = manifest.get("calculation_count")
    frame_count = manifest.get("frame_count")
    label_count = manifest.get("label_count")
    if not _positive_int(source_count) or calculation_count != source_count:
        diagnostics.append(_diagnostic("error", "result.calculation_count", "calculation_count 必须等于 source_count。"))
    if not _positive_int(frame_count) or not _positive_int(label_count):
        diagnostics.append(_diagnostic("error", "result.counts", "frame_count/label_count 必须是正整数。"))
    elif calculation_type == AIMD_TYPE:
        if label_count != frame_count:
            diagnostics.append(_diagnostic("error", "result.label_count", "AIMD 每个 frame 必须恰有一个标签。"))
    elif label_count != source_count:
        diagnostics.append(_diagnostic("error", "result.label_count", "static/relax 每个源结构必须恰有一个标签。"))
    labels_path = attempt / str(manifest.get("labels", "labels.json"))
    labels, _ = _read_json(labels_path)
    records = labels.get("records") if isinstance(labels, Mapping) else None
    if not isinstance(records, list) or len(records) != label_count:
        diagnostics.append(_diagnostic("error", "result.labels", "labels.json 记录数与 label_count 不一致。"))
        return diagnostics
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            diagnostics.append(_diagnostic("error", f"labels[{index}]", "标签必须是对象。"))
            continue
        if record.get("calculation_type") != calculation_type or not _plain_string(
            record.get("calculation_id")
        ):
            diagnostics.append(
                _diagnostic("error", f"labels[{index}].traceability", "标签必须记录 calculation id 与 calculation_type。")
            )
        if not _positive_int(record.get("ionic_step")):
            diagnostics.append(_diagnostic("error", f"labels[{index}].ionic_step", "标签必须记录 ionic step。"))
        if not _finite_number(record.get("energy_ev")):
            diagnostics.append(_diagnostic("error", f"labels[{index}].energy", "标签能量必须是有限数。"))
    raw = manifest.get("raw_outputs")
    if not isinstance(raw, Mapping) or not raw:
        diagnostics.append(_diagnostic("error", "result.raw_outputs", "必须声明 raw VASP 输出。"))
        return diagnostics
    for name, record in raw.items():
        if "POTCAR" in str(name):
            diagnostics.append(_diagnostic("error", "result.raw_outputs.potcar", "POTCAR 不得出现在结果产物中。"))
            continue
        path = attempt / str(name)
        if (
            not isinstance(record, Mapping)
            or record.get("path") != name
            or not _fingerprint(record.get("fingerprint"))
            or not _ordinary_file(path)
            or _sha256(path) != record.get("fingerprint")
        ):
            diagnostics.append(_diagnostic("error", f"result.raw_outputs.{name}", "原始 VASP 输出记录无效。"))
    return diagnostics


class Adapter:
    """Plan one reviewed operation and verify its standardized result."""

    def validate(self, context: Any) -> list[dict[str, str]]:
        if not isinstance(context, Mapping):
            return _base_diagnostics(context)
        if _operation(context) == PREPARE_OPERATION:
            return _validate_prepare(context)
        if _operation(context) == LABEL_OPERATION:
            return _validate_label(context)
        return _base_diagnostics(context)

    def plan(self, context: Any) -> dict[str, Any]:
        if not isinstance(context, Mapping):
            return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": self.validate(context)}
        if _operation(context) == PREPARE_OPERATION:
            return _plan_prepare(context)
        if _operation(context) == LABEL_OPERATION:
            return _plan_label(context)
        return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": self.validate(context)}

    def prepare(self, context: Any, plan: Any) -> dict[str, Any]:
        if not isinstance(plan, Mapping) or plan.get("status") != "READY":
            return {
                "plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False,
                "diagnostics": [_diagnostic("error", "plan.not_ready", "prepare 需要 READY 计划。")],
            }
        return dict(plan)

    def _result_path(self, context: Mapping[str, Any]) -> Path:
        parameters = _mapping(context["parameters"])
        default = "dft-input-manifest.json" if _operation(context) == PREPARE_OPERATION else "dft-labeling-result.json"
        return _path(context["attempt_dir"], parameters.get("result_manifest", default))

    def _verify_result(
        self, context: Mapping[str, Any], manifest: Mapping[str, Any], verify_files: bool
    ) -> list[dict[str, str]]:
        if _operation(context) == PREPARE_OPERATION:
            return _verify_prepare_result(context, manifest, verify_files)
        return _verify_label_result(context, manifest, verify_files)

    def check(self, context: Any) -> dict[str, Any]:
        diagnostics = (
            _validate_prepare(context, require_fresh_outputs=False)
            if isinstance(context, Mapping) and _operation(context) == PREPARE_OPERATION
            else self.validate(context)
        )
        if _errors(diagnostics):
            return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
        assert isinstance(context, Mapping)
        if context.get("backend") == "ssh-slurm" and _operation(context) == LABEL_OPERATION:
            raw_diagnostics, _ = _scheduled_calculations_check(context)
            diagnostics.extend(raw_diagnostics)
            return {
                "plugin_id": PLUGIN_ID,
                "status": "FAIL" if _errors(diagnostics) else "OK",
                "diagnostics": diagnostics,
            }
        result_path = self._result_path(context)
        manifest, read_diagnostic = _read_json(result_path)
        if manifest is None:
            assert read_diagnostic is not None
            status = "WAIT" if read_diagnostic["level"] == "warning" else "FAIL"
            return {"plugin_id": PLUGIN_ID, "status": status, "diagnostics": [read_diagnostic]}
        diagnostics.extend(self._verify_result(context, manifest, verify_files=True))
        return {
            "plugin_id": PLUGIN_ID,
            "status": "FAIL" if _errors(diagnostics) else "OK",
            "diagnostics": diagnostics,
            "result_manifest": str(result_path),
        }

    def collect(self, context: Any) -> dict[str, Any]:
        checked = self.check(context)
        if checked["status"] != "OK":
            return checked
        assert isinstance(context, Mapping)
        if context.get("backend") == "ssh-slurm" and _operation(context) == LABEL_OPERATION:
            return _collect_scheduled_result(context)
        result_path = self._result_path(context)
        manifest, _ = _read_json(result_path)
        assert manifest is not None
        parameters = _mapping(context["parameters"])
        if _operation(context) == PREPARE_OPERATION:
            artifacts: list[dict[str, Any]] = [
                {
                    "name": "dft-input-manifest",
                    "path": parameters.get("result_manifest", "dft-input-manifest.json"),
                    "media_type": "application/json",
                }
            ]
            for calculation in manifest["calculations"]:
                for name in ("POSCAR", "INCAR", "KPOINTS"):
                    record = dict(calculation["files"][name])
                    record["name"] = f"{calculation['structure_id']}-{name.lower()}"
                    artifacts.append(record)
            return {
                "plugin_id": PLUGIN_ID,
                "status": "OK",
                "artifacts": artifacts,
                "metrics": {
                    "structure_count": manifest["structure_count"],
                    "calculation_type": manifest["calculation_type"],
                    "potcar_ready": True,
                    "potcar_collected": False,
                },
            }
        artifacts = [
            {
                "name": "dft-labeling-result",
                "path": parameters.get("result_manifest", "dft-labeling-result.json"),
                "media_type": "application/json",
            },
            *[dict(item) for item in manifest["artifacts"]],
        ]
        return {
            "plugin_id": PLUGIN_ID,
            "status": "OK",
            "artifacts": artifacts,
            "metrics": {
                "source_structure_count": manifest["source_structure_count"],
                "label_count": manifest["label_count"],
                "electronic_converged": True,
                "ionic_converged": manifest["completion"].get("ionic_converged"),
            },
        }

    def replay(self, context: Any) -> dict[str, Any]:
        """Verify a provided standardized result without launching any executable."""

        if isinstance(context, Mapping) and isinstance(context.get("result_manifest"), Mapping):
            diagnostics = (
                _validate_prepare(context, require_fresh_outputs=False)
                if _operation(context) == PREPARE_OPERATION
                else self.validate(context)
            )
            diagnostics.extend(self._verify_result(context, context["result_manifest"], False))
            return {
                "plugin_id": PLUGIN_ID,
                "status": "FAIL" if _errors(diagnostics) else "OK",
                "executable": False,
                "diagnostics": diagnostics,
                "result_manifest": context["result_manifest"],
            }
        collected = self.collect(context)
        collected["executable"] = False
        return collected
