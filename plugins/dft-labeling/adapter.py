"""Thin adapter for user-owned DFT preparation and labeling commands.

No VASP output is heuristically accepted here.  Scientific completion is
established only by a standard result manifest written by the reviewed user
command, including process success, convergence, non-truncation, units and
immutable dataset artifacts.

This module never calls ``sbatch`` or another submission API; job ownership is
reserved for the MLIPFlow core backend.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PLUGIN_ID = "dft-labeling"
SHELL_EXECUTABLES = frozenset(
    {"bash", "csh", "cmd", "dash", "fish", "ksh", "powershell", "pwsh", "sh", "tcsh", "zsh"}
)
CONTEXT_KEYS = (
    "project_root",
    "attempt_dir",
    "inputs",
    "parameters",
    "backend",
    "resources",
)


def _diagnostic(level: str, code: str, message: str) -> dict[str, str]:
    return {"level": level, "code": code, "message": message}


def _errors(diagnostics: list[dict[str, str]]) -> list[dict[str, str]]:
    return [item for item in diagnostics if item["level"] == "error"]


def _safe_relative(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip() or "\x00" in value or "\n" in value:
        return False
    path = Path(value)
    return path != Path(".") and not path.is_absolute() and ".." not in path.parts


def _path(root: str, relative: str) -> Path:
    return Path(root).expanduser().absolute() / relative


def _base_diagnostics(context: Any) -> list[dict[str, str]]:
    if not isinstance(context, dict):
        return [_diagnostic("error", "context.type", "context 必须是对象。")]
    diagnostics: list[dict[str, str]] = []
    missing = [key for key in CONTEXT_KEYS if key not in context]
    if missing:
        diagnostics.append(
            _diagnostic("error", "context.keys", f"缺少上下文字段：{', '.join(missing)}。")
        )
    for key in ("project_root", "attempt_dir"):
        if key in context and (not isinstance(context[key], str) or not context[key].strip()):
            diagnostics.append(_diagnostic("error", f"context.{key}", f"{key} 必须是路径字符串。"))
    for key in ("inputs", "parameters", "resources"):
        if key in context and not isinstance(context[key], dict):
            diagnostics.append(_diagnostic("error", f"context.{key}", f"{key} 必须是对象。"))
    if "backend" in context and context["backend"] != "local":
        diagnostics.append(
            _diagnostic(
                "error",
                "backend.unsupported",
                "当前仅启用 local；调度提交必须由核心托管，插件不会自行 sbatch。",
            )
        )
    return diagnostics


def _read_json(path: Path) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    if not path.is_file():
        return None, _diagnostic("warning", "artifact.missing", f"结果清单尚不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, _diagnostic("error", "artifact.invalid_json", f"无法读取 JSON：{exc}")
    if not isinstance(value, dict):
        return None, _diagnostic("error", "artifact.not_object", "结果清单必须是 JSON 对象。")
    return value, None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


class Adapter:
    """Build one approved script command and verify its DFT result manifest."""

    def validate(self, context: Any) -> list[dict[str, str]]:
        diagnostics = _base_diagnostics(context)
        if not isinstance(context, dict):
            return diagnostics
        inputs = context.get("inputs")
        parameters = context.get("parameters")
        if not isinstance(inputs, dict) or not isinstance(parameters, dict):
            return diagnostics

        for key in ("structures_manifest", "labeling_config"):
            if not _safe_relative(inputs.get(key)):
                diagnostics.append(
                    _diagnostic(
                        "error", f"inputs.{key}", f"{key} 必须是 project_root 下的安全相对路径。"
                    )
                )
        operation = parameters.get("operation")
        if operation != "label":
            diagnostics.append(
                _diagnostic(
                    "error",
                    "parameters.operation",
                    "当前可执行 operation 仅为 label；输入准备可通过显式 prepare_script 交给同一受审命令。",
                )
            )
        if not _safe_relative(parameters.get("label_script")):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "parameters.label_script",
                    "label_script 必须是 project_root 下显式的相对脚本路径。",
                )
            )
        prepare_script = parameters.get("prepare_script")
        if prepare_script is not None and not _safe_relative(prepare_script):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "parameters.prepare_script",
                    "prepare_script 必须是 project_root 下显式的安全相对路径。",
                )
            )
        interpreter = parameters.get("interpreter_argv")
        if not (
            isinstance(interpreter, list)
            and len(interpreter) == 1
            and all(
                isinstance(item, str) and item and "\x00" not in item and "\n" not in item
                for item in interpreter
            )
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "parameters.interpreter_argv",
                    "interpreter_argv 必须只含一个显式可执行文件；不会交给 shell。",
                )
            )
        elif Path(interpreter[0]).name.lower() in SHELL_EXECUTABLES:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "parameters.shell_forbidden",
                    "interpreter_argv 不得选择 shell 或命令字符串模式。",
                )
            )
        if "extra_args" in parameters:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "parameters.extra_args",
                    "不接受 extra_args；固定输入、引擎和结果清单不得被重复参数覆盖。",
                )
            )
        engine = parameters.get("engine")
        if not isinstance(engine, str) or not engine.strip():
            diagnostics.append(_diagnostic("error", "parameters.engine", "engine 必须显式声明。"))
        if not _safe_relative(parameters.get("result_manifest", "dft-labeling-result.json")):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "parameters.result_manifest",
                    "result_manifest 必须是 attempt_dir 下的安全相对路径。",
                )
            )
        completion_policy = parameters.get("completion_policy")
        if not isinstance(completion_policy, dict):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "parameters.completion_policy",
                    "completion_policy 必须显式声明电子/离子收敛要求。",
                )
            )
        elif type(completion_policy.get("require_ionic_convergence")) is not bool:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "parameters.completion_policy",
                    "completion_policy.require_ionic_convergence 必须是布尔值。",
                )
            )
        units = parameters.get("units")
        required_units = {"energy", "length", "force", "stress"}
        if not isinstance(units, dict) or any(
            not isinstance(units.get(key), str) or not units.get(key) for key in required_units
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "parameters.units",
                    "units 必须声明 energy、length、force、stress 四种单位。",
                )
            )
        return diagnostics

    def plan(self, context: Any) -> dict[str, Any]:
        diagnostics = self.validate(context)
        if _errors(diagnostics):
            return {
                "plugin_id": PLUGIN_ID,
                "status": "BLOCKED",
                "executable": False,
                "diagnostics": diagnostics,
            }
        assert isinstance(context, dict)
        inputs = context["inputs"]
        parameters = context["parameters"]
        operation = parameters["operation"]
        script = parameters["label_script"]
        result_path = _path(
            context["attempt_dir"],
            parameters.get("result_manifest", "dft-labeling-result.json"),
        )
        argv = list(parameters["interpreter_argv"])
        argv.extend(
            [
                str(_path(context["project_root"], script)),
                "--operation",
                operation,
                "--structures-manifest",
                str(_path(context["project_root"], inputs["structures_manifest"])),
                "--labeling-config",
                str(_path(context["project_root"], inputs["labeling_config"])),
                "--attempt-dir",
                str(Path(context["attempt_dir"]).expanduser().absolute()),
                "--result-manifest",
                str(result_path),
                "--engine",
                parameters["engine"],
            ]
        )
        if parameters.get("prepare_script") is not None:
            argv.extend(
                [
                    "--prepare-script",
                    str(_path(context["project_root"], parameters["prepare_script"])),
                ]
            )
        return {
            "plugin_id": PLUGIN_ID,
            "status": "READY",
            "executable": True,
            "argv": argv,
            "cwd": str(Path(context["attempt_dir"]).expanduser().absolute()),
            "expected_outputs": [str(result_path)],
            "diagnostics": diagnostics,
            "operation": operation,
            "approval_summary": {
                "expensive": operation == "label",
                "engine": parameters["engine"],
                "backend": context["backend"],
                "resources": context["resources"],
                "completion_policy": parameters["completion_policy"],
            },
        }

    def prepare(self, context: Any, plan: Any) -> dict[str, Any]:
        if not isinstance(plan, dict) or plan.get("status") != "READY":
            return {
                "plugin_id": PLUGIN_ID,
                "status": "BLOCKED",
                "executable": False,
                "diagnostics": [
                    _diagnostic("error", "plan.not_ready", "prepare 需要 READY 计划。")
                ],
            }
        return dict(plan)

    def _result_path(self, context: dict[str, Any]) -> Path:
        return _path(
            context["attempt_dir"],
            context["parameters"].get("result_manifest", "dft-labeling-result.json"),
        )

    def _verify_result(
        self, context: dict[str, Any], manifest: dict[str, Any], verify_files: bool
    ) -> list[dict[str, str]]:
        diagnostics: list[dict[str, str]] = []
        parameters = context["parameters"]
        if manifest.get("schema_version") != 1:
            diagnostics.append(_diagnostic("error", "result.schema_version", "schema_version 必须为 1。"))
        if manifest.get("plugin_id") != PLUGIN_ID:
            diagnostics.append(_diagnostic("error", "result.plugin_id", "结果 plugin_id 不匹配。"))
        if manifest.get("status") != "OK":
            diagnostics.append(_diagnostic("error", "result.status", "DFT 结果未声明 status=OK。"))
        if manifest.get("engine") != parameters.get("engine"):
            diagnostics.append(_diagnostic("error", "result.engine", "结果 engine 与计划不一致。"))

        input_fingerprints = manifest.get("input_fingerprints")
        if not isinstance(input_fingerprints, dict):
            diagnostics.append(
                _diagnostic("error", "result.input_fingerprints", "缺少 input_fingerprints。")
            )
        else:
            input_paths = {
                "structures_manifest": _path(
                    context["project_root"], context["inputs"]["structures_manifest"]
                ),
                "labeling_config": _path(
                    context["project_root"], context["inputs"]["labeling_config"]
                ),
            }
            for key, input_path in input_paths.items():
                declared = input_fingerprints.get(key)
                if not isinstance(declared, str) or not declared.startswith("sha256:"):
                    diagnostics.append(
                        _diagnostic("error", f"result.input_fingerprints.{key}", "必须声明 sha256。")
                    )
                elif verify_files:
                    if not input_path.is_file():
                        diagnostics.append(
                            _diagnostic(
                                "error", f"result.input_fingerprints.{key}", f"输入不存在：{input_path}"
                            )
                        )
                    elif _sha256(input_path) != declared:
                        diagnostics.append(
                            _diagnostic(
                                "error", f"result.input_fingerprints.{key}", "输入 sha256 不匹配。"
                            )
                        )

        completion = manifest.get("completion")
        if not isinstance(completion, dict):
            diagnostics.append(
                _diagnostic("error", "result.completion", "缺少标准 completion 对象。")
            )
        else:
            if completion.get("scheduler_success") is not True:
                diagnostics.append(
                    _diagnostic("error", "completion.scheduler", "调度/进程未成功完成。")
                )
            if completion.get("electronic_converged") is not True:
                diagnostics.append(
                    _diagnostic("error", "completion.electronic", "电子步未收敛，拒绝标签。")
                )
            if completion.get("truncated") is not False:
                diagnostics.append(
                    _diagnostic("error", "completion.truncated", "输出截断状态不安全，拒绝标签。")
                )
            ionic_required = completion.get("ionic_convergence_required")
            ionic_converged = completion.get("ionic_converged")
            if type(ionic_required) is not bool:
                diagnostics.append(
                    _diagnostic(
                        "error", "completion.ionic_policy", "必须声明 ionic_convergence_required。"
                    )
                )
            elif ionic_required is not parameters["completion_policy"].get(
                "require_ionic_convergence"
            ):
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "completion.ionic_policy",
                        "结果的离子收敛要求与已批准 completion_policy 不一致。",
                    )
                )
            elif ionic_required and ionic_converged is not True:
                diagnostics.append(
                    _diagnostic("error", "completion.ionic", "离子步未收敛，拒绝标签。")
                )
            elif not ionic_required and ionic_converged is not True and ionic_converged is not None:
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "completion.ionic",
                        "静态计算的 ionic_converged 应为 null（不适用）或 true，不接受 false。",
                    )
                )

        units = manifest.get("units")
        expected_units = parameters.get("units")
        if not isinstance(units, dict) or units != expected_units:
            diagnostics.append(
                _diagnostic("error", "result.units", "结果单位必须与已批准的 units 完全一致。")
            )
        source_count = manifest.get("source_structure_count")
        label_count = manifest.get("label_count")
        if (
            isinstance(source_count, bool)
            or isinstance(label_count, bool)
            or not isinstance(source_count, int)
            or not isinstance(label_count, int)
            or source_count < 1
            or label_count != source_count
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "result.label_count",
                    "source_structure_count 必须为正整数，且每个结构恰有一个标签。",
                )
            )
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            diagnostics.append(
                _diagnostic("error", "result.artifacts", "必须声明至少一个数据集产物。")
            )
            return diagnostics
        names: set[str] = set()
        for index, artifact in enumerate(artifacts):
            prefix = f"artifacts[{index}]"
            if not isinstance(artifact, dict):
                diagnostics.append(_diagnostic("error", f"{prefix}.type", "产物必须是对象。"))
                continue
            name = artifact.get("name")
            if not isinstance(name, str) or not name or name in names:
                diagnostics.append(
                    _diagnostic("error", f"{prefix}.name", "产物 name 必须非空且唯一。")
                )
            else:
                names.add(name)
            relative = artifact.get("path")
            fingerprint = artifact.get("fingerprint")
            if not _safe_relative(relative):
                diagnostics.append(
                    _diagnostic("error", f"{prefix}.path", "产物路径必须位于 attempt_dir 内。")
                )
                continue
            if not isinstance(fingerprint, str) or not fingerprint.startswith("sha256:"):
                diagnostics.append(
                    _diagnostic("error", f"{prefix}.fingerprint", "产物必须声明 sha256 指纹。")
                )
                continue
            if not isinstance(artifact.get("media_type"), str) or not artifact.get("media_type"):
                diagnostics.append(
                    _diagnostic("error", f"{prefix}.media_type", "产物必须声明 media_type。")
                )
            if verify_files:
                artifact_path = _path(context["attempt_dir"], relative)
                if not artifact_path.is_file() or artifact_path.stat().st_size == 0:
                    diagnostics.append(
                        _diagnostic("error", f"{prefix}.missing", f"数据产物缺失或为空：{artifact_path}")
                    )
                elif _sha256(artifact_path) != fingerprint:
                    diagnostics.append(
                        _diagnostic("error", f"{prefix}.fingerprint", "数据产物 sha256 不匹配。")
                    )
        return diagnostics

    def check(self, context: Any) -> dict[str, Any]:
        diagnostics = self.validate(context)
        if _errors(diagnostics):
            return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
        assert isinstance(context, dict)
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
        assert isinstance(context, dict)
        result_path = self._result_path(context)
        manifest, _ = _read_json(result_path)
        assert manifest is not None
        artifacts = [
            {
                "name": "dft-labeling-result",
                "path": context["parameters"].get(
                    "result_manifest", "dft-labeling-result.json"
                ),
                "media_type": "application/json",
            }
        ]
        for artifact in manifest["artifacts"]:
            item = dict(artifact)
            artifacts.append(item)
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
        """Only verify an existing standardized result; never parse OUTCAR heuristically."""

        if isinstance(context, dict) and isinstance(context.get("result_manifest"), dict):
            diagnostics = self.validate(context)
            diagnostics.extend(
                self._verify_result(context, context["result_manifest"], verify_files=False)
            )
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
