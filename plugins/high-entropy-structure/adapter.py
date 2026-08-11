"""Safe adapter for user-owned, seeded SQS structure generators.

The adapter never implements an SQS objective and never imports a generator.  It
only validates an explicit command contract, builds an argv list (``shell=False``
is part of the plugin manifest), and verifies declared output artifacts.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PLUGIN_ID = "high-entropy-structure"
BUILTIN_SQS = Path(globals().get("__file__", "adapter.py")).absolute().with_name("sqs.py")
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


def _error_diagnostics(diagnostics: list[dict[str, str]]) -> list[dict[str, str]]:
    return [item for item in diagnostics if item["level"] == "error"]


def _safe_relative(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip() or "\x00" in value or "\n" in value:
        return False
    path = Path(value)
    return path != Path(".") and not path.is_absolute() and ".." not in path.parts


def _path(root: str, relative: str) -> Path:
    return Path(root).expanduser().absolute() / relative


def _context_diagnostics(context: Any) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    if not isinstance(context, dict):
        return [_diagnostic("error", "context.type", "context 必须是对象。")]
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
                "当前适配器仅验证过 local；SLURM 完成后尚未接入插件科学检查。",
            )
        )
    return diagnostics


def _load_json(path: Path) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
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
    """Plan a user generator and verify its immutable structure manifest."""

    def validate(self, context: Any) -> list[dict[str, str]]:
        diagnostics = _context_diagnostics(context)
        if not isinstance(context, dict):
            return diagnostics
        inputs = context.get("inputs")
        parameters = context.get("parameters")
        if not isinstance(inputs, dict) or not isinstance(parameters, dict):
            return diagnostics

        for key in ("prototype_structure", "composition_manifest"):
            if not _safe_relative(inputs.get(key)):
                diagnostics.append(
                    _diagnostic(
                        "error",
                        f"inputs.{key}",
                        f"{key} 必须是 project_root 下不含 '..' 的相对路径。",
                    )
                )
        if parameters.get("sqs_script") is not None and not _safe_relative(
            parameters.get("sqs_script")
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "parameters.sqs_script",
                    "sqs_script 若提供，必须是 project_root 下显式的相对脚本路径。",
                )
            )
        interpreter = parameters.get("interpreter_argv", [sys.executable])
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
                    "interpreter_argv 必须只含一个显式可执行文件，例如 ['python']。",
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
                    "不接受 extra_args；固定输出、seed 和候选上限不得被重复参数覆盖。",
                )
            )
        seed = parameters.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            diagnostics.append(
                _diagnostic("error", "parameters.seed", "seed 必须是显式的非负整数。")
            )
        max_candidates = parameters.get("max_candidates")
        if (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or max_candidates < 1
        ):
            diagnostics.append(
                _diagnostic(
                    "error", "parameters.max_candidates", "max_candidates 必须是正整数硬上限。"
                )
            )
        for key, default in (
            ("structures_dir", "structures"),
            ("result_manifest", "generation-result.json"),
        ):
            if not _safe_relative(parameters.get(key, default)):
                diagnostics.append(
                    _diagnostic(
                        "error", f"parameters.{key}", f"{key} 必须是 attempt_dir 下的安全相对路径。"
                    )
                )
        return diagnostics

    def plan(self, context: Any) -> dict[str, Any]:
        diagnostics = self.validate(context)
        if _error_diagnostics(diagnostics):
            return {
                "plugin_id": PLUGIN_ID,
                "status": "BLOCKED",
                "executable": False,
                "diagnostics": diagnostics,
            }
        assert isinstance(context, dict)
        inputs = context["inputs"]
        parameters = context["parameters"]
        project_root = context["project_root"]
        attempt_dir = context["attempt_dir"]
        output_dir = _path(attempt_dir, parameters.get("structures_dir", "structures"))
        result_path = _path(
            attempt_dir, parameters.get("result_manifest", "generation-result.json")
        )
        script_value = parameters.get("sqs_script")
        script_path = (
            _path(project_root, script_value)
            if isinstance(script_value, str)
            else BUILTIN_SQS.absolute()
        )
        argv = list(parameters.get("interpreter_argv", [sys.executable]))
        argv.extend(
            [
                str(script_path),
                "--prototype",
                str(_path(project_root, inputs["prototype_structure"])),
                "--composition-manifest",
                str(_path(project_root, inputs["composition_manifest"])),
                "--seed",
                str(parameters["seed"]),
                "--max-candidates",
                str(parameters["max_candidates"]),
                "--output-dir",
                str(output_dir),
                "--result-manifest",
                str(result_path),
            ]
        )
        return {
            "plugin_id": PLUGIN_ID,
            "status": "READY",
            "executable": True,
            "argv": argv,
            "cwd": str(Path(attempt_dir).expanduser().absolute()),
            "expected_outputs": [str(result_path), str(output_dir)],
            "diagnostics": diagnostics,
            "provenance": {
                "seed": parameters["seed"],
                "max_candidates": parameters["max_candidates"],
                "generator_is_user_supplied": script_value is not None,
                "generator_path": str(script_path),
                "generator_fingerprint": _sha256(script_path) if script_path.is_file() else None,
            },
        }

    def prepare(self, context: Any, plan: Any) -> dict[str, Any]:
        """Return the already-materialized plan; never create input files here."""

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
        parameters = context["parameters"]
        return _path(
            context["attempt_dir"], parameters.get("result_manifest", "generation-result.json")
        )

    def _verify_result(
        self, context: dict[str, Any], manifest: dict[str, Any], verify_files: bool
    ) -> list[dict[str, str]]:
        diagnostics: list[dict[str, str]] = []
        parameters = context["parameters"]
        if manifest.get("schema_version") != 1:
            diagnostics.append(
                _diagnostic("error", "result.schema_version", "schema_version 必须为 1。")
            )
        if manifest.get("plugin_id") != PLUGIN_ID:
            diagnostics.append(_diagnostic("error", "result.plugin_id", "结果 plugin_id 不匹配。"))
        if manifest.get("status") != "OK":
            diagnostics.append(_diagnostic("error", "result.status", "生成器未声明 status=OK。"))
        if manifest.get("seed") != parameters.get("seed"):
            diagnostics.append(
                _diagnostic("error", "result.seed", "结果 seed 与已批准计划不一致。")
            )
        declared_inputs = {
            "prototype_fingerprint": manifest.get("prototype_fingerprint"),
            "composition_manifest_fingerprint": manifest.get("composition_manifest_fingerprint"),
        }
        for key, fingerprint in declared_inputs.items():
            if not isinstance(fingerprint, str) or not fingerprint.startswith("sha256:"):
                diagnostics.append(
                    _diagnostic("error", f"result.{key}", f"必须声明输入 {key} 的 sha256。")
                )
        if verify_files:
            input_paths = {
                "prototype_fingerprint": _path(
                    context["project_root"], context["inputs"]["prototype_structure"]
                ),
                "composition_manifest_fingerprint": _path(
                    context["project_root"], context["inputs"]["composition_manifest"]
                ),
            }
            for key, input_path in input_paths.items():
                if not input_path.is_file():
                    diagnostics.append(
                        _diagnostic("error", f"result.{key}", f"输入文件不存在：{input_path}")
                    )
                elif _sha256(input_path) != declared_inputs[key]:
                    diagnostics.append(
                        _diagnostic("error", f"result.{key}", f"输入 {key} 的 sha256 不匹配。")
                    )
        structures = manifest.get("structures")
        if not isinstance(structures, list) or not structures:
            diagnostics.append(
                _diagnostic("error", "result.structures", "structures 必须是非空数组。")
            )
            return diagnostics
        if len(structures) > parameters.get("max_candidates", 0):
            diagnostics.append(
                _diagnostic("error", "result.count_limit", "结构数量超过 max_candidates 硬上限。")
            )
        seen: set[str] = set()
        for index, structure in enumerate(structures):
            prefix = f"structures[{index}]"
            if not isinstance(structure, dict):
                diagnostics.append(_diagnostic("error", f"{prefix}.type", "结构记录必须是对象。"))
                continue
            structure_id = structure.get("id")
            if not isinstance(structure_id, str) or not structure_id:
                diagnostics.append(_diagnostic("error", f"{prefix}.id", "结构 id 不能为空。"))
            elif structure_id in seen:
                diagnostics.append(_diagnostic("error", f"{prefix}.id", "结构 id 必须唯一。"))
            else:
                seen.add(structure_id)
            relative = structure.get("path")
            fingerprint = structure.get("fingerprint")
            if not _safe_relative(relative):
                diagnostics.append(
                    _diagnostic("error", f"{prefix}.path", "结构路径必须位于 attempt_dir 内。")
                )
                continue
            if not isinstance(fingerprint, str) or not fingerprint.startswith("sha256:"):
                diagnostics.append(
                    _diagnostic("error", f"{prefix}.fingerprint", "必须声明 sha256 指纹。")
                )
                continue
            if not isinstance(structure.get("composition"), dict):
                diagnostics.append(
                    _diagnostic("error", f"{prefix}.composition", "必须声明 composition 对象。")
                )
            if verify_files:
                artifact_path = _path(context["attempt_dir"], relative)
                if not artifact_path.is_file():
                    diagnostics.append(
                        _diagnostic(
                            "error", f"{prefix}.missing", f"结构文件不存在：{artifact_path}"
                        )
                    )
                elif artifact_path.stat().st_size == 0:
                    diagnostics.append(_diagnostic("error", f"{prefix}.empty", "结构文件为空。"))
                elif _sha256(artifact_path) != fingerprint:
                    diagnostics.append(
                        _diagnostic("error", f"{prefix}.fingerprint", "结构文件 sha256 不匹配。")
                    )
        return diagnostics

    def check(self, context: Any) -> dict[str, Any]:
        diagnostics = self.validate(context)
        if _error_diagnostics(diagnostics):
            return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
        assert isinstance(context, dict)
        path = self._result_path(context)
        manifest, read_diagnostic = _load_json(path)
        if manifest is None:
            assert read_diagnostic is not None
            status = "WAIT" if read_diagnostic["level"] == "warning" else "FAIL"
            return {"plugin_id": PLUGIN_ID, "status": status, "diagnostics": [read_diagnostic]}
        diagnostics.extend(self._verify_result(context, manifest, verify_files=True))
        return {
            "plugin_id": PLUGIN_ID,
            "status": "FAIL" if _error_diagnostics(diagnostics) else "OK",
            "diagnostics": diagnostics,
            "result_manifest": str(path),
        }

    def collect(self, context: Any) -> dict[str, Any]:
        checked = self.check(context)
        if checked["status"] != "OK":
            return checked
        assert isinstance(context, dict)
        result_path = self._result_path(context)
        manifest, _ = _load_json(result_path)
        assert manifest is not None
        artifacts = [
            {
                "name": "generation-manifest",
                "path": context["parameters"].get("result_manifest", "generation-result.json"),
                "media_type": "application/json",
            }
        ]
        for structure in manifest["structures"]:
            artifacts.append(
                {
                    "name": structure["id"],
                    "path": structure["path"],
                    "media_type": structure.get("media_type", "chemical/x-cif"),
                    "fingerprint": structure["fingerprint"],
                }
            )
        return {
            "plugin_id": PLUGIN_ID,
            "status": "OK",
            "artifacts": artifacts,
            "metrics": {"structure_count": len(manifest["structures"]), "seed": manifest["seed"]},
        }

    def replay(self, context: Any) -> dict[str, Any]:
        """Verify existing artifacts only; never invoke the user generator."""

        if not isinstance(context, dict):
            return {
                "plugin_id": PLUGIN_ID,
                "status": "FAIL",
                "executable": False,
                "diagnostics": [_diagnostic("error", "context.type", "context 必须是对象。")],
            }
        inline = context.get("result_manifest")
        if isinstance(inline, dict):
            diagnostics = self.validate(context)
            diagnostics.extend(self._verify_result(context, inline, verify_files=False))
            return {
                "plugin_id": PLUGIN_ID,
                "status": "FAIL" if _error_diagnostics(diagnostics) else "OK",
                "executable": False,
                "diagnostics": diagnostics,
                "result_manifest": inline,
            }
        collected = self.collect(context)
        collected["executable"] = False
        return collected
