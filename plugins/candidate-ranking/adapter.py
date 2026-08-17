"""Thin adapter for deterministic single-metric candidate ranking.

The ranking implementation may be bundled or project-owned.  This adapter passes
explicit candidate and metric-results manifests to it and independently
checks that the returned ranking obeys the approved metric, direction, missing
value policy and top-k limit.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


PLUGIN_ID = "candidate-ranking"
BUILTIN_RANK = Path(globals().get("__file__", "adapter.py")).absolute().with_name("rank.py")
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


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


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
                "当前适配器仅验证过 local；未声明远端 staging 或 SLURM 科学检查。",
            )
        )
    return diagnostics


def _read_json(path: Path) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    if not path.is_file():
        return None, _diagnostic("warning", "artifact.missing", f"显式产物尚不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, _diagnostic("error", "artifact.invalid_json", f"无法读取 JSON：{exc}")
    if not isinstance(value, dict):
        return None, _diagnostic("error", "artifact.not_object", "清单必须是 JSON 对象。")
    return value, None


class Adapter:
    """Plan a ranking script and verify its top-k evidence."""

    def validate(self, context: Any) -> list[dict[str, str]]:
        diagnostics = _base_diagnostics(context)
        if not isinstance(context, dict):
            return diagnostics
        inputs = context.get("inputs")
        parameters = context.get("parameters")
        if not isinstance(inputs, dict) or not isinstance(parameters, dict):
            return diagnostics
        for key in ("candidate_manifest", "metric_results_manifest"):
            if not _safe_relative(inputs.get(key)):
                diagnostics.append(
                    _diagnostic(
                        "error", f"inputs.{key}", f"{key} 必须是 project_root 下的安全相对路径。"
                    )
                )
        if parameters.get("ranking_script") is not None and not _safe_relative(
            parameters.get("ranking_script")
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "parameters.ranking_script",
                    "ranking_script 若提供，必须是 project_root 下显式的相对脚本路径。",
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
                    "不接受 extra_args；metric、top-k 与结果路径不得被重复参数覆盖。",
                )
            )
        metric = parameters.get("metric")
        if not isinstance(metric, str) or not metric.strip():
            diagnostics.append(_diagnostic("error", "parameters.metric", "metric 不能为空。"))
        if parameters.get("direction") not in {"minimize", "maximize"}:
            diagnostics.append(
                _diagnostic(
                    "error", "parameters.direction", "direction 必须是 minimize 或 maximize。"
                )
            )
        top_k = parameters.get("top_k")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            diagnostics.append(_diagnostic("error", "parameters.top_k", "top_k 必须是正整数。"))
        if parameters.get("missing_metric_policy") not in {"reject", "error"}:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "parameters.missing_metric_policy",
                    "missing_metric_policy 仅支持 reject 或 error；不做隐式插值。",
                )
            )
        if not _safe_relative(parameters.get("result_manifest", "ranking-result.json")):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "parameters.result_manifest",
                    "result_manifest 必须是 attempt_dir 下的安全相对路径。",
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
        result_path = _path(
            context["attempt_dir"], parameters.get("result_manifest", "ranking-result.json")
        )
        script_value = parameters.get("ranking_script")
        script_path = (
            _path(context["project_root"], script_value)
            if isinstance(script_value, str)
            else BUILTIN_RANK.absolute()
        )
        argv = list(parameters.get("interpreter_argv", [sys.executable]))
        argv.extend(
            [
                str(script_path),
                "--candidate-manifest",
                str(_path(context["project_root"], inputs["candidate_manifest"])),
                "--metric-results-manifest",
                str(_path(context["project_root"], inputs["metric_results_manifest"])),
                "--metric",
                parameters["metric"],
                "--direction",
                parameters["direction"],
                "--top-k",
                str(parameters["top_k"]),
                "--missing-metric-policy",
                parameters["missing_metric_policy"],
                "--result-manifest",
                str(result_path),
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
            "rule": {
                "metric": parameters["metric"],
                "direction": parameters["direction"],
                "top_k": parameters["top_k"],
                "missing_metric_policy": parameters["missing_metric_policy"],
            },
            "implementation": {
                "path": str(script_path),
                "sha256": _sha256(script_path) if script_path.is_file() else None,
                "bundled": script_value is None,
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

    def _paths(self, context: dict[str, Any]) -> tuple[Path, Path, Path]:
        return (
            _path(context["project_root"], context["inputs"]["candidate_manifest"]),
            _path(context["project_root"], context["inputs"]["metric_results_manifest"]),
            _path(
                context["attempt_dir"],
                context["parameters"].get("result_manifest", "ranking-result.json"),
            ),
        )

    def _verify_result(
        self,
        context: dict[str, Any],
        candidate_manifest: dict[str, Any],
        metric_results_manifest: dict[str, Any],
        result: dict[str, Any],
    ) -> list[dict[str, str]]:
        diagnostics: list[dict[str, str]] = []
        parameters = context["parameters"]
        if candidate_manifest.get("schema_version") != 1:
            diagnostics.append(
                _diagnostic(
                    "error", "candidates.schema", "candidate manifest schema_version 必须为 1。"
                )
            )
        if metric_results_manifest.get("schema_version") != 1:
            diagnostics.append(
                _diagnostic(
                    "error", "metrics.schema", "metric-results manifest schema_version 必须为 1。"
                )
            )
        candidates = candidate_manifest.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            diagnostics.append(
                _diagnostic("error", "candidates.empty", "candidate_manifest.candidates 必须非空。")
            )
            return diagnostics
        candidate_ids: list[str] = []
        for item in candidates:
            candidate_id = item.get("id") if isinstance(item, dict) else None
            if not isinstance(candidate_id, str) or not candidate_id:
                diagnostics.append(_diagnostic("error", "candidates.id", "每个候选必须有非空 id。"))
            else:
                candidate_ids.append(candidate_id)
        if len(candidate_ids) != len(set(candidate_ids)):
            diagnostics.append(_diagnostic("error", "candidates.duplicate", "候选 id 必须唯一。"))

        metric_results = metric_results_manifest.get("results")
        if not isinstance(metric_results, list):
            diagnostics.append(
                _diagnostic("error", "metrics.results", "metric results 必须是数组。")
            )
            return diagnostics
        metric_values: dict[str, float] = {}
        seen_metric_results: set[str] = set()
        for item in metric_results:
            if not isinstance(item, dict):
                diagnostics.append(
                    _diagnostic("error", "metrics.item", "metric 记录必须是对象。")
                )
                continue
            candidate_id = item.get("candidate_id")
            metrics = item.get("metrics")
            value = metrics.get(parameters["metric"]) if isinstance(metrics, dict) else None
            if candidate_id not in candidate_ids:
                diagnostics.append(
                    _diagnostic(
                        "error", "metrics.unknown_candidate", "metric 结果引用未知候选。"
                    )
                )
            elif candidate_id in seen_metric_results:
                diagnostics.append(
                    _diagnostic("error", "metrics.duplicate", "同一候选的 metric 结果重复。")
                )
            else:
                seen_metric_results.add(candidate_id)
                if _finite_number(value):
                    metric_values[candidate_id] = float(value)
                else:
                    diagnostics.append(
                        _diagnostic(
                            "error",
                            "metrics.metric",
                            "已声明的 metric 记录必须包含有限目标指标；缺失候选应完全省略。",
                        )
                    )

        expected_rule = {
            "metric": parameters["metric"],
            "direction": parameters["direction"],
            "top_k": parameters["top_k"],
            "missing_metric_policy": parameters["missing_metric_policy"],
        }
        if result.get("schema_version") != 1 or result.get("plugin_id") != PLUGIN_ID:
            diagnostics.append(
                _diagnostic("error", "result.identity", "结果 schema/plugin 标识不正确。")
            )
        if result.get("status") != "OK":
            diagnostics.append(_diagnostic("error", "result.status", "排序结果未声明 status=OK。"))
        if result.get("rule") != expected_rule:
            diagnostics.append(_diagnostic("error", "result.rule", "结果规则与已批准参数不一致。"))

        missing_ids = sorted(set(candidate_ids) - set(metric_values))
        if parameters["missing_metric_policy"] == "error" and missing_ids:
            diagnostics.append(
                _diagnostic(
                    "error", "result.missing_metric", "error 策略下存在缺失 metric。"
                )
            )
        declared_missing = result.get("excluded_missing", [])
        if declared_missing != missing_ids:
            diagnostics.append(
                _diagnostic("error", "result.excluded_missing", "缺失指标候选列表不完整或不稳定。")
            )

        ranked = result.get("ranked_candidates")
        if not isinstance(ranked, list):
            diagnostics.append(
                _diagnostic("error", "result.ranked_candidates", "ranked_candidates 必须是数组。")
            )
            return diagnostics
        expected_count = min(parameters["top_k"], len(metric_values))
        if len(ranked) != expected_count:
            diagnostics.append(
                _diagnostic("error", "result.top_k", "返回数量不符合 top_k 和可用指标数量。")
            )
        seen: set[str] = set()
        for index, item in enumerate(ranked):
            if not isinstance(item, dict):
                diagnostics.append(_diagnostic("error", "result.rank_item", "排名项必须是对象。"))
                continue
            candidate_id = item.get("candidate_id")
            value = item.get("value")
            if candidate_id in seen or candidate_id not in metric_values:
                diagnostics.append(
                    _diagnostic("error", "result.candidate_id", "排名候选未知或重复。")
                )
                continue
            seen.add(candidate_id)
            if item.get("rank") != index + 1:
                diagnostics.append(_diagnostic("error", "result.rank", "rank 必须从 1 连续递增。"))
            if not _finite_number(value) or not math.isclose(
                float(value), metric_values[candidate_id], rel_tol=1e-12, abs_tol=1e-12
            ):
                diagnostics.append(
                    _diagnostic("error", "result.value", "排名值必须等于 metric-results 清单中的原值。")
                )
        reverse = parameters["direction"] == "maximize"
        expected_ids = [
            candidate_id
            for candidate_id, _ in sorted(
                metric_values.items(),
                key=lambda item: ((-item[1]) if reverse else item[1], item[0]),
            )[:expected_count]
        ]
        if [item.get("candidate_id") for item in ranked if isinstance(item, dict)] != expected_ids:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "result.order",
                    "排名必须按指标排序，并以 candidate_id 做确定性并列裁决。",
                )
            )
        if result.get("candidate_count") != len(candidate_ids):
            diagnostics.append(
                _diagnostic("error", "result.candidate_count", "candidate_count 与输入不一致。")
            )
        return diagnostics

    def check(self, context: Any) -> dict[str, Any]:
        diagnostics = self.validate(context)
        if _errors(diagnostics):
            return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
        assert isinstance(context, dict)
        candidate_path, metric_results_path, result_path = self._paths(context)
        manifests: list[dict[str, Any]] = []
        for path in (candidate_path, metric_results_path, result_path):
            manifest, read_diagnostic = _read_json(path)
            if manifest is None:
                assert read_diagnostic is not None
                status = "WAIT" if read_diagnostic["level"] == "warning" else "FAIL"
                return {"plugin_id": PLUGIN_ID, "status": status, "diagnostics": [read_diagnostic]}
            manifests.append(manifest)
        diagnostics.extend(self._verify_result(context, manifests[0], manifests[1], manifests[2]))
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
        candidate_path, _, result_path = self._paths(context)
        candidate_manifest, _ = _read_json(candidate_path)
        result, _ = _read_json(result_path)
        assert candidate_manifest is not None and result is not None
        return {
            "plugin_id": PLUGIN_ID,
            "status": "OK",
            "artifacts": [
                {
                    "name": "ranking-result",
                    "path": context["parameters"].get("result_manifest", "ranking-result.json"),
                    "media_type": "application/json",
                }
            ],
            "metrics": {
                "candidate_count": len(candidate_manifest["candidates"]),
                "selected_count": len(result["ranked_candidates"]),
                "missing_metric_count": len(result["excluded_missing"]),
            },
        }

    def replay(self, context: Any) -> dict[str, Any]:
        """Re-verify explicit manifests without executing the ranking script."""

        collected = self.collect(context)
        collected["executable"] = False
        return collected
