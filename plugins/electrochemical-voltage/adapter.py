"""Unit-explicit electrochemical voltage post-processing adapter."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

from mlipflow.science.voltage import average_intercalation_voltage


PLUGIN_ID = "electrochemical-voltage"
COMPUTE_OPERATION = "compute-from-energies"
REPLAY_OPERATION = "replay-si-table-s11"
REPLAY_OUTPUT_NAMES = ("metrics.json", "model_ranking.json", "provenance.json")
DEFAULT_REPLAY_OUTPUT_SUBDIR = "manuscript-voltage-replay"
_SHA256_PATTERN = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})$")
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


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _safe_relative(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip() or "\x00" in value or "\n" in value:
        return False
    path = Path(value)
    return path != Path(".") and not path.is_absolute() and ".." not in path.parts


def _path(root: str, relative: str) -> Path:
    return Path(root).expanduser().absolute() / relative


def _operation(context: dict[str, Any]) -> str:
    parameters = context.get("parameters")
    raw = parameters.get("operation", COMPUTE_OPERATION) if isinstance(parameters, dict) else None
    # ``compute`` was the original adapter CLI spelling.  Treat it as an alias
    # while emitting only the explicit public operation from new plans.
    return COMPUTE_OPERATION if raw == "compute" else raw


def _normalized_sha256(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = _SHA256_PATTERN.fullmatch(value.strip())
    return match.group(1).lower() if match else None


@lru_cache(maxsize=1)
def _manuscript_replay_module() -> ModuleType:
    path = Path(__file__).with_name("manuscript_replay.py").absolute()
    spec = importlib.util.spec_from_file_location("mlipflow_voltage_manuscript_replay", path)
    if spec is None or spec.loader is None:  # pragma: no cover - packaging corruption
        raise RuntimeError(f"cannot load bundled manuscript replay module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _replay_input_artifacts(
    context: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]] | None, list[dict[str, str]]]:
    """Validate the strict SI Table S11 input and build expected artifacts in memory."""

    diagnostics: list[dict[str, str]] = []
    inputs = context["inputs"]
    parameters = context["parameters"]
    input_path = _path(context["project_root"], inputs["manuscript_voltage_table"])
    try:
        raw = input_path.read_bytes()
    except OSError as exc:
        return None, [
            _diagnostic(
                "error",
                "manuscript_replay.input_unreadable",
                f"无法读取 SI Table S11 CSV：{exc}",
            )
        ]
    try:
        module = _manuscript_replay_module()
        if len(module.CSV_FIELDS) != 15:
            raise RuntimeError("bundled replay contract must contain exactly 15 CSV fields")
        artifacts = module.build_artifacts(raw)
    except (OSError, RuntimeError, ValueError) as exc:
        return None, [
            _diagnostic(
                "error",
                "manuscript_replay.input_invalid",
                f"SI Table S11 证据不满足严格 15 列契约：{exc}",
            )
        ]

    expected_document_sha256 = _normalized_sha256(parameters.get("source_document_sha256"))
    provenance = artifacts.get("provenance.json")
    if not isinstance(provenance, dict):  # pragma: no cover - bundled contract guard
        return None, [
            _diagnostic(
                "error", "manuscript_replay.provenance_missing", "内存重放缺少 provenance。"
            )
        ]
    if provenance.get("source_document_sha256") != expected_document_sha256:
        diagnostics.append(
            _diagnostic(
                "error",
                "manuscript_replay.source_document_sha256",
                "CSV 中的 source_document_sha256 与已批准参数不一致。",
            )
        )
    for name in REPLAY_OUTPUT_NAMES:
        artifact = artifacts.get(name)
        if not isinstance(artifact, dict):
            diagnostics.append(
                _diagnostic(
                    "error", "manuscript_replay.artifact_missing", f"内存重放缺少 {name}。"
                )
            )
            continue
        if artifact.get("evidence_mode") != "manuscript-table-replay":
            diagnostics.append(
                _diagnostic(
                    "error",
                    "manuscript_replay.mode",
                    f"{name} 必须声明 evidence_mode=manuscript-table-replay。",
                )
            )
        if artifact.get("voltage_unit") != "V":
            diagnostics.append(
                _diagnostic(
                    "error", "manuscript_replay.unit", f"{name} 必须声明 voltage_unit=V。"
                )
            )
    if provenance.get("model_execution") is not False:
        diagnostics.append(
            _diagnostic(
                "error",
                "manuscript_replay.model_execution",
                "SI Table S11 replay 必须声明 model_execution=false。",
            )
        )
    if provenance.get("dft_execution") is not False:
        diagnostics.append(
            _diagnostic(
                "error",
                "manuscript_replay.dft_execution",
                "SI Table S11 replay 必须声明 dft_execution=false。",
            )
        )
    if provenance.get("recomputed_from_total_energies") is not False:
        diagnostics.append(
            _diagnostic(
                "error",
                "manuscript_replay.total_energy_claim",
                "SI Table S11 replay 不得声明从总能重算电压。",
            )
        )
    return artifacts, diagnostics


def _series_diagnostics(series: Any) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    if not isinstance(series, dict):
        return [_diagnostic("error", "energy_series.type", "能量清单必须是对象。")]
    if series.get("schema_version") != 1:
        diagnostics.append(_diagnostic("error", "energy_series.schema", "schema_version 必须为 1。"))
    if series.get("energy_unit") != "eV":
        diagnostics.append(
            _diagnostic("error", "energy_series.energy_unit", "当前仅支持明确的 eV 总能单位。")
        )
    formula_units = series.get("formula_units")
    if (
        isinstance(formula_units, bool)
        or not isinstance(formula_units, int)
        or formula_units < 1
    ):
        diagnostics.append(
            _diagnostic("error", "energy_series.formula_units", "formula_units 必须是正整数。")
        )
    states = series.get("states")
    if not isinstance(states, list) or len(states) < 2:
        diagnostics.append(
            _diagnostic("error", "energy_series.states", "至少需要两个 Li 含量不同的能量状态。")
        )
        return diagnostics
    ids: set[str] = set()
    previous_li: float | None = None
    for index, state in enumerate(states):
        prefix = f"states[{index}]"
        if not isinstance(state, dict):
            diagnostics.append(_diagnostic("error", f"{prefix}.type", "状态必须是对象。"))
            continue
        state_id = state.get("id")
        if not isinstance(state_id, str) or not state_id or state_id in ids:
            diagnostics.append(_diagnostic("error", f"{prefix}.id", "状态 id 必须非空且唯一。"))
        else:
            ids.add(state_id)
        li_content = state.get("li_content")
        energy = state.get("total_energy_ev")
        if not _finite(li_content) or float(li_content) < 0:
            diagnostics.append(
                _diagnostic("error", f"{prefix}.li_content", "li_content 必须是非负有限数。")
            )
        elif previous_li is not None and float(li_content) <= previous_li:
            diagnostics.append(
                _diagnostic(
                    "error", f"{prefix}.order", "states 必须按 Li 含量严格递增且无重复。"
                )
            )
        else:
            previous_li = float(li_content)
        if not _finite(energy):
            diagnostics.append(
                _diagnostic("error", f"{prefix}.energy", "total_energy_ev 必须是有限数。")
            )
        if state.get("completion_status") != "OK":
            diagnostics.append(
                _diagnostic(
                    "error",
                    f"{prefix}.completion_status",
                    "只接受上游明确标记 completion_status=OK 的能量。",
                )
            )
        if not isinstance(state.get("energy_source"), str) or not state.get("energy_source"):
            diagnostics.append(
                _diagnostic("error", f"{prefix}.energy_source", "必须记录 energy_source。")
            )
    return diagnostics


def _compute_manifest(
    series: dict[str, Any], lithium_reference_ev: float, electrons_per_li: float
) -> dict[str, Any]:
    """Compute adjacent average voltages in memory without changing files."""

    diagnostics = _series_diagnostics(series)
    if diagnostics:
        raise ValueError("; ".join(item["message"] for item in diagnostics))
    formula_units = series["formula_units"]
    normalized_states = []
    for state in series["states"]:
        normalized_states.append(
            {
                "id": state["id"],
                "li_per_formula_unit": float(state["li_content"]) / formula_units,
                "energy_ev_per_formula_unit": float(state["total_energy_ev"]) / formula_units,
                "energy_source": state["energy_source"],
            }
        )
    steps: list[dict[str, Any]] = []
    for low, high in zip(normalized_states, normalized_states[1:]):
        voltage = average_intercalation_voltage(
            energy_low_li_ev=low["energy_ev_per_formula_unit"],
            energy_high_li_ev=high["energy_ev_per_formula_unit"],
            x_low=low["li_per_formula_unit"],
            x_high=high["li_per_formula_unit"],
            lithium_reference_ev=lithium_reference_ev,
            electrons_per_li=electrons_per_li,
        )
        steps.append(
            {
                "low_state_id": low["id"],
                "high_state_id": high["id"],
                "x_low": low["li_per_formula_unit"],
                "x_high": high["li_per_formula_unit"],
                "delta_li_per_formula_unit": high["li_per_formula_unit"]
                - low["li_per_formula_unit"],
                "average_voltage_v": voltage,
            }
        )
    voltages = [step["average_voltage_v"] for step in steps]
    return {
        "schema_version": 1,
        "plugin_id": PLUGIN_ID,
        "status": "OK",
        "composition_order": "increasing-li",
        "energy_unit": "eV",
        "voltage_unit": "V",
        "reference": {
            "species": "Li",
            "energy_ev_per_atom": lithium_reference_ev,
            "electrons_per_li": electrons_per_li,
        },
        "formula_units": formula_units,
        "states": normalized_states,
        "steps": steps,
        "metrics": {
            "step_count": len(steps),
            "minimum_voltage_v": min(voltages),
            "maximum_voltage_v": max(voltages),
            "mean_voltage_v": sum(voltages) / len(voltages),
        },
    }


class Adapter:
    """Plan and verify deterministic average-voltage post-processing."""

    def validate(self, context: Any) -> list[dict[str, str]]:
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
                diagnostics.append(
                    _diagnostic("error", f"context.{key}", f"{key} 必须是路径字符串。")
                )
        for key in ("inputs", "parameters", "resources"):
            if key in context and not isinstance(context[key], dict):
                diagnostics.append(_diagnostic("error", f"context.{key}", f"{key} 必须是对象。"))
        if "backend" in context and context["backend"] != "local":
            diagnostics.append(
                _diagnostic(
                    "error", "backend.unsupported", "电压后处理当前仅支持 local backend。"
                )
            )
        inputs = context.get("inputs")
        parameters = context.get("parameters")
        if not isinstance(inputs, dict) or not isinstance(parameters, dict):
            return diagnostics
        operation = _operation(context)
        if operation not in {COMPUTE_OPERATION, REPLAY_OPERATION}:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "parameters.operation",
                    f"operation 必须为 {COMPUTE_OPERATION} 或 {REPLAY_OPERATION}。",
                )
            )
            return diagnostics

        if operation == REPLAY_OPERATION:
            input_is_safe = _safe_relative(inputs.get("manuscript_voltage_table"))
            if not input_is_safe:
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "inputs.manuscript_voltage_table",
                        "manuscript_voltage_table 必须是 project_root 下的安全相对路径。",
                    )
                )
            if not _safe_relative(
                parameters.get("replay_output_subdir", DEFAULT_REPLAY_OUTPUT_SUBDIR)
            ):
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "parameters.replay_output_subdir",
                        "replay_output_subdir 必须是 attempt_dir 下的安全相对路径。",
                    )
                )
            if _normalized_sha256(parameters.get("source_document_sha256")) is None:
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "parameters.source_document_sha256",
                        "SI replay 必须提供 64 位 source_document_sha256（可带 sha256: 前缀）。",
                    )
                )
            if input_is_safe and not _errors(diagnostics):
                _, replay_diagnostics = _replay_input_artifacts(context)
                diagnostics.extend(replay_diagnostics)
            return diagnostics

        if not _safe_relative(inputs.get("energy_manifest")):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "inputs.energy_manifest",
                    "energy_manifest 必须是 project_root 下的安全相对路径。",
                )
            )
        if not _safe_relative(parameters.get("result_manifest", "voltage-result.json")):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "parameters.result_manifest",
                    "result_manifest 必须是 attempt_dir 下的安全相对路径。",
                )
            )
        reference = parameters.get("lithium_reference_ev")
        if not _finite(reference):
            diagnostics.append(
                _diagnostic(
                    "error", "parameters.lithium_reference_ev", "必须声明有限的 Li 参考能（eV/atom）。"
                )
            )
        electrons = parameters.get("electrons_per_li", 1.0)
        if not _finite(electrons) or float(electrons) <= 0:
            diagnostics.append(
                _diagnostic("error", "parameters.electrons_per_li", "electrons_per_li 必须为正。")
            )
        if parameters.get("energy_unit") != "eV":
            diagnostics.append(
                _diagnostic("error", "parameters.energy_unit", "当前仅支持 energy_unit=eV。")
            )
        if parameters.get("reference_species") != "Li":
            diagnostics.append(
                _diagnostic("error", "parameters.reference_species", "当前参考物种必须显式为 Li。")
            )
        if parameters.get("composition_order") != "increasing-li":
            diagnostics.append(
                _diagnostic(
                    "error",
                    "parameters.composition_order",
                    "composition_order 必须为 increasing-li。",
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
        operation = _operation(context)
        if operation == REPLAY_OPERATION:
            input_path = _path(
                context["project_root"], context["inputs"]["manuscript_voltage_table"]
            )
            output_relative = context["parameters"].get(
                "replay_output_subdir", DEFAULT_REPLAY_OUTPUT_SUBDIR
            )
            output_dir = _path(context["attempt_dir"], output_relative)
            if output_dir.exists():
                return {
                    "plugin_id": PLUGIN_ID,
                    "status": "BLOCKED",
                    "executable": False,
                    "operation": REPLAY_OPERATION,
                    "mode": "replay",
                    "model_execution": False,
                    "diagnostics": [
                        _diagnostic(
                            "error",
                            "manuscript_replay.output_exists",
                            "SI replay 输出目录必须尚不存在；拒绝覆盖已有或部分结果。",
                        )
                    ],
                }
            replay_script = Path(__file__).with_name("manuscript_replay.py").absolute()
            expected_outputs = [str(output_dir / name) for name in REPLAY_OUTPUT_NAMES]
            return {
                "plugin_id": PLUGIN_ID,
                "status": "READY",
                "executable": True,
                "operation": REPLAY_OPERATION,
                "mode": "replay",
                "model_execution": False,
                "dft_execution": False,
                "recomputed_from_total_energies": False,
                "argv": [
                    sys.executable,
                    str(replay_script),
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                ],
                "shell": False,
                "cwd": str(Path(context["attempt_dir"]).expanduser().absolute()),
                "expected_outputs": expected_outputs,
                "diagnostics": diagnostics,
                "units": {"voltage": "V"},
                "source_document_sha256": _normalized_sha256(
                    context["parameters"]["source_document_sha256"]
                ),
            }

        input_path = _path(context["project_root"], context["inputs"]["energy_manifest"])
        result_relative = context["parameters"].get("result_manifest", "voltage-result.json")
        result_path = _path(context["attempt_dir"], result_relative)
        argv = [
            sys.executable,
            str(Path(__file__).absolute()),
            COMPUTE_OPERATION,
            "--energy-manifest",
            str(input_path),
            "--result-manifest",
            str(result_path),
            "--lithium-reference-ev",
            repr(float(context["parameters"]["lithium_reference_ev"])),
            "--electrons-per-li",
            repr(float(context["parameters"].get("electrons_per_li", 1.0))),
        ]
        return {
            "plugin_id": PLUGIN_ID,
            "status": "READY",
            "executable": True,
            "operation": COMPUTE_OPERATION,
            "mode": "execute",
            "argv": argv,
            "shell": False,
            "cwd": str(Path(context["attempt_dir"]).expanduser().absolute()),
            "expected_outputs": [str(result_path)],
            "diagnostics": diagnostics,
            "units": {"energy": "eV", "voltage": "V"},
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

    def _paths(self, context: dict[str, Any]) -> tuple[Path, Path]:
        return (
            _path(context["project_root"], context["inputs"]["energy_manifest"]),
            _path(
                context["attempt_dir"],
                context["parameters"].get("result_manifest", "voltage-result.json"),
            ),
        )

    def _replay_paths(self, context: dict[str, Any]) -> tuple[Path, dict[str, Path]]:
        input_path = _path(
            context["project_root"], context["inputs"]["manuscript_voltage_table"]
        )
        output_dir = _path(
            context["attempt_dir"],
            context["parameters"].get(
                "replay_output_subdir", DEFAULT_REPLAY_OUTPUT_SUBDIR
            ),
        )
        return input_path, {name: output_dir / name for name in REPLAY_OUTPUT_NAMES}

    def _verify_replay_outputs(
        self, context: dict[str, Any]
    ) -> tuple[dict[str, dict[str, Any]] | None, list[dict[str, str]], str]:
        expected, diagnostics = _replay_input_artifacts(context)
        if expected is None or _errors(diagnostics):
            return None, diagnostics, "FAIL"
        _, output_paths = self._replay_paths(context)
        existence = {name: path.is_file() for name, path in output_paths.items()}
        if not any(existence.values()):
            diagnostics.append(
                _diagnostic(
                    "warning", "artifact.missing", "SI Table S11 replay 三件套尚未生成。"
                )
            )
            return None, diagnostics, "WAIT"
        if not all(existence.values()):
            missing = ", ".join(name for name, exists in existence.items() if not exists)
            diagnostics.append(
                _diagnostic(
                    "error",
                    "manuscript_replay.partial_outputs",
                    f"SI Table S11 replay 产物不完整，缺少：{missing}。",
                )
            )
            return None, diagnostics, "FAIL"
        observed: dict[str, dict[str, Any]] = {}
        for name, path in output_paths.items():
            value, read_diagnostic = _read_json(path)
            if value is None:
                assert read_diagnostic is not None
                diagnostics.append(read_diagnostic)
                status = "WAIT" if read_diagnostic["level"] == "warning" else "FAIL"
                return None, diagnostics, status
            observed[name] = value
        for name in REPLAY_OUTPUT_NAMES:
            if observed[name] != expected[name]:
                diagnostics.append(
                    _diagnostic(
                        "error",
                        f"manuscript_replay.output.{name}",
                        f"{name} 与严格 SI Table S11 输入的内存重算结果不一致。",
                    )
                )
        provenance = observed["provenance.json"]
        if provenance.get("evidence_mode") != "manuscript-table-replay":
            diagnostics.append(
                _diagnostic(
                    "error", "manuscript_replay.mode", "provenance 必须声明 replay mode。"
                )
            )
        if provenance.get("model_execution") is not False:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "manuscript_replay.model_execution",
                    "provenance 必须声明 model_execution=false。",
                )
            )
        if provenance.get("recomputed_from_total_energies") is not False:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "manuscript_replay.total_energy_claim",
                    "SI Table S11 replay 不能声明从总能重算电压。",
                )
            )
        if provenance.get("source_document_sha256") != _normalized_sha256(
            context["parameters"].get("source_document_sha256")
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "manuscript_replay.source_document_sha256",
                    "provenance 的文档 SHA-256 与已批准参数不一致。",
                )
            )
        if any(observed[name].get("voltage_unit") != "V" for name in REPLAY_OUTPUT_NAMES):
            diagnostics.append(
                _diagnostic(
                    "error", "manuscript_replay.unit", "所有 SI replay 产物必须使用 V。"
                )
            )
        return observed, diagnostics, "FAIL" if _errors(diagnostics) else "OK"

    def _verify_result(
        self, context: dict[str, Any], series: dict[str, Any], result: dict[str, Any]
    ) -> list[dict[str, str]]:
        diagnostics = _series_diagnostics(series)
        if diagnostics:
            return diagnostics
        expected = _compute_manifest(
            series,
            float(context["parameters"]["lithium_reference_ev"]),
            float(context["parameters"].get("electrons_per_li", 1.0)),
        )
        for key in (
            "schema_version",
            "plugin_id",
            "status",
            "composition_order",
            "energy_unit",
            "voltage_unit",
            "reference",
            "formula_units",
            "states",
        ):
            if result.get(key) != expected[key]:
                diagnostics.append(
                    _diagnostic("error", f"result.{key}", f"结果字段 {key} 与输入/公式不一致。")
                )
        steps = result.get("steps")
        if not isinstance(steps, list) or len(steps) != len(expected["steps"]):
            diagnostics.append(_diagnostic("error", "result.steps", "电压步数不正确。"))
        else:
            for index, (observed, calculated) in enumerate(zip(steps, expected["steps"])):
                if not isinstance(observed, dict):
                    diagnostics.append(
                        _diagnostic("error", f"result.steps[{index}]", "电压步必须是对象。")
                    )
                    continue
                for key, expected_value in calculated.items():
                    actual = observed.get(key)
                    if isinstance(expected_value, float):
                        if not _finite(actual) or not math.isclose(
                            float(actual), expected_value, rel_tol=1e-12, abs_tol=1e-12
                        ):
                            diagnostics.append(
                                _diagnostic(
                                    "error",
                                    f"result.steps[{index}].{key}",
                                    f"电压步字段 {key} 与核心公式不一致。",
                                )
                            )
                    elif actual != expected_value:
                        diagnostics.append(
                            _diagnostic(
                                "error",
                                f"result.steps[{index}].{key}",
                                f"电压步字段 {key} 不一致。",
                            )
                        )
        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            diagnostics.append(_diagnostic("error", "result.metrics", "缺少 metrics。"))
        else:
            for key, expected_value in expected["metrics"].items():
                actual = metrics.get(key)
                if isinstance(expected_value, float):
                    if not _finite(actual) or not math.isclose(
                        float(actual), expected_value, rel_tol=1e-12, abs_tol=1e-12
                    ):
                        diagnostics.append(
                            _diagnostic("error", f"result.metrics.{key}", f"指标 {key} 不一致。")
                        )
                elif actual != expected_value:
                    diagnostics.append(
                        _diagnostic("error", f"result.metrics.{key}", f"指标 {key} 不一致。")
                    )
        return diagnostics

    def check(self, context: Any) -> dict[str, Any]:
        diagnostics = self.validate(context)
        if _errors(diagnostics):
            return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
        assert isinstance(context, dict)
        if _operation(context) == REPLAY_OPERATION:
            _, diagnostics, status = self._verify_replay_outputs(context)
            _, output_paths = self._replay_paths(context)
            return {
                "plugin_id": PLUGIN_ID,
                "status": status,
                "operation": REPLAY_OPERATION,
                "mode": "replay",
                "model_execution": False,
                "recomputed_from_total_energies": False,
                "diagnostics": diagnostics,
                "result_manifests": {
                    name: str(path) for name, path in output_paths.items()
                },
            }
        energy_path, result_path = self._paths(context)
        values: list[dict[str, Any]] = []
        for path in (energy_path, result_path):
            value, read_diagnostic = _read_json(path)
            if value is None:
                assert read_diagnostic is not None
                status = "WAIT" if read_diagnostic["level"] == "warning" else "FAIL"
                return {"plugin_id": PLUGIN_ID, "status": status, "diagnostics": [read_diagnostic]}
            values.append(value)
        diagnostics.extend(self._verify_result(context, values[0], values[1]))
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
        if _operation(context) == REPLAY_OPERATION:
            observed, diagnostics, status = self._verify_replay_outputs(context)
            if observed is None or status != "OK":
                return {"plugin_id": PLUGIN_ID, "status": status, "diagnostics": diagnostics}
            _, output_paths = self._replay_paths(context)
            metrics = observed["metrics.json"]
            ranking = observed["model_ranking.json"]
            winner = ranking["ranking"][0]
            output_relative = context["parameters"].get(
                "replay_output_subdir", DEFAULT_REPLAY_OUTPUT_SUBDIR
            )
            return {
                "plugin_id": PLUGIN_ID,
                "status": "OK",
                "operation": REPLAY_OPERATION,
                "mode": "replay",
                "model_execution": False,
                "dft_execution": False,
                "recomputed_from_total_energies": False,
                "diagnostics": diagnostics,
                "artifacts": [
                    {
                        "name": "manuscript-replay-metrics",
                        "role": "manuscript-replay-metrics",
                        "path": str(Path(output_relative) / "metrics.json"),
                        "media_type": "application/json",
                    },
                    {
                        "name": "manuscript-replay-ranking",
                        "role": "manuscript-replay-ranking",
                        "path": str(Path(output_relative) / "model_ranking.json"),
                        "media_type": "application/json",
                    },
                    {
                        "name": "manuscript-replay-provenance",
                        "role": "manuscript-replay-provenance",
                        "path": str(Path(output_relative) / "provenance.json"),
                        "media_type": "application/json",
                    },
                ],
                "metrics": {
                    "row_count": metrics["row_count"],
                    "model_count": len(metrics["models"]),
                    "winner_mae_v": winner["mae_v"],
                },
                "selection": {"winner": ranking["winner"], "unit": "V"},
                "result_manifests": {
                    name: str(path) for name, path in output_paths.items()
                },
            }
        _, result_path = self._paths(context)
        result, _ = _read_json(result_path)
        assert result is not None
        return {
            "plugin_id": PLUGIN_ID,
            "status": "OK",
            "artifacts": [
                {
                    "name": "voltage-result",
                    "path": context["parameters"].get("result_manifest", "voltage-result.json"),
                    "media_type": "application/json",
                }
            ],
            "metrics": dict(result["metrics"]),
        }

    def replay(self, context: Any) -> dict[str, Any]:
        """Recompute in memory or verify an existing result; never write a file."""

        diagnostics = self.validate(context)
        if _errors(diagnostics):
            return {
                "plugin_id": PLUGIN_ID,
                "status": "FAIL",
                "executable": False,
                "diagnostics": diagnostics,
            }
        assert isinstance(context, dict)
        if _operation(context) == REPLAY_OPERATION:
            expected, replay_diagnostics = _replay_input_artifacts(context)
            diagnostics.extend(replay_diagnostics)
            if expected is None or _errors(diagnostics):
                return {
                    "plugin_id": PLUGIN_ID,
                    "status": "FAIL",
                    "executable": False,
                    "operation": REPLAY_OPERATION,
                    "mode": "replay",
                    "model_execution": False,
                    "diagnostics": diagnostics,
                }
            _, output_paths = self._replay_paths(context)
            existing = [path.is_file() for path in output_paths.values()]
            if any(existing) and not all(existing):
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "manuscript_replay.partial_outputs",
                        "发现部分 SI replay 产物；无写 replay 拒绝混合部分结果。",
                    )
                )
            elif all(existing):
                observed, verify_diagnostics, _ = self._verify_replay_outputs(context)
                diagnostics.extend(verify_diagnostics)
                if observed is not None:
                    expected = observed
            return {
                "plugin_id": PLUGIN_ID,
                "status": "FAIL" if _errors(diagnostics) else "OK",
                "executable": False,
                "operation": REPLAY_OPERATION,
                "mode": "replay",
                "model_execution": False,
                "dft_execution": False,
                "recomputed_from_total_energies": False,
                "diagnostics": diagnostics,
                "result_manifest": {
                    "schema_version": 1,
                    "mode": "replay",
                    "artifacts": expected,
                },
            }
        energy_path, result_path = self._paths(context)
        series, read_diagnostic = _read_json(energy_path)
        if series is None:
            return {
                "plugin_id": PLUGIN_ID,
                "status": "WAIT" if read_diagnostic and read_diagnostic["level"] == "warning" else "FAIL",
                "executable": False,
                "diagnostics": [read_diagnostic] if read_diagnostic else [],
            }
        diagnostics.extend(_series_diagnostics(series))
        if _errors(diagnostics):
            return {
                "plugin_id": PLUGIN_ID,
                "status": "FAIL",
                "executable": False,
                "diagnostics": diagnostics,
            }
        if result_path.is_file():
            result, result_diagnostic = _read_json(result_path)
            if result is None:
                return {
                    "plugin_id": PLUGIN_ID,
                    "status": "FAIL",
                    "executable": False,
                    "diagnostics": [result_diagnostic] if result_diagnostic else [],
                }
            diagnostics.extend(self._verify_result(context, series, result))
        else:
            result = _compute_manifest(
                series,
                float(context["parameters"]["lithium_reference_ev"]),
                float(context["parameters"].get("electrons_per_li", 1.0)),
            )
        return {
            "plugin_id": PLUGIN_ID,
            "status": "FAIL" if _errors(diagnostics) else "OK",
            "executable": False,
            "diagnostics": diagnostics,
            "result_manifest": result,
        }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute adjacent average Li intercalation voltages")
    parser.add_argument("operation", choices=["compute", COMPUTE_OPERATION])
    parser.add_argument("--energy-manifest", required=True)
    parser.add_argument("--result-manifest", required=True)
    parser.add_argument("--lithium-reference-ev", required=True, type=float)
    parser.add_argument("--electrons-per-li", default=1.0, type=float)
    args = parser.parse_args(argv)
    energy_path = Path(args.energy_manifest)
    result_path = Path(args.result_manifest)
    series, diagnostic = _read_json(energy_path)
    if series is None:
        parser.error(diagnostic["message"] if diagnostic else "无法读取 energy manifest")
    try:
        result = _compute_manifest(series, args.lithium_reference_ev, args.electrons_per_li)
    except ValueError as exc:
        parser.error(str(exc))
    if result_path.exists():
        parser.error(f"refuse to overwrite existing result manifest: {result_path}")
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
