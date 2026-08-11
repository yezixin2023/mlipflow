"""Safe command planner and result collector for MLIP benchmark artifacts.

The adapter deliberately does not import any of the six reviewed model
configurations and never executes a program itself.  A user-owned prediction
CLI is required to accept the argv contract emitted by :meth:`Adapter.plan`
and to write the versioned result manifest consumed by :meth:`Adapter.check`
and :meth:`Adapter.collect`.  The bundled ``benchmark_wrapper.py`` is exposed
through separate normalize operations that recompute metrics only from
explicit pairs or replay existing evidence; neither operation loads an MLIP.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


PLUGIN_ID = "mlip-benchmark"
MODEL_FAMILIES = frozenset(
    {
        "deepmd-se_atten_v2",
        "deepmd-se_e2_a",
        "deepmd-se_e2_r",
        "deepmd-dpa2",
        "m3gnet",
        "chgnet",
    }
)
LEGACY_OPERATIONS = frozenset({"evaluate-static", "collect-existing"})
NORMALIZE_OPERATIONS = frozenset({"normalize-replay", "normalize-execute"})
OPERATIONS = LEGACY_OPERATIONS | NORMALIZE_OPERATIONS
EXECUTION_BACKENDS = frozenset({"local"})
SHELL_EXECUTABLES = frozenset(
    {"bash", "csh", "cmd", "dash", "fish", "ksh", "powershell", "pwsh", "sh", "tcsh", "zsh"}
)
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_CSV_BYTES = 32 * 1024 * 1024
NORMALIZED_OUTPUT_NAMES = (
    "metrics.json",
    "benchmark_summary.csv",
    "model_ranking.json",
    "provenance.json",
)
SUMMARY_COLUMNS = (
    "model",
    "task",
    "scenario",
    "split",
    "metric",
    "value",
    "unit",
    "direction",
    "sample_count",
    "mode",
    "evidence_sha256",
    "source_path",
    "source_format",
    "evidence_locator",
    "unit_provenance",
    "dimensions_json",
)
BUNDLED_WRAPPER = (
    Path(globals().get("__file__", "adapter.py")).absolute().with_name("benchmark_wrapper.py")
)
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


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
    return isinstance(value, str) and bool(value) and "\x00" not in value and "\n" not in value


def _project_path(context: dict[str, Any], value: Any) -> str:
    reference = _reference(value)
    if reference is None:
        raise ValueError("missing path reference")
    path = Path(reference)
    if path.is_absolute():
        return str(path)
    return str(Path(str(context["project_root"])) / path)


def _safe_attempt_output(context: dict[str, Any], value: Any) -> str:
    reference = _reference(value)
    if reference is None:
        raise ValueError("missing output path reference")
    path = Path(reference)
    if path.is_absolute() or path in {Path("."), Path("")} or ".." in path.parts:
        raise ValueError("output must be a relative path inside attempt_dir")
    return str(Path(str(context["attempt_dir"])) / path)


def _operation(context: dict[str, Any]) -> str:
    return str(_mapping(context.get("parameters")).get("operation", "evaluate-static"))


def _normalize_mode(operation: str) -> str:
    if operation == "normalize-replay":
        return "replay"
    if operation == "normalize-execute":
        return "execute"
    raise ValueError("operation is not a normalization operation")


def _portable_locator(value: Any, fallback: str, field: str) -> str:
    if value in (None, ""):
        return fallback
    if not _plain_string(value):
        raise ValueError(f"{field} must be a non-empty single-line string")
    locator = str(value).replace("\\", "/")
    lowered = locator.lower()
    if locator.startswith("/") or re.match(r"^[a-zA-Z]:/", locator) or lowered.startswith("file:"):
        raise ValueError(f"{field} must not expose an absolute local path")
    if "://" not in locator and ".." in Path(locator).parts:
        raise ValueError(f"{field} must not traverse parent directories")
    return locator


def _evidence_specs(context: dict[str, Any]) -> list[dict[str, str]]:
    inputs = _mapping(context.get("inputs"))
    raw = inputs.get("evidence_inputs")
    if not isinstance(raw, list) or not raw:
        raise ValueError("inputs.evidence_inputs must be a non-empty list")
    result: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        reference = _reference(item)
        if reference is None:
            raise ValueError(f"inputs.evidence_inputs[{index}] requires a path")
        resolved = _project_path(context, item)
        locator_value = item.get("evidence_locator") if isinstance(item, dict) else None
        locator = _portable_locator(
            locator_value, Path(reference).name, f"inputs.evidence_inputs[{index}].evidence_locator"
        )
        result.append({"path": resolved, "locator": locator})
    if len(result) > 1:
        custom = [
            index
            for index, (item, spec) in enumerate(zip(raw, result))
            if isinstance(item, dict)
            and item.get("evidence_locator") not in (None, "")
            and item.get("evidence_locator") != Path(spec["path"]).name
        ]
        if custom:
            raise ValueError(
                "custom evidence_locator is supported only for one input; multiple inputs use unique basenames"
            )
        basenames = [Path(item["path"]).name for item in result]
        if len(set(basenames)) != len(basenames):
            raise ValueError("multiple evidence inputs require unique basenames")
        for item in result:
            item["locator"] = Path(item["path"]).name
    return result


def _normalized_output_dir(context: dict[str, Any]) -> Path:
    inputs = _mapping(context.get("inputs"))
    return Path(_safe_attempt_output(context, inputs.get("output_dir")))


def _normalized_output_paths(context: dict[str, Any]) -> dict[str, Path]:
    output_dir = _normalized_output_dir(context)
    return {name: output_dir / name for name in NORMALIZED_OUTPUT_NAMES}


def _source_script_spec(context: dict[str, Any]) -> dict[str, str] | None:
    inputs = _mapping(context.get("inputs"))
    value = inputs.get("source_script")
    if value in (None, ""):
        return None
    reference = _reference(value)
    if reference is None:
        raise ValueError("inputs.source_script requires a path")
    locator_value = value.get("source_script_locator") if isinstance(value, dict) else None
    if locator_value in (None, ""):
        locator_value = _mapping(context.get("parameters")).get("source_script_locator")
    return {
        "path": _project_path(context, value),
        "locator": _portable_locator(
            locator_value, Path(reference).name, "inputs.source_script.source_script_locator"
        ),
    }


def _expected_values(
    parameters: dict[str, Any], singular: str, plural: str, *, models: bool = False
) -> set[str]:
    singular_value = parameters.get(singular)
    plural_value = parameters.get(plural)
    if singular_value not in (None, "") and plural_value not in (None, []):
        raise ValueError(f"parameters.{singular} and parameters.{plural} are mutually exclusive")
    if plural_value not in (None, []):
        if not isinstance(plural_value, list) or not plural_value:
            raise ValueError(f"parameters.{plural} must be a non-empty list")
        values = plural_value
    elif singular_value not in (None, ""):
        values = [singular_value]
    else:
        raise ValueError(f"parameters.{singular} or parameters.{plural} is required")
    normalized: set[str] = set()
    for value in values:
        if not _plain_string(value):
            raise ValueError(f"parameters.{plural} entries must be non-empty strings")
        token = str(value)
        if models:
            if token not in MODEL_FAMILIES:
                raise ValueError(f"unsupported exact model family: {token}")
        elif not _IDENTIFIER.fullmatch(token):
            raise ValueError(f"parameters.{plural} entries must be normalized identifiers")
        normalized.add(token)
    if len(normalized) != len(values):
        raise ValueError(f"parameters.{plural} entries must be unique")
    return normalized


def _expected_identities(parameters: dict[str, Any]) -> dict[str, set[str]]:
    return {
        "model": _expected_values(
            parameters, "model_family", "expected_models", models=True
        ),
        "task": _expected_values(parameters, "task", "expected_tasks"),
        "scenario": _expected_values(parameters, "scenario", "expected_scenarios"),
    }


def _units(parameters: dict[str, Any]) -> dict[str, str]:
    value = parameters.get("units", {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("parameters.units must be an object")
    result: dict[str, str] = {}
    for key, unit in value.items():
        if key not in {"energy", "force", "stress"}:
            raise ValueError("parameters.units keys must be energy, force, or stress")
        if not _plain_string(unit):
            raise ValueError(f"parameters.units.{key} must be a non-empty unit string")
        result[key] = str(unit)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _result_path(context: dict[str, Any]) -> Path:
    inputs = _mapping(context.get("inputs"))
    operation = _mapping(context.get("parameters")).get("operation", "evaluate-static")
    if operation == "collect-existing":
        return Path(_project_path(context, inputs.get("result_manifest")))
    return Path(_safe_attempt_output(context, inputs.get("result_manifest")))


def _read_json(path: Path) -> Any:
    size = path.stat().st_size
    if size > MAX_JSON_BYTES:
        raise ValueError(f"JSON artifact exceeds {MAX_JSON_BYTES} bytes: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _portable_child(base: Path, value: Any, field: str) -> Path:
    reference = _reference(value)
    if reference is None:
        raise ValueError(f"{field} must contain a non-empty path")
    path = Path(reference)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be relative to the result manifest")
    return base / path


def _validate_metric_records(metrics: Any) -> tuple[dict[str, float], list[dict[str, str]]]:
    diagnostics: list[dict[str, str]] = []
    values: dict[str, float] = {}
    if not isinstance(metrics, dict) or not metrics:
        return {}, [_diagnostic("error", "benchmark.metrics_required", "metrics must be a non-empty object")]
    for name, record in sorted(metrics.items()):
        prefix = f"metric {name!r}"
        if not isinstance(name, str) or not name:
            diagnostics.append(_diagnostic("error", "benchmark.metric_name", "metric names must be non-empty strings"))
            continue
        if not isinstance(record, dict):
            diagnostics.append(_diagnostic("error", "benchmark.metric_record", f"{prefix} must be an object"))
            continue
        value = record.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            diagnostics.append(_diagnostic("error", "benchmark.metric_value", f"{prefix} value must be finite"))
        else:
            values[name] = float(value)
        if not _plain_string(record.get("unit")):
            diagnostics.append(_diagnostic("error", "benchmark.metric_unit", f"{prefix} requires an explicit unit"))
        if record.get("direction") not in {"minimize", "maximize"}:
            diagnostics.append(
                _diagnostic("error", "benchmark.metric_direction", f"{prefix} direction must be minimize or maximize")
            )
        count = record.get("sample_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            diagnostics.append(
                _diagnostic("error", "benchmark.metric_sample_count", f"{prefix} sample_count must be a positive integer")
            )
    return values, diagnostics


def _load_result(
    path: Path, context: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, float], list[dict[str, Any]], list[dict[str, str]]]:
    diagnostics: list[dict[str, str]] = []
    artifacts: list[dict[str, Any]] = []
    try:
        raw = _read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return None, {}, [], [_diagnostic("error", "benchmark.result_unreadable", str(exc))]
    if not isinstance(raw, dict):
        return None, {}, [], [_diagnostic("error", "benchmark.result_type", "result manifest must be a JSON object")]
    if raw.get("schema_version") != 1:
        diagnostics.append(_diagnostic("error", "benchmark.result_schema", "schema_version must equal 1"))
    if raw.get("plugin_id") != PLUGIN_ID:
        diagnostics.append(_diagnostic("error", "benchmark.result_plugin", f"plugin_id must equal {PLUGIN_ID}"))
    if raw.get("status") not in {"OK", "FAIL"}:
        diagnostics.append(_diagnostic("error", "benchmark.result_status", "status must be OK or FAIL"))
    parameters = _mapping(context.get("parameters"))
    if raw.get("model_family") not in MODEL_FAMILIES:
        supported = ", ".join(sorted(MODEL_FAMILIES))
        diagnostics.append(
            _diagnostic(
                "error",
                "benchmark.result_family",
                f"model_family must be one of the exact reviewed names: {supported}",
            )
        )
    elif raw.get("model_family") != parameters.get("model_family"):
        diagnostics.append(
            _diagnostic("error", "benchmark.family_mismatch", "result model_family does not match the approved plan")
        )
    for key in ("task", "scenario"):
        if not _plain_string(raw.get(key)):
            diagnostics.append(_diagnostic("error", f"benchmark.result_{key}", f"{key} must be a non-empty string"))
        elif raw.get(key) != parameters.get(key):
            diagnostics.append(
                _diagnostic("error", f"benchmark.{key}_mismatch", f"result {key} does not match the approved plan")
            )
    for key in ("dataset_fingerprint", "model_fingerprint"):
        value = raw.get(key)
        if value is None or value == "":
            diagnostics.append(_diagnostic("error", f"benchmark.result_{key}", f"{key} is required"))
        elif value != parameters.get(key):
            diagnostics.append(
                _diagnostic("error", f"benchmark.{key}_mismatch", f"result {key} does not match the approved plan")
            )
    conventions = raw.get("conventions")
    if not isinstance(conventions, dict):
        diagnostics.append(_diagnostic("error", "benchmark.result_conventions", "conventions must be an object"))
    else:
        for key in ("energy_normalization", "stress_convention"):
            if not _plain_string(conventions.get(key)):
                diagnostics.append(_diagnostic("error", f"benchmark.result_{key}", f"conventions.{key} is required"))
            elif conventions.get(key) != parameters.get(key):
                diagnostics.append(
                    _diagnostic("error", f"benchmark.{key}_mismatch", f"result conventions.{key} does not match the approved plan")
                )

    metrics_raw = raw.get("metrics")
    metrics_path: Path | None = None
    if metrics_raw is None and raw.get("metrics_file") is not None:
        try:
            metrics_path = _portable_child(path.parent, raw["metrics_file"], "metrics_file")
            loaded_metrics = _read_json(metrics_path)
            metrics_raw = loaded_metrics.get("metrics") if isinstance(loaded_metrics, dict) and "metrics" in loaded_metrics else loaded_metrics
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            diagnostics.append(_diagnostic("error", "benchmark.metrics_unreadable", str(exc)))
    values, metric_diagnostics = _validate_metric_records(metrics_raw)
    diagnostics.extend(metric_diagnostics)

    raw_artifacts = raw.get("artifacts", [])
    if not isinstance(raw_artifacts, list):
        diagnostics.append(_diagnostic("error", "benchmark.artifacts_type", "artifacts must be a list"))
    else:
        for index, item in enumerate(raw_artifacts):
            if not isinstance(item, dict):
                diagnostics.append(_diagnostic("error", "benchmark.artifact_record", f"artifacts[{index}] must be an object"))
                continue
            try:
                artifact_path = _portable_child(path.parent, item.get("path"), f"artifacts[{index}].path")
            except ValueError as exc:
                diagnostics.append(_diagnostic("error", "benchmark.artifact_path", str(exc)))
                continue
            if not artifact_path.is_file():
                diagnostics.append(_diagnostic("error", "benchmark.artifact_missing", f"artifact is not a regular file: {artifact_path}"))
            artifacts.append(
                {
                    "path": str(artifact_path),
                    "role": item.get("role", "benchmark-output"),
                    "media_type": item.get("media_type"),
                }
            )
    artifacts.insert(0, {"path": str(path), "role": "result-manifest", "media_type": "application/json"})
    if metrics_path is not None:
        artifacts.append({"path": str(metrics_path), "role": "metrics", "media_type": "application/json"})
    return raw, values, artifacts, diagnostics


def _read_summary(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if path.stat().st_size > MAX_CSV_BYTES:
        raise ValueError(f"CSV artifact exceeds {MAX_CSV_BYTES} bytes: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("benchmark_summary.csv has no header")
        return list(reader.fieldnames), list(reader)


def _expected_sources(context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for spec in _evidence_specs(context):
        path = Path(spec["path"])
        if not path.is_file():
            raise ValueError(f"evidence input is not a regular file: {path}")
        if spec["locator"] in result:
            raise ValueError(f"duplicate portable evidence locator: {spec['locator']}")
        result[spec["locator"]] = {
            "path": path,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return result


def _normalized_metric_key(record: dict[str, Any]) -> str:
    dimensions = json.dumps(record.get("dimensions", {}), sort_keys=True, separators=(",", ":"))
    return "/".join(
        str(record.get(name, ""))
        for name in ("model", "task", "scenario", "split", "metric", "unit")
    ) + "/" + dimensions


def _load_normalized_outputs(
    context: dict[str, Any], paths: dict[str, Path]
) -> tuple[dict[str, Any], dict[str, float], list[dict[str, Any]]]:
    parameters = _mapping(context.get("parameters"))
    mode = _normalize_mode(_operation(context))
    expected_identities = _expected_identities(parameters)
    expected_sources = _expected_sources(context)

    metrics = _read_json(paths["metrics.json"])
    ranking = _read_json(paths["model_ranking.json"])
    provenance = _read_json(paths["provenance.json"])
    summary_fields, summary_rows = _read_summary(paths["benchmark_summary.csv"])
    for name, payload in (
        ("metrics.json", metrics),
        ("model_ranking.json", ranking),
        ("provenance.json", provenance),
    ):
        if not isinstance(payload, dict):
            raise ValueError(f"{name} must contain a JSON object")
        if payload.get("schema_version") != 1:
            raise ValueError(f"{name} schema_version must equal 1")
        if payload.get("plugin_id") != PLUGIN_ID:
            raise ValueError(f"{name} plugin_id must equal {PLUGIN_ID}")

    if metrics.get("mode") != mode or provenance.get("mode") != mode:
        raise ValueError("normalized metrics/provenance mode does not match the approved operation")
    if provenance.get("model_execution") is not False:
        raise ValueError("normalized provenance must state model_execution=false")
    if provenance.get("network_access") is not False:
        raise ValueError("normalized provenance must state network_access=false")
    if not _plain_string(provenance.get("wrapper_version")):
        raise ValueError("normalized provenance requires wrapper_version")

    records = metrics.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("metrics.json records must be a non-empty list")
    if metrics.get("record_count") != len(records):
        raise ValueError("metrics.json record_count does not match records")
    supported = metrics.get("supported_models")
    if not isinstance(supported, list) or set(supported) != set(MODEL_FAMILIES):
        raise ValueError("metrics.json supported_models does not name the six exact models")

    observed_identities = {name: set() for name in expected_identities}
    record_values: dict[str, float] = {}
    expected_source_pairs = {
        (locator, item["sha256"]) for locator, item in expected_sources.items()
    }
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"metrics.json records[{index}] must be an object")
        if record.get("mode") != mode:
            raise ValueError(f"metrics.json records[{index}] mode mismatch")
        for identity in observed_identities:
            value = record.get(identity)
            if not _plain_string(value):
                raise ValueError(f"metrics.json records[{index}] requires {identity}")
            observed_identities[identity].add(str(value))
        if record.get("model") not in MODEL_FAMILIES:
            raise ValueError(f"metrics.json records[{index}] uses an unsupported model")
        value = record.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"metrics.json records[{index}] value must be finite")
        if not _plain_string(record.get("metric")) or not _plain_string(record.get("unit")):
            raise ValueError(f"metrics.json records[{index}] requires metric and unit")
        if record.get("direction") not in {"minimize", "maximize"}:
            raise ValueError(f"metrics.json records[{index}] direction is invalid")
        sample_count = record.get("sample_count")
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
            raise ValueError(f"metrics.json records[{index}] sample_count must be positive")
        source_pair = (record.get("source_path"), record.get("evidence_sha256"))
        if source_pair not in expected_source_pairs:
            raise ValueError(
                f"metrics.json records[{index}] source locator/SHA does not match approved evidence"
            )
        if not _plain_string(record.get("source_format")) or not _plain_string(
            record.get("evidence_locator")
        ):
            raise ValueError(f"metrics.json records[{index}] source provenance is incomplete")
        key = _normalized_metric_key(record)
        if key in record_values:
            raise ValueError(f"metrics.json contains duplicate metric identity: {key}")
        record_values[key] = float(value)
    for identity, expected in expected_identities.items():
        if observed_identities[identity] != expected:
            raise ValueError(
                f"normalized {identity} identities {sorted(observed_identities[identity])} "
                f"do not match approved {sorted(expected)}"
            )

    source_evidence = provenance.get("source_evidence")
    if not isinstance(source_evidence, list) or len(source_evidence) != len(expected_sources):
        raise ValueError("provenance source_evidence count does not match approved inputs")
    actual_sources: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(source_evidence):
        if not isinstance(item, dict) or not _plain_string(item.get("path")):
            raise ValueError(f"provenance source_evidence[{index}] is invalid")
        locator = str(item["path"])
        if locator in actual_sources:
            raise ValueError(f"provenance contains duplicate source locator: {locator}")
        actual_sources[locator] = item
    if set(actual_sources) != set(expected_sources):
        raise ValueError("provenance source locators do not match approved evidence inputs")
    source_script = _source_script_spec(context)
    for locator, expected in expected_sources.items():
        item = actual_sources[locator]
        if (
            item.get("sha256") != expected["sha256"]
            or item.get("size_bytes") != expected["size_bytes"]
            or item.get("read_only_import") is not True
        ):
            raise ValueError(f"provenance fingerprint drift for evidence source {locator}")
        if not _plain_string(item.get("parser")) or not isinstance(
            item.get("normalized_record_count"), int
        ):
            raise ValueError(f"provenance parser/count missing for evidence source {locator}")
        if item["normalized_record_count"] != sum(
            record.get("source_path") == locator for record in records
        ):
            raise ValueError(f"provenance normalized record count drift for {locator}")
        original = item.get("original_implementation")
        if source_script is None:
            if original is not None:
                raise ValueError("unexpected source-script provenance in normalized outputs")
        else:
            script_path = Path(source_script["path"])
            if not script_path.is_file():
                raise ValueError(f"source script is not a regular file: {script_path}")
            if not isinstance(original, dict) or (
                original.get("path") != source_script["locator"]
                or original.get("sha256") != _sha256(script_path)
                or original.get("size_bytes") != script_path.stat().st_size
                or original.get("read_only_reference") is not True
            ):
                raise ValueError("source-script provenance does not match the approved source")

    output_artifacts = provenance.get("output_artifacts")
    if not isinstance(output_artifacts, list):
        raise ValueError("provenance output_artifacts must be a list")
    actual_output_records = {
        item.get("path"): item for item in output_artifacts if isinstance(item, dict)
    }
    expected_provenance_outputs = set(NORMALIZED_OUTPUT_NAMES) - {"provenance.json"}
    if set(actual_output_records) != expected_provenance_outputs:
        raise ValueError("provenance must fingerprint metrics, summary, and ranking exactly")
    for name in expected_provenance_outputs:
        item = actual_output_records[name]
        if item.get("sha256") != _sha256(paths[name]) or item.get(
            "size_bytes"
        ) != paths[name].stat().st_size:
            raise ValueError(f"provenance output fingerprint mismatch for {name}")

    if tuple(summary_fields) != SUMMARY_COLUMNS:
        raise ValueError("benchmark_summary.csv columns do not match the fixed contract")
    if len(summary_rows) != len(records):
        raise ValueError("benchmark_summary.csv row count does not match metrics.json")
    for index, (row, record) in enumerate(zip(summary_rows, records)):
        for field in (
            "model",
            "task",
            "scenario",
            "split",
            "metric",
            "unit",
            "direction",
            "mode",
            "evidence_sha256",
            "source_path",
            "source_format",
            "evidence_locator",
            "unit_provenance",
        ):
            if row.get(field) != str(record.get(field)):
                raise ValueError(f"benchmark_summary.csv row {index + 2} field {field} drift")
        try:
            row_value = float(row["value"])
            row_count = int(row["sample_count"])
            row_dimensions = json.loads(row["dimensions_json"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"benchmark_summary.csv row {index + 2} is malformed") from exc
        if row_value != float(record["value"]) or row_count != record["sample_count"]:
            raise ValueError(f"benchmark_summary.csv row {index + 2} value/count drift")
        if row_dimensions != record.get("dimensions"):
            raise ValueError(f"benchmark_summary.csv row {index + 2} dimensions drift")

    rankings = ranking.get("rankings")
    if not isinstance(rankings, list):
        raise ValueError("model_ranking.json rankings must be a list")
    for index, item in enumerate(rankings):
        if not isinstance(item, dict):
            raise ValueError(f"model_ranking.json rankings[{index}] must be an object")
        if item.get("task") not in expected_identities["task"] or item.get(
            "scenario"
        ) not in expected_identities["scenario"]:
            raise ValueError(f"model_ranking.json rankings[{index}] identity mismatch")
        for field in ("split", "metric", "unit"):
            if not _plain_string(item.get(field)):
                raise ValueError(f"model_ranking.json rankings[{index}] requires {field}")
        if item.get("direction") not in {"minimize", "maximize"}:
            raise ValueError(f"model_ranking.json rankings[{index}] direction is invalid")
        ranking_dimensions = item.get("dimensions")
        if not isinstance(ranking_dimensions, dict):
            raise ValueError(f"model_ranking.json rankings[{index}] dimensions must be an object")
        matching_records = [
            record
            for record in records
            if record.get("task") == item.get("task")
            and record.get("scenario") == item.get("scenario")
            and record.get("split") == item.get("split")
            and record.get("metric") == item.get("metric")
            and record.get("unit") == item.get("unit")
            and record.get("direction") == item.get("direction")
            and {
                key: value
                for key, value in record.get("dimensions", {}).items()
                if key not in {"sheet", "num_structures", "count_in_bin"}
            }
            == ranking_dimensions
        ]
        candidates = item.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"model_ranking.json rankings[{index}] needs candidates")
        if len(matching_records) != len(candidates):
            raise ValueError(f"model_ranking.json rankings[{index}] candidate count drift")
        expected_candidates = sorted(
            matching_records,
            key=lambda record: (
                record["value"]
                if item["direction"] == "minimize"
                else -record["value"],
                record["model"],
            ),
        )
        candidate_models = set()
        previous_value: float | None = None
        expected_rank = 0
        for position, (candidate, record) in enumerate(
            zip(candidates, expected_candidates), start=1
        ):
            if not isinstance(candidate, dict) or candidate.get("model") not in expected_identities[
                "model"
            ]:
                raise ValueError(f"model_ranking.json rankings[{index}] model mismatch")
            if candidate.get("evidence_sha256") not in {
                item["sha256"] for item in expected_sources.values()
            }:
                raise ValueError(f"model_ranking.json rankings[{index}] source SHA mismatch")
            value = candidate.get("value")
            rank = candidate.get("rank")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or isinstance(rank, bool)
                or not isinstance(rank, int)
                or rank < 1
            ):
                raise ValueError(f"model_ranking.json rankings[{index}] candidate is malformed")
            if previous_value is None or float(record["value"]) != previous_value:
                expected_rank = position
            previous_value = float(record["value"])
            if (
                candidate.get("model") != record.get("model")
                or float(candidate["value"]) != float(record["value"])
                or candidate.get("sample_count") != record.get("sample_count")
                or candidate.get("evidence_sha256") != record.get("evidence_sha256")
                or candidate.get("rank") != expected_rank
            ):
                raise ValueError(
                    f"model_ranking.json rankings[{index}] does not match metrics.json"
                )
            candidate_models.add(candidate["model"])
        if item.get("comparable_model_count") != len(candidate_models):
            raise ValueError(f"model_ranking.json rankings[{index}] model count drift")

    artifact_roles = {
        "metrics.json": ("normalized-metrics", "application/json"),
        "benchmark_summary.csv": ("benchmark-summary", "text/csv"),
        "model_ranking.json": ("model-ranking", "application/json"),
        "provenance.json": ("provenance", "application/json"),
    }
    artifacts = [
        {
            "path": str(paths[name]),
            "role": artifact_roles[name][0],
            "media_type": artifact_roles[name][1],
        }
        for name in NORMALIZED_OUTPUT_NAMES
    ]
    return metrics, record_values, artifacts


class Adapter:
    """Plan reviewed external CLIs and parse only their explicit artifacts."""

    def validate(self, context: dict[str, Any]) -> list[dict[str, str]]:
        diagnostics: list[dict[str, str]] = []
        if not isinstance(context, dict):
            return [_diagnostic("error", "context.type", "context must be an object")]
        for key in ("project_root", "attempt_dir"):
            if not _plain_string(context.get(key)):
                diagnostics.append(_diagnostic("error", f"context.{key}", f"{key} must be a non-empty path string"))
        inputs = context.get("inputs")
        parameters = context.get("parameters")
        resources = context.get("resources")
        if not isinstance(inputs, dict):
            diagnostics.append(_diagnostic("error", "context.inputs", "inputs must be an object"))
            inputs = {}
        if not isinstance(parameters, dict):
            diagnostics.append(_diagnostic("error", "context.parameters", "parameters must be an object"))
            parameters = {}
        if not isinstance(resources, dict):
            diagnostics.append(_diagnostic("error", "context.resources", "resources must be an object"))
        if context.get("backend") not in EXECUTION_BACKENDS:
            diagnostics.append(_diagnostic("error", "context.backend", "this adapter currently supports only the local backend"))

        operation = parameters.get("operation", "evaluate-static")
        if operation not in OPERATIONS:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "benchmark.operation",
                    "operation must be evaluate-static, collect-existing, normalize-replay, or normalize-execute",
                )
            )
            return diagnostics

        if operation in NORMALIZE_OPERATIONS:
            try:
                evidence = _evidence_specs(context)
                for item in evidence:
                    path = Path(item["path"])
                    if not path.is_file():
                        raise ValueError(f"evidence input is not a regular file: {path}")
                output_dir = _normalized_output_dir(context)
                if output_dir.exists() and not output_dir.is_dir():
                    raise ValueError(f"normalized output path is not a directory: {output_dir}")
                if output_dir.is_dir() and any(output_dir.iterdir()):
                    raise ValueError(
                        "normalized output directory must be fresh and empty; existing artifacts are never overwritten"
                    )
                _expected_identities(parameters)
                _units(parameters)
                split = parameters.get("split")
                if split not in (None, "") and (
                    not _plain_string(split) or not _IDENTIFIER.fullmatch(str(split))
                ):
                    raise ValueError("parameters.split must be a normalized identifier")
                source_script = _source_script_spec(context)
                if source_script is not None:
                    script_path = Path(source_script["path"])
                    if not script_path.is_file():
                        raise ValueError(f"source script is not a regular file: {script_path}")
                    if operation != "normalize-replay":
                        raise ValueError(
                            "source_script is supported only for normalize-replay provenance"
                        )
                if not BUNDLED_WRAPPER.is_file():
                    raise ValueError(f"bundled benchmark wrapper is missing: {BUNDLED_WRAPPER}")
            except (KeyError, OSError, TypeError, ValueError) as exc:
                diagnostics.append(
                    _diagnostic("error", "benchmark.normalize_contract", str(exc))
                )
            return diagnostics

        if parameters.get("model_family") not in MODEL_FAMILIES:
            supported = ", ".join(sorted(MODEL_FAMILIES))
            diagnostics.append(
                _diagnostic(
                    "error",
                    "benchmark.model_family",
                    f"model_family must be one of the exact reviewed names: {supported}",
                )
            )
        for key in (
            "task",
            "scenario",
            "energy_normalization",
            "stress_convention",
            "dataset_fingerprint",
            "model_fingerprint",
        ):
            if not _plain_string(parameters.get(key)):
                diagnostics.append(_diagnostic("error", f"benchmark.{key}", f"parameters.{key} is required"))

        required_inputs = ["result_manifest"]
        if operation == "evaluate-static":
            required_inputs.extend(["executable", "script", "model", "benchmark_dataset", "benchmark_config"])
        for key in required_inputs:
            if _reference(inputs.get(key)) is None:
                diagnostics.append(_diagnostic("error", f"benchmark.input_{key}", f"inputs.{key} is required"))
        executable = _reference(inputs.get("executable"))
        if executable is not None and Path(executable).name.lower() in SHELL_EXECUTABLES:
            diagnostics.append(_diagnostic("error", "benchmark.shell_forbidden", "a shell executable is not permitted"))
        if operation == "evaluate-static" and _reference(inputs.get("result_manifest")) is not None:
            try:
                _safe_attempt_output(context, inputs["result_manifest"])
            except (KeyError, ValueError) as exc:
                diagnostics.append(_diagnostic("error", "benchmark.output_scope", str(exc)))
        return diagnostics

    def plan(self, context: dict[str, Any]) -> dict[str, Any]:
        diagnostics = self.validate(context)
        parameters = _mapping(context.get("parameters")) if isinstance(context, dict) else {}
        if parameters.get("operation", "evaluate-static") == "collect-existing" and not diagnostics:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "benchmark.collect_is_read_only",
                    "collect-existing is read-only; call check/collect or replay instead of executing a plan",
                )
            )
        if diagnostics:
            return {
                "plugin_id": PLUGIN_ID,
                "status": "BLOCKED",
                "executable": False,
                "diagnostics": diagnostics,
            }
        operation = str(parameters.get("operation", "evaluate-static"))
        if operation in NORMALIZE_OPERATIONS:
            evidence = _evidence_specs(context)
            output_paths = _normalized_output_paths(context)
            argv = [sys.executable, str(BUNDLED_WRAPPER), _normalize_mode(operation)]
            for item in evidence:
                argv.extend(["--input", item["path"]])
            argv.extend(["--output-dir", str(_normalized_output_dir(context))])
            if len(evidence) == 1:
                argv.extend(["--evidence-locator", evidence[0]["locator"]])
            for parameter, flag in (
                ("model_family", "--model"),
                ("task", "--task"),
                ("scenario", "--scenario"),
                ("split", "--split"),
            ):
                value = parameters.get(parameter)
                if value not in (None, ""):
                    argv.extend([flag, str(value)])
            for target, unit in sorted(_units(parameters).items()):
                argv.extend(["--{}-unit".format(target), unit])
            source_script = _source_script_spec(context)
            if source_script is not None:
                argv.extend(["--source-script", source_script["path"]])
                argv.extend(["--source-script-locator", source_script["locator"]])
            input_fingerprints = [
                {
                    "path": item["locator"],
                    "sha256": _sha256(Path(item["path"])),
                    "size_bytes": Path(item["path"]).stat().st_size,
                }
                for item in evidence
            ]
            return {
                "plugin_id": PLUGIN_ID,
                "status": "READY",
                "executable": True,
                "argv": argv,
                "shell": False,
                "cwd": str(context["attempt_dir"]),
                "expected_outputs": [str(output_paths[name]) for name in NORMALIZED_OUTPUT_NAMES],
                "input_fingerprints": input_fingerprints,
                "calculation_claim": (
                    "imported-existing-evidence"
                    if operation == "normalize-replay"
                    else "metrics-recomputed-from-supplied-reference-prediction-pairs; no model execution"
                ),
                "resources": dict(_mapping(context.get("resources"))),
                "diagnostics": [],
            }
        inputs = _mapping(context["inputs"])
        result_manifest = _safe_attempt_output(context, inputs["result_manifest"])
        argv = [
            _reference(inputs["executable"]),
            _project_path(context, inputs["script"]),
            "--model-family",
            str(parameters["model_family"]),
            "--model",
            _project_path(context, inputs["model"]),
            "--dataset",
            _project_path(context, inputs["benchmark_dataset"]),
            "--config",
            _project_path(context, inputs["benchmark_config"]),
            "--task",
            str(parameters["task"]),
            "--scenario",
            str(parameters["scenario"]),
            "--energy-normalization",
            str(parameters["energy_normalization"]),
            "--stress-convention",
            str(parameters["stress_convention"]),
            "--dataset-fingerprint",
            str(parameters["dataset_fingerprint"]),
            "--model-fingerprint",
            str(parameters["model_fingerprint"]),
            "--result-manifest",
            result_manifest,
        ]
        return {
            "plugin_id": PLUGIN_ID,
            "status": "READY",
            "executable": True,
            "argv": argv,
            "shell": False,
            "cwd": str(context["attempt_dir"]),
            "expected_outputs": [result_manifest],
            "resources": dict(_mapping(context.get("resources"))),
            "diagnostics": [],
        }

    def prepare(self, context: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(plan, dict) or plan.get("status") != "READY":
            return {
                "plugin_id": PLUGIN_ID,
                "status": "BLOCKED",
                "executable": False,
                "diagnostics": [_diagnostic("error", "benchmark.plan_required", "a READY plan is required")],
            }
        return {
            "plugin_id": PLUGIN_ID,
            "status": "READY",
            "executable": False,
            "prepared": False,
            "message": "No files were written; the external wrapper consumes existing inputs.",
        }

    def check(self, context: dict[str, Any]) -> dict[str, Any]:
        if _operation(context) in NORMALIZE_OPERATIONS:
            try:
                paths = _normalized_output_paths(context)
            except (KeyError, TypeError, ValueError) as exc:
                return {
                    "plugin_id": PLUGIN_ID,
                    "status": "FAIL",
                    "diagnostics": [
                        _diagnostic("error", "benchmark.normalized_output_path", str(exc))
                    ],
                }
            existing = {name for name, path in paths.items() if path.is_file()}
            if not existing:
                return {
                    "plugin_id": PLUGIN_ID,
                    "status": "WAIT",
                    "diagnostics": [
                        _diagnostic(
                            "info",
                            "benchmark.normalized_outputs_pending",
                            "normalized benchmark four-artifact suite is not present",
                        )
                    ],
                }
            if existing != set(NORMALIZED_OUTPUT_NAMES):
                missing = sorted(set(NORMALIZED_OUTPUT_NAMES) - existing)
                return {
                    "plugin_id": PLUGIN_ID,
                    "status": "FAIL",
                    "diagnostics": [
                        _diagnostic(
                            "error",
                            "benchmark.normalized_outputs_partial",
                            "normalized benchmark output is partial; missing: " + ", ".join(missing),
                        )
                    ],
                }
            output_dir = _normalized_output_dir(context)
            try:
                children = {item.name for item in output_dir.iterdir()}
            except OSError as exc:
                return {
                    "plugin_id": PLUGIN_ID,
                    "status": "FAIL",
                    "diagnostics": [
                        _diagnostic("error", "benchmark.normalized_outputs_unreadable", str(exc))
                    ],
                }
            if children != set(NORMALIZED_OUTPUT_NAMES):
                unexpected = sorted(children - set(NORMALIZED_OUTPUT_NAMES))
                return {
                    "plugin_id": PLUGIN_ID,
                    "status": "FAIL",
                    "diagnostics": [
                        _diagnostic(
                            "error",
                            "benchmark.normalized_outputs_not_fixed",
                            "normalized output directory must contain exactly four artifacts; unexpected: "
                            + ", ".join(unexpected),
                        )
                    ],
                }
            try:
                payload, metric_values, artifacts = _load_normalized_outputs(context, paths)
            except (
                OSError,
                UnicodeError,
                csv.Error,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                return {
                    "plugin_id": PLUGIN_ID,
                    "status": "FAIL",
                    "diagnostics": [
                        _diagnostic("error", "benchmark.normalized_contract", str(exc))
                    ],
                }
            return {
                "plugin_id": PLUGIN_ID,
                "status": "OK",
                "mode": payload["mode"],
                "records": payload["records"],
                "metrics": metric_values,
                "artifacts": artifacts,
                "diagnostics": [],
            }
        try:
            path = _result_path(context)
        except (KeyError, TypeError, ValueError) as exc:
            return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": [_diagnostic("error", "benchmark.result_path", str(exc))]}
        if not path.is_file():
            return {
                "plugin_id": PLUGIN_ID,
                "status": "WAIT",
                "diagnostics": [_diagnostic("info", "benchmark.result_pending", f"result manifest not found: {path}")],
            }
        raw, metrics, artifacts, diagnostics = _load_result(path, context)
        status = "FAIL" if diagnostics or raw is None or raw.get("status") == "FAIL" else "OK"
        if raw is not None and raw.get("status") == "FAIL" and not diagnostics:
            diagnostics.append(_diagnostic("error", "benchmark.reported_failure", "benchmark result reports FAIL"))
        return {
            "plugin_id": PLUGIN_ID,
            "status": status,
            "diagnostics": diagnostics,
            "metrics": metrics,
            "artifacts": artifacts,
        }

    def collect(self, context: dict[str, Any]) -> dict[str, Any]:
        checked = self.check(context)
        if checked["status"] != "OK":
            return checked
        result = {
            "plugin_id": PLUGIN_ID,
            "status": "OK",
            "artifacts": checked["artifacts"],
            "metrics": checked["metrics"],
            "diagnostics": [],
        }
        if "mode" in checked:
            result["mode"] = checked["mode"]
            result["records"] = checked["records"]
        return result

    def replay(self, context: dict[str, Any]) -> dict[str, Any]:
        """Replay is collection only and therefore cannot execute numerical code."""

        return self.collect(context)
