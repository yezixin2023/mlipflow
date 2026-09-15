"""Normalize real benchmark evidence without importing or running an MLIP.

Two deliberately different modes are provided:

``replay``
    Import metrics that were already computed by a manuscript workflow. Every
    normalized record names its source path. No numerical benchmark is claimed
    to have been rerun.

``execute``
    Recompute MAE, RMSE, Pearson correlation, and—when explicit N×3 force
    vectors are supplied—the maximum atomic force-vector error.  This executes
    metric calculations only; it never loads a model or generates predictions.

The implementation has no third-party dependencies.  In particular, the XLSX
reader is a small, read-only OOXML reader so the two historical pandas/openpyxl
workbooks can be replayed in a minimal MLIPFlow installation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import posixpath
import re
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence
from xml.etree import ElementTree

if __package__:
    from mlipflow.plugins import model_runtime
else:
    try:
        import model_runtime
    except ModuleNotFoundError as exc:
        if exc.name != "model_runtime":
            raise
        # Direct source-script execution uses the shared file in its parent package.
        # Uploaded bundles instead provide model_runtime.py beside this script.
        import importlib.util

        runtime_path = Path(__file__).resolve().parent.parent / "model_runtime.py"
        runtime_spec = importlib.util.spec_from_file_location("model_runtime", runtime_path)
        if runtime_spec is None or runtime_spec.loader is None:
            raise ImportError(f"missing bundled model runtime: {runtime_path}") from exc
        model_runtime = importlib.util.module_from_spec(runtime_spec)
        sys.modules["model_runtime"] = model_runtime
        runtime_spec.loader.exec_module(model_runtime)

FRESH_MODEL_FAMILIES = model_runtime.FRESH_MODEL_FAMILIES
MODEL_FAMILIES = model_runtime.MODEL_FAMILIES
RuntimeCompatibilityError = model_runtime.RuntimeCompatibilityError
canonical_model_family = model_runtime.canonical_model_family


PLUGIN_ID = "mlip-benchmark"
WRAPPER_VERSION = "1.1.0"
OUTPUT_NAMES = (
    "metrics.json",
    "benchmark_summary.csv",
    "model_ranking.json",
    "provenance.json",
)

MODEL_NAMES = MODEL_FAMILIES
FRESH_MODEL_NAMES = FRESH_MODEL_FAMILIES

_XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PACKAGE_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_METRIC_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_CELL_REFERENCE = re.compile(r"^([A-Z]+)[0-9]+$")


class BenchmarkNormalizationError(ValueError):
    """Raised when evidence is ambiguous, incomplete, or internally inconsistent."""


def _canonical_model(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkNormalizationError("every metric requires an explicit model")
    try:
        return canonical_model_family(value)
    except RuntimeCompatibilityError as exc:
        raise BenchmarkNormalizationError(str(exc)) from exc


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkNormalizationError(f"{field} must be a non-empty string")
    result = re.sub(r"[^a-z0-9._-]+", "-", value.strip().lower()).strip("-.")
    if not result or not _METRIC_ID.fullmatch(result):
        raise BenchmarkNormalizationError(f"cannot normalize {field}={value!r}")
    return result


def _metric_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkNormalizationError("metric must be a non-empty string")
    result = re.sub(r"[^a-z0-9._-]+", "_", value.strip().lower()).strip("_.-")
    result = result.replace("pearsonr", "pearson_r")
    if not result or not _METRIC_ID.fullmatch(result):
        raise BenchmarkNormalizationError(f"cannot normalize metric={value!r}")
    return result


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise BenchmarkNormalizationError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkNormalizationError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise BenchmarkNormalizationError(f"{field} must be finite")
    return result


def _positive_int(value: Any, field: str) -> int:
    result = _finite_float(value, field)
    if result < 1 or not result.is_integer():
        raise BenchmarkNormalizationError(f"{field} must be a positive integer")
    return int(result)


def _explicit_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkNormalizationError(f"{field} must be a non-empty string")
    result = value.strip()
    if "\x00" in result or "\n" in result or "\r" in result:
        raise BenchmarkNormalizationError(f"{field} must be a single plain-text line")
    return result


def _check_source(path_value: Any) -> Path:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise BenchmarkNormalizationError(f"benchmark evidence is not a regular file: {path}")
    return path


def _public_locator(value: Any, source_path: Path, field: str = "evidence_locator") -> str:
    """Return a stable public locator while keeping the resolved read path private."""

    if value in (None, ""):
        return source_path.name
    locator = _explicit_string(value, field).replace("\\", "/")
    lowered = locator.lower()
    if locator.startswith("/") or re.match(r"^[a-zA-Z]:/", locator) or lowered.startswith("file:"):
        raise BenchmarkNormalizationError(f"{field} must not expose an absolute local path")
    if "://" not in locator and ".." in Path(locator).parts:
        raise BenchmarkNormalizationError(f"{field} must not traverse parent directories")
    return locator


def _lower_keys(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key).strip().lower(): value for key, value in row.items() if key is not None}


def _first(row: dict[str, Any], *names: str) -> Any:
    lowered = _lower_keys(row)
    for name in names:
        if name.lower() in lowered and lowered[name.lower()] not in (None, ""):
            return lowered[name.lower()]
    return None


def _column_index(reference: str) -> int:
    match = _CELL_REFERENCE.match(reference)
    if match is None:
        raise BenchmarkNormalizationError(f"unsupported XLSX cell reference: {reference}")
    result = 0
    for char in match.group(1):
        result = result * 26 + ord(char) - ord("A") + 1
    return result - 1


def _xlsx_member(base: str, target: str) -> str:
    if target.startswith("/"):
        result = posixpath.normpath(target.lstrip("/"))
    else:
        result = posixpath.normpath(posixpath.join(posixpath.dirname(base), target))
    if result.startswith("../") or result == "..":
        raise BenchmarkNormalizationError(f"unsafe XLSX relationship target: {target}")
    return result


def _xlsx_cell(cell: ElementTree.Element, shared: list[str]) -> Any:
    cell_type = cell.get("t")
    if cell_type == "inlineStr":
        return "".join(item.text or "" for item in cell.findall(f".//{_XML_NS}t"))
    value_node = cell.find(f"{_XML_NS}v")
    if value_node is None or value_node.text is None:
        return None
    value = value_node.text
    if cell_type == "s":
        try:
            return shared[int(value)]
        except (IndexError, ValueError) as exc:
            raise BenchmarkNormalizationError("invalid XLSX shared-string reference") from exc
    if cell_type in {"str", "e"}:
        return value
    if cell_type == "b":
        return value == "1"
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def _read_xlsx_tables(path: Path) -> list[dict[str, Any]]:
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise BenchmarkNormalizationError(f"invalid XLSX evidence: {path}") from exc
    with archive:
        names = {item.filename for item in archive.infolist()}
        required = {"xl/workbook.xml", "xl/_rels/workbook.xml.rels"}
        if not required.issubset(names):
            raise BenchmarkNormalizationError("XLSX evidence is missing workbook metadata")

        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{_XML_NS}si"):
                shared.append("".join(node.text or "" for node in item.findall(f".//{_XML_NS}t")))

        relation_root = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            item.get("Id"): _xlsx_member("xl/workbook.xml", item.get("Target", ""))
            for item in relation_root.findall(f"{_PACKAGE_REL_NS}Relationship")
        }
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        tables: list[dict[str, Any]] = []
        for sheet in workbook.findall(f".//{_XML_NS}sheet"):
            title = sheet.get("name") or "unnamed"
            target = targets.get(sheet.get(f"{_REL_NS}id"))
            if target not in names:
                raise BenchmarkNormalizationError(f"XLSX sheet {title!r} has no readable target")
            worksheet = ElementTree.fromstring(archive.read(target))
            matrix: list[tuple[int, list[Any]]] = []
            for row in worksheet.findall(f".//{_XML_NS}sheetData/{_XML_NS}row"):
                cells: dict[int, Any] = {}
                for cell in row.findall(f"{_XML_NS}c"):
                    reference = cell.get("r")
                    if reference is None:
                        continue
                    cells[_column_index(reference)] = _xlsx_cell(cell, shared)
                if cells:
                    width = max(cells) + 1
                    matrix.append(
                        (int(row.get("r", len(matrix) + 1)), [cells.get(i) for i in range(width)])
                    )
            if not matrix:
                continue
            header = [str(item).strip() if item is not None else "" for item in matrix[0][1]]
            if not any(header):
                continue
            rows: list[dict[str, Any]] = []
            for row_number, values in matrix[1:]:
                if not any(value not in (None, "") for value in values):
                    continue
                record = {
                    header[index]: value
                    for index, value in enumerate(values)
                    if index < len(header) and header[index]
                }
                record["__sheet__"] = title
                record["__row__"] = row_number
                rows.append(record)
            tables.append({"sheet": title, "headers": header, "rows": rows})
        return tables


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise BenchmarkNormalizationError(f"CSV evidence has no header: {path}")
            rows = []
            for number, row in enumerate(reader, start=2):
                if not any(value not in (None, "") for value in row.values()):
                    continue
                record = dict(row)
                record["__sheet__"] = "csv"
                record["__row__"] = number
                rows.append(record)
            return rows
    except (OSError, UnicodeError, csv.Error) as exc:
        raise BenchmarkNormalizationError(f"cannot read CSV evidence: {path}") from exc


def _expand_metric_mapping(container: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = container.get("metrics")
    if not isinstance(metrics, dict):
        return [container]
    base = {key: value for key, value in container.items() if key != "metrics"}
    units = base.pop("units", {})
    directions = base.pop("directions", {})
    sample_counts = base.pop("sample_counts", {})
    rows: list[dict[str, Any]] = []
    for name, raw in metrics.items():
        if isinstance(raw, dict):
            row = dict(base)
            row.update(raw)
        else:
            row = dict(base)
            row["value"] = raw
        row["metric"] = name
        if "unit" not in row and isinstance(units, dict):
            row["unit"] = units.get(name)
        if "direction" not in row and isinstance(directions, dict):
            row["direction"] = directions.get(name)
        if "sample_count" not in row and isinstance(sample_counts, dict):
            row["sample_count"] = sample_counts.get(name)
        rows.append(row)
    return rows


def _json_rows(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        records = raw
        defaults: dict[str, Any] = {}
    elif isinstance(raw, dict):
        defaults = {
            key: value
            for key, value in raw.items()
            if key not in {"records", "models", "metrics", "schema_version", "plugin_id"}
        }
        if isinstance(raw.get("records"), list):
            records = raw["records"]
        elif isinstance(raw.get("models"), list):
            records = raw["models"]
        elif "metrics" in raw:
            records = [raw]
            defaults = {}
        elif "metric" in raw:
            records = [raw]
            defaults = {}
        else:
            raise BenchmarkNormalizationError(
                "JSON replay evidence needs records[], models[], a metrics object, or one metric record"
            )
    else:
        raise BenchmarkNormalizationError("JSON replay evidence must be an object or array")

    result: list[dict[str, Any]] = []
    for index, item in enumerate(records, start=1):
        if not isinstance(item, dict):
            raise BenchmarkNormalizationError(f"JSON evidence record {index} must be an object")
        row = dict(defaults)
        row.update(item)
        expanded = _expand_metric_mapping(row)
        for metric_row in expanded:
            metric_row["__sheet__"] = "json"
            metric_row["__row__"] = index
            result.append(metric_row)
    return result


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BenchmarkNormalizationError(f"cannot read JSON evidence: {path}") from exc
    return _json_rows(raw)


def _source_defaults(spec: dict[str, Any]) -> dict[str, Any]:
    units = spec.get("units")
    if units is None:
        units = {}
    if not isinstance(units, dict):
        raise BenchmarkNormalizationError("source units must be an object")
    return {
        "model": spec.get("model"),
        "task": spec.get("task"),
        "scenario": spec.get("scenario"),
        "split": spec.get("split"),
        "units": {str(key).lower(): value for key, value in units.items()},
    }


def _metric_unit(target: str, statistic: str, defaults: dict[str, Any]) -> tuple[str, str]:
    if statistic == "pearson_r":
        return "dimensionless", "defined-by-metric"
    value = defaults["units"].get(target)
    if value is None:
        return "source-unit-unspecified", "not-declared-by-source"
    return _explicit_string(value, f"{target} unit"), "declared-by-invocation"


def _record(
    *,
    model: Any,
    task: Any,
    scenario: Any,
    split: Any,
    metric: Any,
    value: Any,
    unit: Any,
    direction: Any,
    sample_count: Any,
    source_path: str,
    source_format: str,
    mode: str,
    locator: str,
    dimensions: dict[str, Any] | None = None,
    unit_provenance: str = "declared-by-source",
) -> dict[str, Any]:
    direction_value = _explicit_string(direction, "direction").lower()
    if direction_value not in {"minimize", "maximize"}:
        raise BenchmarkNormalizationError("direction must be minimize or maximize")
    unit_value = _explicit_string(unit, "unit")
    result = {
        "model": _canonical_model(model),
        "task": _identifier(task, "task"),
        "scenario": _identifier(scenario, "scenario"),
        "split": _identifier(split, "split"),
        "metric": _metric_name(metric),
        "value": _finite_float(value, "value"),
        "unit": unit_value,
        "direction": direction_value,
        "sample_count": _positive_int(sample_count, "sample_count"),
        "mode": mode,
        "source_path": source_path,
        "source_format": source_format,
        "evidence_locator": _explicit_string(locator, "evidence_locator"),
        "unit_provenance": unit_provenance,
        "dimensions": dimensions or {},
    }
    return result


def _required_context(row: dict[str, Any], defaults: dict[str, Any], name: str) -> Any:
    value = _first(row, name)
    return value if value not in (None, "") else defaults.get(name)


def _normalize_generic_rows(
    rows: Iterable[dict[str, Any]],
    *,
    defaults: dict[str, Any],
    source_locator: str,
    source_format: str,
    mode: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        sheet = str(row.get("__sheet__", source_format))
        row_number = row.get("__row__", "?")
        locator = f"{sheet}!row-{row_number}"
        dimensions = _first(row, "dimensions")
        if isinstance(dimensions, str) and dimensions.strip():
            try:
                dimensions = json.loads(dimensions)
            except json.JSONDecodeError as exc:
                raise BenchmarkNormalizationError(
                    f"dimensions must be valid JSON at {locator}"
                ) from exc
        if dimensions in (None, ""):
            dimensions = {}
        if not isinstance(dimensions, dict):
            raise BenchmarkNormalizationError(f"dimensions must be an object at {locator}")

        records.append(
            _record(
                model=_required_context(row, defaults, "model"),
                task=_required_context(row, defaults, "task"),
                scenario=_required_context(row, defaults, "scenario"),
                split=_required_context(row, defaults, "split") or "all",
                metric=_first(row, "metric", "metric_name"),
                value=_first(row, "value", "metric_value"),
                unit=_first(row, "unit"),
                direction=_first(row, "direction"),
                sample_count=_first(row, "sample_count", "n", "count"),
                source_path=source_locator,
                source_format=source_format,
                mode=mode,
                locator=locator,
                dimensions=dimensions,
                unit_provenance="declared-by-source",
            )
        )
    return records


def _is_chgnet(tables: Sequence[dict[str, Any]]) -> bool:
    required = {"split", "energy_mae", "energy_rmse", "force_mae", "force_rmse"}
    return any(required.issubset({item.lower() for item in table["headers"]}) for table in tables)


def _is_m3gnet(tables: Sequence[dict[str, Any]]) -> bool:
    required = {"split", "target", "r", "mae", "rmse", "n_struct", "n_scalar"}
    return any(required.issubset({item.lower() for item in table["headers"]}) for table in tables)


def _is_deepmd(tables: Sequence[dict[str, Any]]) -> bool:
    required = {
        "split",
        "n_structures(frames)",
        "n_force_components",
        "energy_pearson_r",
        "energy_rmse",
        "force_pearson_r",
        "force_rmse",
    }
    return any(required.issubset({item.lower() for item in table["headers"]}) for table in tables)


def _normalise_target(value: Any) -> str:
    target = _identifier(str(value), "target")
    aliases = {"e": "energy", "energies": "energy", "f": "force", "forces": "force"}
    aliases.update({"s": "stress", "stresses": "stress"})
    return aliases.get(target, target)


def _normalize_chgnet(
    tables: Sequence[dict[str, Any]],
    *,
    defaults: dict[str, Any],
    source_locator: str,
) -> list[dict[str, Any]]:
    if defaults.get("model") not in (None, ""):
        if _canonical_model(defaults["model"]) != "chgnet":
            raise BenchmarkNormalizationError("CHGNet workbook cannot be labelled as another model")
    task = defaults.get("task")
    scenario = defaults.get("scenario")
    records: list[dict[str, Any]] = []
    for table in tables:
        headers = {item.lower() for item in table["headers"]}
        if "split" not in headers or "energy_rmse" not in headers:
            continue
        for row in table["rows"]:
            split = _first(row, "split") or defaults.get("split") or "all"
            dimensions: dict[str, Any] = {
                "sheet": table["sheet"],
                "scope": "by-natoms" if _first(row, "natoms") is not None else "overall",
            }
            if _first(row, "natoms") is not None:
                dimensions["natoms"] = _positive_int(_first(row, "natoms"), "Natoms")
            if _first(row, "numstructures") is not None:
                dimensions["num_structures"] = _positive_int(
                    _first(row, "numstructures"), "NumStructures"
                )
            for target in ("energy", "force", "stress"):
                count = _first(row, f"{target}_n")
                for statistic, direction in (
                    ("pearson_r", "maximize"),
                    ("mae", "minimize"),
                    ("rmse", "minimize"),
                ):
                    source_column = (
                        f"{target}_pearsonr"
                        if statistic == "pearson_r"
                        else f"{target}_{statistic}"
                    )
                    value = _first(row, source_column)
                    if value in (None, ""):
                        continue
                    unit, unit_provenance = _metric_unit(target, statistic, defaults)
                    records.append(
                        _record(
                            model="chgnet",
                            task=task,
                            scenario=scenario,
                            split=split,
                            metric=f"{target}_{statistic}",
                            value=value,
                            unit=unit,
                            direction=direction,
                            sample_count=count,
                            source_path=source_locator,
                            source_format="chgnet-efs-metrics-xlsx",
                            mode="replay",
                            locator=f"{table['sheet']}!row-{row['__row__']}:{source_column}",
                            dimensions=dimensions,
                            unit_provenance=unit_provenance,
                        )
                    )
    if not records:
        raise BenchmarkNormalizationError("CHGNet workbook contains no finite metric records")
    return records


def _normalize_m3gnet(
    tables: Sequence[dict[str, Any]],
    *,
    defaults: dict[str, Any],
    source_locator: str,
) -> list[dict[str, Any]]:
    if defaults.get("model") not in (None, ""):
        if _canonical_model(defaults["model"]) != "m3gnet":
            raise BenchmarkNormalizationError("M3GNet workbook cannot be labelled as another model")
    records: list[dict[str, Any]] = []
    for table in tables:
        headers = {item.lower() for item in table["headers"]}
        if not {"split", "target", "r", "mae", "rmse", "n_scalar"}.issubset(headers):
            continue
        for row in table["rows"]:
            target = _normalise_target(_first(row, "target"))
            dimensions: dict[str, Any] = {
                "sheet": table["sheet"],
                "scope": "by-natoms" if _first(row, "natoms") is not None else "overall",
            }
            if _first(row, "n_struct") is not None:
                dimensions["num_structures"] = _positive_int(_first(row, "n_struct"), "N_struct")
            if _first(row, "natoms") is not None:
                dimensions["natoms"] = _positive_int(_first(row, "natoms"), "natoms")
            if _first(row, "count_in_bin") is not None:
                dimensions["count_in_bin"] = _positive_int(
                    _first(row, "count_in_bin"), "count_in_bin"
                )
            for source_column, statistic, direction in (
                ("r", "pearson_r", "maximize"),
                ("mae", "mae", "minimize"),
                ("rmse", "rmse", "minimize"),
            ):
                value = _first(row, source_column)
                if value in (None, ""):
                    continue
                unit, unit_provenance = _metric_unit(target, statistic, defaults)
                records.append(
                    _record(
                        model="m3gnet",
                        task=defaults.get("task"),
                        scenario=defaults.get("scenario"),
                        split=_first(row, "split") or defaults.get("split") or "all",
                        metric=f"{target}_{statistic}",
                        value=value,
                        unit=unit,
                        direction=direction,
                        sample_count=_first(row, "n_scalar"),
                        source_path=source_locator,
                        source_format="m3gnet-metrics-summary-xlsx",
                        mode="replay",
                        locator=f"{table['sheet']}!row-{row['__row__']}:{source_column}",
                        dimensions=dimensions,
                        unit_provenance=unit_provenance,
                    )
                )
    if not records:
        raise BenchmarkNormalizationError("M3GNet workbook contains no finite metric records")
    return records


def _normalize_deepmd(
    tables: Sequence[dict[str, Any]],
    *,
    defaults: dict[str, Any],
    source_locator: str,
) -> list[dict[str, Any]]:
    model = _canonical_model(defaults.get("model"))
    if not model.startswith("deepmd-"):
        raise BenchmarkNormalizationError("DeepMD workbook cannot be labelled as another model")
    records: list[dict[str, Any]] = []
    for table in tables:
        headers = {item.lower() for item in table["headers"]}
        if not {"split", "energy_rmse", "force_rmse"}.issubset(headers):
            continue
        for row in table["rows"]:
            dimensions: dict[str, Any] = {"sheet": table["sheet"], "scope": "overall"}
            for source_name, normalized_name in (
                ("n_dirs_input", "num_input_directories"),
                ("n_system_dirs", "num_system_directories"),
                ("n_atoms_total(frames*natoms)", "num_atom_frames"),
            ):
                value = _first(row, source_name)
                if value not in (None, ""):
                    dimensions[normalized_name] = _positive_int(value, source_name)
            energy_is_per_atom = _first(row, "energy_is_per_atom")
            if energy_is_per_atom not in (None, ""):
                if isinstance(energy_is_per_atom, bool):
                    normalized_energy_flag = energy_is_per_atom
                elif isinstance(energy_is_per_atom, str) and energy_is_per_atom.lower() in {
                    "true",
                    "false",
                }:
                    normalized_energy_flag = energy_is_per_atom.lower() == "true"
                elif energy_is_per_atom in {0, 1}:
                    normalized_energy_flag = bool(energy_is_per_atom)
                else:
                    raise BenchmarkNormalizationError("energy_is_per_atom must be boolean")
                dimensions["energy_is_per_atom"] = normalized_energy_flag
            for target, count_column in (
                ("energy", "n_structures(frames)"),
                ("force", "n_force_components"),
            ):
                count = _first(row, count_column)
                for source_column, statistic, direction in (
                    (f"{target}_pearson_r", "pearson_r", "maximize"),
                    (f"{target}_rmse", "rmse", "minimize"),
                ):
                    value = _first(row, source_column)
                    if value in (None, ""):
                        continue
                    unit, unit_provenance = _metric_unit(target, statistic, defaults)
                    records.append(
                        _record(
                            model=model,
                            task=defaults.get("task"),
                            scenario=defaults.get("scenario"),
                            split=_first(row, "split") or defaults.get("split") or "all",
                            metric=f"{target}_{statistic}",
                            value=value,
                            unit=unit,
                            direction=direction,
                            sample_count=count,
                            source_path=source_locator,
                            source_format="deepmd-metrics-xlsx",
                            mode="replay",
                            locator=f"{table['sheet']}!row-{row['__row__']}:{source_column}",
                            dimensions=dimensions,
                            unit_provenance=unit_provenance,
                        )
                    )
    if not records:
        raise BenchmarkNormalizationError("DeepMD workbook contains no finite metric records")
    return records


def _normalize_replay_source(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = _check_source(spec.get("path"))
    source_locator = _public_locator(spec.get("evidence_locator"), path)
    defaults = _source_defaults(spec)
    suffix = path.suffix.lower()
    if suffix == ".json":
        rows = _read_json_rows(path)
        parser = "prepared-json"
        records = _normalize_generic_rows(
            rows,
            defaults=defaults,
            source_locator=source_locator,
            source_format=parser,
            mode="replay",
        )
        sheets: list[str] = []
    elif suffix in {".csv", ".tsv"}:
        if suffix == ".tsv":
            raise BenchmarkNormalizationError(
                "TSV is not accepted; use explicit CSV, JSON, or XLSX"
            )
        rows = _read_csv_rows(path)
        parser = "prepared-csv"
        records = _normalize_generic_rows(
            rows,
            defaults=defaults,
            source_locator=source_locator,
            source_format=parser,
            mode="replay",
        )
        sheets = []
    elif suffix == ".xlsx":
        tables = _read_xlsx_tables(path)
        sheets = [str(table["sheet"]) for table in tables]
        if _is_chgnet(tables):
            parser = "chgnet-efs-metrics-xlsx"
            records = _normalize_chgnet(
                tables,
                defaults=defaults,
                source_locator=source_locator,
            )
        elif _is_m3gnet(tables):
            parser = "m3gnet-metrics-summary-xlsx"
            records = _normalize_m3gnet(
                tables,
                defaults=defaults,
                source_locator=source_locator,
            )
        elif _is_deepmd(tables):
            parser = "deepmd-metrics-xlsx"
            records = _normalize_deepmd(
                tables,
                defaults=defaults,
                source_locator=source_locator,
            )
        else:
            parser = "prepared-xlsx"
            rows = [row for table in tables for row in table["rows"]]
            records = _normalize_generic_rows(
                rows,
                defaults=defaults,
                source_locator=source_locator,
                source_format=parser,
                mode="replay",
            )
    else:
        raise BenchmarkNormalizationError(
            f"unsupported replay evidence extension {suffix!r}; use JSON, CSV, or XLSX"
        )
    if not records:
        raise BenchmarkNormalizationError(f"no metric records found in {path}")
    provenance = {
        "path": source_locator,
        "parser": parser,
        "sheets": sheets,
        "normalized_record_count": len(records),
        "read_only_import": True,
    }
    if parser == "chgnet-efs-metrics-xlsx":
        provenance["source_semantics"] = {
            "energy": "raw dataset energy and CHGNet e; normalization and unit not declared by workbook",
            "force": "all reported force components flattened before metrics",
            "stress": "all reported stress components flattened; no Voigt or sign conversion",
            "wrapper_behavior": "import values exactly as reported; perform no unit or stress conversion",
        }
    elif parser == "m3gnet-metrics-summary-xlsx":
        provenance["source_semantics"] = {
            "energy": "reported scalar energy metrics imported as written",
            "force": "reported flattened force metrics imported as written",
            "stress": "reported source-script stress handling imported as written",
            "wrapper_behavior": "perform no post-hoc unit, tensor-order, Voigt, or sign conversion",
        }
    elif parser == "deepmd-metrics-xlsx":
        provenance["source_semantics"] = {
            "energy": "per-atom normalization is taken from the workbook energy_is_per_atom field",
            "force": "force RMSE sample count is the reported flattened force-component count",
            "wrapper_behavior": "import reported Pearson/RMSE values without model execution or rescaling",
        }
    source_script_value = spec.get("source_script")
    if source_script_value not in (None, ""):
        source_script = _check_source(source_script_value)
        provenance["original_implementation"] = {
            "path": _public_locator(
                spec.get("source_script_locator"), source_script, "source_script_locator"
            ),
            "read_only_reference": True,
        }
    return records, provenance


def _parse_numeric_values(value: Any, field: str) -> list[float]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise BenchmarkNormalizationError(
                    f"{field} is not valid JSON numeric data"
                ) from exc
    if isinstance(value, (list, tuple)):
        result: list[float] = []
        for item in value:
            result.extend(_parse_numeric_values(item, field))
        return result
    return [_finite_float(value, field)]


def _atomic_force_errors(reference: Any, prediction: Any) -> list[float] | None:
    """Return per-atom L2 force errors only for explicit N×3 vector evidence."""

    parsed: list[Any] = []
    for value, field in ((reference, "reference"), (prediction, "prediction")):
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped.startswith("["):
                return None
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise BenchmarkNormalizationError(
                    f"{field} is not valid JSON numeric data"
                ) from exc
        parsed.append(value)
    ref_vectors, pred_vectors = parsed
    if (
        not isinstance(ref_vectors, (list, tuple))
        or not isinstance(pred_vectors, (list, tuple))
        or not ref_vectors
        or len(ref_vectors) != len(pred_vectors)
    ):
        return None
    errors: list[float] = []
    for atom_index, (ref_vector, pred_vector) in enumerate(
        zip(ref_vectors, pred_vectors)
    ):
        if (
            not isinstance(ref_vector, (list, tuple))
            or not isinstance(pred_vector, (list, tuple))
            or len(ref_vector) != 3
            or len(pred_vector) != 3
        ):
            return None
        ref = [
            _finite_float(item, f"reference force atom {atom_index}")
            for item in ref_vector
        ]
        pred = [
            _finite_float(item, f"prediction force atom {atom_index}")
            for item in pred_vector
        ]
        errors.append(math.sqrt(sum((b - a) ** 2 for a, b in zip(ref, pred))))
    return errors


def _paired_rows(path: Path) -> tuple[list[dict[str, Any]], str, list[str]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _read_json_rows(path), "paired-json", []
    if suffix == ".csv":
        return _read_csv_rows(path), "paired-csv", []
    if suffix == ".xlsx":
        tables = _read_xlsx_tables(path)
        return (
            [row for table in tables for row in table["rows"]],
            "paired-xlsx",
            [str(table["sheet"]) for table in tables],
        )
    raise BenchmarkNormalizationError("execute evidence must be JSON, CSV, or XLSX")


def _pearson(
    reference: Sequence[float], prediction: Sequence[float]
) -> tuple[float | None, str | None]:
    if len(reference) < 2:
        return None, "insufficient-scalar-pairs"
    ref_mean = sum(reference) / len(reference)
    pred_mean = sum(prediction) / len(prediction)
    numerator = sum((a - ref_mean) * (b - pred_mean) for a, b in zip(reference, prediction))
    ref_norm = math.sqrt(sum((a - ref_mean) ** 2 for a in reference))
    pred_norm = math.sqrt(sum((b - pred_mean) ** 2 for b in prediction))
    if ref_norm == 0.0 or pred_norm == 0.0:
        return None, "constant-reference-or-prediction"
    return numerator / (ref_norm * pred_norm), None


def _execute_source(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = _check_source(spec.get("path"))
    source_locator = _public_locator(spec.get("evidence_locator"), path)
    defaults = _source_defaults(spec)
    rows, parser, sheets = _paired_rows(path)
    groups: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        reference = _first(row, "reference", "reference_value", "y_true", "true")
        prediction = _first(row, "prediction", "prediction_value", "y_pred", "predicted")
        if reference is None or prediction is None:
            raise BenchmarkNormalizationError(
                f"execute row {row.get('__row__', '?')} needs reference and prediction"
            )
        ref_values = _parse_numeric_values(reference, "reference")
        pred_values = _parse_numeric_values(prediction, "prediction")
        if len(ref_values) != len(pred_values):
            raise BenchmarkNormalizationError("reference/prediction scalar counts differ")
        model = _canonical_model(_required_context(row, defaults, "model"))
        task = _identifier(_required_context(row, defaults, "task"), "task")
        scenario = _identifier(_required_context(row, defaults, "scenario"), "scenario")
        split = _identifier(_required_context(row, defaults, "split") or "all", "split")
        target = _normalise_target(_first(row, "target", "property"))
        unit = _first(row, "unit") or defaults["units"].get(target)
        unit = _explicit_string(unit, f"{target} unit")
        key = (model, task, scenario, split, target, unit)
        group = groups.setdefault(
            key,
            {
                "reference": [],
                "prediction": [],
                "locators": [],
                "atomic_force_errors": [],
                "force_vector_grouping_complete": True,
            },
        )
        group["reference"].extend(ref_values)
        group["prediction"].extend(pred_values)
        group["locators"].append(f"{row.get('__sheet__', parser)}!row-{row.get('__row__', '?')}")
        if target == "force":
            atomic_errors = _atomic_force_errors(reference, prediction)
            if atomic_errors is None:
                group["force_vector_grouping_complete"] = False
            else:
                group["atomic_force_errors"].extend(atomic_errors)
    records: list[dict[str, Any]] = []
    unavailable_metrics: list[dict[str, Any]] = []
    for key, values in sorted(groups.items()):
        model, task, scenario, split, target, unit = key
        reference = values["reference"]
        prediction = values["prediction"]
        errors = [pred - ref for ref, pred in zip(reference, prediction)]
        pearson, pearson_reason = _pearson(reference, prediction)
        calculated = {
            "mae": sum(abs(item) for item in errors) / len(errors),
            "rmse": math.sqrt(sum(item * item for item in errors) / len(errors)),
            "pearson_r": pearson,
        }
        for statistic in ("mae", "rmse", "pearson_r"):
            if calculated[statistic] is None:
                unavailable_metrics.append(
                    {
                        "model": model,
                        "task": task,
                        "scenario": scenario,
                        "split": split,
                        "metric": f"{target}_{statistic}",
                        "unit": "dimensionless",
                        "direction": "maximize",
                        "sample_count": len(reference),
                        "mode": "execute",
                        "source_path": source_locator,
                        "source_format": parser,
                        "evidence_locator": f"{len(values['locators'])} prepared row(s)",
                        "unit_provenance": "defined-by-metric",
                        "dimensions": {"scope": "overall", "target": target},
                        "reason": pearson_reason,
                    }
                )
                continue
            records.append(
                _record(
                    model=model,
                    task=task,
                    scenario=scenario,
                    split=split,
                    metric=f"{target}_{statistic}",
                    value=calculated[statistic],
                    unit="dimensionless" if statistic == "pearson_r" else unit,
                    direction="maximize" if statistic == "pearson_r" else "minimize",
                    sample_count=len(reference),
                    source_path=source_locator,
                    source_format=parser,
                    mode="execute",
                    locator=f"{len(values['locators'])} prepared row(s)",
                    dimensions={"scope": "overall", "target": target},
                    unit_provenance=(
                        "defined-by-metric" if statistic == "pearson_r" else "declared-by-source"
                    ),
                )
            )
        atomic_errors = values["atomic_force_errors"]
        if (
            target == "force"
            and values["force_vector_grouping_complete"]
            and atomic_errors
        ):
            records.append(
                _record(
                    model=model,
                    task=task,
                    scenario=scenario,
                    split=split,
                    metric="maximum_atomic_force_error",
                    value=max(atomic_errors),
                    unit=unit,
                    direction="minimize",
                    sample_count=len(atomic_errors),
                    source_path=source_locator,
                    source_format=parser,
                    mode="execute",
                    locator=f"{len(values['locators'])} prepared row(s)",
                    dimensions={
                        "scope": "overall",
                        "target": "force",
                        "error_norm": "atomic-l2-vector",
                        "aggregation": "maximum-over-atoms",
                    },
                    unit_provenance="declared-by-source",
                )
            )
    provenance = {
        "path": source_locator,
        "parser": parser,
        "sheets": sheets,
        "normalized_record_count": len(records),
        "unavailable_metric_count": len(unavailable_metrics),
        "unavailable_metrics": unavailable_metrics,
        "read_only_import": True,
    }
    return records, provenance


def _scientific_dimensions(record: dict[str, Any]) -> dict[str, Any]:
    """Drop workbook bookkeeping while retaining scientific comparison axes."""

    ignored = {"sheet", "num_structures", "count_in_bin"}
    return {key: value for key, value in record["dimensions"].items() if key not in ignored}


def _record_key(record: dict[str, Any]) -> tuple[str, ...]:
    return (
        record["model"],
        record["task"],
        record["scenario"],
        record["split"],
        record["metric"],
        record["unit"],
        json.dumps(_scientific_dimensions(record), sort_keys=True, separators=(",", ":")),
    )


def _validate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, ...], dict[str, Any]] = {}
    for record in records:
        key = _record_key(record)
        if key in seen:
            raise BenchmarkNormalizationError(
                "duplicate scientific metric identity: " + "/".join(key[:6])
            )
        seen[key] = record
    return sorted(records, key=_record_key)


def _ranking(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["dimensions"].get("scope") == "by-natoms":
            continue
        key = (
            record["task"],
            record["scenario"],
            record["split"],
            record["metric"],
            record["unit"],
            record["direction"],
            json.dumps(_scientific_dimensions(record), sort_keys=True, separators=(",", ":")),
        )
        groups[key].append(record)
    rankings: list[dict[str, Any]] = []
    for key, candidates in sorted(groups.items()):
        task, scenario, split, metric, unit, direction, dimensions_json = key
        candidates.sort(
            key=lambda item: (
                item["value"] if direction == "minimize" else -item["value"],
                item["model"],
            )
        )
        ranked = []
        previous: float | None = None
        rank = 0
        for position, item in enumerate(candidates, start=1):
            if previous is None or item["value"] != previous:
                rank = position
            previous = item["value"]
            ranked.append(
                {
                    "rank": rank,
                    "model": item["model"],
                    "value": item["value"],
                    "sample_count": item["sample_count"],
                }
            )
        rankings.append(
            {
                "task": task,
                "scenario": scenario,
                "split": split,
                "metric": metric,
                "unit": unit,
                "direction": direction,
                "dimensions": json.loads(dimensions_json),
                "comparable_model_count": len({item["model"] for item in candidates}),
                "candidates": ranked,
            }
        )
    return {
        "schema_version": 1,
        "plugin_id": PLUGIN_ID,
        "ranking_method": (
            "Per task/scenario/split/metric/unit/scientific-dimensions; numeric direction "
            "from each metric; ties share rank and model name is the deterministic tie-breaker."
        ),
        "rankings": rankings,
    }


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _summary_bytes(records: Sequence[dict[str, Any]]) -> bytes:
    columns = [
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
        "source_path",
        "source_format",
        "evidence_locator",
        "unit_provenance",
        "dimensions_json",
    ]
    temporary = tempfile.SpooledTemporaryFile(mode="w+", encoding="utf-8", newline="")
    try:
        writer = csv.DictWriter(temporary, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in columns if key != "dimensions_json"}
            row["dimensions_json"] = json.dumps(
                record["dimensions"], sort_keys=True, separators=(",", ":")
            )
            writer.writerow(row)
        temporary.seek(0)
        return temporary.read().encode("utf-8")
    finally:
        temporary.close()


def _write_atomic(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def normalize_benchmark(
    sources: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    mode: str = "replay",
    overwrite: bool = False,
) -> dict[str, Path]:
    """Normalize evidence and atomically write the four benchmark artifacts.

    ``sources`` entries require ``path`` and may supply ``model``, ``task``,
    ``scenario``, ``split``, ``units`` (a mapping keyed by target), and a
    portable ``evidence_locator``.  Resolved local read paths are never emitted.
    """

    if mode not in {"execute", "replay"}:
        raise BenchmarkNormalizationError("mode must be execute or replay")
    if not sources:
        raise BenchmarkNormalizationError("at least one source is required")
    output_dir = Path(output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise BenchmarkNormalizationError(f"output path is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {name: output_dir / name for name in OUTPUT_NAMES}
    source_paths = [_check_source(spec.get("path")) for spec in sources if isinstance(spec, dict)]
    for source_path in source_paths:
        for output_path in output_paths.values():
            same_resolved_path = source_path == output_path.resolve()
            same_existing_file = output_path.exists() and os.path.samefile(source_path, output_path)
            if same_resolved_path or same_existing_file:
                raise BenchmarkNormalizationError(
                    f"source evidence must remain read-only and cannot also be an output: {source_path}"
                )
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise BenchmarkNormalizationError(
            "refusing to overwrite benchmark artifacts without overwrite=True: "
            + ", ".join(existing)
        )

    records: list[dict[str, Any]] = []
    provenance_sources: list[dict[str, Any]] = []
    for spec in sources:
        if not isinstance(spec, dict):
            raise BenchmarkNormalizationError("every source specification must be an object")
        if mode == "replay":
            source_records, source_provenance = _normalize_replay_source(spec)
        else:
            source_records, source_provenance = _execute_source(spec)
        records.extend(source_records)
        provenance_sources.append(source_provenance)
    records = _validate_records(records)
    unavailable_metrics = sorted(
        [
            item
            for source in provenance_sources
            for item in source.get("unavailable_metrics", [])
        ],
        key=lambda item: (
            item["model"],
            item["task"],
            item["scenario"],
            item["split"],
            item["metric"],
            item["source_path"],
        ),
    )

    supported_models = (
        FRESH_MODEL_NAMES
        if any(record["model"] not in MODEL_NAMES for record in records)
        else MODEL_NAMES
    )
    metrics_payload = {
        "schema_version": 1,
        "plugin_id": PLUGIN_ID,
        "mode": mode,
        "calculation_claim": (
            "imported-existing-evidence"
            if mode == "replay"
            else "metrics-recomputed-from-supplied-reference-prediction-pairs"
        ),
        "supported_models": list(supported_models),
        "record_count": len(records),
        "records": records,
    }
    if mode == "execute":
        metrics_payload["unavailable_metrics"] = unavailable_metrics
    ranking_payload = _ranking(records)
    payloads = {
        "metrics.json": _json_bytes(metrics_payload),
        "benchmark_summary.csv": _summary_bytes(records),
        "model_ranking.json": _json_bytes(ranking_payload),
    }
    provenance_payload = {
        "schema_version": 1,
        "plugin_id": PLUGIN_ID,
        "wrapper_version": WRAPPER_VERSION if mode == "execute" else "1.0.0",
        "mode": mode,
        "model_execution": False,
        "network_access": False,
        "source_evidence": sorted(provenance_sources, key=lambda item: item["path"]),
        "output_artifacts": [{"path": name} for name in sorted(payloads)],
        "limitations": [
            (
                "Replay imports historical metrics and does not claim a fresh model evaluation."
                if mode == "replay"
                else "Execute recomputes metrics from supplied pairs but does not run an MLIP."
            ),
            "source-unit-unspecified means the historical source did not declare a reliable unit; "
            "it is not a physical unit inference.",
        ],
    }
    if mode == "execute":
        provenance_payload["unavailable_metrics"] = unavailable_metrics
        provenance_payload["limitations"].append(
            "Undefined Pearson correlations are recorded explicitly as unavailable and are not "
            "replaced with synthetic or non-finite values."
        )
    payloads["provenance.json"] = _json_bytes(provenance_payload)
    for name in OUTPUT_NAMES:
        _write_atomic(output_paths[name], payloads[name])
    return output_paths


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize manuscript benchmark evidence without running an MLIP."
    )
    parser.add_argument("mode", choices=("execute", "replay"))
    parser.add_argument(
        "--input", action="append", required=True, help="JSON, CSV, or XLSX evidence"
    )
    parser.add_argument(
        "--output-dir", required=True, help="directory receiving the four artifacts"
    )
    parser.add_argument(
        "--evidence-locator",
        help="portable public locator for one input (defaults to its basename)",
    )
    parser.add_argument(
        "--input-locator",
        action="append",
        help="portable locator paired by order with each --input",
    )
    parser.add_argument("--model", choices=FRESH_MODEL_NAMES)
    parser.add_argument("--task")
    parser.add_argument("--scenario")
    parser.add_argument("--split")
    parser.add_argument("--energy-unit")
    parser.add_argument("--force-unit")
    parser.add_argument("--stress-unit")
    parser.add_argument(
        "--source-script", help="optional read-only original implementation path for provenance"
    )
    parser.add_argument("--source-script-locator", help="portable locator for --source-script")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _cli_parser()
    arguments = parser.parse_args(argv)
    if arguments.evidence_locator is not None and arguments.input_locator is not None:
        parser.error("--evidence-locator and --input-locator are mutually exclusive")
    if arguments.evidence_locator is not None and len(arguments.input) != 1:
        parser.error("--evidence-locator can be used only with one --input")
    if arguments.input_locator is not None and len(arguments.input_locator) != len(arguments.input):
        parser.error("--input-locator must be repeated exactly once for each --input")
    locators = (
        arguments.input_locator
        if arguments.input_locator is not None
        else [arguments.evidence_locator] * len(arguments.input)
    )
    units = {
        key: value
        for key, value in {
            "energy": arguments.energy_unit,
            "force": arguments.force_unit,
            "stress": arguments.stress_unit,
        }.items()
        if value is not None
    }
    sources = [
        {
            "path": value,
            "evidence_locator": locator,
            "model": arguments.model,
            "task": arguments.task,
            "scenario": arguments.scenario,
            "split": arguments.split,
            "units": units,
            "source_script": arguments.source_script,
            "source_script_locator": arguments.source_script_locator,
        }
        for value, locator in zip(arguments.input, locators)
    ]
    try:
        outputs = normalize_benchmark(
            sources,
            Path(arguments.output_dir),
            mode=arguments.mode,
            overwrite=arguments.overwrite,
        )
    except BenchmarkNormalizationError as exc:
        print(f"benchmark normalization failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({name: str(path) for name, path in outputs.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
