"""Executable wrapper for deterministic offline active-learning operations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

try:
    from mlipflow.science.active_learning import ActiveLearningError, execute_operation
except ModuleNotFoundError:
    source_root = Path(__file__).resolve().parents[2] / "src"
    if source_root.is_dir() and str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    for module_name in list(sys.modules):
        if module_name == "mlipflow" or module_name.startswith("mlipflow."):
            del sys.modules[module_name]
    from mlipflow.science.active_learning import ActiveLearningError, execute_operation


PLUGIN_ID = "active-learning"
OPERATIONS = ("committee-evaluate", "select-candidates", "assess-round")


def load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ActiveLearningError(f"input is not an ordinary file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ActiveLearningError(f"cannot read mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ActiveLearningError(f"input root must be an object: {path}")
    return value


def load_direct_selection(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".csv":
        return load_mapping(path)
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = [dict(row) for row in csv.DictReader(stream)]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ActiveLearningError(f"cannot read DIRECT manifest: {exc}") from exc
    if not rows:
        raise ActiveLearningError("DIRECT manifest contains no selected rows")
    try:
        ordered = sorted(rows, key=lambda row: int(row["selected_order"]))
        indexes = [int(row["input_index"]) for row in ordered]
    except (KeyError, TypeError, ValueError) as exc:
        raise ActiveLearningError(
            "DIRECT CSV requires integer selected_order and input_index columns"
        ) from exc
    return {
        "schema_version": 1,
        "method": "DIRECT",
        "selected_input_indexes": indexes,
        "parameters": {"source_contract": "pes-sampling/direct-select-manifest.csv"},
    }


def validation_claim(values: Mapping[str, Any]) -> str:
    modes = []
    for value in values.values():
        if isinstance(value, Mapping):
            modes.extend(
                str(value.get(key, ""))
                for key in ("evidence_mode", "validation_claim")
                if value.get(key)
            )
    if any(mode == "ORACLE_REPLAY_VALIDATION" or mode == "oracle-replay" for mode in modes):
        return "ORACLE_REPLAY_VALIDATION"
    if any(mode == "FRESH_ACTIVE_LEARNING_ROUND" or mode == "fresh" for mode in modes):
        return "FRESH_ACTIVE_LEARNING_ROUND"
    return "CONTRACT_VALIDATION"


def build_result(operation: str, paths: Mapping[str, Path]) -> dict[str, Any]:
    values = {}
    for name, path in paths.items():
        values[name] = (
            load_direct_selection(path) if name == "direct_selection" else load_mapping(path)
        )
    result = execute_operation(operation, values)
    result["validation_claim"] = validation_claim(values)
    result["input_paths"] = {name: str(path) for name, path in sorted(paths.items())}
    return result


def write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("operation", choices=OPERATIONS)
    result.add_argument("--policy", type=Path, required=True)
    result.add_argument("--committee-predictions", type=Path)
    result.add_argument("--committee-evaluation", type=Path)
    result.add_argument("--candidate-manifest", type=Path)
    result.add_argument("--direct-selection", type=Path)
    result.add_argument("--selection-result", type=Path)
    result.add_argument("--audit-benchmark", type=Path)
    result.add_argument("--spot-checks", type=Path)
    result.add_argument("--labeling-result", type=Path)
    result.add_argument("--dataset-split", type=Path)
    result.add_argument("--round-history", type=Path)
    result.add_argument("--campaign", type=Path)
    result.add_argument("--transport-evidence", type=Path)
    result.add_argument("--result-manifest", type=Path, required=True)
    return result


def operation_paths(args: argparse.Namespace) -> dict[str, Path]:
    required = {
        "committee-evaluate": ("committee_predictions",),
        "select-candidates": ("committee_evaluation", "candidate_manifest"),
        "assess-round": (
            "committee_evaluation",
            "selection_result",
            "audit_benchmark",
            "spot_checks",
            "labeling_result",
            "dataset_split",
            "round_history",
            "campaign",
        ),
    }[args.operation]
    values: dict[str, Path] = {"policy": args.policy}
    for name in required:
        path = getattr(args, name)
        if path is None:
            raise ActiveLearningError(
                f"{args.operation} requires --{name.replace('_', '-')}"
            )
        values[name] = path
    if args.operation == "select-candidates" and args.direct_selection is not None:
        values["direct_selection"] = args.direct_selection
    if args.operation == "assess-round" and args.transport_evidence is not None:
        values["transport_evidence"] = args.transport_evidence
    return values


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = build_result(args.operation, operation_paths(args))
        write_new_json(args.result_manifest, result)
    except (ActiveLearningError, OSError, ValueError) as exc:
        parser().error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
