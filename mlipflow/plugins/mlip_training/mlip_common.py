"""Shared validation and provenance helpers for bundled MLIP runners."""

import importlib.metadata
import json
import math
import os
import random
import re
from pathlib import Path

FRAMEWORKS = ("deepmd", "m3gnet", "chgnet", "mace")
OPERATIONS = ("train", "finetune")
PLUGIN_ID = "mlip-training"


class TrainingError(RuntimeError):
    pass


def mapping(value, field):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TrainingError(f"{field} must be an object")
    return dict(value)


def load_config(path):
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        value = yaml.safe_load(text)
    else:
        value = json.loads(text)
    if not isinstance(value, dict):
        raise TrainingError("training config must be an object")
    return dict(value)


def section(config, framework):
    if config.get("framework") not in {None, framework}:
        raise TrainingError("config framework mismatch")
    return mapping(config.get(framework), framework)


def version(*names):
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return "unknown"


def seed_all(seed):
    random.seed(seed)


def work_dir(result_manifest, framework):
    path = result_manifest.parent / f"{framework}-work"
    if path.exists():
        raise TrainingError(f"work directory already exists: {path}")
    path.mkdir(parents=True)
    return path


def records(path):
    files = sorted(path.rglob("*.json*")) if path.is_dir() else [path]
    for file in files:
        if file.suffix.lower() == ".jsonl":
            for line in file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    yield json.loads(line)
        else:
            value = json.loads(file.read_text(encoding="utf-8"))
            items = (
                value if isinstance(value, list) else value.get("records", value.get("data", []))
            )
            for item in items:
                yield item


def predefined_split_files(path, suffix):
    """Return the fixed train/validation/test files for an assembled dataset."""
    if not path.is_dir():
        return None
    files = {
        "train": path / f"train.{suffix}",
        "validation": path / f"valid.{suffix}",
        "test": path / f"test.{suffix}",
    }
    missing = [name for name, file in files.items() if not file.is_file()]
    if missing:
        raise TrainingError("predefined dataset split lacks: " + ", ".join(missing))
    return files


def predefined_split_record(files):
    split_ids, record_ids = set(), {}
    for name, path in files.items():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TrainingError(f"cannot read predefined {name} split: {exc}") from exc
        split_id = value.get("split_id") if isinstance(value, dict) else None
        items = value.get("records") if isinstance(value, dict) else None
        ids = [item.get("record_id") for item in items] if isinstance(items, list) else None
        if not isinstance(split_id, str) or not split_id or not isinstance(ids, list) or any(
            not isinstance(item, str) or not item for item in ids
        ):
            raise TrainingError(f"predefined {name} split lacks split_id or record IDs")
        split_ids.add(split_id)
        record_ids[name] = ids
    if len(split_ids) != 1:
        raise TrainingError("predefined dataset files have different split_id values")
    flattened = [item for name in ("train", "validation", "test") for item in record_ids[name]]
    if len(flattened) != len(set(flattened)):
        raise TrainingError("predefined dataset split contains duplicate record IDs")
    return {
        "source": "predefined",
        "split_id": split_ids.pop(),
        "train_record_ids": record_ids["train"],
        "validation_record_ids": record_ids["validation"],
        "test_record_ids": record_ids["test"],
    }


def write_result(args, status, framework_version, metrics, media_type, provenance, error=None):
    result = Path(args.result_manifest).absolute()
    output = Path(args.output).absolute()
    rel = Path(os.path.relpath(output, result.parent))
    if rel.is_absolute() or ".." in rel.parts:
        raise TrainingError("model output must remain under result directory")
    clean = {}
    for key, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TrainingError(f"metric {key!r} must be a numeric int or float")
        try:
            numeric = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise TrainingError(f"metric {key!r} must be a finite number") from exc
        if not math.isfinite(numeric):
            raise TrainingError(f"metric {key!r} must be a finite number")
        clean[re.sub(r"[^A-Za-z0-9_]+", "_", str(key)).strip("_").lower() or "metric"] = numeric
    payload = {
        "schema_version": 1,
        "plugin_id": PLUGIN_ID,
        "status": status,
        "framework": args.framework,
        "framework_version": framework_version or "unknown",
        "operation": args.operation,
        "seed": args.seed,
        "device": args.device,
        "precision": args.precision,
        "dataset_path": args.data,
        "config_path": args.config,
        "metrics": clean,
        "provenance": dict(provenance),
    }
    if args.operation == "finetune":
        payload["foundation_model_path"] = args.foundation_model
    if status == "OK":
        payload["model_artifact"] = {
            "path": rel.as_posix(),
            "media_type": media_type,
        }
    if error:
        payload["error"] = error
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
