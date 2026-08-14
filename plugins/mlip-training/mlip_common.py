"""Shared validation and provenance helpers for bundled MLIP runners."""

import hashlib
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
    os.environ["PYTHONHASHSEED"] = str(seed)
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


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


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
        "dataset_fingerprint": args.dataset_fingerprint,
        "config_fingerprint": args.config_fingerprint,
        "metrics": clean,
        "provenance": dict(provenance),
    }
    if args.operation == "finetune":
        payload["foundation_model_fingerprint"] = args.foundation_model_fingerprint
    if status == "OK":
        payload["model_artifact"] = {
            "path": rel.as_posix(),
            "media_type": media_type,
            "sha256": _sha(output),
            "size_bytes": output.stat().st_size,
        }
    if error:
        payload["error"] = error
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
