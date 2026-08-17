"""Deterministic single-metric candidate ranking.

This module is the executable implementation used by the
``candidate-ranking`` adapter.  It deliberately does not run an ML model or
calculate a property: it ranks explicit, already-computed metrics and records every
selection rule in a small JSON manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


PLUGIN_ID = "candidate-ranking"


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def build_result(
    candidate_manifest: dict[str, Any],
    metric_results_manifest: dict[str, Any],
    *,
    metric: str,
    direction: str,
    top_k: int,
    missing_metric_policy: str,
    input_fingerprints: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Validate the two inputs and return a deterministic top-k result."""

    if candidate_manifest.get("schema_version") != 1:
        raise ValueError("candidate manifest schema_version must equal 1")
    if metric_results_manifest.get("schema_version") != 1:
        raise ValueError("metric-results manifest schema_version must equal 1")
    if not isinstance(metric, str) or not metric:
        raise ValueError("metric must be a non-empty string")
    if direction not in {"minimize", "maximize"}:
        raise ValueError("direction must be minimize or maximize")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if missing_metric_policy not in {"reject", "error"}:
        raise ValueError("missing_metric_policy must be reject or error")

    candidates = candidate_manifest.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate_manifest.candidates must be a non-empty array")
    candidate_ids: list[str] = []
    for index, candidate in enumerate(candidates):
        candidate_id = candidate.get("id") if isinstance(candidate, dict) else None
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"candidate {index} has no non-empty id")
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate candidate id: {candidate_id}")
        candidate_ids.append(candidate_id)

    results = metric_results_manifest.get("results")
    if not isinstance(results, list):
        raise ValueError("metric_results_manifest.results must be an array")
    values: dict[str, float] = {}
    for index, record in enumerate(results):
        if not isinstance(record, dict):
            raise ValueError(f"metric record {index} must be an object")
        candidate_id = record.get("candidate_id")
        if candidate_id not in candidate_ids:
            raise ValueError(f"metric record references unknown candidate: {candidate_id!r}")
        if candidate_id in values:
            raise ValueError(f"duplicate metric record: {candidate_id}")
        metrics = record.get("metrics")
        value = metrics.get(metric) if isinstance(metrics, dict) else None
        if not _finite(value):
            raise ValueError(f"metric record {candidate_id!r} has no finite metric {metric!r}")
        values[candidate_id] = float(value)

    missing = sorted(set(candidate_ids) - set(values))
    if missing_metric_policy == "error" and missing:
        raise ValueError(f"{len(missing)} candidates have no metric {metric!r}")
    reverse = direction == "maximize"
    ordered = sorted(values.items(), key=lambda item: ((-item[1]) if reverse else item[1], item[0]))
    selected = ordered[: min(top_k, len(ordered))]
    return {
        "schema_version": 1,
        "plugin_id": PLUGIN_ID,
        "status": "OK",
        "rule": {
            "metric": metric,
            "direction": direction,
            "top_k": top_k,
            "missing_metric_policy": missing_metric_policy,
        },
        "candidate_count": len(candidate_ids),
        "evaluated_count": len(values),
        "excluded_missing": missing,
        "ranked_candidates": [
            {"candidate_id": candidate_id, "rank": rank, "value": value}
            for rank, (candidate_id, value) in enumerate(selected, start=1)
        ],
        "input_fingerprints": dict(input_fingerprints or {}),
        "implementation": {
            "name": "mlipflow-deterministic-single-metric-ranking",
            "version": 1,
            "tie_break": "candidate_id-ascending",
        },
    }


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    """Create *path* once; never replace an existing scientific result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--metric-results-manifest", type=Path, required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--direction", choices=("minimize", "maximize"), required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--missing-metric-policy", choices=("reject", "error"), required=True)
    parser.add_argument("--result-manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        candidate_manifest = _load_object(args.candidate_manifest)
        metric_results_manifest = _load_object(args.metric_results_manifest)
        result = build_result(
            candidate_manifest,
            metric_results_manifest,
            metric=args.metric,
            direction=args.direction,
            top_k=args.top_k,
            missing_metric_policy=args.missing_metric_policy,
            input_fingerprints={
                "candidate_manifest": _sha256(args.candidate_manifest),
                "metric_results_manifest": _sha256(args.metric_results_manifest),
            },
        )
        _write_new_json(args.result_manifest, result)
    except (OSError, ValueError) as exc:
        _parser().error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
