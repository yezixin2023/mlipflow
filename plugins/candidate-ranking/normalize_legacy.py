"""Normalize legacy Li10 candidate and conductivity text files.

The manuscript-era files use two incompatible element orders:

* candidate rows: ``Zn,Fe,Cu,Ni,Mn -> structure``;
* conductivity rows: ``Mn#_Fe#_Ni#_Cu#_Zn# value``.

This read-only parser maps both forms by element label and writes the two
versioned manifests consumed by the candidate-ranking adapter. It does not
infer missing candidates or units.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any


TOKEN = re.compile(r"([A-Z][a-z]?)([0-9]+)")


def _symbols(value: str, field: str) -> tuple[str, ...]:
    symbols = tuple(part.strip() for part in value.split(",") if part.strip())
    if not symbols or len(symbols) != len(set(symbols)):
        raise ValueError(f"{field} must contain unique comma-separated element symbols")
    if any(not re.fullmatch(r"[A-Z][a-z]?", symbol) for symbol in symbols):
        raise ValueError(f"{field} contains an invalid element symbol")
    return symbols


def canonical_id(composition: dict[str, int], order: tuple[str, ...]) -> str:
    if set(composition) != set(order):
        raise ValueError("composition elements do not match canonical order")
    return "_".join(f"{symbol}{composition[symbol]}" for symbol in order)


def parse_candidate_line(
    line: str,
    *,
    source_order: tuple[str, ...],
    canonical_order: tuple[str, ...],
    expected_metal_sites: int | None,
) -> tuple[str, dict[str, int], str]:
    left, separator, right = line.partition("->")
    if not separator or not right.strip():
        raise ValueError("candidate row must be '<counts> -> <structure reference>'")
    raw_counts = [part.strip() for part in left.split(",")]
    if len(raw_counts) != len(source_order):
        raise ValueError("candidate count does not match candidate element order")
    try:
        counts = [int(value) for value in raw_counts]
    except ValueError as exc:
        raise ValueError("candidate counts must be integers") from exc
    if any(value < 0 for value in counts):
        raise ValueError("candidate counts must be non-negative")
    composition = dict(zip(source_order, counts))
    if expected_metal_sites is not None and sum(counts) != expected_metal_sites:
        raise ValueError(
            f"candidate has {sum(counts)} metal sites, expected {expected_metal_sites}"
        )
    return canonical_id(composition, canonical_order), composition, right.strip()


def parse_metric_line(
    line: str, *, canonical_order: tuple[str, ...], expected_metal_sites: int | None
) -> tuple[str, dict[str, int], float]:
    fields = line.split()
    if len(fields) != 2:
        raise ValueError("metric row must be '<element-labelled-id> <value>'")
    composition = {symbol: int(count) for symbol, count in TOKEN.findall(fields[0])}
    reconstructed = "_".join(f"{symbol}{count}" for symbol, count in TOKEN.findall(fields[0]))
    if reconstructed != fields[0] or set(composition) != set(canonical_order):
        raise ValueError("metric id must label every canonical element exactly once")
    if len(TOKEN.findall(fields[0])) != len(composition):
        raise ValueError("metric id repeats an element")
    if expected_metal_sites is not None and sum(composition.values()) != expected_metal_sites:
        raise ValueError(
            f"metric row has {sum(composition.values())} metal sites, expected {expected_metal_sites}"
        )
    try:
        value = float(fields[1])
    except ValueError as exc:
        raise ValueError("metric value must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError("metric value must be finite")
    return canonical_id(composition, canonical_order), composition, value


def _source(path: Path) -> dict[str, Any]:
    return {"locator": path.name}


def _nonempty_lines(path: Path) -> list[tuple[int, str]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    return [(index, line.strip()) for index, line in enumerate(lines, 1) if line.strip()]


def normalize(
    candidate_paths: list[Path],
    metric_paths: list[Path],
    *,
    candidate_order: tuple[str, ...],
    canonical_order: tuple[str, ...],
    expected_metal_sites: int | None,
    metric_name: str,
    metric_unit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if set(candidate_order) != set(canonical_order):
        raise ValueError("candidate and canonical element orders must contain the same elements")
    candidates: dict[str, dict[str, Any]] = {}
    for path in candidate_paths:
        for line_number, line in _nonempty_lines(path):
            try:
                candidate_id, composition, source_structure = parse_candidate_line(
                    line,
                    source_order=candidate_order,
                    canonical_order=canonical_order,
                    expected_metal_sites=expected_metal_sites,
                )
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            record = candidates.setdefault(
                candidate_id,
                {
                    "id": candidate_id,
                    "composition": {key: composition[key] for key in canonical_order},
                    "source_structures": [],
                },
            )
            if record["composition"] != {key: composition[key] for key in canonical_order}:
                raise ValueError(f"conflicting composition for candidate {candidate_id}")
            if source_structure not in record["source_structures"]:
                record["source_structures"].append(source_structure)

    metrics: dict[str, dict[str, Any]] = {}
    for path in metric_paths:
        for line_number, line in _nonempty_lines(path):
            try:
                candidate_id, composition, value = parse_metric_line(
                    line,
                    canonical_order=canonical_order,
                    expected_metal_sites=expected_metal_sites,
                )
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if candidate_id not in candidates:
                raise ValueError(
                    f"{path}:{line_number}: metric references candidate absent from candidate files: "
                    f"{candidate_id}"
                )
            existing = metrics.get(candidate_id)
            if existing is not None and not math.isclose(
                existing["value"], value, rel_tol=0.0, abs_tol=0.0
            ):
                raise ValueError(f"conflicting metric values for candidate {candidate_id}")
            if existing is None:
                metrics[candidate_id] = {
                    "value": value,
                    "composition": {key: composition[key] for key in canonical_order},
                    "sources": [],
                }
            metrics[candidate_id]["sources"].append(
                {"locator": path.name, "line": line_number}
            )

    common_provenance = {
        "normalizer": "mlipflow-legacy-li10-screening-v1",
        "candidate_element_order": list(candidate_order),
        "canonical_element_order": list(canonical_order),
        "expected_metal_sites": expected_metal_sites,
        "candidate_sources": [_source(path) for path in candidate_paths],
        "metric_sources": [_source(path) for path in metric_paths],
    }
    candidate_manifest = {
        "schema_version": 1,
        "candidates": [candidates[key] for key in sorted(candidates)],
        "provenance": common_provenance,
    }
    metric_results_manifest = {
        "schema_version": 1,
        "metric_definition": {
            "name": metric_name,
            "unit": metric_unit,
            "source_convention": "legacy-text-value-no-rescaling",
        },
        "results": [
            {
                "candidate_id": candidate_id,
                "metrics": {metric_name: metrics[candidate_id]["value"]},
                "metric_units": {metric_name: metric_unit},
                "composition": metrics[candidate_id]["composition"],
                "source_records": metrics[candidate_id]["sources"],
            }
            for candidate_id in sorted(metrics)
        ],
        "provenance": common_provenance,
    }
    return candidate_manifest, metric_results_manifest


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
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
    parser.add_argument("--candidate-list", type=Path, action="append", required=True)
    parser.add_argument("--metric-file", type=Path, action="append", required=True)
    parser.add_argument("--candidate-order", default="Zn,Fe,Cu,Ni,Mn")
    parser.add_argument("--canonical-order", default="Mn,Fe,Ni,Cu,Zn")
    parser.add_argument("--expected-metal-sites", type=int, default=28)
    parser.add_argument("--metric-name", required=True)
    parser.add_argument("--metric-unit", required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--metric-results-manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        candidate_order = _symbols(args.candidate_order, "candidate_order")
        canonical_order = _symbols(args.canonical_order, "canonical_order")
        if args.expected_metal_sites < 1:
            raise ValueError("expected_metal_sites must be positive")
        if not args.metric_name or not args.metric_unit:
            raise ValueError("metric name and unit must be non-empty")
        if args.candidate_manifest.absolute() == args.metric_results_manifest.absolute():
            raise ValueError("the two output manifests must be different files")
        candidates, metric_results = normalize(
            args.candidate_list,
            args.metric_file,
            candidate_order=candidate_order,
            canonical_order=canonical_order,
            expected_metal_sites=args.expected_metal_sites,
            metric_name=args.metric_name,
            metric_unit=args.metric_unit,
        )
        _write_new_json(args.candidate_manifest, candidates)
        try:
            _write_new_json(args.metric_results_manifest, metric_results)
        except Exception:
            args.candidate_manifest.unlink(missing_ok=True)
            raise
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
