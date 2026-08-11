"""Replay voltage errors from the manuscript SI Table S11 evidence table.

This module deliberately does not calculate voltages from DFT or MLIP total
energies.  It validates a transcription of the published voltage table and
recomputes prediction errors against the table's DFT reference column.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Sequence


SOURCE_TABLE = "Table S11"
MODEL_COLUMNS = (
    "deepmd-se_atten_v2",
    "deepmd-se_e2_a",
    "deepmd-se_e2_r",
    "deepmd-dpa2",
    "m3gnet",
    "chgnet",
)
CSV_FIELDS = (
    "source_table",
    "source_document_sha256",
    "unit_source",
    "prototype",
    "metal_composition",
    "lithium_sequence",
    "stage",
    "reference_dft_v",
    *MODEL_COLUMNS,
    "unit",
)
OUTPUT_NAMES = ("metrics.json", "model_ranking.json", "provenance.json")
DISCLAIMER = (
    "This artifact replays voltage values transcribed from SI Table S11; "
    "it does not recompute voltages from DFT or MLIP total energies."
)
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class ReplayValidationError(ValueError):
    """Raised when manuscript evidence cannot be replayed without ambiguity."""


def _text(value: str | None, *, field: str, row_number: int) -> str:
    if value is None:
        raise ReplayValidationError(f"row {row_number}: missing field {field!r}")
    normalized = value.strip()
    if not normalized:
        raise ReplayValidationError(f"row {row_number}: field {field!r} must not be empty")
    if any(character in normalized for character in ("\x00", "\r", "\n")):
        raise ReplayValidationError(
            f"row {row_number}: field {field!r} contains a forbidden control character"
        )
    return normalized


def _finite_float(value: str | None, *, field: str, row_number: int) -> float:
    normalized = _text(value, field=field, row_number=row_number)
    try:
        number = float(normalized)
    except ValueError as exc:
        raise ReplayValidationError(
            f"row {row_number}: field {field!r} must be a finite number"
        ) from exc
    if not math.isfinite(number):
        raise ReplayValidationError(
            f"row {row_number}: field {field!r} must be a finite number"
        )
    return number


def _parse_csv(raw: bytes) -> tuple[list[dict[str, Any]], dict[str, str]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReplayValidationError("input CSV must be UTF-8 without a byte-order mark") from exc

    try:
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            expected = ",".join(CSV_FIELDS)
            actual = ",".join(reader.fieldnames or ())
            raise ReplayValidationError(
                f"CSV header must exactly match the required fields and order; "
                f"expected {expected!r}, got {actual!r}"
            )
        csv_rows = list(reader)
    except csv.Error as exc:
        raise ReplayValidationError(f"invalid CSV syntax: {exc}") from exc

    if not csv_rows:
        raise ReplayValidationError("input CSV must contain at least one evidence row")

    rows: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str, str]] = set()
    source_document_sha256: str | None = None
    unit_source: str | None = None

    for row_number, csv_row in enumerate(csv_rows, start=2):
        if None in csv_row:
            raise ReplayValidationError(f"row {row_number}: unexpected extra CSV value")
        source_table = _text(
            csv_row.get("source_table"), field="source_table", row_number=row_number
        )
        if source_table != SOURCE_TABLE:
            raise ReplayValidationError(
                f"row {row_number}: source_table must be exactly {SOURCE_TABLE!r}"
            )

        row_document_sha256 = _text(
            csv_row.get("source_document_sha256"),
            field="source_document_sha256",
            row_number=row_number,
        ).lower()
        if not _SHA256_PATTERN.fullmatch(row_document_sha256):
            raise ReplayValidationError(
                f"row {row_number}: source_document_sha256 must be 64 hexadecimal characters"
            )
        if source_document_sha256 is None:
            source_document_sha256 = row_document_sha256
        elif row_document_sha256 != source_document_sha256:
            raise ReplayValidationError(
                f"row {row_number}: source_document_sha256 differs from earlier rows"
            )

        row_unit_source = _text(
            csv_row.get("unit_source"), field="unit_source", row_number=row_number
        )
        if unit_source is None:
            unit_source = row_unit_source
        elif row_unit_source != unit_source:
            raise ReplayValidationError(f"row {row_number}: unit_source differs from earlier rows")

        unit = _text(csv_row.get("unit"), field="unit", row_number=row_number)
        if unit != "V":
            raise ReplayValidationError(f"row {row_number}: unit must be exactly 'V'")

        prototype = _text(
            csv_row.get("prototype"), field="prototype", row_number=row_number
        )
        composition = _text(
            csv_row.get("metal_composition"),
            field="metal_composition",
            row_number=row_number,
        )
        lithium_sequence = _text(
            csv_row.get("lithium_sequence"),
            field="lithium_sequence",
            row_number=row_number,
        )
        stage = _text(csv_row.get("stage"), field="stage", row_number=row_number)
        identity = (prototype, composition, lithium_sequence, stage)
        if identity in identities:
            raise ReplayValidationError(
                "row "
                f"{row_number}: duplicate evidence row for "
                f"{prototype}/{composition}/{lithium_sequence}/{stage}"
            )
        identities.add(identity)

        reference = _finite_float(
            csv_row.get("reference_dft_v"), field="reference_dft_v", row_number=row_number
        )
        predictions = {
            model: _finite_float(csv_row.get(model), field=model, row_number=row_number)
            for model in MODEL_COLUMNS
        }
        signed_errors = {model: predictions[model] - reference for model in MODEL_COLUMNS}
        absolute_errors = {model: abs(signed_errors[model]) for model in MODEL_COLUMNS}
        rows.append(
            {
                "row_id": f"row-{len(rows) + 1:04d}",
                "prototype": prototype,
                "metal_composition": composition,
                "lithium_sequence": lithium_sequence,
                "stage": stage,
                "reference_dft_v": reference,
                "predictions_v": predictions,
                "signed_errors_v": signed_errors,
                "absolute_errors_v": absolute_errors,
            }
        )

    if source_document_sha256 is None or unit_source is None:  # pragma: no cover - guarded above
        raise AssertionError("non-empty CSV did not produce source metadata")
    return rows, {
        "source_table": SOURCE_TABLE,
        "source_document_sha256": source_document_sha256,
        "unit_source": unit_source,
    }


def _model_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for model in MODEL_COLUMNS:
        signed = [row["signed_errors_v"][model] for row in rows]
        absolute = [abs(error) for error in signed]
        count = len(signed)
        metrics.append(
            {
                "model": model,
                "N": count,
                "mae_v": math.fsum(absolute) / count,
                "rmse_v": math.sqrt(math.fsum(error * error for error in signed) / count),
                "max_abs_error_v": max(absolute),
            }
        )
    return metrics


def build_artifacts(raw: bytes) -> dict[str, dict[str, Any]]:
    """Validate a Table S11 CSV byte stream and construct deterministic artifacts."""

    rows, source = _parse_csv(raw)
    model_metrics = _model_metrics(rows)
    ordered = sorted(
        model_metrics,
        key=lambda item: (
            item["mae_v"],
            item["rmse_v"],
            item["max_abs_error_v"],
            item["model"],
        ),
    )
    ranking = [
        {"rank": rank, **metrics}
        for rank, metrics in enumerate(ordered, start=1)
    ]
    evidence_sha256 = hashlib.sha256(raw).hexdigest()
    common = {
        "schema_version": 1,
        "evidence_mode": "manuscript-table-replay",
        "source_table": source["source_table"],
        "voltage_unit": "V",
        "disclaimer": DISCLAIMER,
    }
    return {
        "metrics.json": {
            **common,
            "artifact_type": "manuscript-voltage-replay-metrics",
            "reference_column": "reference_dft_v",
            "row_count": len(rows),
            "models": model_metrics,
            "row_errors": rows,
        },
        "model_ranking.json": {
            **common,
            "artifact_type": "manuscript-voltage-model-ranking",
            "ranking_metric": "mae_v",
            "direction": "minimize",
            "tie_breakers": ["rmse_v", "max_abs_error_v", "model"],
            "ranking": ranking,
            "winner": ranking[0]["model"],
        },
        "provenance.json": {
            **common,
            "artifact_type": "manuscript-voltage-replay-provenance",
            "source_document_sha256": source["source_document_sha256"],
            "input_evidence_sha256": evidence_sha256,
            "unit_source": source["unit_source"],
            "model_execution": False,
            "dft_execution": False,
            "recomputed_from_total_energies": False,
            "recomputed_quantities": [
                "signed_error_v",
                "absolute_error_v",
                "mae_v",
                "rmse_v",
                "max_abs_error_v",
                "model_ranking",
            ],
        },
    }


def replay_table(input_csv: Path | str, output_dir: Path | str) -> dict[str, Path]:
    """Replay a manuscript CSV into new files, refusing to overwrite artifacts."""

    input_path = Path(input_csv)
    destination = Path(output_dir)
    try:
        raw = input_path.read_bytes()
    except OSError as exc:
        raise ReplayValidationError(f"cannot read input CSV: {exc}") from exc
    artifacts = build_artifacts(raw)

    if destination.exists() and not destination.is_dir():
        raise ReplayValidationError(f"output path exists and is not a directory: {destination}")
    output_paths = {name: destination / name for name in OUTPUT_NAMES}
    existing = [path for path in output_paths.values() if path.exists()]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise ReplayValidationError(f"refusing to overwrite existing output(s): {names}")

    try:
        destination.mkdir(parents=True, exist_ok=True)
        for name in OUTPUT_NAMES:
            with output_paths[name].open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(artifacts[name], stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
    except OSError as exc:
        raise ReplayValidationError(f"cannot write replay outputs: {exc}") from exc
    return output_paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay model errors from a strict SI Table S11 voltage CSV."
    )
    parser.add_argument("--input", required=True, type=Path, help="strict UTF-8 Table S11 CSV")
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="directory for three new JSON artifacts"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        paths = replay_table(args.input, args.output_dir)
    except ReplayValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({name: str(path) for name, path in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess smoke tests
    raise SystemExit(main())
