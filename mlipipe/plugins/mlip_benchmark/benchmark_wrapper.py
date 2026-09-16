#!/usr/bin/env python3
"""Compatibility CLI for benchmark metric normalization.

The implementation lives in ``benchmark_normalization.py`` so this file stays
an intentionally small, stable argv entrypoint for existing MLIPipe plans.
"""

from __future__ import annotations

from typing import Sequence


def _implementation():
    if __package__:
        from . import benchmark_normalization as implementation
    else:
        import benchmark_normalization as implementation
    return implementation


_NORMALIZATION = _implementation()
MODEL_NAMES = _NORMALIZATION.MODEL_NAMES
OUTPUT_NAMES = _NORMALIZATION.OUTPUT_NAMES
BenchmarkNormalizationError = _NORMALIZATION.BenchmarkNormalizationError
normalize_benchmark = _NORMALIZATION.normalize_benchmark

# The fresh runner consumes the same metric and ranking implementation.  These
# private compatibility exports avoid a second numerical implementation while
# keeping its existing injected-predictor API stable.
_json_bytes = _NORMALIZATION._json_bytes
_execute_source = _NORMALIZATION._execute_source
_ranking = _NORMALIZATION._ranking
_summary_bytes = _NORMALIZATION._summary_bytes
_validate_records = _NORMALIZATION._validate_records
_write_atomic = _NORMALIZATION._write_atomic


def main(argv: Sequence[str] | None = None) -> int:
    return _NORMALIZATION.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
