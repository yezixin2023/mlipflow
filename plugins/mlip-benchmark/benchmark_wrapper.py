#!/usr/bin/env python3
"""Compatibility CLI for benchmark metric normalization.

The implementation lives in ``benchmark_normalization.py`` so this file stays
an intentionally small, stable argv entrypoint for existing MLIPFlow plans.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Sequence


def _implementation():
    source_root = Path(__file__).resolve().parents[2] / "src"
    if source_root.is_dir() and str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    path = Path(__file__).with_name("benchmark_normalization.py")
    spec = importlib.util.spec_from_file_location("mlip_benchmark_normalization", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load benchmark normalization implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
