from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "pes-sampling"
WRAPPER = PLUGIN / "lasp_ssw.py"
ALIAS = PLUGIN / "lasp_random_walk.py"
EXAMPLE = ROOT / "examples" / "lasp_random_walk" / "lasp.in"
RERUN_EXAMPLE = ROOT / "examples" / "lasp_random_walk" / "lasp-rerun.in"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lasp_random_walk_examples_are_valid_ssw_contracts():
    module = _load(WRAPPER, "lasp_ssw_example_test")
    for path in (EXAMPLE, RERUN_EXAMPLE):
        parameters = module._parse_lasp_input(path)
        module._validate_ssw_input(parameters)
        assert str(parameters["explore_type"]).lower() == "ssw"
        assert int(parameters["SSW.SSWsteps"]) > 0


def test_lasp_random_walk_alias_defaults_to_execute():
    sys.path.insert(0, str(PLUGIN))
    try:
        module = _load(ALIAS, "lasp_random_walk_example_test")
    finally:
        sys.path.remove(str(PLUGIN))
    seen = []
    original = module.lasp_ssw.main
    module.lasp_ssw.main = lambda argv: seen.extend(argv) or 0
    try:
        assert module.main(["--lasp-executable", "/fixture/lasp"]) == 0
    finally:
        module.lasp_ssw.main = original
    assert seen[0] == "execute"
