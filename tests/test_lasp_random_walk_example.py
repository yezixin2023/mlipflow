from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "plugins" / "pes-sampling" / "lasp_ssw.py"
EXAMPLE = ROOT / "examples" / "lasp_random_walk" / "lasp.in"


def _load_wrapper():
    spec = importlib.util.spec_from_file_location("lasp_ssw_example_test", WRAPPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lasp_random_walk_example_is_valid_ssw_contract():
    module = _load_wrapper()
    parameters = module._parse_lasp_input(EXAMPLE)
    module._validate_ssw_input(parameters)
    assert str(parameters["explore_type"]).lower() == "ssw"
    assert int(parameters["SSW.SSWsteps"]) > 0
