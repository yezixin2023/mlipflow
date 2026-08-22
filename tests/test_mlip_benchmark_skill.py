"""Release and intent contract for the MLIP benchmark supervisor Skill."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml

from mlipflow.science.model_runtime import (
    FRESH_MODEL_FAMILIES,
    MODEL_FAMILIES,
    MODEL_FAMILY_FRAMEWORKS,
)


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / ".agents" / "skills" / "mlip-benchmark"


def test_skill_files_metadata_and_catalog_source_are_bounded() -> None:
    skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    reference = (SKILL_ROOT / "references" / "benchmark-contract.md").read_text(
        encoding="utf-8"
    )
    metadata = yaml.safe_load(
        (SKILL_ROOT / "agents" / "openai.yaml").read_text(encoding="utf-8")
    )

    assert metadata["interface"]["display_name"] == "MLIP Benchmark"
    assert metadata["interface"]["default_prompt"].startswith("$mlip-benchmark ")
    assert "src/mlipflow/science/model_runtime.py" in skill
    assert "Do not copy either" in reference
    assert len(MODEL_FAMILIES) == 6
    assert len(FRESH_MODEL_FAMILIES) == 7
    assert sum(family.startswith("deepmd-") for family in MODEL_FAMILIES) == 4
    assert set(FRESH_MODEL_FAMILIES) == set(MODEL_FAMILY_FRAMEWORKS)
    assert not (SKILL_ROOT / "scripts").exists()


def test_natural_language_intents_preserve_execution_and_claim_boundaries() -> None:
    skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    reference = (SKILL_ROOT / "references" / "benchmark-contract.md").read_text(
        encoding="utf-8"
    )

    assert "model artifact plus a labeled" in skill
    assert "Choose `evaluate-fresh`" in skill
    assert "`model_execution=true`" in skill
    assert "Choose `normalize-execute` for supplied reference/prediction pairs" in skill
    assert "without loading a model" in skill
    assert "Choose `normalize-replay` for historical JSON, CSV, XLSX" in skill
    assert "`model_execution=false`" in skill
    assert "historical CHGNet stress unit" in reference
    assert "Report unknown" in reference
    assert "Which of the available exact models is best?" in reference
    assert "no brand preference" in reference
    assert "Use this old workbook for a fresh benchmark" in reference
    assert "Correct the mode to replay" in reference
    assert "Changed input paths or existing output" in reference
    assert "do not bypass or overwrite" in reference


def test_all_skill_files_are_declared_for_wheel_packaging() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = pyproject.split("[tool.setuptools.data-files]", 1)[1].split("\n[", 1)[0]
    prefix = "share/mlipflow/agent-skills/mlip-benchmark"
    declared: set[str] = set()
    for line in section.splitlines():
        match = re.fullmatch(r'"([^"]+)"\s*=\s*(\[.*\])', line.strip())
        if match is not None and (
            match.group(1) == prefix or match.group(1).startswith(prefix + "/")
        ):
            declared.update(ast.literal_eval(match.group(2)))
    expected = {
        ".agents/skills/mlip-benchmark/SKILL.md",
        ".agents/skills/mlip-benchmark/agents/openai.yaml",
        ".agents/skills/mlip-benchmark/references/benchmark-contract.md",
    }
    assert declared == expected
