"""Release and supervision contract for the MLIP active-learning Skill."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / ".agents" / "skills" / "mlip-active-learning"


def test_skill_files_metadata_and_scope_are_bounded():
    skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    reference = (SKILL_ROOT / "references" / "active-learning-contract.md").read_text(
        encoding="utf-8"
    )
    metadata = yaml.safe_load(
        (SKILL_ROOT / "agents" / "openai.yaml").read_text(encoding="utf-8")
    )
    actual_files = {
        path.relative_to(SKILL_ROOT).as_posix()
        for path in SKILL_ROOT.rglob("*")
        if path.is_file()
    }
    assert actual_files == {
        "SKILL.md",
        "agents/openai.yaml",
        "references/active-learning-contract.md",
    }
    assert metadata["interface"]["display_name"] == "MLIP Active Learning"
    assert metadata["interface"]["default_prompt"].startswith("$mlip-active-learning ")
    assert "on-the-fly" in skill
    assert "dynamic DAG" in skill + reference
    assert "Do not invent" in skill
    assert "Never modify the canonical dataset directly" in skill
    assert not (SKILL_ROOT / "scripts").exists()
    assert not (SKILL_ROOT / "assets").exists()


def test_skill_preserves_strategy_math_decision_and_approval_boundaries():
    text = "\n".join(
        [
            (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8"),
            (SKILL_ROOT / "references" / "active-learning-contract.md").read_text(
                encoding="utf-8"
            ),
        ]
    )
    for fragment in (
        "single-model-committee",
        "dual-model-risk-union",
        "combined_risk = max",
        "Cross-framework prediction difference",
        "committee-evaluate",
        "select-candidates",
        "assess-round",
        "SAFE spot-check",
        "ORACLE_REPLAY_VALIDATION",
        "FRESH_ACTIVE_LEARNING_ROUND",
        "CONVERGED_FOR_DECLARED_DOMAIN",
        "BUDGET_EXHAUSTED",
        "BLOCKED_CALIBRATION",
        "BLOCKED_SAMPLING",
        "SCIENTIFIC_REVIEW_REQUIRED",
        "dry-run",
        "boolean approval",
        "scheduler `COMPLETED`",
        "canonical-label path and split IDs",
    ):
        assert fragment in text
    assert "`BUDGET_EXHAUSTED` is never convergence" in text
    assert "PES_ACTIVE_LEARNING_CONVERGENCE" in text
    assert "TRANSPORT_CONVERGENCE" in text


def test_all_skill_files_are_declared_for_wheel_packaging():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = pyproject.split("[tool.setuptools.data-files]", 1)[1].split("\n[", 1)[0]
    prefix = "share/mlipflow/agent-skills/mlip-active-learning"
    declared: set[str] = set()
    for line in section.splitlines():
        match = re.fullmatch(r'"([^"]+)"\s*=\s*(\[.*\])', line.strip())
        if match is not None and (
            match.group(1) == prefix or match.group(1).startswith(prefix + "/")
        ):
            declared.update(ast.literal_eval(match.group(2)))
    assert declared == {
        ".agents/skills/mlip-active-learning/SKILL.md",
        ".agents/skills/mlip-active-learning/agents/openai.yaml",
        ".agents/skills/mlip-active-learning/references/active-learning-contract.md",
    }
