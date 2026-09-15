"""Release and supervision responsibilities for the MLIP active-learning Skill."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / ".agents" / "skills" / "mlip-active-learning"


def _texts() -> tuple[str, str]:
    return (
        (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8"),
        (
            SKILL_ROOT / "references" / "active-learning-contract.md"
        ).read_text(encoding="utf-8"),
    )


def test_skill_files_metadata_and_scope_are_bounded() -> None:
    skill, _ = _texts()
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
    assert "finite, offline" in skill
    assert "on-the-fly or open-ended dynamic loop" in skill
    assert not (SKILL_ROOT / "scripts").exists()
    assert not (SKILL_ROOT / "assets").exists()


def test_skill_routes_operations_and_reuses_round_artifacts() -> None:
    skill, _ = _texts()
    normalized = " ".join(skill.split())

    for operation in ("committee-evaluate", "select-candidates", "assess-round"):
        assert f"`{operation}`" in normalized
    assert "Inventory verified canonical labels" in normalized
    assert "Reuse matching accepted artifacts and start at the earliest missing stage" in (
        normalized
    )
    assert "Never edit the canonical dataset or replace prior rounds" in normalized
    assert "Retry creates a fresh attempt in the same round" in normalized


def test_skill_keeps_human_policy_and_convergence_boundaries() -> None:
    skill, reference = _texts()
    normalized = " ".join((skill + reference).split())

    assert "explicit target domain" in normalized
    assert "Do not derive thresholds from model brand" in normalized
    assert "coverage, independent accuracy, and consecutive stability" in normalized
    assert "Uncertainty-distribution change is diagnostic" in normalized
    assert "`BUDGET_EXHAUSTED` is not convergence" in normalized
    assert "`CONVERGED_FOR_DECLARED_DOMAIN` is bounded to the declared PES domain" in normalized
    assert "does not establish transport convergence" in normalized
    assert "ORACLE_REPLAY_VALIDATION" in normalized
    assert "FRESH_ACTIVE_LEARNING_ROUND" in normalized


def test_advanced_reference_is_conditional_and_execution_uses_effective_plan() -> None:
    skill, reference = _texts()
    normalized_skill = " ".join(skill.split())
    normalized_reference = " ".join(reference.split())

    assert "only when drafting or changing campaign policy" in normalized_skill
    assert "Ordinary operation routing and `assess-round` do not require it" in normalized_skill
    assert "Read this reference only when drafting or changing a campaign policy" in (
        normalized_reference
    )
    assert "effective `approval_required`" in normalized_skill
    assert "mlipflow --project PROJECT" in normalized_skill
    assert "final plugin `OK`" in normalized_skill


def test_all_skill_files_are_declared_for_wheel_packaging() -> None:
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
        "mlipflow/plugins/active_learning/skill/SKILL.md",
        "mlipflow/plugins/active_learning/skill/agents/openai.yaml",
        "mlipflow/plugins/active_learning/skill/references/active-learning-contract.md",
    }
