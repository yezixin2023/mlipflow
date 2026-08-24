"""Release and orchestration responsibilities for the MLIP workflow Skill."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / ".agents" / "skills" / "mlip-workflow"


def _skill_text() -> tuple[str, str]:
    return (
        (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8"),
        (SKILL_ROOT / "references" / "workflow-contract.md").read_text(encoding="utf-8"),
    )


def test_skill_is_a_thin_control_plane_with_current_routes() -> None:
    skill, _ = _skill_text()
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
        "references/workflow-contract.md",
    }
    assert metadata["interface"]["display_name"] == "MLIP Workflow"
    assert metadata["interface"]["default_prompt"].startswith("$mlip-workflow ")
    for specialist in (
        "$high-entropy-structure",
        "$pes-sampling",
        "$dft-labeling",
        "$mlip-training",
        "$mlip-benchmark",
        "$ase-md",
        "$lammps-md",
        "$ionic-transport",
        "$candidate-ranking",
        "$mlip-active-learning",
    ):
        assert specialist in skill
    assert "`electrochemical-voltage` plugin" in skill
    assert ("$" + "electrochemical-voltage") not in skill
    assert not (SKILL_ROOT / "scripts").exists()
    assert not (SKILL_ROOT / "assets").exists()


def test_workflow_starts_at_the_earliest_missing_stage_and_continues_safely() -> None:
    skill, _ = _skill_text()
    normalized = " ".join(skill.split())

    assert "final scientific objective" in normalized
    assert "Reuse artifacts accepted by the producing Adapter" in normalized
    assert "earliest missing prerequisite" in normalized
    assert "Load the current specialist Skill for that stage" in normalized
    assert "Continue through deterministic downstream work" in normalized
    assert "Stop on a real scientific failure, unresolved information" in normalized
    assert "effective `approval_required`" in normalized
    assert "validate/plan/execute/check/collect" in normalized
    assert "without final plugin `OK`" in normalized


def test_workflow_reference_is_exceptional_not_a_normal_prerequisite() -> None:
    skill, reference = _skill_text()
    normalized_skill = " ".join(skill.split())
    normalized_reference = " ".join(reference.split())

    assert "only for manuscript or historical reproduction" in normalized_skill
    assert "not required for ordinary stage selection or artifact handoff" in normalized_skill
    assert "Read this reference only for manuscript or historical reproduction" in normalized_reference
    assert "Ordinary workflow routing should use the thin specialist Skill" in normalized_reference
    assert "structured collection of existing results" in normalized_reference
    assert "plugin manifest" not in normalized_skill.lower()
    assert "for every selected stage" not in normalized_skill.lower()


def test_all_skill_files_are_declared_for_wheel_packaging() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = pyproject.split("[tool.setuptools.data-files]", 1)[1].split("\n[", 1)[0]
    prefix = "share/mlipflow/agent-skills/mlip-workflow"
    declared: set[str] = set()
    for line in section.splitlines():
        match = re.fullmatch(r'"([^"]+)"\s*=\s*(\[.*\])', line.strip())
        if match is not None and (
            match.group(1) == prefix or match.group(1).startswith(prefix + "/")
        ):
            declared.update(ast.literal_eval(match.group(2)))
    assert declared == {
        ".agents/skills/mlip-workflow/SKILL.md",
        ".agents/skills/mlip-workflow/agents/openai.yaml",
        ".agents/skills/mlip-workflow/references/workflow-contract.md",
    }
