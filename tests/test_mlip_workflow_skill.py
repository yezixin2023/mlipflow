"""Release and natural-language routing contract for the MLIP workflow Skill."""

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


def test_skill_is_control_plane_only_and_lists_current_capabilities() -> None:
    skill, reference = _skill_text()
    metadata = yaml.safe_load(
        (SKILL_ROOT / "agents" / "openai.yaml").read_text(encoding="utf-8")
    )
    expected_files = {
        "SKILL.md",
        "agents/openai.yaml",
        "references/workflow-contract.md",
    }
    actual_files = {
        path.relative_to(SKILL_ROOT).as_posix()
        for path in SKILL_ROOT.rglob("*")
        if path.is_file()
    }

    assert actual_files == expected_files
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
        assert specialist in skill + reference
    assert "`electrochemical-voltage` plugin" in skill + reference
    assert ("$" + "electrochemical-voltage") not in skill + reference
    assert not (SKILL_ROOT / "scripts").exists()
    assert not (SKILL_ROOT / "assets").exists()


def test_all_nineteen_natural_language_routes_are_explicit_and_safe() -> None:
    _, reference = _skill_text()
    rows = {}
    for line in reference.splitlines():
        match = re.match(r"\|\s*(\d+)\s*\|", line)
        if match:
            rows[int(match.group(1))] = line
    expected_fragments = {
        1: ("$high-entropy-structure", "do not jump to training"),
        2: ("$dft-labeling", "do not regenerate SQS"),
        3: ("$mlip-training", "explicit framework"),
        4: ("$mlip-benchmark", "labeled test set/evidence"),
        5: ("$ase-md", "$lammps-md", "skip unrelated upstream"),
        6: ("$ionic-transport", "do not rerun MD"),
        7: ("$candidate-ranking", "top_k=10", "missing policy"),
        8: ("prefer replay", "do not default to fresh expensive reruns"),
        9: ("check/collect", "do not claim `OK` yet"),
        10: ("Refuse brand preference", "comparable benchmark/routing evidence"),
        11: ("dry-run", "approval boundary"),
        12: ("Reuse verified labels", "explicit scoped intent"),
        13: ("Do not invent a native pair style", "$ase-md"),
        14: ("electrochemical-voltage", "do not seek a voltage Skill"),
        15: ("$high-entropy-structure", "replay/check", "explicit intent"),
        16: ("Stop downstream", "final `OK`"),
        17: ("dataset-assemble", "no custom converter script"),
        18: ("$ionic-transport", "auto-read md-result/index", "do not request files"),
        19: ("$ionic-transport", "old/new coordinate columns", "do not rerun MD"),
    }

    assert set(rows) == set(range(1, 20))
    for scenario, fragments in expected_fragments.items():
        for fragment in fragments:
            assert fragment in rows[scenario], (scenario, fragment)


def test_workflow_uses_effective_operation_and_batch_approval_boundaries() -> None:
    skill, reference = _skill_text()
    text = skill + reference
    for fragment in (
        "effective `approval_required`",
        "Every SSH-SLURM execution requires approval",
        "One explicitly reviewed batch",
        "no persistent batch approval",
        "continue eligible local deterministic",
        "Approval is not scientific validation",
    ):
        assert fragment in text


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
