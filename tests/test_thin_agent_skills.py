"""Responsibility-boundary checks shared by the priority MLIPFlow Skills."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / ".agents" / "skills"

PRIORITY_SKILLS = {
    "mlip-workflow": (),
    "ionic-transport": ("analyze-existing", "md-smoke-and-analyze"),
    "dft-labeling": ("vasp-prepare", "label", "dataset-assemble"),
    "mlip-training": ("train", "finetune"),
    "mlip-benchmark": ("evaluate-fresh", "normalize-execute", "normalize-replay"),
    "pes-sampling": (
        "direct-select",
        "lasp-input-prepare",
        "merge-structures",
        "lasp-ssw-execute",
        "lasp-ssw-normalize-replay",
    ),
    "ase-md": ("run",),
    "lammps-md": ("lammps-prepare", "execute"),
    "mlip-active-learning": (
        "committee-evaluate",
        "select-candidates",
        "assess-round",
    ),
    "high-entropy-structure": ("generate-sqs",),
    "candidate-ranking": ("rank-candidates",),
}

CONDITIONAL_REFERENCES = {
    "mlip-workflow": "workflow-contract.md",
    "mlip-benchmark": "benchmark-contract.md",
    "mlip-active-learning": "active-learning-contract.md",
    "high-entropy-structure": "sqs-contract.md",
}

CLAIM_BOUNDARIES = {
    "mlip-workflow": "PES coverage does not establish transport convergence",
    "ionic-transport": "not an exact transition temperature or proof of a physical phase transition",
    "dft-labeling": "Preparation `OK` does not establish that VASP ran",
    "mlip-training": "not generalization, physical accuracy",
    "mlip-benchmark": "not fresh inference or independent scientific validation",
    "pes-sampling": "not SSW numerical parity, global PES coverage",
    "ase-md": "does not establish equilibration, diffusion, ionic conductivity",
    "lammps-md": "does not prove equilibration, diffusion, ionic conductivity",
    "mlip-active-learning": "`BUDGET_EXHAUSTED` is not convergence",
    "high-entropy-structure": "does not prove search convergence",
    "candidate-ranking": "not candidate generation, property calculation, model selection",
}


def _normalized_skill(name: str) -> str:
    text = (SKILL_ROOT / name / "SKILL.md").read_text(encoding="utf-8")
    return " ".join(text.split())


def test_priority_skills_route_operations_without_duplicating_adapter_contracts() -> None:
    hard_coded_approval = re.compile(
        r"approval_required\s*:\s*(?:true|false)|"
        r"(?:operation|execution)\s+(?:has|have)\s+approval_required"
    )

    for name, operations in PRIORITY_SKILLS.items():
        skill = _normalized_skill(name)
        for operation in operations:
            assert f"`{operation}`" in skill, (name, operation)
        assert "Reuse" in skill, name
        assert "effective `approval_required`" in skill, name
        assert "validate/plan/execute/check/collect" in skill, name
        assert "final plugin `ok`" in skill.lower(), name
        assert not hard_coded_approval.search(skill), name
        assert "plugin manifest" not in skill.lower(), name
        assert "source implementation" not in skill.lower(), name


def test_reference_reading_is_explicitly_conditional() -> None:
    mandatory_reference = re.compile(
        r"read \[references/[^\]]+\]\([^)]+\) (?:before|whenever|for every)",
        re.IGNORECASE,
    )

    for name, reference_name in CONDITIONAL_REFERENCES.items():
        skill = _normalized_skill(name)
        reference = " ".join(
            (
                SKILL_ROOT / name / "references" / reference_name
            ).read_text(encoding="utf-8").split()
        )
        assert f"references/{reference_name}" in skill
        assert "only when" in skill or "only for" in skill
        assert "not required" in skill or "do not require it" in skill
        assert "Read this reference only" in reference
        assert not mandatory_reference.search(skill)

    for name in set(PRIORITY_SKILLS) - set(CONDITIONAL_REFERENCES):
        skill = _normalized_skill(name)
        assert "references/" not in skill


def test_artifact_reuse_does_not_erase_scientific_claim_boundaries() -> None:
    for name, claim in CLAIM_BOUNDARIES.items():
        skill = _normalized_skill(name)
        assert claim in skill, name


def test_conditional_references_no_longer_copy_execution_contracts() -> None:
    forbidden_details = (
        "approval_required:",
        "shell: false",
        "schema_version:",
        "src/mlipflow/",
        "adapter.py",
        "run.sh",
        "output_subdir",
    )

    for name, reference_name in CONDITIONAL_REFERENCES.items():
        reference = (
            SKILL_ROOT / name / "references" / reference_name
        ).read_text(encoding="utf-8")
        for detail in forbidden_details:
            assert detail not in reference.lower(), (name, detail)
