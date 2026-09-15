"""Release and supervision responsibilities for the MLIP benchmark Skill."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / ".agents" / "skills" / "mlip-benchmark"


def _texts() -> tuple[str, str]:
    return (
        (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8"),
        (SKILL_ROOT / "references" / "benchmark-contract.md").read_text(encoding="utf-8"),
    )


def test_skill_files_and_metadata_are_bounded() -> None:
    skill, _ = _texts()
    metadata = yaml.safe_load(
        (SKILL_ROOT / "agents" / "openai.yaml").read_text(encoding="utf-8")
    )

    assert metadata["interface"]["display_name"] == "MLIP Benchmark"
    assert metadata["interface"]["default_prompt"].startswith("$mlip-benchmark ")
    assert "current exact-family catalog" in skill
    assert "rather than reading source code" in skill
    assert not (SKILL_ROOT / "scripts").exists()


def test_skill_routes_by_evidence_and_preserves_claim_boundaries() -> None:
    skill, _ = _texts()
    normalized = " ".join(skill.split())

    assert "Explicit model artifact plus labeled dataset: `evaluate-fresh`" in normalized
    assert "Supplied reference/prediction pairs: `normalize-execute`" in normalized
    assert "Historical JSON, CSV, XLSX, workbook" in normalized
    assert "`normalize-replay`" in normalized
    assert "`model_execution=false`" in normalized
    assert "Do not choose a model by brand or rank incomparable records" in normalized
    assert "fresh inference, metric recomputation, replay, manuscript parity" in normalized
    assert "structured collection of existing results, not fresh inference" in normalized


def test_historical_reference_is_conditional_and_keeps_source_limitations() -> None:
    skill, reference = _texts()
    normalized_skill = " ".join(skill.split())
    normalized_reference = " ".join(reference.split())

    assert "only when historical or legacy evidence has ambiguous provenance" in normalized_skill
    assert "not required for an ordinary fresh or metric-only benchmark" in normalized_skill
    assert "Read this reference only for historical or legacy benchmark evidence" in (
        normalized_reference
    )
    assert "MISSING_SOURCE" in reference
    assert "Preserve an unknown value as unknown" in normalized_reference
    assert "Never synthesize missing historical evidence" in normalized_reference


def test_skill_uses_artifacts_effective_plan_and_adapter_lifecycle() -> None:
    skill, _ = _texts()
    normalized = " ".join(skill.split())

    assert "Reuse verified prediction evidence" in normalized
    assert "effective `approval_required`" in normalized
    assert "mlipflow --project PROJECT" in normalized
    assert "Require final plugin `OK`" in normalized
    assert "`check.diagnostics`" in normalized


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
    assert declared == {
        "mlipflow/plugins/mlip_benchmark/skill/SKILL.md",
        "mlipflow/plugins/mlip_benchmark/skill/agents/openai.yaml",
        "mlipflow/plugins/mlip_benchmark/skill/references/benchmark-contract.md",
    }
