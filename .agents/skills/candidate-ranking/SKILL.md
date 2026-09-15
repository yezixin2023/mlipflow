---
name: candidate-ranking
description: Supervise deterministic single-metric ranking and top-k selection for an existing candidate manifest and existing numeric metric results through the MLIPFlow candidate-ranking plugin. Use when planning, running, replaying, or verifying rank-candidates with an explicit metric, maximize/minimize direction, top-k, and missing-metric policy; do not use it to generate candidates, run MLIP/MD/DFT, calculate properties, or select models.
---

# Candidate ranking

Use `mlipflow/plugins/candidate_ranking` through MLIPFlow for `rank-candidates`. This Skill
selects the ranking intent; the plugin owns validation, sorting, tie handling, checking,
and collection.

## Artifact first

Require existing candidate and numeric metric evidence. Reuse a verified matching
ranking result instead of recalculating it. Do not generate candidates, compute missing
properties, rerun MLIP/MD/DFT, or impute values in order to make a ranking possible.

## Scientific judgment

Require an explicit metric, its unit and provenance, maximize/minimize direction,
top-k, and missing-metric policy. Do not guess any of these or choose them from
candidate names, composition, or file order. Ranking is meaningful only for comparable
numeric evidence under the declared policy.

## Execute and interpret

Use `mlipflow inspect` and the dry-run to review the actual manifests and ranking rule.
Follow the effective `approval_required` value rather than duplicating it in this Skill.
Never call the ranking script directly, sort candidates in the Agent, or bypass Adapter
`validate/plan/execute/check/collect`. Require final plugin `OK`.

Report the rule, evidence coverage, exclusions, and returned top-k values. The result is
a deterministic ordering under one declared metric and policy; it is not candidate
generation, property calculation, model selection, predictive validation, or
high-fidelity scientific validation. Replay is a structured collection of existing
results, not a new property calculation.
