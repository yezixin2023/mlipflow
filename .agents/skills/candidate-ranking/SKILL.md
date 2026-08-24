---
name: candidate-ranking
description: Supervise deterministic single-metric ranking and top-k selection for an existing candidate manifest and existing numeric metric results through the MLIPFlow candidate-ranking plugin. Use when planning, running, replaying, or verifying rank-candidates with an explicit metric, maximize/minimize direction, top-k, and missing-metric policy; do not use it to generate candidates, run MLIP/MD/DFT, calculate properties, or select models.
---

# Candidate ranking

Use `plugins/candidate-ranking` as the deterministic implementation. Call its
`rank-candidates` operation through MLIPFlow; do not implement sorting in this
Skill or rank candidates by inspection.

## Require the complete contract

Require all of the following without guessing:

- an existing schema-v1 candidate manifest with unique non-empty candidate IDs;
- an existing schema-v1 metric-results manifest whose records reference those IDs
  and contain finite values under `metrics`;
- the exact metric key and its documented unit/provenance;
- `direction: maximize` or `direction: minimize`;
- a positive integer `top_k`;
- `missing_metric_policy: reject` or `missing_metric_policy: error`.

`reject` deterministically excludes candidates with no metric record. `error`
requires complete coverage. A present record with a missing or non-finite selected
metric is invalid under either policy; never impute a value.

## Supervise the plugin

Use only read-only MLIPFlow commands while establishing the node and input
contract. Inspect the `candidate-ranking` node and confirm operation
`rank-candidates`, backend `local`, both manifest paths, the metric rule, and the
result path. `rank-candidates` has `approval_required: false`: inspect the dry-run,
then run it without an approval stop. Never invoke `rank.py` as a substitute for the
plugin lifecycle, and never guess a missing metric, direction, top-k, or missing-value
policy.

After execution or replay, require plugin `status: OK`. Report the requested rule,
input/evaluated/missing counts, and returned top-k values. The deterministic tie
break is candidate ID ascending. Do not reinterpret a ranking as candidate
generation, a property calculation, model selection, predictive validation, or
high-fidelity validation.
