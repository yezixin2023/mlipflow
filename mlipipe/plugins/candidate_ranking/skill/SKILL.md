---
name: candidate-ranking
description: Supervise deterministic single-metric ranking and top-k selection for an existing candidate manifest and existing numeric metric results through the MLIPipe candidate-ranking plugin. Use when planning, running, replaying, or verifying rank-candidates with an explicit metric, maximize/minimize direction, top-k, and missing-metric policy; do not use it to generate candidates, run MLIP/MD/DFT, calculate properties, or select models.
---

# Candidate ranking

Use `mlipipe/plugins/candidate_ranking` through MLIPipe for `rank-candidates`. This Skill
selects the ranking intent; the plugin owns validation, sorting, tie handling, checking,
and collection.

## Inputs and example

Supply `inputs.candidate_manifest` and `inputs.metric_results_manifest`; reuse
matching existing results. Set `metric`, `direction`, `top_k` and
`missing_metric_policy` from the requested rule. The complete, offline example is
`examples/local_ranking/project.yaml`; follow its `USAGE.md` with no model download.

## Scientific judgment

Require an explicit metric, its unit and provenance, maximize/minimize direction,
top-k, and missing-metric policy. Do not guess any of these or choose them from
candidate names, composition, or file order. Ranking is meaningful only for comparable
numeric evidence under the declared policy.

## Run and read the result

For the existing local inputs, the shortest execution is:

```bash
mlipipe --project PROJECT init
mlipipe --project PROJECT --format json run NODE
```

Use `run NODE --dry-run` when reviewing changed inputs or execution scope; it reports
the effective `approval_required` value. `mlipipe --project PROJECT json NODE`
reads the saved result later.

Read `state`, `metrics`, `artifacts[].role` and the full `artifacts[].path` from JSON.
Require final plugin `OK` before using outputs. On failure start with `reason`,
`check.diagnostics`, `manifest_path` and `logs`; `mlipipe --project PROJECT logs NODE`
shows saved stdout/stderr. Example paths refer to the
[repository examples](https://github.com/yezixin2023/mlipipe/tree/public/examples).

Report the rule, evidence coverage, exclusions, and returned top-k values. The result is
a deterministic ordering under one declared metric and policy; it is not candidate
generation, property calculation, model selection, predictive validation, or
high-fidelity scientific validation. Replay is a structured collection of existing
results, not a new property calculation.
