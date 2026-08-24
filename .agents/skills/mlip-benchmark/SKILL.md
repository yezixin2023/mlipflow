---
name: mlip-benchmark
description: Supervise MLIPFlow energy, force, and stress benchmarks and model ranking. Use when running fresh inference from an explicit model plus labeled dataset, recomputing metrics from supplied reference/prediction pairs, replaying historical JSON/CSV/XLSX benchmark evidence, comparing exact MLIP families, or verifying benchmark IDs, paths, provenance, units, conventions, and scientific claim boundaries.
---

# MLIP benchmark

Use `plugins/mlip-benchmark` through MLIPFlow. The Skill selects the evidence mode and
guards comparability and claims; the Adapter and runner own inference, metrics,
normalization, ranking, and schemas.

## Route by scientific inputs

- Explicit model artifact plus labeled dataset: `evaluate-fresh`; a model is executed.
- Supplied reference/prediction pairs: `normalize-execute`; metrics are recomputed
  without loading a model.
- Historical JSON, CSV, XLSX, workbook, or reported metric evidence:
  `normalize-replay`; existing evidence is normalized with
  `model_execution=false`.

Resolve the mode from the inputs, not words such as “fresh” or “reproduce.” Use
`mlipflow inspect` for the current exact-family catalog rather than reading source code
or maintaining a model catalog in this Skill.

Read [references/benchmark-contract.md](references/benchmark-contract.md) only when
historical or legacy evidence has ambiguous provenance, units, conventions, or missing
source support. It is not required for an ordinary fresh or metric-only benchmark.

## Artifact first

Reuse verified prediction evidence or a completed normalized benchmark when its model,
dataset, task, scenario, split, targets, units, and conventions match. Do not rerun a
model merely to normalize already available prediction pairs. For a multi-model
comparison, use one shared labeled test set and normalize the collected evidence
together.

## Scientific judgment

Do not choose a model by brand or rank incomparable records. A between-model claim
requires the same task, scenario, split, metric, unit, direction, and relevant
scientific dimensions across the requested model set. Preserve unavailable or unknown
metrics and units explicitly; never guess them, replace them with numeric values, or
use a one-model metric to declare a winner.

Keep fresh inference, metric recomputation, replay, manuscript parity, and external
validation as separate claims.

## Execute and report

Use `mlipflow inspect` and the dry-run to review the exact model/evidence and dataset
bindings, targets, conventions, backend/resources, and expected artifacts. Follow the
plan's effective `approval_required` value rather than hard-coding approval by
operation or backend in this Skill.

Never invoke a model, historical script, benchmark runner, or scheduler outside
MLIPFlow, and never bypass Adapter `validate/plan/execute/check/collect`. Require final
plugin `OK`; scheduler `COMPLETED` or process exit zero is insufficient.

Report the selected mode, whether a model actually ran, exact family, dataset/scenario/
split, comparable metrics and direction, units/conventions, evidence paths, and
limitations. Replay is a structured collection of existing results, not fresh
inference or independent scientific validation.
