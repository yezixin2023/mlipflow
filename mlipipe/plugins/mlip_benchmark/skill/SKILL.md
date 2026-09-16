---
name: mlip-benchmark
description: Supervise MLIPipe energy, force, and stress benchmarks and model ranking. Use when running fresh inference from an explicit model plus labeled dataset, recomputing metrics from supplied reference/prediction pairs, replaying historical JSON/CSV/XLSX benchmark evidence, comparing exact MLIP families, or verifying benchmark IDs, paths, provenance, units, conventions, and scientific claim boundaries.
---

# MLIP benchmark

Use `mlipipe/plugins/mlip_benchmark` through MLIPipe. The Skill selects the evidence mode and
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
`mlipipe inspect` for the current exact-family catalog rather than reading source code
or maintaining a model catalog in this Skill.

Read [references/benchmark-contract.md](references/benchmark-contract.md) only when
historical or legacy evidence has ambiguous provenance, units, conventions, or missing
source support. It is not required for an ordinary fresh or metric-only benchmark.

## Inputs

Reuse verified prediction evidence for the same dataset/scenario/split and targets.
Fresh evaluation requires an explicit model reference and labeled test dataset;
normalization requires the existing evidence manifest, declared units and metric policy.
The dry-run
reports the resolved evidence mode and expected metrics/model-ranking artifacts.

## Scientific judgment

Do not choose a model by brand or rank incomparable records. A between-model claim
requires the same task, scenario, split, metric, unit, direction, and relevant
scientific dimensions across the requested model set. Preserve unavailable or unknown
metrics and units explicitly; never guess them, replace them with numeric values, or
use a one-model metric to declare a winner.

Keep fresh inference, metric recomputation, replay, manuscript parity, and external
validation as separate claims.

## Run and read the result

Preview the actual task and effective `approval_required` value. Reuse explicit
user authorization for this task and scale; use `--approve` when the plan requires it.

```bash
mlipipe --project PROJECT init
mlipipe --project PROJECT --format json run NODE --dry-run
mlipipe --project PROJECT --format json run NODE --approve
```

For a plan without an approval requirement, `run NODE` suffices. For scheduled
execution, use `mlipipe --project PROJECT --format json advance` when the job
progresses; `mlipipe --project PROJECT json NODE` reads the saved result.

Read `state`, `metrics`, `artifacts[].role` and the full `artifacts[].path` from JSON.
Require final plugin `OK` before using outputs. On failure start with `reason`,
`check.diagnostics`, `manifest_path` and `logs`; `mlipipe --project PROJECT logs NODE`
shows saved stdout/stderr. Example paths refer to the
[repository examples](https://github.com/yezixin2023/mlipipe/tree/public/examples).

Report the selected mode, whether a model actually ran, exact family, dataset/scenario/
split, comparable metrics and direction, units/conventions, evidence paths, and
limitations. Replay is a structured collection of existing results, not fresh
inference or independent scientific validation.
