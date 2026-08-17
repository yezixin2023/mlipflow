---
name: mlip-benchmark
description: Supervise MLIPFlow energy, force, and stress benchmarks and model ranking. Use when running fresh inference from an explicit model plus labeled dataset, recomputing metrics from supplied reference/prediction pairs, replaying historical JSON/CSV/XLSX benchmark evidence, comparing exact MLIP families, or verifying benchmark identities, provenance, units, conventions, and scientific claim boundaries.
---

# MLIP benchmark

Use `plugins/mlip-benchmark` as the deterministic execution layer. Do not
reimplement inference, metrics, ranking, parsing, or fingerprinting in this
Skill. Read [references/benchmark-contract.md](references/benchmark-contract.md)
before planning or interpreting a benchmark.

## Select exactly one mode

- Choose `evaluate-fresh` only for an explicit model artifact plus a labeled
  dataset. It must load the model, generate predictions, and produce provenance
  with `model_execution=true`.
- Choose `normalize-execute` for supplied reference/prediction pairs. Recompute
  MAE, RMSE, Pearson r, and ranking without loading a model; describe this as
  metric-only evaluation.
- Choose `normalize-replay` for historical JSON, CSV, XLSX, workbook, or metric
  evidence. Describe it as read-only evidence normalization/verification with
  `model_execution=false`, never as fresh inference or fresh numerical parity.

An old workbook is replay input even if the user calls it a “fresh benchmark.”
Prediction pairs are metric-only input even if they came from a recent model
run. Resolve the mode from the supplied scientific inputs, not the adjective.

## Establish the contract

Use read-only MLIPFlow commands to inspect persistent state and the node before
execution. Confirm the exact operation, local backend, inputs, task, scenario,
split, target set, units, conventions, and expected fresh output directory.

Read the supported exact model families and framework mapping from
`src/mlipflow/science/model_runtime.py` or the installed MLIPFlow model catalog;
cross-check the `mlip-benchmark` manifest. Never maintain a model list in this
Skill, accept an under-specified DeepMD family, or choose a model by brand.

Require the inputs appropriate to the selected mode:

- Fresh: model, canonical labeled dataset, exact family, model and dataset
  fingerprints, task/scenario/split, targets, target units, total/per-atom
  energy convention, and an explicit stress convention when stress is selected.
- Metric-only: reference/prediction evidence plus explicit or evidence-backed
  model/task/scenario/split/unit identities.
- Replay: historical evidence, portable evidence locator, exact identities, and
  optional read-only historical source reference for provenance.

Do not guess units, total/per-atom normalization, stress ordering/sign, split,
or scalar-count semantics. Ask for the missing value when execution requires it;
otherwise preserve the contract's unknown or `source-unit-unspecified` value.

## Execute through MLIPFlow

Never call the bundled runner, wrapper, framework, or historical script as a
substitute for the Adapter lifecycle. Use MLIPFlow's ordinary plan, dry-run and
approval flow. Confirm `shell=false`, argv-based execution, pinned input
identities, and a fresh empty output directory. Do not overwrite or delete an
existing result; retry must create a new attempt.

After execution, require Adapter `check` and `collect` to return `OK`. Treat
partial output, model/dataset/evidence drift, unsupported identity, non-finite
metrics, or provenance mismatch as `FAIL`. Scheduler or process completion alone
is not scientific success.

## Report the result

Lead with success/failure, selected mode, and whether a model actually ran. Then
report the exact family, dataset/scenario/split, comparable MAE/RMSE/Pearson
values, metric direction and ranking, units/conventions, and the important
model/dataset/evidence/output fingerprints. State missing or unknown scientific
information explicitly. Keep the default answer concise and provide full JSON
only when requested.

Never claim historical replay as fresh inference, metric recomputation as model
execution, contract coverage as production numerical validation, or parity
without an appropriate comparison against real reference evidence.
