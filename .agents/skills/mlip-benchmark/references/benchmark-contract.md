# MLIP benchmark contract

Use this reference to select, validate, and explain the existing
`mlip-benchmark` capability. The plugin and MLIPFlow core remain authoritative.

## Mode semantics

| User input and intent | Operation | Model executed | Permitted claim |
|---|---|---:|---|
| Explicit model + canonical labeled dataset | `evaluate-fresh` | yes | Fresh inference and metrics from newly generated predictions |
| Existing reference/prediction pairs | `normalize-execute` | no | Metric recomputation from supplied pairs |
| Historical JSON/CSV/XLSX/workbook or reported metrics | `normalize-replay` | no | Historical evidence normalization and verification |

Do not infer the mode from words such as “new,” “fresh,” or “reproduce.” Infer
it from the inputs. A workbook cannot become fresh inference without a model and
labeled dataset; supplied prediction pairs never prove that this attempt loaded
the model.

Legacy `evaluate-static` and `collect-existing` remain compatible operations,
but prefer the three explicit modes above for new benchmark supervision.

Operation-level approval follows execution semantics: `evaluate-fresh` requires
approval, while local `normalize-execute`, `normalize-replay`, and compatibility
metric-only paths do not. Any SSH-SLURM execution requires approval. This boundary does
not supply missing units, conventions, splits, models, or datasets; Adapter validation
still blocks those scientific gaps.

## Catalog and model selection

Read the current exact-family catalog, aliases, and framework mapping from
`src/mlipflow/science/model_runtime.py` (or its installed package equivalent)
and confirm the plugin manifest. The fresh executable catalog contains seven
exact families: four DeepMD family IDs that share one DeepMD inference
implementation, M3GNet/MatGL, CHGNet, and MACE. The separate six-family
historical replay catalog deliberately remains stable so existing manuscript
artifacts do not change when fresh runtimes are added. Do not copy either
catalog into the Skill, accept an ambiguous family, or hard-code a preferred
model.

Only rank records that are comparable in task, scenario, split, metric, unit,
direction, and scientific dimensions. Lower is better for MAE and RMSE; higher
is better for Pearson r. Equal values share a rank, with model ID used only as a
deterministic display tie-breaker. Ranking policy, not brand, determines order.

## Fresh input and evidence contract

Fresh evaluation requires:

- an explicit model file or directory path;
- a canonical schema-v1 labeled JSON dataset path;
- an exact model family;
- normalized task, scenario, and split identifiers;
- a non-empty energy/force/stress target subset;
- an explicit unit for every selected target;
- `energy_normalization: total` or `per-atom`;
- an explicit stress convention when stress is selected;
- a fresh, empty attempt-relative output directory.

Local fresh evaluation uses direct model/dataset paths. Reviewed SSH-SLURM fresh
evaluation instead uses small logical model and benchmark-dataset reference manifests;
the remote runner resolves them below the selected site's canonical roots and emits a
cluster report recording the exact family, software version, parameters, and fetched
output paths. Site environments and root paths remain template-owned.

The versioned `prediction_evidence.json` records structure/sample ID,
structure index, atom count, model family/framework/path, dataset path,
task/scenario/split, references, predictions, units, component labels, scalar counts,
runtime/framework/version, source paths, and energy/stress conventions. Fresh evidence and provenance must both state
`mode: fresh` and `model_execution: true`.

Energy contributes one scalar per structure. Force contributes three components
per atom. The fresh canonical stress contract uses the explicitly approved
six-component order; do not convert a historical nine-component table to this
contract without reliable source evidence. Confirm component and structure
counts rather than inferring them from file size or row count.

When force evidence preserves explicit N×3 vectors, the normalized suite also emits
`maximum_atomic_force_error`: the maximum over atoms of the L2 norm of each predicted
minus reference force vector. Its sample count is the atom count, not the flattened
component count. If any force row is flattened or otherwise loses vector grouping,
omit this metric; componentwise MAE/RMSE remain valid, but the maximum atomic-vector
error is unavailable and must not be inferred.

MAE and RMSE are always required for each selected target. Pearson correlation is
mathematically unavailable with fewer than two scalar pairs or when either series is
constant. Preserve that as explicit unavailable-metric provenance and omit it from
numeric ranking; never substitute zero, NaN, or a fabricated coefficient. This permits
a tiny held-out set to retain valid error metrics without overstating correlation
evidence.

## Metric-only and replay inputs

`normalize-execute` consumes actual reference/prediction values, groups them by
their scientific IDs/units, and computes one implementation of MAE, RMSE, and
Pearson r. It also computes the maximum atomic L2 force error when explicit N×3
force vectors are available. Its provenance states `model_execution: false`. It
does not establish how or when predictions were generated.

One joint comparison may consume prediction evidence from multiple fresh model nodes.
Each input must have a distinct portable locator even when all source basenames are
`prediction_evidence.json`. The normalized ranking is valid only for exact shared
task/scenario/split/metric/unit/direction/dimensions; it is the primary static test-set
comparison, not a universal model-quality claim. For a requested model set, only groups
whose `comparable_model_count` covers that entire set may inform the winner. A
single-model stress group remains useful diagnostic evidence but is excluded from a
two-model decision when the other model has no comparable stress prediction.

`normalize-replay` parses existing prepared JSON/CSV/XLSX evidence and preserves
source locator, parser name, units or unit uncertainty, split, sample count, and
the source-script path when supplied. It does not execute
the historical script. Never import or run historical source merely to make a
replay succeed.

## Normalized artifacts and provenance

All three modes emit:

- `metrics.json`
- `benchmark_summary.csv`
- `model_ranking.json`
- `provenance.json`

Fresh additionally emits `prediction_evidence.json`.

Scheduled fresh evaluation additionally emits `cluster-benchmark-report.json`; it must
match the approved model/dataset paths, exact framework/family, return code, and all
five fetched artifact paths.

Inspect normalized records for finite values, positive sample counts, explicit
direction, exact model/task/scenario/split IDs, units, dimensions, and
evidence records. Inspect provenance for mode, `model_execution`, network-access
claim, source paths, output paths, and read-only replay markers. Fresh provenance also
requires exact family, model and dataset paths, runtime/framework/version,
conventions, structure count, and scalar sample counts.

Adapter `check/collect` must reject model, dataset, evidence, implementation
source, mode, task/scenario/split, unit, summary, ranking, or declared-output
mismatches. Never bypass that rejection, silently overwrite output, or edit artifacts
to make them pass.

## Unit and convention policy

Do not derive units from framework reputation or common practice. Use only
explicit dataset/evidence fields or reliable source definitions. If execution
requires a missing unit or convention, ask the user. If replay can preserve the
unknown, retain `source-unit-unspecified` or the corresponding unknown marker.

Apply the same rule to total versus per-atom energy, stress ordering and sign,
dataset split, and sample-count semantics. Distinguish structure count from
energy scalar count, force component count, and stress component count.

## Historical evidence boundary

- The three audited historical DeepMD families have source and workbook evidence.
  Source confirms per-atom energy normalization, but
  units not declared by the source remain unspecified.
- CHGNet has historical source/workbook evidence. Historical
  energy is per-atom; force/stress units and stress ordering/sign are not all
  reliably established, so report them as unknown where applicable.
- The audited M3GNet evaluation source currently has no corresponding historical
  benchmark output, workbook, prediction pairs, or evaluation log. Preserve
  `MISSING_SOURCE` for historical script-level parity.
- DPA-2 currently lacks raw prediction pairs or family-specific richer
  historical benchmark output. Preserve `MISSING_SOURCE` for that evidence
  level; a manuscript summary replay does not fill the gap.

Never synthesize missing historical evidence. `REPLAY_VERIFIED`, metric
recomputation, fresh inference, historical numerical parity, and external
scientific validation are separate claims.

## Intent checks

Use these checks when mode selection is ambiguous:

| Request | Required behavior |
|---|---|
| “Benchmark this labeled dataset with this CHGNet model.” | Fresh; require model/dataset paths and execute the model |
| “I have reference/prediction JSON; calculate RMSE and Pearson, then rank.” | Metric-only; do not run a model |
| “Reproduce this DeepMD historical workbook.” | Replay; require `model_execution=false` |
| “What is the historical CHGNet stress unit?” | Report unknown when evidence cannot prove it; do not guess |
| “Which of the available exact models is best?” | Require comparable metric records and direction; no brand preference |
| “Use this old workbook for a fresh benchmark.” | Correct the mode to replay unless a model and labeled dataset are supplied |
| Changed input paths or existing output | Stop on Adapter validation failure; do not bypass or overwrite |
