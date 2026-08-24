# Historical benchmark evidence guidance

Read this reference only for historical or legacy benchmark evidence with ambiguous
provenance, units, conventions, or missing source support. Ordinary fresh inference and
metric-only normalization should rely on the Skill, inspected plan, and Adapter.

## Evidence mode and permitted claim

| Available scientific input | Mode | Permitted claim |
|---|---|---|
| Explicit model plus labeled dataset | Fresh inference | The model ran in this attempt and predictions were benchmarked. |
| Existing reference/prediction pairs | Metric recomputation | Metrics were recomputed from supplied pairs; model execution is not established. |
| Historical workbook, JSON/CSV, or reported metrics | Replay | Existing evidence was normalized or verified; no model ran. |

Words such as “fresh,” “reproduce,” or “new” do not change the evidence mode. An old
workbook remains replay input, and recent prediction pairs remain metric-only unless
the current attempt actually loads the model and labeled dataset.

## Units, conventions, and comparability

Use only units, energy normalization, stress ordering/sign, split identity, and sample
semantics supported by the evidence. Preserve an unknown value as unknown. Do not infer
it from framework reputation, common practice, filenames, table shape, or another
model's evidence.

A between-model conclusion needs comparable task, scenario, split, metric, unit,
direction, and scientific dimensions for the entire requested model set. A metric
available for only one model may be reported diagnostically, but it cannot select that
model over another with no comparable record.

Unavailable correlation or another mathematically unsupported metric must remain
unavailable. Never replace missing or undefined historical evidence with zero, NaN, or
an Agent estimate.

## Audited historical limitations

- Audited historical DeepMD sources and workbooks support their recorded energy
  normalization, but source-undeclared units remain unspecified.
- Historical CHGNet evidence supports its recorded per-atom energy convention. Force
  or stress units and stress ordering/sign remain unknown where the source cannot prove
  them.
- The audited M3GNet evaluation source has no corresponding historical benchmark
  output, prediction pairs, workbook, or evaluation log. Preserve `MISSING_SOURCE` for
  script-level parity.
- DPA-2 lacks raw prediction pairs or richer family-specific historical output at that
  evidence level. A manuscript summary replay does not fill the gap.

Never synthesize missing historical evidence or run an old source script simply to make
replay succeed.

## Reporting

State the evidence mode, whether a model actually ran, what was normalized or compared,
which units/conventions remain unknown, and which source limitations prevent stronger
claims. Keep `REPLAY_VERIFIED`, metric recomputation, fresh inference, historical
numerical parity, and external scientific validation distinct.
