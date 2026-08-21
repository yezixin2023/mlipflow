# Offline active-learning supervision contract

Use this reference for policy review, artifact routing, round execution, and decision reporting.
Current plugin manifests and the selected specialist Skills remain authoritative for their stages.

## Campaign and round layout

Keep one small campaign record and one immutable MLIPFlow project per round:

```text
campaign/
  campaign.yaml
  round-000/project.yaml
  round-001/project.yaml
```

`campaign.yaml` records only the campaign ID, strategy, target domain, policy, sequential
`round-NNN` entries, cumulative canonical dataset reference, and current decision. It is not a model
registry or workflow engine. A round is not a retry, and a retry never edits or replaces a prior
attempt. The immutable assessment input snapshot uses `current_decision: PENDING` and binds the
current cumulative dataset path and split IDs; only after the plugin emits its decision may the Agent
update the campaign-level record outside that round.

The fixed round DAG may connect these verified stages:

```text
committee training -> exploration MD -> committee evaluation
-> physical/temporal/dedup preparation -> DIRECT -> candidate selection
-> DFT labeling -> canonical dataset assembly -> retraining
-> immutable audit benchmark -> round assessment
```

Skip an upstream numerical stage only when a checker/replay contract accepts matching existing
evidence. Do not repeat expensive work ceremonially.

## Required policy evidence

The policy must declare, without Skill defaults:

- Strategy A with one `primary_model`, or Strategy B with exactly two models;
- each model's same-framework committee seeds (first version varies seeds only);
- the canonical dataset ID, shared split ID, and disjoint train/validation/calibration/audit IDs;
- requested and minimum observed calibration coverage, epsilon, minimum calibration sample count,
  and maximum calibration false-negative rate;
- force unit, `label_threshold`, and larger `abort_threshold`;
- every target condition and condition-level maximum QUERY and UNSAFE fractions;
- temporal stride, per-condition/per-replica quotas, SAFE spot-check count and seed, per-round DFT
  maximum, and total DFT maximum;
- per-model immutable audit limits for energy MAE/RMSE, force MAE/RMSE, and maximum atomic force
  error; add stress only when all compared models share a reliable convention;
- marginal-gain metric and stopping threshold plus required consecutive passing rounds;
- composition, structure families, defect states, temperature and pressure ranges, ensembles,
  supercell range, LASP/SSW high-energy inclusion, phase-change/melting/decomposition permissions,
  mobile species, and intended static/MD/NPT/transport use.

Do not derive a missing threshold from model branding, a plot, training loss, or an empirical
constant from another chemistry.

Fresh scheduled inference requires a model index with one shared training contract per
same-framework committee. The contract records the cumulative canonical label path, exact
framework dataset export, predefined split ID plus train/validation/test record IDs,
foundation-model and training-config paths, precision, and train/finetune operation. Every
member is checked against its collected training result and cluster report before the index is
written, without copying those reports. Strategy B may use
framework-specific dataset exports and configs, but its two committees must have identical
canonical-label path and split IDs.

## Strategy contracts

Strategy A evaluates only the selected model's committee. Its labels remain canonical and may later
train another framework, but they do not affect the current acquisition rule.

Strategy B independently calibrates each same-framework committee over the same calibration IDs.
For each candidate the plugin uses:

```text
combined_risk = max(calibrated_risk_model_a, calibrated_risk_model_b)
```

Any model's QUERY/UNSAFE class prevents a final SAFE class. Cross-framework prediction difference is
diagnostic only and must never replace either committee's calibration. Fair model comparisons use
only metrics supported by both; model-only stress remains diagnostic.

## Mathematical and selection evidence

For atom `i` and `M` committee members, the plugin defines force disagreement as:

```text
sqrt((1/M) * sum_m ||F_i^m - mean_m(F_i^m)||_2^2)
```

The structure quantity is its maximum over atoms; the result also records every per-atom value and
their mean. Energy disagreement is the population standard deviation of total energy divided by atom
count.

Calibration uses the conservative nearest-rank quantile requested by policy:

```text
scale_i = actual_committee_mean_DFT_force_error_i / (u_force_i + epsilon)
estimated_force_error = quantile(scale_i) * u_force
```

The calibration report must expose sample count, requested/observed coverage, scale, correlation,
false negatives, and out-of-calibration-range counts for each model. Insufficient or failed
calibration is a scientific block, not a successful low-risk result.

Candidate selection preserves this order: physical validity, calibrated class, temporal thinning,
exact duplicate removal, near-duplicate removal, condition grouping, existing `pes-sampling`
DIRECT evidence, trigger-model coverage, quotas, and DFT budget. For Strategy B, the deterministic
coverage pass first represents every trigger model available in the DIRECT subset, then fills the
remaining budget by combined risk. If budget or quotas leave an available trigger model
unrepresented, the selection reports that fact and round assessment cannot converge. Severe bad
structures never go directly to DFT. SAFE
spot-checks are a separate deterministic, condition-stratified DFT sample for detecting shared
committee blind spots. QUERY labels and SAFE spot checks together must fit the declared per-round
DFT-label maximum. Any SAFE false negative must make the round-level
`calibration_status` explicitly fail even when the original fixed calibration subset passed.
Each offline UNSAFE candidate records the nearest earlier SAFE frame in the same declared condition
and replica when one exists; a missing earlier boundary remains explicit `null`.

## Operation handoffs

Local `committee-evaluate` consumes policy plus a prediction manifest containing model/framework
records, member IDs and seeds, common calibration/candidate IDs, per-member E/F predictions,
candidate condition/replica/frame metadata, structure ID, near-duplicate group, and upstream
physical-validity evidence. Its output is usable for selection only when
`evaluation_status: READY`.

Reviewed SSH-SLURM `committee-evaluate` instead consumes a policy, a bounded evaluation dataset, and
an immutable committee model index. Every member record records its exact family, seed, logical
artifact ID, canonical model-root-relative path, and kind. The site-owned
`active-learning-committee-canonical` template supplies the framework environment and canonical
model root; the project must not contain a host, partition, module, environment path, executable, or
absolute model path. Collect and verify `committee-predictions.json`,
`committee-evaluation.json`, and `cluster-active-learning-report.json`. The cluster report records
the resolved model and fetched output paths, after which the local
checker recomputes the evaluation from the fetched predictions. This fresh path does not publish,
replace, or edit models and never reads a historical model directory as a canonical root.

`select-candidates` consumes that evaluation, the matching structure manifest, policy, and DIRECT
JSON or collected `manifest.csv` when QUERY candidates remain. It returns separate QUERY labels,
SAFE spot checks, severe-invalid exclusions, selection counts, and exact structures for DFT.

`assess-round` consumes the evaluation and selection plus successful DFT collection, fixed audit
benchmark, completed SAFE spot checks, the current disjoint dataset split, prior-round history,
campaign record, and optional transport evidence. The cumulative dataset path and split IDs may change
as training labels grow, while calibration and immutable audit IDs may not. The labeling handoff must
map every selected QUERY sample to its canonical DFT record and bind the current cumulative dataset
and split IDs; the checker verifies those records are present in the updated training/validation
pool. The audit benchmark manifest must carry the exact immutable audit IDs, and the validation helper
derives that binding only after matching fresh benchmark prediction evidence to those IDs. Ordinary
canonical test records remain excluded from train/validation rather than being silently renamed. The
checker deterministically recomputes the complete result and rejects drift.

Round assessment summarizes the calibrated candidate-risk distribution with count,
minimum/mean/q50/q90/maximum for the combined risk and every committee, and reports mean/q90/max
deltas when the prior round summary is present. This is bounded provenance for marginal-gain review,
not a new uncertainty model or an independent stopping gate.

For the bounded validation, run the packaged `split-seed-review` handoff before cumulative dataset
assembly. It invokes the exact reviewed `dft-labeling/dataset_contract.py`, records the lowest
non-negative deterministic seed that keeps every prior train/validation record and new QUERY record
in train or validation, and records the source paths. The dataset-assemble node must then
independently reproduce that seed and split; the handoff is not a substitute for plugin checking.

When a Strategy A round audits every retrained committee member, aggregate only after checking that
all members used the same immutable sample IDs and units. The bounded validation uses the conservative
maximum member error for each declared audit gate and retains every member value; it never promotes an
arbitrary seed to stand in for the committee.

Use collected artifact bindings from final `OK` direct dependencies. Do not type an attempt path to
smuggle a failed or superseded artifact downstream and do not ask the user to move intermediate
files manually.

## Decisions and claims

- `CONTINUE`: gates are not yet all satisfied; create one next immutable round.
- `CONVERGED_FOR_DECLARED_DOMAIN`: all PES gates and consecutive-round requirement passed for the
  stated domain.
- `BLOCKED_CALIBRATION`: independent calibration evidence is absent or failed.
- `BLOCKED_SAMPLING`: at least one target condition lacks usable evaluated candidates.
- `BUDGET_EXHAUSTED`: the explicit total label maximum was reached while convergence gates failed.
- `SCIENTIFIC_REVIEW_REQUIRED`: collected DFT/other evidence is incomplete or inconsistent.

`BUDGET_EXHAUSTED` is never convergence. Report `PES_ACTIVE_LEARNING_CONVERGENCE` separately from
`TRANSPORT_CONVERGENCE`; a PES result does not establish diffusion or conductivity stability.

Preserve these validation labels:

- `ORACLE_REPLAY_VALIDATION`: existing labels/predictions were structurally replayed;
- `FRESH_ACTIVE_LEARNING_ROUND`: the declared numerical stages executed freshly and their plugins
  returned final `OK`;
- local integration smoke: a bounded software-path test only.

Never strengthen replay or smoke evidence into a fresh production claim.
