---
name: mlip-active-learning
description: Supervise finite, offline, round-based MLIP active-learning campaigns using one same-framework committee or a calibrated two-framework risk union. Use when creating, running, replaying, resuming, or assessing campaigns that reuse MLIPFlow training, MD, DIRECT, DFT labeling, canonical dataset, and benchmark stages; do not use it for on-the-fly MD/DFT, uncalibrated uncertainty, model-brand selection, or open-ended automatic loops.
---

# MLIP active learning

Use `plugins/active-learning` as the deterministic decision layer. Read
[references/active-learning-contract.md](references/active-learning-contract.md) before creating a
campaign, connecting round artifacts, executing an operation, or interpreting a decision.

## Establish the campaign contract

Require an explicit target domain, Strategy A (`single-model-committee`) or Strategy B
(`dual-model-risk-union`), committee member seeds, calibration coverage, force-error thresholds,
target-condition gates, SAFE spot-check policy, DFT budgets, immutable audit thresholds, and
consecutive-round count. Do not invent any missing scientific value.

Bind each committee to one seed-independent training contract: exact cumulative canonical label
path, framework dataset export path, train/validation/test record IDs, foundation-model path,
scientific config, precision, and operation. Members of one committee may differ only by their
declared seeds; Strategy B model families must share the same canonical-label path and split IDs.

Inventory existing verified artifacts first. Reuse accepted canonical labels, model artifacts,
trajectories, and benchmark evidence when their paths, recorded parameters, and conventions match. Historical labels
used as an oracle must remain read-only and the validation claim must be
`ORACLE_REPLAY_VALIDATION`; it is not a fresh active-learning round.

Keep cumulative training, calibration, and immutable audit/test sample IDs disjoint. Both Strategy B
models must use the same split IDs and DFT labels. If calibration or audit evidence is too small or
missing, stop at `BLOCKED_CALIBRATION`; never substitute training error.

## Supervise one immutable round

Each `round-NNN/` is one ordinary fixed-DAG MLIPFlow project. Use the current specialist Skills for
committee training, ASE/LAMMPS exploration MD, `pes-sampling` DIRECT, DFT labeling and canonical
dataset assembly, model benchmark, and optional transport. Read each selected Skill and plugin
manifest when its stage becomes current.

Do not turn MLIPFlow core into a dynamic DAG or infinite loop; campaign progression remains an
Agent-supervised decision between immutable projects.

Call the active-learning operations only through MLIPFlow:

1. `committee-evaluate` either locally over explicit member predictions and calibration DFT
   evidence, or through reviewed SSH-SLURM fresh inference over an immutable committee model index
   and bounded calibration/candidate dataset;
2. `select-candidates` over the checked evaluation, structure manifest, and collected DIRECT
   evidence;
3. `assess-round` over checked selection, fresh/replayed DFT results, fixed audit benchmark, SAFE
   spot checks, the current disjoint split, immutable audit IDs, campaign lineage, and prior-round
   summaries.

Do not compute committee variance, calibrate thresholds, classify frames, rank candidates, or judge
convergence in the Skill. Require the plugin result and final Adapter `OK`; scheduler `COMPLETED` or
process exit zero is insufficient.

## Preserve execution and round boundaries

Training, scheduled inference, exploratory MD, DFT, and other expensive fresh stages retain their
own dry-run, boolean approval, checker, and collection boundaries. Core fresh-attempt retry semantics
also remain in force. A user asking for automatic or
end-to-end execution does not waive them. Local `committee-evaluate`, `select-candidates`, and
`assess-round` have `approval_required: false`; inspect their dry-runs and continue without an
approval stop. An SSH-SLURM `committee-evaluate` requires approval because it submits scheduled
fresh inference. Scientific policy, calibration, selection, and assessment validation remain
independent of approval and must still block on missing or inconsistent evidence.

Never modify the canonical dataset directly. Bind successful collected DFT labels into the existing
dataset assembly handoff. Retry creates a fresh attempt inside the same round; it never creates a
new scientific round. Preserve every prior round and artifact.

Read `round-assessment.json` literally:

- treat `coverage_passed` as the combined result of calibrated committee uncertainty, every
  condition's QUERY/UNSAFE limits, and SAFE spot checks;
- treat `accuracy_passed` as the result of the immutable independent audit energy/force gates;
- require selection, labeling, and Strategy B trigger-model coverage integrity before setting
  `pes_gates_passed_this_round`, without folding that integrity into `coverage_passed`;
- count a round toward consecutive stability only when its PES gates passed;
- create exactly one new `round-(N+1)` only for `CONTINUE`;
- stop and request the missing evidence/review for either `BLOCKED_*` state or
  `SCIENTIFIC_REVIEW_REQUIRED`;
- report `BUDGET_EXHAUSTED` as non-converged;
- stop the PES campaign for `CONVERGED_FOR_DECLARED_DOMAIN` without strengthening it to global or
  transport convergence.

Report cumulative and per-round DFT labels, condition-level QUERY/UNSAFE fractions, immutable audit
errors, calibration coverage, SAFE spot-check false negatives, current uncertainty distribution and
its prior-round change, coverage, accuracy, consecutive stability, PES status, and the independent
transport status. Uncertainty-distribution change is diagnostic only and does not determine
convergence.
