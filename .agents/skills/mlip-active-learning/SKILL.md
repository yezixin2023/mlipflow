---
name: mlip-active-learning
description: Supervise finite, offline, round-based MLIP active-learning campaigns using one same-framework committee or a calibrated two-framework risk union. Use when creating, running, replaying, resuming, or assessing campaigns that reuse MLIPFlow training, MD, DIRECT, DFT labeling, canonical dataset, and benchmark stages; do not use it for on-the-fly MD/DFT, uncalibrated uncertainty, model-brand selection, or open-ended automatic loops.
---

# MLIP active learning

Use `plugins/active-learning` as the deterministic decision layer. Keep campaigns
finite, offline, and divided into immutable round projects; do not turn MLIPFlow into
an on-the-fly or open-ended dynamic loop.

## Route the operation

- Evaluate a declared same-framework committee, or a calibrated two-framework risk
  union, against candidate and calibration evidence: `committee-evaluate`.
- Select DFT candidates from checked evaluation and structure/PES evidence:
  `select-candidates`.
- Assess coverage, independent accuracy, integrity, budgets, and consecutive-round
  stability: `assess-round`.

Use the current specialist Skills for training, MD, DIRECT, DFT, dataset assembly, and
benchmark stages. Do not reimplement their contracts here.

Read [references/active-learning-contract.md](references/active-learning-contract.md)
only when drafting or changing campaign policy, reviewing the calibrated
two-framework strategy, resolving ambiguous historical oracle evidence, or explaining
an exceptional convergence dispute. Ordinary operation routing and `assess-round` do
not require it.

## Artifact first

Inventory verified canonical labels, fixed splits, models, trajectories, committee
evidence, selections, DFT results, and audit benchmarks before planning a round. Reuse
matching accepted artifacts and start at the earliest missing stage. Keep historical
oracle evidence read-only and label it `ORACLE_REPLAY_VALIDATION`; it is not a fresh
round. Never edit the canonical dataset or replace prior rounds.

## Scientific judgment

Require an explicit target domain, committee strategy, calibration and acquisition
policy, DFT budgets, immutable audit policy, and consecutive-round requirement. Do not
derive thresholds from model brand, plots, training loss, or another chemistry.
Calibration and immutable audit evidence must remain independent of training data.

Do not compute committee uncertainty, calibrate risk, rank candidates, or decide
convergence in the Skill. Interpret the plugin result using three distinct concepts:
coverage, independent accuracy, and consecutive stability with complete round
integrity. Uncertainty-distribution change is diagnostic, not a stopping gate.

`BUDGET_EXHAUSTED` is not convergence. `CONVERGED_FOR_DECLARED_DOMAIN` is bounded to
the declared PES domain and does not establish transport convergence.

## Execute and report

Use `mlipflow inspect` and each node's dry-run. Follow its effective
`approval_required` value rather than copying approval rules into this Skill. A request
for end-to-end execution does not waive a new approval requirement or a missing
scientific decision.

Never call plugin runners or schedulers directly or bypass Adapter
`validate/plan/execute/check/collect`. Final plugin `OK`, not scheduler `COMPLETED` or
process exit zero, is required before artifacts or decisions are reused. Retry creates
a fresh attempt in the same round; only a checked `CONTINUE` decision creates one next
scientific round.

Report per-round and cumulative label use, calibration/coverage evidence, independent
audit accuracy, integrity, budgets, consecutive stability, the literal decision, and
transport status separately. Distinguish `FRESH_ACTIVE_LEARNING_ROUND`,
`ORACLE_REPLAY_VALIDATION`, and bounded integration smoke.
