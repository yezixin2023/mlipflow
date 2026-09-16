---
name: mlip-active-learning
description: Supervise finite, offline, round-based MLIP active-learning campaigns using one same-framework committee or a calibrated two-framework risk union. Use when creating, running, replaying, resuming, or assessing campaigns that reuse MLIPipe training, MD, DIRECT, DFT labeling, canonical dataset, and benchmark stages; do not use it for on-the-fly MD/DFT, uncalibrated uncertainty, model-brand selection, or open-ended automatic loops.
---

# MLIP active learning

Use `mlipipe/plugins/active_learning` as the deterministic decision layer. Keep campaigns
finite, offline, and divided into immutable round projects; do not turn MLIPipe into
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

## Inputs and example

Inventory verified canonical labels, fixed splits, models, trajectories, committee
evidence, selections, DFT results, and audit benchmarks. Reuse matching accepted
artifacts and start at the earliest missing stage. Keep historical oracle evidence
read-only as `ORACLE_REPLAY_VALIDATION`. Never edit the canonical dataset or replace
prior rounds. Use the operation-specific configs in
`examples/active_learning_validation/README.md` and its input-preparation example.
Campaign policy, candidate pool, calibration evidence and independent audit data
must be explicit before starting a round.

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

Retry creates a fresh attempt in the same round; only a checked `CONTINUE` decision
creates the next scientific round.

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

Report per-round and cumulative label use, calibration/coverage evidence, independent
audit accuracy, integrity, budgets, consecutive stability, the literal decision, and
transport status separately. Distinguish `FRESH_ACTIVE_LEARNING_ROUND`,
`ORACLE_REPLAY_VALIDATION`, and bounded integration smoke.
