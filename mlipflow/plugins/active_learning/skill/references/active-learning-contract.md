# Advanced offline active-learning guidance

Read this reference only when drafting or changing a campaign policy, reviewing the
calibrated two-framework strategy, resolving ambiguous historical oracle evidence, or
explaining a disputed convergence decision. Ordinary
`committee-evaluate`, `select-candidates`, and `assess-round` supervision should use the
Skill, inspected plan, and Adapter result.

## Campaign and round boundary

Keep a finite campaign record and one immutable MLIPFlow project per scientific round.
A retry is a fresh attempt inside the same round; it does not replace evidence or create
the next round. Create one next round only after a checked `CONTINUE` decision.

Reuse accepted training, MD, selection, DFT, dataset, and benchmark artifacts. Do not
repeat expensive stages ceremonially, edit the canonical dataset, or move artifacts
between attempt directories by hand.

## Policy choices requiring scientific review

The campaign policy must explicitly define:

- the target composition, structure, thermodynamic, defect, and intended-use domain;
- a same-framework committee strategy or a calibrated two-framework risk union;
- independent calibration, acquisition, SAFE spot-check, and immutable audit policies;
- per-round and total DFT budgets;
- condition-level coverage expectations and independent accuracy gates;
- the number of consecutive passing rounds required.

These choices have no universal defaults. Do not derive them from model brand, training
loss, a plot, or thresholds used for another material.

Training, validation, calibration, immutable audit/test, and SAFE spot-check roles must
remain scientifically distinct. Calibration or audit gaps cannot be filled with
training error.

## Two-framework risk union

Each framework uses its own same-framework committee and independent calibration over
the same scientific domain. A candidate is safe only when neither calibrated committee
flags it. Cross-framework prediction disagreement is useful diagnostic evidence but is
not a substitute for either committee's calibration.

Candidate selection must retain representation of the models that triggered risk when
the declared budget and available evidence permit it. Missing trigger-model coverage is
an integrity failure, not evidence that domain coverage passed.

## Convergence interpretation

Interpret a checked round using separate concepts:

1. **Coverage**: calibrated uncertainty, target-condition coverage, and SAFE spot-check
   evidence support the declared domain.
2. **Accuracy**: immutable independent audit evidence satisfies the declared error
   gates for every strategy model.
3. **Consecutive stability and integrity**: coverage and accuracy hold for the required
   consecutive rounds and the selection/labeling evidence is complete.

Uncertainty-distribution change is diagnostic only. New-label saturation or marginal
gain is not an implicit convergence gate.

`BLOCKED_CALIBRATION`, `BLOCKED_SAMPLING`, and
`SCIENTIFIC_REVIEW_REQUIRED` mean evidence is missing or inconsistent; they are not
negative convergence results that should be bypassed. `BUDGET_EXHAUSTED` is
non-converged. `CONVERGED_FOR_DECLARED_DOMAIN` is limited to the stated PES domain and
does not establish diffusion or conductivity convergence.

## Historical and validation labels

- `ORACLE_REPLAY_VALIDATION`: existing labels or predictions were replayed
  structurally; no fresh active-learning round is established.
- `FRESH_ACTIVE_LEARNING_ROUND`: the declared numerical stages ran freshly and all
  relevant plugins returned final `OK`.
- Integration smoke: only a bounded software-path check.

Never strengthen replay or smoke evidence into a fresh production claim. Report PES
active-learning convergence separately from transport convergence.
