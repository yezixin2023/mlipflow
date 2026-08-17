---
name: high-entropy-structure
description: Supervise reproducible, auditable high-entropy or SQS disordered-structure candidate generation through the MLIPFlow high-entropy-structure plugin from an explicit prototype, one alloy sublattice, allowed replacement species, integer candidate counts, supercell, SQS parameters, seed, and hard candidate limit. Use when planning, running, replaying, or verifying seeded icet SQS generation; do not use it to discover compositions, design materials automatically, evaluate MLIP/DFT properties, rank candidates, or claim global SQS optimality.
---

# High-entropy structure

Use `plugins/high-entropy-structure` as the deterministic implementation. Supervise
inputs, approval, execution, checking, and interpretation; never implement or imitate
the SQS search in this Skill.

Read the current plugin manifest and inspect the actual workflow node before planning.
Do not copy parameters from old documentation or invoke `sqs.py` outside the MLIPFlow
lifecycle. Read [references/sqs-contract.md](references/sqs-contract.md) whenever
constructing or validating a composition manifest, reviewing result provenance, or
handling replay.

## Select the mode

- For an explicit prototype and explicit candidate integer counts, prepare a fresh
  generation contract.
- If the user asks to “design some high-entropy compositions” without candidate counts
  or an explicit enumeration rule, stop and request them. Do not invent a search space.
- For an existing standard `generation-result.json`, use replay and verify existing
  inputs/artifacts only. Never run icet or regenerate a structure during replay.
- For “which candidate performs best,” hand off to explicit downstream MLIP/DFT or
  property calculations and then `$candidate-ranking`. This Skill generates no property
  values and performs no ranking.

## Establish the scientific contract

1. Confirm the exact prototype structure and its identity.
2. Confirm one explicit alloy sublattice selected by `prototype_species`. Do not claim
   arbitrary multi-sublattice support or introduce vacancies/charge disorder.
3. Confirm the complete unique `allowed_species` list.
4. Confirm every candidate ID and every species' non-negative integer count. Preserve
   them exactly: never round, rebalance, add, or delete species.
5. Parse the actual prototype and account for `supercell_repeat`. Require each
   candidate's count sum to equal the repeated alloy-sublattice site count. Show a
   mismatch and stop; do not repair it silently.
6. Require explicit `cluster_cutoffs_angstrom`, `supercell_repeat`, and `n_steps` from
   the user or a trusted project convention, and confirm `output_format` as `vasp` or
   `cif`. If a search parameter is absent, explain that it requires scientific judgment
   and either request it or offer a clearly labeled, reviewable proposal—not a unique
   correct default.
7. Require an explicit non-negative deterministic `seed`. If the user has no
   preference, propose one and display it in the plan.
8. Require a positive hard `max_candidates`. For a large enumeration, show the exact
   candidate count, input scope, local resources, outputs, and expected cost in the
   dry-run, then obtain the approval required for this expensive operation.

Default to the bundled generator. Set `sqs_script` only when the user explicitly asks
for a reviewed custom generator that implements the same result and seed contract.
Never search for or execute a historical private script by default.

## Execute and verify

Keep the operation `generate-sqs` local-only and `shell: false`. Use a fresh attempt;
never overwrite an output directory/result manifest or delete an earlier attempt to
simulate retry. Follow the MLIPFlow dry-run and approval token lifecycle before fresh
generation.

After execution, require Adapter `check/collect` and final plugin `OK`. Do not accept
process exit zero or self-reported JSON alone. Confirm candidate IDs/order and coverage,
integer compositions, structure count, per-candidate seed policy, prototype and
composition-manifest identities, structure SHA-256, generator provenance, actual
structure composition, and any declared cluster vector.

## Report without overclaiming

Briefly report mode, prototype, alloy sublattice, allowed species, candidate counts,
supercell, cutoffs, `n_steps`, seed, generated count, output paths/fingerprints,
generator/icet identity, and scientific limitations. Do not dump the full JSON unless
asked.

Call each output a **generated SQS candidate**. Never equate deterministic same-seed
execution with a historically identical structure, a bounded local smoke with a
production-quality converged search, or Adapter `OK` with proof of global SQS
optimality. The historical generator exposed no seed; current seeded output is a
reproducible new execution, not historical byte parity.
