---
name: high-entropy-structure
description: Supervise reproducible, auditable high-entropy or SQS disordered-structure candidate generation through the MLIPFlow high-entropy-structure plugin from an explicit prototype, one alloy sublattice, allowed replacement species, integer candidate counts, supercell, SQS parameters, seed, and hard candidate limit. Use when planning, running, replaying, or verifying seeded icet SQS generation; do not use it to discover compositions, design materials automatically, evaluate MLIP/DFT properties, rank candidates, or claim global SQS optimality.
---

# High-entropy structure

Use `plugins/high-entropy-structure` through MLIPFlow for `generate-sqs`. The Skill
reviews the requested search space and scientific claims; the plugin owns composition
validation, seeded generation, checking, and collection.

## Route the request

- Explicit prototype, one alloy sublattice, allowed species, and integer candidate
  counts: generate or verify SQS candidates.
- Existing standard generation result: reuse or replay it without running icet.
- An open-ended request to design compositions: request an explicit enumeration rule;
  do not invent the search space.
- A request for the best candidate: obtain comparable downstream property evidence,
  then use `$candidate-ranking`.

Read [references/sqs-contract.md](references/sqs-contract.md) only when human review is
needed for SQS search controls, a request exceeds the current one-sublattice model, or
historical/manuscript evidence needs interpretation. It is not required to restate the
ordinary composition schema or checker rules.

## Artifact first

Reuse verified prototype, composition, and generation artifacts when their scientific
contract matches. Do not regenerate candidates merely to confirm a standard result or
to reconstruct a canonical workflow. Never mutate historical evidence during replay.

## Scientific judgment

Preserve the user's exact candidate counts and search scope. The user or an explicit
project convention must determine the composition enumeration, cluster cutoffs,
supercell, search effort, seed, output scope, and any custom generator. Do not round or
rebalance compositions, add disorder modes, infer universal defaults, or search for a
private historical script.

The current capability covers one explicitly selected alloy sublattice. Vacancies,
multiple disordered sublattices, charge balancing, or automated composition discovery
need a different reviewed scientific contract.

## Execute and interpret

Use `mlipflow inspect` and the dry-run to review the exact prototype, composition
scope, SQS controls, candidate count, generator, scale, and outputs. Follow the plan's
effective `approval_required` value rather than hard-coding it in this Skill.

Never call icet or a generator outside MLIPFlow or bypass Adapter
`validate/plan/execute/check/collect`. Final plugin `OK`, not process exit zero or a
self-reported result, is required. Preserve earlier attempts and use a fresh retry.

Report generated versus replay mode, search controls, candidate coverage, provenance,
and output paths. Call outputs generated SQS candidates. A seed supports reproducible
execution under the declared contract; it does not prove search convergence, historical
byte parity, global SQS optimality, material performance, or model accuracy. Replay is
a structured collection of existing results, not fresh generation or independent
scientific validation.
