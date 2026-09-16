---
name: high-entropy-structure
description: Supervise reproducible, auditable high-entropy or SQS disordered-structure candidate generation through the MLIPipe high-entropy-structure plugin from an explicit prototype, one alloy sublattice, allowed replacement species, integer candidate counts, supercell, SQS parameters, seed, and hard candidate limit. Use when planning, running, replaying, or verifying seeded icet SQS generation; do not use it to discover compositions, design materials automatically, evaluate MLIP/DFT properties, rank candidates, or claim global SQS optimality.
---

# High-entropy structure

Use `mlipipe/plugins/high_entropy_structure` through MLIPipe for `generate-sqs`. The Skill
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

## Inputs and example

Reuse accepted prototype/composition artifacts. Supply `inputs.prototype_structure`
and `inputs.composition_manifest`, plus explicit `parameters.seed` and
`parameters.max_candidates`. The composition manifest declares the one alloy
sublattice, allowed species, integer candidate counts, supercell repeat, cluster
cutoffs, search steps and output format. See `examples/sqs/project.yaml` and
`USAGE.md` for a small complete input example. For existing results, the
`structure-replay` node in `examples/high_entropy_sulfide/project.yaml` is a separate
replay example.

## Scientific judgment

Preserve the user's exact candidate counts and search scope. The user or an explicit
project convention must determine the composition enumeration, cluster cutoffs,
supercell, search effort, seed, output scope, and any custom generator. Do not round or
rebalance compositions, add disorder modes, infer universal defaults, or search for a
private historical script.

The current capability covers one explicitly selected alloy sublattice. Vacancies,
multiple disordered sublattices, charge balancing, or automated composition discovery
need a different reviewed scientific contract.

## Run and read the result

For the existing local inputs, the shortest execution is:

```bash
mlipipe --project PROJECT init
mlipipe --project PROJECT --format json run NODE
```

Use `run NODE --dry-run` when reviewing changed inputs or execution scope; it reports
the effective `approval_required` value. `mlipipe --project PROJECT json NODE`
reads the saved result later.

Read `state`, `metrics`, `artifacts[].role` and the full `artifacts[].path` from JSON.
Require final plugin `OK` before using outputs. On failure start with `reason`,
`check.diagnostics`, `manifest_path` and `logs`; `mlipipe --project PROJECT logs NODE`
shows saved stdout/stderr. Example paths refer to the
[repository examples](https://github.com/yezixin2023/mlipipe/tree/public/examples).

Report generated versus replay mode, search controls, candidate coverage, provenance,
and output paths. Call outputs generated SQS candidates. A seed supports reproducible
execution under the declared contract; it does not prove search convergence, historical
byte parity, global SQS optimality, material performance, or model accuracy. Replay is
a structured collection of existing results, not fresh generation or independent
scientific validation.
