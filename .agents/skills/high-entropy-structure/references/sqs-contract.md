# SQS generation contract

Use this reference for detailed, stable scientific and artifact semantics. The current
`plugins/high-entropy-structure/plugin.yaml`, Adapter, and inspected MLIPFlow plan remain
authoritative if the implementation evolves.

## Composition manifest semantics

The schema-v1 JSON composition manifest declares:

- `alloy_sublattice.prototype_species`: the species marking every currently supported
  disordered site in the prototype;
- `alloy_sublattice.allowed_species`: a unique list with at least two entries that
  includes `prototype_species`;
- `cluster_cutoffs_angstrom`: non-empty positive finite cluster cutoffs;
- `supercell_repeat`: exactly three positive integers;
- `n_steps`: a positive integer search budget;
- `output_format`: `vasp` or `cif` (default `vasp`);
- `candidates`: a non-empty ordered list of unique safe IDs and `counts` mappings.

Every candidate `counts` mapping must contain exactly the allowed species. Counts are
non-negative integers; a zero preserves the explicit species contract. Do not accept
fractions, percentages, inferred stoichiometry, rounded values, omitted allowed species,
or extra species.

## Current one-sublattice boundary

The current contract replaces every occurrence of one explicit `prototype_species`.
It does not express multiple independently disordered sublattices, partial site masks,
vacancies, interstitial disorder, coupled charge disorder, oxidation-state balancing,
or automatic site classification. Require another reviewed contract rather than
pretending these cases fit this one.

## Site-count rule

Read the prototype's actual chemical symbols with a reliable structure parser. Let
`N_prototype` be the number of sites labeled `prototype_species`, and let
`R = repeat_a * repeat_b * repeat_c`. The alloy-site count is:

```text
N_alloy = N_prototype * R
```

For every candidate:

```text
sum(candidate.counts.values()) == N_alloy
```

The generated structure's expected full composition is the repeated prototype
composition with all repeated `prototype_species` sites removed, followed by the exact
candidate alloy counts. Both the generator and Adapter enforce the contract. Never
alter candidate counts to make the equality hold.

## SQS parameter policy

`cluster_cutoffs_angstrom`, `supercell_repeat`, and `n_steps` define the scientific
search. They have no universal correct defaults. Reuse them only when explicitly
provided by the user or a relevant, trusted, and disclosed project convention. When
proposing values, label them as a proposal requiring review and explain that changing
them changes the search—not merely formatting or compute metadata.

`output_format` controls the structure artifact format, not scientific quality.

## Seed policy

Fresh generation requires an explicit non-negative base seed. Candidate order in the
approved composition manifest is significant. The fixed policy is:

```text
candidate_random_seed = base_seed + zero_based_candidate_index
```

The Adapter checks each output seed against that order. A user-supplied generator may
not choose, shuffle, or merely self-report unrelated seeds. A seed is reproducibility
provenance, not evidence of sampling convergence or optimality.

## Candidate limit and approval

`max_candidates` is a positive hard cap. The manifest candidate count must not exceed
it, and output must cover exactly the approved ordered candidates—no missing, extra, or
duplicate IDs. Do not use the cap as permission to truncate silently.

The plugin is local-only and marked expensive. Review `run --dry-run`, including exact
candidate count, parameters, local runtime/resources, generator identity, fresh output
locations, and staged argv. Start generation only with the approval token issued for
that plan. Keep retries in fresh attempts and retain earlier evidence.

## Generator and execution boundary

Omitting `sqs_script` selects the bundled, fingerprinted
`icet.generate_sqs_from_supercells` implementation. Prefer it. A custom project-relative
script is allowed only after explicit user request and review; it must accept the fixed
argv contract, run without a shell, honor fresh outputs and the same deterministic seed
policy, and emit the standard result contract.

Do not put a shell command string in `interpreter_argv`, add override arguments, execute
historical private generators implicitly, enable network loading, or add an HPC/
scheduler backend. Numerical SQS implementation belongs to the plugin, not this Skill.

## Result manifest and checker

The standard generation result binds at least:

- schema/plugin/status and the base `seed`;
- exact `candidate_count`;
- prototype and composition-manifest SHA-256;
- generator identity and source SHA-256;
- ordered structure records with ID, safe attempt-relative path, structure SHA-256,
  exact declared alloy composition, media type, and per-candidate `random_seed`;
- method provenance, including seed policy and, for the bundled generator, icet library/
  API/version and ASE version identities;
- optional `cluster_vector` per structure.

If `cluster_vector` is present, it must be a non-empty finite numeric array. Its presence
does not mean the Adapter recomputed the icet objective or established a global optimum.

Adapter `check/collect` independently rereads the approved composition manifest and
actual structure artifacts. It enforces candidate coverage/order/uniqueness, exact
species/counts, alloy site count, seed derivation, candidate cap, input and generator
identity, structure fingerprints, and full structure composition. Missing, changed,
malformed, non-finite, out-of-contract, or self-inconsistent evidence is `FAIL`.

Framework versions are provenance. Same or different ASE/icet version strings do not by
themselves establish scientific structure parity.

## Replay semantics

Replay consumes an existing standard `generation-result.json` plus the referenced,
verifiable approved inputs and structure artifacts. It performs the same contract and
fingerprint checks without importing/executing the generator, running icet, writing new
structures, or mutating source evidence. Describe success as read-only verification of
existing generation evidence, never as regeneration or independent scientific
validation.

## Scientific claim boundaries

- “generated SQS candidate” does not mean “globally optimal SQS.”
- deterministic output under the declared seed policy does not prove convergence.
- a new seeded execution cannot establish historical byte identity when the historical
  generator had no seed.
- `LOCAL_INTEGRATION_SMOKE_PASS` proves only the bounded wrapper/seed/write/manifest
  integration and determinism exercised by that smoke; it is not production-quality SQS.
- Adapter `OK` proves contract consistency and verifiable provenance, not material
  performance, thermodynamic stability, synthesis feasibility, or model accuracy.

## Downstream handoff

This Skill stops at verified structure candidates. Use explicit downstream plugins for
MLIP inference, DFT labeling, MD, or property calculation. Only after comparable numeric
metrics exist should `$candidate-ranking` apply its explicit metric/direction/top-k
policy. Never rank candidates by composition intuition, cluster-vector presence, file
order, or generator brand.
