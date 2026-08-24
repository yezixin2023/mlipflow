# SQS scientific judgment and historical guidance

Read this reference only when choosing SQS search controls, reviewing a request beyond
the current one-sublattice model, or interpreting historical/manuscript evidence.
Ordinary schema validation, seed derivation, generation, and checking belong to the
inspected plan and Adapter.

## Current scientific scope

The current capability replaces every site of one explicitly selected prototype
species with an allowed replacement species. It does not represent multiple
independently disordered sublattices, vacancies, interstitial disorder, coupled charge
disorder, oxidation-state balancing, partial site masks, or automatic composition
design.

Do not squeeze those requests into the one-sublattice contract. They need an explicit
scientific model and a separately reviewed implementation.

## Choosing the search

Candidate integer counts define the requested compositions and must be preserved.
Cluster cutoffs, supercell dimensions, search effort, and candidate scope determine the
scientific search; they are not formatting choices and have no universal correct
defaults.

Reuse values only when the user supplied them or a relevant project convention is
identified and disclosed. A proposed value must be labeled as a proposal requiring
review. Changing these controls changes the search and may change the resulting
candidate.

A seed provides reproducible execution under the declared implementation and
environment. It does not establish sampling convergence or optimality. A changed
library/runtime may legitimately produce a different candidate without implying that
either is a globally optimal SQS.

## Historical and replay boundary

Replay of a standard generation result validates existing inputs and structures without
running icet or writing new candidates. Describe it as a **structured collection of
existing results**, not regeneration or independent scientific validation.

The historical generator exposed no seed. Therefore a current seeded run can be a
reproducible new execution but cannot establish historical byte-for-byte parity.
Missing historical inputs or outputs must remain missing; do not infer them or search
for a private legacy generator.

## Claim boundaries

- A generated SQS candidate is not proven globally optimal.
- Deterministic execution is not search-convergence evidence.
- A bounded integration smoke is not a production-quality SQS search.
- Adapter `OK` does not establish material performance, thermodynamic stability,
  synthesis feasibility, or model accuracy.

Use downstream MLIP/DFT/property calculations for candidate evidence and
`$candidate-ranking` only after a metric and ranking policy are explicit.
