# Changelog

All notable user-facing changes to MLIPFlow are documented here.

MLIPFlow follows semantic versioning for the Python package. Scientific
validation status is tracked separately because software implementation,
numerical parity, and real-HPC validation can advance independently.

## Unreleased

### Documentation and release preparation

- Reworked the top-level README around user goals, quick start, scientific
  capabilities, Agent Skills, environment setup, HPC configuration, model
  routing, and extension points.
- Added clearer package metadata and project links for distribution tooling.
- Added this changelog and a project NOTICE while retaining the standard
  Apache-2.0 license text.
- Refreshed contribution and security guidance for external users and
  contributors.

## 0.1.0 - 2026-08-10

Initial alpha package version.

### Workflow core

- Versioned `project.yaml` workflow configuration with explicit DAG nodes.
- Local SQLite state with immutable attempt lineage and fresh-attempt retries.
- Read-only inspection commands for workflow state, logs, diagnostics, and
  evidence-driven model routing.
- Dry-run planning and plan-bound approval for approval-gated execution.
- Local and controlled SSH + Slurm backend contracts with site-local cluster
  profiles and site-owned templates.
- Structured artifact identity, provenance, replay, scientific checks, and
  collection contracts.

### Scientific plugins

- High-entropy/SQS structure generation.
- PES sampling and LASP/SSW integration.
- DFT/VASP input preparation and labeling contracts.
- DeepMD, M3GNet/MatGL, CHGNet, and MACE training/fine-tuning contracts.
- ASE and LAMMPS molecular-dynamics workflows with restart-aware execution.
- MLIP benchmarking and evidence normalization.
- Ionic-transport analysis.
- Deterministic candidate ranking.
- Electrochemical-voltage analysis and evidence replay.

### Agent supervision

- Bundled Agent Skills for end-to-end MLIP workflow supervision and the major
  scientific capabilities exposed by the package.

### Validation

The 0.1.0 package includes substantial software, local integration, replay,
and scientific-contract tests. Real-HPC and numerical validation are reported
per capability rather than implied by the package version. See
`docs/IMPLEMENTATION_STATUS.md`, `docs/SCIENTIFIC_VALIDATION.md`, and
`docs/HPC_VALIDATION.md` for the current evidence.
