# Changelog

All notable user-facing changes to MLIPFlow are documented here.

MLIPFlow follows semantic versioning for the Python package. Scientific validation status is tracked separately because software implementation, numerical agreement, and real-HPC validation can advance independently.

## Unreleased

### Added

- Finite offline active-learning workflows with calibrated one- or two-model committees, risk-union candidate selection, reviewed DIRECT subsets, and immutable round assessment.
- Direct handoff of verified VASP AIMD `vasprun.xml` trajectories from `dft-labeling` to `ionic-transport`.
- Bounded AIMD-versus-MLIP partial RDF comparison with explicit trajectory windows, atom pairs, radial grids, common-temperature transport errors, and machine-readable comparison evidence.
- Task-separated benchmark normalization for static PES, structural dynamics, and ionic-transport evidence across an explicitly declared model set.
- A documentation index that separates user guidance, HPC setup, extension references, validation evidence, and manuscript-specific reproduction material.

### Changed

- Consolidate Python implementation into the top-level `mlipflow` package with
  standard imports and one adapter entrypoint per built-in capability.
- Separate DFT, PES and transport operations and isolate scheduled/restart helpers.
- Remove nested adapters, lifecycle mixins, duplicate MD collectors and unused
  controller uploads; generate LAMMPS v2 inputs directly from one runner.
- Ship complete replay example inputs and verify installed and standalone execution.
- Add source migration, example navigation and a focused Agent usage guide.
- Simplified active-learning convergence to declared-domain coverage, immutable-audit accuracy,
  and consecutive stability; new-label marginal gain is no longer a stopping gate.
- Reworked the top-level README around researcher tasks, a shorter quick start, direct artifact handoff, environment boundaries, and scientific validation.
- Combined scientific plugins and their corresponding Agent Skills into one capability matrix, and shortened the detailed Agent Skills guide to supervision and maintenance rules.
- Removed the previous maturity qualifier from user-facing documentation and package metadata without changing the package version or creating a release tag.
- Clarified that MLIP framework environments, cluster software, model weights, datasets, and licensed scientific resources remain user- or site-owned.
- Refreshed security guidance, citation metadata, and package-distribution metadata for external users.

## 0.1.0 - 2026-08-10

Initial packaged release.

### Workflow core

- Versioned `project.yaml` workflow configuration with explicit DAG nodes.
- Local SQLite state with immutable attempt lineage and fresh-attempt retries.
- Read-only inspection commands for workflow state, logs, diagnostics, and evidence-driven model routing.
- Dry-run planning and plan-bound approval for approval-gated execution.
- Local and controlled SSH + Slurm backend contracts with site-local cluster profiles and site-owned templates.
- Structured artifact records, generic result replay, scientific checks, and collection contracts.

### Scientific plugins

- High-entropy/SQS structure generation.
- PES sampling and LASP/SSW integration.
- DFT/VASP input preparation and labeling contracts.
- DeepMD, M3GNet/MatGL, CHGNet, and MACE training/fine-tuning contracts.
- ASE and LAMMPS molecular-dynamics workflows with restart-aware execution.
- MLIP benchmarking and evidence normalization.
- Ionic-transport analysis.
- Deterministic candidate ranking.
- Electrochemical-voltage analysis and manuscript-table normalization.

### Agent supervision

- Bundled Agent Skills for end-to-end MLIP workflow supervision and the major scientific capabilities exposed by the package.

### Validation

The 0.1.0 package includes substantial software, local integration, replay, and scientific-contract tests. Real-HPC and numerical validation are reported per capability rather than implied by the package version. See `docs/IMPLEMENTATION_STATUS.md`, `docs/SCIENTIFIC_VALIDATION.md`, and `docs/HPC_VALIDATION.md` for the current evidence.
