# Changelog

All notable user-facing changes to MLIPFlow are documented here.

MLIPFlow follows semantic versioning for the Python package. Scientific validation status is tracked separately because software implementation, numerical agreement, and real-HPC validation can advance independently.

## Unreleased

### Added

- A real offline candidate-ranking example and complete training, ASE MD and ionic
  transport task recipes in the repository.

- Finite offline active-learning workflows with calibrated one- or two-model committees, risk-union candidate selection, reviewed DIRECT subsets, and immutable round assessment.
- Direct handoff of verified VASP AIMD `vasprun.xml` trajectories from `dft-labeling` to `ionic-transport`.
- Bounded AIMD-versus-MLIP partial RDF comparison with explicit trajectory windows, atom pairs, radial grids, common-temperature transport errors, and machine-readable comparison evidence.
- Task-separated benchmark normalization for static PES, structural dynamics, and ionic-transport evidence across an explicitly declared model set.

### Changed

- Require Python 3.12 or newer for the core and all dependency extras.
- Install runtime code and one copy of each Agent Skill; examples and documentation
  are available from the repository.
- Remove standalone configuration JSON Schemas and the `jsonschema` development dependency.
- Preserve artifact paths, reported metrics, checks, manifests and logs in CLI JSON;
  synchronous execution/check/collection failures now return exit code 1 and `ok: false`.
- Choose explicit/default/sole site profiles without
  cross-cluster probes; cross-cluster selection now requires `backend_profile: auto`.
- Accept ID-only dataset references with documented path/kind defaults and report
  unreadable references directly. The explicit
  `validation_profile: deepmd-curve` retains bounded historical trajectory checks.
- Simplified active-learning convergence to declared-domain coverage, immutable-audit accuracy,
  and consecutive stability; new-label marginal gain is no longer a stopping gate.
- Clarified that MLIP framework environments, cluster software, model weights, datasets, and licensed scientific resources remain user- or site-owned.
- Refreshed security guidance, citation metadata, and package-distribution metadata for external users.

## 0.1.0 development baseline

Initial development package; no public release has been published.

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

The 0.1.0 package includes software, local integration, replay, and scientific-contract tests. Real-HPC and numerical validation are reported per capability rather than implied by the package version.
