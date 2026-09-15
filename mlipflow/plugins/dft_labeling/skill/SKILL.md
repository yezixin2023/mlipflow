---
name: dft-labeling
description: Supervise MLIPFlow DFT input preparation and labeling. Use for generating VASP POSCAR/INCAR/KPOINTS/POTCAR sets with pymatgen, choosing static/relax/AIMD contracts, reviewing pseudopotential functional and symbols, launching reviewed local or SSH-SLURM static DFT labels, or interpreting convergence and dataset manifests.
---

# DFT labeling

Use `mlipflow/plugins/dft_labeling` through MLIPFlow. The Skill chooses the scientific task and
reviews it; input generation, DFT execution, validation, and collection belong to the
Adapter and VASP.

## Route the request

- Generate and review VASP inputs: `vasp-prepare`.
- Run static, relaxation, or AIMD labeling from reviewed prepared inputs: `label`.
- Serialize verified canonical labels into requested DeepMD, M3GNet/MatGL, CHGNet, or
  MACE dataset views: `dataset-assemble`.

Keep preparation, DFT execution, and dataset serialization as distinct operations. Do
not use a custom preparation hook to collapse them.

## Artifact first

Reuse a verified `dft-input-manifest.json`, canonical labeled dataset, shared split, or
framework dataset reference when it matches the objective. Do not rerun VASP or
reconvert a dataset merely to reconstruct an end-to-end workflow. A final `OK` AIMD
node supplies its collected trajectory directly to `$ionic-transport`.

For model comparisons, preserve one framework-independent split and its held-out
benchmark records across all requested framework views. Do not create a new split for
each framework or derive benchmark labels from a framework-specific view.

## Scientific judgment

The user or an evidence-backed project convention must determine the structures,
calculation type, convergence settings, pseudopotential functional and symbols, and
dataset split policy. Do not guess unresolved DFT settings, reference energies, units,
seeds, or split choices.

Keep licensed POTCAR material outside the repository and reports. Never display,
collect, package, or commit its content or resolved private path; review only the
portable functional/symbol reference and license acknowledgement.

Treat the optional manuscript preset as evidence-bound. Preserve disclosed differences
between reported manuscript wording and VASP semantics rather than silently
normalizing them.

## Execute and interpret

Use `mlipflow inspect` and the dry-run to review the actual structures, calculation
type, inputs, backend, abstract resources, staged plan, and expected artifacts. DFT
execution requires explicit user intent. Follow the plan's effective
`approval_required` value rather than duplicating operation/backend approval rules in
this Skill.

Never launch VASP or a scheduler directly, guess site-owned cluster details, or bypass
Adapter `validate/plan/execute/check/collect`. Scheduler `COMPLETED` and process exit
zero are not scientific success; require final plugin `OK`. Diagnose calculation-level
failure before proposing a fresh retry, and preserve prior attempts.

Report what actually occurred: input preparation, fresh DFT, dataset assembly, or
replay. Preparation `OK` does not establish that VASP ran. Label `OK` establishes the
declared convergence and dataset contract, not functional accuracy or transferability.
Replay is a structured collection of existing results, not recomputation or independent
scientific validation.
