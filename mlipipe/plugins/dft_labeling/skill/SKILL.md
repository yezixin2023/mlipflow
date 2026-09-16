---
name: dft-labeling
description: Supervise MLIPipe DFT input preparation and labeling. Use for generating VASP POSCAR/INCAR/KPOINTS/POTCAR sets with pymatgen, choosing static/relax/AIMD contracts, reviewing pseudopotential functional and symbols, launching reviewed local or SSH-SLURM static DFT labels, or interpreting convergence and dataset manifests.
---

# DFT labeling

Use `mlipipe/plugins/dft_labeling` through MLIPipe. The Skill chooses the scientific task and
reviews it; input generation, DFT execution, validation, and collection belong to the
Adapter and VASP.

## Route the request

- Generate and review VASP inputs: `vasp-prepare`.
- Run static, relaxation, or AIMD labeling from reviewed prepared inputs: `label`.
- Serialize verified canonical labels into requested DeepMD, M3GNet/MatGL, CHGNet, or
  MACE dataset views: `dataset-assemble`.

Keep preparation, DFT execution, and dataset serialization as distinct operations. Do
not use a custom preparation hook to collapse them.

## Inputs and example

Reuse a verified `dft_input_manifest`, canonical dataset and shared split when they
match. Preparation needs `structures_manifest` and `labeling_config`; label adds the
prepared `dft_input_manifest`. `dataset-assemble` consumes `canonical_dataset` and
an explicit shared split policy. `examples/aimd_reference_validation/project.yaml`
contains the AIMD path; `examples/training_all_models/dft-to-all-training.yaml`
shows static labels and assembly into framework-specific training data.
Pseudopotential functional/symbols and convergence settings must already be chosen.

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

## Run and read the result

Preview the actual task and effective `approval_required` value. Reuse explicit
user authorization for this task and scale; use `--approve` when the plan requires it.

```bash
mlipipe --project PROJECT init
mlipipe --project PROJECT --format json run NODE --dry-run
mlipipe --project PROJECT --format json run NODE --approve
```

For a plan without an approval requirement, `run NODE` suffices. For scheduled
execution, use `mlipipe --project PROJECT --format json advance` when the job
progresses; `mlipipe --project PROJECT json NODE` reads the saved result.

Read `state`, `metrics`, `artifacts[].role` and the full `artifacts[].path` from JSON.
Require final plugin `OK` before using outputs. On failure start with `reason`,
`check.diagnostics`, `manifest_path` and `logs`; `mlipipe --project PROJECT logs NODE`
shows saved stdout/stderr. Example paths refer to the
[repository examples](https://github.com/yezixin2023/mlipipe/tree/public/examples).

Report what actually occurred: input preparation, fresh DFT, dataset assembly, or
replay. Preparation `OK` does not establish that VASP ran. Label `OK` establishes the
declared convergence and dataset contract, not functional accuracy or transferability.
Replay is a structured collection of existing results, not recomputation or independent
scientific validation.
