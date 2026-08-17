---
name: pes-sampling
description: Supervise MLIPFlow potential-energy-surface sampling with local MAML/DIRECT representative-structure selection, local LASP/SSW execution or historical ARC normalization, and reviewed SSH-SLURM LASP execution. Use when planning, running, retrying, or verifying the pes-sampling plugin's direct-select, lasp-ssw-execute, or lasp-ssw-normalize-replay operation.
---

# Potential-energy-surface sampling

Use the current `plugins/pes-sampling` manifest and adapters as the deterministic
implementation. Do not estimate energies, choose structures by intuition, implement
SSW, or infer undocumented LASP inputs in this Skill.

## Choose exactly one operation

- Use `direct-select` for local MAML/DIRECT representative-structure selection.
- Use `lasp-ssw-execute` to run a user- or site-supplied LASP executable, locally or
  through MLIPFlow's `ssh-slurm` lifecycle.
- Use `lasp-ssw-normalize-replay` to normalize and verify an existing LASP archive
  locally without running LASP.

Do not describe DIRECT as LASP, an archive replay as a new LASP calculation, or any of
these operations as DFT labeling, AIMD, MLIP-MD, or model training.

## Inspect before planning

Run only MLIPFlow read-only inspection/status commands until the scientific inputs and
backend are understood. Inspect the selected operation, all important inputs, the
literal `lasp.in`, potential type, auxiliary-file names, output bounds, and requested
resources. Treat repository example inputs as contract fixtures, not real scientific
inputs.

Do not guess a LASP parameter, random seed, potential, auxiliary file, executable,
launcher, or resource request. If an input is missing or ambiguous, report the missing
scientific contract and stop.

## `direct-select`

Use `backend: local`. Require:

- `inputs.direct_script`: the reviewed `direct.py`;
- `inputs.input_dirs`: one or more existing local structure directories;
- non-empty relative `globs`, boolean `recursive`, positive `stride`, `n_clusters`,
  `threshold_init`, and `k_per_cluster`;
- optional positive `max_frames_per_file` and `max_input_structures`;
- optional positive-integer-to-element `lammps_type_map`;
- an explicit non-negative recorded `seed` and
  `acknowledge_uncontrolled_seed: true`.

The reviewed DIRECT CLI has no seed flag. Record the declared seed but state that it
does not control the source CLI. Do not claim seeded reproducibility.

Keep `output_subdir` fresh, relative, confined to the attempt, and disjoint from every
input directory. The source CLI deletes an existing `--out-dir`; MLIPFlow therefore
blocks an existing output rather than allowing destructive reuse. Preserve
`shell: false`. Completion requires a valid non-empty `manifest.csv` and confined
selected structure files; optional plots are diagnostics only.

Use only the current manifest parameter names. Do not invent aliases for cluster count,
threshold, input limits, or output directory.

## Common LASP/SSW contract

Require an explicit `lasp.in` that parses to `explore_type ssw` and integer
`SSW.SSWsteps >= 1`. Inline `#` comments are ignored by the current parser. Also
require:

- a portable `historical_source_id` without a private absolute path;
- positive `selection_stride`;
- finite `energy_max_ev` or null;
- explicit positive `max_frames` no greater than 10000;
- boolean `include_best_arc` and `include_md_arc`;
- `seed_status: HISTORICAL_PARAMETER_UNKNOWN`;
- `acknowledge_uncontrolled_seed: true`;
- `preserve_historical_order: true`.

Keep historical order. Apply the inclusive energy filter first, renumber accepted
frames, and apply `selection_stride` to that accepted order. Never reorder frames by an
inferred notion of quality.

`best.arc` is optional AIMD-seed-candidate lineage and `md.arc` is optional sampled
structure lineage. Include either only when the declared run is expected to produce it.
Their normalization does not run AIMD/MD and does not establish that the structures are
scientifically suitable for those downstream uses.

### Determine the potential without generalizing one case

Read the actual potential declaration and auxiliary-file contract before approving an
execute operation. LASP may use an NN potential, VASP/DFT, or another site-supported
potential; this Skill is not LASP-NN-specific.

- For an NN or other non-DFT potential, require its exact auxiliary files; do not assume
  a filename or element set from an earlier run.
- If `potential vasp` or the runtime would invoke VASP/DFT, disclose the explicit LASP
  settings, required auxiliary-file names, MPI/resources, and DFT call bound derivable
  from declared inputs. Obtain the separate DFT authorization required by repository
  policy before submission. Do not infer a call count from undocumented behavior.
- Never read, display, package, or commit POTCAR content. Keep licensed material outside
  the repository and pass it only through the core input/staging contract.

Do not claim LASP numerical parity. Success proves the declared execution, archive
lineage, selection policy, and checker contract, not independent validation of LASP's
SSW numerics.

## `lasp-ssw-normalize-replay`

Use `backend: local`. Require an ordinary read-only `historical_run_dir` containing
`allstr.arc`, the explicit `lasp.in`, and `best.arc`/`md.arc` only when their include
flags are true. Do not supply a LASP executable, MPI settings, or `lasp_version`.

Write normalized results only in a fresh attempt output. Describe the result as
structured collection and validation of existing LASP artifacts, never as a rerun or
independent scientific validation. Do not mutate the historical source tree.

## Local `lasp-ssw-execute`

Require a user-supplied ordinary executable LASP file, an explicit single-frame ARC
input structure, `lasp.in`, explicit `lasp_version`, and a mapping of safe auxiliary
destination basenames to files. Reserved LASP input/output names cannot be auxiliary
destinations. Require an explicit absolute non-symlink Python executable for the
wrapper.

MPI is optional locally. If used, require an ordinary executable path whose basename is
`mpirun` or `mpiexec` and a positive `mpi_processes`; do not accept either without the
other. The wrapper stages declared inputs into a fresh `raw-run`, invokes argv
with `shell: false`, and keeps an `INCOMPLETE.json` marker until normalization finishes.

## Scheduled `lasp-ssw-execute`

Use `backend: ssh-slurm`, a named `backend_profile`, project-scoped `input_structure`,
`lasp_input`, and declared `lasp_auxiliary_files`. Resources must contain exactly the
abstract fields `cpus`, `gpus`, `memory`, and `walltime`. Scheduled LASP uses MPI
execution semantics: `resources.cpus` is the MPI task/rank count. Do not set
`mpi_processes` in the project, and keep `output_subdir: lasp-ssw` as required by the
current scheduled adapter.

Keep all site-owned knowledge out of the project: SSH host/profile details, partition,
account/QoS, modules, LASP/Python/MPI executable paths, launcher flags, template root,
and work root. The named local site profile selects the site; the persistent remote
`lasp-ssw/run.sh` template supplies executable/module/launcher knowledge, while the
site's `slurm/mpi/cpu.sbatch` or `slurm/mpi/gpu.sbatch` supplies scheduler knowledge.

Treat one physical site as having one canonical template root and one canonical work
root across plugins, partitions, potentials, and validation runs. If `lasp-ssw/run.sh`
is absent, first inventory the canonical root and compare every target, then add only
the missing `lasp-ssw/` family below it. Never create sibling `templates-*` or work
roots, and never silently overwrite an existing template.

Follow the complete lifecycle: review the dry-run's scientific identity, important
inputs, resources, fresh attempt workspace, and expected outputs. Let MLIPFlow core stage, submit, and persist
scheduler identity; never call `sbatch` directly. Poll through MLIPFlow read-only
status/log interfaces. After scheduler `COMPLETED`, ordinary `advance` bounded-fetches
the run's declared outputs and runs pinned checker/collect.

Slurm `COMPLETED` alone is never scientific success. Report `OK` only after bounded
fetch and pinned checker/collect both succeed.

## ARC compatibility and completion

For execute operations, the current wrapper binds the SSW walk archive to canonical
`raw-run/allstr.arc`:

- If LASP produces only `allstr.arc`, keep it as canonical.
- If LASP produces `all.arc`, treat that as the walk archive. When no `allstr.arc`
  exists, copy `all.arc` to canonical `allstr.arc`.
- If both files exist and have identical content, retain them and use `allstr.arc` as
  canonical without rewriting it.
- If both exist and differ, rename the original `allstr.arc` to
  `allstr.native.arc`, then copy `all.arc` to canonical `allstr.arc`. Refuse to
  overwrite a pre-existing `allstr.native.arc`.

`lasp-run-metadata.json` records the source/canonical names, whether canonicalization
occurred, and, when preserved, the native copy's name and size. Its source records bind
canonical `allstr.arc`; the result manifest binds the metadata artifact. In scheduled mode, bounded fetch returns the
canonical `allstr.arc`, not `all.arc` or `allstr.native.arc`. Treat the native record as
preserved provenance, not as the selected scientific trajectory or an independently
fetched archive.

Do not trust a self-reported `OK` JSON. The pinned checker must reparse fetched canonical
`allstr.arc`, enforce frame/byte bounds and finite Energy records, recompute deterministic
structure IDs, historical order, energy acceptance, accepted-order stride, selected
counts, and selected-manifest identity. Scheduled checking must also verify every safe
member of `selected-structures.tar.gz` against its source frame. Reparse declared
`best.arc`/`md.arc` when included. Fail on changed inputs/outputs,
unsafe archive members, missing/oversized files, non-finite data, or lineage mismatch.

## Retry and failure handling

Classify failures before proposing a retry: repair site templates/configuration for a
site/environment failure; stop rather than guess for a scientific-input failure; fix a
general MLIPFlow defect with regression coverage rather than adding site-specific code.

Use `retry` to create a fresh local attempt while preserving the failed attempt and its
artifacts. Retry does not launch the scheduled job. Never delete, overwrite, or reuse the prior workspace to
simulate retry, and never retry automatically.
