---
name: pes-sampling
description: Supervise MLIPFlow potential-energy-surface sampling with local MAML/DIRECT representative-structure selection, deterministic ASE-to-LASP input conversion and DIRECT+LASP structure merge, local LASP/SSW execution or historical ARC normalization, and reviewed SSH-SLURM LASP execution. Use when planning, running, retrying, or verifying the pes-sampling plugin's direct-select, lasp-input-prepare, merge-structures, lasp-ssw-execute, or lasp-ssw-normalize-replay operation.
---

# Potential-energy-surface sampling

Use `mlipflow/plugins/pes_sampling` through MLIPFlow. The Skill chooses the sampling intent and
guards scientific claims; selection, conversion, LASP execution, normalization, and
checking belong to the plugin.

## Route the request

- Select representative structures from existing ensembles with MAML/DIRECT:
  `direct-select`.
- Convert one explicit ASE-readable periodic frame to reviewed LASP input:
  `lasp-input-prepare`.
- Combine verified DIRECT and LASP selections into one DFT-ready structure manifest:
  `merge-structures`.
- Run fresh LASP/SSW locally or through the supported scheduler path:
  `lasp-ssw-execute`.
- Normalize an existing LASP archive without running LASP:
  `lasp-ssw-normalize-replay`.

Do not describe DIRECT as LASP, replay as fresh LASP, or any operation as DFT, MD,
training, or energetic ranking.

## Inputs and example

Reuse collected structure ensembles, DIRECT selections and LASP archives.
DIRECT needs an explicit ensemble and clustering settings; LASP execution needs
reviewed LASP input, potential/auxiliary files, selection policy and output bounds.
Use `examples/lasp_random_walk/CLUSTER.md` for a scheduled node and the adjacent
`lasp.in` as the editable input example. For an existing archive choose
`lasp-ssw-normalize-replay`; for a downstream structure set use `merge-structures`
with explicit matching tolerances. The CLI reports accepted input bindings.

## Scientific judgment

Use DIRECT only when representative/configurational selection is necessary. Require
the actual clustering and input-scope choices; do not invent them. The current bundled
MAML path does not expose control of its selection seed, so a recorded seed must not be
claimed to make the selection reproducible.

For LASP preparation, review the selected source frame and any scientific cell-size
bound. For merge, require scientifically chosen structure-matching tolerances and
scope. The merge normalizes and deduplicates; it does not evaluate energies or decide
which source is scientifically better.

For fresh LASP, review the literal scientific input, potential, auxiliary artifacts,
selection policy, output bounds, and resources. Do not infer undocumented LASP
parameters, seeds, potentials, executables, or launch details. If LASP may invoke
VASP/DFT, disclose that scope and obtain the separate explicit DFT intent required by
repository policy. Never expose or collect POTCAR content.

Historical normalization must preserve source evidence and ordering. Describe it as a
structured collection of existing results, not a rerun, numerical parity, or
independent validation.

## Run and read the result

Preview the actual task and effective `approval_required` value. Reuse explicit
user authorization for this task and scale; use `--approve` when the plan requires it.

```bash
mlipflow --project PROJECT init
mlipflow --project PROJECT --format json run NODE --dry-run
mlipflow --project PROJECT --format json run NODE --approve
```

For a plan without an approval requirement, `run NODE` suffices. For scheduled
execution, use `mlipflow --project PROJECT --format json advance` when the job
progresses; `mlipflow --project PROJECT json NODE` reads the saved result.

Read `state`, `metrics`, `artifacts[].role` and the full `artifacts[].path` from JSON.
Require final plugin `OK` before using outputs. On failure start with `reason`,
`check.diagnostics`, `manifest_path` and `logs`; `mlipflow --project PROJECT logs NODE`
shows saved stdout/stderr. Examples are in the checkout's `examples/` or the
installed environment's `share/mlipflow/examples/`.

Report the selected operation, evidence mode, important scientific choices, collected
structures and lineage, whether LASP or DFT actually ran, and limitations. `OK`
establishes the declared selection/execution and artifact contract, not SSW numerical
parity, global PES coverage, DFT suitability, or downstream transport convergence.
