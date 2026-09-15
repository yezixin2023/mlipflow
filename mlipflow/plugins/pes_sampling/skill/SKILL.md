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

## Artifact first

Reuse accepted structure ensembles, DIRECT selections, prepared LASP inputs, LASP
archives, and merged structure manifests. Pass collected artifacts directly between
operations and to `$dft-labeling`; do not copy, rename, or rebuild them manually. Use
replay for suitable historical archives rather than rerunning LASP to recreate evidence.

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

## Execute and interpret

Use `mlipflow inspect` and the dry-run; let Adapter validation supply detailed
parameter, staging, and output rules. Follow the plan's effective
`approval_required` value instead of maintaining operation/backend approval rules in
this Skill.

Never invoke DIRECT, LASP, VASP, plugin runners, or schedulers outside MLIPFlow; never
guess site-owned cluster configuration or bypass Adapter
`validate/plan/execute/check/collect`. Final plugin `OK`, not scheduler `COMPLETED` or
process exit zero, is required. Diagnose failure before a fresh retry and preserve
earlier attempts.

Report the selected operation, evidence mode, important scientific choices, collected
structures and lineage, whether LASP or DFT actually ran, and limitations. `OK`
establishes the declared selection/execution and artifact contract, not SSW numerical
parity, global PES coverage, DFT suitability, or downstream transport convergence.
