---
name: lammps-md
description: Prepare, submit, verify, and restart portable LAMMPS MLIP workflows for explicit LAMMPS-ready DeepMD, MACE, or MatGL/M3GNet models. Use for CPU/GPU NVT/NPT input generation, reviewed ssh-slurm execution, periodic binary restart salvage, or walltime/preemption continuation.
---

# LAMMPS MLIP molecular dynamics

Use `mlipipe/plugins/lammps_md` through MLIPipe. Keep input generation and numerical execution
as separate operations; the Skill reviews MD intent while the plugin owns deck
generation, execution, restart, checking, and collection.

## Route the request

- Generate and verify portable LAMMPS input artifacts: `lammps-prepare`.
- Execute one reviewed prepared target through the supported scheduler lifecycle:
  `execute`.

Preparation success never means LAMMPS ran. Require an explicit LAMMPS-ready model
reference and a supported interface. Do not pass raw training checkpoints as exported
LAMMPS models or invent a CHGNet pair style; route CHGNet MD to `$ase-md` unless an
explicit supported bridge exists.

## Inputs and example

Reuse the existing prepared-input manifest when it matches. Preparation requires a
periodic structure, LAMMPS-ready model reference and simulation config; execution
requires the prepared manifest and explicit target. The existing NVT/NPT configs,
model reference examples and `USAGE.md` are in `examples/lammps_mlip_inputs/`.
Reuse collected trajectory segments directly in `$ionic-transport`.

## Scientific judgment

Require an explicit periodic structure, ensemble, temperature, timestep, total
duration/steps, output cadence, seed, thermostat controls, NPT pressure/barostat
controls when applicable, target interface/device, and abstract resources. Do not
infer physical parameters, model compatibility, or GPU/rank choices from composition
or framework name.

Use periodic binary restart only when the Adapter validates the immediately preceding
interrupted attempt, salvaged candidates, and runtime compatibility. Do not parse or
select binary restarts in the Skill, silently fresh-start, or call a scientifically
failed completed run resumable. Describe a valid resume as state-continuous under a
compatible runtime, not universally bitwise identical.

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

Report preparation versus actual execution, model/interface, physical controls,
restart status, collected trajectory segments, and checker result. `OK` does not prove
equilibration, diffusion, ionic conductivity, phase stability, sufficient sampling,
model validity, or transport convergence.
