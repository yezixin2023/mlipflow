---
name: lammps-md
description: Prepare, submit, verify, and restart portable LAMMPS MLIP workflows for explicit LAMMPS-ready DeepMD, MACE, or MatGL/M3GNet models. Use for CPU/GPU NVT/NPT input generation, reviewed ssh-slurm execution, periodic binary restart salvage, or walltime/preemption continuation.
---

# LAMMPS MLIP molecular dynamics

Use `plugins/lammps-md` through MLIPFlow. Keep input generation and numerical execution
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

## Artifact first

Reuse accepted prepared-input manifests, model references, checkpoints, and final `OK`
trajectories. Do not regenerate a deck when a matching verified preparation exists. If
a suitable trajectory already exists, pass all collected segments directly to
`$ionic-transport` rather than rerunning MD or asking the user to manipulate files.

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

## Execute and interpret

Use `mlipflow inspect` and the dry-run to review the prepared manifest, model/interface,
physics, total and remaining scale, restart source, backend/resources, and expected
artifacts. Follow the effective `approval_required` value rather than copying
operation/backend approval rules into this Skill.

Never invoke LAMMPS, plugin runners, or schedulers outside MLIPFlow, guess site-owned
cluster details, or bypass Adapter `validate/plan/execute/check/collect`. Scheduler
`COMPLETED` and process exit zero are insufficient; require final plugin `OK`. Let
MLIPFlow own bounded salvage and fresh retry attempts.

Report preparation versus actual execution, model/interface, physical controls,
restart status, collected trajectory segments, and checker result. `OK` does not prove
equilibration, diffusion, ionic conductivity, phase stability, sufficient sampling,
model validity, or transport convergence.
