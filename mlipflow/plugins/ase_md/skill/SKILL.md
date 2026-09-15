---
name: ase-md
description: Supervise scheduled ASE molecular dynamics with explicit DeepMD, M3GNet/MatGL, CHGNet, or MACE model artifacts. Use when planning, submitting, restarting, or verifying MLIP-driven NVT Langevin or isotropic MTK NPT trajectories on an MLIPFlow ssh-slurm cluster profile.
---

# ASE molecular dynamics

Use `mlipflow/plugins/ase_md` through MLIPFlow for the `run` operation. The Skill chooses and
reviews MD physics; calculator construction, integration, restart, checking, and
collection belong to the plugin and ASE.

## Route the request

Use `nvt-langevin` for fixed-cell Langevin NVT and `npt-isotropic-mtk` for the supported
isotropic MTK NPT contract. Do not silently substitute either for NVE, anisotropic or
full-cell NPT, replica studies, temperature sweeps, transport analysis, or another MD
engine.

Require an explicit compatible local model artifact for DeepMD, M3GNet/MatGL, CHGNet,
or MACE. Do not select remote model aliases, cached pretrained names, or a model family
by reputation. NPT additionally needs evidence that the concrete calculator supplies
usable stress.

## Inputs and example

Reuse a compatible collected model and periodic structure. Set `inputs.structure`
and `inputs.model_reference`; preserve the intended ensemble, temperature,
timestep, total steps, output cadence, seed and resources from the project.
`examples/ase_md_cluster/project.yaml` is a complete NVT task; its `USAGE.md` explains
which fields to replace, dependencies, units and output paths. The existing README
contains the NPT and interrupted-attempt variants. If a suitable trajectory already
exists, pass it to `$ionic-transport` and reuse compatible collected segments.

## Scientific judgment

Require explicit temperature, timestep, total duration/steps, output cadence, seed,
ensemble controls, model/structure identity, device/precision intent, cell-size choices,
and abstract resources. NPT pressure and damping choices and NVT friction are
scientific inputs. Do not infer them from composition or typical practice.

Use checkpoint restart only when the prior attempt and salvaged checkpoint satisfy the
plugin's compatibility and interruption rules. Do not promise that a manual stop is
resumable, reinterpret total steps as per-attempt steps, or treat salvage as scientific
success. A successful restart establishes state continuity under the recorded runtime,
not physical convergence.

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

Report the actual ensemble and model, physical controls, segment/total scale, restart
status, collected trajectory artifacts, and checker result. `OK` does not establish
equilibration, diffusion, ionic conductivity, phase stability, sufficient sampling,
model validity, or transport convergence.
