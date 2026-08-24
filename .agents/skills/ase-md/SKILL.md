---
name: ase-md
description: Supervise scheduled ASE molecular dynamics with explicit DeepMD, M3GNet/MatGL, CHGNet, or MACE model artifacts. Use when planning, submitting, restarting, or verifying MLIP-driven NVT Langevin or isotropic MTK NPT trajectories on an MLIPFlow ssh-slurm cluster profile.
---

# ASE molecular dynamics

Use `plugins/ase-md` through MLIPFlow for the `run` operation. The Skill chooses and
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

## Artifact first

Reuse matching final `OK` model and structure references. If a suitable verified
trajectory already exists, route it directly to `$ionic-transport` instead of rerunning
MD. Preserve all collected restart segments; downstream analysis joins compatible
segments without user-side copying, renaming, or concatenation.

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

## Execute and interpret

Use `mlipflow inspect` and the dry-run to review the effective ensemble, segment,
checkpoint source, model/structure bindings, trajectory scale, backend/resources, and
expected artifacts. Follow the plan's effective `approval_required` value rather than
duplicating approval rules here.

Never launch ASE, a framework runner, or a scheduler directly, guess site-owned cluster
details, or bypass Adapter `validate/plan/execute/check/collect`. Scheduler
`COMPLETED` and process exit zero are insufficient; success requires final plugin
`OK`. Let MLIPFlow perform bounded salvage and fresh-attempt retry.

Report the actual ensemble and model, physical controls, segment/total scale, restart
status, collected trajectory artifacts, and checker result. `OK` does not establish
equilibration, diffusion, ionic conductivity, phase stability, sufficient sampling,
model validity, or transport convergence.
