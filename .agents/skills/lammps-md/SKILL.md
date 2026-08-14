---
name: lammps-md
description: Prepare, submit, and verify portable LAMMPS MLIP workflows for explicit LAMMPS-ready DeepMD, MACE, or MatGL/M3GNet models. Use for CPU/GPU LAMMPS input generation or reviewed ssh-slurm NVT/NPT execution.
---

# LAMMPS MLIP molecular dynamics

Use the `lammps-md` plugin as the implementation source of truth. Version 0.2 has two deliberately separate operations:

- `lammps-prepare`: local input generation and verification only;
- `execute`: separately approved `ssh-slurm` execution of one already prepared CPU or GPU target.

Never treat preparation success as evidence that LAMMPS ran.

## Decide whether the model is LAMMPS-ready

Accept only an explicit model reference with a content fingerprint, supported element list, site-root-relative path, and framework-specific exported artifact format.

Supported contracts:

- `deepmd` + `artifact_format: deepmd-lammps-model`;
- `mace` + `artifact_format: mace-lammps-torchscript`;
- `m3gnet` + `artifact_format: matgl-lammps-torchscript`.

Do not treat a raw MACE or MatGL training checkpoint as a LAMMPS model. Require the framework's reviewed LAMMPS export step first.

Do not invent a CHGNet pair style. Version 0.2 still blocks CHGNet because no native export/pair-style bridge is pinned by this repository; use `$ase-md` unless a separate bridge is reviewed.

## Preparation contract

Require an explicit `lammps_config` with ensemble `nvt` or `npt-isotropic`, CPU/GPU targets, type map, temperature, timestep, total steps, thermo/dump intervals, positive velocity seed, thermostat damping, and NPT pressure/barostat damping when applicable.

Do not invent physical inputs from the composition. The generator converts workflow units to LAMMPS `metal` units: fs to ps and GPa to bar.

Version 0.2 still requires one full-rank three-dimensional periodic atomic structure. Do not silently generate molecular topology, charges, bonded terms, hybrid potentials, long-range electrostatics, or nonperiodic boundaries.

The execution-ready preparation wrapper must produce `preparation_contract: lammps-md-input-v2`. Every requested deck must contain `${MODEL_FILE}` and an exact `MLIPFLOW_LAMMPS_COMPLETED step=<steps>` print command placed after `final.data` and `final.restart` writes. The manifest must record that exact marker and every generated file SHA-256/size.

## CPU/GPU framework routing

DeepMD:

- CPU and GPU decks both use `pair_style deepmd`.
- Do not add a Kokkos suffix just because GPU was requested.
- Scheduled GPU execution may request more than one GPU, but the site launcher must map at most one GPU to each MPI rank.

MACE:

- Require the reviewed ML-MACE TorchScript artifact.
- CPU deck uses `pair_style mace`.
- GPU deck uses `pair_style mace no_domain_decomposition` with `-k on g 1 -sf kk` launcher metadata.
- Version 0.2 requires exactly one scheduled GPU for this path.
- Do not silently substitute ML-IAP; it is a separate future interface contract.

MatGL/M3GNet:

- Require a file exported for MatGL LAMMPS.
- CPU deck uses `pair_style matgl`.
- GPU deck uses `pair_style matgl/kk` with Kokkos launcher metadata.
- Version 0.2 requires exactly one scheduled GPU and treats this path as single-rank/single-GPU.

## Keep execution portable

Generated decks must reference `${MODEL_FILE}` and must not contain `model_reference.relative_path` or an absolute cluster path.

For operation `execute`, require exactly:

- `backend: ssh-slurm` and a named `backend_profile`;
- `inputs.lammps_input_manifest` pointing to a project-scoped `lammps-md-input-v2` manifest;
- `parameters.operation: execute`;
- explicit `parameters.target: cpu|gpu`;
- explicit full `parameters.input_manifest_fingerprint`;
- resources containing exactly `cpus`, `gpus`, `memory`, and `walltime`.

CPU execution requires `gpus: 0`. GPU execution requires at least one GPU; MACE/MatGL are restricted to exactly one in v0.2.

The execute adapter must re-read the prepared manifest, recompute its SHA, recompute `structure.data` and selected deck SHA/size, verify the target launcher/model contract, and reject any changed bundle before submission.

## Site boundary

Never put SSH host, account, partition, QoS, module/conda setup, LAMMPS path, MPI/srun command, CUDA architecture, MODEL_ROOT, template root, or work root in the project node.

The selected template family is one of:

- `lammps-deepmd-cpu`
- `lammps-deepmd-gpu`
- `lammps-mace-cpu`
- `lammps-mace-gpu`
- `lammps-m3gnet-cpu`
- `lammps-m3gnet-gpu`

Each site's `run.sh` owns `PYTHON_BIN`, `LAMMPS_BIN`, `MODEL_ROOT`, and a JSON launcher argv. The bundled remote runner parses launcher JSON and launches with `shell=False`.

The compute-node runner must resolve the prepared model relative path only below MODEL_ROOT, verify its SHA before execution, inject the resolved path through `-var MODEL_FILE`, and verify the model SHA again after execution.

## Approval lifecycle

The execute approval must show framework, target, model id/fingerprint/artifact format, prepared-manifest fingerprint, ensemble, temperature, timestep, steps/simulated duration, type map, required package family, resources, and selected template family.

Scheduler `COMPLETED` is not scientific success. After completion, use the normal second `advance --dry-run` approval. The core must inventory and bounded-fetch only the declared outputs before the plugin checker runs.

A successful execute attempt requires:

- `lammps-execution-result.json` and `cluster-run-report.json` with matching approved identities;
- non-empty bounded `trajectory.lammpstrj`, `final.data`, `final.restart`, `lammps.log`, and `lammps.screen.log`;
- matching size/SHA-256 records for every required execution artifact;
- a recorded LAMMPS version;
- `steps_completed == approved steps`;
- the exact approved completion marker in the fetched `lammps.log`.

A zero scheduler exit code alone is insufficient. The marker is intentionally printed only after the final data/restart writes.

If LAMMPS or the scheduler fails, failure salvage may fetch only the already approved diagnostic subset: cluster report and logs. Do not promote a partial trajectory, final data file, or binary restart to scientific success.

## Restart boundary

Version 0.2 collects `final.restart` but does not resume it. Do not reuse the ASE-MD JSON checkpoint logic for LAMMPS binary restarts. LAMMPS restart files are tied to the LAMMPS executable/platform contract and some fixes/commands must be reissued, so restart support must be introduced as a separate reviewed contract.

## Scientific interpretation

`lammps-md` generates/runs trajectories. It does not establish equilibration, diffusion, ionic conductivity, phase stability, or model validity by itself. Keep transport analysis and multi-temperature orchestration separate until those stages are explicitly connected.
