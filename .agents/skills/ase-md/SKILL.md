---
name: ase-md
description: Supervise scheduled ASE molecular dynamics with explicit DeepMD, M3GNet/MatGL, CHGNet, or MACE model artifacts. Use when planning, submitting, or verifying MLIP-driven NVT trajectories on an MLIPFlow ssh-slurm cluster profile.
---

# ASE molecular dynamics

Use the `ase-md` plugin as the deterministic implementation. The Skill decides and explains workflow inputs; it does not implement an integrator or calculator itself.

## Version 0.1 boundary

Version 0.1 supports exactly one fresh, single-temperature `nvt-langevin` trajectory per workflow node on `ssh-slurm`.

Do not silently substitute this contract when the user requests NPT, NVE, restart/resume, replicas, a temperature sweep, diffusion fitting, conductivity, or Arrhenius analysis. Those are separate workflow capabilities. Keep the MD trajectory stage independent until the corresponding contract exists.

## Choose the calculator

Accept one explicit model family:

- `deepmd`: an explicit DeepMD model file, loaded through the ASE DeepMD calculator.
- `m3gnet`: an explicit local MatGL model directory, loaded as a MatGL potential and attached through its ASE PES calculator.
- `chgnet`: an explicit CHGNet model file, loaded through `CHGNetCalculator.from_file`; require `default_dtype: float32`.
- `mace`: an explicit MACE model file, loaded through `MACECalculator`.

Never choose a network model name, pretrained alias, package cache entry, or remote Hub model on the user's behalf. Scheduled MD is network-free and requires an approved model content fingerprint.

## Bind the model without cluster paths

Require `inputs.model_reference` to point to a small project JSON manifest containing:

- `schema_version: 1`
- a logical `model_id`
- a safe `relative_path` below the site-owned model registry
- `kind: file` for DeepMD/CHGNet/MACE or `kind: directory` for M3GNet/MatGL
- the approved `sha256:<64 hex>` content fingerprint

The portable project must never contain the cluster's absolute model path. The site-owned `ase-md-<calculator>/run.sh` supplies `MODEL_ROOT`. The compute-node resolver recomputes the model fingerprint before and after inference.

## Require explicit MD physics

Do not infer scientific MD settings. Require the user/workflow to declare:

- target `temperature_k`
- `timestep_fs`
- total `steps`
- `trajectory_interval`
- `thermo_interval`
- non-negative stochastic `seed`
- `friction_per_fs`
- `fix_com`
- `device` (`cpu` or `cuda`)
- `default_dtype`
- the exact structure SHA-256 and model fingerprint

`device: cuda` requires at least one scheduled GPU. Do not invent a timestep, thermostat friction, duration, or temperature from the chemical system.

The runner seeds both Maxwell-Boltzmann velocity initialization and the ASE Langevin RNG. Explain that framework/GPU kernels are not promised to be bitwise deterministic.

## Cluster boundary

Require `backend: ssh-slurm`, a named `backend_profile`, and exactly the abstract resources `cpus`, `gpus`, `memory`, and `walltime`.

Never put an SSH host, account, partition, QoS, module, conda path, Python path, absolute model path, template root, work root, submit script, or `remote_cwd` in the workflow node. Those belong to the local site profile and persistent remote templates.

The selected template family is one of:

- `ase-md-deepmd`
- `ase-md-m3gnet`
- `ase-md-chgnet`
- `ase-md-mace`

Each site template may activate a separate framework environment while keeping the same MLIPFlow scheduler contract.

## Approval and completion

Before submission, show calculator/model identity, structure identity, NVT settings, simulated duration, frame/thermo record counts, device, resources, template family, and bounded fetch allowlist. Submit only after the matching normal MLIPFlow approval.

A Slurm `COMPLETED` state is not scientific success. After completion, use the normal second `advance --dry-run` / approval boundary. The pinned checker must verify:

- completed steps equal requested steps;
- model and structure fingerprints match the approved identities;
- `trajectory-index.json` contains exactly step 0, every approved interval, and the final step;
- `thermo.csv` contains the matching step/time schedule and only finite values;
- `trajectory.traj`, index, thermo table, final extxyz, and reports match their recorded sizes and SHA-256 values;
- calculator and ASE versions are recorded.

Treat missing, oversized, changed, non-finite, or identity-mismatched output as `FAIL`, not a warning.

## Scientific interpretation

The `ase-md` plugin produces trajectories; it does not by itself establish equilibration, diffusion, ionic conductivity, phase stability, or model validity. Do not report those quantities unless a separate reviewed analysis stage consumes the trajectory. Do not call a successful contract test or short trajectory numerical parity with historical calculations.
