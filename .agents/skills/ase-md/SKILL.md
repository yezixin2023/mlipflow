---
name: ase-md
description: Supervise scheduled ASE molecular dynamics with explicit DeepMD, M3GNet/MatGL, CHGNet, or MACE model artifacts. Use when planning, submitting, or verifying MLIP-driven NVT Langevin or isotropic MTK NPT trajectories on an MLIPFlow ssh-slurm cluster profile.
---

# ASE molecular dynamics

Use the `ase-md` plugin as the deterministic implementation. The Skill decides and explains workflow inputs; it does not implement an integrator or calculator itself.

## Version 0.2 boundary

Version 0.2 supports exactly one fresh, single-temperature trajectory per workflow node on `ssh-slurm`:

- `nvt-langevin`: fixed-cell Langevin NVT.
- `npt-isotropic-mtk`: isotropic Martyna-Tobias-Klein NPT using ASE `IsotropicMTKNPT`.

Do not silently substitute either contract when the user requests NVE, anisotropic/full-cell NPT, restart/resume, replicas, a temperature sweep, diffusion fitting, conductivity, or Arrhenius analysis. Those remain separate workflow capabilities.

## Choose the calculator

Accept one explicit model family:

- `deepmd`: explicit DeepMD model file through the ASE DeepMD calculator.
- `m3gnet`: explicit local MatGL model directory through its ASE PES calculator.
- `chgnet`: explicit CHGNet model file through `CHGNetCalculator.from_file`; require `default_dtype: float32`.
- `mace`: explicit MACE model file through `MACECalculator`.

Never choose a network model name, pretrained alias, package cache entry, or remote Hub model on the user's behalf. Scheduled MD is network-free and requires an approved model content fingerprint.

For NPT, framework name alone is not sufficient evidence of stress capability. The compute-node runner must inspect the concrete loaded calculator's `implemented_properties` and successfully obtain a finite 3x3 ASE stress tensor before any NPT step. If that probe fails, the job must fail before dynamics.

## Bind the model without cluster paths

Require `inputs.model_reference` to point to a small project JSON manifest containing:

- `schema_version: 1`
- a logical `model_id`
- a safe `relative_path` below the site-owned model registry
- `kind: file` for DeepMD/CHGNet/MACE or `kind: directory` for M3GNet/MatGL
- the approved `sha256:<64 hex>` content fingerprint

The portable project must never contain the cluster's absolute model path. The site-owned `ase-md-<calculator>/run.sh` supplies `MODEL_ROOT`. The compute-node resolver recomputes the model fingerprint before and after inference.

## Require explicit common MD physics

For either ensemble, require the user/workflow to declare:

- target `temperature_k`
- `timestep_fs`
- total `steps`
- `trajectory_interval`
- `thermo_interval`
- non-negative `seed`
- `device` (`cpu` or `cuda`)
- `default_dtype`
- `fix_com`
- exact structure SHA-256 and model fingerprint

`device: cuda` requires at least one scheduled GPU. Do not invent a timestep, duration, temperature, or resource request from the chemical system.

DeepMD's ASE calculator uses model-native precision; `default_dtype` is recorded for cross-framework workflow identity but must not be described as recasting a DeepMD model.

## NVT contract

For `ensemble: nvt-langevin` also require an explicit positive `friction_per_fs`.

`fix_com: true` uses ASE `FixCom` rather than deprecated Langevin `fixcm`. The seed controls both Maxwell-Boltzmann initialization and Langevin stochastic forces. Explain that GPU/framework kernels still are not promised bitwise deterministic.

Do not add NPT pressure or damping parameters to an NVT node.

## Isotropic NPT contract

For `ensemble: npt-isotropic-mtk` require all of:

- finite `pressure_gpa` (zero and negative pressure are not silently rejected);
- positive `thermostat_damping_fs`;
- positive `barostat_damping_fs`;
- `fix_com: false`;
- no `friction_per_fs`.

Do not infer the damping times from the timestep. ASE documentation gives typical scales, but they remain scientific inputs and must be explicit in MLIPFlow.

The NPT implementation is deliberately isotropic: only volume changes, preserving the initial cell shape. Version 0.2 pins thermostat/barostat chain lengths to 3/3 and chain integration substeps to 1/1; expose those values in approval/provenance rather than pretending they are user-selected.

NPT requires a full-rank 3D periodic cell and no ASE constraints. Reject a molecular/nonperiodic or constrained structure rather than silently converting it. The seed controls initial Maxwell-Boltzmann velocities only; subsequent MTK integration has no stochastic thermostat RNG.

NPT thermodynamics must additionally record finite `pressure_GPa`, positive `volume_A3`, and positive `cell_a_A`, `cell_b_A`, `cell_c_A` on the approved thermo schedule.

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

Before submission, show calculator/model identity, structure identity, ensemble, temperature, timestep, simulated duration, frame/thermo record counts, device, resources, template family, and bounded fetch allowlist. For NVT also show friction. For NPT also show target pressure, both damping times, stress requirement, isotropic cell mode, no-constraints requirement, and pinned MTK chain configuration.

A Slurm `COMPLETED` state is not scientific success. After completion, use the normal second `advance --dry-run` / approval boundary. The pinned checker must verify:

- completed steps equal requested steps;
- model and structure fingerprints match the approved identities;
- `trajectory-index.json` contains exactly step 0, every approved interval, and the final step;
- `thermo.csv` contains the matching step/time schedule and only finite values;
- NPT pressure and cell metrics satisfy the NPT contract;
- trajectory, index, thermo table, final extxyz, and reports match recorded sizes and SHA-256 values;
- calculator and ASE versions are recorded.

Treat missing, oversized, changed, non-finite, stress-incompatible, or identity-mismatched output as `FAIL`, not a warning.

## Scientific interpretation

The `ase-md` plugin produces trajectories; it does not by itself establish equilibration, diffusion, ionic conductivity, phase stability, or model validity. A stable pressure trace is not proof of a converged NPT ensemble. Do not report those conclusions unless a separate reviewed analysis stage supports them.
