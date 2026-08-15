---
name: ase-md
description: Supervise scheduled ASE molecular dynamics with explicit DeepMD, M3GNet/MatGL, CHGNet, or MACE model artifacts. Use when planning, submitting, restarting, or verifying MLIP-driven NVT Langevin or isotropic MTK NPT trajectories on an MLIPFlow ssh-slurm cluster profile.
---

# ASE molecular dynamics

Use the `ase-md` plugin as the deterministic implementation. The Skill decides and explains workflow inputs and approval boundaries; it does not implement an integrator or calculator itself.

## Version 0.3 boundary

Version 0.3 supports one single-temperature trajectory per workflow node on `ssh-slurm`:

- `nvt-langevin`: fixed-cell Langevin NVT.
- `npt-isotropic-mtk`: isotropic Martyna-Tobias-Klein NPT using ASE `IsotropicMTKNPT`.
- optional periodic checkpointing and exact restart across fresh scheduler attempts.

Do not silently substitute these contracts when the user requests NVE, anisotropic/full-cell NPT, replicas, a temperature sweep, automatic segment concatenation, diffusion fitting, conductivity, or Arrhenius analysis. Those remain separate workflow capabilities.

## Choose the calculator

Accept one explicit model family:

- `deepmd`: explicit DeepMD model file through the ASE DeepMD calculator.
- `m3gnet`: explicit local MatGL model directory through its ASE PES calculator.
- `chgnet`: explicit CHGNet model file through `CHGNetCalculator.from_file`; require `default_dtype: float32`.
- `mace`: explicit MACE model file through `MACECalculator`.

Never choose a network model name, pretrained alias, package cache entry, or remote Hub model on the user's behalf. Scheduled MD is network-free and requires an approved model content fingerprint.

For NPT, framework name alone is not sufficient evidence of stress capability. The compute-node runner must inspect the concrete loaded calculator's `implemented_properties` and successfully obtain a finite 3x3 ASE stress tensor before any fresh or restarted NPT step. If that probe fails, the job must fail before dynamics.

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
- total global `steps`
- `trajectory_interval`
- `thermo_interval`
- non-negative `seed`
- `device` (`cpu` or `cuda`)
- `default_dtype`
- `fix_com`
- exact structure SHA-256 and model fingerprint

`steps` is always the total trajectory target. A retry from step 400000 of a 1000000-step node runs only the remaining 600000 steps; never reinterpret `steps` as a per-attempt count.

`device: cuda` requires at least one scheduled GPU. Do not invent timestep, duration, temperature, pressure, damping, checkpoint cadence, or resource requests from the chemical system.

DeepMD's ASE calculator uses model-native precision; `default_dtype` is recorded for cross-framework workflow identity but must not be described as recasting a DeepMD model.

## NVT contract

For `ensemble: nvt-langevin` also require an explicit positive `friction_per_fs`.

`fix_com: true` uses ASE `FixCom` rather than deprecated Langevin `fixcm`. On a fresh attempt, the explicit seed initializes NumPy `PCG64`, Maxwell-Boltzmann velocities, and Langevin stochastic forces. On restart, do not reseed or reinitialize velocities: restore positions, momenta, cell and the saved PCG64 bit-generator state from the approved checkpoint.

Do not add NPT pressure or damping parameters to an NVT node.

## Isotropic NPT contract

For `ensemble: npt-isotropic-mtk` require all of:

- finite `pressure_gpa`;
- positive `thermostat_damping_fs`;
- positive `barostat_damping_fs`;
- `fix_com: false`;
- no `friction_per_fs`.

Do not infer damping times from the timestep. They are scientific inputs and must be explicit in MLIPFlow.

The NPT implementation is deliberately isotropic: only volume changes, preserving the initial cell shape. Version 0.3 pins thermostat/barostat chain lengths to 3/3 and chain integration substeps to 1/1; expose those values in approval/provenance rather than pretending they are user-selected.

NPT requires a full-rank 3D periodic cell and no ASE constraints. Reject a molecular/nonperiodic or constrained structure rather than silently converting it. A fresh NPT seed controls initial Maxwell-Boltzmann velocities only.

Exact NPT restart restores atomic state plus MTK particle/cell state, thermostat-chain coordinates/momenta and barostat-chain coordinates/momenta. Because these are currently ASE implementation-private fields, require the checkpoint's exact ASE version to match the compute environment. Refuse a version mismatch rather than approximating a restart from the saved geometry.

NPT thermodynamics must additionally record finite `pressure_GPa`, positive `volume_A3`, and positive `cell_a_A`, `cell_b_A`, `cell_c_A` on the approved thermo schedule.

## Checkpoint and restart contract

Checkpointing is opt-in. Accept:

- `checkpoint_interval`: positive global MD-step interval.
- `restart_policy`: `disabled` or `auto-from-previous-attempt`.

`auto-from-previous-attempt` requires `checkpoint_interval`. The runner atomically replaces `output/md-checkpoint.json` using a temporary file, flush/fsync, then rename, so an interrupted write should leave the previous complete checkpoint rather than a partially overwritten one.

A checkpoint is strict JSON, never pickle. It binds model/structure identity, ensemble physics, total target steps, completed global step, ASE version, atomic numbers/order, positions, momenta, cell, masses and PBC. NVT also binds Langevin implementation version and PCG64 RNG state. NPT also binds the complete MTK extended state.

Do not let the Agent choose an arbitrary checkpoint path. A retry may consume only `md-checkpoint.json` already salvaged into the immediately previous local attempt by MLIPFlow core. The retry stages that file as a new fingerprinted scientific input and shows its SHA-256 and completed step in the new approval plan.

Do not auto-resume a normal scientific `FAIL` whose scheduler state was `COMPLETED`. Automatic restart is for scheduler terminal interruption such as `TIMEOUT`, `PREEMPTED`, node/OOM/deadline failures, or an observed scheduler cancellation with an approved salvaged checkpoint. If the checkpoint is absent, identity-mismatched, already complete, or belongs to a non-eligible attempt, block the retry.

## Terminal failure salvage

A scheduler terminal state is not permission to download arbitrary remote files. When checkpointing is enabled, the adapter declares a small `failure_salvage` subset selected only from the already approved normal fetch allowlist.

For a terminal scheduler interruption:

1. Run `advance --dry-run` while the interrupted attempt is still the latest attempt.
2. Show the exact salvage inventory, including which approved checkpoint/partial-segment files exist, their sizes and SHA-256 values, and any oversized items.
3. Require the matching `advance` approval before fetching anything.
4. Core fetches only the approved salvage subset and then leaves the attempt `FAIL` or `STOPPED`; salvage is not scientific success.
5. Only after the checkpoint exists locally may `retry --dry-run` / `retry` create a fresh attempt.
6. The next run plan must show `restart_from_attempt`, checkpoint SHA-256, `segment_start_step`, remaining steps, and the new per-attempt output schedules before submission approval.

Do not describe `retry` itself as fetching remote state. It only creates a fresh attempt; the new adapter plan stages the already salvaged local checkpoint.

An explicit MLIPFlow `stop` that transitions state immediately may not provide the same salvage opportunity as observing a scheduler-side terminal cancellation. Do not promise a resumable manual stop unless a checkpoint has first been safely salvaged through the approved observation flow.

## Segment semantics

Every attempt has a fresh remote workspace and writes its own `trajectory.traj`, `trajectory-index.json`, and `thermo.csv`. Restarted segments use global MD step/time numbers beginning at the checkpoint step. Version 0.3 does not automatically concatenate these segments.

Do not discard prior segments after restart. Keep each attempt's immutable artifacts/provenance so a later reviewed segment-stitching or transport stage can join them while checking the checkpoint boundary and global step continuity.

## Cluster boundary

Require `backend: ssh-slurm`, a named `backend_profile`, and exactly the abstract resources `cpus`, `gpus`, `memory`, and `walltime`.

Never put an SSH host, account, partition, QoS, module, conda path, Python path, absolute model path, template root, work root, submit script, or `remote_cwd` in the workflow node. Those belong to the local site profile and persistent remote templates.

The selected template family is one of:

- `ase-md-deepmd`
- `ase-md-m3gnet`
- `ase-md-chgnet`
- `ase-md-mace`

Each site template may activate a separate framework environment while keeping the same MLIPFlow scheduler/restart contract.

ASE MD has `execution_model: single-python`. `resources.cpus` is the CPU/thread
budget for one Python process, and the site Slurm template must therefore use
`--ntasks=1` with `--cpus-per-task={{CPUS}}`. Do not expose ASE `CPUS` as the
Slurm task count; doing so makes Lightning-backed calculators look like an
unconfigured distributed job. MPI execution models such as VASP/LAMMPS retain
their separate rank-count semantics.

## Approval and completion

Before submission, show calculator/model identity, structure identity, ensemble, temperature, timestep, total target duration, current segment start/remaining steps, frame/thermo record counts, checkpoint policy, device, resources, template family, and bounded fetch allowlist. For NVT also show friction. For NPT also show target pressure, both damping times, stress requirement, isotropic cell mode, no-constraints requirement, and pinned MTK chain configuration.

A Slurm `COMPLETED` state is not scientific success. After completion, use the normal second `advance --dry-run` / approval boundary. The pinned checker must verify:

- completed global steps equal the original requested total;
- model, structure, and any restart-checkpoint fingerprints match approved identities;
- trajectory-index and thermo schedules start at the approved segment start and reach the original final step;
- thermodynamic values are finite and NPT pressure/cell metrics satisfy the NPT contract;
- trajectory, index, thermo table, final extxyz, checkpoint when enabled, and reports match recorded sizes/SHA-256 values;
- calculator and ASE versions are recorded.

Treat missing, oversized, changed, non-finite, stress-incompatible, checkpoint-incompatible, or identity-mismatched output as `FAIL`, not a warning.

## Scientific interpretation

The `ase-md` plugin produces trajectory segments; it does not by itself establish equilibration, diffusion, ionic conductivity, phase stability, or model validity. A successful restart proves continuity of the declared integrator state under the pinned implementation contract, not physical convergence. Do not report transport or convergence conclusions unless a separate reviewed analysis stage supports them.
