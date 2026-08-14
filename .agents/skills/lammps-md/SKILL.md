---
name: lammps-md
description: Prepare, submit, verify, and restart portable LAMMPS MLIP workflows for explicit LAMMPS-ready DeepMD, MACE, or MatGL/M3GNet models. Use for CPU/GPU NVT/NPT input generation, reviewed ssh-slurm execution, periodic binary restart salvage, or walltime/preemption continuation.
---

# LAMMPS MLIP molecular dynamics

Use the `lammps-md` plugin as the implementation source of truth. Version 0.3 has two deliberately separate operations plus an optional restart protocol:

- `lammps-prepare`: local input generation and verification only;
- `execute`: separately approved `ssh-slurm` execution of one already prepared CPU or GPU target;
- execute may opt into periodic binary checkpointing and `auto-from-previous-attempt` restart after scheduler interruption.

Never treat preparation success as evidence that LAMMPS ran. Never treat scheduler `COMPLETED` as scientific success before bounded fetch/check.

## Model readiness

Accept only an explicit model reference with content fingerprint, supported elements, site-root-relative path, and framework-specific LAMMPS export format.

Supported contracts:

- `deepmd` + `artifact_format: deepmd-lammps-model`;
- `mace` + `artifact_format: mace-lammps-torchscript`;
- `m3gnet` + `artifact_format: matgl-lammps-torchscript`.

Do not pass raw MACE/MatGL training checkpoints directly to LAMMPS. Do not invent a CHGNet pair style; version 0.3 still blocks native CHGNet LAMMPS and should route CHGNet MD to `$ase-md` unless a separately reviewed bridge exists.

## Preparation contract

Require explicit NVT or isotropic NPT configuration, target list, type map, temperature, timestep, total global steps, thermo/dump intervals, positive velocity seed, thermostat damping, and NPT pressure/barostat damping when applicable.

Do not invent physical parameters from composition. Preparation converts fs to ps and GPa to bar for LAMMPS `metal` units.

Require one full-rank 3D periodic atomic structure. Do not silently generate molecular topology, charges, bonded terms, hybrid potentials, long-range electrostatics, or nonperiodic boundaries.

The execution-ready prepare wrapper must produce `preparation_contract: lammps-md-input-v2`. Requested decks must use `${MODEL_FILE}` and contain the exact `MLIPFLOW_LAMMPS_COMPLETED step=<total>` print command only after `final.data` and `final.restart` writes.

## CPU/GPU framework routing

DeepMD:

- CPU/GPU decks both use `pair_style deepmd`;
- do not add a Kokkos suffix solely because GPU was selected;
- the site launcher owns MPI/GPU mapping and must respect the supported GPU/rank mapping.

MACE:

- require the reviewed ML-MACE TorchScript artifact;
- CPU uses `pair_style mace`;
- GPU uses `pair_style mace no_domain_decomposition` and Kokkos launcher metadata;
- version 0.3 requires one scheduled GPU for this path;
- do not silently substitute ML-IAP.

MatGL/M3GNet:

- require a LAMMPS-exported MatGL model;
- CPU uses `pair_style matgl`;
- GPU uses `pair_style matgl/kk`;
- version 0.3 requires one GPU for this path.

## Execute contract

For `operation: execute`, require:

- `backend: ssh-slurm` and a named backend profile;
- exactly one project-scoped `inputs.lammps_input_manifest`;
- explicit `target: cpu|gpu`;
- exact `input_manifest_fingerprint`;
- resources containing exactly `cpus`, `gpus`, `memory`, `walltime`;
- optional `checkpoint_interval`;
- optional `restart_policy`, either `disabled` or `auto-from-previous-attempt`.

`auto-from-previous-attempt` requires a positive `checkpoint_interval` no greater than the prepared total steps.

CPU requires `gpus: 0`. GPU requires at least one GPU; MACE/MatGL are restricted to one GPU in this contract.

The adapter must revalidate the prepared manifest, `structure.data`, selected deck, launcher/model contract, and their SHA/size before submission.

## Site boundary

Never put SSH host, account, partition, QoS, module/conda setup, LAMMPS executable, MPI/srun command, CUDA architecture, MODEL_ROOT, template root, or remote work root in the project node.

Template families remain:

- `lammps-deepmd-cpu`
- `lammps-deepmd-gpu`
- `lammps-mace-cpu`
- `lammps-mace-gpu`
- `lammps-m3gnet-cpu`
- `lammps-m3gnet-gpu`

The v0.3 site `run.sh` invokes the staged `lammps_cluster_restart.py`. The site owns `PYTHON_BIN`, `LAMMPS_BIN`, `MODEL_ROOT`, and JSON launcher argv. Keep executable and launcher stable across restart attempts; version 0.3 fingerprints them.

The compute-node runner resolves the prepared model only below MODEL_ROOT, verifies its SHA before execution, passes it via `-var MODEL_FILE`, and verifies it again after execution.

## Periodic restart policy

When `checkpoint_interval` is present, the runner derives an active deck from the reviewed fresh deck and adds exactly:

```lammps
restart <interval> checkpoint.1.restart checkpoint.2.restart
```

Do not replace this with a single overwrite file. The two fixed names alternate, preserving another candidate if interruption occurs during a checkpoint write.

Before starting the long LAMMPS subprocess, write a bounded `restart-runtime.json` that binds:

- attempt;
- framework/target;
- prepared manifest SHA;
- model SHA;
- checkpoint cadence;
- exact abstract resource object;
- resolved LAMMPS executable SHA;
- site launcher prefix identity;
- prepared launcher identity;
- platform system/machine/byteorder.

These are failure-recovery artifacts, not normal successful outputs. After normal success, periodic checkpoint files and the runtime sidecar are removed; `final.restart` remains the ordinary successful restart artifact.

## Failure salvage before retry

For a scheduler-terminal `TIMEOUT`, `PREEMPTED`, `NODE_FAIL`, `OUT_OF_MEMORY`, `FAILED`, `DEADLINE`, `CANCELLED`, or `REVOKED`, do not immediately run `retry`.

First create the normal `advance --dry-run` failure-salvage plan. The core must inventory only allowlisted outputs and bind observed SHA/size before fetching. With periodic restart enabled, the salvage subset may include:

- `checkpoint.1.restart` when present;
- `checkpoint.2.restart` when present;
- `restart-runtime.json` when present;
- approved diagnostic report/logs.

Do not salvage partial trajectory/final state as successful science.

After salvage approval/fetch, the original attempt remains `FAIL` or `STOPPED`. Only then create `retry`, which creates a fresh attempt.

If no periodic checkpoint exists because interruption occurred before the first checkpoint, auto restart must be BLOCKED. Do not silently fresh-start, do not rebuild from `final.data`, and do not reinitialize velocities while calling the result a restart.

A scheduler `COMPLETED` run that later fails its scientific checker is not auto-restart eligible.

## Retry planning

For attempt N > 1 with `auto-from-previous-attempt`, accept only the immediately previous attempt and require:

- previous final manifest state is FAIL/STOPPED;
- scheduler terminal state is restart-eligible;
- `restart-runtime.json` was locally salvaged;
- at least one alternating checkpoint was locally salvaged;
- previous runtime identity matches the new plan's framework, target, prepared manifest, model, checkpoint interval, and resources.

Stage every approved available candidate plus the runtime sidecar into the new fresh attempt. Bind each candidate SHA/size, source attempt, previous executable SHA, and previous platform into the new approval plan.

The control plane must not parse the binary LAMMPS restart or guess its timestep.

## Compute-node resume

Before `read_restart`, require current runtime to match the salvaged sidecar for executable SHA, platform, site launcher identity, prepared launcher identity, resources, framework/target, prepared manifest, model, and checkpoint interval.

Use the same LAMMPS executable to inspect each approved salvaged binary candidate. Ignore corrupt/unreadable/out-of-contract candidates and choose the largest valid timestep that lies on the approved checkpoint cadence and is below the global total step.

The derived resume deck must:

- `read_restart ${RESTART_FILE}`;
- reissue the reviewed MLIP `pair_style` and `pair_coeff`;
- reissue timestep/thermo/dump settings;
- reuse the exact fresh-deck `fix mlipflow all nvt ...` or `fix mlipflow all npt ...` line with the same fix ID/style/arguments;
- never run the fresh `velocity create` line;
- re-enable the same periodic checkpoint cadence;
- `run <original-total-steps> upto`;
- write normal final data/restart and completion marker after reaching the original global target.

`steps` always means the entire trajectory target. A retry runs only the remainder represented by the binary restart timestep.

## Reproducibility statement

Do not promise universal bitwise trajectory identity for LAMMPS restart. Binary restart is runtime-bound, and processor decomposition / floating-point ordering can still alter the resumed numerical trajectory. Record `bitwise_exact_guaranteed: false` even after compatibility checks.

Describe the feature as **state-continuous restart under a pinned compatible runtime**, not cross-platform portable checkpointing.

## Completion

Normal successful execute still requires:

- matching `lammps-execution-result.json` and `cluster-run-report.json`;
- bounded `trajectory.lammpstrj`, `final.data`, `final.restart`, `lammps.log`, `lammps.screen.log`;
- matching output SHA/size records;
- recorded LAMMPS version;
- `steps_completed == approved global steps`;
- exact completion marker in fetched `lammps.log`.

For a resumed attempt, additionally require the selected checkpoint SHA to be one of the newly approved salvaged candidates, a valid periodic start step, the expected source attempt, runtime compatibility confirmation, and cluster/result restart identities that agree.

## Scientific interpretation

`lammps-md` produces trajectories. It does not prove equilibration, diffusion, ionic conductivity, phase stability, or model validity. Version 0.3 also does not stitch trajectory/log segments automatically; keep stitching, multi-temperature orchestration, and transport analysis separate until explicitly implemented.
