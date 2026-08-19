# LAMMPS MLIP preparation, cluster execution, and restart

`lammps-md@0.3` keeps the VASP-like approval split and adds an explicit walltime/preemption restart layer:

1. `lammps-prepare` runs locally and creates a reviewable LAMMPS input bundle.
2. `execute` consumes one reviewed `lammps-md-input-v2` bundle and submits exactly one CPU or GPU target through `ssh-slurm`.
3. Optional periodic binary restart files can be salvaged after scheduler interruption and rebound into a fresh retry attempt.

Preparation success is not execution success. Scheduler `COMPLETED` is also not scientific success: completed output fetch/check remains a separate approval.

## Preparation inputs and outputs

A preparation node supplies one ASE-readable periodic structure, one LAMMPS-ready model-reference JSON, and one LAMMPS MD config JSON. It produces `structure.data`, `lammps-input-manifest.json`, and one or both of `in.cpu.lammps` / `in.gpu.lammps`.

The preparation wrapper records `preparation_contract: lammps-md-input-v2` and an exact `MLIPFLOW_LAMMPS_COMPLETED step=<total>` marker. The marker appears only after `final.data` and `final.restart` are written.

## Portable model binding

Generated input files never contain a cluster model path. They reference `${MODEL_FILE}`. Scheduled execution resolves `model.relative_path` only below the site-owned `MODEL_ROOT`, recomputes the approved file SHA-256 or directory `tree-sha256-v1`, and passes the resolved artifact using LAMMPS `-var MODEL_FILE`.

LAMMPS-ready artifact formats remain:

- DeepMD: `deepmd-lammps-model`.
- MACE: `mace-lammps-torchscript` for the reviewed ML-MACE interface.
- M3GNet/MatGL native: `lammps_interface: matgl`, `kind: file`, and `matgl-lammps-torchscript` exported with `mgl create-lammps-model`.
- M3GNet/MatGL Python bridges: explicit `lammps_interface: gnnp|m3gnet`, `kind: directory`, and `matgl-model-directory`. The directory must be a native `matgl.load_model` artifact, not a mislabeled single checkpoint.
- CHGNet: intentionally unsupported by `lammps-md@0.3`; use `ase-md` until a native LAMMPS bridge is pinned and reviewed.

## Prepare workflow node

```yaml
- id: prepare-lammps-mace
  uses: lammps-md@0
  mode: execute
  backend: local
  inputs:
    structure: inputs/start.extxyz
    model_reference: inputs/mace-lammps-model.json
    lammps_config: inputs/lammps-npt.json
  parameters:
    operation: lammps-prepare
    output_dir: prepared/lammps-mace
    structure_format: extxyz
  resources: {}
```

After preparation, bind the generated manifest by its full SHA-256 in a separate execute node.

## Long-running execute node

```yaml
- id: run-lammps-mace-gpu
  uses: lammps-md@0
  mode: execute
  backend: ssh-slurm
  backend_profile: cluster-a
  inputs:
    lammps_input_manifest: prepared/lammps-mace/lammps-input-manifest.json
  parameters:
    operation: execute
    target: gpu
    input_manifest_fingerprint: sha256:REPLACE_WITH_PREPARED_MANIFEST_SHA256
    checkpoint_interval: 10000
    restart_policy: auto-from-previous-attempt
  resources:
    cpus: 8
    gpus: 1
    memory: 32G
    walltime: "24:00:00"
```

`steps` in the prepared MD config always means the global trajectory target. A retry does not run another `steps`; the restart deck uses `run <original-total-steps> upto` from the salvaged restart timestep.

## What periodic restart writes

When `checkpoint_interval` is set, the compute-node wrapper adds this execute-time command to the reviewed fresh deck:

```lammps
restart 10000 checkpoint.1.restart checkpoint.2.restart
```

LAMMPS alternates the two files. This is deliberately preferable to repeatedly overwriting one binary file: if a hard stop lands during one checkpoint write, the other file may still contain the previous complete checkpoint.

Before the long LAMMPS process starts, the runner also writes `restart-runtime.json`. It binds the attempt, framework/target, prepared-manifest SHA, model SHA, checkpoint cadence, abstract resources, resolved LAMMPS executable SHA, site launcher hash, prepared launcher hash, and platform identity.

These three files are failure-recovery artifacts. On a normal successful run the temporary periodic pair and runtime sidecar are removed; the ordinary successful artifact remains `final.restart`.

## Walltime / preemption recovery sequence

For `TIMEOUT`, `PREEMPTED`, `NODE_FAIL`, `OUT_OF_MEMORY`, or another approved scheduler terminal state, do not call `retry` first.

The intended sequence is:

```text
attempt-1 RUNNING
    ↓
LAMMPS alternates checkpoint.1.restart / checkpoint.2.restart
    ↓
Slurm TIMEOUT or PREEMPTED
    ↓
advance
    ↓
inventory and verify the bounded failure_salvage outputs
    ↓ approval
bounded fetch of available checkpoint files + restart-runtime.json + diagnostics
    ↓
attempt-1 remains FAIL/STOPPED
    ↓
retry creates fresh attempt-2
    ↓
new run plan binds previous checkpoint/runtime SHA values
    ↓ approval
compute node validates runtime compatibility, probes both binary candidates,
chooses the newest valid periodic timestep, then resumes to the original total step
```

If the scheduler interruption happens before the first checkpoint is written, auto restart is blocked. The workflow does not silently fall back to `final.data`, reinitialize velocities, or start a fresh trajectory under the name “restart”.

A scheduler `COMPLETED` run that later fails the scientific checker is also not automatically restart-eligible.

## Resume semantics

The retry input is a LAMMPS binary restart, not a structure-only continuation. The generated restart deck:

- uses `read_restart ${RESTART_FILE}`;
- reissues the reviewed MLIP `pair_style` / `pair_coeff` because the model path is not treated as restart state;
- reuses the exact generated `fix mlipflow all nvt ...` or `fix mlipflow all npt ...` command, preserving the same fix ID/style so LAMMPS can restore stored thermostat/barostat state;
- does **not** execute the original `velocity create` line;
- re-enables the same periodic checkpoint cadence;
- uses `run <original-total-steps> upto`;
- writes the normal `final.data`, `final.restart`, and completion marker only after reaching the global target.

The control plane never parses LAMMPS binary restart bytes. On the compute node the same LAMMPS executable probes every approved salvaged candidate and chooses the largest valid timestep that is below the global total and lies on the approved checkpoint cadence.

## Runtime compatibility boundary

Binary restart is deliberately stricter than ASE-MD JSON checkpointing. Before a resumed run may call `read_restart`, version 0.3 requires the current runtime to match the salvaged sidecar for:

- LAMMPS executable SHA-256;
- operating-system / machine / byte-order identity;
- site launcher JSON identity;
- prepared launcher identity;
- CPU/GPU/memory/walltime resource object;
- framework and target;
- model SHA-256;
- prepared input-manifest SHA-256;
- checkpoint cadence.

Changing one of these makes the retry plan or compute-node runner fail closed. Even when all checks match, the result records `bitwise_exact_guaranteed: false`: processor decomposition and floating-point ordering can still make a resumed trajectory numerically diverge from an uninterrupted run. The promise is state-continuous restart under the bound runtime, not universal bitwise identity.

## CPU/GPU contracts

| Framework | CPU | GPU |
|---|---|---|
| DeepMD | `pair_style deepmd` | `pair_style deepmd`; site launcher owns MPI/GPU mapping |
| MACE | `pair_style mace` | `pair_style mace no_domain_decomposition` + Kokkos; one GPU |
| M3GNet/MatGL native | `pair_style matgl` | `pair_style matgl/kk` + Kokkos; one GPU |
| M3GNet via GNNP bridge | `pair_style gnnp ${INTERFACE_PATH}` | unsupported |
| M3GNet legacy Python bridge | `pair_style m3gnet ${INTERFACE_PATH}` | unsupported |

DeepMD may request more than one GPU, but the site template remains responsible for at most one GPU per MPI rank. Keep the launcher stable across restart attempts because version 0.3 fingerprints it.

## Site templates

Install `run.sh.example` under every framework/target family you expose:

- `<remote_template_root>/lammps-deepmd-cpu/run.sh`
- `<remote_template_root>/lammps-deepmd-gpu/run.sh`
- `<remote_template_root>/lammps-mace-cpu/run.sh`
- `<remote_template_root>/lammps-mace-gpu/run.sh`
- `<remote_template_root>/lammps-m3gnet-cpu/run.sh`
- `<remote_template_root>/lammps-m3gnet-gpu/run.sh`
- `<remote_template_root>/lammps-m3gnet-gnnp-cpu/run.sh`
- `<remote_template_root>/lammps-m3gnet-legacy-cpu/run.sh`

The v0.3 example invokes the staged `lammps_cluster_restart.py` and passes the MLIPFlow attempt number. Each site template still owns `PYTHON_BIN`, `LAMMPS_BIN`, `MODEL_ROOT`, optional Python-bridge `INTERFACE_PATH`, modules/conda setup, and `LAMMPS_LAUNCHER_JSON`; none belong in the portable project.

## Real CPU functional validation

On 2026-08-16, two five-step NVT jobs completed the full MLIPFlow path on the
`cluster-a` SSH-SLURM profile:

| Interface | Slurm job | LAMMPS | Model identity | Prepared manifest | Result |
|---|---:|---|---|---|---|
| DeepMD `pair_style deepmd` | `redacted` | 2 Aug 2023; executable SHA-256 `c86fd0…e856` | `deepmd-lammps-model`; SHA-256 `e137c8…89aa` | SHA-256 `c0be4b…13b8` | 5/5 steps, exit 0, checker/collect `OK` |
| MatGL/M3GNet `pair_style gnnp ${INTERFACE_PATH}` | `redacted` | 2 Aug 2023; executable SHA-256 `adf720…8d43` | `matgl-model-directory`; tree SHA-256 `dfe3f1…425d4` | SHA-256 `b468a3…1268` | 5/5 steps, exit 0, checker/collect `OK` |

Both runs used one CPU rank, no GPU, 400 K NVT, a 1 fs timestep, and the same
explicit `Li P S Mn Fe Ni Cu Zn` type map. The generated decks kept site paths out
of the portable project, wrote `final.data` and `final.restart` before the exact
completion marker, and returned only the bounded approved output set. The GNNP
run preserved a failed first attempt caused by missing site Python initialization;
after the canonical site family was corrected, retry created a fresh attempt.

The machine-readable identities and output fingerprints are in
[`reports/lammps_cluster_cpu_functional_smokes.json`](../../reports/lammps_cluster_cpu_functional_smokes.json).
This is execution validation only. It does not validate force-field accuracy,
equilibration, transport, GPU execution, NPT, or binary restart. The tested GNNP
model does not supply virial pressure, so its pressure output is not a scientific
result.

## Completion and scope

Normal success still requires process exit zero, the exact approved completion marker, recorded LAMMPS version, unchanged input/model identities, bounded output files, matching SHA records, and consistent execution/result manifests.

Version 0.3 supports NVT and isotropic NPT restart for the currently prepared MLIP interfaces. It does not itself run transport analysis, replicas, or charged/molecular/hybrid force fields. `ionic-transport` now consumes collected trajectory segments across attempts, stitches them by global timestep, and accepts both older wrapped dumps and new dumps containing image flags.
