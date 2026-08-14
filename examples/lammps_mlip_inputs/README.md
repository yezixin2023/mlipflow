# LAMMPS MLIP preparation and cluster execution

`lammps-md@0.2` deliberately separates two operations:

1. `lammps-prepare` runs locally and creates a reviewable LAMMPS input bundle.
2. `execute` consumes an already reviewed `lammps-md-input-v2` bundle and submits exactly one CPU or GPU target through `ssh-slurm`.

Preparation success is not execution success. The scheduled calculation always gets a new plan digest and approval.

## Preparation inputs and outputs

A preparation node supplies three project-scoped inputs:

- one ASE-readable periodic structure;
- one LAMMPS-ready model-reference JSON;
- one LAMMPS MD config JSON.

The preparation output contains `structure.data`, `lammps-input-manifest.json`, and one or both of `in.cpu.lammps` / `in.gpu.lammps`.

The v0.2 wrapper adds `preparation_contract: lammps-md-input-v2` and an exact completion marker to every selected deck. That marker is printed only after `final.data` and `final.restart` have been written.

## Portable model binding

The generated input files never contain a cluster model path. They reference `${MODEL_FILE}`. Scheduled execution resolves the prepared `relative_path` only below the site-owned `MODEL_ROOT`, recomputes the approved model SHA-256, then supplies the path through LAMMPS `-var MODEL_FILE`.

The model reference describes a **LAMMPS-ready** artifact, not necessarily the framework's raw training checkpoint:

- DeepMD: `artifact_format: deepmd-lammps-model` for a LAMMPS-loadable frozen model.
- MACE: `artifact_format: mace-lammps-torchscript` for a model exported for the reviewed ML-MACE interface.
- M3GNet/MatGL: `artifact_format: matgl-lammps-torchscript` for a file exported with `mgl create-lammps-model`.
- CHGNet: intentionally unsupported by `lammps-md@0.2`; use `ase-md` until a native LAMMPS bridge is pinned and reviewed.

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
  resources: {}
```

After this node finishes, bind the generated `prepared/lammps-mace/lammps-input-manifest.json` by its full SHA-256 in a separate execute node.

## Execute workflow node

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
  resources:
    cpus: 8
    gpus: 1
    memory: 32G
    walltime: "24:00:00"
```

The execute adapter stages only the prepared manifest, `structure.data`, the selected `in.<target>.lammps`, the project config, and the pinned cluster runner. The model itself is not uploaded.

## CPU/GPU contracts

| Framework | CPU | GPU |
|---|---|---|
| DeepMD | `pair_style deepmd` | `pair_style deepmd`; the site launcher owns MPI/GPU mapping |
| MACE | `pair_style mace` | `pair_style mace no_domain_decomposition` + Kokkos; v0.2 requires one GPU |
| M3GNet/MatGL | `pair_style matgl` | `pair_style matgl/kk` + Kokkos; v0.2 requires one GPU |

DeepMD can request more than one GPU, but the site template remains responsible for launching at most one GPU per MPI rank. No portable MPI rank count is embedded in the project yet.

## Site templates

Copy `run.sh.example` into every framework/target family you expose:

- `<remote_template_root>/lammps-deepmd-cpu/run.sh`
- `<remote_template_root>/lammps-deepmd-gpu/run.sh`
- `<remote_template_root>/lammps-mace-cpu/run.sh`
- `<remote_template_root>/lammps-mace-gpu/run.sh`
- `<remote_template_root>/lammps-m3gnet-cpu/run.sh`
- `<remote_template_root>/lammps-m3gnet-gpu/run.sh`

Each template owns `PYTHON_BIN`, `LAMMPS_BIN`, `MODEL_ROOT`, modules/conda setup, and `LAMMPS_LAUNCHER_JSON`. The portable project never contains SSH hosts, partitions, module names, executable paths, absolute model paths, MPI launcher commands, template roots, or remote work roots.

`LAMMPS_LAUNCHER_JSON` is a JSON argv prefix such as `["srun","--ntasks=1"]`. The cluster runner parses it as argv and launches with `shell=False`.

## Approval and completion

The first execute approval binds framework, CPU/GPU target, model identity/fingerprint, prepared-manifest fingerprint, ensemble, timestep, total steps, type map, required package family, resources, and selected template family.

After Slurm reports `COMPLETED`, use the normal second `advance --dry-run` / approval boundary. MLIPFlow inventories and bounded-fetches only the declared outputs before checking them.

Successful execution requires all of:

- LAMMPS process exit code zero;
- exact `MLIPFLOW_LAMMPS_COMPLETED step=<approved steps>` marker in `lammps.log`;
- recorded LAMMPS version;
- unchanged prepared manifest/model identities;
- non-empty bounded `trajectory.lammpstrj`, `final.data`, `final.restart`, `lammps.log`, and `lammps.screen.log`;
- output size/SHA-256 records matching the fetched files;
- matching `lammps-execution-result.json` and `cluster-run-report.json`.

If the scheduler or LAMMPS fails, the plugin exposes only approved diagnostic logs/report through the core failure-salvage approval path. Partial trajectories or final-state files are not accepted as successful science.

## Units and scope

The preparation config is expressed in workflow-friendly units: temperature in K, timestep/damping in fs, pressure in GPa. The deck uses LAMMPS `metal` units, converting fs to ps and GPa to bar.

Version 0.2 supports NVT and isotropic NPT execution. It collects `final.restart` but does not yet resume from it. Hybrid potentials, charges, molecular topology, long-range electrostatics, replicas, automatic restart continuation, and transport analysis remain separate contracts.
