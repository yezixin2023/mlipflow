# LAMMPS MLIP input preparation

`lammps-md@0.1` prepares reviewable LAMMPS input bundles. It does **not** run LAMMPS yet.

A project node supplies three project-scoped inputs:

- one ASE-readable periodic structure;
- one LAMMPS model-reference JSON;
- one LAMMPS MD config JSON.

The preparation output contains `structure.data`, `lammps-input-manifest.json`, and one or both of `in.cpu.lammps` / `in.gpu.lammps`.

## Portable model binding

The generated input files never contain a cluster model path. They reference the LAMMPS command-line variable `${MODEL_FILE}`. A future scheduled execution layer will resolve the approved model below a site-owned model registry and call LAMMPS with `-var MODEL_FILE <resolved-path>`.

The model reference describes a **LAMMPS-ready** artifact, not necessarily the framework's raw training checkpoint:

- DeepMD: `artifact_format: deepmd-lammps-model` for a LAMMPS-loadable frozen model.
- MACE: `artifact_format: mace-lammps-torchscript` for a model exported with MACE's LAMMPS exporter.
- M3GNet/MatGL: `artifact_format: matgl-lammps-torchscript` for a file exported with `mgl create-lammps-model`.
- CHGNet: intentionally unsupported by `lammps-md@0.1`; use `ase-md` until a native LAMMPS bridge is pinned and reviewed.

`relative_path` is site-registry-relative metadata only and is not copied into the generated LAMMPS deck.

## Workflow node

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
    output_dir: lammps-inputs
  resources: {}
```

The generated CPU/GPU variants are framework-specific:

| Framework | CPU | GPU |
|---|---|---|
| DeepMD | `pair_style deepmd` | `pair_style deepmd` (GPU owned by DeePMD runtime) |
| MACE | `pair_style mace` | `pair_style mace no_domain_decomposition` + Kokkos launcher metadata |
| M3GNet/MatGL | `pair_style matgl` | `pair_style matgl/kk` + Kokkos launcher metadata |

## Units

The config is expressed in workflow-friendly units:

- temperature: K;
- timestep / thermostat / barostat damping: fs;
- pressure: GPa.

The generator writes LAMMPS `metal` units, converting fs to ps and GPa to bar in the input deck.

Version 0.1 supports NVT and isotropic NPT only. It intentionally does not generate hybrid potentials, charges, molecular topology, long-range electrostatics, restart orchestration, or a scheduler submission script.
