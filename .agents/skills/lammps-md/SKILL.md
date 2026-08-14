---
name: lammps-md
description: Prepare and verify portable LAMMPS input bundles for explicit LAMMPS-ready DeepMD, MACE, or MatGL/M3GNet models. Use when the user wants CPU/GPU LAMMPS input files, model/type mapping, NVT/NPT deck generation, or a later LAMMPS execution workflow.
---

# LAMMPS MLIP input preparation

Use the `lammps-md` plugin as the implementation source of truth. Version 0.1 is **prepare-only**: it generates and verifies LAMMPS inputs locally but never launches LAMMPS, MPI, Kokkos, CUDA, SSH, or Slurm.

## Decide whether the model is LAMMPS-ready

Accept only an explicit `model_reference` with a content fingerprint, supported element list, site-root-relative path, and framework-specific exported artifact format.

Supported contracts:

- `deepmd` + `artifact_format: deepmd-lammps-model`.
- `mace` + `artifact_format: mace-lammps-torchscript`.
- `m3gnet` + `artifact_format: matgl-lammps-torchscript`.

Do not treat a raw MACE or MatGL training checkpoint as a LAMMPS model. MACE and MatGL require their own LAMMPS export steps before this preparation contract.

Do not generate a native CHGNet pair style in version 0.1. The adapter deliberately blocks legacy CHGNet references because no native CHGNet LAMMPS export/pair-style contract is pinned here. Suggest the existing `$ase-md` path instead unless a separate reviewed bridge is introduced.

## Require explicit MD input

Require an `lammps_config` with:

- `schema_version: 1` and `engine: lammps`;
- ensemble `nvt` or `npt-isotropic`;
- explicit `targets` containing `cpu`, `gpu`, or both;
- explicit `type_map` in LAMMPS atom-type order;
- target temperature in kelvin;
- timestep in femtoseconds;
- total step count;
- thermo and dump intervals;
- positive LAMMPS velocity seed;
- thermostat damping in femtoseconds;
- for NPT, explicit pressure in GPa and barostat damping in femtoseconds.

Do not invent physical parameters from the material composition. Preparation converts workflow units into LAMMPS `metal` units: fs to ps and GPa to bar.

Version 0.1 requires one full-rank three-dimensional periodic atomic structure. Do not silently generate molecular topology, charges, bonded terms, hybrid potentials, electrostatics, or nonperiodic boundary conditions.

## CPU/GPU framework routing

DeepMD:

- CPU deck: `pair_style deepmd`.
- GPU deck: still `pair_style deepmd`.
- GPU ownership belongs to the DeePMD-enabled LAMMPS runtime; do not add a Kokkos suffix just because `gpu` was requested.

MACE version 0.1:

- Require a pre-exported ML-MACE TorchScript artifact.
- CPU deck: `pair_style mace`.
- GPU deck: `pair_style mace no_domain_decomposition`.
- GPU launcher metadata uses one Kokkos GPU (`-k on g 1 -sf kk`).
- Do not silently substitute the newer ML-IAP interface; add it later as an explicit interface contract because it uses a different exported artifact and build/runtime requirements.

MatGL/M3GNet:

- Require a file exported by `mgl create-lammps-model`.
- CPU deck: `pair_style matgl`.
- GPU deck: `pair_style matgl/kk`.
- Version 0.1 treats the Kokkos path as single-GPU/single-rank.

## Keep model paths portable

Generated decks must reference `${MODEL_FILE}` and must never contain the cluster model root or `model_reference.relative_path`.

A later scheduled execution layer must:

1. resolve the approved logical model below a site-owned model root;
2. recompute and match the approved model fingerprint;
3. call the site-owned LAMMPS executable;
4. inject `-var MODEL_FILE <resolved-model-path>`;
5. own MPI/Kokkos/CUDA/module/conda details outside the portable project.

Do not put SSH hosts, partitions, QoS, executable paths, module names, CUDA architectures, or absolute model paths into this preparation workflow.

## Verify preparation

A successful preparation must produce exactly:

- `structure.data`;
- `lammps-input-manifest.json`;
- `in.cpu.lammps` when CPU was requested;
- `in.gpu.lammps` when GPU was requested.

The checker must recompute the three source-input SHA-256 values and every generated file SHA-256/size. It must also require `${MODEL_FILE}` in each deck and reject embedded site model paths.

Treat a successfully generated deck as **prepared input only**, not evidence that the remote LAMMPS build contains the required pair style or that an MD run has completed.

## Next execution boundary

When scheduled LAMMPS execution is added, keep it as a new approval boundary, analogous to `dft-labeling` preparation versus actual VASP labeling. Reuse the current MLIPFlow ssh-slurm core rather than embedding Slurm commands inside the generated LAMMPS deck.
