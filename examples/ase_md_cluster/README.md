# Scheduled ASE MD

`ase-md@0` is the cluster-first ASE molecular-dynamics plugin. Version 0.1 runs one fresh, single-temperature NVT Langevin trajectory with an explicit DeepMD, M3GNet/MatGL, CHGNet, or MACE model. It does not yet run NPT, restart a trajectory, fit diffusion, or orchestrate a temperature series.

## Site templates

Copy `run.sh.example` into each calculator environment that you want to expose:

- `<remote_template_root>/ase-md-deepmd/run.sh`
- `<remote_template_root>/ase-md-m3gnet/run.sh`
- `<remote_template_root>/ase-md-chgnet/run.sh`
- `<remote_template_root>/ase-md-mace/run.sh`

Each family may activate a different module/conda environment. The site template owns the Python executable and `MODEL_ROOT`; the portable workflow never contains an SSH host, partition, module name, absolute model path, template root, or work root.

The normal MLIPFlow scheduler templates `slurm/cpu.sbatch` and/or `slurm/gpu.sbatch` are still required.

## Model registry reference

The project stores a small `model-reference.json` rather than uploading the model. DeepMD, CHGNet, and MACE use `kind: file`. M3GNet/MatGL uses `kind: directory`. `relative_path` is resolved only below the site-owned `MODEL_ROOT`, and the compute-node runner recomputes the declared content fingerprint before and after MD.

For a directory model, the fingerprint is a deterministic tree hash over sorted relative file names, sizes, and per-file SHA-256 values. The same `ase_md_cluster.py fingerprint` logic can be reused later if a small CLI wrapper is exposed; for now the reference is intentionally prepared by the site/user that owns the model registry.

## Workflow node

```yaml
- id: md-900k
  uses: ase-md@0
  mode: execute
  backend: ssh-slurm
  backend_profile: cluster-a
  inputs:
    structure: inputs/start.extxyz
    model_reference: inputs/mace-model-reference.json
  parameters:
    calculator: mace
    ensemble: nvt-langevin
    temperature_k: 900
    timestep_fs: 1.0
    steps: 1000000
    trajectory_interval: 100
    thermo_interval: 100
    seed: 20260814
    device: cuda
    default_dtype: float64
    friction_per_fs: 0.01
    fix_com: true
    model_fingerprint: sha256:REPLACE_WITH_MODEL_FINGERPRINT
    structure_fingerprint: sha256:REPLACE_WITH_STRUCTURE_SHA256
    input_index: "-1"
  resources:
    cpus: 8
    gpus: 1
    memory: 32G
    walltime: "24:00:00"
```

The approval summary exposes the simulated time, exact output schedules, structure/model identities, device, and abstract resources. After Slurm reports `COMPLETED`, use the normal second `advance` approval so MLIPFlow can inventory and fetch only the bounded declared outputs and run the pinned checker.

## Output contract

A successful attempt fetches `trajectory.traj`, `trajectory-index.json`, `thermo.csv`, `final.extxyz`, `md-result.json`, and `cluster-run-report.json`. The checker verifies completed steps, the deterministic frame/thermo schedule, finite thermodynamic values, model and structure fingerprints, and every output SHA-256. Scheduler exit code alone is never accepted as scientific success.
