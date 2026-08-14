# Scheduled ASE MD

`ase-md@0` is the cluster-first ASE molecular-dynamics plugin. Version 0.2 runs one fresh, single-temperature trajectory per workflow node with an explicit DeepMD, M3GNet/MatGL, CHGNet, or MACE model.

Supported ensembles:

- `nvt-langevin`: fixed-cell Langevin NVT.
- `npt-isotropic-mtk`: isotropic Martyna-Tobias-Klein NPT using ASE `IsotropicMTKNPT`.

It does not yet restart a trajectory, run NVE, perform anisotropic/full-cell NPT, fit diffusion, or orchestrate a temperature series.

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

For a directory model, the fingerprint is a deterministic tree hash over sorted relative file names, sizes, and per-file SHA-256 values.

## NVT workflow node

```yaml
- id: md-900k-nvt
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

## Isotropic NPT workflow node

```yaml
- id: md-900k-npt
  uses: ase-md@0
  mode: execute
  backend: ssh-slurm
  backend_profile: cluster-a
  inputs:
    structure: inputs/start.extxyz
    model_reference: inputs/mace-model-reference.json
  parameters:
    calculator: mace
    ensemble: npt-isotropic-mtk
    temperature_k: 900
    pressure_gpa: 0.0
    timestep_fs: 1.0
    thermostat_damping_fs: 100.0
    barostat_damping_fs: 1000.0
    steps: 1000000
    trajectory_interval: 100
    thermo_interval: 100
    seed: 20260814
    device: cuda
    default_dtype: float64
    fix_com: false
    model_fingerprint: sha256:REPLACE_WITH_MODEL_FINGERPRINT
    structure_fingerprint: sha256:REPLACE_WITH_STRUCTURE_SHA256
    input_index: "-1"
  resources:
    cpus: 8
    gpus: 1
    memory: 32G
    walltime: "24:00:00"
```

NPT intentionally has no `friction_per_fs`. The target pressure and both damping times are explicit workflow inputs rather than inferred defaults. Version 0.2 pins the MTK thermostat/barostat chain lengths to 3/3 and the chain integration substeps to 1/1; these values appear in approval/provenance.

The initial structure must have a full-rank 3D periodic cell and no ASE constraints. The concrete loaded model must also provide finite ASE stress. The compute-node runner performs a stress probe before the first NPT step and fails closed if the model cannot supply a finite 3x3 stress tensor.

## Approval and output contract

The approval summary exposes simulated time, exact output schedules, structure/model identities, device, abstract resources, and ensemble-specific settings. NPT additionally exposes target pressure, thermostat/barostat damping, the stress requirement, isotropic cell mode, no-constraints requirement, and pinned chain configuration.

After Slurm reports `COMPLETED`, use the normal second `advance` approval so MLIPFlow can inventory and fetch only the bounded declared outputs and run the pinned checker.

A successful attempt fetches `trajectory.traj`, `trajectory-index.json`, `thermo.csv`, `final.extxyz`, `md-result.json`, and `cluster-run-report.json`. The checker verifies completed steps, deterministic frame/thermo schedules, finite thermodynamic values, model and structure fingerprints, and every output SHA-256. NPT thermo additionally records and checks `pressure_GPa`, positive volume, and positive cell lengths. Scheduler exit code alone is never accepted as scientific success.
