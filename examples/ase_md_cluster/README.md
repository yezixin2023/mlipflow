# Scheduled ASE MD

`ase-md@0` is the cluster-first ASE molecular-dynamics plugin. Version 0.3 runs a single-temperature NVT Langevin or isotropic MTK NPT trajectory with an explicit DeepMD, M3GNet/MatGL, CHGNet, or MACE model, and can continue that same trajectory across fresh scheduler attempts from an approved checkpoint.

Supported ensembles:

- `nvt-langevin`: fixed-cell Langevin NVT.
- `npt-isotropic-mtk`: isotropic Martyna-Tobias-Klein NPT using ASE `IsotropicMTKNPT`.

Version 0.3 still does not run NVE, anisotropic/full-cell NPT, replica exchange, fit diffusion, orchestrate a temperature series, or automatically concatenate trajectory segments.

## Site templates

Copy `run.sh.example` into each calculator environment that you want to expose:

- `<remote_template_root>/ase-md-deepmd/run.sh`
- `<remote_template_root>/ase-md-m3gnet/run.sh`
- `<remote_template_root>/ase-md-chgnet/run.sh`
- `<remote_template_root>/ase-md-mace/run.sh`

Each family may activate a different module/conda environment. The site template owns the Python executable and `MODEL_ROOT`; the portable workflow never contains an SSH host, partition, module name, absolute model path, template root, or work root.

ASE MD declares `execution_model: single-python`; install `slurm/single-python/cpu.sbatch` and/or `slurm/single-python/gpu.sbatch`. The CPU template must use `--ntasks=1` and `--cpus-per-task={{CPUS}}`, because `resources.cpus` is the thread budget of one Python process.

## Model registry reference

The project stores a small `model-reference.json` rather than uploading the model. DeepMD, CHGNet, and MACE use `kind: file`. M3GNet/MatGL uses `kind: directory`. `relative_path` is resolved only below the site-owned `MODEL_ROOT`; the run record keeps the model ID, path, calculator settings, software version, seed, and outputs.

## NVT workflow node with restart

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
    checkpoint_interval: 10000
    restart_policy: auto-from-previous-attempt
    seed: 20260814
    device: cuda
    default_dtype: float64
    friction_per_fs: 0.01
    fix_com: true
    input_index: "-1"
  resources:
    cpus: 8
    gpus: 1
    memory: 32G
    walltime: "24:00:00"
```

`steps` is the total global target, not a per-Slurm-job count. If attempt 1 times out after a checkpoint at step 430000, attempt 2 is planned from global step 430000 and runs only the remaining 570000 steps.

NVT checkpoints restore the atomic state plus the NumPy `PCG64` bit-generator state used by ASE Langevin. A restarted attempt does not call Maxwell-Boltzmann velocity initialization again and does not reseed Langevin noise.

## Isotropic NPT workflow node with restart

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
    checkpoint_interval: 10000
    restart_policy: auto-from-previous-attempt
    seed: 20260814
    device: cuda
    default_dtype: float64
    fix_com: false
    input_index: "-1"
  resources:
    cpus: 8
    gpus: 1
    memory: 32G
    walltime: "24:00:00"
```

NPT intentionally has no `friction_per_fs`. The target pressure and both damping times are explicit workflow inputs rather than inferred defaults. Version 0.3 pins the MTK thermostat/barostat chain lengths to 3/3 and chain integration substeps to 1/1; these values appear in approval/provenance.

The initial structure must have a full-rank 3D periodic cell and no ASE constraints. The concrete loaded model must provide finite ASE stress. The compute-node runner performs a stress probe before fresh or restarted NPT dynamics and fails closed if the model cannot supply a finite 3x3 stress tensor.

NPT checkpoint restart restores positions, momenta and cell plus ASE's MTK particle/cell state, thermostat-chain state and barostat-chain state. Because those extended variables currently live in implementation-private ASE attributes, the checkpoint's exact ASE version must match the environment used for the restart. MLIPFlow refuses a version mismatch instead of degrading to a geometry-only restart.

## What the checkpoint contains

When `checkpoint_interval` is set, the runner atomically replaces `output/md-checkpoint.json`. It writes a temporary JSON file, flushes and `fsync`s it, then renames it into place. An interruption during a new checkpoint write should therefore leave the previous complete checkpoint available for salvage.

The checkpoint binds:

- plugin/checkpoint schema and exact ASE version;
- calculator, ensemble, model ID/path and original structure path;
- total requested steps, completed global step, timestep, temperature and seed;
- device/dtype and ensemble-specific friction or NPT pressure/damping/chain settings;
- atomic numbers/order, positions, momenta, cell, masses and PBC;
- NVT: Langevin implementation version and PCG64 RNG state;
- NPT: MTK particle/cell variables plus thermostat and barostat chain coordinates/momenta.

Checkpoint data is strict JSON; restart never unpickles remote content.

## TIMEOUT / PREEMPTED restart flow

A scheduler failure does **not** make arbitrary remote files eligible for download. Failure salvage remains limited to the allowlist approved with the run.

1. Submit attempt 1 normally with the approved run plan.
2. If Slurm reports `TIMEOUT`, `PREEMPTED`, `NODE_FAIL`, `OUT_OF_MEMORY`, `DEADLINE`, or another restart-eligible terminal state, run `advance` before creating a retry.
3. Core inventories only the adapter's `failure_salvage` subset of the already approved fetch allowlist, rechecks it during transport, and leaves the attempt in `FAIL` or `STOPPED`. For ASE MD this can include `md-checkpoint.json` and partial trajectory/index/thermo segment files. A salvaged checkpoint is not scientific success.
4. Run `retry`. This creates a **fresh attempt directory**; it does not reuse the failed remote workspace or launch computation.
5. Dry-run the new MD attempt. With `restart_policy: auto-from-previous-attempt`, the adapter reads only the immediately previous local attempt's salvaged `md-checkpoint.json`, verifies the previous scheduler terminal state and matching scientific parameters, and stages the checkpoint as `input/restart/md-checkpoint.json`.
6. Review `restart_from_attempt`, the checkpoint path, `segment_start_step`, `remaining_steps`, and the new segment's frame/thermo schedules, then approve submission.

There is no project parameter for an arbitrary restart path. This prevents an Agent from pointing a retry at an unrelated or unreviewed checkpoint.

A normal scientific `FAIL` after scheduler `COMPLETED` is deliberately not auto-resumable. Fix the scientific/configuration problem and make an explicit new plan instead.

An explicit MLIPFlow `stop` can transition a node immediately and therefore may not provide the same salvage window as observing a scheduler-side terminal cancellation. Do not rely on manual stop for restart unless the checkpoint has already been safely brought local through an approved salvage path.

## Segment semantics

Every attempt is immutable and has its own fresh remote/local workspace. A restart therefore writes a new `trajectory.traj`, `trajectory-index.json`, and `thermo.csv` segment rather than appending into the failed attempt's files.

Segment rows and index entries use **global** MD step/time numbers. For example, a retry from step 430000 starts its new segment at 430000 and continues toward the original step 1000000. This makes later continuity/stitching checks possible without pretending the binary trajectory was safely appended across Slurm jobs.

The producer does not concatenate segments into a replacement file. Keep all attempt artifacts; `ionic-transport` now consumes the collected attempts directly, orders frames by global step, and removes a repeated checkpoint-boundary frame.

## Approval and successful output contract

The submission approval summary exposes total simulated time, current segment start/remaining steps, exact output schedules, checkpoint path, structure/model paths, device, abstract resources, and ensemble-specific settings. NPT additionally exposes target pressure, thermostat/barostat damping, stress requirement, isotropic cell mode, no-constraints requirement, and fixed chain configuration.

After Slurm reports `COMPLETED`, use the normal second `advance` approval so MLIPFlow inventories and fetches only bounded declared outputs and runs the scientific checker.

A successful checkpoint-enabled attempt fetches `trajectory.traj`, `trajectory-index.json`, `thermo.csv`, `md-checkpoint.json`, `final.extxyz`, `md-result.json`, and `cluster-run-report.json`. The checker verifies completed total steps, segment frame/thermo schedules, finite thermodynamic values, the recorded model/structure paths, and checkpoint scientific parameters. NPT thermo additionally checks `pressure_GPa`, positive volume, and positive cell lengths. Scheduler exit code alone is never accepted as scientific success.
