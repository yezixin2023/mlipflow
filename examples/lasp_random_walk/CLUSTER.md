# LASP random walk on an ssh-slurm cluster

LASP stochastic-surface-walking belongs to `pes-sampling`.  The project declares scientific inputs and generic resources; SSH identity, remote roots, modules, LASP executable paths, MPI launchers, partitions, and other site details stay in the user-local site configuration and the cluster's template library.

## 1. Site profile

Example `~/.mlipflow/site.yaml`:

```yaml
schema_version: 1
clusters:
  cluster-a:
    backend: ssh-slurm
    ssh_profile: cluster-a
    remote_template_root: /templates/cluster-a
    work_root: /work/mlipflow
```

LASP declares `execution_model: mpi`. The remote template root must contain `slurm/mpi/cpu.sbatch` and/or `slurm/mpi/gpu.sbatch` plus a LASP family template at `lasp-ssw/run.sh`.

Copy `examples/lasp_random_walk/cluster/run.sh.example` to `/templates/cluster-a/lasp-ssw/run.sh` and edit only the site-owned `PYTHON_BIN`, `LASP_BIN`, and `MPI_BIN` lines.  For a serial LASP build, remove the `--mpi-launcher` and `--mpi-processes` arguments from that template.

## 2. Project node

A scheduled LASP node has no `lasp_executable` input.  The licensed binary is resolved by the remote site template.

```yaml
- id: lasp-walk
  uses: pes-sampling@0
  mode: execute
  backend: ssh-slurm
  backend_profile: cluster-a
  inputs:
    input_structure: inputs/input.arc
    lasp_input: inputs/lasp.in
    lasp_auxiliary_files: {}
  parameters:
    operation: lasp-ssw-execute
    output_subdir: lasp-ssw
    historical_source_id: case://lasp/random-walk-001
    selection_stride: 1
    energy_max_ev: null
    max_frames: 1000
    include_best_arc: false
    include_md_arc: false
    seed_status: HISTORICAL_PARAMETER_UNKNOWN
    acknowledge_uncontrolled_seed: true
    preserve_historical_order: true
    lasp_version: "3.7"
  resources:
    cpus: 16
    gpus: 0
    memory: 64G
    walltime: "12:00:00"
```

`resources.cpus` is available as `{{CPUS}}` and means the MPI task/rank count. The submit template must use `--ntasks={{CPUS}}`; it must not also map `CPUS` to `cpus-per-task`. Do not set `mpi_processes` on an ssh-slurm LASP node because launcher/process topology is site-owned in scheduled mode.

## 3. What MLIPFlow stages

The approved scheduled plan stages only pinned project/plugin files: the project configuration, `input.arc`, `lasp.in`, declared auxiliary files, `lasp_ssw.py`, and `lasp_cluster.py`.  The cluster run template invokes the staged helper inside the Slurm allocation.

## 4. Bounded outputs and verified continuation

The approved run binds a fixed output allowlist. After scheduler completion, ordinary `advance` inventories that allowlist and rechecks sizes/hashes during transport without requiring a second approval. Required LASP outputs include:

- `cluster-run-report.json`
- `sampling-result.json`
- `ssw-structures.json`
- `selected-structures.json`
- `lasp-run-metadata.json`
- `selected-structures.tar.gz`
- raw `allstr.arc`
- the core `completion.json`

`best.arc`, `md.arc`, their manifests, and LASP logs are fetched only when declared/available.  Dynamic selected ARC files are packed remotely into `selected-structures.tar.gz`, which keeps the pre-fetch allowlist finite.

During finalization, the pinned adapter reparses fetched `allstr.arc`, recomputes the energy filter and accepted-order stride, recomputes deterministic structure IDs, verifies the selected manifest, and verifies every member of `selected-structures.tar.gz` against the corresponding source-frame SHA-256 before returning scientific `OK`.

This integration validates scheduling, lineage, and bounded transfer.  It does not claim that a fake/local test establishes LASP numerical correctness; production validation still requires a real LASP run on the target cluster.
