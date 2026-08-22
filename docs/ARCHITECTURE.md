# MLIPFlow architecture

MLIPFlow is a small workflow controller for its built-in AI-for-materials capabilities. It is not a third-party plugin platform, a workflow database, or a portable plan interchange format.

```text
Agent Skills
    ↓
project.yaml scientific DAG
    ↓
built-in capability adapter
    ↓
local process or SSH-Slurm
    ↓
scientific result
```

## Sources of truth

MLIPFlow deliberately keeps each kind of information in one place:

| Information | Authoritative source |
|---|---|
| Intended nodes, inputs, parameters, resources, and `needs` edges | `project.yaml` |
| Current attempt, state, backend, scheduler job, and remote directory | one SQLite `step_runs` table |
| Logs, rendered scripts, staged control files, and collected outputs | the attempt directory |
| Scientific values and completion evidence | capability-specific scientific results |

SQLite does not copy workflow nodes or dependency edges. It has no artifact index, event history, retry counter, provenance store, or topology snapshot. The attempt number itself records retries.

## Built-in capabilities

`src/mlipflow/plugins.py` contains a literal `BUILTIN_CAPABILITIES` mapping. Core uses only the adapter filename, supported execution backends, operation names, approval requirement, and a short description.

There is no filesystem discovery, manifest loading, semantic-version selection, plugin API version, or user-supplied plugin path. A project names a capability directly:

```yaml
uses: dft-labeling
```

The current IDs are:

- `active-learning`
- `ase-md`
- `candidate-ranking`
- `dft-labeling`
- `electrochemical-voltage`
- `high-entropy-structure`
- `ionic-transport`
- `lammps-md`
- `mlip-benchmark`
- `mlip-training`
- `pes-sampling`

Adapters remain ordinary Python implementations with four lifecycle methods where the execution path needs them:

- `validate(context)` checks scientific inputs and prerequisites;
- `plan(context)` describes the concrete local or scheduled action;
- `check(context)` decides whether the scientific computation completed successfully;
- `collect(context)` returns the scientific outputs available to downstream work.

Missing framework or external-program dependencies are reported by the relevant adapter while planning. `doctor` only checks global project, built-in capability, state, site, and SSH configuration.

## Commands and mutation boundaries

The read-only commands are `list`, `status`, `json`, `inspect`, `logs`, `route`, and `doctor`. They do not submit jobs, fetch results, advance workflow state, or create project files.

The mutating commands are `init`, `run`, `advance`, `retry`, and `stop`.

`run NODE --dry-run` builds the concrete plan without creating an attempt directory. Expensive capabilities and every SSH-Slurm submission require a later explicit `run NODE --approve`. The plan contains real paths for the current controller checkout; plans are not designed to move between machines or relocated projects.

## Minimal state

Each SQLite row represents one node attempt and contains only:

- `run_id`, `project_id`, `node_id`, and `attempt`;
- current `state` and `backend`;
- `job_id` and `remote_dir` for scheduled work;
- created, updated, submitted, started, and ended timestamps;
- a diagnostic string.

The useful state flow is:

```text
WAIT ──dependency OK──> READY ──local──> RUNNING ──scientific check──> OK
  │                       │
  └──dependency failed──> BLOCKED

READY ──submit──> SUBMITTED ──queue──> PENDING ──scheduler──> RUNNING
                                                      │
                                                      └──fetch/check──> OK or FAIL
```

`advance` rereads `needs` from the current `project.yaml` and combines those edges with the latest dependency states. It also observes already submitted scheduler jobs and finalizes their scientific results. It never launches a new scientific node.

## Attempt directories and retry

Attempts live at:

```text
.mlipflow/runs/<node-id>/attempt-<N>/
```

The directory is created fresh and is never reused. A failed or stopped node may be retried; retry inserts attempt `N + 1` and leaves all earlier directories and state rows intact.

A small `run-manifest.json` attempt record may contain the node and attempt identity, state, backend/profile, scheduler job, remote directory, command, timestamps, diagnostic, and collected artifact references. It does not serialize the DAG, project inputs, resources, dependency graph, environment, platform, event history, or generic provenance.

Downstream artifact bindings read the relevant attempt records and scientific results directly. There is no second artifact catalog in SQLite.

## Generic replay

A node with `mode: replay` reads `inputs.result_manifest`, requires an explicit successful scientific state, and references its declared artifacts. It does not import or execute the capability adapter and does not run numerical software.

Replay is intentionally described as a structured collection of existing results, not as recomputation or independent scientific validation.

## Local execution

For a local execute node, core:

1. validates and plans through the built-in adapter;
2. creates a fresh attempt directory;
3. runs the adapter's argv with `shell=False`;
4. records stdout and stderr;
5. calls the adapter's scientific `check` and `collect` methods;
6. marks the attempt `OK` only if process execution, checking, and collection succeed.

A zero process exit code alone is not scientific success.

## SSH-Slurm execution

Cluster details stay in the user-owned `site.yaml` and remote template tree. A project supplies abstract resources—CPUs, GPUs, memory, and wall time—and may name `backend_profile` explicitly.

When `backend_profile` is omitted, core compares configured scheduler profiles against the requested resources and current partition availability, then selects a capable cluster. Within that profile, the existing partition router chooses a compatible partition. This routing remains part of the scheduler subsystem rather than the scientific adapters.

The scheduled lifecycle is:

```text
adapter plan
  -> resolve cluster and partition
  -> render site-owned templates
  -> stage declared inputs and runners
  -> sbatch
  -> persist job ID and remote attempt directory
  -> poll during advance
  -> fetch declared required/optional outputs
  -> adapter scientific check
  -> adapter collect
  -> OK or FAIL
```

Schema-3 scheduled plans represent one scheduler job. Schema-4 plans retain independent multi-job submission for batch calculations. Restart-aware ASE and LAMMPS adapters may select checkpoints from preserved previous attempts and continue into a new attempt directory.

`stop NODE` cancels only the job IDs recorded for that node attempt. Automatic cluster selection records the selected cluster in the scheduled attempt plan so later polling, fetch, and cancellation use the same site profile.

## Scientific ownership

Core controls execution and state transitions. It does not calculate or infer energies, forces, stresses, diffusion coefficients, conductivities, voltages, or rankings.

Scientific runners own formulas, model commands, DFT/MD settings, convergence checks, result schemas, and numerical output. Agent Skills supervise parameter selection and approval boundaries. Scheduler completion must always be followed by the capability's scientific check before a node reaches `OK`.

The scheduler backend, site configuration, template rendering, automatic cluster/partition routing, restart logic, and multi-job execution remain substantial engineering subsystems. They are deliberately retained for a later scheduler-focused contraction rather than redesigned here.
