# Architecture

## One-Sentence Boundary

The Agent interprets scientific intent and produces scientific tasks together with abstract resource requirements; MLIPFlow persists and executes explicit plans.

Plugins encapsulate deterministic scientific capabilities. Cluster profiles and remote templates provide site-specific execution knowledge. Backends are responsible for composing execution, running workloads, and coordinating execution state.

The Agent must not guess hosts, partitions, modules, executables, templates, or remote work roots.

Ownership is singular: Agent Skills select and orchestrate scientific stages; plugin Adapters validate inputs, plan execution, check completion, and collect results; core owns execution state, generic replay, retries, approval, and local/SSH-Slurm execution; scientific programs and helper modules own numerical algorithms.

## Command / Query Separation

```text
Queries: list status json inspect logs route doctor
  └─ load only project/plugin/registry manifests
  └─ SQLite mode=ro + query_only
  └─ do not import adapters
  └─ do not call backends
  └─ do not write events

Commands: init run advance retry stop
  └─ explicit write-capable entry points
  └─ expensive/external runs require execution approval
  └─ advance/retry/stop express continuation or termination intent directly
  └─ external programs always run as argv + shell=False
```

`--dry-run` does not create the state database, attempt directories, staging areas, or scheduler submissions.

Local planning performs no network access.

For `ssh-slurm`, planning may use the explicitly selected local site profile to retrieve the required remote templates over SSH in **read-only** mode and display the rendered scripts.

Execution-mode planning loads the selected Python adapter. Plugins therefore belong to the trusted-code boundary in the same sense as build scripts.

Query commands do not import adapters or access backends, except that `doctor` may validate the local site configuration.

The approval summary shows the actual execution implementation, input paths, execution parameters, resources, backend, staged files, and the final scripts that will be executed.

## Agent-Facing Output

The query/planning service always returns one detailed internal object. Execution, auditing, and presentation all use this object as a single source of truth.

The CLI exposes only read-only projections by default:

- `status` shows node, state, attempt, and output roles;
- `inspect` shows node semantics and a plugin summary;
- `route` shows the selected model, scores, metrics, and concise elimination reasons;
- dry-run shows what will run, its inputs and resources, and whether approval is required.

Default text output uses command-specific line-oriented rendering.

Default JSON output uses the same compact projection and does not include complete manifests, raw configuration, or the full internal plan.

`--audit` does not re-plan and does not maintain a second plan model. It only exposes more of the same detailed object, including artifact provenance, the complete plugin manifest, routing contributions and evidence verification, and adapter/HPC plans.

## Approval: Explicit Boolean Confirmation

Plan `schema_version` is `4`.

Computations that require approval are first reviewed with:

```bash
mlipflow run NODE --dry-run --audit
```

After confirmation, execution is started with:

```bash
mlipflow run NODE --approve
```

MLIPFlow does not generate digests, hashes, fingerprints, or content identifiers for plans, approvals, inputs, scripts, or templates.

At execution time, the current plan is written into the attempt directory as ordinary JSON. Subsequent scheduler observation, bounded fetch, and scientific `check/collect` operations continue to use this recorded plan.

### Portability of Adapter-Authored Fields

Adapters naturally reason in absolute local paths. Built-in plugins emit `cwd`; some also emit absolute argv entries, staged `source` paths, and diagnostics containing referenced filenames.

These values remain in the audit plan. Before persistence, the core rewrites paths that fall under known roots into portable forms, and restores them immediately before execution:

```text
<project>/.mlipflow/runs/n/attempt-1/out
  -> {ATTEMPT_DIR}/out

<project>/prepared/POSCAR
  -> {PROJECT_ROOT}/prepared/POSCAR

<plugins>/dft-labeling/helper.py
  -> {PLUGIN_DIR}/helper.py
```

`portable.to_portable()` performs this rewrite inside `make_run_plan`.

`portable.to_runtime()` restores runtime paths inside `_execute_ready`, `_scheduled_contract`, and scientific checker contexts.

Adapters therefore always receive local absolute paths. Their argv and working directories remain unchanged, so scientific behavior is unaffected.

Approval displays the portable description, while execution sees the local runtime implementation.

Paths outside all known roots are preserved unchanged. For example, `resources.python_executable` is an explicitly declared site-level configuration value. On machines using the same `project.yaml`, rewriting such a path would hide information that should remain visible during approval.

Tests cover `BLOCKED`, local `READY`, and `ssh-slurm READY` plan forms and verify that repository-local paths are correctly restored before execution.

## Persistent State

The project state database is stored at:

```text
.mlipflow/state.sqlite3
```

The main entities are:

- `step_runs`: immutable records for every node attempt;
- `dependencies`: DAG edges;
- `artifacts`: artifact URIs and roles;
- `events`: state-transition audit records;
- `submission_intents`: approved plans and their consumption time;
- `metadata`: state schema version.

State-database ownership is determined directly from the project ID stored in attempt rows.

Before a `READY` attempt begins execution, it is bound to a complete snapshot of the node at that time.

Post-submission scheduler observation, fetch, scientific checking, and `stop` all use this snapshot.

Historical attempts therefore remain immutable, while unrelated changes to the current `project.yaml` or future attempts do not invalidate the entire state database.

```text
WAIT ──dependencies completed──> READY ──local──> RUNNING ──check──> OK
 │                                      └───────────────────> FAIL
 ├─upstream failure──> BLOCKED
 └──────────────────────────────────────────────────────────> STOPPED

READY ──submit──> SUBMITTED ──queue──> PENDING ──schedule──> RUNNING

FAIL / STOPPED ──retry──> new READY attempt
                         old attempt is preserved
```

A SLURM `COMPLETED` state is only a scheduler fact.

Remote `completion.json` must record at least:

- project;
- node;
- attempt;
- successful process exit status.

Fetch is restricted to paths declared by the execution plan and is subject to size-limit checks.

After transport completes, the fixed adapter must still run scientific `check/collect` before the attempt can become `OK`.

`dft-labeling.label` with the static VASP contract is the first scientific plugin integrated with this generic scheduled lifecycle.

The remaining built-in adapters currently support local execution only.

## Three-Layer HPC Configuration

```text
Local ~/.mlipflow/site.yaml
  = named cluster selection / control plane
  = backend + SSH config alias + template root + work root

Remote <remote_template_root>/
  = persistent site-specific template library
  = Slurm skeleton + module/environment + launcher/executable knowledge

Remote <work_root>/<project>/<node>/attempt-XXXX/
  = ephemeral per-run workspace
  = rendered scripts + input/output/logs/completion
```

`site.yaml` does not belong to the project or repository.

A project node declares only:

```yaml
backend: ssh-slurm
backend_profile: <name>   # optional
cpus: ...
gpus: ...
memory: ...
walltime: ...
```

If `backend_profile` is omitted, the core selects among site profiles according to the number of currently available nodes that satisfy the requested resources.

An explicitly selected profile always takes precedence.

Project-level `backend_profiles`, `parameters.submit_script`, and `parameters.remote_cwd` are rejected.

Real cluster bootstrap, template installation, credentials, and authentication are independent site-administration procedures and are not part of the portable workflow definition.

## Template Resolution and Remote Workspace

The new `scheduled_execution schema_version=3` provides both a safe `template_family` and an `execution_model`.

The core first selects:

```text
slurm/<execution-model>/cpu.sbatch
```

or:

```text
slurm/<execution-model>/gpu.sbatch
```

It then uses `template_family` to select an application-specific execution template such as:

```text
vasp/run.sh
```

Templates may use only the following fixed placeholders:

```text
PROJECT_ID NODE_ID ATTEMPT RUN_DIR INPUT_DIR OUTPUT_DIR LOG_DIR
CPUS GPUS MEMORY WALLTIME
```

The renderer performs only exact `{{NAME}}` substitution and newline normalization.

It does not support expressions, includes, loops, or arbitrary code templating.

Missing templates, unknown or incomplete variables, missing resources, or nonexistent profiles all fail explicitly before staging.

### Execution Models

The meaning of `CPUS` is fixed by the execution model.

#### `single-python`

`CPUS` is the thread budget for one Python process.

The Slurm template must contain:

```text
--ntasks=1
--cpus-per-task={{CPUS}}
```

#### `mpi`

`CPUS` is the number of MPI tasks/ranks.

The Slurm template must contain:

```text
--ntasks={{CPUS}}
```

and must not simultaneously use `CPUS` as `cpus-per-task`.

MLIP training and ASE MD use `single-python`.

VASP, scheduled LAMMPS, and LASP use `mpi`.

The core validates this mapping before staging. Therefore, fixing CPU allocation for a single Python process cannot accidentally change the MPI layout of VASP, LAMMPS, or LASP.

Attempt numbers come only from existing attempt state in SQLite.

The workspace is fixed as:

```text
<work_root>/<project-id>/<node-id>/attempt-XXXX/
```

Each workspace contains:

```text
submit.sbatch
run.sh
input/
output/
logs/stdout.log
logs/stderr.log
completion.json
```

The directory must be fresh.

`retry` increments the attempt number and never overwrites an older workspace.

The template root and work root must not be nested inside one another.

## SSH-SLURM Execution State Machine

```text
resolve local profile

-> resolve remote templates

-> render deterministic execution plan/scripts

-> create fresh attempt workspace

-> stage reviewed inputs/scripts

-> submit and persist job ID

-> monitor scheduler

-> inventory/fetch outputs, logs, and completion

-> plugin scientific check

-> plugin collect

-> OK
```

Approval for `run` covers profile resolution, template resolution, rendering, staging, and submission.

After the scheduler reaches a terminal state, a normal `advance` reads the remote inventory using the output allowlist already bound to that run, performs bounded fetch, and then enters scientific validation.

No second approval is required.

## Plugin Discovery

The query path discovers plugins only by reading sorted static manifests:

```text
plugins/*/plugin.yaml
```

It validates plugin ID, API version, and supported backends without importing Python.

`implementation.entrypoint` is the sole Adapter entrypoint declaration. The manifest does not repeat the Python lifecycle, core replay/retry policy, shell behavior, or fixed success/failure states.

Only explicit execution paths may load an adapter.

Scientific plugins must not hard-code hosts, private keys, personal absolute paths, or scheduler-specific resource values.

## Artifacts and Replay

Each attempt writes an independent:

```text
run-manifest.json
```

The manifest records:

- paths;
- artifact roles;
- parameters;
- software versions;
- seeds;
- commands;
- outputs.

Large external datasets and models are referenced through safe relative paths under site-defined roots.

Adapter-produced artifacts must be ordinary files located inside the fresh attempt directory.

Replay accepts only ordinary result manifests located under the project root.

A replay manifest must explicitly declare `OK` and may reference only explicitly listed ordinary files within its own directory.

Replay is handled entirely by core. Planning and execution do not import the selected Adapter, call a plugin replay method, or consult a plugin-specific replay declaration.

Replay rejects:

- absolute paths;
- `..`;
- symbolic links.

Replay does not copy datasets and does not run numerical programs.

## Model Routing

The core contains no hard-coded "best model" constant.

Routing proceeds as follows:

1. Filter models by elements, task, and benchmark scenario.
2. Eliminate models missing required metrics and report the reason.
3. Normalize and score the remaining models according to project policy directions and weights.
4. Resolve ties using, in order:
   - element coverage;
   - validation sample count;
   - stable model ID ordering.
5. By default, return:
   - selected model;
   - ranking and metrics;
   - concise elimination reasons.

With `--audit`, routing additionally exposes per-metric score contributions and evidence verification.

Local benchmark evidence records ordinary file paths.

External URIs are explicitly marked as external.

Routing uses only declared task, scenario, metrics, and policy information. File-engineering properties are never treated as scientific metrics.

The DeepMD/CHGNet selections shown in examples come from synthetic benchmark fixtures in the example registry. They are not core preferences.

## Backend Boundaries

### Local

The local backend:

- synchronously executes explicit argv;
- always uses `shell=False`;
- uses a restricted environment;
- determines scientific success only after the fixed adapter runs `check/collect`.

LASP/SSW may execute directly through the local backend.

Local MPI is supported only when the launcher is an explicitly provided ordinary executable whose basename is `mpirun` or `mpiexec`, with the process count expressed as:

```text
-np N
```

This remains local execution and is not a scheduler backend.

### SLURM

The SLURM backend retains the scheduler-command abstraction.

Scientific nodes no longer use a user-supplied complete `sbatch` script as their primary execution contract.

### SSH + SLURM

The SSH+SLURM backend uses only the SSH alias declared by site configuration, the remote template library, and the configured remote work root.

It:

- creates a fresh attempt workspace;
- persists the scheduler job ID;
- reads terminal-state inventory without mutation;
- performs bounded local fetch;
- invokes the scientific checker after transport.

`POTCAR` is never fetched back from the remote system.

Remote staging and submission may only be triggered by an approved `run`.

Subsequent fetch operations may read only the output allowlist already bound to that run and must perform transport validation.

An explicit:

```bash
mlipflow stop NODE
```

expresses cancellation intent.

If query commands gain live scheduler overlays in the future, such information may exist only in memory and must not modify persistent state.
