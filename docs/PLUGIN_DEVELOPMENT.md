# Developing Computational Plugins

Computational plugins are the deterministic scientific execution units of MLIPFlow.

They live under:

```text
plugins/<plugin-id>/
```

A plugin is responsible for validating scientific inputs, preparing a deterministic execution plan, invoking or wrapping scientific software, checking scientific completion, and producing structured results.

Agent Skills are different.

Files under:

```text
.agents/skills/
```

teach the Agent **when** a capability should be used, **what information is required**, and **how results should be interpreted**. They must not contain or replace the primary numerical implementation.

A useful rule of thumb is:

> **Skills explain the workflow. Plugins perform the science.**

---

## Quick Start

A minimal plugin looks like this:

```text
plugins/my-plugin/
  plugin.yaml      # Plugin metadata and execution contract
  adapter.py       # Imported only on explicit execution paths
```

`plugin.yaml` must validate against:

```text
schemas/plugin.schema.json
```

The adapter implements the plugin lifecycle:

```python
class Adapter:
    def validate(self, context): ...
    def plan(self, context): ...
    def prepare(self, context, plan): ...
    def check(self, context): ...
    def collect(self, context): ...
    def replay(self, context): ...
```

The lifecycle is intentionally separated into stages so that planning, execution, scientific validation, result collection, and replay remain auditable.

For most new plugins, development should follow this order:

1. define the scientific inputs and outputs;
2. define completion and failure criteria;
3. implement a side-effect-free `validate()` and `plan()`;
4. wrap the existing scientific program rather than rewriting it;
5. produce structured results;
6. add successful, failed, and incomplete fixtures;
7. add parity tests against trusted small examples;
8. only then declare the corresponding capability as validated.

---

# 1. Plugin Responsibilities

A computational plugin must be a deterministic and verifiable scientific execution unit.

It should answer four questions clearly:

1. **What scientific task does this plugin perform?**
2. **What inputs and parameters are required?**
3. **How do we know the calculation succeeded scientifically?**
4. **What structured outputs does the plugin produce?**

A plugin must not rely on the Agent to infer numerical results from arbitrary logs, plots, or stdout.

For example, the Agent must not be expected to estimate or infer:

- energies;
- forces;
- diffusion coefficients;
- ionic conductivities;
- voltages;
- model rankings.

Those values must come from deterministic scientific code and structured plugin outputs.

---

# 2. Plugin Manifest

Every plugin must provide a `plugin.yaml`.

JSON syntax is also valid YAML and is often convenient because it can be parsed statically in minimal environments.

At minimum, the manifest must declare:

- `id`;
- `name`;
- `description`;
- `version`;
- `api_version`;
- inputs;
- parameters;
- outputs;
- Python dependencies;
- external program dependencies;
- supported execution backends;
- scientific completion criteria;
- scientific failure states;
- retry behavior;
- approval requirements;
- replay capabilities.

Third-party scientific programs must not be bundled unless their licensing and redistribution terms explicitly permit it.

The manifest must describe **capabilities that have actually been implemented and validated**.

Do not advertise a backend merely because MLIPFlow core supports that backend.

For example:

> MLIPFlow core supporting `ssh-slurm` does not automatically mean that a plugin supports `ssh-slurm`.

Similarly, do not use empty capability lists or placeholder fields in a way that implies functionality has already been implemented.

Current built-in plugins may provide local thin adapters and contract-level states such as `adapter-ready` or `implemented`. These labels do **not** imply that:

- the external scientific software is bundled;
- real numerical parity has been demonstrated;
- the implementation has been scientifically validated against production calculations.

A scientific capability should only be promoted to the corresponding validated state after its:

- adapter;
- completion criteria;
- fixtures;
- parity tests

have all been verified.

---

# 3. Adapter Lifecycle

The adapter interface is:

```python
class Adapter:
    def validate(self, context): ...
    def plan(self, context): ...
    def prepare(self, context, plan): ...
    def check(self, context): ...
    def collect(self, context): ...
    def replay(self, context): ...
```

Each method has a distinct responsibility.

## `validate()`

Validate inputs, parameters, dependencies, and scientific preconditions.

Validation should reject malformed or scientifically incomplete configurations before any execution directory is prepared.

## `plan()`

Construct the deterministic execution plan.

`plan()` must behave as a pure planning operation.

It must not:

- create files or directories;
- access the network;
- submit jobs;
- start subprocesses;
- mutate the environment;
- load large datasets;
- import heavyweight GPU frameworks.

A dry run depends on this property.

## `prepare()`

Prepare execution inputs only after validation and planning have succeeded.

This may include creating the fresh attempt workspace and writing validated input files.

## `check()`

Determine whether the scientific calculation completed successfully.

Scheduler success is not scientific success.

For example:

```text
SLURM COMPLETED
```

must not automatically become:

```text
scientific status = OK
```

The plugin must evaluate its own completion criteria.

## `collect()`

Parse validated scientific outputs and produce structured results.

The Agent should consume structured JSON, CSV, manifests, or other explicitly defined result formats rather than arbitrary stdout.

## `replay()`

Read and normalize existing results without rerunning the scientific calculation.

Replay must never become an implicit execution path.

---

# 4. Adapter Safety

Query-only commands do not import plugin adapters.

However:

```text
run --dry-run
```

does import the selected adapter.

Therefore, adapters are trusted code.

Module-level code and the `validate()` / `plan()` paths must not:

- create directories;
- read large datasets;
- import heavyweight GPU frameworks;
- mutate the environment;
- access the network;
- start subprocesses.

In the current implementation, third-party plugins should not be treated as untrusted data unless an independent execution sandbox is available.

Do not generate or execute plans from plugins whose source is not trusted.

---

# 5. Wrapping Existing Scientific Software

MLIPFlow follows a:

> **wrap-before-rewrite**

principle.

If a trusted scientific program or workflow already exists, prefer wrapping it with a deterministic interface rather than reimplementing the underlying scientific method.

When wrapping external scientific software:

1. Record its provenance.

   Document, in `plugin.yaml`, related documentation, or the pull request:

   - source;
   - software version;
   - license;
   - redistribution limitations;
   - modifications made;
   - validation evidence.

2. Replace hard-coded paths and parameters with explicit inputs or CLI arguments.

3. Invoke external programs using an argument list.

   Use:

   ```python
   subprocess.run([...], shell=False)
   ```

   Do not construct shell command strings from user input.

4. Use a controlled working directory and a restricted environment.

   Do not blindly inherit or record sensitive environment variables.

5. Validate the input schema before preparing execution directories.

6. Treat process exit status and scientific completion as separate checks.

   Even if the external program exits with code `0`, the plugin must still run its scientific completion criteria.

7. Produce structured results.

   Prefer:

   - JSON;
   - CSV;
   - result manifests;
   - explicitly defined scientific output files.

   Do not require the Agent to interpret arbitrary stdout.

8. Add parity tests against trusted small reference examples.

9. Refactor unsafe legacy behavior before wrapping it.

   Existing scripts containing:

   - `rm`;
   - destructive `mv`;
   - infinite polling loops;
   - implicit scheduler submission

   must be decomposed into explicit lifecycle steps instead of being executed unchanged.

---

# 6. Local Execution

Built-in adapters should default to the `local` backend unless remote execution has been explicitly implemented and validated.

A local plugin owns the scientific execution contract, including:

- validated input files;
- executable invocation;
- scientific completion criteria;
- output collection.

It should not implicitly depend on scheduler-specific behavior.

---

# 7. SSH-SLURM Execution

Remote scheduler execution has a stricter contract.

A new adapter supporting `ssh-slurm` must declare a complete scientific execution contract rather than merely adding a backend name to the manifest.

The adapter must declare:

```text
schema_version: 3
```

and provide:

- an explicit `execution_model`;
- a safe `template_family`;
- explicit `staged_files`;
- bounded `fetch_outputs`, including per-file size limits.

The adapter declares **scientific requirements**.

MLIPFlow core and the user's site configuration own **cluster-specific infrastructure**.

The adapter must therefore not define or guess:

- SSH hosts;
- SLURM partitions;
- environment modules;
- executable paths;
- launchers;
- remote work roots;
- complete `sbatch` scripts.

These belong to:

- the user's local cluster profile; and
- the remote template library.

The adapter must also not use fields such as:

```text
submit_script
remote_cwd
```

to bypass MLIPFlow's staging, fetch, workspace, or completion-check boundaries.

---

# 8. Remote Execution Ownership

For `ssh-slurm`, MLIPFlow core is responsible for:

1. resolving a fresh workspace from persistent attempt state;
2. selecting the appropriate scheduler template;
3. staging approved inputs;
4. rendering the scheduler script;
5. submitting the job;
6. reconciling scheduler state;
7. performing bounded output fetch during ordinary `advance`;
8. loading the pinned plugin;
9. running the plugin's scientific `check()` and `collect()`.

A scheduler-backed plugin without a complete scientific execution contract must be rejected by the core.

Simply adding:

```text
backend: ssh-slurm
```

to a manifest is not sufficient.

Local `slurm` adapters are not currently supported.

---

# 9. Execution Models

The current remote execution contract supports two execution models.

## `single-python`

This represents one Python process.

```text
resources.cpus
```

means the thread budget for that process.

The SLURM mapping must therefore be equivalent to:

```text
--ntasks=1
--cpus-per-task={{CPUS}}
```

Templates are selected from:

```text
slurm/single-python/cpu.sbatch
slurm/single-python/gpu.sbatch
```

as appropriate.

---

## `mpi`

This represents an MPI calculation.

```text
resources.cpus
```

means the number of MPI tasks/ranks.

The SLURM mapping must therefore be equivalent to:

```text
--ntasks={{CPUS}}
```

The same `{{CPUS}}` value must not also be reused as `cpus-per-task`.

Templates are selected from:

```text
slurm/mpi/cpu.sbatch
slurm/mpi/gpu.sbatch
```

as appropriate.

MLIPFlow core validates this mapping before staging begins.

Legacy schema version 2 and untyped scheduler templates such as:

```text
slurm/cpu.sbatch
slurm/gpu.sbatch
```

are no longer supported.

---

# 10. Canonical Remote Site Layout

A single physical computing site must reuse its canonical:

```text
remote_template_root
```

and:

```text
work_root
```

Do not create sibling roots based on:

- plugin;
- framework;
- scientific target;
- partition;
- validation round.

For example, avoid site layouts such as:

```text
templates-mattersim/
templates-vasp/
templates-validation-2/
```

when all of them refer to the same physical site.

If a required template family is missing:

1. inspect the existing canonical template root;
2. verify its current contents;
3. add the required family or scheduler-specific subdirectory under that root.

Never silently overwrite an existing template.

---

# 11. Example: LASP / SSW

LASP/SSW illustrates the boundary between a scientific adapter and infrastructure configuration.

The execution adapter should only wrap a user-provided executable.

It must require explicit information such as:

- software version;
- ARC input;
- `lasp.in`;
- required auxiliary inputs;
- a fresh attempt workspace.

The executable must be invoked with:

```text
shell=False
```

Optional MPI execution may accept only explicitly supplied ordinary executable paths such as:

```text
mpirun
mpiexec
```

together with a restricted argument pattern such as:

```text
-np N
```

This does **not** imply SLURM support.

Scheduler integration requires the complete `ssh-slurm` scientific contract described above.

A `normalize-replay` capability may parse an existing archive without executing LASP.

A fake executable smoke test demonstrates only that the adapter contract works.

It must not be presented as:

- a real LASP calculation;
- numerical validation;
- scientific parity.

---

# 12. Scientific Result Requirements

Scientific results should record the metadata required to interpret them correctly.

The exact fields depend on the capability, but may include:

- physical units;
- normalization convention;
- number of samples;
- number of tensor/vector components;
- random seed;
- dataset split;
- temperature;
- timestep;
- equilibration interval;
- fitting window;
- simulation volume;
- mobile species or particles;
- DFT settings;
- model path;
- dataset path;
- software version.

The result schema should make these choices explicit rather than relying on undocumented assumptions.

---

# 13. Common Scientific Errors

Plugin implementations should explicitly guard against common scientific mistakes.

## Scheduler success is not scientific success

Do not convert:

```text
COMPLETED
```

directly into:

```text
OK
```

without running the plugin-defined scientific completion checks.

---

## Do not accept unconverged electronic-structure results

For example, a VASP process may terminate while the requested electronic or ionic convergence criteria have not been satisfied.

The plugin must distinguish termination from convergence.

---

## Distinguish total and normalized energies

Do not mix:

```text
total energy
```

with:

```text
energy per atom
```

The result schema must record the normalization convention.

---

## Define stress conventions

When reporting stress, explicitly document:

- units;
- sign convention;
- tensor component order.

Do not assume every scientific program uses the same convention.

---

## Do not hard-code three-dimensional isotropic diffusion assumptions

The expression:

```text
D = slope / 6
```

is appropriate only under the corresponding three-dimensional isotropic diffusion assumptions.

For other dimensionalities or anisotropic systems, the plugin must use the scientifically appropriate formulation.

---

## State Nernst-Einstein assumptions

If ionic conductivity is calculated using the Nernst-Einstein relation, explicitly state assumptions concerning:

- independent charge carriers;
- correlation effects;
- the Haven ratio, if applicable.

Do not silently assume uncorrelated motion.

---

## Model ranking must be deterministic

A model ranking must come from:

- declared benchmark metrics;
- explicit routing criteria;
- deterministic comparison logic.

It must not be generated from:

- visual inspection of plots;
- model branding;
- Agent preference;
- subjective judgment.

---

# 14. Replay Mode

Replay exists to reuse and normalize existing scientific results.

It is not a hidden execution mode.

Replay may:

- read small existing result manifests;
- validate files explicitly referenced by those manifests;
- record source paths;
- record parameters;
- record software versions;
- record existing outputs;
- write a new run manifest for the current project when invoked through an explicitly approved `run`.

Replay must not:

- start numerical programs;
- submit scheduler jobs;
- copy large datasets;
- copy model weights;
- modify source artifacts;
- describe existing results as newly computed.

Replay outputs must be clearly identified as structured collection or normalization of existing results.

They must not be described as:

- recomputation;
- independent scientific validation.

---

# 15. Portable Replay and Result Manifests

Portable replay/result manifests must have an explicit:

```text
status: OK
```

when they represent a valid completed result.

Referenced files must use ordinary relative paths.

They must not:

- contain `..` traversal;
- escape the result directory through symbolic links.

Remote:

```text
completion.json
```

only proves that the remote process terminated.

At minimum, it must bind the completion record to:

```text
project_id
node_id
attempt
status
exit_code
```

MLIPFlow core retains the approved plan associated with that attempt.

Even when `completion.json` reports successful process termination, the pinned plugin must still execute its scientific:

```text
check()
collect()
```

before the result can be considered scientifically valid.

---

# 16. Retry Semantics

A retry must always create a fresh attempt.

Previous attempts and their results must remain available for inspection and provenance.

Do not implement retry by deleting or overwriting the previous run.

In particular, never use:

```text
rm
```

as a substitute for retry semantics.

Retry behavior should define:

- which source states are retryable;
- how the new attempt is created;
- which inputs are reused;
- what changes are allowed;
- the maximum retry count, if bounded.

---

# 17. Testing Requirements

Every plugin should include tests covering both engineering behavior and scientific behavior.

At minimum, test:

- manifest/schema validation;
- duplicate plugin IDs;
- API version compatibility;
- absence of adapter import-time side effects;
- zero-write behavior during planning and dry-run;
- argument-vector injection resistance;
- non-zero process exit handling;
- missing outputs;
- successful completion fixtures;
- failed completion fixtures;
- incomplete completion fixtures;
- run-manifest validation against `run-manifest.schema.json`;
- retry creating a fresh attempt without overwriting previous attempts;
- scientific units;
- parity against trusted small examples.

For scheduler-capable plugins, also test:

- synthetic multi-cluster site configurations;
- fake template libraries;
- fresh workspace generation;
- staging plans;
- scheduler reconciliation;
- SSH argument boundaries;
- remote path boundaries;
- bounded output fetching;
- execution-model-to-SLURM mapping.

---

# 18. Contract Tests vs Scientific Validation

It is important to distinguish different levels of validation.

A fake executable or smoke test may demonstrate that:

- arguments are passed correctly;
- files are staged correctly;
- completion detection works;
- result manifests are produced correctly.

This proves the **execution contract**.

It does not prove:

- correctness of the external scientific program;
- numerical equivalence with a reference implementation;
- scientific parity;
- predictive accuracy.

Scientific parity must be tested separately using trusted small reference calculations.

---

# 19. Recommended Development Checklist

Before considering a plugin ready, verify the following.

### Manifest

- [ ] `plugin.yaml` validates against `schemas/plugin.schema.json`.
- [ ] Plugin ID is unique.
- [ ] API version is supported.
- [ ] Inputs and parameters are explicit.
- [ ] Outputs are structured.
- [ ] Dependencies are declared.
- [ ] Backend claims match implemented capabilities.
- [ ] Approval requirements are correct.
- [ ] Replay behavior is declared.
- [ ] Retry behavior is declared.

### Adapter

- [ ] Importing the adapter has no side effects.
- [ ] `validate()` performs no execution.
- [ ] `plan()` is side-effect free.
- [ ] Dry-run performs zero writes.
- [ ] External commands use argument lists.
- [ ] `shell=False` is used.
- [ ] Input validation occurs before workspace preparation.
- [ ] Scientific completion is checked independently from process termination.
- [ ] Results are returned in structured formats.

### Scientific Validation

- [ ] Units are explicit.
- [ ] Normalization conventions are explicit.
- [ ] Required scientific metadata is recorded.
- [ ] Successful fixtures exist.
- [ ] Failed fixtures exist.
- [ ] Incomplete fixtures exist.
- [ ] Small trusted parity examples pass.

### Retry and Replay

- [ ] Retry creates a fresh attempt.
- [ ] Previous attempts remain intact.
- [ ] Replay does not execute numerical software.
- [ ] Replay does not modify source artifacts.
- [ ] Portable file references cannot escape the result directory.

### SSH-SLURM, if supported

- [ ] `schema_version: 3` is used.
- [ ] `execution_model` is explicit.
- [ ] `template_family` is explicit.
- [ ] `staged_files` are explicit.
- [ ] `fetch_outputs` are bounded.
- [ ] Per-file fetch limits are defined.
- [ ] Cluster-specific settings remain outside the adapter.
- [ ] The canonical site template root is reused.
- [ ] The canonical site work root is reused.
- [ ] Execution-model-to-SLURM mapping is tested.

---

# 20. Running the Core Plugin Tests

Run:

```bash
python -m pytest tests/test_plugin_manifests.py
```

Additional plugin-specific tests should be added for adapter behavior, completion criteria, replay behavior, scheduler contracts, and scientific parity as appropriate.

---

# Design Principle Summary

When designing a plugin, keep ownership boundaries explicit:

| Concern | Owner |
|---|---|
| When a scientific capability should be used | Agent Skill |
| Scientific input/output contract | Plugin |
| Numerical execution | Plugin / external scientific program |
| Scientific completion criteria | Plugin |
| Result parsing and normalization | Plugin |
| Model routing metrics | MLIPFlow routing logic |
| Persistent attempt state | MLIPFlow core |
| Fresh attempt workspace | MLIPFlow core |
| Scheduler submission | MLIPFlow core |
| SSH host and site configuration | User cluster profile |
| SLURM templates | Remote template library |
| Scientific interpretation | Agent, based on structured plugin outputs |

The central design rule is:

> **The Agent supervises, the plugin defines the scientific contract, and MLIPFlow provides controlled execution infrastructure.**

Do not move deterministic numerical work into Agent Skills, and do not move site-specific infrastructure into scientific adapters.