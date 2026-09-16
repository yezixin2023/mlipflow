# MLIPipe Agent Operating Guidelines

These instructions apply to this repository. The agent supervises scientific intent;
MLIPipe plugins and the named scientific programs calculate, execute and check results.
Prefer an existing suitable conda environment for development and execution.

## Normal use

Use the task's specialist Skill, existing project configuration, input artifacts and
CLI output. Start with `mlipipe json` when resuming a project. Reuse accepted outputs
and the existing scientific settings. Use `mlipipe inspect NODE` for the installed
operations, backend support and effective approval requirement; their definitions live
in `mlipipe/plugins/__init__.py`, not a parallel documentation table.

A single task can run directly. Use the workflow Skill only when several stages or
artifact handoffs need coordination. Examples and installation/Skill discovery are
in `README.md`. Source reading is for development or a concrete failure.

## Scientific decisions and authorization

- Never invent energies, forces, diffusion, conductivity, voltages or rankings. All
  numerical work goes through computational plugins or user-specified scientific programs.
- Reuse parameters already fixed by the user, project or a documented default, and
  show them in the plan. Gather missing decisions that change research intent together:
  units, seeds, splits, fitting windows, reference energies and scientific scope must
  not be guessed.
- Before an approval-gated run, review `run NODE --dry-run`: execution semantics,
  actual inputs, scale, backend/resources, staged files and expected outputs. An explicit
  user request to execute that task at that scale is authorization; carry it forward
  with `--approve`. Ask only when authorization or consequential scope is still missing.
- Deletion/cleanup, overwriting data or models, destructive reruns, remote staging,
  DFT/AIMD, long MD, training/fine-tuning and large screening need explicit user intent.
  Do not expand an approved task to cover new costly computations.
- Observation, bounded fetch, result collection and deterministic downstream work
  within the approved scope need no second approval. Diagnose `FAIL` using diagnostics,
  logs and the saved manifest before proposing retry; do not automatically retry a
  scientifically failed calculation.

## Execution and state

`list/status/json/inspect/logs/route/doctor` are strictly read-only: they must not write
state, create run files, contact schedulers to advance jobs, fetch, retry or submit.
`init/run/advance/retry/stop` change state. Use MLIPipe instead of direct `sbatch`,
`scancel`, directory deletion or model replacement. `retry` creates a fresh attempt
and preserves previous results. An explicit `stop NODE` authorizes cancellation of
that node's known jobs only.

Read both the exit code and node state. A successful submission is unfinished;
scientific success requires final `OK`, the capability completion checks and valid
required outputs. A `COMPLETED` scheduler job or process exit zero alone is insufficient.
Use `artifacts[].role` and `artifacts[].path` from JSON for downstream inputs.

Replay is a **structured collection of existing results**. It reads and validates
existing small manifests and references artifacts by path; it does not launch numerics,
submit jobs, copy large data or alter source evidence. Never describe it as recomputation
or independent scientific validation.

## Site and data boundaries

Keep passwords, keys, tokens, POTCAR content, weights and private large datasets out of
the repository. SSH backends refer only to configured SSH profile names. Do not guess
hosts, partitions, modules, executables, template roots or work roots. Do not use
`submit_script` or `remote_cwd` to bypass staging/fetch restrictions.

Reuse each site's canonical `remote_template_root` and `work_root`. If a template
family is missing, inspect the existing canonical root first, then add the required
family there with explicit site-setup intent. Never silently overwrite templates or
create parallel roots for a framework, target or validation round. Ordinary tasks use
an explicit/default/sole profile; cross-cluster selection requires `backend_profile: auto`.

## Development and maintenance

Read and change the relevant source when developing or diagnosing an error. Built-in
adapters are trusted and dry-run may load them; never derive execution plans from
untrusted project code. Test observable behavior, including zero-write queries and
scientific completion checks.

When project code is defective, fix its canonical source and verify the affected
workflow instead of patching only the current attempt. An outdated external runtime
should be selected or updated appropriately, not supported through unnecessary source
compatibility patches. Preserve failed attempts and distinguish program defects from
environment, site configuration, and scientific failures.

Specialist Skills have one maintained source at
`mlipipe/plugins/<capability>/skill/`, discovered through `.agents/skills/` symlinks.
The cross-capability workflow Skill remains in `.agents/skills/mlip-workflow/`.
Scientific algorithms belong in plugins, not Skills. Keep framework details under
training/MD capabilities rather than adding top-level Skills for individual models,
MSD fits, Slurm or OUTCAR parsing.

Keep README edits minimal: the user maintains them personally. Confirm any README
content deletion or reduction with the user before applying it.
