# MLIPFlow Development Handoff

This file records portable development constraints only. Site profiles, scheduler job
IDs, absolute model/data paths, credentials, and unpublished run results
must stay outside the repository.

## Architecture

- The control chain is `Agent -> Skill -> MLIPFlow Core -> Plugin -> scientific software`.
- MLIPFlow is the deterministic execution layer; an Agent supervises scientific intent.
- Scheduler `COMPLETED` is not scientific `OK`. The selected plugin must still validate
  completion semantics and collect only approved artifacts.
- Scheduled work uses a fresh attempt workspace, bounded staging/fetch allowlists, and
  preserved retry history. Never overwrite or delete an earlier attempt to simulate a
  retry.
- Do not add a second workflow engine, artifact registry, scheduler path, or bundled
  scientific implementation when an existing plugin contract already owns the stage.

## Site boundary

- A project names a backend profile and abstract `cpus/gpus/memory/walltime` resources.
- SSH hosts, accounts, partitions, QoS, modules, launchers, executable paths, environment
  activation, template roots, data/model roots, and work roots belong to the user-local
  site profile or site-owned templates.
- Documentation and examples use placeholders such as `cluster-a`, `compute`,
  `/path/to/python`, `/path/to/data`, `/path/to/models`, and `/path/to/work`.
- Credentials, private keys, licensed pseudopotential contents, private model weights,
  raw datasets, trajectories, scheduler logs, and real job identifiers are never
  committed.

## Artifact and scientific contracts

- Producer artifacts may be handed directly to declared downstream dependencies through
  explicit role bindings; the final `OK` producer attempt remains authoritative.
- Canonical DFT labels preserve units, energy convention, force/stress convention,
  structure lineage, and one shared train/validation/test split across framework views.
- Published model references record framework, kind, and a safe site-root-relative path.
- Benchmark selection is task-specific. Undefined Pearson values are recorded as
  unavailable; only metrics shared by the full compared model set may decide a winner.
- ASE/LAMMPS trajectory handoff to ionic transport uses recorded timestep, frame spacing,
  cell/species metadata, and unwrapped-coordinate semantics. Transport calculation and
  transport convergence are separate claims.
- Short MD and out-of-training DFT spot checks are supplemental evidence; the formal
  benchmark remains the primary model-selection evidence.

## Scheduler behavior

- One logical scheduled node may own one job or an explicit set of independent jobs.
- Mixed `PENDING` and `RUNNING` states are normal until every job is terminal.
- Finalization performs bounded per-workspace fetch followed by aggregate plugin checks.
- Per-job resources are site policy. Core does not encode a universal CPU count,
  partition, node, or machine-specific routing rule.
- A failed calculation must be identifiable without allowing one successful scheduler
  state to promote the whole scientific node to `OK`.

## Verification before handoff

Run relevant focused tests, then the complete local suite. Verify an editable install
with `pip install -e .` and invoke `mlipflow` without `PYTHONPATH=src`. Finally scan
tracked files for private paths, site names, job identifiers, runtime state, licensed
files, model weights, datasets, trajectories, logs, and unpublished results.
