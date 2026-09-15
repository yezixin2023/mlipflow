# Source layout migration

## Current simplification changes

Normal execution uses [USAGE.md](USAGE.md); this section is for updating older callers.

- JSON retains output paths and result evidence; `run` failures now return 1 with
  `ok: false`. Queries can succeed while reporting node `FAIL`.
- Set `default_profile` in the existing site config, select a node profile explicitly,
  or use `backend_profile: auto` for intentional cross-cluster selection.
- Scheduled training uses `mlip-<framework>` for all reference shapes. See
  [the short DeepMD replacement example](../examples/training_all_models/USAGE.md#earlier-deepmd-configurations).
  Historical attempts remain readable. Pre-change, unfinished DeepMD jobs using the
  removed `deepmd` launcher should be collected with their original installed version;
  new attempts use the current bundled contract.
- Import backend classes and private helpers from their defining modules. Public
  services such as `run_node`, `make_run_plan`, `query_workflow`, `advance` and `retry`
  remain available from `mlipflow.services`.

## Earlier source relocation interfaces

The CLI, project and site configuration, capability IDs, operation names, approval
rules, state database and scientific result formats retain their existing behavior.
Existing attempt directories and recorded results are preserved. Scientific
calculations do not need to be repeated because of this source relocation.

Install the checkout again after updating it (`python -m pip install -e .` for
development). A previously installed release does not automatically acquire moved
modules. Published packages require the same scientific dependencies as before.

## Python and script locations

| Previous source | Current source |
|---|---|
| `src/mlipflow/` | `mlipflow/` |
| `src/mlipflow/plugins.py` | `mlipflow/plugins/__init__.py` |
| `plugins/<hyphenated-id>/` | `mlipflow/plugins/<underscored_id>/` |
| `src/mlipflow/science/model_runtime.py` | `mlipflow/plugins/model_runtime.py` |
| `src/mlipflow/science/active_learning.py` | `mlipflow/plugins/active_learning/science.py` |
| `src/mlipflow/science/transport.py` | `mlipflow/plugins/ionic_transport/manuscript.py` |
| `src/mlipflow/science/voltage.py` | `mlipflow/plugins/electrochemical_voltage/science.py` |
| `plugins/lammps-md/lammps_prepare_v2.py` | `mlipflow/plugins/lammps_md/lammps_prepare.py` |

Use `mlipflow.plugins.load_adapter("dft-labeling")` for a capability's lifecycle
entrypoint. Internal imports and direct source-script paths must be updated;
there are no compatibility copies under the former source directories.

Every capability has an `adapter.py`. ASE separates `nvt`, `npt` and `restart`;
LAMMPS separates `prepare`, `execute` and `restart`; training keeps its
local and scheduled responsibilities in named modules. DFT separates preparation,
labeling and dataset assembly. PES operation modules cover DIRECT, LASP inputs,
SSW and merging, with one dispatch table in `adapter.py`. These modules use plain
functions rather than additional adapters or lifecycle mixins. Ionic transport
keeps formal analysis and its result checks together in `analysis.py`, alongside
MD handoff and manuscript reproduction. The historical CLI and its formulae live
in the self-contained `mlipflow/plugins/ionic_transport/manuscript.py`.

LAMMPS preparation now directly writes the existing v2 completion contract;
there is no second generator wrapper. The unused `m3gnet_runner.py` placeholder
was removed; M3GNet training continues through `mlip_m3gnet.py`.

## Agent Skill locations

Specialist Skill folders now live at
`mlipflow/plugins/<underscored_id>/skill/`, including their `SKILL.md`,
`agents/openai.yaml` and optional references. Their previous
`.agents/skills/<skill-name>` locations are relative directory symlinks to the
canonical folders, preserving repository discovery and `$skill-name` invocation.
Edit the files beside the corresponding capability; there is one maintained copy.

`mlip-active-learning` maps to `mlipflow/plugins/active_learning/skill/`.
The cross-capability `mlip-workflow` Skill remains in
`.agents/skills/mlip-workflow/`, and `electrochemical-voltage` still has no
dedicated Skill.

Repository skill discovery requires a checkout that preserves directory symlinks.
Release builds use the canonical files directly and do not depend on these links.

## Installed and standalone execution

Plugin Python files are regular package code. They are no longer installed below
`share/mlipflow/plugins`. Specialist Skills are included as package data beside
their installed capability code. Documentation, schemas, small examples and the
existing `share/mlipflow/agent-skills/<skill-name>/` Skill exports remain
distributed below `share/mlipflow`.

The controller imports only its registered package modules. Scientific scripts
also support execution by filename with their explicitly bundled sibling files.
Scheduled plans still declare their staged files, remote filenames and output
contracts; cluster environments do not need the complete controller package.
Only these standalone runners use file-based loading of declared dependencies.

New ASE plans stage `ase_md.py` and `ase_md_cluster.py` as their Python bundle.
They no longer upload the unused controller adapters `adapter-npt.py` and
`adapter-legacy.py`. LAMMPS preparation needs only `lammps_prepare.py`, and
historical transport replay needs only `manuscript.py` plus the standard library.

Previously submitted attempts retain their approved plans and remote files.
Collection does not require old local staging-source paths to remain present.
Retries create new attempts using the current package and preserved checkpoints.

Historical evidence retains its original source locators. The manuscript driver
resolves known source relocations when reading it and compares reports allowing
only those path changes; values, units and scientific claims must still match.
It does not rewrite the committed evidence or recorded reports.
