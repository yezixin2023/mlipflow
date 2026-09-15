# Run MLIPFlow tasks

## Install the tool and find its examples

The workflow core and all dependency extras require **Python 3.12 or newer**.
This common baseline follows the current stable [NumPy 2.5.3](https://pypi.org/project/numpy/2.5.3/)
and [SciPy 1.18.1](https://pypi.org/project/scipy/1.18.1/) Python requirements,
checked on 2026-09-15. CI runs the full local scientific test suite on Python 3.12.

Use an existing suitable conda environment, or create an isolated environment.
Base installation supports the offline ranking example; add `transport` for formal
ionic analysis or `local` for all general scientific dependencies. MLIP frameworks
and licensed programs are installed separately in their execution environments.

```bash
python -m pip install .
mlipflow --version
```

With a distribution file, replace `.` by the wheel path. Published resources are
under the active environment's `share/mlipflow/`:

```bash
python -c 'import sysconfig; print(sysconfig.get_path("data") + "/share/mlipflow")'
```

Start with [the real local ranking example](../examples/local_ranking/USAGE.md).
Task recipes use the same CLI:

- [Training and fine-tuning](../examples/training_all_models/USAGE.md)
- [ASE MD](../examples/ase_md_cluster/USAGE.md)
- [LAMMPS preparation and MD](../examples/lammps_mlip_inputs/USAGE.md)
- [Ionic transport](../examples/ionic_transport/USAGE.md)

## Read the output

Options precede the command: `mlipflow --project PROJECT --format json run NODE`.
`PROJECT` is a directory or a project YAML/JSON file; there is no upward search.
`init` creates local state. A standalone `run NODE` also initializes its state when
necessary. `inspect NODE` reports inputs and the installed capability contract.

JSON run output keeps `node_id`, `state`, `attempt`, `job_id` when available,
`artifacts` with roles and absolute paths, reported `metrics`, `check`, `collection`,
`manifest_path`, and available `logs`. Some capabilities also provide a `summary`.
`json NODE` or `--format json status NODE` reads saved state and the same output evidence.
Older attempts written before these summaries were added retain artifact paths;
their missing summaries are not recomputed by a query.

| Command outcome | Exit code | Top-level `ok` | Task state |
| --- | --- | --- | --- |
| Synchronous execution and checks pass | 0 | true | OK |
| Process, check, or collection fails | 1 | false | FAIL |
| Remote submission accepted | 0 | true | PENDING (unfinished) |
| Status query successfully reads a failed task | 0 | true | FAIL |
| Configuration, approval, or I/O error | 2 | false | See `error` |
| Doctor finds a problem | 1 | false | See diagnostics |

Thus `ok` describes the command outcome. For downstream scientific work, require
the node's `state: OK`. A remote submission needs `advance` to observe, fetch and
check its results. `status` only reads saved state and never contacts the scheduler.
Dry-run reports the exact local argv or scheduled job plan, resolved site/resources,
staged sources, templates, workspace, and expected outputs. Long rendered scripts
are retained in the approved run plan when submitted.

## Choose a site

Keep site configuration outside projects, normally in `~/.mlipflow/site.yaml`.
An explicit node `backend_profile` wins. Otherwise MLIPFlow uses the site's
`default_profile`, or its sole profile. Multiple profiles without a default produce
a configuration error before any cluster probe.

```yaml
# site.yaml, alongside schema_version and clusters
default_profile: cluster-a
```

Set `backend_profile: auto` on a node only when cross-cluster availability selection
is desired. This mode uses the existing scheduler selection policy. Site bootstrap
and environment setup are covered in [CLUSTER_ENVIRONMENTS.md](CLUSTER_ENVIRONMENTS.md).

## Enable a Skill separately

Installing the Python tool does not automatically enable Skills in every agent.
In a checkout, `.agents/skills/<name>` discovers each specialist's canonical
`mlipflow/plugins/<capability>/skill/` directory via a relative symlink. The workflow
Skill lives directly under `.agents/skills/mlip-workflow/`.

Installed canonical Skills are exported under `share/mlipflow/agent-skills/`.
For an agent that discovers project `.agents/skills`, copy the desired Skill directory
there, including its `agents` and `references` directories. For example, using the
resource path printed above:

```bash
mkdir -p .agents/skills
cp -R /PATH/TO/share/mlipflow/agent-skills/candidate-ranking .agents/skills/
```

Use a free destination; do not replace a customized Skill. Other agents may use a
different discovery folder. Follow that agent's configured discovery path, then
invoke `$candidate-ranking` or allow normal skill selection. The Python package and
Skill must refer to the same installed version. Source maintenance edits the canonical
Skill once; wheel copies are generated during packaging.

Ordinary tasks use the Skill, task configuration, inputs and CLI output. Read source
only when developing a feature or diagnosing a concrete error. Architecture and
migration notes are optional references, not prerequisites for normal execution.
