<p align="center">
  <img src="docs/assets/mlipflow-logo.png" alt="MLIPFlow logo" width="720">
</p>

<p align="center"><strong>Deterministic, auditable workflow orchestration for machine-learned interatomic-potential research.</strong></p>

MLIPFlow turns an MLIP research workflow into an explicit, versioned DAG. It coordinates structure generation, PES sampling, DFT labeling, model training, molecular dynamics, benchmarking, transport analysis, candidate ranking, and voltage analysis while keeping execution plans, approvals, attempts, artifacts, model evidence, and provenance inspectable.

MLIPFlow is intentionally a **workflow layer**, not another MLIP framework. It wraps scientific tools and models you already use, gives them a common execution contract, and makes local and HPC runs reproducible enough for both humans and tool-using agents to supervise.

**Current package:** `0.1.0` (alpha) · **Python:** `>=3.9` · **License:** Apache-2.0

## Why MLIPFlow

- **One workflow model.** Define scientific steps and dependencies in `project.yaml` instead of stitching together ad-hoc shell scripts.
- **Deterministic execution.** Plugins expose structured plans, inputs, outputs, checks, retry behavior, and replay contracts.
- **Approval before expensive work.** `run --dry-run` produces the exact execution plan; approval binds to that plan rather than to a vague command.
- **Attempt lineage instead of overwrite.** Retries create fresh attempts and preserve previous state, artifacts, and scheduler history.
- **Local and HPC orchestration.** Run local scientific adapters or use controlled SSH + Slurm profiles with site-owned templates.
- **Evidence-driven model routing.** Rank models from versioned benchmark evidence and explicit policies rather than hard-coded model preferences.
- **Agent-ready supervision.** Bundled Agent Skills teach compatible tool-using agents how to plan and supervise workflows without moving numerical logic into the agent.
- **Replay and provenance.** Reuse existing scientific evidence without pretending it was recomputed, while retaining source paths and run provenance.

## Quick start

### 1. Install from source

```bash
git clone https://github.com/yezixin2023/mlipflow.git
cd mlipflow

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[local]"

mlipflow --version
```

`.[local]` is the recommended complete local MLIPFlow environment. It installs the
common scientific runtime covered by the `science`, `dft`, and `transport` extras
through one supported entry point. Because formal transport support comes from
`pymatgen-analysis-diffusion`, this complete local environment requires Python >=3.10.

The **MLIPFlow local environment** is the general control/science environment above.
An **MLIP framework execution environment** contains a framework and its accelerator
stack. `local` intentionally does not install DeepMD-kit, MACE, CHGNet, or
MatGL/M3GNet. You only need a framework-specific environment if you actually execute
that framework on this machine; remote/HPC execution can continue to use isolated,
site-owned environments and templates.

#### Advanced / minimal installation

Advanced users can keep the control environment smaller by installing only the core
or one fine-grained runtime extra:

```bash
# Workflow core only
python -m pip install -e .

# NumPy / ASE / icet / pandas / Plotly scientific helpers only
python -m pip install -e ".[science]"

# pymatgen support for VASP input preparation only
python -m pip install -e ".[dft]"

# Formal ionic transport only (Python >=3.10)
python -m pip install -e ".[transport]"

# Test, lint, schema validation, and development tools only
python -m pip install -e ".[dev]"
```

MLIPFlow core and historical transport reproduction continue to support Python >=3.9.
The formal `transport` extra requires Python >=3.10 and includes the runner's direct
NumPy, pandas, Plotly, and ASE dependencies.

### 2. Create a project

```bash
mlipflow init my-project
cd my-project
mlipflow doctor
```

`mlipflow init` creates a minimal `project.yaml` and initializes the local state database. Add workflow nodes, inputs, parameters, and routing policies before running scientific work.

### 3. Try the bundled replay example

From the MLIPFlow repository root, copy the example to a disposable working directory so the checked-in fixture stays untouched:

```bash
cd /path/to/mlipflow
cp -R examples/high_entropy_sulfide /tmp/mlipflow-replay-demo

mlipflow --project /tmp/mlipflow-replay-demo init
mlipflow --project /tmp/mlipflow-replay-demo list
mlipflow --project /tmp/mlipflow-replay-demo run structure-replay --dry-run
mlipflow --project /tmp/mlipflow-replay-demo run structure-replay
mlipflow --project /tmp/mlipflow-replay-demo advance
mlipflow --project /tmp/mlipflow-replay-demo status
```

This replay path consumes bundled result manifests rather than launching expensive scientific programs. For a larger evidence-reproduction example, see [`examples/high_entropy_sulfide_reproduction/`](examples/high_entropy_sulfide_reproduction/).

## The command model

MLIPFlow keeps observation, planning, execution, and reconciliation separate.

| Command | Purpose |
|---|---|
| `mlipflow init` | Create or initialize a project and state database. |
| `mlipflow doctor` | Run non-mutating diagnostics for the project, plugins, and site configuration. |
| `mlipflow list` | List workflow nodes and their states. |
| `mlipflow status [NODE]` | Inspect workflow or node state. |
| `mlipflow json [NODE]` | Emit machine-readable status JSON. |
| `mlipflow inspect NODE` | Inspect the selected node and plugin contract. |
| `mlipflow logs NODE` | Read saved logs for an attempt. |
| `mlipflow route ...` | Rank models from benchmark evidence and a project routing policy. |
| `mlipflow run NODE --dry-run` | Build the exact execution plan without running it. |
| `mlipflow run NODE --approve` | Execute approval-gated work after reviewing the dry-run. |
| `mlipflow advance` | Observe and reconcile state; it does not launch new workflow nodes. |
| `mlipflow retry NODE` | Create a fresh retry attempt while preserving lineage. |
| `mlipflow stop NODE` | Explicitly stop/cancel a running node when the backend supports it. |

Add `--format json` for structured output and `--audit` when you need detailed planning and provenance information.

A typical expensive run looks like this:

```bash
mlipflow --project . run train-model --dry-run --audit
# Review command, input paths, backend, resources, staged files, and scripts.
mlipflow --project . run train-model --approve
mlipflow --project . advance
mlipflow --project . status train-model
```

## Bundled scientific capabilities

MLIPFlow currently ships eleven scientific plugins. Plugin manifests are the source of truth for exact operations, inputs, outputs, backend support, dependencies, and completion checks.

| Plugin | What it does | Typical execution |
|---|---|---|
| `high-entropy-structure` | Seeded high-entropy/SQS structure generation with ASE + icet. | Local |
| `pes-sampling` | DIRECT representative selection, LASP/SSW execution, and historical archive normalization/replay. | Local; controlled SSH-Slurm for scheduled LASP execution |
| `dft-labeling` | Prepare/execute VASP static/relax/AIMD labels, collect a canonical labeled dataset, and remotely assemble DeepMD/M3GNet/CHGNet/MACE views. | Local preparation; controlled SSH-Slurm labeling and dataset assembly |
| `mlip-training` | Train or fine-tune DeepMD, M3GNet/MatGL, CHGNet, and MACE models. | Local and controlled SSH-Slurm contracts |
| `ase-md` | Run single-temperature ASE NVT Langevin or isotropic MTK NPT with explicit models and checkpoint/restart. | Controlled SSH-Slurm |
| `lammps-md` | Prepare LAMMPS MLIP decks and execute NVT/NPT with restart-aware scheduled runs. | Local prepare; controlled SSH-Slurm execute |
| `mlip-benchmark` | Normalize/evaluate prediction evidence and produce canonical metrics and rankings. | Local |
| `active-learning` | Evaluate calibrated one/two-model committees, combine model risks, select reviewed DIRECT candidates, and assess immutable offline rounds. | Local deterministic decisions; controlled SSH-SLURM committee inference; other numerical stages remain in their existing plugins |
| `ionic-transport` | Analyze native collected ASE-MD/LAMMPS-MD artifacts or historical ASE/LAMMPS/VASP/MSD data into diffusion, conductivity, and Arrhenius results; restart segments and upstream metadata are handled automatically. | Local |
| `candidate-ranking` | Deterministic ranking/top-k selection from existing candidate and metric manifests. | Local |
| `electrochemical-voltage` | Convert explicit total-energy sequences to voltage or replay reported voltage evidence. | Local |

The software contracts for these plugins are implemented and tested, but scientific validation is deliberately reported separately from software completeness. For the exact current validation level—including which scheduled paths have been exercised on a real cluster—see [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md), [`docs/SCIENTIFIC_VALIDATION.md`](docs/SCIENTIFIC_VALIDATION.md), and [`docs/HPC_VALIDATION.md`](docs/HPC_VALIDATION.md).

Recorded real-HPC evidence includes one bounded DeepMD training run, a real LASP
3.6.0 NN 14-atom CPU tiny SSW smoke, and CPU LAMMPS five-step functional smokes.
Each reached bounded fetch and pinned checker/collect `OK`; these validate execution
lifecycles, not scientific accuracy, LASP numerical parity, or production sampling/MD.
See the [machine-readable LASP smoke report](reports/lasp_cpu_tiny_hpc_smoke.json).

Local integration evidence also includes a real MAML/DIRECT 4-structure smoke that
produced two selected structures and passed the `pes-sampling` checker/collect path.
The [machine-readable DIRECT smoke report](reports/direct_local_integration_smoke.json)
records that this was not a full MLIPFlow core run lifecycle or historical
selection-parity result.

## Bundled Agent Skills

Agent Skills live in [`.agents/skills/`](.agents/skills/). They are **supervision guides**: they help an agent choose operations, request the right evidence, call MLIPFlow, and interpret structured results. The scientific computation remains in the plugins and external tools.

| Skill | Use it for |
|---|---|
| `$mlip-workflow` | Plan and supervise an end-to-end MLIP workflow. |
| `$mlip-active-learning` | Supervise finite offline committee active-learning campaigns and immutable rounds. |
| `$high-entropy-structure` | Build and review high-entropy/SQS structure-generation work. |
| `$pes-sampling` | Supervise representative selection and LASP/SSW sampling/replay. |
| `$dft-labeling` | Prepare and supervise DFT labeling workflows. |
| `$mlip-training` | Plan and supervise training/fine-tuning across supported MLIP frameworks. |
| `$ase-md` | Supervise ASE NVT/NPT production and checkpoint-aware retries. |
| `$lammps-md` | Prepare and supervise LAMMPS MLIP MD and binary-restart recovery. |
| `$mlip-benchmark` | Produce benchmark evidence and model rankings. |
| `$ionic-transport` | Analyze existing trajectories/MSD into transport results. |
| `$candidate-ranking` | Apply deterministic metric-based candidate ranking/top-k. |

Example requests to a compatible agent:

```text
$mlip-workflow Turn my structure -> DFT -> training -> benchmark plan into an MLIPFlow DAG.

$mlip-training Plan a MACE fine-tuning node using this dataset reference and show me the dry-run before execution.

$mlip-workflow Use these verified DFT labels to train DeepMD, M3GNet, CHGNet, and MACE without custom conversion scripts.

$ionic-transport Analyze these three temperature trajectories and report the assumptions used for diffusion and conductivity.
```

See [`docs/AGENT_SKILLS.md`](docs/AGENT_SKILLS.md) for the detailed contracts. The `electrochemical-voltage` plugin is currently used directly from workflows; there is not a separate bundled voltage Agent Skill.

## Project configuration

A project is described by `project.yaml` (JSON syntax is also valid YAML). The main sections are:

```yaml
schema_version: 1
project:
  id: my-mlip-project
  name: My MLIP project

model_registry: model_registry.yaml

workflow:
  nodes:
    - id: benchmark-replay
      uses: mlip-benchmark@0
      mode: replay
      backend: local
      inputs:
        result_manifest: replay/benchmark-result.json
      parameters: {}

routing:
  policies: {}

safety:
  auto_submit: false
```

The full schema is [`schemas/project.schema.json`](schemas/project.schema.json). Real examples are under [`examples/`](examples/), including replay, training, ASE MD, LAMMPS, LASP/SSW, site-template, and manuscript-reproduction material.

### State and attempt layout

By default, project state is local to the project:

```text
.mlipflow/state.sqlite3
.mlipflow/runs/<node>/attempt-<N>/
```

Each retry gets a new attempt. MLIPFlow records that attempt's plan, paths, parameters, versions, seed, and outputs so later project edits do not silently rewrite historical execution state.

## HPC configuration

Cluster-specific information belongs in a **user-local site file**, not in the research project. The default location is:

```text
~/.mlipflow/site.yaml
```

Example:

```yaml
schema_version: 1
clusters:
  lab-gpu:
    backend: ssh-slurm
    ssh_profile: lab-gpu
    remote_template_root: /opt/mlipflow/templates
    work_root: /scratch/myuser/mlipflow
    scheduler:
      partition_candidates:
        - gpu
        - compute
      memory_constraint: reported
```

A project node may select that profile and declares only abstract resources:

```yaml
- id: train-model
  uses: mlip-training@0
  backend: ssh-slurm
  backend_profile: lab-gpu
  resources:
    cpus: 8
    gpus: 1
    memory: 32GB
    walltime: "04:00:00"
  inputs: {}
  parameters: {}
```

If `backend_profile` is omitted, MLIPFlow reads one scheduler snapshot from each
site profile that declares `partition_candidates` and selects the cluster with
the most currently available nodes capable of the requested resources. An
explicit `backend_profile` always wins. Set `memory_constraint: unreported` only
for a site whose Slurm node records do not expose usable memory capacity; the
submitted template remains responsible for that site's memory policy.

The site-owned template library is responsible for details such as modules/conda activation, executable paths, launchers, and Slurm directives. This keeps projects portable across clusters and keeps private infrastructure details out of version-controlled research manifests.

`resources.walltime` accepts a reviewed `HH:MM:SS` value or the explicit literal
`UNLIMITED`. The selected Slurm partition can still impose its own maximum time.

### DFT labels to all training frameworks

A successful scheduled `dft-labeling.label` attempt collects a reusable
`canonical-labeled-dataset.json` with stable `record_id` values. A separate reviewed
`dft-labeling.dataset-assemble` node runs format conversion on the selected cluster and
first creates one deterministic or source-group-aware train/validation/test split,
then publishes split-preserving DeepMD, M3GNet/MatGL, CHGNet, and MACE directories
below the site-owned data root. The four small `*-dataset-reference.json` files are
direct inputs to `mlip-training`.

The converter preserves stable record IDs, source paths, and atom order, records every
stress/unit transformation, uses dpdata for DeepMD, and refuses silent overwrite.
Every framework train/validation/test file is selected from the same `split.json`
record IDs, so the held-out test set is directly comparable. There is no bundle tar,
verification archive, or user-authored data conversion command in this
path. See [`examples/training_all_models/CLUSTER.md`](examples/training_all_models/CLUSTER.md)
for the site template and workflow YAML.

The site schema is [`schemas/site.schema.json`](schemas/site.schema.json); concrete template examples are in [`examples/site_templates/`](examples/site_templates/).
For step-by-step creation, validation, and `PYTHON_BIN` binding of isolated DeepMD,
MACE, CHGNet, and MatGL execution environments, see
[`docs/CLUSTER_ENVIRONMENTS.md`](docs/CLUSTER_ENVIRONMENTS.md).

## Recommended environment layout

Use one general MLIPFlow local environment plus separate execution environments only
for the MLIP frameworks you actually run:

```text
mlipflow-local        -> mlipflow[local] (control/science; no MLIP frameworks)
deepmd-exec           -> DeepMD-kit + project-specific dependencies, when needed
mace-exec             -> MACE + PyTorch stack, when needed
matgl-exec            -> MatGL/M3GNet stack, when needed
chgnet-exec           -> CHGNet stack, when needed
site programs         -> VASP / LAMMPS / LASP / MPI / scheduler modules
```

MLIPFlow records and checks the execution contract, but the scientific environment remains yours to select and validate. For production calculations, record framework versions, model/data paths, units, seeds, device/precision choices, and the scientific parameters that matter to your result.

## Extending MLIPFlow

There are three intentionally separate extension layers.

### 1. Add a scientific plugin

```text
plugins/my-plugin/
  plugin.yaml
  adapter.py
```

`plugin.yaml` describes the public contract. An adapter implements deterministic lifecycle hooks such as:

```python
class Adapter:
    def validate(self, context): ...
    def plan(self, context): ...
    def prepare(self, context, plan): ...
    def check(self, context): ...
    def collect(self, context): ...
    def replay(self, context): ...
```

Use [`schemas/plugin.schema.json`](schemas/plugin.schema.json) and [`docs/PLUGIN_DEVELOPMENT.md`](docs/PLUGIN_DEVELOPMENT.md) when adding a plugin. Plugins may wrap existing scientific codes; you do not need to reimplement the numerical method inside MLIPFlow.

### 2. Add an Agent Skill

Add `.agents/skills/<name>/SKILL.md` when you want an agent-facing workflow guide. Skills should explain **when and how to supervise** a capability, while delegating deterministic computation to plugins and the CLI.

### 3. Add a site template family

Add site-owned Slurm templates when a plugin needs a new execution environment, launcher, CPU/GPU layout, or cluster-specific software stack. Projects continue to refer only to a profile, resources, and scientific inputs.

## Model registry and evidence-driven routing

A project can point to `model_registry.yaml` and define routing policies over benchmark metrics. The CLI then ranks only evidence that satisfies the policy and evidence requirements:

```bash
mlipflow --project . route \
  --task ionic-transport \
  --elements Li P S Cl \
  --scenario production-md
```

This keeps model choice reviewable and task-specific: the model that is best for static PES error does not have to be the model selected for transport or voltage work.

See [`examples/high_entropy_sulfide/model_registry.yaml`](examples/high_entropy_sulfide/model_registry.yaml) and [`docs/TASKFLOW_REFERENCE.md`](docs/TASKFLOW_REFERENCE.md).

## Documentation map

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — core architecture and module boundaries.
- [`docs/TASKFLOW_REFERENCE.md`](docs/TASKFLOW_REFERENCE.md) — workflow lifecycle and CLI behavior.
- [`docs/PLUGIN_DEVELOPMENT.md`](docs/PLUGIN_DEVELOPMENT.md) — writing scientific plugins.
- [`docs/AGENT_SKILLS.md`](docs/AGENT_SKILLS.md) — bundled Agent Skill contracts.
- [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md) — current implementation and validation matrix.
- [`docs/SCIENTIFIC_VALIDATION.md`](docs/SCIENTIFIC_VALIDATION.md) — numerical/scientific validation evidence.
- [`docs/HPC_VALIDATION.md`](docs/HPC_VALIDATION.md) — scheduler and real-cluster validation.
- [`docs/MANUSCRIPT_REPRODUCTION.md`](docs/MANUSCRIPT_REPRODUCTION.md) — manuscript evidence-reproduction workflow.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — development and contribution guide.
- [`SECURITY.md`](SECURITY.md) — vulnerability reporting and security model.
- [`CHANGELOG.md`](CHANGELOG.md) — notable user-facing changes by release.

## Project maturity

MLIPFlow is currently an alpha project with a completed software core and substantial local/scientific integration coverage. Real-HPC scientific validation is intentionally tracked per capability rather than summarized as a single "production ready" flag. This makes it possible to use the parts that are validated for your environment without overstating what has been exercised elsewhere.

If you are evaluating MLIPFlow for a new cluster or production study, start with `mlipflow doctor`, a replay or bounded smoke workflow, and the validation documents above before scaling the same contract to expensive work.

## Contributing

Contributions are welcome across workflow core, plugins, scientific validation, examples, Agent Skills, and site-template patterns. Start with [`CONTRIBUTING.md`](CONTRIBUTING.md) and keep changes small enough that their execution and scientific consequences can be reviewed.

## License and citation

MLIPFlow is licensed under the [Apache License 2.0](LICENSE). See [`NOTICE`](NOTICE) for project and third-party boundary information.

If MLIPFlow contributes to published research, cite the software version or commit using [`CITATION.cff`](CITATION.cff) and also cite the scientific methods, models, datasets, and external codes used by the plugins you executed.
