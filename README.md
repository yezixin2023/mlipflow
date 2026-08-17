<p align="center">
  <img src="docs/assets/mlipflow-logo.png" alt="MLIPFlow logo" width="720">
</p>

<h1 align="center">MLIPFlow</h1>

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
- **Replay and provenance.** Reuse existing scientific evidence without pretending it was recomputed, while retaining source identity and run provenance.

## Quick start

### 1. Install from source

```bash
git clone https://github.com/yezixin2023/mlipflow.git
cd mlipflow

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

mlipflow --version
```

Useful extras:

```bash
# Test, lint, schema validation, and development tools
python -m pip install -e ".[dev]"

# NumPy / ASE / icet / pandas / Plotly helpers used by scientific workflows
python -m pip install -e ".[science]"

# pymatgen support for VASP input preparation
python -m pip install -e ".[dft]"

# A practical development + science environment
python -m pip install -e ".[dev,science,dft]"
```

MLIP frameworks and external scientific programs such as DeepMD-kit, MACE, MatGL/M3GNet, CHGNet, LAMMPS, VASP, and LASP are **not forced into the MLIPFlow control environment**. Install the frameworks you actually execute in their own validated environments or provide them through your HPC site templates.

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
| `mlipflow run NODE --approve TOKEN` | Execute approval-gated work using the reviewed plan digest. |
| `mlipflow advance` | Observe and reconcile state; it does not launch new workflow nodes. |
| `mlipflow retry NODE` | Create a fresh retry attempt while preserving lineage. |
| `mlipflow stop NODE` | Explicitly stop/cancel a running node when the backend supports it. |

Add `--format json` for structured output and `--audit` when you need detailed planning and provenance information.

A typical expensive run looks like this:

```bash
mlipflow --project . run train-model --dry-run --audit
# Review command, inputs, backend, resources, staged files, and plan_digest.
mlipflow --project . run train-model --approve <PLAN_DIGEST>
mlipflow --project . advance
mlipflow --project . status train-model
```

## Bundled scientific capabilities

MLIPFlow currently ships ten scientific plugins. Plugin manifests are the source of truth for exact operations, inputs, outputs, backend support, dependencies, and completion checks.

| Plugin | What it does | Typical execution |
|---|---|---|
| `high-entropy-structure` | Seeded high-entropy/SQS structure generation with ASE + icet. | Local |
| `pes-sampling` | DIRECT representative selection, LASP/SSW execution, and historical archive normalization/replay. | Local; controlled SSH-Slurm for scheduled LASP execution |
| `dft-labeling` | Prepare VASP static/relax/AIMD inputs and execute/collect labeling through an explicit contract. | Local; controlled SSH-Slurm for supported static labeling |
| `mlip-training` | Train or fine-tune DeepMD, M3GNet/MatGL, CHGNet, and MACE models. | Local and controlled SSH-Slurm contracts |
| `ase-md` | Run single-temperature ASE NVT Langevin or isotropic MTK NPT with explicit models and checkpoint/restart. | Controlled SSH-Slurm |
| `lammps-md` | Prepare LAMMPS MLIP decks and execute NVT/NPT with restart-aware scheduled runs. | Local prepare; controlled SSH-Slurm execute |
| `mlip-benchmark` | Normalize/evaluate prediction evidence and produce canonical metrics and rankings. | Local |
| `ionic-transport` | Analyze existing ASE/LAMMPS/VASP trajectories or MSD data into diffusion, conductivity, and Arrhenius results. | Local |
| `candidate-ranking` | Deterministic ranking/top-k selection from existing candidate and metric manifests. | Local |
| `electrochemical-voltage` | Convert explicit total-energy sequences to voltage or replay reported voltage evidence. | Local |

The software contracts for these plugins are implemented and tested, but scientific validation is deliberately reported separately from software completeness. For the exact current validation level—including which scheduled paths have been exercised on a real cluster—see [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md), [`docs/SCIENTIFIC_VALIDATION.md`](docs/SCIENTIFIC_VALIDATION.md), and [`docs/HPC_VALIDATION.md`](docs/HPC_VALIDATION.md).

## Bundled Agent Skills

Agent Skills live in [`.agents/skills/`](.agents/skills/). They are **supervision guides**: they help an agent choose operations, request the right evidence, call MLIPFlow, and interpret structured results. The scientific computation remains in the plugins and external tools.

| Skill | Use it for |
|---|---|
| `$mlip-workflow` | Plan and supervise an end-to-end MLIP workflow. |
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

Each retry gets a new attempt. MLIPFlow records the plan and relevant identities used for that attempt so later project edits do not silently rewrite historical execution state.

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
```

A project node selects that profile and declares only abstract resources:

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

The site-owned template library is responsible for details such as modules/conda activation, executable paths, launchers, and Slurm directives. This keeps projects portable across clusters and keeps private infrastructure details out of version-controlled research manifests.

The site schema is [`schemas/site.schema.json`](schemas/site.schema.json); concrete template examples are in [`examples/site_templates/`](examples/site_templates/).

## Recommended environment layout

For real research, use small, purpose-specific environments rather than one environment containing every MLIP framework:

```text
mlipflow-control      -> mlipflow + PyYAML (+ optional science/dft extras)
deepmd-exec           -> DeepMD-kit + project-specific dependencies
mace-exec             -> MACE + PyTorch stack
matgl-exec            -> MatGL/M3GNet stack
chgnet-exec           -> CHGNet stack
site programs         -> VASP / LAMMPS / LASP / MPI / scheduler modules
```

MLIPFlow records and checks the execution contract, but the scientific environment remains yours to pin and validate. For production calculations, record framework versions, model/data identities, units, seeds, device/precision choices, and the scientific parameters that matter to your result.

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
