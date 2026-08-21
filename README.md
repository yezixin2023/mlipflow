<p align="center">
  <img src="docs/assets/mlipflow-logo.png" alt="MLIPFlow logo" width="550">
</p>

<p align="center"><strong>Deterministic, auditable workflows for machine-learned interatomic-potential research.</strong></p>

MLIPFlow turns structure generation, sampling, DFT labeling, model training, molecular dynamics, benchmarking, transport analysis, active learning, and candidate selection into an explicit, versioned workflow. It keeps execution plans, approvals, attempts, artifacts, checks, and provenance inspectable across local machines and Slurm clusters.

MLIPFlow is a **workflow layer**, not a new interatomic-potential framework. It coordinates the scientific codes and models selected by the researcher while preserving their inputs, versions, execution boundaries, and evidence.

**Python:** core `>=3.9`; complete local environment `>=3.10` · **License:** Apache-2.0

## Why MLIPFlow

- **One research DAG.** Describe scientific stages and dependencies in `project.yaml` instead of maintaining disconnected shell scripts.
- **Review before execution.** A dry run exposes the exact command, backend, resources, staged inputs, templates, and expected outputs before expensive work is approved.
- **Immutable attempt history.** Retries create fresh attempts; previous results, scheduler records, logs, and failure evidence are retained.
- **Local and HPC execution.** Scientific plugins use a common contract while cluster-specific paths, modules, launchers, and partitions remain in user-owned site profiles and templates.
- **Verified artifact handoff.** Downstream nodes consume checked upstream artifacts directly, including canonical DFT datasets, AIMD trajectories, trained-model references, and restart-aware MD segments.
- **Evidence-based decisions.** Model routing, benchmark comparison, active-learning selection, and candidate ranking operate on explicit machine-readable evidence.
- **Agent-compatible supervision.** Bundled Agent Skills help compatible agents plan and supervise workflows; deterministic scientific work remains in plugins and external programs.

## Quick start

### 1. Install

The recommended complete local environment includes the workflow core, scientific helpers, VASP input preparation, and formal transport analysis:

```bash
git clone https://github.com/yezixin2023/mlipflow.git
cd mlipflow

python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ".[local]"

mlipflow --version
```

`.[local]` requires Python 3.10 or newer because formal transport analysis uses `pymatgen-analysis-diffusion`. Python 3.9 remains supported for the workflow core and selected extras; see [Installation and environments](#installation-and-environments).

MLIP frameworks such as DeepMD-kit, MACE, CHGNet, and MatGL/M3GNet are intentionally not installed by `.[local]`. Install only the frameworks you execute, preferably in isolated local or site-owned cluster environments.

### 2. Create a project

```bash
mlipflow init my-project
cd my-project
mlipflow doctor
```

`mlipflow init` creates a minimal `project.yaml` and initializes local workflow state. Add nodes, inputs, parameters, dependencies, and routing policies before running scientific work.

### 3. Run a bundled replay example

From the repository root, copy the example so the checked-in fixture remains unchanged:

```bash
python -c "from shutil import copytree; copytree('examples/high_entropy_sulfide', 'mlipflow-replay-demo')"

mlipflow --project mlipflow-replay-demo init
mlipflow --project mlipflow-replay-demo list
mlipflow --project mlipflow-replay-demo run structure-replay --dry-run
mlipflow --project mlipflow-replay-demo run structure-replay
mlipflow --project mlipflow-replay-demo advance
mlipflow --project mlipflow-replay-demo status
```

This example consumes bundled result manifests and does not launch an expensive scientific program. Additional examples cover DFT-to-training dataset assembly, multi-framework training, ASE and LAMMPS MD, active learning, LASP/SSW, cluster templates, and AIMD-reference validation under [`examples/`](examples/).

## Core workflow model

MLIPFlow separates inspection, planning, execution, and reconciliation:

| Command | Purpose |
|---|---|
| `mlipflow init [PATH]` | Create or initialize a project and its local state database. |
| `mlipflow doctor` | Check project, plugin, dependency, and site configuration without mutation. |
| `mlipflow list`, `status`, `json` | Inspect workflow state in human-readable or machine-readable form. |
| `mlipflow inspect NODE`, `logs NODE` | Review a node contract, attempt details, and collected logs. |
| `mlipflow run NODE --dry-run --audit` | Build the exact execution plan without launching work. |
| `mlipflow run NODE --approve` | Execute approval-gated work after reviewing the plan. |
| `mlipflow advance` | Observe submitted work, fetch bounded outputs, run scientific checks, and reconcile state. It never launches a new node. |
| `mlipflow retry NODE` | Create a fresh attempt while preserving prior lineage. |
| `mlipflow stop NODE` | Cancel or stop a running node when the backend supports it. |
| `mlipflow route ...` | Rank compatible models from versioned benchmark evidence and project policy. |

A typical external or expensive node follows this sequence:

```bash
mlipflow --project . run train-model --dry-run --audit
mlipflow --project . run train-model --approve
mlipflow --project . advance
mlipflow --project . status train-model
```

Scheduler completion is not treated as scientific success. A node reaches `OK` only after its plugin verifies the declared outputs and completion criteria.

## Scientific capabilities and Agent Skills

Plugin manifests define deterministic operations, inputs, outputs, checks, backends, and retry behavior. Agent Skills provide supervision guidance for compatible tool-using agents; they do not replace plugin code or numerical software.

| Research capability | Plugin | Agent Skill | Main scope | Execution |
|---|---|---|---|---|
| End-to-end workflow design | MLIPFlow core | `$mlip-workflow` | DAG construction, evidence handoff, approval boundaries, and workflow supervision | Supervision |
| High-entropy and SQS structures | `high-entropy-structure` | `$high-entropy-structure` | Seeded `icet` SQS generation with explicit composition and supercell contracts | Local |
| PES sampling | `pes-sampling` | `$pes-sampling` | DIRECT selection, LASP input preparation, LASP/SSW execution or replay, and structure-set merging | Local; SSH-Slurm for scheduled LASP |
| DFT labeling and dataset assembly | `dft-labeling` | `$dft-labeling` | VASP input preparation; bounded static, relax, and AIMD labeling; canonical labels; shared-split DeepMD, M3GNet, CHGNet, and MACE datasets | Local preparation; local or SSH-Slurm labeling; SSH-Slurm assembly |
| MLIP training and fine-tuning | `mlip-training` | `$mlip-training` | DeepMD, M3GNet/MatGL, CHGNet, and MACE training or fine-tuning with explicit model and dataset references | Local; SSH-Slurm |
| ASE molecular dynamics | `ase-md` | `$ase-md` | Single-temperature NVT Langevin or isotropic MTK NPT with explicit models, supercells, checkpoints, and fresh-attempt restart | SSH-Slurm |
| LAMMPS molecular dynamics | `lammps-md` | `$lammps-md` | Deterministic input preparation, DeepMD/MACE/MatGL execution, bounded output collection, and binary-restart recovery | Local preparation; SSH-Slurm execution |
| MLIP benchmarking | `mlip-benchmark` | `$mlip-benchmark` | Fresh or normalized static-PES evidence plus task-separated AIMD-reference RDF and transport metrics | Local; SSH-Slurm for fresh inference |
| Offline active learning | `active-learning` | `$mlip-active-learning` | Calibrated one- or two-model committees, risk-union candidate selection, and immutable round assessment | Local decisions; SSH-Slurm committee inference |
| Ionic transport and structural dynamics | `ionic-transport` | `$ionic-transport` | Trajectory/MSD analysis, diffusion, conductivity, Haven ratio, Arrhenius fitting, and bounded AIMD-versus-MLIP partial RDF comparison | Local |
| Candidate ranking | `candidate-ranking` | `$candidate-ranking` | Deterministic top-k selection from an existing finite numeric metric | Local |
| Electrochemical voltage | `electrochemical-voltage` | — | Adjacent average Li intercalation voltages from explicit total energies and deterministic evidence replay | Local |

The voltage plugin is orchestrated directly or through `$mlip-workflow`; it does not currently have a dedicated Agent Skill. Exact operations and limitations are defined by the manifests under [`plugins/`](plugins/) and the corresponding files under [`.agents/skills/`](.agents/skills/).

A common research path is:

```text
structures -> PES sampling -> DFT labels -> canonical dataset -> MLIP training
                                                        |            |
                                                        |            +-> benchmark -> model routing
                                                        |                              |
                                                        +-> active-learning rounds     +-> MD -> transport/RDF
```

## Direct artifact handoff

MLIPFlow passes checked producer artifacts to declared downstream nodes without requiring users to rename or manually relocate internal files. Current handoffs include:

- verified DFT labels to one shared train/validation/test split and framework-specific dataset views;
- final checked AIMD `vasprun.xml` trajectories directly to ionic-transport analysis;
- trained-model references to benchmarking and MD nodes without copying site-owned model weights through the control plane;
- ASE and LAMMPS trajectory segments across restart attempts, ordered by global step with repeated boundaries removed for transport analysis;
- AIMD-versus-MLIP RDF and transport comparison evidence to task-separated benchmark normalization.

Every handoff retains the producing node, attempt, artifact role, path identity, scientific parameters, and completion status.

## Installation and environments

| Install target | Command | Intended use |
|---|---|---|
| Core | `python -m pip install .` | Workflow configuration, state, replay, inspection, and plugins without optional scientific Python dependencies. |
| Scientific helpers | `python -m pip install ".[science]"` | NumPy, ASE, `icet`, pandas, and Plotly helpers. |
| DFT preparation and datasets | `python -m pip install ".[dft]"` | `pymatgen`, `dpdata`, and ASE support for VASP preparation and dataset conversion. |
| Formal transport | `python -m pip install ".[transport]"` | `pymatgen-analysis-diffusion` and its direct analysis dependencies; Python 3.10 or newer. |
| Complete local environment | `python -m pip install ".[local]"` | Supported union of `science`, `dft`, and `transport`; Python 3.10 or newer. |
| Development | `python -m pip install -e ".[dev]"` | Tests, schema validation, linting, and development dependencies. |

Keep accelerator-heavy MLIP frameworks in isolated execution environments when their PyTorch, CUDA, or framework requirements conflict. Cluster templates should bind the exact interpreter, executable, modules, model root, and dataset root used for each scientific node. See [`docs/CLUSTER_ENVIRONMENTS.md`](docs/CLUSTER_ENVIRONMENTS.md).

## HPC configuration

Projects declare only a named backend profile and abstract resources such as CPUs, GPUs, memory, and wall time. Private infrastructure belongs in the user-local site configuration, normally:

```text
~/.mlipflow/site.yaml
```

Site profiles and site-owned templates contain SSH aliases, partitions, accounts, module or environment activation, executable paths, launchers, canonical data/model roots, and remote work roots. This separation keeps `project.yaml` portable and prevents cluster-specific or sensitive details from entering research repositories.

Use `mlipflow doctor` before a new cluster workflow, inspect the dry-run plan, and begin with a replay or bounded smoke case before scaling the same contract. Configuration details and template examples are in [`docs/CLUSTER_ENVIRONMENTS.md`](docs/CLUSTER_ENVIRONMENTS.md) and [`examples/site_templates/`](examples/site_templates/).

## Scientific validation

Software implementation, workflow completion, numerical agreement, and real-cluster validation are distinct claims. MLIPFlow reports validation per capability rather than inferring scientific reliability from package installation or scheduler success.

Before relying on a capability for a study, review:

- [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md) for implemented operations and current limitations;
- [`docs/SCIENTIFIC_VALIDATION.md`](docs/SCIENTIFIC_VALIDATION.md) for numerical and scientific evidence;
- [`docs/HPC_VALIDATION.md`](docs/HPC_VALIDATION.md) for scheduler and real-cluster evidence.

Researchers remain responsible for validating the selected DFT settings, models, datasets, simulation parameters, convergence, and uncertainty for their scientific system.

## Documentation

Start with the [`documentation index`](docs/README.md). It separates user guides, HPC setup, extension references, validation evidence, and manuscript-specific reproduction material.

Key references include:

- [`docs/AGENT_SKILLS.md`](docs/AGENT_SKILLS.md) — detailed Agent Skill contracts;
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — architecture and trust boundaries;
- [`docs/PLUGIN_DEVELOPMENT.md`](docs/PLUGIN_DEVELOPMENT.md) — adding or maintaining scientific plugins;
- [`schemas/`](schemas/) — machine-readable project, site, plugin, and artifact schemas;
- [`examples/`](examples/) — executable or replayable project patterns.

## Contributing, security, and citation

Contributions are welcome across the workflow core, plugins, schemas, tests, examples, Agent Skills, site-template patterns, and validation evidence. See [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a change.

Report security issues according to [`SECURITY.md`](SECURITY.md). Do not commit credentials, private cluster details, licensed pseudopotentials, model weights, large trajectories, or unpublished data.

MLIPFlow is licensed under the [Apache License 2.0](LICENSE). If it contributes to published research, cite the software version or commit using [`CITATION.cff`](CITATION.cff), together with the scientific methods, datasets, models, and external codes used by the executed plugins.
