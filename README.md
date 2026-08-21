<p align="center">
  <img src="docs/assets/mlipflow-logo.png" alt="MLIPFlow logo" width="550">
</p>

<p align="center"><strong>Deterministic, auditable workflows for machine-learned interatomic-potential research.</strong></p>

MLIPFlow connects structure generation, sampling, DFT labeling, dataset assembly, MLIP training, molecular dynamics, benchmarking, active learning, transport analysis, and candidate selection in one versioned workflow. Plans, approvals, attempts, artifacts, checks, and provenance remain inspectable across local machines and Slurm clusters.

MLIPFlow is a **workflow layer**, not a new interatomic-potential framework. Researchers choose the scientific codes, models, datasets, and numerical settings; MLIPFlow coordinates them without hiding execution boundaries or evidence.

**Python:** core `>=3.9`; complete local environment `>=3.10` · **License:** Apache-2.0

## What MLIPFlow provides

- **Explicit research DAGs:** scientific stages and dependencies live in `project.yaml`, not disconnected shell scripts.
- **Review before execution:** dry runs expose commands, backends, resources, staged inputs, templates, and expected outputs before expensive work is approved.
- **Immutable attempts:** retries create fresh attempts while prior results, logs, scheduler records, and failures remain available.
- **Local and HPC execution:** plugins share one contract; private cluster details stay in user-owned site profiles and templates.
- **Verified handoff:** checked DFT labels, AIMD trajectories, model references, and restart-aware MD segments flow directly to downstream nodes.
- **Evidence-based decisions:** routing, benchmarking, active-learning selection, and ranking consume machine-readable evidence rather than hard-coded model preferences.
- **Optional agent supervision:** bundled Agent Skills help compatible agents plan and supervise work; deterministic scientific operations remain in plugins and external programs.

## Quick start

### Install

```bash
git clone https://github.com/yezixin2023/mlipflow.git
cd mlipflow

python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ".[local]"

mlipflow --version
```

`.[local]` includes the workflow core, scientific helpers, VASP input preparation, dataset conversion, and formal transport analysis. It requires Python 3.10 or newer because `pymatgen-analysis-diffusion` does. Python 3.9 remains supported for the core and selected extras.

DeepMD-kit, MACE, CHGNet, and MatGL/M3GNet are not installed automatically. Install only the frameworks you execute, preferably in isolated local or site-owned cluster environments.

### Initialize a project

```bash
mlipflow init my-project
cd my-project
mlipflow doctor
```

`mlipflow init` creates a minimal `project.yaml` and initializes local state. Add workflow nodes, inputs, parameters, dependencies, and routing policy before launching scientific work.

### Run the bundled replay example

From the repository root:

```bash
python -c "from shutil import copytree; copytree('examples/high_entropy_sulfide', 'mlipflow-replay-demo')"

mlipflow --project mlipflow-replay-demo init
mlipflow --project mlipflow-replay-demo list
mlipflow --project mlipflow-replay-demo run structure-replay --dry-run
mlipflow --project mlipflow-replay-demo run structure-replay
mlipflow --project mlipflow-replay-demo advance
mlipflow --project mlipflow-replay-demo status
```

The example consumes bundled manifests and does not launch an expensive scientific program. Other examples cover DFT-to-training data assembly, multi-framework training, ASE and LAMMPS MD, active learning, LASP/SSW, site templates, and AIMD-reference validation under [`examples/`](examples/).

## Workflow lifecycle

| Command | Purpose |
|---|---|
| `mlipflow doctor` | Check project, plugin, dependency, and site configuration without mutation. |
| `mlipflow list`, `status`, `json` | Inspect workflow state in human- or machine-readable form. |
| `mlipflow inspect NODE`, `logs NODE` | Review a node contract, attempts, diagnostics, and collected logs. |
| `mlipflow run NODE --dry-run --audit` | Build the exact execution plan without launching work. |
| `mlipflow run NODE --approve` | Execute approval-gated work after reviewing the plan. |
| `mlipflow advance` | Observe submitted work, fetch bounded outputs, run checks, and reconcile state; it never launches a new node. |
| `mlipflow retry NODE`, `stop NODE` | Create a fresh attempt or explicitly stop supported running work. |
| `mlipflow route ...` | Rank compatible models from versioned benchmark evidence and project policy. |

A typical external or expensive node is operated as follows:

```bash
mlipflow --project . run train-model --dry-run --audit
mlipflow --project . run train-model --approve
mlipflow --project . advance
mlipflow --project . status train-model
```

Scheduler completion is not scientific success. A node reaches `OK` only after its plugin verifies declared outputs and completion criteria.

## Scientific capabilities and Agent Skills

Plugin manifests define deterministic operations, inputs, outputs, checks, backends, and retry behavior. Agent Skills provide supervision guidance; they do not replace numerical software or plugin code.

| Research capability | Plugin | Agent Skill | Main scope | Execution |
|---|---|---|---|---|
| End-to-end workflow design | MLIPFlow core | `$mlip-workflow` | DAG construction, evidence handoff, approval boundaries, and supervision | Supervision |
| High-entropy and SQS structures | `high-entropy-structure` | `$high-entropy-structure` | Seeded `icet` SQS generation with explicit composition and supercell contracts | Local |
| PES sampling | `pes-sampling` | `$pes-sampling` | DIRECT selection, LASP preparation, LASP/SSW execution or replay, and structure-set merging | Local; SSH-Slurm for scheduled LASP |
| DFT labeling and datasets | `dft-labeling` | `$dft-labeling` | VASP preparation; bounded static, relax, and AIMD labeling; canonical labels; shared-split framework datasets | Local preparation; local or SSH-Slurm labeling; SSH-Slurm assembly |
| MLIP training and fine-tuning | `mlip-training` | `$mlip-training` | DeepMD, M3GNet/MatGL, CHGNet, and MACE training or fine-tuning with explicit references | Local; SSH-Slurm |
| ASE molecular dynamics | `ase-md` | `$ase-md` | NVT Langevin or isotropic MTK NPT, supercells, checkpoints, and fresh-attempt restart | SSH-Slurm |
| LAMMPS molecular dynamics | `lammps-md` | `$lammps-md` | Deterministic input preparation, DeepMD/MACE/MatGL execution, collection, and binary restart | Local preparation; SSH-Slurm execution |
| MLIP benchmarking | `mlip-benchmark` | `$mlip-benchmark` | Fresh or normalized static-PES evidence plus task-separated AIMD-reference RDF and transport metrics | Local; SSH-Slurm for fresh inference |
| Offline active learning | `active-learning` | `$mlip-active-learning` | Calibrated one- or two-model committees, risk-union selection, and immutable round assessment | Local decisions; SSH-Slurm inference |
| Ionic transport and dynamics | `ionic-transport` | `$ionic-transport` | Trajectory/MSD analysis, diffusion, conductivity, Haven ratio, Arrhenius fitting, and bounded partial-RDF comparison | Local |
| Candidate ranking | `candidate-ranking` | `$candidate-ranking` | Deterministic top-k selection from an existing finite numeric metric | Local |
| Electrochemical voltage | `electrochemical-voltage` | — | Adjacent average Li intercalation voltages and deterministic evidence replay | Local |

The voltage plugin is orchestrated directly or through `$mlip-workflow`; it does not currently have a dedicated Agent Skill. Exact operations and limitations are defined by manifests under [`plugins/`](plugins/) and files under [`.agents/skills/`](.agents/skills/).

## Artifact handoff and provenance

Current direct handoffs include:

- verified DFT labels to one shared train/validation/test split and framework-specific dataset views;
- checked AIMD `vasprun.xml` trajectories to ionic-transport analysis;
- trained-model references to benchmark and MD nodes without copying site-owned weights through the control plane;
- ordered ASE and LAMMPS trajectory segments across restart attempts, with repeated boundaries removed for analysis;
- AIMD-versus-MLIP RDF and transport evidence to task-separated benchmark normalization.

Each handoff retains the producing node, attempt, artifact role, path identity, scientific parameters, and completion status.

## Environments and HPC

| Install target | Command | Intended use |
|---|---|---|
| Core | `python -m pip install .` | Workflow configuration, state, replay, inspection, and plugins. |
| Scientific helpers | `python -m pip install ".[science]"` | NumPy, ASE, `icet`, pandas, and Plotly helpers. |
| DFT and datasets | `python -m pip install ".[dft]"` | `pymatgen`, `dpdata`, and ASE support for VASP preparation and conversion. |
| Formal transport | `python -m pip install ".[transport]"` | `pymatgen-analysis-diffusion` and direct analysis dependencies; Python `>=3.10`. |
| Complete local | `python -m pip install ".[local]"` | Supported union of `science`, `dft`, and `transport`; Python `>=3.10`. |
| Development | `python -m pip install -e ".[dev]"` | Tests, schema validation, linting, and development dependencies. |

Projects declare a named backend profile and abstract resources such as CPUs, GPUs, memory, and wall time. SSH aliases, partitions, accounts, modules, executables, launchers, canonical data/model roots, and remote work roots belong in the user-local site configuration, normally `~/.mlipflow/site.yaml`, and site-owned templates. This keeps `project.yaml` portable and private infrastructure out of research repositories.

Review [`docs/CLUSTER_ENVIRONMENTS.md`](docs/CLUSTER_ENVIRONMENTS.md) and [`examples/site_templates/`](examples/site_templates/) before using a new cluster. Start with `mlipflow doctor`, inspect the dry-run plan, and validate a replay or bounded smoke case before scaling the same contract.

## Validation and documentation

Software implementation, workflow completion, numerical agreement, and real-cluster validation are different claims. Before relying on a capability, review:

- [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md) — implemented operations and limitations;
- [`docs/SCIENTIFIC_VALIDATION.md`](docs/SCIENTIFIC_VALIDATION.md) — numerical and scientific evidence;
- [`docs/HPC_VALIDATION.md`](docs/HPC_VALIDATION.md) — scheduler and real-cluster evidence.

Researchers remain responsible for validating DFT settings, models, datasets, simulation parameters, convergence, and uncertainty for their system.

The [`documentation index`](docs/README.md) organizes user guidance, HPC setup, extension references, validation evidence, and manuscript-specific reproduction material. Core references include [`ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`AGENT_SKILLS.md`](docs/AGENT_SKILLS.md), [`CLUSTER_ENVIRONMENTS.md`](docs/CLUSTER_ENVIRONMENTS.md), and [`PLUGIN_DEVELOPMENT.md`](docs/PLUGIN_DEVELOPMENT.md).

## Contributing, security, and citation

Contributions are welcome across the core, plugins, schemas, tests, examples, Agent Skills, site templates, and validation evidence. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

Report security issues according to [`SECURITY.md`](SECURITY.md). Do not commit credentials, private cluster details, licensed pseudopotentials, model weights, large trajectories, or unpublished data.

MLIPFlow is licensed under the [Apache License 2.0](LICENSE). For published research, cite the software version or commit using [`CITATION.cff`](CITATION.cff), together with the scientific methods, datasets, models, and external codes used by the executed plugins.
