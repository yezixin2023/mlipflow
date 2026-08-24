<p align="center">
  <img src="docs/assets/mlipflow-logo.png" alt="MLIPFlow logo" width="500">
</p>
<p align="center"><strong>Deterministic workflows for machine-learned interatomic-potential research.</strong></p>

MLIPFlow connects structure generation, sampling, DFT labeling, dataset assembly, MLIP training, molecular dynamics, benchmarking, active learning, transport analysis, and candidate selection in one explicit workflow. Attempts, scheduler jobs, logs, and scientific results remain inspectable across local machines and Slurm clusters.

MLIPFlow is a **workflow layer**, not a new interatomic-potential framework. Researchers choose the scientific codes, models, datasets, and numerical settings; MLIPFlow coordinates them without hiding execution boundaries or evidence.

**Python:** core `>=3.9`; complete local environment `>=3.10` · **License:** Apache-2.0

## What MLIPFlow provides

- **Explicit research workflows:** scientific stages and dependencies live in `project.yaml`, not disconnected shell scripts.
- **Review before execution:** dry runs expose commands, backends, resources, staged inputs, templates, expected outputs, and the effective operation-level approval requirement.
- **Immutable attempts:** retries create fresh attempts while prior results, logs, scheduler records, and failures remain available.
- **Local and HPC execution:** built-in capability adapters support local and SSH-Slurm work; private cluster details stay in user-owned site profiles and templates.
- **Verified handoff:** checked DFT labels, AIMD trajectories, model references, and restart-aware MD segments flow directly to downstream nodes.
- **Evidence-based decisions:** routing, benchmarking, active-learning selection, and ranking consume machine-readable evidence rather than hard-coded model preferences.
- **Optional agent supervision:** bundled Agent Skills help compatible agents plan and supervise work; deterministic scientific operations remain in built-in adapters and external programs.

## Quick start

### Install

```bash
git clone https://github.com/yezixin2023/mlipflow.git
cd mlipflow

python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install ".[local]"

mlipflow --version
```

`.[local]` includes the workflow core, scientific helpers, VASP input preparation, dataset conversion, and formal transport analysis. It requires Python 3.10 or newer because `pymatgen-analysis-diffusion` does. Python 3.9 remains supported for the core and selected extras. Selective dependency extras (`science`, `dft`, `transport`) are available for lightweight or specialized installations.

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

The example consumes bundled scientific results and does not launch an expensive scientific program. Other examples cover DFT-to-training data assembly, multi-framework training, ASE and LAMMPS MD, active learning, LASP/SSW, site templates, and AIMD-reference validation under [`examples/`](examples/).


## Scientific capabilities and Agent Skills

MLIPFlow ships a fixed set of scientific capability adapters. Each adapter validates inputs, plans execution, checks scientific completion, and collects results. MLIPFlow core owns replay, DAG progression, and fresh-attempt retry semantics. Agent Skills provide supervision guidance; they do not replace numerical software or adapter code.

| Research capability | Capability ID | Agent Skill | Main scope | Execution |
|---|---|---|---|---|
| End-to-end workflow design | MLIPFlow core | `$mlip-workflow` | DAG construction, evidence handoff, approval boundaries, and supervision | Supervision |
| High-entropy and SQS structures | `high-entropy-structure` | `$high-entropy-structure` | Seeded `icet` SQS generation with explicit composition and supercell contracts | Local |
| PES sampling | `pes-sampling` | `$pes-sampling` | DIRECT-based structure selection, LASP/SSW sampling, replay of existing sampling runs, and structure-set merging | Local; SSH-Slurm for scheduled LASP |
| DFT labeling and datasets | `dft-labeling` | `$dft-labeling` | VASP preparation and static, relaxation, or AIMD labeling, followed by standardized labeled-dataset assembly | Local preparation/assembly; local or SSH-Slurm labeling/assembly |
| MLIP training and fine-tuning | `mlip-training` | `$mlip-training` | DeepMD, M3GNet/MatGL, CHGNet, and MACE training or fine-tuning with explicit references | Local; SSH-Slurm |
| ASE molecular dynamics | `ase-md` | `$ase-md` | NVT Langevin or isotropic MTK NPT, supercells, checkpoints, and fresh-attempt restart | SSH-Slurm |
| LAMMPS molecular dynamics | `lammps-md` | `$lammps-md` | Deterministic input preparation, DeepMD/MACE/MatGL execution, collection, and binary restart | Local preparation; SSH-Slurm execution |
| MLIP benchmarking | `mlip-benchmark` | `$mlip-benchmark` | Fresh or normalized static-PES evidence plus task-separated AIMD-reference RDF and transport metrics | Local; SSH-Slurm for fresh inference |
| Offline active learning | `active-learning` | `$mlip-active-learning` | Uncertainty/risk-based screening, DFT-label selection, and round-wise declared-domain coverage and independent-accuracy assessment | Local selection and assessment; SSH-Slurm for large-scale inference |
| Ionic transport and dynamics | `ionic-transport` | `$ionic-transport` | Trajectory/MSD analysis, diffusion, conductivity, Haven ratio, Arrhenius fitting, and bounded partial-RDF comparison | Local |
| Candidate ranking | `candidate-ranking` | `$candidate-ranking` | Ranking and top-k selection of candidates according to user-defined quantitative metrics | Local |
| Electrochemical voltage | `electrochemical-voltage` | — | Average Li intercalation voltage calculations between adjacent compositions, with support for replaying existing results | Local |

The voltage capability is orchestrated directly or through `$mlip-workflow`; it does not currently have a dedicated Agent Skill. Implementations live under [`plugins/`](plugins/), while supervision guidance lives under [`.agents/skills/`](.agents/skills/). `mlipflow inspect NODE` reports the node's resolved operation, effective approval requirement, and the operations supported by its built-in capability.

## Environments and HPC

Projects declare an optional backend profile and abstract resources such as CPUs, GPUs, memory, and wall time. If the profile is omitted, MLIPFlow selects a suitable configured cluster. SSH aliases, partitions, accounts, modules, executables, launchers, canonical data/model roots, and remote work roots belong in the user-local site configuration, normally `~/.mlipflow/site.yaml`, and site-owned templates. This keeps private infrastructure out of research repositories.

Review [`docs/CLUSTER_ENVIRONMENTS.md`](docs/CLUSTER_ENVIRONMENTS.md) and [`examples/site_templates/`](examples/site_templates/) before using a new cluster. Start with `mlipflow doctor`, inspect the dry-run plan, and validate a replay or bounded smoke case before scaling the same contract.

### Example agent requests

After initializing an MLIPFlow project, you can describe a scientific task directly to a compatible AI agent. The agent can select the appropriate bundled Skill, or you can explicitly invoke one with `$skill-name`.

These examples assume that the required local software and, when needed, an HPC site profile have already been configured. If a required scientific parameter is missing, the agent should ask for it rather than inventing a value.

**End-to-end workflow — `$mlip-workflow`**

> I already have VASP-labelled structures in this project. Train MACE and DeepMD using the same train/validation/test split, benchmark them on the same held-out structures, and reuse anything that has already been completed instead of rerunning DFT.

> I have structures but no labels yet. Prepare the missing DFT calculations on my configured CPU cluster, assemble the resulting dataset, train MACE on my GPU cluster, run MD with the trained model, and analyze Li-ion transport. Show me each expensive stage before submitting it.

**DFT labeling — `$dft-labeling`**

> Prepare VASP static calculations for the structures in `./selected_structures`, run them on my configured CPU cluster, and collect the converged energies, forces, and stresses into a reusable labeled dataset.

> Run VASP AIMD for these structures at 800 K using the settings already defined in my project, then make the completed trajectories available for ionic-transport analysis.

**MLIP training and fine-tuning — `$mlip-training`**

> Fine-tune my existing MACE foundation model using the assembled dataset in this project. Run it on the configured GPU cluster with one GPU and show me the training plan before submission.

> Train DeepMD, MACE, CHGNet, and M3GNet from the existing framework-specific dataset views while preserving the same train/validation/test split for all four models.

**PES sampling — `$pes-sampling`**

> Use DIRECT to select representative structures from my MD trajectories, then prepare the selected structure set for DFT labeling.

> Start from the final frame of this MD trajectory, prepare a LASP input, run SSW sampling on my configured CPU cluster, and merge the accepted LASP structures with the structures selected by DIRECT.

**LAMMPS molecular dynamics — `$lammps-md`**

> Run a 500 K NVT simulation for 500 ps with my published MACE model using LAMMPS on the configured GPU cluster. Write periodic restart files so the calculation can continue if it reaches the Slurm walltime limit.

> Prepare and run an NPT LAMMPS simulation using my DeepMD model. Use the temperature, pressure, timestep, and run length already declared in the project, and do not submit anything until I review the dry-run.

**ASE molecular dynamics — `$ase-md`**

> Run a 600 K Langevin NVT trajectory with my CHGNet model on the configured GPU cluster and save checkpoints so an interrupted job can resume from the previous attempt.

> Run isotropic NPT MD with my MACE model using the existing structure and model reference. Expand the structure with the supercell repeat already specified in the project and verify that the model provides stress before starting dynamics.

**Ionic transport — `$ionic-transport`**

> Analyze the completed 400, 500, 600, and 700 K Li-ion trajectories from this project. Calculate diffusion coefficients and ionic conductivity at each temperature, fit the Arrhenius activation energy, and extrapolate the conductivity to 300 K.

> Compare the completed AIMD and MACE trajectories at the same temperature using Li diffusion, conductivity, and the requested partial RDF. Reuse the trajectory metadata already collected by MLIPFlow instead of asking me to copy or rename files.

**MLIP benchmarking — `$mlip-benchmark`**

> Compare the trained models on the same held-out test set and report energy, force, and available stress errors. Do not create a new split.

> Run fresh benchmarks for the published MACE, DeepMD, CHGNet, and M3GNet models using the shared benchmark dataset, then give me a joint ranking only for metrics that are directly comparable across all models.

**Offline active learning — `$mlip-active-learning`**

> Run one offline active-learning round using my two trained committee members and the current candidate pool. Evaluate model disagreement, select the configurations that require DFT labeling within the declared budget, and prepare the next cumulative training dataset.

> Resume the existing active-learning campaign from the latest completed round. Reuse the existing calibration, audit, labels, models, and trajectories where valid, and tell me whether the evidence supports another round before launching new expensive calculations.

**High-entropy / SQS structures — `$high-entropy-structure`**

> Generate SQS candidates from this prototype using the compositions, supercell, cluster cutoffs, search steps, and seed already defined in my project. Keep the generated structures as candidates only; do not rank them without property results.

**Candidate ranking — `$candidate-ranking`**

> Rank the existing candidates by formation energy, lower is better, and return the top 20. Use only the metric values already present in the project and report any candidates that are missing the requested metric.

Explicit Skill invocation is optional. For example:

> `$mlip-training` Fine-tune MACE using the existing assembled dataset on my configured GPU cluster.

Without the `$mlip-training` prefix, a compatible agent may infer the appropriate Skill from the scientific request.

## Workflow lifecycle

| Command | Purpose |
|---|---|
| `mlipflow doctor` | Check project, built-in capability, state, and site configuration without mutation. |
| `mlipflow list`, `status`, `json` | Inspect workflow state in human- or machine-readable form. |
| `mlipflow inspect NODE`, `logs NODE` | Review a node, its capability, current state, diagnostics, and collected logs. |
| `mlipflow run NODE --dry-run` | Build the execution plan without launching work. |
| `mlipflow run NODE --approve` | Execute approval-gated work after reviewing the plan. |
| `mlipflow advance` | Observe submitted work, fetch declared outputs, run checks, and reconcile state; it never launches a new node. |
| `mlipflow retry NODE`, `stop NODE` | Create a fresh attempt or explicitly stop supported running work. |
| `mlipflow route ...` | Rank compatible models from versioned benchmark evidence and project policy. |

Approval is operation-level. Replay never requires approval. Every SSH-Slurm execution
does, as do expensive local operations such as DFT labeling, training/fine-tuning,
MD/LASP execution, and fresh model inference. Ordinary local preparation, analysis,
selection, normalization, checking, transport/Arrhenius post-processing, ranking,
voltage analysis, and SQS generation run without `--approve` after their inputs pass
scientific validation.

A typical external or expensive node is operated as follows:

```bash
mlipflow --project . run train-model --dry-run
mlipflow --project . run train-model --approve
mlipflow --project . advance
mlipflow --project . status train-model
```

Scheduler completion is not scientific success. A node reaches `OK` only after its capability adapter verifies declared outputs and completion criteria.

One explicitly reviewed batch of expensive nodes may be approved once, after which an
agent may execute only those reviewed nodes with `--approve`. This is a supervision UX,
not persistent core state. Downstream local deterministic work continues without a new
approval; a newly introduced expensive or scheduled execution needs a new review.

## Validation and documentation

Software implementation, workflow completion, numerical agreement, and real-cluster validation are different claims. Keep site-specific execution evidence outside the public repository, and do not reinterpret an integration smoke as production or independent scientific validation.

Researchers remain responsible for validating DFT settings, models, datasets, simulation parameters, convergence, and uncertainty for their system.

The [`documentation index`](docs/README.md) organizes user guidance, HPC setup, and manuscript-specific reproduction material. Core references include [`ARCHITECTURE.md`](docs/ARCHITECTURE.md), the bundled [Agent Skills](.agents/skills/), and [`CLUSTER_ENVIRONMENTS.md`](docs/CLUSTER_ENVIRONMENTS.md).

## Contributing, security, and citation

Contributions are welcome across the core, built-in capabilities, schemas, tests, examples, Agent Skills, and site templates. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

Report security issues according to [`SECURITY.md`](SECURITY.md). Do not commit credentials, private cluster details, licensed pseudopotentials, model weights, large trajectories, or unpublished data.

MLIPFlow is licensed under the [Apache License 2.0](LICENSE). For published research, cite the software version or commit using [`CITATION.cff`](CITATION.cff), together with the scientific methods, datasets, models, and external codes used by the executed capabilities.
