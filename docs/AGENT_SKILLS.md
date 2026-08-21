# Agent Skills

MLIPFlow bundles Agent Skills under [`.agents/skills/`](../.agents/skills/) for compatible tool-using agents. A Skill explains how to supervise a research task: what evidence to inspect, which workflow boundary to use, when approval is required, and what completion state is acceptable.

Skills do **not** implement numerical methods. Deterministic execution, validation, collection, and replay remain in MLIPFlow core, plugin adapters, and the external scientific programs selected by the researcher.

## Capability map

| Research task | Agent Skill | Deterministic implementation | Use it for |
|---|---|---|---|
| End-to-end workflow design | `$mlip-workflow` | MLIPFlow core plus the selected plugins | Building a DAG, choosing the earliest missing stage, reviewing artifact handoffs, and preserving approval boundaries. |
| High-entropy and SQS structures | `$high-entropy-structure` | `high-entropy-structure` | Seeded `icet` SQS generation with explicit composition, sublattice, cutoff, supercell, and output contracts. |
| PES sampling | `$pes-sampling` | `pes-sampling` | DIRECT representative selection, LASP input preparation, scheduled LASP/SSW execution, archive replay, and checked structure-set merging. |
| DFT labeling and datasets | `$dft-labeling` | `dft-labeling` | VASP input preparation; bounded static, relax, or AIMD labeling; canonical labels; shared-split framework datasets; direct AIMD trajectory handoff. |
| MLIP training and fine-tuning | `$mlip-training` | `mlip-training` | DeepMD, M3GNet/MatGL, CHGNet, or MACE training/fine-tuning with explicit datasets, models, seeds, environments, and publication paths. |
| ASE molecular dynamics | `$ase-md` | `ase-md` | Scheduled single-temperature NVT Langevin or isotropic MTK NPT, including explicit models, supercells, checkpoints, and fresh-attempt restart. |
| LAMMPS molecular dynamics | `$lammps-md` | `lammps-md` | Deterministic deck preparation and scheduled DeepMD, MACE, or MatGL/M3GNet execution with bounded collection and binary-restart recovery. |
| MLIP benchmarking | `$mlip-benchmark` | `mlip-benchmark` | Fresh static-PES inference or normalization of explicit evidence, including task-separated structural-dynamics and ionic-transport metrics. |
| Offline active learning | `$mlip-active-learning` | `active-learning` plus existing training, MD, sampling, DFT, and benchmark plugins | Finite immutable rounds, calibrated one- or two-model committees, risk-union selection, DFT budgets, and round assessment. |
| Ionic transport and AIMD/MLIP dynamics | `$ionic-transport` | `ionic-transport` | Existing trajectory/MSD analysis, diffusion, conductivity, Haven ratio, Arrhenius fitting, and one bounded partial-RDF comparison at a common temperature. |
| Candidate ranking | `$candidate-ranking` | `candidate-ranking` | Deterministic top-k selection from an existing finite numeric metric with an explicit direction and missing-value policy. |
| Electrochemical voltage | `$mlip-workflow` or direct plugin orchestration | `electrochemical-voltage` | Adjacent average Li intercalation voltages from explicit total energies or deterministic evidence replay. There is no dedicated voltage Skill. |

Exact operations, parameters, backends, outputs, limitations, and checks are defined by the corresponding manifests under [`plugins/`](../plugins/). The Skill files define supervision behavior and should not duplicate the full plugin contract.

## Standard supervision sequence

For a workflow node that may execute external or expensive work, an agent should follow the same reviewable lifecycle as a researcher:

```bash
mlipflow --project . doctor
mlipflow --project . inspect NODE
mlipflow --project . run NODE --dry-run --audit
mlipflow --project . run NODE --approve
mlipflow --project . advance
mlipflow --project . status NODE
```

The agent must inspect the dry-run command, backend, resources, staged inputs, site-template family, output allowlist, and scientific completion policy before approval. Scheduler `COMPLETED` or process exit code zero is not sufficient: downstream work may proceed only from a plugin-verified final `OK` attempt.

Read-only commands such as `list`, `status`, `json`, `inspect`, `logs`, `route`, and `doctor` should remain read-only. `advance` observes and reconciles already submitted work; it never launches the next node automatically.

## Artifact-first routing

Use existing verified artifacts rather than repeating expensive computation:

- A final `OK` DFT-labeling AIMD attempt already exposes its collected `vasprun.xml` trajectories to `$ionic-transport`; do not ask the researcher to copy or rename them.
- Completed ASE- or LAMMPS-MD attempts already supply timing, temperature, species/type mapping, model identity, and restart-aware trajectory segments to transport analysis.
- One AIMD reference and one MLIP trajectory set can produce an explicit-pair partial RDF comparison and metric-only evidence for `$mlip-benchmark`.
- A canonical held-out dataset and published model references should feed fresh benchmark inference directly; do not construct a second split for each framework.
- Existing benchmark evidence should drive model routing. A model that is strongest for static PES is not automatically the correct choice for structural dynamics, ionic transport, or voltage work.
- An existing candidate manifest and finite metric should go directly to `$candidate-ranking`; the ranking Skill does not generate candidates or compute properties.

When the required artifact does not exist, `$mlip-workflow` should enter at the earliest missing producer rather than rebuilding the entire pipeline.

## Scientific and infrastructure boundaries

Agent Skills must not:

- embed SSH hosts, accounts, partitions, modules, executable paths, environment activation, model roots, dataset roots, or work roots in a portable project;
- guess a foundation model, hyperparameter set, DFT setting, MD ensemble, convergence threshold, atom selection, or ranking objective that the researcher has not supplied or approved;
- bypass dry-run review or approval for expensive/external work;
- treat a scheduler state, file presence, or process return code as scientific success without the plugin checker;
- copy model weights, licensed pseudopotentials, large datasets, or trajectories through the control plane when a checked reference or direct artifact handoff exists;
- collapse static PES, structural dynamics, ionic transport, and electrochemical tasks into one universal model score unless the project explicitly defines such a policy;
- overwrite a prior attempt or represent a new active-learning round as a retry.

Cluster-specific information belongs in the user-local site profile and site-owned templates. Scientific software and framework environments remain user- or site-supplied unless a plugin explicitly states otherwise.

## Maintaining a Skill

When a plugin or CLI contract changes:

1. update the plugin manifest, schema, adapter, and tests first;
2. update the corresponding `SKILL.md` to reflect the verified interface and evidence boundaries;
3. keep detailed parameters in the manifest rather than copying them into several prose documents;
4. update the combined capability table in the top-level [`README`](../README.md) only when user-visible scope or execution support changes;
5. record validation changes in [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md), [`SCIENTIFIC_VALIDATION.md`](SCIENTIFIC_VALIDATION.md), or [`HPC_VALIDATION.md`](HPC_VALIDATION.md), as appropriate.

The current CLI help, schemas, plugin manifests, and checked adapter behavior take precedence over older prose or historical research scripts.
