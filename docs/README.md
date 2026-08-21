# MLIPFlow documentation

This index separates the main user path from implementation references, validation evidence, and project-specific reproduction material. Plugin manifests and schemas remain the source of truth for exact operations and data contracts.

## Start here

1. Read the top-level [`README`](../README.md) for installation, the command model, the combined plugin/Agent Skill capability matrix, and a replay quick start.
2. Run the small [`high_entropy_sulfide`](../examples/high_entropy_sulfide/) replay example before launching external scientific software.
3. Use [`ARCHITECTURE.md`](ARCHITECTURE.md) for execution boundaries, state, plugins, and artifact flow.
4. Check [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md) before depending on a scientific capability.

## Choose a path

| Goal | Primary documentation | Related examples or references |
|---|---|---|
| Create and operate a workflow | [`README`](../README.md), [`ARCHITECTURE.md`](ARCHITECTURE.md) | [`examples/high_entropy_sulfide/`](../examples/high_entropy_sulfide/) |
| Configure a Slurm cluster | [`CLUSTER_ENVIRONMENTS.md`](CLUSTER_ENVIRONMENTS.md) | [`examples/site_templates/`](../examples/site_templates/), [`HPC_VALIDATION.md`](HPC_VALIDATION.md) |
| Supervise work with an Agent Skill | [`AGENT_SKILLS.md`](AGENT_SKILLS.md) | [`.agents/skills/`](../.agents/skills/) |
| Prepare DFT labels and shared datasets | `dft-labeling` [`plugin.yaml`](../plugins/dft-labeling/plugin.yaml) | [`examples/training_all_models/`](../examples/training_all_models/) |
| Train or fine-tune MLIPs | `mlip-training` [`plugin.yaml`](../plugins/mlip-training/plugin.yaml) | [`examples/training_all_models/`](../examples/training_all_models/), [`CLUSTER_ENVIRONMENTS.md`](CLUSTER_ENVIRONMENTS.md) |
| Run ASE or LAMMPS MD | [`ase-md`](../plugins/ase-md/plugin.yaml) and [`lammps-md`](../plugins/lammps-md/plugin.yaml) | [`examples/ase_md_cluster/`](../examples/ase_md_cluster/), [`examples/lammps_mlip_inputs/`](../examples/lammps_mlip_inputs/) |
| Analyze transport or compare AIMD and MLIP dynamics | [`ionic-transport`](../plugins/ionic-transport/plugin.yaml) and [`mlip-benchmark`](../plugins/mlip-benchmark/plugin.yaml) | [`examples/aimd_reference_validation/`](../examples/aimd_reference_validation/) |
| Run an offline active-learning campaign | `active-learning` [`plugin.yaml`](../plugins/active-learning/plugin.yaml), `$mlip-active-learning` | [`examples/active_learning_validation/`](../examples/active_learning_validation/) |
| Add a scientific plugin | [`PLUGIN_DEVELOPMENT.md`](PLUGIN_DEVELOPMENT.md), [`ARCHITECTURE.md`](ARCHITECTURE.md) | [`schemas/plugin.schema.json`](../schemas/plugin.schema.json), existing plugin tests |
| Contribute to the repository | [`CONTRIBUTING.md`](../CONTRIBUTING.md) | [`SECURITY.md`](../SECURITY.md), [`CHANGELOG.md`](../CHANGELOG.md) |

## Reference documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) describes core services, execution boundaries, trust assumptions, and artifact flow.
- [`PLUGIN_DEVELOPMENT.md`](PLUGIN_DEVELOPMENT.md) defines the plugin lifecycle and review expectations.
- [`AGENT_SKILLS.md`](AGENT_SKILLS.md) explains when each bundled Skill should supervise a capability and what must remain delegated to plugins.
- [`CLUSTER_ENVIRONMENTS.md`](CLUSTER_ENVIRONMENTS.md) documents isolated scientific environments and site-template binding.

## Validation evidence

Validation documents make separate claims about software behavior, scientific calculation, numerical agreement, and real-cluster execution:

- [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md) — operation-by-operation status and limitations;
- [`SCIENTIFIC_VALIDATION.md`](SCIENTIFIC_VALIDATION.md) — scientific and numerical validation evidence;
- [`HPC_VALIDATION.md`](HPC_VALIDATION.md) — scheduler lifecycle and real-HPC evidence.

Machine-readable validation records are stored under [`reports/`](../reports/). These records should be retained with the documents that cite them; they are evidence, not general tutorials.

## Manuscript-specific reproduction

The following material supports a particular evidence-reproduction workflow and is not required for ordinary MLIPFlow use:

- [`MANUSCRIPT_REPRODUCTION.md`](MANUSCRIPT_REPRODUCTION.md);
- [`MANUSCRIPT_BENCHMARK_AUDIT.md`](MANUSCRIPT_BENCHMARK_AUDIT.md);
- [`examples/high_entropy_sulfide_reproduction/`](../examples/high_entropy_sulfide_reproduction/).

Keep reproduction artifacts separate from general user guidance so that project-specific assumptions are not mistaken for universal defaults.

## Documentation rules

- Treat the current plugin manifest, schema, and CLI help as authoritative when prose becomes stale.
- Distinguish scheduler completion from plugin-verified scientific `OK`.
- State whether evidence was freshly computed, normalized, replayed, or imported.
- Do not publish private paths, credentials, licensed pseudopotentials, model weights, raw large datasets, trajectories, or unpublished results.
- Link detailed contracts instead of repeating them across several top-level documents.
