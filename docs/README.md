# MLIPFlow documentation

This index separates the main user path from implementation references and project-specific reproduction material. The CLI, built-in capability lookup, adapters, and deterministic tests are the source of truth for implemented operations.

## Start here

1. Read the top-level [`README`](../README.md) for installation, the command model, the capability/Agent Skill matrix, and a replay quick start.
2. Run the small [`high_entropy_sulfide`](../examples/high_entropy_sulfide/) replay example before launching external scientific software.
3. Use [`ARCHITECTURE.md`](ARCHITECTURE.md) together with `mlipflow --help` and `mlipflow inspect NODE` when building or debugging a project DAG.

## Choose a path

| Goal | Primary documentation | Related examples or references |
|---|---|---|
| Create and operate a workflow | [`README`](../README.md), [`ARCHITECTURE.md`](ARCHITECTURE.md) | [`examples/high_entropy_sulfide/`](../examples/high_entropy_sulfide/) |
| Configure a Slurm cluster | [`CLUSTER_ENVIRONMENTS.md`](CLUSTER_ENVIRONMENTS.md) | [`examples/site_templates/`](../examples/site_templates/) |
| Supervise work with an Agent Skill | [Capability and Skill guide](../README.md#scientific-capabilities-and-agent-skills) | Each capability's `skill/SKILL.md` and references |
| Prepare DFT labels and shared datasets | `dft-labeling` adapter | [`examples/training_all_models/`](../examples/training_all_models/) |
| Train or fine-tune MLIPs | `mlip-training` adapter | [`examples/training_all_models/`](../examples/training_all_models/), [`CLUSTER_ENVIRONMENTS.md`](CLUSTER_ENVIRONMENTS.md) |
| Run ASE or LAMMPS MD | `ase-md` and `lammps-md` adapters | [`examples/ase_md_cluster/`](../examples/ase_md_cluster/), [`examples/lammps_mlip_inputs/`](../examples/lammps_mlip_inputs/) |
| Analyze transport or compare AIMD and MLIP dynamics | `ionic-transport` and `mlip-benchmark` adapters | [`examples/aimd_reference_validation/`](../examples/aimd_reference_validation/) |
| Run an offline active-learning campaign | `active-learning`, `$mlip-active-learning` | [`examples/active_learning_validation/`](../examples/active_learning_validation/) |
| Change a built-in scientific capability | [`ARCHITECTURE.md`](ARCHITECTURE.md), adapter tests | [`mlipflow/plugins/`](../mlipflow/plugins/), [`CONTRIBUTING.md`](../CONTRIBUTING.md) |
| Contribute to the repository | [`CONTRIBUTING.md`](../CONTRIBUTING.md) | [`SECURITY.md`](../SECURITY.md), [`CHANGELOG.md`](../CHANGELOG.md) |

## Reference documentation

- [Source migration](MIGRATION.md) maps previous source and script locations.
- [Agent request examples](AGENT_USAGE.md) provides prompts for each scientific capability.
- [Example index](../examples/README.md) lists all bundled scenarios.


- [`ARCHITECTURE.md`](ARCHITECTURE.md) describes core services, execution boundaries, trust assumptions, and artifact flow.
- The installed `mlipflow --help` output is the authoritative command reference; [`ARCHITECTURE.md`](ARCHITECTURE.md) explains state transitions, approvals, retries, and reconciliation boundaries.
- Specialist supervision contracts and references live in each [`capability's`](../mlipflow/plugins/) `skill/` folder. [`.agents/skills/`](../.agents/skills/) provides discovery links and the standalone `mlip-workflow` Skill.
- [`CLUSTER_ENVIRONMENTS.md`](CLUSTER_ENVIRONMENTS.md) documents isolated scientific environments and site-template binding.

Historical migration, source-audit, and predecessor-project notes are intentionally omitted from this public navigation. They are not required to install, operate, extend, or validate MLIPFlow.

## Manuscript-specific reproduction

The [`high_entropy_sulfide_reproduction`](../examples/high_entropy_sulfide_reproduction/) example supports a particular evidence-reproduction workflow and is not required for ordinary MLIPFlow use.

Keep reproduction artifacts separate from general user guidance so that project-specific assumptions are not mistaken for universal defaults.

## Documentation rules

- Treat the current CLI help, built-in lookup, adapter implementation, and tests as authoritative when prose becomes stale.
- Distinguish scheduler completion from adapter-verified scientific `OK`.
- Record whether the original numerical program ran when that distinction affects scientific interpretation.
- Do not publish private paths, credentials, licensed pseudopotentials, model weights, raw large datasets, trajectories, or unpublished results.
- Link detailed contracts instead of repeating them across several top-level documents.
