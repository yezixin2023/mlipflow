# Examples

Each scenario is self-contained. Begin with the replay example; scheduled examples
need a user-owned site profile and the scientific inputs described in their README.

| Example | Purpose |
|---|---|
| [high_entropy_sulfide](high_entropy_sulfide/) | Small workflow using existing results; the quick-start replay |
| [training_all_models](training_all_models/) | Shared DFT dataset views and multi-framework training |
| [ase_md_cluster](ase_md_cluster/) | Explicit-model ASE molecular dynamics on a configured cluster |
| [lammps_mlip_inputs](lammps_mlip_inputs/) | LAMMPS MLIP input preparation and execution |
| [lasp_random_walk](lasp_random_walk/) | LASP/SSW sampling inputs and cluster guidance |
| [active_learning_validation](active_learning_validation/) | Finite offline active-learning integration |
| [aimd_reference_validation](aimd_reference_validation/) | AIMD reference and MLIP validation workflow |
| [site_templates](site_templates/) | Portable examples of site-owned scheduler templates |

Replay is a **structured collection of existing results**.
Examples contain small public inputs and references; supply private datasets,
models, licensed files and infrastructure settings outside the source repository.
