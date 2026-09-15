# Agent use

For installation, Skill discovery, CLI output and executable task recipes, see
[USAGE.md](USAGE.md). Installing the Python package and enabling a Skill are separate
steps. Routine tasks use the selected Skill, existing configuration and CLI results.
Source inspection is appropriate for development or a specific error.

Reuse parameters and authorization already supplied for the task. Gather consequential
missing decisions in one question; a new costly scope needs new intent. Status checks,
collection and deterministic postprocessing continue within the existing authorization.

## Request examples

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


See the [quick start](../README.md#quick-start) and [capability guide](../README.md#scientific-capabilities-and-agent-skills).
