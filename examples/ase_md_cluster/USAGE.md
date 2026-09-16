# Run one ASE MD task

Copy this directory to a work project. `project.yaml` wraps the existing NVT example
in a complete project: `md-900k-nvt` runs one million 1 fs steps at 900 K. These are
illustrative settings for review, not a recommendation for an unknown material.

Provide `inputs/start.extxyz` with the real periodic structure. Edit
`model-reference.json.example` to reference the matching model under your site's
model root. Set the calculator, ensemble, temperature, timestep, steps, output
intervals, seed and resources in `project.yaml` to your intended task. NVT friction
is in fs^-1. For NPT, use the existing NPT example's pressure in GPa and damping
times in fs. The concrete model must supply stress for NPT.

The site needs ASE and the selected MLIP framework in its calculator environment,
`ase-md-<calculator>/run.sh`, and the appropriate Slurm template. Existing sites and
models can be reused; the model is not downloaded by this example.

```bash
mlipipe --project /PATH/TO/md-project init
mlipipe --project /PATH/TO/md-project inspect md-900k-nvt
mlipipe --project /PATH/TO/md-project --format json run md-900k-nvt --dry-run
mlipipe --project /PATH/TO/md-project --format json run md-900k-nvt --approve
mlipipe --project /PATH/TO/md-project --format json advance
mlipipe --project /PATH/TO/md-project json md-900k-nvt
```

Submission returns `PENDING`; `advance` later fetches and checks the declared
outputs. Read `artifacts` for the trajectory, trajectory index, thermo table,
final structure and MD result paths. Times use the recorded global step and
timestep; temperature is K, energy eV, and volume Å³. Final `OK` confirms the
requested trajectory contract, not equilibration or diffusive sampling.

For transport, pass the collected trajectory artifact or use `from_node`/`role`
bindings in the same workflow. The plugin consumes compatible restart segments;
keep all attempts. The existing README's restart section covers interruptions.
