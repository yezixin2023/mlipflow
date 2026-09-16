# Prepare and execute LAMMPS MD

Use the existing `lammps-nvt.json` or `lammps-npt.json` and the matching model
reference in this directory. Replace the sample model ID/path with your exported
LAMMPS-ready model, and review the structure, temperature (K), timestep (fs in
the JSON config; ps in the generated LAMMPS metal deck), step count, seed,
thermostat/barostat and device target.
Raw training checkpoints are not interchangeable with exported LAMMPS models.

`project.yaml` is a complete two-node task using the existing NPT config and MACE
reference. Supply `inputs/start.extxyz` and edit the model reference for your model.
The execution node consumes the collected preparation manifest by role, so it does
not need a guessed attempt-directory path. Review the selected site and resources.

Preparation requires ASE locally. Execution needs a site template and LAMMPS built
with the declared MLIP interface; `backend_profile` selects that configured site.

```bash
mlipipe --project /PATH/TO/lammps-project init
mlipipe --project /PATH/TO/lammps-project --format json run prepare-lammps-mace --dry-run
mlipipe --project /PATH/TO/lammps-project --format json run prepare-lammps-mace
mlipipe --project /PATH/TO/lammps-project --format json advance
mlipipe --project /PATH/TO/lammps-project --format json run run-lammps-mace-gpu --dry-run
mlipipe --project /PATH/TO/lammps-project --format json run run-lammps-mace-gpu --approve
mlipipe --project /PATH/TO/lammps-project --format json advance
mlipipe --project /PATH/TO/lammps-project json run-lammps-mace-gpu
```

Preparation emits the
prepared manifest and input bundle; final MD `OK` exposes the dump, thermo, result
and restart artifacts through `artifacts[].role` and `artifacts[].path`.
Use the returned trajectory paths for transport. Periodic restart and interruption
rules remain in the existing example's restart section.
