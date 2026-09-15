# Analyze an existing MSD table

Copy this directory to a work project and install `mlipflow[transport]` in the
analysis environment. `project.yaml` is complete; provide the existing input at
`inputs/msd.csv`, with columns `time_ps,msd_A2`. The MSD must represent the intended
mobile-species displacement, with time in ps and squared displacement in Å².

Replace the example mobile species, 800 K temperature and 20–100 ps fit window with
the values established for your data. This example requests an unsmoothed fit and
a direct result at the input temperature. One temperature does not support an
Arrhenius extrapolation. With MSD alone it reports diffusivity; add `inputs.structure`
pointing to the real matching structure when requesting Nernst–Einstein conductivity.
Do not substitute a guessed cell volume or carrier count.

The configuration also supplies the adapter's required trajectory/Arrhenius controls
using the analysis program's documented defaults. The trajectory controls do not
smooth or change this MSD-table fit; `fit_smoothed_msd: false` governs that fit.

```bash
mlipflow --project /PATH/TO/transport-project init
mlipflow --project /PATH/TO/transport-project inspect transport
mlipflow --project /PATH/TO/transport-project --format json run transport --dry-run
mlipflow --project /PATH/TO/transport-project --format json run transport
mlipflow --project /PATH/TO/transport-project json transport
```

The run is synchronous. Check exit code, `state`, `check`, and `collection`.
`artifacts` gives the exact paths to `diffusion_results_by_temperature.csv`,
`arrhenius_summary.json`, `analysis_manifest.json`, and the MSD curves/fit plots
under `.mlipflow/runs/transport/attempt-1/ionic-transport-postprocess/`.
Diffusivity is cm²/s, conductivity mS/cm, activation energy eV. Unavailable
conductivity or extrapolation stays unavailable. On failure, inspect `reason` and
`mlipflow --project /PATH/TO/transport-project logs transport`.

For a collected ASE/LAMMPS/VASP trajectory, replace `source: msd` with
`source: trajectory`, set `input_paths` to the collected artifact, remove the
MSD-only units/temperature/fit-window fields, and set an explicit trajectory window
and `diffusion_analyzer_smoothed` policy. Native collected metadata supplies physical
time and temperature; standalone files need the matching documented timestep fields.
Keep the reviewed fitting and smoothing choices from the existing project.
