---
name: ionic-transport
description: Supervise local MLIPFlow formal ionic-transport analysis through pymatgen-analysis-diffusion from existing ASE, LAMMPS, or VASP trajectories or explicit MSD tables, while keeping historical manuscript reproduction isolated. Use when planning, running, or verifying ionic-transport analyze-existing or the bounded md-smoke-and-analyze integration operation.
---

# Ionic transport

Use `plugins/ionic-transport` as the deterministic implementation. This Skill supervises scientific inputs and interpretation; it never estimates numerical transport results itself.

## Scope and operations

The plugin is local-only and has two formal operations:

- `analyze-existing` analyzes existing trajectory or MSD evidence. MLIPFlow packages `ionic_conductivity.py`; a project supplies only `inputs.input_paths`, never an external `analysis_script`.
- `md-smoke-and-analyze` is a tightly bounded local integration test. It invokes an explicitly supplied local ASE-MD source, then sends its tiny trajectories through the same packaged analysis runner and checker used by `analyze-existing`.

Do not add `ssh-slurm`, scheduler templates, cluster paths, modules, partitions, accounts, MPI launchers, or HPC lifecycle instructions. Use `$ase-md` or `$lammps-md` separately when a user needs production trajectory generation; then pass the completed local trajectory/MSD artifacts to `analyze-existing`.

Formal analysis uses `pymatgen-analysis-diffusion` directly:

```text
ASE/VASP/reliably described LAMMPS trajectory
-> ordered pymatgen Structure frames
-> DiffusionAnalyzer values, including analyzer.dt / 1000 as time_ps

MSD table -> get_diffusivity_from_msd
MSD + real Structure -> D * get_conversion_factor conductivity
multi-temperature D -> fit_arrhenius(..., mode="linear")
```

Do not ask MLIPFlow to refit `analyzer.msd`, reconstruct the analyzer time axis, clone pymatgen's weighted fit, recompute Nernst-Einstein conductivity, or silently fall back to NumPy/MLIPFlow formulas. Formal mode requires the `transport` extra in the same local Python interpreter that runs MLIPFlow. A missing dependency is a blocking error with `pip install 'mlipflow[transport]'`; it must not affect core import or the isolated historical wrapper.

Do not expand this Skill to Green-Kubo, custom anisotropic analysis, piecewise Arrhenius models, bootstrap uncertainty, or replica aggregation. `piecewise` must be `never`. A trajectory Haven ratio is an analyzer result, not an Agent-selected correction.

## Choose and declare the input contract

The packaged runner currently handles:

- ASE directories containing `production.traj`; frame spacing must come from a valid `metadata.json`, a valid `production_log.csv`, or explicit `ase_frame_step_fs`.
- LAMMPS directories containing an unwrapped `traj.lammpstrj`. Formal conversion requires periodic cells, stable frame/atom order, all species, uniform physical timestep, and frame spacing fine enough to preserve motion through ordered pymatgen Structures. Missing information is an error, never a NumPy fallback.
- VASP AIMD `vasprun.xml`; temperature must come from explicit override or VASP temperature metadata, and timestep must come from `POTIM` or explicit `vasp_step_fs`.
- Precomputed text/CSV/TSV/XVG MSD. Require explicit `msd_time_unit` and `msd_unit` whenever the selected column names do not encode them. For step-valued time require explicit `msd_step_ps` unless a companion LAMMPS input/log declares it.

ASE is mandatory when parsing `production.traj` or running `md-smoke-and-analyze`; a missing import must fail that selected path. Do not make pure MSD or non-ASE analysis import ASE when its actual source contract does not need it.

Always require an explicit mobile `specie`, positive temperature, reviewed analysis window, smoothing mode, `min_obs`, `avg_nsteps`, `step_skip`, and physical timestep source. Never infer scientific parameters from typical values.

For MSD-only inputs, require explicit time/MSD units. A real supplied Structure enables `get_conversion_factor`; without one, report D normally and conductivity as unavailable. Do not substitute literal carrier count/volume inputs or a private NE formula. Strict temperature directory names such as `T800`, `800K`, or `temp_800K` are declarations; mixed-token and bare-number directory guesses are not accepted.

Explain that the selected trajectory/MSD window, pymatgen smoothing settings, equilibration evidence, and trajectory length remain scientific judgments even when the checker returns `OK`.

## Carrier-count conventions

Normal formal `analyze-existing` never defaults to seven carriers and never consumes historical carrier-count constants. Trajectory conductivity comes directly from `DiffusionAnalyzer`; MSD-only conductivity requires a real Structure and `get_conversion_factor`.

The plugin-local historical manuscript wrapper has a separate `legacy_script` convention that deliberately preserves the historical Li10 script literal `N=7` for bug-compatible numerical parity. Its `composition_corrected` convention requires an explicit or POSCAR-derived carrier count and keeps diffusivity unchanged while conductivity changes with `N/V`. Historical parity also retains the historical rounded charge and Boltzmann constants so the N=7 versus corrected comparison isolates carrier count.

Never report `N=7` as the physical default, ground truth, or a requirement for other materials. Never mix the legacy and corrected conventions in one result. Historical agreement is parity evidence for the old script, not validation of the carrier choice.

## Local MLIPFlow lifecycle

Inspect the plugin and project first with read-only commands. For execution:

1. Inspect the dry-run's source type, important inputs, temperature/timestep source, carrier count, charge, volume, fit window, MSD method, expected outputs, and overwrite risk.
2. Let MLIPFlow invoke the pinned checker and collect phase. Do not bypass the state store by launching the packaged runner manually for a claimed workflow result.
3. Treat success only as final plugin `OK`, not merely subprocess return code zero.

Each attempt must use a fresh output directory. Do not delete, clear, or reuse an earlier attempt to simulate retry. Record the actual Python, MLIPFlow, pymatgen, pymatgen-analysis-diffusion, NumPy, and source-specific ASE versions.

## Checker and completion contract

The runner writes:

- `diffusion_results_by_temperature.csv`;
- `arrhenius_summary.json`;
- `postprocess_failures.json`;
- `analysis_manifest.json`;
- per-run MSD curves and fit diagnostics.

The analysis manifest binds the declared parameters, runtime versions, and scientific source/result artifacts. The checker rereads original sources and calls the same pymatgen public APIs again:

- trajectory: rebuild Structure frames and rerun `DiffusionAnalyzer`;
- MSD-only: rerun `get_diffusivity_from_msd`;
- MSD + Structure conductivity: rerun `get_conversion_factor`;
- Arrhenius: rerun `fit_arrhenius(..., mode="linear")`.

The checker compares saved values and time/MSD artifacts with those reruns; it must not independently reimplement pymatgen mathematics.

A self-reported JSON status is not completion evidence. Missing, changed, non-finite, parameter-mismatched, equation-inconsistent, or out-of-tree result artifacts must fail. Non-empty `postprocess_failures.json` fails unless partial results were explicitly requested; partial success is never silently promoted.

## Tiny MD smoke boundary

Use `md-smoke-and-analyze` only to test the local handoff from an explicit ASE-MD source to formal analysis. Keep its plugin-enforced temperature count, duration, frame, step, model, and fresh-output bounds. It is not production MD, convergence evidence, model validation, manuscript parity, or a recommended way to generate transport trajectories.

The smoke requires an explicit seed, but that seed covers only the documented NumPy velocity path. Do not claim framework or GPU bitwise determinism. Network-prone default model loading remains forbidden for MACE, CHGNet, M3GNet, and MatGL; use an explicit local model. EMT/LJ may use only the explicit `default` marker.

## Interpretation

`OK` means the declared files, parameters, pymatgen execution, provenance, and schemas are internally consistent. It does not establish equilibration, a diffusive regime, finite-size convergence, sufficient sampling, independent replicas, model accuracy, Nernst-Einstein validity in a correlated conductor, or safe long-range Arrhenius extrapolation. Historical numerical parity remains legacy reproduction only and is not evidence that a new formal pymatgen analysis must match old script conventions.
