---
name: ionic-transport
description: Supervise local MLIPFlow ionic-transport analysis from existing ASE, LAMMPS, or VASP trajectories or explicit MSD tables through MSD, diffusion, uncorrected Nernst-Einstein conductivity, and single-line Arrhenius fitting. Use when planning, running, or verifying ionic-transport analyze-existing or the bounded md-smoke-and-analyze integration operation, or when distinguishing historical N=7 parity from composition-corrected carrier counts.
---

# Ionic transport

Use `plugins/ionic-transport` as the deterministic implementation. This Skill supervises inputs, approval, and interpretation; it never estimates numerical transport results itself.

## Scope and operations

The plugin is local-only and has two formal operations:

- `analyze-existing` analyzes existing trajectory or MSD evidence. MLIPFlow packages `ionic_conductivity.py`; a project supplies only `inputs.input_paths`, never an external `analysis_script`.
- `md-smoke-and-analyze` is a tightly bounded local integration test. It invokes an explicitly supplied local ASE-MD source, then sends its tiny trajectories through the same packaged analysis runner and checker used by `analyze-existing`.

Do not add `ssh-slurm`, scheduler templates, cluster paths, modules, partitions, accounts, MPI launchers, or HPC lifecycle instructions. Use `$ase-md` or `$lammps-md` separately when a user needs production trajectory generation; then pass the completed local trajectory/MSD artifacts to `analyze-existing`.

The formal scientific chain is:

```text
existing MD/AIMD trajectory or MSD
-> MSD in A^2 versus time in ps
-> D = slope / (2 d), with d=3
-> uncorrected Nernst-Einstein conductivity
-> single-line ln(D_cm2/s) versus 1/T Arrhenius fit
```

Do not expand this Skill to Green-Kubo, Haven-ratio estimation, anisotropic transport, piecewise Arrhenius models, bootstrap uncertainty, or replica aggregation. `haven_ratio` must be `1`; `dimensions` must be `3`; `piecewise` must be `never`.

## Choose and declare the input contract

The packaged runner currently handles:

- ASE directories containing `production.traj`; frame spacing must come from a valid `metadata.json`, a valid `production_log.csv`, or explicit `ase_frame_step_fs`.
- LAMMPS directories containing an unwrapped `traj.lammpstrj`. Accept only `xu/yu/zu` or `xsu/ysu/zsu`, not wrapped coordinates. Timestep must come from a parsed LAMMPS input/log or explicit `lammps_timestep_ps`. Mobile identity must come from an `element` column, explicit `mobile_type`, or reviewed data-file metadata.
- VASP AIMD `vasprun.xml`; temperature must come from explicit override or VASP temperature metadata, and timestep must come from `POTIM` or explicit `vasp_step_fs`.
- Precomputed text/CSV/TSV/XVG MSD. Require explicit `msd_time_unit` and `msd_unit` whenever the selected column names do not encode them. For step-valued time require explicit `msd_step_ps` unless a companion LAMMPS input/log declares it.

Always require an explicit mobile `specie`, positive charge number, non-negative fit start, later fit end, and at least three fit points. Temperature, carrier count, and volume may be derived only from actual source metadata/structure. If that evidence is absent or ambiguous, require explicit `temperature_k`/source-specific temperature, `n_mobile_ions`, and `volume_a3`; never infer them from chemistry or typical values.

For MSD-only inputs, require explicit positive `n_mobile_ions` and `volume_a3`. Strict temperature directory names such as `T800`, `800K`, or `temp_800K` are declarations; mixed-token and bare-number directory guesses are not accepted.

Require an explicit drift correction and MSD mode for trajectory inputs. Explain that the chosen fit window, equilibration evidence, drift policy, and trajectory length remain scientific judgments even when the checker returns `OK`.

## Carrier-count conventions

Normal `analyze-existing` never defaults to seven carriers. It uses a structure-derived or explicitly approved composition-corrected count.

The plugin-local historical manuscript wrapper has a separate `legacy_script` convention that deliberately preserves the historical Li10 script literal `N=7` for bug-compatible numerical parity. Its `composition_corrected` convention requires an explicit or POSCAR-derived carrier count and keeps diffusivity unchanged while conductivity changes with `N/V`. Historical parity also retains the historical rounded charge and Boltzmann constants so the N=7 versus corrected comparison isolates carrier count.

Never report `N=7` as the physical default, ground truth, or a requirement for other materials. Never mix the legacy and corrected conventions in one result. Historical agreement is parity evidence for the old script, not validation of the carrier choice.

## Local MLIPFlow lifecycle

Inspect the plugin and project first with read-only commands. For execution:

1. Run the exact `mlipflow run ... --dry-run` plan.
2. Review inputs, hashes, source type, temperature/timestep provenance, carrier count, charge, volume, fit window, MSD method, expected rows/files, local Python executable, and overwrite risk.
3. Execute only with the exact returned `sha256:...` approval digest.
4. Let the local MLIPFlow run invoke the pinned checker and collect phase. Do not bypass the state store by launching the packaged runner manually for a claimed workflow result.
5. Treat success only as final plugin `OK`, not merely subprocess return code zero.

Each attempt must use a fresh output directory. Do not delete, clear, or reuse an earlier attempt to simulate retry.

## Checker and provenance contract

The runner writes:

- `diffusion_results_by_temperature.csv`;
- `arrhenius_summary.json`;
- `postprocess_failures.json`;
- `analysis_manifest.json`;
- per-run MSD curves and fit diagnostics.

The analysis manifest pins approved parameters and SHA-256/size provenance for scientific source and result artifacts. The checker must rehash those files, reparse every MSD curve, select the recorded fit points, and independently verify:

- reported slope, fit endpoints, and R2 against the curve;
- `D = slope/(2d)` and the A²/ps to cm²/s conversion;
- conductivity from reported D, T, carrier count, charge, and volume;
- single-line Arrhenius slope, intercept, activation energy, prefactor, R2, and target-temperature diffusivity.

A self-reported JSON status is not completion evidence. Missing, changed, non-finite, parameter-mismatched, equation-inconsistent, or out-of-tree result artifacts must fail. Non-empty `postprocess_failures.json` fails unless partial results were explicitly approved; partial success is never silently promoted.

## Tiny MD smoke boundary

Use `md-smoke-and-analyze` only to test the local handoff from an explicit ASE-MD source to formal analysis. Keep its plugin-enforced temperature count, duration, frame, step, model, and fresh-output bounds. It is not production MD, convergence evidence, model validation, manuscript parity, or a recommended way to generate transport trajectories.

The smoke requires an explicit seed, but that seed covers only the documented NumPy velocity path. Do not claim framework or GPU bitwise determinism. Network-prone default model loading remains forbidden for MACE, CHGNet, M3GNet, and MatGL; use an explicit local model. EMT/LJ may use only the explicit `default` marker.

## Interpretation

`OK` means the declared files, parameters, formulas, and schemas are internally consistent. It does not establish equilibration, a diffusive regime, finite-size convergence, sufficient sampling, independent replicas, model accuracy, Nernst-Einstein validity in a correlated conductor, or safe long-range Arrhenius extrapolation. Report those limitations with every result.
