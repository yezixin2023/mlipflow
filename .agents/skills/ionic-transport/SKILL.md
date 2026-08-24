---
name: ionic-transport
description: Supervise local MLIPFlow formal ionic-transport analysis through pymatgen-analysis-diffusion from existing ASE, LAMMPS, or VASP trajectories or explicit MSD tables, while keeping historical manuscript reproduction isolated. Use when planning, running, or verifying ionic-transport analyze-existing or the bounded md-smoke-and-analyze integration operation.
---

# Ionic transport

Use `plugins/ionic-transport` as the deterministic implementation. This Skill supervises scientific inputs and interpretation; it never estimates numerical transport results itself.

## Scope and operations

The plugin is local-only and has two formal operations:

- `analyze-existing` analyzes existing trajectory or MSD evidence. When the node depends on completed AIMD `dft-labeling`, `ase-md`, or `lammps-md` nodes, core supplies their collected artifacts automatically; `inputs.input_paths` remains available for historical standalone files and explicit mixed-source selection. Never request an external `analysis_script`.
- `md-smoke-and-analyze` is a tightly bounded local integration test. It invokes an explicitly supplied local ASE-MD source, then sends its tiny trajectories through the same packaged analysis runner and checker used by `analyze-existing`.

Both local operations have `approval_required: false`. Inspect the dry-run and continue
without asking for `--approve`, including multi-temperature Arrhenius fitting and the
bounded smoke handoff. Approval removal does not relax the scientific contract: missing
species, temperatures, timestep sources, units, fitting windows, or required dependencies
still block or fail through Adapter validation.

`analyze-existing` can also request one bounded AIMD-versus-MLIP partial RDF
comparison while it analyzes the same trajectories. Require an explicit atom pair,
radial range, bin count, common temperature when more than one is available, trajectory
window, model family, scenario, and split. The result contains both curves, curve
MAE/RMSE, common-temperature transport errors, and metric-only records consumable by
`$mlip-benchmark`. This is not a general trajectory-analysis framework.

Do not add `ssh-slurm`, scheduler templates, cluster paths, modules, partitions, accounts, MPI launchers, or HPC lifecycle instructions. Use `$ase-md` or `$lammps-md` separately when a user needs production trajectory generation; then pass the completed local trajectory/MSD artifacts to `analyze-existing`.

Formal analysis uses `pymatgen-analysis-diffusion` directly:

```text
ASE/VASP/reliably described LAMMPS trajectory
-> ordered pymatgen Structure frames
-> DiffusionAnalyzer values, including analyzer.dt / 1000 as time_ps

MSD table with physical lag/elapsed time -> get_diffusivity_from_msd
MSD + real Structure -> D * get_conversion_factor conductivity
multi-temperature D -> fit_arrhenius(..., mode="linear")
```

Do not ask MLIPFlow to refit `analyzer.msd`, reconstruct the analyzer time axis, clone pymatgen's weighted fit, recompute Nernst-Einstein conductivity, or silently fall back to NumPy/MLIPFlow formulas. Formal mode requires the `transport` extra in the same local Python interpreter that runs MLIPFlow. A missing dependency is a blocking error with `pip install 'mlipflow[transport]'`; it must not affect core import or the isolated historical wrapper.

Do not expand this Skill to Green-Kubo, custom anisotropic analysis, multiple-breakpoint
models, bootstrap uncertainty, or replica aggregation. The bounded Arrhenius contract
always retains the global pymatgen linear fit and may evaluate at most one breakpoint
between adjacent sampled temperatures. Each branch must contain at least three points
and is fitted with the same `fit_arrhenius(..., mode="linear")` API. `piecewise=auto`
selects the two-branch result only when both the declared BIC improvement and relative
activation-energy-change thresholds pass; `piecewise=always` exposes and selects the
best valid two-branch fit even when that automatic evidence is weak, and must say that
the selection was forced. A trajectory Haven ratio is an analyzer result, not an
Agent-selected correction.

Do not request formal `charge`, `dimensions`, `haven_ratio`, `drift_correction`, or `msd_mode` inputs. They do not control the pymatgen formal calculation and are not part of the formal contract. Optional `fit_start_ps`/`fit_end_ps` bounds apply only to an MSD table; trajectory analysis uses `DiffusionAnalyzer` over the selected trajectory frames.

## Choose and declare the input contract

The packaged runner handles both MLIPFlow-native and historical inputs:

- Collected DFT-labeling AIMD `vasprun.xml` artifacts. Read temperature and timestep from VASP metadata under the same VASP trajectory contract; do not ask the user to copy or locate the XML after a final `OK` AIMD node.
- Native ASE-MD `trajectory.traj` plus `trajectory-index.json`/`md-result.json`. Read temperature, integration timestep, trajectory interval, global steps, physical time, ensemble, model identity, and structure identity from the collected artifacts. Never ask the user for `ase_frame_step_fs`, temperature metadata, a copied file, or a handwritten `metadata.json`.
- Historical ASE `production.traj`. Preserve the existing metadata/log/explicit compatibility path.
- Native LAMMPS-MD `trajectory.lammpstrj` plus `lammps-execution-result.json`, the prepared input manifest when present, or the approved execution identity. Read temperature, timestep, dump interval, type map, ensemble, model identity, and structure identity automatically. Never ask the user for `lammps_timestep_ps`, `temperature`, `mobile_type`, or `lammps_data_name` when native artifacts supply them.
- Historical LAMMPS `traj.lammpstrj`. Preserve the existing standalone-file compatibility path.
- LAMMPS coordinates may be `xu/yu/zu`, `xsu/ysu/zsu`, `x/y/z`, or `xs/ys/zs`. Prefer `ix/iy/iz` image flags when present; otherwise unwrap wrapped coordinates across ordered frames with fractional minimum-image continuity before constructing the common Structure sequence.
- VASP AIMD `vasprun.xml`; temperature must come from explicit override or VASP temperature metadata, and timestep must come from `POTIM` or explicit `vasp_step_fs`.
- Precomputed text/CSV/TSV/XVG MSD. Time values are physical lag/elapsed time and are never rebased to the first row. Require explicit `msd_time_unit` and `msd_unit` whenever the selected column names do not encode them. For step-valued time require explicit `msd_step_ps` unless a companion LAMMPS input/log declares it. With `smoothed=max`, zero lag is excluded from the pymatgen call and marked false in `used_for_analysis`.

ASE is mandatory when parsing `trajectory.traj`, `production.traj`, or running `md-smoke-and-analyze`; a missing import must fail that selected path. Do not make pure MSD or non-ASE analysis import ASE when its actual source contract does not need it.

Collected restart attempts from one MD node form one logical trajectory. Order segments by their global MD step, verify compatible temperature/timestep/species/model/structure identities, and remove a repeated restart-boundary frame. Do not ask the user to concatenate attempts. Multiple completed ASE and LAMMPS nodes, including an explicit mixture, may feed one multi-temperature Arrhenius analysis.

Always require an explicit mobile `specie`, at least one non-mobile framework atom for trajectory-based `DiffusionAnalyzer`, positive temperature, reviewed analysis window, smoothing mode, `min_obs`, `avg_nsteps`, `step_skip`, and physical timestep source. All-mobile elemental trajectories must fail explicitly; do not invent a self-diffusion or drift-correction fallback. Never infer scientific parameters from typical values.

For MSD-only inputs, require explicit time/MSD units. A real supplied Structure enables `get_conversion_factor`; without one, report D normally and conductivity as unavailable. Do not substitute literal carrier count/volume inputs or a private NE formula. Strict temperature directory names such as `T800`, `800K`, or `temp_800K` are declarations; mixed-token and bare-number directory guesses are not accepted.

Explain that the selected trajectory/MSD window, pymatgen smoothing settings, equilibration evidence, and trajectory length remain scientific judgments even when the checker returns `OK`.

## Carrier-count conventions

Normal formal `analyze-existing` never defaults to seven carriers and never consumes historical carrier-count constants. Trajectory conductivity comes directly from `DiffusionAnalyzer`; MSD-only conductivity requires a real Structure and `get_conversion_factor`.

The plugin-local historical manuscript wrapper has a separate `legacy_script` convention that deliberately preserves the historical Li10 script literal `N=7` for bug-compatible numerical parity. Its `composition_corrected` convention requires an explicit or POSCAR-derived carrier count and keeps diffusivity unchanged while conductivity changes with `N/V`. Historical parity also retains the historical rounded charge and Boltzmann constants so the N=7 versus corrected comparison isolates carrier count.

Never report `N=7` as the physical default, ground truth, or a requirement for other materials. Never mix the legacy and corrected conventions in one result. Historical agreement is parity evidence for the old script, not validation of the carrier choice.

## Local MLIPFlow lifecycle

Inspect the plugin and project first with read-only commands. For execution:

1. Inspect the dry-run's source type, important inputs, temperature/timestep source, optional MSD-only subset, pymatgen smoothing settings, expected outputs, and overwrite risk.
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

When RDF comparison is requested it additionally writes `rdf_curves.csv` and
`aimd_mlip_comparison.json`. The checker rebuilds both curves from the original
trajectories. The comparison JSON is metric-only evidence: it does not claim that the
local analysis node executed an MLIP model.

The analysis manifest binds the declared parameters, runtime versions, and scientific source/result artifacts. The checker rereads original sources and calls the same pymatgen public APIs again:

- trajectory: rebuild Structure frames and rerun `DiffusionAnalyzer`;
- MSD-only: rerun `get_diffusivity_from_msd`;
- MSD + Structure conductivity: rerun `get_conversion_factor`;
- Arrhenius: rerun the global `fit_arrhenius(..., mode="linear")` baseline and the same
  declared zero-or-one-breakpoint enumeration, branch fits, BIC comparison, selection,
  and target-regime prediction.

The checker compares saved values and time/MSD artifacts with those reruns; it must not independently reimplement pymatgen mathematics.

A self-reported JSON status is not completion evidence. Missing, changed, non-finite, parameter-mismatched, equation-inconsistent, or out-of-tree result artifacts must fail. Non-empty `postprocess_failures.json` fails unless partial results were explicitly requested; partial success is never silently promoted.

## Tiny MD smoke boundary

Use `md-smoke-and-analyze` only to test the local handoff from an explicit ASE-MD source to formal analysis. Keep its plugin-enforced temperature count, duration, frame, step, model, and fresh-output bounds. It is not production MD, convergence evidence, model validation, manuscript parity, or a recommended way to generate transport trajectories.

The smoke requires an explicit seed, but that seed covers only the documented NumPy velocity path. Do not claim framework or GPU bitwise determinism. Network-prone default model loading remains forbidden for MACE, CHGNet, M3GNet, and MatGL; use an explicit local model. EMT/LJ may use only the explicit `default` marker.

## Interpretation

`ARRHENIUS_REGIME_CHANGE_DETECTED` means only that the bounded two-branch transport
model passed the declared numerical evidence thresholds. Report the breakpoint as the
interval between adjacent sampled temperatures, never as an exact transition
temperature. It does not establish a physical phase transition; any phase-transition
claim requires independent structural evidence. When a piecewise model is selected,
use only the low-temperature branch below the breakpoint interval and only the
high-temperature branch above it. A target inside the interval is ambiguous and must
not receive a silently chosen Arrhenius branch. Keep directly simulated target
conductivity separate from an Arrhenius prediction.

`OK` means the declared files, parameters, pymatgen execution, provenance, and schemas are internally consistent. It does not establish equilibration, a diffusive regime, finite-size convergence, sufficient sampling, independent replicas, model accuracy, Nernst-Einstein validity in a correlated conductor, or safe long-range Arrhenius extrapolation. Historical numerical parity remains legacy reproduction only and is not evidence that a new formal pymatgen analysis must match old script conventions.
