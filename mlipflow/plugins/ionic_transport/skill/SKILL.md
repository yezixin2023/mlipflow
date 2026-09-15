---
name: ionic-transport
description: Supervise local MLIPFlow formal ionic-transport analysis through pymatgen-analysis-diffusion from existing ASE, LAMMPS, or VASP trajectories or explicit MSD tables, while keeping historical manuscript reproduction isolated. Use when planning, running, or verifying ionic-transport analyze-existing or the bounded md-smoke-and-analyze integration operation.
---

# Ionic transport

Use `mlipflow/plugins/ionic_transport` through MLIPFlow. The Skill selects the analysis and
guards scientific interpretation; it does not calculate diffusion or conductivity.

## Route the request

- Existing ASE, LAMMPS, or VASP trajectories, explicit MSD tables, multi-temperature
  Arrhenius analysis, or the supported AIMD/MLIP RDF comparison:
  `analyze-existing`.
- A tiny local integration test from an explicit ASE-MD source:
  `md-smoke-and-analyze`.
- Production trajectory generation: use `$ase-md`, `$lammps-md`, or reviewed AIMD
  first; do not expand the smoke operation into production MD.

## Inputs and example

Reuse collected trajectories, including restart segments, through
`inputs.input_paths`; preserve their physical time and temperature metadata.
For MSD input, supply the column meaning, explicit time/MSD units and temperature;
add the real `inputs.structure` when conductivity is required. Set the scientific
window, species and smoothing policy from the project. A complete MSD task and
commands are in `examples/ionic_transport/project.yaml` and `USAGE.md`.
The bundled runner uses the active environment's `transport` dependencies.
Keep historical manuscript carrier-count conventions isolated; the old `N=7`
value is parity evidence, not a default for formal analysis.

## Scientific judgment

Require the actual mobile species, trajectory or MSD meaning, temperature and physical
time source, units, analysis window, and requested extrapolation. Do not infer missing
scientific inputs from filenames or typical values. A real structure is needed before
MSD-only evidence can support conductivity; otherwise report diffusivity without
inventing a conversion.

Review whether sampling is equilibrated and diffusive, whether the selected window and
smoothing are defensible, and whether extrapolation is scientifically reasonable.
These judgments are not supplied by Adapter `OK`.

An Arrhenius breakpoint is a bounded two-branch transport-model result, not an exact
transition temperature or proof of a physical phase transition. Any phase-transition
claim needs independent structural evidence. Keep a directly simulated target value
separate from an extrapolated prediction, and do not silently select a branch for a
target inside the breakpoint interval.

## Run and read the result

For the existing local inputs, the shortest execution is:

```bash
mlipflow --project PROJECT init
mlipflow --project PROJECT --format json run NODE
```

Use `run NODE --dry-run` when reviewing changed inputs or execution scope; it reports
the effective `approval_required` value. `mlipflow --project PROJECT json NODE`
reads the saved result later.

Read `state`, `metrics`, `artifacts[].role` and the full `artifacts[].path` from JSON.
Require final plugin `OK` before using outputs. On failure start with `reason`,
`check.diagnostics`, `manifest_path` and `logs`; `mlipflow --project PROJECT logs NODE`
shows saved stdout/stderr. Examples are in the checkout's `examples/` or the
installed environment's `share/mlipflow/examples/`.

Report source type, temperatures, mobile species, window, formal analysis mode, direct
versus extrapolated results, and unavailable quantities. `OK` establishes internal
contract consistency, not equilibration, a diffusive regime, finite-size convergence,
replica convergence, model accuracy, Nernst-Einstein validity, or safe long-range
extrapolation. Replay remains a structured collection of existing results, not fresh
analysis.
