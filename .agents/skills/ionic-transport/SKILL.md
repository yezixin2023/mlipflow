---
name: ionic-transport
description: Supervise local MLIPFlow formal ionic-transport analysis through pymatgen-analysis-diffusion from existing ASE, LAMMPS, or VASP trajectories or explicit MSD tables, while keeping historical manuscript reproduction isolated. Use when planning, running, or verifying ionic-transport analyze-existing or the bounded md-smoke-and-analyze integration operation.
---

# Ionic transport

Use `plugins/ionic-transport` through MLIPFlow. The Skill selects the analysis and
guards scientific interpretation; it does not calculate diffusion or conductivity.

## Route the request

- Existing ASE, LAMMPS, or VASP trajectories, explicit MSD tables, multi-temperature
  Arrhenius analysis, or the supported AIMD/MLIP RDF comparison:
  `analyze-existing`.
- A tiny local integration test from an explicit ASE-MD source:
  `md-smoke-and-analyze`.
- Production trajectory generation: use `$ase-md`, `$lammps-md`, or reviewed AIMD
  first; do not expand the smoke operation into production MD.

## Artifact first

Reuse final `OK` trajectory and metadata artifacts from upstream MD or AIMD nodes.
MLIPFlow supplies collected restart segments and their recorded timing, temperature,
species, model, and structure information. Do not ask the user to copy, rename,
concatenate, or manually redescribe native artifacts. Reuse a completed compatible
analysis rather than rerunning MD or transport merely to recreate a workflow shape.

Use standalone historical files only through their supported compatibility path. Keep
the legacy manuscript carrier-count convention isolated: its historical `N=7` value is
parity evidence, not a physical default or a rule for formal analysis.

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

## Execute and interpret

Use `mlipflow inspect` and the dry-run; let Adapter validation define the detailed input
and output contract. Follow the effective `approval_required` value in that plan rather
than a hard-coded Skill rule. Never call the packaged analysis runner directly or bypass
Adapter `validate/plan/execute/check/collect`. Success requires final plugin `OK`.

Report source type, temperatures, mobile species, window, formal analysis mode, direct
versus extrapolated results, and unavailable quantities. `OK` establishes internal
contract consistency, not equilibration, a diffusive regime, finite-size convergence,
replica convergence, model accuracy, Nernst-Einstein validity, or safe long-range
extrapolation. Replay remains a structured collection of existing results, not fresh
analysis.
