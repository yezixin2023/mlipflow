# LASP stochastic surface walking / random-walk sampling

This example is for the LASP code developed by Zhi-Pan Liu's group at Fudan University. LASP's stochastic surface walking (SSW) method explores the potential-energy surface by repeatedly perturbing a structure along stochastic directions and relaxing it. In MLIPFlow this belongs to `pes-sampling`, not `mlip-training`.

The repository implements the execution and normalization path in `plugins/pes-sampling/lasp_ssw.py` and exposes it through the `pes-sampling` adapter operation `lasp-ssw-execute`. `plugins/pes-sampling/lasp_random_walk.py` is the user-facing alias and defaults to the execute path. The wrapper stages `input.arc` and `lasp.in`, runs the user-supplied licensed LASP executable without a shell, parses `allstr.arc`, preserves walk order, and emits `sampling-result.json`, `ssw-structures.json`, and `selected-structures.json`. Optional `best.arc` and `md.arc` can also be normalized.

`lasp.in` in this directory is only a minimal SSW contract fixture. Replace the potential setup and add the auxiliary files required by your real LASP case. The included `potential vasp` line is not a recommendation to use VASP for production sampling.

Direct smoke/run command using your own LASP installation and ARC structure:

```bash
python plugins/pes-sampling/lasp_random_walk.py --lasp-executable /ABS/PATH/TO/lasp --input-structure /ABS/PATH/TO/input.arc --lasp-input examples/lasp_random_walk/lasp.in --output-dir "$PWD/lasp-random-walk-out" --historical-source-id case://lasp/random-walk-demo --selection-stride 1 --max-frames 1000 --seed-status HISTORICAL_PARAMETER_UNKNOWN --lasp-version YOUR_LASP_VERSION
```

For MPI execution, add `--mpi-launcher /ABS/PATH/TO/mpirun --mpi-processes N`.

To keep every third accepted walk structure, set `--selection-stride 3`. To additionally reject structures above an energy threshold, add `--energy-max-ev VALUE`; stride is applied after the energy filter. Add `--include-best-arc` or `--include-md-arc` only when the corresponding LASP outputs are expected.

The current contract intentionally does not synthesize undocumented LASP keywords or attempt to reproduce LASP's internal random number generator. The native `lasp.in` remains the source of truth for SSW settings such as `SSW.SSWsteps` and `SSW.Temp`.
