# Compact AIMD-reference validation

This example shows the smallest artifact path that exercises the implemented
contracts. It does not include numerical results and does not launch calculations by
itself.

`project.yaml` assumes two user-local site profiles:

- `cpu-cluster` owns the reviewed VASP templates and CPU scheduler environment;
- `model-compute` owns the reviewed MLIP model environment and ASE-MD templates.

The project contains neither site paths nor scheduler setup. Review the AIMD
deck first, run its dry-run, and approve the VASP job separately. Do the same for each
MLIP-MD node. The local comparison nodes need no remote model runtime: they receive the
collected trajectories from their direct dependencies and use one
`ionic-transport` implementation for VASP, ASE, and LAMMPS sources.

The path is:

```text
reviewed VASP AIMD on cpu-cluster ─┐
                                  ├─ local transport + explicit-pair RDF ─┐
MLIP-MD on model-compute ─────────┘                                       │
                                                                           ├─ local mlip-benchmark
second MLIP-MD on model-compute ── local transport + explicit-pair RDF ────┘
```

Each comparison writes `aimd_mlip_comparison.json`. These files contain only values
available from the analyzed trajectories; missing Ea or conductivity stays absent.
The final `normalize-execute` node accepts the explicit model list and ranks
`structural-dynamics` and `ionic-transport` separately. Existing static energy/force
`prediction_evidence.json` files may be added to the same node as additional
`static-pes` evidence; no cross-task score is formed.

Add separately reviewed temperature nodes to each trajectory set when D(T), Ea, or
target-temperature quantities are required. Use `fit_scope: dataset` so AIMD and each
MLIP retain separate Arrhenius fits. The RDF temperature selects one shared-temperature
trajectory pair and uses the same trajectory window already approved for transport.
