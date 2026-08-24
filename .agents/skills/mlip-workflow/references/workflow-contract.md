# Exceptional workflow guidance

Read this reference only for manuscript or historical reproduction, ambiguous legacy
artifacts, or claim interpretation that spans several stages. Ordinary workflow
routing should use the thin specialist Skill, `mlipflow inspect`, the effective
dry-run, and Adapter results.

## Historical and manuscript reproduction

Inventory the available source files, standard result manifests, parameters, and
completion evidence for each requested claim. Keep unavailable evidence unavailable;
never synthesize it or launch a fresh expensive calculation merely to make a historical
story look complete.

Use generic replay for an existing standard result manifest. Use a plugin's named
normalization/replay operation when legacy raw files need scientific parsing. In both
cases, describe the outcome as a **structured collection of existing results**. State
whether the original numerical program ran in the current attempt and whether any
numerical comparison against historical evidence was actually possible.

Fresh execution may be offered as a separately scoped task, but it is not historical
parity by itself. Preserve differences in software, inputs, seeds, units, conventions,
and missing source evidence.

## Ambiguous artifacts and handoff

File presence, a familiar filename, scheduler state, or a self-reported status is not
enough to reuse an artifact. First establish whether it was accepted by the applicable
Adapter `check/collect` or can be validated through replay. If it cannot, stop at that
validation boundary rather than repeating unrelated upstream stages.

Bind exact collected artifacts downstream. Do not ask the user to copy files out of
attempt directories, type a superseded attempt path, or reconstruct metadata already
recorded by a producer. Let current core resolution and the downstream Adapter validate
the binding.

When several framework datasets or models will be compared, preserve one canonical
record split and the same held-out IDs. When MD restarts produced several accepted
segments, preserve them all for downstream transport; do not concatenate them by hand.

## Routing questions that need judgment

| Situation | Scientific decision |
|---|---|
| Existing structures, requested DFT labels | Start at `$dft-labeling`; SQS and PES are not ceremonial prerequisites. |
| Existing canonical labels, requested training | Reuse them and create only missing framework views before `$mlip-training`. |
| Existing compatible model, requested trajectory | Choose `$ase-md` or `$lammps-md` from the requested engine and actual runtime compatibility. |
| Existing trajectory or MSD, requested conductivity | Start at `$ionic-transport`; do not rerun MD. |
| Several models, requested “best” | Require comparable `$mlip-benchmark` evidence and a metric policy; do not choose by brand. |
| Candidate metrics, requested top-k | Use `$candidate-ranking`; the ranking policy remains a human choice. |
| Finite offline acquisition campaign | Use `$mlip-active-learning`; begin at the earliest missing artifact in the current round. |
| Li-content energy sequence, requested voltage | Use the `electrochemical-voltage` plugin directly; do not implement its equation here. |

## Cross-stage claim boundaries

| Evidence | Permitted statement | Do not strengthen to |
|---|---|---|
| Contract or unit tests | The implementation contract passed | Production numerical validation |
| Integration smoke | The bounded software path passed | Production-scale or converged science |
| Replay | Existing evidence was read and validated | Fresh computation or independent validation |
| Seeded generation | The declared run is reproducible under its contract | Historical byte parity or global optimum |
| Scheduler `COMPLETED` | The scheduled process ended | Scientific result `OK` |
| Benchmark winner | Best under the stated comparable evidence and policy | Universally best model |
| PES active-learning convergence | Passed declared PES-domain gates | Transport convergence |
| Candidate ranking | Ordered by the stated metric and policy | Material validation |

Never relax a more specific limitation reported by a specialist Skill, an inspected
plan, an Adapter result, or the underlying evidence.
