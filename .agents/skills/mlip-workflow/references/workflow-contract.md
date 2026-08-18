# MLIP workflow orchestration contract

Use this reference to route from existing artifacts and workflow state. Current plugin
manifests, inspected plans, Adapter results, and specialist Skills remain authoritative.

## Contents

- [Capability map](#capability-map)
- [Artifact-first routing](#artifact-first-routing)
- [Optional and conditional stages](#optional-and-conditional-stages)
- [Replay versus fresh execution](#replay-versus-fresh-execution)
- [Approval and compute boundaries](#approval-and-compute-boundaries)
- [Model and MD routing](#model-and-md-routing)
- [Voltage exception](#voltage-exception)
- [Artifact handoff](#artifact-handoff)
- [Failure and retry](#failure-and-retry)
- [Scientific claim vocabulary](#scientific-claim-vocabulary)
- [Natural-language acceptance matrix](#natural-language-acceptance-matrix)

## Capability map

| Goal/stage | Owner | Current routing boundary |
|---|---|---|
| Seeded high-entropy/SQS candidates | `$high-entropy-structure` | Explicit prototype, one alloy sublattice, integer counts; local |
| PES sampling/selection | `$pes-sampling` | DIRECT local; LASP local or reviewed SSH-SLURM; historical ARC replay |
| VASP preparation, DFT labels, and framework datasets | `$dft-labeling` | Separate prepare/label; canonical dataset; reviewed SSH-SLURM DeepMD/M3GNet/CHGNet/MACE assembly |
| MLIP training/fine-tuning | `$mlip-training` | DeepMD, M3GNet/MatGL, CHGNet, MACE; local or supported SSH-SLURM |
| E/F/S benchmark or evidence replay | `$mlip-benchmark` | Fresh inference, metric recomputation, or historical replay; local |
| ASE MLIP MD | `$ase-md` | Explicit DeepMD, M3GNet/MatGL, CHGNet, or MACE; reviewed SSH-SLURM NVT/NPT |
| LAMMPS MLIP MD | `$lammps-md` | LAMMPS-ready DeepMD, MACE, or MatGL/M3GNet; prepare/execute/restart contracts |
| MSD/diffusion/conductivity | `$ionic-transport` | Existing ASE/LAMMPS/VASP trajectory or MSD; local analysis |
| Single-metric top-k | `$candidate-ranking` | Existing candidate and numeric metric manifests; local |
| Li energy-to-voltage or SI replay | `electrochemical-voltage` plugin | Direct local plugin orchestration; no standalone voltage Skill |

Never recreate a standalone electrochemical-voltage Skill. Load each selected specialist Skill at the
time its stage becomes current, then inspect the current plugin/CLI contract rather than
relying on this summary for parameters.

## Artifact-first routing

| Verified inputs already available | Objective | Enter or continue at |
|---|---|---|
| Prototype plus explicit composition/count requirements | Generate disordered structures | `$high-entropy-structure` |
| Existing generated or user-supplied structure candidates | Add labels | `$dft-labeling`; do not regenerate SQS |
| Structure ensemble needing representative selection | Reduce/select PES coverage | `$pes-sampling`, only if selection is scientifically required |
| Verified sampled/selected structures | Produce labels | `$dft-labeling` |
| Verified DFT canonical dataset, no framework view | Train or fine-tune | `$dft-labeling` `dataset-assemble` for only the requested frameworks, then `$mlip-training` |
| Verified matching DeepMD/M3GNet/CHGNet/MACE dataset reference | Train or fine-tune | `$mlip-training`; do not rerun VASP or reconvert |
| Explicit labeled dataset plus one or more trained models | Compare predictions | `$mlip-benchmark` |
| Trained model plus initial structure | Generate a new trajectory | `$ase-md` or `$lammps-md` according to runtime compatibility and goal |
| Existing ASE/LAMMPS/VASP trajectory or MSD | Compute transport | `$ionic-transport`; do not rerun MD |
| Candidate manifest plus comparable numeric metrics | Select top-k | `$candidate-ranking` |
| Li-content total-energy sequence | Compute voltage | `electrochemical-voltage` plugin `compute-from-energies` |
| Existing standard result/evidence manifests | Verify/reconstruct history | Applicable replay/check path; do not default to fresh execution |

An artifact is reusable only when its identity, scientific parameters, and completion
contract are known. File existence, filename, scheduler state, or a JSON `status` field
alone is insufficient. If a supplied artifact lacks checker evidence, enter at its
validation/replay boundary rather than treating its upstream stage as complete.

## Optional and conditional stages

A possible long workflow is:

```text
high-entropy structures -> optional PES sampling -> DFT labels -> training
-> benchmark -> optional MD -> optional transport -> optional ranking
-> optional voltage post-processing
```

This is a map, not a template:

- Skip SQS when fit-for-purpose structures already exist.
- Include PES sampling only when the objective needs representative/configurational
  coverage; it is not a ceremonial prerequisite for labeling.
- Skip training when the user supplies an explicit compatible trained model.
- Benchmark before model selection when a comparison claim is required; do not require
  it merely to honor an explicit user-selected model for a scoped run.
- Skip MD when a suitable verified trajectory already exists.
- Run transport only when transport is an objective.
- Rank only when comparable candidate metrics and a policy exist.
- Run voltage independently when its energy/evidence inputs exist.

Find the earliest missing **necessary** stage for the stated objective after these
conditions are applied.

## Replay versus fresh execution

Prefer replay/check when the user supplies historical evidence or an existing standard
result manifest. Replay reads and validates evidence under the relevant plugin contract;
it does not run the original numerical program, regenerate structures, train a model,
run DFT/MD, or establish independent scientific validation.

For manuscript reproduction, inventory evidence per stage and preserve one of these
labels:

- `REPLAY_VERIFIED`: existing evidence was normalized/checked under a replay contract;
- fresh execution: the numerical implementation ran in this workflow attempt;
- local integration smoke: a bounded integration/determinism exercise only;
- numerical parity: an explicitly defined result comparison supported by source data;
- `MISSING_SOURCE`: required historical evidence is absent.

Never replace `MISSING_SOURCE` with synthesized data. Never launch a fresh expensive
rerun merely because replay cannot establish a stronger claim. Ask whether the user
wants a separately scoped fresh execution.

## Approval and compute boundaries

Treat large SQS generation, DFT/AIMD, MLIP training/fine-tuning, fresh benchmark
inference, long MD, large screening, and scheduled jobs as expensive. Before execution:

1. inspect current persistent state and the selected node;
2. run the appropriate dry-run;
3. show the stage, exact input identities, scale, backend, abstract resources, expected
   artifacts, freshness/overwrite semantics, and why execution is needed;
4. use the approval token returned for that exact plan.

Do not auto-submit because the user requested an “end-to-end” or “fully automatic” run.
Do not widen resources, candidate counts, MD duration, DFT scope, or training schedule
without review. A smoke configuration never silently becomes a production configuration.

The orchestrator may declare supported abstract `backend`, `cpus`, `gpus`, `memory`, and
`walltime`. Site-owned configuration/templates own SSH aliases, hosts, partitions,
accounts/QoS, modules, conda environments, executables/launchers, remote template roots,
and work roots. Do not guess or embed them. Respect each plugin's current backend list;
in particular, keep local-only plugins local.

## Model and MD routing

Choose a model only from:

- the user's explicit exact model/family and artifact;
- comparable `$mlip-benchmark` evidence;
- current routing evidence plus an explicit metric policy.

Without comparable task/scenario/split/metric/unit evidence, report that “best” is not
established. Never hard-code or imply a DeepMD, MACE, CHGNet, or M3GNet brand preference.
A benchmark winner is conditional on its evidence and policy, not universally best.

For a new trajectory:

- route explicit DeepMD, M3GNet/MatGL, CHGNet, or MACE calculators targeting ASE
  NVT/NPT to `$ase-md`;
- route an already LAMMPS-compatible DeepMD, MACE, or MatGL/M3GNet model requiring
  LAMMPS execution/restart to `$lammps-md`;
- do not assume CHGNet has an audited native LAMMPS pair style; prefer `$ase-md` unless
  explicit compatible evidence and a supported contract exist;
- do not choose one MD engine merely for workflow uniformity.

If a suitable trajectory or MSD already exists, skip both MD routes and enter
`$ionic-transport` directly. A final `OK` ASE-MD or LAMMPS-MD node supplies its collected
artifacts natively: do not ask for internal filenames, timestep, temperature, type map,
manual metadata, rename/copy, or concatenation. Preserve all collected restart attempts;
ionic-transport orders their frames by global step and removes a repeated boundary.
Multiple completed ASE and/or LAMMPS temperature nodes may feed one Arrhenius analysis.
If a new trajectory is necessary, accept it downstream only after the MD Skill's
completion checks and collected artifact identity succeed.

## Voltage exception

There is intentionally no standalone electrochemical-voltage Skill. Invoke the existing local
`electrochemical-voltage` plugin directly:

- `compute-from-energies` for an explicit, valid Li-content total-energy sequence;
- `replay-si-table-s11` for the supported historical evidence contract.

Preserve its formula, units, provenance, replay, and claim boundaries. Do not implement
the voltage equation in the orchestrator, infer missing energies, run implicit DFT/MLIP,
or recreate a standalone voltage Skill.

## Artifact handoff

At every edge, bind the exact collected upstream artifact rather than a guessed path.
Record logical identity, fingerprint, schema/version, scientific parameters, producing
plugin/attempt, and completion status where the downstream contract supports them.

Before advancing, require:

- upstream Adapter `check/collect` and final plugin `OK`;
- no fingerprint or parameter drift;
- downstream inputs reference the approved collected artifacts;
- units, composition/model identity, task/scenario/split, and other scientific
  conventions needed downstream are explicit;
- a fresh downstream attempt with no silent overwrite.

Re-inventory after every completed stage because newly collected artifacts may make
planned intermediate stages unnecessary or reveal a new required decision.

For DFT-to-training handoff, keep the canonical `dataset_id`, shared `split_id`, and each
final framework artifact reference explicit. `dataset-assemble` owns the deterministic
or group-aware record assignment and all four split-preserving serializations. Its
collected `*-dataset-reference.json` is the direct `mlip-training.dataset_reference`,
and its necessary final artifact fingerprint is the training node's
`dataset_fingerprint`. Reuse a collected reference; do not generate a new semantic
dataset for a repeated downstream request. Benchmark the four trained models only on
the shared held-out test record IDs.

## Failure and retry

Do not invent workflow-level retry semantics:

- stop downstream on `FAIL`, `BLOCKED`, or `STOPPED`;
- treat scheduler `COMPLETED` only as permission to fetch/check, not scientific success;
- use MLIPFlow core and the specialist Adapter's existing retry/salvage contract;
- create a fresh retry attempt and preserve the old attempt;
- let `$ase-md` or `$lammps-md` own checkpoint/restart compatibility and salvage;
- do not delete outputs, edit result JSON, bypass a checker, or skip a failed stage;
- after repair and final `OK`, re-evaluate downstream eligibility from collected output.

## Scientific claim vocabulary

| Evidence | Permitted statement | Forbidden strengthening |
|---|---|---|
| Implementation/tests complete | Contract implementation is complete | Numerical parity is complete |
| Integration smoke | Bounded integration path passed | Production-scale validation passed |
| Replay | Existing evidence was read/validated | Fresh numerical computation ran |
| Deterministic seeded run | Declared run is reproducible under its contract | Historical byte parity is established |
| Scheduler `COMPLETED` | Scheduler process ended successfully | Scientific result is valid |
| Generated candidate | Candidate was generated and checked | It is the best material |
| Benchmark winner | Winner under stated comparable evidence/policy | Universally best model |
| Candidate ranking | Deterministic ordering by stated metric/policy | High-fidelity validation |

Never relax a stronger or more specific limitation stated by the selected specialist
Skill, plugin, evidence report, or source audit.

## Natural-language acceptance matrix

Use these cases as route checks. Each decision assumes the described artifact is usable;
otherwise validate/replay it first.

| # | User intent/state | Required orchestration decision |
|---:|---|---|
| 1 | “I only have a prototype and want a high-entropy MLIP workflow from scratch.” | Start at `$high-entropy-structure`; do not jump to training. |
| 2 | “The structures already exist; I only want DFT labels.” | Start at `$dft-labeling`; do not regenerate SQS. |
| 3 | “I already have labels; train CHGNet.” | Start at `$mlip-training` with the explicit framework. |
| 4 | “Training is done; compare six models.” | Start at `$mlip-benchmark` with models and a labeled test set/evidence. |
| 5 | “I have a MACE model and structure; run NVT.” | Enter `$ase-md` or `$lammps-md` from runtime compatibility and requested engine; skip unrelated upstream. |
| 6 | “The trajectory exists; calculate conductivity.” | Start at `$ionic-transport`; do not rerun MD. |
| 7 | “I have 247 conductivity results; select top 10.” | Use `$candidate-ranking`; require metric, direction, `top_k=10`, and missing policy. |
| 8 | “Reproduce all paper results.” | Inventory historical evidence and prefer replay; do not default to fresh expensive reruns. |
| 9 | “The Slurm job is COMPLETED, so the result is valid, right?” | Continue fetch and scientific `check/collect`; do not claim `OK` yet. |
| 10 | “Which MLIP is best? Pick one arbitrarily.” | Refuse brand preference without comparable benchmark/routing evidence. |
| 11 | “Run everything automatically; do not ask me.” | Preserve every expensive-operation dry-run and approval boundary. |
| 12 | “I have a DFT dataset; recompute everything from SQS.” | Reuse verified labels and start later by default; rerun only with explicit scoped intent. |
| 13 | “Run CHGNet in LAMMPS.” | Do not invent a native pair style; route to `$ase-md` or require explicit audited LAMMPS compatibility. |
| 14 | “I have a total-energy sequence; compute voltage.” | Call `electrochemical-voltage` plugin directly; do not seek a voltage Skill. |
| 15 | “A generation-result.json exists; regenerate it to confirm.” | Prefer `$high-entropy-structure` replay/check; fresh rerun requires explicit intent. |
| 16 | “The upstream check failed; continue downstream.” | Stop downstream until the upstream stage is repaired and returns final `OK`. |
| 17 | “把这些 DFT labels 分别用于 DeepMD、M3GNet、CHGNet 和 MACE 训练。” | Verify/reuse the canonical DFT dataset, run one reviewed remote `dataset-assemble` for the requested views if missing, then bind its four references to four `$mlip-training` nodes; no custom converter script. |
| 18 | “用我完成的 ASE MD 轨迹计算 Li 离子电导率。” | Route the completed `ase-md` node's collected artifacts directly to `$ionic-transport`; auto-read md-result/index and stitch restart segments; do not request files or MD metadata. |
| 19 | “用我完成的 LAMMPS MD 轨迹计算 Li 离子电导率。” | Route the completed `lammps-md` node's collected artifacts directly to `$ionic-transport`; auto-read result/manifest, type map and timing, accept old/new coordinate columns, and stitch restart segments; do not rerun MD. |
