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
| Seeded high-entropy/SQS candidates | `$high-entropy-structure` | Explicit prototype, one alloy sublattice, integer counts; ordinary local generation is not approval-gated |
| PES sampling/selection | `$pes-sampling` | DIRECT local; LASP local or reviewed SSH-SLURM; verified local DIRECT+LASP merge; historical ARC replay |
| VASP preparation, DFT labels, and framework datasets | `$dft-labeling` | Local preparation/assembly is ungated; label and scheduled assembly require approval; calculation-level fresh retry; canonical dataset; shared split plus framework views and benchmark test reference |
| MLIP training/fine-tuning | `$mlip-training` | DeepMD, M3GNet/MatGL, CHGNet, MACE; local or supported SSH-SLURM |
| E/F/S benchmark or evidence replay | `$mlip-benchmark` | Fresh inference local or reviewed SSH-SLURM; metric recomputation and historical replay local |
| Finite offline active-learning campaign | `$mlip-active-learning` | One- or two-model calibrated committees; immutable round projects; local deterministic decisions with existing training/MD/DIRECT/DFT/benchmark stages |
| ASE MLIP MD | `$ase-md` | Explicit DeepMD, M3GNet/MatGL, CHGNet, or MACE; reviewed SSH-SLURM NVT/NPT |
| LAMMPS MLIP MD | `$lammps-md` | LAMMPS-ready DeepMD, MACE, or MatGL/M3GNet; prepare/execute/restart contracts |
| MSD/diffusion/conductivity and bounded AIMD/MLIP RDF comparison | `$ionic-transport` | Existing ASE/LAMMPS/VASP trajectory or MSD; local analysis |
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
| Shared-split benchmark reference plus published model references | Compare fresh predictions on the exact same test IDs | One scheduled `$mlip-benchmark` per model, then one local joint normalization |
| Explicit active-learning policy plus verified split/models/labels | Run or resume a finite offline campaign | `$mlip-active-learning`; enter at the earliest missing artifact in the current immutable round |
| Trained model plus initial structure | Generate a new trajectory | `$ase-md` or `$lammps-md` according to runtime compatibility and goal |
| Existing ASE/LAMMPS/VASP trajectory or MSD | Compute transport or one explicit-pair AIMD/MLIP RDF comparison | `$ionic-transport`; do not rerun MD |
| Candidate manifest plus comparable numeric metrics | Select top-k | `$candidate-ranking` |
| Li-content total-energy sequence | Compute voltage | `electrochemical-voltage` plugin `compute-from-energies` |
| Existing standard result/evidence manifests | Verify/reconstruct history | Applicable replay/check path; do not default to fresh execution |

An artifact is reusable only when its path, scientific parameters, and completion
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

## Reusing results versus fresh execution

Use generic core replay when the user supplies an existing standard result manifest. If raw
historical files require a plugin operation such as `normalize-replay`, run that named operation
through the normal Adapter execution lifecycle. Neither path runs the original numerical program,
regenerates structures, trains a model, runs DFT/MD, or establishes independent scientific
validation.

For manuscript reproduction, inventory the available files per stage and report plainly whether
the original numerical program ran, whether only bounded integration was tested, and whether an
explicit numerical comparison was performed. Missing historical source data must remain missing;
never synthesize it or launch a fresh expensive rerun merely to strengthen a claim. Ask whether the
user wants a separately scoped fresh execution.

## Approval and compute boundaries

Use each dry-run's effective `approval_required` value. Replay is never approval-gated.
Every SSH-SLURM execution requires approval. Local DFT labeling, MLIP training/fine-tuning,
ASE/LAMMPS MD execution, fresh benchmark inference, and LASP/SSW execution are declared
expensive operations and require approval. Ordinary local analysis, preparation,
selection, checking, normalization, SQS generation, active-learning decision operations,
transport/Arrhenius post-processing, ranking, and voltage analysis do not.

Before an approval-required execution:

1. inspect current persistent state and the selected node;
2. run the appropriate dry-run;
3. show the stage, exact input paths, scale, backend, abstract resources, expected
   artifacts, freshness/overwrite semantics, and why execution is needed;
4. use the explicit boolean approval after review.

One explicitly reviewed batch of expensive nodes may be approved once and then executed
with `--approve` internally, but only those reviewed nodes are covered. Core has no
persistent batch approval. After an approved stage completes, continue downstream local
deterministic check/collect, normalization, transport/Arrhenius analysis, active-learning
assessment, and ranking without another approval. A new expensive or scheduled execution
outside the reviewed batch requires a new approval.

Do not auto-submit because the user requested an “end-to-end” or “fully automatic” run.
Do not widen resources, candidate counts, MD duration, DFT scope, or training schedule
without review. A smoke configuration never silently becomes a production configuration.
Approval is not scientific validation: unresolved species, temperatures, fitting windows,
DFT settings, units, conventions, seeds, or splits still block or fail.

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
Likewise, a final `OK` DFT-labeling AIMD node supplies its collected `vasprun.xml`
trajectory directly. One AIMD reference plus one MLIP trajectory set may request an
explicit-pair RDF comparison and emit metric-only evidence for task-separated
`$mlip-benchmark` normalization.
If a new trajectory is necessary, accept it downstream only after the MD Skill's
completion checks and artifact collection succeed.

## Voltage exception

There is intentionally no standalone electrochemical-voltage Skill. Invoke the existing local
`electrochemical-voltage` plugin directly:

- `compute-from-energies` for an explicit, valid Li-content total-energy sequence;
- `replay-si-table-s11` for the supported historical table operation.

Preserve its formula, units, source paths, and execution boundaries. Do not implement
the voltage equation in the orchestrator, infer missing energies, run implicit DFT/MLIP,
or recreate a standalone voltage Skill.

## Artifact handoff

At every edge, bind the exact collected upstream artifact rather than a guessed path.
Record logical ID, path, schema/version, scientific parameters, producing
plugin/attempt, and completion status where the downstream contract supports them.

For an execute-node input that directly consumes one collected file, use the explicit
project binding `{"from_node": "producer-id", "role": "artifact-role"}`. Put that
binding inside a list when the downstream input expects a list. Use
`"resolve": "parent"` only when the downstream contract consumes the containing
attempt/output directory rather than the file itself. Core resolves bindings only from
a final `OK` direct dependency, requires exactly one unique artifact for the declared
role, confines it to the project, and re-resolves it before execution. Do not write an
`attempt-1` path into the project or turn a failed/retried attempt into an implicit
input.

Before advancing, require:

- upstream Adapter `check/collect` and final plugin `OK`;
- matching paths and scientific parameters;
- downstream inputs reference the approved collected artifacts;
- units, composition/model records, task/scenario/split, and other scientific
  conventions needed downstream are explicit;
- a fresh downstream attempt with no silent overwrite.

Re-inventory after every completed stage because newly collected artifacts may make
planned intermediate stages unnecessary or reveal a new required decision.

For DFT-to-training handoff, keep the canonical `dataset_id`, shared `split_id`, and each
final framework artifact reference explicit. `dataset-assemble` owns the deterministic
or group-aware record assignment and all four split-preserving serializations. Its
collected `*-dataset-reference.json` is the direct `mlip-training.dataset_reference`.
Reuse a collected reference; do not generate a new semantic
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
| 11 | “Run everything automatically; do not ask me.” | Preserve every effective expensive/scheduled dry-run and approval boundary; continue ungated local deterministic work without extra approval. |
| 12 | “I have a DFT dataset; recompute everything from SQS.” | Reuse verified labels and start later by default; rerun only with explicit scoped intent. |
| 13 | “Run CHGNet in LAMMPS.” | Do not invent a native pair style; route to `$ase-md` or require explicit audited LAMMPS compatibility. |
| 14 | “I have a total-energy sequence; compute voltage.” | Call `electrochemical-voltage` plugin directly; do not seek a voltage Skill. |
| 15 | “A generation-result.json exists; regenerate it to confirm.” | Prefer `$high-entropy-structure` replay/check; fresh rerun requires explicit intent. |
| 16 | “The upstream check failed; continue downstream.” | Stop downstream until the upstream stage is repaired and returns final `OK`. |
| 17 | “把这些 DFT labels 分别用于 DeepMD、M3GNet、CHGNet 和 MACE 训练。” | Verify/reuse the canonical DFT dataset, run one reviewed remote `dataset-assemble` for the requested views if missing, then bind its four references to four `$mlip-training` nodes; no custom converter script. |
| 18 | “用我完成的 ASE MD 轨迹计算 Li 离子电导率。” | Route the completed `ase-md` node's collected artifacts directly to `$ionic-transport`; auto-read md-result/index and stitch restart segments; do not request files or MD metadata. |
| 19 | “用我完成的 LAMMPS MD 轨迹计算 Li 离子电导率。” | Route the completed `lammps-md` node's collected artifacts directly to `$ionic-transport`; auto-read result/manifest, type map and timing, accept old/new coordinate columns, and stitch restart segments; do not rerun MD. |
