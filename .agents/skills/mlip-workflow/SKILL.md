---
name: mlip-workflow
description: Orchestrate auditable end-to-end MLIP research workflows by identifying the user's final goal, inventorying existing artifacts and workflow state, selecting the earliest missing necessary stage, and handing each stage to the current MLIPFlow specialist Skill or plugin. Use for multi-stage planning, resuming partial workflows, manuscript replay, avoiding redundant expensive work, or coordinating structure generation, PES sampling, DFT labeling, MLIP training/benchmarking, offline active learning, MD, ionic transport, candidate ranking, and voltage analysis; do not implement scientific algorithms or bypass specialist contracts.
---

# MLIP workflow

Act as the control plane for existing MLIPFlow capabilities. Do not create a new
scientific implementation, force every request through a fixed pipeline, or repeat an
expensive stage merely to make a workflow look complete.

Read [references/workflow-contract.md](references/workflow-contract.md) when selecting
an entry point, connecting stages, choosing replay versus fresh execution, routing MD,
or handling approval/failure. For every selected stage, read its current specialist
Skill and plugin manifest; keep their scientific and safety boundaries authoritative.

## Determine the next stage

1. State the user's final scientific objective.
2. Inspect current MLIPFlow state with read-only commands and inventory actual inputs,
   outputs, paths/parameters, and completion evidence.
3. Distinguish merely present artifacts from artifacts accepted by the applicable
   Adapter `check/collect`. Validate or replay unverified evidence before reuse.
4. Mark completed necessary stages and omit irrelevant optional stages.
5. Select the earliest missing prerequisite for the objective. Start there, not at the
   beginning of a canonical diagram.
6. Delegate that stage to the matching specialist Skill. Use the
   `electrochemical-voltage` plugin directly for voltage; no voltage Skill exists.
7. Expand only the current stage. After it returns final plugin `OK`, bind its verified
   collected artifacts as downstream inputs and reconsider the next stage.

For DFT-to-training requests, treat `dft-labeling` canonical collection and remote
`dataset-assemble` as distinct artifact gates. Reuse exact framework references when
present; otherwise assemble only the requested DeepMD/M3GNet/CHGNet/MACE views, then
hand their collected reference manifests to `$mlip-training`. Never ask the user for a
custom conversion script. Review one framework-independent split before serialization;
all requested framework views must preserve that exact split.

For a same-test-set model comparison, also route the collected
`benchmark-dataset-reference.json` from that assembly to scheduled `$mlip-benchmark`
nodes, then normalize their prediction evidence into one joint ranking. Do not create a
second split or derive benchmark labels from framework-specific training views.

Never continue downstream from `FAIL`, `BLOCKED`, `STOPPED`, a scheduler-only
`COMPLETED`, or process exit zero without scientific completion checks.

For an ionic-transport objective, treat a final `OK` AIMD `dft-labeling`, `ase-md`, or
`lammps-md` node as the trajectory handoff. Pass its collected dependency artifacts to
`$ionic-transport` directly; core includes preserved restart attempts and the transport loader joins them
by global step. Do not ask the user to identify filenames, rename/copy trajectories,
concatenate attempts, write metadata, or repeat timestep, temperature, or atom-type
information already recorded upstream. When several completed temperature nodes are
selected, hand all of them to one analysis for D(T), conductivity, Arrhenius Ea, and
the requested extrapolated temperature. Do not rerun MD merely to change a dump name
or coordinate convention.

## Apply approval and execution boundaries

Use the current node dry-run's effective `approval_required` value. Approval is required
for SSH-SLURM submission and for local operations declared expensive by the capability;
ordinary local analysis, preparation, selection, checking, normalization, replay, and
post-processing continue without an approval stop. Scientific validation remains
independent: missing parameters still block or fail and must never be guessed.

Before approval-required execution, show the stage, verified inputs, expected artifacts,
backend/resources, cost class, and approval requirement. Follow the MLIPFlow
dry-run/boolean-approval lifecycle; do not auto-submit, enlarge resources, convert smoke
settings into production settings, or rerun existing expensive evidence without explicit
intent. One explicitly reviewed batch of expensive nodes may receive one user approval;
execute only those reviewed nodes with `--approve` internally. Do not create or imply a
persistent batch-approval mechanism; there is no persistent batch approval.

After an approved expensive stage reaches final `OK`, continue eligible local deterministic
check, collect, normalization, transport/Arrhenius analysis, active-learning assessment, and
ranking without asking again. Request new approval only for a new expensive or scheduled
execution that was not part of the reviewed batch.

Declare only abstract `backend`, `cpus`, `gpus`, `memory`, and `walltime` when supported
by the selected plugin. Never guess site-owned SSH hosts, partitions, accounts, modules,
environments, executables, remote roots, or templates. Never route a local-only plugin
to `ssh-slurm`.

## Preserve specialist ownership

Use these Skills rather than copying their domain rules:

- `$high-entropy-structure`, `$pes-sampling`, `$dft-labeling`;
- `$mlip-training`, `$mlip-benchmark`;
- `$ase-md`, `$lammps-md`, `$ionic-transport`;
- `$candidate-ranking`, `$mlip-active-learning`.

Do not select a model by brand. Require an explicit user choice or comparable benchmark,
routing, and metric-policy evidence. Do not sort candidates yourself. Do not implement
the voltage formula, SQS search, DFT, training, inference, MD, transport analysis, or
ranking in this Skill.

## Report progressively

For the first response, report only:

- objective;
- verified existing artifacts and important gaps;
- concise suggested stage path, with optional stages labeled;
- current next stage and why;
- whether that stage is expensive or needs approval.

After each stage, report its status, collected artifacts and scientific metadata, next eligible
stage, and any new approval. State whether the original numerical program ran when relevant, and
never strengthen the scientific claim language defined by a specialist Skill.
