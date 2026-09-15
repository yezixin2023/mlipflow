---
name: mlip-workflow
description: Orchestrate auditable end-to-end MLIP research workflows by identifying the user's final goal, inventorying existing artifacts and workflow state, selecting the earliest missing necessary stage, and handing each stage to the current MLIPFlow specialist Skill or plugin. Use for multi-stage planning, resuming partial workflows, manuscript replay, avoiding redundant expensive work, or coordinating structure generation, PES sampling, DFT labeling, MLIP training/benchmarking, offline active learning, MD, ionic transport, candidate ranking, and voltage analysis; do not implement scientific algorithms or bypass specialist contracts.
---

# MLIP workflow

Act as a lightweight orchestrator for existing MLIPFlow capabilities. Do not impose a
canonical pipeline, implement scientific calculations, or duplicate specialist and
Adapter contracts.

## Select the next necessary stage

1. State the user's final scientific objective.
2. Inspect persistent MLIPFlow state and inventory the supplied and collected artifacts.
3. Reuse artifacts accepted by the producing Adapter; validate or replay unverified
   evidence instead of repeating an expensive upstream stage.
4. Skip irrelevant or already completed stages and select the earliest missing
   prerequisite for the objective.
5. Load the current specialist Skill for that stage, then use `mlipflow inspect` and the
   dry-run as the executable contract.
6. After final plugin `OK`, pass collected artifacts downstream and reconsider the next
   necessary stage.

Continue through deterministic downstream work when no new scientific choice or
approval is required. Stop on a real scientific failure, unresolved information, or a
new costly scope whose authorization is missing. Reuse explicit user authorization
for the same task and scale, including its observation and deterministic postprocessing.

Route structure generation to `$high-entropy-structure`, PES selection or LASP to
`$pes-sampling`, VASP preparation/DFT/dataset assembly to `$dft-labeling`, training to
`$mlip-training`, model evaluation to `$mlip-benchmark`, finite offline campaigns to
`$mlip-active-learning`, trajectory generation to `$ase-md` or `$lammps-md`, transport
analysis to `$ionic-transport`, and metric-based top-k selection to
`$candidate-ranking`. Use the `electrochemical-voltage` plugin directly for voltage;
there is no voltage Skill.

Read [references/workflow-contract.md](references/workflow-contract.md) only for
manuscript or historical reproduction, ambiguous legacy evidence, or cross-stage claim
interpretation. It is not required for ordinary stage selection or artifact handoff.

## Preserve scientific decisions

Do not choose a model by brand when the objective requires comparison evidence. Require
an explicit model choice or comparable benchmark/routing evidence and an explicit
metric policy. Preserve shared dataset splits when models are to be compared.

Keep replay, smoke, fresh execution, scheduler completion, and final scientific `OK`
distinct. PES coverage does not establish transport convergence, and a generated or
ranked candidate is not thereby a validated material.

## Execute through MLIPFlow

```bash
mlipflow --project PROJECT json
mlipflow --project PROJECT inspect NODE
mlipflow --project PROJECT --format json run NODE --dry-run
mlipflow --project PROJECT --format json run NODE --approve
```

Use `--approve` for the effective `approval_required` task after its inputs, scale and
resources have been reviewed and authorized. Local deterministic nodes can use
`run NODE` directly. `advance` reconciles scheduled jobs and dependency state; it
does not launch another node.

After final plugin `OK`, use `artifacts[].role` and `artifacts[].path`, or bind the
next input as `{from_node: NODE, role: ROLE}`. `metrics`, `check`, `collection`,
`manifest_path` and `logs` expose the next decision without scanning attempt folders.
Do not continue from a failed node. For a complete multi-stage configuration see
`examples/training_all_models/dft-to-all-training.yaml`; single-task recipes are in
`docs/USAGE.md` and the installed `share/mlipflow/examples/` directory.

## Report

Report the objective, reusable verified artifacts, important gaps, the concise stage
path, the current next stage, and its effective approval requirement. After each stage,
report final status, collected artifacts, scientific limitations, and the next eligible
stage without strengthening the specialist's claim language.
