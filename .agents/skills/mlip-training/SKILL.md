---
name: mlip-training
description: Supervise MLIPFlow training and fine-tuning for DeepMD, M3GNet/MatGL, CHGNet, and MACE. Use when choosing a framework, preparing local or SSH-SLURM training nodes, binding labeled datasets and foundation models, selecting train versus finetune, reviewing abstract compute resources, or verifying model/result artifacts.
---

# MLIP training and fine-tuning

Use `plugins/mlip-training` through MLIPFlow. The Skill selects and reviews training
intent; the framework runner and Adapter own deterministic configuration validation,
execution, checking, and collection.

## Route the request

- Train a new DeepMD, M3GNet/MatGL, CHGNet, or MACE model: `train`.
- Start from an explicit compatible foundation model: `finetune`.

Do not silently change frameworks, convert training to fine-tuning, select a foundation
model, enable a training technique, or choose a model by brand. Use the user's exact
choice or comparable routing/benchmark evidence; otherwise expose the unresolved
alternatives.

## Artifact first

Prefer verified dataset references from `$dft-labeling` and preserve their predefined
split. Reuse matching foundation-model and completed model references instead of
copying large artifacts or repeating training. When downstream work needs a stable
published model, use the plugin's reviewed publication path; never overwrite or
manually republish an existing model.

## Scientific judgment

Require an explicit dataset, framework, operation, scientific configuration, seed,
device/precision choice, and resources appropriate to the intended scale. Fine-tuning
also requires an explicit compatible foundation-model artifact. Do not invent
hyperparameters, labels, dataset splits, model compatibility, or convergence criteria.

Distinguish a smoke or short validation run from production training. Model selection
claims require comparable benchmark evidence; training loss or successful artifact
creation alone does not establish superiority.

## Execute and interpret

Use `mlipflow inspect` and the dry-run to review actual artifact bindings, framework,
operation, configuration, backend, abstract resources, and expected outputs. Follow the
effective `approval_required` value rather than maintaining a separate approval table
in this Skill.

Never invoke framework scripts or the scheduler directly, copy site-owned paths into
the project, or bypass Adapter `validate/plan/execute/check/collect`. Scheduler
`COMPLETED` or process exit zero is insufficient; require final plugin `OK`. Retry must
preserve the failed attempt and create a fresh attempt.

Report the framework, train versus fine-tune mode, dataset/split, foundation model when
used, seed, scale, collected model reference, and checker result. `OK` establishes the
declared training contract and artifacts, not generalization, physical accuracy,
production readiness, or universal model ranking.
