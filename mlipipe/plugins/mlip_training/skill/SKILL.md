---
name: mlip-training
description: Supervise MLIPipe training and fine-tuning for DeepMD, M3GNet/MatGL, CHGNet, and MACE. Use when choosing a framework, preparing local or SSH-SLURM training nodes, binding labeled datasets and foundation models, selecting train versus finetune, reviewing abstract compute resources, or verifying model/result artifacts.
---

# MLIP training and fine-tuning

Use `mlipipe/plugins/mlip_training` through MLIPipe. The Skill selects and reviews training
intent; the framework runner and Adapter own deterministic configuration validation,
execution, checking, and collection.

## Route the request

- Train a new DeepMD, M3GNet/MatGL, CHGNet, or MACE model: `train`.
- Start from an explicit compatible foundation model: `finetune`.

Do not silently change frameworks, convert training to fine-tuning, select a foundation
model, enable a training technique, or choose a model by brand. Use the user's exact
choice or comparable routing/benchmark evidence; otherwise expose the unresolved
alternatives.

## Inputs and example

Reuse the existing framework config and dataset split. For a scheduled task set
`inputs.training_config` and `inputs.dataset_reference`; `finetune` also needs
`foundation_model_reference`. Declare framework, operation, seed, device, precision
and resources. Use `examples/training_all_models/project.yaml` and its `USAGE.md`.
The existing `dft-to-all-training.yaml` reuses assembled datasets in a multi-stage task.
For local execution, supply the wrapper executable/script, config, data, output and
result-manifest paths; the CLI dry-run displays the complete wrapper argv.
Earlier DeepMD references and the optional bounded curve audit are explained in the
example's `USAGE.md`.

## Scientific judgment

Require an explicit dataset, framework, operation, scientific configuration, seed,
device/precision choice, and resources appropriate to the intended scale. Fine-tuning
also requires an explicit compatible foundation-model artifact. Do not invent
hyperparameters, labels, dataset splits, model compatibility, or convergence criteria.

Distinguish a smoke or short validation run from production training. Model selection
claims require comparable benchmark evidence; training loss or successful artifact
creation alone does not establish superiority.

## Run and read the result

Preview the actual task and effective `approval_required` value. Reuse explicit
user authorization for this task and scale; use `--approve` when the plan requires it.

```bash
mlipipe --project PROJECT init
mlipipe --project PROJECT --format json run NODE --dry-run
mlipipe --project PROJECT --format json run NODE --approve
```

For a plan without an approval requirement, `run NODE` suffices. For scheduled
execution, use `mlipipe --project PROJECT --format json advance` when the job
progresses; `mlipipe --project PROJECT json NODE` reads the saved result.

Read `state`, `metrics`, `artifacts[].role` and the full `artifacts[].path` from JSON.
Require final plugin `OK` before using outputs. On failure start with `reason`,
`check.diagnostics`, `manifest_path` and `logs`; `mlipipe --project PROJECT logs NODE`
shows saved stdout/stderr. Example paths refer to the
[repository examples](https://github.com/yezixin2023/mlipipe/tree/public/examples).

Report the framework, train versus fine-tune mode, dataset/split, foundation model when
used, seed, scale, collected model reference, and checker result. `OK` establishes the
declared training contract and artifacts, not generalization, physical accuracy,
production readiness, or universal model ranking.
