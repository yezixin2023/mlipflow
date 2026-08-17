---
name: mlip-training
description: Supervise MLIPFlow training and fine-tuning for DeepMD, M3GNet/MatGL, CHGNet, and MACE. Use when choosing a framework, preparing local or SSH-SLURM training nodes, binding labeled datasets and foundation models, selecting train versus finetune, reviewing abstract compute resources, or verifying model/result artifacts.
---

# MLIP training and fine-tuning

Use the `mlip-training` plugin as the deterministic execution layer. The Skill chooses and reviews workflow intent; framework numerics remain in explicit config files and the bundled `mlip_*.py` runners.

## Choose framework and operation

The supported frameworks are `deepmd`, `m3gnet`, `chgnet`, and `mace`. All four support `train` and `finetune` in the bundled scheduler contract.

Do not choose a framework from name recognition alone. Prefer an explicit user choice, an upstream benchmark/routing artifact, or constraints such as an existing foundation model. If evidence is insufficient, expose the alternatives instead of inventing a ranking.

Framework-specific execution shapes are:

- DeepMD: labeled data is a directory. Fine-tuning uses a file foundation artifact for the currently bundled path and is limited to DeePMD backends whose verified entry points expose `--finetune` (TF/TF2/PyTorch/Paddle in the bundled runner).
- M3GNet/MatGL: labeled data is a JSON/JSONL file. Fine-tuning requires an extracted MatGL model directory.
- CHGNet: labeled data is a JSON/JSONL file. Fine-tuning requires a CHGNet checkpoint file. Precision must be `float32`.
- MACE: labeled data is a framework-supported file such as extxyz/HDF5. Fine-tuning requires a MACE foundation model file. Optional LoRA is configured in the explicit MACE config, never inferred by the Skill.

Never silently convert a `train` request into `finetune`, reuse a checkpoint, change a foundation model, or enable LoRA.

## Prefer the scheduler contract for cluster work

For cluster execution use `backend: ssh-slurm` and a named `backend_profile`. The project node carries only abstract resources:

- `cpus`
- `gpus`
- `memory`
- `walltime`

Never put an SSH host, partition, account, QoS, module command, conda path, framework executable, absolute dataset path, absolute model path, template root, work root, submit script, or `remote_cwd` in the project node. Those are site-owned details selected by `~/.mlipflow/site.yaml` and the remote template library.

The generic template families are:

- `mlip-deepmd/run.sh`
- `mlip-m3gnet/run.sh`
- `mlip-chgnet/run.sh`
- `mlip-mace/run.sh`

Each site template may activate a different framework environment. It supplies the cluster data/model roots and calls the staged `training_cluster.py`; MLIPFlow core alone owns SSH, staging, `sbatch`, polling, bounded fetch, retry, and cancellation.

The bundled training contract has `execution_model: single-python`. It launches
one Python process, so `resources.cpus` means the CPU/thread budget for that one
process. The selected Slurm submit template must use `--ntasks=1` and
`--cpus-per-task={{CPUS}}`. Never map training `CPUS` to `--ntasks`; Lightning
interprets that as a distributed launch. This does not change MPI contracts such
as VASP, LAMMPS, or LASP, where `CPUS` is the task/rank count.

## Bind large data and models logically

Do not copy production datasets or foundation models through the control plane. Scheduled nodes use small project-scoped reference manifests.

A dataset reference declares:

- `schema_version: 1`
- stable `dataset_id`
- safe site-root-relative `relative_path`
- `kind: file` or `directory`
- a verifiable content identity

A fine-tune node additionally needs a foundation-model reference with `model_id`, `relative_path`, `kind`, and verifiable content identity.

The framework determines the allowed kind: DeepMD datasets are directories; M3GNet/CHGNet/MACE datasets are files; M3GNet foundation models are directories; the currently bundled DeepMD/CHGNet/MACE foundation paths are files.

The remote runner verifies the referenced content before training. A mismatch is `FAIL`, never a warning.

## Build a scheduled node

Require these scheduled inputs:

- `training_config`
- `dataset_reference`
- `foundation_model_reference` only for `finetune`

Require these parameters:

- `framework`
- `operation`
- `seed`
- `device`
- `precision`

`result_manifest` may set the fetched result filename. Configs must be JSON on the generic scheduler path. Bind actual dataset/config/foundation-model references; do not invent their identity or substitute similarly named files.

## Execution

Before submission, show the user the framework, operation, config/dataset/foundation-model identity, seed, device, precision, abstract resources, selected backend profile, template family, important staged inputs, expected outputs, and fresh attempt workspace.

Scheduler `COMPLETED` is not scientific success. After completion, ordinary `advance` bounded-fetches the run's declared outputs, then runs `check/collect`.

For the generic path, required remote outputs are:

- `cluster-run-report.json`
- `training-result.json`
- `model-artifact`

Optional training stdout/stderr logs are bounded and may also be fetched.

## Completion checks

Accept `OK` only when the pinned checker confirms:

- framework and operation match the attempt snapshot;
- seed, device, and precision match;
- staged config and cluster-resolved dataset match the requested inputs;
- the foundation model matches for fine-tuning;
- the cluster runner reports return code 0;
- all reported numeric metrics are finite;
- the fetched model content matches `training-result.json`.

Retry always creates a fresh attempt. Do not infer resume behavior from a failed attempt.

## Agent boundary

The Skill may propose framework/operation/resources from explicit evidence, compose project nodes, explain blocked requirements, and route successful model artifacts to downstream benchmark or simulation stages. It must not fabricate benchmark superiority, training convergence, framework versions, cluster paths, foundation models, labels, or hyperparameters.
