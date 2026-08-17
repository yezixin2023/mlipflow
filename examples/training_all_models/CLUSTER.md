# Scheduled MLIP training and fine-tuning

The bundled scheduler contract is the common workflow layer for DeepMD, M3GNet/MatGL, CHGNet, and MACE. The project declares scientific intent; the selected site profile and remote template library own SSH, Slurm, modules, Python environments, dataset roots, and foundation-model roots.

## Site template families

Install one `run.sh` below the selected cluster profile's `remote_template_root` for every framework you enable:

- `mlip-deepmd/run.sh`
- `mlip-m3gnet/run.sh`
- `mlip-chgnet/run.sh`
- `mlip-mace/run.sh`

Start from `cluster/run.sh.example`. Each family may activate a different conda/module environment. The environment needs PyYAML plus the selected framework package. The generic runner itself is staged by MLIPFlow.

## Cluster artifact references

Large datasets and foundation models are not copied through the control plane. The project instead points to small JSON reference manifests. A dataset reference contains `dataset_id`, a safe `relative_path` below the site-owned data root, `kind` (`file` or `directory`), and a content fingerprint. Fine-tuning adds an analogous foundation-model reference below the site-owned model root.

Use `plugins/mlip-training/training_cluster.py fingerprint /ABS/CLUSTER/PATH` on the cluster to compute the expected fingerprint. Files use ordinary SHA-256. Directories use deterministic `tree-sha256-v1`: sorted relative file names, sizes, and per-file SHA-256 values are hashed together. Symlinked files inside a referenced tree are refused.

Framework data shapes are intentionally explicit:

- DeepMD dataset: `kind: directory`.
- M3GNet/MatGL dataset: `kind: file` (JSON/JSONL expected by the bundled runner).
- CHGNet dataset: `kind: file` (JSON/JSONL records expected by the bundled runner).
- MACE dataset: `kind: file` (for example extxyz/HDF5 accepted by MACE).
- M3GNet foundation model: `kind: directory`.
- DeepMD, CHGNet, and MACE foundation model: `kind: file` for the currently bundled fine-tuning paths.

## Project node shape

A scheduled MACE fine-tune node, for example, is shaped like this:

```yaml
- id: finetune-mace
  uses: mlip-training@0
  mode: execute
  backend: ssh-slurm
  backend_profile: cluster-a
  inputs:
    training_config: inputs/mace.json
    dataset_reference: inputs/dataset-reference.json
    foundation_model_reference: inputs/foundation-model-reference.json
  parameters:
    framework: mace
    operation: finetune
    seed: 23
    device: gpu
    precision: float32
    dataset_fingerprint: sha256:DATASET_SHA256
    config_fingerprint: sha256:CONFIG_SHA256
    foundation_model_fingerprint: sha256:FOUNDATION_SHA256
    result_manifest: mace-finetune-result.json
  resources:
    cpus: 16
    gpus: 1
    memory: 64G
    walltime: "12:00:00"
```

The same outer contract works for `deepmd`, `m3gnet`, `chgnet`, and `mace`, and for `train` or `finetune`. Framework-specific config semantics remain in `mlip_deepmd.py`, `mlip_m3gnet.py`, `mlip_chgnet.py`, and `mlip_mace.py`.

## Scheduler lifecycle

The generic path emits `scheduled_execution schema_version=3` with `execution_model: single-python`. MLIPFlow core stages the small approved files and bundled runner, renders the site-owned `slurm/single-python/{cpu,gpu}.sbatch` plus `mlip-<framework>/run.sh`, submits, polls, then inventories and verifies the bounded remote outputs as a continuation of the approved run.

Training is one Python process. `resources.cpus` is its thread budget, so the Slurm template must use `--ntasks=1` and `--cpus-per-task={{CPUS}}`. Mapping `CPUS` to `--ntasks` is rejected before staging.

Required fetched outputs are `cluster-run-report.json`, `training-result.json`, and `model-artifact`. The checker verifies framework/operation/seed/device/precision, staged config SHA-256, cluster-resolved dataset and foundation fingerprints, finite reported metrics, and the exact model size/SHA-256.

Legacy DeepMD fresh-training references without `relative_path`/`kind` keep using the earlier strict DeepMD scheduler contract for backward compatibility. Add the explicit generic reference fields to use the unified bundled scheduler contract.
