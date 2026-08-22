# Scheduled MLIP training and fine-tuning

The bundled scheduler contract is the common workflow layer for DeepMD, M3GNet/MatGL, CHGNet, and MACE. The project declares scientific intent; the selected site profile and remote template library own SSH, Slurm, modules, Python environments, dataset roots, and foundation-model roots.

For from-zero creation, framework-specific import/GPU checks, and `PYTHON_BIN`
binding, see [`docs/CLUSTER_ENVIRONMENTS.md`](../../docs/CLUSTER_ENVIRONMENTS.md).

## Site template families

Install one `run.sh` below the selected cluster profile's `remote_template_root` for every framework you enable:

- `dft-dataset/run.sh` for canonical-to-framework data conversion
- `mlip-deepmd/run.sh`
- `mlip-m3gnet/run.sh`
- `mlip-chgnet/run.sh`
- `mlip-mace/run.sh`

Start training templates from `cluster/run.sh.example`, and the dataset template from
`cluster/dft-dataset.run.sh.example`. The conversion environment needs
dpdata/NumPy for DeepMD and ASE for MACE; M3GNet and CHGNet JSON serialization adds
no framework import. The repository's `dft` extra declares these conversion packages.
All five templates must resolve the same canonical cluster data
root. Each training family may activate a different framework environment. The generic
runners themselves are staged by MLIPFlow.

## DFT dataset assembly

After a scheduled DFT label attempt is final `OK`, point `dataset-assemble` at its
collected canonical artifact and declare the shared split:

```yaml
- id: assemble-datasets
  uses: dft-labeling
  needs: [label-dft]
  backend: ssh-slurm
  backend_profile: cluster-a
  inputs:
    canonical_dataset: .mlipflow/runs/label-dft/attempt-1/canonical-labeled-dataset.json
  parameters:
    operation: dataset-assemble
    frameworks: [deepmd, m3gnet, chgnet, mace]
    split_strategy: deterministic
    split_seed: 23
    split_fractions: {train: 0.8, validation: 0.1, test: 0.1}
  resources:
    cpus: 4
    gpus: 0
    memory: 8G
    walltime: "00:30:00"
```

The remote publish target is `<dataset_id>`. It contains `canonical.json`, `split.json`,
four framework directories, and `assembly-result.json`. A pre-existing target is not
rescanned or reused by the converter; use the already collected references instead.
`collect` returns the split, assembly result, and four small reference manifests while
the large datasets stay under the site data root. Each training node consumes its
matching collected reference path, so users do not run or author a converter. See
[`dft-to-all-training.yaml`](dft-to-all-training.yaml) for the full minimal DAG shape.

## Cluster artifact references

Large datasets and foundation models are not copied through the control plane. The project instead points to small JSON reference manifests. A dataset reference contains `dataset_id`, a safe `relative_path` below the site-owned data root, and `kind` (`file` or `directory`). Fine-tuning adds an analogous foundation-model reference below the site-owned model root.

Framework data shapes are intentionally explicit:

- DFT-assembled DeepMD, M3GNet/MatGL, CHGNet, and MACE datasets: `kind: directory`;
  each directory contains the fixed train/validation/test partitions.
- Legacy user-supplied M3GNet/CHGNet/MACE single-file datasets remain accepted.
- M3GNet foundation model: `kind: directory`.
- DeepMD, CHGNet, and MACE foundation model: `kind: file` for the currently bundled fine-tuning paths.

## Project node shape

A scheduled MACE fine-tune node, for example, is shaped like this:

```yaml
- id: finetune-mace
  uses: mlip-training
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
    result_manifest: mace-finetune-result.json
  resources:
    cpus: 16
    gpus: 1
    memory: 64G
    walltime: "12:00:00"
```

The same outer contract works for `deepmd`, `m3gnet`, `chgnet`, and `mace`, and for `train` or `finetune`. Framework-specific config semantics remain in `mlip_deepmd.py`, `mlip_m3gnet.py`, `mlip_chgnet.py`, and `mlip_mace.py`.

## Scheduler lifecycle

The scheduled path emits `scheduled_execution schema_version=3` with `execution_model: single-python`. MLIPFlow core stages the small approved files and bundled runner, renders the site-owned `slurm/single-python/{cpu,gpu}.sbatch` plus `mlip-<framework>/run.sh`, submits, polls, then fetches and verifies the declared remote outputs as a continuation of the approved run.

Training is one Python process. `resources.cpus` is its thread budget, so the Slurm template must use `--ntasks=1` and `--cpus-per-task={{CPUS}}`. Mapping `CPUS` to `--ntasks` is rejected before staging.

Required fetched outputs are `cluster-run-report.json`, `training-result.json`, and `model-artifact`. The checker verifies framework/operation/seed/device/precision, the recorded config, dataset and foundation-model paths, finite reported metrics, and the declared model output.

Legacy DeepMD fresh-training references without `relative_path`/`kind` keep using the earlier strict DeepMD scheduler contract for backward compatibility. Add the explicit generic reference fields to use the unified bundled scheduler contract.
