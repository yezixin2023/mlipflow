# Run one training task

Copy this directory to a work project. `project.yaml` is a complete single-MACE
training project using the existing `mace.json`; `dft-to-all-training.yaml` remains
the multi-stage example for labels, dataset assembly, and four frameworks.

Before running, set these inputs in your copy:

- `cluster/dataset-reference.json.example`: your dataset ID, path relative to the
  configured site data root, and file/directory kind. Prefer a collected reference
  from `dataset-assemble` to preserve the existing split. A directory contains fixed
  train/validation/test views; an extxyz file follows the split settings in `mace.json`.
- `mace.json`: element reference energies, training controls and scientific settings
  for your data. The checked-in values are an example, not chemistry-independent defaults.
- `project.yaml`: matching framework/config, seed, device, precision and resources.
  Change `backend_profile` to your configured profile, or omit it to use your default.

The selected site needs a compatible MACE environment and the existing
`mlip-mace/run.sh` plus Slurm templates. See [CLUSTER.md](CLUSTER.md) only when setting
up the site or using publication/fine-tuning. No data or model weights are staged
from this example directory.

```bash
mlipflow --project /PATH/TO/training-project init
mlipflow --project /PATH/TO/training-project inspect train-mace
mlipflow --project /PATH/TO/training-project --format json run train-mace --dry-run
mlipflow --project /PATH/TO/training-project --format json run train-mace --approve
mlipflow --project /PATH/TO/training-project --format json advance
mlipflow --project /PATH/TO/training-project json train-mace
```

Review the actual inputs and scale before the approved command. Submission returns
`PENDING`; call `advance` again when the scheduler has progressed. Final `OK` returns
roles `model`, `training-manifest`, and `training-cluster-report`, with absolute paths.
Reported metrics retain their framework-defined names and units; they are not a
cross-model ranking. For fine-tuning, change `operation` to `finetune` and add an
explicit `foundation_model_reference` as illustrated in `CLUSTER.md`.

On failure use `reason`, `check.diagnostics`, and the returned log artifacts. After
diagnosis, `retry train-mace` creates a fresh attempt; the next `run` uses that attempt.

## Earlier DeepMD configurations

All scheduled training now uses the bundled `mlip-<framework>` runner. A legacy
dataset reference `{"schema_version":1,"dataset_id":"my-data"}` is normalized at
input parsing to `relative_path: my-data` and `kind: directory` for DeepMD. Write the
fields explicitly when the dataset is stored elsewhere:

```json
{"schema_version":1,"dataset_id":"my-data","relative_path":"datasets/my-data","kind":"directory"}
```

Use the existing canonical site's `mlip-deepmd/run.sh` family. Missing or malformed
references produce input errors. They never select a different training runner.
Historical attempts can still be read or replayed as existing results.

To retain the historical bounded TensorFlow fresh-training learning-curve audit,
add `validation_profile: deepmd-curve` to `parameters`. It preserves the explicit
seed, system-count, at-most-20,000-step, complete curve, positive learning rate,
normal completion marker and checkpoint checks. The extra collected artifact is
`deepmd-curve-result.json` (role `training-curve-manifest`). Ordinary training does
not implicitly enable this short-trajectory contract.
