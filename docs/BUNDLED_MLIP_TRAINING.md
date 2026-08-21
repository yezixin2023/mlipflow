# Bundled MLIP training

`plugins/mlip-training/training_wrapper.py` is a single entry point for DeepMD, M3GNet/MatGL, CHGNet, and MACE. Numerical work is delegated to each upstream Python API. Use `--dry-run` before a real GPU job.

Install the repository and one framework at a time when accelerator dependencies conflict:

```bash
python -m pip install -e '.[dev]' deepmd-kit
python -m pip install -e '.[dev]' matgl
python -m pip install -e '.[dev]' chgnet
python -m pip install -e '.[dev]' mace-torch
```

DeepMD PyTorch fine-tuning:

```bash
python plugins/mlip-training/training_wrapper.py --framework deepmd --operation finetune --config examples/training_all_models/deepmd-dpa2-pt.json --data /ABS/DEEPMD_DATA --output "$PWD/run-dp/model.pth" --result-manifest "$PWD/run-dp/result.json" --seed 23 --device cuda --precision float64 --foundation-model /ABS/PRETRAINED.pth
```

M3GNet/MatGL fine-tuning uses a MatGL model directory as the foundation model:

```bash
python plugins/mlip-training/training_wrapper.py --framework m3gnet --operation finetune --config examples/training_all_models/m3gnet.json --data /ABS/train.jsonl --output "$PWD/run-m3gnet/model.tar.gz" --result-manifest "$PWD/run-m3gnet/result.json" --seed 23 --device cuda --precision float32 --foundation-model /ABS/MATGL_MODEL_DIR
```

CHGNet fine-tuning expects JSON/JSONL records with `structure`, total `energy`, and `forces`; `efs` also requires VASP-style stress:

```bash
python plugins/mlip-training/training_wrapper.py --framework chgnet --operation finetune --config examples/training_all_models/chgnet.json --data /ABS/train.jsonl --output "$PWD/run-chgnet/model.pth.tar" --result-manifest "$PWD/run-chgnet/result.json" --seed 23 --device cuda --precision float32 --foundation-model /ABS/chgnet.pth.tar
```

MACE fine-tuning uses the upstream parser/run API; `mace-lora.json` enables LoRA:

```bash
python plugins/mlip-training/training_wrapper.py --framework mace --operation finetune --config examples/training_all_models/mace-lora.json --data /ABS/train.extxyz --output "$PWD/run-mace/model.model" --result-manifest "$PWD/run-mace/result.json" --seed 23 --device cuda --precision float32 --foundation-model /ABS/mace-foundation.model
```

Lightweight validation:

```bash
python -m pytest -q tests/test_mlip_runners.py
```

Real training is intentionally not run in repository CI; use your GPU/cluster for production jobs.
