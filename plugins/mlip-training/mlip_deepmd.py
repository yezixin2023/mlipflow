"""DeepMD runner using DeePMD-kit's Python entry points, never a shell."""

import importlib
import json
import os
import shutil
from pathlib import Path
from mlip_common import TrainingError, mapping, work_dir

BACKENDS = {
    "tf": "deepmd.tf.entrypoints.main",
    "tf2": "deepmd.tf2.entrypoints.main",
    "pt": "deepmd.pt.entrypoints.main",
    "pytorch": "deepmd.pt.entrypoints.main",
    "pd": "deepmd.pd.entrypoints.main",
    "paddle": "deepmd.pd.entrypoints.main",
    "jax": "deepmd.jax.entrypoints.main",
    "pt-expt": "deepmd.pt_expt.entrypoints.main",
}
FINETUNE = {"tf", "tf2", "pt", "pytorch", "pd", "paddle"}


def _parts(args, config):
    meta = mapping(config.get("_mlipflow"), "_mlipflow")
    backend = str(meta.get("backend", "tf")).lower()
    if backend not in BACKENDS:
        raise TrainingError(f"unsupported DeepMD backend: {backend}")
    if args.operation == "finetune" and backend not in FINETUNE:
        raise TrainingError("DeepMD fine-tuning is enabled for TF/TF2/PyTorch/Paddle")
    raw = {k: v for k, v in config.items() if k not in {"_mlipflow", "schema_version", "framework"}}
    seed = mapping(raw.get("training"), "training").get("seed")
    if seed is not None and seed != args.seed:
        raise TrainingError("DeepMD training.seed must match --seed")
    return meta, backend, raw


def plan(args, config):
    meta, backend, _ = _parts(args, config)
    cmd = ["train", "WORKDIR/input.json"]
    if args.operation == "finetune":
        cmd += ["--finetune", str(Path(args.foundation_model).absolute())]
        if meta.get("model_branch") is not None:
            cmd += ["--model-branch", str(meta["model_branch"])]
        if meta.get("use_pretrain_script"):
            cmd += ["--use-pretrain-script"]
    return {
        "framework": "deepmd",
        "operation": args.operation,
        "backend": backend,
        "python_entrypoint": BACKENDS[backend],
        "train_args": cmd,
        "output": args.output,
    }


def run(args, config, config_path, data_path):
    meta, backend, raw = _parts(args, config)
    if not data_path.is_dir():
        raise TrainingError("DeepMD --data must be a dataset directory")
    work = work_dir(Path(args.result_manifest).absolute(), "deepmd")
    link = Path(str(meta.get("link_data_as", "data")))
    if link.is_absolute() or ".." in link.parts:
        raise TrainingError("link_data_as must be relative")
    target = work / link
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(data_path.absolute(), target_is_directory=True)
    input_file = work / "input.json"
    input_file.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    module = importlib.import_module(BACKENDS[backend])
    train_args = ["train", input_file.name]
    if args.operation == "finetune":
        train_args += ["--finetune", str(Path(args.foundation_model).absolute())]
        if meta.get("model_branch") is not None:
            train_args += ["--model-branch", str(meta["model_branch"])]
        if meta.get("use_pretrain_script"):
            train_args += ["--use-pretrain-script"]
    previous = Path.cwd()
    try:
        os.chdir(work)
        module.main(train_args)
        suffix = {
            "tf": ".pb",
            "tf2": ".savedmodeltf",
            "pt": ".pth",
            "pytorch": ".pth",
            "pd": "",
            "paddle": "",
            "jax": ".hlo",
            "pt-expt": "",
        }[backend]
        native = work / ("model" + suffix)
        module.main(["freeze", "-o", str(native)])
    finally:
        os.chdir(previous)
    output = Path(args.output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if native.is_file():
        shutil.copy2(native, output)
    else:
        shutil.make_archive(
            str(output.with_suffix("")), "gztar", root_dir=native.parent, base_dir=native.name
        )
        created = output.with_suffix("").with_suffix(".tar.gz")
        shutil.move(created, output)
    return (
        "application/octet-stream",
        {},
        {"backend": backend, "work_dir": str(work)},
    )
