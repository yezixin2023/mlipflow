"""MACE runner using the official Python parser and run() API."""

import shutil
from pathlib import Path
from mlip_common import TrainingError, mapping, section, work_dir

SAFE = {
    "max_num_epochs": "--max_num_epochs",
    "batch_size": "--batch_size",
    "valid_batch_size": "--valid_batch_size",
    "r_max": "--r_max",
    "num_interactions": "--num_interactions",
    "correlation": "--correlation",
    "hidden_irreps": "--hidden_irreps",
    "lr": "--lr",
    "weight_decay": "--weight_decay",
    "energy_weight": "--energy_weight",
    "forces_weight": "--forces_weight",
    "stress_weight": "--stress_weight",
    "ema": "--ema",
    "amsgrad": "--amsgrad",
}


def _argv(args, config, data_path, work):
    cfg = section(config, "mace")
    name = str(cfg.get("name", Path(args.output).stem or "mace-model"))
    opts = mapping(cfg.get("options"), "mace.options")
    unknown = sorted(set(opts) - set(SAFE))
    if unknown:
        raise TrainingError("unsupported MACE options: " + ", ".join(unknown))
    argv = [
        "--name",
        name,
        "--train_file",
        str(data_path),
        "--seed",
        str(args.seed),
        "--device",
        "cuda" if args.device.lower() in {"gpu", "cuda"} else args.device.lower(),
        "--default_dtype",
        args.precision,
        "--work_dir",
        str(work),
        "--model_dir",
        str(work / "models"),
        "--log_dir",
        str(work / "logs"),
        "--checkpoints_dir",
        str(work / "checkpoints"),
        "--results_dir",
        str(work / "results"),
    ]
    for key in sorted(opts):
        argv += [
            SAFE[key],
            str(opts[key])
            if not isinstance(opts[key], bool)
            else ("True" if opts[key] else "False"),
        ]
    if args.operation == "finetune":
        argv += ["--foundation_model", str(Path(args.foundation_model).absolute())]
        lora = mapping(cfg.get("lora"), "mace.lora")
        if lora.get("enabled"):
            argv += [
                "--lora",
                "True",
                "--lora_rank",
                str(int(lora.get("rank", 4))),
                "--lora_alpha",
                str(float(lora.get("alpha", 1.0))),
            ]
    if cfg.get("save_cpu", True):
        argv += ["--save_cpu"]
    return name, argv


def plan(args, config, config_path, data_path):
    name, argv = _argv(args, config, data_path, Path("WORKDIR"))
    return {
        "framework": "mace",
        "operation": args.operation,
        "python_entrypoint": "mace.cli.run_train.run",
        "argv": argv,
        "name": name,
        "output": args.output,
    }


def run(args, config, config_path, data_path):
    if not data_path.is_file():
        raise TrainingError("MACE --data must be an extxyz/HDF5 file")
    from mace import tools
    from mace.cli.run_train import run as mace_run

    work = work_dir(Path(args.result_manifest).absolute(), "mace")
    name, argv = _argv(args, config, data_path, work)
    parsed = tools.build_default_arg_parser().parse_args(argv)
    mace_run(parsed)
    models = list((work / "models").glob(f"{name}*.model"))
    if not models:
        raise TrainingError("MACE completed without a model artifact")
    staged = [p for p in models if p.name.endswith("_stagetwo.model")]
    native = staged[0] if staged else max(models, key=lambda p: p.stat().st_mtime_ns)
    output = Path(args.output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(native, output)
    return (
        "application/x-pytorch",
        {"model_size_bytes": float(output.stat().st_size)},
        {"native_model": str(native), "work_dir": str(work)},
    )
