"""MACE runner using the official Python parser and run() API."""

import hashlib
import json
import math
import platform
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path, PurePosixPath

from mlip_common import TrainingError, mapping, section, version, work_dir

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
    "energy_key": "--energy_key",
    "forces_weight": "--forces_weight",
    "stress_weight": "--stress_weight",
    "valid_fraction": "--valid_fraction",
    "E0s": "--E0s",
    "scaling": "--scaling",
    "ema_decay": "--ema_decay",
    "multiheads_finetuning": "--multiheads_finetuning",
    "ema": "--ema",
    "amsgrad": "--amsgrad",
}
BOOLEAN_FLAGS = {"ema", "amsgrad"}
STRING_BOOLEAN_OPTIONS = {"multiheads_finetuning"}
EPOCH = re.compile(r"\bEpoch\s+(\d+):")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _auxiliary_test_file(cfg, data_path):
    auxiliary = mapping(cfg.get("auxiliary_data"), "mace.auxiliary_data")
    unknown = sorted(set(auxiliary) - {"test_file"})
    if unknown:
        raise TrainingError("unsupported MACE auxiliary data: " + ", ".join(unknown))
    if "test_file" not in auxiliary:
        return None
    reference = mapping(auxiliary["test_file"], "mace.auxiliary_data.test_file")
    if set(reference) != {"relative_path", "fingerprint"}:
        raise TrainingError(
            "mace.auxiliary_data.test_file requires only relative_path and fingerprint"
        )
    relative = reference.get("relative_path")
    path = PurePosixPath(relative) if isinstance(relative, str) else None
    if path is None or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise TrainingError("MACE auxiliary test_file relative_path is unsafe")
    expected = reference.get("fingerprint")
    if (
        not isinstance(expected, str)
        or len(expected) != 71
        or not expected.startswith("sha256:")
        or any(char not in "0123456789abcdef" for char in expected[7:])
    ):
        raise TrainingError("MACE auxiliary test_file fingerprint is invalid")
    root = data_path.parent.resolve()
    candidate = (root / Path(*path.parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise TrainingError("MACE auxiliary test_file escapes the dataset directory") from exc
    if not candidate.is_file():
        raise TrainingError(f"MACE auxiliary test_file does not exist: {candidate}")
    observed = _sha256(candidate)
    if observed != expected:
        raise TrainingError("MACE auxiliary test_file fingerprint mismatch")
    return {"path": candidate, "fingerprint": observed}


def _argv(args, config, data_path, work):
    cfg = section(config, "mace")
    name = str(cfg.get("name", Path(args.output).stem or "mace-model"))
    opts = mapping(cfg.get("options"), "mace.options")
    unknown = sorted(set(opts) - set(SAFE))
    if unknown:
        raise TrainingError("unsupported MACE options: " + ", ".join(unknown))
    predefined = None
    if data_path.is_dir():
        predefined = {
            "train": data_path / "train.extxyz",
            "validation": data_path / "valid.extxyz",
            "test": data_path / "test.extxyz",
        }
        missing = [name for name, path in predefined.items() if not path.is_file()]
        if missing:
            raise TrainingError("predefined MACE split lacks: " + ", ".join(missing))
    train_path = predefined["train"] if predefined else data_path
    argv = [
        "--name",
        name,
        "--train_file",
        str(train_path),
        "--seed",
        str(args.seed),
        "--device",
        "cuda" if args.device.lower() in {"gpu", "cuda"} else args.device.lower(),
        "--default_dtype",
        args.precision,
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
        value = opts[key]
        if key in BOOLEAN_FLAGS:
            if not isinstance(value, bool):
                raise TrainingError(f"MACE option {key} must be boolean")
            if value:
                argv.append(SAFE[key])
            elif key == "amsgrad":
                raise TrainingError("the bundled MACE flag contract cannot disable amsgrad")
        elif key in STRING_BOOLEAN_OPTIONS:
            if not isinstance(value, bool):
                raise TrainingError(f"MACE option {key} must be boolean")
            argv += [SAFE[key], str(value)]
        else:
            argv += [SAFE[key], str(value)]
    auxiliary_test = None if predefined else _auxiliary_test_file(cfg, data_path)
    if predefined:
        argv += ["--valid_file", str(predefined["validation"]), "--test_file", str(predefined["test"])]
        if "valid_fraction" in opts:
            raise TrainingError("mace.options.valid_fraction conflicts with a predefined split")
    elif auxiliary_test is not None:
        argv += ["--test_file", str(auxiliary_test["path"])]
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
    return name, argv, auxiliary_test


def _execute_mace(tools, run_train, argv):
    parsed = tools.build_default_arg_parser().parse_args(argv)
    runner = getattr(run_train, "run", None)
    if callable(runner):
        runner(parsed)
        return
    main = getattr(run_train, "main", None)
    if not callable(main):
        raise TrainingError("installed MACE exposes neither run(args) nor main()")
    original = sys.argv
    sys.argv = ["mace_run_train", *argv]
    try:
        main()
    finally:
        sys.argv = original


def _prepare_output_directories(work):
    for name in ("models", "logs", "checkpoints", "results"):
        (work / name).mkdir(parents=True, exist_ok=True)


def _select_model(directory, name):
    models = list(directory.glob(f"{name}*.model"))
    staged = [path for path in models if path.name.endswith("_stagetwo.model")]
    if staged:
        return max(staged, key=lambda path: path.stat().st_mtime_ns)
    exact = directory / f"{name}.model"
    if exact.is_file():
        return exact
    native = [path for path in models if "_compiled.model" not in path.name]
    if native:
        return max(native, key=lambda path: path.stat().st_mtime_ns)
    raise TrainingError("MACE completed without a native model artifact")


def _walk_finite(value, label):
    if isinstance(value, bool):
        raise TrainingError(f"MACE result {label} contains a boolean numeric value")
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise TrainingError(f"MACE result {label} contains a non-finite value")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _walk_finite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _walk_finite(item, f"{label}[{index}]")


def _completion_evidence(work, requested_epochs):
    if (
        isinstance(requested_epochs, bool)
        or not isinstance(requested_epochs, int)
        or requested_epochs < 1
    ):
        raise TrainingError("MACE max_num_epochs must be a positive integer")
    log_files = sorted((work / "logs").glob("*.log"))
    if not log_files:
        raise TrainingError("MACE completed without a training log")
    log_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in log_files)
    if "Training complete" not in log_text:
        raise TrainingError("MACE log lacks normal Training complete evidence")
    observed_epochs = sorted({int(value) for value in EPOCH.findall(log_text)})
    expected_epochs = list(range(requested_epochs))
    if observed_epochs != expected_epochs:
        raise TrainingError(f"MACE completed epochs {observed_epochs}, expected {expected_epochs}")

    result_files = sorted((work / "results").glob("*.txt"))
    if not result_files:
        raise TrainingError("MACE completed without a results history")
    records = []
    for path in result_files:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="strict").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingError(
                    f"invalid MACE result record {path.name}:{line_number}"
                ) from exc
            _walk_finite(record, f"{path.name}:{line_number}")
            if isinstance(record, dict):
                records.append(record)
    final_eval = next(
        (
            record
            for record in reversed(records)
            if record.get("mode") == "eval" and record.get("epoch") == requested_epochs - 1
        ),
        None,
    )
    if final_eval is None:
        raise TrainingError("MACE results lack final-epoch evaluation metrics")
    metrics = {
        "requested_epochs": float(requested_epochs),
        "completed_epochs": float(len(observed_epochs)),
    }
    for key in ("loss", "mae_e_per_atom", "rmse_e_per_atom", "mae_f", "rmse_f"):
        value = final_eval.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        metrics[f"final_{key}"] = float(value)
    return metrics, {
        "requested_epochs": requested_epochs,
        "completed_epochs": len(observed_epochs),
        "normal_completion": True,
        "all_recorded_metrics_finite": True,
        "log_files": [path.name for path in log_files],
        "result_files": [path.name for path in result_files],
    }


def _reload_native_model(path, torch):
    try:
        model = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        model = torch.load(path, map_location="cpu")
    if not isinstance(model, torch.nn.Module):
        raise TrainingError("MACE native model reload did not yield a torch module")
    return f"{type(model).__module__}.{type(model).__name__}"


def _runtime_environment(torch, mace_source_version):
    driver_version = "unknown"
    executable = shutil.which("nvidia-smi")
    if executable:
        probe = subprocess.run(
            [executable, "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        values = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
        if probe.returncode == 0 and values:
            driver_version = values[0]
    return {
        "compute_node": socket.gethostname(),
        "python_version": platform.python_version(),
        "mace_source_version": str(mace_source_version),
        "mace_dist_version": version("mace-torch", "mace"),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_driver_version": driver_version,
        "gpu_model": torch.cuda.get_device_name(torch.cuda.current_device()),
    }


def plan(args, config, config_path, data_path):
    name, argv, auxiliary_test = _argv(args, config, data_path, Path("WORKDIR"))
    return {
        "framework": "mace",
        "operation": args.operation,
        "python_entrypoint": "mace.cli.run_train.run",
        "argv": argv,
        "name": name,
        "output": args.output,
        "auxiliary_test_file": (
            {
                "path": str(auxiliary_test["path"]),
                "fingerprint": auxiliary_test["fingerprint"],
            }
            if auxiliary_test
            else None
        ),
    }


def run(args, config, config_path, data_path):
    if not data_path.is_file() and not data_path.is_dir():
        raise TrainingError("MACE --data must be an extxyz/HDF5 file or predefined split directory")
    import torch
    from mace import __version__ as mace_source_version
    from mace import tools
    from mace.cli import run_train

    if args.device.lower() in {"gpu", "cuda"} and not torch.cuda.is_available():
        raise TrainingError("MACE CUDA execution requested but torch.cuda is unavailable")
    work = work_dir(Path(args.result_manifest).absolute(), "mace")
    cfg = section(config, "mace")
    opts = mapping(cfg.get("options"), "mace.options")
    requested_epochs = opts.get("max_num_epochs")
    name, argv, auxiliary_test = _argv(args, config, data_path, work)
    environment = _runtime_environment(torch, mace_source_version)
    _prepare_output_directories(work)
    _execute_mace(tools, run_train, argv)
    metrics, completion = _completion_evidence(work, requested_epochs)
    native = _select_model(work / "models", name)
    output = Path(args.output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(native, output)
    model_class = _reload_native_model(output, torch)
    completion["native_model_reload"] = "OK"
    completion["native_model_class"] = model_class
    if auxiliary_test is not None:
        completion["auxiliary_test_file"] = {
            "fingerprint": auxiliary_test["fingerprint"],
            "name": auxiliary_test["path"].name,
        }
    return (
        "application/x-pytorch",
        {**metrics, "model_size_bytes": float(output.stat().st_size)},
        {
            "native_model": str(native),
            "work_dir": str(work),
            "completion": completion,
            "environment": environment,
        },
    )
