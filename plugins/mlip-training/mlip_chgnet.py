"""CHGNet scratch training and checkpoint fine-tuning."""

import hashlib
import importlib.metadata
import json
import math
import platform
import random
import shutil
import socket
import subprocess
from collections.abc import Mapping
from numbers import Real
from pathlib import Path

from mlip_common import TrainingError, mapping, records, section, work_dir

TARGETS = {"e", "ef", "efs", "efm", "efsm"}
DATASET_LAYOUTS = {"records", "columnar"}
FREEZE_MODULES = {
    "atom_embedding",
    "bond_embedding",
    "angle_embedding",
    "bond_basis_expansion",
    "angle_basis_expansion",
    "atom_conv_layers_except_last",
    "bond_conv_layers",
    "angle_layers",
}


def _require_finite_history(value, path):
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_history(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_history(item, f"{path}[{index}]")
        return
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TrainingError(f"CHGNet training history {path} must be numeric")
    if not math.isfinite(float(value)):
        raise TrainingError(f"CHGNet training history {path} is non-finite")


def _validate_training_completion(trainer):
    epochs = getattr(trainer, "epochs", None)
    starting_epoch = getattr(trainer, "starting_epoch", None)
    if (
        isinstance(epochs, bool)
        or not isinstance(epochs, int)
        or isinstance(starting_epoch, bool)
        or not isinstance(starting_epoch, int)
        or starting_epoch < 0
        or epochs <= starting_epoch
    ):
        raise TrainingError("CHGNet trainer must request at least one epoch")
    expected = epochs - starting_epoch
    targets = getattr(trainer, "targets", None)
    history = getattr(trainer, "training_history", None)
    if not isinstance(targets, str) or not targets or not isinstance(history, Mapping):
        raise TrainingError("CHGNet trainer did not expose its training history")
    for target in targets:
        target_history = history.get(target)
        if not isinstance(target_history, Mapping):
            raise TrainingError(f"CHGNet training history is missing target {target!r}")
        for split in ("train", "val"):
            values = target_history.get(split)
            if not isinstance(values, list):
                raise TrainingError(f"CHGNet training history is missing {target}.{split}")
            if len(values) != expected:
                raise TrainingError(
                    f"CHGNet training completed {len(values)} of {expected} requested epochs "
                    f"for {target}.{split}"
                )
    _require_finite_history(history, "training_history")


def _plain_key(value, field):
    if not isinstance(value, str) or not value or "/" in value or "\\" in value:
        raise TrainingError(f"{field} must be a plain non-empty key")
    return value


def _dataset_contract(cfg):
    data = mapping(cfg.get("dataset"), "chgnet.dataset")
    unknown = sorted(
        set(data)
        - {
            "layout",
            "structure_key",
            "energy_key",
            "forces_key",
            "stress_key",
            "magmom_key",
            "energy_is_per_atom",
            "max_records",
        }
    )
    if unknown:
        raise TrainingError("unsupported CHGNet dataset options: " + ", ".join(unknown))
    layout = data.get("layout", "records")
    if layout not in DATASET_LAYOUTS:
        raise TrainingError("chgnet.dataset.layout must be records or columnar")
    energy_is_per_atom = data.get("energy_is_per_atom", cfg.get("energy_is_per_atom", False))
    if not isinstance(energy_is_per_atom, bool):
        raise TrainingError("chgnet.dataset.energy_is_per_atom must be boolean")
    max_records = data.get("max_records")
    if max_records is not None and (
        isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 2
    ):
        raise TrainingError("chgnet.dataset.max_records must be an integer of at least two")
    return {
        "layout": layout,
        "structure_key": _plain_key(data.get("structure_key", "structure"), "structure_key"),
        "energy_key": _plain_key(
            data.get("energy_key", "energy" if layout == "records" else "energy_per_atom"),
            "energy_key",
        ),
        "forces_key": _plain_key(
            data.get("forces_key", "forces" if layout == "records" else "force"),
            "forces_key",
        ),
        "stress_key": _plain_key(data.get("stress_key", "stress"), "stress_key"),
        "magmom_key": _plain_key(data.get("magmom_key", "magmom"), "magmom_key"),
        "energy_is_per_atom": energy_is_per_atom,
        "max_records": max_records,
    }


def _normalized_records(data_path, contract, targets):
    if contract["layout"] == "records":
        source = records(data_path)
    else:
        if not data_path.is_file() or data_path.suffix.lower() != ".json":
            raise TrainingError("CHGNet columnar data must be one JSON file")
        try:
            value = json.loads(data_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TrainingError(f"cannot read CHGNet columnar dataset: {exc}") from exc
        if not isinstance(value, Mapping):
            raise TrainingError("CHGNet columnar dataset must be a JSON object")
        required = [contract[key] for key in ("structure_key", "energy_key", "forces_key")]
        if "s" in targets:
            required.append(contract["stress_key"])
        if "m" in targets:
            required.append(contract["magmom_key"])
        columns = {}
        for key in required:
            column = value.get(key)
            if not isinstance(column, list):
                raise TrainingError(f"CHGNet columnar dataset requires list column {key!r}")
            columns[key] = column
        sizes = {len(column) for column in columns.values()}
        if len(sizes) != 1:
            raise TrainingError("CHGNet columnar dataset columns have different lengths")
        source_count = sizes.pop()
        max_records = contract["max_records"]
        if max_records is not None and max_records > source_count:
            raise TrainingError("chgnet.dataset.max_records exceeds available records")
        count = source_count if max_records is None else max_records
        if count < 2:
            raise TrainingError("CHGNet requires at least two structures")
        source = (
            {key: column[index] for key, column in columns.items()} for index in range(count)
        )
    for index, item in enumerate(source):
        if contract["max_records"] is not None and index >= contract["max_records"]:
            break
        if not isinstance(item, Mapping):
            raise TrainingError(f"CHGNet record {index} must be an object")
        required = {
            "structure": contract["structure_key"],
            "energy": contract["energy_key"],
            "forces": contract["forces_key"],
        }
        if "s" in targets:
            required["stress"] = contract["stress_key"]
        if "m" in targets:
            required["magmom"] = contract["magmom_key"]
        missing = [source_key for source_key in required.values() if source_key not in item]
        if missing:
            raise TrainingError(
                f"CHGNet record {index} is missing required labels: {', '.join(missing)}"
            )
        yield {name: item[source_key] for name, source_key in required.items()}


def _split_contract(cfg, sample_count):
    split = mapping(cfg.get("split"), "chgnet.split")
    unknown = sorted(set(split) - {"train_ratio", "val_ratio", "include_test"})
    if unknown:
        raise TrainingError("unsupported CHGNet split options: " + ", ".join(unknown))
    include_test = split.get("include_test", False)
    if not isinstance(include_test, bool):
        raise TrainingError("chgnet.split.include_test must be boolean")
    val_ratio = split.get("val_ratio", cfg.get("val_ratio", 0.1))
    if isinstance(val_ratio, bool) or not isinstance(val_ratio, Real):
        raise TrainingError("chgnet.split.val_ratio must be numeric")
    val_ratio = float(val_ratio)
    train_ratio = split.get("train_ratio")
    if train_ratio is None:
        train_ratio = 1.0 - val_ratio if not include_test else None
    if train_ratio is None:
        raise TrainingError("chgnet.split.train_ratio is required when include_test is true")
    if isinstance(train_ratio, bool) or not isinstance(train_ratio, Real):
        raise TrainingError("chgnet.split.train_ratio must be numeric")
    train_ratio = float(train_ratio)
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
        raise TrainingError("CHGNet train_ratio and val_ratio must be between zero and one")
    if include_test:
        if train_ratio + val_ratio >= 1:
            raise TrainingError("CHGNet test split requires train_ratio + val_ratio < 1")
    elif not math.isclose(train_ratio + val_ratio, 1.0, rel_tol=0, abs_tol=1e-12):
        raise TrainingError("CHGNet train_ratio + val_ratio must equal one without a test split")
    if include_test:
        train_count = int(train_ratio * sample_count)
        val_count = int(val_ratio * sample_count)
        test_count = sample_count - train_count - val_count
    else:
        val_count = max(1, min(sample_count - 1, round(val_ratio * sample_count)))
        train_count = sample_count - val_count
        test_count = 0
    if min(train_count, val_count) < 1 or (include_test and test_count < 1):
        raise TrainingError("CHGNet split leaves an empty requested partition")
    return {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "include_test": include_test,
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
    }


def _indices_sha256(indices):
    payload = json.dumps(indices, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _split_indices(sample_count, cfg, seed):
    split = _split_contract(cfg, sample_count)
    ids = list(range(sample_count))
    random.Random(seed).shuffle(ids)
    train_end = split["train_count"]
    val_end = train_end + split["val_count"]
    train_ids = ids[:train_end]
    val_ids = ids[train_end:val_end]
    test_ids = ids[val_end:] if split["include_test"] else []
    evidence = {
        **split,
        "seed": seed,
        "train_indices_sha256": _indices_sha256(train_ids),
        "val_indices_sha256": _indices_sha256(val_ids),
        "test_indices_sha256": _indices_sha256(test_ids) if test_ids else None,
    }
    return train_ids, val_ids, test_ids, evidence


def _freeze_modules(model, requested, operation):
    if requested is None:
        return [], 0
    if not isinstance(requested, list) or any(not isinstance(item, str) for item in requested):
        raise TrainingError("chgnet.freeze_modules must be a list of strings")
    if operation != "finetune":
        raise TrainingError("chgnet.freeze_modules is valid only for finetune")
    if len(requested) != len(set(requested)):
        raise TrainingError("chgnet.freeze_modules contains duplicates")
    unknown = sorted(set(requested) - FREEZE_MODULES)
    if unknown:
        raise TrainingError("unsupported CHGNet freeze modules: " + ", ".join(unknown))
    groups = {
        "atom_embedding": [model.atom_embedding],
        "bond_embedding": [model.bond_embedding],
        "angle_embedding": [model.angle_embedding],
        "bond_basis_expansion": [model.bond_basis_expansion],
        "angle_basis_expansion": [model.angle_basis_expansion],
        "atom_conv_layers_except_last": list(model.atom_conv_layers[:-1]),
        "bond_conv_layers": list(model.bond_conv_layers),
        "angle_layers": list(model.angle_layers),
    }
    count = 0
    for name in requested:
        for module in groups[name]:
            for parameter in module.parameters():
                if parameter.requires_grad:
                    parameter.requires_grad = False
                    count += parameter.numel()
    return list(requested), count


def _completion_evidence(trainer):
    _validate_training_completion(trainer)
    completed = trainer.epochs - trainer.starting_epoch
    metrics = {
        "requested_epochs": float(completed),
        "completed_epochs": float(completed),
    }
    for target in trainer.targets:
        history = trainer.training_history[target]
        metrics[f"final_train_{target}_mae"] = float(history["train"][-1])
        metrics[f"final_val_{target}_mae"] = float(history["val"][-1])
        test = history.get("test")
        if isinstance(test, Real) and not isinstance(test, bool):
            metrics[f"test_{target}_mae"] = float(test)
    return metrics, {
        "requested_epochs": completed,
        "completed_epochs": completed,
        "normal_completion": True,
        "all_recorded_metrics_finite": True,
    }


def _runtime_environment(torch):
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
    gpu_model = (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if torch.cuda.is_available()
        else "unavailable"
    )
    return {
        "compute_node": socket.gethostname(),
        "python_version": platform.python_version(),
        "chgnet_version": importlib.metadata.version("chgnet"),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_driver_version": driver_version,
        "gpu_model": gpu_model,
    }


def plan(args, config, data_path):
    cfg = section(config, "chgnet")
    targets = str(cfg.get("targets", "ef"))
    if args.precision != "float32":
        raise TrainingError("CHGNet requires --precision float32")
    if targets not in TARGETS:
        raise TrainingError("unsupported CHGNet targets")
    dataset = _dataset_contract(cfg)
    freeze_modules = cfg.get("freeze_modules")
    if freeze_modules is not None:
        if not isinstance(freeze_modules, list) or any(
            not isinstance(item, str) for item in freeze_modules
        ):
            raise TrainingError("chgnet.freeze_modules must be a list of strings")
        if args.operation != "finetune":
            raise TrainingError("chgnet.freeze_modules is valid only for finetune")
        if len(freeze_modules) != len(set(freeze_modules)):
            raise TrainingError("chgnet.freeze_modules contains duplicates")
        unknown = sorted(set(freeze_modules) - FREEZE_MODULES)
        if unknown:
            raise TrainingError("unsupported CHGNet freeze modules: " + ", ".join(unknown))
    return {
        "framework": "chgnet",
        "operation": args.operation,
        "python_entrypoint": "chgnet.trainer.Trainer",
        "targets": targets,
        "dataset": str(data_path),
        "dataset_contract": dataset,
        "split": mapping(cfg.get("split"), "chgnet.split"),
        "freeze_modules": freeze_modules or [],
        "foundation_model": args.foundation_model if args.operation == "finetune" else None,
        "output": args.output,
    }


def run(args, config, config_path, data_path):
    plan(args, config, data_path)
    cfg = section(config, "chgnet")
    targets = str(cfg.get("targets", "ef"))
    import torch
    from chgnet.data.dataset import StructureData, collate_graphs
    from chgnet.model.model import CHGNet
    from chgnet.trainer import Trainer
    from pymatgen.core import Structure
    from torch.utils.data import DataLoader, Subset

    requested_device = "cuda" if args.device.lower() == "gpu" else args.device.lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise TrainingError("CHGNet CUDA execution requested but torch.cuda is unavailable")
    contract = _dataset_contract(cfg)
    structures = []
    energies = []
    forces = []
    stresses = []
    magmoms = []
    for item in _normalized_records(data_path, contract, targets):
        structure = Structure.from_dict(item["structure"])
        structures.append(structure)
        energy = float(item["energy"])
        energies.append(energy if contract["energy_is_per_atom"] else energy / len(structure))
        forces.append(item["forces"])
        if "s" in targets:
            stresses.append(item["stress"])
        if "m" in targets:
            magmoms.append(item["magmom"])
    if contract["max_records"] is not None and len(structures) != contract["max_records"]:
        raise TrainingError("CHGNet dataset contains fewer records than dataset.max_records")
    if len(structures) < 2:
        raise TrainingError("CHGNet requires at least two structures")
    dataset = StructureData(
        structures,
        energies,
        forces,
        stresses=stresses or None,
        magmoms=magmoms or None,
        shuffle=False,
    )
    train_ids, val_ids, test_ids, split_evidence = _split_indices(len(dataset), cfg, args.seed)
    opts = {
        "batch_size": int(cfg.get("batch_size", 32)),
        "collate_fn": collate_graphs,
        "num_workers": int(cfg.get("num_workers", 0)),
    }
    pin_memory = cfg.get("pin_memory", False)
    if not isinstance(pin_memory, bool):
        raise TrainingError("chgnet.pin_memory must be boolean")
    opts["pin_memory"] = pin_memory
    train = DataLoader(
        Subset(dataset, train_ids),
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **opts,
    )
    valid = DataLoader(Subset(dataset, val_ids), shuffle=False, **opts)
    test = DataLoader(Subset(dataset, test_ids), shuffle=False, **opts) if test_ids else None
    model = (
        CHGNet.from_file(str(Path(args.foundation_model).absolute()))
        if args.operation == "finetune"
        else CHGNet(**mapping(cfg.get("model"), "chgnet.model"))
    )
    frozen_modules, frozen_parameters = _freeze_modules(
        model, cfg.get("freeze_modules"), args.operation
    )
    kw = mapping(cfg.get("trainer"), "chgnet.trainer")
    trainer = Trainer(
        model=model,
        targets=targets,
        torch_seed=args.seed,
        data_seed=args.seed,
        use_device=requested_device,
        **kw,
    )
    work = work_dir(Path(args.result_manifest).absolute(), "chgnet")
    environment = _runtime_environment(torch)
    trainer.train(train, valid, test_loader=test, save_dir=str(work / "checkpoints"))
    metrics, completion = _completion_evidence(trainer)
    output = Path(args.output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(filename=str(output))
    reloaded = CHGNet.from_file(str(output))
    if not isinstance(reloaded, CHGNet):
        raise TrainingError("CHGNet native model reload did not yield a CHGNet model")
    completion["native_model_reload"] = "OK"
    completion["native_model_class"] = f"{type(reloaded).__module__}.{type(reloaded).__name__}"
    return (
        "application/x-pytorch",
        {
            **metrics,
            "model_size_bytes": float(output.stat().st_size),
            "samples": float(len(dataset)),
            "train_samples": float(len(train_ids)),
            "val_samples": float(len(val_ids)),
            "test_samples": float(len(test_ids)),
        },
        {
            "samples": len(dataset),
            "targets": targets,
            "work_dir": str(work),
            "dataset_contract": contract,
            "split": split_evidence,
            "freeze_modules": frozen_modules,
            "frozen_parameters": frozen_parameters,
            "completion": completion,
            "environment": environment,
        },
    )
