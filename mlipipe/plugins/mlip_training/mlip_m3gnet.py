"""M3GNet scratch training and foundation fine-tuning through MatGL."""

from __future__ import annotations

import csv
import importlib.metadata
import inspect
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

if __package__:
    from .mlip_common import (
        TrainingError,
        mapping,
        predefined_split_files,
        predefined_split_record,
        section,
        work_dir,
    )
else:
    from mlip_common import (
        TrainingError,
        mapping,
        predefined_split_files,
        predefined_split_record,
        section,
        work_dir,
    )

DATASET_LAYOUTS = {"records", "columnar"}
API_FLAVORS = {"auto", "high_level", "legacy"}


def _plain_key(value, field):
    if not isinstance(value, str) or not value or "/" in value or "\\" in value:
        raise TrainingError(f"{field} must be a plain non-empty key")
    return value


def _dataset_contract(cfg):
    data = mapping(cfg.get("dataset"), "m3gnet.dataset")
    unknown = sorted(
        set(data)
        - {
            "layout",
            "structure_key",
            "energy_key",
            "forces_key",
            "stress_key",
            "energy_is_per_atom",
            "max_records",
        }
    )
    if unknown:
        raise TrainingError("unsupported M3GNet dataset options: " + ", ".join(unknown))
    layout = data.get("layout", "records")
    if layout not in DATASET_LAYOUTS:
        raise TrainingError("m3gnet.dataset.layout must be records or columnar")
    max_records = data.get("max_records")
    if max_records is not None and (
        isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 3
    ):
        raise TrainingError("m3gnet.dataset.max_records must be an integer of at least three")
    energy_is_per_atom = data.get("energy_is_per_atom", False)
    if not isinstance(energy_is_per_atom, bool):
        raise TrainingError("m3gnet.dataset.energy_is_per_atom must be boolean")
    if energy_is_per_atom:
        raise TrainingError(
            "this M3GNet PES path requires total energies; energy_is_per_atom must be false"
        )
    return {
        "layout": layout,
        "structure_key": _plain_key(data.get("structure_key", "structure"), "structure_key"),
        "energy_key": _plain_key(
            data.get("energy_key", "energy" if layout == "records" else "energies"),
            "energy_key",
        ),
        "forces_key": _plain_key(
            data.get("forces_key", "forces" if layout == "records" else "forces"),
            "forces_key",
        ),
        "stress_key": _plain_key(data.get("stress_key", "stresses"), "stress_key"),
        "energy_is_per_atom": energy_is_per_atom,
        "max_records": max_records,
    }


def _normalized_records(data_path, contract, include_stress):
    if not data_path.is_file():
        raise TrainingError("M3GNet data must be one JSON or JSONL file")
    if contract["layout"] == "columnar":
        if data_path.suffix.lower() != ".json":
            raise TrainingError("M3GNet columnar data must be one JSON file")
        try:
            with data_path.open(encoding="utf-8") as stream:
                value = json.load(stream)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TrainingError(f"cannot read M3GNet columnar dataset: {exc}") from exc
        if not isinstance(value, Mapping):
            raise TrainingError("M3GNet columnar dataset must be a JSON object")
        names = [contract[key] for key in ("structure_key", "energy_key", "forces_key")]
        if include_stress:
            names.append(contract["stress_key"])
        columns = {}
        for name in names:
            column = value.get(name)
            if not isinstance(column, list):
                raise TrainingError(f"M3GNet columnar dataset requires list column {name!r}")
            columns[name] = column
        lengths = {len(column) for column in columns.values()}
        if len(lengths) != 1:
            raise TrainingError("M3GNet columnar dataset columns have different lengths")
        total = next(iter(lengths), 0)
        limit = min(total, contract["max_records"] or total)
        for index in range(limit):
            item = {
                "structure": columns[contract["structure_key"]][index],
                "energy": columns[contract["energy_key"]][index],
                "forces": columns[contract["forces_key"]][index],
            }
            if include_stress:
                item["stress"] = columns[contract["stress_key"]][index]
            yield item
        return
    try:
        if data_path.suffix.lower() == ".jsonl":
            source = (
                json.loads(line)
                for line in data_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        else:
            value = json.loads(data_path.read_text(encoding="utf-8"))
            source = (
                value if isinstance(value, list) else value.get("records", value.get("data", []))
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingError(f"cannot read M3GNet record dataset: {exc}") from exc
    if not isinstance(source, list) and not hasattr(source, "__iter__"):
        raise TrainingError("M3GNet record dataset must contain a list of records")
    required = [contract[key] for key in ("structure_key", "energy_key", "forces_key")]
    if include_stress:
        required.append(contract["stress_key"])
    for index, raw in enumerate(source):
        if contract["max_records"] is not None and index >= contract["max_records"]:
            break
        if not isinstance(raw, Mapping) or any(name not in raw for name in required):
            raise TrainingError(f"M3GNet record {index} lacks required mapped fields")
        item = {
            "structure": raw[contract["structure_key"]],
            "energy": raw[contract["energy_key"]],
            "forces": raw[contract["forces_key"]],
        }
        if isinstance(raw.get("record_id"), str):
            item["record_id"] = raw["record_id"]
        if include_stress:
            item["stress"] = raw[contract["stress_key"]]
        yield item


def _records_and_split(data_path, contract, include_stress, cfg, seed):
    files = predefined_split_files(data_path, "json")
    if files is None:
        records = list(_normalized_records(data_path, contract, include_stress))
        train, validation, test, evidence = _split_indices(len(records), cfg, seed)
        return records, train, validation, test, evidence
    if contract["max_records"] is not None:
        raise TrainingError("dataset.max_records cannot truncate a predefined split")
    partitions = {
        name: list(_normalized_records(path, contract, include_stress))
        for name, path in files.items()
    }
    records = partitions["train"] + partitions["validation"] + partitions["test"]
    train_end = len(partitions["train"])
    valid_end = train_end + len(partitions["validation"])
    return (
        records,
        list(range(train_end)),
        list(range(train_end, valid_end)),
        list(range(valid_end, len(records))),
        predefined_split_record(files),
    )


def _split_indices(sample_count, cfg, seed):
    split = mapping(cfg.get("split"), "m3gnet.split")
    unknown = sorted(set(split) - {"train_ratio", "val_ratio", "include_test"})
    if unknown:
        raise TrainingError("unsupported M3GNet split options: " + ", ".join(unknown))
    train_ratio = split.get("train_ratio", 0.8)
    val_ratio = split.get("val_ratio", 0.1)
    include_test = split.get("include_test", True)
    if (
        isinstance(train_ratio, bool)
        or not isinstance(train_ratio, Real)
        or isinstance(val_ratio, bool)
        or not isinstance(val_ratio, Real)
        or not 0 < float(train_ratio) < 1
        or not 0 < float(val_ratio) < 1
        or float(train_ratio) + float(val_ratio) >= 1
    ):
        raise TrainingError("M3GNet split ratios must be positive and sum to less than one")
    if not isinstance(include_test, bool):
        raise TrainingError("m3gnet.split.include_test must be boolean")
    order = list(range(sample_count))
    random.Random(seed).shuffle(order)
    train_count = int(sample_count * float(train_ratio))
    val_count = int(sample_count * float(val_ratio))
    if train_count < 1 or val_count < 1:
        raise TrainingError("M3GNet split must contain at least one train and validation sample")
    test_count = sample_count - train_count - val_count
    if include_test and test_count < 1:
        raise TrainingError("M3GNet split must contain at least one test sample")
    train = order[:train_count]
    val = order[train_count : train_count + val_count]
    test = order[train_count + val_count :] if include_test else []
    evidence = {
        "seed": seed,
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "include_test": include_test,
        "train_count": len(train),
        "val_count": len(val),
        "test_count": len(test),
        "train_indices": train,
        "validation_indices": val,
        "test_indices": test,
    }
    return train, val, test, evidence


def _config(cfg):
    allowed = {"api", "dataset", "data", "split", "model", "module", "loader", "trainer"}
    unknown = sorted(set(cfg) - allowed)
    if unknown:
        raise TrainingError("unsupported M3GNet config sections: " + ", ".join(unknown))
    data = mapping(cfg.get("data"), "m3gnet.data")
    if set(data) - {
        "cutoff",
        "threebody_cutoff",
        "include_line_graph",
        "include_stress",
        "stress_unit",
    }:
        raise TrainingError("unsupported M3GNet data option")
    model = mapping(cfg.get("model"), "m3gnet.model")
    module = mapping(cfg.get("module"), "m3gnet.module")
    loader = mapping(cfg.get("loader"), "m3gnet.loader")
    trainer = mapping(cfg.get("trainer"), "m3gnet.trainer")
    if set(loader) - {"batch_size", "num_workers", "drop_last"}:
        raise TrainingError("unsupported M3GNet loader option")
    if set(trainer) - {"max_epochs", "deterministic", "log_every_n_steps"}:
        raise TrainingError("unsupported M3GNet trainer option")
    if set(module) - {
        "energy_weight",
        "force_weight",
        "stress_weight",
        "lr",
        "decay_steps",
        "decay_alpha",
        "loss",
    }:
        raise TrainingError("unsupported M3GNet module option")
    epochs = trainer.get("max_epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise TrainingError("m3gnet.trainer.max_epochs must be a positive integer")
    batch_size = loader.get("batch_size", 24)
    num_workers = loader.get("num_workers", 0)
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
        or isinstance(num_workers, bool)
        or not isinstance(num_workers, int)
        or num_workers < 0
    ):
        raise TrainingError("M3GNet loader batch_size/num_workers are invalid")
    for key in ("include_line_graph", "include_stress"):
        if key in data and not isinstance(data[key], bool):
            raise TrainingError(f"m3gnet.data.{key} must be boolean")
    return data, model, module, loader, trainer


def _requested_api(cfg):
    value = cfg.get("api", "auto")
    if value not in API_FLAVORS:
        raise TrainingError("m3gnet.api must be auto, high_level, or legacy")
    return value


def _high_level_available(matgl):
    return callable(getattr(matgl, "MGLDatasetLoader", None)) and callable(
        getattr(matgl, "MGLPotentialTrainer", None)
    )


def _select_api(cfg, matgl):
    requested = _requested_api(cfg)
    available = _high_level_available(matgl)
    if requested == "high_level" and not available:
        raise TrainingError(
            "M3GNet high_level API requires MGLDatasetLoader and MGLPotentialTrainer"
        )
    if requested == "legacy":
        return "legacy"
    if requested == "high_level":
        return "high_level"
    return "high_level" if available else "legacy"


def _finite(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise TrainingError("MatGL recorded a non-finite metric")
    return float(value)


def _history(csv_path, requested_epochs):
    if not csv_path.is_file():
        raise TrainingError("MatGL Lightning metrics.csv is missing")
    train, val = {}, {}
    with csv_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            raw_epoch = row.get("epoch")
            if raw_epoch in {None, ""}:
                continue
            epoch = int(raw_epoch)
            train_values = {
                key: _finite(float(value))
                for key, value in row.items()
                if key.startswith("train_") and value not in {None, ""}
            }
            val_values = {
                key: _finite(float(value))
                for key, value in row.items()
                if key.startswith("val_") and value not in {None, ""}
            }
            if train_values:
                train[epoch] = train_values
            if val_values:
                val[epoch] = val_values
    expected = list(range(requested_epochs))
    if sorted(train) != expected or sorted(val) != expected:
        raise TrainingError("MatGL train/validation history does not match requested epochs")
    return [train[index] for index in expected], [val[index] for index in expected]


def _environment(torch):
    gpu_model = "unavailable"
    if torch.cuda.is_available():
        try:
            gpu_model = torch.cuda.get_device_name(0)
        except Exception:
            gpu_model = "unknown"
    driver = "unknown"
    if shutil.which("nvidia-smi"):
        try:
            driver = (
                subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                .stdout.splitlines()[0]
                .strip()
            )
        except Exception:
            driver = "unknown"
    try:
        graph_backend = "dgl"
        graph_library_version = importlib.metadata.version("dgl")
    except importlib.metadata.PackageNotFoundError:
        graph_backend = "pyg"
        graph_library_version = importlib.metadata.version("torch-geometric")
    return {
        "compute_node": socket.gethostname(),
        "python_version": platform.python_version(),
        "matgl_version": importlib.metadata.version("matgl"),
        "lightning_version": importlib.metadata.version("lightning"),
        "graph_backend": graph_backend,
        "graph_library_version": graph_library_version,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda or "none"),
        "cuda_driver_version": driver,
        "gpu_model": gpu_model,
    }


def plan(args, config, data_path):
    if not data_path.is_file() and predefined_split_files(data_path, "json") is None:
        raise TrainingError("M3GNet data must be JSON/JSONL or a predefined split directory")
    cfg = section(config, "m3gnet")
    contract = _dataset_contract(cfg)
    _, _, _, _, trainer = _config(cfg)
    requested_api = _requested_api(cfg)
    return {
        "framework": "m3gnet",
        "operation": args.operation,
        "api": requested_api,
        "python_entrypoint": (
            "auto:MGLPotentialTrainer|PotentialLightningModule"
            if requested_api == "auto"
            else "matgl.MGLPotentialTrainer"
            if requested_api == "high_level"
            else "matgl.utils.training.PotentialLightningModule"
        ),
        "dataset": str(data_path),
        "dataset_contract": contract,
        "requested_epochs": trainer["max_epochs"],
        "foundation_model": args.foundation_model if args.operation == "finetune" else None,
        "output": args.output,
    }


def _supported_kwargs(function, values):
    parameters = inspect.signature(function).parameters
    if any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
        return dict(values)
    return {key: value for key, value in values.items() if key in parameters}


def _run_high_level(args, cfg, data_path, matgl, torch):
    """Run the current PyG-oriented MatGL convenience API when it is installed."""
    from lightning.pytorch.loggers import CSVLogger
    from matgl.models import M3GNet
    from torch.utils.data import Subset

    contract = _dataset_contract(cfg)
    data_cfg, model_cfg, module_cfg, loader_cfg, trainer_cfg = _config(cfg)
    work = work_dir(Path(args.result_manifest).absolute(), "m3gnet")
    dtype = torch.float64 if args.precision == "float64" else torch.float32
    potential = None
    atomrefs = None
    elements = None
    if args.operation == "finetune":
        source = Path(args.foundation_model).absolute()
        if not source.is_dir():
            raise TrainingError("M3GNet foundation model must be a MatGL model directory")
        potential = matgl.load_model(str(source))
        model = getattr(potential, "model", potential).to(dtype=dtype)
        elements = tuple(model.element_types)
        atomrefs = getattr(potential, "element_refs", None)
        if atomrefs is None:
            raise TrainingError("M3GNet high-level finetune foundation lacks element references")

    loader_type = matgl.MGLDatasetLoader
    from_json = getattr(loader_type, "from_json", None)
    if not callable(from_json):
        instance = loader_type()
        from_json = getattr(instance, "from_json", None)
    if not callable(from_json):
        raise TrainingError(
            "installed MGLDatasetLoader has no local from_json factory; use a pre-built "
            "MatPES dataset integration or select api=legacy for historical JSON"
        )
    def load_json(path, cache_name):
        return from_json(
            path,
            **_supported_kwargs(
                from_json,
                {
                    "cutoff": float(data_cfg.get("cutoff", 5.0)),
                    "element_types": elements,
                    "save_cache": False,
                    "root": str(work / cache_name),
                    "stress_unit": str(data_cfg.get("stress_unit", "kbar")),
                },
            ),
        )

    files = predefined_split_files(data_path, "json")
    if files is not None:
        if contract["max_records"] is not None:
            raise TrainingError("dataset.max_records cannot truncate a predefined split")
        datasets = {name: load_json(path, f"cache-{name}") for name, path in files.items()}
        dataset = datasets["train"]
        train_indices = list(range(len(datasets["train"])))
        val_indices = list(range(len(datasets["validation"])))
        test_indices = list(range(len(datasets["test"])))
        selected_dataset = [None] * sum(map(len, datasets.values()))
        split = predefined_split_record(files)
        splits = {"train": datasets["train"], "valid": datasets["validation"], "test": datasets["test"]}
    else:
        dataset = load_json(data_path, "cache")
        selected_count = min(len(dataset), contract["max_records"] or len(dataset))
        selected_dataset = Subset(dataset, list(range(selected_count)))
        train_indices, val_indices, test_indices, split = _split_indices(len(selected_dataset), cfg, args.seed)
        splits = {
            "train": Subset(selected_dataset, train_indices),
            "valid": Subset(selected_dataset, val_indices),
            "test": Subset(selected_dataset, test_indices),
        }
    if len(selected_dataset) < 3 or any(len(splits[name]) < 1 for name in splits):
        raise TrainingError("M3GNet high-level dataset requires non-empty train/validation/test")
    if model_cfg.get("element_types"):
        configured_elements = tuple(model_cfg.pop("element_types"))
    else:
        configured_elements = tuple(getattr(dataset, "element_types", ())) or tuple(
            getattr(getattr(dataset, "converter", None), "element_types", ())
        )
    if args.operation == "train":
        if not configured_elements:
            raise TrainingError("M3GNet high-level dataset lacks element_types")
        elements = configured_elements
        model_cfg.setdefault("is_intensive", False)
        model = M3GNet(element_types=elements, **model_cfg).to(dtype=dtype)
    elif configured_elements and configured_elements != elements:
        raise TrainingError("M3GNet high-level dataset/model element_types differ")

    logger = CSVLogger(save_dir=str(work / "logs"), name="M3GNet")
    accelerator = "gpu" if args.device.lower() in {"gpu", "cuda"} else args.device.lower()
    if accelerator == "gpu" and not torch.cuda.is_available():
        raise TrainingError("M3GNet requested CUDA but torch.cuda.is_available() is false")
    trainer_values = {
        **module_cfg,
        "batch_size": loader_cfg.get("batch_size", 32),
        "num_workers": loader_cfg.get("num_workers", 0),
        "max_epochs": trainer_cfg["max_epochs"],
        "accelerator": accelerator,
        "devices": 1,
        "random_state": args.seed,
        "seed": args.seed,
        "trainer_kwargs": {
            "logger": logger,
            "enable_checkpointing": False,
            "enable_model_summary": False,
            "enable_progress_bar": False,
            "num_sanity_val_steps": 0,
            "deterministic": trainer_cfg.get("deterministic", True),
            "log_every_n_steps": trainer_cfg.get("log_every_n_steps", 1),
        },
    }
    trainer_type = matgl.MGLPotentialTrainer
    high_level = trainer_type(model, **_supported_kwargs(trainer_type, trainer_values))
    native = work / "model"
    trained = high_level.fit(dataset=splits, atomrefs=atomrefs, save_path=str(native))
    if not native.is_dir():
        saved = trained or getattr(high_level, "potential", None)
        if saved is None or not callable(getattr(saved, "save", None)):
            raise TrainingError("MGLPotentialTrainer did not create a native model directory")
        saved.save(str(native))
    lightning_trainer = getattr(high_level, "trainer", None)
    normal_completion = bool(
        lightning_trainer is not None
        and getattr(getattr(lightning_trainer, "state", None), "finished", False)
    )
    requested_epochs = trainer_cfg["max_epochs"]
    train_history, val_history = _history(Path(logger.log_dir) / "metrics.csv", requested_epochs)
    reloaded = matgl.load_model(str(native))
    native_class = f"{type(reloaded).__module__}.{type(reloaded).__name__}"
    output = Path(args.output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    archive = Path(
        shutil.make_archive(str(work / "packed-model"), "gztar", root_dir=work, base_dir="model")
    )
    shutil.move(str(archive), str(output))
    final_train = train_history[-1]
    final_val = val_history[-1]
    metrics = {
        "samples": float(len(selected_dataset)),
        "train_samples": float(len(train_indices)),
        "val_samples": float(len(val_indices)),
        "test_samples": float(len(test_indices)),
        "requested_epochs": float(requested_epochs),
        "completed_epochs": float(len(train_history)),
        **{f"final_{key}": value for key, value in final_train.items()},
        **{f"final_{key}": value for key, value in final_val.items()},
    }
    return (
        "application/gzip",
        metrics,
        {
            "api": "high_level",
            "dataset_contract": contract,
            "samples": len(selected_dataset),
            "split": split,
            "element_types": list(elements),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "foundation_element_refs": args.operation == "finetune",
            "train_history": train_history,
            "val_history": val_history,
            "test_metrics": {},
            "completion": {
                "requested_epochs": requested_epochs,
                "completed_epochs": len(train_history),
                "normal_completion": normal_completion,
                "all_recorded_metrics_finite": True,
                "native_model_reload": "OK",
                "native_model_class": native_class,
            },
            "environment": _environment(torch),
            "work_dir": str(work),
        },
    )


def run(args, config, config_path, data_path):
    plan(args, config, data_path)
    import matgl
    import torch

    cfg = section(config, "m3gnet")
    if _select_api(cfg, matgl) == "high_level":
        return _run_high_level(args, cfg, data_path, matgl, torch)

    import lightning as pl
    from functools import partial
    from lightning.pytorch.loggers import CSVLogger
    from matgl.config import DEFAULT_ELEMENTS
    from matgl.ext.pymatgen import Structure2Graph
    from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn_pes
    from matgl.models import M3GNet
    from matgl.utils.training import PotentialLightningModule
    from pymatgen.core import Structure
    from torch.utils.data import Subset

    contract = _dataset_contract(cfg)
    data_cfg, model_cfg, module_cfg, loader_cfg, trainer_cfg = _config(cfg)
    include_line_graph = data_cfg.get("include_line_graph", True)
    include_stress = data_cfg.get("include_stress", True)
    records, train_indices, val_indices, test_indices, split = _records_and_split(
        data_path, contract, include_stress, cfg, args.seed
    )
    if len(records) < 3:
        raise TrainingError("M3GNet dataset requires at least three selected records")
    structures = [Structure.from_dict(item["structure"]) for item in records]
    labels = {
        "energies": [item["energy"] for item in records],
        "forces": [item["forces"] for item in records],
    }
    if include_stress:
        labels["stresses"] = [item["stress"] for item in records]

    matgl.set_default_dtype("float", 64 if args.precision == "float64" else 32)
    dtype = torch.float64 if args.precision == "float64" else torch.float32
    work = work_dir(Path(args.result_manifest).absolute(), "m3gnet")
    if args.operation == "finetune":
        source = Path(args.foundation_model).absolute()
        if not source.is_dir():
            raise TrainingError("M3GNet foundation model must be a MatGL model directory")
        potential = matgl.load_model(str(source))
        model = getattr(potential, "model", potential).to(dtype=dtype)
        elements = tuple(model.element_types)
        element_refs = getattr(getattr(potential, "element_refs", None), "property_offset", None)
        if element_refs is None:
            raise TrainingError("M3GNet foundation lacks element reference offsets")
        element_refs = element_refs.to(dtype=dtype)
    else:
        elements = tuple(model_cfg.pop("element_types", ())) or tuple(DEFAULT_ELEMENTS)
        model_cfg.setdefault("is_intensive", False)
        model = M3GNet(element_types=elements, **model_cfg).to(dtype=dtype)
        element_refs = None

    converter = Structure2Graph(
        element_types=elements,
        cutoff=float(data_cfg.get("cutoff", getattr(model, "cutoff", 5.0))),
    )
    dataset = MGLDataset(
        threebody_cutoff=float(
            data_cfg.get("threebody_cutoff", getattr(model, "threebody_cutoff", 4.0))
        ),
        structures=structures,
        converter=converter,
        labels=labels,
        include_line_graph=include_line_graph,
        save_cache=False,
        raw_dir=str(work / "cache"),
    )
    collate = partial(
        collate_fn_pes,
        include_line_graph=include_line_graph,
        include_stress=include_stress,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loaders = MGLDataLoader(
        train_data=Subset(dataset, train_indices),
        val_data=Subset(dataset, val_indices),
        test_data=Subset(dataset, test_indices) if test_indices else None,
        collate_fn=collate,
        batch_size=loader_cfg.get("batch_size", 24),
        num_workers=loader_cfg.get("num_workers", 0),
        drop_last=loader_cfg.get("drop_last", False),
        generator=generator,
    )
    train_loader, val_loader = loaders[:2]
    test_loader = loaders[2] if len(loaders) == 3 else None
    module_cfg.setdefault("include_line_graph", include_line_graph)
    lightning_module = PotentialLightningModule(
        model=model,
        element_refs=element_refs,
        **module_cfg,
    )
    logger = CSVLogger(save_dir=str(work / "logs"), name="M3GNet")
    requested_epochs = trainer_cfg["max_epochs"]
    accelerator = "gpu" if args.device.lower() in {"gpu", "cuda"} else args.device.lower()
    if accelerator == "gpu" and not torch.cuda.is_available():
        raise TrainingError("M3GNet requested CUDA but torch.cuda.is_available() is false")
    trainer = pl.Trainer(
        max_epochs=requested_epochs,
        accelerator=accelerator,
        devices=1,
        logger=logger,
        inference_mode=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        deterministic=trainer_cfg.get("deterministic", True),
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 1),
    )
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    normal_completion = bool(getattr(trainer.state, "finished", False))
    train_history, val_history = _history(Path(logger.log_dir) / "metrics.csv", requested_epochs)
    test_metrics = {}
    if test_loader is not None:
        tested = trainer.test(model=lightning_module, dataloaders=test_loader)
        if len(tested) != 1 or not isinstance(tested[0], Mapping):
            raise TrainingError("MatGL test did not return one metric mapping")
        test_metrics = {key: _finite(value) for key, value in tested[0].items()}

    native = work / "model"
    lightning_module.model.save(str(native))
    reloaded = matgl.load_model(str(native))
    native_class = f"{type(reloaded).__module__}.{type(reloaded).__name__}"
    output = Path(args.output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    archive = Path(
        shutil.make_archive(str(work / "packed-model"), "gztar", root_dir=work, base_dir="model")
    )
    shutil.move(str(archive), str(output))

    final_train = train_history[-1]
    final_val = val_history[-1]
    metrics = {
        "samples": float(len(records)),
        "train_samples": float(len(train_indices)),
        "val_samples": float(len(val_indices)),
        "test_samples": float(len(test_indices)),
        "requested_epochs": float(requested_epochs),
        "completed_epochs": float(len(train_history)),
    }
    metrics.update({f"final_{key}": value for key, value in final_train.items()})
    metrics.update({f"final_{key}": value for key, value in final_val.items()})
    metrics.update(test_metrics)
    return (
        "application/gzip",
        metrics,
        {
            "api": "legacy",
            "dataset_contract": contract,
            "samples": len(records),
            "split": split,
            "element_types": list(elements),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "foundation_element_refs": args.operation == "finetune",
            "train_history": train_history,
            "val_history": val_history,
            "test_metrics": test_metrics,
            "completion": {
                "requested_epochs": requested_epochs,
                "completed_epochs": len(train_history),
                "normal_completion": normal_completion,
                "all_recorded_metrics_finite": True,
                "native_model_reload": "OK",
                "native_model_class": native_class,
            },
            "environment": _environment(torch),
            "work_dir": str(work),
        },
    )
