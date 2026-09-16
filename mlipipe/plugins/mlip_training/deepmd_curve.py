"""Bounded DeepMD learning-curve validation, shared by the bundled training path.

This optional scientific contract preserves the historical fresh-trajectory
checks. It owns no scheduler, staging, or framework execution.
"""
from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

PLUGIN_ID = "mlip-training"
MAX_SCHEDULED_STEPS = 20000
MAX_CURVE_RECORDS = 4096
MAX_DATASET_SYSTEMS = 4096
RESTART_KEYS = frozenset({"init_model", "restart", "init_frz_model", "finetune", "auto_prob_style"})
DATA_PREFIX = "data"
SCHEDULED_RESULT_SCHEMA = 2
DEEPMD_FINISH_MARKER = "finished training"


def export_training_evidence(output: Path, result: dict[str, Any], dataset_id: str) -> None:
    """Copy actual TensorFlow curve/checkpoint evidence for bounded collection."""
    work = Path(result["provenance"]["work_dir"])
    checkpoint = (work / "checkpoint").read_text(encoding="utf-8")
    match = re.search(r'^model_checkpoint_path:\s*"([^"/]+)"', checkpoint, re.MULTILINE)
    if match is None or ".." in match.group(1):
        raise ValueError("DeepMD checkpoint state lacks a safe checkpoint name")
    index = work / (match.group(1) + ".index")
    for source, name in ((work / "lcurve.out", "lcurve.out"),
                         (work / "checkpoint", "checkpoint"), (index, "model.ckpt.index"),
                         (output / "training.stderr.log", "train.stderr")):
        shutil.copyfile(source, output / name)
    (output / "training-report.json").write_text(json.dumps({
        "schema_version": 1, "framework": "deepmd",
        "framework_version": result["framework_version"],
        "framework_backend": result["provenance"].get("backend"),
        "exit_code": 0, "dataset_id": dataset_id,
        "checkpoint_files": [{"name": index.name}],
    }, indent=2) + "\n", encoding="utf-8")


def calculation(
    context: dict[str, Any], config_path: Path, dataset_path: Path
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Validate the explicit short-trajectory inputs and retain their settings."""
    config, error = _parse_deepmd_config(config_path)
    dataset, dataset_error = _parse_dataset_reference(dataset_path)
    diagnostics = []
    for code, message in (("training.config_contract", error),
                          ("training.dataset_contract", dataset_error)):
        if message:
            diagnostics.append(_diagnostic("error", code, message))
    if diagnostics:
        return diagnostics, {}
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if raw.get("_mlipipe", {}).get("backend", "tf") not in {"tf", "tf2"}:
        diagnostics.append(_diagnostic("error", "training.curve_backend",
                                       "deepmd-curve requires TensorFlow checkpoint evidence"))
    if context["parameters"].get("seed") != config["training"]["seed"]:
        diagnostics.append(_diagnostic("error", "training.seed_mismatch",
                                       "parameters.seed must match training.seed"))
    for role, count in dataset["systems"].items():
        if count != len(config["systems"][role]):
            diagnostics.append(_diagnostic("error", f"training.system_count_{role}",
                                           f"dataset declares {count} {role} systems; config differs"))
    return diagnostics, {
        **config, "dataset_id": dataset["dataset_id"],
        "dataset_reference_path": str(dataset_path),
        "system_counts": {role: len(items) for role, items in config["systems"].items()},
    }

def _diagnostic(level: str, code: str, message: str) -> dict[str, str]:
    return {"level": level, "code": code, "message": message}



def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}



def _plain_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and "\x00" not in value and "\n" not in value



def _positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0



def _safe_relative(value: Any) -> bool:
    """True for a bounded POSIX-relative path such as ``data/train/sys-1``.

    This is the guard that keeps a machine-specific dataset location out of the
    approval plan: the reviewed training config may only name systems relative
    to the run directory, and the site alone knows where ``data`` resolves to.
    """

    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        return False
    path = PurePosixPath(value)
    if path.is_absolute():
        return False
    parts = path.parts
    return bool(parts) and all(part not in {"", ".", ".."} for part in parts)



def _ordinary_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False



def _read_json_file(path: Path) -> tuple[Any, str | None]:
    try:
        if not path.is_file():
            return None, f"not a file: {path.name}"
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return None, str(exc)



def _contains_restart_key(value: Any) -> str | None:
    """Find any resume-shaped key anywhere in the reviewed training config."""

    if isinstance(value, dict):
        for key, nested in value.items():
            if isinstance(key, str) and key in RESTART_KEYS:
                return key
            found = _contains_restart_key(nested)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _contains_restart_key(item)
            if found is not None:
                return found
    return None



def _parse_deepmd_config(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Extract the trajectory-relevant settings of a DeepMD training config.

    Nothing here interprets the science; it reads the declared fields so the
    approval summary states what will actually be optimised, and refuses a
    config whose systems are not portable or whose seeds are not explicit.
    """

    raw, error = _read_json_file(path)
    if error is not None:
        return None, f"training config is unreadable: {error}"
    if not isinstance(raw, dict):
        return None, "training config must be a JSON object"
    model = _mapping(raw.get("model"))
    descriptor = _mapping(model.get("descriptor"))
    fitting = _mapping(model.get("fitting_net"))
    learning_rate = _mapping(raw.get("learning_rate"))
    loss = _mapping(raw.get("loss"))
    training = _mapping(raw.get("training"))
    training_data = _mapping(training.get("training_data"))
    validation_data = _mapping(training.get("validation_data"))

    restart = _contains_restart_key(raw)
    if restart is not None:
        return None, f"scheduled training must be a fresh run; config declares {restart!r}"

    type_map = model.get("type_map")
    if not isinstance(type_map, list) or not type_map or not all(
        _plain_string(item) for item in type_map
    ):
        return None, "model.type_map must be a non-empty list of element strings"

    for name, seed in (
        ("model.descriptor.seed", descriptor.get("seed")),
        ("model.fitting_net.seed", fitting.get("seed")),
        ("training.seed", training.get("seed")),
    ):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            return None, f"{name} must be an explicit non-negative integer"

    numb_steps = training.get("numb_steps", training.get("stop_batch"))
    disp_freq = training.get("disp_freq")
    save_freq = training.get("save_freq")
    if not _positive_int(numb_steps) or numb_steps > MAX_SCHEDULED_STEPS:
        return None, (
            f"training.numb_steps must be a positive integer <= {MAX_SCHEDULED_STEPS} "
            "for a scheduled validation run"
        )
    if not _positive_int(disp_freq) or numb_steps % disp_freq != 0:
        return None, "training.disp_freq must be positive and divide training.numb_steps"
    if not _positive_int(save_freq):
        return None, "training.save_freq must be a positive integer"
    if numb_steps // disp_freq + 1 > MAX_CURVE_RECORDS:
        return None, "the requested learning curve would exceed the bounded record count"
    disp_file = training.get("disp_file", "lcurve.out")
    if disp_file != "lcurve.out":
        return None, "training.disp_file must be lcurve.out for the scheduled contract"

    systems: dict[str, list[str]] = {}
    for role, block in (("training", training_data), ("validation", validation_data)):
        declared = block.get("systems")
        if role == "validation" and declared is None:
            systems[role] = []
            continue
        if not isinstance(declared, list) or not declared or len(declared) > MAX_DATASET_SYSTEMS:
            return None, f"training.{role}_data.systems must be a bounded non-empty list"
        normalized: list[str] = []
        for item in declared:
            if not _safe_relative(item):
                return None, (
                    f"training.{role}_data.systems must be relative paths inside the run "
                    f"directory; got {item!r}"
                )
            if PurePosixPath(str(item)).parts[0] != DATA_PREFIX:
                return None, (
                    f"training.{role}_data.systems must start with {DATA_PREFIX!r} so the "
                    "site alone resolves the dataset location"
                )
            normalized.append(str(item))
        if len(set(normalized)) != len(normalized):
            return None, f"training.{role}_data.systems contains a duplicate system"
        systems[role] = normalized

    return {
        "type_map": list(type_map),
        "descriptor": {
            "type": descriptor.get("type"),
            "seed": descriptor.get("seed"),
            "precision": descriptor.get("precision"),
            "rcut": descriptor.get("rcut"),
            "rcut_smth": descriptor.get("rcut_smth"),
            "neuron": descriptor.get("neuron"),
            "axis_neuron": descriptor.get("axis_neuron"),
            "sel": descriptor.get("sel"),
            "resnet_dt": descriptor.get("resnet_dt"),
            "type_one_side": descriptor.get("type_one_side"),
        },
        "fitting_net": {
            "neuron": fitting.get("neuron"),
            "seed": fitting.get("seed"),
            "precision": fitting.get("precision"),
            "resnet_dt": fitting.get("resnet_dt"),
        },
        "learning_rate": dict(sorted(learning_rate.items())),
        "loss": dict(sorted(loss.items())),
        "training": {
            "numb_steps": int(numb_steps),
            "disp_freq": int(disp_freq),
            "save_freq": int(save_freq),
            "seed": training.get("seed"),
            "training_batch_size": training_data.get("batch_size"),
            "validation_batch_size": validation_data.get("batch_size"),
            "validation_numb_btch": validation_data.get("numb_btch"),
        },
        "systems": systems,
    }, None



def _parse_dataset_reference(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read the project's logical dataset reference.

    The project names a dataset by id and path. Where it lives on a given
    cluster is site knowledge and stays in the remote
    template library, exactly like a partition or a module name.
    """

    raw, error = _read_json_file(path)
    if error is not None:
        return None, f"dataset reference is unreadable: {error}"
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        return None, "dataset reference must be a JSON object with schema_version 1"
    dataset_id = raw.get("dataset_id")
    if not _plain_string(dataset_id) or not _safe_relative(dataset_id) or "/" in str(dataset_id):
        return None, "dataset reference dataset_id must be a single safe path segment"
    counts = _mapping(raw.get("systems"))
    for role in ("training", "validation"):
        if role in counts and not _positive_int(counts.get(role)):
            return None, f"dataset reference systems.{role} must be a positive integer"
    return {
        "dataset_id": str(dataset_id),
        "systems": {key: counts[key] for key in sorted(counts) if key in {"training", "validation"}},
        "description": raw.get("description"),
    }, None



def _parse_lcurve(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a DeepMD learning curve using its own header.

    Column names are read from the file rather than assumed, because which
    columns exist depends on the configured loss: a run with no virial
    prefactor simply has no virial columns, and a run without validation data
    prints ``nan``.  Nothing here decides what a column means.
    """

    if not _ordinary_file(path):
        return None, "lcurve.out is missing or empty"
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        return None, f"lcurve.out is unreadable: {exc}"
    columns: list[str] | None = None
    records: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if columns is None:
                tokens = stripped.lstrip("#").split()
                if len(tokens) < 2 or tokens[0] != "step":
                    return None, "lcurve.out header does not start with a step column"
                columns = tokens
            continue
        if columns is None:
            return None, "lcurve.out has data before its header"
        fields = stripped.split()
        if len(fields) != len(columns):
            return None, f"lcurve.out line {number} has {len(fields)} of {len(columns)} columns"
        try:
            step = int(fields[0])
        except ValueError:
            return None, f"lcurve.out line {number} has a non-integer step"
        record: dict[str, Any] = {"step": step}
        for name, raw in zip(columns[1:], fields[1:]):
            try:
                value = float(raw)
            except ValueError:
                return None, f"lcurve.out line {number} column {name} is not a number"
            if not math.isfinite(value):
                return None, f"lcurve.out line {number} column {name} is not finite ({raw})"
            record[name] = value
        records.append(record)
        if len(records) > MAX_CURVE_RECORDS:
            return None, "lcurve.out exceeds the bounded record count"
    if columns is None or not records:
        return None, "lcurve.out contains no training records"
    steps = [record["step"] for record in records]
    if steps != sorted(steps) or len(set(steps)) != len(steps):
        return None, "lcurve.out steps are not strictly increasing"
    return {"columns": columns, "records": records, "steps": steps}, None



def _check_scheduled_training(context: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    """Verify that a COMPLETED scheduler job actually produced a training run.

    A zero scheduler exit code says the launcher returned; it says nothing about
    whether DeepMD reached the requested step, whether every reported loss is
    finite, or whether a checkpoint exists.  Each of those is checked here from
    the fetched artifacts.
    """

    diagnostics: list[dict[str, str]] = []
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    parameters = _mapping(context.get("parameters"))
    planned = _mapping(_mapping(context.get("execution")).get("plan"))
    calculation = _mapping(planned.get("training_calculation"))
    scheduled = _mapping(planned.get("scheduled_execution"))
    hpc_execution = _mapping(_mapping(context.get("execution")).get("hpc_execution"))
    expected_steps = _mapping(calculation.get("training")).get("numb_steps")
    disp_freq = _mapping(calculation.get("training")).get("disp_freq")
    if not _positive_int(expected_steps) or not _positive_int(disp_freq):
        diagnostics.append(
            _diagnostic(
                "error",
                "training.plan",
                "the pinned plan does not declare numb_steps and disp_freq",
            )
        )
        return diagnostics, None

    report, report_error = _read_json_file(attempt / "training-report.json")
    if report_error is not None or not isinstance(report, dict):
        diagnostics.append(
            _diagnostic(
                "error",
                "training.report_unreadable",
                str(report_error or "training-report.json must be a JSON object"),
            )
        )
        return diagnostics, None
    if report.get("schema_version") != 1:
        diagnostics.append(
            _diagnostic("error", "training.report_schema", "training-report.json schema_version must be 1")
        )
    if report.get("framework") != calculation.get("framework", parameters.get("framework")):
        diagnostics.append(
            _diagnostic("error", "training.report_framework", "the remote report names a different framework")
        )
    if report.get("exit_code") != 0:
        diagnostics.append(
            _diagnostic(
                "error",
                "training.report_exit_code",
                f"the training process exited with {report.get('exit_code')!r}",
            )
        )
    if not _plain_string(report.get("framework_version")):
        diagnostics.append(
            _diagnostic("error", "training.report_version", "the remote report records no framework version")
        )
    if report.get("dataset_id") != calculation.get("dataset_id"):
        diagnostics.append(
            _diagnostic("error", "training.report_dataset_id", "the remote report names a different dataset")
        )
    checkpoints = report.get("checkpoint_files")
    if not isinstance(checkpoints, list) or not checkpoints:
        diagnostics.append(
            _diagnostic("error", "training.report_checkpoints", "the remote report lists no checkpoint artifact")
        )
        checkpoints = []
    else:
        for index, item in enumerate(checkpoints):
            if not isinstance(item, dict) or not _plain_string(item.get("name")):
                diagnostics.append(
                    _diagnostic(
                        "error",
                        f"training.report_checkpoint_{index}",
                        "each checkpoint record needs a name",
                    )
                )
    if scheduled.get("template_family") and report.get("template_family") not in {
        None,
        scheduled.get("template_family"),
    }:
        diagnostics.append(
            _diagnostic("error", "training.report_template", "the remote report names a different template family")
        )

    curve, curve_error = _parse_lcurve(attempt / "lcurve.out")
    if curve_error is not None or curve is None:
        diagnostics.append(
            _diagnostic("error", "training.curve_unreadable", str(curve_error or "lcurve.out is unusable"))
        )
        return diagnostics, None
    if "lr" not in curve["columns"]:
        diagnostics.append(
            _diagnostic("error", "training.curve_lr", "lcurve.out has no learning-rate column")
        )
    else:
        if any(record["lr"] <= 0.0 for record in curve["records"]):
            diagnostics.append(
                _diagnostic("error", "training.curve_lr_value", "a reported learning rate is not positive")
            )
    expected_series = list(range(0, int(expected_steps) + 1, int(disp_freq)))
    if curve["steps"] != expected_series:
        diagnostics.append(
            _diagnostic(
                "error",
                "training.curve_steps",
                f"training reported steps {curve['steps'][:3]}..{curve['steps'][-1:]}, "
                f"but the approved plan requires every multiple of {disp_freq} up to {expected_steps}",
            )
        )

    log_path = attempt / "train.stderr"
    if not _ordinary_file(log_path):
        diagnostics.append(
            _diagnostic("error", "training.log_missing", "the DeepMD training log was not fetched")
        )
    else:
        log = log_path.read_text(encoding="utf-8", errors="replace")
        if DEEPMD_FINISH_MARKER not in log:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "training.log_incomplete",
                    "the training log has no normal completion marker; the scheduler exit "
                    "code alone does not establish that training finished",
                )
            )
    for name in ("checkpoint", "model.ckpt.index"):
        if not _ordinary_file(attempt / name):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "training.model_artifact",
                    f"expected model artifact is missing or empty: {name}",
                )
            )
    reported_templates = report.get("templates")
    if hpc_execution and reported_templates is not None and reported_templates != hpc_execution.get(
        "template_paths"
    ):
        diagnostics.append(
            _diagnostic("error", "training.report_templates", "the remote report cites different templates")
        )
    if diagnostics:
        return diagnostics, None
    return diagnostics, {"report": report, "curve": curve, "checkpoints": checkpoints}



def _collect_scheduled_training(context: dict[str, Any]) -> dict[str, Any]:
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    parameters = _mapping(context["parameters"])
    planned = _mapping(_mapping(context.get("execution")).get("plan"))
    calculation = _mapping(planned.get("training_calculation"))
    hpc_execution = _mapping(_mapping(context.get("execution")).get("hpc_execution"))
    report, _ = _read_json_file(attempt / "training-report.json")
    curve, _ = _parse_lcurve(attempt / "lcurve.out")
    assert isinstance(report, dict) and curve is not None

    manifest = {
        "schema_version": SCHEDULED_RESULT_SCHEMA,
        "plugin_id": PLUGIN_ID,
        "status": "OK",
        "framework": str(parameters["framework"]),
        "framework_version": report.get("framework_version"),
        "framework_backend": report.get("framework_backend"),
        "operation": "train",
        "fresh_training": True,
        "device": parameters.get("device"),
        "precision": parameters.get("precision"),
        "seed": parameters.get("seed"),
        "config_path": calculation.get("config_path"),
        "dataset": {
            "id": calculation.get("dataset_id"),
            "reference_path": calculation.get("dataset_reference_path"),
            "system_counts": calculation.get("system_counts"),
            "systems": calculation.get("systems"),
        },
        "model": {
            "type_map": calculation.get("type_map"),
            "descriptor": calculation.get("descriptor"),
            "fitting_net": calculation.get("fitting_net"),
        },
        "optimization": {
            "learning_rate": calculation.get("learning_rate"),
            "loss": calculation.get("loss"),
            "training": calculation.get("training"),
        },
        "training_curve": {
            "columns": curve["columns"],
            "record_count": len(curve["records"]),
            "records": curve["records"],
        },
        "completed_steps": curve["steps"][-1],
        "requested_steps": _mapping(calculation.get("training")).get("numb_steps"),
        "model_artifacts": report["checkpoint_files"],
        "execution": {
            "template_family": _mapping(planned.get("scheduled_execution")).get("template_family"),
            "templates": hpc_execution.get("template_paths"),
            "host": report.get("host"),
            "threads": report.get("threads"),
        },
        "artifacts": [
            {"name": name, "path": name}
            for name in ("lcurve.out", "training-report.json", "checkpoint", "model.ckpt.index")
            if _ordinary_file(attempt / name)
        ],
    }
    result_name = "deepmd-curve-result.json"
    if not _safe_relative(result_name) or "/" in result_name:
        return {
            "plugin_id": PLUGIN_ID,
            "status": "FAIL",
            "diagnostics": [
                _diagnostic("error", "training.result_name", "result_manifest must be a plain file name")
            ],
        }
    result_path = attempt / result_name
    result_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metrics: dict[str, float] = {"completed_steps": float(curve["steps"][-1])}
    final = curve["records"][-1]
    for name, value in final.items():
        if name != "step" and isinstance(value, float):
            metrics[f"final_{name}"] = value
    artifacts = [
        {"path": str(result_path), "role": "training-curve-manifest", "media_type": "application/json"}
    ]
    for name, role in (
        ("lcurve.out", "training-curve"),
        ("training-report.json", "training-report"),
        ("checkpoint", "model-checkpoint-state"),
        ("model.ckpt.index", "model-checkpoint-index"),
    ):
        if _ordinary_file(attempt / name):
            artifacts.append({"path": str(attempt / name), "role": role})
    return {
        "plugin_id": PLUGIN_ID,
        "status": "OK",
        "diagnostics": [],
        "metrics": metrics,
        "artifacts": artifacts,
    }
