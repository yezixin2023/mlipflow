"""Safe external-wrapper planner for MLIP training frameworks.

This module is intentionally framework-free.  It translates reviewed,
explicit context fields into an argv list for a user-owned wrapper and parses
only a versioned result manifest.  It never invokes a shell or a trainer.

Two execution shapes are supported.  On the ``local`` backend the adapter plans
argv for a user-owned wrapper.  On the ``ssh-slurm`` backend it plans one
scheduler job for a DeepMD fresh training run: the reviewed training config and
a logical dataset reference are staged, a site-owned remote template launches
``dp train``, and only bounded, declared artifacts come back.  Neither path
imports or reimplements a training framework, and neither path may carry a
site-specific absolute path into the approval plan.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any


PLUGIN_ID = "mlip-training"
FRAMEWORK_OPERATIONS = {
    "deepmd": frozenset({"train"}),
    "m3gnet": frozenset({"train", "finetune"}),
    "chgnet": frozenset({"train", "finetune"}),
    "mace": frozenset({"train", "finetune"}),
}
LOCAL_BACKEND = "local"
SCHEDULED_BACKEND = "ssh-slurm"
EXECUTION_BACKENDS = frozenset({LOCAL_BACKEND, SCHEDULED_BACKEND})
# Only DeepMD fresh training has a verified scheduler contract; the other three
# frameworks stay on the local wrapper path until their own remote template and
# completion evidence exist.
SCHEDULED_FRAMEWORKS = {"deepmd": frozenset({"train"})}
TEMPLATE_FAMILIES = {"deepmd": "deepmd"}
SHELL_EXECUTABLES = frozenset(
    {"bash", "csh", "cmd", "dash", "fish", "ksh", "powershell", "pwsh", "sh", "tcsh", "zsh"}
)
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_CONFIG_BYTES = 4 * 1024 * 1024
MAX_LCURVE_BYTES = 32 * 1024 * 1024
MAX_TRAINING_LOG_BYTES = 64 * 1024 * 1024
MAX_CHECKPOINT_INDEX_BYTES = 64 * 1024 * 1024
# A validation/reproduction run, not production training.  The bound is a
# refusal, never a silent truncation.
MAX_SCHEDULED_STEPS = 20000
MAX_CURVE_RECORDS = 4096
MAX_DATASET_SYSTEMS = 4096
# Restarting from a checkpoint would make an "early training trajectory" claim
# meaningless, so a scheduled plan refuses any resume-shaped key outright.
RESTART_KEYS = frozenset({"init_model", "restart", "init_frz_model", "finetune", "auto_prob_style"})
DATA_PREFIX = "data"
SCHEDULED_RESULT_SCHEMA = 2
DEEPMD_FINISH_MARKER = "finished training"


def _diagnostic(level: str, code: str, message: str) -> dict[str, str]:
    return {"level": level, "code": code, "message": message}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _reference(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict) and isinstance(value.get("path"), str) and value["path"]:
        return value["path"]
    return None


def _plain_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and "\x00" not in value and "\n" not in value


def _project_path(context: dict[str, Any], value: Any) -> str:
    reference = _reference(value)
    if reference is None:
        raise ValueError("missing path reference")
    path = Path(reference)
    if path.is_absolute():
        return str(path)
    return str(Path(str(context["project_root"])) / path)


def _safe_attempt_output(context: dict[str, Any], value: Any) -> str:
    reference = _reference(value)
    if reference is None:
        raise ValueError("missing output path reference")
    path = Path(reference)
    if path.is_absolute() or path in {Path("."), Path("")} or ".." in path.parts:
        raise ValueError("output must be a relative path inside attempt_dir")
    return str(Path(str(context["attempt_dir"])) / path)


def _read_json(path: Path) -> Any:
    size = path.stat().st_size
    if size > MAX_JSON_BYTES:
        raise ValueError(f"JSON artifact exceeds {MAX_JSON_BYTES} bytes: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _portable_child(base: Path, value: Any, field: str) -> Path:
    reference = _reference(value)
    if reference is None:
        raise ValueError(f"{field} must contain a non-empty path")
    path = Path(reference)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be relative to the result manifest")
    return base / path


def _validate_metrics(value: Any) -> tuple[dict[str, float], list[dict[str, str]]]:
    if value is None:
        return {}, []
    if not isinstance(value, dict):
        return {}, [_diagnostic("error", "training.metrics_type", "metrics must be an object")]
    diagnostics: list[dict[str, str]] = []
    metrics: dict[str, float] = {}
    for name, raw in sorted(value.items()):
        if not isinstance(name, str) or not name:
            diagnostics.append(_diagnostic("error", "training.metric_name", "metric names must be non-empty strings"))
        elif isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            diagnostics.append(_diagnostic("error", "training.metric_value", f"metric {name!r} must be finite"))
        else:
            metrics[name] = float(raw)
    return metrics, diagnostics


def _load_result(path: Path, context: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, float], list[dict[str, Any]], list[dict[str, str]]]:
    diagnostics: list[dict[str, str]] = []
    try:
        raw = _read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return None, {}, [], [_diagnostic("error", "training.result_unreadable", str(exc))]
    if not isinstance(raw, dict):
        return None, {}, [], [_diagnostic("error", "training.result_type", "result manifest must be a JSON object")]
    parameters = _mapping(context.get("parameters"))
    if raw.get("schema_version") != 1:
        diagnostics.append(_diagnostic("error", "training.result_schema", "schema_version must equal 1"))
    if raw.get("plugin_id") != PLUGIN_ID:
        diagnostics.append(_diagnostic("error", "training.result_plugin", f"plugin_id must equal {PLUGIN_ID}"))
    if raw.get("status") not in {"OK", "FAIL"}:
        diagnostics.append(_diagnostic("error", "training.result_status", "status must be OK or FAIL"))
    if raw.get("framework") not in FRAMEWORK_OPERATIONS:
        diagnostics.append(_diagnostic("error", "training.result_framework", "framework is unsupported"))
    elif parameters.get("framework") in FRAMEWORK_OPERATIONS and raw.get("framework") != parameters.get("framework"):
        diagnostics.append(_diagnostic("error", "training.framework_mismatch", "result framework does not match the plan"))
    seed = raw.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        diagnostics.append(_diagnostic("error", "training.result_seed", "seed must be a non-negative integer"))
    elif isinstance(parameters.get("seed"), int) and seed != parameters.get("seed"):
        diagnostics.append(_diagnostic("error", "training.seed_mismatch", "result seed does not match the plan"))
    if not _plain_string(raw.get("framework_version")):
        diagnostics.append(
            _diagnostic("error", "training.result_framework_version", "framework_version must be a non-empty string")
        )
    for key in ("dataset_fingerprint", "config_fingerprint"):
        value = raw.get(key)
        if value is None or value == "":
            diagnostics.append(_diagnostic("error", f"training.result_{key}", f"{key} is required"))
        elif value != parameters.get(key):
            diagnostics.append(
                _diagnostic("error", f"training.{key}_mismatch", f"result {key} does not match the approved plan")
            )
    for key in ("operation", "device", "precision"):
        if not _plain_string(raw.get(key)):
            diagnostics.append(
                _diagnostic("error", f"training.result_{key}", f"result {key} is required")
            )
        elif raw.get(key) != parameters.get(key):
            diagnostics.append(
                _diagnostic("error", f"training.{key}_mismatch", f"result {key} does not match the approved plan")
            )
    if parameters.get("operation") == "finetune":
        value = raw.get("foundation_model_fingerprint")
        if value != parameters.get("foundation_model_fingerprint"):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "training.foundation_model_fingerprint_mismatch",
                    "result foundation model fingerprint does not match the approved plan",
                )
            )

    model_path: Path | None = None
    model = raw.get("model_artifact")
    if not isinstance(model, dict):
        diagnostics.append(_diagnostic("error", "training.model_artifact", "model_artifact must be an object"))
    else:
        try:
            model_path = _portable_child(path.parent, model.get("path"), "model_artifact.path")
        except ValueError as exc:
            diagnostics.append(_diagnostic("error", "training.model_path", str(exc)))
        if model_path is not None:
            if not model_path.is_file():
                diagnostics.append(_diagnostic("error", "training.model_missing", f"model artifact is not a regular file: {model_path}"))
            try:
                expected = Path(_safe_attempt_output(context, _mapping(context.get("inputs")).get("output")))
                if model_path != expected:
                    diagnostics.append(
                        _diagnostic("error", "training.model_output_mismatch", "model artifact does not match inputs.output")
                    )
            except (KeyError, ValueError):
                pass

    metrics, metric_diagnostics = _validate_metrics(raw.get("metrics"))
    diagnostics.extend(metric_diagnostics)
    artifacts = [{"path": str(path), "role": "result-manifest", "media_type": "application/json"}]
    if model_path is not None:
        artifacts.append(
            {
                "path": str(model_path),
                "role": "model",
                "media_type": model.get("media_type", "application/octet-stream") if isinstance(model, dict) else None,
            }
        )
    return raw, metrics, artifacts, diagnostics


# ---------------------------------------------------------------------------
# Scheduled DeepMD training
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _is_fingerprint(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


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


def _ordinary_project_file(path: Path, project_root: Path, max_bytes: int) -> bool:
    try:
        if path.is_symlink() or not path.is_file():
            return False
        resolved = path.resolve()
        resolved.relative_to(project_root.resolve())
        for parent in [path, *path.parents]:
            if parent == project_root:
                break
            if parent.is_symlink():
                return False
        return 0 < resolved.stat().st_size <= max_bytes
    except (OSError, ValueError):
        return False


def _ordinary_file(path: Path, max_bytes: int | None = None) -> bool:
    try:
        if path.is_symlink() or not path.is_file():
            return False
        size = path.stat().st_size
        return size > 0 and (max_bytes is None or size <= max_bytes)
    except OSError:
        return False


def _read_json_file(path: Path, max_bytes: int = MAX_JSON_BYTES) -> tuple[Any, str | None]:
    try:
        if path.is_symlink() or not path.is_file():
            return None, f"not an ordinary file: {path.name}"
        if path.stat().st_size > max_bytes:
            return None, f"exceeds {max_bytes} bytes: {path.name}"
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return None, str(exc)


def _project_file(context: Any, reference: Any) -> Path:
    root = Path(str(context["project_root"])).expanduser().absolute()
    return (root / str(_reference(reference) or "")).absolute()


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
    """Extract the trajectory-relevant identity of a DeepMD training config.

    Nothing here interprets the science; it reads the declared fields so the
    approval summary states what will actually be optimised, and refuses a
    config whose systems are not portable or whose seeds are not explicit.
    """

    raw, error = _read_json_file(path, MAX_CONFIG_BYTES)
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

    The project names a dataset by id and pins its content fingerprint.  Where
    that id lives on a given cluster is site knowledge and stays in the remote
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
    if not _is_fingerprint(raw.get("fingerprint")):
        return None, "dataset reference fingerprint must be sha256:<64 hex>"
    counts = _mapping(raw.get("systems"))
    for role in ("training", "validation"):
        if role in counts and not _positive_int(counts.get(role)):
            return None, f"dataset reference systems.{role} must be a positive integer"
    return {
        "dataset_id": str(dataset_id),
        "fingerprint": str(raw["fingerprint"]),
        "systems": {key: counts[key] for key in sorted(counts) if key in {"training", "validation"}},
        "description": raw.get("description"),
    }, None


def _staged_record(source: Path, remote_name: str) -> dict[str, Any]:
    return {
        "source": str(source.absolute()),
        "remote_name": remote_name,
        "sha256": _sha256(source),
        "size_bytes": source.stat().st_size,
        "sensitive": False,
        "fetch_allowed": False,
    }


def _validate_scheduled(context: dict[str, Any]) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    inputs = _mapping(context.get("inputs"))
    parameters = _mapping(context.get("parameters"))
    framework = parameters.get("framework")
    operation = parameters.get("operation", "train")
    if framework not in SCHEDULED_FRAMEWORKS:
        diagnostics.append(
            _diagnostic(
                "error",
                "training.scheduled_framework",
                "only deepmd has a verified ssh-slurm training contract",
            )
        )
    elif operation not in SCHEDULED_FRAMEWORKS[framework]:
        diagnostics.append(
            _diagnostic(
                "error",
                "training.scheduled_operation",
                f"{framework} supports only fresh train on ssh-slurm",
            )
        )
    for key in ("training_config", "dataset_reference"):
        if _reference(inputs.get(key)) is None:
            diagnostics.append(
                _diagnostic("error", f"training.input_{key}", f"inputs.{key} is required")
            )
    for key in ("executable", "script", "output", "result_manifest"):
        if _reference(inputs.get(key)) is not None:
            diagnostics.append(
                _diagnostic(
                    "error",
                    f"training.local_only_{key}",
                    f"inputs.{key} belongs to the local wrapper contract and must not "
                    "appear in a scheduled plan",
                )
            )
    return diagnostics


def _plan_scheduled_training(context: dict[str, Any]) -> dict[str, Any]:
    """Plan one scheduler job that runs one DeepMD fresh training.

    The plan pins what determines the optimisation trajectory — seeds, learning
    rate schedule, loss prefactors, model architecture, system order, dataset
    fingerprint — so an approval reviewer sees the science, while the site
    template alone knows which ``dp`` binary and which dataset root are used.
    """

    diagnostics: list[dict[str, str]] = []
    inputs = _mapping(context["inputs"])
    parameters = _mapping(context["parameters"])
    project_root = Path(str(context["project_root"])).expanduser().absolute()
    config_path = _project_file(context, inputs["training_config"])
    dataset_path = _project_file(context, inputs["dataset_reference"])
    for name, path, limit in (
        ("training_config", config_path, MAX_CONFIG_BYTES),
        ("dataset_reference", dataset_path, MAX_JSON_BYTES),
    ):
        if not _ordinary_project_file(path, project_root, limit):
            diagnostics.append(
                _diagnostic(
                    "error",
                    f"inputs.{name}",
                    f"{name} must be an ordinary bounded file inside the project root",
                )
            )
    if diagnostics:
        return {
            "plugin_id": PLUGIN_ID,
            "status": "BLOCKED",
            "executable": False,
            "diagnostics": diagnostics,
        }

    config, config_error = _parse_deepmd_config(config_path)
    dataset, dataset_error = _parse_dataset_reference(dataset_path)
    if config_error is not None:
        diagnostics.append(_diagnostic("error", "training.config_contract", config_error))
    if dataset_error is not None:
        diagnostics.append(_diagnostic("error", "training.dataset_contract", dataset_error))
    if diagnostics or config is None or dataset is None:
        return {
            "plugin_id": PLUGIN_ID,
            "status": "BLOCKED",
            "executable": False,
            "diagnostics": diagnostics,
        }

    config_fingerprint = _sha256(config_path)
    if parameters.get("config_fingerprint") != config_fingerprint:
        diagnostics.append(
            _diagnostic(
                "error",
                "training.config_fingerprint_mismatch",
                "parameters.config_fingerprint does not match the reviewed training config",
            )
        )
    if parameters.get("dataset_fingerprint") != dataset["fingerprint"]:
        diagnostics.append(
            _diagnostic(
                "error",
                "training.dataset_fingerprint_mismatch",
                "parameters.dataset_fingerprint does not match the dataset reference",
            )
        )
    if parameters.get("seed") != config["training"]["seed"]:
        diagnostics.append(
            _diagnostic(
                "error",
                "training.seed_mismatch",
                "parameters.seed does not match training.seed in the reviewed config",
            )
        )
    declared = dataset.get("systems", {})
    for role in ("training", "validation"):
        expected = declared.get(role)
        if expected is not None and expected != len(config["systems"].get(role, [])):
            diagnostics.append(
                _diagnostic(
                    "error",
                    f"training.system_count_{role}",
                    f"dataset reference declares {expected} {role} systems but the config "
                    f"lists {len(config['systems'].get(role, []))}",
                )
            )
    if diagnostics:
        return {
            "plugin_id": PLUGIN_ID,
            "status": "BLOCKED",
            "executable": False,
            "diagnostics": diagnostics,
        }

    staged = [
        _staged_record(config_path, "input.json"),
        _staged_record(dataset_path, "dataset.json"),
    ]
    fetch_outputs = [
        {
            "remote_name": "lcurve.out",
            "local_name": "lcurve.out",
            "required": True,
            "max_bytes": MAX_LCURVE_BYTES,
            "role": "training-curve",
        },
        {
            "remote_name": "training-report.json",
            "local_name": "training-report.json",
            "required": True,
            "max_bytes": MAX_JSON_BYTES,
            "role": "training-report",
        },
        {
            "remote_name": "checkpoint",
            "local_name": "checkpoint",
            "required": True,
            "max_bytes": 64 * 1024,
            "role": "model-checkpoint-state",
        },
        {
            "remote_name": "model.ckpt.index",
            "local_name": "model.ckpt.index",
            "required": True,
            "max_bytes": MAX_CHECKPOINT_INDEX_BYTES,
            "role": "model-checkpoint-index",
        },
        {
            "remote_name": "train.stderr",
            "remote_path": "logs/train.stderr",
            "local_name": "train.stderr",
            "required": True,
            "max_bytes": MAX_TRAINING_LOG_BYTES,
            "role": "training-log",
        },
        {
            "remote_name": "train.stdout",
            "remote_path": "logs/train.stdout",
            "local_name": "train.stdout",
            "required": False,
            "max_bytes": MAX_TRAINING_LOG_BYTES,
            "role": "training-log",
        },
    ]
    training = config["training"]
    return {
        "plugin_id": PLUGIN_ID,
        "status": "READY",
        "executable": True,
        "argv": [f"template-family:{TEMPLATE_FAMILIES[str(parameters['framework'])]}"],
        "cwd": "remote-attempt-workspace",
        "expected_outputs": [item["remote_name"] for item in fetch_outputs if item["required"]],
        "diagnostics": [],
        "operation": "train",
        "framework": str(parameters["framework"]),
        "training_identity": {
            "config_fingerprint": config_fingerprint,
            "dataset_id": dataset["dataset_id"],
            "dataset_fingerprint": dataset["fingerprint"],
            "type_map": config["type_map"],
            "descriptor": config["descriptor"],
            "fitting_net": config["fitting_net"],
            "learning_rate": config["learning_rate"],
            "loss": config["loss"],
            "training": training,
            "system_counts": {
                role: len(items) for role, items in sorted(config["systems"].items())
            },
            # Order matters to the optimisation trajectory, so it is pinned by
            # content rather than by an unordered set.
            "system_order_fingerprint": "sha256:"
            + hashlib.sha256(
                "\n".join(
                    f"{role}\t{item}"
                    for role in sorted(config["systems"])
                    for item in config["systems"][role]
                ).encode("utf-8")
            ).hexdigest(),
        },
        "approval_summary": {
            "expensive": True,
            "submits_jobs": True,
            "fresh_training": True,
            "execution_model": "single-python",
            "cpus_meaning": "threads-per-process",
            "framework": str(parameters["framework"]),
            "numb_steps": training["numb_steps"],
            "disp_freq": training["disp_freq"],
            "reporting_points": training["numb_steps"] // training["disp_freq"] + 1,
            "seeds": {
                "training": training["seed"],
                "descriptor": config["descriptor"]["seed"],
                "fitting_net": config["fitting_net"]["seed"],
            },
            "learning_rate": config["learning_rate"],
            "loss": config["loss"],
            "dataset_id": dataset["dataset_id"],
            "system_counts": {
                role: len(items) for role, items in sorted(config["systems"].items())
            },
            "fetch_allowlist": sorted(item["remote_name"] for item in fetch_outputs),
            "staged_file_count": len(staged),
        },
        "input_fingerprints": {
            "training_config": config_fingerprint,
            "dataset_reference": _sha256(dataset_path),
        },
        "scheduled_execution": {
            "schema_version": 3,
            "execution_model": "single-python",
            "template_family": TEMPLATE_FAMILIES[str(parameters["framework"])],
            "staged_files": staged,
            "fetch_outputs": fetch_outputs,
        },
    }


def _parse_lcurve(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a DeepMD learning curve using its own header.

    Column names are read from the file rather than assumed, because which
    columns exist depends on the configured loss: a run with no virial
    prefactor simply has no virial columns, and a run without validation data
    prints ``nan``.  Nothing here decides what a column means.
    """

    if not _ordinary_file(path, MAX_LCURVE_BYTES):
        return None, "lcurve.out is missing, empty or oversized"
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


def _scheduled_plan(context: dict[str, Any]) -> dict[str, Any]:
    return _mapping(_mapping(context.get("execution")).get("plan"))


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
    planned = _scheduled_plan(context)
    identity = _mapping(planned.get("training_identity"))
    scheduled = _mapping(planned.get("scheduled_execution"))
    hpc_execution = _mapping(_mapping(context.get("execution")).get("hpc_execution"))
    expected_steps = _mapping(identity.get("training")).get("numb_steps")
    disp_freq = _mapping(identity.get("training")).get("disp_freq")
    if not _positive_int(expected_steps) or not _positive_int(disp_freq):
        diagnostics.append(
            _diagnostic(
                "error",
                "training.plan_identity",
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
    if report.get("framework") != identity.get("framework", parameters.get("framework")):
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
    if report.get("config_sha256") != identity.get("config_fingerprint"):
        diagnostics.append(
            _diagnostic(
                "error",
                "training.report_config_identity",
                "the staged training config on the cluster is not the approved one",
            )
        )
    if report.get("dataset_id") != identity.get("dataset_id"):
        diagnostics.append(
            _diagnostic("error", "training.report_dataset_id", "the remote report names a different dataset")
        )
    if report.get("dataset_fingerprint") != identity.get("dataset_fingerprint"):
        diagnostics.append(
            _diagnostic(
                "error",
                "training.report_dataset_identity",
                "the dataset recomputed on the cluster does not match the approved fingerprint",
            )
        )
    checkpoints = report.get("checkpoint_files")
    if not isinstance(checkpoints, list) or not checkpoints:
        diagnostics.append(
            _diagnostic("error", "training.report_checkpoints", "the remote report lists no checkpoint artifact")
        )
        checkpoints = []
    else:
        for index, item in enumerate(checkpoints):
            if not isinstance(item, dict) or not _plain_string(item.get("name")) or not _positive_int(
                item.get("size_bytes")
            ):
                diagnostics.append(
                    _diagnostic(
                        "error",
                        f"training.report_checkpoint_{index}",
                        "each checkpoint record needs a name and a positive size",
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
    if not _ordinary_file(log_path, MAX_TRAINING_LOG_BYTES):
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
    for name, limit in (
        ("checkpoint", 64 * 1024),
        ("model.ckpt.index", MAX_CHECKPOINT_INDEX_BYTES),
    ):
        if not _ordinary_file(attempt / name, limit):
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
    diagnostics, analysis = _check_scheduled_training(context)
    if diagnostics or analysis is None:
        return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    parameters = _mapping(context["parameters"])
    planned = _scheduled_plan(context)
    identity = _mapping(planned.get("training_identity"))
    hpc_execution = _mapping(_mapping(context.get("execution")).get("hpc_execution"))
    report = analysis["report"]
    curve = analysis["curve"]

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
        "config_fingerprint": identity.get("config_fingerprint"),
        "dataset": {
            "id": identity.get("dataset_id"),
            "fingerprint": identity.get("dataset_fingerprint"),
            "system_counts": identity.get("system_counts"),
            "system_order_fingerprint": identity.get("system_order_fingerprint"),
        },
        "model": {
            "type_map": identity.get("type_map"),
            "descriptor": identity.get("descriptor"),
            "fitting_net": identity.get("fitting_net"),
        },
        "optimization": {
            "learning_rate": identity.get("learning_rate"),
            "loss": identity.get("loss"),
            "training": identity.get("training"),
        },
        "training_curve": {
            "columns": curve["columns"],
            "record_count": len(curve["records"]),
            "records": curve["records"],
        },
        "completed_steps": curve["steps"][-1],
        "requested_steps": _mapping(identity.get("training")).get("numb_steps"),
        "model_artifacts": analysis["checkpoints"],
        "execution": {
            "template_family": _mapping(planned.get("scheduled_execution")).get("template_family"),
            "templates": hpc_execution.get("template_paths"),
            "host": report.get("host"),
            "threads": report.get("threads"),
        },
        "artifacts": [
            {"name": name, "path": name, "sha256": _sha256(attempt / name), "size_bytes": (attempt / name).stat().st_size}
            for name in ("lcurve.out", "training-report.json", "checkpoint", "model.ckpt.index")
            if _ordinary_file(attempt / name)
        ],
    }
    result_name = str(parameters.get("result_manifest", "mlip-training-result.json"))
    if not _safe_relative(result_name) or "/" in result_name:
        return {
            "plugin_id": PLUGIN_ID,
            "status": "FAIL",
            "diagnostics": [
                _diagnostic("error", "training.result_name", "result_manifest must be a plain file name")
            ],
        }
    result_path = attempt / result_name
    if result_path.is_symlink():
        return {
            "plugin_id": PLUGIN_ID,
            "status": "FAIL",
            "diagnostics": [
                _diagnostic("error", "training.result_symlink", "the result manifest path is a symlink")
            ],
        }
    result_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metrics: dict[str, float] = {"completed_steps": float(curve["steps"][-1])}
    final = curve["records"][-1]
    for name, value in final.items():
        if name != "step" and isinstance(value, float):
            metrics[f"final_{name}"] = value
    artifacts = [
        {"path": str(result_path), "role": "training-manifest", "media_type": "application/json"}
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


class Adapter:
    """Build deterministic wrapper argv for four declared MLIP frameworks."""

    def validate(self, context: dict[str, Any]) -> list[dict[str, str]]:
        diagnostics: list[dict[str, str]] = []
        if not isinstance(context, dict):
            return [_diagnostic("error", "context.type", "context must be an object")]
        for key in ("project_root", "attempt_dir"):
            if not _plain_string(context.get(key)):
                diagnostics.append(_diagnostic("error", f"context.{key}", f"{key} must be a non-empty path string"))
        inputs = context.get("inputs")
        parameters = context.get("parameters")
        if not isinstance(inputs, dict):
            diagnostics.append(_diagnostic("error", "context.inputs", "inputs must be an object"))
            inputs = {}
        if not isinstance(parameters, dict):
            diagnostics.append(_diagnostic("error", "context.parameters", "parameters must be an object"))
            parameters = {}
        if not isinstance(context.get("resources"), dict):
            diagnostics.append(_diagnostic("error", "context.resources", "resources must be an object"))
        backend = context.get("backend")
        if backend not in EXECUTION_BACKENDS:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "context.backend",
                    "this adapter supports the local wrapper backend and ssh-slurm scheduled training",
                )
            )

        framework = parameters.get("framework")
        operation = parameters.get("operation", "train")
        if framework not in FRAMEWORK_OPERATIONS:
            diagnostics.append(_diagnostic("error", "training.framework", "framework must be deepmd, m3gnet, chgnet, or mace"))
        elif operation not in FRAMEWORK_OPERATIONS[framework]:
            allowed = ", ".join(sorted(FRAMEWORK_OPERATIONS[framework]))
            diagnostics.append(_diagnostic("error", "training.operation", f"{framework} operation must be one of: {allowed}"))
        seed = parameters.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            diagnostics.append(_diagnostic("error", "training.seed", "parameters.seed must be a non-negative integer"))
        for key in ("device", "precision", "dataset_fingerprint", "config_fingerprint"):
            if not _plain_string(parameters.get(key)):
                diagnostics.append(_diagnostic("error", f"training.{key}", f"parameters.{key} is required"))

        if backend == SCHEDULED_BACKEND:
            # The scheduled contract has its own inputs: there is no local
            # wrapper argv, and the executable lives in the site's remote
            # template rather than in the portable project.
            diagnostics.extend(_validate_scheduled(context))
            return diagnostics

        for key in ("executable", "script", "config", "data", "output", "result_manifest"):
            if _reference(inputs.get(key)) is None:
                diagnostics.append(_diagnostic("error", f"training.input_{key}", f"inputs.{key} is required"))
        executable = _reference(inputs.get("executable"))
        if executable is not None and Path(executable).name.lower() in SHELL_EXECUTABLES:
            diagnostics.append(_diagnostic("error", "training.shell_forbidden", "a shell executable is not permitted"))
        if operation == "finetune" and _reference(inputs.get("foundation_model")) is None:
            diagnostics.append(
                _diagnostic("error", "training.foundation_model", "inputs.foundation_model is required for finetune")
            )
        if operation == "finetune" and not _plain_string(
            parameters.get("foundation_model_fingerprint")
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "training.foundation_model_fingerprint",
                    "parameters.foundation_model_fingerprint is required for finetune",
                )
            )
        for key in ("output", "result_manifest"):
            if _reference(inputs.get(key)) is not None:
                try:
                    _safe_attempt_output(context, inputs[key])
                except (KeyError, ValueError) as exc:
                    diagnostics.append(_diagnostic("error", f"training.{key}_scope", str(exc)))
        return diagnostics

    def plan(self, context: dict[str, Any]) -> dict[str, Any]:
        diagnostics = self.validate(context)
        if diagnostics:
            return {
                "plugin_id": PLUGIN_ID,
                "status": "BLOCKED",
                "executable": False,
                "diagnostics": diagnostics,
            }
        if context.get("backend") == SCHEDULED_BACKEND:
            return _plan_scheduled_training(context)
        inputs = _mapping(context["inputs"])
        parameters = _mapping(context["parameters"])
        model_output = _safe_attempt_output(context, inputs["output"])
        result_manifest = _safe_attempt_output(context, inputs["result_manifest"])
        argv = [
            _reference(inputs["executable"]),
            _project_path(context, inputs["script"]),
            "--framework",
            str(parameters["framework"]),
            "--operation",
            str(parameters.get("operation", "train")),
            "--config",
            _project_path(context, inputs["config"]),
            "--data",
            _project_path(context, inputs["data"]),
            "--output",
            model_output,
            "--result-manifest",
            result_manifest,
            "--seed",
            str(parameters["seed"]),
            "--device",
            str(parameters["device"]),
            "--precision",
            str(parameters["precision"]),
            "--dataset-fingerprint",
            str(parameters["dataset_fingerprint"]),
            "--config-fingerprint",
            str(parameters["config_fingerprint"]),
        ]
        if parameters.get("operation", "train") == "finetune":
            argv.extend(["--foundation-model", _project_path(context, inputs["foundation_model"])])
            argv.extend(
                [
                    "--foundation-model-fingerprint",
                    str(parameters["foundation_model_fingerprint"]),
                ]
            )
        return {
            "plugin_id": PLUGIN_ID,
            "status": "READY",
            "executable": True,
            "argv": argv,
            "cwd": str(context["attempt_dir"]),
            "expected_outputs": [model_output, result_manifest],
            "resources": dict(_mapping(context.get("resources"))),
            "diagnostics": [],
        }

    def prepare(self, context: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(plan, dict) or plan.get("status") != "READY":
            return {
                "plugin_id": PLUGIN_ID,
                "status": "BLOCKED",
                "executable": False,
                "diagnostics": [_diagnostic("error", "training.plan_required", "a READY plan is required")],
            }
        return {
            "plugin_id": PLUGIN_ID,
            "status": "READY",
            "executable": False,
            "prepared": False,
            "message": (
                "No files were written; the scheduler stages the reviewed config and the "
                "external wrapper consumes existing inputs."
            ),
        }

    def check(self, context: dict[str, Any]) -> dict[str, Any]:
        if context.get("backend") == SCHEDULED_BACKEND:
            diagnostics, analysis = _check_scheduled_training(context)
            if diagnostics or analysis is None:
                return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
            return {
                "plugin_id": PLUGIN_ID,
                "status": "OK",
                "diagnostics": [],
                "metrics": {"completed_steps": float(analysis["curve"]["steps"][-1])},
                "artifacts": [],
            }
        try:
            path = Path(_safe_attempt_output(context, _mapping(context.get("inputs")).get("result_manifest")))
        except (KeyError, TypeError, ValueError) as exc:
            return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": [_diagnostic("error", "training.result_path", str(exc))]}
        if not path.is_file():
            return {
                "plugin_id": PLUGIN_ID,
                "status": "WAIT",
                "diagnostics": [_diagnostic("info", "training.result_pending", f"result manifest not found: {path}")],
            }
        raw, metrics, artifacts, diagnostics = _load_result(path, context)
        status = "FAIL" if diagnostics or raw is None or raw.get("status") == "FAIL" else "OK"
        if raw is not None and raw.get("status") == "FAIL" and not diagnostics:
            diagnostics.append(_diagnostic("error", "training.reported_failure", "training result reports FAIL"))
        return {
            "plugin_id": PLUGIN_ID,
            "status": status,
            "diagnostics": diagnostics,
            "metrics": metrics,
            "artifacts": artifacts,
        }

    def collect(self, context: dict[str, Any]) -> dict[str, Any]:
        if context.get("backend") == SCHEDULED_BACKEND:
            return _collect_scheduled_training(context)
        checked = self.check(context)
        if checked["status"] != "OK":
            return checked
        return {
            "plugin_id": PLUGIN_ID,
            "status": "OK",
            "artifacts": checked["artifacts"],
            "metrics": checked["metrics"],
            "diagnostics": [],
        }

    def replay(self, context: dict[str, Any]) -> dict[str, Any]:
        """Replay only parses an already existing manifest and model reference."""

        return self.collect(context)
