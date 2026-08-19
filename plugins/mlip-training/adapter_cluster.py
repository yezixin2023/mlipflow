"""Cluster-aware facade for bundled MLIP training.

Local execution remains delegated to adapter.py.  The original DeepMD fresh
training scheduler contract remains available for backward compatibility, while
the bundled scheduler contract extends ssh-slurm to train/fine-tune DeepMD,
M3GNet/MatGL, CHGNet, and MACE through one staged remote runner.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from mlipflow.science import model_runtime

HERE = Path(__file__).resolve().parent
LEGACY_PATH = HERE / "adapter.py"
CLUSTER_RUNNER = HERE / "training_cluster.py"
SHARED_MODEL_RUNTIME = Path(model_runtime.__file__).resolve()
BUNDLED_FILES = (
    "training_wrapper.py",
    "mlip_common.py",
    "mlip_deepmd.py",
    "mlip_m3gnet.py",
    "mlip_chgnet.py",
    "mlip_mace.py",
)
PLUGIN_ID = "mlip-training"
FRAMEWORKS = {"deepmd", "m3gnet", "chgnet", "mace"}
OPERATIONS = {"train", "finetune"}
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_MODEL_BYTES = 8 * 1024 * 1024 * 1024
MAX_LOG_BYTES = 128 * 1024 * 1024
HPC_RESOURCES = {"cpus", "gpus", "memory", "walltime"}
GENERIC_CONTRACT = "bundled-mlip-v1"


def _template_family(framework: str, *, publishes_model: bool) -> str:
    suffix = "-publish" if publishes_model else ""
    return f"mlip-{framework}{suffix}"


def _load_legacy():
    spec = importlib.util.spec_from_file_location("mlipflow_training_legacy", LEGACY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load legacy mlip-training adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEGACY = _load_legacy()


def _diag(level: str, code: str, message: str) -> dict[str, str]:
    return {"level": level, "code": code, "message": message}


def _errors(items: list[dict[str, str]]) -> bool:
    return any(str(item.get("level", "")).lower() == "error" for item in items)


def _blocked(items: list[dict[str, str]]) -> dict[str, Any]:
    return {"plugin_id": PLUGIN_ID, "status": "BLOCKED", "executable": False, "diagnostics": items}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _is_fingerprint(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(char in "0123456789abcdef" for char in value[7:])
    )


def _plain(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and "\x00" not in value and "\n" not in value


def _safe_relative(value: Any, *, plain_name: bool = False) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        return False
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return False
    return not plain_name or len(path.parts) == 1


def _project_file(root: Path) -> Path | None:
    for name in ("project.yaml", "project.yml", "project.json"):
        path = root / name
        if path.is_file() and not path.is_symlink():
            return path.resolve()
    return None


def _resolve_project_file(root: Path, value: Any) -> Path | None:
    if isinstance(value, dict):
        value = value.get("path")
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    candidate = candidate if candidate.is_absolute() else root / candidate
    if candidate.is_symlink() or not candidate.is_file():
        return None
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    return candidate.resolve()


def _json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"{path.name} exceeds the JSON size bound")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _artifact_reference(
    path: Path, *, id_key: str, default_kind: str
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        raw = _json(path)
        if raw.get("schema_version") != 1:
            raise ValueError("schema_version must be 1")
        artifact_id = raw.get(id_key)
        if not _plain(artifact_id) or "/" in str(artifact_id) or "\\" in str(artifact_id):
            raise ValueError(f"{id_key} must be a safe logical id")
        relative = raw.get("relative_path", artifact_id)
        if not _safe_relative(relative):
            raise ValueError("relative_path must stay below the site-owned artifact root")
        kind = raw.get("kind", default_kind)
        if kind not in {"file", "directory"}:
            raise ValueError("kind must be file or directory")
        fingerprint = raw.get("fingerprint")
        if not _is_fingerprint(fingerprint):
            raise ValueError("fingerprint must be sha256:<64 hex>")
        return {
            "id": str(artifact_id),
            "relative_path": str(relative),
            "kind": str(kind),
            "fingerprint": str(fingerprint),
        }, None
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return None, str(exc)


def _framework_dataset_kind(framework: str) -> str:
    return "directory" if framework == "deepmd" else "file"


def _framework_dataset_kinds(framework: str) -> set[str]:
    # DFT assembly emits split-preserving directories for every framework.
    # Legacy user-supplied M3GNet/CHGNet/MACE single files remain supported.
    return {"directory"} if framework == "deepmd" else {"file", "directory"}


def _framework_foundation_kind(framework: str) -> str:
    return "directory" if framework == "m3gnet" else "file"


def _legacy_scheduled(context: Mapping[str, Any]) -> bool:
    if context.get("backend") != "ssh-slurm":
        return False
    parameters = context.get("parameters", {})
    inputs = context.get("inputs", {})
    if not isinstance(parameters, Mapping) or not isinstance(inputs, Mapping):
        return False
    if parameters.get("framework") != "deepmd" or parameters.get("operation", "train") != "train":
        return False
    root = Path(str(context.get("project_root", ""))).expanduser().absolute()
    reference = _resolve_project_file(root, inputs.get("dataset_reference"))
    if reference is None:
        return True
    try:
        raw = _json(reference)
    except Exception:
        return True
    return "relative_path" not in raw and "kind" not in raw


def _generic_plan(context: Any) -> bool:
    if not isinstance(context, Mapping) or context.get("backend") != "ssh-slurm":
        return False
    execution = context.get("execution")
    plan = execution.get("plan") if isinstance(execution, Mapping) else None
    return isinstance(plan, Mapping) and plan.get("scheduler_contract") == GENERIC_CONTRACT


def _staged(path: Path, remote_name: str) -> dict[str, Any]:
    return {
        "source": str(path),
        "remote_name": remote_name,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "sensitive": False,
        "fetch_allowed": False,
    }


def _fetch(
    remote_name: str, remote_path: str, local_name: str, required: bool, maximum: int, role: str
) -> dict[str, Any]:
    return {
        "remote_name": remote_name,
        "remote_path": remote_path,
        "local_name": local_name,
        "required": required,
        "max_bytes": maximum,
        "role": role,
    }


def _validate_generic(context: Any) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    if not isinstance(context, Mapping):
        return [_diag("error", "context.type", "context must be an object")]
    root = Path(str(context.get("project_root", ""))).expanduser().absolute()
    if not root.is_dir():
        diagnostics.append(
            _diag("error", "context.project_root", "project_root must be an existing directory")
        )
        return diagnostics
    parameters = context.get("parameters", {})
    inputs = context.get("inputs", {})
    resources = context.get("resources", {})
    if not isinstance(parameters, Mapping) or not isinstance(inputs, Mapping):
        diagnostics.append(
            _diag("error", "context.mapping", "inputs and parameters must be objects")
        )
        return diagnostics
    if not isinstance(resources, Mapping) or set(resources) != HPC_RESOURCES:
        diagnostics.append(
            _diag(
                "error",
                "resource.hpc_contract",
                "ssh-slurm resources must be cpus, gpus, memory, walltime",
            )
        )

    framework = parameters.get("framework")
    operation = parameters.get("operation", "train")
    if framework not in FRAMEWORKS:
        diagnostics.append(
            _diag(
                "error", "training.framework", "framework must be deepmd, m3gnet, chgnet, or mace"
            )
        )
        return diagnostics
    if operation not in OPERATIONS:
        diagnostics.append(
            _diag("error", "training.operation", "operation must be train or finetune")
        )
    seed = parameters.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        diagnostics.append(_diag("error", "training.seed", "seed must be a non-negative integer"))
    for key in ("device", "precision"):
        if not _plain(parameters.get(key)):
            diagnostics.append(_diag("error", f"training.{key}", f"parameters.{key} is required"))
    if parameters.get("precision") not in {"float32", "float64"}:
        diagnostics.append(
            _diag("error", "training.precision_value", "precision must be float32 or float64")
        )
    if framework == "chgnet" and parameters.get("precision") != "float32":
        diagnostics.append(
            _diag(
                "error", "training.chgnet_precision", "CHGNet scheduled training requires float32"
            )
        )
    if not _is_fingerprint(parameters.get("config_fingerprint")):
        diagnostics.append(
            _diag(
                "error",
                "training.config_fingerprint",
                "parameters.config_fingerprint must be sha256:<64 hex>",
            )
        )
    declared_dataset_fingerprint = parameters.get("dataset_fingerprint")
    if declared_dataset_fingerprint is not None and not _is_fingerprint(
        declared_dataset_fingerprint
    ):
        diagnostics.append(
            _diag(
                "error",
                "training.dataset_fingerprint",
                "parameters.dataset_fingerprint must be sha256:<64 hex> when supplied",
            )
        )

    allowed_inputs = {"training_config", "dataset_reference", "foundation_model_reference"}
    unknown = sorted(set(inputs) - allowed_inputs)
    if unknown:
        diagnostics.append(
            _diag(
                "error",
                "training.scheduled_inputs",
                "scheduled bundled training does not accept: " + ", ".join(unknown),
            )
        )
    config = _resolve_project_file(root, inputs.get("training_config"))
    dataset_path = _resolve_project_file(root, inputs.get("dataset_reference"))
    if config is None or config.suffix.lower() != ".json":
        diagnostics.append(
            _diag(
                "error",
                "training.training_config",
                "training_config must be a project-scoped JSON file",
            )
        )
    else:
        try:
            _json(config)
            if _sha256(config) != parameters.get("config_fingerprint"):
                diagnostics.append(
                    _diag(
                        "error",
                        "training.config_fingerprint_mismatch",
                        "config fingerprint differs from parameters.config_fingerprint",
                    )
                )
        except Exception as exc:
            diagnostics.append(_diag("error", "training.config_unreadable", str(exc)))
    if dataset_path is None:
        diagnostics.append(
            _diag("error", "training.dataset_reference", "dataset_reference is required")
        )
    else:
        dataset, error = _artifact_reference(
            dataset_path, id_key="dataset_id", default_kind=_framework_dataset_kind(str(framework))
        )
        if error is not None or dataset is None:
            diagnostics.append(
                _diag("error", "training.dataset_contract", error or "invalid dataset reference")
            )
        else:
            if dataset["kind"] not in _framework_dataset_kinds(str(framework)):
                diagnostics.append(
                    _diag(
                        "error",
                        "training.dataset_kind",
                        f"{framework} scheduled data kind must be one of "
                        + ", ".join(sorted(_framework_dataset_kinds(str(framework)))),
                    )
                )
            if (
                declared_dataset_fingerprint is not None
                and dataset["fingerprint"] != declared_dataset_fingerprint
            ):
                diagnostics.append(
                    _diag(
                        "error",
                        "training.dataset_fingerprint_mismatch",
                        "dataset reference differs from parameters.dataset_fingerprint",
                    )
                )

    foundation_path = _resolve_project_file(root, inputs.get("foundation_model_reference"))
    if operation == "finetune":
        if foundation_path is None:
            diagnostics.append(
                _diag(
                    "error",
                    "training.foundation_reference",
                    "finetune requires foundation_model_reference",
                )
            )
        else:
            foundation, error = _artifact_reference(
                foundation_path,
                id_key="model_id",
                default_kind=_framework_foundation_kind(str(framework)),
            )
            if error is not None or foundation is None:
                diagnostics.append(
                    _diag(
                        "error",
                        "training.foundation_contract",
                        error or "invalid foundation model reference",
                    )
                )
            else:
                if foundation["kind"] != _framework_foundation_kind(str(framework)):
                    diagnostics.append(
                        _diag(
                            "error",
                            "training.foundation_kind",
                            f"{framework} foundation model must be a {_framework_foundation_kind(str(framework))}",
                        )
                    )
                if foundation["fingerprint"] != parameters.get("foundation_model_fingerprint"):
                    diagnostics.append(
                        _diag(
                            "error",
                            "training.foundation_fingerprint_mismatch",
                            "foundation reference differs from parameters.foundation_model_fingerprint",
                        )
                    )
        if not _is_fingerprint(parameters.get("foundation_model_fingerprint")):
            diagnostics.append(
                _diag(
                    "error",
                    "training.foundation_model_fingerprint",
                    "finetune requires foundation_model_fingerprint",
                )
            )
    elif foundation_path is not None or parameters.get("foundation_model_fingerprint") is not None:
        diagnostics.append(
            _diag(
                "error",
                "training.foundation_train",
                "foundation model fields are valid only for finetune",
            )
        )

    publish_id = parameters.get("publish_model_id")
    publish_relative = parameters.get("publish_model_relative_path")
    if (publish_id is None) != (publish_relative is None):
        diagnostics.append(
            _diag(
                "error",
                "training.publish_pair",
                "publish_model_id and publish_model_relative_path must be supplied together",
            )
        )
    elif publish_id is not None:
        if (
            not _plain(publish_id)
            or "/" in str(publish_id)
            or "\\" in str(publish_id)
        ):
            diagnostics.append(
                _diag("error", "training.publish_model_id", "publish_model_id is invalid")
            )
        if not _safe_relative(publish_relative):
            diagnostics.append(
                _diag(
                    "error",
                    "training.publish_model_relative_path",
                    "publish_model_relative_path must stay below the site model root",
                )
            )

    result_name = parameters.get("result_manifest", "mlip-training-result.json")
    if not _safe_relative(result_name, plain_name=True):
        diagnostics.append(
            _diag(
                "error",
                "training.result_manifest",
                "result_manifest must be a plain relative file name",
            )
        )
    if _project_file(root) is None:
        diagnostics.append(
            _diag(
                "error",
                "training.project_file",
                "scheduled training requires project.yaml/project.yml/project.json",
            )
        )
    for name in (CLUSTER_RUNNER.name, *BUNDLED_FILES):
        if not (HERE / name).is_file():
            diagnostics.append(
                _diag("error", "training.bundled_file", f"missing bundled cluster file: {name}")
            )
    if not SHARED_MODEL_RUNTIME.is_file():
        diagnostics.append(
            _diag(
                "error",
                "training.bundled_file",
                "missing shared model runtime: model_runtime.py",
            )
        )
    return diagnostics


def _plan_generic(context: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = _validate_generic(context)
    if _errors(diagnostics):
        return _blocked(diagnostics)
    root = Path(str(context["project_root"])).expanduser().absolute().resolve()
    inputs = dict(context.get("inputs", {}))
    parameters = dict(context.get("parameters", {}))
    framework = str(parameters["framework"])
    operation = str(parameters.get("operation", "train"))
    project = _project_file(root)
    config = _resolve_project_file(root, inputs["training_config"])
    dataset_path = _resolve_project_file(root, inputs["dataset_reference"])
    assert project is not None and config is not None and dataset_path is not None
    dataset, _ = _artifact_reference(
        dataset_path, id_key="dataset_id", default_kind=_framework_dataset_kind(framework)
    )
    assert dataset is not None

    staged = [
        _staged(project, "project.yaml"),
        _staged(config, "training-config.json"),
        _staged(dataset_path, "dataset-reference.json"),
        _staged(CLUSTER_RUNNER, "training_cluster.py"),
    ]
    for name in BUNDLED_FILES:
        staged.append(_staged(HERE / name, name))
    staged.append(_staged(SHARED_MODEL_RUNTIME, "model_runtime.py"))
    foundation: dict[str, Any] | None = None
    if operation == "finetune":
        source = _resolve_project_file(root, inputs["foundation_model_reference"])
        assert source is not None
        foundation, _ = _artifact_reference(
            source, id_key="model_id", default_kind=_framework_foundation_kind(framework)
        )
        assert foundation is not None
        staged.append(_staged(source, "foundation-model-reference.json"))

    result_name = str(parameters.get("result_manifest", "mlip-training-result.json"))
    fetch_outputs = [
        _fetch(
            "cluster-run-report.json",
            "output/cluster-run-report.json",
            "cluster-run-report.json",
            True,
            MAX_JSON_BYTES,
            "training-cluster-report",
        ),
        _fetch(
            "training-result.json",
            "output/training-result.json",
            result_name,
            True,
            MAX_JSON_BYTES,
            "training-manifest",
        ),
        _fetch(
            "model-artifact",
            "output/model-artifact",
            "model-artifact",
            True,
            MAX_MODEL_BYTES,
            "model",
        ),
        _fetch(
            "training.stdout.log",
            "output/training.stdout.log",
            "training.stdout.log",
            False,
            MAX_LOG_BYTES,
            "training-log",
        ),
        _fetch(
            "training.stderr.log",
            "output/training.stderr.log",
            "training.stderr.log",
            False,
            MAX_LOG_BYTES,
            "training-log",
        ),
    ]
    publish_model: dict[str, Any] | None = None
    if parameters.get("publish_model_id") is not None:
        publish_model = {
            "model_id": str(parameters["publish_model_id"]),
            "framework": framework,
            "relative_path": str(parameters["publish_model_relative_path"]),
            "kind": _framework_foundation_kind(framework),
        }
        fetch_outputs.append(
            _fetch(
                "model-reference.json",
                "output/model-reference.json",
                "model-reference.json",
                True,
                MAX_JSON_BYTES,
                "model-reference",
            )
        )
    identity = {
        "framework": framework,
        "operation": operation,
        "seed": parameters["seed"],
        "device": parameters["device"],
        "precision": parameters["precision"],
        "config_fingerprint": parameters["config_fingerprint"],
        "dataset": dataset,
        "foundation_model": foundation,
        "publish_model": publish_model,
    }
    template_family = _template_family(
        framework, publishes_model=publish_model is not None
    )
    return {
        "plugin_id": PLUGIN_ID,
        "status": "READY",
        "executable": True,
        "scheduler_contract": GENERIC_CONTRACT,
        "framework": framework,
        "operation": operation,
        "argv": [f"template-family:{template_family}"],
        "cwd": "remote-attempt-workspace",
        "expected_outputs": [item["remote_name"] for item in fetch_outputs if item["required"]],
        "training_identity": identity,
        "approval_summary": {
            "expensive": True,
            "submits_jobs": True,
            "execution_model": "single-python",
            "cpus_meaning": "threads-per-process",
            "framework": framework,
            "operation": operation,
            "fine_tune": operation == "finetune",
            "dataset_id": dataset["id"],
            "dataset_fingerprint": dataset["fingerprint"],
            "foundation_model_id": foundation["id"] if foundation else None,
            "publish_model": publish_model,
            "seed": parameters["seed"],
            "device": parameters["device"],
            "precision": parameters["precision"],
            "fetch_allowlist": sorted(item["remote_name"] for item in fetch_outputs),
        },
        "input_fingerprints": {
            "training_config": _sha256(config),
            "dataset_reference": _sha256(dataset_path),
            **({"foundation_model_reference": _sha256(source)} if operation == "finetune" else {}),
        },
        "scheduled_execution": {
            "schema_version": 3,
            "execution_model": "single-python",
            "template_family": template_family,
            "staged_files": staged,
            "fetch_outputs": fetch_outputs,
        },
        "diagnostics": [],
    }


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
            raise ValueError(f"missing, unsafe, or oversized JSON file: {path.name}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("top-level JSON value must be an object")
        return value, None
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return None, str(exc)


def _mace_completion_diagnostics(
    context: Mapping[str, Any], result: Mapping[str, Any]
) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    root = Path(str(context.get("project_root", ""))).expanduser().absolute()
    inputs = context.get("inputs", {})
    config_path = (
        _resolve_project_file(root, inputs.get("training_config"))
        if isinstance(inputs, Mapping)
        else None
    )
    requested_epochs: int | None = None
    if config_path is not None:
        try:
            raw = _json(config_path)
            mace = raw.get("mace")
            options = mace.get("options") if isinstance(mace, Mapping) else None
            value = options.get("max_num_epochs") if isinstance(options, Mapping) else None
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                requested_epochs = value
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            requested_epochs = None
    if requested_epochs is None:
        return [
            _diag(
                "error",
                "training.mace_requested_epochs",
                "approved MACE config must declare a positive max_num_epochs",
            )
        ]
    provenance = result.get("provenance")
    completion = provenance.get("completion") if isinstance(provenance, Mapping) else None
    if not isinstance(completion, Mapping):
        return [
            _diag(
                "error",
                "training.mace_completion",
                "MACE result lacks completion evidence",
            )
        ]
    expected = {
        "requested_epochs": requested_epochs,
        "completed_epochs": requested_epochs,
        "normal_completion": True,
        "all_recorded_metrics_finite": True,
        "native_model_reload": "OK",
    }
    for key, value in expected.items():
        if completion.get(key) != value:
            diagnostics.append(
                _diag(
                    "error",
                    f"training.mace_{key}",
                    f"MACE completion {key} differs from the approved requirement",
                )
            )
    environment = provenance.get("environment") if isinstance(provenance, Mapping) else None
    required_environment = {
        "compute_node",
        "python_version",
        "mace_source_version",
        "mace_dist_version",
        "torch_version",
        "torch_cuda_version",
        "gpu_model",
    }
    if not isinstance(environment, Mapping) or any(
        not _plain(environment.get(key)) for key in required_environment
    ):
        diagnostics.append(
            _diag(
                "error",
                "training.mace_environment",
                "MACE result lacks required compute/GPU environment evidence",
            )
        )
    return diagnostics


def _chgnet_completion_diagnostics(
    context: Mapping[str, Any], result: Mapping[str, Any]
) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    root = Path(str(context.get("project_root", ""))).expanduser().absolute()
    inputs = context.get("inputs", {})
    config_path = (
        _resolve_project_file(root, inputs.get("training_config"))
        if isinstance(inputs, Mapping)
        else None
    )
    requested_epochs: int | None = None
    expected_freeze: list[str] | None = None
    if config_path is not None:
        try:
            raw = _json(config_path)
            chgnet = raw.get("chgnet")
            trainer = chgnet.get("trainer") if isinstance(chgnet, Mapping) else None
            epochs = trainer.get("epochs") if isinstance(trainer, Mapping) else None
            starting_epoch = (
                trainer.get("starting_epoch", 0) if isinstance(trainer, Mapping) else None
            )
            if (
                isinstance(epochs, int)
                and not isinstance(epochs, bool)
                and isinstance(starting_epoch, int)
                and not isinstance(starting_epoch, bool)
                and starting_epoch >= 0
                and epochs > starting_epoch
            ):
                requested_epochs = epochs - starting_epoch
            freeze = chgnet.get("freeze_modules") if isinstance(chgnet, Mapping) else None
            if freeze is None:
                expected_freeze = []
            elif isinstance(freeze, list) and all(isinstance(item, str) for item in freeze):
                expected_freeze = list(freeze)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            requested_epochs = None
    if requested_epochs is None:
        return [
            _diag(
                "error",
                "training.chgnet_requested_epochs",
                "approved CHGNet config must declare trainer.epochs greater than starting_epoch",
            )
        ]
    provenance = result.get("provenance")
    completion = provenance.get("completion") if isinstance(provenance, Mapping) else None
    if not isinstance(completion, Mapping):
        return [
            _diag(
                "error",
                "training.chgnet_completion",
                "CHGNet result lacks completion evidence",
            )
        ]
    expected = {
        "requested_epochs": requested_epochs,
        "completed_epochs": requested_epochs,
        "normal_completion": True,
        "all_recorded_metrics_finite": True,
        "native_model_reload": "OK",
    }
    for key, value in expected.items():
        if completion.get(key) != value:
            diagnostics.append(
                _diag(
                    "error",
                    f"training.chgnet_{key}",
                    f"CHGNet completion {key} differs from the approved requirement",
                )
            )
    if expected_freeze is None or provenance.get("freeze_modules") != expected_freeze:
        diagnostics.append(
            _diag(
                "error",
                "training.chgnet_freeze_modules",
                "CHGNet frozen modules differ from the approved config",
            )
        )
    split = provenance.get("split") if isinstance(provenance, Mapping) else None
    generated_split = (
        isinstance(split, Mapping)
        and split.get("seed") == result.get("seed")
        and _is_fingerprint(split.get("train_indices_sha256"))
        and _is_fingerprint(split.get("val_indices_sha256"))
    )
    predefined_split = (
        isinstance(split, Mapping)
        and split.get("source") == "predefined"
        and _plain(split.get("split_id"))
        and all(_is_fingerprint(split.get(key)) for key in (
            "train_record_ids_sha256", "validation_record_ids_sha256", "test_record_ids_sha256"
        ))
    )
    if not generated_split and not predefined_split:
        diagnostics.append(
            _diag(
                "error",
                "training.chgnet_split",
                "CHGNet result lacks the approved seed and split identity evidence",
            )
        )
    environment = provenance.get("environment") if isinstance(provenance, Mapping) else None
    required_environment = {
        "compute_node",
        "python_version",
        "chgnet_version",
        "torch_version",
        "torch_cuda_version",
        "gpu_model",
    }
    if not isinstance(environment, Mapping) or any(
        not _plain(environment.get(key)) for key in required_environment
    ):
        diagnostics.append(
            _diag(
                "error",
                "training.chgnet_environment",
                "CHGNet result lacks required compute/runtime environment evidence",
            )
        )
    elif (
        str(result.get("device")).lower() in {"gpu", "cuda"}
        and environment.get("gpu_model") == "unavailable"
    ):
        diagnostics.append(
            _diag(
                "error",
                "training.chgnet_gpu",
                "CHGNet CUDA execution lacks an observed GPU model",
            )
        )
    return diagnostics


def _m3gnet_completion_diagnostics(
    context: Mapping[str, Any], result: Mapping[str, Any]
) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    root = Path(str(context.get("project_root", ""))).expanduser().absolute()
    inputs = context.get("inputs", {})
    config_path = (
        _resolve_project_file(root, inputs.get("training_config"))
        if isinstance(inputs, Mapping)
        else None
    )
    requested_epochs: int | None = None
    if config_path is not None:
        try:
            raw = _json(config_path)
            m3gnet = raw.get("m3gnet")
            trainer = m3gnet.get("trainer") if isinstance(m3gnet, Mapping) else None
            value = trainer.get("max_epochs") if isinstance(trainer, Mapping) else None
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                requested_epochs = value
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            requested_epochs = None
    if requested_epochs is None:
        return [
            _diag(
                "error",
                "training.m3gnet_requested_epochs",
                "approved M3GNet config must declare a positive trainer.max_epochs",
            )
        ]
    provenance = result.get("provenance")
    completion = provenance.get("completion") if isinstance(provenance, Mapping) else None
    if not isinstance(completion, Mapping):
        return [
            _diag(
                "error",
                "training.m3gnet_completion",
                "M3GNet result lacks completion evidence",
            )
        ]
    expected = {
        "requested_epochs": requested_epochs,
        "completed_epochs": requested_epochs,
        "normal_completion": True,
        "all_recorded_metrics_finite": True,
        "native_model_reload": "OK",
    }
    for key, value in expected.items():
        if completion.get(key) != value:
            diagnostics.append(
                _diag(
                    "error",
                    f"training.m3gnet_{key}",
                    f"M3GNet completion {key} differs from the approved requirement",
                )
            )
    split = provenance.get("split") if isinstance(provenance, Mapping) else None
    generated_split = (
        isinstance(split, Mapping)
        and split.get("seed") == result.get("seed")
        and _is_fingerprint(split.get("train_indices_sha256"))
        and _is_fingerprint(split.get("val_indices_sha256"))
        and (split.get("include_test") is not True or _is_fingerprint(split.get("test_indices_sha256")))
    )
    predefined_split = (
        isinstance(split, Mapping)
        and split.get("source") == "predefined"
        and _plain(split.get("split_id"))
        and all(_is_fingerprint(split.get(key)) for key in (
            "train_record_ids_sha256", "validation_record_ids_sha256", "test_record_ids_sha256"
        ))
    )
    if not generated_split and not predefined_split:
        diagnostics.append(
            _diag(
                "error",
                "training.m3gnet_split",
                "M3GNet result lacks the approved seed and split identity evidence",
            )
        )
    expected_refs = result.get("operation") == "finetune"
    if provenance.get("foundation_element_refs") is not expected_refs:
        diagnostics.append(
            _diag(
                "error",
                "training.m3gnet_foundation_refs",
                "M3GNet foundation element-reference evidence differs from the operation",
            )
        )
    environment = provenance.get("environment") if isinstance(provenance, Mapping) else None
    required_environment = {
        "compute_node",
        "python_version",
        "matgl_version",
        "lightning_version",
        "graph_backend",
        "graph_library_version",
        "torch_version",
        "torch_cuda_version",
        "gpu_model",
    }
    if not isinstance(environment, Mapping) or any(
        not _plain(environment.get(key)) for key in required_environment
    ):
        diagnostics.append(
            _diag(
                "error",
                "training.m3gnet_environment",
                "M3GNet result lacks required compute/runtime environment evidence",
            )
        )
    elif (
        str(result.get("device")).lower() in {"gpu", "cuda"}
        and environment.get("gpu_model") == "unavailable"
    ):
        diagnostics.append(
            _diag(
                "error",
                "training.m3gnet_gpu",
                "M3GNet CUDA execution lacks an observed GPU model",
            )
        )
    return diagnostics


def _check_generic(
    context: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    diagnostics: list[dict[str, str]] = []
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    execution = context.get("execution", {})
    plan = execution.get("plan") if isinstance(execution, Mapping) else None
    identity = plan.get("training_identity") if isinstance(plan, Mapping) else None
    parameters = context.get("parameters", {})
    if not isinstance(identity, Mapping) or not isinstance(parameters, Mapping):
        return [
            _diag("error", "training.plan_identity", "pinned bundled training identity is missing")
        ], None
    result_name = str(parameters.get("result_manifest", "mlip-training-result.json"))
    result, error = _read_json(attempt / result_name)
    if error is not None or result is None:
        return [
            _diag("error", "training.result_unreadable", error or "training result is missing")
        ], None
    report, report_error = _read_json(attempt / "cluster-run-report.json")
    if report_error is not None or report is None:
        return [
            _diag("error", "training.cluster_report", report_error or "cluster report is missing")
        ], None
    expected = {
        "plugin_id": PLUGIN_ID,
        "status": "OK",
        "framework": identity.get("framework"),
        "operation": identity.get("operation"),
        "seed": identity.get("seed"),
        "device": identity.get("device"),
        "precision": identity.get("precision"),
        "config_fingerprint": identity.get("config_fingerprint"),
        "dataset_fingerprint": (identity.get("dataset") or {}).get("fingerprint")
        if isinstance(identity.get("dataset"), Mapping)
        else None,
    }
    if result.get("schema_version") != 1:
        diagnostics.append(
            _diag("error", "training.result_schema", "training result schema_version must be 1")
        )
    for key, value in expected.items():
        if result.get(key) != value:
            diagnostics.append(
                _diag(
                    "error",
                    f"training.result_{key}",
                    f"training result {key} differs from the approved plan",
                )
            )
    if not _plain(result.get("framework_version")):
        diagnostics.append(
            _diag(
                "error",
                "training.framework_version",
                "training result must record framework_version",
            )
        )
    foundation = identity.get("foundation_model")
    if identity.get("operation") == "finetune":
        expected_foundation = (
            foundation.get("fingerprint") if isinstance(foundation, Mapping) else None
        )
        if result.get("foundation_model_fingerprint") != expected_foundation:
            diagnostics.append(
                _diag(
                    "error",
                    "training.foundation_result",
                    "foundation model fingerprint differs from the approved plan",
                )
            )
    published_reference: dict[str, Any] | None = None
    approved_publish = identity.get("publish_model")
    if isinstance(approved_publish, Mapping):
        reference, reference_error = _read_json(attempt / "model-reference.json")
        expected_reference = {
            "schema_version": 1,
            **dict(approved_publish),
        }
        if reference_error is not None or reference is None:
            diagnostics.append(
                _diag(
                    "error",
                    "training.model_reference",
                    reference_error or "published model reference is missing",
                )
            )
        elif (
            any(reference.get(key) != value for key, value in expected_reference.items())
            or not _is_fingerprint(reference.get("fingerprint"))
            or report.get("published_model") != reference
        ):
            diagnostics.append(
                _diag(
                    "error",
                    "training.model_publication",
                    "published model reference differs from the approved destination or cluster report",
                )
            )
        else:
            published_reference = reference
    elif report.get("published_model") is not None:
        diagnostics.append(
            _diag(
                "error",
                "training.unapproved_publication",
                "cluster report contains an unapproved model publication",
            )
        )
    model = result.get("model_artifact")
    model_path = attempt / "model-artifact"
    if not isinstance(model, Mapping) or model.get("path") != "model-artifact":
        diagnostics.append(
            _diag("error", "training.model_record", "training result must reference model-artifact")
        )
    elif model_path.is_symlink() or not model_path.is_file():
        diagnostics.append(
            _diag("error", "training.model_missing", "fetched model-artifact is missing or unsafe")
        )
    else:
        if model.get("size_bytes") != model_path.stat().st_size or model.get("sha256") != _sha256(
            model_path
        ):
            diagnostics.append(
                _diag(
                    "error",
                    "training.model_identity",
                    "fetched model identity differs from training-result.json",
                )
            )
    if (
        report.get("schema_version") != 1
        or report.get("status") != "OK"
        or report.get("return_code") != 0
    ):
        diagnostics.append(
            _diag(
                "error",
                "training.remote_failure",
                "cluster runner did not report successful completion",
            )
        )
    for key in ("framework", "operation", "config_fingerprint"):
        if report.get(key) != identity.get(key):
            diagnostics.append(
                _diag(
                    "error",
                    f"training.cluster_{key}",
                    f"cluster report {key} differs from the approved plan",
                )
            )
    dataset = report.get("dataset")
    approved_dataset = identity.get("dataset")
    if not isinstance(dataset, Mapping) or not isinstance(approved_dataset, Mapping):
        diagnostics.append(
            _diag("error", "training.cluster_dataset", "cluster report lacks dataset identity")
        )
    else:
        if dataset.get("id") != approved_dataset.get("id") or dataset.get(
            "observed_fingerprint"
        ) != approved_dataset.get("fingerprint"):
            diagnostics.append(
                _diag(
                    "error",
                    "training.cluster_dataset_identity",
                    "cluster dataset content differs from approved identity",
                )
            )
    if identity.get("operation") == "finetune":
        reported_foundation = report.get("foundation_model")
        approved_foundation = identity.get("foundation_model")
        if not isinstance(reported_foundation, Mapping) or not isinstance(
            approved_foundation, Mapping
        ):
            diagnostics.append(
                _diag(
                    "error",
                    "training.cluster_foundation",
                    "cluster report lacks foundation model identity",
                )
            )
        elif reported_foundation.get("id") != approved_foundation.get(
            "id"
        ) or reported_foundation.get("observed_fingerprint") != approved_foundation.get(
            "fingerprint"
        ):
            diagnostics.append(
                _diag(
                    "error",
                    "training.cluster_foundation_identity",
                    "cluster foundation model content differs from approved identity",
                )
            )
    metrics = result.get("metrics", {})
    if not isinstance(metrics, Mapping):
        diagnostics.append(_diag("error", "training.metrics", "metrics must be an object"))
        metrics = {}
    else:
        for name, value in metrics.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                diagnostics.append(
                    _diag("error", "training.metric_value", f"metric {name!r} must be finite")
                )
    if identity.get("framework") == "mace":
        diagnostics.extend(_mace_completion_diagnostics(context, result))
    if identity.get("framework") == "chgnet":
        diagnostics.extend(_chgnet_completion_diagnostics(context, result))
    if identity.get("framework") == "m3gnet":
        diagnostics.extend(_m3gnet_completion_diagnostics(context, result))
    if diagnostics:
        return diagnostics, None
    return [], {
        "result": result,
        "report": report,
        "model": model_path,
        "model_reference": published_reference,
        "metrics": dict(metrics),
    }


def _collect_generic(context: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics, analysis = _check_generic(context)
    if diagnostics or analysis is None:
        return {
            "plugin_id": PLUGIN_ID,
            "status": "FAIL",
            "diagnostics": diagnostics,
            "artifacts": [],
            "metrics": {},
        }
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    parameters = context.get("parameters", {})
    result_name = (
        str(parameters.get("result_manifest", "mlip-training-result.json"))
        if isinstance(parameters, Mapping)
        else "mlip-training-result.json"
    )
    artifacts = [
        {
            "path": str(attempt / result_name),
            "role": "training-manifest",
            "media_type": "application/json",
        },
        {
            "path": str(attempt / "model-artifact"),
            "role": "model",
            "media_type": analysis["result"]
            .get("model_artifact", {})
            .get("media_type", "application/octet-stream"),
        },
        {
            "path": str(attempt / "cluster-run-report.json"),
            "role": "training-cluster-report",
            "media_type": "application/json",
        },
    ]
    if analysis.get("model_reference") is not None:
        artifacts.append(
            {
                "path": str(attempt / "model-reference.json"),
                "role": "model-reference",
                "media_type": "application/json",
            }
        )
    for name in ("training.stdout.log", "training.stderr.log"):
        path = attempt / name
        if path.is_file() and not path.is_symlink():
            artifacts.append(
                {"path": str(path), "role": "training-log", "media_type": "text/plain"}
            )
    return {
        "plugin_id": PLUGIN_ID,
        "status": "OK",
        "diagnostics": [],
        "metrics": analysis["metrics"],
        "artifacts": artifacts,
    }


class Adapter:
    """Delegate local/legacy behavior and add the bundled scheduler contract."""

    def __init__(self) -> None:
        self.legacy = LEGACY.Adapter()

    def validate(self, context: dict[str, Any]) -> list[dict[str, str]]:
        if context.get("backend") != "ssh-slurm" or _legacy_scheduled(context):
            return self.legacy.validate(context)
        return _validate_generic(context)

    def plan(self, context: dict[str, Any]) -> dict[str, Any]:
        if context.get("backend") != "ssh-slurm" or _legacy_scheduled(context):
            plan = self.legacy.plan(context)
            if isinstance(plan, dict):
                plan = dict(plan)
                plan["delegated_adapter_sha256"] = _sha256(LEGACY_PATH)
            return plan
        return _plan_generic(context)

    def prepare(self, context: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
        if plan.get("scheduler_contract") != GENERIC_CONTRACT:
            return self.legacy.prepare(context, plan)
        if plan.get("status") != "READY":
            return _blocked([_diag("error", "training.plan_required", "a READY plan is required")])
        return {
            "plugin_id": PLUGIN_ID,
            "status": "READY",
            "executable": False,
            "prepared": False,
            "message": "Core scheduler staging owns all writes; the adapter only declares the approved files and bounded outputs.",
        }

    def check(self, context: dict[str, Any]) -> dict[str, Any]:
        if not _generic_plan(context):
            return self.legacy.check(context)
        diagnostics, analysis = _check_generic(context)
        if diagnostics or analysis is None:
            return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
        return {
            "plugin_id": PLUGIN_ID,
            "status": "OK",
            "diagnostics": [],
            "metrics": analysis["metrics"],
            "artifacts": [],
        }

    def collect(self, context: dict[str, Any]) -> dict[str, Any]:
        if not _generic_plan(context):
            return self.legacy.collect(context)
        return _collect_generic(context)

    def replay(self, context: dict[str, Any]) -> dict[str, Any]:
        if _generic_plan(context):
            return _collect_generic(context)
        return self.legacy.replay(context)
