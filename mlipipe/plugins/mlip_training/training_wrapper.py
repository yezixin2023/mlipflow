"""Unified bundled trainer for DeepMD, M3GNet/MatGL, CHGNet, and MACE."""

from __future__ import annotations
import argparse
import importlib
import json
import sys
from pathlib import Path
if __package__:
    from mlipipe.plugins import model_runtime
else:
    try:
        import model_runtime
    except ModuleNotFoundError as exc:
        if exc.name != "model_runtime":
            raise
        # Direct source-script execution uses the shared file in its parent package.
        # Uploaded bundles instead provide model_runtime.py beside this script.
        import importlib.util

        runtime_path = Path(__file__).resolve().parent.parent / "model_runtime.py"
        runtime_spec = importlib.util.spec_from_file_location("model_runtime", runtime_path)
        if runtime_spec is None or runtime_spec.loader is None:
            raise ImportError(f"missing bundled model runtime: {runtime_path}") from exc
        model_runtime = importlib.util.module_from_spec(runtime_spec)
        sys.modules["model_runtime"] = model_runtime
        runtime_spec.loader.exec_module(model_runtime)
if __package__:
    from .mlip_common import (
        FRAMEWORKS,
        OPERATIONS,
        TrainingError,
        load_config,
        seed_all,
        write_result,
    )
else:
    from mlip_common import (
        FRAMEWORKS,
        OPERATIONS,
        TrainingError,
        load_config,
        seed_all,
        write_result,
    )

FRAMEWORK_MODULES = {
    "deepmd": "mlip_deepmd",
    "m3gnet": "mlip_m3gnet",
    "chgnet": "mlip_chgnet",
    "mace": "mlip_mace",
}


def build_parser():
    p = argparse.ArgumentParser(description="Train or fine-tune bundled MLIP families")
    p.add_argument("--framework", choices=FRAMEWORKS, required=True)
    p.add_argument("--operation", choices=OPERATIONS, required=True)
    for name in (
        "config",
        "data",
        "output",
        "result-manifest",
        "device",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--precision", choices=("float32", "float64"), required=True)
    p.add_argument("--foundation-model")
    p.add_argument("--dry-run", action="store_true")
    return p


def _paths(a):
    if a.seed < 0:
        raise TrainingError("--seed must be non-negative")
    config = Path(a.config).expanduser().absolute()
    data = Path(a.data).expanduser().absolute()
    output = Path(a.output).expanduser().absolute()
    result = Path(a.result_manifest).expanduser().absolute()
    if not config.is_file():
        raise TrainingError(f"config does not exist: {config}")
    if not data.exists():
        raise TrainingError(f"data does not exist: {data}")
    if output == result:
        raise TrainingError("output and result manifest must differ")
    if a.operation == "finetune":
        if not a.foundation_model:
            raise TrainingError("finetune requires foundation model")
        foundation = Path(a.foundation_model).expanduser().absolute()
        if not foundation.exists():
            raise TrainingError(f"foundation model does not exist: {foundation}")
        a.foundation_model = str(foundation)
    elif a.foundation_model:
        raise TrainingError("foundation arguments require finetune")
    a.config = str(config)
    a.data = str(data)
    a.output = str(output)
    a.result_manifest = str(result)
    return config, data


def _framework_version(name):
    return model_runtime.framework_version(name)


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        config_path, data_path = _paths(a)
        config = load_config(config_path)
        module_name = FRAMEWORK_MODULES[a.framework]
        backend = importlib.import_module(f".{module_name}", __package__) if __package__ else importlib.import_module(module_name)
        plan = (
            backend.plan(a, config, config_path, data_path)
            if a.framework == "mace"
            else backend.plan(a, config, data_path)
            if a.framework in {"m3gnet", "chgnet"}
            else backend.plan(a, config)
        )
        if a.dry_run:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.result_manifest).parent.mkdir(parents=True, exist_ok=True)
        seed_all(a.seed)
        media, metrics, provenance = backend.run(a, config, config_path, data_path)
        write_result(a, "OK", _framework_version(a.framework), metrics, media, provenance)
        return 0
    except Exception as exc:
        if not a.dry_run:
            try:
                write_result(
                    a,
                    "FAIL",
                    _framework_version(a.framework),
                    {},
                    "application/octet-stream",
                    {},
                    str(exc),
                )
            except Exception:
                pass
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
