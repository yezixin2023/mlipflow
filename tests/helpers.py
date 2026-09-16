from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any, Iterable

from mlipipe.cli import main


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def project_config(nodes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "project": {"id": "test-project", "name": "Test", "description": "fixture"},
        "model_registry": "model_registry.yaml",
        "workflow": {"nodes": nodes or []},
        "routing": {
            "policies": {
                "ionic-transport": {
                    "metrics": [
                        {"name": "diffusion_mae", "direction": "minimize", "weight": 1.0}
                    ]
                }
            }
        },
    }


def run_cli(argv: Iterable[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = main(list(argv))
    return code, stdout.getvalue(), stderr.getvalue()


def snapshot(root: Path) -> dict[str, bytes | None]:
    result: dict[str, bytes | None] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        result[relative] = path.read_bytes() if path.is_file() else None
    return result


def load_module(path: Path, name: str = "test_script"):
    """Import installed source normally; load isolated script fixtures by path."""
    import importlib
    import importlib.util
    import sys

    package = Path(__file__).resolve().parents[1] / "mlipipe"
    try:
        relative = path.resolve().relative_to(package)
    except ValueError:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load test script: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module("mlipipe." + ".".join(relative.with_suffix("").parts))
