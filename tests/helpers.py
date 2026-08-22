from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any, Iterable

from mlipflow.cli import main


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plugin_manifest(plugin_id: str = "demo") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": plugin_id,
        "version": "1.0.0",
        "api_version": 1,
        "description": "Test-only replay plugin",
        "implementation": {"kind": "replay-only", "entrypoint": "adapter.py:Adapter"},
        "input_schema": {"type": "object"},
        "parameter_schema": {"type": "object"},
        "output_schema": {"type": "object"},
        "metrics_schema": {"type": "object"},
        "dependencies": {"python": [], "executables": []},
        "execution": {"backends": ["local"], "cost_class": "low"},
        "completion": {"requires_scheduler_success": False},
    }


def project_config(nodes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "project": {"id": "test-project", "name": "Test", "description": "fixture"},
        "locations": {},
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
        "safety": {"auto_submit": False},
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
