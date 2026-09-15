"""mlip training: adapter."""

from __future__ import annotations

from typing import Any, Mapping

from . import local
from .scheduled import (
    PLUGIN_ID,
    _check_generic,
    _collect_generic,
    _generic_plan,
    _legacy_scheduled,
    _plan_generic,
    _validate_generic,
)


class Adapter:
    """Delegate local/legacy behavior and add the bundled scheduler contract."""

    def operation(self, context: dict[str, Any]) -> str:
        parameters = context.get("parameters", {}) if isinstance(context, Mapping) else {}
        return (
            str(parameters.get("operation", "train"))
            if isinstance(parameters, Mapping)
            else "train"
        )

    def validate(self, context: dict[str, Any]) -> list[dict[str, str]]:
        if context.get("backend") != "ssh-slurm" or _legacy_scheduled(context):
            return local.validate(context)
        return _validate_generic(context)

    def plan(self, context: dict[str, Any]) -> dict[str, Any]:
        if context.get("backend") != "ssh-slurm" or _legacy_scheduled(context):
            return local.plan(context)
        return _plan_generic(context)

    def check(self, context: dict[str, Any]) -> dict[str, Any]:
        if not _generic_plan(context):
            return local.check(context)
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
            return local.collect(context)
        return _collect_generic(context)
