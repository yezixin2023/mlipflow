"""lammps md: adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import execute, prepare
from .restart import (
    EXECUTE,
    PLUGIN_ID,
    _check_restart,
    _plan_execute,
    _proxy,
    _validate,
)


class Adapter:
    def operation(self, context: Any) -> str:
        return execute._operation(context)

    def validate(self, context: Any) -> list[dict[str, str]]:
        if not isinstance(context, dict) or execute._operation(context) != EXECUTE:
            return prepare.validate(context)
        return execute.validate(_proxy(context)) + _validate(context)

    def plan(self, context: Any) -> dict[str, Any]:
        if not isinstance(context, dict) or execute._operation(context) != EXECUTE:
            return prepare.plan(context)
        return _plan_execute(context)

    def check(self, context: Any) -> dict[str, Any]:
        if not isinstance(context, dict) or execute._operation(context) != EXECUTE:
            return prepare.check(context)
        checked = execute.check(_proxy(context))
        if checked.get("status") != "OK":
            return checked
        diagnostics = _check_restart(context)
        if diagnostics:
            return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
        calculation = prepare._mapping(execute._scheduled_plan(context).get("lammps_calculation"))
        result = execute._read_json(
            Path(str(context["attempt_dir"])) / "lammps-execution-result.json"
        )
        metrics = dict(prepare._mapping(checked.get("metrics")))
        metrics["segment_start_step"] = float(result.get("segment_start_step", 0))
        metrics["resumed"] = 1.0 if calculation.get("restart_from_attempt") is not None else 0.0
        checked["metrics"] = metrics
        return checked

    def collect(self, context: Any) -> dict[str, Any]:
        if not isinstance(context, dict) or execute._operation(context) != EXECUTE:
            return prepare.collect(context)
        collected = execute.collect(_proxy(context))
        calculation = prepare._mapping(execute._scheduled_plan(context).get("lammps_calculation"))
        result = execute._read_json(Path(str(context["attempt_dir"])) / "lammps-execution-result.json")
        collected["metrics"].update(
            segment_start_step=float(result.get("segment_start_step", 0)),
            resumed=1.0 if calculation.get("restart_from_attempt") is not None else 0.0,
        )
        return collected
