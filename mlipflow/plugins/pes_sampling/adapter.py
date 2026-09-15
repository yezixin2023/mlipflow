"""PES lifecycle dispatch to the module that owns each operation."""

from __future__ import annotations

from typing import Any, Mapping

from . import contracts, direct, lasp_input, merge, scheduled, ssw

OPERATIONS = {
    contracts.DIRECT_OPERATION: direct,
    contracts.LASP_INPUT_OPERATION: lasp_input,
    contracts.MERGE_OPERATION: merge,
    **{name: ssw for name in contracts.LASP_OPERATIONS},
}


def _operation_module(context: Any, *, results: bool = False):
    if isinstance(context, Mapping):
        if results:
            plan = contracts._mapping(contracts._mapping(context.get("execution")).get("plan"))
            if (
                isinstance(plan.get("scheduled_execution"), Mapping)
                and plan.get("operation") == "lasp-ssw-execute"
            ):
                return scheduled
        elif context.get("backend") == "ssh-slurm":
            return scheduled
    return OPERATIONS.get(contracts._operation(context))


class Adapter:
    def operation(self, context: Any) -> str:
        return contracts._operation(context)

    def validate(self, context: Any) -> list[dict[str, str]]:
        module = _operation_module(context)
        if module is not None:
            return module.validate(context)
        return [
            contracts._diagnostic(
                "ERROR",
                "operation.unsupported",
                "operation must be one of: %s" % ", ".join(sorted(OPERATIONS)),
            )
        ]

    def plan(self, context: Any) -> dict[str, Any]:
        module = _operation_module(context)
        return (
            module.plan(context)
            if module is not None
            else contracts._blocked(self.validate(context))
        )

    def check(self, context: Any) -> dict[str, Any]:
        module = _operation_module(context, results=True)
        if module is not None:
            return module.check(context)
        return {
            "plugin_id": contracts.PLUGIN_ID,
            "status": "FAIL",
            "diagnostics": self.validate(context),
        }

    def collect(self, context: Any) -> dict[str, Any]:
        module = _operation_module(context, results=True)
        if module is not None:
            return module.collect(context)
        return {
            "plugin_id": contracts.PLUGIN_ID,
            "operation": self.operation(context),
            "status": "FAIL",
            "artifacts": [],
            "metrics": {},
            "diagnostics": self.validate(context),
        }
