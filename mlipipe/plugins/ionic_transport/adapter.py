"""Public lifecycle for formal transport analysis and its MD handoff."""

from __future__ import annotations

from typing import Any, Mapping

from . import analysis, contracts


class Adapter:
    def operation(self, context: Any) -> str:
        parameters = (
            contracts._mapping(context.get("parameters")) if isinstance(context, Mapping) else {}
        )
        return contracts._operation_name(parameters)

    def validate(self, context: Any) -> list[dict[str, str]]:
        return analysis.validate(context)

    def plan(self, context: Any) -> dict[str, Any]:
        return analysis.plan(context)

    def check(self, context: Any) -> dict[str, Any]:
        return analysis.check(context)

    def collect(self, context: Any) -> dict[str, Any]:
        return analysis.collect(context)
