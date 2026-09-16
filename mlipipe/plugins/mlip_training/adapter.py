"""Dispatch the public training lifecycle by its explicit execution backend."""
from __future__ import annotations

from typing import Any
from . import local, scheduled


class Adapter:
    def operation(self, context: dict[str, Any]) -> str:
        return str(context.get("parameters", {}).get("operation", "train"))

    def validate(self, context: dict[str, Any]) -> list[dict[str, str]]:
        return (scheduled if context.get("backend") == "ssh-slurm" else local).validate(context)

    def plan(self, context: dict[str, Any]) -> dict[str, Any]:
        return (scheduled if context.get("backend") == "ssh-slurm" else local).plan(context)

    def check(self, context: dict[str, Any]) -> dict[str, Any]:
        return (scheduled if context.get("backend") == "ssh-slurm" else local).check(context)

    def collect(self, context: dict[str, Any]) -> dict[str, Any]:
        return (scheduled if context.get("backend") == "ssh-slurm" else local).collect(context)
