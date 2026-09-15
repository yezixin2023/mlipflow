"""ASE MD lifecycle with shared artifact collection and restart verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import npt, nvt, restart


class Adapter:
    def operation(self, context: dict[str, Any]) -> str:
        return "run"

    def validate(self, context: dict[str, Any]) -> list[dict[str, str]]:
        return restart._validate_restart(context)

    def plan(self, context: dict[str, Any]) -> dict[str, Any]:
        return restart._plan_restart(context)

    def check(self, context: dict[str, Any]) -> dict[str, Any]:
        checked = npt.check(context)
        if checked.get("status") != "OK":
            return checked
        diagnostics = restart._check_restart_metadata(context)
        if diagnostics:
            return {"plugin_id": nvt.PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
        return checked

    def collect(self, context: dict[str, Any]) -> dict[str, Any]:
        checked = self.check(context)
        if checked.get("status") != "OK":
            return checked
        attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
        roles = (
            ("md-result.json", "md-result", "application/json"),
            ("trajectory.traj", "trajectory", "application/octet-stream"),
            ("trajectory-index.json", "trajectory-index", "application/json"),
            ("thermo.csv", "thermodynamics", "text/csv"),
            ("final.extxyz", "final-structure", "chemical/x-xyz"),
            ("cluster-run-report.json", "cluster-run-report", "application/json"),
            ("md-checkpoint.json", "md-checkpoint", "application/json"),
        )
        return {
            "plugin_id": nvt.PLUGIN_ID,
            "status": "OK",
            "diagnostics": [],
            "metrics": checked.get("metrics", {}),
            "artifacts": [
                {"path": str(attempt / name), "role": role, "media_type": media}
                for name, role, media in roles
                if nvt._ordinary_file(attempt / name)
            ],
        }
