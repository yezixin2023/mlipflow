"""ASE MD lifecycle with shared artifact collection and restart verification."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from . import common, npt, nvt, restart


class Adapter:
    def operation(self, context: dict[str, Any]) -> str:
        return "run"

    def validate(self, context: dict[str, Any]) -> list[dict[str, str]]:
        ensemble = common._mapping(common._mapping(context).get("parameters")).get("ensemble", nvt.NVT)
        diagnostics = (npt if ensemble == npt.NPT else nvt).validate(context)
        return diagnostics + restart.validate(context)

    def plan(self, context: dict[str, Any]) -> dict[str, Any]:
        ensemble = common._mapping(common._mapping(context).get("parameters")).get("ensemble", nvt.NVT)
        plan = (npt if ensemble == npt.NPT else nvt).plan(context)
        return restart.apply_restart(context, plan)

    def check(self, context: dict[str, Any]) -> dict[str, Any]:
        settings = common._mapping(common._scheduled_plan(context).get("md_parameters"))
        ensemble = settings.get("ensemble", common._mapping(context.get("parameters")).get("ensemble", nvt.NVT))
        checked = (npt if ensemble == npt.NPT else nvt).check(context)
        if checked.get("status") != "OK":
            return checked
        diagnostics = restart.check_restart_metadata(context)
        if diagnostics:
            return {"plugin_id": common.PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
        return checked

    def collect(self, context: dict[str, Any]) -> dict[str, Any]:
        attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
        result = common._read_json(attempt / "md-result.json")
        final = {}
        with (attempt / "thermo.csv").open(encoding="utf-8", newline="") as stream:
            for final in csv.DictReader(stream):
                pass
        metrics = {
            "steps_completed": float(result["steps_completed"]),
            "final_temperature_K": float(final["temperature_K"]),
            "final_total_energy_eV": float(final["total_energy_eV"]),
        }
        if result["ensemble"] == npt.NPT:
            metrics.update(final_pressure_GPa=float(final["pressure_GPa"]),
                           final_volume_A3=float(final["volume_A3"]))
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
            "plugin_id": common.PLUGIN_ID,
            "status": "OK",
            "diagnostics": [],
            "metrics": metrics,
            "artifacts": [
                {"path": str(attempt / name), "role": role, "media_type": media}
                for name, role, media in roles
                if common._ordinary_file(attempt / name)
            ],
        }
