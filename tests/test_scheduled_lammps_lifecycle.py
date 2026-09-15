from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from mlipflow.backends import ExecutionResult
from mlipflow.config import load_project
from mlipflow.services import advance, initialize, make_advance_plan, make_run_plan, run_node

from .helpers import project_config, write_json
from .test_scheduled_dft import FakeTemplateLibrary, write_site


RUN_TEMPLATE = """#!/usr/bin/env bash
# {{PROJECT_ID}} {{NODE_ID}} {{ATTEMPT}}
# {{RUN_DIR}} {{INPUT_DIR}} {{OUTPUT_DIR}} {{LOG_DIR}}
# {{CPUS}} {{GPUS}} {{MEMORY}} {{WALLTIME}}
exit 0
"""


def _library() -> FakeTemplateLibrary:
    templates = dict(FakeTemplateLibrary().templates)
    templates["lammps-mace-gpu/run.sh"] = RUN_TEMPLATE
    return FakeTemplateLibrary(templates)


def _prepared(root: Path) -> Path:
    prepared = root / "prepared"
    prepared.mkdir(parents=True)
    structure = prepared / "structure.data"
    structure.write_text("LAMMPS data\n\n1 atoms\n1 atom types\n", encoding="utf-8")
    deck = prepared / "in.gpu.lammps"
    marker = "MLIPFLOW_LAMMPS_COMPLETED step=1000"
    deck.write_text(
        "units metal\nread_data structure.data\npair_style mace no_domain_decomposition\n"
        "pair_coeff * * ${MODEL_FILE} Li\nrun 1000\nwrite_data final.data\n"
        f"write_restart final.restart\nprint \"{marker}\"\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "plugin_id": "lammps-md",
        "operation": "lammps-prepare",
        "status": "OK",
        "preparation_contract": "lammps-md-input-v2",
        "runtime_model_variable": "MODEL_FILE",
        "completion_marker": marker,
        "model": {
            "model_id": "mace-lmp-v1",
            "framework": "mace",
            "relative_path": "mace/model-lammps.pt",
            "kind": "file",
            "artifact_format": "mace-lammps-torchscript",
            "elements": ["Li"],
        },
        "md": {
            "ensemble": "nvt",
            "targets": ["gpu"],
            "type_map": ["Li"],
            "temperature_k": 900.0,
            "timestep_fs": 1.0,
            "steps": 1000,
            "thermo_interval": 10,
            "dump_interval": 10,
            "seed": 7,
            "thermostat_damping_fs": 100.0,
        },
        "generated_files": [
            {"name": "structure.data"},
            {"name": "in.gpu.lammps"},
        ],
        "launchers": [
            {
                "target": "gpu",
                "executable": "site-owned-lammps",
                "argv_after_executable": [
                    "-k", "on", "g", "1", "-sf", "kk", "-in", "in.gpu.lammps",
                    "-var", "MODEL_FILE", "<site-resolved-model-file>",
                ],
                "required_packages": ["ML-MACE", "KOKKOS"],
                "limitations": [],
            }
        ],
    }
    path = prepared / "lammps-input-manifest.json"
    write_json(path, manifest)
    return path


def _build_project(root: Path) -> Path:
    _prepared(root)
    node = {
        "id": "lammps-run",
        "uses": "lammps-md",
        "mode": "execute",
        "backend": "ssh-slurm",
        "backend_profile": "cluster-a",
        "inputs": {"lammps_input_manifest": "prepared/lammps-input-manifest.json"},
        "parameters": {"operation": "execute", "target": "gpu"},
        "resources": {"cpus": 8, "gpus": 1, "memory": "32G", "walltime": "01:00:00"},
    }
    write_json(root / "project.yaml", project_config([node]))
    return write_site(root)


class ScheduledLammpsLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.site = _build_project(self.root)
        self.remote = self.root / "fake-remote"
        initialize(self.root)
        self.project = load_project(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _inspect(self, _self, _cwd, remote_path):
        path = self.remote / remote_path
        if not path.is_file():
            return {"path": remote_path, "exists": False}
        return {
            "path": remote_path,
            "exists": True,
            "size_bytes": path.stat().st_size,
        }

    def _fetch(self, _self, _cwd, remote_path, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.remote / remote_path, destination)
        return destination

    def _write_remote_success(self, calculation: dict[str, Any]) -> None:
        output = self.remote / "output"
        output.mkdir(parents=True)
        files = {
            "trajectory.lammpstrj": b"ITEM: TIMESTEP\n1000\n",
            "final.data": b"final data\n",
            "final.restart": b"binary restart",
            "lammps.log": ("LAMMPS (4 Jul 2026)\n" + calculation["completion_marker"] + "\n").encode(),
            "lammps.screen.log": b"LAMMPS (4 Jul 2026)\n",
        }
        artifacts = []
        for name, payload in files.items():
            path = output / name
            path.write_bytes(payload)
            artifacts.append({"name": name, "path": name})
        result = {
            "schema_version": 1,
            "plugin_id": "lammps-md",
            "status": "OK",
            "operation": "execute",
            "framework": calculation["framework"],
            "target": calculation["target"],
            "lammps_version": "4 Jul 2026",
            "input_manifest_path": calculation["input_manifest_path"],
            "model": {
                "id": calculation["model_id"],
                "path": calculation["model_path"],
                "kind": calculation["model_kind"],
                "artifact_format": calculation["artifact_format"],
                "lammps_interface": calculation["lammps_interface"],
            },
            "ensemble": calculation["ensemble"],
            "steps_requested": calculation["steps"],
            "steps_completed": calculation["steps"],
            "segment_start_step": 0,
            "completion_marker": calculation["completion_marker"],
            "launcher": {
                "site_launcher_used": True,
                "prepared_argv_after_executable": calculation["prepared_launcher"]["argv_after_executable"],
                "required_packages": calculation["prepared_launcher"].get("required_packages", []),
            },
            "restart": {
                "policy": "disabled",
                "checkpoint_interval": None,
                "resumed": False,
                "from_attempt": None,
                "selected_checkpoint": None,
                "runtime_compatibility_checked": False,
                "bitwise_exact_guaranteed": False,
            },
            "artifacts": artifacts,
        }
        write_json(output / "lammps-execution-result.json", result)
        write_json(
            output / "cluster-run-report.json",
            {
                "schema_version": 1,
                "status": "OK",
                "framework": calculation["framework"],
                "target": calculation["target"],
                "model_id": calculation["model_id"],
                "model_path": calculation["model_path"],
                "model_kind": calculation["model_kind"],
                "lammps_interface": calculation["lammps_interface"],
                "input_manifest_path": calculation["input_manifest_path"],
                "steps_completed": calculation["steps"],
                "segment_start_step": 0,
                "restart_from_attempt": None,
                "selected_checkpoint": None,
                "lammps_version": "4 Jul 2026",
            },
        )
        write_json(
            self.remote / "completion.json",
            {
                "schema_version": 1,
                "status": "COMPLETED",
                "exit_code": 0,
                "project_id": self.project.project_id,
                "node_id": "lammps-run",
                "attempt": 1,
            },
        )

    def test_submit_completed_fetch_and_check(self) -> None:
        plan = make_run_plan(self.project, "lammps-run", self.site, _library())
        self.assertEqual(plan["adapter_plan"]["status"], "READY")
        calculation = plan["adapter_plan"]["lammps_calculation"]

        def stage(remote_dir, files):
            return remote_dir

        with patch("mlipflow.backends.SshSlurmBackend.stage_workspace", side_effect=stage), patch(
            "mlipflow.backends.SshSlurmBackend.submit",
            return_value=ExecutionResult(0, "Submitted batch job 91\n", "", "91"),
        ):
            run_node(
                self.project,
                "lammps-run",
                True,
                self.site,
                _library(),
            )
        self._write_remote_success(calculation)
        attempt = self.root / ".mlipflow" / "runs" / "lammps-run" / "attempt-1"
        self.assertFalse((attempt / "lammps-execution-result.json").exists())

        with patch(
            "mlipflow.backends.SshSlurmBackend.status",
            return_value={"state": "COMPLETED", "detail": None, "source": "fake"},
        ), patch(
            "mlipflow.backends.SshSlurmBackend.inspect_file",
            autospec=True,
            side_effect=self._inspect,
        ):
            approved = make_advance_plan(self.project)
        self.assertEqual(approved["details"]["transitions"][0]["action"], "adapter-finalize")
        self.assertFalse((attempt / "lammps-execution-result.json").exists())

        with patch(
            "mlipflow.backends.SshSlurmBackend.status",
            return_value={"state": "COMPLETED", "detail": None, "source": "fake"},
        ), patch(
            "mlipflow.backends.SshSlurmBackend.inspect_file",
            autospec=True,
            side_effect=self._inspect,
        ), patch(
            "mlipflow.backends.SshSlurmBackend.fetch_from",
            autospec=True,
            side_effect=self._fetch,
        ):
            outcome = advance(self.project)

        self.assertEqual(outcome["changed"][0]["state"], "OK", outcome)
        self.assertTrue((attempt / "lammps-execution-result.json").is_file())
        self.assertTrue((attempt / "final.restart").is_file())


if __name__ == "__main__":
    unittest.main()
