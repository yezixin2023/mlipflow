from __future__ import annotations

import hashlib
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
from .test_scheduled_dft import PLUGINS, FakeTemplateLibrary, write_site


RUN_TEMPLATE = """#!/usr/bin/env bash
# {{PROJECT_ID}} {{NODE_ID}} {{ATTEMPT}}
# {{RUN_DIR}} {{INPUT_DIR}} {{OUTPUT_DIR}} {{LOG_DIR}}
# {{CPUS}} {{GPUS}} {{MEMORY}} {{WALLTIME}}
exit 0
"""


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


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
            "fingerprint": "sha256:" + "2" * 64,
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
            {"name": "structure.data", "sha256": _sha(structure), "size_bytes": structure.stat().st_size},
            {"name": "in.gpu.lammps", "sha256": _sha(deck), "size_bytes": deck.stat().st_size},
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
    manifest = _prepared(root)
    node = {
        "id": "lammps-run",
        "uses": "lammps-md@0",
        "mode": "execute",
        "backend": "ssh-slurm",
        "backend_profile": "cluster-a",
        "inputs": {"lammps_input_manifest": "prepared/lammps-input-manifest.json"},
        "parameters": {
            "operation": "execute",
            "target": "gpu",
            "input_manifest_fingerprint": _sha(manifest),
        },
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
            "sha256": _sha(path),
        }

    def _fetch(self, _self, _cwd, remote_path, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.remote / remote_path, destination)
        return destination

    def _write_remote_success(self, identity: dict[str, Any]) -> None:
        output = self.remote / "output"
        output.mkdir(parents=True)
        files = {
            "trajectory.lammpstrj": b"ITEM: TIMESTEP\n1000\n",
            "final.data": b"final data\n",
            "final.restart": b"binary restart",
            "lammps.log": ("LAMMPS (4 Jul 2026)\n" + identity["completion_marker"] + "\n").encode(),
            "lammps.screen.log": b"LAMMPS (4 Jul 2026)\n",
        }
        artifacts = []
        for name, payload in files.items():
            path = output / name
            path.write_bytes(payload)
            artifacts.append({"name": name, "path": name, "sha256": _sha(path), "size_bytes": path.stat().st_size})
        result = {
            "schema_version": 1,
            "plugin_id": "lammps-md",
            "status": "OK",
            "operation": "execute",
            "framework": identity["framework"],
            "target": identity["target"],
            "lammps_version": "4 Jul 2026",
            "input_manifest_fingerprint": identity["input_manifest_fingerprint"],
            "model": {
                "id": identity["model_id"],
                "fingerprint": identity["model_fingerprint"],
                "kind": identity["model_kind"],
                "artifact_format": identity["artifact_format"],
                "lammps_interface": identity["lammps_interface"],
            },
            "ensemble": identity["ensemble"],
            "steps_requested": identity["steps"],
            "steps_completed": identity["steps"],
            "segment_start_step": 0,
            "completion_marker": identity["completion_marker"],
            "launcher": {
                "site_launcher_used": True,
                "prepared_argv_after_executable": identity["prepared_launcher"]["argv_after_executable"],
                "required_packages": identity["prepared_launcher"].get("required_packages", []),
            },
            "restart": {
                "policy": "disabled",
                "checkpoint_interval": None,
                "resumed": False,
                "from_attempt": None,
                "selected_checkpoint_sha256": None,
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
                "framework": identity["framework"],
                "target": identity["target"],
                "model_id": identity["model_id"],
                "model_fingerprint": identity["model_fingerprint"],
                "model_kind": identity["model_kind"],
                "lammps_interface": identity["lammps_interface"],
                "input_manifest_fingerprint": identity["input_manifest_fingerprint"],
                "steps_completed": identity["steps"],
                "segment_start_step": 0,
                "restart_from_attempt": None,
                "selected_checkpoint_sha256": None,
                "lammps_version": "4 Jul 2026",
                "result_sha256": _sha(output / "lammps-execution-result.json"),
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
        plan = make_run_plan(self.project, "lammps-run", PLUGINS, self.site, _library())
        self.assertEqual(plan["adapter_plan"]["status"], "READY")
        identity = plan["adapter_plan"]["lammps_execution_identity"]

        def stage(remote_dir, files):
            return remote_dir

        with patch("mlipflow.services.SshSlurmBackend.stage_workspace", side_effect=stage), patch(
            "mlipflow.services.SshSlurmBackend.submit",
            return_value=ExecutionResult(0, "Submitted batch job 91\n", "", "91"),
        ):
            run_node(
                self.project,
                "lammps-run",
                PLUGINS,
                plan["plan_digest"],
                self.site,
                _library(),
            )
        self._write_remote_success(identity)
        attempt = self.root / ".mlipflow" / "runs" / "lammps-run" / "attempt-1"
        self.assertFalse((attempt / "lammps-execution-result.json").exists())

        with patch(
            "mlipflow.services.SshSlurmBackend.status",
            return_value={"state": "COMPLETED", "detail": None, "source": "fake"},
        ), patch(
            "mlipflow.services.SshSlurmBackend.inspect_file",
            autospec=True,
            side_effect=self._inspect,
        ):
            approved = make_advance_plan(self.project, PLUGINS)
        self.assertEqual(approved["details"]["transitions"][0]["action"], "adapter-finalize")
        self.assertFalse((attempt / "lammps-execution-result.json").exists())

        with patch(
            "mlipflow.services.SshSlurmBackend.status",
            return_value={"state": "COMPLETED", "detail": None, "source": "fake"},
        ), patch(
            "mlipflow.services.SshSlurmBackend.inspect_file",
            autospec=True,
            side_effect=self._inspect,
        ), patch(
            "mlipflow.services.SshSlurmBackend.fetch_from",
            autospec=True,
            side_effect=self._fetch,
        ):
            assert "plan_digest" not in approved
            outcome = advance(self.project, PLUGINS)

        self.assertEqual(outcome["changed"][0]["state"], "OK", outcome)
        self.assertTrue((attempt / "lammps-execution-result.json").is_file())
        self.assertTrue((attempt / "final.restart").is_file())


if __name__ == "__main__":
    unittest.main()
