from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

from mlipflow.backends import ExecutionResult
from mlipflow.config import load_project
from mlipflow.services import advance, initialize, make_advance_plan, make_run_plan, run_node

from .helpers import project_config, write_json


ROOT = Path(__file__).resolve().parents[1]
PLUGINS = ROOT / "plugins"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def prepared_fixture(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = root / "inputs"
    prepared = root / "prepared"
    inputs.mkdir(parents=True)
    prepared.mkdir()
    source = inputs / "li.vasp"
    source.write_text(
        "Li\n1.0\n3 0 0\n0 3 0\n0 0 3\nLi\n1\nDirect\n0 0 0\n",
        encoding="utf-8",
    )
    structures = inputs / "structures.json"
    write_json(
        structures,
        {
            "schema_version": 1,
            "structures": [{"id": "li", "path": "li.vasp", "fingerprint": sha256(source)}],
        },
    )
    labeling = inputs / "labeling.json"
    write_json(
        labeling,
        {
            "schema_version": 1,
            "engine": "vasp",
            "calculation_type": "static",
            "incar": {"ENCUT": 450, "EDIFF": 5e-6, "NSW": 0, "IBRION": -1},
            "kpoints": {"mode": "gamma", "grid": [1, 1, 1], "shift": [0, 0, 0]},
        },
    )
    contents = {
        "POSCAR": source.read_text(encoding="utf-8"),
        "INCAR": "ENCUT = 450\nEDIFF = 5e-6\nNSW = 0\nIBRION = -1\n",
        "KPOINTS": "mesh\n0\nGamma\n1 1 1\n0 0 0\n",
        "POTCAR": "TEST-ONLY-POTCAR\n",
    }
    files: dict[str, Any] = {}
    for name, text in contents.items():
        path = prepared / name
        path.write_text(text, encoding="utf-8")
        files[name] = {
            "path": name,
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
            "collectable": name != "POTCAR",
        }
    prepared_manifest = prepared / "dft-input-manifest.json"
    write_json(
        prepared_manifest,
        {
            "schema_version": 1,
            "plugin_id": "dft-labeling",
            "operation": "vasp-prepare",
            "status": "OK",
            "engine": "vasp",
            "calculation_type": "static",
            "structure_count": 1,
            "calculations": [
                {
                    "structure_id": "li",
                    "atom_count": 1,
                    "files": files,
                }
            ],
        },
    )
    fake_vasp = inputs / "fake_vasp.py"
    fake_vasp.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "Path('OUTCAR').write_text('vasp.6.3.0\\nGeneral timing and accounting informations for this job:\\n')\n"
        "Path('OSZICAR').write_text(' 1 F= -.150000E+01 E0= -.150000E+01\\n')\n"
        "Path('vasprun.xml').write_text('''<modeling>\n"
        "<parameters><separator><i name=\"NELM\">700</i></separator></parameters>\n"
        "<atominfo><array name=\"atoms\"><set><rc><c>Li</c></rc></set></array></atominfo>\n"
        "<structure name=\"finalpos\"><crystal><varray name=\"basis\"><v>3 0 0</v><v>0 3 0</v><v>0 0 3</v></varray></crystal><varray name=\"positions\"><v>0 0 0</v></varray></structure>\n"
        "<calculation><scstep/><energy><i name=\"e_0_energy\">-1.5</i></energy><varray name=\"forces\"><v>0 0 0</v></varray><varray name=\"stress\"><v>1 0 0</v><v>0 1 0</v><v>0 0 1</v></varray></calculation>\n"
        "</modeling>''')\n",
        encoding="utf-8",
    )
    parameters = {
        "operation": "label",
        "scheduler_runner": "bundled-vasp-static-v1",
        "remote_python": str(Path(sys.executable).resolve()),
        "modules": [],
        "vasp_argv": ["srun", "--ntasks=4", "vasp_std"],
        "engine": "vasp",
        "completion_policy": {"require_ionic_convergence": False},
        "units": {
            "energy": "eV",
            "length": "angstrom",
            "force": "eV/angstrom",
            "stress": "kbar-vasp-3x3",
        },
        "result_manifest": "dft-labeling-result.json",
    }
    node = {
        "id": "label-li",
        "uses": "dft-labeling@0",
        "backend": "ssh-slurm",
        "backend_profile": "cpu",
        "inputs": {
            "structures_manifest": "inputs/structures.json",
            "labeling_config": "inputs/labeling.json",
            "dft_input_manifest": "prepared/dft-input-manifest.json",
        },
        "parameters": parameters,
        "resources": {
            "partition": "cpu192",
            "qos": "debug",
            "nodes": 1,
            "ntasks": 4,
            "time": "00:05:00",
        },
    }
    config = project_config([node])
    config["backend_profiles"] = {
        "cpu": {"ssh_profile": "cpu", "remote_root": "."}
    }
    write_json(root / "project.yaml", config)
    return node, parameters


class ScheduledDftTests(unittest.TestCase):
    def test_plan_submission_and_approved_fetch_use_pinned_checker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, parameters = prepared_fixture(root)
            initialize(root)
            project = load_project(root)
            plan = make_run_plan(project, "label-li", PLUGINS)
            scheduled = plan["adapter_plan"]["scheduled_execution"]
            potcar = next(item for item in scheduled["staged_files"] if item["remote_name"] == "POTCAR")
            self.assertTrue(potcar["sensitive"])
            self.assertFalse(potcar["fetch_allowed"])
            self.assertNotIn("POTCAR", {item["remote_name"] for item in scheduled["fetch_outputs"]})
            remote_cwd = plan["scheduled_transport"]["remote_cwd"]
            with patch(
                "mlipflow.services.SshSlurmBackend.stage_fresh",
                return_value=remote_cwd,
            ) as staged, patch(
                "mlipflow.services.SshSlurmBackend.submit",
                return_value=ExecutionResult(0, "Submitted batch job 77\n", "", "77"),
            ):
                submitted = run_node(project, "label-li", PLUGINS, plan["plan_digest"])
            self.assertEqual("PENDING", submitted["step"]["state"])
            staged_names = {item[1] for item in staged.call_args.args[2]}
            self.assertIn("POTCAR", staged_names)
            self.assertIn("run.slurm", staged_names)

            remote = root / "remote-fixture"
            remote.mkdir()
            for name in ("POSCAR", "INCAR", "KPOINTS", "POTCAR"):
                shutil.copy2(root / "prepared" / name, remote / name)
            for source, destination in (
                (root / "inputs/structures.json", remote / "structures.json"),
                (root / "inputs/labeling.json", remote / "labeling.json"),
                (root / "prepared/dft-input-manifest.json", remote / "dft-input-manifest.json"),
                (PLUGINS / "dft-labeling/vasp_label.py", remote / "mlipflow-vasp-label.py"),
            ):
                shutil.copy2(source, destination)
            shutil.copy2(root / "inputs/fake_vasp.py", remote / "srun")
            (remote / "srun").chmod(0o700)
            wrapper = load_module("test_scheduled_vasp_wrapper", remote / "mlipflow-vasp-label.py")
            arguments = argparse.Namespace(
                attempt_dir=str(remote),
                structures_manifest="structures.json",
                labeling_config="labeling.json",
                dft_input_manifest="dft-input-manifest.json",
                result_manifest="dft-labeling-result.json",
                labels="labels.json",
                units_json=json.dumps(parameters["units"]),
                vasp_argv_json=json.dumps(parameters["vasp_argv"]),
            )
            with patch.dict(
                os.environ,
                {"PATH": str(remote) + os.pathsep + os.environ.get("PATH", "")},
            ):
                wrapper.execute(arguments)
            inventory = {
                path.name: {
                    "path": path.name,
                    "exists": True,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in remote.iterdir()
                if path.is_file()
            }

            def inspect(_: object, _cwd: str, remote_name: str) -> dict[str, object]:
                return inventory.get(remote_name, {"path": remote_name, "exists": False})

            def fetch(
                _: object, _cwd: str, remote_name: str, destination: Path
            ) -> Path:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(remote / remote_name, destination)
                return destination

            with patch(
                "mlipflow.services.SshSlurmBackend.status",
                return_value={"state": "COMPLETED", "detail": None, "source": "remote"},
            ), patch(
                "mlipflow.services.SshSlurmBackend.inspect_file", autospec=True, side_effect=inspect
            ):
                advance_plan = make_advance_plan(project, PLUGINS)
            transition = advance_plan["details"]["transitions"][0]
            self.assertEqual("adapter-finalize", transition["action"])
            with patch(
                "mlipflow.services.SshSlurmBackend.status",
                return_value={"state": "COMPLETED", "detail": None, "source": "remote"},
            ), patch(
                "mlipflow.services.SshSlurmBackend.inspect_file", autospec=True, side_effect=inspect
            ), patch(
                "mlipflow.services.SshSlurmBackend.fetch_from", autospec=True, side_effect=fetch
            ):
                finished = advance(project, advance_plan["plan_digest"], PLUGINS)
            self.assertEqual("OK", finished["changed"][0]["state"])
            attempt = root / ".mlipflow/runs/label-li/attempt-1"
            self.assertTrue((attempt / "OUTCAR").is_file())
            self.assertFalse((attempt / "POTCAR").is_file())
            self.assertTrue((attempt / "run-manifest.final.json").is_file())

    def test_remote_output_change_after_advance_plan_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_fixture(root)
            initialize(root)
            project = load_project(root)
            plan = make_run_plan(project, "label-li", PLUGINS)
            remote_cwd = plan["scheduled_transport"]["remote_cwd"]
            with patch(
                "mlipflow.services.SshSlurmBackend.stage_fresh", return_value=remote_cwd
            ), patch(
                "mlipflow.services.SshSlurmBackend.submit",
                return_value=ExecutionResult(0, "Submitted batch job 78\n", "", "78"),
            ):
                run_node(project, "label-li", PLUGINS, plan["plan_digest"])
            def missing(_self: object, _cwd: str, name: str) -> dict[str, object]:
                return {"path": name, "exists": False}
            with patch(
                "mlipflow.services.SshSlurmBackend.status",
                return_value={"state": "COMPLETED", "detail": None, "source": "remote"},
            ), patch(
                "mlipflow.services.SshSlurmBackend.inspect_file", autospec=True, side_effect=missing
            ):
                approved = make_advance_plan(project, PLUGINS)
            def changed(_self: object, _cwd: str, name: str) -> dict[str, object]:
                return {
                    "path": name,
                    "exists": True,
                    "size_bytes": 1,
                    "sha256": "sha256:" + "0" * 64,
                }
            with patch(
                "mlipflow.services.SshSlurmBackend.status",
                return_value={"state": "COMPLETED", "detail": None, "source": "remote"},
            ), patch(
                "mlipflow.services.SshSlurmBackend.inspect_file", autospec=True, side_effect=changed
            ):
                with self.assertRaisesRegex(Exception, "approval digest mismatch|changed after"):
                    advance(project, approved["plan_digest"], PLUGINS)


if __name__ == "__main__":
    unittest.main()
