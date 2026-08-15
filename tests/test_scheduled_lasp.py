"""Lightweight contract tests for scheduled LASP stochastic-surface-walking."""
from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from mlipflow.config import load_project
from mlipflow.services import initialize, make_run_plan
from mlipflow.services.contracts import _scheduled_contract
from mlipflow.plugins import discover_plugins

from .helpers import project_config, write_json
from .test_scheduled_dft import FakeTemplateLibrary, PLUGINS, write_site

ROOT = Path(__file__).resolve().parents[1]
PES = ROOT / "plugins" / "pes-sampling"
ARC_HEADER = "!BIOSYM archive 2\nPBC=ON\n"


def arc_payload(energies: list[float]) -> str:
    chunks = [ARC_HEADER]
    for index, energy in enumerate(energies, 1):
        chunks.extend(
            [
                f"Energy {index} {energy:.8f}\n",
                "!DATE scheduled-fixture\n",
                "PBC 5.0 5.0 5.0 90.0 90.0 90.0\n",
                f"Li {index / 10:.4f} 0.0 0.0 CORE 1 Li Li 0.0 1\n",
                "S 1.0 1.0 1.0 CORE 2 S S 0.0 2\n",
                "end\n",
                "end\n",
            ]
        )
    return "".join(chunks)


def lasp_run_template() -> str:
    return """#!/bin/bash
# project={{PROJECT_ID}} node={{NODE_ID}} attempt={{ATTEMPT}}
# cpus={{CPUS}}
# inputs={{INPUT_DIR}}
# outputs={{OUTPUT_DIR}}
cd {{RUN_DIR}}
"""


def library() -> FakeTemplateLibrary:
    templates = dict(FakeTemplateLibrary().templates)
    templates["lasp-ssw/run.sh"] = lasp_run_template()
    return FakeTemplateLibrary(templates)


def build_project(root: Path) -> Path:
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "input.arc").write_text(arc_payload([-10.0]), encoding="utf-8")
    (inputs / "lasp.in").write_text(
        "potential vasp\nexplore_type ssw\nRun_type 5\nSSW.SSWsteps 4\nSSW.MaxOptstep 300\nSSW.Temp 200.0000\n",
        encoding="utf-8",
    )
    write_json(
        root / "project.yaml",
        project_config(
            [
                {
                    "id": "lasp-walk",
                    "uses": "pes-sampling@0",
                    "mode": "execute",
                    "backend": "ssh-slurm",
                    "backend_profile": "cluster-a",
                    "inputs": {
                        "input_structure": "inputs/input.arc",
                        "lasp_input": "inputs/lasp.in",
                        "lasp_auxiliary_files": {},
                    },
                    "parameters": {
                        "operation": "lasp-ssw-execute",
                        "output_subdir": "lasp-ssw",
                        "historical_source_id": "fixture://lasp/scheduled-walk",
                        "selection_stride": 2,
                        "energy_max_ev": 0.0,
                        "max_frames": 5,
                        "include_best_arc": False,
                        "include_md_arc": False,
                        "seed_status": "HISTORICAL_PARAMETER_UNKNOWN",
                        "acknowledge_uncontrolled_seed": True,
                        "preserve_historical_order": True,
                        "lasp_version": "fixture-1.0",
                    },
                    "resources": {
                        "cpus": 4,
                        "gpus": 0,
                        "memory": "8G",
                        "walltime": "00:10:00",
                    },
                }
            ]
        ),
    )
    return write_site(root)


def load_remote_runner():
    path = PES / "lasp_cluster.py"
    spec = importlib.util.spec_from_file_location("scheduled_lasp_remote_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScheduledLaspPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.site = build_project(self.root)
        initialize(self.root)
        self.project = load_project(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_make_run_plan_produces_lasp_scheduled_contract(self) -> None:
        plan = make_run_plan(self.project, "lasp-walk", PLUGINS, self.site, library())
        adapter = plan["adapter_plan"]
        self.assertEqual("READY", adapter["status"], adapter.get("diagnostics"))
        scheduled = adapter["scheduled_execution"]
        self.assertEqual(3, scheduled["schema_version"])
        self.assertEqual("mpi", scheduled["execution_model"])
        self.assertEqual("lasp-ssw", scheduled["template_family"])
        self.assertEqual("mpi-task-count", adapter["approval_summary"]["cpus_meaning"])
        staged = {item["remote_name"] for item in scheduled["staged_files"]}
        self.assertTrue({"project.yaml", "input.arc", "lasp.in", "lasp_ssw.py", "lasp_cluster.py"}.issubset(staged))
        self.assertNotIn("lasp_executable", staged)
        self.assertNotIn("lasp_executable", adapter["input_fingerprints"])
        self.assertNotIn("lasp_executable", adapter["lasp_scheduled_identity"])
        self.assertTrue(adapter["assumptions"]["cluster_lasp_executable_is_site_owned"])
        fetched = {item["remote_name"] for item in adapter["scheduled_execution"]["fetch_outputs"]}
        self.assertTrue({"cluster-run-report.json", "sampling-result.json", "selected-structures.tar.gz", "allstr.arc"}.issubset(fetched))

    def test_core_accepts_the_lasp_scheduled_execution(self) -> None:
        plan = make_run_plan(self.project, "lasp-walk", PLUGINS, self.site, library())
        plugin = discover_plugins(PLUGINS)["pes-sampling"]
        contract = _scheduled_contract(self.project, plugin, plan, node_id="lasp-walk", attempt=1)
        self.assertEqual(3, contract["schema_version"])
        self.assertEqual("mpi", contract["execution_model"])
        self.assertEqual("lasp-ssw", contract["template_family"])
        names = {item["remote_name"] for item in contract["fetch_outputs"]}
        self.assertIn("completion.json", names)
        self.assertIn("selected-structures.tar.gz", names)

    def test_scheduled_plan_rejects_local_lasp_executable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            site = build_project(root)
            project = json.loads((root / "project.yaml").read_text(encoding="utf-8"))
            project["workflow"]["nodes"][0]["inputs"]["lasp_executable"] = "/site/lasp"
            write_json(root / "project.yaml", project)
            initialize(root)
            current = load_project(root)
            plan = make_run_plan(current, "lasp-walk", PLUGINS, site, library())
            self.assertEqual("BLOCKED", plan["adapter_plan"]["status"])
            codes = {item["code"] for item in plan["adapter_plan"]["diagnostics"]}
            self.assertIn("input.unknown", codes)


class LaspRemoteRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        build_project(self.root)
        self.input_dir = self.root / "remote-input"
        self.output_dir = self.root / "remote-output"
        self.input_dir.mkdir()
        shutil.copy2(self.root / "project.yaml", self.input_dir / "project.yaml")
        shutil.copy2(self.root / "inputs" / "input.arc", self.input_dir / "input.arc")
        shutil.copy2(self.root / "inputs" / "lasp.in", self.input_dir / "lasp.in")
        shutil.copy2(PES / "lasp_ssw.py", self.input_dir / "lasp_ssw.py")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_fake_lasp_is_normalized_and_packed_for_bounded_fetch(self) -> None:
        fake = self.root / "lasp"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "from pathlib import Path\n"
            f"Path('allstr.arc').write_text({arc_payload([-10.0, -9.0, 1.0, -8.0, -7.0])!r}, encoding='utf-8')\n",
            encoding="utf-8",
        )
        fake.chmod(0o755)
        runner = load_remote_runner()
        code = runner.run(
            Namespace(
                project=str(self.input_dir / "project.yaml"),
                node_id="lasp-walk",
                input_dir=str(self.input_dir),
                output_dir=str(self.output_dir),
                lasp_executable=str(fake),
                mpi_launcher=None,
                mpi_processes=None,
            )
        )
        self.assertEqual(0, code)
        report = json.loads((self.output_dir / "cluster-run-report.json").read_text(encoding="utf-8"))
        self.assertEqual("OK", report["status"])
        self.assertEqual(2, report["counts"]["selected_structure_count"])
        archive = self.output_dir / "selected-structures.tar.gz"
        self.assertTrue(archive.is_file())
        import tarfile
        with tarfile.open(archive, "r:gz") as bundle:
            self.assertEqual(["selected/input-000001.arc", "selected/input-000002.arc"], [item.name for item in bundle.getmembers()])


if __name__ == "__main__":
    unittest.main()
