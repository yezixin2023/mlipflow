"""Lightweight contract tests for scheduled LASP stochastic-surface-walking."""
from __future__ import annotations

from tests.helpers import load_module
import json
import shutil
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from mlipflow.config import load_project
from mlipflow.services import initialize, make_run_plan
from mlipflow.services.contracts import _scheduled_contract

from .helpers import project_config, write_json
from .test_scheduled_dft import FakeTemplateLibrary, write_site

ROOT = Path(__file__).resolve().parents[1]
PES = ROOT / "mlipflow" / "plugins" / "pes_sampling"
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
    prepared = inputs / "prepared"
    prepared.mkdir()
    structure = prepared / "input.arc"
    structure.write_text(arc_payload([-10.0]), encoding="utf-8")
    potcar = prepared / "POTCAR"
    potcar.write_bytes(b"FAKE-Li\nFAKE-S\n")
    write_json(
        prepared / "lasp-input-manifest.json",
        {
            "schema_version": 1,
            "plugin_id": "pes-sampling",
            "operation": "lasp-input-prepare",
            "pseudopotential_reference_path": "inputs/pseudopotentials.json",
            "output": {"path": "input.arc"},
            "potcar": {
                "reference_id": "fixture-pbe54-plain-v1",
                "source_env": "PMG_VASP_PSP_DIR",
                "configuration_source": "pymatgen-settings",
                "functional": "PBE_54",
                "elements": ["Li", "S"],
                "symbols": ["Li", "S"],
                "components": [{"symbol": symbol} for symbol in ("Li", "S")],
                "portable_artifact": False,
                "output": {"path": "POTCAR", "collectable": False},
            },
        },
    )
    (inputs / "fixture.pot").write_text("scheduled auxiliary fixture\n", encoding="utf-8")
    (inputs / "lasp.in").write_text(
        "potential vasp\nexplore_type ssw\nRun_type 5\nSSW.SSWsteps 4 # inline LASP comment\nSSW.MaxOptstep 300\nSSW.Temp 200.0000\n",
        encoding="utf-8",
    )
    write_json(
        root / "project.yaml",
        project_config(
            [
                {
                    "id": "lasp-walk",
                    "uses": "pes-sampling",
                    "mode": "execute",
                    "backend": "ssh-slurm",
                    "backend_profile": "cluster-a",
                    "inputs": {
                        "input_structure": "inputs/prepared/input.arc",
                        "lasp_input_manifest": "inputs/prepared/lasp-input-manifest.json",
                        "lasp_input": "inputs/lasp.in",
                        "lasp_auxiliary_files": {"fixture.pot": "inputs/fixture.pot"},
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
    module = load_module(path, 'scheduled_lasp_remote_test')
    return module


def load_cluster_adapter():
    path = PES / "adapter.py"
    module = load_module(path, 'scheduled_lasp_adapter_test')
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
        plan = make_run_plan(self.project, "lasp-walk", self.site, library())
        adapter = plan["adapter_plan"]
        self.assertEqual("READY", adapter["status"], adapter.get("diagnostics"))
        scheduled = adapter["scheduled_execution"]
        self.assertEqual(3, scheduled["schema_version"])
        self.assertEqual("mpi", scheduled["execution_model"])
        self.assertEqual("lasp-ssw", scheduled["template_family"])
        self.assertEqual("mpi-task-count", adapter["approval_summary"]["cpus_meaning"])
        staged = {item["remote_name"] for item in scheduled["staged_files"]}
        self.assertTrue({"project.yaml", "input.arc", "lasp.in", "lasp-input-manifest.json", "POTCAR", "fixture.pot", "lasp_ssw.py", "lasp_cluster.py"}.issubset(staged))
        staged_records = {item["remote_name"]: item for item in scheduled["staged_files"]}
        self.assertTrue(Path(staged_records["POTCAR"]["source"]).is_file())
        self.assertEqual("vasp", adapter["lasp_calculation"]["potential"])
        self.assertEqual(
            "fixture-pbe54-plain-v1",
            adapter["lasp_calculation"]["pseudopotential"]["reference_id"],
        )
        self.assertNotIn("lasp_executable", staged)
        self.assertNotIn("lasp_executable", adapter["input_paths"])
        self.assertNotIn("lasp_executable", adapter["lasp_calculation"])
        self.assertTrue(adapter["assumptions"]["cluster_lasp_executable_is_site_owned"])
        fetched = {item["remote_name"] for item in adapter["scheduled_execution"]["fetch_outputs"]}
        self.assertTrue({"cluster-run-report.json", "sampling-result.json", "selected-structures.tar.gz", "allstr.arc"}.issubset(fetched))
        self.assertNotIn("POTCAR", fetched)

    def test_core_accepts_the_lasp_scheduled_execution(self) -> None:
        plan = make_run_plan(self.project, "lasp-walk", self.site, library())
        contract = _scheduled_contract(
            self.project, "pes-sampling", plan, node_id="lasp-walk", attempt=1
        )
        self.assertEqual(3, contract["schema_version"])
        self.assertEqual("mpi", contract["execution_model"])
        self.assertEqual("lasp-ssw", contract["template_family"])
        names = {item["remote_name"] for item in contract["fetch_outputs"]}
        self.assertIn("completion.json", names)
        self.assertIn("selected-structures.tar.gz", names)

    def test_pseudopotential_check_ignores_site_local_path_prefixes(self) -> None:
        plan = make_run_plan(self.project, "lasp-walk", self.site, library())
        approved = plan["adapter_plan"]["lasp_calculation"]["pseudopotential"]
        reported = json.loads(json.dumps(approved))
        reported["reference_path"] = "/remote/input/pseudopotentials.json"
        reported["manifest_path"] = "/remote/input/lasp-input-manifest.json"
        reported["potcar_path"] = "/remote/input/POTCAR"
        from mlipflow.plugins.pes_sampling import scheduled as adapter
        self.assertEqual(
            adapter._pseudopotential_settings(approved),
            adapter._pseudopotential_settings(reported),
        )
        reported["functional"] = "PBE"
        self.assertNotEqual(
            adapter._pseudopotential_settings(approved),
            adapter._pseudopotential_settings(reported),
        )

    def test_scheduled_plan_rejects_local_lasp_executable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            site = build_project(root)
            project = json.loads((root / "project.yaml").read_text(encoding="utf-8"))
            project["workflow"]["nodes"][0]["inputs"]["lasp_executable"] = "/site/lasp"
            write_json(root / "project.yaml", project)
            initialize(root)
            current = load_project(root)
            plan = make_run_plan(current, "lasp-walk", site, library())
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
        shutil.copy2(self.root / "inputs" / "prepared" / "input.arc", self.input_dir / "input.arc")
        shutil.copy2(
            self.root / "inputs" / "prepared" / "lasp-input-manifest.json",
            self.input_dir / "lasp-input-manifest.json",
        )
        shutil.copy2(self.root / "inputs" / "prepared" / "POTCAR", self.input_dir / "POTCAR")
        shutil.copy2(self.root / "inputs" / "lasp.in", self.input_dir / "lasp.in")
        shutil.copy2(self.root / "inputs" / "fixture.pot", self.input_dir / "fixture.pot")
        shutil.copy2(PES / "lasp_ssw.py", self.input_dir / "lasp_ssw.py")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_fake_lasp_native_all_arc_is_canonicalized_and_packed(self) -> None:
        fake = self.root / "lasp"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "from pathlib import Path\n"
            f"Path('all.arc').write_text({arc_payload([-10.0, -9.0, 1.0, -8.0, -7.0])!r}, encoding='utf-8')\n"
            f"Path('allstr.arc').write_text({arc_payload([-20.0, -19.0, -18.0, -17.0, -16.0, -15.0])!r}, encoding='utf-8')\n",
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
        self.assertEqual("vasp", report["potential"])
        self.assertEqual("fixture-pbe54-plain-v1", report["pseudopotential"]["reference_id"])
        self.assertEqual("pseudopotentials.json", report["pseudopotential"]["reference_path"])
        self.assertEqual(
            "lasp-input-manifest.json", report["pseudopotential"]["manifest_path"]
        )
        self.assertEqual("POTCAR", report["pseudopotential"]["potcar_path"])
        self.assertEqual(2, report["counts"]["selected_structure_count"])
        canonical = self.output_dir / "lasp-ssw" / "raw-run" / "allstr.arc"
        self.assertEqual(
            (self.output_dir / "lasp-ssw" / "raw-run" / "all.arc").read_bytes(),
            canonical.read_bytes(),
        )
        preserved = self.output_dir / "lasp-ssw" / "raw-run" / "allstr.native.arc"
        self.assertEqual(
            arc_payload([-20.0, -19.0, -18.0, -17.0, -16.0, -15.0]),
            preserved.read_text(encoding="utf-8"),
        )
        metadata = json.loads(
            (self.output_dir / "lasp-ssw" / "lasp-run-metadata.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual("all.arc", metadata["execution"]["ssw_archive_source_name"])
        self.assertEqual(
            "allstr.arc", metadata["execution"]["ssw_archive_canonical_name"]
        )
        self.assertTrue(metadata["execution"]["ssw_archive_canonicalized"])
        self.assertEqual(
            "allstr.native.arc",
            metadata["execution"]["native_allstr_preserved"]["name"],
        )
        archive = self.output_dir / "selected-structures.tar.gz"
        self.assertTrue(archive.is_file())
        import tarfile
        with tarfile.open(archive, "r:gz") as bundle:
            self.assertEqual(["selected/input-000001.arc", "selected/input-000002.arc"], [item.name for item in bundle.getmembers()])


if __name__ == "__main__":
    unittest.main()
