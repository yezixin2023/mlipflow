"""Focused orchestration tests for the bounded ionic-transport MD handoff.

The fixtures imitate the two historical source contracts.  The bounded handoff
uses real ASE Atoms/Trajectory I/O, but it does not run a calculator or stand in
for scientific numerical parity.
"""

from __future__ import annotations

from tests.helpers import load_module
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "mlipflow" / "plugins" / "ionic_transport" / "adapter.py"


def load_adapter():
    module = load_module(ADAPTER_PATH, 'ionic_md_handoff_adapter')
    return module.Adapter()


FAKE_MD_SOURCE = '''from pathlib import Path
import json
from ase import Atoms
from ase.io.trajectory import Trajectory

OUTPUT_ROOT = Path("unused")
TIMESTEP_FS = 1.0
TRAJ_INTERVAL = 1
PROD_STEPS = 1
NPT_STEPS = 0
EQUIL_STEPS = 0


def run_single_temperature_md(
    temperature_k,
    structure_path,
    calculator_type,
    model_path,
    input_format=None,
    index="-1",
):
    run_dir = OUTPUT_ROOT / ("T%d" % int(temperature_k))
    run_dir.mkdir(parents=True, exist_ok=False)
    trajectory = run_dir / "production.traj"
    with Trajectory(str(trajectory), "w") as writer:
        scale = float(temperature_k) / 400.0
        for frame in range(5):
            atoms = Atoms(
                "LiHe",
                positions=[[0.08 * scale * frame, 0, 0], [2, 2, 2]],
                cell=[5, 5, 5],
                pbc=True,
            )
            writer.write(atoms)
    metadata = {
        "temperature_K": float(temperature_k),
        "traj_path": str(trajectory),
        "time_step_fs_between_frames": TIMESTEP_FS * TRAJ_INTERVAL,
        "n_frames": 5,
        "calculator_type": calculator_type,
        "model_path": str(model_path),
        "structure_path": str(structure_path),
        "input_format": input_format,
        "input_index": index,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return metadata
'''


class IonicMDHandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.project = Path(self.temporary.name).resolve()
        self.attempt = self.project / "attempt"
        self.attempt.mkdir()
        self.md_script = self.project / "ase_md_only_multi_calc.py"
        self.structure = self.project / "structure.xyz"
        self.md_script.write_text(FAKE_MD_SOURCE, encoding="utf-8")
        self.structure.write_text("1\nfixture\nLi 0 0 0\n", encoding="utf-8")
        self.adapter = load_adapter()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def smoke_context(self) -> dict:
        return {
            "project_root": str(self.project),
            "attempt_dir": str(self.attempt),
            "inputs": {
                "md_script": str(self.md_script),
                "structure": str(self.structure),
                "model": "default",
            },
            "parameters": {
                "operation": "md-smoke-and-analyze",
                "output_subdir": "transport-analysis",
                "md_output_subdir": "md-smoke",
                "integration_manifest": "md-integration-manifest.json",
                "calculator": "emt",
                "temperatures_k": [400, 600],
                "seed": 17,
                "timestep_fs": 1.0,
                "npt_time_ps": 0.0,
                "nvt_equil_time_ps": 0.0,
                "production_time_ps": 0.01,
                "trajectory_interval_steps": 2,
                "device": "cpu",
                "default_dtype": "float64",
                "input_format": "xyz",
                "input_index": "-1",
                "source": "trajectory",
                "specie": "Li",
                "trajectory_start_ps": None,
                "trajectory_end_ps": None,
                "trajectory_msd_engine": "diffusion-analyzer",
                "diffusion_analyzer_smoothed": "none",
                "diffusion_analyzer_min_obs": 3,
                "diffusion_analyzer_avg_nsteps": 3,
                "diffusion_analyzer_step_skip": 1,
                "min_msd_fit_points": 3,
                "msd_smooth_window_points": None,
                "msd_smooth_window_ps": None,
                "fit_smoothed_msd": False,
                "ase_frame_step_fs": 2.0,
                "fit_temperatures": None,
                "exclude_temperatures": None,
                "fit_scope": "all",
                "target_temperature_k": 300.0,
                "piecewise": "never",
                "min_segment_points": 3,
                "piecewise_slope_change": 0.35,
                "piecewise_bic_delta": 2.0,
                "allow_partial_results": False,
            },
            "backend": "local",
            "resources": {"python_executable": sys.executable},
        }

    def test_fake_historical_sources_complete_the_confined_handoff(self) -> None:
        context = self.smoke_context()
        immutable = [self.md_script, self.structure]
        before = {path: path.read_bytes() for path in immutable}
        plan = self.adapter.plan(context)
        self.assertEqual("READY", plan["status"], plan.get("diagnostics"))
        self.assertEqual("md-smoke-and-analyze", plan["operation"])
        self.assertFalse(plan["shell"])
        self.assertIn("integration-smoke-only", plan["assumptions"]["scientific_use"])

        completed = subprocess.run(
            plan["argv"],
            cwd=plan["cwd"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        context["execution"] = {"returncode": completed.returncode, "plan": plan}
        checked = self.adapter.check(context)
        self.assertEqual("OK", checked["status"], checked.get("diagnostics"))
        self.assertEqual("integration-smoke-only", checked["scientific_use"])

        manifest_path = Path(plan["integration_manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual("integration-smoke-only", manifest["scientific_use"])
        self.assertFalse(manifest["boundaries"]["scientific_parity_claimed"])
        self.assertFalse(manifest["randomness"]["framework_rngs_fully_controlled"])
        self.assertEqual(2, len(manifest["trajectory_artifacts"]))
        for group_name in ("trajectory_artifacts", "md_artifacts", "result_artifacts"):
            roles = [item["role"] for item in manifest[group_name]]
            self.assertEqual(len(roles), len(set(roles)), group_name)
            self.assertTrue(
                all(
                    (self.attempt / item["path"]).resolve().is_relative_to(self.attempt)
                    for item in manifest[group_name]
                )
            )
        self.assertTrue(
            {
                "diffusion_results_by_temperature.csv",
                "arrhenius_summary.json",
                "postprocess_failures.json",
                "analysis_manifest.json",
            }.issubset({Path(item["path"]).name for item in manifest["result_artifacts"]})
        )
        for group_name in ("source_artifacts", "input_artifacts", "result_artifacts"):
            self.assertTrue(
                all(set(item) == {"role", "path"} for item in manifest[group_name])
            )
        self.assertFalse(manifest["execution"]["absolute_paths_recorded"])
        self.assertTrue(
            all(Path(path).resolve().is_relative_to(self.attempt) for path in plan["expected_outputs"])
        )
        self.assertEqual(
            before, {path: path.read_bytes() for path in immutable}
        )
        self.assertFalse((self.project / "__pycache__").exists())

        collected = self.adapter.collect(context)
        self.assertEqual("OK", collected["status"])
        roles = {item["role"] for item in collected["artifacts"]}
        self.assertIn("md-integration-manifest", roles)
        self.assertIn("production-trajectory", roles)
        self.assertEqual(
            "integration-smoke-only", collected["metrics"]["assumptions"]["scientific_use"]
        )

        first_trajectory = Path(plan["trajectory_paths"][0])
        first_trajectory.unlink()
        missing = self.adapter.check(context)
        self.assertEqual("FAIL", missing["status"])
        self.assertIn(
            "result.manifest_missing", {item["code"] for item in missing["diagnostics"]}
        )

    def test_smoke_caps_and_network_prone_default_are_blocked(self) -> None:
        context = self.smoke_context()
        context["parameters"]["production_time_ps"] = 100.0
        blocked = self.adapter.plan(context)
        self.assertEqual("BLOCKED", blocked["status"])
        self.assertIn("parameter.production_time_ps", {item["code"] for item in blocked["diagnostics"]})

        context = self.smoke_context()
        context["parameters"]["calculator"] = "m3gnet"
        blocked = self.adapter.plan(context)
        self.assertEqual("BLOCKED", blocked["status"])
        self.assertIn(
            "parameter.default_model_network", {item["code"] for item in blocked["diagnostics"]}
        )

        context = self.smoke_context()
        context["parameters"]["calculator"] = "chgnet"
        blocked = self.adapter.plan(context)
        self.assertEqual("BLOCKED", blocked["status"])
        self.assertIn(
            "parameter.default_model_network", {item["code"] for item in blocked["diagnostics"]}
        )

        context = self.smoke_context()
        context["parameters"]["integration_manifest"] = "md-smoke/nested-manifest.json"
        blocked = self.adapter.plan(context)
        self.assertEqual("BLOCKED", blocked["status"])
        self.assertIn(
            "path.smoke_output_overlap", {item["code"] for item in blocked["diagnostics"]}
        )

    def test_legacy_context_without_operation_still_plans_analysis(self) -> None:
        msd = self.project / "target.msd"
        msd.write_text("time,msd\n0,0\n1,1\n2,2\n", encoding="utf-8")
        context = self.smoke_context()
        context["inputs"] = {
            "input_paths": [str(msd)],
        }
        parameters = context["parameters"]
        for name in (
            "operation",
            "md_output_subdir",
            "integration_manifest",
            "calculator",
            "temperatures_k",
            "timestep_fs",
            "npt_time_ps",
            "nvt_equil_time_ps",
            "production_time_ps",
            "trajectory_interval_steps",
            "device",
            "default_dtype",
            "input_format",
            "input_index",
        ):
            parameters.pop(name, None)
        parameters["output_subdir"] = "analysis-only"
        parameters["source"] = "msd"
        parameters["seed"] = None
        parameters["msd_time_unit"] = "ps"
        parameters["msd_unit"] = "A2"
        parameters["n_mobile_ions"] = None
        parameters["volume_a3"] = None
        plan = self.adapter.plan(context)
        self.assertEqual("READY", plan["status"], plan.get("diagnostics"))
        self.assertEqual("analyze-existing", plan["operation"])


if __name__ == "__main__":
    unittest.main()
