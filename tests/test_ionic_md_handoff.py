"""Focused orchestration tests for the bounded ionic-transport MD handoff.

The fixtures imitate the two historical source contracts.  They do not import
ASE, run a calculator, or stand in for scientific numerical parity.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "plugins" / "ionic-transport" / "adapter.py"


def load_adapter():
    spec = importlib.util.spec_from_file_location("ionic_md_handoff_adapter", ADAPTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Adapter()


FAKE_MD_SOURCE = '''from pathlib import Path
import json

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
    trajectory.write_bytes(("fake-trajectory-%g" % temperature_k).encode("ascii"))
    metadata = {
        "temperature_K": float(temperature_k),
        "traj_path": str(trajectory),
        "time_step_fs_between_frames": TIMESTEP_FS * TRAJ_INTERVAL,
        "n_frames": int((PROD_STEPS + TRAJ_INTERVAL - 1) / TRAJ_INTERVAL),
        "calculator_type": calculator_type,
        "model_path": str(model_path),
        "structure_path": str(structure_path),
        "input_format": input_format,
        "input_index": index,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return metadata
'''


FAKE_ANALYSIS_SOURCE = '''from pathlib import Path
import argparse
import csv
import json


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args, unknown = parser.parse_known_args()
source = Path(args.input)
output = Path(args.output)
output.mkdir(parents=True, exist_ok=False)
rows = []
for run_dir in sorted(source.glob("T*")):
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    temperature = float(metadata["temperature_K"])
    curve = output / ("msd_%d.csv" % int(temperature))
    fit = output / ("msd_%d.html" % int(temperature))
    curve.write_text("time_ps,msd_A2\\n0,0\\n0.004,0.04\\n0.008,0.08\\n", encoding="utf-8")
    fit.write_text("<html>fake fit</html>\\n", encoding="utf-8")
    rows.append({
        "temperature_K": temperature,
        "diffusivity_cm2_s": 1.0e-7 * temperature,
        "fit_start_ps": 0.0,
        "fit_end_ps": 0.008,
        "msd_fit_r2": 0.99,
        "n_mobile_ions": 1,
        "charge": 1.0,
        "volume_A3": 100.0,
        "conductivity_NE_mS_cm": 0.01 * temperature,
        "msd_curve_csv": str(curve),
        "msd_fit_html": str(fit),
    })
with (output / "diffusion_results_by_temperature.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(output / "arrhenius_summary.json").write_text(
    json.dumps({
        "fit_scope": "all",
        "arrhenius_fit_skipped": False,
        "single": {"Ea_eV": 0.123, "r2_lnD": 0.98},
    }),
    encoding="utf-8",
)
(output / "postprocess_failures.json").write_text("[]\\n", encoding="utf-8")
print(json.dumps({"unknown_argv": unknown}, sort_keys=True))
'''


class IonicMDHandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.project = Path(self.temporary.name).resolve()
        self.attempt = self.project / "attempt"
        self.attempt.mkdir()
        self.md_script = self.project / "ase_md_only_multi_calc.py"
        self.analysis_script = self.project / "ionic_conductivity.py"
        self.structure = self.project / "structure.xyz"
        self.md_script.write_text(FAKE_MD_SOURCE, encoding="utf-8")
        self.analysis_script.write_text(FAKE_ANALYSIS_SOURCE, encoding="utf-8")
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
                "analysis_script": str(self.analysis_script),
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
                "charge": 1.0,
                "dimensions": 3,
                "haven_ratio": 1.0,
                "fit_start_ps": 0.0,
                "fit_end_ps": 0.008,
                "trajectory_start_ps": None,
                "trajectory_end_ps": None,
                "drift_correction": "none",
                "msd_mode": "single-origin",
                "trajectory_msd_engine": "numpy",
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
                "min_segment_points": 2,
                "piecewise_slope_change": 0.35,
                "piecewise_bic_delta": 2.0,
                "allow_partial_results": False,
            },
            "backend": "local",
            "resources": {"python_executable": sys.executable},
        }

    def test_committed_real_mace_smoke_is_code_bound_and_non_scientific(self) -> None:
        report_path = ROOT / "reports" / "ionic_md_local_integration_smoke.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for name in ("adapter", "handoff"):
            record = report["implementation"][name]
            source = ROOT / record["locator"]
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), record["sha256"])

        self.assertEqual("LOCAL_INTEGRATION_SMOKE_PASS", report["status"])
        self.assertTrue(report["scientific_claim"]["real_mlip_model_loaded"])
        self.assertTrue(report["scientific_claim"]["integration_handoff_completed"])
        self.assertFalse(report["scientific_claim"]["production_md_executed"])
        self.assertFalse(report["scientific_claim"]["scientific_parity_established"])
        self.assertFalse(report["scientific_claim"]["manuscript_result_recomputed"])
        self.assertFalse(report["external_inputs"]["model"]["included_in_repository"])
        self.assertEqual(3, report["result"]["run_count"])
        self.assertEqual(0, report["result"]["failed_run_count"])
        self.assertEqual("OK", report["result"]["adapter_check_status"])
        self.assertEqual("OK", report["result"]["adapter_collect_status"])
        self.assertLessEqual(report["bounded_parameters"]["production_steps_per_temperature"], 10)
        self.assertLessEqual(report["bounded_parameters"]["production_time_ps"], 0.01)
        for digest in report["result"]["required_analysis_artifacts"].values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn("/Users/", serialized)
        self.assertNotIn("/public/home/", serialized)

    def test_fake_historical_sources_complete_the_confined_handoff(self) -> None:
        context = self.smoke_context()
        immutable = [self.md_script, self.analysis_script, self.structure]
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in immutable}
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
            self.assertTrue(all(item["path_is_attempt_relative"] for item in manifest[group_name]))
        self.assertTrue(
            {
                "diffusion_results_by_temperature.csv",
                "arrhenius_summary.json",
                "postprocess_failures.json",
            }.issubset({Path(item["path"]).name for item in manifest["result_artifacts"]})
        )
        self.assertTrue(all(item["sha256"].startswith("sha256:") for item in manifest["source_artifacts"]))
        self.assertTrue(all(item["sha256"].startswith("sha256:") for item in manifest["input_artifacts"]))
        self.assertTrue(all(item["sha256"].startswith("sha256:") for item in manifest["result_artifacts"]))
        self.assertTrue(all(item["absolute_path_recorded"] is False for item in manifest["source_artifacts"]))
        self.assertTrue(all(item["portable"] is False for item in manifest["source_artifacts"]))
        self.assertFalse(manifest["execution"]["absolute_paths_recorded"])
        self.assertTrue(
            all(Path(path).resolve().is_relative_to(self.attempt) for path in plan["expected_outputs"])
        )
        self.assertEqual(
            before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in immutable}
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
        first_trajectory.write_bytes(b"tampered")
        tampered = self.adapter.check(context)
        self.assertEqual("FAIL", tampered["status"])
        self.assertIn(
            "integration.artifact_sha256", {item["code"] for item in tampered["diagnostics"]}
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
            "analysis_script": str(self.analysis_script),
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
        parameters["n_mobile_ions"] = 1
        parameters["volume_a3"] = 100.0
        plan = self.adapter.plan(context)
        self.assertEqual("READY", plan["status"], plan.get("diagnostics"))
        self.assertEqual("analyze-existing", plan["operation"])


if __name__ == "__main__":
    unittest.main()
