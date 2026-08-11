"""Dependency-free contract tests for the two reviewed claw CLI adapters."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]


def load_adapter(plugin_id: str) -> Any:
    path = ROOT / "plugins" / plugin_id / "adapter.py"
    spec = importlib.util.spec_from_file_location(
        "test_%s_adapter" % plugin_id.replace("-", "_"), path
    )
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Adapter()


def diagnostic_codes(result: Any) -> set[str]:
    diagnostics = result if isinstance(result, list) else result.get("diagnostics", [])
    return {item["code"] for item in diagnostics}


class DirectAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.script = self.root / "direct.py"
        self.script.write_text("# reviewed CLI fixture\n", encoding="utf-8")
        self.inputs = self.root / "structures"
        self.inputs.mkdir()
        (self.inputs / "POSCAR").write_text("fixture\n", encoding="utf-8")
        self.attempt = self.root / ".mlipflow" / "runs" / "sample" / "attempt-1"
        self.adapter = load_adapter("pes-sampling")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def context(self) -> dict[str, Any]:
        return {
            "project_root": str(self.root),
            "attempt_dir": str(self.attempt),
            "inputs": {
                "direct_script": str(self.script),
                "input_dirs": [str(self.inputs)],
            },
            "parameters": {
                "output_subdir": "direct-selected",
                "globs": ["POSCAR*", "*.extxyz"],
                "recursive": False,
                "stride": 5,
                "max_frames_per_file": 100,
                "max_input_structures": 1000,
                "n_clusters": 20,
                "threshold_init": 0.05,
                "k_per_cluster": 1,
                "lammps_type_map": {1: "Li", 2: "S"},
                "seed": 17,
                "acknowledge_uncontrolled_seed": True,
            },
            "backend": "local",
            "resources": {"python_executable": "python3"},
        }

    def test_plan_is_pure_shell_false_exact_argv(self) -> None:
        context = self.context()
        before = {path.relative_to(self.root).as_posix() for path in self.root.rglob("*")}
        diagnostics = self.adapter.validate(context)
        plan = self.adapter.plan(context)
        after = {path.relative_to(self.root).as_posix() for path in self.root.rglob("*")}

        self.assertEqual(before, after)
        self.assertFalse(self.attempt.exists())
        self.assertEqual("READY", plan["status"])
        self.assertTrue(plan["executable"])
        self.assertFalse(plan["shell"])
        self.assertIsInstance(plan["argv"], list)
        self.assertTrue(all(isinstance(value, str) for value in plan["argv"]))
        self.assertEqual("python3", plan["argv"][0])
        self.assertEqual(str(self.script), plan["argv"][1])
        self.assertIn("--no-recursive", plan["argv"])
        self.assertEqual('{"1":"Li","2":"S"}', plan["argv"][-1])
        self.assertIn("seed.provenance_only", diagnostic_codes(diagnostics))
        self.assertEqual("not-exposed-by-source-cli", plan["assumptions"]["seed_control"])

    def test_uncontrolled_seed_needs_explicit_acknowledgement(self) -> None:
        context = self.context()
        context["parameters"]["acknowledge_uncontrolled_seed"] = False
        plan = self.adapter.plan(context)
        self.assertEqual("BLOCKED", plan["status"])
        self.assertIn("seed.no_cli_control", diagnostic_codes(plan))

    def test_unknown_parameter_is_not_silently_ignored(self) -> None:
        context = self.context()
        context["parameters"]["cluster_count"] = 99
        plan = self.adapter.plan(context)
        self.assertEqual("BLOCKED", plan["status"])
        self.assertIn("parameter.unknown", diagnostic_codes(plan))

    def test_existing_output_blocks_without_deleting_it(self) -> None:
        output = self.attempt / "direct-selected"
        output.mkdir(parents=True)
        sentinel = output / "keep.txt"
        sentinel.write_text("do not delete", encoding="utf-8")
        plan = self.adapter.plan(self.context())
        self.assertEqual("BLOCKED", plan["status"])
        self.assertIn("path.output_exists", diagnostic_codes(plan))
        self.assertEqual("do not delete", sentinel.read_text(encoding="utf-8"))

    def result_context(self) -> Tuple[Dict[str, Any], Path, Path]:
        context = self.context()
        output = self.attempt / "direct-selected"
        output.mkdir(parents=True)
        selected = output / "00001_Li2S.vasp"
        selected.write_text("Li2S fixture\n", encoding="utf-8")
        manifest = output / "manifest.csv"
        manifest.write_text(
            "selected_order,input_index,source_kind,source_file,frame_index,output_file,formula\n"
            "1,4,single_structure,%s,0,%s,Li2S\n" % (self.inputs / "POSCAR", selected),
            encoding="utf-8",
        )
        context["execution"] = {
            "returncode": 0,
            "plan": {
                "expected_outputs": [str(manifest)],
                "output_dir": str(output),
            },
        }
        return context, manifest, selected

    def test_check_and_collect_read_only_explicit_manifest(self) -> None:
        context, manifest, selected = self.result_context()
        before = {
            path: (path.stat().st_mtime_ns, path.read_bytes()) for path in (manifest, selected)
        }
        checked = self.adapter.check(context)
        collected = self.adapter.collect(context)
        after = {
            path: (path.stat().st_mtime_ns, path.read_bytes()) for path in (manifest, selected)
        }

        self.assertEqual(before, after)
        self.assertEqual("OK", checked["status"])
        self.assertEqual("OK", collected["status"])
        self.assertEqual(1, collected["metrics"]["selected_structure_count"])
        self.assertEqual(
            {"sample-manifest", "selected-structure"},
            {artifact["role"] for artifact in collected["artifacts"]},
        )

    def test_missing_explicit_manifest_waits(self) -> None:
        context = self.context()
        manifest = self.attempt / "direct-selected" / "manifest.csv"
        context["execution"] = {
            "returncode": 0,
            "plan": {"expected_outputs": [str(manifest)], "output_dir": str(manifest.parent)},
        }
        self.assertEqual("WAIT", self.adapter.check(context)["status"])


class IonicTransportAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.script = self.root / "ionic_conductivity.py"
        self.script.write_text("# reviewed CLI fixture\n", encoding="utf-8")
        self.input_file = self.root / "msd.csv"
        self.input_file.write_text("time_ps,msd_A2\n0,0\n1,1\n", encoding="utf-8")
        self.attempt = self.root / ".mlipflow" / "runs" / "transport" / "attempt-1"
        self.adapter = load_adapter("ionic-transport")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def context(self) -> dict[str, Any]:
        return {
            "project_root": str(self.root),
            "attempt_dir": str(self.attempt),
            "inputs": {
                "analysis_script": str(self.script),
                "input_paths": [str(self.input_file)],
            },
            "parameters": {
                "output_subdir": "ionic-transport-postprocess",
                "source": "msd",
                "specie": "Li",
                "charge": 1.0,
                "dimensions": 3,
                "haven_ratio": 1.0,
                "fit_start_ps": 10.0,
                "fit_end_ps": 100.0,
                "trajectory_start_ps": None,
                "trajectory_end_ps": None,
                "drift_correction": "framework",
                "msd_mode": "multi-origin",
                "trajectory_msd_engine": "numpy",
                "diffusion_analyzer_smoothed": "max",
                "diffusion_analyzer_min_obs": 30,
                "diffusion_analyzer_avg_nsteps": 1000,
                "diffusion_analyzer_step_skip": 1,
                "min_msd_fit_points": 3,
                "msd_smooth_window_points": None,
                "msd_smooth_window_ps": None,
                "fit_smoothed_msd": False,
                "msd_time_unit": "ps",
                "msd_unit": "A2",
                "msd_temperature_k": 800.0,
                "n_mobile_ions": 24,
                "volume_a3": 1200.0,
                "fit_scope": "dataset",
                "target_temperature_k": 300.0,
                "piecewise": "never",
                "min_segment_points": 3,
                "piecewise_slope_change": 0.35,
                "piecewise_bic_delta": 2.0,
                "allow_partial_results": False,
            },
            "backend": "local",
            "resources": {"python_executable": "python3"},
        }

    def test_plan_is_pure_and_records_fixed_scientific_assumptions(self) -> None:
        context = self.context()
        before = {path.relative_to(self.root).as_posix() for path in self.root.rglob("*")}
        plan = self.adapter.plan(context)
        after = {path.relative_to(self.root).as_posix() for path in self.root.rglob("*")}

        self.assertEqual(before, after)
        self.assertFalse(self.attempt.exists())
        self.assertEqual("READY", plan["status"])
        self.assertTrue(plan["executable"])
        self.assertFalse(plan["shell"])
        self.assertIsInstance(plan["argv"], list)
        self.assertIn("--fit-start-ps", plan["argv"])
        self.assertIn("--fit-end-ps", plan["argv"])
        self.assertIn("--n-mobile-ions", plan["argv"])
        self.assertEqual(3, plan["assumptions"]["dimensions"])
        self.assertEqual(1.0, plan["assumptions"]["haven_ratio"])
        self.assertIsNone(plan["assumptions"]["seed"])

    def test_unsupported_dimension_haven_and_seed_block(self) -> None:
        for key, value, code in (
            ("dimensions", 2, "assumption.dimensions_unsupported"),
            ("haven_ratio", 0.7, "assumption.haven_unsupported"),
            ("seed", 1, "parameter.seed_unsupported"),
        ):
            with self.subTest(parameter=key):
                context = self.context()
                context["parameters"][key] = value
                plan = self.adapter.plan(context)
                self.assertEqual("BLOCKED", plan["status"])
                self.assertIn(code, diagnostic_codes(plan))

    def test_invalid_fit_window_and_implicit_msd_units_block(self) -> None:
        context = self.context()
        context["parameters"]["fit_end_ps"] = 5.0
        context["parameters"]["msd_time_unit"] = "auto"
        plan = self.adapter.plan(context)
        self.assertEqual("BLOCKED", plan["status"])
        self.assertIn("parameter.fit_start_ps_fit_end_ps", diagnostic_codes(plan))
        self.assertIn("unit.msd_time_explicit", diagnostic_codes(plan))

    def test_unknown_parameter_is_not_silently_ignored(self) -> None:
        context = self.context()
        context["parameters"]["fit_start"] = 10.0
        plan = self.adapter.plan(context)
        self.assertEqual("BLOCKED", plan["status"])
        self.assertIn("parameter.unknown", diagnostic_codes(plan))

    def test_existing_output_blocks_without_overwrite(self) -> None:
        output = self.attempt / "ionic-transport-postprocess"
        output.mkdir(parents=True)
        sentinel = output / "arrhenius_summary.json"
        sentinel.write_text('{"old": true}\n', encoding="utf-8")
        plan = self.adapter.plan(self.context())
        self.assertEqual("BLOCKED", plan["status"])
        self.assertIn("path.output_exists", diagnostic_codes(plan))
        self.assertEqual({"old": True}, json.loads(sentinel.read_text(encoding="utf-8")))

    def result_context(
        self, failures: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Dict[str, Any], List[Path]]:
        context = self.context()
        output = self.attempt / "ionic-transport-postprocess"
        curve = output / "msd_curves" / "fixture_msd.csv"
        fit = output / "msd_fits" / "fixture_msd_fit.html"
        curve.parent.mkdir(parents=True)
        fit.parent.mkdir(parents=True)
        curve.write_text("time_ps,msd_A2\n10,1\n100,10\n", encoding="utf-8")
        fit.write_text("<html>fit</html>\n", encoding="utf-8")
        results = output / "diffusion_results_by_temperature.csv"
        results.write_text(
            "temperature_K,diffusivity_cm2_s,fit_start_ps,fit_end_ps,msd_fit_r2,"
            "n_mobile_ions,charge,volume_A3,conductivity_NE_mS_cm,msd_curve_csv,msd_fit_html\n"
            "800,1e-6,10,100,0.99,24,1,1200,1.5,%s,%s\n" % (curve, fit),
            encoding="utf-8",
        )
        summary = output / "arrhenius_summary.json"
        summary.write_text(
            json.dumps(
                {
                    "fit_scope": "dataset",
                    "specie": "Li",
                    "arrhenius_fit_skipped": True,
                    "skip_reason": "one temperature",
                }
            ),
            encoding="utf-8",
        )
        failures_path = output / "postprocess_failures.json"
        failures_path.write_text(json.dumps(failures or []), encoding="utf-8")
        expected = [results, summary, failures_path]
        context["execution"] = {
            "returncode": 0,
            "plan": {
                "expected_outputs": [str(path) for path in expected],
                "output_dir": str(output),
            },
        }
        return context, [*expected, curve, fit]

    def test_check_and_collect_are_read_only_and_standardized(self) -> None:
        context, files = self.result_context()
        before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in files}
        checked = self.adapter.check(context)
        collected = self.adapter.collect(context)
        after = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in files}

        self.assertEqual(before, after)
        self.assertEqual("OK", checked["status"])
        self.assertEqual("OK", collected["status"])
        self.assertEqual(1, collected["metrics"]["run_count"])
        self.assertEqual(3, collected["metrics"]["assumptions"]["dimensions"])
        self.assertEqual(1.0, collected["metrics"]["assumptions"]["haven_ratio"])
        self.assertEqual(
            {
                "transport-results",
                "arrhenius-summary",
                "postprocess-failures",
                "msd-curve",
                "msd-fit",
            },
            {artifact["role"] for artifact in collected["artifacts"]},
        )

    def test_partial_results_are_explicitly_opted_in(self) -> None:
        context, _ = self.result_context([{"run_dir": "bad", "error": "fixture"}])
        self.assertEqual("FAIL", self.adapter.check(context)["status"])
        context["parameters"]["allow_partial_results"] = True
        checked = self.adapter.check(context)
        self.assertEqual("OK", checked["status"])
        self.assertIn("result.partial_allowed", diagnostic_codes(checked))

    def test_missing_explicit_results_wait(self) -> None:
        context = self.context()
        output = self.attempt / "ionic-transport-postprocess"
        context["execution"] = {
            "returncode": 0,
            "plan": {
                "expected_outputs": [
                    str(output / name)
                    for name in (
                        "diffusion_results_by_temperature.csv",
                        "arrhenius_summary.json",
                        "postprocess_failures.json",
                    )
                ],
                "output_dir": str(output),
            },
        }
        self.assertEqual("WAIT", self.adapter.check(context)["status"])


if __name__ == "__main__":
    unittest.main()
