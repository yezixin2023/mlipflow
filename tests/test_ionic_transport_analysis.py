from __future__ import annotations

import csv
from tests.helpers import load_module
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "mlipflow" / "plugins" / "ionic_transport" / "adapter.py"
RUNNER_PATH = ROOT / "mlipflow" / "plugins" / "ionic_transport" / "ionic_conductivity.py"


def load_adapter():
    module = load_adapter_module()
    return module.Adapter()


def load_adapter_module():
    module = load_module(ADAPTER_PATH, 'ionic_transport_analysis_adapter')
    return module


def load_runner():
    name = "test_ionic_transport_formal_runner"
    module = load_module(RUNNER_PATH, name)
    return module


def write_lammps_run(root: Path, temperature_k: int, displacement_per_frame: float) -> None:
    run = root / f"T{temperature_k}"
    run.mkdir(parents=True)
    (run / "data.LYC").write_text(
        "2 atoms\n"
        "2 atom types\n\n"
        "0 10 xlo xhi\n"
        "0 10 ylo yhi\n"
        "0 10 zlo zhi\n\n"
        "Masses\n\n"
        "1 6.94 # Li\n"
        "2 4.00 # He\n\n"
        "Atoms # atomic\n\n"
        "1 1 0 0 0\n"
        "2 2 5 5 5\n",
        encoding="utf-8",
    )
    blocks = []
    for frame in range(6):
        blocks.append(
            "ITEM: TIMESTEP\n"
            f"{frame}\n"
            "ITEM: NUMBER OF ATOMS\n"
            "2\n"
            "ITEM: BOX BOUNDS pp pp pp\n"
            "0 10\n0 10\n0 10\n"
            "ITEM: ATOMS id type element xu yu zu\n"
            f"1 1 Li {frame * displacement_per_frame:.12g} 0 0\n"
            "2 2 He 5 5 5\n"
        )
    (run / "traj.lammpstrj").write_text("".join(blocks), encoding="utf-8")


def write_aimd_rdf_run(root: Path) -> None:
    run = root / "T600"
    run.mkdir(parents=True)
    frames = []
    for frame in range(6):
        li_x = 0.10 + 0.01 * frame
        frames.append(
            "<calculation><structure><crystal><varray name=\"basis\">"
            "<v>10 0 0</v><v>0 10 0</v><v>0 0 10</v>"
            "</varray></crystal><varray name=\"positions\">"
            f"<v>{li_x:.8f} 0 0</v><v>0.5 0 0</v>"
            "</varray></structure></calculation>"
        )
    (run / "vasprun.xml").write_text(
        "<modeling><incar>"
        '<i name="POTIM">1.0</i><i name="TEBEG">600</i><i name="TEEND">600</i>'
        "</incar><atominfo><array name=\"atoms\"><set>"
        "<rc><c>Li</c></rc><rc><c>S</c></rc>"
        "</set></array></atominfo>"
        + "".join(frames)
        + "</modeling>",
        encoding="utf-8",
    )


def write_mlip_rdf_run(root: Path) -> None:
    run = root / "T600"
    run.mkdir(parents=True)
    (run / "data.LYC").write_text(
        "2 atoms\n2 atom types\n\n"
        "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n"
        "Masses\n\n1 6.94 # Li\n2 32.06 # S\n\n"
        "Atoms # atomic\n\n1 1 1 0 0\n2 2 5 0 0\n",
        encoding="utf-8",
    )
    blocks = []
    for frame in range(6):
        blocks.append(
            "ITEM: TIMESTEP\n"
            f"{frame}\n"
            "ITEM: NUMBER OF ATOMS\n2\n"
            "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
            "ITEM: ATOMS id type element xu yu zu\n"
            f"1 1 Li {1.0 + 0.2 * frame:.8f} 0 0\n"
            "2 2 S 5 0 0\n"
        )
    (run / "traj.lammpstrj").write_text("".join(blocks), encoding="utf-8")


class IonicTransportTrajectoryRegressionTests(unittest.TestCase):
    def test_target_temperature_conductivity_is_exposed_only_from_available_row(self) -> None:
        import pandas as pd

        runner = load_runner()
        rows = pd.DataFrame(
            {
                "temperature_K": [400.0, 600.0, 800.0],
                "diffusivity_cm2_s": [1.0e-7, 2.0e-7, 4.0e-7],
                "conductivity_NE_mS_cm": [0.5, 1.25, 2.5],
            }
        )
        summary = runner.summarize_arrhenius_one(
            rows,
            SimpleNamespace(
                specie="Li",
                fit_temperatures=None,
                exclude_temperatures=None,
                target_temperature_K=600.0,
                piecewise="never",
                min_segment_points=3,
                piecewise_slope_change=0.35,
                piecewise_bic_delta=2.0,
            ),
        )
        direct = summary["target_prediction"]["direct_simulation"]
        self.assertTrue(direct["available"])
        self.assertEqual(1.25, direct["conductivity_mS_cm"])
        self.assertEqual("single", summary["selected_model"])
        self.assertIsNone(summary["piecewise"])
        self.assertIn("D_600K_cm2_s", summary["single"])

    def test_aimd_and_mlip_share_transport_and_rdf_comparison_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary).resolve()
            aimd = project / "aimd"
            mlip = project / "mlip"
            write_aimd_rdf_run(aimd)
            write_mlip_rdf_run(mlip)
            attempt = project / ".mlipflow" / "runs" / "compare" / "attempt-1"
            context = {
                "project_root": str(project),
                "attempt_dir": str(attempt),
                "inputs": {"input_paths": [str(aimd), str(mlip)]},
                "parameters": {
                    "operation": "analyze-existing",
                    "output_subdir": "analysis",
                    "source": "trajectory",
                    "specie": "Li",
                    "seed": None,
                    "trajectory_start_ps": None,
                    "trajectory_end_ps": None,
                    "trajectory_msd_engine": "diffusion-analyzer",
                    "diffusion_analyzer_smoothed": "none",
                    "diffusion_analyzer_min_obs": 1,
                    "diffusion_analyzer_avg_nsteps": 2,
                    "diffusion_analyzer_step_skip": 1,
                    "min_msd_fit_points": 3,
                    "msd_smooth_window_points": None,
                    "msd_smooth_window_ps": None,
                    "fit_smoothed_msd": False,
                    "ase_frame_step_fs": None,
                    "lammps_timestep_ps": 0.001,
                    "lammps_data_name": "data.LYC",
                    "vasp_file_name": "vasprun.xml",
                    "vasp_step_fs": None,
                    "temperature_k": None,
                    "aimd_temperature_k": None,
                    "mobile_type": None,
                    "msd_file_name": None,
                    "msd_time_column": None,
                    "msd_column": None,
                    "msd_time_unit": "auto",
                    "msd_step_ps": None,
                    "msd_unit": "auto",
                    "msd_temperature_k": None,
                    "n_mobile_ions": None,
                    "volume_a3": None,
                    "fit_temperatures": None,
                    "exclude_temperatures": None,
                    "fit_scope": "dataset",
                    "target_temperature_k": 600.0,
                    "piecewise": "never",
                    "min_segment_points": 3,
                    "piecewise_slope_change": 0.35,
                    "piecewise_bic_delta": 2.0,
                    "rdf_pair": ["Li", "S"],
                    "rdf_r_min_angstrom": 0.0,
                    "rdf_r_max_angstrom": 4.5,
                    "rdf_bins": 9,
                    "rdf_temperature_k": 600.0,
                    "comparison_model": "deepmd-dpa2",
                    "comparison_scenario": "aimd-rdf-smoke",
                    "comparison_split": "window-0-5fs",
                    "allow_partial_results": False,
                },
                "backend": "local",
                "resources": {"python_executable": sys.executable},
            }
            adapter = load_adapter()
            plan = adapter.plan(context)
            self.assertEqual("READY", plan["status"], plan.get("diagnostics"))
            self.assertIn("--rdf-pair", plan["argv"])
            attempt.mkdir(parents=True)
            completed = subprocess.run(
                plan["argv"], cwd=plan["cwd"], check=False, capture_output=True, text=True
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
            context["execution"] = {"returncode": completed.returncode, "plan": plan}
            checked = adapter.check(context)
            collected = adapter.collect(context)
            self.assertEqual("OK", checked["status"], checked.get("diagnostics"))
            self.assertEqual("OK", collected["status"], collected.get("diagnostics"))
            roles = {item["role"] for item in collected["artifacts"]}
            self.assertIn("rdf-curves", roles)
            self.assertIn("aimd-mlip-comparison", roles)

            comparison = json.loads(
                (Path(plan["output_dir"]) / "aimd_mlip_comparison.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(["Li", "S"], comparison["rdf"]["pair"])
            self.assertGreater(comparison["rdf"]["curve_rmse"], 0.0)
            self.assertEqual(
                {"structural-dynamics", "ionic-transport"},
                {record["task"] for record in comparison["records"]},
            )
            self.assertTrue(
                all(record["model"] == "deepmd-dpa2" for record in comparison["records"])
            )
            self.assertNotIn(
                "activation_energy",
                {record["target"] for record in comparison["records"]},
            )
            self.assertTrue(comparison["transport_errors"])
            self.assertTrue(
                all("relative_error" in item for item in comparison["transport_errors"])
            )

            benchmark_path = ROOT / "mlipflow" / "plugins" / "mlip_benchmark" / "adapter.py"
            benchmark_module = load_module(benchmark_path, 'aimd_comparison_benchmark_adapter')
            benchmark_attempt = project / ".mlipflow" / "runs" / "benchmark" / "attempt-1"
            benchmark_attempt.mkdir(parents=True)
            benchmark_context = {
                "project_root": str(project),
                "attempt_dir": str(benchmark_attempt),
                "inputs": {
                    "evidence_inputs": [
                        {
                            "path": str(Path(plan["output_dir"]) / "aimd_mlip_comparison.json"),
                            "evidence_locator": "comparison/deepmd-dpa2.json",
                        }
                    ],
                    "output_dir": "normalized",
                },
                "parameters": {
                    "operation": "normalize-execute",
                    "expected_models": ["deepmd-dpa2"],
                    "expected_tasks": ["structural-dynamics", "ionic-transport"],
                    "scenario": "aimd-rdf-smoke",
                    "split": "window-0-5fs",
                },
                "backend": "local",
                "resources": {"cpus": 1},
            }
            benchmark = benchmark_module.Adapter()
            benchmark_plan = benchmark.plan(benchmark_context)
            self.assertEqual(
                "READY", benchmark_plan["status"], benchmark_plan.get("diagnostics")
            )
            benchmark_run = subprocess.run(
                benchmark_plan["argv"], check=False, capture_output=True, text=True
            )
            self.assertEqual(0, benchmark_run.returncode, benchmark_run.stderr)
            benchmark_checked = benchmark.check(benchmark_context)
            self.assertEqual(
                "OK", benchmark_checked["status"], benchmark_checked.get("diagnostics")
            )

    def test_smoothed_max_trajectory_matches_diffusion_analyzer_and_checker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary).resolve()
            trajectories = project / "trajectories"
            for temperature, displacement in ((400, 0.10), (600, 0.16), (800, 0.23)):
                write_lammps_run(trajectories, temperature, displacement)
            attempt = project / ".mlipflow" / "runs" / "transport" / "attempt-0001"
            context = {
                "project_root": str(project),
                "attempt_dir": str(attempt),
                "inputs": {"input_paths": [str(trajectories)]},
                "parameters": {
                    "operation": "analyze-existing",
                    "output_subdir": "ionic-transport-postprocess",
                    "source": "trajectory",
                    "specie": "Li",
                    "seed": None,
                    "trajectory_start_ps": None,
                    "trajectory_end_ps": None,
                    "trajectory_msd_engine": "diffusion-analyzer",
                    "diffusion_analyzer_smoothed": "max",
                    "diffusion_analyzer_min_obs": 1,
                    "diffusion_analyzer_avg_nsteps": 3,
                    "diffusion_analyzer_step_skip": 1,
                    "min_msd_fit_points": 3,
                    "msd_smooth_window_points": None,
                    "msd_smooth_window_ps": None,
                    "fit_smoothed_msd": False,
                    "ase_frame_step_fs": None,
                    "lammps_timestep_ps": 1.0,
                    "lammps_data_name": "data.LYC",
                    "vasp_file_name": "vasprun.xml",
                    "vasp_step_fs": None,
                    "temperature_k": None,
                    "aimd_temperature_k": None,
                    "mobile_type": None,
                    "msd_file_name": None,
                    "msd_time_column": None,
                    "msd_column": None,
                    "msd_time_unit": "auto",
                    "msd_step_ps": None,
                    "msd_unit": "auto",
                    "msd_temperature_k": None,
                    "n_mobile_ions": None,
                    "volume_a3": None,
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
            adapter = load_adapter()
            plan = adapter.plan(context)
            self.assertEqual("READY", plan["status"], plan.get("diagnostics"))
            attempt.mkdir(parents=True)
            completed = subprocess.run(
                plan["argv"], cwd=plan["cwd"], check=False, capture_output=True, text=True
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
            context["execution"] = {"returncode": completed.returncode, "plan": plan}
            checked = adapter.check(context)
            collected = adapter.collect(context)
            self.assertEqual("OK", checked["status"], checked.get("diagnostics"))
            self.assertEqual("OK", collected["status"], collected.get("diagnostics"))
            self.assertEqual(3, collected["metrics"]["run_count"])
            with (Path(plan["output_dir"]) / "diffusion_results_by_temperature.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([400.0, 600.0, 800.0], [float(row["temperature_K"]) for row in rows])
            self.assertTrue(all(float(row["diffusivity_cm2_s"]) > 0.0 for row in rows))
            self.assertTrue(
                all(row["diffusion_method"] == "pymatgen-diffusion-analyzer" for row in rows)
            )
            self.assertTrue(
                all(
                    row["conductivity_method"] == "pymatgen-diffusion-analyzer"
                    for row in rows
                )
            )

            from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer, fit_arrhenius
            from pymatgen.core import Lattice, Structure

            expected_analyzers = []
            for temperature, displacement in ((400, 0.10), (600, 0.16), (800, 0.23)):
                structures = [
                    Structure(
                        Lattice.cubic(10),
                        ["Li", "He"],
                        [[frame * displacement / 10.0, 0, 0], [0.5, 0.5, 0.5]],
                    )
                    for frame in range(6)
                ]
                expected_analyzers.append(
                    DiffusionAnalyzer.from_structures(
                        structures,
                        specie="Li",
                        temperature=temperature,
                        time_step=1000.0,
                        step_skip=1,
                        smoothed="max",
                        min_obs=1,
                        avg_nsteps=3,
                    )
                )
            for row, analyzer in zip(rows, expected_analyzers):
                self.assertTrue(
                    math.isclose(
                        float(row["diffusivity_cm2_s"]),
                        float(analyzer.diffusivity),
                        rel_tol=1e-12,
                    )
                )
                self.assertTrue(
                    math.isclose(
                        float(row["conductivity_NE_mS_cm"]),
                        float(analyzer.conductivity),
                        rel_tol=1e-12,
                    )
                )
                with Path(row["msd_curve_csv"]).open(encoding="utf-8", newline="") as stream:
                    curve = list(csv.DictReader(stream))
                self.assertEqual(
                    [float(item["time_ps"]) for item in curve],
                    [float(value) / 1000.0 for value in analyzer.dt],
                )

            summary = json.loads(
                (Path(plan["output_dir"]) / "arrhenius_summary.json").read_text(encoding="utf-8")
            )
            expected_ea, expected_prefactor, expected_std = fit_arrhenius(
                [400.0, 600.0, 800.0],
                [float(analyzer.diffusivity) for analyzer in expected_analyzers],
                mode="linear",
            )
            self.assertEqual("pymatgen-fit-arrhenius-linear", summary["arrhenius_method"])
            self.assertTrue(math.isclose(expected_ea, summary["single"]["Ea_eV"], rel_tol=1e-12))
            self.assertTrue(
                math.isclose(expected_prefactor, summary["single"]["D0_cm2_s"], rel_tol=1e-12)
            )
            self.assertTrue(
                math.isclose(expected_std, summary["single"]["Ea_stderr_eV"], rel_tol=1e-12)
            )

    def test_step_skip_time_axis_comes_from_diffusion_analyzer_dt(self) -> None:
        from pymatgen.core import Lattice, Structure

        runner = load_runner()
        structures = [
            Structure(
                Lattice.cubic(10),
                ["Li", "He"],
                [[frame * 0.01, 0, 0], [0.5, 0.5, 0.5]],
            )
            for frame in range(9)
        ]
        args = SimpleNamespace(
            diffusion_analyzer_step_skip=2,
            diffusion_analyzer_smoothed="none",
            diffusion_analyzer_min_obs=3,
            diffusion_analyzer_avg_nsteps=3,
        )
        analyzer, time_ps, _, _ = runner.diffusion_analyzer_from_structures(
            structures, "Li", 600.0, 4.0, args
        )
        self.assertEqual(time_ps.tolist(), (analyzer.dt / 1000.0).tolist())
        self.assertEqual([0.0, 0.008, 0.016, 0.024, 0.032], time_ps.tolist())

    def test_all_mobile_trajectory_requires_framework_atom(self) -> None:
        from pymatgen.core import Lattice, Structure

        runner = load_runner()
        structures = [
            Structure(
                Lattice.cubic(10),
                ["Li", "Li"],
                [[0, 0, 0], [0.5 + frame * 0.01, 0.5, 0.5]],
            )
            for frame in range(2)
        ]
        args = SimpleNamespace(
            diffusion_analyzer_step_skip=1,
            diffusion_analyzer_smoothed="none",
            diffusion_analyzer_min_obs=1,
            diffusion_analyzer_avg_nsteps=1,
        )

        with self.assertRaisesRegex(ValueError, "non-Li framework atom"):
            runner.diffusion_analyzer_from_structures(
                structures, "Li", 600.0, 1.0, args
            )


class IonicTransportArrheniusBreakpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = load_runner()
        self.temperatures = [300.0, 350.0, 400.0, 450.0, 550.0, 650.0, 750.0, 850.0]

    @staticmethod
    def diffusivity(temperature_K: float, ea_eV: float, prefactor_cm2_s: float) -> float:
        from scipy import constants

        return prefactor_cm2_s * math.exp(
            -ea_eV / ((constants.k / constants.e) * temperature_K)
        )

    def args(
        self,
        *,
        piecewise: str = "auto",
        target_temperature_K: float = 300.0,
        piecewise_bic_delta: float = 2.0,
        piecewise_slope_change: float = 0.35,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            specie="Li",
            fit_temperatures=None,
            exclude_temperatures=None,
            target_temperature_K=target_temperature_K,
            piecewise=piecewise,
            min_segment_points=3,
            piecewise_slope_change=piecewise_slope_change,
            piecewise_bic_delta=piecewise_bic_delta,
            fit_scope="all",
        )

    def summary(self, diffusivities, **argument_overrides):
        frame = self.runner.pd.DataFrame(
            {
                "temperature_K": self.temperatures,
                "diffusivity_cm2_s": diffusivities,
                "conductivity_NE_mS_cm": [None] * len(self.temperatures),
            }
        )
        return self.runner.summarize_arrhenius_one(
            frame, self.args(**argument_overrides)
        )

    def two_regime_diffusivities(self):
        return [
            self.diffusivity(temperature, 0.2, 1.0e-3)
            if temperature <= 450.0
            else self.diffusivity(temperature, 0.6, 1.0e-2)
            for temperature in self.temperatures
        ]

    def test_single_arrhenius_data_selects_single_baseline(self) -> None:
        diffusivities = [
            self.diffusivity(temperature, 0.3, 1.0e-3)
            for temperature in self.temperatures
        ]

        summary = self.summary(diffusivities)

        self.assertEqual("single", summary["selected_model"])
        self.assertFalse(summary["regime_change_detected"])
        self.assertIsNotNone(summary["piecewise"])
        self.assertIn("BIC", summary["single"])

    def test_clear_two_regime_data_finds_one_breakpoint_and_two_activation_energies(self) -> None:
        summary = self.summary(self.two_regime_diffusivities())

        self.assertEqual("piecewise", summary["selected_model"])
        self.assertTrue(summary["regime_change_detected"])
        self.assertEqual([450.0, 550.0], summary["piecewise"]["breakpoint_interval_K"])
        self.assertAlmostEqual(
            0.2, summary["piecewise"]["low_temperature"]["Ea_eV"], places=12
        )
        self.assertAlmostEqual(
            0.6, summary["piecewise"]["high_temperature"]["Ea_eV"], places=12
        )

    def test_insufficient_temperatures_keep_single_with_explicit_status(self) -> None:
        temperatures = self.temperatures[:5]
        frame = self.runner.pd.DataFrame(
            {
                "temperature_K": temperatures,
                "diffusivity_cm2_s": [
                    self.diffusivity(temperature, 0.3, 1.0e-3)
                    for temperature in temperatures
                ],
                "conductivity_NE_mS_cm": [None] * len(temperatures),
            }
        )

        summary = self.runner.summarize_arrhenius_one(frame, self.args())

        self.assertEqual("single", summary["selected_model"])
        self.assertIsNone(summary["piecewise"])
        self.assertEqual(
            "insufficient_temperature_points", summary["piecewise_evidence_status"]
        )
        self.assertEqual("insufficient_breakpoint_evidence", summary["selection_reason"])

    def test_weak_bic_improvement_keeps_single_selected(self) -> None:
        summary = self.summary(
            self.two_regime_diffusivities(), piecewise_bic_delta=1.0e6
        )

        self.assertFalse(summary["piecewise"]["BIC_threshold_met"])
        self.assertTrue(summary["piecewise"]["slope_change_threshold_met"])
        self.assertEqual("single", summary["selected_model"])

    def test_large_bic_improvement_without_slope_change_keeps_single_selected(self) -> None:
        diffusivities = [
            self.diffusivity(temperature, 0.3, 1.0e-3)
            if temperature <= 450.0
            else self.diffusivity(temperature, 0.3, 1.0e-5)
            for temperature in self.temperatures
        ]

        summary = self.summary(diffusivities, piecewise_slope_change=0.2)

        self.assertTrue(summary["piecewise"]["BIC_threshold_met"])
        self.assertFalse(summary["piecewise"]["slope_change_threshold_met"])
        self.assertEqual("single", summary["selected_model"])

    def test_piecewise_always_reports_forced_weak_evidence(self) -> None:
        diffusivities = [
            self.diffusivity(temperature, 0.3, 1.0e-3)
            for temperature in self.temperatures
        ]

        summary = self.summary(diffusivities, piecewise="always")

        self.assertEqual("piecewise", summary["selected_model"])
        self.assertFalse(summary["regime_change_detected"])
        self.assertEqual(
            "piecewise_forced_without_automatic_evidence", summary["selection_reason"]
        )

    def test_target_prediction_uses_only_the_selected_piecewise_branch(self) -> None:
        from pymatgen.analysis.diffusion.analyzer import get_extrapolated_diffusivity

        diffusivities = self.two_regime_diffusivities()
        for target, expected_regime, branch_slice in (
            (325.0, "low_temperature", slice(None, 4)),
            (700.0, "high_temperature", slice(4, None)),
        ):
            with self.subTest(target=target):
                summary = self.summary(
                    diffusivities, target_temperature_K=target
                )
                prediction = summary["target_prediction"]["arrhenius_prediction"]
                expected = get_extrapolated_diffusivity(
                    self.temperatures[branch_slice],
                    diffusivities[branch_slice],
                    target,
                    mode="linear",
                )
                self.assertEqual(expected_regime, prediction["regime"])
                self.assertTrue(
                    math.isclose(
                        expected,
                        prediction["diffusivity_cm2_s"],
                        rel_tol=1.0e-12,
                    )
                )

    def test_target_inside_breakpoint_interval_is_ambiguous(self) -> None:
        summary = self.summary(
            self.two_regime_diffusivities(), target_temperature_K=500.0
        )

        prediction = summary["target_prediction"]["arrhenius_prediction"]
        self.assertEqual("ambiguous", prediction["status"])
        self.assertEqual("ambiguous", prediction["regime"])
        self.assertIsNone(prediction["diffusivity_cm2_s"])

    def test_regime_detection_never_claims_a_phase_transition(self) -> None:
        summary = self.summary(self.two_regime_diffusivities())
        serialized = json.dumps(summary, sort_keys=True)

        self.assertEqual(
            "ARRHENIUS_REGIME_CHANGE_DETECTED", summary["interpretation"]
        )
        self.assertNotIn("PHASE_TRANSITION_DETECTED", serialized)
        self.assertIn(
            "independent structural evidence", summary["phase_transition_caveat"]
        )

    def test_checker_reruns_and_rejects_changed_piecewise_evidence(self) -> None:
        diffusivities = self.two_regime_diffusivities()
        frame = self.runner.pd.DataFrame(
            {
                "dataset": ["synthetic"] * len(self.temperatures),
                "temperature_K": self.temperatures,
                "diffusivity_cm2_s": diffusivities,
                "conductivity_NE_mS_cm": [None] * len(self.temperatures),
            }
        )
        args = self.args()
        summary = self.runner.build_arrhenius_summary(frame, args)

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            (output_dir / "analysis_manifest.json").write_text(
                json.dumps({"parameters": vars(args)}), encoding="utf-8"
            )
            from mlipflow.plugins.ionic_transport import analysis as adapter_module
            rows = frame.to_dict(orient="records")
            self.assertEqual(
                [],
                adapter_module._verify_arrhenius_summary(
                    output_dir, rows, summary
                ),
            )

            changed = json.loads(json.dumps(summary))
            changed["piecewise"]["low_temperature"]["Ea_eV"] += 0.1
            diagnostics = adapter_module._verify_arrhenius_summary(
                output_dir, rows, changed
            )
            self.assertEqual(
                ["result.arrhenius_relation"],
                [item["code"] for item in diagnostics],
            )


class IonicTransportMsdOnlyRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = load_runner()
        self.args = SimpleNamespace(
            specie="Li",
            fit_start_ps=None,
            fit_end_ps=None,
            min_msd_fit_points=3,
            diffusion_analyzer_smoothed="max",
            diffusion_analyzer_min_obs=3,
            diffusion_analyzer_avg_nsteps=3,
        )
        self.time_ps = self.runner.np.asarray([0, 1, 2, 3, 4, 5], dtype=float)
        self.msd_A2 = self.runner.np.asarray([1, 2, 4, 7, 11, 16], dtype=float)

    def run_data(self, structure=None):
        return self.runner.RunData(
            dataset="fixture",
            run_dir=Path("."),
            temperature_K=700.0,
            source_kind="msd_file",
            source_path=Path("msd.csv"),
            time_ps=self.time_ps,
            msd_A2=self.msd_A2,
            n_mobile=None if structure is None else 1,
            volume_A3=None if structure is None else float(structure.volume),
            drift_correction="as_in_msd_file",
            structure=structure,
        )

    def test_smoothed_max_msd_excludes_zero_lag_and_matches_pymatgen(self) -> None:
        from pymatgen.analysis.diffusion.analyzer import get_diffusivity_from_msd

        result = self.runner.formal_result_values(self.run_data(), self.args)
        expected = get_diffusivity_from_msd(
            self.msd_A2[1:], self.time_ps[1:] * 1000.0, smoothed="max"
        )
        self.assertEqual(float(expected[0]), result["diffusivity_cm2_s"])
        self.assertEqual(float(expected[1]), result["diffusivity_std_dev_cm2_s"])
        self.assertEqual([False, True, True, True, True, True], result["used"].tolist())
        self.assertIsNone(result["conductivity_mS_cm"])
        self.assertEqual("unavailable", result["conductivity_method"])

    def test_read_msd_table_preserves_physical_lag_and_step_origin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            msd_path = run_dir / "msd.csv"
            msd_path.write_text(
                "time,msd_A2\n5,1\n6,2\n7,4\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                msd_time_column="time",
                msd_column="msd_A2",
                msd_time_unit="step",
                msd_step_ps=0.5,
                lammps_timestep_ps=None,
                msd_unit="A2",
            )

            time_ps, _, notes = self.runner.read_msd_table(msd_path, run_dir, args)

            self.assertEqual([2.5, 3.0, 3.5], time_ps.tolist())
            self.assertIn("time_semantics=physical_lag_or_elapsed", notes)

    def test_msd_structure_conductivity_uses_pymatgen_conversion_factor(self) -> None:
        from pymatgen.analysis.diffusion.analyzer import get_conversion_factor
        from pymatgen.core import Lattice, Structure

        structure = Structure(Lattice.cubic(10), ["Li", "S"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        result = self.runner.formal_result_values(self.run_data(structure), self.args)
        expected = result["diffusivity_cm2_s"] * get_conversion_factor(
            structure, "Li", 700.0
        )
        self.assertEqual(expected, result["conductivity_mS_cm"])
        self.assertEqual("pymatgen-get-conversion-factor", result["conductivity_method"])

    def test_missing_formal_dependency_has_actionable_error_without_breaking_core_import(self) -> None:
        import mlipflow

        original_import = __import__

        def blocked_import(name, *args, **kwargs):
            if name == "pymatgen.analysis.diffusion.analyzer":
                raise ModuleNotFoundError("simulated missing diffusion extra")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=blocked_import):
            with self.assertRaisesRegex(
                ImportError,
                r"Formal ionic transport requires pymatgen-analysis-diffusion.*mlipflow\[transport\]",
            ):
                self.runner.require_formal_diffusion_api()
        self.assertIsNotNone(mlipflow)


if __name__ == "__main__":
    unittest.main()
