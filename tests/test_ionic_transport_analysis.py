from __future__ import annotations

import csv
import importlib.util
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
ADAPTER_PATH = ROOT / "plugins" / "ionic-transport" / "adapter.py"
RUNNER_PATH = ROOT / "plugins" / "ionic-transport" / "ionic_conductivity.py"


def load_adapter():
    spec = importlib.util.spec_from_file_location("ionic_transport_analysis_adapter", ADAPTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Adapter()


def load_runner():
    name = "test_ionic_transport_formal_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
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


class IonicTransportTrajectoryRegressionTests(unittest.TestCase):
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
                    "min_segment_points": 2,
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
