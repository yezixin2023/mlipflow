from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "plugins" / "ionic-transport" / "adapter.py"


def load_adapter():
    spec = importlib.util.spec_from_file_location("ionic_transport_analysis_adapter", ADAPTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Adapter()


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
    def test_tiny_lammps_trajectory_completes_formal_analysis_and_checker(self) -> None:
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
                    "charge": 1.0,
                    "dimensions": 3,
                    "haven_ratio": 1.0,
                    "seed": None,
                    "fit_start_ps": 0.0,
                    "fit_end_ps": 5.0,
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


if __name__ == "__main__":
    unittest.main()
