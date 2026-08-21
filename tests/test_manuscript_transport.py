from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = PROJECT_ROOT / "plugins" / "ionic-transport" / "adapter.py"


def load_transport_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mlipflow_ionic_transport_parity", ADAPTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load ionic-transport adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRANSPORT = load_transport_module()


class ManuscriptTransportFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_target(self, name: str, slope_a2_fs: float, intercept_a2: float = 0.2) -> Path:
        path = self.root / name / "target.msd"
        path.parent.mkdir(parents=True)
        lines = [
            "# Time-averaged data for fix 2",
            "# TimeStep c_msd1[4] c_msd2[4] c_msd3[4] c_msd4[4] "
            "c_msd5[4] c_msd6[4] c_msd7[4] c_msd8[4]",
        ]
        for timestep in range(0, 401, 10):
            li_msd = slope_a2_fs * timestep + intercept_a2
            lines.append("%d %.17g 0 0 0 0 0 0 0" % (timestep, li_msd))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_small_target_msd_reproduces_both_historical_diffusivity_methods(self) -> None:
        target = self.write_target("400K", slope_a2_fs=6.0e-4)
        before = target.read_bytes()
        result = TRANSPORT.analyze_historical_target_msd(target, "mace_li10_legacy_v1")
        after = target.read_bytes()

        self.assertEqual(before, after)
        self.assertEqual("OK", result["status"])
        self.assertEqual(5, result["sampling"]["selected_point_count"])
        self.assertAlmostEqual(
            1.0e-9,
            result["methods"]["linear_msd_fit"]["diffusivity_m2_s"],
            places=20,
        )
        self.assertAlmostEqual(1.0, result["methods"]["linear_msd_fit"]["r2"])
        self.assertNotEqual(
            result["methods"]["linear_msd_fit"]["diffusivity_m2_s"],
            result["methods"]["mean_msd_over_time"]["diffusivity_m2_s"],
        )

    def test_historical_stdout_parser_handles_labeled_and_unlabeled_blocks(self) -> None:
        history = self.root / "Li10_MACE.out"
        history.write_text(
            "scheduler warning that is not scientific evidence\n"
            "11221\n"
            "Diffusion Coefficients of Li+: 1e-10 m^2/s\n"
            "扩散系数 = 2e-10\n"
            "400 电导率(1)： 3 S/m\n"
            "400 电导率(2)： 6 S/m\n\n"
            "Diffusion Coefficients of Li+: 2e-10 m^2/s\n"
            "扩散系数 = 4e-10\n"
            "600 电导率(1)： 4 S/m\n"
            "600 电导率(2)： 8 S/m\n",
            encoding="utf-8",
        )
        parsed = TRANSPORT.parse_historical_transport_output(history)

        self.assertEqual(["11221", "block-2"], [d["dataset_id"] for d in parsed["datasets"]])
        first = parsed["datasets"][0]["by_temperature"][0]
        self.assertEqual(1.0e-10, first["methods"]["linear_msd_fit"]["diffusivity_m2_s"])
        self.assertEqual(60.0, first["methods"]["mean_msd_over_time"]["conductivity_NE_mS_cm"])
        self.assertEqual(1, parsed["provenance"]["ignored_non_scientific_line_count"])

    def test_plugin_local_wrapper_writes_only_a_new_normalized_output(self) -> None:
        history = self.root / "Li10_test.out"
        history.write_text(
            "Diffusion Coefficients of Li+: 1e-10 m^2/s\n"
            "扩散系数 = 2e-10\n"
            "400 电导率(1)： 3 S/m\n"
            "400 电导率(2)： 6 S/m\n",
            encoding="utf-8",
        )
        output = self.root / "normalized.json"
        before = history.read_bytes()
        returncode = TRANSPORT.manuscript_transport_main(
            ["parse-historical-output", str(history), "--output", str(output)]
        )
        normalized = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(0, returncode)
        self.assertEqual("mlipflow.historical_transport_output", normalized["artifact_type"])
        self.assertEqual(before, history.read_bytes())

    def test_legacy_n7_and_corrected_n10_are_separate_and_ratio_is_explicit(self) -> None:
        runs = {
            400: self.write_target("400K", 2.0e-4),
            600: self.write_target("600K", 6.0e-4),
            800: self.write_target("800K", 1.2e-3),
        }
        legacy = TRANSPORT.reproduce_manuscript_transport(
            runs,
            sampling_profile="mace_li10_legacy_v1",
            carrier_convention="legacy_script",
            arrhenius_convention="exact_unrounded_ols_v1",
            volume_a3=1000.0,
        )
        corrected = TRANSPORT.reproduce_manuscript_transport(
            runs,
            sampling_profile="mace_li10_legacy_v1",
            carrier_convention="composition_corrected",
            arrhenius_convention="exact_unrounded_ols_v1",
            n_mobile_ions=10,
            volume_a3=1000.0,
        )

        self.assertEqual(7, legacy["carrier_convention"]["n_mobile_ions"])
        self.assertTrue(legacy["carrier_convention"]["historical_N7_error_preserved"])
        self.assertEqual(10, corrected["carrier_convention"]["n_mobile_ions"])
        self.assertFalse(corrected["carrier_convention"]["historical_N7_error_preserved"])
        for legacy_row, corrected_row in zip(legacy["by_temperature"], corrected["by_temperature"]):
            for method in ("linear_msd_fit", "mean_msd_over_time"):
                self.assertEqual(
                    legacy_row["methods"][method]["diffusivity_m2_s"],
                    corrected_row["methods"][method]["diffusivity_m2_s"],
                )
                ratio = (
                    corrected_row["methods"][method]["conductivity_S_m"]
                    / legacy_row["methods"][method]["conductivity_S_m"]
                )
                self.assertTrue(math.isclose(ratio, 10.0 / 7.0, rel_tol=1e-14))
        self.assertFalse(legacy["execution"]["md_executed"])
        self.assertIn(
            "activation_energy_eV",
            corrected["arrhenius"]["methods"]["linear_msd_fit"],
        )

    def test_corrected_count_can_come_from_poscar_and_disagreement_is_rejected(self) -> None:
        poscar = self.root / "POSCAR"
        coordinates = ["0 0 0"] * 11
        poscar.write_text(
            "Li10P1\n"
            "1.0\n"
            "10 0 0\n"
            "0 10 0\n"
            "0 0 10\n"
            "Li P\n"
            "10 1\n"
            "Direct\n" + "\n".join(coordinates) + "\n",
            encoding="utf-8",
        )
        structure = TRANSPORT.parse_poscar_mobile_ions(poscar, "Li")
        self.assertEqual(10, structure["n_mobile_ions"])
        self.assertEqual(1000.0, structure["volume_A3"])
        runs = {
            400: self.write_target("s400", 2.0e-4),
            600: self.write_target("s600", 6.0e-4),
            800: self.write_target("s800", 1.2e-3),
        }
        result = TRANSPORT.reproduce_manuscript_transport(
            runs,
            sampling_profile="mace_li10_legacy_v1",
            carrier_convention="composition_corrected",
            arrhenius_convention="legacy_get_sigma_v1",
            structure_path=poscar,
            n_mobile_ions=10,
        )
        self.assertEqual(
            "explicit+structure-verified", result["carrier_convention"]["n_mobile_ions_source"]
        )
        self.assertEqual("legacy_get_sigma_v1", result["arrhenius"]["convention"])
        self.assertEqual(
            1.667,
            result["arrhenius"]["methods"]["linear_msd_fit"]["fit_inputs"][1]["x_1000_over_T_K"],
        )
        with self.assertRaisesRegex(TRANSPORT.ManuscriptTransportError, "disagrees"):
            TRANSPORT.reproduce_manuscript_transport(
                runs,
                sampling_profile="mace_li10_legacy_v1",
                carrier_convention="composition_corrected",
                arrhenius_convention="exact_unrounded_ols_v1",
                structure_path=poscar,
                n_mobile_ions=9,
            )

    def test_ambiguous_carrier_inputs_are_blocked_instead_of_mixed(self) -> None:
        runs = {
            400: self.write_target("a400", 2.0e-4),
            600: self.write_target("a600", 6.0e-4),
            800: self.write_target("a800", 1.2e-3),
        }
        with self.assertRaisesRegex(TRANSPORT.ManuscriptTransportError, "fixes N=7"):
            TRANSPORT.reproduce_manuscript_transport(
                runs,
                sampling_profile="mace_li10_legacy_v1",
                carrier_convention="legacy_script",
                arrhenius_convention="exact_unrounded_ols_v1",
                n_mobile_ions=10,
                volume_a3=1000.0,
            )
        with self.assertRaisesRegex(TRANSPORT.ManuscriptTransportError, "requires explicit"):
            TRANSPORT.reproduce_manuscript_transport(
                runs,
                sampling_profile="mace_li10_legacy_v1",
                carrier_convention="composition_corrected",
                arrhenius_convention="exact_unrounded_ols_v1",
                volume_a3=1000.0,
            )
        with self.assertRaisesRegex(
            TRANSPORT.ManuscriptTransportError, "historical_carrier_convention is required"
        ):
            history = self.root / "history.out"
            history.write_text(
                "Diffusion Coefficients of Li+: 1e-10 m^2/s\n"
                "扩散系数 = 2e-10\n"
                "400 电导率(1)： 3 S/m\n"
                "400 电导率(2)： 6 S/m\n",
                encoding="utf-8",
            )
            TRANSPORT.reproduce_manuscript_transport(
                runs,
                sampling_profile="mace_li10_legacy_v1",
                carrier_convention="legacy_script",
                arrhenius_convention="exact_unrounded_ols_v1",
                volume_a3=1000.0,
                historical_output_path=history,
            )


if __name__ == "__main__":
    unittest.main()
