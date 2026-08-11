from __future__ import annotations

import unittest

from mlipflow.science import average_intercalation_voltage, linear_diffusion_from_msd


class ScientificPrimitiveTests(unittest.TestCase):
    def test_average_voltage_sign_and_units(self) -> None:
        voltage = average_intercalation_voltage(
            energy_low_li_ev=-10.0,
            energy_high_li_ev=-13.0,
            x_low=0.0,
            x_high=1.0,
            lithium_reference_ev=-1.0,
        )
        self.assertAlmostEqual(voltage, 2.0)

    def test_voltage_requires_consistent_li_order(self) -> None:
        with self.assertRaises(ValueError):
            average_intercalation_voltage(
                energy_low_li_ev=-10,
                energy_high_li_ev=-11,
                x_low=1,
                x_high=1,
                lithium_reference_ev=-1,
            )

    def test_three_dimensional_einstein_slope(self) -> None:
        result = linear_diffusion_from_msd([0, 1, 2, 3], [1, 7, 13, 19], dimensions=3)
        self.assertAlmostEqual(result["slope_angstrom2_per_ps"], 6.0)
        self.assertAlmostEqual(result["diffusion_angstrom2_per_ps"], 1.0)
        self.assertAlmostEqual(result["diffusion_cm2_per_s"], 1e-4)
        self.assertAlmostEqual(result["r_squared"], 1.0)


if __name__ == "__main__":
    unittest.main()

