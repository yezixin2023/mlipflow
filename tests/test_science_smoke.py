from __future__ import annotations

import math
import unittest

from mlipflow.plugins.electrochemical_voltage.science import average_intercalation_voltage
from mlipflow.plugins.ionic_transport.manuscript import (
    arrhenius_from_diffusivities,
    linear_diffusion_from_msd,
    nernst_einstein_conductivity,
)


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

    def test_known_nernst_einstein_conductivity(self) -> None:
        result = nernst_einstein_conductivity(
            diffusivity_m2_s=2.0e-11,
            temperature_k=500.0,
            carrier_count=4,
            charge_number=1.0,
            volume_angstrom3=800.0,
        )
        expected = (
            (4.0 / (800.0e-30))
            * (1.602176634e-19) ** 2
            * 2.0e-11
            / (1.380649e-23 * 500.0)
        )
        self.assertAlmostEqual(result["conductivity_s_m"], expected)
        self.assertAlmostEqual(result["conductivity_ms_cm"], expected * 10.0)

    def test_known_arrhenius_series(self) -> None:
        temperatures = [400.0, 600.0, 800.0]
        activation_energy_ev = 0.25
        prefactor = 2.0e-4
        diffusivities = [
            prefactor * math.exp(-activation_energy_ev / (8.617333262145e-5 * t))
            for t in temperatures
        ]
        result = arrhenius_from_diffusivities(
            temperatures, diffusivities, target_temperature_k=300.0
        )
        self.assertAlmostEqual(result["activation_energy_ev"], activation_energy_ev)
        self.assertAlmostEqual(result["prefactor_cm2_s"], prefactor)
        self.assertAlmostEqual(result["r_squared"], 1.0)


if __name__ == "__main__":
    unittest.main()
