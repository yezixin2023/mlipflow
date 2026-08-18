"""Unit-explicit transport helpers retained for legacy/historical reproduction.

Formal ionic-transport analysis and its checker use pymatgen-analysis-diffusion
public APIs directly. These helpers remain stable for isolated historical parity
and low-level regression tests; they are not a fallback for formal analysis.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence


BOLTZMANN_EV_PER_K = 8.617333262145e-5
BOLTZMANN_J_PER_K = 1.380649e-23
ELEMENTARY_CHARGE_C = 1.602176634e-19


def _finite_floats(values: Sequence[float], name: str) -> list[float]:
    converted = [float(value) for value in values]
    if not all(math.isfinite(value) for value in converted):
        raise ValueError(f"{name} must contain only finite values")
    return converted


def linear_fit(x_values: Sequence[float], y_values: Sequence[float]) -> Dict[str, float]:
    """Return an unweighted ordinary-least-squares line and diagnostics."""

    x = _finite_floats(x_values, "x_values")
    y = _finite_floats(y_values, "y_values")
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("linear fit requires at least two paired points")
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    sxx = sum((value - x_mean) ** 2 for value in x)
    if sxx <= 0.0:
        raise ValueError("linear fit x values must not all be identical")
    slope = sum((xv - x_mean) * (yv - y_mean) for xv, yv in zip(x, y)) / sxx
    intercept = y_mean - slope * x_mean
    predicted = [intercept + slope * value for value in x]
    residual_sum = sum((actual - fitted) ** 2 for actual, fitted in zip(y, predicted))
    total_sum = sum((value - y_mean) ** 2 for value in y)
    r_squared = (
        1.0
        if total_sum == 0.0 and residual_sum == 0.0
        else 1.0 - residual_sum / total_sum
    )
    slope_stderr: Optional[float] = None
    if len(x) > 2:
        slope_stderr = math.sqrt(max(0.0, residual_sum / (len(x) - 2) / sxx))
    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "residual_sum_squares": residual_sum,
        "slope_stderr": slope_stderr,
        "point_count": len(x),
    }


def linear_diffusion_from_msd(
    times_ps: Sequence[float],
    msd_angstrom2: Sequence[float],
    *,
    dimensions: int = 3,
) -> Dict[str, object]:
    """Fit MSD and apply ``D = slope / (2 d)`` with explicit units."""

    times = _finite_floats(times_ps, "times_ps")
    msd = _finite_floats(msd_angstrom2, "msd_angstrom2")
    if len(times) != len(msd) or len(times) < 3:
        raise ValueError("times and MSD need the same length with at least three points")
    if dimensions not in {1, 2, 3}:
        raise ValueError("dimensions must be 1, 2, or 3")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("times must be strictly increasing")
    fit = linear_fit(times, msd)
    diffusion_a2_ps = float(fit["slope"]) / (2.0 * dimensions)
    return {
        "samples": len(times),
        "dimensions": dimensions,
        "slope_angstrom2_per_ps": fit["slope"],
        "intercept_angstrom2": fit["intercept"],
        "r_squared": fit["r_squared"],
        "residual_sum_squares": fit["residual_sum_squares"],
        "slope_stderr_angstrom2_per_ps": fit["slope_stderr"],
        "diffusion_angstrom2_per_ps": diffusion_a2_ps,
        "diffusion_cm2_per_s": diffusion_a2_ps * 1.0e-4,
        "fit_assumption": "ordinary-least-squares over caller-selected window",
    }


def nernst_einstein_conductivity(
    diffusivity_m2_s: float,
    temperature_k: float,
    carrier_count: int,
    charge_number: float,
    volume_angstrom3: float,
    *,
    elementary_charge_c: float = ELEMENTARY_CHARGE_C,
    boltzmann_j_per_k: float = BOLTZMANN_J_PER_K,
) -> Dict[str, float]:
    """Return uncorrected Nernst--Einstein conductivity.

    Custom constants exist only for the isolated historical-parity path, which
    retains its documented rounded constants without duplicating the equation.
    """

    values = (
        float(diffusivity_m2_s),
        float(temperature_k),
        float(charge_number),
        float(volume_angstrom3),
        float(elementary_charge_c),
        float(boltzmann_j_per_k),
    )
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("Nernst-Einstein inputs must be finite and positive")
    if isinstance(carrier_count, bool) or not isinstance(carrier_count, int) or carrier_count <= 0:
        raise ValueError("carrier_count must be a positive integer")
    volume_m3 = float(volume_angstrom3) * 1.0e-30
    number_density_m3 = carrier_count / volume_m3
    conductivity_s_m = (
        number_density_m3
        * (float(charge_number) * float(elementary_charge_c)) ** 2
        * float(diffusivity_m2_s)
        / (float(boltzmann_j_per_k) * float(temperature_k))
    )
    return {
        "conductivity_s_m": conductivity_s_m,
        "conductivity_ms_cm": conductivity_s_m * 10.0,
        "number_density_m3": number_density_m3,
    }


def arrhenius_from_diffusivities(
    temperatures_k: Sequence[float],
    diffusivities_cm2_s: Sequence[float],
    *,
    target_temperature_k: Optional[float] = None,
    boltzmann_ev_per_k: float = BOLTZMANN_EV_PER_K,
) -> Dict[str, object]:
    """Fit ``ln(D_cm2/s)`` against ``1/T`` and return Arrhenius parameters."""

    temperatures = _finite_floats(temperatures_k, "temperatures_k")
    diffusivities = _finite_floats(diffusivities_cm2_s, "diffusivities_cm2_s")
    if len(temperatures) != len(diffusivities) or len(temperatures) < 2:
        raise ValueError("Arrhenius fit requires at least two paired temperatures")
    if any(value <= 0.0 for value in temperatures):
        raise ValueError("Arrhenius temperatures must be positive")
    if any(value <= 0.0 for value in diffusivities):
        raise ValueError("Arrhenius diffusivities must be positive")
    if not math.isfinite(float(boltzmann_ev_per_k)) or float(boltzmann_ev_per_k) <= 0.0:
        raise ValueError("boltzmann_ev_per_k must be finite and positive")
    inverse_temperatures = [1.0 / value for value in temperatures]
    log_diffusivities = [math.log(value) for value in diffusivities]
    fit = linear_fit(inverse_temperatures, log_diffusivities)
    result: Dict[str, object] = {
        "temperatures_k": temperatures,
        "diffusivities_cm2_s": diffusivities,
        "inverse_temperatures_k_inverse": inverse_temperatures,
        "log_diffusivities": log_diffusivities,
        "slope_k": fit["slope"],
        "intercept_ln_diffusivity": fit["intercept"],
        "activation_energy_ev": -float(fit["slope"]) * float(boltzmann_ev_per_k),
        "prefactor_cm2_s": math.exp(float(fit["intercept"])),
        "r_squared": fit["r_squared"],
        "residual_sum_squares": fit["residual_sum_squares"],
        "slope_stderr_k": fit["slope_stderr"],
        "activation_energy_stderr_ev": (
            None
            if fit["slope_stderr"] is None
            else float(fit["slope_stderr"]) * float(boltzmann_ev_per_k)
        ),
        "point_count": fit["point_count"],
    }
    if target_temperature_k is not None:
        target = float(target_temperature_k)
        if not math.isfinite(target) or target <= 0.0:
            raise ValueError("target_temperature_k must be finite and positive")
        result["target_temperature_k"] = target
        result["target_diffusivity_cm2_s"] = math.exp(
            float(fit["intercept"]) + float(fit["slope"]) / target
        )
    return result
