"""Unit-explicit average intercalation-voltage primitive."""

from __future__ import annotations

import math


def average_intercalation_voltage(
    *,
    energy_low_li_ev: float,
    energy_high_li_ev: float,
    x_low: float,
    x_high: float,
    lithium_reference_ev: float,
    electrons_per_li: float = 1.0,
) -> float:
    """Return average voltage in V between two Li contents.

    Energies are total energies in eV for the same host/formula-unit basis.
    ``x_high`` must contain more Li than ``x_low``. With energies expressed in
    eV, division by one elementary charge gives volts numerically.
    """

    values = (
        energy_low_li_ev,
        energy_high_li_ev,
        x_low,
        x_high,
        lithium_reference_ev,
        electrons_per_li,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("voltage inputs must be finite")
    delta_x = x_high - x_low
    if delta_x <= 0:
        raise ValueError("x_high must be greater than x_low")
    if electrons_per_li <= 0:
        raise ValueError("electrons_per_li must be positive")
    reaction_energy = energy_high_li_ev - energy_low_li_ev - delta_x * lithium_reference_ev
    return -reaction_energy / (delta_x * electrons_per_li)

