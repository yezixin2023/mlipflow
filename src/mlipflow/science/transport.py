"""Minimal, transparent MSD linear-fit primitive.

Production ionic-transport adapters still need equilibration selection,
multiple time origins, correlation diagnostics, uncertainty, anisotropy and
replica handling. This helper deliberately does not guess those choices.
"""

from __future__ import annotations

import math
from typing import Sequence


def linear_diffusion_from_msd(
    times_ps: Sequence[float],
    msd_angstrom2: Sequence[float],
    *,
    dimensions: int = 3,
) -> dict[str, float | int | str]:
    """Fit MSD = intercept + slope*time and return D with explicit units."""

    if len(times_ps) != len(msd_angstrom2) or len(times_ps) < 3:
        raise ValueError("times and MSD need the same length with at least three points")
    if dimensions not in {1, 2, 3}:
        raise ValueError("dimensions must be 1, 2, or 3")
    if not all(math.isfinite(float(value)) for value in (*times_ps, *msd_angstrom2)):
        raise ValueError("times and MSD must be finite")
    if any(right <= left for left, right in zip(times_ps, times_ps[1:])):
        raise ValueError("times must be strictly increasing")
    count = len(times_ps)
    mean_time = sum(times_ps) / count
    mean_msd = sum(msd_angstrom2) / count
    denominator = sum((time - mean_time) ** 2 for time in times_ps)
    if denominator == 0:
        raise ValueError("time variance is zero")
    slope = sum(
        (time - mean_time) * (msd - mean_msd)
        for time, msd in zip(times_ps, msd_angstrom2)
    ) / denominator
    intercept = mean_msd - slope * mean_time
    residual = sum(
        (msd - (intercept + slope * time)) ** 2
        for time, msd in zip(times_ps, msd_angstrom2)
    )
    total = sum((msd - mean_msd) ** 2 for msd in msd_angstrom2)
    r_squared = 1.0 if total == 0 and residual == 0 else 1.0 - residual / total
    diffusion_a2_ps = slope / (2.0 * dimensions)
    return {
        "samples": count,
        "dimensions": dimensions,
        "slope_angstrom2_per_ps": slope,
        "intercept_angstrom2": intercept,
        "r_squared": r_squared,
        "diffusion_angstrom2_per_ps": diffusion_a2_ps,
        "diffusion_cm2_per_s": diffusion_a2_ps * 1.0e-4,
        "fit_assumption": "ordinary-least-squares over caller-selected window",
    }

