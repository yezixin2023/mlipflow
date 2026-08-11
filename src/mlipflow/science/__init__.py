"""Small, unit-explicit scientific primitives used by future adapters.

These functions are not complete workflow plugins and do not replace the
manuscript's validated production scripts.
"""

from .transport import linear_diffusion_from_msd
from .voltage import average_intercalation_voltage

__all__ = ["average_intercalation_voltage", "linear_diffusion_from_msd"]

