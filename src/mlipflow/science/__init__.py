"""Small, unit-explicit scientific primitives used by future adapters.

These functions are not complete workflow plugins and do not replace the
manuscript's validated production scripts.
"""

from .transport import (
    arrhenius_from_diffusivities,
    linear_diffusion_from_msd,
    linear_fit,
    nernst_einstein_conductivity,
)
from .model_runtime import (
    FRESH_MODEL_FAMILIES,
    MODEL_FAMILIES,
    MODEL_FAMILY_ALIASES,
    MODEL_FAMILY_FRAMEWORKS,
    RuntimeCompatibilityError,
    canonical_model_family,
    framework_version,
    load_inference_predictor,
)
from .voltage import average_intercalation_voltage

__all__ = [
    "FRESH_MODEL_FAMILIES",
    "MODEL_FAMILIES",
    "MODEL_FAMILY_ALIASES",
    "MODEL_FAMILY_FRAMEWORKS",
    "RuntimeCompatibilityError",
    "arrhenius_from_diffusivities",
    "average_intercalation_voltage",
    "linear_diffusion_from_msd",
    "linear_fit",
    "nernst_einstein_conductivity",
    "canonical_model_family",
    "framework_version",
    "load_inference_predictor",
]
