"""Historical manuscript reproduction, including its original transport formulae.

These helpers reproduce existing evidence. Formal analysis uses the public
pymatgen-analysis-diffusion APIs in ionic_conductivity.py and analysis.py.
This module also runs by file with only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

MANUSCRIPT_PARITY_VERSION = "1.0"

LEGACY_SCRIPT_CONVENTION = "legacy_script"

COMPOSITION_CORRECTED_CONVENTION = "composition_corrected"

EXACT_ARRHENIUS_CONVENTION = "exact_unrounded_ols_v1"

LEGACY_ARRHENIUS_CONVENTION = "legacy_get_sigma_v1"

_LEGACY_LI10_N_MOBILE_IONS = 7

_HISTORICAL_ELEMENTARY_CHARGE_C = 1.6e-19

_HISTORICAL_BOLTZMANN_J_K = 1.38e-23

_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

_SAMPLING_PROFILES: Dict[str, Dict[str, Any]] = {
    "mace_li10_legacy_v1": {
        "start_line_index": 3,
        "line_modulus": 8,
        "line_offset": 2,
        "timestep_fs": 1.0,
        "description": (
            "Exact line selection used by the historical Li10 MACE get_MSD_from_* scripts: "
            "range(3, len(lines)) and (i - 2) % 8 == 0."
        ),
    },
    "deepmd_li10_legacy_v1": {
        "start_line_index": 990,
        "line_modulus": 12,
        "line_offset": 2,
        "timestep_fs": 1.0,
        "description": (
            "Exact line selection used by the historical Li10 DeepMD screening scripts: "
            "range(990, len(lines)) and (i - 2) % 12 == 0."
        ),
    },
}


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
    if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in values):
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


def _positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0


class ManuscriptTransportError(ValueError):
    """Raised when historical evidence is incomplete or scientifically ambiguous."""


def _read_parity_source(path_value: Any, role: str) -> Tuple[str, Dict[str, Any]]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ManuscriptTransportError("%s must name an existing file: %s" % (role, path))
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManuscriptTransportError("%s is not UTF-8 text: %s" % (role, path)) from exc
    provenance = {
        "role": role,
        "path": str(path),
        "read_only": True,
    }
    return text, provenance


def _transport_value(value: float, unit_kind: str) -> Dict[str, float]:
    if unit_kind == "diffusivity":
        return {
            "diffusivity_m2_s": float(value),
            "diffusivity_cm2_s": float(value) * 1.0e4,
        }
    if unit_kind == "conductivity":
        return {
            "conductivity_S_m": float(value),
            "conductivity_NE_mS_cm": float(value) * 10.0,
        }
    raise AssertionError("unknown transport unit kind: %s" % unit_kind)


def _history_method_record(diffusivity: float, conductivity: float) -> Dict[str, Any]:
    record: Dict[str, Any] = {}
    record.update(_transport_value(diffusivity, "diffusivity"))
    record.update(_transport_value(conductivity, "conductivity"))
    return record


def parse_historical_transport_output(path_value: Any) -> Dict[str, Any]:
    """Parse Li10_test.out/Li10_MACE.out without executing or modifying them.

    Both historical files use the same four-line record: a linear-fit diffusivity,
    a mean(MSD/t) diffusivity, then the corresponding conductivity (1)/(2).
    Labels such as ``11221`` start named blocks; unlabeled blocks receive stable
    ``block-N`` identifiers. Unrelated scheduler diagnostics are counted but not
    interpreted as scientific records.
    """

    text, provenance = _read_parity_source(path_value, "historical_transport_output")
    diffusion_linear_re = re.compile(
        r"^\s*Diffusion\s+Coefficients\s+of\s+Li\+?\s*:\s*(%s)\s*m\^?2/s\s*$" % _FLOAT_PATTERN,
        re.IGNORECASE,
    )
    diffusion_ratio_re = re.compile(r"^\s*扩散系数\s*=\s*(%s)\s*$" % _FLOAT_PATTERN)
    conductivity_re = re.compile(
        r"^\s*(%s)\s*(?:电导率|conductivity)\s*[\(（]\s*([12])\s*[\)）]"
        r"\s*[:：]\s*(%s)\s*S/m\s*$" % (_FLOAT_PATTERN, _FLOAT_PATTERN),
        re.IGNORECASE,
    )
    label_re = re.compile(r"^[A-Za-z0-9_.+-]+$")

    datasets: List[Dict[str, Any]] = []
    current_label: Optional[str] = None
    current_records: List[Dict[str, Any]] = []
    pending: Dict[str, Any] = {}
    ignored_line_count = 0

    def finish_dataset() -> None:
        nonlocal current_label, current_records, pending
        if pending:
            raise ManuscriptTransportError(
                "historical output ended a block with an incomplete four-line record"
            )
        if not current_records:
            current_label = None
            return
        temperatures = [item["temperature_K"] for item in current_records]
        if len(temperatures) != len(set(temperatures)):
            raise ManuscriptTransportError("historical output repeats a temperature in one block")
        block_index = len(datasets) + 1
        datasets.append(
            {
                "dataset_id": current_label or "block-%d" % block_index,
                "source_label": current_label,
                "block_index": block_index,
                "by_temperature": current_records,
            }
        )
        current_label = None
        current_records = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            if current_records:
                finish_dataset()
            continue
        matched = diffusion_linear_re.match(line)
        if matched:
            if pending:
                raise ManuscriptTransportError(
                    "line %d starts a new record before the previous record completed" % line_number
                )
            pending["linear_diffusivity_m2_s"] = float(matched.group(1))
            continue
        matched = diffusion_ratio_re.match(line)
        if matched:
            if "linear_diffusivity_m2_s" not in pending or len(pending) != 1:
                raise ManuscriptTransportError(
                    "line %d has mean(MSD/t) diffusivity out of order" % line_number
                )
            pending["ratio_diffusivity_m2_s"] = float(matched.group(1))
            continue
        matched = conductivity_re.match(line)
        if matched:
            temperature = float(matched.group(1))
            method_number = int(matched.group(2))
            conductivity = float(matched.group(3))
            if method_number == 1:
                expected = {"linear_diffusivity_m2_s", "ratio_diffusivity_m2_s"}
                if set(pending) != expected:
                    raise ManuscriptTransportError(
                        "line %d has conductivity (1) before both diffusivities" % line_number
                    )
                pending["temperature_K"] = temperature
                pending["linear_conductivity_S_m"] = conductivity
                continue
            expected = {
                "linear_diffusivity_m2_s",
                "ratio_diffusivity_m2_s",
                "temperature_K",
                "linear_conductivity_S_m",
            }
            if set(pending) != expected:
                raise ManuscriptTransportError(
                    "line %d has conductivity (2) before conductivity (1)" % line_number
                )
            if not math.isclose(
                temperature, float(pending["temperature_K"]), rel_tol=0.0, abs_tol=1e-12
            ):
                raise ManuscriptTransportError(
                    "line %d changes temperature inside one transport record" % line_number
                )
            current_records.append(
                {
                    "temperature_K": temperature,
                    "methods": {
                        "linear_msd_fit": _history_method_record(
                            float(pending["linear_diffusivity_m2_s"]),
                            float(pending["linear_conductivity_S_m"]),
                        ),
                        "mean_msd_over_time": _history_method_record(
                            float(pending["ratio_diffusivity_m2_s"]), conductivity
                        ),
                    },
                }
            )
            pending = {}
            continue

        if any(
            token in line
            for token in ("Diffusion Coefficients", "扩散系数", "电导率", "conductivity")
        ):
            raise ManuscriptTransportError(
                "line %d resembles a transport result but has an unknown format" % line_number
            )
        if label_re.fullmatch(line) and not pending:
            if current_records:
                finish_dataset()
            current_label = line
        else:
            ignored_line_count += 1

    finish_dataset()
    if not datasets:
        raise ManuscriptTransportError("historical output contains no complete transport records")
    return {
        "schema_version": 1,
        "artifact_type": "mlipipe.historical_transport_output",
        "status": "OK",
        "source_format": "li10_dual_method_stdout_v1",
        "datasets": datasets,
        "units": {
            "temperature_K": "K",
            "diffusivity_m2_s": "m^2/s",
            "diffusivity_cm2_s": "cm^2/s",
            "conductivity_S_m": "S/m",
            "conductivity_NE_mS_cm": "mS/cm",
        },
        "fit": {
            "linear_msd_fit": "historical stdout method (1); fit window is not encoded",
            "mean_msd_over_time": "historical stdout method (2); arithmetic mean of MSD/t",
            "arrhenius": "not encoded in Li10_test.out or Li10_MACE.out",
        },
        "convention": {
            "carrier_count": "not inferable from stdout; inspect the producing script",
            "conductivity_model": "uncorrected Nernst-Einstein in producing scripts",
            "methods_are_not_interchangeable": True,
        },
        "provenance": {
            "parser": "ionic-transport.adapter.parse_historical_transport_output",
            "parser_version": MANUSCRIPT_PARITY_VERSION,
            "ignored_non_scientific_line_count": ignored_line_count,
            "sources": [provenance],
        },
    }


def _sampling_profile(name: str) -> Dict[str, Any]:
    try:
        return dict(_SAMPLING_PROFILES[name])
    except KeyError as exc:
        raise ManuscriptTransportError(
            "sampling_profile must be one of %s" % sorted(_SAMPLING_PROFILES)
        ) from exc


def analyze_historical_target_msd(path_value: Any, sampling_profile: str) -> Dict[str, Any]:
    """Reproduce both diffusivity calculations in the historical MSD scripts."""

    text, provenance = _read_parity_source(path_value, "target_msd")
    lines = text.splitlines()
    profile = _sampling_profile(sampling_profile)
    header_index: Optional[int] = None
    msd_column_index: Optional[int] = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") and "TimeStep" in stripped and "c_msd1[4]" in stripped:
            header = stripped.lstrip("#").split()
            header_index = index
            msd_column_index = header.index("c_msd1[4]")
            break
    if header_index is None or msd_column_index is None:
        raise ManuscriptTransportError("target.msd must declare '# TimeStep ... c_msd1[4] ...'")

    timesteps: List[float] = []
    msd_values: List[float] = []
    sampled_line_indices: List[int] = []
    start = int(profile["start_line_index"])
    modulus = int(profile["line_modulus"])
    offset = int(profile["line_offset"])
    timestep_fs = float(profile["timestep_fs"])
    for line_index in range(start, len(lines)):
        if (line_index - offset) % modulus != 0:
            continue
        fields = lines[line_index].split()
        if len(fields) <= msd_column_index:
            raise ManuscriptTransportError(
                "target.msd line index %d is not a complete numeric row" % line_index
            )
        try:
            timestep = float(fields[0])
            msd = float(fields[msd_column_index])
        except ValueError as exc:
            raise ManuscriptTransportError(
                "target.msd line index %d contains non-numeric data" % line_index
            ) from exc
        if not math.isfinite(timestep) or timestep <= 0:
            raise ManuscriptTransportError(
                "sampled target.msd timestep must be finite and positive at line index %d"
                % line_index
            )
        if not math.isfinite(msd):
            raise ManuscriptTransportError(
                "sampled target.msd MSD must be finite at line index %d" % line_index
            )
        timesteps.append(timestep)
        msd_values.append(msd)
        sampled_line_indices.append(line_index)
    if len(timesteps) < 3:
        raise ManuscriptTransportError(
            "sampling profile %s selected fewer than three target.msd points" % sampling_profile
        )
    if any(right <= left for left, right in zip(timesteps, timesteps[1:])):
        raise ManuscriptTransportError("sampled target.msd timesteps must increase strictly")

    try:
        fit = linear_fit(timesteps, msd_values)
    except ValueError as exc:
        raise ManuscriptTransportError(str(exc)) from exc
    slope_a2_fs = float(fit["slope"]) / timestep_fs
    mean_ratio_a2_fs = sum(
        msd / (timestep * timestep_fs) for timestep, msd in zip(timesteps, msd_values)
    ) / len(timesteps)
    diffusion_linear = slope_a2_fs / 6.0 * 1.0e-5
    diffusion_ratio = mean_ratio_a2_fs / 6.0 * 1.0e-5
    if diffusion_linear <= 0 or diffusion_ratio <= 0:
        raise ManuscriptTransportError(
            "historical MSD analysis produced non-positive diffusivity; Arrhenius log is undefined"
        )

    linear_result: Dict[str, Any] = {
        "method": "linear_msd_fit",
        "slope_A2_fs": slope_a2_fs,
        "intercept_A2": float(fit["intercept"]),
        "r2": float(fit["r_squared"]),
    }
    linear_result.update(_transport_value(diffusion_linear, "diffusivity"))
    ratio_result: Dict[str, Any] = {
        "method": "mean_msd_over_time",
        "mean_MSD_over_t_A2_fs": mean_ratio_a2_fs,
    }
    ratio_result.update(_transport_value(diffusion_ratio, "diffusivity"))
    return {
        "status": "OK",
        "source_format": "lammps_fix_ave_time_target_msd",
        "mobile_msd_column": "c_msd1[4]",
        "sampling": {
            "profile": sampling_profile,
            **profile,
            "selected_point_count": len(timesteps),
            "first_selected_source_line_index": sampled_line_indices[0],
            "last_selected_source_line_index": sampled_line_indices[-1],
            "first_timestep": timesteps[0],
            "last_timestep": timesteps[-1],
            "first_time_ps": timesteps[0] * timestep_fs / 1000.0,
            "last_time_ps": timesteps[-1] * timestep_fs / 1000.0,
        },
        "methods": {
            "linear_msd_fit": linear_result,
            "mean_msd_over_time": ratio_result,
        },
        "units": {
            "raw_timestep": "LAMMPS step",
            "interpreted_timestep": "fs",
            "msd": "angstrom^2",
            "diffusivity_m2_s": "m^2/s",
            "diffusivity_cm2_s": "cm^2/s",
        },
        "fit": {
            "linear_msd_fit": "numpy.polyfit degree=1 on the exact historical line subset",
            "mean_msd_over_time": "numpy.mean(MSD/t)/6 on the same line subset",
            "dimensions": 3,
        },
        "provenance": {
            "parser": "ionic-transport.adapter.analyze_historical_target_msd",
            "parser_version": MANUSCRIPT_PARITY_VERSION,
            "sources": [provenance],
        },
    }


def _determinant_3x3(matrix: Sequence[Sequence[float]]) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def parse_poscar_mobile_ions(path_value: Any, mobile_species: str = "Li") -> Dict[str, Any]:
    """Read a VASP 5 POSCAR carrier count and volume without changing the structure."""

    if re.fullmatch(r"[A-Z][a-z]?", mobile_species) is None:
        raise ManuscriptTransportError("mobile_species must be one element symbol")
    text, provenance = _read_parity_source(path_value, "carrier_structure_poscar")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        raise ManuscriptTransportError("POSCAR is too short to contain a VASP 5 structure")
    try:
        scale_values = [float(value) for value in lines[1].split()]
        lattice = [[float(value) for value in lines[index].split()] for index in range(2, 5)]
    except ValueError as exc:
        raise ManuscriptTransportError("POSCAR scale/lattice contains non-numeric data") from exc
    if len(scale_values) not in {1, 3} or any(len(vector) != 3 for vector in lattice):
        raise ManuscriptTransportError("POSCAR must contain one/three scale values and 3x3 lattice")
    raw_volume = abs(_determinant_3x3(lattice))
    if raw_volume <= 0:
        raise ManuscriptTransportError("POSCAR lattice has zero volume")
    if len(scale_values) == 1:
        scale = scale_values[0]
        if scale == 0:
            raise ManuscriptTransportError("POSCAR scale must be non-zero")
        volume_a3 = abs(scale) if scale < 0 else raw_volume * scale**3
    else:
        if any(value <= 0 for value in scale_values):
            raise ManuscriptTransportError("three-component POSCAR scale must be positive")
        volume_a3 = raw_volume * math.prod(scale_values)

    species = lines[5].split()
    if not species or all(re.fullmatch(r"\d+", token) for token in species):
        raise ManuscriptTransportError(
            "VASP 4 POSCAR lacks species names; corrected carrier count cannot be inferred"
        )
    try:
        counts = [int(value) for value in lines[6].split()]
    except ValueError as exc:
        raise ManuscriptTransportError("POSCAR atom counts must be integers") from exc
    if len(species) != len(counts) or any(value <= 0 for value in counts):
        raise ManuscriptTransportError("POSCAR species/count rows are inconsistent")
    if mobile_species not in species:
        raise ManuscriptTransportError("POSCAR contains no %s species" % mobile_species)
    n_mobile_ions = sum(count for name, count in zip(species, counts) if name == mobile_species)
    return {
        "n_mobile_ions": n_mobile_ions,
        "mobile_species": mobile_species,
        "volume_A3": volume_a3,
        "species_counts": dict(zip(species, counts)),
        "provenance": provenance,
    }


def parse_legacy_transport_script(path_value: Any) -> Dict[str, Any]:
    """Extract literal N/V/kB/q constants from a historical script; never execute it."""

    text, provenance = _read_parity_source(path_value, "legacy_transport_script")
    assignments: Dict[str, float] = {}
    for name in ("N", "V", "kB", "q", "T"):
        matched = re.search(
            r"(?m)^\s*%s\s*=\s*(%s)(?:\s*#.*)?$" % (re.escape(name), _FLOAT_PATTERN),
            text,
        )
        if matched:
            assignments[name] = float(matched.group(1))
    required = {"N", "V", "kB", "q"}
    if not required.issubset(assignments):
        raise ManuscriptTransportError(
            "legacy transport script lacks literal assignments for %s"
            % sorted(required - set(assignments))
        )
    n_value = assignments["N"]
    if not n_value.is_integer() or n_value <= 0:
        raise ManuscriptTransportError("legacy script N must be a positive integer literal")
    if any(assignments[name] <= 0 for name in ("V", "kB", "q")):
        raise ManuscriptTransportError("legacy script V/kB/q constants must be positive")
    return {
        "n_mobile_ions": int(n_value),
        "volume_m3": assignments["V"],
        "volume_A3": assignments["V"] * 1.0e30,
        "boltzmann_J_K": assignments["kB"],
        "elementary_charge_C": assignments["q"],
        "temperature_K": assignments.get("T"),
        "provenance": provenance,
    }


def _positive_explicit_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ManuscriptTransportError("%s must be an explicit positive integer" % name)
    return value


def _resolve_carrier_convention(
    *,
    convention: str,
    n_mobile_ions: Optional[int],
    volume_a3: Optional[float],
    structure_path: Optional[Any],
    mobile_species: str,
    legacy_script_path: Optional[Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if convention not in {LEGACY_SCRIPT_CONVENTION, COMPOSITION_CORRECTED_CONVENTION}:
        raise ManuscriptTransportError(
            "carrier_convention must be legacy_script or composition_corrected"
        )
    sources: List[Dict[str, Any]] = []
    structure: Optional[Dict[str, Any]] = None
    legacy_script: Optional[Dict[str, Any]] = None
    if structure_path is not None:
        structure = parse_poscar_mobile_ions(structure_path, mobile_species)
        sources.append(structure["provenance"])
    if legacy_script_path is not None:
        legacy_script = parse_legacy_transport_script(legacy_script_path)
        sources.append(legacy_script["provenance"])

    volume_candidates: List[Tuple[str, float]] = []
    if volume_a3 is not None:
        if not _positive_number(volume_a3):
            raise ManuscriptTransportError("volume_a3 must be a finite positive number")
        volume_candidates.append(("explicit", float(volume_a3)))
    if structure is not None:
        volume_candidates.append(("structure", float(structure["volume_A3"])))
    if legacy_script is not None:
        volume_candidates.append(("legacy_script", float(legacy_script["volume_A3"])))
    if not volume_candidates:
        raise ManuscriptTransportError(
            "volume must come from volume_a3, a POSCAR, or a reviewed legacy script"
        )
    resolved_volume = volume_candidates[0][1]
    for source, candidate in volume_candidates[1:]:
        if not math.isclose(candidate, resolved_volume, rel_tol=1e-9, abs_tol=1e-8):
            raise ManuscriptTransportError(
                "volume sources disagree: %.17g versus %.17g A^3 from %s"
                % (resolved_volume, candidate, source)
            )

    if legacy_script is not None:
        if not math.isclose(
            float(legacy_script["boltzmann_J_K"]),
            _HISTORICAL_BOLTZMANN_J_K,
            rel_tol=0.0,
            abs_tol=1e-35,
        ) or not math.isclose(
            float(legacy_script["elementary_charge_C"]),
            _HISTORICAL_ELEMENTARY_CHARGE_C,
            rel_tol=0.0,
            abs_tol=1e-30,
        ):
            raise ManuscriptTransportError(
                "legacy script constants differ from kB=1.38e-23 and q=1.6e-19"
            )

    if convention == LEGACY_SCRIPT_CONVENTION:
        if n_mobile_ions is not None or structure_path is not None:
            raise ManuscriptTransportError(
                "legacy_script fixes N=7; do not provide n_mobile_ions or a structure"
            )
        if legacy_script is not None and legacy_script["n_mobile_ions"] != 7:
            raise ManuscriptTransportError(
                "legacy_script convention requires source-script N=7 exactly"
            )
        count = _LEGACY_LI10_N_MOBILE_IONS
        count_source = "historical Li10 MACE script literal N=7"
        legacy_preserved = True
    else:
        explicit_count = (
            None
            if n_mobile_ions is None
            else _positive_explicit_int(n_mobile_ions, "n_mobile_ions")
        )
        structure_count = None if structure is None else int(structure["n_mobile_ions"])
        if explicit_count is None and structure_count is None:
            raise ManuscriptTransportError(
                "composition_corrected requires explicit n_mobile_ions or a VASP 5 POSCAR"
            )
        if (
            explicit_count is not None
            and structure_count is not None
            and explicit_count != structure_count
        ):
            raise ManuscriptTransportError(
                "explicit n_mobile_ions=%d disagrees with structure count=%d"
                % (explicit_count, structure_count)
            )
        count = explicit_count if explicit_count is not None else int(structure_count)
        count_source = (
            "explicit+structure-verified"
            if explicit_count is not None and structure_count is not None
            else ("explicit" if explicit_count is not None else "structure")
        )
        legacy_preserved = False

    return (
        {
            "name": convention,
            "n_mobile_ions": count,
            "n_mobile_ions_source": count_source,
            "mobile_species": mobile_species,
            "volume_A3": resolved_volume,
            "volume_m3": resolved_volume * 1.0e-30,
            "volume_sources": [name for name, _ in volume_candidates],
            "historical_N7_error_preserved": legacy_preserved,
            "legacy_script_n_mobile_ions": (
                None if legacy_script is None else legacy_script["n_mobile_ions"]
            ),
            "conductivity_model": "uncorrected Nernst-Einstein",
            "dimensions": 3,
            "haven_ratio": 1.0,
            "charge_number": 1.0,
            "constants_convention": "historical rounded constants retained to isolate N/V",
        },
        sources,
    )


def _conductivity_s_m(
    diffusivity_m2_s: float, temperature_k: float, carrier: Mapping[str, Any]
) -> float:
    if diffusivity_m2_s <= 0 or temperature_k <= 0:
        raise ManuscriptTransportError("diffusivity and temperature must be positive")
    try:
        return nernst_einstein_conductivity(
            diffusivity_m2_s,
            temperature_k,
            int(carrier["n_mobile_ions"]),
            float(carrier["charge_number"]),
            float(carrier["volume_A3"]),
            elementary_charge_c=_HISTORICAL_ELEMENTARY_CHARGE_C,
            boltzmann_j_per_k=_HISTORICAL_BOLTZMANN_J_K,
        )["conductivity_s_m"]
    except ValueError as exc:
        raise ManuscriptTransportError(str(exc)) from exc


def _ols(x_values: Sequence[float], y_values: Sequence[float]) -> Dict[str, float]:
    try:
        fit = linear_fit(x_values, y_values)
    except ValueError as exc:
        raise ManuscriptTransportError(str(exc)) from exc
    return {
        "slope": float(fit["slope"]),
        "intercept": float(fit["intercept"]),
        "r2": float(fit["r_squared"]),
    }


def _arrhenius_fit(
    rows: Sequence[Mapping[str, Any]],
    method: str,
    convention: str,
    target_temperature_k: float,
    carrier: Mapping[str, Any],
) -> Dict[str, Any]:
    if not _positive_number(target_temperature_k):
        raise ManuscriptTransportError("target_temperature_k must be positive")
    temperatures = [float(row["temperature_K"]) for row in rows]
    diffusivities = [float(row["methods"][method]["diffusivity_m2_s"]) for row in rows]
    if convention == EXACT_ARRHENIUS_CONVENTION:
        x_values = [1000.0 / value for value in temperatures]
        y_values = [math.log10(value) for value in diffusivities]
        target_x = 1000.0 / float(target_temperature_k)
        rounding = "none"
    elif convention == LEGACY_ARRHENIUS_CONVENTION:
        if set(temperatures) != {400.0, 600.0, 800.0} or len(temperatures) != 3:
            raise ManuscriptTransportError(
                "legacy_get_sigma_v1 requires exactly 400, 600, and 800 K"
            )
        historical_x = {400.0: 2.5, 600.0: 1.667, 800.0: 1.25}
        x_values = [historical_x[value] for value in temperatures]
        y_values = [round(math.log10(value), 2) for value in diffusivities]
        if not math.isclose(float(target_temperature_k), 300.0, rel_tol=0.0, abs_tol=1e-12):
            raise ManuscriptTransportError("legacy_get_sigma_v1 fixes target temperature at 300 K")
        target_x = 3.33
        rounding = "log10(D) rounded to 2 decimals; x600=1.667; x300=3.33"
    else:
        raise ManuscriptTransportError(
            "arrhenius_convention must be exact_unrounded_ols_v1 or legacy_get_sigma_v1"
        )
    fit = _ols(x_values, y_values)
    target_log10_d = fit["slope"] * target_x + fit["intercept"]
    target_d = 10.0**target_log10_d
    target_sigma = _conductivity_s_m(target_d, float(target_temperature_k), carrier)
    historical_kb_ev_k = _HISTORICAL_BOLTZMANN_J_K / _HISTORICAL_ELEMENTARY_CHARGE_C
    activation_energy_ev = -fit["slope"] * historical_kb_ev_k * math.log(10.0) * 1000.0
    result: Dict[str, Any] = {
        "method": method,
        "convention": convention,
        "regression": "unweighted OLS of log10(D_m2_s) against 1000/T_K",
        "historical_script_usage": (
            "direct get_MSD_Li10.py -> get_sigma.py parity path"
            if convention == LEGACY_ARRHENIUS_CONVENTION and method == "mean_msd_over_time"
            else (
                "same legacy transform applied as a secondary comparison; not the original shell path"
                if convention == LEGACY_ARRHENIUS_CONVENTION
                else "wrapper-normalized unrounded Arrhenius fit"
            )
        ),
        "slope_log10_D_per_1000_over_K": fit["slope"],
        "intercept_log10_D": fit["intercept"],
        "r2": fit["r2"],
        "activation_energy_eV": activation_energy_ev,
        "activation_energy_status": "derived by this wrapper from the declared fitted slope",
        "prefactor_m2_s": 10.0 ** fit["intercept"],
        "target_temperature_K": float(target_temperature_k),
        "target_log10_diffusivity_m2_s": target_log10_d,
        "fit_inputs": [
            {
                "temperature_K": temperature,
                "x_1000_over_T_K": x_value,
                "log10_diffusivity_m2_s": y_value,
            }
            for temperature, x_value, y_value in zip(temperatures, x_values, y_values)
        ],
        "rounding": rounding,
    }
    result.update(_transport_value(target_d, "diffusivity"))
    result.update(_transport_value(target_sigma, "conductivity"))
    return result


def _select_historical_dataset(
    parsed: Mapping[str, Any], selector: Optional[str]
) -> Mapping[str, Any]:
    datasets = parsed.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ManuscriptTransportError("parsed historical output has no datasets")
    if selector is None:
        if len(datasets) != 1:
            raise ManuscriptTransportError(
                "historical_dataset is required when stdout contains multiple blocks"
            )
        return datasets[0]
    matches = [item for item in datasets if str(item.get("dataset_id")) == str(selector)]
    if len(matches) != 1:
        raise ManuscriptTransportError(
            "historical_dataset=%s does not select exactly one output block" % selector
        )
    return matches[0]


def _relative_close(actual: float, expected: float, rel_tol: float, abs_tol: float) -> bool:
    return math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol)


def _compare_historical_dataset(
    *,
    reproduced_rows: Sequence[Mapping[str, Any]],
    parsed: Mapping[str, Any],
    selector: Optional[str],
    reproduction_carrier: Mapping[str, Any],
    historical_carrier_convention: str,
    rel_tol: float = 1e-12,
    abs_tol: float = 1e-20,
) -> Dict[str, Any]:
    if historical_carrier_convention not in {
        LEGACY_SCRIPT_CONVENTION,
        "same_as_reproduction",
        "unknown",
    }:
        raise ManuscriptTransportError(
            "historical_carrier_convention must be legacy_script, same_as_reproduction, or unknown"
        )
    dataset = _select_historical_dataset(parsed, selector)
    historical_by_temperature = {
        float(row["temperature_K"]): row for row in dataset["by_temperature"]
    }
    reproduced_by_temperature = {float(row["temperature_K"]): row for row in reproduced_rows}
    if set(historical_by_temperature) != set(reproduced_by_temperature):
        raise ManuscriptTransportError(
            "historical and reproduced temperature sets differ: %s versus %s"
            % (
                sorted(historical_by_temperature),
                sorted(reproduced_by_temperature),
            )
        )
    if historical_carrier_convention == LEGACY_SCRIPT_CONVENTION:
        historical_n: Optional[int] = _LEGACY_LI10_N_MOBILE_IONS
    elif historical_carrier_convention == "same_as_reproduction":
        historical_n = int(reproduction_carrier["n_mobile_ions"])
    else:
        historical_n = None
    reproduced_n = int(reproduction_carrier["n_mobile_ions"])
    expected_sigma_ratio = (
        None if historical_n is None else float(reproduced_n) / float(historical_n)
    )

    comparisons: List[Dict[str, Any]] = []
    diffusion_close: List[bool] = []
    conductivity_close: List[bool] = []
    conductivity_ratio_close: List[bool] = []
    for temperature in sorted(reproduced_by_temperature):
        historical_row = historical_by_temperature[temperature]
        reproduced_row = reproduced_by_temperature[temperature]
        for method in ("linear_msd_fit", "mean_msd_over_time"):
            historical_method = historical_row["methods"][method]
            reproduced_method = reproduced_row["methods"][method]
            historical_d = float(historical_method["diffusivity_m2_s"])
            reproduced_d = float(reproduced_method["diffusivity_m2_s"])
            d_close = _relative_close(reproduced_d, historical_d, rel_tol, abs_tol)
            diffusion_close.append(d_close)
            historical_sigma = float(historical_method["conductivity_S_m"])
            reproduced_sigma = float(reproduced_method["conductivity_S_m"])
            sigma_close = _relative_close(reproduced_sigma, historical_sigma, rel_tol, abs_tol)
            conductivity_close.append(sigma_close)
            actual_ratio = reproduced_sigma / historical_sigma
            ratio_close: Optional[bool] = None
            if expected_sigma_ratio is not None:
                ratio_close = _relative_close(actual_ratio, expected_sigma_ratio, rel_tol, abs_tol)
                conductivity_ratio_close.append(ratio_close)
            comparisons.append(
                {
                    "temperature_K": temperature,
                    "method": method,
                    "historical_diffusivity_m2_s": historical_d,
                    "reproduced_diffusivity_m2_s": reproduced_d,
                    "diffusivity_close": d_close,
                    "historical_conductivity_S_m": historical_sigma,
                    "reproduced_conductivity_S_m": reproduced_sigma,
                    "conductivity_close": sigma_close,
                    "conductivity_ratio_reproduced_over_historical": actual_ratio,
                    "expected_conductivity_ratio": expected_sigma_ratio,
                    "conductivity_ratio_close": ratio_close,
                }
            )
    all_d = all(diffusion_close)
    all_sigma = all(conductivity_close)
    all_ratio = bool(conductivity_ratio_close) and all(conductivity_ratio_close)
    if all_d and all_sigma:
        status = "EXACT_NUMERICAL_PARITY"
    elif all_d and expected_sigma_ratio is not None and all_ratio:
        status = "EXPECTED_CARRIER_CONVENTION_DIVERGENCE"
    elif all_d and historical_n is None:
        status = "DIFFUSIVITY_PARITY_CONDUCTIVITY_UNASSESSED"
    else:
        status = "PARITY_MISMATCH"
    return {
        "status": status,
        "historical_dataset_id": dataset["dataset_id"],
        "historical_carrier_convention": historical_carrier_convention,
        "historical_n_mobile_ions": historical_n,
        "reproduced_n_mobile_ions": reproduced_n,
        "diffusivity_all_close": all_d,
        "conductivity_all_close": all_sigma,
        "conductivity_ratio_all_close": all_ratio if historical_n is not None else None,
        "relative_tolerance": rel_tol,
        "absolute_tolerance": abs_tol,
        "comparisons": comparisons,
    }


def reproduce_manuscript_transport(
    target_msd_by_temperature: Mapping[Any, Any],
    *,
    sampling_profile: str,
    carrier_convention: str,
    arrhenius_convention: str,
    target_temperature_k: float = 300.0,
    n_mobile_ions: Optional[int] = None,
    volume_a3: Optional[float] = None,
    structure_path: Optional[Any] = None,
    mobile_species: str = "Li",
    legacy_script_path: Optional[Any] = None,
    historical_output_path: Optional[Any] = None,
    historical_dataset: Optional[str] = None,
    historical_carrier_convention: Optional[str] = None,
) -> Dict[str, Any]:
    """Reproduce a local three-temperature transport series from existing MSD files.

    This is a post-processing wrapper only. It never launches MD, imports a calculator,
    submits a job, or changes any evidence file.
    """

    if not isinstance(target_msd_by_temperature, Mapping) or len(target_msd_by_temperature) < 2:
        raise ManuscriptTransportError(
            "target_msd_by_temperature must map at least two temperatures to target.msd files"
        )
    normalized_inputs: List[Tuple[float, Any]] = []
    for raw_temperature, path in target_msd_by_temperature.items():
        try:
            temperature = float(raw_temperature)
        except (TypeError, ValueError) as exc:
            raise ManuscriptTransportError("temperature keys must be numeric") from exc
        if not math.isfinite(temperature) or temperature <= 0:
            raise ManuscriptTransportError("temperature keys must be finite and positive")
        normalized_inputs.append((temperature, path))
    if len({item[0] for item in normalized_inputs}) != len(normalized_inputs):
        raise ManuscriptTransportError("target MSD mapping contains duplicate temperatures")
    normalized_inputs.sort(key=lambda item: item[0])

    carrier, carrier_sources = _resolve_carrier_convention(
        convention=carrier_convention,
        n_mobile_ions=n_mobile_ions,
        volume_a3=volume_a3,
        structure_path=structure_path,
        mobile_species=mobile_species,
        legacy_script_path=legacy_script_path,
    )
    rows: List[Dict[str, Any]] = []
    source_provenance: List[Dict[str, Any]] = list(carrier_sources)
    for temperature, target_path in normalized_inputs:
        analyzed = analyze_historical_target_msd(target_path, sampling_profile)
        methods: Dict[str, Any] = {}
        for method_name, method_result in analyzed["methods"].items():
            normalized_method = dict(method_result)
            sigma = _conductivity_s_m(
                float(method_result["diffusivity_m2_s"]), temperature, carrier
            )
            normalized_method.update(_transport_value(sigma, "conductivity"))
            methods[method_name] = normalized_method
        rows.append(
            {
                "temperature_K": temperature,
                "methods": methods,
                "msd_fit": analyzed["fit"],
                "sampling": analyzed["sampling"],
            }
        )
        source_provenance.extend(analyzed["provenance"]["sources"])

    arrhenius = {
        method: _arrhenius_fit(
            rows,
            method,
            arrhenius_convention,
            float(target_temperature_k),
            carrier,
        )
        for method in ("linear_msd_fit", "mean_msd_over_time")
    }
    parity: Optional[Dict[str, Any]] = None
    if historical_output_path is not None:
        if historical_carrier_convention is None:
            raise ManuscriptTransportError(
                "historical_carrier_convention is required when comparing historical stdout"
            )
        parsed_history = parse_historical_transport_output(historical_output_path)
        parity = _compare_historical_dataset(
            reproduced_rows=rows,
            parsed=parsed_history,
            selector=historical_dataset,
            reproduction_carrier=carrier,
            historical_carrier_convention=historical_carrier_convention,
        )
        source_provenance.extend(parsed_history["provenance"]["sources"])
    elif historical_dataset is not None or historical_carrier_convention is not None:
        raise ManuscriptTransportError(
            "historical dataset/convention options require historical_output_path"
        )

    return {
        "schema_version": 1,
        "artifact_type": "mlipipe.manuscript_transport_parity",
        "status": "OK",
        "execution": {
            "mode": "read-only-postprocess-existing-msd",
            "md_executed": False,
            "scheduler_used": False,
            "network_used": False,
        },
        "carrier_convention": carrier,
        "sampling_convention": {
            "profile": sampling_profile,
            **_sampling_profile(sampling_profile),
            "same_sampling_used_for_all_carrier_conventions": True,
        },
        "by_temperature": rows,
        "arrhenius": {
            "convention": arrhenius_convention,
            "methods": arrhenius,
        },
        "parity": parity,
        "units": {
            "temperature_K": "K",
            "raw_timestep": "LAMMPS step",
            "interpreted_timestep": "fs",
            "msd": "angstrom^2",
            "volume_A3": "angstrom^3",
            "diffusivity_m2_s": "m^2/s",
            "diffusivity_cm2_s": "cm^2/s",
            "conductivity_S_m": "S/m",
            "conductivity_NE_mS_cm": "mS/cm",
            "activation_energy_eV": "eV",
        },
        "constants": {
            "elementary_charge_C": _HISTORICAL_ELEMENTARY_CHARGE_C,
            "boltzmann_J_K": _HISTORICAL_BOLTZMANN_J_K,
            "reason": (
                "historical rounded q/kB retained in both modes so the corrected result "
                "changes only the explicit carrier convention"
            ),
        },
        "provenance": {
            "wrapper": "ionic-transport.adapter.reproduce_manuscript_transport",
            "wrapper_version": MANUSCRIPT_PARITY_VERSION,
            "sources": source_provenance,
        },
    }


def _parse_temperature_paths(values: Sequence[str]) -> Dict[float, str]:
    result: Dict[float, str] = {}
    for value in values:
        if "=" not in value:
            raise ManuscriptTransportError("--run must use TEMPERATURE=PATH")
        raw_temperature, raw_path = value.split("=", 1)
        try:
            temperature = float(raw_temperature)
        except ValueError as exc:
            raise ManuscriptTransportError("--run temperature must be numeric") from exc
        if temperature in result:
            raise ManuscriptTransportError("--run repeats temperature %.17g" % temperature)
        if not raw_path.strip():
            raise ManuscriptTransportError("--run path must be non-empty")
        result[temperature] = raw_path
    return result


def _emit_parity_json(result: Mapping[str, Any], output: Optional[str]) -> None:
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output is None:
        sys.stdout.write(rendered)
        return
    path = Path(output).expanduser().resolve()
    if not path.parent.is_dir():
        raise ManuscriptTransportError("output parent directory does not exist: %s" % path.parent)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    except FileExistsError as exc:
        raise ManuscriptTransportError(
            "output already exists; parity wrapper never overwrites: %s" % path
        ) from exc


def manuscript_transport_main(argv: Optional[Sequence[str]] = None) -> int:
    """Plugin-local read-only evidence wrapper; this is not a top-level MLIPipe CLI."""

    parser = argparse.ArgumentParser(
        description="Read-only parser/parity wrapper for historical Li10 transport evidence."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    history_parser = subparsers.add_parser(
        "parse-historical-output", help="normalize Li10_test.out or Li10_MACE.out"
    )
    history_parser.add_argument("source")
    history_parser.add_argument("--output")

    reproduce_parser = subparsers.add_parser(
        "reproduce-msd", help="post-process existing target.msd files; never run MD"
    )
    reproduce_parser.add_argument(
        "--run", action="append", required=True, help="repeat TEMPERATURE=PATH"
    )
    reproduce_parser.add_argument(
        "--sampling-profile", choices=sorted(_SAMPLING_PROFILES), required=True
    )
    reproduce_parser.add_argument(
        "--carrier-convention",
        choices=[LEGACY_SCRIPT_CONVENTION, COMPOSITION_CORRECTED_CONVENTION],
        required=True,
    )
    reproduce_parser.add_argument(
        "--arrhenius-convention",
        choices=[EXACT_ARRHENIUS_CONVENTION, LEGACY_ARRHENIUS_CONVENTION],
        required=True,
    )
    reproduce_parser.add_argument("--target-temperature-K", type=float, default=300.0)
    reproduce_parser.add_argument("--n-mobile-ions", type=int)
    reproduce_parser.add_argument("--volume-A3", type=float)
    reproduce_parser.add_argument("--structure")
    reproduce_parser.add_argument("--mobile-species", default="Li")
    reproduce_parser.add_argument("--legacy-script")
    reproduce_parser.add_argument("--historical-output")
    reproduce_parser.add_argument("--historical-dataset")
    reproduce_parser.add_argument(
        "--historical-carrier-convention",
        choices=[LEGACY_SCRIPT_CONVENTION, "same_as_reproduction", "unknown"],
    )
    reproduce_parser.add_argument("--output")

    arguments = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if arguments.command == "parse-historical-output":
            result = parse_historical_transport_output(arguments.source)
            _emit_parity_json(result, arguments.output)
            return 0
        target_paths = _parse_temperature_paths(arguments.run)
        result = reproduce_manuscript_transport(
            target_paths,
            sampling_profile=arguments.sampling_profile,
            carrier_convention=arguments.carrier_convention,
            arrhenius_convention=arguments.arrhenius_convention,
            target_temperature_k=arguments.target_temperature_K,
            n_mobile_ions=arguments.n_mobile_ions,
            volume_a3=arguments.volume_A3,
            structure_path=arguments.structure,
            mobile_species=arguments.mobile_species,
            legacy_script_path=arguments.legacy_script,
            historical_output_path=arguments.historical_output,
            historical_dataset=arguments.historical_dataset,
            historical_carrier_convention=arguments.historical_carrier_convention,
        )
        _emit_parity_json(result, arguments.output)
        return 0
    except ManuscriptTransportError as exc:
        parser.error(str(exc))
    return 2  # pragma: no cover - argparse.error raises SystemExit


if __name__ == "__main__":
    raise SystemExit(manuscript_transport_main())
