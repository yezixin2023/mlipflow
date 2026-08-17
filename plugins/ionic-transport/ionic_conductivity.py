#!/usr/bin/env python3
"""Post-process Li diffusion, ionic conductivity, and Arrhenius Ea.
It does not run MD. It can read either:

* ASE folders containing production.traj and metadata.json
* LAMMPS folders containing traj.lammpstrj/data.LYC
* VASP AIMD folders containing vasprun.xml
* Precomputed MSD tables such as msd.dat, msd.csv, and msd_after_discard.csv

The Arrhenius fit is always performed as a linear fit in ln(D) vs 1/T.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    import mlipflow.science.transport as _transport_core
except ModuleNotFoundError:  # Source-checkout execution before editable installation.
    _source_root = Path(__file__).resolve().parents[2] / "src"
    if not (_source_root / "mlipflow" / "science" / "transport.py").is_file():
        raise
    sys.path.insert(0, str(_source_root))
    import mlipflow.science.transport as _transport_core

from mlipflow.science.transport import (
    arrhenius_from_diffusivities,
    linear_diffusion_from_msd,
    linear_fit,
    nernst_einstein_conductivity,
)


KNOWN_RUN_FILES = {
    "production.traj",
    "traj.lammpstrj",
    "vasprun.xml",
    "msd.dat",
    "MSD.dat",
    "msd.csv",
    "MSD.csv",
    "msd.txt",
    "MSD.txt",
    "msd.xvg",
    "MSD.xvg",
    "msd_after_discard.csv",
}


@dataclass
class RunData:
    dataset: str
    run_dir: Path
    temperature_K: float
    source_kind: str
    source_path: Path
    time_ps: np.ndarray
    msd_A2: np.ndarray
    n_mobile: int | None
    volume_A3: float | None
    drift_correction: str
    notes: str = ""


@dataclass
class LinearFit:
    slope: float
    intercept: float
    r2: float
    sse: float
    slope_stderr: float | None
    n_points: int


def finite_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return finite_or_none(value)
    if isinstance(value, Path):
        return str(value)
    return finite_or_none(value) if isinstance(value, float) else value


def parse_float_list(text: str | None) -> list[float]:
    if text is None or text == "":
        return []
    return [float(part) for part in re.split(r"[,\s]+", text.strip()) if part]


def is_close_to_any(values: np.ndarray, targets: Iterable[float], atol: float = 1e-8):
    targets = np.asarray(list(targets), dtype=float)
    if targets.size == 0:
        return np.zeros(values.shape, dtype=bool)
    return np.any(np.isclose(values[:, None], targets[None, :], rtol=0, atol=atol), axis=1)


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        match = re.search(r"[-+]?[0-9]+(?:\.[0-9]+)?", value)
        if not match:
            return None
        value = match.group(0)
    try:
        return float(value)
    except Exception:
        return None


def temperature_from_metadata(path: Path) -> float | None:
    """Read explicit temperature metadata from this run or one of its parents."""
    keys = ("temperature_K", "temperature", "temp_K", "T_K", "T")
    for folder in (path, *path.parents):
        metadata = folder / "metadata.json"
        if not metadata.exists():
            continue
        try:
            data = json.loads(metadata.read_text())
        except Exception:
            continue
        for key in keys:
            temperature = _float_or_none(data.get(key))
            if temperature is not None:
                return temperature
    return None


def temperature_from_path(path: Path) -> float | None:
    """Infer temperature from metadata first, then conservative directory-name patterns."""
    metadata_temp = temperature_from_metadata(path)
    if metadata_temp is not None:
        return metadata_temp

    patterns = [
        r"^T[_-]?([0-9]+(?:\.[0-9]+)?)(?:K)?$",
        r"^([0-9]+(?:\.[0-9]+)?)[_-]?K(?:elvin)?$",
        r"^temp(?:erature)?[_-]?([0-9]+(?:\.[0-9]+)?)(?:K)?$",
    ]
    for part in reversed(path.parts):
        token = str(part).strip()
        for pattern in patterns:
            match = re.fullmatch(pattern, token, flags=re.IGNORECASE)
            if match:
                return float(match.group(1))

    return None


def args_temperature_override(args, attr: str | None = None) -> float | None:
    if attr is not None and getattr(args, attr, None) is not None:
        return float(getattr(args, attr))
    if getattr(args, "temperature_K", None) is not None:
        return float(args.temperature_K)
    return None


def require_temperature(path: Path, source_label: str, args=None, override_attr: str | None = None) -> float:
    if args is not None:
        override = args_temperature_override(args, override_attr)
        if override is not None:
            return override

    temperature = temperature_from_path(path)
    if temperature is None:
        raise ValueError(
            f"Could not infer temperature for {source_label} in {path}. "
            "Use --temperature-K for a single-temperature run, use the source-specific "
            "override such as --msd-temperature-K or --aimd-temperature-K, add metadata.json "
            "with 'temperature_K', or use a strict directory name such as T800, 800K, "
            "or temp_800K."
        )
    return float(temperature)


def require_temperature_from_path(path: Path, source_label: str) -> float:
    return require_temperature(path, source_label)


def is_msd_filename(name: str) -> bool:
    lower = name.lower()
    return "msd" in lower and Path(lower).suffix in {".dat", ".csv", ".txt", ".tsv", ".xvg"}


def discover_run_dirs(input_paths: list[Path]) -> list[tuple[str, Path]]:
    run_dirs: dict[Path, str] = {}
    for input_path in input_paths:
        root = input_path.expanduser().resolve()
        if root.is_file():
            dataset = root.parent.name if root.name in KNOWN_RUN_FILES or is_msd_filename(root.name) else root.name
            run_dirs[root.parent.resolve()] = dataset
            continue
        dataset = root.name if root.name else root.parent.name
        if not root.exists():
            raise FileNotFoundError(f"Input path does not exist: {root}")
        for dirpath, _, filenames in os.walk(root):
            names = set(filenames)
            if names & KNOWN_RUN_FILES or any(is_msd_filename(name) for name in names):
                run_dirs[Path(dirpath).resolve()] = dataset
    return sorted(((dataset, path) for path, dataset in run_dirs.items()), key=lambda x: str(x[1]))


def infer_ase_dt_fs(run_dir: Path, override_fs: float | None) -> float:
    if override_fs is not None:
        return float(override_fs)

    metadata = run_dir / "metadata.json"
    if metadata.exists():
        try:
            meta = json.loads(metadata.read_text())
            value = meta.get("time_step_fs_between_frames")
            if value is not None:
                return float(value)
        except Exception:
            pass

    log_path = run_dir / "production_log.csv"
    if log_path.exists():
        try:
            log = pd.read_csv(log_path)
            if "time_fs" in log.columns and len(log) > 1:
                diffs = np.diff(log["time_fs"].to_numpy(dtype=float))
                diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
                if diffs.size:
                    return float(np.median(diffs))
        except Exception:
            pass

    raise ValueError(
        "ASE frame spacing is not declared by metadata.json/production_log.csv; "
        "pass --ase-frame-step-fs explicitly"
    )


def slice_by_time_indices(times_ps: np.ndarray, start_ps: float | None, end_ps: float | None) -> np.ndarray:
    mask = np.ones(times_ps.shape, dtype=bool)
    if start_ps is not None:
        mask &= times_ps >= float(start_ps)
    if end_ps is not None:
        mask &= times_ps <= float(end_ps)
    return np.flatnonzero(mask)


def diffusion_analyzer_smoothed_arg(args):
    if args.diffusion_analyzer_smoothed == "none":
        return False
    return args.diffusion_analyzer_smoothed


def diffusion_analyzer_from_structures(structures, specie: str, temperature_K: float, time_step_fs: float, args):
    """Run pymatgen DiffusionAnalyzer and return analyzer plus its MSD/time arrays."""
    try:
        from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer
    except Exception as exc:
        raise ImportError(
            "pymatgen-analysis-diffusion is required for DiffusionAnalyzer. "
            "Install it in the same environment as pymatgen, e.g. `pip install pymatgen-analysis-diffusion`."
        ) from exc

    step_skip = int(args.diffusion_analyzer_step_skip)
    if step_skip < 1:
        raise ValueError("--diffusion-analyzer-step-skip must be >= 1")
    structures = list(structures)[::step_skip]
    if len(structures) < 2:
        raise ValueError("DiffusionAnalyzer needs at least two structures after step skipping")

    smoothed = diffusion_analyzer_smoothed_arg(args)
    analyzer = DiffusionAnalyzer.from_structures(
        structures,
        specie=specie,
        temperature=float(temperature_K),
        time_step=float(time_step_fs),
        step_skip=step_skip,
        smoothed=smoothed,
        min_obs=int(args.diffusion_analyzer_min_obs),
        avg_nsteps=int(args.diffusion_analyzer_avg_nsteps),
    )
    msd = np.asarray(analyzer.msd, dtype=float)
    if msd.ndim != 1 or msd.size < 2:
        raise ValueError("DiffusionAnalyzer returned an invalid or too-short MSD array")

    time_ps = np.arange(msd.size, dtype=float) * float(time_step_fs) * step_skip / 1000.0
    return analyzer, time_ps, msd


def pymatgen_structures_from_frames(symbols, basis_frames: np.ndarray, frac_frames: np.ndarray):
    try:
        from pymatgen.core import Lattice, Structure
    except Exception as exc:
        raise ImportError("pymatgen is required to build Structure objects for DiffusionAnalyzer") from exc

    return [
        Structure(
            lattice=Lattice(np.asarray(basis, dtype=float)),
            species=list(symbols),
            coords=np.asarray(frac, dtype=float),
            coords_are_cartesian=False,
            to_unit_cell=True,
        )
        for basis, frac in zip(basis_frames, frac_frames)
    ]


def compute_msd_series(
    positions: np.ndarray,
    time_ps: np.ndarray,
    mobile_mask: np.ndarray,
    framework_mask: np.ndarray,
    drift_correction: str,
    msd_mode: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    mobile_positions = positions[:, mobile_mask, :].copy()
    used_correction = drift_correction

    if drift_correction == "framework":
        if framework_mask.any():
            frame_disp = positions[:, framework_mask, :] - positions[0:1, framework_mask, :]
            drift = frame_disp.mean(axis=1)
            mobile_positions = mobile_positions - drift[:, None, :]
        else:
            used_correction = "none"
    elif drift_correction == "mobile":
        mobile_disp_from_first = mobile_positions - mobile_positions[0:1, :, :]
        drift = mobile_disp_from_first.mean(axis=1)
        mobile_positions = mobile_positions - drift[:, None, :]
    elif drift_correction != "none":
        raise ValueError("drift correction must be one of: none, framework, mobile")

    if msd_mode == "single-origin":
        mobile_disp = mobile_positions - mobile_positions[0:1, :, :]
        msd = np.mean(np.sum(mobile_disp**2, axis=2), axis=1)
        return time_ps, msd, used_correction

    if msd_mode != "multi-origin":
        raise ValueError("msd_mode must be one of: multi-origin, single-origin")

    n_frames, n_mobile, _ = mobile_positions.shape
    lag_times = time_ps - time_ps[0]
    msd = multi_origin_msd_fft(mobile_positions.reshape(n_frames, -1), n_mobile)

    return lag_times, msd, used_correction


def multi_origin_msd_fft(flat_positions: np.ndarray, n_mobile: int) -> np.ndarray:
    """Time-averaged MSD using FFT autocorrelation.

    flat_positions has shape (n_frames, n_mobile * 3), with unwrapped Cartesian
    coordinates after any selected drift correction.
    """
    positions = np.asarray(flat_positions, dtype=float)
    n_frames = positions.shape[0]
    if n_frames < 2:
        raise ValueError("Need at least two frames for MSD.")

    squared = np.sum(positions * positions, axis=1)
    n_fft = 1 << (2 * n_frames - 1).bit_length()
    fft_pos = np.fft.rfft(positions, n=n_fft, axis=0)
    autocorr = np.fft.irfft(fft_pos * np.conjugate(fft_pos), n=n_fft, axis=0)[:n_frames]
    autocorr_sum = np.sum(autocorr, axis=1)

    prefix = np.concatenate(([0.0], np.cumsum(squared)))
    lags = np.arange(n_frames)
    counts = n_frames - lags
    sum_first = prefix[n_frames - lags] - prefix[0]
    sum_second = prefix[n_frames] - prefix[lags]
    numerator = sum_first + sum_second - 2.0 * autocorr_sum
    msd = numerator / (counts * n_mobile)
    msd[0] = 0.0
    return np.maximum(msd, 0.0)


def load_ase_trajectory(run_dir: Path, dataset: str, args) -> RunData:
    try:
        from ase.io import read
    except ImportError as exc:
        raise ImportError(
            "ASE is required to read production.traj; install ASE in the selected Python runtime"
        ) from exc

    traj_path = run_dir / "production.traj"
    frames = read(str(traj_path), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    if len(frames) < 2:
        raise ValueError(f"Need at least two frames in {traj_path}")

    dt_fs = infer_ase_dt_fs(run_dir, args.ase_frame_step_fs)
    raw_times_ps = np.arange(len(frames), dtype=float) * dt_fs / 1000.0
    idx = slice_by_time_indices(raw_times_ps, args.trajectory_start_ps, args.trajectory_end_ps)
    if idx.size < 2:
        raise ValueError(f"Trajectory segment leaves fewer than two frames: {traj_path}")

    frames = [frames[i] for i in idx]
    time_ps = raw_times_ps[idx] - raw_times_ps[idx[0]]

    symbols = np.asarray(frames[0].get_chemical_symbols())
    mobile_mask = symbols == args.specie
    framework_mask = ~mobile_mask
    if not mobile_mask.any():
        raise ValueError(f"No {args.specie} atoms found in {traj_path}")

    temperature_K = require_temperature(run_dir, "ASE trajectory", args)

    if args.trajectory_msd_engine in {"diffusion-analyzer", "auto"}:
        try:
            basis_frames = np.asarray([np.asarray(atoms.get_cell().array, dtype=float) for atoms in frames], dtype=float)
            frac_frames = np.asarray([np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float) for atoms in frames], dtype=float)
            structures = pymatgen_structures_from_frames(symbols, basis_frames, frac_frames)
            analyzer, da_time_ps, da_msd = diffusion_analyzer_from_structures(
                structures=structures,
                specie=args.specie,
                temperature_K=temperature_K,
                time_step_fs=dt_fs,
                args=args,
            )
            return RunData(
                dataset=dataset,
                run_dir=run_dir,
                temperature_K=temperature_K,
                source_kind="ase_diffusion_analyzer",
                source_path=traj_path,
                time_ps=da_time_ps,
                msd_A2=da_msd,
                n_mobile=int(mobile_mask.sum()),
                volume_A3=float(frames[0].get_volume()),
                drift_correction="DiffusionAnalyzer",
                notes=(
                    f"DiffusionAnalyzer.from_structures; dt_fs_between_frames={dt_fs:g}; "
                    f"da_smoothed={args.diffusion_analyzer_smoothed}; "
                    f"da_step_skip={args.diffusion_analyzer_step_skip}; "
                    f"da_diffusivity_cm2_s={getattr(analyzer, 'diffusivity', None)}"
                ),
            )
        except Exception:
            if args.trajectory_msd_engine == "diffusion-analyzer":
                raise

    cell0 = np.asarray(frames[0].get_cell().array, dtype=float)
    scaled0 = np.asarray(frames[0].get_scaled_positions(wrap=False), dtype=float)
    acc_scaled = scaled0.copy()
    prev_scaled = scaled0.copy()
    unwrapped = [acc_scaled @ cell0]

    for atoms in frames[1:]:
        scaled = np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float)
        delta = scaled - prev_scaled
        delta -= np.round(delta)
        acc_scaled = acc_scaled + delta
        unwrapped.append(acc_scaled @ cell0)
        prev_scaled = scaled

    positions = np.asarray(unwrapped, dtype=float)
    time_ps, msd, used_correction = compute_msd_series(
        positions=positions,
        time_ps=time_ps,
        mobile_mask=mobile_mask,
        framework_mask=framework_mask,
        drift_correction=args.drift_correction,
        msd_mode=args.msd_mode,
    )

    return RunData(
        dataset=dataset,
        run_dir=run_dir,
        temperature_K=temperature_K,
        source_kind="ase_numpy_msd",
        source_path=traj_path,
        time_ps=time_ps,
        msd_A2=msd,
        n_mobile=int(mobile_mask.sum()),
        volume_A3=float(frames[0].get_volume()),
        drift_correction=used_correction,
        notes=f"numpy_fallback; dt_fs_between_frames={dt_fs:g}; msd_mode={args.msd_mode}",
    )


def parse_lammps_data(data_path: Path, specie: str, mobile_type: int | None = None):
    if not data_path.exists():
        return None, None, {}

    lines = data_path.read_text(errors="replace").splitlines()
    bounds = {}
    type_to_element = {}
    atom_types = []
    section = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.endswith("xlo xhi"):
            parts = line.split()
            bounds["x"] = (float(parts[0]), float(parts[1]))
            continue
        if line.endswith("ylo yhi"):
            parts = line.split()
            bounds["y"] = (float(parts[0]), float(parts[1]))
            continue
        if line.endswith("zlo zhi"):
            parts = line.split()
            bounds["z"] = (float(parts[0]), float(parts[1]))
            continue

        if line.startswith("Masses"):
            section = "masses"
            continue
        if line.startswith("Atoms"):
            section = "atoms"
            continue

        if section == "masses":
            if re.match(r"^[A-Za-z]", line):
                section = None
                continue
            parts = line.split("#", 1)
            values = parts[0].split()
            if len(values) >= 2 and parts[1:]:
                try:
                    type_to_element[int(values[0])] = parts[1].strip().split()[0]
                except Exception:
                    pass
            continue

        if section == "atoms":
            if re.match(r"^[A-Za-z]", line):
                section = None
                continue
            values = line.split("#", 1)[0].split()
            if len(values) >= 2:
                try:
                    atom_types.append(int(values[1]))
                except ValueError:
                    pass

    volume_A3 = None
    if all(axis in bounds for axis in ("x", "y", "z")):
        volume_A3 = (
            (bounds["x"][1] - bounds["x"][0])
            * (bounds["y"][1] - bounds["y"][0])
            * (bounds["z"][1] - bounds["z"][0])
        )

    if mobile_type is None:
        matches = [typ for typ, element in type_to_element.items() if element == specie]
        mobile_type = matches[0] if matches else None

    n_mobile = None
    if mobile_type is not None and atom_types:
        n_mobile = int(sum(typ == mobile_type for typ in atom_types))

    return n_mobile, volume_A3, type_to_element


def infer_lammps_timestep_ps(run_dir: Path, override_ps: float | None) -> float:
    if override_ps is not None:
        return float(override_ps)

    for input_path in (run_dir / "in.mace_cond", run_dir / "log.lammps"):
        if not input_path.exists():
            continue
        text = input_path.read_text(errors="replace")
        variables = {
            name: float(value)
            for name, value in re.findall(
                r"^\s*variable\s+(\S+)\s+equal\s+([0-9.eE+-]+)\s*$",
                text,
                flags=re.MULTILINE,
            )
        }
        for value in re.findall(r"^\s*timestep\s+(\S+)\s*$", text, flags=re.MULTILINE):
            if value.startswith("${") and value.endswith("}"):
                name = value[2:-1]
                if name in variables:
                    return float(variables[name])
            else:
                try:
                    return float(value)
                except ValueError:
                    pass

    raise ValueError(
        "LAMMPS timestep is not declared by in.mace_cond/log.lammps; "
        "pass --lammps-timestep-ps explicitly"
    )


def read_lammps_dump_frames(dump_path: Path):
    with dump_path.open(errors="replace") as handle:
        while True:
            line = handle.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            timestep = int(handle.readline().strip())
            if not handle.readline().startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError(f"Unexpected dump format near timestep {timestep}")
            n_atoms = int(handle.readline().strip())
            box_header = handle.readline().strip()
            if not box_header.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"Unexpected dump box format near timestep {timestep}")
            box_lines = [handle.readline().strip() for _ in range(3)]
            atoms_header = handle.readline().strip().split()
            if len(atoms_header) < 3 or atoms_header[:2] != ["ITEM:", "ATOMS"]:
                raise ValueError(f"Unexpected atom header near timestep {timestep}")
            columns = atoms_header[2:]
            rows = [handle.readline().strip().split() for _ in range(n_atoms)]
            yield timestep, columns, rows, box_header, box_lines


def parse_lammps_dump_box(box_header: str, box_lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return origin and cell rows for a LAMMPS dump box."""
    bounds = [line.split() for line in box_lines]
    if len(bounds) != 3 or any(len(row) < 2 for row in bounds):
        raise ValueError(f"Could not parse LAMMPS BOX BOUNDS: {box_lines}")

    has_tilt = "xy" in box_header and len(bounds[0]) >= 3 and len(bounds[1]) >= 3 and len(bounds[2]) >= 3
    if has_tilt:
        xlo_bound, xhi_bound, xy = map(float, bounds[0][:3])
        ylo_bound, yhi_bound, xz = map(float, bounds[1][:3])
        zlo_bound, zhi_bound, yz = map(float, bounds[2][:3])
        xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
        xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
        ylo = ylo_bound - min(0.0, yz)
        yhi = yhi_bound - max(0.0, yz)
        zlo, zhi = zlo_bound, zhi_bound
    else:
        xlo, xhi = map(float, bounds[0][:2])
        ylo, yhi = map(float, bounds[1][:2])
        zlo, zhi = map(float, bounds[2][:2])
        xy = xz = yz = 0.0

    cell = np.asarray(
        [
            [xhi - xlo, 0.0, 0.0],
            [xy, yhi - ylo, 0.0],
            [xz, yz, zhi - zlo],
        ],
        dtype=float,
    )
    origin = np.asarray([xlo, ylo, zlo], dtype=float)
    return origin, cell


def lammps_scaled_to_cartesian(scaled: np.ndarray, origin: np.ndarray, cell: np.ndarray) -> np.ndarray:
    return origin + scaled @ cell


def load_lammps_trajectory(run_dir: Path, dataset: str, args) -> RunData:
    dump_path = run_dir / "traj.lammpstrj"
    timestep_ps = infer_lammps_timestep_ps(run_dir, args.lammps_timestep_ps)
    frames = []
    timesteps = []
    ids_ref = None
    mobile_mask = None
    framework_mask = None

    coordinate_mode = None

    for timestep, columns, rows, box_header, box_lines in read_lammps_dump_frames(dump_path):
        col = {name: i for i, name in enumerate(columns)}
        if "id" not in col:
            raise ValueError(f"LAMMPS dump lacks an id column: {dump_path}")
        if all(name in col for name in ("xu", "yu", "zu")):
            this_mode = "unwrapped_cartesian"
            coord_names = ("xu", "yu", "zu")
        elif all(name in col for name in ("xsu", "ysu", "zsu")):
            this_mode = "unwrapped_scaled"
            coord_names = ("xsu", "ysu", "zsu")
        elif all(name in col for name in ("x", "y", "z")) or all(name in col for name in ("xs", "ys", "zs")):
            raise ValueError(
                "LAMMPS dump contains wrapped coordinates (x/y/z or xs/ys/zs). "
                "These are not safe for MSD. Dump unwrapped Cartesian xu/yu/zu, "
                "or unwrapped scaled xsu/ysu/zsu."
            )
        else:
            raise ValueError(
                f"LAMMPS dump lacks supported unwrapped coordinate columns in {dump_path}. "
                "Expected xu/yu/zu or xsu/ysu/zsu."
            )
        if coordinate_mode is None:
            coordinate_mode = this_mode
        elif coordinate_mode != this_mode:
            raise ValueError(f"LAMMPS coordinate column mode changed within {dump_path}")

        origin, cell = parse_lammps_dump_box(box_header, box_lines)

        parsed = []
        for row in rows:
            atom_id = int(row[col["id"]])
            values = np.asarray([float(row[col[name]]) for name in coord_names], dtype=float)
            if this_mode == "unwrapped_scaled":
                xyz = lammps_scaled_to_cartesian(values, origin, cell)
            else:
                xyz = values
            element = row[col["element"]] if "element" in col else None
            atom_type = int(row[col["type"]]) if "type" in col else None
            parsed.append((atom_id, xyz, element, atom_type))
        parsed.sort(key=lambda item: item[0])

        ids = np.asarray([item[0] for item in parsed], dtype=int)
        positions = np.asarray([item[1] for item in parsed], dtype=float)
        elements = np.asarray([item[2] for item in parsed], dtype=object)
        atom_types = np.asarray([item[3] for item in parsed], dtype=object)

        if ids_ref is None:
            ids_ref = ids
            if elements[0] is not None:
                mobile_mask = elements == args.specie
            elif args.mobile_type is not None:
                mobile_mask = atom_types == args.mobile_type
            else:
                raise ValueError("LAMMPS dump has no element column; pass --mobile-type.")
            framework_mask = ~mobile_mask
        elif not np.array_equal(ids, ids_ref):
            raise ValueError(f"Atom ids changed order/content in {dump_path}")

        timesteps.append(timestep)
        frames.append(positions)

    if len(frames) < 2:
        raise ValueError(f"Need at least two frames in {dump_path}")
    if not np.any(mobile_mask):
        raise ValueError(f"No {args.specie} atoms found in {dump_path}")

    raw_times_ps = (np.asarray(timesteps, dtype=float) - float(timesteps[0])) * timestep_ps
    idx = slice_by_time_indices(raw_times_ps, args.trajectory_start_ps, args.trajectory_end_ps)
    if idx.size < 2:
        raise ValueError(f"Trajectory segment leaves fewer than two frames: {dump_path}")

    positions = np.asarray(frames, dtype=float)[idx]
    time_ps = raw_times_ps[idx] - raw_times_ps[idx[0]]
    time_ps, msd, used_correction = compute_msd_series(
        positions=positions,
        time_ps=time_ps,
        mobile_mask=mobile_mask,
        framework_mask=framework_mask,
        drift_correction=args.drift_correction,
        msd_mode=args.msd_mode,
    )

    data_path = run_dir / args.lammps_data_name
    n_mobile, volume_A3, _ = parse_lammps_data(data_path, args.specie, args.mobile_type)
    if n_mobile is None:
        n_mobile = int(np.sum(mobile_mask))

    return RunData(
        dataset=dataset,
        run_dir=run_dir,
        temperature_K=require_temperature(run_dir, "LAMMPS trajectory", args),
        source_kind="lammps_trajectory",
        source_path=dump_path,
        time_ps=time_ps,
        msd_A2=msd,
        n_mobile=n_mobile,
        volume_A3=volume_A3,
        drift_correction=used_correction,
        notes=f"lammps_timestep_ps={timestep_ps:g}; coordinate_mode={coordinate_mode}; msd_mode={args.msd_mode}",
    )


def strip_xml_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def parse_vasp_numbers(text: str | None) -> list[float]:
    if text is None:
        return []
    return [float(part) for part in text.split()]


def parse_vasp_structure_element(structure_elem) -> tuple[np.ndarray, np.ndarray] | None:
    basis = None
    positions = None
    for varray in structure_elem.iter():
        if strip_xml_namespace(varray.tag) != "varray":
            continue
        name = varray.attrib.get("name")
        if name not in {"basis", "positions"}:
            continue
        rows = [
            parse_vasp_numbers(child.text)
            for child in list(varray)
            if strip_xml_namespace(child.tag) == "v"
        ]
        if name == "basis":
            basis = np.asarray(rows, dtype=float)
        elif name == "positions":
            positions = np.asarray(rows, dtype=float)
    if basis is None or positions is None or len(basis) != 3:
        return None
    return basis, positions


def parse_vasprun_aimd(vasprun_path: Path):
    incar = {}
    symbols = None
    basis_frames = []
    frac_frames = []
    stack = []

    for event, elem in ET.iterparse(str(vasprun_path), events=("start", "end")):
        tag = strip_xml_namespace(elem.tag)
        if event == "start":
            stack.append((tag, elem.attrib.get("name")))
            continue

        in_incar = any(parent_tag == "incar" for parent_tag, _ in stack)
        in_structure = any(parent_tag == "structure" for parent_tag, _ in stack)
        in_atominfo = any(parent_tag == "atominfo" for parent_tag, _ in stack)
        in_calculation = any(parent_tag == "calculation" for parent_tag, _ in stack)

        if in_incar and tag in {"i", "v"}:
            name = elem.attrib.get("name")
            if name and elem.text is not None:
                parts = elem.text.split()
                if len(parts) == 1:
                    raw_value = parts[0]
                    try:
                        value = float(raw_value)
                        if elem.attrib.get("type") == "int":
                            value = int(value)
                    except ValueError:
                        value = raw_value
                    incar[name] = value
                elif parts:
                    try:
                        incar[name] = [float(part) for part in parts]
                    except ValueError:
                        incar[name] = parts

        if tag == "array" and elem.attrib.get("name") == "atoms":
            parsed_symbols = []
            for rc in elem.iter():
                if strip_xml_namespace(rc.tag) != "rc":
                    continue
                cols = [
                    (child.text or "").strip()
                    for child in list(rc)
                    if strip_xml_namespace(child.tag) == "c"
                ]
                if cols:
                    parsed_symbols.append(cols[0])
            if parsed_symbols:
                symbols = parsed_symbols

        if tag == "structure":
            name = elem.attrib.get("name")
            use_structure = name == "initialpos" or (name is None and in_calculation)
            if use_structure:
                parsed = parse_vasp_structure_element(elem)
                if parsed is not None:
                    basis, frac = parsed
                    basis_frames.append(basis)
                    frac_frames.append(frac)

        if tag == "structure" or (tag == "array" and in_atominfo) or (not in_structure and not in_atominfo):
            elem.clear()
        stack.pop()

    if symbols is None:
        raise ValueError(f"Could not read atom symbols from {vasprun_path}")
    if len(frac_frames) < 2:
        raise ValueError(f"Need at least two AIMD frames in {vasprun_path}")
    if any(len(frame) != len(symbols) for frame in frac_frames):
        raise ValueError(f"Atom count mismatch while reading {vasprun_path}")

    return incar, symbols, np.asarray(basis_frames, dtype=float), np.asarray(frac_frames, dtype=float)


def infer_vasp_temperature(run_dir: Path, incar: dict, args) -> float:
    override_temp = args_temperature_override(args, "aimd_temperature_K")
    if override_temp is not None:
        return override_temp
    path_temp = temperature_from_path(run_dir)
    if path_temp is not None:
        return float(path_temp)
    tebeg = incar.get("TEBEG")
    teend = incar.get("TEEND", tebeg)
    if tebeg is not None and teend is not None:
        return 0.5 * (float(tebeg) + float(teend))
    raise ValueError(
        "Could not infer AIMD temperature. Put vasprun.xml under T*/ or pass --aimd-temperature-K."
    )


def unwrap_fractional_trajectory(frac_frames: np.ndarray, basis_frames: np.ndarray) -> np.ndarray:
    acc_frac = frac_frames[0].copy()
    prev_frac = frac_frames[0].copy()
    positions = [acc_frac @ basis_frames[0]]
    for frac, basis in zip(frac_frames[1:], basis_frames[1:]):
        delta = frac - prev_frac
        delta -= np.round(delta)
        acc_frac = acc_frac + delta
        positions.append(acc_frac @ basis)
        prev_frac = frac
    return np.asarray(positions, dtype=float)


def load_vasp_aimd(run_dir: Path, dataset: str, args) -> RunData:
    vasprun_path = run_dir / args.vasp_file_name
    if not vasprun_path.exists():
        raise FileNotFoundError(f"Cannot find AIMD vasprun: {vasprun_path}")

    incar, symbols, basis_frames, frac_frames = parse_vasprun_aimd(vasprun_path)
    temperature_K = infer_vasp_temperature(run_dir, incar, args)
    raw_potim = args.vasp_step_fs if args.vasp_step_fs is not None else incar.get("POTIM")
    if raw_potim is None:
        raise ValueError("VASP AIMD timestep POTIM is absent; pass --vasp-step-fs explicitly")
    potim_fs = float(raw_potim)
    if not math.isfinite(potim_fs) or potim_fs <= 0.0:
        raise ValueError("VASP AIMD timestep must be finite and positive")

    raw_times_ps = np.arange(len(frac_frames), dtype=float) * potim_fs / 1000.0
    idx = slice_by_time_indices(raw_times_ps, args.trajectory_start_ps, args.trajectory_end_ps)
    if idx.size < 2:
        raise ValueError(f"AIMD trajectory segment leaves fewer than two frames: {vasprun_path}")

    frac_frames = frac_frames[idx]
    basis_frames = basis_frames[idx]
    time_ps = raw_times_ps[idx] - raw_times_ps[idx[0]]

    symbols = np.asarray(symbols)
    mobile_mask = symbols == args.specie
    framework_mask = ~mobile_mask
    if not mobile_mask.any():
        raise ValueError(f"No {args.specie} atoms found in {vasprun_path}")

    volumes = np.abs(np.linalg.det(basis_frames))

    if args.trajectory_msd_engine in {"diffusion-analyzer", "auto"}:
        try:
            structures = pymatgen_structures_from_frames(symbols, basis_frames, frac_frames)
            analyzer, da_time_ps, da_msd = diffusion_analyzer_from_structures(
                structures=structures,
                specie=args.specie,
                temperature_K=temperature_K,
                time_step_fs=potim_fs,
                args=args,
            )
            return RunData(
                dataset=dataset,
                run_dir=run_dir,
                temperature_K=temperature_K,
                source_kind="vasp_diffusion_analyzer",
                source_path=vasprun_path,
                time_ps=da_time_ps,
                msd_A2=da_msd,
                n_mobile=int(mobile_mask.sum()),
                volume_A3=float(np.median(volumes)),
                drift_correction="DiffusionAnalyzer",
                notes=(
                    f"DiffusionAnalyzer.from_structures; POTIM_fs={potim_fs:g}; "
                    f"n_raw_frames={len(raw_times_ps)}; n_used_structures={len(structures)}; "
                    f"da_smoothed={args.diffusion_analyzer_smoothed}; "
                    f"da_step_skip={args.diffusion_analyzer_step_skip}; "
                    f"da_diffusivity_cm2_s={getattr(analyzer, 'diffusivity', None)}"
                ),
            )
        except Exception:
            if args.trajectory_msd_engine == "diffusion-analyzer":
                raise

    positions = unwrap_fractional_trajectory(frac_frames, basis_frames)
    time_ps, msd, used_correction = compute_msd_series(
        positions=positions,
        time_ps=time_ps,
        mobile_mask=mobile_mask,
        framework_mask=framework_mask,
        drift_correction=args.drift_correction,
        msd_mode=args.msd_mode,
    )

    return RunData(
        dataset=dataset,
        run_dir=run_dir,
        temperature_K=temperature_K,
        source_kind="vasp_numpy_msd",
        source_path=vasprun_path,
        time_ps=time_ps,
        msd_A2=msd,
        n_mobile=int(mobile_mask.sum()),
        volume_A3=float(np.median(volumes)),
        drift_correction=used_correction,
        notes=(
            f"numpy_fallback; POTIM_fs={potim_fs:g}; n_raw_frames={len(raw_times_ps)}; "
            f"n_used_frames={len(time_ps)}; msd_mode={args.msd_mode}"
        ),
    )


def parse_vasprun_static_metadata(vasprun_path: Path):
    incar = {}
    symbols = None
    initial_basis = None
    stack = []

    for event, elem in ET.iterparse(str(vasprun_path), events=("start", "end")):
        tag = strip_xml_namespace(elem.tag)
        if event == "start":
            stack.append((tag, elem.attrib.get("name")))
            continue

        in_incar = any(parent_tag == "incar" for parent_tag, _ in stack)
        in_atominfo = any(parent_tag == "atominfo" for parent_tag, _ in stack)

        if in_incar and tag in {"i", "v"}:
            name = elem.attrib.get("name")
            if name and elem.text is not None:
                parts = elem.text.split()
                if len(parts) == 1:
                    raw_value = parts[0]
                    try:
                        value = float(raw_value)
                        if elem.attrib.get("type") == "int":
                            value = int(value)
                    except ValueError:
                        value = raw_value
                    incar[name] = value

        if tag == "array" and elem.attrib.get("name") == "atoms":
            parsed_symbols = []
            for rc in elem.iter():
                if strip_xml_namespace(rc.tag) != "rc":
                    continue
                cols = [
                    (child.text or "").strip()
                    for child in list(rc)
                    if strip_xml_namespace(child.tag) == "c"
                ]
                if cols:
                    parsed_symbols.append(cols[0])
            if parsed_symbols:
                symbols = parsed_symbols

        if tag == "structure" and elem.attrib.get("name") == "initialpos":
            parsed = parse_vasp_structure_element(elem)
            if parsed is not None:
                initial_basis, _ = parsed

        if tag == "structure" or (tag == "array" and in_atominfo) or not any(
            parent_tag in {"structure", "atominfo"} for parent_tag, _ in stack
        ):
            elem.clear()
        stack.pop()

        if symbols is not None and initial_basis is not None:
            break

    return incar, symbols, initial_basis


def find_msd_file(run_dir: Path, args) -> Path | None:
    if args.msd_file_name:
        requested = Path(args.msd_file_name)
        path = requested if requested.is_absolute() else run_dir / requested
        return path if path.exists() else None

    priority = [
        "msd.dat",
        "MSD.dat",
        "msd_after_discard.csv",
        "msd.csv",
        "MSD.csv",
        "msd.txt",
        "MSD.txt",
        "msd.tsv",
        "MSD.tsv",
        "msd.xvg",
        "MSD.xvg",
    ]
    for name in priority:
        path = run_dir / name
        if path.exists():
            return path

    candidates = sorted(
        path for path in run_dir.iterdir() if path.is_file() and is_msd_filename(path.name)
    )
    return candidates[0] if candidates else None


def read_text_msd_header(path: Path) -> list[str] | None:
    header = None
    with path.open(errors="replace") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped.startswith(("#", "@")):
                candidate = stripped.lstrip("#@").strip()
                if candidate.startswith(("title", "xaxis", "yaxis", "TYPE")):
                    continue
                tokens = candidate.split()
                if len(tokens) >= 2 and any(re.search(r"[A-Za-z]", token) for token in tokens):
                    header = tokens
                continue
            break
    return header


def resolve_column_index(
    column_spec: str | None,
    names: list[str] | None,
    n_columns: int,
    default_index: int,
) -> int:
    if column_spec is None:
        return default_index
    spec = str(column_spec).strip()
    if re.fullmatch(r"-?\d+", spec):
        index = int(spec)
        return index if index >= 0 else n_columns + index
    if names:
        lower_to_index = {name.lower(): i for i, name in enumerate(names)}
        if spec.lower() in lower_to_index:
            return lower_to_index[spec.lower()]
    raise ValueError(f"Could not resolve column {column_spec!r}; available columns: {names}")


def choose_time_column(names: list[str] | None, n_columns: int, args) -> int:
    if args.msd_time_column is not None:
        return resolve_column_index(args.msd_time_column, names, n_columns, 0)
    if names:
        normalized = [name.lower().replace("(", "_").replace(")", "").replace("/", "_") for name in names]
        preferred = ["time_ps", "t_ps", "time_fs", "time_ns", "time", "t", "timestep", "step"]
        for target in preferred:
            if target in normalized:
                return normalized.index(target)
        for i, name in enumerate(normalized):
            if "time" in name or name in {"step", "timestep"}:
                return i
    return 0


def choose_msd_column(names: list[str] | None, n_columns: int, time_col: int, args) -> int:
    if args.msd_column is not None:
        return resolve_column_index(args.msd_column, names, n_columns, 1 if n_columns > 1 else 0)
    if names:
        normalized = [name.lower() for name in names]
        specie = args.specie.lower()
        candidates = []
        for i, name in enumerate(normalized):
            if i == time_col:
                continue
            if "msd" in name and specie in name and ("[4]" in name or "total" in name or name.endswith(specie)):
                candidates.append(i)
        if candidates:
            return candidates[0]
        for i, name in enumerate(normalized):
            if i == time_col:
                continue
            if "msd" in name and "fw" not in name and "framework" not in name:
                return i
    if n_columns < 2:
        raise ValueError("MSD table must contain at least two columns.")
    return 1 if time_col != 1 else 0


def infer_msd_time_unit(column_name: str | None, args) -> str:
    if args.msd_time_unit != "auto":
        return args.msd_time_unit
    name = (column_name or "").lower()
    if "fs" in name:
        return "fs"
    if "ns" in name:
        return "ns"
    if "ps" in name:
        return "ps"
    if "step" in name:
        return "step"
    raise ValueError(
        "MSD time unit is not encoded in the selected column name; "
        "pass --msd-time-unit explicitly"
    )


def infer_msd_unit(column_name: str | None, args) -> str:
    if args.msd_unit != "auto":
        return args.msd_unit
    name = (column_name or "").lower()
    if "nm" in name:
        return "nm2"
    if "a2" in name or "angstrom" in name or "å" in name:
        return "A2"
    raise ValueError(
        "MSD length unit is not encoded in the selected column name; pass --msd-unit explicitly"
    )


def read_msd_table(msd_path: Path, run_dir: Path, args):
    suffix = msd_path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(msd_path, sep=sep)
        names = [str(col) for col in df.columns]
        data = df.to_numpy(dtype=float)
    else:
        names = read_text_msd_header(msd_path)
        data = np.loadtxt(str(msd_path), comments=("#", "@"), ndmin=2)
        if names and len(names) != data.shape[1]:
            names = None

    if data.shape[0] < 2:
        raise ValueError(f"Need at least two rows in MSD file: {msd_path}")

    time_col = choose_time_column(names, data.shape[1], args)
    msd_col = choose_msd_column(names, data.shape[1], time_col, args)
    time_name = names[time_col] if names else None
    msd_name = names[msd_col] if names else None

    raw_time = data[:, time_col].astype(float)
    msd = data[:, msd_col].astype(float)

    time_unit = infer_msd_time_unit(time_name, args)
    if time_unit == "fs":
        time_ps = raw_time / 1000.0
    elif time_unit == "ns":
        time_ps = raw_time * 1000.0
    elif time_unit == "step":
        step_ps = (
            args.msd_step_ps
            if args.msd_step_ps is not None
            else infer_lammps_timestep_ps(run_dir, args.lammps_timestep_ps)
        )
        time_ps = (raw_time - raw_time[0]) * float(step_ps)
    else:
        time_ps = raw_time

    msd_unit = infer_msd_unit(msd_name, args)
    if msd_unit == "nm2":
        msd = msd * 100.0

    order = np.argsort(time_ps)
    time_ps = time_ps[order]
    msd = msd[order]
    finite = np.isfinite(time_ps) & np.isfinite(msd)
    time_ps = time_ps[finite]
    msd = msd[finite]
    if len(time_ps) < 2:
        raise ValueError(f"MSD file has fewer than two finite rows: {msd_path}")
    time_ps = time_ps - time_ps[0]

    notes = (
        f"msd_file={msd_path.name}; time_column={time_name if time_name else time_col}; "
        f"msd_column={msd_name if msd_name else msd_col}; time_unit={time_unit}; msd_unit={msd_unit}"
    )
    return time_ps, msd, notes


def infer_msd_temperature(run_dir: Path, args) -> float:
    override_temp = args_temperature_override(args, "msd_temperature_K")
    if override_temp is not None:
        return override_temp
    path_temp = temperature_from_path(run_dir)
    if path_temp is not None:
        return float(path_temp)
    if (run_dir / args.vasp_file_name).exists():
        incar, _, _ = parse_vasprun_static_metadata(run_dir / args.vasp_file_name)
        tebeg = incar.get("TEBEG")
        teend = incar.get("TEEND", tebeg)
        if tebeg is not None and teend is not None:
            return 0.5 * (float(tebeg) + float(teend))
    raise ValueError("Could not infer MSD temperature. Put file under T*/ or pass --msd-temperature-K.")


def infer_mobile_count_and_volume(run_dir: Path, args):
    n_mobile = args.n_mobile_ions
    volume_A3 = args.volume_A3

    if (n_mobile is None or volume_A3 is None) and (run_dir / args.lammps_data_name).exists():
        parsed_n, parsed_volume, _ = parse_lammps_data(run_dir / args.lammps_data_name, args.specie, args.mobile_type)
        n_mobile = n_mobile if n_mobile is not None else parsed_n
        volume_A3 = volume_A3 if volume_A3 is not None else parsed_volume

    if (n_mobile is None or volume_A3 is None) and (run_dir / "production.traj").exists():
        try:
            from ase.io import read

            atoms = read(str(run_dir / "production.traj"), index=0)
            symbols = np.asarray(atoms.get_chemical_symbols())
            n_mobile = n_mobile if n_mobile is not None else int(np.sum(symbols == args.specie))
            volume_A3 = volume_A3 if volume_A3 is not None else float(atoms.get_volume())
        except Exception:
            pass

    if (n_mobile is None or volume_A3 is None) and (run_dir / args.vasp_file_name).exists():
        try:
            _, symbols, basis = parse_vasprun_static_metadata(run_dir / args.vasp_file_name)
            if symbols is not None:
                symbols = np.asarray(symbols)
                n_mobile = n_mobile if n_mobile is not None else int(np.sum(symbols == args.specie))
            if basis is not None:
                volume_A3 = volume_A3 if volume_A3 is not None else float(abs(np.linalg.det(basis)))
        except Exception:
            pass

    return n_mobile, volume_A3


def load_msd_file(run_dir: Path, dataset: str, args) -> RunData:
    msd_path = find_msd_file(run_dir, args)
    if msd_path is None:
        raise FileNotFoundError(f"No MSD table found in {run_dir}")

    time_ps, msd_A2, notes = read_msd_table(msd_path, run_dir, args)
    temperature_K = infer_msd_temperature(run_dir, args)
    n_mobile, volume_A3 = infer_mobile_count_and_volume(run_dir, args)

    return RunData(
        dataset=dataset,
        run_dir=run_dir,
        temperature_K=temperature_K,
        source_kind="msd_file",
        source_path=msd_path,
        time_ps=time_ps,
        msd_A2=msd_A2,
        n_mobile=n_mobile,
        volume_A3=volume_A3,
        drift_correction="as_in_msd_file",
        notes=notes,
    )


def load_run_data(dataset: str, run_dir: Path, args) -> RunData:
    if args.source == "msd":
        return load_msd_file(run_dir, dataset, args)

    if args.source in {"trajectory", "vasp"}:
        if (run_dir / args.vasp_file_name).exists():
            return load_vasp_aimd(run_dir, dataset, args)
        if args.source == "vasp":
            raise FileNotFoundError(f"No {args.vasp_file_name} in {run_dir}")
        if (run_dir / "production.traj").exists():
            return load_ase_trajectory(run_dir, dataset, args)
        if (run_dir / "traj.lammpstrj").exists():
            return load_lammps_trajectory(run_dir, dataset, args)
        raise FileNotFoundError(f"No trajectory file in {run_dir}")

    if (run_dir / "production.traj").exists():
        return load_ase_trajectory(run_dir, dataset, args)
    if (run_dir / "traj.lammpstrj").exists():
        return load_lammps_trajectory(run_dir, dataset, args)
    if (run_dir / args.vasp_file_name).exists():
        return load_vasp_aimd(run_dir, dataset, args)
    msd_path = find_msd_file(run_dir, args)
    if msd_path is not None:
        return load_msd_file(run_dir, dataset, args)
    raise FileNotFoundError(f"No known post-processing input in {run_dir}")


def fit_line(x: np.ndarray, y: np.ndarray) -> LinearFit:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        raise ValueError("Need at least two finite points for linear fit")

    fit = linear_fit(x.tolist(), y.tolist())
    return LinearFit(
        slope=float(fit["slope"]),
        intercept=float(fit["intercept"]),
        r2=float(fit["r_squared"]),
        sse=float(fit["residual_sum_squares"]),
        slope_stderr=(
            None if fit["slope_stderr"] is None else float(fit["slope_stderr"])
        ),
        n_points=int(fit["point_count"]),
    )


def conductivity_NE_mS_cm(
    diffusivity_cm2_s: float,
    n_mobile: int | None,
    volume_A3: float | None,
    temperature_K: float,
    charge: float = 1.0,
):
    if n_mobile is None or volume_A3 is None or volume_A3 <= 0:
        return None
    if not math.isfinite(diffusivity_cm2_s) or diffusivity_cm2_s <= 0:
        return None
    if not math.isfinite(charge) or charge <= 0:
        return None

    return nernst_einstein_conductivity(
        diffusivity_cm2_s * 1.0e-4,
        temperature_K,
        int(n_mobile),
        charge,
        volume_A3,
    )["conductivity_ms_cm"]


def density_m3(n_mobile: int | None, volume_A3: float | None):
    if n_mobile is None or volume_A3 is None or volume_A3 <= 0:
        return None
    return n_mobile / (volume_A3 * 1e-30)


def resolve_smooth_window_points(time_ps: np.ndarray, args) -> int | None:
    """Return an odd moving-average window length, or None when smoothing is off."""
    window_points = args.msd_smooth_window_points
    if args.msd_smooth_window_ps is not None:
        diffs = np.diff(np.asarray(time_ps, dtype=float))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            raise ValueError("Cannot use --msd-smooth-window-ps because time spacing is not positive")
        dt_ps = float(np.median(diffs))
        window_points = max(1, int(round(float(args.msd_smooth_window_ps) / dt_ps)))

    if window_points is None or window_points <= 1:
        return None
    window_points = int(window_points)
    if window_points % 2 == 0:
        window_points += 1
    return window_points


def smooth_msd_series(msd_A2: np.ndarray, window_points: int | None) -> np.ndarray:
    """Centered moving-average smoothing with edge padding."""
    msd = np.asarray(msd_A2, dtype=float)
    if window_points is None or window_points <= 1:
        return msd.copy()
    if window_points > msd.size:
        window_points = msd.size if msd.size % 2 == 1 else msd.size - 1
    if window_points <= 1:
        return msd.copy()
    pad = window_points // 2
    padded = np.pad(msd, pad_width=pad, mode="edge")
    kernel = np.ones(window_points, dtype=float) / float(window_points)
    smoothed = np.convolve(padded, kernel, mode="valid")
    smoothed[0] = msd[0]
    return smoothed


def analyze_run(run: RunData, args, output_dir: Path):
    if not math.isfinite(float(run.temperature_K)) or float(run.temperature_K) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if (
        run.n_mobile is None
        or isinstance(run.n_mobile, bool)
        or int(run.n_mobile) <= 0
    ):
        raise ValueError(
            "mobile-ion count cannot be derived reliably; pass --n-mobile-ions explicitly"
        )
    if run.volume_A3 is None or not math.isfinite(float(run.volume_A3)) or run.volume_A3 <= 0:
        raise ValueError("cell volume cannot be derived reliably; pass --volume-A3 explicitly")
    time_ps = np.asarray(run.time_ps, dtype=float)
    msd_A2 = np.asarray(run.msd_A2, dtype=float)
    smooth_window_points = resolve_smooth_window_points(time_ps, args)
    smoothed_msd_A2 = smooth_msd_series(msd_A2, smooth_window_points)
    fit_msd_A2 = smoothed_msd_A2 if args.fit_smoothed_msd else msd_A2
    mask = np.isfinite(time_ps) & np.isfinite(fit_msd_A2)
    if args.fit_start_ps is not None:
        mask &= time_ps >= args.fit_start_ps
    if args.fit_end_ps is not None:
        mask &= time_ps <= args.fit_end_ps
    if int(mask.sum()) < args.min_msd_fit_points:
        raise ValueError(
            f"MSD fit window leaves {int(mask.sum())} points; "
            f"need at least {args.min_msd_fit_points}"
        )

    diffusion = linear_diffusion_from_msd(
        time_ps[mask].tolist(), fit_msd_A2[mask].tolist(), dimensions=3
    )
    fit = LinearFit(
        slope=float(diffusion["slope_angstrom2_per_ps"]),
        intercept=float(diffusion["intercept_angstrom2"]),
        r2=float(diffusion["r_squared"]),
        sse=float(diffusion["residual_sum_squares"]),
        slope_stderr=(
            None
            if diffusion["slope_stderr_angstrom2_per_ps"] is None
            else float(diffusion["slope_stderr_angstrom2_per_ps"])
        ),
        n_points=int(diffusion["samples"]),
    )
    diffusivity = float(diffusion["diffusion_cm2_per_s"])
    diffusivity_stderr = (
        fit.slope_stderr / 6.0 * 1e-4 if fit.slope_stderr is not None else None
    )
    sigma = conductivity_NE_mS_cm(diffusivity, run.n_mobile, run.volume_A3, run.temperature_K, args.charge)

    slug = safe_slug(run)
    curve_path = output_dir / "msd_curves" / f"{slug}_msd.csv"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve_data = {
        "time_ps": time_ps,
        "msd_A2": msd_A2,
        "msd_smoothed_A2": smoothed_msd_A2,
        "fit_msd_A2": fit_msd_A2,
        "used_for_fit": mask,
    }
    pd.DataFrame(curve_data).to_csv(curve_path, index=False)

    plot_path = output_dir / "msd_fits" / f"{slug}_msd_fit.html"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_single_msd(run, fit, mask, plot_path, smoothed_msd_A2=smoothed_msd_A2, fit_smoothed=args.fit_smoothed_msd)

    return {
        "dataset": run.dataset,
        "temperature_K": run.temperature_K,
        "run_dir": str(run.run_dir),
        "source_kind": run.source_kind,
        "source_path": str(run.source_path),
        "drift_correction": run.drift_correction,
        "n_msd_points": int(time_ps.size),
        "n_fit_points": fit.n_points,
        "fit_start_ps": float(np.min(time_ps[mask])),
        "fit_end_ps": float(np.max(time_ps[mask])),
        "msd_start_A2": float(fit_msd_A2[mask][0]),
        "msd_end_A2": float(fit_msd_A2[mask][-1]),
        "msd_smoothing_window_points": smooth_window_points,
        "fit_smoothed_msd": bool(args.fit_smoothed_msd),
        "msd_slope_A2_per_ps": fit.slope,
        "msd_slope_stderr_A2_per_ps": fit.slope_stderr,
        "msd_fit_intercept_A2": fit.intercept,
        "msd_fit_r2": fit.r2,
        "diffusivity_cm2_s": diffusivity if diffusivity > 0 else float("nan"),
        "diffusivity_stderr_cm2_s": diffusivity_stderr,
        "n_mobile_ions": run.n_mobile,
        "charge": args.charge,
        "volume_A3": run.volume_A3,
        "mobile_ion_density_m3": density_m3(run.n_mobile, run.volume_A3),
        "conductivity_NE_mS_cm": sigma,
        "dimensions": 3,
        "haven_ratio": 1.0,
        "msd_curve_csv": str(curve_path),
        "msd_fit_html": str(plot_path),
        "notes": run.notes,
    }


def safe_slug(run: RunData) -> str:
    digest = hashlib.sha1(str(run.run_dir).encode("utf-8")).hexdigest()[:8]
    temp = f"{run.temperature_K:g}K".replace(".", "p")
    dataset = re.sub(r"[^A-Za-z0-9_.-]+", "_", run.dataset)
    return f"{dataset}_{temp}_{digest}"


def plot_single_msd(
    run: RunData,
    fit: LinearFit,
    mask: np.ndarray,
    output_path: Path,
    smoothed_msd_A2: np.ndarray | None = None,
    fit_smoothed: bool = False,
):
    time_ps = np.asarray(run.time_ps, dtype=float)
    msd_A2 = np.asarray(run.msd_A2, dtype=float)
    x_fit = time_ps[mask]
    x_line = np.linspace(float(np.min(x_fit)), float(np.max(x_fit)), 100)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_ps,
            y=msd_A2,
            mode="markers",
            marker={"size": 5},
            name="MSD raw",
        )
    )
    if smoothed_msd_A2 is not None:
        fig.add_trace(
            go.Scatter(
                x=time_ps,
                y=smoothed_msd_A2,
                mode="lines",
                line={"width": 2},
                name="MSD smoothed" + (" used for fit" if fit_smoothed else ""),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=fit.intercept + fit.slope * x_line,
            mode="lines",
            line={"width": 2},
            name="linear fit",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=f"{run.dataset} T={run.temperature_K:g} K",
        xaxis_title="time (ps)",
        yaxis_title="MSD (A^2)",
        legend_title_text="",
    )
    fig.write_html(output_path, include_plotlyjs="cdn")


def bic_from_sse(sse: float, n: int, k: int) -> float:
    sse = max(float(sse), 1e-300)
    return n * math.log(sse / n) + k * math.log(n)


def arrhenius_fit_from_xy(x_invK: np.ndarray, y_lnD: np.ndarray, indices: np.ndarray | None = None):
    if indices is None:
        indices = np.arange(x_invK.size)
    temps = 1.0 / x_invK[indices]
    diffusivities = np.exp(y_lnD[indices])
    shared = arrhenius_from_diffusivities(temps.tolist(), diffusivities.tolist())
    return {
        "indices": indices.astype(int).tolist(),
        "n_points": int(shared["point_count"]),
        "temperature_min_K": float(np.min(temps)),
        "temperature_max_K": float(np.max(temps)),
        "slope_K": shared["slope_k"],
        "intercept_lnD0": shared["intercept_ln_diffusivity"],
        "Ea_eV": shared["activation_energy_ev"],
        "Ea_meV": float(shared["activation_energy_ev"]) * 1000.0,
        "D0_cm2_s": shared["prefactor_cm2_s"],
        "r2_lnD": shared["r_squared"],
        "sse_lnD": shared["residual_sum_squares"],
        "slope_stderr_K": shared["slope_stderr_k"],
        "Ea_stderr_eV": shared["activation_energy_stderr_ev"],
    }


def predict_arrhenius(fit_info: dict, temperature_K: float) -> float:
    lnD = fit_info["intercept_lnD0"] + fit_info["slope_K"] * (1.0 / temperature_K)
    return math.exp(lnD)


def find_piecewise_arrhenius(T: np.ndarray, D: np.ndarray, args):
    x = 1.0 / T
    y = np.log(D)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    T_sorted = T[order]
    n = x.size

    single = arrhenius_fit_from_xy(x, y)
    single["bic"] = bic_from_sse(single["sse_lnD"], n, 2)

    best = None
    min_points = args.min_segment_points
    for split in range(min_points, n - min_points + 1):
        left_idx = np.arange(0, split)
        right_idx = np.arange(split, n)
        left = arrhenius_fit_from_xy(x, y, left_idx)
        right = arrhenius_fit_from_xy(x, y, right_idx)
        left["segment_label"] = "high_temperature_segment"
        right["segment_label"] = "low_temperature_segment"
        left["x_invK_min"] = float(np.min(x[left_idx]))
        left["x_invK_max"] = float(np.max(x[left_idx]))
        right["x_invK_min"] = float(np.min(x[right_idx]))
        right["x_invK_max"] = float(np.max(x[right_idx]))
        sse = left["sse_lnD"] + right["sse_lnD"]
        bic = bic_from_sse(sse, n, 4)
        slope1 = left["slope_K"]
        slope2 = right["slope_K"]
        denom = max(0.5 * (abs(slope1) + abs(slope2)), 1e-300)
        rel_slope_change = abs(slope1 - slope2) / denom
        candidate = {
            "split_index_sorted": split,
            "temperature_sorting": "segments are sorted by increasing 1/T: high-temperature segment first, low-temperature segment second",
            "break_high_temperature_edge_K": float(T_sorted[split - 1]),
            "break_low_temperature_edge_K": float(T_sorted[split]),
            "break_between_temperature_edges_K": [
                float(T_sorted[split - 1]),
                float(T_sorted[split]),
            ],
            "break_midpoint_invK": float(0.5 * (x[split - 1] + x[split])),
            "segments": [left, right],
            "sse_lnD": sse,
            "bic": bic,
            "bic_improvement_vs_single": single["bic"] - bic,
            "relative_slope_change": rel_slope_change,
        }
        if best is None or candidate["bic"] < best["bic"]:
            best = candidate

    accepted = False
    reason = "not_enough_points_for_two_segments"
    if best is not None:
        if args.piecewise == "always":
            accepted = True
            reason = "forced"
        elif args.piecewise == "auto":
            accepted = (
                best["bic_improvement_vs_single"] >= args.piecewise_bic_delta
                and best["relative_slope_change"] >= args.piecewise_slope_change
            )
            reason = "auto_accept" if accepted else "auto_reject"
        else:
            reason = "disabled"

    return {
        "single": single,
        "piecewise_best": best,
        "piecewise_accepted": accepted,
        "piecewise_reason": reason,
        "sort_order_original_indices": order.astype(int).tolist(),
    }


def summarize_arrhenius_one(results_df: pd.DataFrame, args):
    temperatures = results_df["temperature_K"].to_numpy(dtype=float)
    diffusivities = results_df["diffusivity_cm2_s"].to_numpy(dtype=float)
    valid = np.isfinite(temperatures) & np.isfinite(diffusivities) & (diffusivities > 0)

    include_temps = parse_float_list(args.fit_temperatures)
    exclude_temps = parse_float_list(args.exclude_temperatures)
    if include_temps:
        valid &= is_close_to_any(temperatures, include_temps)
    if exclude_temps:
        valid &= ~is_close_to_any(temperatures, exclude_temps)

    T = temperatures[valid]
    D = diffusivities[valid]
    if T.size < 2:
        raise ValueError("Need at least two positive diffusivity values for Arrhenius fit")

    fit = find_piecewise_arrhenius(T, D, args)

    densities = results_df.loc[valid, "mobile_ion_density_m3"].to_numpy(dtype=float)
    densities = densities[np.isfinite(densities) & (densities > 0)]
    reference_density = float(np.median(densities)) if densities.size else None

    def sigma_at_target(D_target):
        if reference_density is None:
            return None
        equivalent_volume_a3 = 1.0e30 / reference_density
        return nernst_einstein_conductivity(
            D_target * 1.0e-4,
            args.target_temperature_K,
            1,
            args.charge,
            equivalent_volume_a3,
        )["conductivity_ms_cm"]

    single_D_target = predict_arrhenius(fit["single"], args.target_temperature_K)
    fit["single"][f"D_{args.target_temperature_K:g}K_cm2_s"] = single_D_target
    fit["single"][f"sigma_NE_{args.target_temperature_K:g}K_mS_cm"] = sigma_at_target(single_D_target)

    used_model = "single"
    if fit["piecewise_accepted"] and fit["piecewise_best"] is not None:
        used_model = "piecewise"
        breakpoint_x = fit["piecewise_best"]["break_midpoint_invK"]
        target_x = 1.0 / args.target_temperature_K
        target_segment_index = 0 if target_x <= breakpoint_x else 1
        segment = fit["piecewise_best"]["segments"][target_segment_index]
        piece_D_target = predict_arrhenius(segment, args.target_temperature_K)
        fit["piecewise_best"][f"D_{args.target_temperature_K:g}K_cm2_s"] = piece_D_target
        fit["piecewise_best"][f"sigma_NE_{args.target_temperature_K:g}K_mS_cm"] = sigma_at_target(piece_D_target)
        fit["piecewise_best"]["target_temperature_invK"] = target_x
        fit["piecewise_best"]["target_temperature_segment_index"] = target_segment_index
        fit["piecewise_best"]["target_temperature_segment_label"] = segment.get("segment_label")

    return {
        "fit_scope": "all",
        "specie": args.specie,
        "arrhenius_fit_space": "ln(D_cm2_s) vs 1/T",
        "target_temperature_K": args.target_temperature_K,
        "fit_temperatures_K": T.tolist(),
        "fit_diffusivities_cm2_s": D.tolist(),
        "fit_row_indices": np.flatnonzero(valid).astype(int).tolist(),
        "reference_mobile_ion_density_m3": reference_density,
        "used_model": used_model,
        **fit,
    }


def summarize_arrhenius(results_df: pd.DataFrame, args):
    if args.fit_scope == "all" or results_df["dataset"].nunique() <= 1:
        return summarize_arrhenius_one(results_df, args)

    summaries = {}
    failures = []
    for dataset, sub_df in results_df.groupby("dataset", sort=True):
        try:
            summaries[dataset] = summarize_arrhenius_one(sub_df.reset_index(drop=True), args)
            summaries[dataset]["fit_scope"] = "dataset"
        except Exception as exc:
            failures.append({"dataset": dataset, "error": str(exc)})

    if not summaries:
        raise ValueError("No dataset has enough valid points for Arrhenius fit")

    return {
        "fit_scope": "dataset",
        "specie": args.specie,
        "arrhenius_fit_space": "ln(D_cm2_s) vs 1/T",
        "target_temperature_K": args.target_temperature_K,
        "datasets": summaries,
        "dataset_fit_failures": failures,
    }


def plot_all_msd(runs: list[RunData], output_path: Path):
    fig = go.Figure()
    for run in sorted(runs, key=lambda item: (item.temperature_K, str(item.run_dir))):
        label = f"{run.dataset} {run.temperature_K:g} K"
        fig.add_trace(
            go.Scatter(
                x=run.time_ps,
                y=run.msd_A2,
                mode="lines",
                name=label,
                hovertemplate="time=%{x:.4g} ps<br>MSD=%{y:.4g} A^2<extra>" + label + "</extra>",
            )
        )
    fig.update_layout(
        template="plotly_white",
        title="MSD by temperature",
        xaxis_title="time (ps)",
        yaxis_title="MSD (A^2)",
        legend_title_text="",
    )
    fig.write_html(output_path, include_plotlyjs="cdn")


def plot_arrhenius(results_df: pd.DataFrame, summary: dict, output_path: Path):
    valid = np.isfinite(results_df["diffusivity_cm2_s"]) & (results_df["diffusivity_cm2_s"] > 0)
    df = results_df.loc[valid].copy()
    fig = go.Figure()
    for dataset, sub in df.groupby("dataset"):
        fig.add_trace(
            go.Scatter(
                x=1000.0 / sub["temperature_K"],
                y=np.log(sub["diffusivity_cm2_s"]),
                mode="markers",
                marker={"size": 8},
                name=dataset,
                customdata=np.stack([sub["temperature_K"], sub["diffusivity_cm2_s"]], axis=1),
                hovertemplate=(
                    "1000/T=%{x:.4g} K^-1<br>"
                    "lnD=%{y:.4g}<br>"
                    "T=%{customdata[0]:.4g} K<br>"
                    "D=%{customdata[1]:.4e} cm^2/s<extra>" + dataset + "</extra>"
                ),
            )
        )

    def add_fit_lines(fit_summary: dict, prefix: str = ""):
        T_all = np.asarray(fit_summary["fit_temperatures_K"], dtype=float)
        if T_all.size:
            T_line = np.linspace(float(np.min(T_all)), float(np.max(T_all)), 300)
            single = fit_summary["single"]
            y_line = single["intercept_lnD0"] + single["slope_K"] / T_line
            fig.add_trace(
                go.Scatter(
                    x=1000.0 / T_line,
                    y=y_line,
                    mode="lines",
                    line={"dash": "dash", "width": 2},
                    name=f"{prefix}single Ea={single['Ea_eV']:.3f} eV",
                )
            )

        if fit_summary.get("piecewise_accepted") and fit_summary.get("piecewise_best"):
            for i, segment in enumerate(fit_summary["piecewise_best"]["segments"], start=1):
                t_min = segment["temperature_min_K"]
                t_max = segment["temperature_max_K"]
                T_line = np.linspace(t_min, t_max, 120)
                y_line = segment["intercept_lnD0"] + segment["slope_K"] / T_line
                segment_label = segment.get("segment_label", f"segment {i}")
                fig.add_trace(
                    go.Scatter(
                        x=1000.0 / T_line,
                        y=y_line,
                        mode="lines",
                        line={"width": 3},
                        name=f"{prefix}{segment_label} Ea={segment['Ea_eV']:.3f} eV",
                    )
                )

    if "datasets" in summary:
        for dataset, fit_summary in summary["datasets"].items():
            add_fit_lines(fit_summary, prefix=f"{dataset} ")
    elif "single" in summary:
        add_fit_lines(summary)
    else:
        pass

    fig.update_layout(
        template="plotly_white",
        title="Arrhenius fit",
        xaxis_title="1000 / T (K^-1)",
        yaxis_title="ln D (cm^2/s)",
        legend_title_text="",
    )
    fig.write_html(output_path, include_plotlyjs="cdn")


def default_inputs() -> list[Path]:
    candidates = [
        Path("ase_li_conductivity_runs_1ns_discard20ps"),
        Path("mace_conductivity_md"),
    ]
    existing = [path for path in candidates if path.exists()]
    return existing if existing else [Path(".")]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_record(path: Path, role: str) -> dict:
    resolved = path.resolve()
    return {
        "role": role,
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def analysis_source_paths(run: RunData, args) -> list[Path]:
    candidates = [run.source_path]
    for name in (
        "metadata.json",
        "production_log.csv",
        args.lammps_data_name,
        "in.mace_cond",
        "log.lammps",
    ):
        path = run.run_dir / name
        if path.is_file():
            candidates.append(path)
    return list(dict.fromkeys(path.resolve() for path in candidates))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Post-process Li MSD, ionic conductivity, and ln(D) Arrhenius Ea."
    )
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        help="Input file/folder. Can be passed multiple times. Default scans known local folders.",
    )
    parser.add_argument("--output", type=Path, default=Path("li_conductivity_postprocess"))
    parser.add_argument("--specie", default="Li")
    parser.add_argument(
        "--charge",
        type=float,
        default=1.0,
        help="Mobile ion charge number z used in Nernst-Einstein conductivity, e.g. Li+=1, Mg2+=2.",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "trajectory", "vasp", "msd"],
        default="auto",
        help="Read trajectories or precomputed MSD tables; auto chooses trajectory first, then MSD.",
    )
    parser.add_argument(
        "--drift-correction",
        choices=["framework", "mobile", "none"],
        default="framework",
        help="Used only when MSD is recomputed from trajectories.",
    )
    parser.add_argument(
        "--msd-mode",
        choices=["multi-origin", "single-origin"],
        default="multi-origin",
        help="Use multiple time origins by default to reduce MSD noise.",
    )
    parser.add_argument(
        "--trajectory-msd-engine",
        choices=["diffusion-analyzer", "numpy", "auto"],
        default="diffusion-analyzer",
        help=(
            "MSD engine for trajectory inputs. diffusion-analyzer uses "
            "pymatgen.analysis.diffusion.analyzer.DiffusionAnalyzer; numpy uses the older in-script MSD; "
            "auto tries DiffusionAnalyzer and falls back to numpy."
        ),
    )
    parser.add_argument(
        "--diffusion-analyzer-smoothed",
        choices=["max", "constant", "none"],
        default="max",
        help="DiffusionAnalyzer built-in smoothing mode. Use none for unsmoothed DiffusionAnalyzer MSD.",
    )
    parser.add_argument(
        "--diffusion-analyzer-min-obs",
        type=int,
        default=30,
        help="min_obs passed to DiffusionAnalyzer.from_structures.",
    )
    parser.add_argument(
        "--diffusion-analyzer-avg-nsteps",
        type=int,
        default=1000,
        help="avg_nsteps passed to DiffusionAnalyzer.from_structures when smoothing is enabled.",
    )
    parser.add_argument(
        "--diffusion-analyzer-step-skip",
        type=int,
        default=1,
        help="Use every Nth structure in DiffusionAnalyzer. The effective time spacing is time_step * step_skip.",
    )
    parser.add_argument("--trajectory-start-ps", type=float, default=None)
    parser.add_argument("--trajectory-end-ps", type=float, default=None)
    parser.add_argument("--fit-start-ps", type=float, default=None)
    parser.add_argument("--fit-end-ps", type=float, default=None)
    parser.add_argument(
        "--msd-smooth-window-points",
        type=int,
        default=None,
        help="Centered moving-average smoothing window in number of MSD points. Even values are rounded up to odd.",
    )
    parser.add_argument(
        "--msd-smooth-window-ps",
        type=float,
        default=None,
        help="Centered moving-average smoothing window in ps. Converted to points using the median time spacing.",
    )
    parser.add_argument(
        "--fit-smoothed-msd",
        action="store_true",
        help="Fit the smoothed MSD curve instead of the raw MSD. By default smoothing is only written/plotted.",
    )
    parser.add_argument("--min-msd-fit-points", type=int, default=3)
    parser.add_argument("--ase-frame-step-fs", type=float, default=None)
    parser.add_argument("--lammps-timestep-ps", type=float, default=None)
    parser.add_argument("--lammps-data-name", default="data.LYC")
    parser.add_argument("--vasp-file-name", default="vasprun.xml")
    parser.add_argument("--vasp-step-fs", type=float, default=None, help="Override VASP POTIM in fs.")
    parser.add_argument(
        "--temperature-K",
        type=float,
        default=None,
        help="Fallback temperature for inputs that do not contain temperature metadata. Best for one run or one-temperature analysis.",
    )
    parser.add_argument(
        "--aimd-temperature-K",
        type=float,
        default=None,
        help="Override AIMD temperature if it cannot be inferred from T*/ or TEBEG/TEEND.",
    )
    parser.add_argument("--mobile-type", type=int, default=None)
    parser.add_argument("--msd-file-name", default=None, help="Specific MSD file name/path to read.")
    parser.add_argument("--msd-time-column", default=None, help="MSD table time column name or 0-based index.")
    parser.add_argument("--msd-column", default=None, help="MSD table MSD column name or 0-based index.")
    parser.add_argument(
        "--msd-time-unit",
        choices=["auto", "ps", "fs", "ns", "step"],
        default="auto",
        help="Unit of the MSD table time column. step uses --msd-step-ps or LAMMPS timestep inference.",
    )
    parser.add_argument("--msd-step-ps", type=float, default=None, help="ps per step for MSD tables.")
    parser.add_argument(
        "--msd-unit",
        choices=["auto", "A2", "nm2"],
        default="auto",
        help="Unit of the MSD column.",
    )
    parser.add_argument("--msd-temperature-K", type=float, default=None)
    parser.add_argument("--n-mobile-ions", type=int, default=None)
    parser.add_argument("--volume-A3", type=float, default=None)
    parser.add_argument("--fit-temperatures", default=None, help="Comma/space separated T values to include.")
    parser.add_argument("--exclude-temperatures", default=None, help="Comma/space separated T values to exclude.")
    parser.add_argument(
        "--fit-scope",
        choices=["dataset", "all"],
        default="dataset",
        help="Fit Arrhenius lines per dataset by default; use all to combine inputs.",
    )
    parser.add_argument("--target-temperature-K", type=float, default=300.0)
    parser.add_argument("--piecewise", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--min-segment-points", type=int, default=3)
    parser.add_argument("--piecewise-slope-change", type=float, default=0.35)
    parser.add_argument("--piecewise-bic-delta", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_paths = args.input if args.input else default_inputs()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    run_dirs = discover_run_dirs(input_paths)
    if not run_dirs:
        print("No run directories found.", file=sys.stderr)
        return 1

    runs = []
    rows = []
    failures = []
    for dataset, run_dir in run_dirs:
        try:
            run = load_run_data(dataset, run_dir, args)
            row = analyze_run(run, args, output_dir)
            runs.append(run)
            rows.append(row)
            print(
                f"T={run.temperature_K:g} K [{run.source_kind}] "
                f"D={row['diffusivity_cm2_s']:.4e} cm^2/s "
                f"sigma_NE={row['conductivity_NE_mS_cm'] if row['conductivity_NE_mS_cm'] is not None else 'nan'} mS/cm"
            )
        except Exception as exc:
            failures.append({"dataset": dataset, "run_dir": str(run_dir), "error": str(exc)})
            print(f"SKIP {run_dir}: {exc}", file=sys.stderr)

    if not rows:
        (output_dir / "postprocess_failures.json").write_text(json.dumps(json_ready(failures), indent=2))
        print("No runs could be processed.", file=sys.stderr)
        return 1

    results_df = pd.DataFrame(rows).sort_values(["temperature_K", "dataset", "run_dir"]).reset_index(drop=True)
    results_path = output_dir / "diffusion_results_by_temperature.csv"
    results_df.to_csv(results_path, index=False)

    try:
        summary = summarize_arrhenius(results_df, args)
    except Exception as exc:
        summary = {
            "fit_scope": args.fit_scope,
            "specie": args.specie,
            "arrhenius_fit_space": "ln(D_cm2_s) vs 1/T",
            "target_temperature_K": args.target_temperature_K,
            "arrhenius_fit_skipped": True,
            "skip_reason": str(exc),
            "available_temperatures_K": results_df["temperature_K"].tolist(),
            "available_diffusivities_cm2_s": results_df["diffusivity_cm2_s"].tolist(),
        }
    summary_path = output_dir / "arrhenius_summary.json"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2))

    plot_all_msd(runs, output_dir / "msd_by_temperature.html")
    plot_arrhenius(results_df, summary, output_dir / "arrhenius_fit.html")
    (output_dir / "postprocess_failures.json").write_text(json.dumps(json_ready(failures), indent=2))

    source_paths = list(
        dict.fromkeys(path for run in runs for path in analysis_source_paths(run, args))
    )
    result_paths = [
        results_path,
        summary_path,
        output_dir / "postprocess_failures.json",
        output_dir / "msd_by_temperature.html",
        output_dir / "arrhenius_fit.html",
        *[Path(row["msd_curve_csv"]) for row in rows],
        *[Path(row["msd_fit_html"]) for row in rows],
    ]
    manifest = {
        "schema_version": 1,
        "plugin_id": "ionic-transport",
        "operation": "analyze-existing",
        "scientific_contract": {
            "dimensions": 3,
            "einstein_relation": "D=slope/(2*d)",
            "conductivity_model": "uncorrected-nernst-einstein",
            "haven_ratio": 1.0,
            "arrhenius_relation": "ln(D_cm2_s) vs 1/T_K",
            "carrier_count_policy": "derived-from-structure-or-explicit; never historical-N7",
        },
        "parameters": json_ready(vars(args)),
        "implementation_artifacts": [
            artifact_record(Path(__file__), "packaged-analysis-runner"),
            artifact_record(Path(_transport_core.__file__), "mlipflow-science-transport"),
        ],
        "source_artifacts": [
            artifact_record(path, f"analysis-source:{index}")
            for index, path in enumerate(source_paths)
        ],
        "result_artifacts": [
            artifact_record(path, f"analysis-result:{index}")
            for index, path in enumerate(dict.fromkeys(path.resolve() for path in result_paths))
        ],
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(json_ready(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    target_key = f"D_{args.target_temperature_K:g}K_cm2_s"
    print()
    print(f"Wrote: {results_path}")
    print(f"Wrote: {summary_path}")

    def print_fit_report(label: str, fit_summary: dict):
        if "single" not in fit_summary:
            print(f"[{label}] Arrhenius fit skipped: {fit_summary.get('skip_reason', 'not enough data')}")
            return
        single_fit = fit_summary["single"]
        print(f"[{label}] single-line lnD fit: Ea={single_fit['Ea_eV']:.4f} eV, R2={single_fit['r2_lnD']:.4f}")
        if fit_summary.get("piecewise_best") is not None:
            best_fit = fit_summary["piecewise_best"]
            print(
                f"[{label}] piecewise check: "
                f"accepted={fit_summary['piecewise_accepted']} "
                f"reason={fit_summary['piecewise_reason']} "
                f"relative_slope_change={best_fit['relative_slope_change']:.3g} "
                f"BIC_improvement={best_fit['bic_improvement_vs_single']:.3g}"
            )
        model = fit_summary["used_model"]
        if model == "piecewise" and fit_summary.get("piecewise_best"):
            model_info = fit_summary["piecewise_best"]
            print(
                f"[{label}] target T segment: "
                f"{model_info.get('target_temperature_segment_label')} "
                f"(index {model_info.get('target_temperature_segment_index')})"
            )
        else:
            model_info = single_fit
        print(f"[{label}] model used for target extrapolation: {model}")
        print(f"[{label}] D_{args.target_temperature_K:g}K={model_info.get(target_key):.4e} cm^2/s")
        sigma_key = f"sigma_NE_{args.target_temperature_K:g}K_mS_cm"
        sigma_value = model_info.get(sigma_key)
        if sigma_value is not None:
            print(f"[{label}] sigma_NE_{args.target_temperature_K:g}K={sigma_value:.4g} mS/cm")

    if summary.get("arrhenius_fit_skipped"):
        print(f"Arrhenius fit skipped: {summary.get('skip_reason')}")
    elif "datasets" in summary:
        for dataset, fit_summary in summary["datasets"].items():
            print_fit_report(dataset, fit_summary)
        if summary.get("dataset_fit_failures"):
            print(f"Dataset fit failures: {len(summary['dataset_fit_failures'])}")
    else:
        print_fit_report("all", summary)
    if failures:
        print(f"Processed with {len(failures)} skipped run(s); see postprocess_failures.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
