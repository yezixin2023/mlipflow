#!/usr/bin/env python3
"""Post-process Li diffusion, ionic conductivity, and Arrhenius Ea.
It does not run MD. It can read either:

* MLIPipe ASE-MD artifacts containing trajectory.traj plus md-result/index
* MLIPipe LAMMPS-MD artifacts containing trajectory.lammpstrj plus result/manifest
* Historical ASE production.traj and LAMMPS traj.lammpstrj folders
* VASP AIMD folders containing vasprun.xml
* Precomputed MSD tables such as msd.dat, msd.csv, and msd_after_discard.csv

The Arrhenius fit is always performed as a linear fit in ln(D) vs 1/T.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go


FORMAL_DEPENDENCY_ERROR = (
    "Formal ionic transport requires pymatgen-analysis-diffusion. "
    "Install with: pip install 'mlipipe[transport]'"
)


KNOWN_RUN_FILES = {
    "trajectory.traj",
    "production.traj",
    "trajectory.lammpstrj",
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
    analyzer: object | None = None
    structure: object | None = None
    structure_path: Path | None = None
    time_step_fs: float | None = None
    step_skip: int = 1
    uses_ase: bool = False
    source_paths: tuple[Path, ...] = ()
    handoff: dict[str, Any] | None = None
    structures: tuple[object, ...] = ()


def require_formal_diffusion_api():
    """Import formal scientific APIs only when formal transport is selected."""

    try:
        from pymatgen.analysis.diffusion.analyzer import (
            DiffusionAnalyzer,
            fit_arrhenius,
            get_conversion_factor,
            get_diffusivity_from_msd,
            get_extrapolated_diffusivity,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(FORMAL_DEPENDENCY_ERROR) from exc
    return {
        "DiffusionAnalyzer": DiffusionAnalyzer,
        "fit_arrhenius": fit_arrhenius,
        "get_conversion_factor": get_conversion_factor,
        "get_diffusivity_from_msd": get_diffusivity_from_msd,
        "get_extrapolated_diffusivity": get_extrapolated_diffusivity,
    }


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_provenance(*, uses_ase: bool) -> dict:
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "mlipipe_version": installed_version("mlipipe"),
        "pymatgen_version": installed_version("pymatgen"),
        "pymatgen_analysis_diffusion_version": installed_version(
            "pymatgen-analysis-diffusion"
        ),
        "numpy_version": installed_version("numpy"),
        "ase_version": installed_version("ase") if uses_ase else None,
    }


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


def _native_node_root(path: Path) -> Path:
    """Collapse one MLIPipe attempt path to its logical workflow-node root."""

    directory = path.parent if path.is_file() else path
    if re.fullmatch(r"attempt-[0-9]+", directory.name):
        return directory.parent
    return directory


def _native_segment_dirs(run_dir: Path, trajectory_name: str) -> list[Path]:
    if (run_dir / trajectory_name).is_file():
        return [run_dir]
    attempts = [
        path
        for path in run_dir.glob("attempt-*")
        if path.is_dir() and (path / trajectory_name).is_file()
    ]

    def attempt_number(path: Path) -> tuple[int, str]:
        match = re.fullmatch(r"attempt-([0-9]+)", path.name)
        return (int(match.group(1)) if match else 0, path.name)

    return sorted(attempts, key=attempt_number)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _nested_mapping(value: Any, *keys: str) -> dict[str, Any]:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


def _approved_parameters(segment_dir: Path, key: str) -> dict[str, Any]:
    approved = _read_json_object(segment_dir / "approved-plan.json")
    return _nested_mapping(approved, "adapter_plan", key)


def _model_record(value: Any, fallback: dict[str, Any]) -> dict[str, Any] | None:
    model = value if isinstance(value, dict) else {}
    record = {
        "id": model.get("id", model.get("model_id", fallback.get("model_id"))),
        "path": model.get("path", fallback.get("model_path")),
    }
    return record if any(item is not None for item in record.values()) else None


def discover_run_dirs(
    input_paths: list[Path],
) -> list[tuple[str, Path, tuple[Path, ...] | None]]:
    run_dirs: dict[Path, str] = {}
    native_segments: dict[Path, set[Path]] = {}
    for input_path in input_paths:
        root = input_path.expanduser().resolve()
        if root.is_file():
            dataset = (
                root.parent.name
                if root.name in KNOWN_RUN_FILES or is_msd_filename(root.name)
                else root.name
            )
            logical = (
                _native_node_root(root)
                if root.name in {"trajectory.traj", "trajectory.lammpstrj"}
                else root.parent.resolve()
            )
            run_dirs[logical] = logical.name if logical != root.parent else dataset
            if logical != root.parent:
                native_segments.setdefault(logical, set()).add(root.parent.resolve())
            continue
        dataset = root.name if root.name else root.parent.name
        if root.name.startswith("attempt-") and root.parent.parent.name == "runs":
            dataset = root.parent.name
        if not root.exists():
            raise FileNotFoundError(f"Input path does not exist: {root}")
        for dirpath, _, filenames in os.walk(root):
            names = set(filenames)
            if names & KNOWN_RUN_FILES or any(is_msd_filename(name) for name in names):
                directory = Path(dirpath).resolve()
                native = names & {"trajectory.traj", "trajectory.lammpstrj"}
                logical = _native_node_root(directory) if native else directory
                run_dirs[logical] = logical.name if native else dataset
                if native:
                    native_segments.setdefault(logical, set()).add(directory)
    return sorted(
        (
            (
                dataset,
                path,
                tuple(sorted(native_segments[path])) if path in native_segments else None,
            )
            for path, dataset in run_dirs.items()
        ),
        key=lambda item: str(item[1]),
    )


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


def diffusion_analyzer_from_structures(
    structures,
    specie: str,
    temperature_K: float,
    time_step_fs: float,
    args,
    *,
    continuous_frac: np.ndarray | None = None,
):
    """Run pymatgen DiffusionAnalyzer and return analyzer plus its MSD/time arrays."""
    DiffusionAnalyzer = require_formal_diffusion_api()["DiffusionAnalyzer"]

    step_skip = int(args.diffusion_analyzer_step_skip)
    if step_skip < 1:
        raise ValueError("--diffusion-analyzer-step-skip must be >= 1")
    structures = list(structures)[::step_skip]
    if continuous_frac is not None:
        continuous_frac = np.asarray(continuous_frac, dtype=float)[::step_skip]
    if len(structures) < 2:
        raise ValueError("DiffusionAnalyzer needs at least two structures after step skipping")
    if not any(site.specie.symbol != specie for site in structures[0]):
        raise ValueError(
            "pymatgen DiffusionAnalyzer trajectory analysis requires at least one "
            f"non-{specie} framework atom"
        )

    smoothed = diffusion_analyzer_smoothed_arg(args)
    analyzer_kwargs = {
        "specie": specie,
        "temperature": float(temperature_K),
        "time_step": float(time_step_fs),
        "step_skip": step_skip,
        "smoothed": smoothed,
        "min_obs": int(args.diffusion_analyzer_min_obs),
        "avg_nsteps": int(args.diffusion_analyzer_avg_nsteps),
    }
    if continuous_frac is None:
        analyzer = DiffusionAnalyzer.from_structures(structures, **analyzer_kwargs)
    else:
        lattices = np.asarray(
            [np.asarray(structure.lattice.matrix, dtype=float) for structure in structures],
            dtype=float,
        )
        fractional_displacements = continuous_frac - continuous_frac[0]
        cartesian_displacements = np.asarray(
            [
                [fractional_displacements[frame, atom] @ lattices[frame] for frame in range(len(structures))]
                for atom in range(fractional_displacements.shape[1])
            ],
            dtype=float,
        )
        analyzer_lattices = lattices[:1] if np.array_equal(lattices[0], lattices[-1]) else lattices
        analyzer = DiffusionAnalyzer(
            structures[0],
            cartesian_displacements,
            lattices=analyzer_lattices,
            structures=structures,
            **analyzer_kwargs,
        )
    msd = np.asarray(analyzer.msd, dtype=float)
    if msd.ndim != 1 or msd.size < 2:
        raise ValueError("DiffusionAnalyzer returned an invalid or too-short MSD array")

    time_ps = np.asarray(analyzer.dt, dtype=float) / 1000.0
    if time_ps.shape != msd.shape or not np.all(np.isfinite(time_ps)):
        raise ValueError("DiffusionAnalyzer returned an invalid dt array")
    return analyzer, time_ps, msd, structures


def pymatgen_structures_from_frames(symbols, basis_frames: np.ndarray, frac_frames: np.ndarray):
    try:
        from pymatgen.core import Lattice, Structure
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(FORMAL_DEPENDENCY_ERROR) from exc

    return [
        Structure(
            lattice=Lattice(np.asarray(basis, dtype=float)),
            species=list(symbols),
            coords=np.asarray(frac, dtype=float),
            coords_are_cartesian=False,
            to_unit_cell=False,
        )
        for basis, frac in zip(basis_frames, frac_frames)
    ]


def unwrap_fractional_continuity(wrapped_frames: np.ndarray) -> np.ndarray:
    """Restore a continuous fractional trajectory using minimum-image steps."""

    wrapped = np.asarray(wrapped_frames, dtype=float)
    if wrapped.ndim != 3 or wrapped.shape[0] == 0 or wrapped.shape[2] != 3:
        raise ValueError("fractional trajectory must have shape (frames, atoms, 3)")
    continuous = np.empty_like(wrapped)
    continuous[0] = wrapped[0]
    for index in range(1, len(wrapped)):
        delta = wrapped[index] - wrapped[index - 1]
        continuous[index] = continuous[index - 1] + delta - np.round(delta)
    return continuous


def _ase_segment_metadata(segment_dir: Path) -> dict[str, Any]:
    result = _read_json_object(segment_dir / "md-result.json")
    index = _read_json_object(segment_dir / "trajectory-index.json")
    parameters = _approved_parameters(segment_dir, "md_parameters")
    return {
        "temperature_K": result.get("temperature_K", parameters.get("temperature_k")),
        "timestep_fs": result.get("timestep_fs", parameters.get("timestep_fs")),
        "trajectory_interval": result.get(
            "trajectory_interval", parameters.get("trajectory_interval")
        ),
        "ensemble": result.get("ensemble", parameters.get("ensemble")),
        "model": _model_record(result.get("model"), parameters),
        "structure_path": result.get("structure_path", parameters.get("structure_path")),
        "steps": index.get("steps"),
        "time_fs": index.get("time_fs"),
        "index_path": segment_dir / "trajectory-index.json",
        "result_path": segment_dir / "md-result.json",
    }


def _same_native_settings(records: list[dict[str, Any]], keys: tuple[str, ...], label: str) -> None:
    for key in keys:
        values = [record.get(key) for record in records if record.get(key) is not None]
        if values and any(value != values[0] for value in values[1:]):
            raise ValueError(f"{label} restart segments disagree on {key}")


def _first_metadata(records: list[dict[str, Any]], key: str) -> Any:
    return next((record[key] for record in records if record.get(key) is not None), None)


def load_ase_trajectory(
    run_dir: Path,
    dataset: str,
    args,
    native_segment_dirs: tuple[Path, ...] | None = None,
) -> RunData:
    require_formal_diffusion_api()
    try:
        from ase.io import read
    except ImportError as exc:
        raise ImportError(
            "ASE is required to read trajectory.traj/production.traj; install ASE"
        ) from exc

    native_segments = list(native_segment_dirs or _native_segment_dirs(run_dir, "trajectory.traj"))
    is_native = bool(native_segments)
    segment_dirs = native_segments or [run_dir]
    trajectory_name = "trajectory.traj" if is_native else "production.traj"
    metadata_records = [_ase_segment_metadata(path) for path in segment_dirs] if is_native else []
    if is_native:
        _same_native_settings(
            metadata_records,
            ("temperature_K", "timestep_fs", "trajectory_interval", "ensemble", "model", "structure_path"),
            "ASE-MD",
        )

    merged: dict[int, tuple[Any, float]] = {}
    source_paths: list[Path] = []
    for segment_index, segment_dir in enumerate(segment_dirs):
        traj_path = segment_dir / trajectory_name
        frames = read(str(traj_path), index=":")
        if not isinstance(frames, list):
            frames = [frames]
        source_paths.append(traj_path)
        if is_native:
            metadata = metadata_records[segment_index]
            steps = metadata.get("steps")
            times_fs = metadata.get("time_fs")
            if not isinstance(steps, list) or not isinstance(times_fs, list):
                raise ValueError(f"Missing trajectory-index step/time data beside {traj_path}")
            if len(steps) != len(frames) or len(times_fs) != len(frames):
                raise ValueError(f"ASE trajectory/index frame count mismatch: {traj_path}")
            source_paths.append(Path(metadata["index_path"]))
            if Path(metadata["result_path"]).is_file():
                source_paths.append(Path(metadata["result_path"]))
        else:
            dt_fs = infer_ase_dt_fs(run_dir, args.ase_frame_step_fs)
            steps = list(range(len(frames)))
            times_fs = [index * dt_fs for index in range(len(frames))]

        for step, time_fs, atoms in zip(steps, times_fs, frames):
            step = int(step)
            if step in merged:
                previous = merged[step][0]
                if (
                    previous.get_chemical_symbols() != atoms.get_chemical_symbols()
                    or not np.allclose(previous.get_cell().array, atoms.get_cell().array)
                    or not np.allclose(previous.get_positions(), atoms.get_positions())
                ):
                    raise ValueError(f"ASE restart boundary step {step} contains different frames")
                continue
            merged[step] = (atoms, float(time_fs))

    ordered = sorted(merged.items())
    frames = [item[1][0] for item in ordered]
    global_steps = np.asarray([item[0] for item in ordered], dtype=int)
    physical_times_fs = np.asarray([item[1][1] for item in ordered], dtype=float)
    traj_path = source_paths[0]
    if is_native:
        integration_timestep = _first_metadata(metadata_records, "timestep_fs")
        trajectory_interval = _first_metadata(metadata_records, "trajectory_interval")
        if integration_timestep is None or trajectory_interval is None:
            raise ValueError("ASE-MD artifact lacks timestep/trajectory interval metadata")
        expected_times = global_steps.astype(float) * float(integration_timestep)
        if not np.allclose(physical_times_fs, expected_times, rtol=0.0, atol=1e-9):
            raise ValueError("ASE trajectory-index physical times disagree with timestep_fs")
        interval = int(trajectory_interval)
        cadence = np.flatnonzero(global_steps % interval == 0)
        if cadence.size >= 2 and cadence.size != len(global_steps):
            frames = [frames[index] for index in cadence]
            global_steps = global_steps[cadence]
            physical_times_fs = physical_times_fs[cadence]
    if len(frames) < 2:
        raise ValueError(f"Need at least two frames in {traj_path}")
    if any(not bool(np.all(atoms.get_pbc())) for atoms in frames):
        raise ValueError("ASE formal transport requires a periodic cell in every frame")
    if any(float(atoms.get_volume()) <= 0.0 for atoms in frames):
        raise ValueError("ASE formal transport requires a finite non-zero cell in every frame")
    time_deltas = np.diff(physical_times_fs)
    if np.any(time_deltas <= 0) or not np.allclose(time_deltas, time_deltas[0]):
        raise ValueError("ASE restart segments must form a strictly ordered uniform frame schedule")
    dt_fs = float(time_deltas[0])
    raw_times_ps = (physical_times_fs - physical_times_fs[0]) / 1000.0
    idx = slice_by_time_indices(raw_times_ps, args.trajectory_start_ps, args.trajectory_end_ps)
    if idx.size < 2:
        raise ValueError(f"Trajectory segment leaves fewer than two frames: {traj_path}")
    frames = [frames[index] for index in idx]
    global_steps = global_steps[idx]
    physical_times_fs = physical_times_fs[idx]

    symbols = np.asarray(frames[0].get_chemical_symbols())
    if any(atoms.get_chemical_symbols() != list(symbols) for atoms in frames[1:]):
        raise ValueError("ASE atom ordering/species changed across trajectory segments")
    mobile_mask = symbols == args.specie
    if not mobile_mask.any():
        raise ValueError(f"No {args.specie} atoms found in {traj_path}")

    native_temperature = _first_metadata(metadata_records, "temperature_K")
    temperature_K = (
        float(native_temperature)
        if is_native and native_temperature is not None
        else require_temperature(run_dir, "ASE trajectory", args)
    )
    basis_frames = np.asarray(
        [np.asarray(atoms.get_cell().array, dtype=float) for atoms in frames], dtype=float
    )
    raw_frac = np.asarray(
        [np.asarray(atoms.get_scaled_positions(wrap=not is_native), dtype=float) for atoms in frames],
        dtype=float,
    )
    continuous_frac = raw_frac if is_native else unwrap_fractional_continuity(raw_frac)
    structures = pymatgen_structures_from_frames(symbols, basis_frames, continuous_frac)
    analyzer, time_ps, msd, used_structures = diffusion_analyzer_from_structures(
        structures=structures,
        specie=args.specie,
        temperature_K=temperature_K,
        time_step_fs=dt_fs,
        args=args,
        continuous_frac=continuous_frac,
    )
    handoff = {
        "producer": "ase-md" if is_native else "historical-ase",
        "segments": len(segment_dirs),
        "global_steps": global_steps.tolist(),
        "physical_time_fs": physical_times_fs.tolist(),
        "temperature_K": temperature_K,
        "frame_step_fs": dt_fs,
        "timestep_fs": _first_metadata(metadata_records, "timestep_fs") if is_native else None,
        "trajectory_interval": _first_metadata(metadata_records, "trajectory_interval") if is_native else None,
        "ensemble": _first_metadata(metadata_records, "ensemble") if is_native else None,
        "model": _first_metadata(metadata_records, "model") if is_native else None,
        "structure_path": _first_metadata(metadata_records, "structure_path") if is_native else None,
    }
    return RunData(
        dataset=dataset,
        run_dir=run_dir,
        temperature_K=temperature_K,
        source_kind="ase_md_artifact" if is_native else "ase_trajectory",
        source_path=traj_path,
        source_paths=tuple(dict.fromkeys(path for path in source_paths if path.is_file())),
        handoff=handoff,
        time_ps=time_ps,
        msd_A2=np.asarray(msd, dtype=float),
        n_mobile=int(mobile_mask.sum()),
        volume_A3=float(frames[0].get_volume()),
        drift_correction="pymatgen-framework-drift",
        notes=f"DiffusionAnalyzer over {len(segment_dirs)} ASE trajectory segment(s)",
        analyzer=analyzer,
        structure=used_structures[0],
        time_step_fs=dt_fs,
        step_skip=int(args.diffusion_analyzer_step_skip),
        uses_ase=True,
        structures=tuple(used_structures),
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


def _project_root_from_attempt(segment_dir: Path) -> Path | None:
    for parent in (segment_dir, *segment_dir.parents):
        if parent.name == ".mlipipe":
            return parent.parent
    return None


def _lammps_input_manifest(segment_dir: Path) -> tuple[dict[str, Any], Path | None]:
    direct = segment_dir / "lammps-input-manifest.json"
    if direct.is_file():
        return _read_json_object(direct), direct
    project_root = _project_root_from_attempt(segment_dir)
    if project_root is None:
        return {}, None
    for name in ("run-manifest.final.json", "run-manifest.json"):
        run_manifest_path = segment_dir / name
        run_manifest = _read_json_object(run_manifest_path)
        inputs = run_manifest.get("inputs")
        if not isinstance(inputs, dict):
            continue
        raw = inputs.get("lammps_input_manifest") or inputs.get("lammps-input-manifest")
        if not isinstance(raw, str) or not raw:
            continue
        candidate = Path(raw).expanduser()
        candidate = candidate if candidate.is_absolute() else project_root / candidate
        if candidate.is_file():
            return _read_json_object(candidate), candidate.resolve()
    return {}, None


def _lammps_segment_metadata(segment_dir: Path) -> dict[str, Any]:
    result_path = segment_dir / "lammps-execution-result.json"
    result = _read_json_object(result_path)
    parameters = _approved_parameters(segment_dir, "lammps_calculation")
    manifest, manifest_path = _lammps_input_manifest(segment_dir)
    md = manifest.get("md") if isinstance(manifest.get("md"), dict) else {}
    manifest_model = manifest.get("model") if isinstance(manifest.get("model"), dict) else {}
    return {
        "temperature_K": result.get(
            "temperature_K", parameters.get("temperature_k", md.get("temperature_k"))
        ),
        "timestep_fs": result.get(
            "timestep_fs", parameters.get("timestep_fs", md.get("timestep_fs"))
        ),
        "dump_interval": result.get(
            "dump_interval", parameters.get("dump_interval", md.get("dump_interval"))
        ),
        "type_map": result.get(
            "type_map", parameters.get("type_map", md.get("type_map"))
        ),
        "ensemble": result.get("ensemble", parameters.get("ensemble", md.get("ensemble"))),
        "model": _model_record(result.get("model") or manifest_model, parameters),
        "structure_path": result.get("structure_path", manifest.get("structure_path")),
        "result_path": result_path,
        "manifest_path": manifest_path,
    }


def _lammps_data_path(segment_dir: Path, requested_name: str | None) -> Path | None:
    for name in ("final.data", "structure.data", requested_name, "data.LYC"):
        if isinstance(name, str) and name and (segment_dir / name).is_file():
            return segment_dir / name
    return None


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


def load_lammps_trajectory(
    run_dir: Path,
    dataset: str,
    args,
    native_segment_dirs: tuple[Path, ...] | None = None,
) -> RunData:
    require_formal_diffusion_api()
    native_segments = list(
        native_segment_dirs or _native_segment_dirs(run_dir, "trajectory.lammpstrj")
    )
    is_native = bool(native_segments)
    segment_dirs = native_segments or [run_dir]
    trajectory_name = "trajectory.lammpstrj" if is_native else "traj.lammpstrj"
    metadata_records = (
        [_lammps_segment_metadata(path) for path in segment_dirs] if is_native else []
    )
    if is_native:
        _same_native_settings(
            metadata_records,
            ("temperature_K", "timestep_fs", "dump_interval", "type_map", "ensemble", "model", "structure_path"),
            "LAMMPS-MD",
        )

    merged: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, bool]] = {}
    ids_ref = None
    symbols_ref = None
    coordinate_modes: list[str] = []
    source_paths: list[Path] = []
    requested_data_name = getattr(args, "lammps_data_name", None)

    for segment_index, segment_dir in enumerate(segment_dirs):
        dump_path = segment_dir / trajectory_name
        source_paths.append(dump_path)
        metadata = metadata_records[segment_index] if is_native else {}
        type_map = metadata.get("type_map")
        type_to_element = (
            {index + 1: str(element) for index, element in enumerate(type_map)}
            if isinstance(type_map, list)
            else {}
        )
        data_path = _lammps_data_path(segment_dir, requested_data_name)
        if data_path is not None:
            _, _, data_type_map = parse_lammps_data(
                data_path, args.specie, getattr(args, "mobile_type", None)
            )
            type_to_element = {**data_type_map, **type_to_element}
            source_paths.append(data_path)
        for key in ("result_path", "manifest_path"):
            path = metadata.get(key)
            if isinstance(path, Path) and path.is_file():
                source_paths.append(path)

        for timestep, columns, rows, box_header, box_lines in read_lammps_dump_frames(dump_path):
            col = {name: i for i, name in enumerate(columns)}
            if "id" not in col:
                raise ValueError(f"LAMMPS dump lacks an id column: {dump_path}")
            has_images = all(name in col for name in ("ix", "iy", "iz"))
            if has_images and all(name in col for name in ("xs", "ys", "zs")):
                this_mode, coord_names, scaled, exact = (
                    "scaled_with_image_flags",
                    ("xs", "ys", "zs"),
                    True,
                    True,
                )
            elif has_images and all(name in col for name in ("x", "y", "z")):
                this_mode, coord_names, scaled, exact = (
                    "cartesian_with_image_flags",
                    ("x", "y", "z"),
                    False,
                    True,
                )
            elif all(name in col for name in ("xu", "yu", "zu")):
                this_mode, coord_names, scaled, exact = (
                    "unwrapped_cartesian",
                    ("xu", "yu", "zu"),
                    False,
                    True,
                )
            elif all(name in col for name in ("xsu", "ysu", "zsu")):
                this_mode, coord_names, scaled, exact = (
                    "unwrapped_scaled",
                    ("xsu", "ysu", "zsu"),
                    True,
                    True,
                )
            elif all(name in col for name in ("x", "y", "z")):
                this_mode, coord_names, scaled, exact = (
                    "wrapped_cartesian_continuity",
                    ("x", "y", "z"),
                    False,
                    False,
                )
            elif all(name in col for name in ("xs", "ys", "zs")):
                this_mode, coord_names, scaled, exact = (
                    "wrapped_scaled_continuity",
                    ("xs", "ys", "zs"),
                    True,
                    False,
                )
            else:
                raise ValueError(
                    f"LAMMPS dump lacks x/y/z, xs/ys/zs, xu/yu/zu, or xsu/ysu/zsu: {dump_path}"
                )
            coordinate_modes.append(this_mode)
            origin, cell = parse_lammps_dump_box(box_header, box_lines)
            inverse_cell = np.linalg.inv(cell)

            parsed = []
            for row in rows:
                atom_id = int(row[col["id"]])
                values = np.asarray(
                    [float(row[col[name]]) for name in coord_names], dtype=float
                )
                fractional = values if scaled else (values - origin) @ inverse_cell
                if this_mode in {"scaled_with_image_flags", "cartesian_with_image_flags"}:
                    fractional = fractional + np.asarray(
                        [int(row[col[name]]) for name in ("ix", "iy", "iz")],
                        dtype=float,
                    )
                element = row[col["element"]] if "element" in col else None
                atom_type = int(row[col["type"]]) if "type" in col else None
                if element is None and atom_type is not None:
                    element = type_to_element.get(atom_type)
                if element is None:
                    raise ValueError(
                        "LAMMPS atom species are absent from both dump element and upstream type_map"
                    )
                parsed.append((atom_id, fractional, element))
            parsed.sort(key=lambda item: item[0])
            ids = np.asarray([item[0] for item in parsed], dtype=int)
            fractional = np.asarray([item[1] for item in parsed], dtype=float)
            elements = np.asarray([item[2] for item in parsed], dtype=object)
            if ids_ref is None:
                ids_ref, symbols_ref = ids, elements
            elif not np.array_equal(ids, ids_ref):
                raise ValueError(f"Atom ids changed order/content in {dump_path}")
            elif not np.array_equal(elements, symbols_ref):
                raise ValueError(f"Atom species changed order/content in {dump_path}")
            if timestep in merged:
                previous = merged[timestep]
                if (
                    not np.allclose(previous[0], cell)
                    or not np.allclose(np.mod(previous[1], 1.0), np.mod(fractional, 1.0))
                    or not np.array_equal(previous[2], elements)
                ):
                    raise ValueError(
                        f"LAMMPS restart boundary timestep {timestep} contains different frames"
                    )
                continue
            merged[int(timestep)] = (cell, fractional, elements, ids, this_mode, exact)

    ordered = sorted(merged.items())
    timesteps = np.asarray([item[0] for item in ordered], dtype=int)
    if len(timesteps) < 2:
        raise ValueError(f"Need at least two LAMMPS frames in {run_dir}")
    assert symbols_ref is not None
    mobile_mask = symbols_ref == args.specie
    if not np.any(mobile_mask):
        raise ValueError(f"No {args.specie} atoms found in {run_dir}")
    timestep_deltas = np.diff(timesteps.astype(float))
    if np.any(timestep_deltas <= 0) or not np.allclose(
        timestep_deltas, timestep_deltas[0], rtol=0.0, atol=1e-12
    ):
        raise ValueError("LAMMPS formal transport requires strictly ordered, uniform frame timesteps")

    if is_native:
        timestep_values = [
            float(record["timestep_fs"])
            for record in metadata_records
            if record.get("timestep_fs") is not None
        ]
        if not timestep_values:
            raise ValueError("LAMMPS-MD artifact lacks timestep_fs in result/manifest/approved plan")
        timestep_ps = timestep_values[0] / 1000.0
        dump_interval = _first_metadata(metadata_records, "dump_interval")
        if dump_interval is not None and not np.allclose(
            timestep_deltas, float(dump_interval), rtol=0.0, atol=1e-12
        ):
            raise ValueError("LAMMPS dump timesteps disagree with upstream dump_interval")
    else:
        timestep_ps = infer_lammps_timestep_ps(
            run_dir, getattr(args, "lammps_timestep_ps", None)
        )
    raw_times_ps = (timesteps.astype(float) - float(timesteps[0])) * timestep_ps
    idx = slice_by_time_indices(raw_times_ps, args.trajectory_start_ps, args.trajectory_end_ps)
    if idx.size < 2:
        raise ValueError(f"Trajectory segment leaves fewer than two frames: {run_dir}")

    continuous: list[np.ndarray] = []
    basis_frames: list[np.ndarray] = []
    for _, (cell, fractional, _, _, _, exact) in ordered:
        if not continuous or exact:
            unwrapped = fractional
        else:
            previous = continuous[-1]
            delta = fractional - np.mod(previous, 1.0)
            unwrapped = previous + delta - np.round(delta)
        continuous.append(np.asarray(unwrapped, dtype=float))
        basis_frames.append(np.asarray(cell, dtype=float))
    selected_unwrapped = np.asarray(continuous, dtype=float)[idx]
    selected_basis = np.asarray(basis_frames, dtype=float)[idx]
    structures = pymatgen_structures_from_frames(
        symbols_ref, selected_basis, selected_unwrapped
    )
    frame_step_fs = float(timestep_deltas[0]) * timestep_ps * 1000.0
    temperature_values = [
        float(record["temperature_K"])
        for record in metadata_records
        if record.get("temperature_K") is not None
    ]
    temperature_K = (
        temperature_values[0]
        if temperature_values
        else require_temperature(run_dir, "LAMMPS trajectory", args)
    )
    analyzer, time_ps, msd, used_structures = diffusion_analyzer_from_structures(
        structures=structures,
        specie=args.specie,
        temperature_K=temperature_K,
        time_step_fs=frame_step_fs,
        args=args,
        continuous_frac=selected_unwrapped,
    )
    handoff = {
        "producer": "lammps-md" if is_native else "historical-lammps",
        "segments": len(segment_dirs),
        "global_steps": timesteps[idx].tolist(),
        "physical_time_ps": (timesteps[idx].astype(float) * timestep_ps).tolist(),
        "temperature_K": temperature_K,
        "timestep_fs": timestep_ps * 1000.0,
        "dump_interval": _first_metadata(metadata_records, "dump_interval") if is_native else int(timestep_deltas[0]),
        "type_map": _first_metadata(metadata_records, "type_map") if is_native else None,
        "ensemble": _first_metadata(metadata_records, "ensemble") if is_native else None,
        "model": _first_metadata(metadata_records, "model") if is_native else None,
        "structure_path": _first_metadata(metadata_records, "structure_path") if is_native else None,
        "coordinate_modes": list(dict.fromkeys(coordinate_modes)),
    }
    return RunData(
        dataset=dataset,
        run_dir=run_dir,
        temperature_K=temperature_K,
        source_kind="lammps_md_artifact" if is_native else "lammps_trajectory",
        source_path=source_paths[0],
        source_paths=tuple(dict.fromkeys(path for path in source_paths if path.is_file())),
        handoff=handoff,
        time_ps=time_ps,
        msd_A2=np.asarray(msd, dtype=float),
        n_mobile=int(np.sum(mobile_mask)),
        volume_A3=float(used_structures[0].volume),
        drift_correction="pymatgen-framework-drift",
        notes=(
            f"DiffusionAnalyzer over {len(segment_dirs)} LAMMPS segment(s); "
            f"lammps_timestep_ps={timestep_ps:g}; coordinate_modes={','.join(dict.fromkeys(coordinate_modes))}"
        ),
        analyzer=analyzer,
        structure=used_structures[0],
        time_step_fs=frame_step_fs,
        step_skip=int(args.diffusion_analyzer_step_skip),
        structures=tuple(used_structures),
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
    require_formal_diffusion_api()
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
    symbols = np.asarray(symbols)
    mobile_mask = symbols == args.specie
    if not mobile_mask.any():
        raise ValueError(f"No {args.specie} atoms found in {vasprun_path}")

    volumes = np.abs(np.linalg.det(basis_frames))
    structures = pymatgen_structures_from_frames(symbols, basis_frames, frac_frames)
    analyzer, time_ps, msd, used_structures = diffusion_analyzer_from_structures(
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
        source_kind="vasp_trajectory",
        source_path=vasprun_path,
        time_ps=time_ps,
        msd_A2=np.asarray(msd, dtype=float),
        n_mobile=int(mobile_mask.sum()),
        volume_A3=float(np.median(volumes)),
        drift_correction="pymatgen-framework-drift",
        notes=(
            f"DiffusionAnalyzer.from_structures over VASP frames; POTIM_fs={potim_fs:g}; "
            f"n_raw_frames={len(raw_times_ps)}"
        ),
        analyzer=analyzer,
        structure=used_structures[0],
        time_step_fs=potim_fs,
        step_skip=int(args.diffusion_analyzer_step_skip),
        structures=tuple(used_structures),
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
        time_ps = raw_time * float(step_ps)
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
    if np.any(time_ps < 0.0):
        raise ValueError(
            f"MSD time must be physical non-negative lag/elapsed time: {msd_path}"
        )
    if np.any(np.diff(time_ps) <= 0.0):
        raise ValueError(f"MSD time must be strictly increasing: {msd_path}")

    notes = (
        f"msd_file={msd_path.name}; time_column={time_name if time_name else time_col}; "
        f"msd_column={msd_name if msd_name else msd_col}; time_unit={time_unit}; "
        f"msd_unit={msd_unit}; time_semantics=physical_lag_or_elapsed"
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


def load_optional_msd_structure(run_dir: Path, args):
    raw_path = getattr(args, "msd_structure", None)
    if raw_path is None:
        return None, None
    structure_path = Path(raw_path)
    if not structure_path.is_absolute():
        structure_path = run_dir / structure_path
    structure_path = structure_path.resolve()
    if not structure_path.is_file():
        raise FileNotFoundError(f"MSD Structure does not exist: {structure_path}")
    try:
        from pymatgen.core import Structure
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(FORMAL_DEPENDENCY_ERROR) from exc
    structure = Structure.from_file(structure_path)
    if float(structure.composition[args.specie]) <= 0:
        raise ValueError(f"MSD Structure contains no {args.specie}")
    return structure, structure_path


def load_msd_file(run_dir: Path, dataset: str, args) -> RunData:
    require_formal_diffusion_api()
    msd_path = find_msd_file(run_dir, args)
    if msd_path is None:
        raise FileNotFoundError(f"No MSD table found in {run_dir}")

    time_ps, msd_A2, notes = read_msd_table(msd_path, run_dir, args)
    temperature_K = infer_msd_temperature(run_dir, args)
    structure, structure_path = load_optional_msd_structure(run_dir, args)
    n_mobile = (
        None if structure is None else int(round(float(structure.composition[args.specie])))
    )
    volume_A3 = None if structure is None else float(structure.volume)

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
        structure=structure,
        structure_path=structure_path,
    )


def load_run_data(
    dataset: str,
    run_dir: Path,
    args,
    native_segment_dirs: tuple[Path, ...] | None = None,
) -> RunData:
    if args.source == "msd":
        return load_msd_file(run_dir, dataset, args)

    if args.source in {"trajectory", "vasp"}:
        if (run_dir / args.vasp_file_name).exists():
            return load_vasp_aimd(run_dir, dataset, args)
        if args.source == "vasp":
            raise FileNotFoundError(f"No {args.vasp_file_name} in {run_dir}")
        if _native_segment_dirs(run_dir, "trajectory.traj") or (run_dir / "production.traj").exists():
            return load_ase_trajectory(run_dir, dataset, args, native_segment_dirs)
        if _native_segment_dirs(run_dir, "trajectory.lammpstrj") or (run_dir / "traj.lammpstrj").exists():
            return load_lammps_trajectory(run_dir, dataset, args, native_segment_dirs)
        raise FileNotFoundError(f"No trajectory file in {run_dir}")

    if _native_segment_dirs(run_dir, "trajectory.traj") or (run_dir / "production.traj").exists():
        return load_ase_trajectory(run_dir, dataset, args, native_segment_dirs)
    if _native_segment_dirs(run_dir, "trajectory.lammpstrj") or (run_dir / "traj.lammpstrj").exists():
        return load_lammps_trajectory(run_dir, dataset, args, native_segment_dirs)
    if (run_dir / args.vasp_file_name).exists():
        return load_vasp_aimd(run_dir, dataset, args)
    msd_path = find_msd_file(run_dir, args)
    if msd_path is not None:
        return load_msd_file(run_dir, dataset, args)
    raise FileNotFoundError(f"No known post-processing input in {run_dir}")


def radial_distribution_curve(
    structures: tuple[object, ...],
    pair: tuple[str, str],
    r_min_angstrom: float,
    r_max_angstrom: float,
    bins: int,
) -> dict[str, np.ndarray]:
    """Average one partial RDF over an explicit trajectory window."""

    if not structures:
        raise ValueError("RDF requires trajectory structures")
    edges = np.linspace(float(r_min_angstrom), float(r_max_angstrom), int(bins) + 1)
    shell_volumes = 4.0 * math.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)
    counts = np.zeros(int(bins), dtype=float)
    normalization = np.zeros(int(bins), dtype=float)
    first_specie, second_specie = pair

    for structure in structures:
        matrix = np.asarray(structure.lattice.matrix, dtype=float)
        volume = float(abs(np.linalg.det(matrix)))
        face_areas = [
            np.linalg.norm(np.cross(matrix[1], matrix[2])),
            np.linalg.norm(np.cross(matrix[0], matrix[2])),
            np.linalg.norm(np.cross(matrix[0], matrix[1])),
        ]
        half_minimum_height = 0.5 * min(volume / area for area in face_areas)
        if float(r_max_angstrom) > half_minimum_height + 1.0e-10:
            raise ValueError(
                "rdf_r_max_angstrom exceeds half the minimum periodic cell height"
            )

        symbols = [str(site.specie.symbol) for site in structure]
        first = np.asarray(
            [index for index, symbol in enumerate(symbols) if symbol == first_specie],
            dtype=int,
        )
        second = np.asarray(
            [index for index, symbol in enumerate(symbols) if symbol == second_specie],
            dtype=int,
        )
        if first.size == 0 or second.size == 0:
            raise ValueError(
                f"RDF pair {first_specie}-{second_specie} is absent from a trajectory frame"
            )

        distances = np.asarray(structure.distance_matrix, dtype=float)
        if first_specie == second_specie:
            if first.size < 2:
                raise ValueError(f"RDF pair {first_specie}-{second_specie} needs two atoms")
            selected = distances[np.ix_(first, first)][np.triu_indices(first.size, k=1)]
            pair_count = first.size * (first.size - 1) / 2.0
        else:
            selected = distances[np.ix_(first, second)].reshape(-1)
            pair_count = float(first.size * second.size)
        counts += np.histogram(selected, bins=edges)[0]
        normalization += pair_count * shell_volumes / volume

    if np.any(normalization <= 0.0):
        raise ValueError("RDF normalization is not positive")
    return {
        "r_min_angstrom": edges[:-1],
        "r_max_angstrom": edges[1:],
        "r_center_angstrom": 0.5 * (edges[:-1] + edges[1:]),
        "g_r": counts / normalization,
    }


def density_m3(n_mobile: int | None, volume_A3: float | None):
    if n_mobile is None or volume_A3 is None or volume_A3 <= 0:
        return None
    return n_mobile / (volume_A3 * 1e-30)


def formal_result_values(run: RunData, args) -> dict:
    """Return formal transport values exclusively from pymatgen public APIs."""

    api = require_formal_diffusion_api()
    if not math.isfinite(float(run.temperature_K)) or float(run.temperature_K) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    time_ps = np.asarray(run.time_ps, dtype=float)
    msd_A2 = np.asarray(run.msd_A2, dtype=float)
    if time_ps.ndim != 1 or msd_A2.shape != time_ps.shape or time_ps.size < 3:
        raise ValueError("formal transport requires paired one-dimensional time/MSD arrays")

    if run.analyzer is not None:
        analyzer = run.analyzer
        used = np.ones(time_ps.shape, dtype=bool)
        diffusivity = float(analyzer.diffusivity)
        diffusivity_std = finite_or_none(analyzer.diffusivity_std_dev)
        conductivity = finite_or_none(analyzer.conductivity)
        conductivity_std = finite_or_none(analyzer.conductivity_std_dev)
        chg_diffusivity = finite_or_none(analyzer.chg_diffusivity)
        chg_conductivity = finite_or_none(analyzer.chg_conductivity)
        haven_ratio = finite_or_none(analyzer.haven_ratio)
        diffusion_method = "pymatgen-diffusion-analyzer"
        conductivity_method = "pymatgen-diffusion-analyzer"
        conductivity_reason = None
    else:
        used = np.isfinite(time_ps) & np.isfinite(msd_A2)
        if args.fit_start_ps is not None:
            used &= time_ps >= float(args.fit_start_ps)
        if args.fit_end_ps is not None:
            used &= time_ps <= float(args.fit_end_ps)
        smoothed = diffusion_analyzer_smoothed_arg(args)
        if smoothed == "max":
            used &= time_ps > 0.0
        if int(used.sum()) < int(args.min_msd_fit_points):
            raise ValueError(
                f"MSD analysis window leaves {int(used.sum())} points; "
                f"need at least {args.min_msd_fit_points}"
            )
        dt_fs = np.asarray(time_ps[used], dtype=float) * 1000.0
        diffusion_pair = api["get_diffusivity_from_msd"](
            np.asarray(msd_A2[used], dtype=float),
            dt_fs,
            smoothed=smoothed,
        )
        diffusivity = float(diffusion_pair[0])
        diffusivity_std = finite_or_none(diffusion_pair[1])
        diffusion_method = "pymatgen-get-diffusivity-from-msd"
        chg_diffusivity = None
        chg_conductivity = None
        haven_ratio = None
        conductivity_std = None
        if run.structure is None:
            conductivity = None
            conductivity_method = "unavailable"
            conductivity_reason = "MSD-only conductivity requires a real Structure"
        else:
            factor = float(
                api["get_conversion_factor"](
                    run.structure, args.specie, float(run.temperature_K)
                )
            )
            conductivity = diffusivity * factor
            conductivity_std = (
                None if diffusivity_std is None else diffusivity_std * factor
            )
            conductivity_method = "pymatgen-get-conversion-factor"
            conductivity_reason = None

    if not math.isfinite(diffusivity) or diffusivity <= 0.0:
        raise ValueError("pymatgen returned a non-positive or non-finite diffusivity")
    return {
        "time_ps": time_ps,
        "msd_A2": msd_A2,
        "used": used,
        "diffusivity_cm2_s": diffusivity,
        "diffusivity_std_dev_cm2_s": diffusivity_std,
        "conductivity_mS_cm": conductivity,
        "conductivity_std_dev_mS_cm": conductivity_std,
        "chg_diffusivity_cm2_s": chg_diffusivity,
        "chg_conductivity_mS_cm": chg_conductivity,
        "haven_ratio": haven_ratio,
        "diffusion_method": diffusion_method,
        "conductivity_method": conductivity_method,
        "conductivity_unavailable_reason": conductivity_reason,
    }


def analyze_run(run: RunData, args, output_dir: Path):
    values = formal_result_values(run, args)
    time_ps = values["time_ps"]
    msd_A2 = values["msd_A2"]
    used = values["used"]

    slug = safe_slug(run)
    curve_path = output_dir / "msd_curves" / f"{slug}_msd.csv"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve_data = {
        "time_ps": time_ps,
        "msd_A2": msd_A2,
        "used_for_analysis": used,
    }
    pd.DataFrame(curve_data).to_csv(curve_path, index=False)

    plot_path = output_dir / "msd_fits" / f"{slug}_msd_fit.html"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_single_msd(run, plot_path)

    return {
        "dataset": run.dataset,
        "temperature_K": run.temperature_K,
        "run_dir": str(run.run_dir),
        "source_kind": run.source_kind,
        "source_path": str(run.source_path),
        "source_file": str(run.source_path),
        "drift_correction": run.drift_correction,
        "n_msd_points": int(time_ps.size),
        "n_analysis_points": int(np.sum(used)),
        "analysis_start_ps": float(np.min(time_ps[used])),
        "analysis_end_ps": float(np.max(time_ps[used])),
        "diffusivity_cm2_s": values["diffusivity_cm2_s"],
        "diffusivity_std_dev_cm2_s": values["diffusivity_std_dev_cm2_s"],
        "diffusivity_stderr_cm2_s": values["diffusivity_std_dev_cm2_s"],
        "chg_diffusivity_cm2_s": values["chg_diffusivity_cm2_s"],
        "n_mobile_ions": run.n_mobile,
        "volume_A3": run.volume_A3,
        "mobile_ion_density_m3": density_m3(run.n_mobile, run.volume_A3),
        "conductivity_NE_mS_cm": values["conductivity_mS_cm"],
        "conductivity_std_dev_mS_cm": values["conductivity_std_dev_mS_cm"],
        "chg_conductivity_mS_cm": values["chg_conductivity_mS_cm"],
        "conductivity_available": values["conductivity_mS_cm"] is not None,
        "conductivity_unavailable_reason": values["conductivity_unavailable_reason"],
        "diffusion_method": values["diffusion_method"],
        "conductivity_method": values["conductivity_method"],
        "arrhenius_method": "pymatgen-fit-arrhenius-linear",
        "haven_ratio": values["haven_ratio"],
        "smoothed": args.diffusion_analyzer_smoothed,
        "min_obs": args.diffusion_analyzer_min_obs,
        "avg_nsteps": args.diffusion_analyzer_avg_nsteps,
        "step_skip": run.step_skip,
        "time_step_fs": run.time_step_fs,
        "species": args.specie,
        "msd_curve_csv": str(curve_path),
        "msd_fit_html": str(plot_path),
        "notes": run.notes,
    }


def safe_slug(run: RunData) -> str:
    temp = f"{run.temperature_K:g}K".replace(".", "p")
    dataset = re.sub(r"[^A-Za-z0-9_.-]+", "_", run.dataset)
    return f"{dataset}_{temp}"


def plot_single_msd(run: RunData, output_path: Path):
    time_ps = np.asarray(run.time_ps, dtype=float)
    msd_A2 = np.asarray(run.msd_A2, dtype=float)
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
    fig.update_layout(
        template="plotly_white",
        title=f"{run.dataset} T={run.temperature_K:g} K",
        xaxis_title="time (ps)",
        yaxis_title="MSD (A^2)",
        legend_title_text="",
    )
    fig.write_html(output_path, include_plotlyjs="cdn")


def _arrhenius_fit_evidence(temperatures, diffusivities, api):
    """Run the declared pymatgen fit and score its returned Arrhenius parameters."""

    from scipy import constants

    T = np.asarray(temperatures, dtype=float)
    D = np.asarray(diffusivities, dtype=float)
    ea_eV, prefactor_cm2_s, ea_std_eV = api["fit_arrhenius"](
        T, D, mode="linear"
    )
    ea_eV = float(ea_eV)
    prefactor_cm2_s = float(prefactor_cm2_s)
    if not math.isfinite(ea_eV) or not math.isfinite(prefactor_cm2_s) or prefactor_cm2_s <= 0:
        raise ValueError("pymatgen returned a non-finite Arrhenius fit")
    boltzmann_eV_K = constants.k / constants.e
    predicted_log_diffusivity = np.log(prefactor_cm2_s) - ea_eV / (
        boltzmann_eV_K * T
    )
    residual_sum_squares = float(
        np.sum((np.log(D) - predicted_log_diffusivity) ** 2)
    )
    if not math.isfinite(residual_sum_squares):
        raise ValueError("Arrhenius log-space residuals are non-finite")
    return {
        "Ea_eV": ea_eV,
        "Ea_stderr_eV": finite_or_none(ea_std_eV),
        "D0_cm2_s": prefactor_cm2_s,
        "residual_sum_squares_log_D": residual_sum_squares,
    }


def _arrhenius_bic(point_count: int, residual_sum_squares: float, parameter_count: int):
    bounded_rss = max(
        float(residual_sum_squares), point_count * np.finfo(float).eps
    )
    return float(
        point_count * math.log(bounded_rss / point_count)
        + parameter_count * math.log(point_count)
    )


def _piecewise_arrhenius_evidence(T, D, args, api, single_bic):
    """Return the best valid one-breakpoint model without implementing another fit."""

    minimum = int(args.min_segment_points)
    if T.size < 2 * minimum:
        return None, "insufficient_temperature_points"

    best = None
    for split in range(minimum, T.size - minimum + 1):
        if not T[split - 1] < T[split]:
            continue
        try:
            low = _arrhenius_fit_evidence(T[:split], D[:split], api)
            high = _arrhenius_fit_evidence(T[split:], D[split:], api)
        except (TypeError, ValueError):
            continue
        combined_rss = (
            low["residual_sum_squares_log_D"]
            + high["residual_sum_squares_log_D"]
        )
        bic = _arrhenius_bic(T.size, combined_rss, 4)
        relative_ea_change = abs(low["Ea_eV"] - high["Ea_eV"]) / max(
            0.5 * (abs(low["Ea_eV"]) + abs(high["Ea_eV"])),
            np.finfo(float).tiny,
        )
        candidate = {
            "breakpoint_interval_K": [float(T[split - 1]), float(T[split])],
            "diagnostic_midpoint_K": float((T[split - 1] + T[split]) / 2.0),
            "low_temperature": {
                **low,
                "temperature_range_K": [float(T[0]), float(T[split - 1])],
                "temperatures_K": T[:split].tolist(),
                "diffusivities_cm2_s": D[:split].tolist(),
            },
            "high_temperature": {
                **high,
                "temperature_range_K": [float(T[split]), float(T[-1])],
                "temperatures_K": T[split:].tolist(),
                "diffusivities_cm2_s": D[split:].tolist(),
            },
            "BIC": bic,
            "BIC_parameter_count": 4,
            "residual_sum_squares_log_D": float(combined_rss),
            "delta_BIC": float(single_bic - bic),
            "relative_Ea_change": float(relative_ea_change),
        }
        candidate["BIC_threshold_met"] = (
            candidate["delta_BIC"] >= float(args.piecewise_bic_delta)
        )
        candidate["slope_change_threshold_met"] = (
            candidate["relative_Ea_change"] >= float(args.piecewise_slope_change)
        )
        candidate["automatic_selection_evidence"] = (
            candidate["BIC_threshold_met"]
            and candidate["slope_change_threshold_met"]
        )
        if best is None or candidate["BIC"] < best["BIC"]:
            best = candidate

    return best, "evaluated" if best is not None else "no_valid_breakpoint"


def _direct_target_evidence(results_df: pd.DataFrame, target_temperature_K: float):
    target_rows = results_df.loc[
        np.isclose(
            results_df["temperature_K"].to_numpy(dtype=float),
            float(target_temperature_K),
            rtol=0.0,
            atol=1.0e-8,
        )
    ]
    if len(target_rows) != 1:
        return {
            "available": False,
            "reason": (
                "not_simulated_at_target_temperature"
                if target_rows.empty
                else "multiple_rows_at_target_temperature"
            ),
        }
    row = target_rows.iloc[0]
    return {
        "available": True,
        "diffusivity_cm2_s": finite_or_none(row.get("diffusivity_cm2_s")),
        "conductivity_mS_cm": finite_or_none(row.get("conductivity_NE_mS_cm")),
    }


def _selected_target_prediction(T, D, target_temperature_K, selected_model, piecewise, api):
    target = float(target_temperature_K)
    if selected_model == "single":
        return {
            "status": "available",
            "regime": "single",
            "diffusivity_cm2_s": float(
                api["get_extrapolated_diffusivity"](T, D, target, mode="linear")
            ),
        }

    interval_low, interval_high = piecewise["breakpoint_interval_K"]
    if interval_low < target < interval_high:
        return {
            "status": "ambiguous",
            "regime": "ambiguous",
            "diffusivity_cm2_s": None,
            "reason": "target_temperature_inside_breakpoint_interval",
        }
    branch_name = "low_temperature" if target <= interval_low else "high_temperature"
    branch = piecewise[branch_name]
    return {
        "status": "available",
        "regime": branch_name,
        "diffusivity_cm2_s": float(
            api["get_extrapolated_diffusivity"](
                branch["temperatures_K"],
                branch["diffusivities_cm2_s"],
                target,
                mode="linear",
            )
        ),
        "fit_temperature_range_K": branch["temperature_range_K"],
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

    fit_row_indices = np.flatnonzero(valid)
    order = np.argsort(temperatures[valid], kind="stable")
    T = temperatures[valid][order]
    D = diffusivities[valid][order]
    fit_row_indices = fit_row_indices[order]
    if T.size < 2:
        raise ValueError("Need at least two positive diffusivity values for Arrhenius fit")
    if int(args.min_segment_points) < 3:
        raise ValueError("min_segment_points must be at least 3")

    api = require_formal_diffusion_api()
    single = _arrhenius_fit_evidence(T, D, api)
    single["BIC"] = _arrhenius_bic(
        T.size, single["residual_sum_squares_log_D"], 2
    )
    single["BIC_parameter_count"] = 2
    target_diffusivity = api["get_extrapolated_diffusivity"](
        T, D, float(args.target_temperature_K), mode="linear"
    )
    single[f"D_{args.target_temperature_K:g}K_cm2_s"] = float(target_diffusivity)

    piecewise = None
    piecewise_status = "disabled"
    if args.piecewise != "never":
        piecewise, piecewise_status = _piecewise_arrhenius_evidence(
            T, D, args, api, single["BIC"]
        )

    evidence_detected = bool(
        piecewise is not None and piecewise["automatic_selection_evidence"]
    )
    if piecewise is not None and (
        args.piecewise == "always" or evidence_detected
    ):
        selected_model = "piecewise"
    else:
        selected_model = "single"

    if args.piecewise == "never":
        selection_reason = "piecewise_disabled"
    elif piecewise is None:
        selection_reason = "insufficient_breakpoint_evidence"
    elif evidence_detected:
        selection_reason = "automatic_evidence_thresholds_met"
    elif args.piecewise == "always":
        selection_reason = "piecewise_forced_without_automatic_evidence"
    else:
        selection_reason = "automatic_evidence_thresholds_not_met"

    target_prediction = {
        "temperature_K": float(args.target_temperature_K),
        "arrhenius_prediction": _selected_target_prediction(
            T,
            D,
            args.target_temperature_K,
            selected_model,
            piecewise,
            api,
        ),
        "direct_simulation": _direct_target_evidence(
            results_df, float(args.target_temperature_K)
        ),
    }

    return {
        "fit_scope": "all",
        "specie": args.specie,
        "arrhenius_fit_space": "ln(D_cm2_s) vs 1/T",
        "target_temperature_K": args.target_temperature_K,
        "fit_temperatures_K": T.tolist(),
        "fit_diffusivities_cm2_s": D.tolist(),
        "fit_row_indices": fit_row_indices.astype(int).tolist(),
        "arrhenius_method": "pymatgen-fit-arrhenius-linear",
        "piecewise_mode": args.piecewise,
        "min_segment_points": int(args.min_segment_points),
        "piecewise_slope_change_threshold": float(args.piecewise_slope_change),
        "piecewise_BIC_delta_threshold": float(args.piecewise_bic_delta),
        "BIC_definition": (
            "n*ln(max(RSS_log_D,n*machine_epsilon)/n)+k*ln(n); "
            "k_single=2, k_piecewise=4"
        ),
        "relative_Ea_change_definition": (
            "abs(Ea_low-Ea_high)/(0.5*(abs(Ea_low)+abs(Ea_high)))"
        ),
        "selected_model": selected_model,
        "selection_reason": selection_reason,
        "regime_change_detected": evidence_detected,
        "piecewise_evidence_status": piecewise_status,
        "interpretation": (
            "ARRHENIUS_REGIME_CHANGE_DETECTED"
            if evidence_detected
            else "NO_REGIME_CHANGE_EVIDENCE"
        ),
        "phase_transition_caveat": (
            "Arrhenius breakpoint evidence diagnoses a transport-regime change only; "
            "a phase-transition claim requires independent structural evidence."
        ),
        "single": single,
        "piecewise": piecewise,
        "target_prediction": target_prediction,
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


def build_arrhenius_summary(results_df: pd.DataFrame, args):
    try:
        return summarize_arrhenius(results_df, args)
    except Exception as exc:
        return {
            "fit_scope": args.fit_scope,
            "specie": args.specie,
            "arrhenius_fit_space": "ln(D_cm2_s) vs 1/T",
            "target_temperature_K": args.target_temperature_K,
            "arrhenius_fit_skipped": True,
            "skip_reason": str(exc),
            "available_temperatures_K": results_df["temperature_K"].tolist(),
            "available_diffusivities_cm2_s": results_df["diffusivity_cm2_s"].tolist(),
        }


def _temperature_pairs(
    reference_runs: list[RunData], candidate_runs: list[RunData]
) -> list[tuple[RunData, RunData]]:
    pairs: list[tuple[RunData, RunData]] = []
    for reference in reference_runs:
        matches = [
            candidate
            for candidate in candidate_runs
            if math.isclose(
                float(candidate.temperature_K),
                float(reference.temperature_K),
                rel_tol=0.0,
                abs_tol=1.0e-8,
            )
        ]
        if len(matches) > 1:
            raise ValueError(
                f"multiple MLIP trajectories match {reference.temperature_K:g} K"
            )
        if matches:
            pairs.append((reference, matches[0]))
    return sorted(pairs, key=lambda item: float(item[0].temperature_K))


def _result_row(rows: list[dict[str, Any]], run: RunData) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if Path(str(row["run_dir"])).resolve() == run.run_dir.resolve()
        and math.isclose(
            float(row["temperature_K"]),
            float(run.temperature_K),
            rel_tol=0.0,
            abs_tol=1.0e-8,
        )
    ]
    if len(matches) != 1:
        raise ValueError(f"transport result does not uniquely match {run.run_dir}")
    return matches[0]


def _dataset_fit(summary: dict[str, Any], dataset: str) -> dict[str, Any] | None:
    datasets = summary.get("datasets")
    if isinstance(datasets, dict):
        value = datasets.get(dataset)
        return value if isinstance(value, dict) else None
    if isinstance(summary.get("single"), dict):
        return summary
    return None


def build_aimd_mlip_comparison(
    runs: list[RunData],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    args,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build RDF curves and metric-only benchmark pairs from analyzed trajectories."""

    reference_runs = [run for run in runs if run.source_kind == "vasp_trajectory"]
    candidate_runs = [
        run
        for run in runs
        if run.source_kind in {
            "ase_md_artifact",
            "ase_trajectory",
            "lammps_md_artifact",
            "lammps_trajectory",
        }
    ]
    if not reference_runs or not candidate_runs or len(reference_runs) + len(candidate_runs) != len(runs):
        raise ValueError(
            "RDF comparison requires only VASP AIMD reference and ASE/LAMMPS MLIP trajectories"
        )
    common = _temperature_pairs(reference_runs, candidate_runs)
    if not common:
        raise ValueError("AIMD and MLIP trajectories have no common temperature")

    requested_temperature = getattr(args, "rdf_temperature_K", None)
    if requested_temperature is None:
        if len(common) != 1:
            raise ValueError(
                "rdf_temperature_k is required when more than one common temperature is available"
            )
        rdf_reference, rdf_candidate = common[0]
    else:
        selected = [
            pair
            for pair in common
            if math.isclose(
                float(pair[0].temperature_K),
                float(requested_temperature),
                rel_tol=0.0,
                abs_tol=1.0e-8,
            )
        ]
        if len(selected) != 1:
            raise ValueError("rdf_temperature_k does not select one common-temperature pair")
        rdf_reference, rdf_candidate = selected[0]

    pair = tuple(args.rdf_pair)
    reference_curve = radial_distribution_curve(
        rdf_reference.structures,
        pair,
        args.rdf_r_min_angstrom,
        args.rdf_r_max_angstrom,
        args.rdf_bins,
    )
    candidate_curve = radial_distribution_curve(
        rdf_candidate.structures,
        pair,
        args.rdf_r_min_angstrom,
        args.rdf_r_max_angstrom,
        args.rdf_bins,
    )
    difference = candidate_curve["g_r"] - reference_curve["g_r"]
    curve_table = {
        "r_min_angstrom": reference_curve["r_min_angstrom"],
        "r_max_angstrom": reference_curve["r_max_angstrom"],
        "r_center_angstrom": reference_curve["r_center_angstrom"],
        "aimd_g_r": reference_curve["g_r"],
        "mlip_g_r": candidate_curve["g_r"],
        "difference": difference,
    }

    def evidence_record(task: str, target: str, unit: str, reference, prediction):
        return {
            "model": args.comparison_model,
            "task": task,
            "scenario": args.comparison_scenario,
            "split": args.comparison_split,
            "target": target,
            "unit": unit,
            "reference": json_ready(reference),
            "prediction": json_ready(prediction),
        }

    temperature_label = f"{rdf_reference.temperature_K:g}k".replace(".", "p")
    pair_label = f"{pair[0]}-{pair[1]}".lower()
    records = [
        evidence_record(
            "structural-dynamics",
            f"rdf_{pair_label}_{temperature_label}",
            "dimensionless",
            reference_curve["g_r"],
            candidate_curve["g_r"],
        )
    ]
    transport_errors = []
    for reference, candidate in common:
        reference_row = _result_row(rows, reference)
        candidate_row = _result_row(rows, candidate)
        temperature = float(reference.temperature_K)
        label = f"{temperature:g}k".replace(".", "p")
        for column, target, unit in (
            ("diffusivity_cm2_s", f"diffusivity_{label}", "cm^2/s"),
            ("conductivity_NE_mS_cm", f"conductivity_{label}", "mS/cm"),
        ):
            reference_value = finite_or_none(reference_row.get(column))
            candidate_value = finite_or_none(candidate_row.get(column))
            if reference_value is None or candidate_value is None:
                continue
            records.append(
                evidence_record(
                    "ionic-transport", target, unit, reference_value, candidate_value
                )
            )
            absolute_error = abs(candidate_value - reference_value)
            transport_errors.append(
                {
                    "target": target,
                    "unit": unit,
                    "reference": reference_value,
                    "prediction": candidate_value,
                    "absolute_error": absolute_error,
                    "relative_error": (
                        absolute_error / abs(reference_value)
                        if reference_value != 0.0
                        else None
                    ),
                }
            )

    reference_fit = _dataset_fit(summary, rdf_reference.dataset)
    candidate_fit = _dataset_fit(summary, rdf_candidate.dataset)
    if reference_fit is not None and candidate_fit is not None:
        reference_single = reference_fit.get("single")
        candidate_single = candidate_fit.get("single")
        if isinstance(reference_single, dict) and isinstance(candidate_single, dict):
            target_keys = [
                ("Ea_eV", "activation_energy", "eV"),
                (
                    f"D_{args.target_temperature_K:g}K_cm2_s",
                    f"diffusivity_{args.target_temperature_K:g}k_extrapolated".replace(".", "p"),
                    "cm^2/s",
                ),
                (
                    f"conductivity_{args.target_temperature_K:g}K_mS_cm",
                    f"conductivity_{args.target_temperature_K:g}k".replace(".", "p"),
                    "mS/cm",
                ),
            ]
            for key, target, unit in target_keys:
                reference_value = finite_or_none(reference_single.get(key))
                candidate_value = finite_or_none(candidate_single.get(key))
                if reference_value is not None and candidate_value is not None:
                    records.append(
                        evidence_record(
                            "ionic-transport", target, unit, reference_value, candidate_value
                        )
                    )

    payload = {
        "schema_version": 1,
        "plugin_id": "ionic-transport",
        "model_execution": False,
        "model": args.comparison_model,
        "scenario": args.comparison_scenario,
        "split": args.comparison_split,
        "rdf": {
            "pair": list(pair),
            "temperature_K": float(rdf_reference.temperature_K),
            "r_min_angstrom": float(args.rdf_r_min_angstrom),
            "r_max_angstrom": float(args.rdf_r_max_angstrom),
            "bins": int(args.rdf_bins),
            "trajectory_start_ps": args.trajectory_start_ps,
            "trajectory_end_ps": args.trajectory_end_ps,
            "reference_source": str(rdf_reference.source_path),
            "candidate_source": str(rdf_candidate.source_path),
            "reference_frame_count": len(rdf_reference.structures),
            "candidate_frame_count": len(rdf_candidate.structures),
            "curve_mae": float(np.mean(np.abs(difference))),
            "curve_rmse": float(np.sqrt(np.mean(difference**2))),
        },
        "transport_errors": transport_errors,
        "records": records,
    }
    return payload, curve_table


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


def artifact_record(path: Path, role: str) -> dict:
    resolved = path.resolve()
    return {
        "role": role,
        "path": str(resolved),
    }


def analysis_source_paths(run: RunData, args) -> list[Path]:
    candidates = list(run.source_paths) if run.source_paths else [run.source_path]
    if run.structure_path is not None:
        candidates.append(run.structure_path)
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
        "--source",
        choices=["auto", "trajectory", "vasp", "msd"],
        default="auto",
        help="Read trajectories or precomputed MSD tables; auto chooses trajectory first, then MSD.",
    )
    parser.add_argument(
        "--trajectory-msd-engine",
        choices=["diffusion-analyzer"],
        default="diffusion-analyzer",
        help="Formal trajectory analysis always uses pymatgen DiffusionAnalyzer.",
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
    parser.add_argument(
        "--fit-start-ps",
        type=float,
        default=None,
        help="Optional physical-lag subset start used only for MSD-table input.",
    )
    parser.add_argument(
        "--fit-end-ps",
        type=float,
        default=None,
        help="Optional physical-lag subset end used only for MSD-table input.",
    )
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
        help="Unit of physical lag/elapsed time. step multiplies values by --msd-step-ps.",
    )
    parser.add_argument("--msd-step-ps", type=float, default=None, help="ps per step for MSD tables.")
    parser.add_argument(
        "--msd-unit",
        choices=["auto", "A2", "nm2"],
        default="auto",
        help="Unit of the MSD column.",
    )
    parser.add_argument("--msd-temperature-K", type=float, default=None)
    parser.add_argument(
        "--msd-structure",
        type=Path,
        default=None,
        help="Real structure used only for pymatgen MSD-only conductivity conversion.",
    )
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
    parser.add_argument("--rdf-pair", nargs=2, default=None)
    parser.add_argument("--rdf-r-min-angstrom", type=float, default=None)
    parser.add_argument("--rdf-r-max-angstrom", type=float, default=None)
    parser.add_argument("--rdf-bins", type=int, default=None)
    parser.add_argument("--rdf-temperature-K", type=float, default=None)
    parser.add_argument("--comparison-model", default=None)
    parser.add_argument("--comparison-scenario", default=None)
    parser.add_argument("--comparison-split", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.min_segment_points < 3:
        parser.error("--min-segment-points must be at least 3")
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
    for dataset, run_dir, native_segment_dirs in run_dirs:
        try:
            run = load_run_data(dataset, run_dir, args, native_segment_dirs)
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

    summary = build_arrhenius_summary(results_df, args)
    summary_path = output_dir / "arrhenius_summary.json"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2))

    comparison_paths = []
    if args.rdf_pair is not None:
        try:
            comparison, curve_table = build_aimd_mlip_comparison(
                runs, rows, summary, args
            )
            curve_path = output_dir / "rdf_curves.csv"
            comparison_path = output_dir / "aimd_mlip_comparison.json"
            pd.DataFrame(curve_table).to_csv(curve_path, index=False)
            comparison["rdf"]["curve_csv"] = str(curve_path)
            comparison_path.write_text(
                json.dumps(json_ready(comparison), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            comparison_paths.extend([curve_path, comparison_path])
        except Exception as exc:
            failures.append(
                {"stage": "aimd-mlip-rdf-comparison", "error": str(exc)}
            )
            print(f"RDF comparison failed: {exc}", file=sys.stderr)

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
        *comparison_paths,
    ]
    manifest = {
        "schema_version": 1,
        "plugin_id": "ionic-transport",
        "operation": "analyze-existing",
        "scientific_contract": {
            "formal_implementation": "pymatgen-analysis-diffusion public API",
            "trajectory_diffusion": "DiffusionAnalyzer",
            "msd_only_diffusion": "get_diffusivity_from_msd",
            "msd_only_conductivity": "get_conversion_factor when a real Structure exists",
            "arrhenius": "fit_arrhenius(mode='linear')",
            "arrhenius_model_selection": (
                "single baseline plus at most one adjacent-temperature breakpoint by BIC "
                "and relative activation-energy change"
            ),
            "rdf": "periodic partial pair distribution over the approved trajectory window",
            "historical_implementation": "separate adapter-only legacy reproduction",
        },
        "parameters": json_ready(vars(args)),
        "runtime_provenance": runtime_provenance(uses_ase=any(run.uses_ase for run in runs)),
        "upstream_handoffs": [json_ready(run.handoff) for run in runs if run.handoff],
        "implementation_artifacts": [
            artifact_record(Path(__file__), "packaged-analysis-runner"),
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
        print(f"[{label}] pymatgen single-line baseline: Ea={single_fit['Ea_eV']:.4f} eV")
        print(
            f"[{label}] single-line D_{args.target_temperature_K:g}K="
            f"{single_fit.get(target_key):.4e} cm^2/s"
        )
        print(
            f"[{label}] selected_model={fit_summary['selected_model']} "
            f"reason={fit_summary['selection_reason']}"
        )
        if fit_summary.get("piecewise") is not None:
            piecewise = fit_summary["piecewise"]
            print(
                f"[{label}] breakpoint_interval_K={piecewise['breakpoint_interval_K']} "
                f"delta_BIC={piecewise['delta_BIC']:.4g} "
                f"relative_Ea_change={piecewise['relative_Ea_change']:.4g}"
            )
        prediction = fit_summary["target_prediction"]["arrhenius_prediction"]
        if prediction["status"] == "ambiguous":
            print(
                f"[{label}] target Arrhenius regime is ambiguous inside the "
                "breakpoint interval"
            )
        if fit_summary["regime_change_detected"]:
            print(f"[{label}] ARRHENIUS_REGIME_CHANGE_DETECTED")

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
