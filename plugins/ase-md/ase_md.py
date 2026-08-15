"""ASE molecular dynamics runner for explicit MLIP model artifacts.

This module owns the framework-neutral MD loop. It never downloads a model and
never shells out. Site-specific model lookup is handled by the cluster wrapper;
this runner receives an already resolved, fingerprinted model path and attaches
one of the supported ASE calculators.

Version 0.3 adds checkpoint/restart to single-temperature NVT Langevin and
isotropic MTK NPT trajectories. A restart restores integrator state rather than
merely starting a new trajectory from the last geometry.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
from typing import Any

CALCULATORS = ("deepmd", "m3gnet", "chgnet", "mace")
ENSEMBLES = ("nvt-langevin", "npt-isotropic-mtk")
DEVICES = ("cpu", "cuda")
DTYPES = ("float32", "float64")
RESULT_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_STATE_VERSION = "ase-md-checkpoint-v1"
CHECKPOINT_FILENAME = "md-checkpoint.json"
RNG_ALGORITHM = "PCG64"
NPT_TCHAIN = 3
NPT_PCHAIN = 3
NPT_TLOOP = 1
NPT_PLOOP = 1


class AseMDError(RuntimeError):
    pass


def _prepare_output_directory(path: Path) -> Path:
    """Claim a fresh output root under the common scheduled-runner contract.

    Standalone callers may provide a path that does not exist. The ssh-slurm
    backend instead pre-creates an empty output/ directory as part of its fresh
    attempt workspace. Both forms are fresh; any existing content remains an
    overwrite error.
    """

    candidate = path.expanduser()
    if candidate.is_symlink():
        raise AseMDError(f"output directory must not be a symlink: {candidate}")
    resolved = candidate.resolve()
    if resolved.exists():
        if not resolved.is_dir():
            raise AseMDError(f"output path is not a directory: {resolved}")
        if next(resolved.iterdir(), None) is not None:
            raise AseMDError(f"output directory is not empty: {resolved}")
        return resolved
    resolved.mkdir(parents=True)
    return resolved


def _final_structure_copy(atoms: Any, input_constraints: list[Any]) -> Any:
    """Return final geometry/state without runtime-only attachments."""

    final_atoms = atoms.copy()
    final_atoms.calc = None
    final_atoms.set_constraint(input_constraints)
    return final_atoms


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def framework_version(calculator: str) -> str:
    return {
        "deepmd": lambda: _version("deepmd-kit", "deepmd"),
        "m3gnet": lambda: _version("matgl"),
        "chgnet": lambda: _version("chgnet"),
        "mace": lambda: _version("mace-torch", "mace"),
    }[calculator]()


def _build_calculator(calculator: str, model: Path, device: str, default_dtype: str):
    """Create a calculator only from an explicit local model artifact."""
    if calculator == "deepmd":
        from deepmd.calculator import DP

        return DP(model=str(model))
    if calculator == "chgnet":
        if default_dtype != "float32":
            raise AseMDError("CHGNet ASE MD requires default_dtype=float32")
        from chgnet.model.dynamics import CHGNetCalculator

        return CHGNetCalculator.from_file(str(model), use_device=device)
    if calculator == "mace":
        from mace.calculators import MACECalculator

        return MACECalculator(
            model_paths=str(model), device=device, default_dtype=default_dtype
        )
    if calculator == "m3gnet":
        import matgl
        import torch
        from matgl.ext.ase import PESCalculator

        setter = getattr(matgl, "set_default_dtype", None)
        if callable(setter):
            setter("float", 64 if default_dtype == "float64" else 32)
        if device == "cuda":
            with torch.device("cuda"):
                potential = matgl.load_model(str(model))
        else:
            potential = matgl.load_model(str(model))
        return PESCalculator(potential)
    raise AseMDError(f"unsupported calculator: {calculator}")


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise AseMDError(f"{name} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AseMDError(f"{name} must be finite and positive") from exc
    if not math.isfinite(result) or result <= 0:
        raise AseMDError(f"{name} must be finite and positive")
    return result


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise AseMDError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AseMDError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise AseMDError(f"{name} must be finite")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AseMDError(f"{name} must be a positive integer")
    return value


def _jsonable(value: Any) -> Any:
    """Convert NumPy-rich state to strict JSON without pickle."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise AseMDError(f"checkpoint state contains a non-JSON value: {type(value).__name__}")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_checkpoint(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AseMDError(f"restart checkpoint is missing or unsafe: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AseMDError(f"restart checkpoint is unreadable: {exc}") from exc
    if not isinstance(raw, dict):
        raise AseMDError("restart checkpoint must contain a JSON object")
    return raw


def _npt_stress_probe(atoms: Any, units: Any) -> float:
    """Verify the concrete calculator/model can provide finite ASE stress."""
    implemented = tuple(getattr(atoms.calc, "implemented_properties", ()) or ())
    if "stress" not in implemented:
        raise AseMDError(
            "NPT requires a calculator/model exposing ASE stress in implemented_properties"
        )
    try:
        stress = atoms.get_stress(voigt=False, include_ideal_gas=False)
    except Exception as exc:
        raise AseMDError(f"NPT stress capability probe failed: {exc}") from exc
    if getattr(stress, "shape", None) != (3, 3):
        raise AseMDError("NPT stress capability probe did not return a 3x3 tensor")
    values = [float(value) for row in stress for value in row]
    if any(not math.isfinite(value) for value in values):
        raise AseMDError("NPT stress capability probe returned a non-finite value")
    pressure_au = -sum(float(stress[index, index]) for index in range(3)) / 3.0
    return pressure_au / units.GPa


def _thermo_header(ensemble: str) -> list[str]:
    common = [
        "step",
        "time_fs",
        "temperature_K",
        "potential_energy_eV",
        "kinetic_energy_eV",
        "total_energy_eV",
        "volume_A3",
    ]
    if ensemble == "npt-isotropic-mtk":
        common.extend(["pressure_GPa", "cell_a_A", "cell_b_A", "cell_c_A"])
    return common


def _segment_steps(start: int, total: int, interval: int) -> list[int]:
    values = [start]
    next_step = ((start // interval) + 1) * interval
    values.extend(range(next_step, total + 1, interval))
    if values[-1] != total:
        values.append(total)
    return values


def _base_checkpoint_identity(
    *,
    calculator: str,
    ensemble: str,
    model_id: str,
    model_fingerprint: str,
    structure_fingerprint: str,
    temperature_k: float,
    timestep_fs: float,
    steps: int,
    seed: int,
    device: str,
    default_dtype: str,
    fix_com: bool,
    friction_per_fs: float | None,
    pressure_gpa: float | None,
    thermostat_damping_fs: float | None,
    barostat_damping_fs: float | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "plugin_id": "ase-md",
        "checkpoint_state_version": CHECKPOINT_STATE_VERSION,
        "calculator": calculator,
        "ensemble": ensemble,
        "model": {"id": model_id, "fingerprint": model_fingerprint},
        "structure_fingerprint": structure_fingerprint,
        "temperature_K": temperature_k,
        "timestep_fs": timestep_fs,
        "steps_requested": steps,
        "seed": seed,
        "device": device,
        "default_dtype": default_dtype,
        "fix_com": bool(fix_com),
    }
    if ensemble == "nvt-langevin":
        payload["friction_per_fs"] = friction_per_fs
        payload["rng_algorithm"] = RNG_ALGORITHM
    else:
        payload.update(
            {
                "pressure_GPa": pressure_gpa,
                "thermostat_damping_fs": thermostat_damping_fs,
                "barostat_damping_fs": barostat_damping_fs,
                "thermostat_chain_length": NPT_TCHAIN,
                "barostat_chain_length": NPT_PCHAIN,
                "thermostat_substeps": NPT_TLOOP,
                "barostat_substeps": NPT_PLOOP,
            }
        )
    return payload


def _validate_checkpoint_identity(
    checkpoint: dict[str, Any], expected: dict[str, Any], ase_version: str
) -> int:
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise AseMDError("restart checkpoint schema_version is unsupported")
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise AseMDError(f"restart checkpoint identity mismatch for {key}")
    if checkpoint.get("ase_version") != ase_version:
        raise AseMDError(
            "restart checkpoint ASE version differs from the current environment; "
            "exact integrator-state restart is refused"
        )
    completed = checkpoint.get("completed_steps")
    if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
        raise AseMDError("restart checkpoint completed_steps is invalid")
    total = int(expected["steps_requested"])
    if completed >= total:
        raise AseMDError("restart checkpoint is already at or beyond the requested total steps")
    return completed


def _restore_atoms_from_checkpoint(atoms: Any, checkpoint: dict[str, Any], np: Any) -> None:
    numbers = checkpoint.get("atomic_numbers")
    if numbers != [int(value) for value in atoms.get_atomic_numbers()]:
        raise AseMDError("restart checkpoint atomic ordering differs from the source structure")
    positions = np.asarray(checkpoint.get("positions_A"), dtype=float)
    momenta = np.asarray(checkpoint.get("momenta"), dtype=float)
    cell = np.asarray(checkpoint.get("cell_A"), dtype=float)
    masses = np.asarray(checkpoint.get("masses_amu"), dtype=float)
    pbc = checkpoint.get("pbc")
    if positions.shape != (len(atoms), 3) or momenta.shape != (len(atoms), 3):
        raise AseMDError("restart checkpoint position/momentum shape is invalid")
    if cell.shape != (3, 3) or masses.shape != (len(atoms),):
        raise AseMDError("restart checkpoint cell/mass shape is invalid")
    if not isinstance(pbc, list) or len(pbc) != 3 or any(type(value) is not bool for value in pbc):
        raise AseMDError("restart checkpoint pbc is invalid")
    arrays = [positions, momenta, cell, masses]
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise AseMDError("restart checkpoint contains a non-finite atom state")
    atoms.set_cell(cell, scale_atoms=False)
    atoms.set_positions(positions)
    atoms.set_masses(masses)
    atoms.set_momenta(momenta)
    atoms.set_pbc(pbc)


def _checkpoint_payload(
    *,
    dyn: Any,
    atoms: Any,
    rng: Any,
    base_identity: dict[str, Any],
    ase_version: str,
    ensemble: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        **base_identity,
        "ase_version": ase_version,
        "completed_steps": int(dyn.get_number_of_steps()),
        "atomic_numbers": [int(value) for value in atoms.get_atomic_numbers()],
        "positions_A": _jsonable(atoms.get_positions()),
        "momenta": _jsonable(atoms.get_momenta()),
        "cell_A": _jsonable(atoms.get_cell().array),
        "masses_amu": _jsonable(atoms.get_masses()),
        "pbc": [bool(value) for value in atoms.get_pbc()],
    }
    if ensemble == "nvt-langevin":
        payload["integrator_state"] = {
            "langevin_version": int(getattr(dyn, "_lgv_version", -1)),
            "rng_state": _jsonable(rng.bit_generator.state),
        }
    else:
        payload["integrator_state"] = {
            "q": _jsonable(dyn._q),
            "p": _jsonable(dyn._p),
            "eps": float(dyn._eps),
            "p_eps": float(dyn._p_eps),
            "cell0": _jsonable(dyn._cell0),
            "volume0": float(dyn._volume0),
            "thermostat_eta": _jsonable(dyn._thermostat._eta),
            "thermostat_p_eta": _jsonable(dyn._thermostat._p_eta),
            "barostat_xi": _jsonable(dyn._barostat._xi),
            "barostat_p_xi": _jsonable(dyn._barostat._p_xi),
        }
    return payload


def _restore_npt_integrator(dyn: Any, checkpoint: dict[str, Any], np: Any) -> None:
    state = checkpoint.get("integrator_state")
    if not isinstance(state, dict):
        raise AseMDError("restart checkpoint lacks NPT integrator_state")
    q = np.asarray(state.get("q"), dtype=float)
    p = np.asarray(state.get("p"), dtype=float)
    cell0 = np.asarray(state.get("cell0"), dtype=float)
    thermostat_eta = np.asarray(state.get("thermostat_eta"), dtype=float)
    thermostat_p_eta = np.asarray(state.get("thermostat_p_eta"), dtype=float)
    barostat_xi = np.asarray(state.get("barostat_xi"), dtype=float)
    barostat_p_xi = np.asarray(state.get("barostat_p_xi"), dtype=float)
    if q.shape != dyn._q.shape or p.shape != dyn._p.shape or cell0.shape != (3, 3):
        raise AseMDError("restart checkpoint NPT particle/cell state shape is invalid")
    if thermostat_eta.shape != dyn._thermostat._eta.shape or thermostat_p_eta.shape != dyn._thermostat._p_eta.shape:
        raise AseMDError("restart checkpoint thermostat chain shape is invalid")
    if barostat_xi.shape != dyn._barostat._xi.shape or barostat_p_xi.shape != dyn._barostat._p_xi.shape:
        raise AseMDError("restart checkpoint barostat chain shape is invalid")
    eps = _finite(state.get("eps"), "checkpoint eps")
    p_eps = _finite(state.get("p_eps"), "checkpoint p_eps")
    volume0 = _finite_positive(state.get("volume0"), "checkpoint volume0")
    arrays = [q, p, cell0, thermostat_eta, thermostat_p_eta, barostat_xi, barostat_p_xi]
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise AseMDError("restart checkpoint NPT integrator state contains non-finite values")
    dyn._q = q.copy()
    dyn._p = p.copy()
    dyn._eps = eps
    dyn._p_eps = p_eps
    dyn._cell0 = cell0.copy()
    dyn._volume0 = volume0
    dyn._thermostat._eta = thermostat_eta.copy()
    dyn._thermostat._p_eta = thermostat_p_eta.copy()
    dyn._barostat._xi = barostat_xi.copy()
    dyn._barostat._p_xi = barostat_p_xi.copy()
    dyn._update_atoms()


def run_md(
    *,
    structure: Path,
    model: Path,
    output_dir: Path,
    calculator: str,
    ensemble: str,
    model_id: str,
    model_fingerprint: str,
    structure_fingerprint: str,
    temperature_k: float,
    timestep_fs: float,
    steps: int,
    trajectory_interval: int,
    thermo_interval: int,
    seed: int,
    device: str,
    default_dtype: str,
    friction_per_fs: float | None,
    fix_com: bool,
    pressure_gpa: float | None = None,
    thermostat_damping_fs: float | None = None,
    barostat_damping_fs: float | None = None,
    checkpoint_interval: int | None = None,
    restart_checkpoint: Path | None = None,
    input_format: str | None = None,
    input_index: str = "-1",
) -> dict[str, Any]:
    """Run or exactly resume one NVT or isotropic NPT trajectory segment."""
    if calculator not in CALCULATORS:
        raise AseMDError("calculator must be deepmd, m3gnet, chgnet, or mace")
    if ensemble not in ENSEMBLES:
        raise AseMDError("ensemble must be nvt-langevin or npt-isotropic-mtk")
    if device not in DEVICES:
        raise AseMDError("device must be cpu or cuda")
    if default_dtype not in DTYPES:
        raise AseMDError("default_dtype must be float32 or float64")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise AseMDError("seed must be a non-negative integer")
    steps = _positive_int(steps, "steps")
    trajectory_interval = _positive_int(trajectory_interval, "trajectory_interval")
    thermo_interval = _positive_int(thermo_interval, "thermo_interval")
    if trajectory_interval > steps or thermo_interval > steps:
        raise AseMDError("trajectory/thermo intervals cannot exceed steps")
    if checkpoint_interval is not None:
        checkpoint_interval = _positive_int(checkpoint_interval, "checkpoint_interval")
        if checkpoint_interval > steps:
            raise AseMDError("checkpoint_interval cannot exceed steps")
    if restart_checkpoint is not None and checkpoint_interval is None:
        raise AseMDError("restart requires checkpoint_interval so the resumed attempt remains recoverable")
    temperature_k = _finite_positive(temperature_k, "temperature_k")
    timestep_fs = _finite_positive(timestep_fs, "timestep_fs")

    nvt_friction: float | None = None
    npt_pressure: float | None = None
    npt_tdamp: float | None = None
    npt_pdamp: float | None = None
    if ensemble == "nvt-langevin":
        nvt_friction = _finite_positive(friction_per_fs, "friction_per_fs")
        if any(value is not None for value in (pressure_gpa, thermostat_damping_fs, barostat_damping_fs)):
            raise AseMDError("pressure/barostat parameters are NPT-only")
    else:
        if friction_per_fs is not None:
            raise AseMDError("friction_per_fs is NVT-only")
        npt_pressure = _finite(pressure_gpa, "pressure_gpa")
        npt_tdamp = _finite_positive(thermostat_damping_fs, "thermostat_damping_fs")
        npt_pdamp = _finite_positive(barostat_damping_fs, "barostat_damping_fs")
        if fix_com:
            raise AseMDError(
                "npt-isotropic-mtk requires fix_com=false because ASE IsotropicMTKNPT "
                "does not support constraints"
            )

    structure = structure.expanduser().resolve()
    model = model.expanduser().resolve()
    output_dir = output_dir.expanduser()
    restart_checkpoint = restart_checkpoint.expanduser().resolve() if restart_checkpoint else None
    if not structure.is_file():
        raise AseMDError(f"structure is not a regular file: {structure}")
    if calculator == "m3gnet":
        if not model.is_dir():
            raise AseMDError("M3GNet/MatGL model must be a local model directory")
    elif not model.is_file():
        raise AseMDError(f"{calculator} model must be a local model file")
    output_dir = _prepare_output_directory(output_dir)

    import numpy as np
    from ase import units
    from ase.constraints import FixCom
    from ase.io import read, write
    from ase.io.trajectory import Trajectory
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    ase_version = _version("ase")
    atoms = read(str(structure), format=input_format, index=input_index)
    if not hasattr(atoms, "get_positions"):
        raise AseMDError("structure selection did not produce exactly one ASE Atoms object")
    if len(atoms) == 0:
        raise AseMDError("structure contains no atoms")
    input_constraints = list(atoms.constraints)
    if ensemble == "npt-isotropic-mtk":
        if atoms.cell.rank != 3 or not all(bool(value) for value in atoms.get_pbc()):
            raise AseMDError("npt-isotropic-mtk requires a full-rank 3D periodic cell")
        if not math.isfinite(float(atoms.get_volume())) or float(atoms.get_volume()) <= 0:
            raise AseMDError("npt-isotropic-mtk requires a finite positive cell volume")
        if len(atoms.constraints) > 0:
            raise AseMDError("npt-isotropic-mtk does not support pre-existing ASE constraints")
    elif fix_com:
        constraints = list(atoms.constraints)
        constraints.append(FixCom())
        atoms.set_constraint(constraints)

    base_checkpoint_identity = _base_checkpoint_identity(
        calculator=calculator,
        ensemble=ensemble,
        model_id=model_id,
        model_fingerprint=model_fingerprint,
        structure_fingerprint=structure_fingerprint,
        temperature_k=temperature_k,
        timestep_fs=timestep_fs,
        steps=steps,
        seed=seed,
        device=device,
        default_dtype=default_dtype,
        fix_com=fix_com,
        friction_per_fs=nvt_friction,
        pressure_gpa=npt_pressure,
        thermostat_damping_fs=npt_tdamp,
        barostat_damping_fs=npt_pdamp,
    )
    checkpoint: dict[str, Any] | None = None
    start_step = 0
    restart_sha256: str | None = None
    if restart_checkpoint is not None:
        checkpoint = _read_checkpoint(restart_checkpoint)
        start_step = _validate_checkpoint_identity(
            checkpoint, base_checkpoint_identity, ase_version
        )
        _restore_atoms_from_checkpoint(atoms, checkpoint, np)
        restart_sha256 = _sha256(restart_checkpoint)

    atoms.calc = _build_calculator(calculator, model, device, default_dtype)
    initial_pressure_gpa: float | None = None
    if ensemble == "npt-isotropic-mtk":
        initial_pressure_gpa = _npt_stress_probe(atoms, units)

    rng = np.random.Generator(np.random.PCG64(seed))
    if checkpoint is None:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k, rng=rng)
    elif ensemble == "nvt-langevin":
        state = checkpoint.get("integrator_state")
        if not isinstance(state, dict) or state.get("langevin_version") != int(getattr(Langevin, "_lgv_version", -1)):
            raise AseMDError("restart checkpoint Langevin implementation version differs")
        rng_state = state.get("rng_state")
        if not isinstance(rng_state, dict) or rng_state.get("bit_generator") != RNG_ALGORITHM:
            raise AseMDError("restart checkpoint RNG state is invalid")
        try:
            rng.bit_generator.state = rng_state
        except Exception as exc:
            raise AseMDError(f"restart checkpoint RNG state cannot be restored: {exc}") from exc

    if ensemble == "nvt-langevin":
        dyn = Langevin(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_k,
            friction=float(nvt_friction) / units.fs,
            rng=rng,
            fixcm=False,
        )
        stochastic_scope = "velocity-initialization-and-langevin"
    else:
        try:
            from ase.md.nose_hoover_chain import IsotropicMTKNPT
        except ImportError as exc:
            raise AseMDError(
                "npt-isotropic-mtk requires an ASE version providing IsotropicMTKNPT"
            ) from exc
        dyn = IsotropicMTKNPT(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_k,
            pressure_au=float(npt_pressure) * units.GPa,
            tdamp=float(npt_tdamp) * units.fs,
            pdamp=float(npt_pdamp) * units.fs,
            tchain=NPT_TCHAIN,
            pchain=NPT_PCHAIN,
            tloop=NPT_TLOOP,
            ploop=NPT_PLOOP,
        )
        stochastic_scope = "velocity-initialization-only"
        if checkpoint is not None:
            _restore_npt_integrator(dyn, checkpoint, np)
    dyn.nsteps = start_step

    trajectory_path = output_dir / "trajectory.traj"
    trajectory_index_path = output_dir / "trajectory-index.json"
    thermo_path = output_dir / "thermo.csv"
    checkpoint_path = output_dir / CHECKPOINT_FILENAME
    final_path = output_dir / "final.extxyz"
    result_path = output_dir / "md-result.json"
    trajectory = Trajectory(str(trajectory_path), "w", atoms)
    thermo_stream = thermo_path.open("x", encoding="utf-8", newline="", buffering=1)
    thermo_writer = csv.writer(thermo_stream)
    thermo_writer.writerow(_thermo_header(ensemble))
    frame_steps: list[int] = []
    thermo_steps: list[int] = []

    def write_index() -> None:
        _write_json_atomic(
            trajectory_index_path,
            {
                "schema_version": 1,
                "segment_start_step": start_step,
                "segment_end_step": int(dyn.get_number_of_steps()),
                "steps": frame_steps,
                "time_fs": [step * timestep_fs for step in frame_steps],
            },
        )

    def write_frame() -> None:
        step = int(dyn.get_number_of_steps())
        trajectory.write(atoms)
        frame_steps.append(step)
        write_index()

    def write_thermo() -> None:
        step = int(dyn.get_number_of_steps())
        epot = float(atoms.get_potential_energy())
        ekin = float(atoms.get_kinetic_energy())
        temp = float(atoms.get_temperature())
        volume = float(atoms.get_volume()) if atoms.cell.rank == 3 else 0.0
        row: list[float | int] = [
            step,
            step * timestep_fs,
            temp,
            epot,
            ekin,
            epot + ekin,
            volume,
        ]
        values = [temp, epot, ekin, epot + ekin, volume]
        if ensemble == "npt-isotropic-mtk":
            stress = atoms.get_stress(voigt=False, include_ideal_gas=True)
            pressure = -sum(float(stress[index, index]) for index in range(3)) / 3.0
            pressure_gpa_value = pressure / units.GPa
            lengths = [float(value) for value in atoms.cell.lengths()]
            row.extend([pressure_gpa_value, *lengths])
            values.extend([pressure_gpa_value, *lengths])
            if volume <= 0 or any(value <= 0 for value in lengths):
                raise AseMDError(f"non-positive NPT cell metric at MD step {step}")
        if any(not math.isfinite(float(value)) for value in values):
            raise AseMDError(f"non-finite thermodynamic value at MD step {step}")
        thermo_writer.writerow(row)
        thermo_stream.flush()
        thermo_steps.append(step)

    def write_checkpoint() -> None:
        if checkpoint_interval is None:
            return
        _write_json_atomic(
            checkpoint_path,
            _checkpoint_payload(
                dyn=dyn,
                atoms=atoms,
                rng=rng,
                base_identity=base_checkpoint_identity,
                ase_version=ase_version,
                ensemble=ensemble,
            ),
        )

    try:
        write_frame()
        write_thermo()
        write_checkpoint()
        dyn.attach(write_frame, interval=trajectory_interval)
        dyn.attach(write_thermo, interval=thermo_interval)
        if checkpoint_interval is not None:
            dyn.attach(write_checkpoint, interval=checkpoint_interval)
        remaining = steps - start_step
        dyn.run(remaining)
        if not frame_steps or frame_steps[-1] != steps:
            write_frame()
        if not thermo_steps or thermo_steps[-1] != steps:
            write_thermo()
        if checkpoint_interval is not None:
            write_checkpoint()
    finally:
        trajectory.close()
        thermo_stream.close()

    write(
        str(final_path),
        _final_structure_copy(atoms, input_constraints),
        format="extxyz",
    )
    expected_frame_steps = _segment_steps(start_step, steps, trajectory_interval)
    expected_thermo_steps = _segment_steps(start_step, steps, thermo_interval)
    if frame_steps != expected_frame_steps:
        raise AseMDError("trajectory callback schedule differs from the deterministic segment schedule")
    if thermo_steps != expected_thermo_steps:
        raise AseMDError("thermo callback schedule differs from the deterministic segment schedule")

    artifacts = []
    artifact_pairs = [
        ("trajectory", trajectory_path),
        ("trajectory-index", trajectory_index_path),
        ("thermo", thermo_path),
        ("final-structure", final_path),
    ]
    if checkpoint_interval is not None:
        artifact_pairs.append(("checkpoint", checkpoint_path))
    for name, path in artifact_pairs:
        artifacts.append(
            {
                "name": name,
                "path": path.name,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "plugin_id": "ase-md",
        "status": "OK",
        "calculator": calculator,
        "calculator_version": framework_version(calculator),
        "ase_version": ase_version,
        "ensemble": ensemble,
        "model": {"id": model_id, "fingerprint": model_fingerprint},
        "structure_fingerprint": structure_fingerprint,
        "temperature_K": temperature_k,
        "timestep_fs": timestep_fs,
        "steps_requested": steps,
        "steps_completed": int(dyn.get_number_of_steps()),
        "segment_start_step": start_step,
        "segment_end_step": int(dyn.get_number_of_steps()),
        "trajectory_interval": trajectory_interval,
        "thermo_interval": thermo_interval,
        "trajectory_frames": len(frame_steps),
        "thermo_records": len(thermo_steps),
        "seed": seed,
        "stochastic_scope": stochastic_scope,
        "device": device,
        "default_dtype": default_dtype,
        "fix_com": bool(fix_com),
        "restart": {
            "resumed": checkpoint is not None,
            "checkpoint_sha256": restart_sha256,
            "checkpoint_start_step": start_step if checkpoint is not None else None,
            "checkpoint_interval": checkpoint_interval,
        },
        "artifacts": artifacts,
    }
    if checkpoint_interval is not None:
        payload["checkpoint"] = {
            "path": CHECKPOINT_FILENAME,
            "sha256": _sha256(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "completed_steps": int(dyn.get_number_of_steps()),
        }
    if ensemble == "nvt-langevin":
        payload["friction_per_fs"] = nvt_friction
    else:
        payload.update(
            {
                "pressure_GPa": npt_pressure,
                "thermostat_damping_fs": npt_tdamp,
                "barostat_damping_fs": npt_pdamp,
                "initial_pressure_GPa": initial_pressure_gpa,
                "thermostat_chain_length": NPT_TCHAIN,
                "barostat_chain_length": NPT_PCHAIN,
                "thermostat_substeps": NPT_TLOOP,
                "barostat_substeps": NPT_PLOOP,
            }
        )
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or resume one explicit-model ASE MD trajectory")
    parser.add_argument("--structure", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calculator", choices=CALCULATORS, required=True)
    parser.add_argument("--ensemble", choices=ENSEMBLES, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument("--structure-fingerprint", required=True)
    parser.add_argument("--temperature-k", type=float, required=True)
    parser.add_argument("--timestep-fs", type=float, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--trajectory-interval", type=int, required=True)
    parser.add_argument("--thermo-interval", type=int, required=True)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--restart-checkpoint")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=DEVICES, required=True)
    parser.add_argument("--default-dtype", choices=DTYPES, required=True)
    parser.add_argument("--friction-per-fs", type=float)
    parser.add_argument("--pressure-gpa", type=float)
    parser.add_argument("--thermostat-damping-fs", type=float)
    parser.add_argument("--barostat-damping-fs", type=float)
    parser.add_argument("--fix-com", action="store_true")
    parser.add_argument("--input-format")
    parser.add_argument("--input-index", default="-1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_md(
        structure=Path(args.structure),
        model=Path(args.model),
        output_dir=Path(args.output_dir),
        calculator=args.calculator,
        ensemble=args.ensemble,
        model_id=args.model_id,
        model_fingerprint=args.model_fingerprint,
        structure_fingerprint=args.structure_fingerprint,
        temperature_k=args.temperature_k,
        timestep_fs=args.timestep_fs,
        steps=args.steps,
        trajectory_interval=args.trajectory_interval,
        thermo_interval=args.thermo_interval,
        checkpoint_interval=args.checkpoint_interval,
        restart_checkpoint=Path(args.restart_checkpoint) if args.restart_checkpoint else None,
        seed=args.seed,
        device=args.device,
        default_dtype=args.default_dtype,
        friction_per_fs=args.friction_per_fs,
        fix_com=args.fix_com,
        pressure_gpa=args.pressure_gpa,
        thermostat_damping_fs=args.thermostat_damping_fs,
        barostat_damping_fs=args.barostat_damping_fs,
        input_format=args.input_format,
        input_index=args.input_index,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
