"""ASE molecular dynamics runner for explicit MLIP model artifacts.

This module owns the framework-neutral MD loop. It never downloads a model and
never shells out. Site-specific model lookup is handled by the cluster wrapper;
this runner receives an already resolved, fingerprinted model path and attaches
one of the supported ASE calculators.

Version 0.2 supports fresh single-temperature NVT Langevin and isotropic MTK NPT
trajectories. Restart/resume, replicas, temperature sweeps and transport analysis
remain separate contracts rather than hidden behavior in this runner.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any

CALCULATORS = ("deepmd", "m3gnet", "chgnet", "mace")
ENSEMBLES = ("nvt-langevin", "npt-isotropic-mtk")
DEVICES = ("cpu", "cuda")
DTYPES = ("float32", "float64")
RESULT_SCHEMA_VERSION = 1
NPT_TCHAIN = 3
NPT_PCHAIN = 3
NPT_TLOOP = 1
NPT_PLOOP = 1


class AseMDError(RuntimeError):
    pass


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
            model_path=str(model), device=device, default_dtype=default_dtype
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
    input_format: str | None = None,
    input_index: str = "-1",
) -> dict[str, Any]:
    """Run one fresh NVT or isotropic NPT trajectory."""
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
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise AseMDError("steps must be a positive integer")
    for name, value in (
        ("trajectory_interval", trajectory_interval),
        ("thermo_interval", thermo_interval),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise AseMDError(f"{name} must be a positive integer")
        if value > steps:
            raise AseMDError(f"{name} cannot exceed steps")
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
    output_dir = output_dir.expanduser().resolve()
    if not structure.is_file():
        raise AseMDError(f"structure is not a regular file: {structure}")
    if calculator == "m3gnet":
        if not model.is_dir():
            raise AseMDError("M3GNet/MatGL model must be a local model directory")
    elif not model.is_file():
        raise AseMDError(f"{calculator} model must be a local model file")
    if output_dir.exists():
        raise AseMDError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    import numpy as np
    from ase import units
    from ase.constraints import FixCom
    from ase.io import read, write
    from ase.io.trajectory import Trajectory
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    atoms = read(str(structure), format=input_format, index=input_index)
    if not hasattr(atoms, "get_positions"):
        raise AseMDError("structure selection did not produce exactly one ASE Atoms object")
    if len(atoms) == 0:
        raise AseMDError("structure contains no atoms")
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

    atoms.calc = _build_calculator(calculator, model, device, default_dtype)
    initial_pressure_gpa: float | None = None
    if ensemble == "npt-isotropic-mtk":
        initial_pressure_gpa = _npt_stress_probe(atoms, units)

    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k, rng=rng)

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

    trajectory_path = output_dir / "trajectory.traj"
    trajectory_index_path = output_dir / "trajectory-index.json"
    thermo_path = output_dir / "thermo.csv"
    final_path = output_dir / "final.extxyz"
    result_path = output_dir / "md-result.json"
    trajectory = Trajectory(str(trajectory_path), "w", atoms)
    thermo_stream = thermo_path.open("x", encoding="utf-8", newline="", buffering=1)
    thermo_writer = csv.writer(thermo_stream)
    thermo_writer.writerow(_thermo_header(ensemble))
    frame_steps: list[int] = []
    thermo_steps: list[int] = []

    def write_frame() -> None:
        step = int(dyn.get_number_of_steps())
        trajectory.write(atoms)
        frame_steps.append(step)

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
        thermo_steps.append(step)

    try:
        write_frame()
        write_thermo()
        dyn.attach(write_frame, interval=trajectory_interval)
        dyn.attach(write_thermo, interval=thermo_interval)
        dyn.run(steps)
        if not frame_steps or frame_steps[-1] != steps:
            write_frame()
        if not thermo_steps or thermo_steps[-1] != steps:
            write_thermo()
    finally:
        trajectory.close()
        thermo_stream.close()

    write(str(final_path), atoms, format="extxyz")
    trajectory_index_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "steps": frame_steps,
                "time_fs": [step * timestep_fs for step in frame_steps],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    expected_frames = 1 + steps // trajectory_interval + (
        1 if steps % trajectory_interval else 0
    )
    expected_thermo = 1 + steps // thermo_interval + (1 if steps % thermo_interval else 0)
    if len(frame_steps) != expected_frames:
        raise AseMDError("trajectory callback count differs from the deterministic schedule")
    if len(thermo_steps) != expected_thermo:
        raise AseMDError("thermo callback count differs from the deterministic schedule")

    artifacts = []
    for name, path in (
        ("trajectory", trajectory_path),
        ("trajectory-index", trajectory_index_path),
        ("thermo", thermo_path),
        ("final-structure", final_path),
    ):
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
        "ase_version": _version("ase"),
        "ensemble": ensemble,
        "model": {"id": model_id, "fingerprint": model_fingerprint},
        "structure_fingerprint": structure_fingerprint,
        "temperature_K": temperature_k,
        "timestep_fs": timestep_fs,
        "steps_requested": steps,
        "steps_completed": int(dyn.get_number_of_steps()),
        "trajectory_interval": trajectory_interval,
        "thermo_interval": thermo_interval,
        "trajectory_frames": len(frame_steps),
        "thermo_records": len(thermo_steps),
        "seed": seed,
        "stochastic_scope": stochastic_scope,
        "device": device,
        "default_dtype": default_dtype,
        "fix_com": bool(fix_com),
        "artifacts": artifacts,
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
    parser = argparse.ArgumentParser(description="Run one explicit-model ASE MD trajectory")
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
