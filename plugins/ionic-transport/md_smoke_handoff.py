"""Bounded handoff from the historical ASE-MD source to transport analysis.

This plugin-local executable owns no MD, MSD, diffusion, conductivity, or
Arrhenius numerical algorithm.  It configures and calls the reviewed
``ase_md_only_multi_calc.py`` per-temperature function, then invokes the
reviewed ``ionic_conductivity.py`` CLI with an argv list and ``shell=False``.
The strict caps make this path suitable only for integration smoke testing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import types
from pathlib import Path
from typing import Any, Mapping, Sequence


PLUGIN_ID = "ionic-transport"
REQUIRED_RESULTS = (
    "diffusion_results_by_temperature.csv",
    "arrhenius_summary.json",
    "postprocess_failures.json",
)
CALCULATORS = {"mace", "chgnet", "m3gnet", "matgl", "emt", "lj"}
DEVICES = {"cpu", "cuda", "mps"}
DTYPES = {"float32", "float64"}
MAX_TEMPERATURES = 4
MAX_EQUILIBRATION_PS = 0.1
MAX_PRODUCTION_PS = 0.2
MAX_TOTAL_STEPS = 5000


class IntegrationSmokeError(ValueError):
    """Raised when the bounded historical-source handoff is unsafe or incomplete."""


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _artifact(path: Path, role: str, attempt_dir: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    locator = resolved.name
    relative = False
    locator_kind = "local-basename"
    portable = False
    if attempt_dir is not None and _is_within(resolved, attempt_dir):
        locator = resolved.relative_to(attempt_dir).as_posix()
        relative = True
        locator_kind = "attempt-relative"
        portable = True
    return {
        "role": role,
        "path": locator,
        "path_is_attempt_relative": relative,
        "locator_kind": locator_kind,
        "portable": portable,
        "absolute_path_recorded": False,
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _load_md_source(path: Path) -> types.ModuleType:
    """Compile a source in memory so no bytecode is written beside it."""

    payload = path.read_bytes()
    module = types.ModuleType("_mlipflow_historical_ase_md")
    module.__file__ = str(path)
    module.__package__ = ""
    exec(compile(payload, str(path), "exec"), module.__dict__)
    if not callable(getattr(module, "run_single_temperature_md", None)):
        raise IntegrationSmokeError("MD source must expose run_single_temperature_md")
    return module


def _exclusive_text(path: Path, value: str) -> None:
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(value)
    except FileExistsError as exc:
        raise IntegrationSmokeError("refusing to overwrite output: %s" % path) from exc


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise IntegrationSmokeError("refusing to overwrite manifest: %s" % path) from exc


def _validate_bounds(
    *,
    temperatures_k: Sequence[float],
    seed: int,
    timestep_fs: float,
    npt_time_ps: float,
    nvt_equil_time_ps: float,
    production_time_ps: float,
    trajectory_interval_steps: int,
    calculator: str,
    model: str,
    device: str,
    default_dtype: str,
    analysis_arguments: Sequence[str],
) -> None:
    if not 2 <= len(temperatures_k) <= MAX_TEMPERATURES:
        raise IntegrationSmokeError("Arrhenius smoke requires 2-%d temperatures" % MAX_TEMPERATURES)
    normalized = [float(value) for value in temperatures_k]
    if any(not math.isfinite(value) or value <= 0 or not value.is_integer() for value in normalized):
        raise IntegrationSmokeError("temperatures must be positive integer-K values")
    if len(set(normalized)) != len(normalized):
        raise IntegrationSmokeError("temperatures must be unique")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise IntegrationSmokeError("seed must be a non-negative integer")
    if not math.isfinite(timestep_fs) or not 0 < timestep_fs <= 2.0:
        raise IntegrationSmokeError("timestep_fs must be in (0, 2]")
    for name, value in (
        ("npt_time_ps", npt_time_ps),
        ("nvt_equil_time_ps", nvt_equil_time_ps),
    ):
        if not math.isfinite(value) or not 0 <= value <= MAX_EQUILIBRATION_PS:
            raise IntegrationSmokeError("%s exceeds the integration-smoke bound" % name)
    if not math.isfinite(production_time_ps) or not 0 < production_time_ps <= MAX_PRODUCTION_PS:
        raise IntegrationSmokeError("production_time_ps exceeds the integration-smoke bound")
    if (
        isinstance(trajectory_interval_steps, bool)
        or not isinstance(trajectory_interval_steps, int)
        or not 1 <= trajectory_interval_steps <= 100
    ):
        raise IntegrationSmokeError("trajectory_interval_steps must be in [1, 100]")
    npt_steps = int(round(npt_time_ps * 1000.0 / timestep_fs))
    nvt_steps = int(round(nvt_equil_time_ps * 1000.0 / timestep_fs))
    production_steps = int(round(production_time_ps * 1000.0 / timestep_fs))
    if production_steps < 3:
        raise IntegrationSmokeError("production must contain at least three MD steps")
    production_frames = int(math.ceil(production_steps / trajectory_interval_steps))
    if production_frames < 3:
        raise IntegrationSmokeError("production must write at least three trajectory frames")
    total_steps = len(normalized) * (npt_steps + nvt_steps + production_steps)
    if total_steps > MAX_TOTAL_STEPS:
        raise IntegrationSmokeError("smoke exceeds %d total MD steps" % MAX_TOTAL_STEPS)
    if calculator not in CALCULATORS:
        raise IntegrationSmokeError("unsupported calculator")
    if calculator in {"mace", "chgnet", "m3gnet", "matgl"} and model == "default":
        raise IntegrationSmokeError(
            "%s requires an explicit local model; implicit default loading is forbidden"
            % calculator
        )
    if calculator in {"emt", "lj"} and model != "default":
        raise IntegrationSmokeError("EMT/LJ must use the explicit default-model marker")
    if device not in DEVICES or default_dtype not in DTYPES:
        raise IntegrationSmokeError("unsupported device or default dtype")
    if not all(
        isinstance(value, str)
        and "\x00" not in value
        and "\n" not in value
        and "\r" not in value
        for value in analysis_arguments
    ):
        raise IntegrationSmokeError("analysis arguments must be plain strings")
    if any(value in {"--input", "--output"} for value in analysis_arguments):
        raise IntegrationSmokeError("analysis arguments cannot override confined input/output paths")


def run_handoff(
    *,
    attempt_dir: Path,
    md_script: Path,
    analysis_script: Path,
    structure: Path,
    model: str,
    calculator: str,
    temperatures_k: Sequence[float],
    seed: int,
    timestep_fs: float,
    npt_time_ps: float,
    nvt_equil_time_ps: float,
    production_time_ps: float,
    trajectory_interval_steps: int,
    device: str,
    default_dtype: str,
    input_format: str | None,
    input_index: str,
    specie: str,
    md_output: Path,
    analysis_output: Path,
    manifest_path: Path,
    analysis_arguments: Sequence[str],
    python_executable: str,
) -> dict[str, Any]:
    _validate_bounds(
        temperatures_k=temperatures_k,
        seed=seed,
        timestep_fs=timestep_fs,
        npt_time_ps=npt_time_ps,
        nvt_equil_time_ps=nvt_equil_time_ps,
        production_time_ps=production_time_ps,
        trajectory_interval_steps=trajectory_interval_steps,
        calculator=calculator,
        model=model,
        device=device,
        default_dtype=default_dtype,
        analysis_arguments=analysis_arguments,
    )
    attempt_dir = attempt_dir.resolve()
    md_script = md_script.resolve()
    analysis_script = analysis_script.resolve()
    structure = structure.resolve()
    md_output = md_output.resolve()
    analysis_output = analysis_output.resolve()
    manifest_path = manifest_path.resolve()
    if not attempt_dir.is_dir():
        raise IntegrationSmokeError("attempt_dir must already exist")
    for path, label in (
        (md_script, "MD source"),
        (analysis_script, "analysis source"),
        (structure, "structure"),
    ):
        if not path.is_file():
            raise IntegrationSmokeError("%s is not a regular file: %s" % (label, path))
    for path, label in (
        (md_output, "MD output"),
        (analysis_output, "analysis output"),
        (manifest_path, "integration manifest"),
    ):
        if not _is_within(path, attempt_dir):
            raise IntegrationSmokeError("%s must stay inside attempt_dir" % label)
        if path.exists():
            raise IntegrationSmokeError("%s already exists; smoke runs never overwrite" % label)
    output_targets = (md_output, analysis_output, manifest_path)
    for index, left in enumerate(output_targets):
        for right in output_targets[index + 1 :]:
            if _is_within(left, right) or _is_within(right, left):
                raise IntegrationSmokeError("MD, analysis, and manifest outputs cannot overlap")

    model_path: Path | None = None
    if model != "default":
        model_path = Path(model).expanduser().resolve()
        if not model_path.is_file():
            raise IntegrationSmokeError("explicit model is not a regular file: %s" % model_path)
        model_value = str(model_path)
    else:
        model_value = "default"
    immutable_paths = [md_script, analysis_script, structure]
    if model_path is not None:
        immutable_paths.append(model_path)
    before = {path: _sha256(path) for path in immutable_paths}

    source_module = _load_md_source(md_script)
    production_steps = int(round(production_time_ps * 1000.0 / timestep_fs))
    npt_steps = int(round(npt_time_ps * 1000.0 / timestep_fs))
    nvt_steps = int(round(nvt_equil_time_ps * 1000.0 / timestep_fs))
    overrides = {
        "OUTPUT_ROOT": md_output,
        "SPECIE": specie,
        "TIMESTEP_FS": timestep_fs,
        "PRODUCTION_TIME_PS": production_time_ps,
        "PROD_STEPS": production_steps,
        "DISCARD_INITIAL_PS": 0.0,
        "TRAJ_INTERVAL": trajectory_interval_steps,
        "RUN_NPT_EQUIL": npt_steps > 0,
        "NPT_TIME_PS": npt_time_ps,
        "NPT_STEPS": npt_steps,
        "RUN_NVT_EQUIL_AFTER_NPT": nvt_steps > 0,
        "EQUIL_TIME_PS": nvt_equil_time_ps,
        "EQUIL_STEPS": nvt_steps,
        "WRITE_XYZ_DUMP": False,
        "DEVICE": device,
        "DEFAULT_DTYPE": default_dtype,
        "RUN_MD": True,
    }
    for name, value in overrides.items():
        setattr(source_module, name, value)

    trajectories: list[Path] = []
    per_temperature: list[dict[str, Any]] = []
    for index, temperature in enumerate(temperatures_k):
        run_seed = seed + index
        random.seed(run_seed)
        source_numpy = getattr(source_module, "np", None)
        if source_numpy is not None and hasattr(getattr(source_numpy, "random", None), "seed"):
            source_numpy.random.seed(run_seed)
        metadata = source_module.run_single_temperature_md(
            temperature_k=float(temperature),
            structure_path=str(structure),
            calculator_type=calculator,
            model_path=model_value,
            input_format=input_format,
            index=input_index,
        )
        if not isinstance(metadata, Mapping):
            raise IntegrationSmokeError("historical MD source returned non-mapping metadata")
        raw_trajectory = metadata.get("traj_path")
        trajectory = (
            Path(str(raw_trajectory)).expanduser()
            if raw_trajectory is not None
            else md_output / ("T%d" % int(float(temperature))) / "production.traj"
        )
        trajectory = (
            (attempt_dir / trajectory).resolve() if not trajectory.is_absolute() else trajectory.resolve()
        )
        if not _is_within(trajectory, md_output) or not trajectory.is_file():
            raise IntegrationSmokeError(
                "historical MD source omitted confined production.traj for %.17g K"
                % float(temperature)
            )
        metadata_path = trajectory.parent / "metadata.json"
        if not metadata_path.is_file():
            raise IntegrationSmokeError("historical MD source omitted metadata.json")
        trajectories.append(trajectory)
        per_temperature.append(
            {
                "temperature_K": float(temperature),
                "velocity_seed": run_seed,
                "trajectory": trajectory.relative_to(attempt_dir).as_posix(),
                "metadata": metadata_path.relative_to(attempt_dir).as_posix(),
            }
        )

    analysis_argv = [
        python_executable,
        str(analysis_script),
        "--input",
        str(md_output),
        "--output",
        str(analysis_output),
        *analysis_arguments,
    ]
    completed = subprocess.run(
        analysis_argv,
        cwd=str(attempt_dir),
        shell=False,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[-4000:]
        raise IntegrationSmokeError(
            "ionic_conductivity.py returned %d: %s" % (completed.returncode, detail)
        )
    if not analysis_output.is_dir():
        raise IntegrationSmokeError("analysis source did not create its output directory")
    stdout_path = analysis_output / "analysis.stdout.log"
    stderr_path = analysis_output / "analysis.stderr.log"
    _exclusive_text(stdout_path, completed.stdout)
    _exclusive_text(stderr_path, completed.stderr)

    result_paths = [analysis_output / name for name in REQUIRED_RESULTS]
    missing = [path.name for path in result_paths if not path.is_file()]
    if missing:
        raise IntegrationSmokeError("analysis source omitted required output(s): %s" % missing)
    try:
        failures = json.loads((analysis_output / "postprocess_failures.json").read_text())
        arrhenius = json.loads((analysis_output / "arrhenius_summary.json").read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IntegrationSmokeError("analysis JSON output is unreadable: %s" % exc) from exc
    if failures != []:
        raise IntegrationSmokeError("integration smoke requires zero post-processing failures")
    if not isinstance(arrhenius, Mapping) or arrhenius.get("arrhenius_fit_skipped") is True:
        raise IntegrationSmokeError("integration smoke did not complete an Arrhenius fit")
    if not isinstance(arrhenius.get("single"), Mapping):
        raise IntegrationSmokeError("integration smoke requires a single-line Arrhenius result")
    after = {path: _sha256(path) for path in immutable_paths}
    if before != after:
        raise IntegrationSmokeError("a source/input artifact changed during the smoke run")

    md_files = sorted(path for path in md_output.rglob("*") if path.is_file())
    analysis_files = sorted(path for path in analysis_output.rglob("*") if path.is_file())
    for path in [*md_files, *analysis_files]:
        if not _is_within(path.resolve(), attempt_dir):
            raise IntegrationSmokeError("generated artifact escapes attempt_dir: %s" % path)
    source_artifacts = [
        _artifact(md_script, "historical-ase-md-source"),
        _artifact(analysis_script, "historical-transport-analysis-source"),
        _artifact(Path(__file__), "mlipflow-handoff-wrapper"),
    ]
    input_artifacts = [_artifact(structure, "structure")]
    if model_path is not None:
        input_artifacts.append(_artifact(model_path, "model"))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "plugin_id": PLUGIN_ID,
        "operation": "md-smoke-and-analyze",
        "status": "OK",
        "scientific_use": "integration-smoke-only",
        "scientific_claim": (
            "The short trajectory verifies orchestration only and is not a manuscript, "
            "production, or scientifically converged transport result."
        ),
        "source_artifacts": source_artifacts,
        "input_artifacts": input_artifacts,
        "model": (
            {
                "kind": "explicit-file",
                "path": model_path.name,
                "locator_kind": "local-basename",
                "portable": False,
                "absolute_path_recorded": False,
                "sha256": _sha256(model_path),
            }
            if model_path is not None
            else {
                "kind": "calculator-default",
                "value": "default",
                "portable": True,
                "absolute_path_recorded": False,
                "sha256": None,
            }
        ),
        "configuration": {
            "calculator": calculator,
            "temperatures_K": [float(value) for value in temperatures_k],
            "base_seed": seed,
            "temperature_seed_policy": "base-seed-plus-temperature-index",
            "timestep_fs": timestep_fs,
            "npt_time_ps": npt_time_ps,
            "nvt_equil_time_ps": nvt_equil_time_ps,
            "production_time_ps": production_time_ps,
            "trajectory_interval_steps": trajectory_interval_steps,
            "device": device,
            "default_dtype": default_dtype,
            "specie": specie,
            "input_format": input_format,
            "input_index": input_index,
        },
        "randomness": {
            "velocity_seed_applied_through_numpy_global_rng": True,
            "framework_rngs_fully_controlled": False,
            "gpu_bitwise_determinism_guaranteed": False,
            "limitation": (
                "The historical source exposes no seed argument. The handoff seeds its NumPy "
                "velocity RNG before each temperature; calculator/framework RNGs and GPU kernels "
                "are not guaranteed bitwise deterministic."
            ),
        },
        "execution": {
            "md_source_called_in_process": True,
            "analysis_argv_template": [
                "<python-executable>",
                "<analysis-script>",
                "--input",
                "<attempt-dir>/%s" % md_output.relative_to(attempt_dir).as_posix(),
                "--output",
                "<attempt-dir>/%s" % analysis_output.relative_to(attempt_dir).as_posix(),
                *analysis_arguments,
            ],
            "analysis_argv_sha256": "sha256:"
            + hashlib.sha256(
                json.dumps(analysis_argv, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "absolute_paths_recorded": False,
            "analysis_shell": False,
            "analysis_returncode": completed.returncode,
        },
        "temperature_runs": per_temperature,
        "trajectory_artifacts": [
            _artifact(
                path,
                "production-trajectory:%s" % path.relative_to(attempt_dir).as_posix(),
                attempt_dir,
            )
            for path in trajectories
        ],
        "md_artifacts": [
            _artifact(
                path,
                "md-output:%s" % path.relative_to(attempt_dir).as_posix(),
                attempt_dir,
            )
            for path in md_files
        ],
        "result_artifacts": [
            _artifact(
                path,
                "transport-result:%s" % path.relative_to(attempt_dir).as_posix(),
                attempt_dir,
            )
            for path in analysis_files
        ],
        "boundaries": {
            "production_parameters_accepted": False,
            "scientific_parity_claimed": False,
            "model_quality_validated": False,
            "network_access_authorized_by_handoff": False,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _exclusive_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--md-script", type=Path, required=True)
    parser.add_argument("--analysis-script", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calculator", choices=sorted(CALCULATORS), required=True)
    parser.add_argument("--temperatures-json", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timestep-fs", type=float, required=True)
    parser.add_argument("--npt-time-ps", type=float, required=True)
    parser.add_argument("--nvt-equil-time-ps", type=float, required=True)
    parser.add_argument("--production-time-ps", type=float, required=True)
    parser.add_argument("--trajectory-interval-steps", type=int, required=True)
    parser.add_argument("--device", choices=sorted(DEVICES), required=True)
    parser.add_argument("--default-dtype", choices=sorted(DTYPES), required=True)
    parser.add_argument("--input-format")
    parser.add_argument("--input-index", required=True)
    parser.add_argument("--specie", required=True)
    parser.add_argument("--md-output", type=Path, required=True)
    parser.add_argument("--analysis-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--analysis-arguments-json", required=True)
    parser.add_argument("--python-executable", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        temperatures = json.loads(args.temperatures_json)
        analysis_arguments = json.loads(args.analysis_arguments_json)
        if not isinstance(temperatures, list):
            raise IntegrationSmokeError("temperatures-json must encode a list")
        if not isinstance(analysis_arguments, list):
            raise IntegrationSmokeError("analysis-arguments-json must encode a list")
        run_handoff(
            attempt_dir=args.attempt_dir,
            md_script=args.md_script,
            analysis_script=args.analysis_script,
            structure=args.structure,
            model=args.model,
            calculator=args.calculator,
            temperatures_k=temperatures,
            seed=args.seed,
            timestep_fs=args.timestep_fs,
            npt_time_ps=args.npt_time_ps,
            nvt_equil_time_ps=args.nvt_equil_time_ps,
            production_time_ps=args.production_time_ps,
            trajectory_interval_steps=args.trajectory_interval_steps,
            device=args.device,
            default_dtype=args.default_dtype,
            input_format=args.input_format,
            input_index=args.input_index,
            specie=args.specie,
            md_output=args.md_output,
            analysis_output=args.analysis_output,
            manifest_path=args.manifest,
            analysis_arguments=analysis_arguments,
            python_executable=args.python_executable,
        )
    except (IntegrationSmokeError, OSError, ValueError, ImportError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
