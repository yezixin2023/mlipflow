"""Normalize LASP/SSW archives and optionally run a user-supplied LASP binary.

This module is a deliberately small local wrapper.  It does not implement SSW,
ship LASP, submit scheduler jobs, or infer undocumented historical parameters.
It stages explicit inputs into a fresh directory, invokes argv with
``shell=False`` for the execute operation, and writes a portable result
contract consumed by :mod:`plugins.pes-sampling.adapter`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


PLUGIN_ID = "pes-sampling"
SCHEMA_VERSION = 1
UNKNOWN = "HISTORICAL_PARAMETER_UNKNOWN"
ARC_HEADER = b"!BIOSYM archive 2\nPBC=ON\n"
MAX_ARC_BYTES = 512 * 1024 * 1024
MAX_FRAME_BYTES = 8 * 1024 * 1024
MAX_LASP_INPUT_BYTES = 1024 * 1024
MAX_LOG_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_FRAMES = 10000
INCOMPLETE_MARKER = "INCOMPLETE.json"
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
RESERVED_STAGE_NAMES = {
    "input.arc",
    "lasp.in",
    "allstr.arc",
    "best.arc",
    "md.arc",
    "sampling-result.json",
}


class ContractError(ValueError):
    """Raised when an explicit LASP/SSW contract is incomplete or unsafe."""


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    for candidate in (absolute, *absolute.parents):
        if candidate.is_symlink():
            return True
    return False


def _ordinary_file(path: Path, field: str) -> Path:
    if _has_symlink_component(path) or not path.is_file():
        raise ContractError(f"{field} must be an existing ordinary non-symlink file")
    return path.resolve()


def _ordinary_directory(path: Path, field: str) -> Path:
    if _has_symlink_component(path) or not path.is_dir():
        raise ContractError(f"{field} must be an existing ordinary non-symlink directory")
    return path.resolve()


def _portable_source_id(value: str) -> str:
    if not value or "\x00" in value or "\n" in value or "\r" in value:
        raise ContractError("historical_source_id must be a non-empty single-line string")
    normalized = value.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized):
        raise ContractError("historical_source_id must not expose an absolute local path")
    if "://" not in normalized and ".." in Path(normalized).parts:
        raise ContractError("historical_source_id must not traverse parent directories")
    return normalized


def _fresh_output(path: Path) -> Path:
    if path.exists() or _has_symlink_component(path):
        raise ContractError("output_dir must not already exist")
    path.mkdir(parents=True, exist_ok=False)
    resolved = path.resolve()
    _json_write(
        resolved / INCOMPLETE_MARKER,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "INCOMPLETE",
            "reason": "removed only after all normalized artifacts and sampling-result.json are written",
        },
    )
    return resolved


def _json_write(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_text(payload, encoding="utf-8")


def _artifact(path: Path, base: Path, role: str, media_type: str) -> dict[str, Any]:
    resolved = _ordinary_file(path, role)
    try:
        portable = resolved.relative_to(base.resolve()).as_posix()
    except ValueError as exc:
        raise ContractError(f"artifact escapes output_dir: {resolved}") from exc
    return {
        "role": role,
        "path": portable,
        "media_type": media_type,
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _parse_lasp_input(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_LASP_INPUT_BYTES:
        raise ContractError(f"lasp.in exceeds {MAX_LASP_INPUT_BYTES} bytes")
    parameters: dict[str, Any] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(("#", "!")):
            continue
        tokens = stripped.split()
        if len(tokens) < 2:
            raise ContractError(f"lasp.in line {line_number} has no explicit value")
        key = tokens[0]
        value: Any = tokens[1] if len(tokens) == 2 else tokens[1:]
        if key in parameters:
            previous = parameters[key]
            parameters[key] = previous + [value] if isinstance(previous, list) else [previous, value]
        else:
            parameters[key] = value
    if not parameters:
        raise ContractError("lasp.in contains no parseable parameters")
    return parameters


def _validate_ssw_input(parameters: dict[str, Any]) -> None:
    if str(parameters.get("explore_type", "")).lower() != "ssw":
        raise ContractError("lasp.in must explicitly declare explore_type ssw")
    steps = parameters.get("SSW.SSWsteps")
    try:
        numeric_steps = int(str(steps))
    except (TypeError, ValueError) as exc:
        raise ContractError("lasp.in must declare an integer SSW.SSWsteps") from exc
    if numeric_steps < 1:
        raise ContractError("lasp-ssw operations require SSW.SSWsteps >= 1")


def _parse_energy_record(line: bytes, path: Path, frame_index: int) -> float:
    try:
        tokens = line.decode("ascii").split()
    except UnicodeDecodeError as exc:
        raise ContractError(f"{path.name} frame {frame_index} Energy record is not ASCII") from exc
    if tokens[:1] != ["Energy"] or len(tokens) not in {3, 4, 5}:
        raise ContractError(f"{path.name} frame {frame_index} has an unsupported Energy record")
    try:
        int(tokens[1])
    except ValueError as exc:
        raise ContractError(f"{path.name} frame {frame_index} Energy index is invalid") from exc
    if len(tokens) == 5 and not re.fullmatch(r"C[0-9]+", tokens[4]):
        raise ContractError(f"{path.name} frame {frame_index} Energy suffix is invalid")
    energy_token = tokens[2] if len(tokens) == 3 else tokens[3]
    try:
        energy = float(energy_token)
    except ValueError as exc:
        raise ContractError(f"{path.name} frame {frame_index} energy is invalid") from exc
    if not math.isfinite(energy):
        raise ContractError(f"{path.name} frame {frame_index} energy is not finite")
    return energy


def _arc_frames(path: Path, max_frames: int) -> list[dict[str, Any]]:
    size = path.stat().st_size
    if size == 0 or size > MAX_ARC_BYTES:
        raise ContractError(f"{path.name} size must be between 1 and {MAX_ARC_BYTES} bytes")
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    starts = [index for index, line in enumerate(lines) if line.lstrip().startswith(b"Energy")]
    if not starts:
        raise ContractError(f"{path.name} contains no LASP Energy frames")
    if len(starts) > max_frames:
        raise ContractError(
            f"{path.name} contains {len(starts)} frames, exceeding max_frames={max_frames}"
        )
    frames: list[dict[str, Any]] = []
    for position, start in enumerate(starts):
        stop = starts[position + 1] if position + 1 < len(starts) else len(lines)
        block_lines = lines[start:stop]
        canonical = ARC_HEADER + b"".join(block_lines)
        if len(canonical) > MAX_FRAME_BYTES:
            raise ContractError(
                f"{path.name} frame {position + 1} exceeds {MAX_FRAME_BYTES} bytes"
            )
        end_count = sum(1 for line in block_lines if line.strip() == b"end")
        if end_count < 2:
            raise ContractError(f"{path.name} frame {position + 1} is truncated")
        energy = _parse_energy_record(block_lines[0], path, position + 1)
        frames.append(
            {
                "frame_index": position + 1,
                "energy_ev": energy,
                "payload": canonical,
                "frame_sha256": _sha256_bytes(canonical),
            }
        )
    return frames


def _structure_id(source_id: str, role: str, frame: dict[str, Any]) -> str:
    identity = "\0".join(
        [source_id, role, str(frame["frame_index"]), str(frame["frame_sha256"])]
    ).encode("utf-8")
    return f"lasp-{role}-{hashlib.sha256(identity).hexdigest()[:20]}"


def _source_file_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "name": path.name,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _select_ssw_frames(
    frames: list[dict[str, Any]],
    source_id: str,
    stride: int,
    energy_max_ev: float | None,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    structures_dir = output_dir / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)
    selected_dir = output_dir / "selected"
    records: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    accepted_order = 0
    for frame in frames:
        structure_path = structures_dir / f"frame-{frame['frame_index']:06d}.arc"
        structure_path.write_bytes(frame["payload"])
        artifacts.append(
            _artifact(
                structure_path,
                output_dir,
                "ssw-generated-structure",
                "chemical/x-biosym-archive",
            )
        )
        accepted = energy_max_ev is None or float(frame["energy_ev"]) <= energy_max_ev
        if accepted:
            accepted_order += 1
        is_selected = accepted and (accepted_order - 1) % stride == 0
        selected_order = len(selected) + 1 if is_selected else None
        record = {
            "structure_id": _structure_id(source_id, "ssw", frame),
            "source_role": "allstr.arc",
            "frame_index": frame["frame_index"],
            "historical_order": frame["frame_index"],
            "energy_ev": frame["energy_ev"],
            "energy_filter_pass": accepted,
            "energy_filter_order": accepted_order if accepted else None,
            "selected": is_selected,
            "selected_order": selected_order,
            "selection_reason": (
                f"energy<={energy_max_ev} then accepted-order stride {stride}"
                if is_selected and energy_max_ev is not None
                else (
                    f"accepted-order stride {stride}"
                    if is_selected
                    else ("energy-filter-rejected" if not accepted else "stride-not-selected")
                )
            ),
            "frame_sha256": frame["frame_sha256"],
            "structure_file": structure_path.relative_to(output_dir).as_posix(),
            "output_file": None,
        }
        if is_selected:
            selected_dir.mkdir(parents=True, exist_ok=True)
            path = selected_dir / f"input-{selected_order:06d}.arc"
            path.write_bytes(frame["payload"])
            record["output_file"] = path.relative_to(output_dir).as_posix()
            selected_record = dict(record)
            selected.append(selected_record)
            artifacts.append(
                _artifact(path, output_dir, "selected-structure", "chemical/x-biosym-archive")
            )
        records.append(record)
    if not selected:
        raise ContractError("selection policy produced no structures")
    return records, selected, artifacts


def _export_role_frames(
    frames: list[dict[str, Any]], source_id: str, role: str, output_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    destination = output_dir / role
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    artifact_role = (
        "aimd-seed-candidate" if role == "aimd-seeds" else "md-sampled-structure"
    )
    for order, frame in enumerate(frames, 1):
        path = destination / f"input-{order:06d}.arc"
        path.write_bytes(frame["payload"])
        record = {
            "structure_id": _structure_id(source_id, role, frame),
            "source_role": "best.arc" if role == "aimd-seeds" else "md.arc",
            "frame_index": frame["frame_index"],
            "historical_order": frame["frame_index"],
            "energy_ev": frame["energy_ev"],
            "frame_sha256": frame["frame_sha256"],
            "output_file": path.relative_to(output_dir).as_posix(),
        }
        records.append(record)
        artifacts.append(
            _artifact(path, output_dir, artifact_role, "chemical/x-biosym-archive")
        )
    return records, artifacts


def _normalize(
    *,
    source_dir: Path,
    lasp_input: Path,
    output_dir: Path,
    operation: str,
    mode: str,
    source_id: str,
    selection_stride: int,
    energy_max_ev: float | None,
    max_frames: int,
    include_best_arc: bool,
    include_md_arc: bool,
    execution_metadata: dict[str, Any],
    extra_artifacts: list[tuple[Path, str, str]],
) -> dict[str, Any]:
    source_dir = _ordinary_directory(source_dir, "source_dir")
    lasp_input = _ordinary_file(lasp_input, "lasp_input")
    allstr = _ordinary_file(source_dir / "allstr.arc", "allstr.arc")
    source_id = _portable_source_id(source_id)
    lasp_parameters = _parse_lasp_input(lasp_input)
    _validate_ssw_input(lasp_parameters)
    frames = _arc_frames(allstr, max_frames)
    records, selected, selected_artifacts = _select_ssw_frames(
        frames, source_id, selection_stride, energy_max_ev, output_dir
    )

    source_files = [
        _source_file_record(lasp_input, "lasp-input"),
        _source_file_record(allstr, "ssw-archive"),
    ]
    artifacts = list(selected_artifacts)
    for path, role, media_type in extra_artifacts:
        if path.is_file() and 0 < path.stat().st_size <= MAX_LOG_ARTIFACT_BYTES:
            artifacts.append(_artifact(path, output_dir, role, media_type))
    structure_manifest = output_dir / "ssw-structures.json"
    _json_write(
        structure_manifest,
        {
            "schema_version": SCHEMA_VERSION,
            "source_id": source_id,
            "source_role": "allstr.arc",
            "structures": records,
        },
    )
    selected_manifest = output_dir / "selected-structures.json"
    _json_write(
        selected_manifest,
        {
            "schema_version": SCHEMA_VERSION,
            "source_id": source_id,
            "selection_policy": {
                "energy_max_ev": energy_max_ev,
                "selection_stride": selection_stride,
                "stride_applies_after_energy_filter": True,
                "preserve_historical_order": True,
            },
            "structures": selected,
        },
    )
    artifacts.extend(
        [
            _artifact(
                structure_manifest,
                output_dir,
                "ssw-structure-manifest",
                "application/json",
            ),
            _artifact(
                selected_manifest,
                output_dir,
                "selected-structure-manifest",
                "application/json",
            ),
        ]
    )

    best_records: list[dict[str, Any]] = []
    if include_best_arc:
        best = _ordinary_file(source_dir / "best.arc", "best.arc")
        source_files.append(_source_file_record(best, "best-archive"))
        best_records, best_artifacts = _export_role_frames(
            _arc_frames(best, max_frames), source_id, "aimd-seeds", output_dir
        )
        best_manifest = output_dir / "aimd-seeds.json"
        _json_write(
            best_manifest,
            {
                "schema_version": SCHEMA_VERSION,
                "source_id": source_id,
                "source_role": "best.arc",
                "structures": best_records,
            },
        )
        artifacts.extend(best_artifacts)
        artifacts.append(
            _artifact(best_manifest, output_dir, "aimd-seed-manifest", "application/json")
        )

    md_records: list[dict[str, Any]] = []
    if include_md_arc:
        md = _ordinary_file(source_dir / "md.arc", "md.arc")
        source_files.append(_source_file_record(md, "md-archive"))
        md_records, md_artifacts = _export_role_frames(
            _arc_frames(md, max_frames), source_id, "md-structures", output_dir
        )
        md_manifest = output_dir / "md-structures.json"
        _json_write(
            md_manifest,
            {
                "schema_version": SCHEMA_VERSION,
                "source_id": source_id,
                "source_role": "md.arc",
                "structures": md_records,
            },
        )
        artifacts.append(
            _artifact(md_manifest, output_dir, "md-structure-manifest", "application/json")
        )
        artifacts.extend(md_artifacts)

    parameters = {
        "lasp_input": lasp_parameters,
        "seed_status": UNKNOWN,
        "selection_stride": selection_stride,
        "energy_max_ev": energy_max_ev,
        "stride_applies_after_energy_filter": True,
        "preserve_historical_order": True,
    }
    metadata_path = output_dir / "lasp-run-metadata.json"
    _json_write(
        metadata_path,
        {
            "schema_version": SCHEMA_VERSION,
            "plugin_id": PLUGIN_ID,
            "operation": operation,
            "mode": mode,
            "source": {"id": source_id, "files": source_files},
            "parameters": parameters,
            "execution": execution_metadata,
            "scientific_boundaries": [
                "The wrapper does not implement stochastic surface walking.",
                "Historical replay verifies parsing and selection only, not LASP numerical parity.",
                "AIMD and MLIP-MD execution are outside this operation.",
            ],
        },
    )
    artifacts.append(
        _artifact(metadata_path, output_dir, "lasp-run-metadata", "application/json")
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "plugin_id": PLUGIN_ID,
        "operation": operation,
        "status": "OK",
        "mode": mode,
        "source_id": source_id,
        "parameters": parameters,
        "counts": {
            "generated_structure_count": len(records),
            "energy_accepted_count": sum(
                1 for record in records if record["energy_filter_pass"]
            ),
            "selected_structure_count": len(selected),
            "aimd_seed_candidate_count": len(best_records),
            "md_structure_count": len(md_records),
        },
        "artifacts": sorted(artifacts, key=lambda item: (item["role"], item["path"])),
        "completion_evidence": {
            "allstr_arc_parsed": True,
            "structure_manifest_validated_on_write": True,
            "selected_structures_nonempty": True,
            "stdout_success_phrase_used": False,
            "incomplete_marker_removed": True,
        },
    }
    _json_write(output_dir / "sampling-result.json", result)
    marker = output_dir / INCOMPLETE_MARKER
    if not marker.is_file() or marker.is_symlink():
        raise ContractError("incomplete marker was unexpectedly removed or replaced")
    marker.unlink()
    return result


def normalize_replay(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed_status != UNKNOWN:
        raise ContractError(f"seed_status must be {UNKNOWN}")
    if args.selection_stride < 1 or not 1 <= args.max_frames <= MAX_FRAMES:
        raise ContractError(
            f"selection_stride must be positive and max_frames must be 1..{MAX_FRAMES}"
        )
    output_dir = _fresh_output(Path(args.output_dir))
    return _normalize(
        source_dir=Path(args.historical_run_dir),
        lasp_input=Path(args.lasp_input),
        output_dir=output_dir,
        operation="lasp-ssw-normalize-replay",
        mode="historical-replay",
        source_id=args.historical_source_id,
        selection_stride=args.selection_stride,
        energy_max_ev=args.energy_max_ev,
        max_frames=args.max_frames,
        include_best_arc=args.include_best_arc,
        include_md_arc=args.include_md_arc,
        execution_metadata={
            "performed": False,
            "external_program": "LASP",
            "reason": "normalized existing LASP/SSW artifacts",
        },
        extra_artifacts=[],
    )


def _parse_auxiliary(values: Iterable[list[str]]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        if len(value) != 2:
            raise ContractError("each auxiliary entry requires a name and a path")
        name, raw_path = value
        if not SAFE_NAME.fullmatch(name):
            raise ContractError("auxiliary names must be safe basenames")
        if name in RESERVED_STAGE_NAMES or name in seen:
            raise ContractError(f"reserved or duplicate auxiliary destination: {name}")
        seen.add(name)
        result.append((name, _ordinary_file(Path(raw_path), f"auxiliary {name}")))
    return result


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed_status != UNKNOWN:
        raise ContractError(f"seed_status must be {UNKNOWN}")
    if args.selection_stride < 1 or not 1 <= args.max_frames <= MAX_FRAMES:
        raise ContractError(
            f"selection_stride must be positive and max_frames must be 1..{MAX_FRAMES}"
        )
    if not args.lasp_version or any(character in args.lasp_version for character in "\x00\r\n"):
        raise ContractError("lasp_version must be an explicit single-line value")
    executable = _ordinary_file(Path(args.lasp_executable), "lasp_executable")
    if executable.stat().st_mode & 0o111 == 0:
        raise ContractError("lasp_executable is not executable")
    input_structure = _ordinary_file(Path(args.input_structure), "input_structure")
    input_frames = _arc_frames(input_structure, 1)
    if len(input_frames) != 1:
        raise ContractError("input_structure must contain exactly one ARC frame")
    lasp_input = _ordinary_file(Path(args.lasp_input), "lasp_input")
    auxiliary = _parse_auxiliary(args.auxiliary)
    output_dir = _fresh_output(Path(args.output_dir))
    raw_run = output_dir / "raw-run"
    raw_run.mkdir()
    shutil.copyfile(input_structure, raw_run / "input.arc")
    shutil.copyfile(lasp_input, raw_run / "lasp.in")
    for name, path in auxiliary:
        shutil.copyfile(path, raw_run / name)
    staged_inputs = [
        {
            "role": "input-structure",
            "destination": "input.arc",
            "sha256": _sha256(raw_run / "input.arc"),
            "size_bytes": (raw_run / "input.arc").stat().st_size,
        },
        {
            "role": "lasp-input",
            "destination": "lasp.in",
            "sha256": _sha256(raw_run / "lasp.in"),
            "size_bytes": (raw_run / "lasp.in").stat().st_size,
        },
    ]
    staged_inputs.extend(
        {
            "role": "auxiliary-input",
            "destination": name,
            "sha256": _sha256(raw_run / name),
            "size_bytes": (raw_run / name).stat().st_size,
        }
        for name, _ in auxiliary
    )

    command = [str(executable)]
    launcher: Path | None = None
    if args.mpi_launcher is not None:
        launcher = _ordinary_file(Path(args.mpi_launcher), "mpi_launcher")
        launcher_name = launcher.name
        if (
            launcher_name not in {"mpirun", "mpiexec"}
            or launcher.stat().st_mode & 0o111 == 0
            or args.mpi_processes is None
            or args.mpi_processes < 1
        ):
            raise ContractError(
                "mpi_launcher must be an executable mpirun/mpiexec file and requires positive mpi_processes"
            )
        command = [str(launcher), "-np", str(args.mpi_processes), str(executable)]
    elif args.mpi_processes is not None:
        raise ContractError("mpi_processes requires mpi_launcher")

    stdout_path = raw_run / "lasp.stdout.log"
    stderr_path = raw_run / "lasp.stderr.log"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        completed = subprocess.run(
            command,
            cwd=raw_run,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            shell=False,
            check=False,
        )
    if completed.returncode != 0:
        raise ContractError(f"LASP process returned {completed.returncode}")

    diagnostic_artifacts = [
        (stdout_path, "lasp-stdout", "text/plain"),
        (stderr_path, "lasp-stderr", "text/plain"),
    ]
    native_log = raw_run / "lasp.out"
    if native_log.is_file() and not native_log.is_symlink():
        diagnostic_artifacts.append((native_log, "lasp-native-log", "text/plain"))

    return _normalize(
        source_dir=raw_run,
        lasp_input=raw_run / "lasp.in",
        output_dir=output_dir,
        operation="lasp-ssw-execute",
        mode="execute",
        source_id=args.historical_source_id,
        selection_stride=args.selection_stride,
        energy_max_ev=args.energy_max_ev,
        max_frames=args.max_frames,
        include_best_arc=args.include_best_arc,
        include_md_arc=args.include_md_arc,
        execution_metadata={
            "performed": True,
            "external_program": "LASP",
            "version": args.lasp_version,
            "executable_name": executable.name,
            "executable_sha256": _sha256(executable),
            "mpi_launcher": launcher.name if launcher else None,
            "mpi_launcher_sha256": _sha256(launcher) if launcher else None,
            "mpi_processes": args.mpi_processes,
            "returncode": completed.returncode,
            "command_uses_shell": False,
            "staged_inputs": staged_inputs,
        },
        extra_artifacts=diagnostic_artifacts,
    )


def _add_normalization_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lasp-input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--historical-source-id", required=True)
    parser.add_argument("--selection-stride", type=int, required=True)
    parser.add_argument("--energy-max-ev", type=float)
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--seed-status", required=True)
    parser.add_argument("--include-best-arc", action="store_true")
    parser.add_argument("--include-md-arc", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local LASP/SSW execute and historical-output normalization contract"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    normalize = subparsers.add_parser("normalize-replay")
    normalize.add_argument("--historical-run-dir", required=True)
    _add_normalization_arguments(normalize)
    normalize.set_defaults(handler=normalize_replay)

    run = subparsers.add_parser("execute")
    run.add_argument("--lasp-executable", required=True)
    run.add_argument("--input-structure", required=True)
    run.add_argument("--lasp-version", required=True)
    run.add_argument("--mpi-launcher")
    run.add_argument("--mpi-processes", type=int)
    run.add_argument(
        "--auxiliary", action="append", nargs=2, metavar=("NAME", "PATH"), default=[]
    )
    _add_normalization_arguments(run)
    run.set_defaults(handler=execute)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (ContractError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"LASP/SSW contract error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
