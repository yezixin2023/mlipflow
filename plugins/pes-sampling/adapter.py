"""Safe, side-effect-free adapter for DIRECT and LASP/SSW contracts.

The adapter only describes local commands and reads explicit result manifests.  It
never imports MAML/pymatgen/LASP, starts a process, creates directories, or submits
jobs.  DIRECT selection and LASP execution/normalization are delegated to bundled
argv wrappers after MLIPFlow approval.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


PLUGIN_ID = "pes-sampling"
DIRECT_OPERATION = "direct-select"
MERGE_OPERATION = "merge-structures"
LASP_INPUT_OPERATION = "lasp-input-prepare"
LASP_OPERATIONS = frozenset({"lasp-ssw-execute", "lasp-ssw-normalize-replay"})
OPERATIONS = (
    frozenset({DIRECT_OPERATION, MERGE_OPERATION, LASP_INPUT_OPERATION})
    | LASP_OPERATIONS
)
UNKNOWN = "HISTORICAL_PARAMETER_UNKNOWN"
BUNDLED_DIRECT_WRAPPER = (
    Path(globals().get("__file__", "adapter.py")).absolute().with_name("direct_select.py")
)
BUNDLED_LASP_WRAPPER = (
    Path(globals().get("__file__", "adapter.py")).absolute().with_name("lasp_ssw.py")
)
BUNDLED_MERGE_WRAPPER = (
    Path(globals().get("__file__", "adapter.py")).absolute().with_name("structure_merge.py")
)
BUNDLED_LASP_INPUT_WRAPPER = (
    Path(globals().get("__file__", "adapter.py"))
    .absolute()
    .with_name("structure_to_lasp.py")
)
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_ARC_BYTES = 512 * 1024 * 1024
MAX_FRAME_BYTES = 8 * 1024 * 1024
MAX_LASP_INPUT_BYTES = 1024 * 1024
MAX_POTCAR_BYTES = 64 * 1024 * 1024
MAX_LASP_FRAMES = 10000
ARC_HEADER = b"!BIOSYM archive 2\nPBC=ON\n"
SAFE_POTCAR_SYMBOL = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
LASP_ARTIFACT_ROLES = frozenset(
    {
        "ssw-generated-structure",
        "selected-structure",
        "ssw-structure-manifest",
        "selected-structure-manifest",
        "aimd-seed-candidate",
        "aimd-seed-manifest",
        "md-sampled-structure",
        "md-structure-manifest",
        "lasp-run-metadata",
        "lasp-stdout",
        "lasp-stderr",
        "lasp-native-log",
    }
)
_DIRECT_COLUMNS = {
    "selected_order",
    "input_index",
    "source_kind",
    "source_file",
    "frame_index",
    "output_file",
    "formula",
}
_OPTIONAL_PLOTS = (
    "direct_pca_coverage.html",
    "direct_pca_explained_variance.html",
    "direct_feature_coverage_score.html",
    "direct_pca_coverage.png",
    "direct_pca_explained_variance.png",
    "direct_feature_coverage_score.png",
)
_DIRECT_PARAMETERS = {
    "operation",
    "output_subdir",
    "globs",
    "recursive",
    "stride",
    "max_frames_per_file",
    "max_input_structures",
    "n_clusters",
    "threshold_init",
    "k_per_cluster",
    "lammps_type_map",
    "seed",
    "acknowledge_uncontrolled_seed",
}
_LASP_PARAMETERS = {
    "operation",
    "output_subdir",
    "historical_source_id",
    "selection_stride",
    "energy_max_ev",
    "max_frames",
    "include_best_arc",
    "include_md_arc",
    "seed_status",
    "acknowledge_uncontrolled_seed",
    "preserve_historical_order",
    "lasp_version",
    "mpi_processes",
}
_MERGE_PARAMETERS = {
    "operation",
    "output_subdir",
    "direct_source_group_id",
    "lasp_source_group_id",
    "matcher_ltol",
    "matcher_stol",
    "matcher_angle_tol_deg",
    "minimum_distance_angstrom",
    "max_structures",
}
_LASP_INPUT_PARAMETERS = {
    "operation",
    "output_subdir",
    "input_format",
    "input_index",
    "minimum_cell_length_angstrom",
}


def _diagnostic(level: str, code: str, message: str) -> Dict[str, str]:
    return {"level": level, "code": code, "message": message}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _resolve(value: Any, base: Path) -> Optional[Path]:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _unresolved(value: Any, base: Path) -> Optional[Path]:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).absolute()


def _has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    return any(candidate.is_symlink() for candidate in (absolute, *absolute.parents))


def _resolve_nonsymlink(value: Any, base: Path) -> Optional[Path]:
    path = _unresolved(value, base)
    if path is None or _has_symlink_component(path):
        return None
    return path.resolve()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _safe_subdirectory(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    path = Path(value)
    return not path.is_absolute() and path != Path(".") and ".." not in path.parts


def _has_errors(diagnostics: List[Dict[str, str]]) -> bool:
    return any(item["level"] == "ERROR" for item in diagnostics)


def _blocked(diagnostics: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "plugin_id": PLUGIN_ID,
        "status": "BLOCKED",
        "executable": False,
        "diagnostics": diagnostics,
    }


def _operation(context: Any) -> str:
    if not isinstance(context, Mapping):
        return DIRECT_OPERATION
    return str(_mapping(context.get("parameters")).get("operation", DIRECT_OPERATION))


def _plain_string(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and "\x00" not in value
        and "\n" not in value
        and "\r" not in value
    )


def _portable_source_id(value: Any) -> bool:
    if not _plain_string(value):
        return False
    normalized = str(value).replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized):
        return False
    return "://" in normalized or ".." not in Path(normalized).parts


def _pseudopotential_reference(path: Path) -> Dict[str, Any]:
    if not _ordinary_file(path) or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("pseudopotential reference must be an ordinary JSON file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("pseudopotential reference must contain an object")
    allowed = {
        "schema_version",
        "reference_id",
        "source_env",
        "license_acknowledged",
        "functional",
        "symbols",
    }
    if set(value) - allowed:
        raise ValueError("pseudopotential reference contains unsupported fields")
    if value.get("schema_version") != 1 or value.get("source_env") != "PMG_VASP_PSP_DIR":
        raise ValueError(
            "pseudopotential reference requires schema_version=1 and PMG_VASP_PSP_DIR"
        )
    if value.get("license_acknowledged") is not True:
        raise ValueError("pseudopotential license acknowledgement is required")
    reference_id = value.get("reference_id")
    if not _portable_source_id(reference_id):
        raise ValueError("pseudopotential reference_id must be portable")
    functional = value.get("functional")
    if not _plain_string(functional):
        raise ValueError("pseudopotential functional must be explicit")
    raw_symbols = value.get("symbols")
    if not isinstance(raw_symbols, Mapping) or not raw_symbols:
        raise ValueError("pseudopotential symbols must be an explicit mapping")
    symbols: Dict[str, str] = {}
    for element, symbol in raw_symbols.items():
        if (
            not isinstance(element, str)
            or re.fullmatch(r"[A-Z][a-z]?", element) is None
            or not isinstance(symbol, str)
            or SAFE_POTCAR_SYMBOL.fullmatch(symbol) is None
        ):
            raise ValueError("pseudopotential symbols mapping is invalid")
        symbols[element] = symbol
    return {
        "reference_id": reference_id,
        "functional": functional,
        "symbols": symbols,
    }


def _lasp_structure_id(role: str, frame_index: int) -> str:
    return f"lasp-{role}-{frame_index:06d}"


def _ordinary_file(path: Optional[Path]) -> bool:
    return path is not None and not path.is_symlink() and path.is_file()


def _explicit_executable(value: Any, base: Path) -> Optional[Path]:
    """Resolve an absolute, non-symlink executable without consulting ``PATH``."""
    if not _plain_string(value):
        return None
    unresolved = Path(str(value)).expanduser()
    if not unresolved.is_absolute():
        return None
    path = _resolve_nonsymlink(unresolved, base)
    if not _ordinary_file(path):
        return None
    assert path is not None
    return path if path.stat().st_mode & 0o111 else None


def _parse_energy_record(line: bytes, source: str, frame_index: int) -> float:
    try:
        tokens = line.decode("ascii").split()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source} frame {frame_index} Energy record is not ASCII") from exc
    if tokens[:1] != ["Energy"] or len(tokens) not in {3, 4, 5}:
        raise ValueError(f"{source} frame {frame_index} has an unsupported Energy record")
    try:
        int(tokens[1])
    except ValueError as exc:
        raise ValueError(f"{source} frame {frame_index} Energy index is invalid") from exc
    if len(tokens) == 5 and not re.fullmatch(r"C[0-9]+", tokens[4]):
        raise ValueError(f"{source} frame {frame_index} Energy suffix is invalid")
    energy_token = tokens[2] if len(tokens) == 3 else tokens[3]
    try:
        energy = float(energy_token)
    except ValueError as exc:
        raise ValueError(f"{source} frame {frame_index} energy is invalid") from exc
    if not math.isfinite(energy):
        raise ValueError(f"{source} frame {frame_index} energy is not finite")
    return energy


def _read_arc_frames(path: Path, max_frames: int) -> List[Dict[str, Any]]:
    size = path.stat().st_size
    if size == 0 or size > MAX_ARC_BYTES:
        raise ValueError(f"{path.name} size must be between 1 and {MAX_ARC_BYTES} bytes")
    lines = path.read_bytes().splitlines(keepends=True)
    starts = [index for index, line in enumerate(lines) if line.lstrip().startswith(b"Energy")]
    if not starts or len(starts) > max_frames:
        raise ValueError(f"{path.name} frame count is zero or exceeds max_frames")
    frames: List[Dict[str, Any]] = []
    for position, start in enumerate(starts):
        stop = starts[position + 1] if position + 1 < len(starts) else len(lines)
        block = lines[start:stop]
        payload = ARC_HEADER + b"".join(block)
        if len(payload) > MAX_FRAME_BYTES:
            raise ValueError(f"{path.name} frame {position + 1} exceeds size limit")
        if sum(1 for line in block if line.strip() == b"end") < 2:
            raise ValueError(f"{path.name} frame {position + 1} is truncated")
        frames.append(
            {
                "frame_index": position + 1,
                "energy_ev": _parse_energy_record(block[0], path.name, position + 1),
                "payload": payload,
            }
        )
    return frames


def _parse_lasp_input(path: Path) -> Dict[str, Any]:
    if path.stat().st_size == 0 or path.stat().st_size > MAX_LASP_INPUT_BYTES:
        raise ValueError("lasp.in is empty or exceeds the size limit")
    parameters: Dict[str, Any] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw_line.split("#", 1)[0].strip()
        if not stripped or stripped.startswith(("#", "!")):
            continue
        tokens = stripped.split()
        if len(tokens) < 2:
            raise ValueError(f"lasp.in line {line_number} has no explicit value")
        key = tokens[0]
        value: Any = tokens[1] if len(tokens) == 2 else tokens[1:]
        if key in parameters:
            previous = parameters[key]
            parameters[key] = previous + [value] if isinstance(previous, list) else [previous, value]
        else:
            parameters[key] = value
    if not parameters:
        raise ValueError("lasp.in contains no parameters")
    return parameters


def _validate_ssw_input(parameters: Mapping[str, Any]) -> None:
    if str(parameters.get("explore_type", "")).lower() != "ssw":
        raise ValueError("lasp.in must explicitly declare explore_type ssw")
    steps = parameters.get("SSW.SSWsteps")
    try:
        numeric_steps = int(str(steps))
    except (TypeError, ValueError) as exc:
        raise ValueError("lasp.in must declare an integer SSW.SSWsteps") from exc
    if numeric_steps < 1:
        raise ValueError("lasp-ssw operations require SSW.SSWsteps >= 1")


class Adapter:
    """Plan reviewed DIRECT or LASP/SSW local argv without executing it."""

    def validate(self, context: Any) -> List[Dict[str, str]]:
        operation = _operation(context)
        if operation == DIRECT_OPERATION:
            return self._validate_direct(context)
        if operation == MERGE_OPERATION:
            return self._validate_merge(context)
        if operation == LASP_INPUT_OPERATION:
            return self._validate_lasp_input(context)
        if operation in LASP_OPERATIONS:
            return self._validate_lasp(context)
        return [
            _diagnostic(
                "ERROR",
                "operation.unsupported",
                "operation must be one of: %s" % ", ".join(sorted(OPERATIONS)),
            )
        ]

    def plan(self, context: Any) -> Dict[str, Any]:
        operation = _operation(context)
        if operation == DIRECT_OPERATION:
            return self._plan_direct(context)
        if operation == MERGE_OPERATION:
            return self._plan_merge(context)
        if operation == LASP_INPUT_OPERATION:
            return self._plan_lasp_input(context)
        if operation in LASP_OPERATIONS:
            return self._plan_lasp(context)
        return _blocked(self.validate(context))

    def _validate_direct(self, context: Any) -> List[Dict[str, str]]:
        diagnostics: List[Dict[str, str]] = []
        if not isinstance(context, Mapping):
            return [_diagnostic("ERROR", "context.mapping_required", "context must be a mapping")]

        project_root = _resolve(context.get("project_root"), Path.cwd())
        if project_root is None or not project_root.is_dir():
            diagnostics.append(
                _diagnostic(
                    "ERROR", "path.project_root", "project_root must name an existing directory"
                )
            )
            project_root = Path.cwd().resolve()

        attempt_dir = _resolve(context.get("attempt_dir"), project_root)
        if attempt_dir is None:
            diagnostics.append(
                _diagnostic("ERROR", "path.attempt_dir", "attempt_dir must be an explicit path")
            )
            attempt_dir = project_root / ".mlipflow-invalid-attempt"
        elif not _is_within(attempt_dir, project_root):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "path.attempt_outside_project",
                    "attempt_dir must stay inside project_root",
                )
            )

        if context.get("backend", "local") != "local":
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "backend.local_only",
                    "DIRECT is exposed as a local argv only; scheduler wrapping belongs to the core",
                )
            )

        inputs = _mapping(context.get("inputs"))
        parameters = _mapping(context.get("parameters"))
        resources = _mapping(context.get("resources"))
        if not isinstance(context.get("inputs", {}), Mapping):
            diagnostics.append(
                _diagnostic("ERROR", "inputs.mapping_required", "inputs must be a mapping")
            )
        if not isinstance(context.get("parameters", {}), Mapping):
            diagnostics.append(
                _diagnostic("ERROR", "parameters.mapping_required", "parameters must be a mapping")
            )
        unknown_parameters = sorted(set(parameters) - _DIRECT_PARAMETERS)
        if unknown_parameters:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "parameter.unknown",
                    "unsupported parameter(s): %s" % ", ".join(unknown_parameters),
                )
            )
        if not isinstance(context.get("resources", {}), Mapping):
            diagnostics.append(
                _diagnostic("ERROR", "resources.mapping_required", "resources must be a mapping")
            )
        unknown_resources = sorted(set(resources) - {"python_executable"})
        if unknown_resources:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "resource.unknown",
                    "unsupported DIRECT resource(s): %s" % ", ".join(unknown_resources),
                )
            )

        unknown_inputs = sorted(set(inputs) - {"input_dirs", "result_manifest"})
        if unknown_inputs:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "input.unknown",
                    "unsupported DIRECT input(s): %s" % ", ".join(unknown_inputs),
                )
            )
        if not BUNDLED_DIRECT_WRAPPER.is_file() or BUNDLED_DIRECT_WRAPPER.is_symlink():
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "path.bundled_direct_wrapper",
                    "the bundled DIRECT runner is missing or is not an ordinary file",
                )
            )

        raw_input_dirs = inputs.get("input_dirs")
        input_dirs: List[Path] = []
        if not isinstance(raw_input_dirs, list) or not raw_input_dirs:
            diagnostics.append(
                _diagnostic(
                    "ERROR", "path.input_dirs", "inputs.input_dirs must be a non-empty list"
                )
            )
        else:
            for index, raw_path in enumerate(raw_input_dirs):
                resolved = _resolve(raw_path, project_root)
                if resolved is None or not resolved.is_dir():
                    diagnostics.append(
                        _diagnostic(
                            "ERROR",
                            "path.input_dir",
                            "input_dirs[%d] must name an existing directory" % index,
                        )
                    )
                else:
                    input_dirs.append(resolved)

        output_subdir = parameters.get("output_subdir", "direct-selected")
        if not _safe_subdirectory(output_subdir):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "path.output_subdir",
                    "output_subdir must be a non-empty relative path without '..'",
                )
            )
            output_dir = attempt_dir / "direct-selected"
        else:
            output_dir = (attempt_dir / str(output_subdir)).resolve()
            if not _is_within(output_dir, attempt_dir):
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "path.output_escape", "DIRECT output must stay inside attempt_dir"
                    )
                )
        if output_dir.exists():
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "path.output_exists",
                    "the bundled DIRECT runner requires a fresh attempt/output_subdir",
                )
            )
        for input_dir in input_dirs:
            if _is_within(output_dir, input_dir) or _is_within(input_dir, output_dir):
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "path.input_output_overlap",
                        "input and output directories must not overlap",
                    )
                )
                break

        globs = parameters.get("globs")
        if (
            not isinstance(globs, list)
            or not globs
            or not all(isinstance(item, str) and item.strip() for item in globs)
        ):
            diagnostics.append(
                _diagnostic("ERROR", "parameter.globs", "globs must be a non-empty string list")
            )
        elif any(Path(item).is_absolute() or ".." in Path(item).parts for item in globs):
            diagnostics.append(
                _diagnostic(
                    "ERROR", "parameter.glob_escape", "globs must be relative patterns without '..'"
                )
            )

        if not isinstance(parameters.get("recursive", True), bool):
            diagnostics.append(
                _diagnostic("ERROR", "parameter.recursive", "recursive must be boolean")
            )
        for name in ("stride", "n_clusters", "k_per_cluster"):
            if not _positive_int(parameters.get(name)):
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "parameter.%s" % name, "%s must be a positive integer" % name
                    )
                )
        for name in ("max_frames_per_file", "max_input_structures"):
            value = parameters.get(name)
            if value is not None and not _positive_int(value):
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "parameter.%s" % name,
                        "%s must be null or a positive integer" % name,
                    )
                )
        if not _positive_number(parameters.get("threshold_init")):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "parameter.threshold_init",
                    "threshold_init must be a finite positive number",
                )
            )

        seed = parameters.get("seed")
        if not _nonnegative_int(seed):
            diagnostics.append(
                _diagnostic(
                    "ERROR", "parameter.seed", "seed must be an explicit non-negative integer"
                )
            )
        if parameters.get("acknowledge_uncontrolled_seed") is not True:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "seed.no_cli_control",
                    "MAML DIRECT exposes no seed control in this selection path; set acknowledge_uncontrolled_seed=true to accept this limitation",
                )
            )
        else:
            diagnostics.append(
                _diagnostic(
                    "WARNING",
                    "seed.provenance_only",
                    "seed is recorded in the plan but cannot be passed to the bundled MAML DIRECT selection path",
                )
            )

        lammps_map = parameters.get("lammps_type_map")
        if lammps_map is not None:
            if not isinstance(lammps_map, Mapping) or not lammps_map:
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "parameter.lammps_type_map",
                        "lammps_type_map must be a non-empty mapping",
                    )
                )
            else:
                for atom_type, symbol in lammps_map.items():
                    try:
                        numeric_type = int(atom_type)
                    except (TypeError, ValueError):
                        numeric_type = 0
                    if (
                        numeric_type < 1
                        or not isinstance(symbol, str)
                        or not re.fullmatch(r"[A-Z][a-z]?", symbol)
                    ):
                        diagnostics.append(
                            _diagnostic(
                                "ERROR",
                                "parameter.lammps_type_map_entry",
                                "LAMMPS types must be positive integers mapped to element symbols",
                            )
                        )
                        break

        python_executable = resources.get("python_executable", sys.executable)
        if not isinstance(python_executable, str) or not python_executable.strip():
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "resource.python_executable",
                    "python_executable must be a non-empty string",
                )
            )
        return diagnostics

    def _plan_direct(self, context: Any) -> Dict[str, Any]:
        diagnostics = self._validate_direct(context)
        if _has_errors(diagnostics):
            return _blocked(diagnostics)

        project_root = _resolve(context["project_root"], Path.cwd())
        attempt_dir = _resolve(context["attempt_dir"], project_root)
        inputs = _mapping(context["inputs"])
        parameters = _mapping(context["parameters"])
        resources = _mapping(context.get("resources"))
        assert project_root is not None and attempt_dir is not None
        script = BUNDLED_DIRECT_WRAPPER.resolve()
        input_dirs = [_resolve(value, project_root) for value in inputs["input_dirs"]]
        output_dir = (
            attempt_dir / str(parameters.get("output_subdir", "direct-selected"))
        ).resolve()

        argv = [
            str(resources.get("python_executable", sys.executable)),
            str(script),
            "--input-dirs",
        ]
        argv.extend(str(path) for path in input_dirs)
        argv.extend(["--globs"])
        argv.extend(str(value) for value in parameters["globs"])
        if parameters.get("recursive", True) is False:
            argv.append("--no-recursive")
        argv.extend(["--stride", str(parameters["stride"])])
        if parameters.get("max_frames_per_file") is not None:
            argv.extend(["--max-frames-per-file", str(parameters["max_frames_per_file"])])
        if parameters.get("max_input_structures") is not None:
            argv.extend(["--max-input-structures", str(parameters["max_input_structures"])])
        argv.extend(["--n-clusters", str(parameters["n_clusters"])])
        argv.extend(["--threshold-init", str(parameters["threshold_init"])])
        argv.extend(["--k-per-cluster", str(parameters["k_per_cluster"])])
        argv.extend(["--out-dir", str(output_dir)])
        if parameters.get("lammps_type_map") is not None:
            encoded = json.dumps(
                {str(key): value for key, value in parameters["lammps_type_map"].items()},
                sort_keys=True,
                separators=(",", ":"),
            )
            argv.extend(["--lammps-type-map", encoded])

        return {
            "plugin_id": PLUGIN_ID,
            "operation": "direct-select",
            "status": "READY",
            "executable": True,
            "argv": argv,
            "cwd": str(attempt_dir),
            "shell": False,
            "expected_outputs": [str(output_dir / "manifest.csv")],
            "output_dir": str(output_dir),
            "input_paths": {
                "direct_wrapper": str(script),
                "input_directories": [str(path) for path in input_dirs],
            },
            "assumptions": {
                "seed": parameters["seed"],
                "seed_control": "not-exposed-by-bundled-maml-interface",
                "selection_implementation": "maml.sampling.direct.DIRECTSampler",
                "fresh_output_directory_required": True,
            },
            "diagnostics": diagnostics,
        }

    def _validate_lasp_input(self, context: Any) -> List[Dict[str, str]]:
        diagnostics: List[Dict[str, str]] = []
        if not isinstance(context, Mapping):
            return [
                _diagnostic("ERROR", "context.mapping_required", "context must be a mapping")
            ]
        project_root = _resolve(context.get("project_root"), Path.cwd())
        if project_root is None or not project_root.is_dir():
            diagnostics.append(
                _diagnostic(
                    "ERROR", "path.project_root", "project_root must name an existing directory"
                )
            )
            project_root = Path.cwd().resolve()
        attempt_dir = _resolve(context.get("attempt_dir"), project_root)
        if attempt_dir is None or not _is_within(attempt_dir, project_root):
            diagnostics.append(
                _diagnostic(
                    "ERROR", "path.attempt_dir", "attempt_dir must stay inside project_root"
                )
            )
            attempt_dir = project_root / ".mlipflow-invalid-attempt"
        if context.get("backend", "local") != "local":
            diagnostics.append(
                _diagnostic(
                    "ERROR", "backend.local_only", "lasp-input-prepare is local-only"
                )
            )

        inputs = _mapping(context.get("inputs"))
        parameters = _mapping(context.get("parameters"))
        resources = _mapping(context.get("resources"))
        unknown_inputs = sorted(
            set(inputs)
            - {
                "input_structure",
                "pseudopotential_reference",
                "result_manifest",
            }
        )
        unknown_parameters = sorted(set(parameters) - _LASP_INPUT_PARAMETERS)
        unknown_resources = sorted(set(resources) - {"python_executable"})
        for code, values in (
            ("input.unknown", unknown_inputs),
            ("parameter.unknown", unknown_parameters),
            ("resource.unknown", unknown_resources),
        ):
            if values:
                diagnostics.append(
                    _diagnostic(
                        "ERROR", code, "unsupported value(s): %s" % ", ".join(values)
                    )
                )

        source = _resolve(inputs.get("input_structure"), project_root)
        if source is None:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "path.input_structure",
                    "input_structure must be a non-empty path",
                )
            )
        elif not source.exists():
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "path.input_structure",
                    f"input_structure file does not exist: {source}",
                )
            )
        elif not _ordinary_file(source):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "path.input_structure",
                    f"input_structure must be an ordinary file: {source}",
                )
            )
        else:
            try:
                with source.open("rb") as stream:
                    stream.read(1)
            except OSError:
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "path.input_structure",
                        f"input_structure must be readable: {source}",
                    )
                )
        pseudopotential = inputs.get("pseudopotential_reference")
        if pseudopotential is not None:
            reference_path = _resolve_nonsymlink(pseudopotential, project_root)
            if (
                not _ordinary_file(reference_path)
                or reference_path is None
                or not _is_within(reference_path, project_root)
            ):
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "path.pseudopotential_reference",
                        "pseudopotential_reference must be an ordinary project file",
                    )
                )
            else:
                try:
                    _pseudopotential_reference(reference_path)
                except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                    diagnostics.append(
                        _diagnostic(
                            "ERROR",
                            "input.pseudopotential_reference",
                            str(exc),
                        )
                    )
        if not _ordinary_file(BUNDLED_LASP_INPUT_WRAPPER):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "path.lasp_input_wrapper",
                    "bundled LASP input converter is missing",
                )
            )
        python_executable = _explicit_executable(
            resources.get("python_executable"), project_root
        )
        if python_executable is None:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "resource.python_executable",
                    "python_executable must be an explicit ordinary executable",
                )
            )
        output_subdir = parameters.get("output_subdir", "lasp-input")
        if not _safe_subdirectory(output_subdir):
            diagnostics.append(
                _diagnostic(
                    "ERROR", "path.output_subdir", "output_subdir must be a safe relative path"
                )
            )
            output_dir = attempt_dir / "lasp-input"
        else:
            output_dir = (attempt_dir / str(output_subdir)).resolve()
        if not _is_within(output_dir, attempt_dir) or output_dir.exists() or output_dir.is_symlink():
            diagnostics.append(
                _diagnostic(
                    "ERROR", "path.output_exists", "LASP input conversion requires a fresh output"
                )
            )
        for name in ("input_format", "input_index"):
            value = parameters.get(name)
            if value is not None and not _plain_string(str(value)):
                diagnostics.append(
                    _diagnostic(
                        "ERROR", f"parameter.{name}", f"{name} must be a plain string"
                    )
                )
        minimum = parameters.get("minimum_cell_length_angstrom")
        if minimum is not None and not _positive_number(minimum):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "parameter.minimum_cell_length_angstrom",
                    "minimum_cell_length_angstrom must be finite and positive",
                )
            )
        return diagnostics

    def _plan_lasp_input(self, context: Any) -> Dict[str, Any]:
        diagnostics = self._validate_lasp_input(context)
        if _has_errors(diagnostics):
            return _blocked(diagnostics)
        project_root = _resolve(context["project_root"], Path.cwd())
        attempt_dir = _resolve(context["attempt_dir"], project_root)
        inputs = _mapping(context["inputs"])
        parameters = _mapping(context["parameters"])
        resources = _mapping(context["resources"])
        assert project_root is not None and attempt_dir is not None
        source = _resolve(inputs["input_structure"], project_root)
        python_executable = _explicit_executable(
            resources["python_executable"], project_root
        )
        assert source is not None and python_executable is not None
        output_dir = (
            attempt_dir / str(parameters.get("output_subdir", "lasp-input"))
        ).resolve()
        wrapper = BUNDLED_LASP_INPUT_WRAPPER.resolve()
        argv = [
            str(python_executable),
            str(wrapper),
            "--input-structure",
            str(source),
            "--output-dir",
            str(output_dir),
            "--input-index",
            str(parameters.get("input_index", "-1")),
        ]
        if parameters.get("input_format") is not None:
            argv.extend(["--input-format", str(parameters["input_format"])])
        if parameters.get("minimum_cell_length_angstrom") is not None:
            argv.extend(
                [
                    "--minimum-cell-length-angstrom",
                    str(parameters["minimum_cell_length_angstrom"]),
                ]
            )
        reference_path = _resolve_nonsymlink(
            inputs.get("pseudopotential_reference"), project_root
        )
        reference = None
        if reference_path is not None:
            reference = _pseudopotential_reference(reference_path)
            argv.extend(["--pseudopotential-reference", str(reference_path)])
        conversion = {
            "source_path": str(source),
            "input_format": parameters.get("input_format"),
            "input_index": str(parameters.get("input_index", "-1")),
            "minimum_cell_length_angstrom": parameters.get(
                "minimum_cell_length_angstrom"
            ),
            "pseudopotential_reference_path": (
                str(reference_path) if reference_path is not None else None
            ),
            "pseudopotential": reference,
        }
        return {
            "plugin_id": PLUGIN_ID,
            "operation": LASP_INPUT_OPERATION,
            "status": "READY",
            "executable": True,
            "argv": argv,
            "cwd": str(attempt_dir),
            "shell": False,
            "expected_outputs": [str(output_dir / "lasp-input-manifest.json")],
            "output_dir": str(output_dir),
            "conversion": conversion,
            "approval_summary": {
                "expensive": False,
                "submits_jobs": False,
                "materializes_licensed_potcar": reference is not None,
                "potcar_collectable": False if reference is not None else None,
                **conversion,
            },
            "input_paths": {
                "source_structure": str(source),
                "conversion_wrapper": str(wrapper),
                "python_executable": str(python_executable),
                **(
                    {"pseudopotential_reference": str(reference_path)}
                    if reference_path is not None
                    else {}
                ),
            },
            "diagnostics": diagnostics,
        }

    def _check_lasp_input(self, context: Any) -> Dict[str, Any]:
        execution = _mapping(context.get("execution")) if isinstance(context, Mapping) else {}
        if execution.get("returncode") not in (None, 0):
            return {
                "plugin_id": PLUGIN_ID,
                "operation": LASP_INPUT_OPERATION,
                "status": "FAIL",
                "diagnostics": [
                    _diagnostic(
                        "ERROR", "execution.nonzero", "LASP input conversion returned nonzero"
                    )
                ],
            }
        project_root = _resolve(context.get("project_root"), Path.cwd()) or Path.cwd()
        plan = _mapping(execution.get("plan"))
        expected = plan.get("expected_outputs")
        path = (
            _unresolved(expected[0], project_root)
            if isinstance(expected, list) and len(expected) == 1
            else _unresolved(_mapping(context.get("inputs")).get("result_manifest"), project_root)
        )
        diagnostics: List[Dict[str, str]] = []
        if not _ordinary_file(path) or path is None or path.stat().st_size > MAX_JSON_BYTES:
            diagnostics.append(
                _diagnostic("ERROR", "result.manifest", "LASP input manifest is missing")
            )
            value: Dict[str, Any] = {}
        else:
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                value = dict(loaded) if isinstance(loaded, Mapping) else {}
            except (OSError, UnicodeError, json.JSONDecodeError):
                value = {}
            if not value:
                diagnostics.append(
                    _diagnostic("ERROR", "result.manifest", "LASP input manifest is invalid")
                )
        conversion = _mapping(plan.get("conversion"))
        if value:
            source = _mapping(value.get("source"))
            for key, expected_value in (
                ("path", conversion.get("source_path")),
                ("input_format", conversion.get("input_format")),
                ("input_index", conversion.get("input_index")),
            ):
                if source.get(key) != expected_value:
                    diagnostics.append(
                        _diagnostic("ERROR", f"result.source_{key}", f"source {key} differs")
                    )
            structure = _mapping(value.get("structure"))
            lengths = structure.get("cell_lengths_A")
            minimum = conversion.get("minimum_cell_length_angstrom")
            if (
                value.get("schema_version") != 1
                or value.get("plugin_id") != PLUGIN_ID
                or value.get("operation") != LASP_INPUT_OPERATION
                or not _positive_int(structure.get("atom_count"))
                or not isinstance(lengths, list)
                or len(lengths) != 3
                or any(not _positive_number(item) for item in lengths)
                or (
                    minimum is not None
                    and any(float(item) <= float(minimum) for item in lengths)
                )
            ):
                diagnostics.append(
                    _diagnostic("ERROR", "result.structure", "converted structure metadata is invalid")
                )
            output = _mapping(value.get("output"))
            arc = path.parent / str(output.get("path", "")) if path is not None else None
            if (
                output.get("path") != "input.arc"
                or not _ordinary_file(arc)
            ):
                diagnostics.append(
                    _diagnostic("ERROR", "result.arc", "converted ARC output is invalid")
                )
            reference = _mapping(conversion.get("pseudopotential"))
            if reference:
                potcar = _mapping(value.get("potcar"))
                potcar_output = _mapping(potcar.get("output"))
                potcar_path = (
                    path.parent / str(potcar_output.get("path", ""))
                    if path is not None
                    else None
                )
                elements = potcar.get("elements")
                symbols = potcar.get("symbols")
                expected_symbols = reference.get("symbols")
                components = potcar.get("components")
                valid_components = (
                    isinstance(symbols, list)
                    and isinstance(components, list)
                    and len(components) == len(symbols)
                    and all(
                        isinstance(component, Mapping)
                        and component.get("symbol") == symbol
                        for component, symbol in zip(components, symbols)
                    )
                )
                valid_symbols = (
                    isinstance(elements, list)
                    and bool(elements)
                    and isinstance(symbols, list)
                    and isinstance(expected_symbols, Mapping)
                    and symbols == [expected_symbols.get(element) for element in elements]
                )
                if (
                    value.get("pseudopotential_reference_path")
                    != conversion.get("pseudopotential_reference_path")
                    or potcar.get("reference_id") != reference.get("reference_id")
                    or potcar.get("source_env") != "PMG_VASP_PSP_DIR"
                    or potcar.get("configuration_source")
                    not in {"environment", "pymatgen-settings"}
                    or potcar.get("functional") != reference.get("functional")
                    or potcar.get("portable_artifact") is not False
                    or not valid_symbols
                    or not valid_components
                    or potcar_output.get("path") != "POTCAR"
                    or potcar_output.get("collectable") is not False
                    or not _ordinary_file(potcar_path)
                    or potcar_path is None
                    or potcar_path.stat().st_size > MAX_POTCAR_BYTES
                ):
                    diagnostics.append(
                        _diagnostic(
                            "ERROR",
                            "result.potcar",
                            "runtime-only POTCAR settings differ from the approved reference",
                        )
                    )
            elif (
                value.get("pseudopotential_reference_path") is not None
                or value.get("potcar") is not None
            ):
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "result.potcar_unapproved",
                        "manifest contains an unapproved POTCAR record",
                    )
                )
        return {
            "plugin_id": PLUGIN_ID,
            "operation": LASP_INPUT_OPERATION,
            "status": "FAIL" if diagnostics else "OK",
            "result_file": str(path) if path is not None else None,
            "diagnostics": diagnostics,
        }

    def _collect_lasp_input(self, context: Any) -> Dict[str, Any]:
        checked = self._check_lasp_input(context)
        if checked["status"] != "OK":
            return {**checked, "artifacts": [], "metrics": {}}
        manifest = Path(str(checked["result_file"]))
        value = json.loads(manifest.read_text(encoding="utf-8"))
        return {
            **checked,
            "artifacts": [
                {
                    "role": "lasp-input-manifest",
                    "path": str(manifest),
                    "media_type": "application/json",
                },
                {
                    "role": "lasp-input-structure",
                    "path": str(manifest.parent / "input.arc"),
                    "media_type": "chemical/x-biosym-archive",
                },
            ],
            "metrics": {
                "atom_count": _mapping(value.get("structure")).get("atom_count", 0)
            },
        }

    def _validate_merge(self, context: Any) -> List[Dict[str, str]]:
        diagnostics: List[Dict[str, str]] = []
        if not isinstance(context, Mapping):
            return [_diagnostic("ERROR", "context.mapping_required", "context must be a mapping")]
        project_root = _resolve(context.get("project_root"), Path.cwd())
        if project_root is None or not project_root.is_dir():
            diagnostics.append(
                _diagnostic("ERROR", "path.project_root", "project_root must name an existing directory")
            )
            project_root = Path.cwd().resolve()
        attempt_dir = _resolve(context.get("attempt_dir"), project_root)
        if attempt_dir is None or not _is_within(attempt_dir, project_root):
            diagnostics.append(
                _diagnostic("ERROR", "path.attempt_dir", "attempt_dir must stay inside project_root")
            )
            attempt_dir = project_root / ".mlipflow-invalid-attempt"
        if context.get("backend", "local") != "local":
            diagnostics.append(
                _diagnostic("ERROR", "backend.local_only", "merge-structures is local-only")
            )

        inputs = _mapping(context.get("inputs"))
        parameters = _mapping(context.get("parameters"))
        resources = _mapping(context.get("resources"))
        if not isinstance(context.get("inputs", {}), Mapping):
            diagnostics.append(_diagnostic("ERROR", "inputs.mapping_required", "inputs must be a mapping"))
        if not isinstance(context.get("parameters", {}), Mapping):
            diagnostics.append(
                _diagnostic("ERROR", "parameters.mapping_required", "parameters must be a mapping")
            )
        if not isinstance(context.get("resources", {}), Mapping):
            diagnostics.append(
                _diagnostic("ERROR", "resources.mapping_required", "resources must be a mapping")
            )
        unknown_inputs = sorted(
            set(inputs)
            - {
                "direct_manifest",
                "lasp_selected_manifest",
                "lasp_selected_archive",
                "result_manifest",
            }
        )
        if unknown_inputs:
            diagnostics.append(
                _diagnostic(
                    "ERROR", "input.unknown", "unsupported merge input(s): " + ", ".join(unknown_inputs)
                )
            )
        for name in ("direct_manifest", "lasp_selected_manifest", "lasp_selected_archive"):
            path = _resolve_nonsymlink(inputs.get(name), project_root)
            if not _ordinary_file(path) or path is None or not _is_within(path, project_root):
                diagnostics.append(
                    _diagnostic(
                        "ERROR", f"path.{name}", f"{name} must be an ordinary project file"
                    )
                )
        unknown_parameters = sorted(set(parameters) - _MERGE_PARAMETERS)
        if unknown_parameters:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "parameter.unknown",
                    "unsupported merge parameter(s): " + ", ".join(unknown_parameters),
                )
            )
        output_subdir = parameters.get("output_subdir", "merged-structures")
        if not _safe_subdirectory(output_subdir):
            diagnostics.append(
                _diagnostic("ERROR", "path.output_subdir", "output_subdir must be a safe relative path")
            )
        else:
            output_dir = (attempt_dir / str(output_subdir)).resolve()
            if not _is_within(output_dir, attempt_dir):
                diagnostics.append(
                    _diagnostic("ERROR", "path.output_escape", "merge output must stay inside attempt_dir")
                )
            elif output_dir.exists():
                diagnostics.append(
                    _diagnostic("ERROR", "path.output_exists", "merge-structures requires fresh output")
                )
        for name in ("direct_source_group_id", "lasp_source_group_id"):
            value = parameters.get(name)
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", value):
                diagnostics.append(
                    _diagnostic("ERROR", f"parameter.{name}", f"{name} must be a portable ID")
                )
        numeric_ranges = {
            "matcher_ltol": (0.0, 1.0),
            "matcher_stol": (0.0, 1.0),
            "matcher_angle_tol_deg": (0.0, 30.0),
            "minimum_distance_angstrom": (0.0, 5.0),
        }
        for name, (lower, upper) in numeric_ranges.items():
            value = parameters.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not lower < float(value) <= upper
            ):
                diagnostics.append(
                    _diagnostic(
                        "ERROR", f"parameter.{name}", f"{name} must be in ({lower}, {upper}]"
                    )
                )
        max_structures = parameters.get("max_structures")
        if not _positive_int(max_structures) or int(max_structures) > 10000:
            diagnostics.append(
                _diagnostic("ERROR", "parameter.max_structures", "max_structures must be 1..10000")
            )
        if set(resources) != {"python_executable"}:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "resource.python_executable",
                    "merge resources must contain only an explicit python_executable",
                )
            )
        elif _explicit_executable(resources.get("python_executable"), project_root) is None:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "resource.python_executable",
                    "python_executable must be an absolute non-symlink executable file",
                )
            )
        if not _ordinary_file(BUNDLED_MERGE_WRAPPER) or BUNDLED_MERGE_WRAPPER.is_symlink():
            diagnostics.append(
                _diagnostic("ERROR", "path.bundled_merge_wrapper", "structure_merge.py is missing")
            )
        return diagnostics

    def _plan_merge(self, context: Any) -> Dict[str, Any]:
        diagnostics = self._validate_merge(context)
        if _has_errors(diagnostics):
            return _blocked(diagnostics)
        project_root = _resolve(context["project_root"], Path.cwd())
        attempt_dir = _resolve(context["attempt_dir"], project_root)
        inputs = _mapping(context["inputs"])
        parameters = _mapping(context["parameters"])
        resources = _mapping(context["resources"])
        assert project_root is not None and attempt_dir is not None
        direct = _resolve_nonsymlink(inputs["direct_manifest"], project_root)
        lasp_manifest = _resolve_nonsymlink(inputs["lasp_selected_manifest"], project_root)
        lasp_archive = _resolve_nonsymlink(inputs["lasp_selected_archive"], project_root)
        python_executable = _explicit_executable(resources["python_executable"], project_root)
        assert direct is not None and lasp_manifest is not None and lasp_archive is not None
        assert python_executable is not None
        output_dir = (attempt_dir / str(parameters.get("output_subdir", "merged-structures"))).resolve()
        script = BUNDLED_MERGE_WRAPPER.resolve()
        argv = [
            str(python_executable),
            str(script),
            "--direct-manifest",
            str(direct),
            "--lasp-selected-manifest",
            str(lasp_manifest),
            "--lasp-selected-archive",
            str(lasp_archive),
            "--output-dir",
            str(output_dir),
            "--direct-source-group-id",
            str(parameters["direct_source_group_id"]),
            "--lasp-source-group-id",
            str(parameters["lasp_source_group_id"]),
            "--matcher-ltol",
            str(parameters["matcher_ltol"]),
            "--matcher-stol",
            str(parameters["matcher_stol"]),
            "--matcher-angle-tol-deg",
            str(parameters["matcher_angle_tol_deg"]),
            "--minimum-distance-angstrom",
            str(parameters["minimum_distance_angstrom"]),
            "--max-structures",
            str(parameters["max_structures"]),
        ]
        return {
            "plugin_id": PLUGIN_ID,
            "operation": MERGE_OPERATION,
            "status": "READY",
            "executable": True,
            "argv": argv,
            "cwd": str(attempt_dir),
            "shell": False,
            "expected_outputs": [str(output_dir / "structures.json")],
            "output_dir": str(output_dir),
            "approval_summary": {
                "expensive": False,
                "submits_jobs": False,
                "input_structure_bound": parameters["max_structures"],
                "near_duplicate_matcher": {
                    "ltol": parameters["matcher_ltol"],
                    "stol": parameters["matcher_stol"],
                    "angle_tol_deg": parameters["matcher_angle_tol_deg"],
                    "scale": False,
                },
                "minimum_distance_angstrom": parameters["minimum_distance_angstrom"],
            },
            "input_paths": {
                "direct_manifest": str(direct),
                "lasp_selected_manifest": str(lasp_manifest),
                "lasp_selected_archive": str(lasp_archive),
                "merge_wrapper": str(script),
                "python_executable": str(python_executable),
            },
            "diagnostics": diagnostics,
        }

    def _validate_lasp(self, context: Any) -> List[Dict[str, str]]:
        diagnostics: List[Dict[str, str]] = []
        if not isinstance(context, Mapping):
            return [_diagnostic("ERROR", "context.mapping_required", "context must be a mapping")]

        operation = _operation(context)
        if operation not in LASP_OPERATIONS:
            return [
                _diagnostic("ERROR", "operation.unsupported", "unsupported LASP/SSW operation")
            ]
        project_root = _resolve(context.get("project_root"), Path.cwd())
        if project_root is None or not project_root.is_dir():
            diagnostics.append(
                _diagnostic(
                    "ERROR", "path.project_root", "project_root must name an existing directory"
                )
            )
            project_root = Path.cwd().resolve()
        attempt_dir = _resolve(context.get("attempt_dir"), project_root)
        if attempt_dir is None:
            diagnostics.append(
                _diagnostic("ERROR", "path.attempt_dir", "attempt_dir must be an explicit path")
            )
            attempt_dir = project_root / ".mlipflow-invalid-attempt"
        elif not _is_within(attempt_dir, project_root):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "path.attempt_outside_project",
                    "attempt_dir must stay inside project_root",
                )
            )
        if context.get("backend", "local") != "local":
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "backend.local_only",
                    "LASP/SSW adapter execution is local-only; it never submits scheduler jobs",
                )
            )

        inputs = _mapping(context.get("inputs"))
        parameters = _mapping(context.get("parameters"))
        resources = _mapping(context.get("resources"))
        if not isinstance(context.get("inputs", {}), Mapping):
            diagnostics.append(
                _diagnostic("ERROR", "inputs.mapping_required", "inputs must be a mapping")
            )
        if not isinstance(context.get("parameters", {}), Mapping):
            diagnostics.append(
                _diagnostic("ERROR", "parameters.mapping_required", "parameters must be a mapping")
            )
        if not isinstance(context.get("resources", {}), Mapping):
            diagnostics.append(
                _diagnostic("ERROR", "resources.mapping_required", "resources must be a mapping")
            )
        unknown_resources = sorted(set(resources) - {"python_executable", "mpi_launcher"})
        if unknown_resources:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "resource.unknown",
                    "unsupported LASP resource(s): %s" % ", ".join(unknown_resources),
                )
            )
        unknown_parameters = sorted(set(parameters) - _LASP_PARAMETERS)
        if unknown_parameters:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "parameter.unknown",
                    "unsupported parameter(s): %s" % ", ".join(unknown_parameters),
                )
            )

        allowed_inputs = (
            {"historical_run_dir", "lasp_input", "result_manifest"}
            if operation == "lasp-ssw-normalize-replay"
            else {
                "lasp_executable",
                "input_structure",
                "lasp_input",
                "lasp_auxiliary_files",
                "result_manifest",
            }
        )
        unknown_inputs = sorted(set(inputs) - allowed_inputs)
        if unknown_inputs:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "input.unknown",
                    "unsupported LASP input(s): %s" % ", ".join(unknown_inputs),
                )
            )

        if not _ordinary_file(BUNDLED_LASP_WRAPPER):
            diagnostics.append(
                _diagnostic(
                    "ERROR", "path.lasp_wrapper", "bundled lasp_ssw.py wrapper is missing"
                )
            )
        lasp_input = _resolve_nonsymlink(inputs.get("lasp_input"), project_root)
        if not _ordinary_file(lasp_input):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "path.lasp_input",
                    "inputs.lasp_input must name an ordinary non-symlink lasp.in file",
                )
            )
        else:
            assert lasp_input is not None
            try:
                _validate_ssw_input(_parse_lasp_input(lasp_input))
            except (OSError, UnicodeError, ValueError) as exc:
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "input.lasp_ssw_contract", f"invalid LASP/SSW input: {exc}"
                    )
                )

        output_subdir = parameters.get("output_subdir", "lasp-ssw")
        if not _safe_subdirectory(output_subdir):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "path.output_subdir",
                    "output_subdir must be a non-empty relative path without '..'",
                )
            )
            output_dir = attempt_dir / "lasp-ssw"
        else:
            output_dir = (attempt_dir / str(output_subdir)).resolve()
            if not _is_within(output_dir, attempt_dir):
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "path.output_escape", "LASP output must stay inside attempt_dir"
                    )
                )
        if output_dir.exists() or output_dir.is_symlink():
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "path.output_exists",
                    "LASP/SSW requires a fresh attempt/output_subdir",
                )
            )

        if operation == "lasp-ssw-normalize-replay":
            source_dir = _resolve_nonsymlink(inputs.get("historical_run_dir"), project_root)
            if source_dir is None or not source_dir.is_dir():
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "path.historical_run_dir",
                        "inputs.historical_run_dir must be an ordinary non-symlink directory",
                    )
                )
            else:
                allstr = source_dir / "allstr.arc"
                if _has_symlink_component(allstr) or not _ordinary_file(allstr):
                    diagnostics.append(
                        _diagnostic(
                            "ERROR",
                            "path.allstr_arc",
                            "historical_run_dir must contain ordinary allstr.arc",
                        )
                    )
                if _is_within(output_dir, source_dir) or _is_within(source_dir, output_dir):
                    diagnostics.append(
                        _diagnostic(
                            "ERROR",
                            "path.input_output_overlap",
                            "historical input and output directories must not overlap",
                        )
                    )
                for flag, name in (
                    (parameters.get("include_best_arc", False), "best.arc"),
                    (parameters.get("include_md_arc", False), "md.arc"),
                ):
                    if flag is True and (
                        _has_symlink_component(source_dir / name)
                        or not _ordinary_file(source_dir / name)
                    ):
                        diagnostics.append(
                            _diagnostic(
                                "ERROR", "path.%s" % name.replace(".", "_"), f"missing {name}"
                            )
                        )
            if parameters.get("lasp_version") not in (None, ""):
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "parameter.lasp_version_replay",
                        "lasp_version is execute-only; replay records the historical input exactly",
                    )
                )
            if parameters.get("mpi_processes") is not None or resources.get(
                "mpi_launcher"
            ) is not None:
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "resource.mpi_replay", "MPI settings are not used by replay"
                    )
                )
        else:
            executable = _resolve_nonsymlink(inputs.get("lasp_executable"), project_root)
            if not _ordinary_file(executable):
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "path.lasp_executable",
                        "inputs.lasp_executable must name a user-supplied ordinary LASP binary",
                    )
                )
            elif executable.stat().st_mode & 0o111 == 0:
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "path.lasp_not_executable", "LASP binary is not executable"
                    )
                )
            input_structure = _resolve_nonsymlink(inputs.get("input_structure"), project_root)
            if not _ordinary_file(input_structure):
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "path.input_structure",
                        "inputs.input_structure must name an ordinary ARC structure file",
                    )
                )
            auxiliary = inputs.get("lasp_auxiliary_files", {})
            if not isinstance(auxiliary, Mapping):
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "input.lasp_auxiliary_files",
                        "lasp_auxiliary_files must map safe destination basenames to files",
                    )
                )
            else:
                seen = set()
                reserved = {
                    "input.arc",
                    "lasp.in",
                    "all.arc",
                    "allstr.arc",
                    "allstr.native.arc",
                    "best.arc",
                    "md.arc",
                    "sampling-result.json",
                }
                for name, raw_path in auxiliary.items():
                    if (
                        not isinstance(name, str)
                        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name)
                        or name in reserved
                        or name in seen
                    ):
                        diagnostics.append(
                            _diagnostic(
                                "ERROR",
                                "input.auxiliary_name",
                                "auxiliary destinations must be unique safe non-reserved basenames",
                            )
                        )
                        break
                    seen.add(name)
                    if not _ordinary_file(_resolve_nonsymlink(raw_path, project_root)):
                        diagnostics.append(
                            _diagnostic(
                                "ERROR",
                                "path.auxiliary_file",
                                f"auxiliary input {name} is missing or is a symlink",
                            )
                        )
                        break
            if not _plain_string(parameters.get("lasp_version")):
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "parameter.lasp_version",
                        "execute requires an explicit single-line lasp_version",
                    )
                )
            launcher = resources.get("mpi_launcher")
            processes = parameters.get("mpi_processes")
            if launcher is not None:
                launcher_path = _resolve_nonsymlink(launcher, project_root)
                if (
                    not _plain_string(launcher)
                    or launcher_path is None
                    or not _ordinary_file(launcher_path)
                    or launcher_path.name not in {"mpirun", "mpiexec"}
                    or launcher_path.stat().st_mode & 0o111 == 0
                ):
                    diagnostics.append(
                        _diagnostic(
                            "ERROR",
                            "resource.mpi_launcher",
                            "mpi_launcher must be an explicit ordinary executable path named mpirun or mpiexec",
                        )
                    )
                if not _positive_int(processes):
                    diagnostics.append(
                        _diagnostic(
                            "ERROR",
                            "parameter.mpi_processes",
                            "mpi_processes must be positive when mpi_launcher is set",
                        )
                    )
            elif processes is not None:
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "parameter.mpi_without_launcher",
                        "mpi_processes requires resources.mpi_launcher",
                    )
                )

        source_id = parameters.get("historical_source_id")
        if not _portable_source_id(source_id):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "parameter.historical_source_id",
                    "historical_source_id must be portable and must not expose an absolute path",
                )
            )
        if not _positive_int(parameters.get("selection_stride")):
            diagnostics.append(
                _diagnostic(
                    "ERROR", "parameter.selection_stride", "selection_stride must be positive"
                )
            )
        max_frames = parameters.get("max_frames")
        if not _positive_int(max_frames) or (
            isinstance(max_frames, int) and max_frames > MAX_LASP_FRAMES
        ):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "parameter.max_frames",
                    f"max_frames must be a positive integer no greater than {MAX_LASP_FRAMES}",
                )
            )
        energy_max = parameters.get("energy_max_ev")
        if energy_max is not None and (
            not isinstance(energy_max, (int, float))
            or isinstance(energy_max, bool)
            or not math.isfinite(float(energy_max))
        ):
            diagnostics.append(
                _diagnostic(
                    "ERROR", "parameter.energy_max_ev", "energy_max_ev must be finite or null"
                )
            )
        for name in ("include_best_arc", "include_md_arc"):
            if not isinstance(parameters.get(name, False), bool):
                diagnostics.append(
                    _diagnostic("ERROR", f"parameter.{name}", f"{name} must be boolean")
                )
        if parameters.get("preserve_historical_order") is not True:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "parameter.preserve_historical_order",
                    "preserve_historical_order must be true",
                )
            )
        if parameters.get("seed_status") != UNKNOWN:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "seed.historical_unknown",
                    f"seed_status must be {UNKNOWN}; the reviewed source did not record a seed",
                )
            )
        if parameters.get("acknowledge_uncontrolled_seed") is not True:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "seed.acknowledgement_required",
                    "acknowledge_uncontrolled_seed must be true",
                )
            )
        else:
            diagnostics.append(
                _diagnostic(
                    "WARNING",
                    "seed.not_reconstructed",
                    "the historical LASP seed is unknown and is not reconstructed",
                )
            )
        python_executable = resources.get("python_executable")
        if _explicit_executable(python_executable, project_root) is None:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "resource.python_executable",
                    "python_executable must be an explicit absolute, executable, non-symlink file",
                )
            )
        return diagnostics

    def _plan_lasp(self, context: Any) -> Dict[str, Any]:
        diagnostics = self._validate_lasp(context)
        if _has_errors(diagnostics):
            return _blocked(diagnostics)
        project_root = _resolve(context["project_root"], Path.cwd())
        attempt_dir = _resolve(context["attempt_dir"], project_root)
        inputs = _mapping(context["inputs"])
        parameters = _mapping(context["parameters"])
        resources = _mapping(context.get("resources"))
        assert project_root is not None and attempt_dir is not None
        operation = _operation(context)
        output_dir = (
            attempt_dir / str(parameters.get("output_subdir", "lasp-ssw"))
        ).resolve()
        input_paths: Dict[str, str] = {}

        def remember_input(name: str, path: Path) -> None:
            input_paths[name] = str(path)

        python_executable = _explicit_executable(
            resources.get("python_executable"),
            project_root,
        )
        assert python_executable is not None
        wrapper = BUNDLED_LASP_WRAPPER.resolve()
        remember_input("python_executable", python_executable)
        remember_input("lasp_wrapper", wrapper)
        argv = [
            str(python_executable),
            str(wrapper),
            "normalize-replay"
            if operation == "lasp-ssw-normalize-replay"
            else "execute",
        ]
        if operation == "lasp-ssw-normalize-replay":
            source_dir = _resolve_nonsymlink(inputs["historical_run_dir"], project_root)
            assert source_dir is not None
            argv.extend(["--historical-run-dir", str(source_dir)])
            remember_input("allstr.arc", source_dir / "allstr.arc")
            if parameters.get("include_best_arc", False):
                remember_input("best.arc", source_dir / "best.arc")
            if parameters.get("include_md_arc", False):
                remember_input("md.arc", source_dir / "md.arc")
        else:
            executable = _resolve_nonsymlink(inputs["lasp_executable"], project_root)
            structure = _resolve_nonsymlink(inputs["input_structure"], project_root)
            assert executable is not None and structure is not None
            argv.extend(
                [
                    "--lasp-executable",
                    str(executable),
                    "--input-structure",
                    str(structure),
                    "--lasp-version",
                    str(parameters["lasp_version"]),
                ]
            )
            remember_input("lasp_executable", executable)
            remember_input("input_structure", structure)
            launcher = resources.get("mpi_launcher")
            if launcher is not None:
                launcher_path = _resolve_nonsymlink(launcher, project_root)
                assert launcher_path is not None
                argv.extend(
                    [
                        "--mpi-launcher",
                        str(launcher_path),
                        "--mpi-processes",
                        str(parameters["mpi_processes"]),
                    ]
                )
                remember_input("mpi_launcher", launcher_path)
            auxiliary = _mapping(inputs.get("lasp_auxiliary_files"))
            for name in sorted(auxiliary):
                path = _resolve_nonsymlink(auxiliary[name], project_root)
                assert path is not None
                argv.extend(["--auxiliary", name, str(path)])
                remember_input(f"auxiliary:{name}", path)
        lasp_input = _resolve_nonsymlink(inputs["lasp_input"], project_root)
        assert lasp_input is not None
        remember_input("lasp_input", lasp_input)
        argv.extend(
            [
                "--lasp-input",
                str(lasp_input),
                "--output-dir",
                str(output_dir),
                "--historical-source-id",
                str(parameters["historical_source_id"]),
                "--selection-stride",
                str(parameters["selection_stride"]),
                "--max-frames",
                str(parameters["max_frames"]),
                "--seed-status",
                UNKNOWN,
            ]
        )
        if parameters.get("energy_max_ev") is not None:
            argv.extend(["--energy-max-ev", str(parameters["energy_max_ev"])])
        if parameters.get("include_best_arc", False):
            argv.append("--include-best-arc")
        if parameters.get("include_md_arc", False):
            argv.append("--include-md-arc")
        return {
            "plugin_id": PLUGIN_ID,
            "operation": operation,
            "status": "READY",
            "executable": True,
            "argv": argv,
            "cwd": str(attempt_dir),
            "shell": False,
            "expected_outputs": [str(output_dir / "sampling-result.json")],
            "output_dir": str(output_dir),
            "input_paths": input_paths,
            "assumptions": {
                "scientific_mode": (
                    "historical-replay"
                    if operation == "lasp-ssw-normalize-replay"
                    else "execute"
                ),
                "seed_status": UNKNOWN,
                "selection_stride_applies_after_energy_filter": True,
                "preserve_historical_order": True,
                "fresh_output_directory_required": True,
                "scheduler_submission": False,
            },
            "diagnostics": diagnostics,
        }


    def _manifest_path(self, context: Any) -> Optional[Path]:
        if not isinstance(context, Mapping):
            return None
        inputs = _mapping(context.get("inputs"))
        project_root = _resolve(context.get("project_root"), Path.cwd()) or Path.cwd().resolve()
        execution = _mapping(context.get("execution"))
        plan = _mapping(execution.get("plan"))
        expected = plan.get("expected_outputs")
        if isinstance(expected, list) and len(expected) == 1:
            return _unresolved(expected[0], project_root)
        explicit = inputs.get("result_manifest")
        if explicit is not None:
            return _unresolved(explicit, project_root)
        return None

    def _read_direct_manifest(
        self, context: Any
    ) -> Tuple[Optional[Path], List[Dict[str, str]], List[Dict[str, str]]]:
        path = self._manifest_path(context)
        diagnostics: List[Dict[str, str]] = []
        if path is None:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "result.manifest_explicit_required",
                    "result manifest must come from execution.plan.expected_outputs or inputs.result_manifest",
                )
            )
            return None, [], diagnostics
        if path.is_symlink():
            diagnostics.append(
                _diagnostic("ERROR", "result.manifest_symlink", "manifest must not be a symlink")
            )
            return path, [], diagnostics
        if not path.is_file():
            diagnostics.append(
                _diagnostic("INFO", "result.manifest_missing", "DIRECT manifest does not exist yet")
            )
            return path, [], diagnostics
        try:
            reader = csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")))
            rows = [dict(row) for row in reader]
        except (OSError, UnicodeError, csv.Error) as exc:
            diagnostics.append(
                _diagnostic(
                    "ERROR", "result.manifest_unreadable", "cannot read manifest.csv: %s" % exc
                )
            )
            return path, [], diagnostics
        if reader.fieldnames is None or not _DIRECT_COLUMNS.issubset(set(reader.fieldnames)):
            diagnostics.append(
                _diagnostic(
                    "ERROR", "result.manifest_columns", "manifest.csv has an unexpected header"
                )
            )
        if not rows:
            diagnostics.append(
                _diagnostic(
                    "ERROR", "result.no_samples", "manifest.csv contains no selected structures"
                )
            )
        manifest_dir = path.parent.resolve()
        seen_orders = set()
        for index, row in enumerate(rows):
            try:
                order = int(row.get("selected_order", ""))
                int(row.get("input_index", ""))
                int(row.get("frame_index", ""))
            except (TypeError, ValueError):
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "result.manifest_integer",
                        "manifest row %d has invalid integer fields" % (index + 1),
                    )
                )
                continue
            if order < 1 or order in seen_orders:
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "result.selected_order",
                        "selected_order values must be unique and positive",
                    )
                )
            seen_orders.add(order)
            output_file = _resolve(row.get("output_file"), manifest_dir)
            if output_file is None or not _is_within(output_file, manifest_dir):
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "result.output_escape",
                        "selected output must stay beside manifest.csv",
                    )
                )
            elif not output_file.is_file() or output_file.stat().st_size == 0:
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "result.output_missing",
                        "selected structure is missing or empty: %s" % output_file,
                    )
                )
        return path, rows, diagnostics

    def _check_direct(self, context: Any) -> Dict[str, Any]:
        execution = _mapping(context.get("execution")) if isinstance(context, Mapping) else {}
        returncode = execution.get("returncode")
        if returncode is not None and returncode != 0:
            return {
                "plugin_id": PLUGIN_ID,
                "status": "FAIL",
                "diagnostics": [
                    _diagnostic(
                        "ERROR", "execution.nonzero", "DIRECT process returned %s" % returncode
                    )
                ],
            }
        path, rows, diagnostics = self._read_direct_manifest(context)
        if path is not None and not path.exists() and not _has_errors(diagnostics):
            return {"plugin_id": PLUGIN_ID, "status": "WAIT", "diagnostics": diagnostics}
        if _has_errors(diagnostics):
            return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
        return {
            "plugin_id": PLUGIN_ID,
            "status": "OK",
            "result_file": str(path),
            "selected_structure_count": len(rows),
            "diagnostics": diagnostics,
        }

    def _collect_direct(self, context: Any) -> Dict[str, Any]:
        checked = self._check_direct(context)
        if checked.get("status") != "OK":
            return {
                "plugin_id": PLUGIN_ID,
                "status": checked.get("status", "FAIL"),
                "artifacts": [],
                "metrics": {},
                "diagnostics": checked.get("diagnostics", []),
            }
        manifest, rows, diagnostics = self._read_direct_manifest(context)
        assert manifest is not None
        artifacts: List[Dict[str, str]] = [
            {"role": "sample-manifest", "path": str(manifest), "media_type": "text/csv"}
        ]
        formulas = set()
        sources = set()
        manifest_dir = manifest.parent.resolve()
        for row in rows:
            output_file = _resolve(row["output_file"], manifest_dir)
            assert output_file is not None
            artifacts.append(
                {
                    "role": "selected-structure",
                    "path": str(output_file),
                    "media_type": "chemical/x-vasp-poscar",
                }
            )
            formulas.add(row.get("formula", ""))
            sources.add(row.get("source_file", ""))
        for name in _OPTIONAL_PLOTS:
            path = manifest_dir / name
            if path.is_file():
                artifacts.append(
                    {
                        "role": "sampling-diagnostic",
                        "path": str(path),
                        "media_type": "text/html" if path.suffix == ".html" else "image/png",
                    }
                )
        return {
            "plugin_id": PLUGIN_ID,
            "status": "OK",
            "artifacts": artifacts,
            "metrics": {
                "selected_structure_count": len(rows),
                "unique_formula_count": len(formulas),
                "source_file_count": len(sources),
            },
            "diagnostics": diagnostics,
        }

    def _read_merge_manifest(
        self, context: Any
    ) -> Tuple[Optional[Path], Dict[str, Any], List[Dict[str, str]], List[Dict[str, Any]]]:
        path = self._manifest_path(context)
        diagnostics: List[Dict[str, str]] = []
        if path is None:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "result.manifest_explicit_required",
                    "merge result manifest must come from the execution plan or result_manifest",
                )
            )
            return None, {}, diagnostics, []
        if path.is_symlink():
            diagnostics.append(
                _diagnostic("ERROR", "result.manifest_symlink", "merge manifest must not be a symlink")
            )
            return path, {}, diagnostics, []
        if not path.is_file():
            diagnostics.append(
                _diagnostic("INFO", "result.manifest_missing", "merge structures.json does not exist yet")
            )
            return path, {}, diagnostics, []
        manifest = self._read_json_object(path, "merge_manifest", diagnostics)
        if manifest is None:
            return path, {}, diagnostics, []
        if (
            manifest.get("schema_version") != 1
            or manifest.get("operation") != MERGE_OPERATION
            or manifest.get("status") != "OK"
        ):
            diagnostics.append(
                _diagnostic("ERROR", "result.merge_contract", "merge manifest schema/status is invalid")
            )
        records = manifest.get("structures")
        if not isinstance(records, list) or not records:
            diagnostics.append(
                _diagnostic("ERROR", "result.structures", "merge manifest must contain structures")
            )
            records = []
        counts = _mapping(manifest.get("counts"))
        duplicates = manifest.get("duplicates")
        rejected = manifest.get("rejected")
        if not isinstance(duplicates, list) or not isinstance(rejected, list):
            diagnostics.append(
                _diagnostic("ERROR", "result.lineage_lists", "duplicates/rejected must be lists")
            )
            duplicates = []
            rejected = []
        expected_input = len(records) + len(duplicates) + len(rejected)
        if (
            counts.get("input") != expected_input
            or counts.get("unique") != len(records)
            or counts.get("rejected_bad") != len(rejected)
            or counts.get("exact_duplicates")
            != sum(isinstance(item, Mapping) and item.get("duplicate_kind") == "exact" for item in duplicates)
            or counts.get("near_duplicates")
            != sum(isinstance(item, Mapping) and item.get("duplicate_kind") == "near" for item in duplicates)
        ):
            diagnostics.append(
                _diagnostic("ERROR", "result.counts", "merge counts do not match manifest records")
            )
        input_artifacts = _mapping(manifest.get("input_artifacts"))
        inputs = _mapping(context.get("inputs")) if isinstance(context, Mapping) else {}
        project_root = _resolve(context.get("project_root"), Path.cwd()) if isinstance(context, Mapping) else None
        if project_root is not None:
            for key in ("direct_manifest", "lasp_selected_manifest", "lasp_selected_archive"):
                source = _resolve_nonsymlink(inputs.get(key), project_root)
                if source is None or not _ordinary_file(source) or input_artifacts.get(key) != str(source):
                    diagnostics.append(
                        _diagnostic("ERROR", f"result.input_{key}", f"merge input {key} path differs")
                    )
        seen_ids: set[str] = set()
        seen_paths: set[str] = set()
        for index, record in enumerate(records, 1):
            prefix = f"structure {index}"
            if not isinstance(record, Mapping):
                diagnostics.append(_diagnostic("ERROR", "result.structure_record", f"{prefix} is invalid"))
                continue
            structure_id = record.get("id")
            relative = record.get("path")
            methods = record.get("source_sampling_methods")
            sources = record.get("source_records")
            if (
                structure_id != f"structure-{index:06d}"
                or structure_id in seen_ids
            ):
                diagnostics.append(_diagnostic("ERROR", "result.structure_id", f"{prefix} id is invalid"))
            else:
                seen_ids.add(structure_id)
            if (
                not isinstance(relative, str)
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or relative in seen_paths
            ):
                diagnostics.append(_diagnostic("ERROR", "result.structure_path", f"{prefix} path is invalid"))
                continue
            seen_paths.add(relative)
            output = (path.parent / relative).absolute()
            if (
                not _is_within(output, path.parent.resolve())
                or output.is_symlink()
                or not output.is_file()
                or output.stat().st_size < 1
                or output.stat().st_size > MAX_ARTIFACT_BYTES
            ):
                diagnostics.append(
                    _diagnostic("ERROR", "result.structure_file", f"{prefix} file is invalid")
                )
            if (
                not isinstance(methods, list)
                or not methods
                or any(method not in {"DIRECT", "LASP_SSW"} for method in methods)
                or not isinstance(sources, list)
                or not sources
                or sorted({item.get("sampling_method") for item in sources if isinstance(item, Mapping)})
                != methods
            ):
                diagnostics.append(
                    _diagnostic("ERROR", "result.source_provenance", f"{prefix} source provenance is invalid")
                )
        if (path.parent / "INCOMPLETE.json").exists():
            diagnostics.append(
                _diagnostic("ERROR", "result.incomplete_marker", "merge output remains incomplete")
            )
        return path, manifest, diagnostics, [dict(item) for item in records if isinstance(item, Mapping)]

    def _check_merge(self, context: Any) -> Dict[str, Any]:
        execution = _mapping(context.get("execution")) if isinstance(context, Mapping) else {}
        returncode = execution.get("returncode")
        if returncode is not None and returncode != 0:
            return {
                "plugin_id": PLUGIN_ID,
                "operation": MERGE_OPERATION,
                "status": "FAIL",
                "diagnostics": [
                    _diagnostic("ERROR", "execution.nonzero", f"merge-structures returned {returncode}")
                ],
            }
        path, manifest, diagnostics, records = self._read_merge_manifest(context)
        if path is not None and not path.exists() and not _has_errors(diagnostics):
            return {"plugin_id": PLUGIN_ID, "operation": MERGE_OPERATION, "status": "WAIT", "diagnostics": diagnostics}
        if _has_errors(diagnostics):
            return {"plugin_id": PLUGIN_ID, "operation": MERGE_OPERATION, "status": "FAIL", "diagnostics": diagnostics}
        return {
            "plugin_id": PLUGIN_ID,
            "operation": MERGE_OPERATION,
            "status": "OK",
            "result_file": str(path),
            "counts": manifest.get("counts", {}),
            "structure_count": len(records),
            "diagnostics": diagnostics,
        }

    def _collect_merge(self, context: Any) -> Dict[str, Any]:
        checked = self._check_merge(context)
        if checked.get("status") != "OK":
            return {
                "plugin_id": PLUGIN_ID,
                "operation": MERGE_OPERATION,
                "status": checked.get("status", "FAIL"),
                "artifacts": [],
                "metrics": {},
                "diagnostics": checked.get("diagnostics", []),
            }
        manifest, value, diagnostics, records = self._read_merge_manifest(context)
        assert manifest is not None
        artifacts: List[Dict[str, str]] = [
            {"role": "structures-manifest", "path": str(manifest), "media_type": "application/json"}
        ]
        for record in records:
            artifacts.append(
                {
                    "role": "merged-structure",
                    "path": str((manifest.parent / record["path"]).resolve()),
                    "media_type": "chemical/x-vasp-poscar",
                }
            )
        counts = _mapping(value.get("counts"))
        return {
            "plugin_id": PLUGIN_ID,
            "operation": MERGE_OPERATION,
            "status": "OK",
            "artifacts": artifacts,
            "metrics": {
                "input_structure_count": counts.get("input", 0),
                "unique_structure_count": counts.get("unique", 0),
                "exact_duplicate_count": counts.get("exact_duplicates", 0),
                "near_duplicate_count": counts.get("near_duplicates", 0),
                "rejected_bad_structure_count": counts.get("rejected_bad", 0),
            },
            "diagnostics": diagnostics,
        }

    def _read_json_object(
        self, path: Path, field: str, diagnostics: List[Dict[str, str]]
    ) -> Optional[Dict[str, Any]]:
        try:
            size = path.stat().st_size
            if size == 0 or size > MAX_JSON_BYTES:
                raise ValueError(f"size must be between 1 and {MAX_JSON_BYTES} bytes")
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("top-level JSON value must be an object")
            return value
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            diagnostics.append(
                _diagnostic("ERROR", f"result.{field}_unreadable", f"cannot read {field}: {exc}")
            )
            return None

    def _portable_artifact_path(
        self,
        manifest: Path,
        value: Any,
        field: str,
        diagnostics: List[Dict[str, str]],
    ) -> Optional[Path]:
        if not isinstance(value, str) or not value:
            diagnostics.append(
                _diagnostic("ERROR", "result.artifact_path", f"{field} requires a path")
            )
            return None
        portable = Path(value)
        if portable.is_absolute() or ".." in portable.parts:
            diagnostics.append(
                _diagnostic(
                    "ERROR", "result.artifact_escape", f"{field} must be a portable relative path"
                )
            )
            return None
        path = manifest.parent / portable
        if path.is_symlink():
            diagnostics.append(
                _diagnostic("ERROR", "result.artifact_symlink", f"{field} must not be a symlink")
            )
            return None
        resolved = path.resolve()
        if not _is_within(resolved, manifest.parent.resolve()):
            diagnostics.append(
                _diagnostic("ERROR", "result.artifact_escape", f"{field} escapes result directory")
            )
            return None
        if (
            not resolved.is_file()
            or resolved.stat().st_size == 0
            or resolved.stat().st_size > MAX_ARTIFACT_BYTES
        ):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "result.artifact_missing",
                    f"{field} is missing, empty, or exceeds {MAX_ARTIFACT_BYTES} bytes",
                )
            )
            return None
        return resolved

    def _read_lasp_result(
        self, context: Any
    ) -> Tuple[
        Optional[Path],
        Dict[str, Any],
        List[Dict[str, str]],
        List[Dict[str, Any]],
    ]:
        diagnostics: List[Dict[str, str]] = []
        manifest = self._manifest_path(context)
        if isinstance(context, Mapping):
            inputs = _mapping(context.get("inputs"))
            execution = _mapping(context.get("execution"))
            plan = _mapping(execution.get("plan"))
            expected = plan.get("expected_outputs")
            explicit = inputs.get("result_manifest")
            if explicit is not None and isinstance(expected, list) and len(expected) == 1:
                project_root = (
                    _resolve(context.get("project_root"), Path.cwd()) or Path.cwd().resolve()
                )
                explicit_path = _resolve(explicit, project_root)
                expected_path = _resolve(expected[0], project_root)
                if explicit_path != expected_path:
                    diagnostics.append(
                        _diagnostic(
                            "ERROR",
                            "result.manifest_plan_mismatch",
                            "inputs.result_manifest must equal execution.plan.expected_outputs[0]",
                        )
                    )
        if manifest is None:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "result.manifest_explicit_required",
                    "sampling-result.json must come from the execution plan or inputs.result_manifest",
                )
            )
            return None, {}, diagnostics, []
        if manifest.is_symlink():
            diagnostics.append(
                _diagnostic(
                    "ERROR", "result.manifest_symlink", "sampling-result.json must not be a symlink"
                )
            )
            return manifest, {}, diagnostics, []
        if not manifest.is_file():
            diagnostics.append(
                _diagnostic(
                    "INFO", "result.manifest_missing", "sampling-result.json does not exist yet"
                )
            )
            return manifest, {}, diagnostics, []
        result = self._read_json_object(manifest, "sampling_result", diagnostics)
        if result is None:
            return manifest, {}, diagnostics, []
        operation = _operation(context)
        expected_mode = (
            "historical-replay"
            if operation == "lasp-ssw-normalize-replay"
            else "execute"
        )
        for key, expected in (
            ("schema_version", 1),
            ("plugin_id", PLUGIN_ID),
            ("operation", operation),
            ("status", "OK"),
            ("mode", expected_mode),
        ):
            if result.get(key) != expected:
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "result.schema",
                        f"sampling result {key} must be {expected!r}",
                    )
                )
        context_parameters = _mapping(context.get("parameters")) if isinstance(context, Mapping) else {}
        if result.get("source_id") != context_parameters.get("historical_source_id"):
            diagnostics.append(
                _diagnostic(
                    "ERROR", "result.source_id", "sampling result source_id does not match the plan"
                )
            )
        result_parameters = result.get("parameters")
        if not isinstance(result_parameters, Mapping):
            diagnostics.append(
                _diagnostic("ERROR", "result.parameters", "result parameters must be an object")
            )
            result_parameters = {}
        expected_energy = context_parameters.get("energy_max_ev")
        for key, expected in (
            ("seed_status", UNKNOWN),
            ("selection_stride", context_parameters.get("selection_stride")),
            ("energy_max_ev", expected_energy),
            ("stride_applies_after_energy_filter", True),
            ("preserve_historical_order", True),
        ):
            if result_parameters.get(key) != expected:
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "result.parameter",
                        f"result parameter {key} does not match the approved plan",
                    )
                )

        counts = result.get("counts")
        required_count_keys = {
            "generated_structure_count",
            "energy_accepted_count",
            "selected_structure_count",
            "aimd_seed_candidate_count",
            "md_structure_count",
        }
        if not isinstance(counts, Mapping) or required_count_keys - set(counts):
            diagnostics.append(
                _diagnostic("ERROR", "result.counts", "result counts are incomplete")
            )
            counts = {}
        else:
            for name in required_count_keys:
                value = counts.get(name)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    diagnostics.append(
                        _diagnostic(
                            "ERROR", "result.count", f"result count {name} must be non-negative"
                        )
                    )
            if counts.get("selected_structure_count", 0) < 1:
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "result.no_samples", "LASP result selected no structures"
                    )
                )
            max_frames = context_parameters.get("max_frames")
            if isinstance(max_frames, int) and counts.get("generated_structure_count", 0) > max_frames:
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "result.frame_limit", "LASP result exceeds approved max_frames"
                    )
                )

        completion = result.get("completion_evidence")
        if not isinstance(completion, Mapping) or any(
            completion.get(key) is not True
            for key in (
                "allstr_arc_parsed",
                "structure_manifest_validated_on_write",
                "selected_structures_nonempty",
                "incomplete_marker_removed",
            )
        ) or completion.get("stdout_success_phrase_used") is not False:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "result.completion_evidence",
                    "completion must come from parsed files, never a stdout success phrase",
                )
            )
        if (manifest.parent / "INCOMPLETE.json").exists() or (
            manifest.parent / "INCOMPLETE.json"
        ).is_symlink():
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "result.incomplete_marker",
                    "result directory is explicitly marked INCOMPLETE",
                )
            )

        raw_artifacts = result.get("artifacts")
        artifacts: List[Dict[str, Any]] = []
        role_paths: Dict[str, List[Path]] = {}
        seen_paths = set()
        if not isinstance(raw_artifacts, list) or not raw_artifacts:
            diagnostics.append(
                _diagnostic("ERROR", "result.artifacts", "result artifacts must be a non-empty list")
            )
            raw_artifacts = []
        for index, item in enumerate(raw_artifacts):
            if not isinstance(item, Mapping) or not _plain_string(item.get("role")):
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "result.artifact_record", f"artifact {index + 1} is invalid"
                    )
                )
                continue
            if item.get("role") not in LASP_ARTIFACT_ROLES:
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "result.artifact_role_unknown",
                        f"artifact {index + 1} has an undeclared role",
                    )
                )
                continue
            path = self._portable_artifact_path(
                manifest, item.get("path"), f"artifacts[{index}]", diagnostics
            )
            if path is None:
                continue
            relative = path.relative_to(manifest.parent.resolve()).as_posix()
            if relative in seen_paths:
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "result.artifact_duplicate", f"duplicate artifact path: {relative}"
                    )
                )
            seen_paths.add(relative)
            role = str(item["role"])
            role_paths.setdefault(role, []).append(path)
            artifacts.append(
                {
                    "role": role,
                    "path": str(path),
                    "media_type": str(item.get("media_type", "application/octet-stream")),
                }
            )
        for role in (
            "ssw-structure-manifest",
            "selected-structure-manifest",
            "lasp-run-metadata",
        ):
            if len(role_paths.get(role, [])) != 1:
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "result.artifact_role", f"exactly one {role} artifact is required"
                    )
                )

        source_frames: List[Dict[str, Any]] = []
        best_source_frames: List[Dict[str, Any]] = []
        md_source_frames: List[Dict[str, Any]] = []
        metadata: Dict[str, Any] = {}
        context_inputs = _mapping(context.get("inputs")) if isinstance(context, Mapping) else {}
        context_resources = (
            _mapping(context.get("resources")) if isinstance(context, Mapping) else {}
        )
        max_frames = context_parameters.get("max_frames")
        if not _positive_int(max_frames) or int(max_frames) > MAX_LASP_FRAMES:
            max_frames = MAX_LASP_FRAMES
        source_paths: Dict[str, Path] = {}
        lasp_input_path: Optional[Path] = None
        if operation == "lasp-ssw-normalize-replay":
            project_root = (
                _resolve(context.get("project_root"), Path.cwd())
                if isinstance(context, Mapping)
                else Path.cwd().resolve()
            ) or Path.cwd().resolve()
            source_dir = _resolve_nonsymlink(
                context_inputs.get("historical_run_dir"), project_root
            )
            lasp_input_path = _resolve_nonsymlink(context_inputs.get("lasp_input"), project_root)
            if source_dir is None or not source_dir.is_dir() or lasp_input_path is None:
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "result.source_inputs_required",
                        "historical_run_dir and lasp_input are required to bind replay results",
                    )
                )
            else:
                source_paths = {
                    "lasp-input": lasp_input_path,
                    "ssw-archive": source_dir / "allstr.arc",
                }
                if context_parameters.get("include_best_arc", False):
                    source_paths["best-archive"] = source_dir / "best.arc"
                if context_parameters.get("include_md_arc", False):
                    source_paths["md-archive"] = source_dir / "md.arc"
        else:
            raw_run = manifest.parent / "raw-run"
            if _has_symlink_component(raw_run) or not raw_run.is_dir():
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "result.raw_run_required",
                        "execute result must retain an ordinary raw-run directory",
                    )
                )
            else:
                lasp_input_path = raw_run / "lasp.in"
                source_paths = {
                    "lasp-input": lasp_input_path,
                    "ssw-archive": raw_run / "allstr.arc",
                }
                if context_parameters.get("include_best_arc", False):
                    source_paths["best-archive"] = raw_run / "best.arc"
                if context_parameters.get("include_md_arc", False):
                    source_paths["md-archive"] = raw_run / "md.arc"

        for role, path in source_paths.items():
            if _has_symlink_component(path) or not _ordinary_file(path):
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "result.source_file", f"bound source file is missing: {role}"
                    )
                )
        if _ordinary_file(source_paths.get("ssw-archive")):
            try:
                source_frames = _read_arc_frames(
                    source_paths["ssw-archive"], int(max_frames)
                )
            except (OSError, UnicodeError, ValueError) as exc:
                diagnostics.append(
                    _diagnostic("ERROR", "result.source_arc", f"cannot parse allstr.arc: {exc}")
                )
        if _ordinary_file(source_paths.get("best-archive")):
            try:
                best_source_frames = _read_arc_frames(
                    source_paths["best-archive"], int(max_frames)
                )
            except (OSError, UnicodeError, ValueError) as exc:
                diagnostics.append(
                    _diagnostic("ERROR", "result.best_arc", f"cannot parse best.arc: {exc}")
                )
        if _ordinary_file(source_paths.get("md-archive")):
            try:
                md_source_frames = _read_arc_frames(source_paths["md-archive"], int(max_frames))
            except (OSError, UnicodeError, ValueError) as exc:
                diagnostics.append(
                    _diagnostic("ERROR", "result.md_arc", f"cannot parse md.arc: {exc}")
                )
        if _ordinary_file(lasp_input_path):
            try:
                parsed_lasp_input = _parse_lasp_input(lasp_input_path)
                _validate_ssw_input(parsed_lasp_input)
                if result_parameters.get("lasp_input") != parsed_lasp_input:
                    diagnostics.append(
                        _diagnostic(
                            "ERROR",
                            "result.lasp_input_parameters",
                            "parsed lasp.in parameters differ from the bound input",
                        )
                    )
            except (OSError, UnicodeError, ValueError) as exc:
                diagnostics.append(
                    _diagnostic("ERROR", "result.lasp_input", f"cannot parse lasp.in: {exc}")
                )

        if len(role_paths.get("lasp-run-metadata", [])) == 1:
            value = self._read_json_object(
                role_paths["lasp-run-metadata"][0], "lasp_run_metadata", diagnostics
            )
            if value is not None:
                metadata = value
        for key, expected in (
            ("schema_version", 1),
            ("plugin_id", PLUGIN_ID),
            ("operation", operation),
            ("mode", expected_mode),
        ):
            if metadata.get(key) != expected:
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "result.metadata_schema", f"metadata {key} is not bound"
                    )
                )
        if metadata.get("parameters") != dict(result_parameters):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "result.metadata_parameters",
                    "metadata parameters differ from sampling-result parameters",
                )
            )
        source_metadata = metadata.get("source")
        raw_source_records = (
            source_metadata.get("files") if isinstance(source_metadata, Mapping) else None
        )
        if (
            not isinstance(source_metadata, Mapping)
            or source_metadata.get("id") != result.get("source_id")
            or not isinstance(raw_source_records, list)
        ):
            diagnostics.append(
                _diagnostic(
                    "ERROR", "result.metadata_source", "metadata source record is invalid"
                )
            )
            raw_source_records = []
        source_records: Dict[str, Mapping[str, Any]] = {}
        for item in raw_source_records:
            if not isinstance(item, Mapping) or not _plain_string(item.get("role")):
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "result.metadata_source_record", "source records are invalid"
                    )
                )
                continue
            role = str(item["role"])
            if role in source_records:
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "result.metadata_source_duplicate", "source roles must be unique"
                    )
                )
            source_records[role] = item
        if set(source_records) != set(source_paths):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "result.metadata_source_set",
                    "metadata source files differ from approved source roles",
                )
            )
        for role, path in source_paths.items():
            record = source_records.get(role, {})
            if _ordinary_file(path) and (
                record.get("name") != path.name
                or record.get("path") != str(path)
            ):
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "result.metadata_source_path",
                        f"metadata does not match bound source file: {role}",
                    )
                )

        execution_metadata = metadata.get("execution")
        if not isinstance(execution_metadata, Mapping):
            diagnostics.append(
                _diagnostic("ERROR", "result.metadata_execution", "execution metadata is missing")
            )
            execution_metadata = {}
        if operation == "lasp-ssw-normalize-replay":
            if execution_metadata.get("performed") is not False:
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "result.metadata_replay", "replay must record performed=false"
                    )
                )
        else:
            project_root = (
                _resolve(context.get("project_root"), Path.cwd())
                if isinstance(context, Mapping)
                else Path.cwd().resolve()
            ) or Path.cwd().resolve()
            executable = _resolve_nonsymlink(
                context_inputs.get("lasp_executable"), project_root
            )
            input_structure = _resolve_nonsymlink(
                context_inputs.get("input_structure"), project_root
            )
            approved_lasp_input = _resolve_nonsymlink(
                context_inputs.get("lasp_input"), project_root
            )
            if not all(
                _ordinary_file(path)
                for path in (executable, input_structure, approved_lasp_input)
            ):
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "result.execute_inputs_required",
                        "execute checker requires the approved executable and staged inputs",
                    )
                )
            else:
                assert executable is not None
                if (
                    execution_metadata.get("performed") is not True
                    or execution_metadata.get("version") != context_parameters.get("lasp_version")
                    or execution_metadata.get("executable_path") != str(executable)
                    or execution_metadata.get("returncode") != 0
                    or execution_metadata.get("command_uses_shell") is not False
                ):
                    diagnostics.append(
                        _diagnostic(
                            "ERROR",
                            "result.executable",
                            "execution metadata does not match the approved LASP executable",
                        )
                    )
            expected_staged: List[Dict[str, Any]] = []
            if _ordinary_file(input_structure):
                assert input_structure is not None
                expected_staged.append(
                    {
                        "role": "input-structure",
                        "destination": "input.arc",
                        "source": str(input_structure),
                    }
                )
            if _ordinary_file(approved_lasp_input):
                assert approved_lasp_input is not None
                expected_staged.append(
                    {
                        "role": "lasp-input",
                        "destination": "lasp.in",
                        "source": str(approved_lasp_input),
                    }
                )
            auxiliary = _mapping(context_inputs.get("lasp_auxiliary_files"))
            for name in sorted(auxiliary):
                path = _resolve_nonsymlink(auxiliary[name], project_root)
                if not _ordinary_file(path):
                    diagnostics.append(
                        _diagnostic(
                            "ERROR", "result.auxiliary_input", f"approved auxiliary {name} is missing"
                        )
                    )
                    continue
                assert path is not None
                expected_staged.append(
                    {
                        "role": "auxiliary-input",
                        "destination": name,
                        "source": str(path),
                    }
                )
            raw_staged = execution_metadata.get("staged_inputs")
            staged_valid = isinstance(raw_staged, list) and all(
                isinstance(item, Mapping) for item in raw_staged
            )
            normalized_staged = (
                [dict(item) for item in raw_staged] if staged_valid else []
            )
            if not staged_valid or sorted(
                normalized_staged, key=lambda item: str(item.get("destination"))
            ) != sorted(expected_staged, key=lambda item: item["destination"]):
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "result.staged_inputs",
                        "staged input paths differ from the approved inputs",
                    )
                )
            if staged_valid:
                for item in normalized_staged:
                    destination = item.get("destination")
                    if (
                        not isinstance(destination, str)
                        or Path(destination).name != destination
                    ):
                        diagnostics.append(
                            _diagnostic(
                                "ERROR",
                                "result.staged_input_path",
                                "staged input destinations must be safe basenames",
                            )
                        )
                        continue
                    staged_path = raw_run / destination
                    if (
                        _has_symlink_component(staged_path)
                        or not _ordinary_file(staged_path)
                    ):
                        diagnostics.append(
                            _diagnostic(
                                "ERROR",
                                "result.staged_input",
                                f"staged input is missing: {destination}",
                            )
                        )
            launcher_value = context_resources.get("mpi_launcher")
            if launcher_value is None:
                launcher_path_value = None
            else:
                launcher_path = _resolve_nonsymlink(launcher_value, project_root)
                launcher_path_value = (
                    str(launcher_path) if _ordinary_file(launcher_path) else None
                )
            if (
                execution_metadata.get("mpi_launcher") != launcher_path_value
                or execution_metadata.get("mpi_processes")
                != context_parameters.get("mpi_processes")
            ):
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "result.mpi", "MPI launcher settings differ from the plan"
                    )
                )

        structure_records: List[Dict[str, Any]] = []
        selected_records: List[Dict[str, Any]] = []
        if len(role_paths.get("ssw-structure-manifest", [])) == 1:
            value = self._read_json_object(
                role_paths["ssw-structure-manifest"][0], "ssw_structure_manifest", diagnostics
            )
            if value is not None:
                raw = value.get("structures")
                if value.get("schema_version") != 1 or value.get("source_id") != result.get(
                    "source_id"
                ) or not isinstance(raw, list):
                    diagnostics.append(
                        _diagnostic(
                            "ERROR",
                            "result.structure_manifest",
                            "SSW structure manifest schema or records are invalid",
                        )
                    )
                else:
                    structure_records = [item for item in raw if isinstance(item, dict)]
                    if len(structure_records) != len(raw):
                        diagnostics.append(
                            _diagnostic(
                                "ERROR", "result.structure_record", "SSW records must be objects"
                            )
                        )
        if len(role_paths.get("selected-structure-manifest", [])) == 1:
            value = self._read_json_object(
                role_paths["selected-structure-manifest"][0],
                "selected_structure_manifest",
                diagnostics,
            )
            if value is not None:
                raw = value.get("structures")
                if value.get("schema_version") != 1 or value.get("source_id") != result.get(
                    "source_id"
                ) or not isinstance(raw, list):
                    diagnostics.append(
                        _diagnostic(
                            "ERROR",
                            "result.selected_manifest",
                            "selected structure manifest schema or records are invalid",
                        )
                    )
                else:
                    selected_records = [item for item in raw if isinstance(item, dict)]
                    if len(selected_records) != len(raw):
                        diagnostics.append(
                            _diagnostic(
                                "ERROR", "result.selected_record", "selected records must be objects"
                            )
                        )

        if counts:
            if len(structure_records) != counts.get("generated_structure_count"):
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "result.generated_count", "generated structure count mismatch"
                    )
                )
            if len(source_frames) != counts.get("generated_structure_count"):
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "result.source_generated_count",
                        "bound allstr.arc frame count differs from the result",
                    )
                )
            if len(selected_records) != counts.get("selected_structure_count"):
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "result.selected_count", "selected structure count mismatch"
                    )
                )
            accepted = sum(
                1 for record in structure_records if record.get("energy_filter_pass") is True
            )
            if accepted != counts.get("energy_accepted_count"):
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "result.accepted_count", "energy accepted count mismatch"
                    )
                )

        identifiers: List[str] = []
        selected_from_all: List[Dict[str, Any]] = []
        accepted_order = 0
        recomputed_selected_order = 0
        approved_stride = context_parameters.get("selection_stride")
        if not _positive_int(approved_stride):
            approved_stride = 1
        for index, record in enumerate(structure_records):
            identifier = record.get("structure_id")
            frame_index = record.get("frame_index")
            energy = record.get("energy_ev")
            if not _plain_string(identifier):
                diagnostics.append(
                    _diagnostic("ERROR", "result.structure_id", "structure_id is required")
                )
            else:
                identifiers.append(str(identifier))
            if frame_index != index + 1 or record.get("historical_order") != index + 1:
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "result.structure_order", "historical frame order is invalid"
                    )
                )
            energy_valid = not (
                not isinstance(energy, (int, float))
                or isinstance(energy, bool)
                or not math.isfinite(float(energy))
            )
            if not energy_valid:
                diagnostics.append(
                    _diagnostic("ERROR", "result.energy", f"frame {index + 1} energy is invalid")
                )
            if (
                isinstance(frame_index, int)
                and not isinstance(frame_index, bool)
                and identifier
                != _lasp_structure_id("ssw", frame_index)
            ):
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "result.structure_id_order",
                        "structure_id does not match the frame order",
                    )
                )
            if record.get("source_role") != "allstr.arc":
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "result.source_role", "SSW records must come from allstr.arc"
                    )
                )
            if index >= len(source_frames) or any(
                record.get(key) != source_frames[index].get(key)
                for key in ("frame_index", "energy_ev")
            ):
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "result.source_frame",
                        "SSW record differs from the corresponding bound allstr.arc frame",
                    )
                )
            expected_accepted = bool(
                energy_valid
                and (
                    expected_energy is None
                    or float(energy) <= float(expected_energy)
                )
            )
            if expected_accepted:
                accepted_order += 1
            expected_selected = expected_accepted and (accepted_order - 1) % int(
                approved_stride
            ) == 0
            if expected_selected:
                recomputed_selected_order += 1
            if (
                record.get("energy_filter_pass") is not expected_accepted
                or record.get("energy_filter_order")
                != (accepted_order if expected_accepted else None)
                or record.get("selected") is not expected_selected
                or record.get("selected_order")
                != (recomputed_selected_order if expected_selected else None)
            ):
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "result.selection_recomputed",
                        "energy filter, accepted order, or stride selection differs from the plan",
                    )
                )
            if expected_selected:
                selected_from_all.append(record)
        if len(set(identifiers)) != len(identifiers):
            diagnostics.append(
                _diagnostic("ERROR", "result.structure_id_duplicate", "structure IDs must be unique")
            )

        generated_artifact_paths = {
            path.relative_to(manifest.parent.resolve()).as_posix(): path
            for path in role_paths.get("ssw-generated-structure", [])
        }
        if len(generated_artifact_paths) != len(structure_records):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "result.generated_artifacts",
                    "generated structure artifact count does not match SSW records",
                )
            )
        for record in structure_records:
            path = generated_artifact_paths.get(record.get("structure_file"))
            if path is None:
                diagnostics.append(
                    _diagnostic(
                        "ERROR",
                        "result.generated_path",
                        "SSW structure_file is not a generated artifact",
                    )
                )
            else:
                try:
                    parsed = _read_arc_frames(path, 1)
                    frame_index = int(record.get("frame_index", 0))
                    if (
                        len(parsed) != 1
                        or frame_index < 1
                        or frame_index > len(source_frames)
                        or parsed[0].get("energy_ev") != record.get("energy_ev")
                        or parsed[0].get("payload")
                        != source_frames[frame_index - 1].get("payload")
                    ):
                        raise ValueError("ARC structure does not match its record")
                except (OSError, UnicodeError, ValueError) as exc:
                    diagnostics.append(
                        _diagnostic(
                            "ERROR",
                            "result.generated_arc",
                            f"generated structure is not a bound LASP ARC frame: {exc}",
                        )
                    )

        selected_ids = [record.get("structure_id") for record in selected_records]
        expected_ids = [
            record.get("structure_id")
            for record in sorted(selected_from_all, key=lambda item: item.get("selected_order", 0))
        ]
        if selected_ids != expected_ids:
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "result.selection_records",
                    "selected manifest does not match selected SSW records in historical order",
                )
            )
        selected_artifact_paths = {
            path.relative_to(manifest.parent.resolve()).as_posix(): path
            for path in role_paths.get("selected-structure", [])
        }
        if len(selected_artifact_paths) != len(selected_records):
            diagnostics.append(
                _diagnostic(
                    "ERROR",
                    "result.selected_artifacts",
                    "selected structure artifact count does not match selected records",
                )
            )
        for order, record in enumerate(selected_records, 1):
            if record.get("selected_order") != order:
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "result.selected_order", "selected_order must be contiguous"
                    )
                )
            path = selected_artifact_paths.get(record.get("output_file"))
            if path is None:
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "result.selected_path", "selected output_file is not an artifact"
                    )
                )
            else:
                try:
                    parsed = _read_arc_frames(path, 1)
                    frame_index = int(record.get("frame_index", 0))
                    if (
                        len(parsed) != 1
                        or frame_index < 1
                        or frame_index > len(source_frames)
                        or parsed[0].get("energy_ev") != record.get("energy_ev")
                        or parsed[0].get("payload")
                        != source_frames[frame_index - 1].get("payload")
                    ):
                        raise ValueError("selected ARC does not match its record")
                except (OSError, UnicodeError, ValueError) as exc:
                    diagnostics.append(
                        _diagnostic(
                            "ERROR",
                            "result.selected_arc",
                            f"selected structure is not a bound LASP ARC frame: {exc}",
                        )
                    )

        optional_manifest_counts = (
            (
                "aimd-seed-manifest",
                "aimd_seed_candidate_count",
                "aimd-seed-candidate",
                "aimd-seeds",
                "best.arc",
                best_source_frames,
            ),
            (
                "md-structure-manifest",
                "md_structure_count",
                "md-sampled-structure",
                "md-structures",
                "md.arc",
                md_source_frames,
            ),
        )
        for (
            role,
            count_key,
            structure_role,
            record_role,
            source_role,
            bound_frames,
        ) in optional_manifest_counts:
            expected_count = counts.get(count_key, 0) if counts else 0
            paths = role_paths.get(role, [])
            if expected_count == 0 and paths:
                diagnostics.append(
                    _diagnostic(
                        "ERROR", "result.optional_manifest", f"unexpected {role} artifact"
                    )
                )
            elif expected_count > 0:
                if len(paths) != 1:
                    diagnostics.append(
                        _diagnostic(
                            "ERROR", "result.optional_manifest", f"exactly one {role} is required"
                        )
                    )
                else:
                    value = self._read_json_object(paths[0], role.replace("-", "_"), diagnostics)
                    records = value.get("structures") if isinstance(value, dict) else None
                    if (
                        not isinstance(value, dict)
                        or value.get("schema_version") != 1
                        or value.get("source_id") != result.get("source_id")
                        or not isinstance(records, list)
                        or len(records) != expected_count
                    ):
                        diagnostics.append(
                            _diagnostic(
                                "ERROR", "result.optional_count", f"{role} count mismatch"
                            )
                        )
                    else:
                        if len(bound_frames) != expected_count:
                            diagnostics.append(
                                _diagnostic(
                                    "ERROR",
                                    "result.optional_source_count",
                                    f"bound {source_role} frame count differs from result",
                                )
                            )
                        artifact_paths = {
                            path.relative_to(manifest.parent.resolve()).as_posix(): path
                            for path in role_paths.get(structure_role, [])
                        }
                        if len(artifact_paths) != expected_count:
                            diagnostics.append(
                                _diagnostic(
                                    "ERROR",
                                    "result.optional_artifacts",
                                    f"{structure_role} artifact count mismatch",
                                )
                            )
                        seen_optional_ids = set()
                        for order, record in enumerate(records, 1):
                            if not isinstance(record, Mapping):
                                diagnostics.append(
                                    _diagnostic(
                                        "ERROR",
                                        "result.optional_record",
                                        f"{role} records must be objects",
                                    )
                                )
                                continue
                            identifier = record.get("structure_id")
                            path = artifact_paths.get(record.get("output_file"))
                            energy = record.get("energy_ev")
                            if not _plain_string(identifier) or identifier in seen_optional_ids:
                                diagnostics.append(
                                    _diagnostic(
                                        "ERROR",
                                        "result.optional_id",
                                        f"{role} structure IDs must be unique",
                                    )
                                )
                            elif isinstance(identifier, str):
                                seen_optional_ids.add(identifier)
                            expected_identifier = _lasp_structure_id(record_role, order)
                            if identifier != expected_identifier or record.get(
                                "source_role"
                            ) != source_role:
                                diagnostics.append(
                                    _diagnostic(
                                        "ERROR",
                                        "result.optional_source",
                                        f"{role} ID or source role differs from its record order",
                                    )
                                )
                            if (
                                record.get("frame_index") != order
                                or record.get("historical_order") != order
                                or not isinstance(energy, (int, float))
                                or isinstance(energy, bool)
                                or not math.isfinite(float(energy))
                            ):
                                diagnostics.append(
                                    _diagnostic(
                                        "ERROR",
                                        "result.optional_order",
                                        f"{role} order or energy is invalid",
                                    )
                                )
                            if path is None:
                                diagnostics.append(
                                    _diagnostic(
                                        "ERROR",
                                        "result.optional_path",
                                        f"{role} output path is invalid",
                                    )
                                )
                            else:
                                try:
                                    parsed = _read_arc_frames(path, 1)
                                    if (
                                        len(parsed) != 1
                                        or order > len(bound_frames)
                                        or any(
                                            record.get(key) != bound_frames[order - 1].get(key)
                                            for key in ("frame_index", "energy_ev")
                                        )
                                        or parsed[0].get("energy_ev") != record.get("energy_ev")
                                        or parsed[0].get("payload")
                                        != bound_frames[order - 1].get("payload")
                                    ):
                                        raise ValueError("ARC differs from bound source frame")
                                except (OSError, UnicodeError, ValueError) as exc:
                                    diagnostics.append(
                                        _diagnostic(
                                            "ERROR",
                                            "result.optional_arc",
                                            f"{role} structure is not bound to its source: {exc}",
                                        )
                                    )
        return manifest, result, diagnostics, artifacts

    def check(self, context: Any) -> Dict[str, Any]:
        operation = _operation(context)
        if operation == DIRECT_OPERATION:
            return self._check_direct(context)
        if operation == MERGE_OPERATION:
            return self._check_merge(context)
        if operation == LASP_INPUT_OPERATION:
            return self._check_lasp_input(context)
        if operation not in LASP_OPERATIONS:
            return {
                "plugin_id": PLUGIN_ID,
                "status": "FAIL",
                "diagnostics": self.validate(context),
            }
        execution = _mapping(context.get("execution")) if isinstance(context, Mapping) else {}
        returncode = execution.get("returncode")
        if returncode is not None and returncode != 0:
            return {
                "plugin_id": PLUGIN_ID,
                "operation": operation,
                "status": "FAIL",
                "diagnostics": [
                    _diagnostic(
                        "ERROR", "execution.nonzero", f"LASP/SSW wrapper returned {returncode}"
                    )
                ],
            }
        manifest, result, diagnostics, _ = self._read_lasp_result(context)
        if manifest is not None and not manifest.exists() and not _has_errors(diagnostics):
            return {
                "plugin_id": PLUGIN_ID,
                "operation": operation,
                "status": "WAIT",
                "diagnostics": diagnostics,
            }
        if _has_errors(diagnostics):
            return {
                "plugin_id": PLUGIN_ID,
                "operation": operation,
                "status": "FAIL",
                "diagnostics": diagnostics,
            }
        return {
            "plugin_id": PLUGIN_ID,
            "operation": operation,
            "status": "OK",
            "result_file": str(manifest),
            "counts": result.get("counts", {}),
            "diagnostics": diagnostics,
        }

    def collect(self, context: Any) -> Dict[str, Any]:
        operation = _operation(context)
        if operation == DIRECT_OPERATION:
            return self._collect_direct(context)
        if operation == MERGE_OPERATION:
            return self._collect_merge(context)
        if operation == LASP_INPUT_OPERATION:
            return self._collect_lasp_input(context)
        checked = self.check(context)
        if checked.get("status") != "OK":
            return {
                "plugin_id": PLUGIN_ID,
                "operation": operation,
                "status": checked.get("status", "FAIL"),
                "artifacts": [],
                "metrics": {},
                "diagnostics": checked.get("diagnostics", []),
            }
        manifest, result, diagnostics, artifacts = self._read_lasp_result(context)
        assert manifest is not None
        counts = _mapping(result.get("counts"))
        return {
            "plugin_id": PLUGIN_ID,
            "operation": operation,
            "status": "OK",
            "artifacts": [
                {
                    "role": "sampling-result",
                    "path": str(manifest),
                    "media_type": "application/json",
                },
                *artifacts,
            ],
            "metrics": {
                "generated_structure_count": counts.get("generated_structure_count", 0),
                "energy_accepted_count": counts.get("energy_accepted_count", 0),
                "selected_structure_count": counts.get("selected_structure_count", 0),
                "aimd_seed_candidate_count": counts.get("aimd_seed_candidate_count", 0),
                "md_structure_count": counts.get("md_structure_count", 0),
                "lasp_execution": operation == "lasp-ssw-execute",
                "lasp_replay": operation == "lasp-ssw-normalize-replay",
            },
            "diagnostics": diagnostics,
        }
