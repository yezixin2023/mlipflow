"""Remote-side runner for approved LASP stochastic-surface-walking jobs.

This file is staged by the pes-sampling adapter.  Cluster-specific executable
paths stay in the site-owned ``lasp-ssw/run.sh`` template; scientific and
post-processing parameters are re-read from the staged portable project file.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import re
import shutil
import tarfile
import sys
from pathlib import Path
from typing import Any

UNKNOWN = "HISTORICAL_PARAMETER_UNKNOWN"
MAX_POTCAR_BYTES = 64 * 1024 * 1024
FINGERPRINT = re.compile(r"sha256:[0-9a-f]{64}")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lasp_input_arc import canonicalize_prepared_arc  # noqa: E402


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        import yaml
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("project file must contain a mapping")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _lasp_potential(path: Path) -> str:
    values: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        tokens = raw_line.split("#", 1)[0].split()
        if tokens[:1] == ["potential"]:
            if len(tokens) != 2:
                raise ValueError("lasp.in potential declaration is invalid")
            values.append(tokens[1].lower())
    if len(values) != 1:
        raise ValueError("lasp.in must declare exactly one explicit potential")
    return values[0]


def _pseudopotential_identity(input_dir: Path) -> dict[str, Any]:
    manifest_path = input_dir / "lasp-input-manifest.json"
    potcar_path = input_dir / "POTCAR"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("staged lasp-input-manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("plugin_id") != "pes-sampling"
        or manifest.get("operation") != "lasp-input-prepare"
    ):
        raise ValueError("staged LASP input manifest identity is invalid")
    structure = manifest.get("output")
    if (
        not isinstance(structure, dict)
        or structure.get("path") != "input.arc"
        or structure.get("sha256") != _sha256(input_dir / "input.arc")
        or structure.get("size_bytes") != (input_dir / "input.arc").stat().st_size
    ):
        raise ValueError("staged LASP input manifest does not bind input.arc")
    reference_sha256 = manifest.get("pseudopotential_reference_sha256")
    potcar = manifest.get("potcar")
    if (
        not isinstance(reference_sha256, str)
        or FINGERPRINT.fullmatch(reference_sha256) is None
        or not isinstance(potcar, dict)
    ):
        raise ValueError("staged LASP input manifest lacks pseudopotential identity")
    elements = potcar.get("elements")
    symbols = potcar.get("symbols")
    components = potcar.get("components")
    output = potcar.get("output")
    if (
        not isinstance(potcar.get("reference_id"), str)
        or not potcar.get("reference_id")
        or potcar.get("source_env") != "PMG_VASP_PSP_DIR"
        or potcar.get("configuration_source") not in {"environment", "pymatgen-settings"}
        or not isinstance(potcar.get("functional"), str)
        or not potcar.get("functional")
        or potcar.get("portable_artifact") is not False
        or not isinstance(elements, list)
        or not elements
        or any(not isinstance(element, str) or re.fullmatch(r"[A-Z][a-z]?", element) is None for element in elements)
        or not isinstance(symbols, list)
        or len(symbols) != len(elements)
        or any(not isinstance(symbol, str) or SAFE_NAME.fullmatch(symbol) is None for symbol in symbols)
        or not isinstance(components, list)
        or len(components) != len(symbols)
        or any(
            not isinstance(component, dict)
            or component.get("symbol") != symbol
            or not isinstance(component.get("sha256"), str)
            or FINGERPRINT.fullmatch(component["sha256"]) is None
            for component, symbol in zip(components, symbols)
        )
        or not isinstance(output, dict)
        or output.get("path") != "POTCAR"
        or output.get("collectable") is not False
        or output.get("sha256") != potcar.get("combined_sha256")
        or not isinstance(output.get("sha256"), str)
        or FINGERPRINT.fullmatch(output["sha256"]) is None
        or isinstance(output.get("size_bytes"), bool)
        or not isinstance(output.get("size_bytes"), int)
        or output.get("size_bytes") < 1
        or potcar_path.is_symlink()
        or not potcar_path.is_file()
        or potcar_path.stat().st_size > MAX_POTCAR_BYTES
        or potcar_path.stat().st_size != output.get("size_bytes")
        or _sha256(potcar_path) != output.get("sha256")
    ):
        raise ValueError("staged runtime-only POTCAR identity is invalid")
    return {
        "reference_id": potcar.get("reference_id"),
        "reference_sha256": reference_sha256,
        "functional": potcar.get("functional"),
        "elements": elements,
        "symbols": symbols,
        "components": components,
        "combined_sha256": output.get("sha256"),
        "configuration_source": potcar.get("configuration_source"),
        "manifest_sha256": _sha256(manifest_path),
        "portable_or_collectable": False,
    }


def _node(project: dict[str, Any], node_id: str) -> dict[str, Any]:
    workflow = project.get("workflow")
    nodes = workflow.get("nodes") if isinstance(workflow, dict) else None
    if not isinstance(nodes, list):
        raise ValueError("project workflow.nodes is missing")
    matches = [item for item in nodes if isinstance(item, dict) and item.get("id") == node_id]
    if len(matches) != 1:
        raise ValueError(f"project must contain exactly one node {node_id!r}")
    node = matches[0]
    if not str(node.get("uses", "")).split("@", 1)[0] == "pes-sampling":
        raise ValueError("scheduled LASP node must use pes-sampling")
    if node.get("backend") != "ssh-slurm":
        raise ValueError("scheduled LASP node must use ssh-slurm")
    parameters = node.get("parameters", {})
    if not isinstance(parameters, dict) or parameters.get("operation") != "lasp-ssw-execute":
        raise ValueError("scheduled LASP node must use operation lasp-ssw-execute")
    return node


def _load_wrapper(path: Path):
    spec = importlib.util.spec_from_file_location("mlipflow_remote_lasp_ssw", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load staged wrapper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _deterministic_selected_archive(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise RuntimeError("LASP normalization produced no selected directory")
    tar_path = destination.with_suffix("")
    with tarfile.open(tar_path, "w") as archive:
        for path in sorted(source.glob("*.arc")):
            payload = path.read_bytes()
            info = tarfile.TarInfo(f"selected/{path.name}")
            info.size = len(payload)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            import io
            archive.addfile(info, io.BytesIO(payload))
    with tar_path.open("rb") as src, destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            shutil.copyfileobj(src, compressed)
    tar_path.unlink()


def run(args: argparse.Namespace) -> int:
    project_path = Path(args.project).resolve()
    input_dir = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    project = _load_mapping(project_path)
    node = _node(project, args.node_id)
    parameters = node.get("parameters", {})
    inputs = node.get("inputs", {})
    if not isinstance(inputs, dict):
        raise ValueError("node inputs must be a mapping")
    potential = _lasp_potential(input_dir / "lasp.in")
    pseudopotential = None
    if potential == "vasp":
        if "lasp_input_manifest" not in inputs:
            raise ValueError("potential vasp requires lasp_input_manifest")
        pseudopotential = _pseudopotential_identity(input_dir)
    elif "lasp_input_manifest" in inputs:
        raise ValueError("lasp_input_manifest is accepted only for potential vasp")

    wrapper_path = input_dir / "lasp_ssw.py"
    wrapper = _load_wrapper(wrapper_path)
    output_root.mkdir(parents=True, exist_ok=True)
    source_input = input_dir / "input.arc"
    canonical_payload, arc_conversion = canonicalize_prepared_arc(source_input.read_bytes())
    canonical_input = output_root / "input.canonical.arc"
    canonical_input.write_bytes(canonical_payload)
    target = output_root / "lasp-ssw"
    argv = [
        "execute",
        "--lasp-executable", str(Path(args.lasp_executable).resolve()),
        "--input-structure", str(canonical_input),
        "--lasp-input", str(input_dir / "lasp.in"),
        "--output-dir", str(target),
        "--historical-source-id", str(parameters["historical_source_id"]),
        "--selection-stride", str(parameters["selection_stride"]),
        "--max-frames", str(parameters["max_frames"]),
        "--seed-status", UNKNOWN,
        "--lasp-version", str(parameters["lasp_version"]),
    ]
    if parameters.get("energy_max_ev") is not None:
        argv += ["--energy-max-ev", str(parameters["energy_max_ev"])]
    if parameters.get("include_best_arc"):
        argv.append("--include-best-arc")
    if parameters.get("include_md_arc"):
        argv.append("--include-md-arc")
    auxiliary = inputs.get("lasp_auxiliary_files", {})
    if auxiliary is not None:
        if not isinstance(auxiliary, dict):
            raise ValueError("lasp_auxiliary_files must be a mapping")
        for name in sorted(auxiliary):
            argv += ["--auxiliary", str(name), str(input_dir / str(name))]
    if potential == "vasp":
        if isinstance(auxiliary, dict) and "POTCAR" in auxiliary:
            raise ValueError("POTCAR must come from the verified LASP input manifest")
        argv += ["--auxiliary", "POTCAR", str(input_dir / "POTCAR")]
    if args.mpi_launcher:
        if args.mpi_processes is None or args.mpi_processes < 1:
            raise ValueError("--mpi-launcher requires positive --mpi-processes")
        argv += ["--mpi-launcher", str(Path(args.mpi_launcher).resolve()), "--mpi-processes", str(args.mpi_processes)]

    returncode = int(wrapper.main(argv))
    if returncode != 0:
        return returncode

    archive = output_root / "selected-structures.tar.gz"
    _deterministic_selected_archive(target / "selected", archive)
    result = json.loads((target / "sampling-result.json").read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "plugin_id": "pes-sampling",
        "operation": "lasp-ssw-execute",
        "status": "OK",
        "node_id": args.node_id,
        "lasp_version": str(parameters["lasp_version"]),
        "potential": potential,
        "pseudopotential": pseudopotential,
        "input_structure": {
            "source_sha256": _sha256(source_input),
            "canonical_sha256": _sha256(canonical_input),
            "conversion": arc_conversion,
            "canonical_collected": False,
        },
        "selection_policy": {
            "selection_stride": parameters["selection_stride"],
            "energy_max_ev": parameters.get("energy_max_ev"),
            "max_frames": parameters["max_frames"],
            "preserve_historical_order": parameters.get("preserve_historical_order"),
        },
        "counts": result.get("counts", {}),
        "selected_archive": {
            "path": "selected-structures.tar.gz",
            "sha256": _sha256(archive),
            "size_bytes": archive.stat().st_size,
        },
        "source_outputs": {
            "allstr_arc_sha256": _sha256(target / "raw-run" / "allstr.arc"),
        },
    }
    (output_root / "cluster-run-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an approved LASP SSW job inside an ssh-slurm allocation")
    parser.add_argument("--project", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lasp-executable", required=True)
    parser.add_argument("--mpi-launcher")
    parser.add_argument("--mpi-processes", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except Exception as exc:
        print(f"LASP cluster runner error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
