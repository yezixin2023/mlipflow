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
import shutil
import tarfile
from pathlib import Path
from typing import Any

UNKNOWN = "HISTORICAL_PARAMETER_UNKNOWN"


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

    wrapper_path = input_dir / "lasp_ssw.py"
    wrapper = _load_wrapper(wrapper_path)
    target = output_root / "lasp-ssw"
    argv = [
        "execute",
        "--lasp-executable", str(Path(args.lasp_executable).resolve()),
        "--input-structure", str(input_dir / "input.arc"),
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
    if args.mpi_launcher:
        if args.mpi_processes is None or args.mpi_processes < 1:
            raise ValueError("--mpi-launcher requires positive --mpi-processes")
        argv += ["--mpi-launcher", str(Path(args.mpi_launcher).resolve()), "--mpi-processes", str(args.mpi_processes)]

    output_root.mkdir(parents=True, exist_ok=True)
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
