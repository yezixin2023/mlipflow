"""Remote-side resolver for scheduled ASE MD.

The portable project contains a structure, a model-reference manifest and,
for approved retries, an explicitly staged restart checkpoint. The site-owned
run template supplies the cluster-local model root. This module resolves the
declared model path and invokes the staged ase_md.py implementation in-process.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import traceback
from pathlib import Path, PurePosixPath
from typing import Any

MODEL_KINDS = {"deepmd": "file", "m3gnet": "directory", "chgnet": "file", "mace": "file"}
ENSEMBLES = {"nvt-langevin", "npt-isotropic-mtk"}


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        value = yaml.safe_load(text)
    else:
        value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _reference(value: Any) -> str:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict) and isinstance(value.get("path"), str) and value["path"]:
        return value["path"]
    raise ValueError("input reference must contain a non-empty path")


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("relative_path must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe relative_path: {value!r}")
    return value


def _resolve_under(root_value: str, relative: str, kind: str) -> Path:
    if not root_value:
        raise ValueError("site model root is required")
    root = Path(root_value).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"site model root does not exist: {root}")
    candidate = (root / _safe_relative(relative)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("model reference escapes the site model root") from exc
    if kind == "file" and not candidate.is_file():
        raise ValueError(f"referenced model file does not exist: {candidate}")
    if kind == "directory" and not candidate.is_dir():
        raise ValueError(f"referenced model directory does not exist: {candidate}")
    return candidate


def _model_reference(path: Path, calculator: str) -> dict[str, str]:
    raw = _load_mapping(path)
    if raw.get("schema_version") != 1:
        raise ValueError("model reference schema_version must be 1")
    model_id = raw.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model reference requires model_id")
    relative = _safe_relative(raw.get("relative_path", model_id))
    expected_kind = MODEL_KINDS[calculator]
    kind = raw.get("kind", expected_kind)
    if kind != expected_kind:
        raise ValueError(f"{calculator} model reference kind must be {expected_kind}")
    return {
        "id": model_id,
        "relative_path": relative,
        "kind": kind,
    }


def _project_node(project: Path, node_id: str) -> dict[str, Any]:
    raw = _load_mapping(project)
    workflow = raw.get("workflow")
    nodes = workflow.get("nodes") if isinstance(workflow, dict) else None
    if not isinstance(nodes, list):
        raise ValueError("project workflow.nodes must be a list")
    matches = [item for item in nodes if isinstance(item, dict) and item.get("id") == node_id]
    if len(matches) != 1:
        raise ValueError(f"project must contain exactly one node {node_id!r}")
    return matches[0]


def _load_runner(input_dir: Path):
    path = input_dir / "ase_md.py"
    spec = importlib.util.spec_from_file_location("_mlipflow_staged_ase_md", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load staged ase_md.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_report(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cluster-run-report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _normalize_result_artifacts(output_dir: Path, result: dict[str, Any]) -> None:
    """Keep checkpoint provenance separate from the stable four-artifact MD set."""
    artifacts = result.get("artifacts")
    if isinstance(artifacts, list):
        result["artifacts"] = [
            item
            for item in artifacts
            if not (isinstance(item, dict) and item.get("name") == "checkpoint")
        ]
    (output_dir / "md-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    report: dict[str, Any] = {"schema_version": 1, "status": "FAIL", "node_id": args.node_id}
    try:
        node = _project_node(Path(args.project).expanduser().resolve(), args.node_id)
        uses = str(node.get("uses", "")).split("@", 1)[0]
        if uses != "ase-md":
            raise ValueError("scheduled node does not use ase-md")
        if node.get("backend") != "ssh-slurm":
            raise ValueError("scheduled ASE MD requires backend ssh-slurm")
        inputs = node.get("inputs")
        parameters = node.get("parameters")
        if not isinstance(inputs, dict) or not isinstance(parameters, dict):
            raise ValueError("ASE MD node inputs and parameters must be objects")
        calculator = parameters.get("calculator")
        if calculator not in MODEL_KINDS:
            raise ValueError("unsupported ASE MD calculator")
        ensemble = parameters.get("ensemble", "nvt-langevin")
        if ensemble not in ENSEMBLES:
            raise ValueError("unsupported ASE MD ensemble")

        structure_reference = _reference(inputs.get("structure"))
        structure_name = Path(structure_reference).name
        if structure_name in {"", ".", ".."}:
            raise ValueError("structure input must have a safe basename")
        structure = input_dir / "structure" / structure_name
        model_reference_path = input_dir / "model-reference.json"
        if not structure.is_file() or not model_reference_path.is_file():
            raise ValueError("staged structure and model-reference.json are required")
        model_ref = _model_reference(model_reference_path, str(calculator))
        model_root = args.model_root or os.environ.get("MLIPFLOW_MODEL_ROOT", "")
        model = _resolve_under(model_root, model_ref["relative_path"], model_ref["kind"])

        restart_path = input_dir / "restart" / "md-checkpoint.json"
        restart_checkpoint = restart_path if restart_path.is_file() else None
        if restart_path.exists() and restart_checkpoint is None:
            raise ValueError("staged restart checkpoint is not a regular file")

        runner = _load_runner(input_dir)
        result = runner.run_md(
            structure=structure,
            model=model,
            output_dir=output_dir,
            calculator=str(calculator),
            ensemble=str(ensemble),
            model_id=model_ref["id"],
            model_record_path=model_ref["relative_path"],
            structure_record_path=f"structure/{structure_name}",
            temperature_k=parameters.get("temperature_k"),
            timestep_fs=parameters.get("timestep_fs"),
            steps=parameters.get("steps"),
            trajectory_interval=parameters.get("trajectory_interval"),
            thermo_interval=parameters.get("thermo_interval"),
            checkpoint_interval=parameters.get("checkpoint_interval"),
            restart_checkpoint=restart_checkpoint,
            seed=parameters.get("seed"),
            device=parameters.get("device"),
            default_dtype=parameters.get("default_dtype"),
            friction_per_fs=parameters.get("friction_per_fs"),
            fix_com=parameters.get("fix_com"),
            pressure_gpa=parameters.get("pressure_gpa"),
            thermostat_damping_fs=parameters.get("thermostat_damping_fs"),
            barostat_damping_fs=parameters.get("barostat_damping_fs"),
            input_format=parameters.get("input_format"),
            input_index=str(parameters.get("input_index", "-1")),
            supercell_repeat=parameters.get("supercell_repeat"),
            minimum_initial_cell_length_angstrom=parameters.get(
                "minimum_initial_cell_length_angstrom"
            ),
        )
        _normalize_result_artifacts(output_dir, result)
        report.update(
            {
                "status": "OK",
                "calculator": calculator,
                "ensemble": ensemble,
                "model": model_ref,
                "structure_path": f"structure/{structure_name}",
                "supercell_repeat": result.get("supercell_repeat"),
                "source_atom_count": result.get("source_atom_count"),
                "atom_count": result.get("atom_count"),
                "initial_cell_lengths_A": result.get("initial_cell_lengths_A"),
                "minimum_initial_cell_length_A": result.get(
                    "minimum_initial_cell_length_A"
                ),
                "observed_stability": result.get("observed_stability"),
                "steps_completed": result.get("steps_completed"),
                "segment_start_step": result.get("segment_start_step"),
                "restart": result.get("restart"),
            }
        )
        if ensemble == "npt-isotropic-mtk":
            report.update(
                {
                    "pressure_GPa": result.get("pressure_GPa"),
                    "thermostat_damping_fs": result.get("thermostat_damping_fs"),
                    "barostat_damping_fs": result.get("barostat_damping_fs"),
                }
            )
        _write_report(output_dir, report)
        return 0
    except Exception as exc:
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc(limit=8)
        _write_report(output_dir, report)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolve site model and run scheduled ASE MD")
    parser.add_argument("--project", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-root")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
