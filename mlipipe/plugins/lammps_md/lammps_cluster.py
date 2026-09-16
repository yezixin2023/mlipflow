"""Remote-side resolver and executor for scheduled LAMMPS MLIP MD.

The project stages a reviewed LAMMPS input bundle, never a model or cluster
executable. The site template supplies MODEL_ROOT, LAMMPS_BIN and an optional
JSON launcher argv. This runner checks the staged paths, resolves the
model below MODEL_ROOT, launches LAMMPS with shell=False, and writes a small
machine-readable result for the local checker.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import traceback
from pathlib import Path, PurePosixPath
from typing import Any

PREPARATION_CONTRACT = "lammps-md-input-v2"
FRAMEWORKS = {"deepmd", "mace", "m3gnet"}
TARGETS = {"cpu", "gpu"}
MODEL_CONTRACTS = {
    ("deepmd", None): ("file", "deepmd-lammps-model", {"cpu", "gpu"}),
    ("mace", None): ("file", "mace-lammps-torchscript", {"cpu", "gpu"}),
    ("m3gnet", "matgl"): ("file", "matgl-lammps-torchscript", {"cpu", "gpu"}),
    ("m3gnet", "gnnp"): ("directory", "matgl-model-directory", {"cpu"}),
    ("m3gnet", "m3gnet"): ("directory", "matgl-model-directory", {"cpu"}),
}
LAMMPS_VERSION = re.compile(r"LAMMPS\s*\(([^\r\n)]+)\)")


def _ordinary_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _read_mapping(path: Path) -> dict[str, Any]:
    if not _ordinary_file(path):
        raise ValueError(f"missing or empty file: {path.name}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        value = yaml.safe_load(text)
    else:
        value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _input_manifest_record(node_inputs: Any) -> str:
    if not isinstance(node_inputs, dict):
        raise ValueError("scheduled LAMMPS node inputs must be a mapping")
    value = node_inputs.get("lammps_input_manifest")
    if isinstance(value, str) and value:
        return "lammps-input-manifest.json"
    if (
        isinstance(value, dict)
        and set(value) <= {"from_node", "role", "resolve"}
        and isinstance(value.get("from_node"), str)
        and value.get("from_node")
        and isinstance(value.get("role"), str)
        and value.get("role")
        and value.get("resolve", "artifact") == "artifact"
    ):
        return "lammps-input-manifest.json"
    raise ValueError("scheduled LAMMPS node must record its input manifest")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("model relative_path must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("model relative_path is unsafe")
    return value


def _project_node(project: Path, node_id: str) -> dict[str, Any]:
    raw = _read_mapping(project)
    workflow = raw.get("workflow")
    nodes = workflow.get("nodes") if isinstance(workflow, dict) else None
    if not isinstance(nodes, list):
        raise ValueError("project workflow.nodes must be a list")
    matches = [item for item in nodes if isinstance(item, dict) and item.get("id") == node_id]
    if len(matches) != 1:
        raise ValueError(f"project must contain exactly one node {node_id!r}")
    return matches[0]


def _generated_records(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = manifest.get("generated_files")
    if not isinstance(raw, list):
        raise ValueError("input manifest generated_files must be a list")
    records: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("generated file record is invalid")
        name = item["name"]
        if name in records:
            raise ValueError(f"duplicate generated file record: {name}")
        records[name] = item
    return records


def _launcher(manifest: dict[str, Any], target: str) -> dict[str, Any]:
    launchers = manifest.get("launchers")
    if not isinstance(launchers, list):
        raise ValueError("input manifest launchers must be a list")
    matches = [item for item in launchers if isinstance(item, dict) and item.get("target") == target]
    if len(matches) != 1:
        raise ValueError(f"input manifest must contain exactly one {target} launcher")
    launcher = matches[0]
    if launcher.get("executable") != "site-owned-lammps":
        raise ValueError("launcher executable contract is invalid")
    argv = launcher.get("argv_after_executable")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(item, str) or not item or "\x00" in item for item in argv)
    ):
        raise ValueError("launcher argv_after_executable must be a non-empty string list")
    if argv.count("<site-resolved-model-file>") != 1:
        raise ValueError("launcher argv must contain exactly one model placeholder")
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise ValueError("input manifest model record is missing")
    interface = model.get("lammps_interface") if model.get("framework") == "m3gnet" else None
    requires_interface_path = interface in {"gnnp", "m3gnet"}
    if bool(launcher.get("requires_interface_path", False)) != requires_interface_path:
        raise ValueError("launcher interface-path contract differs from model interface")
    if argv.count("<site-resolved-interface-path>") != (1 if requires_interface_path else 0):
        raise ValueError("launcher interface-path placeholder count is invalid")
    if launcher.get("lammps_interface", interface or model.get("framework")) != (
        interface or model.get("framework")
    ):
        raise ValueError("launcher interface differs from model interface")
    return launcher


def _resolve_model(model_root_value: str, model: dict[str, Any]) -> Path:
    root = Path(model_root_value).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("site model root does not exist")
    kind = model.get("kind")
    if kind not in {"file", "directory"}:
        raise ValueError("scheduled LAMMPS model kind must be file or directory")
    relative = _safe_relative(model.get("relative_path"))
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("model reference escapes site model root") from exc
    if kind == "file" and not _ordinary_file(candidate):
        raise ValueError("resolved LAMMPS model file is missing")
    if kind == "directory" and not candidate.is_dir():
        raise ValueError("resolved LAMMPS model directory is missing")
    return candidate


def _resolve_interface_path(value: str, required: bool) -> Path | None:
    if not required:
        if value:
            raise ValueError("site interface path was provided for an interface that does not use it")
        return None
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("site LAMMPS interface path is required")
    resolved = Path(value).expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError("site LAMMPS interface path is not a directory")
    return resolved


def _launcher_prefix(raw: str) -> list[str]:
    if raw in {"", "[]"}:
        return []
    value = json.loads(raw)
    if (
        not isinstance(value, list)
        or len(value) > 32
        or any(not isinstance(item, str) or not item or "\x00" in item for item in value)
    ):
        raise ValueError("site launcher JSON must be a bounded string list")
    return value


def _replace_model(
    argv: list[str], model: Path, interface_path: Path | None = None
) -> list[str]:
    replaced = [str(model) if item == "<site-resolved-model-file>" else item for item in argv]
    if replaced == argv:
        raise ValueError("launcher argv lacks the site model placeholder")
    if replaced.count(str(model)) != 1:
        raise ValueError("launcher argv must contain exactly one model placeholder")
    interface_placeholders = replaced.count("<site-resolved-interface-path>")
    if interface_path is None and interface_placeholders:
        raise ValueError("launcher argv requires a site interface path")
    if interface_path is not None and interface_placeholders != 1:
        raise ValueError("launcher argv must contain exactly one interface-path placeholder")
    if interface_path is not None:
        replaced = [
            str(interface_path) if item == "<site-resolved-interface-path>" else item
            for item in replaced
        ]
    return replaced


def _version_from_logs(*paths: Path) -> str:
    for path in paths:
        if not _ordinary_file(path):
            continue
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for _ in range(128):
                line = stream.readline()
                if not line:
                    break
                match = LAMMPS_VERSION.search(line)
                if match:
                    return match.group(1).strip()
    raise ValueError("LAMMPS version banner was not found in logs")


def _contains_marker(path: Path, marker: str) -> bool:
    if not _ordinary_file(path):
        return False
    needle = marker.encode("utf-8")
    with path.open("rb") as stream:
        carry = b""
        while True:
            chunk = stream.read(1 << 20)
            if not chunk:
                return needle in carry
            data = carry + chunk
            if needle in data:
                return True
            carry = data[-max(len(needle) - 1, 0) :]


def _artifact(path: Path, name: str) -> dict[str, Any]:
    if not _ordinary_file(path):
        raise ValueError(f"required LAMMPS artifact is missing or empty: {name}")
    return {"name": name, "path": name}


def run(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"schema_version": 1, "status": "FAIL", "node_id": args.node_id}
    try:
        node = _project_node(Path(args.project).expanduser().resolve(), args.node_id)
        if node.get("uses") != "lammps-md":
            raise ValueError("scheduled node does not use lammps-md")
        if node.get("backend") != "ssh-slurm":
            raise ValueError("scheduled LAMMPS execute requires ssh-slurm")
        parameters = node.get("parameters")
        if not isinstance(parameters, dict) or parameters.get("operation") != "execute":
            raise ValueError("scheduled LAMMPS node must declare operation=execute")
        target = parameters.get("target")
        if target not in TARGETS:
            raise ValueError("target must be cpu or gpu")

        manifest_path = input_dir / "lammps-input-manifest.json"
        manifest = _read_mapping(manifest_path)
        input_manifest_path = _input_manifest_record(node.get("inputs"))
        if (
            manifest.get("schema_version") != 1
            or manifest.get("plugin_id") != "lammps-md"
            or manifest.get("operation") != "lammps-prepare"
            or manifest.get("status") != "OK"
            or manifest.get("preparation_contract") != PREPARATION_CONTRACT
            or manifest.get("runtime_model_variable") != "MODEL_FILE"
        ):
            raise ValueError("staged LAMMPS input manifest is not execution-ready")
        marker = manifest.get("completion_marker")
        if not isinstance(marker, str) or not marker.startswith("MLIPIPE_LAMMPS_COMPLETED step="):
            raise ValueError("input manifest completion marker is invalid")

        model = manifest.get("model")
        md = manifest.get("md")
        if not isinstance(model, dict) or not isinstance(md, dict):
            raise ValueError("input manifest model/MD record is incomplete")
        framework = model.get("framework")
        if framework not in FRAMEWORKS:
            raise ValueError("unsupported LAMMPS framework")
        interface = model.get("lammps_interface") if framework == "m3gnet" else None
        if framework == "m3gnet" and interface is None:
            interface = "matgl"
        model_contract = MODEL_CONTRACTS.get((str(framework), interface))
        if model_contract is None:
            raise ValueError("unsupported LAMMPS model interface")
        expected_kind, expected_format, supported_targets = model_contract
        if (
            model.get("kind") != expected_kind
            or model.get("artifact_format") != expected_format
            or target not in supported_targets
        ):
            raise ValueError("LAMMPS model artifact/interface contract is inconsistent")
        if target not in md.get("targets", []):
            raise ValueError("selected execution target was not prepared")
        steps = md.get("steps")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("approved MD step count is invalid")
        if marker != f"MLIPIPE_LAMMPS_COMPLETED step={steps}":
            raise ValueError("completion marker does not bind the approved step count")

        generated = _generated_records(manifest)
        selected_name = f"in.{target}.lammps"
        for name in ("structure.data", selected_name):
            record = generated.get(name)
            staged = input_dir / "lammps" / name
            if record is None or not _ordinary_file(staged):
                raise ValueError(f"staged prepared input is missing: {name}")
        deck_text = (input_dir / "lammps" / selected_name).read_text(encoding="utf-8")
        requires_interface_path = interface in {"gnnp", "m3gnet"}
        if (
            marker not in deck_text
            or "${MODEL_FILE}" not in deck_text
            or ("${INTERFACE_PATH}" in deck_text) != requires_interface_path
        ):
            raise ValueError("selected input deck lacks its approved portable runtime contract")
        relative_model = str(model.get("relative_path", ""))
        if relative_model and relative_model in deck_text:
            raise ValueError("selected input deck embeds the site-relative model path")

        model_root = args.model_root or os.environ.get("MLIPIPE_MODEL_ROOT", "")
        model_path = _resolve_model(model_root, model)

        shutil.copy2(input_dir / "lammps" / "structure.data", output_dir / "structure.data")
        shutil.copy2(input_dir / "lammps" / selected_name, output_dir / selected_name)
        launcher = _launcher(manifest, str(target))
        interface_value = args.interface_path or os.environ.get(
            "MLIPIPE_LAMMPS_INTERFACE_PATH", ""
        )
        interface_path = _resolve_interface_path(
            interface_value, requires_interface_path
        )
        launch_argv = _replace_model(
            list(launcher["argv_after_executable"]), model_path, interface_path
        )
        lammps_bin = args.lammps_bin
        if not isinstance(lammps_bin, str) or not lammps_bin or "\x00" in lammps_bin:
            raise ValueError("site LAMMPS executable is required")
        prefix = _launcher_prefix(args.launcher_json)
        log_path = output_dir / "lammps.log"
        screen_path = output_dir / "lammps.screen.log"
        stdout_path = output_dir / "lammps.stdout.log"
        stderr_path = output_dir / "lammps.stderr.log"
        command = [*prefix, lammps_bin, *launch_argv, "-log", log_path.name, "-screen", screen_path.name]
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            completed = subprocess.run(
                command,
                cwd=output_dir,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                shell=False,
                check=False,
            )
        if completed.returncode != 0:
            raise ValueError(f"LAMMPS exited with code {completed.returncode}")
        if not _contains_marker(log_path, marker):
            raise ValueError("LAMMPS log lacks the approved end-of-script completion marker")
        lammps_version = _version_from_logs(log_path, screen_path, stdout_path)
        artifacts = [
            _artifact(output_dir / "trajectory.lammpstrj", "trajectory.lammpstrj"),
            _artifact(output_dir / "final.data", "final.data"),
            _artifact(output_dir / "final.restart", "final.restart"),
            _artifact(log_path, "lammps.log"),
            _artifact(screen_path, "lammps.screen.log"),
        ]
        if _ordinary_file(stdout_path):
            artifacts.append(_artifact(stdout_path, "lammps.stdout.log"))
        if _ordinary_file(stderr_path):
            artifacts.append(_artifact(stderr_path, "lammps.stderr.log"))

        result = {
            "schema_version": 1,
            "plugin_id": "lammps-md",
            "status": "OK",
            "operation": "execute",
            "framework": framework,
            "target": target,
            "lammps_version": lammps_version,
            "input_manifest_path": input_manifest_path,
            "model": {
                "id": model.get("model_id"),
                "path": model.get("relative_path"),
                "kind": model.get("kind"),
                "artifact_format": model.get("artifact_format"),
                "lammps_interface": interface,
            },
            "ensemble": md.get("ensemble"),
            "temperature_K": md.get("temperature_k"),
            "timestep_fs": md.get("timestep_fs"),
            "dump_interval": md.get("dump_interval"),
            "type_map": md.get("type_map"),
            "structure_path": manifest.get("input_paths", {}).get("structure"),
            "steps_requested": steps,
            "steps_completed": steps,
            "completion_marker": marker,
            "launcher": {
                "site_launcher_used": bool(prefix),
                "prepared_argv_after_executable": launcher["argv_after_executable"],
                "required_packages": launcher.get("required_packages", []),
            },
            "artifacts": artifacts,
        }
        result_path = output_dir / "lammps-execution-result.json"
        _write_json(result_path, result)
        report.update(
            {
                "status": "OK",
                "framework": framework,
                "target": target,
                "model_id": model.get("model_id"),
                "model_path": model.get("relative_path"),
                "model_kind": model.get("kind"),
                "lammps_interface": interface,
                "input_manifest_path": input_manifest_path,
                "steps_completed": steps,
                "lammps_version": lammps_version,
            }
        )
        _write_json(output_dir / "cluster-run-report.json", report)
        return 0
    except Exception as exc:
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc(limit=8)
        _write_json(output_dir / "cluster-run-report.json", report)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reviewed LAMMPS MLIP input on a cluster")
    parser.add_argument("--project", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-root")
    parser.add_argument("--interface-path")
    parser.add_argument("--lammps-bin", required=True)
    parser.add_argument("--launcher-json", default="[]")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
