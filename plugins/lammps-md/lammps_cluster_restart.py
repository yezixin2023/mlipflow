"""Restart-aware remote LAMMPS executor.

This facade preserves the reviewed v0.2 execution checks while adding periodic
binary restart files and state-continuous retry after scheduler interruption.
Binary restarts are deliberately treated as site-runtime-bound artifacts: a
retry must match the previous LAMMPS executable path, platform, launcher
arguments, resources, prepared bundle, model path, target, and checkpoint cadence.
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import lammps_cluster as base
import lammps_restart as restart_deck

RUNTIME_CONTRACT = "lammps-restart-runtime-v1"
RESTART_DISABLED = "disabled"
RESTART_AUTO = "auto-from-previous-attempt"
RESTART_POLICIES = {RESTART_DISABLED, RESTART_AUTO}
# -restart2info runs read_restart + info system/group/computes/fixes.  Current
# LAMMPS info.cpp prints "Current timestep number = N"; accepting the normal
# run-setup "Current step : N" spelling too makes the parser tolerant without
# weakening the numeric/cadence checks that follow.
RESTART_STEP = re.compile(
    r"Current\s+(?:time\s*)?step(?:\s+number)?\s*(?:=|:)\s*([0-9]+)",
    re.IGNORECASE,
)
RESTART_STEP_MARKER = re.compile(r"MLIPFLOW_RESTART_STEP=([0-9]+)")


def _attempt(value: str | None, output_dir: Path) -> int:
    if value is not None:
        try:
            number = int(value)
        except ValueError as exc:
            raise ValueError("attempt must be a positive integer") from exc
    else:
        name = output_dir.parent.name
        if not name.startswith("attempt-"):
            raise ValueError("cannot infer attempt number from remote workspace")
        try:
            number = int(name.rsplit("-", 1)[1])
        except ValueError as exc:
            raise ValueError("cannot infer attempt number from remote workspace") from exc
    if number < 1:
        raise ValueError("attempt must be positive")
    return number


def _resolve_executable(value: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("site LAMMPS executable is required")
    raw = Path(value).expanduser()
    if raw.is_absolute() or "/" in value:
        candidate = raw
    else:
        found = shutil.which(value)
        if not found:
            raise ValueError("site LAMMPS executable is not resolvable")
        candidate = Path(found)
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise ValueError("resolved LAMMPS executable is not an ordinary file")
    return resolved


def _runtime_record(
    *,
    attempt: int,
    node: dict[str, Any],
    framework: str,
    target: str,
    manifest_path: str,
    model_path: str,
    model_kind: str,
    lammps_interface: str | None,
    interval: int,
    executable: Path,
    launcher_prefix: list[str],
    prepared_launcher: dict[str, Any],
) -> dict[str, Any]:
    resources = node.get("resources")
    if not isinstance(resources, dict):
        raise ValueError("scheduled node resources are missing")
    return {
        "schema_version": 1,
        "runtime_contract": RUNTIME_CONTRACT,
        "attempt": attempt,
        "framework": framework,
        "target": target,
        "input_manifest_path": manifest_path,
        "model_path": model_path,
        "model_kind": model_kind,
        "lammps_interface": lammps_interface,
        "checkpoint_interval": interval,
        "resources": resources,
        "lammps_executable": str(executable),
        "launcher_prefix": launcher_prefix,
        "prepared_launcher": prepared_launcher.get("argv_after_executable", []),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "byteorder": sys.byteorder,
        },
    }


def _validate_previous_runtime(
    previous: dict[str, Any], current: dict[str, Any], attempt: int
) -> None:
    if (
        previous.get("schema_version") != 1
        or previous.get("runtime_contract") != RUNTIME_CONTRACT
    ):
        raise ValueError("previous restart runtime contract is invalid")
    if previous.get("attempt") != attempt - 1:
        raise ValueError(
            "previous restart runtime does not come from the immediately preceding attempt"
        )
    for key in (
        "framework",
        "target",
        "input_manifest_path",
        "model_path",
        "model_kind",
        "lammps_interface",
        "checkpoint_interval",
        "resources",
        "lammps_executable",
        "launcher_prefix",
        "prepared_launcher",
        "platform",
    ):
        if previous.get(key) != current.get(key):
            raise ValueError(f"binary restart runtime mismatch for {key}")


def _restart_probe_options(prepared_launcher: dict[str, Any]) -> list[str]:
    argv = prepared_launcher.get("argv_after_executable")
    if not isinstance(argv, list):
        raise ValueError("prepared launcher argv is invalid")
    try:
        stop = argv.index("-in")
    except ValueError as exc:
        raise ValueError("prepared launcher lacks -in") from exc
    options = argv[:stop]
    if any(token in {"-var", "-v"} for token in options):
        raise ValueError(
            "prepared launcher has variables before -in and cannot be probed safely"
        )
    return list(options)


def _parse_restart_step(text: str) -> int | None:
    match = RESTART_STEP.search(text)
    return int(match.group(1)) if match is not None else None


def _restart_step(executable: Path, options: list[str], candidate: Path) -> int | None:
    if not base._ordinary_file(candidate, base.MAX_RESTART_BYTES):
        return None
    completed = subprocess.run(
        [str(executable), *options, "-restart2info", str(candidate)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        check=False,
        timeout=120,
    )
    text = (completed.stdout + b"\n" + completed.stderr).decode(
        "utf-8", errors="replace"
    )
    step = _parse_restart_step(text)
    if completed.returncode == 0 and step is not None:
        return step

    # LAMMPS versions before -restart2info can still load their own binary
    # restart and expose the saved global step through an equal-style variable.
    probe = (
        f'read_restart "{candidate}"\n'
        "variable mlipflow_restart_step equal step\n"
        'print "MLIPFLOW_RESTART_STEP=${mlipflow_restart_step}"\n'
    ).encode("utf-8")
    completed = subprocess.run(
        [str(executable), *options, "-log", "none"],
        input=probe,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0:
        return None
    text = (completed.stdout + b"\n" + completed.stderr).decode(
        "utf-8", errors="replace"
    )
    match = RESTART_STEP_MARKER.search(text)
    return int(match.group(1)) if match is not None else None


def _select_restart(
    input_dir: Path,
    executable: Path,
    prepared_launcher: dict[str, Any],
    interval: int,
    total_steps: int,
) -> tuple[Path, int, str]:
    options = _restart_probe_options(prepared_launcher)
    candidates: list[tuple[int, str, Path]] = []
    for name in restart_deck.CHECKPOINT_FILES:
        path = input_dir / "restart" / name
        step = _restart_step(executable, options, path)
        if step is None:
            continue
        if step <= 0 or step >= total_steps or step % interval != 0:
            continue
        candidates.append((step, name, path))
    if not candidates:
        raise ValueError(
            "no valid approved periodic LAMMPS restart candidate can be resumed"
        )
    step, name, path = sorted(
        candidates, key=lambda item: (item[0], item[1])
    )[-1]
    return path, step, name


def _active_launcher(
    prepared_launcher: dict[str, Any],
    model_path: Path,
    interface_path: Path | None,
    input_name: str,
    restart_path: Path | None,
) -> list[str]:
    raw = prepared_launcher.get("argv_after_executable")
    if not isinstance(raw, list):
        raise ValueError("prepared launcher argv is invalid")
    argv = base._replace_model(list(raw), model_path, interface_path)
    try:
        index = argv.index("-in")
    except ValueError as exc:
        raise ValueError("prepared launcher lacks -in") from exc
    if index + 1 >= len(argv):
        raise ValueError("prepared launcher -in has no input filename")
    argv[index + 1] = input_name
    if restart_path is not None:
        argv.extend(
            ["-var", restart_deck.RESTART_VARIABLE, str(restart_path)]
        )
    return argv


def _write_runtime(path: Path, value: dict[str, Any]) -> None:
    base._write_json(path, value)
    if not base._ordinary_file(path, base.MAX_JSON_BYTES):
        raise ValueError("restart runtime contract was not written safely")


def _cleanup_success(output_dir: Path) -> None:
    for name in (*restart_deck.CHECKPOINT_FILES, "restart-runtime.json"):
        path = output_dir / name
        try:
            if path.is_file() and not path.is_symlink():
                path.unlink()
        except OSError:
            pass


def run(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "FAIL",
        "node_id": args.node_id,
    }
    try:
        attempt = _attempt(args.attempt, output_dir)
        node = base._project_node(
            Path(args.project).expanduser().resolve(), args.node_id
        )
        if str(node.get("uses", "")).split("@", 1)[0] != "lammps-md":
            raise ValueError("scheduled node does not use lammps-md")
        if node.get("backend") != "ssh-slurm":
            raise ValueError("scheduled LAMMPS execute requires ssh-slurm")
        parameters = node.get("parameters")
        if (
            not isinstance(parameters, dict)
            or parameters.get("operation") != "execute"
        ):
            raise ValueError("scheduled LAMMPS node must declare operation=execute")
        target = parameters.get("target")
        if target not in base.TARGETS:
            raise ValueError("target must be cpu or gpu")
        policy = parameters.get("restart_policy", RESTART_DISABLED)
        if policy not in RESTART_POLICIES:
            raise ValueError("restart_policy is invalid")
        interval = parameters.get("checkpoint_interval")
        if policy == RESTART_AUTO:
            if (
                isinstance(interval, bool)
                or not isinstance(interval, int)
                or interval <= 0
            ):
                raise ValueError(
                    "auto restart requires a positive checkpoint_interval"
                )
        elif interval is not None and (
            isinstance(interval, bool)
            or not isinstance(interval, int)
            or interval <= 0
        ):
            raise ValueError("checkpoint_interval must be positive when provided")

        manifest_path = input_dir / "lammps-input-manifest.json"
        manifest = base._read_mapping(manifest_path)
        input_manifest_path = base._input_manifest_record(node.get("inputs"))
        if (
            manifest.get("schema_version") != 1
            or manifest.get("plugin_id") != "lammps-md"
            or manifest.get("operation") != "lammps-prepare"
            or manifest.get("status") != "OK"
            or manifest.get("preparation_contract") != base.PREPARATION_CONTRACT
            or manifest.get("runtime_model_variable") != "MODEL_FILE"
        ):
            raise ValueError(
                "staged LAMMPS input manifest is not execution-ready"
            )
        marker = manifest.get("completion_marker")
        model = manifest.get("model")
        md = manifest.get("md")
        if not isinstance(model, dict) or not isinstance(md, dict):
            raise ValueError("input manifest model/MD record is incomplete")
        framework = model.get("framework")
        if framework not in base.FRAMEWORKS:
            raise ValueError("unsupported LAMMPS framework")
        interface = model.get("lammps_interface") if framework == "m3gnet" else None
        if framework == "m3gnet" and interface is None:
            interface = "matgl"
        model_contract = base.MODEL_CONTRACTS.get((str(framework), interface))
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
        if marker != f"MLIPFLOW_LAMMPS_COMPLETED step={steps}":
            raise ValueError(
                "completion marker does not bind the approved step count"
            )
        if interval is not None and interval > steps:
            raise ValueError(
                "checkpoint_interval cannot exceed approved total steps"
            )

        generated = base._generated_records(manifest)
        selected_name = f"in.{target}.lammps"
        for name in ("structure.data", selected_name):
            record = generated.get(name)
            staged = input_dir / "lammps" / name
            if record is None or not base._ordinary_file(
                staged, base.MAX_INPUT_BYTES
            ):
                raise ValueError(f"staged prepared input is missing: {name}")
        source_deck = (input_dir / "lammps" / selected_name).read_text(
            encoding="utf-8"
        )
        requires_interface_path = interface in {"gnnp", "m3gnet"}
        if (
            marker not in source_deck
            or "${MODEL_FILE}" not in source_deck
            or ("${INTERFACE_PATH}" in source_deck) != requires_interface_path
        ):
            raise ValueError(
                "selected input deck lacks its approved portable runtime contract"
            )
        relative_model = str(model.get("relative_path", ""))
        if relative_model and relative_model in source_deck:
            raise ValueError(
                "selected input deck embeds the site-relative model path"
            )

        model_root = args.model_root or os.environ.get(
            "MLIPFLOW_MODEL_ROOT", ""
        )
        model_path = base._resolve_model(model_root, model)
        prepared_launcher = base._launcher(manifest, str(target))
        interface_value = args.interface_path or os.environ.get(
            "MLIPFLOW_LAMMPS_INTERFACE_PATH", ""
        )
        interface_path = base._resolve_interface_path(
            interface_value, requires_interface_path
        )
        launcher_prefix = base._launcher_prefix(args.launcher_json)
        executable = _resolve_executable(args.lammps_bin)

        resumed = policy == RESTART_AUTO and attempt > 1
        if resumed and interval is None:
            raise ValueError("restart attempt lacks checkpoint interval")
        segment_start_step = 0
        restart_path: Path | None = None
        restart_name: str | None = None
        restart_from_attempt: int | None = None

        current_runtime = _runtime_record(
            attempt=attempt,
            node=node,
            framework=str(framework),
            target=str(target),
            manifest_path=input_manifest_path,
            model_path=str(model.get("relative_path")),
            model_kind=str(model.get("kind")),
            lammps_interface=str(interface) if interface is not None else None,
            interval=int(interval or steps),
            executable=executable,
            launcher_prefix=launcher_prefix,
            prepared_launcher=prepared_launcher,
        )
        if resumed:
            previous_runtime = base._read_mapping(
                input_dir / "restart" / "restart-runtime.json"
            )
            _validate_previous_runtime(previous_runtime, current_runtime, attempt)
            restart_path, segment_start_step, restart_name = _select_restart(
                input_dir, executable, prepared_launcher, int(interval), steps
            )
            restart_from_attempt = attempt - 1
            active_text = restart_deck.build_resume(
                source_deck, int(interval), steps
            )
        elif interval is not None:
            active_text = restart_deck.instrument_fresh(
                source_deck, int(interval), steps
            )
        else:
            active_text = (
                source_deck
                if source_deck.endswith("\n")
                else source_deck + "\n"
            )

        shutil.copy2(
            input_dir / "lammps" / "structure.data",
            output_dir / "structure.data",
        )
        active_name = "in.active.lammps"
        active_path = output_dir / active_name
        active_path.write_text(active_text, encoding="utf-8")
        current_runtime.update(
            {
                "resumed": resumed,
                "segment_start_step": segment_start_step,
                "restart_from_attempt": restart_from_attempt,
                "selected_restart": restart_name,
                "active_deck_path": str(active_path),
            }
        )
        runtime_path = output_dir / "restart-runtime.json"
        if interval is not None:
            _write_runtime(runtime_path, current_runtime)

        launch_argv = _active_launcher(
            prepared_launcher,
            model_path,
            interface_path,
            active_name,
            restart_path,
        )
        log_path = output_dir / "lammps.log"
        screen_path = output_dir / "lammps.screen.log"
        stdout_path = output_dir / "lammps.stdout.log"
        stderr_path = output_dir / "lammps.stderr.log"
        command = [
            *launcher_prefix,
            str(executable),
            *launch_argv,
            "-log",
            log_path.name,
            "-screen",
            screen_path.name,
        ]
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
        if not base._contains_marker(log_path, str(marker)):
            raise ValueError(
                "LAMMPS log lacks the approved end-of-script completion marker"
            )
        lammps_version = base._version_from_logs(
            log_path, screen_path, stdout_path
        )
        artifacts = [
            base._artifact(
                output_dir / "trajectory.lammpstrj",
                "trajectory.lammpstrj",
                base.MAX_TRAJECTORY_BYTES,
            ),
            base._artifact(
                output_dir / "final.data",
                "final.data",
                base.MAX_FINAL_DATA_BYTES,
            ),
            base._artifact(
                output_dir / "final.restart",
                "final.restart",
                base.MAX_RESTART_BYTES,
            ),
            base._artifact(
                log_path, "lammps.log", base.MAX_LOG_BYTES
            ),
            base._artifact(
                screen_path, "lammps.screen.log", base.MAX_LOG_BYTES
            ),
        ]
        if base._ordinary_file(stdout_path, base.MAX_LOG_BYTES):
            artifacts.append(
                base._artifact(
                    stdout_path, "lammps.stdout.log", base.MAX_LOG_BYTES
                )
            )
        if base._ordinary_file(stderr_path, base.MAX_LOG_BYTES):
            artifacts.append(
                base._artifact(
                    stderr_path, "lammps.stderr.log", base.MAX_LOG_BYTES
                )
            )

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
            "segment_start_step": segment_start_step,
            "completion_marker": marker,
            "launcher": {
                "site_launcher_used": bool(launcher_prefix),
                "prepared_argv_after_executable": prepared_launcher[
                    "argv_after_executable"
                ],
                "required_packages": prepared_launcher.get(
                    "required_packages", []
                ),
            },
            "restart": {
                "policy": policy,
                "checkpoint_interval": interval,
                "resumed": resumed,
                "from_attempt": restart_from_attempt,
                "selected_checkpoint": restart_name,
                "runtime_compatibility_checked": resumed,
                "lammps_executable": current_runtime["lammps_executable"],
                "launcher_prefix": current_runtime["launcher_prefix"],
                "prepared_launcher": current_runtime["prepared_launcher"],
                "platform": current_runtime["platform"],
                "bitwise_exact_guaranteed": False,
            },
            "artifacts": artifacts,
        }
        result_path = output_dir / "lammps-execution-result.json"
        base._write_json(result_path, result)
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
                "segment_start_step": segment_start_step,
                "restart_from_attempt": restart_from_attempt,
                "selected_checkpoint": restart_name,
                "lammps_version": lammps_version,
            }
        )
        base._write_json(output_dir / "cluster-run-report.json", report)
        if interval is not None:
            _cleanup_success(output_dir)
        return 0
    except Exception as exc:
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc(limit=8)
        base._write_json(output_dir / "cluster-run-report.json", report)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run/restart reviewed LAMMPS MLIP input on a cluster"
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-root")
    parser.add_argument("--interface-path")
    parser.add_argument("--lammps-bin", required=True)
    parser.add_argument("--launcher-json", default="[]")
    parser.add_argument("--attempt")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
