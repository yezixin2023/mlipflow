"""Static, read-only audit of historical MLIP training entrypoints.

The audit records source paths and extracts enough Python/shell structure to
decide whether a historical script is suitable for a production wrapper.  It
never imports a training framework and never executes the inspected source.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any


PLUGIN_ID = "mlip-training"
ABSOLUTE_PATH = re.compile(r"/(?:public|home|Users|opt|usr|data|scratch)/[^\s\"']+")


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def _call_name(call: ast.Call) -> str:
    parts: list[str] = []
    value: ast.AST = call.func
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _system_summary(value: Any) -> dict[str, Any]:
    systems = _mapping(value).get("systems", [])
    if isinstance(systems, str):
        systems = [systems]
    if not isinstance(systems, list) or not all(isinstance(item, str) for item in systems):
        systems = []
    portable_ids = sorted(Path(item.rstrip("/")).name for item in systems)
    return {
        "system_count": len(systems),
        "unique_system_id_count": len(set(portable_ids)),
        "system_names": portable_ids,
        "batch_size": _mapping(value).get("batch_size"),
    }


def _deepmd_json_audit(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return {
            "language": "json",
            "parse_status": "FAIL",
            "issues": [
                {
                    "severity": "high",
                    "code": "deepmd.invalid_json",
                    "message": f"line {exc.lineno}: {exc.msg}",
                }
            ],
        }
    if not isinstance(value, dict):
        return {
            "language": "json",
            "parse_status": "FAIL",
            "issues": [
                {
                    "severity": "high",
                    "code": "deepmd.config_not_object",
                    "message": "DeepMD config root is not an object",
                }
            ],
        }
    model = _mapping(value.get("model"))
    descriptor = _mapping(model.get("descriptor"))
    fitting = _mapping(model.get("fitting_net"))
    training = _mapping(value.get("training"))
    train_summary = _system_summary(training.get("training_data"))
    validation_summary = _system_summary(training.get("validation_data"))
    absolute_paths = sorted(set(ABSOLUTE_PATH.findall(text)))
    issues: list[dict[str, str]] = []
    if absolute_paths:
        issues.append(
            {
                "severity": "high",
                "code": "deepmd.absolute_paths",
                "message": "config contains host-specific absolute dataset paths",
            }
        )
    if training.get("seed") is None:
        issues.append(
            {
                "severity": "high",
                "code": "deepmd.training_seed_missing",
                "message": "training.seed is not explicit",
            }
        )
    issues.append(
        {
            "severity": "high",
            "code": "training.standard_result_missing",
            "message": "historical source does not emit the MLIPFlow training result contract",
        }
    )
    return {
        "language": "json",
        "parse_status": "OK",
        "absolute_paths": absolute_paths,
        "configuration": {
            "type_map": model.get("type_map"),
            "descriptor": {
                key: descriptor.get(key)
                for key in (
                    "type",
                    "rcut_smth",
                    "rcut",
                    "sel",
                    "neuron",
                    "axis_neuron",
                    "seed",
                    "precision",
                    "attn",
                    "attn_layer",
                )
                if key in descriptor
            },
            "fitting_net": {
                key: fitting.get(key)
                for key in ("neuron", "resnet_dt", "seed", "precision")
                if key in fitting
            },
            "learning_rate": _mapping(value.get("learning_rate")),
            "loss": _mapping(value.get("loss")),
            "training": {
                key: training.get(key)
                for key in ("numb_steps", "seed", "disp_file", "disp_freq", "save_freq")
                if key in training
            },
        },
        "data": {
            "training": train_summary,
            "validation": validation_summary,
        },
        "issues": issues,
    }


def _python_audit(path: Path, text: str) -> dict[str, Any]:
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return {
            "language": "python",
            "parse_status": "FAIL",
            "issues": [
                {
                    "severity": "high",
                    "code": "python.syntax_error",
                    "message": f"line {exc.lineno}: {exc.msg}",
                }
            ],
        }
    imports: set[str] = set()
    absolute_paths: set[str] = set()
    calls: list[dict[str, Any]] = []
    seeds: list[dict[str, Any]] = []
    devices: list[dict[str, Any]] = []
    split_settings: list[dict[str, Any]] = []
    outputs: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            absolute_paths.update(ABSOLUTE_PATH.findall(node.value))
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            keywords = {
                keyword.arg: _literal(keyword.value)
                for keyword in node.keywords
                if keyword.arg is not None
            }
            interesting = {
                key: value
                for key, value in keywords.items()
                if key
                in {
                    "accelerator",
                    "batch_size",
                    "criterion",
                    "epochs",
                    "frac_list",
                    "learning_rate",
                    "max_epochs",
                    "optimizer",
                    "precision",
                    "random_state",
                    "scheduler",
                    "seed",
                    "shuffle",
                    "targets",
                    "train_ratio",
                    "use_device",
                    "val_ratio",
                }
            }
            if interesting:
                calls.append({"call": name, "line": node.lineno, "keywords": interesting})
            for key in ("seed", "random_state"):
                if key in keywords:
                    seeds.append({"call": name, "key": key, "value": keywords[key]})
            for key in ("device", "use_device", "accelerator"):
                if key in keywords:
                    devices.append({"call": name, "key": key, "value": keywords[key]})
            if any(key in keywords for key in ("frac_list", "train_ratio", "val_ratio", "shuffle")):
                split_settings.append({"call": name, "line": node.lineno, "keywords": interesting})
            if name.endswith((".save", ".to_excel", ".to_csv", ".write")) and node.args:
                value = _literal(node.args[0])
                if isinstance(value, str):
                    outputs.append(value)

    top_level_execution = any(
        isinstance(node, (ast.Expr, ast.For, ast.While, ast.With, ast.Try))
        or (isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Call))
        for node in tree.body
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
    )
    issues: list[dict[str, str]] = []
    if absolute_paths:
        issues.append(
            {
                "severity": "high",
                "code": "python.absolute_paths",
                "message": "source contains host-specific absolute paths",
            }
        )
    if not seeds:
        issues.append(
            {
                "severity": "high",
                "code": "python.seed_missing",
                "message": "no explicit seed/random_state keyword was found",
            }
        )
    if top_level_execution:
        issues.append(
            {
                "severity": "medium",
                "code": "python.top_level_execution",
                "message": "importing the source can execute training or write files",
            }
        )
    framework_roots = sorted(
        root
        for root in {name.split(".")[0] for name in imports}
        if root in {"chgnet", "m3gnet", "matgl", "lightning", "pytorch_lightning", "tensorflow"}
    )
    if "m3gnet" in framework_roots and "matgl" in framework_roots:
        issues.append(
            {
                "severity": "high",
                "code": "python.mixed_m3gnet_matgl",
                "message": "source mixes legacy M3GNet and MatGL APIs and needs version-specific refactoring",
            }
        )
    issues.append(
        {
            "severity": "high",
            "code": "training.standard_result_missing",
            "message": "historical source does not emit the MLIPFlow training result contract",
        }
    )
    return {
        "language": "python",
        "parse_status": "OK",
        "imports": sorted(imports),
        "framework_roots": framework_roots,
        "absolute_paths": sorted(absolute_paths),
        "calls": calls,
        "seeds": seeds,
        "devices": devices,
        "split_settings": split_settings,
        "declared_outputs": sorted(set(outputs)),
        "top_level_execution": top_level_execution,
        "issues": issues,
    }


def _shell_audit(path: Path, text: str) -> dict[str, Any]:
    lines = text.splitlines()
    directives: dict[str, str | bool] = {}
    absolute_paths = sorted(set(ABSOLUTE_PATH.findall(text)))
    issues: list[dict[str, str]] = []
    commands: list[dict[str, Any]] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#SBATCH"):
            body = stripped.removeprefix("#SBATCH").strip()
            if "=" in body:
                key, value = body.split("=", 1)
                directives[key] = value
            else:
                fields = body.split(maxsplit=1)
                directives[fields[0]] = fields[1] if len(fields) == 2 else True

    command_start: int | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("mace_run_train") or (
            stripped.startswith("python") and "run_train.py" in stripped
        ):
            command_start = index
            break
    flags: dict[str, Any] = {}
    if command_start is not None:
        command_lines: list[str] = []
        index = command_start
        while index < len(lines):
            stripped = lines[index].strip()
            command_lines.append(stripped.removesuffix("\\").rstrip())
            continued = stripped.endswith("\\")
            next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
            if not continued:
                if next_line.startswith("--"):
                    issues.append(
                        {
                            "severity": "critical",
                            "code": "shell.missing_continuation",
                            "message": (
                                f"line {index + 1} ends the trainer command, but line {index + 2} "
                                "starts with another option"
                            ),
                        }
                    )
                break
            index += 1
        command_text = " ".join(command_lines)
        try:
            tokens = shlex.split(command_text)
        except ValueError as exc:
            issues.append({"severity": "high", "code": "shell.parse_error", "message": str(exc)})
            tokens = []
        option_index = 0
        while option_index < len(tokens):
            token = tokens[option_index]
            if token.startswith("--"):
                body = token[2:]
                if "=" in body:
                    key, value = body.split("=", 1)
                    flags[key] = value
                elif option_index + 1 < len(tokens) and not tokens[option_index + 1].startswith(
                    "--"
                ):
                    flags[body] = tokens[option_index + 1]
                    option_index += 1
                else:
                    flags[body] = True
            option_index += 1
        commands.append(
            {
                "line": command_start + 1,
                "argv": tokens,
                "captured_through_line": index + 1,
            }
        )
    else:
        issues.append(
            {
                "severity": "high",
                "code": "shell.trainer_missing",
                "message": "no recognized training command was found",
            }
        )
    if absolute_paths:
        issues.append(
            {
                "severity": "high",
                "code": "shell.absolute_paths",
                "message": "source contains host-specific absolute paths",
            }
        )
    if "seed" not in flags:
        issues.append(
            {
                "severity": "high",
                "code": "shell.seed_missing",
                "message": "captured trainer command has no explicit --seed",
            }
        )
    issues.append(
        {
            "severity": "high",
            "code": "training.standard_result_missing",
            "message": "historical source does not emit the MLIPFlow training result contract",
        }
    )
    return {
        "language": "shell",
        "parse_status": "OK",
        "scheduler_directives": directives,
        "absolute_paths": absolute_paths,
        "commands": commands,
        "trainer_flags": flags,
        "issues": issues,
    }


def audit_source(framework: str, path: Path) -> dict[str, Any]:
    if framework not in {"deepmd", "m3gnet", "chgnet", "mace"}:
        raise ValueError(f"unsupported framework label: {framework}")
    if not path.is_file():
        raise ValueError(f"source is not a regular file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read source {path}: {exc}") from exc
    if framework == "deepmd" and path.suffix.lower() == ".json":
        detail = _deepmd_json_audit(text)
    elif path.suffix == ".py":
        detail = _python_audit(path, text)
    else:
        detail = _shell_audit(path, text)
    return {
        "framework": framework,
        "source_name": path.name,
        **detail,
    }


def _redact_private_paths(value: Any) -> Any:
    """Remove host/user path values while preserving issue structure."""

    if isinstance(value, dict):
        result = {
            key: _redact_private_paths(item)
            for key, item in value.items()
            if key != "absolute_paths"
        }
        if isinstance(value.get("absolute_paths"), list):
            result["absolute_path_count"] = len(value["absolute_paths"])
        return result
    if isinstance(value, list):
        return [_redact_private_paths(item) for item in value]
    if isinstance(value, str):
        return ABSOLUTE_PATH.sub("<redacted-absolute-path>", value)
    return value


def build_report(sources: list[tuple[str, Path]]) -> dict[str, Any]:
    if not sources:
        raise ValueError("at least one source is required")
    records = [
        _redact_private_paths(audit_source(framework, path)) for framework, path in sources
    ]
    counts: dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for record in records:
        for issue in record["issues"]:
            counts[issue["severity"]] += 1
    return {
        "schema_version": 1,
        "plugin_id": PLUGIN_ID,
        "status": "OK",
        "mode": "read-only-source-audit",
        "scientific_execution_status": "EXTERNAL_VALIDATION_PENDING",
        "sources": records,
        "summary": {"source_count": len(records), "issue_counts": counts},
        "claims": {
            "training_executed": False,
            "numerical_parity_established": False,
            "source_code_executed": False,
        },
        "redaction": {
            "private_absolute_path_values_removed": True,
            "source_record": "basename",
        },
    }


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_argument(value: str) -> tuple[str, Path]:
    framework, separator, path = value.partition("=")
    if not separator or not framework or not path:
        raise argparse.ArgumentTypeError("source must be FRAMEWORK=PATH")
    return framework, Path(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=_source_argument, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        _write_new_json(args.result, build_report(args.source))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
