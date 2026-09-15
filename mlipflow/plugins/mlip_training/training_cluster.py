"""Remote-side runner for scheduled bundled MLIP training.

The site-owned run.sh selects the Python/framework environment and provides
cluster-local data/model roots. This helper resolves only approved relative
references below those roots and calls the
bundled training_wrapper in-process (never through a shell).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import sys
import tarfile
import traceback
from pathlib import Path, PurePosixPath
from typing import Any

FRAMEWORKS = {"deepmd", "m3gnet", "chgnet", "mace"}
OPERATIONS = {"train", "finetune"}
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")


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


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("relative_path must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe relative_path: {value!r}")
    return value


def _site_root(cli_value: str | None, env_name: str) -> Path:
    value = cli_value or os.environ.get(env_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"site template must provide {env_name} or the matching CLI root")
    return Path(value).expanduser()


def _resolve_under(root: Path, relative: str, kind: str) -> Path:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"site artifact root does not exist: {root}")
    candidate = (root / _safe_relative(relative)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"artifact reference escapes site root: {relative}") from exc
    if kind == "file" and not candidate.is_file():
        raise ValueError(f"referenced file does not exist: {candidate}")
    if kind == "directory" and not candidate.is_dir():
        raise ValueError(f"referenced directory does not exist: {candidate}")
    return candidate


def _reference(path: Path, *, id_key: str, default_kind: str) -> dict[str, str]:
    raw = _load_mapping(path)
    if raw.get("schema_version") != 1:
        raise ValueError(f"{path.name} schema_version must be 1")
    artifact_id = raw.get(id_key)
    if not isinstance(artifact_id, str) or not artifact_id:
        raise ValueError(f"{path.name} requires {id_key}")
    relative = _safe_relative(raw.get("relative_path", artifact_id))
    kind = raw.get("kind", default_kind)
    if kind not in {"file", "directory"}:
        raise ValueError(f"{path.name} kind must be file or directory")
    return {
        "id": artifact_id,
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
        raise ValueError(f"project does not contain exactly one node {node_id!r}")
    return matches[0]


def _framework_default_dataset_kind(framework: str) -> str:
    return "directory" if framework == "deepmd" else "file"


def _write_report(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cluster-run-report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _publish_model(
    source: Path,
    model_root: Path,
    framework: str,
    model_id: str,
    relative_path: str,
) -> dict[str, str]:
    """Publish one verified model without overwriting an existing registry entry."""

    if not SAFE_ID.fullmatch(model_id):
        raise ValueError("publish_model_id must be a safe logical id")
    relative = _safe_relative(relative_path)
    root = model_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"site model root does not exist: {root}")
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.parent.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError("published model path escapes site model root") from exc
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"published model destination already exists: {relative}")

    created = False
    try:
        if framework == "m3gnet":
            destination.mkdir()
            created = True
            file_count = 0
            with tarfile.open(source, "r:gz") as archive:
                for member in archive.getmembers():
                    member_path = PurePosixPath(member.name)
                    if (
                        member_path.is_absolute()
                        or not member_path.parts
                        or member_path.parts[0] != "model"
                        or any(part in {"", ".", ".."} for part in member_path.parts)
                        or member.issym()
                        or member.islnk()
                        or not (member.isdir() or member.isfile())
                    ):
                        raise ValueError("M3GNet model archive contains an unsafe member")
                    stripped = member_path.parts[1:]
                    if not stripped:
                        continue
                    target = destination.joinpath(*stripped)
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError("M3GNet model archive member is unreadable")
                    with target.open("xb") as output:
                        shutil.copyfileobj(stream, output)
                    file_count += 1
            if file_count == 0:
                raise ValueError("M3GNet model archive contains no files")
            kind = "directory"
        else:
            if not source.is_file() or source.is_symlink():
                raise ValueError("trained model artifact must be an ordinary file")
            with source.open("rb") as input_stream, destination.open("xb") as output:
                created = True
                shutil.copyfileobj(input_stream, output)
            kind = "file"
    except Exception:
        if created:
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            elif destination.is_file() and not destination.is_symlink():
                destination.unlink()
        raise
    return {
        "schema_version": 1,
        "model_id": model_id,
        "framework": framework,
        "relative_path": relative,
        "kind": kind,
    }


def run(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "FAIL",
        "node_id": args.node_id,
    }
    try:
        node = _project_node(Path(args.project).expanduser().resolve(), args.node_id)
        parameters = node.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError("training node parameters must be an object")
        framework = parameters.get("framework")
        operation = parameters.get("operation", "train")
        if framework not in FRAMEWORKS or operation not in OPERATIONS:
            raise ValueError("unsupported framework/operation")
        seed = parameters.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")

        config = input_dir / "training-config.json"
        dataset_file = input_dir / "dataset-reference.json"
        if not config.is_file() or not dataset_file.is_file():
            raise ValueError("staged training-config.json and dataset-reference.json are required")
        dataset_ref = _reference(
            dataset_file,
            id_key="dataset_id",
            default_kind=_framework_default_dataset_kind(str(framework)),
        )
        data_root = _site_root(args.data_root, "MLIPFLOW_DATA_ROOT")
        data_path = _resolve_under(
            data_root, dataset_ref["relative_path"], dataset_ref["kind"]
        )

        wrapper_args = [
            "--framework",
            str(framework),
            "--operation",
            str(operation),
            "--config",
            str(config),
            "--data",
            str(data_path),
            "--output",
            str(output_dir / "model-artifact"),
            "--result-manifest",
            str(output_dir / "training-result.json"),
            "--seed",
            str(seed),
            "--device",
            str(parameters.get("device")),
            "--precision",
            str(parameters.get("precision")),
        ]
        foundation_ref: dict[str, str] | None = None
        foundation_report: dict[str, str] | None = None
        if operation == "finetune":
            foundation_file = input_dir / "foundation-model-reference.json"
            if not foundation_file.is_file():
                raise ValueError("finetune requires foundation-model-reference.json")
            default_kind = "directory" if framework == "m3gnet" else "file"
            foundation_ref = _reference(
                foundation_file,
                id_key="model_id",
                default_kind=default_kind,
            )
            foundation_root_value = (
                args.foundation_model_root
                or os.environ.get("MLIPFLOW_FOUNDATION_MODEL_ROOT")
                or args.model_root
                or os.environ.get("MLIPFLOW_MODEL_ROOT")
            )
            foundation_root = _site_root(
                foundation_root_value, "MLIPFLOW_FOUNDATION_MODEL_ROOT"
            )
            foundation_path = _resolve_under(
                foundation_root,
                foundation_ref["relative_path"],
                foundation_ref["kind"],
            )
            foundation_report = {
                **foundation_ref,
                "resolved_path": str(foundation_path),
            }
            wrapper_args.extend(
                [
                    "--foundation-model",
                    str(foundation_path),
                ]
            )

        sys.path.insert(0, str(input_dir))
        import training_wrapper

        output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = output_dir / "training.stdout.log"
        stderr_path = output_dir / "training.stderr.log"
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                return_code = int(training_wrapper.main(wrapper_args))
        result_path = output_dir / "training-result.json"
        result = _load_mapping(result_path) if result_path.is_file() else None
        if parameters.get("validation_profile") == "deepmd-curve" and return_code == 0:
            if __package__:
                from .deepmd_curve import export_training_evidence
            else:
                from deepmd_curve import export_training_evidence
            export_training_evidence(output_dir, result, dataset_ref["id"])
        published_model: dict[str, str] | None = None
        publish_id = parameters.get("publish_model_id")
        publish_relative = parameters.get("publish_model_relative_path")
        if publish_id is not None or publish_relative is not None:
            if not isinstance(publish_id, str) or not isinstance(publish_relative, str):
                raise ValueError(
                    "publish_model_id and publish_model_relative_path must be supplied together"
                )
            model_root = _site_root(args.model_root, "MLIPFLOW_MODEL_ROOT")
            published_model = _publish_model(
                output_dir / "model-artifact",
                model_root,
                str(framework),
                publish_id,
                publish_relative,
            )
            (output_dir / "model-reference.json").write_text(
                json.dumps(published_model, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        report.update(
            {
                "status": "OK" if return_code == 0 else "FAIL",
                "return_code": return_code,
                "framework": framework,
                "operation": operation,
                "dataset": {**dataset_ref, "resolved_path": str(data_path)},
                "config_path": str(config),
                "foundation_model": foundation_report,
                "published_model": published_model,
                "framework_version": result.get("framework_version")
                if isinstance(result, dict)
                else None,
            }
        )
        _write_report(output_dir, report)
        return return_code
    except Exception as exc:
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc(limit=8)
        _write_report(output_dir, report)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an approved MLIP training node on a cluster"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--project", required=True)
    execute.add_argument("--node-id", required=True)
    execute.add_argument("--input-dir", required=True)
    execute.add_argument("--output-dir", required=True)
    execute.add_argument("--data-root")
    execute.add_argument("--model-root")
    execute.add_argument("--foundation-model-root")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
