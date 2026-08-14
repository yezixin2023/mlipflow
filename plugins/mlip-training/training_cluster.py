"""Remote-side runner for scheduled bundled MLIP training.

The site-owned run.sh selects the Python/framework environment and provides
cluster-local data/model roots. This helper resolves only approved relative
references below those roots, verifies content fingerprints, and calls the
bundled training_wrapper in-process (never through a shell).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path, PurePosixPath
from typing import Any

FRAMEWORKS = {"deepmd", "m3gnet", "chgnet", "mace"}
OPERATIONS = {"train", "finetune"}
FINGERPRINT_PREFIX = "sha256:"


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


def _valid_fingerprint(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith(FINGERPRINT_PREFIX)
        and len(value) == 71
        and all(char in "0123456789abcdef" for char in value[7:])
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return FINGERPRINT_PREFIX + digest.hexdigest()


def fingerprint(path: Path) -> str:
    """Return file SHA-256 or deterministic tree-sha256-v1 for a directory."""
    path = path.resolve()
    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir():
        raise ValueError(f"artifact does not exist: {path}")
    digest = hashlib.sha256()
    files = [item for item in path.rglob("*") if item.is_file()]
    if not files:
        raise ValueError(f"artifact directory is empty: {path}")
    for item in sorted(files, key=lambda value: value.relative_to(path).as_posix()):
        if item.is_symlink():
            raise ValueError(f"artifact tree contains symlink: {item}")
        relative = item.relative_to(path).as_posix()
        record = f"{relative}\0{item.stat().st_size}\0{_sha256_file(item)}\n"
        digest.update(record.encode("utf-8"))
    return FINGERPRINT_PREFIX + digest.hexdigest()


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
    declared = raw.get("fingerprint")
    if not _valid_fingerprint(declared):
        raise ValueError(f"{path.name} fingerprint must be sha256:<64 lowercase hex>")
    return {
        "id": artifact_id,
        "relative_path": relative,
        "kind": kind,
        "fingerprint": str(declared),
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
        config_fingerprint = _sha256_file(config)
        if config_fingerprint != parameters.get("config_fingerprint"):
            raise ValueError("staged config fingerprint differs from approved parameters")

        dataset_ref = _reference(
            dataset_file,
            id_key="dataset_id",
            default_kind=_framework_default_dataset_kind(str(framework)),
        )
        if dataset_ref["fingerprint"] != parameters.get("dataset_fingerprint"):
            raise ValueError("dataset reference fingerprint differs from approved parameters")
        data_root = _site_root(args.data_root, "MLIPFLOW_DATA_ROOT")
        data_path = _resolve_under(
            data_root, dataset_ref["relative_path"], dataset_ref["kind"]
        )
        observed_dataset = fingerprint(data_path)
        if observed_dataset != dataset_ref["fingerprint"]:
            raise ValueError("cluster dataset content fingerprint differs from approved reference")

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
            "--dataset-fingerprint",
            observed_dataset,
            "--config-fingerprint",
            config_fingerprint,
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
            approved_foundation = parameters.get("foundation_model_fingerprint")
            if foundation_ref["fingerprint"] != approved_foundation:
                raise ValueError("foundation reference fingerprint differs from approved parameters")
            model_root = _site_root(args.model_root, "MLIPFLOW_MODEL_ROOT")
            foundation_path = _resolve_under(
                model_root,
                foundation_ref["relative_path"],
                foundation_ref["kind"],
            )
            observed_foundation = fingerprint(foundation_path)
            if observed_foundation != foundation_ref["fingerprint"]:
                raise ValueError("cluster foundation model fingerprint differs from approved reference")
            foundation_report = {
                **foundation_ref,
                "observed_fingerprint": observed_foundation,
            }
            wrapper_args.extend(
                [
                    "--foundation-model",
                    str(foundation_path),
                    "--foundation-model-fingerprint",
                    observed_foundation,
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
        report.update(
            {
                "status": "OK" if return_code == 0 else "FAIL",
                "return_code": return_code,
                "framework": framework,
                "operation": operation,
                "dataset": {**dataset_ref, "observed_fingerprint": observed_dataset},
                "config_fingerprint": config_fingerprint,
                "foundation_model": foundation_report,
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
    fingerprint_parser = sub.add_parser("fingerprint")
    fingerprint_parser.add_argument("path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fingerprint":
        print(fingerprint(Path(args.path)))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
