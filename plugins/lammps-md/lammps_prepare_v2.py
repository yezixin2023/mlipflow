"""Execution-ready facade for deterministic LAMMPS input preparation.

The reviewed v0.1 generator remains the source of the scientific input deck. This
facade adds one machine-readable completion marker after the final restart write
and rebinds the generated-file fingerprints. The marker lets scheduled execution
prove that LAMMPS reached the end of the approved input script without parsing
human-oriented log formatting.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

PREPARATION_CONTRACT = "lammps-md-input-v2"


def _load_legacy():
    path = Path(__file__).resolve().with_name("lammps_prepare.py")
    spec = importlib.util.spec_from_file_location("_mlipflow_lammps_prepare_v1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load bundled lammps_prepare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = _load_legacy()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def prepare(
    structure: Path,
    model_reference: Path,
    config_path: Path,
    output_dir: Path,
    structure_format: str | None = None,
) -> dict[str, Any]:
    manifest = legacy.prepare(
        structure,
        model_reference,
        config_path,
        output_dir,
        structure_format,
    )
    steps = manifest.get("md", {}).get("steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise legacy.ContractError("prepared manifest has invalid MD step count")
    marker = f"MLIPFLOW_LAMMPS_COMPLETED step={steps}"

    generated = manifest.get("generated_files")
    if not isinstance(generated, list):
        raise legacy.ContractError("prepared manifest generated_files must be a list")
    by_name = {
        str(item.get("name")): item
        for item in generated
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    targets = manifest.get("md", {}).get("targets")
    if not isinstance(targets, list) or not targets:
        raise legacy.ContractError("prepared manifest targets must be a non-empty list")

    for target in targets:
        name = f"in.{target}.lammps"
        path = output_dir / name
        if name not in by_name or not path.is_file():
            raise legacy.ContractError(f"prepared input deck is missing: {name}")
        text = path.read_text(encoding="utf-8")
        if "MLIPFLOW_LAMMPS_COMPLETED" in text:
            raise legacy.ContractError("legacy deck unexpectedly contains a completion marker")
        if not text.endswith("\n"):
            text += "\n"
        text += f'print           "{marker}"\n'
        path.write_text(text, encoding="utf-8")
        by_name[name]["sha256"] = _sha256(path)
        by_name[name]["size_bytes"] = path.stat().st_size

    manifest["preparation_contract"] = PREPARATION_CONTRACT
    manifest["completion_marker"] = marker
    notes = manifest.setdefault("notes", [])
    if isinstance(notes, list):
        notes.append(
            "Each execution-ready deck prints the exact approved completion marker only after final.data and final.restart are written."
        )
    _write_json(output_dir / "lammps-input-manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare execution-ready deterministic MLIP LAMMPS inputs"
    )
    parser.add_argument("--structure", required=True)
    parser.add_argument("--model-reference", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--structure-format")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepare(
        Path(args.structure),
        Path(args.model_reference),
        Path(args.config),
        Path(args.output_dir),
        args.structure_format,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
