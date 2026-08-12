"""Run and normalize one prepared static VASP calculation without a shell.

The wrapper is staged by MLIPFlow into a fresh SSH-SLURM run directory.  It is
deliberately limited to one already-reviewed static input set; batching, relax,
AIMD, continuation, and scheduler submission remain separate contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence


FORBIDDEN_EXECUTABLES = frozenset({"ssh", "scp", "sftp", "sbatch", "scancel"})
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+\-]*")


class ContractError(ValueError):
    """Raised when staged inputs or VASP outputs violate the approved contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _ordinary(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _read_json(path: Path) -> dict[str, Any]:
    if not _ordinary(path):
        raise ContractError(f"required staged file is missing or unsafe: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path.name}")
    return value


def _validate_argv(value: object) -> list[str]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item and "\x00" not in item for item in value
    ):
        raise ContractError("vasp argv must be a non-empty string list")
    executable = Path(value[0]).name
    if executable in FORBIDDEN_EXECUTABLES:
        raise ContractError(f"nested transport/scheduler executable is forbidden: {executable}")
    return list(value)


def _float_rows(element: ET.Element | None) -> list[list[float]]:
    if element is None:
        raise ContractError("required VASP XML array is missing")
    rows: list[list[float]] = []
    for vector in element.findall("v"):
        values = [float(token) for token in (vector.text or "").split()]
        if not values or any(not math.isfinite(value) for value in values):
            raise ContractError("VASP XML contains a non-finite or empty vector")
        rows.append(values)
    if not rows:
        raise ContractError("required VASP XML array is empty")
    return rows


def _named_float(parent: ET.Element, name: str) -> float:
    candidate = parent.find(f"./energy/i[@name='{name}']")
    if candidate is None or candidate.text is None:
        raise ContractError(f"VASP XML lacks energy {name}")
    value = float(candidate.text)
    if not math.isfinite(value):
        raise ContractError(f"VASP XML energy {name} is non-finite")
    return value


def _parse_vasprun(path: Path, expected_atoms: int) -> dict[str, Any]:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ContractError(f"vasprun.xml is truncated or invalid: {exc}") from exc
    calculations = root.findall("./calculation")
    if not calculations:
        raise ContractError("vasprun.xml contains no calculation")
    final = calculations[-1]
    parameters = root.findall(".//parameters//i[@name='NELM']")
    if not parameters or parameters[-1].text is None:
        raise ContractError("vasprun.xml does not record NELM")
    nelm = int(float(parameters[-1].text))
    scstep_count = len(final.findall("./scstep"))
    structures = root.findall("./structure")
    final_structure = next(
        (item for item in reversed(structures) if item.get("name") == "finalpos"),
        structures[-1] if structures else None,
    )
    if final_structure is None:
        raise ContractError("vasprun.xml lacks a final structure")
    basis = _float_rows(final_structure.find("./crystal/varray[@name='basis']"))
    positions = _float_rows(final_structure.find("./varray[@name='positions']"))
    forces = _float_rows(final.find("./varray[@name='forces']"))
    stress = _float_rows(final.find("./varray[@name='stress']"))
    atoms = root.findall("./atominfo/array[@name='atoms']/set/rc")
    species = [
        str(row.findtext("c", default="")).strip()
        for row in atoms
    ]
    if (
        len(species) != expected_atoms
        or len(positions) != expected_atoms
        or len(forces) != expected_atoms
        or any(not symbol for symbol in species)
    ):
        raise ContractError("VASP output atom count differs from the prepared manifest")
    if len(basis) != 3 or any(len(row) != 3 for row in basis):
        raise ContractError("final lattice is not 3x3")
    if len(stress) != 3 or any(len(row) != 3 for row in stress):
        raise ContractError("stress is not a 3x3 tensor")
    return {
        "energy_ev": _named_float(final, "e_0_energy"),
        "species": species,
        "lattice_angstrom": basis,
        "fractional_coordinates": positions,
        "forces_ev_per_angstrom": forces,
        "stress_kbar_vasp_3x3": stress,
        "electronic_converged": scstep_count > 0 and scstep_count < nelm,
        "electronic_steps": scstep_count,
        "nelm": nelm,
    }


def _write_json_fresh(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ContractError(f"output already exists: {path.name}")
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def execute(arguments: argparse.Namespace) -> None:
    attempt = Path(arguments.attempt_dir).resolve()
    if not attempt.is_dir() or attempt.is_symlink():
        raise ContractError("attempt directory must be an ordinary directory")
    names = {
        "structures_manifest": arguments.structures_manifest,
        "labeling_config": arguments.labeling_config,
        "dft_input_manifest": arguments.dft_input_manifest,
        "POSCAR": "POSCAR",
        "INCAR": "INCAR",
        "KPOINTS": "KPOINTS",
        "POTCAR": "POTCAR",
    }
    if any(not SAFE_NAME.fullmatch(value) for value in names.values()):
        raise ContractError("all staged inputs must use safe basenames")
    paths = {name: attempt / value for name, value in names.items()}
    if any(not _ordinary(path) for path in paths.values()):
        raise ContractError("one or more staged VASP inputs are missing or unsafe")
    structures = _read_json(paths["structures_manifest"])
    labeling = _read_json(paths["labeling_config"])
    prepared = _read_json(paths["dft_input_manifest"])
    calculations = prepared.get("calculations")
    if (
        prepared.get("status") != "OK"
        or prepared.get("calculation_type") != "static"
        or not isinstance(calculations, list)
        or len(calculations) != 1
        or not isinstance(calculations[0], dict)
    ):
        raise ContractError("bundled scheduled wrapper requires one prepared static calculation")
    calculation = calculations[0]
    declared_files = calculation.get("files")
    if not isinstance(declared_files, dict):
        raise ContractError("prepared manifest lacks VASP file records")
    for name in ("POSCAR", "INCAR", "KPOINTS", "POTCAR"):
        record = declared_files.get(name)
        if not isinstance(record, dict) or record.get("sha256") != _sha256(paths[name]):
            raise ContractError(f"staged {name} differs from the prepared manifest")
    raw_structures = structures.get("structures")
    if not isinstance(raw_structures, list) or len(raw_structures) != 1:
        raise ContractError("structures manifest must contain exactly one source structure")
    if labeling.get("calculation_type") != "static":
        raise ContractError("labeling config must describe a static calculation")
    vasp_argv = _validate_argv(json.loads(arguments.vasp_argv_json))
    stdout_path = attempt / "vasp.stdout"
    stderr_path = attempt / "vasp.stderr"
    if stdout_path.exists() or stderr_path.exists():
        raise ContractError("VASP log outputs must be fresh")
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            vasp_argv,
            cwd=str(attempt),
            stdout=stdout,
            stderr=stderr,
            shell=False,
            check=False,
        )
    if completed.returncode != 0:
        raise ContractError(f"VASP command exited {completed.returncode}")
    raw_paths = {name: attempt / name for name in ("OUTCAR", "OSZICAR", "vasprun.xml")}
    if any(not _ordinary(path) or path.stat().st_size == 0 for path in raw_paths.values()):
        raise ContractError("VASP did not produce all required ordinary outputs")
    outcar_text = raw_paths["OUTCAR"].read_text(encoding="utf-8", errors="replace")
    footer = "General timing and accounting informations for this job:"
    if footer not in outcar_text:
        raise ContractError("OUTCAR lacks the normal-completion footer")
    parsed = _parse_vasprun(raw_paths["vasprun.xml"], int(calculation.get("atom_count", 0)))
    if not parsed["electronic_converged"]:
        raise ContractError("electronic convergence was not reached before NELM")
    version_match = re.search(r"\bvasp\.([0-9][A-Za-z0-9._-]*)", outcar_text, re.I)
    labels_path = attempt / arguments.labels
    units = json.loads(arguments.units_json)
    if not isinstance(units, dict):
        raise ContractError("units must decode to an object")
    label_document = {
        "schema_version": 1,
        "records": [
            {
                "structure_id": calculation.get("structure_id"),
                **{key: value for key, value in parsed.items() if key not in {"electronic_converged", "electronic_steps", "nelm"}},
            }
        ],
        "units": units,
    }
    _write_json_fresh(labels_path, label_document)
    result_path = attempt / arguments.result_manifest
    raw_records = {
        name: {
            "path": name,
            "fingerprint": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in raw_paths.items()
    }
    result = {
        "schema_version": 1,
        "plugin_id": "dft-labeling",
        "status": "OK",
        "engine": "vasp",
        "input_fingerprints": {
            name: _sha256(paths[name])
            for name in ("structures_manifest", "labeling_config", "dft_input_manifest")
        },
        "completion": {
            "scheduler_success": True,
            "electronic_converged": True,
            "ionic_convergence_required": False,
            "ionic_converged": None,
            "truncated": False,
        },
        "units": units,
        "source_structure_count": 1,
        "label_count": 1,
        "execution": {
            "vasp_argv": vasp_argv,
            "vasp_version": version_match.group(1) if version_match else "UNKNOWN",
            "wrapper_sha256": _sha256(Path(__file__)),
            "electronic_steps": parsed["electronic_steps"],
            "nelm": parsed["nelm"],
        },
        "raw_outputs": raw_records,
        "artifacts": [
            {
                "name": "labels-json",
                "path": arguments.labels,
                "media_type": "application/json",
                "fingerprint": _sha256(labels_path),
            }
        ],
    }
    _write_json_fresh(result_path, result)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--attempt-dir", required=True)
    value.add_argument("--structures-manifest", required=True)
    value.add_argument("--labeling-config", required=True)
    value.add_argument("--dft-input-manifest", required=True)
    value.add_argument("--result-manifest", required=True)
    value.add_argument("--labels", required=True)
    value.add_argument("--units-json", required=True)
    value.add_argument("--vasp-argv-json", required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = parser().parse_args(argv)
        for name in (
            arguments.structures_manifest,
            arguments.labeling_config,
            arguments.dft_input_manifest,
            arguments.result_manifest,
            arguments.labels,
        ):
            if not SAFE_NAME.fullmatch(name):
                raise ContractError(f"unsafe staged basename: {name!r}")
        execute(arguments)
    except (ContractError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"dft-labeling scheduled wrapper failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
