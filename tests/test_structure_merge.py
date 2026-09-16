"""DIRECT+LASP structure-set merge and DFT handoff contract."""

from __future__ import annotations

import csv
from tests.helpers import load_module as import_test_module
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest
from ase import Atoms
from ase.io import write
from ase.io.dmol import write_dmol_arc
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Poscar


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "mlipipe" / "plugins" / "pes_sampling"
ADAPTER_PATH = PLUGIN_ROOT / "adapter.py"
DFT_PREPARE_PATH = ROOT / "mlipipe" / "plugins" / "dft_labeling" / "vasp_prepare.py"


def load_module(name: str, path: Path) -> Any:
    module = import_test_module(path, name)
    return module


def arc_payload(symbols: list[str], scaled: list[list[float]]) -> bytes:
    atoms = Atoms(symbols=symbols, scaled_positions=scaled, cell=[12.0, 12.0, 12.0], pbc=True)
    stream = io.StringIO()
    write_dmol_arc(stream, [atoms])
    return stream.getvalue().encode("utf-8")


def write_fixture(root: Path) -> tuple[Path, Path, Path]:
    direct_dir = root / "direct"
    direct_dir.mkdir()
    direct_structure = Structure(
        Lattice.cubic(12.0), ["Li", "S"], [[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]]
    )
    direct_poscar = direct_dir / "00001_LiS.vasp"
    Poscar(direct_structure, sort_structure=True).write_file(direct_poscar)
    direct_manifest = direct_dir / "manifest.csv"
    with direct_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "selected_order",
                "input_index",
                "source_kind",
                "source_file",
                "frame_index",
                "output_file",
                "formula",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "selected_order": 1,
                "input_index": 0,
                "source_kind": "ase",
                "source_file": "trajectory.traj",
                "frame_index": 10,
                "output_file": direct_poscar.name,
                "formula": "LiS",
            }
        )

    payloads = {
        "selected/input-000001.arc": arc_payload(
            ["Li", "S"], [[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]]
        ),
        "selected/input-000002.arc": arc_payload(
            ["Li", "S"], [[0.1001, 0.1, 0.1], [0.5, 0.5, 0.5]]
        ),
        "selected/input-000003.arc": arc_payload(
            ["Li", "P"], [[0.2, 0.2, 0.2], [0.7, 0.7, 0.7]]
        ),
        "selected/input-000004.arc": arc_payload(
            ["Li", "S"], [[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]]
        ),
    }
    lasp_manifest = root / "selected-structures.json"
    lasp_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_id": "fixture://lasp",
                "structures": [
                    {
                        "structure_id": f"lasp-ssw-{index:02d}",
                        "selected_order": index,
                        "output_file": name,
                    }
                    for index, name in enumerate(payloads, 1)
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    archive = root / "selected-structures.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
    return direct_manifest, lasp_manifest, archive


def context(root: Path) -> dict[str, Any]:
    direct, lasp_manifest, archive = write_fixture(root)
    return {
        "project_root": str(root),
        "attempt_dir": str(root / ".mlipipe" / "runs" / "merge" / "attempt-1"),
        "inputs": {
            "direct_manifest": str(direct),
            "lasp_selected_manifest": str(lasp_manifest),
            "lasp_selected_archive": str(archive),
        },
        "parameters": {
            "operation": "merge-structures",
            "output_subdir": "merged",
            "direct_source_group_id": "exploratory-md-1",
            "lasp_source_group_id": "lasp-ssw-1",
            "matcher_ltol": 0.01,
            "matcher_stol": 0.05,
            "matcher_angle_tol_deg": 1.0,
            "minimum_distance_angstrom": 0.5,
            "max_structures": 10,
        },
        "backend": "local",
        "resources": {"python_executable": str(Path(sys.executable).resolve())},
    }


def test_merge_runner_adapter_and_dft_ready_manifest(tmp_path: Path) -> None:
    adapter = load_module("test_structure_merge_adapter", ADAPTER_PATH).Adapter()
    value = context(tmp_path)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    diagnostics = adapter.validate(value)
    plan = adapter.plan(value)
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert before == after
    assert not any(item["level"] == "ERROR" for item in diagnostics), diagnostics
    assert plan["status"] == "READY"
    assert plan["operation"] == "merge-structures"
    assert plan["shell"] is False
    assert plan["approval_summary"]["expensive"] is False

    attempt = Path(value["attempt_dir"])
    attempt.mkdir(parents=True)
    completed = subprocess.run(plan["argv"], cwd=plan["cwd"], check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    value["execution"] = {"returncode": completed.returncode, "plan": plan}
    checked = adapter.check(value)
    collected = adapter.collect(value)
    assert checked["status"] == "OK", checked
    assert collected["status"] == "OK", collected

    manifest_path = Path(checked["result_file"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"] == {
        "input": 5,
        "unique": 2,
        "exact_duplicates": 1,
        "near_duplicates": 1,
        "rejected_bad": 1,
    }
    assert manifest["source_statistics"]["DIRECT"]["input"] == 1
    assert manifest["source_statistics"]["LASP_SSW"]["input"] == 4
    assert any(record["source_sampling_methods"] == ["DIRECT", "LASP_SSW"] for record in manifest["structures"])
    assert len([item for item in collected["artifacts"] if item["role"] == "merged-structure"]) == 2

    dft_prepare = load_module("test_structure_merge_dft_prepare", DFT_PREPARE_PATH)
    records = dft_prepare._structure_records(manifest, manifest_path, tmp_path, 10)
    assert [record["id"] for record in records] == [item["id"] for item in manifest["structures"]]


def test_merge_checker_accepts_nonscientific_text_change(
    tmp_path: Path,
) -> None:
    adapter = load_module("test_structure_merge_adapter_changed", ADAPTER_PATH).Adapter()
    value = context(tmp_path)
    plan = adapter.plan(value)
    attempt = Path(value["attempt_dir"])
    attempt.mkdir(parents=True)
    completed = subprocess.run(plan["argv"], cwd=plan["cwd"], check=False)
    assert completed.returncode == 0
    value["execution"] = {"returncode": 0, "plan": plan}
    manifest = json.loads(Path(plan["expected_outputs"][0]).read_text(encoding="utf-8"))
    structure = Path(plan["expected_outputs"][0]).parent / manifest["structures"][0]["path"]
    structure.write_text(structure.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    checked = adapter.check(value)
    assert checked["status"] == "OK"


def test_lasp_input_prepare_converts_selected_ase_frame(tmp_path: Path) -> None:
    source = tmp_path / "trajectory.extxyz"
    frames = [
        Atoms(
            symbols=["Li", "S"],
            scaled_positions=[[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]],
            cell=[12.0, 13.0, 14.0],
            pbc=True,
        ),
        Atoms(
            symbols=["Li", "P"],
            scaled_positions=[[0.2, 0.2, 0.2], [0.7, 0.7, 0.7]],
            cell=[12.5, 13.5, 14.5],
            pbc=True,
        ),
    ]
    write(source, frames, format="extxyz")
    attempt = tmp_path / ".mlipipe" / "runs" / "lasp-input" / "attempt-1"
    value = {
        "project_root": str(tmp_path),
        "attempt_dir": str(attempt),
        "inputs": {"input_structure": str(source)},
        "parameters": {
            "operation": "lasp-input-prepare",
            "output_subdir": "prepared",
            "input_format": "extxyz",
            "input_index": "-1",
            "minimum_cell_length_angstrom": 10.0,
        },
        "backend": "local",
        "resources": {"python_executable": str(Path(sys.executable).resolve())},
    }
    adapter = load_module("test_structure_to_lasp_adapter", ADAPTER_PATH).Adapter()
    diagnostics = adapter.validate(value)
    plan = adapter.plan(value)
    assert not any(item["level"] == "ERROR" for item in diagnostics), diagnostics
    assert plan["status"] == "READY"
    assert plan["operation"] == "lasp-input-prepare"
    assert plan["shell"] is False
    assert plan["approval_summary"]["expensive"] is False

    attempt.mkdir(parents=True)
    completed = subprocess.run(
        plan["argv"], cwd=plan["cwd"], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    value["execution"] = {"returncode": completed.returncode, "plan": plan}
    checked = adapter.check(value)
    collected = adapter.collect(value)
    assert checked["status"] == "OK", checked
    assert collected["status"] == "OK", collected
    assert collected["metrics"] == {"atom_count": 2}

    manifest = json.loads(Path(checked["result_file"]).read_text(encoding="utf-8"))
    assert manifest["source"]["input_index"] == "-1"
    assert manifest["structure"]["composition"] == {"Li": 1, "P": 1}
    assert manifest["structure"]["cell_lengths_A"] == [12.5, 13.5, 14.5]
    input_arc = Path(checked["result_file"]).parent / "input.arc"
    assert input_arc.read_bytes().startswith(b"!BIOSYM archive 2\nPBC=ON\nEnergy 1 ")
    assert manifest["arc_contract"]["energy_semantics"] == "input-format-placeholder-not-a-label"


@pytest.mark.parametrize("path_kind", ["project-relative", "parent-relative", "absolute"])
def test_lasp_input_structure_path_runs_full_conversion_chain(
    tmp_path: Path, path_kind: str
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    if path_kind == "project-relative":
        source = project / "inputs" / "A.extxyz"
        value = "inputs/A.extxyz"
    else:
        source = tmp_path / ("shared" if path_kind == "parent-relative" else "external") / "A.extxyz"
        value = "../shared/A.extxyz" if path_kind == "parent-relative" else str(source)
    source.parent.mkdir()
    write(
        source,
        Atoms(
            symbols=["Li", "P"],
            scaled_positions=[[0.2, 0.2, 0.2], [0.7, 0.7, 0.7]],
            cell=[12.5, 13.5, 14.5],
            pbc=True,
        ),
        format="extxyz",
    )
    attempt = project / ".mlipipe" / "runs" / "lasp-input" / "attempt-1"
    context = {
        "project_root": str(project),
        "attempt_dir": str(attempt),
        "inputs": {"input_structure": value},
        "parameters": {
            "operation": "lasp-input-prepare",
            "output_subdir": "prepared",
            "input_format": "extxyz",
            "input_index": "-1",
        },
        "backend": "local",
        "resources": {"python_executable": str(Path(sys.executable).resolve())},
    }
    adapter = load_module(f"test_structure_to_lasp_path_{path_kind}", ADAPTER_PATH).Adapter()

    plan = adapter.plan(context)

    assert plan["status"] == "READY", plan.get("diagnostics")
    assert plan["input_paths"]["source_structure"] == str(source.resolve())
    attempt.mkdir(parents=True)
    completed = subprocess.run(
        plan["argv"], cwd=plan["cwd"], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    context["execution"] = {"returncode": completed.returncode, "plan": plan}
    checked = adapter.check(context)
    collected = adapter.collect(context)
    assert checked["status"] == "OK", checked
    assert collected["status"] == "OK", collected
    manifest = json.loads(Path(checked["result_file"]).read_text(encoding="utf-8"))
    assert manifest["source"]["path"] == str(source.resolve())
    assert (Path(checked["result_file"]).parent / "input.arc").is_file()


@pytest.mark.parametrize("path_kind", ["parent-relative", "absolute"])
def test_lasp_input_missing_structure_path_is_blocked(
    tmp_path: Path, path_kind: str
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    missing = tmp_path / "shared" / "missing.extxyz"
    value = "../shared/missing.extxyz" if path_kind == "parent-relative" else str(missing)
    context = {
        "project_root": str(project),
        "attempt_dir": str(project / ".mlipipe" / "runs" / "lasp-input" / "attempt-1"),
        "inputs": {"input_structure": value},
        "parameters": {"operation": "lasp-input-prepare"},
        "backend": "local",
        "resources": {"python_executable": str(Path(sys.executable).resolve())},
    }
    adapter = load_module(f"test_structure_to_lasp_missing_{path_kind}", ADAPTER_PATH).Adapter()

    plan = adapter.plan(context)

    assert plan["status"] == "BLOCKED"
    messages = [item["message"] for item in plan["diagnostics"]]
    assert any("input_structure file does not exist" in message for message in messages)
    assert str(missing.resolve()) in "\n".join(messages)


def test_lasp_input_prepare_materializes_noncollectable_potcar(
    tmp_path: Path, monkeypatch: Any
) -> None:
    source = tmp_path / "structure.extxyz"
    write(
        source,
        Atoms(
            symbols=["Li", "P"],
            scaled_positions=[[0.2, 0.2, 0.2], [0.7, 0.7, 0.7]],
            cell=[12.5, 13.5, 14.5],
            pbc=True,
        ),
        format="extxyz",
    )

    class FakePotcarSingle:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def __str__(self) -> str:
            return f"FAKE-{self.symbol}\n"

    class FakePotcar(list[FakePotcarSingle]):
        def __init__(self, symbols: list[str], functional: str) -> None:
            assert functional == "PBE_54"
            super().__init__([FakePotcarSingle(symbol) for symbol in symbols])

        def write_file(self, path: str) -> None:
            Path(path).write_text(
                "".join(str(component) for component in self), encoding="utf-8"
            )

    reference = tmp_path / "pseudopotentials.json"
    reference.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reference_id": "fixture-pbe54-plain-v1",
                "source_env": "PMG_VASP_PSP_DIR",
                "license_acknowledged": True,
                "functional": "PBE_54",
                "symbols": {"Li": "Li", "P": "P"},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    psp_root = tmp_path / "licensed-psp"
    psp_root.mkdir()
    monkeypatch.setenv("PMG_VASP_PSP_DIR", str(psp_root))
    attempt = tmp_path / ".mlipipe" / "runs" / "lasp-input-potcar" / "attempt-1"
    value = {
        "project_root": str(tmp_path),
        "attempt_dir": str(attempt),
        "inputs": {
            "input_structure": str(source),
            "pseudopotential_reference": str(reference),
        },
        "parameters": {
            "operation": "lasp-input-prepare",
            "output_subdir": "prepared",
            "input_format": "extxyz",
            "input_index": "-1",
            "minimum_cell_length_angstrom": 10.0,
        },
        "backend": "local",
        "resources": {"python_executable": str(Path(sys.executable).resolve())},
    }
    adapter = load_module("test_structure_to_lasp_potcar_adapter", ADAPTER_PATH).Adapter()
    plan = adapter.plan(value)
    assert plan["status"] == "READY", plan
    assert plan["approval_summary"]["materializes_licensed_potcar"] is True
    assert plan["approval_summary"]["potcar_collectable"] is False

    wrapper = load_module(
        "test_structure_to_lasp_potcar_wrapper", PLUGIN_ROOT / "structure_to_lasp.py"
    )
    attempt.mkdir(parents=True)
    args = wrapper.build_parser().parse_args(plan["argv"][2:])
    wrapper.run(
        args,
        api=wrapper.PymatgenApi(
            Potcar=FakePotcar, configured_psp_root=str(psp_root)
        ),
    )
    value["execution"] = {"returncode": 0, "plan": plan}
    checked = adapter.check(value)
    collected = adapter.collect(value)
    assert checked["status"] == "OK", checked
    assert collected["status"] == "OK", collected
    assert {item["role"] for item in collected["artifacts"]} == {
        "lasp-input-manifest",
        "lasp-input-structure",
    }
    assert all(Path(item["path"]).name != "POTCAR" for item in collected["artifacts"])
    manifest = json.loads(Path(checked["result_file"]).read_text(encoding="utf-8"))
    assert manifest["potcar"]["output"]["collectable"] is False
    assert manifest["potcar"]["symbols"] == ["Li", "P"]
