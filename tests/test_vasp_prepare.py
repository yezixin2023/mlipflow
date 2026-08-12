"""Contract, provenance, and integrity tests for ``dft-labeling.vasp-prepare``."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "plugins" / "dft-labeling"
PYTHON_EXECUTABLE = str(Path(sys.executable).resolve())


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fingerprint_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def fingerprint(path: Path) -> str:
    return fingerprint_bytes(path.read_bytes())


class FakeStructure:
    composition = SimpleNamespace(reduced_formula="LiPS")

    @classmethod
    def from_file(cls, path: str) -> "FakeStructure":
        if not Path(path).is_file():
            raise ValueError("missing structure")
        return cls()

    def __len__(self) -> int:
        return 3


class FakePoscar:
    site_symbols = ["Li", "P", "S"]

    def __init__(self, structure: FakeStructure, sort_structure: bool) -> None:
        self.structure = structure
        self.sort_structure = sort_structure

    def write_file(self, path: str) -> None:
        Path(path).write_text(
            "LiPS\n1.0\n1 0 0\n0 1 0\n0 0 1\nLi P S\n1 1 1\nDirect\n0 0 0\n0 0 0\n0 0 0\n",
            encoding="utf-8",
        )


def incar_value(value: Any) -> str:
    if value is True:
        return ".TRUE."
    if value is False:
        return ".FALSE."
    if isinstance(value, list):
        return " ".join(incar_value(item) for item in value)
    return str(value)


class FakeIncar:
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values

    def write_file(self, path: str) -> None:
        Path(path).write_text(
            "".join(f"{key} = {incar_value(value)}\n" for key, value in sorted(self.values.items())),
            encoding="utf-8",
        )


class FakeKpoints:
    def __init__(self, mode: str, kpts: tuple[int, int, int], shift: tuple[float, float, float]):
        self.mode = mode
        self.kpts = kpts
        self.shift = shift

    @classmethod
    def gamma_automatic(cls, kpts: tuple[int, int, int], shift: tuple[float, float, float]):
        return cls("Gamma", kpts, shift)

    @classmethod
    def monkhorst_automatic(
        cls, kpts: tuple[int, int, int], shift: tuple[float, float, float]
    ):
        return cls("Monkhorst-Pack", kpts, shift)

    def write_file(self, path: str) -> None:
        grid = " ".join(str(item) for item in self.kpts)
        shift = " ".join(str(item) for item in self.shift)
        Path(path).write_text(
            f"Automatic mesh\n0\n{self.mode}\n{grid}\n{shift}\n", encoding="utf-8"
        )


class FakePotcarSingle:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    def __str__(self) -> str:
        return f"FAKE-{self.symbol}\n"


class FakePotcar(list[FakePotcarSingle]):
    def __init__(self, symbols: list[str], functional: str) -> None:
        if functional != "PBE_54":
            raise ValueError("unexpected functional")
        super().__init__([FakePotcarSingle(symbol) for symbol in symbols])

    def write_file(self, path: str) -> None:
        Path(path).write_text("".join(str(item) for item in self), encoding="utf-8")


class VaspPrepareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter_module = load_module("test_dft_adapter", PLUGIN_ROOT / "adapter.py")
        self.wrapper = load_module("test_vasp_prepare", PLUGIN_ROOT / "vasp_prepare.py")

    def fixture(self, root: Path, calculation_type: str = "static") -> tuple[dict[str, Any], Any]:
        project = root.resolve()
        attempt = project / "attempt"
        source_dir = project / "input"
        attempt.mkdir(parents=True)
        source_dir.mkdir()
        structure = source_dir / "s1.cif"
        structure.write_text("data_s1\n_cell_length_a 1\n", encoding="utf-8")
        structures = source_dir / "structures.json"
        write_json(
            structures,
            {
                "schema_version": 1,
                "structures": [
                    {"id": "s1", "path": "s1.cif", "fingerprint": fingerprint(structure)}
                ],
            },
        )
        if calculation_type == "static":
            config_value = {
                "schema_version": 1,
                "engine": "vasp",
                "calculation_type": "static",
                "preset": "manuscript-static-v1",
                "incar": {},
            }
        elif calculation_type == "relax":
            config_value = {
                "schema_version": 1,
                "engine": "vasp",
                "calculation_type": "relax",
                "incar": {"ENCUT": 450, "EDIFF": 5e-6, "NSW": 100, "IBRION": 2, "ISIF": 3, "EDIFFG": -0.02},
                "kpoints": {"mode": "gamma", "grid": [2, 2, 2], "shift": [0, 0, 0]},
            }
        else:
            config_value = {
                "schema_version": 1,
                "engine": "vasp",
                "calculation_type": "aimd",
                "incar": {"ENCUT": 450, "EDIFF": 5e-6, "NSW": 500, "IBRION": 0, "POTIM": 2.0, "TEBEG": 500, "TEEND": 800, "MDALGO": 2},
                "kpoints": {"mode": "gamma", "grid": [1, 1, 1], "shift": [0, 0, 0]},
            }
        config = source_dir / "labeling.json"
        write_json(config, config_value)
        combined = b"FAKE-Li_sv\nFAKE-P\nFAKE-S\n"
        reference = source_dir / "pseudopotentials.json"
        write_json(
            reference,
            {
                "schema_version": 1,
                "reference_id": "user-licensed-pbe54-v1",
                "source_env": "PMG_VASP_PSP_DIR",
                "license_acknowledged": True,
                "functional": "PBE_54",
                "symbols": {"Li": "Li_sv", "P": "P", "S": "S"},
                "expected_component_sha256": {
                    symbol: fingerprint_bytes(f"FAKE-{symbol}\n".encode())
                    for symbol in ("Li_sv", "P", "S")
                },
                "expected_combined_sha256": fingerprint_bytes(combined),
            },
        )
        psp_dir = project / "licensed-psp"
        psp_dir.mkdir()
        context = {
            "project_root": str(project),
            "attempt_dir": str(attempt),
            "inputs": {
                "structures_manifest": "input/structures.json",
                "labeling_config": "input/labeling.json",
                "pseudopotential_reference": "input/pseudopotentials.json",
            },
            "parameters": {
                "operation": "vasp-prepare",
                "interpreter_argv": [PYTHON_EXECUTABLE],
                "engine": "vasp",
                "result_manifest": "dft-input-manifest.json",
                "output_subdir": "vasp-inputs",
                "max_structures": 10,
            },
            "backend": "local",
            "resources": {"cpus": 1},
        }
        args = argparse.Namespace(
            project_root=str(project),
            attempt_dir=str(attempt),
            structures_manifest=str(structures),
            labeling_config=str(config),
            pseudopotential_reference=str(reference),
            result_manifest=str(attempt / "dft-input-manifest.json"),
            output_subdir="vasp-inputs",
            max_structures=10,
        )
        return context, SimpleNamespace(args=args, psp_dir=psp_dir)

    def fake_api(self):
        return self.wrapper.PymatgenApi(
            FakeStructure,
            FakeIncar,
            FakeKpoints,
            FakePoscar,
            FakePotcar,
            "2025.10.7-test",
            None,
        )

    def test_static_preset_plan_generate_check_and_collect_excludes_potcar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context, fixture = self.fixture(Path(directory))
            adapter = self.adapter_module.Adapter()
            before = sorted(Path(directory).rglob("*"))
            self.assertEqual([], adapter.validate(context))
            plan = adapter.plan(context)
            self.assertEqual("READY", plan["status"])
            self.assertFalse(plan["approval_summary"]["runs_vasp"])
            self.assertFalse(plan["approval_summary"]["submits_jobs"])
            self.assertIn("prepare_wrapper", plan["input_fingerprints"])
            self.assertEqual(450, plan["approval_summary"]["effective_incar"]["ENCUT"])
            self.assertEqual([1, 1, 1], plan["approval_summary"]["kpoints"]["grid"])
            self.assertEqual(
                {"Li": "Li_sv", "P": "P", "S": "S"},
                plan["approval_summary"]["pseudopotential_symbols"],
            )
            self.assertEqual(before, sorted(Path(directory).rglob("*")), "validate/plan must not write")

            with patch.dict(os.environ, {"PMG_VASP_PSP_DIR": str(fixture.psp_dir)}):
                manifest = self.wrapper.prepare_inputs(fixture.args, self.fake_api())
            self.assertEqual(450, manifest["incar"]["effective"]["ENCUT"])
            self.assertEqual(5e-6, manifest["incar"]["effective"]["EDIFF"])
            self.assertEqual(38, manifest["incar"]["effective"]["IALGO"])
            self.assertEqual([1, 1, 1], manifest["kpoints"]["grid"])
            self.assertRegex(manifest["runtime"]["prepare_wrapper_sha256"], r"^sha256:[0-9a-f]{64}$")
            self.assertFalse(manifest["calculations"][0]["files"]["POTCAR"]["collectable"])
            self.assertEqual("OK", adapter.check(context)["status"])
            collected = adapter.collect(context)
            self.assertEqual("OK", collected["status"])
            self.assertFalse(collected["metrics"]["potcar_collected"])
            self.assertNotIn("POTCAR", [Path(item["path"]).name for item in collected["artifacts"]])

    def test_incar_and_potcar_tampering_fail_even_when_manifest_hashes_are_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context, fixture = self.fixture(Path(directory))
            adapter = self.adapter_module.Adapter()
            with patch.dict(os.environ, {"PMG_VASP_PSP_DIR": str(fixture.psp_dir)}):
                self.wrapper.prepare_inputs(fixture.args, self.fake_api())
            result_path = Path(context["attempt_dir"]) / "dft-input-manifest.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            incar = Path(context["attempt_dir"]) / result["calculations"][0]["files"]["INCAR"]["path"]
            incar.write_text("ENCUT = 100\nNSW = 0\nIBRION = -1\n", encoding="utf-8")
            result["calculations"][0]["files"]["INCAR"]["sha256"] = fingerprint(incar)
            result["calculations"][0]["files"]["INCAR"]["size_bytes"] = incar.stat().st_size
            write_json(result_path, result)
            self.assertEqual("FAIL", adapter.check(context)["status"])

            with tempfile.TemporaryDirectory() as second_directory:
                second_context, second_fixture = self.fixture(Path(second_directory))
                with patch.dict(os.environ, {"PMG_VASP_PSP_DIR": str(second_fixture.psp_dir)}):
                    self.wrapper.prepare_inputs(second_fixture.args, self.fake_api())
                second_result_path = Path(second_context["attempt_dir"]) / "dft-input-manifest.json"
                second_result = json.loads(second_result_path.read_text(encoding="utf-8"))
                potcar = Path(second_context["attempt_dir"]) / second_result["calculations"][0]["files"]["POTCAR"]["path"]
                potcar.write_text("forged\n", encoding="utf-8")
                new_digest = fingerprint(potcar)
                second_result["calculations"][0]["files"]["POTCAR"]["sha256"] = new_digest
                second_result["calculations"][0]["files"]["POTCAR"]["size_bytes"] = potcar.stat().st_size
                second_result["calculations"][0]["potcar"]["combined_sha256"] = new_digest
                write_json(second_result_path, second_result)
                self.assertEqual("FAIL", self.adapter_module.Adapter().check(second_context)["status"])

    def test_static_relax_and_aimd_contracts_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            static_context, _ = self.fixture(Path(directory) / "static", "static")
            self.assertEqual([], self.adapter_module.Adapter().validate(static_context))
        with tempfile.TemporaryDirectory() as directory:
            relax_context, _ = self.fixture(Path(directory) / "relax", "relax")
            self.assertEqual([], self.adapter_module.Adapter().validate(relax_context))
        with tempfile.TemporaryDirectory() as directory:
            aimd_context, _ = self.fixture(Path(directory) / "aimd", "aimd")
            self.assertEqual([], self.adapter_module.Adapter().validate(aimd_context))
            config = Path(aimd_context["project_root"]) / "input/labeling.json"
            value = json.loads(config.read_text(encoding="utf-8"))
            del value["incar"]["TEBEG"]
            write_json(config, value)
            codes = {item["code"] for item in self.adapter_module.Adapter().validate(aimd_context)}
            self.assertIn("inputs.labeling_config.contract", codes)

    def test_prepare_blocks_bad_runtime_paths_and_waits_for_missing_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context, _ = self.fixture(Path(directory))
            adapter = self.adapter_module.Adapter()
            self.assertEqual("WAIT", adapter.check(context)["status"])
            for executable in ("python", str(Path(directory) / "missing-python")):
                changed = json.loads(json.dumps(context))
                changed["parameters"]["interpreter_argv"] = [executable]
                self.assertEqual("BLOCKED", adapter.plan(changed)["status"])

    def test_wrapper_requires_approved_pseudopotential_environment_and_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context, fixture = self.fixture(Path(directory))
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(self.wrapper.ContractError, "PMG_VASP_PSP_DIR"):
                    self.wrapper.prepare_inputs(fixture.args, self.fake_api())
            reference = Path(context["project_root"]) / "input/pseudopotentials.json"
            value = json.loads(reference.read_text(encoding="utf-8"))
            value["expected_combined_sha256"] = "sha256:" + "0" * 64
            write_json(reference, value)
            # A failed runtime leaves explicit INCOMPLETE evidence in its fresh attempt.
            retry_attempt = Path(context["project_root"]) / "retry-attempt"
            retry_attempt.mkdir()
            retry_args = argparse.Namespace(**{**vars(fixture.args), "attempt_dir": str(retry_attempt), "result_manifest": str(retry_attempt / "dft-input-manifest.json")})
            with patch.dict(os.environ, {"PMG_VASP_PSP_DIR": str(fixture.psp_dir)}):
                with self.assertRaisesRegex(self.wrapper.ContractError, "combined POTCAR hash"):
                    self.wrapper.prepare_inputs(retry_args, self.fake_api())
            self.assertTrue((retry_attempt / "vasp-inputs/INCOMPLETE.json").is_file())

    def test_label_can_bind_prepared_manifest_but_remains_a_separate_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory).resolve()
            attempt = project / "label-attempt"
            attempt.mkdir()
            context = {
                "project_root": str(project),
                "attempt_dir": str(attempt),
                "inputs": {
                    "structures_manifest": "structures.json",
                    "labeling_config": "labeling.json",
                    "dft_input_manifest": "dft-input-manifest.json",
                },
                "parameters": {
                    "operation": "label",
                    "label_script": "scripts/run_dft.py",
                    "interpreter_argv": ["python"],
                    "engine": "vasp",
                    "completion_policy": {"require_ionic_convergence": False},
                    "units": {"energy": "eV", "length": "angstrom", "force": "eV/angstrom", "stress": "GPa"},
                },
                "backend": "local",
                "resources": {"cpus": 16},
            }
            plan = self.adapter_module.Adapter().plan(context)
            self.assertEqual("label", plan["operation"])
            self.assertTrue(plan["approval_summary"]["runs_vasp"])
            self.assertIn("--dft-input-manifest", plan["argv"])
            self.assertNotIn("vasp_prepare.py", " ".join(plan["argv"]))
            combined = json.loads(json.dumps(context))
            combined["parameters"]["prepare_script"] = "scripts/prepare_and_run.py"
            self.assertEqual("BLOCKED", self.adapter_module.Adapter().plan(combined)["status"])

    def test_real_pymatgen_writes_poscar_incar_and_kpoints_when_available(self) -> None:
        try:
            real = self.wrapper._load_pymatgen()
        except self.wrapper.ContractError:
            self.skipTest("pymatgen is not installed in this test interpreter")
        with tempfile.TemporaryDirectory() as directory:
            context, fixture = self.fixture(Path(directory))
            source_dir = Path(context["project_root"]) / "input"
            source = source_dir / "source.vasp"
            source.write_text(
                "LiPS\n1.0\n5 0 0\n0 5 0\n0 0 5\nLi P S\n1 1 1\nDirect\n"
                "0 0 0\n0.25 0.25 0.25\n0.5 0.5 0.5\n",
                encoding="utf-8",
            )
            structures = source_dir / "structures.json"
            write_json(
                structures,
                {
                    "schema_version": 1,
                    "structures": [
                        {"id": "s1", "path": "source.vasp", "fingerprint": fingerprint(source)}
                    ],
                },
            )
            api = self.wrapper.PymatgenApi(
                real.Structure,
                real.Incar,
                real.Kpoints,
                real.Poscar,
                FakePotcar,
                real.version,
                real.configured_psp_root,
            )
            runtime_environment = (
                {} if real.configured_psp_root else {"PMG_VASP_PSP_DIR": str(fixture.psp_dir)}
            )
            with patch.dict(os.environ, runtime_environment, clear=True):
                manifest = self.wrapper.prepare_inputs(fixture.args, api)
            self.assertEqual("pymatgen", manifest["generator"]["name"])
            expected_source = "pymatgen-settings" if real.configured_psp_root else "environment"
            self.assertEqual(
                expected_source, manifest["potcar_policy"]["configuration_source"]
            )
            checked = self.adapter_module.Adapter().check(context)
            self.assertEqual("OK", checked["status"], checked)


if __name__ == "__main__":
    unittest.main()
