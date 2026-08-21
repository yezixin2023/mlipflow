"""Focused tests for the bundled local MAML/DIRECT runner."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "plugins" / "pes-sampling" / "direct_select.py"


def load_runner():
    name = "test_pes_direct_select_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load %s" % RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


class DirectRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = load_runner()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_file_discovery_is_stable_and_deduplicated(self) -> None:
        nested = self.root / "nested"
        nested.mkdir()
        first = self.root / "a.cif"
        second = nested / "b.cif"
        first.write_text("a\n", encoding="utf-8")
        second.write_text("b\n", encoding="utf-8")

        files = self.runner.gather_files([self.root], ["*.cif", "a.*"], recursive=True)

        self.assertEqual([first, second], files)

    def test_lammps_type_map_is_strict(self) -> None:
        self.assertEqual(
            {1: "Li", 2: "S"},
            self.runner.parse_lammps_type_map('{"1":"Li","2":"S"}'),
        )
        with self.assertRaisesRegex(ValueError, "positive integers"):
            self.runner.parse_lammps_type_map('{"0":"Li"}')

    def test_selection_delegates_to_maml_direct_classes(self) -> None:
        calls = {}

        class FakeBirch:
            def __init__(self, n, threshold_init):
                calls["birch"] = (n, threshold_init)

        class FakeSelect:
            def __init__(self, k):
                calls["select"] = k

        class FakeSampler:
            def __init__(self, clustering, select_k_from_clusters):
                calls["sampler"] = (clustering, select_k_from_clusters)

            def fit_transform(self, structures):
                calls["structures"] = list(structures)
                return {"selected_indexes": [2, 0], "PCAfeatures": []}

        direct_module = types.ModuleType("maml.sampling.direct")
        direct_module.BirchClustering = FakeBirch
        direct_module.DIRECTSampler = FakeSampler
        direct_module.SelectKFromClusters = FakeSelect
        modules = {
            "maml": types.ModuleType("maml"),
            "maml.sampling": types.ModuleType("maml.sampling"),
            "maml.sampling.direct": direct_module,
        }
        records = [
            self.runner.StructRecord("s0", self.root / "0.cif", 0, "single"),
            self.runner.StructRecord("s1", self.root / "1.cif", 0, "single"),
            self.runner.StructRecord("s2", self.root / "2.cif", 0, "single"),
        ]

        with patch.dict(sys.modules, modules):
            _, _, indexes, effective_clusters = self.runner.run_direct(
                records, n_clusters=9, threshold_init=0.05, k_per_cluster=1
            )

        self.assertEqual((3, 0.05), calls["birch"])
        self.assertEqual(1, calls["select"])
        self.assertEqual(["s0", "s1", "s2"], calls["structures"])
        self.assertEqual([0, 2], indexes)
        self.assertEqual(3, effective_clusters)

    def test_existing_output_is_rejected_without_deletion(self) -> None:
        output = self.root / "selected"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("keep\n", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            self.runner.write_selected_outputs([], [], output)

        self.assertEqual("keep\n", sentinel.read_text(encoding="utf-8"))

    def test_manifest_uses_confined_relative_output_names(self) -> None:
        class Composition:
            reduced_formula = "Li2S"

        class Structure:
            composition = Composition()

        class FakePoscar:
            def __init__(self, structure):
                self.structure = structure

            def write_file(self, path):
                Path(path).write_text("POSCAR\n", encoding="utf-8")

        pymatgen = types.ModuleType("pymatgen")
        pymatgen_io = types.ModuleType("pymatgen.io")
        pymatgen_vasp = types.ModuleType("pymatgen.io.vasp")
        pymatgen_vasp.Poscar = FakePoscar
        modules = {
            "pymatgen": pymatgen,
            "pymatgen.io": pymatgen_io,
            "pymatgen.io.vasp": pymatgen_vasp,
        }
        source = self.root / "input.cif"
        source.write_text("fixture\n", encoding="utf-8")
        record = self.runner.StructRecord(Structure(), source, 0, "single_structure")
        output = self.root / "selected"

        with patch.dict(sys.modules, modules):
            manifest = self.runner.write_selected_outputs([record], [0], output)

        with manifest.open(newline="", encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual("00001_Li2S.vasp", row["output_file"])
        self.assertFalse(Path(row["output_file"]).is_absolute())
        self.assertEqual("POSCAR\n", (output / row["output_file"]).read_text())


class DirectSmokeReportTests(unittest.TestCase):
    def test_committed_real_runtime_smoke_is_bounded_and_nonsecret(self) -> None:
        report_path = ROOT / "reports" / "direct_local_integration_smoke.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        runner = ROOT / report["implementation"]["runner_locator"]

        self.assertEqual("LOCAL_INTEGRATION_SMOKE_PASS", report["status"])
        self.assertTrue(runner.is_file())
        self.assertTrue(report["scientific_claim"]["real_maml_direct_executed"])
        self.assertTrue(report["scientific_claim"]["adapter_checker_ok"])
        self.assertTrue(report["scientific_claim"]["adapter_collect_ok"])
        self.assertFalse(
            report["scientific_claim"]["mlipflow_core_run_lifecycle_executed"]
        )
        self.assertFalse(
            report["scientific_claim"]["historical_selection_parity_established"]
        )
        self.assertFalse(report["scientific_claim"]["production_sampling_executed"])
        self.assertEqual(4, report["input"]["loaded_structure_count"])
        self.assertEqual(2, report["result"]["selected_structure_count"])
        self.assertEqual(
            [2, 3],
            [item["input_index"] for item in report["result"]["selected_structures"]],
        )
        self.assertEqual("OK", report["result"]["checker_status"])
        self.assertEqual("OK", report["result"]["collect_status"])
        self.assertIsNone(report["bounded_parameters"]["declared_seed"])
        self.assertFalse(
            report["bounded_parameters"]["acknowledgement_contract_exercised"]
        )
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn("/Users/", serialized)
        self.assertNotIn("/public/home/", serialized)
        self.assertNotIn("hfe" + "shell", serialized.lower())


if __name__ == "__main__":
    unittest.main()
