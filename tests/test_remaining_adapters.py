"""Focused fixtures for the four safe thin adapters in this workstream."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_adapter(plugin_id: str) -> ModuleType:
    path = ROOT / "plugins" / plugin_id / "adapter.py"
    spec = importlib.util.spec_from_file_location(f"test_adapter_{plugin_id.replace('-', '_')}", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fingerprint(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def context(project: Path, attempt: Path, inputs: dict, parameters: dict) -> dict:
    return {
        "project_root": str(project),
        "attempt_dir": str(attempt),
        "inputs": inputs,
        "parameters": parameters,
        "backend": "local",
        "resources": {"cpus": 1},
    }


class HighEntropyAdapterTests(unittest.TestCase):
    def test_seeded_plan_and_fingerprint_checked_collection(self) -> None:
        module = load_adapter("high-entropy-structure")
        adapter = module.Adapter()
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            attempt = project / "attempt"
            attempt.mkdir()
            ctx = context(
                project,
                attempt,
                {"prototype_structure": "prototype.cif", "composition_manifest": "composition.json"},
                {
                    "sqs_script": "scripts/sqs.py",
                    "interpreter_argv": ["python"],
                    "seed": 17,
                    "max_candidates": 2,
                },
            )
            before = sorted(project.rglob("*"))
            self.assertEqual([], adapter.validate(ctx))
            plan = adapter.plan(ctx)
            self.assertEqual("READY", plan["status"])
            self.assertIsInstance(plan["argv"], list)
            self.assertIn("17", plan["argv"])
            self.assertEqual(before, sorted(project.rglob("*")), "validate/plan must be pure")

            prototype = project / "prototype.cif"
            prototype.write_text("data_prototype\n", encoding="utf-8")
            composition = project / "composition.json"
            write_json(composition, {"schema_version": 1, "species": ["Li", "P", "S"]})
            structure = attempt / "structures" / "sqs-001.cif"
            structure.parent.mkdir()
            structure.write_text("data_sqs\n_cell_length_a 10\n", encoding="utf-8")
            write_json(
                attempt / "generation-result.json",
                {
                    "schema_version": 1,
                    "plugin_id": "high-entropy-structure",
                    "status": "OK",
                    "seed": 17,
                    "prototype_fingerprint": fingerprint(prototype),
                    "composition_manifest_fingerprint": fingerprint(composition),
                    "structures": [
                        {
                            "id": "sqs-001",
                            "path": "structures/sqs-001.cif",
                            "fingerprint": fingerprint(structure),
                            "composition": {"Li": 8, "P": 8, "S": 32},
                            "media_type": "chemical/x-cif",
                        }
                    ],
                },
            )
            self.assertEqual("OK", adapter.check(ctx)["status"])
            collected = adapter.collect(ctx)
            self.assertEqual("OK", collected["status"])
            self.assertEqual(1, collected["metrics"]["structure_count"])
            self.assertEqual("structures/sqs-001.cif", collected["artifacts"][1]["path"])
            structure.write_text("tampered\n", encoding="utf-8")
            self.assertEqual("FAIL", adapter.check(ctx)["status"])


class DFTLabelingAdapterTests(unittest.TestCase):
    def test_only_standard_converged_result_is_accepted(self) -> None:
        module = load_adapter("dft-labeling")
        adapter = module.Adapter()
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            attempt = project / "attempt"
            attempt.mkdir()
            units = {
                "energy": "eV",
                "length": "angstrom",
                "force": "eV/angstrom",
                "stress": "GPa-voigt-xx-yy-zz-yz-xz-xy",
            }
            ctx = context(
                project,
                attempt,
                {"structures_manifest": "structures.json", "labeling_config": "labeling.json"},
                {
                    "operation": "label",
                    "label_script": "scripts/run_dft.py",
                    "interpreter_argv": ["python"],
                    "engine": "vasp",
                    "completion_policy": {"require_ionic_convergence": False},
                    "units": units,
                },
            )
            self.assertEqual([], adapter.validate(ctx))
            plan = adapter.plan(ctx)
            self.assertEqual("READY", plan["status"])
            self.assertEqual("label", plan["operation"])
            self.assertIsInstance(plan["argv"], list)

            structures_manifest = project / "structures.json"
            labeling_config = project / "labeling.json"
            write_json(structures_manifest, {"schema_version": 1, "structures": [{"id": "s1"}]})
            write_json(labeling_config, {"schema_version": 1, "engine": "vasp"})
            dataset = attempt / "labels.extxyz"
            dataset.write_text("1\nProperties=species:S:1\nLi\n", encoding="utf-8")
            result = {
                "schema_version": 1,
                "plugin_id": "dft-labeling",
                "status": "OK",
                "engine": "vasp",
                "input_fingerprints": {
                    "structures_manifest": fingerprint(structures_manifest),
                    "labeling_config": fingerprint(labeling_config),
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
                "artifacts": [
                    {
                        "name": "labels-extxyz",
                        "path": "labels.extxyz",
                        "media_type": "chemical/x-extxyz",
                        "fingerprint": fingerprint(dataset),
                    }
                ],
            }
            write_json(attempt / "dft-labeling-result.json", result)
            self.assertEqual("OK", adapter.check(ctx)["status"])
            self.assertEqual("labels.extxyz", adapter.collect(ctx)["artifacts"][1]["path"])

            result["completion"]["electronic_converged"] = False
            write_json(attempt / "dft-labeling-result.json", result)
            checked = adapter.check(ctx)
            self.assertEqual("FAIL", checked["status"])
            self.assertIn("completion.electronic", {item["code"] for item in checked["diagnostics"]})


class CompositionScreeningAdapterTests(unittest.TestCase):
    def test_top_k_rule_is_recomputed_from_explicit_manifests(self) -> None:
        module = load_adapter("composition-screening")
        adapter = module.Adapter()
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            attempt = project / "attempt"
            attempt.mkdir()
            write_json(
                project / "candidates.json",
                {"schema_version": 1, "candidates": [{"id": "A"}, {"id": "B"}, {"id": "C"}]},
            )
            write_json(
                project / "transport.json",
                {
                    "schema_version": 1,
                    "results": [
                        {"candidate_id": "A", "metrics": {"conductivity": 1.0}},
                        {"candidate_id": "B", "metrics": {"conductivity": 2.0}},
                    ],
                },
            )
            parameters = {
                "screening_script": "scripts/screen.py",
                "interpreter_argv": ["python"],
                "metric": "conductivity",
                "direction": "maximize",
                "top_k": 2,
                "missing_metric_policy": "reject",
            }
            ctx = context(
                project,
                attempt,
                {
                    "candidate_manifest": "candidates.json",
                    "transport_results_manifest": "transport.json",
                },
                parameters,
            )
            self.assertEqual("READY", adapter.plan(ctx)["status"])
            result = {
                "schema_version": 1,
                "plugin_id": "composition-screening",
                "status": "OK",
                "rule": {
                    "metric": "conductivity",
                    "direction": "maximize",
                    "top_k": 2,
                    "missing_metric_policy": "reject",
                },
                "candidate_count": 3,
                "excluded_missing": ["C"],
                "ranked_candidates": [
                    {"candidate_id": "B", "rank": 1, "value": 2.0},
                    {"candidate_id": "A", "rank": 2, "value": 1.0},
                ],
            }
            write_json(attempt / "screening-result.json", result)
            self.assertEqual("OK", adapter.check(ctx)["status"])
            self.assertEqual(2, adapter.collect(ctx)["metrics"]["selected_count"])
            result["ranked_candidates"].reverse()
            write_json(attempt / "screening-result.json", result)
            self.assertEqual("FAIL", adapter.check(ctx)["status"])


class VoltageAdapterTests(unittest.TestCase):
    def test_cli_formula_and_read_only_replay(self) -> None:
        module = load_adapter("electrochemical-voltage")
        adapter = module.Adapter()
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            attempt = project / "attempt"
            attempt.mkdir()
            energy_manifest = {
                "schema_version": 1,
                "energy_unit": "eV",
                "formula_units": 1,
                "states": [
                    {
                        "id": "Li0",
                        "li_content": 0,
                        "total_energy_ev": -10.0,
                        "completion_status": "OK",
                        "energy_source": "dft://Li0/result.json",
                    },
                    {
                        "id": "Li1",
                        "li_content": 1,
                        "total_energy_ev": -13.0,
                        "completion_status": "OK",
                        "energy_source": "dft://Li1/result.json",
                    },
                    {
                        "id": "Li2",
                        "li_content": 2,
                        "total_energy_ev": -15.0,
                        "completion_status": "OK",
                        "energy_source": "dft://Li2/result.json",
                    },
                ],
            }
            write_json(project / "energies.json", energy_manifest)
            ctx = context(
                project,
                attempt,
                {"energy_manifest": "energies.json"},
                {
                    "lithium_reference_ev": -1.0,
                    "electrons_per_li": 1.0,
                    "energy_unit": "eV",
                    "reference_species": "Li",
                    "composition_order": "increasing-li",
                },
            )
            plan = adapter.plan(ctx)
            self.assertEqual("READY", plan["status"])
            self.assertEqual("compute-from-energies", plan["operation"])
            self.assertEqual("compute-from-energies", plan["argv"][2])
            explicit_ctx = dict(ctx)
            explicit_ctx["parameters"] = {
                **ctx["parameters"],
                "operation": "compute-from-energies",
            }
            self.assertEqual("READY", adapter.plan(explicit_ctx)["status"])
            self.assertIsInstance(plan["argv"], list)
            replayed = adapter.replay(ctx)
            self.assertEqual("OK", replayed["status"])
            self.assertFalse((attempt / "voltage-result.json").exists(), "replay must not write")
            self.assertEqual(
                [2.0, 1.0],
                [step["average_voltage_v"] for step in replayed["result_manifest"]["steps"]],
            )

            exit_code = module._main(
                [
                    "compute",
                    "--energy-manifest",
                    str(project / "energies.json"),
                    "--result-manifest",
                    str(attempt / "voltage-result.json"),
                    "--lithium-reference-ev",
                    "-1.0",
                    "--electrons-per-li",
                    "1.0",
                ]
            )
            self.assertEqual(0, exit_code)
            self.assertEqual("OK", adapter.check(ctx)["status"])
            self.assertEqual(2, adapter.collect(ctx)["metrics"]["step_count"])

            energy_manifest["states"][1]["completion_status"] = "FAIL"
            write_json(project / "energies.json", energy_manifest)
            self.assertEqual("FAIL", adapter.check(ctx)["status"])


class AdapterSafetyBoundaryTests(unittest.TestCase):
    def test_external_wrappers_reject_shells_overrides_and_nonlocal_backend(self) -> None:
        cases = [
            (
                "high-entropy-structure",
                {"prototype_structure": "prototype.cif", "composition_manifest": "composition.json"},
                {
                    "sqs_script": "scripts/sqs.py",
                    "interpreter_argv": ["python"],
                    "seed": 4,
                    "max_candidates": 10,
                },
            ),
            (
                "dft-labeling",
                {"structures_manifest": "structures.json", "labeling_config": "labeling.json"},
                {
                    "operation": "label",
                    "label_script": "scripts/label.py",
                    "interpreter_argv": ["python"],
                    "engine": "vasp",
                    "completion_policy": {"require_ionic_convergence": False},
                    "units": {
                        "energy": "eV",
                        "length": "angstrom",
                        "force": "eV/angstrom",
                        "stress": "GPa",
                    },
                },
            ),
            (
                "composition-screening",
                {
                    "candidate_manifest": "candidates.json",
                    "transport_results_manifest": "transport.json",
                },
                {
                    "screening_script": "scripts/screen.py",
                    "interpreter_argv": ["python"],
                    "metric": "conductivity",
                    "direction": "maximize",
                    "top_k": 3,
                    "missing_metric_policy": "reject",
                },
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / "attempt"
            for plugin_id, inputs, parameters in cases:
                with self.subTest(plugin=plugin_id, attack="shell"):
                    adapter = load_adapter(plugin_id).Adapter()
                    attacked = dict(parameters)
                    attacked["interpreter_argv"] = ["sh"]
                    plan = adapter.plan(context(root, attempt, inputs, attacked))
                    self.assertEqual("BLOCKED", plan["status"])
                    self.assertFalse(plan["executable"])
                    self.assertIn(
                        "parameters.shell_forbidden",
                        {item["code"] for item in plan["diagnostics"]},
                    )
                with self.subTest(plugin=plugin_id, attack="override"):
                    attacked = dict(parameters)
                    attacked["extra_args"] = ["--result-manifest=/tmp/escape.json"]
                    plan = adapter.plan(context(root, attempt, inputs, attacked))
                    self.assertEqual("BLOCKED", plan["status"])
                    self.assertIn(
                        "parameters.extra_args",
                        {item["code"] for item in plan["diagnostics"]},
                    )
                with self.subTest(plugin=plugin_id, attack="backend"):
                    nonlocal_context = context(root, attempt, inputs, parameters)
                    nonlocal_context["backend"] = "ssh-slurm"
                    plan = adapter.plan(nonlocal_context)
                    self.assertEqual("BLOCKED", plan["status"])
                    expected_code = (
                        "parameters.scheduler_runner"
                        if plugin_id == "dft-labeling"
                        else "backend.unsupported"
                    )
                    self.assertIn(expected_code, {item["code"] for item in plan["diagnostics"]})


if __name__ == "__main__":
    unittest.main()
