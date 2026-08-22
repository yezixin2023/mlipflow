"""Focused fixtures for the four safe thin adapters in this workstream."""

from __future__ import annotations

import copy
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


def write_poscar(path: Path, counts: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atom_count = sum(counts.values())
    coordinates = [f"{index / max(atom_count, 1):.8f} 0.0 0.0" for index in range(atom_count)]
    path.write_text(
        "\n".join(
            [
                "contract fixture",
                "1.0",
                "10.0 0.0 0.0",
                "0.0 10.0 0.0",
                "0.0 0.0 10.0",
                " ".join(counts),
                " ".join(str(value) for value in counts.values()),
                "Direct",
                *coordinates,
                "",
            ]
        ),
        encoding="utf-8",
    )


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
    def _valid_case(self, project: Path, attempt: Path) -> tuple[dict, dict]:
        script = project / "scripts" / "sqs.py"
        script.parent.mkdir()
        script.write_text("raise SystemExit('generator must not run during checking')\n")
        prototype = project / "prototype.vasp"
        write_poscar(prototype, {"Li": 1, "Zn": 2})
        composition = project / "composition.json"
        write_json(
            composition,
            {
                "schema_version": 1,
                "alloy_sublattice": {
                    "prototype_species": "Zn",
                    "allowed_species": ["Zn", "Fe"],
                },
                "cluster_cutoffs_angstrom": [5.0],
                "supercell_repeat": [1, 1, 1],
                "n_steps": 100,
                "output_format": "vasp",
                "candidates": [
                    {"id": "sqs-001", "counts": {"Zn": 1, "Fe": 1}},
                    {"id": "sqs-002", "counts": {"Zn": 0, "Fe": 2}},
                ],
            },
        )
        structure_one = attempt / "structures" / "sqs-001.vasp"
        structure_two = attempt / "structures" / "sqs-002.vasp"
        write_poscar(structure_one, {"Li": 1, "Zn": 1, "Fe": 1})
        write_poscar(structure_two, {"Li": 1, "Fe": 2})
        ctx = context(
            project,
            attempt,
            {
                "prototype_structure": "prototype.vasp",
                "composition_manifest": "composition.json",
            },
            {
                "sqs_script": "scripts/sqs.py",
                "interpreter_argv": ["python"],
                "seed": 17,
                "max_candidates": 2,
            },
        )
        result = {
            "schema_version": 1,
            "plugin_id": "high-entropy-structure",
            "status": "OK",
            "seed": 17,
            "candidate_count": 2,
            "input_paths": {
                "prototype": str(prototype.resolve()),
                "composition_manifest": str(composition.resolve()),
            },
            "generator": {"name": "scripts/sqs.py"},
            "method": {"random_seed_policy": "base-seed-plus-candidate-index"},
            "structures": [
                {
                    "id": "sqs-001",
                    "path": "structures/sqs-001.vasp",
                    "composition": {"Zn": 1, "Fe": 1},
                    "media_type": "chemical/x-vasp-poscar",
                    "random_seed": 17,
                    "cluster_vector": [0.0, 1.0],
                },
                {
                    "id": "sqs-002",
                    "path": "structures/sqs-002.vasp",
                    "composition": {"Zn": 0, "Fe": 2},
                    "media_type": "chemical/x-vasp-poscar",
                    "random_seed": 18,
                },
            ],
        }
        return ctx, result

    def test_seeded_plan_and_scientific_collection(self) -> None:
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

            ctx, result = self._valid_case(project, attempt)
            write_json(attempt / "generation-result.json", result)
            self.assertEqual("OK", adapter.check(ctx)["status"])
            collected = adapter.collect(ctx)
            self.assertEqual("OK", collected["status"])
            self.assertEqual(2, collected["metrics"]["structure_count"])
            self.assertEqual("structures/sqs-001.vasp", collected["artifacts"][1]["path"])
            structure = attempt / "structures" / "sqs-001.vasp"
            structure.write_text("tampered\n", encoding="utf-8")
            self.assertEqual("FAIL", adapter.check(ctx)["status"])

    def test_generation_result_rejects_candidate_contract_failures(self) -> None:
        module = load_adapter("high-entropy-structure")
        adapter = module.Adapter()
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            attempt = project / "attempt"
            attempt.mkdir()
            ctx, valid = self._valid_case(project, attempt)

            cases: dict[str, tuple[dict, str]] = {}
            wrong_id = copy.deepcopy(valid)
            wrong_id["structures"][0]["id"] = "not-approved"
            cases["wrong candidate id"] = (wrong_id, "structures[0].approved_id")

            missing = copy.deepcopy(valid)
            missing["structures"].pop()
            missing["candidate_count"] = 1
            cases["missing candidate"] = (missing, "result.candidate_coverage")

            extra = copy.deepcopy(valid)
            extra["structures"].append(
                {
                    **copy.deepcopy(valid["structures"][1]),
                    "id": "sqs-extra",
                    "random_seed": 19,
                }
            )
            extra["candidate_count"] = 3
            cases["extra candidate"] = (extra, "result.candidate_coverage")

            wrong_count = copy.deepcopy(valid)
            wrong_count["structures"][0]["composition"] = {"Zn": 2, "Fe": 0}
            cases["wrong species count"] = (wrong_count, "structures[0].composition")

            wrong_species = copy.deepcopy(valid)
            wrong_species["structures"][0]["composition"] = {"Zn": 1, "Ni": 1}
            cases["wrong composition species"] = (
                wrong_species,
                "structures[0].composition",
            )

            wrong_seed = copy.deepcopy(valid)
            wrong_seed["structures"][1]["random_seed"] = 999
            cases["wrong candidate seed"] = (wrong_seed, "structures[1].random_seed")

            duplicate = copy.deepcopy(valid)
            duplicate["structures"][1]["id"] = "sqs-001"
            cases["duplicate candidate"] = (duplicate, "structures[1].id")

            empty_vector = copy.deepcopy(valid)
            empty_vector["structures"][0]["cluster_vector"] = []
            cases["empty cluster vector"] = (
                empty_vector,
                "structures[0].cluster_vector",
            )

            nonfinite_vector = copy.deepcopy(valid)
            nonfinite_vector["structures"][0]["cluster_vector"] = [float("inf")]
            cases["non-finite cluster vector"] = (
                nonfinite_vector,
                "structures[0].cluster_vector",
            )

            wrong_generator = copy.deepcopy(valid)
            wrong_generator["generator"]["name"] = "unapproved-generator"
            cases["wrong generator name"] = (
                wrong_generator,
                "result.generator_name",
            )

            wrong_seed_policy = copy.deepcopy(valid)
            wrong_seed_policy["method"]["random_seed_policy"] = "result-selected-seed"
            cases["wrong seed policy"] = (wrong_seed_policy, "result.seed_policy")

            wrong_candidate_count = copy.deepcopy(valid)
            wrong_candidate_count["candidate_count"] = 1
            cases["wrong candidate count"] = (
                wrong_candidate_count,
                "result.candidate_count",
            )

            for name, (result, expected_code) in cases.items():
                with self.subTest(name=name):
                    write_json(attempt / "generation-result.json", result)
                    checked = adapter.check(ctx)
                    self.assertEqual("FAIL", checked["status"])
                    self.assertIn(expected_code, {item["code"] for item in checked["diagnostics"]})

    def test_bundled_generator_requires_icet_and_ase_versions(self) -> None:
        module = load_adapter("high-entropy-structure")
        adapter = module.Adapter()
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            attempt = project / "attempt"
            attempt.mkdir()
            ctx, valid = self._valid_case(project, attempt)
            del ctx["parameters"]["sqs_script"]
            valid["generator"] = {"name": "icet.generate_sqs_from_supercells"}
            valid["method"] = {
                "library": "icet",
                "library_version": "recorded-version-not-runtime-parity",
                "api": "generate_sqs_from_supercells",
                "ase_version": "recorded-version-not-runtime-parity",
                "random_seed_policy": "base-seed-plus-candidate-index",
            }
            write_json(attempt / "generation-result.json", valid)

            self.assertEqual("OK", adapter.check(ctx)["status"])

    def test_generation_result_uses_semantics_and_rejects_structure_mismatch(self) -> None:
        module = load_adapter("high-entropy-structure")
        adapter = module.Adapter()
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            attempt = project / "attempt"
            attempt.mkdir()
            ctx, valid = self._valid_case(project, attempt)
            result_path = attempt / "generation-result.json"
            write_json(result_path, valid)

            prototype = project / "prototype.vasp"
            original_prototype = prototype.read_text(encoding="utf-8")
            prototype.write_text(
                original_prototype.replace("contract fixture", "drifted prototype"),
                encoding="utf-8",
            )
            checked = adapter.check(ctx)
            self.assertEqual("OK", checked["status"], "non-scientific title change")
            prototype.write_text(original_prototype, encoding="utf-8")

            composition = project / "composition.json"
            original_composition = composition.read_text(encoding="utf-8")
            composition.write_text(original_composition + " ", encoding="utf-8")
            checked = adapter.check(ctx)
            self.assertEqual("OK", checked["status"], "insignificant JSON whitespace")
            composition.write_text(original_composition, encoding="utf-8")

            structure = attempt / "structures" / "sqs-001.vasp"
            write_poscar(structure, {"Li": 1, "Zn": 2})
            actual_mismatch = copy.deepcopy(valid)
            write_json(result_path, actual_mismatch)
            checked = adapter.check(ctx)
            self.assertEqual("FAIL", checked["status"])
            self.assertIn(
                "structures[0].structure_composition",
                {item["code"] for item in checked["diagnostics"]},
            )

    def test_collect_reads_existing_result_without_executing_generator(self) -> None:
        module = load_adapter("high-entropy-structure")
        adapter = module.Adapter()
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            attempt = project / "attempt"
            attempt.mkdir()
            ctx, valid = self._valid_case(project, attempt)
            marker = project / "generator-executed"
            script = project / "scripts" / "sqs.py"
            script.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
            write_json(attempt / "generation-result.json", valid)

            collected = adapter.collect(ctx)
            self.assertEqual("OK", collected["status"])
            self.assertFalse(marker.exists())


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


class CandidateRankingAdapterTests(unittest.TestCase):
    def test_top_k_rule_is_recomputed_from_explicit_manifests(self) -> None:
        module = load_adapter("candidate-ranking")
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
                project / "metrics.json",
                {
                    "schema_version": 1,
                    "results": [
                        {"candidate_id": "A", "metrics": {"conductivity": 1.0}},
                        {"candidate_id": "B", "metrics": {"conductivity": 2.0}},
                    ],
                },
            )
            parameters = {
                "ranking_script": "scripts/rank.py",
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
                    "metric_results_manifest": "metrics.json",
                },
                parameters,
            )
            self.assertEqual("READY", adapter.plan(ctx)["status"])
            result = {
                "schema_version": 1,
                "plugin_id": "candidate-ranking",
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
            write_json(attempt / "ranking-result.json", result)
            self.assertEqual("OK", adapter.check(ctx)["status"])
            self.assertEqual(2, adapter.collect(ctx)["metrics"]["selected_count"])
            result["ranked_candidates"].reverse()
            write_json(attempt / "ranking-result.json", result)
            self.assertEqual("FAIL", adapter.check(ctx)["status"])


class VoltageAdapterTests(unittest.TestCase):
    def test_cli_formula_check_and_collection(self) -> None:
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
                "candidate-ranking",
                {
                    "candidate_manifest": "candidates.json",
                    "metric_results_manifest": "metrics.json",
                },
                {
                    "ranking_script": "scripts/rank.py",
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
                        "parameters.local_runner"
                        if plugin_id == "dft-labeling"
                        else "backend.unsupported"
                    )
                    self.assertIn(expected_code, {item["code"] for item in plan["diagnostics"]})


if __name__ == "__main__":
    unittest.main()
