"""Acceptance tests for the compact manuscript evidence reproduction.

The tests normalize and route already-reported evidence only.  They never load
an MLIP, run MD/AIMD/DFT, train a model, contact a cluster, or modify either
supplied manuscript document.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from mlipflow.config import load_project
from mlipflow.routing import load_registry


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "high_entropy_sulfide_reproduction"
REPRODUCE_PATH = EXAMPLE / "reproduce.py"
BENCHMARK_WRAPPER = ROOT / "plugins" / "mlip-benchmark" / "benchmark_wrapper.py"
SCHEMAS = ROOT / "schemas"
MAIN_SHA256 = "22bfc6c405bc972bca86c141a14239ad4a91718b086e9418784a51ab0aa67e9b"
SI_SHA256 = "0e421d4236d8c9389eddd4e61f9503ada6ff0fd07f70f5bd1c32eea35214e732"


def _load_reproduce():
    spec = importlib.util.spec_from_file_location(
        "test_high_entropy_sulfide_reproduce", REPRODUCE_PATH
    )
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load {}".format(REPRODUCE_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


class ManuscriptReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reproduce = _load_reproduce()
        cls.summary = cls.reproduce.reproduce(
            EXAMPLE,
            benchmark_wrapper_path=BENCHMARK_WRAPPER,
            write=False,
        )

    def test_pinned_real_evidence_and_schema_valid_registry(self) -> None:
        provenance = json.loads(
            (EXAMPLE / "evidence" / "transcription_provenance.json").read_text(encoding="utf-8")
        )
        sources = {item["id"]: item for item in provenance["source_documents"]}
        self.assertEqual(MAIN_SHA256, sources["manuscript-main"]["sha256"])
        self.assertEqual(SI_SHA256, sources["supporting-information"]["sha256"])
        self.assertTrue(all(item["read_only"] for item in sources.values()))
        self.assertTrue(all(not item["included_in_example"] for item in sources.values()))

        project = load_project(EXAMPLE)
        registry = load_registry(EXAMPLE / "model_registry.yaml")
        self.assertEqual("high-entropy-sulfide-manuscript-reproduction", project.project_id)
        self.assertEqual(6, len(registry["models"]))
        self.assertTrue(all(not model["recommended_tasks"] for model in registry["models"]))
        self.assertEqual(
            {"sha256-verified"},
            {item["status"] for item in registry["_evidence_verification"]},
        )

        try:
            from jsonschema import Draft202012Validator
        except ImportError:
            Draft202012Validator = None
        if Draft202012Validator is not None:
            for filename, schema_name in (
                ("project.yaml", "project.schema.json"),
                ("model_registry.yaml", "model-registry.schema.json"),
            ):
                value = json.loads((EXAMPLE / filename).read_text(encoding="utf-8"))
                schema = json.loads((SCHEMAS / schema_name).read_text(encoding="utf-8"))
                Draft202012Validator(schema).validate(value)

        table_s2 = _csv(EXAMPLE / "evidence" / "table_s2_model_implementation.csv")
        by_family = {row["family"]: row for row in table_s2}
        self.assertEqual("66739", by_family["DeePMD"]["high_entropy_sulfides_count"])
        self.assertEqual("train-from-scratch", by_family["DeePMD"]["training_mode"])
        self.assertEqual(
            "se_atten_v2;se_e2_a;se_e2_r;DPA-2",
            by_family["DeePMD"]["architecture_or_checkpoint"],
        )
        self.assertEqual("fine-tune", by_family["M3GNet"]["training_mode"])
        self.assertEqual("fine-tune", by_family["CHGNet"]["training_mode"])

    def test_real_benchmark_replay_and_task_aware_routes(self) -> None:
        summary = self.summary
        self.assertEqual("REPLAY_VERIFIED", summary["scientific_status"])
        self.assertEqual("evidence-replay", summary["mode"])
        normalization = summary["benchmark_normalization"]
        self.assertEqual("replay", normalization["mode"])
        self.assertFalse(normalization["model_execution"])
        self.assertFalse(normalization["network_access"])
        implementation_files = normalization["implementation_files"]
        self.assertEqual(
            {
                "plugins/mlip-benchmark/benchmark_wrapper.py",
                "plugins/mlip-benchmark/plugin.yaml",
            },
            {item["locator"] for item in implementation_files},
        )
        for item in implementation_files:
            self.assertEqual(_sha256(ROOT / item["locator"]), item["sha256"])
        self.assertEqual(
            normalization["implementation_sha256"],
            _sha256(ROOT / normalization["implementation"]),
        )
        self.assertEqual(
            {
                "benchmark/metrics.json",
                "benchmark/benchmark_summary.csv",
                "benchmark/model_ranking.json",
                "benchmark/provenance.json",
            },
            {item["path"] for item in normalization["artifacts"]},
        )

        routes = summary["routing"]
        self.assertEqual("deepmd-dpa2", routes["static-pes"]["selected_model"])
        self.assertEqual("deepmd-se_atten_v2", routes["ionic-transport"]["selected_model"])
        self.assertEqual("chgnet", routes["electrochemical-voltage"]["selected_model"])
        self.assertNotEqual(
            routes["static-pes"]["selected_model"],
            routes["ionic-transport"]["selected_model"],
        )

        for route in routes.values():
            self.assertEqual("sha256-verified", route["ranking"][0]["evidence_status"])
            self.assertTrue(route["decision_provenance"]["source_chain"])
            for source in route["decision_provenance"]["source_chain"]:
                self.assertEqual(71, len(source["source_document_sha256"]))
                self.assertEqual(71, len(source["evidence_file_sha256"]))
        transport_source = routes["ionic-transport"]["decision_provenance"]["source_chain"][0]
        self.assertEqual(
            "sha256:" + MAIN_SHA256,
            transport_source["unit_source_document_sha256"],
        )

        ranking = json.loads(
            (EXAMPLE / "benchmark" / "model_ranking.json").read_text(encoding="utf-8")
        )
        winners = {
            (item["task"], item["metric"]): item["candidates"][0]["model"]
            for item in ranking["rankings"]
            if item["comparable_model_count"] == 6
        }
        self.assertEqual("deepmd-dpa2", winners[("static-pes", "energy_rmse")])
        self.assertEqual("deepmd-dpa2", winners[("static-pes", "force_rmse")])
        self.assertEqual("deepmd-se_atten_v2", winners[("ionic-transport", "conductivity_mae")])
        self.assertEqual("chgnet", winners[("electrochemical-voltage", "voltage_mae")])

    def test_key_values_units_and_speedups_are_recomputed(self) -> None:
        values = self.summary["key_numerical_reconstruction"]
        self.assertEqual(6.63, values["static_pes"]["test_energy_rmse"])
        self.assertEqual("meV/atom", values["static_pes"]["test_energy_rmse_unit"])
        self.assertEqual(151.0, values["static_pes"]["test_force_rmse"])
        self.assertEqual("meV/angstrom", values["static_pes"]["test_force_rmse_unit"])
        self.assertAlmostEqual(
            0.09599531, values["ionic_transport"]["conductivity_mae_against_aimd"]
        )
        self.assertEqual("mS/cm", values["ionic_transport"]["unit"])
        self.assertAlmostEqual(
            0.182777777777778,
            values["electrochemical_voltage"]["voltage_mae_against_dft"],
        )
        self.assertEqual("V", values["electrochemical_voltage"]["unit"])

        expected = {
            "I": ("table_s8_runtime.json", 116),
            "II": ("table_s9_runtime.json", 114),
            "III": ("table_s10_runtime.json", 173),
        }
        observed = {item["prototype"]: item for item in values["simulation_efficiency"]}
        for prototype, (filename, approximate) in expected.items():
            raw = json.loads((EXAMPLE / "evidence" / filename).read_text(encoding="utf-8"))
            deepmd = sum(item["deepmd-se_atten_v2_seconds"] for item in raw["runs"])
            aimd = sum(item["aimd_seconds"] for item in raw["runs"])
            self.assertTrue(
                math.isclose(
                    aimd / deepmd,
                    observed[prototype]["recomputed_acceleration_ratio"],
                    rel_tol=1.0e-14,
                )
            )
            self.assertEqual(
                "dimensionless AIMD_time/MLIP_time",
                observed[prototype]["acceleration_unit"],
            )
            self.assertEqual(approximate, observed[prototype]["published_approximate_ratio"])

    def test_voltage_long_form_is_exact_expansion_of_table_s11(self) -> None:
        wide = _csv(EXAMPLE / "evidence" / "table_s11_voltage.csv")
        long_rows = _csv(EXAMPLE / "evidence" / "table_s11_voltage_long.csv")
        self.assertEqual(9, len(wide))
        self.assertEqual(18, len(long_rows))
        models = (
            "deepmd-se_atten_v2",
            "deepmd-se_e2_a",
            "deepmd-se_e2_r",
            "deepmd-dpa2",
            "m3gnet",
            "chgnet",
        )
        indexed = {
            (
                row["prototype"],
                row["metal_composition"],
                row["lithium_sequence"],
                int(row["stage"]),
            ): row
            for row in long_rows
        }
        for row in wide:
            for stage in (1, 2):
                expanded = indexed[
                    (
                        row["prototype"],
                        row["metal_composition"],
                        row["lithium_sequence"],
                        stage,
                    )
                ]
                self.assertEqual(
                    float(row["dft_stage_{}".format(stage)]), float(expanded["reference_dft_v"])
                )
                for model in models:
                    self.assertEqual(
                        float(row["{}_stage_{}".format(model, stage)]),
                        float(expanded[model]),
                    )

    def test_large_supercell_screening_replay_is_complete_and_portable(self) -> None:
        screening = self.summary["large_supercell_screening"]
        self.assertEqual("REPLAY_VERIFIED", screening["status"])
        self.assertEqual("PINNED_LOCAL_RESULT_ARTIFACT", screening["evidence_level"])
        inputs = screening["input_set"]
        self.assertEqual(247, inputs["candidate_count"])
        self.assertEqual(247, inputs["source_structure_count"])
        self.assertEqual(247, inputs["unique_source_structure_count"])
        self.assertEqual(247, inputs["evaluated_count"])
        self.assertEqual(0, inputs["excluded_missing_count"])
        self.assertEqual(28, inputs["expected_metal_sites_per_candidate"])
        self.assertEqual(
            ["Mn", "Fe", "Ni", "Cu", "Zn"], inputs["canonical_element_order"]
        )
        self.assertEqual("S/m", screening["metric"]["unit"])
        self.assertEqual("maximize", screening["rule"]["direction"])
        self.assertEqual("candidate_id-ascending", screening["rule"]["tie_break"])
        self.assertEqual("0.3.0", screening["implementation"]["plugin_version"])

        artifacts = screening["source_artifacts"]
        self.assertEqual(
            "sha256:30ada4653684618ab1594b4b428502ba324908b208028910d68b8e4785f5793a",
            artifacts["ranking"]["sha256"],
        )
        self.assertEqual(
            "sha256:d030d2fdbd9c042b13a1a15c6a124ef2b953d62ca1170e963c272a57b43aa90d",
            artifacts["candidate_manifest"]["sha256"],
        )
        self.assertEqual(
            "sha256:e3c4f4466b872b575345e4a10560b311fff19a8ed049383a82c18132534cd1c0",
            artifacts["transport_results_manifest"]["sha256"],
        )

        top = screening["top_candidates"]
        self.assertEqual(10, len(top))
        self.assertEqual(list(range(1, 11)), [item["rank"] for item in top])
        self.assertEqual("Mn6_Fe3_Ni8_Cu4_Zn7", top[0]["candidate_id"])
        self.assertAlmostEqual(18.420176685825222, top[0]["value"], places=14)
        self.assertEqual(
            sorted(top, key=lambda item: (-item["value"], item["candidate_id"])), top
        )
        for item in top:
            self.assertEqual(28, sum(item["composition"].values()))
            self.assertEqual(set(inputs["canonical_element_order"]), set(item["composition"]))

        sources = screening["legacy_source_collection"]
        self.assertEqual(11, sources["candidate_source_file_count"])
        self.assertEqual(11, sources["metric_source_file_count"])
        for item in sources["candidate_sources"] + sources["metric_sources"]:
            self.assertEqual(item["basename"], Path(item["basename"]).name)
            self.assertEqual(71, len(item["sha256"]))
        for item in screening["implementation"]["files"]:
            self.assertFalse(Path(item["locator"]).is_absolute())
            self.assertEqual(71, len(item["sha256"]))
        legacy_implementation = {
            item["locator"]: item["sha256"]
            for item in screening["implementation"]["files"]
        }
        self.assertEqual(self.reproduce.LEGACY_RANKING_IMPLEMENTATION, legacy_implementation)
        rendered = json.dumps(screening, sort_keys=True)
        for prefix in ("/Users/", "/public/", "/private/", "/tmp/"):
            self.assertNotIn(prefix, rendered)

    def test_top_candidate_link_separates_postprocess_from_high_fidelity(self) -> None:
        validation = self.summary["top_candidate_validation"]
        identity = validation["identity_link"]
        self.assertEqual("EXACT_IDENTITY_MAPPING", identity["status"])
        self.assertEqual("Mn6_Fe3_Ni8_Cu4_Zn7", identity["ranked_candidate_id"])
        self.assertEqual("6_3_8_4_7", identity["historical_folder_id"])
        self.assertEqual(
            ["Mn", "Fe", "Ni", "Cu", "Zn"], identity["historical_token_order"]
        )
        self.assertEqual(
            {"Mn": 6, "Fe": 3, "Ni": 8, "Cu": 4, "Zn": 7},
            identity["parsed_composition"],
        )

        parity = validation["historical_transport_parity"]
        self.assertEqual("EXACT_NUMERICAL_PARITY", parity["status"])
        self.assertEqual(
            "READ_ONLY_HISTORICAL_POSTPROCESS_PARITY", parity["evidence_level"]
        )
        self.assertTrue(parity["not_high_fidelity_validation"])
        self.assertEqual("block-1", parity["historical_comparison"]["dataset_id"])
        self.assertTrue(parity["historical_comparison"]["diffusivity_all_close"])
        self.assertTrue(parity["historical_comparison"]["conductivity_all_close"])
        self.assertAlmostEqual(
            18.420176685825222,
            parity["arrhenius"]["mean_msd_over_time"]["conductivity_S_m"],
            places=14,
        )
        self.assertAlmostEqual(
            validation["screening_metric"]["value"],
            parity["screening_value_match"]["replayed_mean_msd_value_S_m"],
            places=14,
        )
        implementation = parity["implementation_provenance"]
        self.assertEqual("1.0", implementation["wrapper_version"])
        self.assertEqual(
            _sha256(ROOT / implementation["wrapper_locator"]),
            implementation["wrapper_sha256"],
        )
        self.assertEqual(2, len(implementation["historical_scripts"]))
        for name in (
            "model_executed",
            "md_executed",
            "aimd_executed",
            "dft_executed",
            "scheduler_used",
            "network_used",
        ):
            self.assertFalse(parity["execution"][name])

        high_fidelity = validation["high_fidelity_validation"]
        self.assertEqual("EXTERNAL_VALIDATION_PENDING", high_fidelity["status"])
        self.assertEqual(
            "NO_AIMD_OR_DFT_REFERENCE_ARTIFACT", high_fidelity["evidence_level"]
        )
        self.assertIsNone(high_fidelity["reference_artifact"])
        self.assertEqual(
            "NOT_TESTABLE_WITH_AVAILABLE_DATA", high_fidelity["numerical_parity"]
        )
        self.assertGreaterEqual(len(high_fidelity["blockers"]), 2)
        scope = self.summary["scope_boundaries"]
        self.assertEqual(
            "EXACT_NUMERICAL_PARITY", scope["top_candidate_historical_transport"]
        )
        self.assertEqual(
            "EXTERNAL_VALIDATION_PENDING",
            scope["top_candidate_aimd_dft_high_fidelity"],
        )

    def test_unseen_transfer_is_claim_level_and_not_numerical_parity(self) -> None:
        unseen = self.summary["unseen_transfer_validation"]
        self.assertEqual("CLAIM_LEVEL_DOCUMENT_EVIDENCE", unseen["evidence_level"])
        self.assertEqual("EXTERNAL_VALIDATION_PENDING", unseen["status"])
        self.assertEqual(
            "sha256:" + MAIN_SHA256, unseen["source_document"]["sha256"]
        )
        self.assertTrue(unseen["source_document"]["read_only"])
        system = unseen["document_claims"]["unseen_system"]
        self.assertEqual("Li24M12(PS4)16", system["reported_formula"])
        self.assertEqual("Li6M3(PS4)4", system["expanded_formula"])
        self.assertEqual(787, system["reported_aimd_configuration_count"])
        transfer = unseen["document_claims"]["composition_transfer"]
        self.assertEqual(3, transfer["reported_comparison_count"])
        self.assertFalse(transfer["composition_ids_available_in_supplied_tables"])
        self.assertFalse(transfer["per_composition_values_available_in_supplied_tables"])
        self.assertEqual(
            "NOT_TESTABLE_WITH_AVAILABLE_DATA", unseen["numerical_parity"]["status"]
        )
        self.assertFalse(unseen["numerical_parity"]["performed"])
        self.assertGreaterEqual(len(unseen["blockers"]), 4)
        scope = self.summary["scope_boundaries"]
        self.assertEqual(
            "CLAIM_LEVEL_DOCUMENT_EVIDENCE", scope["unseen_transfer_evidence_level"]
        )
        self.assertEqual(
            "EXTERNAL_VALIDATION_PENDING", scope["unseen_transfer_validation"]
        )
        self.assertEqual(
            "NOT_TESTABLE_WITH_AVAILABLE_DATA", scope["unseen_transfer_numerical_parity"]
        )

    def test_replay_is_portable_and_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            relocated = Path(temporary) / "reproduction"
            shutil.copytree(EXAMPLE, relocated)
            self.reproduce.reproduce(
                relocated,
                benchmark_wrapper_path=BENCHMARK_WRAPPER,
                write=True,
            )
            for folder, names in (
                ("benchmark", self.reproduce.BENCHMARK_OUTPUTS),
                ("reports", self.reproduce.REPORT_OUTPUTS),
            ):
                for name in names:
                    self.assertEqual(
                        (EXAMPLE / folder / name).read_bytes(),
                        (relocated / folder / name).read_bytes(),
                    )
            rendered = b"".join(
                (relocated / "benchmark" / name).read_bytes()
                for name in self.reproduce.BENCHMARK_OUTPUTS
            )
            self.assertNotIn(str(relocated).encode("utf-8"), rendered)

            long_evidence = relocated / "evidence" / "table_s11_voltage_long.csv"
            long_evidence.write_text(
                long_evidence.read_text(encoding="utf-8").replace(",2.91,V", ",9.91,V", 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                self.reproduce.ReproductionEvidenceError, "evidence (size|SHA-256) mismatch"
            ):
                self.reproduce.reproduce(
                    relocated,
                    benchmark_wrapper_path=BENCHMARK_WRAPPER,
                    write=False,
                )

    def test_repository_locator_hash_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary) / "mlipflow"
            relocated = checkout / "examples" / "high_entropy_sulfide_reproduction"
            shutil.copytree(EXAMPLE, relocated)
            locators = (
                "plugins/mlip-benchmark/benchmark_wrapper.py",
                "plugins/mlip-benchmark/plugin.yaml",
                "plugins/candidate-ranking/rank.py",
                "plugins/candidate-ranking/normalize_legacy.py",
                "plugins/candidate-ranking/adapter.py",
                "plugins/candidate-ranking/plugin.yaml",
                "plugins/ionic-transport/adapter.py",
            )
            for locator in locators:
                target = checkout / locator
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / locator, target)

            copied_wrapper = checkout / self.reproduce.BENCHMARK_WRAPPER_LOCATOR
            self.reproduce.reproduce(
                relocated,
                benchmark_wrapper_path=copied_wrapper,
                write=True,
            )
            tampered = checkout / "plugins" / "candidate-ranking" / "plugin.yaml"
            tampered.write_text(
                tampered.read_text(encoding="utf-8").replace(
                    '"id": "candidate-ranking"', '"id": "candidate-rankings"', 1
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                self.reproduce.ReproductionEvidenceError,
                "current candidate-ranking manifest identity drift",
            ):
                self.reproduce.reproduce(
                    relocated,
                    benchmark_wrapper_path=copied_wrapper,
                    write=False,
                )

    def test_fixture_is_compact_and_contains_no_source_document(self) -> None:
        self.assertFalse(any(EXAMPLE.rglob("*.docx")))
        size = sum(path.stat().st_size for path in EXAMPLE.rglob("*") if path.is_file())
        self.assertLess(size, 500_000)

    def test_top_level_acceptance_report_pins_the_canonical_report(self) -> None:
        index = json.loads(
            (ROOT / "reports" / "manuscript_reproduction_summary.json").read_text(encoding="utf-8")
        )
        canonical = ROOT / index["canonical_report"]["path"]
        digest = "sha256:" + hashlib.sha256(canonical.read_bytes()).hexdigest()
        self.assertEqual(index["canonical_report"]["sha256"], digest)
        self.assertEqual("REPLAY_VERIFIED", index["status"])
        self.assertEqual("deepmd-dpa2", index["selected_models"]["static-pes"])
        self.assertEqual("deepmd-se_atten_v2", index["selected_models"]["ionic-transport"])
        self.assertEqual("chgnet", index["selected_models"]["electrochemical-voltage"])
        self.assertFalse(index["claim_boundary"]["model_execution"])


if __name__ == "__main__":
    unittest.main()
