#!/usr/bin/env python3
"""Reconstruct manuscript model-selection conclusions from recorded evidence.

This example driver never imports an MLIP framework and never runs a model.
It verifies the read-only transcriptions, independently recomputes transparent
table reductions, calls the real ``mlip-benchmark`` replay normalizer, and then
uses MLIPFlow's ordinary task-aware routing implementation.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCENARIO = "high-entropy-sulfide-manuscript-v1"
BENCHMARK_INPUT = "evidence/benchmark_records.json"
BENCHMARK_OUTPUTS = (
    "metrics.json",
    "benchmark_summary.csv",
    "model_ranking.json",
    "provenance.json",
)
REPORT_OUTPUTS = (
    "manuscript_reproduction_summary.json",
    "manuscript_reproduction_summary.md",
)
BENCHMARK_WRAPPER_LOCATOR = "plugins/mlip-benchmark/benchmark_wrapper.py"
BENCHMARK_NORMALIZATION_LOCATOR = "plugins/mlip-benchmark/benchmark_normalization.py"
MODEL_RUNTIME_LOCATOR = "src/mlipflow/science/model_runtime.py"
BENCHMARK_ADAPTER_LOCATOR = "plugins/mlip-benchmark/adapter.py"
LEGACY_RANKING_IMPLEMENTATION = (
    "plugins/composition-screening/screen.py",
    "plugins/composition-screening/normalize_legacy.py",
    "plugins/composition-screening/adapter.py",
    "plugins/composition-screening/plugin.yaml",
)
CURRENT_RANKING_ADAPTER_LOCATOR = "plugins/candidate-ranking/adapter.py"
CURRENT_RANKER_LOCATOR = "plugins/candidate-ranking/rank.py"


class ReproductionEvidenceError(ValueError):
    """Raised when recorded evidence or a deterministic reduction is invalid."""


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _repository_file(repository_root: Path, locator: Any, label: str) -> Path:
    """Resolve one portable repository locator without allowing path escape."""

    if not isinstance(locator, str) or not locator:
        raise ReproductionEvidenceError("{} locator is missing".format(label))
    relative = Path(locator)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ReproductionEvidenceError("{} locator is not portable: {}".format(label, locator))
    repository_root = repository_root.resolve()
    path = (repository_root / relative).resolve()
    try:
        path.relative_to(repository_root)
    except ValueError as exc:
        raise ReproductionEvidenceError(
            "{} locator escapes the repository: {}".format(label, locator)
        ) from exc
    if not path.is_file():
        raise ReproductionEvidenceError(
            "{} locator does not resolve to a file: {}".format(label, locator)
        )
    return path


def _repository_root_from_benchmark_wrapper(benchmark_wrapper_path: Path) -> Path:
    """Derive the checkout root while preserving the wrapper's portable locator."""

    relative = Path(BENCHMARK_WRAPPER_LOCATOR)
    repository_root = benchmark_wrapper_path.resolve()
    for _part in relative.parts:
        repository_root = repository_root.parent
    resolved = _repository_file(repository_root, BENCHMARK_WRAPPER_LOCATOR, "benchmark wrapper")
    if resolved != benchmark_wrapper_path.resolve():
        raise ReproductionEvidenceError(
            "benchmark wrapper path does not match {}".format(BENCHMARK_WRAPPER_LOCATOR)
        )
    return repository_root


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ReproductionEvidenceError(
            "{} mismatch: derived {!r}, prepared {!r}".format(label, actual, expected)
        )


def _verify_transcription_files(root: Path) -> Dict[str, Any]:
    evidence_root = root / "evidence"
    provenance = _json(evidence_root / "transcription_provenance.json")
    if provenance.get("not_a_fresh_calculation") is not True:
        raise ReproductionEvidenceError("transcription must explicitly reject a fresh-calculation claim")

    source_documents = {
        item["id"]: item for item in provenance.get("source_documents", [])
    }
    if len(source_documents) != 2:
        raise ReproductionEvidenceError("exactly the supplied main manuscript and SI must be recorded")
    for item in source_documents.values():
        if not isinstance(item.get("basename"), str) or not item["basename"]:
            raise ReproductionEvidenceError("source document requires a basename")
        if item.get("included_in_example") is not False or item.get("read_only") is not True:
            raise ReproductionEvidenceError("source documents must remain external and read-only")

    evidence_files: Dict[str, Dict[str, Any]] = {}
    for item in provenance.get("evidence_files", []):
        relative = item.get("path")
        if not isinstance(relative, str) or not relative:
            raise ReproductionEvidenceError("every evidence file needs a portable relative path")
        path = evidence_root / relative
        if not path.is_file():
            raise ReproductionEvidenceError("missing evidence file: {}".format(relative))
        if relative in evidence_files:
            raise ReproductionEvidenceError("duplicate evidence provenance: {}".format(relative))
        evidence_files[relative] = item
    return {
        "manifest": provenance,
        "source_documents": source_documents,
        "evidence_files": evidence_files,
    }


def _prepared_lookup(root: Path) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, str], Dict[str, Any]]]:
    records = _json(root / BENCHMARK_INPUT).get("records", [])
    if not isinstance(records, list):
        raise ReproductionEvidenceError("benchmark_records.json needs records[]")
    lookup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for record in records:
        key = (record["model"], record["task"], record["metric"])
        if key in lookup and record["task"] != "simulation-efficiency":
            raise ReproductionEvidenceError("duplicate prepared benchmark record: {}".format(key))
        if record["task"] != "simulation-efficiency":
            lookup[key] = record
    return records, lookup


def _verify_prepared_benchmark_reductions(root: Path) -> Dict[str, Any]:
    records, lookup = _prepared_lookup(root)
    evidence = root / "evidence"

    static_rows = _csv(evidence / "table_3_pes_rmse.csv")
    static_models = set()
    for row in static_rows:
        model = row["model_id"]
        static_models.add(model)
        for metric, column, unit in (
            ("energy_rmse", "energy_test_rmse_mev_atom", "meV/atom"),
            ("force_rmse", "force_test_rmse_mev_a", "meV/angstrom"),
        ):
            prepared = lookup[(model, "static-pes", metric)]
            _assert_close(float(row[column]), float(prepared["value"]), "{}/{!s}".format(model, metric))
            if prepared["unit"] != unit or prepared["direction"] != "minimize":
                raise ReproductionEvidenceError("static metric unit/direction drift")

    conductivity_rows = _csv(evidence / "table_s3_ionic_conductivity.csv")
    if {row["unit"] for row in conductivity_rows} != {"mS cm^-1"}:
        raise ReproductionEvidenceError("Table S3 conductivity unit provenance drift")
    for model in sorted(static_models):
        errors = [
            abs(float(row[model]) - float(row["aimd"])) for row in conductivity_rows
        ]
        derived = sum(errors) / len(errors)
        prepared = lookup[(model, "ionic-transport", "conductivity_mae")]
        _assert_close(derived, float(prepared["value"]), model + "/conductivity_mae")
        if prepared["unit"] != "mS/cm" or prepared["sample_count"] != len(errors):
            raise ReproductionEvidenceError("conductivity unit/sample-count drift")

    voltage_rows = _csv(evidence / "table_s11_voltage.csv")
    if {row["unit"] for row in voltage_rows} != {"V"}:
        raise ReproductionEvidenceError("Table S11 voltage unit provenance drift")
    voltage_long_rows = _csv(evidence / "table_s11_voltage_long.csv")
    if len(voltage_long_rows) != 2 * len(voltage_rows):
        raise ReproductionEvidenceError("Table S11 long-form expansion must contain 18 rows")
    long_lookup = {
        (
            row["prototype"],
            row["metal_composition"],
            row["lithium_sequence"],
            int(row["stage"]),
        ): row
        for row in voltage_long_rows
    }
    if len(long_lookup) != len(voltage_long_rows):
        raise ReproductionEvidenceError("Table S11 long-form stage identities are not unique")
    for wide in voltage_rows:
        for stage in (1, 2):
            key = (
                wide["prototype"],
                wide["metal_composition"],
                wide["lithium_sequence"],
                stage,
            )
            long = long_lookup.get(key)
            if long is None:
                raise ReproductionEvidenceError("Table S11 long-form row missing: {}".format(key))
            _assert_close(
                float(wide["dft_stage_{}".format(stage)]),
                float(long["reference_dft_v"]),
                "Table S11 long DFT/{}".format(key),
            )
            for model in sorted(static_models):
                _assert_close(
                    float(wide["{}_stage_{}".format(model, stage)]),
                    float(long[model]),
                    "Table S11 long {}/{}".format(model, key),
                )
            if long["unit"] != "V":
                raise ReproductionEvidenceError("Table S11 long-form unit drift")
    for model in sorted(static_models):
        errors = [
            abs(float(row[model]) - float(row["reference_dft_v"]))
            for row in voltage_long_rows
        ]
        derived = sum(errors) / len(errors)
        prepared = lookup[(model, "electrochemical-voltage", "voltage_mae")]
        _assert_close(derived, float(prepared["value"]), model + "/voltage_mae")
        if prepared["unit"] != "V" or prepared["sample_count"] != len(errors):
            raise ReproductionEvidenceError("voltage unit/sample-count drift")

    efficiency_records = {
        record["split"]: record
        for record in records
        if record["task"] == "simulation-efficiency"
    }
    speedups: List[Dict[str, Any]] = []
    for table_number in (8, 9, 10):
        table = _json(evidence / "table_s{}_runtime.json".format(table_number))
        runs = table["runs"]
        deepmd_seconds = sum(item["deepmd-se_atten_v2_seconds"] for item in runs)
        aimd_seconds = sum(item["aimd_seconds"] for item in runs)
        compositions = {item["metal_composition"] for item in runs}
        split = "prototype-{}".format(table["prototype"].lower())
        prepared = efficiency_records[split]
        speedup = aimd_seconds / deepmd_seconds
        _assert_close(speedup, float(prepared["value"]), split + "/acceleration_ratio")
        if prepared["unit"] != "1" or prepared["sample_count"] != len(runs):
            raise ReproductionEvidenceError("acceleration unit/sample-count drift")
        speedups.append(
            {
                "prototype": table["prototype"],
                "source_table": table["source_table"],
                "timing_pair_count": len(runs),
                "composition_count": len(compositions),
                "deepmd_total_seconds": deepmd_seconds,
                "aimd_total_seconds": aimd_seconds,
                "deepmd_mean_hours_per_composition": deepmd_seconds / len(compositions) / 3600.0,
                "aimd_mean_hours_per_composition": aimd_seconds / len(compositions) / 3600.0,
                "reported_deepmd_average_hours": table["reported_average_hours"]["deepmd-se_atten_v2"],
                "reported_aimd_average_hours": table["reported_average_hours"]["aimd"],
                "recomputed_acceleration_ratio": speedup,
                "acceleration_unit": "dimensionless AIMD_time/MLIP_time",
                "published_approximate_ratio": prepared["dimensions"]["published_approximate_ratio"],
            }
        )

    expected_record_count = len(static_models) * 4 + 3
    if len(records) != expected_record_count:
        raise ReproductionEvidenceError(
            "prepared benchmark record count drift: {} != {}".format(
                len(records), expected_record_count
            )
        )
    return {
        "models": sorted(static_models),
        "record_count": len(records),
        "speedups": speedups,
    }


def _require_portable_evidence(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_portable_evidence(item, "{}/{}".format(label, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_portable_evidence(item, "{}/{}".format(label, index))
    elif isinstance(value, str) and (Path(value).is_absolute() or value.startswith("~")):
        raise ReproductionEvidenceError(
            "{} contains a non-portable absolute locator".format(label)
        )


def _candidate_composition(candidate_id: str, element_order: Sequence[str]) -> Dict[str, int]:
    parts = candidate_id.split("_")
    if len(parts) != len(element_order):
        raise ReproductionEvidenceError("candidate ID has the wrong number of element tokens")
    composition: Dict[str, int] = {}
    for part, element in zip(parts, element_order):
        match = re.fullmatch(re.escape(element) + r"([1-9][0-9]*)", part)
        if match is None:
            raise ReproductionEvidenceError(
                "candidate ID token {!r} does not match {}<count>".format(part, element)
            )
        composition[element] = int(match.group(1))
    return composition


def _verify_screening_evidence(
    root: Path,
    evidence_context: Mapping[str, Any],
    repository_root: Path,
) -> Dict[str, Any]:
    evidence = root / "evidence"
    screening = _json(evidence / "large_supercell_screening_top10.json")
    top_link = _json(evidence / "top_candidate_transport_link.json")
    unseen = _json(evidence / "unseen_transfer_claim.json")
    for name, value in (
        ("large_supercell_screening_top10", screening),
        ("top_candidate_transport_link", top_link),
        ("unseen_transfer_claim", unseen),
    ):
        _require_portable_evidence(value, name)

    if screening.get("status") != "REPLAY_VERIFIED":
        raise ReproductionEvidenceError("large-supercell screening is not replay-verified")
    if screening.get("evidence_level") != "RECORDED_LOCAL_RESULT_ARTIFACT":
        raise ReproductionEvidenceError("large-supercell screening evidence level drift")
    inputs = screening.get("input_set", {})
    expected_counts = {
        "candidate_count": 247,
        "source_structure_count": 247,
        "unique_source_structure_count": 247,
        "evaluated_count": 247,
        "excluded_missing_count": 0,
        "expected_metal_sites_per_candidate": 28,
    }
    for name, expected in expected_counts.items():
        if inputs.get(name) != expected:
            raise ReproductionEvidenceError(
                "large-supercell input {} drift: {!r}".format(name, inputs.get(name))
            )
    canonical_order = ["Mn", "Fe", "Ni", "Cu", "Zn"]
    if inputs.get("canonical_element_order") != canonical_order:
        raise ReproductionEvidenceError("canonical screening element order drift")
    if inputs.get("candidate_element_order") != ["Zn", "Fe", "Cu", "Ni", "Mn"]:
        raise ReproductionEvidenceError("legacy screening element order drift")
    if inputs.get("normalizer") != "mlipflow-legacy-li10-screening-v1":
        raise ReproductionEvidenceError("large-supercell normalizer drift")

    source_artifacts = screening.get("source_artifacts", {})
    expected_artifacts = {
        "ranking": "external-screening-artifact/ranking.json",
        "candidate_manifest": "external-screening-artifact/candidates.json",
        "transport_results_manifest": "external-screening-artifact/transport.json",
    }
    for name, locator in expected_artifacts.items():
        item = source_artifacts.get(name, {})
        if item.get("locator") != locator or item.get("read_only") is not True:
            raise ReproductionEvidenceError("screening source artifacts must be read-only")

    collection = screening.get("legacy_source_collection", {})
    for key, count_key in (
        ("candidate_sources", "candidate_source_file_count"),
        ("metric_sources", "metric_source_file_count"),
    ):
        records = collection.get(key, [])
        if len(records) != 11 or collection.get(count_key) != 11:
            raise ReproductionEvidenceError("screening {} collection count drift".format(key))
        if len({item.get("basename") for item in records}) != len(records):
            raise ReproductionEvidenceError("screening {} basenames are not unique".format(key))
        for item in records:
            basename = item.get("basename")
            if not isinstance(basename, str) or Path(basename).name != basename:
                raise ReproductionEvidenceError("screening source requires basename-only locator")

    metric = screening.get("metric", {})
    if metric != {
        "name": "ionic_conductivity_300k_s_per_m",
        "unit": "S/m",
        "source_convention": "legacy-text-value-no-rescaling",
    }:
        raise ReproductionEvidenceError("large-supercell metric definition drift")
    rule = screening.get("rule", {})
    if rule != {
        "direction": "maximize",
        "missing_metric_policy": "error",
        "top_k": 10,
        "tie_break": "candidate_id-ascending",
    }:
        raise ReproductionEvidenceError("large-supercell ranking rule drift")
    implementation = screening.get("implementation", {})
    if (
        implementation.get("ranking_implementation")
        != "mlipflow-deterministic-single-metric-screen"
        or implementation.get("ranking_implementation_version") != 1
        or implementation.get("plugin_id") != "composition-screening"
        or implementation.get("plugin_version") != "0.3.0"
    ):
        raise ReproductionEvidenceError("large-supercell implementation provenance drift")
    implementation_files = implementation.get("files", [])
    if len(implementation_files) != 4:
        raise ReproductionEvidenceError("screening implementation file provenance is incomplete")
    observed_legacy_implementation = tuple(
        item.get("locator") for item in implementation_files
    )
    if observed_legacy_implementation != LEGACY_RANKING_IMPLEMENTATION:
        raise ReproductionEvidenceError("legacy screening implementation provenance drift")

    top_candidates = screening.get("top_candidates", [])
    if len(top_candidates) != 10:
        raise ReproductionEvidenceError("large-supercell compact ranking must have ten entries")
    if [item.get("rank") for item in top_candidates] != list(range(1, 11)):
        raise ReproductionEvidenceError("large-supercell ranks must be contiguous 1..10")
    if len({item.get("candidate_id") for item in top_candidates}) != 10:
        raise ReproductionEvidenceError("large-supercell top candidates must be unique")
    for item in top_candidates:
        value = item.get("value")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ReproductionEvidenceError("large-supercell ranking contains a non-finite value")
        parsed = _candidate_composition(item["candidate_id"], canonical_order)
        if item.get("composition") != parsed:
            raise ReproductionEvidenceError("candidate ID/composition mismatch")
        if sum(parsed.values()) != inputs["expected_metal_sites_per_candidate"]:
            raise ReproductionEvidenceError("candidate does not contain 28 metal sites")
    expected_order = sorted(
        top_candidates, key=lambda item: (-float(item["value"]), item["candidate_id"])
    )
    if top_candidates != expected_order:
        raise ReproductionEvidenceError("large-supercell top ten violates value/tie ordering")

    _repository_file(
        repository_root,
        CURRENT_RANKING_ADAPTER_LOCATOR,
        "current candidate-ranking adapter",
    )

    ranker_path = _repository_file(
        repository_root,
        CURRENT_RANKER_LOCATOR,
        "current candidate-ranking implementation",
    )
    ranker_spec = importlib.util.spec_from_file_location(
        "mlipflow_manuscript_candidate_ranking_compatibility", ranker_path
    )
    if ranker_spec is None or ranker_spec.loader is None:
        raise ReproductionEvidenceError("cannot load current candidate-ranking implementation")
    ranker = importlib.util.module_from_spec(ranker_spec)
    ranker_spec.loader.exec_module(ranker)
    compatibility_result = ranker.build_result(
        {
            "schema_version": 1,
            "candidates": [{"id": item["candidate_id"]} for item in top_candidates],
        },
        {
            "schema_version": 1,
            "results": [
                {
                    "candidate_id": item["candidate_id"],
                    "metrics": {metric["name"]: item["value"]},
                }
                for item in top_candidates
            ],
        },
        metric=metric["name"],
        direction=rule["direction"],
        top_k=rule["top_k"],
        missing_metric_policy=rule["missing_metric_policy"],
    )
    if (
        compatibility_result.get("plugin_id") != "candidate-ranking"
        or compatibility_result.get("ranked_candidates")
        != [
            {
                "candidate_id": item["candidate_id"],
                "rank": item["rank"],
                "value": item["value"],
            }
            for item in top_candidates
        ]
    ):
        raise ReproductionEvidenceError("current candidate-ranking compatibility replay drift")

    candidate_mapping = top_link.get("candidate_mapping", {})
    top_candidate = top_candidates[0]
    if candidate_mapping.get("status") != "EXACT_COMPOSITION_MAPPING":
        raise ReproductionEvidenceError("top-candidate composition mapping is not exact")
    if candidate_mapping.get("ranked_candidate_id") != top_candidate["candidate_id"]:
        raise ReproductionEvidenceError("top-candidate mapping does not match screening rank 1")
    if candidate_mapping.get("historical_token_order") != canonical_order:
        raise ReproductionEvidenceError("historical folder token order drift")
    historical_tokens = str(candidate_mapping.get("historical_folder_id", "")).split("_")
    if len(historical_tokens) != len(canonical_order) or any(
        not token.isdigit() or int(token) <= 0 for token in historical_tokens
    ):
        raise ReproductionEvidenceError("historical folder ID is not five positive counts")
    mapped = {
        element: int(token) for element, token in zip(canonical_order, historical_tokens)
    }
    if (
        mapped != candidate_mapping.get("parsed_composition")
        or mapped != top_candidate["composition"]
    ):
        raise ReproductionEvidenceError("historical folder/canonical candidate mapping drift")

    screening_metric = top_link.get("screening_metric", {})
    if (
        screening_metric.get("name") != metric["name"]
        or screening_metric.get("unit") != metric["unit"]
        or screening_metric.get("method") != "mean_msd_over_time"
    ):
        raise ReproductionEvidenceError("top-candidate linked metric definition drift")
    _assert_close(
        float(screening_metric.get("value")),
        float(top_candidate["value"]),
        "top-candidate linked screening value",
    )

    parity = top_link.get("historical_transport_parity", {})
    if parity.get("status") != "EXACT_NUMERICAL_PARITY":
        raise ReproductionEvidenceError("historical transport parity status drift")
    if parity.get("evidence_level") != "READ_ONLY_HISTORICAL_POSTPROCESS_PARITY":
        raise ReproductionEvidenceError("historical transport evidence level drift")
    if parity.get("not_high_fidelity_validation") is not True:
        raise ReproductionEvidenceError("historical parity must reject a high-fidelity claim")
    execution = parity.get("execution", {})
    for name in (
        "model_executed",
        "md_executed",
        "aimd_executed",
        "dft_executed",
        "scheduler_used",
        "network_used",
    ):
        if execution.get(name) is not False:
            raise ReproductionEvidenceError("historical parity execution flag {} drift".format(name))
    comparison = parity.get("historical_comparison", {})
    if (
        comparison.get("dataset_id") != "block-1"
        or comparison.get("historical_carrier_convention") != "same_as_reproduction"
        or comparison.get("diffusivity_all_close") is not True
        or comparison.get("conductivity_all_close") is not True
    ):
        raise ReproductionEvidenceError("historical transport comparison drift")
    arrhenius = parity.get("arrhenius", {})
    if arrhenius.get("convention") != "legacy_get_sigma_v1":
        raise ReproductionEvidenceError("historical Arrhenius convention drift")
    replayed_value = arrhenius.get("mean_msd_over_time", {}).get("conductivity_S_m")
    _assert_close(
        float(replayed_value),
        float(top_candidate["value"]),
        "top-candidate historical transport parity",
    )
    match = parity.get("screening_value_match", {})
    if match.get("status") != "EXACT_NUMERICAL_PARITY":
        raise ReproductionEvidenceError("screening/postprocess value parity status drift")
    _assert_close(
        float(match.get("screening_value_S_m")),
        float(match.get("replayed_mean_msd_value_S_m")),
        "screening/postprocess value match",
    )
    parity_sources = parity.get("source_artifacts", [])
    if len(parity_sources) != 4:
        raise ReproductionEvidenceError("historical transport source provenance is incomplete")
    for item in parity_sources:
        if not isinstance(item.get("locator"), str) or item.get("read_only") is not True:
            raise ReproductionEvidenceError("historical transport sources must be read-only")
    parity_implementation = parity.get("implementation_provenance", {})
    if (
        parity_implementation.get("wrapper")
        != "ionic-transport.adapter.reproduce_manuscript_transport"
        or parity_implementation.get("wrapper_version") != "1.0"
        or parity_implementation.get("wrapper_locator")
        != "plugins/ionic-transport/adapter.py"
    ):
        raise ReproductionEvidenceError("historical transport wrapper provenance drift")
    _repository_file(
        repository_root,
        parity_implementation.get("wrapper_locator"),
        "historical transport wrapper",
    )
    historical_scripts = parity_implementation.get("historical_scripts", [])
    expected_script_locators = (
        "historical-transport/code/get_MSD_Li10.py",
        "historical-transport/code/get_sigma.py",
    )
    if tuple(item.get("locator") for item in historical_scripts) != expected_script_locators:
        raise ReproductionEvidenceError("historical transport script provenance drift")

    high_fidelity = top_link.get("high_fidelity_validation", {})
    if high_fidelity.get("status") != "EXTERNAL_VALIDATION_PENDING":
        raise ReproductionEvidenceError("top-candidate high-fidelity status must remain pending")
    if high_fidelity.get("reference_artifact") is not None:
        raise ReproductionEvidenceError("top-candidate high-fidelity reference was not supplied")
    if high_fidelity.get("numerical_parity") != "NOT_TESTABLE_WITH_AVAILABLE_DATA":
        raise ReproductionEvidenceError("top-candidate high-fidelity parity is overstated")
    if len(high_fidelity.get("blockers", [])) < 2:
        raise ReproductionEvidenceError("top-candidate high-fidelity blockers are incomplete")

    if unseen.get("status") != "EXTERNAL_VALIDATION_PENDING":
        raise ReproductionEvidenceError("unseen/transfer validation must remain pending")
    if unseen.get("evidence_level") != "CLAIM_LEVEL_DOCUMENT_EVIDENCE":
        raise ReproductionEvidenceError("unseen/transfer evidence level is overstated")
    main_source = evidence_context["source_documents"]["manuscript-main"]
    unseen_source = unseen.get("source_document", {})
    if (
        unseen_source.get("id") != "manuscript-main"
        or unseen_source.get("basename") != main_source["basename"]
        or unseen_source.get("read_only") is not True
    ):
        raise ReproductionEvidenceError("unseen/transfer source-document pin drift")
    claims = unseen.get("document_claims", {})
    unseen_system = claims.get("unseen_system", {})
    transfer = claims.get("composition_transfer", {})
    if (
        unseen_system.get("reported_formula") != "Li24M12(PS4)16"
        or unseen_system.get("expanded_formula") != "Li6M3(PS4)4"
        or unseen_system.get("reported_aimd_configuration_count") != 787
    ):
        raise ReproductionEvidenceError("unseen-system claim transcription drift")
    if (
        transfer.get("reported_comparison_count") != 3
        or transfer.get("composition_ids_available_in_supplied_tables") is not False
        or transfer.get("per_composition_values_available_in_supplied_tables") is not False
    ):
        raise ReproductionEvidenceError("composition-transfer claim boundary drift")
    numerical_parity = unseen.get("numerical_parity", {})
    if (
        numerical_parity.get("status") != "NOT_TESTABLE_WITH_AVAILABLE_DATA"
        or numerical_parity.get("performed") is not False
    ):
        raise ReproductionEvidenceError("unseen/transfer numerical parity is overstated")
    if len(unseen.get("blockers", [])) < 4:
        raise ReproductionEvidenceError("unseen/transfer blockers are incomplete")

    return {
        "large_supercell_screening": screening,
        "top_candidate_validation": {
            "status": top_link["status"],
            "candidate_mapping": candidate_mapping,
            "screening_metric": screening_metric,
            "historical_transport_parity": parity,
            "high_fidelity_validation": high_fidelity,
        },
        "unseen_transfer_validation": unseen,
    }


def _load_benchmark_wrapper(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("mlipflow_manuscript_benchmark_wrapper", path)
    if spec is None or spec.loader is None:
        raise ReproductionEvidenceError("cannot load benchmark wrapper: {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_to_temporary(root: Path, wrapper_path: Path, temporary: Path) -> Dict[str, Path]:
    wrapper = _load_benchmark_wrapper(wrapper_path)
    outputs = wrapper.normalize_benchmark(
        [
            {
                "path": root / BENCHMARK_INPUT,
                "evidence_locator": BENCHMARK_INPUT,
            }
        ],
        temporary / "benchmark",
        mode="replay",
    )
    return {name: Path(path) for name, path in outputs.items()}


def _verify_registry_matches_records(root: Path, normalized_metrics: Mapping[str, Any]) -> None:
    registry = _json(root / "model_registry.yaml")
    records = normalized_metrics["records"]
    normalized = {
        (item["model"], item["task"], item["metric"]): item for item in records
    }
    for model in registry["models"]:
        if model.get("recommended_tasks"):
            raise ReproductionEvidenceError(
                "recommended_tasks must stay empty; the example may not encode winners"
            )
        for benchmark in model["benchmarks"]:
            task = benchmark["task"]
            for metric, value in benchmark["metrics"].items():
                item = normalized[(model["id"], task, metric)]
                _assert_close(float(value), float(item["value"]), "registry/{}/{}/{}".format(model["id"], task, metric))
                if benchmark["units"][metric] != item["unit"]:
                    raise ReproductionEvidenceError("registry metric unit drift")


def _route_models(root: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from mlipflow.config import load_project
    from mlipflow.services import query_route

    project = load_project(root)
    registry = _json(root / "model_registry.yaml")
    models = registry["models"]
    elements = set(models[0]["elements"])
    if any(set(model["elements"]) != elements for model in models):
        raise ReproductionEvidenceError("model element coverage differs inside the manuscript registry")
    policies = project.raw["routing"]["policies"]
    routes = {
        task: query_route(
            project,
            task=task,
            elements=elements,
            scenario=SCENARIO,
        )
        for task in sorted(policies)
    }
    return routes, project.raw


def _source_chain(
    task: str,
    selected_model: str,
    normalized_records: Iterable[Mapping[str, Any]],
    evidence_context: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    files = sorted(
        {
            item.get("dimensions", {}).get("source_file")
            for item in normalized_records
            if item["task"] == task and item["model"] == selected_model
        }
        - {None}
    )
    result = []
    for relative in files:
        file_record = evidence_context["evidence_files"][relative]
        source_document = evidence_context["source_documents"][file_record["source_document_id"]]
        entry = {
            "evidence_file": "evidence/" + relative,
            "source_location": file_record["source_location"],
            "source_document": source_document["basename"],
            "transcription": file_record["transformation"],
        }
        unit_source_id = file_record.get("unit_source_document_id")
        if unit_source_id:
            unit_source = evidence_context["source_documents"][unit_source_id]
            entry["unit_source_document"] = unit_source["basename"]
        result.append(entry)
    return result


def _compact_route(
    task: str,
    route: Mapping[str, Any],
    project_raw: Mapping[str, Any],
    normalized_records: Sequence[Mapping[str, Any]],
    evidence_context: Mapping[str, Any],
) -> Dict[str, Any]:
    selected = route["selected_model"]
    selected_candidate = next(item for item in route["ranking"] if item["model_id"] == selected)
    units = {
        item["metric"]: item["unit"]
        for item in normalized_records
        if item["task"] == task and item["model"] == selected
    }
    return {
        "selected_model": selected,
        "scenario": route["scenario"],
        "policy": project_raw["routing"]["policies"][task],
        "selected_metrics": selected_candidate["metrics"],
        "selected_metric_units": {
            name: units[name] for name in selected_candidate["metrics"]
        },
        "ranking": [
            {
                "rank": index,
                "model": item["model_id"],
                "score": item["score"],
                "metrics": item["metrics"],
                "contributions": item["contributions"],
            }
            for index, item in enumerate(route["ranking"], start=1)
        ],
        "decision_provenance": {
            "prepared_benchmark_input": BENCHMARK_INPUT,
            "source_chain": _source_chain(
                task, selected, normalized_records, evidence_context
            ),
        },
    }


def _build_summary(
    root: Path,
    repository_root: Path,
    evidence_context: Dict[str, Any],
    reduction_context: Mapping[str, Any],
    screening_context: Mapping[str, Any],
    normalized_outputs: Mapping[str, Path],
    routes: Mapping[str, Any],
    project_raw: Mapping[str, Any],
) -> Dict[str, Any]:
    normalized_metrics = _json(normalized_outputs["metrics.json"])
    normalized_provenance = _json(normalized_outputs["provenance.json"])
    evidence_context = dict(evidence_context)
    evidence_context["root"] = str(root)
    compact_routes = {
        task: _compact_route(
            task,
            route,
            project_raw,
            normalized_metrics["records"],
            evidence_context,
        )
        for task, route in sorted(routes.items())
    }
    source_documents = [
        {
            "id": item["id"],
            "basename": item["basename"],
            "included_in_example": False,
            "read_only": True,
        }
        for item in evidence_context["manifest"]["source_documents"]
    ]
    artifact_records = [
        {"path": "benchmark/" + name}
        for name in sorted(normalized_outputs)
    ]
    benchmark_implementation_files = [
        {"locator": locator}
        for locator in (
            BENCHMARK_WRAPPER_LOCATOR,
            BENCHMARK_NORMALIZATION_LOCATOR,
            MODEL_RUNTIME_LOCATOR,
            BENCHMARK_ADAPTER_LOCATOR,
        )
    ]
    static = compact_routes["static-pes"]
    transport = compact_routes["ionic-transport"]
    voltage = compact_routes["electrochemical-voltage"]
    return {
        "schema_version": 1,
        "report_id": "high-entropy-sulfide-manuscript-reproduction-v1",
        "mode": "evidence-replay",
        "scientific_status": "REPLAY_VERIFIED",
        "calculation_claim": "No MLIP, MD, AIMD, DFT, training, or structure generation was executed.",
        "source_documents": source_documents,
        "evidence_integrity": {
            "transcription_manifest": "evidence/transcription_provenance.json",
            "verified_evidence_file_count": len(evidence_context["evidence_files"]),
            "prepared_benchmark_record_count": reduction_context["record_count"],
            "all_evidence_paths_present": True,
            "all_transparent_reductions_recomputed": True,
        },
        "benchmark_normalization": {
            "implementation": BENCHMARK_WRAPPER_LOCATOR,
            "implementation_files": benchmark_implementation_files,
            "wrapper_version": normalized_provenance["wrapper_version"],
            "mode": normalized_provenance["mode"],
            "model_execution": normalized_provenance["model_execution"],
            "network_access": normalized_provenance["network_access"],
            "input": normalized_provenance["source_evidence"],
            "artifacts": artifact_records,
        },
        "registered_models": reduction_context["models"],
        "routing": compact_routes,
        "key_numerical_reconstruction": {
            "static_pes": {
                "selected_model": static["selected_model"],
                "test_energy_rmse": static["selected_metrics"]["energy_rmse"],
                "test_energy_rmse_unit": static["selected_metric_units"]["energy_rmse"],
                "test_force_rmse": static["selected_metrics"]["force_rmse"],
                "test_force_rmse_unit": static["selected_metric_units"]["force_rmse"],
                "source": "SI table captioned Table 3",
            },
            "ionic_transport": {
                "selected_model": transport["selected_model"],
                "conductivity_mae_against_aimd": transport["selected_metrics"]["conductivity_mae"],
                "unit": transport["selected_metric_units"]["conductivity_mae"],
                "paired_compositions": 10,
                "source": "SI Table S3; unit cross-checked against main Figure 3 y-axis",
            },
            "electrochemical_voltage": {
                "selected_model": voltage["selected_model"],
                "voltage_mae_against_dft": voltage["selected_metrics"]["voltage_mae"],
                "unit": voltage["selected_metric_units"]["voltage_mae"],
                "paired_voltage_stages": 18,
                "source": "SI Table S11; unit cross-checked against SI Figure S11 y-axis",
            },
            "simulation_efficiency": reduction_context["speedups"],
        },
        "large_supercell_screening": screening_context["large_supercell_screening"],
        "top_candidate_validation": screening_context["top_candidate_validation"],
        "unseen_transfer_validation": screening_context["unseen_transfer_validation"],
        "reconstructed_conclusions": {
            "static_and_transport_rankings_are_not_assumed_identical": (
                static["selected_model"] != transport["selected_model"]
            ),
            "transport_and_voltage_rankings_are_not_assumed_identical": (
                transport["selected_model"] != voltage["selected_model"]
            ),
            "statements": [
                "{} is selected for static PES by the two equally weighted test RMSE metrics.".format(
                    static["selected_model"]
                ),
                "{} is selected for ionic transport by the lowest Table-S3 conductivity MAE against AIMD.".format(
                    transport["selected_model"]
                ),
                "{} is selected for voltage by the lowest Table-S11 voltage MAE against DFT.".format(
                    voltage["selected_model"]
                ),
                "Tables S8-S10 reconstruct acceleration ratios of {:.3f}, {:.3f}, and {:.3f} for prototypes I, II, and III.".format(
                    *[
                        item["recomputed_acceleration_ratio"]
                        for item in reduction_context["speedups"]
                    ]
                ),
            ],
        },
        "scope_boundaries": {
            "fresh_model_prediction": "NOT_PERFORMED",
            "production_md_aimd_or_dft": "NOT_PERFORMED",
            "model_training": "NOT_PERFORMED",
            "large_supercell_screening": "REPLAY_VERIFIED",
            "top_candidate_historical_transport": "EXACT_NUMERICAL_PARITY",
            "top_candidate_aimd_dft_high_fidelity": "EXTERNAL_VALIDATION_PENDING",
            "unseen_transfer_evidence_level": "CLAIM_LEVEL_DOCUMENT_EVIDENCE",
            "unseen_transfer_validation": "EXTERNAL_VALIDATION_PENDING",
            "unseen_transfer_numerical_parity": "NOT_TESTABLE_WITH_AVAILABLE_DATA",
            "claim_boundary": "This report reconstructs evidence, reductions, and routing decisions; it does not independently validate the underlying model predictions."
        },
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Manuscript reproduction summary",
        "",
        "Status: **{}** (`{}`).".format(summary["scientific_status"], summary["mode"]),
        "",
        summary["calculation_claim"],
        "",
        "## Source documents",
        "",
        "| Document | Included |",
        "| --- | --- |",
    ]
    for item in summary["source_documents"]:
        lines.append(
            "| {} | no; read-only external source |".format(item["basename"])
        )
    lines.extend(
        [
            "",
            "## Evidence-driven routes",
            "",
            "| Task | Selected model | Selected evidence metrics |",
            "| --- | --- | --- |",
        ]
    )
    for task, route in summary["routing"].items():
        metrics = ", ".join(
            "{}={} {}".format(name, value, route["selected_metric_units"][name])
            for name, value in route["selected_metrics"].items()
        )
        lines.append("| {} | `{}` | {} |".format(task, route["selected_model"], metrics))
    lines.extend(
        [
            "",
            "The registry contains no `recommended_tasks` hints; each route is obtained from the registered metrics and the policy recorded in `project.yaml`.",
            "",
            "## Recomputed timing evidence",
            "",
            "| Prototype | DeePMD mean (h) | AIMD mean (h) | Exact AIMD/DeePMD ratio | Manuscript approximation |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in summary["key_numerical_reconstruction"]["simulation_efficiency"]:
        lines.append(
            "| {} | {:.6f} | {:.6f} | {:.6f} | {} |".format(
                item["prototype"],
                item["deepmd_mean_hours_per_composition"],
                item["aimd_mean_hours_per_composition"],
                item["recomputed_acceleration_ratio"],
                item["published_approximate_ratio"],
            )
        )
    screening = summary["large_supercell_screening"]
    screening_inputs = screening["input_set"]
    lines.extend(
        [
            "",
            "## Large-supercell screening replay",
            "",
            "Status: **{}**. Candidate/evaluated count: **{}/{}**; excluded for missing metrics: **{}**.".format(
                screening["status"],
                screening_inputs["candidate_count"],
                screening_inputs["evaluated_count"],
                screening_inputs["excluded_missing_count"],
            ),
            "",
            "Metric: `{}` ({}; direction `{}`). Candidate and transport records: `{}` and `{}`.".format(
                screening["metric"]["name"],
                screening["metric"]["unit"],
                screening["rule"]["direction"],
                screening["source_artifacts"]["candidate_manifest"]["locator"],
                screening["source_artifacts"]["transport_results_manifest"]["locator"],
            ),
            "",
            "| Rank | Candidate | Composition (Mn/Fe/Ni/Cu/Zn) | Conductivity (S/m) |",
            "| ---: | --- | --- | ---: |",
        ]
    )
    order = screening_inputs["canonical_element_order"]
    for item in screening["top_candidates"]:
        composition = "/".join(str(item["composition"][element]) for element in order)
        lines.append(
            "| {} | `{}` | {} | {} |".format(
                item["rank"], item["candidate_id"], composition, item["value"]
            )
        )

    top_validation = summary["top_candidate_validation"]
    candidate_mapping = top_validation["candidate_mapping"]
    historical = top_validation["historical_transport_parity"]
    high_fidelity = top_validation["high_fidelity_validation"]
    lines.extend(
        [
            "",
            "## Top-candidate validation status",
            "",
            "`{}` maps exactly to historical folder ID `{}` under the declared Mn/Fe/Ni/Cu/Zn token order.".format(
                candidate_mapping["ranked_candidate_id"],
                candidate_mapping["historical_folder_id"],
            ),
            "",
            "Historical transport status: **{}** (`{}`). The exact match is limited to read-only replay of existing target.msd/stdout post-processing; it is **not** AIMD/DFT high-fidelity validation.".format(
                historical["status"], historical["evidence_level"]
            ),
            "",
            "AIMD/DFT high-fidelity status: **{}**; numerical parity: **{}**.".format(
                high_fidelity["status"], high_fidelity["numerical_parity"]
            ),
        ]
    )

    unseen = summary["unseen_transfer_validation"]
    lines.extend(
        [
            "",
            "## Unseen/transfer claim representation",
            "",
            "Evidence level: **{}**; status: **{}**; numerical parity: **{}**.".format(
                unseen["evidence_level"],
                unseen["status"],
                unseen["numerical_parity"]["status"],
            ),
            "",
            "Recorded source: `{}`. The document reports 787 AIMD configurations for Li24M12(PS4)16 and three qualitative conductivity comparisons, but supplies no paired values needed for numerical parity.".format(
                unseen["source_document"]["basename"]
            ),
            "",
            "Blocking evidence:",
            "",
        ]
    )
    lines.extend("- " + item for item in unseen["blockers"])
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            summary["scope_boundaries"]["claim_boundary"],
            "The 247-candidate screening and historical top-candidate post-processing are replay-verified. Top-candidate AIMD/DFT validation remains external and pending; unseen/transfer evidence remains claim-level and cannot support numerical parity with the available data.",
            "",
        ]
    )
    return "\n".join(lines)


def reproduce(
    root: Path,
    *,
    benchmark_wrapper_path: Optional[Path] = None,
    write: bool = False,
) -> Dict[str, Any]:
    """Verify or regenerate the compact evidence-replay artifacts."""

    root = Path(root).resolve()
    if not (root / "project.yaml").is_file():
        raise ReproductionEvidenceError("not a reproduction example root: {}".format(root))
    if benchmark_wrapper_path is None:
        benchmark_wrapper_path = (
            root.parents[1] / "plugins" / "mlip-benchmark" / "benchmark_wrapper.py"
        )
    benchmark_wrapper_path = Path(benchmark_wrapper_path).resolve()
    repository_root = _repository_root_from_benchmark_wrapper(benchmark_wrapper_path)
    evidence_context = _verify_transcription_files(root)
    reduction_context = _verify_prepared_benchmark_reductions(root)
    screening_context = _verify_screening_evidence(
        root, evidence_context, repository_root
    )

    with tempfile.TemporaryDirectory(prefix="mlipflow-manuscript-replay-") as temporary_name:
        temporary = Path(temporary_name)
        normalized_outputs = _normalize_to_temporary(
            root, benchmark_wrapper_path, temporary
        )
        normalized_metrics = _json(normalized_outputs["metrics.json"])
        _verify_registry_matches_records(root, normalized_metrics)
        routes, project_raw = _route_models(root)
        summary = _build_summary(
            root,
            repository_root,
            evidence_context,
            reduction_context,
            screening_context,
            normalized_outputs,
            routes,
            project_raw,
        )
        expected: Dict[Path, bytes] = {
            root / "benchmark" / name: path.read_bytes()
            for name, path in normalized_outputs.items()
        }
        expected[root / "reports" / REPORT_OUTPUTS[0]] = _json_bytes(summary)
        expected[root / "reports" / REPORT_OUTPUTS[1]] = _markdown(summary).encode("utf-8")

        if write:
            for path, payload in expected.items():
                _write_atomic(path, payload)
        else:
            for path, payload in expected.items():
                if not path.is_file():
                    raise ReproductionEvidenceError("missing generated artifact: {}".format(path))
                if path.read_bytes() != payload:
                    raise ReproductionEvidenceError(
                        "generated artifact drift; run reproduce.py --write: {}".format(path)
                    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify manuscript evidence replay without running a model."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="reproduction example root",
    )
    parser.add_argument(
        "--benchmark-wrapper",
        type=Path,
        help="path to plugins/mlip-benchmark/benchmark_wrapper.py",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify committed outputs (default)")
    mode.add_argument("--write", action="store_true", help="regenerate normalized outputs and reports")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    summary = reproduce(
        arguments.root,
        benchmark_wrapper_path=arguments.benchmark_wrapper,
        write=arguments.write,
    )
    print(
        json.dumps(
            {
                "status": summary["scientific_status"],
                "mode": summary["mode"],
                "selected_models": {
                    task: item["selected_model"]
                    for task, item in summary["routing"].items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
