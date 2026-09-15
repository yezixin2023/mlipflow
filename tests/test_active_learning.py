"""Deterministic Strategy A/B and round-decision tests for active learning."""

from __future__ import annotations

from tests.helpers import load_module
import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from mlipflow.plugins.active_learning import science as al
from mlipflow.config import load_project
from mlipflow.services.commands import (
    advance,
    initialize,
    make_advance_plan,
    make_run_plan,
    run_node,
)
from mlipflow.services.queries import query_workflow


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "mlipflow" / "plugins" / "active_learning" / "adapter.py"


def load_adapter():
    module = load_module(ADAPTER_PATH, 'test_active_learning_adapter')
    return module.Adapter()


def target_domain():
    return {
        "composition": "Li-test",
        "structure_families": ["solid-test"],
        "defect_states": ["stoichiometric"],
        "temperature_range_k": [400, 800],
        "pressure_range_gpa": [0, 0],
        "ensembles": ["NVT"],
        "supercell_range": [1, 1],
        "include_lasp_ssw_high_energy": False,
        "allow_phase_change": False,
        "allow_melting": False,
        "allow_decomposition": False,
        "mobile_species": ["Li"],
        "intended_use": "MD",
    }


def policy(strategy="dual", consecutive=2, maximum_total=20):
    if strategy == "single":
        strategy_record = {"mode": al.STRATEGY_SINGLE, "primary_model": "model-a"}
        models = ["model-a"]
    else:
        strategy_record = {
            "mode": al.STRATEGY_DUAL,
            "models": ["model-a", "model-b"],
        }
        models = ["model-a", "model-b"]
    audit_thresholds = []
    for model in models:
        for metric, unit in (
            ("energy_mae", "eV/atom"),
            ("energy_rmse", "eV/atom"),
            ("force_mae", "eV/angstrom"),
            ("force_rmse", "eV/angstrom"),
            ("maximum_atomic_force_error", "eV/angstrom"),
        ):
            audit_thresholds.append(
                {"model": model, "metric": metric, "maximum": 0.25, "unit": unit}
            )
    return {
        "schema_version": 1,
        "contract": al.POLICY_CONTRACT,
        "policy_id": f"oracle-{strategy}-v1",
        "strategy": strategy_record,
        "target_domain": target_domain(),
        "target_conditions": [
            {"id": "400k", "temperature_k": 400, "ensemble": "NVT"},
            {"id": "800k", "temperature_k": 800, "ensemble": "NVT"},
        ],
        "committee": {
            "models": [
                {"model_id": model, "member_seeds": [11, 29]} for model in models
            ],
            "requested_coverage": 0.75,
            "minimum_observed_coverage": 0.75,
            "epsilon": 1e-9,
            "minimum_calibration_samples": 4,
            "maximum_calibration_false_negative_rate": 0.0,
        },
        "thresholds": {
            "force": {
                "unit": "eV/angstrom",
                "label_threshold": 0.25,
                "abort_threshold": 0.6,
            }
        },
        "selection": {
            "time_stride": 1,
            "maximum_labels_per_round": 3,
            "maximum_total_labels": maximum_total,
            "safe_spot_checks_per_round": 1,
            "safe_spot_check_seed": 47,
            "include_safe_spot_checks_in_training": False,
            "per_condition_quota": {"400k": 1, "800k": 1},
            "per_replica_quota": {"r0": 2, "r1": 2},
        },
        "assessment": {
            "audit_thresholds": audit_thresholds,
            "condition_gates": {
                "400k": {"maximum_query_fraction": 1.0, "maximum_unsafe_fraction": 1.0},
                "800k": {"maximum_query_fraction": 1.0, "maximum_unsafe_fraction": 1.0},
            },
            "maximum_spot_check_false_negative_rate": 0.0,
            "required_consecutive_rounds": consecutive,
        },
    }


def prediction_record(
    sample_id,
    split,
    disagreement,
    *,
    condition_id=None,
    replica="r0",
    frame_index=0,
    structure_path=None,
    near_group=None,
    valid=True,
    severe=False,
):
    record = {
        "sample_id": sample_id,
        "split": split,
        "atom_count": 1,
        "energy": 0.0,
        "forces": [[0.0, 0.0, 0.0]],
    }
    if split == "candidate":
        record.update(
            {
                "condition_id": condition_id,
                "condition": {
                    "temperature_k": 400 if condition_id == "400k" else 800,
                    "ensemble": "NVT",
                },
                "replica": replica,
                "frame_index": frame_index,
                "structure_path": structure_path or f"structures/{sample_id}.vasp",
                "near_duplicate_group": near_group or f"near-{sample_id}",
                "physical_validity": {
                    "valid": valid,
                    "severe": severe,
                    "reasons": [] if valid else ["fixture-invalid"],
                },
            }
        )
    return record


def prediction_bundle(strategy="dual"):
    selected_policy = policy(strategy)
    model_ids = (
        ["model-a"] if strategy == "single" else ["model-a", "model-b"]
    )
    calibration = {f"cal-{index}": value for index, value in enumerate((0.1, 0.2, 0.3, 0.4), 1)}
    candidate_disagreements = {
        "model-a": {
            "safe": 0.1,
            "union": 0.2,
            "dup": 0.35,
            "query2": 0.3,
            "unsafe": 0.7,
            "severe": 0.2,
        },
        "model-b": {
            "safe": 0.1,
            "union": 0.3,
            "dup": 0.35,
            "query2": 0.3,
            "unsafe": 0.2,
            "severe": 0.2,
        },
    }
    metadata = {
        "safe": ("400k", "r0", 0, "fp-safe", "near-safe", True, False),
        "union": ("400k", "r0", 1, "exact-shared", "near-shared", True, False),
        "dup": ("400k", "r0", 2, "exact-shared", "near-shared", True, False),
        "unsafe": ("400k", "r1", 3, "fp-unsafe", "near-unsafe", True, False),
        "severe": ("400k", "r1", 4, "fp-severe", "near-severe", False, True),
        "query2": ("800k", "r1", 0, "fp-query2", "near-query2", True, False),
    }
    models = []
    for model_id in model_ids:
        framework = "framework-a" if model_id == "model-a" else "framework-b"
        members = []
        for member_index, seed in enumerate((11, 29)):
            records = []
            for sample_id, disagreement in calibration.items():
                record = prediction_record(sample_id, "calibration", disagreement)
                record["forces"] = [
                    [0.0 if member_index == 0 else 2 * disagreement, 0.0, 0.0]
                ]
                records.append(record)
            for short_id, disagreement in candidate_disagreements[model_id].items():
                condition, replica, frame, structure_path, group, valid, severe = metadata[short_id]
                record = prediction_record(
                    f"cand-{short_id}",
                    "candidate",
                    disagreement,
                    condition_id=condition,
                    replica=replica,
                    frame_index=frame,
                    structure_path=structure_path,
                    near_group=group,
                    valid=valid,
                    severe=severe,
                )
                record["forces"] = [
                    [0.0 if member_index == 0 else 2 * disagreement, 0.0, 0.0]
                ]
                records.append(record)
            members.append(
                {
                    "member_id": f"{model_id}-seed-{seed}",
                    "seed": seed,
                    "predictions": records,
                }
            )
        models.append(
            {
                "model_id": model_id,
                "framework": framework,
                "supports_stress": model_id == "model-a",
                "members": members,
            }
        )
    return {
        "schema_version": 1,
        "contract": al.PREDICTION_CONTRACT,
        "validation_claim": "ORACLE_REPLAY_VALIDATION",
        "strategy": selected_policy["strategy"],
        "units": {"energy": "eV", "force": "eV/angstrom"},
        "dataset_split": {
            "dataset_id": "oracle-dataset",
            "split_id": "immutable-split-v1",
            "train_ids": ["train-1"],
            "validation_ids": ["validation-1"],
            "calibration_ids": list(calibration),
            "audit_ids": ["audit-1", "audit-2"],
        },
        "calibration_labels": [
            {"sample_id": sample_id, "forces": [[0.0, 0.0, 0.0]]}
            for sample_id in calibration
        ],
        "models": models,
    }


def candidate_manifest(evaluation):
    return {
        "schema_version": 1,
        "contract": al.CANDIDATE_CONTRACT,
        "candidates": [
            {
                "id": item["sample_id"],
                "structure_path": item["structure_path"],
            }
            for item in evaluation["candidates"]
        ],
    }


def selection_fixture(evaluation, selected_policy):
    direct = {
        "schema_version": 1,
        "method": "DIRECT",
        "input_candidate_ids": ["cand-dup", "cand-query2"],
        "selected_candidate_ids": ["cand-dup", "cand-query2"],
        "parameters": {"n_clusters": 2},
    }
    return al.select_candidates(
        evaluation, candidate_manifest(evaluation), selected_policy, direct
    )


def audit_fixture(models, force_rmse=0.10):
    records = []
    for model in models:
        for metric, value, unit in (
            ("energy_mae", 0.05, "eV/atom"),
            ("energy_rmse", 0.06, "eV/atom"),
            ("force_mae", 0.08, "eV/angstrom"),
            ("force_rmse", force_rmse, "eV/angstrom"),
            ("maximum_atomic_force_error", 0.20, "eV/angstrom"),
        ):
            records.append(
                {
                    "model": model,
                    "metric": metric,
                    "value": value,
                    "unit": unit,
                    "split": "audit",
                }
            )
    return {
        "schema_version": 1,
        "audit_ids": ["audit-1", "audit-2"],
        "records": records,
    }


def assessment_inputs(consecutive=2, cumulative=7):
    selected_policy = policy("dual", consecutive=consecutive)
    bundle = prediction_bundle("dual")
    evaluation = al.evaluate_committee(bundle, selected_policy)
    selection = selection_fixture(evaluation, selected_policy)
    models = ["model-a", "model-b"]
    by_id = {item["sample_id"]: item for item in evaluation["candidates"]}
    spot_ids = [item["sample_id"] for item in selection["selected_safe_spot_checks"]]
    spot_records = [
        {"sample_id": sample_id, "record_id": f"spot-record-{sample_id}"}
        for sample_id in spot_ids
    ]
    query_ids = [item["sample_id"] for item in selection["selected_query_candidates"]]
    query_records = [
        {"sample_id": sample_id, "record_id": f"query-record-{sample_id}"}
        for sample_id in query_ids
    ]
    current_split = {
        "dataset_id": "oracle-cumulative-dataset",
        "split_id": "oracle-cumulative-split-v2",
        "train_ids": bundle["dataset_split"]["train_ids"]
        + [item["record_id"] for item in query_records],
        "validation_ids": bundle["dataset_split"]["validation_ids"],
        "calibration_ids": bundle["dataset_split"]["calibration_ids"],
        "audit_ids": bundle["dataset_split"]["audit_ids"],
    }
    spot_samples = []
    for sample_id in spot_ids:
        for model in models:
            spot_samples.append(
                {
                    "sample_id": sample_id,
                    "model": model,
                    "predicted_force_error": by_id[sample_id]["models"][model][
                        "estimated_force_error"
                    ],
                    "actual_force_error": 0.12,
                    "unit": "eV/angstrom",
                }
            )
    prior = {
        "round_index": 0,
        "dataset_id": "oracle-dataset",
        "split_id": "immutable-split-v1",
        "audit_ids": ["audit-1", "audit-2"],
        "coverage_passed": True,
        "accuracy_passed": True,
        "pes_gates_passed_this_round": True,
    }
    current_round = 1 if consecutive > 1 else 0
    previous_rounds = [prior] if current_round else []
    campaign = {
        "campaign_id": "oracle-campaign",
        "policy_id": selected_policy["policy_id"],
        "strategy": selected_policy["strategy"],
        "target_domain": selected_policy["target_domain"],
        "current_cumulative_dataset": {
            "dataset_id": current_split["dataset_id"],
            "split_id": current_split["split_id"],
            "reference": "canonical/current.json",
        },
        "current_decision": "PENDING",
        "rounds": [
            {"index": index, "directory": f"round-{index:03d}"}
            for index in range(current_round + 1)
        ],
        "validation_claim": "ORACLE_REPLAY_VALIDATION",
    }
    return {
        "evaluation": evaluation,
        "selection": selection,
        "audit": audit_fixture(models),
        "spot": {"schema_version": 1, "samples": spot_samples},
        "labeling": {
            "schema_version": 1,
            "status": "OK",
            "successful_query_records": query_records,
            "successful_spot_check_records": spot_records,
            "cumulative_dft_label_count": cumulative,
            "cumulative_dataset_id": current_split["dataset_id"],
            "cumulative_split_id": current_split["split_id"],
        },
        "split": current_split,
        "history": {"current_round": current_round, "rounds": previous_rounds},
        "campaign": campaign,
        "policy": selected_policy,
    }


def run_assessment(values):
    return al.assess_round(
        values["evaluation"],
        values["selection"],
        values["audit"],
        values["spot"],
        values["labeling"],
        values["split"],
        values["history"],
        values["campaign"],
        values["policy"],
    )


def test_force_disagreement_and_calibrated_error_are_recomputable():
    summary = al.force_disagreement_summary(
        [
            [[-1.0, 0.0, 0.0], [0.0, -2.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        ]
    )
    assert summary["per_atom_force_disagreement"] == [1.0, 2.0]
    assert summary["maximum_atomic_force_disagreement"] == 2.0
    assert summary["mean_atomic_force_disagreement"] == 1.5
    assert al.energy_disagreement([-2.0, 2.0], 2) == 1.0

    report = al.calibrate_force_uncertainty(
        [
            {"sample_id": "a", "disagreement": 1.0, "actual_error": 2.0},
            {"sample_id": "b", "disagreement": 2.0, "actual_error": 2.0},
        ],
        requested_coverage=1.0,
        minimum_observed_coverage=1.0,
        epsilon=1e-12,
        minimum_sample_count=2,
        label_threshold=3.0,
        maximum_false_negative_rate=0.0,
    )
    assert report["status"] == "PASS"
    assert report["scale_quantile"] == pytest.approx(2.0 / (1.0 + 1e-12))
    assert report["samples"][1]["estimated_force_error"] == pytest.approx(
        report["scale_quantile"] * 2.0
    )


def test_strategy_a_and_b_use_independent_committees_and_max_risk_union():
    evaluation_a = al.evaluate_committee(prediction_bundle("single"), policy("single"))
    by_id_a = {item["sample_id"]: item for item in evaluation_a["candidates"]}
    assert evaluation_a["evaluation_status"] == "READY"
    assert by_id_a["cand-union"]["classification"] == "SAFE"

    evaluation_b = al.evaluate_committee(prediction_bundle("dual"), policy("dual"))
    by_id_b = {item["sample_id"]: item for item in evaluation_b["candidates"]}
    union = by_id_b["cand-union"]
    assert union["models"]["model-a"]["classification"] == "SAFE"
    assert union["models"]["model-b"]["classification"] == "QUERY"
    assert union["classification"] == "QUERY"
    assert union["combined_risk"] == max(
        union["models"][model]["estimated_force_error"] for model in ("model-a", "model-b")
    )
    assert union["trigger_models"] == ["model-b"]
    assert set(evaluation_b["calibration"]) == {"model-a", "model-b"}
    assert evaluation_b["committees"]["model-a"]["supports_stress"] is True
    assert evaluation_b["committees"]["model-b"]["supports_stress"] is False
    assert "cross" not in json.dumps(evaluation_b).lower()


def test_committee_records_framework_members_and_policy_seeds():
    bundle = prediction_bundle("dual")
    evaluation = al.evaluate_committee(bundle, policy("dual"))
    assert evaluation["committees"]["model-a"] == {
        "framework": "framework-a",
        "supports_stress": True,
        "member_count": 2,
        "members": [
            {"member_id": "model-a-seed-11", "seed": 11},
            {"member_id": "model-a-seed-29", "seed": 29},
        ],
    }
    bundle["models"][1]["members"][1]["seed"] = 31
    with pytest.raises(al.ActiveLearningError, match="member seeds differ from policy"):
        al.evaluate_committee(bundle, policy("dual"))


def test_assessment_requires_common_audit_metrics_but_not_marginal_gain():
    model_only_stress = assessment_inputs(consecutive=1)
    model_only_stress["policy"]["assessment"]["audit_thresholds"].append(
        {
            "model": "model-a",
            "metric": "stress_rmse",
            "maximum": 0.1,
            "unit": "eV/angstrom^3",
        }
    )
    with pytest.raises(al.ActiveLearningError, match="common energy/force metrics"):
        run_assessment(model_only_stress)

    no_marginal_gain = assessment_inputs(consecutive=1)
    result = run_assessment(no_marginal_gain)
    assert result["decision"] == "CONVERGED_FOR_DECLARED_DOMAIN"
    assert result["coverage_passed"] is True
    assert result["accuracy_passed"] is True
    assert "marginal_gain" not in result


def test_target_condition_metadata_and_replica_statistics_are_explicit():
    selected_policy = policy("dual")
    evaluation = al.evaluate_committee(prediction_bundle("dual"), selected_policy)
    condition = next(
        item for item in evaluation["per_condition"] if item["condition_id"] == "400k"
    )
    assert condition["condition"] == {"temperature_k": 400, "ensemble": "NVT"}
    replicas = {item["replica"]: item for item in condition["replicas"]}
    assert replicas["r0"]["candidate_count"] == 3
    assert replicas["r1"]["candidate_count"] == 2
    assert set(replicas["r0"]["models"]) == {"model-a", "model-b"}

    drift = prediction_bundle("dual")
    for model in drift["models"]:
        for member in model["members"]:
            for record in member["predictions"]:
                if record["sample_id"] == "cand-safe":
                    record["condition"]["temperature_k"] = 999
    with pytest.raises(al.ActiveLearningError, match="condition metadata differs"):
        al.evaluate_committee(drift, selected_policy)


def test_calibration_block_and_split_leakage_are_not_hidden():
    selected_policy = policy("dual")
    selected_policy["committee"]["minimum_calibration_samples"] = 5
    blocked = al.evaluate_committee(prediction_bundle("dual"), selected_policy)
    assert blocked["evaluation_status"] == "BLOCKED_CALIBRATION"
    assert all(item["classification"] is None for item in blocked["candidates"])

    leaked = prediction_bundle("dual")
    leaked["dataset_split"]["train_ids"].append("cal-1")
    with pytest.raises(al.ActiveLearningError, match="appears in both"):
        al.evaluate_committee(leaked, policy("dual"))


def test_unsafe_frames_record_the_nearest_prior_safe_boundary():
    bundle = prediction_bundle("dual")
    for model in bundle["models"]:
        for member in model["members"]:
            for record in member["predictions"]:
                if record["sample_id"] == "cand-unsafe":
                    record["replica"] = "r0"
    evaluation = al.evaluate_committee(bundle, policy("dual"))
    unsafe = next(
        item for item in evaluation["candidates"] if item["sample_id"] == "cand-unsafe"
    )
    safe = next(
        item for item in evaluation["candidates"] if item["sample_id"] == "cand-safe"
    )
    assert unsafe["classification"] == "UNSAFE"
    boundary = unsafe["nearest_prior_safe_boundary"]
    assert boundary["sample_id"] == "cand-safe"
    assert boundary["frame_index"] == 0
    assert boundary["combined_risk"] == safe["combined_risk"]


def test_selection_is_deterministic_uses_direct_and_excludes_severe_structures():
    selected_policy = policy("dual")
    evaluation = al.evaluate_committee(prediction_bundle("dual"), selected_policy)
    selection = selection_fixture(evaluation, selected_policy)
    assert selection["direct_input_candidate_ids"] == ["cand-dup", "cand-query2"]
    assert [item["sample_id"] for item in selection["selected_query_candidates"]] == [
        "cand-dup",
        "cand-query2",
    ]
    assert len(selection["selected_safe_spot_checks"]) == 1
    assert "cand-severe" in selection["severe_invalid_not_sent_to_dft"]
    assert "cand-severe" not in {
        item["sample_id"] for item in selection["selected_for_dft"]
    }
    assert selection == selection_fixture(evaluation, selected_policy)

    with pytest.raises(al.ActiveLearningError, match="DIRECT"):
        al.select_candidates(
            evaluation, candidate_manifest(evaluation), selected_policy, None
        )


def test_strategy_b_selection_covers_each_available_trigger_model_before_risk_fill():
    selected_policy = policy("dual")
    selected_policy["selection"]["per_condition_quota"]["400k"] = 3
    selected_policy["selection"]["per_replica_quota"]["r0"] = 3
    evaluation = al.evaluate_committee(prediction_bundle("dual"), selected_policy)
    by_id = {item["sample_id"]: item for item in evaluation["candidates"]}
    configured = {
        "cand-dup": (0.9, ["model-a"], "fp-a-high", "near-a-high"),
        "cand-union": (0.8, ["model-a"], "fp-a-low", "near-a-low"),
        "cand-query2": (0.3, ["model-b"], "fp-b", "near-b"),
    }
    for sample_id, (risk, triggers, structure_path, near_group) in configured.items():
        candidate = by_id[sample_id]
        candidate["classification"] = "QUERY"
        candidate["combined_risk"] = risk
        candidate["trigger_models"] = triggers
        candidate["structure_path"] = structure_path
        candidate["near_duplicate_group"] = near_group
    manifest = candidate_manifest(evaluation)
    prefiltered, _ = al._prefilter_query_candidates(
        evaluation["candidates"], al._selection_policy(selected_policy)
    )
    direct_ids = [item["sample_id"] for item in prefiltered]
    selection = al.select_candidates(
        evaluation,
        manifest,
        selected_policy,
        {
            "method": "DIRECT",
            "input_candidate_ids": direct_ids,
            "selected_candidate_ids": direct_ids,
            "parameters": {"fixture": "trigger-coverage"},
        },
    )

    assert {
        model_id
        for item in selection["selected_query_candidates"]
        for model_id in item["trigger_models"]
    } == {"model-a", "model-b"}
    assert selection["trigger_model_coverage"] == {
        "available_trigger_models": ["model-a", "model-b"],
        "selected_trigger_models": ["model-a", "model-b"],
        "missing_trigger_models": [],
        "status": "COMPLETE",
    }


def test_round_dft_budget_includes_safe_spot_checks():
    selected_policy = policy("dual")
    selected_policy["selection"]["maximum_labels_per_round"] = 2
    evaluation = al.evaluate_committee(prediction_bundle("dual"), selected_policy)
    selection = selection_fixture(evaluation, selected_policy)
    assert selection["counts"]["query_dft_selected_count"] == 1
    assert selection["counts"]["safe_spot_check_count"] == 1
    assert selection["counts"]["dft_selected_count"] == 2
    assert any(item["reason"] == "round-dft-budget" for item in selection["rejected"])

    invalid_policy = policy("dual")
    invalid_policy["selection"]["maximum_labels_per_round"] = 1
    invalid_policy["selection"]["safe_spot_checks_per_round"] = 2
    with pytest.raises(al.ActiveLearningError, match="safe_spot_checks_per_round"):
        al.select_candidates(
            evaluation,
            candidate_manifest(evaluation),
            invalid_policy,
            None,
        )


def test_policy_controls_whether_safe_spot_labels_enter_training():
    excluded = assessment_inputs(consecutive=1)
    excluded_result = run_assessment(excluded)
    assert not any(
        "cumulative dataset handoff" in reason
        for reason in excluded_result["round_evidence"]["failed_reasons"]
    )
    assert excluded_result["dft_labels"]["safe_spot_checks_in_training"] is False

    missing = assessment_inputs(consecutive=1)
    missing["policy"]["selection"]["include_safe_spot_checks_in_training"] = True
    missing["selection"]["include_safe_spot_checks_in_training"] = True
    missing_result = run_assessment(missing)
    assert missing_result["decision"] == "SCIENTIFIC_REVIEW_REQUIRED"
    assert missing_result["coverage_passed"] is True
    assert missing_result["accuracy_passed"] is True
    assert missing_result["pes_gates_passed_this_round"] is False
    assert any(
        "cumulative dataset handoff" in reason
        for reason in missing_result["round_evidence"]["failed_reasons"]
    )

    included = assessment_inputs(consecutive=1)
    included["policy"]["selection"]["include_safe_spot_checks_in_training"] = True
    included["selection"]["include_safe_spot_checks_in_training"] = True
    included["split"]["train_ids"].extend(
        item["record_id"]
        for item in included["labeling"]["successful_spot_check_records"]
    )
    included_result = run_assessment(included)
    assert not any(
        "cumulative dataset handoff" in reason
        for reason in included_result["round_evidence"]["failed_reasons"]
    )
    assert included_result["dft_labels"]["safe_spot_checks_in_training"] is True


def test_false_negative_consecutive_round_and_budget_decisions_are_distinct():
    values = assessment_inputs(consecutive=2)
    converged = run_assessment(values)
    assert converged["decision"] == "CONVERGED_FOR_DECLARED_DOMAIN"
    assert converged["pes_status"] == "CONVERGED"
    assert converged["transport_status"] == "NOT_ESTABLISHED"
    assert converged["coverage_passed"] is True
    assert converged["accuracy_passed"] is True
    assert converged["pes_gates_passed_this_round"] is True
    assert converged["passed_consecutive_rounds"] == 2

    not_consecutive = assessment_inputs(consecutive=2)
    not_consecutive["history"]["rounds"][0]["pes_gates_passed_this_round"] = False
    continued = run_assessment(not_consecutive)
    assert continued["decision"] == "CONTINUE"
    assert continued["coverage_passed"] is True
    assert continued["accuracy_passed"] is True
    assert continued["passed_consecutive_rounds"] == 1
    assert continued["consecutive_stability"]["passed"] is False

    false_negative = assessment_inputs(consecutive=2)
    false_negative["spot"]["samples"][0]["actual_force_error"] = 0.4
    failed_spot = run_assessment(false_negative)
    assert failed_spot["decision"] == "CONTINUE"
    assert failed_spot["coverage_passed"] is False
    assert failed_spot["accuracy_passed"] is True
    assert failed_spot["coverage"]["safe_spot_checks"]["false_negative_count"] == 1
    assert failed_spot["coverage"]["calibration"]["status"] == "PASS"
    assert "model-a" in failed_spot["recommended_focus"]

    exhausted = assessment_inputs(consecutive=2, cumulative=20)
    exhausted["audit"] = audit_fixture(["model-a", "model-b"], force_rmse=0.4)
    budget = run_assessment(exhausted)
    assert budget["decision"] == "BUDGET_EXHAUSTED"
    assert budget["pes_status"] == "NOT_CONVERGED"
    assert budget["coverage_passed"] is True
    assert budget["accuracy_passed"] is False

    stability_exhausted = assessment_inputs(consecutive=2, cumulative=20)
    stability_exhausted["history"]["rounds"][0][
        "pes_gates_passed_this_round"
    ] = False
    stability_budget = run_assessment(stability_exhausted)
    assert stability_budget["decision"] == "BUDGET_EXHAUSTED"
    assert stability_budget["pes_status"] == "NOT_CONVERGED"
    assert stability_budget["coverage_passed"] is True
    assert stability_budget["accuracy_passed"] is True
    assert stability_budget["passed_consecutive_rounds"] == 1


@pytest.mark.parametrize("metric", sorted(al.COMMON_AUDIT_METRICS))
def test_accuracy_requires_every_immutable_audit_metric(metric):
    values = assessment_inputs(consecutive=1)
    record = next(
        item
        for item in values["audit"]["records"]
        if item["model"] == "model-a" and item["metric"] == metric
    )
    record["value"] = 0.3

    result = run_assessment(values)

    assert result["decision"] == "CONTINUE"
    assert result["coverage_passed"] is True
    assert result["accuracy_passed"] is False
    assert result["pes_gates_passed_this_round"] is False
    assert any(metric in reason for reason in result["accuracy"]["failed_reasons"])


def test_missing_target_condition_is_blocked_sampling_and_not_coverage():
    values = assessment_inputs(consecutive=1)
    condition = next(
        item
        for item in values["evaluation"]["per_condition"]
        if item["condition_id"] == "800k"
    )
    condition["candidate_count"] = 0

    result = run_assessment(values)

    assert result["decision"] == "BLOCKED_SAMPLING"
    assert result["coverage_passed"] is False
    assert result["accuracy_passed"] is True
    assert result["pes_status"] == "NOT_CONVERGED"


def test_assessment_reports_uncertainty_distribution_change():
    values = assessment_inputs(consecutive=2)
    current = al._candidate_uncertainty_distribution(
        values["evaluation"], ["model-a", "model-b"]
    )
    previous = json.loads(json.dumps(current))
    for statistics in [
        previous["combined_risk"],
        *previous["models"].values(),
    ]:
        for key in ("mean", "q90", "maximum"):
            statistics[key] += 0.1
    values["history"]["rounds"][0]["uncertainty_distribution"] = previous

    result = run_assessment(values)
    report = result["uncertainty_distribution"]
    assert report["current"] == current
    assert report["previous"] == previous
    assert report["change"]["status"] == "AVAILABLE"
    assert report["change"]["combined_risk"]["mean_delta"] == pytest.approx(-0.1)
    assert report["change"]["models"]["model-b"]["q90_delta"] == pytest.approx(
        -0.1
    )


def test_strategy_b_condition_and_spot_check_gates_are_per_model():
    values = assessment_inputs(consecutive=2)
    values["policy"]["assessment"]["maximum_spot_check_false_negative_rate"] = 0.5
    values["spot"]["samples"][0]["actual_force_error"] = 0.4
    failed_spot = run_assessment(values)
    assert failed_spot["decision"] == "CONTINUE"
    spot_report = failed_spot["coverage"]["safe_spot_checks"]
    assert spot_report["false_negative_rate"] == 0.5
    assert spot_report["per_model"]["model-a"]["passed"] is False
    assert failed_spot["coverage_passed"] is False

    values = assessment_inputs(consecutive=2)
    condition = next(
        item for item in values["evaluation"]["per_condition"] if item["condition_id"] == "400k"
    )
    condition.update(
        {
            "safe_count": 0,
            "query_count": 0,
            "unsafe_count": condition["candidate_count"],
            "safe_fraction": 0.0,
            "query_fraction": 0.0,
            "unsafe_fraction": 1.0,
        }
    )
    condition["models"]["model-a"].update(
        {"safe_fraction": 0.5, "query_fraction": 0.5, "unsafe_fraction": 0.0}
    )
    condition["models"]["model-b"].update(
        {"safe_fraction": 0.0, "query_fraction": 0.0, "unsafe_fraction": 1.0}
    )
    values["policy"]["assessment"]["condition_gates"]["400k"] = {
        "maximum_query_fraction": 0.0,
        "maximum_unsafe_fraction": 1.0,
    }
    failed_condition = run_assessment(values)
    assert failed_condition["decision"] == "CONTINUE"
    assert failed_condition["coverage_passed"] is False
    assert (
        "400k/model-a query fraction exceeds policy"
        in failed_condition["coverage"]["failed_reasons"]
    )

    missing_trigger = assessment_inputs(consecutive=2)
    missing_trigger["selection"]["trigger_model_coverage"] = {
        "available_trigger_models": ["model-a", "model-b"],
        "selected_trigger_models": ["model-a"],
        "missing_trigger_models": ["model-b"],
        "status": "INCOMPLETE",
    }
    trigger_result = run_assessment(missing_trigger)
    assert trigger_result["decision"] == "CONTINUE"
    assert trigger_result["coverage_passed"] is True
    assert trigger_result["accuracy_passed"] is True
    assert trigger_result["pes_gates_passed_this_round"] is False
    assert any(
        "omitted available trigger models" in reason
        for reason in trigger_result["round_evidence"]["failed_reasons"]
    )


def test_strategy_b_rejects_model_calibration_failure_and_audit_drift():
    values = assessment_inputs(consecutive=2)
    values["evaluation"]["calibration"]["model-b"]["status"] = "BLOCKED_CALIBRATION"
    values["evaluation"]["calibration"]["model-b"]["failed_gates"] = ["fixture failure"]
    blocked = run_assessment(values)
    assert blocked["decision"] == "BLOCKED_CALIBRATION"
    assert blocked["coverage_passed"] is False
    assert blocked["coverage"]["calibration"]["status"] == "BLOCKED_CALIBRATION"

    drift = assessment_inputs(consecutive=2)
    drift["split"]["audit_ids"] = ["changed-audit"]
    with pytest.raises(al.ActiveLearningError, match="differ"):
        run_assessment(drift)

    missing_query = assessment_inputs(consecutive=2)
    query_record = missing_query["labeling"]["successful_query_records"][0]["record_id"]
    missing_query["split"]["train_ids"].remove(query_record)
    assert run_assessment(missing_query)["decision"] == "SCIENTIFIC_REVIEW_REQUIRED"

    growing = assessment_inputs(consecutive=2)
    growing["history"]["rounds"][0]["dataset_id"] = "previous-cumulative-dataset"
    growing["history"]["rounds"][0]["split_id"] = "previous-training-split"
    assert run_assessment(growing)["decision"] == "CONVERGED_FOR_DECLARED_DOMAIN"


def test_assessment_binds_metrics_to_exact_immutable_audit_ids():
    values = assessment_inputs()
    values["audit"]["audit_ids"] = list(reversed(values["split"]["audit_ids"]))

    with pytest.raises(al.ActiveLearningError, match="immutable audit split"):
        run_assessment(values)

    aggregate_drift = assessment_inputs()
    aggregate_drift["audit"]["records"][0].update(
        {
            "aggregation": "maximum-over-committee-members",
            "member_values": {"seed-11": 0.01, "seed-23": 0.02},
        }
    )
    with pytest.raises(al.ActiveLearningError, match="maximum committee-member"):
        run_assessment(aggregate_drift)


def test_assessment_binds_campaign_to_policy_and_current_cumulative_split():
    policy_drift = assessment_inputs()
    policy_drift["campaign"]["policy_id"] = "different-policy"
    with pytest.raises(al.ActiveLearningError, match="policy_id differs"):
        run_assessment(policy_drift)

    dataset_drift = assessment_inputs()
    dataset_drift["campaign"]["current_cumulative_dataset"]["split_id"] = "old-split"
    with pytest.raises(al.ActiveLearningError, match="cumulative dataset record differs"):
        run_assessment(dataset_drift)

    decision_drift = assessment_inputs()
    decision_drift["campaign"]["current_decision"] = "CONVERGED_FOR_DECLARED_DOMAIN"
    with pytest.raises(al.ActiveLearningError, match="must be PENDING"):
        run_assessment(decision_drift)

    retry_as_round = assessment_inputs()
    retry_as_round["campaign"]["rounds"][-1]["directory"] = "attempt-2"
    with pytest.raises(al.ActiveLearningError, match="round-NNN"):
        run_assessment(retry_as_round)


def test_adapter_executes_checks_and_collects_oracle_evidence():
    with tempfile.TemporaryDirectory() as directory:
        project = Path(directory)
        attempt = project / "attempt"
        attempt.mkdir()
        (project / "policy.json").write_text(json.dumps(policy("dual")), encoding="utf-8")
        (project / "predictions.json").write_text(
            json.dumps(prediction_bundle("dual")), encoding="utf-8"
        )
        context = {
            "project_root": str(project),
            "attempt_dir": str(attempt),
            "inputs": {
                "policy": "policy.json",
                "committee_predictions": "predictions.json",
            },
            "parameters": {"operation": "committee-evaluate"},
            "backend": "local",
            "resources": {"cpus": 1},
        }
        adapter = load_adapter()
        plan = adapter.plan(context)
        assert plan["status"] == "READY", plan["diagnostics"]
        completed = subprocess.run(plan["argv"], cwd=plan["cwd"], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        checked = adapter.check(context)
        assert checked["status"] == "OK", checked["diagnostics"]
        assert checked["validation_claim"] == "ORACLE_REPLAY_VALIDATION"
        collected = adapter.collect(context)
        assert collected["status"] == "OK"
        assert collected["metrics"]["model_count"] == 2

        result_path = attempt / "committee-evaluation.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["candidates"][0]["combined_risk"] = 999
        result_path.write_text(json.dumps(result), encoding="utf-8")
        assert adapter.check(context)["status"] == "FAIL"


@pytest.mark.parametrize("strategy", ["single", "dual"])
def test_oracle_full_chain_runs_through_core_and_preserves_prior_round(strategy):
    with tempfile.TemporaryDirectory() as directory:
        campaign_root = Path(directory) / f"campaign-{strategy}"
        previous_round = campaign_root / "round-000"
        current_round = campaign_root / "round-001"
        previous_round.mkdir(parents=True)
        current_round.mkdir()
        sentinel = previous_round / "immutable.txt"
        sentinel.write_text("prior-round-artifact\n", encoding="utf-8")

        selected_policy = policy(strategy)
        bundle = prediction_bundle(strategy)
        evaluation = al.evaluate_committee(bundle, selected_policy)
        selection = selection_fixture(evaluation, selected_policy)
        models = ["model-a"] if strategy == "single" else ["model-a", "model-b"]
        by_id = {item["sample_id"]: item for item in evaluation["candidates"]}
        spot_ids = [item["sample_id"] for item in selection["selected_safe_spot_checks"]]
        spot_records = [
            {"sample_id": sample_id, "record_id": f"spot-record-{sample_id}"}
            for sample_id in spot_ids
        ]
        query_ids = [
            item["sample_id"] for item in selection["selected_query_candidates"]
        ]
        query_records = [
            {"sample_id": sample_id, "record_id": f"query-record-{sample_id}"}
            for sample_id in query_ids
        ]
        current_split = {
            "dataset_id": "oracle-cumulative-dataset",
            "split_id": "oracle-cumulative-split-v2",
            "train_ids": bundle["dataset_split"]["train_ids"]
            + [item["record_id"] for item in query_records],
            "validation_ids": bundle["dataset_split"]["validation_ids"],
            "calibration_ids": bundle["dataset_split"]["calibration_ids"],
            "audit_ids": bundle["dataset_split"]["audit_ids"],
        }
        spot_samples = [
            {
                "sample_id": sample_id,
                "model": model,
                "predicted_force_error": by_id[sample_id]["models"][model][
                    "estimated_force_error"
                ],
                "actual_force_error": 0.12,
                "unit": "eV/angstrom",
            }
            for sample_id in spot_ids
            for model in models
        ]
        files = {
            "policy.json": selected_policy,
            "predictions.json": bundle,
            "candidates.json": candidate_manifest(evaluation),
            "direct.json": {
                "schema_version": 1,
                "method": "DIRECT",
                "input_candidate_ids": ["cand-dup", "cand-query2"],
                "selected_candidate_ids": ["cand-dup", "cand-query2"],
                "parameters": {"fixture": "oracle"},
            },
            "audit.json": audit_fixture(models),
            "spot.json": {"schema_version": 1, "samples": spot_samples},
            "labeling.json": {
                "schema_version": 1,
                "status": "OK",
                "successful_query_records": query_records,
                "successful_spot_check_records": spot_records,
                "cumulative_dft_label_count": 7,
                "cumulative_dataset_id": current_split["dataset_id"],
                "cumulative_split_id": current_split["split_id"],
            },
            "split.json": current_split,
            "history.json": {
                "current_round": 1,
                "rounds": [
                    {
                        "round_index": 0,
                        "dataset_id": "oracle-dataset",
                        "split_id": "immutable-split-v1",
                        "audit_ids": ["audit-1", "audit-2"],
                        "coverage_passed": True,
                        "accuracy_passed": True,
                        "pes_gates_passed_this_round": True,
                    }
                ],
            },
            "campaign.json": {
                "campaign_id": f"oracle-{strategy}",
                "policy_id": selected_policy["policy_id"],
                "strategy": selected_policy["strategy"],
                "target_domain": selected_policy["target_domain"],
                "current_cumulative_dataset": {
                    "dataset_id": current_split["dataset_id"],
                    "split_id": current_split["split_id"],
                    "reference": "canonical/current.json",
                },
                "current_decision": "PENDING",
                "rounds": [
                    {"index": 0, "directory": "round-000"},
                    {"index": 1, "directory": "round-001"},
                ],
                "validation_claim": "ORACLE_REPLAY_VALIDATION",
            },
        }
        for name, value in files.items():
            (current_round / name).write_text(json.dumps(value), encoding="utf-8")
        project_data = {
            "schema_version": 1,
            "project": {
                "id": f"oracle-active-learning-{strategy}",
                "name": f"Oracle active learning {strategy}",
            },
            "workflow": {
                "nodes": [
                    {
                        "id": "committee-evaluate",
                        "uses": "active-learning",
                        "backend": "local",
                        "inputs": {
                            "policy": "policy.json",
                            "committee_predictions": "predictions.json",
                        },
                        "parameters": {"operation": "committee-evaluate"},
                    },
                    {
                        "id": "select-candidates",
                        "uses": "active-learning",
                        "needs": ["committee-evaluate"],
                        "backend": "local",
                        "inputs": {
                            "policy": "policy.json",
                            "committee_evaluation": {
                                "from_node": "committee-evaluate",
                                "role": "committee-evaluation",
                            },
                            "candidate_manifest": "candidates.json",
                            "direct_selection": "direct.json",
                        },
                        "parameters": {"operation": "select-candidates"},
                    },
                    {
                        "id": "assess-round",
                        "uses": "active-learning",
                        "needs": ["committee-evaluate", "select-candidates"],
                        "backend": "local",
                        "inputs": {
                            "policy": "policy.json",
                            "committee_evaluation": {
                                "from_node": "committee-evaluate",
                                "role": "committee-evaluation",
                            },
                            "selection_result": {
                                "from_node": "select-candidates",
                                "role": "candidate-selection",
                            },
                            "audit_benchmark": "audit.json",
                            "spot_checks": "spot.json",
                            "labeling_result": "labeling.json",
                            "dataset_split": "split.json",
                            "round_history": "history.json",
                            "campaign": "campaign.json",
                        },
                        "parameters": {"operation": "assess-round"},
                    },
                ]
            },
        }
        (current_round / "project.yaml").write_text(
            json.dumps(project_data), encoding="utf-8"
        )
        initialize(current_round)
        project = load_project(current_round)
        for _ in range(8):
            workflow = query_workflow(project)
            for step in workflow["steps"]:
                if step["state"] == "READY":
                    make_run_plan(project, step["node_id"])
                    run_node(
                        project,
                        step["node_id"],
                        approval=True,
                    )
            workflow = query_workflow(project)
            if all(step["state"] == "OK" for step in workflow["steps"]):
                break
            make_advance_plan(project)
            advance(project)
        completed = query_workflow(project)
        assert completed["counts"] == {"OK": 3}
        assessment = json.loads(
            (
                current_round
                / ".mlipflow"
                / "runs"
                / "assess-round"
                / "attempt-1"
                / "round-assessment.json"
            ).read_text(encoding="utf-8")
        )
        assert assessment["decision"] == "CONVERGED_FOR_DECLARED_DOMAIN"
        assert assessment["validation_claim"] == "ORACLE_REPLAY_VALIDATION"
        assert sentinel.read_text(encoding="utf-8") == "prior-round-artifact\n"
