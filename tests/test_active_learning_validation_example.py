from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlipflow.science.active_learning import CANDIDATE_CONTRACT


def test_validation_helper_uses_the_plugin_candidate_contract() -> None:
    helper = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "active_learning_validation"
        / "prepare_round_inputs.py"
    )
    spec = importlib.util.spec_from_file_location("active_learning_validation_helper", helper)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.ACTIVE_CANDIDATE_CONTRACT == CANDIDATE_CONTRACT


def test_validation_example_and_cluster_template_are_packaged_portably() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    section = pyproject.split("[tool.setuptools.data-files]", 1)[1].split("\n[", 1)[0]
    declared = set()
    for line in section.splitlines():
        match = re.fullmatch(r'"([^"]+)"\s*=\s*(\[.*\])', line.strip())
        if match is not None and "active_learning_validation" in match.group(1):
            declared.update(ast.literal_eval(match.group(2)))
    assert declared == {
        "examples/active_learning_validation/*.md",
        "examples/active_learning_validation/*.py",
        "examples/active_learning_validation/cluster/*.example",
    }

    template = (
        root
        / "examples"
        / "active_learning_validation"
        / "cluster"
        / "run.sh.example"
    ).read_text(encoding="utf-8")
    for placeholder in (
        "{{RUN_DIR}}",
        "{{INPUT_DIR}}",
        "{{OUTPUT_DIR}}",
        "{{LOG_DIR}}",
        "{{PROJECT_ID}}",
        "{{NODE_ID}}",
        "{{ATTEMPT}}",
        "{{CPUS}}",
    ):
        assert placeholder in template
    assert "/Users/" not in template
    assert "/public/home/" not in template
    assert "ssh " not in template


def test_assessment_helper_does_not_fold_test_records_into_training_pool(
    tmp_path: Path,
) -> None:
    helper = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "active_learning_validation"
        / "prepare_round_inputs.py"
    )
    spec = importlib.util.spec_from_file_location("active_learning_assessment_helper", helper)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def write(name: str, value: object) -> str:
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return str(path)

    metrics = [
        {"metric": name, "value": 0.0, "unit": "test-unit"}
        for name in (
            "energy_mae",
            "energy_rmse",
            "force_mae",
            "force_rmse",
            "maximum_atomic_force_error",
        )
    ]
    args = SimpleNamespace(
        policy=write(
            "policy.json",
            {"selection": {"include_safe_spot_checks_in_training": False}},
        ),
        committee_evaluation=write(
            "evaluation.json",
            {"dataset_split": {"calibration_ids": [], "audit_ids": ["audit-1"]}},
        ),
        committee_predictions=write("predictions.json", {"models": []}),
        selection_result=write(
            "selection.json",
            {
                "selected_query_candidates": [{"sample_id": "query-1"}],
                "selected_safe_spot_checks": [],
            },
        ),
        query_canonical=write(
            "query.json",
            {"records": [{"source_structure_id": "query-1", "record_id": "record-query"}]},
        ),
        spot_canonical=write("spot.json", {"records": []}),
        audit_metrics=[f"seed-11={write('metrics.json', {'records': metrics})}"],
        audit_evidence=[
            "seed-11="
            + write(
                "evidence.json",
                {
                    "split": "test",
                    "records": [{"sample_id": "audit-1", "target": "energy"}],
                },
            )
        ],
        cumulative_split=write(
            "split.json",
            {
                "dataset_id": "dataset-1",
                "split_id": "split-1",
                "train_record_ids": ["record-old-train"],
                "validation_record_ids": ["record-old-validation"],
                "test_record_ids": ["record-query"],
            },
        ),
        audit_output=str(tmp_path / "audit.json"),
        spot_output=str(tmp_path / "spot-output.json"),
        labeling_output=str(tmp_path / "labeling.json"),
        dataset_split_output=str(tmp_path / "assessment-split.json"),
    )

    with pytest.raises(ValueError, match="train/validation"):
        module.assessment_inputs(args)


def test_evaluation_split_keeps_canonical_test_records_excluded() -> None:
    helper = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "active_learning_validation"
        / "prepare_round_inputs.py"
    )
    spec = importlib.util.spec_from_file_location("active_learning_split_helper", helper)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module._active_learning_dataset_split(
        "bootstrap-dataset",
        {
            "split_id": "bootstrap-split",
            "train_record_ids": ["calibration-1"],
            "validation_record_ids": ["calibration-2"],
            "test_record_ids": ["audit-1"],
        },
        {
            "dataset_id": "training-dataset",
            "split_id": "training-split",
            "train_record_ids": ["train-1"],
            "validation_record_ids": ["validation-1"],
            "test_record_ids": ["ordinary-test-1"],
        },
    )

    assert result["train_ids"] == ["train-1"]
    assert result["validation_ids"] == ["validation-1"]
    assert "ordinary-test-1" not in {
        *result["train_ids"],
        *result["validation_ids"],
        *result["calibration_ids"],
        *result["audit_ids"],
    }


def test_prediction_split_rebind_preserves_numerical_evidence(tmp_path: Path) -> None:
    helper = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "active_learning_validation"
        / "prepare_round_inputs.py"
    )
    spec = importlib.util.spec_from_file_location("active_learning_rebind_helper", helper)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source = {
        "dataset_split": {
            "dataset_id": "old",
            "split_id": "old",
            "train_ids": ["train"],
            "validation_ids": ["validation", "ordinary-test"],
            "calibration_ids": ["calibration"],
            "audit_ids": ["audit"],
        },
        "models": [{"model_id": "model", "members": [{"predictions": [1, 2, 3]}]}],
    }
    corrected = {
        "dataset_split": {
            "dataset_id": "new",
            "split_id": "new",
            "train_ids": ["train"],
            "validation_ids": ["validation"],
            "calibration_ids": ["calibration"],
            "audit_ids": ["audit"],
        }
    }
    source_path = tmp_path / "source.json"
    evaluation_path = tmp_path / "evaluation.json"
    output_path = tmp_path / "corrected.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    evaluation_path.write_text(json.dumps(corrected), encoding="utf-8")

    module.prediction_split_rebind(
        SimpleNamespace(
            committee_predictions=str(source_path),
            evaluation_dataset=str(evaluation_path),
            output=str(output_path),
        )
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["dataset_split"] == corrected["dataset_split"]
    assert result["models"] == source["models"]
    assert result["split_rebind"]["numerical_predictions_changed"] is False


def test_direct_selection_replay_preserves_exact_candidate_ids(tmp_path: Path) -> None:
    helper = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "active_learning_validation"
        / "prepare_round_inputs.py"
    )
    spec = importlib.util.spec_from_file_location("active_learning_direct_helper", helper)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source_path = tmp_path / "selection.json"
    output_path = tmp_path / "direct.json"
    source_path.write_text(
        json.dumps(
            {
                "direct_input_candidate_ids": ["candidate-1", "candidate-2"],
                "direct_selected_candidate_ids": ["candidate-2"],
                "diversity_selection": {
                    "method": "DIRECT",
                    "parameters": {"n_clusters": 1},
                },
            }
        ),
        encoding="utf-8",
    )

    module.direct_selection_replay(
        SimpleNamespace(selection_result=str(source_path), output=str(output_path))
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["input_candidate_ids"] == ["candidate-1", "candidate-2"]
    assert result["selected_candidate_ids"] == ["candidate-2"]
    assert result["parameters"]["replay_mode"] == "existing-result-structured-capture"


def test_audit_handoff_uses_conservative_committee_maximum(tmp_path: Path) -> None:
    helper = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "active_learning_validation"
        / "prepare_round_inputs.py"
    )
    spec = importlib.util.spec_from_file_location("active_learning_audit_helper", helper)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    metric_names = (
        "energy_mae",
        "energy_rmse",
        "force_mae",
        "force_rmse",
        "maximum_atomic_force_error",
    )
    metric_args = []
    evidence_args = []
    for member, value in (("seed-11", 0.1), ("seed-23", 0.2)):
        metrics_path = tmp_path / f"{member}-metrics.json"
        evidence_path = tmp_path / f"{member}-evidence.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "records": [
                        {"metric": metric, "value": value, "unit": "unit"}
                        for metric in metric_names
                    ]
                }
            ),
            encoding="utf-8",
        )
        evidence_path.write_text(
            json.dumps(
                {
                    "split": "test",
                    "records": [
                        {"sample_id": "audit-1", "target": "energy"},
                        {"sample_id": "audit-1", "target": "force"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        metric_args.append(f"{member}={metrics_path}")
        evidence_args.append(f"{member}={evidence_path}")

    result = module._audit_handoff(metric_args, evidence_args, ["audit-1"])

    assert result["aggregation"] == "maximum-over-committee-members"
    assert {record["value"] for record in result["records"]} == {0.2}
    assert all(
        record["member_values"] == {"seed-11": 0.1, "seed-23": 0.2}
        for record in result["records"]
    )


def test_split_seed_review_preserves_prior_training_and_new_queries(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    helper_path = (
        root / "examples" / "active_learning_validation" / "prepare_round_inputs.py"
    )
    spec = importlib.util.spec_from_file_location(
        "active_learning_split_seed_helper", helper_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from tests.test_dft_dataset_handoff import _canonical, _contract

    contract = _contract()
    initial = _canonical(4, project_id="initial")
    query_400k = _canonical(1, project_id="query-400k", start=10)
    query_600k = _canonical(1, project_id="query-600k", start=20)
    previous = contract.build_split_manifest(
        initial,
        strategy="deterministic",
        seed=7,
        fractions={"train": 0.5, "validation": 0.25, "test": 0.25},
    )

    def write(name: str, value: object) -> str:
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return str(path)

    output = tmp_path / "review.json"
    module.split_seed_review(
        SimpleNamespace(
            dataset_contract=str(root / "plugins/dft-labeling/dataset_contract.py"),
            canonical_source=[
                write("initial.json", initial),
                write("query-400k.json", query_400k),
                write("query-600k.json", query_600k),
            ],
            previous_training_split=write("previous-split.json", previous),
            query_source_structure_id=["structure-010", "structure-020"],
            train_fraction=0.6,
            validation_fraction=0.2,
            test_fraction=0.2,
            maximum_seed=1000,
            output=str(output),
        )
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    selected = result["split_manifest"]
    training_ids = set(selected["train_record_ids"]) | set(
        selected["validation_record_ids"]
    )
    previous_training_ids = set(previous["train_record_ids"]) | set(
        previous["validation_record_ids"]
    )
    query_ids = {
        query_400k["records"][0]["record_id"],
        query_600k["records"][0]["record_id"],
    }
    assert previous_training_ids | query_ids <= training_ids
    assert set(selected["test_record_ids"]) == set(previous["test_record_ids"])
    assert result["selected_seed"] == selected["seed"]
    for seed in range(result["selected_seed"]):
        earlier = contract.build_split_manifest(
            contract.merge_canonical_datasets([initial, query_400k, query_600k]),
            strategy="deterministic",
            seed=seed,
            fractions={"train": 0.6, "validation": 0.2, "test": 0.2},
        )
        earlier_training = set(earlier["train_record_ids"]) | set(
            earlier["validation_record_ids"]
        )
        assert not previous_training_ids | query_ids <= earlier_training


def test_model_index_records_shared_training_inputs(tmp_path: Path) -> None:
    helper = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "active_learning_validation"
        / "prepare_round_inputs.py"
    )
    spec = importlib.util.spec_from_file_location("active_learning_model_index_helper", helper)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from tests.test_dft_dataset_handoff import _canonical

    canonical = _canonical(2, project_id="committee-training")
    canonical_path = tmp_path / "canonical.json"
    canonical_path.write_text(json.dumps(canonical), encoding="utf-8")
    dataset_contract = (
        Path(__file__).resolve().parents[1]
        / "plugins"
        / "dft-labeling"
        / "dataset_contract.py"
    )

    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "strategy": {"primary_model": "chgnet-primary"},
                "committee": {
                    "models": [
                        {"model_id": "chgnet-primary", "member_seeds": [11, 23]}
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    member_references = []
    for seed in (11, 23):
        attempt = tmp_path / f"seed-{seed}"
        attempt.mkdir()
        reference = {
            "schema_version": 1,
            "model_id": f"model-{seed}",
            "framework": "chgnet",
            "relative_path": f"committee/model-{seed}.pt",
            "kind": "file",
        }
        report = {
            "status": "OK",
            "return_code": 0,
            "framework": "chgnet",
            "operation": "finetune",
            "dataset": {
                "id": f"dft-merged-{canonical['dataset_id']}-chgnet-split-1",
                "relative_path": "datasets/chgnet-split-1",
            },
            "foundation_model": {
                "id": "foundation",
                "relative_path": "foundation/chgnet.pt",
            },
            "published_model": reference,
            "framework_version": "0.test",
        }
        result = {
            "status": "OK",
            "seed": seed,
            "framework": "chgnet",
            "operation": "finetune",
            "framework_version": "0.test",
            "dataset_path": "datasets/chgnet-split-1",
            "config_path": "input/config.json",
            "foundation_model_path": "foundation/chgnet.pt",
            "precision": "float32",
            "model_artifact": {"path": "model-artifact"},
            "provenance": {
                "split": {
                    "source": "predefined",
                    "split_id": "split-1",
                    "train_record_ids": ["record-train"],
                    "validation_record_ids": ["record-validation"],
                    "test_record_ids": ["record-test"],
                }
            },
        }
        reference_path = attempt / "model-reference.json"
        reference_path.write_text(json.dumps(reference), encoding="utf-8")
        (attempt / "cluster-run-report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        (attempt / "training-result.json").write_text(json.dumps(result), encoding="utf-8")
        member_references.append(f"{seed}={reference_path}")

    output = tmp_path / "model-index.json"
    module.model_index(
        SimpleNamespace(
            policy=str(policy_path),
            dataset_contract=str(dataset_contract),
            canonical_source=[str(canonical_path)],
            member_reference=member_references,
            output=str(output),
        )
    )
    index = json.loads(output.read_text(encoding="utf-8"))
    model = index["models"][0]
    assert model["training"]["canonical_dataset_id"] == f"dft-merged-{canonical['dataset_id']}"
    assert model["training"]["split"]["split_id"] == "split-1"
    assert {member["seed"] for member in model["members"]} == {11, 23}

    drifted = json.loads(
        (tmp_path / "seed-23" / "training-result.json").read_text(encoding="utf-8")
    )
    drifted["precision"] = "float64"
    (tmp_path / "seed-23" / "training-result.json").write_text(
        json.dumps(drifted), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="must share the training dataset, split, and parameters"):
        module.model_index(
            SimpleNamespace(
                policy=str(policy_path),
                dataset_contract=str(dataset_contract),
                canonical_source=[str(canonical_path)],
                member_reference=member_references,
                output=str(tmp_path / "drifted-index.json"),
            )
        )
