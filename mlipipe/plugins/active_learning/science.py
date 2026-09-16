"""Deterministic mathematics for offline, round-based active learning.

The module consumes explicit committee predictions and DFT evidence.  It never
loads a model, runs MD/DFT, mutates a dataset, or chooses scientific thresholds.
Those stages remain owned by the corresponding MLIPipe plugins.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Mapping, Sequence


STRATEGY_SINGLE = "single-model-committee"
STRATEGY_DUAL = "dual-model-risk-union"
STRATEGIES = frozenset({STRATEGY_SINGLE, STRATEGY_DUAL})
RISK_ORDER = {"SAFE": 0, "QUERY": 1, "UNSAFE": 2}
DECISIONS = frozenset(
    {
        "CONTINUE",
        "CONVERGED_FOR_DECLARED_DOMAIN",
        "BLOCKED_CALIBRATION",
        "BLOCKED_SAMPLING",
        "BUDGET_EXHAUSTED",
        "SCIENTIFIC_REVIEW_REQUIRED",
    }
)
COMMON_AUDIT_METRICS = frozenset(
    {
        "energy_mae",
        "energy_rmse",
        "force_mae",
        "force_rmse",
        "maximum_atomic_force_error",
    }
)

POLICY_CONTRACT = "mlipipe/active-learning-policy"
PREDICTION_CONTRACT = "mlipipe/active-learning-committee-predictions"
CANDIDATE_CONTRACT = "mlipipe/active-learning-candidates"
EVALUATION_CONTRACT = "mlipipe/active-learning-committee-evaluation"
SELECTION_CONTRACT = "mlipipe/active-learning-selection"
ASSESSMENT_CONTRACT = "mlipipe/active-learning-round-assessment"


class ActiveLearningError(ValueError):
    """Raised when active-learning evidence violates its scientific contract."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ActiveLearningError(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise ActiveLearningError(f"{field} must be an array")
    return value


def _plain(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise ActiveLearningError(f"{field} must be a non-empty single-line string")
    return value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActiveLearningError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ActiveLearningError(f"{field} must be finite")
    return result


def _positive_number(value: Any, field: str) -> float:
    result = _number(value, field)
    if result <= 0:
        raise ActiveLearningError(f"{field} must be positive")
    return result


def _fraction(value: Any, field: str) -> float:
    result = _number(value, field)
    if not 0 <= result <= 1:
        raise ActiveLearningError(f"{field} must be between zero and one")
    return result


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ActiveLearningError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ActiveLearningError(f"{field} must be a non-negative integer")
    return value


def _vector(value: Any, length: int, field: str) -> list[float]:
    raw = _sequence(value, field)
    if len(raw) != length:
        raise ActiveLearningError(f"{field} must contain {length} values")
    return [_number(item, field) for item in raw]


def _forces(value: Any, atom_count: int, field: str) -> list[list[float]]:
    raw = _sequence(value, field)
    if len(raw) != atom_count:
        raise ActiveLearningError(f"{field} must contain {atom_count} atomic forces")
    return [_vector(row, 3, f"{field}[{index}]") for index, row in enumerate(raw)]


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def population_standard_deviation(values: Sequence[float]) -> float:
    """Return ``sqrt(mean((x - mean(x))**2))`` for at least two values."""

    normalized = [_number(value, "values") for value in values]
    if len(normalized) < 2:
        raise ActiveLearningError("population standard deviation requires two values")
    center = _mean(normalized)
    return math.sqrt(math.fsum((value - center) ** 2 for value in normalized) / len(normalized))


def atomic_force_disagreements(
    member_forces: Sequence[Sequence[Sequence[float]]],
) -> list[float]:
    """Return population force-vector disagreement for every atom.

    For atom ``i`` and ``M`` committee members the definition is::

        sqrt((1 / M) * sum_m ||F_i^m - mean_m(F_i^m)||_2^2)

    The maximum of this vector is the primary structure acquisition quantity.
    """

    raw_members = _sequence(member_forces, "member_forces")
    if len(raw_members) < 2:
        raise ActiveLearningError("force disagreement requires at least two committee members")
    first = _sequence(raw_members[0], "member_forces[0]")
    atom_count = len(first)
    if atom_count < 1:
        raise ActiveLearningError("force disagreement requires at least one atom")
    normalized = [
        _forces(value, atom_count, f"member_forces[{index}]")
        for index, value in enumerate(raw_members)
    ]
    result = []
    for atom_index in range(atom_count):
        mean_force = [
            _mean([member[atom_index][component] for member in normalized])
            for component in range(3)
        ]
        squared = math.fsum(
            math.fsum(
                (member[atom_index][component] - mean_force[component]) ** 2
                for component in range(3)
            )
            for member in normalized
        )
        result.append(math.sqrt(squared / len(normalized)))
    return result


def force_disagreement_summary(
    member_forces: Sequence[Sequence[Sequence[float]]],
) -> dict[str, Any]:
    per_atom = atomic_force_disagreements(member_forces)
    return {
        "per_atom_force_disagreement": per_atom,
        "maximum_atomic_force_disagreement": max(per_atom),
        "mean_atomic_force_disagreement": _mean(per_atom),
        "atom_count": len(per_atom),
        "member_count": len(member_forces),
    }


def energy_disagreement(member_total_energies: Sequence[float], atom_count: int) -> float:
    """Return population standard deviation of total energy divided by atom count."""

    _positive_int(atom_count, "atom_count")
    return population_standard_deviation(member_total_energies) / atom_count


def committee_mean_force_error(
    member_forces: Sequence[Sequence[Sequence[float]]],
    reference_forces: Sequence[Sequence[float]],
) -> float:
    """Maximum atomic vector error between the committee mean and DFT forces."""

    raw_members = _sequence(member_forces, "member_forces")
    if len(raw_members) < 2:
        raise ActiveLearningError("committee force error requires at least two members")
    atom_count = len(_sequence(raw_members[0], "member_forces[0]"))
    normalized = [
        _forces(value, atom_count, f"member_forces[{index}]")
        for index, value in enumerate(raw_members)
    ]
    reference = _forces(reference_forces, atom_count, "reference_forces")
    errors = []
    for atom_index in range(atom_count):
        mean_force = [
            _mean([member[atom_index][component] for member in normalized])
            for component in range(3)
        ]
        errors.append(
            math.sqrt(
                math.fsum(
                    (mean_force[component] - reference[atom_index][component]) ** 2
                    for component in range(3)
                )
            )
        )
    return max(errors)


def conservative_empirical_quantile(values: Sequence[float], coverage: float) -> float:
    """Nearest-rank empirical quantile using ``ceil(coverage * n)``."""

    requested = _fraction(coverage, "coverage")
    if requested <= 0:
        raise ActiveLearningError("coverage must be greater than zero")
    ordered = sorted(_number(value, "quantile values") for value in values)
    if not ordered:
        raise ActiveLearningError("quantile requires at least one value")
    rank = max(1, math.ceil(requested * len(ordered)))
    return ordered[rank - 1]


def _distribution_summary(values: Sequence[float]) -> dict[str, Any]:
    normalized = sorted(_number(value, "uncertainty distribution") for value in values)
    if not normalized:
        return {
            "count": 0,
            "minimum": None,
            "mean": None,
            "q50": None,
            "q90": None,
            "maximum": None,
        }
    return {
        "count": len(normalized),
        "minimum": normalized[0],
        "mean": _mean(normalized),
        "q50": conservative_empirical_quantile(normalized, 0.5),
        "q90": conservative_empirical_quantile(normalized, 0.9),
        "maximum": normalized[-1],
    }


def _pearson_diagnostic(xs: Sequence[float], ys: Sequence[float]) -> dict[str, Any]:
    if len(xs) < 2:
        return {"pearson_r": None, "reason": "insufficient-samples"}
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    numerator = math.fsum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_norm = math.sqrt(math.fsum((x - x_mean) ** 2 for x in xs))
    y_norm = math.sqrt(math.fsum((y - y_mean) ** 2 for y in ys))
    if x_norm == 0 or y_norm == 0:
        return {"pearson_r": None, "reason": "zero-variance"}
    return {"pearson_r": numerator / (x_norm * y_norm), "reason": None}


def classify_estimated_error(
    estimated_error: float, label_threshold: float, abort_threshold: float
) -> str:
    value = _number(estimated_error, "estimated_error")
    label = _positive_number(label_threshold, "label_threshold")
    abort = _positive_number(abort_threshold, "abort_threshold")
    if abort <= label:
        raise ActiveLearningError("abort_threshold must exceed label_threshold")
    if value <= label:
        return "SAFE"
    if value <= abort:
        return "QUERY"
    return "UNSAFE"


def calibrate_force_uncertainty(
    samples: Sequence[Mapping[str, Any]],
    *,
    requested_coverage: float,
    minimum_observed_coverage: float,
    epsilon: float,
    minimum_sample_count: int,
    label_threshold: float,
    maximum_false_negative_rate: float,
) -> dict[str, Any]:
    """Fit and report the explicit quantile scale from calibration evidence."""

    coverage = _fraction(requested_coverage, "requested_coverage")
    if coverage <= 0:
        raise ActiveLearningError("requested_coverage must be greater than zero")
    minimum_coverage = _fraction(minimum_observed_coverage, "minimum_observed_coverage")
    eps = _positive_number(epsilon, "epsilon")
    minimum = _positive_int(minimum_sample_count, "minimum_sample_count")
    threshold = _positive_number(label_threshold, "label_threshold")
    maximum_false_negative = _fraction(
        maximum_false_negative_rate, "maximum_false_negative_rate"
    )
    normalized = []
    seen = set()
    for index, raw in enumerate(samples):
        item = _mapping(raw, f"samples[{index}]")
        sample_id = _plain(item.get("sample_id"), f"samples[{index}].sample_id")
        if sample_id in seen:
            raise ActiveLearningError(f"duplicate calibration sample: {sample_id}")
        seen.add(sample_id)
        disagreement = _number(item.get("disagreement"), f"{sample_id}.disagreement")
        actual_error = _number(item.get("actual_error"), f"{sample_id}.actual_error")
        if disagreement < 0 or actual_error < 0:
            raise ActiveLearningError("calibration disagreements and errors must be non-negative")
        normalized.append(
            {
                "sample_id": sample_id,
                "disagreement": disagreement,
                "actual_error": actual_error,
                "scale": actual_error / (disagreement + eps),
            }
        )
    sample_count = len(normalized)
    if not normalized:
        return {
            "status": "BLOCKED_CALIBRATION",
            "sample_count": 0,
            "minimum_sample_count": minimum,
            "requested_coverage": coverage,
            "minimum_observed_coverage": minimum_coverage,
            "epsilon": eps,
            "scale_quantile": None,
            "observed_coverage": None,
            "correlation_diagnostics": {"pearson_r": None, "reason": "no-samples"},
            "false_negative_count": 0,
            "false_negative_rate": None,
            "maximum_calibration_disagreement": None,
            "samples": [],
            "failed_gates": ["calibration sample count is zero"],
        }
    scale = conservative_empirical_quantile([item["scale"] for item in normalized], coverage)
    false_negative_count = 0
    covered = 0
    evidence = []
    for item in normalized:
        estimated = scale * item["disagreement"]
        calibration_coverage_bound = scale * (item["disagreement"] + eps)
        is_covered = item["actual_error"] <= calibration_coverage_bound
        false_negative = estimated <= threshold < item["actual_error"]
        covered += int(is_covered)
        false_negative_count += int(false_negative)
        evidence.append(
            {
                **item,
                "estimated_force_error": estimated,
                "calibration_coverage_bound": calibration_coverage_bound,
                "covered": is_covered,
                "false_negative": false_negative,
            }
        )
    observed = covered / sample_count
    false_negative_rate = false_negative_count / sample_count
    failed = []
    if sample_count < minimum:
        failed.append(f"calibration sample count {sample_count} is below {minimum}")
    if observed < minimum_coverage:
        failed.append(
            f"observed calibration coverage {observed:.12g} is below {minimum_coverage:.12g}"
        )
    if false_negative_rate > maximum_false_negative:
        failed.append(
            "calibration false-negative rate "
            f"{false_negative_rate:.12g} exceeds {maximum_false_negative:.12g}"
        )
    return {
        "status": "PASS" if not failed else "BLOCKED_CALIBRATION",
        "sample_count": sample_count,
        "minimum_sample_count": minimum,
        "requested_coverage": coverage,
        "minimum_observed_coverage": minimum_coverage,
        "epsilon": eps,
        "scale_quantile": scale,
        "quantile_method": "nearest-rank-ceil",
        "observed_coverage": observed,
        "correlation_diagnostics": _pearson_diagnostic(
            [item["disagreement"] for item in normalized],
            [item["actual_error"] for item in normalized],
        ),
        "false_negative_count": false_negative_count,
        "false_negative_rate": false_negative_rate,
        "maximum_calibration_disagreement": max(
            item["disagreement"] for item in normalized
        ),
        "samples": evidence,
        "failed_gates": failed,
    }


def _policy_models(policy: Mapping[str, Any]) -> tuple[str, list[str], dict[str, list[int]]]:
    strategy = _mapping(policy.get("strategy"), "policy.strategy")
    mode = strategy.get("mode")
    if mode not in STRATEGIES:
        raise ActiveLearningError(f"strategy.mode must be one of {sorted(STRATEGIES)}")
    if mode == STRATEGY_SINGLE:
        models = [_plain(strategy.get("primary_model"), "strategy.primary_model")]
    else:
        raw_models = _sequence(strategy.get("models"), "strategy.models")
        models = [_plain(value, "strategy.models") for value in raw_models]
        if len(models) != 2 or len(set(models)) != 2:
            raise ActiveLearningError("dual-model-risk-union requires exactly two distinct models")
    committee = _mapping(policy.get("committee"), "policy.committee")
    records = _sequence(committee.get("models"), "policy.committee.models")
    seeds: dict[str, list[int]] = {}
    for index, raw in enumerate(records):
        record = _mapping(raw, f"policy.committee.models[{index}]")
        model_id = _plain(record.get("model_id"), f"committee.models[{index}].model_id")
        raw_seeds = _sequence(record.get("member_seeds"), f"{model_id}.member_seeds")
        member_seeds = []
        for seed in raw_seeds:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ActiveLearningError(f"{model_id}.member_seeds must contain integers")
            member_seeds.append(seed)
        if len(member_seeds) < 2 or len(set(member_seeds)) != len(member_seeds):
            raise ActiveLearningError(
                f"{model_id} requires at least two distinct explicit committee seeds"
            )
        if model_id in seeds:
            raise ActiveLearningError(f"duplicate committee policy for model {model_id}")
        seeds[model_id] = member_seeds
    if set(seeds) != set(models):
        raise ActiveLearningError("committee models and strategy models must match exactly")
    return str(mode), models, seeds


_TARGET_DOMAIN_FIELDS = frozenset(
    {
        "composition",
        "structure_families",
        "defect_states",
        "temperature_range_k",
        "pressure_range_gpa",
        "ensembles",
        "supercell_range",
        "include_lasp_ssw_high_energy",
        "allow_phase_change",
        "allow_melting",
        "allow_decomposition",
        "mobile_species",
        "intended_use",
    }
)


def validate_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    policy = _mapping(policy, "policy")
    if policy.get("schema_version") != 1 or policy.get("contract") != POLICY_CONTRACT:
        raise ActiveLearningError("policy must use schema_version=1 and the active-learning contract")
    policy_id = _plain(policy.get("policy_id"), "policy.policy_id")
    mode, models, seeds = _policy_models(policy)
    target_domain = _mapping(policy.get("target_domain"), "policy.target_domain")
    missing_domain = sorted(_TARGET_DOMAIN_FIELDS - set(target_domain))
    if missing_domain:
        raise ActiveLearningError("target_domain is missing: " + ", ".join(missing_domain))
    target_conditions = _sequence(policy.get("target_conditions"), "policy.target_conditions")
    conditions_by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(target_conditions):
        record = _mapping(raw, f"target_conditions[{index}]")
        condition_id = _plain(record.get("id"), f"target_conditions[{index}].id")
        if condition_id in conditions_by_id:
            raise ActiveLearningError("target_conditions require unique non-empty ids")
        condition = {str(key): value for key, value in record.items() if key != "id"}
        if not condition:
            raise ActiveLearningError(
                f"target_conditions[{index}] requires explicit condition metadata"
            )
        conditions_by_id[condition_id] = condition
    condition_ids = list(conditions_by_id)
    if not condition_ids:
        raise ActiveLearningError("target_conditions require unique non-empty ids")

    committee = _mapping(policy.get("committee"), "policy.committee")
    requested_coverage = _fraction(
        committee.get("requested_coverage"), "committee.requested_coverage"
    )
    if requested_coverage <= 0:
        raise ActiveLearningError("committee.requested_coverage must be greater than zero")
    minimum_observed_coverage = _fraction(
        committee.get("minimum_observed_coverage"),
        "committee.minimum_observed_coverage",
    )
    epsilon = _positive_number(committee.get("epsilon"), "committee.epsilon")
    minimum_samples = _positive_int(
        committee.get("minimum_calibration_samples"),
        "committee.minimum_calibration_samples",
    )
    maximum_calibration_fn = _fraction(
        committee.get("maximum_calibration_false_negative_rate"),
        "committee.maximum_calibration_false_negative_rate",
    )
    thresholds = _mapping(policy.get("thresholds"), "policy.thresholds")
    force = _mapping(thresholds.get("force"), "policy.thresholds.force")
    force_unit = _plain(force.get("unit"), "thresholds.force.unit")
    label_threshold = _positive_number(
        force.get("label_threshold"), "thresholds.force.label_threshold"
    )
    abort_threshold = _positive_number(
        force.get("abort_threshold"), "thresholds.force.abort_threshold"
    )
    if abort_threshold <= label_threshold:
        raise ActiveLearningError("force abort_threshold must exceed label_threshold")
    return {
        "policy_id": policy_id,
        "mode": mode,
        "models": models,
        "seeds": seeds,
        "condition_ids": condition_ids,
        "conditions_by_id": conditions_by_id,
        "requested_coverage": requested_coverage,
        "minimum_observed_coverage": minimum_observed_coverage,
        "epsilon": epsilon,
        "minimum_calibration_samples": minimum_samples,
        "maximum_calibration_false_negative_rate": maximum_calibration_fn,
        "force_unit": force_unit,
        "label_threshold": label_threshold,
        "abort_threshold": abort_threshold,
    }


def _validate_splits(raw: Any) -> dict[str, Any]:
    splits = _mapping(raw, "dataset_split")
    dataset_id = _plain(splits.get("dataset_id"), "dataset_split.dataset_id")
    split_id = _plain(splits.get("split_id"), "dataset_split.split_id")
    names = ("train_ids", "validation_ids", "calibration_ids", "audit_ids")
    normalized: dict[str, list[str]] = {}
    ownership: dict[str, str] = {}
    for name in names:
        values = [
            _plain(item, f"dataset_split.{name}")
            for item in _sequence(splits.get(name), f"dataset_split.{name}")
        ]
        if name in {"calibration_ids", "audit_ids"} and not values:
            raise ActiveLearningError(f"dataset_split.{name} must be non-empty")
        if len(values) != len(set(values)):
            raise ActiveLearningError(f"dataset_split.{name} contains duplicate sample ids")
        for sample_id in values:
            if sample_id in ownership:
                raise ActiveLearningError(
                    f"sample {sample_id} appears in both {ownership[sample_id]} and {name}"
                )
            ownership[sample_id] = name
        normalized[name] = values
    return {"dataset_id": dataset_id, "split_id": split_id, **normalized}


def _prediction_map(member: Mapping[str, Any], model_id: str) -> dict[str, Mapping[str, Any]]:
    raw_predictions = _sequence(member.get("predictions"), f"{model_id}.member.predictions")
    predictions: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_predictions):
        item = _mapping(raw, f"{model_id}.predictions[{index}]")
        sample_id = _plain(item.get("sample_id"), f"{model_id}.predictions[{index}].sample_id")
        if sample_id in predictions:
            raise ActiveLearningError(f"duplicate prediction for {model_id}/{sample_id}")
        if item.get("split") not in {"calibration", "candidate"}:
            raise ActiveLearningError(
                f"{model_id}/{sample_id}.split must be calibration or candidate"
            )
        predictions[sample_id] = item
    return predictions


_CANDIDATE_METADATA = (
    "condition_id",
    "condition",
    "replica",
    "frame_index",
    "structure_path",
    "near_duplicate_group",
    "physical_validity",
)


def _candidate_metadata(item: Mapping[str, Any], sample_id: str) -> dict[str, Any]:
    condition_id = _plain(item.get("condition_id"), f"{sample_id}.condition_id")
    condition = dict(_mapping(item.get("condition"), f"{sample_id}.condition"))
    replica = _plain(item.get("replica"), f"{sample_id}.replica")
    frame_index = _nonnegative_int(item.get("frame_index"), f"{sample_id}.frame_index")
    structure_path = _plain(item.get("structure_path"), f"{sample_id}.structure_path")
    near_group = _plain(
        item.get("near_duplicate_group"), f"{sample_id}.near_duplicate_group"
    )
    validity = _mapping(item.get("physical_validity"), f"{sample_id}.physical_validity")
    valid = validity.get("valid")
    severe = validity.get("severe")
    reasons = _sequence(validity.get("reasons"), f"{sample_id}.physical_validity.reasons")
    if not isinstance(valid, bool) or not isinstance(severe, bool):
        raise ActiveLearningError(f"{sample_id}.physical_validity flags must be booleans")
    normalized_reasons = [_plain(reason, f"{sample_id}.physical_validity.reasons") for reason in reasons]
    if severe and valid:
        raise ActiveLearningError(f"{sample_id} cannot be both severe and physically valid")
    return {
        "condition_id": condition_id,
        "condition": condition,
        "replica": replica,
        "frame_index": frame_index,
        "structure_path": structure_path,
        "near_duplicate_group": near_group,
        "physical_validity": {
            "valid": valid,
            "severe": severe,
            "reasons": normalized_reasons,
        },
    }


def _calibration_labels(raw: Any) -> dict[str, Mapping[str, Any]]:
    labels: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(_sequence(raw, "calibration_labels")):
        item = _mapping(value, f"calibration_labels[{index}]")
        sample_id = _plain(item.get("sample_id"), f"calibration_labels[{index}].sample_id")
        if sample_id in labels:
            raise ActiveLearningError(f"duplicate calibration label: {sample_id}")
        labels[sample_id] = item
    return labels


def _safe_force_evaluation(member_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    atom_count = _positive_int(member_records[0].get("atom_count"), "atom_count")
    if any(record.get("atom_count") != atom_count for record in member_records):
        raise ActiveLearningError("committee members disagree on atom_count")
    force_values = [record.get("forces") for record in member_records]
    summary = force_disagreement_summary(force_values)
    energies = [record.get("energy") for record in member_records]
    if all(value is not None for value in energies):
        summary["energy_disagreement_per_atom"] = energy_disagreement(energies, atom_count)
    else:
        summary["energy_disagreement_per_atom"] = None
    return summary


def evaluate_committee(
    predictions: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate and calibrate one or two same-framework committees."""

    policy_info = validate_policy(policy)
    predictions = _mapping(predictions, "committee_predictions")
    if (
        predictions.get("schema_version") != 1
        or predictions.get("contract") != PREDICTION_CONTRACT
    ):
        raise ActiveLearningError("committee predictions use an unsupported contract")
    if predictions.get("strategy") != policy.get("strategy"):
        raise ActiveLearningError("prediction strategy differs from the approved policy")
    units = _mapping(predictions.get("units"), "committee_predictions.units")
    if units.get("force") != policy_info["force_unit"]:
        raise ActiveLearningError("prediction and force-threshold units differ")
    splits = _validate_splits(predictions.get("dataset_split"))
    labels = _calibration_labels(predictions.get("calibration_labels"))
    if set(labels) != set(splits["calibration_ids"]):
        raise ActiveLearningError("calibration labels must match calibration_ids exactly")

    raw_models = _sequence(predictions.get("models"), "committee_predictions.models")
    models_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_models):
        model = _mapping(raw, f"models[{index}]")
        model_id = _plain(model.get("model_id"), f"models[{index}].model_id")
        if model_id in models_by_id:
            raise ActiveLearningError(f"duplicate model committee: {model_id}")
        models_by_id[model_id] = model
    if set(models_by_id) != set(policy_info["models"]):
        raise ActiveLearningError("prediction committees and policy models must match exactly")

    calibration_reports: dict[str, Any] = {}
    candidate_model_results: dict[str, dict[str, Any]] = {}
    shared_candidate_ids: set[str] | None = None
    shared_metadata: dict[str, dict[str, Any]] = {}
    member_evidence: dict[str, Any] = {}

    for model_id in policy_info["models"]:
        model = models_by_id[model_id]
        framework = _plain(model.get("framework"), f"{model_id}.framework")
        supports_stress = model.get("supports_stress")
        if not isinstance(supports_stress, bool):
            raise ActiveLearningError(f"{model_id}.supports_stress must be boolean")
        raw_members = _sequence(model.get("members"), f"{model_id}.members")
        members: list[dict[str, Any]] = []
        seen_member_ids = set()
        seen_seeds = set()
        for member_index, raw_member in enumerate(raw_members):
            member = _mapping(raw_member, f"{model_id}.members[{member_index}]")
            member_id = _plain(
                member.get("member_id"), f"{model_id}.members[{member_index}].member_id"
            )
            seed = member.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ActiveLearningError(f"{model_id}/{member_id}.seed must be an integer")
            if member_id in seen_member_ids or seed in seen_seeds:
                raise ActiveLearningError(f"{model_id} member ids and seeds must be unique")
            seen_member_ids.add(member_id)
            seen_seeds.add(seed)
            members.append(
                {"member_id": member_id, "seed": seed, "predictions": _prediction_map(member, model_id)}
            )
        if set(seen_seeds) != set(policy_info["seeds"][model_id]):
            raise ActiveLearningError(f"{model_id} member seeds differ from policy")
        sample_sets = [set(member["predictions"]) for member in members]
        if any(sample_set != sample_sets[0] for sample_set in sample_sets[1:]):
            raise ActiveLearningError(f"{model_id} committee members must predict identical sample ids")
        calibration_ids = {
            sample_id
            for sample_id, item in members[0]["predictions"].items()
            if item.get("split") == "calibration"
        }
        candidate_ids = sample_sets[0] - calibration_ids
        if calibration_ids != set(splits["calibration_ids"]):
            raise ActiveLearningError(f"{model_id} calibration predictions do not match split ids")
        if sample_sets[0] & (
            set(splits["train_ids"]) | set(splits["validation_ids"]) | set(splits["audit_ids"])
        ):
            raise ActiveLearningError("train/validation/audit samples entered committee evaluation")
        if shared_candidate_ids is None:
            shared_candidate_ids = set(candidate_ids)
        elif candidate_ids != shared_candidate_ids:
            raise ActiveLearningError("Strategy B models must evaluate identical candidate ids")

        calibration_samples = []
        for sample_id in sorted(calibration_ids):
            records = [member["predictions"][sample_id] for member in members]
            summary = _safe_force_evaluation(records)
            actual = committee_mean_force_error(
                [record.get("forces") for record in records], labels[sample_id].get("forces")
            )
            calibration_samples.append(
                {
                    "sample_id": sample_id,
                    "disagreement": summary["maximum_atomic_force_disagreement"],
                    "actual_error": actual,
                }
            )
        report = calibrate_force_uncertainty(
            calibration_samples,
            requested_coverage=policy_info["requested_coverage"],
            minimum_observed_coverage=policy_info["minimum_observed_coverage"],
            epsilon=policy_info["epsilon"],
            minimum_sample_count=policy_info["minimum_calibration_samples"],
            label_threshold=policy_info["label_threshold"],
            maximum_false_negative_rate=policy_info[
                "maximum_calibration_false_negative_rate"
            ],
        )
        report.update(
            {
                "model_id": model_id,
                "framework": framework,
                "member_count": len(members),
                "member_seeds": [member["seed"] for member in members],
                "force_unit": policy_info["force_unit"],
            }
        )
        calibration_reports[model_id] = report
        member_evidence[model_id] = {
            "framework": framework,
            "supports_stress": supports_stress,
            "member_count": len(members),
            "members": [
                {"member_id": member["member_id"], "seed": member["seed"]}
                for member in members
            ],
        }

        candidate_model_results[model_id] = {}
        for sample_id in sorted(candidate_ids):
            records = [member["predictions"][sample_id] for member in members]
            metadata = _candidate_metadata(records[0], sample_id)
            for record in records[1:]:
                if _candidate_metadata(record, sample_id) != metadata:
                    raise ActiveLearningError(
                        f"committee members disagree on candidate metadata for {sample_id}"
                    )
            if sample_id in shared_metadata and shared_metadata[sample_id] != metadata:
                raise ActiveLearningError(
                    f"Strategy B models disagree on candidate metadata for {sample_id}"
                )
            shared_metadata[sample_id] = metadata
            expected_condition = policy_info["conditions_by_id"].get(
                metadata["condition_id"]
            )
            if metadata["condition"] != expected_condition:
                raise ActiveLearningError(
                    f"candidate {sample_id} condition metadata differs from policy"
                )
            try:
                summary = _safe_force_evaluation(records)
                disagreement = summary["maximum_atomic_force_disagreement"]
                scale = report["scale_quantile"]
                if scale is None:
                    estimated = None
                    classification = None
                    out_of_range = None
                else:
                    estimated = float(scale) * disagreement
                    maximum_calibration = report["maximum_calibration_disagreement"]
                    out_of_range = disagreement > float(maximum_calibration)
                    classification = (
                        "UNSAFE"
                        if out_of_range
                        else classify_estimated_error(
                            estimated,
                            policy_info["label_threshold"],
                            policy_info["abort_threshold"],
                        )
                    )
                candidate_model_results[model_id][sample_id] = {
                    **summary,
                    "estimated_force_error": estimated,
                    "classification": classification,
                    "out_of_calibration_range": out_of_range,
                    "nonfinite_prediction": False,
                }
            except ActiveLearningError as exc:
                candidate_model_results[model_id][sample_id] = {
                    "per_atom_force_disagreement": None,
                    "maximum_atomic_force_disagreement": None,
                    "mean_atomic_force_disagreement": None,
                    "energy_disagreement_per_atom": None,
                    "atom_count": records[0].get("atom_count"),
                    "member_count": len(members),
                    "estimated_force_error": None,
                    "classification": "UNSAFE",
                    "out_of_calibration_range": True,
                    "nonfinite_prediction": True,
                    "diagnostic": str(exc),
                }

    calibration_ok = all(
        report["status"] == "PASS" for report in calibration_reports.values()
    )
    candidates = []
    condition_counts: dict[str, dict[str, int]] = {
        condition_id: {"total": 0, "SAFE": 0, "QUERY": 0, "UNSAFE": 0}
        for condition_id in policy_info["condition_ids"]
    }
    model_condition_counts: dict[str, dict[str, dict[str, int]]] = {
        model_id: {
            condition_id: {"total": 0, "SAFE": 0, "QUERY": 0, "UNSAFE": 0}
            for condition_id in policy_info["condition_ids"]
        }
        for model_id in policy_info["models"]
    }
    replica_counts: dict[tuple[str, str], dict[str, int]] = {}
    model_replica_counts: dict[str, dict[tuple[str, str], dict[str, int]]] = {
        model_id: {} for model_id in policy_info["models"]
    }
    for sample_id in sorted(shared_candidate_ids or set()):
        metadata = shared_metadata[sample_id]
        condition_id = metadata["condition_id"]
        if condition_id not in condition_counts:
            raise ActiveLearningError(
                f"candidate {sample_id} uses undeclared target condition {condition_id}"
            )
        model_results = {
            model_id: candidate_model_results[model_id][sample_id]
            for model_id in policy_info["models"]
        }
        classes = [record.get("classification") for record in model_results.values()]
        if not calibration_ok:
            classification = None
            combined_risk = None
            trigger_models: list[str] = []
        else:
            classification = max(classes, key=lambda value: RISK_ORDER[str(value)])
            numeric_risks = [
                float(record["estimated_force_error"])
                for record in model_results.values()
                if record.get("estimated_force_error") is not None
            ]
            combined_risk = max(numeric_risks) if numeric_risks else None
            trigger_models = [
                model_id
                for model_id, record in model_results.items()
                if record.get("classification") in {"QUERY", "UNSAFE"}
            ]
            validity = metadata["physical_validity"]
            if not validity["valid"]:
                classification = "UNSAFE"
        if classification is not None:
            condition_counts[condition_id]["total"] += 1
            condition_counts[condition_id][classification] += 1
            replica_key = (condition_id, metadata["replica"])
            combined_replica = replica_counts.setdefault(
                replica_key, {"total": 0, "SAFE": 0, "QUERY": 0, "UNSAFE": 0}
            )
            combined_replica["total"] += 1
            combined_replica[classification] += 1
            for model_id, record in model_results.items():
                model_class = str(record["classification"])
                model_condition_counts[model_id][condition_id]["total"] += 1
                model_condition_counts[model_id][condition_id][model_class] += 1
                model_replica = model_replica_counts[model_id].setdefault(
                    replica_key,
                    {"total": 0, "SAFE": 0, "QUERY": 0, "UNSAFE": 0},
                )
                model_replica["total"] += 1
                model_replica[model_class] += 1
        candidates.append(
            {
                "sample_id": sample_id,
                **metadata,
                "models": model_results,
                "combined_risk": combined_risk,
                "classification": classification,
                "trigger_models": trigger_models,
                "simultaneous_trigger": len(trigger_models) == len(policy_info["models"]),
                "nearest_prior_safe_boundary": None,
            }
        )

    by_trajectory: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_trajectory[(candidate["condition_id"], candidate["replica"])].append(candidate)
    for trajectory in by_trajectory.values():
        last_safe = None
        for candidate in sorted(
            trajectory, key=lambda item: (item["frame_index"], item["sample_id"])
        ):
            if candidate["classification"] == "SAFE":
                last_safe = {
                    "sample_id": candidate["sample_id"],
                    "frame_index": candidate["frame_index"],
                    "combined_risk": candidate["combined_risk"],
                }
            elif candidate["classification"] == "UNSAFE":
                candidate["nearest_prior_safe_boundary"] = last_safe

    per_condition = []
    for condition_id in policy_info["condition_ids"]:
        counts = condition_counts[condition_id]
        total = counts["total"]
        model_records = {}
        for model_id in policy_info["models"]:
            model_counts = model_condition_counts[model_id][condition_id]
            model_total = model_counts["total"]
            model_records[model_id] = {
                "candidate_count": model_total,
                "safe_count": model_counts["SAFE"],
                "query_count": model_counts["QUERY"],
                "unsafe_count": model_counts["UNSAFE"],
                "safe_fraction": model_counts["SAFE"] / model_total if model_total else None,
                "query_fraction": model_counts["QUERY"] / model_total if model_total else None,
                "unsafe_fraction": model_counts["UNSAFE"] / model_total if model_total else None,
            }
        replica_records = []
        condition_replicas = sorted(
            replica
            for candidate_condition, replica in replica_counts
            if candidate_condition == condition_id
        )
        for replica in condition_replicas:
            replica_key = (condition_id, replica)
            replica_count = replica_counts[replica_key]
            replica_total = replica_count["total"]
            replica_models = {}
            for model_id in policy_info["models"]:
                model_count = model_replica_counts[model_id][replica_key]
                model_total = model_count["total"]
                replica_models[model_id] = {
                    "candidate_count": model_total,
                    "safe_count": model_count["SAFE"],
                    "query_count": model_count["QUERY"],
                    "unsafe_count": model_count["UNSAFE"],
                    "safe_fraction": model_count["SAFE"] / model_total,
                    "query_fraction": model_count["QUERY"] / model_total,
                    "unsafe_fraction": model_count["UNSAFE"] / model_total,
                }
            replica_records.append(
                {
                    "replica": replica,
                    "candidate_count": replica_total,
                    "safe_count": replica_count["SAFE"],
                    "query_count": replica_count["QUERY"],
                    "unsafe_count": replica_count["UNSAFE"],
                    "safe_fraction": replica_count["SAFE"] / replica_total,
                    "query_fraction": replica_count["QUERY"] / replica_total,
                    "unsafe_fraction": replica_count["UNSAFE"] / replica_total,
                    "models": replica_models,
                }
            )
        per_condition.append(
            {
                "condition_id": condition_id,
                "condition": policy_info["conditions_by_id"][condition_id],
                "candidate_count": total,
                "safe_count": counts["SAFE"],
                "query_count": counts["QUERY"],
                "unsafe_count": counts["UNSAFE"],
                "safe_fraction": counts["SAFE"] / total if total else None,
                "query_fraction": counts["QUERY"] / total if total else None,
                "unsafe_fraction": counts["UNSAFE"] / total if total else None,
                "models": model_records,
                "replicas": replica_records,
            }
        )

    blocked_reasons = [
        f"{model_id}: {reason}"
        for model_id, report in calibration_reports.items()
        for reason in report["failed_gates"]
    ]
    out_of_range_counts = {
        model_id: sum(
            record.get("out_of_calibration_range") is True
            for record in candidate_model_results[model_id].values()
        )
        for model_id in policy_info["models"]
    }
    return {
        "schema_version": 1,
        "contract": EVALUATION_CONTRACT,
        "plugin_id": "active-learning",
        "operation": "committee-evaluate",
        "status": "OK",
        "evaluation_status": "READY" if calibration_ok else "BLOCKED_CALIBRATION",
        "strategy": dict(_mapping(policy.get("strategy"), "policy.strategy")),
        "policy_id": policy_info["policy_id"],
        "target_domain": dict(_mapping(policy.get("target_domain"), "policy.target_domain")),
        "dataset_split": splits,
        "units": {"force": policy_info["force_unit"], "energy": units.get("energy")},
        "force_disagreement_definition": (
            "sqrt(mean_over_members(||F_i_member - mean_member(F_i)||_2^2)); "
            "structure=max_over_atoms; population denominator M"
        ),
        "energy_disagreement_definition": "population_std(total_energy)/atom_count",
        "calibration_method": (
            "nearest-rank quantile of actual_force_error/(force_disagreement+epsilon); "
            "estimated_force_error=scale_quantile*force_disagreement"
        ),
        "committees": member_evidence,
        "calibration": calibration_reports,
        "blocked_reasons": blocked_reasons,
        "out_of_calibration_range_counts": out_of_range_counts,
        "candidates": candidates,
        "per_condition": per_condition,
    }


def _candidate_manifest(raw: Any) -> dict[str, dict[str, Any]]:
    manifest = _mapping(raw, "candidate_manifest")
    if manifest.get("schema_version") != 1 or manifest.get("contract") != CANDIDATE_CONTRACT:
        raise ActiveLearningError("candidate manifest uses an unsupported contract")
    result: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(_sequence(manifest.get("candidates"), "candidates")):
        item = dict(_mapping(value, f"candidates[{index}]"))
        sample_id = _plain(item.get("id"), f"candidates[{index}].id")
        if sample_id in result:
            raise ActiveLearningError(f"duplicate candidate manifest id: {sample_id}")
        _plain(item.get("structure_path"), f"{sample_id}.structure_path")
        result[sample_id] = item
    return result


def _selection_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    selection = _mapping(policy.get("selection"), "policy.selection")
    time_stride = _positive_int(selection.get("time_stride"), "selection.time_stride")
    maximum_labels = _positive_int(
        selection.get("maximum_labels_per_round"), "selection.maximum_labels_per_round"
    )
    maximum_total = _positive_int(
        selection.get("maximum_total_labels"), "selection.maximum_total_labels"
    )
    spot_count = _nonnegative_int(
        selection.get("safe_spot_checks_per_round"),
        "selection.safe_spot_checks_per_round",
    )
    if spot_count > maximum_labels:
        raise ActiveLearningError(
            "selection.safe_spot_checks_per_round cannot exceed maximum_labels_per_round"
        )
    if maximum_labels > maximum_total:
        raise ActiveLearningError(
            "selection.maximum_labels_per_round cannot exceed maximum_total_labels"
        )
    spot_seed = selection.get("safe_spot_check_seed")
    if isinstance(spot_seed, bool) or not isinstance(spot_seed, int):
        raise ActiveLearningError("selection.safe_spot_check_seed must be an integer")
    include_spot = selection.get("include_safe_spot_checks_in_training")
    if not isinstance(include_spot, bool):
        raise ActiveLearningError(
            "selection.include_safe_spot_checks_in_training must be boolean"
        )
    condition_quotas_raw = _mapping(
        selection.get("per_condition_quota"), "selection.per_condition_quota"
    )
    replica_quotas_raw = _mapping(
        selection.get("per_replica_quota"), "selection.per_replica_quota"
    )
    condition_quotas = {
        str(key): _positive_int(value, f"per_condition_quota.{key}")
        for key, value in condition_quotas_raw.items()
    }
    replica_quotas = {
        str(key): _positive_int(value, f"per_replica_quota.{key}")
        for key, value in replica_quotas_raw.items()
    }
    return {
        "time_stride": time_stride,
        "maximum_labels_per_round": maximum_labels,
        "maximum_total_labels": maximum_total,
        "safe_spot_checks_per_round": spot_count,
        "safe_spot_check_seed": spot_seed,
        "include_safe_spot_checks_in_training": include_spot,
        "per_condition_quota": condition_quotas,
        "per_replica_quota": replica_quotas,
    }


def _risk_sort_key(candidate: Mapping[str, Any]) -> tuple[float, str]:
    risk = candidate.get("combined_risk")
    numeric = _number(risk, f"{candidate.get('sample_id')}.combined_risk")
    return (-numeric, str(candidate["sample_id"]))


def _keep_highest_risk(
    candidates: Sequence[Mapping[str, Any]], key_name: str
) -> tuple[list[Mapping[str, Any]], list[dict[str, str]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[str(candidate[key_name])].append(candidate)
    kept = []
    removed = []
    for group in sorted(grouped):
        ordered = sorted(grouped[group], key=_risk_sort_key)
        kept.append(ordered[0])
        for item in ordered[1:]:
            removed.append(
                {
                    "sample_id": str(item["sample_id"]),
                    "reason": f"duplicate:{key_name}",
                    "kept_sample_id": str(ordered[0]["sample_id"]),
                }
            )
    return kept, removed


def _prefilter_query_candidates(
    evaluation_candidates: Sequence[Mapping[str, Any]], selection_policy: Mapping[str, Any]
) -> tuple[list[Mapping[str, Any]], list[dict[str, str]]]:
    rejected = []
    query = []
    for candidate in evaluation_candidates:
        validity = _mapping(candidate.get("physical_validity"), "physical_validity")
        if candidate.get("classification") != "QUERY":
            continue
        if not validity.get("valid") or validity.get("severe"):
            rejected.append(
                {"sample_id": str(candidate["sample_id"]), "reason": "physical-invalid"}
            )
            continue
        query.append(candidate)

    stride_kept = []
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in query:
        groups[(str(candidate["condition_id"]), str(candidate["replica"]))].append(candidate)
    for group in sorted(groups):
        ordered = sorted(
            groups[group], key=lambda item: (int(item["frame_index"]), str(item["sample_id"]))
        )
        keep_ids = {
            item["sample_id"]
            for index, item in enumerate(ordered)
            if index % int(selection_policy["time_stride"]) == 0
        }
        for item in ordered:
            if item["sample_id"] in keep_ids:
                stride_kept.append(item)
            else:
                rejected.append(
                    {"sample_id": str(item["sample_id"]), "reason": "time-downsampling"}
                )

    near_kept, near_removed = _keep_highest_risk(stride_kept, "near_duplicate_group")
    rejected.extend(near_removed)
    return sorted(near_kept, key=lambda item: str(item["sample_id"])), rejected


def _direct_selected_ids(
    direct_selection: Mapping[str, Any] | None, expected_input_ids: list[str]
) -> tuple[list[str], dict[str, Any]]:
    if not expected_input_ids:
        return [], {"method": "DIRECT", "input_count": 0, "selected_count": 0}
    if direct_selection is None:
        raise ActiveLearningError("QUERY candidates require a reviewed DIRECT selection result")
    direct = _mapping(direct_selection, "direct_selection")
    if direct.get("method") != "DIRECT":
        raise ActiveLearningError("diversity selection method must be DIRECT")
    declared_inputs = direct.get("input_candidate_ids")
    if declared_inputs is not None:
        input_ids = [_plain(value, "direct input id") for value in _sequence(declared_inputs, "input_candidate_ids")]
        if input_ids != expected_input_ids:
            raise ActiveLearningError("DIRECT input candidate ids differ from deterministic prefilter")
    if direct.get("selected_candidate_ids") is not None:
        selected = [
            _plain(value, "direct selected id")
            for value in _sequence(direct.get("selected_candidate_ids"), "selected_candidate_ids")
        ]
    else:
        indexes = [
            _nonnegative_int(value, "direct selected input index")
            for value in _sequence(direct.get("selected_input_indexes"), "selected_input_indexes")
        ]
        if any(index >= len(expected_input_ids) for index in indexes):
            raise ActiveLearningError("DIRECT selected an out-of-range input index")
        selected = [expected_input_ids[index] for index in indexes]
    if len(selected) != len(set(selected)) or not set(selected) <= set(expected_input_ids):
        raise ActiveLearningError("DIRECT selected ids must be a unique subset of its inputs")
    return selected, {
        "method": "DIRECT",
        "input_count": len(expected_input_ids),
        "selected_count": len(selected),
        "parameters": dict(_mapping(direct.get("parameters", {}), "direct.parameters")),
    }


def _deterministic_spot_checks(
    candidates: Sequence[Mapping[str, Any]], count: int, seed: int
) -> list[Mapping[str, Any]]:
    if count == 0:
        return []
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        validity = _mapping(candidate.get("physical_validity"), "physical_validity")
        if candidate.get("classification") == "SAFE" and validity.get("valid"):
            groups[str(candidate["condition_id"])].append(candidate)
    ordered_groups = []
    sampler = random.Random(seed)
    for condition_id in sorted(groups):
        group = sorted(groups[condition_id], key=lambda item: str(item["sample_id"]))
        sampler.shuffle(group)
        ordered_groups.append(group)
    selected = []
    while ordered_groups and len(selected) < count:
        remaining = []
        for group in ordered_groups:
            if group and len(selected) < count:
                selected.append(group.pop(0))
            if group:
                remaining.append(group)
        ordered_groups = remaining
    return selected


def select_candidates(
    evaluation: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    direct_selection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Apply the declared prefilter, DIRECT evidence, quotas, and DFT budget."""

    policy_info = validate_policy(policy)
    selection_policy = _selection_policy(policy)
    evaluation = _mapping(evaluation, "committee_evaluation")
    if evaluation.get("contract") != EVALUATION_CONTRACT:
        raise ActiveLearningError("committee evaluation uses an unsupported contract")
    if evaluation.get("policy_id") != policy_info["policy_id"]:
        raise ActiveLearningError("committee evaluation and policy identities differ")
    if evaluation.get("strategy") != policy.get("strategy"):
        raise ActiveLearningError("committee evaluation strategy drift")
    manifest_candidates = _candidate_manifest(candidate_manifest)
    raw_evaluation_candidates = _sequence(evaluation.get("candidates"), "evaluation.candidates")
    evaluation_candidates = [
        dict(_mapping(value, f"evaluation.candidates[{index}]"))
        for index, value in enumerate(raw_evaluation_candidates)
    ]
    evaluation_ids = [str(item.get("sample_id")) for item in evaluation_candidates]
    if len(evaluation_ids) != len(set(evaluation_ids)) or set(evaluation_ids) != set(
        manifest_candidates
    ):
        raise ActiveLearningError("evaluation and candidate manifest ids must match exactly")
    if evaluation.get("evaluation_status") != "READY":
        return {
            "schema_version": 1,
            "contract": SELECTION_CONTRACT,
            "plugin_id": "active-learning",
            "operation": "select-candidates",
            "status": "OK",
            "selection_status": "BLOCKED_CALIBRATION",
            "strategy": dict(_mapping(policy.get("strategy"), "policy.strategy")),
            "policy_id": policy_info["policy_id"],
            "selected_query_candidates": [],
            "selected_safe_spot_checks": [],
            "include_safe_spot_checks_in_training": selection_policy[
                "include_safe_spot_checks_in_training"
            ],
            "selected_for_dft": [],
            "counts": {
                "candidate_count": len(evaluation_candidates),
                "query_count": 0,
                "unique_query_count": 0,
                "direct_selected_count": 0,
                "dft_selected_count": 0,
            },
            "blocked_reasons": list(evaluation.get("blocked_reasons", [])),
        }

    prefiltered, rejected = _prefilter_query_candidates(
        evaluation_candidates, selection_policy
    )
    expected_direct_ids = [str(item["sample_id"]) for item in prefiltered]
    direct_ids, direct_report = _direct_selected_ids(direct_selection, expected_direct_ids)
    direct_candidates = {
        str(item["sample_id"]): item for item in prefiltered if item["sample_id"] in direct_ids
    }
    ordered = sorted(direct_candidates.values(), key=_risk_sort_key)
    spot_checks = _deterministic_spot_checks(
        evaluation_candidates,
        int(selection_policy["safe_spot_checks_per_round"]),
        int(selection_policy["safe_spot_check_seed"]),
    )
    query_budget = selection_policy["maximum_labels_per_round"] - len(spot_checks)
    selected_query = []
    selected_query_ids: set[str] = set()
    condition_counts: dict[str, int] = defaultdict(int)
    replica_counts: dict[str, int] = defaultdict(int)

    def quota_reason(candidate: Mapping[str, Any]) -> str | None:
        condition_id = str(candidate["condition_id"])
        replica = str(candidate["replica"])
        condition_limit = selection_policy["per_condition_quota"].get(
            condition_id, selection_policy["maximum_labels_per_round"]
        )
        replica_limit = selection_policy["per_replica_quota"].get(
            replica, selection_policy["maximum_labels_per_round"]
        )
        if condition_counts[condition_id] >= condition_limit:
            return "per-condition-quota"
        if replica_counts[replica] >= replica_limit:
            return "per-replica-quota"
        return None

    def add_candidate(candidate: Mapping[str, Any]) -> None:
        sample_id = str(candidate["sample_id"])
        selected_query.append(candidate)
        selected_query_ids.add(sample_id)
        condition_counts[str(candidate["condition_id"])] += 1
        replica_counts[str(candidate["replica"])] += 1

    available_trigger_models = sorted(
        {
            model_id
            for candidate in ordered
            for model_id in candidate.get("trigger_models", [])
            if model_id in policy_info["models"]
        }
    )
    uncovered_trigger_models = set(available_trigger_models)
    if policy_info["mode"] == STRATEGY_DUAL:
        while uncovered_trigger_models and len(selected_query) < query_budget:
            eligible = [
                candidate
                for candidate in ordered
                if str(candidate["sample_id"]) not in selected_query_ids
                and set(candidate.get("trigger_models", [])) & uncovered_trigger_models
                and quota_reason(candidate) is None
            ]
            if not eligible:
                break
            selected = min(
                eligible,
                key=lambda candidate: (
                    -len(
                        set(candidate.get("trigger_models", []))
                        & uncovered_trigger_models
                    ),
                    *_risk_sort_key(candidate),
                ),
            )
            add_candidate(selected)
            uncovered_trigger_models -= set(selected.get("trigger_models", []))

    for candidate in ordered:
        if str(candidate["sample_id"]) in selected_query_ids:
            continue
        reason = quota_reason(candidate)
        if reason is not None:
            rejected.append(
                {"sample_id": str(candidate["sample_id"]), "reason": reason}
            )
            continue
        if len(selected_query) >= query_budget:
            rejected.append(
                {"sample_id": str(candidate["sample_id"]), "reason": "round-dft-budget"}
            )
            continue
        add_candidate(candidate)

    selected_trigger_models = sorted(
        {
            model_id
            for candidate in selected_query
            for model_id in candidate.get("trigger_models", [])
            if model_id in policy_info["models"]
        }
    )
    missing_trigger_models = sorted(
        set(available_trigger_models) - set(selected_trigger_models)
    )

    def selected_record(candidate: Mapping[str, Any], selection_kind: str) -> dict[str, Any]:
        sample_id = str(candidate["sample_id"])
        manifest = manifest_candidates[sample_id]
        return {
            "sample_id": sample_id,
            "selection_kind": selection_kind,
            "classification": candidate["classification"],
            "combined_risk": candidate["combined_risk"],
            "trigger_models": list(candidate["trigger_models"]),
            "condition_id": candidate["condition_id"],
            "replica": candidate["replica"],
            "frame_index": candidate["frame_index"],
            "structure_path": manifest["structure_path"],
        }

    query_records = [selected_record(item, "QUERY") for item in selected_query]
    spot_records = [selected_record(item, "SAFE_SPOT_CHECK") for item in spot_checks]
    severe_ids = [
        str(item["sample_id"])
        for item in evaluation_candidates
        if _mapping(item.get("physical_validity"), "physical_validity").get("severe")
    ]
    unsafe_count = sum(item.get("classification") == "UNSAFE" for item in evaluation_candidates)
    return {
        "schema_version": 1,
        "contract": SELECTION_CONTRACT,
        "plugin_id": "active-learning",
        "operation": "select-candidates",
        "status": "OK",
        "selection_status": "READY_FOR_DFT",
        "strategy": dict(_mapping(policy.get("strategy"), "policy.strategy")),
        "policy_id": policy_info["policy_id"],
        "selection_order": [
            "physical-validity",
            "calibrated-uncertainty",
            "time-downsampling",
            "exact-duplicate-removal",
            "near-duplicate-removal",
            "target-condition-grouping",
            "DIRECT",
            "trigger-model-coverage",
            "per-condition-and-replica-quota",
            "round-dft-budget",
        ],
        "diversity_selection": direct_report,
        "trigger_model_coverage": {
            "available_trigger_models": available_trigger_models,
            "selected_trigger_models": selected_trigger_models,
            "missing_trigger_models": missing_trigger_models,
            "status": "COMPLETE" if not missing_trigger_models else "INCOMPLETE",
        },
        "direct_input_candidate_ids": expected_direct_ids,
        "direct_selected_candidate_ids": direct_ids,
        "selected_query_candidates": query_records,
        "selected_safe_spot_checks": spot_records,
        "include_safe_spot_checks_in_training": selection_policy[
            "include_safe_spot_checks_in_training"
        ],
        "selected_for_dft": query_records + spot_records,
        "severe_invalid_not_sent_to_dft": sorted(severe_ids),
        "rejected": sorted(rejected, key=lambda item: (item["reason"], item["sample_id"])),
        "counts": {
            "candidate_count": len(evaluation_candidates),
            "query_count": sum(
                item.get("classification") == "QUERY" for item in evaluation_candidates
            ),
            "unsafe_count": unsafe_count,
            "unique_query_count": len(prefiltered),
            "direct_selected_count": len(direct_ids),
            "query_dft_selected_count": len(query_records),
            "safe_spot_check_count": len(spot_records),
            "dft_selected_count": len(query_records) + len(spot_records),
        },
        "budgets": {
            "maximum_labels_per_round": selection_policy["maximum_labels_per_round"],
            "maximum_total_labels": selection_policy["maximum_total_labels"],
        },
    }


def _audit_records(
    raw: Any, expected_audit_ids: Sequence[str]
) -> dict[tuple[str, str], dict[str, Any]]:
    manifest = _mapping(raw, "audit_benchmark")
    if manifest.get("schema_version") != 1:
        raise ActiveLearningError("audit benchmark must use schema_version=1")
    audit_ids = [
        _plain(value, "audit_benchmark.audit_ids")
        for value in _sequence(manifest.get("audit_ids"), "audit_benchmark.audit_ids")
    ]
    if audit_ids != list(expected_audit_ids):
        raise ActiveLearningError(
            "audit benchmark sample IDs differ from the immutable audit split"
        )
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for index, value in enumerate(_sequence(manifest.get("records"), "audit_benchmark.records")):
        item = dict(_mapping(value, f"audit_benchmark.records[{index}]"))
        model = _plain(item.get("model"), f"audit record {index}.model")
        metric = _plain(item.get("metric"), f"audit record {index}.metric")
        _number(item.get("value"), f"audit {model}/{metric}.value")
        _plain(item.get("unit"), f"audit {model}/{metric}.unit")
        if item.get("split") not in {"audit", "test"}:
            raise ActiveLearningError("audit benchmark records must use audit/test split")
        aggregation = item.get("aggregation")
        if aggregation is not None:
            if aggregation != "maximum-over-committee-members":
                raise ActiveLearningError("audit benchmark uses an unsupported aggregation")
            member_values = _mapping(
                item.get("member_values"),
                f"audit {model}/{metric}.member_values",
            )
            normalized_values = [
                _number(value, f"audit {model}/{metric}.member_values.{member}")
                for member, value in member_values.items()
            ]
            if not normalized_values or not math.isclose(
                max(normalized_values),
                float(item["value"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ActiveLearningError(
                    "audit aggregate differs from maximum committee-member error"
                )
        key = (model, metric)
        if key in records:
            raise ActiveLearningError(f"duplicate audit metric: {model}/{metric}")
        records[key] = item
    return records


def _assessment_policy(policy: Mapping[str, Any], models: Sequence[str]) -> dict[str, Any]:
    assessment = _mapping(policy.get("assessment"), "policy.assessment")
    raw_thresholds = _sequence(assessment.get("audit_thresholds"), "assessment.audit_thresholds")
    thresholds = []
    required_base = {
        (model, metric)
        for model in models
        for metric in COMMON_AUDIT_METRICS
    }
    for index, value in enumerate(raw_thresholds):
        item = _mapping(value, f"assessment.audit_thresholds[{index}]")
        model = _plain(item.get("model"), f"audit_thresholds[{index}].model")
        metric = _plain(item.get("metric"), f"audit_thresholds[{index}].metric")
        if model not in models:
            raise ActiveLearningError("audit threshold model must be a strategy model")
        if metric not in COMMON_AUDIT_METRICS:
            raise ActiveLearningError(
                "first-version audit gates support only common energy/force metrics"
            )
        maximum = _positive_number(item.get("maximum"), f"{model}/{metric}.maximum")
        unit = _plain(item.get("unit"), f"{model}/{metric}.unit")
        thresholds.append({"model": model, "metric": metric, "maximum": maximum, "unit": unit})
    declared = {(item["model"], item["metric"]) for item in thresholds}
    if len(declared) != len(thresholds):
        raise ActiveLearningError("audit thresholds contain duplicate model/metric gates")
    if not required_base <= declared:
        missing = sorted(f"{model}/{metric}" for model, metric in required_base - declared)
        raise ActiveLearningError("audit thresholds are missing: " + ", ".join(missing))
    condition_gates_raw = _mapping(
        assessment.get("condition_gates"), "assessment.condition_gates"
    )
    condition_gates = {}
    for condition, value in condition_gates_raw.items():
        gate = _mapping(value, f"condition_gates.{condition}")
        condition_gates[str(condition)] = {
            "maximum_query_fraction": _fraction(
                gate.get("maximum_query_fraction"),
                f"condition_gates.{condition}.maximum_query_fraction",
            ),
            "maximum_unsafe_fraction": _fraction(
                gate.get("maximum_unsafe_fraction"),
                f"condition_gates.{condition}.maximum_unsafe_fraction",
            ),
        }
    return {
        "audit_thresholds": thresholds,
        "condition_gates": condition_gates,
        "maximum_spot_check_false_negative_rate": _fraction(
            assessment.get("maximum_spot_check_false_negative_rate"),
            "assessment.maximum_spot_check_false_negative_rate",
        ),
        "required_consecutive_rounds": _positive_int(
            assessment.get("required_consecutive_rounds"),
            "assessment.required_consecutive_rounds",
        ),
    }


def _spot_check_report(
    raw: Any,
    selected_spot_ids: set[str],
    models: Sequence[str],
    label_threshold: float,
    unit: str,
    maximum_rate: float,
) -> dict[str, Any]:
    manifest = _mapping(raw, "spot_checks")
    samples = []
    observed_pairs = set()
    false_negative_count = 0
    per_model_counts = {
        model: {"sample_count": 0, "false_negative_count": 0} for model in models
    }
    for index, value in enumerate(_sequence(manifest.get("samples"), "spot_checks.samples")):
        item = _mapping(value, f"spot_checks.samples[{index}]")
        sample_id = _plain(item.get("sample_id"), f"spot_checks[{index}].sample_id")
        model = _plain(item.get("model"), f"spot_checks[{index}].model")
        if sample_id not in selected_spot_ids or model not in models:
            raise ActiveLearningError("spot-check evidence references an unexpected sample/model")
        pair = (sample_id, model)
        if pair in observed_pairs:
            raise ActiveLearningError("duplicate spot-check sample/model evidence")
        observed_pairs.add(pair)
        if item.get("unit") != unit:
            raise ActiveLearningError("spot-check force unit differs from policy")
        predicted = _number(item.get("predicted_force_error"), "predicted_force_error")
        actual = _number(item.get("actual_force_error"), "actual_force_error")
        false_negative = predicted <= label_threshold < actual
        false_negative_count += int(false_negative)
        per_model_counts[model]["sample_count"] += 1
        per_model_counts[model]["false_negative_count"] += int(false_negative)
        samples.append(
            {
                "sample_id": sample_id,
                "model": model,
                "predicted_force_error": predicted,
                "actual_force_error": actual,
                "false_negative": false_negative,
            }
        )
    expected_pairs = {(sample_id, model) for sample_id in selected_spot_ids for model in models}
    missing = expected_pairs - observed_pairs
    rate = false_negative_count / len(samples) if samples else 0.0
    failed = []
    if missing:
        failed.append(f"missing {len(missing)} SAFE spot-check model results")
    per_model = {}
    for model in models:
        counts = per_model_counts[model]
        model_rate = (
            counts["false_negative_count"] / counts["sample_count"]
            if counts["sample_count"]
            else 0.0
        )
        passed = counts["sample_count"] == len(selected_spot_ids) and model_rate <= maximum_rate
        per_model[model] = {**counts, "false_negative_rate": model_rate, "passed": passed}
        if model_rate > maximum_rate:
            failed.append(
                f"{model} SAFE spot-check false-negative rate "
                f"{model_rate:.12g} exceeds {maximum_rate:.12g}"
            )
    return {
        "sample_model_count": len(samples),
        "selected_structure_count": len(selected_spot_ids),
        "false_negative_count": false_negative_count,
        "false_negative_rate": rate,
        "per_model": per_model,
        "samples": samples,
        "status": "PASS" if not failed else "FAIL",
        "failed_gates": failed,
    }


def _validate_campaign(
    campaign: Any,
    round_index: int,
    policy: Mapping[str, Any],
    current_split: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _mapping(campaign, "campaign")
    campaign_id = _plain(raw.get("campaign_id"), "campaign.campaign_id")
    policy_id = _plain(raw.get("policy_id"), "campaign.policy_id")
    if policy_id != policy.get("policy_id"):
        raise ActiveLearningError("campaign policy_id differs from policy")
    if raw.get("strategy") != policy.get("strategy"):
        raise ActiveLearningError("campaign strategy differs from policy")
    if raw.get("target_domain") != policy.get("target_domain"):
        raise ActiveLearningError("campaign target domain differs from policy")
    cumulative = _mapping(
        raw.get("current_cumulative_dataset"),
        "campaign.current_cumulative_dataset",
    )
    cumulative_record = {
        "dataset_id": _plain(
            cumulative.get("dataset_id"),
            "campaign.current_cumulative_dataset.dataset_id",
        ),
        "split_id": _plain(
            cumulative.get("split_id"),
            "campaign.current_cumulative_dataset.split_id",
        ),
        "reference": _plain(
            cumulative.get("reference"),
            "campaign.current_cumulative_dataset.reference",
        ),
    }
    if (
        cumulative_record["dataset_id"] != current_split.get("dataset_id")
        or cumulative_record["split_id"] != current_split.get("split_id")
    ):
        raise ActiveLearningError(
            "campaign cumulative dataset record differs from assessment split"
        )
    current_decision = _plain(raw.get("current_decision"), "campaign.current_decision")
    if current_decision != "PENDING":
        raise ActiveLearningError(
            "campaign current_decision must be PENDING before round assessment"
        )
    rounds = _sequence(raw.get("rounds"), "campaign.rounds")
    normalized = []
    for index, value in enumerate(rounds):
        item = _mapping(value, f"campaign.rounds[{index}]")
        declared_index = _nonnegative_int(item.get("index"), f"campaign.rounds[{index}].index")
        directory = _plain(item.get("directory"), f"campaign.rounds[{index}].directory")
        expected_directory = f"round-{declared_index:03d}"
        if declared_index != index or directory != expected_directory:
            raise ActiveLearningError("campaign rounds must be sequential round-NNN directories")
        normalized.append({"index": declared_index, "directory": directory})
    if not normalized or normalized[-1]["index"] != round_index:
        raise ActiveLearningError("current round must be the final declared campaign round")
    return {
        "campaign_id": campaign_id,
        "policy_id": policy_id,
        "rounds": normalized,
        "current_round": round_index,
        "current_cumulative_dataset": cumulative_record,
        "current_decision": current_decision,
    }


def _candidate_uncertainty_distribution(
    evaluation: Mapping[str, Any], models: Sequence[str]
) -> dict[str, Any]:
    candidates = _sequence(evaluation.get("candidates"), "evaluation.candidates")
    combined = []
    model_values = {model: [] for model in models}
    for index, raw in enumerate(candidates):
        candidate = _mapping(raw, f"evaluation.candidates[{index}]")
        if candidate.get("combined_risk") is not None:
            combined.append(
                _number(candidate["combined_risk"], "candidate.combined_risk")
            )
        records = _mapping(candidate.get("models"), "candidate.models")
        if set(records) != set(models):
            raise ActiveLearningError(
                "candidate uncertainty records must contain every strategy model"
            )
        for model in models:
            record = _mapping(records[model], f"candidate.models.{model}")
            value = record.get("estimated_force_error")
            if value is not None:
                model_values[model].append(
                    _number(value, f"candidate.models.{model}.estimated_force_error")
                )
    units = _mapping(evaluation.get("units"), "evaluation.units")
    return {
        "unit": _plain(units.get("force"), "evaluation.units.force"),
        "quantile_method": "nearest-rank-ceil",
        "combined_risk": _distribution_summary(combined),
        "models": {
            model: _distribution_summary(model_values[model]) for model in models
        },
    }


def _uncertainty_distribution_change(
    current: Mapping[str, Any], previous: Any
) -> dict[str, Any]:
    if previous is None:
        return {"status": "NOT_AVAILABLE", "reason": "no prior round"}
    prior = _mapping(previous, "previous uncertainty_distribution")
    if prior.get("unit") != current.get("unit"):
        raise ActiveLearningError("uncertainty distribution unit changed across rounds")
    current_models = _mapping(current.get("models"), "current uncertainty models")
    prior_models = _mapping(prior.get("models"), "previous uncertainty models")
    if set(current_models) != set(prior_models):
        raise ActiveLearningError("uncertainty distribution models changed across rounds")

    def delta(current_stats: Any, prior_stats: Any, field: str) -> dict[str, Any]:
        current_record = _mapping(current_stats, f"current uncertainty {field}")
        prior_record = _mapping(prior_stats, f"previous uncertainty {field}")
        result = {}
        for statistic in ("mean", "q90", "maximum"):
            current_value = current_record.get(statistic)
            prior_value = prior_record.get(statistic)
            result[f"{statistic}_delta"] = (
                None
                if current_value is None or prior_value is None
                else _number(current_value, statistic) - _number(prior_value, statistic)
            )
        return result

    return {
        "status": "AVAILABLE",
        "combined_risk": delta(
            current.get("combined_risk"), prior.get("combined_risk"), "combined_risk"
        ),
        "models": {
            model: delta(current_models[model], prior_models[model], model)
            for model in current_models
        },
    }


def assess_round(
    evaluation: Mapping[str, Any],
    selection: Mapping[str, Any],
    audit_benchmark: Mapping[str, Any],
    spot_checks: Mapping[str, Any],
    labeling_result: Mapping[str, Any],
    dataset_split: Mapping[str, Any],
    round_history: Mapping[str, Any],
    campaign: Mapping[str, Any],
    policy: Mapping[str, Any],
    transport_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assess coverage, independent accuracy, and consecutive stability."""

    policy_info = validate_policy(policy)
    assessment_policy = _assessment_policy(policy, policy_info["models"])
    evaluation = _mapping(evaluation, "committee_evaluation")
    selection = _mapping(selection, "selection_result")
    if evaluation.get("contract") != EVALUATION_CONTRACT:
        raise ActiveLearningError("round assessment requires committee evaluation evidence")
    if selection.get("contract") != SELECTION_CONTRACT:
        raise ActiveLearningError("round assessment requires selection evidence")
    if evaluation.get("policy_id") != policy_info["policy_id"] or selection.get(
        "policy_id"
    ) != policy_info["policy_id"]:
        raise ActiveLearningError("round evidence and policy identities differ")
    evaluation_split = _validate_splits(evaluation.get("dataset_split"))
    current_split = _validate_splits(dataset_split)
    if current_split["calibration_ids"] != evaluation_split["calibration_ids"]:
        raise ActiveLearningError(
            "assessment calibration ids differ from committee evaluation"
        )
    if current_split["audit_ids"] != evaluation_split["audit_ids"]:
        raise ActiveLearningError(
            "assessment immutable audit ids differ from committee evaluation"
        )
    previous_training_ids = set(evaluation_split["train_ids"]) | set(
        evaluation_split["validation_ids"]
    )
    current_training_ids = set(current_split["train_ids"]) | set(
        current_split["validation_ids"]
    )
    if not previous_training_ids <= current_training_ids:
        raise ActiveLearningError(
            "assessment cumulative split dropped a prior training/validation record"
        )

    history = _mapping(round_history, "round_history")
    round_index = _nonnegative_int(history.get("current_round"), "round_history.current_round")
    previous_rounds = [
        dict(_mapping(value, f"round_history.rounds[{index}]"))
        for index, value in enumerate(
            _sequence(history.get("rounds", []), "round_history.rounds")
        )
    ]
    if len(previous_rounds) != round_index:
        raise ActiveLearningError("round history must contain exactly the prior rounds")
    for index, item in enumerate(previous_rounds):
        if item.get("round_index") != index:
            raise ActiveLearningError("round history indexes must be sequential")
        if item.get("audit_ids") != current_split["audit_ids"]:
            raise ActiveLearningError("immutable audit ids changed across rounds")
    lineage = _validate_campaign(campaign, round_index, policy, current_split)
    current_uncertainty = _candidate_uncertainty_distribution(
        evaluation, policy_info["models"]
    )
    previous_uncertainty = (
        previous_rounds[-1].get("uncertainty_distribution")
        if previous_rounds
        else None
    )
    uncertainty_change = _uncertainty_distribution_change(
        current_uncertainty, previous_uncertainty
    )

    audit = _audit_records(audit_benchmark, current_split["audit_ids"])
    audit_checks = []
    audit_failed = []
    current_audit_values = {}
    for threshold in assessment_policy["audit_thresholds"]:
        key = (threshold["model"], threshold["metric"])
        record = audit.get(key)
        if record is None:
            audit_failed.append(f"missing immutable audit metric {key[0]}/{key[1]}")
            continue
        if record["unit"] != threshold["unit"]:
            raise ActiveLearningError(f"audit unit drift for {key[0]}/{key[1]}")
        value = float(record["value"])
        passed = value <= threshold["maximum"]
        current_audit_values[f"{key[0]}:{key[1]}"] = value
        audit_checks.append(
            {
                **threshold,
                "value": value,
                "aggregation": record.get("aggregation", "single-result"),
                "member_values": record.get("member_values"),
                "passed": passed,
            }
        )
        if not passed:
            audit_failed.append(
                f"{key[0]} {key[1]} {value:.12g} exceeds {threshold['maximum']:.12g}"
            )

    calibration_failed = []
    calibration_passed = evaluation.get("evaluation_status") == "READY"
    for model_id in policy_info["models"]:
        report = _mapping(
            _mapping(evaluation.get("calibration"), "evaluation.calibration").get(model_id),
            f"evaluation.calibration.{model_id}",
        )
        if report.get("status") != "PASS":
            calibration_passed = False
            reasons = list(report.get("failed_gates", []))
            calibration_failed.extend(
                f"{model_id}: {reason}" for reason in reasons
            )
            if not reasons:
                calibration_failed.append(f"{model_id}: calibration did not pass")

    condition_records = {
        str(item.get("condition_id")): item
        for item in _sequence(evaluation.get("per_condition"), "evaluation.per_condition")
        if isinstance(item, Mapping)
    }
    condition_checks = []
    condition_failed = []
    sampling_blocked = []
    if set(assessment_policy["condition_gates"]) != set(policy_info["condition_ids"]):
        raise ActiveLearningError("condition gates must cover every target condition exactly")
    for condition_id in policy_info["condition_ids"]:
        record = condition_records.get(condition_id)
        if record is None or record.get("candidate_count") == 0:
            sampling_blocked.append(f"target condition {condition_id} has no evaluated candidates")
            continue
        gate = assessment_policy["condition_gates"][condition_id]
        query_fraction = _fraction(record.get("query_fraction"), f"{condition_id}.query_fraction")
        unsafe_fraction = _fraction(record.get("unsafe_fraction"), f"{condition_id}.unsafe_fraction")
        passed = (
            query_fraction <= gate["maximum_query_fraction"]
            and unsafe_fraction <= gate["maximum_unsafe_fraction"]
        )
        raw_model_records = _mapping(record.get("models"), f"{condition_id}.models")
        if set(raw_model_records) != set(policy_info["models"]):
            raise ActiveLearningError(
                f"{condition_id} condition coverage must contain every strategy model"
            )
        model_checks = {}
        for model_id in policy_info["models"]:
            model_record = _mapping(
                raw_model_records[model_id], f"{condition_id}.models.{model_id}"
            )
            if model_record.get("candidate_count") != record["candidate_count"]:
                raise ActiveLearningError(
                    f"{condition_id}/{model_id} condition candidate count drift"
                )
            model_query = _fraction(
                model_record.get("query_fraction"),
                f"{condition_id}.{model_id}.query_fraction",
            )
            model_unsafe = _fraction(
                model_record.get("unsafe_fraction"),
                f"{condition_id}.{model_id}.unsafe_fraction",
            )
            model_passed = (
                model_query <= gate["maximum_query_fraction"]
                and model_unsafe <= gate["maximum_unsafe_fraction"]
            )
            model_checks[model_id] = {
                "candidate_count": model_record["candidate_count"],
                "safe_fraction": model_record["safe_fraction"],
                "query_fraction": model_query,
                "unsafe_fraction": model_unsafe,
                "passed": model_passed,
            }
            passed = passed and model_passed
            if model_query > gate["maximum_query_fraction"]:
                condition_failed.append(
                    f"{condition_id}/{model_id} query fraction exceeds policy"
                )
            if model_unsafe > gate["maximum_unsafe_fraction"]:
                condition_failed.append(
                    f"{condition_id}/{model_id} unsafe fraction exceeds policy"
                )
        condition_checks.append(
            {
                "condition_id": condition_id,
                "condition": record.get("condition"),
                "candidate_count": record["candidate_count"],
                "safe_fraction": record["safe_fraction"],
                "query_fraction": query_fraction,
                "unsafe_fraction": unsafe_fraction,
                "models": model_checks,
                "replicas": record.get("replicas", []),
                **gate,
                "passed": passed,
            }
        )
        if query_fraction > gate["maximum_query_fraction"]:
            condition_failed.append(
                f"{condition_id} query fraction exceeds policy"
            )
        if unsafe_fraction > gate["maximum_unsafe_fraction"]:
            condition_failed.append(
                f"{condition_id} unsafe fraction exceeds policy"
            )

    selected_spot = {
        str(item.get("sample_id"))
        for item in _sequence(
            selection.get("selected_safe_spot_checks", []),
            "selection.selected_safe_spot_checks",
        )
        if isinstance(item, Mapping)
    }
    spot_report = _spot_check_report(
        spot_checks,
        selected_spot,
        policy_info["models"],
        policy_info["label_threshold"],
        policy_info["force_unit"],
        assessment_policy["maximum_spot_check_false_negative_rate"],
    )

    labeling = _mapping(labeling_result, "labeling_result")
    labeling_status = labeling.get("status")
    successful_query_records = _sequence(
        labeling.get("successful_query_records"), "successful_query_records"
    )
    successful_query = set()
    successful_query_record_ids = set()
    for index, value in enumerate(successful_query_records):
        item = _mapping(value, f"successful_query_records[{index}]")
        sample_id = _plain(item.get("sample_id"), f"successful_query_records[{index}].sample_id")
        record_id = _plain(item.get("record_id"), f"successful_query_records[{index}].record_id")
        if sample_id in successful_query or record_id in successful_query_record_ids:
            raise ActiveLearningError("successful query sample and record ids must be unique")
        successful_query.add(sample_id)
        successful_query_record_ids.add(record_id)
    successful_spot_records = _sequence(
        labeling.get("successful_spot_check_records"),
        "successful_spot_check_records",
    )
    successful_spot = set()
    successful_spot_record_ids = set()
    for index, value in enumerate(successful_spot_records):
        item = _mapping(value, f"successful_spot_check_records[{index}]")
        sample_id = _plain(
            item.get("sample_id"), f"successful_spot_check_records[{index}].sample_id"
        )
        record_id = _plain(
            item.get("record_id"), f"successful_spot_check_records[{index}].record_id"
        )
        if sample_id in successful_spot or record_id in successful_spot_record_ids:
            raise ActiveLearningError(
                "successful spot-check sample and record ids must be unique"
            )
        successful_spot.add(sample_id)
        successful_spot_record_ids.add(record_id)
    selected_query = {
        str(item.get("sample_id"))
        for item in _sequence(
            selection.get("selected_query_candidates", []),
            "selection.selected_query_candidates",
        )
        if isinstance(item, Mapping)
    }
    labeling_failed = []
    cumulative_dataset_matches = (
        labeling.get("cumulative_dataset_id") == current_split["dataset_id"]
        and labeling.get("cumulative_split_id") == current_split["split_id"]
    )
    query_records_are_cumulative = successful_query_record_ids <= current_training_ids
    selection_policy = _selection_policy(policy)
    include_spot = selection_policy["include_safe_spot_checks_in_training"]
    if selection.get("include_safe_spot_checks_in_training") is not include_spot:
        raise ActiveLearningError("selection SAFE spot-check training policy drift")
    spot_membership_matches = (
        successful_spot_record_ids <= current_training_ids
        if include_spot
        else successful_spot_record_ids.isdisjoint(current_training_ids)
    )
    if (
        labeling_status != "OK"
        or successful_query != selected_query
        or successful_spot != selected_spot
        or not cumulative_dataset_matches
        or not query_records_are_cumulative
        or not spot_membership_matches
    ):
        labeling_failed.append(
            "DFT labeling or cumulative dataset handoff did not bind every selected structure"
        )
    trigger_coverage_failed = []
    if policy_info["mode"] == STRATEGY_DUAL:
        trigger_coverage = _mapping(
            selection.get("trigger_model_coverage"),
            "selection.trigger_model_coverage",
        )
        available_triggers = {
            _plain(value, "available_trigger_models")
            for value in _sequence(
                trigger_coverage.get("available_trigger_models"),
                "available_trigger_models",
            )
        }
        selected_triggers = {
            _plain(value, "selected_trigger_models")
            for value in _sequence(
                trigger_coverage.get("selected_trigger_models"),
                "selected_trigger_models",
            )
        }
        missing_triggers = {
            _plain(value, "missing_trigger_models")
            for value in _sequence(
                trigger_coverage.get("missing_trigger_models"),
                "missing_trigger_models",
            )
        }
        expected_missing = available_triggers - selected_triggers
        expected_status = "COMPLETE" if not expected_missing else "INCOMPLETE"
        if (
            missing_triggers != expected_missing
            or trigger_coverage.get("status") != expected_status
        ):
            raise ActiveLearningError(
                "Strategy B trigger-model coverage report is inconsistent"
            )
        if missing_triggers:
            trigger_coverage_failed.append(
                "Strategy B QUERY selection omitted available trigger models: "
                + ", ".join(sorted(missing_triggers))
            )
    cumulative_labels = _nonnegative_int(
        labeling.get("cumulative_dft_label_count"),
        "labeling_result.cumulative_dft_label_count",
    )
    new_labels = len(successful_query) + len(successful_spot)

    coverage_failed = (
        calibration_failed
        + condition_failed
        + spot_report["failed_gates"]
        + sampling_blocked
    )
    coverage_passed = (
        calibration_passed
        and not condition_failed
        and spot_report["status"] == "PASS"
        and not sampling_blocked
    )
    accuracy_passed = not audit_failed
    round_integrity_failed = labeling_failed + trigger_coverage_failed
    current_pes_gates_passed = (
        coverage_passed and accuracy_passed and not round_integrity_failed
    )
    consecutive = 1 if current_pes_gates_passed else 0
    if current_pes_gates_passed:
        for previous in reversed(previous_rounds):
            if previous.get("pes_gates_passed_this_round") is True:
                consecutive += 1
            else:
                break
    required_consecutive = assessment_policy["required_consecutive_rounds"]
    stability_failed = []
    if current_pes_gates_passed and consecutive < required_consecutive:
        stability_failed.append(
            f"passed consecutive rounds {consecutive} is below {required_consecutive}"
        )

    maximum_total = selection_policy["maximum_total_labels"]
    if not calibration_passed or evaluation.get("evaluation_status") == "BLOCKED_CALIBRATION":
        decision = "BLOCKED_CALIBRATION"
    elif sampling_blocked:
        decision = "BLOCKED_SAMPLING"
    elif labeling_failed:
        decision = "SCIENTIFIC_REVIEW_REQUIRED"
    elif current_pes_gates_passed and consecutive >= required_consecutive:
        decision = "CONVERGED_FOR_DECLARED_DOMAIN"
    elif cumulative_labels >= maximum_total:
        decision = "BUDGET_EXHAUSTED"
    else:
        decision = "CONTINUE"
    if decision not in DECISIONS:
        raise AssertionError(decision)

    if transport_evidence is None:
        transport_status = "NOT_ESTABLISHED"
        transport_report: dict[str, Any] = {"status": transport_status}
    else:
        transport = _mapping(transport_evidence, "transport_evidence")
        if transport.get("status") not in {"CONVERGED", "NOT_CONVERGED", "NOT_ESTABLISHED"}:
            raise ActiveLearningError("transport evidence has an unsupported status")
        transport_status = str(transport["status"])
        transport_report = dict(transport)

    recommended_focus = sorted(
        {
            check["condition_id"]
            for check in condition_checks
            if not check["passed"]
        }
        | {
            model_id
            for model_id, report in _mapping(
                evaluation.get("calibration"), "evaluation.calibration"
            ).items()
            if isinstance(report, Mapping) and report.get("status") != "PASS"
        }
        | {
            model_id
            for model_id, report in spot_report["per_model"].items()
            if not report["passed"]
        }
    )
    calibration_status = "PASS" if calibration_passed else "BLOCKED_CALIBRATION"
    return {
        "schema_version": 1,
        "contract": ASSESSMENT_CONTRACT,
        "plugin_id": "active-learning",
        "operation": "assess-round",
        "status": "OK",
        "decision": decision,
        "strategy": dict(_mapping(policy.get("strategy"), "policy.strategy")),
        "policy_id": policy_info["policy_id"],
        "campaign_id": lineage["campaign_id"],
        "round_index": round_index,
        "target_domain": dict(_mapping(policy.get("target_domain"), "policy.target_domain")),
        "pes_status": (
            "CONVERGED" if decision == "CONVERGED_FOR_DECLARED_DOMAIN" else "NOT_CONVERGED"
        ),
        "transport_status": transport_status,
        "transport_evidence": transport_report,
        "coverage_passed": coverage_passed,
        "accuracy_passed": accuracy_passed,
        "pes_gates_passed_this_round": current_pes_gates_passed,
        "passed_consecutive_rounds": consecutive,
        "required_consecutive_rounds": required_consecutive,
        "recommended_focus": recommended_focus,
        "maximum_new_dft_labels": selection_policy["maximum_labels_per_round"],
        "coverage": {
            "passed": coverage_passed,
            "failed_reasons": coverage_failed,
            "calibration": {
                "status": calibration_status,
                "models": dict(
                    _mapping(evaluation.get("calibration"), "evaluation.calibration")
                ),
            },
            "conditions": condition_checks,
            "safe_spot_checks": spot_report,
        },
        "accuracy": {
            "passed": accuracy_passed,
            "failed_reasons": audit_failed,
            "audit_checks": audit_checks,
            "audit_metrics": current_audit_values,
        },
        "consecutive_stability": {
            "passed": current_pes_gates_passed and consecutive >= required_consecutive,
            "failed_reasons": stability_failed,
        },
        "round_evidence": {
            "failed_reasons": round_integrity_failed,
            "labeling_failed_reasons": labeling_failed,
            "strategy_selection": {
                "failed_reasons": trigger_coverage_failed,
                "trigger_model_coverage": selection.get("trigger_model_coverage"),
            },
        },
        "uncertainty_distribution": {
            "current": current_uncertainty,
            "previous": previous_uncertainty,
            "change": uncertainty_change,
        },
        "selection_counts": dict(_mapping(selection.get("counts"), "selection.counts")),
        "dft_labels": {
            "new_this_round": new_labels,
            "successful_query_labels": len(successful_query),
            "successful_safe_spot_checks": len(successful_spot),
            "safe_spot_checks_in_training": include_spot,
            "cumulative": cumulative_labels,
            "maximum_total": maximum_total,
        },
        "dataset_split": current_split,
        "round_lineage": lineage,
    }


def execute_operation(operation: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Dispatch a plugin operation over already-loaded mappings."""

    values = _mapping(inputs, "inputs")
    policy = _mapping(values.get("policy"), "inputs.policy")
    if operation == "committee-evaluate":
        return evaluate_committee(
            _mapping(values.get("committee_predictions"), "inputs.committee_predictions"),
            policy,
        )
    if operation == "select-candidates":
        direct = values.get("direct_selection")
        return select_candidates(
            _mapping(values.get("committee_evaluation"), "inputs.committee_evaluation"),
            _mapping(values.get("candidate_manifest"), "inputs.candidate_manifest"),
            policy,
            _mapping(direct, "inputs.direct_selection") if direct is not None else None,
        )
    if operation == "assess-round":
        transport = values.get("transport_evidence")
        return assess_round(
            _mapping(values.get("committee_evaluation"), "inputs.committee_evaluation"),
            _mapping(values.get("selection_result"), "inputs.selection_result"),
            _mapping(values.get("audit_benchmark"), "inputs.audit_benchmark"),
            _mapping(values.get("spot_checks"), "inputs.spot_checks"),
            _mapping(values.get("labeling_result"), "inputs.labeling_result"),
            _mapping(values.get("dataset_split"), "inputs.dataset_split"),
            _mapping(values.get("round_history"), "inputs.round_history"),
            _mapping(values.get("campaign"), "inputs.campaign"),
            policy,
            _mapping(transport, "inputs.transport_evidence") if transport is not None else None,
        )
    raise ActiveLearningError(f"unsupported active-learning operation: {operation}")
