"""Evidence-driven, task-aware model routing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .errors import ConfigError
from .io import load_mapping


@dataclass
class Candidate:
    model_id: str
    eligible: bool
    reasons: list[str]
    metrics: dict[str, float]
    contributions: dict[str, float]
    score: float = 0.0
    element_coverage: int = 0
    validation_samples: int = 0
    registry_recommended: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "eligible": self.eligible,
            "reasons": self.reasons,
            "metrics": self.metrics,
            "contributions": self.contributions,
            "score": self.score,
            "element_coverage": self.element_coverage,
            "validation_samples": self.validation_samples,
            "registry_recommended": self.registry_recommended,
        }


def load_registry(path: Path) -> dict[str, Any]:
    registry = load_mapping(path)
    if registry.get("schema_version") != 1 or not isinstance(registry.get("models"), list):
        raise ConfigError(f"invalid model registry: {path}")
    ids = [item.get("id") for item in registry["models"] if isinstance(item, dict)]
    if len(ids) != len(registry["models"]) or any(not isinstance(item, str) for item in ids):
        raise ConfigError(f"every model in {path} needs a string id")
    if len(set(ids)) != len(ids):
        raise ConfigError(f"duplicate model ids in {path}")
    registry["_evidence_files"] = verify_benchmark_evidence(registry, path.parent)
    return registry


def verify_benchmark_evidence(
    registry: dict[str, Any], root: Path
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for model in registry.get("models", []):
        for benchmark in model.get("benchmarks", []):
            artifact = benchmark.get("artifact", {})
            uri = artifact.get("uri")
            if not isinstance(uri, str):
                raise ConfigError(f"benchmark evidence for {model.get('id')} needs a uri")
            parsed = urlparse(uri)
            if parsed.scheme and parsed.scheme != "file":
                results.append(
                    {
                        "model_id": model.get("id"),
                        "run_id": benchmark.get("run_id"),
                        "uri": uri,
                        "status": "external",
                    }
                )
                continue
            if parsed.scheme == "file":
                evidence_path = Path(unquote(parsed.path))
            else:
                relative = unquote(parsed.path)
                evidence_path = Path(relative)
                if not evidence_path.is_absolute():
                    evidence_path = root / evidence_path
            if not evidence_path.is_file():
                raise ConfigError(f"benchmark evidence does not exist: {evidence_path}")
            results.append(
                {
                    "model_id": model.get("id"),
                    "run_id": benchmark.get("run_id"),
                    "uri": uri,
                    "status": "available",
                }
            )
    return results


def route_models(
    registry: dict[str, Any],
    *,
    task: str,
    elements: set[str],
    scenario: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    metric_rules = policy.get("metrics", [])
    if not isinstance(metric_rules, list) or not metric_rules:
        raise ConfigError(f"routing policy for {task} has no metrics")
    candidates: list[Candidate] = []
    for model in registry["models"]:
        model_id = str(model["id"])
        supported_elements = set(model.get("elements", []))
        supported_tasks = set(model.get("tasks", []))
        reasons: list[str] = []
        if not elements.issubset(supported_elements):
            reasons.append(f"missing elements: {sorted(elements - supported_elements)}")
        if task not in supported_tasks:
            reasons.append(f"task {task!r} is not declared")
        benchmark = _select_benchmark(model.get("benchmarks", []), task, scenario)
        metrics: dict[str, float] = {}
        samples = 0
        if benchmark is None:
            reasons.append(f"no benchmark for task={task}, scenario={scenario}")
        else:
            samples = int(benchmark.get("validation_samples", 0))
            raw_metrics = benchmark.get("metrics", {})
            for rule in metric_rules:
                name = rule.get("name")
                if isinstance(name, str) and isinstance(raw_metrics.get(name), (int, float)):
                    metrics[name] = float(raw_metrics[name])
                elif rule.get("required", True):
                    reasons.append(f"required metric {name!r} is missing")
        candidates.append(
            Candidate(
                model_id=model_id,
                eligible=not reasons,
                reasons=reasons,
                metrics=metrics,
                contributions={},
                element_coverage=len(supported_elements),
                validation_samples=samples,
                registry_recommended=task in set(model.get("recommended_tasks", [])),
            )
        )

    eligible = [candidate for candidate in candidates if candidate.eligible]
    for rule in metric_rules:
        name = str(rule["name"])
        direction = rule.get("direction", "minimize")
        if direction not in {"minimize", "maximize"}:
            raise ConfigError(f"invalid direction for metric {name}: {direction}")
        values = [candidate.metrics[name] for candidate in eligible if name in candidate.metrics]
        if not values:
            continue
        minimum, maximum = min(values), max(values)
        for candidate in eligible:
            if name not in candidate.metrics:
                candidate.contributions[name] = 0.0
                continue
            if maximum == minimum:
                normalized = 1.0
            else:
                normalized = (candidate.metrics[name] - minimum) / (maximum - minimum)
                if direction == "minimize":
                    normalized = 1.0 - normalized
            contribution = normalized * float(rule.get("weight", 1.0))
            candidate.contributions[name] = contribution
            candidate.score += contribution
    total_weight = sum(float(rule.get("weight", 1.0)) for rule in metric_rules)
    if total_weight <= 0:
        raise ConfigError("routing metric weights must sum to a positive value")
    for candidate in eligible:
        candidate.score = round(candidate.score / total_weight, 12)

    ranked = sorted(
        eligible,
        key=lambda candidate: (
            -candidate.score,
            -candidate.element_coverage,
            -candidate.validation_samples,
            candidate.model_id,
        ),
    )
    return {
        "task": task,
        "elements": sorted(elements),
        "scenario": scenario,
        "selected_model": ranked[0].model_id if ranked else None,
        "ranking": [candidate.to_dict() for candidate in ranked],
        "rejected": [candidate.to_dict() for candidate in candidates if not candidate.eligible],
        "evidence_files": registry.get("_evidence_files", []),
    }


def _select_benchmark(benchmarks: Any, task: str, scenario: str) -> dict[str, Any] | None:
    if not isinstance(benchmarks, list):
        return None
    matches = [
        item
        for item in benchmarks
        if isinstance(item, dict) and item.get("task") == task and item.get("scenario") == scenario
    ]
    if not matches:
        return None
    matches.sort(key=lambda item: str(item.get("run_id", "")))
    return matches[-1]
