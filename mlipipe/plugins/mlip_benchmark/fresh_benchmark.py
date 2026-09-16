"""First-party fresh MLIP inference and canonical benchmark evidence.

The heavy ML frameworks are imported lazily only when their exact model family
is selected.  Tests may inject a predictor object implementing ``predict``;
that exercises the complete dataset, evidence, metric, ranking, and provenance
contracts without installing a scientific framework.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Protocol, Sequence

if __package__:
    from mlipipe.plugins import model_runtime
else:
    from benchmark_normalization import model_runtime


PLUGIN_ID = "mlip-benchmark"
RUNNER_VERSION = "1.1.0"
PREDICTION_CONTRACT = "mlip-benchmark/prediction-evidence"
PREDICTION_SCHEMA_VERSION = 1
MODEL_FRAMEWORKS = model_runtime.MODEL_FAMILY_FRAMEWORKS
TARGETS = ("energy", "force", "stress")
OUTPUT_NAMES = (
    "prediction_evidence.json",
    "metrics.json",
    "benchmark_summary.csv",
    "model_ranking.json",
    "provenance.json",
)


class FreshBenchmarkError(RuntimeError):
    """Raised when fresh evaluation cannot produce trustworthy evidence."""


class Predictor(Protocol):
    runtime_details: dict[str, Any]
    prediction_units: dict[str, str]

    def predict(self, sample: dict[str, Any]) -> dict[str, Any]: ...


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise FreshBenchmarkError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FreshBenchmarkError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise FreshBenchmarkError(f"{field} must be finite")
    return result


def _matrix(value: Any, rows: int, columns: int, field: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != rows:
        raise FreshBenchmarkError(f"{field} must contain {rows} rows")
    result = []
    for row in value:
        if not isinstance(row, list) or len(row) != columns:
            raise FreshBenchmarkError(f"{field} rows must contain {columns} values")
        result.append([_finite(item, field) for item in row])
    return result


def _vector(value: Any, length: int, field: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise FreshBenchmarkError(f"{field} must contain {length} values")
    return [_finite(item, field) for item in value]


def _load_dataset(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.is_file() or path.suffix.lower() != ".json":
        raise FreshBenchmarkError("fresh benchmark dataset must be one canonical JSON file")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FreshBenchmarkError(f"cannot read benchmark dataset: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise FreshBenchmarkError("dataset schema_version must equal 1")
    units = raw.get("units")
    if not isinstance(units, dict):
        raise FreshBenchmarkError("dataset units must be an object")
    energy_convention = raw.get("energy_convention")
    if energy_convention not in {"total", "per-atom"}:
        raise FreshBenchmarkError("dataset energy_convention must be total or per-atom")
    stress_convention = raw.get("stress_convention")
    samples = raw.get("samples")
    if not isinstance(samples, list) or not samples:
        raise FreshBenchmarkError("dataset samples must be a non-empty list")
    normalized = []
    identities: set[str] = set()
    for index, item in enumerate(samples):
        if not isinstance(item, dict):
            raise FreshBenchmarkError(f"samples[{index}] must be an object")
        sample_id = item.get("id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in identities:
            raise FreshBenchmarkError("every dataset sample needs a unique non-empty id")
        identities.add(sample_id)
        species = item.get("species")
        if not isinstance(species, list) or not species or not all(
            isinstance(value, str) and value for value in species
        ):
            raise FreshBenchmarkError(f"sample {sample_id} requires a species list")
        natoms = len(species)
        pbc = item.get("pbc", True)
        if not isinstance(pbc, bool) and not (
            isinstance(pbc, list) and len(pbc) == 3 and all(isinstance(value, bool) for value in pbc)
        ):
            raise FreshBenchmarkError(f"sample {sample_id} pbc must be a boolean or three booleans")
        references = item.get("references")
        if not isinstance(references, dict):
            raise FreshBenchmarkError(f"sample {sample_id} requires references")
        normalized.append(
            {
                "id": sample_id,
                "species": list(species),
                "positions": _matrix(item.get("positions"), natoms, 3, f"{sample_id}.positions"),
                "cell": _matrix(item.get("cell"), 3, 3, f"{sample_id}.cell"),
                "pbc": pbc,
                "references": dict(references),
                "natoms": natoms,
            }
        )
    return {
        "units": dict(units),
        "energy_convention": energy_convention,
        "stress_convention": stress_convention,
    }, normalized


_UNIT_FACTORS = {
    "energy": {"ev": 1.0, "mev": 0.001},
    "force": {"ev/angstrom": 1.0, "mev/angstrom": 0.001},
    "stress": {"ev/angstrom^3": 1.0, "gpa": 1.0 / 160.21766208, "kbar": 0.1 / 160.21766208},
}


def _unit_token(value: Any, target: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FreshBenchmarkError(f"{target} unit must be a non-empty string")
    token = value.strip().lower().replace(" ", "")
    aliases = {
        "ev/å": "ev/angstrom",
        "ev/a": "ev/angstrom",
        "ev/ang": "ev/angstrom",
        "mev/å": "mev/angstrom",
        "mev/a": "mev/angstrom",
        "ev/å^3": "ev/angstrom^3",
        "ev/a^3": "ev/angstrom^3",
    }
    token = aliases.get(token, token)
    if target == "energy":
        token = token.replace("/atom", "")
    if token not in _UNIT_FACTORS[target]:
        raise FreshBenchmarkError(f"unsupported {target} unit conversion: {value}")
    return token


def _convert(values: Any, source: str, destination: str, target: str) -> Any:
    source_factor = _UNIT_FACTORS[target][_unit_token(source, target)]
    destination_factor = _UNIT_FACTORS[target][_unit_token(destination, target)]
    factor = source_factor / destination_factor

    def visit(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return [visit(item) for item in value]
        return _finite(value, target) * factor

    return visit(values)


def _shared_runtime_path() -> Path:
    return Path(model_runtime.__file__).resolve()


def _shared_runtime_module():
    """Compatibility accessor for callers that inspect the shared loader."""

    return model_runtime


def load_predictor(model: Path, family: str, device: str) -> Predictor:
    """Load one exact family through MLIPipe's shared scientific runtime."""

    try:
        return model_runtime.load_inference_predictor(model, family, device)
    except model_runtime.RuntimeCompatibilityError as exc:
        raise FreshBenchmarkError(str(exc)) from exc


def _normalizer_module():
    if __package__:
        from . import benchmark_wrapper
    else:
        import benchmark_wrapper
    return benchmark_wrapper


def _flatten_count(value: Any) -> int:
    if isinstance(value, list):
        return sum(_flatten_count(item) for item in value)
    return 1


def evaluate_fresh(
    *,
    model: Path,
    dataset: Path,
    output_dir: Path,
    model_family: str,
    task: str,
    scenario: str,
    split: str,
    targets: Sequence[str],
    units: dict[str, str],
    energy_normalization: str,
    stress_convention: str | None,
    device: str = "cpu",
    predictor: Predictor | None = None,
) -> dict[str, Path]:
    """Execute a model and atomically emit evidence plus normalized artifacts."""

    model = Path(model).resolve()
    dataset = Path(dataset).resolve()
    output_dir = Path(output_dir).resolve()
    if model_family not in MODEL_FRAMEWORKS:
        raise FreshBenchmarkError(f"unsupported exact model family: {model_family}")
    selected_targets = tuple(dict.fromkeys(targets))
    if not selected_targets or any(target not in TARGETS for target in selected_targets):
        raise FreshBenchmarkError("targets must be a non-empty subset of energy, force, stress")
    if energy_normalization not in {"total", "per-atom"}:
        raise FreshBenchmarkError("energy_normalization must be total or per-atom")
    if "stress" in selected_targets and not stress_convention:
        raise FreshBenchmarkError("stress target requires an explicit stress_convention")
    for target in selected_targets:
        _unit_token(units.get(target), target)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FreshBenchmarkError("fresh output directory must not already contain artifacts")

    metadata, samples = _load_dataset(dataset)
    for target in selected_targets:
        if target not in metadata["units"]:
            raise FreshBenchmarkError(f"dataset does not declare a {target} unit")
    if "stress" in selected_targets and metadata["stress_convention"] != stress_convention:
        raise FreshBenchmarkError("dataset stress convention differs from the approved convention")
    runtime = predictor if predictor is not None else load_predictor(model, model_family, device)
    runtime_info = getattr(runtime, "runtime_details", None)
    prediction_units = getattr(runtime, "prediction_units", None)
    if not isinstance(runtime_info, dict) or runtime_info.get("framework") != MODEL_FRAMEWORKS[model_family]:
        raise FreshBenchmarkError("predictor runtime framework does not match the exact model family")
    if runtime_info.get("exact_model_family") not in {None, model_family}:
        raise FreshBenchmarkError("predictor exact model family differs")
    if not isinstance(prediction_units, dict):
        raise FreshBenchmarkError("predictor must declare prediction_units")
    for target in selected_targets:
        _unit_token(prediction_units.get(target), target)
    if "energy" in selected_targets and runtime_info.get("energy_convention") != "total":
        raise FreshBenchmarkError("predictor must declare total-energy model output")
    if "stress" in selected_targets and runtime_info.get("stress_convention") != stress_convention:
        raise FreshBenchmarkError("predictor stress convention differs from the approved convention")

    evidence_records = []
    scalar_counts = {target: 0 for target in selected_targets}
    for structure_index, sample in enumerate(samples):
        prediction = runtime.predict(sample)
        if not isinstance(prediction, dict):
            raise FreshBenchmarkError(f"predictor returned no mapping for {sample['id']}")
        for target in selected_targets:
            reference = sample["references"].get(target)
            predicted = prediction.get(target)
            if reference is None or predicted is None:
                raise FreshBenchmarkError(f"partial {target} output for sample {sample['id']}")
            if target == "energy":
                reference = _finite(reference, f"{sample['id']}.reference.energy")
                predicted = _finite(predicted, f"{sample['id']}.prediction.energy")
                if metadata["energy_convention"] != energy_normalization:
                    if metadata["energy_convention"] == "total":
                        reference /= sample["natoms"]
                    else:
                        reference *= sample["natoms"]
                if energy_normalization == "per-atom":
                    predicted /= sample["natoms"]
                reference = _convert(reference, metadata["units"][target], units[target], target)
                predicted = _convert(predicted, prediction_units[target], units[target], target)
                components = ["scalar"]
            elif target == "force":
                reference = _matrix(reference, sample["natoms"], 3, f"{sample['id']}.reference.force")
                predicted = _matrix(predicted, sample["natoms"], 3, f"{sample['id']}.prediction.force")
                reference = _convert(reference, metadata["units"][target], units[target], target)
                predicted = _convert(predicted, prediction_units[target], units[target], target)
                components = ["x", "y", "z"]
            else:
                reference = _vector(reference, 6, f"{sample['id']}.reference.stress")
                predicted = _vector(predicted, 6, f"{sample['id']}.prediction.stress")
                reference = _convert(reference, metadata["units"][target], units[target], target)
                predicted = _convert(predicted, prediction_units[target], units[target], target)
                components = ["xx", "yy", "zz", "yz", "xz", "xy"]
            count = _flatten_count(reference)
            scalar_counts[target] += count
            evidence_records.append({
                "sample_id": sample["id"], "structure_index": structure_index,
                "natoms": sample["natoms"], "target": target, "components": components,
                "reference": reference, "prediction": predicted, "unit": units[target],
                "scalar_count": count, "model": model_family, "task": task,
                "scenario": scenario, "split": split,
            })

    evidence = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "contract": PREDICTION_CONTRACT,
        "plugin_id": PLUGIN_ID,
        "mode": "fresh",
        "model_execution": True,
        "model": {"family": model_family, "framework": MODEL_FRAMEWORKS[model_family], "path": str(model)},
        "dataset": {"path": str(dataset), "structure_count": len(samples)},
        "source_paths": {
            "fresh_runner": str(Path(__file__).resolve()),
            "metric_normalization": str(Path(__file__).with_name("benchmark_normalization.py").resolve()),
            "shared_model_runtime": str(_shared_runtime_path()),
        },
        "runtime": dict(runtime_info),
        "task": task, "scenario": scenario, "split": split,
        "targets": list(selected_targets),
        "units": {target: units[target] for target in selected_targets},
        "conventions": {
            "energy_normalization": energy_normalization,
            "dataset_energy_convention": metadata["energy_convention"],
            "model_energy_convention": "total",
            "stress_convention": stress_convention,
            "stress_component_order": ["xx", "yy", "zz", "yz", "xz", "xy"],
        },
        "structure_count": len(samples),
        "scalar_sample_counts": scalar_counts,
        "records": evidence_records,
    }
    evidence_payload = _json_bytes(evidence)
    normalizer = _normalizer_module()
    output_parent = output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fresh-benchmark-", dir=str(output_parent)) as temporary:
        evidence_path = Path(temporary) / "prediction_evidence.json"
        evidence_path.write_bytes(evidence_payload)
        metric_records, metric_provenance = normalizer._execute_source({
            "path": evidence_path, "evidence_locator": "prediction_evidence.json",
            "model": model_family, "task": task, "scenario": scenario, "split": split,
            "units": units,
        })
    for record in metric_records:
        record["mode"] = "fresh"
    unavailable_metrics = metric_provenance["unavailable_metrics"]
    for record in unavailable_metrics:
        record["mode"] = "fresh"
    metric_records = normalizer._validate_records(metric_records)
    metrics = {
        "schema_version": 1, "plugin_id": PLUGIN_ID, "mode": "fresh",
        "calculation_claim": "fresh-model-inference-and-metrics-from-canonical-prediction-evidence",
        "supported_models": list(MODEL_FRAMEWORKS), "record_count": len(metric_records),
        "records": metric_records, "unavailable_metrics": unavailable_metrics,
    }
    ranking = normalizer._ranking(metric_records)
    payloads = {
        "prediction_evidence.json": evidence_payload,
        "metrics.json": normalizer._json_bytes(metrics),
        "benchmark_summary.csv": normalizer._summary_bytes(metric_records),
        "model_ranking.json": normalizer._json_bytes(ranking),
    }
    provenance = {
        "schema_version": 1, "plugin_id": PLUGIN_ID, "runner_version": RUNNER_VERSION,
        "mode": "fresh", "model_execution": True, "network_access": False,
        "exact_model_family": model_family, "model_path": str(model),
        "dataset_path": str(dataset), "runtime": dict(runtime_info),
        "source_paths": evidence["source_paths"],
        "task": task, "scenario": scenario, "split": split,
        "units": {target: units[target] for target in selected_targets},
        "conventions": evidence["conventions"], "structure_count": len(samples),
        "scalar_sample_counts": scalar_counts,
        "unavailable_metrics": unavailable_metrics,
        "output_artifacts": [{"path": name} for name in sorted(payloads)],
    }
    payloads["provenance.json"] = _json_bytes(provenance)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for name in OUTPUT_NAMES:
            normalizer._write_atomic(output_dir / name, payloads[name])
    except BaseException:
        for name in OUTPUT_NAMES:
            try:
                (output_dir / name).unlink()
            except FileNotFoundError:
                pass
        raise
    return {name: output_dir / name for name in OUTPUT_NAMES}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run first-party fresh MLIP benchmark inference")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-family", choices=tuple(MODEL_FRAMEWORKS), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--target", action="append", required=True, choices=TARGETS)
    parser.add_argument("--energy-unit")
    parser.add_argument("--force-unit")
    parser.add_argument("--stress-unit")
    parser.add_argument("--energy-normalization", choices=("total", "per-atom"), required=True)
    parser.add_argument("--stress-convention")
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    units = {target: getattr(arguments, f"{target}_unit") for target in arguments.target}
    try:
        outputs = evaluate_fresh(
            model=Path(arguments.model), dataset=Path(arguments.dataset),
            output_dir=Path(arguments.output_dir), model_family=arguments.model_family,
            task=arguments.task,
            scenario=arguments.scenario, split=arguments.split, targets=arguments.target,
            units=units, energy_normalization=arguments.energy_normalization,
            stress_convention=arguments.stress_convention, device=arguments.device,
        )
    except FreshBenchmarkError as exc:
        print(f"fresh benchmark failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({name: str(path) for name, path in outputs.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
