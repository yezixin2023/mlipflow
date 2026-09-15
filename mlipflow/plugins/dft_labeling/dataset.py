"""dft labeling: dataset."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from . import contracts, label
from . import dataset_contract as DATASETS


def _dataset_frameworks(parameters: Mapping[str, Any]) -> list[str]:
    value = parameters.get("frameworks", list(contracts.DATASET_FRAMEWORKS))
    if not isinstance(value, list):
        return []
    requested = [str(item).lower() for item in value if isinstance(item, str)]
    return [name for name in contracts.DATASET_FRAMEWORKS if name in requested]


def _canonical_dataset_material(
    context: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], list[tuple[str, Path]]]:
    project_root = Path(str(context.get("project_root", ""))).expanduser().absolute().resolve()
    inputs = contracts._mapping(context.get("inputs"))
    path = contracts._project_input_path(project_root, inputs.get("canonical_dataset"))
    if path is None or not contracts._ordinary_project_file(path, project_root):
        raise ValueError("canonical_dataset must be an ordinary bounded project file")
    value, read_error = contracts._read_json(path)
    if value is None:
        raise ValueError(
            str(read_error["message"] if read_error else "canonical dataset is invalid")
        )
    if value.get("contract") != "mlipflow/canonical-dataset-merge":
        errors = DATASETS.validate_canonical_dataset(value)
        if errors:
            raise ValueError("; ".join(errors))
        return path, value, []
    sources = value.get("sources")
    if value.get("schema_version") != 1 or not isinstance(sources, list) or not sources:
        raise ValueError("canonical merge manifest requires schema_version=1 and sources")
    staged_sources: list[tuple[str, Path]] = []
    datasets = []
    seen_names = {"canonical.json", "dataset_convert.py", "dataset_contract.py"}
    for index, item in enumerate(sources):
        if not isinstance(item, Mapping):
            raise ValueError(f"canonical merge source {index} must be an object")
        relative = item.get("path")
        if not DATASETS.safe_relative(relative) or str(relative) in seen_names:
            raise ValueError(f"canonical merge source {index} path is unsafe or reserved")
        seen_names.add(str(relative))
        source = (path.parent / str(relative)).resolve()
        try:
            source.relative_to(project_root)
        except ValueError as exc:
            raise ValueError("canonical merge source escapes project_root") from exc
        if not contracts._ordinary_project_file(source, project_root):
            raise ValueError(f"canonical merge source is not an ordinary file: {relative}")
        dataset, source_error = contracts._read_json(source)
        if dataset is None:
            raise ValueError(
                str(
                    source_error["message"]
                    if source_error
                    else f"canonical merge source is invalid: {relative}"
                )
            )
        datasets.append(dataset)
        staged_sources.append((str(relative), source))
    merged = DATASETS.merge_canonical_datasets(datasets)
    return path, merged, staged_sources


def _validate_dataset_assemble(context: Mapping[str, Any]) -> list[dict[str, str]]:
    diagnostics = contracts._base_diagnostics(context, frozenset({"ssh-slurm"}))
    parameters = contracts._mapping(context.get("parameters"))
    contracts._validate_input_path(diagnostics, context, "canonical_dataset")
    allowed_parameters = {
        "operation",
        "frameworks",
        "dataset_relative_path",
        "result_manifest",
        "split_strategy",
        "split_seed",
        "split_fractions",
    }
    unknown = sorted(set(parameters) - allowed_parameters)
    if unknown:
        diagnostics.append(
            contracts._diagnostic(
                "error",
                "parameters.unknown",
                "dataset-assemble 不支持参数：" + ", ".join(unknown),
            )
        )
    raw_frameworks = parameters.get("frameworks", list(contracts.DATASET_FRAMEWORKS))
    frameworks = _dataset_frameworks(parameters)
    if (
        not isinstance(raw_frameworks, list)
        or not raw_frameworks
        or len(raw_frameworks) != len(set(str(item) for item in raw_frameworks))
        or any(
            not isinstance(item, str) or item not in contracts.DATASET_FRAMEWORKS for item in raw_frameworks
        )
        or len(frameworks) != len(raw_frameworks)
    ):
        diagnostics.append(
            contracts._diagnostic(
                "error",
                "parameters.frameworks",
                "frameworks 必须是 deepmd/m3gnet/chgnet/mace 的非空、无重复列表。",
            )
        )
    relative = parameters.get("dataset_relative_path")
    if relative is not None and not DATASETS.safe_relative(relative):
        diagnostics.append(
            contracts._diagnostic(
                "error",
                "parameters.dataset_relative_path",
                "dataset_relative_path 必须是站点 data root 下的安全相对路径。",
            )
        )
    strategy = parameters.get("split_strategy", "deterministic")
    if strategy not in {"deterministic", "group-aware"}:
        diagnostics.append(
            contracts._diagnostic(
                "error",
                "parameters.split_strategy",
                "split_strategy 必须是 deterministic 或 group-aware。",
            )
        )
    seed = parameters.get("split_seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        diagnostics.append(
            contracts._diagnostic("error", "parameters.split_seed", "split_seed 必须是非负整数。")
        )
    fractions = contracts._mapping(
        parameters.get("split_fractions", {"train": 0.8, "validation": 0.1, "test": 0.1})
    )
    if (
        set(fractions) != {"train", "validation", "test"}
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
            for value in fractions.values()
        )
        or not math.isclose(
            sum(float(value) for value in fractions.values()), 1.0, rel_tol=0, abs_tol=1e-12
        )
    ):
        diagnostics.append(
            contracts._diagnostic(
                "error",
                "parameters.split_fractions",
                "split_fractions 必须包含正数 train/validation/test 且总和为 1。",
            )
        )
    result_name = parameters.get("result_manifest", "dataset-assembly-result.json")
    if not contracts._safe_relative(result_name) or len(Path(str(result_name)).parts) != 1:
        diagnostics.append(
            contracts._diagnostic(
                "error",
                "parameters.result_manifest",
                "dataset-assemble result_manifest 必须是安全 basename。",
            )
        )
    if contracts._errors(diagnostics):
        return diagnostics
    try:
        _, canonical, _ = _canonical_dataset_material(context)
    except (OSError, TypeError, ValueError) as exc:
        diagnostics.append(
            contracts._diagnostic(
                "error",
                "inputs.canonical_dataset",
                str(exc),
            )
        )
        return diagnostics
    if relative is not None and relative != canonical.get("dataset_id"):
        diagnostics.append(
            contracts._diagnostic(
                "error",
                "parameters.dataset_relative_path",
                "dataset_relative_path 必须等于 canonical dataset_id。",
            )
        )
    try:
        DATASETS.build_split_manifest(canonical, strategy=strategy, seed=seed, fractions=fractions)
    except ValueError as exc:
        diagnostics.append(contracts._diagnostic("error", "parameters.split", str(exc)))
    return diagnostics


def _plan_dataset_assemble(context: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = _validate_dataset_assemble(context)
    if contracts._errors(diagnostics):
        return {
            "plugin_id": contracts.PLUGIN_ID,
            "status": "BLOCKED",
            "executable": False,
            "diagnostics": diagnostics,
        }
    parameters = contracts._mapping(context["parameters"])
    canonical_path, canonical, canonical_sources = _canonical_dataset_material(context)
    frameworks = _dataset_frameworks(parameters)
    relative = str(parameters.get("dataset_relative_path", canonical["dataset_id"]))
    result_name = str(parameters.get("result_manifest", "dataset-assembly-result.json"))
    staged = [
        contracts._staged_record(canonical_path, "canonical.json"),
        *[contracts._staged_record(source, remote_name) for remote_name, source in canonical_sources],
        contracts._staged_record(contracts.DATASET_CONVERTER_PATH, "dataset_convert.py"),
        contracts._staged_record(contracts.DATASET_CONTRACT_PATH, "dataset_contract.py"),
    ]
    fetch_outputs = [
        {
            "remote_name": result_name,
            "remote_path": f"output/{result_name}",
            "local_name": result_name,
            "required": True,
            "role": "dataset-assembly-result",
        },
        {
            "remote_name": "split.json",
            "remote_path": "output/split.json",
            "local_name": "split.json",
            "required": True,
            "role": "split-manifest",
        },
    ]
    fetch_outputs.extend(
        {
            "remote_name": f"{name}-dataset-reference.json",
            "remote_path": f"output/{name}-dataset-reference.json",
            "local_name": f"{name}-dataset-reference.json",
            "required": True,
            "role": f"{name}-dataset-reference",
        }
        for name in frameworks
    )
    fetch_outputs.extend(
        [
            {
                "remote_name": "benchmark-test.json",
                "remote_path": "output/benchmark-test.json",
                "local_name": "benchmark-test.json",
                "required": True,
                "role": "benchmark-dataset",
            },
            {
                "remote_name": "benchmark-dataset-reference.json",
                "remote_path": "output/benchmark-dataset-reference.json",
                "local_name": "benchmark-dataset-reference.json",
                "required": True,
                "role": "benchmark-dataset-reference",
            },
        ]
    )
    strategy = str(parameters.get("split_strategy", "deterministic"))
    seed = int(parameters.get("split_seed", 0))
    fractions = contracts._mapping(
        parameters.get("split_fractions", {"train": 0.8, "validation": 0.1, "test": 0.1})
    )
    attempt_dir = Path(str(context["attempt_dir"])).expanduser().absolute()
    attempt_index = contracts._attempt_index(attempt_dir)
    reuse_existing = False
    reused_attempt: int | None = None
    if attempt_index is not None and attempt_index > 1:
        previous = attempt_dir.parent / f"attempt-{attempt_index - 1}"
        previous_result, _ = contracts._read_json(previous / result_name)
        previous_completion, _ = contracts._read_json(previous / "completion.json")
        previous_context = {**dict(context), "attempt_dir": str(previous)}
        if (
            previous_result is not None
            and previous_completion is not None
            and previous_completion.get("status") == "COMPLETED"
            and previous_completion.get("exit_code") == 0
            and previous_completion.get("attempt") == attempt_index - 1
            and not contracts._errors(_verify_dataset_assembly(previous_context, previous_result))
        ):
            reuse_existing = True
            reused_attempt = attempt_index - 1
    template_family = "dft-dataset-reuse" if reuse_existing else "dft-dataset"
    if reuse_existing and reused_attempt is not None:
        previous = attempt_dir.parent / f"attempt-{reused_attempt}"
        for name in [*frameworks, "benchmark"]:
            filename = (
                "benchmark-dataset-reference.json"
                if name == "benchmark"
                else f"{name}-dataset-reference.json"
            )
            staged.append(contracts._staged_record(previous / filename, f"reuse/{filename}"))
    return {
        "plugin_id": contracts.PLUGIN_ID,
        "status": "READY",
        "executable": True,
        "operation": contracts.DATASET_OPERATION,
        "argv": [f"template-family:{template_family}"],
        "cwd": "remote-attempt-workspace",
        "expected_outputs": [item["remote_name"] for item in fetch_outputs],
        "input_paths": {
            "canonical_dataset": str(canonical_path),
            **{
                f"canonical_source_{index:04d}": str(source)
                for index, (_, source) in enumerate(canonical_sources, start=1)
            },
            "dataset_converter": str(contracts.DATASET_CONVERTER_PATH),
            "dataset_contract": str(contracts.DATASET_CONTRACT_PATH),
        },
        "approval_summary": {
            "expensive": False,
            "submits_jobs": True,
            "execution_model": "single-python",
            "cpus_meaning": "threads-per-process",
            "operation": contracts.DATASET_OPERATION,
            "dataset_id": canonical["dataset_id"],
            "record_count": canonical["record_count"],
            "canonical_source_count": len(canonical_sources) or 1,
            "frameworks": frameworks,
            "split_strategy": strategy,
            "split_seed": seed,
            "split_fractions": dict(fractions),
            "site_data_relative_path": relative,
            "remote_publish": not reuse_existing,
            "reuse_existing_verified_dataset": reuse_existing,
            "reused_dataset_attempt": reused_attempt,
            "silent_overwrite": False,
            "fetch_allowlist": [item["remote_name"] for item in fetch_outputs],
        },
        "scheduled_execution": {
            "schema_version": 3,
            "execution_model": "single-python",
            "template_family": template_family,
            "template_variables": {
                "PLUGIN_FRAMEWORKS": ",".join(frameworks),
                "PLUGIN_DATASET_ID": relative,
                "PLUGIN_SPLIT_STRATEGY": strategy,
                "PLUGIN_SPLIT_SEED": str(seed),
                "PLUGIN_TRAIN_FRACTION": str(fractions["train"]),
                "PLUGIN_VALIDATION_FRACTION": str(fractions["validation"]),
                "PLUGIN_TEST_FRACTION": str(fractions["test"]),
                "PLUGIN_RESULT_NAME": result_name,
            },
            "staged_files": staged,
            "fetch_outputs": fetch_outputs,
        },
        "diagnostics": diagnostics,
    }


def _verify_dataset_assembly(
    context: Mapping[str, Any], manifest: Mapping[str, Any]
) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    parameters = contracts._mapping(context["parameters"])
    try:
        _, canonical, _ = _canonical_dataset_material(context)
    except (OSError, TypeError, ValueError):
        canonical = None
    split, _ = contracts._read_json(attempt / "split.json")
    if canonical is None or split is None:
        return [contracts._diagnostic("error", "dataset.inputs", "canonical 或 split.json 不可读。")]
    frameworks = _dataset_frameworks(parameters)
    strategy = str(parameters.get("split_strategy", "deterministic"))
    seed = int(parameters.get("split_seed", 0))
    fractions = contracts._mapping(
        parameters.get("split_fractions", {"train": 0.8, "validation": 0.1, "test": 0.1})
    )
    try:
        expected_split = DATASETS.build_split_manifest(
            canonical, strategy=strategy, seed=seed, fractions=fractions
        )
    except ValueError as exc:
        return [contracts._diagnostic("error", "dataset.split", str(exc))]
    if split != expected_split:
        diagnostics.append(
            contracts._diagnostic("error", "dataset.split", "split.json 与批准的确定性 split 不一致。")
        )
    relative = str(parameters.get("dataset_relative_path", canonical["dataset_id"]))
    if manifest.get("dataset_id") != canonical["dataset_id"]:
        diagnostics.append(
            contracts._diagnostic("error", "dataset.dataset_id", "assembly dataset_id 不一致。")
        )
    if manifest.get("split_id") != expected_split["split_id"]:
        diagnostics.append(contracts._diagnostic("error", "dataset.split_id", "assembly split_id 不一致。"))
    if manifest.get("counts") != expected_split["counts"]:
        diagnostics.append(contracts._diagnostic("error", "dataset.counts", "assembly split counts 不一致。"))
    expected_paths = {name: f"{relative}/{name}" for name in frameworks}
    if manifest.get("framework_output_paths") != expected_paths:
        diagnostics.append(contracts._diagnostic("error", "dataset.paths", "framework output paths 不一致。"))
    if set(contracts._mapping(manifest.get("formats"))) != set(frameworks):
        diagnostics.append(
            contracts._diagnostic("error", "dataset.formats", "framework format table 不完整。")
        )
    if manifest.get("benchmark_output_path") != f"{relative}/benchmark/test.json":
        diagnostics.append(
            contracts._diagnostic("error", "dataset.benchmark_path", "benchmark test path 不一致。")
        )
    if manifest.get("benchmark_record_ids") != expected_split["test_record_ids"]:
        diagnostics.append(
            contracts._diagnostic("error", "dataset.benchmark_ids", "benchmark 未使用同一 test split。")
        )
    conventions = contracts._mapping(manifest.get("units_and_conventions"))
    if (
        conventions.get("energy") != "total eV per configuration, unchanged"
        or conventions.get("forces") != "eV/angstrom, unchanged"
    ):
        diagnostics.append(
            contracts._diagnostic("error", "dataset.units", "energy/force convention 不完整。")
        )
    for framework in frameworks:
        reference, _ = contracts._read_json(attempt / f"{framework}-dataset-reference.json")
        if (
            reference is None
            or reference.get("schema_version") != 1
            or reference.get("relative_path") != expected_paths[framework]
            or reference.get("kind") != "directory"
            or reference.get("split_id") != expected_split["split_id"]
        ):
            diagnostics.append(
                contracts._diagnostic(
                    "error",
                    f"dataset.reference.{framework}",
                    "mlip-training dataset reference 无效。",
                )
            )
    benchmark, _ = contracts._read_json(attempt / "benchmark-test.json")
    expected_benchmark = DATASETS.benchmark_dataset(canonical, expected_split)
    if not label._same_numeric_tree(benchmark, expected_benchmark):
        diagnostics.append(
            contracts._diagnostic(
                "error",
                "dataset.benchmark",
                "benchmark test dataset 与 canonical test split 不一致。",
            )
        )
    benchmark_reference, _ = contracts._read_json(attempt / "benchmark-dataset-reference.json")
    if (
        benchmark_reference is None
        or benchmark_reference.get("schema_version") != 1
        or benchmark_reference.get("relative_path") != f"{relative}/benchmark/test.json"
        or benchmark_reference.get("kind") != "file"
        or benchmark_reference.get("split_id") != expected_split["split_id"]
        or benchmark_reference.get("split") != "test"
        or benchmark is None
    ):
        diagnostics.append(
            contracts._diagnostic(
                "error", "dataset.benchmark_reference", "benchmark dataset reference 无效。"
            )
        )
    return diagnostics


def _collect_dataset_assembly(
    context: Mapping[str, Any], manifest: Mapping[str, Any], diagnostics: list[dict[str, str]]
) -> dict[str, Any]:
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    parameters = contracts._mapping(context["parameters"])
    frameworks = _dataset_frameworks(parameters)
    result_name = str(parameters.get("result_manifest", "dataset-assembly-result.json"))
    artifacts: list[dict[str, Any]] = [
        {
            "name": "dataset-assembly-result",
            "role": "dataset-assembly-result",
            "path": result_name,
            "media_type": "application/json",
        },
        {
            "name": "split-manifest",
            "role": "split-manifest",
            "path": "split.json",
            "media_type": "application/json",
        },
        {
            "name": "benchmark-dataset",
            "role": "benchmark-dataset",
            "path": "benchmark-test.json",
            "media_type": "application/json",
        },
        {
            "name": "benchmark-dataset-reference",
            "role": "benchmark-dataset-reference",
            "path": "benchmark-dataset-reference.json",
            "media_type": "application/json",
        },
    ]
    for framework in frameworks:
        reference_name = f"{framework}-dataset-reference.json"
        reference, _ = contracts._read_json(attempt / reference_name)
        assert reference is not None
        artifacts.extend(
            [
                {
                    "name": f"{framework}-dataset-reference",
                    "role": f"{framework}-dataset-reference",
                    "path": reference_name,
                    "media_type": "application/json",
                },
                {
                    "name": f"{framework}-dataset",
                    "role": f"{framework}-dataset",
                    "uri": f"mlipflow-data:///{reference['relative_path']}",
                    "metadata": {
                        "kind": "directory",
                        "dataset_id": reference["dataset_id"],
                        "split_id": manifest["split_id"],
                    },
                },
            ]
        )
    benchmark_reference, _ = contracts._read_json(attempt / "benchmark-dataset-reference.json")
    assert benchmark_reference is not None
    artifacts.append(
        {
            "name": "published-benchmark-dataset",
            "role": "published-benchmark-dataset",
            "uri": f"mlipflow-data:///{benchmark_reference['relative_path']}",
            "metadata": {
                "kind": "file",
                "dataset_id": benchmark_reference["dataset_id"],
                "split_id": manifest["split_id"],
                "split": "test",
            },
        }
    )
    return {
        "plugin_id": contracts.PLUGIN_ID,
        "status": "OK",
        "diagnostics": diagnostics,
        "artifacts": artifacts,
        "metrics": {
            "dataset_id": manifest["dataset_id"],
            "split_id": manifest["split_id"],
            **manifest["counts"],
            "frameworks": frameworks,
        },
    }
