"""dft labeling: label."""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping

from . import contracts
from . import dataset_contract as DATASETS


def _validate_label(context: Mapping[str, Any]) -> list[dict[str, str]]:
    backend = context.get("backend")
    diagnostics = contracts._base_diagnostics(context, frozenset({"local", "ssh-slurm"}))
    parameters = contracts._mapping(context.get("parameters"))
    for key in ("structures_manifest", "labeling_config"):
        contracts._validate_input_path(diagnostics, context, key)
    contracts._validate_input_path(diagnostics, context, "dft_input_manifest", required=False)
    allowed_parameters = {
        "operation",
        "label_script",
        "interpreter_argv",
        "engine",
        "completion_policy",
        "units",
        "result_manifest",
        "calculation_concurrency",
    }
    unknown = sorted(set(parameters) - allowed_parameters)
    if "extra_args" in parameters:
        diagnostics.append(
            contracts._diagnostic(
                "error",
                "parameters.extra_args",
                "不接受 extra_args；固定输入、输出和引擎参数不得被覆盖。",
            )
        )
    if unknown:
        diagnostics.append(
            contracts._diagnostic("error", "parameters.unknown", "不支持参数：" + ", ".join(unknown))
        )
    if backend == "local":
        if not contracts._safe_relative(parameters.get("label_script")):
            diagnostics.append(
                contracts._diagnostic(
                    "error",
                    "parameters.label_script",
                    "local label_script 必须是 project_root 下的安全相对路径。",
                )
            )
        interpreter = parameters.get("interpreter_argv")
        if not (
            isinstance(interpreter, list)
            and len(interpreter) == 1
            and contracts._plain_string(interpreter[0])
        ):
            diagnostics.append(
                contracts._diagnostic(
                    "error",
                    "parameters.interpreter_argv",
                    "local interpreter_argv 必须只含一个可执行文件。",
                )
            )
        elif Path(interpreter[0]).name.lower() in contracts.SHELL_EXECUTABLES:
            diagnostics.append(
                contracts._diagnostic(
                    "error", "parameters.shell_forbidden", "interpreter_argv 不得选择 shell。"
                )
            )
    elif backend == "ssh-slurm":
        if (
            parameters.get("label_script") is not None
            or parameters.get("interpreter_argv") is not None
        ):
            diagnostics.append(
                contracts._diagnostic(
                    "error",
                    "parameters.local_runner",
                    "ssh-slurm 使用站点模板，不接受 local label_script/interpreter_argv。",
                )
            )
        if contracts._mapping(context.get("inputs")).get("dft_input_manifest") is None:
            diagnostics.append(
                contracts._diagnostic(
                    "error",
                    "inputs.dft_input_manifest",
                    "ssh-slurm label 必须绑定已审核的 dft_input_manifest。",
                )
            )
    concurrency = parameters.get("calculation_concurrency", 1)
    if not contracts._positive_int(concurrency) or int(concurrency) > contracts.MAX_SCHEDULED_CALCULATIONS:
        diagnostics.append(
            contracts._diagnostic(
                "error",
                "parameters.calculation_concurrency",
                f"calculation_concurrency 必须在 1..{contracts.MAX_SCHEDULED_CALCULATIONS}。",
            )
        )
    elif backend != "ssh-slurm" and "calculation_concurrency" in parameters:
        diagnostics.append(
            contracts._diagnostic(
                "error",
                "parameters.calculation_concurrency",
                "calculation_concurrency 只适用于 ssh-slurm label。",
            )
        )
    if not contracts._plain_string(parameters.get("engine")):
        diagnostics.append(contracts._diagnostic("error", "parameters.engine", "engine 必须显式声明。"))
    elif backend == "ssh-slurm" and parameters.get("engine") != "vasp":
        diagnostics.append(
            contracts._diagnostic(
                "error", "parameters.engine", "vasp template family 的 engine 必须是 vasp。"
            )
        )
    if not contracts._safe_relative(parameters.get("result_manifest", "dft-labeling-result.json")):
        diagnostics.append(
            contracts._diagnostic(
                "error", "parameters.result_manifest", "result_manifest 必须是安全相对路径。"
            )
        )
    elif backend == "ssh-slurm" and not contracts.SAFE_ID.fullmatch(
        str(parameters.get("result_manifest", "dft-labeling-result.json"))
    ):
        diagnostics.append(
            contracts._diagnostic(
                "error",
                "parameters.result_manifest",
                "ssh-slurm result_manifest 必须是安全 basename。",
            )
        )
    completion = parameters.get("completion_policy")
    if (
        not isinstance(completion, Mapping)
        or type(completion.get("require_ionic_convergence")) is not bool
    ):
        diagnostics.append(
            contracts._diagnostic(
                "error",
                "parameters.completion_policy",
                "completion_policy 必须显式声明 require_ionic_convergence。",
            )
        )
    units = parameters.get("units")
    required_units = {"energy", "length", "force", "stress"}
    if not isinstance(units, Mapping) or any(
        not contracts._plain_string(units.get(key)) for key in required_units
    ):
        diagnostics.append(
            contracts._diagnostic(
                "error", "parameters.units", "units 必须声明 energy、length、force、stress。"
            )
        )
    elif backend == "ssh-slurm" and units != {
        "energy": "eV",
        "length": "angstrom",
        "force": "eV/angstrom",
        "stress": "kbar-vasp-3x3",
    }:
        diagnostics.append(
            contracts._diagnostic(
                "error",
                "parameters.units",
                "bundled static runner 固定输出 eV/angstrom/kbar-vasp-3x3 单位。",
            )
        )
    if backend == "ssh-slurm":
        config_path = contracts._mapping(context.get("inputs")).get("labeling_config")
        config: Mapping[str, Any] | None = None
        resolved_config = contracts._project_input_path(context.get("project_root"), config_path)
        if resolved_config is not None:
            config, _ = contracts._parse_labeling_contract(resolved_config)
        calculation_type = config.get("calculation_type") if isinstance(config, Mapping) else None
        if calculation_type is not None and calculation_type not in contracts.SCHEDULED_TYPES:
            diagnostics.append(
                contracts._diagnostic(
                    "error",
                    "inputs.labeling_config",
                    "scheduler runner 支持 static、relax 与 aimd。",
                )
            )
        # Ionic convergence is a completion criterion for relax and only for
        # relax: static has a single ionic step, and AIMD is a trajectory rather
        # than a structural optimisation.
        expected_ionic = calculation_type == contracts.RELAX_TYPE
        if (
            isinstance(completion, Mapping)
            and completion.get("require_ionic_convergence") is not expected_ionic
        ):
            diagnostics.append(
                contracts._diagnostic(
                    "error",
                    "parameters.completion_policy",
                    f"{calculation_type or 'scheduled'} 计算要求 require_ionic_convergence="
                    f"{str(expected_ionic).lower()}。",
                )
            )
    return diagnostics


def _plan_label(context: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = _validate_label(context)
    if contracts._errors(diagnostics):
        return {
            "plugin_id": contracts.PLUGIN_ID,
            "status": "BLOCKED",
            "executable": False,
            "diagnostics": diagnostics,
        }
    if context.get("backend") == "ssh-slurm":
        return _plan_scheduled_label(context, diagnostics)
    inputs = contracts._mapping(context["inputs"])
    parameters = contracts._mapping(context["parameters"])
    result_path = contracts._path(
        context["attempt_dir"], parameters.get("result_manifest", "dft-labeling-result.json")
    )
    argv = list(parameters["interpreter_argv"])
    argv.extend(
        [
            str(contracts._path(context["project_root"], parameters["label_script"])),
            "--operation",
            contracts.LABEL_OPERATION,
            "--structures-manifest",
            str(contracts._path(context["project_root"], inputs["structures_manifest"])),
            "--labeling-config",
            str(contracts._path(context["project_root"], inputs["labeling_config"])),
            "--attempt-dir",
            str(Path(str(context["attempt_dir"])).expanduser().absolute()),
            "--result-manifest",
            str(result_path),
            "--engine",
            str(parameters["engine"]),
        ]
    )
    if inputs.get("dft_input_manifest") is not None:
        argv.extend(
            [
                "--dft-input-manifest",
                str(contracts._path(context["project_root"], inputs["dft_input_manifest"])),
            ]
        )
    return {
        "plugin_id": contracts.PLUGIN_ID,
        "status": "READY",
        "executable": True,
        "argv": argv,
        "cwd": str(Path(str(context["attempt_dir"])).expanduser().absolute()),
        "expected_outputs": [str(result_path)],
        "diagnostics": diagnostics,
        "operation": contracts.LABEL_OPERATION,
        "approval_summary": {
            "expensive": True,
            "runs_vasp": True,
            "engine": parameters["engine"],
            "backend": context["backend"],
            "resources": context["resources"],
            "completion_policy": parameters["completion_policy"],
            "prepared_input_manifest": inputs.get("dft_input_manifest"),
        },
    }


def _plan_scheduled_label(
    context: Mapping[str, Any], diagnostics: list[dict[str, str]]
) -> dict[str, Any]:
    """Plan one sequential job or independently scheduled VASP calculations.

    Calculation order is taken from the prepare manifest and frozen into
    calc-0001..calc-N, so the remote layout is deterministic and never depends
    on directory iteration order.  The calculation type is shared by the whole
    batch; it selects which outputs are required, not how the job is launched.
    """

    inputs = contracts._mapping(context["inputs"])
    project_root = Path(str(context["project_root"])).expanduser().absolute().resolve()
    paths = {
        "structures_manifest": contracts._path(project_root, inputs["structures_manifest"]),
        "labeling_config": contracts._path(project_root, inputs["labeling_config"]),
        "dft_input_manifest": contracts._path(project_root, inputs["dft_input_manifest"]),
    }
    for name, path in paths.items():
        if not contracts._ordinary_project_file(path, project_root):
            diagnostics.append(
                contracts._diagnostic(
                    "error", f"inputs.{name}", f"{name} 必须是 project_root 内的普通小文件。"
                )
            )
    prepared, read_error = contracts._read_json(paths["dft_input_manifest"])
    calculations = prepared.get("calculations") if isinstance(prepared, Mapping) else None
    calculation_type = prepared.get("calculation_type") if isinstance(prepared, Mapping) else None
    if (
        not isinstance(prepared, Mapping)
        or prepared.get("status") != "OK"
        or prepared.get("operation") != contracts.PREPARE_OPERATION
        or prepared.get("engine") != "vasp"
        or calculation_type not in contracts.SCHEDULED_TYPES
        or not isinstance(calculations, list)
        or not 1 <= len(calculations) <= contracts.MAX_SCHEDULED_CALCULATIONS
        or any(not isinstance(item, Mapping) for item in calculations)
    ):
        diagnostics.append(
            contracts._diagnostic(
                "error",
                "inputs.dft_input_manifest.contract",
                str(
                    read_error["message"]
                    if read_error
                    else "scheduler runner 需要一个 OK 的 static/relax/aimd prepare manifest，"
                    f"且 calculation 数量在 1..{contracts.MAX_SCHEDULED_CALCULATIONS}。"
                ),
            )
        )
    if contracts._errors(diagnostics):
        return {
            "plugin_id": contracts.PLUGIN_ID,
            "status": "BLOCKED",
            "executable": False,
            "diagnostics": diagnostics,
        }
    assert isinstance(prepared, Mapping) and isinstance(calculations, list)
    assert isinstance(calculation_type, str)

    config, config_error = contracts._parse_labeling_contract(paths["labeling_config"])
    if config is None or config.get("calculation_type") != calculation_type:
        diagnostics.append(
            contracts._diagnostic(
                "error",
                "inputs.labeling_config",
                str(
                    config_error
                    or "labeling_config 与 prepare manifest 的 calculation_type 不一致。"
                ),
            )
        )
        return {
            "plugin_id": contracts.PLUGIN_ID,
            "status": "BLOCKED",
            "executable": False,
            "diagnostics": diagnostics,
        }

    prepared_root = paths["dft_input_manifest"].parent
    staged: list[dict[str, Any]] = [
        contracts._staged_record(paths["structures_manifest"], "structures.json"),
        contracts._staged_record(paths["labeling_config"], "labeling.json"),
        contracts._staged_record(paths["dft_input_manifest"], "dft-input-manifest.json"),
    ]
    planned: list[dict[str, Any]] = []
    for index, calculation in enumerate(calculations):
        calc_id = _calculation_id(index)
        files = calculation.get("files")
        if not isinstance(files, Mapping) or set(files) != contracts.EXPECTED_FILE_NAMES:
            diagnostics.append(
                contracts._diagnostic(
                    "error",
                    f"inputs.dft_input_manifest.{calc_id}.files",
                    f"{calc_id} 必须且只能声明四个 VASP 输入。",
                )
            )
            continue
        for name in ("POSCAR", "INCAR", "KPOINTS", "POTCAR"):
            record = files.get(name)
            relative = record.get("path") if isinstance(record, Mapping) else None
            if not contracts._safe_relative(relative):
                diagnostics.append(
                    contracts._diagnostic(
                        "error",
                        f"inputs.dft_input_manifest.{calc_id}.{name}",
                        f"{calc_id} 的 {name} 路径不安全。",
                    )
                )
                continue
            source = prepared_root / str(relative)
            if not contracts._ordinary_project_file(source, project_root):
                diagnostics.append(
                    contracts._diagnostic(
                        "error",
                        f"inputs.dft_input_manifest.{calc_id}.{name}",
                        f"{calc_id} 的 {name} 缺失。",
                    )
                )
                continue
            # Nested remote name: calc-0001/POSCAR. POTCAR remains staged-only.
            staged.append(contracts._staged_record(source, f"{calc_id}/{name}", sensitive=name == "POTCAR"))
        planned.append(
            {
                "id": calc_id,
                "structure_id": calculation.get("structure_id"),
                "atom_count": calculation.get("atom_count"),
            }
        )
    if contracts._errors(diagnostics):
        return {
            "plugin_id": contracts.PLUGIN_ID,
            "status": "BLOCKED",
            "executable": False,
            "diagnostics": diagnostics,
        }

    active_ids = [str(item["id"]) for item in planned]

    parameters = contracts._mapping(context["parameters"])
    resources = contracts._mapping(context["resources"])
    requested_concurrency = int(parameters.get("calculation_concurrency", 1))
    independent_jobs = requested_concurrency > 1
    if independent_jobs and requested_concurrency < len(active_ids):
        diagnostics.append(
            contracts._diagnostic(
                "error",
                "parameters.calculation_concurrency",
                "独立作业模式要求 calculation_concurrency 不小于本次要提交的 calculation 数量。",
            )
        )
        return {
            "plugin_id": contracts.PLUGIN_ID,
            "status": "BLOCKED",
            "executable": False,
            "diagnostics": diagnostics,
        }
    mpi_ranks_per_calculation = int(resources["cpus"])
    submitted_job_count = len(active_ids) if independent_jobs else 1
    template_family = "vasp-batch" if independent_jobs else "vasp"

    required, optional, _ = _scheduled_output_spec(calculation_type)
    fetch_outputs: list[dict[str, Any]] = []
    for item in planned:
        calc_id = str(item["id"])
        for name in required:
            fetch_outputs.append(
                {
                    "remote_name": f"{calc_id}/{name}",
                    "local_name": f"{calc_id}/{name}",
                    "required": True,
                    "role": (
                        "aimd-trajectory"
                        if calculation_type == contracts.AIMD_TYPE and name == "vasprun.xml"
                        else "vasp-output"
                    ),
                }
            )
        for name in optional:
            fetch_outputs.append(
                {
                    "remote_name": f"{calc_id}/{name}",
                    "local_name": f"{calc_id}/{name}",
                    "required": False,
                    "role": "vasp-output",
                }
            )
        for stream in ("stdout", "stderr"):
            fetch_outputs.append(
                {
                    "remote_name": f"{calc_id}.{stream}",
                    "remote_path": f"logs/{calc_id}.{stream}",
                    "local_name": f"{calc_id}/{stream}.log",
                    "required": False,
                    "role": "scheduler-log",
                }
            )
    if independent_jobs:
        shared_staged = [item for item in staged if "/" not in str(item["remote_name"])]
        submissions = []
        for calc_id in active_ids:
            submissions.append(
                {
                    "id": calc_id,
                    "template_variables": {
                        "PLUGIN_CALCULATION_IDS": calc_id,
                        "PLUGIN_CALCULATION_CONCURRENCY": "1",
                        "PLUGIN_MPI_RANKS_PER_CALCULATION": str(mpi_ranks_per_calculation),
                    },
                    "staged_files": [
                        *shared_staged,
                        *[
                            item
                            for item in staged
                            if str(item["remote_name"]).startswith(f"{calc_id}/")
                        ],
                    ],
                    "fetch_outputs": [
                        item
                        for item in fetch_outputs
                        if str(item["remote_name"]).startswith(calc_id)
                    ],
                }
            )
        scheduled_execution = {
            "schema_version": 4,
            "submission_strategy": "independent-jobs",
            "execution_model": "mpi",
            "template_family": template_family,
            "submissions": submissions,
        }
    else:
        scheduled_execution = {
            "schema_version": 3,
            "execution_model": "mpi",
            "template_family": template_family,
            "template_variables": {
                "PLUGIN_CALCULATION_IDS": ",".join(active_ids),
                "PLUGIN_CALCULATION_CONCURRENCY": "1",
                "PLUGIN_MPI_RANKS_PER_CALCULATION": str(mpi_ranks_per_calculation),
            },
            "staged_files": staged,
            "fetch_outputs": fetch_outputs,
        }
    return {
        "plugin_id": contracts.PLUGIN_ID,
        "status": "READY",
        "executable": True,
        "argv": [f"template-family:{template_family}"],
        "cwd": "remote-attempt-workspace",
        "expected_outputs": [item["remote_name"] for item in fetch_outputs if item["required"]],
        "diagnostics": diagnostics,
        "operation": contracts.LABEL_OPERATION,
        "calculation_type": calculation_type,
        "calculations": planned,
        "failure_salvage": {
            "schema_version": 1,
            "fetch_remote_names": [
                *[str(item["remote_name"]) for item in fetch_outputs],
                "completion.json",
            ],
        },
        "approval_summary": {
            "calculation_type": calculation_type,
            "calculation_count": len(planned),
            "submitted_calculation_count": len(active_ids),
            "submitted_calculation_ids": active_ids,
            "runs_vasp": True,
            "submits_jobs": True,
            "execution_model": "mpi",
            "cpus_meaning": "mpi-task-count",
            "requested_calculation_concurrency": requested_concurrency,
            "submission_strategy": (
                "independent-jobs" if independent_jobs else "single-job-sequential"
            ),
            "submitted_job_count": submitted_job_count,
            "jobs_may_run_or_queue_independently": independent_jobs,
            "mpi_ranks_per_calculation": mpi_ranks_per_calculation,
            "maximum_concurrent_mpi_ranks": (submitted_job_count * mpi_ranks_per_calculation),
            # POTCAR is staged but licensed, so it never appears here and never
            # becomes a fetched or collected artifact.
            "fetch_allowlist": sorted({*required, *optional}),
            "staged_file_count": len(staged),
        },
        "input_paths": contracts._file_paths(paths),
        "scheduled_execution": scheduled_execution,
    }


def _verify_label_result(
    context: Mapping[str, Any], manifest: Mapping[str, Any], verify_files: bool
) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    parameters = contracts._mapping(context["parameters"])
    for key, expected in (
        ("schema_version", 1),
        ("plugin_id", contracts.PLUGIN_ID),
        ("status", "OK"),
        ("engine", parameters.get("engine")),
    ):
        if manifest.get(key) != expected:
            diagnostics.append(contracts._diagnostic("error", f"result.{key}", f"结果 {key} 不匹配。"))
    input_paths = {
        "structures_manifest": contracts._path(
            context["project_root"], contracts._mapping(context["inputs"])["structures_manifest"]
        ),
        "labeling_config": contracts._path(
            context["project_root"], contracts._mapping(context["inputs"])["labeling_config"]
        ),
    }
    if contracts._mapping(context["inputs"]).get("dft_input_manifest") is not None:
        input_paths["dft_input_manifest"] = contracts._path(
            context["project_root"], contracts._mapping(context["inputs"])["dft_input_manifest"]
        )
    if verify_files:
        for name, path in input_paths.items():
            if not contracts._ordinary_file(path):
                diagnostics.append(contracts._diagnostic("error", f"result.inputs.{name}", "输入文件缺失。"))
    completion = manifest.get("completion")
    if not isinstance(completion, Mapping):
        diagnostics.append(contracts._diagnostic("error", "result.completion", "缺少标准 completion。"))
    else:
        if completion.get("scheduler_success") is not True:
            diagnostics.append(contracts._diagnostic("error", "completion.scheduler", "调度/进程未成功。"))
        if completion.get("electronic_converged") is not True:
            diagnostics.append(contracts._diagnostic("error", "completion.electronic", "电子步未收敛。"))
        if completion.get("truncated") is not False:
            diagnostics.append(contracts._diagnostic("error", "completion.truncated", "输出截断状态不安全。"))
        ionic_required = completion.get("ionic_convergence_required")
        ionic_converged = completion.get("ionic_converged")
        expected_required = contracts._mapping(parameters.get("completion_policy")).get(
            "require_ionic_convergence"
        )
        if type(ionic_required) is not bool or ionic_required is not expected_required:
            diagnostics.append(
                contracts._diagnostic("error", "completion.ionic_policy", "离子收敛策略与批准计划不一致。")
            )
        elif ionic_required and ionic_converged is not True:
            diagnostics.append(contracts._diagnostic("error", "completion.ionic", "离子步未收敛。"))
        elif not ionic_required and ionic_converged not in {None, True}:
            diagnostics.append(
                contracts._diagnostic(
                    "error", "completion.ionic", "静态计算 ionic_converged 只能为 null 或 true。"
                )
            )
    if manifest.get("units") != parameters.get("units"):
        diagnostics.append(contracts._diagnostic("error", "result.units", "结果单位与批准 units 不一致。"))
    source_count = manifest.get("source_structure_count")
    label_count = manifest.get("label_count")
    if not contracts._positive_int(source_count) or label_count != source_count:
        diagnostics.append(
            contracts._diagnostic("error", "result.label_count", "每个源结构必须恰有一个标签。")
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        diagnostics.append(contracts._diagnostic("error", "result.artifacts", "必须声明至少一个数据集产物。"))
        return diagnostics
    names: set[str] = set()
    for index, artifact in enumerate(artifacts):
        prefix = f"artifacts[{index}]"
        if not isinstance(artifact, Mapping):
            diagnostics.append(contracts._diagnostic("error", prefix, "产物必须是对象。"))
            continue
        name = artifact.get("name")
        if not contracts._plain_string(name) or name in names:
            diagnostics.append(contracts._diagnostic("error", f"{prefix}.name", "产物 name 必须非空且唯一。"))
        else:
            names.add(str(name))
        relative = artifact.get("path")
        if not contracts._safe_relative(relative):
            diagnostics.append(contracts._diagnostic("error", f"{prefix}.path", "产物必须有安全路径。"))
            continue
        if not contracts._plain_string(artifact.get("media_type")):
            diagnostics.append(
                contracts._diagnostic("error", f"{prefix}.media_type", "产物必须声明 media_type。")
            )
        if verify_files:
            path = contracts._path(context["attempt_dir"], relative)
            if not contracts._ordinary_file(path) or path.stat().st_size == 0:
                diagnostics.append(contracts._diagnostic("error", f"{prefix}.path", "数据产物缺失或为空。"))
    if context.get("backend") == "ssh-slurm":
        diagnostics.extend(_verify_scheduled_vasp_result(context, manifest, verify_files))
    return diagnostics


def _xml_rows(element: ET.Element | None) -> list[list[float]] | None:
    if element is None:
        return None
    try:
        rows = [
            [float(token) for token in (item.text or "").split()] for item in element.findall("v")
        ]
    except ValueError:
        return None
    if not rows or any(not row or any(not math.isfinite(value) for value in row) for row in rows):
        return None
    return rows


def _same_numeric_tree(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-10)
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_same_numeric_tree(a, e) for a, e in zip(actual, expected))
        )
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_same_numeric_tree(actual[key], value) for key, value in expected.items())
        )
    return actual == expected


def _calculation_id(index: int) -> str:
    return f"calc-{index + 1:04d}"


def _scheduled_output_spec(
    calculation_type: str,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, int]]:
    """Required outputs, optional outputs and byte bounds for one type."""

    if calculation_type == contracts.RELAX_TYPE:
        return (
            ("OUTCAR", "OSZICAR", "vasprun.xml", "CONTCAR"),
            (),
            dict(contracts._STATIC_RELAX_LIMITS),
        )
    if calculation_type == contracts.AIMD_TYPE:
        return (
            ("OUTCAR", "OSZICAR", "vasprun.xml"),
            ("XDATCAR",),
            dict(contracts._AIMD_LIMITS),
        )
    return ("OUTCAR", "OSZICAR", "vasprun.xml"), (), dict(contracts._STATIC_RELAX_LIMITS)


def _frame_from_calculation(element: ET.Element) -> dict[str, Any] | None:
    """Build one ionic-step frame from a single ``<calculation>`` element."""

    scsteps = element.findall("./scstep")
    energy = math.nan
    if scsteps:
        node = scsteps[-1].find("./energy/i[@name='e_0_energy']")
        if node is not None:
            energy = float(str(node.text))
    if not math.isfinite(energy):
        fallback = element.find("./energy/i[@name='e_wo_entrp']")
        energy = float(str(fallback.text)) if fallback is not None else math.nan
    structure = element.find("./structure")
    lattice = (
        _xml_rows(structure.find("./crystal/varray[@name='basis']"))
        if structure is not None
        else None
    )
    positions = (
        _xml_rows(structure.find("./varray[@name='positions']")) if structure is not None else None
    )
    forces = _xml_rows(element.find("./varray[@name='forces']"))
    stress = _xml_rows(element.find("./varray[@name='stress']"))
    # lattice/positions may be absent when a writer emits only a document-level
    # finalpos structure; the caller fills the final frame from it.
    if not math.isfinite(energy) or forces is None or not scsteps:
        return None
    return {
        "energy_ev": energy,
        "lattice_angstrom": lattice,
        "fractional_coordinates": positions,
        "forces_ev_per_angstrom": forces,
        # AIMD steps do not always carry a stress tensor; that is legitimate and
        # is recorded as absent rather than invented.
        "stress_kbar_vasp_3x3": stress,
        "electronic_steps": len(scsteps),
    }


def _parse_vasprun_trajectory(
    path: Path, max_frames: int = contracts.MAX_AIMD_FRAMES
) -> dict[str, Any] | None:
    """Stream every ionic step out of one vasprun.xml.

    ``iterparse`` is used and each ``<calculation>`` is released as soon as its
    frame has been extracted, so a long AIMD trajectory is never materialised as
    one in-memory tree.  ``NELM`` and ``atominfo`` both precede the calculations
    in VASP's output, so a single forward pass suffices.
    """

    nelm: int | None = None
    species: list[str] = []
    frames: list[dict[str, Any]] = []
    final_lattice: Any = None
    final_positions: Any = None
    exceeded = False
    try:
        for _, element in ET.iterparse(str(path), events=("end",)):
            tag = element.tag
            if tag == "separator" and element.get("name") == "electronic convergence":
                node = element.find("./i[@name='NELM']")
                if node is not None and nelm is None:
                    nelm = int(float(str(node.text)))
                element.clear()
            elif tag == "atominfo":
                if not species:
                    species = [
                        str(row.findtext("c", default="")).strip()
                        for row in element.findall("./array[@name='atoms']/set/rc")
                    ]
                element.clear()
            elif tag == "structure" and element.get("name") == "finalpos":
                # Document-level final structure; only used to complete the last
                # frame when the writer omitted a per-calculation structure.
                final_lattice = _xml_rows(element.find("./crystal/varray[@name='basis']"))
                final_positions = _xml_rows(element.find("./varray[@name='positions']"))
                element.clear()
            elif tag == "calculation":
                if len(frames) >= max_frames:
                    exceeded = True
                    break
                frame = _frame_from_calculation(element)
                if frame is None:
                    return None
                frames.append(frame)
                element.clear()
    except (ET.ParseError, OSError, ValueError, IndexError):
        return None
    if nelm is None or not species or not frames:
        return None
    if frames[-1]["lattice_angstrom"] is None:
        frames[-1]["lattice_angstrom"] = final_lattice
    if frames[-1]["fractional_coordinates"] is None:
        frames[-1]["fractional_coordinates"] = final_positions
    if any(
        frame["lattice_angstrom"] is None or frame["fractional_coordinates"] is None
        for frame in frames
    ):
        return None
    return {
        "nelm": nelm,
        "species": species,
        "frames": frames,
        "frame_limit_exceeded": exceeded,
        # The document-level finalpos is the geometry *after* the last ionic
        # update, so it matches CONTCAR but not the last force evaluation.  It is
        # reported separately rather than mixed into the frames.
        "final_lattice_angstrom": final_lattice,
        "final_fractional_coordinates": final_positions,
    }


def _parse_scheduled_vasprun(path: Path) -> dict[str, Any] | None:
    """Final-frame view of one vasprun.xml, used by the single-frame paths.

    Built on the streaming trajectory parser so static, relax and AIMD all read
    NELM and the sigma->0 energy through exactly one implementation.  Two real
    VASP 5.4.x layouts are handled there and must stay handled: NELM appears
    twice (electronic convergence, and 1 under response functions), and the
    final <calculation><energy> block reports the electronic entropy in
    e_0_energy while the real E0 lives in the last <scstep>.
    """

    parsed = _parse_vasprun_trajectory(path)
    if parsed is None or parsed["frame_limit_exceeded"]:
        return None
    final = parsed["frames"][-1]
    value = {
        "energy_ev": final["energy_ev"],
        "species": parsed["species"],
        "lattice_angstrom": final["lattice_angstrom"],
        "fractional_coordinates": final["fractional_coordinates"],
        "forces_ev_per_angstrom": final["forces_ev_per_angstrom"],
        "stress_kbar_vasp_3x3": final["stress_kbar_vasp_3x3"],
        "electronic_steps": final["electronic_steps"],
        "nelm": parsed["nelm"],
    }
    if any(
        value[key] is None
        for key in (
            "lattice_angstrom",
            "fractional_coordinates",
            "forces_ev_per_angstrom",
            "stress_kbar_vasp_3x3",
        )
    ):
        return None
    return value


def _verify_scheduled_vasp_result(
    context: Mapping[str, Any], manifest: Mapping[str, Any], verify_files: bool
) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    execution = manifest.get("execution")
    planned = contracts._mapping(contracts._mapping(context.get("execution")).get("plan"))
    scheduled = contracts._mapping(planned.get("scheduled_execution"))
    hpc_execution = contracts._mapping(contracts._mapping(context.get("execution")).get("hpc_execution"))
    if (
        not isinstance(execution, Mapping)
        or execution.get("template_family") != scheduled.get("template_family")
        or execution.get("templates") != hpc_execution.get("template_paths")
    ):
        diagnostics.append(
            contracts._diagnostic("error", "result.execution", "结果引用的远端模板与批准计划不一致。")
        )
    raw = manifest.get("raw_outputs")
    required = {"OUTCAR", "OSZICAR", "vasprun.xml"}
    if not isinstance(raw, Mapping) or set(raw) != required:
        diagnostics.append(
            contracts._diagnostic(
                "error", "result.raw_outputs", "必须且只能声明 OUTCAR、OSZICAR、vasprun.xml。"
            )
        )
        return diagnostics
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    for name in sorted(required):
        record = raw.get(name)
        path = attempt / name
        if not isinstance(record, Mapping) or record.get("path") != name:
            diagnostics.append(
                contracts._diagnostic("error", f"result.raw_outputs.{name}", "原始 VASP 输出记录无效。")
            )
            continue
        if verify_files and not contracts._ordinary_file(path):
            diagnostics.append(
                contracts._diagnostic("error", f"result.raw_outputs.{name}", "拉回的原始 VASP 输出缺失。")
            )
    if not verify_files or contracts._errors(diagnostics):
        return diagnostics
    outcar = (attempt / "OUTCAR").read_text(encoding="utf-8", errors="replace")
    if "General timing and accounting informations for this job:" not in outcar:
        diagnostics.append(
            contracts._diagnostic("error", "completion.outcar", "OUTCAR 不含正常结束 footer。")
        )
    parsed = _parse_scheduled_vasprun(attempt / "vasprun.xml")
    if parsed is None:
        diagnostics.append(contracts._diagnostic("error", "completion.vasprun", "vasprun.xml 无法完整解析。"))
        return diagnostics
    if not (0 < parsed["electronic_steps"] < parsed["nelm"]):
        diagnostics.append(
            contracts._diagnostic("error", "completion.electronic", "独立解析显示电子步达到 NELM 或为空。")
        )
    if isinstance(execution, Mapping) and (
        execution.get("electronic_steps") != parsed["electronic_steps"]
        or execution.get("nelm") != parsed["nelm"]
    ):
        diagnostics.append(
            contracts._diagnostic("error", "result.execution.steps", "电子步数/NELM 与独立解析不一致。")
        )
    artifacts = manifest.get("artifacts")
    labels_record = (
        next(
            (
                item
                for item in artifacts
                if isinstance(item, Mapping) and item.get("name") == "labels-json"
            ),
            None,
        )
        if isinstance(artifacts, list)
        else None
    )
    labels_path = (
        attempt / str(labels_record.get("path", ""))
        if isinstance(labels_record, Mapping)
        else attempt / ""
    )
    labels, _ = contracts._read_json(labels_path)
    records = labels.get("records") if isinstance(labels, Mapping) else None
    record = (
        records[0]
        if isinstance(records, list) and len(records) == 1 and isinstance(records[0], Mapping)
        else None
    )
    if record is None:
        diagnostics.append(
            contracts._diagnostic("error", "result.labels", "labels.json 必须包含恰好一个标签记录。")
        )
        return diagnostics
    expected = {
        key: parsed[key]
        for key in (
            "energy_ev",
            "species",
            "lattice_angstrom",
            "fractional_coordinates",
            "forces_ev_per_angstrom",
            "stress_kbar_vasp_3x3",
        )
    }
    if any(not _same_numeric_tree(record.get(key), value) for key, value in expected.items()):
        diagnostics.append(
            contracts._diagnostic(
                "error",
                "result.labels.values",
                "标签能量/结构/力/应力与独立 vasprun.xml 解析不一致。",
            )
        )
    return diagnostics


def _scheduled_calculation_type(context: Mapping[str, Any]) -> str:
    """Calculation type of the approved plan, defaulting to static."""

    planned = contracts._mapping(contracts._mapping(context.get("execution")).get("plan"))
    declared = planned.get("calculation_type")
    if declared in contracts.SCHEDULED_TYPES:
        return str(declared)
    prepared, _ = contracts._read_json(
        contracts._path(context["project_root"], contracts._mapping(context["inputs"])["dft_input_manifest"])
    )
    value = prepared.get("calculation_type") if isinstance(prepared, Mapping) else None
    return str(value) if value in contracts.SCHEDULED_TYPES else contracts.STATIC_TYPE


def _requested_nsw(context: Mapping[str, Any]) -> int | None:
    config, _ = contracts._parse_labeling_contract(
        contracts._path(context["project_root"], contracts._mapping(context["inputs"])["labeling_config"])
    )
    if config is None:
        return None
    value = contracts._mapping(config.get("effective_incar")).get("NSW")
    return int(value) if contracts._positive_int(value) else None


def _check_one_calculation(
    attempt: Path,
    calc_id: str,
    calculation_type: str,
    expected_atoms: Any,
    requested_nsw: int | None,
) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    """Scientific completion for a single calculation.

    Every diagnostic is prefixed with the calculation id, so a batch failure
    names the calculation and the reason instead of collapsing into
    "batch failed".
    """

    diagnostics: list[dict[str, str]] = []
    directory = attempt / calc_id
    required, _, limits = _scheduled_output_spec(calculation_type)
    for name in required:
        path = directory / name
        if not contracts._ordinary_file(path) or path.stat().st_size == 0:
            diagnostics.append(
                contracts._diagnostic("error", f"{calc_id}.raw.{name}", f"{calc_id} 缺少非空的 {name}。")
            )
        elif path.stat().st_size > limits.get(name, 0):
            diagnostics.append(
                contracts._diagnostic(
                    "error", f"{calc_id}.raw.{name}", f"{calc_id} 的 {name} 超过已批准的大小上限。"
                )
            )
    if contracts._errors(diagnostics):
        return diagnostics, None

    outcar = (directory / "OUTCAR").read_text(encoding="utf-8", errors="replace")
    if "General timing and accounting informations for this job:" not in outcar:
        diagnostics.append(
            contracts._diagnostic(
                "error",
                f"{calc_id}.completion.outcar",
                f"{calc_id} 的 OUTCAR 不含正常结束 footer。",
            )
        )
    max_frames = contracts.MAX_AIMD_FRAMES if calculation_type == contracts.AIMD_TYPE else 1024
    parsed = _parse_vasprun_trajectory(directory / "vasprun.xml", max_frames)
    if parsed is None:
        diagnostics.append(
            contracts._diagnostic(
                "error", f"{calc_id}.completion.vasprun", f"{calc_id} 的 vasprun.xml 无法完整解析。"
            )
        )
        return diagnostics, None
    if parsed["frame_limit_exceeded"]:
        diagnostics.append(
            contracts._diagnostic(
                "error",
                f"{calc_id}.completion.frames",
                f"{calc_id} 的 ionic step 数超过已批准上限 {max_frames}。",
            )
        )
        return diagnostics, None

    frames = parsed["frames"]
    nelm = parsed["nelm"]
    # Electronic convergence is required for every ionic step, not just the last
    # one: a single unconverged SCF step poisons that frame's forces.
    for position, frame in enumerate(frames, start=1):
        if not 0 < frame["electronic_steps"] < nelm:
            diagnostics.append(
                contracts._diagnostic(
                    "error",
                    f"{calc_id}.completion.electronic",
                    f"{calc_id} 的第 {position} 个 ionic step 电子步达到 NELM({nelm}) 或为空。",
                )
            )
    species = parsed["species"]
    if contracts._positive_int(expected_atoms) and len(species) != int(expected_atoms):
        diagnostics.append(
            contracts._diagnostic(
                "error", f"{calc_id}.result.atom_count", f"{calc_id} 的原子数与准备清单不一致。"
            )
        )
    for position, frame in enumerate(frames, start=1):
        if len(frame["fractional_coordinates"]) != len(species) or len(
            frame["forces_ev_per_angstrom"]
        ) != len(species):
            diagnostics.append(
                contracts._diagnostic(
                    "error",
                    f"{calc_id}.result.frame_shape",
                    f"{calc_id} 的第 {position} 帧原子数与 species 不一致。",
                )
            )

    ionic_converged: bool | None = None
    if calculation_type == contracts.STATIC_TYPE:
        if len(frames) != 1:
            diagnostics.append(
                contracts._diagnostic(
                    "error",
                    f"{calc_id}.completion.ionic",
                    f"{calc_id} 是 static，必须恰好一个 ionic step。",
                )
            )
    elif calculation_type == contracts.RELAX_TYPE:
        # VASP states ionic convergence explicitly; a zero exit code does not.
        ionic_converged = contracts.RELAX_CONVERGENCE_MARKER in outcar
        if not ionic_converged:
            diagnostics.append(
                contracts._diagnostic(
                    "error",
                    f"{calc_id}.completion.ionic",
                    f"{calc_id} 的离子步未收敛：OUTCAR 未报告 reached required accuracy。",
                )
            )
        if requested_nsw is not None and len(frames) > requested_nsw:
            diagnostics.append(
                contracts._diagnostic(
                    "error",
                    f"{calc_id}.completion.ionic_steps",
                    f"{calc_id} 的 ionic step 数超过 NSW。",
                )
            )
        contcar = directory / "CONTCAR"
        if not contracts._ordinary_file(contcar) or contcar.stat().st_size == 0:
            diagnostics.append(
                contracts._diagnostic("error", f"{calc_id}.raw.CONTCAR", f"{calc_id} 缺少非空的 CONTCAR。")
            )
        else:
            contcar_lattice = _contcar_lattice(contcar)
            # CONTCAR holds the geometry *after* the final ionic update, which is
            # what vasprun reports as finalpos -- not the geometry the last
            # forces were evaluated at.  Comparing it against the last frame
            # would flag every converged relaxation as inconsistent.
            reference = parsed.get("final_lattice_angstrom") or frames[-1]["lattice_angstrom"]
            if contcar_lattice is None:
                diagnostics.append(
                    contracts._diagnostic(
                        "error", f"{calc_id}.result.contcar", f"{calc_id} 的 CONTCAR 无法解析。"
                    )
                )
            elif not _same_lattice_across_formats(contcar_lattice, reference):
                diagnostics.append(
                    contracts._diagnostic(
                        "error",
                        f"{calc_id}.result.contcar",
                        f"{calc_id} 的 CONTCAR 晶格与 vasprun finalpos 不一致。",
                    )
                )
    elif calculation_type == contracts.AIMD_TYPE:
        # AIMD is not a structural optimisation, so ionic convergence is not a
        # completion criterion; the trajectory being complete is.
        if requested_nsw is None:
            diagnostics.append(
                contracts._diagnostic(
                    "error",
                    f"{calc_id}.completion.nsw",
                    f"{calc_id} 的 labeling_config 未声明可用的 NSW。",
                )
            )
        elif len(frames) != requested_nsw:
            diagnostics.append(
                contracts._diagnostic(
                    "error",
                    f"{calc_id}.completion.trajectory",
                    f"{calc_id} 完成 {len(frames)} 个 ionic step，与请求的 NSW={requested_nsw} 不一致。",
                )
            )
    return diagnostics, {
        "calc_id": calc_id,
        "calculation_type": calculation_type,
        "species": species,
        "nelm": nelm,
        "frames": frames,
        "ionic_converged": ionic_converged,
        "final_lattice_angstrom": parsed.get("final_lattice_angstrom"),
        "final_fractional_coordinates": parsed.get("final_fractional_coordinates"),
        "outcar_text_head": outcar[:4096],
    }


def _same_lattice_across_formats(actual: Any, expected: Any) -> bool:
    """Compare a CONTCAR lattice with a vasprun lattice.

    These are two different files at two different printed precisions: CONTCAR
    carries ~16 significant digits while vasprun.xml prints 8 decimals, so the
    same cell differs by ~1e-9.  The strict 1e-10 tolerance used elsewhere is
    correct when re-parsing one file twice, but here it would reject every real
    relaxation.  The bound below is tied to vasprun's printed precision.
    """

    if not isinstance(actual, list) or not isinstance(expected, list):
        return False
    if len(actual) != len(expected):
        return False
    for actual_row, expected_row in zip(actual, expected):
        if not isinstance(actual_row, list) or len(actual_row) != len(expected_row):
            return False
        for left, right in zip(actual_row, expected_row):
            if not contracts._finite_number(left) or not contracts._finite_number(right):
                return False
            if not math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-6):
                return False
    return True


def _contcar_lattice(path: Path) -> list[list[float]] | None:
    """Read the three lattice vectors from a CONTCAR, scale applied."""

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        scale = float(lines[1].split()[0])
        rows = [[float(x) * scale for x in lines[i].split()[:3]] for i in (2, 3, 4)]
    except (OSError, ValueError, IndexError):
        return None
    if any(len(row) != 3 for row in rows):
        return None
    return rows


def _scheduled_calculations_check(
    context: Mapping[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, Any]] | None]:
    """Check every calculation in the current attempt as one complete batch."""

    diagnostics: list[dict[str, str]] = []
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    calculation_type = _scheduled_calculation_type(context)
    requested_nsw = _requested_nsw(context)
    prepared, _ = contracts._read_json(
        contracts._path(context["project_root"], contracts._mapping(context["inputs"])["dft_input_manifest"])
    )
    calculations = prepared.get("calculations") if isinstance(prepared, Mapping) else None
    if not isinstance(calculations, list) or not calculations:
        diagnostics.append(
            contracts._diagnostic("error", "result.calculations", "准备清单缺少 calculations。")
        )
        return diagnostics, None
    expected_ids = [_calculation_id(index) for index in range(len(calculations))]
    diagnostics.extend(_verify_completion_calculations(context, expected_ids))
    completion, _ = contracts._read_json(attempt / "completion.json")
    source_identity = (
        {
            "project_id": completion.get("project_id"),
            "node_id": completion.get("node_id"),
            "attempt": completion.get("attempt"),
        }
        if isinstance(completion, Mapping)
        else None
    )
    if (
        not isinstance(source_identity, Mapping)
        or not isinstance(source_identity.get("project_id"), str)
        or not contracts.SAFE_ID.fullmatch(str(source_identity["project_id"]))
        or not isinstance(source_identity.get("node_id"), str)
        or not contracts.SAFE_ID.fullmatch(str(source_identity["node_id"]))
        or source_identity.get("attempt") != contracts._attempt_index(attempt)
    ):
        diagnostics.append(
            contracts._diagnostic(
                "error",
                "completion.identity",
                "completion.json 未绑定当前 project/node/attempt。",
            )
        )
        return diagnostics, None
    analyses: list[dict[str, Any]] = []
    for index, calculation in enumerate(calculations):
        calc_id = _calculation_id(index)
        expected_atoms = calculation.get("atom_count") if isinstance(calculation, Mapping) else None
        calc_diagnostics, analysis = _check_one_calculation(
            attempt, calc_id, calculation_type, expected_atoms, requested_nsw
        )
        diagnostics.extend(calc_diagnostics)
        if analysis is not None:
            analysis["structure_id"] = (
                calculation.get("structure_id") if isinstance(calculation, Mapping) else None
            )
            analysis["source_dft_attempt"] = source_identity
            analyses.append(analysis)
    if contracts._errors(diagnostics):
        return diagnostics, None
    return diagnostics, analyses


def _verify_completion_calculations(
    context: Mapping[str, Any], expected_ids: list[str]
) -> list[dict[str, str]]:
    """Cross-check the remote completion record against the approved batch.

    The core only guarantees project/node/attempt identity and the overall exit
    status; the per-calculation list is a plugin-level contract, so it is
    verified here rather than in the scheduler lifecycle.
    """

    diagnostics: list[dict[str, str]] = []
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    completion, _ = contracts._read_json(attempt / "completion.json")
    if not isinstance(completion, Mapping):
        return diagnostics
    reported = completion.get("calculations")
    if reported is None:
        return diagnostics
    if (
        not isinstance(reported, list)
        or [item.get("id") if isinstance(item, Mapping) else None for item in reported]
        != expected_ids
    ):
        diagnostics.append(
            contracts._diagnostic(
                "error",
                "completion.calculations",
                "completion.json 的 calculation 列表与批准计划不一致。",
            )
        )
        return diagnostics
    for item in reported:
        if item.get("exit_code") != 0:
            diagnostics.append(
                contracts._diagnostic(
                    "error",
                    f"{item.get('id')}.completion.exit_code",
                    f"{item.get('id')} 的 VASP 进程退出码为 {item.get('exit_code')!r}。",
                )
            )
    return diagnostics


def _write_fresh_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"refusing to overwrite result: {path}")
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _collect_scheduled_result(context: Mapping[str, Any]) -> dict[str, Any]:
    """Build the labelled dataset for a whole batch.

    Label cardinality follows the calculation type: static and relax yield one
    label per source structure (the single, respectively final converged,
    configuration), while AIMD yields one label per ionic step.  Every label
    carries back-references so a row can be traced to its source structure,
    calculation and ionic step.
    """

    diagnostics, analyses = _scheduled_calculations_check(context)
    if contracts._errors(diagnostics) or analyses is None:
        return {"plugin_id": contracts.PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    parameters = contracts._mapping(context["parameters"])
    inputs = contracts._mapping(context["inputs"])
    calculation_type = _scheduled_calculation_type(context)

    records: list[dict[str, Any]] = []
    frame_count = 0
    for analysis in analyses:
        frames = analysis["frames"]
        selected = (
            list(enumerate(frames, start=1))
            if calculation_type == contracts.AIMD_TYPE
            # static has one frame; relax labels only the final converged
            # configuration.  Intermediate relaxation steps are deliberately not
            # treated as training labels in this change.
            else [(len(frames), frames[-1])]
        )
        frame_count += len(frames)
        for ionic_step, frame in selected:
            records.append(
                {
                    "structure_id": analysis.get("structure_id"),
                    "calculation_id": analysis["calc_id"],
                    "calculation_type": calculation_type,
                    "source_dft_attempt": analysis["source_dft_attempt"],
                    "ionic_step": ionic_step,
                    "species": analysis["species"],
                    "energy_ev": frame["energy_ev"],
                    "lattice_angstrom": frame["lattice_angstrom"],
                    "fractional_coordinates": frame["fractional_coordinates"],
                    "forces_ev_per_angstrom": frame["forces_ev_per_angstrom"],
                    "stress_kbar_vasp_3x3": frame["stress_kbar_vasp_3x3"],
                }
            )
    raw_outputs: dict[str, Any] = {}
    required, optional, _ = _scheduled_output_spec(calculation_type)
    for analysis in analyses:
        calc_id = analysis["calc_id"]
        for name in (*required, *optional):
            path = attempt / calc_id / name
            if not contracts._ordinary_file(path):
                continue
            raw_outputs[f"{calc_id}/{name}"] = {
                "path": f"{calc_id}/{name}",
                "source_dft_attempt": analysis["source_dft_attempt"],
            }
    labels_path = attempt / "labels.json"
    _write_fresh_json(
        labels_path,
        {"schema_version": 2, "records": records, "units": parameters["units"]},
    )
    first = analyses[0]
    version = re.search(r"\bvasp\.([0-9][A-Za-z0-9._-]*)", first["outcar_text_head"], re.I)
    ionic_required = calculation_type == contracts.RELAX_TYPE
    ionic_converged = (
        all(item["ionic_converged"] is True for item in analyses) if ionic_required else None
    )
    execution_context = contracts._mapping(context.get("execution"))
    hpc_execution = contracts._mapping(execution_context.get("hpc_execution"))
    scheduled_plan = contracts._mapping(contracts._mapping(execution_context.get("plan")).get("scheduled_execution"))
    completion_path = attempt / "completion.json"
    completion_identity, completion_error = contracts._read_json(completion_path)
    if completion_identity is None:
        return {
            "plugin_id": contracts.PLUGIN_ID,
            "status": "FAIL",
            "diagnostics": [
                completion_error
                or contracts._diagnostic(
                    "error", "completion.identity", "缺少 scheduler completion identity。"
                )
            ],
        }
    source_attempt_identity = {
        "project_id": completion_identity.get("project_id"),
        "node_id": completion_identity.get("node_id"),
        "attempt": completion_identity.get("attempt"),
    }
    execution = {
        "template_family": scheduled_plan.get("template_family"),
        "templates": hpc_execution.get("template_paths"),
        "vasp_version": version.group(1) if version else "UNKNOWN",
        "calculations": [
            {
                "id": item["calc_id"],
                "structure_id": item.get("structure_id"),
                "source_dft_attempt": item["source_dft_attempt"],
                "nelm": item["nelm"],
                "ionic_steps": len(item["frames"]),
                "ionic_converged": item["ionic_converged"],
                "relaxed_structure": (
                    {
                        "lattice_angstrom": item["final_lattice_angstrom"],
                        "fractional_coordinates": item["final_fractional_coordinates"],
                    }
                    if calculation_type == contracts.RELAX_TYPE
                    else None
                ),
            }
            for item in analyses
        ],
    }
    structures_path = contracts._path(context["project_root"], inputs["structures_manifest"])
    structures_manifest, structure_error = contracts._read_json(structures_path)
    if structures_manifest is None:
        return {
            "plugin_id": contracts.PLUGIN_ID,
            "status": "FAIL",
            "diagnostics": [
                structure_error
                or contracts._diagnostic("error", "dataset.structures", "无法读取 structures manifest。")
            ],
        }
    try:
        canonical = DATASETS.build_canonical_dataset(
            label_records=records,
            units=parameters["units"],
            calculation_type=calculation_type,
            source_attempt_identity=source_attempt_identity,
            structures_manifest=structures_manifest,
            raw_outputs=raw_outputs,
        )
    except ValueError as exc:
        return {
            "plugin_id": contracts.PLUGIN_ID,
            "status": "FAIL",
            "diagnostics": [
                contracts._diagnostic("error", "dataset.canonical", f"canonical dataset 组装失败：{exc}")
            ],
        }
    canonical_path = attempt / "canonical-labeled-dataset.json"
    _write_fresh_json(canonical_path, canonical)
    artifact_paths = {
        "labels-json": labels_path,
        "canonical-labeled-dataset": canonical_path,
    }
    manifest = {
        "schema_version": 3,
        "plugin_id": contracts.PLUGIN_ID,
        "status": "OK",
        "engine": "vasp",
        "calculation_type": calculation_type,
        "completion": {
            "scheduler_success": True,
            "electronic_converged": True,
            "ionic_convergence_required": ionic_required,
            "ionic_converged": ionic_converged,
            "truncated": False,
        },
        "units": parameters["units"],
        "source_count": len(analyses),
        "calculation_count": len(analyses),
        "frame_count": frame_count,
        "label_count": len(records),
        "labels": "labels.json",
        "execution": execution,
        "raw_outputs": raw_outputs,
        "dataset_id": canonical["dataset_id"],
        "canonical_dataset": canonical_path.name,
        "artifacts": [
            {
                "name": name,
                "path": path.name,
                "media_type": "application/json",
            }
            for name, path in artifact_paths.items()
        ],
    }
    result_path = attempt / str(parameters.get("result_manifest", "dft-labeling-result.json"))
    _write_fresh_json(result_path, manifest)
    diagnostics.extend(_verify_scheduled_label_result(context, manifest))
    if contracts._errors(diagnostics):
        return {"plugin_id": contracts.PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
    return _collected_scheduled_label(context, manifest, diagnostics)


def _verify_scheduled_label_result(
    context: Mapping[str, Any], manifest: Mapping[str, Any]
) -> list[dict[str, str]]:
    """Schema-2 verification of a scheduled result manifest.

    Kept separate from the local wrapper contract (schema 1), whose
    "one label per source structure" rule is only correct for single static
    calculations and is wrong for AIMD.
    """

    diagnostics: list[dict[str, str]] = []
    parameters = contracts._mapping(context["parameters"])
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    calculation_type = manifest.get("calculation_type")
    if calculation_type not in contracts.SCHEDULED_TYPES:
        diagnostics.append(
            contracts._diagnostic("error", "result.calculation_type", "calculation_type 必须显式且受支持。")
        )
        return diagnostics
    if manifest.get("units") != parameters.get("units"):
        diagnostics.append(contracts._diagnostic("error", "result.units", "结果单位与批准 units 不一致。"))
    completion = contracts._mapping(manifest.get("completion"))
    if (
        completion.get("scheduler_success") is not True
        or completion.get("electronic_converged") is not True
    ):
        diagnostics.append(contracts._diagnostic("error", "result.completion", "completion 状态不安全。"))
    if calculation_type == contracts.RELAX_TYPE and completion.get("ionic_converged") is not True:
        diagnostics.append(
            contracts._diagnostic("error", "completion.ionic", "relax 结果必须报告 ionic_converged=true。")
        )
    if calculation_type != contracts.RELAX_TYPE and completion.get("ionic_convergence_required") is not False:
        diagnostics.append(
            contracts._diagnostic("error", "completion.ionic_policy", "仅 relax 要求 ionic convergence。")
        )
    source_count = manifest.get("source_count")
    calculation_count = manifest.get("calculation_count")
    frame_count = manifest.get("frame_count")
    label_count = manifest.get("label_count")
    if not contracts._positive_int(source_count) or calculation_count != source_count:
        diagnostics.append(
            contracts._diagnostic(
                "error", "result.calculation_count", "calculation_count 必须等于 source_count。"
            )
        )
    if not contracts._positive_int(frame_count) or not contracts._positive_int(label_count):
        diagnostics.append(
            contracts._diagnostic("error", "result.counts", "frame_count/label_count 必须是正整数。")
        )
    elif calculation_type == contracts.AIMD_TYPE:
        if label_count != frame_count:
            diagnostics.append(
                contracts._diagnostic("error", "result.label_count", "AIMD 每个 frame 必须恰有一个标签。")
            )
    elif label_count != source_count:
        diagnostics.append(
            contracts._diagnostic("error", "result.label_count", "static/relax 每个源结构必须恰有一个标签。")
        )
    labels_path = attempt / str(manifest.get("labels", "labels.json"))
    labels, _ = contracts._read_json(labels_path)
    records = labels.get("records") if isinstance(labels, Mapping) else None
    if not isinstance(records, list) or len(records) != label_count:
        diagnostics.append(
            contracts._diagnostic("error", "result.labels", "labels.json 记录数与 label_count 不一致。")
        )
        return diagnostics
    completion, _ = contracts._read_json(attempt / "completion.json")
    expected_source_attempt = (
        {
            "project_id": completion.get("project_id"),
            "node_id": completion.get("node_id"),
            "attempt": completion.get("attempt"),
        }
        if isinstance(completion, Mapping)
        else None
    )
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            diagnostics.append(contracts._diagnostic("error", f"labels[{index}]", "标签必须是对象。"))
            continue
        if record.get("calculation_type") != calculation_type or not contracts._plain_string(
            record.get("calculation_id")
        ):
            diagnostics.append(
                contracts._diagnostic(
                    "error",
                    f"labels[{index}].traceability",
                    "标签必须记录 calculation id 与 calculation_type。",
                )
            )
        if not contracts._positive_int(record.get("ionic_step")):
            diagnostics.append(
                contracts._diagnostic("error", f"labels[{index}].ionic_step", "标签必须记录 ionic step。")
            )
        if not contracts._finite_number(record.get("energy_ev")):
            diagnostics.append(
                contracts._diagnostic("error", f"labels[{index}].energy", "标签能量必须是有限数。")
            )
        source_attempt = record.get("source_dft_attempt")
        if manifest.get("schema_version") == 3 and source_attempt != expected_source_attempt:
            diagnostics.append(
                contracts._diagnostic(
                    "error",
                    f"labels[{index}].source_dft_attempt",
                    "canonical 标签必须记录当前 DFT attempt。",
                )
            )
    raw = manifest.get("raw_outputs")
    if not isinstance(raw, Mapping) or not raw:
        diagnostics.append(contracts._diagnostic("error", "result.raw_outputs", "必须声明 raw VASP 输出。"))
        return diagnostics
    for name, record in raw.items():
        if "POTCAR" in str(name):
            diagnostics.append(
                contracts._diagnostic("error", "result.raw_outputs.potcar", "POTCAR 不得出现在结果产物中。")
            )
            continue
        source_attempt = record.get("source_dft_attempt") if isinstance(record, Mapping) else None
        path = attempt / str(name)
        if (
            not isinstance(record, Mapping)
            or record.get("path") != name
            or source_attempt != expected_source_attempt
            or not contracts._ordinary_file(path)
        ):
            diagnostics.append(
                contracts._diagnostic("error", f"result.raw_outputs.{name}", "原始 VASP 输出记录无效。")
            )
    if manifest.get("schema_version") == 3:
        diagnostics.extend(_verify_canonical_label_bundle(context, manifest, records))
    elif manifest.get("schema_version") != 2:
        diagnostics.append(
            contracts._diagnostic(
                "error", "result.schema_version", "scheduled label 结果 schema 必须是 2 或 3。"
            )
        )
    return diagnostics


def _verify_canonical_label_bundle(
    context: Mapping[str, Any],
    manifest: Mapping[str, Any],
    label_records: list[Any],
) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute()
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return [
            contracts._diagnostic("error", "dataset.artifacts", "canonical dataset artifacts 必须是列表。")
        ]
    by_name = {
        item.get("name"): item
        for item in artifacts
        if isinstance(item, Mapping) and contracts._plain_string(item.get("name"))
    }
    if set(by_name) != {"labels-json", "canonical-labeled-dataset"}:
        return [
            contracts._diagnostic(
                "error", "dataset.artifacts.roles", "只应收集 labels 与 canonical dataset。"
            )
        ]
    for name, item in by_name.items():
        path = attempt / str(item.get("path"))
        if not contracts._safe_relative(item.get("path")) or not contracts._ordinary_file(path):
            diagnostics.append(
                contracts._diagnostic("error", f"dataset.artifacts.{name}", "artifact 路径无效。")
            )
    canonical_path = attempt / str(by_name["canonical-labeled-dataset"]["path"])
    canonical, _ = contracts._read_json(canonical_path)
    completion, _ = contracts._read_json(attempt / "completion.json")
    structures_path = contracts._path(
        context["project_root"], contracts._mapping(context["inputs"])["structures_manifest"]
    )
    structures, _ = contracts._read_json(structures_path)
    if canonical is None or completion is None or structures is None:
        return diagnostics + [contracts._diagnostic("error", "dataset.json", "canonical 重建输入不可读。")]
    try:
        expected = DATASETS.build_canonical_dataset(
            label_records=label_records,
            units=contracts._mapping(context["parameters"])["units"],
            calculation_type=str(manifest["calculation_type"]),
            source_attempt_identity={
                "project_id": completion.get("project_id"),
                "node_id": completion.get("node_id"),
                "attempt": completion.get("attempt"),
            },
            structures_manifest=structures,
            raw_outputs=contracts._mapping(manifest.get("raw_outputs")),
        )
    except (ValueError, KeyError) as exc:
        return diagnostics + [
            contracts._diagnostic("error", "dataset.rebuild", f"canonical dataset 无法重建：{exc}")
        ]
    if canonical != expected:
        diagnostics.append(
            contracts._diagnostic(
                "error",
                "dataset.canonical_identity",
                "canonical dataset 不是当前标签的确定性序列化。",
            )
        )
    diagnostics.extend(
        contracts._diagnostic("error", "dataset.canonical_contract", message)
        for message in DATASETS.validate_canonical_dataset(canonical)
    )
    if (
        manifest.get("dataset_id") != expected["dataset_id"]
        or manifest.get("canonical_dataset") != canonical_path.name
    ):
        diagnostics.append(
            contracts._diagnostic("error", "dataset.result", "结果未绑定 canonical dataset identity。")
        )
    return diagnostics


def _collected_scheduled_label(
    context: Mapping[str, Any],
    manifest: Mapping[str, Any],
    diagnostics: list[dict[str, str]],
) -> dict[str, Any]:
    parameters = contracts._mapping(context["parameters"])
    result_name = str(parameters.get("result_manifest", "dft-labeling-result.json"))
    if manifest.get("schema_version") == 2:
        return {
            "plugin_id": contracts.PLUGIN_ID,
            "status": "OK",
            "diagnostics": diagnostics,
            "artifacts": [
                {"path": "labels.json", "role": "labels", "media_type": "application/json"},
                {"path": result_name, "role": "dft-label-result", "media_type": "application/json"},
            ],
            "metrics": {
                "source_count": float(manifest["source_count"]),
                "calculation_count": float(manifest["calculation_count"]),
                "frame_count": float(manifest["frame_count"]),
                "label_count": float(manifest["label_count"]),
            },
        }
    artifacts = [
        {
            "name": "dft-labeling-result",
            "role": "dft-label-result",
            "path": result_name,
            "media_type": "application/json",
        },
        *[{**dict(item), "role": item["name"]} for item in manifest["artifacts"]],
    ]
    if manifest.get("calculation_type") == contracts.AIMD_TYPE:
        artifacts.extend(
            {
                "name": f"aimd-trajectory-{index}",
                "role": "aimd-trajectory",
                "path": name,
                "media_type": "application/xml",
            }
            for index, name in enumerate(sorted(contracts._mapping(manifest.get("raw_outputs"))), start=1)
            if name.endswith("/vasprun.xml")
        )
    return {
        "plugin_id": contracts.PLUGIN_ID,
        "status": "OK",
        "diagnostics": diagnostics,
        "artifacts": artifacts,
        "metrics": {
            "source_structure_count": manifest["source_count"],
            "label_count": manifest["label_count"],
            "dataset_id": manifest["dataset_id"],
            "electronic_converged": True,
            "ionic_converged": manifest["completion"].get("ionic_converged"),
        },
    }
