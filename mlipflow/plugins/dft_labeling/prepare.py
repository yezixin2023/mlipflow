"""dft labeling: prepare."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from . import contracts


def _validate_prepare(
    context: Mapping[str, Any], require_fresh_outputs: bool = True
) -> list[dict[str, str]]:
    diagnostics = contracts._base_diagnostics(context)
    inputs = contracts._mapping(context.get("inputs"))
    parameters = contracts._mapping(context.get("parameters"))
    for key in ("structures_manifest", "labeling_config", "pseudopotential_reference"):
        path = contracts._validate_input_path(diagnostics, context, key)
        if path is not None:
            project_root = Path(str(context.get("project_root"))).expanduser().absolute()
            if not contracts._ordinary_project_file(path, project_root):
                diagnostics.append(
                    contracts._diagnostic("error", f"inputs.{key}.missing", f"{key} 必须是现有普通文件。")
                )
    structures_path = contracts._project_input_path(
        context.get("project_root"), inputs.get("structures_manifest")
    )
    if structures_path is not None:
        structures_manifest, _ = contracts._read_json(structures_path)
        structures = (
            structures_manifest.get("structures")
            if isinstance(structures_manifest, Mapping)
            else None
        )
        if isinstance(structures, list):
            for index, structure in enumerate(structures):
                if not isinstance(structure, Mapping):
                    continue
                source = contracts._structure_input_path(
                    structures_path,
                    structure.get("path", structure.get("output_file")),
                )
                code = f"inputs.structures_manifest.structures[{index}].path"
                if source is None:
                    diagnostics.append(contracts._diagnostic("error", code, "结构输入路径必须是非空字符串。"))
                elif not source.exists():
                    diagnostics.append(
                        contracts._diagnostic("error", f"{code}.missing", f"结构输入文件不存在：{source}")
                    )
                elif not contracts._readable_structure_file(source):
                    diagnostics.append(
                        contracts._diagnostic("error", code, f"结构输入必须是非空可读普通文件：{source}")
                    )
    allowed_parameters = {
        "operation",
        "interpreter_argv",
        "engine",
        "result_manifest",
        "output_subdir",
        "max_structures",
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
    if parameters.get("engine") != "vasp":
        diagnostics.append(
            contracts._diagnostic("error", "parameters.engine", "vasp-prepare 的 engine 必须是 vasp。")
        )
    interpreter = parameters.get("interpreter_argv")
    executable = None
    if isinstance(interpreter, list) and len(interpreter) == 1:
        executable = contracts._explicit_executable(interpreter[0])
    if executable is None or executable.name.lower() in contracts.SHELL_EXECUTABLES:
        diagnostics.append(
            contracts._diagnostic(
                "error",
                "parameters.interpreter_argv",
                "vasp-prepare 需要一个显式绝对、可执行、无符号链接的 Python 文件。",
            )
        )
    result_manifest = parameters.get("result_manifest", "dft-input-manifest.json")
    output_subdir = parameters.get("output_subdir", "vasp-inputs")
    if not contracts._safe_relative(result_manifest) or len(Path(str(result_manifest)).parts) != 1:
        diagnostics.append(
            contracts._diagnostic(
                "error",
                "parameters.result_manifest",
                "result_manifest 必须是 attempt_dir 的直接子文件。",
            )
        )
    if not contracts._safe_relative(output_subdir):
        diagnostics.append(
            contracts._diagnostic("error", "parameters.output_subdir", "output_subdir 必须是安全相对路径。")
        )
    elif (
        contracts._safe_relative(result_manifest)
        and Path(str(result_manifest)) in Path(str(output_subdir)).parents
    ):
        diagnostics.append(
            contracts._diagnostic(
                "error",
                "path.output_result_overlap",
                "output_subdir 不得位于 result_manifest 同名路径下。",
            )
        )
    elif contracts._safe_relative(result_manifest) and str(output_subdir) == str(result_manifest):
        diagnostics.append(
            contracts._diagnostic(
                "error", "path.output_result_overlap", "output_subdir 不得与 result_manifest 相同。"
            )
        )
    max_structures = parameters.get("max_structures", contracts.MAX_STRUCTURES)
    if not contracts._positive_int(max_structures) or int(max_structures) > contracts.MAX_STRUCTURES:
        diagnostics.append(
            contracts._diagnostic(
                "error", "parameters.max_structures", f"max_structures 必须在 1..{contracts.MAX_STRUCTURES}。"
            )
        )
    attempt = Path(str(context.get("attempt_dir", ""))).expanduser().absolute()
    if (
        require_fresh_outputs
        and contracts._safe_relative(result_manifest)
        and (attempt / str(result_manifest)).exists()
    ):
        diagnostics.append(
            contracts._diagnostic("error", "path.result_exists", "vasp-prepare 要求新的 result_manifest。")
        )
    if (
        require_fresh_outputs
        and contracts._safe_relative(output_subdir)
        and (attempt / str(output_subdir)).exists()
    ):
        diagnostics.append(
            contracts._diagnostic("error", "path.output_exists", "vasp-prepare 要求新的 output_subdir。")
        )
    labeling_path = contracts._project_input_path(context.get("project_root"), inputs.get("labeling_config"))
    if labeling_path is not None:
        config, message = contracts._parse_labeling_contract(labeling_path)
        if config is None:
            diagnostics.append(
                contracts._diagnostic("error", "inputs.labeling_config.contract", str(message))
            )
    reference_path = contracts._project_input_path(
        context.get("project_root"), inputs.get("pseudopotential_reference")
    )
    if reference_path is not None:
        reference, message = contracts._parse_pseudopotential_reference(reference_path)
        if reference is None:
            diagnostics.append(
                contracts._diagnostic("error", "inputs.pseudopotential_reference.contract", str(message))
            )
    return diagnostics


def _plan_prepare(context: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = _validate_prepare(context)
    if contracts._errors(diagnostics):
        return {
            "plugin_id": contracts.PLUGIN_ID,
            "status": "BLOCKED",
            "executable": False,
            "diagnostics": diagnostics,
        }
    inputs = contracts._mapping(context["inputs"])
    parameters = contracts._mapping(context["parameters"])
    project_root = Path(str(context["project_root"])).expanduser().absolute().resolve()
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute().resolve()
    executable = contracts._explicit_executable(parameters["interpreter_argv"][0])
    assert executable is not None
    paths = {
        "structures_manifest": contracts._path(project_root, inputs["structures_manifest"]),
        "labeling_config": contracts._path(project_root, inputs["labeling_config"]),
        "pseudopotential_reference": contracts._path(project_root, inputs["pseudopotential_reference"]),
        "python_executable": executable,
        "prepare_wrapper": contracts.BUNDLED_PREPARE_WRAPPER,
    }
    result_name = str(parameters.get("result_manifest", "dft-input-manifest.json"))
    output_subdir = str(parameters.get("output_subdir", "vasp-inputs"))
    result_path = attempt / result_name
    argv = [
        str(executable),
        str(contracts.BUNDLED_PREPARE_WRAPPER),
        "--project-root",
        str(project_root),
        "--attempt-dir",
        str(attempt),
        "--structures-manifest",
        str(paths["structures_manifest"]),
        "--labeling-config",
        str(paths["labeling_config"]),
        "--pseudopotential-reference",
        str(paths["pseudopotential_reference"]),
        "--result-manifest",
        str(result_path),
        "--output-subdir",
        output_subdir,
        "--max-structures",
        str(parameters.get("max_structures", contracts.MAX_STRUCTURES)),
    ]
    config, _ = contracts._parse_labeling_contract(paths["labeling_config"])
    reference, _ = contracts._parse_pseudopotential_reference(paths["pseudopotential_reference"])
    assert config is not None and reference is not None
    return {
        "plugin_id": contracts.PLUGIN_ID,
        "status": "READY",
        "executable": True,
        "argv": argv,
        "cwd": str(attempt),
        "expected_outputs": [str(result_path)],
        "diagnostics": diagnostics,
        "operation": contracts.PREPARE_OPERATION,
        "input_paths": contracts._file_paths(paths),
        "approval_summary": {
            "expensive": False,
            "runs_vasp": False,
            "submits_jobs": False,
            "engine": "vasp",
            "calculation_type": config["calculation_type"],
            "preset": config["preset"],
            "effective_incar": config["effective_incar"],
            "kpoints": config["kpoints"],
            "structure_count": contracts._read_structure_count(paths["structures_manifest"]),
            "pseudopotential_reference": reference["reference_id"],
            "pseudopotential_functional": reference["functional"],
            "pseudopotential_symbols": reference["symbols"],
            "potcar_from_environment": "PMG_VASP_PSP_DIR",
            "potcar_collectable": False,
            "python_executable": str(executable),
            "generator": "pymatgen (version captured at execution)",
            "output_subdir": output_subdir,
        },
    }


def _parse_incar_scalar(token: str) -> Any:
    stripped = token.strip()
    upper = stripped.upper().strip(".")
    if upper in {"TRUE", "T"}:
        return True
    if upper in {"FALSE", "F"}:
        return False
    try:
        return int(stripped)
    except ValueError:
        try:
            value = float(stripped.replace("D", "E").replace("d", "e"))
            return value if math.isfinite(value) else stripped
        except ValueError:
            return stripped


def _parse_incar_file(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].split("!", 1)[0]
        for statement in line.split(";"):
            if "=" not in statement:
                continue
            raw_key, raw_value = statement.split("=", 1)
            key = raw_key.strip().upper()
            tokens = raw_value.split()
            if key:
                result[key] = (
                    _parse_incar_scalar(tokens[0])
                    if len(tokens) == 1
                    else [_parse_incar_scalar(token) for token in tokens]
                )
    return result


def _same_incar_value(actual: Any, expected: Any) -> bool:
    if (
        isinstance(expected, float)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        return math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-15)
    if isinstance(expected, list) and isinstance(actual, list) and len(actual) == len(expected):
        return all(_same_incar_value(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, str) and isinstance(actual, str):
        return actual.lower() == expected.lower()
    return actual == expected


def _verify_kpoints(path: Path, expected: Mapping[str, Any]) -> bool:
    try:
        lines = [
            line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        mode = lines[2].lower()
        grid = [int(value) for value in lines[3].split()]
        shift = [float(value) for value in lines[4].split()] if len(lines) > 4 else [0.0, 0.0, 0.0]
    except (OSError, UnicodeError, ValueError, IndexError):
        return False
    expected_mode = str(expected["mode"])
    mode_ok = mode.startswith("g") if expected_mode == "gamma" else mode.startswith("m")
    return (
        mode_ok
        and grid == list(expected["grid"])
        and all(
            math.isclose(a, float(b), rel_tol=0, abs_tol=1e-12)
            for a, b in zip(shift, expected["shift"])
        )
    )


def _verify_prepare_result(
    context: Mapping[str, Any], manifest: Mapping[str, Any], verify_files: bool
) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    inputs = contracts._mapping(context["inputs"])
    parameters = contracts._mapping(context["parameters"])
    project_root = Path(str(context["project_root"])).expanduser().absolute().resolve()
    attempt = Path(str(context["attempt_dir"])).expanduser().absolute().resolve()
    expected_fields = {
        "schema_version": 1,
        "plugin_id": contracts.PLUGIN_ID,
        "operation": contracts.PREPARE_OPERATION,
        "status": "OK",
        "engine": "vasp",
        "execution_ready": True,
    }
    for key, expected in expected_fields.items():
        if manifest.get(key) != expected:
            diagnostics.append(
                contracts._diagnostic("error", f"result.{key}", f"结果 {key} 与已批准合同不一致。")
            )
    generator = manifest.get("generator")
    if (
        not isinstance(generator, Mapping)
        or generator.get("name") != "pymatgen"
        or not contracts._plain_string(generator.get("version"))
    ):
        diagnostics.append(
            contracts._diagnostic("error", "result.generator", "必须记录 pymatgen 名称与版本。")
        )
    runtime = manifest.get("runtime")
    interpreter = parameters.get("interpreter_argv")
    expected_executable = (
        contracts._explicit_executable(interpreter[0])
        if isinstance(interpreter, list) and len(interpreter) == 1
        else None
    )
    if not isinstance(runtime, Mapping) or (
        expected_executable is None
        or runtime.get("python_executable") != str(expected_executable)
        or not contracts._plain_string(runtime.get("python_version"))
        or not contracts._plain_string(runtime.get("pymatgen_version"))
        or not isinstance(generator, Mapping)
        or runtime.get("pymatgen_version") != generator.get("version")
    ):
        diagnostics.append(
            contracts._diagnostic(
                "error",
                "result.runtime.provenance",
                "Python executable/version 或 pymatgen version 与已批准运行时不一致。",
            )
        )
    input_paths = {
        "structures_manifest": contracts._path(project_root, inputs["structures_manifest"]),
        "labeling_config": contracts._path(project_root, inputs["labeling_config"]),
        "pseudopotential_reference": contracts._path(project_root, inputs["pseudopotential_reference"]),
    }
    if verify_files:
        for name, path in input_paths.items():
            if not contracts._ordinary_project_file(path, project_root):
                diagnostics.append(contracts._diagnostic("error", f"result.inputs.{name}", "输入文件缺失。"))
    config, config_error = contracts._parse_labeling_contract(input_paths["labeling_config"])
    reference, reference_error = contracts._parse_pseudopotential_reference(
        input_paths["pseudopotential_reference"]
    )
    if config is None:
        diagnostics.append(contracts._diagnostic("error", "result.labeling_config", str(config_error)))
        return diagnostics
    if reference is None:
        diagnostics.append(
            contracts._diagnostic("error", "result.pseudopotential_reference", str(reference_error))
        )
        return diagnostics
    if (
        manifest.get("calculation_type") != config["calculation_type"]
        or manifest.get("preset") != config["preset"]
    ):
        diagnostics.append(
            contracts._diagnostic(
                "error", "result.calculation_type", "calculation_type/preset 与输入配置不一致。"
            )
        )
    incar_record = manifest.get("incar")
    if (
        not isinstance(incar_record, Mapping)
        or incar_record.get("effective") != config["effective_incar"]
    ):
        diagnostics.append(
            contracts._diagnostic("error", "result.incar", "记录的有效 INCAR 参数与输入配置不一致。")
        )
    if manifest.get("kpoints") != config["kpoints"]:
        diagnostics.append(
            contracts._diagnostic("error", "result.kpoints", "记录的 KPOINTS 规则与输入配置不一致。")
        )
    if manifest.get("sort_structure") is not config["sort_structure"]:
        diagnostics.append(
            contracts._diagnostic("error", "result.sort_structure", "结构排序策略与输入配置不一致。")
        )
    provenance = manifest.get("parameter_provenance")
    if config["preset"] == contracts.MANUSCRIPT_STATIC_PRESET:
        paper = provenance.get("paper") if isinstance(provenance, Mapping) else None
        historical = (
            provenance.get("historical_template") if isinstance(provenance, Mapping) else None
        )
        if (
            not isinstance(paper, Mapping)
            or paper.get("locator") != "manuscript-supplement://SI_0510zdl.docx#page=2&figure=S3"
            or paper.get("declared_parameters") != {"ENCUT": 450, "EDIFF": 5e-6, "IALGO": 38}
            or not isinstance(historical, Mapping)
            or historical.get("incar_locator") != "remote-mlp://4-Element/Li8/scf/single/INCAR"
            or historical.get("kpoints_locator") != "remote-mlp://4-Element/Li8/scf/single/KPOINTS"
        ):
            diagnostics.append(
                contracts._diagnostic(
                    "error",
                    "result.parameter_provenance",
                    "论文/历史模板 provenance 不完整或已变化。",
                )
            )
    elif provenance != {}:
        diagnostics.append(
            contracts._diagnostic(
                "error",
                "result.parameter_provenance",
                "非 preset 输入不得冒充论文参数 provenance。",
            )
        )
    potcar_policy = manifest.get("potcar_policy")
    if (
        not isinstance(potcar_policy, Mapping)
        or potcar_policy.get("source_env") != "PMG_VASP_PSP_DIR"
        or potcar_policy.get("configuration_source") not in {"environment", "pymatgen-settings"}
        or potcar_policy.get("materialized_in_attempt") is not True
        or potcar_policy.get("portable_or_collectable") is not False
    ):
        diagnostics.append(
            contracts._diagnostic("error", "result.potcar_policy", "POTCAR 策略不满足运行时生成且禁止收集。")
        )
    calculations = manifest.get("calculations")
    count = manifest.get("structure_count")
    if not isinstance(calculations, list) or not contracts._positive_int(count) or len(calculations) != count:
        diagnostics.append(
            contracts._diagnostic(
                "error", "result.structure_count", "structure_count 与 calculations 不一致。"
            )
        )
        return diagnostics
    structures_manifest, structure_error = contracts._read_json(input_paths["structures_manifest"])
    raw_structures = (
        structures_manifest.get("structures") if isinstance(structures_manifest, Mapping) else None
    )
    if not isinstance(raw_structures, list) or len(raw_structures) != count:
        diagnostics.append(
            contracts._diagnostic(
                "error", "result.source_structures", str(structure_error or "源结构数量不一致。")
            )
        )
        return diagnostics
    output_subdir = str(parameters.get("output_subdir", "vasp-inputs"))
    expected_reference_id = reference.get("reference_id")
    for index, (calculation, source) in enumerate(zip(calculations, raw_structures), 1):
        prefix = f"calculations[{index - 1}]"
        if not isinstance(calculation, Mapping) or not isinstance(source, Mapping):
            diagnostics.append(contracts._diagnostic("error", prefix, "结构计算记录必须是对象。"))
            continue
        structure_id = source.get("id", source.get("structure_id"))
        source_path = source.get("path", source.get("output_file"))
        if not isinstance(structure_id, str) or not contracts.SAFE_ID.fullmatch(structure_id):
            diagnostics.append(contracts._diagnostic("error", f"{prefix}.structure_id", "源结构 ID 无效。"))
            continue
        actual_source = contracts._structure_input_path(input_paths["structures_manifest"], source_path)
        if actual_source is None:
            diagnostics.append(
                contracts._diagnostic("error", f"{prefix}.source", "源结构必须有非空输入路径。")
            )
            continue
        expected_dir = f"{output_subdir}/{index:06d}-{structure_id}"
        if (
            calculation.get("order") != index
            or calculation.get("structure_id") != structure_id
            or calculation.get("directory") != expected_dir
        ):
            diagnostics.append(
                contracts._diagnostic("error", f"{prefix}.record", "结构顺序、ID 或目录不一致。")
            )
        result_source = calculation.get("source")
        if not isinstance(result_source, Mapping) or result_source.get("path") != source_path:
            diagnostics.append(contracts._diagnostic("error", f"{prefix}.source", "结构来源路径不一致。"))
        if verify_files:
            if not contracts._readable_structure_file(actual_source):
                diagnostics.append(contracts._diagnostic("error", f"{prefix}.source", "源结构缺失或不可读。"))
        files = calculation.get("files")
        if not isinstance(files, Mapping) or set(files) != contracts.EXPECTED_FILE_NAMES:
            diagnostics.append(
                contracts._diagnostic(
                    "error", f"{prefix}.files", "必须且只能声明 POSCAR/INCAR/KPOINTS/POTCAR。"
                )
            )
            continue
        for name in sorted(contracts.EXPECTED_FILE_NAMES):
            record = files.get(name)
            expected_path = f"{expected_dir}/{name}"
            if not isinstance(record, Mapping) or record.get("path") != expected_path:
                diagnostics.append(contracts._diagnostic("error", f"{prefix}.files.{name}", "文件路径无效。"))
                continue
            if record.get("collectable") is not (name != "POTCAR"):
                diagnostics.append(
                    contracts._diagnostic(
                        "error",
                        f"{prefix}.files.{name}.collectable",
                        "POTCAR 必须禁止收集，其余输入必须可收集。",
                    )
                )
            if verify_files:
                actual = attempt / expected_path
                if not contracts._ordinary_project_file(actual, project_root):
                    diagnostics.append(
                        contracts._diagnostic("error", f"{prefix}.files.{name}", f"{name} 缺失。")
                    )
        potcar = calculation.get("potcar")
        expected_symbols = reference.get("symbols")
        if (
            not isinstance(potcar, Mapping)
            or potcar.get("reference_id") != expected_reference_id
            or potcar.get("source_env") != "PMG_VASP_PSP_DIR"
            or potcar.get("functional") != reference.get("functional")
            or potcar.get("portable_artifact") is not False
        ):
            diagnostics.append(
                contracts._diagnostic("error", f"{prefix}.potcar", "POTCAR 来源与批准引用不一致。")
            )
        elif isinstance(expected_symbols, Mapping):
            elements = potcar.get("elements")
            symbols = potcar.get("symbols")
            if (
                not isinstance(elements, list)
                or not isinstance(symbols, list)
                or symbols != [expected_symbols.get(element) for element in elements]
            ):
                diagnostics.append(
                    contracts._diagnostic(
                        "error", f"{prefix}.potcar.symbols", "POTCAR symbols 与显式元素映射不一致。"
                    )
                )
            components = potcar.get("components")
            components_valid = isinstance(symbols, list) and isinstance(components, list)
            if components_valid:
                components_valid = len(components) == len(symbols) and all(
                    isinstance(component, Mapping) and component.get("symbol") == symbol
                    for component, symbol in zip(components, symbols)
                )
            if not components_valid:
                diagnostics.append(
                    contracts._diagnostic(
                        "error",
                        f"{prefix}.potcar.components",
                        "POTCAR component symbols 与引用不一致。",
                    )
                )
        if verify_files and isinstance(files.get("INCAR"), Mapping):
            incar_path = attempt / str(files["INCAR"].get("path", ""))
            if contracts._ordinary_project_file(incar_path, project_root):
                parsed_incar = _parse_incar_file(incar_path)
                if any(
                    not _same_incar_value(parsed_incar.get(key), value)
                    for key, value in config["effective_incar"].items()
                ):
                    diagnostics.append(
                        contracts._diagnostic(
                            "error", f"{prefix}.incar.parameters", "INCAR 内容与批准参数不一致。"
                        )
                    )
        if verify_files and isinstance(files.get("KPOINTS"), Mapping):
            kpoints_path = attempt / str(files["KPOINTS"].get("path", ""))
            if contracts._ordinary_project_file(kpoints_path, project_root) and not _verify_kpoints(
                kpoints_path, config["kpoints"]
            ):
                diagnostics.append(
                    contracts._diagnostic("error", f"{prefix}.kpoints.rule", "KPOINTS 内容与批准规则不一致。")
                )
    return diagnostics
