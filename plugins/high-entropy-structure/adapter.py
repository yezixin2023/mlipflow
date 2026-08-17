"""Safe adapter for user-owned, seeded SQS structure generators.

The adapter never implements an SQS objective and never imports a generator.  It
only validates an explicit command contract, builds an argv list (``shell=False``
is part of the plugin manifest), and verifies declared output artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PLUGIN_ID = "high-entropy-structure"
BUILTIN_SQS = Path(globals().get("__file__", "adapter.py")).absolute().with_name("sqs.py")
SEED_POLICY = "base-seed-plus-candidate-index"
BUILTIN_GENERATOR_IDENTITY = "mlipflow-bundled:icet.generate_sqs_from_supercells"
IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
SHELL_EXECUTABLES = frozenset(
    {"bash", "csh", "cmd", "dash", "fish", "ksh", "powershell", "pwsh", "sh", "tcsh", "zsh"}
)
CONTEXT_KEYS = (
    "project_root",
    "attempt_dir",
    "inputs",
    "parameters",
    "backend",
    "resources",
)


def _diagnostic(level: str, code: str, message: str) -> dict[str, str]:
    return {"level": level, "code": code, "message": message}


def _error_diagnostics(diagnostics: list[dict[str, str]]) -> list[dict[str, str]]:
    return [item for item in diagnostics if item["level"] == "error"]


def _safe_relative(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip() or "\x00" in value or "\n" in value:
        return False
    path = Path(value)
    return path != Path(".") and not path.is_absolute() and ".." not in path.parts


def _path(root: str, relative: str) -> Path:
    return Path(root).expanduser().absolute() / relative


def _context_diagnostics(context: Any) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    if not isinstance(context, dict):
        return [_diagnostic("error", "context.type", "context 必须是对象。")]
    missing = [key for key in CONTEXT_KEYS if key not in context]
    if missing:
        diagnostics.append(
            _diagnostic("error", "context.keys", f"缺少上下文字段：{', '.join(missing)}。")
        )
    for key in ("project_root", "attempt_dir"):
        if key in context and (not isinstance(context[key], str) or not context[key].strip()):
            diagnostics.append(_diagnostic("error", f"context.{key}", f"{key} 必须是路径字符串。"))
    for key in ("inputs", "parameters", "resources"):
        if key in context and not isinstance(context[key], dict):
            diagnostics.append(_diagnostic("error", f"context.{key}", f"{key} 必须是对象。"))
    if "backend" in context and context["backend"] != "local":
        diagnostics.append(
            _diagnostic(
                "error",
                "backend.unsupported",
                "当前适配器仅验证过 local；SLURM 完成后尚未接入插件科学检查。",
            )
        )
    return diagnostics


def _load_json(path: Path) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    if not path.is_file():
        return None, _diagnostic("warning", "artifact.missing", f"结果清单尚不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, _diagnostic("error", "artifact.invalid_json", f"无法读取 JSON：{exc}")
    if not isinstance(value, dict):
        return None, _diagnostic("error", "artifact.not_object", "结果清单必须是 JSON 对象。")
    return value, None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _generator_contract(context: dict[str, Any]) -> tuple[str, Path]:
    script_value = context["parameters"].get("sqs_script")
    if isinstance(script_value, str):
        return f"user-supplied:{script_value}", _path(context["project_root"], script_value)
    return BUILTIN_GENERATOR_IDENTITY, BUILTIN_SQS.absolute()


def _positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _read_structure_counts(path: Path) -> Counter[str]:
    """Read actual atom counts, using ASE with a strict POSCAR fallback.

    ASE is a declared execution dependency and is the reliable general parser.  The
    fallback keeps checker-only environments useful for VASP 5 files without
    pretending to support arbitrary structure formats.
    """

    try:
        from ase.io import read
    except ImportError:
        read = None
    if read is not None:
        try:
            atoms = read(str(path), index=-1)
        except Exception as exc:
            raise ValueError(f"ASE 无法解析结构 {path}: {exc}") from exc
        symbols = atoms.get_chemical_symbols()
        if not symbols:
            raise ValueError(f"结构不含原子：{path}")
        return Counter(symbols)

    if path.suffix.lower() not in {".vasp", ".poscar", ".contcar"} and path.name.upper() not in {
        "POSCAR",
        "CONTCAR",
    }:
        raise ValueError("缺少 ASE 时只能可靠验证 VASP 5 POSCAR/CONTCAR")
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 7:
        raise ValueError("缺少 ASE，且结构不是可严格解析的 VASP 5 POSCAR")
    try:
        scale = float(lines[1].split()[0])
        lattice = [[float(item) for item in lines[index].split()] for index in range(2, 5)]
    except (IndexError, ValueError) as exc:
        raise ValueError("缺少 ASE，且 VASP 5 scale/lattice 无效") from exc
    if scale == 0 or any(len(vector) != 3 for vector in lattice):
        raise ValueError("缺少 ASE，且 VASP 5 scale/lattice 无效")
    species = lines[5].split()
    raw_counts = lines[6].split()
    if not species or len(species) != len(raw_counts) or len(species) != len(set(species)):
        raise ValueError("缺少 ASE，且 VASP 5 species/count 行无效")
    try:
        counts = [int(item) for item in raw_counts]
    except ValueError as exc:
        raise ValueError("缺少 ASE，且 VASP 5 count 行不是整数") from exc
    if any(count < 0 for count in counts) or sum(counts) < 1:
        raise ValueError("VASP 5 count 必须为非负整数且总数大于零")
    coordinate_mode_index = 7
    if lines[coordinate_mode_index].strip().lower().startswith("s"):
        coordinate_mode_index += 1
    if (
        len(lines) <= coordinate_mode_index
        or not lines[coordinate_mode_index].strip().lower().startswith(("d", "c", "k"))
    ):
        raise ValueError("缺少 ASE，且 VASP 5 coordinate mode 无效")
    coordinate_start = coordinate_mode_index + 1
    if len(lines) < coordinate_start + sum(counts):
        raise ValueError("VASP 5 坐标行少于 species count 声明")
    try:
        coordinates = [
            [float(item) for item in line.split()[:3]]
            for line in lines[coordinate_start : coordinate_start + sum(counts)]
        ]
    except ValueError as exc:
        raise ValueError("缺少 ASE，且 VASP 5 坐标不是数值") from exc
    if any(len(coordinate) != 3 for coordinate in coordinates):
        raise ValueError("缺少 ASE，且 VASP 5 坐标列不足")
    return Counter(dict(zip(species, counts)))


def _load_composition_contract(
    path: Path, prototype_counts: Counter[str]
) -> dict[str, Any]:
    """Independently normalize the approved generator input contract."""

    value, diagnostic = _load_json(path)
    if value is None:
        assert diagnostic is not None
        raise ValueError(diagnostic["message"])
    if value.get("schema_version") != 1:
        raise ValueError("composition manifest schema_version 必须为 1")
    sublattice = value.get("alloy_sublattice")
    if not isinstance(sublattice, dict):
        raise ValueError("alloy_sublattice 必须是对象")
    prototype_species = sublattice.get("prototype_species")
    allowed_species = sublattice.get("allowed_species")
    if not isinstance(prototype_species, str) or not prototype_species:
        raise ValueError("alloy_sublattice.prototype_species 必须是非空字符串")
    if not (
        isinstance(allowed_species, list)
        and len(allowed_species) >= 2
        and all(isinstance(item, str) and item for item in allowed_species)
        and len(set(allowed_species)) == len(allowed_species)
        and prototype_species in allowed_species
    ):
        raise ValueError("allowed_species 必须唯一且包含 prototype_species")
    cutoffs = value.get("cluster_cutoffs_angstrom")
    if not (
        isinstance(cutoffs, list)
        and cutoffs
        and all(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
            and float(item) > 0
            for item in cutoffs
        )
    ):
        raise ValueError("cluster_cutoffs_angstrom 必须包含正 finite 数值")
    repeat = value.get("supercell_repeat")
    if not (isinstance(repeat, list) and len(repeat) == 3 and all(map(_positive_int, repeat))):
        raise ValueError("supercell_repeat 必须包含三个正整数")
    n_steps = value.get("n_steps")
    if not _positive_int(n_steps):
        raise ValueError("n_steps 必须是正整数")
    if value.get("output_format", "vasp") not in {"vasp", "cif"}:
        raise ValueError("output_format 必须为 vasp 或 cif")
    primitive_alloy_sites = prototype_counts.get(prototype_species, 0)
    if primitive_alloy_sites < 1:
        raise ValueError(f"prototype 不含 alloy sublattice species {prototype_species!r}")
    repeat_factor = math.prod(repeat)
    alloy_site_count = primitive_alloy_sites * repeat_factor
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates 必须是非空数组")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    allowed_set = set(allowed_species)
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"candidate {index} 必须是对象")
        candidate_id = candidate.get("id")
        if not isinstance(candidate_id, str) or not IDENTIFIER.fullmatch(candidate_id):
            raise ValueError(f"candidate {index} id 无效")
        if candidate_id in seen:
            raise ValueError(f"candidate id 重复：{candidate_id}")
        seen.add(candidate_id)
        counts = candidate.get("counts")
        if not isinstance(counts, dict) or set(counts) != allowed_set:
            raise ValueError(f"candidate {candidate_id} species 必须与 allowed_species 完全一致")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in counts.values()
        ):
            raise ValueError(f"candidate {candidate_id} count 必须是非负整数")
        if sum(counts.values()) != alloy_site_count:
            raise ValueError(
                f"candidate {candidate_id} count 总和与 alloy site count {alloy_site_count} 不一致"
            )
        expected_structure = Counter(
            {
                species: count * repeat_factor
                for species, count in prototype_counts.items()
                if species != prototype_species
            }
        )
        expected_structure.update(counts)
        normalized.append(
            {
                "id": candidate_id,
                "counts": dict(counts),
                "expected_structure_counts": {
                    species: count for species, count in expected_structure.items() if count > 0
                },
            }
        )
    return {
        "candidates": normalized,
        "alloy_site_count": alloy_site_count,
        "allowed_species": list(allowed_species),
    }


class Adapter:
    """Plan a user generator and verify its immutable structure manifest."""

    def validate(self, context: Any) -> list[dict[str, str]]:
        diagnostics = _context_diagnostics(context)
        if not isinstance(context, dict):
            return diagnostics
        inputs = context.get("inputs")
        parameters = context.get("parameters")
        if not isinstance(inputs, dict) or not isinstance(parameters, dict):
            return diagnostics

        for key in ("prototype_structure", "composition_manifest"):
            if not _safe_relative(inputs.get(key)):
                diagnostics.append(
                    _diagnostic(
                        "error",
                        f"inputs.{key}",
                        f"{key} 必须是 project_root 下不含 '..' 的相对路径。",
                    )
                )
        if parameters.get("sqs_script") is not None and not _safe_relative(
            parameters.get("sqs_script")
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "parameters.sqs_script",
                    "sqs_script 若提供，必须是 project_root 下显式的相对脚本路径。",
                )
            )
        interpreter = parameters.get("interpreter_argv", [sys.executable])
        if not (
            isinstance(interpreter, list)
            and len(interpreter) == 1
            and all(
                isinstance(item, str) and item and "\x00" not in item and "\n" not in item
                for item in interpreter
            )
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "parameters.interpreter_argv",
                    "interpreter_argv 必须只含一个显式可执行文件，例如 ['python']。",
                )
            )
        elif Path(interpreter[0]).name.lower() in SHELL_EXECUTABLES:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "parameters.shell_forbidden",
                    "interpreter_argv 不得选择 shell 或命令字符串模式。",
                )
            )
        if "extra_args" in parameters:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "parameters.extra_args",
                    "不接受 extra_args；固定输出、seed 和候选上限不得被重复参数覆盖。",
                )
            )
        seed = parameters.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            diagnostics.append(
                _diagnostic("error", "parameters.seed", "seed 必须是显式的非负整数。")
            )
        max_candidates = parameters.get("max_candidates")
        if (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or max_candidates < 1
        ):
            diagnostics.append(
                _diagnostic(
                    "error", "parameters.max_candidates", "max_candidates 必须是正整数硬上限。"
                )
            )
        for key, default in (
            ("structures_dir", "structures"),
            ("result_manifest", "generation-result.json"),
        ):
            if not _safe_relative(parameters.get(key, default)):
                diagnostics.append(
                    _diagnostic(
                        "error", f"parameters.{key}", f"{key} 必须是 attempt_dir 下的安全相对路径。"
                    )
                )
        return diagnostics

    def plan(self, context: Any) -> dict[str, Any]:
        diagnostics = self.validate(context)
        if _error_diagnostics(diagnostics):
            return {
                "plugin_id": PLUGIN_ID,
                "status": "BLOCKED",
                "executable": False,
                "diagnostics": diagnostics,
            }
        assert isinstance(context, dict)
        inputs = context["inputs"]
        parameters = context["parameters"]
        project_root = context["project_root"]
        attempt_dir = context["attempt_dir"]
        output_dir = _path(attempt_dir, parameters.get("structures_dir", "structures"))
        result_path = _path(
            attempt_dir, parameters.get("result_manifest", "generation-result.json")
        )
        script_value = parameters.get("sqs_script")
        generator_identity, script_path = _generator_contract(context)
        argv = list(parameters.get("interpreter_argv", [sys.executable]))
        argv.extend(
            [
                str(script_path),
                "--prototype",
                str(_path(project_root, inputs["prototype_structure"])),
                "--composition-manifest",
                str(_path(project_root, inputs["composition_manifest"])),
                "--seed",
                str(parameters["seed"]),
                "--max-candidates",
                str(parameters["max_candidates"]),
                "--output-dir",
                str(output_dir),
                "--result-manifest",
                str(result_path),
            ]
        )
        return {
            "plugin_id": PLUGIN_ID,
            "status": "READY",
            "executable": True,
            "argv": argv,
            "cwd": str(Path(attempt_dir).expanduser().absolute()),
            "expected_outputs": [str(result_path), str(output_dir)],
            "diagnostics": diagnostics,
            "provenance": {
                "seed": parameters["seed"],
                "max_candidates": parameters["max_candidates"],
                "generator_is_user_supplied": script_value is not None,
                "generator_identity": generator_identity,
                "generator_path": str(script_path),
                "generator_fingerprint": _sha256(script_path) if script_path.is_file() else None,
                "seed_policy": SEED_POLICY,
            },
        }

    def prepare(self, context: Any, plan: Any) -> dict[str, Any]:
        """Return the already-materialized plan; never create input files here."""

        if not isinstance(plan, dict) or plan.get("status") != "READY":
            return {
                "plugin_id": PLUGIN_ID,
                "status": "BLOCKED",
                "executable": False,
                "diagnostics": [
                    _diagnostic("error", "plan.not_ready", "prepare 需要 READY 计划。")
                ],
            }
        return dict(plan)

    def _result_path(self, context: dict[str, Any]) -> Path:
        parameters = context["parameters"]
        return _path(
            context["attempt_dir"], parameters.get("result_manifest", "generation-result.json")
        )

    def _verify_result(
        self, context: dict[str, Any], manifest: dict[str, Any]
    ) -> list[dict[str, str]]:
        diagnostics: list[dict[str, str]] = []
        parameters = context["parameters"]
        if manifest.get("schema_version") != 1:
            diagnostics.append(
                _diagnostic("error", "result.schema_version", "schema_version 必须为 1。")
            )
        if manifest.get("plugin_id") != PLUGIN_ID:
            diagnostics.append(_diagnostic("error", "result.plugin_id", "结果 plugin_id 不匹配。"))
        if manifest.get("status") != "OK":
            diagnostics.append(_diagnostic("error", "result.status", "生成器未声明 status=OK。"))
        if manifest.get("seed") != parameters.get("seed"):
            diagnostics.append(
                _diagnostic("error", "result.seed", "结果 seed 与已批准计划不一致。")
            )

        generator_identity, generator_path = _generator_contract(context)
        generator = manifest.get("generator")
        expected_generator_fingerprint: str | None = None
        if generator_path.is_file():
            expected_generator_fingerprint = _sha256(generator_path)
        else:
            diagnostics.append(
                _diagnostic(
                    "error", "result.generator_source", f"已批准 generator 不存在：{generator_path}"
                )
            )
        if not isinstance(generator, dict):
            diagnostics.append(
                _diagnostic("error", "result.generator", "必须声明 generator identity 和 fingerprint。")
            )
        else:
            if generator.get("identity") != generator_identity:
                diagnostics.append(
                    _diagnostic(
                        "error", "result.generator_identity", "generator identity 与已批准计划不一致。"
                    )
                )
            if generator.get("fingerprint") != expected_generator_fingerprint:
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "result.generator_fingerprint",
                        "generator fingerprint 与当前已批准脚本不一致。",
                    )
                )

        method = manifest.get("method")
        if not isinstance(method, dict):
            diagnostics.append(_diagnostic("error", "result.method", "必须声明 method 对象。"))
        else:
            if method.get("random_seed_policy") != SEED_POLICY:
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "result.seed_policy",
                        f"random_seed_policy 必须为 {SEED_POLICY}。",
                    )
                )
            if generator_identity == BUILTIN_GENERATOR_IDENTITY:
                bundled_identity = {
                    "library": "icet",
                    "api": "generate_sqs_from_supercells",
                }
                for key, expected in bundled_identity.items():
                    if method.get(key) != expected:
                        diagnostics.append(
                            _diagnostic(
                                "error", f"result.method.{key}", f"bundled method {key} 不匹配。"
                            )
                        )
                for key in ("library_version", "ase_version"):
                    if not isinstance(method.get(key), str) or not method[key]:
                        diagnostics.append(
                            _diagnostic(
                                "error",
                                f"result.method.{key}",
                                f"bundled method 必须记录 {key} identity；版本本身不代表结构科学等价。",
                            )
                        )

        declared_inputs = {
            "prototype_fingerprint": manifest.get("prototype_fingerprint"),
            "composition_manifest_fingerprint": manifest.get("composition_manifest_fingerprint"),
        }
        for key, fingerprint in declared_inputs.items():
            if not isinstance(fingerprint, str) or not SHA256.fullmatch(fingerprint):
                diagnostics.append(
                    _diagnostic("error", f"result.{key}", f"必须声明输入 {key} 的 sha256。")
                )
        input_paths = {
            "prototype_fingerprint": _path(
                context["project_root"], context["inputs"]["prototype_structure"]
            ),
            "composition_manifest_fingerprint": _path(
                context["project_root"], context["inputs"]["composition_manifest"]
            ),
        }
        for key, input_path in input_paths.items():
            if not input_path.is_file():
                diagnostics.append(
                    _diagnostic("error", f"result.{key}", f"输入文件不存在：{input_path}")
                )
            elif _sha256(input_path) != declared_inputs[key]:
                diagnostics.append(
                    _diagnostic("error", f"result.{key}", f"输入 {key} 的 sha256 不匹配。")
                )

        approved: dict[str, Any] | None = None
        prototype_path = input_paths["prototype_fingerprint"]
        composition_path = input_paths["composition_manifest_fingerprint"]
        if prototype_path.is_file() and composition_path.is_file():
            try:
                prototype_counts = _read_structure_counts(prototype_path)
                approved = _load_composition_contract(composition_path, prototype_counts)
            except (OSError, UnicodeError, ValueError) as exc:
                diagnostics.append(
                    _diagnostic(
                        "error", "result.approved_input_contract", f"无法重建批准输入 contract：{exc}"
                    )
                )
        if approved is not None and len(approved["candidates"]) > parameters["max_candidates"]:
            diagnostics.append(
                _diagnostic(
                    "error", "result.approved_count_limit", "批准 manifest 超过 max_candidates 硬上限。"
                )
            )

        structures = manifest.get("structures")
        if not isinstance(structures, list) or not structures:
            diagnostics.append(
                _diagnostic("error", "result.structures", "structures 必须是非空数组。")
            )
            return diagnostics
        if len(structures) > parameters.get("max_candidates", 0):
            diagnostics.append(
                _diagnostic("error", "result.count_limit", "结构数量超过 max_candidates 硬上限。")
            )
        if manifest.get("candidate_count") != len(structures):
            diagnostics.append(
                _diagnostic(
                    "error", "result.candidate_count", "candidate_count 必须等于 structures 数量。"
                )
            )
        approved_candidates: list[dict[str, Any]] = (
            approved["candidates"] if approved is not None else []
        )
        if approved is not None and len(structures) != len(approved_candidates):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "result.candidate_coverage",
                    "输出 candidate 数量必须与批准且应执行的 candidate 完全一致。",
                )
            )
        seen: set[str] = set()
        for index, structure in enumerate(structures):
            prefix = f"structures[{index}]"
            if not isinstance(structure, dict):
                diagnostics.append(_diagnostic("error", f"{prefix}.type", "结构记录必须是对象。"))
                continue
            structure_id = structure.get("id")
            if not isinstance(structure_id, str) or not structure_id:
                diagnostics.append(_diagnostic("error", f"{prefix}.id", "结构 id 不能为空。"))
            elif structure_id in seen:
                diagnostics.append(_diagnostic("error", f"{prefix}.id", "结构 id 必须唯一。"))
            else:
                seen.add(structure_id)
            expected = approved_candidates[index] if index < len(approved_candidates) else None
            if expected is None or structure_id != expected["id"]:
                diagnostics.append(
                    _diagnostic(
                        "error",
                        f"{prefix}.approved_id",
                        "candidate id/顺序必须与批准 composition manifest 完全一致。",
                    )
                )
            expected_seed = parameters["seed"] + index
            if structure.get("random_seed") != expected_seed:
                diagnostics.append(
                    _diagnostic(
                        "error",
                        f"{prefix}.random_seed",
                        f"random_seed 必须等于 base seed + candidate index ({expected_seed})。",
                    )
                )
            relative = structure.get("path")
            fingerprint = structure.get("fingerprint")
            if not _safe_relative(relative):
                diagnostics.append(
                    _diagnostic("error", f"{prefix}.path", "结构路径必须位于 attempt_dir 内。")
                )
                continue
            if not isinstance(fingerprint, str) or not SHA256.fullmatch(fingerprint):
                diagnostics.append(
                    _diagnostic("error", f"{prefix}.fingerprint", "必须声明 sha256 指纹。")
                )
                continue
            declared_composition = structure.get("composition")
            if not isinstance(declared_composition, dict):
                diagnostics.append(
                    _diagnostic("error", f"{prefix}.composition", "必须声明 composition 对象。")
                )
            elif expected is not None and declared_composition != expected["counts"]:
                diagnostics.append(
                    _diagnostic(
                        "error",
                        f"{prefix}.composition",
                        "结果声明的 species/count 必须与批准 composition manifest 完全一致。",
                    )
                )
            cluster_vector = structure.get("cluster_vector")
            if "cluster_vector" in structure and not (
                isinstance(cluster_vector, list)
                and cluster_vector
                and all(
                    not isinstance(item, bool)
                    and isinstance(item, (int, float))
                    and math.isfinite(float(item))
                    for item in cluster_vector
                )
            ):
                diagnostics.append(
                    _diagnostic(
                        "error",
                        f"{prefix}.cluster_vector",
                        "cluster_vector 若声明，必须是非空 finite numeric array。",
                    )
                )
            artifact_path = _path(context["attempt_dir"], relative)
            if not artifact_path.is_file():
                diagnostics.append(
                    _diagnostic("error", f"{prefix}.missing", f"结构文件不存在：{artifact_path}")
                )
            elif artifact_path.stat().st_size == 0:
                diagnostics.append(_diagnostic("error", f"{prefix}.empty", "结构文件为空。"))
            elif _sha256(artifact_path) != fingerprint:
                diagnostics.append(
                    _diagnostic("error", f"{prefix}.fingerprint", "结构文件 sha256 不匹配。")
                )
            elif expected is not None:
                try:
                    actual_counts = dict(_read_structure_counts(artifact_path))
                except (OSError, UnicodeError, ValueError) as exc:
                    diagnostics.append(
                        _diagnostic(
                            "error",
                            f"{prefix}.structure_composition",
                            f"无法可靠验证结构实际 composition：{exc}",
                        )
                    )
                else:
                    if actual_counts != expected["expected_structure_counts"]:
                        diagnostics.append(
                            _diagnostic(
                                "error",
                                f"{prefix}.structure_composition",
                                "结构实际 species/count 与 prototype + 批准 alloy counts 不一致。",
                            )
                        )
        return diagnostics

    def check(self, context: Any) -> dict[str, Any]:
        diagnostics = self.validate(context)
        if _error_diagnostics(diagnostics):
            return {"plugin_id": PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
        assert isinstance(context, dict)
        path = self._result_path(context)
        manifest, read_diagnostic = _load_json(path)
        if manifest is None:
            assert read_diagnostic is not None
            status = "WAIT" if read_diagnostic["level"] == "warning" else "FAIL"
            return {"plugin_id": PLUGIN_ID, "status": status, "diagnostics": [read_diagnostic]}
        diagnostics.extend(self._verify_result(context, manifest))
        return {
            "plugin_id": PLUGIN_ID,
            "status": "FAIL" if _error_diagnostics(diagnostics) else "OK",
            "diagnostics": diagnostics,
            "result_manifest": str(path),
        }

    def collect(self, context: Any) -> dict[str, Any]:
        checked = self.check(context)
        if checked["status"] != "OK":
            return checked
        assert isinstance(context, dict)
        result_path = self._result_path(context)
        manifest, _ = _load_json(result_path)
        assert manifest is not None
        artifacts = [
            {
                "name": "generation-manifest",
                "path": context["parameters"].get("result_manifest", "generation-result.json"),
                "media_type": "application/json",
            }
        ]
        for structure in manifest["structures"]:
            artifacts.append(
                {
                    "name": structure["id"],
                    "path": structure["path"],
                    "media_type": structure.get("media_type", "chemical/x-cif"),
                    "fingerprint": structure["fingerprint"],
                }
            )
        return {
            "plugin_id": PLUGIN_ID,
            "status": "OK",
            "artifacts": artifacts,
            "metrics": {"structure_count": len(manifest["structures"]), "seed": manifest["seed"]},
        }

    def replay(self, context: Any) -> dict[str, Any]:
        """Verify existing artifacts only; never invoke the user generator."""

        if not isinstance(context, dict):
            return {
                "plugin_id": PLUGIN_ID,
                "status": "FAIL",
                "executable": False,
                "diagnostics": [_diagnostic("error", "context.type", "context 必须是对象。")],
            }
        inline = context.get("result_manifest")
        if isinstance(inline, dict):
            diagnostics = self.validate(context)
            if _error_diagnostics(diagnostics):
                return {
                    "plugin_id": PLUGIN_ID,
                    "status": "FAIL",
                    "executable": False,
                    "diagnostics": diagnostics,
                    "result_manifest": inline,
                }
            diagnostics.extend(self._verify_result(context, inline))
            return {
                "plugin_id": PLUGIN_ID,
                "status": "FAIL" if _error_diagnostics(diagnostics) else "OK",
                "executable": False,
                "diagnostics": diagnostics,
                "result_manifest": inline,
            }
        collected = self.collect(context)
        collected["executable"] = False
        return collected
