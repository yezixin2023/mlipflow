"""Contract tests for LASP/SSW execution and historical-output normalization."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "plugins" / "pes-sampling"
ADAPTER_PATH = PLUGIN_ROOT / "adapter.py"
WRAPPER_PATH = PLUGIN_ROOT / "lasp_ssw.py"
PLUGIN_MANIFEST_PATH = PLUGIN_ROOT / "plugin.yaml"
UNKNOWN_SEED = "HISTORICAL_PARAMETER_UNKNOWN"
ARC_HEADER = b"!BIOSYM archive 2\nPBC=ON\n"
PYTHON_EXECUTABLE = str(Path(sys.executable).resolve())


def load_adapter() -> Any:
    spec = importlib.util.spec_from_file_location("test_lasp_ssw_adapter", ADAPTER_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {ADAPTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Adapter()


def diagnostic_codes(result: Any) -> set[str]:
    diagnostics = result if isinstance(result, list) else result.get("diagnostics", [])
    return {str(item["code"]) for item in diagnostics}


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def file_snapshot(root: Path) -> dict[str, tuple[int, bytes]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_mtime_ns, path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    }


def arc_payload(energies: list[float]) -> bytes:
    chunks = ["!BIOSYM archive 2\n", "PBC=ON\n"]
    for index, energy in enumerate(energies, 1):
        chunks.extend(
            [
                f"Energy {index} {energy:.8f}\n",
                "!DATE fixture\n",
                "PBC 5.0 5.0 5.0 90.0 90.0 90.0\n",
                f"Li {index / 10:.4f} 0.0 0.0 CORE 1 Li Li 0.0 1\n",
                "S 1.0 1.0 1.0 CORE 2 S S 0.0 2\n",
                "end\n",
                "end\n",
            ]
        )
    return "".join(chunks).encode("utf-8")


def read_structures(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    structures = value.get("structures")
    if not isinstance(structures, list) or not all(isinstance(item, dict) for item in structures):
        raise AssertionError(f"{path} must contain a structures list")
    return structures


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def refresh_result_artifact(output: Path, artifact_path: Path) -> None:
    result_path = output / "sampling-result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    relative = artifact_path.relative_to(output).as_posix()
    for artifact in result["artifacts"]:
        if artifact["path"] == relative:
            payload = artifact_path.read_bytes()
            artifact["sha256"] = sha256_bytes(payload)
            artifact["size_bytes"] = len(payload)
            break
    else:
        raise AssertionError(f"artifact not listed in sampling-result.json: {relative}")
    write_json(result_path, result)


class LaspFixtureMixin:
    root: Path
    historical: Path
    lasp_input: Path

    def write_historical_fixture(self) -> None:
        self.historical = self.root / "historical-lasp-run"
        self.historical.mkdir()
        (self.historical / "allstr.arc").write_bytes(
            arc_payload([-10.0, -9.0, 1.0, -8.0, -7.0])
        )
        (self.historical / "best.arc").write_bytes(arc_payload([-10.0]))
        (self.historical / "md.arc").write_bytes(arc_payload([-9.0, -8.0]))
        self.lasp_input = self.root / "lasp.in"
        self.lasp_input.write_text(
            "potential vasp\nexplore_type ssw\nSSW.SSWsteps 5\nSSW.Temp 300\n",
            encoding="utf-8",
        )

    def shared_parameters(self, operation: str, output_subdir: str) -> dict[str, Any]:
        return {
            "operation": operation,
            "output_subdir": output_subdir,
            "historical_source_id": "fixture://lasp/ssw-five-frames-v1",
            "selection_stride": 3,
            "energy_max_ev": 0.0,
            "max_frames": 5,
            "include_best_arc": True,
            "include_md_arc": True,
            "seed_status": UNKNOWN_SEED,
            "acknowledge_uncontrolled_seed": True,
            "preserve_historical_order": True,
        }


class LaspWrapperTests(LaspFixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.write_historical_fixture()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def normalize(
        self,
        output: Path,
        *,
        historical: Path | None = None,
        max_frames: int = 5,
        source_id: str = "fixture://lasp/ssw-five-frames-v1",
    ) -> None:
        historical = historical or self.historical
        completed = subprocess.run(
            [
                PYTHON_EXECUTABLE,
                str(WRAPPER_PATH),
                "normalize-replay",
                "--historical-run-dir",
                str(historical),
                "--lasp-input",
                str(self.lasp_input),
                "--output-dir",
                str(output),
                "--historical-source-id",
                source_id,
                "--selection-stride",
                "3",
                "--energy-max-ev",
                "0",
                "--max-frames",
                str(max_frames),
                "--seed-status",
                UNKNOWN_SEED,
                "--include-best-arc",
                "--include-md-arc",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)

    def test_four_source_frames_select_only_first_after_accepted_order_renumbering(self) -> None:
        historical = self.root / "historical-four-frames"
        historical.mkdir()
        (historical / "allstr.arc").write_bytes(arc_payload([-10.0, -9.0, 1.0, -8.0]))
        (historical / "best.arc").write_bytes(arc_payload([-10.0]))
        (historical / "md.arc").write_bytes(arc_payload([-9.0]))
        output = self.root / "normalized-four-frames"
        self.normalize(
            output,
            historical=historical,
            max_frames=4,
            source_id="fixture://lasp/ssw-four-frames-v1",
        )

        selected = read_structures(output / "selected-structures.json")
        self.assertEqual([1], [item["frame_index"] for item in selected])
        self.assertEqual([1], [item["selected_order"] for item in selected])

    def test_normalize_historical_filter_then_stride_ids_hashes_and_order_are_stable(self) -> None:
        historical_before = file_snapshot(self.historical)
        first = self.root / "normalized-first"
        self.normalize(first)
        historical_after_first = file_snapshot(self.historical)
        second = self.root / "normalized-second"
        self.normalize(second)

        # Normalization reads but never mutates the historical source tree.
        self.assertEqual(historical_before, historical_after_first)
        self.assertEqual(historical_after_first, file_snapshot(self.historical))

        all_first = read_structures(first / "ssw-structures.json")
        all_second = read_structures(second / "ssw-structures.json")
        selected_first = read_structures(first / "selected-structures.json")
        selected_second = read_structures(second / "selected-structures.json")

        self.assertEqual([1, 2, 3, 4, 5], [item["frame_index"] for item in all_first])
        self.assertEqual([1, 2, 3, 4, 5], [item["historical_order"] for item in all_first])
        self.assertEqual(
            [-10.0, -9.0, 1.0, -8.0, -7.0],
            [item["energy_ev"] for item in all_first],
        )
        # Historical single.py discards energy > 0, renumbers accepted frames,
        # then selects accepted-order 1, 4, 7, ... .  Source frames 1 and 5
        # therefore survive this five-frame fixture.
        self.assertEqual([1, 5], [item["frame_index"] for item in selected_first])
        self.assertEqual([1, 2], [item["selected_order"] for item in selected_first])

        def identity(item: dict[str, Any]) -> tuple[Any, ...]:
            return (
                item["structure_id"],
                item["frame_index"],
                item["historical_order"],
                item["frame_sha256"],
            )
        self.assertEqual(list(map(identity, all_first)), list(map(identity, all_second)))
        self.assertEqual(list(map(identity, selected_first)), list(map(identity, selected_second)))
        self.assertEqual(len({item["structure_id"] for item in all_first}), 5)

        for output, records in ((first, selected_first), (second, selected_second)):
            for record in records:
                artifact = output / record["output_file"]
                self.assertTrue(artifact.is_file())
                self.assertEqual(record["frame_sha256"], sha256_bytes(artifact.read_bytes()))

        result = json.loads((first / "sampling-result.json").read_text(encoding="utf-8"))
        self.assertEqual("OK", result["status"])
        self.assertEqual("lasp-ssw-normalize-replay", result["operation"])
        self.assertEqual("historical-replay", result["mode"])
        self.assertEqual(
            {
                "generated_structure_count": 5,
                "energy_accepted_count": 4,
                "selected_structure_count": 2,
                "aimd_seed_candidate_count": 1,
                "md_structure_count": 2,
            },
            result["counts"],
        )
        for required in (
            "sampling-result.json",
            "ssw-structures.json",
            "selected-structures.json",
            "lasp-run-metadata.json",
            "aimd-seeds.json",
            "md-structures.json",
        ):
            self.assertTrue((first / required).is_file(), required)


class LaspAdapterTests(LaspFixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.write_historical_fixture()
        self.adapter = load_adapter()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def normalize_context(self, attempt_name: str = "attempt-1") -> dict[str, Any]:
        attempt = self.root / ".mlipflow" / "runs" / "lasp-normalize" / attempt_name
        return {
            "project_root": str(self.root),
            "attempt_dir": str(attempt),
            "inputs": {
                "historical_run_dir": str(self.historical),
                "lasp_input": str(self.lasp_input),
            },
            "parameters": self.shared_parameters(
                "lasp-ssw-normalize-replay", "lasp-normalized"
            ),
            "backend": "local",
            "resources": {"python_executable": PYTHON_EXECUTABLE},
        }

    def run_adapter_plan(self, context: dict[str, Any]) -> tuple[dict[str, Any], Path]:
        plan = self.adapter.plan(context)
        self.assertEqual("READY", plan["status"], plan.get("diagnostics"))
        attempt = Path(context["attempt_dir"])
        attempt.mkdir(parents=True)
        completed = subprocess.run(
            plan["argv"],
            cwd=plan["cwd"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        context["execution"] = {"returncode": completed.returncode, "plan": plan}
        return plan, Path(plan["output_dir"])

    def test_normalize_plan_and_prepare_are_zero_write_shell_false(self) -> None:
        context = self.normalize_context()
        before = file_snapshot(self.root)
        diagnostics = self.adapter.validate(context)
        plan = self.adapter.plan(context)
        prepared = self.adapter.prepare(context, plan)
        after = file_snapshot(self.root)

        self.assertEqual(before, after)
        self.assertFalse(Path(context["attempt_dir"]).exists())
        self.assertFalse(any(item["level"] == "ERROR" for item in diagnostics), diagnostics)
        self.assertEqual("READY", plan["status"])
        self.assertEqual("lasp-ssw-normalize-replay", plan["operation"])
        self.assertTrue(plan["executable"])
        self.assertFalse(plan["shell"])
        self.assertIsInstance(plan["argv"], list)
        self.assertTrue(all(isinstance(item, str) for item in plan["argv"]))
        self.assertIn(str(WRAPPER_PATH), plan["argv"])
        self.assertNotIn("--lasp-executable", plan["argv"])
        self.assertIn("sampling-result.json", Path(plan["expected_outputs"][0]).name)
        python_path = Path(PYTHON_EXECUTABLE)
        self.assertEqual(PYTHON_EXECUTABLE, plan["argv"][0])
        self.assertEqual(
            {
                "sha256": sha256_bytes(python_path.read_bytes()),
                "size_bytes": python_path.stat().st_size,
            },
            plan["input_fingerprints"]["python_executable"],
        )
        self.assertFalse(prepared["writes_files"])
        self.assertFalse(prepared["prepared"])

    def test_python_executable_must_be_explicit_absolute_nonsymlink_executable_file(
        self,
    ) -> None:
        nonexecutable = self.root / "python-not-executable"
        nonexecutable.write_text("not executable\n", encoding="utf-8")
        nonexecutable.chmod(0o644)
        directory = self.root / "python-directory"
        directory.mkdir()
        real_bin = self.root / "real-bin"
        real_bin.mkdir()
        executable = real_bin / "python"
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
        linked_bin = self.root / "linked-bin"
        linked_bin.symlink_to(real_bin, target_is_directory=True)

        invalid_values: tuple[tuple[str, str | None], ...] = (
            ("missing", None),
            ("relative", "python3"),
            ("nonexecutable", str(nonexecutable)),
            ("directory", str(directory)),
            ("symlink-component", str(linked_bin / executable.name)),
        )
        for operation in ("lasp-ssw-normalize-replay", "lasp-ssw-execute"):
            for case, value in invalid_values:
                with self.subTest(operation=operation, case=case):
                    context = self.normalize_context(f"attempt-{operation}-{case}")
                    if operation == "lasp-ssw-execute":
                        context["parameters"]["operation"] = operation
                        context["parameters"]["lasp_version"] = "fixture-1.0"
                        context["inputs"] = {
                            "lasp_executable": PYTHON_EXECUTABLE,
                            "input_structure": str(self.historical / "allstr.arc"),
                            "lasp_input": str(self.lasp_input),
                        }
                    if value is None:
                        context["resources"].pop("python_executable")
                    else:
                        context["resources"]["python_executable"] = value

                    blocked = self.adapter.plan(context)

                    self.assertEqual("BLOCKED", blocked["status"])
                    self.assertIn("resource.python_executable", diagnostic_codes(blocked))

    def test_plan_blocks_output_escape_and_never_overwrites_existing_output(self) -> None:
        escaped = self.normalize_context()
        escaped["parameters"]["output_subdir"] = "../escape"
        blocked = self.adapter.plan(escaped)
        self.assertEqual("BLOCKED", blocked["status"])
        self.assertIn("path.output_subdir", diagnostic_codes(blocked))

        existing = self.normalize_context("attempt-existing")
        output = Path(existing["attempt_dir"]) / existing["parameters"]["output_subdir"]
        output.mkdir(parents=True)
        sentinel = output / "keep.txt"
        sentinel.write_text("do not overwrite\n", encoding="utf-8")
        blocked = self.adapter.plan(existing)
        self.assertEqual("BLOCKED", blocked["status"])
        self.assertIn("path.output_exists", diagnostic_codes(blocked))
        self.assertEqual("do not overwrite\n", sentinel.read_text(encoding="utf-8"))

    def test_plan_enforces_historical_seed_and_order_contract(self) -> None:
        mutations = (
            ("seed_status", "KNOWN", "seed.historical_unknown"),
            ("acknowledge_uncontrolled_seed", False, "seed.acknowledgement_required"),
            ("preserve_historical_order", False, "parameter.preserve_historical_order"),
        )
        for parameter, value, expected_code in mutations:
            with self.subTest(parameter=parameter):
                context = self.normalize_context(f"attempt-{parameter}")
                context["parameters"][parameter] = value
                blocked = self.adapter.plan(context)
                self.assertEqual("BLOCKED", blocked["status"])
                self.assertIn(expected_code, diagnostic_codes(blocked))

    def test_check_collect_are_read_only_and_tampered_hash_fails(self) -> None:
        context = self.normalize_context()
        _, output = self.run_adapter_plan(context)
        before = file_snapshot(output)
        checked = self.adapter.check(context)
        collected = self.adapter.collect(context)
        after = file_snapshot(output)

        self.assertEqual(before, after)
        self.assertEqual("OK", checked["status"], checked.get("diagnostics"))
        self.assertEqual("OK", collected["status"], collected.get("diagnostics"))
        self.assertEqual(5, collected["metrics"]["generated_structure_count"])
        self.assertEqual(2, collected["metrics"]["selected_structure_count"])
        artifact_names = {Path(item["path"]).name for item in collected["artifacts"]}
        self.assertTrue(
            {
                "sampling-result.json",
                "ssw-structures.json",
                "selected-structures.json",
                "lasp-run-metadata.json",
                "input-000001.arc",
                "input-000002.arc",
            }.issubset(artifact_names)
        )

        selected = read_structures(output / "selected-structures.json")
        selected_path = output / selected[0]["output_file"]
        selected_path.write_bytes(selected_path.read_bytes() + b"tampered\n")
        failed = self.adapter.check(context)
        self.assertEqual("FAIL", failed["status"])
        self.assertTrue(
            any("hash" in code or "fingerprint" in code for code in diagnostic_codes(failed)),
            failed.get("diagnostics"),
        )

    def test_missing_sampling_result_waits(self) -> None:
        context = self.normalize_context()
        plan = self.adapter.plan(context)
        self.assertEqual("READY", plan["status"], plan.get("diagnostics"))
        context["execution"] = {"returncode": 0, "plan": plan}
        checked = self.adapter.check(context)
        self.assertEqual("WAIT", checked["status"])

    def test_artifact_path_escape_fails_completion(self) -> None:
        context = self.normalize_context()
        _, output = self.run_adapter_plan(context)
        result_path = output / "sampling-result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["artifacts"][0]["path"] = "../escape.arc"
        write_json(result_path, result)

        failed = self.adapter.check(context)
        self.assertEqual("FAIL", failed["status"])
        self.assertIn("result.artifact_escape", diagnostic_codes(failed))

    def test_checker_recomputes_energy_filter_instead_of_trusting_manifest_flags(self) -> None:
        context = self.normalize_context()
        _, output = self.run_adapter_plan(context)
        structures_path = output / "ssw-structures.json"
        structure_manifest = json.loads(structures_path.read_text(encoding="utf-8"))
        positive_energy = structure_manifest["structures"][2]
        self.assertEqual(1.0, positive_energy["energy_ev"])
        positive_energy["energy_filter_pass"] = True
        positive_energy["energy_filter_order"] = 3
        write_json(structures_path, structure_manifest)
        refresh_result_artifact(output, structures_path)
        result_path = output / "sampling-result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["counts"]["energy_accepted_count"] = 5
        write_json(result_path, result)

        failed = self.adapter.check(context)
        self.assertEqual("FAIL", failed["status"])

    def test_checker_recomputes_deterministic_structure_ids(self) -> None:
        context = self.normalize_context()
        _, output = self.run_adapter_plan(context)
        structures_path = output / "ssw-structures.json"
        selected_path = output / "selected-structures.json"
        structures = json.loads(structures_path.read_text(encoding="utf-8"))
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        self.assertEqual(
            structures["structures"][0]["structure_id"],
            selected["structures"][0]["structure_id"],
        )
        structures["structures"][0]["structure_id"] = "forged-structure-id"
        selected["structures"][0]["structure_id"] = "forged-structure-id"
        write_json(structures_path, structures)
        write_json(selected_path, selected)
        refresh_result_artifact(output, structures_path)
        refresh_result_artifact(output, selected_path)

        failed = self.adapter.check(context)
        self.assertEqual("FAIL", failed["status"])

    def test_checker_binds_result_to_approved_source_fingerprints(self) -> None:
        context = self.normalize_context()
        _, _ = self.run_adapter_plan(context)
        allstr = self.historical / "allstr.arc"
        allstr.write_bytes(allstr.read_bytes() + arc_payload([-6.0])[len(ARC_HEADER) :])

        failed = self.adapter.check(context)
        self.assertEqual("FAIL", failed["status"])
        self.assertTrue(
            {
                "result.approved_input_fingerprint",
                "result.metadata_source_hash",
                "result.source_generated_count",
            }
            & diagnostic_codes(failed)
        )

    def test_check_fails_when_approved_plan_omits_input_fingerprints(self) -> None:
        context = self.normalize_context()
        self.run_adapter_plan(context)
        del context["execution"]["plan"]["input_fingerprints"]

        failed = self.adapter.check(context)

        self.assertEqual("FAIL", failed["status"])
        self.assertTrue(
            any("fingerprint" in code for code in diagnostic_codes(failed)),
            failed.get("diagnostics"),
        )

    def test_explicit_result_manifest_cannot_override_execution_plan(self) -> None:
        context = self.normalize_context()
        _, output = self.run_adapter_plan(context)
        alternate = output / "alternate-result.json"
        alternate.write_bytes((output / "sampling-result.json").read_bytes())
        context["inputs"]["result_manifest"] = str(alternate)

        failed = self.adapter.check(context)
        self.assertEqual("FAIL", failed["status"])
        self.assertIn("result.manifest_plan_mismatch", diagnostic_codes(failed))

    def test_symlink_unknown_resource_and_excessive_frame_limit_are_blocked(self) -> None:
        symlink = self.root / "lasp-input-link"
        symlink.symlink_to(self.lasp_input)
        context = self.normalize_context("attempt-symlink")
        context["inputs"]["lasp_input"] = str(symlink)
        blocked = self.adapter.plan(context)
        self.assertEqual("BLOCKED", blocked["status"])
        self.assertIn("path.lasp_input", diagnostic_codes(blocked))

        context = self.normalize_context("attempt-resource")
        context["resources"]["mpi_lancher"] = "typo"
        blocked = self.adapter.plan(context)
        self.assertEqual("BLOCKED", blocked["status"])
        self.assertIn("resource.unknown", diagnostic_codes(blocked))

        context = self.normalize_context("attempt-limit")
        context["parameters"]["max_frames"] = 10001
        blocked = self.adapter.plan(context)
        self.assertEqual("BLOCKED", blocked["status"])
        self.assertIn("parameter.max_frames", diagnostic_codes(blocked))

        single_point = self.root / "single-point.lasp.in"
        single_point.write_text(
            "potential vasp\nexplore_type ssw\nSSW.SSWsteps 0\n", encoding="utf-8"
        )
        context = self.normalize_context("attempt-not-ssw")
        context["inputs"]["lasp_input"] = str(single_point)
        blocked = self.adapter.plan(context)
        self.assertEqual("BLOCKED", blocked["status"])
        self.assertIn("input.lasp_ssw_contract", diagnostic_codes(blocked))

    def test_undeclared_artifact_role_is_not_collected(self) -> None:
        context = self.normalize_context()
        _, output = self.run_adapter_plan(context)
        undeclared = output / "unreviewed-private-input"
        undeclared.write_text("must not be collected\n", encoding="utf-8")
        result_path = output / "sampling-result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["artifacts"].append(
            {
                "role": "undeclared-private-input",
                "path": undeclared.name,
                "media_type": "application/octet-stream",
                "sha256": sha256_bytes(undeclared.read_bytes()),
                "size_bytes": undeclared.stat().st_size,
            }
        )
        write_json(result_path, result)

        failed = self.adapter.check(context)
        self.assertEqual("FAIL", failed["status"])
        self.assertIn("result.artifact_role_unknown", diagnostic_codes(failed))

    def test_failed_normalization_retains_explicit_incomplete_marker(self) -> None:
        broken = self.root / "broken-history"
        broken.mkdir()
        (broken / "allstr.arc").write_text("not an ARC archive\n", encoding="utf-8")
        output = self.root / "broken-output"
        completed = subprocess.run(
            [
                PYTHON_EXECUTABLE,
                str(WRAPPER_PATH),
                "normalize-replay",
                "--historical-run-dir",
                str(broken),
                "--lasp-input",
                str(self.lasp_input),
                "--output-dir",
                str(output),
                "--historical-source-id",
                "fixture://broken",
                "--selection-stride",
                "1",
                "--max-frames",
                "5",
                "--seed-status",
                UNKNOWN_SEED,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(0, completed.returncode)
        self.assertTrue((output / "INCOMPLETE.json").is_file())
        self.assertFalse((output / "sampling-result.json").exists())

    def test_execute_fake_binary_minimal_smoke(self) -> None:
        executable = self.root / "fake-lasp"
        fake_source = (
            f"#!{PYTHON_EXECUTABLE}\n"
            "from pathlib import Path\n"
            f"Path('allstr.arc').write_bytes({arc_payload([-10.0, -9.0, 1.0, -8.0, -7.0])!r})\n"
            f"Path('best.arc').write_bytes({arc_payload([-10.0])!r})\n"
            f"Path('md.arc').write_bytes({arc_payload([-9.0, -8.0])!r})\n"
        )
        executable.write_text(fake_source, encoding="utf-8")
        executable.chmod(0o755)
        input_structure = self.root / "input.arc"
        input_structure.write_bytes(arc_payload([-10.0]))
        attempt = self.root / ".mlipflow" / "runs" / "lasp-execute" / "attempt-1"
        parameters = self.shared_parameters("lasp-ssw-execute", "lasp-executed")
        parameters["lasp_version"] = "fixture-1.0"
        context = {
            "project_root": str(self.root),
            "attempt_dir": str(attempt),
            "inputs": {
                "lasp_executable": str(executable),
                "input_structure": str(input_structure),
                "lasp_input": str(self.lasp_input),
            },
            "parameters": parameters,
            "backend": "local",
            "resources": {"python_executable": PYTHON_EXECUTABLE},
        }

        plan = self.adapter.plan(context)
        self.assertEqual("READY", plan["status"], plan.get("diagnostics"))
        self.assertEqual("lasp-ssw-execute", plan["operation"])
        self.assertFalse(plan["shell"])
        _, output = self.run_adapter_plan(context)
        result = json.loads((output / "sampling-result.json").read_text(encoding="utf-8"))
        self.assertEqual("execute", result["mode"])
        self.assertEqual("lasp-ssw-execute", result["operation"])
        self.assertEqual(2, result["counts"]["selected_structure_count"])
        self.assertEqual("OK", self.adapter.check(context)["status"])

        staged_input = output / "raw-run" / "input.arc"
        original_staged_input = staged_input.read_bytes()
        staged_input.write_bytes(original_staged_input + b"tampered\n")
        staged_tamper = self.adapter.check(context)
        self.assertEqual("FAIL", staged_tamper["status"])
        self.assertIn("result.staged_input_content", diagnostic_codes(staged_tamper))
        staged_input.write_bytes(original_staged_input)

        executable.write_text(fake_source + "# changed after approval\n", encoding="utf-8")
        executable.chmod(0o755)
        rebound = self.adapter.check(context)
        self.assertEqual("FAIL", rebound["status"])
        self.assertTrue(
            {
                "result.executable_identity",
                "result.approved_input_fingerprint",
            }
            & diagnostic_codes(rebound)
        )


class LaspHistoricalReplayReportTests(unittest.TestCase):
    def test_committed_report_is_bounded_fingerprinted_and_nonsecret(self) -> None:
        report_path = ROOT / "reports" / "lasp_ssw_historical_replay.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual("HISTORICAL_POSTPROCESS_REPLAY_PASS", report["status"])
        claim = report["scientific_claim"]
        self.assertTrue(claim["historical_archive_postprocessing_replayed"])
        self.assertFalse(claim["licensed_lasp_executed"])
        self.assertFalse(claim["ssw_numerical_parity_established"])
        self.assertFalse(claim["scheduler_or_hpc_integration_validated"])
        self.assertEqual(
            sha256_bytes(ADAPTER_PATH.read_bytes()).removeprefix("sha256:"),
            report["implementation"]["adapter"]["sha256"],
        )
        self.assertEqual(
            sha256_bytes(WRAPPER_PATH.read_bytes()).removeprefix("sha256:"),
            report["implementation"]["wrapper"]["sha256"],
        )
        self.assertEqual(
            sha256_bytes(PLUGIN_MANIFEST_PATH.read_bytes()).removeprefix("sha256:"),
            report["implementation"]["plugin_manifest"]["sha256"],
        )
        self.assertEqual(
            {
                "generated_structure_count": 6,
                "energy_accepted_count": 5,
                "selected_structure_count": 5,
                "aimd_seed_candidate_count": 2,
                "md_structure_count": 0,
            },
            {
                key: report["results"]["initial_run"][key]
                for key in (
                    "generated_structure_count",
                    "energy_accepted_count",
                    "selected_structure_count",
                    "aimd_seed_candidate_count",
                    "md_structure_count",
                )
            },
        )
        self.assertEqual(
            (66, 25, 9, 1, 17),
            tuple(
                report["results"]["rerun"][key]
                for key in (
                    "generated_structure_count",
                    "energy_accepted_count",
                    "selected_structure_count",
                    "aimd_seed_candidate_count",
                    "md_structure_count",
                )
            ),
        )
        audit = report["source_audit"]
        self.assertFalse(audit["remote_files_modified"])
        self.assertFalse(audit["raw_structure_files_included_in_repository"])
        self.assertFalse(audit["temporary_local_copies_retained_after_audit"])
        serialized = json.dumps(report, sort_keys=True)
        for forbidden in ("/Users/", "IdentityFile", "HostName", "RsaKey"):
            self.assertNotIn(forbidden, serialized)


if __name__ == "__main__":
    unittest.main()
