"""Focused tests for safe benchmark/training thin adapters.

No scientific framework is installed or invoked.  The tests exercise only
validation, argv planning, and collection of small explicit JSON fixtures.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_adapter(plugin_id: str):
    path = ROOT / "plugins" / plugin_id / "adapter.py"
    spec = importlib.util.spec_from_file_location(f"test_{plugin_id.replace('-', '_')}", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Adapter()


class BenchmarkAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = load_adapter("mlip-benchmark")
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.attempt = self.root / "attempt"
        self.context = {
            "project_root": str(self.root),
            "attempt_dir": str(self.attempt),
            "inputs": {
                "executable": "/usr/bin/python3",
                "script": "tools/benchmark.py",
                "model": "models/chgnet.pth",
                "benchmark_dataset": "data/test.json",
                "benchmark_config": "configs/benchmark.json",
                "result_manifest": "benchmark-result.json",
            },
            "parameters": {
                "operation": "evaluate-static",
                "model_family": "chgnet",
                "task": "ionic-transport",
                "scenario": "static-holdout-v1",
                "energy_normalization": "per-atom",
                "stress_convention": "VASP sign; GPa; xx,yy,zz,xy,yz,zx",
                "dataset_fingerprint": "sha256:dataset",
                "model_fingerprint": "sha256:model",
            },
            "backend": "local",
            "resources": {"cpus": 4},
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_result(self, *, status: str = "OK", metrics_file: bool = False) -> Path:
        self.attempt.mkdir(parents=True, exist_ok=True)
        metric_records = {
            "energy_rmse_ev_atom": {
                "value": 0.031,
                "unit": "eV/atom",
                "direction": "minimize",
                "sample_count": 24,
            },
            "force_rmse_ev_a": {
                "value": 0.052,
                "unit": "eV/angstrom",
                "direction": "minimize",
                "sample_count": 1728,
            },
        }
        payload = {
            "schema_version": 1,
            "plugin_id": "mlip-benchmark",
            "status": status,
            "model_family": "chgnet",
            "task": "ionic-transport",
            "scenario": "static-holdout-v1",
            "dataset_fingerprint": "sha256:dataset",
            "model_fingerprint": "sha256:model",
            "conventions": {
                "energy_normalization": "per-atom",
                "stress_convention": "VASP sign; GPa; xx,yy,zz,xy,yz,zx",
            },
            "artifacts": [],
        }
        if metrics_file:
            (self.attempt / "metrics.json").write_text(
                json.dumps({"metrics": metric_records}), encoding="utf-8"
            )
            payload["metrics_file"] = "metrics.json"
        else:
            payload["metrics"] = metric_records
        path = self.attempt / "benchmark-result.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_plan_is_pure_and_builds_explicit_argv(self) -> None:
        before = set(self.root.rglob("*"))
        plan = self.adapter.plan(self.context)
        self.assertEqual("READY", plan["status"])
        self.assertTrue(plan["executable"])
        self.assertEqual("/usr/bin/python3", plan["argv"][0])
        self.assertIn("--result-manifest", plan["argv"])
        self.assertIn(str(self.attempt / "benchmark-result.json"), plan["argv"])
        self.assertEqual([str(self.attempt / "benchmark-result.json")], plan["expected_outputs"])
        self.assertEqual(before, set(self.root.rglob("*")))

    def test_shell_and_output_escape_are_blocked(self) -> None:
        self.context["inputs"]["executable"] = "/bin/bash"
        self.context["inputs"]["result_manifest"] = "../overwrite.json"
        plan = self.adapter.plan(self.context)
        self.assertEqual("BLOCKED", plan["status"])
        codes = {item["code"] for item in plan["diagnostics"]}
        self.assertIn("benchmark.shell_forbidden", codes)
        self.assertIn("benchmark.output_scope", codes)

    def test_stdout_is_never_treated_as_result(self) -> None:
        self.attempt.mkdir(parents=True)
        (self.attempt / "stdout.log").write_text("RMSE = 0.001\n", encoding="utf-8")
        checked = self.adapter.check(self.context)
        self.assertEqual("WAIT", checked["status"])

    def test_collects_standard_inline_metrics(self) -> None:
        self._write_result()
        collected = self.adapter.collect(self.context)
        self.assertEqual("OK", collected["status"])
        self.assertEqual(0.031, collected["metrics"]["energy_rmse_ev_atom"])
        self.assertTrue(any(item["role"] == "result-manifest" for item in collected["artifacts"]))

    def test_collects_explicit_metrics_json(self) -> None:
        self._write_result(metrics_file=True)
        collected = self.adapter.collect(self.context)
        self.assertEqual("OK", collected["status"])
        self.assertTrue(any(item["role"] == "metrics" for item in collected["artifacts"]))

    def test_nonfinite_or_under_specified_metric_fails(self) -> None:
        path = self._write_result()
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["metrics"]["force_rmse_ev_a"].pop("unit")
        path.write_text(json.dumps(payload), encoding="utf-8")
        checked = self.adapter.check(self.context)
        self.assertEqual("FAIL", checked["status"])
        self.assertIn("benchmark.metric_unit", {item["code"] for item in checked["diagnostics"]})

    def test_result_identity_must_match_approved_plan(self) -> None:
        path = self._write_result()
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["scenario"] = "different-scenario"
        payload["model_fingerprint"] = "sha256:different-model"
        path.write_text(json.dumps(payload), encoding="utf-8")
        checked = self.adapter.check(self.context)
        self.assertEqual("FAIL", checked["status"])
        codes = {item["code"] for item in checked["diagnostics"]}
        self.assertIn("benchmark.scenario_mismatch", codes)
        self.assertIn("benchmark.model_fingerprint_mismatch", codes)

    def test_collect_existing_is_read_only_and_not_an_execution_plan(self) -> None:
        path = self._write_result()
        self.context["inputs"] = {"result_manifest": str(path)}
        self.context["parameters"]["operation"] = "collect-existing"
        plan = self.adapter.plan(self.context)
        self.assertEqual("BLOCKED", plan["status"])
        self.assertEqual("OK", self.adapter.collect(self.context)["status"])


class TrainingAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = load_adapter("mlip-training")
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.attempt = self.root / "attempt"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def context(self, framework: str, operation: str = "train") -> dict:
        inputs = {
            "executable": "/usr/bin/python3",
            "script": f"wrappers/{framework}_train.py",
            "config": f"configs/{framework}.json",
            "data": "data/labeled",
            "output": "model.bin",
            "result_manifest": "training-result.json",
        }
        if operation == "finetune":
            inputs["foundation_model"] = "models/foundation.model"
        return {
            "project_root": str(self.root),
            "attempt_dir": str(self.attempt),
            "inputs": inputs,
            "parameters": {
                "framework": framework,
                "operation": operation,
                "seed": 20260810,
                "device": "cpu",
                "precision": "float64",
                "dataset_fingerprint": "sha256:dataset",
                "config_fingerprint": "sha256:config",
            },
            "backend": "local",
            "resources": {"cpus": 8},
        }

    def _write_result(self, context: dict, *, framework: str, seed: int = 20260810) -> None:
        self.attempt.mkdir(parents=True, exist_ok=True)
        (self.attempt / "model.bin").write_bytes(b"small-test-model")
        payload = {
            "schema_version": 1,
            "plugin_id": "mlip-training",
            "status": "OK",
            "framework": framework,
            "framework_version": "test-only",
            "operation": context["parameters"]["operation"],
            "seed": seed,
            "device": context["parameters"]["device"],
            "precision": context["parameters"]["precision"],
            "dataset_fingerprint": "sha256:dataset",
            "config_fingerprint": "sha256:config",
            "model_artifact": {"path": "model.bin", "media_type": "application/octet-stream"},
            "metrics": {"best_validation_loss": 0.012},
        }
        (self.attempt / "training-result.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_all_four_frameworks_have_explicit_wrapper_plans(self) -> None:
        for framework in ("deepmd", "m3gnet", "chgnet", "mace"):
            with self.subTest(framework=framework):
                context = self.context(framework)
                before = set(self.root.rglob("*"))
                plan = self.adapter.plan(context)
                self.assertEqual("READY", plan["status"])
                self.assertEqual("/usr/bin/python3", plan["argv"][0])
                for flag in ("--framework", "--config", "--data", "--output", "--seed"):
                    self.assertIn(flag, plan["argv"])
                self.assertEqual(before, set(self.root.rglob("*")))

    def test_finetune_support_and_deepmd_boundary(self) -> None:
        for framework in ("m3gnet", "chgnet", "mace"):
            context = self.context(framework, "finetune")
            context["parameters"]["foundation_model_fingerprint"] = "sha256:foundation"
            plan = self.adapter.plan(context)
            self.assertEqual("READY", plan["status"])
            self.assertIn("--foundation-model", plan["argv"])
        blocked = self.adapter.plan(self.context("deepmd", "finetune"))
        self.assertEqual("BLOCKED", blocked["status"])
        self.assertIn("training.operation", {item["code"] for item in blocked["diagnostics"]})

    def test_training_shell_and_output_escape_are_blocked(self) -> None:
        context = self.context("mace")
        context["inputs"]["executable"] = "zsh"
        context["inputs"]["output"] = "/tmp/model.pt"
        result = self.adapter.plan(context)
        self.assertEqual("BLOCKED", result["status"])
        codes = {item["code"] for item in result["diagnostics"]}
        self.assertIn("training.shell_forbidden", codes)
        self.assertIn("training.output_scope", codes)

    def test_check_waits_for_explicit_manifest(self) -> None:
        self.assertEqual("WAIT", self.adapter.check(self.context("mace"))["status"])

    def test_collects_model_and_training_metrics(self) -> None:
        context = self.context("mace")
        self._write_result(context, framework="mace")
        collected = self.adapter.collect(context)
        self.assertEqual("OK", collected["status"])
        self.assertEqual(0.012, collected["metrics"]["best_validation_loss"])
        self.assertEqual({"model", "result-manifest"}, {item["role"] for item in collected["artifacts"]})

    def test_seed_or_model_output_mismatch_fails(self) -> None:
        context = self.context("chgnet")
        self._write_result(context, framework="chgnet", seed=7)
        checked = self.adapter.check(context)
        self.assertEqual("FAIL", checked["status"])
        self.assertIn("training.seed_mismatch", {item["code"] for item in checked["diagnostics"]})


if __name__ == "__main__":
    unittest.main()
