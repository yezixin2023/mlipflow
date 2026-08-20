"""Scheduled fresh benchmark planning and core staging contract tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mlipflow.config import load_project
from mlipflow.plugins import discover_plugins, load_adapter
from mlipflow.services import initialize, make_run_plan
from mlipflow.services.contracts import _scheduled_contract

from .helpers import project_config, write_json
from .test_scheduled_dft import FakeTemplateLibrary, PLUGINS, write_site


def library() -> FakeTemplateLibrary:
    templates = dict(FakeTemplateLibrary().templates)
    templates["benchmark-chgnet-canonical/run.sh"] = """#!/bin/bash
# project={{PROJECT_ID}} node={{NODE_ID}} attempt={{ATTEMPT}}
# cpus={{CPUS}}
# inputs={{INPUT_DIR}}
# outputs={{OUTPUT_DIR}}
cd {{RUN_DIR}}
"""
    return FakeTemplateLibrary(templates)


def build_project(root: Path) -> Path:
    write_json(
        root / "model-reference.json",
        {
            "schema_version": 1,
            "model_id": "chgnet-loop-test",
            "framework": "chgnet",
            "relative_path": "chgnet/chgnet-loop-test.pt",
            "kind": "file",
        },
    )
    write_json(
        root / "benchmark-dataset-reference.json",
        {
            "schema_version": 1,
            "dataset_id": "loop-test-dataset",
            "relative_path": "loop-test-dataset/benchmark/test.json",
            "kind": "file",
        },
    )
    node = {
        "id": "benchmark-chgnet",
        "uses": "mlip-benchmark@0",
        "mode": "execute",
        "backend": "ssh-slurm",
        "backend_profile": "cluster-a",
        "inputs": {
            "model_reference": "model-reference.json",
            "benchmark_dataset_reference": "benchmark-dataset-reference.json",
            "output_dir": "fresh",
        },
        "parameters": {
            "operation": "evaluate-fresh",
            "model_family": "chgnet",
            "task": "static-pes",
            "scenario": "loop-test-v1",
            "split": "test",
            "targets": ["energy", "force", "stress"],
            "units": {
                "energy": "eV/atom",
                "force": "eV/angstrom",
                "stress": "eV/angstrom^3",
            },
            "energy_normalization": "per-atom",
            "stress_convention": "ase-voigt-xx-yy-zz-yz-xz-xy",
            "device": "cpu",
        },
        "resources": {
            "cpus": 1,
            "gpus": 0,
            "memory": "2G",
            "walltime": "00:05:00",
        },
    }
    write_json(root / "project.yaml", project_config([node]))
    return write_site(root)


class ScheduledBenchmarkPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.site = build_project(self.root)
        initialize(self.root)
        self.project = load_project(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_core_accepts_package_owned_benchmark_runtime_staging(self) -> None:
        plan = make_run_plan(
            self.project,
            "benchmark-chgnet",
            PLUGINS,
            self.site,
            library(),
        )
        adapter = plan["adapter_plan"]
        self.assertEqual("READY", adapter["status"], adapter.get("diagnostics"))
        portable_sources = {
            item["remote_name"]: item["source"]
            for item in adapter["scheduled_execution"]["staged_files"]
        }
        self.assertTrue(portable_sources["model_runtime.py"].startswith("{PACKAGE_DIR}/"))

        plugin = discover_plugins(PLUGINS)["mlip-benchmark"]
        contract = _scheduled_contract(
            self.project,
            plugin,
            plan,
            node_id="benchmark-chgnet",
            attempt=1,
        )
        self.assertEqual("benchmark-chgnet-canonical", contract["template_family"])
        staged = {item["remote_name"] for item in contract["staged_files"]}
        self.assertTrue(
            {
                "model-reference.json",
                "benchmark-dataset-reference.json",
                "fresh_benchmark_cluster.py",
                "fresh_benchmark.py",
                "model_runtime.py",
            }.issubset(staged)
        )
        fetched = {item["remote_name"] for item in contract["fetch_outputs"]}
        self.assertTrue(
            {
                "prediction_evidence.json",
                "metrics.json",
                "benchmark_summary.csv",
                "model_ranking.json",
                "provenance.json",
                "cluster-benchmark-report.json",
                "completion.json",
            }.issubset(fetched)
        )

    def test_upstream_references_supply_readable_paths(self) -> None:
        plugin = discover_plugins(PLUGINS)["mlip-benchmark"]
        adapter = load_adapter(plugin)
        node = self.project.node("benchmark-chgnet")
        parameters = dict(node["parameters"])
        context = {
            "project_root": str(self.root),
            "attempt_dir": str(self.root / ".mlipflow/runs/benchmark-chgnet/attempt-1"),
            "backend": "ssh-slurm",
            "inputs": {
                "model_reference": str((self.root / "model-reference.json").resolve()),
                "benchmark_dataset_reference": str(
                    (self.root / "benchmark-dataset-reference.json").resolve()
                ),
                "output_dir": "fresh",
            },
            "parameters": parameters,
            "resources": dict(node["resources"]),
        }

        plan = adapter.plan(context)

        self.assertEqual("READY", plan["status"], plan.get("diagnostics"))
        calculation = plan["fresh_calculation"]
        self.assertEqual("chgnet-loop-test", calculation["model"]["id"])
        self.assertEqual(
            "chgnet/chgnet-loop-test.pt", calculation["model"]["relative_path"]
        )
        self.assertEqual("loop-test-dataset", calculation["dataset"]["id"])
        self.assertEqual(
            "loop-test-dataset/benchmark/test.json",
            calculation["dataset"]["relative_path"],
        )


if __name__ == "__main__":
    unittest.main()
