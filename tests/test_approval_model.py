from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlipflow.backends import ExecutionResult
from mlipflow.config import load_project
from mlipflow.errors import ApprovalError, CapabilityError
from mlipflow.services import initialize, make_run_plan, retry, run_node, state_path
from mlipflow.state import RunState, StateStore

from .helpers import project_config, run_cli, write_json


class FixtureAdapter:
    def __init__(self, default_operation="train"):
        self.default_operation = default_operation

    def operation(self, context):
        return context.get("parameters", {}).get(
            "operation", self.default_operation
        )

    def validate(self, context):
        return []

    def plan(self, context):
        return {
            "status": "READY",
            "executable": True,
            "argv": ["true"],
            "cwd": context["attempt_dir"],
        }

    def check(self, context):
        return {"status": "OK"}

    def collect(self, context):
        return {"status": "OK", "artifacts": [], "metrics": {}}


class ApprovalTests(unittest.TestCase):
    def test_inspect_reports_effective_node_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(
                root / "project.yaml",
                project_config(
                    [
                        {
                            "id": "prepare",
                            "uses": "dft-labeling",
                            "parameters": {"operation": "vasp-prepare"},
                        }
                    ]
                ),
            )
            code, stdout, stderr = run_cli(
                [
                    "--project",
                    str(root),
                    "--format",
                    "json",
                    "inspect",
                    "prepare",
                ]
            )
            self.assertEqual(0, code, stderr)
            data = json.loads(stdout)["data"]
            self.assertEqual("vasp-prepare", data["action"])
            self.assertFalse(data["approval_required"])
            self.assertEqual(["label"], data["capability"]["approval_operations"])
            self.assertNotIn("approval_required", data["capability"])

    def test_cli_explains_expensive_or_scheduled_approval_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(
                root / "project.yaml",
                project_config([{"id": "train", "uses": "mlip-training"}]),
            )
            code, _, stderr = run_cli(
                ["--project", str(root), "run", "train"]
            )
            self.assertEqual(2, code)
            self.assertIn(
                "approval is required for expensive or scheduled execution", stderr
            )

    def test_operation_level_approval_matrix(self) -> None:
        cases = {
            ("ionic-transport", "analyze-existing"): False,
            ("ionic-transport", "md-smoke-and-analyze"): False,
            ("dft-labeling", "vasp-prepare"): False,
            ("dft-labeling", "dataset-assemble"): False,
            ("dft-labeling", "label"): True,
            ("lammps-md", "lammps-prepare"): False,
            ("lammps-md", "execute"): True,
            ("active-learning", "committee-evaluate"): False,
            ("active-learning", "select-candidates"): False,
            ("active-learning", "assess-round"): False,
            ("mlip-benchmark", "normalize-execute"): False,
            ("mlip-benchmark", "normalize-replay"): False,
            ("mlip-benchmark", "evaluate-static"): False,
            ("mlip-benchmark", "evaluate-fresh"): True,
            ("candidate-ranking", "rank-candidates"): False,
            ("electrochemical-voltage", "compute-from-energies"): False,
            ("high-entropy-structure", "generate-sqs"): False,
            ("pes-sampling", "direct-select"): False,
            ("pes-sampling", "lasp-input-prepare"): False,
            ("pes-sampling", "merge-structures"): False,
            ("pes-sampling", "lasp-ssw-normalize-replay"): False,
            ("pes-sampling", "lasp-ssw-execute"): True,
            ("mlip-training", "train"): True,
            ("mlip-training", "finetune"): True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nodes = [
                {
                    "id": f"node-{index}",
                    "uses": capability_id,
                    "parameters": {"operation": operation},
                }
                for index, (capability_id, operation) in enumerate(cases)
            ]
            write_json(root / "project.yaml", project_config(nodes))
            project = load_project(root)
            for node, expected in zip(nodes, cases.values()):
                with self.subTest(node=node["id"]):
                    plan = make_run_plan(project, node["id"])
                    self.assertEqual(expected, plan["approval_required"])
                    self.assertEqual(
                        node["parameters"]["operation"], plan["operation"]
                    )

    def test_arrhenius_style_transport_analysis_requires_no_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(
                root / "project.yaml",
                project_config(
                    [
                        {
                            "id": "arrhenius",
                            "uses": "ionic-transport",
                            "parameters": {
                                "operation": "analyze-existing",
                                "arrhenius_extrapolate_temperature_k": 300.0,
                            },
                        }
                    ]
                ),
            )
            plan = make_run_plan(load_project(root), "arrhenius")
            self.assertFalse(plan["approval_required"])

    def test_local_analysis_runs_without_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(
                root / "project.yaml",
                project_config(
                    [
                        {
                            "id": "transport",
                            "uses": "ionic-transport",
                            "parameters": {"operation": "analyze-existing"},
                        }
                    ]
                ),
            )
            initialize(root)
            project = load_project(root)
            adapter = FixtureAdapter("analyze-existing")
            with patch(
                "mlipflow.services.commands.load_adapter", return_value=adapter
            ), patch(
                "mlipflow.services.execution.load_adapter", return_value=adapter
            ), patch(
                "mlipflow.services.LocalBackend.run",
                return_value=ExecutionResult(0, "", ""),
            ):
                result = run_node(project, "transport")
            self.assertEqual("OK", result["step"]["state"])

    def test_ssh_slurm_always_requires_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(
                root / "project.yaml",
                project_config(
                    [
                        {
                            "id": node_id,
                            "uses": capability_id,
                            "backend": "ssh-slurm",
                            "parameters": {"operation": operation},
                            "resources": {
                                "cpus": 1,
                                "gpus": 0,
                                "memory": "1G",
                                "walltime": "00:10:00",
                            },
                        }
                        for node_id, capability_id, operation in (
                            (
                                "scheduled-selection",
                                "active-learning",
                                "select-candidates",
                            ),
                            (
                                "scheduled-assembly",
                                "dft-labeling",
                                "dataset-assemble",
                            ),
                        )
                    ]
                ),
            )
            project = load_project(root)
            for node_id in ("scheduled-selection", "scheduled-assembly"):
                with self.subTest(node=node_id):
                    plan = make_run_plan(project, node_id)
                    self.assertTrue(plan["approval_required"])

    def test_scientific_validation_blocks_independently_of_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(
                root / "project.yaml",
                project_config(
                    [
                        {
                            "id": "invalid-prepare",
                            "uses": "dft-labeling",
                            "parameters": {"operation": "vasp-prepare"},
                        },
                        {
                            "id": "invalid-label",
                            "uses": "dft-labeling",
                            "parameters": {"operation": "label"},
                        }
                    ]
                ),
            )
            initialize(root)
            project = load_project(root)
            plan = make_run_plan(project, "invalid-prepare")
            self.assertFalse(plan["approval_required"])
            self.assertEqual("BLOCKED", plan["adapter_plan"]["status"])
            with self.assertRaisesRegex(CapabilityError, "blocked execution"):
                run_node(project, "invalid-prepare")
            expensive_plan = make_run_plan(project, "invalid-label")
            self.assertTrue(expensive_plan["approval_required"])
            self.assertEqual("BLOCKED", expensive_plan["adapter_plan"]["status"])
            with self.assertRaisesRegex(CapabilityError, "blocked execution"):
                run_node(project, "invalid-label", approval=True)

    def test_expensive_run_requires_explicit_boolean_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(
                root / "project.yaml",
                project_config([{"id": "train", "uses": "mlip-training"}]),
            )
            initialize(root)
            project = load_project(root)
            adapter = FixtureAdapter("train")
            with patch(
                "mlipflow.services.commands.load_adapter", return_value=adapter
            ):
                plan = make_run_plan(project, "train")
                self.assertTrue(plan["approval_required"])
                with self.assertRaises(ApprovalError):
                    run_node(project, "train")
            self.assertFalse((root / ".mlipflow/runs/train/attempt-1").exists())

            with patch(
                "mlipflow.services.commands.load_adapter", return_value=adapter
            ), patch(
                "mlipflow.services.execution.load_adapter", return_value=adapter
            ), patch(
                "mlipflow.services.LocalBackend.run",
                return_value=ExecutionResult(0, "", ""),
            ):
                result = run_node(project, "train", approval=True)
            self.assertEqual("OK", result["step"]["state"])

    def test_retry_uses_attempt_number_as_its_only_counter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(
                root / "project.yaml",
                project_config(
                    [
                        {"id": "x", "uses": "candidate-ranking", "mode": "replay"},
                        {"id": "y", "uses": "candidate-ranking", "mode": "replay"},
                    ]
                ),
            )
            initialize(root)
            project = load_project(root)
            first = make_run_plan(project, "x")
            other = make_run_plan(project, "y")
            with StateStore(state_path(project), readonly=False) as store:
                step = store.latest_step(project.project_id, "x")
                store.transition(step.run_id, RunState.RUNNING)
                store.transition(step.run_id, RunState.FAIL)
            retry(project, "x")
            second = make_run_plan(project, "x")
            self.assertEqual(("x", 1), (first["node_id"], first["attempt"]))
            self.assertEqual(("y", 1), (other["node_id"], other["attempt"]))
            self.assertEqual(("x", 2), (second["node_id"], second["attempt"]))


if __name__ == "__main__":
    unittest.main()
