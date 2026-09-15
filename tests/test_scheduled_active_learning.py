"""Scheduled committee inference planning and fresh-evidence tests."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mlipflow.config import load_project
from mlipflow.services import initialize, make_run_plan
from mlipflow.services.contracts import _scheduled_contract

from .helpers import project_config, write_json
from .test_active_learning import policy
from .test_scheduled_benchmark import library as benchmark_library
from .test_scheduled_dft import FakeTemplateLibrary, write_site


ROOT = Path(__file__).resolve().parents[1]
CLUSTER_RUNNER = ROOT / "mlipflow" / "plugins" / "active_learning" / "committee_inference_cluster.py"
ACTIVE_SCIENCE = ROOT / "mlipflow" / "plugins" / "active_learning" / "science.py"
def evaluation_dataset() -> dict:
    samples = []
    calibration_ids = []
    for index in range(4):
        sample_id = f"cal-{index}"
        calibration_ids.append(sample_id)
        samples.append(
            {
                "sample_id": sample_id,
                "split": "calibration",
                "species": ["Li"],
                "positions": [[0.0, 0.0, float(index)]],
                "cell": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
                "pbc": True,
                "reference_forces": [[0.05, 0.0, 0.0]],
            }
        )
    for index in range(2):
        sample_id = f"candidate-{index}"
        samples.append(
            {
                "sample_id": sample_id,
                "split": "candidate",
                "species": ["Li"],
                "positions": [[0.0, 0.0, 0.1 * index]],
                "cell": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
                "pbc": True,
                "condition_id": "400k",
                "condition": {"temperature_k": 400, "ensemble": "NVT"},
                "replica": "r0",
                "frame_index": index,
                "structure_path": f"structures/candidate-{index}.vasp",
                "near_duplicate_group": f"near-{index}",
                "physical_validity": {"valid": True, "severe": False, "reasons": []},
            }
        )
    return {
        "schema_version": 1,
        "contract": "mlipflow/active-learning-evaluation-dataset",
        "units": {"energy": "eV", "force": "eV/angstrom"},
        "dataset_split": {
            "dataset_id": "scheduled-active-test",
            "split_id": "scheduled-active-test-split",
            "train_ids": ["train-1"],
            "validation_ids": ["validation-1"],
            "calibration_ids": calibration_ids,
            "audit_ids": ["audit-1"],
        },
        "samples": samples,
    }


def model_index() -> dict:
    return {
        "schema_version": 1,
        "contract": "mlipflow/active-learning-committee-model-index",
        "strategy": {"mode": "single-model-committee", "primary_model": "model-a"},
        "models": [
            {
                "model_id": "model-a",
                "model_family": "chgnet",
                "framework": "chgnet",
                "supports_stress": True,
                "members": [
                    {
                        "member_id": "member-11",
                        "seed": 11,
                        "relative_path": "chgnet/model-seed-11.pt",
                        "kind": "file",
                    },
                    {
                        "member_id": "member-29",
                        "seed": 29,
                        "relative_path": "chgnet/model-seed-29.pt",
                        "kind": "file",
                    },
                ],
            }
        ],
    }


def active_library() -> FakeTemplateLibrary:
    templates = dict(benchmark_library().templates)
    templates["active-learning-committee-canonical/run.sh"] = """#!/bin/bash
# project={{PROJECT_ID}} node={{NODE_ID}} attempt={{ATTEMPT}}
# cpus={{CPUS}} gpus={{GPUS}}
# inputs={{INPUT_DIR}} outputs={{OUTPUT_DIR}}
cd {{RUN_DIR}}
"""
    return FakeTemplateLibrary(templates)


class ScheduledActiveLearningTests(unittest.TestCase):
    def test_core_accepts_committee_runtime_staging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected_policy = policy("single")
            write_json(root / "policy.json", selected_policy)
            write_json(root / "evaluation-dataset.json", evaluation_dataset())
            write_json(root / "committee-model-index.json", model_index())
            node = {
                "id": "committee-evaluate",
                "uses": "active-learning",
                "mode": "execute",
                "backend": "ssh-slurm",
                "backend_profile": "cluster-a",
                "inputs": {
                    "policy": "policy.json",
                    "committee_model_index": "committee-model-index.json",
                    "evaluation_dataset": "evaluation-dataset.json",
                },
                "parameters": {"operation": "committee-evaluate", "device": "cpu"},
                "resources": {
                    "cpus": 1,
                    "gpus": 0,
                    "memory": "2G",
                    "walltime": "00:05:00",
                },
            }
            write_json(root / "project.yaml", project_config([node]))
            site = write_site(root)
            initialize(root)
            project = load_project(root)
            plan = make_run_plan(
                project, "committee-evaluate", site, active_library()
            )
            adapter = plan["adapter_plan"]
            self.assertEqual("READY", adapter["status"], adapter.get("diagnostics"))
            self.assertTrue(plan["approval_required"])
            self.assertEqual(2, adapter["approval_summary"]["member_model_count"])
            contract = _scheduled_contract(
                project,
                "active-learning",
                plan,
                node_id="committee-evaluate",
                attempt=1,
            )
            staged = {item["remote_name"] for item in contract["staged_files"]}
            self.assertTrue(
                {
                    "committee-model-index.json",
                    "evaluation-dataset.json",
                    "committee_inference_cluster.py",
                    "model_runtime.py",
                    "active_learning_science.py",
                }.issubset(staged)
            )
            fetched = {item["remote_name"] for item in contract["fetch_outputs"]}
            self.assertTrue(
                {
                    "committee-predictions.json",
                    "committee-evaluation.json",
                    "cluster-active-learning-report.json",
                    "completion.json",
                }.issubset(fetched)
            )

    def test_cluster_runner_generates_fresh_predictions_and_calibrated_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output_dir = root / "output" / "active-learning"
            model_root = root / "models"
            input_dir.mkdir()
            (model_root / "chgnet").mkdir(parents=True)
            first = model_root / "chgnet" / "model-seed-11.pt"
            second = model_root / "chgnet" / "model-seed-29.pt"
            first.write_text("0.1", encoding="utf-8")
            second.write_text("-0.1", encoding="utf-8")
            write_json(input_dir / "policy.json", policy("single"))
            write_json(input_dir / "evaluation-dataset.json", evaluation_dataset())
            write_json(
                input_dir / "committee-model-index.json",
                model_index(),
            )
            write_json(
                input_dir / "project.yaml",
                project_config(
                    [
                        {
                            "id": "committee-evaluate",
                            "uses": "active-learning",
                            "parameters": {
                                "operation": "committee-evaluate",
                                "device": "cpu",
                            },
                        }
                    ]
                ),
            )
            shutil.copy2(ACTIVE_SCIENCE, input_dir / "active_learning_science.py")
            (input_dir / "model_runtime.py").write_text(
                """
from pathlib import Path
MODEL_FAMILY_FRAMEWORKS = {"chgnet": "chgnet"}
class Predictor:
    prediction_units = {"energy": "eV", "force": "eV/angstrom"}
    def __init__(self, path):
        self.offset = float(Path(path).read_text())
    def predict(self, sample):
        return {"energy": self.offset, "force": [[self.offset, 0.0, 0.0] for _ in sample["species"]]}
def load_inference_predictor(path, family, device):
    return Predictor(path)
""".lstrip(),
                encoding="utf-8",
            )
            report = root / "output" / "cluster-active-learning-report.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(CLUSTER_RUNNER),
                    "--project",
                    str(input_dir / "project.yaml"),
                    "--node-id",
                    "committee-evaluate",
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                    "--report",
                    str(report),
                    "--model-root",
                    str(model_root),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, completed.returncode, completed.stderr + report.read_text())
            evaluation = json.loads((output_dir / "committee-evaluation.json").read_text())
            predictions = json.loads((output_dir / "committee-predictions.json").read_text())
            cluster_report = json.loads(report.read_text())
            self.assertEqual("READY", evaluation["evaluation_status"])
            self.assertEqual("FRESH_ACTIVE_LEARNING_ROUND", evaluation["validation_claim"])
            self.assertEqual([11, 29], [item["seed"] for item in predictions["models"][0]["members"]])
            self.assertEqual(2, len(evaluation["candidates"]))
            self.assertEqual("OK", cluster_report["status"])
            self.assertEqual(2, len(cluster_report["models"]))
            self.assertEqual(
                ["chgnet/model-seed-11.pt", "chgnet/model-seed-29.pt"],
                [item["relative_path"] for item in cluster_report["models"]],
            )


if __name__ == "__main__":
    unittest.main()
