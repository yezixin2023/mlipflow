"""Contract tests for scheduled DeepMD fresh training over ssh-slurm.

These exercise READY plans through to a final MLIPFlow state: a BLOCKED plan
proves nothing about staging, fetching, or the checks that decide whether a
training node may reach OK.  The cluster is simulated by writing exactly the
layout the bundled training runner is contracted to produce.

The point of the checks under test is that a COMPLETED scheduler job is not
evidence of training.  Reaching the requested step count, finite losses, a
parseable learning rate, a real checkpoint, and the requested dataset record all
have to hold independently.
"""

from __future__ import annotations

from tests.helpers import load_module
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from mlipflow.backends import ExecutionResult
from mlipflow.config import load_project
from mlipflow.services import advance, initialize, make_advance_plan, make_run_plan, run_node
from mlipflow.services.contracts import _scheduled_contract

from .helpers import project_config, write_json
from .test_scheduled_dft import PLUGINS, FakeTemplateLibrary, write_site


DEEPMD_RUN_TEMPLATE = """#!/bin/bash
# {{PROJECT_ID}} attempt {{ATTEMPT}}
# inputs={{INPUT_DIR}}
# outputs={{OUTPUT_DIR}}
cd {{RUN_DIR}}
"""
CURVE_HEADER = (
    "#  step      rmse_val    rmse_trn    rmse_e_val  rmse_e_trn"
    "    rmse_f_val  rmse_f_trn         lr"
)
CURVE_NOTE = "# If there is no available reference data, rmse_*_{val,trn} will print nan"


def deepmd_config(
    *,
    numb_steps: int = 500,
    disp_freq: int = 100,
    save_freq: int = 500,
    seed: int = 10,
    systems: list[str] | None = None,
    validation: list[str] | None = None,
    extra_training: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = {
        "model": {
            "type_map": ["Li", "P", "S"],
            "descriptor": {
                "type": "se_e2_a",
                "sel": "auto",
                "rcut_smth": 0.5,
                "rcut": 6.0,
                "neuron": [50, 100, 150],
                "axis_neuron": 32,
                "resnet_dt": False,
                "type_one_side": True,
                "precision": "float64",
                "seed": 1,
            },
            "fitting_net": {
                "neuron": [384, 384, 384],
                "resnet_dt": True,
                "precision": "float64",
                "seed": 1,
            },
        },
        "learning_rate": {
            "type": "exp",
            "decay_steps": 5000,
            "start_lr": 0.001,
            "stop_lr": 3.51e-08,
        },
        "loss": {
            "type": "ener",
            "start_pref_e": 0.02,
            "limit_pref_e": 1,
            "start_pref_f": 1000,
            "limit_pref_f": 1,
            "start_pref_v": 0,
            "limit_pref_v": 0,
        },
        "training": {
            "training_data": {
                "systems": systems if systems is not None else ["data/train/sys-1", "data/train/sys-2"],
                "batch_size": "auto",
            },
            "validation_data": {
                "systems": validation if validation is not None else ["data/test/sys-1"],
                "batch_size": "auto",
                "numb_btch": 32,
            },
            "numb_steps": numb_steps,
            "seed": seed,
            "disp_file": "lcurve.out",
            "disp_freq": disp_freq,
            "save_freq": save_freq,
        },
    }
    if extra_training:
        config["training"].update(extra_training)
    return config


def build_project(
    root: Path,
    *,
    config: dict[str, Any] | None = None,
    dataset: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
) -> Path:
    """Write a portable project whose only dataset reference is a logical id."""

    config_path = root / "inputs" / "deepmd-input.json"
    dataset_path = root / "inputs" / "dataset.json"
    write_json(config_path, config if config is not None else deepmd_config())
    write_json(
        dataset_path,
        dataset
        if dataset is not None
        else {
            "schema_version": 1,
            "dataset_id": "demo-set",
            "systems": {"training": 2, "validation": 1},
        },
    )
    node_parameters = {
        "framework": "deepmd",
        "operation": "train",
        "validation_profile": "deepmd-curve",
        "seed": 10,
        "device": "cpu",
        "precision": "float64",
        "result_manifest": "mlip-training-result.json",
    }
    node_parameters.update(parameters or {})
    node_inputs = {
        "training_config": "inputs/deepmd-input.json",
        "dataset_reference": "inputs/dataset.json",
    }
    node_inputs.update(inputs or {})
    write_json(
        root / "project.yaml",
        project_config(
            [
                {
                    "id": "train-deepmd",
                    "uses": "mlip-training",
                    "mode": "execute",
                    "backend": "ssh-slurm",
                    "backend_profile": "cluster-a",
                    "inputs": node_inputs,
                    "parameters": node_parameters,
                    "resources": {
                        "cpus": 16,
                        "gpus": 0,
                        "memory": "64G",
                        "walltime": "00:40:00",
                    },
                }
            ]
        ),
    )
    return write_site(root)


def curve_text(
    steps: list[int], *, lr: float = 1.0e-3, bad: dict[int, str] | None = None
) -> str:
    lines = [CURVE_HEADER, CURVE_NOTE]
    for index, step in enumerate(steps):
        value = 2.0 + index
        row = [
            f"{step:>7}",
            f"{value:.2e}",
            f"{value + 0.1:.2e}",
            f"{value / 100:.2e}",
            f"{value / 90:.2e}",
            f"{value / 30:.2e}",
            f"{value / 29:.2e}",
            (bad or {}).get(step, f"{lr:.1e}"),
        ]
        lines.append("   ".join(row))
    return "\n".join(lines) + "\n"


def library() -> FakeTemplateLibrary:
    templates = dict(FakeTemplateLibrary().templates)
    templates["mlip-deepmd/run.sh"] = DEEPMD_RUN_TEMPLATE
    return FakeTemplateLibrary(templates)


class TrainingLifecycle:
    """Drive one scheduled training node from plan to final state."""

    def __init__(self, root: Path, **project_kwargs: Any):
        self.root = root
        self.site = build_project(root, **project_kwargs)
        self.remote = root / "fake-remote"
        self.staged: dict[str, str] = {}
        initialize(root)
        self.project = load_project(root)

    def plan(self) -> dict[str, Any]:
        return make_run_plan(self.project, "train-deepmd", self.site, library())

    def submit(self) -> dict[str, Any]:
        plan = self.plan()

        def stage(remote_dir, files):
            for source, relative in files:
                self.staged[relative] = str(source)
            return remote_dir

        with patch(
            "mlipflow.backends.SshSlurmBackend.stage_workspace", side_effect=stage
        ), patch(
            "mlipflow.backends.SshSlurmBackend.submit",
            return_value=ExecutionResult(0, "Submitted batch job 71\n", "", "71"),
        ):
            run_node(
                self.project,
                "train-deepmd",
                True,
                self.site,
                library(),
            )
        return plan

    def write_remote_outputs(
        self,
        *,
        steps: list[int] | None = None,
        report: dict[str, Any] | None = None,
        log: str | None = None,
        curve: str | None = None,
        skip: tuple[str, ...] = (),
        completion_exit: int = 0,
        checkpoint_index: bytes = b"index-bytes",
    ) -> None:
        output = self.remote / "output"
        logs = self.remote / "logs"
        output.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        steps = steps if steps is not None else [0, 100, 200, 300, 400, 500]
        payload = {
            "schema_version": 1,
            "framework": "deepmd",
            "framework_backend": "tensorflow",
            "framework_version": "DeePMD-kit v3.0.0b1",
            "template_family": "mlip-deepmd",
            "exit_code": 0,
            "dataset_id": "demo-set",
            "host": "node-1",
            "threads": {"omp_num_threads": "16"},
            "checkpoint_files": [{"name": "model.ckpt-500.index"}],
        }
        if report is not None:
            payload.update(report)
        write_json(output / "training-result.json", {
            "schema_version": 1, "plugin_id": "mlip-training", "status": "OK",
            "framework": "deepmd", "operation": "train", "seed": 10,
            "device": "cpu", "precision": "float64",
            "framework_version": payload["framework_version"],
            "model_artifact": {"path": "model-artifact"}, "metrics": {},
        })
        (output / "model-artifact").write_bytes(b"fixture model")
        write_json(output / "cluster-run-report.json", {
            "schema_version": 1, "status": "OK", "return_code": 0,
            "framework": "deepmd", "operation": "train",
            "dataset": {"id": "demo-set", "relative_path": "demo-set",
                        "resolved_path": "/site/data/demo-set"},
        })
        if "training-report.json" not in skip:
            write_json(output / "training-report.json", payload)
        if "lcurve.out" not in skip:
            (output / "lcurve.out").write_text(
                curve if curve is not None else curve_text(steps), encoding="utf-8"
            )
        if "checkpoint" not in skip:
            (output / "checkpoint").write_text(
                'model_checkpoint_path: "model.ckpt-500"\n', encoding="utf-8"
            )
        if "model.ckpt.index" not in skip:
            (output / "model.ckpt.index").write_bytes(checkpoint_index)
        if "train.stderr" not in skip:
            (output / "train.stderr").write_text(
                log
                if log is not None
                else "DEEPMD INFO batch 500\nDEEPMD INFO finished training\n",
                encoding="utf-8",
            )
        (logs / "train.stdout").write_text("", encoding="utf-8")
        write_json(
            self.remote / "completion.json",
            {
                "schema_version": 1,
                "status": "COMPLETED" if completion_exit == 0 else "FAILED",
                "exit_code": completion_exit,
                "project_id": self.project.project_id,
                "node_id": "train-deepmd",
                "attempt": 1,
            },
        )

    def _hooks(self):
        def inspect(_self, _cwd, remote_path):
            path = self.remote / remote_path
            if not path.is_file():
                return {"path": remote_path, "exists": False}
            return {
                "path": remote_path,
                "exists": True,
                "size_bytes": path.stat().st_size,
            }

        def fetch(_self, _cwd, remote_path, destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.remote / remote_path, destination)
            return destination

        return inspect, fetch

    def finish(self) -> dict[str, Any]:
        inspect, fetch = self._hooks()
        with patch(
            "mlipflow.backends.SshSlurmBackend.status",
            return_value={"state": "COMPLETED", "detail": None, "source": "fake"},
        ), patch(
            "mlipflow.backends.SshSlurmBackend.inspect_file",
            autospec=True,
            side_effect=inspect,
        ):
            make_advance_plan(self.project)
        with patch(
            "mlipflow.backends.SshSlurmBackend.status",
            return_value={"state": "COMPLETED", "detail": None, "source": "fake"},
        ), patch(
            "mlipflow.backends.SshSlurmBackend.inspect_file",
            autospec=True,
            side_effect=inspect,
        ), patch(
            "mlipflow.backends.SshSlurmBackend.fetch_from",
            autospec=True,
            side_effect=fetch,
        ):
            return advance(self.project)

    def run(self, **overrides: Any) -> dict[str, Any]:
        self.submit()
        self.write_remote_outputs(**overrides)
        return self.finish()

    @property
    def attempt(self) -> Path:
        return self.root / ".mlipflow" / "runs" / "train-deepmd" / "attempt-1"

    def result(self) -> dict[str, Any]:
        return json.loads(
            (self.attempt / "deepmd-curve-result.json").read_text(encoding="utf-8")
        )


class TemporaryProjectTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def state(self, outcome: dict[str, Any]) -> str:
        return str(outcome["changed"][0]["state"])


class ScheduledTrainingPlanTests(TemporaryProjectTest):
    def plan(self, **kwargs: Any) -> dict[str, Any]:
        site = build_project(self.root, **kwargs)
        initialize(self.root)
        project = load_project(self.root)
        return make_run_plan(project, "train-deepmd", site, library())

    def test_well_formed_node_produces_a_ready_plan(self) -> None:
        plan = self.plan()
        adapter_plan = plan["adapter_plan"]
        self.assertEqual(adapter_plan["status"], "READY")
        self.assertTrue(adapter_plan["executable"])
        self.assertEqual(
            adapter_plan["scheduled_execution"]["template_family"], "mlip-deepmd"
        )
        self.assertEqual(adapter_plan["scheduled_execution"]["schema_version"], 3)
        self.assertEqual(
            adapter_plan["scheduled_execution"]["execution_model"], "single-python"
        )
        self.assertEqual(
            adapter_plan["approval_summary"]["cpus_meaning"], "threads-per-process"
        )
        hpc = plan["hpc_execution"]
        self.assertEqual(hpc["execution_model"], "single-python")
        self.assertEqual(
            hpc["template_paths"]["submit.sbatch"],
            "slurm/single-python/cpu.sbatch",
        )
        self.assertIn(
            "#SBATCH --ntasks=1",
            hpc["rendered_scripts"]["submit.sbatch"],
        )
        self.assertIn(
            "#SBATCH --cpus-per-task=16",
            hpc["rendered_scripts"]["submit.sbatch"],
        )

    def test_plan_stages_config_references_and_bundled_runner(self) -> None:
        staged = self.plan()["adapter_plan"]["scheduled_execution"]["staged_files"]
        names = {item["remote_name"] for item in staged}
        self.assertTrue({"training-config.json", "dataset-reference.json", "training_cluster.py"} <= names)
        for item in staged:
            self.assertTrue(Path(item["source"]).is_file())

    def test_fetch_outputs_declare_required_artifacts(self) -> None:
        outputs = self.plan()["adapter_plan"]["scheduled_execution"]["fetch_outputs"]
        by_name = {item["remote_name"]: item for item in outputs}
        self.assertTrue({"checkpoint", "lcurve.out", "model.ckpt.index", "train.stderr",
                         "training-report.json", "training-result.json", "model-artifact"} <= set(by_name))
        self.assertTrue(by_name["lcurve.out"]["required"])
        self.assertTrue(by_name["training-report.json"]["required"])
        self.assertFalse(by_name["training.stdout.log"]["required"])
        self.assertEqual(by_name["train.stderr"]["remote_path"], "output/train.stderr")

    def test_plan_records_trajectory_relevant_parameters(self) -> None:
        calculation = self.plan()["adapter_plan"]["training_calculation"]
        self.assertEqual(calculation["training"]["numb_steps"], 500)
        self.assertEqual(calculation["training"]["disp_freq"], 100)
        self.assertEqual(calculation["training"]["seed"], 10)
        self.assertEqual(calculation["descriptor"]["seed"], 1)
        self.assertEqual(calculation["fitting_net"]["seed"], 1)
        self.assertEqual(calculation["learning_rate"]["start_lr"], 0.001)
        self.assertEqual(calculation["loss"]["start_pref_f"], 1000)
        self.assertEqual(calculation["system_counts"], {"training": 2, "validation": 1})

    def test_system_order_is_recorded_directly(self) -> None:
        first = self.plan()["adapter_plan"]["training_calculation"]
        shutil.rmtree(self.root)
        self.root.mkdir()
        reordered = deepmd_config(systems=["data/train/sys-2", "data/train/sys-1"])
        second = self.plan(config=reordered)["adapter_plan"]["training_calculation"]
        self.assertEqual(first["system_counts"], second["system_counts"])
        self.assertNotEqual(first["systems"], second["systems"])

    def test_core_contract_accepts_the_produced_scheduled_execution(self) -> None:
        site = build_project(self.root)
        initialize(self.root)
        project = load_project(self.root)
        plan = make_run_plan(project, "train-deepmd", site, library())
        contract = _scheduled_contract(
            project, "mlip-training", plan, node_id="train-deepmd", attempt=1
        )
        self.assertEqual(contract["template_family"], "mlip-deepmd")
        self.assertEqual(contract["schema_version"], 3)
        self.assertEqual(contract["execution_model"], "single-python")
        self.assertTrue(any(item["remote_name"] == "training_cluster.py" for item in contract["staged_files"]))
        fetched = {item["remote_name"] for item in contract["fetch_outputs"]}
        self.assertIn("completion.json", fetched)
        self.assertIn("lcurve.out", fetched)


class ScheduledTrainingRefusalTests(TemporaryProjectTest):
    def blocked(self, **kwargs: Any) -> dict[str, Any]:
        site = build_project(self.root, **kwargs)
        initialize(self.root)
        project = load_project(self.root)
        plan = make_run_plan(project, "train-deepmd", site, library())
        adapter_plan = plan["adapter_plan"]
        self.assertEqual(adapter_plan["status"], "BLOCKED")
        return adapter_plan

    def codes(self, adapter_plan: dict[str, Any]) -> set[str]:
        return {item["code"] for item in adapter_plan["diagnostics"]}

    def test_absolute_system_path_is_refused(self) -> None:
        plan = self.blocked(config=deepmd_config(systems=["/cluster/scratch/data/sys-1"]))
        self.assertIn("training.config_contract", self.codes(plan))

    def test_parent_traversal_in_system_path_is_refused(self) -> None:
        plan = self.blocked(config=deepmd_config(systems=["data/../../etc/sys-1"]))
        self.assertIn("training.config_contract", self.codes(plan))

    def test_system_outside_the_data_prefix_is_refused(self) -> None:
        plan = self.blocked(config=deepmd_config(systems=["elsewhere/sys-1"]))
        self.assertIn("training.config_contract", self.codes(plan))

    def test_restart_shaped_key_is_refused_as_not_fresh_training(self) -> None:
        plan = self.blocked(
            config=deepmd_config(extra_training={"init_model": "model.ckpt"})
        )
        self.assertIn("training.config_contract", self.codes(plan))
        self.assertTrue(
            any("fresh run" in item["message"] for item in plan["diagnostics"])
        )

    def test_missing_descriptor_seed_is_refused(self) -> None:
        config = deepmd_config()
        del config["model"]["descriptor"]["seed"]
        self.assertIn("training.config_contract", self.codes(self.blocked(config=config)))

    def test_disp_freq_that_does_not_divide_numb_steps_is_refused(self) -> None:
        plan = self.blocked(config=deepmd_config(numb_steps=500, disp_freq=300))
        self.assertIn("training.config_contract", self.codes(plan))

    def test_production_scale_step_count_is_refused_not_truncated(self) -> None:
        plan = self.blocked(config=deepmd_config(numb_steps=300000, disp_freq=100))
        self.assertIn("training.config_contract", self.codes(plan))

    def test_seed_disagreement_between_plan_and_config_is_refused(self) -> None:
        plan = self.blocked(parameters={"seed": 11})
        self.assertIn("training.seed_mismatch", self.codes(plan))

    def test_declared_system_count_must_match_the_config(self) -> None:
        plan = self.blocked(
            dataset={
                "schema_version": 1,
                "dataset_id": "demo-set",
                "systems": {"training": 7, "validation": 1},
            }
        )
        self.assertIn("training.system_count_training", self.codes(plan))

    def test_dataset_id_with_a_path_separator_is_refused(self) -> None:
        plan = self.blocked(
            dataset={
                "schema_version": 1,
                "dataset_id": "../secrets",
            }
        )
        self.assertIn("training.dataset_contract", self.codes(plan))

    def test_local_wrapper_inputs_are_refused_on_a_scheduled_node(self) -> None:
        plan = self.blocked(inputs={"script": "inputs/wrapper.py"})
        self.assertIn("training.scheduled_inputs", self.codes(plan))


def training_module():
    module = load_module(PLUGINS / 'mlip_training' / 'deepmd_curve.py', 'scheduled_training_adapter_under_test')
    return module


class LearningCurveParsingTests(unittest.TestCase):
    def parse(self, text: str):
        _parse_lcurve = training_module()._parse_lcurve

        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "lcurve.out"
            path.write_text(text, encoding="utf-8")
            return _parse_lcurve(path)

    def test_columns_come_from_the_header_not_from_an_assumption(self) -> None:
        text = (
            "#  step  rmse_trn  rmse_e_trn  rmse_f_trn  rmse_v_trn  lr\n"
            "      0  1.00e+00  2.00e-01  3.00e-01  4.00e-01  1.0e-03\n"
        )
        curve, error = self.parse(text)
        self.assertIsNone(error)
        assert curve is not None
        self.assertEqual(
            curve["columns"],
            ["step", "rmse_trn", "rmse_e_trn", "rmse_f_trn", "rmse_v_trn", "lr"],
        )
        self.assertEqual(curve["records"][0]["rmse_v_trn"], 0.4)

    def test_a_curve_without_virial_columns_parses(self) -> None:
        curve, error = self.parse(curve_text([0, 100]))
        self.assertIsNone(error)
        assert curve is not None
        self.assertNotIn("rmse_v_trn", curve["columns"])
        self.assertEqual(curve["steps"], [0, 100])

    def test_non_finite_loss_is_refused(self) -> None:
        text = (
            "#  step  rmse_trn  lr\n"
            "      0  nan  1.0e-03\n"
        )
        curve, error = self.parse(text)
        self.assertIsNone(curve)
        assert error is not None
        self.assertIn("not finite", error)

    def test_column_count_mismatch_is_refused(self) -> None:
        text = "#  step  rmse_trn  lr\n      0  1.0e+00\n"
        curve, error = self.parse(text)
        self.assertIsNone(curve)
        assert error is not None
        self.assertIn("columns", error)

    def test_non_monotonic_steps_are_refused(self) -> None:
        text = (
            "#  step  rmse_trn  lr\n"
            "    100  1.0e+00  1.0e-03\n"
            "      0  1.0e+00  1.0e-03\n"
        )
        curve, error = self.parse(text)
        self.assertIsNone(curve)
        assert error is not None
        self.assertIn("increasing", error)


class ScheduledTrainingCompletionTests(TemporaryProjectTest):
    def test_complete_run_reaches_ok(self) -> None:
        lifecycle = TrainingLifecycle(self.root)
        outcome = lifecycle.run()
        self.assertEqual(self.state(outcome), "OK")

    def test_exact_partial_fetch_is_reused_after_interrupted_finalization(self) -> None:
        lifecycle = TrainingLifecycle(self.root)
        lifecycle.submit()
        lifecycle.write_remote_outputs()
        shutil.copy2(
            lifecycle.remote / "output" / "lcurve.out",
            lifecycle.attempt / "lcurve.out",
        )
        outcome = lifecycle.finish()
        self.assertEqual(self.state(outcome), "OK")

    def test_result_records_curve_and_provenance(self) -> None:
        lifecycle = TrainingLifecycle(self.root)
        lifecycle.run()
        result = lifecycle.result()
        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(result["framework"], "deepmd")
        self.assertEqual(result["framework_version"], "DeePMD-kit v3.0.0b1")
        self.assertTrue(result["fresh_training"])
        self.assertEqual(result["completed_steps"], 500)
        self.assertEqual(result["requested_steps"], 500)
        self.assertEqual(result["training_curve"]["record_count"], 6)
        self.assertEqual(
            [record["step"] for record in result["training_curve"]["records"]],
            [0, 100, 200, 300, 400, 500],
        )
        self.assertEqual(result["dataset"]["id"], "demo-set")
        self.assertEqual(result["dataset"]["system_counts"], {"training": 2, "validation": 1})
        self.assertEqual(result["seed"], 10)
        self.assertEqual(result["model"]["type_map"], ["Li", "P", "S"])
        self.assertTrue(result["model_artifacts"])

    def test_short_training_run_fails_even_though_the_scheduler_completed(self) -> None:
        lifecycle = TrainingLifecycle(self.root)
        outcome = lifecycle.run(steps=[0, 100, 200])
        self.assertEqual(self.state(outcome), "FAIL")

    def test_missing_completion_marker_fails(self) -> None:
        lifecycle = TrainingLifecycle(self.root)
        outcome = lifecycle.run(log="DEEPMD INFO batch 500\nKilled\n")
        self.assertEqual(self.state(outcome), "FAIL")

    def test_cluster_report_dataset_id_must_match(self) -> None:
        lifecycle = TrainingLifecycle(self.root)
        outcome = lifecycle.run(report={"dataset_id": "different-set"})
        self.assertEqual(self.state(outcome), "FAIL")

    def test_cluster_template_record_must_match_when_reported(self) -> None:
        lifecycle = TrainingLifecycle(self.root)
        outcome = lifecycle.run(report={"template_family": "different-template"})
        self.assertEqual(self.state(outcome), "FAIL")

    def test_nonzero_training_exit_code_fails(self) -> None:
        lifecycle = TrainingLifecycle(self.root)
        outcome = lifecycle.run(report={"exit_code": 1}, completion_exit=0)
        self.assertEqual(self.state(outcome), "FAIL")

    def test_missing_checkpoint_index_fails(self) -> None:
        lifecycle = TrainingLifecycle(self.root)
        outcome = lifecycle.run(skip=("model.ckpt.index",))
        self.assertEqual(self.state(outcome), "FAIL")

    def test_empty_checkpoint_index_fails(self) -> None:
        lifecycle = TrainingLifecycle(self.root)
        outcome = lifecycle.run(checkpoint_index=b"")
        self.assertEqual(self.state(outcome), "FAIL")

    def test_report_without_a_framework_version_fails(self) -> None:
        lifecycle = TrainingLifecycle(self.root)
        outcome = lifecycle.run(report={"framework_version": ""})
        self.assertEqual(self.state(outcome), "FAIL")

    def test_non_positive_learning_rate_fails(self) -> None:
        lifecycle = TrainingLifecycle(self.root)
        outcome = lifecycle.run(
            curve=curve_text([0, 100, 200, 300, 400, 500], bad={300: "0.0e+00"})
        )
        self.assertEqual(self.state(outcome), "FAIL")

    def test_staged_remote_names_match_the_approved_bundle(self) -> None:
        lifecycle = TrainingLifecycle(self.root)
        expected = {"input/" + item["remote_name"] for item in
                    lifecycle.plan()["adapter_plan"]["scheduled_execution"]["staged_files"]}
        lifecycle.submit()
        self.assertEqual(set(lifecycle.staged), expected | {"run.sh", "submit.sbatch"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
