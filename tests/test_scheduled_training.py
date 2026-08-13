"""Contract tests for scheduled DeepMD fresh training over ssh-slurm.

These exercise READY plans through to a final MLIPFlow state: a BLOCKED plan
proves nothing about staging, fetching, or the checks that decide whether a
training node may reach OK.  The cluster is simulated by writing exactly the
layout the site's ``deepmd/run.sh`` template is contracted to produce.

The point of the checks under test is that a COMPLETED scheduler job is not
evidence of training.  Reaching the requested step count, finite losses, a
parseable learning rate, a real checkpoint, and a dataset whose fingerprint was
recomputed on the cluster all have to hold independently.
"""

from __future__ import annotations

import copy
import importlib.util
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
from mlipflow.plugins import discover_plugins

from .helpers import project_config, write_json
from .test_scheduled_dft import PLUGINS, FakeTemplateLibrary, sha256, write_site


DEEPMD_RUN_TEMPLATE = """#!/bin/bash
# {{PROJECT_ID}} attempt {{ATTEMPT}}
# inputs={{INPUT_DIR}}
# outputs={{OUTPUT_DIR}}
cd {{RUN_DIR}}
"""
DATASET_FINGERPRINT = "sha256:" + "a1" * 32
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
            "fingerprint": DATASET_FINGERPRINT,
            "systems": {"training": 2, "validation": 1},
        },
    )
    node_parameters = {
        "framework": "deepmd",
        "operation": "train",
        "seed": 10,
        "device": "cpu",
        "precision": "float64",
        "dataset_fingerprint": DATASET_FINGERPRINT,
        "config_fingerprint": sha256(config_path),
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
                    "uses": "mlip-training@0",
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
    templates["deepmd/run.sh"] = DEEPMD_RUN_TEMPLATE
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
        return make_run_plan(self.project, "train-deepmd", PLUGINS, self.site, library())

    def submit(self) -> dict[str, Any]:
        plan = self.plan()

        def stage(remote_dir, files):
            for source, relative, _ in files:
                self.staged[relative] = str(source)
            return remote_dir

        with patch(
            "mlipflow.services.SshSlurmBackend.stage_workspace", side_effect=stage
        ), patch(
            "mlipflow.services.SshSlurmBackend.submit",
            return_value=ExecutionResult(0, "Submitted batch job 71\n", "", "71"),
        ):
            run_node(
                self.project,
                "train-deepmd",
                PLUGINS,
                plan["plan_digest"],
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
            "template_family": "deepmd",
            "exit_code": 0,
            "dataset_id": "demo-set",
            "dataset_fingerprint": DATASET_FINGERPRINT,
            "config_sha256": sha256(self.root / "inputs" / "deepmd-input.json"),
            "host": "node-1",
            "threads": {"omp_num_threads": "16"},
            "checkpoint_files": [{"name": "model.ckpt-500.index", "size_bytes": 4405}],
        }
        if report is not None:
            payload.update(report)
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
            (logs / "train.stderr").write_text(
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
                "sha256": sha256(path),
            }

        def fetch(_self, _cwd, remote_path, destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.remote / remote_path, destination)
            return destination

        return inspect, fetch

    def finish(self) -> dict[str, Any]:
        inspect, fetch = self._hooks()
        with patch(
            "mlipflow.services.SshSlurmBackend.status",
            return_value={"state": "COMPLETED", "detail": None, "source": "fake"},
        ), patch(
            "mlipflow.services.SshSlurmBackend.inspect_file",
            autospec=True,
            side_effect=inspect,
        ):
            approved = make_advance_plan(self.project, PLUGINS, self.site)
        with patch(
            "mlipflow.services.SshSlurmBackend.status",
            return_value={"state": "COMPLETED", "detail": None, "source": "fake"},
        ), patch(
            "mlipflow.services.SshSlurmBackend.inspect_file",
            autospec=True,
            side_effect=inspect,
        ), patch(
            "mlipflow.services.SshSlurmBackend.fetch_from",
            autospec=True,
            side_effect=fetch,
        ):
            return advance(self.project, approved["plan_digest"], PLUGINS, self.site)

    def run(self, **overrides: Any) -> dict[str, Any]:
        self.submit()
        self.write_remote_outputs(**overrides)
        return self.finish()

    @property
    def attempt(self) -> Path:
        return self.root / ".mlipflow" / "runs" / "train-deepmd" / "attempt-1"

    def result(self) -> dict[str, Any]:
        return json.loads(
            (self.attempt / "mlip-training-result.json").read_text(encoding="utf-8")
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
        return make_run_plan(project, "train-deepmd", PLUGINS, site, library())

    def test_well_formed_node_produces_a_ready_plan(self) -> None:
        plan = self.plan()
        adapter_plan = plan["adapter_plan"]
        self.assertEqual(adapter_plan["status"], "READY")
        self.assertTrue(adapter_plan["executable"])
        self.assertEqual(
            adapter_plan["scheduled_execution"]["template_family"], "deepmd"
        )

    def test_plan_stages_exactly_the_config_and_dataset_reference(self) -> None:
        staged = self.plan()["adapter_plan"]["scheduled_execution"]["staged_files"]
        self.assertEqual(
            [item["remote_name"] for item in staged], ["input.json", "dataset.json"]
        )
        for item in staged:
            self.assertGreater(item["size_bytes"], 0)
            self.assertTrue(str(item["sha256"]).startswith("sha256:"))
            self.assertFalse(item["fetch_allowed"])

    def test_fetch_outputs_declare_bounded_required_artifacts(self) -> None:
        outputs = self.plan()["adapter_plan"]["scheduled_execution"]["fetch_outputs"]
        by_name = {item["remote_name"]: item for item in outputs}
        self.assertEqual(
            sorted(by_name),
            [
                "checkpoint",
                "lcurve.out",
                "model.ckpt.index",
                "train.stderr",
                "train.stdout",
                "training-report.json",
            ],
        )
        self.assertTrue(by_name["lcurve.out"]["required"])
        self.assertTrue(by_name["training-report.json"]["required"])
        self.assertFalse(by_name["train.stdout"]["required"])
        self.assertEqual(by_name["train.stderr"]["remote_path"], "logs/train.stderr")
        for item in outputs:
            self.assertIsInstance(item["max_bytes"], int)
            self.assertGreater(item["max_bytes"], 0)

    def test_plan_pins_the_trajectory_relevant_identity(self) -> None:
        identity = self.plan()["adapter_plan"]["training_identity"]
        self.assertEqual(identity["training"]["numb_steps"], 500)
        self.assertEqual(identity["training"]["disp_freq"], 100)
        self.assertEqual(identity["training"]["seed"], 10)
        self.assertEqual(identity["descriptor"]["seed"], 1)
        self.assertEqual(identity["fitting_net"]["seed"], 1)
        self.assertEqual(identity["learning_rate"]["start_lr"], 0.001)
        self.assertEqual(identity["loss"]["start_pref_f"], 1000)
        self.assertEqual(identity["system_counts"], {"training": 2, "validation": 1})
        self.assertTrue(identity["system_order_fingerprint"].startswith("sha256:"))

    def test_system_order_change_changes_the_pinned_identity(self) -> None:
        first = self.plan()["adapter_plan"]["training_identity"]
        shutil.rmtree(self.root)
        self.root.mkdir()
        reordered = deepmd_config(systems=["data/train/sys-2", "data/train/sys-1"])
        second = self.plan(config=reordered)["adapter_plan"]["training_identity"]
        self.assertEqual(first["system_counts"], second["system_counts"])
        self.assertNotEqual(
            first["system_order_fingerprint"], second["system_order_fingerprint"]
        )

    def test_plan_carries_no_absolute_site_path(self) -> None:
        adapter_plan = self.plan()["adapter_plan"]
        portable = copy.deepcopy(adapter_plan)
        # staged_files hold real local sources by construction; every other field
        # must be free of machine-specific absolute paths.
        portable["scheduled_execution"].pop("staged_files")
        serialized = json.dumps(portable)
        self.assertNotIn("/public/home/", serialized)
        self.assertNotIn(str(self.root), serialized)
        self.assertNotIn("/tmp/", serialized)

    def test_core_contract_accepts_the_produced_scheduled_execution(self) -> None:
        site = build_project(self.root)
        initialize(self.root)
        project = load_project(self.root)
        plan = make_run_plan(project, "train-deepmd", PLUGINS, site, library())
        plugin = discover_plugins(PLUGINS)["mlip-training"]
        contract = _scheduled_contract(
            project, plugin, plan, node_id="train-deepmd", attempt=1
        )
        self.assertEqual(contract["template_family"], "deepmd")
        self.assertEqual(len(contract["staged_files"]), 2)
        fetched = {item["remote_name"] for item in contract["fetch_outputs"]}
        self.assertIn("completion.json", fetched)
        self.assertIn("lcurve.out", fetched)


class ScheduledTrainingRefusalTests(TemporaryProjectTest):
    def blocked(self, **kwargs: Any) -> dict[str, Any]:
        site = build_project(self.root, **kwargs)
        initialize(self.root)
        project = load_project(self.root)
        plan = make_run_plan(project, "train-deepmd", PLUGINS, site, library())
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

    def test_config_fingerprint_mismatch_is_refused(self) -> None:
        plan = self.blocked(parameters={"config_fingerprint": "sha256:" + "0" * 64})
        self.assertIn("training.config_fingerprint_mismatch", self.codes(plan))

    def test_dataset_fingerprint_mismatch_is_refused(self) -> None:
        plan = self.blocked(parameters={"dataset_fingerprint": "sha256:" + "0" * 64})
        self.assertIn("training.dataset_fingerprint_mismatch", self.codes(plan))

    def test_seed_disagreement_between_plan_and_config_is_refused(self) -> None:
        plan = self.blocked(parameters={"seed": 11})
        self.assertIn("training.seed_mismatch", self.codes(plan))

    def test_declared_system_count_must_match_the_config(self) -> None:
        plan = self.blocked(
            dataset={
                "schema_version": 1,
                "dataset_id": "demo-set",
                "fingerprint": DATASET_FINGERPRINT,
                "systems": {"training": 7, "validation": 1},
            }
        )
        self.assertIn("training.system_count_training", self.codes(plan))

    def test_dataset_id_with_a_path_separator_is_refused(self) -> None:
        plan = self.blocked(
            dataset={
                "schema_version": 1,
                "dataset_id": "../secrets",
                "fingerprint": DATASET_FINGERPRINT,
            }
        )
        self.assertIn("training.dataset_contract", self.codes(plan))

    def test_local_wrapper_inputs_are_refused_on_a_scheduled_node(self) -> None:
        plan = self.blocked(inputs={"script": "inputs/wrapper.py"})
        self.assertIn("training.local_only_script", self.codes(plan))


def training_module():
    spec = importlib.util.spec_from_file_location(
        "scheduled_training_adapter_under_test", PLUGINS / "mlip-training" / "adapter.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
        self.assertEqual(result["dataset"]["fingerprint"], DATASET_FINGERPRINT)
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

    def test_cluster_recomputed_dataset_fingerprint_must_match(self) -> None:
        lifecycle = TrainingLifecycle(self.root)
        outcome = lifecycle.run(
            report={"dataset_fingerprint": "sha256:" + "b2" * 32}
        )
        self.assertEqual(self.state(outcome), "FAIL")

    def test_staged_config_hash_must_match_the_approved_plan(self) -> None:
        lifecycle = TrainingLifecycle(self.root)
        outcome = lifecycle.run(report={"config_sha256": "sha256:" + "c3" * 32})
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

    def test_staged_remote_names_are_the_two_declared_inputs(self) -> None:
        lifecycle = TrainingLifecycle(self.root)
        lifecycle.submit()
        self.assertEqual(
            sorted(lifecycle.staged),
            ["input/dataset.json", "input/input.json", "run.sh", "submit.sbatch"],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
