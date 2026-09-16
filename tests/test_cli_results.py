"""CLI contracts needed to run a task and consume its outputs directly."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from mlipipe.backends import ExecutionResult
from mlipipe.services import commands
from .helpers import project_config, run_cli, write_json
from .test_adapter_execution import FixtureAdapter


@pytest.fixture
def task(tmp_path):
    write_json(tmp_path / "project.yaml", project_config([
        {"id": "rank", "uses": "candidate-ranking", "backend": "local"}
    ]))
    return tmp_path


def invoke(task, *args):
    code, out, err = run_cli(["--project", str(task), "--format", "json", *args])
    return code, json.loads(out or err)


class ResultAdapter(FixtureAdapter):
    def check(self, context):
        return {"status": "OK", "metrics": {"checked_count": 3}, "diagnostics": []}

    def collect(self, context):
        path = Path(context["attempt_dir"]) / "ranked.json"
        write_json(path, {"candidate_ids": ["a", "b"]})
        return {"status": "OK", "metrics": {"selected_count": 2},
                "artifacts": [{"role": "ranked-candidates", "path": str(path)}]}


def test_run_plans_once_and_exposes_results_and_saved_status(task):
    adapter = ResultAdapter([sys.executable, "-c", "pass"])
    original_plan = commands.make_run_plan
    with patch("mlipipe.services.commands.load_adapter", return_value=adapter), patch(
        "mlipipe.services.execution.load_adapter", return_value=adapter
    ), patch.object(commands, "make_run_plan", wraps=original_plan) as planner, patch(
        "mlipipe.cli.make_run_plan", wraps=original_plan
    ) as cli_planner:
        code, envelope = invoke(task, "run", "rank")
    assert code == 0
    assert planner.call_count + cli_planner.call_count == 1
    data = envelope["data"]
    assert data["state"] == "OK"
    assert data["metrics"] == {"checked_count": 3, "selected_count": 2}
    assert data["check"]["status"] == "OK"
    assert data["collection"]["status"] == "OK"
    artifact = next(a for a in data["artifacts"] if a["role"] == "ranked-candidates")
    assert Path(artifact["path"]).is_file()
    assert Path(data["manifest_path"]).is_file()
    assert Path(data["logs"]["stdout"]).is_file()
    _, status = invoke(task, "status", "rank")
    saved = status["data"]["nodes"][0]
    for key in ("metrics", "check", "collection", "artifacts", "manifest_path", "logs"):
        assert saved[key] == data[key]


@pytest.mark.parametrize("failure", ["process", "check", "collection"])
def test_synchronous_failure_is_nonzero_but_status_query_succeeds(task, failure):
    adapter = ResultAdapter(["fixture"])
    reason = "failure detail " * 40
    if failure != "process":
        setattr(adapter, failure if failure == "check" else "collect",
                lambda context: {"status": "FAIL", "diagnostics": [{"message": reason}]})
    with patch("mlipipe.services.commands.load_adapter", return_value=adapter), patch(
        "mlipipe.services.execution.load_adapter", return_value=adapter
    ), patch("mlipipe.backends.LocalBackend.run",
             return_value=ExecutionResult(7 if failure == "process" else 0, "", "")):
        code, envelope = invoke(task, "run", "rank")
    assert code == 1
    assert envelope["ok"] is False
    assert envelope["data"]["state"] == "FAIL"
    assert envelope["data"]["returncode"] == (7 if failure == "process" else 0)
    if failure != "process":
        assert reason in envelope["data"]["reason"]
    code, status = invoke(task, "status", "rank")
    assert code == 0 and status["ok"] is True
    assert status["data"]["nodes"][0]["state"] == "FAIL"


def test_scheduled_submission_reports_pending_success(task):
    with patch("mlipipe.cli.run_node", return_value={"step": {
        "node_id": "rank", "state": "PENDING", "attempt": 1,
        "backend": "ssh-slurm", "job_id": "123",
    }}), patch("mlipipe.cli.make_run_plan", return_value={"approval_required": True}):
        code, envelope = invoke(task, "run", "rank", "--approve")
    assert code == 0 and envelope["ok"] is True
    assert envelope["data"]["state"] == "PENDING"
    assert envelope["data"]["job_id"] == "123"


def test_json_dry_run_preserves_exact_argv_cwd_and_outputs(task):
    argv = [sys.executable, "/some/full/path/tool.py", *[f"arg-{i}" for i in range(20)]]

    class PlannedAdapter(FixtureAdapter):
        def plan(self, context):
            return {**super().plan(context), "expected_outputs": ["ranked.json"]}

    with patch("mlipipe.services.commands.load_adapter", return_value=PlannedAdapter(argv)):
        code, envelope = invoke(task, "run", "rank", "--dry-run")
    assert code == 0
    data = envelope["data"]
    assert data["attempt"] == 1
    assert data["adapter_plan"]["argv"] == argv
    assert data["adapter_plan"]["cwd"] == str(task / ".mlipipe/runs/rank/attempt-1")
    assert data["adapter_plan"]["expected_outputs"] == ["ranked.json"]
    assert not (task / ".mlipipe").exists()


def test_scheduled_plan_and_collected_json_preserve_actionable_details(tmp_path):
    from .test_scheduled_training import TrainingLifecycle

    lifecycle = TrainingLifecycle(tmp_path)
    plan = lifecycle.plan()
    with patch("mlipipe.cli.make_run_plan", return_value=plan):
        code, envelope = invoke(tmp_path, "run", "train-deepmd", "--dry-run")
    assert code == 0
    data = envelope["data"]
    assert data["state"] == "READY"
    for key in ("resources", "workspace", "template_paths", "execution_model"):
        assert data["hpc_execution"][key] == plan["hpc_execution"][key]
    assert data["adapter_plan"]["scheduled_execution"] == plan["adapter_plan"]["scheduled_execution"]
    assert "rendered_scripts" not in data["hpc_execution"]

    lifecycle.submit()
    code, envelope = invoke(tmp_path, "status", "train-deepmd")
    assert code == 0 and envelope["ok"] is True
    pending = envelope["data"]["nodes"][0]
    assert pending["state"] == "PENDING" and pending["job_id"] == "71"
    lifecycle.write_remote_outputs()
    inspect, fetch = lifecycle._hooks()
    with patch("mlipipe.backends.SshSlurmBackend.status", return_value={
        "state": "COMPLETED", "detail": None, "source": "fixture"
    }), patch("mlipipe.backends.SshSlurmBackend.inspect_file", autospec=True,
              side_effect=inspect), patch("mlipipe.backends.SshSlurmBackend.fetch_from",
                                         autospec=True, side_effect=fetch):
        code, envelope = invoke(tmp_path, "advance")
    assert code == 0
    node = envelope["data"]["changed"][0]
    assert node["state"] == "OK"
    assert node["check"]["status"] == node["collection"]["status"] == "OK"
    assert node["metrics"]["completed_steps"] == 500
    assert Path(node["manifest_path"]).is_file()
    assert {item["role"] for item in node["artifacts"]} >= {"model", "training-manifest"}
    assert all(Path(item["path"]).is_file() for item in node["artifacts"])
    code, envelope = invoke(tmp_path, "json", "train-deepmd")
    assert code == 0
    saved = envelope["data"]["nodes"][0]
    for key in ("state", "job_id", "artifacts", "metrics", "check", "collection", "manifest_path"):
        assert saved[key] == node[key]
