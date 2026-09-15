"""Documented single-task configurations exercise real public CLI contracts."""
import json
import shutil
from pathlib import Path

import pytest

from .helpers import run_cli

ROOT = Path(__file__).resolve().parents[1]


def test_documented_local_ranking_runs_and_exposes_top_k(tmp_path):
    shutil.copytree(ROOT / "examples/local_ranking", tmp_path / "task")
    base = ["--project", str(tmp_path / "task"), "--format", "json"]
    for command in (["init"], ["run", "rank", "--dry-run"], ["run", "rank"], ["json", "rank"]):
        code, out, err = run_cli([*base, *command])
        assert code == 0, err or out
    node = json.loads(out)["data"]["nodes"][0]
    assert node["state"] == "OK"
    assert node["check"]["status"] == node["collection"]["status"] == "OK"
    assert node["metrics"]["selected_count"] == 2
    assert [item["candidate_id"] for item in node["summary"]["ranked_candidates"]] == ["b", "c"]
    assert node["summary"]["excluded_missing"] == ["missing"]
    result = next(item for item in node["artifacts"] if item["role"] == "ranking-result")
    assert Path(result["path"]).is_file()


def test_high_frequency_projects_can_be_inspected_without_inputs_or_execution():
    for directory, node_id in (("training_all_models", "train-mace"),
                               ("ase_md_cluster", "md-900k-nvt"),
                               ("lammps_mlip_inputs", "run-lammps-mace-gpu"),
                               ("ionic_transport", "transport")):
        code, out, err = run_cli(["--project", str(ROOT / "examples" / directory),
                                  "--format", "json", "inspect", node_id])
        assert code == 0, err
        assert json.loads(out)["data"]["inputs"]


def test_documented_transport_recipe_runs_existing_msd_fixture(tmp_path):
    from .test_pes_transport_adapters import IonicTransportAdapterTests

    fixture = IonicTransportAdapterTests()
    fixture.setUp()
    try:
        shutil.copytree(ROOT / "examples/ionic_transport", tmp_path / "task")
        inputs = tmp_path / "task/inputs"
        inputs.mkdir()
        shutil.copy2(fixture.input_file, inputs / "msd.csv")
    finally:
        fixture.tearDown()
    for command in (["init"], ["run", "transport", "--dry-run"], ["run", "transport"]):
        code, out, err = run_cli(["--project", str(tmp_path / "task"),
                                  "--format", "json", *command])
        assert code == 0, err or out
    node = json.loads(out)["data"]
    assert node["state"] == "OK"
    assert node["check"]["status"] == node["collection"]["status"] == "OK"
    paths = [Path(item["path"]) for item in node["artifacts"]]
    assert all(path.is_file() for path in paths)
    assert {path.name for path in paths} >= {
        "diffusion_results_by_temperature.csv", "arrhenius_summary.json"
    }


@pytest.mark.parametrize("directory,node_id,family", [
    ("training_all_models", "train-mace", "mlip-mace"),
    ("ase_md_cluster", "md-900k-nvt", "ase-md-mace"),
    ("lammps_mlip_inputs", "run-lammps-mace-gpu", "lammps-mace-gpu"),
])
def test_documented_scheduled_recipes_plan_with_site_templates(tmp_path, directory, node_id, family):
    from mlipflow.config import load_project
    from mlipflow.services import make_run_plan
    from .test_scheduled_dft import FakeTemplateLibrary, RUN_TEMPLATE, write_site
    from .test_lammps_prepare import _context

    # Reuse an existing structure fixture; no model inference or dynamics runs.
    (tmp_path / "fixture").mkdir()
    _context(tmp_path / "fixture", "mace")
    task = tmp_path / "task"
    shutil.copytree(ROOT / "examples" / directory, task)
    (task / "inputs").mkdir()
    shutil.copy2(tmp_path / "fixture/inputs/start.extxyz", task / "inputs/start.extxyz")
    if directory == "lammps_mlip_inputs":
        code, out, err = run_cli(["--project", str(task), "--format", "json",
                                  "run", "prepare-lammps-mace"])
        assert code == 0, err or out
        assert json.loads(out)["data"]["state"] == "OK"
    templates = dict(FakeTemplateLibrary().templates)
    templates[f"{family}/run.sh"] = RUN_TEMPLATE
    plan = make_run_plan(load_project(task), node_id, write_site(task), FakeTemplateLibrary(templates))
    assert plan["adapter_plan"]["status"] == "READY", plan["adapter_plan"].get("diagnostics")
    assert plan["hpc_execution"]["resources"]["gpus"] == 1
    assert plan["adapter_plan"]["scheduled_execution"]["template_family"] == family
