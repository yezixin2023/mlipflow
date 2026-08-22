from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from mlipflow.config import load_project
from mlipflow.errors import ConfigError
from mlipflow.services import (
    advance,
    initialize,
    make_advance_plan,
    make_run_plan,
    query_route,
    query_workflow,
    run_node,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "high_entropy_sulfide"
AIMD_EXAMPLE = ROOT / "examples" / "aimd_reference_validation"
SCHEMAS = ROOT / "schemas"


class HighEntropySulfideExampleTests(unittest.TestCase):
    def test_schema_plugin_coverage_and_elements(self) -> None:
        project_data = json.loads((EXAMPLE / "project.yaml").read_text(encoding="utf-8"))
        registry_data = json.loads((EXAMPLE / "model_registry.yaml").read_text(encoding="utf-8"))
        try:
            from jsonschema import Draft202012Validator
        except ImportError:
            Draft202012Validator = None
        if Draft202012Validator is not None:
            for value, schema_name in (
                (project_data, "project.schema.json"),
                (registry_data, "model-registry.schema.json"),
            ):
                schema = json.loads((SCHEMAS / schema_name).read_text(encoding="utf-8"))
                Draft202012Validator(schema).validate(value)
        uses = {node["uses"] for node in project_data["workflow"]["nodes"]}
        self.assertEqual(
            {
                "high-entropy-structure",
                "pes-sampling",
                "dft-labeling",
                "mlip-training",
                "mlip-benchmark",
                "ionic-transport",
                "candidate-ranking",
                "electrochemical-voltage",
            },
            uses,
        )
        expected = {"Li", "Mn", "Fe", "Ni", "Cu", "Zn", "P", "S"}
        for model in registry_data["models"]:
            self.assertEqual(expected, set(model["elements"]))

    def test_full_replay_dag_and_routes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "example"
            shutil.copytree(EXAMPLE, root)
            initialize(root)
            project = load_project(root)
            for _ in range(20):
                workflow = query_workflow(project)
                for step in workflow["steps"]:
                    if step["state"] == "READY":
                        plan = make_run_plan(project, step["node_id"])
                        self.assertEqual(step["capability"], plan["capability"])
                        run_node(project, step["node_id"], True)
                workflow = query_workflow(project)
                if all(step["state"] == "OK" for step in workflow["steps"]):
                    break
                make_advance_plan(project)
                advance(project)
            completed = query_workflow(project)
            self.assertEqual({"OK": 9}, completed["counts"])
            for step in completed["steps"]:
                record = json.loads(
                    Path(step["manifest_path"]).read_text(encoding="utf-8")
                )
                self.assertEqual("OK", record["state"])
                self.assertEqual(step["capability"], record["capability"])
            elements = {"Li", "Mn", "Fe", "Ni", "Cu", "Zn", "P", "S"}
            transport = query_route(
                project,
                task="ionic-transport",
                elements=elements,
                scenario="synthetic-high-entropy-sulfide-v1",
            )
            voltage = query_route(
                project,
                task="electrochemical-voltage",
                elements=elements,
                scenario="synthetic-high-entropy-sulfide-v1",
            )
            self.assertEqual("deepmd-demo", transport["selected_model"])
            self.assertEqual("chgnet-demo", voltage["selected_model"])

    def test_fixture_remains_small(self) -> None:
        size = sum(path.stat().st_size for path in EXAMPLE.rglob("*") if path.is_file())
        self.assertLess(size, 100_000)

    def test_missing_benchmark_evidence_blocks_routing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "example"
            shutil.copytree(EXAMPLE, root)
            evidence = root / "replay" / "model_benchmark.csv"
            evidence.unlink()
            project = load_project(root)
            with self.assertRaisesRegex(ConfigError, "does not exist"):
                query_route(
                    project,
                    task="ionic-transport",
                    elements={"Li", "Mn", "Fe", "Ni", "Cu", "Zn", "P", "S"},
                    scenario="synthetic-high-entropy-sulfide-v1",
                )


class AimdReferenceValidationExampleTests(unittest.TestCase):
    def test_compact_example_keeps_compute_and_analysis_boundaries_explicit(self) -> None:
        project = load_project(AIMD_EXAMPLE)
        nodes = {node["id"]: node for node in project.nodes}
        self.assertEqual("ssh-slurm", nodes["aimd-600k"]["backend"])
        self.assertEqual("cpu-cluster", nodes["aimd-600k"]["backend_profile"])
        self.assertEqual(0, nodes["aimd-600k"]["resources"]["gpus"])
        self.assertEqual(
            {"model-compute"},
            {
                nodes["deepmd-md-600k"]["backend_profile"],
                nodes["chgnet-md-600k"]["backend_profile"],
            },
        )
        for node_id in ("deepmd-md-600k", "chgnet-md-600k"):
            self.assertEqual("ssh-slurm", nodes[node_id]["backend"])
            self.assertEqual(1, nodes[node_id]["resources"]["gpus"])
        for node_id in ("compare-deepmd", "compare-chgnet", "rank-aimd-agreement"):
            self.assertEqual("local", nodes[node_id]["backend"])
        ranking = nodes["rank-aimd-agreement"]
        self.assertEqual(
            ["deepmd-dpa2", "chgnet"], ranking["parameters"]["expected_models"]
        )
        self.assertEqual(
            ["structural-dynamics", "ionic-transport"],
            ranking["parameters"]["expected_tasks"],
        )


if __name__ == "__main__":
    unittest.main()
