from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mlipflow.config import load_project
from mlipflow.errors import ConfigError, StateError
from mlipflow.services.commands import initialize, make_run_plan, run_node
from mlipflow.services.paths import state_path
from mlipflow.state import StateStore

from .helpers import project_config, write_json


class ConfigTests(unittest.TestCase):
    def test_project_relative_state_database_path_is_honoured(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = project_config([])
            config["state"] = {"database_path": ".mlipflow/alternate.sqlite3"}
            write_json(root / "project.yaml", config)
            initialized = initialize(root)
            self.assertEqual(
                str((root / ".mlipflow/alternate.sqlite3").resolve()),
                initialized["state_database"],
            )

    def test_state_database_path_cannot_escape_project(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = project_config([])
            config["state"] = {"database_path": "../outside.sqlite3"}
            write_json(root / "project.yaml", config)
            with self.assertRaisesRegex(ConfigError, "inside the project root"):
                initialize(root)

    def test_invalid_dag_nodes_are_rejected(self) -> None:
        cases = (
            [
                {"id": "a", "uses": "candidate-ranking", "needs": ["b"]},
                {"id": "b", "uses": "candidate-ranking", "needs": ["a"]},
            ],
            [
                {"id": "same", "uses": "candidate-ranking"},
                {"id": "same", "uses": "candidate-ranking"},
            ],
        )
        for nodes in cases:
            with self.subTest(nodes=nodes), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                write_json(root / "project.yaml", project_config(nodes))
                with self.assertRaises(ConfigError):
                    load_project(root)

    def test_node_id_cannot_escape_run_directory(self) -> None:
        for unsafe in ("../outside", "/tmp/outside", "nested/node", ".hidden", "NodeA"):
            with self.subTest(node_id=unsafe), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                write_json(
                    root / "project.yaml",
                    project_config(
                        [{"id": unsafe, "uses": "candidate-ranking"}]
                    ),
                )
                with self.assertRaises(ConfigError):
                    load_project(root)

    def test_project_embedded_backend_profiles_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = project_config([])
            config["backend_profiles"] = {"legacy": {"ssh_profile": "cluster"}}
            write_json(root / "project.yaml", config)
            with self.assertRaisesRegex(ConfigError, "site.yaml"):
                load_project(root)

    def test_hpc_node_rejects_submit_script_and_remote_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for key in ("submit_script", "remote_cwd"):
                node = {
                    "id": "hpc",
                    "uses": "dft-labeling",
                    "backend": "ssh-slurm",
                    "backend_profile": "cluster-a",
                    "parameters": {key: "legacy-value"},
                    "resources": {
                        "cpus": 4,
                        "gpus": 0,
                        "memory": "8G",
                        "walltime": "01:00:00",
                    },
                }
                write_json(root / "project.yaml", project_config([node]))
                with self.subTest(key=key), self.assertRaisesRegex(ConfigError, key):
                    load_project(root)

    def test_hpc_project_cannot_embed_site_owned_partition_routing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = {
                "id": "hpc",
                "uses": "dft-labeling",
                "backend": "ssh-slurm",
                "backend_profile": "cluster-a",
                "resources": {
                    "cpus": 4,
                    "gpus": 0,
                    "memory": "8G",
                    "walltime": "01:00:00",
                },
            }
            for field, value in (
                ("partition", "gpu1"),
                ("partition_candidates", ["gpu1"]),
                ("scheduler", {"partition_candidates": ["gpu1"]}),
                ("ssh_profile", "login"),
            ):
                write_json(
                    root / "project.yaml", project_config([{**base, field: value}])
                )
                with self.subTest(field=field), self.assertRaisesRegex(
                    ConfigError, "site-owned"
                ):
                    load_project(root)

    def test_hpc_resource_and_cluster_fields_are_retained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nodes = [
                {
                    "id": "explicit",
                    "uses": "dft-labeling",
                    "backend": "ssh-slurm",
                    "backend_profile": "cluster-a",
                    "resources": {
                        "cpus": 128,
                        "gpus": 0,
                        "memory": "128G",
                        "walltime": "UNLIMITED",
                    },
                },
                {
                    "id": "automatic",
                    "uses": "dft-labeling",
                    "backend": "ssh-slurm",
                    "resources": {
                        "cpus": 128,
                        "gpus": 0,
                        "memory": "128G",
                        "walltime": "48:00:00",
                    },
                },
            ]
            write_json(root / "project.yaml", project_config(nodes))
            project = load_project(root)
            self.assertEqual(
                "UNLIMITED", project.node("explicit")["resources"]["walltime"]
            )
            self.assertNotIn("backend_profile", project.node("automatic"))

    def test_standalone_slurm_backend_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(
                root / "project.yaml",
                project_config(
                    [{"id": "hpc", "uses": "dft-labeling", "backend": "slurm"}]
                ),
            )
            with self.assertRaisesRegex(ConfigError, "local or ssh-slurm"):
                load_project(root)


class PlanTests(unittest.TestCase):
    def test_empty_workflow_can_bootstrap_nodes_on_later_initialize(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "project.yaml", project_config([]))
            initialize(root)
            write_json(
                root / "project.yaml",
                project_config(
                    [{"id": "x", "uses": "candidate-ranking", "mode": "replay"}]
                ),
            )
            initialize(root)
            project = load_project(root)
            with StateStore(state_path(project), readonly=True) as store:
                self.assertEqual("x", store.latest_step(project.project_id, "x").node_id)

    def test_project_yaml_remains_the_dag_source_of_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(
                root / "result.json",
                {"schema_version": 1, "status": "OK", "artifacts": []},
            )
            config = project_config(
                [
                    {"id": "x", "uses": "candidate-ranking", "mode": "replay"},
                    {
                        "id": "y",
                        "uses": "candidate-ranking",
                        "mode": "replay",
                        "needs": ["x"],
                        "inputs": {"result_manifest": "result.json"},
                    },
                ]
            )
            write_json(root / "project.yaml", config)
            initialize(root)
            config["workflow"]["nodes"][1]["needs"] = []
            write_json(root / "project.yaml", config)
            project = load_project(root)
            result = run_node(project, "y")
            self.assertEqual("OK", result["step"]["state"])

    def test_unrelated_config_changes_do_not_block_existing_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = project_config(
                [{"id": "x", "uses": "candidate-ranking", "mode": "replay"}]
            )
            write_json(root / "project.yaml", config)
            initialize(root)
            config["project"]["description"] = "changed after initialization"
            write_json(root / "project.yaml", config)
            self.assertEqual("x", make_run_plan(load_project(root), "x")["node_id"])

    def test_project_identity_still_owns_its_state_database(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = project_config(
                [{"id": "x", "uses": "candidate-ranking", "mode": "replay"}]
            )
            write_json(root / "project.yaml", config)
            initialize(root)
            config["project"]["id"] = "different-project"
            write_json(root / "project.yaml", config)
            with self.assertRaisesRegex(StateError, "belongs to project"):
                make_run_plan(load_project(root), "x")

    def test_replay_plan_records_the_result_path_not_file_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_path = root / "result.json"
            write_json(result_path, {"schema_version": 1, "artifacts": []})
            node = {
                "id": "x",
                "uses": "candidate-ranking",
                "mode": "replay",
                "inputs": {"result_manifest": "result.json"},
            }
            write_json(root / "project.yaml", project_config([node]))
            project = load_project(root)
            first = make_run_plan(project, "x")
            write_json(
                result_path,
                {"schema_version": 1, "metrics": {"changed": 1}, "artifacts": []},
            )
            second = make_run_plan(project, "x")
            self.assertEqual(first, second)
            self.assertEqual("result.json", second["inputs"]["result_manifest"])


if __name__ == "__main__":
    unittest.main()
