from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlipflow.config import load_project
from mlipflow.errors import ConfigError, StateError
from mlipflow.services import initialize, make_run_plan, run_node, state_path
from mlipflow.state import StateStore

from .helpers import plugin_manifest, project_config, write_json


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

    def test_cycle_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(
                root / "project.yaml",
                project_config(
                    [
                        {"id": "a", "uses": "demo@1", "needs": ["b"]},
                        {"id": "b", "uses": "demo@1", "needs": ["a"]},
                    ]
                ),
            )
            with self.assertRaises(ConfigError):
                load_project(root)

    def test_duplicate_node_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(
                root / "project.yaml",
                project_config(
                    [
                        {"id": "same", "uses": "demo@1"},
                        {"id": "same", "uses": "demo@1"},
                    ]
                ),
            )
            with self.assertRaises(ConfigError):
                load_project(root)

    def test_node_id_cannot_escape_run_directory(self) -> None:
        for unsafe in ("../outside", "/tmp/outside", "nested/node", ".hidden", "NodeA"):
            with self.subTest(node_id=unsafe), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                write_json(
                    root / "project.yaml",
                    project_config([{"id": unsafe, "uses": "demo@1"}]),
                )
                with self.assertRaises(ConfigError):
                    load_project(root)

    def test_project_embedded_backend_profiles_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = project_config([])
            config["backend_profiles"] = {
                "legacy": {"ssh_profile": "cluster"}
            }
            write_json(root / "project.yaml", config)
            with self.assertRaisesRegex(ConfigError, "site.yaml"):
                load_project(root)

    def test_hpc_node_rejects_submit_script_and_remote_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for key in ("submit_script", "remote_cwd"):
                with self.subTest(key=key):
                    node = {
                        "id": "hpc",
                        "uses": "demo@1",
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
                    with self.assertRaisesRegex(ConfigError, key):
                        load_project(root)

    def test_hpc_project_cannot_embed_site_owned_partition_routing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = {
                "id": "hpc",
                "uses": "demo@1",
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
                with self.subTest(field=field):
                    write_json(
                        root / "project.yaml",
                        project_config([{**base, field: value}]),
                    )
                    with self.assertRaisesRegex(ConfigError, "site-owned"):
                        load_project(root)


class PlanTests(unittest.TestCase):
    def test_unrelated_config_changes_do_not_block_existing_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugins = root / "plugins"
            manifest = plugin_manifest()
            manifest["execution"]["backends"] = ["local"]
            write_json(plugins / "demo" / "plugin.yaml", manifest)
            config = project_config(
                [
                    {"id": "x", "uses": "demo@1", "mode": "replay", "parameters": {}},
                    {"id": "y", "uses": "demo@1", "mode": "replay", "parameters": {}},
                ]
            )
            write_json(root / "project.yaml", config)
            initialize(root)
            config["project"]["description"] = "changed after initialization"
            config["workflow"]["nodes"][1]["parameters"] = {"unrelated": True}
            write_json(root / "project.yaml", config)
            plan = make_run_plan(load_project(root), "x", plugins)
            self.assertEqual("x", plan["node_id"])

    def test_project_identity_still_owns_its_state_database(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugins = root / "plugins"
            write_json(plugins / "demo" / "plugin.yaml", plugin_manifest())
            config = project_config(
                [{"id": "x", "uses": "demo@1", "mode": "replay", "parameters": {}}]
            )
            write_json(root / "project.yaml", config)
            initialize(root)

            config["project"]["id"] = "different-project"
            write_json(root / "project.yaml", config)
            with self.assertRaisesRegex(StateError, "belongs to project"):
                make_run_plan(load_project(root), "x", plugins)

    def test_attempt_snapshot_binds_at_execution_and_then_stays_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugins = root / "plugins"
            write_json(plugins / "demo" / "plugin.yaml", plugin_manifest())
            write_json(
                root / "result.json",
                {"schema_version": 1, "status": "OK", "metrics": {}, "artifacts": []},
            )
            config = project_config(
                [
                    {
                        "id": "x",
                        "uses": "demo@1",
                        "mode": "replay",
                        "inputs": {"result_manifest": "result.json"},
                        "parameters": {"revision": 1},
                    }
                ]
            )
            write_json(root / "project.yaml", config)
            initialize(root)

            config["workflow"]["nodes"][0]["parameters"]["revision"] = 2
            write_json(root / "project.yaml", config)
            project = load_project(root)
            run_node(project, "x", plugins)
            with StateStore(state_path(project), readonly=True) as store:
                step = store.latest_step(project.project_id, "x")
                self.assertEqual(2, store.node_snapshot(step.run_id)["parameters"]["revision"])

            config["workflow"]["nodes"][0]["parameters"]["revision"] = 3
            write_json(root / "project.yaml", config)
            with StateStore(state_path(project), readonly=True) as store:
                self.assertEqual(2, store.node_snapshot(step.run_id)["parameters"]["revision"])

    def test_generic_file_and_directory_inputs_are_fingerprinted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugins = root / "plugins"
            manifest = plugin_manifest()
            manifest["implementation"]["status"] = "contract-only"
            manifest["execution"]["backends"] = ["local"]
            write_json(plugins / "demo" / "plugin.yaml", manifest)
            structures = root / "structures.xyz"
            structures.write_text("one\n", encoding="utf-8")
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "part.dat").write_text("alpha\n", encoding="utf-8")
            node = {
                "id": "x",
                "uses": "demo@1",
                "mode": "execute",
                "inputs": {"structures": "structures.xyz", "data": "dataset"},
                "parameters": {"argv": ["true"]},
            }
            write_json(root / "project.yaml", project_config([node]))
            project = load_project(root)
            first = make_run_plan(project, "x", plugins)
            structures.write_text("two\n", encoding="utf-8")
            second = make_run_plan(project, "x", plugins)
            self.assertNotEqual(first["plan_digest"], second["plan_digest"])
            (dataset / "part.dat").write_text("beta\n", encoding="utf-8")
            third = make_run_plan(project, "x", plugins)
            self.assertNotEqual(second["plan_digest"], third["plan_digest"])
            self.assertEqual(third["input_identities"]["data"]["content_mode"], "tree-full")
            self.assertEqual(third["input_identities"]["data"]["locator"], "dataset")
            self.assertEqual(third["input_identities"]["structures"]["locator"], "structures.xyz")

    def test_input_change_invalidates_digest_but_manifest_text_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugins = root / "plugins"
            manifest_path = plugins / "demo" / "plugin.yaml"
            manifest = plugin_manifest()
            manifest["execution"]["backends"] = ["local"]
            write_json(manifest_path, manifest)
            result_path = root / "result.json"
            write_json(result_path, {"schema_version": 1, "artifacts": []})
            node = {
                "id": "x",
                "uses": "demo@1",
                "mode": "replay",
                "inputs": {"result_manifest": "result.json"},
            }
            write_json(root / "project.yaml", project_config([node]))
            project = load_project(root)
            first = make_run_plan(project, "x", plugins)
            write_json(result_path, {"schema_version": 1, "metrics": {"changed": 1}, "artifacts": []})
            second = make_run_plan(project, "x", plugins)
            self.assertNotEqual(first["plan_digest"], second["plan_digest"])
            manifest["description"] = "Changed without a version bump, still changes approval"
            write_json(manifest_path, manifest)
            third = make_run_plan(project, "x", plugins)
            self.assertEqual(second["plan_digest"], third["plan_digest"])
            self.assertNotIn("manifest_digest", third["plugin"])

    def test_slurm_dry_plan_never_calls_scheduler(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugins = root / "plugins"
            manifest = plugin_manifest()
            manifest["implementation"]["kind"] = "external-command"
            manifest["execution"]["backends"] = ["slurm"]
            write_json(plugins / "demo" / "plugin.yaml", manifest)
            write_json(
                root / "project.yaml",
                project_config(
                    [
                        {
                            "id": "hpc",
                            "uses": "demo@1",
                            "backend": "slurm",
                            "parameters": {},
                        }
                    ]
                ),
            )
            with patch("mlipflow.backends.subprocess.run", side_effect=AssertionError("submitted")):
                plan = make_run_plan(load_project(root), "hpc", plugins)
            self.assertEqual(plan["backend"], "slurm")
            self.assertIn("submit", " ".join(plan["warnings"]).lower())


if __name__ == "__main__":
    unittest.main()
