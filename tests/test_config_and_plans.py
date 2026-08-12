from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlipflow.config import load_project
from mlipflow.errors import ConfigError, StateError
from mlipflow.services import initialize, make_run_plan

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


class PlanTests(unittest.TestCase):
    def test_initialized_node_config_drift_requires_explicit_migration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugins = root / "plugins"
            manifest = plugin_manifest()
            manifest["implementation"]["kind"] = "external-command"
            manifest["execution"]["backends"] = ["local"]
            write_json(plugins / "demo" / "plugin.yaml", manifest)
            node = {
                "id": "x",
                "uses": "demo@1",
                "backend": "local",
                "parameters": {"argv": ["true"]},
            }
            config = project_config([node])
            write_json(root / "project.yaml", config)
            initialize(root)
            config["workflow"]["nodes"][0]["parameters"]["argv"] = ["false"]
            write_json(root / "project.yaml", config)
            with self.assertRaisesRegex(StateError, "configuration differs"):
                make_run_plan(load_project(root), "x", plugins)

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
            self.assertEqual(third["input_fingerprints"]["data"]["fingerprint_mode"], "tree-full")

    def test_input_or_plugin_change_invalidates_digest(self) -> None:
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
            self.assertNotEqual(second["plan_digest"], third["plan_digest"])

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
