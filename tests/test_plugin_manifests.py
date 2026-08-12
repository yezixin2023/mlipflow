"""Contract tests for built-in computational plugin declarations."""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = ROOT / "schemas"
PLUGIN_ROOT = ROOT / "plugins"
EXPECTED_PLUGINS = {
    "high-entropy-structure",
    "pes-sampling",
    "dft-labeling",
    "mlip-training",
    "mlip-benchmark",
    "ionic-transport",
    "composition-screening",
    "electrochemical-voltage",
}
CONTRACT_METHODS = {"validate", "plan", "prepare", "check", "collect", "replay"}
EXACT_EXECUTION_OPERATIONS = {
    "composition-screening": ["rank-candidates"],
    "dft-labeling": ["vasp-prepare", "label"],
    "high-entropy-structure": ["generate-sqs"],
    "mlip-benchmark": ["evaluate-static", "normalize-replay", "normalize-execute"],
    "mlip-training": ["train", "finetune"],
    "pes-sampling": [
        "direct-select",
        "lasp-ssw-execute",
        "lasp-ssw-normalize-replay",
    ],
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise AssertionError(f"{path} must contain a JSON object")
    return value


class SchemaTests(unittest.TestCase):
    def test_five_draft_2020_12_schemas_are_json(self) -> None:
        expected = {
            "project.schema.json",
            "plugin.schema.json",
            "run-manifest.schema.json",
            "model-registry.schema.json",
            "site.schema.json",
        }
        self.assertEqual(expected, {path.name for path in SCHEMA_ROOT.glob("*.json")})
        for name in sorted(expected):
            schema = load_json(SCHEMA_ROOT / name)
            self.assertEqual(
                "https://json-schema.org/draft/2020-12/schema", schema.get("$schema"), name
            )
            self.assertTrue(str(schema.get("$id", "")).startswith("urn:mlipflow:schema:"), name)

    def test_schemas_are_valid_when_jsonschema_is_installed(self) -> None:
        try:
            from jsonschema import Draft202012Validator
        except ModuleNotFoundError:
            self.skipTest("jsonschema is an optional dev dependency")
        for schema_path in sorted(SCHEMA_ROOT.glob("*.json")):
            Draft202012Validator.check_schema(load_json(schema_path))


class PluginManifestTests(unittest.TestCase):
    def manifests(self) -> list[tuple[Path, dict]]:
        return [
            (path, load_json(path)) for path in sorted(PLUGIN_ROOT.glob("*/plugin.yaml"))
        ]

    def test_exact_builtin_plugin_set(self) -> None:
        self.assertEqual(EXPECTED_PLUGINS, {path.parent.name for path, _ in self.manifests()})

    def test_manifests_obey_static_contract(self) -> None:
        required = {
            "schema_version",
            "id",
            "name",
            "version",
            "api_version",
            "description",
            "category",
            "contract",
            "implementation",
            "execution",
            "replay",
            "inputs",
            "parameters",
            "outputs",
            "dependencies",
            "safety",
            "completion",
            "retry",
        }
        for path, manifest in self.manifests():
            with self.subTest(plugin=path.parent.name):
                self.assertFalse(required - set(manifest))
                self.assertEqual(path.parent.name, manifest["id"])
                self.assertEqual(1, manifest["schema_version"])
                self.assertEqual(1, manifest["api_version"])
                self.assertEqual(CONTRACT_METHODS, set(manifest["contract"]["methods"]))

                implementation = manifest["implementation"]
                self.assertIn(implementation["status"], {"adapter-ready", "implemented"})
                self.assertEqual("python-adapter", implementation["kind"])
                self.assertTrue(implementation["limitations"])

                execution = manifest["execution"]
                self.assertIn(execution["mode"], {"external-command", "python-library"})
                expected_backends = (
                    ["local", "ssh-slurm"]
                    if manifest["id"] == "dft-labeling"
                    else ["local"]
                )
                self.assertEqual(expected_backends, execution["backends"])
                self.assertFalse(execution["shell"])
                self.assertFalse(execution["submits_jobs"])
                self.assertTrue(execution["operations"])

                replay = manifest["replay"]
                self.assertTrue(replay["supported"])
                self.assertIn(
                    replay["behavior"], {"reference-only", "parse-existing-artifacts"}
                )
                self.assertIn("application/json", replay["accepted_media_types"])

                self.assertTrue(manifest["parameters"])
                completion = manifest["completion"]
                self.assertEqual("implemented", completion["status"])
                self.assertTrue(completion["checks"])
                self.assertTrue(all(check["implemented"] for check in completion["checks"]))
                retry = manifest["retry"]
                self.assertEqual("delegated-to-core", retry["status"])
                self.assertEqual({"FAIL", "STOPPED"}, set(retry["allowed_from"]))
                self.assertTrue(retry["creates_new_attempt"])
                self.assertFalse(retry["reuses_previous_run_directory"])
                self.assertTrue(retry["requires_approval"])
                self.assertFalse(retry["limit_enforced"])

                adapter_file, separator, object_name = implementation["entrypoint"].partition(":")
                self.assertEqual(":", separator)
                self.assertEqual("Adapter", object_name)
                self.assertEqual(adapter_file, execution["entrypoint"]["path"])
                self.assertTrue((path.parent / adapter_file).is_file())

    def test_manifests_validate_when_jsonschema_is_installed(self) -> None:
        try:
            from jsonschema import Draft202012Validator
        except ModuleNotFoundError:
            self.skipTest("jsonschema is an optional dev dependency")
        validator = Draft202012Validator(load_json(SCHEMA_ROOT / "plugin.schema.json"))
        for path, manifest in self.manifests():
            errors = sorted(validator.iter_errors(manifest), key=lambda error: list(error.path))
            self.assertEqual([], errors, f"{path}: {[error.message for error in errors]}")

    def test_execution_operations_match_adapter_dispatch(self) -> None:
        manifests = {path.parent.name: manifest for path, manifest in self.manifests()}
        for plugin_id, operations in EXACT_EXECUTION_OPERATIONS.items():
            with self.subTest(plugin=plugin_id):
                self.assertEqual(operations, manifests[plugin_id]["execution"]["operations"])

    def test_adapter_planners_never_spawn_or_delete_process_data(self) -> None:
        forbidden_modules = {"subprocess", "os", "shutil"}
        forbidden_calls = {
            "system",
            "popen",
            "Popen",
            "run",
            "call",
            "check_call",
            "unlink",
            "rmtree",
            "copy",
            "move",
        }

        for path, manifest in self.manifests():
            adapter_path = path.parent / manifest["execution"]["entrypoint"]["path"]
            source = adapter_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(adapter_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertFalse({alias.name.split(".", 1)[0] for alias in node.names} & forbidden_modules)
                elif isinstance(node, ast.ImportFrom):
                    self.assertNotIn((node.module or "").split(".", 1)[0], forbidden_modules)
                elif isinstance(node, ast.Call):
                    called = node.func.attr if isinstance(node.func, ast.Attribute) else None
                    if isinstance(node.func, ast.Name):
                        called = node.func.id
                    self.assertNotIn(called, forbidden_calls)

            namespace: dict = {}
            exec(compile(source, str(adapter_path), "exec"), namespace)
            adapter = namespace["Adapter"]()
            blocked = adapter.plan({})
            self.assertEqual("BLOCKED", blocked["status"])
            self.assertFalse(blocked["executable"])

    def test_training_exposes_four_thin_framework_adapters_with_limitations(self) -> None:
        training = load_json(PLUGIN_ROOT / "mlip-training" / "plugin.yaml")
        backends = {item["id"]: item for item in training["computational_backends"]}
        self.assertEqual({"deepmd", "m3gnet", "chgnet", "mace"}, set(backends))
        self.assertTrue(all(item["status"] == "adapter-ready" for item in backends.values()))
        self.assertTrue(all(item["limitations"] for item in backends.values()))


if __name__ == "__main__":
    unittest.main()
