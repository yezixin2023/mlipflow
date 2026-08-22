"""Contract tests for built-in computational plugin declarations."""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from mlipflow.plugins import discover_plugins, load_adapter


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = ROOT / "schemas"
PLUGIN_ROOT = ROOT / "plugins"
EXPECTED_PLUGINS = {
    "active-learning",
    "ase-md",
    "high-entropy-structure",
    "pes-sampling",
    "dft-labeling",
    "lammps-md",
    "mlip-training",
    "mlip-benchmark",
    "ionic-transport",
    "candidate-ranking",
    "electrochemical-voltage",
}
ADAPTER_METHODS = {"validate", "plan", "check", "collect"}
EXACT_EXECUTION_OPERATIONS = {
    "active-learning": ["committee-evaluate", "select-candidates", "assess-round"],
    "ase-md": ["run"],
    "candidate-ranking": ["rank-candidates"],
    "dft-labeling": ["vasp-prepare", "label", "dataset-assemble"],
    "electrochemical-voltage": ["compute-from-energies", "replay-si-table-s11"],
    "high-entropy-structure": ["generate-sqs"],
    "ionic-transport": ["analyze-existing", "md-smoke-and-analyze"],
    "lammps-md": ["lammps-prepare", "execute"],
    "mlip-benchmark": [
        "evaluate-fresh",
        "evaluate-static",
        "normalize-replay",
        "normalize-execute",
    ],
    "mlip-training": ["train", "finetune"],
    "pes-sampling": [
        "direct-select",
        "lasp-input-prepare",
        "merge-structures",
        "lasp-ssw-execute",
        "lasp-ssw-normalize-replay",
    ],
}
EXPECTED_EXECUTION_BACKENDS = {
    "active-learning": ["local", "ssh-slurm"],
    "ase-md": ["ssh-slurm"],
    "candidate-ranking": ["local"],
    "dft-labeling": ["local", "ssh-slurm"],
    "electrochemical-voltage": ["local"],
    "high-entropy-structure": ["local"],
    "ionic-transport": ["local"],
    "lammps-md": ["local", "ssh-slurm"],
    "mlip-benchmark": ["local", "ssh-slurm"],
    "mlip-training": ["local", "ssh-slurm"],
    "pes-sampling": ["local", "ssh-slurm"],
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
        discovered = {path.parent.name for path, _ in self.manifests()}
        self.assertEqual(EXPECTED_PLUGINS, discovered)
        self.assertEqual(EXPECTED_PLUGINS, set(EXACT_EXECUTION_OPERATIONS))
        self.assertEqual(EXPECTED_PLUGINS, set(EXPECTED_EXECUTION_BACKENDS))

    def test_manifests_use_the_reduced_capability_contract(self) -> None:
        required = {
            "schema_version",
            "id",
            "name",
            "version",
            "api_version",
            "description",
            "category",
            "implementation",
            "execution",
            "inputs",
            "parameters",
            "outputs",
            "dependencies",
            "safety",
            "completion",
        }
        removed = {"contract", "replay", "retry"}
        for path, manifest in self.manifests():
            with self.subTest(plugin=path.parent.name):
                self.assertFalse(required - set(manifest))
                self.assertFalse(removed & set(manifest))
                self.assertEqual(path.parent.name, manifest["id"])
                self.assertEqual(1, manifest["schema_version"])
                self.assertEqual(1, manifest["api_version"])

                implementation = manifest["implementation"]
                self.assertEqual(
                    {"kind", "entrypoint", "status", "limitations"},
                    set(implementation),
                )
                self.assertIn(implementation["status"], {"adapter-ready", "implemented"})
                self.assertEqual("python-adapter", implementation["kind"])
                self.assertTrue(implementation["limitations"])

                execution = manifest["execution"]
                self.assertEqual({"mode", "backends", "operations"}, set(execution))
                self.assertIn(execution["mode"], {"external-command", "python-library"})
                self.assertEqual(
                    EXPECTED_EXECUTION_BACKENDS[manifest["id"]], execution["backends"]
                )
                self.assertTrue(execution["operations"])

                self.assertTrue(manifest["parameters"])
                completion = manifest["completion"]
                self.assertEqual({"checks"}, set(completion))
                self.assertTrue(completion["checks"])
                self.assertTrue(all(check["implemented"] for check in completion["checks"]))

                adapter_file, separator, object_name = implementation["entrypoint"].partition(":")
                self.assertEqual(":", separator)
                self.assertEqual("Adapter", object_name)
                self.assertTrue((path.parent / adapter_file).is_file())

    def test_all_builtin_adapter_classes_use_the_four_method_contract(self) -> None:
        removed = {"prepare", "replay"}
        for adapter_path in sorted(PLUGIN_ROOT.glob("*/adapter*.py")):
            tree = ast.parse(adapter_path.read_text(encoding="utf-8"), filename=str(adapter_path))
            classes = [
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "Adapter"
            ]
            self.assertEqual(1, len(classes), adapter_path)
            methods = {
                item.name
                for item in classes[0].body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            self.assertTrue(ADAPTER_METHODS <= methods, adapter_path)
            self.assertFalse(removed & methods, adapter_path)

    def test_loaded_adapters_expose_only_the_required_lifecycle_methods(self) -> None:
        for plugin_id, spec in discover_plugins(PLUGIN_ROOT).items():
            with self.subTest(plugin=plugin_id):
                adapter = load_adapter(spec)
                self.assertTrue(
                    all(callable(getattr(adapter, method, None)) for method in ADAPTER_METHODS)
                )
                self.assertFalse(hasattr(adapter, "prepare"))
                self.assertFalse(hasattr(adapter, "replay"))

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

        specs = discover_plugins(PLUGIN_ROOT)
        for path, manifest in self.manifests():
            adapter_file = manifest["implementation"]["entrypoint"].partition(":")[0]
            adapter_path = path.parent / adapter_file
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

            adapter = load_adapter(specs[manifest["id"]])
            blocked = adapter.plan({})
            self.assertEqual("BLOCKED", blocked["status"])
            self.assertFalse(blocked["executable"])

    def test_training_exposes_four_implemented_framework_adapters_with_limitations(self) -> None:
        training = load_json(PLUGIN_ROOT / "mlip-training" / "plugin.yaml")
        backends = {item["id"]: item for item in training["computational_backends"]}
        self.assertEqual({"deepmd", "m3gnet", "chgnet", "mace"}, set(backends))
        self.assertTrue(all(item["status"] == "implemented" for item in backends.values()))
        self.assertTrue(all(item["limitations"] for item in backends.values()))


if __name__ == "__main__":
    unittest.main()
