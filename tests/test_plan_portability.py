"""A signed plan must describe the work, not the machine that planned it.

``content_identity`` removed ``mtime_ns`` and absolute paths from the fields the
*core* contributes.  Adapters, however, legitimately think in absolute paths —
every one of the eight emits ``cwd``, and several emit absolute argv entries,
staged ``source`` values, or diagnostic messages naming files.  All of that lands
in ``adapter_plan`` / ``adapter_diagnostics``, both of which are signed.

The core therefore rewrites those roots to ``{PROJECT_ROOT}`` / ``{ATTEMPT_DIR}``
/ ``{PLUGIN_DIR}`` before signing and resolves them again before execution.  These
tests hold that line for every bundled plugin at once, so a new adapter cannot
reintroduce machine dependence unnoticed.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlipflow.backends import ExecutionResult
from mlipflow.config import load_project
from mlipflow.plugins import discover_plugins
from mlipflow.portable import (
    ATTEMPT_DIR_TOKEN as ATTEMPT_TOKEN,
    TOKENS,
    PortableRoots,
    contains_absolute_root,
    to_runtime,
)
from mlipflow.services import make_run_plan, run_node

from .helpers import project_config, write_json


ROOT = Path(__file__).resolve().parents[1]
PLUGINS = ROOT / "plugins"
EXAMPLE = ROOT / "examples" / "high_entropy_sulfide"

# One node per bundled plugin.  With minimal inputs these come back BLOCKED,
# which still exercises the diagnostics path — an adapter that quotes the file it
# could not find would leak an absolute path into a signed plan.
PLUGIN_NODES = {
    "ase-md": {"calculator": "mace", "ensemble": "nvt-langevin"},
    "candidate-ranking": {"operation": "rank-candidates"},
    "dft-labeling": {"operation": "vasp-prepare"},
    "electrochemical-voltage": {"operation": "compute-from-energies"},
    "high-entropy-structure": {"operation": "generate-sqs"},
    "ionic-transport": {"operation": "analyze-existing"},
    "lammps-md": {"operation": "lammps-prepare"},
    "mlip-benchmark": {"operation": "evaluate-static"},
    "mlip-training": {"operation": "train"},
    "pes-sampling": {"operation": "direct-select"},
}
PLUGIN_BACKENDS = {"ase-md": "ssh-slurm"}

# A BLOCKED adapter returns before it builds argv or cwd, so those plans contain
# no paths at all and prove little on their own.  These two reach READY with
# purely declared inputs, so their plans carry the real argv/cwd an approval
# would authorise — that is where portability actually has to hold.
READY_NODES = {
    "mlip-benchmark": (
        {
            "executable": "/usr/bin/python3",
            "script": "tools/benchmark.py",
            "model": "models/chgnet.pth",
            "benchmark_dataset": "data/test.json",
            "benchmark_config": "configs/benchmark.json",
            "result_manifest": "benchmark-result.json",
        },
        {
            "operation": "evaluate-static",
            "model_family": "chgnet",
            "task": "ionic-transport",
            "scenario": "static-holdout-v1",
            "energy_normalization": "per-atom",
            "stress_convention": "VASP sign; GPa; xx,yy,zz,xy,yz,zx",
            "dataset_fingerprint": "sha256:dataset",
            "model_fingerprint": "sha256:model",
        },
    ),
    "mlip-training": (
        {
            "executable": "/usr/bin/python3",
            "script": "wrappers/mace_train.py",
            "config": "configs/mace.json",
            "data": "data/labeled",
            "output": "model.bin",
            "result_manifest": "training-result.json",
        },
        {
            "framework": "mace",
            "operation": "train",
            "seed": 20260810,
            "device": "cpu",
            "precision": "float64",
            "dataset_fingerprint": "sha256:dataset",
            "config_fingerprint": "sha256:config",
        },
    ),
}


def leaves(value, trail=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, f"{trail}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, f"{trail}[{index}]")
    else:
        yield trail, value


def build_checkout(
    root: Path,
    plugin_id: str,
    parameters: dict,
    inputs: dict | None = None,
) -> None:
    """Materialise a self-contained checkout: plugins, inputs and project."""

    shutil.copytree(PLUGINS, root / "plugins")
    (root / "inputs").mkdir()
    (root / "inputs" / "candidates.json").write_text(
        json.dumps({"schema_version": 1, "candidates": []}), encoding="utf-8"
    )
    (root / "inputs" / "structure.vasp").write_text(
        "Li\n1.0\n3 0 0\n0 3 0\n0 0 3\nLi\n1\nDirect\n0 0 0\n", encoding="utf-8"
    )
    node = {
        "id": "n",
        "uses": f"{plugin_id}@0",
        "mode": "execute",
        "backend": PLUGIN_BACKENDS.get(plugin_id, "local"),
        "inputs": inputs
        if inputs is not None
        else {
            "candidates": "inputs/candidates.json",
            "structure": "inputs/structure.vasp",
        },
        "parameters": parameters,
    }
    if node["backend"] == "ssh-slurm":
        node["backend_profile"] = "cluster-a"
        node["resources"] = {
            "cpus": 1,
            "gpus": 0,
            "memory": "1G",
            "walltime": "00:01:00",
        }
    write_json(root / "project.yaml", project_config([node]))


def build_ready_checkout(root: Path, plugin_id: str) -> None:
    inputs, parameters = READY_NODES[plugin_id]
    build_checkout(root, plugin_id, parameters, inputs)


def plan_for(root: Path) -> dict:
    return make_run_plan(load_project(root), "n", root / "plugins")


def plan_for_ready(root: Path, plugin_id: str) -> dict:
    build_ready_checkout(root, plugin_id)
    return plan_for(root)


class AdapterPlanPortabilityTests(unittest.TestCase):
    """Every bundled adapter, planned twice under different roots."""

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.base = Path(self._temporary.name).resolve()

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _two_checkouts(self, plugin_id: str) -> tuple[dict, dict, Path, Path]:
        first = self.base / "alpha"
        second = self.base / "nested" / "deeper" / "beta-with-longer-name"
        first.mkdir(parents=True)
        second.mkdir(parents=True)
        parameters = PLUGIN_NODES[plugin_id]
        build_checkout(first, plugin_id, parameters)
        build_checkout(second, plugin_id, parameters)
        return plan_for(first), plan_for(second), first, second

    def test_every_bundled_plugin_plans_identically_in_two_checkouts(self) -> None:
        for plugin_id in sorted(PLUGIN_NODES):
            with self.subTest(plugin=plugin_id):
                first, second, _, _ = self._two_checkouts(plugin_id)
                self.assertEqual(first["plan_digest"], second["plan_digest"])
                shutil.rmtree(self.base / "alpha")
                shutil.rmtree(self.base / "nested")

    def test_no_plan_leaf_names_either_checkout_root(self) -> None:
        for plugin_id in sorted(PLUGIN_NODES):
            with self.subTest(plugin=plugin_id):
                first, _, root_a, root_b = self._two_checkouts(plugin_id)
                roots = PortableRoots(
                    project_root=root_a,
                    attempt_dir=root_a / ".mlipflow" / "runs" / "n" / "attempt-1",
                    plugin_dir=root_a / "plugins" / plugin_id,
                )
                self.assertEqual([], contains_absolute_root(first, roots))
                self.assertEqual(
                    [],
                    [
                        trail
                        for trail, value in leaves(first)
                        if isinstance(value, str) and str(root_b) in value
                    ],
                )
                shutil.rmtree(self.base / "alpha")
                shutil.rmtree(self.base / "nested")

    def test_no_plan_leaf_carries_mtime(self) -> None:
        for plugin_id in sorted(PLUGIN_NODES):
            with self.subTest(plugin=plugin_id):
                first, _, _, _ = self._two_checkouts(plugin_id)
                self.assertEqual(
                    [], [trail for trail, _ in leaves(first) if "mtime" in trail.lower()]
                )
                shutil.rmtree(self.base / "alpha")
                shutil.rmtree(self.base / "nested")

    def test_relocating_an_identical_checkout_preserves_the_digest(self) -> None:
        """The clone/rsync case, end to end through a real adapter."""

        source = self.base / "origin"
        source.mkdir()
        build_checkout(source, "dft-labeling", PLUGIN_NODES["dft-labeling"])
        before = plan_for(source)["plan_digest"]

        moved = self.base / "relocated" / "somewhere" / "else"
        moved.parent.mkdir(parents=True)
        shutil.copytree(source, moved)
        self.assertEqual(before, plan_for(moved)["plan_digest"])

    def test_changed_input_bytes_still_change_the_digest(self) -> None:
        root = self.base / "alpha"
        root.mkdir()
        build_checkout(root, "dft-labeling", PLUGIN_NODES["dft-labeling"])
        before = plan_for(root)["plan_digest"]

        (root / "inputs" / "structure.vasp").write_text(
            "Li\n1.0\n4 0 0\n0 4 0\n0 0 4\nLi\n1\nDirect\n0 0 0\n", encoding="utf-8"
        )
        self.assertNotEqual(before, plan_for(root)["plan_digest"])

    def test_changed_adapter_source_still_changes_the_digest(self) -> None:
        root = self.base / "alpha"
        root.mkdir()
        build_checkout(root, "dft-labeling", PLUGIN_NODES["dft-labeling"])
        before = plan_for(root)["plan_digest"]

        adapter = root / "plugins" / "dft-labeling" / "adapter.py"
        adapter.write_text(
            adapter.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8"
        )
        self.assertNotEqual(before, plan_for(root)["plan_digest"])

class ReadyPlanPortabilityTests(unittest.TestCase):
    """The cases that matter: plans an approval could actually execute."""

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.base = Path(self._temporary.name).resolve()

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_ready_plans_really_are_ready(self) -> None:
        """Anti-vacuity: a BLOCKED plan proves nothing about argv or cwd."""

        for plugin_id in sorted(READY_NODES):
            with self.subTest(plugin=plugin_id):
                root = self.base / plugin_id
                root.mkdir(parents=True)
                plan = plan_for_ready(root, plugin_id)
                self.assertEqual("READY", plan["adapter_plan"]["status"])
                self.assertTrue(plan["adapter_plan"]["executable"])

    def test_ready_plans_carry_portable_tokens_not_absolute_paths(self) -> None:
        for plugin_id in sorted(READY_NODES):
            with self.subTest(plugin=plugin_id):
                root = self.base / plugin_id
                root.mkdir(parents=True)
                plan = plan_for_ready(root, plugin_id)
                rendered = json.dumps(plan)
                self.assertTrue(
                    any(token in rendered for token in TOKENS),
                    f"{plugin_id} plan should contain a portable token: "
                    f"cwd={plan['adapter_plan'].get('cwd')!r}",
                )
                self.assertEqual(ATTEMPT_TOKEN, plan["adapter_plan"]["cwd"])
                roots = PortableRoots(
                    project_root=root,
                    attempt_dir=root / ".mlipflow" / "runs" / "n" / "attempt-1",
                    plugin_dir=root / "plugins" / plugin_id,
                )
                self.assertEqual([], contains_absolute_root(plan, roots))

    def test_ready_plans_digest_identically_across_checkouts(self) -> None:
        for plugin_id in sorted(READY_NODES):
            with self.subTest(plugin=plugin_id):
                first = self.base / plugin_id / "alpha"
                second = self.base / plugin_id / "nested" / "deeper" / "beta-longer"
                first.mkdir(parents=True)
                second.mkdir(parents=True)
                self.assertEqual(
                    plan_for_ready(first, plugin_id)["plan_digest"],
                    plan_for_ready(second, plugin_id)["plan_digest"],
                )

    def test_ready_plan_still_reacts_to_a_real_adapter_change(self) -> None:
        for plugin_id in sorted(READY_NODES):
            with self.subTest(plugin=plugin_id):
                root = self.base / plugin_id
                root.mkdir(parents=True)
                before = plan_for_ready(root, plugin_id)["plan_digest"]
                manifest = json.loads(
                    (root / "plugins" / plugin_id / "plugin.yaml").read_text(
                        encoding="utf-8"
                    )
                )
                adapter = root / "plugins" / plugin_id / manifest["implementation"][
                    "entrypoint"
                ].split(":", 1)[0]
                adapter.write_text(
                    adapter.read_text(encoding="utf-8") + "\n# tampered\n",
                    encoding="utf-8",
                )
                self.assertNotEqual(before, plan_for(root)["plan_digest"])

    def test_ready_plan_argv_resolves_back_to_this_machine(self) -> None:
        """Runtime resolution must undo the rewrite exactly."""

        root = self.base / "mlip-training"
        root.mkdir(parents=True)
        plan = plan_for_ready(root, "mlip-training")
        roots = PortableRoots(
            project_root=root,
            attempt_dir=root / ".mlipflow" / "runs" / "n" / "attempt-1",
            plugin_dir=root / "plugins" / "mlip-training",
        )
        runtime = to_runtime(plan["adapter_plan"], roots)
        self.assertEqual(str(roots.attempt_dir), runtime["cwd"])
        self.assertEqual([], contains_absolute_root(plan["adapter_plan"], roots))
        self.assertNotEqual([], contains_absolute_root(runtime, roots))


class RuntimeResolutionTests(unittest.TestCase):
    """Execution must undo the rewrite, or the subprocess gets a literal token.

    Every shipped example node is ``mode: replay``, which never touches
    ``adapter_plan``, so nothing else in the suite drives a real bundled adapter
    through the local execute path.  This does, and asserts on the argv that
    actually reached the backend.
    """

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.base = Path(self._temporary.name).resolve()

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_executed_argv_and_cwd_are_resolved_absolute_paths(self) -> None:
        root = self.base / "project"
        root.mkdir(parents=True)
        plan = plan_for_ready(root, "mlip-training")
        attempt = root / ".mlipflow" / "runs" / "n" / "attempt-1"
        captured: dict = {}

        def fake_run(_self, argv, cwd, environment=None):
            captured["argv"] = list(argv)
            captured["cwd"] = str(cwd)
            # The adapter's contract: the wrapper leaves a result manifest and
            # the model artifact behind in the attempt directory.
            Path(cwd).mkdir(parents=True, exist_ok=True)
            (Path(cwd) / "model.bin").write_bytes(b"small-test-model")
            (Path(cwd) / "training-result.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "plugin_id": "mlip-training",
                        "status": "OK",
                        "framework": "mace",
                        "framework_version": "test-only",
                        "operation": "train",
                        "seed": 20260810,
                        "device": "cpu",
                        "precision": "float64",
                        "dataset_fingerprint": "sha256:dataset",
                        "config_fingerprint": "sha256:config",
                        "model_artifact": {
                            "path": "model.bin",
                            "media_type": "application/octet-stream",
                        },
                        "metrics": {"best_validation_loss": 0.012},
                    }
                ),
                encoding="utf-8",
            )
            return ExecutionResult(0, "", "")

        with patch("mlipflow.services.LocalBackend.run", autospec=True, side_effect=fake_run):
            result = run_node(
                load_project(root), "n", root / "plugins", plan["plan_digest"]
            )

        self.assertEqual("OK", result["step"]["state"])
        self.assertEqual(str(attempt), captured["cwd"])
        rendered = " ".join(captured["argv"])
        for token in TOKENS:
            self.assertNotIn(token, rendered, f"{token} reached the backend unresolved")
        self.assertIn(str(attempt), rendered)
        self.assertEqual("/usr/bin/python3", captured["argv"][0])

    def test_signed_plan_kept_tokens_while_execution_used_real_paths(self) -> None:
        """Both halves of the contract, on one run."""

        root = self.base / "project"
        root.mkdir(parents=True)
        plan = plan_for_ready(root, "mlip-training")
        self.assertEqual(ATTEMPT_TOKEN, plan["adapter_plan"]["cwd"])
        self.assertIn(
            ATTEMPT_TOKEN, " ".join(plan["adapter_plan"]["argv"])
        )


class ExamplePlanPortabilityTests(unittest.TestCase):
    """The shipped example project, planned from two different checkouts."""

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.base = Path(self._temporary.name).resolve()

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _copy_example(self, name: str) -> tuple[Path, Path]:
        root = self.base / name
        root.mkdir(parents=True)
        shutil.copytree(EXAMPLE, root / "project")
        shutil.copytree(PLUGINS, root / "plugins")
        return root / "project", root / "plugins"

    def test_every_example_node_digests_identically_across_checkouts(self) -> None:
        first_project, first_plugins = self._copy_example("alpha")
        second_project, second_plugins = self._copy_example("beta/nested/deeper")

        first = load_project(first_project)
        second = load_project(second_project)
        node_ids = [str(node["id"]) for node in first.nodes]
        self.assertTrue(node_ids)
        for node_id in node_ids:
            with self.subTest(node=node_id):
                self.assertEqual(
                    make_run_plan(first, node_id, first_plugins)["plan_digest"],
                    make_run_plan(second, node_id, second_plugins)["plan_digest"],
                )

    def test_example_plans_carry_no_machine_specific_leaf(self) -> None:
        project_dir, plugins_dir = self._copy_example("alpha")
        project = load_project(project_dir)
        for node in project.nodes:
            node_id = str(node["id"])
            with self.subTest(node=node_id):
                plan = make_run_plan(project, node_id, plugins_dir)
                roots = PortableRoots(
                    project_root=project_dir,
                    attempt_dir=project_dir
                    / ".mlipflow"
                    / "runs"
                    / node_id
                    / "attempt-1",
                    plugin_dir=plugins_dir / str(node["uses"]).split("@", 1)[0],
                )
                self.assertEqual([], contains_absolute_root(plan, roots))
                self.assertEqual(
                    [], [trail for trail, _ in leaves(plan) if "mtime" in trail.lower()]
                )


class DiscoveryTests(unittest.TestCase):
    def test_plugin_node_table_covers_every_bundled_plugin(self) -> None:
        """A new plugin must be added here rather than silently skipped."""

        self.assertEqual(
            set(discover_plugins(PLUGINS)), set(PLUGIN_NODES)
        )


if __name__ == "__main__":
    unittest.main()
