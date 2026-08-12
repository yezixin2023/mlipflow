from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from mlipflow.backends import ExecutionResult
from mlipflow.config import load_project
from mlipflow.services import (
    advance,
    initialize,
    make_advance_plan,
    make_retry_plan,
    make_run_plan,
    make_stop_plan,
    retry,
    run_node,
)
from mlipflow.state import RunState, StateStore

from .helpers import project_config, write_json


ROOT = Path(__file__).resolve().parents[1]
PLUGINS = ROOT / "plugins"
SUBMIT_TEMPLATE = """#!/bin/bash
# {{PROJECT_ID}} {{NODE_ID}} attempt {{ATTEMPT}}
#SBATCH --cpus-per-task={{CPUS}}
#SBATCH --gpus={{GPUS}}
#SBATCH --mem={{MEMORY}}
#SBATCH --time={{WALLTIME}}
#SBATCH --output={{LOG_DIR}}/stdout.log
#SBATCH --error={{LOG_DIR}}/stderr.log
cd {{RUN_DIR}}
bash {{RUN_DIR}}/run.sh
"""
RUN_TEMPLATE = """#!/bin/bash
# inputs={{INPUT_DIR}}
# outputs={{OUTPUT_DIR}}
cd {{RUN_DIR}}
"""


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


class FakeTemplateLibrary:
    def __init__(self, templates: dict[str, str] | None = None):
        self.templates = templates or {
            "slurm/cpu.sbatch": SUBMIT_TEMPLATE,
            "slurm/gpu.sbatch": SUBMIT_TEMPLATE,
            "vasp/run.sh": RUN_TEMPLATE,
        }

    def read_template(self, _root: str, relative: str) -> dict[str, Any]:
        content = self.templates.get(relative)
        if content is None:
            return {"relative_path": relative, "exists": False}
        payload = content.encode("utf-8")
        return {
            "relative_path": relative,
            "exists": True,
            "content": content,
            "size_bytes": len(payload),
            "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        }


def write_site(root: Path) -> Path:
    path = root / "site.yaml"
    write_json(
        path,
        {
            "schema_version": 1,
            "clusters": {
                "cluster-a": {
                    "backend": "ssh-slurm",
                    "ssh_profile": "cluster-a",
                    "remote_template_root": "/templates/cluster-a",
                    "work_root": "/work/cluster-a",
                }
            },
        },
    )
    return path


def prepared_fixture(root: Path) -> tuple[dict[str, Any], Path]:
    inputs = root / "inputs"
    prepared = root / "prepared"
    inputs.mkdir(parents=True)
    prepared.mkdir()
    source = inputs / "li.vasp"
    source.write_text(
        "Li\n1.0\n3 0 0\n0 3 0\n0 0 3\nLi\n1\nDirect\n0 0 0\n",
        encoding="utf-8",
    )
    structures = inputs / "structures.json"
    write_json(
        structures,
        {
            "schema_version": 1,
            "structures": [{"id": "li", "path": "li.vasp", "fingerprint": sha256(source)}],
        },
    )
    labeling = inputs / "labeling.json"
    write_json(
        labeling,
        {
            "schema_version": 1,
            "engine": "vasp",
            "calculation_type": "static",
            "incar": {"ENCUT": 450, "EDIFF": 5e-6, "NSW": 0, "IBRION": -1},
            "kpoints": {"mode": "gamma", "grid": [1, 1, 1], "shift": [0, 0, 0]},
        },
    )
    contents = {
        "POSCAR": source.read_text(encoding="utf-8"),
        "INCAR": "ENCUT = 450\nEDIFF = 5e-6\nNSW = 0\nIBRION = -1\n",
        "KPOINTS": "mesh\n0\nGamma\n1 1 1\n0 0 0\n",
        "POTCAR": "TEST-ONLY-POTCAR\n",
    }
    files: dict[str, Any] = {}
    for name, text in contents.items():
        path = prepared / name
        path.write_text(text, encoding="utf-8")
        files[name] = {
            "path": name,
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
            "collectable": name != "POTCAR",
        }
    write_json(
        prepared / "dft-input-manifest.json",
        {
            "schema_version": 1,
            "plugin_id": "dft-labeling",
            "operation": "vasp-prepare",
            "status": "OK",
            "engine": "vasp",
            "calculation_type": "static",
            "structure_count": 1,
            "calculations": [
                {"structure_id": "li", "atom_count": 1, "files": files}
            ],
        },
    )
    parameters = {
        "operation": "label",
        "engine": "vasp",
        "completion_policy": {"require_ionic_convergence": False},
        "units": {
            "energy": "eV",
            "length": "angstrom",
            "force": "eV/angstrom",
            "stress": "kbar-vasp-3x3",
        },
        "result_manifest": "dft-labeling-result.json",
    }
    node = {
        "id": "label-li",
        "uses": "dft-labeling@0",
        "backend": "ssh-slurm",
        "backend_profile": "cluster-a",
        "inputs": {
            "structures_manifest": "inputs/structures.json",
            "labeling_config": "inputs/labeling.json",
            "dft_input_manifest": "prepared/dft-input-manifest.json",
        },
        "parameters": parameters,
        "resources": {"cpus": 4, "gpus": 0, "memory": "4G", "walltime": "00:05:00"},
    }
    write_json(root / "project.yaml", project_config([node]))
    return parameters, write_site(root)


def write_vasp_outputs(remote: Path, *, converged: bool = True) -> None:
    output = remote / "output"
    logs = remote / "logs"
    output.mkdir(parents=True)
    logs.mkdir()
    (output / "OUTCAR").write_text(
        "vasp.6.3.0\nGeneral timing and accounting informations for this job:\n",
        encoding="utf-8",
    )
    (output / "OSZICAR").write_text(
        " 1 F= -.150000E+01 E0= -.150000E+01\n", encoding="utf-8"
    )
    steps = "<scstep/>" if converged else "".join("<scstep/>" for _ in range(2))
    nelm = 700 if converged else 2
    (output / "vasprun.xml").write_text(
        "<modeling>"
        f'<parameters><separator><i name="NELM">{nelm}</i></separator></parameters>'
        '<atominfo><array name="atoms"><set><rc><c>Li</c></rc></set></array></atominfo>'
        '<structure name="finalpos"><crystal><varray name="basis">'
        '<v>3 0 0</v><v>0 3 0</v><v>0 0 3</v></varray></crystal>'
        '<varray name="positions"><v>0 0 0</v></varray></structure>'
        f'<calculation>{steps}<energy><i name="e_0_energy">-1.5</i></energy>'
        '<varray name="forces"><v>0 0 0</v></varray>'
        '<varray name="stress"><v>1 0 0</v><v>0 1 0</v><v>0 0 1</v></varray>'
        "</calculation></modeling>",
        encoding="utf-8",
    )
    (logs / "stdout.log").write_text("done\n", encoding="utf-8")
    (logs / "stderr.log").write_text("", encoding="utf-8")


class ScheduledDftTests(unittest.TestCase):
    def _submit(self, root: Path) -> tuple[Any, dict[str, Any], Path, FakeTemplateLibrary]:
        _, site = prepared_fixture(root)
        initialize(root)
        project = load_project(root)
        library = FakeTemplateLibrary()
        plan = make_run_plan(project, "label-li", PLUGINS, site, library)
        remote_dir = plan["hpc_execution"]["workspace"]["run_dir"]
        with patch(
            "mlipflow.services.SshSlurmBackend.stage_workspace",
            return_value=remote_dir,
        ) as staged, patch(
            "mlipflow.services.SshSlurmBackend.submit",
            return_value=ExecutionResult(0, "Submitted batch job 77\n", "", "77"),
        ):
            run_node(
                project, "label-li", PLUGINS, plan["plan_digest"], site, library
            )
        staged_paths = {item[1] for item in staged.call_args.args[1]}
        self.assertIn("input/POTCAR", staged_paths)
        self.assertIn("submit.sbatch", staged_paths)
        self.assertIn("run.sh", staged_paths)
        self.assertNotIn("POTCAR", plan["adapter_plan"]["approval_summary"]["fetch_allowlist"])
        return project, plan, site, library

    def _inventory_hooks(self, remote: Path):
        def inspect(_self: object, _cwd: str, remote_path: str) -> dict[str, object]:
            path = remote / remote_path
            if not path.is_file():
                return {"path": remote_path, "exists": False}
            return {
                "path": remote_path,
                "exists": True,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }

        def fetch(
            _self: object, _cwd: str, remote_path: str, destination: Path
        ) -> Path:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(remote / remote_path, destination)
            return destination

        return inspect, fetch

    def test_plan_stage_fetch_check_collect_reaches_ok(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, plan, site, _ = self._submit(root)
            self.assertEqual(
                "/work/cluster-a/test-project/label-li/attempt-0001",
                plan["hpc_execution"]["workspace"]["run_dir"],
            )
            self.assertEqual(
                "slurm/cpu.sbatch",
                plan["hpc_execution"]["templates"]["submit.sbatch"]["relative_path"],
            )
            remote = root / "fake-remote"
            write_vasp_outputs(remote)
            write_json(
                remote / "completion.json",
                {
                    "schema_version": 1,
                    "status": "COMPLETED",
                    "exit_code": 0,
                    "project_id": project.project_id,
                    "node_id": "label-li",
                    "attempt": 1,
                },
            )
            inspect, fetch = self._inventory_hooks(remote)
            with patch(
                "mlipflow.services.SshSlurmBackend.status",
                return_value={"state": "COMPLETED", "detail": None, "source": "fake"},
            ), patch(
                "mlipflow.services.SshSlurmBackend.inspect_file",
                autospec=True,
                side_effect=inspect,
            ):
                approved = make_advance_plan(project, PLUGINS, site)
            self.assertEqual(
                "adapter-finalize", approved["details"]["transitions"][0]["action"]
            )
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
                finished = advance(project, approved["plan_digest"], PLUGINS, site)
            self.assertEqual("OK", finished["changed"][0]["state"])
            attempt = root / ".mlipflow/runs/label-li/attempt-1"
            self.assertTrue((attempt / "dft-labeling-result.json").is_file())
            self.assertTrue((attempt / "labels.json").is_file())
            self.assertFalse((attempt / "POTCAR").is_file())

    def test_scheduler_completed_but_scientific_check_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, _, site, _ = self._submit(root)
            remote = root / "fake-remote"
            write_vasp_outputs(remote, converged=False)
            write_json(
                remote / "completion.json",
                {
                    "schema_version": 1,
                    "status": "COMPLETED",
                    "exit_code": 0,
                    "project_id": project.project_id,
                    "node_id": "label-li",
                    "attempt": 1,
                },
            )
            inspect, fetch = self._inventory_hooks(remote)
            patches = (
                patch(
                    "mlipflow.services.SshSlurmBackend.status",
                    return_value={"state": "COMPLETED", "detail": None, "source": "fake"},
                ),
                patch(
                    "mlipflow.services.SshSlurmBackend.inspect_file",
                    autospec=True,
                    side_effect=inspect,
                ),
            )
            with patches[0], patches[1]:
                approved = make_advance_plan(project, PLUGINS, site)
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
                finished = advance(project, approved["plan_digest"], PLUGINS, site)
            self.assertEqual("FAIL", finished["changed"][0]["state"])
            self.assertIn("completion check", finished["changed"][0]["diagnostic"])

    def test_retry_uses_new_persisted_attempt_in_remote_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, _, site, library = self._submit(root)
            with StateStore(root / ".mlipflow/state.sqlite3", readonly=False) as store:
                submitted = store.latest_step(project.project_id, "label-li")
                store.transition(submitted.run_id, RunState.FAIL, diagnostic="synthetic failure")
            retry_plan = make_retry_plan(project, "label-li", PLUGINS)
            retried = retry(
                project, "label-li", retry_plan["plan_digest"], PLUGINS
            )
            self.assertEqual(2, retried["step"]["attempt"])
            second = make_run_plan(project, "label-li", PLUGINS, site, library)
            self.assertEqual(
                "/work/cluster-a/test-project/label-li/attempt-0002",
                second["hpc_execution"]["workspace"]["run_dir"],
            )

    def test_invalid_completion_identity_fails_before_scientific_collect(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, _, site, _ = self._submit(root)
            remote = root / "fake-remote"
            write_vasp_outputs(remote)
            write_json(
                remote / "completion.json",
                {
                    "schema_version": 1,
                    "status": "COMPLETED",
                    "exit_code": 0,
                    "project_id": project.project_id,
                    "node_id": "label-li",
                    "attempt": 999,
                },
            )
            inspect, fetch = self._inventory_hooks(remote)
            with patch(
                "mlipflow.services.SshSlurmBackend.status",
                return_value={"state": "COMPLETED", "detail": None, "source": "fake"},
            ), patch(
                "mlipflow.services.SshSlurmBackend.inspect_file",
                autospec=True,
                side_effect=inspect,
            ):
                approved = make_advance_plan(project, PLUGINS, site)
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
                finished = advance(project, approved["plan_digest"], PLUGINS, site)
            self.assertEqual("FAIL", finished["changed"][0]["state"])
            self.assertIn("field attempt", finished["changed"][0]["diagnostic"])
            attempt = root / ".mlipflow/runs/label-li/attempt-1"
            self.assertFalse((attempt / "labels.json").exists())

    def _submit_with_plugin_copy(self, root: Path):
        """Submit using a private copy of the plugin tree we may safely edit."""

        plugins = root / "plugin-copy"
        shutil.copytree(PLUGINS, plugins)
        _, site = prepared_fixture(root)
        initialize(root)
        project = load_project(root)
        library = FakeTemplateLibrary()
        plan = make_run_plan(project, "label-li", plugins, site, library)
        remote_dir = plan["hpc_execution"]["workspace"]["run_dir"]
        with patch(
            "mlipflow.services.SshSlurmBackend.stage_workspace",
            return_value=remote_dir,
        ), patch(
            "mlipflow.services.SshSlurmBackend.submit",
            return_value=ExecutionResult(0, "Submitted batch job 77\n", "", "77"),
        ):
            run_node(project, "label-li", plugins, plan["plan_digest"], site, library)
        return project, plugins, site

    def test_pinned_plan_survives_touching_every_file_after_submission(self) -> None:
        """A queued job must not become unadvanceable because mtimes moved.

        Under plan schema 1 this failed with ``pinned scheduled plan field
        changed: plugin``: re-staging, an rsync, or any tool that rewrites a file
        in place stranded the submitted job permanently.
        """

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, plugins, site = self._submit_with_plugin_copy(root)

            times = (1_700_000_000, 1_700_000_000)
            for tree in (plugins, root / "prepared"):
                for path in sorted(tree.rglob("*"), reverse=True):
                    os.utime(path, times, follow_symlinks=False)

            remote = root / "fake-remote"
            write_vasp_outputs(remote)
            write_json(
                remote / "completion.json",
                {
                    "schema_version": 1,
                    "status": "COMPLETED",
                    "exit_code": 0,
                    "project_id": project.project_id,
                    "node_id": "label-li",
                    "attempt": 1,
                },
            )
            inspect, _ = self._inventory_hooks(remote)
            with patch(
                "mlipflow.services.SshSlurmBackend.status",
                return_value={"state": "COMPLETED", "detail": None, "source": "fake"},
            ), patch(
                "mlipflow.services.SshSlurmBackend.inspect_file",
                autospec=True,
                side_effect=inspect,
            ):
                approved = make_advance_plan(project, plugins, site)
            self.assertEqual(
                "adapter-finalize", approved["details"]["transitions"][0]["action"]
            )

    def _advance_expecting_rejection(self, project, plugins, site) -> str:
        """Advance with every remote call trapped: rejection must precede them."""

        with patch(
            "mlipflow.services.SshSlurmBackend.status",
            return_value={"state": "COMPLETED", "detail": None, "source": "fake"},
        ), patch(
            "mlipflow.services.SshSlurmBackend.inspect_file",
            side_effect=AssertionError("a rejected plan must not touch the cluster"),
        ), patch(
            "mlipflow.backends.subprocess.run",
            side_effect=AssertionError("no subprocess may be spawned"),
        ):
            observed = make_advance_plan(project, plugins, site)
        self.assertEqual([], observed["details"]["transitions"])
        return str(observed["details"]["observations"][0]["reason"])

    def test_pinned_plan_still_rejects_a_real_adapter_source_change(self) -> None:
        """The guarantee that must survive dropping mtime from the digest."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, plugins, site = self._submit_with_plugin_copy(root)

            adapter = plugins / "dft-labeling" / "adapter.py"
            adapter.write_text(
                adapter.read_text(encoding="utf-8") + "\n# tampered\n",
                encoding="utf-8",
            )
            self.assertIn(
                "pinned scheduled plan field changed: plugin",
                self._advance_expecting_rejection(project, plugins, site),
            )

    def test_pinned_plan_still_rejects_a_real_input_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, plugins, site = self._submit_with_plugin_copy(root)

            manifest = root / "prepared" / "dft-input-manifest.json"
            original = manifest.read_text(encoding="utf-8")
            manifest.write_text(original + "\n", encoding="utf-8")
            self.assertNotEqual(original, manifest.read_text(encoding="utf-8"))
            self.assertIn(
                "pinned scheduled plan field changed: input_identities",
                self._advance_expecting_rejection(project, plugins, site),
            )

    def test_core_plan_fields_carry_no_mtime_and_no_absolute_path(self) -> None:
        """Everything the core contributes to a plan must be relocatable.

        ``adapter_plan`` is excluded because it is authored by the plugin, and
        ``dft-labeling`` still emits absolute ``source``/``path`` values there.
        That is a known remaining source of machine dependence, tracked as plugin
        work; this test locks down the core so the boundary cannot quietly widen,
        and asserts that *no* part of a plan — adapter included — carries mtime.
        """

        def leaves(value, trail=""):
            if isinstance(value, dict):
                for key, item in value.items():
                    yield from leaves(item, f"{trail}.{key}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    yield from leaves(item, f"{trail}[{index}]")
            else:
                yield trail, value

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            _, site = prepared_fixture(root)
            initialize(root)
            project = load_project(root)
            plan = make_run_plan(
                project, "label-li", PLUGINS, site, FakeTemplateLibrary()
            )

            self.assertEqual(2, plan["schema_version"])
            self.assertEqual(
                [], [trail for trail, _ in leaves(plan) if "mtime" in trail.lower()]
            )
            core = {key: value for key, value in plan.items() if key != "adapter_plan"}
            self.assertEqual(
                [],
                [
                    trail
                    for trail, value in leaves(core)
                    if isinstance(value, str)
                    and (str(root) in value or value.startswith("file:///"))
                ],
            )
            self.assertEqual(
                "adapter.py", plan["plugin"]["implementation_identity"]["locator"]
            )
            self.assertEqual(
                "prepared/dft-input-manifest.json",
                plan["input_identities"]["dft_input_manifest"]["locator"],
            )

    def test_scheduled_plan_digests_identically_across_checkouts(self) -> None:
        """The full ssh-slurm READY plan, planned twice under different roots.

        This is the richest adapter plan the project produces: staged sources,
        per-file fingerprints and a rendered remote workspace.  It is also the one
        whose approval has to survive a clone, because staging and advancing may
        well happen from a different machine than planning did.
        """

        with tempfile.TemporaryDirectory() as first_temporary, tempfile.TemporaryDirectory() as second_temporary:
            first = Path(first_temporary).resolve()
            second = Path(second_temporary).resolve() / "nested" / "deeper"
            second.mkdir(parents=True)

            digests = []
            for root in (first, second):
                _, site = prepared_fixture(root)
                initialize(root)
                plan = make_run_plan(
                    load_project(root), "label-li", PLUGINS, site, FakeTemplateLibrary()
                )
                self.assertEqual("READY", plan["adapter_plan"]["status"])
                digests.append(plan["plan_digest"])
                self.assertEqual(
                    "{PROJECT_ROOT}/prepared/POSCAR",
                    next(
                        item["source"]
                        for item in plan["adapter_plan"]["scheduled_execution"][
                            "staged_files"
                        ]
                        if item["remote_name"] == "POSCAR"
                    ),
                )
            self.assertEqual(digests[0], digests[1])

    def test_stop_plan_binds_full_cluster_profile_and_site_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, _, site, _ = self._submit(root)
            first = make_stop_plan(project, "label-li", site)
            details = first["details"]
            self.assertEqual("cluster-a", details["scheduler_target"]["name"])
            self.assertEqual(
                "/templates/cluster-a",
                details["scheduler_target"]["remote_template_root"],
            )
            raw = json.loads(site.read_text(encoding="utf-8"))
            raw["clusters"]["cluster-a"]["ssh_profile"] = "changed-alias"
            write_json(site, raw)
            changed = make_stop_plan(project, "label-li", site)
            self.assertNotEqual(first["plan_digest"], changed["plan_digest"])


if __name__ == "__main__":
    unittest.main()
