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


OUTCAR_FOOTER = "General timing and accounting informations for this job:"
RELAX_MARKER = "reached required accuracy - stopping structural energy minimisation"


def vasprun_xml(
    *,
    nelm: int = 700,
    frames: int = 1,
    electronic_steps: int = 1,
    scale: float = 3.0,
    species: tuple[str, ...] = ("Li",),
) -> str:
    """Build a vasprun.xml that mirrors real VASP 5.4.x layout.

    Two quirks a hand-written minimal XML hides, and that a real cluster run
    exposed, are reproduced here: NELM appears twice under <parameters>, and the
    per-ionic-step energy lives in the last <scstep> while the trailing
    <calculation><energy> block reports the electronic entropy in e_0_energy.
    Each <calculation> also carries its own <structure>, which is what makes an
    AIMD trajectory readable frame by frame.
    """

    atoms = "".join(f"<rc><c>{symbol}</c></rc>" for symbol in species)
    positions = "".join("<v>0 0 0</v>" for _ in species)
    forces = "".join("<v>0 0 0</v>" for _ in species)
    calculations = []
    for index in range(frames):
        step_energy = -1.5 - 0.01 * index
        scstep_block = "".join(
            "<scstep><energy>"
            f'<i name="e_fr_energy">{step_energy - 0.0005}</i>'
            f'<i name="e_wo_entrp">{step_energy + 0.0005}</i>'
            f'<i name="e_0_energy">{step_energy}</i>'
            "</energy></scstep>"
            for _ in range(electronic_steps)
        )
        calculations.append(
            f"<calculation>{scstep_block}"
            "<structure><crystal><varray name=\"basis\">"
            f"<v>{scale} 0 0</v><v>0 {scale} 0</v><v>0 0 {scale}</v>"
            "</varray></crystal>"
            f'<varray name="positions">{positions}</varray></structure>'
            f'<varray name="forces">{forces}</varray>'
            '<varray name="stress"><v>1 0 0</v><v>0 1 0</v><v>0 0 1</v></varray>'
            "<energy>"
            f'<i name="e_fr_energy">{step_energy - 0.0005}</i>'
            f'<i name="e_wo_entrp">{step_energy}</i>'
            '<i name="e_0_energy">-0.00081482</i>'
            "</energy>"
            "</calculation>"
        )
    return (
        "<modeling>"
        "<parameters>"
        '<separator name="electronic convergence">'
        f'<i type="int" name="NELM">{nelm}</i>'
        "</separator>"
        '<separator name="response functions">'
        '<i type="int" name="NELM">1</i>'
        "</separator>"
        "</parameters>"
        f'<atominfo><array name="atoms"><set>{atoms}</set></array></atominfo>'
        + "".join(calculations)
        + '<structure name="finalpos"><crystal><varray name="basis">'
        f"<v>{scale} 0 0</v><v>0 {scale} 0</v><v>0 0 {scale}</v></varray></crystal>"
        f'<varray name="positions">{positions}</varray></structure>'
        "</modeling>"
    )


def contcar_text(scale: float = 3.0) -> str:
    return (
        "Li1\n1.0\n"
        f"  {scale} 0.0 0.0\n  0.0 {scale} 0.0\n  0.0 0.0 {scale}\n"
        "Li\n1\nDirect\n 0.0 0.0 0.0\n"
    )


def write_calculation_outputs(
    remote: Path,
    calc_id: str = "calc-0001",
    *,
    calculation_type: str = "static",
    converged: bool = True,
    frames: int = 1,
    nelm: int = 700,
    ionic_converged: bool = True,
    scale: float = 3.0,
    footer: bool = True,
) -> None:
    """Write one calculation's outputs under output/<calc_id>/."""

    directory = remote / "output" / calc_id
    directory.mkdir(parents=True, exist_ok=True)
    (remote / "logs").mkdir(parents=True, exist_ok=True)
    outcar = "vasp.5.4.4.18Apr17\n"
    if calculation_type == "relax" and ionic_converged:
        outcar += RELAX_MARKER + "\n"
    if footer:
        outcar += OUTCAR_FOOTER + "\n"
    (directory / "OUTCAR").write_text(outcar, encoding="utf-8")
    (directory / "OSZICAR").write_text(" 1 F= -.150000E+01 E0= -.150000E+01\n", encoding="utf-8")
    (directory / "vasprun.xml").write_text(
        vasprun_xml(
            nelm=nelm,
            frames=frames,
            electronic_steps=1 if converged else nelm,
            scale=scale,
        ),
        encoding="utf-8",
    )
    if calculation_type == "relax":
        (directory / "CONTCAR").write_text(contcar_text(scale), encoding="utf-8")
    for stream in ("stdout", "stderr"):
        (remote / "logs" / f"{calc_id}.{stream}").write_text("", encoding="utf-8")


def write_completion(
    remote: Path,
    project_id: str,
    node_id: str,
    attempt: int,
    calc_ids: tuple[str, ...] = ("calc-0001",),
    exit_codes: tuple[int, ...] | None = None,
) -> None:
    codes = exit_codes if exit_codes is not None else tuple(0 for _ in calc_ids)
    write_json(
        remote / "completion.json",
        {
            "schema_version": 1,
            "status": "COMPLETED",
            "exit_code": 0,
            "project_id": project_id,
            "node_id": node_id,
            "attempt": attempt,
            "calculations": [
                {"id": calc_id, "exit_code": code}
                for calc_id, code in zip(calc_ids, codes)
            ],
        },
    )


def write_vasp_outputs(remote: Path, *, converged: bool = True) -> None:
    """Single static calculation, the shape the first real smoke produced."""

    (remote / "logs").mkdir(parents=True, exist_ok=True)
    write_calculation_outputs(remote, "calc-0001", converged=converged)
    (remote / "logs" / "stdout.log").write_text("done\n", encoding="utf-8")
    (remote / "logs" / "stderr.log").write_text("", encoding="utf-8")


class RealVasprunParsingTests(unittest.TestCase):
    """Guards for two VASP 5.4.x layouts that a real cluster run exposed.

    Both defects were invisible to an idealised fixture and both fail silently
    in the wrong direction: one rejects every converged run, the other records
    an energy three orders of magnitude off as a training label.
    """

    def setUp(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "dft_labeling_probe", PLUGINS / "dft-labeling" / "adapter.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.parse = module._parse_scheduled_vasprun
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, remote: Path, *, converged: bool = True) -> Path:
        write_vasp_outputs(remote, converged=converged)
        return remote / "output" / "calc-0001" / "vasprun.xml"

    def test_nelm_comes_from_electronic_convergence_not_response_functions(self) -> None:
        parsed = self.parse(self._write(self.root / "run"))
        self.assertIsNotNone(parsed)
        self.assertEqual(700, parsed["nelm"])
        self.assertEqual(1, parsed["electronic_steps"])
        self.assertLess(parsed["electronic_steps"], parsed["nelm"])

    def test_energy_is_the_scstep_sigma_zero_value_not_the_entropy(self) -> None:
        parsed = self.parse(self._write(self.root / "run"))
        self.assertIsNotNone(parsed)
        self.assertAlmostEqual(-1.5, parsed["energy_ev"], places=10)
        self.assertNotAlmostEqual(-0.00081482, parsed["energy_ev"], places=8)

    def test_non_converged_run_is_still_detected(self) -> None:
        parsed = self.parse(self._write(self.root / "run", converged=False))
        self.assertIsNotNone(parsed)
        self.assertGreaterEqual(parsed["electronic_steps"], parsed["nelm"])

    def test_contcar_and_vasprun_lattices_compare_at_printed_precision(self) -> None:
        """CONTCAR carries ~16 digits, vasprun prints 8; the same cell differs.

        A real relaxation on a cluster produced 1.7141491381523322 in CONTCAR and
        1.71414914 in vasprun.xml.  The 1e-10 tolerance used when re-parsing one
        file twice rejects that, so the cross-format comparison has its own bound.
        """

        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "dft_labeling_lattice", PLUGINS / "dft-labeling" / "adapter.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        contcar = [[-1.7141491381523322, 1.7141491381523322, 1.7141491381523322]]
        vasprun = [[-1.71414914, 1.71414914, 1.71414914]]
        self.assertTrue(module._same_lattice_across_formats(contcar, vasprun))
        # The strict comparator is still strict; it is used for same-file checks.
        self.assertFalse(module._same_numeric_tree(contcar, vasprun))
        # A genuinely different cell must still be rejected.
        self.assertFalse(
            module._same_lattice_across_formats(contcar, [[-1.72, 1.72, 1.72]])
        )
        self.assertFalse(module._same_lattice_across_formats(contcar, [[1.0, 2.0]]))

    def test_energy_falls_back_when_scsteps_carry_no_energy_block(self) -> None:
        path = self.root / "bare.xml"
        path.write_text(
            "<modeling>"
            '<parameters><separator name="electronic convergence">'
            '<i type="int" name="NELM">60</i></separator></parameters>'
            '<atominfo><array name="atoms"><set><rc><c>Li</c></rc></set></array></atominfo>'
            '<structure name="finalpos"><crystal><varray name="basis">'
            "<v>3 0 0</v><v>0 3 0</v><v>0 0 3</v></varray></crystal>"
            '<varray name="positions"><v>0 0 0</v></varray></structure>'
            "<calculation><scstep/>"
            '<energy><i name="e_wo_entrp">-2.25</i>'
            '<i name="e_0_energy">-0.0009</i></energy>'
            '<varray name="forces"><v>0 0 0</v></varray>'
            '<varray name="stress"><v>1 0 0</v><v>0 1 0</v><v>0 0 1</v></varray>'
            "</calculation></modeling>",
            encoding="utf-8",
        )
        parsed = self.parse(path)
        self.assertIsNotNone(parsed)
        self.assertAlmostEqual(-2.25, parsed["energy_ev"], places=10)


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
        self.assertIn("input/calc-0001/POTCAR", staged_paths)
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

    def test_selected_partition_is_persisted_only_in_execution_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, site = prepared_fixture(root)
            site_value = json.loads(site.read_text(encoding="utf-8"))
            site_value["clusters"]["cluster-a"]["scheduler"] = {
                "partition_candidates": ["gpu3", "gpu2"]
            }
            write_json(site, site_value)
            initialize(root)
            project = load_project(root)
            library = FakeTemplateLibrary()
            plan = make_run_plan(project, "label-li", PLUGINS, site, library)
            remote_dir = plan["hpc_execution"]["workspace"]["run_dir"]
            routing = {
                "candidate_partitions": ["gpu3", "gpu2"],
                "observed_partition_availability": [
                    {
                        "partition": "gpu3",
                        "exists": True,
                        "state": "UP",
                        "schedulable": True,
                        "currently_available": True,
                        "observed_node_availability": {
                            "total": 1,
                            "healthy": 1,
                            "capable_for_request": 1,
                            "available_now": 1,
                            "busy_capable": 0,
                            "states": {"IDLE": 1},
                            "max_healthy_node_capacity": {
                                "cpus": 64,
                                "gpus": 4,
                                "memory_mib": 262144,
                            },
                        },
                        "reason": (
                            "a healthy capable node has the requested resources "
                            "available now"
                        ),
                    }
                ],
                "selected_partition": "gpu3",
                "selection_mode": "available-now",
                "snapshot_at": "2026-08-15T00:00:00Z",
                "snapshot_scope": (
                    "submission-time observation only; no node or resource was reserved"
                ),
            }
            with patch(
                "mlipflow.services.SshSlurmBackend.stage_workspace",
                return_value=remote_dir,
            ), patch(
                "mlipflow.services.SshSlurmBackend.submit",
                return_value=ExecutionResult(
                    0,
                    "Submitted batch job 77\n",
                    "",
                    "77",
                    routing,
                ),
            ) as submitted:
                run_node(
                    project,
                    "label-li",
                    PLUGINS,
                    plan["plan_digest"],
                    site,
                    library,
                )

            manifest = json.loads(
                (
                    root
                    / ".mlipflow/runs/label-li/attempt-1/run-manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual("gpu3", manifest["provenance"]["selected_partition"])
            self.assertEqual(
                ["gpu3", "gpu2"],
                manifest["provenance"]["candidate_partitions"],
            )
            self.assertNotIn("partition", project.raw["workflow"]["nodes"][0])
            self.assertNotIn("scheduler", project.raw["workflow"]["nodes"][0])
            self.assertEqual(
                ["gpu3", "gpu2"],
                submitted.call_args.kwargs["partition_candidates"],
            )

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
            write_completion(remote, project.project_id, "label-li", 1)
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
            self.assertFalse((attempt / "calc-0001" / "POTCAR").is_file())

    def test_scheduler_completed_but_scientific_check_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, _, site, _ = self._submit(root)
            remote = root / "fake-remote"
            write_vasp_outputs(remote, converged=False)
            write_completion(remote, project.project_id, "label-li", 1)
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
            write_completion(remote, project.project_id, "label-li", 999)
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
            write_completion(remote, project.project_id, "label-li", 1)
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
                        if item["remote_name"] == "calc-0001/POSCAR"
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
