"""End-to-end contract tests for scheduled static / relax / AIMD / batch runs.

One node drives one attempt. Its 1..N VASP calculations run sequentially by
default or as explicitly approved independent scheduler jobs.
These tests exercise READY plans all the way to a
final MLIPipe state, because a BLOCKED plan proves nothing about staging,
fetching, or the scientific checks that decide whether a node may reach OK.

The remote side is simulated by writing the layout the site template produces
and by intercepting the four backend calls, so the whole lifecycle is covered
without a cluster.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from mlipipe.backends import ExecutionResult
from mlipipe.config import load_project
from mlipipe.services.commands import (
    advance,
    initialize,
    make_advance_plan,
    make_run_plan,
    run_node,
)
from mlipipe.services.queries import query_workflow

from .helpers import project_config, write_json
from .test_scheduled_dft import (
    FakeTemplateLibrary,
    write_calculation_outputs,
    write_completion,
    write_site,
)


INCAR_BY_TYPE = {
    "static": {"ENCUT": 450, "EDIFF": 5e-6, "NSW": 0, "IBRION": -1},
    "relax": {
        "ENCUT": 450,
        "EDIFF": 5e-6,
        "NSW": 8,
        "IBRION": 2,
        "ISIF": 3,
        "EDIFFG": -0.01,
    },
    "aimd": {
        "ENCUT": 450,
        "EDIFF": 5e-6,
        "IBRION": 0,
        "NSW": 4,
        "POTIM": 2.0,
        "TEBEG": 300,
        "TEEND": 300,
        "MDALGO": 2,
    },
}


def calc_id(index: int) -> str:
    return f"calc-{index + 1:04d}"


def build_project(
    root: Path,
    calculation_type: str = "static",
    count: int = 1,
    concurrency: int = 1,
) -> tuple[Path, list[str]]:
    """Create a project whose prepare manifest declares ``count`` calculations."""

    inputs = root / "inputs"
    prepared = root / "prepared"
    inputs.mkdir(parents=True)
    prepared.mkdir()
    structure_ids = [f"li-{index + 1}" for index in range(count)]

    sources = []
    for index, structure_id in enumerate(structure_ids):
        source = inputs / f"{structure_id}.vasp"
        scale = 3.0 + index * 0.1
        source.write_text(
            f"Li\n{scale}\n1 0 0\n0 1 0\n0 0 1\nLi\n1\nDirect\n0 0 0\n",
            encoding="utf-8",
        )
        sources.append(source)
    write_json(
        inputs / "structures.json",
        {
            "schema_version": 1,
            "structures": [
                {"id": sid, "path": src.name}
                for sid, src in zip(structure_ids, sources)
            ],
        },
    )
    write_json(
        inputs / "labeling.json",
        {
            "schema_version": 1,
            "engine": "vasp",
            "calculation_type": calculation_type,
            "incar": INCAR_BY_TYPE[calculation_type],
            "kpoints": {"mode": "gamma", "grid": [1, 1, 1], "shift": [0, 0, 0]},
        },
    )

    calculations = []
    for index, (structure_id, source) in enumerate(zip(structure_ids, sources)):
        directory = prepared / f"{index + 1:06d}-{structure_id}"
        directory.mkdir()
        contents = {
            "POSCAR": source.read_text(encoding="utf-8"),
            "INCAR": f"ENCUT = 450\n# {structure_id}\n",
            "KPOINTS": "mesh\n0\nGamma\n1 1 1\n0 0 0\n",
            "POTCAR": f"TEST-ONLY-POTCAR {structure_id}\n",
        }
        files: dict[str, Any] = {}
        for name, text in contents.items():
            path = directory / name
            path.write_text(text, encoding="utf-8")
            files[name] = {
                "path": f"{directory.name}/{name}",
                "collectable": name != "POTCAR",
            }
        calculations.append(
            {"structure_id": structure_id, "atom_count": 1, "files": files}
        )
    write_json(
        prepared / "dft-input-manifest.json",
        {
            "schema_version": 1,
            "plugin_id": "dft-labeling",
            "operation": "vasp-prepare",
            "status": "OK",
            "engine": "vasp",
            "calculation_type": calculation_type,
            "structure_count": count,
            "calculations": calculations,
        },
    )
    node = {
        "id": "label-li",
        "uses": "dft-labeling",
        "backend": "ssh-slurm",
        "backend_profile": "cluster-a",
        "inputs": {
            "structures_manifest": "inputs/structures.json",
            "labeling_config": "inputs/labeling.json",
            "dft_input_manifest": "prepared/dft-input-manifest.json",
        },
        "parameters": {
            "operation": "label",
            "engine": "vasp",
            "completion_policy": {
                "require_ionic_convergence": calculation_type == "relax"
            },
            "units": {
                "energy": "eV",
                "length": "angstrom",
                "force": "eV/angstrom",
                "stress": "kbar-vasp-3x3",
            },
            "result_manifest": "dft-labeling-result.json",
            "calculation_concurrency": concurrency,
        },
        "resources": {
            "cpus": 2 if concurrency > 1 else 4,
            "gpus": 0,
            "memory": "2G" if concurrency > 1 else "4G",
            "walltime": "00:05:00",
        },
    }
    write_json(root / "project.yaml", project_config([node]))
    return write_site(root), structure_ids


class ScheduledLifecycle:
    """Drive one node from plan to final state against a simulated cluster."""

    def __init__(
        self,
        root: Path,
        calculation_type: str = "static",
        count: int = 1,
        concurrency: int = 1,
    ):
        self.root = root
        self.calculation_type = calculation_type
        self.count = count
        self.concurrency = concurrency
        self.site, self.structure_ids = build_project(
            root, calculation_type, count, concurrency
        )
        self.remote = root / "fake-remote"
        self.staged: set[str] = set()
        initialize(root)
        self.project = load_project(root)

    def plan(self) -> dict[str, Any]:
        return make_run_plan(
            self.project, "label-li", self.site, FakeTemplateLibrary()
        )

    def submit(self) -> dict[str, Any]:
        plan = self.plan()

        def stage(remote_dir, files):
            for _, relative in files:
                self.staged.add(relative)
            return remote_dir

        submitted = iter(
            ExecutionResult(0, f"Submitted batch job {91 + index}\n", "", str(91 + index))
            for index in range(max(1, self.count if self.concurrency > 1 else 1))
        )
        with patch(
            "mlipipe.backends.SshSlurmBackend.stage_workspace", side_effect=stage
        ), patch(
            "mlipipe.backends.SshSlurmBackend.submit",
            side_effect=lambda *_args, **_kwargs: next(submitted),
        ):
            run_node(
                self.project,
                "label-li",
                True,
                self.site,
                FakeTemplateLibrary(),
            )
        return plan

    def write_remote_outputs(self, **overrides: Any) -> None:
        """Materialise the layout the site template is contracted to produce."""

        frames = overrides.pop("frames", None)
        if frames is None:
            frames = 4 if self.calculation_type == "aimd" else 1
        skip = overrides.pop("skip", ())
        unconverged = overrides.pop("unconverged_calcs", ())
        exit_codes = overrides.pop("exit_codes", None)
        for index in range(self.count):
            identifier = calc_id(index)
            if identifier in skip:
                continue
            remote = (
                self.remote / identifier if self.concurrency > 1 else self.remote
            )
            write_calculation_outputs(
                remote,
                identifier,
                calculation_type=self.calculation_type,
                converged=identifier not in unconverged,
                frames=frames,
                **overrides,
            )
        if self.concurrency > 1:
            for index in range(self.count):
                identifier = calc_id(index)
                code = None if exit_codes is None else (exit_codes[index],)
                write_completion(
                    self.remote / identifier,
                    self.project.project_id,
                    "label-li",
                    1,
                    (identifier,),
                    code,
                )
        else:
            write_completion(
                self.remote,
                self.project.project_id,
                "label-li",
                1,
                tuple(calc_id(index) for index in range(self.count)),
                exit_codes,
            )

    def _hooks(self):
        def inspect(_self, _cwd, remote_path):
            submission_id = str(_cwd).rsplit("/", 1)[-1]
            remote = (
                self.remote / submission_id
                if self.concurrency > 1
                else self.remote
            )
            path = remote / remote_path
            if not path.is_file():
                return {"path": remote_path, "exists": False}
            return {
                "path": remote_path,
                "exists": True,
                "size_bytes": path.stat().st_size,
            }

        def fetch(_self, _cwd, remote_path, destination):
            submission_id = str(_cwd).rsplit("/", 1)[-1]
            remote = (
                self.remote / submission_id
                if self.concurrency > 1
                else self.remote
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(remote / remote_path, destination)
            return destination

        return inspect, fetch

    def finish(self) -> dict[str, Any]:
        inspect, fetch = self._hooks()
        with patch(
            "mlipipe.backends.SshSlurmBackend.status",
            return_value={"state": "COMPLETED", "detail": None, "source": "fake"},
        ), patch(
            "mlipipe.backends.SshSlurmBackend.inspect_file", autospec=True,
            side_effect=inspect,
        ):
            make_advance_plan(self.project)
        with patch(
            "mlipipe.backends.SshSlurmBackend.status",
            return_value={"state": "COMPLETED", "detail": None, "source": "fake"},
        ), patch(
            "mlipipe.backends.SshSlurmBackend.inspect_file", autospec=True,
            side_effect=inspect,
        ), patch(
            "mlipipe.backends.SshSlurmBackend.fetch_from", autospec=True,
            side_effect=fetch,
        ):
            return advance(self.project)

    def run(self, **overrides: Any) -> dict[str, Any]:
        self.submit()
        self.write_remote_outputs(**overrides)
        return self.finish()

    @property
    def attempt(self) -> Path:
        return self.root / ".mlipipe" / "runs" / "label-li" / "attempt-1"

    def result(self) -> dict[str, Any]:
        return json.loads(
            (self.attempt / "dft-labeling-result.json").read_text(encoding="utf-8")
        )

    def labels(self) -> dict[str, Any]:
        return json.loads((self.attempt / "labels.json").read_text(encoding="utf-8"))


class TemporaryProjectTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def lifecycle(
        self,
        calculation_type: str = "static",
        count: int = 1,
        concurrency: int = 1,
    ):
        return ScheduledLifecycle(self.root, calculation_type, count, concurrency)


class ReadyPlanTests(TemporaryProjectTest):
    """All four supported shapes must produce an executable plan."""

    def test_single_static_is_ready(self) -> None:
        plan = self.lifecycle("static").plan()
        adapter = plan["adapter_plan"]
        self.assertEqual("READY", adapter["status"])
        self.assertTrue(adapter["executable"])
        self.assertEqual("static", adapter["calculation_type"])

    def test_single_relax_is_ready(self) -> None:
        adapter = self.lifecycle("relax").plan()["adapter_plan"]
        self.assertEqual("READY", adapter["status"])
        self.assertEqual("relax", adapter["calculation_type"])
        self.assertIn("CONTCAR", adapter["approval_summary"]["fetch_allowlist"])

    def test_single_aimd_is_ready(self) -> None:
        adapter = self.lifecycle("aimd").plan()["adapter_plan"]
        self.assertEqual("READY", adapter["status"])
        self.assertEqual("aimd", adapter["calculation_type"])
        self.assertIn("XDATCAR", adapter["approval_summary"]["fetch_allowlist"])

    def test_three_structure_static_batch_is_ready(self) -> None:
        adapter = self.lifecycle("static", 3).plan()["adapter_plan"]
        self.assertEqual("READY", adapter["status"])
        self.assertEqual(3, adapter["approval_summary"]["calculation_count"])
        self.assertEqual(
            ["calc-0001", "calc-0002", "calc-0003"],
            [item["id"] for item in adapter["calculations"]],
        )

    def test_four_structure_batch_plans_four_independent_resource_bound_jobs(self) -> None:
        plan = self.lifecycle("static", 4, concurrency=4).plan()
        adapter = plan["adapter_plan"]
        scheduled = adapter["scheduled_execution"]
        summary = adapter["approval_summary"]

        self.assertEqual(4, scheduled["schema_version"])
        self.assertEqual("independent-jobs", scheduled["submission_strategy"])
        self.assertEqual("vasp-batch", scheduled["template_family"])
        self.assertEqual(
            ["calc-0001", "calc-0002", "calc-0003", "calc-0004"],
            [item["id"] for item in scheduled["submissions"]],
        )
        for submission in scheduled["submissions"]:
            self.assertEqual(
                submission["id"],
                submission["template_variables"]["PLUGIN_CALCULATION_IDS"],
            )
            self.assertEqual(
                "2",
                submission["template_variables"][
                    "PLUGIN_MPI_RANKS_PER_CALCULATION"
                ],
            )
        self.assertEqual(4, summary["submitted_job_count"])
        self.assertEqual(2, summary["mpi_ranks_per_calculation"])
        self.assertEqual(8, summary["maximum_concurrent_mpi_ranks"])
        self.assertEqual(4, len(plan["hpc_executions"]))
        for item in plan["hpc_executions"]:
            execution = item["hpc_execution"]
            self.assertEqual(2, execution["resources"]["cpus"])
            self.assertIn("#SBATCH --ntasks=2", execution["rendered_scripts"]["submit.sbatch"])


class NestedPathTests(TemporaryProjectTest):
    def test_batch_staged_paths_do_not_collide(self) -> None:
        """Three POSCARs in one job used to be an outright name collision."""

        lifecycle = self.lifecycle("static", 3)
        staged = lifecycle.plan()["adapter_plan"]["scheduled_execution"]["staged_files"]
        names = [item["remote_name"] for item in staged]
        self.assertEqual(len(names), len(set(names)))
        for index in range(3):
            for name in ("POSCAR", "INCAR", "KPOINTS", "POTCAR"):
                self.assertIn(f"{calc_id(index)}/{name}", names)

    def test_nested_fetch_paths_are_per_calculation(self) -> None:
        """The core turns each declared output into a bounded nested path."""

        from mlipipe.services.contracts import _scheduled_contract

        lifecycle = self.lifecycle("static", 3)
        plan = lifecycle.plan()
        contract = _scheduled_contract(lifecycle.project, "dft-labeling", plan)
        remote_paths = {item["remote_path"] for item in contract["fetch_outputs"]}
        for index in range(3):
            self.assertIn(f"output/{calc_id(index)}/OUTCAR", remote_paths)
            self.assertIn(f"logs/{calc_id(index)}.stdout", remote_paths)
        self.assertIn("completion.json", remote_paths)
        local_names = {item["local_name"] for item in contract["fetch_outputs"]}
        self.assertIn("calc-0003/OUTCAR", local_names)

    def test_backend_creates_nested_parents_before_upload(self) -> None:
        """The adapter must never have to issue its own ssh/mkdir."""

        from mlipipe.backends import SshSlurmBackend

        source = self.root / "payload"
        source.write_text("x", encoding="utf-8")
        commands: list[list[str]] = []

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(argv, **_):
            commands.append(argv)
            return Result()

        backend = SshSlurmBackend("alias")
        with patch("mlipipe.backends.subprocess.run", side_effect=fake_run), patch.object(
            SshSlurmBackend,
            "inspect_file",
            return_value={"exists": True, "size_bytes": 1},
        ):
            backend.stage_workspace(
                "/work/p/n/attempt-0001",
                [(source, "input/calc-0002/POSCAR")],
            )
        mkdir = commands[0][-1]
        self.assertIn("input/calc-0002", mkdir)
        self.assertIn("mkdir -p", mkdir)
        self.assertLess(
            mkdir.index("attempt-0001'"), mkdir.index("input/calc-0002")
        )

    def test_unsafe_nested_names_are_rejected(self) -> None:
        """Nesting relaxed the name rule; it must not have relaxed the boundary."""

        from mlipipe.services.contracts import _safe_remote_relative

        for unsafe in (
            "../escape/POSCAR",
            "/etc/passwd",
            "calc-0001//POSCAR",
            "calc-0001/",
            "calc-0001/../../etc/passwd",
            "calc-0001/.hidden",
            "",
            "calc 0001/POSCAR",
            "calc-0001/POSCAR\n",
        ):
            with self.subTest(name=unsafe):
                self.assertFalse(_safe_remote_relative(unsafe))
        for safe in ("POSCAR", "calc-0001/POSCAR", "logs/calc-0001.stdout"):
            with self.subTest(name=safe):
                self.assertTrue(_safe_remote_relative(safe))

    def test_duplicate_full_remote_path_is_still_rejected(self) -> None:
        from mlipipe.errors import CapabilityError
        from mlipipe.services.contracts import _scheduled_contract

        lifecycle = self.lifecycle("static", 2)
        plan = lifecycle.plan()
        staged = plan["adapter_plan"]["scheduled_execution"]["staged_files"]
        staged.append(dict(staged[0]))
        with self.assertRaisesRegex(CapabilityError, "duplicate"):
            _scheduled_contract(lifecycle.project, "dft-labeling", plan)


class StaticLifecycleTests(TemporaryProjectTest):
    def test_single_static_reaches_ok_with_one_label(self) -> None:
        lifecycle = self.lifecycle("static")
        changed = lifecycle.run()
        self.assertEqual("OK", changed["changed"][0]["state"])
        result = lifecycle.result()
        self.assertEqual("static", result["calculation_type"])
        self.assertEqual(1, result["source_count"])
        self.assertEqual(1, result["label_count"])
        self.assertEqual(1, result["frame_count"])

    def test_batch_of_three_produces_three_labels(self) -> None:
        lifecycle = self.lifecycle("static", 3)
        changed = lifecycle.run()
        self.assertEqual("OK", changed["changed"][0]["state"])
        result = lifecycle.result()
        self.assertEqual(3, result["source_count"])
        self.assertEqual(3, result["calculation_count"])
        self.assertEqual(3, result["label_count"])
        records = lifecycle.labels()["records"]
        self.assertEqual(
            ["calc-0001", "calc-0002", "calc-0003"],
            [record["calculation_id"] for record in records],
        )
        self.assertEqual(
            lifecycle.structure_ids, [record["structure_id"] for record in records]
        )

    def test_four_independent_jobs_are_fetched_and_collected_together(self) -> None:
        lifecycle = self.lifecycle("static", 4, concurrency=4)
        changed = lifecycle.run()
        self.assertEqual("OK", changed["changed"][0]["state"])
        self.assertEqual(4, lifecycle.result()["label_count"])
        submissions = json.loads(
            (lifecycle.attempt / "scheduler-submissions.json").read_text(
                encoding="utf-8"
            )
        )["submissions"]
        self.assertEqual(["91", "92", "93", "94"], [item["job_id"] for item in submissions])
        self.assertTrue(all("submission_routing" not in item for item in submissions))
        self.assertTrue((lifecycle.attempt / "completion.json").is_file())

    def test_potcar_is_never_fetched_or_collected(self) -> None:
        lifecycle = self.lifecycle("static", 2)
        lifecycle.run()
        for index in range(2):
            self.assertFalse((lifecycle.attempt / calc_id(index) / "POTCAR").is_file())
        serialized = json.dumps(lifecycle.result())
        self.assertNotIn("POTCAR", serialized)

    def test_calculation_type_is_never_null(self) -> None:
        for calculation_type in ("static", "relax", "aimd"):
            with self.subTest(calculation_type=calculation_type):
                with tempfile.TemporaryDirectory() as temporary:
                    lifecycle = ScheduledLifecycle(Path(temporary), calculation_type)
                    lifecycle.run()
                    result = lifecycle.result()
                    self.assertEqual(calculation_type, result["calculation_type"])
                    for record in lifecycle.labels()["records"]:
                        self.assertEqual(calculation_type, record["calculation_type"])


class BatchFailureTests(TemporaryProjectTest):
    def test_missing_one_calculation_output_fails_the_node(self) -> None:
        lifecycle = self.lifecycle("static", 3)
        changed = lifecycle.run(skip=("calc-0002",))
        self.assertEqual("FAIL", changed["changed"][0]["state"])
        self.assertIn("calc-0002", str(changed["changed"][0]["diagnostic"]))

    def test_one_unconverged_calculation_fails_and_names_it(self) -> None:
        """Scheduler COMPLETED must not carry a scientifically failed batch."""

        lifecycle = self.lifecycle("static", 3)
        changed = lifecycle.run(unconverged_calcs=("calc-0002",))
        step = changed["changed"][0]
        self.assertEqual("FAIL", step["state"])
        diagnostic = str(step["diagnostic"])
        self.assertIn("calc-0002", diagnostic)
        self.assertIn("electronic", diagnostic)
        self.assertNotIn("calc-0001.completion.electronic", diagnostic)

    def test_nonzero_calculation_exit_code_fails(self) -> None:
        lifecycle = self.lifecycle("static", 2)
        changed = lifecycle.run(exit_codes=(0, 3))
        self.assertEqual("FAIL", changed["changed"][0]["state"])
        self.assertIn("calc-0002", str(changed["changed"][0]["diagnostic"]))

class RelaxTests(TemporaryProjectTest):
    def test_relax_converged_with_contcar_reaches_ok(self) -> None:
        lifecycle = self.lifecycle("relax")
        changed = lifecycle.run(frames=3)
        self.assertEqual("OK", changed["changed"][0]["state"])
        result = lifecycle.result()
        self.assertEqual("relax", result["calculation_type"])
        self.assertTrue(result["completion"]["ionic_convergence_required"])
        self.assertTrue(result["completion"]["ionic_converged"])
        # Only the final converged configuration is labelled.
        self.assertEqual(3, result["frame_count"])
        self.assertEqual(1, result["label_count"])
        self.assertEqual(3, lifecycle.labels()["records"][0]["ionic_step"])

    def test_relax_electronically_converged_but_ionically_not_fails(self) -> None:
        lifecycle = self.lifecycle("relax")
        changed = lifecycle.run(frames=3, ionic_converged=False)
        step = changed["changed"][0]
        self.assertEqual("FAIL", step["state"])
        self.assertIn("ionic", str(step["diagnostic"]))

    def test_relax_without_contcar_fails(self) -> None:
        lifecycle = self.lifecycle("relax")
        lifecycle.submit()
        lifecycle.write_remote_outputs(frames=2)
        (lifecycle.remote / "output" / "calc-0001" / "CONTCAR").unlink()
        changed = lifecycle.finish()
        self.assertEqual("FAIL", changed["changed"][0]["state"])
        self.assertIn("CONTCAR", str(changed["changed"][0]["diagnostic"]))


class AimdTests(TemporaryProjectTest):
    def test_aimd_complete_trajectory_yields_one_label_per_frame(self) -> None:
        lifecycle = self.lifecycle("aimd")
        changed = lifecycle.run(frames=4)
        self.assertEqual("OK", changed["changed"][0]["state"])
        result = lifecycle.result()
        self.assertEqual("aimd", result["calculation_type"])
        self.assertEqual(1, result["source_count"])
        self.assertEqual(4, result["frame_count"])
        self.assertEqual(4, result["label_count"])
        self.assertFalse(result["completion"]["ionic_convergence_required"])
        records = lifecycle.labels()["records"]
        self.assertEqual([1, 2, 3, 4], [record["ionic_step"] for record in records])
        # Frames must be distinct, not the final one repeated.
        self.assertEqual(4, len({record["energy_ev"] for record in records}))

        step = query_workflow(lifecycle.project, "label-li")["steps"][0]
        trajectories = [
            item for item in step["artifacts"] if item["role"] == "aimd-trajectory"
        ]
        self.assertEqual(1, len(trajectories))
        self.assertTrue(trajectories[0]["uri"].endswith("calc-0001/vasprun.xml"))

    def test_aimd_step_count_must_match_requested_nsw(self) -> None:
        lifecycle = self.lifecycle("aimd")
        changed = lifecycle.run(frames=2)
        step = changed["changed"][0]
        self.assertEqual("FAIL", step["state"])
        diagnostic = str(step["diagnostic"])
        self.assertIn("NSW", diagnostic)

    def test_aimd_single_unconverged_scf_frame_fails(self) -> None:
        lifecycle = self.lifecycle("aimd")
        changed = lifecycle.run(frames=4, unconverged_calcs=("calc-0001",))
        step = changed["changed"][0]
        self.assertEqual("FAIL", step["state"])
        self.assertIn("electronic", str(step["diagnostic"]))

    def test_aimd_labels_carry_full_traceability(self) -> None:
        lifecycle = self.lifecycle("aimd")
        lifecycle.run(frames=4)
        for record in lifecycle.labels()["records"]:
            for key in (
                "structure_id",
                "calculation_id",
                "calculation_type",
                "ionic_step",
                "energy_ev",
                "forces_ev_per_angstrom",
                "lattice_angstrom",
                "species",
            ):
                self.assertIn(key, record)


if __name__ == "__main__":
    unittest.main()
