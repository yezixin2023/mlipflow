from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mlipflow.errors import ConfigError
from mlipflow.hpc import (
    resolve_hpc_execution_plan,
    remote_attempt_workspace,
    render_template,
    validate_hpc_resources,
    validate_slurm_cpu_semantics,
)
from mlipflow.site import load_site_config

from .helpers import write_json


MPI_SUBMIT = """#!/bin/bash
# {{PROJECT_ID}} {{NODE_ID}} {{ATTEMPT}}
#SBATCH --ntasks={{CPUS}}
#SBATCH --cpus-per-task=1
# gpus={{GPUS}} memory={{MEMORY}} time={{WALLTIME}}
# logs={{LOG_DIR}}
cd {{RUN_DIR}}
"""
SINGLE_PYTHON_SUBMIT = """#!/bin/bash
# {{PROJECT_ID}} {{NODE_ID}} {{ATTEMPT}}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={{CPUS}}
# gpus={{GPUS}} memory={{MEMORY}} time={{WALLTIME}}
# logs={{LOG_DIR}}
cd {{RUN_DIR}}
"""
RUN = """#!/bin/bash
# {{INPUT_DIR}} -> {{OUTPUT_DIR}}
cd {{RUN_DIR}}
"""


class FakeLibrary:
    def __init__(
        self, values: dict[str, str] | None = None, *, root_exists: bool = True
    ):
        self.values = values or {
            "slurm/mpi/cpu.sbatch": MPI_SUBMIT,
            "slurm/mpi/gpu.sbatch": MPI_SUBMIT,
            "slurm/single-python/cpu.sbatch": SINGLE_PYTHON_SUBMIT,
            "slurm/single-python/gpu.sbatch": SINGLE_PYTHON_SUBMIT,
            "vasp/run.sh": RUN,
        }
        self.root_exists = root_exists
        self.reads: list[tuple[str, str]] = []

    def read_template(self, root: str, relative: str):
        self.reads.append((root, relative))
        if not self.root_exists:
            return {"relative_path": relative, "root_exists": False, "exists": False}
        content = self.values.get(relative)
        if content is None:
            return {"relative_path": relative, "exists": False}
        return {"content": content}


def site_file(root: Path) -> Path:
    path = root / "site.yaml"
    write_json(
        path,
        {
            "schema_version": 1,
            "clusters": {
                "cluster-a": {
                    "backend": "ssh-slurm",
                    "ssh_profile": "alpha-login",
                    "remote_template_root": "/srv/templates/a",
                    "work_root": "/scratch/runs/a",
                },
                "cluster-b": {
                    "backend": "ssh-slurm",
                    "ssh_profile": "beta-login",
                    "remote_template_root": "/srv/templates/b",
                    "work_root": "/scratch/runs/b",
                },
            },
        },
    )
    return path


class HpcArchitectureTests(unittest.TestCase):
    def test_checked_in_site_examples_preserve_execution_model_cpu_semantics(self) -> None:
        examples = Path(__file__).resolve().parents[1] / "examples" / "site_templates" / "slurm"
        for execution_model in ("single-python", "mpi"):
            for device in ("cpu", "gpu"):
                template = examples / execution_model / f"{device}.sbatch.example"
                self.assertTrue(template.is_file(), template)
                validate_slurm_cpu_semantics(
                    template.read_text(encoding="utf-8"),
                    execution_model,
                    template_name=str(template),
                )

    def test_multi_cluster_site_config_and_profile_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            site = load_site_config(site_file(Path(temporary)))
        self.assertEqual({"cluster-a", "cluster-b"}, set(site.clusters))
        selected = site.cluster("cluster-b")
        self.assertEqual("beta-login", selected.ssh_profile)
        self.assertEqual("/srv/templates/b", selected.remote_template_root)
        self.assertEqual("/scratch/runs/b", selected.work_root)
        with self.assertRaisesRegex(ConfigError, "no cluster profile"):
            site.cluster("missing")

    def test_cluster_profiles_keep_distinct_site_owned_partition_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "site.yaml"
            write_json(
                path,
                {
                    "schema_version": 1,
                    "clusters": {
                        "cluster-a": {
                            "backend": "ssh-slurm",
                            "ssh_profile": "alpha-login",
                            "remote_template_root": "/srv/templates/a",
                            "work_root": "/scratch/runs/a",
                            "scheduler": {
                                "partition_candidates": ["cpu-fast", "cpu-slow"]
                            },
                        },
                        "cluster-b": {
                            "backend": "ssh-slurm",
                            "ssh_profile": "beta-login",
                            "remote_template_root": "/srv/templates/b",
                            "work_root": "/scratch/runs/b",
                            "scheduler": {
                                "partition_candidates": ["gpu3", "gpu2", "gpu1"],
                                "memory_constraint": "unreported",
                            },
                        },
                    },
                },
            )
            site = load_site_config(path)

        self.assertEqual(
            ["cpu-fast", "cpu-slow"],
            site.cluster("cluster-a").to_plan_dict()["scheduler"][
                "partition_candidates"
            ],
        )
        self.assertEqual(
            ["gpu3", "gpu2", "gpu1"],
            site.cluster("cluster-b").to_plan_dict()["scheduler"][
                "partition_candidates"
            ],
        )
        self.assertEqual(
            "reported",
            site.cluster("cluster-a").scheduler.memory_constraint,
        )
        self.assertEqual(
            "unreported",
            site.cluster("cluster-b").scheduler.memory_constraint,
        )

    def test_partition_candidates_reject_duplicates_and_unsafe_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "site.yaml"
            base = {
                "backend": "ssh-slurm",
                "ssh_profile": "alpha-login",
                "remote_template_root": "/srv/templates/a",
                "work_root": "/scratch/runs/a",
            }
            for candidates in (["same", "same"], ["safe", "bad;partition"]):
                with self.subTest(candidates=candidates):
                    write_json(
                        path,
                        {
                            "schema_version": 1,
                            "clusters": {
                                "cluster-a": {
                                    **base,
                                    "scheduler": {
                                        "partition_candidates": candidates
                                    },
                                }
                            },
                        },
                    )
                    with self.assertRaisesRegex(ConfigError, "partition_candidates"):
                        load_site_config(path)

    def test_missing_site_and_invalid_roots_fail_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ConfigError, "site config is missing"):
                load_site_config(root / "missing.yaml")
            real = site_file(root)
            linked = root / "linked-site.yaml"
            linked.symlink_to(real)
            self.assertEqual("cluster-a", load_site_config(linked).cluster("cluster-a").name)
            path = root / "site.yaml"
            write_json(
                path,
                {
                    "schema_version": 1,
                    "clusters": {
                        "bad": {
                            "backend": "ssh-slurm",
                            "ssh_profile": "bad",
                            "remote_template_root": "/same",
                            "work_root": "/same",
                        }
                    },
                },
            )
            with self.assertRaisesRegex(ConfigError, "must be disjoint"):
                load_site_config(path)
            write_json(
                path,
                {
                    "schema_version": 1,
                    "clusters": {
                        "bad": {
                            "backend": "ssh-slurm",
                            "ssh_profile": "bad",
                            "remote_template_root": "/site/templates",
                            "work_root": "/site/templates/runs",
                        }
                    },
                },
            )
            with self.assertRaisesRegex(ConfigError, "must be disjoint"):
                load_site_config(path)

    def test_remote_workspace_uses_persisted_attempt_number(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = load_site_config(site_file(Path(temporary))).cluster("cluster-a")
        first = remote_attempt_workspace(profile, "project-x", "node-y", 1)
        twelfth = remote_attempt_workspace(profile, "project-x", "node-y", 12)
        self.assertEqual(
            "/scratch/runs/a/project-x/node-y/attempt-0001", first["run_dir"]
        )
        self.assertEqual(
            "/scratch/runs/a/project-x/node-y/attempt-0012", twelfth["run_dir"]
        )
        self.assertEqual(first["run_dir"] + "/input", first["input_dir"])
        self.assertEqual(first["run_dir"] + "/output", first["output_dir"])
        self.assertEqual(first["run_dir"] + "/logs", first["log_dir"])

    def test_cpu_gpu_selection_and_deterministic_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = load_site_config(site_file(Path(temporary))).cluster("cluster-a")
        cpu_library = FakeLibrary()
        cpu = resolve_hpc_execution_plan(
            profile=profile,
            project_id="project-x",
            node_id="node-y",
            attempt=2,
            resources_value={
                "cpus": 16,
                "gpus": 0,
                "memory": "64G",
                "walltime": "04:00:00",
            },
            scheduled_execution={
                "template_family": "vasp",
                "execution_model": "mpi",
            },
            library=cpu_library,
        )
        self.assertIn(
            ("/srv/templates/a", "slurm/mpi/cpu.sbatch"), cpu_library.reads
        )
        self.assertIn(
            "#SBATCH --ntasks=16",
            cpu["rendered_scripts"]["submit.sbatch"],
        )
        self.assertIn(
            "# gpus=0 memory=64G time=04:00:00",
            cpu["rendered_scripts"]["submit.sbatch"],
        )
        repeated = resolve_hpc_execution_plan(
            profile=profile,
            project_id="project-x",
            node_id="node-y",
            attempt=2,
            resources_value=cpu["resources"],
            scheduled_execution={
                "template_family": "vasp",
                "execution_model": "mpi",
            },
            library=FakeLibrary(),
        )
        self.assertEqual(cpu, repeated)
        gpu_library = FakeLibrary()
        resolve_hpc_execution_plan(
            profile=profile,
            project_id="project-x",
            node_id="node-y",
            attempt=2,
            resources_value={
                "cpus": 16,
                "gpus": 2,
                "memory": "64G",
                "walltime": "04:00:00",
            },
            scheduled_execution={
                "template_family": "vasp",
                "execution_model": "mpi",
            },
            library=gpu_library,
        )
        self.assertIn(
            ("/srv/templates/a", "slurm/mpi/gpu.sbatch"), gpu_library.reads
        )

    def test_v3_execution_models_bind_distinct_cpu_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = load_site_config(site_file(Path(temporary))).cluster("cluster-a")
        resources = {
            "cpus": 16,
            "gpus": 0,
            "memory": "64G",
            "walltime": "04:00:00",
        }
        single_library = FakeLibrary()
        single = resolve_hpc_execution_plan(
            profile=profile,
            project_id="project-x",
            node_id="python-node",
            attempt=1,
            resources_value=resources,
            scheduled_execution={
                "schema_version": 3,
                "execution_model": "single-python",
                "template_family": "vasp",
            },
            library=single_library,
        )
        self.assertIn(
            ("/srv/templates/a", "slurm/single-python/cpu.sbatch"),
            single_library.reads,
        )
        single_submit = single["rendered_scripts"]["submit.sbatch"]
        self.assertIn("#SBATCH --ntasks=1", single_submit)
        self.assertIn("#SBATCH --cpus-per-task=16", single_submit)
        self.assertEqual("threads-per-process", single["cpu_resource_semantics"]["cpus_meaning"])

        mpi_library = FakeLibrary()
        mpi = resolve_hpc_execution_plan(
            profile=profile,
            project_id="project-x",
            node_id="mpi-node",
            attempt=1,
            resources_value=resources,
            scheduled_execution={
                "schema_version": 3,
                "execution_model": "mpi",
                "template_family": "vasp",
            },
            library=mpi_library,
        )
        self.assertIn(
            ("/srv/templates/a", "slurm/mpi/cpu.sbatch"), mpi_library.reads
        )
        mpi_submit = mpi["rendered_scripts"]["submit.sbatch"]
        self.assertIn("#SBATCH --ntasks=16", mpi_submit)
        self.assertIn("#SBATCH --cpus-per-task=1", mpi_submit)
        self.assertEqual("mpi-task-count", mpi["cpu_resource_semantics"]["cpus_meaning"])

    def test_single_python_rejects_cpus_mapped_to_ntasks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = load_site_config(site_file(Path(temporary))).cluster("cluster-a")
        bad = FakeLibrary(
            {
                "slurm/single-python/cpu.sbatch": MPI_SUBMIT,
                "vasp/run.sh": RUN,
            }
        )
        with self.assertRaisesRegex(
            ConfigError, "--ntasks=1 and --cpus-per-task=\\{\\{CPUS\\}\\}"
        ):
            resolve_hpc_execution_plan(
                profile=profile,
                project_id="project-x",
                node_id="python-node",
                attempt=1,
                resources_value={
                    "cpus": 64,
                    "gpus": 0,
                    "memory": "128G",
                    "walltime": "01:00:00",
                },
                scheduled_execution={
                    "schema_version": 3,
                    "execution_model": "single-python",
                    "template_family": "vasp",
                },
                library=bad,
            )

    def test_mpi_rejects_cpus_mapped_to_cpus_per_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = load_site_config(site_file(Path(temporary))).cluster("cluster-a")
        bad = FakeLibrary(
            {
                "slurm/mpi/cpu.sbatch": SINGLE_PYTHON_SUBMIT,
                "vasp/run.sh": RUN,
            }
        )
        with self.assertRaisesRegex(ConfigError, "--ntasks=\\{\\{CPUS\\}\\}"):
            resolve_hpc_execution_plan(
                profile=profile,
                project_id="project-x",
                node_id="mpi-node",
                attempt=1,
                resources_value={
                    "cpus": 64,
                    "gpus": 0,
                    "memory": "128G",
                    "walltime": "01:00:00",
                },
                scheduled_execution={
                    "schema_version": 3,
                    "execution_model": "mpi",
                    "template_family": "vasp",
                },
                library=bad,
            )

    def test_missing_or_incomplete_template_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = load_site_config(site_file(Path(temporary))).cluster("cluster-a")
        arguments = {
            "profile": profile,
            "project_id": "p",
            "node_id": "n",
            "attempt": 1,
            "resources_value": {
                "cpus": 1,
                "gpus": 0,
                "memory": "1G",
                "walltime": "00:01:00",
            },
            "scheduled_execution": {
                "template_family": "vasp",
                "execution_model": "mpi",
            },
        }
        with self.assertRaisesRegex(ConfigError, "required remote template is missing"):
            resolve_hpc_execution_plan(
                **arguments,
                library=FakeLibrary({"slurm/mpi/cpu.sbatch": MPI_SUBMIT}),
            )
        with self.assertRaisesRegex(ConfigError, "template root is missing"):
            resolve_hpc_execution_plan(
                **arguments, library=FakeLibrary(root_exists=False)
            )
        incomplete = FakeLibrary(
            {
                "slurm/mpi/cpu.sbatch": (
                    "#!/bin/bash\n#SBATCH --ntasks={{CPUS}}\ncd {{RUN_DIR}}\n"
                ),
                "vasp/run.sh": RUN,
            }
        )
        with self.assertRaisesRegex(ConfigError, "lacks required variables"):
            resolve_hpc_execution_plan(**arguments, library=incomplete)
        with self.assertRaisesRegex(ConfigError, "unknown variables"):
            render_template(
                "{{RUN_DIR}} {{ARBITRARY_CODE}}",
                {"RUN_DIR": "/work"},
                template_name="bad",
                required=frozenset({"RUN_DIR"}),
            )

    def _render(self, text: str) -> str:
        return render_template(
            text,
            {"RUN_DIR": "/work/attempt-0001", "ATTEMPT": "0001"},
            template_name="probe",
            required=frozenset({"RUN_DIR"}),
        )

    def test_ordinary_shell_and_json_braces_are_not_template_syntax(self) -> None:
        """A real VASP/JSON template must be writable.

        The renderer used to reject any surviving ``{{`` or ``}}``, which made
        these ordinary constructs impossible: a shell expansion immediately
        before a literal brace, and a JSON object closing after a nested one.
        """

        cases = {
            "brace expansion then json close": (
                'cd {{RUN_DIR}}\n'
                'printf \'{"attempt": %s}\\n\' "${value}"\n'
                'cat <<EOF\n{"a": "${value}"}\nEOF\n'
            ),
            "nested json object": 'cd {{RUN_DIR}}\necho \'{"outer":{"inner":1}}\'\n',
            "expansion directly before closing brace": (
                'cd {{RUN_DIR}}\necho "{\\"n\\": ${count}}"\n'
            ),
            "awk field program": "cd {{RUN_DIR}}\nawk '{print $1}' values.txt\n",
            "arithmetic base ten": "cd {{RUN_DIR}}\nattempt=$((10#{{ATTEMPT}}))\n",
        }
        for name, text in cases.items():
            with self.subTest(case=name):
                rendered = self._render(text)
                self.assertIn("/work/attempt-0001", rendered)
                self.assertNotIn("{{RUN_DIR}}", rendered)

    def test_residual_placeholder_syntax_is_still_rejected(self) -> None:
        """Anything that tried to be a placeholder and was not recognised."""

        cases = {
            "lowercase name": "cd {{RUN_DIR}}\necho {{run_dir}}\n",
            "padded name": "cd {{RUN_DIR}}\necho {{ RUN_DIR }}\n",
            "jinja filter": "cd {{RUN_DIR}}\necho {{RUN_DIR|upper}}\n",
            "empty placeholder": "cd {{RUN_DIR}}\necho {{}}\n",
            "dotted path": "cd {{RUN_DIR}}\necho {{node.id}}\n",
        }
        for name, text in cases.items():
            with self.subTest(case=name):
                with self.assertRaisesRegex(ConfigError, "unsupported template syntax"):
                    self._render(text)

    def test_rejection_message_quotes_the_offending_fragment(self) -> None:
        with self.assertRaisesRegex(ConfigError, r"\{\{ RUN_DIR \}\}"):
            self._render("cd {{RUN_DIR}}\necho {{ RUN_DIR }}\n")

    def test_unknown_uppercase_placeholder_is_still_caught_first(self) -> None:
        """The narrower residual check must not weaken the explicit checks."""

        with self.assertRaisesRegex(ConfigError, "unknown variables"):
            self._render("cd {{RUN_DIR}}\necho {{ARBITRARY_CODE}}\n")

    def test_missing_resource_requirement_fails(self) -> None:
        with self.assertRaisesRegex(ConfigError, "resources lack"):
            validate_hpc_resources({"cpus": 4, "gpus": 0, "memory": "8G"})
        with self.assertRaisesRegex(ConfigError, "site-specific"):
            validate_hpc_resources(
                {
                    "cpus": 4,
                    "gpus": 0,
                    "memory": "8G",
                    "walltime": "01:00:00",
                    "partition": "somewhere",
                }
            )

    def test_unlimited_walltime_is_valid_and_rendered_verbatim(self) -> None:
        resources = validate_hpc_resources(
            {
                "cpus": 128,
                "gpus": 0,
                "memory": "128G",
                "walltime": "UNLIMITED",
            }
        )
        rendered = render_template(
            "#SBATCH --time={{WALLTIME}}\ncd {{RUN_DIR}}\n",
            {"RUN_DIR": "/work/attempt-0001", "WALLTIME": resources.walltime},
            template_name="probe",
            required=frozenset({"RUN_DIR", "WALLTIME"}),
        )

        self.assertIn("#SBATCH --time=UNLIMITED", rendered)

if __name__ == "__main__":
    unittest.main()
