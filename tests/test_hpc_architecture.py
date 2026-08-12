from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from mlipflow.errors import ConfigError
from mlipflow.hpc import (
    resolve_hpc_execution_plan,
    remote_attempt_workspace,
    render_template,
    validate_hpc_resources,
)
from mlipflow.site import load_site_config

from .helpers import write_json


SUBMIT = """#!/bin/bash
# {{PROJECT_ID}} {{NODE_ID}} {{ATTEMPT}}
# cpus={{CPUS}} gpus={{GPUS}} memory={{MEMORY}} time={{WALLTIME}}
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
            "slurm/cpu.sbatch": SUBMIT,
            "slurm/gpu.sbatch": SUBMIT,
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
        payload = content.encode("utf-8")
        return {
            "content": content,
            "size_bytes": len(payload),
            "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        }


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

    def test_missing_site_and_invalid_roots_fail_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ConfigError, "site config is missing"):
                load_site_config(root / "missing.yaml")
            real = site_file(root)
            linked = root / "linked-site.yaml"
            linked.symlink_to(real)
            with self.assertRaisesRegex(ConfigError, "missing or unsafe"):
                load_site_config(linked)
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
            scheduled_execution={"template_family": "vasp"},
            library=cpu_library,
        )
        self.assertIn(
            ("/srv/templates/a", "slurm/cpu.sbatch"), cpu_library.reads
        )
        self.assertIn("# cpus=16 gpus=0 memory=64G time=04:00:00", cpu["rendered_scripts"]["submit.sbatch"]["content"])
        repeated = resolve_hpc_execution_plan(
            profile=profile,
            project_id="project-x",
            node_id="node-y",
            attempt=2,
            resources_value=cpu["resources"],
            scheduled_execution={"template_family": "vasp"},
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
            scheduled_execution={"template_family": "vasp"},
            library=gpu_library,
        )
        self.assertIn(
            ("/srv/templates/a", "slurm/gpu.sbatch"), gpu_library.reads
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
            "scheduled_execution": {"template_family": "vasp"},
        }
        with self.assertRaisesRegex(ConfigError, "required remote template is missing"):
            resolve_hpc_execution_plan(**arguments, library=FakeLibrary({"slurm/cpu.sbatch": SUBMIT}))
        with self.assertRaisesRegex(ConfigError, "template root is missing"):
            resolve_hpc_execution_plan(
                **arguments, library=FakeLibrary(root_exists=False)
            )
        incomplete = FakeLibrary(
            {"slurm/cpu.sbatch": "#!/bin/bash\ncd {{RUN_DIR}}\n", "vasp/run.sh": RUN}
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

    def test_execution_implementation_has_no_cluster_specific_launch_knowledge(self) -> None:
        root = Path(__file__).resolve().parents[1]
        sources = list((root / "src/mlipflow").glob("*.py")) + [
            root / "plugins/dft-labeling/adapter.py"
        ]
        forbidden = (
            "/public/software",
            "module load",
            "setvars.sh",
            "vasp_std",
            "cpu192",
            "mpirun ",
            "srun ",
        )
        for path in sources:
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, f"site launch knowledge leaked into {path}")


if __name__ == "__main__":
    unittest.main()
