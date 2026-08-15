from __future__ import annotations

import subprocess
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlipflow.backends import LocalBackend, SlurmBackend, SshSlurmBackend
from mlipflow.errors import BackendError


def slurm_snapshot(partitions: list[str], nodes: list[str]) -> str:
    return "\n".join(
        partitions + ["__MLIPFLOW_SCONTROL_NODES__"] + nodes
    ) + "\n"


def slurm_partition(name: str, state: str = "UP") -> str:
    return f"PartitionName={name} State={state} TotalNodes=1"


def slurm_node(
    name: str,
    partition: str,
    *,
    state: str = "IDLE",
    cpus: int = 32,
    allocated_cpus: int = 0,
    gpus: int = 0,
    allocated_gpus: int = 0,
    memory_mib: int = 65536,
    allocated_memory_mib: int = 0,
) -> str:
    return (
        f"NodeName={name} CPUTot={cpus} CPUAlloc={allocated_cpus} "
        f"RealMemory={memory_mib} AllocMem={allocated_memory_mib} State={state} "
        f"Partitions={partition} "
        f"CfgTRES=cpu={cpus},mem={memory_mib}M,gres/gpu={gpus} "
        f"AllocTRES=cpu={allocated_cpus},gres/gpu={allocated_gpus}"
    )


REQUEST = {"cpus": 8, "gpus": 0, "memory": "8G", "walltime": "01:00:00"}


class BackendTests(unittest.TestCase):
    def test_local_uses_argv_and_shell_false(self) -> None:
        completed = subprocess.CompletedProcess(["tool", "a;touch bad"], 0, "ok", "")
        with tempfile.TemporaryDirectory() as temporary, patch(
            "mlipflow.backends.subprocess.run", return_value=completed
        ) as mocked:
            with patch.dict(
                "mlipflow.backends.os.environ",
                {"PATH": "/usr/bin", "OPENAI_API_KEY": "must-not-leak"},
                clear=True,
            ):
                result = LocalBackend().run(["tool", "a;touch bad"], Path(temporary))
        self.assertEqual(result.returncode, 0)
        self.assertEqual(mocked.call_args.args[0], ["tool", "a;touch bad"])
        self.assertFalse(mocked.call_args.kwargs["shell"])
        self.assertEqual(mocked.call_args.kwargs["env"], {"PATH": "/usr/bin"})

    def test_slurm_requires_parsed_job_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "run.slurm"
            script.write_text("#!/bin/sh\n", encoding="utf-8")
            completed = subprocess.CompletedProcess(
                ["sbatch", str(script)], 0, "Submitted batch job 12345\n", ""
            )
            with patch("mlipflow.backends.subprocess.run", return_value=completed):
                result = SlurmBackend().submit(script, Path(temporary))
            self.assertEqual(result.job_id, "12345")

    def test_ssh_profile_and_fetch_path_reject_injection(self) -> None:
        with self.assertRaises(BackendError):
            SshSlurmBackend("host;touch-pwned")
        backend = SshSlurmBackend("safe-profile")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(BackendError), patch(
                "mlipflow.backends.subprocess.run", side_effect=AssertionError("scp invoked")
            ):
                backend.fetch("/safe/path;touch-pwned", Path(temporary) / "file")

    def test_local_backend_cannot_bypass_transport_or_embed_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch(
            "mlipflow.backends.subprocess.run", side_effect=AssertionError("process invoked")
        ):
            with self.assertRaises(BackendError):
                LocalBackend().run(["ssh", "cluster", "hostname"], Path(temporary))
            with self.assertRaises(BackendError):
                LocalBackend().run(["tool", "--api-key=secret"], Path(temporary))
            with self.assertRaises(BackendError):
                LocalBackend().run(
                    ["tool", "Authorization: Bearer secret-value"], Path(temporary)
                )
            with self.assertRaises(BackendError):
                LocalBackend().run(
                    ["tool", "https://user:password@example.invalid/data"], Path(temporary)
                )

    def test_ssh_stage_workspace_is_attempt_scoped_and_verifies_each_hash(self) -> None:
        backend = SshSlurmBackend("safe-profile")
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "POSCAR"
            source.write_text("Li\n", encoding="utf-8")
            digest = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
            responses = [
                subprocess.CompletedProcess(["ssh"], 0, "", ""),
                subprocess.CompletedProcess(["scp"], 0, "", ""),
                subprocess.CompletedProcess(
                    ["ssh"], 0, f"3\n{digest.removeprefix('sha256:')}  input/POSCAR\n", ""
                ),
            ]
            with patch("mlipflow.backends.subprocess.run", side_effect=responses) as invoked:
                remote = backend.stage_workspace(
                    "/work/project/node/attempt-0001",
                    [(source, "input/POSCAR", digest)],
                )
            self.assertEqual("/work/project/node/attempt-0001", remote)
            self.assertIn("mkdir --", invoked.call_args_list[0].args[0][-1])
            self.assertIn("/input", invoked.call_args_list[0].args[0][-1])
            self.assertIn("/output", invoked.call_args_list[0].args[0][-1])
            self.assertIn("/logs", invoked.call_args_list[0].args[0][-1])
            self.assertEqual("scp", invoked.call_args_list[1].args[0][0])

    def test_ssh_fetch_rejects_overwrite_and_remote_path_escape(self) -> None:
        backend = SshSlurmBackend("safe-profile")
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "OUTCAR"
            destination.write_text("existing", encoding="utf-8")
            with self.assertRaisesRegex(BackendError, "fresh"), patch(
                "mlipflow.backends.subprocess.run", side_effect=AssertionError("scp invoked")
            ):
                backend.fetch_from("mlipflow-run-1", "OUTCAR", destination)
            with self.assertRaises(BackendError):
                backend.inspect_file("../escape", "OUTCAR")

    def test_ssh_remote_template_read_is_bounded_and_parsed(self) -> None:
        backend = SshSlurmBackend("safe-profile")
        content = "#!/bin/bash\n# {{RUN_DIR}}\n"
        digest = hashlib.sha256(content.encode()).hexdigest()
        completed = subprocess.CompletedProcess(
            ["ssh"],
            0,
            f"{len(content.encode())}\n{digest}  slurm/cpu.sbatch\n{content}",
            "",
        )
        with patch("mlipflow.backends.subprocess.run", return_value=completed) as invoked:
            observed = backend.read_template(
                "/remote/templates", "slurm/cpu.sbatch"
            )
        self.assertEqual(content, observed["content"])
        self.assertEqual("sha256:" + digest, observed["sha256"])
        self.assertTrue(observed["root_exists"])
        self.assertFalse(invoked.call_args.kwargs["shell"])
        with self.assertRaises(BackendError), patch(
            "mlipflow.backends.subprocess.run", side_effect=AssertionError("ssh invoked")
        ):
            backend.read_template("/remote/templates", "../escape")

    def test_ssh_status_falls_back_to_sacct_when_squeue_no_longer_knows_job(self) -> None:
        backend = SshSlurmBackend("safe-profile")
        completed = subprocess.CompletedProcess(
            ["ssh"], 0, "FAILED|NonZeroExitCode\n", ""
        )
        with patch("mlipflow.backends.subprocess.run", return_value=completed) as invoked:
            status = backend.status("3704996")
        self.assertEqual("FAILED", status["state"])
        remote_command = invoked.call_args.args[0][-1]
        self.assertIn("squeue", remote_command)
        self.assertIn("sacct", remote_command)
        self.assertIn("|| true", remote_command)

    def test_ssh_status_falls_back_to_scontrol_when_accounting_is_unavailable(self) -> None:
        backend = SshSlurmBackend("safe-profile")
        completed = subprocess.CompletedProcess(
            ["ssh"], 0, "FAILED|NonZeroExitCode\n", ""
        )
        with patch("mlipflow.backends.subprocess.run", return_value=completed) as invoked:
            status = backend.status("893999")

        self.assertEqual(
            {"state": "FAILED", "detail": "NonZeroExitCode", "source": "remote"},
            status,
        )
        remote_command = invoked.call_args.args[0][-1]
        self.assertIn("sacct -n -X -j 893999", remote_command)
        self.assertIn("scontrol show job -o 893999", remote_command)
        self.assertIn("JobState=*", remote_command)
        self.assertIn("Reason=*", remote_command)

    def test_partition_routing_selects_first_available_and_injects_sbatch_argument(self) -> None:
        backend = SshSlurmBackend("safe-profile")
        snapshot = slurm_snapshot(
            [slurm_partition("preferred"), slurm_partition("fallback")],
            [
                slurm_node("node-a", "preferred"),
                slurm_node("node-b", "fallback"),
            ],
        )
        responses = [
            subprocess.CompletedProcess(["ssh"], 0, snapshot, ""),
            subprocess.CompletedProcess(
                ["ssh"], 0, "Submitted batch job 12345\n", ""
            ),
        ]
        with patch("mlipflow.backends.subprocess.run", side_effect=responses) as invoked:
            result = backend.submit(
                "submit.sbatch",
                "/work/project/node/attempt-0001",
                partition_candidates=["preferred", "fallback"],
                resources=REQUEST,
            )

        self.assertEqual("12345", result.job_id)
        self.assertEqual("preferred", result.submission_provenance["selected_partition"])
        self.assertEqual("available-now", result.submission_provenance["selection_mode"])
        self.assertIn(
            "sbatch --partition='preferred' -- 'submit.sbatch'",
            invoked.call_args_list[1].args[0][-1],
        )
        self.assertEqual("safe-profile", invoked.call_args_list[0].args[0][2])

    def test_partition_routing_prefers_later_available_over_busy_candidate(self) -> None:
        backend = SshSlurmBackend("safe-profile")
        snapshot = slurm_snapshot(
            [slurm_partition("preferred"), slurm_partition("fallback")],
            [
                slurm_node("node-a", "preferred", allocated_cpus=32),
                slurm_node("node-b", "fallback"),
            ],
        )
        completed = subprocess.CompletedProcess(["ssh"], 0, snapshot, "")
        with patch("mlipflow.backends.subprocess.run", return_value=completed):
            routing = backend.select_partition(["preferred", "fallback"], REQUEST)
        self.assertEqual("fallback", routing["selected_partition"])

    def test_partition_routing_skips_down_and_draining_candidates(self) -> None:
        backend = SshSlurmBackend("safe-profile")
        for unavailable_state in ("DOWN", "DRAIN"):
            with self.subTest(state=unavailable_state):
                snapshot = slurm_snapshot(
                    [
                        slurm_partition("preferred", unavailable_state),
                        slurm_partition("fallback"),
                    ],
                    [
                        slurm_node("node-a", "preferred"),
                        slurm_node("node-b", "fallback"),
                    ],
                )
                completed = subprocess.CompletedProcess(["ssh"], 0, snapshot, "")
                with patch("mlipflow.backends.subprocess.run", return_value=completed):
                    routing = backend.select_partition(
                        ["preferred", "fallback"], REQUEST
                    )
                self.assertEqual("fallback", routing["selected_partition"])

    def test_partition_routing_queues_on_preferred_when_all_healthy_candidates_busy(self) -> None:
        backend = SshSlurmBackend("safe-profile")
        snapshot = slurm_snapshot(
            [slurm_partition("preferred"), slurm_partition("fallback")],
            [
                slurm_node("node-a", "preferred", allocated_cpus=32),
                slurm_node("node-b", "fallback", allocated_cpus=32),
            ],
        )
        completed = subprocess.CompletedProcess(["ssh"], 0, snapshot, "")
        with patch("mlipflow.backends.subprocess.run", return_value=completed):
            routing = backend.select_partition(["preferred", "fallback"], REQUEST)
        self.assertEqual("preferred", routing["selected_partition"])
        self.assertEqual("healthy-queue", routing["selection_mode"])
        self.assertIn("no node or resource was reserved", routing["snapshot_scope"])

    def test_partition_routing_fails_when_all_candidates_are_unusable(self) -> None:
        backend = SshSlurmBackend("safe-profile")
        snapshot = slurm_snapshot(
            [slurm_partition("down", "DOWN")],
            [slurm_node("node-a", "down", state="DOWN")],
        )
        completed = subprocess.CompletedProcess(["ssh"], 0, snapshot, "")
        with patch("mlipflow.backends.subprocess.run", return_value=completed):
            with self.assertRaisesRegex(BackendError, "missing: partition does not exist"):
                backend.select_partition(["missing", "down"], REQUEST)

    def test_gpu_request_skips_partition_without_gpu_capability(self) -> None:
        backend = SshSlurmBackend("safe-profile")
        snapshot = slurm_snapshot(
            [slurm_partition("cpu"), slurm_partition("gpu")],
            [
                slurm_node("node-a", "cpu", gpus=0),
                slurm_node("node-b", "gpu", gpus=2),
            ],
        )
        requested = {**REQUEST, "gpus": 1}
        completed = subprocess.CompletedProcess(["ssh"], 0, snapshot, "")
        with patch("mlipflow.backends.subprocess.run", return_value=completed):
            routing = backend.select_partition(["cpu", "gpu"], requested)
        self.assertEqual("gpu", routing["selected_partition"])

    def test_cpu_request_skips_partition_without_single_node_capability(self) -> None:
        backend = SshSlurmBackend("safe-profile")
        snapshot = slurm_snapshot(
            [slurm_partition("small"), slurm_partition("large")],
            [
                slurm_node("node-a", "small", cpus=16),
                slurm_node("node-b", "large", cpus=64),
            ],
        )
        requested = {**REQUEST, "cpus": 32}
        completed = subprocess.CompletedProcess(["ssh"], 0, snapshot, "")
        with patch("mlipflow.backends.subprocess.run", return_value=completed):
            routing = backend.select_partition(["small", "large"], requested)
        self.assertEqual("large", routing["selected_partition"])
        first = routing["observed_partition_availability"][0]
        self.assertEqual(0, first["observed_node_availability"]["capable_for_request"])


if __name__ == "__main__":
    unittest.main()
