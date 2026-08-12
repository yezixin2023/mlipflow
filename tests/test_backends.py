from __future__ import annotations

import subprocess
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlipflow.backends import LocalBackend, SlurmBackend, SshSlurmBackend
from mlipflow.errors import BackendError


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


if __name__ == "__main__":
    unittest.main()
