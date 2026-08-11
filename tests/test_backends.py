from __future__ import annotations

import subprocess
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


if __name__ == "__main__":
    unittest.main()
