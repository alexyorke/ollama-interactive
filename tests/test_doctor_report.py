from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import doctor_report


class DoctorReportTests(unittest.TestCase):
    def test_build_doctor_report_records_green_command(self) -> None:
        command = ["python", "-m", "ollama_code", "--doctor", "--quiet"]

        def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            if args[:1] == ["git"]:
                if args[1:] == ["rev-parse", "--short", "HEAD"]:
                    return subprocess.CompletedProcess(args, 0, stdout="abc123\n", stderr="")
                if args[1:] == ["status", "--short"]:
                    return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            return subprocess.CompletedProcess(args, 0, stdout="Ollama Code doctor\nmodel: ok granite4.1:8b\n", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(doctor_report.subprocess, "run", side_effect=fake_run):
                payload = doctor_report.build_doctor_report(Path(tmp), command=command, now_iso="2026-07-03T00:00:00+00:00")

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["status"], "pass")
        self.assertEqual(payload["git_commit"], "abc123")
        self.assertFalse(payload["git_dirty"])
        self.assertEqual(payload["summary"], "Ollama Code doctor")

    def test_build_doctor_report_records_timeout_as_failure(self) -> None:
        command = ["python", "-m", "ollama_code", "--doctor", "--quiet"]
        timeout = subprocess.TimeoutExpired(command, timeout=1.0, output="partial", stderr="busy")

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(doctor_report.subprocess, "run", side_effect=timeout):
                payload = doctor_report.build_doctor_report(Path(tmp), command=command, timeout_s=1.0)

        self.assertFalse(payload["ok"])
        self.assertEqual(payload["status"], "fail")
        self.assertEqual(payload["returncode"], 124)
        self.assertTrue(payload["timed_out"])
        self.assertIn("timed out", payload["stderr"])


if __name__ == "__main__":
    unittest.main()
