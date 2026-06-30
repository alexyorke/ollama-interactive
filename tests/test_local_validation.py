import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import local_validation


class LocalValidationTests(unittest.TestCase):
    def test_auto_runner_prefers_pytest_with_xdist_for_agent(self) -> None:
        with patch.object(local_validation, "_has_module", side_effect=lambda name: name in {"pytest", "xdist"}):
            commands = local_validation._tier_commands("agent", runner="auto", jobs="auto")

        self.assertEqual(len(commands), 1)
        name, command = commands[0]
        self.assertEqual(name, "agent")
        self.assertEqual(command[:6], [sys.executable, "-m", "pytest", "-q", "-n", "auto"])
        self.assertIn("tests/test_tools.py", command)

    def test_auto_runner_falls_back_to_unittest_when_pytest_missing(self) -> None:
        with patch.object(local_validation, "_has_module", return_value=False):
            commands = local_validation._tier_commands("smoke", runner="auto", jobs="auto")

        self.assertEqual(len(commands), 1)
        name, command = commands[0]
        self.assertEqual(name, "smoke")
        self.assertEqual(command[:3], [sys.executable, "-m", "unittest"])
        self.assertIn("tests.test_live_model_gate", command)

    def test_jobs_off_disables_xdist(self) -> None:
        with patch.object(local_validation, "_has_module", side_effect=lambda name: name in {"pytest", "xdist"}):
            commands = local_validation._tier_commands("agent", runner="pytest", jobs="off")

        _name, command = commands[0]
        self.assertEqual(command[:4], [sys.executable, "-m", "pytest", "-q"])
        self.assertNotIn("-n", command)

    def test_full_tier_prefers_pytest_broad_final_gate(self) -> None:
        with patch.object(local_validation, "_has_module", side_effect=lambda name: name in {"pytest", "xdist"}):
            commands = local_validation._tier_commands("full", runner="auto", jobs="auto")

        self.assertEqual([name for name, _ in commands], ["smoke", "agent", "full-suite"])
        self.assertEqual(commands[-1][1][:6], [sys.executable, "-m", "pytest", "-q", "-n", "auto"])
        self.assertEqual(commands[-1][1][-1], "tests")

    def test_full_tier_unittest_runner_keeps_discover_final_gate(self) -> None:
        with patch.object(local_validation, "_has_module", return_value=False):
            commands = local_validation._tier_commands("full", runner="auto", jobs="auto")

        self.assertEqual([name for name, _ in commands], ["smoke", "agent", "full-discover"])
        self.assertEqual(commands[-1][1], [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])

    def test_run_records_runner_jobs_and_target_count(self) -> None:
        command = [sys.executable, "-m", "pytest", "-q", "-n", "auto", "tests/test_local_validation.py"]

        with patch.object(local_validation.subprocess, "run", return_value=local_validation.subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")):
            row = local_validation._run(Path.cwd(), "smoke", command, runner="pytest", resolved_jobs="auto")

        self.assertEqual(row["runner"], "pytest")
        self.assertEqual(row["resolved_jobs"], "auto")
        self.assertEqual(row["target_count"], 1)
        self.assertTrue(row["ok"])

    def test_run_validation_records_remaining_tiers_after_failure(self) -> None:
        def fake_run(
            repo_root: Path,
            name: str,
            command: list[str],
            *,
            runner: str,
            resolved_jobs: str,
        ) -> dict[str, object]:
            return {
                "name": name,
                "command": command,
                "runner": runner,
                "resolved_jobs": resolved_jobs,
                "target_count": 1,
                "ok": False,
                "returncode": 1,
                "elapsed_s": 0.1,
                "output_tail": "failed",
            }

        with patch.object(local_validation, "_has_module", side_effect=lambda name: name in {"pytest", "xdist"}):
            with patch.object(local_validation, "_run", side_effect=fake_run):
                payload = local_validation.run_validation("full", repo_root=Path.cwd(), runner="auto", jobs="auto")

        self.assertEqual(payload["resolved_runner"], "pytest")
        self.assertEqual(payload["resolved_jobs"], "auto")
        self.assertTrue(payload["pytest_available"])
        self.assertTrue(payload["xdist_available"])
        self.assertEqual(payload["planned_tiers"], ["smoke", "agent", "full-suite"])
        self.assertEqual(payload["tiers_completed"], ["smoke"])
        self.assertEqual(payload["remaining_tiers"], ["agent", "full-suite"])
        self.assertTrue(payload["stopped_after_failure"])
        self.assertFalse(payload["ok"])


if __name__ == "__main__":
    unittest.main()
