import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from scripts import local_validation


class LocalValidationTests(unittest.TestCase):
    def _tier_commands_with_pytest(self, tier: str, *, cpu_count: int = 32, runner: str = "auto", jobs: str = "auto") -> list[tuple[str, list[str]]]:
        with patch.object(local_validation, "_has_module", side_effect=lambda name: name in {"pytest", "xdist"}):
            with patch.object(local_validation.os, "cpu_count", return_value=cpu_count):
                return local_validation._tier_commands(tier, runner=runner, jobs=jobs)

    def test_auto_runner_prefers_pytest_with_capped_xdist_for_agent(self) -> None:
        commands = self._tier_commands_with_pytest("agent")

        self.assertEqual(len(commands), 1)
        name, command = commands[0]
        self.assertEqual(name, "agent")
        self.assertEqual(command[:6], [sys.executable, "-m", "pytest", "-q", "-n", "16"])
        self.assertIn("tests/test_tools.py", command)

    def test_auto_jobs_turn_off_xdist_on_single_cpu(self) -> None:
        commands = self._tier_commands_with_pytest("agent", cpu_count=1)

        _name, command = commands[0]
        self.assertEqual(command[:4], [sys.executable, "-m", "pytest", "-q"])
        self.assertNotIn("-n", command)

    def test_auto_runner_falls_back_to_unittest_when_pytest_missing(self) -> None:
        with patch.object(local_validation, "_has_module", return_value=False):
            commands = local_validation._tier_commands("smoke", runner="auto", jobs="auto")

        self.assertEqual(len(commands), 1)
        name, command = commands[0]
        self.assertEqual(name, "smoke")
        self.assertEqual(command[:3], [sys.executable, "-m", "unittest"])
        self.assertIn("tests.test_live_model_gate", command)

    def test_jobs_off_disables_xdist(self) -> None:
        commands = self._tier_commands_with_pytest("agent", runner="pytest", jobs="off")

        _name, command = commands[0]
        self.assertEqual(command[:4], [sys.executable, "-m", "pytest", "-q"])
        self.assertNotIn("-n", command)

    def test_full_tier_prefers_pytest_broad_final_gate(self) -> None:
        commands = self._tier_commands_with_pytest("full")

        self.assertEqual([name for name, _ in commands], ["smoke", "agent", "full-remaining"])
        self.assertEqual(commands[-1][1][:6], [sys.executable, "-m", "pytest", "-q", "-n", "16"])
        self.assertIn("tests/test_cli.py", commands[-1][1])
        self.assertNotIn("tests/test_local_validation.py", commands[-1][1])
        self.assertNotIn("tests/test_tools.py", commands[-1][1])

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

    def test_timing_summary_orders_slowest_commands(self) -> None:
        summary = local_validation._timing_summary(
            [
                {"name": "agent", "elapsed_s": 4.0, "target_count": 6, "ok": True},
                {"name": "smoke", "elapsed_s": 1.5, "target_count": 10, "ok": True},
                {"name": "full-remaining", "elapsed_s": 2.0, "target_count": 25, "ok": False},
            ],
            elapsed_s=7.5,
        )

        self.assertEqual(summary["command_count"], 3)
        self.assertEqual(summary["successful_commands"], 2)
        self.assertEqual(summary["failed_commands"], 1)
        self.assertEqual(
            [row["name"] for row in summary["slowest_commands"]],
            ["agent", "full-remaining", "smoke"],
        )

    def test_coverage_summary_partitions_pytest_full_plan_without_duplicates(self) -> None:
        discovered = (
            *(local_validation._module_to_path(module) for module in local_validation.SMOKE_MODULES),
            *(local_validation._module_to_path(module) for module in local_validation.AGENT_MODULES),
            "tests/test_cli.py",
            "tests/test_public_benchmark_eval.py",
        )

        with patch.object(local_validation, "_all_pytest_targets", return_value=tuple(discovered)):
            commands = self._tier_commands_with_pytest("full", jobs="off")
            summary = local_validation._coverage_summary(
                "full",
                repo_root=Path.cwd(),
                runner="pytest",
                jobs="off",
                resolved_jobs="off",
            )

        self.assertEqual([name for name, _ in commands], ["smoke", "agent", "full-remaining"])
        self.assertEqual(summary["mode"], "pytest_target_paths")
        self.assertEqual(summary["discovered_test_target_count"], len(discovered))
        self.assertEqual(summary["full_plan_duplicate_target_count"], 0)
        self.assertEqual(summary["full_plan_uncovered_target_count"], 0)
        self.assertEqual(summary["full_plan_extra_target_count"], 0)
        self.assertTrue(summary["full_plan_covers_all_discovered_targets"])
        self.assertEqual(
            summary["full_plan_targets_by_tier"]["full-remaining"],
            ["tests/test_cli.py", "tests/test_public_benchmark_eval.py"],
        )

    def test_run_counts_unittest_module_targets(self) -> None:
        command = [sys.executable, "-m", "unittest", "tests.test_live_model_gate", "tests.test_nightly_self_improvement_report", "-q"]

        with patch.object(local_validation.subprocess, "run", return_value=local_validation.subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")):
            row = local_validation._run(Path.cwd(), "smoke", command, runner="unittest", resolved_jobs="off")

        self.assertEqual(row["target_count"], 2)

    def test_run_counts_unittest_discover_as_single_target(self) -> None:
        command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]

        with patch.object(local_validation.subprocess, "run", return_value=local_validation.subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")):
            row = local_validation._run(Path.cwd(), "full-discover", command, runner="unittest", resolved_jobs="off")

        self.assertEqual(row["target_count"], 1)

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
            with patch.object(local_validation.os, "cpu_count", return_value=32):
                with patch.object(local_validation, "_run", side_effect=fake_run):
                    payload = local_validation.run_validation("full", repo_root=Path.cwd(), runner="auto", jobs="auto")

        self.assertEqual(payload["resolved_runner"], "pytest")
        self.assertEqual(payload["resolved_jobs"], "16")
        self.assertTrue(payload["pytest_available"])
        self.assertTrue(payload["xdist_available"])
        self.assertEqual(payload["planned_tiers"], ["smoke", "agent", "full-remaining"])
        self.assertEqual(payload["tiers_completed"], ["smoke"])
        self.assertEqual(payload["remaining_tiers"], ["agent", "full-remaining"])
        self.assertTrue(payload["stopped_after_failure"])
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["commands"][0]["elapsed_share_pct"], 100.0)
        self.assertEqual(payload["timing_summary"]["command_count"], 1)
        self.assertEqual(payload["timing_summary"]["slowest_commands"][0]["name"], "smoke")
        self.assertEqual(payload["coverage_summary"]["mode"], "pytest_target_paths")
        self.assertIn("full_plan_covers_all_discovered_targets", payload["coverage_summary"])

    def test_run_validation_records_optional_unittest_baseline_compare(self) -> None:
        def fake_run(
            repo_root: Path,
            name: str,
            command: list[str],
            *,
            runner: str,
            resolved_jobs: str,
        ) -> dict[str, object]:
            elapsed_s = 0.1
            if name == "unittest-baseline":
                elapsed_s = 0.6
            return {
                "name": name,
                "command": command,
                "runner": runner,
                "resolved_jobs": resolved_jobs,
                "target_count": 1,
                "ok": True,
                "returncode": 0,
                "elapsed_s": elapsed_s,
                "output_tail": "ok",
            }

        with patch.object(local_validation, "_has_module", side_effect=lambda name: name in {"pytest", "xdist"}):
            with patch.object(local_validation.os, "cpu_count", return_value=32):
                with patch.object(local_validation, "_run", side_effect=fake_run):
                    payload = local_validation.run_validation(
                        "full",
                        repo_root=Path.cwd(),
                        runner="auto",
                        jobs="auto",
                        compare_unittest_baseline=True,
                    )

        comparison = payload["baseline_compare"]
        self.assertTrue(comparison["requested"])
        self.assertTrue(comparison["ran"])
        self.assertEqual(comparison["preferred_elapsed_s"], 0.3)
        self.assertEqual(comparison["preferred_vs_unittest_ratio"], 0.5)
        self.assertEqual(comparison["preferred_minus_unittest_s"], -0.3)
        self.assertTrue(comparison["preferred_faster_than_unittest"])
        self.assertEqual(comparison["unittest_discover"]["name"], "unittest-baseline")
        self.assertEqual([row["elapsed_share_pct"] for row in payload["commands"]], [33.3, 33.3, 33.3])
        self.assertEqual(payload["timing_summary"]["slowest_commands"][0]["name"], "agent")
        self.assertEqual(payload["coverage_summary"]["requested_tier"], "full")

    def test_run_validation_skips_unittest_baseline_compare_for_non_full_tier(self) -> None:
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
                "ok": True,
                "returncode": 0,
                "elapsed_s": 0.1,
                "output_tail": "ok",
            }

        with patch.object(local_validation, "_has_module", side_effect=lambda name: name in {"pytest", "xdist"}):
            with patch.object(local_validation.os, "cpu_count", return_value=32):
                with patch.object(local_validation, "_run", side_effect=fake_run):
                    payload = local_validation.run_validation(
                        "smoke",
                        repo_root=Path.cwd(),
                        runner="auto",
                        jobs="auto",
                        compare_unittest_baseline=True,
                    )

        comparison = payload["baseline_compare"]
        self.assertTrue(comparison["requested"])
        self.assertFalse(comparison["ran"])
        self.assertEqual(comparison["skipped_reason"], "comparison only runs for the full tier")

    def test_main_prints_slowest_timing_summary(self) -> None:
        payload = {
            "ok": True,
            "commands": [
                {
                    "name": "smoke",
                    "runner": "pytest",
                    "resolved_jobs": "16",
                    "ok": True,
                    "elapsed_s": 6.5,
                    "returncode": 0,
                    "elapsed_share_pct": 100.0,
                }
            ],
            "timing_summary": {
                "total_elapsed_s": 6.5,
                "slowest_commands": [
                    {"name": "smoke", "elapsed_s": 6.5, "target_count": 10, "ok": True}
                ],
            },
            "coverage_summary": {
                "mode": "pytest_target_paths",
                "requested_tier": "smoke",
                "requested_unique_target_count": 10,
                "full_plan_unique_target_count": 37,
                "discovered_test_target_count": 37,
                "full_plan_covers_all_discovered_targets": True,
                "full_plan_duplicate_target_count": 0,
                "full_plan_uncovered_target_count": 0,
            },
            "baseline_compare": {"requested": False, "ran": False, "skipped_reason": None},
        }
        output_path = Path.cwd() / "scratch" / "validation" / "test-local-validation-main.json"
        written_text: dict[str, str] = {}

        def fake_write_text(text: str, encoding: str = "utf-8") -> int:
            written_text["value"] = text
            return len(text)

        stdout = StringIO()
        with patch.object(local_validation, "run_validation", return_value=payload):
            with patch.object(Path, "write_text", autospec=True, side_effect=lambda self, text, encoding="utf-8": fake_write_text(text, encoding)):
                with patch.object(sys, "stdout", stdout):
                    exit_code = local_validation.main(["--tier", "smoke", "--output", str(output_path)])

        self.assertEqual(exit_code, 0)
        self.assertIn("slowest=smoke", stdout.getvalue())
        self.assertIn("share_pct=100.0", stdout.getvalue())
        self.assertIn("coverage tier=smoke", stdout.getvalue())
        self.assertIn("full_plan_complete=True", stdout.getvalue())
        self.assertIn('"timing_summary"', written_text["value"])


if __name__ == "__main__":
    unittest.main()
