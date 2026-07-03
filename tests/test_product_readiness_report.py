import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from scripts import product_readiness_report


class ProductReadinessReportTests(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict[str, object], *, mtime: datetime | None = None) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        if mtime is not None:
            timestamp = mtime.timestamp()
            path.touch()
            import os

            os.utime(path, (timestamp, timestamp))
        return path

    def _write_green_artifacts(self, root: Path, *, now: datetime) -> dict[str, Path]:
        doctor = self._write_json(
            root / "scratch" / "validation" / "doctor-report.json",
            {
                "ok": True,
                "status": "pass",
                "summary": "Ollama Code doctor",
                "git_commit": "abc123",
                "git_dirty": False,
            },
            mtime=now,
        )
        local_validation = self._write_json(
            root / "scratch" / "validation" / "local-validation-summary.json",
            {
                "ok": True,
                "command_ok": True,
                "requested_tier": "agent",
                "resolved_runner": "pytest",
                "git_commit": "abc123",
                "git_dirty": False,
            },
            mtime=now,
        )
        live_gate = self._write_json(
            root / "scratch" / "live-model-gate" / "live-model-gate-summary.json",
            {
                "generated_at": now.isoformat(),
                "git_commit": "abc123",
                "git_dirty": False,
                "benchmark_suite": "local-small",
                "selected_default_model": "granite4.1:8b",
                "selection_reason": "Selected granite4.1:8b because it had the highest benchmark pass count.",
                "ok": True,
                "models": [
                    {
                        "model": "granite4.1:8b",
                        "e2e_ok": True,
                        "verification_ok": True,
                        "benchmark_ok": True,
                        "benchmark_passes": 8,
                        "benchmark_runs": 8,
                        "benchmark_total_tokens": 2000,
                        "benchmark_total_llm_calls": 4,
                        "benchmark_median_latency_s": 10.0,
                        "benchmark_artifact": "scratch/live-model-gate/coding-benchmark-granite4.1-8b.json",
                    }
                ],
            },
            mtime=now,
        )
        local_small = self._write_json(
            root / "scratch" / "coding-benchmark" / "local-small.json",
            {
                "suite": "local-small",
                "git_commit": "abc123",
                "git_dirty": False,
                "summary": {"runs": 8, "pass": 8},
                "accuracy_regressions": [],
                "budget_failures": [],
                "llm_bypass_failures": [],
            },
            mtime=now,
        )
        hard_cases = self._write_json(
            root / "scratch" / "coding-benchmark" / "local-full.json",
            {
                "suite": "local-full",
                "git_commit": "abc123",
                "git_dirty": False,
                "results": [{"case": "task_due_filter", "status": "pass"}],
            },
            mtime=now,
        )
        return {
            "doctor": doctor,
            "local_validation": local_validation,
            "live_gate": live_gate,
            "local_small": local_small,
            "hard_cases": hard_cases,
        }

    def _build(self, root: Path, *, now: datetime, **overrides: object) -> dict[str, object]:
        with patch.object(product_readiness_report, "_git_head_commit", return_value="abc123"):
            with patch.object(product_readiness_report, "_git_dirty", return_value=False):
                return product_readiness_report.build_report(root, now=now, **overrides)

    def test_build_report_passes_with_current_green_artifacts(self) -> None:
        now = datetime(2026, 7, 3, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_green_artifacts(root, now=now)

            payload = self._build(root, now=now)

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["blocking_checks"], [])
        self.assertIn("Product readiness: PASS", product_readiness_report.render_text(payload))

    def test_dirty_git_state_blocks_readiness_by_default(self) -> None:
        now = datetime(2026, 7, 3, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_green_artifacts(root, now=now)
            with patch.object(product_readiness_report, "_git_head_commit", return_value="abc123"):
                with patch.object(product_readiness_report, "_git_dirty", return_value=True):
                    payload = product_readiness_report.build_report(root, now=now)

        self.assertFalse(payload["ok"])
        self.assertIn("git", payload["blocking_checks"])

    def test_missing_live_gate_blocks_readiness(self) -> None:
        now = datetime(2026, 7, 3, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self._write_green_artifacts(root, now=now)
            artifacts["live_gate"].unlink()

            payload = self._build(root, now=now)

        self.assertFalse(payload["ok"])
        self.assertIn("live_model_gate", payload["blocking_checks"])
        live_gate = next(check for check in payload["checks"] if check["name"] == "live_model_gate")
        self.assertIn("scripts/live_model_gate.py", live_gate["refresh_command"])

    def test_render_text_includes_refresh_commands_for_blocking_checks(self) -> None:
        now = datetime(2026, 7, 3, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self._write_green_artifacts(root, now=now)
            artifacts["doctor"].unlink()

            payload = self._build(root, now=now)
            rendered = product_readiness_report.render_text(payload)

        self.assertFalse(payload["ok"])
        self.assertIn("Refresh commands:", rendered)
        self.assertIn("doctor: python scripts/doctor_report.py", rendered)

    def test_missing_doctor_artifact_blocks_readiness(self) -> None:
        now = datetime(2026, 7, 3, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self._write_green_artifacts(root, now=now)
            artifacts["doctor"].unlink()

            payload = self._build(root, now=now)

        self.assertFalse(payload["ok"])
        self.assertIn("doctor", payload["blocking_checks"])

    def test_failing_doctor_artifact_blocks_readiness(self) -> None:
        now = datetime(2026, 7, 3, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_green_artifacts(root, now=now)
            self._write_json(
                root / "scratch" / "validation" / "doctor-report.json",
                {
                    "ok": False,
                    "status": "fail",
                    "summary": "model missing",
                    "git_commit": "abc123",
                    "git_dirty": False,
                },
                mtime=now,
            )

            payload = self._build(root, now=now)

        self.assertFalse(payload["ok"])
        self.assertIn("doctor", payload["blocking_checks"])

    def test_failing_hard_case_blocks_readiness(self) -> None:
        now = datetime(2026, 7, 3, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_green_artifacts(root, now=now)
            self._write_json(
                root / "scratch" / "coding-benchmark" / "local-full.json",
                {
                    "suite": "local-full",
                    "git_commit": "abc123",
                    "git_dirty": False,
                    "results": [{"case": "task_due_filter", "status": "fail"}],
                },
                mtime=now,
            )

            payload = self._build(root, now=now)

        self.assertFalse(payload["ok"])
        self.assertIn("hard_cases", payload["blocking_checks"])

    def test_stale_artifact_blocks_readiness(self) -> None:
        now = datetime(2026, 7, 3, tzinfo=timezone.utc)
        stale = now - timedelta(hours=10)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_green_artifacts(root, now=now)
            self._write_json(
                root / "scratch" / "coding-benchmark" / "local-small.json",
                {
                    "suite": "local-small",
                    "git_commit": "abc123",
                    "git_dirty": False,
                    "summary": {"runs": 8, "pass": 8},
                    "accuracy_regressions": [],
                    "budget_failures": [],
                    "llm_bypass_failures": [],
                },
                mtime=stale,
            )

            payload = self._build(root, now=now, max_age_hours=1.0)

        self.assertFalse(payload["ok"])
        self.assertIn("local_small", payload["blocking_checks"])

    def test_local_small_can_use_selected_live_gate_benchmark_artifact(self) -> None:
        now = datetime(2026, 7, 3, tzinfo=timezone.utc)
        stale = now - timedelta(hours=10)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_green_artifacts(root, now=now)
            selected_artifact = root / "scratch" / "live-model-gate" / "coding-benchmark-granite4.1-8b.json"
            self._write_json(
                selected_artifact,
                {
                    "suite": "local-small",
                    "git_commit": "abc123",
                    "git_dirty": False,
                    "summary": {"runs": 8, "pass": 8},
                    "accuracy_regressions": [],
                    "budget_failures": [],
                    "llm_bypass_failures": [],
                },
                mtime=now,
            )
            self._write_json(
                root / "scratch" / "coding-benchmark" / "local-small.json",
                {
                    "suite": "local-small",
                    "git_commit": "old123",
                    "git_dirty": False,
                    "summary": {"runs": 1, "pass": 1},
                    "accuracy_regressions": [],
                    "budget_failures": [],
                    "llm_bypass_failures": [],
                },
                mtime=stale,
            )

            payload = self._build(root, now=now)
            local_small = next(check for check in payload["checks"] if check["name"] == "local_small")

        self.assertTrue(local_small["ok"])
        self.assertIn("scratch", local_small["details"]["path"])
        self.assertNotIn("local_small", payload["blocking_checks"])

    def test_stale_doctor_git_metadata_blocks_readiness(self) -> None:
        now = datetime(2026, 7, 3, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_green_artifacts(root, now=now)
            self._write_json(
                root / "scratch" / "validation" / "doctor-report.json",
                {
                    "ok": True,
                    "status": "pass",
                    "summary": "Ollama Code doctor",
                    "git_commit": "old123",
                    "git_dirty": False,
                },
                mtime=now,
            )

            payload = self._build(root, now=now)

        self.assertFalse(payload["ok"])
        self.assertIn("doctor", payload["blocking_checks"])

    def test_stale_local_validation_git_metadata_blocks_readiness(self) -> None:
        now = datetime(2026, 7, 3, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_green_artifacts(root, now=now)
            self._write_json(
                root / "scratch" / "validation" / "local-validation-summary.json",
                {
                    "ok": True,
                    "command_ok": True,
                    "requested_tier": "agent",
                    "resolved_runner": "pytest",
                    "git_commit": "old123",
                    "git_dirty": False,
                },
                mtime=now,
            )

            payload = self._build(root, now=now)

        self.assertFalse(payload["ok"])
        self.assertIn("local_validation", payload["blocking_checks"])

    def test_stale_benchmark_git_metadata_blocks_readiness(self) -> None:
        now = datetime(2026, 7, 3, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_green_artifacts(root, now=now)
            self._write_json(
                root / "scratch" / "coding-benchmark" / "local-full.json",
                {
                    "suite": "local-full",
                    "git_commit": "old123",
                    "git_dirty": False,
                    "results": [{"case": "task_due_filter", "status": "pass"}],
                },
                mtime=now,
            )

            payload = self._build(root, now=now)

        self.assertFalse(payload["ok"])
        self.assertIn("hard_cases", payload["blocking_checks"])


if __name__ == "__main__":
    unittest.main()
