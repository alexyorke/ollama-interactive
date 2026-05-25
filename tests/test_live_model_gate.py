import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import live_model_gate


class LiveModelGateTests(unittest.TestCase):
    def test_build_steps_includes_llm_required_benchmark_flags(self) -> None:
        repo_root = Path("C:/repo")
        output_dir = repo_root / "scratch" / "live-model-gate"

        steps = live_model_gate.build_steps(
            repo_root,
            output_dir,
            ["gemma4:e4b"],
            run_e2e=True,
            run_verification=True,
            run_benchmarks=True,
            e2e_scenarios=["scenario_transcripted_tool_use", "scenario_run_test"],
            e2e_timeout=600,
            verification_timeout=500,
            benchmark_timeout=400,
            benchmark_suite="local-small",
            benchmark_modes=["off"],
            benchmark_feature_profiles=["trajectory-guards"],
            benchmark_classes=["agent", "controller"],
            benchmark_jobs=1,
        )

        self.assertEqual([step.name for step in steps], ["e2e:gemma4:e4b", "verification:gemma4:e4b", "benchmark:gemma4:e4b"])
        benchmark = steps[2]
        self.assertIn("--require-llm-for-agent-benchmarks", benchmark.command)
        self.assertIn("--strict-accuracy", benchmark.command)
        self.assertIn("--strict-budget", benchmark.command)

    @patch("scripts.live_model_gate.installed_models", return_value=["gemma4:e4b", "qwen3:8b"])
    @patch("scripts.live_model_gate.shutil.which", return_value="C:/bin/ollama.exe")
    def test_preflight_resolves_requested_models(self, _which: object, _models: object) -> None:
        payload = live_model_gate.preflight(["gemma4:e4b", "granite4.1:8b"])

        self.assertEqual(payload["resolved_models"], ["gemma4:e4b"])
        self.assertIn("qwen3:8b", payload["available_models"])

    @patch("scripts.live_model_gate.preflight")
    @patch("scripts.live_model_gate.run_step")
    def test_run_gate_writes_summary_shape(self, run_step_mock: object, preflight_mock: object) -> None:
        preflight_mock.return_value = {
            "ollama_path": "C:/bin/ollama.exe",
            "ollama_host": "http://127.0.0.1:11434",
            "available_models": ["gemma4:e4b"],
            "resolved_models": ["gemma4:e4b"],
        }
        run_step_mock.side_effect = [
            {"name": "e2e:gemma4:e4b", "model": "gemma4:e4b", "ok": True, "latency_s": 1.2, "artifact": None, "returncode": 0, "stdout_tail": "", "stderr_tail": "", "command": ["python"], "timeout_s": 600},
            {"name": "verification:gemma4:e4b", "model": "gemma4:e4b", "ok": True, "latency_s": 2.4, "artifact": None, "returncode": 0, "stdout_tail": "", "stderr_tail": "", "command": ["python"], "timeout_s": 900},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            output_dir = repo_root / "scratch" / "live-model-gate"
            payload = live_model_gate.run_gate(
                repo_root,
                output_dir,
                requested_models=["gemma4:e4b"],
                run_e2e=True,
                run_verification=True,
                run_benchmarks=False,
                e2e_scenarios=["scenario_transcripted_tool_use"],
                e2e_timeout=600,
                verification_timeout=600,
                benchmark_timeout=600,
                benchmark_suite="local-small",
                benchmark_modes=["off"],
                benchmark_feature_profiles=["trajectory-guards"],
                benchmark_classes=["agent"],
                benchmark_jobs=1,
                continue_on_failure=False,
            )

            self.assertTrue(payload["ok"])
            self.assertEqual(payload["resolved_models"], ["gemma4:e4b"])
            self.assertEqual(payload["steps_completed"], ["e2e:gemma4:e4b", "verification:gemma4:e4b"])

    @patch("scripts.live_model_gate.run_gate")
    def test_main_writes_summary_file(self, run_gate_mock: object) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "gate"
            run_gate_mock.return_value = {
                "generated_at": "2026-01-01T00:00:00+00:00",
                "preflight": {"ollama_host": "http://127.0.0.1:11434"},
                "resolved_models": ["gemma4:e4b"],
                "ok": True,
                "elapsed_s": 3.14,
                "step_results": [],
            }

            exit_code = live_model_gate.main(["--models", "gemma4:e4b", "--skip-e2e", "--skip-verification", "--skip-benchmarks", "--output-dir", str(output_dir)])

            self.assertEqual(exit_code, 0)
            summary = output_dir / "live-model-gate-summary.json"
            self.assertTrue(summary.exists())
            payload = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(payload["resolved_models"], ["gemma4:e4b"])

    @patch("scripts.live_model_gate.run_gate", side_effect=RuntimeError("offline"))
    def test_main_writes_failure_summary_when_preflight_fails(self, _run_gate: object) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "gate"

            exit_code = live_model_gate.main(["--models", "gemma4:e4b", "--output-dir", str(output_dir)])

            self.assertEqual(exit_code, 1)
            payload = json.loads((output_dir / "live-model-gate-summary.json").read_text(encoding="utf-8"))
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["preflight_error"], "offline")


if __name__ == "__main__":
    unittest.main()
