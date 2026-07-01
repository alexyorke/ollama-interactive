import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import live_model_gate


class LiveModelGateTests(unittest.TestCase):
    @patch("scripts.live_model_gate.subprocess.run")
    def test_git_provenance_helpers_capture_commit_and_dirty_state(self, run_mock: object) -> None:
        run_mock.side_effect = [
            type("Completed", (), {"returncode": 0, "stdout": "abc123\n"})(),
            type("Completed", (), {"returncode": 0, "stdout": " M scripts/live_model_gate.py\n"})(),
        ]

        repo_root = Path("C:/repo")
        self.assertEqual(live_model_gate._git_commit(repo_root), "abc123")
        self.assertTrue(live_model_gate._git_dirty(repo_root))

    def test_choose_default_model_prefers_lower_tokens_after_pass_tie(self) -> None:
        model, reason = live_model_gate.choose_default_model(
            [
                {
                    "model": "granite4.1:8b",
                    "benchmark_passes": 8,
                    "benchmark_runs": 8,
                    "benchmark_total_tokens": 2047,
                    "benchmark_median_latency_s": 9.1,
                },
                {
                    "model": "gemma4:e4b",
                    "benchmark_passes": 8,
                    "benchmark_runs": 8,
                    "benchmark_total_tokens": 2435,
                    "benchmark_median_latency_s": 8.6,
                },
                {
                    "model": "qwen3:8b",
                    "benchmark_passes": 8,
                    "benchmark_runs": 8,
                    "benchmark_total_tokens": 2534,
                    "benchmark_median_latency_s": 22.8,
                },
            ]
        )

        self.assertEqual(model, "granite4.1:8b")
        self.assertIsNotNone(reason)
        self.assertIn("fewest benchmark tokens", reason or "")

    def test_choose_default_model_prefers_granite_when_all_tie_breakers_match(self) -> None:
        model, reason = live_model_gate.choose_default_model(
            [
                {
                    "model": "granite4.1:8b",
                    "benchmark_passes": 8,
                    "benchmark_runs": 8,
                    "benchmark_total_tokens": 2000,
                    "benchmark_median_latency_s": 9.0,
                },
                {
                    "model": "gemma4:e4b",
                    "benchmark_passes": 8,
                    "benchmark_runs": 8,
                    "benchmark_total_tokens": 2000,
                    "benchmark_median_latency_s": 9.0,
                },
            ]
        )

        self.assertEqual(model, "granite4.1:8b")
        self.assertIsNotNone(reason)
        self.assertIn("tie-break", reason or "")

    def test_resolve_requested_models_dedupes_latest_aliases_while_preserving_order(self) -> None:
        models = live_model_gate.resolve_requested_models(
            ["gemma4:e4b", "gemma4:e4b:latest", "qwen3", "qwen3:latest", "gemma4:e4b"],
            {"gemma4:e4b", "gemma4:e4b:latest", "qwen3:latest"},
        )

        self.assertEqual(models, ["gemma4:e4b", "gemma4:e4b:latest", "qwen3:latest"])

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
    @patch("scripts.live_model_gate._git_dirty", return_value=False)
    @patch("scripts.live_model_gate._git_commit", return_value="abc123")
    def test_run_gate_writes_summary_shape(
        self,
        _git_commit: object,
        _git_dirty: object,
        run_step_mock: object,
        preflight_mock: object,
    ) -> None:
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
            benchmark_artifact = output_dir / "coding-benchmark-gemma4-e4b.json"
            output_dir.mkdir(parents=True, exist_ok=True)
            benchmark_artifact.write_text(
                json.dumps(
                    {
                        "summary": {"pass": 8, "runs": 8, "total_tokens": 2435, "total_llm_calls": 4},
                        "results": [{"latency_s": 8.6}, {"latency_s": 9.4}],
                    }
                ),
                encoding="utf-8",
            )
            run_step_mock.side_effect = [
                {"name": "e2e:gemma4:e4b", "model": "gemma4:e4b", "ok": True, "latency_s": 1.2, "artifact": None, "returncode": 0, "stdout_tail": "", "stderr_tail": "", "command": ["python"], "timeout_s": 600},
                {"name": "verification:gemma4:e4b", "model": "gemma4:e4b", "ok": True, "latency_s": 2.4, "artifact": None, "returncode": 0, "stdout_tail": "", "stderr_tail": "", "command": ["python"], "timeout_s": 900},
                {"name": "benchmark:gemma4:e4b", "model": "gemma4:e4b", "ok": True, "latency_s": 12.0, "artifact": str(benchmark_artifact), "returncode": 0, "stdout_tail": "", "stderr_tail": "", "command": ["python"], "timeout_s": 900},
            ]
            payload = live_model_gate.run_gate(
                repo_root,
                output_dir,
                requested_models=["gemma4:e4b"],
                run_e2e=True,
                run_verification=True,
                run_benchmarks=True,
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
            self.assertEqual(payload["benchmark_suite"], "local-small")
            self.assertEqual(payload["git_commit"], "abc123")
            self.assertFalse(payload["git_dirty"])
            self.assertEqual(payload["resolved_models"], ["gemma4:e4b"])
            self.assertEqual(payload["steps_completed"], ["e2e:gemma4:e4b", "verification:gemma4:e4b", "benchmark:gemma4:e4b"])
            self.assertEqual(payload["selected_default_model"], "gemma4:e4b")
            self.assertEqual(
                payload["models"],
                [
                    {
                        "model": "gemma4:e4b",
                        "e2e_ok": True,
                        "verification_ok": True,
                        "benchmark_ok": True,
                        "benchmark_passes": 8,
                        "benchmark_runs": 8,
                        "benchmark_total_tokens": 2435,
                        "benchmark_total_llm_calls": 4,
                        "benchmark_median_latency_s": 9.0,
                        "benchmark_artifact": str(benchmark_artifact),
                    }
                ],
            )

    @patch("scripts.live_model_gate.run_gate")
    def test_main_writes_summary_file(self, run_gate_mock: object) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "gate"
            mirror_dir = Path(tmp) / "scratch" / "live-model-gate"
            run_gate_mock.return_value = {
                "generated_at": "2026-01-01T00:00:00+00:00",
                "git_commit": "abc123",
                "git_dirty": False,
                "benchmark_suite": "local-small",
                "preflight": {"ollama_host": "http://127.0.0.1:11434"},
                "resolved_models": ["gemma4:e4b"],
                "selected_default_model": "gemma4:e4b",
                "selection_reason": "Selected gemma4:e4b because it had the highest benchmark pass count (8/8).",
                "ok": True,
                "elapsed_s": 3.14,
                "models": [],
                "step_results": [],
            }
            with patch.object(live_model_gate, "DEFAULT_OUTPUT_DIR", mirror_dir):
                exit_code = live_model_gate.main(["--models", "gemma4:e4b", "--skip-e2e", "--skip-verification", "--skip-benchmarks", "--output-dir", str(output_dir)])

            self.assertEqual(exit_code, 0)
            summary = output_dir / "live-model-gate-summary.json"
            self.assertTrue(summary.exists())
            payload = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(payload["resolved_models"], ["gemma4:e4b"])
            self.assertEqual(payload["selected_default_model"], "gemma4:e4b")
            self.assertEqual(payload["git_commit"], "abc123")
            self.assertFalse(payload["git_dirty"])
            mirror = mirror_dir / "live-model-gate-summary.json"
            self.assertTrue(mirror.exists())
            mirror_payload = json.loads(mirror.read_text(encoding="utf-8"))
            self.assertEqual(mirror_payload["selected_default_model"], "gemma4:e4b")
            self.assertEqual(mirror_payload["git_commit"], "abc123")

    @patch("scripts.live_model_gate._git_dirty", return_value=True)
    @patch("scripts.live_model_gate._git_commit", return_value="def456")
    @patch("scripts.live_model_gate.run_gate", side_effect=RuntimeError("offline"))
    def test_main_writes_failure_summary_when_preflight_fails(self, _run_gate: object, _git_commit: object, _git_dirty: object) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "gate"
            mirror_dir = Path(tmp) / "scratch" / "live-model-gate"

            with patch.object(live_model_gate, "DEFAULT_OUTPUT_DIR", mirror_dir):
                exit_code = live_model_gate.main(["--models", "gemma4:e4b", "--output-dir", str(output_dir)])

            self.assertEqual(exit_code, 1)
            payload = json.loads((output_dir / "live-model-gate-summary.json").read_text(encoding="utf-8"))
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["preflight_error"], "offline")
            self.assertEqual(payload["benchmark_suite"], "local-small")
            self.assertEqual(payload["git_commit"], "def456")
            self.assertTrue(payload["git_dirty"])
            self.assertEqual(payload["models"], [])
            self.assertIsNone(payload["selected_default_model"])
            mirror_payload = json.loads((mirror_dir / "live-model-gate-summary.json").read_text(encoding="utf-8"))
            self.assertEqual(mirror_payload["preflight_error"], "offline")
            self.assertEqual(mirror_payload["git_commit"], "def456")
            self.assertTrue(mirror_payload["git_dirty"])

    def test_summary_contract_requires_canonical_fields(self) -> None:
        self.assertTrue(
            live_model_gate.summary_contract_ok(
                {
                    "benchmark_suite": "local-small",
                    "git_commit": "abc123",
                    "git_dirty": False,
                    "ok": True,
                    "selected_default_model": "granite4.1:8b",
                    "selection_reason": "Granite won the token tie-break.",
                    "models": [
                        {
                            "model": "granite4.1:8b",
                            "benchmark_passes": 8,
                            "benchmark_runs": 8,
                            "benchmark_total_tokens": 2047,
                            "benchmark_median_latency_s": 9.0,
                        }
                    ],
                }
            )
        )
        self.assertFalse(live_model_gate.summary_contract_ok({"models": []}))
        self.assertFalse(
            live_model_gate.summary_contract_ok(
                {
                    "benchmark_suite": "local-small",
                    "git_commit": "abc123",
                    "git_dirty": False,
                    "ok": True,
                    "selected_default_model": None,
                    "selection_reason": None,
                    "models": [],
                }
            )
        )
        self.assertFalse(
            live_model_gate.summary_contract_ok(
                {
                    "benchmark_suite": "local-small",
                    "git_commit": "abc123",
                    "git_dirty": False,
                    "ok": True,
                    "selected_default_model": "gemma4:e4b",
                    "selection_reason": "Granite won the token tie-break.",
                    "models": [
                        {
                            "model": "granite4.1:8b",
                            "benchmark_passes": 8,
                            "benchmark_runs": 8,
                            "benchmark_total_tokens": 2047,
                            "benchmark_median_latency_s": 9.0,
                        }
                    ],
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
