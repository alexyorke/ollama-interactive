import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import nightly_self_improvement_report as report


class NightlySelfImprovementReportTests(unittest.TestCase):
    def _args(self, output_dir: Path, trajectory_data_root: Path) -> argparse.Namespace:
        return argparse.Namespace(
            output_dir=output_dir,
            compare=None,
            models=["gemma4:e4b"],
            suite="local-small",
            generated_files=100,
            case_timeout=60,
            command_timeout=60,
            tool_timeout=30,
            sdk_index_limit=100,
            sdk_use_embeddings=False,
            sdk_embedding_model=None,
            skip_llm=False,
            skip_trajectories=False,
            trajectory_data_root=trajectory_data_root,
            trajectory_max_rows=25,
            strict_accuracy=True,
            strict_budget=True,
        )

    def test_default_compare_path_uses_latest_prior_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            root = repo_root / "scratch" / "nightly-self-improvement"
            older = root / "20260101T000000Z"
            newer = root / "20260102T000000Z"
            current = root / "20260103T000000Z"
            older.mkdir(parents=True)
            newer.mkdir(parents=True)
            current.mkdir(parents=True)
            (older / "report.json").write_text("{}", encoding="utf-8")
            (newer / "report.json").write_text("{}", encoding="utf-8")
            (current / "report.json").write_text("{}", encoding="utf-8")

            selected = report._default_compare_path(repo_root, current)

        self.assertEqual(selected, newer / "report.json")

    def test_resolve_ollama_host_prefers_first_reachable_candidate(self) -> None:
        with patch.dict("os.environ", {"OLLAMA_HOST": ""}, clear=False), patch.object(
            report, "_host_responds", side_effect=lambda host: host == "http://[::1]:11434"
        ):
            selected = report._resolve_ollama_host()

        self.assertEqual(selected, "http://[::1]:11434")

    def test_build_report_includes_question_and_trajectory_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            output_dir = repo_root / "out"
            data_root = repo_root / "datasets"
            (data_root / "nebius-swe-agent-trajectories").mkdir(parents=True)
            args = self._args(output_dir, data_root)

            def fake_run_command(repo: Path, name: str, command: list[str], timeout: int, output_path: Path | None = None) -> dict[str, object]:
                self.assertEqual(repo, repo_root)
                self.assertGreater(timeout, 0)
                return {
                    "name": name,
                    "command": command,
                    "exit_code": 0,
                    "elapsed_s": 0.1,
                    "ok": True,
                    "output_path": str(output_path) if output_path else None,
                    "output_tail": "",
                }

            def fake_run_command_with_env(
                repo: Path,
                name: str,
                command: list[str],
                timeout: int,
                output_path: Path | None = None,
                env_overrides: dict[str, str] | None = None,
            ) -> dict[str, object]:
                self.assertEqual(repo, repo_root)
                self.assertEqual(env_overrides, {"OLLAMA_HOST": "http://[::1]:11434"})
                self.assertGreater(timeout, 0)
                return {
                    "name": name,
                    "command": command,
                    "exit_code": 0,
                    "elapsed_s": 0.1,
                    "ok": True,
                    "output_path": str(output_path) if output_path else None,
                    "output_tail": "",
                }

            def fake_load_json(path: Path) -> dict[str, object]:
                name = path.name
                if name == "tool_speed.json":
                    return {"generated_files": 100, "rows": [{"name": "search", "elapsed_ms": 12.0}]}
                if name == "python_sdk_search.json":
                    return {"summary": {"fail": 0}, "refresh": {}, "mode": "fts"}
                if name == "question-quality.json":
                    return {"summary": {"cases": 5, "passed": 5, "failed": 0}}
                if name == "trajectory-dataset-catalog.json":
                    return {
                        "summary": {
                            "entries": 3,
                            "local_entries": 1,
                            "analysis_ready_local_entries": 1,
                            "public_missing_entries": 1,
                            "gated_entries": 1,
                            "high_priority_public_missing": [{"id": "nvidia/SWE-Hero-openhands-trajectories", "reason": "public"}],
                        }
                    }
                if name == "trajectory-profile.json":
                    return {"datasets": [{"rows_profiled": 10}], "portfolio_recommendations": [{"id": "loop-cap"}]}
                if name == "trajectory-error-profile.json":
                    return {"datasets": [{"error_counts": {"test_assertion": 4, "timeout": 1}}]}
                if name == "trajectory-evidence-report.json":
                    return {"portfolio_fix_coverage": [{"id": "mechanical-router", "evidence_count": 12, "status": "partial"}]}
                if name == "token_efficiency.json":
                    return {"summary": {"fail": 0, "total_tokens": 100}, "results": [], "accuracy_regressions": []}
                if name == "coding_benchmark.json":
                    return {
                        "summary": {"fail": 0, "total_tokens": 90},
                        "results": [],
                        "accuracy_regressions": [],
                        "budget_failures": [],
                        "llm_bypass_failures": [],
                    }
                return {}

            with patch.object(report, "_repo_root", return_value=repo_root), patch.object(report, "_run_command", side_effect=fake_run_command), patch.object(report, "_run_command_with_env", side_effect=fake_run_command_with_env), patch.object(report, "_load_json", side_effect=fake_load_json), patch.object(report, "_git_value", return_value="main"), patch.object(report, "_resolve_ollama_host", return_value="http://[::1]:11434"):
                payload = report.build_report(args)

        command_names = [item["name"] for item in payload["commands"]]
        self.assertIn("question_quality", command_names)
        self.assertIn("trajectory_dataset_catalog", command_names)
        self.assertIn("trajectory_profile", command_names)
        self.assertIn("trajectory_error_profile", command_names)
        self.assertIn("trajectory_evidence_report", command_names)
        self.assertEqual(payload["metrics"]["question_quality"]["failed"], 0)
        self.assertEqual(payload["metrics"]["trajectory_profile"]["rows_profiled"], 10)
        self.assertEqual(payload["metrics"]["trajectory_error_profile"]["top_errors"][0]["name"], "test_assertion")
        self.assertEqual(payload["metrics"]["trajectory_evidence_report"]["top_fix_coverage"][0]["id"], "mechanical-router")
        self.assertEqual(payload["metrics"]["trajectory_dataset_catalog"]["public_missing_entries"], 1)
        self.assertEqual(payload["runtime"]["resolved_ollama_host"], "http://[::1]:11434")
        self.assertEqual(payload["runtime"]["llm_skip_reason"], "")
        self.assertTrue(any("nvidia/SWE-Hero-openhands-trajectories" in item for item in payload["metrics"]["suggested_implementation_targets"]))


if __name__ == "__main__":
    unittest.main()
