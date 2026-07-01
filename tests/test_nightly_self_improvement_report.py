import argparse
from contextlib import contextmanager
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import nightly_self_improvement_report as report


@contextmanager
def _temp_root():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


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
        with _temp_root() as repo_root:
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

    def test_default_compare_path_uses_latest_timestamped_report_for_custom_output_dir(self) -> None:
        with _temp_root() as repo_root:
            root = repo_root / "scratch" / "nightly-self-improvement"
            older = root / "20260101T000000Z"
            newer = root / "20260102T000000Z"
            custom = repo_root / "out"
            older.mkdir(parents=True)
            newer.mkdir(parents=True)
            custom.mkdir(parents=True)
            (older / "report.json").write_text("{}", encoding="utf-8")
            (newer / "report.json").write_text("{}", encoding="utf-8")

            selected = report._default_compare_path(repo_root, custom)

        self.assertEqual(selected, newer / "report.json")

    def test_default_compare_path_falls_back_to_legacy_self_improvement_runs(self) -> None:
        with _temp_root() as repo_root:
            legacy_root = repo_root / ".ollama-code" / "self-improvement-runs"
            older = legacy_root / "20260516-211548"
            newer = legacy_root / "20260516-224919"
            current = repo_root / "scratch" / "nightly-self-improvement" / "20260517T010000Z"
            older.mkdir(parents=True)
            newer.mkdir(parents=True)
            current.mkdir(parents=True)
            (older / "report.json").write_text("{}", encoding="utf-8")
            (newer / "report.json").write_text("{}", encoding="utf-8")

            selected = report._default_compare_path(repo_root, current)

        self.assertEqual(selected, newer / "report.json")

    def test_default_compare_path_does_not_select_future_timestamped_report(self) -> None:
        with _temp_root() as repo_root:
            root = repo_root / "scratch" / "nightly-self-improvement"
            future = root / "20260102T000000Z"
            current = root / "20260101T000000Z"
            future.mkdir(parents=True)
            current.mkdir(parents=True)
            (future / "report.json").write_text("{}", encoding="utf-8")
            (current / "report.json").write_text("{}", encoding="utf-8")

            selected = report._default_compare_path(repo_root, current)

        self.assertIsNone(selected)

    def test_default_compare_path_skips_malformed_latest_prior_report(self) -> None:
        with _temp_root() as repo_root:
            root = repo_root / "scratch" / "nightly-self-improvement"
            older = root / "20260101T000000Z"
            malformed = root / "20260102T000000Z"
            current = root / "20260103T000000Z"
            older.mkdir(parents=True)
            malformed.mkdir(parents=True)
            current.mkdir(parents=True)
            (older / "report.json").write_text("{}", encoding="utf-8")
            (malformed / "report.json").write_text("{not-json", encoding="utf-8")
            (current / "report.json").write_text("{}", encoding="utf-8")

            selected = report._default_compare_path(repo_root, current)

        self.assertEqual(selected, older / "report.json")

    def test_trajectory_summaries_mark_missing_payloads_unavailable(self) -> None:
        expected = ["nebius-swe-agent-trajectories"]

        profile = report._trajectory_profile_summary({}, expected_datasets=expected)
        error = report._trajectory_error_summary({}, expected_datasets=expected)
        evidence = report._trajectory_evidence_summary({}, expected_datasets=expected)

        self.assertFalse(profile["available"])
        self.assertEqual(profile["expected_datasets"], expected)
        self.assertFalse(error["available"])
        self.assertEqual(error["expected_datasets"], expected)
        self.assertFalse(evidence["available"])
        self.assertEqual(evidence["expected_datasets"], expected)

    def test_collect_probe_slowest_ignores_non_numeric_elapsed_values(self) -> None:
        rows = report._collect_probe_slowest(
            {
                "rows": [
                    {"name": "bad", "elapsed_ms": "n/a"},
                    {"name": "good", "elapsed_ms": 12.5},
                ]
            }
        )

        self.assertEqual([row["name"] for row in rows], ["good"])

    def test_question_quality_summary_ignores_non_numeric_counts(self) -> None:
        summary = report._question_quality_summary(
            {"summary": {"cases": "n/a", "passed": 4, "failed": "n/a"}}
        )

        self.assertEqual(summary, {"cases": 0, "passed": 4, "failed": 0})

    def test_trajectory_error_summary_ignores_non_numeric_error_counts(self) -> None:
        summary = report._trajectory_error_summary(
            {
                "datasets": [
                    {"error_counts": {"bad": "n/a", "good": 2}},
                ]
            },
            expected_datasets=["nebius-swe-agent-trajectories"],
        )

        self.assertEqual(summary["top_errors"], [{"name": "good", "count": 2}])

    def test_trajectory_profile_summary_ignores_non_numeric_rows_profiled(self) -> None:
        summary = report._trajectory_profile_summary(
            {
                "datasets": [
                    {"rows_profiled": "n/a"},
                    {"rows_profiled": 5},
                ],
                "portfolio_recommendations": [{"id": "loop-cap"}],
            },
            expected_datasets=["nebius-swe-agent-trajectories"],
        )

        self.assertEqual(summary["rows_profiled"], 5)
        self.assertEqual(summary["top_recommendations"], ["loop-cap"])

    def test_trajectory_evidence_summary_ignores_non_numeric_evidence_counts(self) -> None:
        summary = report._trajectory_evidence_summary(
            {
                "portfolio_fix_coverage": [
                    {"id": "bad", "evidence_count": "n/a", "status": "partial"},
                    {"id": "good", "evidence_count": 3, "status": "implemented"},
                ]
            },
            expected_datasets=["nebius-swe-agent-trajectories"],
        )

        self.assertEqual(
            summary["top_fix_coverage"],
            [{"id": "good", "evidence_count": 3, "status": "implemented"}],
        )

    def test_trajectory_catalog_summary_ignores_non_numeric_counts(self) -> None:
        summary = report._trajectory_catalog_summary(
            {
                "summary": {
                    "entries": "n/a",
                    "local_entries": 2,
                    "analysis_ready_local_entries": "n/a",
                    "public_missing_entries": 1,
                    "gated_entries": "n/a",
                    "high_priority_public_missing": [{"id": "x"}],
                }
            }
        )

        self.assertEqual(summary["entries"], 0)
        self.assertEqual(summary["local_entries"], 2)
        self.assertEqual(summary["analysis_ready_local_entries"], 0)
        self.assertEqual(summary["public_missing_entries"], 1)
        self.assertEqual(summary["gated_entries"], 0)
        self.assertEqual(summary["high_priority_public_missing"], [{"id": "x"}])

    def test_trajectory_local_manifest_summary_ignores_non_numeric_file_count(self) -> None:
        with _temp_root() as data_root:
            dataset_dir = data_root / "nebius-swe-agent-trajectories"
            dataset_dir.mkdir(parents=True)
            (
                dataset_dir / report.trajectory_dataset_fetch.MANIFEST_NAME
            ).write_text(
                '{"repo_id":"nebius/SWE-agent-trajectories","resolved_revision":"abc123","downloaded_at":"2026-06-19T00:00:00+00:00","file_count":"n/a","files":["data/train-00000.parquet"]}',
                encoding="utf-8",
            )

            summary = report._trajectory_local_manifest_summary(
                data_root, ["nebius-swe-agent-trajectories"]
            )

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["file_count"], 0)

    def test_suggest_targets_ignores_non_numeric_metric_fields(self) -> None:
        suggestions = report._suggest_targets(
            {
                "question_quality": {"failed": "n/a"},
                "token_efficiency": {"summary": {"fail": "n/a", "fail_closed": "n/a"}, "delta": {"total_tokens": "n/a"}},
                "coding_benchmark": {"summary": {"fail": 0, "fail_closed": "n/a"}, "delta": {"total_tokens": "n/a"}},
                "python_sdk_search": {"summary": {"fail": "n/a"}},
                "slowest_tools": [{"tool": "search"}],
            },
            [{"name": "tool_speed_probe", "ok": True}],
        )

        self.assertEqual(
            suggestions,
            ["Profile deterministic tool path for search; it is the slowest recurring tool."],
        )

    def test_resolve_ollama_host_prefers_first_reachable_candidate(self) -> None:
        with patch.dict("os.environ", {"OLLAMA_HOST": ""}, clear=False), patch.object(
            report, "_host_responds", side_effect=lambda host: host == "http://[::1]:11434"
        ):
            selected = report._resolve_ollama_host()

        self.assertEqual(selected, "http://[::1]:11434")

    def test_build_report_includes_question_and_trajectory_commands(self) -> None:
        with _temp_root() as repo_root:
            output_dir = repo_root / "out"
            data_root = repo_root / "datasets"
            live_gate_summary_path = repo_root / "scratch" / "live-model-gate" / "live-model-gate-summary.json"
            (data_root / "nebius-swe-agent-trajectories").mkdir(parents=True)
            (
                data_root
                / "nebius-swe-agent-trajectories"
                / report.trajectory_dataset_fetch.MANIFEST_NAME
            ).write_text(
                '{"repo_id":"nebius/SWE-agent-trajectories","resolved_revision":"abc123","downloaded_at":"2026-06-19T00:00:00+00:00","file_count":1,"files":["data/train-00000.parquet"]}',
                encoding="utf-8",
            )
            (data_root / "nebius-swe-agent-trajectories" / "data").mkdir(parents=True)
            (data_root / "nebius-swe-agent-trajectories" / "data" / "train-00000.parquet").write_text("", encoding="utf-8")
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
                if name == "live-model-gate-summary.json":
                    return {
                        "ok": True,
                        "benchmark_suite": "local-small",
                        "selected_default_model": "granite4.1:8b",
                        "selection_reason": "Granite won the token tie-break.",
                        "models": [
                            {
                                "model": "granite4.1:8b",
                                "benchmark_passes": 8,
                                "benchmark_runs": 8,
                                "benchmark_total_tokens": 2040,
                                "benchmark_total_llm_calls": 4,
                                "benchmark_median_latency_s": 9.21,
                                "benchmark_artifact": "scratch/live-model-gate/coding-benchmark-granite4.1-8b.json",
                            }
                        ],
                    }
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

            with patch.object(report, "_repo_root", return_value=repo_root), patch.object(report, "_run_command", side_effect=fake_run_command), patch.object(report, "_run_command_with_env", side_effect=fake_run_command_with_env), patch.object(report, "_load_json", side_effect=fake_load_json), patch.object(report, "_git_value", return_value="main"), patch.object(report, "_resolve_ollama_host", return_value="http://[::1]:11434"), patch.object(report, "_latest_live_gate_summary_path", return_value=live_gate_summary_path):
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
        self.assertEqual(payload["metrics"]["trajectory_dataset_catalog"]["local_manifests"][0]["resolved_revision"], "abc123")
        self.assertEqual(payload["metrics"]["slowest_probe_tools"][0]["name"], "search")
        self.assertEqual(payload["metrics"]["slowest_probe_tools"][0]["elapsed_ms"], 12.0)
        self.assertEqual(payload["metrics"]["live_model_gate"]["selected_default_model"], "granite4.1:8b")
        self.assertEqual(payload["runtime"]["resolved_ollama_host"], "http://[::1]:11434")
        self.assertEqual(payload["runtime"]["llm_skip_reason"], "")
        self.assertEqual(payload["runtime"]["trajectory_skip_reason"], "")
        self.assertEqual(payload["runtime"]["trajectory_data_root"], str(data_root.resolve(strict=False)))
        self.assertEqual(payload["runtime"]["trajectory_profile_path"], str(output_dir / "trajectory-profile.json"))
        self.assertEqual(payload["runtime"]["trajectory_error_profile_path"], str(output_dir / "trajectory-error-profile.json"))
        self.assertEqual(payload["runtime"]["trajectory_evidence_report_path"], str(output_dir / "trajectory-evidence-report.json"))
        self.assertEqual(payload["runtime"]["live_gate_summary_path"], str(live_gate_summary_path))
        self.assertEqual(payload["summary"]["selected_default_model"], "granite4.1:8b")
        self.assertTrue(payload["summary"]["live_gate"]["available"])
        self.assertEqual(payload["summary"]["live_gate"]["benchmark_suite"], "local-small")
        self.assertTrue(payload["summary"]["trajectory"]["available"])
        self.assertTrue(any("nvidia/SWE-Hero-openhands-trajectories" in item for item in payload["metrics"]["suggested_implementation_targets"]))

    def test_live_model_gate_summary_rejects_incomplete_payload(self) -> None:
        path = Path("scratch/live-model-gate/live-model-gate-summary.json")

        summary = report._live_model_gate_summary({"models": []}, path=path)

        self.assertFalse(summary["available"])
        self.assertEqual(summary["path"], str(path))
        self.assertIsNone(summary["benchmark_suite"])

    def test_latest_live_gate_summary_path_skips_stale_incomplete_summary(self) -> None:
        with _temp_root() as repo_root:
            fixed_dir = repo_root / "scratch" / "live-model-gate"
            fixed_dir.mkdir(parents=True, exist_ok=True)
            fixed = fixed_dir / "live-model-gate-summary.json"
            fixed.write_text(json.dumps({"models": []}), encoding="utf-8")
            valid_dir = repo_root / "scratch" / "live-model-gate-20260629-local-small"
            valid_dir.mkdir(parents=True, exist_ok=True)
            valid = valid_dir / "live-model-gate-summary.json"
            valid.write_text(
                json.dumps(
                    {
                        "benchmark_suite": "local-small",
                        "selected_default_model": "granite4.1:8b",
                        "selection_reason": "Granite won the token tie-break.",
                        "models": [],
                    }
                ),
                encoding="utf-8",
            )

            selected = report._latest_live_gate_summary_path(repo_root)

        self.assertEqual(selected, valid)

    def test_build_report_ignores_malformed_explicit_compare_path(self) -> None:
        with _temp_root() as repo_root:
            output_dir = repo_root / "out"
            data_root = repo_root / "datasets"
            compare_path = repo_root / "broken-report.json"
            compare_path.write_text("{not-json", encoding="utf-8")
            args = self._args(output_dir, data_root)
            args.skip_llm = True
            args.compare = compare_path

            def fake_run_command(repo: Path, name: str, command: list[str], timeout: int, output_path: Path | None = None) -> dict[str, object]:
                self.assertEqual(repo, repo_root)
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
                    return {"summary": {"entries": 1, "local_entries": 0, "analysis_ready_local_entries": 0, "public_missing_entries": 1, "gated_entries": 0, "high_priority_public_missing": []}}
                return {}

            with patch.object(report, "_repo_root", return_value=repo_root), patch.object(report, "_run_command", side_effect=fake_run_command), patch.object(report, "_load_json", side_effect=fake_load_json), patch.object(report, "_git_value", return_value="main"), patch.object(report, "_resolve_ollama_host", side_effect=AssertionError("should not resolve ollama host when skip_llm is set")), patch.object(report, "_latest_live_gate_summary_path", return_value=None):
                payload = report.build_report(args)

        self.assertIsNone(payload["compare_path"])
        self.assertEqual(payload["metrics"]["token_efficiency"]["delta"], {})
        self.assertEqual(payload["metrics"]["coding_benchmark"]["delta"], {})

    def test_build_report_records_skip_llm_reason(self) -> None:
        with _temp_root() as repo_root:
            output_dir = repo_root / "out"
            data_root = repo_root / "datasets"
            (data_root / "nebius-swe-agent-trajectories").mkdir(parents=True)
            (
                data_root
                / "nebius-swe-agent-trajectories"
                / report.trajectory_dataset_fetch.MANIFEST_NAME
            ).write_text(
                '{"repo_id":"nebius/SWE-agent-trajectories","resolved_revision":"abc123","downloaded_at":"2026-06-19T00:00:00+00:00","file_count":1,"files":["data/train-00000.parquet"]}',
                encoding="utf-8",
            )
            (data_root / "nebius-swe-agent-trajectories" / "data").mkdir(parents=True)
            (data_root / "nebius-swe-agent-trajectories" / "data" / "train-00000.parquet").write_text("", encoding="utf-8")
            args = self._args(output_dir, data_root)
            args.skip_llm = True

            def fake_run_command(repo: Path, name: str, command: list[str], timeout: int, output_path: Path | None = None) -> dict[str, object]:
                self.assertEqual(repo, repo_root)
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
                    return {"summary": {"entries": 1, "local_entries": 1, "analysis_ready_local_entries": 1, "public_missing_entries": 0, "gated_entries": 0, "high_priority_public_missing": []}}
                if name == "trajectory-profile.json":
                    return {"datasets": [{"rows_profiled": 10}], "portfolio_recommendations": [{"id": "loop-cap"}]}
                if name == "trajectory-error-profile.json":
                    return {"datasets": [{"error_counts": {"test_assertion": 4}}]}
                if name == "trajectory-evidence-report.json":
                    return {"portfolio_fix_coverage": [{"id": "mechanical-router", "evidence_count": 12, "status": "partial"}]}
                return {}

            with patch.object(report, "_repo_root", return_value=repo_root), patch.object(report, "_run_command", side_effect=fake_run_command), patch.object(report, "_load_json", side_effect=fake_load_json), patch.object(report, "_git_value", return_value="main"), patch.object(report, "_resolve_ollama_host", side_effect=AssertionError("should not resolve ollama host when skip_llm is set")), patch.object(report, "_latest_live_gate_summary_path", return_value=None):
                payload = report.build_report(args)

        self.assertIsNone(payload["runtime"]["resolved_ollama_host"])
        self.assertEqual(payload["runtime"]["llm_skip_reason"], "LLM commands were skipped because --skip-llm was requested.")
        self.assertEqual(payload["runtime"]["trajectory_skip_reason"], "")
        self.assertEqual(payload["summary"]["selected_default_model"], None)
        self.assertFalse(payload["summary"]["live_gate"]["available"])
        self.assertTrue(payload["summary"]["trajectory"]["available"])

    def test_build_report_records_trajectory_skip_reason_when_no_local_datasets(self) -> None:
        with _temp_root() as repo_root:
            output_dir = repo_root / "out"
            data_root = repo_root / "datasets"
            args = self._args(output_dir, data_root)
            args.skip_llm = True

            def fake_run_command(repo: Path, name: str, command: list[str], timeout: int, output_path: Path | None = None) -> dict[str, object]:
                self.assertEqual(repo, repo_root)
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
                    return {"summary": {"entries": 1, "local_entries": 0, "analysis_ready_local_entries": 0, "public_missing_entries": 1, "gated_entries": 0, "high_priority_public_missing": [{"id": "nebius/SWE-agent-trajectories", "reason": "public"}]}}
                return {}

            with patch.object(report, "_repo_root", return_value=repo_root), patch.object(report, "_run_command", side_effect=fake_run_command), patch.object(report, "_load_json", side_effect=fake_load_json), patch.object(report, "_git_value", return_value="main"), patch.object(report, "_resolve_ollama_host", side_effect=AssertionError("should not resolve ollama host when skip_llm is set")), patch.object(report, "_latest_live_gate_summary_path", return_value=None):
                payload = report.build_report(args)

        self.assertEqual(
            payload["runtime"]["trajectory_skip_reason"],
            f"No local trajectory datasets found under {data_root}; skipped trajectory profile, error, and evidence commands.",
        )
        self.assertIsNone(payload["runtime"]["trajectory_profile_path"])
        self.assertFalse(payload["summary"]["trajectory"]["available"])

    def test_build_report_ignores_trajectory_dirs_without_valid_manifest(self) -> None:
        with _temp_root() as repo_root:
            output_dir = repo_root / "out"
            data_root = repo_root / "datasets"
            (data_root / "nebius-swe-agent-trajectories").mkdir(parents=True)
            args = self._args(output_dir, data_root)
            args.skip_llm = True

            def fake_run_command(repo: Path, name: str, command: list[str], timeout: int, output_path: Path | None = None) -> dict[str, object]:
                self.assertEqual(repo, repo_root)
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
                    return {"summary": {"entries": 1, "local_entries": 0, "analysis_ready_local_entries": 0, "public_missing_entries": 1, "gated_entries": 0, "high_priority_public_missing": [{"id": "nebius/SWE-agent-trajectories", "reason": "public"}]}}
                return {}

            with patch.object(report, "_repo_root", return_value=repo_root), patch.object(report, "_run_command", side_effect=fake_run_command), patch.object(report, "_load_json", side_effect=fake_load_json), patch.object(report, "_git_value", return_value="main"), patch.object(report, "_resolve_ollama_host", side_effect=AssertionError("should not resolve ollama host when skip_llm is set")), patch.object(report, "_latest_live_gate_summary_path", return_value=None):
                payload = report.build_report(args)

        command_names = [item["name"] for item in payload["commands"]]
        self.assertNotIn("trajectory_profile", command_names)
        self.assertNotIn("trajectory_error_profile", command_names)
        self.assertNotIn("trajectory_evidence_report", command_names)
        self.assertEqual(
            payload["runtime"]["trajectory_skip_reason"],
            f"No local trajectory datasets found under {data_root}; skipped trajectory profile, error, and evidence commands.",
        )

    def test_build_report_ignores_stale_trajectory_manifest_with_missing_files(self) -> None:
        with _temp_root() as repo_root:
            output_dir = repo_root / "out"
            data_root = repo_root / "datasets"
            dataset_dir = data_root / "nebius-swe-agent-trajectories"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / report.trajectory_dataset_fetch.MANIFEST_NAME).write_text(
                (
                    '{"repo_id":"nebius/SWE-agent-trajectories",'
                    '"resolved_revision":"abc123",'
                    '"downloaded_at":"2026-06-19T00:00:00+00:00",'
                    '"file_count":1,'
                    '"files":["data/train-00000.parquet"]}'
                ),
                encoding="utf-8",
            )
            args = self._args(output_dir, data_root)
            args.skip_llm = True

            def fake_run_command(repo: Path, name: str, command: list[str], timeout: int, output_path: Path | None = None) -> dict[str, object]:
                self.assertEqual(repo, repo_root)
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
                    return {"summary": {"entries": 1, "local_entries": 0, "analysis_ready_local_entries": 0, "public_missing_entries": 1, "gated_entries": 0, "high_priority_public_missing": [{"id": "nebius/SWE-agent-trajectories", "reason": "public"}]}}
                return {}

            with patch.object(report, "_repo_root", return_value=repo_root), patch.object(report, "_run_command", side_effect=fake_run_command), patch.object(report, "_load_json", side_effect=fake_load_json), patch.object(report, "_git_value", return_value="main"), patch.object(report, "_resolve_ollama_host", side_effect=AssertionError("should not resolve ollama host when skip_llm is set")), patch.object(report, "_latest_live_gate_summary_path", return_value=None):
                payload = report.build_report(args)

        command_names = [item["name"] for item in payload["commands"]]
        self.assertNotIn("trajectory_profile", command_names)
        self.assertNotIn("trajectory_error_profile", command_names)
        self.assertNotIn("trajectory_evidence_report", command_names)
        self.assertEqual(
            payload["runtime"]["trajectory_skip_reason"],
            f"No local trajectory datasets found under {data_root}; skipped trajectory profile, error, and evidence commands.",
        )


if __name__ == "__main__":
    unittest.main()
