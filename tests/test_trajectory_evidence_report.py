import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import trajectory_evidence_report as report


class TrajectoryEvidenceReportTests(unittest.TestCase):
    def test_extract_message_records_preserves_message_and_tool_call(self) -> None:
        row = {
            "trajectory": [
                {
                    "role": "assistant",
                    "content": "I'll inspect the file first.",
                    "tool_calls": [
                        {"function": {"name": "read_file", "arguments": "{\"path\":\"a.py\"}"}},
                    ],
                },
                {"role": "tool", "name": "read_file", "content": "def f():\n    return 1\n"},
            ]
        }

        records = report.extract_message_records("sample", "openhands", row, 0)

        self.assertEqual(records[0].kind, "message")
        self.assertEqual(records[0].tool_call_names, ("read_file",))
        self.assertEqual(records[1].kind, "tool_call")
        self.assertEqual(records[1].name, "read_file")

    def test_extract_message_records_infers_inline_tool_call_for_smith(self) -> None:
        row = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "I will inspect the tree first.\n<function=bash> <parameter=command>find /testbed -maxdepth 2</parameter> </function>",
                }
            ]
        }

        records = report.extract_message_records("sample", "smith", row, 0)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].kind, "message")
        self.assertEqual(records[0].tool_call_names, ("run_shell",))
        self.assertEqual(records[1].kind, "tool_call")
        self.assertEqual(records[1].name, "run_shell")

    def test_extract_message_records_does_not_label_swe_agent_system_prompt_as_tool(self) -> None:
        row = {
            "trajectory": [
                {"role": "system", "system_prompt": "You can use read_file and search to inspect the repo."},
                {"role": "assistant", "text": "I will open the file next."},
            ]
        }

        records = report.extract_message_records("sample", "swe_agent", row, 0)

        self.assertEqual(records[0].name, "")
        self.assertEqual(records[0].tool_call_names, ())
        self.assertEqual(records[1].tool_call_names, ("read_file",))
        self.assertEqual(records[2].kind, "tool_call")
        self.assertEqual(records[2].name, "read_file")

    def test_classify_message_themes_finds_large_failure_blob(self) -> None:
        record = report._make_message_record(
            dataset="sample",
            row_id="row-1",
            message_index=3,
            role="tool",
            kind="message",
            name="run_test",
            category="test",
            content=("Traceback (most recent call last):\n" + ("AssertionError\n" * 80)),
        )

        themes = report.classify_message_themes(record)

        self.assertIn("large-failure-blob", themes)
        self.assertEqual(record.error_class, "test_assertion")

    def test_summarize_dataset_groups_row_patterns_and_fixes(self) -> None:
        rows = [
            {
                "trajectory": [
                    {"role": "user", "content": "Use read_file on app.py and reply with the token only."},
                    {"role": "assistant", "content": "Phase 1. I will inspect several files before deciding.", "tool_calls": []},
                    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "search", "arguments": "{}"}}]},
                    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "search", "arguments": "{}"}}]},
                    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "search", "arguments": "{}"}}]},
                    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "search", "arguments": "{}"}}]},
                    {"role": "tool", "name": "run_test", "content": "FAILED test_app.py::test_x\n" + ("AssertionError: 1 != 2\n" * 40)},
                ]
            }
        ]

        summary = report.summarize_dataset("sample", "openhands", rows)
        fixes = report._fix_evidence_rows(summary)

        self.assertEqual(summary["rows_profiled"], 1)
        self.assertEqual(summary["tool_call_events"], 4)
        self.assertEqual(summary["message_tool_call_records"], 4)
        self.assertGreater(summary["message_theme_counts"]["large-failure-blob"], 0)
        self.assertGreater(summary["message_theme_counts"]["explicit-tool-request"], 0)
        self.assertEqual(summary["row_pattern_counts"]["context-loop-row"], 1)
        fix_ids = {row["id"] for row in fixes}
        self.assertIn("loop-cap", fix_ids)
        self.assertIn("diagnose-test-failure", fix_ids)
        fix_map = {row["id"]: row for row in fixes}
        self.assertEqual(fix_map["loop-cap"]["evidence_count"], 1)

    def test_format_markdown_mentions_fix_coverage(self) -> None:
        payload = {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "data_root": "scratch/external/datasets",
            "max_rows_per_dataset": 1,
            "portfolio_fix_coverage": [
                {
                    "id": "mechanical-router",
                    "evidence_count": 12,
                    "datasets": ["sample"],
                    "status": "partial",
                    "summary": "Deterministic routing exists.",
                    "references": ["ollama_code/controller/navigation_validation.py"],
                    "next_gap": "More extraction needed.",
                }
            ],
            "datasets": [
                {
                    "dataset": "sample",
                    "rows_profiled": 1,
                    "messages_profiled": 2,
                    "tool_call_events": 1,
                    "avg_tool_calls": 1.0,
                    "context_loop_rows_pct": 0.0,
                    "edit_without_prior_context_pct": 0.0,
                    "edit_without_later_test_pct": 0.0,
                    "mechanical_turn_candidates_pct": 100.0,
                    "message_theme_counts": {"explicit-tool-request": 1},
                    "error_counts": {"test_assertion": 2},
                    "row_pattern_counts": {"mechanical-turn-row": 1},
                    "message_theme_examples": {"explicit-tool-request": [{"dataset": "sample", "row_id": "row-1", "message_index": 0, "role": "user", "excerpt": "Use read_file"}]},
                    "error_examples": {},
                    "recommendations": [],
                }
            ],
        }

        markdown = report.format_markdown(payload)

        self.assertIn("Trajectory Evidence Report", markdown)
        self.assertIn("Portfolio Fix Coverage", markdown)
        self.assertIn("mechanical-router", markdown)

    def test_build_report_attaches_dataset_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            dataset_root = data_root / "nebius-swe-agent-trajectories"
            (dataset_root / "data").mkdir(parents=True)
            (dataset_root / report.trajectory_dataset_fetch.MANIFEST_NAME).write_text(
                '{"repo_id":"nebius/SWE-agent-trajectories","resolved_revision":"abc123","file_count":1,"downloaded_at":"2026-06-19T00:00:00+00:00"}',
                encoding="utf-8",
            )
            with patch.object(report, "_iter_dataset_rows", return_value=("swe_agent", iter([{"trajectory": []}]))):
                payload = report.build_report(data_root, ["nebius-swe-agent-trajectories"], max_rows=1)

        summary = payload["datasets"][0]
        self.assertEqual(summary["source_manifest"]["repo_id"], "nebius/SWE-agent-trajectories")
        self.assertEqual(summary["source_manifest"]["resolved_revision"], "abc123")


if __name__ == "__main__":
    unittest.main()
