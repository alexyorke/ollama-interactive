from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts import trajectory_profile as profile


class TrajectoryProfileTests(unittest.TestCase):
    def test_extract_openhands_events_reads_tool_calls(self) -> None:
        row = {
            "trajectory": [
                {"role": "system", "content": "sys"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "read_file", "arguments": "{\"path\":\"a.py\"}"}},
                        {"function": {"name": "run_test", "arguments": "{\"command\":\"pytest\"}"}},
                    ],
                },
                {"role": "tool", "name": "read_file", "content": "file body"},
            ]
        }

        events = profile._extract_events("openhands", row)

        self.assertEqual([event.name for event in events[:2]], ["read_file", "run_test"])
        self.assertEqual(events[0].category, "read")
        self.assertEqual(events[1].category, "test")

    def test_shell_tool_call_with_pytest_is_reclassified_as_test(self) -> None:
        row = {
            "trajectory": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "execute_bash",
                                "arguments": "{\"command\":\"pytest -q tests/test_app.py\"}",
                            }
                        }
                    ],
                }
            ]
        }

        events = profile._extract_events("openhands", row)

        self.assertEqual(events[0].name, "execute_bash")
        self.assertEqual(events[0].category, "test")

    def test_extract_smith_events_accepts_json_string_messages(self) -> None:
        messages = (
            '[{"role":"assistant","tool_calls":[{"function":{"name":"replace_symbols","arguments":"{}"}}]},'
            '"noise",'
            '{"role":"tool","name":"run_test","content":"ok"}]'
        )

        events = profile._extract_smith_events(messages)

        self.assertEqual([event.name for event in events], ["replace_symbols", "run_test"])
        self.assertEqual([event.category for event in events], ["edit", "test"])

    def test_summarize_dataset_flags_context_loops_and_missing_tests(self) -> None:
        rows = [
            {
                "trajectory": [
                    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "search", "arguments": "{}"}}]},
                    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": "{}"}}]},
                    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "search_symbols", "arguments": "{}"}}]},
                    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_symbol", "arguments": "{}"}}]},
                    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "replace_symbol", "arguments": "{}"}}]},
                ]
            },
            {
                "trajectory": [
                    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": "{}"}}]},
                    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "search", "arguments": "{}"}}]},
                    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "replace_symbol", "arguments": "{}"}}]},
                    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "run_test", "arguments": "{}"}}]},
                ]
            },
        ]

        summary = profile._summarize_dataset("sample", "openhands", rows)

        self.assertEqual(summary["rows_profiled"], 2)
        self.assertGreater(summary["context_loop_rows_pct"], 0)
        self.assertGreater(summary["edit_without_later_test_pct"], 0)
        self.assertTrue(summary["recommendations"])
        recommendation_ids = {item["id"] for item in summary["recommendations"]}
        self.assertIn("context-planner", recommendation_ids)
        self.assertIn("loop-cap", recommendation_ids)

    def test_summarize_dataset_counts_mechanical_turn_candidates(self) -> None:
        rows = [
            {
                "trajectory": [
                    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "search", "arguments": "{}"}}]},
                    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": "{}"}}]},
                ]
            },
            {
                "trajectory": [
                    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "git_status", "arguments": "{}"}}]},
                    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "git_diff", "arguments": "{}"}}]},
                ]
            },
            {
                "trajectory": [
                    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": "{}"}}]},
                    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "replace_symbol", "arguments": "{}"}}]},
                ]
            },
        ]

        summary = profile._summarize_dataset("sample", "openhands", rows)

        self.assertEqual(summary["mechanical_turn_candidates_pct"], 66.67)
        self.assertEqual(summary["mechanical_turn_category_counts"]["read_search_only"], 1)
        self.assertEqual(summary["mechanical_turn_category_counts"]["git_only"], 1)
        recommendation_ids = {item["id"] for item in summary["recommendations"]}
        self.assertIn("mechanical-router", recommendation_ids)

    def test_portfolio_recommendations_merge_by_id(self) -> None:
        datasets = [
            {
                "dataset": "a",
                "recommendations": [
                    {
                        "id": "loop-cap",
                        "priority": "high",
                        "title": "Force loop-breaking after repeated context-only actions",
                        "change_type": "controller",
                        "trigger": "context_loop_rows_pct=30",
                        "rationale": "loop",
                        "expected_effect": "less looping",
                        "experiments": ["exp1", "exp2"],
                    }
                ],
            },
            {
                "dataset": "b",
                "recommendations": [
                    {
                        "id": "loop-cap",
                        "priority": "high",
                        "title": "Force loop-breaking after repeated context-only actions",
                        "change_type": "controller",
                        "trigger": "context_loop_rows_pct=40",
                        "rationale": "loop",
                        "expected_effect": "less looping",
                        "experiments": ["exp1", "exp2"],
                    }
                ],
            },
        ]

        portfolio = profile._portfolio_recommendations(datasets)

        self.assertEqual(len(portfolio), 1)
        self.assertEqual(portfolio[0]["id"], "loop-cap")
        self.assertEqual(portfolio[0]["seen_in_datasets"], 2)

    def test_iter_parquet_rows_honors_max_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.parquet"
            table = pa.table({"value": [1, 2, 3, 4]})
            pq.write_table(table, path)

            rows = list(profile._iter_parquet_rows([path], max_rows=2))

        self.assertEqual(rows, [{"value": 1}, {"value": 2}])


if __name__ == "__main__":
    unittest.main()
