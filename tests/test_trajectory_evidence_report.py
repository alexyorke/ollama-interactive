import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    pa = None
    pq = None

from scripts import trajectory_evidence_report as report


class TrajectoryEvidenceReportTests(unittest.TestCase):
    def test_build_parser_accepts_legacy_output_flag(self) -> None:
        args = report._build_parser().parse_args(["--output", "scratch/evidence-report.json"])

        self.assertEqual(args.output, Path("scratch/evidence-report.json"))

    def test_resolve_output_paths_from_legacy_output_json_path(self) -> None:
        output_json, output_md = report._resolve_output_paths(
            output=Path("scratch/evidence-report.json"),
            output_json=None,
            output_md=None,
        )

        self.assertEqual(output_json, Path("scratch/evidence-report.json"))
        self.assertEqual(output_md, Path("scratch/evidence-report.md"))

    def test_iter_dataset_rows_preserves_openhands_messages_column(self) -> None:
        if pa is None or pq is None:
            self.skipTest("pyarrow is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            dataset = "coderforge-preview-swe-bench-verified-trajectories"
            path = data_root / dataset / "trajectory" / "train-00000-of-00001.parquet"
            path.parent.mkdir(parents=True)
            table = pa.table(
                {
                    "trajectory_id": ["demo-1"],
                    "messages": ['[{"role":"assistant","content":"pytest -q tests/test_app.py"}]'],
                }
            )
            pq.write_table(table, path)

            adapter, rows = report._iter_dataset_rows(data_root, dataset, max_rows=1)
            records = report.extract_message_records(dataset, adapter, next(iter(rows)), 0)

        self.assertEqual(adapter, "openhands")
        self.assertTrue(any(record.tool_call_names == ("run_test",) for record in records))

    def test_iter_dataset_rows_preserves_openhands_messages_json_column(self) -> None:
        if pa is None or pq is None:
            self.skipTest("pyarrow is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            dataset = "nebius-swe-rebench-openhands-trajectories"
            path = data_root / dataset / "trajectories.parquet"
            path.parent.mkdir(parents=True)
            table = pa.table(
                {
                    "trajectory_id": ["demo-1"],
                    "messages_json": [
                        (
                            '[{"role":"assistant","content":"",'
                            '"tool_calls_json":"[{\\"function\\":{\\"name\\":\\"execute_bash\\",\\"arguments\\":\\"{\\\\\\"command\\\\\\":\\\\\\"pytest -q\\\\\\"}\\"}}]"},'
                            '{"role":"tool","name":"execute_bash","content":"AssertionError: expected 1"}]'
                        )
                    ],
                }
            )
            pq.write_table(table, path)

            adapter, rows = report._iter_dataset_rows(data_root, dataset, max_rows=1)
            records = report.extract_message_records(dataset, adapter, next(iter(rows)), 0)

        self.assertEqual(adapter, "openhands")
        self.assertTrue(any(record.kind == "tool_call" and record.name == "execute_bash" for record in records))

    def test_iter_dataset_rows_preserves_cc_bench_trajectory_column(self) -> None:
        if pa is None or pq is None:
            self.skipTest("pyarrow is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            dataset = "cc-bench-trajectories"
            path = data_root / dataset / "train.parquet"
            path.parent.mkdir(parents=True)
            table = pa.table(
                {
                    "task_id": ["cc-1"],
                    "trajectory": [
                        json.dumps(
                            [
                                {
                                    "type": "assistant",
                                    "message": {
                                        "role": "assistant",
                                        "content": [
                                            {
                                                "type": "tool_use",
                                                "id": "call_1",
                                                "name": "Bash",
                                                "input": {"command": "pytest -q tests/test_app.py"},
                                            }
                                        ],
                                    },
                                },
                                {
                                    "type": "user",
                                    "message": {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "tool_result",
                                                "tool_use_id": "call_1",
                                                "content": "FAILED tests/test_app.py::test_x\nE   assert 1 == 2",
                                            }
                                        ],
                                    },
                                },
                            ]
                        )
                    ],
                }
            )
            pq.write_table(table, path)

            adapter, rows = report._iter_dataset_rows(data_root, dataset, max_rows=1)
            records = report.extract_message_records(dataset, adapter, next(iter(rows)), 0)

        self.assertEqual(adapter, "cc_bench")
        self.assertTrue(any(record.kind == "tool_call" and record.name == "bash" for record in records))
        self.assertTrue(any(record.error_class == "test_assertion" for record in records))

    def test_iter_dataset_rows_preserves_thoughtworks_messages_json_column(self) -> None:
        if pa is None or pq is None:
            self.skipTest("pyarrow is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            dataset = "thoughtworks-agentic-coding-trajectories"
            path = data_root / dataset / "sessions.parquet"
            path.parent.mkdir(parents=True)
            table = pa.table(
                {
                    "agent_framework": ["mini-swe-agent"],
                    "session_id": ["tw-1"],
                    "messages_json": [
                        (
                            '[{"role":"assistant","content":"THOUGHT: repro\\n```bash\\npytest -q tests/test_app.py\\n```"},'
                            '{"role":"user","content":"Traceback\\nAssertionError: expected 1"}]'
                        )
                    ],
                }
            )
            pq.write_table(table, path)

            adapter, rows = report._iter_dataset_rows(data_root, dataset, max_rows=1)
            records = report.extract_message_records(dataset, adapter, next(iter(rows)), 0)

        self.assertEqual(adapter, "thoughtworks")
        self.assertTrue(any(record.kind == "tool_call" and record.name == "run_test" for record in records))
        self.assertTrue(any(record.error_class == "test_assertion" for record in records))

    def test_iter_dataset_rows_preserves_trace_commons_messages_column(self) -> None:
        if pa is None or pq is None:
            self.skipTest("pyarrow is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            dataset = "trace-commons-agent-traces"
            path = data_root / dataset / "data" / "train-00000-of-00001.parquet"
            path.parent.mkdir(parents=True)
            table = pa.table(
                {
                    "session_id": ["tc-1"],
                    "messages": [
                        [
                            '{"role":"assistant","content":"","tool_calls":[{"function":{"name":"Bash","arguments":{"command":"pytest -q tests/test_app.py"}}}]}',
                            '{"role":"tool","name":"Bash","content":"AssertionError: expected 1"}',
                        ]
                    ],
                }
            )
            pq.write_table(table, path)

            adapter, rows = report._iter_dataset_rows(data_root, dataset, max_rows=1)
            records = report.extract_message_records(dataset, adapter, next(iter(rows)), 0)

        self.assertEqual(adapter, "trace_commons")
        self.assertTrue(any(record.kind == "tool_call" and record.name == "bash" for record in records))
        self.assertTrue(any(record.error_class == "test_assertion" for record in records))

    def test_iter_dataset_rows_preserves_trace_commons_messages_json_column(self) -> None:
        if pa is None or pq is None:
            self.skipTest("pyarrow is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            dataset = "trace-commons-agent-traces"
            path = data_root / dataset / "data" / "train-00000-of-00001.parquet"
            path.parent.mkdir(parents=True)
            table = pa.table(
                {
                    "session_id": ["tc-json"],
                    "messages_json": [
                        (
                            '[{"role":"assistant","content":"","tool_calls":[{"function":{"name":"Bash","arguments":{"command":"pytest -q tests/test_app.py"}}}]},'
                            '{"role":"tool","name":"Bash","content":"FAILED tests/test_app.py::test_x\\nE   assert 1 == 2"}]'
                        )
                    ],
                }
            )
            pq.write_table(table, path)

            adapter, rows = report._iter_dataset_rows(data_root, dataset, max_rows=1)
            records = report.extract_message_records(dataset, adapter, next(iter(rows)), 0)

        self.assertEqual(adapter, "trace_commons")
        self.assertTrue(any(record.kind == "tool_call" and record.name == "bash" for record in records))
        self.assertTrue(any(record.error_class == "test_assertion" for record in records))

    def test_iter_dataset_rows_preserves_trace_commons_trace_column(self) -> None:
        if pa is None or pq is None:
            self.skipTest("pyarrow is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            dataset = "trace-commons-agent-traces"
            path = data_root / dataset / "data" / "train-00000-of-00001.parquet"
            path.parent.mkdir(parents=True)
            table = pa.table(
                {
                    "session_id": ["tc-structured"],
                    "messages": [json.dumps([])],
                    "trace": [
                        json.dumps(
                            {
                                "messages": [
                                    {
                                        "info": {"role": "assistant"},
                                        "parts": [
                                            {"type": "reasoning", "text": "I will search for expert sources first."},
                                            {"type": "tool", "tool": "websearch", "state": {"input": {"query": "kernel anti cheat security"}}},
                                        ],
                                    }
                                ]
                            }
                        )
                    ],
                }
            )
            pq.write_table(table, path)

            adapter, rows = report._iter_dataset_rows(data_root, dataset, max_rows=1)
            records = report.extract_message_records(dataset, adapter, next(iter(rows)), 0)

        self.assertEqual(adapter, "trace_commons")
        self.assertTrue(any(record.kind == "tool_call" and record.name == "websearch" for record in records))

    def test_iter_dataset_rows_preserves_agent_race_jsonl_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            dataset = "agent-race-traces"
            path = data_root / dataset / "pi-kimi.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text(
                "\n".join(
                    [
                        (
                            '{"type":"message","message":{"role":"assistant","content":['
                            '{"type":"toolCall","id":"functions.bash:0","name":"bash","arguments":{"command":"python -m pytest tests/test_app.py -q"}}'
                            ']}}'
                        ),
                        (
                            '{"type":"message","message":{"role":"toolResult","toolCallId":"functions.bash:0","toolName":"bash","content":['
                            '{"type":"text","text":"FAILED tests/test_app.py::test_x\\nE   assert 1 == 2"}'
                            ']}}'
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            adapter, rows = report._iter_dataset_rows(data_root, dataset, max_rows=1)
            records = report.extract_message_records(dataset, adapter, next(iter(rows)), 0)

        self.assertEqual(adapter, "agent_race")
        self.assertTrue(any(record.kind == "tool_call" and record.name == "bash" for record in records))
        self.assertTrue(any(record.error_class == "test_assertion" for record in records))

    def test_reference_row_pattern_counts_ignores_non_numeric_reference_metrics(self) -> None:
        counts = report._reference_row_pattern_counts(
            {
                "reference_trajectory_metrics": {
                    "rows_profiled": "n/a",
                    "rows_with_edit_pct": "n/a",
                    "mechanical_turn_candidates_pct": "n/a",
                    "context_loop_rows_pct": 25.0,
                    "edit_without_prior_context_pct": "n/a",
                    "edit_without_later_test_pct": 50.0,
                }
            }
        )

        self.assertEqual(
            counts,
            {
                "mechanical-turn-row": 0,
                "context-loop-row": 0,
                "edit-without-context-row": 0,
                "edit-without-test-row": 0,
            },
        )

    def test_build_report_treats_max_rows_zero_as_unbounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            dataset = "agent-race-traces"
            first = data_root / dataset / "pi-kimi.jsonl"
            first.parent.mkdir(parents=True)
            first.write_text(
                "\n".join(
                    [
                        '{"type":"message","message":{"role":"assistant","content":[{"type":"toolCall","id":"functions.bash:0","name":"bash","arguments":{"command":"python -m pytest tests/test_a.py -q"}}]}}',
                        '{"type":"message","message":{"role":"toolResult","toolCallId":"functions.bash:0","toolName":"bash","content":[{"type":"text","text":"FAILED tests/test_a.py::test_x\\nE   assert 1 == 2"}]}}',
                    ]
                ),
                encoding="utf-8",
            )
            second = data_root / dataset / "claude-code.jsonl"
            second.write_text(
                "\n".join(
                    [
                        '{"type":"message","message":{"role":"assistant","content":[{"type":"toolCall","id":"functions.bash:1","name":"bash","arguments":{"command":"python -m pytest tests/test_b.py -q"}}]}}',
                        '{"type":"message","message":{"role":"toolResult","toolCallId":"functions.bash:1","toolName":"bash","content":[{"type":"text","text":"FAILED tests/test_b.py::test_y\\nE   assert 2 == 3"}]}}',
                    ]
                ),
                encoding="utf-8",
            )

            payload = report.build_report(data_root, [dataset], max_rows=0)

        summary = payload["datasets"][0]
        self.assertEqual(summary["rows_profiled"], 2)
        self.assertEqual(summary["error_counts"]["test_assertion"], 2)

    def test_iter_dataset_rows_skips_terminalbench_null_steps_before_max_rows(self) -> None:
        if pa is None or pq is None:
            self.skipTest("pyarrow is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            dataset = "terminalbench-trajectories"
            path = data_root / dataset / "data" / "train-00000-of-00001.parquet"
            path.parent.mkdir(parents=True)
            table = pa.table(
                {
                    "steps": [
                        "null",
                        '[{"src":"agent","msg":"Executed Bash","tools":[{"fn":"bash_command","cmd":"python -m pytest -q"}],"obs":"Exit code 0"}]',
                    ],
                    "trial_id": ["skip", "keep"],
                    "trial_name": ["skip-trial", "keep-trial"],
                    "task_name": ["warmup", "real-task"],
                    "agent": ["agent", "agent"],
                    "model": ["model", "model"],
                }
            )
            pq.write_table(table, path)

            adapter, rows = report._iter_dataset_rows(data_root, dataset, max_rows=1)
            row = next(iter(rows))

        self.assertEqual(adapter, "terminalbench")
        self.assertEqual(row["trial_id"], "keep")
        self.assertIn("bash_command", row["steps"])

    def test_extract_openhands_messages_accepts_json_string_messages(self) -> None:
        row = {
            "instance_id": "demo-1",
            "messages": (
                '[{"role":"assistant","tool_calls":[{"function":{"name":"execute_bash","arguments":"{\\"command\\":\\"pytest -q\\"}"}}]},'
                '{"role":"tool","name":"execute_bash","content":"AssertionError: expected 1"}]'
            ),
        }

        records = report.extract_message_records("sample", "openhands", row, 0)

        self.assertGreaterEqual(len(records), 3)
        self.assertIn("execute_bash", [record.name for record in records if record.kind == "tool_call"])
        shell_calls = [record for record in records if record.kind == "tool_call" and record.name == "execute_bash"]
        self.assertEqual(shell_calls[0].tool_arguments, {"command": "pytest -q"})
        self.assertTrue(any(record.error_class == "test_assertion" for record in records))

    def test_extract_openhands_messages_accepts_messages_json_and_tool_calls_json(self) -> None:
        row = {
            "instance_id": "demo-1",
            "messages_json": (
                '[{"role":"assistant","content":"",'
                '"tool_calls_json":"[{\\"function\\":{\\"name\\":\\"execute_bash\\",\\"arguments\\":\\"{\\\\\\"command\\\\\\":\\\\\\"pytest -q\\\\\\"}\\"}}]"},'
                '{"role":"tool","name":"execute_bash","content":"AssertionError: expected 1"}]'
            ),
        }

        records = report.extract_message_records("sample", "openhands", row, 0)

        self.assertTrue(any(record.kind == "tool_call" and record.name == "execute_bash" for record in records))
        shell_calls = [record for record in records if record.kind == "tool_call" and record.name == "execute_bash"]
        self.assertEqual(shell_calls[0].tool_arguments, {"command": "pytest -q"})
        self.assertTrue(any(record.error_class == "test_assertion" for record in records))

    def test_extract_openhands_messages_include_reasoning_content(self) -> None:
        row = {
            "instance_id": "demo-1",
            "trajectory": [
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "I should inspect src/app.py and then run pytest -q tests/test_app.py.",
                    "tool_calls": [
                        {
                            "id": "chatcmpl-tool-1",
                            "function": {
                                "name": "execute_bash",
                                "arguments": "{\"command\":\"pytest -q tests/test_app.py\"}",
                            },
                        }
                    ],
                }
            ],
        }

        records = report.extract_message_records("sample", "openhands", row, 0)

        assistant_messages = [record for record in records if record.kind == "message" and record.role == "assistant"]
        self.assertEqual(len(assistant_messages), 1)
        self.assertIn("inspect src/app.py", assistant_messages[0].content)
        self.assertIn("pytest -q tests/test_app.py", assistant_messages[0].content)

    def test_extract_openhands_messages_exclude_embedded_tool_use_json_from_message_content(self) -> None:
        row = {
            "instance_id": "demo-1",
            "trajectory": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "chatcmpl-tool-1",
                            "name": "execute_bash",
                            "input": {"command": "pytest -q tests/test_app.py"},
                        }
                    ],
                    "reasoning_content": "I should inspect src/app.py and then run pytest -q tests/test_app.py.",
                }
            ],
        }

        records = report.extract_message_records("sample", "openhands", row, 0)

        assistant_message = next(record for record in records if record.kind == "message" and record.role == "assistant")
        self.assertEqual(
            assistant_message.content,
            "I should inspect src/app.py and then run pytest -q tests/test_app.py.",
        )

    def test_summarize_dataset_does_not_count_invalid_args_from_str_replace_editor_cat_output(self) -> None:
        row = {
            "trajectory": [
                {
                    "role": "tool",
                    "name": "str_replace_editor",
                    "content": (
                        "Here's the result of running `cat -n` on /workspace/tests/test_schemapi.py:\n"
                        "1 import copy\n"
                        "2 import io\n"
                        "3 parser.add_argument('--timeout', default=None)\n"
                        "4 msg = 'error: unrecognized arguments: --timeout=60'\n"
                        "5 assert 'usage:' in msg.lower()\n"
                    ),
                }
            ]
        }

        summary = report.summarize_dataset("sample", "openhands", [row])

        self.assertEqual(summary["error_counts"], {})
        self.assertEqual(summary["error_examples"], {})

    def test_extract_message_record_does_not_assign_shell_error_to_str_replace_editor_cat_output(self) -> None:
        record = report._make_message_record(
            dataset="sample",
            row_id="row-1",
            message_index=17,
            role="tool",
            kind="message",
            name="str_replace_editor",
            category="edit",
            content=(
                "Here's the result of running `cat -n` on /workspace/tests/test_schemapi.py:\n"
                "1 import copy\n"
                "2 import io\n"
                "3 parser.add_argument('--timeout', default=None)\n"
                "4 msg = 'error: unrecognized arguments: --timeout=60'\n"
                "5 assert 'usage:' in msg.lower()\n"
            ),
        )

        self.assertIsNone(record.error_class)
        self.assertIsNone(record.shell_error)

    def test_extract_message_record_does_not_assign_other_shell_error_to_str_replace_editor_status_text(self) -> None:
        record = report._make_message_record(
            dataset="sample",
            row_id="row-1",
            message_index=18,
            role="tool",
            kind="message",
            name="str_replace_editor",
            category="edit",
            content="The file has been edited. Here is the snippet with error-handling logic and validation status.",
        )

        self.assertIsNone(record.error_class)
        self.assertIsNone(record.shell_error)

    def test_extract_message_record_does_not_assign_other_shell_error_to_str_replace_editor_code_snippet(self) -> None:
        record = report._make_message_record(
            dataset="sample",
            row_id="row-1",
            message_index=53,
            role="tool",
            kind="message",
            name="str_replace_editor",
            category="edit",
            content=(
                "Here's the result of running `cat -n` on /workspace/index.js:\n"
                '655    "UNKNOWN_LAUNCH_FUNCTION": "try telling the application what to do instead of opening it",\n'
                "656    // when a request type was not recognized\n"
                '657    "invalid_request_type": "error: not a valid request",\n'
            ),
        )

        self.assertIsNone(record.error_class)
        self.assertIsNone(record.shell_error)

    def test_extract_message_record_does_not_assign_invalid_args_to_user_issue_description(self) -> None:
        record = report._make_message_record(
            dataset="sample",
            row_id="row-1",
            message_index=1,
            role="user",
            kind="message",
            name="",
            category="other",
            content=(
                "<issue_description>\n"
                "epubcheck 5.1.0 does not recognize `--failonwarnings+`, nor `--failonwarnings-`\n"
                "According to the issue report, the CLI does not recognize these switches.\n"
                "</issue_description>"
            ),
        )

        self.assertIsNone(record.error_class)
        self.assertIsNone(record.shell_error)

    def test_extract_message_record_does_not_assign_invalid_args_to_assistant_reasoning(self) -> None:
        record = report._make_message_record(
            dataset="sample",
            row_id="row-1",
            message_index=24,
            role="assistant",
            kind="message",
            name="",
            category="edit",
            content=(
                "Now I've reproduced the panic. The issue is that the unrecognized argument error is generated\n"
                "during validation, so I should inspect the parser and then patch the recovery path.\n"
            ),
        )

        self.assertIsNone(record.error_class)
        self.assertIsNone(record.shell_error)

    def test_extract_openhands_messages_recovers_unknown_tool_name_from_tool_call_id(self) -> None:
        row = {
            "session_id": "tc-1",
            "messages": [
                '{"role":"assistant","content":"","tool_calls":[{"id":"toolu_123","function":{"name":"Write","arguments":{"file_path":"PROJECT.md","content":"x"}}}]}',
                '{"role":"tool","tool_call_id":"toolu_123","name":"unknown_tool","content":"File created successfully"}',
            ],
        }

        records = report.extract_message_records("sample", "trace_commons", row, 0)

        tool_messages = [record for record in records if record.role == "tool" and record.kind == "message"]
        self.assertEqual(tool_messages[0].name, "write")
        self.assertTrue(any(record.kind == "tool_call" and record.name == "write" for record in records))

    def test_extract_trace_commons_messages_accepts_structured_trace_fallback(self) -> None:
        row = {
            "session_id": "tc-structured",
            "messages": json.dumps([]),
            "trace": json.dumps(
                {
                    "messages": [
                        {
                            "info": {"role": "user"},
                            "parts": [{"type": "text", "text": "Investigate kernel anti-cheat tradeoffs."}],
                        },
                        {
                            "info": {"role": "assistant"},
                            "parts": [
                                {"type": "reasoning", "text": "I will search for expert sources first."},
                                {"type": "tool", "tool": "websearch", "state": {"input": {"query": "kernel anti cheat security"}}},
                            ],
                        },
                    ]
                }
            ),
        }

        records = report.extract_message_records("sample", "trace_commons", row, 0)

        self.assertTrue(any(record.kind == "message" and record.role == "assistant" for record in records))
        self.assertTrue(any(record.kind == "tool_call" and record.name == "websearch" for record in records))

    def test_shell_command_from_record_accepts_powershell_tool(self) -> None:
        row = {
            "session_id": "tc-1",
            "messages": [
                '{"role":"assistant","content":"","tool_calls":[{"function":{"name":"PowerShell","arguments":{"command":"pytest -q tests/test_app.py"}}}]}'
            ],
        }

        records = report.extract_message_records("sample", "trace_commons", row, 0)
        tool_call = next(record for record in records if record.kind == "tool_call")

        self.assertEqual(tool_call.name, "powershell")
        self.assertEqual(report._shell_command_from_record(tool_call), "pytest -q tests/test_app.py")

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
        self.assertEqual(records[1].tool_arguments, {"command": "find /testbed -maxdepth 2"})

    def test_extract_smith_messages_fall_back_to_messages_json_when_messages_is_empty_list_string(self) -> None:
        row = {
            "messages": "[]",
            "messages_json": (
                '[{"role":"assistant","content":"THOUGHT: repro\\n```bash\\npytest -q tests/test_app.py\\n```"},'
                '{"role":"user","content":"Traceback\\nAssertionError: expected 1"}]'
            ),
        }

        records = report.extract_message_records("sample", "smith", row, 0)

        self.assertTrue(any(record.kind == "tool_call" and record.name == "run_test" for record in records))
        self.assertTrue(any(record.kind == "message" and record.role == "user" and record.name == "run_test" for record in records))

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

    def test_extract_message_records_preserves_swe_agent_shell_command(self) -> None:
        row = {
            "trajectory": [
                {
                    "role": "assistant",
                    "text": (
                        "Let's run the checker.\n"
                        "```bash\n"
                        "python -m flake8 fail.py\n"
                        "```"
                    ),
                }
            ]
        }

        records = report.extract_message_records("sample", "swe_agent", row, 0)

        self.assertEqual(records[0].tool_call_names, ("run_shell",))
        self.assertEqual(records[1].kind, "tool_call")
        self.assertEqual(records[1].name, "run_shell")
        self.assertEqual(records[1].tool_arguments, {"command": "python -m flake8 fail.py"})

    def test_extract_message_records_preserves_swe_agent_observation_context(self) -> None:
        row = {
            "trajectory": [
                {"role": "user", "text": "The issue description mentions SyntaxError but this is only prompt text."},
                {
                    "role": "assistant",
                    "text": "Let's run the checker.\n```bash\npython -m flake8 fail.py\n```",
                },
                {
                    "role": "user",
                    "text": "Traceback (most recent call last):\nImportError: cannot import name 'x' from 'pkg'",
                },
            ]
        }

        records = report.extract_message_records("sample", "swe_agent", row, 0)

        observation = next(record for record in records if record.kind == "message" and record.error_class == "import_error")
        self.assertEqual(observation.name, "run_shell")
        self.assertEqual(observation.category, "shell")

    def test_extract_message_records_ignore_import_error_in_successful_read_file_source_snippet(self) -> None:
        row = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Read the file.\n<function=read_file>\n<parameter=path>theme.py</parameter>\n</function>",
                },
                {
                    "role": "user",
                    "content": (
                        "OBSERVATION:\n"
                        "<returncode>0</returncode>\n"
                        "<output>\n"
                        "from __future__ import annotations\n\n"
                        "from .prettytable import PrettyTable\n\n"
                        "try:\n"
                        "    from colorama import init\n"
                        "except ImportError:\n"
                        "    pass\n"
                        "</output>"
                    ),
                },
            ]
        }

        records = report.extract_message_records("sample", "smith", row, 0)

        observation = next(record for record in records if record.kind == "message" and record.role == "user")
        self.assertEqual(observation.name, "read_file")
        self.assertIsNone(observation.error_class)

    def test_extract_message_records_ignore_test_assertion_in_successful_read_file_source_snippet(self) -> None:
        row = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Read the file.\n<function=read_file>\n<parameter=path>test_app.py</parameter>\n</function>",
                },
                {
                    "role": "user",
                    "content": (
                        "OBSERVATION:\n"
                        "<returncode>0</returncode>\n"
                        "<output>\n"
                        "def test_example() -> None:\n"
                        "    if actual != expected:\n"
                        "        raise AssertionError(f'{actual} != {expected}')\n"
                        "    return None\n"
                        "</output>"
                    ),
                },
            ]
        }

        records = report.extract_message_records("sample", "smith", row, 0)

        observation = next(record for record in records if record.kind == "message" and record.role == "user")
        self.assertEqual(observation.name, "read_file")
        self.assertIsNone(observation.error_class)

    def test_extract_message_records_ignore_path_missing_in_str_replace_editor_cat_preview(self) -> None:
        row = {
            "trajectory": [
                {
                    "role": "tool",
                    "name": "str_replace_editor",
                    "content": (
                        "Here's the result of running `cat -n` on /workspace/app.py:\n"
                        "     1\tdef build_message() -> str:\n"
                        "     2\t    return 'No such file or directory'\n"
                        "     3\t\n"
                        "     4\tdef main() -> None:\n"
                        "     5\t    print(build_message())\n"
                    ),
                }
            ]
        }

        records = report.extract_message_records("sample", "openhands", row, 0)

        observation = next(record for record in records if record.kind == "message" and record.role == "tool")
        self.assertEqual(observation.name, "str_replace_editor")
        self.assertIsNone(observation.error_class)

    def test_extract_message_records_ignore_expected_syntax_error_example_in_passing_smoke_test(self) -> None:
        row = {
            "messages": [
                {
                    "role": "tool",
                    "name": "bash",
                    "content": (
                        "FrostCore Stage 1 -- Smoke Test\n"
                        "================================\n"
                        "  PASS [rt_sys]: eval(1 + 1) => ok\n"
                        "  PASS [rt_sys/syntax]: eval(this is not valid JS %%%) => error: SyntaxError: expecting ';'\n"
                        "  PASS [rt_count/ref]: eval(undeclaredVariable.x.y.z) => error: ReferenceError: undeclaredVariable is not defined\n"
                        "================================\n"
                        "All smoke tests PASSED.\n"
                    ),
                }
            ]
        }

        records = report.extract_message_records("sample", "trace_commons", row, 0)

        observation = next(record for record in records if record.kind == "message" and record.role == "tool")
        self.assertEqual(observation.name, "bash")
        self.assertIsNone(observation.error_class)

    def test_extract_message_records_ignore_expected_invalid_args_example_in_cli_test_transcript(self) -> None:
        row = {
            "trajectory": [
                {
                    "role": "assistant",
                    "text": "Run the CLI checks.\n```bash\nnode test-cli.js\n```",
                },
                {
                    "role": "user",
                    "text": (
                        "OBSERVATION:\n"
                        "Test CLI:\n"
                        "Test 1 - Default (bare):\n"
                        "const a = 1;\n"
                        "---\n"
                        "Test 4 - --no-bare with --use-js-modules (should fail):\n"
                        "Error: --no-bare cannot be used with --modernize-js or --use-js-modules\n"
                        "Exit code: 1\n"
                        "---\n"
                        "Test 6 - Invalid option (should fail):\n"
                        "Error: unrecognized option '--invalid-option'\n"
                        "Exit code: 1\n"
                    ),
                }
            ]
        }

        records = report.extract_message_records("sample", "swe_agent", row, 0)

        observation = next(record for record in records if record.kind == "message" and record.role == "user")
        self.assertEqual(observation.name, "run_shell")
        self.assertIsNone(observation.error_class)

    def test_extract_message_records_ignore_expected_invalid_args_in_parser_recovery_suite(self) -> None:
        row = {
            "trajectory": [
                {
                    "role": "tool",
                    "name": "bash",
                    "content": (
                        "PASS 35_pragma_jsx.jsx\n"
                        "  PASS 36_pragma_auto.jsx\n"
                        "  FAIL [exit=1] 44_recover_mode.jsx\n"
                        "    stderr: error: unknown option '--recover'\n"
                        "  PASS [exit=1] err_01_mismatched_tags.jsx\n"
                        "  PASS [exit=1] err_02_unclosed_tag.jsx\n"
                        "Results: \n"
                        "  47 passed\n"
                        "  1 failed\n"
                    ),
                }
            ]
        }

        records = report.extract_message_records("sample", "openhands", row, 0)

        observation = next(record for record in records if record.kind == "message" and record.role == "tool")
        self.assertEqual(observation.name, "bash")
        self.assertIsNone(observation.error_class)

    def test_extract_message_records_ignore_expected_import_error_in_success_wrapper(self) -> None:
        row = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Run the example suite.\n<function=bash>\n<parameter=command>python test_examples.py</parameter>\n</function>",
                },
                {
                    "role": "user",
                    "content": (
                        "<returncode>0</returncode>\n"
                        "<output>\n"
                        "Test case 1: Basic case\n"
                        "example.py: \"my-plugin\" failed during execution due to ValueError('Something went wrong')\n\n"
                        "Test case 2: Different exception type\n"
                        "test_file.py: \"another-plugin\" failed during execution due to ImportError('Module not found')\n"
                        "</output>"
                    ),
                },
            ]
        }

        records = report.extract_message_records("sample", "smith", row, 0)

        observation = next(record for record in records if record.kind == "message" and record.role == "user")
        self.assertEqual(observation.name, "run_shell")
        self.assertIsNone(observation.error_class)

    def test_extract_message_records_ignore_expected_assertion_error_in_success_wrapper(self) -> None:
        row = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Run the example suite.\n<function=bash>\n<parameter=command>python test_examples.py</parameter>\n</function>",
                },
                {
                    "role": "user",
                    "content": (
                        "<returncode>0</returncode>\n"
                        "<output>\n"
                        "Keys: ['id', 'name', 'email']\n"
                        "Values: [1, 'John', 'john@example.com']\n"
                        "Error: \n"
                        "Type: <class 'AssertionError'>\n"
                        "</output>"
                    ),
                },
            ]
        }

        records = report.extract_message_records("sample", "smith", row, 0)

        observation = next(record for record in records if record.kind == "message" and record.role == "user")
        self.assertEqual(observation.name, "run_shell")
        self.assertIsNone(observation.error_class)

    def test_extract_message_records_classify_timed_out_tool_result(self) -> None:
        row = {
            "trajectory": [
                {
                    "role": "tool",
                    "name": "bash",
                    "content": "Command timed out after 2m 0.0s\nServing HTTP on 0.0.0.0 port 8000",
                }
            ]
        }

        records = report.extract_message_records("sample", "openhands", row, 0)

        observation = next(record for record in records if record.kind == "message" and record.role == "tool")
        self.assertEqual(observation.name, "bash")
        self.assertEqual(observation.error_class, "timeout")

    def test_extract_message_records_ignore_shell_code_snippet_invalid_args(self) -> None:
        row = {
            "trajectory": [
                {
                    "role": "tool",
                    "name": "bash",
                    "content": (
                        "OBSERVATION:\n"
                        "import { RulesetValidationError } from '@stoplight/spectral-core';\n"
                        "import testFunction from './__helpers__/tester';\n"
                        "const runXor = testFunction.bind(null, xor);\n"
                        "new RulesetValidationError('invalid-function-options', 'bad options', ['rules', 'my-rule']);\n"
                        "it.each([[null]])\n"
                    ),
                }
            ]
        }

        records = report.extract_message_records("sample", "openhands", row, 0)

        observation = next(record for record in records if record.kind == "message" and record.role == "tool")
        self.assertEqual(observation.name, "bash")
        self.assertIsNone(observation.error_class)

    def test_extract_message_records_ignore_shell_code_snippet_import_error(self) -> None:
        row = {
            "trajectory": [
                {
                    "role": "tool",
                    "name": "execute_bash",
                    "content": (
                        "func TestGenerateObservation(t *testing.T) {\n"
                        '\tid := 0\n'
                        '\towner := "chief"\n'
                        '\tpathWithNamespace := "path/to/cool-project"\n'
                        '\tnameWithNamespace := "name/to/cool-project"\n'
                        '\timportError := "none"\n'
                        "\treturn\n"
                        "}\n"
                        "[The command completed with exit code 0.]\n"
                        "[Current working directory: /workspace/crossplane-contrib__provider-gitlab__1.0]\n"
                        "[Command finished with exit code 0]\n"
                    ),
                }
            ]
        }

        records = report.extract_message_records("sample", "openhands", row, 0)

        observation = next(record for record in records if record.kind == "message" and record.role == "tool")
        self.assertEqual(observation.name, "execute_bash")
        self.assertIsNone(observation.error_class)

    def test_extract_message_records_preserves_swe_agent_edit_observation_context(self) -> None:
        row = {
            "trajectory": [
                {
                    "role": "assistant",
                    "text": "I will patch it.\n``` edit 138:138 print('x') ```",
                },
                {
                    "role": "user",
                    "text": "Your proposed edit has introduced new syntax error(s).\nERRORS:\n- E999 IndentationError: unexpected indent",
                },
            ]
        }

        records = report.extract_message_records("sample", "swe_agent", row, 0)

        observation = next(record for record in records if record.kind == "message" and record.error_class == "syntax_error")
        self.assertEqual(observation.name, "edit")
        self.assertEqual(observation.category, "edit")

    def test_extract_message_records_preserves_plain_python_observation_context(self) -> None:
        row = {
            "trajectory": [
                {
                    "role": "assistant",
                    "text": "Let's run it.\n``` python reproduce.py ```",
                },
                {
                    "role": "user",
                    "text": "Traceback (most recent call last):\nImportError: cannot import name 'Config' from 'lexicon.config'",
                },
            ]
        }

        records = report.extract_message_records("sample", "swe_agent", row, 0)

        observation = next(record for record in records if record.kind == "message" and record.error_class == "import_error")
        self.assertEqual(observation.name, "run_shell")
        self.assertEqual(observation.category, "shell")

    def test_extract_terminalbench_records_preserve_shell_command_and_observation(self) -> None:
        row = {
            "trial_id": "trial-1",
            "steps": (
                '[{"src":"agent","msg":"Executed Bash","tools":[{"fn":"bash_command","cmd":"python -m pytest tests/test_app.py -q"}],'
                '"obs":"Exit code 1\\nFAILED tests/test_app.py::test_x\\nE   assert 1 == 2"}]'
            ),
        }

        records = report.extract_message_records("sample", "terminalbench", row, 0)

        tool_call = next(record for record in records if record.kind == "tool_call")
        observation = next(record for record in records if record.kind == "message" and record.role == "tool")
        self.assertEqual(tool_call.name, "run_shell")
        self.assertEqual(tool_call.tool_arguments, {"command": "python -m pytest tests/test_app.py -q"})
        self.assertEqual(observation.name, "run_shell")
        self.assertEqual(observation.error_class, "test_assertion")

    def test_extract_terminalbench_records_normalize_bash_and_shell_to_run_shell(self) -> None:
        row = {
            "trial_id": "trial-1",
            "steps": (
                '[{"src":"agent","msg":"Executed Bash","tools":[{"fn":"Bash","cmd":"which R"}],"obs":"Exit code 1"},'
                '{"src":"agent","msg":"Executed shell","tools":[{"fn":"shell","cmd":["bash","-lc","R --version"]}],"obs":"bash: line 1: R: command not found"}]'
            ),
        }

        records = report.extract_message_records("sample", "terminalbench", row, 0)

        tool_calls = [record for record in records if record.kind == "tool_call"]
        observations = [record for record in records if record.kind == "message" and record.role == "tool"]
        self.assertEqual([record.name for record in tool_calls], ["run_shell", "run_shell"])
        self.assertEqual([record.name for record in observations], ["run_shell", "run_shell"])

    def test_extract_terminalbench_records_use_task_name_when_user_prompt_is_warmup(self) -> None:
        row = {
            "trial_id": "trial-1",
            "task_name": "adaptive-rejection-sampler",
            "steps": '[{"src":"user","msg":"Warmup","tools":null,"obs":null}]',
        }

        records = report.extract_message_records("sample", "terminalbench", row, 0)

        user_message = next(record for record in records if record.kind == "message" and record.role == "user")
        self.assertEqual(user_message.content, "adaptive-rejection-sampler")

    def test_extract_terminalbench_records_strip_environment_context_prefix_from_user_prompt(self) -> None:
        row = {
            "trial_id": "trial-1",
            "task_name": "adaptive-rejection-sampler",
            "steps": (
                '[{"src":"user","msg":"<environment_context>\\n<cwd>/app</cwd>\\n<approval_policy>never</approval_policy>\\n</environment_context>\\n'
                'Implement adaptive rejection sampling in sampler.py and run tests.","tools":null,"obs":null}]'
            ),
        }

        records = report.extract_message_records("sample", "terminalbench", row, 0)

        user_message = next(record for record in records if record.kind == "message" and record.role == "user")
        self.assertEqual(user_message.content, "Implement adaptive rejection sampling in sampler.py and run tests.")

    def test_extract_terminalbench_records_use_task_name_when_user_prompt_is_placeholder(self) -> None:
        row = {
            "trial_id": "trial-1",
            "task_name": "adaptive-rejection-sampler",
            "steps": '[{"src":"user","msg":"$31","tools":null,"obs":null}]',
        }

        records = report.extract_message_records("sample", "terminalbench", row, 0)

        user_message = next(record for record in records if record.kind == "message" and record.role == "user")
        self.assertEqual(user_message.content, "adaptive-rejection-sampler")

    def test_extract_agent_race_records_preserve_tool_calls_and_results(self) -> None:
        row = {
            "session_id": "race-1",
            "events": [
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "inspect"},
                            {
                                "type": "toolCall",
                                "id": "functions.bash:0",
                                "name": "bash",
                                "arguments": {"command": "python -m pytest tests/test_app.py -q"},
                            },
                        ],
                    },
                },
                {
                    "type": "message",
                    "message": {
                        "role": "toolResult",
                        "toolCallId": "functions.bash:0",
                        "toolName": "bash",
                        "content": [{"type": "text", "text": "FAILED tests/test_app.py::test_x\nE   assert 1 == 2"}],
                    },
                },
            ],
        }

        records = report.extract_message_records("sample", "agent_race", row, 0)

        tool_call = next(record for record in records if record.kind == "tool_call")
        observation = next(record for record in records if record.kind == "message" and record.role == "tool")
        self.assertEqual(tool_call.name, "bash")
        self.assertEqual(tool_call.tool_arguments, {"command": "python -m pytest tests/test_app.py -q"})
        self.assertEqual(observation.name, "bash")
        self.assertEqual(observation.error_class, "test_assertion")

    def test_extract_agent_race_tool_result_wrapper_does_not_duplicate_message_record(self) -> None:
        row = {
            "session_id": "race-1",
            "events": [
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tooluse_2",
                                "name": "bash",
                                "input": {"command": "python -m pytest tests/test_app.py -q"},
                            }
                        ],
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "tooluse_2", "content": "FAILED tests/test_app.py::test_x\nE   assert 1 == 2"},
                        ],
                    },
                },
            ],
        }

        records = report.extract_message_records("sample", "agent_race", row, 0)

        tool_messages = [record for record in records if record.kind == "message" and record.role == "tool"]
        user_messages = [record for record in records if record.kind == "message" and record.role == "user"]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(len(user_messages), 0)
        self.assertEqual(tool_messages[0].content, "FAILED tests/test_app.py::test_x\nE   assert 1 == 2")

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
        self.assertIn("failure-compression", fix_ids)
        fix_map = {row["id"]: row for row in fixes}
        self.assertEqual(fix_map["loop-cap"]["evidence_count"], 1)
        self.assertGreater(fix_map["failure-compression"]["evidence_count"], 0)

    def test_summarize_dataset_error_counts_ignore_prompt_mentions_but_keep_observations(self) -> None:
        swe_agent_rows = [
            {
                "trajectory": [
                    {"role": "user", "text": "The issue description mentions SyntaxError but this is only prompt text."},
                    {"role": "assistant", "text": "Let's run the checker.\n```bash\npython -m flake8 fail.py\n```"},
                    {"role": "user", "text": "Traceback (most recent call last):\nImportError: cannot import name 'x' from 'pkg'"},
                ]
            }
        ]
        openhands_rows = [
            {
                "trajectory": [
                    {"role": "user", "content": "Task text says AssertionError but this is only background context."},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"function": {"name": "run_test", "arguments": "{}"}}],
                    },
                    {"role": "tool", "name": "run_test", "content": "FAILED tests/test_app.py::test_x\nE   assert 1 == 2"},
                ]
            }
        ]

        swe_agent_summary = report.summarize_dataset("agent-sample", "swe_agent", swe_agent_rows)
        openhands_summary = report.summarize_dataset("oh-sample", "openhands", openhands_rows)

        self.assertEqual(swe_agent_summary["error_counts"], {"import_error": 1})
        self.assertEqual(openhands_summary["error_counts"], {"test_assertion": 1})

    def test_summarize_dataset_counts_shell_command_shapes(self) -> None:
        rows = [
            {
                "trajectory": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"function": {"name": "execute_bash", "arguments": '{"command":"cd /testbed && find . -name \\"*test*\\" -type d"}'}},
                            {"function": {"name": "execute_bash", "arguments": '{"command":"grep needle README.md"}'}},
                            {"function": {"name": "execute_bash", "arguments": '{"command":"python -m pytest tests/test_app.py -v"}'}},
                            {"function": {"name": "execute_bash", "arguments": '{"command":"python -c \\"print(1)\\""}'}},
                            {"function": {"name": "execute_bash", "arguments": '{"command":"python reproduce_issue.py"}'}},
                            {"function": {"name": "execute_bash", "arguments": '{"command":"find . -name \\"*.py\\" -exec grep -l needle {} ;"}'}},
                        ],
                    }
                ]
            }
        ]

        summary = report.summarize_dataset("sample", "openhands", rows)

        self.assertEqual(summary["shell_command_counts"]["find"], 2)
        self.assertEqual(summary["shell_command_counts"]["grep"], 1)
        self.assertEqual(summary["shell_command_shape_counts"]["find PATH -name GLOB -type ARG"], 1)
        self.assertEqual(summary["shell_command_shape_counts"]["grep ARG ARG"], 1)
        self.assertEqual(summary["shell_command_intent_counts"]["file-discovery"], 1)
        self.assertEqual(summary["shell_command_intent_counts"]["text-search"], 1)
        self.assertEqual(summary["shell_command_intent_counts"]["python-test-command"], 1)
        self.assertEqual(summary["shell_command_intent_counts"]["python-inline-probe"], 1)
        self.assertEqual(summary["shell_command_intent_counts"]["python-script-run"], 1)
        self.assertEqual(summary["shell_command_intent_counts"]["find-grep-search"], 1)

    def test_shell_command_family_normalizes_find_dot_to_find(self) -> None:
        family, shape = report._shell_command_family_and_shape('find. -name "*.py" -exec grep -H "def main(" {} \\;')
        intent = report._shell_command_intent('find. -name "*.py" -exec grep -H "def main(" {} \\;')

        self.assertEqual(family, "find")
        self.assertEqual(shape, "find -name GLOB -exec ARG -h ARG {}")
        self.assertEqual(intent, "find-grep-search")

    def test_shell_command_helpers_ignore_pseudo_shell_wrappers(self) -> None:
        command = '# setup\nif ! command -v R >/dev/null; then echo installing && apt-get install -y r-base; fi'
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "apt-get")
        self.assertEqual(shape, "apt-get ARG -y ARG")
        self.assertEqual(intent, "dependency-install")
        self.assertEqual(report._shell_command_family_and_shape("$37"), ("", ""))
        self.assertEqual(report._shell_command_intent("$37"), "unknown")
        self.assertEqual(report._shell_command_family_and_shape("#' roxygen comment only"), ("", ""))
        self.assertEqual(report._shell_command_intent("#' roxygen comment only"), "unknown")
        self.assertEqual(report._shell_command_family_and_shape("q"), ("", ""))
        self.assertEqual(report._shell_command_intent("q"), "unknown")
        self.assertEqual(report._shell_command_family_and_shape("C-z"), ("", ""))
        self.assertEqual(report._shell_command_intent("C-z"), "unknown")
        self.assertEqual(report._shell_command_family_and_shape("C-d"), ("", ""))
        self.assertEqual(report._shell_command_intent("C-d"), "unknown")

    def test_shell_command_helpers_unwrap_python_list_bash_wrappers(self) -> None:
        family, shape = report._shell_command_family_and_shape("['bash', '-lc', 'pytest -q tests/test_app.py']")
        intent = report._shell_command_intent("['bash', '-lc', 'pytest -q tests/test_app.py']")

        self.assertEqual(family, "pytest")
        self.assertEqual(shape, "pytest -q PATH")
        self.assertEqual(intent, "test-command")

    def test_shell_command_helpers_skip_echo_wrapper_before_real_command(self) -> None:
        command = json.dumps(["bash", "-lc", "echo 'Running test()'; Rscript -e \"source('/app/ars.R'); test()\""])
        family, shape = report._shell_command_family_and_shape(
            command
        )
        intent = report._shell_command_intent(
            command
        )

        self.assertEqual(family, "rscript")
        self.assertEqual(shape, "rscript -e PATH")
        self.assertEqual(intent, "r-test-command")

    def test_shell_command_helpers_classify_dependency_probe(self) -> None:
        family, shape = report._shell_command_family_and_shape("which R")
        intent = report._shell_command_intent("which R")

        self.assertEqual(family, "which")
        self.assertEqual(shape, "which ARG")
        self.assertEqual(intent, "dependency-probe")

    def test_shell_command_helpers_classify_r_heredoc_test_command(self) -> None:
        command = "Rscript << 'EOF'\nsource(\"ars.R\")\ntest()\nEOF"
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "rscript")
        self.assertEqual(shape, "rscript ARG ARG ARG ARG ARG")
        self.assertEqual(intent, "r-test-command")

    def test_shell_command_helpers_classify_timeout_wrapped_r_test_command(self) -> None:
        command = "timeout 120 Rscript -e 'source(\"ars.R\"); test()' 2>&1"
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "timeout")
        self.assertEqual(shape, "timeout N ARG -e ARG ARG")
        self.assertEqual(intent, "r-test-command")

    def test_shell_command_helpers_classify_update_then_install_as_dependency_install(self) -> None:
        family, shape = report._shell_command_family_and_shape("apt-get update && apt-get install -y r-base")
        intent = report._shell_command_intent("apt-get update && apt-get install -y r-base")

        self.assertEqual(family, "apt-get")
        self.assertEqual(shape, "apt-get ARG")
        self.assertEqual(intent, "dependency-install")

    def test_shell_command_helpers_classify_wc_as_file_inspection(self) -> None:
        family, shape = report._shell_command_family_and_shape("wc -l /app/ars.R")
        intent = report._shell_command_intent("wc -l /app/ars.R")

        self.assertEqual(family, "wc")
        self.assertEqual(shape, "wc -l PATH")
        self.assertEqual(intent, "file-inspection")

    def test_shell_command_helpers_classify_cp_as_file_mutation(self) -> None:
        family, shape = report._shell_command_family_and_shape("cp /app/ars_fixed.R /app/ars.R")
        intent = report._shell_command_intent("cp /app/ars_fixed.R /app/ars.R")

        self.assertEqual(family, "cp")
        self.assertEqual(shape, "cp PATH PATH")
        self.assertEqual(intent, "file-mutation")

    def test_shell_command_helpers_skip_line_continued_echo_wrappers(self) -> None:
        command = (
            'echo "=== FINAL VERIFICATION ===" && \\\n'
            'echo "" && \\\n'
            'echo "1. Implementation File:" && \\\n'
            'ls -lh /app/ars.R'
        )
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "ls")
        self.assertEqual(shape, "ls -lh PATH")
        self.assertEqual(intent, "file-inspection")

    def test_shell_command_helpers_skip_sleep_wrapper_before_cat(self) -> None:
        command = "sleep 30 && cat /tmp/claude/-app/tasks/b548a15.output 2>/dev/null || echo 'Output file not found'"
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "cat")
        self.assertEqual(shape, "cat PATH PATH")
        self.assertEqual(intent, "file-inspection")

    def test_shell_command_helpers_classify_nl_sed_pipeline_as_file_inspection(self) -> None:
        command = """['bash', '-lc', "nl -ba /app/ars.R | sed -n '1,260p'"]"""
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "nl")
        self.assertEqual(shape, "nl -ba PATH ARG ARG -n ARG")
        self.assertEqual(intent, "file-inspection")

    def test_shell_command_helpers_classify_shell_wrapped_apply_patch_as_file_mutation(self) -> None:
        command = (
            "['bash', '-lc', \"apply_patch << 'PATCH'\\n"
            "*** Begin Patch\\n"
            "*** Update File: /app/ars.R\\n"
            "@@\\n"
            "-old\\n"
            "+new\\n"
            "*** End Patch\\n"
            "PATCH\"]"
        )
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "apply_patch")
        self.assertEqual(shape, "apply_patch ARG ARG GLOB ARG ARG GLOB ARG")
        self.assertEqual(intent, "file-mutation")

    def test_shell_command_helpers_classify_list_wrapped_apply_patch_as_file_mutation(self) -> None:
        command = "['apply_patch', '*** Begin Patch\\n*** Update File: ars.R\\n@@\\n-old\\n+new\\n*** End Patch']"
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "apply_patch")
        self.assertEqual(shape, "apply_patch GLOB ARG ARG GLOB ARG ARG ARG")
        self.assertEqual(intent, "file-mutation")

    def test_summarize_dataset_classifies_terminalbench_list_wrapped_apply_patch(self) -> None:
        rows = [
            {
                "trial_id": "trial-1",
                "steps": json.dumps(
                    [
                        {
                            "src": "agent",
                            "msg": "Executed shell patch",
                            "tools": [
                                {"fn": "shell", "cmd": "['apply_patch', '*** Begin Patch\\n*** Update File: ars.R\\n@@\\n-old\\n+new\\n*** End Patch']"},
                            ],
                            "obs": "",
                        }
                    ]
                ),
            }
        ]

        summary = report.summarize_dataset("sample", "terminalbench", rows)

        self.assertEqual(summary["shell_command_counts"]["apply_patch"], 1)
        self.assertEqual(summary["shell_command_intent_counts"]["file-mutation"], 1)

    def test_shell_command_helpers_classify_sed_print_as_file_inspection(self) -> None:
        family, shape = report._shell_command_family_and_shape("sed -n '80,95p' /app/ars.R")
        intent = report._shell_command_intent("sed -n '80,95p' /app/ars.R")

        self.assertEqual(family, "sed")
        self.assertEqual(shape, "sed -n ARG PATH")
        self.assertEqual(intent, "file-inspection")

    def test_summarize_dataset_classifies_terminalbench_sed_print_as_file_inspection(self) -> None:
        rows = [
            {
                "trial_id": "trial-1",
                "steps": json.dumps(
                    [
                        {
                            "src": "agent",
                            "msg": "Executed shell read",
                            "tools": [
                                {"fn": "run_shell", "cmd": "sed -n '80,95p' /app/ars.R"},
                            ],
                            "obs": "",
                        }
                    ]
                ),
            }
        ]

        summary = report.summarize_dataset("sample", "terminalbench", rows)

        self.assertEqual(summary["shell_command_counts"]["sed"], 1)
        self.assertEqual(summary["shell_command_intent_counts"]["file-inspection"], 1)

    def test_shell_command_helpers_classify_ps_grep_pipeline_as_process_inspection(self) -> None:
        command = 'ps aux | grep -i apt | grep -v grep || echo "No apt process running"'
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "ps")
        self.assertEqual(shape, "ps ARG ARG ARG -i ARG ARG ARG")
        self.assertEqual(intent, "process-inspection")

    def test_shell_command_helpers_classify_killall_as_process_control(self) -> None:
        command = "killall python"
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "killall")
        self.assertEqual(shape, "killall ARG")
        self.assertEqual(intent, "process-control")

    def test_shell_command_helpers_classify_kill_pipeline_as_process_control(self) -> None:
        command = "kill -9 $(ps aux | grep python | grep -v grep | awk '{print $2}')"
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "kill")
        self.assertEqual(shape, "kill -9 ARG ARG ARG ARG ARG ARG")
        self.assertEqual(intent, "process-control")

    def test_shell_command_helpers_classify_pkill_as_process_control(self) -> None:
        command = 'pkill -f python || echo "No python processes to kill"'
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "pkill")
        self.assertEqual(shape, "pkill -f ARG")
        self.assertEqual(intent, "process-control")

    def test_shell_command_helpers_classify_timeout_wrapped_r_probe_as_dependency_probe(self) -> None:
        command = "timeout 300 bash -c 'while ! command -v R &> /dev/null; do sleep 5; done; R --version'"
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "timeout")
        self.assertEqual(shape, "timeout N ARG -c PATH")
        self.assertEqual(intent, "dependency-probe")

    def test_shell_command_helpers_classify_wait_then_which_r_as_dependency_probe(self) -> None:
        command = "wait 5017 2>/dev/null; sleep 5; which R && R --version | head -3"
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "which")
        self.assertEqual(shape, "which ARG")
        self.assertEqual(intent, "dependency-probe")

    def test_shell_command_helpers_classify_timeout_wrapped_dpkg_probe_as_dependency_probe(self) -> None:
        command = "timeout 10 bash -c 'dpkg -l | grep r-base' || echo \"R not installed\""
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "timeout")
        self.assertEqual(shape, "timeout N ARG -c ARG")
        self.assertEqual(intent, "dependency-probe")

    def test_shell_command_helpers_classify_sed_in_place_as_file_mutation(self) -> None:
        command = "sed -i '345s/ars/result <- ars/' /app/ars.R"
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "sed")
        self.assertEqual(shape, "sed -i PATH PATH")
        self.assertEqual(intent, "file-mutation")

    def test_summarize_dataset_classifies_terminalbench_sed_in_place_as_file_mutation(self) -> None:
        rows = [
            {
                "trial_id": "trial-1",
                "steps": json.dumps(
                    [
                        {
                            "src": "agent",
                            "msg": "Executed shell edit",
                            "tools": [
                                {"fn": "run_shell", "cmd": "sed -i '345s/ars/result <- ars/' /app/ars.R"},
                            ],
                            "obs": "",
                        }
                    ]
                ),
            }
        ]

        summary = report.summarize_dataset("sample", "terminalbench", rows)

        self.assertEqual(summary["shell_command_counts"]["sed"], 1)
        self.assertEqual(summary["shell_command_intent_counts"]["file-mutation"], 1)

    def test_shell_command_helpers_classify_awk_file_probe_as_file_inspection(self) -> None:
        command = r"awk '/^[[:space:]]*\{/ {count++} /^[[:space:]]*\}/ {count--} END {print count}' /app/ars.R"
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "awk")
        self.assertEqual(shape, "awk GLOB PATH")
        self.assertEqual(intent, "file-inspection")

    def test_shell_command_helpers_classify_pkill_then_r_version_as_dependency_probe(self) -> None:
        command = 'pkill apt-get && sleep 2 && /usr/bin/R --version 2>&1 || ls /usr/lib/R/bin/R 2>&1 || echo "R still not available"'
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "pkill")
        self.assertEqual(shape, "pkill ARG")
        self.assertEqual(intent, "dependency-probe")

    def test_shell_command_helpers_classify_pkill_then_dpkg_configure_as_dependency_recovery(self) -> None:
        command = "pkill -9 apt-get && sleep 2 && rm -f /var/lib/dpkg/lock* /var/cache/apt/archives/lock && dpkg --configure -a"
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "pkill")
        self.assertEqual(shape, "pkill -9 ARG")
        self.assertEqual(intent, "dependency-recovery")

    def test_shell_command_helpers_classify_cd_then_pip_install_as_dependency_install(self) -> None:
        command = "cd /workspace/project && pip install -e ."
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "pip")
        self.assertEqual(shape, "pip ARG -e PATH")
        self.assertEqual(intent, "dependency-install")

    def test_shell_command_helpers_classify_cd_then_rm_as_file_mutation(self) -> None:
        command = "cd /testbed && rm -f reproduce_issue.py verify_fix.py"
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "rm")
        self.assertEqual(shape, "rm -f ARG ARG")
        self.assertEqual(intent, "file-mutation")

    def test_shell_command_helpers_classify_cd_then_awk_as_file_inspection(self) -> None:
        command = (
            "cd /workspace/pandas-dev__pandas__1.0 && "
            "awk '/^class.*:/,/^[^[:space:]]/ { if (/^class/) print NR \": \" $0; if (/def value_counts/) print NR \": \" $0 }' "
            "pandas/core/groupby/generic.py"
        )
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "awk")
        self.assertEqual(shape, "awk GLOB PATH")
        self.assertEqual(intent, "file-inspection")

    def test_shell_command_helpers_classify_env_prefixed_pytest_as_python_test_command(self) -> None:
        command = "PYTHONPATH=src python -m pytest tests/test_app.py -q"
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "python")
        self.assertEqual(shape, "python -m ARG PATH -q")
        self.assertEqual(intent, "python-test-command")

    def test_shell_command_helpers_classify_python27_inline_probe(self) -> None:
        command = 'python2.7 -c "import sys; print(sys.version)"'
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "python2.7")
        self.assertEqual(shape, "python2.7 -c ARG")
        self.assertEqual(intent, "python-inline-probe")

    def test_shell_command_helpers_classify_export_then_pytest_as_python_test_command(self) -> None:
        command = "export PYTHONPATH=src && python -m pytest tests/test_app.py -q"
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "python")
        self.assertEqual(shape, "python -m ARG PATH -q")
        self.assertEqual(intent, "python-test-command")

    def test_shell_command_helpers_classify_source_then_pytest_as_test_command(self) -> None:
        command = "source /opt/venv/bin/activate && pytest -q"
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "pytest")
        self.assertEqual(shape, "pytest -q")
        self.assertEqual(intent, "test-command")

    def test_shell_command_helpers_classify_pwd_as_file_inspection(self) -> None:
        family, shape = report._shell_command_family_and_shape("pwd")
        intent = report._shell_command_intent("pwd")

        self.assertEqual(family, "pwd")
        self.assertEqual(shape, "pwd")
        self.assertEqual(intent, "file-inspection")

    def test_shell_command_helpers_classify_mkdir_as_file_mutation(self) -> None:
        command = "mkdir -p /testbed/test_app"
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "mkdir")
        self.assertEqual(shape, "mkdir -p PATH")
        self.assertEqual(intent, "file-mutation")

    def test_shell_command_helpers_classify_pip_list_grep_as_dependency_probe(self) -> None:
        command = 'cd /workspace/Pylons__pyramid__1.0 && pip list | grep -E "(webob|WebOb)"'
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "pip")
        self.assertTrue(shape.startswith("pip "))
        self.assertEqual(intent, "dependency-probe")

    def test_shell_command_helpers_classify_make_unit_as_test_command(self) -> None:
        command = "cd /workspace/tobymao__sqlglot__26.0 && make unit"
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "make")
        self.assertEqual(shape, "make ARG")
        self.assertEqual(intent, "test-command")

    def test_shell_command_helpers_classify_runtests_py_as_test_command(self) -> None:
        command = "cd /testbed && ./tests/runtests.py test_email_token_issue"
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "runtests.py")
        self.assertEqual(shape, "runtests.py ARG")
        self.assertEqual(intent, "test-command")

    def test_shell_command_helpers_classify_chmod_then_run_tests_sh_as_test_command(self) -> None:
        command = "cd /workspace/datalad__datalad__1.0 && chmod +x run_tests.sh && ./run_tests.sh --help"
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "run_tests.sh")
        self.assertEqual(shape, "run_tests.sh --help")
        self.assertEqual(intent, "test-command")

    def test_shell_command_helpers_classify_chmod_as_file_mutation(self) -> None:
        command = "chmod +x reproduce_issue.py"
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "chmod")
        self.assertEqual(shape, "chmod ARG ARG")
        self.assertEqual(intent, "file-mutation")

    def test_shell_command_helpers_classify_psql_help_grep_as_dependency_probe(self) -> None:
        command = "psql --help | grep -i ssl"
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "psql")
        self.assertEqual(shape, "psql --help ARG ARG -i ARG")
        self.assertEqual(intent, "dependency-probe")

    def test_shell_command_helpers_classify_man_psql_grep_as_dependency_probe(self) -> None:
        command = 'man psql 2>/dev/null | grep -A5 -B5 ssl || echo "No man page found"'
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "man")
        self.assertEqual(shape, "man ARG PATH ARG ARG -a5 -b5 ARG")
        self.assertEqual(intent, "dependency-probe")

    def test_shell_command_helpers_classify_file_glob_as_file_inspection(self) -> None:
        command = "cd /testbed && file /testbed/*.png"
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "file")
        self.assertEqual(shape, "file GLOB")
        self.assertEqual(intent, "file-inspection")

    def test_shell_command_helpers_classify_stat_glob_as_file_inspection(self) -> None:
        command = "cd /testbed && stat /testbed/*.png"
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "stat")
        self.assertEqual(shape, "stat GLOB")
        self.assertEqual(intent, "file-inspection")

    def test_shell_command_helpers_classify_curl_grep_doc_search_as_text_search(self) -> None:
        command = 'cd /testbed && curl -s https://www.python.org/dev/peps/pep-0257/ | grep -A 20 -B 5 "Handling Docstring Indentation"'
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "curl")
        self.assertEqual(shape, "curl -s PATH ARG ARG -a N -b")
        self.assertEqual(intent, "text-search")

    def test_shell_command_helpers_classify_curl_grep_api_search_as_text_search(self) -> None:
        command = 'cd /testbed && curl -s https://api.github.com/repos/django/django/pulls/13023 | grep -E "title|body" -A 5'
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "curl")
        self.assertEqual(shape, "curl -s PATH ARG ARG -e ARG -a")
        self.assertEqual(intent, "text-search")

    def test_shell_command_helpers_classify_curl_sed_preview_as_file_inspection(self) -> None:
        command = "curl -s https://raw.githubusercontent.com/django/django/38a21f2d9ed4f556af934498ec6a242f6a20418a/django/views/debug.py | sed -n '390,410p'"
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "curl")
        self.assertEqual(shape, "curl -s PATH ARG ARG -n ARG")
        self.assertEqual(intent, "file-inspection")

    def test_shell_command_helpers_classify_curl_head_preview_as_file_inspection(self) -> None:
        command = "cd /testbed && curl -s https://api.github.com/repos/django/django/pulls/13614/commits | head -20"
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "curl")
        self.assertEqual(shape, "curl -s PATH ARG ARG -20")
        self.assertEqual(intent, "file-inspection")

    def test_shell_command_helpers_classify_plain_curl_fetch_as_file_inspection(self) -> None:
        command = "cd /testbed && curl -s https://patch-diff.githubusercontent.com/raw/django/django/pull/13023.diff"
        family, shape = report._shell_command_family_and_shape(command)
        intent = report._shell_command_intent(command)

        self.assertEqual(family, "curl")
        self.assertEqual(shape, "curl -s PATH")
        self.assertEqual(intent, "file-inspection")

    def test_summarize_dataset_counts_shell_commands_for_smith_and_swe_agent(self) -> None:
        smith_rows = [
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "I will inspect first.\n<function=bash> <parameter=command>find /testbed -maxdepth 2</parameter> </function>",
                    }
                ]
            }
        ]
        swe_agent_rows = [
            {
                "trajectory": [
                    {
                        "role": "assistant",
                        "text": "Let's run the checker.\n```bash\npython -m flake8 fail.py\n```",
                    }
                ]
            }
        ]

        smith_summary = report.summarize_dataset("smith-sample", "smith", smith_rows)
        swe_agent_summary = report.summarize_dataset("agent-sample", "swe_agent", swe_agent_rows)

        self.assertEqual(smith_summary["shell_command_counts"]["find"], 1)
        self.assertEqual(smith_summary["shell_command_intent_counts"]["file-discovery"], 1)
        self.assertEqual(swe_agent_summary["shell_command_counts"]["python"], 1)
        self.assertEqual(swe_agent_summary["shell_command_intent_counts"]["python-other"], 1)

    def test_summarize_dataset_ignores_terminalbench_comment_only_shell_commands(self) -> None:
        rows = [
            {
                "trial_id": "trial-1",
                "steps": json.dumps(
                    [
                        {
                            "src": "agent",
                            "msg": "Executed Bash",
                            "tools": [
                                {"fn": "bash_command", "cmd": "#' Adaptive Rejection Sampler"},
                                {"fn": "bash_command", "cmd": "['bash', '-lc', 'pytest -q tests/test_app.py']"},
                            ],
                            "obs": "",
                        }
                    ]
                ),
            }
        ]

        summary = report.summarize_dataset("sample", "terminalbench", rows)

        self.assertNotIn("#'", summary["shell_command_counts"])
        self.assertNotIn("[bash", summary["shell_command_counts"])
        self.assertEqual(summary["shell_command_counts"]["pytest"], 1)
        self.assertEqual(summary["shell_command_intent_counts"]["test-command"], 1)

    def test_summarize_dataset_ignores_non_shell_pseudo_commands(self) -> None:
        rows = [
            {
                "trajectory": [
                    {"role": "assistant", "text": "```bash\nedit 258:259\n```"},
                    {"role": "assistant", "text": "```bash\nsubmit\n```"},
                    {"role": "assistant", "text": "```bash\nscroll_down 500\n```"},
                    {"role": "assistant", "text": "```bash\ncd repo\n```"},
                    {"role": "assistant", "text": "```bash\nC-c\n```"},
                    {"role": "assistant", "text": "```bash\nsearch_file \"needle\" app.py\n```"},
                    {"role": "assistant", "text": "```bash\nsearch_dir \"needle\"\n```"},
                    {"role": "assistant", "text": "```bash\nfind_file \"base32.py\"\n```"},
                    {"role": "assistant", "text": "```bash\npython -m flake8 fail.py\n```"},
                ]
            }
        ]

        summary = report.summarize_dataset("sample", "swe_agent", rows)

        self.assertEqual(summary["shell_command_counts"], {"python": 1})
        self.assertEqual(summary["shell_command_intent_counts"], {"python-other": 1})

    def test_fix_coverage_counts_shell_specific_errors(self) -> None:
        summary = {
            "dataset": "sample",
            "shell_error_counts": {
                "bash_unexpected_token": 3,
                "missing_file_or_dir": 2,
                "permission_denied": 1,
            },
        }

        fixes = report._fix_evidence_rows(summary)
        fix_map = {row["id"]: row for row in fixes}

        self.assertEqual(fix_map["validated-cli-preflight"]["evidence_count"], 3)
        self.assertEqual(fix_map["path-repair-guard"]["evidence_count"], 2)
        self.assertEqual(fix_map["fail-closed-permission"]["evidence_count"], 1)

    def test_fix_evidence_rows_ignores_non_numeric_message_theme_counts(self) -> None:
        summary = {
            "dataset": "sample",
            "message_theme_counts": {
                "explicit-tool-request": "n/a",
                "large-failure-blob": 2,
            },
        }

        fixes = report._fix_evidence_rows(summary)
        fix_map = {row["id"]: row for row in fixes}

        self.assertIn("failure-compression", fix_map)
        self.assertNotIn("mechanical-router", fix_map)

    def test_fix_evidence_rows_ignores_non_numeric_error_counts(self) -> None:
        summary = {
            "dataset": "sample",
            "error_counts": {
                "test_assertion": "n/a",
                "path_missing": 2,
            },
        }

        fixes = report._fix_evidence_rows(summary)
        fix_map = {row["id"]: row for row in fixes}

        self.assertEqual(fix_map["path-repair-guard"]["evidence_count"], 2)
        self.assertNotIn("diagnose-test-failure", fix_map)

    def test_portfolio_ignores_non_numeric_evidence_count_rows(self) -> None:
        with patch.object(
            report,
            "_fix_evidence_rows",
            return_value=[
                {"id": "bad", "evidence_count": "n/a", "datasets": ["sample"]},
                {"id": "good", "evidence_count": 2, "datasets": ["sample"]},
            ],
        ):
            rows = report._portfolio({"datasets": [{"dataset": "sample"}]})

        row_map = {row["id"]: row for row in rows}
        self.assertIn("good", row_map)
        self.assertEqual(row_map["good"]["evidence_count"], 2)
        self.assertNotIn("bad", row_map)

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
                    "shell_command_counts": {"grep": 2},
                    "shell_command_shape_counts": {"grep ARG PATH": 2},
                    "shell_command_intent_counts": {"text-search": 2},
                    "error_counts": {"test_assertion": 2},
                    "row_pattern_counts": {"mechanical-turn-row": 1},
                    "message_theme_examples": {"explicit-tool-request": [{"dataset": "sample", "row_id": "row-1", "message_index": 0, "role": "user", "excerpt": "Use read_file"}]},
                    "error_examples": {},
                    "shell_command_intent_examples": {
                        "text-search": [
                            {
                                "dataset": "sample",
                                "row_id": "row-2",
                                "message_index": 1,
                                "role": "assistant",
                                "command": "grep needle README.md",
                                "shape": "grep ARG PATH",
                            }
                        ]
                    },
                    "recommendations": [],
                }
            ],
        }

        markdown = report.format_markdown(payload)

        self.assertIn("Trajectory Evidence Report", markdown)
        self.assertIn("Portfolio Fix Coverage", markdown)
        self.assertIn("mechanical-router", markdown)
        self.assertIn("Shell Command Shapes", markdown)
        self.assertIn("Top intents", markdown)
        self.assertIn("text-search", markdown)
        self.assertIn("grep ARG PATH", markdown)
        self.assertIn("shell-intent:text-search", markdown)
        self.assertIn("grep needle README.md", markdown)

    def test_format_markdown_ignores_non_numeric_message_theme_counts(self) -> None:
        payload = {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "data_root": "scratch/external/datasets",
            "max_rows_per_dataset": 1,
            "portfolio_fix_coverage": [],
            "datasets": [
                {
                    "dataset": "sample",
                    "rows_profiled": 1,
                    "messages_profiled": 1,
                    "tool_call_events": 1,
                    "avg_tool_calls": 1.0,
                    "context_loop_rows_pct": 0.0,
                    "edit_without_prior_context_pct": 0.0,
                    "edit_without_later_test_pct": 0.0,
                    "mechanical_turn_candidates_pct": 0.0,
                    "message_theme_counts": {"explicit-tool-request": "n/a", "large-failure-blob": 2},
                    "shell_command_counts": {},
                    "shell_command_shape_counts": {},
                    "shell_command_intent_counts": {},
                    "error_counts": {},
                    "row_pattern_counts": {},
                    "message_theme_examples": {},
                    "error_examples": {},
                    "shell_command_intent_examples": {},
                    "recommendations": [],
                }
            ],
        }

        markdown = report.format_markdown(payload)

        self.assertIn("large-failure-blob", markdown)
        self.assertNotIn("`explicit-tool-request` | n/a", markdown)

    def test_format_markdown_orders_shell_intent_examples_by_count(self) -> None:
        payload = {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "data_root": "scratch/external/datasets",
            "max_rows_per_dataset": 1,
            "portfolio_fix_coverage": [],
            "datasets": [
                {
                    "dataset": "sample",
                    "rows_profiled": 1,
                    "messages_profiled": 1,
                    "tool_call_events": 1,
                    "avg_tool_calls": 1.0,
                    "context_loop_rows_pct": 0.0,
                    "edit_without_prior_context_pct": 0.0,
                    "edit_without_later_test_pct": 0.0,
                    "mechanical_turn_candidates_pct": 0.0,
                    "message_theme_counts": {},
                    "shell_command_counts": {"python": 10},
                    "shell_command_shape_counts": {"python -m ARG PATH -v": 10},
                    "shell_command_intent_counts": {
                        "python-test-command": 10,
                        "python-inline-probe": 5,
                        "text-search": 3,
                        "file-discovery": 1,
                    },
                    "error_counts": {},
                    "row_pattern_counts": {},
                    "message_theme_examples": {},
                    "error_examples": {},
                    "shell_command_intent_examples": {
                        "file-discovery": [{"dataset": "sample", "row_id": "row-1", "message_index": 1, "role": "assistant", "command": "find . -name *.py", "shape": "find PATH -name GLOB"}],
                        "text-search": [{"dataset": "sample", "row_id": "row-2", "message_index": 2, "role": "assistant", "command": "grep needle README.md", "shape": "grep ARG PATH"}],
                        "python-inline-probe": [{"dataset": "sample", "row_id": "row-3", "message_index": 3, "role": "assistant", "command": "python -c \"print(1)\"", "shape": "python -c ARG"}],
                        "python-test-command": [{"dataset": "sample", "row_id": "row-4", "message_index": 4, "role": "assistant", "command": "python -m pytest tests/test_app.py -v", "shape": "python -m ARG PATH -v"}],
                    },
                    "recommendations": [],
                }
            ],
        }

        markdown = report.format_markdown(payload)

        self.assertIn("shell-intent:python-test-command", markdown)
        self.assertIn("python -m pytest tests/test_app.py -v", markdown)
        self.assertIn("shell-intent:python-inline-probe", markdown)
        self.assertIn("shell-intent:text-search", markdown)
        self.assertNotIn("shell-intent:file-discovery", markdown)

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
