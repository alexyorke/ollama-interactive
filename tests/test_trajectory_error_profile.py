import unittest
import tempfile
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    pa = None
    pq = None

from scripts import trajectory_error_profile as error_profile


class TrajectoryErrorProfileTests(unittest.TestCase):
    def test_classifies_tool_result_errors_without_task_description_matches(self) -> None:
        rows = [
            {
                "trajectory": [
                    {"role": "user", "content": "Task mentions SyntaxError as text only."},
                    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "run_test", "arguments": "{}"}}]},
                    {"role": "tool", "name": "run_test", "content": "FAILED tests/test_app.py::test_x\nE   assert 1 == 2"},
                    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "run_shell", "arguments": "{}"}}]},
                    {"role": "tool", "name": "run_shell", "content": "usage: pytest [options]\nerror: unrecognized arguments: --wat"},
                ]
            }
        ]

        summary = error_profile.summarize_dataset("sample", "openhands", rows)

        self.assertEqual(summary["error_counts"]["test_assertion"], 1)
        self.assertEqual(summary["error_counts"]["invalid_args"], 1)
        self.assertEqual(summary["top_shell_errors"][0]["shell_error"], "unrecognized_argument")
        self.assertNotIn("syntax_error", summary["error_counts"])
        self.assertEqual(summary["examples"]["test_assertion"][0]["tool"], "run_test")

    def test_classifies_shell_syntax_and_command_errors(self) -> None:
        rows = [
            {
                "trajectory": [
                    {"role": "tool", "name": "run_shell", "content": "bash: -c: line 2: syntax error: unexpected end of file"},
                    {"role": "tool", "name": "execute_bash", "content": "bash: foozle: command not found"},
                    {"role": "tool", "name": "run_shell", "content": "syntax error near unexpected token `fi'"},
                ]
            }
        ]

        summary = error_profile.summarize_dataset("sample", "openhands", rows)
        shell_counts = {item["shell_error"]: item["count"] for item in summary["top_shell_errors"]}

        self.assertEqual(shell_counts["bash_unexpected_eof"], 1)
        self.assertEqual(shell_counts["command_not_found"], 1)
        self.assertEqual(shell_counts["bash_unexpected_token"], 1)

    def test_openhands_missing_tool_name_inherits_previous_tool_call_name(self) -> None:
        rows = [
            {
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
                    },
                    {
                        "role": "tool",
                        "content": "FAILED tests/test_app.py::test_x\nE   assert 1 == 2",
                    },
                ]
            }
        ]

        summary = error_profile.summarize_dataset("sample", "openhands", rows)

        self.assertEqual(summary["result_events"], 1)
        self.assertEqual(summary["error_counts"]["test_assertion"], 1)
        self.assertEqual(summary["top_tool_errors"][0]["tool_error"], "execute_bash:test_assertion")

    def test_openhands_unknown_tool_name_uses_tool_call_id_mapping(self) -> None:
        rows = [
            {
                "messages": [
                    '{"role":"assistant","content":"","tool_calls":[{"id":"toolu_123","function":{"name":"Bash","arguments":{"command":"pytest -q tests/test_app.py"}}}]}',
                    '{"role":"tool","tool_call_id":"toolu_123","name":"unknown_tool","content":"FAILED tests/test_app.py::test_x\\nE   assert 1 == 2"}',
                ]
            }
        ]

        summary = error_profile.summarize_dataset("sample", "trace_commons", rows)

        self.assertEqual(summary["result_events"], 1)
        self.assertEqual(summary["error_counts"]["test_assertion"], 1)
        self.assertEqual(summary["top_tool_errors"][0]["tool_error"], "bash:test_assertion")

    def test_openhands_result_events_accept_messages_json_and_tool_calls_json(self) -> None:
        rows = [
            {
                "messages_json": (
                    '[{"role":"assistant","content":"",'
                    '"tool_calls_json":"[{\\"function\\":{\\"name\\":\\"execute_bash\\",\\"arguments\\":\\"{\\\\\\"command\\\\\\":\\\\\\"pytest -q\\\\\\"}\\"}}]"},'
                    '{"role":"tool","name":"execute_bash","content":"FAILED tests/test_app.py::test_x\\nE   assert 1 == 2"}]'
                )
            }
        ]

        summary = error_profile.summarize_dataset("sample", "openhands", rows)

        self.assertEqual(summary["result_events"], 1)
        self.assertEqual(summary["error_counts"]["test_assertion"], 1)
        self.assertEqual(summary["top_tool_errors"][0]["tool_error"], "execute_bash:test_assertion")

    def test_repeated_error_loop_counts_same_tool_error(self) -> None:
        rows = [
            {
                "trajectory": [
                    {"role": "tool", "name": "read_file", "content": "FileNotFoundError: missing.py"},
                    {"role": "tool", "name": "read_file", "content": "FileNotFoundError: missing.py"},
                ]
            }
        ]

        summary = error_profile.summarize_dataset("sample", "openhands", rows)

        self.assertEqual(summary["repeated_error_loops"][0]["tool_error"], "read_file:path_missing")

    def test_smith_observation_after_function_call_counts_as_error_result(self) -> None:
        rows = [
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": (
                            "Let's run the reproduction script.\n"
                            "<function=bash>\n"
                            "<parameter=command>cd /testbed && python reproduce_issue.py</parameter>\n"
                            "</function>"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "OBSERVATION:\n"
                            "Traceback (most recent call last):\n"
                            "ImportError: cannot import name 'ConnectionState' from 'h11'"
                        ),
                    },
                ]
            }
        ]

        summary = error_profile.summarize_dataset("sample", "smith", rows)

        self.assertEqual(summary["result_events"], 1)
        self.assertEqual(summary["error_counts"]["import_error"], 1)
        self.assertEqual(summary["examples"]["import_error"][0]["tool"], "run_shell")

    def test_build_profile_reads_coderforge_messages_column(self) -> None:
        if pa is None or pq is None:
            self.skipTest("pyarrow is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            dataset = "coderforge-preview-swe-bench-verified-trajectories"
            path = data_root / dataset / "trajectory" / "train-00000-of-00001.parquet"
            path.parent.mkdir(parents=True)
            table = pa.table(
                {
                    "messages": [
                        (
                            '[{"role":"assistant","tool_calls":[{"function":{"name":"execute_bash","arguments":"{\\"command\\":\\"pytest -q\\"}"}}]},'
                            '{"role":"tool","name":"execute_bash","content":"FAILED tests/test_app.py::test_x\\nE   assert 1 == 2"}]'
                        )
                    ]
                }
            )
            pq.write_table(table, path)

            payload = error_profile.build_profile(data_root, [dataset], max_rows=1)

        summary = payload["datasets"][0]
        self.assertEqual(summary["rows_profiled"], 1)
        self.assertEqual(summary["result_events"], 1)
        self.assertEqual(summary["error_counts"]["test_assertion"], 1)

    def test_build_profile_reads_openhands_messages_json_column(self) -> None:
        if pa is None or pq is None:
            self.skipTest("pyarrow is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            dataset = "nebius-swe-rebench-openhands-trajectories"
            path = data_root / dataset / "trajectories.parquet"
            path.parent.mkdir(parents=True)
            table = pa.table(
                {
                    "messages_json": [
                        (
                            '[{"role":"assistant","content":"",'
                            '"tool_calls_json":"[{\\"function\\":{\\"name\\":\\"execute_bash\\",\\"arguments\\":\\"{\\\\\\"command\\\\\\":\\\\\\"pytest -q\\\\\\"}\\"}}]"},'
                            '{"role":"tool","name":"execute_bash","content":"FAILED tests/test_app.py::test_x\\nE   assert 1 == 2"}]'
                        )
                    ]
                }
            )
            pq.write_table(table, path)

            payload = error_profile.build_profile(data_root, [dataset], max_rows=1)

        summary = payload["datasets"][0]
        self.assertEqual(summary["rows_profiled"], 1)
        self.assertEqual(summary["result_events"], 1)
        self.assertEqual(summary["error_counts"]["test_assertion"], 1)

    def test_build_profile_reads_thoughtworks_messages_json_with_framework_dispatch(self) -> None:
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

            payload = error_profile.build_profile(data_root, [dataset], max_rows=1)

        summary = payload["datasets"][0]
        self.assertEqual(summary["rows_profiled"], 1)
        self.assertEqual(summary["result_events"], 1)
        self.assertEqual(summary["error_counts"]["test_assertion"], 1)

    def test_build_profile_reads_trace_commons_messages_column(self) -> None:
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
                            '{"role":"tool","name":"Bash","content":"FAILED tests/test_app.py::test_x\\nE   assert 1 == 2"}',
                        ]
                    ],
                }
            )
            pq.write_table(table, path)

            payload = error_profile.build_profile(data_root, [dataset], max_rows=1)

        summary = payload["datasets"][0]
        self.assertEqual(summary["rows_profiled"], 1)
        self.assertEqual(summary["result_events"], 1)
        self.assertEqual(summary["error_counts"]["test_assertion"], 1)

    def test_terminalbench_observation_counts_as_tool_result(self) -> None:
        rows = [
            {
                "steps": (
                    '[{"src":"agent","msg":"Executed Bash","tools":[{"fn":"bash_command","cmd":"pytest -q"}],'
                    '"obs":"Exit code 1\\nFAILED tests/test_app.py::test_x\\nE   assert 1 == 2"}]'
                )
            }
        ]

        summary = error_profile.summarize_dataset("sample", "terminalbench", rows)

        self.assertEqual(summary["result_events"], 1)
        self.assertEqual(summary["error_counts"]["test_assertion"], 1)
        self.assertEqual(summary["top_tool_errors"][0]["tool_error"], "run_shell:test_assertion")


if __name__ == "__main__":
    unittest.main()
