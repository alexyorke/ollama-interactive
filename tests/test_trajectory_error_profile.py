import json
from contextlib import contextmanager
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


@contextmanager
def _temp_root():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


class TrajectoryErrorProfileTests(unittest.TestCase):
    def test_main_accepts_argv_and_writes_output(self) -> None:
        with _temp_root() as root:
            output_path = root / "trajectory-error-profile.json"

            exit_code = error_profile.main(
                [
                    "--data-root",
                    str(root),
                    "--datasets",
                    "trace-commons-agent-traces",
                    "--max-rows",
                    "0",
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(len(payload["datasets"]), 1)
        self.assertEqual(payload["datasets"][0]["dataset"], "trace-commons-agent-traces")
        self.assertEqual(payload["datasets"][0]["rows_profiled"], 0)
        self.assertEqual(payload["datasets"][0]["result_events"], 0)
        self.assertEqual(payload["data_root"], root.as_posix())

    def test_does_not_classify_code_snippets_as_test_or_timeout_errors(self) -> None:
        rows = [
            {
                "trajectory": [
                    {
                        "role": "tool",
                        "name": "read_file",
                        "content": (
                            "58\tsetTimeout(async () => {\n"
                            "59\t    system.setScreen('desktop');\n"
                            "60\t});\n"
                            "61\tset timeout=3\n"
                            "133\t    kprint(\"WARNING: failed to mount root filesystem\\n\");\n"
                            "109\t    /* Count this as a failed attempt; will retry next tick */\n"
                            "1\tVerify against current code before asserting as fact.\n"
                        ),
                    }
                ]
            }
        ]

        summary = error_profile.summarize_dataset("sample", "openhands", rows)

        self.assertEqual(summary["result_events"], 1)
        self.assertEqual(summary["error_counts"], {})
        self.assertEqual(summary["top_tool_errors"], [])
        self.assertEqual(summary["top_shell_errors"], [])

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

    def test_cc_bench_result_events_read_embedded_tool_result(self) -> None:
        rows = [
            {
                "trajectory": json.dumps(
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
            }
        ]

        summary = error_profile.summarize_dataset("sample", "cc_bench", rows)

        self.assertEqual(summary["error_counts"], {"test_assertion": 1})

    def test_does_not_classify_successful_cli_help_as_invalid_args(self) -> None:
        rows = [
            {
                "trajectory": [
                    {
                        "role": "tool",
                        "name": "run_shell",
                        "content": (
                            "Usage: hf [OPTIONS] COMMAND [ARGS]...\n\n"
                            "  Hugging Face Hub CLI\n\n"
                            "Options:\n"
                            "  -h, --help  Show this message and exit.\n\n"
                            "Main commands:\n"
                            "  auth  Manage authentication.\n"
                        ),
                    },
                    {
                        "role": "tool",
                        "name": "run_shell",
                        "content": (
                            "Usage: hf jobs uv run [OPTIONS] SCRIPT [SCRIPT_ARGS]...\n\n"
                            "  Run a UV script (local file or URL) on HF infrastructure\n\n"
                            "Arguments:\n"
                            "  SCRIPT            UV script to run (local file or URL)  [required]\n"
                            "  [SCRIPT_ARGS]...  Arguments for the script\n"
                        ),
                    },
                ]
            }
        ]

        summary = error_profile.summarize_dataset("sample", "openhands", rows)

        self.assertEqual(summary["result_events"], 2)
        self.assertEqual(summary["error_counts"], {})
        self.assertEqual(summary["top_tool_errors"], [])

    def test_does_not_classify_str_replace_editor_cat_output_as_invalid_args(self) -> None:
        rows = [
            {
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
        ]

        summary = error_profile.summarize_dataset("sample", "openhands", rows)

        self.assertEqual(summary["result_events"], 1)
        self.assertEqual(summary["error_counts"], {})
        self.assertEqual(summary["top_tool_errors"], [])

    def test_does_not_classify_successful_read_file_source_snippet_as_import_error(self) -> None:
        rows = [
            {
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
        ]

        summary = error_profile.summarize_dataset("sample", "smith", rows)

        self.assertEqual(summary["result_events"], 1)
        self.assertEqual(summary["error_counts"], {})
        self.assertEqual(summary["top_tool_errors"], [])

    def test_does_not_classify_successful_read_file_source_snippet_as_test_assertion(self) -> None:
        rows = [
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "Inspect the edited file.\n<function=read_file>\n<parameter=path>test_app.py</parameter>\n</function>",
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
        ]

        summary = error_profile.summarize_dataset("sample", "smith", rows)

        self.assertEqual(summary["result_events"], 1)
        self.assertEqual(summary["error_counts"], {})
        self.assertEqual(summary["top_tool_errors"], [])

    def test_does_not_classify_str_replace_editor_cat_preview_as_path_missing(self) -> None:
        rows = [
            {
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
        ]

        summary = error_profile.summarize_dataset("sample", "openhands", rows)

        self.assertEqual(summary["result_events"], 1)
        self.assertEqual(summary["error_counts"], {})
        self.assertEqual(summary["top_tool_errors"], [])

    def test_does_not_classify_expected_syntax_error_example_in_passing_smoke_test(self) -> None:
        rows = [
            {
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
        ]

        summary = error_profile.summarize_dataset("sample", "trace_commons", rows)

        self.assertEqual(summary["result_events"], 1)
        self.assertEqual(summary["error_counts"], {})
        self.assertEqual(summary["top_tool_errors"], [])

    def test_does_not_classify_expected_invalid_args_example_in_cli_test_transcript(self) -> None:
        rows = [
            {
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
        ]

        summary = error_profile.summarize_dataset("sample", "swe_agent", rows)

        self.assertEqual(summary["result_events"], 1)
        self.assertEqual(summary["error_counts"], {})
        self.assertEqual(summary["top_tool_errors"], [])

    def test_does_not_classify_expected_invalid_args_in_parser_recovery_suite(self) -> None:
        rows = [
            {
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
        ]

        summary = error_profile.summarize_dataset("sample", "openhands", rows)

        self.assertEqual(summary["result_events"], 1)
        self.assertEqual(summary["error_counts"], {})
        self.assertEqual(summary["top_tool_errors"], [])

    def test_does_not_classify_expected_import_error_example_in_success_wrapper(self) -> None:
        rows = [
            {
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
        ]

        summary = error_profile.summarize_dataset("sample", "smith", rows)

        self.assertEqual(summary["result_events"], 1)
        self.assertEqual(summary["error_counts"], {})
        self.assertEqual(summary["top_tool_errors"], [])

    def test_does_not_classify_expected_assertion_error_example_in_success_wrapper(self) -> None:
        rows = [
            {
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
        ]

        summary = error_profile.summarize_dataset("sample", "smith", rows)

        self.assertEqual(summary["result_events"], 1)
        self.assertEqual(summary["error_counts"], {})
        self.assertEqual(summary["top_tool_errors"], [])

    def test_does_not_classify_shell_code_snippet_as_invalid_args(self) -> None:
        rows = [
            {
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
        ]

        summary = error_profile.summarize_dataset("sample", "openhands", rows)

        self.assertEqual(summary["result_events"], 1)
        self.assertEqual(summary["error_counts"], {})
        self.assertEqual(summary["top_tool_errors"], [])

    def test_does_not_classify_shell_code_snippet_as_import_error(self) -> None:
        rows = [
            {
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
                            '\treturn\n'
                            "}\n"
                            "[The command completed with exit code 0.]\n"
                            "[Current working directory: /workspace/crossplane-contrib__provider-gitlab__1.0]\n"
                            "[Command finished with exit code 0]\n"
                        ),
                    }
                ]
            }
        ]

        summary = error_profile.summarize_dataset("sample", "openhands", rows)

        self.assertEqual(summary["result_events"], 1)
        self.assertEqual(summary["error_counts"], {})
        self.assertEqual(summary["top_tool_errors"], [])

    def test_does_not_classify_timeout_mentions_without_failure_signal(self) -> None:
        rows = [
            {
                "trajectory": [
                    {
                        "role": "tool",
                        "name": "run_shell",
                        "content": (
                            "Usage: hf jobs uv run [OPTIONS] SCRIPT [SCRIPT_ARGS]...\n\n"
                            "  --timeout TEXT  Max duration: int/float with s (seconds), m (minutes), h (hours) or d (days).\n"
                            "  -d, --detach    Run the Job in the background.\n"
                        ),
                    },
                    {
                        "role": "tool",
                        "name": "read_file",
                        "content": (
                            "Recommended command:\n"
                            "hf jobs uv run --flavor a10g-small --timeout 2h --secrets HF_TOKEN train.py\n"
                            "This is just documentation, not a failed command.\n"
                        ),
                    },
                ]
            }
        ]

        summary = error_profile.summarize_dataset("sample", "openhands", rows)

        self.assertEqual(summary["result_events"], 2)
        self.assertEqual(summary["error_counts"], {})
        self.assertEqual(summary["top_tool_errors"], [])

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

    def test_powershell_result_is_treated_as_shell_tool_error(self) -> None:
        rows = [
            {
                "messages": [
                    '{"role":"assistant","content":"","tool_calls":[{"id":"toolu_123","function":{"name":"PowerShell","arguments":{"command":"pytest -q tests/test_app.py"}}}]}',
                    '{"role":"tool","tool_call_id":"toolu_123","name":"PowerShell","content":"bash: foozle: command not found"}',
                ]
            }
        ]

        summary = error_profile.summarize_dataset("sample", "trace_commons", rows)

        self.assertEqual(summary["result_events"], 1)
        self.assertEqual(summary["top_tool_errors"][0]["tool_error"], "powershell:command_not_found")
        self.assertEqual(summary["top_shell_errors"][0]["shell_error"], "command_not_found")

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

    def test_smith_result_events_fall_back_to_messages_json_when_messages_is_empty_list_string(self) -> None:
        rows = [
            {
                "messages": "[]",
                "messages_json": (
                    '[{"role":"assistant","content":"THOUGHT: repro\\n```bash\\npytest -q tests/test_app.py\\n```"},'
                    '{"role":"user","content":"Traceback\\nAssertionError: expected 1"}]'
                ),
            }
        ]

        summary = error_profile.summarize_dataset("sample", "smith", rows)

        self.assertEqual(summary["result_events"], 1)
        self.assertEqual(summary["error_counts"]["test_assertion"], 1)

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
        with _temp_root() as data_root:
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
        with _temp_root() as data_root:
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
        with _temp_root() as data_root:
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

    def test_build_profile_thoughtworks_falls_back_to_messages_json_when_messages_is_empty_list_string(self) -> None:
        if pa is None or pq is None:
            self.skipTest("pyarrow is not installed")
        with _temp_root() as data_root:
            dataset = "thoughtworks-agentic-coding-trajectories"
            path = data_root / dataset / "sessions.parquet"
            path.parent.mkdir(parents=True)
            table = pa.table(
                {
                    "agent_framework": ["mini-swe-agent"],
                    "session_id": ["tw-1"],
                    "messages": ["[]"],
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
        with _temp_root() as data_root:
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

    def test_build_profile_reads_trace_commons_messages_json_column(self) -> None:
        if pa is None or pq is None:
            self.skipTest("pyarrow is not installed")
        with _temp_root() as data_root:
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

            payload = error_profile.build_profile(data_root, [dataset], max_rows=1)

        summary = payload["datasets"][0]
        self.assertEqual(summary["rows_profiled"], 1)
        self.assertEqual(summary["result_events"], 1)
        self.assertEqual(summary["error_counts"]["test_assertion"], 1)

    def test_build_profile_reads_trace_commons_trace_column(self) -> None:
        if pa is None or pq is None:
            self.skipTest("pyarrow is not installed")
        with _temp_root() as data_root:
            dataset = "trace-commons-agent-traces"
            path = data_root / dataset / "data" / "train-00000-of-00001.parquet"
            path.parent.mkdir(parents=True)
            table = pa.table(
                {
                    "session_id": ["tc-structured"],
                    "messages": [json.dumps([])],
                    "trace": [
                        json.dumps(
                            [
                                {
                                    "role": "assistant",
                                    "message": {
                                        "content": [
                                            {
                                                "type": "tool_use",
                                                "id": "toolu_1",
                                                "name": "Bash",
                                                "input": {"command": "pytest -q tests/test_app.py"},
                                            }
                                        ]
                                    },
                                },
                                {
                                    "role": "tool",
                                    "message": {
                                        "tool_call_id": "toolu_1",
                                        "name": "Bash",
                                        "content": "FAILED tests/test_app.py::test_x\nE   assert 1 == 2",
                                    },
                                },
                            ]
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

    def test_build_profile_reads_agent_race_jsonl_sessions(self) -> None:
        with _temp_root() as data_root:
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

            payload = error_profile.build_profile(data_root, [dataset], max_rows=1)

        summary = payload["datasets"][0]
        self.assertEqual(summary["rows_profiled"], 1)
        self.assertEqual(summary["result_events"], 1)
        self.assertEqual(summary["error_counts"]["test_assertion"], 1)
        self.assertEqual(summary["top_tool_errors"][0]["tool_error"], "bash:test_assertion")

    def test_build_profile_treats_max_rows_zero_as_unbounded(self) -> None:
        with _temp_root() as data_root:
            dataset = "agent-race-traces"
            path = data_root / dataset / "pi-kimi.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text(
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

            payload = error_profile.build_profile(data_root, [dataset], max_rows=0)

        summary = payload["datasets"][0]
        self.assertEqual(summary["rows_profiled"], 2)
        self.assertEqual(summary["result_events"], 2)
        self.assertEqual(summary["error_counts"]["test_assertion"], 2)

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

    def test_terminalbench_killshell_success_is_not_classified_as_timeout(self) -> None:
        rows = [
            {
                "steps": (
                    '[{"src":"agent","msg":"Executed KillShell toolu_123","tools":[{"fn":"KillShell","cmd":""}],'
                    '"obs":"{\\"message\\":\\"Successfully killed shell: d657ce (Rscript -e \\\\\\"source(\'ars.R\'); test()\\\\\\")\\",\\"shell_id\\":\\"d657ce\\"}"}]'
                )
            }
        ]

        summary = error_profile.summarize_dataset("sample", "terminalbench", rows)

        self.assertEqual(summary["result_events"], 1)
        self.assertEqual(summary["error_counts"], {})
        self.assertEqual(summary["top_tool_errors"], [])

    def test_build_profile_skips_terminalbench_null_rows_before_max_rows(self) -> None:
        if pa is None or pq is None:
            self.skipTest("pyarrow is not installed")
        with _temp_root() as data_root:
            dataset = "terminalbench-trajectories"
            path = data_root / dataset / "data" / "train-00000-of-00001.parquet"
            path.parent.mkdir(parents=True)
            table = pa.table(
                {
                    "steps": [
                        "null",
                        (
                            '[{"src":"agent","msg":"Executed Bash","tools":[{"fn":"bash_command","cmd":"pytest -q"}],'
                            '"obs":"Exit code 1\\nFAILED tests/test_app.py::test_x\\nE   assert 1 == 2"}]'
                        ),
                    ]
                }
            )
            pq.write_table(table, path)

            payload = error_profile.build_profile(data_root, [dataset], max_rows=1)

        summary = payload["datasets"][0]
        self.assertEqual(summary["rows_profiled"], 1)
        self.assertEqual(summary["result_events"], 1)
        self.assertEqual(summary["error_counts"]["test_assertion"], 1)


if __name__ == "__main__":
    unittest.main()
