from __future__ import annotations

from contextlib import contextmanager
import json
import tempfile
import unittest
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    pa = None
    pq = None

from scripts import trajectory_profile as profile


@contextmanager
def _temp_root():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


class TrajectoryProfileTests(unittest.TestCase):
    def test_main_accepts_empty_data_root_without_formatter_crash(self) -> None:
        with _temp_root() as root:
            output_path = root / "trajectory-profile.json"

            exit_code = profile.main(
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
            self.assertEqual(payload["data_root"], str(root.resolve(strict=False)))
            self.assertEqual(len(payload["datasets"]), 1)
            self.assertEqual(payload["datasets"][0]["dataset"], "trace-commons-agent-traces")
            self.assertEqual(payload["datasets"][0]["status"], "missing")
            self.assertEqual(payload["datasets"][0]["rows_profiled"], 0)

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

    def test_extract_openhands_events_accepts_json_string_messages(self) -> None:
        row = {
            "messages": (
                '[{"role":"assistant","tool_calls":[{"function":{"name":"str_replace_editor","arguments":"{}"}}]},'
                '{"role":"assistant","content":"pytest -q tests/test_app.py"}]'
            )
        }

        events = profile._extract_events("openhands", row)

        self.assertEqual([event.name for event in events], ["str_replace_editor", "run_test"])
        self.assertEqual([event.category for event in events], ["edit", "test"])

    def test_extract_openhands_events_accepts_open_swe_trajectory_rows(self) -> None:
        row = {
            "trajectory": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "Let me inspect the repo first.",
                    "tool_calls": [
                        {
                            "id": "chatcmpl-tool-1",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": "{\"command\": \"pytest -q tests/test_app.py\"}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "chatcmpl-tool-1",
                    "content": "OBSERVATION:\nFAILED tests/test_app.py::test_x\nE   assert 1 == 2",
                },
            ]
        }

        events = profile._extract_events("openhands", row)

        self.assertEqual([event.name for event in events], ["bash", "bash"])
        self.assertEqual(events[0].kind, "tool_call")
        self.assertEqual(events[0].category, "test")
        self.assertEqual(events[1].kind, "tool_result")

    def test_extract_openhands_events_infers_tool_from_reasoning_content(self) -> None:
        row = {
            "trajectory": [
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "First I should run pytest -q tests/test_app.py to reproduce the failure.",
                }
            ]
        }

        events = profile._extract_events("openhands", row)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].name, "run_test")
        self.assertEqual(events[0].category, "test")

    def test_extract_openhands_events_accepts_messages_json_and_tool_calls_json(self) -> None:
        row = {
            "messages_json": (
                '[{"role":"assistant","content":"",'
                '"tool_calls_json":"[{\\"function\\":{\\"name\\":\\"execute_bash\\",\\"arguments\\":\\"{\\\\\\"command\\\\\\":\\\\\\"pytest -q\\\\\\"}\\"}}]"},'
                '{"role":"tool","name":"execute_bash","content":"FAILED tests/test_app.py::test_x\\nE   assert 1 == 2"}]'
            )
        }

        events = profile._extract_events("openhands", row)

        self.assertEqual([event.name for event in events], ["execute_bash", "execute_bash"])

    def test_extract_openhands_events_falls_back_to_messages_json_when_messages_is_empty_list_string(self) -> None:
        row = {
            "messages": "[]",
            "messages_json": (
                '[{"role":"assistant","content":"",'
                '"tool_calls_json":"[{\\"function\\":{\\"name\\":\\"execute_bash\\",\\"arguments\\":\\"{\\\\\\"command\\\\\\":\\\\\\"pytest -q\\\\\\"}\\"}}]"},'
                '{"role":"tool","name":"execute_bash","content":"AssertionError: expected 1"}]'
            ),
        }

        events = profile._extract_events("openhands", row)

        self.assertEqual([event.name for event in events], ["execute_bash", "execute_bash"])

    def test_extract_smith_events_falls_back_to_messages_json_when_messages_is_empty_list_string(self) -> None:
        row = {
            "messages": "[]",
            "messages_json": (
                '[{"role":"assistant","content":"THOUGHT: repro\\n```bash\\npytest -q tests/test_app.py\\n```"},'
                '{"role":"user","content":"Traceback\\nAssertionError: expected 1"}]'
            ),
        }

        events = profile._extract_events("smith", row)

        self.assertEqual([event.name for event in events], ["run_test"])
        self.assertEqual(events[0].category, "test")
        self.assertEqual(events[0].kind, "tool_call")

    def test_extract_openhands_events_accepts_message_list_of_json_strings(self) -> None:
        row = {
            "messages": [
                '{"role":"assistant","content":"","tool_calls":[{"function":{"name":"Bash","arguments":{"command":"pytest -q tests/test_app.py"}}}]}',
                '{"role":"tool","name":"Bash","content":"FAILED tests/test_app.py::test_x\\nE   assert 1 == 2"}',
            ]
        }

        events = profile._extract_events("trace_commons", row)

        self.assertEqual([event.name for event in events], ["bash", "bash"])
        self.assertEqual(events[0].kind, "tool_call")
        self.assertEqual(events[1].kind, "tool_result")

    def test_extract_openhands_events_reclassifies_embedded_bash_tool_use_from_input_command(self) -> None:
        row = {
            "trajectory": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "rg -n TODO src/"},
                        }
                    ],
                }
            ]
        }

        events = profile._extract_events("openhands", row)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].name, "bash")
        self.assertIn(events[0].category, {"read", "search"})

    def test_extract_openhands_events_reclassifies_compound_bash_context_command(self) -> None:
        row = {
            "trajectory": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "cd /repo && echo ===src=== && ls src/ && echo ===readme=== && cat README.md"},
                        }
                    ],
                }
            ]
        }

        events = profile._extract_events("openhands", row)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].name, "bash")
        self.assertIn(events[0].category, {"read", "search"})

    def test_extract_cc_bench_events_reads_embedded_tool_use_and_result(self) -> None:
        row = {
            "trajectory": json.dumps(
                [
                    {"type": "user", "message": {"role": "user", "content": "Build a page."}},
                    {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "I will write the file now."},
                                {
                                    "type": "tool_use",
                                    "id": "call_1",
                                    "name": "Write",
                                    "input": {"file_path": "/app/index.html", "content": "<h1>hi</h1>"},
                                },
                            ],
                        },
                    },
                    {
                        "type": "user",
                        "message": {
                            "role": "user",
                            "content": [
                                {"type": "tool_result", "tool_use_id": "call_1", "content": "File created successfully"}
                            ],
                        },
                    },
                ]
            )
        }

        events = profile._extract_events("cc_bench", row)

        self.assertEqual([event.name for event in events], ["write", "write"])
        self.assertEqual(events[0].kind, "tool_call")
        self.assertEqual(events[0].category, "edit")
        self.assertEqual(events[1].kind, "tool_result")

    def test_extract_openhands_events_recovers_unknown_tool_name_from_tool_call_id(self) -> None:
        row = {
            "messages": [
                '{"role":"assistant","content":"","tool_calls":[{"id":"toolu_123","function":{"name":"Write","arguments":{"file_path":"PROJECT.md","content":"x"}}}]}',
                '{"role":"tool","tool_call_id":"toolu_123","name":"unknown_tool","content":"File created successfully"}',
            ]
        }

        events = profile._extract_events("trace_commons", row)

        self.assertEqual([event.name for event in events], ["write", "write"])
        self.assertEqual(events[0].kind, "tool_call")
        self.assertEqual(events[1].kind, "tool_result")

    def test_extract_trace_commons_events_accepts_structured_trace_fallback(self) -> None:
        row = {
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

        events = profile._extract_events("trace_commons", row)

        self.assertEqual([event.name for event in events], ["websearch"])
        self.assertEqual(events[0].kind, "tool_call")
        self.assertEqual(events[0].category, "other")

    def test_extract_openhands_events_does_not_infer_tool_from_issue_description_user_prompt(self) -> None:
        row = {
            "trajectory": [
                {
                    "role": "user",
                    "content": (
                        "<uploaded_files>\n/workspace/getmoto__moto__1.0\n</uploaded_files>\n"
                        "I've uploaded a python code repository in the directory getmoto__moto__1.0.\n"
                        "Consider the following issue description:\n\n"
                        "<issue_description>\n"
                        "resourcegroupstaggingapi get_resources doesn't properly support filters in tests.\n"
                        "The current error message is misleading.\n"
                        "</issue_description>\n"
                    ),
                }
            ]
        }

        events = profile._extract_events("openhands", row)

        self.assertEqual(events, [])

    def test_extract_openhands_events_keeps_explicit_user_tool_request(self) -> None:
        row = {
            "trajectory": [
                {
                    "role": "user",
                    "content": "Use read_file on src/app.py, then run pytest -q tests/test_app.py.",
                }
            ]
        }

        events = profile._extract_events("openhands", row)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].name, "read_file")
        self.assertEqual(events[0].category, "read")

    def test_infer_tool_name_from_plain_swe_agent_edit_and_create_commands(self) -> None:
        self.assertEqual(profile._infer_tool_name_from_text("``` edit 138:138 print('x') ```"), "edit")
        self.assertEqual(profile._infer_tool_name_from_text("``` create ./test.py\nprint(1)\n```"), "write_file")
        self.assertEqual(profile._infer_tool_name_from_text("``` python reproduce.py ```"), "run_shell")

    def test_shell_observation_traceback_with_tree_variable_is_not_classified_as_search(self) -> None:
        content = (
            "OBSERVATION: + /opt/miniconda3/envs/testbed/bin/ninja\n"
            "Traceback (most recent call last):\n"
            '  File "/testbed/reproduce.py", line 1, in <module>\n'
            "    import pandas as pd\n"
            '  File "/opt/miniconda3/envs/testbed/lib/python3.10/site-packages/_pandas_editable_loader.py", line 268, in find_spec\n'
            "    tree = self.rebuild()\n"
        )

        self.assertIsNone(profile._command_category_from_content(content))
        self.assertEqual(profile._tool_category("run_shell", content), "shell")

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

    def test_shell_tool_call_with_make_test_is_reclassified_as_test(self) -> None:
        row = {
            "messages": [
                '{"role":"assistant","content":"","tool_calls":[{"function":{"name":"Bash","arguments":{"command":"make test 2>&1"}}}]}'
            ]
        }

        events = profile._extract_events("trace_commons", row)

        self.assertEqual(events[0].name, "bash")
        self.assertEqual(events[0].category, "test")

    def test_powershell_tool_call_with_pytest_is_reclassified_as_test(self) -> None:
        row = {
            "messages": [
                '{"role":"assistant","content":"","tool_calls":[{"function":{"name":"PowerShell","arguments":{"command":"pytest -q tests/test_app.py"}}}]}'
            ]
        }

        events = profile._extract_events("trace_commons", row)

        self.assertEqual(events[0].name, "powershell")
        self.assertEqual(events[0].category, "test")

    def test_shell_tool_call_with_git_remote_is_reclassified_as_git(self) -> None:
        row = {
            "messages": [
                '{"role":"assistant","content":"","tool_calls":[{"function":{"name":"Bash","arguments":{"command":"git remote -v | head -2"}}}]}'
            ]
        }

        events = profile._extract_events("trace_commons", row)

        self.assertEqual(events[0].name, "bash")
        self.assertEqual(events[0].category, "git")

    def test_shell_tool_call_with_compound_typecheck_and_node_test_is_reclassified_as_test(self) -> None:
        row = {
            "messages": [
                '{"role":"assistant","content":"","tool_calls":[{"function":{"name":"Bash","arguments":{"command":"cd /repo/web && npm run -s typecheck && node --experimental-strip-types --test test/native.test.ts 2>&1 | tail -20"}}}]}'
            ]
        }

        events = profile._extract_events("trace_commons", row)

        self.assertEqual(events[0].name, "bash")
        self.assertEqual(events[0].category, "test")

    def test_shell_tool_call_with_adb_install_and_launch_is_reclassified_as_test(self) -> None:
        row = {
            "messages": [
                '{"role":"assistant","content":"","tool_calls":[{"function":{"name":"Bash","arguments":{"command":"adb=~/Android/Sdk/platform-tools/adb && $adb install -r app/build/outputs/apk/debug/app-debug.apk 2>&1 | tail -5 && echo \\"=== launch ===\\" && $adb shell am start -n net.opendsp.x4x4/.MainActivity 2>&1 | tail -3"}}}]}'
            ]
        }

        events = profile._extract_events("trace_commons", row)

        self.assertEqual(events[0].name, "bash")
        self.assertEqual(events[0].category, "test")

    def test_shell_tool_call_with_gh_run_watch_is_reclassified_as_test(self) -> None:
        row = {
            "messages": [
                '{"role":"assistant","content":"","tool_calls":[{"function":{"name":"Bash","arguments":{"command":"gh run watch 27685043250 --exit-status > /tmp/ghrun.log 2>&1; echo \\"WATCH_EXIT=$?\\"; echo \\"=== final status ===\\" && gh run view 27685043250 2>&1 | head -30"}}}]}'
            ]
        }

        events = profile._extract_events("trace_commons", row)

        self.assertEqual(events[0].name, "bash")
        self.assertEqual(events[0].category, "test")

    def test_shell_tool_call_with_build_apk_script_is_reclassified_as_test(self) -> None:
        row = {
            "messages": [
                '{"role":"assistant","content":"","tool_calls":[{"function":{"name":"Bash","arguments":{"command":"export ANDROID_HOME=$HOME/Android/Sdk JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 && ./build-apk.sh > /tmp/apk-build.log 2>&1; echo \\"EXIT=$?\\""}}}]}'
            ]
        }

        events = profile._extract_events("trace_commons", row)

        self.assertEqual(events[0].name, "bash")
        self.assertEqual(events[0].category, "test")

    def test_shell_tool_call_with_node_cdp_eval_is_reclassified_as_test(self) -> None:
        row = {
            "messages": [
                '{"role":"assistant","content":"","tool_calls":[{"function":{"name":"Bash","arguments":{"command":"cd /tmp && node cdp-eval.mjs \\"(async()=>{const s=[];for(let k=0;k<8;k++){s.push(globalThis.device.channels.map(c=>+(c.meter||0).toFixed(2)));await new Promise(r=>setTimeout(r,200));}return s;})()\\""}}}]}'
            ]
        }

        events = profile._extract_events("trace_commons", row)

        self.assertEqual(events[0].name, "bash")
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

    def test_extract_terminalbench_events_reads_step_tools(self) -> None:
        row = {
            "steps": (
                '[{"src":"user","msg":"Fix the task","tools":null,"obs":null},'
                '{"src":"agent","msg":"Executed tools","tools":['
                '{"fn":"bash_command","cmd":"pytest -q"},'
                '{"fn":"Read","cmd":"src/app.py"}],"obs":"Exit code 1\\nFAILED tests/test_app.py::test_x"}]'
            )
        }

        events = profile._extract_events("terminalbench", row)

        self.assertEqual([event.name for event in events], ["run_shell", "read"])
        self.assertEqual([event.category for event in events], ["test", "read"])

    def test_extract_terminalbench_events_normalizes_bash_and_shell_to_run_shell(self) -> None:
        row = {
            "steps": (
                '[{"src":"agent","msg":"Executed Bash","tools":[{"fn":"Bash","cmd":"which R"}],"obs":"Exit code 1"},'
                '{"src":"agent","msg":"Executed shell","tools":[{"fn":"shell","cmd":["bash","-lc","R --version"]}],"obs":"bash: line 1: R: command not found"}]'
            )
        }

        events = profile._extract_events("terminalbench", row)

        self.assertEqual([event.name for event in events], ["run_shell", "run_shell"])
        self.assertEqual([event.category for event in events], ["shell", "shell"])

    def test_extract_terminalbench_events_classifies_r_test_command_as_test(self) -> None:
        row = {
            "steps": (
                '[{"src":"agent","msg":"Executed Bash","tools":[{"fn":"Bash","cmd":"Rscript -e \\"source(\'ars.R\'); test()\\""}],"obs":"ok"}]'
            )
        }

        events = profile._extract_events("terminalbench", row)

        self.assertEqual([event.name for event in events], ["run_shell"])
        self.assertEqual([event.category for event in events], ["test"])

    def test_extract_agent_race_events_reads_pi_tool_calls_and_results(self) -> None:
        row = {
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
                                "arguments": {"command": "hf --help"},
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
                        "content": [{"type": "text", "text": "/bin/sh: 1: hf: not found"}],
                    },
                },
            ]
        }

        events = profile._extract_events("agent_race", row)

        self.assertEqual([event.name for event in events], ["bash", "bash"])

    def test_row_has_trajectory_content_accepts_agent_race_json_string_events(self) -> None:
        row = {
            "events": (
                '[{"type":"message","message":{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"bash","input":{"command":"pytest -q"}}]}},'
                '{"type":"message","message":{"role":"toolResult","toolCallId":"call_1","toolName":"bash","content":[{"type":"text","text":"FAILED tests/test_app.py::test_x\\nE   assert 1 == 2"}]}}]'
            )
        }

        self.assertTrue(profile._row_has_trajectory_content("agent_race", row))

    def test_extract_agent_race_events_reads_tool_use_results_from_user_messages(self) -> None:
        row = {
            "events": [
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tooluse_1",
                                "name": "plan_tool",
                                "input": {"todos": [{"id": "1", "content": "plan", "status": "in_progress"}]},
                            },
                            {
                                "type": "tool_use",
                                "id": "tooluse_2",
                                "name": "bash",
                                "input": {"command": "python -m pytest tests/test_app.py -q"},
                            },
                        ],
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "tooluse_1", "content": "Plan updated"},
                            {"type": "tool_result", "tool_use_id": "tooluse_2", "content": "FAILED tests/test_app.py::test_x"},
                        ],
                    },
                },
            ]
        }

        events = profile._extract_events("agent_race", row)
        metrics = profile._trajectory_metrics(events)

        self.assertEqual([event.name for event in events], ["plan_tool", "bash", "plan_tool", "bash"])
        self.assertEqual(metrics["tool_names"], ["bash"])
        self.assertEqual(metrics["categories"], ["test"])

    def test_iter_agent_race_rows_emits_one_row_per_jsonl_file(self) -> None:
        with _temp_root() as root:
            (root / "claude-code.jsonl").write_text(
                '{"type":"message","message":{"role":"user","content":"task"},"sessionId":"s1"}\n',
                encoding="utf-8",
            )
            (root / "pi-kimi.jsonl").write_text(
                '{"type":"message","message":{"role":"user","content":[{"type":"text","text":"task"}]},"sessionId":"s2"}\n',
                encoding="utf-8",
            )

            rows = list(profile._iter_agent_race_rows(sorted(root.glob("*.jsonl"))))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["session_id"], "s1")
        self.assertEqual(rows[1]["session_id"], "s2")

    def test_terminalbench_helper_tools_do_not_shift_first_edit_or_hide_context_loop(self) -> None:
        row = {
            "steps": (
                '['
                '{"src":"agent","msg":"Executed search","tools":[{"fn":"Grep","cmd":"grep -n todo src/app.py"}],"obs":"ok"},'
                '{"src":"agent","msg":"Executed read","tools":[{"fn":"Read","cmd":"src/app.py"}],"obs":"ok"},'
                '{"src":"agent","msg":"Executed TodoWrite","tools":[{"fn":"TodoWrite","cmd":""}],"obs":"$1"},'
                '{"src":"agent","msg":"Executed search","tools":[{"fn":"Grep","cmd":"grep -n main src/app.py"}],"obs":"ok"},'
                '{"src":"agent","msg":"Executed read","tools":[{"fn":"Read","cmd":"src/lib.py"}],"obs":"ok"},'
                '{"src":"agent","msg":"Executed write","tools":[{"fn":"Write","cmd":"src/app.py"}],"obs":"ok"}'
                ']'
            )
        }

        metrics = profile._trajectory_metrics(profile._extract_events("terminalbench", row))

        self.assertTrue(metrics["context_loop"])
        self.assertEqual(metrics["first_edit_index"], 4)
        self.assertEqual(metrics["tool_names"], ["grep", "read", "grep", "read", "write"])

    def test_terminalbench_killshell_is_ignored_in_profile_metrics(self) -> None:
        row = {
            "steps": (
                '['
                '{"src":"agent","msg":"Executed Bash","tools":[{"fn":"Bash","cmd":"Rscript -e \\"source(\'ars.R\'); test()\\""}],"obs":"ok"},'
                '{"src":"agent","msg":"Executed KillShell","tools":[{"fn":"KillShell","cmd":""}],"obs":"{\\"message\\":\\"Successfully killed shell: abc123\\"}"},'
                '{"src":"agent","msg":"Executed Write","tools":[{"fn":"Write","cmd":"ars.R"}],"obs":"ok"}'
                ']'
            )
        }

        metrics = profile._trajectory_metrics(profile._extract_events("terminalbench", row))

        self.assertEqual(metrics["tool_names"], ["run_shell", "write"])
        self.assertEqual(metrics["categories"], ["test", "edit"])
        self.assertEqual(metrics["first_edit_index"], 1)

    def test_trace_commons_helper_tools_do_not_shift_first_edit_metrics(self) -> None:
        row = {
            "messages": [
                '{"role":"assistant","content":"","tool_calls":[{"function":{"name":"ToolSearch","arguments":{"query":"search tools"}}}]}',
                '{"role":"assistant","content":"","tool_calls":[{"function":{"name":"WebFetch","arguments":{"url":"https://example.com"}}}]}',
                '{"role":"assistant","content":"","tool_calls":[{"function":{"name":"Write","arguments":{"file_path":"app.py","content":"x"}}}]}',
            ]
        }

        metrics = profile._trajectory_metrics(profile._extract_events("trace_commons", row))

        self.assertEqual(metrics["tool_names"], ["write"])
        self.assertEqual(metrics["categories"], ["edit"])
        self.assertEqual(metrics["first_edit_index"], 0)

    def test_extract_thoughtworks_events_dispatches_by_agent_framework(self) -> None:
        openhands_row = {
            "agent_framework": "openhands",
            "messages_json": (
                '[{"role":"assistant","content":"",'
                '"tool_calls_json":"[{\\"function\\":{\\"name\\":\\"execute_bash\\",\\"arguments\\":\\"{\\\\\\"command\\\\\\":\\\\\\"pytest -q\\\\\\"}\\"}}]"}]'
            ),
        }
        mini_swe_agent_row = {
            "agent_framework": "mini-swe-agent",
            "messages_json": (
                '[{"role":"assistant","content":"THOUGHT: inspect\\n```bash\\npytest -q tests/test_app.py\\n```"},'
                '{"role":"user","content":"<returncode>1</returncode>"}]'
            ),
        }

        openhands_events = profile._extract_events("thoughtworks", openhands_row)
        mini_events = profile._extract_events("thoughtworks", mini_swe_agent_row)

        self.assertEqual([event.name for event in openhands_events], ["execute_bash"])
        self.assertEqual(openhands_events[0].category, "test")
        self.assertEqual([event.name for event in mini_events], ["run_test"])
        self.assertEqual(mini_events[0].category, "test")

    def test_row_has_trajectory_content_uses_messages_json_when_messages_is_empty_list_string(self) -> None:
        row = {
            "messages": "[]",
            "messages_json": (
                '[{"role":"assistant","content":"",'
                '"tool_calls_json":"[{\\"function\\":{\\"name\\":\\"execute_bash\\",\\"arguments\\":\\"{\\\\\\"command\\\\\\":\\\\\\"pytest -q\\\\\\"}\\"}}]"}]'
            ),
        }

        self.assertTrue(profile._row_has_trajectory_content("openhands", row))

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

    def test_resolve_dataset_paths_supports_thoughtworks(self) -> None:
        with _temp_root() as root:
            dataset_root = root / "thoughtworks-agentic-coding-trajectories"
            dataset_root.mkdir(parents=True)
            (dataset_root / "sessions.parquet").write_text("stub\n", encoding="utf-8")

            adapter, paths = profile._resolve_dataset_paths(root, "thoughtworks-agentic-coding-trajectories")

        self.assertEqual(adapter, "thoughtworks")
        self.assertEqual([path.name for path in paths], ["sessions.parquet"])

    def test_resolve_dataset_paths_supports_trace_commons(self) -> None:
        with _temp_root() as root:
            dataset_root = root / "trace-commons-agent-traces" / "data"
            dataset_root.mkdir(parents=True)
            (dataset_root / "train-00000-of-00001.parquet").write_text("stub\n", encoding="utf-8")

            adapter, paths = profile._resolve_dataset_paths(root, "trace-commons-agent-traces")

        self.assertEqual(adapter, "trace_commons")
        self.assertEqual([path.name for path in paths], ["train-00000-of-00001.parquet"])

    def test_resolve_dataset_paths_supports_cc_bench(self) -> None:
        with _temp_root() as root:
            dataset_root = root / "cc-bench-trajectories"
            dataset_root.mkdir(parents=True)
            (dataset_root / "train.parquet").write_text("stub\n", encoding="utf-8")

            adapter, paths = profile._resolve_dataset_paths(root, "cc-bench-trajectories")

        self.assertEqual(adapter, "cc_bench")
        self.assertEqual([path.name for path in paths], ["train.parquet"])

    def test_extract_events_trace_commons_supports_raw_session_trace_rows(self) -> None:
        row = {
            "trace": json.dumps(
                [
                    {
                        "type": "user",
                        "message": {"role": "user", "content": "Fix the parser and rerun the tests."},
                    },
                    {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "tool_use", "name": "Read", "input": {"file_path": "parser.py"}},
                                {"type": "tool_use", "name": "Bash", "input": {"command": "pytest -q"}},
                            ],
                        },
                    },
                ]
            )
        }

        events = profile._extract_events("trace_commons", row)
        metrics = profile._trajectory_metrics(events)

        self.assertEqual(metrics["tool_calls"], 2)
        self.assertEqual(metrics["categories"], ["read", "test"])

    def test_iter_parquet_rows_honors_max_rows(self) -> None:
        if pa is None or pq is None:
            self.skipTest("pyarrow is not installed")
        with _temp_root() as root:
            path = root / "sample.parquet"
            table = pa.table({"value": [1, 2, 3, 4]})
            pq.write_table(table, path)

            rows = list(profile._iter_parquet_rows([path], max_rows=2))

        self.assertEqual(rows, [{"value": 1}, {"value": 2}])

    def test_iter_rows_with_trajectory_content_skips_terminalbench_null_rows_before_max_rows(self) -> None:
        rows = [
            {"steps": "null", "trial_id": "skip-1"},
            {"steps": '[{"src":"agent","msg":"Executed Bash","tools":[{"fn":"bash_command","cmd":"pytest -q"}],"obs":"ok"}]', "trial_id": "keep-1"},
            {"steps": '[{"src":"agent","msg":"Executed Bash","tools":[{"fn":"bash_command","cmd":"pytest -q"}],"obs":"ok"}]', "trial_id": "keep-2"},
        ]

        filtered = list(profile._iter_rows_with_trajectory_content("terminalbench", rows, max_rows=1))

        self.assertEqual([row["trial_id"] for row in filtered], ["keep-1"])

    def test_iter_rows_with_trajectory_content_skips_empty_openhands_rows_before_max_rows(self) -> None:
        rows = [
            {"trajectory": None, "instance_id": "skip-1"},
            {
                "trajectory": '[{"role":"assistant","content":"","tool_calls":[{"function":{"name":"read_file","arguments":"{}"}}]}]',
                "instance_id": "keep-1",
            },
            {
                "trajectory": '[{"role":"assistant","content":"","tool_calls":[{"function":{"name":"run_test","arguments":"{\\"command\\":\\"pytest -q\\"}"}}]}]',
                "instance_id": "keep-2",
            },
        ]

        filtered = list(profile._iter_rows_with_trajectory_content("openhands", rows, max_rows=1))

        self.assertEqual([row["instance_id"] for row in filtered], ["keep-1"])

    def test_iter_dataset_rows_skips_terminalbench_null_rows_before_max_rows(self) -> None:
        if pa is None or pq is None:
            self.skipTest("pyarrow is not installed")
        with _temp_root() as root:
            dataset_root = root / "terminalbench-trajectories" / "data"
            dataset_root.mkdir(parents=True)
            table = pa.table(
                {
                    "trial_id": ["skip-1", "keep-1", "keep-2"],
                    "steps": [
                        "null",
                        '[{"src":"agent","msg":"Executed Bash","tools":[{"fn":"bash_command","cmd":"pytest -q"}],"obs":"ok"}]',
                        '[{"src":"agent","msg":"Executed Bash","tools":[{"fn":"bash_command","cmd":"pytest -q"}],"obs":"ok"}]',
                    ],
                }
            )
            pq.write_table(table, dataset_root / "train-00000-of-00001.parquet")

            adapter, rows = profile._iter_dataset_rows(root, "terminalbench-trajectories", 1)
            filtered = list(rows)

        self.assertEqual(adapter, "terminalbench")
        self.assertEqual([row["trial_id"] for row in filtered], ["keep-1"])

    def test_iter_dataset_rows_skips_empty_openhands_rows_before_max_rows(self) -> None:
        if pa is None or pq is None:
            self.skipTest("pyarrow is not installed")
        with _temp_root() as root:
            dataset_root = root / "nebius-swe-rebench-openhands-trajectories"
            dataset_root.mkdir(parents=True)
            table = pa.table(
                {
                    "instance_id": ["skip-1", "keep-1", "keep-2"],
                    "trajectory": [
                        None,
                        '[{"role":"assistant","content":"","tool_calls":[{"function":{"name":"read_file","arguments":"{}"}}]}]',
                        '[{"role":"assistant","content":"","tool_calls":[{"function":{"name":"run_test","arguments":"{\\"command\\":\\"pytest -q\\"}"}}]}]',
                    ],
                }
            )
            pq.write_table(table, dataset_root / "trajectories.parquet")

            adapter, rows = profile._iter_dataset_rows(root, "nebius-swe-rebench-openhands-trajectories", 1)
            filtered = list(rows)

        self.assertEqual(adapter, "openhands")
        self.assertEqual([row["instance_id"] for row in filtered], ["keep-1"])


if __name__ == "__main__":
    unittest.main()
