import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ollama_code.agent import OllamaCodeAgent
from ollama_code.features import ENV_OLLAMA_CODE_FEATURE_PROFILE
from ollama_code.tools import ToolExecutor
from tests.agent_test_support import AgentTestBase, CountingToolExecutor, FakeClient


class AgentFailureCompressionTests(AgentTestBase):
    def _cwd_agent(self, client: FakeClient | None = None, **kwargs: object) -> OllamaCodeAgent:
        resolved_client = client if client is not None else FakeClient([])
        return OllamaCodeAgent(
            client=resolved_client,
            tools=ToolExecutor(self._workspace_scratch(), approval_mode="auto"),
            model="fake-model",
            **kwargs,
        )

    def test_trajectory_failure_delta_compacts_repeated_test_failure(self) -> None:
        root = self._workspace_scratch()
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)

        delta = agent._failure_delta_summary(
            "FAILED test_ops.py::test_value | AssertionError: expected 1 got 0",
            "FAILED test_ops.py::test_value | AssertionError: expected 1 got 2",
        )

        self.assertIn("expected 1 got 2", delta)
        self.assertNotIn("expected 1 got 0", delta)

    def test_agent_compacts_large_tool_results_in_follow_up_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            large_lines = "\n".join(f"line {index} " + ("x" * 80) for index in range(1, 201)) + "\n"
            (root / "big.txt").write_text(large_lines, encoding="utf-8")
            client = FakeClient(
                [
                    '{"type":"tool","name":"read_file","arguments":{"path":"big.txt"}}',
                    '{"type":"final","message":"done"}',
                ]
            )
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
            result = agent.handle_user("Summarize big.txt.")

        self.assertEqual(result.message, "done")
        tool_feedback = next(
            message["content"]
            for message in agent.messages
            if message["role"] == "user" and "read_file" in message["content"] and "... truncated ..." in message["content"]
        )
        self.assertIn("... truncated ...", tool_feedback)
        self.assertNotIn(" 200 |", tool_feedback)

    def test_agent_compacts_large_write_arguments_in_assistant_history_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content = "\n".join(f"# line {index} TOKEN_FULL_EVENT" for index in range(80)) + "\n"
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "big.py", "content": content}}),
                    json.dumps({"type": "final", "message": "big.py updated"}),
                ]
            )
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
            result = agent.handle_user("Write big.py with generated content.")

        self.assertEqual(result.message, "big.py updated")
        assistant_tool = next(
            message["content"]
            for message in agent.messages
            if message["role"] == "assistant" and '"name":"write_file"' in message["content"]
        )
        self.assertIn("[omitted", assistant_tool)
        self.assertIn("do not copy", assistant_tool)
        self.assertNotIn("line 79 TOKEN_FULL_EVENT", assistant_tool)
        tool_call = next(event for event in agent.events if event["type"] == "tool_call" and event.get("name") == "write_file")
        self.assertIn("line 79 TOKEN_FULL_EVENT", tool_call["arguments"]["content"])

    def test_agent_compacts_run_test_output_to_actionable_failure(self) -> None:
        agent = self._cwd_agent()
        noisy_status = "\n".join(f"test_noise_{index} (suite.Case.test_noise_{index}) ... FAIL" for index in range(80))
        output = (
            noisy_status
            + "\n"
            + "-" * 70
            + "\nFAIL: test_add_negative (tests.test_math.MathTests.test_add_negative)\n"
            + "-" * 70
            + "\nTraceback (most recent call last):\n"
            + '  File "tests/test_math.py", line 42, in test_add_negative\n'
            + "    self.assertEqual(add(-2, -3), -5)\n"
            + "AssertionError: -4 != -5\n\n"
            + "unhelpful tail " + ("x" * 3000) + "\n"
            + "Ran 81 tests in 0.123s\nFAILED (failures=81)\n"
        )

        payload = agent._compact_tool_result_for_context("run_test", {"ok": False, "output": output})

        compact = payload["output"]
        self.assertIn("FAILED (failures=81)", compact)
        self.assertIn("FAIL: test_add_negative", compact)
        self.assertIn("AssertionError: -4 != -5", compact)
        self.assertNotIn("test_noise_0", compact)
        self.assertNotIn("unhelpful tail", compact)
        self.assertLessEqual(len(compact), 700)

    def test_agent_run_test_feedback_points_at_syntax_file_not_import_guess(self) -> None:
        agent = self._cwd_agent()
        output = (
            "ERROR: sample_test (unittest.loader._FailedTest.sample_test)\n"
            "Traceback (most recent call last):\n"
            '  File "C:\\workspace\\sample.py", line 2\n'
            "    def f():\n"
            "IndentationError: unexpected indent\n"
        )

        feedback = agent._tool_result_feedback_message("run_test", {"ok": False, "output": output}, real_tool_use=True)

        self.assertIn("IndentationError: unexpected indent at sample.py:2", feedback)
        self.assertIn("Do not blame imports unless error is ModuleNotFoundError", feedback)

    def test_agent_run_test_feedback_includes_failing_source_excerpt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "list_ops.py").write_text("def foldr(function, items, initial):\n    return initial\n", encoding="utf-8")
            test_file = root / "list_ops_test.py"
            test_file.write_text(
                "from list_ops import foldr\n\n"
                "import unittest\n\n"
                "class ListOpsTest(unittest.TestCase):\n"
                "    def test_foldr_add_string(self):\n"
                "        self.assertEqual(\n"
                "            foldr(lambda acc, el: el + acc, ['e', 'x'], '!'),\n"
                "            'ex!'\n"
                "        )\n",
                encoding="utf-8",
            )
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")
            output = (
                "FAIL: test_foldr_add_string (list_ops_test.ListOpsTest.test_foldr_add_string)\n"
                "Traceback (most recent call last):\n"
                f'  File "{test_file}", line 7, in test_foldr_add_string\n'
                "    self.assertEqual(\n"
                "AssertionError: '!xe' != 'ex!'\n"
            )

            feedback = agent._tool_result_feedback_message("run_test", {"ok": False, "output": output}, real_tool_use=True)

            self.assertIn("AssertionError: '!xe' != 'ex!'", feedback)
            self.assertIn("Diagnosis:", feedback)
            self.assertIn("actual='!xe' expected='ex!'", feedback)
            self.assertIn("list_ops.py", feedback)
            self.assertIn("Failing source excerpt", feedback)
            self.assertIn("list_ops_test.py:7", feedback)
            self.assertIn("foldr(lambda acc, el: el + acc", feedback)

    def test_agent_failed_omitted_context_write_points_to_partial_edit_tools(self) -> None:
        agent = self._cwd_agent()

        feedback = agent._tool_result_feedback_message(
            "write_file",
            {
                "ok": False,
                "summary": "Refusing to write omitted-context marker as file content. Reconstruct complete file content instead.",
            },
            real_tool_use=False,
        )

        self.assertIn("Use replace_symbol/replace_in_file", feedback)
        self.assertIn("content was abbreviated", feedback)

    def test_agent_failed_cli_proof_points_to_parser_repair(self) -> None:
        agent = self._cwd_agent()

        feedback = agent._tool_result_feedback_message(
            "run_shell",
            {
                "ok": False,
                "exit_code": 2,
                "output": (
                    "usage: notes_cli.py [-h] [--tag TAG] [--json]\n"
                    "notes_cli.py: error: unrecognized arguments: --limit 1\n"
                ),
            },
            real_tool_use=False,
            arguments={"command": f"{sys.executable} notes_cli.py --tag work --limit 1 --json"},
        )

        self.assertIn("app parser rejected --limit", feedback)
        self.assertIn("Repair the source command surface first", feedback)
        self.assertIn("do not use replace_symbol on a local parser variable", feedback)
        self.assertIn("rerun the same shell proof and tests", feedback)

    def test_agent_failed_pytest_flag_does_not_point_to_app_parser_repair(self) -> None:
        agent = self._cwd_agent()

        feedback = agent._tool_result_feedback_message(
            "run_shell",
            {
                "ok": False,
                "exit_code": 4,
                "output": "usage: pytest [options]\npytest: error: unrecognized arguments: --wat\n",
            },
            real_tool_use=False,
            arguments={"command": "python -m pytest --wat"},
        )

        self.assertIn("Correct the command flags", feedback)
        self.assertNotIn("app parser", feedback)

    def test_agent_missing_edit_symbol_feedback_rejects_invented_helper(self) -> None:
        agent = self._cwd_agent()

        feedback = agent._tool_result_feedback_message(
            "edit_intent",
            {"ok": False, "summary": "change Python function body: Symbol not found: _parse_args"},
            real_tool_use=False,
            arguments={"path": "notes_cli.py", "intent": "replace_body", "target": "_parse_args"},
        )

        self.assertIn("Edit target `_parse_args` was not found", feedback)
        self.assertIn("Do not invent helper symbols", feedback)
        self.assertIn("existing symbol such as main", feedback)

    def test_agent_missing_edit_function_feedback_rejects_import_target(self) -> None:
        agent = self._cwd_agent()

        feedback = agent._tool_result_feedback_message(
            "edit_intent",
            {"ok": False, "summary": "replace Python function body: Function not found: list_bookmarks"},
            real_tool_use=False,
            arguments={"path": "bookmarks/cli.py", "intent": "replace_body", "target": "list_bookmarks"},
        )

        self.assertIn("Edit target `list_bookmarks` was not found", feedback)
        self.assertIn("Imported names are not implementation symbols", feedback)
        self.assertIn("edit the defining module", feedback)

    def test_code_outline_retry_message_distinguishes_imports_from_implementation_symbols(self) -> None:
        agent = self._cwd_agent()

        feedback = agent._context_planner_probe_retry_message(
            probe_name="code_outline",
            probe_arguments={"path": "bookmarks/cli.py"},
            probe_result={
                "ok": True,
                "path": "bookmarks/cli.py",
                "output": "imports: from .store import list_bookmarks\n9-12 function main",
            },
            mutation_required=True,
            code_mutation_required=True,
            test_run_required=True,
            required_mutation_paths=set(),
            mutated_paths_this_turn=set(),
            successful_tool_results=[],
            broad=False,
        )

        self.assertIn("Only edit functions/classes listed in the outline", feedback)
        self.assertIn("imports are dependencies", feedback)
        self.assertIn("defining module", feedback)

    def test_context_guard_for_mutation_forbids_more_context_after_limit(self) -> None:
        agent = self._cwd_agent()

        feedback = agent._context_guard_retry_message(
            mutation_required=True,
            code_mutation_required=True,
            test_run_required=True,
            required_mutation_paths=set(),
            mutated_paths_this_turn=set(),
            successful_tool_results=[],
            broad=False,
        )

        self.assertIn("Do not call read_file", feedback)
        self.assertIn("Use edit_intent", feedback)
        self.assertIn("then run_test", feedback)
        self.assertIn("fail closed", feedback)

    def test_final_chance_auto_run_test_records_auto_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
            command = f"{sys.executable} -c \"print('ok')\""
            client = FakeClient(
                [
                    json.dumps(
                        {
                            "type": "tool",
                            "name": "write_file",
                            "arguments": {"path": "app.py", "content": "VALUE = 2\n"},
                        }
                    )
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=1)

            with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "baseline"}):
                result = agent.handle_user("Change app.py and run tests.")

        self.assertTrue(result.completed)
        self.assertIn("Ran tests after the latest edit", result.message)
        auto_run_tests = [
            event
            for event in agent.events
            if event.get("type") == "tool_result" and event.get("name") == "run_test" and event.get("auto") is True
        ]
        self.assertTrue(auto_run_tests)
        self.assertEqual(auto_run_tests[-1].get("arguments"), {})

    def test_candidate_cli_proof_commands_include_requested_limit_flag(self) -> None:
        agent = self._cwd_agent()

        commands = agent._candidate_cli_proof_commands(
            "notes_cli.py",
            (
                "parser.add_argument('--tag')\n"
                "parser.add_argument('--json', action='store_true')\n"
                "parser.add_argument('--limit', type=int)\n"
            ),
            "Add a --limit N option that works with --json.",
        )

        self.assertTrue(any("--tag work --limit 1 --json" in command for command in commands), commands)

    def test_agent_blocks_repeated_failed_run_test_until_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fail_command = subprocess.list2cmdline([sys.executable, "-c", "import sys; sys.exit(1)"])
            pass_command = subprocess.list2cmdline([sys.executable, "-c", "print('ok')"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": fail_command}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": fail_command}}),
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": "def f():\n    return 1\n"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": pass_command}}),
                    json.dumps({"type": "final", "message": "changed app.py"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
                result = agent.handle_user("Fix app.py, run tests, and rerun tests after editing.")

        self.assertEqual(result.message, "changed app.py")
        self.assertEqual(tools.execute_counts.get("run_test"), 2)
        self.assertEqual(tools.execute_counts.get("write_file"), 1)

        tool_names = [
            event.get("name")
            for event in agent.events
            if event.get("type") == "tool_call" and event.get("name") != "context_pack"
        ]
        self.assertEqual(tool_names[:4], ["run_test", "diagnose_test_failure", "write_file", "run_test"])
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Use the diagnosis above to edit implementation before rerunning run_test", feedback)

    def test_agent_blocks_false_test_success_after_failed_run_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fail_command = subprocess.list2cmdline([sys.executable, "-c", "import sys; print('FAILED'); sys.exit(1)"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": fail_command}}),
                    json.dumps({"type": "final", "message": "All tests passed successfully."}),
                    json.dumps({"type": "final", "message": "Tests failed with exit code 1."}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=3)

            with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
                result = agent.handle_user("Run tests and summarize the result.")

        self.assertFalse(result.completed)
        self.assertNotIn("All tests passed", result.message)
        self.assertEqual(tools.execute_counts.get("run_test"), 1)
        self.assertTrue(any("do not claim tests passed" in message["content"] for message in agent.messages if message["role"] == "user"))

    def test_agent_requires_edit_after_failed_tests_for_fix_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fail_command = subprocess.list2cmdline([sys.executable, "-c", "import sys; print('AssertionError: None != 1'); sys.exit(1)"])
            pass_command = subprocess.list2cmdline([sys.executable, "-c", "print('OK')"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": fail_command}}),
                    json.dumps({"type": "final", "message": "The test failure shows the bug."}),
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": "def f():\n    return 1\n"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": pass_command}}),
                    json.dumps({"type": "final", "message": "Fixed app.py and tests passed."}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=12)

            with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
                result = agent.handle_user("Fix app.py, run tests, and summarize.")

        self.assertEqual(result.message, "Fixed app.py and tests passed.")
        self.assertEqual(tools.execute_counts.get("write_file"), 1)
        self.assertEqual(tools.execute_counts.get("run_test"), 2)
        self.assertTrue(any("no implementation edit succeeded" in message["content"] for message in agent.messages if message["role"] == "user"))

    def test_agent_reconciles_failed_test_artifact_and_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fail_command = subprocess.list2cmdline([sys.executable, "-c", "import sys; print('AssertionError: 0 != 1'); sys.exit(1)"])
            pass_command = subprocess.list2cmdline([sys.executable, "-c", "print('OK')"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": fail_command}}),
                    json.dumps(
                        {
                            "verdict": "retry",
                            "reason": "The failing test needs implementation repair before final.",
                            "repair_plan": ["edit implementation", "rerun tests"],
                            "required_tools": ["write_file"],
                            "forbidden_tools": [],
                        }
                    ),
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": "def f():\n    return 1\n"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": pass_command}}),
                    json.dumps({"type": "final", "message": "Fixed app.py and tests passed."}),
                ],
                script_reconciliation=True,
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, reconcile_mode="auto")

            with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
                result = agent.handle_user("Fix app.py, run tests, and summarize.")

        self.assertEqual(result.message, "Fixed app.py and tests passed.")
        reconciliations = [event for event in agent.events if event["type"] == "reconciliation"]
        self.assertEqual([event["verdict"] for event in reconciliations], ["retry"])
        self.assertTrue(any("Artifact reconciliation rejected" in message["content"] for message in agent.messages if message["role"] == "user"))
        self.assertEqual([call["think"] for call in client.calls if str(call["messages"][0]["content"]).startswith("You are an artifact reconciliation critic")], [False])

    def test_agent_reconcile_off_skips_failed_test_artifact_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fail_command = subprocess.list2cmdline([sys.executable, "-c", "import sys; print('FAILED'); sys.exit(1)"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": fail_command}}),
                    json.dumps({"type": "final", "message": "Tests failed with exit code 1."}),
                ],
                script_reconciliation=True,
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(
                client=client,
                tools=tools,
                model="fake-model",
                debate_enabled=False,
                reconcile_mode="off",
                max_tool_rounds=2,
            )

            with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
                result = agent.handle_user("Run tests and summarize the result.")

        self.assertFalse(result.completed)
        self.assertFalse(any(event["type"] == "reconciliation" for event in agent.events))

    def test_agent_reconcile_auto_skips_failed_edit_artifact(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient([])
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", reconcile_mode="auto")

        needs_reconciliation = agent._tool_result_needs_reconciliation(
            request_text="Fix app.py and run tests.",
            name="replace_symbol",
            result={"ok": False, "summary": "Symbol not found: f"},
            cache_hit=False,
            session_memory_request=False,
            mutation_required=True,
            test_run_required=True,
        )

        self.assertFalse(needs_reconciliation)

    def test_agent_runs_final_chance_test_after_last_round_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pass_command = subprocess.list2cmdline([sys.executable, "-c", "print('OK')"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": "def f():\n    return 1\n"}}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=pass_command)
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=1)

            with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
                result = agent.handle_user("Edit app.py and run tests.")

        self.assertTrue(result.completed)
        self.assertEqual(result.message, "Ran tests after the latest edit: passed.")
        self.assertEqual(tools.execute_counts.get("write_file"), 1)
        self.assertEqual(tools.execute_counts.get("run_test"), 1)
        auto_results = [event for event in agent.events if event["type"] == "tool_result" and event["name"] == "run_test"]
        self.assertTrue(auto_results[0]["auto"])

    def test_agent_requires_edits_to_explicitly_named_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "docs").mkdir()
            pass_command = subprocess.list2cmdline([sys.executable, "-c", "print('OK')"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "src/app.py", "content": "def f():\n    return 1\n"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": pass_command}}),
                    json.dumps({"type": "final", "message": "Tests passed."}),
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "docs/app.md", "content": "Updated docs.\n"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": pass_command}}),
                    json.dumps({"type": "final", "message": "Updated src and docs."}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=10)

            with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
                result = agent.handle_user("Update src/app.py and docs/app.md, then run tests.")

        self.assertEqual(result.message, "Updated src and docs.")
        self.assertIn("docs/app.md", " ".join(message["content"] for message in agent.messages if message["role"] == "user"))
        self.assertEqual(tools.execute_counts.get("write_file"), 2)
        self.assertEqual(tools.execute_counts.get("run_test"), 2)

    def test_agent_fails_closed_after_reconciliation_retry_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fail_commands = [
                subprocess.list2cmdline([sys.executable, "-c", f"import sys; print('FAIL {index}'); sys.exit(1)"])
                for index in range(3)
            ]
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": fail_commands[0]}}),
                    json.dumps({"verdict": "retry", "reason": "repair first", "repair_plan": ["edit"], "required_tools": [], "forbidden_tools": []}),
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": "def f():\n    return 1\n"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": fail_commands[1]}}),
                    json.dumps({"verdict": "retry", "reason": "still failing", "repair_plan": ["edit again"], "required_tools": [], "forbidden_tools": []}),
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": "def f():\n    return 2\n"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": fail_commands[2]}}),
                    json.dumps({"verdict": "retry", "reason": "still no approved path", "repair_plan": ["stop"], "required_tools": [], "forbidden_tools": []}),
                ],
                script_reconciliation=True,
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, reconcile_mode="on", max_tool_rounds=6)

            with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
                result = agent.handle_user("Fix app.py, run tests, and keep repairing until tests pass.")

        self.assertFalse(result.completed)
        self.assertIn("Validation already failed after a prior edit", result.message)
        reconciliations = [event for event in agent.events if event["type"] == "reconciliation"]
        self.assertEqual(len(reconciliations), 2)

    def test_trajectory_loop_cap_blocks_fourth_context_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient(
                [
                    '{"type":"tool","name":"search_symbols","arguments":{"query":"greet","path":"app.py"}}',
                    '{"type":"tool","name":"read_symbol","arguments":{"path":"app.py","symbol":"greet"}}',
                    '{"type":"tool","name":"code_outline","arguments":{"path":"app.py"}}',
                    '{"type":"tool","name":"read_file","arguments":{"path":"app.py"}}',
                    '{"type":"final","message":"hello world"}',
                ]
            )
            (root / "app.py").write_text("def greet():\n    return 'hello world'\n", encoding="utf-8")
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=5)

            with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
                result = agent.handle_user("Inspect app.py and summarize the relevant text.")

        self.assertEqual(result.message, "hello world")
        self.assertTrue(any(event.get("type") == "controller_guard" and event.get("guard") == "loop-cap" for event in agent.events))
        self.assertEqual([event.get("name") for event in agent.events if event.get("type") == "tool_call"], ["search_symbols", "read_symbol", "code_outline"])

    def test_context_planner_blocks_third_broad_context_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient(
                [
                    '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                    '{"type":"tool","name":"search","arguments":{"query":"hello"}}',
                    '{"type":"tool","name":"read_file","arguments":{"path":"note.txt","start":1,"end":1}}',
                    '{"type":"final","message":"hello world"}',
                ]
            )
            (root / "note.txt").write_text("hello world\n", encoding="utf-8")
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=5)

            with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
                result = agent.handle_user("Inspect this repo and find the relevant hello text.")

        self.assertEqual(result.message, "hello world")
        self.assertTrue(any(event.get("type") == "controller_guard" and event.get("guard") == "context-planner" for event in agent.events))
        self.assertEqual([event.get("name") for event in agent.events if event.get("type") == "tool_call"], ["read_file", "search"])

    def test_agent_blocks_third_identical_cached_symbol_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "core.py").write_text("def wrapped():\n    return 'ok'\n", encoding="utf-8")
            client = FakeClient(
                [
                    '{"type":"tool","name":"search_symbols","arguments":{"query":"wrapped","path":"src"}}',
                    '{"type":"tool","name":"search_symbols","arguments":{"query":"wrapped","path":"src"}}',
                    '{"type":"tool","name":"search_symbols","arguments":{"query":"wrapped","path":"src"}}',
                    '{"type":"final","message":"wrapped is in src/core.py"}',
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=5)
            result = agent.handle_user("Find the wrapped implementation in src and summarize the match.")

        self.assertEqual(result.message, "wrapped is in src/core.py")
        self.assertEqual(tools.execute_counts.get("search_symbols"), 1)
        search_tool_calls = [event for event in agent.events if event.get("type") == "tool_call" and event.get("name") == "search_symbols"]
        self.assertEqual(len(search_tool_calls), 2)
        tool_results = [event for event in agent.events if event.get("type") == "tool_result" and event.get("name") == "search_symbols"]
        self.assertEqual(len(tool_results), 2)
        self.assertTrue(tool_results[1].get("cached", False))
        self.assertTrue(any(event.get("type") == "controller_guard" and event.get("guard") == "loop-cap" for event in agent.events))

    def test_trajectory_failure_compression_auto_diagnoses_repeated_run_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fail_command = subprocess.list2cmdline(
                [sys.executable, "-c", "import sys; print('AssertionError: 0 != 1'); sys.exit(1)"]
            )
            pass_command = subprocess.list2cmdline([sys.executable, "-c", "print('ok')"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": fail_command}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": fail_command}}),
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": "def f():\n    return 1\n"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": pass_command}}),
                    json.dumps({"type": "final", "message": "changed app.py"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
                result = agent.handle_user("Fix app.py, run tests, and rerun tests after editing.")

        self.assertEqual(result.message, "changed app.py")
        self.assertEqual(tools.execute_counts.get("run_test"), 2)
        self.assertEqual(tools.execute_counts.get("diagnose_test_failure"), 1)
        self.assertTrue(any(event.get("type") == "controller_guard" and event.get("guard") == "failure-compression" for event in agent.events))
        tool_names = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_names[:4], ["run_test", "diagnose_test_failure", "write_file", "run_test"])
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Use the diagnosis above to edit implementation before rerunning run_test.", feedback)

    def test_trajectory_failure_compression_diagnoses_first_failed_test_before_more_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def f():\n    return 0\n", encoding="utf-8")
            fail_command = subprocess.list2cmdline(
                [sys.executable, "-c", "import sys; print('AssertionError: 0 != 1'); sys.exit(1)"]
            )
            pass_command = subprocess.list2cmdline([sys.executable, "-c", "print('ok')"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": fail_command}}),
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "app.py"}}),
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": "def f():\n    return 1\n"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": pass_command}}),
                    json.dumps({"type": "final", "message": "changed app.py"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
                result = agent.handle_user("Fix app.py, run tests, and rerun tests after editing.")
            final_text = (root / "app.py").read_text(encoding="utf-8")

        self.assertEqual(result.message, "changed app.py")
        self.assertEqual(tools.execute_counts.get("run_test"), 2)
        self.assertEqual(tools.execute_counts.get("diagnose_test_failure"), 1)
        self.assertIsNone(tools.execute_counts.get("read_file"))
        self.assertEqual(final_text, "def f():\n    return 1\n")
        self.assertTrue(any(event.get("type") == "controller_guard" and event.get("guard") == "diagnose-first-failed-test" for event in agent.events))
