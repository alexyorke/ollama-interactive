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


class AgentShellCommandPreflightTests(AgentTestBase):
    def test_agent_audits_shell_tool_under_debate(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"run_shell","arguments":{"command":"python -c \\"print(123)\\""}}',
                '{"type":"final","message":"done"}',
                '{"verdict":"accept"}',
            ]
        )
        root = self._workspace_scratch()
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        result = agent.handle_user('Use run_shell to execute python -c "print(123)" and then say done.')

        self.assertEqual(result.message, "done")
        audits = [event for event in agent.events if event["type"] == "assumption_audit"]
        self.assertEqual(len(audits), 1)
        self.assertEqual(audits[0]["tool"], "run_shell")

    def test_agent_recovers_exact_shell_command_after_invalid_json(self) -> None:
        client = FakeClient(["not json"])
        root = self._workspace_scratch()
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")
        exact_command = 'python -c "import sys; print(\'boom\'); sys.exit(5)"'

        result = agent.handle_user(
            f"Use run_shell to execute exactly: {exact_command}. Then tell me the exit code and the printed word."
        )

        self.assertIn("Exit code: 5", result.message)
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_calls[0]["arguments"]["command"], exact_command)

    def test_agent_recovers_exact_shell_command_skips_assumption_audit_under_debate(self) -> None:
        client = FakeClient(["not json"])
        root = self._workspace_scratch()
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")
        exact_command = 'python -c "import sys; print(\'boom\'); sys.exit(5)"'

        result = agent.handle_user(
            f"Use run_shell to execute exactly: {exact_command}. Then tell me the exit code and the printed word."
        )

        self.assertIn("Exit code: 5", result.message)
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_calls[0]["arguments"]["command"], exact_command)
        audits = [event for event in agent.events if event["type"] == "assumption_audit"]
        self.assertEqual(audits, [])

    def test_agent_salvages_malformed_assumption_audit_payload(self) -> None:
        root = self._workspace_scratch()
        exact_command = subprocess.list2cmdline([sys.executable, "-c", "print(42)"])
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": exact_command}}),
                '{"verdict":"accept","reason":"Command directly answers the request.","assumptions":["Python is available."],"validation_steps":["Run the exact command."],"required_tools:["run_shell"],"forbidden_tools":[]}',
                json.dumps({"type": "final", "message": "done"}),
            ],
            script_assumption_audit=True,
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        result = agent.handle_user(f"Use run_shell to execute exactly: {exact_command}. Then say done.")

        self.assertEqual(result.message, "done")
        audits = [event for event in agent.events if event["type"] == "assumption_audit"]
        self.assertEqual(len(audits), 1)
        self.assertEqual(audits[0]["verdict"], "accept")
        self.assertIn("Run the exact command.", audits[0]["validation_steps"])
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_calls[0]["name"], "run_shell")

    def test_agent_normalizes_vague_run_test_to_configured_command(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient(['{"type":"tool","name":"run_test","arguments":{"command":"test"}}'])
        tools = ToolExecutor(root, approval_mode="auto", test_command='python -c "print(\'test_sample OK\')"')
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        result = agent.handle_user("Use run_test and tell me whether tests passed and which test module ran.")

        self.assertIn("Tests passed: yes", result.message)
        self.assertIn("test_sample", result.message)
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_calls[0]["arguments"]["command"], 'python -c "print(\'test_sample OK\')"')

    def test_agent_normalizes_test_tool_alias_to_configured_run_test(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient(['{"type":"tool","name":"test","arguments":{}}'])
        tools = ToolExecutor(root, approval_mode="auto", test_command='python -c "print(\'test_alias OK\')"')
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        result = agent.handle_user("Run tests and tell me whether tests passed.")

        self.assertIn("Tests passed: yes", result.message)
        self.assertEqual(len(client.calls), 0)
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_calls[0]["name"], "run_test")
        self.assertEqual(tool_calls[0]["arguments"]["command"], 'python -c "print(\'test_alias OK\')"')

    def test_agent_normalizes_shell_test_to_configured_run_test(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient(['{"type":"tool","name":"run_shell","arguments":{"command":"python -m unittest example_test.py"}}'])
        tools = ToolExecutor(root, approval_mode="auto", test_command='python -c "print(\'test_polyglot OK\')"')
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        result = agent.handle_user("Validate the project and tell me whether tests passed.")

        self.assertIn("Tests passed: yes", result.message)
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_calls[0]["name"], "run_test")
        self.assertEqual(tool_calls[0]["arguments"]["command"], 'python -c "print(\'test_polyglot OK\')"')
        normalizations = [event for event in agent.events if event["type"] == "tool_normalized"]
        self.assertEqual(normalizations[0]["normalized_name"], "run_test")

    def test_agent_normalizes_shell_test_to_original_run_test_without_configured_command(self) -> None:
        root = self._workspace_scratch()
        (root / "tests").mkdir()
        (root / "tests" / "test_sample.py").write_text(
            "import unittest\n\nclass SampleTests(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n",
            encoding="utf-8",
        )
        client = FakeClient(['{"type":"tool","name":"run_shell","arguments":{"command":"python -m unittest discover -s tests -v"}}'])
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        result = agent.handle_user("Validate the project and tell me whether tests passed.")

        self.assertIn("Tests passed: yes", result.message)
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_calls[0]["name"], "run_test")
        self.assertEqual(tool_calls[0]["arguments"]["command"], "python -m unittest discover -s tests -v")
        normalizations = [event for event in agent.events if event["type"] == "tool_normalized"]
        self.assertEqual(normalizations[0]["normalized_name"], "run_test")
        self.assertIn("original command", normalizations[0]["reason"])

    def test_agent_preserves_exact_user_requested_shell_test_command(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient(['{"type":"tool","name":"run_shell","arguments":{"command":"python -m pytest --version"}}'])
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        result = agent.handle_user("Use run_shell to execute exactly: python -m pytest --version. Then tell me the exit code.")

        self.assertIn("exit code", result.message.lower())
        self.assertEqual(tools.execute_counts.get("run_shell"), 1)
        self.assertIsNone(tools.execute_counts.get("run_test"))

    def test_agent_treats_disabled_tools_as_forbidden_up_front(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient(
                [
                    '{"type":"tool","name":"run_shell","arguments":{"command":"python scripts/e2e_suite.py --model gemma4:e4b --scenarios scenario_run_test scenario_git_tools"}}',
                    '{"type":"tool","name":"run_test","arguments":{"command":"python -c \\"print(\'e2e OK\')\\""}}',
                    '{"type":"final","message":"done"}',
                ]
            )
            tools = ToolExecutor(
                root,
                approval_mode="auto",
                disabled_tools=["run_shell"],
            )
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user(
                "Run exact e2e commands with run_test, not run_shell. "
                "Use python scripts/e2e_suite.py --model gemma4:e4b --scenarios scenario_run_test scenario_git_tools."
            )

        self.assertEqual(result.message, "done")
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertTrue(tool_calls)
        self.assertTrue(all(event["name"] == "run_test" for event in tool_calls))
        self.assertFalse(any(event["name"] == "run_shell" for event in tool_calls))

    def test_agent_reprompts_forbidden_run_shell_with_run_test_alternative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient(
                [
                    '{"type":"tool","name":"run_shell","arguments":{"command":"python scripts/e2e_suite.py --model gemma4:e4b --scenarios scenario_run_test scenario_git_tools"}}',
                    '{"type":"tool","name":"run_test","arguments":{"command":"python -c \\"print(\'e2e OK\')\\""}}',
                    '{"type":"final","message":"done"}',
                ]
            )
            tools = ToolExecutor(
                root,
                approval_mode="auto",
                disabled_tools=["run_shell"],
            )
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user(
                "Run exact e2e commands with run_test, not run_shell. "
                "Use python scripts/e2e_suite.py --model gemma4:e4b --scenarios scenario_run_test scenario_git_tools."
            )

        self.assertEqual(result.message, "done")
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertTrue(tool_calls)
        self.assertTrue(all(event["name"] == "run_test" for event in tool_calls))
        normalizations = [event for event in agent.events if event["type"] == "tool_normalized"]
        self.assertTrue(any(event["normalized_name"] == "run_test" for event in normalizations))

    def test_agent_forbids_non_available_tools_from_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient(
                [
                    '{"type":"tool","name":"run_shell","arguments":{"command":"python scripts/e2e_suite.py --model gemma4:e4b --scenarios scenario_run_test scenario_git_tools"}}',
                    '{"type":"tool","name":"run_test","arguments":{"command":"python -c \\"print(\'e2e OK\')\\""}}',
                    '{"type":"final","message":"done"}',
                ]
            )
            tools = ToolExecutor(
                root,
                approval_mode="auto",
                enabled_tools=["run_test", "read_file"],
            )
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user(
                "Run exact e2e commands with run_test, not run_shell. "
                "Use python scripts/e2e_suite.py --model gemma4:e4b --scenarios scenario_run_test scenario_git_tools."
            )

        self.assertEqual(result.message, "done")
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertTrue(tool_calls)
        self.assertTrue(all(event["name"] == "run_test" for event in tool_calls))

    def test_agent_normalizes_run_shell_to_run_test_for_explicit_benchmark_request(self) -> None:
        root = self._workspace_scratch()
        tools = ToolExecutor(
            root,
            approval_mode="auto",
            enabled_tools=["run_test", "read_file"],
        )
        client = FakeClient([])
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

        name, arguments, reason = agent._normalize_shell_test_call(
            "run_shell",
            {"command": "python -c \"print('bench OK')\""},
            request_text="Run one concrete coding benchmark case with run_test, not run_shell: `python -c \"print('bench OK')\"`.",
            exact_shell_command=None,
        )

        self.assertEqual(name, "run_test")
        self.assertEqual(arguments["command"], "python -c \"print('bench OK')\"")
        self.assertIn("explicitly requires run_test", reason)

    def test_agent_normalizes_run_shell_to_run_test_for_explicit_e2e_request(self) -> None:
        root = self._workspace_scratch()
        tools = ToolExecutor(
            root,
            approval_mode="auto",
            enabled_tools=["run_test", "read_file"],
        )
        client = FakeClient([])
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

        name, arguments, reason = agent._normalize_shell_test_call(
            "run_shell",
            {"command": "python -c \"print('e2e OK')\""},
            request_text="Run exact e2e commands with run_test, not run_shell: `python -c \"print('e2e OK')\"`.",
            exact_shell_command=None,
        )

        self.assertEqual(name, "run_test")
        self.assertEqual(arguments["command"], "python -c \"print('e2e OK')\"")
        self.assertIn("explicitly requires run_test", reason)

    def test_agent_normalizes_bare_python_test_file_shell_command_to_run_test(self) -> None:
        root = self._workspace_scratch()
        (root / "tests").mkdir()
        (root / "tests" / "test_sample.py").write_text(
            "def test_ok():\n    assert True\n",
            encoding="utf-8",
        )
        tools = ToolExecutor(root, approval_mode="auto", test_command='python -c "print(\'bare_test OK\')"')
        agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)

        name, arguments, reason = agent._normalize_shell_test_call(
            "run_shell",
            {"command": "tests/test_sample.py"},
            request_text="Validate the project and summarize failures.",
            exact_shell_command=None,
        )

        self.assertEqual(name, "run_test")
        self.assertEqual(arguments["command"], 'python -c "print(\'bare_test OK\')"')
        self.assertIn("bare Python test-file", reason)

    def test_agent_normalizes_bare_python_test_file_shell_command_to_pytest_without_config(self) -> None:
        root = self._workspace_scratch()
        (root / "tests").mkdir()
        (root / "tests" / "test_sample.py").write_text(
            "def test_ok():\n    assert True\n",
            encoding="utf-8",
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)

        name, arguments, reason = agent._normalize_shell_test_call(
            "run_shell",
            {"command": "tests/test_sample.py"},
            request_text="Validate the project and summarize failures.",
            exact_shell_command=None,
        )

        self.assertEqual(name, "run_test")
        self.assertEqual(arguments["command"], "python -m pytest tests/test_sample.py")
        self.assertIn("pytest", reason)

    def test_agent_preserves_exact_user_requested_bare_python_test_file_shell_command(self) -> None:
        root = self._workspace_scratch()
        (root / "tests").mkdir()
        (root / "tests" / "test_sample.py").write_text("print('shell only')\n", encoding="utf-8")
        tools = ToolExecutor(root, approval_mode="auto", test_command='python -c "print(\'bare_test OK\')"')
        agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)

        name, arguments, reason = agent._normalize_shell_test_call(
            "run_shell",
            {"command": "tests/test_sample.py"},
            request_text="Use run_shell to execute exactly: tests/test_sample.py.",
            exact_shell_command="tests/test_sample.py",
        )

        self.assertEqual(name, "run_shell")
        self.assertEqual(arguments, {"command": "tests/test_sample.py"})
        self.assertIsNone(reason)

    def test_shell_cat_inspection_normalizes_to_read_file(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"run_shell","arguments":{"command":"cat README.md"}}',
                '{"type":"final","message":"hello from docs"}',
            ]
        )
        root = self._workspace_scratch()
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=3)
        (root / "README.md").write_text("hello from docs\n", encoding="utf-8")

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Inspect README.md and summarize it.")

        self.assertEqual(result.message, "hello from docs")
        self.assertEqual(tools.execute_counts.get("read_file"), 1)
        self.assertIsNone(tools.execute_counts.get("run_shell"))
        normalized = [event for event in agent.events if event.get("type") == "tool_normalized"]
        self.assertEqual(normalized[0].get("original_name"), "run_shell")
        self.assertEqual(normalized[0].get("normalized_name"), "read_file")

    def test_shell_ls_inspection_normalizes_to_list_files(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"run_shell","arguments":{"command":"ls ."}}',
                '{"type":"final","message":"README.md exists"}',
            ]
        )
        root = self._workspace_scratch()
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=3)
        (root / "README.md").write_text("overview\n", encoding="utf-8")

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Inspect the project files and summarize them.")

        self.assertEqual(result.message, "README.md exists")
        self.assertEqual(tools.execute_counts.get("list_files"), 1)
        self.assertIsNone(tools.execute_counts.get("run_shell"))
        normalized = [event for event in agent.events if event.get("type") == "tool_normalized"]
        self.assertEqual(normalized[0].get("original_name"), "run_shell")
        self.assertEqual(normalized[0].get("normalized_name"), "list_files")

    def test_shell_grep_inspection_normalizes_to_search(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"run_shell","arguments":{"command":"grep \\"needle phrase\\" README.md"}}',
                '{"type":"final","message":"needle phrase is in README.md"}',
            ]
        )
        root = self._workspace_scratch()
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=3)
        (root / "README.md").write_text("alpha\nneedle phrase\n", encoding="utf-8")

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Inspect README.md for the needle phrase and summarize.")

        self.assertEqual(result.message, "needle phrase is in README.md")
        self.assertEqual(tools.execute_counts.get("search"), 1)
        self.assertIsNone(tools.execute_counts.get("run_shell"))
        normalized = [event for event in agent.events if event.get("type") == "tool_normalized"]
        self.assertEqual(normalized[0].get("original_name"), "run_shell")
        self.assertEqual(normalized[0].get("normalized_name"), "search")
        self.assertEqual(normalized[0].get("normalized_arguments"), {"query": "needle phrase", "path": "README.md"})

    def test_shell_grep_with_flags_does_not_normalize_to_search(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"run_shell","arguments":{"command":"rg -i needle README.md"}}',
                '{"type":"final","message":"searched with grep"}',
            ]
        )
        root = self._workspace_scratch()
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=3)
        (root / "README.md").write_text("needle\n", encoding="utf-8")

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Inspect the project with a shell-style lookup if useful and summarize.")

        self.assertEqual(result.message, "searched with grep")
        self.assertEqual(tools.execute_counts.get("run_shell"), 1)
        self.assertIsNone(tools.execute_counts.get("search"))
        self.assertFalse(any(event.get("type") == "tool_normalized" for event in agent.events))

    def test_shell_grep_line_number_inspection_normalizes_to_search(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"run_shell","arguments":{"command":"grep -n needle README.md"}}',
                '{"type":"final","message":"needle found"}',
            ]
        )
        root = self._workspace_scratch()
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=3)
        (root / "README.md").write_text("alpha\nneedle\n", encoding="utf-8")

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Inspect README.md for needle and summarize.")

        self.assertEqual(result.message, "needle found")
        self.assertEqual(tools.execute_counts.get("search"), 1)
        self.assertIsNone(tools.execute_counts.get("run_shell"))
        normalized = [event for event in agent.events if event.get("type") == "tool_normalized"]
        self.assertEqual(normalized[0].get("normalized_name"), "search")
        self.assertEqual(normalized[0].get("normalized_arguments"), {"query": "needle", "path": "README.md"})

    def test_shell_recursive_grep_inspection_normalizes_to_search(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"run_shell","arguments":{"command":"grep -r \\"THREADPOOL\\" azure_functions_worker"}}',
                '{"type":"final","message":"THREADPOOL is in constants.py"}',
            ]
        )
        root = self._workspace_scratch()
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=4)
        (root / "azure_functions_worker").mkdir()
        (root / "azure_functions_worker" / "constants.py").write_text("PYTHON_THREADPOOL_THREAD_COUNT = 1\n", encoding="utf-8")

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Inspect azure_functions_worker for THREADPOOL and summarize.")

        self.assertEqual(result.message, "THREADPOOL is in constants.py")
        self.assertEqual(tools.execute_counts.get("search"), 1)
        self.assertIsNone(tools.execute_counts.get("run_shell"))
        normalized = [event for event in agent.events if event.get("type") == "tool_normalized"]
        self.assertEqual(normalized[0].get("normalized_name"), "search")
        self.assertEqual(normalized[0].get("normalized_arguments"), {"query": "THREADPOOL", "path": "azure_functions_worker"})

    def test_shell_recursive_grep_with_unsupported_flags_does_not_normalize(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "src" / "app.py").write_text("needle\n", encoding="utf-8")
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)

        normalized_name, normalized_arguments, reason = agent._normalize_shell_inspection_call(
            "run_shell",
            {"command": "grep -ri needle src"},
            request_text="Inspect src with recursive ignore-case grep if useful and summarize.",
            exact_shell_command=None,
        )

        self.assertEqual(normalized_name, "run_shell")
        self.assertEqual(normalized_arguments, {"command": "grep -ri needle src"})
        self.assertIsNone(reason)

    def test_shell_head_inspection_normalizes_to_bounded_read_file(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"run_shell","arguments":{"command":"head -n 2 README.md"}}',
                '{"type":"final","message":"read the first two lines"}',
            ]
        )
        root = self._workspace_scratch()
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=3)
        (root / "README.md").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Preview README.md and summarize it.")

        self.assertEqual(result.message, "read the first two lines")
        self.assertEqual(tools.execute_counts.get("read_file"), 1)
        self.assertIsNone(tools.execute_counts.get("run_shell"))
        normalized = [event for event in agent.events if event.get("type") == "tool_normalized"]
        self.assertEqual(normalized[0].get("normalized_name"), "read_file")
        self.assertEqual(normalized[0].get("normalized_arguments"), {"path": "README.md", "start": 1, "end": 2})

    def test_shell_tail_inspection_normalizes_to_bounded_read_file(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"run_shell","arguments":{"command":"tail -2 README.md"}}',
                '{"type":"final","message":"read the last two lines"}',
            ]
        )
        root = self._workspace_scratch()
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=3)
        (root / "README.md").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Preview README.md and summarize it.")

        self.assertEqual(result.message, "read the last two lines")
        self.assertEqual(tools.execute_counts.get("read_file"), 1)
        self.assertIsNone(tools.execute_counts.get("run_shell"))
        normalized = [event for event in agent.events if event.get("type") == "tool_normalized"]
        self.assertEqual(normalized[0].get("normalized_name"), "read_file")
        self.assertEqual(normalized[0].get("normalized_arguments"), {"path": "README.md", "start": 3, "end": 4})

    def test_shell_head_with_unsupported_flags_does_not_normalize(self) -> None:
        root = self._workspace_scratch()
        (root / "README.md").write_text("needle\n", encoding="utf-8")
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)

        normalized_name, normalized_arguments, reason = agent._normalize_shell_inspection_call(
            "run_shell",
            {"command": "head -q README.md"},
            request_text="Inspect README.md and summarize it.",
            exact_shell_command=None,
        )

        self.assertEqual(normalized_name, "run_shell")
        self.assertEqual(normalized_arguments, {"command": "head -q README.md"})
        self.assertIsNone(reason)

    def test_shell_find_file_inspection_normalizes_to_file_search(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"run_shell","arguments":{"command":"find src -name \\"debug.py\\" -type f"}}',
                '{"type":"final","message":"debug.py exists"}',
            ]
        )
        root = self._workspace_scratch()
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=3)
        (root / "src").mkdir()
        (root / "src" / "debug.py").write_text("VALUE = 1\n", encoding="utf-8")

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Find the debug.py file and summarize it.")

        self.assertEqual(result.message, "debug.py exists")
        self.assertEqual(tools.execute_counts.get("file_search"), 1)
        self.assertIsNone(tools.execute_counts.get("run_shell"))
        normalized = [event for event in agent.events if event.get("type") == "tool_normalized"]
        self.assertEqual(normalized[0].get("normalized_name"), "file_search")
        self.assertEqual(normalized[0].get("normalized_arguments"), {"query": "debug.py", "path": "src", "limit": 100})

    def test_shell_find_directory_inspection_normalizes_to_directory_search(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"run_shell","arguments":{"command":"find tests -name \\"*mysql*\\" -type d"}}',
                '{"type":"final","message":"mysql test directory exists"}',
            ]
        )
        root = self._workspace_scratch()
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=3)
        (root / "tests" / "mysql").mkdir(parents=True)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Find matching test directories.")

        self.assertEqual(result.message, "mysql test directory exists")
        self.assertEqual(tools.execute_counts.get("directory_search"), 1)
        self.assertIsNone(tools.execute_counts.get("run_shell"))
        normalized = [event for event in agent.events if event.get("type") == "tool_normalized"]
        self.assertEqual(normalized[0].get("normalized_name"), "directory_search")
        self.assertEqual(normalized[0].get("normalized_arguments"), {"query": "*mysql*", "path": "tests", "limit": 100})

    def test_shell_find_exec_grep_inspection_normalizes_to_filtered_search(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"run_shell","arguments":{"command":"find . -name \\"*.py\\" -exec grep -l needle {} ;"}}',
                '{"type":"final","message":"needle found in Python files"}',
            ]
        )
        root = self._workspace_scratch()
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=3)
        (root / "app.py").write_text("needle\n", encoding="utf-8")
        (root / "README.md").write_text("needle\n", encoding="utf-8")

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Find Python files containing needle and summarize.")

        self.assertIn("app.py", result.message)
        self.assertEqual(tools.execute_counts.get("search"), 1)
        self.assertIsNone(tools.execute_counts.get("run_shell"))
        normalized = [event for event in agent.events if event.get("type") == "tool_normalized"]
        self.assertEqual(normalized[0].get("normalized_name"), "search")
        self.assertEqual(normalized[0].get("normalized_arguments"), {"query": "needle", "path": ".", "file_glob": "*.py"})

    def test_shell_find_exec_grep_with_type_file_normalizes_to_filtered_search(self) -> None:
        root = self._workspace_scratch()
        (root / "app.py").write_text("needle\n", encoding="utf-8")
        (root / "README.md").write_text("needle\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"run_shell","arguments":{"command":"find . -name \\"*.py\\" -type f -exec grep -l needle {} ;"}}',
                '{"type":"final","message":"needle found in Python files"}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=3)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Find Python files containing needle and summarize.")

        self.assertIn("app.py", result.message)
        self.assertEqual(tools.execute_counts.get("search"), 1)
        self.assertIsNone(tools.execute_counts.get("run_shell"))
        normalized = [event for event in agent.events if event.get("type") == "tool_normalized"]
        self.assertEqual(normalized[0].get("normalized_name"), "search")
        self.assertEqual(normalized[0].get("normalized_arguments"), {"query": "needle", "path": ".", "file_glob": "*.py"})

    def test_shell_find_exec_grep_with_unsupported_flags_does_not_normalize(self) -> None:
        root = self._workspace_scratch()
        (root / "app.py").write_text("needle\n", encoding="utf-8")
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)

        normalized_name, normalized_arguments, reason = agent._normalize_shell_inspection_call(
            "run_shell",
            {"command": 'find . -name "*.py" -exec grep -i needle {} ;'},
            request_text="Find Python files containing needle.",
            exact_shell_command=None,
        )

        self.assertEqual(normalized_name, "run_shell")
        self.assertEqual(normalized_arguments, {"command": 'find . -name "*.py" -exec grep -i needle {} ;'})
        self.assertIsNone(reason)

    def test_shell_find_dot_exec_grep_h_normalizes_to_filtered_search(self) -> None:
        root = self._workspace_scratch()
        (root / "app.py").write_text("def main():\n    return 0\n", encoding="utf-8")
        (root / "README.md").write_text("def main(): docs only\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"run_shell","arguments":{"command":"find. -name \\"*.py\\" -exec grep -H \\"def main(\\" {} ;"}}',
                '{"type":"final","message":"main is in app.py"}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=3)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Find Python files containing def main( and summarize.")

        self.assertIn("app.py", result.message)
        self.assertEqual(tools.execute_counts.get("search"), 1)
        self.assertIsNone(tools.execute_counts.get("run_shell"))
        normalized = [event for event in agent.events if event.get("type") == "tool_normalized"]
        self.assertEqual(normalized[0].get("normalized_name"), "search")
        self.assertEqual(normalized[0].get("normalized_arguments"), {"query": "def main(", "path": ".", "file_glob": "*.py"})

    def test_shell_find_dot_exec_grep_unsupported_flags_does_not_normalize(self) -> None:
        root = self._workspace_scratch()
        (root / "app.py").write_text("needle\n", encoding="utf-8")
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)

        normalized_name, normalized_arguments, reason = agent._normalize_shell_inspection_call(
            "run_shell",
            {"command": 'find. -name "*.py" -exec grep -i needle {} ;'},
            request_text="Find Python files containing needle.",
            exact_shell_command=None,
        )

        self.assertEqual(normalized_name, "run_shell")
        self.assertEqual(normalized_arguments, {"command": 'find. -name "*.py" -exec grep -i needle {} ;'})
        self.assertIsNone(reason)

    def test_tool_error_guard_auto_diagnoses_repeated_missing_dependency_failure(self) -> None:
        root = self._workspace_scratch()
        command = subprocess.list2cmdline([sys.executable, "-c", "import definitely_missing_package_12345"])
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": command}}),
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": command}}),
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": command}}),
                json.dumps({"type": "final", "message": "Dependency is missing."}),
                json.dumps({"type": "final", "message": "Dependency is missing."}),
                json.dumps({"type": "final", "message": "Dependency is missing."}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=5)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Check whether the helper command works, but do not loop on dependency failures.")

        self.assertIn("Dependency is missing.", result.message)
        tool_calls = [event for event in agent.events if event.get("type") == "tool_call" and event.get("name") == "run_shell"]
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tools.execute_counts.get("diagnose_dependency_error"), 1)
        guard_events = [event for event in agent.events if event.get("type") == "tool_error_guard"]
        self.assertEqual(len(guard_events), 1)
        self.assertEqual(guard_events[0].get("error_class"), "missing_dependency")
        controller_guards = [event for event in agent.events if event.get("type") == "controller_guard" and event.get("guard") == "dependency-or-import-guard"]
        self.assertEqual(len(controller_guards), 1)
        self.assertTrue(any("Use the diagnosis above to report the exact missing dependency/import" in message["content"] for message in agent.messages if message["role"] == "user"))

    def test_tool_error_guard_blocks_repeated_shell_syntax_failure(self) -> None:
        root = self._workspace_scratch()
        command = "if true; then echo hi"
        syntax_result = subprocess.CompletedProcess(
            args=["bash", "-n", "-c", command],
            returncode=2,
            stdout="",
            stderr="bash: -c: line 2: syntax error: unexpected end of file\n",
        )
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": command}}),
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": command}}),
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": command}}),
                json.dumps({"type": "final", "message": "Shell syntax is invalid."}),
                json.dumps({"type": "final", "message": "Shell syntax is invalid."}),
                json.dumps({"type": "final", "message": "Shell syntax is invalid."}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=5)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            with patch("ollama_code.tools.shutil.which", return_value="bash"):
                with patch("ollama_code.tools.subprocess.run", return_value=syntax_result):
                    result = agent.handle_user("Try the shell command if useful, but do not loop on shell syntax failures.")

        self.assertIn("Shell syntax is invalid.", result.message)
        tool_calls = [event for event in agent.events if event.get("type") == "tool_call" and event.get("name") == "run_shell"]
        self.assertEqual(len(tool_calls), 2)
        guard_events = [event for event in agent.events if event.get("type") == "tool_error_guard"]
        self.assertEqual(len(guard_events), 1)
        self.assertEqual(guard_events[0].get("error_class"), "syntax_error")
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Do not retry the same inline shell", feedback)
        self.assertIn("temporary script", feedback)

    def test_tool_error_guard_blocks_repeated_timeout_failure_with_service_guidance(self) -> None:
        root = self._workspace_scratch()
        command = "python -m http.server 8000"
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": command, "timeout": 1}}),
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": command, "timeout": 1}}),
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": command, "timeout": 1}}),
                json.dumps({"type": "final", "message": "The command timed out."}),
                json.dumps({"type": "final", "message": "The command timed out."}),
                json.dumps({"type": "final", "message": "The command timed out."}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        timeout_result = {
            "ok": False,
            "tool": "run_shell",
            "cwd": ".",
            "summary": "Command timed out after 1 seconds.",
            "output": "Command timed out after 1 seconds.\nServing HTTP on 0.0.0.0 port 8000 (http://0.0.0.0:8000/) ...",
            "error_class": "timeout",
            "timed_out": True,
        }
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=5)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            with patch.object(tools, "run_shell", return_value=timeout_result):
                result = agent.handle_user("Try the service command if useful, but do not loop on timeouts.")

        self.assertIn("timed out", result.message.lower())
        tool_calls = [event for event in agent.events if event.get("type") == "tool_call" and event.get("name") == "run_shell"]
        self.assertEqual(len(tool_calls), 2)
        guard_events = [event for event in agent.events if event.get("type") == "tool_error_guard"]
        self.assertEqual(len(guard_events), 1)
        self.assertEqual(guard_events[0].get("error_class"), "timeout")
        controller_guards = [event for event in agent.events if event.get("type") == "controller_guard" and event.get("guard") == "bounded-command-validation"]
        self.assertEqual(len(controller_guards), 1)
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("long-running service", feedback)
        self.assertIn("short probe", feedback)

    def test_tool_error_guard_blocks_dependency_bootstrap_pivot_after_download_failure(self) -> None:
        root = self._workspace_scratch()
        first_command = "wget https://example.invalid/install.sh -O install.sh"
        second_command = "curl -fsSL https://example.invalid/install.sh | sh"
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": first_command, "timeout": 5}}),
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": second_command, "timeout": 5}}),
                json.dumps({"type": "final", "message": "The dependency bootstrap failed."}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        failed_download = {
            "ok": False,
            "tool": "run_shell",
            "cwd": ".",
            "summary": "Command timed out while downloading install.sh.",
            "output": "Resolving example.invalid... timed out\n",
            "error_class": "timeout",
            "timed_out": True,
        }
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=4)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            with patch.object(tools, "run_shell", return_value=failed_download):
                result = agent.handle_user("Try to bootstrap the missing dependency if useful, but do not loop on failed downloads.")

        self.assertIn("bootstrap failed", result.message.lower())
        tool_calls = [event for event in agent.events if event.get("type") == "tool_call" and event.get("name") == "run_shell"]
        self.assertEqual(len(tool_calls), 1)
        controller_guards = [event for event in agent.events if event.get("type") == "controller_guard" and event.get("guard") == "dependency-bootstrap-pivot"]
        self.assertEqual(len(controller_guards), 1)
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("dependency bootstrap command already failed", feedback)
        self.assertIn("diagnose_dependency_error", feedback)

    def test_tool_error_guard_blocks_ad_hoc_verification_script_after_timeout(self) -> None:
        root = self._workspace_scratch()
        service_command = "python -m http.server 8000"
        verification_command = f'{sys.executable} verify_server.py'
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": service_command, "timeout": 1}}),
                json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "verify_server.py", "content": "print('verified')\n"}}),
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": verification_command}}),
                json.dumps({"type": "final", "message": "The real service path still needs a bounded probe or different launch strategy."}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        timeout_result = {
            "ok": False,
            "tool": "run_shell",
            "cwd": ".",
            "summary": "Command timed out after 1 seconds.",
            "output": "Command timed out after 1 seconds.\nServing HTTP on 0.0.0.0 port 8000 (http://0.0.0.0:8000/) ...",
            "error_class": "timeout",
            "timed_out": True,
        }
        verification_result = {
            "ok": True,
            "tool": "run_shell",
            "cwd": ".",
            "summary": "Verification helper printed verified.",
            "output": "verified\n",
        }
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=5)

        def fake_run_shell(command: str, cwd: str = ".", timeout: int = 30) -> dict[str, object]:
            if "http.server" in command:
                return timeout_result
            return verification_result

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            with patch.object(tools, "run_shell", side_effect=fake_run_shell):
                result = agent.handle_user("Start the service and verify it works, but do not loop on timeouts.")

        self.assertIn("bounded probe", result.message)
        self.assertEqual(tools.execute_counts.get("run_shell"), 1)
        controller_guards = [
            event
            for event in agent.events
            if event.get("type") == "controller_guard" and event.get("guard") == "fail-closed-timeout-verification"
        ]
        self.assertEqual(len(controller_guards), 1)
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("verification script", feedback)
        self.assertIn("original timed-out command path", feedback)

    def test_timeout_verification_guard_clears_after_successful_changed_strategy(self) -> None:
        root = self._workspace_scratch()
        service_command = "python -m http.server 8000"
        background_launch_command = f'{sys.executable} -c "print(\'server launched in background\')"'
        verification_command = f'{sys.executable} verify_server.py'
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": service_command, "timeout": 1}}),
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": background_launch_command}}),
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": verification_command}}),
                json.dumps({"type": "final", "message": "Background launch succeeded and verification passed."}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        timeout_result = {
            "ok": False,
            "tool": "run_shell",
            "cwd": ".",
            "summary": "Command timed out after 1 seconds.",
            "output": "Command timed out after 1 seconds.\nServing HTTP on 0.0.0.0 port 8000 (http://0.0.0.0:8000/) ...",
            "error_class": "timeout",
            "timed_out": True,
        }
        success_result = {
            "ok": True,
            "tool": "run_shell",
            "cwd": ".",
            "summary": "Background-safe launch succeeded.",
            "output": "server launched in background\n",
        }
        verification_result = {
            "ok": True,
            "tool": "run_shell",
            "cwd": ".",
            "summary": "Verification probe succeeded.",
            "output": "verified\n",
        }
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=6)

        def fake_run_shell(command: str, cwd: str = ".", timeout: int = 30) -> dict[str, object]:
            if "http.server" in command:
                return timeout_result
            if "server launched in background" in command:
                return success_result
            return verification_result

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            with patch.object(tools, "run_shell", side_effect=fake_run_shell):
                result = agent.handle_user("Start the service, switch to a safe launch strategy if needed, then verify it works.")

        self.assertTrue(result.completed)
        self.assertIn("verification passed", result.message.lower())
        self.assertEqual(tools.execute_counts.get("run_shell"), 3)
        controller_guards = [
            event
            for event in agent.events
            if event.get("type") == "controller_guard" and event.get("guard") == "fail-closed-timeout-verification"
        ]
        self.assertEqual(controller_guards, [])

    def test_timeout_verification_guard_allows_preexisting_probe_script_after_timeout(self) -> None:
        root = self._workspace_scratch()
        (root / "verify_server.py").write_text("print('verified')\n", encoding="utf-8")
        service_command = "python -m http.server 8000"
        verification_command = f'{sys.executable} verify_server.py'
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": service_command, "timeout": 1}}),
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": verification_command}}),
                json.dumps({"type": "final", "message": "The preexisting probe confirmed the service state."}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        timeout_result = {
            "ok": False,
            "tool": "run_shell",
            "cwd": ".",
            "summary": "Command timed out after 1 seconds.",
            "output": "Command timed out after 1 seconds.\nServing HTTP on 0.0.0.0 port 8000 (http://0.0.0.0:8000/) ...",
            "error_class": "timeout",
            "timed_out": True,
        }
        verification_result = {
            "ok": True,
            "tool": "run_shell",
            "cwd": ".",
            "summary": "Preexisting verification probe succeeded.",
            "output": "verified\n",
        }
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=5)

        def fake_run_shell(command: str, cwd: str = ".", timeout: int = 30) -> dict[str, object]:
            if "http.server" in command:
                return timeout_result
            return verification_result

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            with patch.object(tools, "run_shell", side_effect=fake_run_shell):
                result = agent.handle_user("Start the service, then use the existing verify_server.py probe to check it.")

        self.assertTrue(result.completed)
        self.assertIn("preexisting probe", result.message.lower())
        self.assertEqual(tools.execute_counts.get("run_shell"), 2)
        controller_guards = [
            event
            for event in agent.events
            if event.get("type") == "controller_guard" and event.get("guard") == "fail-closed-timeout-verification"
        ]
        self.assertEqual(controller_guards, [])

    def test_final_success_claim_is_blocked_after_timeout_until_summary_is_accurate(self) -> None:
        root = self._workspace_scratch()
        service_command = "python -m http.server 8000"
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": service_command, "timeout": 1}}),
                json.dumps({"type": "final", "message": "The service is working and verification passed."}),
                json.dumps({"type": "final", "message": "The command timed out; the service still needs a bounded probe or different launch strategy."}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        timeout_result = {
            "ok": False,
            "tool": "run_shell",
            "cwd": ".",
            "summary": "Command timed out after 1 seconds.",
            "output": "Command timed out after 1 seconds.\nServing HTTP on 0.0.0.0 port 8000 (http://0.0.0.0:8000/) ...",
            "error_class": "timeout",
            "timed_out": True,
        }
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=4)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            with patch.object(tools, "run_shell", return_value=timeout_result):
                result = agent.handle_user("Start the service and verify it works.")

        self.assertTrue(result.completed)
        self.assertIn("timed out", result.message.lower())
        self.assertIn("bounded probe", result.message.lower())
        controller_guards = [
            event
            for event in agent.events
            if event.get("type") == "controller_guard" and event.get("guard") == "timeout-final-claim"
        ]
        self.assertEqual(len(controller_guards), 1)
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("latest run_shell timed out", feedback)
        self.assertIn("do not claim the command works", feedback)

    def test_final_success_claim_is_blocked_after_command_not_found_until_summary_is_accurate(self) -> None:
        root = self._workspace_scratch()
        command = "go version"
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": command}}),
                json.dumps({"type": "final", "message": "The command works and verification passed."}),
                json.dumps({"type": "final", "message": "The command failed because go is not installed in this environment."}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        failure_result = {
            "ok": False,
            "tool": "run_shell",
            "cwd": ".",
            "summary": "go: command not found",
            "output": "/usr/bin/bash: line 1: go: command not found\n",
            "error_class": "command_not_found",
        }
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", max_tool_rounds=4)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            with patch.object(tools, "run_shell", return_value=failure_result):
                result = agent.handle_user("Run `go version` and tell me whether it works.")

        self.assertTrue(result.completed)
        self.assertIn("not installed", result.message.lower())
        controller_guards = [
            event
            for event in agent.events
            if event.get("type") == "controller_guard" and event.get("guard") == "run-shell-final-claim"
        ]
        self.assertEqual(len(controller_guards), 1)
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("latest run_shell failed", feedback)
        self.assertIn("do not claim the command works", feedback)

    def test_accurate_failure_summary_is_allowed_after_command_not_found(self) -> None:
        root = self._workspace_scratch()
        command = "go version"
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": command}}),
                json.dumps({"type": "final", "message": "The command failed because go is not installed in this environment."}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        failure_result = {
            "ok": False,
            "tool": "run_shell",
            "cwd": ".",
            "summary": "go: command not found",
            "output": "/usr/bin/bash: line 1: go: command not found\n",
            "error_class": "command_not_found",
        }
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=3)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            with patch.object(tools, "run_shell", return_value=failure_result):
                result = agent.handle_user("Run `go version` and tell me whether it works.")

        self.assertTrue(result.completed)
        self.assertIn("not installed", result.message.lower())
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertNotIn("This request requires real tool use in this turn.", feedback)

    def test_command_validation_event_records_rejected_common_command(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient(
            [
                '{"type":"tool","name":"run_shell","arguments":{"command":"git reset --hard"}}',
                '{"type":"final","message":"Command rejected."}',
                '{"type":"final","message":"Command rejected."}',
                '{"type":"final","message":"Command rejected."}',
            ]
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=4)

        result = agent.handle_user("Try a shell command if useful and summarize.")

        self.assertFalse(result.completed)
        validation_events = [event for event in agent.events if event.get("type") == "command_validation"]
        self.assertEqual(len(validation_events), 1)
        self.assertFalse(validation_events[0].get("valid"))
        self.assertEqual(validation_events[0].get("family"), "git")
