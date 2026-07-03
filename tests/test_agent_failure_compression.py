import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ollama_code.agent import OllamaCodeAgent
from ollama_code.features import ENV_OLLAMA_CODE_FEATURE_PROFILE
from tests.agent_test_support import AgentTestBase, CountingToolExecutor, FakeClient


class AgentFailureCompressionTests(AgentTestBase):
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
