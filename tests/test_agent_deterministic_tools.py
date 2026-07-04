import subprocess
import tempfile
from pathlib import Path

from ollama_code.agent import OllamaCodeAgent
from ollama_code.tools import ToolExecutor
from tests.agent_test_support import AgentTestBase, FakeClient


class AgentDeterministicToolTests(AgentTestBase):
    def _cwd_agent(self, client: FakeClient | None = None, **kwargs: object) -> OllamaCodeAgent:
        resolved_client = client if client is not None else FakeClient([])
        return OllamaCodeAgent(
            client=resolved_client,
            tools=ToolExecutor(self._workspace_scratch(), approval_mode="auto"),
            model="fake-model",
            **kwargs,
        )

    def test_agent_deterministically_handles_git_diff_without_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_git_repo_or_skip(root)
            (root / "app.py").write_text("def value():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "app.py"], cwd=root, capture_output=True, text=True, check=True)
            subprocess.run(["git", "commit", "-m", "base"], cwd=root, capture_output=True, text=True, check=True)
            (root / "app.py").write_text("def value():\n    return 2\n", encoding="utf-8")
            client = FakeClient(['{"type":"final","message":"model diff"}'])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Show git diff app.py.")

        self.assertIn("+    return 2", result.message)
        self.assertEqual(len(client.calls), 0)
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_calls[0]["name"], "git_diff")

    def test_agent_require_llm_for_turn_bypasses_deterministic_git_diff_shortcut(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_git_repo_or_skip(root)
            (root / "app.py").write_text("def value():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "app.py"], cwd=root, capture_output=True, text=True, check=True)
            subprocess.run(["git", "commit", "-m", "base"], cwd=root, capture_output=True, text=True, check=True)
            (root / "app.py").write_text("def value():\n    return 2\n", encoding="utf-8")
            client = FakeClient(['{"type":"tool","name":"git_diff","arguments":{"path":"app.py"}}'])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(
                client=client,
                tools=tools,
                model="fake-model",
                debate_enabled=False,
                require_llm_for_turn=True,
            )

            result = agent.handle_user("Show git diff app.py.")

        self.assertIn("+    return 2", result.message)
        self.assertEqual(len(client.calls), 1)
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_calls[0]["name"], "git_diff")

    def test_agent_retries_after_unknown_tool_does_not_count_as_real_tool_use(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"bogus_tool","arguments":{}}',
                '{"type":"tool","name":"list_files","arguments":{}}',
                '{"type":"final","message":"listed workspace"}',
            ]
        )
        agent = self._cwd_agent(client, debate_enabled=False)

        result = agent.handle_user("Inspect the workspace with a tool.")

        self.assertEqual(result.message, "listed workspace")
        tool_results = [event for event in agent.events if event["type"] == "tool_result"]
        self.assertEqual(tool_results[0]["name"], "bogus_tool")
        self.assertEqual(tool_results[0]["result"]["summary"], "Unknown tool: bogus_tool")
        self.assertEqual(tool_results[1]["name"], "list_files")

    def test_agent_rejects_forbidden_tool_and_retries(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"README.md"}}',
                '{"type":"tool","name":"list_files","arguments":{}}',
                '{"type":"final","message":"listed workspace"}',
            ]
        )
        agent = self._cwd_agent(client, debate_enabled=False)

        result = agent.handle_user("List files in the workspace. Do not use read_file.")

        self.assertTrue(result.message)
        self.assertEqual(len(client.calls), 0)
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "list_files")

    def test_agent_rejects_mutating_tool_for_read_only_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.txt").write_text("hello\n", encoding="utf-8")
            client = FakeClient(
                [
                    '{"type":"tool","name":"replace_in_file","arguments":{"path":"note.txt","old":"hello","new":"goodbye"}}',
                    '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                    '{"type":"final","message":"line 1 is hello"}',
                ]
            )
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Read note.txt and tell me what line 1 says.")
            final_content = (root / "note.txt").read_text(encoding="utf-8")

        self.assertEqual(result.message, "line 1 is hello")
        self.assertEqual(final_content, "hello\n")
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "read_file")

    def test_agent_rejects_tools_for_session_memory_question(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"list_files","arguments":{}}',
                '{"type":"final","message":"CONTINUE_TOKEN_99"}',
            ]
        )
        agent = self._cwd_agent(client, debate_enabled=False)
        agent.messages.append({"role": "user", "content": "Remember the exact token CONTINUE_TOKEN_99 for this session."})

        result = agent.handle_user("What token did I ask you to remember earlier in this session? Reply with the token only.")

        self.assertEqual(result.message, "CONTINUE_TOKEN_99")
        self.assertFalse(any(event["type"] == "tool_call" for event in agent.events))

    def test_agent_skips_verification_for_session_memory_question(self) -> None:
        client = FakeClient(['{"type":"final","message":"CONTINUE_TOKEN_99"}'], script_verification=True)
        agent = self._cwd_agent(client)
        agent.messages.append({"role": "user", "content": "Remember the exact token CONTINUE_TOKEN_99 for this session."})

        result = agent.handle_user("What token did I ask you to remember earlier in this session? Reply with the token only.")

        self.assertEqual(result.message, "CONTINUE_TOKEN_99")
        self.assertEqual(len(client.calls), 1)
        self.assertFalse(any(event["type"] == "verification" for event in agent.events))
