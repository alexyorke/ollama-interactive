import sys
from unittest.mock import patch

from ollama_code.agent import OllamaCodeAgent
from ollama_code.tools import ToolExecutor
from tests.agent_test_support import AgentTestBase, FakeClient


class AgentSubagentTests(AgentTestBase):
    def test_agent_can_run_subagent(self) -> None:
        root = self._workspace_scratch()
        (root / "note.txt").write_text("helper data\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"run_agent","arguments":{"prompt":"Read note.txt and summarize it.","approval_mode":"read-only"}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                '{"type":"final","message":"helper saw helper data"}',
                '{"type":"final","message":"parent got: helper saw helper data"}',
            ]
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
        result = agent.handle_user("delegate this")

        self.assertEqual(result.message, "parent got: helper saw helper data")
        self.assertEqual(agent.events[1]["name"], "run_agent")
        self.assertEqual(agent.events[2]["result"]["tool"], "run_agent")

    def test_subagent_inherits_parent_tool_configuration(self) -> None:
        root = self._workspace_scratch()
        (root / "note.txt").write_text("helper data\n", encoding="utf-8")
        marker = root / "marker.txt"
        code = "from pathlib import Path; Path('marker.txt').write_text('ok\\n')"
        command = f'"{sys.executable}" -c "{code}"'
        client = FakeClient(
            [
                '{"type":"tool","name":"run_test","arguments":{}}',
                '{"type":"final","message":"child done"}',
            ]
        )
        tools = ToolExecutor(root, approval_mode="auto", test_command=command, disabled_tools=["read_file"])
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

        result = agent._run_sub_agent({"prompt": "Read note.txt and run tests.", "approval_mode": "auto", "max_tool_rounds": "2"})

        self.assertTrue(result["ok"], result)
        self.assertEqual(marker.read_text(encoding="utf-8"), "ok\n")
        system_prompt = client.calls[0]["messages"][0]["content"]
        self.assertNotIn("read_file(path", str(system_prompt))

    def test_subagent_inherits_parent_enabled_tool_allowlist(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient(['{"type":"final","message":"child done"}'])
        tools = ToolExecutor(
            root,
            approval_mode="auto",
            enabled_tools=["run_test", "read_file"],
        )
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
        captured: dict[str, object] = {}

        class SpyToolExecutor(ToolExecutor):
            def __init__(self, *args: object, **kwargs: object) -> None:
                captured["enabled_tools"] = kwargs.get("enabled_tools")
                super().__init__(*args, **kwargs)

        with patch("ollama_code.agent.ToolExecutor", SpyToolExecutor):
            result = agent._run_sub_agent({"prompt": "Say ok.", "approval_mode": "read-only"})

        self.assertTrue(result["ok"], result)
        self.assertEqual(set(captured["enabled_tools"]), {"run_test", "read_file"})

    def test_agent_normalizes_near_match_subagent_model(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient(
            [
                '{"type":"tool","name":"run_agent","arguments":{"prompt":"Say ok.","model":"granite4.1:8b-q8_O","approval_mode":"read-only"}}',
                '{"type":"final","message":"ok"}',
                '{"type":"final","message":"parent got ok"}',
            ],
            models=["fake-model", "granite4.1:8b-q8_0"],
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
        result = agent.handle_user("Use run_agent with the requested model.")

        self.assertEqual(result.message, "parent got ok")
        tool_result = next(event["result"] for event in agent.events if event.get("type") == "tool_result" and event.get("name") == "run_agent")
        self.assertTrue(tool_result["ok"])
        self.assertEqual(tool_result["model"], "granite4.1:8b-q8_0")
        self.assertIn("Normalized sub-agent model", tool_result["model_note"])
        self.assertEqual(client.calls[1]["model"], "granite4.1:8b-q8_0")

    def test_agent_skips_assumption_audit_for_explicit_subagent(self) -> None:
        root = self._workspace_scratch()
        (root / "note.txt").write_text("helper data\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"run_agent","arguments":{"prompt":"Read note.txt and summarize it.","approval_mode":"read-only"}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                '{"type":"final","message":"helper saw helper data"}',
                '{"type":"final","message":"parent got: helper saw helper data"}',
            ]
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")
        result = agent.handle_user("Use run_agent to delegate this in read-only mode.")

        self.assertEqual(result.message, "parent got: helper saw helper data")
        audits = [event for event in agent.events if event["type"] == "assumption_audit"]
        self.assertEqual(len(audits), 0)
