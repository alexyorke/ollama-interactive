from ollama_code.agent import OllamaCodeAgent
from ollama_code.tools import ToolExecutor
from tests.agent_test_support import AgentTestBase, FakeClient


class AgentVerificationTests(AgentTestBase):
    def test_agent_runs_verification_by_default_and_accepts_candidate(self) -> None:
        root = self._workspace_scratch()
        (root / "note.txt").write_text("hello\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                '{"type":"final","message":"line 1 is hello"}',
                '{"verdict":"accept"}',
            ],
            script_verification=True,
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        result = agent.handle_user("Use read_file on note.txt and tell me line 1.")

        self.assertEqual(result.message, "line 1 is hello")
        self.assertEqual(len(client.calls), 3)
        self.assertFalse(client.calls[0]["think"])
        self.assertFalse(client.calls[1]["think"])
        self.assertFalse(client.calls[2]["think"])
        assumption_audits = [event for event in agent.events if event["type"] == "assumption_audit"]
        self.assertEqual(len(assumption_audits), 0)
        verification_events = [event for event in agent.events if event["type"] == "verification"]
        self.assertEqual(len(verification_events), 1)
        self.assertEqual(verification_events[0]["verdict"], "accept")

    def test_agent_skips_verification_for_low_risk_final(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient(['{"type":"final","message":"base answer"}'], script_verification=True)
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        result = agent.handle_user("Say something brief.")

        self.assertEqual(result.message, "base answer")
        self.assertEqual(len(client.calls), 1)
        self.assertIsNone(client.calls[0]["think"])
        self.assertFalse(any(event["type"] == "verification" for event in agent.events))

    def test_agent_verification_can_be_disabled(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient(['{"type":"final","message":"base answer"}'])
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

        result = agent.handle_user("Say something brief.")

        self.assertEqual(result.message, "base answer")
        self.assertEqual(len(client.calls), 1)
