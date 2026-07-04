import tempfile
import unittest
from pathlib import Path

from ollama_code.agent import OllamaCodeAgent, extract_json_response
from ollama_code.ollama_client import ChatResponse, TokenUsage
from ollama_code.tools import ToolExecutor
from tests.agent_test_support import AgentTestBase, FakeClient


class AgentCoreLoopTests(AgentTestBase):
    def test_agent_runs_tool_loop(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                '{"type":"final","message":"The file says hello world."}',
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")
            (root / "note.txt").write_text("hello world\n", encoding="utf-8")
            result = agent.handle_user("Summarize note.txt")

        self.assertEqual(result.message, "The file says hello world.")
        self.assertEqual(result.rounds, 2)
        self.assertTrue(any(event.get("type") == "tool_call" and event.get("name") == "read_file" for event in agent.events))
        tool_result = next(event for event in agent.events if event.get("type") == "tool_result" and event.get("name") == "read_file")
        self.assertIsInstance(tool_result.get("duration_ms"), float)

    def test_agent_normalizes_tool_named_final_into_final_answer(self) -> None:
        client = FakeClient(['{"type":"tool","name":"final","arguments":{"message":"done"}}'])
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, disable_spec_guided_repair=True)
            result = agent.handle_user("Say done.")

        self.assertEqual(result.message, "done")
        self.assertFalse(any(event.get("type") == "tool_call" and event.get("name") == "final" for event in agent.events))

    def test_agent_records_llm_call_usage_events(self) -> None:
        class UsageClient(FakeClient):
            def chat(self, **kwargs: object) -> ChatResponse:
                response = super().chat(**kwargs)  # type: ignore[arg-type]
                return ChatResponse(
                    content=response.content,
                    model=response.model,
                    raw=response.raw,
                    thinking=response.thinking,
                    usage=TokenUsage(prompt_tokens=10, output_tokens=2, total_tokens=12, total_duration_ns=100),
                )

        client = UsageClient(['{"type":"final","message":"done"}'])
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
            result = agent.handle_user("Say done.")

        self.assertEqual(result.message, "done")
        llm_calls = [event for event in agent.events if event["type"] == "llm_call"]
        self.assertEqual(len(llm_calls), 1)
        self.assertEqual(llm_calls[0]["purpose"], "primary")
        self.assertEqual(llm_calls[0]["prompt_tokens"], 10)
        self.assertEqual(llm_calls[0]["output_tokens"], 2)
        self.assertEqual(llm_calls[0]["total_tokens"], 12)
        self.assertGreater(llm_calls[0]["message_count"], 0)
        self.assertIn("system", llm_calls[0]["prompt_chars_by_role"])
        self.assertGreater(llm_calls[0]["top_prompt_messages"][0]["chars"], 0)
        telemetry_types = [event["type"] for event in agent.llm_telemetry_events]
        self.assertIn("llm_call_started", telemetry_types)

    def test_extract_json_response_repairs_truncated_tool_object(self) -> None:
        payload = extract_json_response(
            '{"type":"tool","name":"edit_intent","arguments":{"path":"docs/pricing.md","intent":"replace_text","target":"total(prices)","replacement":"cart_total(prices)"}'
        )

        self.assertEqual(
            payload,
            {
                "type": "tool",
                "name": "edit_intent",
                "arguments": {
                    "path": "docs/pricing.md",
                    "intent": "replace_text",
                    "target": "total(prices)",
                    "replacement": "cart_total(prices)",
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
