from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ollama_code.agent import OllamaCodeAgent
from ollama_code.ollama_client import ChatResponse
from ollama_code.tools import ToolExecutor


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def chat(self, *, model: str, messages: list[dict[str, str]], response_format: str = "json") -> ChatResponse:
        self.calls.append({"model": model, "messages": list(messages), "response_format": response_format})
        return ChatResponse(content=self.responses.pop(0), model=model, raw={})


class AgentTests(unittest.TestCase):
    def test_agent_runs_tool_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.txt").write_text("hello world\n", encoding="utf-8")
            client = FakeClient(
                [
                    '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                    '{"type":"final","message":"The file says hello world."}',
                ]
            )
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")
            result = agent.handle_user("Summarize note.txt")

        self.assertEqual(result.message, "The file says hello world.")
        self.assertEqual(result.rounds, 2)
        self.assertEqual(agent.events[1]["type"], "tool_call")
        self.assertEqual(agent.events[2]["type"], "tool_result")

    def test_agent_stops_after_max_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient(['{"type":"tool","name":"list_files","arguments":{}}'] * 2)
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", max_tool_rounds=1)
            result = agent.handle_user("loop forever")

        self.assertIn("maximum tool rounds", result.message)

    def test_agent_can_run_subagent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
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
            result = agent.handle_user("delegate this")

        self.assertEqual(result.message, "parent got: helper saw helper data")
        self.assertEqual(agent.events[1]["name"], "run_agent")
        self.assertEqual(agent.events[2]["result"]["tool"], "run_agent")

    def test_agent_normalizes_tool_name_in_type_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.txt").write_text("helper data\n", encoding="utf-8")
            client = FakeClient(
                [
                    '{"type":"run_agent","arguments":{"prompt":"Read note.txt and summarize it.","approval_mode":"read-only"}}',
                    '{"type":"read_file","arguments":{"path":"note.txt"}}',
                    '{"type":"final","message":"helper saw helper data"}',
                    '{"type":"final","message":"parent got: helper saw helper data"}',
                ]
            )
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")
            result = agent.handle_user("delegate this")

        self.assertEqual(result.message, "parent got: helper saw helper data")
        self.assertEqual(agent.events[1]["type"], "tool_call")
        self.assertEqual(agent.events[1]["name"], "run_agent")

    def test_agent_retries_when_tools_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.txt").write_text("retry me\n", encoding="utf-8")
            client = FakeClient(
                [
                    '{"type":"final","message":"retry me"}',
                    '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                    '{"type":"final","message":"retry me"}',
                ]
            )
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")
            result = agent.handle_user("Read note.txt and tell me what it says.")

        self.assertEqual(result.message, "retry me")
        self.assertEqual(agent.events[1]["type"], "tool_call")

    def test_agent_retries_when_git_tools_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient(
                [
                    '{"type":"final","message":"repo is clean"}',
                    '{"type":"tool","name":"git_status","arguments":{}}',
                    '{"type":"final","message":"repo is clean"}',
                ]
            )
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")
            result = agent.handle_user("Tell me the git status of this workspace.")

        self.assertEqual(result.message, "repo is clean")
        self.assertEqual(agent.events[1]["type"], "tool_call")
        self.assertEqual(agent.events[1]["name"], "git_status")

    def test_agent_retries_after_empty_model_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient(
                [
                    "",
                    '{"type":"final","message":"done"}',
                ]
            )
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")
            result = agent.handle_user("Say done.")

        self.assertEqual(result.message, "done")
        self.assertEqual(len(client.calls), 2)

    def test_agent_retries_after_non_json_model_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient(
                [
                    "</html>",
                    '{"type":"final","message":"done"}',
                ]
            )
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")
            result = agent.handle_user("Say done.")

        self.assertEqual(result.message, "done")
        self.assertEqual(len(client.calls), 2)

    def test_relative_transcript_paths_use_workspace_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", session_file="scratch/session.json")
            saved = agent.save_transcript("scratch/manual.json")

        self.assertEqual(agent.session_file, (root / "scratch" / "session.json").resolve())
        self.assertEqual(saved, (root / "scratch" / "manual.json").resolve())

    def test_agent_can_load_saved_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            session = root / ".ollama-code" / "sessions" / "saved.json"
            session.parent.mkdir(parents=True)
            session.write_text(
                '{"model":"fake-model","approval_mode":"auto","workspace_root":"'
                + root.as_posix()
                + '","messages":[{"role":"system","content":"sys"},{"role":"user","content":"remember TOKEN_42"}],"events":[{"type":"user","content":"remember TOKEN_42"}]}',
                encoding="utf-8",
            )
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="ask")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", session_file="scratch/current.json")
            loaded = agent.load_session(session)

        self.assertEqual(loaded, session.resolve())
        self.assertEqual(agent.session_path(), session.resolve())
        self.assertEqual(agent.messages[1]["content"], "remember TOKEN_42")
        self.assertEqual(agent.events[0]["content"], "remember TOKEN_42")

    def test_agent_rejects_shell_mutation_when_file_tools_fit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient(
                [
                    '{"type":"tool","name":"run_shell","arguments":{"command":"echo hi > scratch/file.txt"}}',
                    '{"type":"tool","name":"write_file","arguments":{"path":"scratch/file.txt","content":"hi\\n"}}',
                    '{"type":"final","message":"done"}',
                ]
            )
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")
            result = agent.handle_user("Create scratch/file.txt with hi and a newline.")

        self.assertEqual(result.message, "done")
        self.assertEqual(agent.events[1]["name"], "write_file")
        self.assertFalse(any(event.get("name") == "run_shell" for event in agent.events if event["type"] == "tool_call"))
