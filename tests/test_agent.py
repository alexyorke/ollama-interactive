from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from ollama_code.agent import OllamaCodeAgent, _workspace_roots_match
from ollama_code.ollama_client import ChatResponse
from ollama_code.tools import ToolExecutor


class FakeClient:
    def __init__(self, responses: list[str], *, script_debate: bool = False) -> None:
        self.responses = list(responses)
        self.script_debate = script_debate
        self.calls: list[dict[str, object]] = []
        self.interrupt_events: list[object | None] = []

    def set_interrupt_event(self, event: object | None) -> None:
        self.interrupt_events.append(event)

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: str = "json",
        on_thinking: object | None = None,
    ) -> ChatResponse:
        system_prompt = messages[0]["content"] if messages else ""
        self.calls.append(
            {
                "model": model,
                "messages": list(messages),
                "response_format": response_format,
                "on_thinking": on_thinking,
            }
        )
        if not self.script_debate and isinstance(system_prompt, str):
            if system_prompt.startswith("You are Against Agent."):
                return ChatResponse(content="Candidate risk low.", model=model, raw={})
            if system_prompt.startswith("You are For Agent."):
                return ChatResponse(content="Candidate fits request.", model=model, raw={})
            if system_prompt.startswith("You are Judge Agent for a coding CLI controller."):
                payload = json.loads(messages[-1]["content"])
                candidate = str(payload.get("candidate_response", "")).strip()
                return ChatResponse(content=candidate, model=model, raw={})
        return ChatResponse(content=self.responses.pop(0), model=model, raw={})


class AgentTests(unittest.TestCase):
    def _workspace_scratch(self) -> Path:
        root = (Path.cwd() / "verify_scratch" / f"test-agent-{uuid4().hex}").resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return root

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
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
            result = agent.handle_user("Summarize note.txt")

        self.assertEqual(result.message, "The file says hello world.")
        self.assertEqual(result.rounds, 2)
        self.assertEqual(agent.events[1]["type"], "tool_call")
        self.assertEqual(agent.events[2]["type"], "tool_result")

    def test_system_prompt_requires_assumption_checking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        prompt = agent.messages[0]["content"]
        self.assertIn("Question your assumptions before acting.", prompt)
        self.assertIn("prove or disprove it with the available tools", prompt)
        self.assertIn("Do not guess about workspace contents", prompt)

    def test_system_prompt_enables_caveman_lite_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        prompt = agent.messages[0]["content"]
        self.assertIn("Default reply style: caveman-lite.", prompt)
        self.assertIn("keep all technical terms, code, file paths, commands, errors, and JSON exact", prompt)
        self.assertIn("Do not let terse style reduce investigation depth", prompt)
        self.assertIn("Tool arguments, JSON wrappers, code, diffs, and commands must stay syntactically correct and complete.", prompt)

    def test_agent_stops_after_max_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient(['{"type":"tool","name":"list_files","arguments":{}}'] * 2)
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", max_tool_rounds=1, debate_enabled=False)
            result = agent.handle_user("loop forever")

        self.assertIn("maximum tool rounds", result.message)

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

    def test_agent_runs_debate_by_default_and_uses_judge_result(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient(
            [
                '{"type":"final","message":"base answer"}',
                "Candidate ignores a stronger direct answer.",
                "Candidate is acceptable but could be clearer.",
                '{"type":"final","message":"judged answer"}',
            ],
            script_debate=True,
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        result = agent.handle_user("Say something brief.")

        self.assertEqual(result.message, "judged answer")
        self.assertEqual(len(client.calls), 4)
        debate_events = [event for event in agent.events if event["type"] == "debate"]
        self.assertEqual(len(debate_events), 1)
        self.assertTrue(debate_events[0]["applied"])

    def test_agent_debate_can_be_disabled(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient(['{"type":"final","message":"base answer"}'])
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

        result = agent.handle_user("Say something brief.")

        self.assertEqual(result.message, "base answer")
        self.assertEqual(len(client.calls), 1)

    def test_agent_keeps_primary_candidate_when_judge_returns_invalid_json(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient(
            [
                '{"type":"final","message":"base answer"}',
                "Skeptic note.",
                "Defender note.",
                "not json",
            ],
            script_debate=True,
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        result = agent.handle_user("Say something brief.")

        self.assertEqual(result.message, "base answer")
        debate_events = [event for event in agent.events if event["type"] == "debate"]
        self.assertEqual(len(debate_events), 1)
        self.assertFalse(debate_events[0]["applied"])

    def test_agent_skips_debate_for_tool_candidates(self) -> None:
        root = self._workspace_scratch()
        (root / "note.txt").write_text("hello\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                '{"type":"final","message":"done"}',
                "Skeptic note.",
                "Defender note.",
                '{"type":"final","message":"done"}',
            ],
            script_debate=True,
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        result = agent.handle_user("Use read_file on note.txt and then tell me when you are done.")

        self.assertEqual(result.message, "done")
        self.assertEqual(agent.events[2]["name"], "read_file")
        self.assertEqual(len(client.calls), 5)
        debate_events = [event for event in agent.events if event["type"] == "debate"]
        self.assertEqual(len(debate_events), 1)
        self.assertTrue(debate_events[0]["applied"])

    def test_agent_requires_explicitly_named_tool_before_final_answer(self) -> None:
        root = self._workspace_scratch()
        (root / "note.txt").write_text("hello\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"list_files","arguments":{}}',
                '{"type":"final","message":"done"}',
                '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                '{"type":"final","message":"done"}',
            ]
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

        result = agent.handle_user("Use read_file on note.txt and then tell me when you are done.")

        self.assertEqual(result.message, "done")
        self.assertEqual(agent.events[1]["name"], "list_files")
        self.assertEqual(agent.events[3]["name"], "read_file")

    def test_agent_allows_final_answer_after_requested_tool_failure_details(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"../outside.txt"}}',
                '{"type":"final","message":"Path escapes the workspace: ../outside.txt"}',
            ]
        )
        tools = ToolExecutor(Path.cwd(), approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

        result = agent.handle_user("Use read_file on ../outside.txt and tell me the exact tool error.")

        self.assertIn("escapes the workspace", result.message)
        self.assertEqual(len(client.calls), 2)

    def test_agent_retries_after_unverified_file_mutation_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.txt").write_text("hello\n", encoding="utf-8")
            client = FakeClient(
                [
                    '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                    '{"type":"final","message":"note.txt has been updated."}',
                    '{"type":"final","message":"note.txt line 1 is hello"}',
                ]
            )
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Use read_file on note.txt and tell me what line 1 says.")

        self.assertEqual(result.message, "note.txt line 1 is hello")
        self.assertEqual(len(client.calls), 3)

    def test_agent_allows_file_mutation_claim_after_write_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient(
                [
                    '{"type":"tool","name":"write_file","arguments":{"path":"note.txt","content":"changed\\n"}}',
                    '{"type":"final","message":"note.txt has been updated."}',
                ]
            )
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Create note.txt with changed on line 1.")
            self.assertEqual((root / "note.txt").read_text(encoding="utf-8"), "changed\n")

        self.assertEqual(result.message, "note.txt has been updated.")

    def test_agent_retries_git_diff_with_working_tree_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init"], cwd=root, capture_output=True, text=True, check=True)
            subprocess.run(["git", "config", "user.name", "Tests"], cwd=root, capture_output=True, text=True, check=True)
            subprocess.run(["git", "config", "user.email", "tests@example.com"], cwd=root, capture_output=True, text=True, check=True)
            (root / "app.py").write_text("def answer() -> int:\n    return 42\n", encoding="utf-8")
            subprocess.run(["git", "add", "app.py"], cwd=root, capture_output=True, text=True, check=True)
            subprocess.run(["git", "commit", "-m", "initial"], cwd=root, capture_output=True, text=True, check=True)
            (root / "app.py").write_text("def answer() -> int:\n    return 99\n", encoding="utf-8")
            client = FakeClient(
                [
                    '{"type":"tool","name":"git_diff","arguments":{"path":"app.py","cached":true}}',
                    '{"type":"tool","name":"git_diff","arguments":{"path":"app.py"}}',
                    '{"type":"final","message":"app.py diff adds return 99"}',
                ]
            )
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Use git_diff on app.py for the working tree only and tell me whether it adds return 99.")

        self.assertEqual(result.message, "app.py diff adds return 99")
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "git_diff")
        self.assertNotIn("cached", tool_calls[0]["arguments"])

    def test_agent_blocks_subagent_approval_escalation(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient(
            [
                '{"type":"tool","name":"run_agent","arguments":{"prompt":"Write a file.","approval_mode":"auto"}}',
                '{"type":"final","message":"blocked"}',
            ]
        )
        tools = ToolExecutor(root, approval_mode="ask")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
        result = agent.handle_user("Delegate this carefully.")

        self.assertEqual(result.message, "blocked")
        self.assertEqual(agent.events[1]["name"], "run_agent")
        self.assertFalse(agent.events[2]["result"]["ok"])
        self.assertIn("more permissive", agent.events[2]["result"]["summary"])

    def test_agent_allows_subagent_to_be_more_restrictive(self) -> None:
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
        tools = ToolExecutor(root, approval_mode="ask")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
        result = agent.handle_user("delegate this")

        self.assertEqual(result.message, "parent got: helper saw helper data")
        self.assertTrue(agent.events[2]["result"]["ok"])
        self.assertEqual(agent.events[2]["result"]["approval_mode"], "read-only")

    def test_agent_retries_after_failed_subagent_does_not_count_as_real_tool_use(self) -> None:
        root = self._workspace_scratch()
        (root / "note.txt").write_text("helper data\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"run_agent","arguments":{"prompt":"Read note.txt and summarize it.","approval_mode":"read-only","max_tool_rounds":1}}',
                '{"type":"final","message":"helper saw helper data"}',
                '{"type":"final","message":"parent got: helper saw helper data"}',
                '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                '{"type":"final","message":"parent read helper data"}',
            ]
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
        result = agent.handle_user("Delegate reading note.txt, then tell me what it says.")

        self.assertEqual(result.message, "parent read helper data")
        self.assertFalse(agent.events[2]["result"]["ok"])
        self.assertIn("Sub-agent failed", agent.events[2]["result"]["summary"])
        self.assertIn("maximum tool rounds", agent.events[2]["result"]["summary"])
        self.assertEqual(agent.events[3]["name"], "read_file")
        self.assertEqual(len(client.calls), 5)

    def test_agent_normalizes_tool_name_in_type_field(self) -> None:
        root = self._workspace_scratch()
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
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
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
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
            result = agent.handle_user("Read note.txt and tell me what it says.")

        self.assertEqual(result.message, "retry me")
        self.assertEqual(agent.events[1]["type"], "tool_call")

    def test_agent_retries_when_git_tools_are_required(self) -> None:
        client = FakeClient(
            [
                '{"type":"final","message":"repo status checked"}',
                '{"type":"tool","name":"git_status","arguments":{}}',
                '{"type":"final","message":"repo status checked"}',
            ]
        )
        tools = ToolExecutor(Path.cwd(), approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
        result = agent.handle_user("Tell me the git status of this workspace.")

        self.assertEqual(result.message, "repo status checked")
        self.assertEqual(agent.events[1]["type"], "tool_call")
        self.assertEqual(agent.events[1]["name"], "git_status")

    def test_agent_retries_after_unknown_tool_does_not_count_as_real_tool_use(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"bogus_tool","arguments":{}}',
                '{"type":"tool","name":"list_files","arguments":{}}',
                '{"type":"final","message":"listed workspace"}',
            ]
        )
        tools = ToolExecutor(Path.cwd(), approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

        result = agent.handle_user("List files in the workspace.")

        self.assertEqual(result.message, "listed workspace")
        self.assertEqual(agent.events[1]["name"], "bogus_tool")
        self.assertEqual(agent.events[2]["result"]["summary"], "Unknown tool: bogus_tool")
        self.assertEqual(agent.events[3]["name"], "list_files")

    def test_agent_retries_after_bad_tool_arguments_do_not_count_as_real_tool_use(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"start":1}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"README.md"}}',
                '{"type":"final","message":"readme loaded"}',
            ]
        )
        tools = ToolExecutor(Path.cwd(), approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

        result = agent.handle_user("Read README.md and summarize it.")

        self.assertEqual(result.message, "readme loaded")
        self.assertIn("Bad arguments for read_file", agent.events[2]["result"]["summary"])
        self.assertEqual(agent.events[3]["name"], "read_file")

    def test_agent_retries_after_tool_failure_does_not_count_as_real_tool_use(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"../secret.txt"}}',
                '{"type":"final","message":"secret loaded"}',
                '{"type":"tool","name":"read_file","arguments":{"path":"README.md"}}',
                '{"type":"final","message":"readme loaded"}',
            ]
        )
        tools = ToolExecutor(Path.cwd(), approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

        result = agent.handle_user("Read README.md and summarize it.")

        self.assertEqual(result.message, "readme loaded")
        self.assertIn("escapes the workspace", agent.events[2]["result"]["summary"])
        self.assertEqual(agent.events[3]["name"], "read_file")
        self.assertEqual(len(client.calls), 4)

    def test_agent_retries_after_approval_denial_does_not_count_as_real_tool_use(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"run_shell","arguments":{"command":"cat README.md"}}',
                '{"type":"final","message":"README loaded from shell"}',
                '{"type":"tool","name":"read_file","arguments":{"path":"README.md"}}',
                '{"type":"final","message":"README loaded from file"}',
            ]
        )
        tools = ToolExecutor(Path.cwd(), approval_mode="read-only")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

        result = agent.handle_user("Read README.md and summarize it.")

        self.assertEqual(result.message, "README loaded from file")
        self.assertIn("denied because approval mode is read-only", agent.events[2]["result"]["summary"])
        self.assertEqual(agent.events[3]["name"], "read_file")
        self.assertEqual(len(client.calls), 4)

    def test_agent_retries_after_empty_model_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient(
                [
                    "",
                    '{"type":"final","message":"done"}',
                ]
            )
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
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
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
            result = agent.handle_user("Say done.")

        self.assertEqual(result.message, "done")
        self.assertEqual(len(client.calls), 2)

    def test_agent_prefers_last_agent_payload_in_mixed_model_output(self) -> None:
        client = FakeClient(
            [
                'Context {"foo":1}\n{"type":"final","message":"done"}',
            ]
        )
        tools = ToolExecutor(Path.cwd(), approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
        result = agent.handle_user("Say done.")

        self.assertEqual(result.message, "done")
        self.assertEqual(result.rounds, 1)
        self.assertEqual(len(client.calls), 1)

    def test_relative_transcript_paths_use_workspace_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", session_file="scratch/session.json", debate_enabled=False)
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
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", session_file="scratch/current.json", debate_enabled=False)
            loaded = agent.load_session(session)

        self.assertEqual(loaded, session.resolve())
        self.assertEqual(agent.session_path(), session.resolve())
        self.assertEqual(agent.model, "fake-model")
        self.assertEqual(agent.approval_mode(), "auto")
        self.assertEqual(agent.messages[1]["content"], "remember TOKEN_42")
        self.assertEqual(agent.events[0]["content"], "remember TOKEN_42")

    def test_agent_load_session_restores_runtime_settings_without_rewriting_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            session = root / ".ollama-code" / "sessions" / "saved.json"
            session.parent.mkdir(parents=True)
            original = (
                '{"model":"saved-model","approval_mode":"read-only","workspace_root":"'
                + root.as_posix()
                + '","messages":[{"role":"system","content":"sys"},{"role":"user","content":"restore me"}],"events":[{"type":"user","content":"restore me"}]}'
            )
            session.write_text(original, encoding="utf-8")
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="current-model", session_file="scratch/current.json", debate_enabled=False)

            loaded = agent.load_session(session)
            on_disk = session.read_text(encoding="utf-8")

        self.assertEqual(loaded, session.resolve())
        self.assertEqual(agent.model, "saved-model")
        self.assertEqual(agent.approval_mode(), "read-only")
        self.assertEqual(on_disk, original)

    def test_workspace_roots_match_accepts_wsl_alias_for_windows_path(self) -> None:
        if Path.cwd().drive:
            saved_root = "/mnt/c/Users/yorke/OneDrive/Desktop/ollama-interactive"
            current_root = Path("C:/Users/yorke/OneDrive/Desktop/ollama-interactive")
        else:
            saved_root = "C:/Users/yorke/OneDrive/Desktop/ollama-interactive"
            current_root = Path("/mnt/c/Users/yorke/OneDrive/Desktop/ollama-interactive")
        self.assertTrue(_workspace_roots_match(saved_root, current_root))

    def test_workspace_roots_match_rejects_different_workspace(self) -> None:
        current_root = Path(__file__).resolve().parents[1]
        self.assertFalse(_workspace_roots_match(str(current_root.parent / "other-workspace"), current_root))

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
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
            result = agent.handle_user("Create scratch/file.txt with hi and a newline.")

        self.assertEqual(result.message, "done")
        self.assertEqual(agent.events[1]["name"], "write_file")
        self.assertFalse(any(event.get("name") == "run_shell" for event in agent.events if event["type"] == "tool_call"))
