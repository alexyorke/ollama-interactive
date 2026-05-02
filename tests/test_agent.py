from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from ollama_code.agent import OllamaCodeAgent, _workspace_roots_match
from ollama_code.ollama_client import ChatResponse, TokenUsage
from ollama_code.tools import ToolExecutor


class FakeClient:
    def __init__(
        self,
        responses: list[str],
        *,
        script_verification: bool = False,
        script_assumption_audit: bool = False,
        script_final_rewrite: bool = False,
    ) -> None:
        self.responses = list(responses)
        self.script_verification = script_verification
        self.script_assumption_audit = script_assumption_audit
        self.script_final_rewrite = script_final_rewrite
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
        think: bool | None = None,
    ) -> ChatResponse:
        system_prompt = messages[0]["content"] if messages else ""
        self.calls.append(
            {
                "model": model,
                "messages": list(messages),
                "response_format": response_format,
                "on_thinking": on_thinking,
                "think": think,
            }
        )
        if isinstance(system_prompt, str):
            if system_prompt.startswith("You are a grounded final verifier") and not self.script_verification:
                return ChatResponse(content='{"verdict":"accept"}', model=model, raw={})
            if system_prompt.startswith("You are a tool-step assumption auditor") and not self.script_assumption_audit:
                return ChatResponse(
                    content='{"verdict":"accept","reason":"","assumptions":["tool is needed"],"validation_steps":["run the tool to gather evidence"],"required_tools":[],"forbidden_tools":[]}',
                    model=model,
                    raw={},
                )
            if system_prompt.startswith("You are an evidence-backed final rewriter") and not self.script_final_rewrite:
                payload = json.loads(messages[1]["content"])
                return ChatResponse(
                    content=json.dumps({"type": "final", "message": payload.get("candidate_final_answer", "")}),
                    model=model,
                    raw={},
                )
        return ChatResponse(content=self.responses.pop(0), model=model, raw={})


class CountingToolExecutor(ToolExecutor):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.execute_counts: dict[str, int] = {}

    def execute(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        self.execute_counts[name] = self.execute_counts.get(name, 0) + 1
        return super().execute(name, arguments)


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
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")
            result = agent.handle_user("Summarize note.txt")

        self.assertEqual(result.message, "The file says hello world.")
        self.assertEqual(result.rounds, 2)
        self.assertEqual(agent.events[1]["type"], "tool_call")
        self.assertEqual(agent.events[2]["type"], "tool_result")

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

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = UsageClient(['{"type":"final","message":"done"}'])
            tools = ToolExecutor(root, approval_mode="auto")
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

    def test_system_prompt_requires_assumption_checking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

        prompt = agent.messages[0]["content"]
        self.assertIn("Question your assumptions before acting.", prompt)
        self.assertIn("prove or disprove it with the available tools", prompt)
        self.assertIn("Do not guess about workspace contents", prompt)
        self.assertIn("prefer search_symbols, code_outline, then read_symbol before broad read_file", prompt)

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

    def test_agent_retries_after_assumption_audit_rejects_tool_and_recovers(self) -> None:
        root = self._workspace_scratch()
        (root / "note.txt").write_text("hello\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"list_files","arguments":{}}',
                '{"verdict":"retry","reason":"Listing files does not validate line 1.","assumptions":["You need file contents."],"validation_steps":["Read note.txt directly."],"required_tools":["read_file"],"forbidden_tools":["list_files"]}',
                '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                '{"verdict":"accept","reason":"","assumptions":["note.txt exists"],"validation_steps":["read_file will show line 1"],"required_tools":[],"forbidden_tools":[]}',
                '{"type":"final","message":"line 1 is hello"}',
            ],
            script_assumption_audit=True,
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        result = agent.handle_user("Read note.txt and tell me what it says.")

        self.assertEqual(result.message, "line 1 is hello")
        audits = [event for event in agent.events if event["type"] == "assumption_audit"]
        self.assertEqual([event["verdict"] for event in audits], ["retry", "accept"])
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual([event["name"] for event in tool_calls], ["read_file"])

    def test_agent_fails_closed_after_assumption_audit_retry_cap(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient(
            [
                '{"type":"tool","name":"list_files","arguments":{}}',
                '{"verdict":"retry","reason":"Need direct evidence.","assumptions":["list_files is enough"],"validation_steps":["Read the file instead."],"required_tools":["read_file"],"forbidden_tools":["list_files"]}',
                '{"type":"tool","name":"search","arguments":{"query":"hello","path":"."}}',
                '{"verdict":"retry","reason":"Search is still indirect.","assumptions":["Search proves line 1."],"validation_steps":["Read the file directly."],"required_tools":["read_file"],"forbidden_tools":[]}',
                '{"type":"tool","name":"git_status","arguments":{}}',
                '{"verdict":"retry","reason":"Git status does not answer the question.","assumptions":["Repo state helps."],"validation_steps":["Use read_file."],"required_tools":["read_file"],"forbidden_tools":[]}',
            ],
            script_assumption_audit=True,
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        result = agent.handle_user("Read README.md and tell me line 1.")

        self.assertFalse(result.completed)
        self.assertIn("assumption audit could not approve", result.message)
        audits = [event for event in agent.events if event["type"] == "assumption_audit"]
        self.assertEqual(len(audits), 3)
        self.assertTrue(all(event["verdict"] == "retry" for event in audits))

    def test_agent_skips_assumption_audit_for_cached_read_only_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.txt").write_text("hello world\n", encoding="utf-8")
            client = FakeClient(
                [
                    '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                    '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                    '{"type":"final","message":"done"}',
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")
            result = agent.handle_user("Read note.txt twice, then say done.")

        self.assertEqual(result.message, "done")
        audits = [event for event in agent.events if event["type"] == "assumption_audit"]
        self.assertEqual(len(audits), 0)

    def test_final_verifier_receives_evidence_without_low_risk_audit(self) -> None:
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
        verifier_payload = json.loads(str(client.calls[-1]["messages"][1]["content"]))
        self.assertEqual(len(verifier_payload["accepted_assumption_audits"]), 0)
        self.assertEqual(verifier_payload["candidate_claims"], ["line 1 is hello"])
        self.assertEqual(len(verifier_payload["evidence_table"]), 1)
        self.assertEqual(verifier_payload["evidence_table"][0]["tool"], "read_file")

    def test_agent_rewrites_from_evidence_after_verifier_retry(self) -> None:
        root = self._workspace_scratch()
        (root / "note.txt").write_text("hello\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                '{"type":"final","message":"line 1 is goodbye"}',
                '{"verdict":"retry","reason":"Tool result says hello, not goodbye.","required_tools":[],"forbidden_tools":[],"claim_checks":[{"claim":"line 1 is goodbye","status":"contradicted","evidence":"E1","correction":"line 1 is hello"}],"rewrite_guidance":["Use the verified file contents."],"rewrite_from_evidence":true}',
                '{"type":"final","message":"line 1 is hello"}',
                '{"verdict":"accept","claim_checks":[{"claim":"line 1 is hello","status":"supported","evidence":"E1"}]}',
            ],
            script_verification=True,
            script_final_rewrite=True,
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        result = agent.handle_user("Read note.txt and tell me what line 1 says.")

        self.assertTrue(result.completed)
        self.assertEqual(result.message, "line 1 is hello")
        rewrite_events = [event for event in agent.events if event["type"] == "verification_rewrite"]
        self.assertEqual(len(rewrite_events), 1)
        self.assertEqual(rewrite_events[0]["verdict"], "accept")
        self.assertEqual(client.calls[3]["messages"][0]["content"].splitlines()[0], "You are an evidence-backed final rewriter for a coding CLI controller.")

    def test_agent_uses_verifier_model_for_verification_and_rewrite(self) -> None:
        root = self._workspace_scratch()
        (root / "note.txt").write_text("hello\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                '{"type":"final","message":"line 1 is goodbye"}',
                '{"verdict":"retry","reason":"Tool result says hello, not goodbye.","claim_checks":[{"claim":"line 1 is goodbye","status":"contradicted","evidence":"E1","correction":"line 1 is hello"}],"rewrite_guidance":["Use the verified file contents."],"rewrite_from_evidence":true}',
                '{"type":"final","message":"line 1 is hello"}',
                '{"verdict":"accept"}',
            ],
            script_verification=True,
            script_final_rewrite=True,
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="base-model", verifier_model="judge-model")

        result = agent.handle_user("Read note.txt and tell me what line 1 says.")

        self.assertEqual(result.message, "line 1 is hello")
        self.assertEqual(client.calls[0]["model"], "base-model")
        self.assertEqual(client.calls[1]["model"], "base-model")
        self.assertEqual(client.calls[2]["model"], "judge-model")
        self.assertEqual(client.calls[3]["model"], "judge-model")
        self.assertEqual(client.calls[4]["model"], "judge-model")

    def test_agent_retries_after_verifier_rejects_candidate_and_recovers(self) -> None:
        root = self._workspace_scratch()
        (root / "note.txt").write_text("hello\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                '{"type":"final","message":"line 1 is goodbye"}',
                '{"verdict":"retry","reason":"Tool result says hello, not goodbye.","required_tools":["read_file"],"forbidden_tools":[]}',
                '{"type":"final","message":"line 1 is hello"}',
                '{"verdict":"accept"}',
            ],
            script_verification=True,
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        result = agent.handle_user("Read note.txt and tell me what line 1 says.")

        self.assertEqual(result.message, "line 1 is hello")
        assumption_audits = [event for event in agent.events if event["type"] == "assumption_audit"]
        self.assertEqual(len(assumption_audits), 0)
        verification_events = [event for event in agent.events if event["type"] == "verification"]
        self.assertEqual(len(verification_events), 2)
        self.assertEqual(verification_events[0]["verdict"], "retry")
        self.assertEqual(verification_events[1]["verdict"], "accept")
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_calls[0]["name"], "read_file")
        self.assertFalse(client.calls[0]["think"])
        self.assertFalse(client.calls[1]["think"])
        self.assertFalse(client.calls[2]["think"])
        self.assertFalse(client.calls[3]["think"])
        self.assertFalse(client.calls[4]["think"])

    def test_agent_fails_closed_after_verification_retry_cap(self) -> None:
        root = self._workspace_scratch()
        (root / "note.txt").write_text("hello\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                '{"type":"final","message":"answer one"}',
                '{"verdict":"retry","reason":"Need grounded answer.","required_tools":["read_file"],"forbidden_tools":[]}',
                '{"type":"final","message":"answer two"}',
                '{"verdict":"retry","reason":"Still not grounded.","required_tools":["read_file"],"forbidden_tools":[]}',
                '{"type":"final","message":"answer three"}',
                '{"verdict":"retry","reason":"Still not grounded.","required_tools":["read_file"],"forbidden_tools":[]}',
            ],
            script_verification=True,
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        result = agent.handle_user("Use read_file on note.txt and answer carefully.")

        self.assertFalse(result.completed)
        self.assertIn("grounded final verification", result.message)
        assumption_audits = [event for event in agent.events if event["type"] == "assumption_audit"]
        self.assertEqual(len(assumption_audits), 0)
        verification_events = [event for event in agent.events if event["type"] == "verification"]
        self.assertEqual(len(verification_events), 3)

    def test_agent_fails_closed_when_model_repeats_rejected_final(self) -> None:
        root = self._workspace_scratch()
        (root / "note.txt").write_text("hello\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                '{"type":"final","message":"line 1 is goodbye"}',
                '{"verdict":"retry","reason":"Tool result says hello, not goodbye.","required_tools":["read_file"],"forbidden_tools":[]}',
                '{"type":"final","message":"line 1 is goodbye"}',
            ],
            script_verification=True,
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        result = agent.handle_user("Read note.txt and tell me what line 1 says.")

        self.assertFalse(result.completed)
        self.assertIn("grounded final verification", result.message)
        assumption_audits = [event for event in agent.events if event["type"] == "assumption_audit"]
        self.assertEqual(len(assumption_audits), 0)
        verification_events = [event for event in agent.events if event["type"] == "verification"]
        self.assertEqual(len(verification_events), 1)

    def test_agent_skips_verification_for_tool_candidates(self) -> None:
        root = self._workspace_scratch()
        (root / "note.txt").write_text("hello\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                '{"type":"final","message":"done"}',
                '{"verdict":"accept"}',
            ],
            script_verification=True,
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        result = agent.handle_user("Use read_file on note.txt and then tell me when you are done.")

        self.assertEqual(result.message, "done")
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_calls[0]["name"], "read_file")
        self.assertEqual(len(client.calls), 3)
        assumption_audits = [event for event in agent.events if event["type"] == "assumption_audit"]
        self.assertEqual(len(assumption_audits), 0)
        verification_events = [event for event in agent.events if event["type"] == "verification"]
        self.assertEqual(len(verification_events), 1)
        self.assertEqual(verification_events[0]["verdict"], "accept")

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
        self.assertEqual(len(client.calls), 0)
        self.assertTrue(any(event["type"] == "assistant_synthesized" for event in agent.events))

    def test_agent_short_circuits_exact_tool_error_with_debate_enabled(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"../outside.txt"}}',
            ]
        )
        tools = ToolExecutor(Path.cwd(), approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        result = agent.handle_user("Use read_file on ../outside.txt and tell me the exact tool error.")

        self.assertIn("escapes the workspace", result.message)
        self.assertEqual(len(client.calls), 0)
        assumption_audits = [event for event in agent.events if event["type"] == "assumption_audit"]
        self.assertEqual(len(assumption_audits), 0)

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

    def test_agent_rejects_forbidden_tool_and_retries(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"README.md"}}',
                '{"type":"tool","name":"list_files","arguments":{}}',
                '{"type":"final","message":"listed workspace"}',
            ]
        )
        tools = ToolExecutor(Path.cwd(), approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

        result = agent.handle_user("List files in the workspace. Do not use read_file.")

        self.assertEqual(result.message, "listed workspace")
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
        tools = ToolExecutor(Path.cwd(), approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
        agent.messages.append({"role": "user", "content": "Remember the exact token CONTINUE_TOKEN_99 for this session."})

        result = agent.handle_user("What token did I ask you to remember earlier in this session? Reply with the token only.")

        self.assertEqual(result.message, "CONTINUE_TOKEN_99")
        self.assertFalse(any(event["type"] == "tool_call" for event in agent.events))

    def test_agent_skips_verification_for_session_memory_question(self) -> None:
        client = FakeClient(['{"type":"final","message":"CONTINUE_TOKEN_99"}'], script_verification=True)
        tools = ToolExecutor(Path.cwd(), approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")
        agent.messages.append({"role": "user", "content": "Remember the exact token CONTINUE_TOKEN_99 for this session."})

        result = agent.handle_user("What token did I ask you to remember earlier in this session? Reply with the token only.")

        self.assertEqual(result.message, "CONTINUE_TOKEN_99")
        self.assertEqual(len(client.calls), 1)
        self.assertFalse(any(event["type"] == "verification" for event in agent.events))

    def test_agent_requires_exact_readback_match_before_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient(
                [
                    '{"type":"tool","name":"write_file","arguments":{"path":"note.txt","content":"APROVED\\n"}}',
                    '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                    '{"type":"final","message":"APROVED"}',
                ]
            )
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user(
                "Create note.txt with exactly the single line APPROVED followed by a newline. Then use read_file to confirm it and reply with APPROVED only."
            )
            final_content = (root / "note.txt").read_text(encoding="utf-8")

        self.assertEqual(result.message, "APPROVED")
        self.assertEqual(final_content, "APPROVED\n")
        self.assertEqual(len(client.calls), 0)
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual([event["name"] for event in tool_calls], ["write_file", "read_file"])
        self.assertEqual(tool_calls[0]["arguments"], {"path": "note.txt", "content": "APPROVED\n"})
        self.assertEqual(tool_calls[1]["arguments"], {"path": "note.txt", "start": 1, "end": 1})
        assistant_synthesized = [event for event in agent.events if event["type"] == "assistant_synthesized"]
        self.assertEqual(len(assistant_synthesized), 1)
        self.assertEqual(assistant_synthesized[0]["content"], "APPROVED")
        self.assertFalse(any(event["type"] == "assumption_audit" for event in agent.events))

    def test_agent_synthesizes_exact_token_reply_after_read_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "docs" / "guide.md").write_text("TOKEN_42 lives here.\n", encoding="utf-8")
            client = FakeClient(['{"type":"tool","name":"read_file","arguments":{"path":"docs/guide.md"}}'])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

            result = agent.handle_user("Use read_file on docs/guide.md and reply with the uppercase token only.")

        self.assertEqual(result.message, "TOKEN_42")
        self.assertEqual(len(client.calls), 0)
        self.assertTrue(any(event["type"] == "assistant_synthesized" for event in agent.events))

    def test_agent_normalizes_target_line_read_and_synthesizes_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            lines = [f"line {index}: filler" for index in range(1, 501)]
            lines[249] = "line 250: NEEDLE_FAST_250"
            (root / "docs" / "large.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
            client = FakeClient(['{"type":"tool","name":"read_file","arguments":{"path":"docs/large.md"}}'])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user(
                "Use read_file on docs/large.md with the smallest useful line range around line 250, then reply with the exact marker token on that line only."
            )

        self.assertEqual(result.message, "NEEDLE_FAST_250")
        self.assertEqual(len(client.calls), 0)
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_calls[0]["arguments"], {"path": "docs/large.md", "start": 245, "end": 255})
        self.assertFalse(any(event["type"] == "tool_normalized" for event in agent.events))

    def test_agent_normalizes_exact_shell_command_and_synthesizes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient(
                [
                    '{"type":"tool","name":"run_shell","arguments":{"command":"python -c \\"print(1)\\""}}',
                ]
            )
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
            exact_command = 'python -c "import sys; print(\'boom\'); sys.exit(5)"'

            result = agent.handle_user(
                f"Use run_shell to execute exactly: {exact_command}. Then tell me the exit code and the printed word."
            )

        self.assertIn("Exit code: 5", result.message)
        self.assertIn("boom", result.message)
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_calls[0]["arguments"]["command"], exact_command)
        self.assertEqual(len(client.calls), 0)
        normalized = [event for event in agent.events if event["type"] == "tool_normalized"]
        self.assertEqual(len(normalized), 0)

    def test_agent_audits_shell_tool_under_debate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient(
                [
                    '{"type":"tool","name":"run_shell","arguments":{"command":"python -c \\"print(123)\\""}}',
                    '{"type":"final","message":"done"}',
                    '{"verdict":"accept"}',
                ]
            )
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

            result = agent.handle_user('Use run_shell to execute python -c "print(123)" and then say done.')

        self.assertEqual(result.message, "done")
        audits = [event for event in agent.events if event["type"] == "assumption_audit"]
        self.assertEqual(len(audits), 1)
        self.assertEqual(audits[0]["tool"], "run_shell")

    def test_agent_recovers_exact_shell_command_after_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient(["not json"])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")
            exact_command = 'python -c "import sys; print(\'boom\'); sys.exit(5)"'

            result = agent.handle_user(
                f"Use run_shell to execute exactly: {exact_command}. Then tell me the exit code and the printed word."
            )

        self.assertIn("Exit code: 5", result.message)
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_calls[0]["arguments"]["command"], exact_command)

    def test_agent_normalizes_vague_run_test_to_configured_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient(['{"type":"tool","name":"run_test","arguments":{"command":"test"}}'])
            tools = ToolExecutor(root, approval_mode="auto", test_command='python -c "print(\'test_sample OK\')"')
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

            result = agent.handle_user("Use run_test and tell me whether tests passed and which test module ran.")

        self.assertIn("Tests passed: yes", result.message)
        self.assertIn("test_sample", result.message)
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_calls[0]["arguments"]["command"], 'python -c "print(\'test_sample OK\')"')

    def test_agent_audits_mutating_tool_under_debate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.txt").write_text("old\n", encoding="utf-8")
            client = FakeClient(
                [
                    '{"type":"tool","name":"replace_in_file","arguments":{"path":"note.txt","old":"old","new":"new"}}',
                    '{"type":"final","message":"updated"}',
                ]
            )
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

            result = agent.handle_user("Update note.txt by replacing old with new.")

        self.assertEqual(result.message, "updated")
        audits = [event for event in agent.events if event["type"] == "assumption_audit"]
        self.assertEqual(len(audits), 1)
        self.assertEqual(audits[0]["tool"], "replace_in_file")

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

    def test_agent_caches_repeated_read_only_tool_calls_within_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.txt").write_text("hello world\n", encoding="utf-8")
            client = FakeClient(
                [
                    '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                    '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                    '{"type":"final","message":"done"}',
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
            result = agent.handle_user("Read note.txt twice, then say done.")

        self.assertEqual(result.message, "done")
        self.assertEqual(tools.execute_counts.get("read_file"), 1)
        tool_results = [event for event in agent.events if event["type"] == "tool_result"]
        self.assertEqual(len(tool_results), 2)
        self.assertFalse(tool_results[0].get("cached", False))
        self.assertTrue(tool_results[1].get("cached", False))

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
        tool_feedback = next(message["content"] for message in agent.messages if message["role"] == "user" and message["content"].startswith("Tool result summary:\n"))
        self.assertIn("... truncated ...", tool_feedback)
        self.assertNotIn(" 200 |", tool_feedback)

    def test_agent_compacts_primary_context_without_dropping_current_request(self) -> None:
        client = FakeClient(['{"type":"final","message":"done"}'])
        tools = ToolExecutor(Path.cwd(), approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
        for index in range(30):
            agent.messages.append(
                {
                    "role": "user" if index % 2 == 0 else "assistant",
                    "content": f"OLD_CONTEXT_{index:02d} " + ("x" * 5000),
                }
            )

        result = agent.handle_user("Say done.")

        self.assertEqual(result.message, "done")
        sent_messages = client.calls[0]["messages"]
        self.assertLessEqual(len(sent_messages), 16)
        sent_text = "\n".join(str(message["content"]) for message in sent_messages)
        full_text = "\n".join(message["content"] for message in agent.messages)
        self.assertIn("Earlier conversation omitted", sent_text)
        self.assertIn("Say done.", sent_text)
        self.assertNotIn("OLD_CONTEXT_00", sent_text)
        self.assertLess(len(sent_text), len(full_text) // 3)

    def test_agent_keeps_full_context_for_session_memory_requests(self) -> None:
        client = FakeClient(['{"type":"final","message":"MEMORY_TOKEN_77"}'], script_verification=True)
        tools = ToolExecutor(Path.cwd(), approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")
        for index in range(20):
            agent.messages.append({"role": "user", "content": f"memory chunk {index} " + ("x" * 1000)})
        agent.messages.append({"role": "user", "content": "Remember MEMORY_TOKEN_77."})

        result = agent.handle_user("What token did I ask you to remember earlier in this session? Reply with the token only.")

        self.assertEqual(result.message, "MEMORY_TOKEN_77")
        sent_messages = client.calls[0]["messages"]
        sent_text = "\n".join(message["content"] for message in sent_messages)
        self.assertGreater(len(sent_messages), 20)
        self.assertIn("memory chunk 0", sent_text)
        self.assertNotIn("Earlier conversation omitted", sent_text)

    def test_agent_handles_multiturn_refactor_test_and_diff_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "tests").mkdir()
            (root / "src" / "calculator.py").write_text(
                "def add(a, b):\n"
                "    return a + b\n",
                encoding="utf-8",
            )
            (root / "tests" / "test_calculator.py").write_text(
                "import unittest\n"
                "from src.calculator import add\n\n"
                "class CalculatorTests(unittest.TestCase):\n"
                "    def test_add(self):\n"
                "        self.assertEqual(add(2, 3), 5)\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "init"], cwd=root, capture_output=True, text=True, check=True)
            subprocess.run(["git", "config", "user.name", "Tests"], cwd=root, capture_output=True, text=True, check=True)
            subprocess.run(["git", "config", "user.email", "tests@example.com"], cwd=root, capture_output=True, text=True, check=True)
            subprocess.run(["git", "add", "."], cwd=root, capture_output=True, text=True, check=True)
            subprocess.run(["git", "commit", "-m", "initial"], cwd=root, capture_output=True, text=True, check=True)

            calculator_after = (
                "def _coerce_number(value):\n"
                "    return int(value)\n\n"
                "def add(a, b):\n"
                "    return _coerce_number(a) + _coerce_number(b)\n\n"
                "def multiply(a, b):\n"
                "    return _coerce_number(a) * _coerce_number(b)\n"
            )
            tests_after = (
                "import unittest\n"
                "from src.calculator import add, multiply\n\n"
                "class CalculatorTests(unittest.TestCase):\n"
                "    def test_add(self):\n"
                "        self.assertEqual(add('2', 3), 5)\n\n"
                "    def test_multiply(self):\n"
                "        self.assertEqual(multiply('4', 5), 20)\n"
            )
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "src/calculator.py"}}),
                    json.dumps({"type": "final", "message": "calculator has add only"}),
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "src/calculator.py"}}),
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "src/calculator.py", "content": calculator_after}}),
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "tests/test_calculator.py", "content": tests_after}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": "test"}}),
                    json.dumps({"type": "final", "message": "refactor complete; tests pass"}),
                    json.dumps({"type": "tool", "name": "git_status", "arguments": {}}),
                    json.dumps({"type": "tool", "name": "git_diff", "arguments": {"path": "src/calculator.py"}}),
                    json.dumps({"type": "final", "message": "diff shows _coerce_number and multiply"}),
                ]
            )
            tools = ToolExecutor(root, approval_mode="auto", test_command="python -m unittest discover -s tests")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

            first = agent.handle_user("Inspect calculator module and summarize current functions.")
            second = agent.handle_user("Refactor calculator to coerce numeric strings, add multiply, update tests, and run tests.")
            third = agent.handle_user("Use git_status and git_diff to summarize the calculator refactor.")
            calculator_final = (root / "src" / "calculator.py").read_text(encoding="utf-8")
            tests_final = (root / "tests" / "test_calculator.py").read_text(encoding="utf-8")
            events = list(agent.events)

        self.assertEqual(first.message, "calculator has add only")
        self.assertEqual(second.message, "refactor complete; tests pass")
        self.assertEqual(third.message, "diff shows _coerce_number and multiply")
        self.assertIn("def multiply", calculator_final)
        self.assertIn("test_multiply", tests_final)
        tool_names = [event["name"] for event in events if event["type"] == "tool_call"]
        self.assertIn("read_file", tool_names)
        self.assertIn("write_file", tool_names)
        self.assertIn("run_test", tool_names)
        self.assertIn("git_status", tool_names)
        self.assertIn("git_diff", tool_names)
        run_test_results = [event for event in events if event["type"] == "tool_result" and event["name"] == "run_test"]
        self.assertTrue(run_test_results[0]["result"]["ok"])
        diff_results = [event for event in events if event["type"] == "tool_result" and event["name"] == "git_diff"]
        self.assertIn("multiply", diff_results[0]["result"]["output"])

    def test_agent_can_use_symbol_tools_instead_of_full_file_reads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            filler = "\n\n".join(f"def filler_{index}():\n    return {index}" for index in range(160))
            (root / "src" / "large_pricing.py").write_text(
                f"{filler}\n\n"
                "def calculate_discount(cart, percentage):\n"
                "    marker = 'TOKEN_SYMBOL_750'\n"
                "    return marker\n",
                encoding="utf-8",
            )
            client = FakeClient(
                [
                    '{"type":"tool","name":"search_symbols","arguments":{"query":"calculate_discount","path":"src"}}',
                    '{"type":"tool","name":"read_symbol","arguments":{"path":"src/large_pricing.py","symbol":"calculate_discount","include_context":0}}',
                    '{"type":"final","message":"TOKEN_SYMBOL_750"}',
                ]
            )
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user(
                "Use search_symbols to find calculate_discount in src/large_pricing.py. Then use read_symbol on the exact match. Do not use read_file. Reply with the uppercase TOKEN_SYMBOL marker from that symbol only."
            )

        self.assertEqual(result.message, "TOKEN_SYMBOL_750")
        self.assertEqual(len(client.calls), 2)
        self.assertFalse(any(event["type"] == "assumption_audit" for event in agent.events))
        tool_names = [event["name"] for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_names, ["search_symbols", "read_symbol"])
        symbol_results = [event for event in agent.events if event["type"] == "tool_result" and event["name"] == "read_symbol"]
        self.assertIn("TOKEN_SYMBOL_750", symbol_results[0]["result"]["output"])
        self.assertNotIn("filler_0", symbol_results[0]["result"]["output"])
        self.assertTrue(any(event["type"] == "assistant_synthesized" for event in agent.events))
