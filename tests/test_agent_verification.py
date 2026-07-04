import json
import tempfile
from pathlib import Path

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

    def test_verification_retry_preserves_explicit_run_test_constraint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="auto", disabled_tools=["run_shell"])
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")
            decision = agent._stabilize_retry_tool_constraints(
                {"verdict": "retry", "reason": "Retry with a required tool.", "required_tools": ["run_shell"], "forbidden_tools": []},
                sticky_required_tool_names={"run_test"},
                sticky_forbidden_tool_names={"run_shell"},
            )

        self.assertEqual(decision["required_tools"], ["run_test"])
        self.assertEqual(decision["forbidden_tools"], ["run_shell"])
        retry_prompt = agent._verification_retry_message(decision)
        self.assertIn("Required tools for this turn: run_test.", retry_prompt)
        self.assertIn("Forbidden tools for this turn: run_shell.", retry_prompt)

    def test_dynamic_mcp_tool_names_are_parsed_from_request_constraints(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient([])
        tools = ToolExecutor(
            root,
            approval_mode="auto",
            enabled_tools=["mcp_call", "read_file"],
            mcp_servers={"demo": {"command": ["python", "-c", "pass"]}},
        )
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        text = "Use mcp.demo.echo before answering, and do not use mcp.demo.delete."
        forbidden = agent._forbidden_tool_names(text)
        requested = agent._requested_tool_names(text, forbidden_tool_names=forbidden)

        self.assertEqual(forbidden, {"mcp.demo.delete"})
        self.assertEqual(requested, {"mcp.demo.echo"})

    def test_verification_retry_preserves_explicit_dynamic_mcp_constraint(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient([])
        tools = ToolExecutor(
            root,
            approval_mode="auto",
            enabled_tools=["mcp_call", "read_file"],
            mcp_servers={"demo": {"command": ["python", "-c", "pass"]}},
        )
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        decision = agent._stabilize_retry_tool_constraints(
            {
                "verdict": "retry",
                "reason": "Use the allowed MCP tool only.",
                "required_tools": ["read_file"],
                "forbidden_tools": ["mcp.demo.delete"],
            },
            sticky_required_tool_names={"mcp.demo.echo"},
            sticky_forbidden_tool_names={"mcp.demo.delete"},
        )

        self.assertEqual(decision["required_tools"], ["mcp.demo.echo", "read_file"])
        self.assertEqual(decision["forbidden_tools"], ["mcp.demo.delete"])
        retry_prompt = agent._verification_retry_message(decision)
        self.assertIn("Required tools for this turn: mcp.demo.echo, read_file.", retry_prompt)
        self.assertIn("Forbidden tools for this turn: mcp.demo.delete.", retry_prompt)

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
