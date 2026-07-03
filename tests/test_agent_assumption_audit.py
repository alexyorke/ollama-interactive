import json
import subprocess
import sys
import tempfile
from pathlib import Path

from ollama_code.agent import OllamaCodeAgent
from ollama_code.tools import ToolExecutor
from tests.agent_test_support import AgentTestBase, CountingToolExecutor, FakeClient


class AgentAssumptionAuditTests(AgentTestBase):
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

    def test_failed_tool_prunes_accepted_assumption_audit_context(self) -> None:
        root = self._workspace_scratch()
        original = "def existing():\n    return 'old'\n"
        updated = "def existing():\n    return 'new'\n"
        (root / "sample.py").write_text(original, encoding="utf-8")
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "sample.py"}}),
                json.dumps(
                    {
                        "type": "tool",
                        "name": "edit_intent",
                        "arguments": {
                            "path": "sample.py",
                            "intent": "add_function",
                            "target": "existing",
                            "replacement": updated,
                        },
                    }
                ),
                json.dumps({"type": "tool", "name": "git_status", "arguments": {}}),
                json.dumps({"type": "final", "message": "stopped"}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", max_tool_rounds=4)

        agent.handle_user("Update sample.py.")

        audit_payloads = [
            json.loads(call["messages"][1]["content"])
            for call in client.calls
            if str(call["messages"][0]["content"]).startswith("You are a tool-step assumption auditor")
        ]
        self.assertGreaterEqual(len(audit_payloads), 2)
        self.assertEqual(audit_payloads[0]["proposed_tool"]["name"], "edit_intent")
        self.assertEqual(audit_payloads[1]["proposed_tool"]["name"], "git_status")
        self.assertEqual(audit_payloads[1]["accepted_assumption_audits"], [])

    def test_grounded_write_file_skips_audit_after_other_tool_forbidden(self) -> None:
        root = self._workspace_scratch()
        original = "def existing():\n    return 'old'\n"
        updated = "def existing():\n    return 'new'\n"
        (root / "sample.py").write_text(original, encoding="utf-8")
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "sample.py"}}),
                json.dumps(
                    {
                        "type": "tool",
                        "name": "edit_intent",
                        "arguments": {
                            "path": "sample.py",
                            "intent": "add_function",
                            "target": "existing",
                            "replacement": updated,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "tool",
                        "name": "edit_intent",
                        "arguments": {
                            "path": "sample.py",
                            "intent": "replace_function_body",
                            "target": "existing",
                            "replacement": "    return 'new'\n",
                        },
                    }
                ),
                json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "sample.py", "content": updated}}),
                json.dumps({"type": "final", "message": "stopped"}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", max_tool_rounds=5)

        agent.handle_user("Update sample.py.")

        audit_payloads = [
            json.loads(call["messages"][1]["content"])
            for call in client.calls
            if str(call["messages"][0]["content"]).startswith("You are a tool-step assumption auditor")
        ]
        audited_tools = [payload["proposed_tool"]["name"] for payload in audit_payloads]
        self.assertIn("edit_intent", audited_tools)
        self.assertNotIn("write_file", audited_tools)
        self.assertEqual(tools.execute_counts.get("write_file"), 1)
        self.assertEqual((root / "sample.py").read_text(encoding="utf-8"), updated)

    def test_grounded_replace_symbol_skips_audit_after_other_tool_forbidden(self) -> None:
        root = self._workspace_scratch()
        original = "def existing():\n    return 'old'\n"
        updated = "def existing():\n    return 'new'\n"
        (root / "sample.py").write_text(original, encoding="utf-8")
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "sample.py"}}),
                json.dumps(
                    {
                        "type": "tool",
                        "name": "edit_intent",
                        "arguments": {
                            "path": "sample.py",
                            "intent": "add_function",
                            "target": "existing",
                            "replacement": updated,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "tool",
                        "name": "edit_intent",
                        "arguments": {
                            "path": "sample.py",
                            "intent": "replace_function_body",
                            "target": "existing",
                            "replacement": "    return 'new'\n",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "tool",
                        "name": "replace_symbol",
                        "arguments": {"path": "sample.py", "symbol": "existing", "content": updated},
                    }
                ),
                json.dumps({"type": "final", "message": "stopped"}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", max_tool_rounds=5)

        agent.handle_user("Update sample.py.")

        audit_payloads = [
            json.loads(call["messages"][1]["content"])
            for call in client.calls
            if str(call["messages"][0]["content"]).startswith("You are a tool-step assumption auditor")
        ]
        audited_tools = [payload["proposed_tool"]["name"] for payload in audit_payloads]
        self.assertIn("edit_intent", audited_tools)
        self.assertNotIn("replace_symbol", audited_tools)
        self.assertEqual(tools.execute_counts.get("replace_symbol"), 1)
        self.assertEqual((root / "sample.py").read_text(encoding="utf-8"), updated)

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

    def test_agent_skips_assumption_audit_for_grounded_replacement(self) -> None:
        root = self._workspace_scratch()
        (root / "sample.py").write_text("def f():\n    return 'old'\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"sample.py"}}',
                '{"type":"tool","name":"replace_in_file","arguments":{"path":"sample.py","old":"   return \'old\'","new":"    return \'new\'"}}',
                '{"type":"final","message":"updated sample.py"}',
            ]
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        result = agent.handle_user("Inspect sample.py, change f to return 'new', and summarize.")

        self.assertEqual(result.message, "updated sample.py")
        self.assertEqual((root / "sample.py").read_text(encoding="utf-8"), "def f():\n    return 'new'\n")
        audits = [event for event in agent.events if event["type"] == "assumption_audit"]
        self.assertEqual(len(audits), 0)

    def test_agent_skips_assumption_audit_for_explicit_run_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = subprocess.list2cmdline([sys.executable, "-c", "print('test_fast OK')"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": command}}),
                ]
            )
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

            result = agent.handle_user(f"Use run_test to execute {command} and tell me whether tests passed.")

        self.assertIn("Tests passed: yes", result.message)
        audits = [event for event in agent.events if event["type"] == "assumption_audit"]
        self.assertEqual(len(audits), 0)

    def test_agent_skips_assumption_audit_for_inspection_after_failed_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.txt").write_text("failure clue\n", encoding="utf-8")
            command = subprocess.list2cmdline([sys.executable, "-c", "import sys; sys.exit(1)"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": command}}),
                    '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                    '{"type":"final","message":"inspected failure clue"}',
                ]
            )
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

            result = agent.handle_user(f"Run this failing test command: {command}. Then read note.txt and summarize.")

        self.assertEqual(result.message, "inspected failure clue")
        audits = [event for event in agent.events if event["type"] == "assumption_audit"]
        self.assertEqual(len(audits), 0)
