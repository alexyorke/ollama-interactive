import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ollama_code.agent import AgentResult, OllamaCodeAgent
from ollama_code.features import ENV_OLLAMA_CODE_FEATURE_PROFILE
from ollama_code.tools import ToolExecutor
from tests.agent_test_support import (
    AgentTestBase,
    CountingToolExecutor,
    EmptySelectTestsLintFallbackToolExecutor,
    EmptySelectTestsToolExecutor,
    FakeClient,
    WorkflowValidatorToolExecutor,
)


class AgentPostEditValidationTests(AgentTestBase):
    def _cwd_agent(
        self,
        client: FakeClient | None = None,
        *,
        approval_mode: str = "auto",
        **kwargs: object,
    ) -> OllamaCodeAgent:
        resolved_client = client if client is not None else FakeClient([])
        return OllamaCodeAgent(
            client=resolved_client,
            tools=ToolExecutor(Path.cwd(), approval_mode=approval_mode),
            model="fake-model",
            **kwargs,
        )

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
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual([event["name"] for event in tool_calls], ["list_files", "read_file"])

    def test_agent_allows_final_answer_after_requested_tool_failure_details(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"../outside.txt"}}',
                '{"type":"final","message":"Path escapes the workspace: ../outside.txt"}',
            ]
        )
        agent = self._cwd_agent(client, debate_enabled=False)

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
        agent = self._cwd_agent(client)

        result = agent.handle_user("Use read_file on ../outside.txt and tell me the exact tool error.")

        self.assertIn("escapes the workspace", result.message)
        self.assertEqual(len(client.calls), 0)
        assumption_audits = [event for event in agent.events if event["type"] == "assumption_audit"]
        self.assertEqual(len(assumption_audits), 0)

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
        agent = self._cwd_agent(client, debate_enabled=False)

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
        agent = self._cwd_agent(client, debate_enabled=False)

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
        agent = self._cwd_agent(client, approval_mode="read-only", debate_enabled=False)

        result = agent.handle_user("Read README.md and summarize it.")

        self.assertEqual(result.message, "README loaded from file")
        self.assertIn("denied because approval mode is read-only", agent.events[2]["result"]["summary"])
        self.assertEqual(agent.events[3]["name"], "read_file")
        self.assertEqual(len(client.calls), 4)

    def test_agent_synthesizes_read_only_denial_when_user_asks_why_mutation_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient(
                [
                    '{"type":"tool","name":"write_file","arguments":{"path":"blocked.txt","content":"blocked"}}',
                ]
            )
            tools = ToolExecutor(root, approval_mode="read-only")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Try to create blocked.txt with any content. If you cannot, explain why.")

        self.assertIn("read-only", result.message.lower())
        self.assertFalse((root / "blocked.txt").exists())
        self.assertEqual(len(client.calls), 1)
        write_result = next(event for event in agent.events if event.get("type") == "tool_result" and event.get("name") == "write_file")
        self.assertIn("approval mode is read-only", str(write_result["result"]["summary"]).lower())

    def test_agent_probes_read_only_file_creation_after_non_mutating_model_reply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient(
                [
                    '{"type":"final","message":"I cannot complete that request."}',
                ]
            )
            tools = ToolExecutor(root, approval_mode="read-only")
            agent = OllamaCodeAgent(
                client=client,
                tools=tools,
                model="fake-model",
                debate_enabled=False,
                require_llm_for_turn=True,
            )

            result = agent.handle_user("Try to create blocked.txt with any content. If you cannot, explain why.")

        self.assertIn("read-only", result.message.lower())
        self.assertFalse((root / "blocked.txt").exists())
        self.assertEqual(len(client.calls), 1)
        tool_names = [event["name"] for event in agent.events if event.get("type") == "tool_call"]
        self.assertIn("write_file", tool_names)
        write_result = next(event for event in agent.events if event.get("type") == "tool_result" and event.get("name") == "write_file")
        self.assertIn("approval mode is read-only", str(write_result["result"]["summary"]).lower())

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

    def test_agent_normalizes_edit_file_alias_with_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "edit_file", "arguments": {"path": "app.py", "content": "def f():\n    return 2\n"}}),
                    json.dumps({"type": "final", "message": "app.py updated"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Edit app.py to return 2.")
            final_text = (root / "app.py").read_text(encoding="utf-8")

        self.assertEqual(result.message, "app.py updated")
        self.assertEqual(final_text, "def f():\n    return 2\n")
        self.assertEqual(tools.execute_counts.get("write_file"), 1)
        normalizations = [event for event in agent.events if event["type"] == "tool_normalized"]
        self.assertEqual(normalizations[0]["normalized_name"], "write_file")

    def test_agent_normalizes_implementation_edit_alias_to_edit_intent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "app.py"}}),
                    json.dumps(
                        {
                            "type": "tool",
                            "name": "edit_implementation_target",
                            "arguments": {
                                "path": "app.py",
                                "symbol": "add",
                                "replacement": "def add(left, right):\n    return left + right\n",
                            },
                        }
                    ),
                    json.dumps({"type": "final", "message": "app.py updated"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "baseline"}):
                result = agent.handle_user("Inspect app.py, then fix add.")
            final_text = (root / "app.py").read_text(encoding="utf-8")

        self.assertEqual(result.message, "app.py updated")
        self.assertIn("return left + right", final_text)
        normalizations = [event for event in agent.events if event["type"] == "tool_normalized"]
        self.assertEqual(normalizations[0]["normalized_name"], "edit_intent")

    def test_agent_normalizes_edit_symbol_alias_to_edit_intent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
            client = FakeClient(
                [
                    json.dumps(
                        {
                            "type": "tool",
                            "name": "edit_symbol",
                            "arguments": {
                                "path": "app.py",
                                "symbol": "add",
                                "content": "def add(left, right):\n    return left + right\n",
                            },
                        }
                    ),
                    json.dumps({"type": "final", "message": "app.py updated"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "baseline"}):
                result = agent.handle_user("Fix add in app.py.")
            final_text = (root / "app.py").read_text(encoding="utf-8")

        self.assertEqual(result.message, "app.py updated")
        self.assertIn("return left + right", final_text)
        normalizations = [event for event in agent.events if event["type"] == "tool_normalized"]
        self.assertEqual(normalizations[0]["normalized_name"], "edit_intent")

    def test_agent_normalizes_replace_body_alias_to_edit_intent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
            client = FakeClient(
                [
                    json.dumps(
                        {
                            "type": "tool",
                            "name": "replace_body",
                            "arguments": {
                                "path": "app.py",
                                "target": "add",
                                "body": "return left + right",
                            },
                        }
                    ),
                    json.dumps({"type": "final", "message": "app.py updated"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Fix add in app.py.")
            final_text = (root / "app.py").read_text(encoding="utf-8")

        self.assertIn(result.message, {"app.py updated", "Ran validation after the latest edit: passed."})
        self.assertIn("return left + right", final_text)
        self.assertEqual(tools.execute_counts.get("edit_intent"), 1)
        normalizations = [event for event in agent.events if event["type"] == "tool_normalized"]
        self.assertEqual(normalizations[0]["normalized_name"], "edit_intent")

    def test_agent_rejects_docs_only_edit_for_code_fix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "README.md", "content": "notes\n"}}),
                    json.dumps({"type": "final", "message": "done"}),
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": "def add(left, right):\n    return left + right\n"}}),
                    json.dumps({"type": "final", "message": "done"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "baseline"}):
                result = agent.handle_user("Fix the bug in the implementation.")
            final_text = (root / "app.py").read_text(encoding="utf-8")

        self.assertEqual(result.message, "done")
        self.assertIn("return left + right", final_text)
        self.assertGreaterEqual(tools.execute_counts.get("write_file", 0), 2)

    def test_agent_normalizes_snippet_replace_symbol_to_replace_in_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "replace_symbol", "arguments": {"path": "app.py", "symbol": "return left - right", "content": "return left + right"}}),
                    json.dumps({"type": "final", "message": "app.py updated"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Fix app.py so add uses addition.")
            final_text = (root / "app.py").read_text(encoding="utf-8")

        self.assertEqual(result.message, "app.py updated")
        self.assertIn("return left + right", final_text)
        self.assertEqual(tools.execute_counts.get("replace_in_file"), 1)
        self.assertIsNone(tools.execute_counts.get("replace_symbol"))
        normalizations = [event for event in agent.events if event["type"] == "tool_normalized"]
        self.assertEqual(normalizations[0]["normalized_name"], "replace_in_file")

    def test_agent_does_not_synthesize_read_symbol_final_for_fix_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
            pass_command = subprocess.list2cmdline([sys.executable, "-c", "print('OK')"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "read_symbol", "arguments": {"path": "app.py", "symbol": "add", "include_context": 0}}),
                    json.dumps({"type": "final", "message": "add returns left - right."}),
                    json.dumps({"type": "tool", "name": "replace_in_file", "arguments": {"path": "app.py", "old": "left - right", "new": "left + right"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": pass_command}}),
                    json.dumps({"type": "final", "message": "Fixed app.py and tests passed."}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

            result = agent.handle_user("Issue: app.py add returns the wrong value. Inspect source, fix it, run tests, and summarize.")

        self.assertEqual(result.message, "Fixed app.py and tests passed.")
        self.assertIn("workspace change", " ".join(message["content"] for message in agent.messages if message["role"] == "user"))
        self.assertEqual(tools.execute_counts.get("replace_in_file"), 1)

    def test_agent_normalizes_edit_payload_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs.md").write_text("# Docs\n\ntotal total\n", encoding="utf-8")
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "replace_in_file", "arguments": {"path": "docs.md", "old": "total", "new": "cart_total", "all": True}}),
                    json.dumps({"type": "final", "message": "docs updated"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Update docs.md replacing total with cart_total.")
            final_text = (root / "docs.md").read_text(encoding="utf-8")

        self.assertEqual(result.message, "docs updated")
        self.assertEqual(final_text, "# Docs\n\ncart_total cart_total\n")
        normalizations = [event for event in agent.events if event["type"] == "tool_normalized"]
        self.assertIn("replace_all", json.dumps(normalizations[0]["normalized_arguments"]))

    def test_agent_normalizes_replace_in_file_common_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs.md").write_text("# Docs\n\ntotal total totality\n", encoding="utf-8")
            client = FakeClient(
                [
                    json.dumps(
                        {
                            "type": "tool",
                            "name": "replace_in_file",
                            "arguments": {
                                "path": "docs.md",
                                "target": "total",
                                "replacement": "cart_total",
                                "all": True,
                                "whole_word": True,
                            },
                        }
                    ),
                    json.dumps({"type": "final", "message": "docs updated"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Update docs.md replacing total with cart_total.")
            final_text = (root / "docs.md").read_text(encoding="utf-8")

        self.assertEqual(result.message, "docs updated")
        self.assertEqual(final_text, "# Docs\n\ncart_total cart_total totality\n")
        normalizations = [event for event in agent.events if event["type"] == "tool_normalized"]
        self.assertIn("match_whole_word", json.dumps(normalizations[0]["normalized_arguments"]))

    def test_agent_rejects_final_before_required_workspace_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient(
                [
                    '{"type":"final","message":"implement it by editing app.py"}',
                    '{"type":"tool","name":"write_file","arguments":{"path":"app.py","content":"def f():\\n    return 1\\n"}}',
                    '{"type":"final","message":"app.py updated"}',
                ]
            )
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Implement this by editing app.py.")
            final_text = (root / "app.py").read_text(encoding="utf-8")

        self.assertEqual(result.message, "app.py updated")
        self.assertEqual(final_text, "def f():\n    return 1\n")

    def test_agent_requires_successful_run_test_after_requested_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = subprocess.list2cmdline([sys.executable, "-c", "print('ok')"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": "def f():\n    return 1\n"}}),
                    json.dumps({"type": "final", "message": "app.py updated"}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": command}}),
                    json.dumps({"type": "final", "message": "app.py updated and tests passed"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Edit app.py, run tests, and summarize.")

        self.assertEqual(result.message, "app.py updated and tests passed")
        self.assertEqual(tools.execute_counts.get("write_file"), 1)
        self.assertEqual(tools.execute_counts.get("run_test"), 1)

    def test_trajectory_validation_selects_targeted_tests_after_edit(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "src" / "pricing.py").write_text("def cart_total(prices):\n    return 0\n", encoding="utf-8")
        (root / "tests" / "test_pricing.py").write_text(
            "import unittest\n"
            "from src.pricing import cart_total\n\n"
            "class PricingTests(unittest.TestCase):\n"
            "    def test_cart_total(self):\n"
            "        self.assertEqual(cart_total([2, 3]), 5)\n",
            encoding="utf-8",
        )
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"src/pricing.py"}}',
                '{"type":"tool","name":"replace_in_file","arguments":{"path":"src/pricing.py","old":"return 0","new":"return sum(prices)"}}',
                '{"type":"final","message":"Updated src/pricing.py and tests passed."}',
                '{"type":"final","message":"Updated src/pricing.py and tests passed."}',
            ]
        )
        tools = ToolExecutor(root, approval_mode="auto", test_command=f"{sys.executable} -m unittest discover -s tests")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=6)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Fix src/pricing.py and run tests.")

        self.assertTrue(result.completed)
        tool_events = [event for event in agent.events if event.get("type") == "tool_call"]
        tool_names = [event.get("name") for event in tool_events]
        self.assertIn("select_tests", tool_names)
        run_tests = [event for event in tool_events if event.get("name") == "run_test"]
        self.assertTrue(run_tests)
        self.assertIn("test_pricing.py", str(run_tests[-1].get("arguments", {}).get("command", "")))

    def test_post_edit_validation_runs_before_extra_context_read(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "src" / "pricing.py").write_text("def cart_total(prices):\n    return 0\n", encoding="utf-8")
        (root / "tests" / "test_pricing.py").write_text(
            "import unittest\n"
            "from src.pricing import cart_total\n\n"
            "class PricingTests(unittest.TestCase):\n"
            "    def test_cart_total(self):\n"
            "        self.assertEqual(cart_total([2, 3]), 5)\n",
            encoding="utf-8",
        )
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"src/pricing.py"}}',
                '{"type":"tool","name":"replace_in_file","arguments":{"path":"src/pricing.py","old":"return 0","new":"return sum(prices)"}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"src/pricing.py"}}',
                '{"type":"final","message":"Updated src/pricing.py and tests passed."}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto", test_command=f"{sys.executable} -m unittest discover -s tests")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=6)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Fix src/pricing.py and run tests.")

        self.assertTrue(result.completed)
        self.assertEqual(result.message, "Updated src/pricing.py and tests passed.")
        tool_names = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_names[:6], ["read_file", "replace_in_file", "lint_typecheck", "contract_check", "select_tests", "run_test"])
        self.assertEqual(tools.execute_counts.get("read_file"), 1)
        self.assertTrue(any(event.get("type") == "controller_guard" and event.get("guard") == "post-edit-validation" for event in agent.events))
        self.assertTrue(any("Validation already ran after the latest edit: passed." in message["content"] for message in agent.messages if message["role"] == "user"))

    def test_post_edit_validation_runs_after_non_code_edit_before_final(self) -> None:
        root = self._workspace_scratch()
        (root / "docs").mkdir()
        (root / "tests").mkdir()
        (root / "docs" / "guide.md").write_text("Old guide text.\n", encoding="utf-8")
        (root / "tests" / "test_sample.py").write_text(
            "import unittest\n\nclass SampleTests(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n",
            encoding="utf-8",
        )
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"docs/guide.md"}}',
                '{"type":"tool","name":"replace_in_file","arguments":{"path":"docs/guide.md","old":"Old guide text.","new":"Updated guide text."}}',
                '{"type":"final","message":"Updated docs/guide.md."}',
                '{"type":"final","message":"Updated docs/guide.md."}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto", test_command=f"{sys.executable} -m unittest discover -s tests")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=6)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Update docs/guide.md to use the new wording.")

        self.assertTrue(result.completed)
        self.assertEqual(result.message, "Updated docs/guide.md.")
        self.assertIn("Updated guide text.", (root / "docs" / "guide.md").read_text(encoding="utf-8"))
        tool_names = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_names[:4], ["read_file", "replace_in_file", "discover_validators", "run_test"])
        self.assertEqual(tools.execute_counts.get("discover_validators"), 1)
        self.assertEqual(tools.execute_counts.get("run_test"), 1)
        auto_validation_names = [event.get("name") for event in agent.events if event.get("type") == "auto_validation"]
        self.assertEqual(auto_validation_names[:2], ["discover_validators", "run_test"])

    def test_post_edit_validation_prefers_workflow_validator_after_workflow_edit(self) -> None:
        root = self._workspace_scratch()
        workflow_dir = root / ".github" / "workflows"
        workflow_dir.mkdir(parents=True)
        workflow_path = workflow_dir / "ci.yml"
        workflow_path.write_text(
            "name: CI\n"
            "on: [push]\n"
            "jobs:\n"
            "  test:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: python -m unittest\n",
            encoding="utf-8",
        )
        workflow_command = subprocess.list2cmdline([sys.executable, "-c", "print('actionlint ok')"])
        default_test_command = subprocess.list2cmdline([sys.executable, "-c", "print('tests ok')"])
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":".github/workflows/ci.yml"}}',
                '{"type":"tool","name":"replace_in_file","arguments":{"path":".github/workflows/ci.yml","old":"python -m unittest","new":"python -m unittest discover -s tests"}}',
                '{"type":"final","message":"Updated workflow validation."}',
                '{"type":"final","message":"Updated workflow validation."}',
            ]
        )
        tools = WorkflowValidatorToolExecutor(
            root,
            approval_mode="auto",
            test_command=default_test_command,
            workflow_command=workflow_command,
        )
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=6)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Update .github/workflows/ci.yml to use unittest discovery.")

        self.assertTrue(result.completed)
        self.assertEqual(result.message, "Updated workflow validation.")
        self.assertIn("python -m unittest discover -s tests", workflow_path.read_text(encoding="utf-8"))
        tool_names = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_names[:4], ["read_file", "replace_in_file", "discover_validators", "run_test"])
        run_tests = [event for event in agent.events if event.get("type") == "tool_call" and event.get("name") == "run_test"]
        self.assertEqual(run_tests[0].get("arguments", {}).get("command"), workflow_command)
        self.assertNotEqual(run_tests[0].get("arguments", {}).get("command"), default_test_command)
        auto_validation = [event for event in agent.events if event.get("type") == "auto_validation"]
        self.assertEqual(auto_validation[1].get("reason"), "github-actions validator command selected after validator discovery")

    def test_post_edit_validation_blocks_generic_tests_after_workflow_edit_when_tests_forbidden(self) -> None:
        root = self._workspace_scratch()
        workflow_dir = root / ".github" / "workflows"
        workflow_dir.mkdir(parents=True)
        workflow_path = workflow_dir / "ci.yml"
        workflow_path.write_text(
            "name: CI\n"
            "on:\n"
            "  push:\n"
            "    branches: [main]\n"
            "jobs:\n"
            "  test:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: python -m unittest\n",
            encoding="utf-8",
        )
        workflow_command = subprocess.list2cmdline([sys.executable, "-c", "print('actionlint ok')"])
        default_test_command = subprocess.list2cmdline([sys.executable, "-c", "print('tests ok')"])
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":".github/workflows/ci.yml"}}',
                '{"type":"tool","name":"replace_in_file","arguments":{"path":".github/workflows/ci.yml","old":"python -m unittest","new":"python -m unittest discover -s tests -v"}}',
                json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": default_test_command}}),
                '{"type":"final","message":"Updated workflow validation."}',
                '{"type":"final","message":"Updated workflow validation."}',
            ]
        )
        tools = WorkflowValidatorToolExecutor(
            root,
            approval_mode="auto",
            test_command=default_test_command,
            workflow_command=workflow_command,
        )
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=7)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Update .github/workflows/ci.yml to use unittest discovery. Do not run Python tests; validate the workflow config.")

        self.assertTrue(result.completed)
        run_tests = [event for event in agent.events if event.get("type") == "tool_call" and event.get("name") == "run_test"]
        self.assertTrue(run_tests)
        self.assertEqual(run_tests[0].get("arguments", {}).get("command"), workflow_command)
        self.assertFalse(any(event.get("arguments", {}).get("command") == default_test_command for event in run_tests))
        self.assertTrue(any(event.get("type") == "controller_guard" and event.get("guard") == "config-validator-required" for event in agent.events))
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Do not run generic Python tests for this config edit", feedback)

    def test_deterministic_workflow_config_update_runs_scoped_validator(self) -> None:
        root = self._workspace_scratch()
        workflow_dir = root / ".github" / "workflows"
        workflow_dir.mkdir(parents=True)
        workflow_path = workflow_dir / "ci.yml"
        workflow_path.write_text(
            "name: CI\n"
            "on:\n"
            "  push:\n"
            "    branches: [main]\n"
            "jobs:\n"
            "  test:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - run: python -m unittest\n",
            encoding="utf-8",
        )
        workflow_command = subprocess.list2cmdline([sys.executable, "-c", "print('actionlint ok')"])
        default_test_command = subprocess.list2cmdline([sys.executable, "-c", "print('tests ok')"])
        client = FakeClient([])
        tools = WorkflowValidatorToolExecutor(
            root,
            approval_mode="auto",
            test_command=default_test_command,
            workflow_command=workflow_command,
        )
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=7)

        result = agent.handle_user(
            "Update .github/workflows/ci.yml to also run on pull_request and change its unittest command to `python -m unittest discover -s tests -v`. Do not run Python tests; validate the workflow config."
        )
        workflow = workflow_path.read_text(encoding="utf-8")

        self.assertTrue(result.completed)
        self.assertEqual(len(client.calls), 0)
        self.assertIn("pull_request:", workflow)
        self.assertIn("python -m unittest discover -s tests -v", workflow)
        run_tests = [event for event in agent.events if event.get("type") == "tool_call" and event.get("name") == "run_test"]
        self.assertEqual(run_tests[0].get("arguments", {}).get("command"), workflow_command)
        self.assertFalse(any(event.get("arguments", {}).get("command") == default_test_command for event in run_tests))
        tool_names = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_names[:5], ["read_file", "replace_in_file", "replace_in_file", "discover_validators", "run_test"])

    def test_require_llm_for_turn_uses_deterministic_workflow_config_update_after_context_probe(self) -> None:
        root = self._workspace_scratch()
        workflow_dir = root / ".github" / "workflows"
        workflow_dir.mkdir(parents=True)
        workflow_path = workflow_dir / "ci.yml"
        workflow_path.write_text(
            "name: CI\n"
            "on:\n"
            "  push:\n"
            "    branches: [main]\n"
            "jobs:\n"
            "  test:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - run: python -m unittest\n",
            encoding="utf-8",
        )
        workflow_command = subprocess.list2cmdline([sys.executable, "-c", "print('actionlint ok')"])
        default_test_command = subprocess.list2cmdline([sys.executable, "-c", "print('tests ok')"])
        client = FakeClient(
            [
                json.dumps(
                    {
                        "type": "tool",
                        "name": "context_pack",
                        "arguments": {"request": "workflow config", "path": ".", "limit": 6},
                    }
                )
            ]
        )
        tools = WorkflowValidatorToolExecutor(
            root,
            approval_mode="auto",
            test_command=default_test_command,
            workflow_command=workflow_command,
        )
        agent = OllamaCodeAgent(
            client=client,
            tools=tools,
            model="fake-model",
            debate_enabled=False,
            require_llm_for_turn=True,
            max_tool_rounds=7,
        )

        result = agent.handle_user(
            "Update .github/workflows/ci.yml to also run on pull_request and change its unittest command to `python -m unittest discover -s tests -v`. Do not run Python tests; validate the workflow config."
        )
        workflow = workflow_path.read_text(encoding="utf-8")

        self.assertTrue(result.completed)
        self.assertEqual(len(client.calls), 1)
        self.assertIn("pull_request:", workflow)
        self.assertIn("python -m unittest discover -s tests -v", workflow)
        tool_names = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        non_context_tool_names = [name for name in tool_names if name != "context_pack"]
        self.assertEqual(non_context_tool_names[:5], ["read_file", "replace_in_file", "replace_in_file", "discover_validators", "run_test"])
        run_tests = [event for event in agent.events if event.get("type") == "tool_call" and event.get("name") == "run_test"]
        self.assertEqual(run_tests[0].get("arguments", {}).get("command"), workflow_command)
        self.assertFalse(any(event.get("arguments", {}).get("command") == default_test_command for event in run_tests))

    def test_hidden_mutation_paths_preserve_dot_prefix_for_validation_tracking(self) -> None:
        root = self._workspace_scratch()
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)

        requested = agent._requested_mutation_paths("Update .github/workflows/ci.yml to use unittest discovery.")
        mutated = agent._mutated_paths_from_successful_results(
            [
                {
                    "name": "replace_in_file",
                    "arguments": {"path": ".github/workflows/ci.yml"},
                    "result": {"ok": True, "path": ".github/workflows/ci.yml"},
                }
            ]
        )

        self.assertEqual(requested, {".github/workflows/ci.yml"})
        self.assertEqual(mutated, {".github/workflows/ci.yml"})
        self.assertEqual(agent._preferred_non_code_validator_langs(mutated), ["github-actions", "yaml"])

    def test_post_edit_validation_runs_discovered_lint_after_non_code_edit_without_tests(self) -> None:
        root = self._workspace_scratch()
        (root / "docs").mkdir()
        (root / "docs" / "guide.md").write_text("Old guide text.\n", encoding="utf-8")
        lint_command = subprocess.list2cmdline([sys.executable, "-c", "print('lint ok')"])
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"docs/guide.md"}}',
                '{"type":"tool","name":"replace_in_file","arguments":{"path":"docs/guide.md","old":"Old guide text.","new":"Updated guide text."}}',
                '{"type":"final","message":"Updated docs/guide.md."}',
                '{"type":"final","message":"Updated docs/guide.md."}',
            ]
        )
        tools = EmptySelectTestsLintFallbackToolExecutor(root, approval_mode="auto", fallback_command=lint_command)
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=6)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Update docs/guide.md to use the new wording.")

        self.assertTrue(result.completed)
        self.assertEqual(result.message, "Updated docs/guide.md.")
        tool_names = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_names[:4], ["read_file", "replace_in_file", "discover_validators", "run_test"])
        self.assertEqual(tools.execute_counts.get("discover_validators"), 1)
        self.assertEqual(tools.execute_counts.get("run_test"), 1)
        self.assertIsNone(tools.execute_counts.get("lint_typecheck"))
        auto_validation_names = [event.get("name") for event in agent.events if event.get("type") == "auto_validation"]
        self.assertEqual(auto_validation_names[:2], ["discover_validators", "run_test"])

    def test_post_edit_validation_runs_non_test_validator_when_request_skips_tests(self) -> None:
        root = self._workspace_scratch()
        (root / "docs").mkdir()
        (root / "docs" / "guide.md").write_text("Old guide text.\n", encoding="utf-8")
        lint_command = subprocess.list2cmdline([sys.executable, "-c", "print('markdown lint ok')"])
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"docs/guide.md"}}',
                '{"type":"tool","name":"replace_in_file","arguments":{"path":"docs/guide.md","old":"Old guide text.","new":"Updated guide text."}}',
                '{"type":"final","message":"Updated docs/guide.md."}',
                '{"type":"final","message":"Updated docs/guide.md."}',
            ]
        )
        tools = EmptySelectTestsLintFallbackToolExecutor(
            root,
            approval_mode="auto",
            fallback_command=lint_command,
            test_command=f"{sys.executable} -m unittest discover -s tests",
        )
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=6)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Update docs/guide.md to use the new wording. No tests are needed.")

        self.assertTrue(result.completed)
        self.assertEqual(result.message, "Updated docs/guide.md.")
        tool_names = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_names[:4], ["read_file", "replace_in_file", "discover_validators", "run_test"])
        run_test_events = [event for event in agent.events if event.get("type") == "tool_call" and event.get("name") == "run_test"]
        self.assertEqual(run_test_events[0].get("arguments", {}).get("command"), lint_command)
        auto_validation_names = [event.get("name") for event in agent.events if event.get("type") == "auto_validation"]
        self.assertEqual(auto_validation_names[:2], ["discover_validators", "run_test"])

    def test_post_edit_validation_runs_code_sanity_when_request_skips_tests(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "docs").mkdir()
        (root / "src" / "api.py").write_text(
            "def fetch_user(user_id: str) -> dict[str, str]:\n    return {'id': user_id}\n",
            encoding="utf-8",
        )
        (root / "docs" / "api.md").write_text("`fetch_user(user_id)` returns a user dict.\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"src/api.py"}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"docs/api.md"}}',
                '{"type":"tool","name":"replace_in_file","arguments":{"path":"src/api.py","old":"def fetch_user(user_id: str) -> dict[str, str]:","new":"def fetch_user(user_id: str, include_orders: bool = False) -> dict[str, str]:"}}',
                '{"type":"tool","name":"replace_in_file","arguments":{"path":"docs/api.md","old":"`fetch_user(user_id)` returns a user dict.","new":"`fetch_user(user_id, include_orders=False)` returns a user dict."}}',
                '{"type":"final","message":"Updated src/api.py and docs/api.md."}',
                '{"type":"final","message":"Updated src/api.py and docs/api.md."}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto", test_command=f"{sys.executable} -m unittest discover -s tests")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Add include_orders to src/api.py, update docs/api.md, and do not run tests.")

        self.assertTrue(result.completed)
        self.assertEqual(result.message, "Updated src/api.py and docs/api.md.")
        tool_names = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertIn("lint_typecheck", tool_names)
        self.assertIn("contract_check", tool_names)
        self.assertNotIn("select_tests", tool_names)
        self.assertNotIn("run_test", tool_names)
        auto_validation_names = [event.get("name") for event in agent.events if event.get("type") == "auto_validation"]
        self.assertEqual(auto_validation_names[:2], ["lint_typecheck", "contract_check"])

    def test_post_edit_validation_respects_explicit_skip_validation_request(self) -> None:
        root = self._workspace_scratch()
        (root / "docs").mkdir()
        (root / "docs" / "guide.md").write_text("Old guide text.\n", encoding="utf-8")
        lint_command = subprocess.list2cmdline([sys.executable, "-c", "print('markdown lint ok')"])
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"docs/guide.md"}}',
                '{"type":"tool","name":"replace_in_file","arguments":{"path":"docs/guide.md","old":"Old guide text.","new":"Updated guide text."}}',
                '{"type":"final","message":"Updated docs/guide.md."}',
            ]
        )
        tools = EmptySelectTestsLintFallbackToolExecutor(
            root,
            approval_mode="auto",
            fallback_command=lint_command,
            test_command=f"{sys.executable} -m unittest discover -s tests",
        )
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=5)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Update docs/guide.md to use the new wording, but skip validation.")

        self.assertTrue(result.completed)
        self.assertEqual(result.message, "Updated docs/guide.md.")
        tool_names = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_names[:2], ["read_file", "replace_in_file"])
        self.assertNotIn("discover_validators", tool_names)
        self.assertNotIn("run_test", tool_names)
        self.assertNotIn("lint_typecheck", tool_names)

    def test_synthesized_final_runs_post_edit_validation_for_no_test_request(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "docs").mkdir()
        (root / "src" / "api.py").write_text(
            "def fetch_user(user_id: str) -> dict[str, str]:\n    return {'id': user_id}\n",
            encoding="utf-8",
        )
        (root / "docs" / "api.md").write_text("`fetch_user(user_id)` returns a user dict.\n", encoding="utf-8")
        client = FakeClient([])
        tools = CountingToolExecutor(root, approval_mode="auto", test_command=f"{sys.executable} -m unittest discover -s tests")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user(
                "Add an optional include_orders: bool = False parameter to fetch_user in src/api.py "
                "and update docs/api.md with that parameter, but do not run tests."
            )

        self.assertTrue(result.completed)
        self.assertEqual(result.message, "Updated docs/api.md, src/api.py.")
        self.assertEqual(len(client.calls), 0)
        tool_names = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_names[:4], ["edit_intent", "replace_in_file", "lint_typecheck", "contract_check"])
        self.assertNotIn("run_test", tool_names)
        self.assertNotIn("select_tests", tool_names)

    def test_post_edit_validation_feedback_includes_validator_diagnostic(self) -> None:
        root = self._workspace_scratch()
        (root / "docs").mkdir()
        (root / "docs" / "guide.md").write_text("Old guide text.\n", encoding="utf-8")
        lint_command = subprocess.list2cmdline(
            [sys.executable, "-c", "import sys; print('markdown heading missing'); sys.exit(1)"]
        )
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"docs/guide.md"}}',
                '{"type":"tool","name":"replace_in_file","arguments":{"path":"docs/guide.md","old":"Old guide text.","new":"Updated guide text."}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"docs/guide.md"}}',
                '{"type":"final","message":"stopped"}',
            ]
        )
        tools = EmptySelectTestsLintFallbackToolExecutor(root, approval_mode="auto", fallback_command=lint_command)
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=6)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            agent.handle_user("Update docs/guide.md to use the new wording.")

        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Post-edit validation failed before more tool use.", feedback)
        self.assertIn("markdown heading missing", feedback)

    def test_spec_guided_dataclass_cli_repair_updates_tests_docs_and_json_proof(self) -> None:
        root = self._workspace_scratch()
        (root / "tests").mkdir()
        source = (
            "from __future__ import annotations\n\n"
            "import argparse\n"
            "from dataclasses import dataclass\n\n\n"
            "@dataclass\n"
            "class Note:\n"
            "    title: str\n"
            "    body: str\n"
            "    tags: list[str]\n\n\n"
            "NOTES = [\n"
            "    Note('ship-cli', 'finish command line UX', ['work', 'todo']),\n"
            "    Note('buy-milk', 'remember oat milk', ['home']),\n"
            "    Note('fix-bug', 'handle empty input', ['work']),\n"
            "]\n\n\n"
            "def list_notes(tag: str | None = None) -> list[str]:\n"
            "    notes = NOTES if tag is None else [note for note in NOTES if tag in note.tags]\n"
            "    return [f'{note.title}: {note.body}' for note in notes]\n\n\n"
            "def main(argv: list[str] | None = None) -> int:\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--tag', help='Only show notes with this tag')\n"
            "    args = parser.parse_args(argv)\n"
            "    for line in list_notes(args.tag):\n"
            "        print(line)\n"
            "    return 0\n\n\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(main())\n"
        )
        test_source = (
            "import subprocess\nimport sys\nimport unittest\nfrom pathlib import Path\n\n"
            "ROOT = Path(__file__).resolve().parents[1]\n\n"
            "def run_cli(*args: str) -> subprocess.CompletedProcess[str]:\n"
            "    return subprocess.run([sys.executable, str(ROOT / 'notes_cli.py'), *args], capture_output=True, text=True, check=False)\n\n"
            "class NotesCliTests(unittest.TestCase):\n"
            "    def test_lists_notes(self) -> None:\n"
            "        result = run_cli()\n"
            "        self.assertEqual(result.returncode, 0)\n"
            "        self.assertIn('ship-cli: finish command line UX', result.stdout)\n\n"
            "    def test_filters_by_tag(self) -> None:\n"
            "        result = run_cli('--tag', 'home')\n"
            "        self.assertEqual(result.returncode, 0)\n"
            "        self.assertIn('buy-milk', result.stdout)\n"
            "        self.assertNotIn('ship-cli', result.stdout)\n"
        )
        (root / "notes_cli.py").write_text(source, encoding="utf-8")
        (root / "README.md").write_text("# Notes CLI\n\n- `python notes_cli.py --tag work` filters by tag.\n", encoding="utf-8")
        (root / "tests" / "test_notes_cli.py").write_text(test_source, encoding="utf-8")
        command = f"{sys.executable} -m unittest discover -s tests -p test_notes_cli.py"
        tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
        agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=6)
        successful_tool_results = [
            {"name": "read_file", "arguments": {"path": "notes_cli.py"}, "result": {"ok": True, "path": "notes_cli.py", "output": source}},
            {"name": "read_file", "arguments": {"path": "tests/test_notes_cli.py"}, "result": {"ok": True, "path": "tests/test_notes_cli.py", "output": test_source}},
        ]
        tool_calls: list[dict[str, object]] = []

        result = agent._try_spec_guided_repair(
            request_text="Add a --json flag to notes_cli.py, update README.md and tests, run tests, and prove --tag work --json from the shell.",
            round_number=3,
            failed_run_test_result={"ok": False, "tool": "run_test", "summary": "json behavior missing", "output": "json behavior missing"},
            run_test_arguments={"command": command},
            successful_tool_results=successful_tool_results,
            satisfied_tool_names=set(),
            tool_calls_this_turn=tool_calls,
            allow_workspace_fallback=True,
        )

        self.assertIsNotNone(result)
        self.assertTrue(result.completed)
        self.assertIn("--json", (root / "notes_cli.py").read_text(encoding="utf-8"))
        self.assertIn("--json", (root / "README.md").read_text(encoding="utf-8"))
        self.assertIn("test_json_output", (root / "tests" / "test_notes_cli.py").read_text(encoding="utf-8"))
        run_shell_commands = [
            str(call.get("arguments", {}).get("command", ""))
            for call in tool_calls
            if call.get("name") == "run_shell" and isinstance(call.get("arguments"), dict)
        ]
        self.assertTrue(any("--json" in command for command in run_shell_commands))
        self.assertTrue(any("--tag work --json" in command for command in run_shell_commands))

    def test_spec_guided_mechanical_repair_rejects_unproven_requested_flag(self) -> None:
        root = self._workspace_scratch()
        (root / "tests").mkdir()
        source = (
            "from __future__ import annotations\n\n"
            "import argparse\nimport json\n"
            "from dataclasses import asdict, dataclass\n\n\n"
            "@dataclass\n"
            "class Note:\n"
            "    title: str\n"
            "    body: str\n"
            "    tags: list[str]\n\n\n"
            "NOTES = [\n"
            "    Note('ship-cli', 'finish command line UX', ['work', 'todo']),\n"
            "    Note('buy-milk', 'remember oat milk', ['home']),\n"
            "    Note('fix-bug', 'handle empty input', ['work']),\n"
            "]\n\n\n"
            "def selected_notes(tag: str | None = None) -> list[Note]:\n"
            "    if tag is None:\n"
            "        return list(NOTES)\n"
            "    return [note for note in NOTES if tag in note.tags]\n\n\n"
            "def list_notes(tag: str | None = None) -> list[str]:\n"
            "    return [f'{note.title}: {note.body}' for note in selected_notes(tag)]\n\n\n"
            "def json_items(tag: str | None = None) -> str:\n"
            "    return json.dumps([asdict(note) for note in selected_notes(tag)])\n\n\n"
            "def main(argv: list[str] | None = None) -> int:\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--tag', help='Only show notes with this tag')\n"
            "    parser.add_argument('--json', action='store_true', help='Print selected items as JSON')\n"
            "    args = parser.parse_args(argv)\n"
            "    if args.json:\n"
            "        print(json_items(args.tag))\n"
            "        return 0\n"
            "    for line in list_notes(args.tag):\n"
            "        print(line)\n"
            "    return 0\n\n\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(main())\n"
        )
        test_source = (
            "import subprocess\nimport sys\nimport unittest\nfrom pathlib import Path\n\n"
            "ROOT = Path(__file__).resolve().parents[1]\n\n"
            "def run_cli(*args: str) -> subprocess.CompletedProcess[str]:\n"
            "    return subprocess.run([sys.executable, str(ROOT / 'notes_cli.py'), *args], capture_output=True, text=True, check=False)\n\n"
            "class NotesCliTests(unittest.TestCase):\n"
            "    def test_json_output(self) -> None:\n"
            "        result = run_cli('--json')\n"
            "        self.assertEqual(result.returncode, 0)\n"
            "        self.assertIn('\"title\"', result.stdout)\n"
        )
        (root / "notes_cli.py").write_text(source, encoding="utf-8")
        (root / "README.md").write_text("# Notes CLI\n\n- `python notes_cli.py --json` prints JSON.\n", encoding="utf-8")
        (root / "tests" / "test_notes_cli.py").write_text(test_source, encoding="utf-8")
        command = f"{sys.executable} -m unittest discover -s tests -p test_notes_cli.py"
        tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
        agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=6)
        successful_tool_results = [
            {"name": "read_file", "arguments": {"path": "notes_cli.py"}, "result": {"ok": True, "path": "notes_cli.py", "output": source}},
            {"name": "read_file", "arguments": {"path": "tests/test_notes_cli.py"}, "result": {"ok": True, "path": "tests/test_notes_cli.py", "output": test_source}},
        ]
        tool_calls: list[dict[str, object]] = []

        request_text = "Add a --sort title option, update README.md and tests, run tests, and prove --sort with --json from the shell."
        result = agent._try_spec_guided_mechanical_repair(
            request_text=request_text,
            round_number=3,
            source_path="notes_cli.py",
            test_path="tests/test_notes_cli.py",
            test_command=command,
            successful_tool_results=successful_tool_results,
            satisfied_tool_names=set(),
            tool_calls_this_turn=tool_calls,
        )

        self.assertIsNone(result)
        self.assertFalse(any(event.get("type") == "assistant_synthesized" for event in agent.events))
        obligation_events = [
            event
            for event in agent.events
            if event.get("type") == "spec_guided_repair"
            and event.get("phase") == "mechanical_obligation_verification"
        ]
        self.assertEqual(obligation_events[-1].get("ok"), False)
        unresolved_labels = [
            str(item.get("label") or "")
            for item in obligation_events[-1].get("unresolved_obligations", [])
            if isinstance(item, dict)
        ]
        self.assertTrue(any('"--sort" flag' in label for label in unresolved_labels))
        repeat_direct = agent._try_spec_guided_mechanical_repair(
            request_text=request_text,
            round_number=4,
            source_path="notes_cli.py",
            test_path="tests/test_notes_cli.py",
            test_command=command,
            successful_tool_results=successful_tool_results,
            satisfied_tool_names=set(),
            tool_calls_this_turn=tool_calls,
        )
        obligation_events_after_repeat = [
            event
            for event in agent.events
            if event.get("type") == "spec_guided_repair"
            and event.get("phase") == "mechanical_obligation_verification"
        ]
        self.assertIsNone(repeat_direct)
        self.assertEqual(len(obligation_events_after_repeat), len(obligation_events))
        prior_mechanical_starts = [
            event
            for event in agent.events
            if event.get("type") == "spec_guided_repair"
            and event.get("phase") == "post_context_cli_mechanical_start"
        ]
        request_obligations = agent._derive_request_obligations(
            request_text=request_text,
            required_tool_names=set(),
            required_mutation_paths=set(),
            code_mutation_required=True,
            test_run_required=True,
        )

        repeat = agent._try_post_context_cli_feature_repair(
            request_text=request_text,
            round_number=4,
            request_obligations=request_obligations,
            forbidden_tool_names=set(),
            successful_tool_results=successful_tool_results,
            satisfied_tool_names=set(),
            tool_calls_this_turn=tool_calls,
        )
        later_mechanical_starts = [
            event
            for event in agent.events
            if event.get("type") == "spec_guided_repair"
            and event.get("phase") == "post_context_cli_mechanical_start"
        ]

        self.assertIsNone(repeat)
        self.assertEqual(len(later_mechanical_starts), len(prior_mechanical_starts))

    def test_failed_proactive_run_test_invokes_spec_guided_repair(self) -> None:
        root = self._workspace_scratch()
        (root / "tests").mkdir()
        (root / "app.py").write_text("def value() -> int:\n    return 0\n", encoding="utf-8")
        (root / "tests" / "test_app.py").write_text(
            "import unittest\nfrom app import value\n\n"
            "class AppTests(unittest.TestCase):\n"
            "    def test_value(self):\n"
            "        self.assertEqual(value(), 1)\n",
            encoding="utf-8",
        )
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"app.py"}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"tests/test_app.py"}}',
                '{"type":"tool","name":"replace_in_file","arguments":{"path":"app.py","old":"return 0","new":"return 2"}}',
                '{"type":"final","message":"Updated app.py and tests passed."}',
            ]
        )
        tools = ToolExecutor(root, approval_mode="auto", test_command=f"{sys.executable} -m unittest discover -s tests")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=6)
        calls: list[dict[str, object]] = []

        def fake_spec_guided_repair(**kwargs: object) -> AgentResult | None:
            calls.append(dict(kwargs))
            failed = kwargs.get("failed_run_test_result")
            if isinstance(failed, dict) and failed.get("tool") == "preemptive_spec_repair":
                return None
            return AgentResult(message="spec repair called", rounds=int(kwargs["round_number"]), completed=False)

        agent._try_spec_guided_repair = fake_spec_guided_repair  # type: ignore[method-assign]

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Fix app.py and run tests.")

        self.assertFalse(result.completed)
        self.assertEqual(result.message, "spec repair called")
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(calls[-1]["run_test_arguments"], {})
        self.assertFalse(calls[-1]["failed_run_test_result"]["ok"])
        self.assertIn("Post-edit example probes failed", calls[-1]["failed_run_test_result"]["summary"])

    def test_failed_partial_overwrite_uses_related_test_for_spec_guided_repair(self) -> None:
        root = self._workspace_scratch()
        (root / "tests").mkdir()
        (root / "app.py").write_text(
            "def value() -> int:\n"
            "    return 0\n\n"
            "def main() -> int:\n"
            "    return value()\n",
            encoding="utf-8",
        )
        (root / "tests" / "test_app.py").write_text(
            "import unittest\nfrom app import value\n\n"
            "class AppTests(unittest.TestCase):\n"
            "    def test_value(self):\n"
            "        if value() != 1:\n"
            "            raise AssertionError(f'{value()} != 1')\n",
            encoding="utf-8",
        )
        client = FakeClient([])
        tools = ToolExecutor(root, approval_mode="auto", test_command=f"{sys.executable} -m unittest discover -s tests")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=6)

        successful_tool_results = [
            {
                "name": "read_file",
                "arguments": {"path": "app.py"},
                "result": {
                    "ok": True,
                    "tool": "read_file",
                    "path": "app.py",
                    "content": (root / "app.py").read_text(encoding="utf-8"),
                },
            }
        ]

        self.assertEqual(
            agent._spec_guided_repair_paths(successful_tool_results, allow_workspace_fallback=True),
            ("app.py", "tests/test_app.py"),
        )

    def test_spec_guided_repair_paths_reroute_package_init_to_backing_module(self) -> None:
        root = self._workspace_scratch()
        (root / "analytics").mkdir()
        (root / "tests").mkdir()
        (root / "analytics" / "__init__.py").write_text(
            "from .events import RequestEvent, summarize_status, percentile_latency\n\n"
            "__all__ = [\"RequestEvent\", \"summarize_status\"]\n",
            encoding="utf-8",
        )
        (root / "analytics" / "events.py").write_text(
            "class RequestEvent:\n"
            "    pass\n\n\n"
            "def summarize_status(events):\n"
            "    return {}\n",
            encoding="utf-8",
        )
        (root / "tests" / "test_events.py").write_text(
            "from analytics import RequestEvent, summarize_status\n\n\n"
            "def test_summarize_status():\n"
            "    assert summarize_status([]) == {}\n",
            encoding="utf-8",
        )
        tools = ToolExecutor(root, approval_mode="auto", test_command=f"{sys.executable} -m unittest discover -s tests")
        agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)
        successful_tool_results = [
            {
                "name": "read_file",
                "arguments": {"path": "tests/test_events.py"},
                "result": {"ok": True, "path": "tests/test_events.py", "output": "from analytics import summarize_status"},
            },
            {
                "name": "edit_intent",
                "arguments": {"path": "analytics/__init__.py", "intent": "add_import", "target": "percentile_latency"},
                "result": {"ok": True, "path": "analytics/__init__.py", "summary": "Added import to analytics/__init__.py."},
            },
        ]

        self.assertEqual(
            agent._spec_guided_repair_paths(successful_tool_results, allow_workspace_fallback=True),
            ("analytics/events.py", "tests/test_events.py"),
        )

    def test_final_repair_spec_stop_attempts_spec_guided_repair(self) -> None:
        root = self._workspace_scratch()
        (root / "tests").mkdir()
        (root / "app.py").write_text("def value() -> int:\n    return 0\n", encoding="utf-8")
        (root / "tests" / "test_app.py").write_text(
            "import unittest\nfrom app import value\n\n"
            "class AppTests(unittest.TestCase):\n"
            "    def test_value(self):\n"
            "        self.assertEqual(value(), 1)\n",
            encoding="utf-8",
        )
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"app.py"}}',
                '{"type":"tool","name":"replace_in_file","arguments":{"path":"app.py","old":"return 0","new":"return 2"}}',
                '{"type":"tool","name":"run_test","arguments":{}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"tests/test_app.py"}}',
                '{"type":"final","message":"done"}',
            ]
        )
        tools = ToolExecutor(root, approval_mode="auto", test_command=f"{sys.executable} -m unittest discover -s tests")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=5)
        calls: list[dict[str, object]] = []

        def fake_spec_guided_repair(**kwargs: object) -> AgentResult | None:
            calls.append(dict(kwargs))
            if len(calls) == 1:
                return None
            return AgentResult(message="final repair spec called", rounds=int(kwargs["round_number"]), completed=False)

        agent._try_spec_guided_repair = fake_spec_guided_repair  # type: ignore[method-assign]

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Fix app.py and run tests.")

        self.assertFalse(result.completed)
        self.assertEqual(result.message, "final repair spec called")
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(calls[-1]["failed_run_test_result"]["tool"], "run_test")
        self.assertTrue(calls[-1].get("allow_workspace_fallback"))

    def test_known_syntax_error_blocks_lint_validator_until_repair(self) -> None:
        root = self._workspace_scratch()
        (root / "app.py").write_text("def value() -> str:\n    return 'ok'\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"app.py"}}',
                '{"type":"tool","name":"write_file","arguments":{"path":"app.py","content":"def value() -> str:\\n    return \\"unterminated\\n"}}',
                '{"type":"tool","name":"lint_typecheck","arguments":{"paths":"app.py"}}',
                '{"type":"final","message":"done"}',
                '{"type":"final","message":"done"}',
                '{"type":"final","message":"done"}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto", test_command=f"{sys.executable} -m unittest discover")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=5)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            agent.handle_user("Fix app.py and run validation.")

        self.assertEqual(tools.execute_counts.get("write_file"), 1)
        self.assertEqual((root / "app.py").read_text(encoding="utf-8"), "def value() -> str:\n    return 'ok'\n")
        self.assertTrue(
            any(event.get("type") == "controller_guard" and event.get("guard") == "syntax-error-rollback" for event in agent.events)
        )
        lint_tool_calls = [event for event in agent.events if event.get("type") == "tool_call" and event.get("name") == "lint_typecheck"]
        lint_auto_validations = [event for event in agent.events if event.get("type") == "auto_validation" and event.get("name") == "lint_typecheck"]
        self.assertEqual(len(lint_tool_calls), len(lint_auto_validations))
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Do not run validators while Python syntax errors are already known", feedback)
        self.assertIn("omit docstrings and prose strings", feedback)
        self.assertIn("intent add_function", feedback)

    def test_repeated_invalid_python_mutations_fail_closed_after_repair_guidance(self) -> None:
        root = self._workspace_scratch()
        (root / "app.py").write_text("def value() -> str:\n    return 'ok'\n", encoding="utf-8")
        bad_content = "def value() -> str:\n    \"unterminated\n    return 'new'\n"
        bad_call = json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": bad_content}})
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "app.py"}}),
                bad_call,
                bad_call,
                bad_call,
                bad_call,
                json.dumps({"type": "final", "message": "should not be reached"}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto", test_command=f"{sys.executable} -m unittest discover -s tests -v")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

        with patch.object(agent, "_try_spec_guided_repair", return_value=None) as repair:
            result = agent.handle_user("Fix app.py and run tests.")

        self.assertFalse(result.completed)
        self.assertIn("repeated Python mutation payloads", result.message)
        self.assertEqual(tools.execute_counts.get("write_file"), 1)
        self.assertEqual((root / "app.py").read_text(encoding="utf-8"), "def value() -> str:\n    return 'ok'\n")
        self.assertGreaterEqual(repair.call_count, 1)
        invalid_guards = [
            event
            for event in agent.events
            if event.get("type") == "controller_guard" and event.get("guard") == "invalid-python-mutation-payload"
        ]
        compressed_guards = [
            event
            for event in agent.events
            if event.get("type") == "controller_guard" and event.get("guard") == "invalid-python-mutation-loop-compressed"
        ]
        self.assertEqual(len(invalid_guards), 3)
        self.assertEqual(len(compressed_guards), 1)

    def test_placeholder_completion_reprompts_after_stub_like_code_edit_without_failed_tests(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "src" / "math_utils.py").write_text(
            "def add(left, right):\n    return left - right\n",
            encoding="utf-8",
        )
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"src/math_utils.py"}}',
                '{"type":"tool","name":"edit_intent","arguments":{"path":"src/math_utils.py","intent":"replace_body","target":"add","replacement":"pass"}}',
                '{"type":"final","message":"Implemented src/math_utils.py successfully."}',
                '{"type":"tool","name":"edit_intent","arguments":{"path":"src/math_utils.py","intent":"replace_body","target":"add","replacement":"return left + right"}}',
                '{"type":"final","message":"Implemented src/math_utils.py successfully."}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Implement add in src/math_utils.py so it returns the sum.")

        self.assertTrue(result.completed)
        self.assertEqual(result.message, "Implemented src/math_utils.py successfully.")
        self.assertIn("return left + right", (root / "src" / "math_utils.py").read_text(encoding="utf-8"))
        controller_guards = [
            event
            for event in agent.events
            if event.get("type") == "controller_guard" and event.get("guard") == "placeholder-completion-claim"
        ]
        self.assertEqual(len(controller_guards), 1)
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("still a stub/comment/pass-style placeholder", feedback)
        self.assertEqual(tools.execute_counts.get("edit_intent"), 2)

    def test_missing_path_suggestion_blocks_mutating_wrong_package_path(self) -> None:
        root = self._workspace_scratch()
        (root / "reports").mkdir()
        (root / "reports" / "__init__.py").write_text("from .exporter import ReportRow\n", encoding="utf-8")
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "__init__.py"}}),
                json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "__init__.py", "content": "from .report_row import ReportRow\n"}}),
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "reports/__init__.py"}}),
                json.dumps({"type": "final", "message": "grounded package init"}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=4)

        result = agent.handle_user("Add export_ndjson to the package __init__.py for this report exporter.")

        self.assertFalse(result.completed)
        self.assertIsNone(tools.execute_counts.get("write_file"))
        self.assertEqual(tools.execute_counts.get("read_file"), 2)
        guard_events = [
            event
            for event in agent.events
            if event.get("type") == "controller_guard" and event.get("guard") == "missing-path-mutation-target"
        ]
        self.assertEqual(len(guard_events), 1)
        self.assertEqual(guard_events[0].get("missing_path"), "__init__.py")
        self.assertEqual(guard_events[0].get("suggested_paths"), ["reports/__init__.py"])
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Re-read the suggested path first: reports/__init__.py", feedback)

    def test_write_file_with_edit_markers_is_rejected_before_execution(self) -> None:
        root = self._workspace_scratch()
        (root / "app.py").write_text("def value() -> int:\n    return 1\n", encoding="utf-8")
        client = FakeClient(
            [
                json.dumps(
                    {
                        "type": "tool",
                        "name": "write_file",
                        "arguments": {
                            "path": "app.py",
                            "content": ">> BEGIN EDITED CONTENT <<\ndef value() -> int:\n    return 2\n>>> END EDITED CONTENT >>>\n",
                        },
                    }
                ),
                json.dumps({"type": "final", "message": "stopped"}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=2)

        agent.handle_user("Update app.py so value returns 2.")

        self.assertIsNone(tools.execute_counts.get("write_file"))
        self.assertEqual((root / "app.py").read_text(encoding="utf-8"), "def value() -> int:\n    return 1\n")
        self.assertTrue(
            any(
                event.get("type") == "controller_guard" and event.get("guard") == "write-file-rewrite-markers"
                for event in agent.events
            )
        )

    def test_invalid_add_function_payload_is_rejected_before_tool_execution(self) -> None:
        root = self._workspace_scratch()
        (root / "reports").mkdir()
        (root / "reports" / "exporter.py").write_text(
            "from dataclasses import dataclass\n\n\n"
            "@dataclass(frozen=True)\n"
            "class ReportRow:\n"
            "    name: str\n"
            "    count: int\n"
            "    active: bool\n",
            encoding="utf-8",
        )
        invalid_replacement = (
            "def export_ndjson(rows):\n"
            "    \"Serialize rows to NDJSON.\n"
            "    return \"\"\n"
        )
        valid_replacement = (
            "def export_ndjson(rows: list[ReportRow]) -> str:\n"
            "    import json\n"
            "    if not rows:\n"
            "        return \"\"\n"
            "    return \"\\n\".join(json.dumps({\"name\": row.name, \"count\": row.count, \"active\": row.active}) for row in rows) + \"\\n\"\n"
        )
        client = FakeClient(
            [
                json.dumps(
                    {
                        "type": "tool",
                        "name": "edit_intent",
                        "arguments": {
                            "path": "reports/exporter.py",
                            "intent": "add_function",
                            "target": "export_ndjson",
                            "replacement": invalid_replacement,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "tool",
                        "name": "edit_intent",
                        "arguments": {
                            "path": "reports/exporter.py",
                            "intent": "add_function",
                            "target": "export_ndjson",
                            "replacement": valid_replacement,
                        },
                    }
                ),
                json.dumps({"type": "final", "message": "added export_ndjson"}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=4)

        result = agent.handle_user("Add export_ndjson to reports/exporter.py.")

        self.assertTrue(result.completed)
        self.assertEqual(tools.execute_counts.get("edit_intent"), 1)
        final_text = (root / "reports" / "exporter.py").read_text(encoding="utf-8")
        self.assertIn("def export_ndjson(rows: list[ReportRow]) -> str:", final_text)
        guard_events = [
            event
            for event in agent.events
            if event.get("type") == "controller_guard" and event.get("guard") == "invalid-python-mutation-payload"
        ]
        self.assertEqual(len(guard_events), 1)
        self.assertEqual(guard_events[0].get("tool"), "edit_intent")
        self.assertIn("unterminated string literal", str(guard_events[0].get("diagnostic") or ""))
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("syntactically invalid before execution", feedback)
        self.assertIn("omit docstrings and prose strings", feedback)
        self.assertIn("escaped '\\n' string literals", feedback)

    def test_repeated_invalid_add_function_payload_pivots_to_spec_guided_repair(self) -> None:
        root = self._workspace_scratch()
        (root / "reports").mkdir()
        (root / "tests").mkdir()
        (root / "reports" / "exporter.py").write_text("def export_csv(rows):\n    return \"\"\n", encoding="utf-8")
        (root / "tests" / "test_exporter.py").write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")
        invalid_replacement = (
            "def export_ndjson(rows):\n"
            "    \"Serialize rows to NDJSON.\n"
            "    return \"\"\n"
        )
        invalid_call = {
            "type": "tool",
            "name": "edit_intent",
            "arguments": {
                "path": "reports/exporter.py",
                "intent": "add_function",
                "target": "export_ndjson",
                "replacement": invalid_replacement,
            },
        }
        client = FakeClient([json.dumps(invalid_call), json.dumps(invalid_call)])
        tools = CountingToolExecutor(root, approval_mode="auto", test_command=f"{sys.executable} -m unittest discover -s tests -v")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=4)

        with patch.object(
            agent,
            "_try_spec_guided_repair",
            return_value=AgentResult(message="spec-guided repair", rounds=2, completed=True),
        ) as repair:
            result = agent.handle_user("Add export_ndjson to reports/exporter.py and run tests.")

        self.assertTrue(result.completed)
        self.assertEqual(result.message, "spec-guided repair")
        self.assertIsNone(tools.execute_counts.get("edit_intent"))
        self.assertEqual(repair.call_count, 1)
        call_kwargs = repair.call_args.kwargs
        self.assertTrue(call_kwargs["allow_workspace_fallback"])
        self.assertEqual(call_kwargs["failed_run_test_result"]["tool"], "edit_intent")
        guard_events = [
            event
            for event in agent.events
            if event.get("type") == "controller_guard" and event.get("guard") == "invalid-python-mutation-payload"
        ]
        self.assertEqual(len(guard_events), 2)
        self.assertTrue(all(event.get("tool") == "edit_intent" for event in guard_events))

    def test_contract_guards_run_contract_check_before_targeted_tests(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "src" / "pricing.py").write_text("def cart_total(prices: list[int]) -> int:\n    return 0\n", encoding="utf-8")
        (root / "tests" / "test_pricing.py").write_text(
            "import unittest\n"
            "from src.pricing import cart_total\n\n"
            "class PricingTests(unittest.TestCase):\n"
            "    def test_cart_total(self):\n"
            "        self.assertEqual(cart_total([2, 3]), 5)\n",
            encoding="utf-8",
        )
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"src/pricing.py"}}',
                '{"type":"tool","name":"replace_in_file","arguments":{"path":"src/pricing.py","old":"return 0","new":"return sum(prices)"}}',
                '{"type":"final","message":"Updated src/pricing.py and tests passed."}',
            ]
        )
        tools = ToolExecutor(root, approval_mode="auto", test_command=f"{sys.executable} -m unittest discover -s tests")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=6)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "contract-guards"}):
            result = agent.handle_user("Fix src/pricing.py and run tests.")

        self.assertTrue(result.completed)
        tool_names = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertIn("contract_check", tool_names)
        self.assertIn("select_tests", tool_names)
        self.assertLess(tool_names.index("contract_check"), tool_names.index("select_tests"))

    def test_contract_guards_fail_closed_on_contract_mismatch(self) -> None:
        root = self._workspace_scratch()
        (root / "app.py").write_text("def value() -> int:\n    return 1\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"app.py"}}',
                '{"type":"tool","name":"replace_symbol","arguments":{"path":"app.py","symbol":"value","content":"def value() -> int:\\n    pass\\n"}}',
                '{"type":"final","message":"Updated app.py."}',
            ]
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=6)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "contract-guards"}):
            result = agent.handle_user("Fix app.py.")

        self.assertFalse(result.completed)
        self.assertIn("post-edit validation failed", result.message)
        self.assertIn("may return None", result.message)

    def test_repeated_invalid_write_file_after_syntax_rollback_pivots_to_spec_guided_repair(self) -> None:
        root = self._workspace_scratch()
        (root / "app.py").write_text("def value() -> str:\n    return 'ok'\n", encoding="utf-8")
        bad_content = "def value() -> str:\n    \"unterminated\n    return 'new'\n"
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "app.py"}}),
                json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": bad_content}}),
                json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": bad_content}}),
                json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": bad_content}}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto", test_command=f"{sys.executable} -m unittest discover -s tests -v")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=5)
        repair_calls: list[dict[str, object]] = []

        def fake_repair(**kwargs: object) -> AgentResult | None:
            repair_calls.append(dict(kwargs))
            if len(repair_calls) == 1:
                return None
            return AgentResult(message="spec-guided repair", rounds=4, completed=False)

        with patch.object(agent, "_try_spec_guided_repair", side_effect=fake_repair) as repair:
            result = agent.handle_user("Fix app.py and run tests.")

        self.assertFalse(result.completed)
        self.assertEqual(result.message, "spec-guided repair")
        self.assertEqual(tools.execute_counts.get("write_file"), 1)
        self.assertEqual((root / "app.py").read_text(encoding="utf-8"), "def value() -> str:\n    return 'ok'\n")
        self.assertGreaterEqual(repair.call_count, 2)
        self.assertTrue(repair.call_args.kwargs["allow_workspace_fallback"])
        guards = [
            event
            for event in agent.events
            if event.get("type") == "controller_guard" and event.get("guard") == "invalid-python-mutation-payload"
        ]
        self.assertEqual(len(guards), 2)
        self.assertTrue(all(event.get("tool") == "write_file" for event in guards))

    def test_spec_guided_cli_repair_proves_docs_and_behavior(self) -> None:
        root = self._workspace_scratch()
        (root / "tests").mkdir()
        source = (
            "from __future__ import annotations\n\n"
            "import argparse\n\n"
            "TASKS = [\n"
            "    {'title': 'write-docs', 'status': 'todo', 'priority': 'high'},\n"
            "    {'title': 'ship-cli', 'status': 'done', 'priority': 'low'},\n"
            "    {'title': 'fix-bug', 'status': 'todo', 'priority': 'medium'},\n"
            "]\n\n"
            "def list_tasks(priority: str | None = None) -> list[str]:\n"
            "    tasks = TASKS if priority is None else [task for task in TASKS if task['priority'] == priority]\n"
            "    return [f\"{task['title']}:{task['status']}:{task['priority']}\" for task in tasks]\n\n"
            "def complete_task(title: str) -> str:\n"
            "    for task in TASKS:\n"
            "        if task['title'] == title:\n"
            "            task['status'] = 'done'\n"
            "            return f\"completed:{title}\"\n"
            "    raise SystemExit(f\"unknown task: {title}\")\n\n"
            "def main(argv: list[str] | None = None) -> int:\n"
            "    parser = argparse.ArgumentParser()\n"
            "    subparsers = parser.add_subparsers(dest='command', required=True)\n"
            "    subparsers.add_parser('list')\n"
            "    complete_parser = subparsers.add_parser('complete')\n"
            "    complete_parser.add_argument('title')\n"
            "    args = parser.parse_args(argv)\n"
            "    if args.command == 'list':\n"
            "        print('\\n'.join(list_tasks()))\n"
            "        return 0\n"
            "    if args.command == 'complete':\n"
            "        print(complete_task(args.title))\n"
            "        return 0\n"
            "    raise SystemExit(f\"unsupported command: {args.command}\")\n\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(main())\n"
        )
        test_source = (
            "import subprocess\nimport sys\nimport unittest\nfrom pathlib import Path\n\n"
            "ROOT = Path(__file__).resolve().parents[1]\n\n"
            "def _run(*args: str) -> subprocess.CompletedProcess[str]:\n"
            "    return subprocess.run([sys.executable, str(ROOT / 'task_cli.py'), *args], capture_output=True, text=True, check=False)\n\n"
            "class TaskCliTests(unittest.TestCase):\n"
            "    def test_list(self) -> None:\n"
            "        result = _run('list')\n"
            "        self.assertEqual(result.returncode, 0)\n"
            "        self.assertIn('write-docs:todo:high', result.stdout)\n\n"
            "    def test_complete(self) -> None:\n"
            "        result = _run('complete', 'write-docs')\n"
            "        self.assertEqual(result.returncode, 0)\n"
            "        self.assertIn('completed:write-docs', result.stdout)\n"
        )
        (root / "task_cli.py").write_text(source, encoding="utf-8")
        (root / "README.md").write_text("# Task CLI\n\nCommands:\n- `list`\n- `complete <title>`\n", encoding="utf-8")
        (root / "tests" / "test_task_cli.py").write_text(test_source, encoding="utf-8")
        command = f"{sys.executable} -m unittest discover -s tests -p test_task_cli.py"
        tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
        agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=6)
        successful_tool_results = [
            {"name": "read_file", "arguments": {"path": "task_cli.py"}, "result": {"ok": True, "path": "task_cli.py", "output": source}},
            {"name": "read_file", "arguments": {"path": "tests/test_task_cli.py"}, "result": {"ok": True, "path": "tests/test_task_cli.py", "output": test_source}},
        ]
        tool_calls: list[dict[str, object]] = []

        result = agent._try_spec_guided_repair(
            request_text="Add a stats command that prints counts by status and priority, add --priority filtering to list, update README.md, and keep tests green.",
            round_number=4,
            failed_run_test_result={"ok": False, "tool": "run_test", "summary": "test_list failed", "output": "test_list failed"},
            run_test_arguments={"command": command},
            successful_tool_results=successful_tool_results,
            satisfied_tool_names=set(),
            tool_calls_this_turn=tool_calls,
            allow_workspace_fallback=True,
        )

        self.assertIsNotNone(result)
        self.assertTrue(result.completed)
        self.assertIn("direct CLI proof passed", result.message)
        self.assertIn("add_parser('stats')", (root / "task_cli.py").read_text(encoding="utf-8"))
        self.assertIn("--priority", (root / "task_cli.py").read_text(encoding="utf-8"))
        readme = (root / "README.md").read_text(encoding="utf-8")
        self.assertIn("stats", readme)
        self.assertIn("--priority", readme)
        run_shell_commands = [
            str(call.get("arguments", {}).get("command", ""))
            for call in tool_calls
            if call.get("name") == "run_shell" and isinstance(call.get("arguments"), dict)
        ]
        self.assertTrue(any(" stats" in command for command in run_shell_commands))
        self.assertTrue(any("--priority" in command for command in run_shell_commands))

    def test_bookmark_archive_package_repair_updates_code_tests_docs_and_proof(self) -> None:
        root = self._workspace_scratch()
        (root / "bookmarks").mkdir()
        (root / "tests").mkdir()
        (root / "bookmarks" / "__init__.py").write_text("", encoding="utf-8")
        (root / "bookmarks" / "cli.py").write_text(
            "from __future__ import annotations\n\n"
            "import argparse\n"
            "from pathlib import Path\n\n"
            "from .store import add_bookmark, list_bookmarks\n\n\n"
            "def format_bookmark(item: dict[str, object]) -> str:\n"
            "    tags = ','.join(str(tag) for tag in item.get('tags', []))\n"
            "    return f\"{item['id']} | {item['title']} | {item['url']} | {tags}\"\n\n\n"
            "def main(argv: list[str] | None = None) -> int:\n"
            "    parser = argparse.ArgumentParser(prog='bookmarks')\n"
            "    parser.add_argument('--data', type=Path, default=Path('bookmarks.json'))\n"
            "    subparsers = parser.add_subparsers(dest='command', required=True)\n"
            "    list_parser = subparsers.add_parser('list')\n"
            "    list_parser.add_argument('--tag')\n"
            "    add_parser = subparsers.add_parser('add')\n"
            "    add_parser.add_argument('id')\n"
            "    add_parser.add_argument('title')\n"
            "    add_parser.add_argument('url')\n"
            "    add_parser.add_argument('--tag', action='append', default=[])\n"
            "    args = parser.parse_args(argv)\n"
            "    if args.command == 'list':\n"
            "        for item in list_bookmarks(args.data, tag=args.tag):\n"
            "            print(format_bookmark(item))\n"
            "        return 0\n"
            "    if args.command == 'add':\n"
            "        item = add_bookmark(args.data, args.id, args.title, args.url, args.tag)\n"
            "        print(format_bookmark(item))\n"
            "        return 0\n"
            "    raise SystemExit(f'unsupported command: {args.command}')\n\n\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(main())\n",
            encoding="utf-8",
        )
        (root / "bookmarks" / "store.py").write_text(
            "from __future__ import annotations\n\n"
            "import json\n"
            "from pathlib import Path\n"
            "from typing import Any\n\n"
            "DEFAULT_BOOKMARKS = [\n"
            "    {'id': 'docs', 'title': 'Docs', 'url': 'https://example.com/docs', 'tags': ['work']},\n"
            "    {'id': 'news', 'title': 'News', 'url': 'https://example.com/news', 'tags': ['reading']},\n"
            "]\n\n\n"
            "def load_bookmarks(path: Path) -> list[dict[str, Any]]:\n"
            "    if not path.exists():\n"
            "        return [dict(item) for item in DEFAULT_BOOKMARKS]\n"
            "    return json.loads(path.read_text(encoding='utf-8'))\n\n\n"
            "def save_bookmarks(path: Path, bookmarks: list[dict[str, Any]]) -> None:\n"
            "    path.write_text(json.dumps(bookmarks, indent=2, sort_keys=True) + '\\n', encoding='utf-8')\n\n\n"
            "def add_bookmark(path: Path, bookmark_id: str, title: str, url: str, tags: list[str]) -> dict[str, Any]:\n"
            "    bookmarks = load_bookmarks(path)\n"
            "    item = {'id': bookmark_id, 'title': title, 'url': url, 'tags': tags}\n"
            "    bookmarks.append(item)\n"
            "    save_bookmarks(path, bookmarks)\n"
            "    return item\n\n\n"
            "def list_bookmarks(path: Path, tag: str | None = None) -> list[dict[str, Any]]:\n"
            "    bookmarks = load_bookmarks(path)\n"
            "    if tag is None:\n"
            "        return bookmarks\n"
            "    return [item for item in bookmarks if tag in item.get('tags', [])]\n",
            encoding="utf-8",
        )
        (root / "tests" / "test_bookmarks.py").write_text(
            "import json\nimport subprocess\nimport sys\nimport tempfile\nimport unittest\nfrom pathlib import Path\n\n"
            "ROOT = Path(__file__).resolve().parents[1]\n\n"
            "def run_cli(*args: str) -> subprocess.CompletedProcess[str]:\n"
            "    return subprocess.run([sys.executable, '-m', 'bookmarks.cli', *args], cwd=ROOT, capture_output=True, text=True, check=False)\n\n"
            "class BookmarkTests(unittest.TestCase):\n"
            "    def test_lists_defaults(self) -> None:\n"
            "        with tempfile.TemporaryDirectory() as tmp:\n"
            "            result = run_cli('--data', str(Path(tmp) / 'bookmarks.json'), 'list')\n"
            "        self.assertEqual(result.returncode, 0, result.stderr)\n"
            "        self.assertIn('docs | Docs', result.stdout)\n",
            encoding="utf-8",
        )
        (root / "README.md").write_text("# Bookmarks CLI\n", encoding="utf-8")
        command = f"{sys.executable} -m unittest discover -s tests -v"
        tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
        agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)
        successful_tool_results: list[dict[str, object]] = []
        satisfied_tool_names: set[str] = set()
        tool_calls: list[dict[str, object]] = []

        result = agent._try_bookmark_archive_package_repair(
            request_text=(
                "Add an archive subcommand. The list command should hide archived bookmarks by default, "
                "list --all should include archived bookmarks. Add a rename subcommand that preserves archived "
                "status. Update README, add tests, run tests and prove archive and rename."
            ),
            round_number=2,
            successful_tool_results=successful_tool_results,  # type: ignore[arg-type]
            satisfied_tool_names=satisfied_tool_names,
            tool_calls_this_turn=tool_calls,
        )
        hidden = subprocess.run(
            [sys.executable, "-m", "bookmarks.cli", "--data", "rename-proof-bookmarks.json", "list"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        shown = subprocess.run(
            [sys.executable, "-m", "bookmarks.cli", "--data", "rename-proof-bookmarks.json", "list", "--all"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertIsNotNone(result)
        self.assertTrue(result.completed)
        self.assertIn("archive_bookmark", (root / "bookmarks" / "store.py").read_text(encoding="utf-8"))
        self.assertIn("rename_bookmark", (root / "bookmarks" / "store.py").read_text(encoding="utf-8"))
        self.assertIn("archive_parser", (root / "bookmarks" / "cli.py").read_text(encoding="utf-8"))
        self.assertIn("rename_parser", (root / "bookmarks" / "cli.py").read_text(encoding="utf-8"))
        self.assertIn("test_archive_hides_by_default_and_all_shows", (root / "tests" / "test_bookmarks.py").read_text(encoding="utf-8"))
        self.assertIn("test_rename_preserves_url_tags_and_archived_status", (root / "tests" / "test_bookmarks.py").read_text(encoding="utf-8"))
        self.assertIn("archive docs2", (root / "README.md").read_text(encoding="utf-8"))
        self.assertIn("rename docs2", (root / "README.md").read_text(encoding="utf-8"))
        self.assertNotIn("Renamed Docs", hidden.stdout)
        self.assertIn("docs | Renamed Docs", shown.stdout)

    def test_json_store_status_package_repair_updates_tasks_package(self) -> None:
        root = self._workspace_scratch()
        (root / "tasks").mkdir()
        (root / "tests").mkdir()
        (root / "tasks" / "__init__.py").write_text("", encoding="utf-8")
        (root / "tasks" / "store.py").write_text(
            "from __future__ import annotations\n\n"
            "import json\n"
            "from pathlib import Path\n"
            "from typing import Any\n\n"
            "DEFAULT_TASKS = [\n"
            "    {'id': 'docs', 'title': 'Write docs', 'priority': 'high', 'tags': ['work']},\n"
            "    {'id': 'shop', 'title': 'Buy milk', 'priority': 'low', 'tags': ['home']},\n"
            "]\n\n\n"
            "def load_tasks(path: Path) -> list[dict[str, Any]]:\n"
            "    if not path.exists():\n"
            "        return [{**item, 'tags': list(item.get('tags', []))} for item in DEFAULT_TASKS]\n"
            "    return json.loads(path.read_text(encoding='utf-8'))\n\n\n"
            "def save_tasks(path: Path, tasks: list[dict[str, Any]]) -> None:\n"
            "    path.write_text(json.dumps(tasks, indent=2, sort_keys=True) + '\\n', encoding='utf-8')\n\n\n"
            "def add_task(path: Path, task_id: str, title: str, priority: str, tags: list[str]) -> dict[str, Any]:\n"
            "    tasks = load_tasks(path)\n"
            "    item = {'id': task_id, 'title': title, 'priority': priority, 'tags': tags}\n"
            "    tasks.append(item)\n"
            "    save_tasks(path, tasks)\n"
            "    return item\n\n\n"
            "def list_tasks(path: Path, tag: str | None = None) -> list[dict[str, Any]]:\n"
            "    tasks = load_tasks(path)\n"
            "    if tag is None:\n"
            "        return tasks\n"
            "    return [item for item in tasks if tag in item.get('tags', [])]\n",
            encoding="utf-8",
        )
        (root / "tasks" / "cli.py").write_text(
            "from __future__ import annotations\n\n"
            "import argparse\n"
            "from pathlib import Path\n\n"
            "from .store import add_task, list_tasks\n\n\n"
            "def format_task(item: dict[str, object]) -> str:\n"
            "    tags = ','.join(str(tag) for tag in item.get('tags', []))\n"
            "    return f\"{item['id']} | {item['title']} | {item['priority']} | {tags}\"\n\n\n"
            "def main(argv: list[str] | None = None) -> int:\n"
            "    parser = argparse.ArgumentParser(prog='tasks')\n"
            "    parser.add_argument('--data', type=Path, default=Path('tasks.json'))\n"
            "    subparsers = parser.add_subparsers(dest='command', required=True)\n"
            "    list_parser = subparsers.add_parser('list')\n"
            "    list_parser.add_argument('--tag')\n"
            "    add_parser = subparsers.add_parser('add')\n"
            "    add_parser.add_argument('id')\n"
            "    add_parser.add_argument('title')\n"
            "    add_parser.add_argument('--priority', default='normal')\n"
            "    add_parser.add_argument('--tag', action='append', default=[])\n"
            "    args = parser.parse_args(argv)\n"
            "    if args.command == 'list':\n"
            "        for item in list_tasks(args.data, tag=args.tag):\n"
            "            print(format_task(item))\n"
            "        return 0\n"
            "    if args.command == 'add':\n"
            "        item = add_task(args.data, args.id, args.title, args.priority, args.tag)\n"
            "        print(format_task(item))\n"
            "        return 0\n"
            "    raise SystemExit(f'unsupported command: {args.command}')\n\n\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(main())\n",
            encoding="utf-8",
        )
        (root / "tests" / "test_tasks.py").write_text(
            "import json\nimport subprocess\nimport sys\nimport tempfile\nimport unittest\nfrom pathlib import Path\n\n"
            "ROOT = Path(__file__).resolve().parents[1]\n\n"
            "def run_cli(*args: str) -> subprocess.CompletedProcess[str]:\n"
            "    return subprocess.run([sys.executable, '-m', 'tasks.cli', *args], cwd=ROOT, capture_output=True, text=True, check=False)\n\n"
            "class TaskCliTests(unittest.TestCase):\n"
            "    def test_lists_defaults(self) -> None:\n"
            "        with tempfile.TemporaryDirectory() as tmp:\n"
            "            result = run_cli('--data', str(Path(tmp) / 'tasks.json'), 'list')\n"
            "        self.assertEqual(result.returncode, 0, result.stderr)\n"
            "        self.assertIn('docs | Write docs', result.stdout)\n",
            encoding="utf-8",
        )
        (root / "README.md").write_text("# Tasks CLI\n", encoding="utf-8")
        command = f"{sys.executable} -m unittest discover -s tests -v"
        tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
        agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)
        successful_tool_results: list[dict[str, object]] = []
        satisfied_tool_names: set[str] = set()
        tool_calls: list[dict[str, object]] = []

        result = agent._try_json_store_status_package_repair(
            request_text=(
                "Add a complete subcommand to this tasks CLI. It should mark a task completed in the JSON data by id. "
                "The list command should hide completed tasks by default, and list --all should include completed tasks. "
                "Update README, add tests, run tests and prove the behavior with a shell command."
            ),
            round_number=2,
            successful_tool_results=successful_tool_results,  # type: ignore[arg-type]
            satisfied_tool_names=satisfied_tool_names,
            tool_calls_this_turn=tool_calls,
        )
        hidden = subprocess.run(
            [sys.executable, "-m", "tasks.cli", "--data", "complete-proof-tasks.json", "list"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        shown = subprocess.run(
            [sys.executable, "-m", "tasks.cli", "--data", "complete-proof-tasks.json", "list", "--all"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertIsNotNone(result)
        self.assertTrue(result.completed)
        self.assertIn("complete_task", (root / "tasks" / "store.py").read_text(encoding="utf-8"))
        self.assertIn("complete_parser", (root / "tasks" / "cli.py").read_text(encoding="utf-8"))
        self.assertIn("test_complete_hides_by_default_and_all_shows", (root / "tests" / "test_tasks.py").read_text(encoding="utf-8"))
        self.assertIn("complete docs", (root / "README.md").read_text(encoding="utf-8"))
        self.assertNotIn("Write docs", hidden.stdout)
        self.assertIn("Write docs", shown.stdout)

    def test_json_store_field_update_package_repair_updates_inventory_price(self) -> None:
        root = self._workspace_scratch()
        (root / "inventory").mkdir()
        (root / "tests").mkdir()
        (root / "inventory" / "__init__.py").write_text("", encoding="utf-8")
        (root / "inventory" / "store.py").write_text(
            "from __future__ import annotations\n\n"
            "import json\n"
            "from pathlib import Path\n"
            "from typing import Any\n\n"
            "DEFAULT_ITEMS = [\n"
            "    {'sku': 'pen', 'name': 'Gel Pen', 'price': 2.50, 'tags': ['office']},\n"
            "    {'sku': 'mug', 'name': 'Coffee Mug', 'price': 8.00, 'tags': ['kitchen']},\n"
            "]\n\n\n"
            "def load_items(path: Path) -> list[dict[str, Any]]:\n"
            "    if not path.exists():\n"
            "        return [{**item, 'tags': list(item.get('tags', []))} for item in DEFAULT_ITEMS]\n"
            "    return json.loads(path.read_text(encoding='utf-8'))\n\n\n"
            "def save_items(path: Path, items: list[dict[str, Any]]) -> None:\n"
            "    path.write_text(json.dumps(items, indent=2, sort_keys=True) + '\\n', encoding='utf-8')\n\n\n"
            "def add_item(path: Path, sku: str, name: str, price: float, tags: list[str]) -> dict[str, Any]:\n"
            "    items = load_items(path)\n"
            "    item = {'sku': sku, 'name': name, 'price': price, 'tags': tags}\n"
            "    items.append(item)\n"
            "    save_items(path, items)\n"
            "    return item\n\n\n"
            "def list_items(path: Path, tag: str | None = None) -> list[dict[str, Any]]:\n"
            "    items = load_items(path)\n"
            "    if tag is None:\n"
            "        return items\n"
            "    return [item for item in items if tag in item.get('tags', [])]\n",
            encoding="utf-8",
        )
        (root / "inventory" / "cli.py").write_text(
            "from __future__ import annotations\n\n"
            "import argparse\n"
            "from pathlib import Path\n\n"
            "from .store import add_item, list_items\n\n\n"
            "def format_item(item: dict[str, object]) -> str:\n"
            "    tags = ','.join(str(tag) for tag in item.get('tags', []))\n"
            "    return f\"{item['sku']} | {item['name']} | ${float(item['price']):.2f} | {tags}\"\n\n\n"
            "def main(argv: list[str] | None = None) -> int:\n"
            "    parser = argparse.ArgumentParser(prog='inventory')\n"
            "    parser.add_argument('--data', type=Path, default=Path('inventory.json'))\n"
            "    subparsers = parser.add_subparsers(dest='command', required=True)\n"
            "    list_parser = subparsers.add_parser('list')\n"
            "    list_parser.add_argument('--tag')\n"
            "    add_parser = subparsers.add_parser('add')\n"
            "    add_parser.add_argument('sku')\n"
            "    add_parser.add_argument('name')\n"
            "    add_parser.add_argument('price', type=float)\n"
            "    add_parser.add_argument('--tag', action='append', default=[])\n"
            "    args = parser.parse_args(argv)\n"
            "    if args.command == 'list':\n"
            "        for item in list_items(args.data, tag=args.tag):\n"
            "            print(format_item(item))\n"
            "        return 0\n"
            "    if args.command == 'add':\n"
            "        item = add_item(args.data, args.sku, args.name, args.price, args.tag)\n"
            "        print(format_item(item))\n"
            "        return 0\n"
            "    raise SystemExit(f'unsupported command: {args.command}')\n\n\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(main())\n",
            encoding="utf-8",
        )
        (root / "tests" / "test_inventory.py").write_text(
            "import json\nimport subprocess\nimport sys\nimport tempfile\nimport unittest\nfrom pathlib import Path\n\n"
            "ROOT = Path(__file__).resolve().parents[1]\n\n"
            "def run_cli(*args: str) -> subprocess.CompletedProcess[str]:\n"
            "    return subprocess.run([sys.executable, '-m', 'inventory.cli', *args], cwd=ROOT, capture_output=True, text=True, check=False)\n\n"
            "class InventoryCliTests(unittest.TestCase):\n"
            "    def test_lists_defaults(self) -> None:\n"
            "        with tempfile.TemporaryDirectory() as tmp:\n"
            "            result = run_cli('--data', str(Path(tmp) / 'inventory.json'), 'list')\n"
            "        self.assertEqual(result.returncode, 0, result.stderr)\n"
            "        self.assertIn('pen | Gel Pen | $2.50', result.stdout)\n",
            encoding="utf-8",
        )
        (root / "README.md").write_text("# Inventory CLI\n", encoding="utf-8")
        command = f"{sys.executable} -m unittest discover -s tests -v"
        tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
        agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)
        successful_tool_results: list[dict[str, object]] = []
        satisfied_tool_names: set[str] = set()
        tool_calls: list[dict[str, object]] = []

        result = agent._try_json_store_field_update_package_repair(
            request_text=(
                "Add a set-price subcommand to this inventory CLI. It should update an item's price "
                "in the JSON data by sku while preserving its name and tags. Update README, add tests, "
                "run tests and prove the behavior with a shell command."
            ),
            round_number=2,
            successful_tool_results=successful_tool_results,  # type: ignore[arg-type]
            satisfied_tool_names=satisfied_tool_names,
            tool_calls_this_turn=tool_calls,
        )
        listed = subprocess.run(
            [sys.executable, "-m", "inventory.cli", "--data", "set-price-proof-items.json", "list"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertIsNotNone(result)
        self.assertTrue(result.completed)
        self.assertIn("set_price_item", (root / "inventory" / "store.py").read_text(encoding="utf-8"))
        cli_after = (root / "inventory" / "cli.py").read_text(encoding="utf-8")
        self.assertIn("set-price", cli_after)
        self.assertIn('default=Path("inventory.json")', cli_after)
        self.assertIn("test_set_price_updates_price_and_preserves_existing_fields", (root / "tests" / "test_inventory.py").read_text(encoding="utf-8"))
        readme_after = (root / "README.md").read_text(encoding="utf-8")
        self.assertIn("--data inventory.json", readme_after)
        self.assertIn("set-price pen 3.75", readme_after)
        self.assertIn("pen | Gel Pen | $3.75 | office", listed.stdout)

    def test_invoice_discount_cap_package_repair_updates_api_tests_docs_and_proof(self) -> None:
        root = self._workspace_scratch()
        (root / "invoice").mkdir()
        (root / "tests").mkdir()
        (root / "invoice" / "__init__.py").write_text(
            "from .calculator import InvoiceLine, calculate_invoice\n\n"
            "__all__ = [\"InvoiceLine\", \"calculate_invoice\"]\n",
            encoding="utf-8",
        )
        (root / "invoice" / "rules.py").write_text(
            "from __future__ import annotations\n\n\n"
            "def category_discount_rate(category: str) -> float:\n"
            "    rates = {\n"
            "        \"books\": 0.10,\n"
            "        \"software\": 0.05,\n"
            "        \"hardware\": 0.00,\n"
            "    }\n"
            "    return rates.get(category, 0.0)\n\n\n"
            "def loyalty_discount_rate(customer_tier: str) -> float:\n"
            "    if customer_tier == \"gold\":\n"
            "        return 0.05\n"
            "    return 0.0\n",
            encoding="utf-8",
        )
        (root / "invoice" / "calculator.py").write_text(
            "from __future__ import annotations\n\n"
            "from dataclasses import dataclass\n\n"
            "from .rules import category_discount_rate, loyalty_discount_rate\n\n\n"
            "@dataclass(frozen=True)\n"
            "class InvoiceLine:\n"
            "    sku: str\n"
            "    category: str\n"
            "    unit_price: float\n"
            "    quantity: int\n\n\n"
            "def calculate_invoice(lines: list[InvoiceLine], customer_tier: str = \"standard\") -> dict[str, float]:\n"
            "    subtotal = sum(line.unit_price * line.quantity for line in lines)\n"
            "    category_discount = sum(\n"
            "        line.unit_price * line.quantity * category_discount_rate(line.category)\n"
            "        for line in lines\n"
            "    )\n"
            "    loyalty_discount = subtotal * loyalty_discount_rate(customer_tier)\n"
            "    discount = round(category_discount + loyalty_discount, 2)\n"
            "    total = round(subtotal - discount, 2)\n"
            "    return {\"subtotal\": round(subtotal, 2), \"discount\": discount, \"total\": total}\n",
            encoding="utf-8",
        )
        (root / "tests" / "test_invoice.py").write_text(
            "import unittest\n\n"
            "from invoice import InvoiceLine, calculate_invoice\n\n\n"
            "class InvoiceTests(unittest.TestCase):\n"
            "    def test_category_discount(self) -> None:\n"
            "        result = calculate_invoice([InvoiceLine(\"book-1\", \"books\", 20.0, 2)])\n"
            "        self.assertEqual(result, {\"subtotal\": 40.0, \"discount\": 4.0, \"total\": 36.0})\n\n"
            "    def test_gold_loyalty_discount(self) -> None:\n"
            "        result = calculate_invoice([InvoiceLine(\"app\", \"software\", 100.0, 1)], customer_tier=\"gold\")\n"
            "        self.assertEqual(result, {\"subtotal\": 100.0, \"discount\": 10.0, \"total\": 90.0})\n\n\n"
            "if __name__ == \"__main__\":\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )
        (root / "README.md").write_text(
            "# Invoice Discount API\n\nGold customers receive 5% off the subtotal.\n",
            encoding="utf-8",
        )
        command = f"{sys.executable} -m unittest discover -s tests -v"
        tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
        agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)
        successful_tool_results: list[dict[str, object]] = []
        satisfied_tool_names: set[str] = set()
        tool_calls: list[dict[str, object]] = []
        request_text = (
            "Add a vip customer tier to this invoice discount API. VIP customers should receive a 12% "
            "loyalty discount, but total combined discounts must be capped at 20% of the subtotal. "
            "Preserve the existing category discounts and gold behavior. Update README with the new vip tier "
            "and cap. Add tests for vip discount and for the 20% cap when vip is combined with category discounts. "
            "Run the tests and prove the behavior with a shell command."
        )
        obligations = agent._derive_request_obligations(
            request_text=request_text,
            required_tool_names=set(),
            required_mutation_paths=set(),
            code_mutation_required=True,
            test_run_required=True,
        )

        result = agent._try_invoice_discount_cap_package_repair(
            request_text=request_text,
            round_number=2,
            request_obligations=obligations,
            forbidden_tool_names=set(),
            successful_tool_results=successful_tool_results,  # type: ignore[arg-type]
            satisfied_tool_names=satisfied_tool_names,
            tool_calls_this_turn=tool_calls,
        )

        self.assertIsNotNone(result)
        self.assertTrue(result.completed)
        self.assertIn("customer_tier == \"vip\"", (root / "invoice" / "rules.py").read_text(encoding="utf-8"))
        self.assertIn("subtotal * 0.20", (root / "invoice" / "calculator.py").read_text(encoding="utf-8"))
        self.assertIn("test_vip_loyalty_discount", (root / "tests" / "test_invoice.py").read_text(encoding="utf-8"))
        self.assertIn("VIP customers receive 12%", (root / "README.md").read_text(encoding="utf-8"))
        proof = subprocess.run(
            [
                sys.executable,
                "-c",
                "from invoice import InvoiceLine, calculate_invoice; print(calculate_invoice([InvoiceLine('book','books',100.0,1)], customer_tier='vip'))",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proof.returncode, 0, proof.stderr)
        self.assertIn("'discount': 20.0", proof.stdout)

    def test_config_env_override_package_repair_updates_settings_tests_docs_and_proof(self) -> None:
        root = self._workspace_scratch()
        (root / "app").mkdir()
        (root / "tests").mkdir()
        (root / "app" / "__init__.py").write_text(
            "from .settings import AppSettings, load_settings\n\n"
            "__all__ = [\"AppSettings\", \"load_settings\"]\n",
            encoding="utf-8",
        )
        (root / "app" / "settings.py").write_text(
            "from __future__ import annotations\n\n"
            "import json\n"
            "from dataclasses import dataclass\n"
            "from pathlib import Path\n\n\n"
            "@dataclass(frozen=True)\n"
            "class AppSettings:\n"
            "    host: str\n"
            "    port: int\n"
            "    debug: bool\n\n\n"
            "def _as_bool(value: object) -> bool:\n"
            "    if isinstance(value, bool):\n"
            "        return value\n"
            "    if isinstance(value, str):\n"
            "        return value.lower() in {\"1\", \"true\", \"yes\", \"on\"}\n"
            "    return bool(value)\n\n\n"
            "def load_settings(path: Path) -> AppSettings:\n"
            "    data = json.loads(path.read_text(encoding=\"utf-8\"))\n"
            "    port = int(data.get(\"port\", 8000))\n"
            "    if port < 1 or port > 65535:\n"
            "        raise ValueError(\"port must be between 1 and 65535\")\n"
            "    return AppSettings(host=str(data.get(\"host\", \"127.0.0.1\")), port=port, debug=_as_bool(data.get(\"debug\", False)))\n",
            encoding="utf-8",
        )
        (root / "tests" / "test_settings.py").write_text(
            "import json\nimport tempfile\nimport unittest\nfrom pathlib import Path\n\n"
            "from app import AppSettings, load_settings\n\n\n"
            "class SettingsTests(unittest.TestCase):\n"
            "    def write_config(self, data: dict[str, object]) -> Path:\n"
            "        tmp = tempfile.TemporaryDirectory()\n"
            "        self.addCleanup(tmp.cleanup)\n"
            "        path = Path(tmp.name) / \"settings.json\"\n"
            "        path.write_text(json.dumps(data), encoding=\"utf-8\")\n"
            "        return path\n\n"
            "    def test_loads_json_settings(self) -> None:\n"
            "        path = self.write_config({\"host\": \"0.0.0.0\", \"port\": 9000, \"debug\": True})\n"
            "        self.assertEqual(load_settings(path), AppSettings(host=\"0.0.0.0\", port=9000, debug=True))\n\n"
            "    def test_defaults_missing_values(self) -> None:\n"
            "        path = self.write_config({})\n"
            "        self.assertEqual(load_settings(path), AppSettings(host=\"127.0.0.1\", port=8000, debug=False))\n\n"
            "    def test_rejects_invalid_port(self) -> None:\n"
            "        path = self.write_config({\"port\": 70000})\n"
            "        with self.assertRaisesRegex(ValueError, \"port must be\"):\n"
            "            load_settings(path)\n\n\n"
            "if __name__ == \"__main__\":\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )
        (root / "README.md").write_text("# Config Loader\n", encoding="utf-8")
        command = f"{sys.executable} -m unittest discover -s tests -v"
        tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
        agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)
        successful_tool_results: list[dict[str, object]] = []
        satisfied_tool_names: set[str] = set()
        tool_calls: list[dict[str, object]] = []
        request_text = (
            "Add environment variable overrides to this config loader. APP_HOST, APP_PORT, and APP_DEBUG "
            "should override the JSON file values when present. Keep the existing port validation for both JSON "
            "and environment values. Update README with the environment variable names. Add tests for host, port, "
            "and debug overrides plus invalid APP_PORT. Run the tests and prove the behavior with a shell command."
        )
        obligations = agent._derive_request_obligations(
            request_text=request_text,
            required_tool_names=set(),
            required_mutation_paths=set(),
            code_mutation_required=True,
            test_run_required=True,
        )

        result = agent._try_config_env_override_package_repair(
            request_text=request_text,
            round_number=2,
            request_obligations=obligations,
            forbidden_tool_names=set(),
            successful_tool_results=successful_tool_results,  # type: ignore[arg-type]
            satisfied_tool_names=satisfied_tool_names,
            tool_calls_this_turn=tool_calls,
        )

        self.assertIsNotNone(result)
        self.assertTrue(result.completed)
        settings_after = (root / "app" / "settings.py").read_text(encoding="utf-8")
        tests_after = (root / "tests" / "test_settings.py").read_text(encoding="utf-8")
        readme_after = (root / "README.md").read_text(encoding="utf-8")
        self.assertIn("import os", settings_after)
        self.assertIn("APP_HOST", settings_after)
        self.assertIn("test_environment_overrides_host_port_and_debug", tests_after)
        self.assertIn("test_invalid_environment_port_is_rejected", tests_after)
        self.assertIn("APP_PORT", readme_after)
        proof = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json, os, tempfile; from pathlib import Path; from app import load_settings; "
                    "d=tempfile.TemporaryDirectory(); p=Path(d.name)/'settings.json'; "
                    "p.write_text(json.dumps({'host':'0.0.0.0','port':9000,'debug':False}), encoding='utf-8'); "
                    "os.environ.update({'APP_HOST':'localhost','APP_PORT':'9100','APP_DEBUG':'true'}); "
                    "print(load_settings(p))"
                ),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proof.returncode, 0, proof.stderr)
        self.assertIn("host='localhost'", proof.stdout)
        self.assertIn("port=9100", proof.stdout)
        self.assertIn("debug=True", proof.stdout)

    def test_catalog_patch_tags_package_repair_updates_router_tests_docs_and_proof(self) -> None:
        root = self._workspace_scratch()
        (root / "catalog").mkdir()
        (root / "tests").mkdir()
        (root / "catalog" / "__init__.py").write_text(
            "from .router import Response, route_request\n\n"
            "__all__ = [\"Response\", \"route_request\"]\n",
            encoding="utf-8",
        )
        (root / "catalog" / "store.py").write_text(
            "from __future__ import annotations\n\n"
            "from copy import deepcopy\n"
            "from typing import Any\n\n"
            "DEFAULT_ITEMS = {\n"
            "    \"pen\": {\"sku\": \"pen\", \"name\": \"Gel Pen\", \"tags\": [\"office\"]},\n"
            "    \"mug\": {\"sku\": \"mug\", \"name\": \"Coffee Mug\", \"tags\": [\"kitchen\"]},\n"
            "}\n\n\n"
            "def new_store() -> dict[str, dict[str, Any]]:\n"
            "    return deepcopy(DEFAULT_ITEMS)\n\n\n"
            "def get_item(store: dict[str, dict[str, Any]], sku: str) -> dict[str, Any] | None:\n"
            "    item = store.get(sku)\n"
            "    return deepcopy(item) if item is not None else None\n\n\n"
            "def list_items(store: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:\n"
            "    return [deepcopy(item) for item in store.values()]\n",
            encoding="utf-8",
        )
        (root / "catalog" / "router.py").write_text(
            "from __future__ import annotations\n\n"
            "import json\n"
            "from dataclasses import dataclass\n"
            "from typing import Any\n\n"
            "from .store import get_item, list_items, new_store\n\n\n"
            "@dataclass(frozen=True)\n"
            "class Response:\n"
            "    status: int\n"
            "    body: dict[str, Any]\n\n\n"
            "def route_request(method: str, path: str, body: str | None = None, *, store: dict[str, dict[str, Any]] | None = None) -> Response:\n"
            "    active_store = new_store() if store is None else store\n"
            "    method = method.upper()\n"
            "    if method == \"GET\" and path == \"/health\":\n"
            "        return Response(200, {\"ok\": True})\n"
            "    if method == \"GET\" and path == \"/items\":\n"
            "        return Response(200, {\"items\": list_items(active_store)})\n"
            "    if method == \"GET\" and path.startswith(\"/items/\"):\n"
            "        sku = path.removeprefix(\"/items/\")\n"
            "        item = get_item(active_store, sku)\n"
            "        if item is None:\n"
            "            return Response(404, {\"error\": \"item not found\"})\n"
            "        return Response(200, {\"item\": item})\n"
            "    return Response(404, {\"error\": \"not found\"})\n",
            encoding="utf-8",
        )
        (root / "tests" / "test_router.py").write_text(
            "import unittest\n\n"
            "from catalog import route_request\n"
            "from catalog.store import new_store\n\n\n"
            "class RouterTests(unittest.TestCase):\n"
            "    def test_health(self) -> None:\n"
            "        response = route_request(\"GET\", \"/health\")\n"
            "        self.assertEqual(response.status, 200)\n"
            "        self.assertEqual(response.body, {\"ok\": True})\n\n"
            "    def test_get_item(self) -> None:\n"
            "        response = route_request(\"GET\", \"/items/pen\")\n"
            "        self.assertEqual(response.status, 200)\n"
            "        self.assertEqual(response.body[\"item\"][\"name\"], \"Gel Pen\")\n\n\n"
            "if __name__ == \"__main__\":\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )
        (root / "README.md").write_text("# Catalog Router\n\n- `GET /items/{sku}`\n", encoding="utf-8")
        command = f"{sys.executable} -m unittest discover -s tests -v"
        tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
        agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)
        successful_tool_results: list[dict[str, object]] = []
        satisfied_tool_names: set[str] = set()
        tool_calls: list[dict[str, object]] = []
        request_text = (
            "Add a PATCH /items/{sku}/tags route to this catalog router. The request body should be JSON "
            "with a tags array of strings. It should replace the item's tags in the supplied store and return "
            "the updated item. Unknown items should return 404, and invalid JSON or non-string tags should return 400. "
            "Update README with the new route. Add tests for success, missing item, and invalid tags. Run the tests "
            "and prove the behavior with a shell command."
        )
        obligations = agent._derive_request_obligations(
            request_text=request_text,
            required_tool_names=set(),
            required_mutation_paths=set(),
            code_mutation_required=True,
            test_run_required=True,
        )

        result = agent._try_catalog_patch_tags_package_repair(
            request_text=request_text,
            round_number=2,
            request_obligations=obligations,
            forbidden_tool_names=set(),
            successful_tool_results=successful_tool_results,  # type: ignore[arg-type]
            satisfied_tool_names=satisfied_tool_names,
            tool_calls_this_turn=tool_calls,
        )

        self.assertIsNotNone(result)
        self.assertTrue(result.completed)
        router_after = (root / "catalog" / "router.py").read_text(encoding="utf-8")
        tests_after = (root / "tests" / "test_router.py").read_text(encoding="utf-8")
        readme_after = (root / "README.md").read_text(encoding="utf-8")
        self.assertIn('method == "PATCH"', router_after)
        self.assertIn("replace_item_tags", router_after)
        self.assertIn("test_patch_tags_replaces_tags_in_supplied_store", tests_after)
        self.assertIn("PATCH /items/{sku}/tags", readme_after)
        proof = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from catalog import route_request; from catalog.store import new_store; "
                    "s=new_store(); r=route_request('PATCH','/items/pen/tags','{\"tags\":[\"office\",\"favorite\"]}',store=s); "
                    "print(r); print(s['pen']['tags'])"
                ),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proof.returncode, 0, proof.stderr)
        self.assertIn("favorite", proof.stdout)

    def test_final_chance_test_success_does_not_complete_unproven_obligations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pass_command = subprocess.list2cmdline([sys.executable, "-c", "print('OK')"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": "def f():\n    return 1\n"}}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=pass_command)
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=1)

            result = agent.handle_user("Edit app.py, add tests for it, run tests, and prove it with a shell command.")

        self.assertFalse(result.completed)
        self.assertIn("final-chance tests passed but requested deliverables remain unproven", result.message)
        self.assertIn("add or update the requested tests", result.message)
        self.assertIn("prove the requested behavior with a shell command", result.message)
        self.assertEqual(tools.execute_counts.get("write_file"), 1)
        self.assertEqual(tools.execute_counts.get("run_test"), 1)

    def test_pending_repair_spec_fails_closed_without_repeated_auto_lint(self) -> None:
        root = self._workspace_scratch()
        (root / "app.py").write_text("import os\n\n\ndef value() -> str:\n    return os.name\n", encoding="utf-8")
        client = FakeClient(
            [
                json.dumps(
                    {
                        "type": "tool",
                        "name": "write_file",
                        "arguments": {
                            "path": "app.py",
                            "content": "import os\nimport os\n\n\ndef value() -> str:\n    return os.name\n",
                        },
                    }
                ),
                '{"type":"tool","name":"read_file","arguments":{"path":"app.py"}}',
                '{"type":"final","message":"still repairing"}',
                '{"type":"final","message":"still repairing"}',
                '{"type":"final","message":"still repairing"}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto", test_command=f"{sys.executable} -m unittest discover")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=5)

        result = agent.handle_user("Update app.py and keep validation green.")

        self.assertFalse(result.completed)
        self.assertEqual(tools.execute_counts.get("write_file"), 1)
        self.assertEqual(tools.execute_counts.get("lint_typecheck"), 1)
        guards = [event for event in agent.events if event.get("type") == "controller_guard"]
        self.assertTrue(any(event.get("guard") == "post-edit-validation" for event in guards))
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Do not rerun validators until you make the broader repair", feedback)

    def test_unproven_feature_obligations_fail_before_final_verifier(self) -> None:
        root = self._workspace_scratch()
        (root / "app.py").write_text("def main():\n    return 0\n", encoding="utf-8")
        (root / "README.md").write_text("# App\n", encoding="utf-8")
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "app.py"}}),
                json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": "def main():\n    return 0\n"}}),
                json.dumps({"type": "tool", "name": "run_test", "arguments": {}}),
                *[json.dumps({"type": "final", "message": "Implemented archive and --all."}) for _ in range(10)],
            ]
        )
        tools = CountingToolExecutor(
            root,
            approval_mode="auto",
            test_command=subprocess.list2cmdline([sys.executable, "-c", "print('OK')"]),
        )
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

        result = agent.handle_user("Add an archive command and --all flag. Update README and run tests.")

        self.assertFalse(result.completed)
        self.assertIn("requested deliverables remain unproven before final verification", result.message)
        self.assertIn('prove the "archive" command exists', result.message)
        verifier_calls = [
            call
            for call in client.calls
            if call["messages"] and str(call["messages"][0]["content"]).startswith("You are a grounded final verifier")
        ]
        self.assertEqual(verifier_calls, [])

    def test_syntax_bad_mutation_invokes_spec_guided_repair_with_workspace_fallback(self) -> None:
        root = self._workspace_scratch()
        (root / "tests").mkdir()
        (root / "app.py").write_text("def value() -> str:\n    return 'ok'\n", encoding="utf-8")
        (root / "tests" / "test_app.py").write_text(
            "import unittest\nfrom app import value\n\n"
            "class AppTests(unittest.TestCase):\n"
            "    def test_value(self):\n"
            "        self.assertEqual(value(), 'fixed')\n",
            encoding="utf-8",
        )
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"app.py"}}',
                '{"type":"tool","name":"write_file","arguments":{"path":"app.py","content":"def value() -> str:\\n    return \\"unterminated\\n"}}',
                '{"type":"final","message":"done"}',
            ]
        )
        tools = ToolExecutor(root, approval_mode="auto", test_command=f"{sys.executable} -m unittest discover -s tests")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)
        calls: list[dict[str, object]] = []

        def fake_spec_guided_repair(**kwargs: object) -> AgentResult | None:
            calls.append(dict(kwargs))
            return AgentResult(message="syntax spec repair called", rounds=int(kwargs["round_number"]), completed=False)

        agent._try_spec_guided_repair = fake_spec_guided_repair  # type: ignore[method-assign]

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Fix app.py and run tests.")

        self.assertFalse(result.completed)
        self.assertEqual(result.message, "syntax spec repair called")
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].get("allow_workspace_fallback"))
        self.assertIn("Post-edit syntax check failed", calls[0]["failed_run_test_result"]["summary"])

    def test_trajectory_final_chance_validation_selects_tests_without_explicit_test_request(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "src" / "pricing.py").write_text("def cart_total(prices):\n    return 0\n", encoding="utf-8")
        (root / "tests" / "test_pricing.py").write_text(
            "import unittest\n"
            "from src.pricing import cart_total\n\n"
            "class PricingTests(unittest.TestCase):\n"
            "    def test_cart_total(self):\n"
            "        self.assertEqual(cart_total([2, 3]), 5)\n",
            encoding="utf-8",
        )
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"src/pricing.py"}}',
                '{"type":"tool","name":"replace_in_file","arguments":{"path":"src/pricing.py","old":"return 0","new":"return sum(prices)"}}',
            ]
        )
        tools = ToolExecutor(root, approval_mode="auto", test_command=f"{sys.executable} -m unittest discover -s tests")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=2)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Fix src/pricing.py.")

        self.assertTrue(result.completed)
        self.assertEqual(result.message, "Ran validation after the latest edit: passed.")
        tool_events = [event for event in agent.events if event.get("type") == "tool_call"]
        tool_names = [event.get("name") for event in tool_events]
        self.assertIn("select_tests", tool_names)
        run_tests = [event for event in tool_events if event.get("name") == "run_test"]
        self.assertTrue(run_tests)
        self.assertIn("test_pricing.py", str(run_tests[-1].get("arguments", {}).get("command", "")))

    def test_trajectory_final_chance_validation_falls_back_to_default_test_command_when_no_targeted_tests(self) -> None:
        root = self._workspace_scratch()
        (root / "app.py").write_text("def value():\n    return 0\n", encoding="utf-8")
        pass_command = subprocess.list2cmdline([sys.executable, "-c", "print('OK')"])
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"app.py"}}',
                '{"type":"tool","name":"replace_in_file","arguments":{"path":"app.py","old":"return 0","new":"return 1"}}',
            ]
        )
        tools = EmptySelectTestsToolExecutor(root, approval_mode="auto", test_command=pass_command)
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=2)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Fix app.py.")

        self.assertTrue(result.completed)
        self.assertEqual(result.message, "Ran validation after the latest edit: passed.")
        self.assertEqual(tools.execute_counts.get("select_tests"), 1)
        self.assertEqual(tools.execute_counts.get("run_test"), 1)
        run_tests = [event for event in agent.events if event.get("type") == "tool_call" and event.get("name") == "run_test"]
        self.assertTrue(run_tests)
        self.assertEqual(run_tests[-1].get("arguments", {}).get("command"), pass_command)

    def test_trajectory_final_chance_validation_discovers_repo_test_command_after_empty_targeted_selection(self) -> None:
        root = self._workspace_scratch()
        (root / "app.py").write_text("def value():\n    return 0\n", encoding="utf-8")
        (root / "tests").mkdir()
        (root / "tests" / "test_app.py").write_text(
            "import unittest\n\nclass AppTests(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n",
            encoding="utf-8",
        )
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"app.py"}}',
                '{"type":"tool","name":"replace_in_file","arguments":{"path":"app.py","old":"return 0","new":"return 1"}}',
            ]
        )
        tools = EmptySelectTestsToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=2)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Fix app.py.")

        self.assertTrue(result.completed)
        self.assertEqual(result.message, "Ran validation after the latest edit: passed.")
        self.assertEqual(tools.execute_counts.get("select_tests"), 1)
        self.assertEqual(tools.execute_counts.get("discover_validators"), 1)
        self.assertEqual(tools.execute_counts.get("run_test"), 1)
        run_tests = [event for event in agent.events if event.get("type") == "tool_call" and event.get("name") == "run_test"]
        self.assertTrue(run_tests)
        command = str(run_tests[-1].get("arguments", {}).get("command", ""))
        self.assertIn("unittest", command)
        self.assertIn("discover", command)
        self.assertIn("tests", command)

    def test_trajectory_final_chance_validation_avoids_rediscovery_after_successful_lint(self) -> None:
        root = self._workspace_scratch()
        (root / "app.py").write_text("def value():\n    return 0\n", encoding="utf-8")
        lint_command = subprocess.list2cmdline([sys.executable, "-c", "print('lint ok')"])
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"app.py"}}',
                '{"type":"tool","name":"replace_in_file","arguments":{"path":"app.py","old":"return 0","new":"return 1"}}',
            ]
        )
        tools = EmptySelectTestsLintFallbackToolExecutor(root, approval_mode="auto", fallback_command=lint_command)
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=2)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Fix app.py.")

        self.assertTrue(result.completed)
        self.assertEqual(result.message, "Ran validation after the latest edit: passed.")
        self.assertEqual(tools.execute_counts.get("select_tests"), 1)
        self.assertEqual(tools.execute_counts.get("lint_typecheck"), 1)
        self.assertEqual(tools.execute_counts.get("contract_check"), 1)
        self.assertEqual(tools.execute_counts.get("discover_validators"), 1)
        self.assertIsNone(tools.execute_counts.get("run_test"))
        lint_calls = [event for event in agent.events if event.get("type") == "tool_call" and event.get("name") == "lint_typecheck"]
        self.assertTrue(lint_calls)
        self.assertIsNone(lint_calls[-1].get("arguments", {}).get("command"))

    def test_post_edit_verification_rejects_docs_only_feature_completion_until_code_proof_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def add(left, right):\n    return left + right\n", encoding="utf-8")
            (root / "README.md").write_text("Usage\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "test_app.py").write_text("import unittest\n\n\nclass AppTests(unittest.TestCase):\n    def test_placeholder(self) -> None:\n        self.assertTrue(True)\n\n\nif __name__ == '__main__':\n    unittest.main()\n", encoding="utf-8")
            client = FakeClient(
                [
                    '{"type":"tool","name":"write_file","arguments":{"path":"README.md","content":"Use the stats command.\\n"}}',
                    *['{"type":"final","message":"Added the stats command and updated README."}' for _ in range(10)],
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=f"{sys.executable} -m unittest discover -s tests -v")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

            result = agent.handle_user("Add a stats command to app.py and update README.md.")

        self.assertFalse(result.completed)
        self.assertIn("requested deliverables", result.message)
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertIn("write_file", tool_calls)
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn('prove the "stats" command exists', feedback)

    def test_accurate_missing_path_summary_is_allowed_for_exists_question(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "missing.py"}}),
                json.dumps({"type": "final", "message": "missing.py does not exist in the workspace."}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=3)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Read missing.py and tell me whether it exists.")

        self.assertTrue(result.completed)
        self.assertIn("does not exist", result.message.lower())
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertNotIn("This request requires real tool use in this turn.", feedback)

    def test_false_exists_claim_is_blocked_after_missing_path_failure(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "missing.py"}}),
                json.dumps({"type": "final", "message": "missing.py exists."}),
                json.dumps({"type": "final", "message": "missing.py does not exist in the workspace."}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=4)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Read missing.py and tell me whether it exists.")

        self.assertTrue(result.completed)
        self.assertIn("does not exist", result.message.lower())
        controller_guards = [
            event
            for event in agent.events
            if event.get("type") == "controller_guard" and event.get("guard") == "path-exists-final-claim"
        ]
        self.assertEqual(len(controller_guards), 1)
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("latest path lookup failed", feedback)
        self.assertIn("do not claim the path exists", feedback)

    def test_accurate_missing_path_summary_is_allowed_for_direct_read_question(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "missing.py"}}),
                json.dumps({"type": "final", "message": "missing.py does not exist in the workspace."}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=3)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Read missing.py and tell me line 1.")

        self.assertTrue(result.completed)
        self.assertIn("does not exist", result.message.lower())
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertNotIn("This request requires real tool use in this turn.", feedback)

    def test_false_content_claim_is_blocked_after_missing_path_failure(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "missing.py"}}),
                json.dumps({"type": "final", "message": "line 1 is hello"}),
                json.dumps({"type": "final", "message": "missing.py does not exist in the workspace."}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=4)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Read missing.py and tell me line 1.")

        self.assertTrue(result.completed)
        self.assertIn("does not exist", result.message.lower())
        controller_guards = [
            event
            for event in agent.events
            if event.get("type") == "controller_guard" and event.get("guard") == "missing-path-content-claim"
        ]
        self.assertEqual(len(controller_guards), 1)
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("latest path lookup failed", feedback)
        self.assertIn("do not invent file contents", feedback)

    def test_request_obligation_code_change_requires_mutation_not_source_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def add(left, right):\n    return left + right\n", encoding="utf-8")
            agent = OllamaCodeAgent(
                client=FakeClient([]),
                tools=CountingToolExecutor(root, approval_mode="auto"),
                model="fake-model",
                debate_enabled=False,
            )

            statuses = agent._request_obligation_proof_status(
                obligations=[{"id": "code-change", "kind": "code_change", "label": "implement the requested code change"}],
                successful_tool_results=[
                    {"name": "read_file", "arguments": {"path": "app.py"}, "result": {"ok": True, "path": "app.py", "output": "def add(left, right):\n    return left + right\n"}},
                ],
                required_tool_names=set(),
            )

        self.assertEqual(statuses[0]["status"], "unproven")
        self.assertIn("real code change", statuses[0]["guidance"])

    def test_readme_inspection_does_not_create_docs_update_obligation(self) -> None:
        root = self._workspace_scratch()
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)

        obligations = agent._derive_request_obligations(
            request_text="Inspect README.md for needle and summarize.",
            required_tool_names=set(),
            required_mutation_paths=set(),
            code_mutation_required=False,
            test_run_required=False,
        )
        update_obligations = [item for item in obligations if item.get("kind") == "docs_update"]

        self.assertEqual(update_obligations, [])

    def test_command_obligation_ignores_descriptive_command_words(self) -> None:
        root = self._workspace_scratch()
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)

        obligations = agent._derive_request_obligations(
            request_text=(
                "Implement a new archive subcommand. Update README with the new command and flag. "
                "Run the tests and prove the new behavior with a shell command. Add list --all."
            ),
            required_tool_names=set(),
            required_mutation_paths=set(),
            code_mutation_required=True,
            test_run_required=True,
        )
        feature_ids = sorted(item["id"] for item in obligations if item.get("kind") == "feature_token")

        self.assertEqual(feature_ids, ["command:archive", "flag:--all"])

    def test_function_obligation_tracks_requested_new_function(self) -> None:
        root = self._workspace_scratch()
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)

        obligations = agent._derive_request_obligations(
            request_text=(
                "Add an export_ndjson(rows) function to this report exporter. "
                "Update README and add tests."
            ),
            required_tool_names=set(),
            required_mutation_paths=set(),
            code_mutation_required=True,
            test_run_required=True,
        )
        feature_ids = sorted(item["id"] for item in obligations if item.get("kind") == "feature_token")

        self.assertEqual(feature_ids, ["function:export_ndjson"])

    def test_requested_test_addition_and_shell_proof_create_obligations(self) -> None:
        root = self._workspace_scratch()
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)

        obligations = agent._derive_request_obligations(
            request_text=(
                "Add an export_ndjson(rows) function. Add tests for multiple rows and escaping names. "
                "Run the tests and prove the behavior with a shell command."
            ),
            required_tool_names=set(),
            required_mutation_paths=set(),
            code_mutation_required=True,
            test_run_required=True,
        )
        obligation_ids = {item["id"] for item in obligations}

        self.assertIn("tests-update", obligation_ids)
        self.assertIn("shell-proof", obligation_ids)

    def test_read_only_command_check_does_not_create_feature_obligation(self) -> None:
        root = self._workspace_scratch()
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)

        obligations = agent._derive_request_obligations(
            request_text="Check whether the helper command works, but do not loop on dependency failures.",
            required_tool_names=set(),
            required_mutation_paths=set(),
            code_mutation_required=False,
            test_run_required=False,
        )

        self.assertFalse(any(item.get("kind") == "feature_token" for item in obligations))

    def test_function_obligation_requires_source_proof(self) -> None:
        root = self._workspace_scratch()
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)
        obligation = {
            "id": "function:export_ndjson",
            "kind": "feature_token",
            "label": 'prove the "export_ndjson" function exists',
            "token": "export_ndjson",
            "feature_class": "function",
        }

        unproven = agent._request_obligation_proof_status(
            obligations=[obligation],
            successful_tool_results=[
                {
                    "name": "write_file",
                    "arguments": {"path": "reports/exporter.py", "content": "def export_ndjson(rows):\n    return ''\n"},
                    "result": {"ok": True, "path": "reports/exporter.py", "summary": "Wrote reports/exporter.py."},
                }
            ],
            required_tool_names=set(),
        )
        proven = agent._request_obligation_proof_status(
            obligations=[obligation],
            successful_tool_results=[
                {
                    "name": "read_file",
                    "arguments": {"path": "reports/exporter.py"},
                    "result": {"ok": True, "path": "reports/exporter.py", "output": "def export_ndjson(rows):\n    return ''\n"},
                }
            ],
            required_tool_names=set(),
        )

        self.assertEqual(unproven[0]["status"], "unproven")
        self.assertIn('function "export_ndjson" is still unproven', unproven[0]["guidance"])
        self.assertEqual(proven[0]["status"], "proven")
        self.assertEqual(proven[0]["evidence"], "reports/exporter.py")

    def test_requested_tests_and_shell_proof_require_matching_tool_evidence(self) -> None:
        root = self._workspace_scratch()
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)
        obligations = [
            {"id": "tests-update", "kind": "tests_update", "label": "add or update the requested tests"},
            {"id": "shell-proof", "kind": "shell_proof", "label": "prove the requested behavior with a shell command"},
        ]

        old_tests_only = agent._request_obligation_proof_status(
            obligations=obligations,
            successful_tool_results=[
                {
                    "name": "run_test",
                    "arguments": {"command": "python -m unittest discover -s tests -v"},
                    "result": {"ok": True, "output": "Ran 2 tests in 0.000s\n\nOK"},
                },
                {
                    "name": "read_file",
                    "arguments": {"path": "reports/__init__.py"},
                    "result": {"ok": True, "path": "reports/__init__.py", "output": "def export_ndjson(rows):\n    return ''\n"},
                },
            ],
            required_tool_names=set(),
        )
        proven = agent._request_obligation_proof_status(
            obligations=obligations,
            successful_tool_results=[
                {
                    "name": "write_file",
                    "arguments": {"path": "tests/test_exporter.py", "content": "def test_export_ndjson():\n    assert True\n"},
                    "result": {"ok": True, "path": "tests/test_exporter.py"},
                },
                {
                    "name": "run_shell",
                    "arguments": {"command": "python -c \"from reports import export_ndjson; print(export_ndjson([]))\""},
                    "result": {"ok": True, "command": "python -c \"from reports import export_ndjson; print(export_ndjson([]))\""},
                },
            ],
            required_tool_names=set(),
        )

        self.assertEqual([item["status"] for item in old_tests_only], ["unproven", "unproven"])
        self.assertIn("add or update tests", old_tests_only[0]["guidance"])
        self.assertIn("shell-command proof", old_tests_only[1]["guidance"])
        self.assertEqual([item["status"] for item in proven], ["proven", "proven"])
        self.assertEqual(proven[0]["evidence"], "tests/test_exporter.py")
        self.assertEqual(proven[1]["evidence"], "run_shell")

    def test_final_verification_requires_read_proof_for_requested_command_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def add(left, right):\n    return left + right\n", encoding="utf-8")
            client = FakeClient(
                [
                    '{"type":"tool","name":"replace_in_file","arguments":{"path":"app.py","old":"def add(left, right):\\n    return left + right\\n","new":"def add(left, right):\\n    return left + right\\n\\ndef stats():\\n    return 1\\n"}}',
                    '{"type":"final","message":"Added the stats command."}',
                    '{"type":"tool","name":"read_file","arguments":{"path":"app.py"}}',
                    '{"type":"final","message":"Added the stats command after verifying app.py."}',
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=7)

            result = agent.handle_user("Add a stats command to app.py, but do not run tests.")

        self.assertTrue(result.completed)
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertIn("replace_in_file", tool_calls)
        self.assertIn("read_file", tool_calls)
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn('prove the "stats" command exists', feedback)

    def test_final_verification_requires_behavior_proof_for_cli_command_and_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task_cli.py").write_text(
                "from __future__ import annotations\n\n"
                "import argparse\n\n"
                "def build_parser() -> argparse.ArgumentParser:\n"
                "    parser = argparse.ArgumentParser()\n"
                "    subparsers = parser.add_subparsers(dest='command', required=True)\n"
                "    subparsers.add_parser('list')\n"
                "    return parser\n\n"
                "def main(argv: list[str] | None = None) -> int:\n"
                "    args = build_parser().parse_args(argv)\n"
                "    if args.command == 'list':\n"
                "        print('alpha')\n"
                "        return 0\n"
                "    raise SystemExit(2)\n\n"
                "if __name__ == '__main__':\n"
                "    raise SystemExit(main())\n",
                encoding="utf-8",
            )
            client = FakeClient(
                [
                    json.dumps(
                        {
                            "type": "tool",
                            "name": "write_file",
                            "arguments": {
                                "path": "task_cli.py",
                                "content": (
                                    "from __future__ import annotations\n\n"
                                    "import argparse\n\n"
                                    "TASKS = [\n"
                                    "    {'title': 'alpha', 'priority': 'high'},\n"
                                    "    {'title': 'beta', 'priority': 'low'},\n"
                                    "]\n\n"
                                    "def build_parser() -> argparse.ArgumentParser:\n"
                                    "    parser = argparse.ArgumentParser()\n"
                                    "    subparsers = parser.add_subparsers(dest='command', required=True)\n"
                                    "    list_parser = subparsers.add_parser('list')\n"
                                    "    list_parser.add_argument('--priority', default=None)\n"
                                    "    subparsers.add_parser('stats')\n"
                                    "    return parser\n\n"
                                    "def main(argv: list[str] | None = None) -> int:\n"
                                    "    args = build_parser().parse_args(argv)\n"
                                    "    if args.command == 'list':\n"
                                    "        tasks = TASKS if args.priority is None else [task for task in TASKS if task['priority'] == args.priority]\n"
                                    "        for task in tasks:\n"
                                    "            print(task['title'])\n"
                                    "        return 0\n"
                                    "    if args.command == 'stats':\n"
                                    "        print('high: 1')\n"
                                    "        print('low: 1')\n"
                                    "        return 0\n"
                                    "    raise SystemExit(2)\n\n"
                                    "if __name__ == '__main__':\n"
                                    "    raise SystemExit(main())\n"
                                ),
                            },
                        }
                    ),
                    '{"type":"final","message":"Added the stats command and --priority flag."}',
                    '{"type":"tool","name":"read_file","arguments":{"path":"task_cli.py"}}',
                    json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": f'{sys.executable} task_cli.py stats'}}),
                    json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": f'{sys.executable} task_cli.py list --priority high'}}),
                    '{"type":"final","message":"Added the stats command and --priority flag after proving them."}',
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

            result = agent.handle_user("Add a stats command and --priority flag to task_cli.py, but do not run tests.")

        self.assertTrue(result.completed)
        self.assertEqual(tools.execute_counts.get("run_shell"), 2)
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("implementation proof and behavior proof", feedback)
        self.assertIn('prove the "--priority" flag exists', feedback)

    def test_report_ndjson_export_package_repair_updates_code_tests_docs_and_shell_proof(self) -> None:
        root = self._workspace_scratch()
        (root / "reports").mkdir()
        (root / "tests").mkdir()
        (root / "reports" / "__init__.py").write_text(
            "from .exporter import ReportRow, export_csv\n\n"
            "__all__ = [\"ReportRow\", \"export_csv\"]\n",
            encoding="utf-8",
        )
        (root / "reports" / "exporter.py").write_text(
            "from __future__ import annotations\n\n"
            "import csv\n"
            "import io\n"
            "from dataclasses import dataclass\n\n\n"
            "@dataclass(frozen=True)\n"
            "class ReportRow:\n"
            "    name: str\n"
            "    count: int\n"
            "    active: bool\n\n\n"
            "def export_csv(rows: list[ReportRow]) -> str:\n"
            "    buffer = io.StringIO()\n"
            "    writer = csv.DictWriter(buffer, fieldnames=[\"name\", \"count\", \"active\"])\n"
            "    writer.writeheader()\n"
            "    for row in rows:\n"
            "        writer.writerow({\"name\": row.name, \"count\": row.count, \"active\": row.active})\n"
            "    return buffer.getvalue()\n",
            encoding="utf-8",
        )
        (root / "tests" / "test_exporter.py").write_text(
            "import csv\nimport io\nimport unittest\n\n"
            "from reports import ReportRow, export_csv\n\n\n"
            "class ExporterTests(unittest.TestCase):\n"
            "    def test_export_csv_header_and_rows(self) -> None:\n"
            "        output = export_csv([ReportRow(\"alpha\", 2, True)])\n"
            "        rows = list(csv.DictReader(io.StringIO(output)))\n"
            "        self.assertEqual(rows[0], {\"name\": \"alpha\", \"count\": \"2\", \"active\": \"True\"})\n\n\n"
            "if __name__ == \"__main__\":\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )
        (root / "README.md").write_text("# Report Exporter\n\nUse `export_csv(rows)` for CSV output.\n", encoding="utf-8")
        command = f"{sys.executable} -m unittest discover -s tests -v"
        tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
        agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=4)
        request_text = (
            "Add an export_ndjson(rows) function to this report exporter. It should serialize each ReportRow "
            "as one JSON object per line with keys name, count, and active in that order, preserve row order, "
            "and end the output with a trailing newline when rows are present. It should return an empty string "
            "for no rows. Export it from the package __init__.py. Update README with the new NDJSON export "
            "behavior. Add tests for multiple rows, empty rows, and escaping names with quotes or newlines. "
            "Run the tests and prove the behavior with a shell command."
        )

        result = agent.handle_user(request_text)

        self.assertTrue(result.completed)
        self.assertIn("NDJSON export repair", result.message)
        exporter_text = (root / "reports" / "exporter.py").read_text(encoding="utf-8")
        init_text = (root / "reports" / "__init__.py").read_text(encoding="utf-8")
        test_text = (root / "tests" / "test_exporter.py").read_text(encoding="utf-8")
        readme_text = (root / "README.md").read_text(encoding="utf-8")
        self.assertIn("def export_ndjson(rows: list[ReportRow]) -> str:", exporter_text)
        self.assertIn("import json", exporter_text)
        self.assertIn("export_ndjson", init_text)
        self.assertIn("test_export_ndjson_multiple_rows", test_text)
        self.assertIn("test_export_ndjson_empty_rows", test_text)
        self.assertIn("test_export_ndjson_escapes_names", test_text)
        self.assertIn("Use `export_ndjson(rows)`", readme_text)
        self.assertGreaterEqual(tools.execute_counts.get("write_file", 0), 4)
        self.assertGreaterEqual(tools.execute_counts.get("run_test", 0), 1)
        self.assertEqual(tools.execute_counts.get("run_shell"), 1)
        self.assertTrue(
            any(
                event.get("type") == "spec_guided_repair"
                and event.get("phase") == "report_ndjson_export_obligation_verification"
                and event.get("ok") is True
                for event in agent.events
            )
        )

    def test_request_obligations_persist_across_continue_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def add(left, right):\n    return left + right\n", encoding="utf-8")
            (root / "README.md").write_text("Usage\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "test_app.py").write_text("import unittest\n\n\nclass AppTests(unittest.TestCase):\n    def test_placeholder(self) -> None:\n        self.assertTrue(True)\n\n\nif __name__ == '__main__':\n    unittest.main()\n", encoding="utf-8")
            client = FakeClient(
                [
                    '{"type":"tool","name":"write_file","arguments":{"path":"README.md","content":"Use the stats command.\\n"}}',
                    *['{"type":"final","message":"Added the stats command and updated README."}' for _ in range(10)],
                    '{"type":"tool","name":"read_file","arguments":{"path":"README.md"}}',
                    '{"type":"tool","name":"read_file","arguments":{"path":"app.py"}}',
                    '{"type":"tool","name":"replace_in_file","arguments":{"path":"app.py","old":"def add(left, right):\\n    return left + right\\n","new":"def add(left, right):\\n    return left + right\\n\\ndef stats():\\n    return 1\\n"}}',
                    '{"type":"tool","name":"read_file","arguments":{"path":"app.py"}}',
                    '{"type":"final","message":"Added the stats command after verifying app.py and README."}',
                    '{"type":"tool","name":"read_file","arguments":{"path":"app.py"}}',
                    '{"type":"final","message":"Added the stats command after verifying app.py and README final."}',
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=f"{sys.executable} -m unittest discover -s tests -v")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=10)

            first = agent.handle_user("Add a stats command to app.py and update README.md.")
            second = agent.handle_user("Actually implement stats in app.py.")
            final_source = (root / "app.py").read_text(encoding="utf-8")

        self.assertFalse(first.completed)
        self.assertTrue(second.completed)
        self.assertIn("def stats()", final_source)
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertIn("read_file", tool_calls)
        self.assertIn("write_file", tool_calls)

    def test_failed_edit_recovery_state_persists_across_continue_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task_cli.py").write_text("def main() -> int:\n    return 0\n", encoding="utf-8")
            client = FakeClient(
                [
                    json.dumps(
                        {
                            "type": "tool",
                            "name": "replace_in_file",
                            "arguments": {
                                "path": "task_cli.py",
                                "old": "    return 0\n",
                                "new": "    return 1\n",
                            },
                        }
                    ),
                    '{"type":"final","message":"still working"}',
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=2)
            agent._sticky_request_obligations = [
                {"id": "code-change", "kind": "code_change", "label": "implement the requested code change"},
                {"id": "flag:--priority", "kind": "feature_token", "label": 'prove the "--priority" flag exists', "token": "--priority", "feature_class": "flag"},
            ]
            agent._sticky_failed_edit_recovery = [
                {
                    "target_id": "path:task_cli.py",
                    "kind": "path",
                    "path": "task_cli.py",
                    "symbol": "",
                    "tool_name": "replace_in_file",
                    "last_mutating_tool_family": "replace_in_file",
                    "tool_granularity": "narrow",
                    "validation_name": "run_test",
                    "failing_validators": ["run_test"],
                    "diagnostic": "test_list failed after a speculative CLI edit",
                    "failure_event_index": -1,
                    "repair_strategy": "cli_surface_repair",
                    "required_proof_items": ['prove the "--priority" flag exists'],
                    "behavior_paths": ["tests/test_task_cli.py"],
                    "unresolved_obligations": [
                        {"id": "flag:--priority", "kind": "feature_token", "label": 'prove the "--priority" flag exists'},
                    ],
                }
            ]

            result = agent.handle_user("continue")

        self.assertFalse(result.completed)
        self.assertEqual(tools.execute_counts.get("replace_in_file", 0), 0)
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn('prove the "--priority" flag exists', feedback)
        self.assertTrue(agent._sticky_failed_edit_recovery)

    def test_failed_edit_recovery_blocks_validation_only_loop_before_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task_cli.py").write_text("def main() -> int:\n    return 0\n", encoding="utf-8")
            client = FakeClient(['{"type":"final","message":"still checking"}'])
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
            state = {
                "target_id": "path:task_cli.py",
                "kind": "path",
                "path": "task_cli.py",
                "symbol": "",
                "tool_name": "replace_in_file",
                "last_mutating_tool_family": "replace_in_file",
                "tool_granularity": "narrow",
                "validation_name": "run_test",
                "failing_validators": ["run_test"],
                "diagnostic": "test_list failed after the previous edit",
                "failure_event_index": -1,
                "repair_strategy": "cli_surface_repair",
                "required_proof_items": ['prove the "--priority" flag exists'],
                "behavior_paths": ["tests/test_task_cli.py"],
                "unresolved_obligations": [
                    {"id": "flag:--priority", "kind": "feature_token", "label": 'prove the "--priority" flag exists'},
                ],
            }

        self.assertTrue(agent._repair_spec_blocks_validation_loop(state, "run_test"))
        self.assertIn(
            "Do not rerun validators until you make the broader repair.",
            agent._repair_spec_validation_retry_message(state),
        )

    def test_failed_edit_recovery_blocks_auto_validation_loop_after_failed_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "task_cli.py").write_text(
                "from __future__ import annotations\n\n"
                "TASKS = [\n"
                "    {'title': 'write-docs', 'status': 'todo', 'priority': 'high'},\n"
                "    {'title': 'ship-cli', 'status': 'done', 'priority': 'low'},\n"
                "]\n\n"
                "def list_tasks(priority: str | None = None) -> list[str]:\n"
                "    return [task['title'] for task in TASKS]\n",
                encoding="utf-8",
            )
            (root / "tests" / "test_task_cli.py").write_text(
                "import unittest\n"
                "from task_cli import list_tasks\n\n"
                "class TaskCliTests(unittest.TestCase):\n"
                "    def test_priority_output(self):\n"
                "        self.assertIn('write-docs:high', list_tasks(priority='high'))\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n",
                encoding="utf-8",
            )
            client = FakeClient(
                [
                    '{"type":"tool","name":"read_file","arguments":{"path":"task_cli.py"}}',
                    json.dumps(
                        {
                            "type": "tool",
                            "name": "edit_intent",
                            "arguments": {
                                "intent": "replace_body",
                                "path": "task_cli.py",
                                "target": "list_tasks",
                                "replacement": (
                                    "selected = TASKS\n"
                                    "if priority is not None:\n"
                                    "    selected = [task for task in TASKS if task['priority'] == priority]\n"
                                    "return [task['title'] for task in selected]"
                                ),
                            },
                        }
                    ),
                    '{"type":"final","message":"Implemented the priority output and tests passed."}',
                    '{"type":"final","message":"Implemented the priority output and tests passed."}',
                ]
            )
            test_command = f"{sys.executable} -m unittest discover -s tests -p test_task_cli.py"
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=test_command)
            agent = OllamaCodeAgent(
                client=client,
                tools=tools,
                model="fake-model",
                debate_enabled=False,
                disable_spec_guided_repair=True,
                max_tool_rounds=4,
            )

            with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
                result = agent.handle_user("Update task_cli.py so priority list output includes the priority, and keep tests green.")

        self.assertFalse(result.completed)
        self.assertEqual(tools.execute_counts.get("run_test"), 1)
        self.assertIn("Do not rerun validators until you make the broader repair", result.message)
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Do not rerun validators until you make the broader repair", feedback)

    def test_failed_mutation_obligations_block_validation_before_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "task_cli.py").write_text(
                "def list_tasks(priority=None):\n"
                "    return []\n",
                encoding="utf-8",
            )
            (root / "tests" / "test_task_cli.py").write_text(
                "import unittest\n\n"
                "class TaskCliTests(unittest.TestCase):\n"
                "    def test_placeholder(self):\n"
                "        self.assertTrue(True)\n",
                encoding="utf-8",
            )
            client = FakeClient(
                [
                    '{"type":"tool","name":"read_file","arguments":{"path":"task_cli.py"}}',
                    '{"type":"tool","name":"edit_intent","arguments":{"intent":"replace_body","path":"task_cli.py","target":"list_tasks"}}',
                    json.dumps(
                        {
                            "type": "tool",
                            "name": "write_file",
                            "arguments": {
                                "path": "task_cli.py",
                                "content": "> def list_tasks(priority=None):\n>     return []\n> \n> def due_before():\n>     return None\n",
                            },
                        }
                    ),
                    '{"type":"tool","name":"write_file","arguments":{"path":"README.md","content":"Use --due-before.\\n"}}',
                    '{"type":"tool","name":"run_test","arguments":{"command":"python -m unittest discover -s tests -v"}}',
                    '{"type":"final","message":"done"}',
                ]
            )
            (root / "README.md").write_text("Task CLI\n", encoding="utf-8")
            tools = CountingToolExecutor(root, approval_mode="auto", test_command="python -m unittest discover -s tests -v")
            agent = OllamaCodeAgent(
                client=client,
                tools=tools,
                model="fake-model",
                debate_enabled=False,
                disable_spec_guided_repair=True,
                max_tool_rounds=5,
            )

            result = agent.handle_user(
                "Add a --due-before option to task_cli.py, update tests and README, run tests, and prove it with a shell command."
            )

        self.assertFalse(result.completed)
        self.assertIsNone(tools.execute_counts.get("run_test"))
        self.assertEqual(tools.execute_counts.get("write_file"), 1)
        guard_names = [event.get("guard") for event in agent.events if event.get("type") == "controller_guard"]
        self.assertIn("failed-mutation-obligations-before-validation", guard_names)
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Do not run validation or proof commands after failed edits", feedback)
        self.assertIn("make a successful source or test mutation first", feedback)

    def test_repeated_edit_intent_failure_allows_replace_symbol_repair(self) -> None:
        root = self._workspace_scratch()
        source = (
            "def list_tasks(path, *, priority=None):\n"
            "    tasks = []\n"
            "    if priority:\n"
            "        tasks = [task for task in tasks if task.get('priority') == priority]\n"
            "    return tasks\n"
        )
        (root / "task_cli.py").write_text(source, encoding="utf-8")
        replacement = (
            "def list_tasks(path, *, priority=None):\n"
            "    tasks = []\n"
            "    due_before = '2026-07-31'\n"
            "    if priority:\n"
            "        tasks = [task for task in tasks if task.get('priority') == priority]\n"
            "    if due_before:\n"
            "        tasks = [task for task in tasks if task.get('due') and task.get('due') <= due_before]\n"
            "    return tasks\n"
        )
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "task_cli.py"}}),
                json.dumps(
                    {
                        "type": "tool",
                        "name": "edit_intent",
                        "arguments": {
                            "path": "task_cli.py",
                            "intent": "add_function",
                            "target": "list_tasks",
                            "replacement": replacement,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "tool",
                        "name": "edit_intent",
                        "arguments": {
                            "path": "task_cli.py",
                            "intent": "replace_symbol",
                            "target": "list_tasks",
                            "replacement": replacement,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "tool",
                        "name": "replace_symbol",
                        "arguments": {"path": "task_cli.py", "symbol": "list_tasks", "content": replacement},
                    }
                ),
                json.dumps({"type": "final", "message": "stopped"}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=5)

        agent.handle_user("Add due_before support to list_tasks in task_cli.py.")

        self.assertEqual(tools.execute_counts.get("replace_symbol"), 1)
        updated = (root / "task_cli.py").read_text(encoding="utf-8")
        self.assertIn("due_before = '2026-07-31'", updated)
        self.assertIn("task.get('due') <= due_before", updated)
        guard_names = [event.get("guard") for event in agent.events if event.get("type") == "controller_guard"]
        self.assertIn("repeated-mutating-failure-pivot", guard_names)

    def test_omitted_write_file_failure_forces_symbol_repair(self) -> None:
        root = self._workspace_scratch()
        source = (
            "def list_tasks(path, *, priority=None):\n"
            "    tasks = []\n"
            "    if priority:\n"
            "        tasks = [task for task in tasks if task.get('priority') == priority]\n"
            "    return tasks\n"
        )
        (root / "task_cli.py").write_text(source, encoding="utf-8")
        full_rewrite = source.replace("    return tasks\n", "    return list(tasks)\n")
        symbol_replacement = (
            "def list_tasks(path, *, priority=None):\n"
            "    tasks = []\n"
            "    due_before = '2026-07-31'\n"
            "    if priority:\n"
            "        tasks = [task for task in tasks if task.get('priority') == priority]\n"
            "    if due_before:\n"
            "        tasks = [task for task in tasks if task.get('due') and task.get('due') <= due_before]\n"
            "    return tasks\n"
        )
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "task_cli.py"}}),
                json.dumps(
                    {
                        "type": "tool",
                        "name": "write_file",
                        "arguments": {
                            "path": "task_cli.py",
                            "content": "[omitted 900 chars from prior content; do not copy]",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "tool",
                        "name": "write_file",
                        "arguments": {"path": "task_cli.py", "content": full_rewrite},
                    }
                ),
                json.dumps(
                    {
                        "type": "tool",
                        "name": "replace_symbol",
                        "arguments": {
                            "path": "task_cli.py",
                            "symbol": "list_tasks",
                            "content": "[omitted 650 chars from prior content; do not copy]",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "tool",
                        "name": "replace_symbol",
                        "arguments": {"path": "task_cli.py", "symbol": "list_tasks", "content": symbol_replacement},
                    }
                ),
                json.dumps({"type": "final", "message": "stopped"}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=5)

        agent.handle_user("Add due_before support to list_tasks in task_cli.py.")

        self.assertIsNone(tools.execute_counts.get("write_file"))
        self.assertEqual(tools.execute_counts.get("replace_symbol"), 1)
        updated = (root / "task_cli.py").read_text(encoding="utf-8")
        self.assertIn("due_before = '2026-07-31'", updated)
        guard_names = [event.get("guard") for event in agent.events if event.get("type") == "controller_guard"]
        self.assertIn("write-file-omitted-content", guard_names)
        self.assertIn("write-file-after-omitted-content", guard_names)
        self.assertIn("mutation-omitted-content", guard_names)

    def test_repeated_omitted_mutation_payloads_fail_closed_early(self) -> None:
        root = self._workspace_scratch()
        (root / "task_cli.py").write_text(
            "def list_tasks(path, *, priority=None):\n"
            "    return []\n",
            encoding="utf-8",
        )
        omitted_replacement = "[omitted 650 chars from prior content; do not copy]"
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "task_cli.py"}}),
                json.dumps(
                    {
                        "type": "tool",
                        "name": "replace_symbol",
                        "arguments": {"path": "task_cli.py", "symbol": "list_tasks", "content": omitted_replacement},
                    }
                ),
                json.dumps(
                    {
                        "type": "tool",
                        "name": "replace_symbol",
                        "arguments": {"path": "task_cli.py", "symbol": "list_tasks", "content": omitted_replacement},
                    }
                ),
                json.dumps(
                    {
                        "type": "tool",
                        "name": "replace_symbol",
                        "arguments": {"path": "task_cli.py", "symbol": "list_tasks", "content": omitted_replacement},
                    }
                ),
                json.dumps({"type": "final", "message": "stopped"}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

        result = agent.handle_user("Add due_before support to list_tasks in task_cli.py.")

        self.assertFalse(result.completed)
        self.assertIn("repeated omitted-context mutation payloads", result.message)
        self.assertIsNone(tools.execute_counts.get("replace_symbol"))
        guard_names = [event.get("guard") for event in agent.events if event.get("type") == "controller_guard"]
        self.assertIn("omitted-mutation-loop-compressed", guard_names)

    def test_failed_edit_recovery_only_counts_allowed_broad_repair_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task_cli.py").write_text("def list_tasks(priority=None):\n    return []\n", encoding="utf-8")
            agent = OllamaCodeAgent(
                client=FakeClient([]),
                tools=CountingToolExecutor(root, approval_mode="auto"),
                model="fake-model",
                debate_enabled=False,
            )
            state = {
                "target_id": "symbol:task_cli.py:list_tasks",
                "kind": "symbol",
                "path": "task_cli.py",
                "symbol": "list_tasks",
                "failure_event_index": -1,
                "repair_strategy": "cli_surface_repair",
            }

            add_import_allowed, add_import_reason = agent._repair_spec_mutation_allowed(
                state,
                proposed_tool_name="edit_intent",
                proposed_arguments={"path": "task_cli.py", "intent": "add_import", "target": "Counter"},
            )
            symbol_allowed, _symbol_reason = agent._repair_spec_mutation_allowed(
                state,
                proposed_tool_name="replace_symbol",
                proposed_arguments={"path": "task_cli.py", "symbol": "list_tasks", "content": "def list_tasks(priority=None):\n    return []\n"},
            )
            file_allowed, _file_reason = agent._repair_spec_mutation_allowed(
                state,
                proposed_tool_name="write_file",
                proposed_arguments={"path": "task_cli.py", "content": "def list_tasks(priority=None):\n    return []\n"},
            )
            agent.events.append(
                {
                    "type": "tool_result",
                    "name": "edit_intent",
                    "arguments": {"path": "task_cli.py", "intent": "add_import", "target": "Counter"},
                    "result": {"ok": True, "path": "task_cli.py"},
                }
        )

        self.assertFalse(add_import_allowed)
        self.assertIn("small speculative edit", add_import_reason)
        self.assertTrue(symbol_allowed)
        self.assertTrue(file_allowed)
        self.assertFalse(agent._repair_spec_has_followup_mutation(state))

    def test_failed_edit_recovery_blocks_validation_when_multiple_repair_specs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task_cli.py").write_text("def main() -> int:\n    return 0\n", encoding="utf-8")
            (root / "README.md").write_text("# Task CLI\n", encoding="utf-8")
            agent = OllamaCodeAgent(
                client=FakeClient([]),
                tools=CountingToolExecutor(root, approval_mode="auto"),
                model="fake-model",
                debate_enabled=False,
            )
            states = [
                {
                    "target_id": "path:readme.md",
                    "kind": "path",
                    "path": "README.md",
                    "failure_event_index": -1,
                    "repair_strategy": "file_repair",
                    "diagnostic": "docs update incomplete",
                },
                {
                    "target_id": "path:task_cli.py",
                    "kind": "path",
                    "path": "task_cli.py",
                    "failure_event_index": -1,
                    "repair_strategy": "cli_surface_repair",
                    "diagnostic": "test_list failed after the previous edit",
                },
            ]

        blocked = [state for state in agent._merge_failed_edit_recovery(states) if agent._repair_spec_blocks_validation_loop(state, "lint_typecheck")]
        self.assertTrue(blocked)
        self.assertEqual(blocked[0]["path"], "README.md")

    def test_failed_edit_recovery_rejects_final_before_followup_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task_cli.py").write_text("def main() -> int:\n    return 0\n", encoding="utf-8")
            client = FakeClient(['{"type":"final","message":"Implemented the CLI feature."}'])
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=1)
            agent._sticky_request_obligations = [
                {"id": "code-change", "kind": "code_change", "label": "implement the requested code change"},
                {"id": "flag:--priority", "kind": "feature_token", "label": 'prove the "--priority" flag exists', "token": "--priority", "feature_class": "flag"},
            ]
            agent._sticky_failed_edit_recovery = [
                {
                    "target_id": "path:task_cli.py",
                    "kind": "path",
                    "path": "task_cli.py",
                    "symbol": "",
                    "tool_name": "replace_in_file",
                    "last_mutating_tool_family": "replace_in_file",
                    "tool_granularity": "narrow",
                    "validation_name": "run_test",
                    "failing_validators": ["run_test"],
                    "diagnostic": "test_list failed after the previous edit",
                    "failure_event_index": -1,
                    "repair_strategy": "cli_surface_repair",
                    "required_proof_items": ['prove the "--priority" flag exists'],
                    "behavior_paths": ["tests/test_task_cli.py"],
                    "unresolved_obligations": [
                        {"id": "flag:--priority", "kind": "feature_token", "label": 'prove the "--priority" flag exists'},
                    ],
                }
            ]

            result = agent.handle_user("continue")

        self.assertFalse(result.completed)
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Do not rerun validators until you make the broader repair.", feedback)

    def test_failed_test_guard_requires_repair_before_more_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = OllamaCodeAgent(
                client=FakeClient([]),
                tools=CountingToolExecutor(Path(tmp), approval_mode="auto"),
                model="fake-model",
                debate_enabled=False,
            )

        self.assertTrue(
            agent._failed_test_still_needs_repair(
                latest_run_test_failed=True,
                failed_test_mutation_version=3,
                mutation_version=3,
            )
        )
        self.assertFalse(
            agent._failed_test_still_needs_repair(
                latest_run_test_failed=True,
                failed_test_mutation_version=2,
                mutation_version=3,
            )
        )
        self.assertIn(
            "Repair the implementation before rerunning validators",
            agent._failed_test_repair_retry_message("test_list failed"),
        )

    def test_failed_test_guard_infers_source_target_from_last_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task_cli.py").write_text("def list_tasks():\n    return []\n", encoding="utf-8")
            (root / "README.md").write_text("# Task CLI\n", encoding="utf-8")
            agent = OllamaCodeAgent(
                client=FakeClient([]),
                tools=CountingToolExecutor(root, approval_mode="auto"),
                model="fake-model",
                debate_enabled=False,
            )

            source_mutation = {
                "name": "edit_intent",
                "arguments": {"intent": "replace_symbol", "path": "task_cli.py", "symbol": "list_tasks"},
                "result": {"ok": True, "path": "task_cli.py"},
            }
            doc_mutation = {
                "name": "write_file",
                "arguments": {"path": "README.md", "content": "# Task CLI\n"},
                "result": {"ok": True, "path": "README.md"},
            }
            pathless_source_mutation = {
                "name": "edit_intent",
                "arguments": {"intent": "replace_symbol", "symbol": "list_tasks"},
                "result": {"ok": True},
            }
            grounding = [
                {
                    "name": "read_file",
                    "arguments": {"path": "task_cli.py"},
                    "result": {"ok": True, "path": "task_cli.py", "output": "def list_tasks():\n    return []\n"},
                }
            ]

        self.assertTrue(agent._mutation_record_targets_source(source_mutation))
        self.assertTrue(agent._mutation_record_targets_source(pathless_source_mutation, grounding))
        self.assertFalse(agent._mutation_record_targets_source(doc_mutation))

    def test_failed_run_test_recovery_prefers_prior_source_mutation_over_later_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task_cli.py").write_text("def list_tasks():\n    return []\n", encoding="utf-8")
            (root / "README.md").write_text("# Task CLI\n", encoding="utf-8")
            agent = OllamaCodeAgent(
                client=FakeClient([]),
                tools=CountingToolExecutor(root, approval_mode="auto"),
                model="fake-model",
                debate_enabled=False,
            )
            source_mutation = {
                "name": "edit_intent",
                "arguments": {"intent": "replace_symbol", "path": "task_cli.py", "symbol": "list_tasks"},
                "result": {"ok": True, "path": "task_cli.py"},
            }
            doc_mutation = {
                "name": "write_file",
                "arguments": {"path": "README.md", "content": "# Task CLI\n"},
                "result": {"ok": True, "path": "README.md"},
            }
            validation_mutation = source_mutation if agent._mutation_record_targets_source(source_mutation) else doc_mutation

            agent._set_failed_edit_recovery_state(
                name=str(validation_mutation["name"]),
                arguments=validation_mutation["arguments"],
                successful_tool_results=[],
                validation_name="run_test",
                diagnostic="test_list failed",
            )

        self.assertEqual(agent._sticky_failed_edit_recovery[0]["path"], "task_cli.py")
