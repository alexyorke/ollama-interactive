import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ollama_code.agent import OllamaCodeAgent
from ollama_code.features import ENV_OLLAMA_CODE_FEATURE_PROFILE
from ollama_code.ollama_client import ChatResponse, OllamaError
from ollama_code.tools import ToolExecutor
from tests.agent_test_support import AgentTestBase, CountingToolExecutor, FakeClient


class AgentGroundingPathRepairTests(AgentTestBase):
    def test_agent_context_pack_profile_preloads_context(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "src" / "calc.py").write_text("def sum_values(a, b):\n    return a - b\n", encoding="utf-8")
        client = FakeClient(['{"type":"final","message":"inspected"}'])
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", max_tool_rounds=4, debate_enabled=False)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "context-pack,evidence-handles"}):
            agent.handle_user("Use context_pack to inspect relevant context for src/calc.py and summarize only.")

        calls = [event["name"] for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(calls[0], "context_pack")
        tool_messages = [message["content"] for message in agent.messages if message["role"] == "user" and str(message["content"]).startswith("Evidence:")]
        self.assertTrue(tool_messages)
        self.assertIn("context_pack", tool_messages[0])

    def test_trajectory_ground_guard_rejects_ungrounded_mutation_then_allows_after_read(self) -> None:
        root = self._workspace_scratch()
        (root / "app.py").write_text("def value():\n    return 'old'\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"replace_in_file","arguments":{"path":"app.py","old":"return \'old\'","new":"return \'new\'"}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"app.py"}}',
                '{"type":"tool","name":"replace_in_file","arguments":{"path":"app.py","old":"return \'old\'","new":"return \'new\'"}}',
                '{"type":"final","message":"Updated app.py."}',
            ]
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=6)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Fix app.py by changing old to new.")

        self.assertTrue(result.completed)
        self.assertIn("return 'new'", (root / "app.py").read_text(encoding="utf-8"))
        self.assertTrue(any(event.get("type") == "controller_guard" and event.get("guard") == "ground-before-mutate" for event in agent.events))
        self.assertTrue(any(event.get("type") == "auto_validation" and event.get("name") == "lint_typecheck" for event in agent.events))

    def test_trajectory_ground_guard_auto_reads_explicit_target_before_retry(self) -> None:
        root = self._workspace_scratch()
        (root / "app.py").write_text("def value():\n    return 'old'\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"replace_in_file","arguments":{"path":"app.py","old":"return \'old\'","new":"return \'new\'"}}',
                '{"type":"tool","name":"replace_in_file","arguments":{"path":"app.py","old":"return \'old\'","new":"return \'new\'"}}',
                '{"type":"final","message":"Updated app.py."}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=6)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Fix app.py by changing old to new.")

        self.assertTrue(result.completed)
        self.assertIn("return 'new'", (root / "app.py").read_text(encoding="utf-8"))
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_calls[:2], ["read_file", "replace_in_file"])
        self.assertEqual(tools.execute_counts.get("read_file"), 1)
        self.assertTrue(any("Use the grounded file app.py to make the edit now." in message["content"] for message in agent.messages if message["role"] == "user"))

    def test_trajectory_ground_guard_requires_target_path_grounding_after_unrelated_read(self) -> None:
        root = self._workspace_scratch()
        (root / "notes.txt").write_text("todo\n", encoding="utf-8")
        (root / "app.py").write_text("def value():\n    return 'old'\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"notes.txt"}}',
                '{"type":"tool","name":"replace_in_file","arguments":{"path":"app.py","old":"return \'old\'","new":"return \'new\'"}}',
                '{"type":"tool","name":"replace_in_file","arguments":{"path":"app.py","old":"return \'old\'","new":"return \'new\'"}}',
                '{"type":"final","message":"Updated app.py."}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=7)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Read notes, then fix app.py by changing old to new.")

        self.assertTrue(result.completed)
        self.assertIn("return 'new'", (root / "app.py").read_text(encoding="utf-8"))
        self.assertEqual(tools.execute_counts.get("read_file"), 2)
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_calls[:3], ["read_file", "read_file", "replace_in_file"])
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Use the grounded file app.py to make the edit now.", feedback)

    def test_trajectory_ground_guard_auto_reads_symbol_before_symbol_edit_retry(self) -> None:
        root = self._workspace_scratch()
        (root / "app.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"replace_symbol","arguments":{"path":"app.py","symbol":"add","content":"def add(left, right):\\n    return left + right\\n"}}',
                '{"type":"tool","name":"replace_symbol","arguments":{"path":"app.py","symbol":"add","content":"def add(left, right):\\n    return left + right\\n"}}',
                '{"type":"final","message":"Updated app.py."}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=6)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Fix add in app.py.")

        self.assertTrue(result.completed)
        self.assertIn("return left + right", (root / "app.py").read_text(encoding="utf-8"))
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_calls[:2], ["read_symbol", "replace_symbol"])
        self.assertEqual(tools.execute_counts.get("read_symbol"), 1)
        self.assertTrue(any("Use the grounded symbol add in app.py to make the edit now." in message["content"] for message in agent.messages if message["role"] == "user"))

    def test_trajectory_ground_guard_falls_back_to_code_outline_after_failed_symbol_grounding(self) -> None:
        root = self._workspace_scratch()
        (root / "app.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"replace_symbol","arguments":{"path":"app.py","symbol":"TaskCLI","content":"def add(left, right):\\n    return left + right\\n"}}',
                '{"type":"tool","name":"replace_symbol","arguments":{"path":"app.py","symbol":"TaskCLI","content":"def add(left, right):\\n    return left + right\\n"}}',
                '{"type":"tool","name":"replace_in_file","arguments":{"path":"app.py","old":"return left - right","new":"return left + right"}}',
                '{"type":"final","message":"Updated app.py."}',
                '{"type":"tool","name":"replace_in_file","arguments":{"path":"app.py","old":"return left - right","new":"return left + right"}}',
                '{"type":"final","message":"Updated app.py after grounding app.py."}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Fix app.py so add returns left + right.")

        self.assertTrue(result.completed)
        self.assertIn("return left + right", (root / "app.py").read_text(encoding="utf-8"))
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_calls[:4], ["read_symbol", "code_outline", "read_file", "replace_in_file"])
        self.assertEqual(tools.execute_counts.get("read_symbol"), 1)
        self.assertEqual(tools.execute_counts.get("code_outline"), 1)

    def test_trajectory_ground_guard_auto_reads_symbol_for_pathless_edit_after_source_context(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "src" / "core.py").write_text("def wrapped():\n    return 'old'\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"src/core.py"}}',
                '{"type":"tool","name":"edit_intent","arguments":{"intent":"replace_body","target":"wrapped","replacement":"return \'new\'"}}',
                '{"type":"tool","name":"replace_symbol","arguments":{"path":"src/core.py","symbol":"wrapped","content":"def wrapped():\\n    return \'new\'\\n"}}',
                '{"type":"final","message":"Updated src/core.py."}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=6)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Inspect src/core.py and update wrapped to return new.")

        self.assertTrue(result.completed)
        self.assertIn("return 'new'", (root / "src" / "core.py").read_text(encoding="utf-8"))
        self.assertEqual(tools.execute_counts.get("read_symbol"), 1)
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_calls[:3], ["read_file", "read_symbol", "replace_symbol"])
        self.assertTrue(any(event.get("type") == "controller_guard" and event.get("guard") == "ground-before-mutate" for event in agent.events))
        self.assertTrue(any("You now have current-turn grounding for src/core.py. Edit the grounded target now." in message["content"] for message in agent.messages if message["role"] == "user"))

    def test_trajectory_ground_guard_blocks_pathless_edit_after_unrelated_read_until_source_is_grounded(self) -> None:
        root = self._workspace_scratch()
        (root / "notes.txt").write_text("todo\n", encoding="utf-8")
        (root / "app.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"notes.txt"}}',
                '{"type":"tool","name":"edit_intent","arguments":{"intent":"replace_body","target":"add","replacement":"return left + right"}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"app.py"}}',
                '{"type":"tool","name":"edit_intent","arguments":{"intent":"replace_body","target":"add","replacement":"return left + right"}}',
                '{"type":"tool","name":"replace_symbol","arguments":{"path":"app.py","symbol":"add","content":"def add(left, right):\\n    return left + right\\n"}}',
                '{"type":"final","message":"Updated app.py."}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Read notes, then fix add in app.py.")

        self.assertTrue(result.completed)
        self.assertIn("return left + right", (root / "app.py").read_text(encoding="utf-8"))
        self.assertIsNone(tools.execute_counts.get("edit_intent"))
        self.assertEqual(tools.execute_counts.get("search_symbols"), 1)
        self.assertEqual(tools.execute_counts.get("read_symbol"), 1)
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_calls[:5], ["read_file", "search_symbols", "read_file", "read_symbol", "replace_symbol"])
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Search found a unique symbol match for add in app.py. Read that exact symbol now before editing.", feedback)
        self.assertIn("Use the grounded symbol add in app.py to make the edit now.", feedback)

    def test_trajectory_ground_guard_auto_searches_then_reads_symbol_for_pathless_edit_without_context(self) -> None:
        root = self._workspace_scratch()
        (root / "app.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"edit_intent","arguments":{"intent":"replace_body","target":"add","replacement":"return left + right"}}',
                '{"type":"tool","name":"edit_intent","arguments":{"intent":"replace_body","target":"add","replacement":"return left + right"}}',
                '{"type":"tool","name":"replace_symbol","arguments":{"path":"app.py","symbol":"add","content":"def add(left, right):\\n    return left + right\\n"}}',
                '{"type":"final","message":"Updated app.py."}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Fix add so it returns left + right.")

        self.assertTrue(result.completed)
        self.assertIn("return left + right", (root / "app.py").read_text(encoding="utf-8"))
        self.assertIsNone(tools.execute_counts.get("edit_intent"))
        self.assertEqual(tools.execute_counts.get("search_symbols"), 1)
        self.assertEqual(tools.execute_counts.get("read_symbol"), 1)
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_calls[:3], ["search_symbols", "read_symbol", "replace_symbol"])
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Search found a unique symbol match for add in app.py. Read that exact symbol now before editing.", feedback)
        self.assertIn("Use the grounded symbol add in app.py to make the edit now.", feedback)

    def test_trajectory_ground_guard_prefers_single_non_test_source_after_ambiguous_repo_search(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "src" / "app.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
        (root / "tests" / "helpers.py").write_text("def add(left, right):\n    return 0\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"edit_intent","arguments":{"intent":"replace_body","target":"add","replacement":"return left + right"}}',
                '{"type":"tool","name":"edit_intent","arguments":{"intent":"replace_body","target":"add","replacement":"return left + right"}}',
                '{"type":"tool","name":"edit_intent","arguments":{"intent":"replace_body","target":"add","replacement":"return left + right"}}',
                '{"type":"tool","name":"replace_symbol","arguments":{"path":"src/app.py","symbol":"add","content":"def add(left, right):\\n    return left + right\\n"}}',
                '{"type":"final","message":"Updated src/app.py."}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=10)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Fix add so it returns left + right.")

        self.assertTrue(result.completed)
        self.assertIn("return left + right", (root / "src" / "app.py").read_text(encoding="utf-8"))
        self.assertEqual(tools.execute_counts.get("search_symbols"), 1)
        self.assertEqual(tools.execute_counts.get("read_symbol"), 1)
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_calls[:2], ["search_symbols", "read_symbol"])
        self.assertIn(tool_calls[2], {"edit_intent", "replace_symbol"})
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Search found a unique symbol match for add in src/app.py. Read that exact symbol now before editing.", feedback)
        self.assertIn("Use the grounded symbol add in src/app.py to make the edit now.", feedback)

    def test_trajectory_ground_guard_prefers_test_affined_source_after_multi_source_repo_search(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "scripts").mkdir()
        (root / "tests").mkdir()
        (root / "src" / "core.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
        (root / "scripts" / "add.py").write_text("def add(left, right):\n    return 0\n", encoding="utf-8")
        (root / "tests" / "test_core.py").write_text(
            "import unittest\n"
            "from src.core import add\n\n"
            "class CoreTests(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(add(1, 2), 3)\n",
            encoding="utf-8",
        )
        client = FakeClient(
            [
                '{"type":"tool","name":"edit_intent","arguments":{"intent":"replace_body","target":"add","replacement":"return left + right"}}',
                '{"type":"tool","name":"edit_intent","arguments":{"intent":"replace_body","target":"add","replacement":"return left + right"}}',
                '{"type":"tool","name":"replace_symbol","arguments":{"path":"src/core.py","symbol":"add","content":"def add(left, right):\\n    return left + right\\n"}}',
                '{"type":"final","message":"Updated src/core.py."}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Fix add so it returns left + right.")

        self.assertTrue(result.completed)
        self.assertIn("return left + right", (root / "src" / "core.py").read_text(encoding="utf-8"))
        self.assertIn("return 0", (root / "scripts" / "add.py").read_text(encoding="utf-8"))
        self.assertEqual(tools.execute_counts.get("search_symbols"), 1)
        self.assertEqual(tools.execute_counts.get("read_symbol"), 1)
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_calls[:3], ["search_symbols", "read_symbol", "replace_symbol"])
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Use the grounded symbol add in src/core.py to make the edit now.", feedback)

    def test_trajectory_ground_guard_prefers_explicit_request_source_path_over_unrelated_recent_source(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "src" / "core.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
        (root / "src" / "other.py").write_text("def add(left, right):\n    return 0\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"src/other.py"}}',
                '{"type":"tool","name":"edit_intent","arguments":{"intent":"replace_body","target":"add","replacement":"return left + right"}}',
                '{"type":"tool","name":"replace_symbol","arguments":{"path":"src/core.py","symbol":"add","content":"def add(left, right):\\n    return left + right\\n"}}',
                '{"type":"final","message":"Updated src/core.py."}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Inspect the helper implementation, then fix add in src/core.py.")

        self.assertTrue(result.completed)
        self.assertIn("return left + right", (root / "src" / "core.py").read_text(encoding="utf-8"))
        self.assertIn("return 0", (root / "src" / "other.py").read_text(encoding="utf-8"))
        self.assertIsNone(tools.execute_counts.get("search_symbols"))
        self.assertEqual(tools.execute_counts.get("read_symbol"), 1)
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_calls[:3], ["read_file", "read_symbol", "replace_symbol"])
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("You now have current-turn grounding for src/core.py. Edit the grounded target now.", feedback)

    def test_pathless_mutation_grounding_probe_does_not_auto_pick_when_request_names_multiple_sources(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "src" / "alpha.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
        (root / "src" / "beta.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)

        probe = agent._pathless_mutation_grounding_probe(
            name="edit_intent",
            arguments={"intent": "replace_body", "target": "add", "replacement": "return left + right"},
            requested_mutation_paths={"src/alpha.py", "src/beta.py"},
            successful_tool_results=[],
        )

        self.assertEqual(probe, ("search_symbols", {"query": "add", "path": "."}))

    def test_pathless_mutation_grounding_probe_uses_unique_context_pack_symbol_before_repo_search(self) -> None:
        root = self._workspace_scratch()
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)
        successful_tool_results = [
            {
                "name": "context_pack",
                "arguments": {"request": "Fix calculate_discount", "path": ".", "limit": 8},
                "result": {
                    "ok": True,
                    "tool": "context_pack",
                    "suggested_next_tool": "read_symbol",
                    "ranked_paths": ["src/pricing.py"],
                    "ranked_symbols": [{"path": "src/pricing.py", "qualname": "calculate_discount"}],
                    "output": "context_pack:\nsuggested_next_tool=read_symbol",
                },
            }
        ]

        probe = agent._pathless_mutation_grounding_probe(
            name="edit_intent",
            arguments={"intent": "replace_body", "target": "calculate_discount", "replacement": "return 0"},
            requested_mutation_paths=set(),
            successful_tool_results=successful_tool_results,
        )

        self.assertEqual(probe, ("read_symbol", {"path": "src/pricing.py", "symbol": "calculate_discount", "include_context": 0}))

    def test_context_planner_probe_prefers_unique_context_pack_outline_without_source_context(self) -> None:
        root = self._workspace_scratch()
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)
        successful_tool_results = [
            {
                "name": "context_pack",
                "arguments": {"request": "Inspect implementation structure", "path": ".", "limit": 8},
                "result": {
                    "ok": True,
                    "tool": "context_pack",
                    "suggested_next_tool": "code_outline",
                    "ranked_paths": ["src/pricing.py"],
                    "ranked_symbols": [{"path": "src/pricing.py", "qualname": "calculate_discount"}],
                    "output": "context_pack:\nsuggested_next_tool=code_outline",
                },
            }
        ]

        self.assertTrue(
            agent._context_planner_blocks(
                name="search",
                tool_calls=[],
                latest_run_test_failed=False,
                successful_tool_results=successful_tool_results,
            )
        )
        probe = agent._context_planner_probe(
            successful_tool_results=successful_tool_results,
            forbidden_tool_names=set(),
        )

        self.assertEqual(probe, ("code_outline", {"path": "src/pricing.py"}))

    def test_context_planner_auto_narrows_identifier_search_to_search_symbols(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"src/core.py"}}',
                '{"type":"tool","name":"search","arguments":{"query":"wrapped","path":"src/core.py"}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"src/core.py","start":1,"end":4}}',
                '{"type":"final","message":"wrapped is defined in src/core.py."}',
            ]
        )
        root = self._workspace_scratch()
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=5)
        (root / "src").mkdir()
        (root / "src" / "core.py").write_text("def wrapped():\n    return 'ok'\n", encoding="utf-8")

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Inspect src/core.py and find the wrapped implementation.")

        self.assertEqual(result.message, "wrapped is defined in src/core.py.")
        self.assertEqual(tools.execute_counts.get("search_symbols"), 1)
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_calls[:3], ["read_file", "search", "search_symbols"])
        self.assertTrue(any("Use the symbol-level matches for wrapped in src/core.py." in message["content"] for message in agent.messages if message["role"] == "user"))

    def test_context_planner_auto_narrows_identifier_search_without_source_context(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"search","arguments":{"query":"wrapped"}}',
                '{"type":"tool","name":"search","arguments":{"query":"wrapped"}}',
                '{"type":"final","message":"wrapped is defined in src/core.py."}',
            ]
        )
        root = self._workspace_scratch()
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=5)
        (root / "README.md").write_text("overview\n", encoding="utf-8")
        (root / "src").mkdir()
        (root / "src" / "core.py").write_text("def wrapped():\n    return 'ok'\n", encoding="utf-8")

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Inspect the repo and find the wrapped implementation.")

        self.assertEqual(result.message, "wrapped is defined in src/core.py.")
        self.assertEqual(tools.execute_counts.get("search_symbols"), 1)
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_calls[:2], ["search", "search_symbols"])
        self.assertTrue(any("Use the symbol-level matches for wrapped in src/core.py." in message["content"] for message in agent.messages if message["role"] == "user"))

    def test_context_planner_blocks_for_unique_context_pack_target_without_source_context(self) -> None:
        root = self._workspace_scratch()
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)
        successful_tool_results = [
            {
                "name": "context_pack",
                "arguments": {"request": "Fix calculate_discount", "path": ".", "limit": 8},
                "result": {
                    "ok": True,
                    "tool": "context_pack",
                    "suggested_next_tool": "read_symbol",
                    "ranked_paths": ["src/pricing.py"],
                    "ranked_symbols": [{"path": "src/pricing.py", "qualname": "calculate_discount"}],
                    "output": "context_pack:\nsuggested_next_tool=read_symbol",
                },
            }
        ]

        self.assertTrue(
            agent._context_planner_blocks(
                name="search",
                tool_calls=[],
                latest_run_test_failed=False,
                successful_tool_results=successful_tool_results,
            )
        )

    def test_context_planner_probe_prefers_unique_context_pack_symbol_without_source_context(self) -> None:
        root = self._workspace_scratch()
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)
        successful_tool_results = [
            {
                "name": "context_pack",
                "arguments": {"request": "Fix calculate_discount", "path": ".", "limit": 8},
                "result": {
                    "ok": True,
                    "tool": "context_pack",
                    "suggested_next_tool": "read_symbol",
                    "ranked_paths": ["src/pricing.py"],
                    "ranked_symbols": [{"path": "src/pricing.py", "qualname": "calculate_discount"}],
                    "output": "context_pack:\nsuggested_next_tool=read_symbol",
                },
            }
        ]

        probe = agent._context_planner_probe(
            successful_tool_results=successful_tool_results,
            forbidden_tool_names=set(),
        )

        self.assertEqual(probe, ("read_symbol", {"path": "src/pricing.py", "symbol": "calculate_discount", "include_context": 0}))

    def test_context_pack_auto_outlines_unique_ranked_source_before_broad_search(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"search","arguments":{"query":"implementation structure"}}',
                '{"type":"final","message":"src/core.py contains wrapped."}',
            ]
        )
        root = self._workspace_scratch()
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=3)
        (root / "README.md").write_text("overview\n", encoding="utf-8")
        (root / "src").mkdir()
        (root / "src" / "core.py").write_text(
            "# implementation structure\n\n"
            "def wrapped():\n"
            "    return 'ok'\n",
            encoding="utf-8",
        )

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "context-pack,trajectory-guards"}):
            result = agent.handle_user("Inspect the repo and summarize implementation structure.")

        self.assertEqual(result.message, "src/core.py contains wrapped.")
        self.assertEqual(tools.execute_counts.get("context_pack"), 1)
        self.assertEqual(tools.execute_counts.get("code_outline"), 1)
        self.assertIsNone(tools.execute_counts.get("search"))
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_calls[:2], ["context_pack", "code_outline"])
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Use the code outline for src/core.py.", feedback)

    def test_context_planner_does_not_auto_outline_ambiguous_list_files_code_paths(self) -> None:
        root = self._workspace_scratch()
        (root / "app.py").write_text("def app():\n    return 'app'\n", encoding="utf-8")
        (root / "worker.py").write_text("def worker():\n    return 'worker'\n", encoding="utf-8")
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)
        successful_tool_results = [
            {
                "name": "list_files",
                "arguments": {"path": "."},
                "result": {"ok": True, "tool": "list_files", "path": ".", "output": "app.py\nworker.py\nREADME.md"},
            }
        ]

        self.assertFalse(
            agent._context_planner_blocks(
                name="read_file",
                tool_calls=[],
                latest_run_test_failed=False,
                successful_tool_results=successful_tool_results,
            )
        )
        self.assertIsNone(agent._context_planner_probe(successful_tool_results=successful_tool_results, forbidden_tool_names=set()))

    def test_context_planner_auto_outlines_recent_source_before_more_broad_reads(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "src" / "core.py").write_text("def wrapped():\n    return 'ok'\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"src/core.py"}}',
                '{"type":"tool","name":"search","arguments":{"query":"business logic","path":"src/core.py"}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"src/core.py","start":1,"end":4}}',
                '{"type":"final","message":"wrapped is the only function in src/core.py."}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=5)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Inspect src/core.py and summarize the relevant implementation structure.")

        self.assertEqual(result.message, "wrapped is the only function in src/core.py.")
        self.assertEqual(tools.execute_counts.get("code_outline"), 1)
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_calls[:3], ["read_file", "search", "code_outline"])
        self.assertTrue(any("Use the code outline for src/core.py." in message["content"] for message in agent.messages if message["role"] == "user"))

    def test_context_planner_auto_reads_grounded_implementation_target_symbol_before_more_broad_reads(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "src" / "core.py").write_text("def wrapped():\n    return 'ok'\n", encoding="utf-8")
        (root / "tests" / "test_pkg.py").write_text(
            "import unittest\n"
            "from src.core import wrapped\n\n"
            "class PackageTests(unittest.TestCase):\n"
            "    def test_wrapped(self):\n"
            "        self.assertEqual(wrapped(), 'ok')\n",
            encoding="utf-8",
        )
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"tests/test_pkg.py"}}',
                '{"type":"tool","name":"search","arguments":{"query":"wrapped"}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"tests/test_pkg.py","start":1,"end":4}}',
                '{"type":"tool","name":"search","arguments":{"query":"wrapped"}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"tests/test_pkg.py","start":1,"end":4}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"tests/test_pkg.py","start":1,"end":4}}',
                '{"type":"final","message":"wrapped is defined in src/core.py."}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Inspect tests/test_pkg.py and find the wrapped implementation.")

        self.assertEqual(result.message, "wrapped is defined in src/core.py.")
        self.assertEqual(tools.execute_counts.get("find_implementation_target"), 1)
        self.assertEqual(tools.execute_counts.get("read_symbol"), 1)
        self.assertIsNone(tools.execute_counts.get("search_symbols"))
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_calls[:5], ["read_file", "search", "find_implementation_target", "read_symbol", "read_file"])
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Use the grounded implementation target(s) src/core.py.", feedback)
        self.assertIn("Use the grounded symbol wrapped in src/core.py. Continue from that narrower symbol target or answer from current evidence.", feedback)

    def test_context_planner_uses_recent_identifier_to_pick_symbol_from_grounded_implementation_target(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "src" / "core.py").write_text(
            "def helper():\n    return 'helper'\n\n"
            "def wrapped():\n    return 'ok'\n",
            encoding="utf-8",
        )
        (root / "tests" / "test_pkg.py").write_text(
            "import unittest\n"
            "from src.core import helper, wrapped\n\n"
            "class PackageTests(unittest.TestCase):\n"
            "    def test_wrapped(self):\n"
            "        self.assertEqual(wrapped(), 'ok')\n",
            encoding="utf-8",
        )
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"tests/test_pkg.py"}}',
                '{"type":"tool","name":"search","arguments":{"query":"wrapped"}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"tests/test_pkg.py","start":1,"end":5}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"tests/test_pkg.py","start":1,"end":5}}',
                '{"type":"final","message":"wrapped is defined in src/core.py."}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Inspect tests/test_pkg.py and find the wrapped implementation.")

        self.assertEqual(result.message, "wrapped is defined in src/core.py.")
        self.assertEqual(tools.execute_counts.get("find_implementation_target"), 1)
        self.assertEqual(tools.execute_counts.get("read_symbol"), 1)
        self.assertIsNone(tools.execute_counts.get("search_symbols"))
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_calls[:4], ["read_file", "search", "find_implementation_target", "read_symbol"])
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Use the grounded implementation target(s) src/core.py.", feedback)
        self.assertIn("Use the grounded symbol wrapped in src/core.py. Continue from that narrower symbol target or answer from current evidence.", feedback)

    def test_context_planner_auto_reads_single_outlined_symbol_before_more_broad_reads(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "src" / "core.py").write_text("def wrapped():\n    return 'ok'\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"src/core.py"}}',
                '{"type":"tool","name":"search","arguments":{"query":"business logic","path":"src/core.py"}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"src/core.py","start":1,"end":4}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"src/core.py","start":1,"end":4}}',
                '{"type":"final","message":"wrapped returns ok."}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Inspect src/core.py and summarize the relevant implementation structure.")

        self.assertEqual(result.message, "wrapped returns ok.")
        self.assertEqual(tools.execute_counts.get("code_outline"), 1)
        self.assertEqual(tools.execute_counts.get("read_symbol"), 1)
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_calls[:4], ["read_file", "search", "code_outline", "read_symbol"])
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Use the code outline for src/core.py.", feedback)
        self.assertIn("Use the grounded symbol wrapped in src/core.py. Continue from that narrower symbol target or answer from current evidence.", feedback)

    def test_context_planner_auto_reads_single_outlined_symbol_after_narrowed_repo_search_without_source_context(self) -> None:
        root = self._workspace_scratch()
        (root / "README.md").write_text("overview\n", encoding="utf-8")
        (root / "src").mkdir()
        (root / "src" / "core.py").write_text("# business logic\n\ndef wrapped():\n    return 'ok'\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"search","arguments":{"query":"business logic"}}',
                '{"type":"tool","name":"search","arguments":{"query":"business logic"}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"README.md"}}',
                '{"type":"final","message":"wrapped returns ok."}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=6)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Inspect the repo and summarize the relevant implementation structure.")

        self.assertEqual(result.message, "wrapped returns ok.")
        self.assertEqual(tools.execute_counts.get("code_outline"), 1)
        self.assertEqual(tools.execute_counts.get("read_symbol"), 1)
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_calls[:3], ["search", "code_outline", "read_symbol"])
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Use the code outline for src/core.py.", feedback)
        self.assertIn("Use the grounded symbol wrapped in src/core.py. Continue from that narrower symbol target or answer from current evidence.", feedback)

    def test_context_planner_auto_reads_symbol_match_before_more_broad_reads(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "src" / "core.py").write_text("def wrapped():\n    return 'ok'\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"src/core.py"}}',
                '{"type":"tool","name":"search","arguments":{"query":"wrapped","path":"src/core.py"}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"src/core.py","start":1,"end":4}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"src/core.py","start":1,"end":4}}',
                '{"type":"final","message":"wrapped returns ok."}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Inspect src/core.py and find the wrapped implementation.")

        self.assertEqual(result.message, "wrapped returns ok.")
        self.assertEqual(tools.execute_counts.get("search_symbols"), 1)
        self.assertEqual(tools.execute_counts.get("read_symbol"), 1)
        self.assertEqual(tools.execute_counts.get("code_outline"), None)
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_calls[:4], ["read_file", "search", "search_symbols", "read_symbol"])
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Use the symbol-level matches for wrapped in src/core.py.", feedback)
        self.assertIn("Use the grounded symbol wrapped in src/core.py. Continue from that narrower symbol target or answer from current evidence.", feedback)

    def test_context_planner_auto_reads_symbol_match_without_source_context_after_repo_search(self) -> None:
        root = self._workspace_scratch()
        (root / "README.md").write_text("overview\n", encoding="utf-8")
        (root / "src").mkdir()
        (root / "src" / "core.py").write_text("def wrapped():\n    return 'ok'\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"search","arguments":{"query":"wrapped"}}',
                '{"type":"tool","name":"search","arguments":{"query":"wrapped"}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"README.md"}}',
                '{"type":"final","message":"wrapped returns ok."}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=6)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Inspect the repo and find the wrapped implementation.")

        self.assertEqual(result.message, "wrapped returns ok.")
        self.assertEqual(tools.execute_counts.get("search_symbols"), 1)
        self.assertEqual(tools.execute_counts.get("read_symbol"), 1)
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_calls[:3], ["search", "search_symbols", "read_symbol"])
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Use the symbol-level matches for wrapped in src/core.py.", feedback)
        self.assertIn("Use the grounded symbol wrapped in src/core.py. Continue from that narrower symbol target or answer from current evidence.", feedback)

    def test_successful_symbol_search_match_keeps_ambiguity_when_test_affinity_ties(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "src" / "alpha.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
        (root / "src" / "beta.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
        (root / "tests" / "test_alpha.py").write_text(
            "import unittest\n"
            "from src.alpha import add\n\n"
            "class AlphaTests(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(add(1, 2), 3)\n",
            encoding="utf-8",
        )
        (root / "tests" / "test_beta.py").write_text(
            "import unittest\n"
            "from src.beta import add\n\n"
            "class BetaTests(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(add(1, 2), 3)\n",
            encoding="utf-8",
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)
        search_result = tools.search_symbols("add", path=".")
        successful_tool_results = [{"name": "search_symbols", "arguments": {"query": "add", "path": "."}, "result": search_result}]

        self.assertIsNone(agent._successful_symbol_search_match(query="add", successful_tool_results=successful_tool_results))
        self.assertIsNone(agent._successful_symbol_search_source_path(query="add", successful_tool_results=successful_tool_results))

    def test_successful_symbol_search_prefers_recent_source_among_tied_repo_candidates(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "src" / "alpha.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
        (root / "src" / "beta.py").write_text("def add(left, right):\n    return 0\n", encoding="utf-8")
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)
        search_result = tools.search_symbols("add", path=".")
        successful_tool_results = [
            {
                "name": "read_file",
                "arguments": {"path": "src/alpha.py"},
                "result": {
                    "ok": True,
                    "path": "src/alpha.py",
                    "output": "def add(left, right):\n    return left - right\n",
                },
            },
            {"name": "search_symbols", "arguments": {"query": "add", "path": "."}, "result": search_result},
        ]

        self.assertEqual(agent._successful_symbol_search_source_path(query="add", successful_tool_results=successful_tool_results), "src/alpha.py")
        self.assertEqual(
            agent._successful_symbol_search_match(query="add", successful_tool_results=successful_tool_results),
            ("src/alpha.py", "add"),
        )

    def test_successful_symbol_search_prefers_recent_test_bridge_among_tied_repo_candidates(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "src" / "alpha.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
        (root / "src" / "beta.py").write_text("def add(left, right):\n    return 0\n", encoding="utf-8")
        (root / "tests" / "test_alpha.py").write_text(
            "import unittest\n"
            "from src.alpha import add\n\n"
            "class AlphaTests(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(add(1, 2), 3)\n",
            encoding="utf-8",
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)
        search_result = tools.search_symbols("add", path=".")
        successful_tool_results = [
            {
                "name": "read_file",
                "arguments": {"path": "tests/test_alpha.py"},
                "result": {
                    "ok": True,
                    "path": "tests/test_alpha.py",
                    "output": "from src.alpha import add",
                },
            },
            {"name": "search_symbols", "arguments": {"query": "add", "path": "."}, "result": search_result},
        ]

        self.assertEqual(agent._successful_symbol_search_source_path(query="add", successful_tool_results=successful_tool_results), "src/alpha.py")
        self.assertEqual(
            agent._successful_symbol_search_match(query="add", successful_tool_results=successful_tool_results),
            ("src/alpha.py", "add"),
        )

    def test_pathless_mutation_grounding_probe_prefers_unique_symbol_match_among_multiple_explicit_sources(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "src" / "alpha.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
        (root / "src" / "beta.py").write_text("def subtract(left, right):\n    return left - right\n", encoding="utf-8")
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)

        probe = agent._pathless_mutation_grounding_probe(
            name="edit_intent",
            arguments={"intent": "replace_body", "target": "add", "replacement": "return left + right"},
            requested_mutation_paths={"src/alpha.py", "src/beta.py"},
            successful_tool_results=[],
        )

        self.assertEqual(probe, ("read_symbol", {"path": "src/alpha.py", "symbol": "add", "include_context": 0}))

    def test_trajectory_grounding_probe_keeps_ambiguous_multiple_explicit_sources_unresolved_without_context_pack(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "src" / "alpha.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
        (root / "src" / "beta.py").write_text("def add(left, right):\n    return 0\n", encoding="utf-8")
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)
        search_result = tools.search_symbols("add", path=".")
        successful_tool_results = [{"name": "search_symbols", "arguments": {"query": "add", "path": "."}, "result": search_result}]

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "baseline"}):
            probe = agent._trajectory_grounding_probe(
                request_text="Fix add in src/alpha.py and src/beta.py.",
                name="edit_intent",
                arguments={"intent": "replace_body", "target": "add", "replacement": "return left + right"},
                required_mutation_paths={"src/alpha.py", "src/beta.py"},
                forbidden_tool_names=set(),
                successful_tool_results=successful_tool_results,
                latest_run_test_failed=False,
                latest_run_test_failure_output="",
            )

        self.assertIsNone(probe)

    def test_trajectory_grounding_probe_uses_context_pack_for_ambiguous_multiple_explicit_sources(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "src" / "alpha.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
        (root / "src" / "beta.py").write_text("def add(left, right):\n    return 0\n", encoding="utf-8")
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)
        search_result = tools.search_symbols("add", path=".")
        successful_tool_results = [{"name": "search_symbols", "arguments": {"query": "add", "path": "."}, "result": search_result}]

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "context-pack"}):
            probe = agent._trajectory_grounding_probe(
                request_text="Fix add in src/alpha.py and src/beta.py.",
                name="edit_intent",
                arguments={"intent": "replace_body", "target": "add", "replacement": "return left + right"},
                required_mutation_paths={"src/alpha.py", "src/beta.py"},
                forbidden_tool_names=set(),
                successful_tool_results=successful_tool_results,
                latest_run_test_failed=False,
                latest_run_test_failure_output="",
            )

        self.assertEqual(probe, ("context_pack", {"request": "Fix add in src/alpha.py and src/beta.py.", "path": ".", "limit": 6}))

    def test_trajectory_ground_guard_lists_ambiguous_pathless_candidates_after_search(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "src" / "alpha.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
        (root / "src" / "beta.py").write_text("def add(left, right):\n    return 0\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"search_symbols","arguments":{"query":"add","path":"."}}',
                '{"type":"tool","name":"edit_intent","arguments":{"intent":"replace_body","target":"add","replacement":"return left + right"}}',
            ],
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=2)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Fix add so it returns the sum.")

        self.assertFalse(result.completed)
        self.assertIn("return left - right", (root / "src" / "alpha.py").read_text(encoding="utf-8"))
        self.assertIn("return 0", (root / "src" / "beta.py").read_text(encoding="utf-8"))
        self.assertEqual(tools.execute_counts.get("search_symbols"), 1)
        self.assertIsNone(tools.execute_counts.get("edit_intent"))
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Pathless mutation for add is ambiguous across src/alpha.py, src/beta.py.", feedback)
        self.assertIn("retry with an explicit target path", feedback)

    def test_pathless_mutation_grounding_probe_prefers_test_affined_symbol_match_among_multiple_explicit_sources(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "src" / "alpha.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
        (root / "src" / "beta.py").write_text("def add(left, right):\n    return 0\n", encoding="utf-8")
        (root / "tests" / "test_alpha.py").write_text(
            "import unittest\n"
            "from src.alpha import add\n\n"
            "class AlphaTests(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(add(1, 2), 3)\n",
            encoding="utf-8",
        )
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)

        probe = agent._pathless_mutation_grounding_probe(
            name="edit_intent",
            arguments={"intent": "replace_body", "target": "add", "replacement": "return left + right"},
            requested_mutation_paths={"src/alpha.py", "src/beta.py"},
            successful_tool_results=[],
        )

        self.assertEqual(probe, ("read_symbol", {"path": "src/alpha.py", "symbol": "add", "include_context": 0}))

    def test_contextual_explicit_source_path_prefers_recent_test_bridge_within_named_sources(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "src" / "alpha.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
        (root / "src" / "beta.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
        (root / "tests" / "test_alpha.py").write_text(
            "import unittest\n"
            "from src.alpha import add\n\n"
            "class AlphaTests(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(add(1, 2), 3)\n",
            encoding="utf-8",
        )
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)
        successful_tool_results = [
            {
                "name": "read_file",
                "arguments": {"path": "tests/test_alpha.py"},
                "result": {
                    "ok": True,
                    "path": "tests/test_alpha.py",
                    "output": "from src.alpha import add",
                },
            }
        ]

        self.assertEqual(
            agent._contextual_explicit_source_path(
                candidate_paths=["src/alpha.py", "src/beta.py"],
                successful_tool_results=successful_tool_results,
            ),
            "src/alpha.py",
        )

    def test_pathless_mutation_grounding_probe_prefers_recent_named_source_when_explicit_symbol_matches_tie(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "src" / "alpha.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
        (root / "src" / "beta.py").write_text("def add(left, right):\n    return 0\n", encoding="utf-8")
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)
        successful_tool_results = [
            {
                "name": "read_file",
                "arguments": {"path": "src/alpha.py"},
                "result": {
                    "ok": True,
                    "path": "src/alpha.py",
                    "output": "def add(left, right):\n    return left - right\n",
                },
            }
        ]

        probe = agent._pathless_mutation_grounding_probe(
            name="edit_intent",
            arguments={"intent": "replace_body", "target": "add", "replacement": "return left + right"},
            requested_mutation_paths={"src/alpha.py", "src/beta.py"},
            successful_tool_results=successful_tool_results,
        )

        self.assertEqual(probe, ("read_symbol", {"path": "src/alpha.py", "symbol": "add", "include_context": 0}))

    def test_pathless_mutation_grounding_probe_ignores_unrelated_recent_source_when_explicit_symbol_matches_tie(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "src" / "alpha.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
        (root / "src" / "beta.py").write_text("def add(left, right):\n    return 0\n", encoding="utf-8")
        (root / "src" / "other.py").write_text("def add(left, right):\n    return 7\n", encoding="utf-8")
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)
        successful_tool_results = [
            {
                "name": "read_file",
                "arguments": {"path": "src/other.py"},
                "result": {
                    "ok": True,
                    "path": "src/other.py",
                    "output": "def add(left, right):\n    return 7\n",
                },
            }
        ]

        probe = agent._pathless_mutation_grounding_probe(
            name="edit_intent",
            arguments={"intent": "replace_body", "target": "add", "replacement": "return left + right"},
            requested_mutation_paths={"src/alpha.py", "src/beta.py"},
            successful_tool_results=successful_tool_results,
        )

        self.assertEqual(probe, ("search_symbols", {"query": "add", "path": "."}))

    def test_explicit_source_path_for_symbol_keeps_ambiguity_when_test_affinity_ties(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "src" / "alpha.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
        (root / "src" / "beta.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
        (root / "tests" / "test_alpha.py").write_text(
            "import unittest\n"
            "from src.alpha import add\n\n"
            "class AlphaTests(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(add(1, 2), 3)\n",
            encoding="utf-8",
        )
        (root / "tests" / "test_beta.py").write_text(
            "import unittest\n"
            "from src.beta import add\n\n"
            "class BetaTests(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(add(1, 2), 3)\n",
            encoding="utf-8",
        )
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)

        self.assertIsNone(agent._explicit_source_path_for_symbol(candidate_paths=["src/alpha.py", "src/beta.py"], symbol="add"))

    def test_trajectory_ground_probe_retry_message_lists_ambiguous_repo_candidates(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "src" / "alpha.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
        (root / "src" / "beta.py").write_text("def add(left, right):\n    return 0\n", encoding="utf-8")
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)
        search_result = tools.search_symbols("add", path=".")

        message = agent._trajectory_ground_probe_retry_message(
            request_text="Fix add so it returns left + right.",
            probe_name="search_symbols",
            probe_arguments={"query": "add", "path": "."},
            probe_result=search_result,
            required_mutation_paths=set(),
            mutated_paths_this_turn=set(),
            test_run_required=False,
        )

        self.assertIn("Search found multiple implementation candidates for add: src/alpha.py, src/beta.py.", message)
        self.assertIn("Read the exact implementation symbol or name the target path before editing.", message)

    def test_trajectory_ground_probe_retry_message_guides_read_after_context_pack(self) -> None:
        root = self._workspace_scratch()
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)

        message = agent._trajectory_ground_probe_retry_message(
            request_text="Fix add in src/alpha.py and src/beta.py.",
            probe_name="context_pack",
            probe_arguments={"request": "Fix add in src/alpha.py and src/beta.py.", "path": ".", "limit": 6},
            probe_result={
                "ok": True,
                "tool": "context_pack",
                "path": ".",
                "suggested_next_tool": "read_symbol",
                "output": "context_pack:\nsuggested_next_tool=read_symbol",
            },
            required_mutation_paths={"src/alpha.py", "src/beta.py"},
            mutated_paths_this_turn=set(),
            test_run_required=False,
        )

        self.assertEqual(
            message,
            "Context pack ranked likely implementation matches. Read the most relevant implementation symbol now before editing. Next JSON only.",
        )

    def test_trajectory_ground_probe_retry_message_lists_ranked_context_pack_symbols(self) -> None:
        root = self._workspace_scratch()
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)

        message = agent._trajectory_ground_probe_retry_message(
            request_text="Fix add.",
            probe_name="context_pack",
            probe_arguments={"request": "Fix add.", "path": ".", "limit": 6},
            probe_result={
                "ok": True,
                "tool": "context_pack",
                "path": ".",
                "suggested_next_tool": "read_symbol",
                "ranked_paths": ["src/alpha.py", "src/beta.py"],
                "ranked_symbols": [
                    {"path": "src/alpha.py", "qualname": "add"},
                    {"path": "src/beta.py", "qualname": "add"},
                ],
            },
            required_mutation_paths=set(),
            mutated_paths_this_turn=set(),
            test_run_required=False,
        )

        self.assertIn("Context pack ranked multiple implementation symbols: src/alpha.py:add, src/beta.py:add.", message)
        self.assertIn("do not mutate from ranking alone", message)

    def test_trajectory_ground_probe_retry_message_lists_ranked_context_pack_files(self) -> None:
        root = self._workspace_scratch()
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)

        message = agent._trajectory_ground_probe_retry_message(
            request_text="Fix the parser.",
            probe_name="context_pack",
            probe_arguments={"request": "Fix the parser.", "path": ".", "limit": 6},
            probe_result={
                "ok": True,
                "tool": "context_pack",
                "path": ".",
                "suggested_next_tool": "read_file",
                "ranked_paths": ["src/cli.py", "src/parser.py"],
            },
            required_mutation_paths=set(),
            mutated_paths_this_turn=set(),
            test_run_required=False,
        )

        self.assertIn("Context pack ranked multiple implementation files: src/cli.py, src/parser.py.", message)
        self.assertIn("Read the intended ranked file before editing", message)

    def test_trajectory_ground_guard_maps_failed_test_to_implementation_before_pathless_edit(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "src" / "core.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
        (root / "tests" / "test_core.py").write_text(
            "import unittest\n"
            "from src.core import add\n\n"
            "class CoreTests(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(add(1, 2), 3)\n",
            encoding="utf-8",
        )
        fail_command = subprocess.list2cmdline(
            [
                sys.executable,
                "-c",
                "import sys; "
                "print('Traceback (most recent call last):'); "
                "print('  File \"tests/test_core.py\", line 6, in test_add'); "
                "print('    self.assertEqual(add(1, 2), 3)'); "
                "print('AssertionError: -1 != 3'); "
                "sys.exit(1)",
            ]
        )
        pass_command = subprocess.list2cmdline([sys.executable, "-c", "print('ok')"])
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": fail_command}}),
                '{"type":"tool","name":"edit_intent","arguments":{"intent":"replace_body","target":"add","replacement":"return left + right"}}',
                '{"type":"tool","name":"replace_symbol","arguments":{"path":"src/core.py","symbol":"add","content":"def add(left, right):\\n    return left + right\\n"}}',
                json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": pass_command}}),
                '{"type":"final","message":"fixed src/core.py"}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Run tests, fix the failing implementation, rerun tests, and summarize.")

        self.assertEqual(result.message, "fixed src/core.py")
        self.assertEqual(tools.execute_counts.get("find_implementation_target"), 1)
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_calls[:4], ["run_test", "find_implementation_target", "replace_symbol", "run_test"])
        self.assertTrue(any(event.get("type") == "controller_guard" and event.get("guard") == "ground-before-mutate" for event in agent.events))
        self.assertTrue(any("Use the grounded implementation target(s) src/core.py." in message["content"] for message in agent.messages if message["role"] == "user"))

    def test_trajectory_ground_guard_auto_diagnoses_failed_test_before_pathless_edit(self) -> None:
        root = self._workspace_scratch()
        (root / "app.py").write_text("def f():\n    return 0\n", encoding="utf-8")
        fail_command = subprocess.list2cmdline([sys.executable, "-c", "import sys; print('AssertionError: 0 != 1'); sys.exit(1)"])
        pass_command = subprocess.list2cmdline([sys.executable, "-c", "print('ok')"])
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": fail_command}}),
                json.dumps({"type": "tool", "name": "edit_intent", "arguments": {"intent": "replace_body", "target": "f", "replacement": "return 1"}}),
                json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": "def f():\n    return 1\n"}}),
                json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": pass_command}}),
                json.dumps({"type": "final", "message": "fixed app.py"}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Run tests, fix the bug, rerun tests, and summarize.")

        self.assertEqual(result.message, "fixed app.py")
        self.assertEqual(tools.execute_counts.get("diagnose_test_failure"), 1)
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_calls[:4], ["run_test", "diagnose_test_failure", "write_file", "run_test"])
        self.assertTrue(any(event.get("type") == "controller_guard" and event.get("guard") == "ground-before-mutate" for event in agent.events))
        self.assertTrue(any("Use the diagnosis above to edit implementation, then run_test." in message["content"] for message in agent.messages if message["role"] == "user"))

    def test_trajectory_ground_guard_allows_explicit_new_file_creation(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient(
            [
                '{"type":"tool","name":"write_file","arguments":{"path":"new_app.py","content":"def value():\\n    return 1\\n"}}',
                '{"type":"final","message":"Created new_app.py."}',
            ]
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=4)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Create new_app.py with a value function.")

        self.assertTrue(result.completed)
        self.assertTrue((root / "new_app.py").exists())
        self.assertFalse(any(event.get("type") == "controller_guard" and event.get("guard") == "ground-before-mutate" for event in agent.events))
        self.assertTrue(any(event.get("type") == "auto_validation" for event in agent.events))

    def test_tool_error_guard_blocks_third_duplicate_path_failure(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"missing.py"}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"missing.py"}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"missing.py"}}',
                '{"type":"final","message":"Path is missing."}',
                '{"type":"final","message":"Path is missing."}',
                '{"type":"final","message":"Path is missing."}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=5)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Check missing.py if useful, but do not loop.")

        self.assertFalse(result.completed)
        tool_calls = [event for event in agent.events if event.get("type") == "tool_call" and event.get("name") == "read_file"]
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tools.execute_counts.get("diagnose_dependency_error"), 1)
        guard_events = [event for event in agent.events if event.get("type") == "tool_error_guard"]
        self.assertEqual(len(guard_events), 1)
        self.assertEqual(guard_events[0].get("error_class"), "path_missing")
        controller_guards = [event for event in agent.events if event.get("type") == "controller_guard" and event.get("guard") == "path-repair-guard"]
        self.assertEqual(len(controller_guards), 1)
        self.assertTrue(any("Use the diagnosis above to repair the path/cwd" in message["content"] for message in agent.messages if message["role"] == "user"))

    def test_path_missing_on_single_source_repo_auto_grounds_real_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task_cli.py").write_text("def main() -> int:\n    return 0\n", encoding="utf-8")
            client = FakeClient(
                [
                    '{"type":"tool","name":"code_outline","arguments":{"path":"tasks.py"}}',
                    '{"type":"final","message":"No implementation found."}',
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=2)

            with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
                agent.handle_user("Add a stats command to this CLI repo and keep tests green.")

        tool_calls = [event for event in agent.events if event.get("type") == "tool_call"]
        outline_paths = [event.get("arguments", {}).get("path") for event in tool_calls if event.get("name") == "code_outline"]
        self.assertEqual(outline_paths[:2], ["tasks.py", "task_cli.py"])
        self.assertTrue(
            any(event.get("type") == "controller_guard" and event.get("guard") == "single-source-path-fallback" for event in agent.events)
        )
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("only non-test Python source was grounded instead: task_cli.py", feedback)

    def test_missing_mutation_target_blocks_before_python_payload_syntax_guard(self) -> None:
        root = self._workspace_scratch()
        (root / "reports").mkdir()
        (root / "reports" / "__init__.py").write_text("from .exporter import ReportRow\n", encoding="utf-8")
        client = FakeClient(
            [
                json.dumps(
                    {
                        "type": "tool",
                        "name": "edit_intent",
                        "arguments": {
                            "path": "__init__.py",
                            "intent": "add_function",
                            "target": "export_ndjson",
                            "replacement": "def export_ndjson(rows):\n    \"unterminated\n",
                        },
                    }
                ),
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "reports/__init__.py"}}),
                *[json.dumps({"type": "final", "message": "grounded package init"}) for _ in range(6)],
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=4)

        result = agent.handle_user("Add export_ndjson to the package __init__.py for this report exporter.")

        self.assertFalse(result.completed)
        self.assertIsNone(tools.execute_counts.get("edit_intent"))
        self.assertEqual(tools.execute_counts.get("read_file"), 1)
        guard_names = [event.get("guard") for event in agent.events if event.get("type") == "controller_guard"]
        self.assertIn("missing-mutation-target", guard_names)
        self.assertNotIn("invalid-python-mutation-payload", guard_names)
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Read the existing candidate first: reports/__init__.py", feedback)

    def test_add_function_to_package_init_redirects_to_backing_module(self) -> None:
        root = self._workspace_scratch()
        (root / "logtools").mkdir()
        (root / "logtools" / "__init__.py").write_text(
            "from .summary import LogEvent, summarize_by_level\n\n"
            "__all__ = [\"LogEvent\", \"summarize_by_level\"]\n",
            encoding="utf-8",
        )
        (root / "logtools" / "summary.py").write_text(
            "class LogEvent:\n"
            "    pass\n\n\n"
            "def summarize_by_level(events):\n"
            "    return {}\n",
            encoding="utf-8",
        )
        invalid_replacement = (
            "def slowest_services(events, limit=3):\n"
            "    \"Return slowest services.\n"
            "    return []\n"
        )
        client = FakeClient(
            [
                json.dumps(
                    {
                        "type": "tool",
                        "name": "edit_intent",
                        "arguments": {
                            "path": "logtools/__init__.py",
                            "intent": "add_function",
                            "target": "slowest_services",
                            "replacement": invalid_replacement,
                        },
                    }
                ),
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "logtools/summary.py"}}),
                *[json.dumps({"type": "final", "message": "grounded implementation module"}) for _ in range(6)],
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=4)

        result = agent.handle_user("Add slowest_services to the package __init__.py and export it.")

        self.assertFalse(result.completed)
        self.assertIsNone(tools.execute_counts.get("edit_intent"))
        self.assertEqual(tools.execute_counts.get("read_file"), 1)
        guards = [event for event in agent.events if event.get("type") == "controller_guard"]
        self.assertTrue(any(event.get("guard") == "package-init-implementation-target" for event in guards))
        self.assertFalse(any(event.get("guard") == "invalid-python-mutation-payload" for event in guards))
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Read and edit the backing implementation module `logtools/summary.py` first", feedback)

    def test_package_init_wildcard_export_is_rejected_before_write(self) -> None:
        root = self._workspace_scratch()
        (root / "analytics").mkdir()
        (root / "analytics" / "__init__.py").write_text(
            "from .events import RequestEvent, summarize_status\n\n"
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
        client = FakeClient(
            [
                json.dumps(
                    {
                        "type": "tool",
                        "name": "write_file",
                        "arguments": {"path": "analytics/__init__.py", "content": "from .events import *\n"},
                    }
                ),
                *[json.dumps({"type": "final", "message": "exported"}) for _ in range(4)],
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=3)

        result = agent.handle_user("Add percentile_latency to the package and export it from analytics/__init__.py.")

        self.assertFalse(result.completed)
        self.assertIsNone(tools.execute_counts.get("write_file"))
        self.assertEqual(
            (root / "analytics" / "__init__.py").read_text(encoding="utf-8"),
            "from .events import RequestEvent, summarize_status\n\n__all__ = [\"RequestEvent\", \"summarize_status\"]\n",
        )
        guards = [event for event in agent.events if event.get("type") == "controller_guard"]
        self.assertTrue(any(event.get("guard") == "package-init-wildcard-export" for event in guards))
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Use explicit named imports and update `__all__`", feedback)

    def test_package_init_implementation_write_is_rejected_before_execution(self) -> None:
        root = self._workspace_scratch()
        (root / "analytics").mkdir()
        (root / "analytics" / "__init__.py").write_text(
            "from .events import RequestEvent, summarize_status\n\n"
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
        polluted_init = (
            "from .events import RequestEvent\n\n\n"
            "def percentile_latency(events, percentile=95):\n"
            "    return None\n\n\n"
            "class AnalyticsTests:\n"
            "    def test_percentile_latency(self):\n"
            "        assert percentile_latency([]) is None\n"
        )
        client = FakeClient(
            [
                json.dumps(
                    {
                        "type": "tool",
                        "name": "write_file",
                        "arguments": {"path": "analytics/__init__.py", "content": polluted_init},
                    }
                ),
                *[json.dumps({"type": "final", "message": "done"}) for _ in range(4)],
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=3)

        result = agent.handle_user(
            "Add percentile_latency to this analytics package, export it, update README, add tests, and run tests."
        )

        self.assertFalse(result.completed)
        self.assertIsNone(tools.execute_counts.get("write_file"))
        self.assertEqual(
            (root / "analytics" / "__init__.py").read_text(encoding="utf-8"),
            "from .events import RequestEvent, summarize_status\n\n__all__ = [\"RequestEvent\", \"summarize_status\"]\n",
        )
        guards = [event for event in agent.events if event.get("type") == "controller_guard"]
        self.assertTrue(any(event.get("guard") == "package-init-implementation-write" for event in guards))
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Do not write feature implementations or test classes into package export file", feedback)
        self.assertIn("Put implementation code in `analytics/events.py` first", feedback)

    def test_repeated_add_function_after_success_routes_to_remaining_obligations(self) -> None:
        root = self._workspace_scratch()
        (root / "analytics").mkdir()
        (root / "analytics" / "__init__.py").write_text(
            "from .events import RequestEvent, summarize_status\n\n"
            "__all__ = [\"RequestEvent\", \"summarize_status\"]\n",
            encoding="utf-8",
        )
        (root / "analytics" / "events.py").write_text(
            "class RequestEvent:\n"
            "    def __init__(self, duration_ms):\n"
            "        self.duration_ms = duration_ms\n\n\n"
            "def summarize_status(events):\n"
            "    return {}\n",
            encoding="utf-8",
        )
        (root / "tests").mkdir()
        (root / "tests" / "test_events.py").write_text(
            "from analytics import summarize_status\n\n\n"
            "def test_summarize_status():\n"
            "    assert summarize_status([]) == {}\n",
            encoding="utf-8",
        )
        valid_add = (
            "def percentile_latency(events, percentile=95):\n"
            "    durations = sorted(event.duration_ms for event in events if event.duration_ms >= 0)\n"
            "    return durations[-1] if durations else None\n"
        )
        invalid_repeat = (
            "def percentile_latency(events, percentile=95):\n"
            "    \"Return nearest rank percentile.\n"
            "    return None\n"
        )
        client = FakeClient(
            [
                json.dumps(
                    {
                        "type": "tool",
                        "name": "edit_intent",
                        "arguments": {
                            "path": "analytics/events.py",
                            "intent": "add_function",
                            "target": "percentile_latency",
                            "replacement": valid_add,
                        },
                    }
                ),
                json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": "python -m unittest discover -s tests -v"}}),
                json.dumps({"type": "final", "message": "Implemented percentile_latency."}),
                json.dumps(
                    {
                        "type": "tool",
                        "name": "edit_intent",
                        "arguments": {
                            "path": "analytics/events.py",
                            "intent": "add_function",
                            "target": "percentile_latency",
                            "replacement": invalid_repeat,
                        },
                    }
                ),
                json.dumps({"type": "final", "message": "done"}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto", test_command="python -m unittest discover -s tests -v")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=5)

        result = agent.handle_user(
            "Add percentile_latency, update README, add tests for it, run tests, and prove it with a shell command."
        )

        self.assertFalse(result.completed)
        self.assertEqual(tools.execute_counts.get("edit_intent"), 1)
        guards = [event for event in agent.events if event.get("type") == "controller_guard"]
        self.assertTrue(any(event.get("guard") == "repeated-add-function-after-success" for event in guards))
        self.assertFalse(any(event.get("guard") == "invalid-python-mutation-payload" for event in guards))
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Do not use add_function for the same symbol again", feedback)
        self.assertIn("Remaining unproven deliverables", feedback)

    def test_replace_body_full_function_payload_is_rejected_before_tool_execution(self) -> None:
        root = self._workspace_scratch()
        (root / "analytics.py").write_text(
            "def percentile_latency(events, percentile=95):\n"
            "    return None\n",
            encoding="utf-8",
        )
        client = FakeClient(
            [
                json.dumps(
                    {
                        "type": "tool",
                        "name": "edit_intent",
                        "arguments": {
                            "path": "analytics.py",
                            "intent": "replace_body",
                            "target": "percentile_latency",
                            "replacement": "def percentile_latency(events, percentile=95):\n    return 1\n",
                        },
                    }
                ),
                *[json.dumps({"type": "final", "message": "updated"}) for _ in range(3)],
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=3)

        result = agent.handle_user("Fix percentile_latency in analytics.py and run tests.")

        self.assertFalse(result.completed)
        self.assertIsNone(tools.execute_counts.get("edit_intent"))
        guards = [event for event in agent.events if event.get("type") == "controller_guard"]
        self.assertTrue(any(event.get("guard") == "replace-body-full-definition-payload" for event in guards))
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("replace_body", feedback)
        self.assertIn("use replace_symbol", feedback)

    def test_write_file_with_quote_prefixed_source_is_rejected_before_execution(self) -> None:
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
                            "content": "> def value() -> int:\n>     return 2\n> \n> def other() -> int:\n>     return 3\n",
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
                event.get("type") == "controller_guard" and event.get("guard") == "write-file-quote-prefixed-content"
                for event in agent.events
            )
        )
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Remove every leading `>` quote marker", feedback)

    def test_write_file_dropping_existing_symbols_is_rejected_before_execution(self) -> None:
        root = self._workspace_scratch()
        original = (
            "class RequestEvent:\n"
            "    pass\n\n\n"
            "def summarize_status(events):\n"
            "    return {}\n"
        )
        (root / "analytics.py").write_text(original, encoding="utf-8")
        client = FakeClient(
            [
                json.dumps(
                    {
                        "type": "tool",
                        "name": "write_file",
                        "arguments": {
                            "path": "analytics.py",
                            "content": (
                                "class RequestEvent:\n"
                                "    pass\n\n\n"
                                "def percentile_latency(events, percentile=95):\n"
                                "    return None\n"
                            ),
                        },
                    }
                ),
                json.dumps({"type": "final", "message": "stopped"}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=2)

        agent.handle_user("Add percentile_latency to analytics.py without removing existing behavior.")

        self.assertIsNone(tools.execute_counts.get("write_file"))
        self.assertEqual((root / "analytics.py").read_text(encoding="utf-8"), original)
        self.assertTrue(
            any(
                event.get("type") == "controller_guard"
                and event.get("guard") == "write-file-drops-existing-python-symbols"
                and event.get("dropped_symbols") == ["summarize_status"]
                for event in agent.events
            )
        )
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("summarize_status", feedback)
        self.assertIn("preserve existing public API", feedback)

    def test_write_file_inventing_object_model_after_failed_symbol_edit_is_rejected(self) -> None:
        root = self._workspace_scratch()
        original = (
            "from __future__ import annotations\n\n"
            "import argparse\n"
            "import json\n"
            "from pathlib import Path\n\n"
            "DATA_FILE = Path(\"tasks.json\")\n\n\n"
            "def load_tasks(path: Path = DATA_FILE) -> list[dict[str, str]]:\n"
            "    if not path.exists():\n"
            "        return []\n"
            "    return json.loads(path.read_text(encoding=\"utf-8\"))\n\n\n"
            "def save_tasks(tasks: list[dict[str, str]], path: Path = DATA_FILE) -> None:\n"
            "    path.write_text(json.dumps(tasks), encoding=\"utf-8\")\n\n\n"
            "def add_task(title: str, priority: str = \"normal\", path: Path = DATA_FILE) -> dict[str, str]:\n"
            "    tasks = load_tasks(path)\n"
            "    task = {\"title\": title, \"priority\": priority, \"status\": \"todo\"}\n"
            "    tasks.append(task)\n"
            "    save_tasks(tasks, path)\n"
            "    return task\n\n\n"
            "def list_tasks(path: Path = DATA_FILE, *, priority: str | None = None) -> list[dict[str, str]]:\n"
            "    tasks = load_tasks(path)\n"
            "    if priority:\n"
            "        tasks = [task for task in tasks if task.get(\"priority\") == priority]\n"
            "    return tasks\n\n\n"
            "def build_parser() -> argparse.ArgumentParser:\n"
            "    parser = argparse.ArgumentParser(description=\"Manage local tasks\")\n"
            "    parser.add_argument(\"--priority\")\n"
            "    return parser\n\n\n"
            "def main(argv: list[str] | None = None) -> int:\n"
            "    return 0\n"
        )
        invented_rewrite = (
            "from __future__ import annotations\n"
            "import argparse\n"
            "import json\n"
            "from datetime import datetime\n"
            "from pathlib import Path\n\n"
            "class TaskCLI:\n"
            "   DATA_FILE = Path(\"tasks.json\")\n"
            "   @classmethod\n"
            "   def load_tasks(cls) -> list[Task]:\n"
            "       if not cls.DATA_FILE.exists():\n"
            "           return []\n"
            "       raw = json.loads(cls.DATA_FILE.read_text())\n"
            "       return [Task(**t) for t in raw]\n"
            "   @classmethod\n"
            "   def run_list_command(cls, args: argparse.Namespace) -> None:\n"
            "       tasks = cls.load_tasks()\n"
            "       for task in tasks:\n"
            "           print(task.title)\n\n"
            "class Task:\n"
            "   def __init__(self, title: str, due_date: str):\n"
            "       self.title = title\n"
            "       self.due_date = datetime.strptime(due_date, \"%Y-%m-%d\").date()\n"
            "if __name__ == \"__main__\":\n"
            "   exit(TaskCLI.main())\n"
        )
        (root / "task_cli.py").write_text(original, encoding="utf-8")
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "task_cli.py"}}),
                json.dumps(
                    {
                        "type": "tool",
                        "name": "edit_intent",
                        "arguments": {
                            "path": "task_cli.py",
                            "intent": "replace_symbol",
                            "target": "TaskCLI.list_command",
                            "replacement": "def list_command(args):\n    pass\n",
                        },
                    }
                ),
                json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "task_cli.py", "content": invented_rewrite}}),
                json.dumps({"type": "final", "message": "stopped"}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=4)

        agent.handle_user("Add --due-before to task_cli.py without removing existing functions.")

        self.assertIsNone(tools.execute_counts.get("write_file"))
        self.assertEqual((root / "task_cli.py").read_text(encoding="utf-8"), original)
        guard_events = [
            event
            for event in agent.events
            if event.get("type") == "controller_guard" and event.get("guard") == "write-file-drops-existing-python-symbols"
        ]
        self.assertEqual(len(guard_events), 1)
        self.assertIn("add_task", guard_events[0].get("dropped_symbols") or [])
        self.assertIn("list_tasks", guard_events[0].get("dropped_symbols") or [])

    def test_failed_edit_recovery_guard_requires_reground_then_broad_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = (
                "from __future__ import annotations\n\n"
                "def stats_lines() -> list[str]:\n"
                "    return ['todo: 1']\n"
            )
            (root / "task_cli.py").write_text(source, encoding="utf-8")
            client = FakeClient(['{"verdict":"accept","reason":"grounded broader repair"}'])
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
            successful_tool_results = [
                {
                    "name": "read_file",
                    "arguments": {"path": "task_cli.py"},
                    "result": {"ok": True, "path": "task_cli.py", "output": source},
                }
            ]

            agent._set_failed_edit_recovery_state(
                name="replace_in_file",
                arguments={"path": "task_cli.py", "old": "return ['todo: 1']\n", "new": "return stats_lines(priority)\n"},
                successful_tool_results=successful_tool_results,
                validation_name="run_test",
                diagnostic="TypeError: stats_lines() takes 0 positional arguments but 1 was given",
            )
            blocked_before_reground = agent._audit_tool_candidate(
                request_text="Add a stats command and keep tests green.",
                round_number=1,
                proposed_tool_name="replace_in_file",
                proposed_arguments={"path": "task_cli.py", "old": "return ['todo: 1']\n", "new": "return stats_lines(priority)\n"},
                tool_calls=[],
                successful_tool_results=successful_tool_results,
                accepted_assumption_audits=[],
                required_tool_names=set(),
                forbidden_tool_names=set(),
                mutation_allowed=True,
                expected_exact_file_line=None,
                expected_exact_reply_text=None,
            )
            agent._record_event(
                "tool_result",
                name="read_file",
                arguments={"path": "task_cli.py"},
                result={"ok": True, "path": "task_cli.py", "output": source},
            )
            blocked_after_reground = agent._audit_tool_candidate(
                request_text="Add a stats command and keep tests green.",
                round_number=2,
                proposed_tool_name="replace_in_file",
                proposed_arguments={"path": "task_cli.py", "old": "return ['todo: 1']\n", "new": "return stats_lines(priority)\n"},
                tool_calls=[],
                successful_tool_results=successful_tool_results,
                accepted_assumption_audits=[],
                required_tool_names=set(),
                forbidden_tool_names=set(),
                mutation_allowed=True,
                expected_exact_file_line=None,
                expected_exact_reply_text=None,
            )
            accepted_broad_repair = agent._audit_tool_candidate(
                request_text="Add a stats command and keep tests green.",
                round_number=3,
                proposed_tool_name="write_file",
                proposed_arguments={"path": "task_cli.py", "content": source + "\n# repaired\n"},
                tool_calls=[],
                successful_tool_results=successful_tool_results,
                accepted_assumption_audits=[],
                required_tool_names=set(),
                forbidden_tool_names=set(),
                mutation_allowed=True,
                expected_exact_file_line=None,
                expected_exact_reply_text=None,
            )

        self.assertEqual(blocked_before_reground["verdict"], "retry")
        self.assertIn("Re-ground task_cli.py", blocked_before_reground["reason"])
        self.assertEqual(blocked_after_reground["verdict"], "retry")
        self.assertIn("broader", blocked_after_reground["reason"])
        self.assertEqual(accepted_broad_repair["verdict"], "accept")

    def test_repair_pivot_model_timeout_fails_closed(self) -> None:
        class TimeoutAfterScriptedClient(FakeClient):
            def chat(
                self,
                *,
                model: str,
                messages: list[dict[str, str]],
                response_format: str = "json",
                on_thinking: object | None = None,
                think: bool | None = None,
                options: dict[str, object] | None = None,
            ) -> ChatResponse:
                if not self.responses:
                    raise OllamaError("Ollama timed out after 60 seconds.")
                return super().chat(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    on_thinking=on_thinking,
                    think=think,
                    options=options,
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def add(left, right):\n    pass\n", encoding="utf-8")
            client = TimeoutAfterScriptedClient(
                [
                    json.dumps({"type": "tool", "name": "edit_intent", "arguments": {"path": "app.py", "intent": "replace_text", "target": "return missing", "replacement": "return left + right"}}),
                    json.dumps({"type": "tool", "name": "edit_intent", "arguments": {"path": "app.py", "intent": "replace_text", "target": "return other_missing", "replacement": "return left + right"}}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

            with patch.object(OllamaCodeAgent, "_spec_guided_repair_has_actionable_spec", return_value=False):
                result = agent.handle_user("Fix app.py.")

        self.assertFalse(result.completed)
        self.assertIn("timed out while responding to the bounded repair prompt", result.message)
        self.assertEqual(tools.execute_counts.get("edit_intent"), 1)
        guards = [event for event in agent.events if event.get("guard") == "repeated-mutating-failure-pivot"]
        self.assertEqual(len(guards), 1)

    def test_final_round_repeated_mutating_failure_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def add(left, right):\n    pass\n", encoding="utf-8")
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "edit_intent", "arguments": {"path": "app.py", "intent": "replace_text", "target": "return missing", "replacement": "return left + right"}}),
                    json.dumps({"type": "tool", "name": "edit_intent", "arguments": {"path": "app.py", "intent": "replace_text", "target": "return other_missing", "replacement": "return left + right"}}),
                    json.dumps({"type": "final", "message": "should not be reached"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=2)

            with patch.object(OllamaCodeAgent, "_spec_guided_repair_has_actionable_spec", return_value=False):
                result = agent.handle_user("Fix app.py.")

        self.assertFalse(result.completed)
        self.assertIn("repeated failed mutation reached the final tool round", result.message)
        self.assertIn("Stop retrying narrow edit operations on app.py", result.message)
        self.assertEqual(tools.execute_counts.get("edit_intent"), 1)
        self.assertEqual(len(client.responses), 1)
        guards = [event for event in agent.events if event.get("guard") == "repeated-mutating-failure-pivot"]
        self.assertEqual(len(guards), 1)

    def test_failed_edit_recovery_guard_requires_behavior_surface_read_for_cli_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            source = (
                "from __future__ import annotations\n\n"
                "def main(argv: list[str] | None = None) -> int:\n"
                "    return 0\n"
            )
            test_source = (
                "import unittest\n\n"
                "class TaskCliTests(unittest.TestCase):\n"
                "    def test_list(self) -> None:\n"
                "        self.assertTrue(True)\n"
            )
            (root / "task_cli.py").write_text(source, encoding="utf-8")
            (root / "tests" / "test_task_cli.py").write_text(test_source, encoding="utf-8")
            client = FakeClient(['{"verdict":"accept","reason":"grounded broader repair"}'])
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
            successful_tool_results = [
                {
                    "name": "read_file",
                    "arguments": {"path": "task_cli.py"},
                    "result": {"ok": True, "path": "task_cli.py", "output": source},
                }
            ]

            agent._set_failed_edit_recovery_state(
                name="replace_in_file",
                arguments={"path": "task_cli.py", "old": "    return 0\n", "new": "    return 1\n"},
                successful_tool_results=successful_tool_results,
                validation_name="run_test",
                diagnostic="test_list failed after the previous edit",
                request_obligations=[
                    {"id": "command:stats", "kind": "feature_token", "label": 'prove the "stats" command exists', "token": "stats", "feature_class": "command"},
                    {"id": "flag:--priority", "kind": "feature_token", "label": 'prove the "--priority" flag exists', "token": "--priority", "feature_class": "flag"},
                ],
            )
            agent._record_event(
                "tool_result",
                name="read_file",
                arguments={"path": "task_cli.py"},
                result={"ok": True, "path": "task_cli.py", "output": source},
            )
            blocked_without_behavior_read = agent._audit_tool_candidate(
                request_text="Add a stats command and --priority flag to task_cli.py and keep tests green.",
                round_number=2,
                proposed_tool_name="write_file",
                proposed_arguments={"path": "task_cli.py", "content": source + "\n# repaired\n"},
                tool_calls=[],
                successful_tool_results=successful_tool_results,
                accepted_assumption_audits=[],
                required_tool_names=set(),
                forbidden_tool_names=set(),
                mutation_allowed=True,
                expected_exact_file_line=None,
                expected_exact_reply_text=None,
            )
            agent._record_event(
                "tool_result",
                name="read_file",
                arguments={"path": "tests/test_task_cli.py"},
                result={"ok": True, "path": "tests/test_task_cli.py", "output": test_source},
            )
            accepted_after_behavior_read = agent._audit_tool_candidate(
                request_text="Add a stats command and --priority flag to task_cli.py and keep tests green.",
                round_number=3,
                proposed_tool_name="write_file",
                proposed_arguments={"path": "task_cli.py", "content": source + "\n# repaired\n"},
                tool_calls=[],
                successful_tool_results=successful_tool_results,
                accepted_assumption_audits=[],
                required_tool_names=set(),
                forbidden_tool_names=set(),
                mutation_allowed=True,
                expected_exact_file_line=None,
                expected_exact_reply_text=None,
            )

        self.assertEqual(blocked_without_behavior_read["verdict"], "retry")
        self.assertIn("behavior surface", blocked_without_behavior_read["reason"])
        self.assertIn("test_task_cli.py", " ".join(blocked_without_behavior_read["validation_steps"]))
        self.assertEqual(accepted_after_behavior_read["verdict"], "accept")

    def test_spec_guided_repair_uses_context_pack_test_files_as_recent_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)
            successful_tool_results = [
                {
                    "name": "context_pack",
                    "arguments": {"request": "add stats command", "path": ".", "limit": 6},
                    "result": {
                        "ok": True,
                        "tool": "context_pack",
                        "test_files": ["tests/test_task_cli.py"],
                        "output": "context_pack:\ntest_files=tests/test_task_cli.py",
                    },
                }
            ]

            paths = agent._recent_test_paths(successful_tool_results)

        self.assertEqual(paths, ["tests/test_task_cli.py"])

    def test_failed_edit_recovery_state_carries_into_later_mutation_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def add(left: int, right: int) -> int:\n    return left + right\n", encoding="utf-8")
            client = FakeClient(
                [
                    json.dumps(
                        {
                            "type": "tool",
                            "name": "replace_in_file",
                            "arguments": {
                                "path": "app.py",
                                "old": "return left + right\n",
                                "new": "return add(1)\n",
                            },
                        }
                    ),
                    '{"type":"final","message":"Still working on it."}',
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=True, max_tool_rounds=2)
            agent._sticky_failed_edit_recovery = [
                {
                    "target_id": "path:app.py",
                    "kind": "path",
                    "path": "app.py",
                    "symbol": "",
                    "tool_name": "replace_in_file",
                    "tool_granularity": "narrow",
                    "validation_name": "run_test",
                    "diagnostic": "TypeError: add() missing 1 required positional argument: 'right'",
                    "failure_event_index": -1,
                }
            ]

            result = agent.handle_user("Actually implement app.py and keep tests green.")

        self.assertFalse(result.completed)
        self.assertEqual(tools.execute_counts.get("replace_in_file", 0), 0)
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Re-ground app.py", feedback)
        self.assertIn("broader repair", feedback)
