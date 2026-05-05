from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from ollama_code.agent import GROUNDING_EVIDENCE_TOOL_NAMES, OllamaCodeAgent, _workspace_roots_match
from ollama_code.features import ENV_OLLAMA_CODE_FEATURE_PROFILE
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
        script_reconciliation: bool = False,
    ) -> None:
        self.responses = list(responses)
        self.script_verification = script_verification
        self.script_assumption_audit = script_assumption_audit
        self.script_final_rewrite = script_final_rewrite
        self.script_reconciliation = script_reconciliation
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
        options: dict[str, object] | None = None,
    ) -> ChatResponse:
        system_prompt = messages[0]["content"] if messages else ""
        self.calls.append(
            {
                "model": model,
                "messages": list(messages),
                "response_format": response_format,
                "on_thinking": on_thinking,
                "think": think,
                "options": options,
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
            if system_prompt.startswith("You are an artifact reconciliation critic") and not self.script_reconciliation:
                return ChatResponse(content='{"verdict":"accept","reason":"","repair_plan":[],"required_tools":[],"forbidden_tools":[]}', model=model, raw={})
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

    def _init_git_repo_or_skip(self, root: Path) -> None:
        if shutil.which("git") is None:
            self.skipTest("git is not installed")
        try:
            subprocess.run(["git", "init"], cwd=root, capture_output=True, text=True, check=True)
            subprocess.run(["git", "config", "user.name", "Tests"], cwd=root, capture_output=True, text=True, check=True)
            subprocess.run(["git", "config", "user.email", "tests@example.com"], cwd=root, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            message = (exc.stderr or exc.stdout or str(exc)).strip()
            self.skipTest(f"git repo init is unavailable in this environment: {message}")

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
        self.assertTrue(any(event.get("type") == "tool_call" and event.get("name") == "read_file" for event in agent.events))
        tool_result = next(event for event in agent.events if event.get("type") == "tool_result" and event.get("name") == "read_file")
        self.assertIsInstance(tool_result.get("duration_ms"), float)

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
        self.assertIn("system", llm_calls[0]["prompt_chars_by_role"])
        self.assertGreater(llm_calls[0]["top_prompt_messages"][0]["chars"], 0)

    def test_agent_uses_schema_and_num_predict_feature_profile(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient(['{"type":"final","message":"ok"}'])
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "schema,num-predict-caps"}):
            result = agent.handle_user("say ok")

        self.assertEqual(result.message, "ok")
        call = client.calls[0]
        self.assertIsInstance(call["response_format"], dict)
        self.assertEqual(call["options"], {"num_predict": 256})

    def test_primary_think_defaults_off_for_broad_coding_prompt(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient([])
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

        think = agent._primary_think_override(
            request_text="Implement this Python exercise, read tests and source, edit implementation files, and run tests.",
            requires_tools=False,
            mutation_required=True,
            test_run_required=True,
            round_number=1,
            tool_used_this_turn=False,
        )

        self.assertFalse(think)

    def test_primary_think_keeps_default_for_simple_non_tool_prompt(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient([])
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

        think = agent._primary_think_override(
            request_text="Say ok.",
            requires_tools=False,
            mutation_required=False,
            test_run_required=False,
            round_number=1,
            tool_used_this_turn=False,
        )

        self.assertIsNone(think)

    def test_context_pack_preload_requires_path_for_focused_edit_prompt(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient([])
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "context-pack"}):
            should_preload = agent._should_preload_context_pack(
                request_text="Fix src/app.py and run tests.",
                session_memory_request=False,
                mutation_required=True,
                test_run_required=True,
                required_tool_names=set(),
                forbidden_tool_names=set(),
            )
            should_skip = agent._should_preload_context_pack(
                request_text="Implement this Python exercise, read tests and source, edit implementation files, and run tests.",
                session_memory_request=False,
                mutation_required=True,
                test_run_required=True,
                required_tool_names=set(),
                forbidden_tool_names=set(),
            )

        self.assertTrue(should_preload)
        self.assertFalse(should_skip)

    def test_trajectory_loop_cap_blocks_fourth_context_tool(self) -> None:
        root = self._workspace_scratch()
        (root / "note.txt").write_text("hello world\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                '{"type":"tool","name":"search","arguments":{"query":"hello"}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"note.txt","start":1,"end":1}}',
                '{"type":"tool","name":"search","arguments":{"query":"world"}}',
                '{"type":"final","message":"hello world"}',
            ]
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=5)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Inspect this repo and summarize the relevant text.")

        self.assertEqual(result.message, "hello world")
        self.assertTrue(any(event.get("type") == "controller_guard" and event.get("guard") == "loop-cap" for event in agent.events))
        self.assertEqual([event.get("name") for event in agent.events if event.get("type") == "tool_call"], ["read_file", "search", "read_file"])

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
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=5)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Check missing.py if useful, but do not loop.")

        self.assertFalse(result.completed)
        tool_calls = [event for event in agent.events if event.get("type") == "tool_call" and event.get("name") == "read_file"]
        self.assertEqual(len(tool_calls), 2)
        guard_events = [event for event in agent.events if event.get("type") == "tool_error_guard"]
        self.assertEqual(len(guard_events), 1)
        self.assertEqual(guard_events[0].get("error_class"), "path_missing")

    def test_command_validation_event_records_rejected_common_command(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient(
            [
                '{"type":"tool","name":"run_shell","arguments":{"command":"git reset --hard"}}',
                '{"type":"final","message":"Command rejected."}',
                '{"type":"final","message":"Command rejected."}',
                '{"type":"final","message":"Command rejected."}',
            ]
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=4)

        result = agent.handle_user("Try a shell command if useful and summarize.")

        self.assertFalse(result.completed)
        validation_events = [event for event in agent.events if event.get("type") == "command_validation"]
        self.assertEqual(len(validation_events), 1)
        self.assertFalse(validation_events[0].get("valid"))
        self.assertEqual(validation_events[0].get("family"), "git")

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

    def test_trajectory_failure_delta_compacts_repeated_test_failure(self) -> None:
        root = self._workspace_scratch()
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)

        delta = agent._failure_delta_summary(
            "FAILED test_ops.py::test_value | AssertionError: expected 1 got 0",
            "FAILED test_ops.py::test_value | AssertionError: expected 1 got 2",
        )

        self.assertIn("expected 1 got 2", delta)
        self.assertNotIn("expected 1 got 0", delta)

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

    def test_system_prompt_requires_assumption_checking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=12)

        prompt = agent.messages[0]["content"]
        self.assertIn("Question your assumptions before acting", prompt)
        self.assertIn("prove or disprove with tools", prompt)
        self.assertIn("do not guess", prompt)
        self.assertIn("prefer search_symbols, code_outline, then read_symbol before broad read_file", prompt)
        self.assertIn("use systems_lens early", prompt)
        self.assertIn("explicit boundary, observer/metric, categories, state/scale", prompt)
        self.assertIn("feedback, delays, stocks/flows, coupling, model limits, and intervention tests", prompt)

    def test_system_prompt_enables_caveman_lite_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        prompt = agent.messages[0]["content"]
        self.assertIn("caveman-lite concise", prompt)
        self.assertIn("keep code, paths, commands, errors, JSON exact", prompt)
        self.assertIn("syntactically complete", prompt)

    def test_primary_tools_include_systems_lens_for_complex_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        selected = agent._primary_tool_names_for_request(
            "Profile the slow edit pipeline and debug the controller design.",
            requires_tools=True,
            session_memory_request=False,
            mutation_allowed=False,
            mutation_required=False,
            test_run_required=False,
            required_tool_names=set(),
            forbidden_tool_names=set(),
        )
        self.assertIn("systems_lens", selected)
        self.assertNotIn("systems_lens", GROUNDING_EVIDENCE_TOOL_NAMES)

    def test_primary_prompt_omits_tool_signatures_for_simple_final(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient(['{"type":"final","message":"done"}'])
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Say done.")

        self.assertEqual(result.message, "done")
        system_prompt = client.calls[0]["messages"][0]["content"]
        self.assertNotIn("replace_symbols(path", system_prompt)
        self.assertNotIn("run_shell(command", system_prompt)
        self.assertLess(len(system_prompt), len(agent.messages[0]["content"]))

    def test_primary_prompt_uses_edit_tool_palette_for_mutation_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient(
                [
                    '{"type":"tool","name":"write_file","arguments":{"path":"src/app.py","content":"def f():\\n    return 1\\n"}}',
                    '{"type":"final","message":"edited"}',
                ]
            )
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Edit src/app.py.")

        self.assertEqual(result.message, "edited")
        system_prompt = client.calls[0]["messages"][0]["content"]
        self.assertIn("replace_symbols(path", system_prompt)
        self.assertIn("write_file(path", system_prompt)
        self.assertNotIn("git_commit(message", system_prompt)

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

    def test_agent_allows_explanatory_implementation_question_without_mutation(self) -> None:
        client = FakeClient(['{"type":"final","message":"explain plan"}'])
        tools = ToolExecutor(Path.cwd(), approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

        result = agent.handle_user("How should I implement this feature?")

        self.assertEqual(result.message, "explain plan")

    def test_agent_retries_git_diff_with_working_tree_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_git_repo_or_skip(root)
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

    def test_agent_normalizes_unquoted_exact_text_write_with_newline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient(
                [
                    '{"type":"tool","name":"write_file","arguments":{"path":"scratch/repl.txt","content":"repl ok"}}',
                    '{"type":"tool","name":"read_file","arguments":{"path":"scratch/repl.txt"}}',
                    '{"type":"final","message":"repl ok"}',
                ]
            )
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Create scratch/repl.txt with exactly the text repl ok followed by a newline.")
            final_content = (root / "scratch" / "repl.txt").read_text(encoding="utf-8")

        self.assertEqual(result.message, "repl ok")
        self.assertEqual(final_content, "repl ok\n")
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_calls[0]["arguments"], {"path": "scratch/repl.txt", "content": "repl ok\n"})

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

    def test_agent_synthesizes_exact_shell_artifact_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scratch").mkdir()
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
            exact_command = subprocess.list2cmdline(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('scratch/artifact.txt').write_text('ok\\n')",
                ]
            )

            result = agent.handle_user(
                f"Use run_shell to execute exactly: {exact_command}. Then tell me what artifact was written."
            )
            artifact_exists = (root / "scratch" / "artifact.txt").exists()

        self.assertEqual(result.message, "Artifact written: scratch/artifact.txt.")
        self.assertEqual(len(client.calls), 0)
        self.assertTrue(artifact_exists)

    def test_agent_synthesizes_file_from_search_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scratch").mkdir()
            (root / "scratch" / "repl.txt").write_text("repl ok\n", encoding="utf-8")
            client = FakeClient(['{"type":"tool","name":"search","arguments":{"query":"repl ok","path":"."}}'])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Use search to find repl ok and tell me which file contains it.")

        self.assertEqual(result.message, "scratch/repl.txt contains the match.")
        self.assertTrue(any(event["type"] == "assistant_synthesized" for event in agent.events))

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

    def test_agent_normalizes_test_tool_alias_to_configured_run_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient(['{"type":"tool","name":"test","arguments":{}}'])
            tools = ToolExecutor(root, approval_mode="auto", test_command='python -c "print(\'test_alias OK\')"')
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

            result = agent.handle_user("Run tests and tell me whether tests passed.")

        self.assertIn("Tests passed: yes", result.message)
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_calls[0]["name"], "run_test")
        self.assertEqual(tool_calls[0]["arguments"]["command"], 'python -c "print(\'test_alias OK\')"')

    def test_agent_normalizes_shell_test_to_configured_run_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient(['{"type":"tool","name":"run_shell","arguments":{"command":"python -m unittest example_test.py"}}'])
            tools = ToolExecutor(root, approval_mode="auto", test_command='python -c "print(\'test_polyglot OK\')"')
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

            result = agent.handle_user("Run tests and tell me whether tests passed.")

        self.assertIn("Tests passed: yes", result.message)
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_calls[0]["name"], "run_test")
        self.assertEqual(tool_calls[0]["arguments"]["command"], 'python -c "print(\'test_polyglot OK\')"')
        normalizations = [event for event in agent.events if event["type"] == "tool_normalized"]
        self.assertEqual(normalizations[0]["normalized_name"], "run_test")

    def test_agent_normalizes_unittest_file_path_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "tests" / "test_sample.py").write_text("import unittest\n\nclass SampleTests(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n", encoding="utf-8")
            client = FakeClient(['{"type":"tool","name":"run_test","arguments":{"command":"python -m unittest tests/test_sample.py"}}'])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

            result = agent.handle_user("Run tests and tell me whether tests passed.")

        self.assertIn("Tests passed: yes", result.message)
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_calls[0]["arguments"]["command"], "python -m unittest discover -s tests -p test_sample.py")

    def test_agent_preserves_explicit_shell_test_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = subprocess.list2cmdline([sys.executable, "-m", "unittest", "--help"])
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="auto", test_command='python -c "print(\'test_polyglot OK\')"')
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user(f"Use run_shell to execute exactly: {command}. Then tell me the exit code.")

        self.assertIn("Exit code: 0", result.message)
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_calls[0]["name"], "run_shell")
        self.assertEqual(tool_calls[0]["arguments"]["command"], command)

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
                '{"model":"saved-model","approval_mode":"read-only","reconcile_mode":"on","workspace_root":"'
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
        self.assertEqual(agent.reconcile_mode(), "on")
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
        write_calls = [event for event in agent.events if event.get("type") == "tool_call" and event.get("name") == "write_file"]
        self.assertTrue(write_calls)
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
        tool_feedback = next(
            message["content"]
            for message in agent.messages
            if message["role"] == "user" and "read_file" in message["content"] and "... truncated ..." in message["content"]
        )
        self.assertIn("... truncated ...", tool_feedback)
        self.assertNotIn(" 200 |", tool_feedback)

    def test_agent_compacts_large_write_arguments_in_assistant_history_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content = "\n".join(f"# line {index} TOKEN_FULL_EVENT" for index in range(80)) + "\n"
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "big.py", "content": content}}),
                    json.dumps({"type": "final", "message": "big.py updated"}),
                ]
            )
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
            result = agent.handle_user("Write big.py with generated content.")

        self.assertEqual(result.message, "big.py updated")
        assistant_tool = next(
            message["content"]
            for message in agent.messages
            if message["role"] == "assistant" and '"name":"write_file"' in message["content"]
        )
        self.assertIn("[omitted", assistant_tool)
        self.assertIn("do not copy", assistant_tool)
        self.assertNotIn("line 79 TOKEN_FULL_EVENT", assistant_tool)
        tool_call = next(event for event in agent.events if event["type"] == "tool_call" and event.get("name") == "write_file")
        self.assertIn("line 79 TOKEN_FULL_EVENT", tool_call["arguments"]["content"])

    def test_agent_compacts_run_test_output_to_actionable_failure(self) -> None:
        client = FakeClient([])
        tools = ToolExecutor(Path.cwd(), approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")
        noisy_status = "\n".join(f"test_noise_{index} (suite.Case.test_noise_{index}) ... FAIL" for index in range(80))
        output = (
            noisy_status
            + "\n"
            + "-" * 70
            + "\nFAIL: test_add_negative (tests.test_math.MathTests.test_add_negative)\n"
            + "-" * 70
            + "\nTraceback (most recent call last):\n"
            + '  File "tests/test_math.py", line 42, in test_add_negative\n'
            + "    self.assertEqual(add(-2, -3), -5)\n"
            + "AssertionError: -4 != -5\n\n"
            + "unhelpful tail " + ("x" * 3000) + "\n"
            + "Ran 81 tests in 0.123s\nFAILED (failures=81)\n"
        )

        payload = agent._compact_tool_result_for_context("run_test", {"ok": False, "output": output})

        compact = payload["output"]
        self.assertIn("FAILED (failures=81)", compact)
        self.assertIn("FAIL: test_add_negative", compact)
        self.assertIn("AssertionError: -4 != -5", compact)
        self.assertNotIn("test_noise_0", compact)
        self.assertNotIn("unhelpful tail", compact)
        self.assertLessEqual(len(compact), 700)

    def test_agent_run_test_feedback_points_at_syntax_file_not_import_guess(self) -> None:
        client = FakeClient([])
        tools = ToolExecutor(Path.cwd(), approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")
        output = (
            "ERROR: sample_test (unittest.loader._FailedTest.sample_test)\n"
            "Traceback (most recent call last):\n"
            "  File \"C:\\workspace\\sample.py\", line 2\n"
            "    def f():\n"
            "IndentationError: unexpected indent\n"
        )

        feedback = agent._tool_result_feedback_message("run_test", {"ok": False, "output": output}, real_tool_use=True)

        self.assertIn("IndentationError: unexpected indent at sample.py:2", feedback)
        self.assertIn("Do not blame imports unless error is ModuleNotFoundError", feedback)

    def test_agent_run_test_feedback_includes_failing_source_excerpt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "list_ops.py").write_text("def foldr(function, items, initial):\n    return initial\n", encoding="utf-8")
            test_file = root / "list_ops_test.py"
            test_file.write_text(
                "from list_ops import foldr\n\n"
                "import unittest\n\n"
                "class ListOpsTest(unittest.TestCase):\n"
                "    def test_foldr_add_string(self):\n"
                "        self.assertEqual(\n"
                "            foldr(lambda acc, el: el + acc, ['e', 'x'], '!'),\n"
                "            'ex!'\n"
                "        )\n",
                encoding="utf-8",
            )
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")
            output = (
                "FAIL: test_foldr_add_string (list_ops_test.ListOpsTest.test_foldr_add_string)\n"
                "Traceback (most recent call last):\n"
                f'  File "{test_file}", line 7, in test_foldr_add_string\n'
                "    self.assertEqual(\n"
                "AssertionError: '!xe' != 'ex!'\n"
            )

            feedback = agent._tool_result_feedback_message("run_test", {"ok": False, "output": output}, real_tool_use=True)

            self.assertIn("AssertionError: '!xe' != 'ex!'", feedback)
            self.assertIn("Diagnosis:", feedback)
            self.assertIn("actual='!xe' expected='ex!'", feedback)
            self.assertIn("list_ops.py", feedback)
            self.assertIn("Failing source excerpt", feedback)
            self.assertIn("list_ops_test.py:7", feedback)
            self.assertIn("foldr(lambda acc, el: el + acc", feedback)

    def test_agent_failed_omitted_context_write_points_to_partial_edit_tools(self) -> None:
        client = FakeClient([])
        tools = ToolExecutor(Path.cwd(), approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

        feedback = agent._tool_result_feedback_message(
            "write_file",
            {
                "ok": False,
                "summary": "Refusing to write omitted-context marker as file content. Reconstruct complete file content instead.",
            },
            real_tool_use=False,
        )

        self.assertIn("Use replace_symbol/replace_in_file", feedback)
        self.assertIn("content was abbreviated", feedback)

    def test_agent_blocks_repeated_failed_run_test_until_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fail_command = subprocess.list2cmdline([sys.executable, "-c", "import sys; sys.exit(1)"])
            pass_command = subprocess.list2cmdline([sys.executable, "-c", "print('ok')"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": fail_command}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": fail_command}}),
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": "def f():\n    return 1\n"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": pass_command}}),
                    json.dumps({"type": "final", "message": "changed app.py"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Fix app.py, run tests, and rerun tests after editing.")

        self.assertEqual(result.message, "changed app.py")
        self.assertEqual(tools.execute_counts.get("run_test"), 2)
        self.assertEqual(tools.execute_counts.get("write_file"), 1)

    def test_agent_blocks_false_test_success_after_failed_run_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fail_command = subprocess.list2cmdline([sys.executable, "-c", "import sys; print('FAILED'); sys.exit(1)"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": fail_command}}),
                    json.dumps({"type": "final", "message": "All tests passed successfully."}),
                    json.dumps({"type": "final", "message": "Tests failed with exit code 1."}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Run tests and summarize the result.")

        self.assertEqual(result.message, "Tests failed with exit code 1.")
        self.assertEqual(tools.execute_counts.get("run_test"), 1)
        self.assertTrue(any("do not claim tests passed" in message["content"] for message in agent.messages if message["role"] == "user"))

    def test_agent_requires_edit_after_failed_tests_for_fix_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fail_command = subprocess.list2cmdline([sys.executable, "-c", "import sys; print('AssertionError: None != 1'); sys.exit(1)"])
            pass_command = subprocess.list2cmdline([sys.executable, "-c", "print('OK')"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": fail_command}}),
                    json.dumps({"type": "final", "message": "The test failure shows the bug."}),
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": "def f():\n    return 1\n"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": pass_command}}),
                    json.dumps({"type": "final", "message": "Fixed app.py and tests passed."}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=12)

            result = agent.handle_user("Fix app.py, run tests, and summarize.")

        self.assertEqual(result.message, "Fixed app.py and tests passed.")
        self.assertEqual(tools.execute_counts.get("write_file"), 1)
        self.assertEqual(tools.execute_counts.get("run_test"), 2)
        self.assertTrue(any("no implementation edit succeeded" in message["content"] for message in agent.messages if message["role"] == "user"))

    def test_agent_reconciles_failed_test_artifact_and_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fail_command = subprocess.list2cmdline([sys.executable, "-c", "import sys; print('AssertionError: 0 != 1'); sys.exit(1)"])
            pass_command = subprocess.list2cmdline([sys.executable, "-c", "print('OK')"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": fail_command}}),
                    json.dumps(
                        {
                            "verdict": "retry",
                            "reason": "The failing test needs implementation repair before final.",
                            "repair_plan": ["edit implementation", "rerun tests"],
                            "required_tools": ["write_file"],
                            "forbidden_tools": [],
                        }
                    ),
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": "def f():\n    return 1\n"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": pass_command}}),
                    json.dumps({"type": "final", "message": "Fixed app.py and tests passed."}),
                ],
                script_reconciliation=True,
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, reconcile_mode="auto")

            result = agent.handle_user("Fix app.py, run tests, and summarize.")

        self.assertEqual(result.message, "Fixed app.py and tests passed.")
        reconciliations = [event for event in agent.events if event["type"] == "reconciliation"]
        self.assertEqual([event["verdict"] for event in reconciliations], ["retry"])
        self.assertTrue(any("Artifact reconciliation rejected" in message["content"] for message in agent.messages if message["role"] == "user"))
        self.assertEqual([call["think"] for call in client.calls if str(call["messages"][0]["content"]).startswith("You are an artifact reconciliation critic")], [False])

    def test_agent_reconcile_off_skips_failed_test_artifact_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fail_command = subprocess.list2cmdline([sys.executable, "-c", "import sys; print('FAILED'); sys.exit(1)"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": fail_command}}),
                    json.dumps({"type": "final", "message": "Tests failed with exit code 1."}),
                ],
                script_reconciliation=True,
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, reconcile_mode="off")

            result = agent.handle_user("Run tests and summarize the result.")

        self.assertEqual(result.message, "Tests failed with exit code 1.")
        self.assertFalse(any(event["type"] == "reconciliation" for event in agent.events))

    def test_agent_reconcile_auto_skips_failed_edit_artifact(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient([])
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", reconcile_mode="auto")

        needs_reconciliation = agent._tool_result_needs_reconciliation(
            request_text="Fix app.py and run tests.",
            name="replace_symbol",
            result={"ok": False, "summary": "Symbol not found: f"},
            cache_hit=False,
            session_memory_request=False,
            mutation_required=True,
            test_run_required=True,
        )

        self.assertFalse(needs_reconciliation)

    def test_agent_runs_final_chance_test_after_last_round_edit(self) -> None:
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

            result = agent.handle_user("Edit app.py and run tests.")

        self.assertTrue(result.completed)
        self.assertEqual(result.message, "Ran tests after the latest edit: passed.")
        self.assertEqual(tools.execute_counts.get("write_file"), 1)
        self.assertEqual(tools.execute_counts.get("run_test"), 1)
        auto_results = [event for event in agent.events if event["type"] == "tool_result" and event["name"] == "run_test"]
        self.assertTrue(auto_results[0]["auto"])

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

    def test_agent_requires_edits_to_explicitly_named_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "docs").mkdir()
            pass_command = subprocess.list2cmdline([sys.executable, "-c", "print('OK')"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "src/app.py", "content": "def f():\n    return 1\n"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": pass_command}}),
                    json.dumps({"type": "final", "message": "Tests passed."}),
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "docs/app.md", "content": "Updated docs.\n"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": pass_command}}),
                    json.dumps({"type": "final", "message": "Updated src and docs."}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=10)

            result = agent.handle_user("Update src/app.py and docs/app.md, then run tests.")

        self.assertEqual(result.message, "Updated src and docs.")
        self.assertIn("docs/app.md", " ".join(message["content"] for message in agent.messages if message["role"] == "user"))
        self.assertEqual(tools.execute_counts.get("write_file"), 2)
        self.assertEqual(tools.execute_counts.get("run_test"), 2)

    def test_agent_fails_closed_after_reconciliation_retry_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fail_commands = [
                subprocess.list2cmdline([sys.executable, "-c", f"import sys; print('FAIL {index}'); sys.exit(1)"])
                for index in range(3)
            ]
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": fail_commands[0]}}),
                    json.dumps({"verdict": "retry", "reason": "repair first", "repair_plan": ["edit"], "required_tools": [], "forbidden_tools": []}),
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": "def f():\n    return 1\n"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": fail_commands[1]}}),
                    json.dumps({"verdict": "retry", "reason": "still failing", "repair_plan": ["edit again"], "required_tools": [], "forbidden_tools": []}),
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": "def f():\n    return 2\n"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": fail_commands[2]}}),
                    json.dumps({"verdict": "retry", "reason": "still no approved path", "repair_plan": ["stop"], "required_tools": [], "forbidden_tools": []}),
                ],
                script_reconciliation=True,
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, reconcile_mode="on", max_tool_rounds=10)

            result = agent.handle_user("Fix app.py, run tests, and keep repairing until tests pass.")

        self.assertFalse(result.completed)
        self.assertIn("artifact reconciliation could not approve", result.message)
        reconciliations = [event for event in agent.events if event["type"] == "reconciliation"]
        self.assertEqual(len(reconciliations), 3)

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
            (root / "docs.md").write_text("total total\n", encoding="utf-8")
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
        self.assertEqual(final_text, "cart_total cart_total\n")
        normalizations = [event for event in agent.events if event["type"] == "tool_normalized"]
        self.assertIn("replace_all", json.dumps(normalizations[0]["normalized_arguments"]))

    def test_agent_normalizes_replace_symbol_text_edit_on_non_code_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs.md").write_text("Use total(prices).\n", encoding="utf-8")
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "replace_symbol", "arguments": {"path": "docs.md", "symbol": "total", "content": "cart_total"}}),
                    json.dumps({"type": "final", "message": "docs updated"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Update docs.md replacing total with cart_total.")
            final_text = (root / "docs.md").read_text(encoding="utf-8")

        self.assertEqual(result.message, "docs updated")
        self.assertEqual(final_text, "Use cart_total(prices).\n")
        self.assertEqual(tools.execute_counts.get("replace_in_file"), 1)

    def test_agent_normalizes_edit_file_alias_with_symbol_content(self) -> None:
        root = self._workspace_scratch()
        (root / "app.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "edit_file", "arguments": {"path": "app.py", "symbol": "f", "content": "def f():\n    return 2\n"}}),
                json.dumps({"type": "final", "message": "app.py updated"}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

        result = agent.handle_user("Edit function f in app.py to return 2.")
        final_text = (root / "app.py").read_text(encoding="utf-8")

        self.assertEqual(result.message, "app.py updated")
        self.assertIn("return 2", final_text)
        self.assertEqual(tools.execute_counts.get("replace_symbol"), 1)
        normalizations = [event for event in agent.events if event["type"] == "tool_normalized"]
        self.assertEqual(normalizations[0]["normalized_name"], "replace_symbol")

    def test_agent_normalizes_edit_file_alias_with_symbol_replacements(self) -> None:
        root = self._workspace_scratch()
        (root / "app.py").write_text("def f():\n    return 1\n\n\ndef g():\n    return 2\n", encoding="utf-8")
        client = FakeClient(
            [
                json.dumps(
                    {
                        "type": "tool",
                        "name": "edit_file",
                        "arguments": {
                            "path": "app.py",
                            "replacements": [
                                {"symbol": "f", "content": "def f():\n    return 10\n"},
                                {"symbol": "g", "content": "def g():\n    return 20\n"},
                            ],
                        },
                    }
                ),
                json.dumps({"type": "final", "message": "app.py updated"}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

        result = agent.handle_user("Edit functions f and g in app.py.")
        final_text = (root / "app.py").read_text(encoding="utf-8")

        self.assertEqual(result.message, "app.py updated")
        self.assertIn("return 10", final_text)
        self.assertIn("return 20", final_text)
        self.assertEqual(tools.execute_counts.get("replace_symbols"), 1)
        normalizations = [event for event in agent.events if event["type"] == "tool_normalized"]
        self.assertEqual(normalizations[0]["normalized_name"], "replace_symbols")

    def test_agent_deterministically_handles_strip_lower_source_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "formatter.py").write_text("def normalize_email(value: str) -> str:\n    return value.strip()\n", encoding="utf-8")
            command = subprocess.list2cmdline([sys.executable, "-c", "print('OK')"])
            client = FakeClient([])
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Please make normalize_email in src/formatter.py return stripped lowercase email text. Inspect first, edit, run tests.")
            final_text = (root / "src" / "formatter.py").read_text(encoding="utf-8")

        self.assertIn("tests passed", result.message)
        self.assertIn("return value.strip().lower()", final_text)
        self.assertEqual(tools.execute_counts.get("replace_in_file"), 1)
        self.assertEqual(tools.execute_counts.get("run_test"), 1)
        self.assertEqual(len(client.calls), 0)

    def test_agent_deterministically_handles_return_literal_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "session_task.py").write_text("def session_value() -> str:\n    return 'todo'\n", encoding="utf-8")
            command = subprocess.list2cmdline([sys.executable, "-c", "print('OK')"])
            client = FakeClient([])
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Change session_value in src/session_task.py to return 'SESSION_OK' instead of 'todo'. Then run tests.")
            final_text = (root / "src" / "session_task.py").read_text(encoding="utf-8")

        self.assertIn("tests passed", result.message)
        self.assertIn("return 'SESSION_OK'", final_text)
        self.assertEqual(tools.execute_counts.get("replace_in_file"), 1)
        self.assertEqual(tools.execute_counts.get("run_test"), 1)
        self.assertEqual(len(client.calls), 0)

    def test_agent_deterministically_handles_slugify_spaces_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "slug.py").write_text("def slugify(value: str) -> str:\n    return value.strip().lower()\n", encoding="utf-8")
            command = subprocess.list2cmdline(
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.path.insert(0, 'src'); from slug import slugify; assert slugify('  Many   Spaces ') == 'many-spaces'; print('OK')",
                ]
            )
            client = FakeClient([])
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Run the tests, inspect the failure, fix slugify in src/slug.py so spaces collapse to single hyphens, rerun tests, and summarize.")
            final_text = (root / "src" / "slug.py").read_text(encoding="utf-8")

        self.assertIn("tests passed", result.message)
        self.assertIn("re.sub", final_text)
        self.assertEqual(tools.execute_counts.get("write_file"), 1)
        self.assertEqual(tools.execute_counts.get("run_test"), 1)
        self.assertEqual(len(client.calls), 0)

    def test_agent_rejects_unrequested_git_commit_during_edit_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "git_commit", "arguments": {"message": "unexpected"}}),
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": "def f():\n    return 1\n"}}),
                    json.dumps({"type": "final", "message": "app.py updated"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Edit app.py to return 1.")

        self.assertEqual(result.message, "app.py updated")
        self.assertIsNone(tools.execute_counts.get("git_commit"))
        self.assertEqual(tools.execute_counts.get("write_file"), 1)
        self.assertTrue(any("Do not create commits" in message["content"] for message in agent.messages if message["role"] == "user"))

    def test_agent_rejects_test_file_edit_when_request_says_implementation_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "sample_test.py", "content": "bad\n"}}),
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "sample.py", "content": "def f():\n    return 1\n"}}),
                    json.dumps({"type": "final", "message": "sample.py updated"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Edit only implementation files to fix sample.py.")

        self.assertEqual(result.message, "sample.py updated")
        self.assertFalse((root / "sample_test.py").exists())
        self.assertEqual(tools.execute_counts.get("write_file"), 1)
        self.assertTrue(any("Do not edit test files" in message["content"] for message in agent.messages if message["role"] == "user"))

    def test_agent_rejects_test_rewrite_that_drops_import_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "tests" / "test_app.py").write_text(
                "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\nfrom app import f\n",
                encoding="utf-8",
            )
            pass_command = subprocess.list2cmdline([sys.executable, "-c", "print('OK')"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "tests/test_app.py", "content": "from app import f\n"}}),
                    json.dumps({"type": "tool", "name": "replace_in_file", "arguments": {"path": "tests/test_app.py", "old": "from app import f", "new": "from app import f"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": pass_command}}),
                    json.dumps({"type": "final", "message": "tests preserved"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

            result = agent.handle_user("Update tests/test_app.py as needed and run tests.")

        self.assertEqual(result.message, "tests preserved")
        self.assertTrue(any("sys.path bootstrap" in message["content"] for message in agent.messages if message["role"] == "user"))
        self.assertIsNone(tools.execute_counts.get("write_file"))
        self.assertEqual(tools.execute_counts.get("replace_in_file"), 1)

    def test_request_requires_mutation_when_only_tests_are_read_only(self) -> None:
        client = FakeClient([])
        tools = ToolExecutor(Path.cwd(), approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")
        prompt = (
            "Implement this Python Exercism exercise. Read tests and source, edit only implementation files, "
            "do not edit tests, replace stubs with complete code, run tests with configured test command."
        )

        self.assertTrue(agent._request_requires_mutation(prompt))

    def test_agent_rejects_new_unimported_python_file_for_test_driven_fix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "list_ops.py").write_text("def reverse(items):\n    return None\n", encoding="utf-8")
            (root / "list_ops_test.py").write_text(
                "from list_ops import reverse\n\n"
                "def test_reverse():\n"
                "    assert reverse([1, 2]) == [2, 1]\n",
                encoding="utf-8",
            )
            pass_command = subprocess.list2cmdline([sys.executable, "-c", "print('OK')"])
            client = FakeClient(
                [
                    json.dumps(
                        {
                            "type": "tool",
                            "name": "write_file",
                            "arguments": {"path": "palindrome_solution.py", "content": "def is_palindrome(s):\n    return True\n"},
                        }
                    ),
                    json.dumps(
                        {
                            "type": "tool",
                            "name": "write_file",
                            "arguments": {"path": "list_ops.py", "content": "def reverse(items):\n    return items[::-1]\n"},
                        }
                    ),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": pass_command}}),
                    json.dumps({"type": "final", "message": "list_ops.py fixed; tests passed."}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=pass_command)
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "baseline"}):
                result = agent.handle_user(
                    "Implement this Python Exercism exercise. Read tests and source, edit only implementation files, "
                    "do not edit tests, replace stubs with complete code, run tests with configured test command."
                )

            self.assertEqual(result.message, "list_ops.py fixed; tests passed.")
            self.assertFalse((root / "palindrome_solution.py").exists())
            self.assertEqual((root / "list_ops.py").read_text(encoding="utf-8"), "def reverse(items):\n    return items[::-1]\n")
            self.assertTrue(any("Existing tests import implementation file(s): list_ops.py" in message["content"] for message in agent.messages if message["role"] == "user"))

    def test_agent_rejects_test_edit_when_fix_names_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "tests/test_slug.py", "content": "bad\n"}}),
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "src/slug.py", "content": "def slugify(value):\n    return value\n"}}),
                    json.dumps({"type": "final", "message": "src/slug.py updated"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Fix slugify in src/slug.py.")

        self.assertEqual(result.message, "src/slug.py updated")
        self.assertFalse((root / "tests" / "test_slug.py").exists())
        self.assertEqual(tools.execute_counts.get("write_file"), 1)
        self.assertTrue(any("Do not edit test files" in message["content"] for message in agent.messages if message["role"] == "user"))

    def test_agent_blocks_final_and_run_test_while_python_syntax_error_known(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = subprocess.list2cmdline([sys.executable, "-c", "print('ok')"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": "def f():\nreturn 1\n"}}),
                    json.dumps({"type": "final", "message": "app.py updated"}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": command}}),
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": "def f():\n    return 1\n"}}),
                    json.dumps({"type": "final", "message": "app.py updated"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Implement this by editing app.py.")
            final_text = (root / "app.py").read_text(encoding="utf-8")

        self.assertEqual(result.message, "app.py updated")
        self.assertEqual(final_text, "def f():\n    return 1\n")
        self.assertEqual(tools.execute_counts.get("write_file"), 2)
        self.assertIsNone(tools.execute_counts.get("run_test"))

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
            self._init_git_repo_or_skip(root)
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
        self.assertEqual(len(client.calls), 0)
        self.assertFalse(any(event["type"] == "assumption_audit" for event in agent.events))
        tool_names = [event["name"] for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_names, ["search_symbols", "read_symbol"])
        symbol_results = [event for event in agent.events if event["type"] == "tool_result" and event["name"] == "read_symbol"]
        self.assertIn("TOKEN_SYMBOL_750", symbol_results[0]["result"]["output"])
        self.assertNotIn("filler_0", symbol_results[0]["result"]["output"])
        self.assertTrue(any(event["type"] == "assistant_synthesized" for event in agent.events))

    def test_agent_synthesizes_symbol_return_value_without_model_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("def meaning() -> int:\n    return 42\n", encoding="utf-8")
            client = FakeClient(['{"type":"final","message":"wrong"}'])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

            result = agent.handle_user(
                "Use search_symbols to locate meaning in src/app.py, then use read_symbol on the exact match. Do not use read_file. Summarize what value it returns."
            )

        self.assertEqual(result.message, "meaning returns 42.")
        self.assertEqual(len(client.calls), 0)
        tool_names = [event["name"] for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_names, ["search_symbols", "read_symbol"])
        self.assertTrue(any(event["type"] == "assistant_synthesized" for event in agent.events))
