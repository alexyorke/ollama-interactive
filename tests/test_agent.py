from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ollama_code.agent import (
    AgentResult,
    GROUNDING_EVIDENCE_TOOL_NAMES,
    OllamaCodeAgent,
    extract_json_response,
)
from ollama_code.features import ENV_OLLAMA_CODE_FEATURE_PROFILE
from ollama_code.ollama_client import OllamaError, TokenUsage
from ollama_code.tools import ToolExecutor
from tests.agent_test_support import (
    AgentTestBase,
    CountingToolExecutor,
    EmptySelectTestsLintFallbackToolExecutor,
    EmptySelectTestsToolExecutor,
    FakeClient,
    RawImplementationSpecCountingToolExecutor,
    WorkflowValidatorToolExecutor,
)


class AgentTests(AgentTestBase):
    def _primary_tools_for_request(
        self,
        request: str,
        *,
        requires_tools: bool = True,
        mutation_allowed: bool = False,
        mutation_required: bool = False,
        test_run_required: bool = False,
    ) -> set[str]:
        root = self._workspace_scratch()
        _client, _tools, agent = self._build_agent(root)
        return agent._primary_tool_names_for_request(
            request,
            requires_tools=requires_tools,
            session_memory_request=False,
            mutation_allowed=mutation_allowed,
            mutation_required=mutation_required,
            test_run_required=test_run_required,
            required_tool_names=set(),
            forbidden_tool_names=set(),
        )

    def _build_agent(
        self,
        root: Path,
        client: FakeClient | None = None,
        *,
        approval_mode: str = "auto",
        tool_cls: type[ToolExecutor] = ToolExecutor,
        tool_kwargs: dict[str, object] | None = None,
        **kwargs: object,
    ) -> tuple[FakeClient, ToolExecutor, OllamaCodeAgent]:
        resolved_client = client if client is not None else FakeClient([])
        tools = tool_cls(root, approval_mode=approval_mode, **(tool_kwargs or {}))
        return resolved_client, tools, OllamaCodeAgent(client=resolved_client, tools=tools, model="fake-model", **kwargs)

    def _cwd_agent(
        self,
        client: FakeClient | None = None,
        *,
        approval_mode: str = "auto",
        **kwargs: object,
    ) -> OllamaCodeAgent:
        _client, _tools, agent = self._build_agent(Path.cwd(), client, approval_mode=approval_mode, **kwargs)
        return agent

    def test_system_prompt_advertises_add_function_edit_intent(self) -> None:
        root = self._workspace_scratch()
        _client, _tools, agent = self._build_agent(root, debate_enabled=False)

        prompt = agent._system_prompt_for_tools({"edit_intent"})

        self.assertIn("add_import|add_function", prompt)
        self.assertIn("edit_intent(path,intent=", prompt)

    def _workspace_agent(
        self,
        client: FakeClient | None = None,
        *,
        approval_mode: str = "auto",
        tool_cls: type[ToolExecutor] = ToolExecutor,
        tool_kwargs: dict[str, object] | None = None,
        **kwargs: object,
    ) -> tuple[Path, FakeClient, ToolExecutor, OllamaCodeAgent]:
        root = self._workspace_scratch()
        return root, *self._build_agent(
            root, client, approval_mode=approval_mode, tool_cls=tool_cls, tool_kwargs=tool_kwargs, **kwargs
        )

    @contextmanager
    def _temp_agent(
        self,
        client: FakeClient | None = None,
        *,
        approval_mode: str = "auto",
        tool_cls: type[ToolExecutor] = ToolExecutor,
        tool_kwargs: dict[str, object] | None = None,
        **kwargs: object,
    ) -> Iterator[tuple[Path, FakeClient, ToolExecutor, OllamaCodeAgent]]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            yield root, *self._build_agent(
                root, client, approval_mode=approval_mode, tool_cls=tool_cls, tool_kwargs=tool_kwargs, **kwargs
            )

    @contextmanager
    def _temp_python_agent(
        self,
        files: dict[str, str],
        client: FakeClient | None = None,
        *,
        tool_cls: type[ToolExecutor] = ToolExecutor,
        test_command: str | None = None,
        test_discover_args: tuple[str, ...] = ("-p", "*_test.py", "-v"),
        tool_kwargs: dict[str, object] | None = None,
        **kwargs: object,
    ) -> Iterator[tuple[Path, FakeClient, ToolExecutor, OllamaCodeAgent]]:
        resolved_tool_kwargs = dict(tool_kwargs or {})
        resolved_tool_kwargs.setdefault(
            "test_command",
            test_command or subprocess.list2cmdline([sys.executable, "-m", "unittest", "discover", *test_discover_args]),
        )
        with self._temp_agent(client, tool_cls=tool_cls, tool_kwargs=resolved_tool_kwargs, **kwargs) as (root, resolved_client, tools, agent):
            for relative_path, content in files.items():
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            yield root, resolved_client, tools, agent

    def _assert_repo_tool_then_git_status_without_model_loop(
        self,
        request: str,
        *,
        first_tool_name: str,
        expected_message_fragment: str,
    ) -> None:
        with self._temp_agent(debate_enabled=False) as (root, client, _tools, agent):
            self._init_git_repo_or_skip(root)
            (root / "docs").mkdir()
            (root / "src").mkdir()
            (root / "docs" / "guide.md").write_text("TOKEN_42 lives here.\n", encoding="utf-8")
            (root / "src" / "app.py").write_text("def answer() -> int:\n    return 42\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=root, capture_output=True, text=True, check=True)
            subprocess.run(["git", "commit", "-m", "initial"], cwd=root, capture_output=True, text=True, check=True)
            (root / "src" / "app.py").write_text("def answer() -> int:\n    return 99\n", encoding="utf-8")
            result = agent.handle_user(request)

        self.assertIn(expected_message_fragment, result.message)
        self.assertIn("src/app.py", result.message)
        self.assertEqual(len(client.calls), 0)
        tool_names = [event["name"] for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_names, [first_tool_name, "git_status"])

    def _assert_follow_up_tool_chain_without_model_loop(
        self,
        request: str,
        *,
        expected_tool_names: list[str],
        expected_message_fragments: list[str],
        acceptable_follow_up_fragments: list[str] | None = None,
        create_docs_fixture: bool = False,
        test_file_content: str | None = None,
    ) -> None:
        with self._temp_agent(debate_enabled=False) as (root, client, _tools, agent):
            if create_docs_fixture:
                (root / "docs").mkdir()
                (root / "docs" / "guide.md").write_text("TOKEN_42 lives here.\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "test_sample.py").write_text(
                test_file_content
                or "import unittest\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n",
                encoding="utf-8",
            )
            result = agent.handle_user(request)

        for fragment in expected_message_fragments:
            self.assertIn(fragment, result.message)
        if acceptable_follow_up_fragments is not None:
            self.assertTrue(any(fragment in result.message for fragment in acceptable_follow_up_fragments))
        self.assertEqual(len(client.calls), 0)
        tool_names = [event["name"] for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_names, expected_tool_names)

    def test_agent_runs_tool_loop(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                '{"type":"final","message":"The file says hello world."}',
            ]
        )
        with self._temp_agent(client) as (root, _client, _tools, agent):
            (root / "note.txt").write_text("hello world\n", encoding="utf-8")
            result = agent.handle_user("Summarize note.txt")

        self.assertEqual(result.message, "The file says hello world.")
        self.assertEqual(result.rounds, 2)
        self.assertTrue(any(event.get("type") == "tool_call" and event.get("name") == "read_file" for event in agent.events))
        tool_result = next(event for event in agent.events if event.get("type") == "tool_result" and event.get("name") == "read_file")
        self.assertIsInstance(tool_result.get("duration_ms"), float)

    def test_agent_normalizes_tool_named_final_into_final_answer(self) -> None:
        client = FakeClient(['{"type":"tool","name":"final","arguments":{"message":"done"}}'])
        with self._temp_agent(client, debate_enabled=False, disable_spec_guided_repair=True) as (_root, _client, _tools, agent):
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
        with self._temp_agent(client, debate_enabled=False) as (_root, _client, _tools, agent):
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

    def test_agent_uses_schema_and_num_predict_feature_profile(self) -> None:
        client = FakeClient(['{"type":"final","message":"ok"}'])
        _root, _client, _tools, agent = self._workspace_agent(
            client, debate_enabled=False, tool_kwargs={"test_command": "python -m unittest discover -s tests -v"}
        )

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "schema,num-predict-caps"}):
            result = agent.handle_user("say ok")

        self.assertEqual(result.message, "ok")
        call = client.calls[0]
        self.assertIsInstance(call["response_format"], dict)
        self.assertEqual(call["options"], {"num_predict": 256})

    def test_primary_think_defaults_off_for_broad_coding_prompt(self) -> None:
        _root, _client, _tools, agent = self._workspace_agent(
            debate_enabled=False, tool_kwargs={"test_command": "python -m unittest discover -s tests -v"}
        )

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
        _root, _client, _tools, agent = self._workspace_agent(debate_enabled=False)

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
        root, _client, _tools, agent = self._workspace_agent(
            debate_enabled=False, tool_kwargs={"test_command": "python -m unittest discover -s tests -v"}
        )
        (root / "src").mkdir(exist_ok=True)
        (root / "docs").mkdir(exist_ok=True)
        (root / "src" / "client.py").write_text("def fetch_data(url: str) -> dict:\n    return {}\n", encoding="utf-8")
        (root / "docs" / "client.md").write_text("Call `fetch_data(url)` to fetch data.\n", encoding="utf-8")

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
            should_skip_project_rename = agent._should_preload_context_pack(
                request_text=(
                    "Refactor the pricing API from total(prices) to cart_total(prices). "
                    "Update src/pricing.py, tests, and docs/pricing.md. Run tests."
                ),
                session_memory_request=False,
                mutation_required=True,
                test_run_required=True,
                required_tool_names=set(),
                forbidden_tool_names=set(),
            )
            should_skip_optional_parameter_docs = agent._should_preload_context_pack(
                request_text=(
                    "Add an optional timeout: int = None parameter to fetch_data in src/client.py, update docs/client.md, "
                    "and run tests."
                ),
                session_memory_request=False,
                mutation_required=True,
                test_run_required=True,
                required_tool_names=set(),
                forbidden_tool_names=set(),
            )
            should_skip_import_repair = agent._should_preload_context_pack(
                request_text="Fix the import bug in src/app.py because it uses from .helpers import slugify when run as a script, then run tests.",
                session_memory_request=False,
                mutation_required=True,
                test_run_required=True,
                required_tool_names=set(),
                forbidden_tool_names=set(),
            )

        self.assertTrue(should_preload)
        self.assertFalse(should_skip)
        self.assertFalse(should_skip_project_rename)
        self.assertFalse(should_skip_optional_parameter_docs)
        self.assertFalse(should_skip_import_repair)

    def test_clarification_planner_asks_after_local_evidence(self) -> None:
        client = FakeClient(
            [
                json.dumps(
                    {
                        "verdict": "ask",
                        "reason": "The request names a checkout workflow, but evidence does not identify which user path defines success.",
                        "ambiguities": [
                            {
                                "kind": "acceptance",
                                "detail": "Guest checkout and subscription renewal could require different edits.",
                                "evidence": "context_pack found checkout.py only.",
                            }
                        ],
                        "questions": [
                            {
                                "question": "Which checkout path should define the fix: guest checkout or subscription renewal?",
                                "why_it_matters": "The target behavior and tests differ by path.",
                                "recommended_default": "Use guest checkout because it is the broadest first-use path.",
                                "choices": ["guest checkout", "subscription renewal"],
                            }
                        ],
                    }
                )
            ],
            script_question_planner=True,
        )
        root, _client, _tools, agent = self._workspace_agent(client, debate_enabled=False)
        (root / "src").mkdir()
        (root / "src" / "checkout.py").write_text(
            "def checkout_total(items, discounts):\n    return sum(items)\n",
            encoding="utf-8",
        )

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "all"}):
            result = agent.handle_user("Improve the checkout workflow and fix the most important issue.")

        self.assertTrue(result.completed)
        self.assertIn("Which checkout path", result.message)
        self.assertIn("Recommended default: Use guest checkout", result.message)
        self.assertNotIn("should I proceed", result.message.lower())
        event_types = [event.get("type") for event in agent.events]
        self.assertLess(event_types.index("tool_call"), event_types.index("clarification_plan"))
        tool_names = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_names[:2], ["context_pack", "systems_lens"])
        self.assertFalse(any(name in {"edit_intent", "replace_in_file", "write_file"} for name in tool_names))

    def test_clarification_planner_proceeds_when_no_high_value_question(self) -> None:
        client = FakeClient(
            [
                '{"verdict":"proceed","reason":"Local evidence gives enough scope for a conservative throughput inspection.","ambiguities":[],"questions":[]}',
                '{"type":"final","message":"Proceeding from local evidence."}',
            ],
            script_question_planner=True,
        )
        root, _client, _tools, agent = self._workspace_agent(client, debate_enabled=False)
        (root / "app.py").write_text("def value():\n    return 1\n", encoding="utf-8")

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "all"}):
            result = agent.handle_user("Improve app throughput.")

        self.assertEqual(result.message, "Proceeding from local evidence.")
        self.assertTrue(any(event.get("type") == "clarification_plan" and event.get("verdict") == "proceed" for event in agent.events))
        system_prompts = [str(call["messages"][0]["content"]) for call in client.calls]  # type: ignore[index]
        self.assertTrue(system_prompts[0].startswith("You are a clarification planner"))
        self.assertTrue(any(prompt.startswith("Ollama Code.") for prompt in system_prompts))

    def test_clarification_planner_falls_back_when_model_notes_vagueness_but_proceeds(self) -> None:
        client = FakeClient(
            [
                json.dumps(
                    {
                        "verdict": "proceed",
                        "reason": "The request is too vague and requires defining the workflow scope and acceptance signal.",
                        "ambiguities": [],
                        "questions": [],
                    }
                )
            ],
            script_question_planner=True,
        )
        root, _client, _tools, agent = self._workspace_agent(client, debate_enabled=False)
        (root / "src").mkdir()
        (root / "src" / "checkout.py").write_text("def checkout():\n    return 'ok'\n", encoding="utf-8")

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "all"}):
            result = agent.handle_user("Improve the checkout workflow and fix the most important issue.")

        self.assertTrue(result.completed)
        self.assertIn("Which workflow should define success", result.message)
        self.assertIn("Recommended default:", result.message)
        self.assertTrue(any(event.get("type") == "clarification_plan" and event.get("verdict") == "ask" for event in agent.events))

    def test_clarification_planner_uses_schema_and_eba_fallback_for_broad_architecture_request(self) -> None:
        client = FakeClient(
            [
                json.dumps(
                    {
                        "verdict": "ask",
                        "reason": "The request is broad and asks for a rewrite.",
                        "ambiguities": [],
                        "questions": [{"question": "Should I proceed?", "choices": ["yes", "no"]}],
                    }
                )
            ],
            script_question_planner=True,
        )
        root, _client, _tools, agent = self._workspace_agent(client, debate_enabled=False)
        (root / "app.py").write_text("def value():\n    return 1\n", encoding="utf-8")

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "all"}):
            result = agent.handle_user("Refactor the architecture heavily, but keep the important external surfaces stable.")

        self.assertTrue(result.completed)
        self.assertIn("Which boundary should stay fixed in this pass", result.message)
        self.assertIn("Choices (pick one):", result.message)
        self.assertIn("Recommended default:", result.message)
        self.assertIsInstance(client.calls[0]["response_format"], dict)
        self.assertTrue(any(event.get("type") == "clarification_plan" and event.get("verdict") == "ask" for event in agent.events))

    def test_clarification_planner_honors_explicit_question_request_even_when_model_proceeds(self) -> None:
        client = FakeClient(
            ['{"verdict":"proceed","reason":"Enough information available.","ambiguities":[],"questions":[]}'],
            script_question_planner=True,
        )
        root, _client, _tools, agent = self._workspace_agent(client, debate_enabled=False)
        (root / "app.py").write_text("def value():\n    return 1\n", encoding="utf-8")

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "all"}):
            result = agent.handle_user(
                "Before you edit anything, ask me one clarification question first and do not assume. "
                "Refactor the architecture heavily, but keep the important external surfaces stable."
            )

        self.assertTrue(result.completed)
        self.assertIn("Need one clarification before continuing:", result.message)
        self.assertIn("Which boundary should stay fixed in this pass", result.message)
        self.assertTrue(any(event.get("type") == "clarification_plan" and event.get("verdict") == "ask" for event in agent.events))

    def test_question_planner_normalization_prefers_high_quality_eba_question(self) -> None:
        _root, _client, _tools, agent = self._workspace_agent(debate_enabled=False)

        decision = agent._normalize_question_planner_payload(
            {
                "verdict": "ask",
                "reason": "The request is broad and leaves the rewrite boundary unresolved.",
                "questions": [
                    {"question": "Should I proceed?", "why_it_matters": "Need permission first.", "choices": ["yes", "no"]},
                    {
                        "question": "Which boundary should stay fixed in this pass: CLI surface, session/transcript format, tool contracts, or benchmark comparability?",
                        "why_it_matters": "That boundary determines how aggressive the internal rewrite can be.",
                        "recommended_default": "Keep the CLI surface stable first.",
                        "choices": ["CLI surface", "session/transcript format", "tool contracts", "benchmark comparability"],
                    },
                ],
            },
            request_text="Refactor the architecture heavily, but keep the important external surfaces stable.",
        )

        self.assertEqual(decision["verdict"], "ask")
        self.assertEqual(len(decision["questions"]), 1)
        question = decision["questions"][0]
        self.assertTrue(question["eba_style"])
        self.assertGreaterEqual(question["quality_score"], 6)
        self.assertIn("Which boundary should stay fixed", question["question"])
        self.assertEqual(question["choices"], ["CLI surface", "session/transcript format", "tool contracts", "benchmark comparability"])

    def test_question_planner_fallback_generates_architecture_boundary_question(self) -> None:
        _root, _client, _tools, agent = self._workspace_agent(debate_enabled=False)

        decision = agent._normalize_question_planner_payload(
            {
                "verdict": "proceed",
                "reason": "The request is broad and leaves the rewrite boundary unspecified.",
                "questions": [],
                "ambiguities": [],
            },
            request_text="Refactor the architecture heavily, but do not break the surfaces that matter most.",
        )

        self.assertEqual(decision["verdict"], "ask")
        self.assertEqual(len(decision["questions"]), 1)
        question = decision["questions"][0]
        self.assertIn("Which boundary should stay fixed in this pass", question["question"])
        self.assertTrue(question["eba_style"])
        self.assertIn("CLI surface", question["choices"])
        self.assertIn("tool contracts", question["choices"])

    def test_clarification_planner_skips_focused_path_edit(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"replace_in_file","arguments":{"path":"app.py","old":"return \'old\'","new":"return \'new\'"}}',
                '{"type":"final","message":"Updated app.py."}',
            ],
            script_question_planner=True,
        )
        root, _client, _tools, agent = self._workspace_agent(client, debate_enabled=False)
        (root / "app.py").write_text("def value():\n    return 'old'\n", encoding="utf-8")

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "all"}):
            result = agent.handle_user("Fix app.py by changing old to new.")

        self.assertTrue(result.completed)
        self.assertIn("return 'new'", (root / "app.py").read_text(encoding="utf-8"))
        self.assertFalse(any(event.get("type") == "clarification_plan" for event in agent.events))

    # Focused loop-cap and broad-context planner coverage lives in test_agent_failure_compression.py.

    def test_context_planner_auto_maps_recent_test_to_implementation_target(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"tests/test_pkg.py"}}',
                '{"type":"tool","name":"search","arguments":{"query":"wrapped"}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"tests/test_pkg.py","start":1,"end":4}}',
                '{"type":"final","message":"Relevant implementation file is src/core.py."}',
            ]
        )
        root, _client, tools, agent = self._workspace_agent(
            client, debate_enabled=False, max_tool_rounds=5, tool_cls=CountingToolExecutor
        )
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

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Inspect tests/test_pkg.py and identify the relevant implementation file.")

        self.assertEqual(result.message, "Relevant implementation file is src/core.py.")
        self.assertEqual(tools.execute_counts.get("find_implementation_target"), 1)
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_calls[:3], ["read_file", "search", "find_implementation_target"])
        self.assertTrue(any("Use the grounded implementation target(s) src/core.py." in message["content"] for message in agent.messages if message["role"] == "user"))

    # Focused identifier-search and context-pack grounding coverage lives in test_agent_grounding_path_repair.py.

    def test_context_planner_auto_outlines_narrowed_repo_search_without_source_context(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"search","arguments":{"query":"business logic"}}',
                '{"type":"tool","name":"search","arguments":{"query":"business logic"}}',
                '{"type":"final","message":"src/core.py contains the relevant implementation."}',
            ]
        )
        root, _client, tools, agent = self._workspace_agent(
            client, debate_enabled=False, max_tool_rounds=5, tool_cls=CountingToolExecutor
        )
        (root / "README.md").write_text("overview\n", encoding="utf-8")
        (root / "src").mkdir()
        (root / "src" / "core.py").write_text("# business logic\n\ndef wrapped():\n    return 'ok'\n", encoding="utf-8")

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Inspect the repo and summarize the relevant implementation structure.")

        self.assertEqual(result.message, "src/core.py contains the relevant implementation.")
        self.assertEqual(tools.execute_counts.get("code_outline"), 1)
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_calls[:2], ["search", "code_outline"])
        self.assertTrue(any("Use the code outline for src/core.py." in message["content"] for message in agent.messages if message["role"] == "user"))

    def test_context_planner_auto_outlines_single_code_file_from_list_files(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"list_files","arguments":{"path":"."}}',
                '{"type":"tool","name":"read_file","arguments":{"path":"README.md"}}',
                '{"type":"final","message":"app.py contains the relevant implementation."}',
            ]
        )
        root, _client, tools, agent = self._workspace_agent(
            client, debate_enabled=False, max_tool_rounds=5, tool_cls=CountingToolExecutor
        )
        (root / "README.md").write_text("overview\n", encoding="utf-8")
        (root / "app.py").write_text("def wrapped():\n    return 'ok'\n", encoding="utf-8")

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Inspect the repo and summarize the relevant implementation structure.")

        self.assertEqual(result.message, "app.py contains the relevant implementation.")
        self.assertEqual(tools.execute_counts.get("code_outline"), 1)
        tool_calls = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_calls[:2], ["list_files", "code_outline"])
        self.assertTrue(any("Use the code outline for app.py." in message["content"] for message in agent.messages if message["role"] == "user"))

    # Focused shell-inspection normalization coverage lives in test_agent_shell_command_preflight.py.


    def test_passing_old_tests_do_not_satisfy_package_feature_request(self) -> None:
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
            "from dataclasses import dataclass\n\n\n"
            "@dataclass(frozen=True)\n"
            "class ReportRow:\n"
            "    name: str\n"
            "    count: int\n"
            "    active: bool\n\n\n"
            "def export_csv(rows: list[ReportRow]) -> str:\n"
            "    lines = [\"name,count,active\"]\n"
            "    for row in rows:\n"
            "        lines.append(f\"{row.name},{row.count},{str(row.active).lower()}\")\n"
            "    return \"\\n\".join(lines) + \"\\n\"\n",
            encoding="utf-8",
        )
        (root / "tests" / "test_exporter.py").write_text(
            "import unittest\n\n"
            "from reports import ReportRow, export_csv\n\n\n"
            "class ExporterTests(unittest.TestCase):\n"
            "    def test_export_csv(self) -> None:\n"
            "        self.assertEqual(export_csv([ReportRow(\"alpha\", 2, True)]), \"name,count,active\\nalpha,2,true\\n\")\n\n\n"
            "if __name__ == \"__main__\":\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )
        (root / "README.md").write_text("# Reports\n\nUse `export_csv(rows)` for CSV output.\n", encoding="utf-8")
        command = f"{sys.executable} -m unittest discover -s tests -v"
        tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
        agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)
        request_text = (
            "Add an export_ndjson(rows) function to this report exporter. It should serialize each ReportRow "
            "as one JSON object per line with keys name, count, and active in that order, preserve row order, "
            "and end the output with a trailing newline when rows are present. It should return an empty string "
            "for no rows. Export it from the package __init__.py. Update README with the new NDJSON export "
            "behavior. Add tests for multiple rows, empty rows, and escaping names with quotes or newlines. "
            "Run the tests and prove the behavior with a shell command."
        )
        obligations = agent._derive_request_obligations(
            request_text=request_text,
            required_tool_names=set(),
            required_mutation_paths=set(),
            code_mutation_required=True,
            test_run_required=True,
        )
        blocked_shortcuts: set[str] = set()

        result = agent._try_handle_deterministic_turn(
            request_text=request_text,
            exact_file_write=None,
            target_line_read=None,
            symbol_read=None,
            exact_shell_command=None,
            expected_exact_reply_text=None,
            required_tool_names=set(),
            forbidden_tool_names=set(),
            session_memory_request=False,
            requested_git_diff_mode=None,
            successful_tool_results=[],
            request_obligations=obligations,
            blocked_deterministic_shortcuts=blocked_shortcuts,
        )

        self.assertIsNone(result)
        self.assertEqual(tools.execute_counts.get("run_test"), 1)
        self.assertEqual(blocked_shortcuts, {"old_tests_only_success"})
        repeated_result = agent._try_handle_deterministic_turn(
            request_text=request_text,
            exact_file_write=None,
            target_line_read=None,
            symbol_read=None,
            exact_shell_command=None,
            expected_exact_reply_text=None,
            required_tool_names=set(),
            forbidden_tool_names=set(),
            session_memory_request=False,
            requested_git_diff_mode=None,
            successful_tool_results=[],
            request_obligations=obligations,
            blocked_deterministic_shortcuts=blocked_shortcuts,
        )

        self.assertIsNone(repeated_result)
        self.assertEqual(tools.execute_counts.get("run_test"), 1)
        self.assertTrue(
            any(
                event.get("type") == "deterministic_turn"
                and event.get("phase") == "blocked_old_tests_only_success"
                for event in agent.events
            )
        )
        self.assertFalse(
            any(
                event.get("type") == "assistant_synthesized"
                and event.get("content") == "Tests already pass."
                for event in agent.events
            )
        )




    # Focused shell preview and find normalization coverage lives in test_agent_shell_command_preflight.py.

    # Focused context-planner grounding refinement coverage lives in test_agent_grounding_path_repair.py.






    # Focused symbol-search disambiguation coverage lives in test_agent_grounding_path_repair.py.


    # Focused mutation-guard coverage lives in test_agent_post_edit_validation.py.

    def test_keep_tests_green_creates_test_run_obligation(self) -> None:
        tools = ToolExecutor(self._workspace_scratch(), approval_mode="auto")
        agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)

        self.assertTrue(agent._request_requires_test_run("Add a stats command and keep tests green."))
        self.assertTrue(agent._request_requires_test_run("Update the CLI and keep the tests passing."))











    # Focused missing-path final-claim coverage lives in test_agent_post_edit_validation.py.


    # Focused contract-guard coverage lives in test_agent_post_edit_validation.py.







    # Focused spec-guided post-edit repair coverage lives in test_agent_post_edit_validation.py.

    # Focused package-repair feature-delivery coverage lives in test_agent_post_edit_validation.py.

    # Focused failure-delta coverage lives in test_agent_failure_compression.py.

    # Focused context-pack preload coverage lives in test_agent_grounding_path_repair.py.

    # Focused prompt and primary-tool policy coverage lives in test_agent_prompt_policy.py.

    def test_agent_stops_after_max_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient(['{"type":"tool","name":"list_files","arguments":{}}'] * 2)
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", max_tool_rounds=1, debate_enabled=False)
            result = agent.handle_user("loop forever")

        self.assertIn("maximum tool rounds", result.message)

    # Focused failed-edit recovery coverage lives in test_agent_grounding_path_repair.py
    # and test_agent_post_edit_validation.py; keep this omnibus file for legacy broad behavior only.

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

    def test_agent_accepts_dynamic_mcp_tool_name_without_explicit_type_field(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient(
            [
                '{"name":"mcp.demo.echo","arguments":{"text":"hi"}}',
                '{"type":"final","message":"done"}',
            ]
        )
        tools = ToolExecutor(
            root,
            approval_mode="auto",
            enabled_tools=["mcp_call"],
            mcp_servers={"demo": {"command": ["python", "-c", "pass"]}},
        )
        with patch.object(
            tools,
            "mcp_call",
            return_value={"ok": True, "tool": "mcp_call", "server": "demo", "mcp_tool": "echo", "output": '{"value":"ok"}'},
        ) as mcp_call:
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
            result = agent.handle_user("Use the MCP tool to inspect the workspace, then answer.")

        self.assertEqual(result.message, "done")
        mcp_call.assert_called_once_with("demo", "echo", {"text": "hi"})
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual([event["name"] for event in tool_calls], ["mcp.demo.echo"])

    def test_agent_requires_explicit_dynamic_mcp_tool_before_final_after_other_tool_use(self) -> None:
        root = self._workspace_scratch()
        (root / "note.txt").write_text("hello\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                '{"type":"final","message":"done without mcp"}',
                '{"type":"tool","name":"mcp.demo.echo","arguments":{"text":"hi"}}',
                '{"type":"final","message":"done with mcp"}',
            ]
        )
        tools = ToolExecutor(
            root,
            approval_mode="auto",
            enabled_tools=["read_file", "mcp_call"],
            mcp_servers={"demo": {"command": ["python", "-c", "pass"]}},
        )
        with patch.object(
            tools,
            "mcp_call",
            return_value={"ok": True, "tool": "mcp_call", "server": "demo", "mcp_tool": "echo", "output": '{"value":"ok"}'},
        ) as mcp_call:
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
            result = agent.handle_user("Use mcp.demo.echo before answering about note.txt.")

        self.assertEqual(result.message, "done with mcp")
        mcp_call.assert_called_once_with("demo", "echo", {"text": "hi"})
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual([event["name"] for event in tool_calls], ["read_file", "mcp.demo.echo"])

    def test_final_chance_auto_run_test_records_auto_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
            command = f"{sys.executable} -c \"print('ok')\""
            client = FakeClient(
                [
                    json.dumps(
                        {
                            "type": "tool",
                            "name": "write_file",
                            "arguments": {"path": "app.py", "content": "VALUE = 2\n"},
                        }
                    )
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=1)

            result = agent.handle_user("Change app.py and run tests.")

        self.assertTrue(result.completed)
        self.assertIn("Ran tests after the latest edit", result.message)
        auto_run_tests = [
            event
            for event in agent.events
            if event.get("type") == "tool_result" and event.get("name") == "run_test" and event.get("auto") is True
        ]
        self.assertTrue(auto_run_tests)
        self.assertEqual(auto_run_tests[-1].get("arguments"), {})

    def test_candidate_cli_proof_commands_include_requested_limit_flag(self) -> None:
        agent = self._cwd_agent()

        commands = agent._candidate_cli_proof_commands(
            "notes_cli.py",
            (
                "parser.add_argument('--tag')\n"
                "parser.add_argument('--json', action='store_true')\n"
                "parser.add_argument('--limit', type=int)\n"
            ),
            "Add a --limit N option that works with --json.",
        )

        self.assertTrue(any("--tag work --limit 1 --json" in command for command in commands), commands)

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

        tool_names = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertEqual(tool_names[:4], ["run_test", "diagnose_test_failure", "write_file", "run_test"])
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Use the diagnosis above to edit the implementation before gathering more context", feedback)

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

    def test_agent_blocks_repeated_identical_syntax_error_edit_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
            bad_edit = {
                "type": "tool",
                "name": "edit_intent",
                "arguments": {
                    "path": "app.py",
                    "intent": "replace_body",
                    "target": "add",
                    "replacement": "if ':",
                },
            }
            client = FakeClient(
                [
                    json.dumps(bad_edit),
                    json.dumps(bad_edit),
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": "def add(left, right):\n    return left + right\n"}}),
                    json.dumps({"type": "final", "message": "app.py updated"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=6)

            with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "baseline"}):
                result = agent.handle_user("Fix app.py.")
            final_text = (root / "app.py").read_text(encoding="utf-8")

        self.assertIn(result.message, {"app.py updated", "Ran validation after the latest edit: passed."})
        self.assertEqual(tools.execute_counts.get("edit_intent"), 1)
        self.assertIn("return left + right", final_text)
        guards = [event for event in agent.events if event["type"] == "tool_error_guard"]
        self.assertEqual(guards[0]["error_class"], "syntax_error")

    def test_agent_blocks_stub_like_repair_edit_after_failed_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def add(left, right):\n    pass\n", encoding="utf-8")
            fail_command = subprocess.list2cmdline([sys.executable, "-c", "import sys; print('AssertionError: None != 3'); sys.exit(1)"])
            pass_command = subprocess.list2cmdline([sys.executable, "-c", "print('OK')"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": fail_command}}),
                    json.dumps({"type": "tool", "name": "edit_intent", "arguments": {"path": "app.py", "intent": "replace_body", "target": "add", "replacement": "# Implementation goes here"}}),
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": "def add(left, right):\n    return left + right\n"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": pass_command}}),
                    json.dumps({"type": "final", "message": "fixed"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

            with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "baseline"}):
                result = agent.handle_user("Fix app.py and run tests.")
            final_text = (root / "app.py").read_text(encoding="utf-8")

        self.assertEqual(result.message, "fixed")
        self.assertIsNone(tools.execute_counts.get("edit_intent"))
        self.assertEqual(tools.execute_counts.get("write_file"), 1)
        self.assertIn("return left + right", final_text)
        guards = [event for event in agent.events if event.get("guard") == "stub-repair-edit"]
        self.assertEqual(len(guards), 1)

    def test_agent_pivots_after_repeated_failed_mutating_edits_on_same_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def add(left, right):\n    pass\n", encoding="utf-8")
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "edit_intent", "arguments": {"path": "app.py", "intent": "replace_text", "target": "return missing", "replacement": "return left + right"}}),
                    json.dumps({"type": "tool", "name": "edit_intent", "arguments": {"path": "app.py", "intent": "replace_text", "target": "return other_missing", "replacement": "return left + right"}}),
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "app.py"}}),
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": "def add(left, right):\n    return left + right\n"}}),
                    json.dumps({"type": "final", "message": "fixed"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

            with patch.object(OllamaCodeAgent, "_spec_guided_repair_has_actionable_spec", return_value=False):
                result = agent.handle_user("Fix app.py.")
            final_text = (root / "app.py").read_text(encoding="utf-8")

        self.assertEqual(result.message, "fixed")
        self.assertEqual(tools.execute_counts.get("edit_intent"), 1)
        self.assertEqual(tools.execute_counts.get("read_file"), 1)
        self.assertEqual(tools.execute_counts.get("write_file"), 1)
        self.assertIn("return left + right", final_text)
        guards = [event for event in agent.events if event.get("guard") == "repeated-mutating-failure-pivot"]
        self.assertEqual(len(guards), 1)



    def test_failed_tests_feedback_includes_stubs_and_unittest_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def add(left, right):\n    pass\n", encoding="utf-8")
            (root / "app_test.py").write_text(
                "import unittest\nfrom app import add\n\n"
                "class AppTest(unittest.TestCase):\n"
                "    def test_adds(self):\n"
                "        self.assertEqual(add(1, 2), 3)\n",
                encoding="utf-8",
            )
            command = subprocess.list2cmdline([sys.executable, "-m", "unittest", "discover", "-p", "*_test.py", "-v"])
            client = FakeClient(
                [
                    json.dumps({"strategy": "normal_loop", "reason": "exercise failed-test feedback in the normal loop"}),
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "app.py"}}),
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "app_test.py"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {}}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=3)

            agent.handle_user("Implement app.py and run tests.")

        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Remaining stubs: app.py::add", feedback)
        self.assertIn("Test examples:", feedback)
        self.assertIn("add(1, 2) -> 3", feedback)

    def test_no_edit_guard_blocks_read_loop_after_failed_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def add(left, right):\n    pass\n", encoding="utf-8")
            (root / "app_test.py").write_text(
                "import unittest\nfrom app import add\n\n"
                "class AppTest(unittest.TestCase):\n"
                "    def test_adds(self):\n"
                "        self.assertEqual(add(1, 2), 3)\n",
                encoding="utf-8",
            )
            command = subprocess.list2cmdline([sys.executable, "-m", "unittest", "discover", "-p", "*_test.py", "-v"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "app.py"}}),
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "app_test.py"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {}}),
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "app.py"}}),
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "app_test.py"}}),
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "app.py"}}),
                    json.dumps({"type": "tool", "name": "edit_intent", "arguments": {"path": "app.py", "intent": "replace_body", "target": "add", "replacement": "return left + right"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {}}),
                    json.dumps({"type": "final", "message": "fixed"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=10)

            with patch.object(OllamaCodeAgent, "_spec_guided_repair_has_actionable_spec", return_value=False):
                result = agent.handle_user("Implement app.py and run tests.")
            final_text = (root / "app.py").read_text(encoding="utf-8")

        self.assertEqual(result.message, "fixed")
        read_calls = [event for event in agent.events if event.get("type") == "tool_call" and event.get("name") == "read_file"]
        self.assertEqual(len(read_calls), 3)
        self.assertEqual(tools.execute_counts.get("edit_intent"), 1)
        self.assertEqual(tools.execute_counts.get("diagnose_test_failure"), 1)
        diagnosis_guards = [event for event in agent.events if event.get("guard") == "diagnose-first-failed-test"]
        no_edit_guards = [event for event in agent.events if event.get("guard") == "no-edit-after-failed-test"]
        self.assertEqual(len(diagnosis_guards), 1)
        self.assertEqual(len(no_edit_guards), 1)
        self.assertIn("return left + right", final_text)

    def test_bulk_stub_guard_blocks_rerun_until_compact_stub_file_is_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def add(left, right):\n    pass\n\n\ndef sub(left, right):\n    pass\n", encoding="utf-8")
            (root / "app_test.py").write_text(
                "import unittest\nfrom app import add, sub\n\n"
                "class AppTest(unittest.TestCase):\n"
                "    def test_math(self):\n"
                "        self.assertEqual(add(1, 2), 3)\n"
                "        self.assertEqual(sub(3, 1), 2)\n",
                encoding="utf-8",
            )
            command = subprocess.list2cmdline([sys.executable, "-m", "unittest", "discover", "-p", "*_test.py", "-v"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "app.py"}}),
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "app_test.py"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {}}),
                    json.dumps({"type": "tool", "name": "edit_intent", "arguments": {"path": "app.py", "intent": "replace_body", "target": "add", "replacement": "return left + right"}}),
                    json.dumps(
                        {
                            "type": "tool",
                            "name": "replace_symbols",
                            "arguments": {
                                "path": "app.py",
                                "replacements": [
                                    {"symbol": "add", "content": "def add(left, right):\n    return left + right\n"},
                                    {"symbol": "sub", "content": "def sub(left, right):\n    return left - right\n"},
                                ],
                            },
                        }
                    ),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {}}),
                    json.dumps({"type": "final", "message": "fixed"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=9)

            with patch.object(OllamaCodeAgent, "_spec_guided_repair_has_actionable_spec", return_value=False):
                result = agent.handle_user("Implement app.py and run tests.")
            final_text = (root / "app.py").read_text(encoding="utf-8")

        self.assertEqual(result.message, "fixed")
        self.assertEqual(tools.execute_counts.get("edit_intent"), None)
        self.assertEqual(tools.execute_counts.get("replace_symbols"), 1)
        self.assertEqual(tools.execute_counts.get("run_test"), 2)
        guards = [event for event in agent.events if event.get("guard") == "bulk-stub-complete-edit"]
        self.assertEqual(len(guards), 1)
        self.assertIn("return left + right", final_text)
        self.assertIn("return left - right", final_text)

    def test_bulk_stub_guard_allows_repeated_partial_edit_after_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def add(left, right):\n    pass\n\n\ndef sub(left, right):\n    pass\n", encoding="utf-8")
            (root / "app_test.py").write_text(
                "import unittest\nfrom app import add, sub\n\n"
                "class AppTest(unittest.TestCase):\n"
                "    def test_math(self):\n"
                "        self.assertEqual(add(1, 2), 3)\n"
                "        self.assertEqual(sub(3, 1), 2)\n",
                encoding="utf-8",
            )
            command = subprocess.list2cmdline([sys.executable, "-m", "unittest", "discover", "-p", "*_test.py", "-v"])
            partial_edit = {
                "type": "tool",
                "name": "edit_intent",
                "arguments": {"path": "app.py", "intent": "replace_body", "target": "add", "replacement": "return left + right"},
            }
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "app.py"}}),
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "app_test.py"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {}}),
                    json.dumps(partial_edit),
                    json.dumps(partial_edit),
                    json.dumps(
                        {
                            "type": "tool",
                            "name": "replace_symbols",
                            "arguments": {
                                "path": "app.py",
                                "replacements": [
                                    {"symbol": "sub", "content": "def sub(left, right):\n    return left - right\n"},
                                ],
                            },
                        }
                    ),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {}}),
                    json.dumps({"type": "final", "message": "fixed"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=10)

            with patch.object(OllamaCodeAgent, "_spec_guided_repair_has_actionable_spec", return_value=False):
                result = agent.handle_user("Implement app.py and run tests.")
            final_text = (root / "app.py").read_text(encoding="utf-8")

        self.assertEqual(result.message, "fixed")
        self.assertEqual(tools.execute_counts.get("edit_intent"), 1)
        guards = [event for event in agent.events if event.get("guard") == "bulk-stub-complete-edit"]
        self.assertEqual(len(guards), 1)
        self.assertIn("return left + right", final_text)
        self.assertIn("return left - right", final_text)

    def test_spec_guided_repair_applies_validated_candidate_after_failed_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ops.py").write_text(
                "def add(left, right):\n    pass\n\n"
                "def sub(left, right):\n    pass\n\n"
                "def double(value):\n    pass\n",
                encoding="utf-8",
            )
            (root / "ops_test.py").write_text(
                "import unittest\nfrom ops import add, sub, double\n\n"
                "class OpsTest(unittest.TestCase):\n"
                "    def test_add_one(self):\n"
                "        self.assertEqual(add(1, 2), 3)\n"
                "    def test_add_two(self):\n"
                "        self.assertEqual(add(-1, 4), 3)\n"
                "    def test_sub_one(self):\n"
                "        self.assertEqual(sub(5, 2), 3)\n"
                "    def test_sub_two(self):\n"
                "        self.assertEqual(sub(2, 5), -3)\n"
                "    def test_double_one(self):\n"
                "        self.assertEqual(double(4), 8)\n"
                "    def test_double_two(self):\n"
                "        self.assertEqual(double(-3), -6)\n",
                encoding="utf-8",
            )
            command = subprocess.list2cmdline([sys.executable, "-m", "unittest", "discover", "-p", "*_test.py", "-v"])
            candidate = (
                "def add(left, right):\n"
                "    return left + right\n\n"
                "def sub(left, right):\n"
                "    return left - right\n\n"
                "def double(value):\n"
                "    return value * 2\n"
            )
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "ops.py"}}),
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "ops_test.py"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {}}),
                    candidate,
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

            result = agent.handle_user("Implement this Python exercise and run tests.")
            final_text = (root / "ops.py").read_text(encoding="utf-8")

        self.assertTrue(result.completed)
        self.assertIn("Spec-guided", result.message)
        self.assertIn("return left + right", final_text)
        self.assertIn("return left - right", final_text)
        self.assertIn("return value * 2", final_text)
        self.assertEqual(tools.execute_counts.get("edit_intent"), None)
        self.assertGreaterEqual(tools.execute_counts.get("implementation_spec", 0), 1)
        phases = [event.get("phase") for event in agent.events if event.get("type") == "spec_guided_repair"]
        self.assertTrue(any(phase in {"candidate_validation", "mechanical_candidate_validation"} for phase in phases))
        candidate_calls = [call for call in client.calls if call["response_format"] is None]
        if candidate_calls:
            candidate_prompt = candidate_calls[0]["messages"][1]["content"]  # type: ignore[index]
            self.assertIn("--- implementation spec ---", candidate_prompt)
            self.assertIn("test method", candidate_prompt)

    def test_spec_guided_repair_runs_for_small_single_stub_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scoreboard.py").write_text("def score_delta(base, bonus):\n    pass\n", encoding="utf-8")
            (root / "scoreboard_test.py").write_text(
                "import unittest\nfrom scoreboard import score_delta\n\n"
                "class ScoreboardTest(unittest.TestCase):\n"
                "    def test_positive_values(self):\n"
                "        self.assertEqual(score_delta(2, 3), 5)\n"
                "    def test_mixed_values(self):\n"
                "        self.assertEqual(score_delta(-1, 4), 3)\n"
                "    def test_zero_values(self):\n"
                "        self.assertEqual(score_delta(0, 0), 0)\n"
                "    def test_negative_values(self):\n"
                "        self.assertEqual(score_delta(-5, -2), -7)\n",
                encoding="utf-8",
            )
            command = subprocess.list2cmdline([sys.executable, "-m", "unittest", "discover", "-p", "*_test.py", "-v"])
            candidate = "def score_delta(base, bonus):\n    return base + bonus\n"
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "scoreboard.py"}}),
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "scoreboard_test.py"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {}}),
                    candidate,
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

            result = agent.handle_user("Implement this Python exercise and run tests.")
            final_text = (root / "scoreboard.py").read_text(encoding="utf-8")

        self.assertTrue(result.completed)
        self.assertIn("Spec-guided", result.message)
        self.assertEqual(final_text, candidate)
        self.assertGreaterEqual(tools.execute_counts.get("implementation_spec", 0), 1)
        phases = [event.get("phase") for event in agent.events if event.get("type") == "spec_guided_repair"]
        self.assertTrue(any(phase in {"candidate_validation", "mechanical_candidate_validation"} for phase in phases))
        candidate_calls = [call for call in client.calls if call["response_format"] is None]
        if candidate_calls:
            candidate_prompt = candidate_calls[0]["messages"][1]["content"]  # type: ignore[index]
            self.assertIn("score_delta(2, 3) -> 5", candidate_prompt)

    def test_spec_guided_repair_runs_for_small_single_non_stub_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "inventory.py").write_text("def total_units(counts):\n    return sum(counts) - 1\n", encoding="utf-8")
            (root / "inventory_test.py").write_text(
                "import unittest\nfrom inventory import total_units\n\n"
                "class InventoryTest(unittest.TestCase):\n"
                "    def test_sums_units(self):\n"
                "        self.assertEqual(total_units([2, 3, 4]), 9)\n"
                "    def test_empty(self):\n"
                "        self.assertEqual(total_units([]), 0)\n",
                encoding="utf-8",
            )
            command = subprocess.list2cmdline([sys.executable, "-m", "unittest", "discover", "-p", "*_test.py", "-v"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "inventory.py"}}),
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "inventory_test.py"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {}}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

            result = agent.handle_user("Run tests, fix inventory.py so total_units is correct, rerun tests, and summarize briefly.")
            final_text = (root / "inventory.py").read_text(encoding="utf-8")

        self.assertTrue(result.completed)
        self.assertIn("Spec-guided", result.message)
        self.assertIn("return sum(counts)", final_text)
        self.assertGreaterEqual(tools.execute_counts.get("implementation_spec", 0), 1)
        phases = [event.get("phase") for event in agent.events if event.get("type") == "spec_guided_repair"]
        self.assertIn("mechanical_candidate_validation", phases)

    def test_structured_test_repair_recovers_bad_configured_test_command_before_repair(self) -> None:
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "src/inventory.py"}}),
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "tests/test_inventory.py"}}),
                json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": "pytesst -q"}}),
            ]
        )
        with self._temp_python_agent(
            {
                "src/inventory.py": "def total_units(counts: list[int]) -> int:\n    return sum(counts) - 1\n",
                "tests/test_inventory.py": (
                "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                + "from inventory import *\nimport unittest\n\n"
                + "class InventoryTests(unittest.TestCase):\n"
                + "    def test_sums_units(self) -> None:\n        self.assertEqual(total_units([2, 3, 4]), 9)\n"
                + "    def test_empty(self) -> None:\n        self.assertEqual(total_units([]), 0)\n\n"
                + "if __name__ == '__main__':\n    unittest.main()\n"
                ),
            },
            client,
            tool_cls=CountingToolExecutor,
            test_command="pytesst -q",
            debate_enabled=False,
            max_tool_rounds=8,
        ) as (root, _client, tools, agent):
            result = agent.handle_user("Run tests, fix src/inventory.py so total_units is correct, rerun tests, and summarize briefly.")
            final_text = (root / "src" / "inventory.py").read_text(encoding="utf-8")

        self.assertTrue(result.completed)
        self.assertIn("Spec-guided", result.message)
        self.assertIn("return sum(counts)", final_text)
        tool_names = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        self.assertGreaterEqual(len(tool_names), 2)
        self.assertEqual(tool_names[0], "run_test")
        run_test_results = [event for event in agent.events if event.get("type") == "tool_result" and event.get("name") == "run_test"]
        self.assertGreaterEqual(len(run_test_results), 2)
        first_result = run_test_results[0].get("result", {})
        self.assertTrue(first_result.get("recovered"))
        self.assertEqual(first_result.get("original_command"), "pytesst -q")
        self.assertIn("unittest discover", str(first_result.get("command", "")))
        last_result = run_test_results[-1].get("result", {})
        self.assertTrue(last_result.get("ok"))
        self.assertIn("unittest discover", str(last_result.get("command", "")))

    def test_spec_guided_repair_runs_for_small_single_non_stub_module_with_one_direct_example(self) -> None:
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "src/balance.py"}}),
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "tests/test_balance_credit.py"}}),
                json.dumps({"type": "tool", "name": "run_test", "arguments": {}}),
            ]
        )
        with self._temp_python_agent(
            {
                "src/balance.py": "def apply_credit(amount: int, credit: int) -> int:\n    return amount - credit\n",
                "tests/test_balance_credit.py": (
                "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\nfrom balance import *\nimport unittest\n\n"
                "class BalanceTest(unittest.TestCase):\n"
                "    def test_apply_credit(self):\n"
                "        self.assertEqual(apply_credit(10, 3), 13)\n"
                "\n"
                ),
            },
            client,
            tool_cls=CountingToolExecutor,
            test_discover_args=("-s", "tests", "-v"),
            debate_enabled=False,
            max_tool_rounds=8,
        ) as (root, _client, tools, agent):
            result = agent.handle_user("Bug in src/balance.py: apply_credit(amount, credit) returns the wrong value. Read source and tests, fix it, run tests, and summarize briefly.")
            final_text = (root / "src" / "balance.py").read_text(encoding="utf-8")

        self.assertTrue(result.completed)
        self.assertIn("Spec-guided", result.message)
        self.assertIn("return amount + credit", final_text)
        self.assertGreaterEqual(tools.execute_counts.get("implementation_spec", 0), 1)
        phases = [event.get("phase") for event in agent.events if event.get("type") == "spec_guided_repair"]
        self.assertIn("mechanical_candidate_validation", phases)

    def test_structured_test_repair_skips_preemptive_spec_guided_for_import_bug_request(self) -> None:
        with self._temp_python_agent(
            {
                "src/pkg/__init__.py": "",
                "src/pkg/core.py": "from helpers import label\n\ndef wrapped() -> str:\n    return label('ok')\n",
                "src/pkg/helpers.py": "def label(value: str) -> str:\n    return f'[{value}]'\n",
                "tests/test_pkg.py": (
                "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                "import unittest\nfrom pkg.core import wrapped\n\n"
                "class PackageTests(unittest.TestCase):\n"
                "    def test_wrapped(self):\n"
                "        self.assertEqual(wrapped(), '[ok]')\n"
                ),
            },
            tool_cls=CountingToolExecutor,
            test_discover_args=("-s", "tests", "-v"),
            debate_enabled=False,
            max_tool_rounds=8,
        ) as (root, _client, tools, agent):
            result = agent._try_structured_test_driven_repair(
                request_text="Run tests, fix the package import bug in src/pkg/core.py, rerun tests, and summarize.",
                session_memory_request=False,
                mutation_required=True,
                test_run_required=True,
                required_tool_names=set(),
                forbidden_tool_names=set(),
                successful_tool_results=[],
                satisfied_tool_names=set(),
                tool_calls_this_turn=[],
            )

        self.assertIsNone(result)
        self.assertIsNone(tools.execute_counts.get("write_file"))
        self.assertIsNone(tools.execute_counts.get("run_test"))

    def test_structured_test_repair_skips_multi_file_refactor_request(self) -> None:
        with self._temp_python_agent(
            {
                "src/pricing.py": "def total(prices: list[int]) -> int:\n    return sum(prices)\n",
                "src/checkout.py": "from pricing import total\n\n\ndef checkout_total(prices: list[int]) -> int:\n    return total(prices)\n",
                "tests/test_checkout_refactor.py": (
                "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                "from pricing import total\nfrom checkout import checkout_total\nimport unittest\n\n"
                "class CheckoutRefactorTests(unittest.TestCase):\n"
                "    def test_total(self):\n"
                "        self.assertEqual(total([2, 3]), 5)\n"
                ),
                "docs/pricing.md": "Call `total(prices)`.\n",
            },
            tool_cls=CountingToolExecutor,
            test_discover_args=("-s", "tests", "-v"),
            debate_enabled=False,
            max_tool_rounds=8,
        ) as (root, _client, tools, agent):
            result = agent._try_structured_test_driven_repair(
                request_text="Refactor the pricing API from total(prices) to cart_total(prices). Update src/pricing.py, src/checkout.py, tests, and docs/pricing.md. Run tests and summarize briefly.",
                session_memory_request=False,
                mutation_required=True,
                test_run_required=True,
                required_tool_names=set(),
                forbidden_tool_names=set(),
                successful_tool_results=[],
                satisfied_tool_names=set(),
                tool_calls_this_turn=[],
            )

        self.assertIsNone(result)
        self.assertIsNone(tools.execute_counts.get("write_file"))
        self.assertIsNone(tools.execute_counts.get("run_test"))

    def test_preemptive_mechanical_spec_guided_repair_skips_primary_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "wordy.py").write_text("def answer(question):\n    pass\n", encoding="utf-8")
            (root / "wordy_test.py").write_text(
                "import unittest\nfrom wordy import answer\n\n"
                "class WordyTest(unittest.TestCase):\n"
                "    def test_just_a_number(self):\n"
                "        self.assertEqual(answer('What is 5?'), 5)\n"
                "    def test_addition(self):\n"
                "        self.assertEqual(answer('What is 1 plus 1?'), 2)\n"
                "    def test_subtraction(self):\n"
                "        self.assertEqual(answer('What is 4 minus -12?'), 16)\n"
                "    def test_multiplication(self):\n"
                "        self.assertEqual(answer('What is -3 multiplied by 25?'), -75)\n"
                "    def test_division(self):\n"
                "        self.assertEqual(answer('What is 33 divided by -3?'), -11)\n"
                "    def test_multiple_operations(self):\n"
                "        self.assertEqual(answer('What is 17 minus 6 plus 3?'), 14)\n"
                "    def test_unknown_operation(self):\n"
                "        with self.assertRaises(ValueError) as err:\n"
                "            answer('What is 52 cubed?')\n"
                "        self.assertEqual(err.exception.args[0], 'unknown operation')\n"
                "    def test_syntax_error(self):\n"
                "        with self.assertRaises(ValueError) as err:\n"
                "            answer('What is 1 plus?')\n"
                "        self.assertEqual(err.exception.args[0], 'syntax error')\n",
                encoding="utf-8",
            )
            command = subprocess.list2cmdline([sys.executable, "-m", "unittest", "discover", "-p", "*_test.py", "-v"])
            client = FakeClient([])
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

            result = agent.handle_user("Implement this Python exercise and run tests.")
            final_text = (root / "wordy.py").read_text(encoding="utf-8")

        self.assertTrue(result.completed)
        self.assertEqual(len(client.calls), 0)
        self.assertIn("Spec-guided mechanical repair applied", result.message)
        self.assertIn("plus", final_text)
        self.assertEqual(tools.execute_counts.get("run_test"), 1)
        phases = [event.get("phase") for event in agent.events if event.get("type") == "spec_guided_repair"]
        self.assertIn("preemptive_mechanical_start", phases)
        self.assertIn("mechanical_candidate_validation", phases)
        validation_events = [
            event
            for event in agent.events
            if event.get("type") == "spec_guided_repair" and event.get("phase") == "mechanical_candidate_validation"
        ]
        self.assertTrue(any(event.get("preemptive") is True for event in validation_events))

    def test_agent_require_llm_for_turn_skips_preemptive_mechanical_repair(self) -> None:
        class SingleToolClient(FakeClient):
            def __init__(self) -> None:
                super().__init__([])

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
                return ChatResponse(
                    content='{"type":"tool","name":"replace_in_file","arguments":{"path":"app.py","old":"return 1","new":"return 2"}}',
                    model=model,
                    raw={},
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def value():\n    return 1\n", encoding="utf-8")
            client = SingleToolClient()
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(
                client=client,
                tools=tools,
                model="fake-model",
                debate_enabled=False,
                require_llm_for_turn=True,
            )

            with patch.object(
                agent,
                "_try_preemptive_mechanical_spec_guided_repair",
                side_effect=AssertionError("preemptive repair should be skipped when an LLM call is required"),
            ):
                result = agent.handle_user("Edit app.py to return 2.")
            final_text = (root / "app.py").read_text(encoding="utf-8")

        self.assertTrue(result.completed)
        self.assertEqual(final_text, "def value():\n    return 2\n")
        self.assertGreaterEqual(len(client.calls), 1)
        self.assertGreaterEqual(tools.execute_counts.get("replace_in_file", 0), 1)

    def test_agent_require_llm_for_turn_uses_structured_test_driven_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scoreboard.py").write_text("def score_delta(base, bonus):\n    pass\n", encoding="utf-8")
            (root / "scoreboard_test.py").write_text(
                "import unittest\nfrom scoreboard import score_delta\n\n"
                "class ScoreboardTest(unittest.TestCase):\n"
                "    def test_positive_values(self):\n"
                "        self.assertEqual(score_delta(2, 3), 5)\n"
                "    def test_mixed_values(self):\n"
                "        self.assertEqual(score_delta(-1, 4), 3)\n"
                "    def test_zero_values(self):\n"
                "        self.assertEqual(score_delta(0, 0), 0)\n"
                "    def test_negative_values(self):\n"
                "        self.assertEqual(score_delta(-5, -2), -7)\n",
                encoding="utf-8",
            )
            command = subprocess.list2cmdline([sys.executable, "-m", "unittest", "discover", "-p", "*_test.py", "-v"])
            candidate = "def score_delta(base, bonus):\n    return base + bonus\n"
            client = FakeClient(
                [
                    json.dumps({"strategy": "spec_guided_repair", "reason": "small Python stub with executable tests"}),
                    candidate,
                ]
            )
            tools = RawImplementationSpecCountingToolExecutor(root, approval_mode="auto", test_command=command)
            agent = OllamaCodeAgent(
                client=client,
                tools=tools,
                model="fake-model",
                debate_enabled=False,
                require_llm_for_turn=True,
                max_tool_rounds=8,
            )

            with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "all"}):
                result = agent.handle_user("Implement scoreboard.py from the tests, run tests, and summarize briefly.")
            final_text = (root / "scoreboard.py").read_text(encoding="utf-8")

        self.assertTrue(result.completed)
        self.assertIn("Spec-guided", result.message)
        self.assertEqual(final_text, candidate)
        self.assertEqual(tools.execute_counts.get("implementation_spec"), 1)
        self.assertEqual(tools.raw_implementation_spec_calls, 1)
        self.assertEqual(tools.execute_counts.get("context_pack"), None)
        strategy_events = [event for event in agent.events if event.get("type") == "repair_strategy"]
        self.assertEqual(len(strategy_events), 1)
        self.assertEqual(strategy_events[0].get("strategy"), "spec_guided_repair")
        self.assertGreaterEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["messages"][0]["content"].startswith("You are a repair strategy planner"), True)  # type: ignore[index]

    def test_final_chance_validation_failure_uses_spec_guided_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scoreboard.py").write_text("def score_delta(base, bonus):\n    pass\n", encoding="utf-8")
            (root / "scoreboard_test.py").write_text(
                "import unittest\nfrom scoreboard import score_delta\n\n"
                "class ScoreboardTest(unittest.TestCase):\n"
                "    def test_positive_values(self):\n"
                "        self.assertEqual(score_delta(2, 3), 5)\n"
                "    def test_mixed_values(self):\n"
                "        self.assertEqual(score_delta(-1, 4), 3)\n",
                encoding="utf-8",
            )
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "scoreboard.py"}}),
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "scoreboard_test.py"}}),
                    json.dumps(
                        {
                            "type": "tool",
                            "name": "replace_in_file",
                            "arguments": {"path": "scoreboard.py", "old": "pass", "new": "return new_score - old_score"},
                        }
                    ),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=3)
            recovery_calls: list[str] = []

            def fake_repair(**kwargs: object) -> AgentResult | None:
                failed = kwargs.get("failed_run_test_result")
                tool = str(failed.get("tool", "")) if isinstance(failed, dict) else ""
                recovery_calls.append(tool)
                if tool in {"contract_check", "lint_typecheck", "run_test"}:
                    round_number = int(kwargs.get("round_number", 0))
                    return AgentResult("Recovered after validation failure.", rounds=round_number, completed=True)
                return None

            with patch.object(agent, "_try_structured_test_driven_repair", return_value=None):
                with patch.object(agent, "_try_spec_guided_repair", side_effect=fake_repair):
                    result = agent.handle_user("Implement scoreboard.py from the tests and run tests.")

        self.assertTrue(result.completed)
        self.assertEqual(result.message, "Recovered after validation failure.")
        self.assertIn("preemptive_spec_repair", recovery_calls)
        self.assertTrue(any(tool in {"contract_check", "lint_typecheck", "run_test"} for tool in recovery_calls))

    def test_preemptive_mechanical_repair_without_explicit_test_request_phrase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "wordy.py").write_text("def answer(question):\n    pass\n", encoding="utf-8")
            (root / "wordy_test.py").write_text(
                "import unittest\nfrom wordy import answer\n\n"
                "class WordyTest(unittest.TestCase):\n"
                "    def test_just_a_number(self):\n"
                "        self.assertEqual(answer('What is 5?'), 5)\n"
                "    def test_addition(self):\n"
                "        self.assertEqual(answer('What is 1 plus 1?'), 2)\n"
                "    def test_subtraction(self):\n"
                "        self.assertEqual(answer('What is 4 minus -12?'), 16)\n"
                "    def test_multiplication(self):\n"
                "        self.assertEqual(answer('What is -3 multiplied by 25?'), -75)\n"
                "    def test_division(self):\n"
                "        self.assertEqual(answer('What is 33 divided by -3?'), -11)\n"
                "    def test_multiple_operations(self):\n"
                "        self.assertEqual(answer('What is 17 minus 6 plus 3?'), 14)\n"
                "    def test_unknown_operation(self):\n"
                "        with self.assertRaises(ValueError) as err:\n"
                "            answer('What is 52 cubed?')\n"
                "        self.assertEqual(err.exception.args[0], 'unknown operation')\n"
                "    def test_syntax_error(self):\n"
                "        with self.assertRaises(ValueError) as err:\n"
                "            answer('What is 1 plus?')\n"
                "        self.assertEqual(err.exception.args[0], 'syntax error')\n",
                encoding="utf-8",
            )
            command = subprocess.list2cmdline([sys.executable, "-m", "unittest", "discover", "-p", "*_test.py", "-v"])
            client = FakeClient([])
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

            result = agent.handle_user("Implement this Python exercise.")
            final_text = (root / "wordy.py").read_text(encoding="utf-8")

        self.assertTrue(result.completed)
        self.assertEqual(len(client.calls), 0)
        self.assertIn("Spec-guided mechanical repair applied", result.message)
        self.assertNotIn("pass", final_text)
        phases = [event.get("phase") for event in agent.events if event.get("type") == "spec_guided_repair"]
        self.assertIn("preemptive_mechanical_start", phases)
        self.assertIn("mechanical_candidate_validation", phases)
        validation_events = [
            event
            for event in agent.events
            if event.get("type") == "spec_guided_repair" and event.get("phase") == "mechanical_candidate_validation"
        ]
        self.assertTrue(any(event.get("preemptive") is True for event in validation_events))

    def test_preemptive_simple_expression_repair_skips_primary_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "calc.py").write_text("def add(left: int, right: int) -> int:\n    return left - right\n", encoding="utf-8")
            (root / "calc_test.py").write_text(
                "import unittest\nfrom calc import add\n\n"
                "class CalcTest(unittest.TestCase):\n"
                "    def test_adds_values(self):\n"
                "        self.assertEqual(add(2, 3), 5)\n"
                "    def test_adds_negative_values(self):\n"
                "        self.assertEqual(add(-2, 5), 3)\n",
                encoding="utf-8",
            )
            command = subprocess.list2cmdline([sys.executable, "-m", "unittest", "discover", "-p", "*_test.py", "-v"])
            client = FakeClient([])
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

            result = agent.handle_user("Issue: calc.py add(left, right) returns the wrong value. Inspect source/tests, fix it, run tests.")
            final_text = (root / "calc.py").read_text(encoding="utf-8")

        self.assertTrue(result.completed)
        self.assertEqual(len(client.calls), 0)
        self.assertIn("return left + right", final_text)
        self.assertEqual(tools.execute_counts.get("run_test"), 1)

    def test_deterministic_project_function_rename_skips_primary_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "tests").mkdir()
            (root / "docs").mkdir()
            (root / "src" / "pricing.py").write_text("def total(prices: list[int]) -> int:\n    return sum(prices)\n", encoding="utf-8")
            (root / "tests" / "test_pricing.py").write_text(
                "import sys\nimport unittest\nsys.path.insert(0, 'src')\nfrom pricing import cart_total\n\n"
                "class PricingTest(unittest.TestCase):\n"
                "    def test_total(self):\n"
                "        self.assertEqual(cart_total([2, 3]), 5)\n",
                encoding="utf-8",
            )
            (root / "docs" / "pricing.md").write_text("Call `total(prices)` to compute totals.\n", encoding="utf-8")
            command = subprocess.list2cmdline([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
            client = FakeClient([])
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=8)

            result = agent.handle_user("Refactor the pricing API from total(prices) to cart_total(prices). Update src/pricing.py, tests, and docs/pricing.md. Run tests.")
            source = (root / "src" / "pricing.py").read_text(encoding="utf-8")
            docs = (root / "docs" / "pricing.md").read_text(encoding="utf-8")

        self.assertTrue(result.completed)
        self.assertEqual(len(client.calls), 0)
        self.assertIn("def cart_total", source)
        self.assertIn("cart_total(prices)", docs)
        self.assertEqual(tools.execute_counts.get("edit_intent"), 1)
        self.assertEqual(tools.execute_counts.get("run_test"), 1)

    def test_require_llm_for_turn_allows_deterministic_followup_after_first_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "tests").mkdir()
            (root / "docs").mkdir()
            (root / "src" / "pricing.py").write_text("def total(prices: list[int]) -> int:\n    return sum(prices)\n", encoding="utf-8")
            (root / "tests" / "test_pricing.py").write_text(
                "import sys\nimport unittest\nsys.path.insert(0, 'src')\nfrom pricing import cart_total\n\n"
                "class PricingTest(unittest.TestCase):\n"
                "    def test_total(self):\n"
                "        self.assertEqual(cart_total([2, 3]), 5)\n",
                encoding="utf-8",
            )
            (root / "docs" / "pricing.md").write_text("Call `total(prices)` to compute totals.\n", encoding="utf-8")
            command = subprocess.list2cmdline([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
            client = FakeClient(
                [
                    json.dumps(
                        {
                            "type": "tool",
                            "name": "edit_intent",
                            "arguments": {"path": "src/pricing.py", "intent": "rename_symbol", "target": "total", "replacement": "cart_total"},
                        }
                    ),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, require_llm_for_turn=True, max_tool_rounds=8)

            result = agent.handle_user("Refactor the pricing API from total(prices) to cart_total(prices). Update src/pricing.py, tests, and docs/pricing.md. Run tests.")
            source = (root / "src" / "pricing.py").read_text(encoding="utf-8")
            docs = (root / "docs" / "pricing.md").read_text(encoding="utf-8")

        self.assertTrue(result.completed)
        self.assertIn("Renamed symbol project-wide", result.message)
        self.assertEqual(len(client.calls), 1)
        self.assertIn("def cart_total", source)
        self.assertIn("cart_total(prices)", docs)
        self.assertEqual(tools.execute_counts.get("edit_intent"), 1)
        self.assertEqual(tools.execute_counts.get("run_test"), 1)

    def test_require_llm_for_turn_normalizes_initial_list_files_for_project_rename_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "tests").mkdir()
            (root / "docs").mkdir()
            (root / "src" / "pricing.py").write_text("def total(prices: list[int]) -> int:\n    return sum(prices)\n", encoding="utf-8")
            (root / "tests" / "test_pricing.py").write_text(
                "import sys\nimport unittest\nsys.path.insert(0, 'src')\nfrom pricing import cart_total\n\n"
                "class PricingTest(unittest.TestCase):\n"
                "    def test_total(self):\n"
                "        self.assertEqual(cart_total([2, 3]), 5)\n",
                encoding="utf-8",
            )
            (root / "docs" / "pricing.md").write_text("Call `total(prices)` to compute totals.\n", encoding="utf-8")
            command = subprocess.list2cmdline([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
            client = FakeClient([json.dumps({"type": "tool", "name": "list_files", "arguments": {"path": "."}})])
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, require_llm_for_turn=True, max_tool_rounds=8)

            result = agent.handle_user("Refactor the pricing API from total(prices) to cart_total(prices). Update src/pricing.py, tests, and docs/pricing.md. Run tests.")
            source = (root / "src" / "pricing.py").read_text(encoding="utf-8")
            docs = (root / "docs" / "pricing.md").read_text(encoding="utf-8")

        self.assertTrue(result.completed)
        self.assertIn("Renamed symbol project-wide", result.message)
        self.assertEqual(len(client.calls), 1)
        self.assertIsNone(tools.execute_counts.get("list_files"))
        self.assertEqual(tools.execute_counts.get("edit_intent"), 1)
        self.assertEqual(tools.execute_counts.get("run_test"), 1)
        self.assertIn("def cart_total", source)
        self.assertIn("cart_total(prices)", docs)
        tool_names = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        non_context_tool_names = [name for name in tool_names if name != "context_pack"]
        self.assertGreaterEqual(len(non_context_tool_names), 1)
        self.assertEqual(non_context_tool_names[0], "edit_intent")
        normalized = [event for event in agent.events if event.get("type") == "tool_normalized"]
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0].get("original_name"), "list_files")
        self.assertEqual(normalized[0].get("normalized_name"), "edit_intent")

    def test_spec_guided_repair_uses_verifier_model_after_candidate_failure(self) -> None:
        bad_candidate = (
            "def add(left, right):\n"
            "    return left - right\n\n"
            "def sub(left, right):\n"
            "    return left - right\n\n"
            "def double(value):\n"
            "    return value * 2\n"
        )
        good_candidate = (
            "def add(left, right):\n"
            "    return left + right\n\n"
            "def sub(left, right):\n"
            "    return left - right\n\n"
            "def double(value):\n"
            "    return value * 2\n"
        )
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "ops.py"}}),
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "ops_test.py"}}),
                json.dumps({"type": "tool", "name": "run_test", "arguments": {}}),
                bad_candidate,
                good_candidate,
            ],
            models=["fake-model", "strong-model"],
        )
        with self._temp_python_agent(
            {
                "ops.py": (
                "def add(left, right):\n    pass\n\n"
                "def sub(left, right):\n    pass\n\n"
                "def double(value):\n    pass\n"
                ),
                "ops_test.py": (
                "import unittest\nfrom ops import add, sub, double\n\n"
                "class OpsTest(unittest.TestCase):\n"
                "    def test_add_one(self):\n"
                "        self.assertEqual(add(1, 2), 3)\n"
                "    def test_add_two(self):\n"
                "        self.assertEqual(add(-1, 4), 3)\n"
                "    def test_sub_one(self):\n"
                "        self.assertEqual(sub(5, 2), 3)\n"
                "    def test_sub_two(self):\n"
                "        self.assertEqual(sub(2, 5), -3)\n"
                "    def test_double_one(self):\n"
                "        self.assertEqual(double(4), 8)\n"
                "    def test_double_two(self):\n"
                "        self.assertEqual(double(-3), -6)\n"
                ),
            },
            client,
            tool_cls=CountingToolExecutor,
            debate_enabled=False,
            verifier_model="strong-model",
            max_tool_rounds=8,
        ) as (root, _client, tools, agent):
            result = agent.handle_user("Implement this Python exercise and run tests.")
            validation_models = [
                event.get("model")
                for event in agent.events
                if event.get("type") == "spec_guided_repair" and event.get("phase") == "candidate_validation"
            ]

        self.assertTrue(result.completed)
        self.assertIn("Spec-guided", result.message)
        if validation_models:
            self.assertEqual(validation_models, ["fake-model", "strong-model"])

    def test_spec_guided_repair_fails_closed_after_candidate_failures(self) -> None:
        bad_candidate = "def add(left, right):\n    return left - right\n"
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "ops.py"}}),
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "ops_test.py"}}),
                json.dumps({"type": "tool", "name": "run_test", "arguments": {}}),
                bad_candidate,
                bad_candidate,
                bad_candidate,
            ],
            models=["fake-model"],
        )
        with self._temp_python_agent(
            {
                "ops.py": "def add(left, right):\n    pass\n",
                "ops_test.py": (
                "import unittest\nfrom ops import add\n\n"
                "class OpsTest(unittest.TestCase):\n"
                "    def test_add_one(self):\n"
                "        self.assertEqual(add(1, 2), 3)\n"
                "    def test_add_two(self):\n"
                "        self.assertEqual(add(-1, 4), 3)\n"
                "    def test_add_three(self):\n"
                "        self.assertEqual(add(0, 0), 0)\n"
                "    def test_add_four(self):\n"
                "        self.assertEqual(add(10, -7), 3)\n"
                "    def test_add_five(self):\n"
                "        self.assertEqual(add(2, 2), 4)\n"
                "    def test_add_six(self):\n"
                "        self.assertEqual(add(-3, -4), -7)\n"
                ),
            },
            client,
            tool_cls=CountingToolExecutor,
            debate_enabled=False,
            max_tool_rounds=8,
        ) as (root, _client, tools, agent):
            with patch.object(
                tools,
                "synthesize_simple_expression_candidate",
                return_value={"ok": False, "tool": "synthesize_simple_expression_candidate", "summary": "disabled in fail-closed test"},
            ):
                result = agent.handle_user("Implement this Python exercise and run tests.")
            final_text = (root / "ops.py").read_text(encoding="utf-8")

        self.assertFalse(result.completed)
        self.assertIn("Spec-guided candidate repair did not produce", result.message)
        self.assertIn("expected 3", result.message)
        self.assertIn("pass", final_text)
        phases = [event.get("phase") for event in agent.events if event.get("type") == "spec_guided_repair"]
        self.assertIn("failed_closed", phases)

    def test_spec_guided_generation_timeout_falls_back_to_primary_loop(self) -> None:
        class TimeoutCandidateClient(FakeClient):
            def chat(self, **kwargs: object) -> ChatResponse:
                messages = kwargs.get("messages")
                if isinstance(messages, list) and messages and str(messages[0].get("content", "")).startswith(
                    "You generate complete Python source files"
                ):
                    self.calls.append(
                        {
                            "model": kwargs.get("model"),
                            "messages": list(messages),
                            "response_format": kwargs.get("response_format"),
                            "think": kwargs.get("think"),
                            "options": kwargs.get("options"),
                        }
                    )
                    raise TimeoutError("candidate timed out")
                return super().chat(**kwargs)  # type: ignore[arg-type]

        candidate = "def wrap(text):\n    return '[' + text + ']'\n"
        client = TimeoutCandidateClient(
            [
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "ops.py"}}),
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "ops_test.py"}}),
                json.dumps({"type": "tool", "name": "run_test", "arguments": {}}),
                json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "ops.py", "content": candidate}}),
                json.dumps({"type": "tool", "name": "run_test", "arguments": {}}),
                json.dumps({"type": "final", "message": "done"}),
            ],
            models=["fake-model"],
        )
        with self._temp_python_agent(
            {
                "ops.py": "def wrap(text):\n    pass\n",
                "ops_test.py": (
                "import unittest\nfrom ops import wrap\n\n"
                "class OpsTest(unittest.TestCase):\n"
                "    def test_one(self):\n"
                "        self.assertEqual(wrap('A1'), '[A1]')\n"
                "    def test_two(self):\n"
                "        self.assertEqual(wrap('xy'), '[xy]')\n"
                "    def test_three(self):\n"
                "        self.assertEqual(wrap('Q'), '[Q]')\n"
                "    def test_four(self):\n"
                "        self.assertEqual(wrap(''), '[]')\n"
                "    def test_five(self):\n"
                "        self.assertEqual(wrap('ab!'), '[ab!]')\n"
                "    def test_six(self):\n"
                "        self.assertEqual(wrap('Z9?'), '[Z9?]')\n"
                ),
            },
            client,
            tool_cls=CountingToolExecutor,
            debate_enabled=False,
            max_tool_rounds=10,
        ) as (root, _client, tools, agent):
            result = agent.handle_user("Implement this Python exercise and run tests.")
            final_text = (root / "ops.py").read_text(encoding="utf-8")

        self.assertTrue(result.completed)
        self.assertIn("return '[' + text + ']'", final_text)
        self.assertEqual(tools.execute_counts.get("write_file"), 1)
        phases = [event.get("phase") for event in agent.events if event.get("type") == "spec_guided_repair"]
        self.assertIn("soft_failed", phases)
        self.assertNotIn("failed_closed", phases)

    def test_spec_guided_repair_runs_after_malformed_edit_with_tests_read(self) -> None:
        candidate = "def add(left, right):\n    return left + right\n"
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "ops.py"}}),
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "ops_test.py"}}),
                json.dumps(
                    {
                        "type": "tool",
                        "name": "edit_intent",
                        "arguments": {"intent": "replace_body", "path": "ops.py", "symbol": "add", "body": "return \""},
                    }
                ),
                candidate,
            ]
        )
        with self._temp_python_agent(
            {
                "ops.py": "def add(left, right):\n    pass\n",
                "ops_test.py": (
                "import unittest\nfrom ops import add\n\n"
                "class OpsTest(unittest.TestCase):\n"
                "    def test_add_one(self):\n"
                "        self.assertEqual(add(1, 2), 3)\n"
                "    def test_add_two(self):\n"
                "        self.assertEqual(add(-1, 4), 3)\n"
                "    def test_add_three(self):\n"
                "        self.assertEqual(add(0, 0), 0)\n"
                "    def test_add_four(self):\n"
                "        self.assertEqual(add(10, -7), 3)\n"
                "    def test_add_five(self):\n"
                "        self.assertEqual(add(2, 2), 4)\n"
                "    def test_add_six(self):\n"
                "        self.assertEqual(add(-3, -4), -7)\n"
                ),
            },
            client,
            tool_cls=CountingToolExecutor,
            debate_enabled=False,
            max_tool_rounds=8,
        ) as (root, _client, tools, agent):
            result = agent.handle_user("Implement this Python exercise and run tests.")
            final_text = (root / "ops.py").read_text(encoding="utf-8")

        self.assertTrue(result.completed)
        self.assertIn("Spec-guided", result.message)
        self.assertIn("return left + right", final_text)
        self.assertIsNone(tools.execute_counts.get("edit_intent"))
        self.assertGreaterEqual(tools.execute_counts.get("implementation_spec", 0), 1)
        phases = [event.get("phase") for event in agent.events if event.get("type") == "spec_guided_repair"]
        self.assertIn("start", phases)
        candidate_calls = [call for call in client.calls if call["response_format"] is None]
        if candidate_calls:
            candidate_prompt = candidate_calls[0]["messages"][1]["content"]  # type: ignore[index]
            self.assertIn("Direct edit skipped", candidate_prompt)
            self.assertNotIn('return "', candidate_prompt)

    def test_spec_guided_repair_runs_after_post_edit_probe_failure(self) -> None:
        fixed_candidate = "def add(left, right):\n    return left + right\n"
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "ops.py"}}),
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "ops_test.py"}}),
                json.dumps(
                    {
                        "type": "tool",
                        "name": "edit_intent",
                        "arguments": {"intent": "replace_body", "path": "ops.py", "symbol": "add", "body": "return left - right"},
                    }
                ),
                fixed_candidate,
            ]
        )
        with self._temp_python_agent(
            {
                "ops.py": "def add(left, right):\n    pass\n",
                "ops_test.py": (
                "import unittest\nfrom ops import add\n\n"
                "class OpsTest(unittest.TestCase):\n"
                "    def test_add_one(self):\n"
                "        self.assertEqual(add(1, 2), 3)\n"
                "    def test_add_two(self):\n"
                "        self.assertEqual(add(-1, 4), 3)\n"
                "    def test_add_three(self):\n"
                "        self.assertEqual(add(0, 0), 0)\n"
                "    def test_add_four(self):\n"
                "        self.assertEqual(add(10, -7), 3)\n"
                "    def test_add_five(self):\n"
                "        self.assertEqual(add(2, 2), 4)\n"
                "    def test_add_six(self):\n"
                "        self.assertEqual(add(-3, -4), -7)\n"
                ),
            },
            client,
            tool_cls=CountingToolExecutor,
            debate_enabled=False,
            max_tool_rounds=8,
        ) as (root, _client, tools, agent):
            result = agent.handle_user("Implement this Python exercise and run tests.")
            final_text = (root / "ops.py").read_text(encoding="utf-8")

        self.assertTrue(result.completed)
        self.assertIn("Spec-guided", result.message)
        self.assertIn("return left + right", final_text)
        self.assertIsNone(tools.execute_counts.get("edit_intent"))
        phases = [event.get("phase") for event in agent.events if event.get("type") == "spec_guided_repair"]
        self.assertTrue(any(phase in {"candidate_validation", "mechanical_candidate_validation"} for phase in phases))

    def test_spec_guided_repair_applies_mechanical_prefix_rotation_candidate(self) -> None:
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "pig.py"}}),
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "pig_test.py"}}),
                json.dumps({"type": "tool", "name": "edit_intent", "arguments": {"path": "pig.py", "intent": "replace_body", "target": "translate", "replacement": "return text"}}),
            ]
        )
        with self._temp_python_agent(
            {
                "pig.py": "def translate(text):\n    pass\n",
                "pig_test.py": (
                "import unittest\nfrom pig import translate\n\n"
                "class PigTest(unittest.TestCase):\n"
                "    def test_word_beginning_with_a_vowel(self):\n"
                "        self.assertEqual(translate('apple'), 'appleay')\n"
                "    def test_word_beginning_with_p(self):\n"
                "        self.assertEqual(translate('pig'), 'igpay')\n"
                "    def test_word_beginning_with_qu(self):\n"
                "        self.assertEqual(translate('queen'), 'eenquay')\n"
                "    def test_word_with_consonant_before_qu(self):\n"
                "        self.assertEqual(translate('square'), 'aresquay')\n"
                "    def test_word_beginning_with_xr(self):\n"
                "        self.assertEqual(translate('xray'), 'xrayay')\n"
                "    def test_y_after_consonant_cluster(self):\n"
                "        self.assertEqual(translate('rhythm'), 'ythmrhay')\n"
                "    def test_phrase(self):\n"
                "        self.assertEqual(translate('quick fast run'), 'ickquay astfay unray')\n"
                ),
            },
            client,
            tool_cls=CountingToolExecutor,
            debate_enabled=False,
            max_tool_rounds=8,
        ) as (root, _client, tools, agent):
            result = agent.handle_user("Implement this Python exercise and run tests.")
            final_text = (root / "pig.py").read_text(encoding="utf-8")

        self.assertTrue(result.completed)
        self.assertIn("mechanical repair applied", result.message)
        self.assertIsNone(tools.execute_counts.get("edit_intent"))
        self.assertIn("move_prefixes", final_text)
        phases = [event.get("phase") for event in agent.events if event.get("type") == "spec_guided_repair"]
        self.assertIn("mechanical_candidate_validation", phases)

    def test_spec_guided_repair_applies_mechanical_word_arithmetic_candidate(self) -> None:
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "wordy.py"}}),
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "wordy_test.py"}}),
                json.dumps({"type": "tool", "name": "edit_intent", "arguments": {"path": "wordy.py", "intent": "replace_body", "target": "answer", "replacement": "return 0"}}),
            ]
        )
        with self._temp_python_agent(
            {
                "wordy.py": "def answer(question):\n    pass\n",
                "wordy_test.py": (
                    "import unittest\nfrom wordy import answer\n\n"
                    "class WordyTest(unittest.TestCase):\n"
                    "    def test_just_a_number(self):\n"
                    "        self.assertEqual(answer('What is 5?'), 5)\n"
                    "    def test_addition(self):\n"
                    "        self.assertEqual(answer('What is 1 plus 1?'), 2)\n"
                    "    def test_subtraction(self):\n"
                    "        self.assertEqual(answer('What is 4 minus -12?'), 16)\n"
                    "    def test_multiplication(self):\n"
                    "        self.assertEqual(answer('What is -3 multiplied by 25?'), -75)\n"
                    "    def test_division(self):\n"
                    "        self.assertEqual(answer('What is 33 divided by -3?'), -11)\n"
                    "    def test_multiple_operations(self):\n"
                    "        self.assertEqual(answer('What is 17 minus 6 plus 3?'), 14)\n"
                    "    def test_unknown_operation(self):\n"
                    "        with self.assertRaises(ValueError) as err:\n"
                    "            answer('What is 52 cubed?')\n"
                    "        self.assertEqual(err.exception.args[0], 'unknown operation')\n"
                    "    def test_syntax_error(self):\n"
                    "        with self.assertRaises(ValueError) as err:\n"
                    "            answer('What is 1 plus?')\n"
                    "        self.assertEqual(err.exception.args[0], 'syntax error')\n"
                    "    def test_empty_question(self):\n"
                    "        with self.assertRaises(ValueError) as err:\n"
                    "            answer('What is?')\n"
                    "        self.assertEqual(err.exception.args[0], 'syntax error')\n"
                    "    def test_non_math_question(self):\n"
                    "        with self.assertRaises(ValueError) as err:\n"
                    "            answer('Who is the President of the United States?')\n"
                    "        self.assertEqual(err.exception.args[0], 'unknown operation')\n"
                ),
            },
            client,
            tool_cls=CountingToolExecutor,
            debate_enabled=False,
            max_tool_rounds=8,
        ) as (root, _client, tools, agent):
            result = agent.handle_user("Implement this Python exercise and run tests.")
            final_text = (root / "wordy.py").read_text(encoding="utf-8")

        self.assertTrue(result.completed)
        self.assertIn("mechanical repair applied", result.message)
        self.assertIsNone(tools.execute_counts.get("edit_intent"))
        self.assertIn("multiplied by", final_text)
        self.assertIn("raise ValueError(unknown_message)", final_text)
        phases = [event.get("phase") for event in agent.events if event.get("type") == "spec_guided_repair"]
        self.assertIn("mechanical_candidate_validation", phases)

    def test_spec_guided_repair_applies_mechanical_text_matrix_transpose_candidate(self) -> None:
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "transpose.py"}}),
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "transpose_test.py"}}),
                json.dumps({"type": "tool", "name": "edit_intent", "arguments": {"path": "transpose.py", "intent": "replace_body", "target": "transpose", "replacement": "return text"}}),
            ]
        )
        with self._temp_python_agent(
            {
                "transpose.py": "def transpose(text):\n    pass\n",
                "transpose_test.py": (
                    "import unittest\nfrom transpose import transpose\n\n"
                    "class TransposeTest(unittest.TestCase):\n"
                    "    def test_two_characters_in_a_row(self):\n"
                    "        self.assertEqual(transpose('A1'), 'A\\n1')\n"
                    "    def test_two_characters_in_a_column(self):\n"
                    "        self.assertEqual(transpose('A\\n1'), 'A1')\n"
                    "    def test_simple(self):\n"
                    "        self.assertEqual(transpose('ABC\\n123'), 'A1\\nB2\\nC3')\n"
                    "    def test_single_line(self):\n"
                    "        self.assertEqual(transpose('A B'), 'A\\n \\nB')\n"
                    "    def test_rectangle(self):\n"
                    "        self.assertEqual(transpose('FRA\\nOUT'), 'FO\\nRU\\nAT')\n"
                    "    def test_jagged_triangle(self):\n"
                    "        self.assertEqual(transpose('11\\n2\\n3333\\n444\\n555555\\n66666'), '123456\\n1 3456\\n  3456\\n  3 56\\n    56\\n    5')\n"
                ),
            },
            client,
            tool_cls=CountingToolExecutor,
            debate_enabled=False,
            max_tool_rounds=8,
        ) as (root, _client, tools, agent):
            result = agent.handle_user("Implement this Python exercise and run tests.")
            final_text = (root / "transpose.py").read_text(encoding="utf-8")

        self.assertTrue(result.completed)
        self.assertIn("mechanical repair applied", result.message)
        self.assertIsNone(tools.execute_counts.get("edit_intent"))
        self.assertIn("for column in range(width)", final_text)
        phases = [event.get("phase") for event in agent.events if event.get("type") == "spec_guided_repair"]
        self.assertIn("mechanical_candidate_validation", phases)

    def test_spec_guided_repair_applies_mechanical_cyclic_scale_candidate(self) -> None:
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "scale_generator.py"}}),
                json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "scale_generator_test.py"}}),
                json.dumps({"type": "tool", "name": "edit_intent", "arguments": {"path": "scale_generator.py", "intent": "replace_body", "target": "Scale.interval", "replacement": "return []"}}),
            ]
        )
        with self._temp_python_agent(
            {
                "scale_generator.py": (
                    "class Scale:\n"
                    "    def __init__(self, tonic):\n"
                    "        pass\n"
                    "    def chromatic(self):\n"
                    "        pass\n"
                    "    def interval(self, intervals):\n"
                    "        pass\n"
                ),
                "scale_generator_test.py": (
                    "import unittest\nfrom scale_generator import Scale\n\n"
                    "class ScaleTest(unittest.TestCase):\n"
                    "    def test_chromatic_scale_with_sharps(self):\n"
                    "        self.assertEqual(Scale('C').chromatic(), ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])\n"
                    "    def test_chromatic_scale_with_flats(self):\n"
                    "        self.assertEqual(Scale('F').chromatic(), ['F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E'])\n"
                    "    def test_simple_major_scale(self):\n"
                    "        self.assertEqual(Scale('C').interval('MMmMMMm'), ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C'])\n"
                    "    def test_major_scale_with_flats(self):\n"
                    "        self.assertEqual(Scale('F').interval('MMmMMMm'), ['F', 'G', 'A', 'Bb', 'C', 'D', 'E', 'F'])\n"
                    "    def test_minor_scale_with_sharps(self):\n"
                    "        self.assertEqual(Scale('f#').interval('MmMMmMM'), ['F#', 'G#', 'A', 'B', 'C#', 'D', 'E', 'F#'])\n"
                    "    def test_enigmatic(self):\n"
                    "        self.assertEqual(Scale('G').interval('mAMMMmm'), ['G', 'G#', 'B', 'C#', 'D#', 'F', 'F#', 'G'])\n"
                ),
            },
            client,
            tool_cls=CountingToolExecutor,
            debate_enabled=False,
            max_tool_rounds=8,
        ) as (root, _client, tools, agent):
            result = agent.handle_user("Implement this Python exercise and run tests.")
            final_text = (root / "scale_generator.py").read_text(encoding="utf-8")

        self.assertTrue(result.completed)
        self.assertIn("mechanical repair applied", result.message)
        self.assertIsNone(tools.execute_counts.get("edit_intent"))
        self.assertIn("STEPS", final_text)
        phases = [event.get("phase") for event in agent.events if event.get("type") == "spec_guided_repair"]
        self.assertIn("mechanical_candidate_validation", phases)

    def test_static_sanity_guard_blocks_tests_after_bad_python_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def make():\n    pass\n", encoding="utf-8")
            (root / "app_test.py").write_text(
                "import unittest\nfrom app import make\n\n"
                "class AppTest(unittest.TestCase):\n"
                "    def test_make(self):\n"
                "        self.assertEqual(make(), 'ok')\n",
                encoding="utf-8",
            )
            command = subprocess.list2cmdline([sys.executable, "-m", "unittest", "discover", "-p", "*_test.py", "-v"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "app.py"}}),
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "app_test.py"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {}}),
                    json.dumps({"type": "tool", "name": "edit_intent", "arguments": {"path": "app.py", "intent": "replace_body", "target": "make", "replacement": "return missing_name"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {}}),
                    json.dumps({"type": "tool", "name": "edit_intent", "arguments": {"path": "app.py", "intent": "replace_body", "target": "make", "replacement": "return 'ok'"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {}}),
                    json.dumps({"type": "final", "message": "fixed"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=10)

            with patch.object(OllamaCodeAgent, "_spec_guided_repair_has_actionable_spec", return_value=False):
                result = agent.handle_user("Implement app.py and run tests.")

        self.assertEqual(result.message, "fixed")
        self.assertEqual(tools.execute_counts.get("run_test"), 2)
        guards = [event for event in agent.events if event.get("guard") == "static-sanity-before-test"]
        self.assertEqual(len(guards), 1)
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("undefined local/global name 'missing_name'", feedback)

    def test_example_probe_guard_blocks_full_tests_after_wrong_return_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "transpose.py").write_text("def transpose(text):\n    pass\n", encoding="utf-8")
            (root / "transpose_test.py").write_text(
                "import unittest\nfrom transpose import transpose\n\n"
                "class TransposeTest(unittest.TestCase):\n"
                "    def test_two_characters_in_a_row(self):\n"
                "        self.assertEqual(transpose('A1'), 'A\\n1')\n",
                encoding="utf-8",
            )
            command = subprocess.list2cmdline([sys.executable, "-m", "unittest", "discover", "-p", "*_test.py", "-v"])
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "transpose.py"}}),
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "transpose_test.py"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {}}),
                    json.dumps({"type": "tool", "name": "edit_intent", "arguments": {"path": "transpose.py", "intent": "replace_body", "target": "transpose", "replacement": "return []"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {}}),
                    json.dumps({"type": "tool", "name": "edit_intent", "arguments": {"path": "transpose.py", "intent": "replace_body", "target": "transpose", "replacement": "return '\\n'.join(text)"}}),
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {}}),
                    json.dumps({"type": "final", "message": "fixed"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=10)

            with patch.object(OllamaCodeAgent, "_spec_guided_repair_has_actionable_spec", return_value=False):
                result = agent.handle_user("Implement transpose.py and run tests.")

        self.assertEqual(result.message, "fixed")
        self.assertEqual(tools.execute_counts.get("run_test"), 2)
        guards = [event for event in agent.events if event.get("guard") == "example-probe-before-test"]
        self.assertEqual(len(guards), 1)
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("transpose('A1') expected 'A\\n1' (str), got [] (list)", feedback)

    def test_agent_normalizes_replace_symbol_text_edit_on_non_code_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs.md").write_text("# Docs\n\nUse total(prices).\n", encoding="utf-8")
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
        self.assertEqual(final_text, "# Docs\n\nUse cart_total(prices).\n")
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

    def test_agent_deterministically_handles_constant_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "flags.py").write_text("OLD_FLAG = 'old'\n\ndef flag_name() -> str:\n    return OLD_FLAG\n", encoding="utf-8")
            command = subprocess.list2cmdline([sys.executable, "-c", "print('OK')"])
            client = FakeClient([])
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Update the OLD_FLAG constant in src/flags.py to 'new' while keeping flag_name() as the public function. Run tests.")
            final_text = (root / "src" / "flags.py").read_text(encoding="utf-8")

        self.assertIn("tests passed", result.message)
        self.assertIn("OLD_FLAG = 'new'", final_text)
        self.assertEqual(tools.execute_counts.get("replace_in_file"), 1)
        self.assertEqual(tools.execute_counts.get("run_test"), 1)
        self.assertEqual(len(client.calls), 0)

    def test_agent_deterministically_handles_optional_parameter_docs_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "docs").mkdir()
            (root / "src" / "api.py").write_text("def fetch_user(user_id: str) -> dict[str, str]:\n    return {'id': user_id}\n", encoding="utf-8")
            (root / "docs" / "api.md").write_text("`fetch_user(user_id)` returns a user dict.\n", encoding="utf-8")
            client = FakeClient([])
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user(
                "Add an optional include_orders: bool = False parameter to fetch_user in src/api.py and update docs/api.md with that parameter. No tests are needed."
            )
            source = (root / "src" / "api.py").read_text(encoding="utf-8")
            docs = (root / "docs" / "api.md").read_text(encoding="utf-8")

        self.assertIn("Updated", result.message)
        self.assertIn("include_orders: bool = False", source)
        self.assertIn("include_orders=False", docs)
        self.assertEqual(tools.execute_counts.get("edit_intent"), 1)
        self.assertEqual(tools.execute_counts.get("replace_in_file"), 1)
        self.assertEqual(len(client.calls), 0)

    def test_require_llm_for_turn_normalizes_initial_search_symbols_for_optional_parameter_docs_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "docs").mkdir()
            (root / "src" / "api.py").write_text("def fetch_user(user_id: str) -> dict[str, str]:\n    return {'id': user_id}\n", encoding="utf-8")
            (root / "docs" / "api.md").write_text("`fetch_user(user_id)` returns a user dict.\n", encoding="utf-8")
            client = FakeClient(
                [
                    json.dumps(
                        {
                            "type": "tool",
                            "name": "search_symbols",
                            "arguments": {"query": "fetch_user", "path": "src/api.py"},
                        }
                    )
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, require_llm_for_turn=True, max_tool_rounds=8)

            result = agent.handle_user(
                "Add an optional include_orders: bool = False parameter to fetch_user in src/api.py and update docs/api.md with that parameter. No tests are needed."
            )
            source = (root / "src" / "api.py").read_text(encoding="utf-8")
            docs = (root / "docs" / "api.md").read_text(encoding="utf-8")

        self.assertIn("Updated", result.message)
        self.assertEqual(len(client.calls), 1)
        self.assertIsNone(tools.execute_counts.get("search_symbols"))
        self.assertEqual(tools.execute_counts.get("edit_intent"), 1)
        self.assertEqual(tools.execute_counts.get("replace_in_file"), 1)
        self.assertIn("include_orders: bool = False", source)
        self.assertIn("include_orders=False", docs)
        tool_names = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        non_context_tool_names = [name for name in tool_names if name != "context_pack"]
        self.assertGreaterEqual(len(non_context_tool_names), 1)
        self.assertEqual(non_context_tool_names[0], "edit_intent")
        normalized = [event for event in agent.events if event.get("type") == "tool_normalized"]
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0].get("original_name"), "search_symbols")
        self.assertEqual(normalized[0].get("normalized_name"), "edit_intent")

    def test_agent_deterministically_repairs_test_grounded_return_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
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
            command = subprocess.list2cmdline([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
            client = FakeClient([])
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user(
                "Fix tests/test_core.py by changing add(left, right) so it returns left + right instead of left - right. Then run tests."
            )
            final_text = (root / "src" / "core.py").read_text(encoding="utf-8")

        self.assertIn("tests passed", result.message)
        self.assertIn("return left + right", final_text)
        self.assertEqual(tools.execute_counts.get("find_implementation_target"), 1)
        self.assertEqual(tools.execute_counts.get("replace_in_file"), 1)
        self.assertEqual(tools.execute_counts.get("run_test"), 1)
        self.assertEqual(len(client.calls), 0)
        tool_names = [event["name"] for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_names, ["find_implementation_target", "replace_in_file", "run_test"])

    def test_agent_preemptively_repairs_string_normalizer_from_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "tests").mkdir()
            (root / "src" / "normalizer.py").write_text("def normalize_words(value: str) -> str:\n    pass\n", encoding="utf-8")
            (root / "tests" / "test_normalizer.py").write_text(
                "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                "from normalizer import normalize_words\nimport unittest\n\n"
                "class NormalizerTests(unittest.TestCase):\n"
                "    def test_spaces(self):\n"
                "        self.assertEqual(normalize_words('Hello Local Model'), 'hello-local-model')\n"
                "    def test_edges(self):\n"
                "        self.assertEqual(normalize_words('  Mixed Case  '), 'mixed-case')\n",
                encoding="utf-8",
            )
            command = subprocess.list2cmdline([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
            client = FakeClient([])
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user(
                "Implement normalize_words in src/normalizer.py from the tests. Read source and tests, replace the stub with complete code so whitespace collapses to single hyphens and output is lowercase, run tests, and summarize briefly."
            )
            final_text = (root / "src" / "normalizer.py").read_text(encoding="utf-8")

        self.assertIn("tests passed", result.message)
        self.assertIn("split()", final_text)
        self.assertEqual(tools.execute_counts.get("write_file"), 1)
        self.assertEqual(tools.execute_counts.get("run_test"), 1)
        self.assertEqual(len(client.calls), 0)

    def test_agent_mechanically_repairs_package_relative_import_after_failed_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src" / "pkg").mkdir(parents=True)
            (root / "tests").mkdir()
            (root / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
            (root / "src" / "pkg" / "helpers.py").write_text("def label(value):\n    return f'[{value}]'\n", encoding="utf-8")
            (root / "src" / "pkg" / "core.py").write_text("from helpers import label\n\ndef wrapped():\n    return label('ok')\n", encoding="utf-8")
            (root / "tests" / "test_pkg.py").write_text(
                "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                "import unittest\nfrom pkg.core import wrapped\n\n"
                "class PackageTests(unittest.TestCase):\n"
                "    def test_wrapped(self):\n"
                "        self.assertEqual(wrapped(), '[ok]')\n",
                encoding="utf-8",
            )
            command = subprocess.list2cmdline([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
            client = FakeClient([])
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Run tests, fix the package import bug in src/pkg/core.py, rerun tests, and summarize.")
            final_text = (root / "src" / "pkg" / "core.py").read_text(encoding="utf-8")

        self.assertIn("tests passed", result.message)
        self.assertIn("from .helpers import label", final_text)
        self.assertEqual(len(client.calls), 0)
        self.assertEqual(tools.execute_counts.get("write_file"), 1)
        self.assertGreaterEqual(tools.execute_counts.get("run_test", 0), 2)

    def test_require_llm_for_turn_normalizes_initial_list_files_for_import_bug_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src" / "pkg").mkdir(parents=True)
            (root / "tests").mkdir()
            (root / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
            (root / "src" / "pkg" / "helpers.py").write_text("def label(value):\n    return f'[{value}]'\n", encoding="utf-8")
            (root / "src" / "pkg" / "core.py").write_text("from helpers import label\n\ndef wrapped():\n    return label('ok')\n", encoding="utf-8")
            (root / "tests" / "test_pkg.py").write_text(
                "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                "import unittest\nfrom pkg.core import wrapped\n\n"
                "class PackageTests(unittest.TestCase):\n"
                "    def test_wrapped(self):\n"
                "        self.assertEqual(wrapped(), '[ok]')\n",
                encoding="utf-8",
            )
            command = subprocess.list2cmdline([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
            client = FakeClient([json.dumps({"type": "tool", "name": "list_files", "arguments": {"path": "."}})])
            tools = CountingToolExecutor(root, approval_mode="auto", test_command=command)
            agent = OllamaCodeAgent(
                client=client,
                tools=tools,
                model="fake-model",
                debate_enabled=False,
                require_llm_for_turn=True,
                max_tool_rounds=8,
            )

            result = agent.handle_user("Run tests, fix the package import bug in src/pkg/core.py, rerun tests, and summarize.")
            final_text = (root / "src" / "pkg" / "core.py").read_text(encoding="utf-8")

        self.assertTrue(result.completed)
        self.assertIn("relative import repair", result.message)
        self.assertIn("from .helpers import label", final_text)
        self.assertEqual(len(client.calls), 1)
        self.assertIsNone(tools.execute_counts.get("list_files"))
        self.assertEqual(tools.execute_counts.get("write_file"), 1)
        tool_names = [event.get("name") for event in agent.events if event.get("type") == "tool_call"]
        non_context_tool_names = [name for name in tool_names if name != "context_pack"]
        self.assertGreaterEqual(len(non_context_tool_names), 1)
        self.assertEqual(non_context_tool_names[0], "run_test")
        normalized = [event for event in agent.events if event.get("type") == "tool_normalized"]
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0].get("original_name"), "list_files")
        self.assertEqual(normalized[0].get("normalized_name"), "run_test")

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
        agent = self._cwd_agent()
        prompt = (
            "Implement this Python Exercism exercise. Read tests and source, edit only implementation files, "
            "do not edit tests, replace stubs with complete code, run tests with configured test command."
        )

        self.assertTrue(agent._request_requires_mutation(prompt))

    def test_request_requires_mutation_for_issue_report_prompt(self) -> None:
        agent = self._cwd_agent()
        prompt = (
            "Modeling's `separability_matrix` does not compute separability correctly for nested CompoundModels. "
            "See `astropy/astropy/modeling/separable.py`. This feels like a bug to me."
        )

        self.assertTrue(agent._request_requires_mutation(prompt))

    def test_request_requires_test_run_when_only_test_edits_are_forbidden(self) -> None:
        agent = self._cwd_agent()
        prompt = (
            "Implement this Python Exercism exercise. Read tests and source, edit only implementation files, "
            "do not edit tests, replace stubs with complete code, run tests with configured test command."
        )

        self.assertTrue(agent._request_requires_test_run(prompt))

    def test_agent_rejects_explanatory_final_for_issue_report_prompt_until_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "app.py"
            target.write_text("def answer():\n    return 0\n", encoding="utf-8")
            client = FakeClient(
                [
                    json.dumps({"type": "final", "message": "This looks like expected behavior, not a bug."}),
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": "def answer():\n    return 1\n"}}),
                    json.dumps({"type": "final", "message": "Updated app.py."}),
                ]
            )
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=6)

            result = agent.handle_user("`app.py` answer() returns the wrong value. This feels like a bug to me.")
            final_text = target.read_text(encoding="utf-8")

        self.assertEqual(result.message, "Updated app.py.")
        self.assertIn("return 1", final_text)
        self.assertTrue(
            any(
                "Do not finish until write_file, replace_symbol, replace_symbols, replace_in_file, or git_commit succeeds" in message["content"]
                for message in agent.messages
                if message["role"] == "user"
            )
        )

    def test_agent_blocks_test_file_edit_for_issue_report_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            source = root / "app.py"
            test_file = root / "tests" / "test_app.py"
            source.write_text("def answer():\n    return 0\n", encoding="utf-8")
            test_file.write_text("from app import answer\n", encoding="utf-8")
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "tests/test_app.py", "content": "from app import answer\n# changed\n"}}),
                    json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "app.py", "content": "def answer():\n    return 1\n"}}),
                    json.dumps({"type": "final", "message": "Updated app.py."}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=6)

            result = agent.handle_user("`app.py` answer() returns the wrong value. This feels like a bug to me.")
            final_source = source.read_text(encoding="utf-8")
            final_test = test_file.read_text(encoding="utf-8")

        self.assertEqual(tools.execute_counts.get("write_file"), 1)
        self.assertIn("return 1", final_source)
        self.assertNotIn("# changed", final_test)
        self.assertTrue(
            any(
                "Do not edit test files unless the user explicitly asks to update tests" in message["content"]
                for message in agent.messages
                if message["role"] == "user"
            )
        )

    def test_test_to_source_bridge_maps_recent_test_evidence_to_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "tests").mkdir()
            (root / "src" / "ops.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            (root / "tests" / "test_ops.py").write_text(
                "from src.ops import add\n\n"
                "def test_add():\n"
                "    assert add(1, 2) == 3\n",
                encoding="utf-8",
            )
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
            successful_tool_results = [
                {
                    "name": "read_file",
                    "arguments": {"path": "tests/test_ops.py"},
                    "result": {"ok": True, "path": "tests/test_ops.py", "output": "from src.ops import add"},
                }
            ]

            bridge = agent._test_to_source_bridge(successful_tool_results)

        self.assertEqual(bridge, ("tests/test_ops.py", "src/ops.py"))

    def test_failed_test_output_paths_guide_spec_repair_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "tests").mkdir()
            (root / "src" / "core.py").write_text("def wrapped():\n    return 1\n", encoding="utf-8")
            (root / "tests" / "test_pkg.py").write_text(
                "from pkg import wrapped\n\n"
                "def test_wrapped():\n"
                "    assert wrapped() == 2\n",
                encoding="utf-8",
            )
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
            failure_output = (
                "ImportError: Failed to import test module: test_pkg\n"
                '  File "tests/test_pkg.py", line 4, in <module>\n'
                "    from pkg import wrapped\n"
                '  File "src/core.py", line 1, in <module>\n'
                "    from helpers import label\n"
            )
            successful_tool_results = [
                {"name": "read_file", "result": {"ok": True, "path": "src/ops.py", "output": "old"}},
                {"name": "run_test", "result": {"ok": False, "output": failure_output}},
            ]
            paths = agent._spec_guided_repair_paths(successful_tool_results)
            hint = agent._failed_test_edit_target_hint(successful_tool_results)

        self.assertEqual(paths, ("src/core.py", "tests/test_pkg.py"))
        self.assertEqual(hint, "src/core.py")

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
                    json.dumps({"strategy": "normal_loop", "reason": "exercise the generic write-file guard path"}),
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
                with patch.object(agent, "_try_structured_test_driven_repair", return_value=None):
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
        agent = self._cwd_agent(client, debate_enabled=False)
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
        agent = self._cwd_agent(client)
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

    def test_agent_synthesizes_plain_language_symbol_return_without_model_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("def meaning() -> int:\n    return 42\n", encoding="utf-8")
            client = FakeClient(['{"type":"final","message":"wrong"}'])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")

            result = agent.handle_user("What does function meaning in src/app.py return?")

        self.assertEqual(result.message, "meaning returns 42.")
        self.assertEqual(len(client.calls), 0)
        tool_names = [event["name"] for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_names, ["read_symbol"])
        self.assertTrue(any(event["type"] == "assistant_synthesized" for event in agent.events))

    def test_agent_deterministically_finds_implementation_file_without_model_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "tests").mkdir()
            (root / "src" / "core.py").write_text("def add(left, right):\n    return left + right\n", encoding="utf-8")
            (root / "tests" / "test_core.py").write_text(
                "import unittest\nfrom src.core import add\n\n"
                "class CoreTests(unittest.TestCase):\n"
                "    def test_add(self):\n"
                "        self.assertEqual(add(1, 2), 3)\n",
                encoding="utf-8",
            )
            client = FakeClient(['{"type":"final","message":"wrong"}'])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Which implementation file corresponds to tests/test_core.py?")

        self.assertEqual(result.message, "Relevant implementation file: src/core.py.")
        self.assertEqual(len(client.calls), 0)
        tool_names = [event["name"] for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_names, ["find_implementation_target"])
        self.assertTrue(any(event["type"] == "assistant_synthesized" for event in agent.events))

    def test_agent_deterministically_handles_explicit_find_implementation_target_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "tests").mkdir()
            (root / "src" / "core.py").write_text("def add(left, right):\n    return left + right\n", encoding="utf-8")
            (root / "tests" / "test_core.py").write_text(
                "import unittest\nfrom src.core import add\n\n"
                "class CoreTests(unittest.TestCase):\n"
                "    def test_add(self):\n"
                "        self.assertEqual(add(1, 2), 3)\n",
                encoding="utf-8",
            )
            client = FakeClient(['{"type":"final","message":"wrong"}'])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Use find_implementation_target for tests/test_core.py and reply with the implementation file only.")

        self.assertEqual(result.message, "src/core.py")
        self.assertEqual(len(client.calls), 0)
        tool_names = [event["name"] for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_names, ["find_implementation_target"])
        self.assertTrue(any(event["type"] == "assistant_synthesized" for event in agent.events))

    def test_agent_synthesizes_search_symbols_name_only_without_model_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "session_task.py").write_text("def session_value() -> str:\n    return 'todo'\n", encoding="utf-8")
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Use search_symbols to find session_value in src/session_task.py. Reply with the function name only.")

        self.assertEqual(result.message, "session_value")
        self.assertEqual(len(client.calls), 0)
        tool_names = [event["name"] for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_names, ["search_symbols"])

    def test_agent_synthesizes_code_outline_summary_without_model_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("def meaning() -> int:\n    return 42\n", encoding="utf-8")
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Use code_outline on src/app.py and tell me which function is defined there.")

        self.assertEqual(result.message, "The function defined in src/app.py is meaning.")
        self.assertEqual(len(client.calls), 0)
        tool_names = [event["name"] for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_names, ["code_outline"])

    def test_agent_deterministically_updates_js_return_after_symbol_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "math.js").write_text("export function double(n) {\n  return n + n;\n}\n", encoding="utf-8")
            client = FakeClient([])
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Use search_symbols and read_symbol on src/math.js, then change double(n) so it returns n * 2 instead of n + n. Do not use read_file.")
            final_text = (root / "src" / "math.js").read_text(encoding="utf-8")

        self.assertIn("Updated", result.message)
        self.assertIn("return n * 2;", final_text)
        self.assertEqual(len(client.calls), 0)
        tool_names = [event["name"] for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_names[:3], ["search_symbols", "read_symbol", "replace_in_file"])
        self.assertEqual(tool_names[3:], ["lint_typecheck", "select_tests", "discover_validators"])
