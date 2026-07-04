import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from ollama_code.features import ENV_OLLAMA_CODE_FEATURE_PROFILE
from ollama_code.agent import GROUNDING_EVIDENCE_TOOL_NAMES, OllamaCodeAgent
from ollama_code.tools import ToolExecutor
from tests.agent_test_support import AgentTestBase, FakeClient


class AgentPromptPolicyTests(AgentTestBase):
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
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model")
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

    def _cwd_agent(self) -> OllamaCodeAgent:
        return OllamaCodeAgent(
            client=FakeClient([]),
            tools=ToolExecutor(self._workspace_scratch(), approval_mode="auto"),
            model="fake-model",
        )

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
        resolved_client = client if client is not None else FakeClient([])
        tools = tool_cls(root, approval_mode=approval_mode, **(tool_kwargs or {}))
        return root, resolved_client, tools, OllamaCodeAgent(client=resolved_client, tools=tools, model="fake-model", **kwargs)

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
        selected = self._primary_tools_for_request(
            "Profile the slow edit pipeline and debug the controller design.",
        )
        self.assertIn("systems_lens", selected)
        self.assertNotIn("systems_lens", GROUNDING_EVIDENCE_TOOL_NAMES)

    def test_primary_tools_keep_specialty_tools_out_of_generic_code_prompt(self) -> None:
        selected = self._primary_tools_for_request(
            "Find the implementation of calculate_discount in this repo.",
        )

        self.assertIn("repo_index_search", selected)
        self.assertIn("read_symbol", selected)
        self.assertNotIn("ast_search", selected)
        self.assertNotIn("semgrep_scan", selected)
        self.assertNotIn("lsp_references", selected)
        self.assertNotIn("mcp_list_tools", selected)
        self.assertNotIn("fts_refresh", selected)
        self.assertNotIn("verified_function_search", selected)

    def test_primary_tools_add_specialty_tools_by_intent(self) -> None:
        selected = self._primary_tools_for_request(
            "Use LSP references and ast-grep structural search to inspect this symbol.",
        )

        self.assertIn("lsp_references", selected)
        self.assertIn("lsp_definition", selected)
        self.assertIn("ast_search", selected)
        self.assertIn("semgrep_scan", selected)

    def test_primary_tools_add_python_sdk_search_by_intent(self) -> None:
        selected = self._primary_tools_for_request(
            "Find the current Python stdlib API for parsing JSON strings.",
        )

        self.assertIn("python_sdk_search", selected)
        self.assertIn("inspect_library_source", selected)

    def test_primary_tools_add_verified_function_tools_by_intent(self) -> None:
        selected = self._primary_tools_for_request(
            "Find a verified reusable function card and compose it; do not invent code.",
        )

        self.assertIn("verified_function_search", selected)
        self.assertIn("verified_function_show", selected)
        self.assertIn("compose_verified_functions", selected)
        self.assertIn("promote_verified_function", selected)

    def test_git_recovery_request_requires_tools_and_git_surface(self) -> None:
        request = "I checked out master and can't find my changes. Help me merge them back."
        selected = self._primary_tools_for_request(
            request,
        )

        self.assertTrue(self._cwd_agent()._request_requires_tools(request))
        self.assertIn("git_status", selected)
        self.assertIn("git_diff", selected)

    def test_primary_tools_include_todos_for_complex_work(self) -> None:
        selected = self._primary_tools_for_request(
            "Implement todo list support, update tests, and run the suite.",
            mutation_allowed=True,
            mutation_required=True,
            test_run_required=True,
        )

        self.assertIn("todo_read", selected)
        self.assertIn("todo_write", selected)

    def test_primary_context_truncates_old_messages_before_recent_limit(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient([])
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model")
        current_request = "Summarize current request with TOKEN_CURRENT_END."
        agent.messages.extend(
            [
                {"role": "user", "content": "old " + ("x" * 5000) + " TOKEN_OLD_END"},
                {"role": "user", "content": current_request},
            ]
        )

        messages = agent._primary_messages_for_model(
            session_memory_request=False,
            current_request=current_request,
            tool_names={"read_file"},
        )

        old_message = next(message["content"] for message in messages if message["role"] == "user" and message["content"].startswith("old "))
        current_message = next(message["content"] for message in messages if "TOKEN_CURRENT_END" in message["content"])
        self.assertIn("... truncated ...", old_message)
        self.assertNotIn("TOKEN_OLD_END", old_message)
        self.assertEqual(current_message, current_request)

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
        self.assertNotIn("todo_write(items", system_prompt)
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
