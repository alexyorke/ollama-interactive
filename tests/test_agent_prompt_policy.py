import tempfile
from pathlib import Path

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
