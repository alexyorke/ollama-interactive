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
