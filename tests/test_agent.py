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

    # Focused loop-cap and broad-context planner coverage lives in test_agent_failure_compression.py.

    # Focused identifier-search and context-pack grounding coverage lives in test_agent_grounding_path_repair.py.

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
