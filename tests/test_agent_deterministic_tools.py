import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from ollama_code.agent import OllamaCodeAgent
from ollama_code.tools import ToolExecutor
from tests.agent_test_support import AgentTestBase, CountingToolExecutor, FakeClient


class AgentDeterministicToolTests(AgentTestBase):
    def _cwd_agent(self, client: FakeClient | None = None, **kwargs: object) -> OllamaCodeAgent:
        resolved_client = client if client is not None else FakeClient([])
        return OllamaCodeAgent(
            client=resolved_client,
            tools=ToolExecutor(self._workspace_scratch(), approval_mode="auto"),
            model="fake-model",
            **kwargs,
        )

    def _assert_repo_tool_then_git_status_without_model_loop(
        self,
        request: str,
        *,
        first_tool_name: str,
        expected_message_fragment: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_git_repo_or_skip(root)
            (root / "docs").mkdir()
            (root / "src").mkdir()
            (root / "docs" / "guide.md").write_text("TOKEN_42 lives here.\n", encoding="utf-8")
            (root / "src" / "app.py").write_text("def answer() -> int:\n    return 42\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=root, capture_output=True, text=True, check=True)
            subprocess.run(["git", "commit", "-m", "initial"], cwd=root, capture_output=True, text=True, check=True)
            (root / "src" / "app.py").write_text("def answer() -> int:\n    return 99\n", encoding="utf-8")
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
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
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            if create_docs_fixture:
                (root / "docs").mkdir()
                (root / "docs" / "guide.md").write_text("TOKEN_42 lives here.\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "test_sample.py").write_text(
                test_file_content
                or "import unittest\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n",
                encoding="utf-8",
            )
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
            result = agent.handle_user(request)

        for fragment in expected_message_fragments:
            self.assertIn(fragment, result.message)
        if acceptable_follow_up_fragments is not None:
            self.assertTrue(any(fragment in result.message for fragment in acceptable_follow_up_fragments))
        self.assertEqual(len(client.calls), 0)
        tool_names = [event["name"] for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_names, expected_tool_names)

    def test_agent_deterministically_handles_git_diff_without_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_git_repo_or_skip(root)
            (root / "app.py").write_text("def value():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "app.py"], cwd=root, capture_output=True, text=True, check=True)
            subprocess.run(["git", "commit", "-m", "base"], cwd=root, capture_output=True, text=True, check=True)
            (root / "app.py").write_text("def value():\n    return 2\n", encoding="utf-8")
            client = FakeClient(['{"type":"final","message":"model diff"}'])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Show git diff app.py.")

        self.assertIn("+    return 2", result.message)
        self.assertEqual(len(client.calls), 0)
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_calls[0]["name"], "git_diff")

    def test_agent_require_llm_for_turn_bypasses_deterministic_git_diff_shortcut(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_git_repo_or_skip(root)
            (root / "app.py").write_text("def value():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "app.py"], cwd=root, capture_output=True, text=True, check=True)
            subprocess.run(["git", "commit", "-m", "base"], cwd=root, capture_output=True, text=True, check=True)
            (root / "app.py").write_text("def value():\n    return 2\n", encoding="utf-8")
            client = FakeClient(['{"type":"tool","name":"git_diff","arguments":{"path":"app.py"}}'])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(
                client=client,
                tools=tools,
                model="fake-model",
                debate_enabled=False,
                require_llm_for_turn=True,
            )

            result = agent.handle_user("Show git diff app.py.")

        self.assertIn("+    return 2", result.message)
        self.assertEqual(len(client.calls), 1)
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_calls[0]["name"], "git_diff")

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

    def test_agent_retries_after_unknown_tool_does_not_count_as_real_tool_use(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"bogus_tool","arguments":{}}',
                '{"type":"tool","name":"list_files","arguments":{}}',
                '{"type":"final","message":"listed workspace"}',
            ]
        )
        agent = self._cwd_agent(client, debate_enabled=False)

        result = agent.handle_user("Inspect the workspace with a tool.")

        self.assertEqual(result.message, "listed workspace")
        tool_results = [event for event in agent.events if event["type"] == "tool_result"]
        self.assertEqual(tool_results[0]["name"], "bogus_tool")
        self.assertEqual(tool_results[0]["result"]["summary"], "Unknown tool: bogus_tool")
        self.assertEqual(tool_results[1]["name"], "list_files")

    def test_agent_rejects_forbidden_tool_and_retries(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"read_file","arguments":{"path":"README.md"}}',
                '{"type":"tool","name":"list_files","arguments":{}}',
                '{"type":"final","message":"listed workspace"}',
            ]
        )
        agent = self._cwd_agent(client, debate_enabled=False)

        result = agent.handle_user("List files in the workspace. Do not use read_file.")

        self.assertTrue(result.message)
        self.assertEqual(len(client.calls), 0)
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
        agent = self._cwd_agent(client, debate_enabled=False)
        agent.messages.append({"role": "user", "content": "Remember the exact token CONTINUE_TOKEN_99 for this session."})

        result = agent.handle_user("What token did I ask you to remember earlier in this session? Reply with the token only.")

        self.assertEqual(result.message, "CONTINUE_TOKEN_99")
        self.assertFalse(any(event["type"] == "tool_call" for event in agent.events))

    def test_agent_skips_verification_for_session_memory_question(self) -> None:
        client = FakeClient(['{"type":"final","message":"CONTINUE_TOKEN_99"}'], script_verification=True)
        agent = self._cwd_agent(client)
        agent.messages.append({"role": "user", "content": "Remember the exact token CONTINUE_TOKEN_99 for this session."})

        result = agent.handle_user("What token did I ask you to remember earlier in this session? Reply with the token only.")

        self.assertEqual(result.message, "CONTINUE_TOKEN_99")
        self.assertEqual(len(client.calls), 1)
        self.assertFalse(any(event["type"] == "verification" for event in agent.events))

    def test_agent_synthesizes_todo_statuses_after_todo_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            client = FakeClient(
                [
                    '{"type":"tool","name":"todo_write","arguments":{"items":[{"content":"inspect","status":"completed"},{"content":"report","status":"pending"}]}}',
                    '{"type":"tool","name":"todo_read","arguments":{}}',
                ]
            )
            tools = ToolExecutor(root, approval_mode="read-only")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user(
                "Use todo_write to create a todo list with one completed item named inspect and one pending item named report. Then use todo_read and reply with the todo statuses only."
            )

        self.assertIn("[completed] inspect", result.message)
        self.assertIn("[pending] report", result.message)
        self.assertTrue(any(event["type"] == "assistant_synthesized" and event.get("tool") == "todo_read" for event in agent.events))

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

    def test_agent_does_not_treat_do_not_modify_as_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "docs" / "spec.md").write_text("MAGIC_TOKEN appears here.\n", encoding="utf-8")
            client = FakeClient(['{"type":"tool","name":"read_file","arguments":{"path":"docs/spec.md"}}'])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user(
                "Do not modify any files. Read docs/spec.md, and report the exact uppercase token that already exists in the file. Then reply with that token only."
            )

        self.assertEqual(result.message, "MAGIC_TOKEN")
        self.assertEqual(len(client.calls), 1)
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

    def test_agent_synthesizes_exact_lowercase_line_after_read_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notes").mkdir()
            (root / "notes" / "alpha.txt").write_text("ORBIT\nsecond line\n", encoding="utf-8")
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Use read_file on notes/alpha.txt and reply with exactly the text on line 2.")

        self.assertEqual(result.message, "second line")
        self.assertEqual(len(client.calls), 0)

    def test_agent_deterministically_reads_single_file_contents_without_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.txt").write_text("hello world\n", encoding="utf-8")
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("What does note.txt say?")

        self.assertEqual(result.message, "hello world")
        self.assertEqual(len(client.calls), 0)
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual([event["name"] for event in tool_calls], ["read_file"])

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

    def test_agent_synthesizes_exact_shell_output_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
            exact_command = subprocess.list2cmdline(
                [
                    sys.executable,
                    "-c",
                    "import os; print(os.getcwd()); print(6*7)",
                ]
            )

            result = agent.handle_user(
                f"Use run_shell to execute exactly: {exact_command}. Then tell me the number and the directory."
            )

        self.assertIn("42", result.message)
        self.assertIn(str(root), result.message)
        self.assertEqual(len(client.calls), 0)

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
            (root / "notes").mkdir()
            (root / "notes" / "repl.txt").write_text("repl ok\n", encoding="utf-8")
            client = FakeClient(['{"type":"tool","name":"search","arguments":{"query":"repl ok","path":"."}}'])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("Use search to find repl ok and tell me which file contains it.")

        self.assertEqual(result.message, "notes/repl.txt contains the match.")
        self.assertEqual(len(client.calls), 0)
        self.assertTrue(any(event["type"] == "assistant_synthesized" for event in agent.events))

    def test_agent_synthesizes_discover_validators_for_natural_phrase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "tests" / "test_sample.py").write_text(
                "import unittest\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n",
                encoding="utf-8",
            )
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)

            result = agent.handle_user("List the test and validation commands for this repo.")

        self.assertIn("test python:", result.message)
        self.assertEqual(len(client.calls), 0)
        tool_names = [event["name"] for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_names, ["discover_validators"])

    def test_agent_chains_search_then_run_test_without_model_loop(self) -> None:
        self._assert_follow_up_tool_chain_without_model_loop(
            "Search for TOKEN_42 in the repo, then run tests and tell me whether tests passed.",
            expected_tool_names=["search", "run_test"],
            expected_message_fragments=["docs/guide.md contains the match.", "Tests passed: yes"],
            create_docs_fixture=True,
            test_file_content="import unittest\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        self.assertEqual(6 * 7, 42)\n",
        )

    def test_agent_chains_discover_validators_then_run_test_without_model_loop(self) -> None:
        self._assert_follow_up_tool_chain_without_model_loop(
            "Discover the test commands for this repo, then run tests and tell me whether they passed.",
            expected_tool_names=["discover_validators", "run_test"],
            expected_message_fragments=["test python:", "Tests passed: yes"],
        )

    def test_agent_chains_search_then_git_status_without_model_loop(self) -> None:
        self._assert_repo_tool_then_git_status_without_model_loop(
            "Search for TOKEN_42 in the repo, then show git status.",
            first_tool_name="search",
            expected_message_fragment="docs/guide.md contains the match.",
        )

    def test_agent_chains_search_and_git_status_without_then(self) -> None:
        self._assert_repo_tool_then_git_status_without_model_loop(
            "Search for TOKEN_42 in the repo and show git status.",
            first_tool_name="search",
            expected_message_fragment="docs/guide.md contains the match.",
        )

    def test_agent_chains_search_after_that_git_status_without_model_loop(self) -> None:
        self._assert_repo_tool_then_git_status_without_model_loop(
            "Search for TOKEN_42 in the repo, after that show git status.",
            first_tool_name="search",
            expected_message_fragment="docs/guide.md contains the match.",
        )

    def test_agent_chains_list_files_and_git_status_without_model_loop(self) -> None:
        self._assert_repo_tool_then_git_status_without_model_loop(
            "List files in the workspace and show git status.",
            first_tool_name="list_files",
            expected_message_fragment="docs/guide.md",
        )

    def test_agent_handles_literal_list_files_tool_request_without_model_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "notes").mkdir()
            (root / "src").mkdir()
            (root / "src" / "sample.py").write_text("def answer() -> int:\n    return 42\n", encoding="utf-8")
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
            result = agent.handle_user("Use list_files on . with a high enough limit to inspect the workspace. Reply with docs, notes, and src only.")

        self.assertIn("docs", result.message)
        self.assertIn("notes", result.message)
        self.assertIn("src", result.message)
        self.assertEqual(len(client.calls), 0)
        tool_names = [event["name"] for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual(tool_names, ["list_files"])

    def test_agent_chains_discover_validators_then_lint_without_model_loop(self) -> None:
        self._assert_follow_up_tool_chain_without_model_loop(
            "List the test and validation commands for this repo, then run lint.",
            expected_tool_names=["discover_validators", "lint_typecheck"],
            expected_message_fragments=["lint python:"],
            acceptable_follow_up_fragments=["All checks passed!", "Lint/typecheck passed.", "syntax ok:"],
        )

    def test_agent_chains_discover_validators_and_lint_without_then(self) -> None:
        self._assert_follow_up_tool_chain_without_model_loop(
            "List the test and validation commands for this repo and run lint.",
            expected_tool_names=["discover_validators", "lint_typecheck"],
            expected_message_fragments=["lint python:"],
            acceptable_follow_up_fragments=["All checks passed!", "Lint/typecheck passed.", "syntax ok:"],
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
        with (
            patch.object(tools, "mcp_list_tools", return_value={"ok": True, "tool": "mcp_list_tools", "output": "mcp.demo.echo: Echo"}),
            patch.object(
                tools,
                "mcp_call",
                return_value={"ok": True, "tool": "mcp_call", "server": "demo", "mcp_tool": "echo", "output": '{"value":"ok"}'},
            ) as mcp_call,
        ):
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
        with (
            patch.object(tools, "mcp_list_tools", return_value={"ok": True, "tool": "mcp_list_tools", "output": "mcp.demo.echo: Echo"}),
            patch.object(
                tools,
                "mcp_call",
                return_value={"ok": True, "tool": "mcp_call", "server": "demo", "mcp_tool": "echo", "output": '{"value":"ok"}'},
            ) as mcp_call,
        ):
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
            result = agent.handle_user("Use mcp.demo.echo before answering about note.txt.")

        self.assertEqual(result.message, "done with mcp")
        mcp_call.assert_called_once_with("demo", "echo", {"text": "hi"})
        tool_calls = [event for event in agent.events if event["type"] == "tool_call"]
        self.assertEqual([event["name"] for event in tool_calls], ["read_file", "mcp.demo.echo"])

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
