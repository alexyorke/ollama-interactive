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
