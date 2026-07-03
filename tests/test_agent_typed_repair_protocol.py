from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ollama_code.agent import OllamaCodeAgent
from tests.test_agent import CountingToolExecutor, FakeClient


class AgentTypedRepairProtocolTests(unittest.TestCase):
    def test_cli_bundle_emits_typed_state_and_blocks_premature_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task_cli.py").write_text("def main() -> int:\n    return 0\n", encoding="utf-8")
            (root / "README.md").write_text("# Task CLI\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "test_task_cli.py").write_text("import unittest\n", encoding="utf-8")
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "run_test", "arguments": {"command": "python -m unittest discover -s tests"}}),
                    json.dumps({"type": "final", "message": "done"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto", test_command="python -m unittest discover -s tests")
            agent = OllamaCodeAgent(
                client=client,
                tools=tools,
                model="fake-model",
                debate_enabled=False,
                disable_spec_guided_repair=True,
                require_llm_for_turn=True,
                max_tool_rounds=2,
            )

            result = agent.handle_user(
                "Add a --due-before option to task_cli.py, update tests and README, run tests, and prove it with a shell command."
            )

        self.assertFalse(result.completed)
        self.assertIsNone(tools.execute_counts.get("run_test"))
        event_types = [event.get("type") for event in agent.events]
        self.assertIn("task_state", event_types)
        self.assertIn("patch_plan", event_types)
        self.assertIn("allowed_next_actions", event_types)
        repair_decisions = [event for event in agent.events if event.get("type") == "repair_decision"]
        self.assertTrue(any(event.get("tool") == "run_test" and event.get("allowed") is False for event in repair_decisions))
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Typed repair protocol", feedback)
        self.assertIn("grounded implementation target", feedback)

    def test_cli_bundle_instruction_after_grounding_requires_all_surfaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task_cli.py").write_text("def main() -> int:\n    return 0\n", encoding="utf-8")
            (root / "README.md").write_text("# Task CLI\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "test_task_cli.py").write_text("import unittest\n", encoding="utf-8")
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "read_file", "arguments": {"path": "task_cli.py"}}),
                    json.dumps({"type": "final", "message": "implemented"}),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto", test_command="python -m unittest discover -s tests")
            agent = OllamaCodeAgent(
                client=client,
                tools=tools,
                model="fake-model",
                debate_enabled=False,
                disable_spec_guided_repair=True,
                require_llm_for_turn=True,
                max_tool_rounds=2,
            )

            result = agent.handle_user(
                "Add a --due-before option to task_cli.py, update tests and README, run tests, and prove it with a shell command."
            )

        self.assertFalse(result.completed)
        self.assertEqual(tools.execute_counts.get("read_file"), 1)
        patch_events = [event for event in agent.events if event.get("type") == "patch_plan"]
        self.assertTrue(any(event.get("complete") is True for event in patch_events))
        final_decisions = [event for event in agent.events if event.get("type") == "repair_decision" and event.get("tool") == "final"]
        self.assertTrue(any(event.get("allowed") is False for event in final_decisions))
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("parser change", feedback)
        self.assertIn("callable behavior", feedback)
        self.assertIn("tests", feedback)
        self.assertIn("docs", feedback)
        self.assertIn("direct CLI behavior proof", feedback)

    def test_cli_bundle_blocks_body_only_edit_intent_for_flag_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task_cli.py").write_text(
                "def list_tasks(priority=None):\n"
                "    return []\n",
                encoding="utf-8",
            )
            client = FakeClient(
                [
                    json.dumps({"type": "tool", "name": "read_symbol", "arguments": {"path": "task_cli.py", "symbol": "list_tasks"}}),
                    json.dumps(
                        {
                            "type": "tool",
                            "name": "edit_intent",
                            "arguments": {
                                "path": "task_cli.py",
                                "intent": "replace_body",
                                "target": "list_tasks",
                                "replacement": "if due_before:\n    return []\nreturn []",
                            },
                        }
                    ),
                ]
            )
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(
                client=client,
                tools=tools,
                model="fake-model",
                debate_enabled=False,
                disable_spec_guided_repair=True,
                require_llm_for_turn=True,
                max_tool_rounds=2,
            )

            result = agent.handle_user("Add a --due-before option to task_cli.py and prove it with a shell command.")

        self.assertFalse(result.completed)
        self.assertIsNone(tools.execute_counts.get("edit_intent"))
        decisions = [event for event in agent.events if event.get("type") == "repair_decision"]
        self.assertTrue(any(event.get("tool") == "edit_intent" and event.get("violation") == "narrow_cli_mutation" for event in decisions))
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("full command-surface bundle", feedback)

    def test_cli_flag_bundle_skips_legacy_mechanical_repair_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task_cli.py").write_text("def list_tasks(priority=None):\n    return []\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "test_task_cli.py").write_text("import unittest\n", encoding="utf-8")
            client = FakeClient([])
            tools = CountingToolExecutor(root, approval_mode="auto", test_command="python -m unittest discover -s tests")
            agent = OllamaCodeAgent(
                client=client,
                tools=tools,
                model="fake-model",
                debate_enabled=False,
            )
            obligations = agent._derive_request_obligations(
                request_text="Add a --due-before option to task_cli.py, update tests, run tests, and prove it with a shell command.",
                required_tool_names=set(),
                required_mutation_paths=set(),
                code_mutation_required=True,
                test_run_required=True,
            )

            result = agent._try_post_context_cli_feature_repair(
                request_text="Add a --due-before option to task_cli.py, update tests, run tests, and prove it with a shell command.",
                round_number=1,
                request_obligations=obligations,
                forbidden_tool_names=set(),
                successful_tool_results=[],
                satisfied_tool_names=set(),
                tool_calls_this_turn=[],
            )

        self.assertIsNone(result)
        spec_events = [event for event in agent.events if event.get("type") == "spec_guided_repair"]
        self.assertTrue(
            any(event.get("phase") == "post_context_cli_mechanical_skipped" for event in spec_events)
        )
        self.assertIsNone(tools.execute_counts.get("implementation_spec"))

    def test_cli_flag_bundle_skips_preemptive_mechanical_repair_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task_cli.py").write_text("def list_tasks(priority=None):\n    return []\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "test_task_cli.py").write_text("import unittest\n", encoding="utf-8")
            tools = CountingToolExecutor(root, approval_mode="auto", test_command="python -m unittest discover -s tests")
            agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)

            result = agent._try_preemptive_mechanical_spec_guided_repair(
                request_text="Add a --due-before option to task_cli.py and run tests.",
                forbidden_tool_names=set(),
                successful_tool_results=[],
                satisfied_tool_names=set(),
                tool_calls_this_turn=[],
                required_mutation_paths=set(),
            )

        self.assertIsNone(result)
        spec_events = [event for event in agent.events if event.get("type") == "spec_guided_repair"]
        self.assertTrue(any(event.get("phase") == "preemptive_mechanical_skipped" for event in spec_events))
        self.assertIsNone(tools.execute_counts.get("implementation_spec"))

    def test_cli_flag_bundle_skips_shared_mechanical_repair_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task_cli.py").write_text("def list_tasks(priority=None):\n    return []\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "test_task_cli.py").write_text("import unittest\n", encoding="utf-8")
            tools = CountingToolExecutor(root, approval_mode="auto", test_command="python -m unittest discover -s tests")
            agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)

            result = agent._try_spec_guided_mechanical_repair(
                request_text="Add a --due-before option to task_cli.py and run tests.",
                round_number=1,
                source_path="task_cli.py",
                test_path="tests/test_task_cli.py",
                test_command="python -m unittest discover -s tests",
                successful_tool_results=[],
                satisfied_tool_names=set(),
                tool_calls_this_turn=[],
            )

        self.assertIsNone(result)
        spec_events = [event for event in agent.events if event.get("type") == "spec_guided_repair"]
        self.assertTrue(any(event.get("phase") == "mechanical_repair_skipped" for event in spec_events))

    def test_cli_flag_bundle_skips_structured_and_spec_guided_repair_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task_cli.py").write_text("def list_tasks(priority=None):\n    return []\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "test_task_cli.py").write_text("import unittest\n", encoding="utf-8")
            tools = CountingToolExecutor(root, approval_mode="auto", test_command="python -m unittest discover -s tests")
            agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)
            request = "Add a --due-before option to task_cli.py, update tests, run tests, and prove it with a shell command."

            structured = agent._try_structured_test_driven_repair(
                request_text=request,
                session_memory_request=False,
                mutation_required=True,
                test_run_required=True,
                required_tool_names=set(),
                forbidden_tool_names=set(),
                successful_tool_results=[],
                satisfied_tool_names=set(),
                tool_calls_this_turn=[],
            )
            spec = agent._try_spec_guided_repair(
                request_text=request,
                round_number=1,
                failed_run_test_result={"ok": False, "tool": "run_test"},
                run_test_arguments={"command": "python -m unittest discover -s tests"},
                successful_tool_results=[],
                satisfied_tool_names=set(),
                tool_calls_this_turn=[],
                forced_paths=("task_cli.py", "tests/test_task_cli.py"),
            )

        self.assertIsNone(structured)
        self.assertIsNone(spec)
        spec_events = [event for event in agent.events if event.get("type") == "spec_guided_repair"]
        self.assertTrue(any(event.get("phase") == "structured_test_driven_skipped" for event in spec_events))
        self.assertTrue(any(event.get("phase") == "spec_guided_repair_skipped" for event in spec_events))

    def test_controller_tool_wrapper_rejects_due_before_narrow_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task_cli.py").write_text("def list_tasks(priority=None):\n    return []\n", encoding="utf-8")
            tools = CountingToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)
            successful: list[dict[str, object]] = []
            satisfied: set[str] = set()
            calls: list[dict[str, object]] = []

            result = agent._execute_controller_tool(
                name="edit_intent",
                arguments={"path": "task_cli.py", "intent": "replace_body", "target": "list_tasks", "replacement": "return []"},
                request_text="Add a --due-before option to task_cli.py.",
                round_number=0,
                successful_tool_results=successful,
                satisfied_tool_names=satisfied,
                tool_calls_this_turn=calls,
            )

        self.assertFalse(result["ok"])
        self.assertIsNone(tools.execute_counts.get("edit_intent"))
        self.assertIn("Typed repair protocol rejected", str(result.get("summary")))


if __name__ == "__main__":
    unittest.main()
