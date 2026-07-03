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

    def test_cli_flag_bundle_legacy_mechanical_repair_path_allows_typed_synthesis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task_cli.py").write_text(
                "from __future__ import annotations\n\n"
                "import argparse\n\n"
                "TASKS = [\n"
                "    {'title': 'write-docs', 'status': 'todo', 'priority': 'high', 'due': '2026-07-01'},\n"
                "    {'title': 'ship-cli', 'status': 'done', 'priority': 'low', 'due': '2026-07-05'},\n"
                "    {'title': 'fix-bug', 'status': 'todo', 'priority': 'medium', 'due': '2026-07-10'},\n"
                "]\n\n"
                "def list_tasks(priority: str | None = None) -> list[str]:\n"
                "    tasks = TASKS if priority is None else [task for task in TASKS if task['priority'] == priority]\n"
                "    return [f\"{task['title']}:{task['status']}:{task['priority']}:{task['due']}\" for task in tasks]\n\n"
                "def main(argv: list[str] | None = None) -> int:\n"
                "    parser = argparse.ArgumentParser()\n"
                "    subparsers = parser.add_subparsers(dest='command', required=True)\n"
                "    list_parser = subparsers.add_parser('list')\n"
                "    list_parser.add_argument('--priority')\n"
                "    args = parser.parse_args(argv)\n"
                "    if args.command == 'list':\n"
                "        print('\\n'.join(list_tasks(args.priority)))\n"
                "        return 0\n"
                "    return 1\n",
                encoding="utf-8",
            )
            (root / "tests").mkdir()
            (root / "README.md").write_text("# Task CLI\n", encoding="utf-8")
            (root / "tests" / "test_task_cli.py").write_text(
                "import subprocess\nimport sys\nimport unittest\nfrom pathlib import Path\n\n"
                "ROOT = Path(__file__).resolve().parents[1]\n\n"
                "def _run(*args: str) -> subprocess.CompletedProcess[str]:\n"
                "    return subprocess.run([sys.executable, str(ROOT / 'task_cli.py'), *args], capture_output=True, text=True, check=False)\n\n"
                "class TaskCliTests(unittest.TestCase):\n"
                "    def test_priority_filter(self) -> None:\n"
                "        result = _run('list', '--priority', 'high')\n"
                "        self.assertEqual(result.returncode, 0)\n"
                "        self.assertIn('write-docs:todo:high:2026-07-01', result.stdout)\n"
                "        self.assertNotIn('ship-cli', result.stdout)\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n",
                encoding="utf-8",
            )
            client = FakeClient([])
            tools = CountingToolExecutor(root, approval_mode="auto", test_command="python -m unittest discover -s tests")
            agent = OllamaCodeAgent(
                client=client,
                tools=tools,
                model="fake-model",
                debate_enabled=False,
            )
            request_text = "Add a --due-before option to task_cli.py, update tests and README, run tests, and prove it with a shell command."
            obligations = agent._derive_request_obligations(
                request_text=request_text,
                required_tool_names=set(),
                required_mutation_paths=set(),
                code_mutation_required=True,
                test_run_required=True,
            )

            result = agent._try_post_context_cli_feature_repair(
                request_text=request_text,
                round_number=1,
                request_obligations=obligations,
                forbidden_tool_names=set(),
                successful_tool_results=[],
                satisfied_tool_names=set(),
                tool_calls_this_turn=[],
            )
            task_source = (root / "task_cli.py").read_text(encoding="utf-8")
            readme_text = (root / "README.md").read_text(encoding="utf-8")
            test_source = (root / "tests" / "test_task_cli.py").read_text(encoding="utf-8")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.completed)
        self.assertIn("--due-before", task_source)
        self.assertIn("--due-before", readme_text)
        self.assertIn("test_due_before_filter", test_source)

    def test_cli_flag_bundle_preemptive_mechanical_repair_path_is_not_skipped(self) -> None:
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
        self.assertFalse(any(event.get("phase") == "preemptive_mechanical_skipped" for event in spec_events))
        self.assertIsNone(tools.execute_counts.get("implementation_spec"))

    def test_cli_flag_bundle_shared_mechanical_repair_path_is_not_skipped(self) -> None:
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
        self.assertFalse(any(event.get("phase") == "mechanical_repair_skipped" for event in spec_events))

    def test_cli_flag_bundle_structured_and_spec_guided_repair_paths_are_not_skipped(self) -> None:
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
        self.assertFalse(any(event.get("phase") == "structured_test_driven_skipped" for event in spec_events))
        self.assertFalse(any(event.get("phase") == "spec_guided_repair_skipped" for event in spec_events))

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
