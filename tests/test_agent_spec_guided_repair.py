from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from ollama_code.agent import OllamaCodeAgent
from ollama_code.tools import ToolExecutor
from tests.agent_test_support import AgentTestBase, CountingToolExecutor, FakeClient


class AgentSpecGuidedRepairTests(AgentTestBase):
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
        with self._temp_agent(client, tool_cls=tool_cls, tool_kwargs=resolved_tool_kwargs, **kwargs) as (
            root,
            resolved_client,
            tools,
            agent,
        ):
            for relative_path, content in files.items():
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            yield root, resolved_client, tools, agent

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
