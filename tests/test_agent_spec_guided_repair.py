from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from ollama_code.agent import AgentResult, OllamaCodeAgent
from ollama_code.features import ENV_OLLAMA_CODE_FEATURE_PROFILE
from ollama_code.ollama_client import ChatResponse
from ollama_code.tools import ToolExecutor
from tests.agent_test_support import (
    AgentTestBase,
    CountingToolExecutor,
    FakeClient,
    RawImplementationSpecCountingToolExecutor,
)


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
        self.assertTrue(any(phase in {"start", "mechanical_candidate_validation"} for phase in phases))
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
