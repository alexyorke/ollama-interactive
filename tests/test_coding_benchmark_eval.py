from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import coding_benchmark_eval as bench
from scripts import public_benchmark_eval as public_bench


class CodingBenchmarkEvalTests(unittest.TestCase):
    def _context(self, workspace: Path, session: dict[str, object], *, stdout: str = "") -> bench.BenchmarkContext:
        case = bench.BenchmarkCase(
            name="unit",
            suite="local-small",
            turns=("prompt",),
            validate=lambda ctx: "pass",
        )
        return bench.BenchmarkContext(
            workspace=workspace,
            session=session,
            stdout=stdout,
            stderr="",
            returncodes=(0,),
            results=(subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=""),),
            case=case,
        )

    def test_selected_suites_have_expected_scope(self) -> None:
        small = bench.selected_cases("local-small")
        full = bench.selected_cases("local-full")

        self.assertEqual(len(small), 8)
        self.assertGreaterEqual(len(full), 25)
        self.assertTrue({case.name for case in small}.issubset({case.name for case in full}))

    def test_default_benchmark_jobs_prefers_explicit_env_then_ollama_host(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(bench.default_benchmark_jobs(), 1)
        with patch.dict("os.environ", {"OLLAMA_HOST": "http://127.0.0.1:11434"}, clear=True):
            self.assertEqual(bench.default_benchmark_jobs(), 8)
        with patch.dict("os.environ", {"OLLAMA_HOST": "http://127.0.0.1:11437"}, clear=True):
            self.assertEqual(bench.default_benchmark_jobs(), 12)
        with patch.dict("os.environ", {"OLLAMA_HOST": "127.0.0.1:11437"}, clear=True):
            self.assertEqual(bench.default_benchmark_jobs(), 12)
        with patch.dict("os.environ", {"OLLAMA_HOST": "http://127.0.0.1:11434", "OLLAMA_CODE_BENCH_JOBS": "6"}, clear=True):
            self.assertEqual(bench.default_benchmark_jobs(), 6)

    def test_selected_cases_can_filter_benchmark_class(self) -> None:
        agent_cases = bench.selected_cases("local-full", benchmark_classes={"agent"})
        controller_cases = bench.selected_cases("local-full", benchmark_classes={"controller"})

        self.assertTrue(agent_cases)
        self.assertTrue(controller_cases)
        self.assertTrue(all(bench.benchmark_class_for_case(case) == "agent" for case in agent_cases))
        self.assertTrue(all(bench.benchmark_class_for_case(case) == "controller" for case in controller_cases))

    def test_evaluate_case_batch_parallelizes_and_preserves_case_order(self) -> None:
        cases = [
            bench.BenchmarkCase(name="one", suite="local-small", turns=("a",), validate=lambda ctx: "pass"),
            bench.BenchmarkCase(name="two", suite="local-small", turns=("b",), validate=lambda ctx: "pass"),
            bench.BenchmarkCase(name="three", suite="local-small", turns=("c",), validate=lambda ctx: "pass"),
        ]
        active = 0
        peak = 0
        lock = threading.Lock()

        def fake_evaluate_case(
            repo_root: Path,
            model: str,
            verifier_model: str | None,
            mode: str,
            case: bench.BenchmarkCase,
            timeout: int,
            *,
            reconcile: str = "auto",
            feature_profile: str = "baseline",
        ) -> dict[str, object]:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return {"case": case.name, "status": "pass"}

        with patch.object(bench, "evaluate_case", side_effect=fake_evaluate_case):
            outcomes = bench.evaluate_case_batch(
                Path("."),
                "gemma4:e4b",
                None,
                "off",
                cases,
                60,
                jobs=3,
            )

        self.assertEqual([outcome["case"] for outcome in outcomes], ["one", "two", "three"])
        self.assertGreaterEqual(peak, 2)

    def test_main_passes_jobs_to_parallel_batch_runner(self) -> None:
        case = bench.BenchmarkCase(name="only", suite="local-small", turns=("prompt",), validate=lambda ctx: "pass")
        seen_jobs: list[int] = []

        def fake_batch(
            repo_root: Path,
            model: str,
            verifier_model: str | None,
            mode: str,
            cases: list[bench.BenchmarkCase],
            timeout: int,
            *,
            reconcile: str = "auto",
            feature_profile: str = "baseline",
            jobs: int = 1,
        ) -> list[dict[str, object]]:
            seen_jobs.append(jobs)
            return [{"case": case.name, "suite": case.suite, "status": "pass", "acceptable": ["pass"], "usage": {"llm_calls": 0, "total_tokens": 0}, "latency_s": 0.0}]

        with (
            patch.object(bench, "selected_cases", return_value=[case]),
            patch.object(bench, "installed_models", return_value=["gemma4:e4b"]),
            patch.object(bench, "resolve_requested_model", side_effect=lambda requested, available: requested if requested in available else None),
            patch.object(bench, "evaluate_case_batch", side_effect=fake_batch),
            patch.object(bench, "print_table"),
            patch.object(bench, "write_results_payload"),
            patch.object(bench, "unload_model"),
        ):
            rc = bench.main(["--suite", "local-small", "--models", "gemma4:e4b", "--modes", "off", "--reconcile-modes", "off", "--jobs", "3"])

        self.assertEqual(rc, 0)
        self.assertEqual(seen_jobs, [3])

    def test_llm_bypass_failures_only_flag_zero_llm_coding_accuracy(self) -> None:
        results = [
            {"case": "coding-zero", "benchmark_kind": "coding_accuracy", "benchmark_class": "agent", "usage": {"llm_calls": 0}},
            {"case": "coding-one", "benchmark_kind": "coding_accuracy", "benchmark_class": "agent", "usage": {"llm_calls": 1}},
            {"case": "tool-zero", "benchmark_kind": "tool_contract", "benchmark_class": "controller", "usage": {"llm_calls": 0}},
        ]

        failures = bench.llm_bypass_failures(results)

        self.assertEqual([item["case"] for item in failures], ["coding-zero"])
        self.assertIn("agent benchmark", failures[0]["llm_bypass_reason"])

    def test_main_can_fail_when_coding_accuracy_uses_zero_llm_calls(self) -> None:
        case = bench.BenchmarkCase(name="only", suite="local-small", turns=("prompt",), validate=lambda ctx: "pass")

        def fake_batch(
            repo_root: Path,
            model: str,
            verifier_model: str | None,
            mode: str,
            cases: list[bench.BenchmarkCase],
            timeout: int,
            *,
            reconcile: str = "auto",
            feature_profile: str = "baseline",
            jobs: int = 1,
        ) -> list[dict[str, object]]:
            return [
                {
                    "case": case.name,
                    "suite": case.suite,
                    "benchmark_kind": "coding_accuracy",
                    "status": "pass",
                    "acceptable": ["pass"],
                    "usage": {"llm_calls": 0, "total_tokens": 0},
                    "latency_s": 0.0,
                }
            ]

        with (
            patch.object(bench, "selected_cases", return_value=[case]),
            patch.object(bench, "installed_models", return_value=["gemma4:e4b"]),
            patch.object(bench, "resolve_requested_model", side_effect=lambda requested, available: requested if requested in available else None),
            patch.object(bench, "evaluate_case_batch", side_effect=fake_batch),
            patch.object(bench, "print_table"),
            patch.object(bench, "unload_model"),
        ):
            rc = bench.main(["--suite", "local-small", "--models", "gemma4:e4b", "--modes", "off", "--reconcile-modes", "off", "--require-llm-for-agent-benchmarks"])

        self.assertEqual(rc, 1)

    def test_main_returns_nonzero_when_no_requested_models_are_available(self) -> None:
        case = bench.BenchmarkCase(name="only", suite="local-small", turns=("prompt",), validate=lambda ctx: "pass")

        with (
            patch.object(bench, "selected_cases", return_value=[case]),
            patch.object(bench, "installed_models", return_value=[]),
        ):
            rc = bench.main(["--suite", "local-small", "--models", "demo-model", "--modes", "off", "--reconcile-modes", "off"])

        self.assertEqual(rc, 1)

    def test_coding_accuracy_prompts_do_not_leak_synthetic_answers(self) -> None:
        cases = bench.selected_cases("local-full")
        violations = {case.name: bench.prompt_integrity_findings(case) for case in cases if bench.prompt_integrity_findings(case)}

        self.assertEqual(violations, {})

    def test_runtime_source_does_not_special_case_public_hard_task_slugs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        runtime_text = "\n".join(path.read_text(encoding="utf-8") for path in (repo_root / "ollama_code").glob("*.py"))

        audited_task_slugs = [task for task in public_bench.HARD_POLYGLOT_TASKS if task != "transpose"]
        for forbidden in audited_task_slugs:
            self.assertNotIn(forbidden, runtime_text)

    def test_local_full_includes_generic_renamed_hidden_cases(self) -> None:
        cases = {case.name: case for case in bench.selected_cases("local-full")}

        self.assertIn("bad_test_command_recovery", cases)
        self.assertIn("docs_sync_without_tests_still_validates", cases)
        self.assertEqual(cases["bad_test_command_recovery"].benchmark_kind, "coding_accuracy")
        self.assertEqual(cases["docs_sync_without_tests_still_validates"].benchmark_kind, "coding_accuracy")
        for name in (
            "renamed_simple_expression_hidden",
            "renamed_prefix_rotation_hidden",
            "renamed_word_arithmetic_hidden",
            "renamed_text_matrix_hidden",
        ):
            self.assertIn(name, cases)
            self.assertEqual(cases[name].benchmark_kind, "coding_accuracy")

    def test_docs_sync_without_tests_still_validates_requires_non_test_validation(self) -> None:
        cases = {case.name: case for case in bench.selected_cases("local-full")}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = cases["docs_sync_without_tests_still_validates"]
            assert case.prepare is not None
            case.prepare(root)
            (root / "src" / "api.py").write_text(
                "def fetch_user(user_id: str, include_orders: bool = False) -> dict[str, str]:\n    return {'id': user_id}\n",
                encoding="utf-8",
            )
            (root / "docs" / "api.md").write_text(
                "`fetch_user(user_id, include_orders=False)` returns a user dict.\n",
                encoding="utf-8",
            )
            passing = bench.BenchmarkContext(
                workspace=root,
                session={
                    "events": [
                        {"type": "tool_call", "name": "lint_typecheck", "arguments": {"paths": ["src/api.py"]}},
                        {"type": "tool_result", "name": "lint_typecheck", "result": {"ok": True}},
                        {"type": "tool_call", "name": "contract_check", "arguments": {"changed_files": ["src/api.py"]}},
                        {"type": "tool_result", "name": "contract_check", "result": {"ok": True}},
                    ]
                },
                stdout="",
                stderr="",
                returncodes=(),
                results=(),
                case=case,
            )
            failing = bench.BenchmarkContext(
                workspace=root,
                session={
                    "events": [
                        {"type": "tool_call", "name": "lint_typecheck", "arguments": {"paths": ["src/api.py"]}},
                        {"type": "tool_result", "name": "lint_typecheck", "result": {"ok": True}},
                        {"type": "tool_call", "name": "run_test", "arguments": {"command": "python -m unittest discover -s tests"}},
                        {"type": "tool_result", "name": "run_test", "result": {"ok": True}},
                    ]
                },
                stdout="",
                stderr="",
                returncodes=(),
                results=(),
                case=case,
            )

            self.assertEqual(case.validate(passing), "pass")
            self.assertEqual(case.validate(failing), "fail")

    def test_renamed_hidden_case_validators_enforce_hidden_behavior(self) -> None:
        cases = {case.name: case for case in bench.selected_cases("local-full")}
        correct_sources = {
            "renamed_simple_expression_hidden": "def score_delta(base: int, bonus: int) -> int:\n    return base + bonus\n",
            "renamed_prefix_rotation_hidden": (
                "def transform_words(text: str) -> str:\n"
                "    def convert(word: str) -> str:\n"
                "        lower = word.lower()\n"
                "        if lower.startswith(('xr', 'yt')) or lower[:1] in 'aeiou':\n"
                "            return word + 'ay'\n"
                "        index = 0\n"
                "        while index < len(word):\n"
                "            if lower.startswith('qu', index):\n"
                "                index += 2\n"
                "                continue\n"
                "            if lower[index] in 'aeiouy' and not (index == 0 and lower[index] == 'y'):\n"
                "                break\n"
                "            index += 1\n"
                "        return word[index:] + word[:index] + 'ay'\n"
                "    return ' '.join(convert(word) for word in text.split())\n"
            ),
            "renamed_word_arithmetic_hidden": (
                "def solve(question):\n"
                "    if not isinstance(question, str) or not question.startswith('What is') or not question.endswith('?'):\n"
                "        raise ValueError('unknown operation')\n"
                "    tokens = question[len('What is'):-1].strip().split()\n"
                "    if not tokens:\n"
                "        raise ValueError('syntax error')\n"
                "    try:\n"
                "        value = int(tokens[0])\n"
                "    except ValueError as exc:\n"
                "        raise ValueError('syntax error') from exc\n"
                "    index = 1\n"
                "    while index < len(tokens):\n"
                "        token = tokens[index]\n"
                "        if token == 'plus':\n"
                "            op = 'plus'\n"
                "            index += 1\n"
                "        elif token == 'minus':\n"
                "            op = 'minus'\n"
                "            index += 1\n"
                "        elif token == 'multiplied' and index + 1 < len(tokens) and tokens[index + 1] == 'by':\n"
                "            op = 'multiplied'\n"
                "            index += 2\n"
                "        elif token == 'divided' and index + 1 < len(tokens) and tokens[index + 1] == 'by':\n"
                "            op = 'divided'\n"
                "            index += 2\n"
                "        else:\n"
                "            raise ValueError('unknown operation')\n"
                "        if index >= len(tokens):\n"
                "            raise ValueError('syntax error')\n"
                "        try:\n"
                "            operand = int(tokens[index])\n"
                "        except ValueError as exc:\n"
                "            raise ValueError('syntax error') from exc\n"
                "        index += 1\n"
                "        if op == 'plus':\n"
                "            value += operand\n"
                "        elif op == 'minus':\n"
                "            value -= operand\n"
                "        elif op == 'multiplied':\n"
                "            value *= operand\n"
                "        else:\n"
                "            value = int(value / operand)\n"
                "    return value\n"
            ),
            "renamed_text_matrix_hidden": (
                "def flip_text(block):\n"
                "    if block == '':\n"
                "        return ''\n"
                "    rows = block.split('\\n')\n"
                "    width = max((len(row) for row in rows), default=0)\n"
                "    transposed = []\n"
                "    for column in range(width):\n"
                "        cells = [row[column] if column < len(row) else None for row in rows]\n"
                "        while cells and cells[-1] is None:\n"
                "            cells.pop()\n"
                "        transposed.append(''.join(cell if cell is not None else ' ' for cell in cells))\n"
                "    return '\\n'.join(transposed)\n"
            ),
        }

        for name, source in correct_sources.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                case = cases[name]
                assert case.prepare is not None
                case.prepare(root)
                target = {
                    "renamed_simple_expression_hidden": root / "src" / "scoreboard.py",
                    "renamed_prefix_rotation_hidden": root / "src" / "syllables.py",
                    "renamed_word_arithmetic_hidden": root / "src" / "story_solver.py",
                    "renamed_text_matrix_hidden": root / "src" / "text_grid.py",
                }[name]
                target.write_text(source, encoding="utf-8")
                ctx = bench.BenchmarkContext(
                    workspace=root,
                    session={},
                    stdout="",
                    stderr="",
                    returncodes=(),
                    results=(),
                    case=case,
                )
                self.assertEqual(case.validate(ctx), "pass")

    def test_tool_contract_cases_can_use_exact_tool_prompts_without_counting_as_coding_accuracy(self) -> None:
        cases = {case.name: case for case in bench.selected_cases("local-full")}

        self.assertEqual(cases["exact_literal_write_readback"].benchmark_kind, "tool_contract")
        self.assertEqual(cases["forbidden_tool_efficiency"].benchmark_kind, "tool_contract")
        self.assertEqual(cases["large_repo_symbol_nav"].benchmark_kind, "tool_contract")
        self.assertEqual(cases["single_file_literal_read"].benchmark_kind, "tool_contract")
        self.assertEqual(cases["discover_validators_natural"].benchmark_kind, "tool_contract")
        self.assertEqual(cases["search_then_run_test_summary"].benchmark_kind, "tool_contract")
        self.assertEqual(cases["search_then_git_status"].benchmark_kind, "tool_contract")
        self.assertEqual(cases["search_and_git_status"].benchmark_kind, "tool_contract")
        self.assertEqual(cases["list_files_and_git_status"].benchmark_kind, "tool_contract")
        self.assertEqual(cases["discover_validators_then_lint"].benchmark_kind, "tool_contract")
        self.assertEqual(cases["discover_validators_and_lint"].benchmark_kind, "tool_contract")
        self.assertEqual(bench.benchmark_class_for_case(cases["large_repo_symbol_nav"]), "controller")
        self.assertEqual(bench.benchmark_class_for_case(cases["single_file_literal_read"]), "controller")
        self.assertTrue(cases["forbidden_tool_efficiency"].requires_git)
        self.assertTrue(cases["staged_vs_worktree_diff"].requires_git)

    def test_prompt_integrity_flags_leaked_answers_for_coding_accuracy_cases(self) -> None:
        case = bench.BenchmarkCase(
            name="leaky",
            suite="local-small",
            turns=("Implement transpose.py, use git_status and tell me whether return 99 appears. Reply with TOKEN_42.",),
            validate=lambda ctx: "pass",
        )

        findings = bench.prompt_integrity_findings(case)

        self.assertIn("synthetic marker token", findings)
        self.assertIn("public benchmark module name", findings)
        self.assertIn("exact git-diff answer", findings)
        self.assertIn("forced git tool path", findings)

    def test_prompt_integrity_flags_expanded_task_slug_leaks(self) -> None:
        case = bench.BenchmarkCase(
            name="leaky",
            suite="local-small",
            turns=("Implement affine-cipher from the tests and keep the behavior exact.",),
            benchmark_kind="coding_accuracy",
            validate=lambda ctx: "pass",
        )

        findings = bench.prompt_integrity_findings(case)

        self.assertIn("public benchmark task slug", findings)
        self.assertIn("public benchmark module name", findings)

    def test_usage_totals_aggregates_prompt_profile(self) -> None:
        session = {
            "events": [
                {
                    "type": "llm_call",
                    "purpose": "primary",
                    "prompt_tokens": 10,
                    "output_tokens": 3,
                    "total_tokens": 13,
                    "prompt_chars": 100,
                    "response_chars": 20,
                    "prompt_chars_by_role": {"system": 70, "user": 30},
                    "top_prompt_messages": [{"role": "system", "chars": 70, "preview": "rules"}],
                },
                {
                    "type": "llm_call",
                    "purpose": "assumption_audit",
                    "prompt_tokens": 5,
                    "output_tokens": 1,
                    "total_tokens": 6,
                    "prompt_chars_by_role": {"user": 10},
                    "top_prompt_messages": [{"role": "user", "chars": 10, "preview": "audit"}],
                },
            ]
        }

        totals = bench.usage_totals(session)

        self.assertEqual(totals["llm_calls"], 2)
        self.assertEqual(totals["prompt_tokens"], 15)
        self.assertEqual(totals["output_tokens"], 4)
        self.assertEqual(totals["total_tokens"], 19)
        self.assertEqual(totals["purposes"]["primary"]["calls"], 1)
        self.assertEqual(totals["prompt_chars_by_role"]["user"], 40)
        self.assertEqual(totals["top_prompt_messages"][0]["chars"], 70)

    def test_budget_violations_check_calls_and_tokens(self) -> None:
        case = bench.BenchmarkCase(
            name="budgeted",
            suite="local-small",
            turns=("prompt",),
            validate=lambda ctx: "pass",
            budget_off=bench.BenchmarkBudget(max_llm_calls=1, max_total_tokens=10),
        )
        outcome = {"debate": "off", "usage": {"llm_calls": 2, "total_tokens": 11}}

        violations = bench.budget_violations(outcome, case)

        self.assertEqual(violations, ["llm_calls 2>1", "total_tokens 11>10"])

    def test_max_rounds_stop_counts_as_fail_closed(self) -> None:
        self.assertTrue(bench.is_fail_closed_message("Stopped after reaching the maximum tool rounds."))

    def test_comparison_rows_report_status_and_token_deltas(self) -> None:
        baseline = [
            {
                "suite": "local-small",
                "model": "gemma3:4b",
                "verifier_model": None,
                "debate": "on",
                "case": "issue",
                "status": "pass",
                "latency_s": 10,
                "usage": {"total_tokens": 100, "llm_calls": 4},
            }
        ]
        current = [
            {
                "suite": "local-small",
                "model": "gemma3:4b",
                "verifier_model": None,
                "debate": "on",
                "case": "issue",
                "status": "fail",
                "latency_s": 8,
                "usage": {"total_tokens": 75, "llm_calls": 3},
            }
        ]

        rows = bench.comparison_rows(current, baseline)

        self.assertEqual(rows[0]["before_status"], "pass")
        self.assertEqual(rows[0]["after_status"], "fail")
        self.assertEqual(rows[0]["total_token_delta_pct"], -25.0)
        self.assertEqual(rows[0]["before_llm_calls"], 4)
        self.assertEqual(rows[0]["after_llm_calls"], 3)

    def test_comparison_rows_separate_feature_profiles(self) -> None:
        baseline = [
            {
                "suite": "local-small",
                "model": "gemma3:4b",
                "verifier_model": None,
                "debate": "off",
                "reconcile": "auto",
                "feature_profile": "baseline",
                "case": "issue",
                "status": "pass",
                "usage": {"total_tokens": 100, "llm_calls": 4},
            }
        ]
        current = [
            {
                "suite": "local-small",
                "model": "gemma3:4b",
                "verifier_model": None,
                "debate": "off",
                "reconcile": "auto",
                "feature_profile": "all",
                "case": "issue",
                "status": "pass",
                "usage": {"total_tokens": 70, "llm_calls": 3},
            }
        ]

        self.assertEqual(bench.comparison_rows(current, baseline), [])

    def test_comparison_rows_separate_benchmark_classes(self) -> None:
        baseline = [
            {
                "suite": "local-small",
                "model": "gemma4:e4b",
                "verifier_model": None,
                "debate": "off",
                "reconcile": "auto",
                "feature_profile": "all",
                "benchmark_kind": "coding_accuracy",
                "benchmark_class": "agent",
                "case": "shared",
                "status": "pass",
                "usage": {"total_tokens": 100, "llm_calls": 4},
                "latency_s": 1.0,
            }
        ]
        current = [
            {
                "suite": "local-small",
                "model": "gemma4:e4b",
                "verifier_model": None,
                "debate": "off",
                "reconcile": "auto",
                "feature_profile": "all",
                "benchmark_kind": "tool_contract",
                "benchmark_class": "controller",
                "case": "shared",
                "status": "pass",
                "usage": {"total_tokens": 0, "llm_calls": 0},
                "latency_s": 0.5,
            }
        ]

        self.assertEqual(bench.comparison_rows(current, baseline), [])

    def test_summarize_reports_agent_and_controller_classes_separately(self) -> None:
        results = [
            {"status": "pass", "benchmark_kind": "coding_accuracy", "benchmark_class": "agent", "usage": {"llm_calls": 2, "total_tokens": 10}},
            {"status": "pass", "benchmark_kind": "tool_contract", "benchmark_class": "controller", "usage": {"llm_calls": 0, "total_tokens": 0}},
            {"status": "fail", "benchmark_kind": "tool_contract", "benchmark_class": "controller", "usage": {"llm_calls": 1, "total_tokens": 3}},
        ]

        summary = bench.summarize(results)

        self.assertEqual(summary["by_benchmark_class"]["agent"]["runs"], 1)
        self.assertEqual(summary["by_benchmark_class"]["agent"]["total_tokens"], 10)
        self.assertEqual(summary["by_benchmark_class"]["controller"]["runs"], 2)
        self.assertEqual(summary["by_benchmark_class"]["controller"]["total_llm_calls"], 1)

    def test_issue_validator_passes_hidden_solution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            bench.prepare_issue_fix_hidden_tests(workspace)
            (workspace / "src" / "calculator.py").write_text("def add(left: int, right: int) -> int:\n    return left + right\n", encoding="utf-8")

            status = bench.validate_issue_fix_hidden_tests(self._context(workspace, {"events": []}))

        self.assertEqual(status, "pass")

    def test_multi_file_refactor_validator_checks_docs_and_hidden_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            bench.prepare_multi_file_refactor(workspace)
            (workspace / "src" / "pricing.py").write_text("def cart_total(prices: list[int]) -> int:\n    return sum(prices)\n", encoding="utf-8")
            (workspace / "docs" / "pricing.md").write_text("Call `cart_total(prices)` to compute a cart total.\n", encoding="utf-8")

            status = bench.validate_multi_file_refactor(self._context(workspace, {"events": []}))

        self.assertEqual(status, "pass")

    def test_bad_test_command_recovery_validator_requires_fallback_and_hidden_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            bench.prepare_bad_test_command_recovery(workspace)
            (workspace / "src" / "inventory.py").write_text("def total_units(counts: list[int]) -> int:\n    return sum(counts)\n", encoding="utf-8")
            session = {
                "events": [
                    {
                        "type": "tool_result",
                        "name": "run_test",
                        "result": {
                            "ok": True,
                            "recovered": True,
                            "original_command": "pytesst -q",
                            "command": f"{sys.executable} -m unittest discover -s tests -v",
                        },
                    }
                ]
            }

            status = bench.validate_bad_test_command_recovery(self._context(workspace, session))

        self.assertEqual(status, "pass")

    def test_forbidden_tool_validator_rejects_read_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            session = {
                "events": [
                    {"type": "tool_call", "name": "read_file", "arguments": {"path": "src/app.py"}},
                    {"type": "tool_result", "name": "git_status", "result": {"ok": True, "output": " M src/app.py"}},
                    {"type": "tool_result", "name": "git_diff", "result": {"ok": True, "output": "+    return 99"}},
                ]
            }

            status = bench.validate_forbidden_tool_efficiency(self._context(workspace, session, stdout="return 99"))

        self.assertEqual(status, "fail")

    def test_test_repair_validator_accepts_shell_test_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            bench.prepare_test_repair_task(workspace)
            (workspace / "src" / "slug.py").write_text(
                "import re\n\n"
                "def slugify(value: str) -> str:\n"
                "    value = re.sub(r'[^\\w\\s-]', '', value.strip().lower())\n"
                "    value = re.sub(r'[\\s-]+', '-', value)\n"
                "    return value.strip('-')\n",
                encoding="utf-8",
            )
            session = {"events": [{"type": "tool_call", "name": "run_shell", "arguments": {"command": "python -m unittest"}}]}

            status = bench.validate_test_repair_task(self._context(workspace, session))

        self.assertEqual(status, "pass")

    def test_regression_token_trap_rejects_duplicate_symbol_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            session = {
                "events": [
                    {"type": "tool_call", "name": "search_symbols", "arguments": {"query": "trap_value"}},
                    {"type": "tool_call", "name": "read_symbol", "arguments": {"path": "src/trap.py", "symbol": "trap_value"}},
                    {"type": "tool_call", "name": "read_symbol", "arguments": {"path": "src/trap.py", "symbol": "trap_value"}},
                ]
            }

            status = bench.validate_regression_token_traps(self._context(workspace, session, stdout="314159"))

        self.assertEqual(status, "fail")

    def test_evaluate_case_uses_resume_for_multiturn_and_collects_usage(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_run_cli(
            repo_root: Path,
            workspace: Path,
            model: str,
            prompt: str,
            *,
            session_file: Path | None = None,
            extra_args: list[str] | None = None,
            extra_env: dict[str, str] | None = None,
            require_llm_for_turn: bool = True,
            **_: object,
        ) -> subprocess.CompletedProcess[str]:
            self.assertIsNotNone(session_file)
            calls.append(
                {
                    "prompt": prompt,
                    "extra_args": list(extra_args or []),
                    "extra_env": dict(extra_env or {}),
                    "require_llm_for_turn": require_llm_for_turn,
                }
            )
            session_file = Path(session_file or workspace / "scratch" / "session.json")
            session_file.parent.mkdir(parents=True, exist_ok=True)
            session_file.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "type": "llm_call",
                                "purpose": "primary",
                                "prompt_tokens": 7,
                                "output_tokens": 2,
                                "total_tokens": 9,
                            },
                            {"type": "assistant", "content": "ok"},
                        ],
                        "messages": [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}],
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr="")

        case = bench.BenchmarkCase(
            name="fake_multiturn",
            suite="local-small",
            turns=("one", "two"),
            validate=lambda ctx: "pass" if len(ctx.results) == 2 else "fail",
        )

        def fake_build_workspace(workspace: Path, *, init_git: bool = True) -> bool:
            workspace.mkdir(parents=True, exist_ok=True)
            return init_git

        with patch.object(bench, "build_workspace", side_effect=fake_build_workspace), patch.object(bench, "run_cli", side_effect=fake_run_cli):
            outcome = bench.evaluate_case(Path.cwd(), "fake-model", None, "on", case, timeout=30, feature_profile="all")

        self.assertEqual(outcome["status"], "pass")
        self.assertEqual(outcome["feature_profile"], "all")
        self.assertEqual(outcome["usage"]["llm_calls"], 1)
        self.assertEqual(calls[0]["extra_env"]["OLLAMA_CODE_FEATURE_PROFILE"], "all")
        self.assertIn("GIT_CEILING_DIRECTORIES", calls[0]["extra_env"])
        self.assertNotIn("--resume", calls[0]["extra_args"])
        self.assertIn("--resume", calls[1]["extra_args"])
        self.assertTrue(calls[0]["require_llm_for_turn"])

    def test_evaluate_case_disables_require_llm_for_controller_benchmarks(self) -> None:
        seen_require_llm: list[bool] = []

        def fake_run_cli(
            repo_root: Path,
            workspace: Path,
            model: str,
            prompt: str,
            *,
            session_file: Path | None = None,
            require_llm_for_turn: bool = True,
            **_: object,
        ) -> subprocess.CompletedProcess[str]:
            self.assertIsNotNone(session_file)
            seen_require_llm.append(require_llm_for_turn)
            session_file = Path(session_file or workspace / "scratch" / "session.json")
            session_file.parent.mkdir(parents=True, exist_ok=True)
            session_file.write_text(json.dumps({"events": [{"type": "assistant", "content": "ok"}], "messages": []}), encoding="utf-8")
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr="")

        case = bench.BenchmarkCase(
            name="controller_case",
            suite="local-small",
            turns=("prompt",),
            benchmark_kind="tool_contract",
            validate=lambda ctx: "pass",
        )

        def fake_build_workspace(workspace: Path, *, init_git: bool = True) -> bool:
            workspace.mkdir(parents=True, exist_ok=True)
            return init_git

        with patch.object(bench, "build_workspace", side_effect=fake_build_workspace), patch.object(bench, "run_cli", side_effect=fake_run_cli):
            outcome = bench.evaluate_case(Path.cwd(), "fake-model", None, "off", case, timeout=30)

        self.assertEqual(outcome["benchmark_class"], "controller")
        self.assertEqual(seen_require_llm, [False])

    def test_evaluate_case_skips_git_required_case_when_nested_git_unavailable(self) -> None:
        case = bench.BenchmarkCase(
            name="git_case",
            suite="local-small",
            turns=("prompt",),
            validate=lambda ctx: "pass",
            requires_git=True,
        )

        def fake_build_workspace(workspace: Path, *, init_git: bool = True) -> bool:
            workspace.mkdir(parents=True, exist_ok=True)
            self.assertTrue(init_git)
            return False

        with patch.object(bench, "build_workspace", side_effect=fake_build_workspace), patch.object(bench, "run_cli") as run_cli:
            outcome = bench.evaluate_case(Path.cwd(), "fake-model", None, "off", case, timeout=30, feature_profile="all")

        self.assertEqual(outcome["status"], "skip")
        self.assertIn("skip", outcome["acceptable"])
        self.assertEqual(outcome["skip_reason"], "nested git workspace unavailable")
        run_cli.assert_not_called()

    def test_git_required_benchmark_uses_external_temp_root_by_default(self) -> None:
        case = bench.BenchmarkCase(
            name="git_case",
            suite="local-small",
            turns=("prompt",),
            validate=lambda ctx: "pass",
            requires_git=True,
        )
        seen: list[Path] = []

        def fake_build_workspace(workspace: Path, *, init_git: bool = True) -> bool:
            seen.append(workspace)
            workspace.mkdir(parents=True, exist_ok=True)
            return False

        with patch.dict("os.environ", {}, clear=False), patch.object(bench, "build_workspace", side_effect=fake_build_workspace):
            bench.evaluate_case(Path.cwd(), "fake-model", None, "off", case, timeout=30, feature_profile="all")

        self.assertTrue(seen)
        self.assertIn(".codex", seen[0].as_posix())

    def test_evaluate_case_retries_transient_ollama_timeout_once_for_agent_benchmark(self) -> None:
        seen_workspaces: list[Path] = []

        def fake_build_workspace(workspace: Path, *, init_git: bool = True) -> bool:
            workspace.mkdir(parents=True, exist_ok=True)
            return init_git

        def fake_run_cli(
            repo_root: Path,
            workspace: Path,
            model: str,
            prompt: str,
            *,
            session_file: Path | None = None,
            **_: object,
        ) -> subprocess.CompletedProcess[str]:
            self.assertIsNotNone(session_file)
            seen_workspaces.append(workspace)
            session_file = Path(session_file or workspace / "scratch" / "session.json")
            session_file.parent.mkdir(parents=True, exist_ok=True)
            session_file.write_text(json.dumps({"events": [{"type": "assistant", "content": "ok"}], "messages": []}), encoding="utf-8")
            if len(seen_workspaces) == 1:
                return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="error: Ollama timed out after 300 seconds.\n")
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr="")

        case = bench.BenchmarkCase(
            name="retryable_agent_case",
            suite="local-small",
            turns=("prompt",),
            validate=lambda ctx: "pass",
        )

        with patch.object(bench, "build_workspace", side_effect=fake_build_workspace), patch.object(bench, "run_cli", side_effect=fake_run_cli):
            outcome = bench.evaluate_case(Path.cwd(), "fake-model", None, "off", case, timeout=30)

        self.assertEqual(outcome["status"], "pass")
        self.assertEqual(outcome.get("transient_ollama_retries"), 1)
        self.assertEqual(len(seen_workspaces), 2)
        self.assertNotEqual(seen_workspaces[0], seen_workspaces[1])

    def test_evaluate_case_does_not_retry_non_timeout_failure(self) -> None:
        seen_workspaces: list[Path] = []

        def fake_build_workspace(workspace: Path, *, init_git: bool = True) -> bool:
            workspace.mkdir(parents=True, exist_ok=True)
            return init_git

        def fake_run_cli(
            repo_root: Path,
            workspace: Path,
            model: str,
            prompt: str,
            *,
            session_file: Path | None = None,
            **_: object,
        ) -> subprocess.CompletedProcess[str]:
            self.assertIsNotNone(session_file)
            seen_workspaces.append(workspace)
            session_file = Path(session_file or workspace / "scratch" / "session.json")
            session_file.parent.mkdir(parents=True, exist_ok=True)
            session_file.write_text(json.dumps({"events": [], "messages": []}), encoding="utf-8")
            return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="error: model unavailable\n")

        case = bench.BenchmarkCase(
            name="non_retryable_agent_case",
            suite="local-small",
            turns=("prompt",),
            validate=lambda ctx: "pass",
        )

        with patch.object(bench, "build_workspace", side_effect=fake_build_workspace), patch.object(bench, "run_cli", side_effect=fake_run_cli):
            outcome = bench.evaluate_case(Path.cwd(), "fake-model", None, "off", case, timeout=30)

        self.assertEqual(outcome["status"], "fail")
        self.assertEqual(len(seen_workspaces), 1)
        self.assertNotIn("transient_ollama_retries", outcome)


if __name__ == "__main__":
    unittest.main()
