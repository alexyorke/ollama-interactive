from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import coding_benchmark_eval as bench


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
        self.assertGreaterEqual(len(full), 20)
        self.assertTrue({case.name for case in small}.issubset({case.name for case in full}))

    def test_coding_accuracy_prompts_do_not_leak_synthetic_answers(self) -> None:
        cases = bench.selected_cases("local-full")
        violations = {case.name: bench.prompt_integrity_findings(case) for case in cases if bench.prompt_integrity_findings(case)}

        self.assertEqual(violations, {})

    def test_tool_contract_cases_can_use_exact_tool_prompts_without_counting_as_coding_accuracy(self) -> None:
        cases = {case.name: case for case in bench.selected_cases("local-full")}

        self.assertEqual(cases["exact_literal_write_readback"].benchmark_kind, "tool_contract")
        self.assertEqual(cases["forbidden_tool_efficiency"].benchmark_kind, "tool_contract")
        self.assertEqual(cases["large_repo_symbol_nav"].benchmark_kind, "tool_contract")

    def test_prompt_integrity_flags_leaked_answers_for_coding_accuracy_cases(self) -> None:
        case = bench.BenchmarkCase(
            name="leaky",
            suite="local-small",
            turns=("Use git_status and tell me whether return 99 appears. Reply with TOKEN_42.",),
            validate=lambda ctx: "pass",
        )

        findings = bench.prompt_integrity_findings(case)

        self.assertIn("synthetic marker token", findings)
        self.assertIn("exact git-diff answer", findings)
        self.assertIn("forced git tool path", findings)

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
            **_: object,
        ) -> subprocess.CompletedProcess[str]:
            self.assertIsNotNone(session_file)
            calls.append({"prompt": prompt, "extra_args": list(extra_args or [])})
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

        def fake_build_workspace(workspace: Path) -> None:
            workspace.mkdir(parents=True, exist_ok=True)

        with patch.object(bench, "build_workspace", side_effect=fake_build_workspace), patch.object(bench, "run_cli", side_effect=fake_run_cli):
            outcome = bench.evaluate_case(Path.cwd(), "fake-model", None, "on", case, timeout=30)

        self.assertEqual(outcome["status"], "pass")
        self.assertEqual(outcome["usage"]["llm_calls"], 1)
        self.assertNotIn("--resume", calls[0]["extra_args"])
        self.assertIn("--resume", calls[1]["extra_args"])


if __name__ == "__main__":
    unittest.main()
