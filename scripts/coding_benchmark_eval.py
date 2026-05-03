from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from e2e_suite import build_workspace, commit_all, installed_models, load_session, run_cli
    from workspace_temp import workspace_temp_dir
except ModuleNotFoundError:  # Imported as scripts.coding_benchmark_eval in unit tests.
    from scripts.e2e_suite import build_workspace, commit_all, installed_models, load_session, run_cli
    from scripts.workspace_temp import workspace_temp_dir


_LOADED_MODELS: set[str] = set()
Command = list[str] | str
FAIL_CLOSED_MESSAGES = {
    "Stopped because grounded final verification could not accept a final answer.",
    "Stopped because assumption audit could not approve a next tool step.",
}


def unload_model(model: str) -> None:
    subprocess.run(["ollama", "stop", model], capture_output=True, text=True, check=False)


def _cleanup_loaded_models() -> None:
    for model in sorted(_LOADED_MODELS):
        unload_model(model)


atexit.register(_cleanup_loaded_models)


@dataclass(frozen=True)
class BenchmarkBudget:
    max_llm_calls: int | None = None
    max_total_tokens: int | None = None


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    suite: str
    turns: tuple[str, ...]
    validate: Callable[["BenchmarkContext"], str]
    benchmark_kind: str = "coding_accuracy"
    prepare: Callable[[Path], None] | None = None
    test_cmd: str | None = None
    acceptable: tuple[str, ...] = ("pass",)
    budget_off: BenchmarkBudget = BenchmarkBudget(max_llm_calls=12, max_total_tokens=80_000)
    budget_on: BenchmarkBudget = BenchmarkBudget(max_llm_calls=16, max_total_tokens=120_000)
    timeout: int | None = None


def prompt_integrity_findings(case: BenchmarkCase) -> list[str]:
    if case.benchmark_kind != "coding_accuracy":
        return []
    text = "\n".join(case.turns)
    checks = {
        "synthetic marker token": r"\b(?:BENCH|TOKEN|NEEDLE|EXACT)_[A-Z0-9_]+\b",
        "exact git-diff answer": r"\breturn\s+(?:22|99)\b",
        "exact shell command": r"\bexecute exactly\b",
        "forced read_file path": r"\bUse read_file\b",
        "forced git tool path": r"\bUse git_(?:status|diff)\b",
        "forbidden-tool routing clause": r"\bDo not use read_file\b",
    }
    findings: list[str] = []
    for label, pattern in checks.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            findings.append(label)
    return findings


@dataclass(frozen=True)
class BenchmarkContext:
    workspace: Path
    session: dict[str, Any]
    stdout: str
    stderr: str
    returncodes: tuple[int, ...]
    results: tuple[subprocess.CompletedProcess[str], ...]
    case: BenchmarkCase


def resolve_requested_model(model: str, available: set[str]) -> str | None:
    if model in available:
        return model
    latest = f"{model}:latest"
    if latest in available:
        return latest
    return None


def tool_results(session: dict[str, Any], tool_name: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for event in session.get("events", []):
        if event.get("type") == "tool_result" and event.get("name") == tool_name:
            result = event.get("result")
            if isinstance(result, dict):
                results.append(result)
    return results


def tool_calls(session: dict[str, Any]) -> list[str]:
    return [
        str(event["name"])
        for event in session.get("events", [])
        if event.get("type") == "tool_call" and isinstance(event.get("name"), str)
    ]


def tool_call_args(session: dict[str, Any], tool_name: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for event in session.get("events", []):
        if event.get("type") == "tool_call" and event.get("name") == tool_name:
            arguments = event.get("arguments")
            if isinstance(arguments, dict):
                calls.append(arguments)
    return calls


def final_assistant_message(session: dict[str, Any]) -> str:
    for event in reversed(session.get("events", [])):
        if event.get("type") == "assistant" and isinstance(event.get("content"), str):
            return str(event["content"]).strip()
    return ""


def is_fail_closed_message(message: str) -> bool:
    return message in FAIL_CLOSED_MESSAGES


def usage_totals(session: dict[str, Any]) -> dict[str, Any]:
    events = [event for event in session.get("events", []) if event.get("type") == "llm_call"]
    totals: dict[str, Any] = {
        "llm_calls": len(events),
        "prompt_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_duration_ns": 0,
        "prompt_chars": 0,
        "response_chars": 0,
        "purposes": {},
        "prompt_chars_by_role": {},
        "top_prompt_messages": [],
    }
    purposes: dict[str, dict[str, int]] = {}
    prompt_chars_by_role: dict[str, int] = {}
    top_messages: list[dict[str, Any]] = []
    for event in events:
        purpose = str(event.get("purpose", "unknown"))
        bucket = purposes.setdefault(purpose, {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        bucket["calls"] += 1
        for key in ("prompt_tokens", "output_tokens", "total_tokens"):
            value = event.get(key)
            if isinstance(value, int):
                totals[key] += value
                bucket[key] += value
        for key in ("total_duration_ns", "prompt_chars", "response_chars"):
            value = event.get(key)
            if isinstance(value, int):
                totals[key] += value
        role_chars = event.get("prompt_chars_by_role")
        if isinstance(role_chars, dict):
            for role, value in role_chars.items():
                if isinstance(role, str) and isinstance(value, int):
                    prompt_chars_by_role[role] = prompt_chars_by_role.get(role, 0) + value
        event_top = event.get("top_prompt_messages")
        if isinstance(event_top, list):
            for item in event_top:
                if not isinstance(item, dict):
                    continue
                chars = item.get("chars")
                if not isinstance(chars, int):
                    continue
                top_messages.append(
                    {
                        "purpose": purpose,
                        "role": str(item.get("role", "")),
                        "chars": chars,
                        "preview": str(item.get("preview", ""))[:100],
                    }
                )
    totals["purposes"] = purposes
    totals["prompt_chars_by_role"] = dict(sorted(prompt_chars_by_role.items()))
    totals["top_prompt_messages"] = sorted(top_messages, key=lambda item: int(item["chars"]), reverse=True)[:8]
    return totals


def _run(command: Command, cwd: Path, *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout, shell=isinstance(command, str), check=False)


def _hidden_python(workspace: Path, code: str) -> bool:
    result = _run([sys.executable, "-c", code], workspace, timeout=120)
    return result.returncode == 0


def _python_test_cmd() -> str:
    return _shell_join([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])


def _shell_join(parts: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(parts)
    return " ".join(shlex.quote(part) for part in parts)


def _python_command(code: str) -> str:
    return _shell_join([sys.executable, "-c", code])


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _standard_test_import(module: str) -> str:
    return (
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
        f"from {module} import *\n\n"
    )


def _tool_success(session: dict[str, Any], name: str) -> bool:
    return any(result.get("ok") is True for result in tool_results(session, name))


def _status_or_fail_closed(ctx: BenchmarkContext, condition: bool) -> str:
    if condition:
        return "pass"
    return "fail_closed" if is_fail_closed_message(final_assistant_message(ctx.session)) else "fail"


def _git_status_short(workspace: Path) -> list[str]:
    result = _run(["git", "status", "--short"], workspace, timeout=60)
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def _changed_files(workspace: Path) -> list[str]:
    files: list[str] = []
    for line in _git_status_short(workspace):
        path = line[3:].strip() if len(line) > 3 else line.strip()
        if path:
            files.append(path.replace("\\", "/"))
    return sorted(set(files))


def _validator_output(ctx: BenchmarkContext, message: str) -> str:
    return message[:1200]


def _run_default_tests(workspace: Path) -> bool:
    return _run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], workspace, timeout=180).returncode == 0


def prepare_issue_fix_hidden_tests(workspace: Path) -> None:
    _write(workspace / "src" / "calculator.py", "def add(left: int, right: int) -> int:\n    return left - right\n")
    _write(
        workspace / "tests" / "test_calculator.py",
        _standard_test_import("calculator")
        + "import unittest\n\n\nclass CalculatorTests(unittest.TestCase):\n"
        + "    def test_adds_positive_numbers(self) -> None:\n        self.assertEqual(add(2, 3), 5)\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )


def validate_issue_fix_hidden_tests(ctx: BenchmarkContext) -> str:
    hidden = "import sys; sys.path.insert(0, 'src'); from calculator import add; assert add(-2, 5) == 3; assert add(0, 0) == 0"
    return "pass" if _hidden_python(ctx.workspace, hidden) else "fail"


def prepare_multi_file_refactor(workspace: Path) -> None:
    _write(workspace / "src" / "pricing.py", "def total(prices: list[int]) -> int:\n    return sum(prices)\n")
    _write(
        workspace / "tests" / "test_pricing.py",
        _standard_test_import("pricing")
        + "import unittest\n\n\nclass PricingTests(unittest.TestCase):\n"
        + "    def test_cart_total(self) -> None:\n        self.assertEqual(cart_total([2, 3, 4]), 9)\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )
    _write(workspace / "docs" / "pricing.md", "Call `total(prices)` to compute a cart total.\n")


def validate_multi_file_refactor(ctx: BenchmarkContext) -> str:
    source = (ctx.workspace / "src" / "pricing.py").read_text(encoding="utf-8")
    docs = (ctx.workspace / "docs" / "pricing.md").read_text(encoding="utf-8")
    hidden = "import sys; sys.path.insert(0, 'src'); from pricing import cart_total; assert cart_total([1, 2, 7]) == 10"
    if "def cart_total" not in source or "`total(prices)`" in docs or "cart_total" not in docs:
        return "fail"
    return "pass" if _hidden_python(ctx.workspace, hidden) and _run_default_tests(ctx.workspace) else "fail"


def prepare_large_repo_symbol_nav(workspace: Path) -> None:
    for index in range(30):
        _write(workspace / "src" / f"distractor_{index}.py", "\n\n".join(f"def helper_{index}_{n}():\n    return {n}" for n in range(20)) + "\n")
    before = "\n\n".join(f"def pre_{index}():\n    return {index}" for index in range(160))
    after = "\n\n".join(f"def post_{index}():\n    return {index}" for index in range(160))
    target = "def calculate_discount(cart):\n    marker = 'BENCH_SYMBOL_TOKEN_817'\n    return marker\n"
    _write(workspace / "src" / "large_pricing.py", f"{before}\n\n{target}\n\n{after}\n")


def validate_large_repo_symbol_nav(ctx: BenchmarkContext) -> str:
    calls = tool_calls(ctx.session)
    reads = tool_results(ctx.session, "read_symbol")
    if "read_file" in calls or "search_symbols" not in calls or "read_symbol" not in calls:
        return "fail"
    ok = any(result.get("ok") and "BENCH_SYMBOL_TOKEN_817" in str(result.get("output", "")) for result in reads)
    return "pass" if ok and "BENCH_SYMBOL_TOKEN_817" in ctx.stdout else "fail"


def prepare_instructed_edit(workspace: Path) -> None:
    _write(workspace / "src" / "formatter.py", "def normalize_email(value: str) -> str:\n    return value.strip()\n")
    _write(
        workspace / "tests" / "test_formatter.py",
        _standard_test_import("formatter")
        + "import unittest\n\n\nclass FormatterTests(unittest.TestCase):\n"
        + "    def test_normalizes_email(self) -> None:\n        self.assertEqual(normalize_email('  A@EXAMPLE.COM  '), 'a@example.com')\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )


def validate_instructed_edit(ctx: BenchmarkContext) -> str:
    hidden = "import sys; sys.path.insert(0, 'src'); from formatter import normalize_email; assert normalize_email('\\tUSER@EXAMPLE.COM ') == 'user@example.com'"
    return "pass" if _hidden_python(ctx.workspace, hidden) and _run_default_tests(ctx.workspace) else "fail"


def prepare_terminal_artifact_task(workspace: Path) -> None:
    _write(workspace / "data" / "names.txt", "zara\namy\nzara\nbob\n")


def validate_terminal_artifact_task(ctx: BenchmarkContext) -> str:
    target = ctx.workspace / "scratch" / "names_sorted.txt"
    if not target.exists():
        return "fail"
    return "pass" if target.read_text(encoding="utf-8") == "amy\nbob\nzara\n" and _tool_success(ctx.session, "run_shell") else "fail"


def prepare_test_repair_task(workspace: Path) -> None:
    _write(workspace / "src" / "slug.py", "def slugify(value: str) -> str:\n    return value.strip().lower()\n")
    _write(
        workspace / "tests" / "test_slug.py",
        _standard_test_import("slug")
        + "import unittest\n\n\nclass SlugTests(unittest.TestCase):\n"
        + "    def test_slugifies_spaces(self) -> None:\n        self.assertEqual(slugify('Hello Local Model'), 'hello-local-model')\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )


def validate_test_repair_task(ctx: BenchmarkContext) -> str:
    hidden = "import sys; sys.path.insert(0, 'src'); from slug import slugify; assert slugify('  Many   Spaces ') == 'many-spaces'"
    return "pass" if _hidden_python(ctx.workspace, hidden) and _run_default_tests(ctx.workspace) else "fail"


def prepare_forbidden_tool_efficiency(workspace: Path) -> None:
    commit_all(workspace, "checkpoint before forbidden benchmark")
    _write(workspace / "src" / "app.py", "def meaning() -> int:\n    return 99\n")


def validate_forbidden_tool_efficiency(ctx: BenchmarkContext) -> str:
    calls = tool_calls(ctx.session)
    diffs = tool_results(ctx.session, "git_diff")
    condition = (
        "read_file" not in calls
        and _tool_success(ctx.session, "git_status")
        and any(result.get("ok") and "return 99" in str(result.get("output", "")) for result in diffs)
        and "return 99" in ctx.stdout
    )
    return _status_or_fail_closed(ctx, condition)


def prepare_multi_turn_session_task(workspace: Path) -> None:
    _write(workspace / "src" / "session_task.py", "def session_value() -> str:\n    return 'todo'\n")
    _write(
        workspace / "tests" / "test_session_task.py",
        _standard_test_import("session_task")
        + "import unittest\n\n\nclass SessionTaskTests(unittest.TestCase):\n"
        + "    def test_session_value(self) -> None:\n        self.assertEqual(session_value(), 'SESSION_OK')\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )


def validate_multi_turn_session_task(ctx: BenchmarkContext) -> str:
    hidden = "import sys; sys.path.insert(0, 'src'); from session_task import session_value; assert session_value() == 'SESSION_OK'"
    messages = ctx.session.get("messages", [])
    has_all_turns = sum(1 for message in messages if isinstance(message, dict) and message.get("role") == "user") >= 3
    return "pass" if has_all_turns and _hidden_python(ctx.workspace, hidden) and _tool_success(ctx.session, "run_test") else "fail"


def prepare_cross_language_smoke(workspace: Path) -> None:
    _write(workspace / "src" / "math.js", "export function double(n) {\n  return n + n;\n}\n")


def validate_cross_language_smoke(ctx: BenchmarkContext) -> str:
    source = (ctx.workspace / "src" / "math.js").read_text(encoding="utf-8")
    return "pass" if "return n * 2" in source and "search_symbols" in tool_calls(ctx.session) else "fail"


def prepare_regression_token_traps(workspace: Path) -> None:
    body = "\n\n".join(f"def helper_{index}():\n    return {index}" for index in range(120))
    target = "def trap_value():\n    return 314159\n"
    _write(workspace / "src" / "trap.py", f"{body}\n\n{target}\n")


def validate_regression_token_traps(ctx: BenchmarkContext) -> str:
    calls = tool_call_args(ctx.session, "read_symbol")
    duplicate = any(calls[index] == calls[index - 1] for index in range(1, len(calls)))
    if duplicate or "read_file" in tool_calls(ctx.session):
        return "fail"
    return "pass" if "314159" in ctx.stdout and "search_symbols" in tool_calls(ctx.session) else "fail"


def prepare_replace_all_refactor(workspace: Path) -> None:
    _write(workspace / "src" / "flags.py", "OLD_FLAG = 'old'\n\ndef flag_name() -> str:\n    return OLD_FLAG\n")
    _write(
        workspace / "tests" / "test_flags.py",
        _standard_test_import("flags")
        + "import unittest\n\n\nclass FlagTests(unittest.TestCase):\n"
        + "    def test_new_flag(self) -> None:\n        self.assertEqual(flag_name(), 'new')\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )


def validate_replace_all_refactor(ctx: BenchmarkContext) -> str:
    source = (ctx.workspace / "src" / "flags.py").read_text(encoding="utf-8")
    return "pass" if "OLD_FLAG = 'new'" in source and _run_default_tests(ctx.workspace) else "fail"


def prepare_docs_sync_after_api_change(workspace: Path) -> None:
    _write(workspace / "src" / "api.py", "def fetch_user(user_id: str) -> dict[str, str]:\n    return {'id': user_id}\n")
    _write(workspace / "docs" / "api.md", "`fetch_user(user_id)` returns a user dict.\n")


def validate_docs_sync_after_api_change(ctx: BenchmarkContext) -> str:
    source = (ctx.workspace / "src" / "api.py").read_text(encoding="utf-8")
    docs = (ctx.workspace / "docs" / "api.md").read_text(encoding="utf-8")
    return "pass" if "include_orders" in source and "include_orders" in docs else "fail"


def prepare_nested_package_import_fix(workspace: Path) -> None:
    _write(workspace / "src" / "pkg" / "__init__.py", "")
    _write(workspace / "src" / "pkg" / "core.py", "from helpers import label\n\ndef wrapped() -> str:\n    return label('ok')\n")
    _write(workspace / "src" / "pkg" / "helpers.py", "def label(value: str) -> str:\n    return f'[{value}]'\n")
    _write(
        workspace / "tests" / "test_pkg.py",
        "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
        + "import unittest\nfrom pkg.core import wrapped\n\n\nclass PackageTests(unittest.TestCase):\n"
        + "    def test_wrapped(self) -> None:\n        self.assertEqual(wrapped(), '[ok]')\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )


def validate_nested_package_import_fix(ctx: BenchmarkContext) -> str:
    return "pass" if _run_default_tests(ctx.workspace) else "fail"


def prepare_staged_vs_worktree_diff(workspace: Path) -> None:
    commit_all(workspace, "checkpoint before staged diff")
    _write(workspace / "src" / "app.py", "def meaning() -> int:\n    return 11\n")
    subprocess.run(["git", "add", "src/app.py"], cwd=workspace, capture_output=True, text=True, check=True)
    _write(workspace / "src" / "app.py", "def meaning() -> int:\n    return 22\n")


def validate_staged_vs_worktree_diff(ctx: BenchmarkContext) -> str:
    diffs = tool_results(ctx.session, "git_diff")
    condition = any(result.get("ok") and "return 22" in str(result.get("output", "")) for result in diffs) and "return 22" in ctx.stdout
    return _status_or_fail_closed(ctx, condition)


def prepare_large_file_targeted_read(workspace: Path) -> None:
    lines = [f"line {index}: filler" for index in range(1, 701)]
    lines[419] = "line 420: BENCH_NEEDLE_420"
    _write(workspace / "docs" / "huge.md", "\n".join(lines) + "\n")


def validate_large_file_targeted_read(ctx: BenchmarkContext) -> str:
    targeted = False
    for args in tool_call_args(ctx.session, "read_file"):
        path = str(args.get("path", "")).replace("\\", "/")
        start = int(args.get("start", 1))
        end = int(args.get("end", 9999))
        if path.endswith("docs/huge.md") and start <= 420 <= end and (end - start) <= 80:
            targeted = True
    return "pass" if targeted and "BENCH_NEEDLE_420" in ctx.stdout else "fail"


def validate_exact_literal_write_readback(ctx: BenchmarkContext) -> str:
    target = ctx.workspace / "scratch" / "bench_exact.txt"
    return "pass" if target.exists() and target.read_text(encoding="utf-8") == "BENCH_EXACT\n" and "BENCH_EXACT" in ctx.stdout else "fail"


def validate_path_escape_error(ctx: BenchmarkContext) -> str:
    text = ctx.stdout + "\n" + final_assistant_message(ctx.session)
    return "pass" if "escapes the workspace" in text and any("escapes the workspace" in str(result.get("summary", "")) for result in tool_results(ctx.session, "read_file")) else "fail"


def validate_shell_failure_exact_command(ctx: BenchmarkContext) -> str:
    shells = tool_results(ctx.session, "run_shell")
    return "pass" if any(result.get("exit_code") == 5 and "bench-boom" in str(result.get("output", "")) for result in shells) and "5" in ctx.stdout else "fail"


def validate_run_test_summary(ctx: BenchmarkContext) -> str:
    runs = tool_results(ctx.session, "run_test")
    return "pass" if any(result.get("ok") and "OK" in str(result.get("output", "")) for result in runs) and "Tests passed" in ctx.stdout else "fail"


def validate_code_outline_summary(ctx: BenchmarkContext) -> str:
    calls = tool_calls(ctx.session)
    outlines = tool_results(ctx.session, "code_outline")
    return "pass" if "code_outline" in calls and any(result.get("ok") and "meaning" in str(result.get("output", "")) for result in outlines) else "fail"


ZERO_LLM = BenchmarkBudget(max_llm_calls=0, max_total_tokens=0)
SMALL_BUDGET_OFF = BenchmarkBudget(max_llm_calls=10, max_total_tokens=70_000)
SMALL_BUDGET_ON = BenchmarkBudget(max_llm_calls=14, max_total_tokens=110_000)
TERMINAL_ARTIFACT_COMMAND = _python_command(
    "from pathlib import Path; names=sorted(set(Path('data/names.txt').read_text().splitlines())); Path('scratch/names_sorted.txt').write_text(chr(10).join(names)+chr(10))"
)
SHELL_FAILURE_COMMAND = _python_command("import sys; print('bench-boom'); sys.exit(5)")


LOCAL_CASES: list[BenchmarkCase] = [
    BenchmarkCase(
        name="issue_fix_hidden_tests",
        suite="local-small",
        turns=("Issue: src/calculator.py add(left, right) returns the wrong value. Inspect source/tests, fix it, run tests, and summarize changed files.",),
        prepare=prepare_issue_fix_hidden_tests,
        validate=validate_issue_fix_hidden_tests,
        test_cmd=_python_test_cmd(),
        budget_off=BenchmarkBudget(max_llm_calls=12, max_total_tokens=80_000),
        budget_on=BenchmarkBudget(max_llm_calls=16, max_total_tokens=120_000),
    ),
    BenchmarkCase(
        name="multi_file_refactor",
        suite="local-small",
        turns=("Refactor the pricing API from total(prices) to cart_total(prices). Update src/pricing.py, tests, and docs/pricing.md. Run tests.",),
        prepare=prepare_multi_file_refactor,
        validate=validate_multi_file_refactor,
        test_cmd=_python_test_cmd(),
        budget_off=BenchmarkBudget(max_llm_calls=12, max_total_tokens=85_000),
        budget_on=BenchmarkBudget(max_llm_calls=16, max_total_tokens=130_000),
    ),
    BenchmarkCase(
        name="large_repo_symbol_nav",
        suite="local-small",
        turns=("Use search_symbols to find calculate_discount in src/large_pricing.py. Then use read_symbol on the exact match. Do not use read_file. Reply with the uppercase BENCH_SYMBOL token from that symbol only.",),
        benchmark_kind="tool_contract",
        prepare=prepare_large_repo_symbol_nav,
        validate=validate_large_repo_symbol_nav,
        budget_off=ZERO_LLM,
        budget_on=ZERO_LLM,
    ),
    BenchmarkCase(
        name="instructed_edit",
        suite="local-small",
        turns=("Please make normalize_email in src/formatter.py return stripped lowercase email text. Inspect first, edit, run tests, and keep the final short.",),
        prepare=prepare_instructed_edit,
        validate=validate_instructed_edit,
        test_cmd=_python_test_cmd(),
        budget_off=SMALL_BUDGET_OFF,
        budget_on=SMALL_BUDGET_ON,
    ),
    BenchmarkCase(
        name="terminal_artifact_task",
        suite="local-small",
        turns=(
            f"Use run_shell to execute exactly: {TERMINAL_ARTIFACT_COMMAND}. Then tell me what artifact was written.",
        ),
        benchmark_kind="tool_contract",
        prepare=prepare_terminal_artifact_task,
        validate=validate_terminal_artifact_task,
        budget_off=ZERO_LLM,
        budget_on=ZERO_LLM,
    ),
    BenchmarkCase(
        name="test_repair_task",
        suite="local-small",
        turns=("Run the tests, inspect the failure, fix slugify in src/slug.py so spaces collapse to single hyphens, rerun tests, and summarize.",),
        prepare=prepare_test_repair_task,
        validate=validate_test_repair_task,
        test_cmd=_python_test_cmd(),
        budget_off=BenchmarkBudget(max_llm_calls=12, max_total_tokens=85_000),
        budget_on=BenchmarkBudget(max_llm_calls=16, max_total_tokens=130_000),
    ),
    BenchmarkCase(
        name="forbidden_tool_efficiency",
        suite="local-small",
        turns=("Use git_status on src/app.py. Then use git_diff on src/app.py for the working tree only; omit cached or set cached to false. Do not use read_file. Reply with src/app.py and whether the diff adds return 99.",),
        benchmark_kind="tool_contract",
        prepare=prepare_forbidden_tool_efficiency,
        validate=validate_forbidden_tool_efficiency,
        acceptable=("pass", "fail_closed"),
        budget_off=ZERO_LLM,
        budget_on=ZERO_LLM,
    ),
    BenchmarkCase(
        name="multi_turn_session_task",
        suite="local-small",
        turns=(
            "Use search_symbols to find session_value in src/session_task.py. Reply with the function name only.",
            "Change session_value in src/session_task.py to return 'SESSION_OK' instead of 'todo'. Then run tests.",
            "Use run_test and reply whether tests pass.",
        ),
        benchmark_kind="tool_contract",
        prepare=prepare_multi_turn_session_task,
        validate=validate_multi_turn_session_task,
        test_cmd=_python_test_cmd(),
        budget_off=BenchmarkBudget(max_llm_calls=20, max_total_tokens=140_000),
        budget_on=BenchmarkBudget(max_llm_calls=28, max_total_tokens=210_000),
        timeout=900,
    ),
    BenchmarkCase(
        name="cross_language_smoke",
        suite="local-full",
        turns=("Use search_symbols and read_symbol on src/math.js, then change double(n) so it returns n * 2 instead of n + n. Do not use read_file.",),
        benchmark_kind="tool_contract",
        prepare=prepare_cross_language_smoke,
        validate=validate_cross_language_smoke,
        budget_off=SMALL_BUDGET_OFF,
        budget_on=SMALL_BUDGET_ON,
    ),
    BenchmarkCase(
        name="regression_token_traps",
        suite="local-full",
        turns=("Use search_symbols to locate trap_value in src/trap.py, then use read_symbol once on the exact match. Do not use read_file. Reply with the numeric return value only.",),
        benchmark_kind="tool_contract",
        prepare=prepare_regression_token_traps,
        validate=validate_regression_token_traps,
        budget_off=ZERO_LLM,
        budget_on=ZERO_LLM,
    ),
    BenchmarkCase(
        name="replace_all_refactor",
        suite="local-full",
        turns=("Change src/flags.py so flag_name() returns 'new' while keeping the existing function name. Run tests.",),
        prepare=prepare_replace_all_refactor,
        validate=validate_replace_all_refactor,
        test_cmd=_python_test_cmd(),
        budget_off=SMALL_BUDGET_OFF,
        budget_on=SMALL_BUDGET_ON,
    ),
    BenchmarkCase(
        name="docs_sync_after_api_change",
        suite="local-full",
        turns=("Add an optional include_orders: bool = False parameter to fetch_user in src/api.py and update docs/api.md with that parameter. No tests are needed.",),
        prepare=prepare_docs_sync_after_api_change,
        validate=validate_docs_sync_after_api_change,
        budget_off=SMALL_BUDGET_OFF,
        budget_on=SMALL_BUDGET_ON,
    ),
    BenchmarkCase(
        name="nested_package_import_fix",
        suite="local-full",
        turns=("Run tests, fix the package import bug in src/pkg/core.py, rerun tests, and summarize.",),
        prepare=prepare_nested_package_import_fix,
        validate=validate_nested_package_import_fix,
        test_cmd=_python_test_cmd(),
        budget_off=SMALL_BUDGET_OFF,
        budget_on=SMALL_BUDGET_ON,
    ),
    BenchmarkCase(
        name="staged_vs_worktree_diff",
        suite="local-full",
        turns=("Use git_status on src/app.py, then git_diff on src/app.py for working tree only, not cached. Do not use read_file. Tell me whether the working-tree diff adds return 22.",),
        benchmark_kind="tool_contract",
        prepare=prepare_staged_vs_worktree_diff,
        validate=validate_staged_vs_worktree_diff,
        acceptable=("pass", "fail_closed"),
        budget_off=ZERO_LLM,
        budget_on=ZERO_LLM,
    ),
    BenchmarkCase(
        name="large_file_targeted_read",
        suite="local-full",
        turns=("Use read_file on docs/huge.md with the smallest useful line range around line 420, then reply with the exact marker token on that line only.",),
        benchmark_kind="tool_contract",
        prepare=prepare_large_file_targeted_read,
        validate=validate_large_file_targeted_read,
        budget_off=ZERO_LLM,
        budget_on=ZERO_LLM,
    ),
    BenchmarkCase(
        name="exact_literal_write_readback",
        suite="local-full",
        turns=("Create scratch/bench_exact.txt with exactly the single line BENCH_EXACT followed by a newline. Then use read_file to confirm it and reply with BENCH_EXACT only.",),
        benchmark_kind="tool_contract",
        validate=validate_exact_literal_write_readback,
        budget_off=ZERO_LLM,
        budget_on=ZERO_LLM,
    ),
    BenchmarkCase(
        name="path_escape_error",
        suite="local-full",
        turns=("Use read_file on ../outside.txt and tell me the exact tool error.",),
        benchmark_kind="tool_contract",
        validate=validate_path_escape_error,
        budget_off=ZERO_LLM,
        budget_on=ZERO_LLM,
    ),
    BenchmarkCase(
        name="shell_failure_exact_command",
        suite="local-full",
        turns=(f"Use run_shell to execute exactly: {SHELL_FAILURE_COMMAND}. Then tell me the exit code and printed word.",),
        benchmark_kind="tool_contract",
        validate=validate_shell_failure_exact_command,
        budget_off=ZERO_LLM,
        budget_on=ZERO_LLM,
    ),
    BenchmarkCase(
        name="run_test_summary",
        suite="local-full",
        turns=("Use run_test and tell me whether tests passed and which test module ran.",),
        benchmark_kind="tool_contract",
        validate=validate_run_test_summary,
        test_cmd=_python_test_cmd(),
        budget_off=ZERO_LLM,
        budget_on=ZERO_LLM,
    ),
    BenchmarkCase(
        name="code_outline_summary",
        suite="local-full",
        turns=("Use code_outline on src/app.py and tell me which function is defined there.",),
        benchmark_kind="tool_contract",
        validate=validate_code_outline_summary,
        budget_off=BenchmarkBudget(max_llm_calls=4, max_total_tokens=30_000),
        budget_on=BenchmarkBudget(max_llm_calls=6, max_total_tokens=45_000),
    ),
]


def selected_cases(suite: str, requested: set[str] | None = None) -> list[BenchmarkCase]:
    if suite == "local-small":
        cases = [case for case in LOCAL_CASES if case.suite == "local-small"]
    elif suite == "local-full":
        cases = list(LOCAL_CASES)
    else:
        cases = []
    if requested:
        cases = [case for case in cases if case.name in requested]
    return cases


def _load_session_if_exists(session_file: Path) -> dict[str, Any]:
    if not session_file.exists():
        return {"events": [], "messages": []}
    return load_session(session_file)


def evaluate_case(
    repo_root: Path,
    model: str,
    verifier_model: str | None,
    mode: str,
    case: BenchmarkCase,
    timeout: int,
) -> dict[str, Any]:
    with workspace_temp_dir("ollama-code-bench-", repo_root) as tmp:
        workspace = Path(tmp)
        build_workspace(workspace)
        if case.prepare is not None:
            case.prepare(workspace)
        session_file = workspace / "scratch" / f"{case.name}-{mode}.json"
        results: list[subprocess.CompletedProcess[str]] = []
        started = time.perf_counter()
        for index, prompt in enumerate(case.turns):
            extra_args = ["--debate", mode, "--quiet"]
            if mode == "on":
                extra_args.extend(["--max-tool-rounds", "24"])
            if verifier_model:
                extra_args.extend(["--verifier-model", verifier_model])
            if case.test_cmd:
                extra_args.extend(["--test-cmd", case.test_cmd])
            if index > 0:
                extra_args.extend(["--resume", str(session_file)])
            try:
                result = run_cli(
                    repo_root,
                    workspace,
                    model,
                    prompt,
                    session_file=session_file,
                    timeout=case.timeout or timeout,
                    extra_args=extra_args,
                )
            except subprocess.TimeoutExpired as exc:
                elapsed = time.perf_counter() - started
                session = _load_session_if_exists(session_file)
                return {
                    "case": case.name,
                    "suite": case.suite,
                    "benchmark_kind": case.benchmark_kind,
                    "model": model,
                    "verifier_model": verifier_model,
                    "debate": mode,
                    "status": "fail",
                    "acceptable": list(case.acceptable),
                    "latency_s": round(elapsed, 2),
                    "usage": usage_totals(session),
                    "tool_calls": tool_calls(session),
                    "failed_tools": failed_tools(session),
                    "assumption_audits": event_count(session, "assumption_audit"),
                    "assumption_audit_retries": event_count(session, "assumption_audit", verdict="retry"),
                    "verification_retries": event_count(session, "verification", verdict="retry"),
                    "verification_rewrites": event_count(session, "verification_rewrite"),
                    "changed_files": _changed_files(workspace),
                    "tests_run": tests_run(session),
                    "validator_output": _validator_output(BenchmarkContext(workspace, session, "", str(exc), tuple(), tuple(), case), "timeout"),
                    "final": final_assistant_message(session),
                    "stdout_tail": "",
                    "stderr_tail": str(exc)[-1200:],
                }
            results.append(result)
            if result.returncode != 0:
                break
        elapsed = time.perf_counter() - started
        session = _load_session_if_exists(session_file)
        ctx = BenchmarkContext(
            workspace=workspace,
            session=session,
            stdout="\n".join(result.stdout for result in results),
            stderr="\n".join(result.stderr for result in results),
            returncodes=tuple(result.returncode for result in results),
            results=tuple(results),
            case=case,
        )
        status = case.validate(ctx)
        if any(code != 0 for code in ctx.returncodes):
            status = "fail"
        outcome = {
            "case": case.name,
            "suite": case.suite,
            "benchmark_kind": case.benchmark_kind,
            "model": model,
            "verifier_model": verifier_model,
            "debate": mode,
            "status": status,
            "acceptable": list(case.acceptable),
            "latency_s": round(elapsed, 2),
            "usage": usage_totals(session),
            "tool_calls": tool_calls(session),
            "failed_tools": failed_tools(session),
            "assumption_audits": event_count(session, "assumption_audit"),
            "assumption_audit_retries": event_count(session, "assumption_audit", verdict="retry"),
            "verification_retries": event_count(session, "verification", verdict="retry"),
            "verification_rewrites": event_count(session, "verification_rewrite"),
            "changed_files": _changed_files(workspace),
            "tests_run": tests_run(session),
            "validator_output": _validator_output(ctx, status),
            "final": final_assistant_message(session),
            "stdout_tail": ctx.stdout[-1200:],
            "stderr_tail": ctx.stderr[-1200:],
        }
        return outcome


def failed_tools(session: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for event in session.get("events", []):
        if event.get("type") != "tool_result":
            continue
        result = event.get("result")
        if not isinstance(result, dict) or result.get("ok") is not False:
            continue
        failures.append(
            {
                "name": event.get("name"),
                "summary": str(result.get("summary") or result.get("output") or "")[:240],
            }
        )
    return failures


def event_count(session: dict[str, Any], event_type: str, *, verdict: str | None = None) -> int:
    count = 0
    for event in session.get("events", []):
        if event.get("type") != event_type:
            continue
        if verdict is not None and event.get("verdict") != verdict:
            continue
        count += 1
    return count


def tests_run(session: dict[str, Any]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for result in tool_results(session, "run_test"):
        runs.append({"ok": result.get("ok"), "command": result.get("command"), "output_tail": str(result.get("output", ""))[-300:]})
    return runs


def budget_for(case: BenchmarkCase, mode: str) -> BenchmarkBudget:
    return case.budget_on if mode == "on" else case.budget_off


def budget_violations(outcome: dict[str, Any], case: BenchmarkCase) -> list[str]:
    budget = budget_for(case, str(outcome.get("debate", "off")))
    usage = outcome.get("usage") if isinstance(outcome.get("usage"), dict) else {}
    violations: list[str] = []
    llm_calls = int(usage.get("llm_calls", 0))
    total_tokens = int(usage.get("total_tokens", 0))
    if budget.max_llm_calls is not None and llm_calls > budget.max_llm_calls:
        violations.append(f"llm_calls {llm_calls}>{budget.max_llm_calls}")
    if budget.max_total_tokens is not None and total_tokens > budget.max_total_tokens:
        violations.append(f"total_tokens {total_tokens}>{budget.max_total_tokens}")
    return violations


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_tokens = [int(item["usage"]["total_tokens"]) for item in results if isinstance(item.get("usage"), dict)]
    by_kind: dict[str, dict[str, int]] = {}
    for item in results:
        kind = str(item.get("benchmark_kind") or "unknown")
        bucket = by_kind.setdefault(kind, {"runs": 0, "pass": 0, "fail_closed": 0, "fail": 0})
        bucket["runs"] += 1
        status = str(item.get("status"))
        if status in {"pass", "fail_closed", "fail"}:
            bucket[status] += 1
    return {
        "runs": len(results),
        "pass": sum(1 for item in results if item.get("status") == "pass"),
        "fail_closed": sum(1 for item in results if item.get("status") == "fail_closed"),
        "fail": sum(1 for item in results if item.get("status") == "fail"),
        "total_llm_calls": sum(int(item["usage"]["llm_calls"]) for item in results if isinstance(item.get("usage"), dict)),
        "total_tokens": sum(total_tokens),
        "median_total_tokens": median(total_tokens),
        "by_benchmark_kind": dict(sorted(by_kind.items())),
    }


def median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def comparison_rows(current: list[dict[str, Any]], baseline: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if baseline is None:
        return []
    index = {
        (item.get("suite"), item.get("model"), item.get("verifier_model"), item.get("debate"), item.get("case")): item
        for item in baseline
        if isinstance(item, dict)
    }
    rows: list[dict[str, Any]] = []
    for item in current:
        key = (item.get("suite"), item.get("model"), item.get("verifier_model"), item.get("debate"), item.get("case"))
        before = index.get(key)
        if before is None:
            continue
        before_usage = before.get("usage") if isinstance(before.get("usage"), dict) else {}
        after_usage = item.get("usage") if isinstance(item.get("usage"), dict) else {}
        before_tokens = int(before_usage.get("total_tokens", 0))
        after_tokens = int(after_usage.get("total_tokens", 0))
        delta_pct = 0.0 if before_tokens == 0 else round((after_tokens - before_tokens) * 100 / before_tokens, 1)
        rows.append(
            {
                "suite": item.get("suite"),
                "model": item.get("model"),
                "verifier_model": item.get("verifier_model"),
                "debate": item.get("debate"),
                "case": item.get("case"),
                "before_status": before.get("status"),
                "after_status": item.get("status"),
                "before_total_tokens": before_tokens,
                "after_total_tokens": after_tokens,
                "total_token_delta_pct": delta_pct,
                "before_llm_calls": before_usage.get("llm_calls", 0),
                "after_llm_calls": after_usage.get("llm_calls", 0),
                "before_latency_s": before.get("latency_s"),
                "after_latency_s": item.get("latency_s"),
            }
        )
    return rows


def print_table(results: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> None:
    for item in results:
        usage = item["usage"]
        print(
            "[coding-bench]"
            f" suite={item['suite']}"
            f" kind={item.get('benchmark_kind') or '-'}"
            f" model={item.get('model') or '-'}"
            f" verifier={item.get('verifier_model') or '-'}"
            f" debate={item.get('debate') or '-'}"
            f" case={item['case']}"
            f" status={item['status']}"
            f" calls={usage.get('llm_calls', 0)}"
            f" prompt={usage.get('prompt_tokens', 0)}"
            f" output={usage.get('output_tokens', 0)}"
            f" total={usage.get('total_tokens', 0)}"
            f" latency_s={item['latency_s']}"
            f" tools={','.join(item.get('tool_calls', [])) or '-'}"
        )
    if comparisons:
        print("[coding-bench] comparison")
        for row in comparisons:
            print(
                "[coding-bench]"
                f" suite={row['suite']}"
                f" model={row['model']}"
                f" verifier={row['verifier_model'] or '-'}"
                f" debate={row['debate']}"
                f" case={row['case']}"
                f" status={row['before_status']}->{row['after_status']}"
                f" total_tokens={row['before_total_tokens']}->{row['after_total_tokens']}"
                f" delta_pct={row['total_token_delta_pct']}"
                f" calls={row['before_llm_calls']}->{row['after_llm_calls']}"
            )


def external_smoke_results() -> list[dict[str, Any]]:
    checks = [
        ("terminal_bench_preflight", ["tb", "--help"], "terminal-bench CLI"),
        ("docker_preflight", ["docker", "--version"], "Docker"),
        ("uv_preflight", ["uv", "--version"], "uv"),
        ("swe_bench_preflight", [sys.executable, "-m", "swebench.harness.run_evaluation", "--help"], "SWE-bench harness"),
    ]
    results: list[dict[str, Any]] = []
    for name, command, label in checks:
        started = time.perf_counter()
        if shutil.which(command[0]) is None and command[0] != sys.executable:
            status = "fail_closed"
            stdout = ""
            stderr = f"{label} is not installed."
        else:
            result = _run(command, Path.cwd(), timeout=30)
            status = "pass" if result.returncode == 0 else "fail_closed"
            stdout = result.stdout
            stderr = result.stderr
        results.append(
            {
                "case": name,
                "suite": "external-smoke",
                "model": None,
                "verifier_model": None,
                "debate": None,
                "status": status,
                "acceptable": ["pass", "fail_closed"],
                "latency_s": round(time.perf_counter() - started, 2),
                "usage": usage_totals({"events": []}),
                "tool_calls": [],
                "failed_tools": [],
                "assumption_audits": 0,
                "assumption_audit_retries": 0,
                "verification_retries": 0,
                "verification_rewrites": 0,
                "changed_files": [],
                "tests_run": [],
                "validator_output": stderr[:600] or stdout[:600],
                "final": "",
                "stdout_tail": stdout[-1200:],
                "stderr_tail": stderr[-1200:],
            }
        )
    return results


def _git_commit(repo_root: Path) -> str:
    result = _run(["git", "rev-parse", "--short", "HEAD"], repo_root, timeout=30)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def write_results_payload(
    output: Path,
    *,
    repo_root: Path,
    suite: str,
    results: list[dict[str, Any]],
    comparisons: list[dict[str, Any]] | None = None,
    accuracy_regressions: list[dict[str, Any]] | None = None,
    budget_failures: list[dict[str, Any]] | None = None,
    partial: bool = False,
) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "suite": suite,
        "partial": partial,
        "summary": summarize(results),
        "results": results,
        "comparisons": comparisons or [],
        "accuracy_regressions": accuracy_regressions or [],
        "budget_failures": budget_failures or [],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run serial coding accuracy + token-efficiency benchmarks.")
    parser.add_argument("--suite", choices=["local-small", "local-full", "external-smoke"], default="local-small")
    parser.add_argument("--models", nargs="+", default=["gemma3:4b", "qwen3:8b", "granite4.1:8b"], help="Primary models for local suites.")
    parser.add_argument("--verifier-pairs", nargs="*", default=[], help="Optional primary=verifier entries, debate-on only.")
    parser.add_argument("--modes", nargs="+", choices=["off", "on"], default=["off", "on"])
    parser.add_argument("--cases", nargs="*", default=None, help="Case names to run.")
    parser.add_argument("--output", default=None, help="Raw JSON output path. Defaults under scratch/coding-benchmark/.")
    parser.add_argument("--compare", default=None, help="Optional prior JSON path for token/accuracy deltas.")
    parser.add_argument("--timeout", type=int, default=600, help="Per-turn timeout in seconds.")
    parser.add_argument("--strict-accuracy", action="store_true", help="Fail if any run status is not acceptable.")
    parser.add_argument("--strict-budget", action="store_true", help="Fail if a local case exceeds its token or LLM-call budget.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    output = Path(args.output) if args.output else repo_root / "scratch" / "coding-benchmark" / f"{args.suite}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.suite == "external-smoke":
        results = external_smoke_results()
        comparisons: list[dict[str, Any]] = []
        accuracy_regressions: list[dict[str, Any]] = []
        budget_failures: list[dict[str, Any]] = []
        print_table(results, [])
    else:
        requested_cases = set(args.cases) if args.cases else None
        cases = selected_cases(args.suite, requested_cases)
        if not cases:
            raise SystemExit("No benchmark cases selected.")
        integrity_violations = [
            {"case": case.name, "findings": prompt_integrity_findings(case)}
            for case in cases
            if prompt_integrity_findings(case)
        ]
        if integrity_violations:
            raise SystemExit(f"Coding-accuracy prompt integrity violation(s): {json.dumps(integrity_violations)}")
        cases_by_name = {case.name: case for case in cases}
        available = set(installed_models())
        matrix: list[tuple[str, str | None, list[str]]] = []
        for requested in args.models:
            model = resolve_requested_model(requested, available)
            if model is not None:
                matrix.append((model, None, list(args.modes)))
        for pair in args.verifier_pairs:
            if "=" not in pair:
                raise SystemExit(f"Bad verifier pair {pair!r}; use primary=verifier.")
            primary_raw, verifier_raw = pair.split("=", 1)
            primary = resolve_requested_model(primary_raw, available)
            verifier = resolve_requested_model(verifier_raw, available)
            if primary is not None and verifier is not None:
                matrix.append((primary, verifier, ["on"]))
        if not matrix:
            raise SystemExit("No requested models are installed.")
        results = []
        for model, verifier_model, modes in matrix:
            _LOADED_MODELS.add(model)
            if verifier_model:
                _LOADED_MODELS.add(verifier_model)
            for mode in modes:
                for case in cases:
                    outcome = evaluate_case(repo_root, model, verifier_model, mode, case, args.timeout)
                    results.append(outcome)
                    print_table([outcome], [])
                    write_results_payload(output, repo_root=repo_root, suite=args.suite, results=results, partial=True)
            unload_model(model)
            _LOADED_MODELS.discard(model)
            if verifier_model:
                unload_model(verifier_model)
                _LOADED_MODELS.discard(verifier_model)

        baseline_results = None
        if args.compare:
            baseline_payload = json.loads(Path(args.compare).read_text(encoding="utf-8"))
            baseline_results = baseline_payload.get("results") if isinstance(baseline_payload.get("results"), list) else []
        comparisons = comparison_rows(results, baseline_results)
        accuracy_regressions = [row for row in comparisons if row.get("before_status") == "pass" and row.get("after_status") != "pass"]
        budget_failures = []
        if args.strict_budget:
            for outcome in results:
                case = cases_by_name.get(str(outcome.get("case")))
                if case is None:
                    continue
                violations = budget_violations(outcome, case)
                if violations:
                    budget_failures.append({**outcome, "budget_violations": violations})
        if comparisons:
            print_table([], comparisons)

    write_results_payload(
        output,
        repo_root=repo_root,
        suite=args.suite,
        results=results,
        comparisons=comparisons,
        accuracy_regressions=accuracy_regressions,
        budget_failures=budget_failures,
        partial=False,
    )

    failures: list[Any] = []
    if args.strict_accuracy:
        failures.extend([item for item in results if item.get("status") not in set(item.get("acceptable", ["pass"]))])
    failures.extend(accuracy_regressions)
    if args.strict_budget:
        failures.extend(budget_failures)
    if failures:
        print(f"[coding-bench] strict failures: {len(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
