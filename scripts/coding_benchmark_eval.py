from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from urllib.parse import urlsplit
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
FEATURE_PROFILES = ("baseline", "schema", "context-pack", "evidence-handles", "num-predict-caps", "structured-edits", "trajectory-guards", "contract-guards", "all")
BENCHMARK_CLASSES = ("agent", "controller")
UNKNOWN_BENCHMARK_CLASS = "unknown"
TRANSIENT_OLLAMA_TIMEOUT_MARKER = "error: Ollama timed out after "
MAX_TRANSIENT_OLLAMA_TIMEOUT_RETRIES = 1
FAIL_CLOSED_MESSAGES = {
    "Stopped because grounded final verification could not accept a final answer.",
    "Stopped because assumption audit could not approve a next tool step.",
    "Stopped because artifact reconciliation could not approve a repair path.",
    "Stopped after reaching the maximum tool rounds.",
}


def _public_hard_slug_patterns() -> tuple[str, ...]:
    fallback = (
        "list-ops",
        "pig-latin",
        "wordy",
        "phone-number",
        "grade-school",
        "variable-length-quantity",
        "robot-name",
        "simple-linked-list",
        "transpose",
        "scale-generator",
    )
    try:
        from scripts import public_benchmark_eval as public_bench
    except (ModuleNotFoundError, ImportError):
        try:
            scripts_root = Path(__file__).resolve().parent
            repo_root = scripts_root.parent
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            import public_benchmark_eval as public_bench
        except ModuleNotFoundError:
            return fallback
    if not hasattr(public_bench, "HARD_POLYGLOT_TASKS"):
        return fallback
    slugs = tuple(getattr(public_bench, "HARD_POLYGLOT_TASKS"))  # type: ignore[assignment]
    if not slugs:
        return fallback
    return slugs


def _public_benchmark_patterns() -> tuple[str, ...]:
    try:
        try:
            from scripts import public_benchmark_eval as public_bench
        except ModuleNotFoundError:
            import public_benchmark_eval as public_bench
        return tuple(public_bench.public_task_set("expanded"))
    except Exception:
        return _public_hard_slug_patterns()


def unload_model(model: str) -> None:
    subprocess.run(["ollama", "stop", model], capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)


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
    requires_git: bool = False


def benchmark_class_for_kind(kind: str | None) -> str:
    if kind == "coding_accuracy":
        return "agent"
    if kind == "tool_contract":
        return "controller"
    return UNKNOWN_BENCHMARK_CLASS


def benchmark_class_for_case(case: BenchmarkCase) -> str:
    return benchmark_class_for_kind(case.benchmark_kind)


def benchmark_class_for_outcome(outcome: dict[str, Any]) -> str:
    value = outcome.get("benchmark_class")
    if isinstance(value, str) and value in BENCHMARK_CLASSES:
        return value
    kind = outcome.get("benchmark_kind")
    return benchmark_class_for_kind(kind if isinstance(kind, str) else None)


def prompt_integrity_findings(case: BenchmarkCase) -> list[str]:
    if case.benchmark_kind != "coding_accuracy":
        return []
    text = "\n".join(case.turns)
    slugs = [slug for slug in _public_benchmark_patterns() if isinstance(slug, str)]
    slug_pattern = "|".join(re.escape(slug) for slug in sorted(set(slug.lower() for slug in slugs)))
    module_names = sorted({slug for slug in slugs} | {slug.replace("-", "_") for slug in slugs})
    module_pattern = "|".join(re.escape(name) for name in module_names)
    checks = {
        "synthetic marker token": r"\b(?:BENCH|TOKEN|NEEDLE|EXACT)_[A-Z0-9_]+\b",
        "public benchmark task slug": rf"(?i)\b(?:{slug_pattern})\b",
        "public benchmark module name": rf"(?i)\b(?:{module_pattern})(?:\.py)?\b",
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


def tool_profile(session: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    totals: dict[str, dict[str, Any]] = {}
    for event in session.get("events", []):
        if event.get("type") != "tool_result":
            continue
        name = str(event.get("name") or "")
        if not name:
            continue
        raw_duration = event.get("duration_ms")
        duration_ms = float(raw_duration) if isinstance(raw_duration, (int, float)) else 0.0
        result = event.get("result") if isinstance(event.get("result"), dict) else {}
        rows.append(
            {
                "name": name,
                "duration_ms": round(duration_ms, 3),
                "cached": event.get("cached") is True,
                "ok": result.get("ok") is True,
                "summary": str(result.get("summary") or result.get("output") or "")[:160],
            }
        )
        bucket = totals.setdefault(name, {"calls": 0, "duration_ms": 0.0, "failed": 0, "cached": 0})
        bucket["calls"] += 1
        bucket["duration_ms"] += duration_ms
        if result.get("ok") is False:
            bucket["failed"] += 1
        if event.get("cached") is True:
            bucket["cached"] += 1
    return {
        "total_duration_ms": round(sum(float(item["duration_ms"]) for item in rows), 3),
        "by_tool": {name: {**bucket, "duration_ms": round(float(bucket["duration_ms"]), 3)} for name, bucket in sorted(totals.items())},
        "calls": rows,
    }


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


def _git_status_short(workspace: Path) -> list[str] | None:
    env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(workspace.parent)}
    result = subprocess.run(["git", "status", "--short"], cwd=workspace, capture_output=True, text=True, timeout=60, check=False, env=env)
    if result.returncode != 0:
        return None
    return [line for line in result.stdout.splitlines() if line.strip()]


def _workspace_snapshot(workspace: Path) -> dict[str, bytes]:
    ignored_roots = {".git", ".ollama-code", "scratch", "__pycache__"}
    snapshot: dict[str, bytes] = {}
    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(workspace).as_posix()
        if any(part in ignored_roots for part in Path(rel).parts):
            continue
        snapshot[rel] = path.read_bytes()
    return snapshot


def _snapshot_changed_files(before: dict[str, bytes], workspace: Path) -> list[str]:
    after = _workspace_snapshot(workspace)
    changed = [path for path in sorted(set(before) | set(after)) if before.get(path) != after.get(path)]
    return changed


def _changed_files_payload(workspace: Path, *, before_snapshot: dict[str, bytes] | None = None) -> dict[str, Any]:
    git_status = _git_status_short(workspace)
    if git_status is not None:
        return {"changed_files": _changed_files_from_git_status(git_status), "changed_files_source": "git_status"}
    if before_snapshot is not None:
        return {"changed_files": _snapshot_changed_files(before_snapshot, workspace), "changed_files_source": "snapshot_diff"}
    return {"changed_files": [], "changed_files_source": "unavailable"}


def _changed_files_from_git_status(lines: list[str]) -> list[str]:
    files: list[str] = []
    for line in lines:
        path = line[3:].strip() if len(line) > 3 else line.strip()
        if path:
            files.append(path.replace("\\", "/"))
    return sorted(set(files))


def _validator_output(ctx: BenchmarkContext, message: str) -> str:
    return message[:1200]


def _run_default_tests(workspace: Path) -> bool:
    return _run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], workspace, timeout=180).returncode == 0


def _hidden_python_and_default_tests(workspace: Path, code: str) -> bool:
    runner = (
        "import unittest\n"
        f"{code}\n"
        "suite = unittest.defaultTestLoader.discover('tests')\n"
        "result = unittest.TextTestRunner(verbosity=2).run(suite)\n"
        "raise SystemExit(0 if result.wasSuccessful() else 1)\n"
    )
    return _run([sys.executable, "-c", runner], workspace, timeout=180).returncode == 0


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
    return "pass" if _hidden_python_and_default_tests(ctx.workspace, hidden) else "fail"


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
    return "pass" if _hidden_python_and_default_tests(ctx.workspace, hidden) else "fail"


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
    return "pass" if _hidden_python_and_default_tests(ctx.workspace, hidden) else "fail"


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


def prepare_docs_sync_without_tests_still_validates(workspace: Path) -> None:
    prepare_docs_sync_after_api_change(workspace)


def validate_docs_sync_without_tests_still_validates(ctx: BenchmarkContext) -> str:
    source = (ctx.workspace / "src" / "api.py").read_text(encoding="utf-8")
    docs = (ctx.workspace / "docs" / "api.md").read_text(encoding="utf-8")
    calls = tool_calls(ctx.session)
    return _status_or_fail_closed(
        ctx,
        "include_orders" in source
        and "include_orders" in docs
        and _tool_success(ctx.session, "lint_typecheck")
        and _tool_success(ctx.session, "contract_check")
        and "run_test" not in calls
        and "select_tests" not in calls,
    )


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


def prepare_bad_test_command_recovery(workspace: Path) -> None:
    _write(workspace / "src" / "inventory.py", "def total_units(counts: list[int]) -> int:\n    return sum(counts) - 1\n")
    _write(
        workspace / "tests" / "test_inventory.py",
        _standard_test_import("inventory")
        + "import unittest\n\n\nclass InventoryTests(unittest.TestCase):\n"
        + "    def test_sums_units(self) -> None:\n        self.assertEqual(total_units([2, 3, 4]), 9)\n"
        + "    def test_empty(self) -> None:\n        self.assertEqual(total_units([]), 0)\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )


def validate_bad_test_command_recovery(ctx: BenchmarkContext) -> str:
    hidden = "import sys; sys.path.insert(0, 'src'); from inventory import total_units; assert total_units([5, 0, 2]) == 7; assert total_units([1]) == 1"
    recovered = any(
        result.get("recovered") is True
        and "pytesst -q" in str(result.get("original_command", ""))
        and "unittest discover" in str(result.get("command", ""))
        for result in tool_results(ctx.session, "run_test")
    )
    return "pass" if recovered and _hidden_python_and_default_tests(ctx.workspace, hidden) else "fail"


def prepare_renamed_simple_expression_hidden(workspace: Path) -> None:
    _write(workspace / "src" / "scoreboard.py", "def score_delta(base: int, bonus: int) -> int:\n    pass\n")
    _write(
        workspace / "tests" / "test_scoreboard.py",
        _standard_test_import("scoreboard")
        + "import unittest\n\n\nclass ScoreboardTests(unittest.TestCase):\n"
        + "    def test_positive_values(self) -> None:\n        self.assertEqual(score_delta(2, 3), 5)\n"
        + "    def test_mixed_values(self) -> None:\n        self.assertEqual(score_delta(-1, 4), 3)\n"
        + "    def test_zero_values(self) -> None:\n        self.assertEqual(score_delta(0, 0), 0)\n"
        + "    def test_negative_values(self) -> None:\n        self.assertEqual(score_delta(-5, -2), -7)\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )


def validate_renamed_simple_expression_hidden(ctx: BenchmarkContext) -> str:
    hidden = "import sys; sys.path.insert(0, 'src'); from scoreboard import score_delta; assert score_delta(9, -4) == 5; assert score_delta(-8, 3) == -5"
    return "pass" if _hidden_python_and_default_tests(ctx.workspace, hidden) else "fail"


def prepare_renamed_prefix_rotation_hidden(workspace: Path) -> None:
    _write(workspace / "src" / "syllables.py", "def transform_words(text: str) -> str:\n    pass\n")
    _write(
        workspace / "tests" / "test_syllables.py",
        _standard_test_import("syllables")
        + "import unittest\n\n\nclass SyllableTests(unittest.TestCase):\n"
        + "    def test_word_beginning_with_a_vowel(self) -> None:\n        self.assertEqual(transform_words('apple'), 'appleay')\n"
        + "    def test_word_beginning_with_p(self) -> None:\n        self.assertEqual(transform_words('pig'), 'igpay')\n"
        + "    def test_word_beginning_with_qu(self) -> None:\n        self.assertEqual(transform_words('queen'), 'eenquay')\n"
        + "    def test_word_with_consonant_before_qu(self) -> None:\n        self.assertEqual(transform_words('square'), 'aresquay')\n"
        + "    def test_word_beginning_with_xr(self) -> None:\n        self.assertEqual(transform_words('xray'), 'xrayay')\n"
        + "    def test_y_after_consonant_cluster(self) -> None:\n        self.assertEqual(transform_words('rhythm'), 'ythmrhay')\n"
        + "    def test_phrase(self) -> None:\n        self.assertEqual(transform_words('quick fast run'), 'ickquay astfay unray')\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )


def validate_renamed_prefix_rotation_hidden(ctx: BenchmarkContext) -> str:
    hidden = (
        "import sys; sys.path.insert(0, 'src'); from syllables import transform_words; "
        "assert transform_words('therapy square apple') == 'erapythay aresquay appleay'; "
        "assert transform_words('rhythm pig') == 'ythmrhay igpay'"
    )
    return "pass" if _hidden_python_and_default_tests(ctx.workspace, hidden) else "fail"


def prepare_renamed_word_arithmetic_hidden(workspace: Path) -> None:
    _write(workspace / "src" / "story_solver.py", "def solve(question):\n    pass\n")
    _write(
        workspace / "tests" / "test_story_solver.py",
        _standard_test_import("story_solver")
        + "import unittest\n\n\nclass StorySolverTests(unittest.TestCase):\n"
        + "    def test_just_a_number(self) -> None:\n        self.assertEqual(solve('What is 5?'), 5)\n"
        + "    def test_addition(self) -> None:\n        self.assertEqual(solve('What is 1 plus 1?'), 2)\n"
        + "    def test_subtraction(self) -> None:\n        self.assertEqual(solve('What is 4 minus -12?'), 16)\n"
        + "    def test_multiplication(self) -> None:\n        self.assertEqual(solve('What is -3 multiplied by 25?'), -75)\n"
        + "    def test_division(self) -> None:\n        self.assertEqual(solve('What is 33 divided by -3?'), -11)\n"
        + "    def test_multiple_operations(self) -> None:\n        self.assertEqual(solve('What is 17 minus 6 plus 3?'), 14)\n"
        + "    def test_unknown_operation(self) -> None:\n"
        + "        with self.assertRaises(ValueError) as err:\n            solve('What is 52 cubed?')\n"
        + "        self.assertEqual(err.exception.args[0], 'unknown operation')\n"
        + "    def test_syntax_error(self) -> None:\n"
        + "        with self.assertRaises(ValueError) as err:\n            solve('What is 1 plus?')\n"
        + "        self.assertEqual(err.exception.args[0], 'syntax error')\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )


def validate_renamed_word_arithmetic_hidden(ctx: BenchmarkContext) -> str:
    hidden = (
        "import sys; sys.path.insert(0, 'src'); from story_solver import solve; "
        "assert solve('What is 10 minus 3 multiplied by 2?') == 14; "
        "assert solve('What is -6 divided by 3 plus 5?') == 3"
    )
    return "pass" if _hidden_python_and_default_tests(ctx.workspace, hidden) else "fail"


def prepare_renamed_text_matrix_hidden(workspace: Path) -> None:
    _write(workspace / "src" / "text_grid.py", "def flip_text(block):\n    pass\n")
    _write(
        workspace / "tests" / "test_text_grid.py",
        _standard_test_import("text_grid")
        + "import unittest\n\n\nclass TextGridTests(unittest.TestCase):\n"
        + "    def test_empty(self) -> None:\n        self.assertEqual(flip_text(''), '')\n"
        + "    def test_single_row(self) -> None:\n        self.assertEqual(flip_text('A1'), 'A\\n1')\n"
        + "    def test_single_column(self) -> None:\n        self.assertEqual(flip_text('A\\n1'), 'A1')\n"
        + "    def test_square(self) -> None:\n        self.assertEqual(flip_text('ABC\\n123'), 'A1\\nB2\\nC3')\n"
        + "    def test_with_space(self) -> None:\n        self.assertEqual(flip_text('A B'), 'A\\n \\nB')\n"
        + "    def test_ragged(self) -> None:\n        self.assertEqual(flip_text('11\\n2\\n3333\\n444\\n555555\\n66666'), '123456\\n1 3456\\n  3456\\n  3 56\\n    56\\n    5')\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )


def validate_renamed_text_matrix_hidden(ctx: BenchmarkContext) -> str:
    hidden = (
        "import sys; sys.path.insert(0, 'src'); from text_grid import flip_text; "
        "assert flip_text('AB\\n12\\nxy') == 'A1x\\nB2y'; "
        "assert flip_text('NO\\nUP') == 'NU\\nOP'"
    )
    return "pass" if _hidden_python_and_default_tests(ctx.workspace, hidden) else "fail"


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


def prepare_single_file_literal_read(workspace: Path) -> None:
    _write(workspace / "notes" / "hello.txt", "hello bench\n")


def validate_single_file_literal_read(ctx: BenchmarkContext) -> str:
    calls = tool_calls(ctx.session)
    reads = tool_results(ctx.session, "read_file")
    used_read = len(calls) == 1 and calls[0] == "read_file"
    readback_ok = any(result.get("ok") and "hello bench" in str(result.get("output", "")) for result in reads)
    no_llm = usage_totals(ctx.session).get("llm_calls", 0) == 0
    return "pass" if used_read and readback_ok and no_llm and "hello bench" in ctx.stdout else "fail"


def validate_discover_validators_natural(ctx: BenchmarkContext) -> str:
    calls = tool_calls(ctx.session)
    validators = tool_results(ctx.session, "discover_validators")
    no_llm = usage_totals(ctx.session).get("llm_calls", 0) == 0
    return "pass" if calls == ["discover_validators"] and validators and no_llm and "test python:" in ctx.stdout else "fail"


def validate_search_then_run_test_summary(ctx: BenchmarkContext) -> str:
    calls = tool_calls(ctx.session)
    no_llm = usage_totals(ctx.session).get("llm_calls", 0) == 0
    return (
        "pass"
        if calls == ["search", "run_test"] and no_llm and "docs/guide.md contains the match." in ctx.stdout and "Tests passed: yes" in ctx.stdout
        else "fail"
    )


def prepare_search_then_git_status(workspace: Path) -> None:
    _write(workspace / "src" / "app.py", "def meaning() -> int:\n    return 99\n")


def validate_search_then_git_status(ctx: BenchmarkContext) -> str:
    calls = tool_calls(ctx.session)
    no_llm = usage_totals(ctx.session).get("llm_calls", 0) == 0
    return (
        "pass"
        if calls == ["search", "git_status"] and no_llm and "docs/guide.md contains the match." in ctx.stdout and "src/app.py" in ctx.stdout
        else "fail"
    )


def validate_list_files_then_git_status(ctx: BenchmarkContext) -> str:
    calls = tool_calls(ctx.session)
    no_llm = usage_totals(ctx.session).get("llm_calls", 0) == 0
    return "pass" if calls == ["list_files", "git_status"] and no_llm and "docs/guide.md" in ctx.stdout and "src/app.py" in ctx.stdout else "fail"


def validate_discover_validators_then_lint(ctx: BenchmarkContext) -> str:
    calls = tool_calls(ctx.session)
    no_llm = usage_totals(ctx.session).get("llm_calls", 0) == 0
    lint_ok = _tool_success(ctx.session, "lint_typecheck")
    return (
        "pass"
        if calls == ["discover_validators", "lint_typecheck"] and no_llm and lint_ok and "test python:" in ctx.stdout
        else "fail"
    )


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
        requires_git=True,
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
        turns=("Update the OLD_FLAG constant in src/flags.py to 'new' while keeping flag_name() as the public function. Run tests.",),
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
        name="docs_sync_without_tests_still_validates",
        suite="local-full",
        turns=("Add an optional include_orders: bool = False parameter to fetch_user in src/api.py and update docs/api.md with that parameter, but do not run tests.",),
        prepare=prepare_docs_sync_without_tests_still_validates,
        validate=validate_docs_sync_without_tests_still_validates,
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
        name="bad_test_command_recovery",
        suite="local-full",
        turns=("Run tests, fix src/inventory.py so total_units is correct, rerun tests, and summarize briefly.",),
        prepare=prepare_bad_test_command_recovery,
        validate=validate_bad_test_command_recovery,
        test_cmd="pytesst -q",
        budget_off=SMALL_BUDGET_OFF,
        budget_on=SMALL_BUDGET_ON,
    ),
    BenchmarkCase(
        name="renamed_simple_expression_hidden",
        suite="local-full",
        turns=("Implement src/scoreboard.py from the tests. Read source and tests, replace stubs with complete code, run tests, and summarize briefly.",),
        prepare=prepare_renamed_simple_expression_hidden,
        validate=validate_renamed_simple_expression_hidden,
        test_cmd=_python_test_cmd(),
        budget_off=SMALL_BUDGET_OFF,
        budget_on=SMALL_BUDGET_ON,
    ),
    BenchmarkCase(
        name="renamed_prefix_rotation_hidden",
        suite="local-full",
        turns=("Implement src/syllables.py from the tests. Read source and tests, replace stubs with complete code, run tests, and summarize briefly.",),
        prepare=prepare_renamed_prefix_rotation_hidden,
        validate=validate_renamed_prefix_rotation_hidden,
        test_cmd=_python_test_cmd(),
        budget_off=SMALL_BUDGET_OFF,
        budget_on=SMALL_BUDGET_ON,
    ),
    BenchmarkCase(
        name="renamed_word_arithmetic_hidden",
        suite="local-full",
        turns=("Implement src/story_solver.py from the tests. Read source and tests, replace stubs with complete code, run tests, and summarize briefly.",),
        prepare=prepare_renamed_word_arithmetic_hidden,
        validate=validate_renamed_word_arithmetic_hidden,
        test_cmd=_python_test_cmd(),
        budget_off=SMALL_BUDGET_OFF,
        budget_on=SMALL_BUDGET_ON,
    ),
    BenchmarkCase(
        name="renamed_text_matrix_hidden",
        suite="local-full",
        turns=("Implement src/text_grid.py from the tests. Read source and tests, replace stubs with complete code, run tests, and summarize briefly.",),
        prepare=prepare_renamed_text_matrix_hidden,
        validate=validate_renamed_text_matrix_hidden,
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
        requires_git=True,
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
        name="single_file_literal_read",
        suite="local-full",
        turns=("What does notes/hello.txt say?",),
        benchmark_kind="tool_contract",
        prepare=prepare_single_file_literal_read,
        validate=validate_single_file_literal_read,
        budget_off=ZERO_LLM,
        budget_on=ZERO_LLM,
    ),
    BenchmarkCase(
        name="discover_validators_natural",
        suite="local-full",
        turns=("List the test and validation commands for this repo.",),
        benchmark_kind="tool_contract",
        validate=validate_discover_validators_natural,
        budget_off=ZERO_LLM,
        budget_on=ZERO_LLM,
    ),
    BenchmarkCase(
        name="search_then_run_test_summary",
        suite="local-full",
        turns=("Search for TOKEN_42 in the repo, then run tests and tell me whether tests passed.",),
        benchmark_kind="tool_contract",
        validate=validate_search_then_run_test_summary,
        test_cmd=_python_test_cmd(),
        budget_off=ZERO_LLM,
        budget_on=ZERO_LLM,
    ),
    BenchmarkCase(
        name="search_then_git_status",
        suite="local-full",
        turns=("Search for TOKEN_42 in the repo, then show git status.",),
        benchmark_kind="tool_contract",
        prepare=prepare_search_then_git_status,
        validate=validate_search_then_git_status,
        budget_off=ZERO_LLM,
        budget_on=ZERO_LLM,
        requires_git=True,
    ),
    BenchmarkCase(
        name="search_and_git_status",
        suite="local-full",
        turns=("Search for TOKEN_42 in the repo and show git status.",),
        benchmark_kind="tool_contract",
        prepare=prepare_search_then_git_status,
        validate=validate_search_then_git_status,
        budget_off=ZERO_LLM,
        budget_on=ZERO_LLM,
        requires_git=True,
    ),
    BenchmarkCase(
        name="list_files_and_git_status",
        suite="local-full",
        turns=("List files in the workspace and show git status.",),
        benchmark_kind="tool_contract",
        prepare=prepare_search_then_git_status,
        validate=validate_list_files_then_git_status,
        budget_off=ZERO_LLM,
        budget_on=ZERO_LLM,
        requires_git=True,
    ),
    BenchmarkCase(
        name="discover_validators_then_lint",
        suite="local-full",
        turns=("List the test and validation commands for this repo, then run lint.",),
        benchmark_kind="tool_contract",
        validate=validate_discover_validators_then_lint,
        budget_off=ZERO_LLM,
        budget_on=ZERO_LLM,
    ),
    BenchmarkCase(
        name="discover_validators_and_lint",
        suite="local-full",
        turns=("List the test and validation commands for this repo and run lint.",),
        benchmark_kind="tool_contract",
        validate=validate_discover_validators_then_lint,
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


def selected_cases(suite: str, requested: set[str] | None = None, benchmark_classes: set[str] | None = None) -> list[BenchmarkCase]:
    if suite == "local-small":
        cases = [case for case in LOCAL_CASES if case.suite == "local-small"]
    elif suite == "local-full":
        cases = list(LOCAL_CASES)
    else:
        cases = []
    if requested:
        cases = [case for case in cases if case.name in requested]
    if benchmark_classes:
        cases = [case for case in cases if benchmark_class_for_case(case) in benchmark_classes]
    return cases


def _load_session_if_exists(session_file: Path) -> dict[str, Any]:
    if not session_file.exists():
        return {"events": [], "messages": []}
    return load_session(session_file)


def _benchmark_workspace_parent(repo_root: Path, *, requires_git: bool) -> Path:
    configured = os.environ.get("OLLAMA_CODE_BENCH_WORKSPACE_ROOT", "").strip()
    if configured:
        return Path(configured)
    if requires_git:
        fallback = Path.home() / ".codex" / "memories" / "ollama-code-bench-temp"
        try:
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback
        except OSError:
            return repo_root
    return repo_root


def default_benchmark_jobs() -> int:
    configured = os.environ.get("OLLAMA_CODE_BENCH_JOBS", "").strip()
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            return 1
    host = os.environ.get("OLLAMA_HOST", "").strip()
    if not host:
        return 1
    normalized = host if host.startswith(("http://", "https://")) else f"http://{host}"
    try:
        parts = urlsplit(normalized)
    except ValueError:
        return 8
    if parts.hostname in {"127.0.0.1", "localhost", "::1"} and parts.port == 11437:
        return 12
    return 8


def _outcome_is_transient_ollama_timeout(outcome: dict[str, Any]) -> bool:
    if str(outcome.get("status") or "") != "fail":
        return False
    stderr_tail = str(outcome.get("stderr_tail") or "")
    return TRANSIENT_OLLAMA_TIMEOUT_MARKER in stderr_tail


def _evaluate_case_once(
    repo_root: Path,
    model: str,
    verifier_model: str | None,
    mode: str,
    case: BenchmarkCase,
    timeout: int,
    reconcile: str = "auto",
    feature_profile: str = "baseline",
) -> dict[str, Any]:
    benchmark_class = benchmark_class_for_case(case)
    workspace_parent = _benchmark_workspace_parent(repo_root, requires_git=case.requires_git)
    with workspace_temp_dir("ollama-code-bench-", workspace_parent) as tmp:
        workspace = Path(tmp)
        git_available = build_workspace(workspace, init_git=case.requires_git)
        if case.requires_git and not git_available:
            return {
                "case": case.name,
                "suite": case.suite,
                "benchmark_kind": case.benchmark_kind,
                "benchmark_class": benchmark_class,
                "model": model,
                "verifier_model": verifier_model,
                "debate": mode,
                "reconcile": reconcile,
                "feature_profile": feature_profile,
                "status": "skip",
                "acceptable": sorted(set(case.acceptable) | {"skip"}),
                "skip_reason": "nested git workspace unavailable",
                "latency_s": 0.0,
                "usage": {"llm_calls": 0, "prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "tool_calls": [],
                "tool_profile": {},
                "failed_tools": [],
                "assumption_audits": 0,
                "assumption_audit_retries": 0,
                "reconciliations": 0,
                "reconciliation_retries": 0,
                "verification_retries": 0,
                "verification_rewrites": 0,
                "changed_files": [],
                "tests_run": [],
                "validator_output": "skipped: nested git workspace unavailable",
                "final": "",
                "stdout_tail": "",
                "stderr_tail": "",
            }
        if case.prepare is not None:
            case.prepare(workspace)
        before_snapshot = _workspace_snapshot(workspace)
        session_file = workspace / "scratch" / f"{case.name}-{mode}-{reconcile}.json"
        results: list[subprocess.CompletedProcess[str]] = []
        started = time.perf_counter()
        for index, prompt in enumerate(case.turns):
            extra_args = ["--debate", mode, "--reconcile", reconcile, "--quiet"]
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
                    extra_env={
                        "GIT_CEILING_DIRECTORIES": str(workspace.parent),
                        "OLLAMA_CODE_FEATURE_PROFILE": feature_profile,
                    },
                    require_llm_for_turn=benchmark_class == "agent",
                )
            except subprocess.TimeoutExpired as exc:
                elapsed = time.perf_counter() - started
                session = _load_session_if_exists(session_file)
                return {
                    "case": case.name,
                    "suite": case.suite,
                    "benchmark_kind": case.benchmark_kind,
                    "benchmark_class": benchmark_class,
                    "model": model,
                    "verifier_model": verifier_model,
                    "debate": mode,
                    "reconcile": reconcile,
                    "feature_profile": feature_profile,
                    "status": "fail",
                    "acceptable": list(case.acceptable),
                    "latency_s": round(elapsed, 2),
                    "usage": usage_totals(session),
                    "tool_calls": tool_calls(session),
                    "tool_profile": tool_profile(session),
                    "failed_tools": failed_tools(session),
                    "assumption_audits": event_count(session, "assumption_audit"),
                    "assumption_audit_retries": event_count(session, "assumption_audit", verdict="retry"),
                    "reconciliations": event_count(session, "reconciliation"),
                    "reconciliation_retries": event_count(session, "reconciliation", verdict="retry"),
                    "verification_retries": event_count(session, "verification", verdict="retry"),
                    "verification_rewrites": event_count(session, "verification_rewrite"),
                    **_changed_files_payload(workspace, before_snapshot=before_snapshot),
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
        final_message = final_assistant_message(session)
        if is_fail_closed_message(final_message):
            status = "fail_closed"
        outcome = {
            "case": case.name,
            "suite": case.suite,
            "benchmark_kind": case.benchmark_kind,
            "benchmark_class": benchmark_class,
            "model": model,
            "verifier_model": verifier_model,
            "debate": mode,
            "reconcile": reconcile,
            "feature_profile": feature_profile,
            "status": status,
            "acceptable": list(case.acceptable),
            "latency_s": round(elapsed, 2),
            "usage": usage_totals(session),
            "tool_calls": tool_calls(session),
            "tool_profile": tool_profile(session),
            "failed_tools": failed_tools(session),
            "assumption_audits": event_count(session, "assumption_audit"),
            "assumption_audit_retries": event_count(session, "assumption_audit", verdict="retry"),
            "reconciliations": event_count(session, "reconciliation"),
            "reconciliation_retries": event_count(session, "reconciliation", verdict="retry"),
            "verification_retries": event_count(session, "verification", verdict="retry"),
            "verification_rewrites": event_count(session, "verification_rewrite"),
            **_changed_files_payload(workspace, before_snapshot=before_snapshot),
            "tests_run": tests_run(session),
            "validator_output": _validator_output(ctx, status),
            "final": final_message,
            "stdout_tail": ctx.stdout[-1200:],
            "stderr_tail": ctx.stderr[-1200:],
        }
        return outcome


def evaluate_case(
    repo_root: Path,
    model: str,
    verifier_model: str | None,
    mode: str,
    case: BenchmarkCase,
    timeout: int,
    reconcile: str = "auto",
    feature_profile: str = "baseline",
) -> dict[str, Any]:
    benchmark_class = benchmark_class_for_case(case)
    retries_used = 0
    total_attempts = 1 + (MAX_TRANSIENT_OLLAMA_TIMEOUT_RETRIES if benchmark_class == "agent" else 0)
    for attempt in range(total_attempts):
        outcome = _evaluate_case_once(
            repo_root,
            model,
            verifier_model,
            mode,
            case,
            timeout,
            reconcile=reconcile,
            feature_profile=feature_profile,
        )
        if attempt < total_attempts - 1 and _outcome_is_transient_ollama_timeout(outcome):
            retries_used += 1
            continue
        if retries_used:
            outcome["transient_ollama_retries"] = retries_used
        return outcome
    raise AssertionError("benchmark evaluation exhausted without returning an outcome")


def evaluate_case_batch(
    repo_root: Path,
    model: str,
    verifier_model: str | None,
    mode: str,
    cases: list[BenchmarkCase],
    timeout: int,
    *,
    reconcile: str = "auto",
    feature_profile: str = "baseline",
    jobs: int = 1,
) -> list[dict[str, Any]]:
    if not cases:
        return []
    if jobs <= 1 or len(cases) <= 1:
        return [
            evaluate_case(
                repo_root,
                model,
                verifier_model,
                mode,
                case,
                timeout,
                reconcile=reconcile,
                feature_profile=feature_profile,
            )
            for case in cases
        ]
    ordered: list[dict[str, Any] | None] = [None] * len(cases)
    with ThreadPoolExecutor(max_workers=min(jobs, len(cases))) as executor:
        future_to_index = {
            executor.submit(
                evaluate_case,
                repo_root,
                model,
                verifier_model,
                mode,
                case,
                timeout,
                reconcile=reconcile,
                feature_profile=feature_profile,
            ): index
            for index, case in enumerate(cases)
        }
        for future in as_completed(future_to_index):
            ordered[future_to_index[future]] = future.result()
    return [outcome for outcome in ordered if outcome is not None]


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


def llm_bypass_failures(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for outcome in results:
        if benchmark_class_for_outcome(outcome) != "agent":
            continue
        usage = outcome.get("usage") if isinstance(outcome.get("usage"), dict) else {}
        if int(usage.get("llm_calls", 0)) != 0:
            continue
        failures.append({**outcome, "llm_bypass_reason": "agent benchmark completed with zero LLM calls"})
    return failures


def _summary_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_tokens = [int(item["usage"]["total_tokens"]) for item in rows if isinstance(item.get("usage"), dict)]
    return {
        "runs": len(rows),
        "pass": sum(1 for item in rows if item.get("status") == "pass"),
        "fail_closed": sum(1 for item in rows if item.get("status") == "fail_closed"),
        "fail": sum(1 for item in rows if item.get("status") == "fail"),
        "skip": sum(1 for item in rows if item.get("status") == "skip"),
        "total_llm_calls": sum(int(item["usage"]["llm_calls"]) for item in rows if isinstance(item.get("usage"), dict)),
        "total_tokens": sum(total_tokens),
        "median_total_tokens": median(total_tokens),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_tokens = [int(item["usage"]["total_tokens"]) for item in results if isinstance(item.get("usage"), dict)]
    by_kind_rows: dict[str, list[dict[str, Any]]] = {}
    by_class_rows: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        kind = str(item.get("benchmark_kind") or "unknown")
        by_kind_rows.setdefault(kind, []).append(item)
        by_class_rows.setdefault(benchmark_class_for_outcome(item), []).append(item)
    return {
        "runs": len(results),
        "pass": sum(1 for item in results if item.get("status") == "pass"),
        "fail_closed": sum(1 for item in results if item.get("status") == "fail_closed"),
        "fail": sum(1 for item in results if item.get("status") == "fail"),
        "skip": sum(1 for item in results if item.get("status") == "skip"),
        "total_llm_calls": sum(int(item["usage"]["llm_calls"]) for item in results if isinstance(item.get("usage"), dict)),
        "total_tokens": sum(total_tokens),
        "median_total_tokens": median(total_tokens),
        "by_benchmark_kind": {name: _summary_bucket(rows) for name, rows in sorted(by_kind_rows.items())},
        "by_benchmark_class": {name: _summary_bucket(rows) for name, rows in sorted(by_class_rows.items())},
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
        (
            item.get("suite"),
            item.get("model"),
            item.get("verifier_model"),
            item.get("debate"),
            item.get("reconcile"),
            item.get("feature_profile", "baseline"),
            benchmark_class_for_outcome(item),
            item.get("case"),
        ): item
        for item in baseline
        if isinstance(item, dict)
    }
    rows: list[dict[str, Any]] = []
    for item in current:
        key = (
            item.get("suite"),
            item.get("model"),
            item.get("verifier_model"),
            item.get("debate"),
            item.get("reconcile"),
            item.get("feature_profile", "baseline"),
            benchmark_class_for_outcome(item),
            item.get("case"),
        )
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
                "reconcile": item.get("reconcile"),
                "feature_profile": item.get("feature_profile", "baseline"),
                "benchmark_class": benchmark_class_for_outcome(item),
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
            f" class={benchmark_class_for_outcome(item)}"
            f" kind={item.get('benchmark_kind') or '-'}"
            f" model={item.get('model') or '-'}"
            f" verifier={item.get('verifier_model') or '-'}"
            f" debate={item.get('debate') or '-'}"
            f" reconcile={item.get('reconcile') or '-'}"
            f" profile={item.get('feature_profile', 'baseline')}"
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
                f" class={row.get('benchmark_class') or UNKNOWN_BENCHMARK_CLASS}"
                f" model={row['model']}"
                f" verifier={row['verifier_model'] or '-'}"
                f" debate={row['debate']}"
                f" reconcile={row.get('reconcile') or '-'}"
                f" profile={row.get('feature_profile', 'baseline')}"
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
                "benchmark_kind": "preflight",
                "benchmark_class": UNKNOWN_BENCHMARK_CLASS,
                "model": None,
                "verifier_model": None,
                "debate": None,
                "reconcile": None,
                "feature_profile": None,
                "status": status,
                "acceptable": ["pass", "fail_closed"],
                "latency_s": round(time.perf_counter() - started, 2),
                "usage": usage_totals({"events": []}),
                "tool_calls": [],
                "failed_tools": [],
                "assumption_audits": 0,
                "assumption_audit_retries": 0,
                "reconciliations": 0,
                "reconciliation_retries": 0,
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
    llm_bypass_failures: list[dict[str, Any]] | None = None,
    partial: bool = False,
) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "suite": suite,
        "partial": partial,
        "benchmark_classes": list(BENCHMARK_CLASSES),
        "summary": summarize(results),
        "results": results,
        "comparisons": comparisons or [],
        "accuracy_regressions": accuracy_regressions or [],
        "budget_failures": budget_failures or [],
        "llm_bypass_failures": llm_bypass_failures or [],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run serial coding accuracy + token-efficiency benchmarks.")
    parser.add_argument("--suite", choices=["local-small", "local-full", "external-smoke"], default="local-small")
    parser.add_argument("--models", nargs="+", default=["gemma4:e4b", "qwen3:8b", "granite4.1:8b"], help="Primary models for local suites.")
    parser.add_argument("--verifier-pairs", nargs="*", default=[], help="Optional primary=verifier entries, debate-on only.")
    parser.add_argument("--modes", nargs="+", choices=["off", "on"], default=["off", "on"])
    parser.add_argument("--reconcile-modes", nargs="+", choices=["off", "on", "auto"], default=["auto"], help="Artifact reconciliation modes to run.")
    parser.add_argument("--cases", nargs="*", default=None, help="Case names to run.")
    parser.add_argument("--output", default=None, help="Raw JSON output path. Defaults under scratch/coding-benchmark/.")
    parser.add_argument("--compare", default=None, help="Optional prior JSON path for token/accuracy deltas.")
    parser.add_argument("--feature-profiles", nargs="+", choices=FEATURE_PROFILES, default=["baseline"], help="A/B controller feature profiles.")
    parser.add_argument("--benchmark-classes", nargs="+", choices=BENCHMARK_CLASSES, default=list(BENCHMARK_CLASSES), help="Benchmark classes to run.")
    parser.add_argument("--timeout", type=int, default=600, help="Per-turn timeout in seconds.")
    parser.add_argument("--jobs", type=int, default=default_benchmark_jobs(), help="Parallel benchmark cases per model/profile.")
    parser.add_argument("--strict-accuracy", action="store_true", help="Fail if any run status is not acceptable.")
    parser.add_argument("--strict-budget", action="store_true", help="Fail if a local case exceeds its token or LLM-call budget.")
    parser.add_argument("--require-llm-for-agent-benchmarks", action="store_true", help="Fail if any agent benchmark completes with zero LLM calls.")
    parser.add_argument("--require-llm-for-coding-accuracy", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    output = Path(args.output) if args.output else repo_root / "scratch" / "coding-benchmark" / f"{args.suite}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.suite == "external-smoke":
        results = external_smoke_results()
        comparisons: list[dict[str, Any]] = []
        accuracy_regressions: list[dict[str, Any]] = []
        budget_failures: list[dict[str, Any]] = []
        llm_bypass_rows: list[dict[str, Any]] = []
        print_table(results, [])
    else:
        requested_cases = set(args.cases) if args.cases else None
        requested_classes = set(args.benchmark_classes)
        cases = selected_cases(args.suite, requested_cases, requested_classes)
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
            print("No requested models are installed.")
            return 1
        results = []
        for model, verifier_model, modes in matrix:
            _LOADED_MODELS.add(model)
            if verifier_model:
                _LOADED_MODELS.add(verifier_model)
            for mode in modes:
                for reconcile in args.reconcile_modes:
                    for feature_profile in args.feature_profiles:
                        batch_results = evaluate_case_batch(
                            repo_root,
                            model,
                            verifier_model,
                            mode,
                            cases,
                            args.timeout,
                            reconcile=reconcile,
                            feature_profile=feature_profile,
                            jobs=max(1, args.jobs),
                        )
                        for outcome in batch_results:
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
        require_llm_for_agent = bool(args.require_llm_for_agent_benchmarks or args.require_llm_for_coding_accuracy)
        llm_bypass_rows = llm_bypass_failures(results) if require_llm_for_agent else []
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
        llm_bypass_failures=llm_bypass_rows,
        partial=False,
    )

    failures: list[Any] = []
    if args.strict_accuracy:
        failures.extend([item for item in results if item.get("status") not in set(item.get("acceptable", ["pass"]))])
    failures.extend(accuracy_regressions)
    if args.strict_budget:
        failures.extend(budget_failures)
    if args.require_llm_for_agent_benchmarks or args.require_llm_for_coding_accuracy:
        failures.extend(llm_bypass_rows)
    if failures:
        print(f"[coding-bench] strict failures: {len(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
