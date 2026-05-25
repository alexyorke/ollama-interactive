from __future__ import annotations

import argparse
import atexit
import json
import os
import secrets
import shlex
import string
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Callable

try:
    from workspace_temp import workspace_temp_dir
except ModuleNotFoundError:
    from scripts.workspace_temp import workspace_temp_dir


_LOADED_MODELS: set[str] = set()


def unload_model(model: str) -> None:
    subprocess.run(["ollama", "stop", model], capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)


def _cleanup_loaded_models() -> None:
    for model in sorted(_LOADED_MODELS):
        unload_model(model)


atexit.register(_cleanup_loaded_models)


def init_git_repo(root: Path) -> bool:
    for command in (
        ["git", "init"],
        ["git", "config", "user.name", "Ollama Code Tests"],
        ["git", "config", "user.email", "tests@example.com"],
        ["git", "add", "."],
        ["git", "commit", "-m", "initial"],
    ):
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f"[e2e] git workspace unavailable; skipping git-only scenario: {' '.join(command)} failed")
            return False
    return True


def commit_all(root: Path, message: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=root, capture_output=True, text=True, check=True)
    status = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root, capture_output=True, text=True, check=False)
    if status.returncode != 0:
        subprocess.run(["git", "commit", "-m", message], cwd=root, capture_output=True, text=True, check=True)


def ollama_host() -> str:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host.rstrip("/")


def installed_models() -> list[str]:
    request = urllib.request.Request(f"{ollama_host()}/api/tags")
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    models = payload.get("models") if isinstance(payload, dict) else []
    names: list[str] = []
    for item in models:
        if isinstance(item, dict) and item.get("name"):
            names.append(str(item["name"]))
    return names


def resolve_model(requested: str, available: set[str]) -> str:
    if requested in available:
        return requested
    latest = f"{requested}:latest"
    if latest in available:
        return latest
    raise SystemExit(f"Requested model {requested!r} is not installed on {ollama_host()}. Available: {sorted(available)}")


def build_workspace(root: Path, *, init_git: bool = True) -> bool:
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "scratch").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / ".gitignore").write_text(".ollama-code/\n", encoding="utf-8")
    (root / "docs" / "guide.md").write_text(
        "# Guide\n\nTOKEN_42 lives here.\nThe CLI should read this file with tools.\n",
        encoding="utf-8",
    )
    (root / "src" / "app.py").write_text(
        "def meaning() -> int:\n    return 42\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_sample.py").write_text(
        "import unittest\n\n\nclass SampleTests(unittest.TestCase):\n    def test_truth(self) -> None:\n        self.assertEqual(6 * 7, 42)\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
        encoding="utf-8",
    )
    if not init_git:
        return False
    return init_git_repo(root)


def _shell_join(parts: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(parts)
    return " ".join(shlex.quote(part) for part in parts)


def _python_test_cmd() -> str:
    return _shell_join([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _nonce(prefix: str, *, width: int = 3) -> str:
    suffix = "".join(secrets.choice(string.digits) for _ in range(width))
    return f"{prefix}_{suffix}"


def _standard_test_import(module: str) -> str:
    return (
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
        f"from {module} import *\n\n"
    )


def _run(command: list[str], cwd: Path, *, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)


def _hidden_python(workspace: Path, code: str) -> bool:
    return _run([sys.executable, "-c", code], workspace, timeout=120).returncode == 0


def _run_default_tests(workspace: Path) -> bool:
    return _run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], workspace, timeout=180).returncode == 0


def run_cli(
    repo_root: Path,
    workspace: Path,
    model: str,
    prompt: str,
    *,
    approval: str = "auto",
    timeout: int = 420,
    session_file: Path | None = None,
    stdin_text: str | None = None,
    extra_args: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    require_llm_for_turn: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "ollama_code.cli",
        "--cwd",
        str(workspace),
        "--model",
        model,
        "--approval",
        approval,
        "--max-tool-rounds",
        "12",
        "--max-agent-depth",
        "2",
    ]
    if require_llm_for_turn:
        command.append("--require-llm-for-turn")
    if extra_args:
        command.extend(extra_args)
    if session_file is not None:
        command.extend(["--session-file", str(session_file)])
    command.append(prompt)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        input=stdin_text,
        timeout=timeout,
        env=env,
        check=False,
    )


def run_repl(
    repo_root: Path,
    workspace: Path,
    model: str,
    stdin_text: str,
    *,
    approval: str = "ask",
    timeout: int = 420,
    session_file: Path | None = None,
    require_llm_for_turn: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "ollama_code.cli",
        "--cwd",
        str(workspace),
        "--model",
        model,
        "--approval",
        approval,
        "--max-tool-rounds",
        "12",
        "--max-agent-depth",
        "2",
    ]
    if require_llm_for_turn:
        command.append("--require-llm-for-turn")
    if session_file is not None:
        command.extend(["--session-file", str(session_file)])
    return subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        input=stdin_text,
        timeout=timeout,
        env=os.environ.copy(),
        check=False,
    )


def load_session(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str, *, stdout: str = "", stderr: str = "", session: dict[str, Any] | None = None) -> None:
    if condition:
        return
    extra = []
    if stdout:
        extra.append(f"STDOUT:\n{stdout}")
    if stderr:
        extra.append(f"STDERR:\n{stderr}")
    if session is not None:
        extra.append(f"SESSION:\n{json.dumps(session, indent=2)}")
    suffix = "\n".join(extra)
    raise AssertionError(f"{message}\n{suffix}".rstrip())


def event_names(session: dict[str, Any]) -> list[str]:
    return [event.get("type", "") for event in session.get("events", [])]


def tool_results(session: dict[str, Any], tool_name: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for event in session.get("events", []):
        if event.get("type") == "tool_result" and event.get("name") == tool_name:
            result = event.get("result")
            if isinstance(result, dict):
                results.append(result)
    return results


def tool_calls(session: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for event in session.get("events", []):
        if event.get("type") == "tool_call" and isinstance(event.get("name"), str):
            names.append(str(event["name"]))
    return names


def llm_call_count(session: dict[str, Any]) -> int:
    return sum(1 for event in session.get("events", []) if event.get("type") == "llm_call")


def require_llm_used(
    session: dict[str, Any],
    *,
    minimum: int,
    message: str,
    stdout: str = "",
    stderr: str = "",
) -> None:
    require(llm_call_count(session) >= minimum, message, stdout=stdout, stderr=stderr, session=session)


def scenario_transcripted_tool_use(repo_root: Path, workspace: Path, model: str) -> None:
    session_file = workspace / "scratch" / "tool-use.json"
    result = run_cli(
        repo_root,
        workspace,
        model,
        "Use read_file on docs/guide.md and reply with the uppercase token only.",
        session_file=session_file,
    )
    session = load_session(session_file)
    reads = tool_results(session, "read_file")
    require(result.returncode == 0, "transcripted tool-use command failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require_llm_used(session, minimum=1, message="transcripted tool-use did not call the LLM", stdout=result.stdout, stderr=result.stderr)
    require(any(result.get("ok") for result in reads), "read_file did not succeed in transcript", stdout=result.stdout, stderr=result.stderr, session=session)
    require("TOKEN_42" in result.stdout, "final answer did not contain TOKEN_42", stdout=result.stdout, stderr=result.stderr, session=session)


def scenario_approval_accept(repo_root: Path, workspace: Path, model: str) -> None:
    session_file = workspace / "scratch" / "approve-yes.json"
    target = workspace / "scratch" / "approved.txt"
    result = run_cli(
        repo_root,
        workspace,
        model,
        "Create scratch/approved.txt with exactly the single line APPROVED followed by a newline. Then use read_file to confirm it and reply with APPROVED only.",
        approval="ask",
        session_file=session_file,
        stdin_text="y\ny\ny\ny\ny\n",
    )
    session = load_session(session_file)
    writes = tool_results(session, "write_file")
    require(result.returncode == 0, "approval accept command failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require_llm_used(session, minimum=1, message="approval accept did not call the LLM", stdout=result.stdout, stderr=result.stderr)
    require(target.exists() and target.read_text(encoding="utf-8") == "APPROVED\n", "approved file content mismatch", stdout=result.stdout, stderr=result.stderr, session=session)
    require(any(result.get("ok") for result in writes), "write_file was not approved successfully", stdout=result.stdout, stderr=result.stderr, session=session)


def scenario_approval_reject(repo_root: Path, workspace: Path, model: str) -> None:
    session_file = workspace / "scratch" / "approve-no.json"
    target = workspace / "scratch" / "rejected.txt"
    result = run_cli(
        repo_root,
        workspace,
        model,
        "Create scratch/rejected.txt with exactly the single line REJECTED followed by a newline. If it does not happen, tell me what happened.",
        approval="ask",
        session_file=session_file,
        stdin_text="n\nn\nn\nn\nn\n",
    )
    session = load_session(session_file)
    writes = tool_results(session, "write_file")
    require(result.returncode == 0, "approval reject command failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require_llm_used(session, minimum=1, message="approval reject did not call the LLM", stdout=result.stdout, stderr=result.stderr)
    require(not target.exists(), "rejected file should not exist", stdout=result.stdout, stderr=result.stderr, session=session)
    require(any(not result.get("ok") and "rejected" in str(result.get("summary", "")).lower() for result in writes), "write rejection was not recorded", stdout=result.stdout, stderr=result.stderr, session=session)


def scenario_path_escape(repo_root: Path, workspace: Path, model: str) -> None:
    session_file = workspace / "scratch" / "escape.json"
    result = run_cli(
        repo_root,
        workspace,
        model,
        "Use read_file on ../outside.txt and tell me the exact tool error.",
        session_file=session_file,
    )
    session = load_session(session_file)
    reads = tool_results(session, "read_file")
    require(result.returncode == 0, "path escape command failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require_llm_used(session, minimum=1, message="path escape scenario did not call the LLM", stdout=result.stdout, stderr=result.stderr)
    require(any("escapes the workspace" in str(item.get("summary", "")) for item in reads), "workspace escape was not blocked", stdout=result.stdout, stderr=result.stderr, session=session)


def scenario_shell_failure(repo_root: Path, workspace: Path, model: str) -> None:
    session_file = workspace / "scratch" / "shell-fail.json"
    result = run_cli(
        repo_root,
        workspace,
        model,
        "Use run_shell to execute exactly: python3 -c \"import sys; print('boom'); sys.exit(5)\". Then tell me the exit code and the printed word.",
        session_file=session_file,
    )
    session = load_session(session_file)
    shells = tool_results(session, "run_shell")
    require(result.returncode == 0, "shell failure command failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require_llm_used(session, minimum=1, message="shell failure scenario did not call the LLM", stdout=result.stdout, stderr=result.stderr)
    require(any(item.get("exit_code") == 5 and item.get("output") == "boom" for item in shells), "shell failure details were not recorded", stdout=result.stdout, stderr=result.stderr, session=session)


def scenario_run_test(repo_root: Path, workspace: Path, model: str) -> None:
    session_file = workspace / "scratch" / "run-test.json"
    result = run_cli(
        repo_root,
        workspace,
        model,
        "Use run_test and tell me whether tests passed and which test module ran.",
        session_file=session_file,
        extra_args=["--test-cmd", "python3 -m unittest discover -s tests -v"],
    )
    session = load_session(session_file)
    test_runs = tool_results(session, "run_test")
    require(result.returncode == 0, "run_test command failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require_llm_used(session, minimum=1, message="run_test scenario did not call the LLM", stdout=result.stdout, stderr=result.stderr)
    require(any(item.get("ok") and "test_sample" in str(item.get("output", "")) and "OK" in str(item.get("output", "")) for item in test_runs), "run_test result was not captured", stdout=result.stdout, stderr=result.stderr, session=session)


def scenario_issue_fix_hidden_tests(repo_root: Path, workspace: Path, model: str) -> None:
    _write(workspace / "src" / "calculator.py", "def add(left: int, right: int) -> int:\n    return left - right\n")
    _write(
        workspace / "tests" / "test_calculator.py",
        _standard_test_import("calculator")
        + "import unittest\n\n\nclass CalculatorTests(unittest.TestCase):\n"
        + "    def test_adds_positive_numbers(self) -> None:\n        self.assertEqual(add(2, 3), 5)\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )
    session_file = workspace / "scratch" / "issue-fix-hidden.json"
    result = run_cli(
        repo_root,
        workspace,
        model,
        "Issue: src/calculator.py add(left, right) returns the wrong value. Inspect source/tests, fix it, run tests, and summarize changed files.",
        session_file=session_file,
        extra_args=["--test-cmd", _python_test_cmd()],
        timeout=900,
    )
    session = load_session(session_file)
    hidden = "import sys; sys.path.insert(0, 'src'); from calculator import add; assert add(-2, 5) == 3; assert add(0, 0) == 0"
    source = (workspace / "src" / "calculator.py").read_text(encoding="utf-8")
    require(result.returncode == 0, "issue-fix-hidden-tests command failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require_llm_used(session, minimum=1, message="issue-fix-hidden-tests did not call the LLM", stdout=result.stdout, stderr=result.stderr)
    require("return left + right" in source, "calculator.py was not fixed", stdout=result.stdout, stderr=result.stderr, session=session)
    require(_hidden_python(workspace, hidden), "hidden calculator assertions failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require(_run_default_tests(workspace), "workspace tests do not pass after calculator fix", stdout=result.stdout, stderr=result.stderr, session=session)
    require(any(item.get("ok") for item in tool_results(session, "run_test")), "run_test did not succeed for calculator fix", stdout=result.stdout, stderr=result.stderr, session=session)


def scenario_multi_file_refactor(repo_root: Path, workspace: Path, model: str) -> None:
    _write(workspace / "src" / "pricing.py", "def total(prices: list[int]) -> int:\n    return sum(prices)\n")
    _write(
        workspace / "tests" / "test_pricing.py",
        _standard_test_import("pricing")
        + "import unittest\n\n\nclass PricingTests(unittest.TestCase):\n"
        + "    def test_cart_total(self) -> None:\n        self.assertEqual(cart_total([2, 3, 4]), 9)\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )
    _write(workspace / "docs" / "pricing.md", "Call `total(prices)` to compute a cart total.\n")
    session_file = workspace / "scratch" / "multi-file-refactor.json"
    result = run_cli(
        repo_root,
        workspace,
        model,
        "Refactor the pricing API from total(prices) to cart_total(prices). Update src/pricing.py, tests, and docs/pricing.md. Run tests.",
        session_file=session_file,
        extra_args=["--test-cmd", _python_test_cmd()],
        timeout=900,
    )
    session = load_session(session_file)
    source = (workspace / "src" / "pricing.py").read_text(encoding="utf-8")
    docs = (workspace / "docs" / "pricing.md").read_text(encoding="utf-8")
    hidden = "import sys; sys.path.insert(0, 'src'); from pricing import cart_total; assert cart_total([1, 2, 7]) == 10"
    require(result.returncode == 0, "multi-file refactor command failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require_llm_used(session, minimum=1, message="multi-file refactor did not call the LLM", stdout=result.stdout, stderr=result.stderr)
    require("def cart_total" in source and "def total" not in source, "pricing.py was not refactored", stdout=result.stdout, stderr=result.stderr, session=session)
    require("cart_total" in docs and "`total(prices)`" not in docs, "pricing docs were not updated", stdout=result.stdout, stderr=result.stderr, session=session)
    require(_hidden_python(workspace, hidden), "hidden pricing assertions failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require(_run_default_tests(workspace), "workspace tests do not pass after pricing refactor", stdout=result.stdout, stderr=result.stderr, session=session)
    require(any(item.get("ok") for item in tool_results(session, "run_test")), "run_test did not succeed for pricing refactor", stdout=result.stdout, stderr=result.stderr, session=session)


def scenario_large_repo_symbol_nav(repo_root: Path, workspace: Path, model: str) -> None:
    for index in range(30):
        _write(workspace / "src" / f"distractor_{index}.py", "\n\n".join(f"def helper_{index}_{n}():\n    return {n}" for n in range(20)) + "\n")
    before = "\n\n".join(f"def pre_{index}():\n    return {index}" for index in range(160))
    after = "\n\n".join(f"def post_{index}():\n    return {index}" for index in range(160))
    target = "def calculate_discount(cart):\n    marker = 'BENCH_SYMBOL_TOKEN_817'\n    return marker\n"
    _write(workspace / "src" / "large_pricing.py", f"{before}\n\n{target}\n\n{after}\n")
    session_file = workspace / "scratch" / "large-symbol-nav.json"
    result = run_cli(
        repo_root,
        workspace,
        model,
        "Use search_symbols to find calculate_discount in src/large_pricing.py. Then use read_symbol on the exact match. Do not use read_file. Reply with the uppercase BENCH_SYMBOL token from that symbol only.",
        session_file=session_file,
        timeout=900,
    )
    session = load_session(session_file)
    reads = tool_results(session, "read_symbol")
    calls = tool_calls(session)
    require(result.returncode == 0, "large symbol navigation command failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require_llm_used(session, minimum=1, message="large symbol navigation did not call the LLM", stdout=result.stdout, stderr=result.stderr)
    require("search_symbols" in calls and "read_symbol" in calls, "symbol-navigation tools were not used", stdout=result.stdout, stderr=result.stderr, session=session)
    require("read_file" not in calls, "large symbol navigation used forbidden read_file", stdout=result.stdout, stderr=result.stderr, session=session)
    require(any(item.get("ok") and "BENCH_SYMBOL_TOKEN_817" in str(item.get("output", "")) for item in reads), "read_symbol did not capture the target token", stdout=result.stdout, stderr=result.stderr, session=session)
    require("BENCH_SYMBOL_TOKEN_817" in result.stdout, "final answer did not contain the symbol token", stdout=result.stdout, stderr=result.stderr, session=session)


def scenario_nested_package_import_fix(repo_root: Path, workspace: Path, model: str) -> None:
    _write(workspace / "src" / "pkg" / "__init__.py", "")
    _write(workspace / "src" / "pkg" / "core.py", "from helpers import label\n\ndef wrapped() -> str:\n    return label('ok')\n")
    _write(workspace / "src" / "pkg" / "helpers.py", "def label(value: str) -> str:\n    return f'[{value}]'\n")
    _write(
        workspace / "tests" / "test_pkg.py",
        "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
        + "import unittest\nfrom pkg.core import wrapped\n\n\nclass PackageTests(unittest.TestCase):\n"
        + "    def test_wrapped(self) -> None:\n        self.assertEqual(wrapped(), '[ok]')\n\n\nif __name__ == '__main__':\n        unittest.main()\n",
    )
    session_file = workspace / "scratch" / "nested-package-import-fix.json"
    result = run_cli(
        repo_root,
        workspace,
        model,
        "Run tests, fix the package import bug in src/pkg/core.py, rerun tests, and summarize.",
        session_file=session_file,
        extra_args=["--test-cmd", _python_test_cmd()],
        timeout=900,
    )
    session = load_session(session_file)
    source = (workspace / "src" / "pkg" / "core.py").read_text(encoding="utf-8")
    require(result.returncode == 0, "nested package import fix command failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require_llm_used(session, minimum=1, message="nested package import fix did not call the LLM", stdout=result.stdout, stderr=result.stderr)
    require("from .helpers import label" in source, "pkg/core.py import was not fixed", stdout=result.stdout, stderr=result.stderr, session=session)
    require(_run_default_tests(workspace), "workspace tests do not pass after package import fix", stdout=result.stdout, stderr=result.stderr, session=session)
    require(any(item.get("ok") for item in tool_results(session, "run_test")), "run_test did not succeed for package import fix", stdout=result.stdout, stderr=result.stderr, session=session)


def scenario_bad_test_command_recovery(repo_root: Path, workspace: Path, model: str) -> None:
    _write(workspace / "src" / "inventory.py", "def total_units(counts: list[int]) -> int:\n    return sum(counts) - 1\n")
    _write(
        workspace / "tests" / "test_inventory.py",
        _standard_test_import("inventory")
        + "import unittest\n\n\nclass InventoryTests(unittest.TestCase):\n"
        + "    def test_sums_units(self) -> None:\n        self.assertEqual(total_units([2, 3, 4]), 9)\n"
        + "    def test_empty(self) -> None:\n        self.assertEqual(total_units([]), 0)\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )
    session_file = workspace / "scratch" / "bad-test-command-recovery.json"
    result = run_cli(
        repo_root,
        workspace,
        model,
        "Run tests, fix src/inventory.py so total_units is correct, rerun tests, and summarize briefly.",
        session_file=session_file,
        extra_args=["--test-cmd", "pytesst -q"],
        timeout=900,
    )
    session = load_session(session_file)
    source = (workspace / "src" / "inventory.py").read_text(encoding="utf-8")
    hidden = "import sys; sys.path.insert(0, 'src'); from inventory import total_units; assert total_units([5, 0, 2]) == 7; assert total_units([1]) == 1"
    test_runs = tool_results(session, "run_test")
    recovered = any(
        item.get("recovered") is True
        and "pytesst -q" in str(item.get("original_command", ""))
        and "unittest discover" in str(item.get("command", ""))
        for item in test_runs
    )
    require(result.returncode == 0, "bad test command recovery command failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require_llm_used(session, minimum=1, message="bad test command recovery did not call the LLM", stdout=result.stdout, stderr=result.stderr)
    require("return sum(counts)" in source, "inventory.py was not fixed", stdout=result.stdout, stderr=result.stderr, session=session)
    require(recovered, "run_test did not recover from the broken configured command", stdout=result.stdout, stderr=result.stderr, session=session)
    require(_hidden_python(workspace, hidden), "hidden inventory assertions failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require(_run_default_tests(workspace), "workspace tests do not pass after inventory fix", stdout=result.stdout, stderr=result.stderr, session=session)


def scenario_api_docs_callsite_sync(repo_root: Path, workspace: Path, model: str) -> None:
    fetch_user = _nonce("fetch_user")
    build_summary = _nonce("build_summary")
    api_module = "api"
    dashboard_module = "dashboard"
    docs_name = "api.md"
    _write(
        workspace / "src" / f"{api_module}.py",
        f"def {fetch_user}(user_id: str, include_orders: bool = False) -> dict[str, object]:\n    return {{'id': user_id}}\n",
    )
    _write(
        workspace / "src" / f"{dashboard_module}.py",
        f"from {api_module} import {fetch_user}\n\n\ndef {build_summary}(user_id: str):\n    return {fetch_user}(user_id)\n",
    )
    _write(workspace / "docs" / docs_name, f"`{fetch_user}(user_id, include_orders=False)` returns a user dict.\n")
    _write(
        workspace / "tests" / "test_api_dashboard.py",
        _standard_test_import(api_module)
        + f"from {dashboard_module} import {build_summary}\n"
        + "import unittest\n\n\nclass ApiDashboardTests(unittest.TestCase):\n"
        + f"    def test_fetch_user_default(self) -> None:\n        self.assertEqual({fetch_user}('7'), {{'id': '7'}})\n"
        + f"    def test_build_summary_includes_orders(self) -> None:\n        self.assertEqual({build_summary}('7'), {{'id': '7', 'orders': []}})\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )
    session_file = workspace / "scratch" / "api-docs-callsite-sync.json"
    result = run_cli(
        repo_root,
        workspace,
        model,
        (
            f"Update {fetch_user} in src/{api_module}.py so include_orders=True returns a user dict with an empty orders list. "
            f"Update src/{dashboard_module}.py so {build_summary} passes include_orders=True. "
            f"Update tests and docs/{docs_name}, run tests, and summarize briefly."
        ),
        session_file=session_file,
        extra_args=["--test-cmd", _python_test_cmd()],
        timeout=1200,
    )
    session = load_session(session_file)
    api_source = (workspace / "src" / f"{api_module}.py").read_text(encoding="utf-8")
    dashboard_source = (workspace / "src" / f"{dashboard_module}.py").read_text(encoding="utf-8")
    docs = (workspace / "docs" / docs_name).read_text(encoding="utf-8")
    hidden = (
        f"import sys; sys.path.insert(0, 'src'); from {api_module} import {fetch_user}; from {dashboard_module} import {build_summary}; "
        f"assert {fetch_user}('2') == {{'id': '2'}}; "
        f"assert {fetch_user}('2', include_orders=True) == {{'id': '2', 'orders': []}}; "
        f"assert {build_summary}('3') == {{'id': '3', 'orders': []}}"
    )
    require(result.returncode == 0, "api docs callsite sync command failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require_llm_used(session, minimum=1, message="api docs callsite sync did not call the LLM", stdout=result.stdout, stderr=result.stderr)
    require("include_orders" in api_source and "'orders': []" in api_source, "api.py was not updated for include_orders", stdout=result.stdout, stderr=result.stderr, session=session)
    require("include_orders=True" in dashboard_source, "dashboard.py did not update the fetch_user call site", stdout=result.stdout, stderr=result.stderr, session=session)
    require(fetch_user in docs and "include_orders" in docs, "api docs were not updated", stdout=result.stdout, stderr=result.stderr, session=session)
    require(_hidden_python(workspace, hidden), "hidden api/dashboard assertions failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require(_run_default_tests(workspace), "workspace tests do not pass after api/docs sync", stdout=result.stdout, stderr=result.stderr, session=session)
    require(any(item.get("ok") for item in tool_results(session, "run_test")), "run_test did not succeed for api/docs sync", stdout=result.stdout, stderr=result.stderr, session=session)


def scenario_scoreboard_hidden(repo_root: Path, workspace: Path, model: str) -> None:
    module = "scoreboard"
    function_name = "score_delta"
    _write(workspace / "src" / f"{module}.py", f"def {function_name}(base: int, bonus: int) -> int:\n    pass\n")
    _write(
        workspace / "tests" / "test_scoreboard.py",
        _standard_test_import(module)
        + "import unittest\n\n\nclass ScoreboardTests(unittest.TestCase):\n"
        + f"    def test_positive_values(self) -> None:\n        self.assertEqual({function_name}(2, 3), 5)\n"
        + f"    def test_mixed_values(self) -> None:\n        self.assertEqual({function_name}(-1, 4), 3)\n"
        + f"    def test_zero_values(self) -> None:\n        self.assertEqual({function_name}(0, 0), 0)\n"
        + f"    def test_negative_values(self) -> None:\n        self.assertEqual({function_name}(-5, -2), -7)\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )
    session_file = workspace / "scratch" / "scoreboard-hidden.json"
    result = run_cli(
        repo_root,
        workspace,
        model,
        f"Implement {function_name} in src/{module}.py from the tests. Read source and tests, replace stubs with complete code, run tests, and summarize briefly.",
        session_file=session_file,
        extra_args=["--test-cmd", _python_test_cmd()],
        timeout=1200,
    )
    session = load_session(session_file)
    source = (workspace / "src" / f"{module}.py").read_text(encoding="utf-8")
    hidden = (
        f"import sys; sys.path.insert(0, 'src'); from {module} import {function_name}; "
        f"assert {function_name}(9, -4) == 5; assert {function_name}(-8, 3) == -5"
    )
    require(result.returncode == 0, "scoreboard hidden command failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require_llm_used(session, minimum=1, message="scoreboard hidden did not call the LLM", stdout=result.stdout, stderr=result.stderr)
    require("pass" not in source, "scoreboard module still contains a stub", stdout=result.stdout, stderr=result.stderr, session=session)
    require(_hidden_python(workspace, hidden), "hidden scoreboard assertions failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require(_run_default_tests(workspace), "workspace tests do not pass after scoreboard implementation", stdout=result.stdout, stderr=result.stderr, session=session)
    require(any(item.get("ok") for item in tool_results(session, "run_test")), "run_test did not succeed for scoreboard", stdout=result.stdout, stderr=result.stderr, session=session)


def scenario_checkout_callsite_refactor(repo_root: Path, workspace: Path, model: str) -> None:
    old_name = "total"
    new_name = "cart_total"
    caller_name = "checkout_total"
    _write(workspace / "src" / "pricing.py", f"def {old_name}(prices: list[int]) -> int:\n    return sum(prices)\n")
    _write(
        workspace / "src" / "checkout.py",
        f"from pricing import {old_name}\n\n\ndef {caller_name}(prices: list[int]) -> int:\n    return {old_name}(prices)\n",
    )
    _write(
        workspace / "tests" / "test_checkout_refactor.py",
        _standard_test_import("pricing")
        + f"from checkout import {caller_name}\n"
        + "import unittest\n\n\nclass CheckoutRefactorTests(unittest.TestCase):\n"
        + f"    def test_public_pricing_api(self) -> None:\n        self.assertEqual({new_name}([2, 3, 4]), 9)\n"
        + f"    def test_checkout_callsite(self) -> None:\n        self.assertEqual({caller_name}([1, 5]), 6)\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )
    _write(workspace / "docs" / "pricing.md", f"Call `{old_name}(prices)` to compute a cart total.\n")
    session_file = workspace / "scratch" / "checkout-callsite-refactor.json"
    result = run_cli(
        repo_root,
        workspace,
        model,
        (
            f"Refactor the pricing API from {old_name}(prices) to {new_name}(prices). Update src/pricing.py, "
            f"src/checkout.py, tests, and docs/pricing.md. Run tests and summarize briefly."
        ),
        session_file=session_file,
        extra_args=["--test-cmd", _python_test_cmd()],
        timeout=1200,
    )
    session = load_session(session_file)
    pricing_source = (workspace / "src" / "pricing.py").read_text(encoding="utf-8")
    checkout_source = (workspace / "src" / "checkout.py").read_text(encoding="utf-8")
    docs = (workspace / "docs" / "pricing.md").read_text(encoding="utf-8")
    hidden = (
        f"import sys; sys.path.insert(0, 'src'); from pricing import {new_name}; from checkout import {caller_name}; "
        f"assert {new_name}([9, 1]) == 10; assert {caller_name}([4, 6]) == 10"
    )
    require(result.returncode == 0, "checkout callsite refactor command failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require_llm_used(session, minimum=1, message="checkout callsite refactor did not call the LLM", stdout=result.stdout, stderr=result.stderr)
    require(f"def {new_name}" in pricing_source and f"def {old_name}(" not in pricing_source, "pricing.py was not refactored", stdout=result.stdout, stderr=result.stderr, session=session)
    require(f"from pricing import {new_name}" in checkout_source and f"return {new_name}(prices)" in checkout_source and f"from pricing import {old_name}" not in checkout_source and f"return {old_name}(prices)" not in checkout_source, "checkout.py did not update the pricing call site", stdout=result.stdout, stderr=result.stderr, session=session)
    require(new_name in docs and f"`{old_name}(prices)`" not in docs, "pricing docs were not updated", stdout=result.stdout, stderr=result.stderr, session=session)
    require(_hidden_python(workspace, hidden), "hidden checkout refactor assertions failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require(_run_default_tests(workspace), "workspace tests do not pass after checkout refactor", stdout=result.stdout, stderr=result.stderr, session=session)
    require(any(item.get("ok") for item in tool_results(session, "run_test")), "run_test did not succeed for checkout refactor", stdout=result.stdout, stderr=result.stderr, session=session)


def scenario_retry_policy_hidden(repo_root: Path, workspace: Path, model: str) -> None:
    should_retry = "should_retry"
    next_action = "next_action"
    _write(
        workspace / "src" / "retry.py",
        f"def {should_retry}(status_code: int, attempt: int) -> bool:\n    return status_code >= 500 and attempt > 2\n",
    )
    _write(
        workspace / "src" / "client.py",
        f"from retry import {should_retry}\n\n\ndef {next_action}(status_code: int, attempt: int) -> str:\n    return 'retry' if {should_retry}(status_code, attempt) else 'stop'\n",
    )
    _write(
        workspace / "tests" / "test_retry_policy.py",
        _standard_test_import("retry")
        + f"from client import {next_action}\n"
        + "import unittest\n\n\nclass RetryPolicyTests(unittest.TestCase):\n"
        + f"    def test_retries_initial_attempts(self) -> None:\n        self.assertTrue({should_retry}(503, 0))\n"
        + f"    def test_does_not_retry_non_server_error(self) -> None:\n        self.assertFalse({should_retry}(404, 0))\n"
        + f"    def test_retries_upper_server_range(self) -> None:\n        self.assertTrue({should_retry}(599, 2))\n"
        + f"    def test_does_not_retry_client_error_boundary(self) -> None:\n        self.assertFalse({should_retry}(499, 0))\n"
        + f"    def test_client_retry_action(self) -> None:\n        self.assertEqual({next_action}(503, 1), 'retry')\n"
        + f"    def test_client_stop_after_budget(self) -> None:\n        self.assertEqual({next_action}(503, 3), 'stop')\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )
    session_file = workspace / "scratch" / "retry-policy-hidden.json"
    result = run_cli(
        repo_root,
        workspace,
        model,
        (
            f"The retry policy in src/retry.py is wrong. Attempts 0, 1, and 2 should retry for HTTP status codes 500 through 599; "
            f"attempt 3 and above should stop, and 499 or 404 should not retry. Read source and tests, fix {should_retry}, run tests, and summarize briefly."
        ),
        session_file=session_file,
        extra_args=["--test-cmd", _python_test_cmd()],
        timeout=1200,
    )
    session = load_session(session_file)
    hidden = (
        f"import sys; sys.path.insert(0, 'src'); from retry import {should_retry}; from client import {next_action}; "
        f"assert {should_retry}(502, 2) is True; assert {should_retry}(502, 3) is False; assert {next_action}(500, 2) == 'retry'"
    )
    require(result.returncode == 0, "retry policy hidden command failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require_llm_used(session, minimum=1, message="retry policy hidden did not call the LLM", stdout=result.stdout, stderr=result.stderr)
    require(_hidden_python(workspace, hidden), "hidden retry policy assertions failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require(_run_default_tests(workspace), "workspace tests do not pass after retry policy fix", stdout=result.stdout, stderr=result.stderr, session=session)
    require(any(item.get("ok") for item in tool_results(session, "run_test")), "run_test did not succeed for retry policy", stdout=result.stdout, stderr=result.stderr, session=session)


def scenario_balance_credit_hidden(repo_root: Path, workspace: Path, model: str) -> None:
    _write(workspace / "src" / "balance.py", "def apply_credit(amount: int, credit: int) -> int:\n    return amount - credit\n")
    _write(
        workspace / "src" / "report.py",
        "from balance import apply_credit\n\n\ndef closing_balance(amount: int, credit: int) -> str:\n    return f'closing={apply_credit(amount, credit)}'\n",
    )
    _write(
        workspace / "tests" / "test_balance_credit.py",
        _standard_test_import("balance")
        + "from report import closing_balance\n"
        + "import unittest\n\n\nclass BalanceCreditTests(unittest.TestCase):\n"
        + "    def test_apply_credit(self) -> None:\n        self.assertEqual(apply_credit(10, 3), 13)\n"
        + "    def test_report_uses_credit(self) -> None:\n        self.assertEqual(closing_balance(10, 3), 'closing=13')\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )
    session_file = workspace / "scratch" / "balance-credit-hidden.json"
    result = run_cli(
        repo_root,
        workspace,
        model,
        "Bug in src/balance.py: apply_credit(amount, credit) returns the wrong value. Read source and tests, fix it, run tests, and summarize briefly.",
        session_file=session_file,
        extra_args=["--test-cmd", _python_test_cmd()],
        timeout=1200,
    )
    session = load_session(session_file)
    source = (workspace / "src" / "balance.py").read_text(encoding="utf-8")
    hidden = (
        "import sys; sys.path.insert(0, 'src'); from balance import apply_credit; from report import closing_balance; "
        "assert apply_credit(0, 0) == 0; assert apply_credit(-2, 5) == 3; assert closing_balance(7, 1) == 'closing=8'"
    )
    require(result.returncode == 0, "balance credit hidden command failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require_llm_used(session, minimum=1, message="balance credit hidden did not call the LLM", stdout=result.stdout, stderr=result.stderr)
    require("return amount + credit" in source, "balance.py was not fixed", stdout=result.stdout, stderr=result.stderr, session=session)
    require(_hidden_python(workspace, hidden), "hidden balance credit assertions failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require(_run_default_tests(workspace), "workspace tests do not pass after balance credit fix", stdout=result.stdout, stderr=result.stderr, session=session)
    require(any(item.get("ok") for item in tool_results(session, "run_test")), "run_test did not succeed for balance credit", stdout=result.stdout, stderr=result.stderr, session=session)


def scenario_slug_repair_hidden(repo_root: Path, workspace: Path, model: str) -> None:
    module = "slug"
    function_name = _nonce("slugify")
    _write(workspace / "src" / f"{module}.py", f"def {function_name}(value: str) -> str:\n    pass\n")
    _write(
        workspace / "tests" / "test_slug.py",
        _standard_test_import(module)
        + "import unittest\n\n\nclass SlugTests(unittest.TestCase):\n"
        + f"    def test_slugifies_spaces(self) -> None:\n        self.assertEqual({function_name}('Hello Local Model'), 'hello-local-model')\n"
        + f"    def test_trims_edges(self) -> None:\n        self.assertEqual({function_name}('  Mixed Case  '), 'mixed-case')\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )
    session_file = workspace / "scratch" / "slug-repair-hidden.json"
    result = run_cli(
        repo_root,
        workspace,
        model,
        (
            f"Implement {function_name} in src/{module}.py from the tests. Read source and tests, replace the stub with "
            f"complete code so whitespace collapses to single hyphens and output is lowercase, run tests, and summarize briefly."
        ),
        session_file=session_file,
        extra_args=["--test-cmd", _python_test_cmd()],
        timeout=1200,
    )
    session = load_session(session_file)
    source = (workspace / "src" / f"{module}.py").read_text(encoding="utf-8")
    hidden = (
        f"import sys; sys.path.insert(0, 'src'); from {module} import {function_name}; "
        f"assert {function_name}('  Many   Spaces ') == 'many-spaces'; "
        f"assert {function_name}('Tabs\\tand\\nlines') == 'tabs-and-lines'"
    )
    require(result.returncode == 0, "slug repair hidden command failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require_llm_used(session, minimum=1, message="slug repair hidden did not call the LLM", stdout=result.stdout, stderr=result.stderr)
    require("lower" in source and "-" in source, "slug repair module was not updated meaningfully", stdout=result.stdout, stderr=result.stderr, session=session)
    require(_hidden_python(workspace, hidden), "hidden slug repair assertions failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require(_run_default_tests(workspace), "workspace tests do not pass after slug repair", stdout=result.stdout, stderr=result.stderr, session=session)
    require(any(item.get("ok") for item in tool_results(session, "run_test")), "run_test did not succeed for slug repair", stdout=result.stdout, stderr=result.stderr, session=session)


def scenario_flag_constant_sync(repo_root: Path, workspace: Path, model: str) -> None:
    module = "flags"
    consumer_module = "consumer"
    constant_name = _nonce("old_flag").upper()
    function_name = _nonce("flag_name")
    consumer_name = _nonce("describe_flag")
    docs_name = "flags.md"
    _write(
        workspace / "src" / f"{module}.py",
        f"{constant_name} = 'old'\n\n\ndef {function_name}() -> str:\n    return {constant_name}\n",
    )
    _write(
        workspace / "src" / f"{consumer_module}.py",
        f"from {module} import {function_name}\n\n\ndef {consumer_name}() -> str:\n    return f'flag={{{function_name}()}}'\n",
    )
    _write(
        workspace / "tests" / "test_flags_consumer.py",
        _standard_test_import(module)
        + f"from {consumer_module} import {consumer_name}\n"
        + "import unittest\n\n\nclass FlagSyncTests(unittest.TestCase):\n"
        + f"    def test_public_function(self) -> None:\n        self.assertEqual({function_name}(), 'new')\n"
        + f"    def test_consumer_output(self) -> None:\n        self.assertEqual({consumer_name}(), 'flag=new')\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    )
    _write(workspace / "docs" / docs_name, f"`{function_name}()` returns the current flag value from `{constant_name}`. Right now `{constant_name}` is `'old'`.\n")
    session_file = workspace / "scratch" / "flag-constant-sync.json"
    result = run_cli(
        repo_root,
        workspace,
        model,
        (
            f"Change {constant_name} in src/{module}.py from 'old' to 'new' while keeping {function_name}() as the public function. "
            f"Update src/{consumer_module}.py, tests, and docs/{docs_name} if needed. Run tests and summarize briefly."
        ),
        session_file=session_file,
        extra_args=["--test-cmd", _python_test_cmd()],
        timeout=1200,
    )
    session = load_session(session_file)
    source = (workspace / "src" / f"{module}.py").read_text(encoding="utf-8")
    consumer_source = (workspace / "src" / f"{consumer_module}.py").read_text(encoding="utf-8")
    docs = (workspace / "docs" / docs_name).read_text(encoding="utf-8")
    hidden = (
        f"import sys; sys.path.insert(0, 'src'); from {module} import {function_name}; from {consumer_module} import {consumer_name}; "
        f"assert {function_name}() == 'new'; assert {consumer_name}() == 'flag=new'"
    )
    require(result.returncode == 0, "flag constant sync command failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require_llm_used(session, minimum=1, message="flag constant sync did not call the LLM", stdout=result.stdout, stderr=result.stderr)
    require(f"{constant_name} = 'new'" in source, "constant was not updated", stdout=result.stdout, stderr=result.stderr, session=session)
    require(function_name in consumer_source and "flag=" in consumer_source, "consumer module was damaged", stdout=result.stdout, stderr=result.stderr, session=session)
    require(constant_name in docs and "'new'" in docs, "flag docs were not updated", stdout=result.stdout, stderr=result.stderr, session=session)
    require(_hidden_python(workspace, hidden), "hidden flag sync assertions failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require(_run_default_tests(workspace), "workspace tests do not pass after flag sync", stdout=result.stdout, stderr=result.stderr, session=session)
    require(any(item.get("ok") for item in tool_results(session, "run_test")), "run_test did not succeed for flag sync", stdout=result.stdout, stderr=result.stderr, session=session)


def scenario_subagent_transcript(repo_root: Path, workspace: Path, model: str) -> None:
    session_file = workspace / "scratch" / "subagent.json"
    result = run_cli(
        repo_root,
        workspace,
        model,
        f"Use run_agent with model {model}. Tell the helper to read docs/guide.md in read-only mode and return the exact uppercase token. Then answer with that token only.",
        session_file=session_file,
    )
    session = load_session(session_file)
    subagents = tool_results(session, "run_agent")
    require(result.returncode == 0, "subagent command failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require_llm_used(session, minimum=1, message="subagent scenario did not call the parent LLM", stdout=result.stdout, stderr=result.stderr)
    require(any(item.get("ok") and item.get("event_count", 0) >= 3 and "TOKEN_42" in str(item.get("output", "")) for item in subagents), "subagent result was not captured correctly", stdout=result.stdout, stderr=result.stderr, session=session)


def scenario_git_tools(repo_root: Path, workspace: Path, model: str) -> None:
    session_file = workspace / "scratch" / "git-tools.json"
    commit_all(workspace, "checkpoint before git scenario")
    target = workspace / "src" / "app.py"
    target.write_text("def meaning() -> int:\n    return 99\n", encoding="utf-8")
    result = run_cli(
        repo_root,
        workspace,
        model,
        "Use git_status on src/app.py. Then use git_diff on src/app.py for the working tree only; omit cached or set cached to false. Do not use read_file. Reply with src/app.py and whether the diff adds return 99.",
        session_file=session_file,
    )
    session = load_session(session_file)
    statuses = tool_results(session, "git_status")
    diffs = tool_results(session, "git_diff")
    calls = tool_calls(session)
    require(result.returncode == 0, "git tool command failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require_llm_used(session, minimum=1, message="git tool scenario did not call the LLM", stdout=result.stdout, stderr=result.stderr)
    require(any(item.get("ok") and "src/app.py" in str(item.get("output", "")) for item in statuses), "git_status result was not captured", stdout=result.stdout, stderr=result.stderr, session=session)
    require(any(item.get("ok") and "return 99" in str(item.get("output", "")) for item in diffs), "git_diff result was not captured", stdout=result.stdout, stderr=result.stderr, session=session)
    require("read_file" not in calls, "git tool command used forbidden read_file", stdout=result.stdout, stderr=result.stderr, session=session)


def scenario_continue_session(repo_root: Path, workspace: Path, model: str) -> None:
    session_dir = workspace / ".ollama-code" / "sessions"
    first = run_cli(
        repo_root,
        workspace,
        model,
        "Remember the exact token CONTINUE_TOKEN_99 for this session and reply with remembered.",
    )
    require(first.returncode == 0, "initial continue-session command failed", stdout=first.stdout, stderr=first.stderr)
    require(session_dir.exists(), "continue-session did not create a session directory", stdout=first.stdout, stderr=first.stderr)
    second = run_cli(
        repo_root,
        workspace,
        model,
        "What token did I ask you to remember earlier in this session? Reply with the token only.",
        extra_args=["--continue"],
    )
    require(second.returncode == 0, "continue-session follow-up failed", stdout=second.stdout, stderr=second.stderr)
    session_files = sorted(session_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
    require(bool(session_files), "continue-session did not save any transcript", stdout=second.stdout, stderr=second.stderr)
    session = load_session(session_files[-1])
    messages = session.get("messages", [])
    require_llm_used(session, minimum=2, message="continue-session prompts did not each call the LLM", stdout=second.stdout, stderr=second.stderr)
    require(
        any(isinstance(message, dict) and "CONTINUE_TOKEN_99" in str(message.get("content", "")) for message in messages),
        "continued session transcript did not preserve the prior token",
        stdout=second.stdout,
        stderr=second.stderr,
        session=session,
    )
    require(
        any(
            isinstance(message, dict)
            and message.get("role") == "user"
            and "What token did I ask you to remember earlier in this session?" in str(message.get("content", ""))
            for message in messages
        ),
        "continued session transcript did not append the follow-up prompt",
        stdout=second.stdout,
        stderr=second.stderr,
        session=session,
    )


def scenario_multiturn_repl(repo_root: Path, workspace: Path, model: str) -> None:
    session_file = workspace / "scratch" / "repl.json"
    repl_input = (
        "Create notes/repl.txt with exactly the text repl ok followed by a newline.\n"
        "Use read_file on notes/repl.txt and reply with the full line.\n"
        "Use search to find repl ok and tell me which file contains it.\n"
        "/save scratch/repl-saved.json\n"
        "/quit\n"
    )
    result = run_repl(
        repo_root,
        workspace,
        model,
        repl_input,
        approval="auto",
        session_file=session_file,
        timeout=1200,
    )
    session = load_session(session_file)
    saved = workspace / "scratch" / "repl-saved.json"
    require(result.returncode == 0, "multiturn repl failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require_llm_used(session, minimum=3, message="multiturn repl prompts did not each call the LLM", stdout=result.stdout, stderr=result.stderr)
    require((workspace / "notes" / "repl.txt").read_text(encoding="utf-8") == "repl ok\n", "repl file content mismatch", stdout=result.stdout, stderr=result.stderr, session=session)
    require(saved.exists(), "manual /save did not create transcript", stdout=result.stdout, stderr=result.stderr, session=session)
    require("tool_call" in event_names(session) and "assistant" in event_names(session), "repl transcript missing expected events", stdout=result.stdout, stderr=result.stderr, session=session)


def scenario_clarifying_question_eba(repo_root: Path, workspace: Path, model: str) -> None:
    session_file = workspace / "scratch" / "clarify-eba.json"
    result = run_cli(
        repo_root,
        workspace,
        model,
        (
            "Before you edit anything, ask me one clarification question first and do not assume. "
            "Refactor the architecture heavily, but keep the important external surfaces stable."
        ),
        session_file=session_file,
    )
    session = load_session(session_file)
    clarification_events = [
        event for event in session.get("events", []) if isinstance(event, dict) and event.get("type") == "clarification_plan"
    ]
    require(result.returncode == 0, "clarifying-question scenario failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require_llm_used(session, minimum=1, message="clarifying-question scenario did not call the LLM", stdout=result.stdout, stderr=result.stderr)
    require(any(event.get("verdict") == "ask" for event in clarification_events), "clarification plan did not ask", stdout=result.stdout, stderr=result.stderr, session=session)
    require("Need one clarification before continuing:" in result.stdout, "final answer did not ask for clarification", stdout=result.stdout, stderr=result.stderr, session=session)
    require("Choices (pick one):" in result.stdout, "clarification answer omitted explicit choices", stdout=result.stdout, stderr=result.stderr, session=session)
    require("Recommended default:" in result.stdout, "clarification answer omitted recommended default", stdout=result.stdout, stderr=result.stderr, session=session)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run stricter end-to-end checks against a real Ollama Code model.")
    parser.add_argument("--model", default="gemma4:e4b", help="Model to test.")
    parser.add_argument("--scenarios", nargs="+", help="Optional scenario names to run.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    model = resolve_model(args.model, set(installed_models()))

    scenarios: list[tuple[str, Callable[[Path, Path, str], None]]] = [
        ("scenario_transcripted_tool_use", scenario_transcripted_tool_use),
        ("scenario_approval_accept", scenario_approval_accept),
        ("scenario_approval_reject", scenario_approval_reject),
        ("scenario_path_escape", scenario_path_escape),
        ("scenario_shell_failure", scenario_shell_failure),
        ("scenario_run_test", scenario_run_test),
        ("scenario_issue_fix_hidden_tests", scenario_issue_fix_hidden_tests),
        ("scenario_multi_file_refactor", scenario_multi_file_refactor),
        ("scenario_large_repo_symbol_nav", scenario_large_repo_symbol_nav),
        ("scenario_nested_package_import_fix", scenario_nested_package_import_fix),
        ("scenario_bad_test_command_recovery", scenario_bad_test_command_recovery),
        ("scenario_scoreboard_hidden", scenario_scoreboard_hidden),
        ("scenario_checkout_callsite_refactor", scenario_checkout_callsite_refactor),
        ("scenario_retry_policy_hidden", scenario_retry_policy_hidden),
        ("scenario_balance_credit_hidden", scenario_balance_credit_hidden),
        ("scenario_slug_repair_hidden", scenario_slug_repair_hidden),
        ("scenario_subagent_transcript", scenario_subagent_transcript),
        ("scenario_git_tools", scenario_git_tools),
        ("scenario_continue_session", scenario_continue_session),
        ("scenario_multiturn_repl", scenario_multiturn_repl),
        ("scenario_clarifying_question_eba", scenario_clarifying_question_eba),
    ]
    if args.scenarios:
        requested = set(args.scenarios)
        selected = [item for item in scenarios if item[0] in requested]
        missing = sorted(requested - {name for name, _ in selected})
        if missing:
            raise SystemExit(f"Unknown scenario(s): {', '.join(missing)}")
        scenarios = selected

    with workspace_temp_dir("ollama-code-e2e-", repo_root) as tmp:
        run_root = Path(tmp)
        print(f"[e2e] model={model}")
        _LOADED_MODELS.add(model)
        for name, scenario in scenarios:
            workspace = run_root / name
            workspace.mkdir(parents=True, exist_ok=True)
            git_available = build_workspace(workspace)
            if name == "scenario_git_tools" and not git_available:
                print(f"[e2e]   {name} (skipped: git workspace unavailable)")
                continue
            print(f"[e2e]   {name}")
            scenario(repo_root, workspace, model)
    unload_model(model)
    _LOADED_MODELS.discard(model)
    print("[e2e] all scenarios passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
