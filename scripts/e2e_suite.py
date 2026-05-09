from __future__ import annotations

import argparse
import atexit
import json
import os
import shlex
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
        "--require-llm-for-turn",
        "--max-tool-rounds",
        "12",
        "--max-agent-depth",
        "2",
    ]
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
        "--require-llm-for-turn",
        "--max-tool-rounds",
        "12",
        "--max-agent-depth",
        "2",
    ]
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run stricter end-to-end checks against a real Ollama Code model.")
    parser.add_argument("--model", default="gemma4:e4b", help="Model to test.")
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
        ("scenario_subagent_transcript", scenario_subagent_transcript),
        ("scenario_git_tools", scenario_git_tools),
        ("scenario_continue_session", scenario_continue_session),
        ("scenario_multiturn_repl", scenario_multiturn_repl),
    ]

    with workspace_temp_dir("ollama-code-e2e-", repo_root) as tmp:
        workspace = Path(tmp)
        git_available = build_workspace(workspace)
        print(f"[e2e] model={model}")
        _LOADED_MODELS.add(model)
        for name, scenario in scenarios:
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
