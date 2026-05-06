from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Callable

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


def init_git_repo(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, capture_output=True, text=True, check=True)
    subprocess.run(["git", "config", "user.name", "Ollama Code Tests"], cwd=root, capture_output=True, text=True, check=True)
    subprocess.run(["git", "config", "user.email", "tests@example.com"], cwd=root, capture_output=True, text=True, check=True)
    subprocess.run(["git", "add", "."], cwd=root, capture_output=True, text=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=root, capture_output=True, text=True, check=True)


def commit_all(root: Path, message: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=root, capture_output=True, text=True, check=True)
    status = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root, capture_output=True, text=True, check=False)
    if status.returncode != 0:
        subprocess.run(["git", "commit", "-m", message], cwd=root, capture_output=True, text=True, check=True)


def build_workspace(root: Path) -> None:
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "notes").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / ".gitignore").write_text(".ollama-code/\n", encoding="utf-8")
    (root / "notes" / "alpha.txt").write_text("ORBIT\nsecond line\n", encoding="utf-8")
    (root / "docs" / "spec.md").write_text(
        "# Spec\n\nMAGIC_TOKEN appears here.\nSubagents should read this file.\n",
        encoding="utf-8",
    )
    (root / "src" / "sample.py").write_text(
        "def answer() -> int:\n    return 7 * 6\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_sample.py").write_text(
        "import unittest\n\n\nclass SampleTests(unittest.TestCase):\n    def test_truth(self) -> None:\n        self.assertEqual(6 * 7, 42)\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
        encoding="utf-8",
    )
    init_git_repo(root)


def python_command(*args: str) -> str:
    return subprocess.list2cmdline([sys.executable, *args])


def run_cli(
    repo_root: Path,
    workspace: Path,
    model: str,
    prompt: str,
    *,
    approval: str = "auto",
    timeout: int = 420,
    session_file: Path | None = None,
    extra_args: list[str] | None = None,
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
    if session_file is not None:
        command.extend(["--session-file", str(session_file)])
    if extra_args:
        command.extend(extra_args)
    command.append(prompt)
    return subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=os.environ.copy(),
        check=False,
    )


def last_non_status_line(stdout: str) -> str:
    lines = [line.strip() for line in stdout.splitlines() if line.strip() and not line.startswith("[status]")]
    return lines[-1] if lines else ""


def load_session(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def tool_results(session: dict[str, object], tool_name: str) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    events = session.get("events")
    if not isinstance(events, list):
        return results
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("type") != "tool_result" or event.get("name") != tool_name:
            continue
        result = event.get("result")
        if isinstance(result, dict):
            results.append(result)
    return results


def tool_calls(session: dict[str, object]) -> list[str]:
    names: list[str] = []
    events = session.get("events")
    if not isinstance(events, list):
        return names
    for event in events:
        if isinstance(event, dict) and event.get("type") == "tool_call" and isinstance(event.get("name"), str):
            names.append(str(event["name"]))
    return names


def require(result: subprocess.CompletedProcess[str], predicate: Callable[[subprocess.CompletedProcess[str]], bool], message: str) -> None:
    if result.returncode != 0:
        raise AssertionError(f"{message}: exit code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    if not predicate(result):
        raise AssertionError(f"{message}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


def scenario_filesystem(repo_root: Path, workspace: Path, model: str) -> None:
    result = run_cli(
        repo_root,
        workspace,
        model,
        "Use list_files on . with a high enough limit to inspect the workspace. Reply with docs, notes, and src only.",
    )
    require(result, lambda r: "[status] tool list_files" in r.stdout and "docs" in r.stdout and "notes" in r.stdout and "src" in r.stdout, "filesystem inspection failed")


def scenario_read_file(repo_root: Path, workspace: Path, model: str) -> None:
    result = run_cli(
        repo_root,
        workspace,
        model,
        "Use read_file on notes/alpha.txt and reply with exactly the text on line 2.",
    )
    require(result, lambda r: "[status] tool read_file" in r.stdout and last_non_status_line(r.stdout) == "second line", "read_file scenario failed")


def scenario_search(repo_root: Path, workspace: Path, model: str) -> None:
    result = run_cli(
        repo_root,
        workspace,
        model,
        "Use the search tool to find MAGIC_TOKEN and tell me which file contains it.",
    )
    require(result, lambda r: "[status] tool search" in r.stdout and "docs/spec.md" in r.stdout, "search scenario failed")


def scenario_shell(repo_root: Path, workspace: Path, model: str) -> None:
    session_file = workspace / ".ollama-code" / "shell.json"
    command = python_command("-c", "import os; print(os.getcwd()); print(6*7)")
    result = run_cli(
        repo_root,
        workspace,
        model,
        f"Use run_shell to execute exactly: {command}. Then tell me the number and the directory.",
        session_file=session_file,
    )
    session = load_session(session_file)
    shells = tool_results(session, "run_shell")
    require(result, lambda r: "[status] tool run_shell" in r.stdout and "42" in r.stdout, "shell scenario failed")
    expected_workspace = str(workspace).replace("\\", "/")
    if not any(
        item.get("ok") and "42" in str(item.get("output", "")) and expected_workspace in str(item.get("output", "")).replace("\\", "/")
        for item in shells
    ):
        raise AssertionError(f"shell tool output was not captured correctly\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


def scenario_run_test(repo_root: Path, workspace: Path, model: str) -> None:
    session_file = workspace / ".ollama-code" / "run-test.json"
    result = run_cli(
        repo_root,
        workspace,
        model,
        "Use run_test and tell me whether the test suite passed.",
        session_file=session_file,
        extra_args=["--test-cmd", python_command("-m", "unittest", "discover", "-s", "tests", "-v")],
    )
    session = load_session(session_file)
    tests = tool_results(session, "run_test")
    require(
        result,
        lambda r: "[status] tool run_test" in r.stdout and ("passed" in r.stdout.lower() or "\nok\n" in r.stdout.lower()),
        "run_test scenario failed",
    )
    if not any(item.get("ok") and "OK" in str(item.get("output", "")) for item in tests):
        raise AssertionError(f"run_test tool output was not captured correctly\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


def scenario_write(repo_root: Path, workspace: Path, model: str) -> None:
    result = run_cli(
        repo_root,
        workspace,
        model,
        "Create edits/result.txt with exactly the text 'status ok' followed by a newline. Use the write_file tool and then confirm the final contents.",
    )
    target = workspace / "edits" / "result.txt"
    require(
        result,
        lambda r: "[status] tool write_file" in r.stdout and target.exists() and target.read_text(encoding="utf-8") == "status ok\n",
        "write scenario failed",
    )


def scenario_replace(repo_root: Path, workspace: Path, model: str) -> None:
    target = workspace / "edits" / "result.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("status ok\n", encoding="utf-8")
    result = run_cli(
        repo_root,
        workspace,
        model,
        "Use replace_in_file on edits/result.txt to replace the standalone word ok with passed. Set match_whole_word to true. Then reply with the final file contents only.",
    )
    require(
        result,
        lambda r: "[status] tool replace_in_file" in r.stdout and target.read_text(encoding="utf-8") == "status passed\n",
        "replace scenario failed",
    )


def scenario_subagent(repo_root: Path, workspace: Path, model: str, helper_model: str) -> None:
    result = run_cli(
        repo_root,
        workspace,
        model,
        f"Use run_agent to start a helper agent with model {helper_model}. Tell the helper: do not modify any files, read docs/spec.md, and report the exact uppercase token that already exists in the file. Then reply with that token only.",
    )
    require(
        result,
        lambda r: "[status] tool run_agent" in r.stdout and "subagent" in r.stdout and "MAGIC_TOKEN" in last_non_status_line(r.stdout),
        "sub-agent scenario failed",
    )


def scenario_git_tools(repo_root: Path, workspace: Path, model: str) -> None:
    session_file = workspace / ".ollama-code" / "git-tools.json"
    commit_all(workspace, "checkpoint before git scenario")
    target = workspace / "src" / "sample.py"
    target.write_text("def answer() -> int:\n    return 99\n", encoding="utf-8")
    result = run_cli(
        repo_root,
        workspace,
        model,
        "Use git_status on src/sample.py. Then use git_diff on src/sample.py for the working tree only; omit cached or set cached to false. Do not use read_file. Reply with src/sample.py and whether the diff adds return 99.",
        session_file=session_file,
    )
    session = load_session(session_file)
    statuses = tool_results(session, "git_status")
    diffs = tool_results(session, "git_diff")
    calls = tool_calls(session)
    require(
        result,
        lambda r: "[status] tool git_status" in r.stdout and "[status] tool git_diff" in r.stdout,
        "git tool scenario failed",
    )
    if not any(item.get("ok") and "src/sample.py" in str(item.get("output", "")) for item in statuses):
        raise AssertionError(f"git_status tool output was not captured correctly\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    if not any(item.get("ok") and "return 99" in str(item.get("output", "")) for item in diffs):
        raise AssertionError(f"git_diff tool output was not captured correctly\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    if "read_file" in calls:
        raise AssertionError(f"git tool scenario used forbidden read_file\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


def scenario_read_only(repo_root: Path, workspace: Path, model: str) -> None:
    blocked = workspace / "blocked.txt"
    session_file = workspace / ".ollama-code" / "read-only.json"
    result = run_cli(
        repo_root,
        workspace,
        model,
        "Try to create blocked.txt with any content. If you cannot, explain why.",
        approval="read-only",
        session_file=session_file,
    )
    session = load_session(session_file)
    writes = tool_results(session, "write_file")
    require(
        result,
        lambda r: not blocked.exists()
        and "[status] tool write_file" in r.stdout
        and (
            "unable" in r.stdout.lower()
            or "denied" in r.stdout.lower()
            or "read-only" in r.stdout.lower()
            or "grounded final verification could not accept a final answer" in r.stdout.lower()
        ),
        "read-only guard scenario failed",
    )
    if not any(not item.get("ok") and "read-only" in str(item.get("summary", "")).lower() for item in writes):
        raise AssertionError(f"read-only denial was not captured in transcript\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


def scenario_repl(repo_root: Path, workspace: Path, model: str) -> None:
    command = [
        sys.executable,
        "-m",
        "ollama_code.cli",
        "--cwd",
        str(workspace),
        "--model",
        model,
        "--quiet",
    ]
    result = subprocess.run(
        command,
        cwd=repo_root,
        input="/models\n/status\n/quit\n",
        capture_output=True,
        text=True,
        timeout=120,
        env=os.environ.copy(),
        check=False,
    )
    require(result, lambda r: model in r.stdout and "max_agent_depth=" in r.stdout and "session=" in r.stdout, "REPL slash-command scenario failed")


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


def resolve_requested_models(requested: list[str], available: set[str]) -> list[str]:
    resolved: list[str] = []
    for model in requested:
        if model in available:
            resolved.append(model)
            continue
        latest = f"{model}:latest"
        if latest in available:
            resolved.append(latest)
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run live Ollama Code smoke tests against real local models.")
    parser.add_argument("--models", nargs="+", default=["gemma4:e4b", "granite4.1:8b", "gemma3:4b", "qwen3:8b"], help="Models to test.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    available = set(installed_models())
    requested = resolve_requested_models(args.models, available)
    if not requested:
        print("No requested models are installed locally.", file=sys.stderr)
        return 1

    scenarios = [
        scenario_filesystem,
        scenario_read_file,
        scenario_search,
        scenario_shell,
        scenario_run_test,
        scenario_write,
        scenario_replace,
        scenario_git_tools,
        scenario_read_only,
        scenario_repl,
    ]
    for model in requested:
        with workspace_temp_dir("ollama-code-live-", repo_root) as tmp:
            workspace = Path(tmp)
            build_workspace(workspace)
            print(f"[live] model={model}")
            _LOADED_MODELS.add(model)
            for scenario in scenarios:
                print(f"[live]   {scenario.__name__}")
                scenario(repo_root, workspace, model)
            print("[live]   scenario_subagent")
            scenario_subagent(repo_root, workspace, model, model)
        unload_model(model)
        _LOADED_MODELS.discard(model)
    print("[live] all scenarios passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
