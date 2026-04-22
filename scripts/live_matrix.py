from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable


def build_workspace(root: Path) -> None:
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "notes").mkdir(parents=True, exist_ok=True)
    (root / "notes" / "alpha.txt").write_text("ORBIT\nsecond line\n", encoding="utf-8")
    (root / "docs" / "spec.md").write_text(
        "# Spec\n\nMAGIC_TOKEN appears here.\nSubagents should read this file.\n",
        encoding="utf-8",
    )
    (root / "src" / "sample.py").write_text(
        "def answer() -> int:\n    return 7 * 6\n",
        encoding="utf-8",
    )


def run_cli(repo_root: Path, workspace: Path, model: str, prompt: str, *, approval: str = "auto", timeout: int = 420) -> subprocess.CompletedProcess[str]:
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
        prompt,
    ]
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
        "Use your tools to inspect this workspace and tell me the three top-level entries. Mention docs, notes, and src.",
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
    result = run_cli(
        repo_root,
        workspace,
        model,
        "Use run_shell to execute exactly: pwd && python3 -c 'print(6*7)'. Then tell me the number and the directory.",
    )
    require(result, lambda r: "[status] tool run_shell" in r.stdout and "42" in r.stdout and str(workspace).replace("\\", "/") in r.stdout, "shell scenario failed")


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


def scenario_read_only(repo_root: Path, workspace: Path, model: str) -> None:
    blocked = workspace / "blocked.txt"
    result = run_cli(
        repo_root,
        workspace,
        model,
        "Try to create blocked.txt with any content. If you cannot, explain why.",
        approval="read-only",
    )
    require(
        result,
        lambda r: not blocked.exists() and "[status] tool write_file" in r.stdout and ("unable" in r.stdout.lower() or "denied" in r.stdout.lower() or "read-only" in r.stdout.lower()),
        "read-only guard scenario failed",
    )


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
    require(result, lambda r: model in r.stdout and "max_agent_depth=" in r.stdout, "REPL slash-command scenario failed")


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
    parser.add_argument("--models", nargs="+", default=["batiai/gemma4-26b:iq4", "gemma4"], help="Models to test.")
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
        scenario_write,
        scenario_replace,
        scenario_read_only,
        scenario_repl,
    ]
    for model in requested:
        with tempfile.TemporaryDirectory(prefix="ollama-code-live-", dir=repo_root) as tmp:
            workspace = Path(tmp)
            build_workspace(workspace)
            print(f"[live] model={model}")
            for scenario in scenarios:
                print(f"[live]   {scenario.__name__}")
                scenario(repo_root, workspace, model)
            print(f"[live]   scenario_subagent")
            scenario_subagent(repo_root, workspace, model, model)
    print("[live] all scenarios passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
