from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Callable


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


def build_workspace(root: Path) -> None:
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "scratch").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "guide.md").write_text(
        "# Guide\n\nTOKEN_42 lives here.\nThe CLI should read this file with tools.\n",
        encoding="utf-8",
    )
    (root / "src" / "app.py").write_text(
        "def meaning() -> int:\n    return 42\n",
        encoding="utf-8",
    )


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
    command.append(prompt)
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
    require(any(result.get("ok") for result in reads), "read_file did not succeed in transcript", stdout=result.stdout, stderr=result.stderr, session=session)
    require("TOKEN_42" in result.stdout, "final answer did not contain TOKEN_42", stdout=result.stdout, stderr=result.stderr, session=session)


def scenario_approval_accept(repo_root: Path, workspace: Path, model: str) -> None:
    session_file = workspace / "scratch" / "approve-yes.json"
    target = workspace / "scratch" / "approved.txt"
    result = run_cli(
        repo_root,
        workspace,
        model,
        "Create scratch/approved.txt with exactly the text approval accepted followed by a newline, then confirm the contents.",
        approval="ask",
        session_file=session_file,
        stdin_text="y\n",
    )
    session = load_session(session_file)
    writes = tool_results(session, "write_file")
    require(result.returncode == 0, "approval accept command failed", stdout=result.stdout, stderr=result.stderr, session=session)
    require(target.exists() and target.read_text(encoding="utf-8") == "approval accepted\n", "approved file content mismatch", stdout=result.stdout, stderr=result.stderr, session=session)
    require(any(result.get("ok") for result in writes), "write_file was not approved successfully", stdout=result.stdout, stderr=result.stderr, session=session)


def scenario_approval_reject(repo_root: Path, workspace: Path, model: str) -> None:
    session_file = workspace / "scratch" / "approve-no.json"
    target = workspace / "scratch" / "rejected.txt"
    result = run_cli(
        repo_root,
        workspace,
        model,
        "Create scratch/rejected.txt with any content and then tell me what happened.",
        approval="ask",
        session_file=session_file,
        stdin_text="n\n",
    )
    session = load_session(session_file)
    writes = tool_results(session, "write_file")
    require(result.returncode == 0, "approval reject command failed", stdout=result.stdout, stderr=result.stderr, session=session)
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
    require(any(item.get("exit_code") == 5 and item.get("output") == "boom" for item in shells), "shell failure details were not recorded", stdout=result.stdout, stderr=result.stderr, session=session)


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
    require(any(item.get("ok") and item.get("event_count", 0) >= 3 and "TOKEN_42" in str(item.get("output", "")) for item in subagents), "subagent result was not captured correctly", stdout=result.stdout, stderr=result.stderr, session=session)


def scenario_multiturn_repl(repo_root: Path, workspace: Path, model: str) -> None:
    session_file = workspace / "scratch" / "repl.json"
    repl_input = (
        "Create scratch/repl.txt with exactly the text repl ok followed by a newline.\n"
        "Use read_file on scratch/repl.txt and reply with the full line.\n"
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
    require((workspace / "scratch" / "repl.txt").read_text(encoding="utf-8") == "repl ok\n", "repl file content mismatch", stdout=result.stdout, stderr=result.stderr, session=session)
    require(saved.exists(), "manual /save did not create transcript", stdout=result.stdout, stderr=result.stderr, session=session)
    require("tool_call" in event_names(session) and "assistant" in event_names(session), "repl transcript missing expected events", stdout=result.stdout, stderr=result.stderr, session=session)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run stricter end-to-end checks against a real Ollama Code model.")
    parser.add_argument("--model", default="batiai/gemma4-26b:iq4", help="Model to test.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    model = resolve_model(args.model, set(installed_models()))

    scenarios: list[tuple[str, Callable[[Path, Path, str], None]]] = [
        ("scenario_transcripted_tool_use", scenario_transcripted_tool_use),
        ("scenario_approval_accept", scenario_approval_accept),
        ("scenario_approval_reject", scenario_approval_reject),
        ("scenario_path_escape", scenario_path_escape),
        ("scenario_shell_failure", scenario_shell_failure),
        ("scenario_subagent_transcript", scenario_subagent_transcript),
        ("scenario_multiturn_repl", scenario_multiturn_repl),
    ]

    with tempfile.TemporaryDirectory(prefix="ollama-code-e2e-", dir=repo_root) as tmp:
        workspace = Path(tmp)
        build_workspace(workspace)
        print(f"[e2e] model={model}")
        for name, scenario in scenarios:
            print(f"[e2e]   {name}")
            scenario(repo_root, workspace, model)
    print("[e2e] all scenarios passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
