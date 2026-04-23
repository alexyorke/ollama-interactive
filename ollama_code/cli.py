from __future__ import annotations

import argparse
import io
import os
import shlex
import sys
import threading
from pathlib import Path
from typing import Callable

from ollama_code.agent import OllamaCodeAgent
from ollama_code.config import (
    DEFAULT_APPROVAL_MODE,
    DEFAULT_MAX_AGENT_DEPTH,
    DEFAULT_MAX_TOOL_ROUNDS,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    ENV_OLLAMA_CODE_MODEL,
    ENV_OLLAMA_CODE_TEST_CMD,
    ENV_OLLAMA_HOST,
    load_config,
)
from ollama_code.interrupts import InterruptController, OperationInterrupted
from ollama_code.ollama_client import OllamaClient, OllamaError
from ollama_code.sessions import latest_session_path, load_transcript_payload, new_session_path, resolve_transcript_path
from ollama_code.tools import ToolExecutor


class CliStatusRenderer:
    def __init__(self, stream: io.TextIOBase | None = None, *, use_ansi: bool | None = None) -> None:
        self.stream = stream or sys.stdout
        self.use_ansi = self.stream.isatty() if use_ansi is None else use_ansi
        self._thinking_lines = 0
        self._lock = threading.Lock()

    def _write_locked(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()

    def _clear_thinking_locked(self) -> None:
        if not self.use_ansi or self._thinking_lines <= 0:
            return
        for _ in range(self._thinking_lines):
            self._write_locked("\x1b[1A\r\x1b[2K\x1b[M")
        self._write_locked("\r")
        self._thinking_lines = 0

    def clear_thinking(self) -> None:
        with self._lock:
            self._clear_thinking_locked()

    def status(self, message: str) -> None:
        with self._lock:
            self._clear_thinking_locked()
            self._write_locked(f"[status] {message}\n")

    def write(self, message: str = "") -> None:
        with self._lock:
            self._clear_thinking_locked()
            self._write_locked(f"{message}\n")

    def show_thinking(self, thinking: str) -> None:
        if not self.use_ansi:
            return
        lines = [line.rstrip() for line in thinking.splitlines() if line.strip()]
        if not lines:
            return
        tail = lines[-3:]
        with self._lock:
            self._clear_thinking_locked()
            for line in tail:
                self._write_locked(f"\x1b[90m{line}\x1b[0m\n")
            self._thinking_lines = len(tail)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ollama-code",
        description="Local coding CLI powered by Ollama.",
    )
    parser.add_argument("prompt", nargs="*", help="Optional one-shot prompt.")
    parser.add_argument("--model", default=None, help="Ollama model name.")
    parser.add_argument("--host", default=None, help="Override the Ollama host.")
    parser.add_argument("--config", default=None, help="Optional config file path. Defaults to .ollama-code/config.json inside the workspace.")
    parser.add_argument("--cwd", default=".", help="Workspace root.")
    parser.add_argument(
        "--approval",
        choices=["ask", "auto", "read-only"],
        default=None,
        help="Approval policy for writes and shell commands.",
    )
    session_group = parser.add_mutually_exclusive_group()
    session_group.add_argument("--continue", dest="continue_session", action="store_true", help="Resume the most recent saved session in the workspace.")
    session_group.add_argument("--resume", default=None, help="Resume a saved session from a transcript path.")
    parser.add_argument("--max-tool-rounds", type=int, default=None, help="Maximum tool rounds per user turn.")
    parser.add_argument("--max-agent-depth", type=int, default=None, help="Maximum nested sub-agent depth.")
    parser.add_argument("--timeout", type=int, default=None, help="Ollama request timeout in seconds.")
    parser.add_argument("--test-cmd", default=None, help="Optional default test command for the run_test tool and /test.")
    parser.add_argument("--session-file", default=None, help="Optional JSON transcript path. Defaults to a local auto-saved session file.")
    parser.add_argument("--quiet", action="store_true", help="Suppress banner and status lines.")
    return parser


def _non_empty_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def build_agent(
    args: argparse.Namespace,
    *,
    input_func: Callable[[str], str] = input,
    status_printer: Callable[[str], None] | None = None,
    thinking_printer: Callable[[str], None] | None = None,
) -> OllamaCodeAgent:
    workspace_root = Path(args.cwd).resolve()
    config = load_config(workspace_root, args.config)
    restored_payload: dict[str, object] | None = None
    resume_path: Path | None = None
    if args.resume:
        resume_path = resolve_transcript_path(workspace_root, args.resume)
        restored_payload = load_transcript_payload(resume_path)
    elif args.continue_session:
        resume_path = latest_session_path(workspace_root)
        if resume_path is None:
            raise ValueError(f"No saved sessions found in {workspace_root.as_posix()}/.ollama-code/sessions")
        restored_payload = load_transcript_payload(resume_path)

    model = config.model or DEFAULT_MODEL
    env_model = _non_empty_string(os.environ.get(ENV_OLLAMA_CODE_MODEL))
    if env_model:
        model = env_model
    approval = config.approval or DEFAULT_APPROVAL_MODE
    if restored_payload is not None:
        saved_model = restored_payload.get("model")
        saved_approval = restored_payload.get("approval_mode")
        explicit_model = _non_empty_string(args.model)
        if explicit_model is None and isinstance(saved_model, str) and saved_model.strip():
            model = saved_model
        if args.approval is None and saved_approval in {"ask", "auto", "read-only"}:
            approval = str(saved_approval)
    explicit_model = _non_empty_string(args.model)
    if explicit_model:
        model = explicit_model
    host = config.host
    env_host = _non_empty_string(os.environ.get(ENV_OLLAMA_HOST))
    if env_host:
        host = env_host
    explicit_host = _non_empty_string(args.host)
    if explicit_host:
        host = explicit_host
    if args.approval is not None:
        approval = args.approval
    max_tool_rounds = args.max_tool_rounds if args.max_tool_rounds is not None else (config.max_tool_rounds or DEFAULT_MAX_TOOL_ROUNDS)
    max_agent_depth = args.max_agent_depth if args.max_agent_depth is not None else (config.max_agent_depth or DEFAULT_MAX_AGENT_DEPTH)
    timeout = args.timeout if args.timeout is not None else (config.timeout or DEFAULT_TIMEOUT)
    session_file = args.session_file
    if session_file is None:
        session_file = resume_path or new_session_path(workspace_root)
    active_session_file = None if restored_payload is not None else session_file
    client = OllamaClient(host=host, timeout=timeout)
    test_command = config.test_cmd
    env_test_command = _non_empty_string(os.environ.get(ENV_OLLAMA_CODE_TEST_CMD))
    if env_test_command:
        test_command = env_test_command
    explicit_test_command = _non_empty_string(args.test_cmd)
    if explicit_test_command:
        test_command = explicit_test_command
    tools = ToolExecutor(
        workspace_root,
        approval_mode=approval,
        input_func=input_func,
        test_command=test_command,
    )
    resolved_status_printer = status_printer or ((lambda message: None) if args.quiet else (lambda message: print(f"[status] {message}")))
    agent = OllamaCodeAgent(
        client=client,
        tools=tools,
        model=model,
        max_tool_rounds=max_tool_rounds,
        session_file=active_session_file,
        status_printer=resolved_status_printer,
        thinking_printer=thinking_printer,
        max_agent_depth=max_agent_depth,
    )
    if restored_payload is not None:
        agent.restore_transcript(restored_payload)
        agent.session_file = resolve_transcript_path(workspace_root, session_file)
    return agent


def print_banner(agent: OllamaCodeAgent) -> None:
    print("Ollama Code")
    print(f"workspace: {agent.workspace_root().as_posix()}")
    print(f"model: {agent.model}")
    print(f"approval: {agent.approval_mode()}")
    if agent.configured_test_command():
        print(f"test_cmd: {agent.configured_test_command()}")
    if agent.session_path() is not None:
        print(f"session: {agent.session_path().as_posix()}")
    print("press Esc to interrupt the current model or tool call")
    print("commands: /help /status /models /model /approval /reset /save /sessions /load /git /diff /commit /test /tools /quit")


def _strip_matching_quotes(text: str) -> str:
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        return stripped[1:-1]
    return stripped


def handle_meta_command(command: str, agent: OllamaCodeAgent, writer: Callable[[str], None]) -> bool | None:
    if not command.startswith("/"):
        return None
    head, _, tail = command.strip().partition(" ")
    action = head.lower()
    remainder = tail.strip()
    if action in {"/quit", "/exit"}:
        return False
    if action == "/help":
        writer(
            "Slash commands: /help /status /models /model <name> /approval <mode> /reset /save [path] /sessions [limit] "
            "/load <path> /git /diff [--cached] [path] /commit <message> /test [command] /tools /quit\nPress Esc during model "
            "or tool execution to interrupt the current task."
        )
        return True
    if action == "/status":
        session = agent.session_path().as_posix() if agent.session_path() is not None else "(none)"
        test_command = agent.configured_test_command() or "(none)"
        writer(
            f"workspace={agent.workspace_root().as_posix()} model={agent.model} approval={agent.approval_mode()} max_tool_rounds={agent.max_tool_rounds} max_agent_depth={agent.max_agent_depth} test_cmd={test_command} session={session}"
        )
        return True
    if action == "/models":
        models = agent.list_models()
        writer(", ".join(models) if models else "(no local models)")
        return True
    if action == "/model":
        model_name = _strip_matching_quotes(remainder)
        if not model_name:
            writer("Usage: /model <name>")
            return True
        agent.set_model(model_name)
        writer(f"model set to {agent.model}")
        return True
    if action == "/approval":
        mode = _strip_matching_quotes(remainder)
        if mode not in {"ask", "auto", "read-only"}:
            writer("Usage: /approval ask|auto|read-only")
            return True
        agent.set_approval_mode(mode)
        writer(f"approval set to {agent.approval_mode()}")
        return True
    if action == "/reset":
        agent.reset()
        writer("conversation reset")
        return True
    if action == "/save":
        target = _strip_matching_quotes(remainder) or None
        saved = agent.save_transcript(target)
        writer(f"saved transcript to {saved.as_posix()}")
        return True
    if action == "/sessions":
        limit = 10
        if remainder:
            try:
                limit = max(1, int(_strip_matching_quotes(remainder)))
            except ValueError:
                writer("Usage: /sessions [limit]")
                return True
        sessions = agent.list_sessions(limit=limit)
        if not sessions:
            writer("(no saved sessions)")
            return True
        lines = [
            f"{index}. {item.path.name} | model={item.model or '(unknown)'} | approval={item.approval_mode or '(unknown)'} | messages={item.message_count} | updated={item.updated_at.strftime('%Y-%m-%d %H:%M:%S')} | {item.summary}"
            for index, item in enumerate(sessions, start=1)
        ]
        writer("\n".join(lines))
        return True
    if action == "/load":
        target = _strip_matching_quotes(remainder)
        if not target:
            writer("Usage: /load <path>")
            return True
        loaded = agent.load_session(target)
        writer(f"loaded session {loaded.as_posix()}")
        return True
    if action == "/git":
        result = agent.git_status()
        writer(str(result.get("output") or result.get("summary", "(no output)")))
        return True
    if action == "/diff":
        try:
            diff_parts = shlex.split(remainder, posix=os.name != "nt") if remainder else []
        except ValueError:
            writer("Usage: /diff [--cached] [path]")
            return True
        cached = False
        path = None
        for token in (_strip_matching_quotes(part) for part in diff_parts):
            if token == "--cached":
                cached = True
            elif path is None:
                path = token
            else:
                writer("Usage: /diff [--cached] [path]")
                return True
        result = agent.git_diff(cached=cached, path=path)
        writer(str(result.get("output") or result.get("summary", "(no output)")))
        return True
    if action == "/commit":
        message = _strip_matching_quotes(remainder)
        if not message:
            writer("Usage: /commit <message>")
            return True
        result = agent.git_commit(message)
        output = str(result.get("output") or result.get("summary", "(no output)"))
        writer(output)
        return True
    if action == "/test":
        command_text = remainder or None
        result = agent.run_test(command_text)
        output = str(result.get("output") or result.get("summary", "(no output)"))
        writer(output)
        return True
    if action == "/tools":
        writer(agent.tool_help())
        return True
    writer(f"Unknown command: {command}")
    return True


def run_repl(agent: OllamaCodeAgent, *, quiet: bool = False, renderer: CliStatusRenderer | None = None) -> int:
    renderer = renderer or CliStatusRenderer()
    if not quiet:
        print_banner(agent)
    interrupt_controller = InterruptController()
    while True:
        try:
            raw = input("ollama-code> ").strip()
        except EOFError:
            print()
            return 0
        if not raw:
            continue
        try:
            with interrupt_controller.watch() as interrupt_event:
                agent.set_interrupt_event(interrupt_event)
                handled = handle_meta_command(raw, agent, renderer.write)
                if handled is False:
                    return 0
                if handled is True:
                    continue
                result = agent.handle_user(raw)
        except OllamaError as exc:
            renderer.clear_thinking()
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except OperationInterrupted:
            renderer.write("interrupted")
            continue
        finally:
            agent.set_interrupt_event(None)
        renderer.write(result.message)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    renderer = CliStatusRenderer()
    try:
        agent = build_agent(args, status_printer=renderer.status, thinking_printer=renderer.show_thinking)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.prompt:
        try:
            with InterruptController(lambda _: None).watch() as interrupt_event:
                agent.set_interrupt_event(interrupt_event)
                result = agent.handle_user(" ".join(args.prompt))
        except OllamaError as exc:
            renderer.clear_thinking()
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except OperationInterrupted:
            renderer.clear_thinking()
            print("interrupted", file=sys.stderr)
            return 130
        finally:
            agent.set_interrupt_event(None)
        renderer.write(result.message)
        return 0
    return run_repl(agent, quiet=args.quiet, renderer=renderer)


if __name__ == "__main__":
    raise SystemExit(main())
