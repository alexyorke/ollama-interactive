from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Callable

from ollama_code.agent import OllamaCodeAgent
from ollama_code.ollama_client import OllamaClient, OllamaError
from ollama_code.tools import ToolExecutor

DEFAULT_MODEL = "batiai/gemma4-26b:iq4"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ollama-code",
        description="Local coding CLI powered by Ollama.",
    )
    parser.add_argument("prompt", nargs="*", help="Optional one-shot prompt.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument("--host", default=None, help="Override the Ollama host.")
    parser.add_argument("--cwd", default=".", help="Workspace root.")
    parser.add_argument(
        "--approval",
        choices=["ask", "auto", "read-only"],
        default="ask",
        help="Approval policy for writes and shell commands.",
    )
    parser.add_argument("--max-tool-rounds", type=int, default=8, help="Maximum tool rounds per user turn.")
    parser.add_argument("--max-agent-depth", type=int, default=2, help="Maximum nested sub-agent depth.")
    parser.add_argument("--timeout", type=int, default=300, help="Ollama request timeout in seconds.")
    parser.add_argument("--session-file", default=None, help="Optional JSON transcript path.")
    parser.add_argument("--quiet", action="store_true", help="Suppress banner and status lines.")
    return parser


def build_agent(args: argparse.Namespace, *, input_func: Callable[[str], str] = input) -> OllamaCodeAgent:
    client = OllamaClient(host=args.host, timeout=args.timeout)
    tools = ToolExecutor(
        Path(args.cwd).resolve(),
        approval_mode=args.approval,
        input_func=input_func,
    )
    status_printer = (lambda message: None) if args.quiet else (lambda message: print(f"[status] {message}"))
    return OllamaCodeAgent(
        client=client,
        tools=tools,
        model=args.model,
        max_tool_rounds=args.max_tool_rounds,
        session_file=args.session_file,
        status_printer=status_printer,
        max_agent_depth=args.max_agent_depth,
    )


def print_banner(agent: OllamaCodeAgent) -> None:
    print("Ollama Code")
    print(f"workspace: {agent.workspace_root().as_posix()}")
    print(f"model: {agent.model}")
    print(f"approval: {agent.approval_mode()}")
    print("commands: /help /status /models /model /approval /reset /save /tools /quit")


def handle_meta_command(command: str, agent: OllamaCodeAgent, writer: Callable[[str], None]) -> bool | None:
    if not command.startswith("/"):
        return None
    parts = shlex.split(command)
    action = parts[0].lower()
    if action in {"/quit", "/exit"}:
        return False
    if action == "/help":
        writer("Slash commands: /help /status /models /model <name> /approval <mode> /reset /save [path] /tools /quit")
        return True
    if action == "/status":
        writer(
            f"workspace={agent.workspace_root().as_posix()} model={agent.model} approval={agent.approval_mode()} max_tool_rounds={agent.max_tool_rounds} max_agent_depth={agent.max_agent_depth}"
        )
        return True
    if action == "/models":
        models = agent.list_models()
        writer(", ".join(models) if models else "(no local models)")
        return True
    if action == "/model":
        if len(parts) < 2:
            writer("Usage: /model <name>")
            return True
        agent.set_model(parts[1])
        writer(f"model set to {agent.model}")
        return True
    if action == "/approval":
        if len(parts) < 2 or parts[1] not in {"ask", "auto", "read-only"}:
            writer("Usage: /approval ask|auto|read-only")
            return True
        agent.set_approval_mode(parts[1])
        writer(f"approval set to {agent.approval_mode()}")
        return True
    if action == "/reset":
        agent.reset()
        writer("conversation reset")
        return True
    if action == "/save":
        target = parts[1] if len(parts) > 1 else None
        saved = agent.save_transcript(target)
        writer(f"saved transcript to {saved.as_posix()}")
        return True
    if action == "/tools":
        writer(agent.tool_help())
        return True
    writer(f"Unknown command: {command}")
    return True


def run_repl(agent: OllamaCodeAgent, *, quiet: bool = False) -> int:
    if not quiet:
        print_banner(agent)
    while True:
        try:
            raw = input("ollama-code> ").strip()
        except EOFError:
            print()
            return 0
        if not raw:
            continue
        handled = handle_meta_command(raw, agent, print)
        if handled is False:
            return 0
        if handled is True:
            continue
        try:
            result = agent.handle_user(raw)
        except OllamaError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(result.message)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    agent = build_agent(args)
    if args.prompt:
        try:
            result = agent.handle_user(" ".join(args.prompt))
        except OllamaError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(result.message)
        return 0
    return run_repl(agent, quiet=args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
