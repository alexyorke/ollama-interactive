from __future__ import annotations

import argparse
import io
import os
import re
import shutil
import shlex
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from ollama_code.agent import OllamaCodeAgent
from ollama_code.config import (
    DEFAULT_APPROVAL_MODE,
    DEFAULT_DEBATE_ENABLED,
    DEFAULT_MAX_AGENT_DEPTH,
    DEFAULT_MAX_TOOL_ROUNDS,
    DEFAULT_MODEL,
    DEFAULT_RECONCILE_MODE,
    DEFAULT_TIMEOUT,
    ENV_OLLAMA_CODE_DEBATE,
    ENV_OLLAMA_CODE_MODEL,
    ENV_OLLAMA_CODE_RECONCILE,
    ENV_OLLAMA_CODE_TEST_CMD,
    ENV_OLLAMA_CODE_DISABLE_SPEC_GUIDED_REPAIR,
    ENV_OLLAMA_CODE_REQUIRE_LLM_FOR_TURN,
    ENV_OLLAMA_CODE_VERIFIER_MODEL,
    ENV_OLLAMA_HOST,
    OFFICIAL_GRANITE_8B_MODEL,
    load_config,
)
from ollama_code.indexer import BackgroundIndexer
from ollama_code.interrupts import InterruptController, OperationInterrupted
from ollama_code.ollama_client import OllamaClient, OllamaError
from ollama_code.sessions import latest_restorable_session, load_transcript_payload, new_session_path, resolve_transcript_path
from ollama_code.tool_dependencies import dependency_statuses
from ollama_code.tools import ToolExecutor

PREFERRED_FALLBACK_MODELS = [
    DEFAULT_MODEL,
    "gemma4:e4b-it-q4_K_M",
    "gemma4:26b",
    OFFICIAL_GRANITE_8B_MODEL,
    "gemma3:12b",
    "gemma3:4b",
    "qwen3:8b",
    "gpt-oss:20b",
]
DEFAULT_MODEL_PULL_HINT = f"Install the recommended default with: ollama pull {DEFAULT_MODEL}"
WINDOWS_DRIVE_PATH = re.compile(r"^(?P<drive>[A-Za-z]):(?:[\\/](?P<rest>.*))?$")
WSL_MOUNT_PATH = re.compile(r"^/mnt/(?P<drive>[A-Za-z])(?:/(?P<rest>.*))?$")


class CliStatusRenderer:
    def __init__(
        self,
        stream: io.TextIOBase | None = None,
        *,
        use_ansi: bool | None = None,
        update_interval: float = 0.1,
    ) -> None:
        self.stream = stream or sys.stdout
        self.use_ansi = self.stream.isatty() if use_ansi is None else use_ansi
        self.update_interval = max(0.0, update_interval)
        self._thinking_lines = 0
        self._last_thinking_tail: tuple[str, ...] = ()
        self._last_render_at = 0.0
        self._lock = threading.Lock()

    def _write_locked(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()

    def _clear_thinking_locked(self) -> None:
        if not self.use_ansi or self._thinking_lines <= 0:
            self._last_thinking_tail = ()
            self._last_render_at = 0.0
            return
        for _ in range(self._thinking_lines):
            self._write_locked("\x1b[1A\r\x1b[2K\x1b[M")
        self._write_locked("\r")
        self._thinking_lines = 0
        self._last_thinking_tail = ()
        self._last_render_at = 0.0

    def _rewrite_last_thinking_line_locked(self, line: str) -> None:
        if not self.use_ansi or self._thinking_lines <= 0:
            return
        self._write_locked("\x1b[1A\r\x1b[2K")
        self._write_locked(f"\x1b[90m{line}\x1b[0m\n")

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
        tail = tuple(lines[-3:])
        with self._lock:
            now = time.monotonic()
            if tail == self._last_thinking_tail:
                return
            if self.update_interval > 0 and self._last_thinking_tail and (now - self._last_render_at) < self.update_interval:
                return
            previous_tail = self._last_thinking_tail
            if (
                previous_tail
                and len(previous_tail) == len(tail)
                and previous_tail[:-1] == tail[:-1]
                and len(previous_tail) <= self._thinking_lines
            ):
                self._rewrite_last_thinking_line_locked(tail[-1])
                self._last_thinking_tail = tail
                self._last_render_at = now
                return
            self._clear_thinking_locked()
            for line in tail:
                self._write_locked(f"\x1b[90m{line}\x1b[0m\n")
            self._thinking_lines = len(tail)
            self._last_thinking_tail = tail
            self._last_render_at = now


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
    parser.add_argument("--debate", choices=["on", "off"], default=None, help="Enable or disable tool-step assumption auditing plus grounded risky-final verification.")
    parser.add_argument("--reconcile", choices=["off", "on", "auto"], default=None, help="Artifact reconciliation after failed tests/edits. Default: auto.")
    parser.add_argument("--verifier-model", default=None, help="Optional model override for grounded final verification and evidence-backed rewrite.")
    parser.add_argument("--session-file", default=None, help="Optional JSON transcript path. Defaults to a local auto-saved session file.")
    parser.add_argument("--no-indexer", action="store_true", help="Disable the background repo indexer for this run.")
    parser.add_argument("--disable-spec-guided-repair", action="store_true", help="Disable spec-guided mechanical repair.")
    parser.add_argument("--require-llm-for-turn", action="store_true", help="Require at least one real model call for each prompt turn.")
    parser.add_argument("--doctor", action="store_true", help="Check Ollama, model, workspace, and optional local tool availability, then exit.")
    parser.add_argument("--quiet", action="store_true", help="Suppress banner and status lines.")
    return parser


def _non_empty_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _positive_int_argument(value: object, flag: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{flag} must be a positive integer.")
    return value


def _bool_from_text(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return None


def _split_meta_args(value: str) -> list[str]:
    posix_mode = os.name != "nt"
    if posix_mode and re.search(r"\\\S", value):
        return shlex.split(value, posix=False)
    return shlex.split(value, posix=posix_mode)


def _parse_single_meta_path(value: str) -> str | None:
    stripped = value.strip()
    if not stripped:
        return None
    parts = _split_meta_args(stripped)
    if len(parts) != 1:
        raise ValueError("expected a single path argument")
    return _strip_matching_quotes(parts[0])


def _resolve_workspace_root(raw_path: str | Path) -> Path:
    text = str(raw_path).strip()
    candidate = Path(text)
    if not candidate.is_absolute():
        normalized = text.replace("\\", "/")
        windows_match = WINDOWS_DRIVE_PATH.match(normalized)
        if windows_match:
            drive = windows_match.group("drive").lower()
            rest = (windows_match.group("rest") or "").strip("/")
            suffix = f"/{rest}" if rest else ""
            candidate = Path(f"/mnt/{drive}{suffix}")
        else:
            wsl_match = WSL_MOUNT_PATH.match(normalized)
            if wsl_match:
                drive = wsl_match.group("drive").upper()
                rest = (wsl_match.group("rest") or "").strip("/")
                candidate = Path(f"{drive}:/{rest}") if rest else Path(f"{drive}:/")
            elif os.name != "nt" and "\\" in text:
                candidate = Path(normalized)
    return candidate.resolve(strict=False)


def _reconcile_from_text(value: str | None) -> str | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    return lowered if lowered in {"off", "on", "auto"} else None


def _llm_call_count(agent: OllamaCodeAgent) -> int:
    return sum(1 for event in agent.events if isinstance(event, dict) and event.get("type") == "llm_call")


def build_agent(
    args: argparse.Namespace,
    *,
    input_func: Callable[[str], str] = input,
    status_printer: Callable[[str], None] | None = None,
    thinking_printer: Callable[[str], None] | None = None,
) -> OllamaCodeAgent:
    workspace_root = _resolve_workspace_root(args.cwd)
    config = load_config(workspace_root, args.config)
    explicit_max_tool_rounds = _positive_int_argument(args.max_tool_rounds, "--max-tool-rounds")
    explicit_max_agent_depth = _positive_int_argument(args.max_agent_depth, "--max-agent-depth")
    explicit_timeout = _positive_int_argument(args.timeout, "--timeout")
    restored_payload: dict[str, object] | None = None
    resume_path: Path | None = None
    if args.resume:
        resume_path = resolve_transcript_path(workspace_root, args.resume)
        restored_payload = load_transcript_payload(resume_path)
    elif args.continue_session:
        latest_session = latest_restorable_session(workspace_root)
        if latest_session is None:
            raise ValueError(f"No saved sessions found in {workspace_root.as_posix()}/.ollama-code/sessions")
        resume_path, restored_payload = latest_session

    explicit_model = _non_empty_string(args.model)
    session_model: str | None = None
    model = config.model or DEFAULT_MODEL
    model_source = f"config:{config.path.as_posix()}" if config.model and config.path is not None else "built-in default"
    env_model = _non_empty_string(os.environ.get(ENV_OLLAMA_CODE_MODEL))
    explicit_model_from_env = env_model is not None
    if env_model:
        model = env_model
        model_source = ENV_OLLAMA_CODE_MODEL
    approval = config.approval or DEFAULT_APPROVAL_MODE
    if restored_payload is not None:
        saved_model = restored_payload.get("model")
        saved_approval = restored_payload.get("approval_mode")
        if explicit_model is None and isinstance(saved_model, str) and saved_model.strip():
            session_model = saved_model.strip()
            model = session_model
            model_source = f"session:{resume_path.as_posix() if resume_path is not None else '(unknown)'}"
        if args.approval is None and saved_approval in {"ask", "auto", "read-only"}:
            approval = str(saved_approval)
    if explicit_model:
        model = explicit_model
        model_source = "--model"

    # If model selection is not explicitly controlled by CLI/env/session, allow startup to
    # repair stale persisted config values by resolving to installed defaults.
    explicit_runtime_model = explicit_model is not None or explicit_model_from_env or session_model is not None
    allow_model_fallback = not explicit_runtime_model
    verifier_model = config.verifier_model
    env_verifier_model = _non_empty_string(os.environ.get(ENV_OLLAMA_CODE_VERIFIER_MODEL))
    if env_verifier_model:
        verifier_model = env_verifier_model
    explicit_verifier_model = _non_empty_string(args.verifier_model)
    if restored_payload is not None:
        saved_verifier_model = restored_payload.get("verifier_model")
        if explicit_verifier_model is None and env_verifier_model is None and isinstance(saved_verifier_model, str) and saved_verifier_model.strip():
            verifier_model = saved_verifier_model.strip()
    if explicit_verifier_model:
        verifier_model = explicit_verifier_model
    host = config.host
    env_host = _non_empty_string(os.environ.get(ENV_OLLAMA_HOST))
    if env_host:
        host = env_host
    explicit_host = _non_empty_string(args.host)
    if explicit_host:
        host = explicit_host
    if args.approval is not None:
        approval = args.approval
    debate_enabled = config.debate if config.debate is not None else DEFAULT_DEBATE_ENABLED
    env_debate = _bool_from_text(_non_empty_string(os.environ.get(ENV_OLLAMA_CODE_DEBATE)))
    if env_debate is not None:
        debate_enabled = env_debate
    if args.debate is not None:
        debate_enabled = args.debate == "on"
    reconcile_mode = config.reconcile or DEFAULT_RECONCILE_MODE
    env_reconcile = _reconcile_from_text(_non_empty_string(os.environ.get(ENV_OLLAMA_CODE_RECONCILE)))
    if env_reconcile is not None:
        reconcile_mode = env_reconcile
    explicit_reconcile = _reconcile_from_text(args.reconcile)
    if restored_payload is not None:
        saved_reconcile = restored_payload.get("reconcile_mode")
        if explicit_reconcile is None and env_reconcile is None and isinstance(saved_reconcile, str):
            reconcile_mode = _reconcile_from_text(saved_reconcile) or reconcile_mode
    if explicit_reconcile is not None:
        reconcile_mode = explicit_reconcile
    max_tool_rounds = explicit_max_tool_rounds if explicit_max_tool_rounds is not None else (config.max_tool_rounds or DEFAULT_MAX_TOOL_ROUNDS)
    max_agent_depth = explicit_max_agent_depth if explicit_max_agent_depth is not None else (config.max_agent_depth or DEFAULT_MAX_AGENT_DEPTH)
    timeout = explicit_timeout if explicit_timeout is not None else (config.timeout or DEFAULT_TIMEOUT)
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
    resolved_status_printer = status_printer or ((lambda message: None) if args.quiet else (lambda message: print(f"[status] {message}")))
    indexer = BackgroundIndexer(
        workspace_root,
        enabled=config.indexer_enabled and not args.no_indexer,
        watch=config.indexer_watch,
        poll_interval_ms=config.indexer_poll_interval_ms,
        status_printer=resolved_status_printer,
    )
    disable_spec_guided_repair = bool(
        _bool_from_text(_non_empty_string(os.environ.get(ENV_OLLAMA_CODE_DISABLE_SPEC_GUIDED_REPAIR)))
        or args.disable_spec_guided_repair
    )
    require_llm_for_turn = bool(
        _bool_from_text(_non_empty_string(os.environ.get(ENV_OLLAMA_CODE_REQUIRE_LLM_FOR_TURN)))
        or args.require_llm_for_turn
    )
    tools = ToolExecutor(
        workspace_root,
        approval_mode=approval,
        input_func=input_func,
        test_command=test_command,
        default_tools_enabled=config.tools_default_enabled,
        enabled_tools=config.enabled_tools,
        disabled_tools=config.disabled_tools,
        mcp_servers=config.mcp_servers,
        browser_enabled=config.browser_enabled,
        security_enabled=config.security_enabled,
        indexer=indexer,
    )
    agent = OllamaCodeAgent(
        client=client,
        tools=tools,
        model=model,
        max_tool_rounds=max_tool_rounds,
        session_file=active_session_file,
        status_printer=resolved_status_printer,
        thinking_printer=thinking_printer,
        max_agent_depth=max_agent_depth,
        debate_enabled=debate_enabled,
        verifier_model=verifier_model,
        reconcile_mode=reconcile_mode,
        disable_spec_guided_repair=disable_spec_guided_repair,
        require_llm_for_turn=require_llm_for_turn,
    )
    if restored_payload is not None:
        agent.restore_transcript(restored_payload)
        agent.session_file = resolve_transcript_path(workspace_root, session_file)
    agent.config_path = config.path
    agent.model_source = model_source
    setattr(agent, "_allow_model_fallback", allow_model_fallback)
    return agent


def _env_model_is_explicit() -> bool:
    return _non_empty_string(os.environ.get(ENV_OLLAMA_CODE_MODEL)) is not None


def _should_resolve_runtime_default_model(args: argparse.Namespace) -> bool:
    if _non_empty_string(args.model) is not None:
        return False
    if _env_model_is_explicit():
        return False
    if args.resume or args.continue_session:
        return False
    return True


def _resolve_model_candidate(candidate: str, available: set[str]) -> str | None:
    if candidate in available:
        return candidate
    if not candidate.endswith(":latest"):
        latest = f"{candidate}:latest"
        if latest in available:
            return latest
    return None


def ensure_runtime_default_model(
    agent: OllamaCodeAgent,
    args: argparse.Namespace,
    renderer: CliStatusRenderer,
    *,
    quiet: bool = False,
    allow_model_fallback: bool,
) -> None:
    if not _should_resolve_runtime_default_model(args):
        return
    available_models = agent.list_models()
    if not available_models:
        if not quiet:
            renderer.status(f"no Ollama models found. {DEFAULT_MODEL_PULL_HINT}")
        return
    available = set(available_models)
    current = _resolve_model_candidate(agent.model, available)
    if current is not None:
        if current != agent.model:
            agent.set_model(current)
            agent.model_source = f"{getattr(agent, 'model_source', 'built-in default')} -> installed tag"
        return
    if not allow_model_fallback:
        if not quiet:
            renderer.status(f"configured model {agent.model} is not installed locally.")
        return
    for candidate in PREFERRED_FALLBACK_MODELS:
        resolved = _resolve_model_candidate(candidate, available)
        if resolved is None:
            continue
        agent.set_model(resolved)
        agent.model_source = f"runtime fallback:{resolved}"
        if not quiet:
            renderer.status(f"default model {DEFAULT_MODEL} is not installed; using {resolved}. {DEFAULT_MODEL_PULL_HINT}")
        return
    if not quiet:
        renderer.status(
            f"default model {DEFAULT_MODEL} is not installed and no preferred fallback model is available. "
            f"{DEFAULT_MODEL_PULL_HINT} or pass --model explicitly."
        )


def startup_help_text(agent: OllamaCodeAgent) -> str:
    index_status = agent.index_status()
    index_label = "on" if index_status.get("enabled") else "off"
    lines = [
        "Ollama Code",
        f"workspace: {agent.workspace_root().as_posix()}",
        f"model: {agent.model} | approval: {agent.approval_mode()} | debate: {'on' if agent.debate_mode() else 'off'} | reconcile: {agent.reconcile_mode()} | indexer: {index_label}",
    ]
    if agent.configured_test_command():
        lines.append(f"test_cmd: {agent.configured_test_command()}")
    if agent.session_path() is not None:
        lines.append(f"session: {agent.session_path().as_posix()}")
    lines.extend(
        [
            "",
            "Type a coding request, for example:",
            "  fix the failing tests, inspect first, then run tests",
            "  summarize the changes in this repo",
            "",
            "Useful commands:",
            "  /status                         show current model/workspace/session",
            "  /model <name>                    switch local Ollama model",
            "  /approval ask|auto|read-only     control writes and shell commands",
            "  /test [command]                  run configured or explicit tests",
            "  /doctor                          check first-use setup",
            "  /index status|refresh|stop|start  manage the repo-local search indexer",
            "  /todos                           show current todo list",
            "  /tools                           show compact model-facing tools",
            "  /tools groups                    show grouped model-facing tool families",
            "  /tools missing                   show optional OSS integrations not installed",
            "  /tools install <tool-id>          prompt to install one optional integration",
            "  /help                            show all slash commands",
            "  /quit                            exit",
            "",
            "Repo search, validator discovery, todos, and verified-function cards are enabled by default.",
            "Press Esc to interrupt the current model or tool call.",
        ]
    )
    return "\n".join(lines)


def slash_help_text() -> str:
    return "\n".join(
        [
            "Slash commands:",
            "  /help                            show this help",
            "  /status                          show workspace/model/session/config",
            "  /models                          list local Ollama models",
            "  /model <name>                    switch model",
            "  /approval ask|auto|read-only     control writes and shell commands",
            "  /debate on|off                   toggle tool audits and final verification",
            "  /reconcile off|on|auto           toggle artifact reconciliation",
            "  /doctor                          check Ollama/model/workspace setup",
            "  /index status|refresh|stop|start  manage the repo-local search indexer",
            "  /reset                           clear conversation memory",
            "  /todos [clear]                   show or clear current todo list",
            "  /save [path]                     save transcript",
            "  /sessions [limit]                list saved sessions",
            "  /load <path>                     load a saved session",
            "  /git                             show git status",
            "  /diff [--cached] [path]          show git diff",
            "  /commit <message>                commit via git_commit tool",
            "  /test [command]                  run tests",
            "  /tools [full|groups]             show compact tools, grouped tools, or full descriptions",
            "  /tools missing                   show optional OSS integrations not installed",
            "  /tools install <tool-id>|--recommended",
            "  /quit                            exit",
            "",
            "Tips:",
            "  Ask normally: \"fix failing tests, inspect source first, then run tests\".",
            "  Run /doctor if the first request is slow or fails before tool use.",
            "  The background indexer keeps file/repo/FTS and verified-function search caches hot under .ollama-code/index.",
            "  Use /tools for a compact list; /tools full is verbose.",
            "  Use /approval read-only for inspection-only sessions.",
            "  Press Esc during model or tool execution to interrupt.",
        ]
    )


def print_banner(agent: OllamaCodeAgent) -> None:
    print(startup_help_text(agent))


def _strip_matching_quotes(text: str) -> str:
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        return stripped[1:-1]
    return stripped


def doctor_report(agent: OllamaCodeAgent) -> tuple[str, bool]:
    ok = True
    lines = [
        "Ollama Code doctor",
        f"workspace: ok {agent.workspace_root().as_posix()}",
    ]
    session = agent.session_path()
    lines.append(f"session: ok {session.as_posix() if session is not None else '(none)'}")
    config_path = getattr(agent, "config_path", None)
    lines.append(f"config: {'ok ' + config_path.as_posix() if isinstance(config_path, Path) else '(none)'}")
    lines.append(f"model_source: {getattr(agent, 'model_source', '(unknown)')}")
    if agent.configured_test_command():
        lines.append(f"test_cmd: ok {agent.configured_test_command()}")
    else:
        lines.append("test_cmd: not configured; use --test-cmd or let discover_validators choose focused commands")

    try:
        models = agent.list_models()
    except OllamaError as exc:
        ok = False
        lines.append(f"ollama: error {exc}")
        models = []
    else:
        if models:
            lines.append(f"ollama: ok {len(models)} local model(s)")
        else:
            ok = False
            lines.append(f"ollama: no local models. {DEFAULT_MODEL_PULL_HINT}")

    if models:
        resolved = _resolve_model_candidate(agent.model, set(models))
        if resolved is not None:
            lines.append(f"model: ok {resolved}")
        else:
            ok = False
            lines.append(f"model: missing {agent.model}. {DEFAULT_MODEL_PULL_HINT}")

    index_status = agent.index_status()
    if index_status.get("enabled"):
        state = "ready" if index_status.get("ready") else ("starting" if index_status.get("running") else "idle")
        lines.append(f"indexer: ok enabled ({state}); cache={index_status.get('cache_dir')}")
    else:
        lines.append("indexer: disabled")

    available_tools: set[str] | None = None
    tools = getattr(agent, "tools", None)
    available = getattr(tools, "available_tool_names", None)
    if callable(available):
        try:
            available_tools = {str(name) for name in available()}
        except Exception:
            available_tools = None
    if available_tools is not None:
        verified_tools = {
            "verified_function_index",
            "verified_function_search",
            "verified_function_show",
            "verify_function_contract",
            "compose_verified_functions",
            "promote_verified_function",
        }
        missing_verified = sorted(verified_tools - available_tools)
        if missing_verified:
            lines.append("verified functions: disabled/missing " + ", ".join(missing_verified))
        else:
            lines.append("verified functions: ok default-on Python cards; cache=.ollama-code/index/verified_functions.sqlite")
        sdk_tools = {"python_sdk_search", "python_sdk_refresh"}
        missing_sdk = sorted(sdk_tools - available_tools)
        if missing_sdk:
            lines.append("python sdk index: disabled/missing " + ", ".join(missing_sdk))
        else:
            lines.append("python sdk index: ok on-demand stdlib/API search; cache=.ollama-code/index/python_sdk.sqlite")
            sdk_embed_model = os.environ.get("OLLAMA_CODE_SDK_EMBED_MODEL", "").strip()
            if sdk_embed_model and sdk_embed_model.lower() not in {"0", "false", "none", "off"}:
                lines.append(f"python sdk embeddings: ok on-demand candidate rerank via {sdk_embed_model}")
            else:
                lines.append("python sdk embeddings: disabled; set OLLAMA_CODE_SDK_EMBED_MODEL to enable")

    tool_rows = dependency_statuses(workspace_root=agent.workspace_root())
    installed_tools = [str(row["id"]) for row in tool_rows if row.get("installed")]
    missing_recommended = [
        str(row["id"])
        for row in tool_rows
        if row.get("recommended") and row.get("supported") and not row.get("installed")
    ]
    unsupported_recommended = [
        str(row["id"])
        for row in tool_rows
        if row.get("recommended") and not row.get("supported")
    ]
    lines.append(f"optional tools: installed {len(installed_tools)}/{len(tool_rows)}")
    docker_host = (
        os.environ.get("OLLAMA_CODE_DOCKER_HOST")
        or os.environ.get("OLLAMA_CODE_REMOTE_DOCKER_HOST")
        or os.environ.get("DOCKER_HOST")
    )
    if docker_host:
        normalized_docker_host = docker_host if "://" in docker_host else f"ssh://{docker_host}"
        lines.append(f"docker tools: remote host {normalized_docker_host}")
    elif any(row.get("id") == "docker" and row.get("installed") for row in tool_rows):
        lines.append("docker tools: local client detected; set OLLAMA_CODE_DOCKER_HOST=ssh://host for remote container tools")
    if missing_recommended:
        lines.append("optional tools: recommended missing " + ", ".join(missing_recommended))
        lines.append("optional tools: install interactively with /tools install --recommended or inspect with /tools missing")
    if unsupported_recommended:
        lines.append("optional tools: unsupported here " + ", ".join(unsupported_recommended))
    if shutil.which("ollama-code"):
        lines.append("console script: ok ollama-code")
    else:
        lines.append("console script: not on PATH; run python -m ollama_code or python -m pip install -e .")
    return "\n".join(lines), ok


def _format_index_status(status: dict[str, object]) -> str:
    if status.get("ok") is False and not status.get("enabled"):
        return str(status.get("summary", "indexer disabled"))
    running = "yes" if status.get("running") else "no"
    ready = "yes" if status.get("ready") else "no"
    enabled = "yes" if status.get("enabled") else "no"
    pending = status.get("pending_paths", 0)
    refresh_count = status.get("refresh_count", 0)
    summary = status.get("summary", "")
    return f"indexer enabled={enabled} running={running} ready={ready} pending={pending} refreshes={refresh_count} summary={summary}"


def handle_meta_command(command: str, agent: OllamaCodeAgent, writer: Callable[[str], None]) -> bool | None:
    if not command.startswith("/"):
        return None
    head, _, tail = command.strip().partition(" ")
    action = head.lower()
    remainder = tail.strip()
    if action in {"/quit", "/exit"}:
        return False
    if action == "/help":
        writer(slash_help_text())
        return True
    if action == "/status":
        session = agent.session_path().as_posix() if agent.session_path() is not None else "(none)"
        test_command = agent.configured_test_command() or "(none)"
        config_path = getattr(agent, "config_path", None)
        config_label = config_path.as_posix() if isinstance(config_path, Path) else "(none)"
        model_source = getattr(agent, "model_source", "(unknown)")
        writer(
            f"workspace={agent.workspace_root().as_posix()} model={agent.model} model_source={model_source} verifier_model={agent.verifier_model_name() or '-'} approval={agent.approval_mode()} debate={'on' if agent.debate_mode() else 'off'} reconcile={agent.reconcile_mode()} max_tool_rounds={agent.max_tool_rounds} max_agent_depth={agent.max_agent_depth} test_cmd={test_command} config={config_label} session={session}"
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
    if action == "/debate":
        mode = _strip_matching_quotes(remainder)
        if mode not in {"on", "off"}:
            writer("Usage: /debate on|off")
            return True
        agent.set_debate_enabled(mode == "on")
        writer(f"debate set to {'on' if agent.debate_mode() else 'off'}")
        return True
    if action == "/reconcile":
        mode = _strip_matching_quotes(remainder)
        if mode not in {"off", "on", "auto"}:
            writer("Usage: /reconcile off|on|auto")
            return True
        agent.set_reconcile_mode(mode)
        writer(f"reconcile set to {agent.reconcile_mode()}")
        return True
    if action == "/doctor":
        report, _ = doctor_report(agent)
        writer(report)
        return True
    if action == "/index":
        command = _strip_matching_quotes(remainder).lower() or "status"
        if command == "status":
            writer(_format_index_status(agent.index_status()))
            return True
        if command == "refresh":
            result = agent.refresh_index()
            writer(str(result.get("summary", "index refresh queued")))
            return True
        if command == "stop":
            agent.stop_indexer()
            writer("indexer stopped")
            return True
        if command == "start":
            started = agent.start_indexer()
            writer("indexer started" if started else "indexer disabled")
            return True
        writer("Usage: /index status|refresh|stop|start")
        return True
    if action == "/todos":
        command = _strip_matching_quotes(remainder).lower()
        if command == "clear":
            result = agent.todo_clear()
            writer(str(result.get("output") or result.get("summary", "(no todos)")))
            return True
        if command:
            writer("Usage: /todos [clear]")
            return True
        result = agent.todo_read()
        writer(str(result.get("output") or result.get("summary", "(no todos)")))
        return True
    if action == "/reset":
        agent.reset()
        writer("conversation reset")
        return True
    if action == "/save":
        try:
            target = _parse_single_meta_path(remainder)
        except ValueError:
            writer("Usage: /save [path]")
            return True
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
        try:
            target = _parse_single_meta_path(remainder)
        except ValueError:
            writer("Usage: /load <path>")
            return True
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
            diff_parts = _split_meta_args(remainder) if remainder else []
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
        command_text = _strip_matching_quotes(remainder) or None
        result = agent.run_test(command_text)
        output = str(result.get("output") or result.get("summary", "(no output)"))
        writer(output)
        return True
    if action == "/tools":
        try:
            args = _split_meta_args(remainder) if remainder else []
        except ValueError:
            writer("Usage: /tools [full|groups|missing|recommended|all|install <tool-id>|install --recommended]")
            return True
        if not args:
            writer(agent.tool_help(compact=True))
            return True
        subcommand = _strip_matching_quotes(args[0]).lower()
        if subcommand == "full":
            writer(agent.tool_help(compact=False))
            return True
        if subcommand == "groups":
            group_method = getattr(agent, "tool_group_help", None)
            writer(group_method() if callable(group_method) else agent.tool_help(compact=True))
            return True
        status_method = getattr(agent, "tool_dependency_status", None)
        install_method = getattr(agent, "tool_dependency_install", None)
        if subcommand in {"missing", "recommended", "all"}:
            if not callable(status_method):
                writer("optional tool dependency status is not available for this agent")
                return True
            scope = "missing" if subcommand == "missing" else subcommand
            result = status_method(scope=scope)
            writer(str(result.get("output") or result.get("summary", "(no output)")))
            return True
        if subcommand == "install":
            if not callable(install_method):
                writer("optional tool install is not available for this agent")
                return True
            if len(args) != 2:
                writer("Usage: /tools install <tool-id>|--recommended")
                return True
            target = _strip_matching_quotes(args[1])
            if not target:
                writer("Usage: /tools install <tool-id>|--recommended")
                return True
            result = install_method(None if target == "--recommended" else target, all_recommended=target == "--recommended", confirm=True)
            writer(str(result.get("output") or result.get("summary", "(no output)")))
            return True
        writer("Usage: /tools [full|groups|missing|recommended|all|install <tool-id>|install --recommended]")
        return True
    writer(f"Unknown command: {command}")
    return True


def run_repl(agent: OllamaCodeAgent, *, quiet: bool = False, renderer: CliStatusRenderer | None = None) -> int:
    renderer = renderer or CliStatusRenderer()
    agent.start_indexer()
    if not quiet:
        print_banner(agent)
    interrupt_controller = InterruptController()
    try:
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
                    llm_calls_before = _llm_call_count(agent)
                    result = agent.handle_user(raw)
                    if agent.require_llm_for_turn and _llm_call_count(agent) == llm_calls_before:
                        renderer.clear_thinking()
                        print("error: require-llm-for-turn completed without any llm_call event", file=sys.stderr)
                        return 1
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
    finally:
        agent.stop_indexer()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    renderer = CliStatusRenderer()
    status_printer = (lambda _message: None) if args.quiet else renderer.status
    try:
        agent = build_agent(args, status_printer=status_printer, thinking_printer=renderer.show_thinking)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not args.doctor:
        try:
            ensure_runtime_default_model(
                agent,
                args,
                renderer,
                quiet=args.quiet,
                allow_model_fallback=getattr(agent, "_allow_model_fallback", True),
            )
        except OllamaError as exc:
            renderer.clear_thinking()
            print(f"error: {exc}", file=sys.stderr)
            return 1
    if args.doctor:
        report, ok = doctor_report(agent)
        print(report)
        return 0 if ok else 1
    if args.prompt:
        agent.start_indexer()
        try:
            with InterruptController(lambda _: None).watch() as interrupt_event:
                agent.set_interrupt_event(interrupt_event)
                llm_calls_before = _llm_call_count(agent)
                result = agent.handle_user(" ".join(args.prompt))
                if agent.require_llm_for_turn and _llm_call_count(agent) == llm_calls_before:
                    renderer.clear_thinking()
                    print("error: require-llm-for-turn completed without any llm_call event", file=sys.stderr)
                    return 1
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
            agent.stop_indexer()
        renderer.write(result.message)
        return 0
    return run_repl(agent, quiet=args.quiet, renderer=renderer)


if __name__ == "__main__":
    raise SystemExit(main())
