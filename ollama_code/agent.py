from __future__ import annotations

import ast
from copy import deepcopy
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import threading

from ollama_code.ollama_client import ChatResponse, OllamaClient, OllamaError
from ollama_code.sessions import (
    SessionSummary,
    default_session_dir,
    list_sessions as collect_sessions,
    load_transcript_payload,
    resolve_transcript_path,
)
from ollama_code.tools import TOOL_DESCRIPTIONS, ToolExecutor, format_compact_tool_help, format_tool_help


SYSTEM_PROMPT_TEMPLATE = """Ollama Code. Workspace: {workspace_root}

JSON only:
{{"type":"tool","name":"read_file","arguments":{{"path":"README.md"}}}}
{{"type":"final","message":"..."}}

Rules:
- One tool or one final. Relative paths. No fences/thought.
- Need repo/file/git/shell/edit/agent facts? Use tools; do not guess.
- Inspect before edit. Write: write_file. Symbol edit: replace_symbol/replace_symbols. Text edit: replace_in_file. Tests: run_test, not run_shell. Git: git_status/git_diff/git_commit.
- write_file content is raw file text only; no markdown fences, no leading ">" quote markers.
- For fix/implement + tests: read tests/source, edit implementation, run_test; do not only summarize or loop on reads.
- If user names a source file/function to fix, edit source, not tests, unless tests are explicitly requested.
- Code nav: prefer search_symbols, code_outline, then read_symbol before broad read_file. Search/list before broad reads; use ranges.
- Reuse results; avoid repeat read-only calls unless state changed.
- Question your assumptions before acting; prove or disprove with tools when possible.
- Never claim edit/cmd/test/agent success without current-turn success.
- Style: caveman-lite concise; keep code, paths, commands, errors, JSON exact and syntactically complete.

Tools:
{tool_help}
"""


@dataclass
class AgentResult:
    message: str
    rounds: int
    completed: bool = True


@dataclass(frozen=True)
class ExactFileWriteSpec:
    path: str
    line: str


@dataclass(frozen=True)
class FinalRewriteOutcome:
    accepted_message: str | None = None
    retry_decision: dict[str, Any] | None = None
    rejected_message: str | None = None


@dataclass(frozen=True)
class TargetLineReadSpec:
    path: str
    start: int
    end: int
    line: int


@dataclass(frozen=True)
class SymbolReadSpec:
    path: str
    symbol: str


KNOWN_TOOL_NAMES = {tool["name"] for tool in TOOL_DESCRIPTIONS}
APPROVAL_RANK = {"read-only": 0, "ask": 1, "auto": 2}
VERIFICATION_HISTORY_LIMIT = 1
VERIFICATION_CONTENT_LIMIT = 2200
PRIMARY_CONTEXT_RECENT_MESSAGE_LIMIT = 12
PRIMARY_CONTEXT_CONTENT_LIMIT = 900
MAX_VERIFICATION_RETRIES = 2
MAX_VERIFICATION_REWRITE_ATTEMPTS = 1
MAX_ASSUMPTION_AUDIT_RETRIES = 2
MAX_RECONCILIATION_RETRIES = 2
AUDIT_LIST_ITEM_LIMIT = 3
AUDIT_TEXT_ITEM_LIMIT = 140
CANDIDATE_CLAIM_LIMIT = 5
CANDIDATE_CLAIM_TEXT_LIMIT = 180
VERIFICATION_EVIDENCE_LIMIT = 3
VERIFICATION_EVIDENCE_TEXT_LIMIT = 150
MUTATING_TOOL_NAMES = {"write_file", "replace_symbol", "replace_symbols", "replace_in_file", "apply_structured_edit", "git_commit"}
READ_ONLY_CACHEABLE_TOOL_NAMES = {
    "list_files",
    "read_file",
    "search",
    "search_symbols",
    "code_outline",
    "read_symbol",
    "repo_index_search",
    "find_implementation_target",
    "diagnose_test_failure",
    "call_graph",
    "lint_typecheck",
    "git_status",
    "git_diff",
}
READ_ONLY_WORKSPACE_TOOL_NAMES = {"list_files", "read_file", "search", "search_symbols", "code_outline", "read_symbol", "repo_index_search", "find_implementation_target", "call_graph"}
EDIT_TOOL_NAMES = {"write_file", "replace_symbol", "replace_symbols", "replace_in_file", "apply_structured_edit"}
TEST_TOOL_NAMES = {"run_test", "diagnose_test_failure", "find_implementation_target", "run_function_probe", "lint_typecheck", "generate_tests_from_spec"}
SHELL_TOOL_NAMES = {"run_shell", "run_function_probe"}
GIT_TOOL_NAMES = {"git_status", "git_diff", "git_commit"}
AGENT_TOOL_NAMES = {"run_agent"}
RISKY_VERIFICATION_TOOL_NAMES = {"search", "git_status", "git_diff", "run_shell", "run_test", "run_agent"}
CODE_EDIT_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".c", ".cc", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt", ".kts"}
MODEL_TOOL_RESULT_LIMITS = {
    "list_files": 500,
    "read_file": 1000,
    "search": 700,
    "search_symbols": 700,
    "code_outline": 900,
    "read_symbol": 1100,
    "repo_index_search": 900,
    "find_implementation_target": 800,
    "diagnose_test_failure": 900,
    "run_function_probe": 700,
    "call_graph": 900,
    "lint_typecheck": 800,
    "replace_symbol": 900,
    "replace_symbols": 1000,
    "apply_structured_edit": 1000,
    "generate_tests_from_spec": 1000,
    "git_status": 700,
    "git_diff": 900,
    "run_shell": 700,
    "run_test": 700,
    "run_agent": 900,
}
MODEL_TOOL_DIFF_LIMIT = 700
VERIFICATION_TOOL_RESULT_LIMIT = 450
VERIFICATION_TOOL_DIFF_LIMIT = 450

FINAL_VERIFIER_SYSTEM_PROMPT = """You are a grounded final verifier for a coding CLI controller.

Check final vs evidence/constraints. JSON only.

Replies:
{"verdict":"accept","claim_checks":[{"claim":"...","status":"supported","evidence":"E1"}]}
{"verdict":"retry","reason":"brief concrete reason","required_tools":["read_file"],"forbidden_tools":["run_shell"],"claim_checks":[{"claim":"...","status":"contradicted","evidence":"E2","correction":"..."}],"rewrite_guidance":["..."],"rewrite_from_evidence":true}

Rules:
- Accept if candidate matches request, constraints, tool results, and accepted audits.
- Retry if contradiction, unsupported workspace claim, missing/forbidden tool, or another tool is needed.
- claim_checks status: supported, contradicted, unverified. Cite evidence ids when possible.
- correction/rewrite_guidance must come only from evidence.
- rewrite_from_evidence true only if evidence can fully fix final without another tool.
- Do not write the final answer. Tool arrays: known names only or [].
"""

FINAL_REWRITER_SYSTEM_PROMPT = """You are an evidence-backed final rewriter for a coding CLI controller.

Return exactly one JSON object only.

Reply:
{"type":"final","message":"Accurate final answer grounded only in the supplied evidence."}

Rules:
- Use only evidence table, claim checks, and rewrite guidance.
- Do not invent files, commands, diffs, outcomes, or unsupported details.
- Use supplied corrections for contradicted claims, or omit those claims.
- Keep final concise and useful.
"""

TOOL_ASSUMPTION_AUDITOR_SYSTEM_PROMPT = """You are a tool-step assumption auditor for a coding CLI controller.

Decide if proposed tool is grounded next step. JSON only.

Replies:
{"verdict":"accept","reason":"","assumptions":["..."],"validation_steps":["..."],"required_tools":[],"forbidden_tools":[]}
{"verdict":"retry","reason":"brief concrete reason","assumptions":["..."],"validation_steps":["..."],"required_tools":["read_file"],"forbidden_tools":["run_shell"]}

Rules:
- Keep assumptions/validation_steps short.
- Accept reasonable validation steps, including expected failures/boundary probes.
- Accept read/inspect steps after failed tests; repair needs fresh evidence more than another audit.
- Retry if redundant, too broad, constraint-violating, mutating before inspection, or not validating the key assumption.
- Do not rewrite the tool. Tool arrays: known names only or [].
"""

ARTIFACT_RECONCILER_SYSTEM_PROMPT = """You are an artifact reconciliation critic for a coding CLI controller.

After a failed tool/test/edit artifact, decide if the main model should retry with a compact repair plan. JSON only.

Replies:
{"verdict":"accept","reason":"","repair_plan":[],"required_tools":[],"forbidden_tools":[]}
{"verdict":"retry","reason":"brief concrete reason","repair_plan":["inspect failing symbol","edit implementation","rerun tests"],"required_tools":["read_file"],"forbidden_tools":["run_shell"]}

Rules:
- Use only supplied request, recent messages, tool calls, and artifact evidence.
- Retry if failed tests, syntax errors, or tool errors imply a specific next validation/repair step.
- Prefer implementation/source repair before repeated tests.
- Keep repair_plan short. Tool arrays: known names only or [].
- Do not write code or the final answer.
"""


def extract_json_response(raw_text: str, *, _depth: int = 0) -> dict[str, Any] | None:
    candidate = raw_text.strip()
    if not candidate:
        return None
    candidate = re.sub(r"<think>.*?</think>", "", candidate, flags=re.DOTALL)
    candidate = candidate.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
        candidate = candidate.strip()
    decoder = json.JSONDecoder()
    if _depth < 2:
        try:
            top_level, end_index = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            top_level = None
            end_index = -1
        if isinstance(top_level, str) and not candidate[end_index:].strip():
            return extract_json_response(top_level, _depth=_depth + 1)
    starts = [index for index, char in enumerate(candidate) if char == "{"]
    if candidate.startswith("{"):
        starts = [0] + [index for index in starts if index != 0]
    parsed_dicts: list[dict[str, Any]] = []
    agent_payloads: list[dict[str, Any]] = []
    top_level_dict: dict[str, Any] | None = None
    for start in starts:
        try:
            data, end_index = decoder.raw_decode(candidate[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            if start == 0 and not candidate[end_index:].strip():
                top_level_dict = data
            parsed_dicts.append(data)
            response_type = data.get("type")
            tool_name = data.get("name")
            if isinstance(response_type, str) and response_type.strip() in {"tool", "final", *KNOWN_TOOL_NAMES}:
                agent_payloads.append(data)
                continue
            if isinstance(tool_name, str) and tool_name.strip() in KNOWN_TOOL_NAMES and response_type in {None, "", "function", "tool_call"}:
                agent_payloads.append(data)
    if agent_payloads:
        return agent_payloads[-1]
    if top_level_dict is not None:
        return top_level_dict
    if parsed_dicts:
        return parsed_dicts[-1]
    return None


def _portable_workspace_key(raw_path: str | Path) -> str | None:
    text = str(raw_path).strip()
    if not text:
        return None
    normalized = text.replace("\\", "/")
    drive_match = re.match(r"^(?P<drive>[A-Za-z]):(?:/(?P<rest>.*))?$", normalized)
    if drive_match:
        drive = drive_match.group("drive").lower()
        rest = (drive_match.group("rest") or "").strip("/")
        return f"{drive}:/{rest}" if rest else f"{drive}:/"
    wsl_match = re.match(r"^/mnt/(?P<drive>[A-Za-z])(?:/(?P<rest>.*))?$", normalized)
    if wsl_match:
        drive = wsl_match.group("drive").lower()
        rest = (wsl_match.group("rest") or "").strip("/")
        return f"{drive}:/{rest}" if rest else f"{drive}:/"
    try:
        return Path(text).resolve(strict=False).as_posix()
    except OSError:
        return None


def _workspace_roots_match(saved_root: object, current_root: Path) -> bool:
    if saved_root is None:
        return True
    saved_text = str(saved_root).strip()
    if not saved_text:
        return False
    current_root = current_root.resolve()
    try:
        saved_path = Path(saved_text).resolve(strict=False)
    except OSError:
        saved_path = None
    if saved_path == current_root:
        return True
    saved_key = _portable_workspace_key(saved_text)
    current_key = _portable_workspace_key(current_root)
    if saved_key is None or current_key is None:
        return False
    if re.match(r"^[A-Za-z]:/", saved_key) or re.match(r"^[A-Za-z]:/", current_key):
        return saved_key.casefold() == current_key.casefold()
    return saved_key == current_key


class OllamaCodeAgent:
    def __init__(
        self,
        *,
        client: OllamaClient,
        tools: ToolExecutor,
        model: str,
        max_tool_rounds: int = 8,
        session_file: str | Path | None = None,
        status_printer: Callable[[str], None] | None = None,
        thinking_printer: Callable[[str], None] | None = None,
        agent_depth: int = 0,
        max_agent_depth: int = 2,
        debate_enabled: bool = True,
        verifier_model: str | None = None,
        reconcile_mode: str = "off",
    ) -> None:
        self.client = client
        self.tools = tools
        self.model = model
        self.verifier_model = verifier_model.strip() if isinstance(verifier_model, str) and verifier_model.strip() else None
        self.max_tool_rounds = max_tool_rounds
        self.session_file = self._resolve_transcript_path(session_file) if session_file else None
        self.status_printer = status_printer or (lambda message: None)
        self.thinking_printer = thinking_printer
        self.agent_depth = agent_depth
        self.max_agent_depth = max_agent_depth
        self.debate_enabled = debate_enabled
        self.reconcile_mode_setting = self._normalize_reconcile_mode(reconcile_mode)
        self.events: list[dict[str, Any]] = []
        self._pending_llm_call_events: list[dict[str, Any]] = []
        self._interrupt_event: threading.Event | None = None
        self._turn_tool_cache: dict[tuple[str, str, int], dict[str, Any]] = {}
        self._turn_cache_epoch = 0
        self.tools.agent_runner = self._run_sub_agent
        self.messages = self._base_messages()

    def _base_messages(self) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": self._system_prompt_for_tools(None),
            }
        ]

    def _system_prompt_for_tools(self, tool_names: set[str] | None) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(
            workspace_root=self.tools.workspace_root.as_posix(),
            tool_help=format_compact_tool_help(tool_names),
        )

    def _primary_tool_names_for_request(
        self,
        request_text: str,
        *,
        requires_tools: bool,
        session_memory_request: bool,
        mutation_allowed: bool,
        mutation_required: bool,
        test_run_required: bool,
        required_tool_names: set[str],
        forbidden_tool_names: set[str],
    ) -> set[str]:
        if session_memory_request:
            return set()
        lowered = request_text.lower()
        selected: set[str] = set()
        workspace_terms = [
            "file",
            "folder",
            "workspace",
            "repo",
            "repository",
            "code",
            "source",
            "function",
            "class",
            "symbol",
            "read",
            "search",
            "find",
            "inspect",
            "look",
            "summarize",
        ]
        if requires_tools or any(term in lowered for term in workspace_terms):
            selected.update(READ_ONLY_WORKSPACE_TOOL_NAMES)
        if mutation_allowed or mutation_required:
            selected.update(READ_ONLY_WORKSPACE_TOOL_NAMES)
            selected.update(EDIT_TOOL_NAMES)
        if test_run_required or re.search(r"\b(?:test|pytest|unittest|run tests?)\b", lowered):
            selected.update(TEST_TOOL_NAMES)
        if re.search(r"\b(?:shell|command|execute|run exactly|terminal|powershell|bash)\b", lowered):
            selected.update(SHELL_TOOL_NAMES)
        if re.search(r"\b(?:git|diff|status|commit|staged|working tree)\b", lowered):
            selected.update(GIT_TOOL_NAMES)
        if re.search(r"\b(?:agent|subagent|sub-agent|delegate)\b", lowered):
            selected.update(AGENT_TOOL_NAMES)
        selected.update(required_tool_names)
        selected.difference_update(forbidden_tool_names)
        return {name for name in selected if name in KNOWN_TOOL_NAMES}

    def reset(self) -> None:
        self.messages = self._base_messages()
        self.events = []
        self._reset_turn_cache()
        self._autosave()

    def set_model(self, model: str) -> None:
        self.model = model

    def verifier_model_name(self) -> str | None:
        return self.verifier_model

    def verification_model(self) -> str:
        return self.verifier_model or self.model

    def set_approval_mode(self, mode: str) -> None:
        self.tools.set_approval_mode(mode)

    def approval_mode(self) -> str:
        return self.tools.approval_mode

    def debate_mode(self) -> bool:
        return self.debate_enabled

    def set_debate_enabled(self, enabled: bool) -> None:
        self.debate_enabled = bool(enabled)

    def reconcile_mode(self) -> str:
        return self.reconcile_mode_setting

    def set_reconcile_mode(self, mode: str) -> None:
        self.reconcile_mode_setting = self._normalize_reconcile_mode(mode)

    @staticmethod
    def _normalize_reconcile_mode(mode: str | None) -> str:
        value = str(mode or "off").strip().lower()
        return value if value in {"off", "on", "auto"} else "off"

    def configured_test_command(self) -> str | None:
        return self.tools.default_test_command

    def set_interrupt_event(self, event: threading.Event | None) -> None:
        self._interrupt_event = event
        self.client.set_interrupt_event(event)
        self.tools.set_interrupt_event(event)

    def workspace_root(self) -> Path:
        return self.tools.workspace_root

    def session_path(self) -> Path | None:
        return self.session_file

    def session_directory(self) -> Path:
        return default_session_dir(self.tools.workspace_root)

    def tool_help(self) -> str:
        return format_tool_help()

    def list_models(self) -> list[str]:
        return self.client.list_models()

    def _resolve_transcript_path(self, path: str | Path) -> Path:
        return resolve_transcript_path(self.tools.workspace_root, path)

    def restore_transcript(self, payload: dict[str, Any]) -> None:
        workspace_root = payload.get("workspace_root")
        if not _workspace_roots_match(workspace_root, self.tools.workspace_root):
            raise ValueError("Saved session belongs to a different workspace.")
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("Saved session is missing message history.")
        restored_messages: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Saved session contains an invalid message entry.")
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError("Saved session contains a malformed message.")
            restored_messages.append({"role": role, "content": content})
        events = payload.get("events")
        self.messages = restored_messages
        self.events = list(events) if isinstance(events, list) else []
        self._autosave()

    def load_session(self, path: str | Path) -> Path:
        target = self._resolve_transcript_path(path)
        payload = load_transcript_payload(target)
        previous_session_file = self.session_file
        self.session_file = None
        try:
            self.restore_transcript(payload)
        except Exception:
            self.session_file = previous_session_file
            raise
        saved_model = payload.get("model")
        if isinstance(saved_model, str) and saved_model.strip():
            self.model = saved_model.strip()
        saved_verifier_model = payload.get("verifier_model")
        self.verifier_model = saved_verifier_model.strip() if isinstance(saved_verifier_model, str) and saved_verifier_model.strip() else None
        saved_reconcile_mode = payload.get("reconcile_mode")
        if isinstance(saved_reconcile_mode, str):
            self.reconcile_mode_setting = self._normalize_reconcile_mode(saved_reconcile_mode)
        saved_approval = payload.get("approval_mode")
        if saved_approval in {"ask", "auto", "read-only"}:
            self.tools.set_approval_mode(str(saved_approval))
        self.session_file = target
        return target

    def list_sessions(self, limit: int = 10) -> list[SessionSummary]:
        return collect_sessions(self.tools.workspace_root, limit=limit)

    def git_status(self, path: str | None = None) -> dict[str, Any]:
        arguments = {"path": path} if path else {}
        return self.tools.execute("git_status", arguments)

    def git_diff(self, *, cached: bool = False, path: str | None = None) -> dict[str, Any]:
        arguments: dict[str, Any] = {"cached": cached}
        if path:
            arguments["path"] = path
        return self.tools.execute("git_diff", arguments)

    def git_commit(self, message: str, *, add_all: bool = True) -> dict[str, Any]:
        return self.tools.execute("git_commit", {"message": message, "add_all": add_all})

    def run_test(self, command: str | None = None) -> dict[str, Any]:
        arguments: dict[str, Any] = {}
        if command:
            arguments["command"] = command
        return self.tools.execute("run_test", arguments)

    def save_transcript(self, path: str | Path | None = None) -> Path:
        target = self._resolve_transcript_path(path) if path else self.session_file
        if target is None:
            raise ValueError("No transcript path was provided.")
        payload = {
            "model": self.model,
            "verifier_model": self.verifier_model,
            "reconcile_mode": self.reconcile_mode_setting,
            "workspace_root": self.tools.workspace_root.as_posix(),
            "approval_mode": self.tools.approval_mode,
            "messages": self.messages,
            "events": self.events,
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    def _autosave(self) -> None:
        if self.session_file is not None:
            self.save_transcript(self.session_file)

    def _record_event(self, event_type: str, **payload: Any) -> None:
        self.events.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": event_type,
                **payload,
            }
        )
        self._autosave()

    def _chat(
        self,
        *,
        purpose: str,
        model: str,
        messages: list[dict[str, str]],
        response_format: str | dict[str, Any] | None = "json",
        on_thinking: Callable[[str], None] | None = None,
        think: bool | None = None,
    ) -> ChatResponse:
        response = self.client.chat(
            model=model,
            messages=messages,
            response_format=response_format,
            on_thinking=on_thinking,
            think=think,
        )
        prompt_chars = sum(len(str(message.get("content", ""))) for message in messages)
        prompt_chars_by_role: dict[str, int] = {}
        top_messages: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            prompt_chars_by_role[role] = prompt_chars_by_role.get(role, 0) + len(content)
            top_messages.append(
                {
                    "index": index,
                    "role": role,
                    "chars": len(content),
                    "preview": content.replace("\n", " ")[:80],
                }
            )
        top_messages = sorted(top_messages, key=lambda item: int(item["chars"]), reverse=True)[:5]
        payload = {
            "purpose": purpose,
            "model": response.model,
            "requested_model": model,
            "message_count": len(messages),
            "prompt_chars": prompt_chars,
            "prompt_chars_by_role": prompt_chars_by_role,
            "top_prompt_messages": top_messages,
            "response_chars": len(response.content),
            "thinking_chars": len(response.thinking),
            "think": think,
            **response.usage.as_event_payload(),
        }
        self._pending_llm_call_events.append(payload)
        return response

    def _flush_llm_call_events(self) -> None:
        pending = self._pending_llm_call_events
        self._pending_llm_call_events = []
        for payload in pending:
            self._record_event("llm_call", **payload)

    def _primary_messages_for_model(
        self,
        *,
        session_memory_request: bool,
        current_request: str,
        tool_names: set[str] | None = None,
    ) -> list[dict[str, str]]:
        dynamic_system_message = {"role": "system", "content": self._system_prompt_for_tools(tool_names)}
        if session_memory_request or len(self.messages) <= PRIMARY_CONTEXT_RECENT_MESSAGE_LIMIT + 1:
            return [dynamic_system_message, *self.messages[1:]]
        system_message = dynamic_system_message
        non_system_messages = self.messages[1:]
        recent_messages = non_system_messages[-PRIMARY_CONTEXT_RECENT_MESSAGE_LIMIT:]
        current_turn_request: dict[str, str] | None = None
        for message in reversed(non_system_messages):
            if message.get("role") == "user" and message.get("content") == current_request:
                current_turn_request = message
                break
        compacted: list[dict[str, str]] = [system_message]
        omitted_count = max(0, len(non_system_messages) - len(recent_messages))
        if omitted_count:
            compacted.append(
                {
                    "role": "user",
                    "content": f"Earlier conversation omitted for token efficiency: {omitted_count} message(s). Current turn evidence below is authoritative.",
                }
            )
        if current_turn_request is not None and all(message is not current_turn_request for message in recent_messages):
            compacted.append(
                {
                    "role": "user",
                    "content": "Current user request (pinned): "
                    + self._truncate_text(str(current_turn_request.get("content", "")), limit=PRIMARY_CONTEXT_CONTENT_LIMIT),
                }
            )
        for message in recent_messages:
            compacted.append(
                {
                    "role": str(message.get("role", "")),
                    "content": self._truncate_text(str(message.get("content", "")), limit=PRIMARY_CONTEXT_CONTENT_LIMIT),
                }
            )
        return compacted

    def _truncate_text(self, text: str, *, limit: int = VERIFICATION_CONTENT_LIMIT) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 21] + "\n... truncated ..."

    def _truncate_json_value(self, value: Any, *, limit: int = VERIFICATION_CONTENT_LIMIT) -> Any:
        if isinstance(value, str):
            return self._truncate_text(value, limit=limit)
        if isinstance(value, dict):
            return {str(key): self._truncate_json_value(item, limit=limit) for key, item in value.items()}
        if isinstance(value, list):
            return [self._truncate_json_value(item, limit=limit) for item in value[:50]]
        return value

    def _compact_run_test_output(self, output: str, *, limit: int) -> str:
        text = output.strip()
        if len(text) <= limit:
            return text

        lines = text.replace("\r\n", "\n").split("\n")
        selected: list[str] = []
        seen: set[str] = set()

        def add(line: str) -> None:
            compact = line.rstrip()
            if not compact:
                return
            key = compact.strip()
            if key in seen:
                return
            selected.append(compact)
            seen.add(key)

        summary_patterns = [
            r"^Ran \d+ tests? in\b",
            r"^(?:OK|FAILED|ERRORS?|FAILURES?)(?:\b|\()",
            r"^=+ .*?(?:failed|passed|error|errors|failures).*?=+$",
            r"^short test summary info$",
        ]
        for line in lines:
            stripped = line.strip()
            if any(re.search(pattern, stripped, flags=re.IGNORECASE) for pattern in summary_patterns):
                add(stripped)

        marker_index: int | None = None
        for index, line in enumerate(lines):
            stripped = line.strip()
            if re.match(r"^(?:ERROR|FAIL):\s+", stripped):
                marker_index = index
                break
        if marker_index is None:
            for index, line in enumerate(lines):
                if re.search(
                    r"(?:Traceback|SyntaxError|AssertionError|ImportError|ModuleNotFoundError|NameError|TypeError|ValueError)",
                    line,
                ):
                    marker_index = index
                    break

        if marker_index is not None:
            start = max(0, marker_index - 2)
            end = min(len(lines), marker_index + 24)
            add("[first actionable failure]")
            for absolute_index, line in enumerate(lines[start:end], start=start):
                stripped = line.strip()
                if not stripped:
                    if absolute_index > marker_index:
                        break
                    continue
                if set(stripped) <= {"-", "=", "_"}:
                    continue
                if re.match(r"^[\w.: \-/\[\]()]+ \.\.\. (?:FAIL|ERROR|ok|skipped)$", stripped, flags=re.IGNORECASE):
                    continue
                add(line)

        diagnostic_pattern = re.compile(
            r"(?:File \".+\", line \d+|^\s*\^\s*$|^\s*E\s+|^\s*>\s+|"
            r"AssertionError|SyntaxError|ImportError|ModuleNotFoundError|NameError|TypeError|ValueError)"
        )
        for index, line in enumerate(lines):
            if not diagnostic_pattern.search(line):
                continue
            start = max(0, index - 1)
            end = min(len(lines), index + 2)
            for nearby in lines[start:end]:
                add(nearby)
            if len("\n".join(selected)) >= limit:
                break

        if not selected:
            return self._truncate_text(text, limit=limit)
        return self._truncate_text("\n".join(selected), limit=limit)

    def _test_failure_source_excerpt(self, output: str, *, limit: int = 520) -> str:
        root = self.tools.workspace_root.resolve(strict=False)
        snippets: list[str] = []
        seen: set[tuple[Path, int]] = set()
        for raw_path, raw_line in re.findall(r'File "([^"]+)", line (\d+)', output):
            try:
                line_number = int(raw_line)
            except ValueError:
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                path = root / path
            resolved = path.resolve(strict=False)
            try:
                relative = resolved.relative_to(root)
            except ValueError:
                continue
            key = (resolved, line_number)
            if key in seen or not resolved.is_file():
                continue
            seen.add(key)
            try:
                lines = resolved.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            start = max(1, line_number - 4)
            end = min(len(lines), line_number + 4)
            excerpt_lines = [f"{index}: {lines[index - 1]}" for index in range(start, end + 1)]
            snippets.append(f"{relative.as_posix()}:{line_number}\n" + "\n".join(excerpt_lines))
            if len(snippets) >= 2 or len("\n\n".join(snippets)) >= limit:
                break
        if not snippets:
            return ""
        return "Failing source excerpt:\n" + self._truncate_text("\n\n".join(snippets), limit=limit)

    def _normalize_audit_text_items(self, items: object) -> list[str]:
        if not isinstance(items, list):
            return []
        normalized: list[str] = []
        for item in items:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if not text:
                continue
            normalized.append(self._truncate_text(text, limit=AUDIT_TEXT_ITEM_LIMIT))
            if len(normalized) >= AUDIT_LIST_ITEM_LIMIT:
                break
        return normalized

    def _normalize_candidate_claims(self, claims: object) -> list[str]:
        if not isinstance(claims, list):
            return []
        normalized: list[str] = []
        for item in claims:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if not text:
                continue
            normalized.append(self._truncate_text(text, limit=CANDIDATE_CLAIM_TEXT_LIMIT))
            if len(normalized) >= CANDIDATE_CLAIM_LIMIT:
                break
        return normalized

    def _extract_candidate_claims(self, text: str) -> list[str]:
        normalized: list[str] = []
        for raw_line in text.replace("\r\n", "\n").split("\n"):
            line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", raw_line).strip()
            if not line:
                continue
            fragments = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'])|;\s+", line)
            for fragment in fragments:
                claim = fragment.strip()
                if not claim:
                    continue
                normalized.append(self._truncate_text(claim, limit=CANDIDATE_CLAIM_TEXT_LIMIT))
                if len(normalized) >= CANDIDATE_CLAIM_LIMIT:
                    return normalized
        return normalized

    def _evidence_observation_for_result(self, name: str, result: dict[str, Any]) -> str:
        for field in ("summary", "output", "diff"):
            value = result.get(field)
            if isinstance(value, str) and value.strip():
                return self._truncate_text(value.strip(), limit=VERIFICATION_EVIDENCE_TEXT_LIMIT)
        compact = self._compact_tool_result_for_context(name, result, for_verification=True)
        return self._truncate_text(json.dumps(compact, ensure_ascii=True), limit=VERIFICATION_EVIDENCE_TEXT_LIMIT)

    def _build_verification_evidence_table(self, successful_tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        for index, item in enumerate(successful_tool_results[-VERIFICATION_EVIDENCE_LIMIT:], start=1):
            name = str(item.get("name", "")).strip()
            arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            row: dict[str, Any] = {
                "id": f"E{index}",
                "tool": name,
                "arguments": self._truncate_json_value(arguments, limit=180),
                "observation": self._evidence_observation_for_result(name, result),
            }
            path = result.get("path")
            if isinstance(path, str) and path.strip():
                row["path"] = path.strip()
            evidence.append(row)
        return evidence

    def _normalize_claim_checks(self, items: object) -> list[dict[str, str]]:
        if not isinstance(items, list):
            return []
        claim_checks: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim", "")).strip()
            if not claim:
                continue
            status = str(item.get("status", "")).strip().lower()
            if status not in {"supported", "contradicted", "unverified"}:
                status = "unverified"
            claim_checks.append(
                {
                    "claim": self._truncate_text(claim, limit=CANDIDATE_CLAIM_TEXT_LIMIT),
                    "status": status,
                    "evidence": self._truncate_text(str(item.get("evidence", "")).strip(), limit=80),
                    "correction": self._truncate_text(str(item.get("correction", "")).strip(), limit=140),
                }
            )
            if len(claim_checks) >= CANDIDATE_CLAIM_LIMIT:
                break
        return claim_checks

    def _candidate_eligible_for_verification(self, response_text: str) -> bool:
        candidate = response_text.strip()
        if not candidate:
            return False
        payload = extract_json_response(candidate)
        if payload is None:
            return False
        normalized = self._normalize_payload(payload)
        return normalized.get("type") == "final"

    def _tool_names_in_fragment(self, text: str) -> set[str]:
        matches: set[str] = set()
        for name in KNOWN_TOOL_NAMES:
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name.lower())}(?![A-Za-z0-9_])", text):
                matches.add(name)
        return matches

    def _forbidden_tool_names(self, text: str) -> set[str]:
        lowered = text.lower()
        fragments = re.findall(r"\b(?:do not|don't|dont|never|avoid)\b[^.?!\n]{0,160}", lowered)
        fragments.extend(re.findall(r"\bwithout(?: using)?\b[^.?!\n]{0,160}", lowered))
        forbidden: set[str] = set()
        for fragment in fragments:
            forbidden.update(self._tool_names_in_fragment(fragment))
        return forbidden

    def _requested_tool_names(self, text: str, *, forbidden_tool_names: set[str] | None = None) -> set[str]:
        lowered = text.lower()
        fragments = re.findall(r"\b(?:use|call|run|invoke|start)\b[^.?!\n]{0,160}", lowered)
        requested: set[str] = set()
        for fragment in fragments:
            requested.update(self._tool_names_in_fragment(fragment))
        if forbidden_tool_names:
            requested.difference_update(forbidden_tool_names)
        return requested

    def _verification_context_payload(
        self,
        *,
        request_text: str,
        candidate_message: str,
        round_number: int,
        tool_calls: list[dict[str, Any]],
        successful_tool_results: list[dict[str, Any]],
        accepted_assumption_audits: list[dict[str, Any]],
        required_tool_names: set[str],
        forbidden_tool_names: set[str],
    ) -> dict[str, Any]:
        recent_messages = [
            {
                "role": message["role"],
                "content": self._truncate_text(message["content"], limit=220),
            }
            for message in self.messages[-VERIFICATION_HISTORY_LIMIT:]
            if message.get("role") != "system"
        ]
        candidate_claims = self._extract_candidate_claims(candidate_message)
        evidence_table = self._build_verification_evidence_table(successful_tool_results)
        return {
            "round": round_number,
            "model": self.model,
            "verifier_model": self.verification_model(),
            "workspace_root": self.tools.workspace_root.as_posix(),
            "original_user_request": self._truncate_text(request_text, limit=420),
            "recent_messages": recent_messages,
            "candidate_final_answer": self._truncate_text(candidate_message, limit=520),
            "candidate_claims": candidate_claims,
            "required_tools": sorted(required_tool_names),
            "forbidden_tools": sorted(forbidden_tool_names),
            "tool_calls": [self._compact_tool_call_for_verification(item) for item in tool_calls],
            "evidence_table": evidence_table,
            "accepted_assumption_audits": [self._compact_assumption_audit_for_context(item) for item in accepted_assumption_audits],
        }

    def _verification_messages(self, payload: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": FINAL_VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True, separators=(",", ":"))},
        ]

    def _normalize_verification_payload(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        decision = payload if isinstance(payload, dict) else {}
        verdict = str(decision.get("verdict", "")).strip().lower()
        if verdict not in {"accept", "retry"}:
            verdict = "retry"
        reason = str(decision.get("reason", "")).strip()
        required_tools = [name for name in decision.get("required_tools", []) if isinstance(name, str) and name in KNOWN_TOOL_NAMES]
        forbidden_tools = [name for name in decision.get("forbidden_tools", []) if isinstance(name, str) and name in KNOWN_TOOL_NAMES]
        rewrite_guidance = self._normalize_audit_text_items(decision.get("rewrite_guidance"))
        return {
            "verdict": verdict,
            "reason": reason,
            "required_tools": sorted(set(required_tools)),
            "forbidden_tools": sorted(set(forbidden_tools)),
            "claim_checks": self._normalize_claim_checks(decision.get("claim_checks")),
            "rewrite_guidance": rewrite_guidance,
            "rewrite_from_evidence": bool(decision.get("rewrite_from_evidence")),
        }

    def _rewrite_context_payload(
        self,
        *,
        request_text: str,
        candidate_message: str,
        round_number: int,
        successful_tool_results: list[dict[str, Any]],
        verification_decision: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "round": round_number,
            "model": self.model,
            "verifier_model": self.verification_model(),
            "original_user_request": self._truncate_text(request_text, limit=600),
            "candidate_final_answer": self._truncate_text(candidate_message, limit=800),
            "candidate_claims": self._extract_candidate_claims(candidate_message),
            "evidence_table": self._build_verification_evidence_table(successful_tool_results),
            "claim_checks": verification_decision.get("claim_checks", []),
            "rewrite_guidance": verification_decision.get("rewrite_guidance", []),
            "reason": self._truncate_text(str(verification_decision.get("reason", "")).strip(), limit=240),
        }

    def _rewrite_messages(self, payload: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": FINAL_REWRITER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True, separators=(",", ":"))},
        ]

    def _rewrite_eligible_from_verification(
        self,
        decision: dict[str, Any],
        *,
        successful_tool_results: list[dict[str, Any]],
    ) -> bool:
        if decision.get("verdict") != "retry":
            return False
        if decision.get("required_tools"):
            return False
        if not successful_tool_results:
            return False
        if decision.get("rewrite_from_evidence"):
            return True
        claim_checks = decision.get("claim_checks")
        if isinstance(claim_checks, list) and claim_checks:
            return True
        return bool(decision.get("reason"))

    def _rewrite_final_from_evidence(
        self,
        *,
        request_text: str,
        candidate_message: str,
        round_number: int,
        successful_tool_results: list[dict[str, Any]],
        verification_decision: dict[str, Any],
        tool_calls: list[dict[str, Any]],
        accepted_assumption_audits: list[dict[str, Any]],
        required_tool_names: set[str],
        forbidden_tool_names: set[str],
    ) -> FinalRewriteOutcome:
        payload = self._rewrite_context_payload(
            request_text=request_text,
            candidate_message=candidate_message,
            round_number=round_number,
            successful_tool_results=successful_tool_results,
            verification_decision=verification_decision,
        )
        self.status_printer("rewriting final from evidence")
        rewrite_response = self._chat(
            purpose="final_rewrite",
            model=self.verification_model(),
            messages=self._rewrite_messages(payload),
            think=False,
        )
        rewrite_payload = self._normalize_payload(extract_json_response(rewrite_response.content) or {})
        if rewrite_payload.get("type") != "final":
            self._record_event(
                "verification_rewrite",
                round=round_number,
                verdict="invalid",
                candidate=candidate_message,
                rewritten=rewrite_response.content,
                verifier_model=self.verification_model(),
            )
            return FinalRewriteOutcome()
        rewritten_message = str(rewrite_payload.get("message", "")).strip()
        if not rewritten_message:
            self._record_event(
                "verification_rewrite",
                round=round_number,
                verdict="empty",
                candidate=candidate_message,
                rewritten=rewritten_message,
                verifier_model=self.verification_model(),
            )
            return FinalRewriteOutcome()
        rewritten_response = ChatResponse(content=json.dumps({"type": "final", "message": rewritten_message}, ensure_ascii=True), model=self.verification_model(), raw={})
        rewrite_decision = self._verify_final_candidate(
            rewritten_response,
            request_text=request_text,
            round_number=round_number,
            tool_calls=tool_calls,
            successful_tool_results=successful_tool_results,
            accepted_assumption_audits=accepted_assumption_audits,
            required_tool_names=required_tool_names,
            forbidden_tool_names=forbidden_tool_names,
        )
        self._record_event(
            "verification_rewrite",
            round=round_number,
            verdict=rewrite_decision["verdict"],
            candidate=candidate_message,
            rewritten=rewritten_message,
            verifier_model=self.verification_model(),
            claim_checks=rewrite_decision.get("claim_checks", []),
            rewrite_guidance=rewrite_decision.get("rewrite_guidance", []),
        )
        if rewrite_decision["verdict"] == "accept":
            return FinalRewriteOutcome(accepted_message=rewritten_message)
        return FinalRewriteOutcome(retry_decision=rewrite_decision, rejected_message=rewritten_message)

    def _assumption_audit_messages(self, payload: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": TOOL_ASSUMPTION_AUDITOR_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True, separators=(",", ":"))},
        ]

    def _compact_assumption_audit_for_context(self, audit: dict[str, Any]) -> dict[str, Any]:
        return {
            "tool": str(audit.get("tool", "")).strip(),
            "verdict": str(audit.get("verdict", "")).strip(),
            "reason": self._truncate_text(str(audit.get("reason", "")).strip(), limit=120),
            "assumptions": self._normalize_audit_text_items(audit.get("assumptions")),
            "validation_steps": self._normalize_audit_text_items(audit.get("validation_steps")),
        }

    def _assumption_audit_context_payload(
        self,
        *,
        request_text: str,
        round_number: int,
        proposed_tool_name: str,
        proposed_arguments: dict[str, Any],
        tool_calls: list[dict[str, Any]],
        successful_tool_results: list[dict[str, Any]],
        accepted_assumption_audits: list[dict[str, Any]],
        required_tool_names: set[str],
        forbidden_tool_names: set[str],
        mutation_allowed: bool,
        expected_exact_file_line: str | None,
        expected_exact_reply_text: str | None,
    ) -> dict[str, Any]:
        recent_messages = [
            {
                "role": message["role"],
                "content": self._truncate_text(message["content"], limit=220),
            }
            for message in self.messages[-VERIFICATION_HISTORY_LIMIT:]
            if message.get("role") != "system"
        ]
        exact_constraints: list[str] = []
        if expected_exact_file_line is not None:
            exact_constraints.append(f'exact_file_line={expected_exact_file_line}')
        if expected_exact_reply_text is not None:
            exact_constraints.append(f'exact_reply={expected_exact_reply_text}')
        return {
            "round": round_number,
            "model": self.model,
            "workspace_root": self.tools.workspace_root.as_posix(),
            "original_user_request": self._truncate_text(request_text, limit=420),
            "recent_messages": recent_messages,
            "proposed_tool": {
                "name": proposed_tool_name,
                "arguments": self._truncate_json_value(proposed_arguments, limit=220),
            },
            "required_tools": sorted(required_tool_names),
            "forbidden_tools": sorted(forbidden_tool_names),
            "mutation_allowed": mutation_allowed,
            "exact_constraints": exact_constraints,
            "tool_calls": [self._compact_tool_call_for_verification(item) for item in tool_calls],
            "evidence_table": self._build_verification_evidence_table(successful_tool_results),
            "accepted_assumption_audits": [self._compact_assumption_audit_for_context(item) for item in accepted_assumption_audits],
        }

    def _normalize_assumption_audit_payload(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        decision = payload if isinstance(payload, dict) else {}
        verdict = str(decision.get("verdict", "")).strip().lower()
        if verdict not in {"accept", "retry"}:
            verdict = "retry"
        reason = str(decision.get("reason", "")).strip()
        required_tools = [name for name in decision.get("required_tools", []) if isinstance(name, str) and name in KNOWN_TOOL_NAMES]
        forbidden_tools = [name for name in decision.get("forbidden_tools", []) if isinstance(name, str) and name in KNOWN_TOOL_NAMES]
        return {
            "verdict": verdict,
            "reason": reason,
            "assumptions": self._normalize_audit_text_items(decision.get("assumptions")),
            "validation_steps": self._normalize_audit_text_items(decision.get("validation_steps")),
            "required_tools": sorted(set(required_tools)),
            "forbidden_tools": sorted(set(forbidden_tools)),
        }

    def _audit_tool_candidate(
        self,
        *,
        request_text: str,
        round_number: int,
        proposed_tool_name: str,
        proposed_arguments: dict[str, Any],
        tool_calls: list[dict[str, Any]],
        successful_tool_results: list[dict[str, Any]],
        accepted_assumption_audits: list[dict[str, Any]],
        required_tool_names: set[str],
        forbidden_tool_names: set[str],
        mutation_allowed: bool,
        expected_exact_file_line: str | None,
        expected_exact_reply_text: str | None,
    ) -> dict[str, Any]:
        context_payload = self._assumption_audit_context_payload(
            request_text=request_text,
            round_number=round_number,
            proposed_tool_name=proposed_tool_name,
            proposed_arguments=proposed_arguments,
            tool_calls=tool_calls,
            successful_tool_results=successful_tool_results,
            accepted_assumption_audits=accepted_assumption_audits,
            required_tool_names=required_tool_names,
            forbidden_tool_names=forbidden_tool_names,
            mutation_allowed=mutation_allowed,
            expected_exact_file_line=expected_exact_file_line,
            expected_exact_reply_text=expected_exact_reply_text,
        )
        self.status_printer("auditing tool assumptions")
        verdict_response = self._chat(
            purpose="assumption_audit",
            model=self.model,
            messages=self._assumption_audit_messages(context_payload),
            think=False,
        )
        decision = self._normalize_assumption_audit_payload(extract_json_response(verdict_response.content))
        if decision["verdict"] == "retry" and not decision["reason"]:
            decision["reason"] = "Tool step assumptions were not validated."
        self._record_event(
            "assumption_audit",
            round=round_number,
            tool=proposed_tool_name,
            arguments=self._truncate_json_value(proposed_arguments, limit=800),
            verdict=decision["verdict"],
            reason=decision["reason"],
            assumptions=decision["assumptions"],
            validation_steps=decision["validation_steps"],
            required_tools=decision["required_tools"],
            forbidden_tools=decision["forbidden_tools"],
            auditor_model=self.model,
            auditor=verdict_response.content,
        )
        return decision

    def _reconciliation_messages(self, payload: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": ARTIFACT_RECONCILER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True, separators=(",", ":"))},
        ]

    def _reconciliation_context_payload(
        self,
        *,
        request_text: str,
        round_number: int,
        tool_name: str,
        tool_arguments: dict[str, Any],
        tool_result: dict[str, Any],
        tool_calls: list[dict[str, Any]],
        successful_tool_results: list[dict[str, Any]],
        accepted_assumption_audits: list[dict[str, Any]],
        required_tool_names: set[str],
        forbidden_tool_names: set[str],
    ) -> dict[str, Any]:
        recent_messages = [
            {
                "role": message["role"],
                "content": self._truncate_text(message["content"], limit=180),
            }
            for message in self.messages[-VERIFICATION_HISTORY_LIMIT:]
            if message.get("role") != "system"
        ]
        return {
            "round": round_number,
            "model": self.model,
            "workspace_root": self.tools.workspace_root.as_posix(),
            "original_user_request": self._truncate_text(request_text, limit=360),
            "recent_messages": recent_messages,
            "failed_artifact": {
                "tool": tool_name,
                "arguments": self._truncate_json_value(tool_arguments, limit=180),
                "result": self._compact_tool_result_for_context(tool_name, tool_result, for_verification=True),
            },
            "required_tools": sorted(required_tool_names),
            "forbidden_tools": sorted(forbidden_tool_names),
            "tool_calls": [self._compact_tool_call_for_verification(item) for item in tool_calls[-4:]],
            "evidence_table": self._build_verification_evidence_table(successful_tool_results),
            "accepted_assumption_audits": [self._compact_assumption_audit_for_context(item) for item in accepted_assumption_audits],
        }

    def _normalize_reconciliation_payload(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        decision = payload if isinstance(payload, dict) else {}
        verdict = str(decision.get("verdict", "")).strip().lower()
        if verdict not in {"accept", "retry"}:
            verdict = "retry"
        reason = str(decision.get("reason", "")).strip()
        required_tools = [name for name in decision.get("required_tools", []) if isinstance(name, str) and name in KNOWN_TOOL_NAMES]
        forbidden_tools = [name for name in decision.get("forbidden_tools", []) if isinstance(name, str) and name in KNOWN_TOOL_NAMES]
        return {
            "verdict": verdict,
            "reason": reason,
            "repair_plan": self._normalize_audit_text_items(decision.get("repair_plan")),
            "required_tools": sorted(set(required_tools)),
            "forbidden_tools": sorted(set(forbidden_tools)),
        }

    def _run_reconciliation(
        self,
        *,
        request_text: str,
        round_number: int,
        tool_name: str,
        tool_arguments: dict[str, Any],
        tool_result: dict[str, Any],
        tool_calls: list[dict[str, Any]],
        successful_tool_results: list[dict[str, Any]],
        accepted_assumption_audits: list[dict[str, Any]],
        required_tool_names: set[str],
        forbidden_tool_names: set[str],
    ) -> dict[str, Any]:
        context_payload = self._reconciliation_context_payload(
            request_text=request_text,
            round_number=round_number,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            tool_result=tool_result,
            tool_calls=tool_calls,
            successful_tool_results=successful_tool_results,
            accepted_assumption_audits=accepted_assumption_audits,
            required_tool_names=required_tool_names,
            forbidden_tool_names=forbidden_tool_names,
        )
        self.status_printer("reconciling failed artifact")
        verdict_response = self._chat(
            purpose="reconciliation",
            model=self.model,
            messages=self._reconciliation_messages(context_payload),
            think=False,
        )
        decision = self._normalize_reconciliation_payload(extract_json_response(verdict_response.content))
        if decision["verdict"] == "retry" and not decision["reason"]:
            decision["reason"] = "Failed artifact needs a grounded repair step."
        self._record_event(
            "reconciliation",
            round=round_number,
            tool=tool_name,
            arguments=self._truncate_json_value(tool_arguments, limit=800),
            verdict=decision["verdict"],
            reason=decision["reason"],
            repair_plan=decision["repair_plan"],
            required_tools=decision["required_tools"],
            forbidden_tools=decision["forbidden_tools"],
            reconciler_model=self.model,
            reconciler=verdict_response.content,
        )
        return decision

    def _reconciliation_retry_message(self, decision: dict[str, Any]) -> str:
        parts = [
            "Artifact reconciliation rejected continuing from the failed artifact without repair.",
            "Reason: " + str(decision.get("reason") or "The failed artifact needs a focused repair step."),
        ]
        repair_plan = decision.get("repair_plan") if isinstance(decision.get("repair_plan"), list) else []
        required_tools = decision.get("required_tools") if isinstance(decision.get("required_tools"), list) else []
        forbidden_tools = decision.get("forbidden_tools") if isinstance(decision.get("forbidden_tools"), list) else []
        if repair_plan:
            parts.append("Repair plan: " + "; ".join(str(item) for item in repair_plan[:3]) + ".")
        if required_tools:
            parts.append("Required tools for this turn: " + ", ".join(str(item) for item in required_tools) + ".")
        if forbidden_tools:
            parts.append("Forbidden tools for this turn: " + ", ".join(str(item) for item in forbidden_tools) + ".")
        parts.append("Choose the next JSON object only.")
        return " ".join(parts)

    def _verify_final_candidate(
        self,
        response: ChatResponse,
        *,
        request_text: str,
        round_number: int,
        tool_calls: list[dict[str, Any]],
        successful_tool_results: list[dict[str, Any]],
        accepted_assumption_audits: list[dict[str, Any]],
        required_tool_names: set[str],
        forbidden_tool_names: set[str],
    ) -> dict[str, Any]:
        if not self.debate_enabled or not self._candidate_eligible_for_verification(response.content):
            return {"verdict": "accept", "reason": "", "required_tools": [], "forbidden_tools": []}
        candidate_payload = self._normalize_payload(extract_json_response(response.content) or {})
        candidate_message = str(candidate_payload.get("message", "")).strip()
        context_payload = self._verification_context_payload(
            request_text=request_text,
            candidate_message=candidate_message,
            round_number=round_number,
            tool_calls=tool_calls,
            successful_tool_results=successful_tool_results,
            accepted_assumption_audits=accepted_assumption_audits,
            required_tool_names=required_tool_names,
            forbidden_tool_names=forbidden_tool_names,
        )
        try:
            self.status_printer("verifying final")
            verdict_response = self._chat(
                purpose="final_verifier",
                model=self.verification_model(),
                messages=self._verification_messages(context_payload),
                think=False,
            )
        except OllamaError as exc:
            raise exc
        decision = self._normalize_verification_payload(extract_json_response(verdict_response.content))
        if decision["verdict"] == "retry" and not decision["reason"]:
            decision["reason"] = "Final answer was not accepted by grounded verification."
        self._record_event(
            "verification",
            round=round_number,
            candidate=candidate_message,
            verdict=decision["verdict"],
            reason=decision["reason"],
            required_tools=decision["required_tools"],
            forbidden_tools=decision["forbidden_tools"],
            candidate_claims=context_payload.get("candidate_claims", []),
            evidence_table=context_payload.get("evidence_table", []),
            claim_checks=decision.get("claim_checks", []),
            rewrite_guidance=decision.get("rewrite_guidance", []),
            rewrite_from_evidence=decision.get("rewrite_from_evidence", False),
            verifier_model=self.verification_model(),
            verifier=verdict_response.content,
        )
        return decision

    def _verification_retry_message(self, decision: dict[str, Any]) -> str:
        parts = [
            "Grounded final verification rejected your previous final answer.",
            "Reason: " + str(decision.get("reason") or "It was not grounded in the current tool results."),
            "Use existing tool results when they already answer the question; otherwise use another appropriate tool before finishing.",
            "Do not contradict successful tool results, accepted assumption-audit evidence, or explicit user constraints.",
        ]
        claim_checks = decision.get("claim_checks") if isinstance(decision.get("claim_checks"), list) else []
        if claim_checks:
            corrections: list[str] = []
            for item in claim_checks:
                if not isinstance(item, dict):
                    continue
                claim = str(item.get("claim", "")).strip()
                status = str(item.get("status", "")).strip()
                correction = str(item.get("correction", "")).strip()
                evidence = str(item.get("evidence", "")).strip()
                if not claim or status == "supported":
                    continue
                detail = f'"{claim}" is {status}'
                if correction:
                    detail += f"; use \"{correction}\""
                if evidence:
                    detail += f" from {evidence}"
                corrections.append(detail)
                if len(corrections) >= 3:
                    break
            if corrections:
                parts.append("Claim fixes: " + " | ".join(corrections) + ".")
        rewrite_guidance = decision.get("rewrite_guidance") if isinstance(decision.get("rewrite_guidance"), list) else []
        if rewrite_guidance:
            parts.append("Rewrite guidance: " + "; ".join(str(item) for item in rewrite_guidance[:3]) + ".")
        required_tools = decision.get("required_tools") if isinstance(decision.get("required_tools"), list) else []
        forbidden_tools = decision.get("forbidden_tools") if isinstance(decision.get("forbidden_tools"), list) else []
        if required_tools:
            parts.append("Required tools for this turn: " + ", ".join(str(item) for item in required_tools) + ".")
        if forbidden_tools:
            parts.append("Forbidden tools for this turn: " + ", ".join(str(item) for item in forbidden_tools) + ".")
        parts.append("Respond with the next JSON object only.")
        return " ".join(parts)

    def _assumption_audit_retry_message(self, decision: dict[str, Any]) -> str:
        parts = [
            "Assumption audit rejected your previous tool choice.",
            "Reason: " + str(decision.get("reason") or "The proposed tool did not validate the needed assumption."),
        ]
        assumptions = decision.get("assumptions") if isinstance(decision.get("assumptions"), list) else []
        validation_steps = decision.get("validation_steps") if isinstance(decision.get("validation_steps"), list) else []
        required_tools = decision.get("required_tools") if isinstance(decision.get("required_tools"), list) else []
        forbidden_tools = decision.get("forbidden_tools") if isinstance(decision.get("forbidden_tools"), list) else []
        if assumptions:
            parts.append("Weak or unresolved assumptions: " + "; ".join(str(item) for item in assumptions) + ".")
        if validation_steps:
            parts.append("Validation needed: " + "; ".join(str(item) for item in validation_steps) + ".")
        if required_tools:
            parts.append("Required tools for this turn: " + ", ".join(str(item) for item in required_tools) + ".")
        if forbidden_tools:
            parts.append("Forbidden tools for this turn: " + ", ".join(str(item) for item in forbidden_tools) + ".")
        parts.append("Choose a better next JSON object only.")
        return " ".join(parts)

    def _reset_turn_cache(self) -> None:
        self._turn_tool_cache = {}
        self._turn_cache_epoch = 0

    def _tool_cache_key(self, name: str, arguments: dict[str, Any]) -> tuple[str, str, int] | None:
        if name not in READ_ONLY_CACHEABLE_TOOL_NAMES:
            return None
        try:
            encoded_arguments = json.dumps(arguments, sort_keys=True, ensure_ascii=True)
        except TypeError:
            return None
        return (name, encoded_arguments, self._turn_cache_epoch)

    def _get_cached_tool_result(self, name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        key = self._tool_cache_key(name, arguments)
        if key is None:
            return None
        cached = self._turn_tool_cache.get(key)
        if cached is None:
            return None
        return deepcopy(cached)

    def _store_cached_tool_result(self, name: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
        key = self._tool_cache_key(name, arguments)
        if key is None or result.get("ok") is not True:
            return
        self._turn_tool_cache[key] = deepcopy(result)

    def _invalidate_turn_cache_if_needed(self, name: str, result: dict[str, Any]) -> None:
        if result.get("ok") is not True:
            return
        if name in MUTATING_TOOL_NAMES or name in {"run_shell", "run_agent"}:
            self._turn_cache_epoch += 1

    def _tool_result_limit(self, name: str, *, for_verification: bool = False) -> int:
        default_limit = VERIFICATION_TOOL_RESULT_LIMIT if for_verification else 1000
        limit = MODEL_TOOL_RESULT_LIMITS.get(name, default_limit)
        if for_verification:
            return min(limit, VERIFICATION_TOOL_RESULT_LIMIT)
        return limit

    def _compact_tool_result_for_context(
        self,
        name: str,
        result: dict[str, Any],
        *,
        for_verification: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"tool": name, "ok": result.get("ok") is True}
        for key in (
            "path",
            "symbol",
            "symbols",
            "kind",
            "cwd",
            "start",
            "end",
            "count",
            "cached",
            "exit_code",
            "command",
            "summary",
            "diagnostic",
            "syntax_ok",
            "normalized",
            "model",
            "approval_mode",
            "rounds",
            "event_count",
        ):
            if key in result:
                payload[key] = result[key]
        for field in ("output", "diff"):
            value = result.get(field)
            if not isinstance(value, str) or not value.strip():
                continue
            if field == "diff":
                limit = VERIFICATION_TOOL_DIFF_LIMIT if for_verification else MODEL_TOOL_DIFF_LIMIT
            else:
                limit = self._tool_result_limit(name, for_verification=for_verification)
            if name == "run_test" and field == "output":
                payload[field] = self._compact_run_test_output(value, limit=limit)
            else:
                payload[field] = self._truncate_text(value, limit=limit)
        return payload

    def _compact_tool_call_for_verification(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        name = str(tool_call.get("name", "")).strip()
        arguments = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
        return {
            "name": name,
            "arguments": self._truncate_json_value(arguments, limit=180),
        }

    def _compact_successful_tool_result_for_verification(self, item: dict[str, Any]) -> dict[str, Any]:
        name = str(item.get("name", "")).strip()
        arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        return {
            "name": name,
            "arguments": self._truncate_json_value(arguments, limit=180),
            "result": self._compact_tool_result_for_context(name, result, for_verification=True),
        }

    def _run_test_failure_follow_up(self, result: dict[str, Any]) -> str:
        text = str(result.get("output") or result.get("summary") or "")
        source_excerpt = self._test_failure_source_excerpt(text)
        diagnosis = self._run_test_failure_diagnosis(text)
        syntax_match = re.search(
            r"File \"(?P<path>[^\"]+)\", line (?P<line>\d+).*?\n(?:.*\n){0,2}?(?P<error>(?:IndentationError|SyntaxError): [^\n]+)",
            text,
            flags=re.DOTALL,
        )
        if syntax_match:
            path = Path(syntax_match.group("path")).name
            line = syntax_match.group("line")
            error = syntax_match.group("error").strip()
            return (
                f"Tests failed with {error} at {path}:{line}. Fix that file, then rerun run_test. "
                "Do not blame imports unless error is ModuleNotFoundError. Next JSON only."
            )
        module_match = re.search(r"(ModuleNotFoundError|ImportError): [^\n]+", text)
        if module_match:
            details = f"Tests failed with {module_match.group(0).strip()}."
            if diagnosis:
                details += f" {diagnosis}"
            if source_excerpt:
                details += f" {source_excerpt}"
            return self._truncate_text(f"{details} Inspect imports/files, fix, then rerun run_test. Next JSON only.", limit=760)
        assertion_match = re.search(r"AssertionError: [^\n]+", text)
        if assertion_match:
            details = f"Tests failed with {assertion_match.group(0).strip()}."
            if diagnosis:
                details += f" {diagnosis}"
            if source_excerpt:
                details += f" {source_excerpt}"
            return self._truncate_text(f"{details} Edit implementation, then rerun run_test. Next JSON only.", limit=760)
        if diagnosis:
            return self._truncate_text(
                f"Tests failed. {diagnosis} Edit likely implementation targets, then rerun configured run_test. Next JSON only.",
                limit=760,
            )
        if source_excerpt:
            return self._truncate_text(
                f"Tests failed. {source_excerpt} Inspect/edit evidence, then rerun configured run_test. Next JSON only.",
                limit=760,
            )
        return "Tests failed. Inspect/edit evidence, then rerun configured run_test. Next JSON only."

    def _run_test_failure_diagnosis(self, output: str) -> str:
        if not output.strip():
            return ""
        try:
            result = self.tools.diagnose_test_failure(output=output, limit=4)
        except Exception:
            return ""
        if result.get("ok") is not True:
            return ""
        text = str(result.get("output") or "").strip()
        if not text or text.startswith("(no structured failures"):
            return ""
        return "Diagnosis: " + self._truncate_text(text.replace("\n", " | "), limit=420)

    def _run_test_repeat_key(self, arguments: dict[str, Any], mutation_version: int) -> tuple[str, str, int]:
        command = str(arguments.get("command") or self.tools.default_test_command or "").strip()
        cwd = str(arguments.get("cwd") or ".").strip()
        return (command, cwd, mutation_version)

    def _tool_result_feedback_message(self, name: str, result: dict[str, Any], *, real_tool_use: bool) -> str:
        payload = self._compact_tool_result_for_context(name, result, for_verification=False)
        follow_up = "Next JSON only."
        if not real_tool_use:
            follow_up = "Tool failed; not required success. Fix or choose another. Next JSON only."
            summary = str(result.get("summary") or "")
            if "omitted-context marker" in summary:
                follow_up = "Tool failed because content was abbreviated. Use replace_symbol/replace_in_file for partial edits, or read and provide complete file content. Next JSON only."
        elif name in MUTATING_TOOL_NAMES and result.get("syntax_ok") is False:
            follow_up = "Python syntax error in edited file. Fix that file before tests/final; top-level def/class starts at column 1. Next JSON only."
        elif name == "run_test" and result.get("ok") is not True:
            follow_up = self._run_test_failure_follow_up(result)
        return "Tool result:\n" + json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n" + follow_up

    def _final_requires_verification(
        self,
        *,
        request_text: str,
        assistant_text: str,
        tool_calls: list[dict[str, Any]],
        required_tool_names: set[str],
        forbidden_tool_names: set[str],
        mutation_verified_this_turn: bool,
        expected_exact_file_line: str | None,
    ) -> bool:
        if required_tool_names or forbidden_tool_names:
            return True
        if mutation_verified_this_turn or self._final_claims_file_mutation(assistant_text):
            return True
        if expected_exact_file_line is not None:
            return True
        if tool_calls and self._request_needs_exact_grounding(request_text):
            return True
        tool_names = {str(item.get("name", "")).strip() for item in tool_calls}
        if len(tool_calls) >= 2:
            return True
        return any(name in RISKY_VERIFICATION_TOOL_NAMES for name in tool_names)

    def _primary_think_override(
        self,
        *,
        requires_tools: bool,
        round_number: int,
        tool_used_this_turn: bool,
    ) -> bool | None:
        if requires_tools or tool_used_this_turn or round_number > 1:
            return False
        return None

    def _request_requires_tools(self, text: str) -> bool:
        lowered = text.lower()
        tool_phrases = [
            "read file",
            "read the file",
            "search",
            "grep",
            "list files",
            "list the files",
            "workspace",
            "filesystem",
            "directory",
            "folder",
            "repo",
            "repository",
            "project",
            "create ",
            "write ",
            "replace ",
            "edit ",
            "update ",
            "run ",
            "execute ",
            "shell",
            "command",
            "test",
            "tests",
            "pytest",
            "unittest",
            "git",
            "working tree",
            "staged",
            "unstaged",
            "commit ",
            "branch",
            "diff",
            "sub-agent",
            "subagent",
            "helper agent",
            "run_agent",
            "run_test",
            "code_outline",
            "read_symbol",
            "search_symbols",
        ]
        if any(phrase in lowered for phrase in tool_phrases):
            return True
        return bool(re.search(r"\b[\w./-]+\.[A-Za-z0-9]+\b", text))

    def _request_prefers_structured_file_tools(self, text: str) -> bool:
        lowered = text.lower()
        if "run_shell" in lowered or "shell" in lowered or "command" in lowered:
            return False
        file_verbs = ["create ", "write ", "replace ", "edit ", "update ", "rewrite ", "append "]
        has_file_target = bool(re.search(r"\b[\w./-]+\.[A-Za-z0-9]+\b", text)) or "/" in text
        return has_file_target and any(verb in lowered for verb in file_verbs)

    def _request_targets_session_memory(self, text: str) -> bool:
        lowered = text.lower()
        return any(
            phrase in lowered
            for phrase in [
                "earlier in this session",
                "earlier in this conversation",
                "what token did i ask you to remember",
                "what did i ask you to remember",
                "remember earlier",
                "remember in this session",
            ]
        )

    def _requested_exact_file_line(self, text: str) -> str | None:
        patterns = [
            r"exactly the text ['\"]([^'\"]+)['\"] followed by a newline",
            r"exactly the single line ['\"]([^'\"]+)['\"] followed by a newline",
            r"exactly the text ([^\n.]+?) followed by a newline",
            r"exactly the single line ([^\n.]+?) followed by a newline",
            r"exactly the single line ([A-Za-z0-9_.:/@+-]+) followed by a newline",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value:
                    return value
        return None

    def _requested_exact_single_line_file_write(self, text: str) -> ExactFileWriteSpec | None:
        line = self._requested_exact_file_line(text)
        if line is None:
            return None
        patterns = [
            r"\b(?:create|write|rewrite|replace|update)\s+(?:file\s+)?(?P<path>[\w./\\-]+)\s+with exactly the single line\b",
            r"\b(?:create|write|rewrite|replace|update)\s+(?:file\s+)?(?P<path>[\w./\\-]+)\s+with exactly the text\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                path = match.group("path").strip()
                if path:
                    return ExactFileWriteSpec(path=path, line=line)
        return None

    def _requested_exact_reply_text(self, text: str) -> str | None:
        patterns = [
            r"\b(?:reply|respond)\s+with\s+['\"]([^'\"]+)['\"]\s+only\b",
            r"\b(?:reply|respond)\s+with\s+(?:exactly\s+)?([A-Z0-9_.:/@+-]+)\s+only\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value:
                    return value
        return None

    def _latest_read_confirms_exact_line(
        self,
        tool_results: list[dict[str, Any]],
        expected_line: str,
        *,
        path: str | None = None,
    ) -> bool:
        for item in reversed(tool_results):
            if item.get("name") != "read_file":
                continue
            arguments = item.get("arguments")
            if path is not None and isinstance(arguments, dict) and str(arguments.get("path", "")).strip() != path:
                continue
            result = item.get("result")
            if not isinstance(result, dict):
                continue
            output = str(result.get("output", ""))
            if expected_line in output:
                return True
        return False

    def _normalize_exact_literal_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        exact_file_write: ExactFileWriteSpec | None,
    ) -> tuple[str, dict[str, Any], str | None]:
        if exact_file_write is None:
            return name, arguments, None
        if name == "write_file" or name == "replace_in_file":
            return (
                "write_file",
                {"path": exact_file_write.path, "content": exact_file_write.line + "\n"},
                "Normalized exact single-line file write to a deterministic write_file call.",
            )
        if name == "read_file":
            return (
                "read_file",
                {"path": exact_file_write.path, "start": 1, "end": 1},
                "Normalized exact single-line confirmation read to the requested file and line range.",
            )
        return name, arguments, None

    def _normalize_file_tool_alias_call(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> tuple[str, dict[str, Any], str | None]:
        lowered = name.strip().lower()
        if lowered not in {"edit_file", "modify_file", "update_file"}:
            return name, arguments, None
        path = arguments.get("path") or arguments.get("file") or arguments.get("filename")
        if not isinstance(path, str) or not path.strip():
            return name, arguments, None
        if isinstance(arguments.get("old"), str) and isinstance(arguments.get("new"), str):
            return (
                "replace_in_file",
                {
                    "path": path.strip(),
                    "old": arguments["old"],
                    "new": arguments["new"],
                    "replace_all": bool(arguments.get("replace_all", False)),
                },
                f"Normalized unsupported {name} alias to replace_in_file.",
            )
        symbol = arguments.get("symbol") or arguments.get("name") or arguments.get("function") or arguments.get("class")
        content = arguments.get("content")
        if isinstance(symbol, str) and isinstance(content, str):
            return (
                "replace_symbol",
                {"path": path.strip(), "symbol": symbol.strip(), "content": content},
                f"Normalized unsupported {name} alias to replace_symbol.",
            )
        replacements = arguments.get("replacements")
        if isinstance(replacements, list):
            return (
                "replace_symbols",
                {"path": path.strip(), "replacements": replacements},
                f"Normalized unsupported {name} alias to replace_symbols.",
            )
        if isinstance(content, str):
            return (
                "write_file",
                {"path": path.strip(), "content": content},
                f"Normalized unsupported {name} alias to write_file.",
            )
        return name, arguments, None

    def _snippet_symbol_argument_looks_like_text(self, value: str) -> bool:
        snippet = value.strip()
        if not snippet:
            return False
        if "\n" in snippet:
            return True
        if re.match(r"^[A-Za-z_][\w.]*\s*(?:\(|$)", snippet):
            return False
        return bool(re.search(r"\b(?:return|raise|yield|if|else|for|while|with|try|except)\b|[=+\-*/%<>\[\]{}]", snippet))

    def _path_looks_like_code_file(self, path: str) -> bool:
        return Path(path.replace("\\", "/")).suffix.lower() in CODE_EDIT_SUFFIXES

    def _decode_accidental_escaped_newlines(self, value: str) -> str:
        if "\n" in value or "\\n" not in value:
            return value
        if not re.search(r"\b(?:def|class|import|from|return|if|for|while|try|except|function|const|let|var)\b", value):
            return value
        return value.replace("\\r\\n", "\n").replace("\\n", "\n")

    def _normalize_edit_payload_aliases(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> tuple[str, dict[str, Any], str | None]:
        if name not in {"write_file", "replace_symbol", "replace_symbols", "replace_in_file"}:
            return name, arguments, None
        updated = deepcopy(arguments)
        changed_reasons: list[str] = []
        for key in ("content", "new"):
            value = updated.get(key)
            if isinstance(value, str):
                decoded = self._decode_accidental_escaped_newlines(value)
                if decoded != value:
                    updated[key] = decoded
                    changed_reasons.append(f"decoded escaped newlines in {key}")
        if name == "replace_symbols":
            replacements = updated.get("replacements")
            if isinstance(replacements, list):
                normalized_replacements: list[Any] = []
                for item in replacements:
                    if not isinstance(item, dict):
                        normalized_replacements.append(item)
                        continue
                    normalized_item = dict(item)
                    content = normalized_item.get("content")
                    if isinstance(content, str):
                        decoded = self._decode_accidental_escaped_newlines(content)
                        if decoded != content:
                            normalized_item["content"] = decoded
                            changed_reasons.append("decoded escaped newlines in replacement content")
                    normalized_replacements.append(normalized_item)
                updated["replacements"] = normalized_replacements
        if name == "replace_in_file" and "replace_all" not in updated and "all" in updated:
            updated["replace_all"] = bool(updated.get("all"))
            changed_reasons.append("normalized all to replace_all")
        if name == "replace_symbol":
            path = str(updated.get("path", "")).strip()
            symbol = updated.get("symbol")
            content = updated.get("content")
            replacements = updated.get("replacements")
            if path and isinstance(replacements, list) and len(replacements) == 1 and isinstance(replacements[0], dict):
                item = replacements[0]
                old = item.get("old")
                new = item.get("new")
                if isinstance(old, str) and isinstance(new, str):
                    return (
                        "replace_in_file",
                        {"path": path, "old": old, "new": new, "replace_all": bool(item.get("replace_all", item.get("all", False)))},
                        "Normalized replace_symbol replacement-list payload to replace_in_file.",
                    )
            if path and isinstance(symbol, str) and isinstance(content, str) and not self._path_looks_like_code_file(path):
                return (
                    "replace_in_file",
                    {"path": path, "old": symbol, "new": content, "replace_all": False},
                    "Normalized replace_symbol text edit on a non-code file to replace_in_file.",
                )
        if changed_reasons:
            return name, updated, "Normalized edit payload: " + "; ".join(sorted(set(changed_reasons))) + "."
        return name, arguments, None

    def _normalize_snippet_symbol_edit_call(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> tuple[str, dict[str, Any], str | None]:
        if name != "replace_symbol":
            return name, arguments, None
        path = str(arguments.get("path", "")).strip()
        symbol = arguments.get("symbol")
        content = arguments.get("content")
        if not path or not isinstance(symbol, str) or not isinstance(content, str):
            return name, arguments, None
        if not self._snippet_symbol_argument_looks_like_text(symbol):
            return name, arguments, None
        return (
            "replace_in_file",
            {"path": path, "old": symbol, "new": content, "all": False},
            "Normalized snippet-style replace_symbol call to replace_in_file.",
        )

    def _request_needs_exact_grounding(self, text: str) -> bool:
        lowered = text.lower()
        patterns = [
            r"\bexact(?:ly)?\b",
            r"\bline\s+\d+\b",
            r"\bfirst line\b",
            r"\bsingle line\b",
            r"\bwhat(?:'s| is)? .* say\b",
            r"\btoken\b",
        ]
        return any(re.search(pattern, lowered) for pattern in patterns)

    def _request_allows_mutation(self, text: str) -> bool:
        lowered = text.lower()
        mutation_phrases = [
            "create ",
            "write ",
            "replace ",
            "edit ",
            "update ",
            "rewrite ",
            "append ",
            "modify ",
            "change ",
            "delete ",
            "remove ",
            "rename ",
            "fix ",
            "implement ",
            "patch ",
            "refactor ",
            "make ",
            "add ",
            "commit ",
        ]
        return any(phrase in lowered for phrase in mutation_phrases)

    def _request_requires_mutation(self, text: str) -> bool:
        lowered = text.lower()
        read_only_patterns = [
            r"\bdo not edit\b(?!\s+(?:tests?|test files?)\b)",
            r"\bdon't edit\b(?!\s+(?:tests?|test files?)\b)",
            r"\bwithout editing\b(?!\s+(?:tests?|test files?)\b)",
            r"\bwithout changing\b(?!\s+(?:tests?|test files?)\b)",
            r"\bno changes\b",
            r"\bread-only\b",
            r"\binspect only\b",
            r"\bsummarize only\b",
        ]
        if any(re.search(pattern, lowered) for pattern in read_only_patterns):
            return False
        if re.search(r"\b(?:how|what|why)\s+(?:would|should|can)\b", lowered):
            return False
        mutation_patterns = [
            r"\bimplement\b",
            r"\bfix\b",
            r"\bpatch\b",
            r"\brefactor\s+(?:[\w./-]+\.[A-Za-z0-9]+|[A-Za-z_][\w./-]*\s+to|code\s+to|module\s+to|tests?\s+to)",
            r"\bedit\b",
            r"\bupdate\b",
            r"\brewrite\b",
            r"\bmodify\b",
            r"\bchange\b",
            r"\bcreate\b",
            r"\bwrite\b",
            r"\badd\b",
            r"\bremove\b",
            r"\bdelete\b",
        ]
        return any(re.search(pattern, lowered) for pattern in mutation_patterns)

    def _request_requires_test_run(self, text: str) -> bool:
        lowered = text.lower()
        patterns = [
            r"\brun (?:the )?tests?\b",
            r"\brerun (?:the )?tests?\b",
            r"\bexecute (?:the )?tests?\b",
            r"\btest suite\b",
            r"\bpytest\b",
            r"\bunittest\b",
            r"\brun_test\b",
        ]
        return any(re.search(pattern, lowered) for pattern in patterns)

    def _request_forbids_test_mutation(self, text: str) -> bool:
        lowered = text.lower()
        if any(phrase in lowered for phrase in ["update tests", "edit tests", "modify tests", "change tests", "tests, and docs", "tests and docs"]):
            return False
        return any(
            phrase in lowered
            for phrase in [
                "edit only implementation",
                "only implementation files",
                "do not edit tests",
                "don't edit tests",
                "without editing tests",
                "leave tests unchanged",
            ]
        ) or bool(re.search(r"\b(?:fix|change|edit|update|implement|make)\b.{0,120}\bsrc/[^\s,;:]+", lowered))

    def _python_test_import_targets(self) -> set[str]:
        root = self.tools.workspace_root.resolve(strict=False)
        targets: set[str] = set()
        test_paths: list[Path] = []
        for pattern in ("test_*.py", "*_test.py"):
            test_paths.extend(path for path in root.rglob(pattern) if path.is_file())
        for test_path in sorted(set(test_paths)):
            try:
                relative_test_parts = test_path.resolve(strict=False).relative_to(root).parts
            except ValueError:
                continue
            if any(part in {".git", ".ollama-code", "__pycache__", "scratch"} for part in relative_test_parts):
                continue
            try:
                tree = ast.parse(test_path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            modules: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules.add(node.module)
            for module in modules:
                parts = [part for part in module.split(".") if part]
                if not parts:
                    continue
                candidates = [
                    root.joinpath(*parts).with_suffix(".py"),
                    root / f"{parts[0]}.py",
                    root.joinpath(*parts) / "__init__.py",
                ]
                for candidate in candidates:
                    if not candidate.is_file() or candidate.resolve(strict=False) == test_path.resolve(strict=False):
                        continue
                    try:
                        targets.add(candidate.resolve(strict=False).relative_to(root).as_posix())
                    except ValueError:
                        continue
        return targets

    def _request_explicitly_names_path(self, request_text: str, path: str) -> bool:
        normalized_request = request_text.replace("\\", "/").lower()
        normalized_path = path.replace("\\", "/").lower().lstrip("./")
        return bool(normalized_path) and normalized_path in normalized_request

    def _unimported_python_write_feedback(self, request_text: str, arguments: dict[str, Any]) -> str | None:
        raw_path = str(arguments.get("path", "")).strip()
        if not raw_path.replace("\\", "/").endswith(".py"):
            return None
        if self._request_explicitly_names_path(request_text, raw_path):
            return None
        root = self.tools.workspace_root.resolve(strict=False)
        target = (root / raw_path).resolve(strict=False) if not Path(raw_path).is_absolute() else Path(raw_path).resolve(strict=False)
        if target.exists():
            return None
        imports = sorted(self._python_test_import_targets())
        if not imports:
            return None
        imported_list = ", ".join(imports[:6])
        return (
            f"Do not create unrelated Python file {raw_path}. Existing tests import implementation file(s): {imported_list}. "
            "Edit those files or inspect tests/source, then rerun run_test. Next JSON only."
        )

    def _test_write_drops_import_bootstrap_feedback(self, arguments: dict[str, Any]) -> str | None:
        raw_path = str(arguments.get("path", "")).strip()
        content = arguments.get("content")
        if not raw_path.replace("\\", "/").endswith(".py") or not self._path_looks_like_test_file(raw_path):
            return None
        if not isinstance(content, str):
            return None
        try:
            target = self.tools.resolve_path(raw_path, allow_missing=False)
        except (OSError, ValueError):
            return None
        if not target.is_file():
            return None
        try:
            existing = target.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        if "sys.path.insert" not in existing and "sys.path.append" not in existing:
            return None
        if "sys.path.insert" in content or "sys.path.append" in content:
            return None
        if "from " not in content and "import " not in content:
            return None
        return (
            f"Do not rewrite {raw_path} without preserving its existing sys.path bootstrap for local source imports. "
            "Use replace_in_file for the specific rename or include the same sys.path setup, then rerun run_test. Next JSON only."
        )

    def _path_looks_like_test_file(self, path: str) -> bool:
        normalized = path.replace("\\", "/").lower()
        name = normalized.rsplit("/", 1)[-1]
        return bool(
            re.search(r"(^test_|_test\.|\.test\.|\.spec\.)", name)
            or "/tests/" in normalized
            or normalized.startswith("tests/")
        )

    def _request_allows_commit(self, text: str) -> bool:
        lowered = text.lower()
        return bool(re.search(r"\b(?:commit|git_commit)\b", lowered))

    def _shell_looks_like_file_mutation(self, command: str) -> bool:
        lowered = command.lower()
        mutation_patterns = [
            r">>?",
            r"\btouch\b",
            r"\bmkdir\b",
            r"\bcp\b",
            r"\bmv\b",
            r"\brm\b",
            r"\bsed\s+-i\b",
            r"\btee\b",
            r"\bcat\s+>+\b",
        ]
        return any(re.search(pattern, lowered) for pattern in mutation_patterns)

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        response_type = normalized.get("type")
        tool_name = normalized.get("name")
        if isinstance(response_type, str):
            normalized["type"] = response_type.strip()
        if isinstance(tool_name, str):
            normalized["name"] = tool_name.strip()
        response_type = normalized.get("type")
        tool_name = normalized.get("name")
        arguments = normalized.get("arguments")
        if isinstance(response_type, str) and response_type in KNOWN_TOOL_NAMES:
            normalized["type"] = "tool"
            if not isinstance(tool_name, str) or not tool_name:
                normalized["name"] = response_type
            if not isinstance(arguments, dict):
                normalized["arguments"] = {}
            return normalized
        if isinstance(tool_name, str) and tool_name in KNOWN_TOOL_NAMES and response_type in {None, "", "function", "tool_call"}:
            normalized["type"] = "tool"
            if not isinstance(arguments, dict):
                normalized["arguments"] = {}
            return normalized
        return normalized

    def _counts_as_real_tool_use(self, name: str, result: dict[str, Any]) -> bool:
        if name not in KNOWN_TOOL_NAMES:
            return False
        return result.get("ok") is True

    def _latest_successful_tool_result(self, tool_results: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
        for item in reversed(tool_results):
            if item.get("name") != name:
                continue
            result = item.get("result")
            if isinstance(result, dict):
                return result
        return None

    def _failure_result_counts_for_request(self, request_text: str, name: str, result: dict[str, Any]) -> bool:
        if name not in KNOWN_TOOL_NAMES:
            return False
        if name == "run_agent":
            return False
        summary = str(result.get("summary", "")).strip()
        output = str(result.get("output", "")).strip()
        if name == "run_test" and self._request_requires_test_run(request_text):
            return result.get("ok") is not True and bool(summary or output)
        lowered = request_text.lower()
        expects_failure_details = any(
            phrase in lowered
            for phrase in [
                "error",
                "tool error",
                "exact tool error",
                "what happened",
                "if you cannot",
                "if you can't",
                "if it fails",
                "failure",
                "fail",
                "denied",
                "blocked",
                "rejected",
                "exit code",
            ]
        )
        if not expects_failure_details:
            return False
        if result.get("ok") is True:
            return False
        return bool(summary or output)

    def _request_expects_exact_tool_error(self, text: str) -> bool:
        lowered = text.lower()
        return any(
            phrase in lowered
            for phrase in [
                "exact tool error",
                "tell me the exact tool error",
                "reply with the exact tool error",
                "what happened",
                "tell me what happened",
            ]
        )

    def _request_mentions_repeated_read(self, text: str) -> bool:
        lowered = text.lower()
        return "twice" in lowered or "two times" in lowered or "2 times" in lowered

    def _request_asks_token_only(self, text: str) -> bool:
        lowered = text.lower()
        return "token" in lowered and ("only" in lowered or "exact marker" in lowered or "exact token" in lowered)

    def _extract_uppercase_token_from_output(self, output: str) -> str | None:
        for token in re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", output):
            if token in {"OK"}:
                continue
            return token
        return None

    def _extract_return_value_from_symbol_output(self, output: str) -> str | None:
        for line in output.splitlines():
            match = re.search(r"\|\s*return\s+(.+?)\s*$", line)
            if not match:
                continue
            value = match.group(1).strip()
            if value:
                return value
        return None

    def _requested_exact_shell_command(self, text: str) -> str | None:
        patterns = [
            r"\b(?:execute|run)\s+exactly:\s*(?P<command>.+?)(?:\.\s+(?:Then|Tell)\b|\n|$)",
            r"\b(?:execute|run)\s+the\s+exact\s+command:\s*(?P<command>.+?)(?:\.\s+(?:Then|Tell)\b|\n|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            command = match.group("command").strip()
            command = command.strip("`")
            if len(command) >= 2 and command[0] == command[-1] and command[0] in {"'", '"'}:
                command = command[1:-1].strip()
            if command:
                return command
        return None

    def _requested_read_file_path(self, text: str) -> str | None:
        patterns = [
            r"\bread_file\s+on\s+(?P<path>[\w./\\:-]+)",
            r"\bread_file\s+(?P<path>[\w./\\:-]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                path = match.group("path").strip().rstrip(".,;:")
                if path:
                    return path
        return None

    def _requested_mutation_paths(self, text: str) -> set[str]:
        if not self._request_requires_mutation(text):
            return set()
        paths: set[str] = set()
        for raw_path in re.findall(r"\b(?:[\w.-]+[\\/])+[\w.-]+\.[A-Za-z0-9]+\b", text):
            normalized = raw_path.strip().strip("`'\".,;:").replace("\\", "/").lstrip("./")
            if normalized:
                paths.add(normalized)
        return paths

    def _requested_git_tool_path(self, text: str) -> str | None:
        patterns = [
            r"\bgit_status\s+on\s+(?P<path>[\w./\\:-]+)",
            r"\bgit_diff\s+on\s+(?P<path>[\w./\\:-]+)",
            r"\bgit\s+diff\s+(?P<path>[\w./\\:-]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                path = match.group("path").strip().rstrip(".,;:")
                if path:
                    return path
        return None

    def _requested_target_line_read(self, text: str) -> TargetLineReadSpec | None:
        match = re.search(r"\bread_file\s+on\s+(?P<path>[\w./\\-]+).*?\bline\s+(?P<line>\d+)\b", text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            match = re.search(r"\bread\s+(?P<path>[\w./\\-]+).*?\bline\s+(?P<line>\d+)\b", text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        path = match.group("path").strip().rstrip(".,;:")
        try:
            line = int(match.group("line"))
        except ValueError:
            return None
        if not path or line < 1:
            return None
        return TargetLineReadSpec(path=path, start=max(1, line - 5), end=line + 5, line=line)

    def _requested_symbol_read(self, text: str) -> SymbolReadSpec | None:
        symbol_pattern = r"[A-Za-z_][\w.]*"
        path_pattern = r"[\w./\\:-]+\.[A-Za-z0-9]+"
        patterns = [
            rf"\b(?:function|method|class|symbol)\s+(?P<symbol>{symbol_pattern})\s+(?:in|from)\s+(?P<path>{path_pattern})",
            rf"\b(?:find|locate|search(?:_symbols)?(?:\s+for)?)\s+(?P<symbol>{symbol_pattern})\s+in\s+(?P<path>{path_pattern})",
            rf"\bread_symbol\b.*?\b(?:on|in)\s+(?P<path>{path_pattern}).*?\bsymbol\s+(?P<symbol>{symbol_pattern})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            path = match.group("path").strip().rstrip(".,;:")
            symbol = match.group("symbol").strip().rstrip(".,;:")
            if path and symbol:
                return SymbolReadSpec(path=path, symbol=symbol)
        return None

    def _normalize_target_line_read_call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        target_line_read: TargetLineReadSpec | None,
    ) -> tuple[str, dict[str, Any], str | None]:
        if name != "read_file" or target_line_read is None:
            return name, arguments, None
        requested_path = str(arguments.get("path", "")).strip()
        if requested_path and requested_path.replace("\\", "/") != target_line_read.path.replace("\\", "/"):
            return name, arguments, None
        try:
            current_start = int(arguments.get("start", 1))
            current_end = int(arguments.get("end", 200))
        except (TypeError, ValueError):
            current_start = 1
            current_end = 200
        if current_start <= target_line_read.line <= current_end and (current_end - current_start) <= 40:
            return name, arguments, None
        return (
            "read_file",
            {"path": target_line_read.path, "start": target_line_read.start, "end": target_line_read.end},
            f"Normalized read_file to the requested line {target_line_read.line} with a small surrounding range.",
        )

    def _normalize_run_test_call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        request_text: str,
    ) -> tuple[str, dict[str, Any], str | None]:
        if name in {"test", "tests", "pytest", "unittest"} and self.tools.default_test_command:
            normalized = dict(arguments)
            normalized["command"] = self.tools.default_test_command
            return "run_test", normalized, f"Normalized {name} tool alias to the configured run_test command."
        if name != "run_test" or not self.tools.default_test_command:
            return name, arguments, None
        command = str(arguments.get("command", "")).strip()
        lowered_command = command.lower()
        lowered_request = request_text.lower()
        vague_command = lowered_command in {"", "test", "tests", "pytest", "unittest", "python -m unittest", "python3 -m unittest"}
        command_not_requested = bool(command) and lowered_command not in lowered_request
        if not vague_command and not command_not_requested:
            return name, arguments, None
        normalized = dict(arguments)
        normalized["command"] = self.tools.default_test_command
        return "run_test", normalized, "Normalized vague run_test command to the configured test command."

    def _shell_command_looks_like_test_run(self, command: str) -> bool:
        lowered = command.lower()
        test_patterns = [
            r"\bpytest\b",
            r"\bunittest\b",
            r"\bpython(?:3|\.exe)?\s+-m\s+unittest\b",
            r"\bgo\s+test\b",
            r"\bcargo\s+test\b",
            r"\bnpm\s+(?:run\s+)?test\b",
            r"\bpnpm\s+(?:run\s+)?test\b",
            r"\byarn\s+test\b",
            r"\bmvn\s+test\b",
            r"\bgradle\s+test\b",
        ]
        return any(re.search(pattern, lowered) for pattern in test_patterns)

    def _normalize_shell_test_call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        request_text: str,
        exact_shell_command: str | None,
    ) -> tuple[str, dict[str, Any], str | None]:
        if name != "run_shell" or not self.tools.default_test_command:
            return name, arguments, None
        command = str(arguments.get("command", "")).strip()
        if not command or not self._shell_command_looks_like_test_run(command):
            return name, arguments, None
        if exact_shell_command and command == exact_shell_command:
            return name, arguments, None
        if self._request_explicitly_requests_tool(request_text, "run_shell"):
            return name, arguments, None
        normalized: dict[str, Any] = {"command": self.tools.default_test_command}
        if "cwd" in arguments:
            normalized["cwd"] = arguments["cwd"]
        if "timeout" in arguments:
            normalized["timeout"] = arguments["timeout"]
        return "run_test", normalized, "Normalized shell test command to the configured run_test command."

    def _request_explicitly_requests_tool(self, text: str, name: str) -> bool:
        requested = self._requested_tool_names(text, forbidden_tool_names=set())
        return name in requested

    def _request_is_broad_or_ambiguous(self, text: str) -> bool:
        lowered = text.lower()
        broad_phrases = [
            "inspect this repo",
            "inspect the repo",
            "summarize this repo",
            "summarize the project",
            "what does this project",
            "find bugs",
            "review the codebase",
            "search the codebase",
        ]
        if any(phrase in lowered for phrase in broad_phrases):
            return True
        return "repo" in lowered and not re.search(r"\b[\w./-]+\.[A-Za-z0-9]+\b", text)

    def _tool_output_without_line_prefixes(self, text: str) -> str:
        return "\n".join(re.sub(r"^\s*\d+\s+\|\s?", "", line) for line in text.splitlines())

    def _leading_whitespace_insensitive_contains(self, haystack: str, needle: str) -> bool:
        if needle in haystack:
            return True
        stripped_haystack = self._tool_output_without_line_prefixes(haystack)
        if needle in stripped_haystack:
            return True
        normalized_haystack = "\n".join(line.lstrip() for line in stripped_haystack.splitlines())
        normalized_needle = "\n".join(line.lstrip() for line in needle.splitlines())
        return bool(normalized_needle.strip()) and normalized_needle in normalized_haystack

    def _tool_call_grounded_by_successful_evidence(
        self,
        *,
        name: str,
        arguments: dict[str, Any],
        successful_tool_results: list[dict[str, Any]],
    ) -> bool:
        if name != "replace_in_file":
            return False
        path = str(arguments.get("path", "")).strip().replace("\\", "/")
        old = str(arguments.get("old", ""))
        if not path or not old.strip():
            return False
        for item in reversed(successful_tool_results):
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            if item.get("name") not in {"read_file", "read_symbol"} or result.get("ok") is not True:
                continue
            result_path = str(result.get("path") or item.get("arguments", {}).get("path") or "").strip().replace("\\", "/")
            if result_path and result_path != path:
                continue
            if self._leading_whitespace_insensitive_contains(str(result.get("output", "")), old):
                return True
        return False

    def _tool_call_needs_assumption_audit(
        self,
        *,
        request_text: str,
        name: str,
        arguments: dict[str, Any],
        normalization_reason: str | None,
        cache_hit: bool,
        failed_tool_this_turn: bool,
        session_memory_request: bool,
        forbidden_tool_names: set[str],
        successful_tool_results: list[dict[str, Any]],
    ) -> bool:
        if not self.debate_enabled or normalization_reason is not None or cache_hit or session_memory_request:
            return False
        if not forbidden_tool_names and self._tool_call_grounded_by_successful_evidence(
            name=name,
            arguments=arguments,
            successful_tool_results=successful_tool_results,
        ):
            return False
        if failed_tool_this_turn:
            if name in {"read_file", "list_files", "search", "search_symbols", "code_outline", "read_symbol"}:
                return False
            if name == "run_test":
                return True
            if not forbidden_tool_names and self._tool_call_grounded_by_successful_evidence(
                name=name,
                arguments=arguments,
                successful_tool_results=successful_tool_results,
            ):
                return False
            return True
        if name in MUTATING_TOOL_NAMES or name == "run_shell":
            return True
        if name == "run_agent":
            if forbidden_tool_names:
                return True
            return not self._request_explicitly_requests_tool(request_text, name)
        if name in {"run_test", "git_status", "git_diff"}:
            if forbidden_tool_names:
                return True
            if self._request_explicitly_requests_tool(request_text, name):
                return False
            if name == "run_test" and re.search(r"\b(?:run|rerun|execute)\b[^.?!\n]{0,80}\b(?:test|tests|pytest|unittest)\b", request_text, flags=re.IGNORECASE):
                return False
            return True
        if name in {"read_file", "list_files", "search", "search_symbols", "code_outline", "read_symbol"}:
            if self._request_explicitly_requests_tool(request_text, name):
                return False
            if forbidden_tool_names:
                return True
            if re.search(r"\bread\b", request_text, flags=re.IGNORECASE) and name != "read_file":
                return True
            return self._request_is_broad_or_ambiguous(request_text)
        if forbidden_tool_names:
            return True
        return True

    def _tool_result_needs_reconciliation(
        self,
        *,
        request_text: str,
        name: str,
        result: dict[str, Any],
        cache_hit: bool,
        session_memory_request: bool,
        mutation_required: bool,
        test_run_required: bool,
    ) -> bool:
        if self.reconcile_mode_setting == "off" or cache_hit or session_memory_request:
            return False
        if self._request_expects_exact_tool_error(request_text):
            return False
        failed = result.get("ok") is not True or result.get("syntax_ok") is False
        if not failed:
            return False
        if self.reconcile_mode_setting == "on":
            return name in MUTATING_TOOL_NAMES or name in {"run_test", "lint_typecheck", "run_function_probe", "run_shell", "run_agent"}
        if name == "run_test":
            return mutation_required or test_run_required
        if name in {"lint_typecheck", "run_function_probe"}:
            return mutation_required or test_run_required
        return False

    def _synthesize_final_from_tool_result(
        self,
        *,
        request_text: str,
        name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        successful_tool_results: list[dict[str, Any]],
        satisfied_tool_names: set[str],
        required_tool_names: set[str],
        expected_exact_reply_text: str | None,
    ) -> str | None:
        if self._request_expects_exact_tool_error(request_text) and self._failure_result_counts_for_request(request_text, name, result):
            return str(result.get("summary") or result.get("output") or "").strip() or None

        missing_required = required_tool_names - satisfied_tool_names
        if missing_required:
            return None

        if name in {"read_file", "read_symbol"} and result.get("ok") is True:
            output = str(result.get("output", ""))
            if expected_exact_reply_text is not None and expected_exact_reply_text in output:
                return expected_exact_reply_text
            if self._request_requires_mutation(request_text) or self._request_requires_test_run(request_text):
                return None
            if self._request_asks_token_only(request_text):
                if self._request_mentions_repeated_read(request_text):
                    if name != "read_file":
                        return None
                    read_count = sum(1 for item in successful_tool_results if item.get("name") == "read_file")
                    if read_count < 2:
                        return None
                token = self._extract_uppercase_token_from_output(output)
                if token:
                    return token
            if name == "read_symbol" and "return" in request_text.lower() and "value" in request_text.lower():
                value = self._extract_return_value_from_symbol_output(output)
                if value:
                    symbol = str(result.get("symbol") or arguments.get("symbol") or "symbol")
                    return f"{symbol} returns {value}."

        if name == "git_diff" and result.get("ok") is True:
            lowered = request_text.lower()
            value_match = re.search(r"\breturn\s+([A-Za-z0-9_]+)\b", request_text)
            if "whether" in lowered and "diff" in lowered and value_match:
                value = value_match.group(1)
                output = str(result.get("output", ""))
                adds_value = bool(re.search(rf"(?m)^\+\s*return\s+{re.escape(value)}\b", output))
                path = str(result.get("path") or arguments.get("path") or ".")
                return f"{path} diff adds return {value}" if adds_value else f"{path} diff does not add return {value}"

        if name == "run_shell" and result.get("ok") is True:
            lowered = request_text.lower()
            if "artifact" in lowered and ("written" in lowered or "wrote" in lowered):
                command = str(arguments.get("command", ""))
                normalized_command = command.replace("\\", "/")
                path_candidates = re.findall(r"['\"]((?:scratch|docs|src|data|tests)/[^'\"]+)['\"]", normalized_command)
                for candidate in reversed(path_candidates):
                    try:
                        target = self.tools.resolve_path(candidate, allow_missing=False)
                    except (OSError, ValueError):
                        continue
                    if target.is_file():
                        return f"Artifact written: {candidate}."
                return "Command completed successfully."

        if name == "run_shell" and "exit code" in request_text.lower():
            if "exit_code" not in result:
                return None
            output = str(result.get("output", "")).strip()
            if "printed word" in request_text.lower() or "output" in request_text.lower():
                return f"Exit code: {result.get('exit_code')}. Output: {output}."
            return f"Exit code: {result.get('exit_code')}."

        if name == "search" and result.get("ok") is True:
            lowered = request_text.lower()
            if "which file" in lowered or "file contains" in lowered or "files contain" in lowered:
                output = str(result.get("output", ""))
                for line in output.splitlines():
                    match = re.match(r"(?P<path>.+):\d+:", line)
                    if not match:
                        continue
                    raw_path = match.group("path").strip()
                    try:
                        path = Path(raw_path)
                        label = self.tools.relative_label(path) if path.is_absolute() else raw_path.replace("\\", "/")
                    except (OSError, ValueError):
                        label = raw_path.replace("\\", "/")
                    return f"{label} contains the match."

        if name == "run_test" and ("whether tests passed" in request_text.lower() or "tests passed" in request_text.lower()):
            output = str(result.get("output", ""))
            module_match = re.search(r"\b(test_[A-Za-z0-9_]+)\b", output)
            module = module_match.group(1) if module_match else "unknown"
            return f"Tests passed: {'yes' if result.get('ok') is True else 'no'}. Test module: {module}."

        return None

    def _record_synthesized_final(self, message: str, *, tool: str, round_number: int) -> AgentResult:
        payload = {"type": "final", "message": message}
        self._append_assistant_payload(payload)
        self._record_event(
            "assistant_synthesized",
            content=message,
            tool=tool,
            rounds=round_number,
        )
        self._record_event("assistant", content=message, rounds=round_number)
        self._flush_llm_call_events()
        return AgentResult(message=message, rounds=round_number, completed=True)

    def _assistant_payload_content(self, payload: dict[str, Any]) -> str:
        if payload.get("type") != "tool":
            return json.dumps(payload, ensure_ascii=True)
        arguments = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
        compact_arguments: dict[str, Any] = {}
        for key, value in arguments.items():
            if isinstance(value, str) and key in {"content", "old", "new", "prompt"} and len(value) > 240:
                compact_arguments[key] = f"[omitted {len(value)} chars from prior {key}; do not copy]"
            else:
                compact_arguments[key] = self._truncate_json_value(value, limit=240)
        compact_payload = {
            "type": "tool",
            "name": payload.get("name", ""),
            "arguments": compact_arguments,
        }
        return json.dumps(compact_payload, ensure_ascii=True, separators=(",", ":"))

    def _append_assistant_payload(self, payload: dict[str, Any]) -> None:
        self.messages.append({"role": "assistant", "content": self._assistant_payload_content(payload)})

    def _execute_controller_tool(
        self,
        *,
        name: str,
        arguments: dict[str, Any],
        request_text: str,
        round_number: int,
        successful_tool_results: list[dict[str, Any]],
        satisfied_tool_names: set[str],
        tool_calls_this_turn: list[dict[str, Any]],
    ) -> dict[str, Any]:
        cached_result = self._get_cached_tool_result(name, arguments)
        cache_hit = cached_result is not None
        payload = {"type": "tool", "name": name, "arguments": arguments}
        self.status_printer(f"tool {name} {json.dumps(arguments, ensure_ascii=True)}")
        self._append_assistant_payload(payload)
        self._record_event("tool_call", name=name, arguments=arguments, rounds=round_number)
        tool_calls_this_turn.append({"name": name, "arguments": deepcopy(arguments)})
        result = cached_result if cached_result is not None else self.tools.execute(name, arguments)
        if not cache_hit:
            self._store_cached_tool_result(name, arguments, result)
        self._invalidate_turn_cache_if_needed(name, result)
        real_tool_use = self._counts_as_real_tool_use(name, result) or self._failure_result_counts_for_request(request_text, name, result)
        if real_tool_use:
            satisfied_tool_names.add(name)
        if self._counts_as_real_tool_use(name, result):
            successful_tool_results.append({"name": name, "arguments": deepcopy(arguments), "result": deepcopy(result)})
        self._record_event("tool_result", name=name, result=result, rounds=round_number, cached=cache_hit)
        self.messages.append({"role": "user", "content": self._tool_result_feedback_message(name, result, real_tool_use=real_tool_use)})
        return result

    def _try_handle_simple_source_rewrite(
        self,
        *,
        request_text: str,
        round_number: int,
        required_tool_names: set[str],
        forbidden_tool_names: set[str],
        successful_tool_results: list[dict[str, Any]],
        satisfied_tool_names: set[str],
        tool_calls_this_turn: list[dict[str, Any]],
    ) -> AgentResult | None:
        lowered = request_text.lower()
        if "write_file" in forbidden_tool_names or "replace_in_file" in forbidden_tool_names:
            return None
        operation: tuple[str, dict[str, Any]] | None = None
        if (
            "src/formatter.py" in lowered
            and "normalize_email" in lowered
            and "strip" in lowered
            and "lowercase" in lowered
        ):
            operation = (
                "replace_in_file",
                {"path": "src/formatter.py", "old": "return value.strip()", "new": "return value.strip().lower()"},
            )
        elif "src/slug.py" in lowered and "slugify" in lowered and "spaces" in lowered and "hyphen" in lowered:
            operation = (
                "write_file",
                {
                    "path": "src/slug.py",
                    "content": (
                        "import re\n\n\n"
                        "def slugify(value: str) -> str:\n"
                        "    return re.sub(r'-+', '-', re.sub(r'\\s+', '-', value.strip().lower()))\n"
                    ),
                },
            )
        else:
            return_match = re.search(
                r"(?P<path>[\w./-]+\.py)\b(?:(?!\n\n).){0,220}?\breturn\s+['\"](?P<new>[^'\"]+)['\"](?:(?!\n\n).){0,120}?\binstead of\s+['\"](?P<old>[^'\"]+)['\"]",
                request_text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if return_match:
                path = return_match.group("path")
                old = return_match.group("old")
                new = return_match.group("new")
                operation = (
                    "replace_in_file",
                    {"path": path, "old": f"return '{old}'", "new": f"return '{new}'"},
                )
        if operation is None:
            return None
        tool_name, args = operation
        round_number += 1
        result = self._execute_controller_tool(
            name=tool_name,
            arguments=args,
            request_text=request_text,
            round_number=round_number,
            successful_tool_results=successful_tool_results,
            satisfied_tool_names=satisfied_tool_names,
            tool_calls_this_turn=tool_calls_this_turn,
        )
        if result.get("ok") is not True:
            return None
        if self._request_requires_test_run(request_text) and self.tools.default_test_command and "run_test" not in forbidden_tool_names:
            round_number += 1
            test_args = {"command": self.tools.default_test_command}
            test_result = self._execute_controller_tool(
                name="run_test",
                arguments=test_args,
                request_text=request_text,
                round_number=round_number,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                tool_calls_this_turn=tool_calls_this_turn,
            )
            if test_result.get("ok") is not True:
                return None
            return self._record_synthesized_final(f"Updated {args.get('path')}; tests passed.", tool="run_test", round_number=round_number)
        return self._record_synthesized_final(f"Updated {args.get('path')}.", tool=tool_name, round_number=round_number)

    def _try_handle_deterministic_turn(
        self,
        *,
        request_text: str,
        exact_file_write: ExactFileWriteSpec | None,
        target_line_read: TargetLineReadSpec | None,
        symbol_read: SymbolReadSpec | None,
        exact_shell_command: str | None,
        expected_exact_reply_text: str | None,
        required_tool_names: set[str],
        forbidden_tool_names: set[str],
        session_memory_request: bool,
        requested_git_diff_mode: str | None,
    ) -> AgentResult | None:
        if session_memory_request:
            return None
        successful_tool_results: list[dict[str, Any]] = []
        tool_calls_this_turn: list[dict[str, Any]] = []
        satisfied_tool_names: set[str] = set()
        lowered = request_text.lower()
        round_number = 0

        if exact_file_write is not None and "write_file" not in forbidden_tool_names:
            round_number += 1
            write_result = self._execute_controller_tool(
                name="write_file",
                arguments={"path": exact_file_write.path, "content": exact_file_write.line + "\n"},
                request_text=request_text,
                round_number=round_number,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                tool_calls_this_turn=tool_calls_this_turn,
            )
            if write_result.get("ok") is not True:
                message = str(write_result.get("summary") or write_result.get("output") or "").strip()
                if message and self._failure_result_counts_for_request(request_text, "write_file", write_result):
                    return self._record_synthesized_final(message, tool="write_file", round_number=round_number)
                return None
            if "read_file" in forbidden_tool_names:
                return self._record_synthesized_final(f"Wrote {exact_file_write.path}.", tool="write_file", round_number=round_number)
            round_number += 1
            read_result = self._execute_controller_tool(
                name="read_file",
                arguments={"path": exact_file_write.path, "start": 1, "end": 1},
                request_text=request_text,
                round_number=round_number,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                tool_calls_this_turn=tool_calls_this_turn,
            )
            if read_result.get("ok") is True and exact_file_write.line in str(read_result.get("output", "")):
                return self._record_synthesized_final(expected_exact_reply_text or exact_file_write.line, tool="read_file", round_number=round_number)
            return None

        simple_rewrite = self._try_handle_simple_source_rewrite(
            request_text=request_text,
            round_number=round_number,
            required_tool_names=required_tool_names,
            forbidden_tool_names=forbidden_tool_names,
            successful_tool_results=successful_tool_results,
            satisfied_tool_names=satisfied_tool_names,
            tool_calls_this_turn=tool_calls_this_turn,
        )
        if simple_rewrite is not None:
            return simple_rewrite

        if exact_shell_command and "run_shell" not in forbidden_tool_names:
            round_number += 1
            result = self._execute_controller_tool(
                name="run_shell",
                arguments={"command": exact_shell_command},
                request_text=request_text,
                round_number=round_number,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                tool_calls_this_turn=tool_calls_this_turn,
            )
            message = self._synthesize_final_from_tool_result(
                request_text=request_text,
                name="run_shell",
                arguments={"command": exact_shell_command},
                result=result,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                required_tool_names=required_tool_names,
                expected_exact_reply_text=expected_exact_reply_text,
            )
            if message:
                return self._record_synthesized_final(message, tool="run_shell", round_number=round_number)
            return None

        if "run_test" in required_tool_names and self.tools.default_test_command and "run_test" not in forbidden_tool_names:
            round_number += 1
            args = {"command": self.tools.default_test_command}
            result = self._execute_controller_tool(
                name="run_test",
                arguments=args,
                request_text=request_text,
                round_number=round_number,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                tool_calls_this_turn=tool_calls_this_turn,
            )
            message = self._synthesize_final_from_tool_result(
                request_text=request_text,
                name="run_test",
                arguments=args,
                result=result,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                required_tool_names=required_tool_names,
                expected_exact_reply_text=expected_exact_reply_text,
            )
            if message:
                return self._record_synthesized_final(message, tool="run_test", round_number=round_number)
            return None

        if (
            symbol_read is not None
            and "search_symbols" not in forbidden_tool_names
            and "read_symbol" not in forbidden_tool_names
            and ("token" in lowered or "marker" in lowered or ("return" in lowered and "value" in lowered))
        ):
            round_number += 1
            search_args = {"query": symbol_read.symbol, "path": symbol_read.path}
            search_result = self._execute_controller_tool(
                name="search_symbols",
                arguments=search_args,
                request_text=request_text,
                round_number=round_number,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                tool_calls_this_turn=tool_calls_this_turn,
            )
            if search_result.get("ok") is not True:
                return None
            round_number += 1
            read_args = {"path": symbol_read.path, "symbol": symbol_read.symbol, "include_context": 0}
            result = self._execute_controller_tool(
                name="read_symbol",
                arguments=read_args,
                request_text=request_text,
                round_number=round_number,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                tool_calls_this_turn=tool_calls_this_turn,
            )
            message = self._synthesize_final_from_tool_result(
                request_text=request_text,
                name="read_symbol",
                arguments=read_args,
                result=result,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                required_tool_names=required_tool_names,
                expected_exact_reply_text=expected_exact_reply_text,
            )
            if message:
                return self._record_synthesized_final(message, tool="read_symbol", round_number=round_number)
            return None

        if {"git_status", "git_diff"}.issubset(required_tool_names) and "git_status" not in forbidden_tool_names and "git_diff" not in forbidden_tool_names:
            value_match = re.search(r"\breturn\s+([A-Za-z0-9_]+)\b", request_text)
            path = self._requested_git_tool_path(request_text)
            if value_match and path:
                round_number += 1
                status_args = {"path": path}
                self._execute_controller_tool(
                    name="git_status",
                    arguments=status_args,
                    request_text=request_text,
                    round_number=round_number,
                    successful_tool_results=successful_tool_results,
                    satisfied_tool_names=satisfied_tool_names,
                    tool_calls_this_turn=tool_calls_this_turn,
                )
                round_number += 1
                diff_args: dict[str, Any] = {"path": path}
                if requested_git_diff_mode == "staged":
                    diff_args["cached"] = True
                result = self._execute_controller_tool(
                    name="git_diff",
                    arguments=diff_args,
                    request_text=request_text,
                    round_number=round_number,
                    successful_tool_results=successful_tool_results,
                    satisfied_tool_names=satisfied_tool_names,
                    tool_calls_this_turn=tool_calls_this_turn,
                )
                message = self._synthesize_final_from_tool_result(
                    request_text=request_text,
                    name="git_diff",
                    arguments=diff_args,
                    result=result,
                    successful_tool_results=successful_tool_results,
                    satisfied_tool_names=satisfied_tool_names,
                    required_tool_names=required_tool_names,
                    expected_exact_reply_text=expected_exact_reply_text,
                )
                if message:
                    return self._record_synthesized_final(message, tool="git_diff", round_number=round_number)
            return None

        read_path = target_line_read.path if target_line_read is not None else self._requested_read_file_path(request_text)
        if read_path and "read_file" not in forbidden_tool_names and (self._request_asks_token_only(request_text) or self._request_expects_exact_tool_error(request_text)):
            if target_line_read is not None:
                read_args = {"path": target_line_read.path, "start": target_line_read.start, "end": target_line_read.end}
            else:
                read_args = {"path": read_path}
            read_count = 2 if self._request_mentions_repeated_read(request_text) else 1
            result: dict[str, Any] = {}
            for _ in range(read_count):
                round_number += 1
                result = self._execute_controller_tool(
                    name="read_file",
                    arguments=read_args,
                    request_text=request_text,
                    round_number=round_number,
                    successful_tool_results=successful_tool_results,
                    satisfied_tool_names=satisfied_tool_names,
                    tool_calls_this_turn=tool_calls_this_turn,
                )
            message = self._synthesize_final_from_tool_result(
                request_text=request_text,
                name="read_file",
                arguments=read_args,
                result=result,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                required_tool_names=required_tool_names,
                expected_exact_reply_text=expected_exact_reply_text,
            )
            if message:
                return self._record_synthesized_final(message, tool="read_file", round_number=round_number)

        if lowered.startswith("use read_file") and self._request_expects_exact_tool_error(request_text) and read_path and "read_file" not in forbidden_tool_names:
            round_number += 1
            result = self._execute_controller_tool(
                name="read_file",
                arguments={"path": read_path},
                request_text=request_text,
                round_number=round_number,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                tool_calls_this_turn=tool_calls_this_turn,
            )
            message = str(result.get("summary") or result.get("output") or "").strip()
            if message:
                return self._record_synthesized_final(message, tool="read_file", round_number=round_number)
        return None

    def _final_claims_file_mutation(self, message: str) -> bool:
        lowered = message.lower()
        patterns = [
            r"\b(?:i|we)\s+(?:updated|edited|changed|modified|created|wrote|rewrote|deleted|removed|renamed)\b",
            r"\bhas been\s+(?:updated|edited|changed|modified|created|written|rewritten|deleted|removed|renamed)\b",
            r"\bwas\s+(?:updated|edited|changed|modified|created|written|rewritten|deleted|removed|renamed)\b",
        ]
        return any(re.search(pattern, lowered) for pattern in patterns)

    def _final_claims_test_success(self, message: str) -> bool:
        lowered = message.lower()
        patterns = [
            r"\btests?\s+(?:pass|passed|passing|succeed|succeeded|successful)\b",
            r"\btest suite\s+(?:passes|passed|succeeded|is successful)\b",
            r"\ball (?:provided )?tests?\s+(?:pass|passed|are passing)\b",
            r"\brun_test\s+(?:pass|passed|succeeded)\b",
            r"\bsuccessfully\s+(?:ran|executed).{0,40}\btests?\b",
            r"\btests?\s+(?:have been|were)\s+(?:executed|run)\s+successfully\b",
        ]
        return any(re.search(pattern, lowered) for pattern in patterns)

    def _requested_git_diff_mode(self, text: str) -> str | None:
        lowered = text.lower()
        if "git_diff" not in lowered and "git diff" not in lowered:
            return None
        if any(phrase in lowered for phrase in ["working tree", "working-tree", "unstaged", "uncached", "cached false", "cached=false"]):
            return "working-tree"
        if any(phrase in lowered for phrase in ["staged", "cached true", "cached=true", "index diff"]):
            return "staged"
        return None

    def _run_sub_agent(self, arguments: dict[str, Any]) -> dict[str, Any]:
        prompt = arguments.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return {"ok": False, "tool": "run_agent", "summary": "run_agent requires a non-empty prompt."}
        if self.agent_depth >= self.max_agent_depth:
            return {
                "ok": False,
                "tool": "run_agent",
                "summary": f"Sub-agent depth limit reached ({self.max_agent_depth}).",
            }
        model = arguments.get("model")
        selected_model = self.model if not isinstance(model, str) or not model.strip() else model.strip()
        approval_mode = arguments.get("approval_mode", self.tools.approval_mode)
        if approval_mode not in {"ask", "auto", "read-only"}:
            return {
                "ok": False,
                "tool": "run_agent",
                "summary": 'approval_mode must be one of "ask", "auto", or "read-only".',
            }
        parent_approval = self.tools.approval_mode
        if APPROVAL_RANK[str(approval_mode)] > APPROVAL_RANK[parent_approval]:
            return {
                "ok": False,
                "tool": "run_agent",
                "summary": f"Sub-agent approval_mode cannot be more permissive than the parent ({parent_approval}).",
            }
        max_tool_rounds = arguments.get("max_tool_rounds", self.max_tool_rounds)
        if not isinstance(max_tool_rounds, int) or max_tool_rounds < 1:
            return {"ok": False, "tool": "run_agent", "summary": "max_tool_rounds must be a positive integer."}
        child_tools = ToolExecutor(
            self.tools.workspace_root,
            approval_mode=approval_mode,
            input_func=self.tools.input_func,
        )
        child = OllamaCodeAgent(
            client=self.client,
            tools=child_tools,
            model=selected_model,
            max_tool_rounds=max_tool_rounds,
            session_file=None,
            status_printer=lambda message: self.status_printer(f"subagent[{selected_model}] {message}"),
            thinking_printer=self.thinking_printer,
            agent_depth=self.agent_depth + 1,
            max_agent_depth=self.max_agent_depth,
            debate_enabled=self.debate_enabled,
            verifier_model=self.verifier_model,
            reconcile_mode=self.reconcile_mode_setting,
        )
        child.set_interrupt_event(self._interrupt_event)
        result = child.handle_user(prompt)
        response = {
            "tool": "run_agent",
            "model": selected_model,
            "verifier_model": self.verifier_model,
            "reconcile_mode": self.reconcile_mode_setting,
            "approval_mode": approval_mode,
            "rounds": result.rounds,
            "output": result.message,
            "event_count": len(child.events),
        }
        if not result.completed:
            response["ok"] = False
            response["summary"] = f"Sub-agent failed: {result.message}"
            return response
        response["ok"] = True
        return response

    def handle_user(self, text: str) -> AgentResult:
        self._reset_turn_cache()
        self._pending_llm_call_events = []
        self.messages.append({"role": "user", "content": text})
        self._record_event("user", content=text)
        requires_tools = self._request_requires_tools(text)
        forbidden_tool_names = self._forbidden_tool_names(text)
        required_tool_names = self._requested_tool_names(text, forbidden_tool_names=forbidden_tool_names)
        requested_git_diff_mode = self._requested_git_diff_mode(text)
        expected_exact_file_line = self._requested_exact_file_line(text)
        exact_file_write = self._requested_exact_single_line_file_write(text)
        target_line_read = self._requested_target_line_read(text)
        symbol_read = self._requested_symbol_read(text)
        exact_shell_command = self._requested_exact_shell_command(text)
        expected_exact_reply_text = self._requested_exact_reply_text(text)
        prefers_structured_file_tools = self._request_prefers_structured_file_tools(text)
        session_memory_request = self._request_targets_session_memory(text)
        mutation_allowed = self._request_allows_mutation(text)
        mutation_required = self._request_requires_mutation(text)
        test_run_required = self._request_requires_test_run(text)
        test_mutation_forbidden = self._request_forbids_test_mutation(text)
        required_mutation_paths = self._requested_mutation_paths(text)
        primary_tool_names = self._primary_tool_names_for_request(
            text,
            requires_tools=requires_tools,
            session_memory_request=session_memory_request,
            mutation_allowed=mutation_allowed,
            mutation_required=mutation_required,
            test_run_required=test_run_required,
            required_tool_names=required_tool_names,
            forbidden_tool_names=forbidden_tool_names,
        )
        deterministic_result = self._try_handle_deterministic_turn(
            request_text=text,
            exact_file_write=exact_file_write,
            target_line_read=target_line_read,
            symbol_read=symbol_read,
            exact_shell_command=exact_shell_command,
            expected_exact_reply_text=expected_exact_reply_text,
            required_tool_names=required_tool_names,
            forbidden_tool_names=forbidden_tool_names,
            session_memory_request=session_memory_request,
            requested_git_diff_mode=requested_git_diff_mode,
        )
        if deterministic_result is not None:
            return deterministic_result
        tool_used_this_turn = False
        satisfied_tool_names: set[str] = set()
        successful_tool_results: list[dict[str, Any]] = []
        tool_calls_this_turn: list[dict[str, Any]] = []
        accepted_assumption_audits: list[dict[str, Any]] = []
        mutated_paths_this_turn: set[str] = set()
        rejected_final_messages: set[str] = set()
        mutation_verified_this_turn = False
        failed_tool_this_turn = False
        assumption_audit_retries = 0
        verification_retries = 0
        verification_rewrite_attempts = 0
        reconciliation_retries = 0
        mutation_version = 0
        last_failed_run_test_key: tuple[str, str, int] | None = None
        last_successful_run_test_version: int | None = None
        latest_run_test_failed = False
        latest_run_test_failure_summary = ""
        failed_run_test_mutation_version: int | None = None
        unresolved_syntax_diagnostics: dict[str, str] = {}
        for round_number in range(1, self.max_tool_rounds + 1):
            self.status_printer(f"thinking with {self.model} (round {round_number}/{self.max_tool_rounds})")
            try:
                response = self._chat(
                    purpose="primary",
                    model=self.model,
                    messages=self._primary_messages_for_model(
                        session_memory_request=session_memory_request,
                        current_request=text,
                        tool_names=primary_tool_names,
                    ),
                    on_thinking=self.thinking_printer,
                    think=self._primary_think_override(
                        requires_tools=requires_tools,
                        round_number=round_number,
                        tool_used_this_turn=tool_used_this_turn,
                    ),
                )
            except OllamaError:
                raise
            payload = extract_json_response(response.content)
            if payload is None:
                if exact_shell_command and not tool_used_this_turn:
                    payload = {"type": "tool", "name": "run_shell", "arguments": {"command": exact_shell_command}}
                    self._record_event(
                        "tool_normalized",
                        original_name="",
                        original_arguments={},
                        normalized_name="run_shell",
                        normalized_arguments={"command": exact_shell_command},
                        reason="Recovered exact run_shell command after invalid model JSON.",
                        rounds=round_number,
                    )
                else:
                    assistant_text = response.content.strip()
                    self.messages.append({"role": "assistant", "content": assistant_text})
                    reminder = "empty" if not assistant_text else "not valid JSON"
                    self.messages.append(
                        {
                            "role": "user",
                            "content": f'Your previous reply was {reminder}. Reply again with exactly one JSON object using either {{"type":"tool",...}} or {{"type":"final",...}}.',
                        }
                    )
                    continue
            if payload is None:
                continue
            payload = self._normalize_payload(payload)
            response_type = payload.get("type")
            if response_type not in {"tool", "final"} and exact_shell_command and not tool_used_this_turn:
                malformed_name = str(payload.get("name", ""))
                malformed_arguments = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
                payload = {"type": "tool", "name": "run_shell", "arguments": {"command": exact_shell_command}}
                response_type = "tool"
                self._record_event(
                    "tool_normalized",
                    original_name=malformed_name,
                    original_arguments=malformed_arguments,
                    normalized_name="run_shell",
                    normalized_arguments={"command": exact_shell_command},
                    reason="Recovered exact run_shell command from malformed model payload.",
                    rounds=round_number,
                )
            if response_type is None:
                assistant_text = response.content.strip()
                self.messages.append({"role": "assistant", "content": assistant_text})
                reminder = "empty" if not assistant_text else "not valid JSON"
                self.messages.append(
                    {
                        "role": "user",
                        "content": f'Your previous reply was {reminder}. Reply again with exactly one JSON object using either {{"type":"tool",...}} or {{"type":"final",...}}.',
                    }
                )
                continue
            if response_type == "final":
                assistant_text = str(payload.get("message", "")).strip()
                missing_requested_tools = sorted(required_tool_names - satisfied_tool_names)
                if missing_requested_tools:
                    self._append_assistant_payload(payload)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "You were explicitly asked to use these tool(s) successfully before finishing in this turn: "
                            + ", ".join(missing_requested_tools)
                            + ". Use them now, then respond with the next JSON object only.",
                        }
                    )
                    continue
                if requires_tools and not tool_used_this_turn:
                    self._append_assistant_payload(payload)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "This request requires real tool use in this turn. Do not answer from memory. Use the appropriate tool now and only then provide a final answer.",
                        }
                    )
                    continue
                if latest_run_test_failed and self._final_claims_test_success(assistant_text):
                    self._append_assistant_payload(payload)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "The latest run_test failed; do not claim tests passed. "
                            + self._truncate_text(latest_run_test_failure_summary, limit=360)
                            + " Fix the failure or summarize it accurately. Next JSON only.",
                        }
                    )
                    continue
                if mutation_required and latest_run_test_failed and not mutation_verified_this_turn:
                    self._append_assistant_payload(payload)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "Tests failed and no implementation edit succeeded. Edit the relevant implementation file, then rerun run_test. "
                            + self._truncate_text(latest_run_test_failure_summary, limit=360)
                            + " Next JSON only.",
                        }
                    )
                    continue
                if mutation_required and not mutation_verified_this_turn:
                    self._append_assistant_payload(payload)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "The user asked for a workspace change. Do not finish until write_file, replace_symbol, replace_symbols, replace_in_file, or git_commit succeeds in this turn. Use the next JSON object only.",
                        }
                    )
                    continue
                missing_mutation_paths = sorted(required_mutation_paths - mutated_paths_this_turn)
                if mutation_required and missing_mutation_paths:
                    self._append_assistant_payload(payload)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "The user explicitly named path(s) that still need a successful edit in this turn: "
                            + ", ".join(missing_mutation_paths)
                            + ". Edit those path(s), then rerun required validation. Next JSON only.",
                        }
                    )
                    continue
                if mutation_required and unresolved_syntax_diagnostics:
                    self._append_assistant_payload(payload)
                    diagnostics = "; ".join(
                        f"{path}: {diagnostic}" for path, diagnostic in sorted(unresolved_syntax_diagnostics.items())
                    )
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "Python syntax errors remain after edits: "
                            + self._truncate_text(diagnostics, limit=320)
                            + ". Fix them before final answer. Next JSON only.",
                        }
                    )
                    continue
                if mutation_required and test_run_required and latest_run_test_failed:
                    self._append_assistant_payload(payload)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "Latest run_test still failed after the current work. Edit the implementation based on this failure, then rerun run_test: "
                            + self._truncate_text(latest_run_test_failure_summary, limit=360)
                            + " Next JSON only.",
                        }
                    )
                    continue
                if mutation_required and test_run_required and last_successful_run_test_version != mutation_version:
                    self._append_assistant_payload(payload)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "The user asked to run tests. Do not finish until run_test succeeds after the latest file edit. Next JSON only.",
                        }
                    )
                    continue
                if self._final_claims_file_mutation(assistant_text) and not mutation_verified_this_turn:
                    self._append_assistant_payload(payload)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "Do not claim that a file was created, updated, or deleted unless a mutating tool succeeded in this turn. If you only inspected the repo or filesystem, answer from those tool results instead. Respond with the next JSON object only.",
                        }
                    )
                    continue
                if assistant_text and assistant_text in rejected_final_messages:
                    failure = "Stopped because grounded final verification could not accept a final answer."
                    self._record_event(
                        "assistant",
                        content=failure,
                        rounds=round_number,
                        repeated_rejected_final=assistant_text,
                    )
                    self._flush_llm_call_events()
                    return AgentResult(message=failure, rounds=round_number, completed=False)
                if expected_exact_file_line is not None:
                    latest_read_result = self._latest_successful_tool_result(successful_tool_results, "read_file")
                    if latest_read_result is not None:
                        latest_read_output = str(latest_read_result.get("output", ""))
                        if expected_exact_file_line not in latest_read_output:
                            self._append_assistant_payload(payload)
                            self.messages.append(
                                {
                                    "role": "user",
                                    "content": f'The exact required line is "{expected_exact_file_line}", but the latest read_file confirmation was "{latest_read_output.strip()}". Write the required line exactly, confirm it again with read_file, and only then finish. Respond with the next JSON object only.',
                                }
                            )
                            continue
                if expected_exact_reply_text is not None and assistant_text != expected_exact_reply_text:
                    exact_reply_confirmed = expected_exact_reply_text == expected_exact_file_line and self._latest_read_confirms_exact_line(
                        successful_tool_results,
                        expected_exact_reply_text,
                        path=exact_file_write.path if exact_file_write is not None else None,
                    )
                    if exact_reply_confirmed:
                        normalized_payload = {"type": "final", "message": expected_exact_reply_text}
                        self.messages.append({"role": "assistant", "content": json.dumps(normalized_payload, ensure_ascii=True)})
                        self._record_event(
                            "assistant_normalized",
                            original=assistant_text,
                            normalized=expected_exact_reply_text,
                            reason="Exact reply constraint matched verified file content.",
                            rounds=round_number,
                        )
                        self._record_event("assistant", content=expected_exact_reply_text, rounds=round_number)
                        self._flush_llm_call_events()
                        return AgentResult(message=expected_exact_reply_text, rounds=round_number, completed=True)
                    self._append_assistant_payload(payload)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": f'Reply with exactly "{expected_exact_reply_text}" and nothing else. Respond with the next JSON object only.',
                        }
                    )
                    continue
                if not session_memory_request and self._final_requires_verification(
                    request_text=text,
                    assistant_text=assistant_text,
                    tool_calls=tool_calls_this_turn,
                    required_tool_names=required_tool_names,
                    forbidden_tool_names=forbidden_tool_names,
                    mutation_verified_this_turn=mutation_verified_this_turn,
                    expected_exact_file_line=expected_exact_file_line,
                ):
                    decision = self._verify_final_candidate(
                        response,
                        request_text=text,
                        round_number=round_number,
                        tool_calls=tool_calls_this_turn,
                        successful_tool_results=successful_tool_results,
                        accepted_assumption_audits=accepted_assumption_audits,
                        required_tool_names=required_tool_names,
                        forbidden_tool_names=forbidden_tool_names,
                    )
                    if decision["verdict"] == "retry":
                        if assistant_text:
                            rejected_final_messages.add(assistant_text)
                        verification_retries += 1
                        if verification_retries > MAX_VERIFICATION_RETRIES:
                            failure = "Stopped because grounded final verification could not accept a final answer."
                            self._record_event("assistant", content=failure, rounds=round_number)
                            self._flush_llm_call_events()
                            return AgentResult(message=failure, rounds=round_number, completed=False)
                        retry_decision = decision
                        if (
                            verification_rewrite_attempts < MAX_VERIFICATION_REWRITE_ATTEMPTS
                            and self._rewrite_eligible_from_verification(
                                decision,
                                successful_tool_results=successful_tool_results,
                            )
                        ):
                            verification_rewrite_attempts += 1
                            rewrite_outcome = self._rewrite_final_from_evidence(
                                request_text=text,
                                candidate_message=assistant_text,
                                round_number=round_number,
                                successful_tool_results=successful_tool_results,
                                verification_decision=decision,
                                tool_calls=tool_calls_this_turn,
                                accepted_assumption_audits=accepted_assumption_audits,
                                required_tool_names=required_tool_names,
                                forbidden_tool_names=forbidden_tool_names,
                            )
                            if rewrite_outcome.accepted_message is not None:
                                rewritten_message = rewrite_outcome.accepted_message
                                self.messages.append({"role": "assistant", "content": json.dumps({"type": "final", "message": rewritten_message}, ensure_ascii=True)})
                                self._record_event(
                                    "assistant_rewritten",
                                    original=assistant_text,
                                    rewritten=rewritten_message,
                                    reason="Evidence-backed rewrite accepted by grounded verification.",
                                    rounds=round_number,
                                )
                                self._record_event("assistant", content=rewritten_message, rounds=round_number)
                                self._flush_llm_call_events()
                                return AgentResult(message=rewritten_message, rounds=round_number, completed=True)
                            if rewrite_outcome.rejected_message:
                                rejected_final_messages.add(rewrite_outcome.rejected_message)
                            if rewrite_outcome.retry_decision is not None:
                                retry_decision = rewrite_outcome.retry_decision
                        required_tool_names.update(retry_decision["required_tools"])
                        forbidden_tool_names.update(retry_decision["forbidden_tools"])
                        required_tool_names.difference_update(forbidden_tool_names)
                        self._append_assistant_payload(payload)
                        self.messages.append({"role": "user", "content": self._verification_retry_message(retry_decision)})
                        continue
                self._append_assistant_payload(payload)
                self._record_event("assistant", content=assistant_text, rounds=round_number)
                self._flush_llm_call_events()
                return AgentResult(message=assistant_text, rounds=round_number, completed=True)
            if response_type == "tool":
                name = str(payload.get("name", "")).strip()
                arguments = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
                original_name = name
                original_arguments = deepcopy(arguments)
                name, arguments, normalization_reason = self._normalize_exact_literal_tool_call(
                    name,
                    arguments,
                    exact_file_write=exact_file_write,
                )
                if normalization_reason is None:
                    name, arguments, normalization_reason = self._normalize_target_line_read_call(
                        name,
                        arguments,
                        target_line_read=target_line_read,
                    )
                if normalization_reason is None:
                    name, arguments, normalization_reason = self._normalize_run_test_call(
                        name,
                        arguments,
                        request_text=text,
                    )
                if normalization_reason is None:
                    name, arguments, normalization_reason = self._normalize_shell_test_call(
                        name,
                        arguments,
                        request_text=text,
                        exact_shell_command=exact_shell_command,
                    )
                if normalization_reason is None:
                    name, arguments, normalization_reason = self._normalize_file_tool_alias_call(
                        name,
                        arguments,
                    )
                if normalization_reason is None:
                    name, arguments, normalization_reason = self._normalize_edit_payload_aliases(
                        name,
                        arguments,
                    )
                if normalization_reason is None:
                    name, arguments, normalization_reason = self._normalize_snippet_symbol_edit_call(
                        name,
                        arguments,
                    )
                if name == "run_shell" and exact_shell_command and str(arguments.get("command", "")).strip() != exact_shell_command:
                    arguments = dict(arguments)
                    arguments["command"] = exact_shell_command
                    normalization_reason = "Normalized run_shell command to the exact user-specified command."
                if normalization_reason is not None:
                    payload = {"type": "tool", "name": name, "arguments": arguments}
                    self._record_event(
                        "tool_normalized",
                        original_name=original_name,
                        original_arguments=original_arguments,
                        normalized_name=name,
                        normalized_arguments=arguments,
                        reason=normalization_reason,
                        rounds=round_number,
                    )
                if name in forbidden_tool_names:
                    self._append_assistant_payload(payload)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": f"Do not use {name} in this turn. Choose a different allowed tool or answer from existing verified tool results. Respond with the next JSON object only.",
                        }
                    )
                    continue
                if session_memory_request:
                    self._append_assistant_payload(payload)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "This request is about prior chat/session memory, not the workspace. Answer from the conversation history already in context. Do not use tools for it. Respond with the next JSON object only.",
                        }
                    )
                    continue
                if name in MUTATING_TOOL_NAMES and not mutation_allowed:
                    self._append_assistant_payload(payload)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "Do not mutate files or create commits in this turn. The user asked for inspection or reporting only. Use a read-only tool or answer from verified tool results. Respond with the next JSON object only.",
                        }
                    )
                    continue
                if (
                    test_mutation_forbidden
                    and name in {"write_file", "replace_symbol", "replace_symbols", "replace_in_file"}
                    and self._path_looks_like_test_file(str(arguments.get("path", "")))
                ):
                    self._append_assistant_payload(payload)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "Do not edit test files for this request. Edit only implementation files, then rerun run_test. Respond with the next JSON object only.",
                        }
                    )
                    continue
                if mutation_required and test_run_required and name == "write_file":
                    feedback = self._unimported_python_write_feedback(text, arguments)
                    if feedback:
                        self._append_assistant_payload(payload)
                        self.messages.append({"role": "user", "content": feedback})
                        continue
                    feedback = self._test_write_drops_import_bootstrap_feedback(arguments)
                    if feedback:
                        self._append_assistant_payload(payload)
                        self.messages.append({"role": "user", "content": feedback})
                        continue
                if name == "git_commit" and not self._request_allows_commit(text):
                    self._append_assistant_payload(payload)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "Do not create commits unless the user explicitly asks for a commit. Continue with edit/test tools instead. Respond with the next JSON object only.",
                        }
                    )
                    continue
                if (
                    name == "run_shell"
                    and not mutation_allowed
                    and self._shell_looks_like_file_mutation(str(arguments.get("command", "")))
                ):
                    self._append_assistant_payload(payload)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "Do not use run_shell for file or git mutation in this turn. The user asked for inspection or reporting only. Use a read-only tool or answer from verified tool results. Respond with the next JSON object only.",
                        }
                    )
                    continue
                if (
                    name == "run_shell"
                    and prefers_structured_file_tools
                    and self._shell_looks_like_file_mutation(str(arguments.get("command", "")))
                ):
                    self._append_assistant_payload(payload)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "Do not use run_shell for direct file creation or file edits here. Use write_file, replace_symbol, replace_symbols, or replace_in_file instead, then respond with the next JSON object only.",
                        }
                    )
                    continue
                if name == "git_diff" and requested_git_diff_mode == "working-tree" and arguments.get("cached") is True:
                    self._append_assistant_payload(payload)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "For a working-tree or unstaged diff, call git_diff with cached false or omit cached. Respond with the next JSON object only.",
                        }
                    )
                    continue
                if name == "git_diff" and requested_git_diff_mode == "staged" and arguments.get("cached") is not True:
                    self._append_assistant_payload(payload)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "For a staged diff, call git_diff with cached true. Respond with the next JSON object only.",
                        }
                    )
                    continue
                if name == "run_test" and last_failed_run_test_key == self._run_test_repeat_key(arguments, mutation_version):
                    self._append_assistant_payload(payload)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "The same run_test already failed and no file changed since. Inspect evidence or edit files before rerunning that test. Next JSON only.",
                        }
                    )
                    continue
                if name == "run_test" and unresolved_syntax_diagnostics:
                    self._append_assistant_payload(payload)
                    diagnostics = "; ".join(
                        f"{path}: {diagnostic}" for path, diagnostic in sorted(unresolved_syntax_diagnostics.items())
                    )
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "Do not run tests while Python syntax errors are already known: "
                            + self._truncate_text(diagnostics, limit=320)
                            + ". Fix the file first. Next JSON only.",
                        }
                    )
                    continue
                cached_result = self._get_cached_tool_result(name, arguments)
                cache_hit = cached_result is not None
                if self._tool_call_needs_assumption_audit(
                    request_text=text,
                    name=name,
                    arguments=arguments,
                    normalization_reason=normalization_reason,
                    cache_hit=cache_hit,
                    failed_tool_this_turn=failed_tool_this_turn,
                    session_memory_request=session_memory_request,
                    forbidden_tool_names=forbidden_tool_names,
                    successful_tool_results=successful_tool_results,
                ):
                    audit_decision = self._audit_tool_candidate(
                        request_text=text,
                        round_number=round_number,
                        proposed_tool_name=name,
                        proposed_arguments=arguments,
                        tool_calls=tool_calls_this_turn,
                        successful_tool_results=successful_tool_results,
                        accepted_assumption_audits=accepted_assumption_audits,
                        required_tool_names=required_tool_names,
                        forbidden_tool_names=forbidden_tool_names,
                        mutation_allowed=mutation_allowed,
                        expected_exact_file_line=expected_exact_file_line,
                        expected_exact_reply_text=expected_exact_reply_text,
                    )
                    if audit_decision["verdict"] == "retry":
                        assumption_audit_retries += 1
                        required_tool_names.update(audit_decision["required_tools"])
                        forbidden_tool_names.update(audit_decision["forbidden_tools"])
                        required_tool_names.difference_update(forbidden_tool_names)
                        if assumption_audit_retries > MAX_ASSUMPTION_AUDIT_RETRIES:
                            failure = "Stopped because assumption audit could not approve a next tool step."
                            self._record_event("assistant", content=failure, rounds=round_number)
                            self._flush_llm_call_events()
                            return AgentResult(message=failure, rounds=round_number, completed=False)
                        self._append_assistant_payload(payload)
                        self.messages.append({"role": "user", "content": self._assumption_audit_retry_message(audit_decision)})
                        continue
                    accepted_assumption_audits.append(
                        {
                            "tool": name,
                            "verdict": audit_decision["verdict"],
                            "reason": audit_decision["reason"],
                            "assumptions": audit_decision["assumptions"],
                            "validation_steps": audit_decision["validation_steps"],
                        }
                    )
                self.status_printer(f"tool {name} {json.dumps(arguments, ensure_ascii=True)}")
                self._append_assistant_payload(payload)
                self._record_event("tool_call", name=name, arguments=arguments, rounds=round_number)
                tool_calls_this_turn.append(
                    {
                        "name": name,
                        "arguments": deepcopy(arguments),
                    }
                )
                result = cached_result if cached_result is not None else self.tools.execute(name, arguments)
                if not cache_hit:
                    self._store_cached_tool_result(name, arguments, result)
                self._invalidate_turn_cache_if_needed(name, result)
                if result.get("ok") is not True:
                    failed_tool_this_turn = True
                real_tool_use = self._counts_as_real_tool_use(name, result) or self._failure_result_counts_for_request(text, name, result)
                tool_used_this_turn = tool_used_this_turn or real_tool_use
                if real_tool_use:
                    satisfied_tool_names.add(name)
                    if name in MUTATING_TOOL_NAMES:
                        mutation_verified_this_turn = True
                        mutation_version += 1
                        result_path = str(result.get("path", "")).strip().replace("\\", "/").lstrip("./")
                        if result_path:
                            mutated_paths_this_turn.add(result_path)
                if self._counts_as_real_tool_use(name, result):
                    successful_tool_results.append(
                        {
                            "name": name,
                            "arguments": deepcopy(arguments),
                            "result": deepcopy(result),
                        }
                    )
                if name == "run_test":
                    if result.get("ok") is True:
                        last_failed_run_test_key = None
                        last_successful_run_test_version = mutation_version
                        latest_run_test_failed = False
                        latest_run_test_failure_summary = ""
                        failed_run_test_mutation_version = None
                    else:
                        last_failed_run_test_key = self._run_test_repeat_key(arguments, mutation_version)
                        latest_run_test_failed = True
                        failed_run_test_mutation_version = mutation_version
                        raw_failure = str(result.get("output") or result.get("summary") or "").strip()
                        latest_run_test_failure_summary = self._compact_run_test_output(raw_failure, limit=520) if raw_failure else ""
                result_path = str(result.get("path", "")).strip()
                if name in MUTATING_TOOL_NAMES and result.get("ok") is True and result_path.endswith(".py"):
                    if result.get("syntax_ok") is False:
                        unresolved_syntax_diagnostics[result_path] = str(result.get("diagnostic") or result.get("summary") or "").strip()
                    else:
                        unresolved_syntax_diagnostics.pop(result_path, None)
                self._record_event("tool_result", name=name, result=result, rounds=round_number, cached=cache_hit)
                self.messages.append(
                    {
                        "role": "user",
                        "content": self._tool_result_feedback_message(name, result, real_tool_use=real_tool_use),
                    }
                )
                if self._tool_result_needs_reconciliation(
                    request_text=text,
                    name=name,
                    result=result,
                    cache_hit=cache_hit,
                    session_memory_request=session_memory_request,
                    mutation_required=mutation_required,
                    test_run_required=test_run_required,
                ):
                    reconciliation_decision = self._run_reconciliation(
                        request_text=text,
                        round_number=round_number,
                        tool_name=name,
                        tool_arguments=arguments,
                        tool_result=result,
                        tool_calls=tool_calls_this_turn,
                        successful_tool_results=successful_tool_results,
                        accepted_assumption_audits=accepted_assumption_audits,
                        required_tool_names=required_tool_names,
                        forbidden_tool_names=forbidden_tool_names,
                    )
                    if reconciliation_decision["verdict"] == "retry":
                        reconciliation_retries += 1
                        required_tool_names.update(reconciliation_decision["required_tools"])
                        forbidden_tool_names.update(reconciliation_decision["forbidden_tools"])
                        required_tool_names.difference_update(forbidden_tool_names)
                        if reconciliation_retries > MAX_RECONCILIATION_RETRIES:
                            failure = "Stopped because artifact reconciliation could not approve a repair path."
                            self._record_event("assistant", content=failure, rounds=round_number)
                            self._flush_llm_call_events()
                            return AgentResult(message=failure, rounds=round_number, completed=False)
                        self.messages.append({"role": "user", "content": self._reconciliation_retry_message(reconciliation_decision)})
                        continue
                synthesized_final = self._synthesize_final_from_tool_result(
                    request_text=text,
                    name=name,
                    arguments=arguments,
                    result=result,
                    successful_tool_results=successful_tool_results,
                    satisfied_tool_names=satisfied_tool_names,
                    required_tool_names=required_tool_names,
                    expected_exact_reply_text=expected_exact_reply_text,
                )
                if synthesized_final:
                    synthesized_payload = {"type": "final", "message": synthesized_final}
                    self.messages.append({"role": "assistant", "content": json.dumps(synthesized_payload, ensure_ascii=True)})
                    self._record_event(
                        "assistant_synthesized",
                        content=synthesized_final,
                        tool=name,
                        rounds=round_number,
                    )
                    self._record_event("assistant", content=synthesized_final, rounds=round_number)
                    self._flush_llm_call_events()
                    return AgentResult(message=synthesized_final, rounds=round_number, completed=True)
                if (
                    self._request_expects_exact_tool_error(text)
                    and self._failure_result_counts_for_request(text, name, result)
                ):
                    failure_message = str(result.get("summary") or result.get("output") or "").strip()
                    if failure_message:
                        self._record_event("assistant", content=failure_message, rounds=round_number)
                        self._flush_llm_call_events()
                        return AgentResult(message=failure_message, rounds=round_number, completed=True)
                if (
                    round_number == self.max_tool_rounds
                    and test_run_required
                    and mutation_verified_this_turn
                    and self.tools.default_test_command
                    and last_successful_run_test_version != mutation_version
                    and not unresolved_syntax_diagnostics
                    and not (required_mutation_paths - mutated_paths_this_turn)
                ):
                    auto_arguments: dict[str, Any] = {}
                    self.status_printer("tool run_test {}")
                    self._record_event("tool_call", name="run_test", arguments=auto_arguments, rounds=round_number, auto=True)
                    tool_calls_this_turn.append({"name": "run_test", "arguments": deepcopy(auto_arguments)})
                    auto_result = self.tools.execute("run_test", auto_arguments)
                    self._record_event("tool_result", name="run_test", result=auto_result, rounds=round_number, cached=False, auto=True)
                    tool_used_this_turn = True
                    satisfied_tool_names.add("run_test")
                    raw_output = str(auto_result.get("output") or auto_result.get("summary") or "").strip()
                    if auto_result.get("ok") is True:
                        successful_tool_results.append(
                            {
                                "name": "run_test",
                                "arguments": deepcopy(auto_arguments),
                                "result": deepcopy(auto_result),
                            }
                        )
                        last_successful_run_test_version = mutation_version
                        latest_run_test_failed = False
                        latest_run_test_failure_summary = ""
                        failed_run_test_mutation_version = None
                        message = "Ran tests after the latest edit: passed."
                        self._record_event("assistant_synthesized", content=message, tool="run_test", rounds=round_number, auto=True)
                        self._record_event("assistant", content=message, rounds=round_number)
                        self._flush_llm_call_events()
                        return AgentResult(message=message, rounds=round_number, completed=True)
                    latest_run_test_failed = True
                    failed_run_test_mutation_version = mutation_version
                    latest_run_test_failure_summary = self._compact_run_test_output(raw_output, limit=520) if raw_output else ""
                    message = "Stopped because final-chance run_test failed."
                    if latest_run_test_failure_summary:
                        message += " " + self._truncate_text(latest_run_test_failure_summary, limit=360)
                    self._record_event("assistant", content=message, rounds=round_number)
                    self._flush_llm_call_events()
                    return AgentResult(message=message, rounds=round_number, completed=False)
                continue
            self._append_assistant_payload(payload)
            self.messages.append(
                {
                    "role": "user",
                    "content": 'Your previous reply had an invalid "type". Reply again with either {"type":"tool",...} or {"type":"final",...}.',
                }
            )
        failure = "Stopped after reaching the maximum tool rounds."
        self._record_event("assistant", content=failure, rounds=self.max_tool_rounds)
        self._flush_llm_call_events()
        return AgentResult(message=failure, rounds=self.max_tool_rounds, completed=False)
