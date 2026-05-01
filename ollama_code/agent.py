from __future__ import annotations

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
from ollama_code.tools import TOOL_DESCRIPTIONS, ToolExecutor, format_tool_help


SYSTEM_PROMPT_TEMPLATE = """You are Ollama Code, a local coding assistant running in a terminal.

Workspace root: {workspace_root}

You can inspect and modify files in that workspace with the provided tools.
Return exactly one JSON object and nothing else.

Valid response shapes:
{{"type":"tool","name":"read_file","arguments":{{"path":"README.md"}}}}
{{"type":"final","message":"Your final answer to the user."}}

Rules:
- Use at most one tool call per response.
- Prefer inspecting files before editing them.
- For broad repo questions, prefer search or list_files before read_file.
- When you use read_file, request the narrowest useful start/end range instead of rereading large files.
- Reuse recent tool results already in the conversation when they still answer the question. Avoid repeating identical read-only tools unless state changed.
- Question your assumptions before acting.
- Identify what you are assuming, then prove or disprove it with the available tools whenever a file read, search, git inspection, test run, or shell command can verify it.
- Do not guess about workspace contents, command output, repo state, or whether an edit worked when you can check instead.
- Default reply style: caveman-lite. Be terse and information-dense. Drop filler, pleasantries, hedging, and redundant transitions.
- Prefer short sentences or fragments in natural-language replies, but keep all technical terms, code, file paths, commands, errors, and JSON exact.
- Do not let terse style reduce investigation depth, verification, correctness, or willingness to use tools.
- Use normal high-clarity prose instead of caveman-lite when handling warnings, irreversible actions, security-sensitive advice, or any case where extra clarity prevents mistakes.
- These style rules apply to natural-language content only. Tool arguments, JSON wrappers, code, diffs, and commands must stay syntactically correct and complete.
- Use relative workspace paths.
- Keep final answers concise and practical.
- Do not emit markdown fences.
- Do not emit chain-of-thought.
- If the user asks about filesystem state, search results, command output, file edits, or helper agents, you must use the relevant tool instead of guessing.
- Never claim a file was changed, a command was run, or a helper agent replied unless you have a successful tool result for it in the current turn.
- For new files use write_file. For edits use replace_in_file or write_file. For shell work use run_shell. For project tests prefer run_test when available. For delegated work use run_agent.
- For git inspection prefer git_status and git_diff. For git commits use git_commit.

Available tools:
{tool_help}

Example when you need a helper agent:
{{"type":"tool","name":"run_agent","arguments":{{"prompt":"Read README.md and summarize setup steps.","approval_mode":"read-only"}}}}
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


KNOWN_TOOL_NAMES = {tool["name"] for tool in TOOL_DESCRIPTIONS}
APPROVAL_RANK = {"read-only": 0, "ask": 1, "auto": 2}
VERIFICATION_HISTORY_LIMIT = 4
VERIFICATION_CONTENT_LIMIT = 6000
MAX_VERIFICATION_RETRIES = 2
MAX_ASSUMPTION_AUDIT_RETRIES = 2
AUDIT_LIST_ITEM_LIMIT = 3
AUDIT_TEXT_ITEM_LIMIT = 180
MUTATING_TOOL_NAMES = {"write_file", "replace_in_file", "git_commit"}
READ_ONLY_CACHEABLE_TOOL_NAMES = {"list_files", "read_file", "search", "git_status", "git_diff"}
RISKY_VERIFICATION_TOOL_NAMES = {"search", "git_status", "git_diff", "run_shell", "run_test", "run_agent"}
MODEL_TOOL_RESULT_LIMITS = {
    "list_files": 1200,
    "read_file": 2200,
    "search": 1600,
    "git_status": 1000,
    "git_diff": 1800,
    "run_shell": 1400,
    "run_test": 1400,
    "run_agent": 1200,
}
MODEL_TOOL_DIFF_LIMIT = 1400
VERIFICATION_TOOL_RESULT_LIMIT = 1200
VERIFICATION_TOOL_DIFF_LIMIT = 1000

FINAL_VERIFIER_SYSTEM_PROMPT = """You are a grounded final verifier for a coding CLI controller.

You will receive the original user request, the candidate final answer, current-turn tool calls, current-turn successful tool results, accepted tool-step assumption audits, and explicit required/forbidden tool constraints.
Return exactly one JSON object and nothing else.

Valid replies:
{"verdict":"accept"}
{"verdict":"retry","reason":"brief concrete reason","required_tools":["read_file"],"forbidden_tools":["run_shell"]}

Rules:
- Check whether the candidate final answer should be accepted as grounded and compliant.
- Prefer accept when the candidate already matches the tool results and request.
- Return retry if the candidate contradicts tool results, ignores required tools, violates forbidden-tool constraints, hallucinates workspace state, or needs another tool/result first.
- Consider accepted assumption-audit events as additional grounding context. Return retry if the candidate leaves those assumptions unresolved or contradicts the evidence they gathered.
- Do not rewrite the final answer yourself.
- Keep the reason brief and concrete.
- required_tools and forbidden_tools must be arrays of known tool names or empty arrays.
"""

TOOL_ASSUMPTION_AUDITOR_SYSTEM_PROMPT = """You are a tool-step assumption auditor for a coding CLI controller.

You will receive the original user request, recent messages, a proposed tool call, current-turn tool calls, current-turn successful tool results, accepted tool-step assumption audits, and explicit required/forbidden tool constraints.
Return exactly one JSON object and nothing else.

Valid replies:
{"verdict":"accept","reason":"","assumptions":["..."],"validation_steps":["..."],"required_tools":[],"forbidden_tools":[]}
{"verdict":"retry","reason":"brief concrete reason","assumptions":["..."],"validation_steps":["..."],"required_tools":["read_file"],"forbidden_tools":["run_shell"]}

Rules:
- Check whether the proposed tool call is a grounded next step.
- assumptions must be short concrete assumptions that the tool relies on or tests.
- validation_steps must be short concrete descriptions of how the tool call validates those assumptions or gathers the needed evidence.
- Prefer accept when the proposed tool is a reasonable next validation step, even if the tool may fail and the user explicitly asked for that exact tool error or boundary failure.
- Return retry if the tool is redundant, too broad, violates required/forbidden-tool constraints, mutates when inspection should happen first, skips a needed validation step, or fails to test the key assumption behind the next step.
- Do not rewrite the tool call yourself.
- Keep reason brief and concrete.
- required_tools and forbidden_tools must be arrays of known tool names or empty arrays.
"""


def extract_json_response(raw_text: str) -> dict[str, Any] | None:
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
    starts = [index for index, char in enumerate(candidate) if char == "{"]
    if candidate.startswith("{"):
        starts = [0] + [index for index in starts if index != 0]
    parsed_dicts: list[dict[str, Any]] = []
    agent_payloads: list[dict[str, Any]] = []
    for start in starts:
        try:
            data, _ = decoder.raw_decode(candidate[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
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
    ) -> None:
        self.client = client
        self.tools = tools
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        self.session_file = self._resolve_transcript_path(session_file) if session_file else None
        self.status_printer = status_printer or (lambda message: None)
        self.thinking_printer = thinking_printer
        self.agent_depth = agent_depth
        self.max_agent_depth = max_agent_depth
        self.debate_enabled = debate_enabled
        self.events: list[dict[str, Any]] = []
        self._interrupt_event: threading.Event | None = None
        self._turn_tool_cache: dict[tuple[str, str, int], dict[str, Any]] = {}
        self._turn_cache_epoch = 0
        self.tools.agent_runner = self._run_sub_agent
        self.messages = self._base_messages()

    def _base_messages(self) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_TEMPLATE.format(
                    workspace_root=self.tools.workspace_root.as_posix(),
                    tool_help=format_tool_help(),
                ),
            }
        ]

    def reset(self) -> None:
        self.messages = self._base_messages()
        self.events = []
        self._reset_turn_cache()
        self._autosave()

    def set_model(self, model: str) -> None:
        self.model = model

    def set_approval_mode(self, mode: str) -> None:
        self.tools.set_approval_mode(mode)

    def approval_mode(self) -> str:
        return self.tools.approval_mode

    def debate_mode(self) -> bool:
        return self.debate_enabled

    def set_debate_enabled(self, enabled: bool) -> None:
        self.debate_enabled = bool(enabled)

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
                "content": self._truncate_text(message["content"], limit=800),
            }
            for message in self.messages[-VERIFICATION_HISTORY_LIMIT:]
            if message.get("role") != "system"
        ]
        return {
            "round": round_number,
            "model": self.model,
            "workspace_root": self.tools.workspace_root.as_posix(),
            "original_user_request": self._truncate_text(request_text),
            "recent_messages": recent_messages,
            "candidate_final_answer": self._truncate_text(candidate_message),
            "required_tools": sorted(required_tool_names),
            "forbidden_tools": sorted(forbidden_tool_names),
            "tool_calls": [self._compact_tool_call_for_verification(item) for item in tool_calls],
            "successful_tool_results": [self._compact_successful_tool_result_for_verification(item) for item in successful_tool_results],
            "accepted_assumption_audits": [self._compact_assumption_audit_for_context(item) for item in accepted_assumption_audits],
        }

    def _verification_messages(self, payload: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": FINAL_VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=2, ensure_ascii=True)},
        ]

    def _normalize_verification_payload(self, payload: dict[str, Any] | None) -> dict[str, Any]:
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
            "required_tools": sorted(set(required_tools)),
            "forbidden_tools": sorted(set(forbidden_tools)),
        }

    def _assumption_audit_messages(self, payload: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": TOOL_ASSUMPTION_AUDITOR_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=2, ensure_ascii=True)},
        ]

    def _compact_assumption_audit_for_context(self, audit: dict[str, Any]) -> dict[str, Any]:
        return {
            "tool": str(audit.get("tool", "")).strip(),
            "verdict": str(audit.get("verdict", "")).strip(),
            "reason": self._truncate_text(str(audit.get("reason", "")).strip(), limit=240),
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
                "content": self._truncate_text(message["content"], limit=800),
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
            "original_user_request": self._truncate_text(request_text),
            "recent_messages": recent_messages,
            "proposed_tool": {
                "name": proposed_tool_name,
                "arguments": self._truncate_json_value(proposed_arguments, limit=800),
            },
            "required_tools": sorted(required_tool_names),
            "forbidden_tools": sorted(forbidden_tool_names),
            "mutation_allowed": mutation_allowed,
            "exact_constraints": exact_constraints,
            "tool_calls": [self._compact_tool_call_for_verification(item) for item in tool_calls],
            "successful_tool_results": [self._compact_successful_tool_result_for_verification(item) for item in successful_tool_results],
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
        verdict_response = self.client.chat(
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
            auditor=verdict_response.content,
        )
        return decision

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
            verdict_response = self.client.chat(
                model=self.model,
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
            "cwd",
            "start",
            "end",
            "count",
            "cached",
            "exit_code",
            "command",
            "summary",
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
            payload[field] = self._truncate_text(value, limit=limit)
        return payload

    def _compact_tool_call_for_verification(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        name = str(tool_call.get("name", "")).strip()
        arguments = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
        return {
            "name": name,
            "arguments": self._truncate_json_value(arguments, limit=600),
        }

    def _compact_successful_tool_result_for_verification(self, item: dict[str, Any]) -> dict[str, Any]:
        name = str(item.get("name", "")).strip()
        arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        return {
            "name": name,
            "arguments": self._truncate_json_value(arguments, limit=600),
            "result": self._compact_tool_result_for_context(name, result, for_verification=True),
        }

    def _tool_result_feedback_message(self, name: str, result: dict[str, Any], *, real_tool_use: bool) -> str:
        payload = self._compact_tool_result_for_context(name, result, for_verification=False)
        follow_up = "Respond with the next JSON object only."
        if not real_tool_use:
            follow_up = (
                "That tool call did not complete successfully, so it does not satisfy the tool-use requirement. "
                "Fix the issue or use a different appropriate tool, then respond with the next JSON object only."
            )
        return "Tool result summary:\n" + json.dumps(payload, indent=2, ensure_ascii=True) + "\n" + follow_up

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
        summary = str(result.get("summary", "")).strip()
        output = str(result.get("output", "")).strip()
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

    def _final_claims_file_mutation(self, message: str) -> bool:
        lowered = message.lower()
        patterns = [
            r"\b(?:i|we)\s+(?:updated|edited|changed|modified|created|wrote|rewrote|deleted|removed|renamed)\b",
            r"\bhas been\s+(?:updated|edited|changed|modified|created|written|rewritten|deleted|removed|renamed)\b",
            r"\bwas\s+(?:updated|edited|changed|modified|created|written|rewritten|deleted|removed|renamed)\b",
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
        )
        child.set_interrupt_event(self._interrupt_event)
        result = child.handle_user(prompt)
        response = {
            "tool": "run_agent",
            "model": selected_model,
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
        self.messages.append({"role": "user", "content": text})
        self._record_event("user", content=text)
        requires_tools = self._request_requires_tools(text)
        forbidden_tool_names = self._forbidden_tool_names(text)
        required_tool_names = self._requested_tool_names(text, forbidden_tool_names=forbidden_tool_names)
        requested_git_diff_mode = self._requested_git_diff_mode(text)
        expected_exact_file_line = self._requested_exact_file_line(text)
        exact_file_write = self._requested_exact_single_line_file_write(text)
        expected_exact_reply_text = self._requested_exact_reply_text(text)
        prefers_structured_file_tools = self._request_prefers_structured_file_tools(text)
        session_memory_request = self._request_targets_session_memory(text)
        mutation_allowed = self._request_allows_mutation(text)
        tool_used_this_turn = False
        satisfied_tool_names: set[str] = set()
        successful_tool_results: list[dict[str, Any]] = []
        tool_calls_this_turn: list[dict[str, Any]] = []
        accepted_assumption_audits: list[dict[str, Any]] = []
        rejected_final_messages: set[str] = set()
        mutation_verified_this_turn = False
        assumption_audit_retries = 0
        verification_retries = 0
        for round_number in range(1, self.max_tool_rounds + 1):
            self.status_printer(f"thinking with {self.model} (round {round_number}/{self.max_tool_rounds})")
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=self.messages,
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
            payload = self._normalize_payload(payload)
            response_type = payload.get("type")
            if response_type == "final":
                missing_requested_tools = sorted(required_tool_names - satisfied_tool_names)
                if missing_requested_tools:
                    self.messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=True)})
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
                    self.messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=True)})
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "This request requires real tool use in this turn. Do not answer from memory. Use the appropriate tool now and only then provide a final answer.",
                        }
                    )
                    continue
                assistant_text = str(payload.get("message", "")).strip()
                if self._final_claims_file_mutation(assistant_text) and not mutation_verified_this_turn:
                    self.messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=True)})
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
                    return AgentResult(message=failure, rounds=round_number, completed=False)
                if expected_exact_file_line is not None:
                    latest_read_result = self._latest_successful_tool_result(successful_tool_results, "read_file")
                    if latest_read_result is not None:
                        latest_read_output = str(latest_read_result.get("output", ""))
                        if expected_exact_file_line not in latest_read_output:
                            self.messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=True)})
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
                        return AgentResult(message=expected_exact_reply_text, rounds=round_number, completed=True)
                    self.messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=True)})
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
                        required_tool_names.update(decision["required_tools"])
                        forbidden_tool_names.update(decision["forbidden_tools"])
                        required_tool_names.difference_update(forbidden_tool_names)
                        if verification_retries > MAX_VERIFICATION_RETRIES:
                            failure = "Stopped because grounded final verification could not accept a final answer."
                            self._record_event("assistant", content=failure, rounds=round_number)
                            return AgentResult(message=failure, rounds=round_number, completed=False)
                        self.messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=True)})
                        self.messages.append({"role": "user", "content": self._verification_retry_message(decision)})
                        continue
                self.messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=True)})
                self._record_event("assistant", content=assistant_text, rounds=round_number)
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
                    self.messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=True)})
                    self.messages.append(
                        {
                            "role": "user",
                            "content": f"Do not use {name} in this turn. Choose a different allowed tool or answer from existing verified tool results. Respond with the next JSON object only.",
                        }
                    )
                    continue
                if session_memory_request:
                    self.messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=True)})
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "This request is about prior chat/session memory, not the workspace. Answer from the conversation history already in context. Do not use tools for it. Respond with the next JSON object only.",
                        }
                    )
                    continue
                if name in MUTATING_TOOL_NAMES and not mutation_allowed:
                    self.messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=True)})
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "Do not mutate files or create commits in this turn. The user asked for inspection or reporting only. Use a read-only tool or answer from verified tool results. Respond with the next JSON object only.",
                        }
                    )
                    continue
                if (
                    name == "run_shell"
                    and not mutation_allowed
                    and self._shell_looks_like_file_mutation(str(arguments.get("command", "")))
                ):
                    self.messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=True)})
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
                    self.messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=True)})
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "Do not use run_shell for direct file creation or file edits here. Use write_file or replace_in_file instead, then respond with the next JSON object only.",
                        }
                    )
                    continue
                if name == "git_diff" and requested_git_diff_mode == "working-tree" and arguments.get("cached") is True:
                    self.messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=True)})
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "For a working-tree or unstaged diff, call git_diff with cached false or omit cached. Respond with the next JSON object only.",
                        }
                    )
                    continue
                if name == "git_diff" and requested_git_diff_mode == "staged" and arguments.get("cached") is not True:
                    self.messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=True)})
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "For a staged diff, call git_diff with cached true. Respond with the next JSON object only.",
                        }
                    )
                    continue
                cached_result = self._get_cached_tool_result(name, arguments)
                cache_hit = cached_result is not None
                should_skip_assumption_audit = (
                    not self.debate_enabled
                    or normalization_reason is not None
                    or cache_hit
                )
                if not should_skip_assumption_audit:
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
                            return AgentResult(message=failure, rounds=round_number, completed=False)
                        self.messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=True)})
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
                self.messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=True)})
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
                real_tool_use = self._counts_as_real_tool_use(name, result) or self._failure_result_counts_for_request(text, name, result)
                tool_used_this_turn = tool_used_this_turn or real_tool_use
                if real_tool_use:
                    satisfied_tool_names.add(name)
                    if name in MUTATING_TOOL_NAMES:
                        mutation_verified_this_turn = True
                if self._counts_as_real_tool_use(name, result):
                    successful_tool_results.append(
                        {
                            "name": name,
                            "arguments": deepcopy(arguments),
                            "result": deepcopy(result),
                        }
                    )
                self._record_event("tool_result", name=name, result=result, rounds=round_number, cached=cache_hit)
                self.messages.append(
                    {
                        "role": "user",
                        "content": self._tool_result_feedback_message(name, result, real_tool_use=real_tool_use),
                    }
                )
                if (
                    self._request_expects_exact_tool_error(text)
                    and self._failure_result_counts_for_request(text, name, result)
                ):
                    failure_message = str(result.get("summary") or result.get("output") or "").strip()
                    if failure_message:
                        self._record_event("assistant", content=failure_message, rounds=round_number)
                        return AgentResult(message=failure_message, rounds=round_number, completed=True)
                continue
            self.messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=True)})
            self.messages.append(
                {
                    "role": "user",
                    "content": 'Your previous reply had an invalid "type". Reply again with either {"type":"tool",...} or {"type":"final",...}.',
                }
            )
        failure = "Stopped after reaching the maximum tool rounds."
        self._record_event("assistant", content=failure, rounds=self.max_tool_rounds)
        return AgentResult(message=failure, rounds=self.max_tool_rounds, completed=False)
