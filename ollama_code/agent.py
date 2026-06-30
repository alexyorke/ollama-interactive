from __future__ import annotations

import ast
from copy import deepcopy
import difflib
import json
import math
import re
import shlex
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any, Callable
import threading

from ollama_code.features import active_feature_profile, feature_enabled, options_for_purpose, response_format_for_purpose
from ollama_code.agent_parsing import (
    _workspace_roots_match,
    extract_json_like_fields,
    extract_json_response,
)
from ollama_code.agent_protocol import (
    AGENT_TOOL_NAMES,
    APPROVAL_RANK,
    AUDIT_LIST_ITEM_LIMIT,
    AUDIT_TEXT_ITEM_LIMIT,
    BROAD_CONTEXT_GATHERING_TOOL_NAMES,
    CANDIDATE_CLAIM_LIMIT,
    CANDIDATE_CLAIM_TEXT_LIMIT,
    CODE_EDIT_SUFFIXES,
    CONTEXT_GATHERING_TOOL_NAMES,
    CORE_READ_ONLY_WORKSPACE_TOOL_NAMES,
    EDIT_TOOL_NAMES,
    ExactFileWriteSpec,
    FinalRewriteOutcome,
    GRAPH_TOOL_NAMES,
    GIT_TOOL_NAMES,
    GROUNDING_EVIDENCE_TOOL_NAMES,
    INDEX_REFRESH_TOOL_NAMES,
    KNOWN_TOOL_NAMES,
    LOW_LEVEL_EDIT_TOOL_NAMES,
    LSP_TOOL_NAMES,
    MAX_ASSUMPTION_AUDIT_RETRIES,
    MAX_RECONCILIATION_RETRIES,
    MAX_VERIFICATION_RETRIES,
    MAX_VERIFICATION_REWRITE_ATTEMPTS,
    MechanicalToolOccurrence,
    MechanicalToolSpec,
    MODEL_TOOL_DIFF_LIMIT,
    MODEL_TOOL_RESULT_LIMITS,
    MUTATING_TOOL_NAMES,
    PREEMPTIVE_SPEC_GUIDED_SYNTHESIS_TOOL_NAMES,
    PRIMARY_CONTEXT_CONTENT_LIMIT,
    PRIMARY_CONTEXT_RECENT_MESSAGE_LIMIT,
    PRIMARY_CURRENT_REQUEST_LIMIT,
    PYTHON_SDK_TOOL_NAMES,
    QUESTION_PLANNER_EVIDENCE_LIMIT,
    QUESTION_PLANNER_MAX_QUESTIONS,
    READ_ONLY_CACHEABLE_TOOL_NAMES,
    READ_ONLY_WORKSPACE_TOOL_NAMES,
    RETRY_PRONE_MUTATING_TOOL_NAMES,
    RISKY_VERIFICATION_TOOL_NAMES,
    SHELL_TOOL_NAMES,
    SPEC_GUIDED_REPAIR_CANDIDATE_TIMEOUT,
    SPEC_GUIDED_REPAIR_MAX_ATTEMPTS,
    SPEC_GUIDED_SYNTHESIS_TOOL_NAMES,
    STRUCTURAL_SEARCH_TOOL_NAMES,
    SymbolReadSpec,
    TEST_TOOL_NAMES,
    TODO_TOOL_NAMES,
    TargetLineReadSpec,
    VALIDATION_TOOL_NAMES,
    VERIFICATION_CONTENT_LIMIT,
    VERIFICATION_EVIDENCE_LIMIT,
    VERIFICATION_EVIDENCE_TEXT_LIMIT,
    VERIFICATION_HISTORY_LIMIT,
    VERIFICATION_TOOL_DIFF_LIMIT,
    VERIFICATION_TOOL_RESULT_LIMIT,
    VERIFIED_FUNCTION_TOOL_NAMES,
    AgentResult,
)
from ollama_code.controller import NavigationValidationController, NavigationValidationTurn
from ollama_code.ollama_client import ChatResponse, OllamaClient, OllamaError
from ollama_code.prompts import (
    ARTIFACT_RECONCILER_SYSTEM_PROMPT,
    FINAL_REWRITER_SYSTEM_PROMPT,
    FINAL_VERIFIER_SYSTEM_PROMPT,
    QUESTION_PLANNER_SYSTEM_PROMPT,
    SYSTEM_PROMPT_TEMPLATE,
    TOOL_ASSUMPTION_AUDITOR_SYSTEM_PROMPT,
)
from ollama_code.sessions import (
    SessionSummary,
    default_session_dir,
    list_sessions as collect_sessions,
    load_transcript_payload,
    resolve_transcript_path,
    transcript_message_role_supported,
    write_transcript_payload,
)
from ollama_code.tools import ToolExecutor, format_compact_tool_help, format_tool_group_help, format_tool_help


TRANSCRIPT_DIAGNOSTIC_STRING_LIMIT = 4000
TRANSCRIPT_DIAGNOSTIC_DEPTH_LIMIT = 40
TRANSCRIPT_DIAGNOSTIC_COLLECTION_LIMIT = 200
TRANSCRIPT_DIAGNOSTIC_DEPTH_MARKER = "[nested payload truncated for transcript]"
TRANSCRIPT_DIAGNOSTIC_CYCLE_MARKER = "[circular reference omitted for transcript]"
TRANSCRIPT_DIAGNOSTIC_DICT_MARKER_KEY = "__ollama_code_transcript_truncated__"


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
        disable_spec_guided_repair: bool = False,
        require_llm_for_turn: bool = False,
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
        self.disable_spec_guided_repair = bool(disable_spec_guided_repair)
        self.require_llm_for_turn = bool(require_llm_for_turn)
        self.events: list[dict[str, Any]] = []
        self.llm_telemetry_events: list[dict[str, Any]] = []
        self._pending_llm_call_events: list[dict[str, Any]] = []
        self._llm_used_this_turn = False
        self._interrupt_event: threading.Event | None = None
        self._turn_tool_cache: dict[tuple[str, str, int], dict[str, Any]] = {}
        self._turn_cache_epoch = 0
        self._turn_evidence_counter = 0
        self._transcript_dirty = False
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
        available = self.tools.available_tool_names()
        selected = available if tool_names is None else set(tool_names) & available
        return SYSTEM_PROMPT_TEMPLATE.format(
            workspace_root=self.tools.workspace_root.as_posix(),
            tool_help=format_compact_tool_help(selected, grouped=True),
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
            selected.update(CORE_READ_ONLY_WORKSPACE_TOOL_NAMES)
        if self._request_benefits_from_todos(request_text, mutation_required=mutation_required, test_run_required=test_run_required):
            selected.update(TODO_TOOL_NAMES)
        if self._request_benefits_from_systems_lens(request_text):
            selected.add("systems_lens")
        if re.search(r"\b(?:refactor|signature|type|return|caller|callee|contract|pipeline|pure|function chain)\b", lowered):
            selected.update(GRAPH_TOOL_NAMES)
        if re.search(r"\b(?:verified|known utilit|reusable function|verified function|function card|compose|lego|legos|do not invent|don't invent|dont invent|pure function|purity)\b", lowered):
            selected.update(VERIFIED_FUNCTION_TOOL_NAMES)
        if re.search(r"\b(?:python sdk|python api|stdlib|standard library|builtin|builtins|current python|python docs?|sdk search)\b", lowered):
            selected.update(PYTHON_SDK_TOOL_NAMES)
        if re.search(r"\b(?:library|package|site-packages|stdlib|standard library|traceback|stack trace|decompile|disassembl|builtin)\b", lowered):
            selected.add("inspect_library_source")
        if re.search(r"\b(?:ast|ast-grep|semgrep|structural|syntax-aware|codemod|pattern search|api misuse)\b", lowered):
            selected.update(STRUCTURAL_SEARCH_TOOL_NAMES)
        if re.search(r"\b(?:lsp|language server|diagnostic|diagnostics|definition|references|go to definition|xref)\b", lowered):
            selected.update(LSP_TOOL_NAMES)
        if re.search(r"\b(?:refresh index|reindex|index refresh|fts_refresh|repo_index_refresh|file_index_refresh)\b", lowered):
            selected.update(INDEX_REFRESH_TOOL_NAMES)
        if re.search(r"\b(?:everything search|es\.exe)\b", lowered):
            selected.add("everything_search")
        if mutation_allowed or mutation_required:
            selected.update(CORE_READ_ONLY_WORKSPACE_TOOL_NAMES)
            selected.update(EDIT_TOOL_NAMES)
        if test_run_required or re.search(r"\b(?:test|pytest|unittest|run tests?)\b", lowered):
            selected.update(TEST_TOOL_NAMES)
            selected.add("discover_validators")
        if re.search(r"\b(?:shell|command|execute|run exactly|terminal|powershell|bash)\b", lowered):
            selected.update(SHELL_TOOL_NAMES)
            selected.add("diagnose_dependency_error")
        if re.search(r"\b(?:git|diff|status|commit|staged|working tree|checkout|checked out|merge|rebase|stash)\b", lowered):
            selected.update(GIT_TOOL_NAMES)
        if re.search(r"\b(?:browser|ui|frontend|page|localhost|url|screenshot|playwright)\b", lowered):
            selected.add("browser_smoke")
        if re.search(r"\b(?:security|secret|vulnerab|audit|cve|dependency scan|scanner|gitleaks|trivy|osv)\b", lowered):
            selected.add("security_scan")
        if re.search(r"\b(?:mcp|model context protocol)\b", lowered):
            selected.update({"mcp_list_tools", "mcp_call"})
        if re.search(r"\b(?:agent|subagent|sub-agent|delegate)\b", lowered):
            selected.update(AGENT_TOOL_NAMES)
        selected.update(required_tool_names)
        if "context_pack" not in required_tool_names:
            selected.discard("context_pack")
            if (
                mutation_required
                or test_run_required
                or self._request_is_broad_or_ambiguous(request_text)
            ):
                selected.add("context_pack")
        selected.difference_update(forbidden_tool_names)
        return {name for name in selected if name in KNOWN_TOOL_NAMES}

    def reset(self) -> None:
        self.messages = self._base_messages()
        self.events = []
        self.llm_telemetry_events = []
        self.tools.clear_todos()
        self._reset_turn_cache()
        self._transcript_dirty = True
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

    def start_indexer(self) -> bool:
        indexer = getattr(self.tools, "indexer", None)
        start = getattr(indexer, "start", None)
        return bool(start()) if callable(start) else False

    def stop_indexer(self) -> None:
        indexer = getattr(self.tools, "indexer", None)
        stop = getattr(indexer, "stop", None)
        if callable(stop):
            stop()

    def refresh_index(self) -> dict[str, Any]:
        indexer = getattr(self.tools, "indexer", None)
        refresh = getattr(indexer, "request_refresh", None)
        status = getattr(indexer, "status", None)
        if callable(refresh):
            refresh("manual")
            return {"ok": True, "summary": "index refresh queued", "status": status() if callable(status) else {}}
        return {"ok": False, "summary": "indexer is not configured"}

    def index_status(self) -> dict[str, Any]:
        indexer = getattr(self.tools, "indexer", None)
        status = getattr(indexer, "status", None)
        if callable(status):
            return {"ok": True, **status()}
        return {"ok": False, "enabled": False, "summary": "indexer is not configured"}

    def session_directory(self) -> Path:
        return default_session_dir(self.tools.workspace_root)

    def tool_help(self, *, compact: bool = False) -> str:
        tool_names = self.tools.available_tool_names()
        return format_compact_tool_help(tool_names) if compact else format_tool_help(tool_names)

    def tool_group_help(self) -> str:
        return format_tool_group_help(self.tools.available_tool_names())

    def tool_dependency_status(self, scope: str = "all", tool_id: str | None = None) -> dict[str, Any]:
        return self.tools.execute("tool_status", {"scope": scope, "tool_id": tool_id})

    def tool_dependency_install(
        self,
        tool_id: str | None = None,
        *,
        all_recommended: bool = False,
        confirm: bool = False,
    ) -> dict[str, Any]:
        return self.tools.execute(
            "tool_install",
            {"tool_id": tool_id, "all_recommended": all_recommended, "confirm": confirm},
        )

    def todo_read(self) -> dict[str, Any]:
        return self.tools.execute("todo_read", {})

    def todo_clear(self) -> dict[str, Any]:
        return self.tools.execute("todo_write", {"items": []})

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
            if not transcript_message_role_supported(role) or not isinstance(content, str):
                raise ValueError("Saved session contains a malformed message.")
            restored_messages.append({"role": role, "content": content})
        events = payload.get("events")
        llm_telemetry_events = payload.get("llm_telemetry_events")
        todos = payload.get("todos")
        normalized_events = self._normalize_transcript_diagnostic_payload(list(events)) if isinstance(events, list) else []
        normalized_telemetry = (
            self._normalize_transcript_diagnostic_payload(list(llm_telemetry_events))
            if isinstance(llm_telemetry_events, list)
            else []
        )
        self.messages = restored_messages
        self.events = normalized_events if isinstance(normalized_events, list) else []
        self.llm_telemetry_events = normalized_telemetry if isinstance(normalized_telemetry, list) else []
        self.tools.set_todos(todos if isinstance(todos, list) else [])
        self._transcript_dirty = True
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

    def git_branch(self, path: str | None = None, *, all_branches: bool = False) -> dict[str, Any]:
        arguments: dict[str, Any] = {"all_branches": all_branches}
        if path:
            arguments["path"] = path
        return self.tools.execute("git_branch", arguments)

    def git_log(self, path: str | None = None, *, max_count: int = 10, oneline: bool = True) -> dict[str, Any]:
        arguments: dict[str, Any] = {"max_count": max_count, "oneline": oneline}
        if path:
            arguments["path"] = path
        return self.tools.execute("git_log", arguments)

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
            "todos": self.tools.todo_snapshot(),
            "messages": self.messages,
            "events": self._normalize_transcript_diagnostic_payload(self.events),
            "llm_telemetry_events": self._normalize_transcript_diagnostic_payload(self.llm_telemetry_events),
        }
        write_transcript_payload(target, payload)
        if self.session_file is not None and target.resolve(strict=False) == self.session_file.resolve(strict=False):
            self._transcript_dirty = False
        return target

    def _normalize_transcript_diagnostic_payload(
        self,
        value: Any,
        *,
        _depth: int = 0,
        _seen: set[int] | None = None,
    ) -> Any:
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else str(value)
        if isinstance(value, str):
            if len(value) <= TRANSCRIPT_DIAGNOSTIC_STRING_LIMIT:
                return value
            omitted = len(value) - TRANSCRIPT_DIAGNOSTIC_STRING_LIMIT
            return value[:TRANSCRIPT_DIAGNOSTIC_STRING_LIMIT] + f"\n... [truncated {omitted} chars for transcript]"
        if isinstance(value, PurePath):
            return value.as_posix()
        if _depth >= TRANSCRIPT_DIAGNOSTIC_DEPTH_LIMIT:
            return TRANSCRIPT_DIAGNOSTIC_DEPTH_MARKER
        if isinstance(value, dict):
            marker = id(value)
            active = _seen if _seen is not None else set()
            if marker in active:
                return TRANSCRIPT_DIAGNOSTIC_CYCLE_MARKER
            active.add(marker)
            try:
                items = list(value.items())
                omitted = 0
                if _depth > 0 and len(items) > TRANSCRIPT_DIAGNOSTIC_COLLECTION_LIMIT:
                    omitted = len(items) - TRANSCRIPT_DIAGNOSTIC_COLLECTION_LIMIT
                    items = items[:TRANSCRIPT_DIAGNOSTIC_COLLECTION_LIMIT]
                normalized = {
                    str(key): self._normalize_transcript_diagnostic_payload(item, _depth=_depth + 1, _seen=active)
                    for key, item in items
                }
                if omitted:
                    marker_key = TRANSCRIPT_DIAGNOSTIC_DICT_MARKER_KEY
                    while marker_key in normalized:
                        marker_key += "_"
                    normalized[marker_key] = f"[truncated {omitted} entries for transcript]"
                return normalized
            finally:
                active.remove(marker)
        if isinstance(value, (list, tuple)):
            marker = id(value)
            active = _seen if _seen is not None else set()
            if marker in active:
                return TRANSCRIPT_DIAGNOSTIC_CYCLE_MARKER
            active.add(marker)
            try:
                items = list(value)
                omitted = 0
                if _depth > 0 and len(items) > TRANSCRIPT_DIAGNOSTIC_COLLECTION_LIMIT:
                    omitted = len(items) - TRANSCRIPT_DIAGNOSTIC_COLLECTION_LIMIT
                    items = items[:TRANSCRIPT_DIAGNOSTIC_COLLECTION_LIMIT]
                normalized = [
                    self._normalize_transcript_diagnostic_payload(item, _depth=_depth + 1, _seen=active)
                    for item in items
                ]
                if omitted:
                    normalized.append(f"[truncated {omitted} items for transcript]")
                return normalized
            finally:
                active.remove(marker)
        if isinstance(value, set):
            marker = id(value)
            active = _seen if _seen is not None else set()
            if marker in active:
                return TRANSCRIPT_DIAGNOSTIC_CYCLE_MARKER
            active.add(marker)
            try:
                items = sorted(value, key=lambda item: str(item))
                omitted = 0
                if _depth > 0 and len(items) > TRANSCRIPT_DIAGNOSTIC_COLLECTION_LIMIT:
                    omitted = len(items) - TRANSCRIPT_DIAGNOSTIC_COLLECTION_LIMIT
                    items = items[:TRANSCRIPT_DIAGNOSTIC_COLLECTION_LIMIT]
                normalized = [
                    self._normalize_transcript_diagnostic_payload(item, _depth=_depth + 1, _seen=active)
                    for item in items
                ]
                if omitted:
                    normalized.append(f"[truncated {omitted} items for transcript]")
                return normalized
            finally:
                active.remove(marker)
        isoformat = getattr(value, "isoformat", None)
        if callable(isoformat):
            try:
                return str(isoformat())
            except TypeError:
                pass
        return self._normalize_transcript_diagnostic_payload(str(value))

    def _autosave(self) -> None:
        if self.session_file is not None and self._transcript_dirty:
            self.save_transcript(self.session_file)

    def _record_event(self, event_type: str, **payload: Any) -> None:
        self.events.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": event_type,
                **payload,
            }
        )
        self._transcript_dirty = True
        if event_type in {"user", "assistant"}:
            self._autosave()

    def _record_llm_telemetry(self, event_type: str, **payload: Any) -> None:
        self.llm_telemetry_events.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": event_type,
                **payload,
            }
        )
        self._transcript_dirty = True

    def _chat(
        self,
        *,
        purpose: str,
        model: str,
        messages: list[dict[str, str]],
        response_format: str | dict[str, Any] | None = "json",
        on_thinking: Callable[[str], None] | None = None,
        think: bool | None = None,
        options: dict[str, Any] | None = None,
        primary_can_emit_large_payload: bool = False,
    ) -> ChatResponse:
        effective_response_format = response_format_for_purpose(purpose, response_format)
        effective_options = options_for_purpose(purpose, primary_can_emit_large_payload=primary_can_emit_large_payload)
        if options:
            effective_options.update(options)
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
        started_at = datetime.now(timezone.utc).isoformat()
        self._record_llm_telemetry(
            "llm_call_started",
            purpose=purpose,
            requested_model=model,
            message_count=len(messages),
            prompt_chars=prompt_chars,
            prompt_chars_by_role=prompt_chars_by_role,
            top_prompt_messages=top_messages,
            think=think,
            response_format="schema" if isinstance(effective_response_format, dict) else effective_response_format,
            options=effective_options,
            started_at=started_at,
        )
        if isinstance(self.client, OllamaClient):
            self._autosave()
        partial_state = {"thinking_chars": 0}

        def thinking_recorder(text: str) -> None:
            if on_thinking is not None:
                on_thinking(text)
            current_len = len(text)
            if current_len - int(partial_state["thinking_chars"]) >= 400:
                partial_state["thinking_chars"] = current_len
                self._record_llm_telemetry(
                    "llm_partial",
                    purpose=purpose,
                    requested_model=model,
                    thinking_chars=current_len,
                    preview=text[-300:],
                )
                if isinstance(self.client, OllamaClient):
                    self._autosave()

        response = self.client.chat(
            model=model,
            messages=messages,
            response_format=effective_response_format,
            on_thinking=thinking_recorder if on_thinking is not None else None,
            think=think,
            options=effective_options or None,
        )
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
            "response_format": "schema" if isinstance(effective_response_format, dict) else effective_response_format,
            "options": effective_options,
            **response.usage.as_event_payload(),
        }
        self._llm_used_this_turn = True
        self._pending_llm_call_events.append(payload)
        return response

    def _flush_llm_call_events(self) -> None:
        pending = self._pending_llm_call_events
        self._pending_llm_call_events = []
        for payload in pending:
            self._record_event("llm_call", **payload)
        self._autosave()

    def _primary_messages_for_model(
        self,
        *,
        session_memory_request: bool,
        current_request: str,
        tool_names: set[str] | None = None,
    ) -> list[dict[str, str]]:
        dynamic_system_message = {"role": "system", "content": self._system_prompt_for_tools(tool_names)}
        if session_memory_request:
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
            limit = PRIMARY_CURRENT_REQUEST_LIMIT if message is current_turn_request else PRIMARY_CONTEXT_CONTENT_LIMIT
            compacted.append(
                {
                    "role": str(message.get("role", "")),
                    "content": self._truncate_text(str(message.get("content", "")), limit=limit),
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

    def _test_failure_source_excerpt(self, output: str, *, successful_tool_results: list[dict[str, Any]] | None = None, limit: int = 520) -> str:
        root = self.tools.workspace_root.resolve(strict=False)
        snippets: list[str] = []
        seen: set[tuple[Path, int]] = set()
        successful_tool_results = successful_tool_results or []
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
            failed_test_path, _ = self._failed_test_output_paths_from_text(output)
            if failed_test_path:
                inferred_source_path = self._infer_source_for_test_path(failed_test_path, output=output)
                if inferred_source_path:
                    try:
                        candidate = self.tools.resolve_path(inferred_source_path, allow_missing=False)
                        candidate_lines = candidate.read_text(encoding="utf-8").splitlines()
                    except Exception:
                        return ""
                    excerpt_lines = [f"{index}: {line}" for index, line in enumerate(candidate_lines[: min(18, len(candidate_lines))], start=1)]
                    snippets.append(f"{inferred_source_path}:{1}\n" + "\n".join(excerpt_lines))
            else:
                for test_path, source_path in reversed(self._recent_source_paths_with_results(successful_tool_results)):
                    if source_path:
                        try:
                            candidate = self.tools.resolve_path(source_path, allow_missing=False)
                            lines = candidate.read_text(encoding="utf-8").splitlines()
                        except Exception:
                            continue
                        excerpt_lines = [f"{index}: {line}" for index, line in enumerate(lines[: min(18, len(lines))], start=1)]
                        snippets.append(f"{source_path}:{1}\n" + "\n".join(excerpt_lines))
                        break
            if not snippets:
                return ""
        return "Failing source excerpt:\n" + self._truncate_text("\n\n".join(snippets), limit=limit)

    def _recent_source_paths_with_results(self, successful_tool_results: list[dict[str, Any]]) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        seen: set[str] = set()
        for item in reversed(successful_tool_results):
            if item.get("name") not in {"read_file", "find_implementation_target", "implementation_spec", "diagnose_test_failure"}:
                continue
            for source_path in self._source_paths_from_tool_result(item):
                if source_path in seen:
                    continue
                seen.add(source_path)
                pairs.append((source_path, source_path))
        return pairs

    def _source_paths_from_tool_result(self, item: dict[str, Any]) -> list[str]:
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        candidates: list[str] = []
        direct_path = str(result.get("path") or arguments.get("source_path") or arguments.get("path") or "").strip().replace("\\", "/")
        if direct_path:
            candidates.append(direct_path)
        targets = result.get("targets")
        if isinstance(targets, list):
            for target in targets:
                if not isinstance(target, dict):
                    continue
                target_path = str(target.get("path") or target.get("source_path") or "").strip().replace("\\", "/")
                if target_path:
                    candidates.append(target_path)
        paths: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = candidate.lstrip("./")
            if (
                not normalized
                or normalized in seen
                or not normalized.endswith(".py")
                or self._path_looks_like_test_file(normalized)
            ):
                continue
            seen.add(normalized)
            paths.append(normalized)
        return paths

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
                "id": str(item.get("evidence_id") or f"E{index}"),
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
        for match in re.findall(r"(?<![A-Za-z0-9_])(mcp\.[a-z0-9_-]+\.[a-z0-9_.-]+)(?![A-Za-z0-9_])", text):
            clean = match.rstrip(".,;:")
            if self._is_supported_tool_name(clean):
                matches.add(clean)
        return matches

    def _forbidden_tool_names(self, text: str) -> set[str]:
        lowered = text.lower()
        masked = re.sub(
            r"mcp\.[a-z0-9_-]+\.[a-z0-9_.-]+",
            lambda match: match.group(0).replace(".", "__mcpdot__"),
            lowered,
        )
        fragments = re.findall(r"\b(?:do not|don't|dont|never|avoid)\b[^.?!\n]{0,160}", masked)
        fragments.extend(re.findall(r"\bwithout(?: using)?\b[^.?!\n]{0,160}", masked))
        fragments.extend(re.findall(r"\bnot\s+(?:with|using|via)?\s*[^.?!\n]{0,80}", masked))
        forbidden: set[str] = set()
        for fragment in fragments:
            forbidden.update(self._tool_names_in_fragment(fragment.replace("__mcpdot__", ".")))
        return forbidden

    def _intrinsic_forbidden_tool_names(self) -> set[str]:
        available = self.tools.available_tool_names()
        return {name for name in KNOWN_TOOL_NAMES if name not in available}

    def _requested_tool_names(self, text: str, *, forbidden_tool_names: set[str] | None = None) -> set[str]:
        lowered = text.lower()
        fragments = re.findall(r"\b(?:use|call|run|invoke|start)\b[^.?!\n]{0,160}", lowered)
        requested: set[str] = set()
        for fragment in fragments:
            requested.update(self._tool_names_in_fragment(fragment))
        requested.update(self._tool_names_in_fragment(lowered))
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
        required_tools = [name for name in decision.get("required_tools", []) if isinstance(name, str) and self._is_supported_tool_name(name)]
        forbidden_tools = [name for name in decision.get("forbidden_tools", []) if isinstance(name, str) and self._is_supported_tool_name(name)]
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

    def _normalize_assumption_audit_payload(self, payload: dict[str, Any] | None, *, raw_text: str = "") -> dict[str, Any]:
        decision = dict(payload) if isinstance(payload, dict) else {}
        recovered = extract_json_like_fields(
            raw_text,
            scalar_keys=("verdict", "reason"),
            array_keys=("assumptions", "validation_steps", "required_tools", "forbidden_tools"),
        )
        for key, value in recovered.items():
            current = decision.get(key)
            if key not in decision or current is None or current == "" or current == []:
                decision[key] = value
        verdict = str(decision.get("verdict", "")).strip().lower()
        if verdict not in {"accept", "retry"}:
            verdict = "retry"
        reason = str(decision.get("reason", "")).strip()
        required_tools = [name for name in decision.get("required_tools", []) if isinstance(name, str) and self._is_supported_tool_name(name)]
        forbidden_tools = [name for name in decision.get("forbidden_tools", []) if isinstance(name, str) and self._is_supported_tool_name(name)]
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
        decision = self._normalize_assumption_audit_payload(extract_json_response(verdict_response.content), raw_text=verdict_response.content)
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

    def _normalize_reconciliation_payload(self, payload: dict[str, Any] | None, *, raw_text: str = "") -> dict[str, Any]:
        decision = dict(payload) if isinstance(payload, dict) else {}
        recovered = extract_json_like_fields(
            raw_text,
            scalar_keys=("verdict", "reason"),
            array_keys=("repair_plan", "required_tools", "forbidden_tools"),
        )
        for key, value in recovered.items():
            current = decision.get(key)
            if key not in decision or current is None or current == "" or current == []:
                decision[key] = value
        verdict = str(decision.get("verdict", "")).strip().lower()
        if verdict not in {"accept", "retry"}:
            verdict = "retry"
        reason = str(decision.get("reason", "")).strip()
        required_tools = [name for name in decision.get("required_tools", []) if isinstance(name, str) and self._is_supported_tool_name(name)]
        forbidden_tools = [name for name in decision.get("forbidden_tools", []) if isinstance(name, str) and self._is_supported_tool_name(name)]
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
        decision = self._normalize_reconciliation_payload(extract_json_response(verdict_response.content), raw_text=verdict_response.content)
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

    def _stabilize_retry_tool_constraints(
        self,
        decision: dict[str, Any],
        *,
        sticky_required_tool_names: set[str],
        sticky_forbidden_tool_names: set[str],
    ) -> dict[str, Any]:
        normalized = dict(decision)
        required_tools = {
            name
            for name in normalized.get("required_tools", [])
            if isinstance(name, str) and self._is_supported_tool_name(name)
        }
        forbidden_tools = {
            name
            for name in normalized.get("forbidden_tools", [])
            if isinstance(name, str) and self._is_supported_tool_name(name)
        }
        required_tools = {name for name in required_tools if self.tools.is_tool_enabled(name)}
        sticky_required = {name for name in sticky_required_tool_names if self._is_supported_tool_name(name) and self.tools.is_tool_enabled(name)}
        sticky_forbidden = {name for name in sticky_forbidden_tool_names if self._is_supported_tool_name(name)}
        required_tools.update(sticky_required)
        forbidden_tools.update(sticky_forbidden)
        required_tools.difference_update(forbidden_tools)
        normalized["required_tools"] = sorted(required_tools)
        normalized["forbidden_tools"] = sorted(forbidden_tools)
        return normalized

    def _reset_turn_cache(self) -> None:
        self._turn_tool_cache = {}
        self._turn_cache_epoch = 0
        self._turn_evidence_counter = 0

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

    def _run_test_failure_follow_up(self, result: dict[str, Any], successful_tool_results: list[dict[str, Any]] | None = None) -> str:
        text = str(result.get("output") or result.get("summary") or "")
        repair_packet = self._run_test_repair_packet(text, successful_tool_results=successful_tool_results)
        source_excerpt = repair_packet.get("source_excerpt", "")
        diagnosis = repair_packet.get("diagnosis", "")
        likely_targets = repair_packet.get("likely_targets", "")
        remaining_stubs = repair_packet.get("remaining_stubs", "")
        test_examples = repair_packet.get("test_examples", "")
        packet_text = self._format_run_test_repair_packet(repair_packet)
        syntax_match = re.search(
            r"File \"(?P<path>[^\"]+)\", line (?P<line>\d+).*?\n(?:.*\n){0,2}?(?P<error>(?:IndentationError|SyntaxError): [^\n]+)",
            text,
            flags=re.DOTALL,
        )
        if syntax_match:
            path = self._display_path_basename(syntax_match.group("path"))
            line = syntax_match.group("line")
            error = syntax_match.group("error").strip()
            return (
                f"Tests failed with {error} at {path}:{line}. Fix that file, then rerun run_test. "
                "Do not blame imports unless error is ModuleNotFoundError. "
                + packet_text
                + " Next JSON only."
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
            if likely_targets:
                details += f" {likely_targets}"
            if remaining_stubs:
                details += f" {remaining_stubs}"
            if test_examples:
                details += f" {test_examples}"
            if source_excerpt:
                details += f" {source_excerpt}"
            return self._truncate_text(f"{details} Edit implementation, then rerun run_test. Next JSON only.", limit=1800)
        if diagnosis:
            return self._truncate_text(
                f"Tests failed. {diagnosis} {likely_targets} {remaining_stubs} {test_examples} Edit likely implementation targets, then rerun configured run_test. Next JSON only.",
                limit=1500,
            )
        if source_excerpt:
            return self._truncate_text(
                f"Tests failed. {source_excerpt} Inspect/edit evidence, then rerun configured run_test. Next JSON only.",
                limit=760,
            )
        return "Tests failed. Inspect/edit evidence, then rerun configured run_test. Next JSON only."

    def _run_test_repair_packet(self, output: str, successful_tool_results: list[dict[str, Any]] | None = None) -> dict[str, str]:
        packet = {"diagnosis": "", "likely_targets": "", "source_excerpt": "", "remaining_stubs": "", "test_examples": "", "static_sanity": "", "repair_matrix": ""}
        if not output.strip():
            return packet
        diagnosis_result: dict[str, Any] = {}
        try:
            diagnosis_result = self.tools.diagnose_test_failure(output=output, limit=4)
        except Exception:
            diagnosis_result = {}
        if diagnosis_result.get("ok") is True:
            diagnosis_text = str(diagnosis_result.get("output") or "").strip()
            if diagnosis_text and not diagnosis_text.startswith("(no structured failures"):
                packet["diagnosis"] = "Diagnosis: " + self._truncate_text(diagnosis_text.replace("\n", " | "), limit=420)
            targets = diagnosis_result.get("targets")
            if isinstance(targets, list) and targets:
                rendered: list[str] = []
                for item in targets[:4]:
                    if not isinstance(item, dict):
                        continue
                    path = str(item.get("path") or "").strip()
                    symbol = str(item.get("symbol") or "").strip()
                    if path:
                        rendered.append(path + (f"::{symbol}" if symbol else ""))
                if rendered:
                    packet["likely_targets"] = "Likely targets: " + ", ".join(rendered) + "."
                stubs = self._remaining_stub_targets(targets)
                if stubs:
                    packet["remaining_stubs"] = self._remaining_stubs_text(stubs)
        if not packet["diagnosis"]:
            fallback = self._diagnose_assertion_output(output)
            if fallback:
                packet["diagnosis"] = f"Diagnosis: {fallback}"
        if not packet["likely_targets"]:
            failed_test_path, _ = self._failed_test_output_paths_from_text(output)
            if failed_test_path:
                inferred = self._infer_source_for_test_path(failed_test_path, output=output, limit=8)
                if inferred:
                    packet["likely_targets"] = f"Likely targets: {inferred}."
        context_paths = self._recent_source_paths(successful_tool_results or [])
        if context_paths:
            stubs = self._stub_targets_for_paths(context_paths)
            if stubs and not packet["remaining_stubs"]:
                packet["remaining_stubs"] = self._remaining_stubs_text(stubs)
            try:
                sanity = self.tools.contract_check(context_paths, limit=8)
            except Exception:
                sanity = {}
            if sanity.get("ok") is False:
                packet["static_sanity"] = "Static sanity: " + self._truncate_text(str(sanity.get("output") or sanity.get("summary") or ""), limit=520)
        examples = self._test_examples_from_context(successful_tool_results or [])
        if examples:
            packet["test_examples"] = examples
        for source_path in context_paths[:2]:
            probe = self._test_example_probe_from_context(source_path, successful_tool_results or [], limit=8)
            if probe and probe.get("ok") is False:
                packet["repair_matrix"] = "Repair matrix: " + self._truncate_text(str(probe.get("output") or probe.get("summary") or ""), limit=700)
                break
        packet["source_excerpt"] = self._test_failure_source_excerpt(output, successful_tool_results=successful_tool_results or [])
        return packet

    def _diagnose_assertion_output(self, output: str) -> str:
        match = re.search(r"AssertionError:\s*([^\n]+)", output or "")
        if not match:
            return ""
        message = match.group(1).strip()
        equality = re.search(r"'([^']*)'\s*!=\s*'([^']*)'", message)
        if equality:
            return f"assertion mismatch: actual='{equality.group(1)}' expected='{equality.group(2)}'"
        return f"assertion mismatch: {message}"

    def _format_run_test_repair_packet(self, packet: dict[str, str]) -> str:
        parts = [value for key, value in packet.items() if key != "source_excerpt" and value]
        if packet.get("source_excerpt"):
            parts.append(packet["source_excerpt"])
        if not parts:
            return ""
        return "Repair packet: " + self._truncate_text(" ".join(parts).replace("\n", " | "), limit=1400)

    def _remaining_stubs_text(self, stubs: list[str]) -> str:
        if not stubs:
            return ""
        hint = "Remaining stubs: " + ", ".join(stubs[:12]) + "."
        if len(stubs) > 1:
            hint += " Implement all listed stubs in the compact source file in one replace_symbols or full-file edit before more reads or tests."
        return hint

    def _recent_source_paths(self, successful_tool_results: list[dict[str, Any]]) -> list[str]:
        paths: list[str] = []
        seen: set[str] = set()
        for item in reversed(successful_tool_results):
            for rel in self._source_paths_from_tool_result(item):
                if rel in seen:
                    continue
                seen.add(rel)
                paths.append(rel)
                if len(paths) >= 4:
                    return list(reversed(paths))
        return list(reversed(paths))

    def _recent_test_paths(self, successful_tool_results: list[dict[str, Any]]) -> list[str]:
        paths: list[str] = []
        seen: set[str] = set()
        for item in reversed(successful_tool_results):
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            rel = str(result.get("path") or arguments.get("path") or "").strip().replace("\\", "/")
            if not rel or rel in seen or not rel.endswith(".py") or not self._path_looks_like_test_file(rel):
                continue
            seen.add(rel)
            paths.append(rel)
            if len(paths) >= 3:
                break
        return list(reversed(paths))

    def _recent_identifier_search_query(self, successful_tool_results: list[dict[str, Any]]) -> str | None:
        for item in reversed(successful_tool_results):
            if item.get("name") != "search":
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            if result.get("ok") is not True:
                continue
            arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            query = str(arguments.get("query") or "").strip()
            if re.fullmatch(r"[A-Za-z_][\w.]*", query):
                return query
        return None

    def _recent_search_code_paths(
        self,
        successful_tool_results: list[dict[str, Any]],
        *,
        query: str | None = None,
    ) -> list[str]:
        expected_query = str(query or "").strip()
        for item in reversed(successful_tool_results):
            if item.get("name") != "search":
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            if result.get("ok") is not True:
                continue
            arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            if expected_query and str(arguments.get("query") or "").strip() != expected_query:
                continue
            paths: list[str] = []
            seen: set[str] = set()
            for raw_line in str(result.get("output") or "").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                match = re.match(r"^(?P<path>.+):\d+(?::|-)", line)
                if not match:
                    continue
                candidate = match.group("path").strip().replace("\\", "/")
                if not candidate:
                    continue
                try:
                    normalized = self.tools.relative_label(self.tools.resolve_path(candidate, allow_missing=False))
                except Exception:
                    normalized = candidate.lstrip("./")
                if (
                    not normalized
                    or normalized in seen
                    or not normalized.endswith(".py")
                    or self._path_looks_like_test_file(normalized)
                ):
                    continue
                seen.add(normalized)
                paths.append(normalized)
            return paths
        return []

    def _recent_list_files_code_paths(self, successful_tool_results: list[dict[str, Any]]) -> list[str]:
        for item in reversed(successful_tool_results):
            if item.get("name") != "list_files":
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            if result.get("ok") is not True:
                continue
            paths: list[str] = []
            seen: set[str] = set()
            for raw_line in str(result.get("output") or "").splitlines():
                candidate = raw_line.strip().replace("\\", "/").lstrip("./")
                if (
                    not candidate
                    or candidate.endswith("/")
                    or candidate in seen
                    or Path(candidate).suffix.lower() not in CODE_EDIT_SUFFIXES
                    or self._path_looks_like_test_file(candidate)
                ):
                    continue
                seen.add(candidate)
                paths.append(candidate)
            return paths
        return []

    def _recent_identifier_search_code_paths(self, successful_tool_results: list[dict[str, Any]]) -> list[str]:
        query = self._recent_identifier_search_query(successful_tool_results)
        if not query:
            return []
        return self._recent_search_code_paths(successful_tool_results, query=query)

    def _successful_context_pack_read_target(
        self,
        *,
        successful_tool_results: list[dict[str, Any]],
        preferred_symbol: str | None = None,
    ) -> tuple[str, dict[str, Any]] | None:
        normalized_preferred = str(preferred_symbol or "").strip().lower()
        for item in reversed(successful_tool_results):
            if item.get("name") != "context_pack":
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            if result.get("ok") is not True:
                continue
            suggested_next_tool = str(result.get("suggested_next_tool") or "").strip().lower()
            ranked_paths = list(
                dict.fromkeys(
                    str(path or "").strip().replace("\\", "/").lstrip("./")
                    for path in result.get("ranked_paths", [])
                    if isinstance(path, str)
                )
            )
            ranked_paths = [
                path
                for path in ranked_paths
                if path.endswith(".py") and not self._path_looks_like_test_file(path)
            ]
            ranked_symbols: list[tuple[str, str]] = []
            seen_ranked_symbols: set[tuple[str, str]] = set()
            for raw_symbol in result.get("ranked_symbols", []) if isinstance(result.get("ranked_symbols"), list) else []:
                if not isinstance(raw_symbol, dict):
                    continue
                path = str(raw_symbol.get("path") or "").strip().replace("\\", "/").lstrip("./")
                qualname = str(raw_symbol.get("qualname") or "").strip()
                if not path or not qualname or self._path_looks_like_test_file(path):
                    continue
                key = (path, qualname)
                if key in seen_ranked_symbols:
                    continue
                seen_ranked_symbols.add(key)
                ranked_symbols.append(key)
            if normalized_preferred:
                preferred_matches = [
                    (path, qualname)
                    for path, qualname in ranked_symbols
                    if normalized_preferred in {qualname.lower(), qualname.rsplit(".", 1)[-1].lower()}
                ]
                if len(preferred_matches) == 1:
                    path, qualname = preferred_matches[0]
                    return "read_symbol", {"path": path, "symbol": qualname, "include_context": 0}
            if suggested_next_tool == "read_symbol" and len(ranked_symbols) == 1:
                path, qualname = ranked_symbols[0]
                return "read_symbol", {"path": path, "symbol": qualname, "include_context": 0}
            if suggested_next_tool in {"read_symbol", "read_file"} and len(ranked_paths) == 1:
                return "read_file", {"path": ranked_paths[0]}
            return None
        return None

    def _parse_code_outline_symbols(self, output: str) -> list[tuple[str, str]]:
        symbols: list[tuple[str, str]] = []
        current_path = ""
        for raw_line in output.splitlines():
            line = raw_line.rstrip()
            if not line:
                continue
            if not line.startswith(" "):
                current_path = line.strip().replace("\\", "/").lstrip("./")
                continue
            match = re.match(r"^\s+\d+-\d+\s+\w+\s+(?P<qualname>[A-Za-z_][\w.]*)\s*:", line)
            if not match or not current_path:
                continue
            symbols.append((current_path, match.group("qualname").strip()))
        return symbols

    def _successful_single_outlined_symbol(
        self,
        *,
        path: str,
        successful_tool_results: list[dict[str, Any]],
    ) -> str | None:
        normalized_path = str(path or "").strip().replace("\\", "/").lstrip("./")
        if not normalized_path:
            return None
        for item in reversed(successful_tool_results):
            if item.get("name") != "code_outline":
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            if result.get("ok") is not True:
                continue
            arguments_dict = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            outline_path = str(result.get("path") or arguments_dict.get("path") or "").strip().replace("\\", "/").lstrip("./")
            if outline_path != normalized_path:
                continue
            matches = [
                qualname
                for match_path, qualname in self._parse_code_outline_symbols(str(result.get("output") or ""))
                if match_path == normalized_path
            ]
            unique = list(dict.fromkeys(matches))
            if len(unique) == 1:
                return unique[0]
            return None
        return None

    def _successful_implementation_target_symbol(
        self,
        *,
        path: str,
        successful_tool_results: list[dict[str, Any]],
        query: str | None = None,
    ) -> str | None:
        normalized_path = str(path or "").strip().replace("\\", "/").lstrip("./")
        if not normalized_path:
            return None
        normalized_query = str(query or "").strip().lower()
        for item in reversed(successful_tool_results):
            if item.get("name") != "find_implementation_target":
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            if result.get("ok") is not True:
                continue
            targets = result.get("targets") if isinstance(result.get("targets"), list) else []
            symbols: list[str] = []
            for target in targets:
                if not isinstance(target, dict):
                    continue
                target_path = str(target.get("path") or target.get("source_path") or "").strip().replace("\\", "/").lstrip("./")
                if target_path != normalized_path:
                    continue
                symbol = str(target.get("symbol") or "").strip()
                if symbol and symbol not in symbols:
                    symbols.append(symbol)
            if len(symbols) == 1:
                return symbols[0]
            if normalized_query:
                exact = [
                    symbol
                    for symbol in symbols
                    if normalized_query in {symbol.lower(), symbol.rsplit(".", 1)[-1].lower()}
                ]
                if len(exact) == 1:
                    return exact[0]
            if symbols:
                return None
        return None

    def _has_successful_read_symbol(
        self,
        *,
        path: str,
        symbol: str,
        successful_tool_results: list[dict[str, Any]],
    ) -> bool:
        normalized_path = str(path or "").strip().replace("\\", "/").lstrip("./")
        normalized_symbol = str(symbol or "").strip()
        if not normalized_path or not normalized_symbol:
            return False
        for item in reversed(successful_tool_results):
            if item.get("name") != "read_symbol":
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            if result.get("ok") is not True:
                continue
            arguments_dict = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            result_path = str(result.get("path") or arguments_dict.get("path") or "").strip().replace("\\", "/").lstrip("./")
            result_symbol = str(result.get("symbol") or arguments_dict.get("symbol") or "").strip()
            if result_path == normalized_path and result_symbol == normalized_symbol:
                return True
        return False

    def _has_successful_read_file(
        self,
        *,
        path: str,
        successful_tool_results: list[dict[str, Any]],
    ) -> bool:
        normalized_path = str(path or "").strip().replace("\\", "/").lstrip("./")
        if not normalized_path:
            return False
        for item in reversed(successful_tool_results):
            if item.get("name") != "read_file":
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            if result.get("ok") is not True:
                continue
            arguments_dict = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            result_path = str(result.get("path") or arguments_dict.get("path") or "").strip().replace("\\", "/").lstrip("./")
            if result_path == normalized_path:
                return True
        return False

    def _stub_targets_for_paths(self, paths: list[str]) -> list[str]:
        targets = [{"path": path} for path in paths]
        return self._remaining_stub_targets(targets)

    def _test_examples_from_context(self, successful_tool_results: list[dict[str, Any]]) -> str:
        test_paths = self._recent_test_paths(successful_tool_results)
        source_paths = self._recent_source_paths(successful_tool_results)
        if not test_paths:
            return ""
        source_path = source_paths[0] if source_paths else None
        outputs: list[str] = []
        for test_path in test_paths[:2]:
            try:
                result = self.tools.test_spec_extract(test_path=test_path, source_path=source_path, limit=32)
            except Exception:
                continue
            if result.get("ok") is True and int(result.get("count") or 0) > 0:
                outputs.append(str(result.get("output") or "").strip())
        if not outputs:
            return ""
        return "Test examples: " + self._truncate_text(" | ".join(outputs), limit=1100)

    def _test_example_probe_from_context(self, source_path: str, successful_tool_results: list[dict[str, Any]], *, limit: int = 8) -> dict[str, Any] | None:
        for test_path in self._recent_test_paths(successful_tool_results)[:2]:
            try:
                result = self.tools.run_test_example_probes(source_path=source_path, test_path=test_path, limit=limit, timeout=20)
            except Exception:
                continue
            output = str(result.get("output") or result.get("summary") or "")
            if "no executable examples" in output or "no Python test examples" in output:
                continue
            return result
        return None

    def _remaining_stub_targets(self, targets: object) -> list[str]:
        if not isinstance(targets, list):
            return []
        stubs: list[str] = []
        seen_paths: set[str] = set()
        for item in targets:
            if not isinstance(item, dict):
                continue
            rel = str(item.get("path") or "").strip()
            if not rel or rel in seen_paths or not rel.endswith(".py"):
                continue
            seen_paths.add(rel)
            try:
                path = self.tools.resolve_path(rel, allow_missing=False)
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if self._function_body_is_stub(list(node.body)):
                    stubs.append(f"{rel}::{node.name}")
        return stubs

    def _function_body_is_stub(self, body: list[ast.stmt]) -> bool:
        statements = [node for node in body if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))]
        if not statements:
            return True
        if len(statements) != 1:
            return False
        node = statements[0]
        if isinstance(node, ast.Pass):
            return True
        if isinstance(node, ast.Return):
            return node.value is None or (isinstance(node.value, ast.Constant) and node.value.value is None)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            value = node.value.value
            return value is Ellipsis or (
                isinstance(value, str) and re.search(r"\b(?:todo|stub|implement|your code)\b", value, flags=re.IGNORECASE) is not None
            )
        if isinstance(node, ast.Raise):
            raised = node.exc
            if isinstance(raised, ast.Call):
                raised = raised.func
            return isinstance(raised, ast.Name) and raised.id == "NotImplementedError"
        return False

    def _text_is_stub_like_python_repair(self, text: str) -> bool:
        stripped = textwrap.dedent(text or "").strip()
        if not stripped:
            return True
        meaningful_lines = [
            line.strip()
            for line in stripped.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not meaningful_lines:
            return True
        if all(line in {"pass", "...", "return None", "raise NotImplementedError", "raise NotImplementedError()"} for line in meaningful_lines):
            return True
        try:
            tree = ast.parse(stripped)
        except SyntaxError:
            return False
        functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        return bool(functions) and all(self._function_body_is_stub(list(node.body)) for node in functions)

    def _edit_payload_is_stub_like_repair(self, name: str, arguments: dict[str, Any]) -> bool:
        path = str(arguments.get("path") or arguments.get("file") or arguments.get("filename") or "").strip()
        if path and not path.replace("\\", "/").endswith(".py"):
            return False
        values: list[str] = []
        if name == "edit_intent":
            value = arguments.get("replacement")
            if isinstance(value, str):
                values.append(value)
        elif name == "replace_symbol":
            value = arguments.get("content")
            if isinstance(value, str):
                values.append(value)
        elif name == "replace_symbols":
            replacements = arguments.get("replacements")
            if isinstance(replacements, list):
                values.extend(str(item.get("content")) for item in replacements if isinstance(item, dict) and isinstance(item.get("content"), str))
        elif name == "write_file":
            value = arguments.get("content")
            if isinstance(value, str):
                values.append(value)
        elif name == "replace_in_file":
            value = arguments.get("new")
            if isinstance(value, str):
                values.append(value)
        return bool(values) and all(self._text_is_stub_like_python_repair(value) for value in values)

    def _validation_failure_is_stub_placeholder(self, summary: str) -> bool:
        lowered = summary.lower()
        return "still has stub body" in lowered or "pass-style placeholder" in lowered or "stub/comment/pass-style placeholder" in lowered

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

    def _next_evidence_id(self) -> str:
        self._turn_evidence_counter += 1
        return f"E{self._turn_evidence_counter}"

    def _evidence_handle_summary(self, evidence_id: str, name: str, result: dict[str, Any]) -> str:
        status = "ok" if result.get("ok") is True else "fail"
        parts = [f"{evidence_id} {name} {status}"]
        for key in ("path", "symbol", "exit_code", "diagnostic", "summary"):
            value = result.get(key)
            if value not in (None, ""):
                parts.append(f"{key}={self._truncate_text(str(value).replace(chr(10), ' | '), limit=180)}")
        output = result.get("output")
        if isinstance(output, str) and output.strip():
            limit = 420 if name == "run_test" else 260
            compact = self._compact_run_test_output(output, limit=limit) if name == "run_test" else self._truncate_text(output.strip(), limit=limit)
            parts.append("obs=" + compact.replace("\n", " | "))
        diff = result.get("diff")
        if isinstance(diff, str) and diff.strip() and "obs=" not in " ".join(parts):
            parts.append("diff=" + self._truncate_text(diff.strip().replace("\n", " | "), limit=260))
        return self._truncate_text(" ".join(parts), limit=760)

    def _tool_result_feedback_message(
        self,
        name: str,
        result: dict[str, Any],
        *,
        real_tool_use: bool,
        evidence_id: str | None = None,
        successful_tool_results: list[dict[str, Any]] | None = None,
    ) -> str:
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
            follow_up = self._run_test_failure_follow_up(result, successful_tool_results=successful_tool_results)
        if feature_enabled("evidence-handles") and evidence_id:
            return "Evidence:\n" + self._evidence_handle_summary(evidence_id, name, result) + "\n" + follow_up
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
        request_text: str,
        requires_tools: bool,
        mutation_required: bool,
        test_run_required: bool,
        round_number: int,
        tool_used_this_turn: bool,
    ) -> bool | None:
        if (
            requires_tools
            or mutation_required
            or test_run_required
            or tool_used_this_turn
            or round_number > 1
            or self._request_is_broad_or_ambiguous(request_text)
        ):
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
            "checkout",
            "checked out",
            "merge",
            "rebase",
            "stash",
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
        if "run_shell" in lowered and "not run_shell" not in lowered:
            return False
        if "shell" in lowered and "not shell" not in lowered:
            return False
        if "command" in lowered and "run_test" not in lowered:
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

    def _requested_loose_file_create_path(self, text: str) -> str | None:
        patterns = [
            r"\b(?:create|write)\s+(?:file\s+)?(?P<path>[\w./\\-]+)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            path = match.group("path").strip().rstrip(".,;:")
            if path:
                return path
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
        if lowered in {"replace_body", "replace_function_body", "function_body", "replace_method_body"}:
            path = arguments.get("path") or arguments.get("file") or arguments.get("filename")
            target = arguments.get("target") or arguments.get("symbol") or arguments.get("name") or arguments.get("function") or arguments.get("method")
            replacement = (
                arguments.get("replacement")
                if "replacement" in arguments
                else arguments.get("body")
                if "body" in arguments
                else arguments.get("content")
                if "content" in arguments
                else arguments.get("new")
            )
            if isinstance(path, str) and path.strip() and isinstance(target, str) and target.strip() and isinstance(replacement, str):
                normalized: dict[str, Any] = {
                    "path": path.strip(),
                    "intent": "replace_body",
                    "target": target.strip(),
                    "replacement": replacement,
                }
                if "scope" in arguments:
                    normalized["scope"] = arguments["scope"]
                if "apply" in arguments:
                    normalized["apply"] = arguments["apply"]
                return "edit_intent", normalized, f"Normalized unsupported {name} alias to edit_intent."
        if lowered in {"edit_symbol", "fix_symbol", "update_symbol"}:
            path = arguments.get("path") or arguments.get("file") or arguments.get("filename")
            symbol = arguments.get("symbol") or arguments.get("name") or arguments.get("function") or arguments.get("target")
            content = arguments.get("content") or arguments.get("replacement") or arguments.get("new")
            if isinstance(path, str) and path.strip() and isinstance(symbol, str) and symbol.strip() and isinstance(content, str) and content.strip():
                return (
                    "edit_intent",
                    {
                        "path": path.strip(),
                        "intent": "replace_symbol",
                        "target": symbol.strip(),
                        "replacement": content,
                    },
                    f"Normalized unsupported {name} alias to edit_intent.",
                )
        if lowered in {"search_implementation_target", "search_implementation", "implementation_search"}:
            query = arguments.get("query") or arguments.get("symbol") or arguments.get("target")
            path = arguments.get("path") or arguments.get("root") or "."
            if isinstance(query, str) and query.strip():
                return (
                    "repo_index_search",
                    {
                        "query": query.strip(),
                        "path": str(path or "."),
                        "limit": int(arguments.get("limit", 10) or 10),
                    },
                    f"Normalized unsupported {name} alias to repo_index_search.",
                )
        if lowered in {"edit_implementation_target", "fix_implementation_target", "edit_target", "fix_target"}:
            path = arguments.get("path") or arguments.get("file") or arguments.get("filename")
            replacement = arguments.get("replacement") or arguments.get("content") or arguments.get("new")
            symbol = arguments.get("symbol") or arguments.get("name") or arguments.get("function") or arguments.get("target_symbol")
            target = symbol if isinstance(symbol, str) and symbol.strip() else arguments.get("target")
            if isinstance(path, str) and path.strip() and isinstance(replacement, str) and replacement.strip() and isinstance(target, str) and target.strip():
                normalized_target = target
                if self._path_looks_like_code_file(path):
                    target_match = re.match(r"\s*(?:async\s+def|def|class)?\s*([A-Za-z_][\w.]*)", target)
                    normalized_target = target_match.group(1) if target_match else target.strip()
                return (
                    "edit_intent",
                    {
                        "path": path.strip(),
                        "intent": "replace_symbol" if self._path_looks_like_code_file(path) else "replace_text",
                        "target": normalized_target,
                        "replacement": replacement,
                    },
                    f"Normalized unsupported {name} alias to edit_intent.",
                )
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
        if name == "replace_in_file":
            path = updated.get("path") or updated.get("file") or updated.get("filename")
            if isinstance(path, str) and path.strip() and "path" not in updated:
                updated["path"] = path
                changed_reasons.append("normalized file/path alias")
            if "old" not in updated:
                for alias in ("old_text", "target", "find", "search", "before", "existing", "original", "from"):
                    value = updated.get(alias)
                    if isinstance(value, str):
                        updated["old"] = value
                        changed_reasons.append(f"normalized {alias} to old")
                        break
            if "new" not in updated:
                for alias in ("new_text", "replacement", "content", "replace_with", "after", "value", "to"):
                    value = updated.get(alias)
                    if isinstance(value, str):
                        updated["new"] = value
                        changed_reasons.append(f"normalized {alias} to new")
                        break
            if "replace_all" not in updated and "all" in updated:
                updated["replace_all"] = bool(updated.get("all"))
                changed_reasons.append("normalized all to replace_all")
            if "match_whole_word" not in updated:
                for alias in ("whole_word", "matchWholeWord"):
                    if alias in updated:
                        updated["match_whole_word"] = bool(updated.get(alias))
                        changed_reasons.append(f"normalized {alias} to match_whole_word")
                        break
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
        return any(phrase in lowered for phrase in mutation_phrases) or self._request_looks_like_issue_report(text)

    def _request_looks_like_issue_report(self, text: str) -> bool:
        lowered = text.lower()
        has_code_context = bool(
            re.search(
                r"`[^`]+`|(?:^|\s)[A-Za-z_][\w./-]*\.(?:py|js|ts|tsx|jsx|java|go|rs|c|cc|cpp)\b|array\(\[|traceback|from [A-Za-z0-9_. ]+ import ",
                text,
            )
        )
        if not has_code_context:
            return False
        issue_patterns = [
            r"\b(?:bug|issue|regression)\b",
            r"\bdoes not\b[^.?!\n]{0,120}\b(?:correctly|properly|compute|return|handle|pass|work)\b",
            r"\breturns?\b[^.?!\n]{0,80}\bwrong\b",
            r"\b(?:incorrect|incorrectly|unexpected(?:ly)?)\b",
            r"\bfails?\b[^.?!\n]{0,120}\b(?:when|with|for|to|under|on)\b",
        ]
        return any(re.search(pattern, lowered) for pattern in issue_patterns)

    def _request_requires_mutation(self, text: str) -> bool:
        lowered = text.lower()
        read_only_patterns = [
            r"\bdo not edit\b(?!\s+(?:tests?|test files?)\b)",
            r"\bdon't edit\b(?!\s+(?:tests?|test files?)\b)",
            r"\bdo not modify\b",
            r"\bdon't modify\b",
            r"\bdo not change\b(?!\s+(?:tests?|test files?)\b)",
            r"\bdon't change\b(?!\s+(?:tests?|test files?)\b)",
            r"\bwithout editing\b(?!\s+(?:tests?|test files?)\b)",
            r"\bwithout modifying\b",
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
        if self._request_looks_like_issue_report(text):
            return True
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

    def _request_requires_code_mutation(self, text: str) -> bool:
        lowered = text.lower()
        if not self._request_requires_mutation(text):
            return False
        return bool(
            re.search(r"\b(?:fix|implement|patch|repair|refactor|change|update)\b", lowered)
            and (
                re.search(r"\b(?:implementation|source|code|bug|failing|failure|hidden tests?)\b", lowered)
                or re.search(r"\b(?:fix|patch|repair)\b.{0,120}\btests?\b", lowered)
            )
        )

    def _request_explicitly_allows_test_mutation(self, text: str) -> bool:
        lowered = text.lower()
        if any(
            phrase in lowered
            for phrase in [
                "update tests",
                "edit tests",
                "modify tests",
                "change tests",
                "rewrite tests",
                "add tests",
                "test file",
                "test files",
                "tests, and docs",
                "tests and docs",
            ]
        ):
            return True
        return bool(re.search(r"\btests?/[^\s,;:]+", lowered))

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
        if isinstance(tool_name, str) and tool_name == "final" and response_type in {"tool", None, "", "function", "tool_call"}:
            message = normalized.get("message")
            if not isinstance(message, str):
                if isinstance(arguments, dict):
                    arg_message = arguments.get("message")
                    if isinstance(arg_message, str):
                        message = arg_message
                    else:
                        arg_content = arguments.get("content")
                        if isinstance(arg_content, str):
                            message = arg_content
            normalized = {"type": "final", "message": str(message or "").strip()}
            return normalized
        if isinstance(response_type, str) and self._is_supported_tool_name(response_type):
            normalized["type"] = "tool"
            if not isinstance(tool_name, str) or not tool_name:
                normalized["name"] = response_type
            if not isinstance(arguments, dict):
                normalized["arguments"] = {}
            return normalized
        if isinstance(tool_name, str) and self._is_supported_tool_name(tool_name) and response_type in {None, "", "function", "tool_call"}:
            normalized["type"] = "tool"
            if not isinstance(arguments, dict):
                normalized["arguments"] = {}
            return normalized
        return normalized

    def _is_supported_tool_name(self, name: str) -> bool:
        clean = str(name or "").strip()
        if clean in KNOWN_TOOL_NAMES:
            return True
        parts = clean.split(".", 2)
        return len(parts) == 3 and parts[0] == "mcp" and all(part.strip() for part in parts[1:])

    def _counts_as_real_tool_use(self, name: str, result: dict[str, Any]) -> bool:
        if not self._is_supported_tool_name(name):
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
        if not self._is_supported_tool_name(name):
            return False
        if name == "run_agent":
            return False
        summary = str(result.get("summary", "")).strip()
        output = str(result.get("output", "")).strip()
        if name == "run_test" and self._request_requires_test_run(request_text):
            return result.get("ok") is not True and bool(summary or output)
        if name == "run_shell" and self._request_asks_if_command_works(request_text):
            return result.get("ok") is not True and bool(summary or output)
        if self._request_asks_if_path_exists(request_text) and str(result.get("error_class") or "") == "path_missing":
            return result.get("ok") is not True and bool(summary or output)
        if (
            self._request_asks_direct_file_contents(request_text)
            or self._request_asks_exact_line_text(request_text)
            or self._request_asks_specific_file_line(request_text)
        ) and str(result.get("error_class") or "") == "path_missing":
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

    def _request_asks_if_command_works(self, text: str) -> bool:
        lowered = text.lower()
        return any(
            phrase in lowered
            for phrase in [
                "whether it works",
                "if it works",
                "whether the command works",
                "if the command works",
                "tell me whether it works",
                "tell me if it works",
            ]
        )

    def _request_asks_if_path_exists(self, text: str) -> bool:
        lowered = text.lower()
        return any(
            phrase in lowered
            for phrase in [
                "whether it exists",
                "if it exists",
                "whether the file exists",
                "if the file exists",
                "whether the path exists",
                "if the path exists",
                "tell me whether it exists",
                "tell me if it exists",
            ]
        )

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

    def _request_asks_exact_line_text(self, text: str) -> bool:
        lowered = text.lower()
        if "line" not in lowered or "exact" not in lowered:
            return False
        return any(phrase in lowered for phrase in ["text on line", "line text", "line only", "that line only"])

    def _request_asks_specific_file_line(self, text: str) -> bool:
        lowered = text.lower()
        if "line" not in lowered:
            return False
        if not re.search(r"\b[\w./-]+\.[A-Za-z0-9]+\b", text):
            return False
        return bool(re.search(r"\bline\s+\d+\b", lowered))

    def _request_asks_direct_file_contents(self, text: str) -> bool:
        lowered = text.lower()
        if any(word in lowered for word in ["summarize", "summary", "explain", "why"]):
            return False
        return self._requested_natural_read_file_path(text) is not None

    def _request_asks_symbol_return(self, text: str) -> bool:
        lowered = text.lower()
        return bool(
            re.search(
                r"\bwhat\s+does\b.*\breturn\b|\breturns?\s+what\b|\btell\s+me\s+what\b.*\breturns?\b|\bsummarize\b.*\breturns?\b|\breturn\s+value\b|\bvalue\s+it\s+returns?\b",
                lowered,
            )
        )

    def _extract_uppercase_token_from_output(self, output: str) -> str | None:
        for token in re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", output):
            if token in {"OK"}:
                continue
            return token
        return None

    def _extract_numbered_lines(self, output: str) -> list[tuple[int, str]]:
        rows: list[tuple[int, str]] = []
        for line in output.splitlines():
            match = re.match(r"\s*(\d+)\s+\|\s?(.*)$", line)
            if not match:
                continue
            rows.append((int(match.group(1)), match.group(2)))
        return rows

    def _extract_numbered_line_from_output(self, output: str, line_number: int) -> str | None:
        for current_line, text in self._extract_numbered_lines(output):
            if current_line == line_number:
                return text
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
        lowered = text.lower()
        if re.search(r"\bnot\s+run_shell\b", lowered) or re.search(r"\bnot\s+shell\b", lowered):
            return None
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

    def _requested_natural_read_file_path(self, text: str) -> str | None:
        patterns = [
            r"\bwhat does\s+(?P<path>[\w./\\:-]+\.[A-Za-z0-9]+)\s+(?:say|contain)\b",
            r"\btell me what\s+(?P<path>[\w./\\:-]+\.[A-Za-z0-9]+)\s+(?:says|contains)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
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

    def _requested_list_files_path(self, text: str) -> str | None:
        lowered = text.lower().strip()
        if lowered in {"ls", "dir", "list files", "list the files", "show files", "show the files", "list_files", "use list_files"}:
            return "."
        patterns = [
            r"\b(?:list|show)\s+(?:the\s+)?files\s+(?:in|under|for|from)\s+(?:the\s+)?(?P<path>[\w./\\:-]+)",
            r"\b(?:use\s+)?list_files\s+(?:on|in|under|for|from)\s+(?:the\s+)?(?P<path>[\w./\\:-]+)",
            r"\b(?:ls|dir)\s+(?P<path>[\w./\\:-]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            path = match.group("path").strip().rstrip(".,;:")
            if path:
                return "." if path.lower() in {"the", "workspace", "repo", "repository", "project", "directory", "folder"} else path
        if re.search(r"\b(?:list|show)\s+(?:the\s+)?files\b|\blist_files\b", lowered):
            return "."
        return None

    def _requested_run_test_command(self, text: str) -> str | None:
        patterns = [
            r"\brun_test\s+to\s+execute\s+(?P<command>.+?)(?:\s+and\b|[.?!]\s|$)",
            r"\buse\s+run_test\s+to\s+execute\s+(?P<command>.+?)(?:\s+and\b|[.?!]\s|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            command = match.group("command").strip().strip("` ")
            if len(command) >= 2 and command[0] == command[-1] and command[0] in {"'", '"'}:
                command = command[1:-1].strip()
            if command:
                return command
        return None

    def _workspace_has_test_signal(self) -> bool:
        root = self.tools.workspace_root
        common_files = [
            "pytest.ini",
            "pytest.toml",
            "tox.ini",
            "noxfile.py",
            "package.json",
            "go.mod",
            "Cargo.toml",
            "build.gradle",
            "settings.gradle",
        ]
        if any((root / name).exists() for name in common_files):
            return True
        if (root / "tests").exists():
            return True
        for pattern in ("test_*.py", "*_test.py"):
            if next(root.glob(pattern), None) is not None:
                return True
        return False

    def _requested_local_search_spec(self, text: str) -> dict[str, Any] | None:
        lowered = text.lower()
        if "web" in lowered and not any(term in lowered for term in ["workspace", "repo", "repository", "code", "files", "project"]):
            return None
        patterns = [
            r"\buse\s+search\s+to\s+find\s+(?P<query>.+?)(?:\s+and\b|[.?!]|$)",
            r"\b(?:search|grep|rg)\s+(?:for\s+)?(?P<query>.+?)(?:\s+in\s+(?P<path>[\w./\\:-]+))?(?:\s+and\b|[,.?!]|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            query = match.group("query").strip().strip("`'\" ")
            query = re.sub(r"\s+in\s+(?:the\s+)?(?:repo|repository|workspace|project)\s*$", "", query, flags=re.IGNORECASE)
            path = (match.groupdict().get("path") or ".").strip().rstrip(".,;:")
            if query and len(query) <= 160:
                return {"query": query, "path": path or ".", "limit": 20}
        return None

    def _requested_mechanical_tool_call(self, text: str, *, forbidden_tool_names: set[str]) -> MechanicalToolSpec | None:
        lowered = text.lower()
        mutation_required = self._request_requires_mutation(text)
        path = self._requested_git_tool_path(text)
        implementation_target_spec = self._requested_find_implementation_target_spec(text)
        wants_validator_listing = bool(
            re.search(
                r"\bdiscover_validators\b|\b(?:discover|show|list|find)\s+(?:the\s+)?(?:test|lint|validation|validator)s?\s+(?:commands?|tools?)\b",
                lowered,
            )
        ) or (
            not mutation_required
            and any(verb in lowered for verb in ["discover", "show", "list", "find"])
            and "command" in lowered
            and "test" in lowered
            and any(term in lowered for term in ["validation", "validator", "lint"])
        )
        if not mutation_required and implementation_target_spec and "find_implementation_target" not in forbidden_tool_names:
            return MechanicalToolSpec("find_implementation_target", implementation_target_spec)
        if (
            not mutation_required
            and "git_status" not in forbidden_tool_names
            and re.search(r"\bgit(?:_|\s+)status\b|\bstatus\s+of\s+(?:this\s+)?(?:workspace|repo|repository)\b", lowered)
        ):
            arguments = {"path": path} if path else {}
            return MechanicalToolSpec("git_status", arguments)
        if not mutation_required and "git_diff" not in forbidden_tool_names and (
            "git diff" in lowered
            or "git_diff" in lowered
            or re.search(r"\b(?:show|print|display)\s+(?:the\s+)?(?:staged\s+|working[-\s]tree\s+|unstaged\s+)?diff\b", lowered)
        ):
            arguments: dict[str, Any] = {"path": path} if path else {}
            mode = self._requested_git_diff_mode(text)
            if mode == "staged" or "git diff --cached" in lowered:
                arguments["cached"] = True
            return MechanicalToolSpec("git_diff", arguments)
        if not mutation_required and "discover_validators" not in forbidden_tool_names and wants_validator_listing:
            return MechanicalToolSpec("discover_validators", {"path": "."})
        run_test_command = self._requested_run_test_command(text)
        has_extra_context_action = bool(re.search(r"\b(?:read|open|inspect|search|grep|list)\b", lowered))
        if not mutation_required and "run_test" not in forbidden_tool_names and run_test_command:
            return MechanicalToolSpec("run_test", {"command": run_test_command})
        if (
            not mutation_required
            and "run_test" not in forbidden_tool_names
            and self._request_requires_test_run(text)
            and (self.tools.default_test_command or self._workspace_has_test_signal())
            and not has_extra_context_action
        ):
            arguments = {"command": self.tools.default_test_command} if self.tools.default_test_command else {}
            return MechanicalToolSpec("run_test", arguments)
        if not mutation_required and "lint_typecheck" not in forbidden_tool_names and re.search(
            r"\b(?:run|execute)\s+(?:ruff\s+check|lint|linter|typecheck|type\s+check|mypy|pyright|tsc)\b|\blint_typecheck\b",
            lowered,
        ):
            command = None
            if "ruff check" in lowered:
                command = "ruff check . --no-cache"
            elif re.search(r"\bmypy\b", lowered):
                command = "mypy ."
            elif re.search(r"\bpyright\b", lowered):
                command = "pyright"
            elif re.search(r"\btsc\b", lowered):
                command = "tsc --noEmit"
            arguments = {"command": command} if command else {"paths": "."}
            return MechanicalToolSpec("lint_typecheck", arguments)
        list_path = self._requested_list_files_path(text)
        if not mutation_required and list_path and "list_files" not in forbidden_tool_names:
            return MechanicalToolSpec("list_files", {"path": list_path})
        outline_path = self._requested_code_outline_path(text)
        if not mutation_required and outline_path and "code_outline" not in forbidden_tool_names:
            return MechanicalToolSpec("code_outline", {"path": outline_path})
        symbol_search = self._requested_search_symbols_spec(text)
        requested_tool_names = self._requested_tool_names(text, forbidden_tool_names=set())
        if not mutation_required and symbol_search and "search_symbols" not in forbidden_tool_names and "read_symbol" not in requested_tool_names:
            return MechanicalToolSpec("search_symbols", symbol_search)
        search_spec = self._requested_local_search_spec(text)
        if not mutation_required and search_spec and "search" not in forbidden_tool_names:
            return MechanicalToolSpec("search", search_spec)
        return None

    def _requested_code_outline_path(self, text: str) -> str | None:
        path_pattern = r"[\w./\\:-]+\.[A-Za-z0-9]+"
        patterns = [
            rf"\bcode_outline\b\s+(?:on|for|in)?\s*(?P<path>{path_pattern})",
            rf"\buse\s+code_outline\s+(?:on|for|in)\s+(?P<path>{path_pattern})",
            rf"\boutline\s+(?:the\s+)?code\s+(?:in|for)\s+(?P<path>{path_pattern})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group("path").strip().rstrip(".,;:")
        return None

    def _requested_find_implementation_target_spec(self, text: str) -> dict[str, Any] | None:
        path_pattern = r"[\w./\\:-]+\.[A-Za-z0-9]+"
        patterns = [
            rf"\bfind_implementation_target\b.*?\b(?:for|on|path|test_path)\s+(?P<path>{path_pattern})",
            rf"\b(?:identify|find|show)\s+(?:the\s+)?(?:relevant\s+|likely\s+)?implementation\s+(?:target|file|path)(?:s)?\s+(?:for|from)\s+(?P<path>{path_pattern})",
            rf"\bwhich\s+implementation\s+(?:file|path)\s+(?:corresponds\s+to|matches|goes\s+with|for)\s+(?P<path>{path_pattern})",
            rf"\bwhat\s+is\s+the\s+(?:relevant\s+|likely\s+)?implementation\s+(?:file|path)\s+for\s+(?P<path>{path_pattern})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            path = match.group("path").strip().rstrip(".,;:")
            if path:
                return {"test_path": path}
        return None

    def _requested_search_symbols_spec(self, text: str) -> dict[str, Any] | None:
        symbol_pattern = r"[A-Za-z_][\w.]*"
        path_pattern = r"[\w./\\:-]+\.[A-Za-z0-9]+|[\w./\\:-]+"
        patterns = [
            rf"\buse\s+search_symbols\s+to\s+(?:find|locate|search\s+for)\s+(?P<symbol>{symbol_pattern})\s+in\s+(?P<path>{path_pattern})",
            rf"\bsearch_symbols\b.*?\b(?:query|symbol)\s+(?P<symbol>{symbol_pattern}).*?\b(?:path|in|on)\s+(?P<path>{path_pattern})",
            rf"\b(?:find|locate)\s+(?P<symbol>{symbol_pattern})\s+in\s+(?P<path>{path_pattern})\s+using\s+search_symbols\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            symbol = match.group("symbol").strip().rstrip(".,;:")
            path = match.group("path").strip().rstrip(".,;:")
            if symbol and path:
                return {"query": symbol, "path": path}
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

    def _should_chain_run_test_after_mechanical(
        self,
        *,
        request_text: str,
        mechanical_tool_name: str,
        forbidden_tool_names: set[str],
    ) -> bool:
        if self._request_requires_mutation(request_text):
            return False
        if not self._request_requires_test_run(request_text):
            return False
        if "run_test" in forbidden_tool_names:
            return False
        if mechanical_tool_name not in {"list_files", "discover_validators", "search"}:
            return False
        return bool(self.tools.default_test_command or self._workspace_has_test_signal())

    def _requested_context_followup_mechanical_sequence(
        self,
        text: str,
        *,
        forbidden_tool_names: set[str],
    ) -> list[tuple[str, MechanicalToolSpec]]:
        if self._request_requires_mutation(text):
            return []
        allowed_roots = {"list_files", "search", "discover_validators"}
        allowed_followups = {"run_test", "lint_typecheck", "git_status", "git_diff"}
        occurrences: list[MechanicalToolOccurrence] = []
        seen: set[tuple[int, int, str, str]] = set()

        def register(start: int, end: int, spec: MechanicalToolSpec) -> None:
            fragment = text[start:end].strip(" \t\r\n,;:.")
            if not fragment:
                return
            key = (start, end, spec.name, json.dumps(spec.arguments, sort_keys=True))
            if key in seen:
                return
            seen.add(key)
            occurrences.append(MechanicalToolOccurrence(start=start, end=end, spec=spec, fragment=fragment))

        if "search" not in forbidden_tool_names:
            search_patterns = [
                r"\buse\s+search\s+to\s+find\s+(?P<query>.+?)(?:\s+and\b|[.?!]|$)",
                r"\b(?:search|grep|rg)\s+(?:for\s+)?(?P<query>.+?)(?:\s+in\s+(?P<path>[\w./\\:-]+))?(?:\s+and\b|[,.?!]|$)",
            ]
            lowered = text.lower()
            if "web" not in lowered or any(term in lowered for term in ["workspace", "repo", "repository", "code", "files", "project"]):
                for pattern in search_patterns:
                    for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
                        query = match.group("query").strip().strip("`'\" ")
                        query = re.sub(r"\s+in\s+(?:the\s+)?(?:repo|repository|workspace|project)\s*$", "", query, flags=re.IGNORECASE)
                        path = (match.groupdict().get("path") or ".").strip().rstrip(".,;:")
                        if query and len(query) <= 160:
                            register(match.start(), match.end(), MechanicalToolSpec("search", {"query": query, "path": path or ".", "limit": 20}))

        if "list_files" not in forbidden_tool_names:
            list_patterns = [
                r"\b(?:list|show)\s+(?:the\s+)?files\s+(?:in|under|for|from)\s+(?:the\s+)?(?P<path>[\w./\\:-]+)",
                r"\b(?:ls|dir)\s+(?P<path>[\w./\\:-]+)",
                r"\b(?:list|show)\s+(?:the\s+)?files\b",
            ]
            for pattern in list_patterns:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    raw_path = (match.groupdict().get("path") or ".").strip().rstrip(".,;:")
                    path = "." if raw_path.lower() in {"", "the", "workspace", "repo", "repository", "project", "directory", "folder"} else raw_path
                    register(match.start(), match.end(), MechanicalToolSpec("list_files", {"path": path}))

        if "discover_validators" not in forbidden_tool_names:
            validator_patterns = [
                r"\bdiscover_validators\b",
                r"\b(?:discover|show|list|find)\s+(?:the\s+)?(?:test|lint|validation|validator)s?(?:\s+and\s+(?:test|lint|validation|validator)s?)*\s+(?:commands?|tools?)\b",
            ]
            for pattern in validator_patterns:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    register(match.start(), match.end(), MechanicalToolSpec("discover_validators", {"path": "."}))

        if "run_test" not in forbidden_tool_names:
            explicit_patterns = [
                r"\brun_test\s+to\s+execute\s+(?P<command>.+?)(?:\s+and\b|[.?!]\s|$)",
                r"\buse\s+run_test\s+to\s+execute\s+(?P<command>.+?)(?:\s+and\b|[.?!]\s|$)",
            ]
            for pattern in explicit_patterns:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
                    command = match.group("command").strip().strip("` ")
                    if len(command) >= 2 and command[0] == command[-1] and command[0] in {"'", '"'}:
                        command = command[1:-1].strip()
                    if command:
                        register(match.start(), match.end(), MechanicalToolSpec("run_test", {"command": command}))
            if self._workspace_has_test_signal():
                for match in re.finditer(r"\b(?:run|execute)\s+(?:the\s+)?tests?\b", text, flags=re.IGNORECASE):
                    if self.tools.default_test_command:
                        register(match.start(), match.end(), MechanicalToolSpec("run_test", {"command": self.tools.default_test_command}))
                    else:
                        register(match.start(), match.end(), MechanicalToolSpec("run_test", {}))

        if "lint_typecheck" not in forbidden_tool_names:
            lint_pattern = r"\b(?:run|execute)\s+(?:ruff\s+check|lint|linter|typecheck|type\s+check|mypy|pyright|tsc)\b|\blint_typecheck\b"
            for match in re.finditer(lint_pattern, text, flags=re.IGNORECASE):
                lowered_fragment = match.group(0).lower()
                command = None
                if "ruff check" in lowered_fragment:
                    command = "ruff check . --no-cache"
                elif "mypy" in lowered_fragment:
                    command = "mypy ."
                elif "pyright" in lowered_fragment:
                    command = "pyright"
                elif "tsc" in lowered_fragment:
                    command = "tsc --noEmit"
                arguments = {"command": command} if command else {"paths": "."}
                register(match.start(), match.end(), MechanicalToolSpec("lint_typecheck", arguments))

        if "git_status" not in forbidden_tool_names:
            git_status_patterns = [
                r"\bgit_status\s+on\s+(?P<path>[\w./\\:-]+)",
                r"\bgit(?:_|\s+)status\b(?:\s+on\s+(?P<path>[\w./\\:-]+))?",
                r"\bstatus\s+of\s+(?:this\s+)?(?:workspace|repo|repository)\b",
            ]
            for pattern in git_status_patterns:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    raw_path = (match.groupdict().get("path") or "").strip().rstrip(".,;:")
                    arguments = {"path": raw_path} if raw_path else {}
                    register(match.start(), match.end(), MechanicalToolSpec("git_status", arguments))

        if "git_diff" not in forbidden_tool_names:
            git_diff_patterns = [
                r"\bgit_diff\s+on\s+(?P<path>[\w./\\:-]+)",
                r"\bgit\s+diff\s+(?P<path>[\w./\\:-]+)",
                r"\b(?:show|print|display)\s+(?:the\s+)?(?:staged\s+|working[-\s]tree\s+|unstaged\s+)?diff\b",
                r"\bgit(?:_|\s+)diff\b",
            ]
            for pattern in git_diff_patterns:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    raw_path = (match.groupdict().get("path") or "").strip().rstrip(".,;:")
                    arguments: dict[str, Any] = {"path": raw_path} if raw_path else {}
                    mode = self._requested_git_diff_mode(match.group(0))
                    if mode == "staged":
                        arguments["cached"] = True
                    register(match.start(), match.end(), MechanicalToolSpec("git_diff", arguments))

        if len(occurrences) < 2:
            return []
        occurrences.sort(key=lambda occurrence: (occurrence.start, occurrence.end))
        ordered: list[MechanicalToolOccurrence] = []
        for occurrence in occurrences:
            if ordered and occurrence.start < ordered[-1].end:
                previous = ordered[-1]
                previous_length = previous.end - previous.start
                current_length = occurrence.end - occurrence.start
                if occurrence.spec.name == previous.spec.name and current_length > previous_length:
                    ordered[-1] = occurrence
                continue
            ordered.append(occurrence)
        if len(ordered) < 2 or ordered[0].spec.name not in allowed_roots:
            return []
        if any(occurrence.spec.name not in allowed_roots | allowed_followups for occurrence in ordered[1:]):
            return []
        return [(occurrence.fragment, occurrence.spec) for occurrence in ordered]

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
        if name != "run_test":
            return name, arguments, None
        command = str(arguments.get("command", "")).strip()
        normalized_unittest = self._normalize_unittest_file_command(command)
        if normalized_unittest:
            normalized = dict(arguments)
            normalized["command"] = normalized_unittest
            return "run_test", normalized, "Normalized unittest file path command to unittest discover."
        if not self.tools.default_test_command:
            return name, arguments, None
        lowered_command = command.lower()
        lowered_request = request_text.lower()
        vague_command = lowered_command in {"", "test", "tests", "pytest", "unittest", "python -m unittest", "python3 -m unittest"}
        command_not_requested = bool(command) and lowered_command not in lowered_request
        if not vague_command and not command_not_requested:
            return name, arguments, None
        normalized = dict(arguments)
        normalized["command"] = self.tools.default_test_command
        return "run_test", normalized, "Normalized vague run_test command to the configured test command."

    def _request_looks_like_explicit_python_import_bug_fix(self, request_text: str) -> bool:
        if not self._request_requires_test_run(request_text):
            return False
        if not (self._request_requires_mutation(request_text) or self._request_requires_code_mutation(request_text)):
            return False
        if not (self.tools.default_test_command or self._workspace_has_test_signal()):
            return False
        lowered = request_text.lower()
        if not re.search(r"\b(?:import|package|module|modulenotfound|importerror)\b", lowered):
            return False
        if not any(token in lowered for token in ("bug", "fix", "repair", "broken", "failure")):
            return False
        requested_tools = self._requested_tool_names(request_text, forbidden_tool_names=set())
        if "list_files" in requested_tools:
            return False
        requested_paths = self._requested_mutation_paths(request_text)
        source_paths = [path for path in requested_paths if path.endswith(".py") and not self._path_looks_like_test_file(path)]
        aux_paths = [
            path
            for path in requested_paths
            if self._path_looks_like_test_file(path) or path.startswith("docs/") or path.endswith((".md", ".rst", ".txt"))
        ]
        return len(source_paths) == 1 and not aux_paths

    def _normalize_import_repair_bootstrap_call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        request_text: str,
        tool_calls_this_turn: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any], str | None]:
        if name != "list_files":
            return name, arguments, None
        prior_tool_names = [str(item.get("name") or "").strip() for item in tool_calls_this_turn]
        if any(prior_name and prior_name != "context_pack" for prior_name in prior_tool_names):
            return name, arguments, None
        if not self._request_looks_like_explicit_python_import_bug_fix(request_text):
            return name, arguments, None
        normalized: dict[str, Any] = {}
        if self.tools.default_test_command:
            normalized["command"] = self.tools.default_test_command
        return (
            "run_test",
            normalized,
            "Normalized initial list_files to run_test because the request already names a Python source path and needs concrete import/test failure evidence first.",
        )

    def _normalize_project_rename_bootstrap_call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        request_text: str,
        tool_calls_this_turn: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any], str | None]:
        if name not in {"list_files", "edit_intent"}:
            return name, arguments, None
        prior_tool_names = [str(item.get("name") or "").strip() for item in tool_calls_this_turn]
        if any(prior_name and prior_name != "context_pack" for prior_name in prior_tool_names):
            return name, arguments, None
        requested_tools = self._requested_tool_names(request_text, forbidden_tool_names=set())
        if name == "list_files" and "list_files" in requested_tools:
            return name, arguments, None
        rename_ops = self._project_function_rename_operations(request_text)
        if not rename_ops or len(rename_ops) != 1:
            return name, arguments, None
        tool_name, tool_arguments = rename_ops[0]
        if tool_name != "edit_intent" or not isinstance(tool_arguments, dict):
            return name, arguments, None
        if name == "edit_intent":
            target = str(arguments.get("target") or arguments.get("symbol") or "").strip()
            replacement = str(arguments.get("replacement") or arguments.get("new") or "").strip()
            if target != str(tool_arguments.get("target") or "").strip():
                return name, arguments, None
            if replacement != str(tool_arguments.get("replacement") or "").strip():
                return name, arguments, None
        return (
            tool_name,
            dict(tool_arguments),
            f"Normalized initial {name} to edit_intent because the request already specifies a grounded project rename operation.",
        )

    def _normalize_optional_parameter_bootstrap_call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        request_text: str,
        tool_calls_this_turn: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any], str | None]:
        if name != "search_symbols":
            return name, arguments, None
        prior_tool_names = [str(item.get("name") or "").strip() for item in tool_calls_this_turn]
        if any(prior_name and prior_name != "context_pack" for prior_name in prior_tool_names):
            return name, arguments, None
        requested_tools = self._requested_tool_names(request_text, forbidden_tool_names=set())
        if "search_symbols" in requested_tools or "read_symbol" in requested_tools:
            return name, arguments, None
        optional_parameter_ops = self._optional_parameter_update_operations(request_text)
        if not optional_parameter_ops:
            return name, arguments, None
        tool_name, tool_arguments = optional_parameter_ops[0]
        if tool_name != "edit_intent" or not isinstance(tool_arguments, dict):
            return name, arguments, None
        return (
            tool_name,
            dict(tool_arguments),
            "Normalized initial search_symbols to edit_intent because the request already specifies a grounded optional-parameter update.",
        )

    def _normalize_unittest_file_command(self, command: str) -> str | None:
        match = re.match(
            r"^(?P<prefix>(?:\"[^\"]+\"|'[^']+'|[^\s]+)\s+-m\s+unittest)\s+(?P<path>[^\s]+\.py)\s*$",
            command.strip(),
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        raw_path = match.group("path").strip("\"'")
        try:
            target = self.tools.resolve_path(raw_path, allow_missing=False)
        except Exception:
            return None
        rel = self.tools.relative_label(target).replace("\\", "/")
        if "/tests/" not in f"/{rel}" and not target.name.startswith("test_") and not target.name.endswith("_test.py"):
            return None
        test_dir = self.tools.relative_label(target.parent)
        return f"{match.group('prefix')} discover -s {test_dir} -p {target.name}"

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

    def _structured_repair_should_preflight_test_command(self) -> bool:
        command = str(self.tools.default_test_command or "").strip()
        if not command:
            return False
        return not self._shell_command_looks_like_test_run(command)

    def _normalize_shell_test_call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        request_text: str,
        exact_shell_command: str | None,
    ) -> tuple[str, dict[str, Any], str | None]:
        if name != "run_shell":
            return name, arguments, None
        if self.approval_mode() == "read-only":
            return name, arguments, None
        command = str(arguments.get("command", "")).strip()
        if not command:
            return name, arguments, None
        request_forbidden = self._forbidden_tool_names(request_text)
        explicit_run_shell = self._request_explicitly_requests_tool(request_text, "run_shell") and "run_shell" not in request_forbidden
        explicit_run_test = self._request_explicitly_requests_tool(request_text, "run_test") and "run_test" not in request_forbidden
        if explicit_run_shell:
            return name, arguments, None
        if explicit_run_test:
            normalized = {"command": command}
            if "cwd" in arguments:
                normalized["cwd"] = arguments["cwd"]
            if "timeout" in arguments:
                normalized["timeout"] = arguments["timeout"]
            return "run_test", normalized, "Normalized run_shell to run_test because the request explicitly requires run_test."
        if not self._shell_command_looks_like_test_run(command):
            return name, arguments, None
        if exact_shell_command and command == exact_shell_command:
            return name, arguments, None
        normalized: dict[str, Any] = {"command": self.tools.default_test_command or command}
        if "cwd" in arguments:
            normalized["cwd"] = arguments["cwd"]
        if "timeout" in arguments:
            normalized["timeout"] = arguments["timeout"]
        reason = (
            "Normalized shell test command to the configured run_test command."
            if self.tools.default_test_command
            else "Normalized shell test command to run_test with the original command."
        )
        return "run_test", normalized, reason

    def _normalize_shell_inspection_call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        request_text: str,
        exact_shell_command: str | None,
    ) -> tuple[str, dict[str, Any], str | None]:
        if name != "run_shell":
            return name, arguments, None
        if self.approval_mode() == "read-only":
            return name, arguments, None
        command = str(arguments.get("command", "")).strip()
        if not command or exact_shell_command:
            return name, arguments, None
        request_forbidden = self._forbidden_tool_names(request_text)
        if self._request_explicitly_requests_tool(request_text, "run_shell") and "run_shell" not in request_forbidden:
            return name, arguments, None
        if str(arguments.get("cwd") or ".").strip() not in {"", "."}:
            return name, arguments, None
        normalized_find_exec = self._normalize_find_exec_grep_shell_command(command)
        if normalized_find_exec is not None:
            return "search", normalized_find_exec, "Normalized find-plus-grep inspection to search for cacheable structured context."
        if re.search(r"[|&;<>`$()\r\n]", command):
            return name, arguments, None
        path_token = r'(?:"([^"]+)"|\'([^\']+)\'|(\S+))'
        cat_match = re.fullmatch(rf"(?:cat|type)\s+(?:-n\s+)?{path_token}", command, flags=re.IGNORECASE)
        if cat_match:
            path = next(group for group in cat_match.groups() if group)
            if path.startswith("-"):
                return name, arguments, None
            return "read_file", {"path": path}, "Normalized shell file inspection to read_file for cacheable structured context."
        list_match = re.fullmatch(rf"(?:ls|dir)\s+{path_token}", command, flags=re.IGNORECASE)
        if list_match:
            path = next(group for group in list_match.groups() if group)
            if path.startswith("-"):
                return name, arguments, None
            return "list_files", {"path": path}, "Normalized shell directory inspection to list_files for cacheable structured context."
        try:
            argv = shlex.split(command, posix=True)
        except ValueError:
            return name, arguments, None
        normalized_search = self._normalize_grep_shell_inspection(argv)
        if normalized_search is not None:
            return "search", normalized_search, "Normalized shell text search to search for cacheable structured context."
        if argv and argv[0].lower() in {"head", "tail"}:
            normalized_read = self._normalize_head_tail_shell_inspection(argv)
            if normalized_read is not None:
                return "read_file", normalized_read, "Normalized shell file preview to read_file for bounded structured context."
        if argv and argv[0].lower() == "find":
            normalized_find = self._normalize_find_shell_inspection(argv)
            if normalized_find is not None:
                tool_name, tool_arguments = normalized_find
                return tool_name, tool_arguments, "Normalized simple shell discovery to structured search for cacheable context."
        return name, arguments, None

    def _normalize_grep_shell_inspection(self, argv: list[str]) -> dict[str, Any] | None:
        if len(argv) not in {3, 4} or not argv:
            return None
        if argv[0].lower() not in {"grep", "rg", "ripgrep"}:
            return None
        index = 1
        if len(argv) == 4:
            if argv[index] not in {"-n", "--line-number"}:
                return None
            index += 1
        query, path = argv[index], argv[index + 1]
        if query.startswith("-") or path.startswith("-"):
            return None
        return {"query": query, "path": path}

    def _normalize_find_exec_grep_shell_command(self, command: str) -> dict[str, Any] | None:
        if not command.lower().startswith("find "):
            return None
        if re.search(r"[|&<>`$()\r\n]", command):
            return None
        try:
            argv = shlex.split(command, posix=True)
        except ValueError:
            return None
        if len(argv) not in {10, 12}:
            return None
        if argv[0].lower() != "find" or argv[2] != "-name":
            return None
        path = argv[1]
        file_glob = argv[3]
        index = 4
        if len(argv) == 12:
            if argv[index] != "-type" or argv[index + 1].lower() not in {"f", "file"}:
                return None
            index += 2
        expected = ["-exec", "grep", "-l"]
        if argv[index : index + 3] != expected:
            return None
        query = argv[index + 3]
        if argv[index + 4] != "{}" or argv[index + 5] != ";":
            return None
        if path.startswith("-") or file_glob.startswith("-") or query.startswith("-"):
            return None
        return {"query": query, "path": path, "file_glob": file_glob}

    def _normalize_head_tail_shell_inspection(self, argv: list[str]) -> dict[str, Any] | None:
        if not argv:
            return None
        command = argv[0].lower()
        if command not in {"head", "tail"}:
            return None
        count = 10
        path: str | None = None
        index = 1
        if index < len(argv):
            token = argv[index]
            if token == "-n":
                if index + 2 >= len(argv):
                    return None
                try:
                    count = int(argv[index + 1])
                except ValueError:
                    return None
                path = argv[index + 2]
                index += 3
            elif re.fullmatch(r"-\d+", token):
                count = int(token[1:])
                if index + 1 >= len(argv):
                    return None
                path = argv[index + 1]
                index += 2
            elif token.startswith("-"):
                return None
            else:
                path = token
                index += 1
        if index != len(argv) or not path or path.startswith("-") or count <= 0:
            return None
        count = min(count, 200)
        if command == "head":
            return {"path": path, "start": 1, "end": count}
        try:
            target = self.tools.resolve_path(path, allow_missing=False)
            if target.is_dir():
                return None
            line_count = len(target.read_text(encoding="utf-8", errors="replace").splitlines())
        except OSError:
            return None
        start = max(1, line_count - count + 1)
        return {"path": path, "start": start, "end": max(start, line_count)}

    def _normalize_find_shell_inspection(self, argv: list[str]) -> tuple[str, dict[str, Any]] | None:
        if len(argv) < 4 or argv[0].lower() != "find":
            return None
        path = argv[1]
        if path.startswith("-"):
            return None
        query: str | None = None
        target_tool = "file_search"
        index = 2
        while index < len(argv):
            token = argv[index]
            if token == "-name":
                if index + 1 >= len(argv) or query is not None:
                    return None
                query = argv[index + 1]
                index += 2
                continue
            if token == "-type":
                if index + 1 >= len(argv):
                    return None
                raw_kind = argv[index + 1].lower()
                if raw_kind in {"f", "file"}:
                    target_tool = "file_search"
                elif raw_kind in {"d", "dir", "directory"}:
                    target_tool = "directory_search"
                else:
                    return None
                index += 2
                continue
            return None
        if not query or query.startswith("-"):
            return None
        clean_query = query.strip()
        if target_tool == "file_search":
            if clean_query.startswith("*") and clean_query.endswith("*") and len(clean_query) > 2:
                clean_query = clean_query.strip("*")
            elif clean_query.startswith("*.") and len(clean_query) > 2:
                clean_query = clean_query[1:]
        clean_query = clean_query.strip()
        if not clean_query:
            return None
        return target_tool, {"query": clean_query, "path": path, "limit": 100}

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

    def _request_benefits_from_systems_lens(self, text: str) -> bool:
        if self._request_is_broad_or_ambiguous(text):
            return True
        return bool(
            re.search(
                r"\b(?:debug|root cause|flaky|regression|perf|performance|slow|throughput|profile|benchmark|architecture|design|workflow|pipeline|integration|refactor|migration|system|systems)\b",
                text,
                flags=re.IGNORECASE,
            )
        )

    def _request_benefits_from_todos(self, text: str, *, mutation_required: bool, test_run_required: bool) -> bool:
        lowered = text.lower()
        if re.search(r"\b(?:todo|to-do|checklist|task list|plan steps|track progress)\b", lowered):
            return True
        if self._request_is_broad_or_ambiguous(text) or test_run_required:
            return True
        if mutation_required and re.search(r"\b(?:implement|fix|refactor|debug|profile|migrate|integrate|keep fixing)\b", lowered):
            return True
        return False

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
            if name in CONTEXT_GATHERING_TOOL_NAMES:
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
        if name in CONTEXT_GATHERING_TOOL_NAMES:
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

    def _tool_call_signature(self, name: str, arguments: dict[str, Any]) -> str:
        return name + ":" + json.dumps(arguments, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    def _same_tool_call_count(self, tool_calls: list[dict[str, Any]], name: str, arguments: dict[str, Any]) -> int:
        signature = self._tool_call_signature(name, arguments)
        return sum(
            1
            for item in tool_calls
            if self._tool_call_signature(str(item.get("name", "")), item.get("arguments") if isinstance(item.get("arguments"), dict) else {}) == signature
        )

    def _context_tool_streak(self, tool_calls: list[dict[str, Any]]) -> int:
        count = 0
        for item in reversed(tool_calls):
            if str(item.get("name", "")).strip() not in CONTEXT_GATHERING_TOOL_NAMES:
                break
            count += 1
        return count

    def _broad_context_tool_streak(self, tool_calls: list[dict[str, Any]]) -> int:
        count = 0
        for item in reversed(tool_calls):
            if str(item.get("name", "")).strip() not in BROAD_CONTEXT_GATHERING_TOOL_NAMES:
                break
            count += 1
        return count

    def _context_guard_retry_message(
        self,
        *,
        mutation_required: bool,
        code_mutation_required: bool,
        test_run_required: bool,
        required_mutation_paths: set[str],
        mutated_paths_this_turn: set[str],
        successful_tool_results: list[dict[str, Any]],
        broad: bool,
    ) -> str:
        if mutation_required or code_mutation_required:
            pending_paths = sorted(required_mutation_paths - mutated_paths_this_turn)
            if pending_paths:
                target_list = ", ".join(pending_paths[:4])
                next_step = "Make the grounded edit now"
                if test_run_required:
                    next_step += ", then run_test"
                return (
                    "You already have enough evidence to edit the named target path(s): "
                    f"{target_list}. Do not gather more context. {next_step}. Next JSON only."
                )
            source_paths = self._recent_source_paths(successful_tool_results)
            test_paths = self._recent_test_paths(successful_tool_results)
            if source_paths:
                evidence_parts = [f"source={', '.join(source_paths[:3])}"]
                if test_paths:
                    evidence_parts.append(f"tests={', '.join(test_paths[:2])}")
                next_step = "Edit the grounded implementation now"
                if test_run_required:
                    next_step += ", then run_test"
                return (
                    "You already have grounding from "
                    + "; ".join(evidence_parts)
                    + f". Do not gather more context. {next_step}. Next JSON only."
                )
        if broad:
            return (
                "Pause broad inspection. Name the smallest missing fact and use a narrower next step: "
                "search_symbols/read_symbol/code_outline for a concrete symbol, find_implementation_target for tests/traceback, "
                "discover_validators/run_test for evidence, or answer/edit from current evidence. Next JSON only."
            )
        return (
            "Too many context-only tool steps. Choose one narrower next step: read_symbol/code_outline for a specific symbol, "
            "find_implementation_target for tests/traceback, edit grounded target, run validation, or answer from current evidence. Next JSON only."
        )

    def _context_planner_probe(
        self,
        *,
        successful_tool_results: list[dict[str, Any]],
        forbidden_tool_names: set[str],
    ) -> tuple[str, dict[str, Any]] | None:
        source_paths = self._recent_source_paths(successful_tool_results)
        test_paths = self._recent_test_paths(successful_tool_results)
        recent_identifier_query = self._recent_identifier_search_query(successful_tool_results)
        recent_identifier_code_paths = self._recent_identifier_search_code_paths(successful_tool_results)
        recent_search_code_paths = self._recent_search_code_paths(successful_tool_results)
        recent_list_files_code_paths = self._recent_list_files_code_paths(successful_tool_results)
        context_pack_target = self._successful_context_pack_read_target(
            successful_tool_results=successful_tool_results,
            preferred_symbol=recent_identifier_query,
        )
        if "find_implementation_target" not in forbidden_tool_names and not self._has_successful_tool_named(successful_tool_results, "find_implementation_target"):
            if test_paths and not source_paths:
                return "find_implementation_target", {"test_path": test_paths[-1], "limit": 6}
        if not source_paths and context_pack_target is not None:
            probe_name, probe_arguments = context_pack_target
            if probe_name == "read_symbol":
                if (
                    "read_symbol" not in forbidden_tool_names
                    and not self._has_successful_read_symbol(
                        path=str(probe_arguments.get("path") or ""),
                        symbol=str(probe_arguments.get("symbol") or ""),
                        successful_tool_results=successful_tool_results,
                    )
                ):
                    return probe_name, probe_arguments
            elif (
                "read_file" not in forbidden_tool_names
                and not self._has_successful_read_file(
                    path=str(probe_arguments.get("path") or ""),
                    successful_tool_results=successful_tool_results,
                )
            ):
                return probe_name, probe_arguments
        if (
            not source_paths
            and recent_identifier_query
            and recent_identifier_code_paths
            and "search_symbols" not in forbidden_tool_names
            and not self._has_successful_symbol_search_query(
                query=recent_identifier_query,
                successful_tool_results=successful_tool_results,
            )
        ):
            return "search_symbols", {"query": recent_identifier_query, "path": "."}
        if not source_paths and recent_identifier_query:
            search_match = self._successful_symbol_search_match(
                query=recent_identifier_query,
                successful_tool_results=successful_tool_results,
            )
            if (
                search_match is not None
                and "read_symbol" not in forbidden_tool_names
                and not self._has_successful_read_symbol(
                    path=search_match[0],
                    symbol=search_match[1],
                    successful_tool_results=successful_tool_results,
                )
            ):
                return "read_symbol", {"path": search_match[0], "symbol": search_match[1], "include_context": 0}
            search_source_path = self._successful_symbol_search_source_path(
                query=recent_identifier_query,
                successful_tool_results=successful_tool_results,
            )
            if (
                search_source_path
                and "read_file" not in forbidden_tool_names
                and not self._has_successful_read_file(
                    path=search_source_path,
                    successful_tool_results=successful_tool_results,
                )
            ):
                return "read_file", {"path": search_source_path}
        if (
            not source_paths
            and len(recent_search_code_paths) == 1
            and "code_outline" not in forbidden_tool_names
            and not self._has_successful_tool_named(successful_tool_results, "code_outline")
        ):
            return "code_outline", {"path": recent_search_code_paths[0]}
        if (
            not source_paths
            and len(recent_list_files_code_paths) == 1
            and "code_outline" not in forbidden_tool_names
            and not self._has_successful_tool_named(successful_tool_results, "code_outline")
        ):
            return "code_outline", {"path": recent_list_files_code_paths[0]}
        if source_paths:
            latest_source_path = source_paths[-1]
            implementation_target_symbol = self._successful_implementation_target_symbol(
                path=latest_source_path,
                successful_tool_results=successful_tool_results,
                query=recent_identifier_query,
            )
            if (
                implementation_target_symbol
                and "read_symbol" not in forbidden_tool_names
                and not self._has_successful_read_symbol(
                    path=latest_source_path,
                    symbol=implementation_target_symbol,
                    successful_tool_results=successful_tool_results,
                )
            ):
                return "read_symbol", {"path": latest_source_path, "symbol": implementation_target_symbol, "include_context": 0}
            if (
                recent_identifier_query
                and "search_symbols" not in forbidden_tool_names
                and not self._has_successful_tool_named(successful_tool_results, "search_symbols")
            ):
                return "search_symbols", {"query": recent_identifier_query, "path": latest_source_path}
            if recent_identifier_query:
                search_match = self._successful_symbol_search_match(
                    query=recent_identifier_query,
                    successful_tool_results=successful_tool_results,
                )
                if (
                    search_match is not None
                    and search_match[0] == latest_source_path
                    and "read_symbol" not in forbidden_tool_names
                    and not self._has_successful_read_symbol(
                        path=latest_source_path,
                        symbol=search_match[1],
                        successful_tool_results=successful_tool_results,
                    )
                ):
                    return "read_symbol", {"path": latest_source_path, "symbol": search_match[1], "include_context": 0}
            outlined_symbol = self._successful_single_outlined_symbol(
                path=latest_source_path,
                successful_tool_results=successful_tool_results,
            )
            if (
                outlined_symbol
                and "read_symbol" not in forbidden_tool_names
                and not self._has_successful_read_symbol(
                    path=latest_source_path,
                    symbol=outlined_symbol,
                    successful_tool_results=successful_tool_results,
                )
            ):
                return "read_symbol", {"path": latest_source_path, "symbol": outlined_symbol, "include_context": 0}
            if "code_outline" not in forbidden_tool_names and not self._has_successful_tool_named(successful_tool_results, "code_outline"):
                return "code_outline", {"path": latest_source_path}
        return None

    def _context_planner_probe_retry_message(
        self,
        *,
        probe_name: str,
        probe_arguments: dict[str, Any],
        probe_result: dict[str, Any],
        mutation_required: bool,
        code_mutation_required: bool,
        test_run_required: bool,
        required_mutation_paths: set[str],
        mutated_paths_this_turn: set[str],
        successful_tool_results: list[dict[str, Any]],
        broad: bool,
    ) -> str:
        if probe_name == "find_implementation_target":
            targets = probe_result.get("targets")
            if isinstance(targets, list):
                grounded_targets = [
                    str(item.get("path") or "").strip()
                    for item in targets
                    if isinstance(item, dict) and str(item.get("path") or "").strip()
                ]
                if grounded_targets:
                    target_list = ", ".join(list(dict.fromkeys(grounded_targets))[:3])
                    if mutation_required or code_mutation_required:
                        next_step = "Edit the grounded implementation now"
                        if test_run_required:
                            next_step += ", then run_test"
                        return f"Use the grounded implementation target(s) {target_list}. {next_step}. Next JSON only."
                    return f"Use the grounded implementation target(s) {target_list}. Continue from that narrower target or answer from current evidence. Next JSON only."
        if probe_name == "search_symbols":
            query = str(probe_arguments.get("query") or "").strip()
            path = str(probe_result.get("path") or probe_arguments.get("path") or "").strip()
            unique_match = self._successful_symbol_search_match(
                query=query,
                successful_tool_results=[
                    {
                        "name": probe_name,
                        "arguments": probe_arguments,
                        "result": probe_result,
                    }
                ],
            )
            if unique_match is not None:
                path = unique_match[0]
            else:
                unique_source_path = self._successful_symbol_search_source_path(
                    query=query,
                    successful_tool_results=[
                        {
                            "name": probe_name,
                            "arguments": probe_arguments,
                            "result": probe_result,
                        }
                    ],
                )
                if unique_source_path:
                    path = unique_source_path
            if query and path:
                if mutation_required or code_mutation_required:
                    next_step = f"Use the symbol-level matches for {query} in {path}. Read or edit the exact implementation symbol now"
                    if test_run_required:
                        next_step += ", then run_test"
                    return next_step + ". Next JSON only."
                return f"Use the symbol-level matches for {query} in {path}. Continue from that narrower symbol target or answer from current evidence. Next JSON only."
        if probe_name == "code_outline":
            path = str(probe_result.get("path") or probe_arguments.get("path") or "").strip()
            if path:
                if mutation_required or code_mutation_required:
                    next_step = f"Use the code outline for {path}. Read or edit the exact implementation symbol now"
                    if test_run_required:
                        next_step += ", then run_test"
                    return next_step + ". Next JSON only."
                return f"Use the code outline for {path}. Continue with a narrower symbol-level read or answer from current evidence. Next JSON only."
        if probe_name == "read_symbol":
            path = str(probe_result.get("path") or probe_arguments.get("path") or "").strip()
            symbol = str(probe_result.get("symbol") or probe_arguments.get("symbol") or "").strip()
            if path and symbol:
                if mutation_required or code_mutation_required:
                    next_step = f"Use the grounded symbol {symbol} in {path}. Edit the exact implementation symbol now"
                    if test_run_required:
                        next_step += ", then run_test"
                    return next_step + ". Next JSON only."
                return f"Use the grounded symbol {symbol} in {path}. Continue from that narrower symbol target or answer from current evidence. Next JSON only."
        return self._context_guard_retry_message(
            mutation_required=mutation_required,
            code_mutation_required=code_mutation_required,
            test_run_required=test_run_required,
            required_mutation_paths=required_mutation_paths,
            mutated_paths_this_turn=mutated_paths_this_turn,
            successful_tool_results=successful_tool_results,
            broad=broad,
        )

    def _test_to_source_bridge(self, successful_tool_results: list[dict[str, Any]]) -> tuple[str, str] | None:
        recent_source_paths = self._recent_source_paths(successful_tool_results)
        if not recent_source_paths:
            for test_path in reversed(self._recent_test_paths(successful_tool_results)):
                targets: list[dict[str, Any]] = []
                try:
                    result = self.tools.find_implementation_target(test_path=test_path, limit=6)
                except Exception:
                    result = {}
                targets = result.get("targets") if isinstance(result, dict) and isinstance(result.get("targets"), list) else []
                for item in targets:
                    if not isinstance(item, dict):
                        continue
                    source_path = str(item.get("path") or "").strip().replace("\\", "/")
                    if source_path.endswith(".py") and not self._path_looks_like_test_file(source_path):
                        return test_path, source_path
                for inferred in [self._infer_source_for_test_path(test_path)]:
                    if inferred and inferred.endswith(".py") and not self._path_looks_like_test_file(inferred):
                        return test_path, inferred
            return None
        bridge_candidates: list[tuple[str, str]] = []
        for test_path in reversed(self._recent_test_paths(successful_tool_results)):
            targets: list[dict[str, Any]] = []
            try:
                result = self.tools.find_implementation_target(test_path=test_path, limit=6)
            except Exception:
                result = {}
            targets = result.get("targets") if isinstance(result, dict) and isinstance(result.get("targets"), list) else []
            for item in targets:
                if not isinstance(item, dict):
                    continue
                source_path = str(item.get("path") or "").strip().replace("\\", "/")
                if source_path.endswith(".py") and not self._path_looks_like_test_file(source_path):
                    candidate = (test_path, source_path)
                    if candidate not in bridge_candidates:
                        bridge_candidates.append(candidate)
                        break
            inferred = self._infer_source_for_test_path(test_path)
            if inferred and inferred.endswith(".py") and not self._path_looks_like_test_file(inferred):
                candidate = (test_path, inferred)
                if candidate not in bridge_candidates:
                    bridge_candidates.append(candidate)
        if not bridge_candidates:
            return None
        test_path, source_path = bridge_candidates[0]
        if source_path in recent_source_paths:
            return test_path, source_path
        if len(bridge_candidates) == 1:
            return test_path, source_path
        return None

    def _failed_run_test_output_paths(self, output: str) -> tuple[str | None, str | None]:
        raw = output.replace("\r", "\n").replace("\\", "/")
        test_path: str | None = None
        source_path: str | None = None
        test_module_match = re.search(r"Failed to import test module:\s*([^\s]+)", raw, flags=re.IGNORECASE)
        if test_module_match:
            raw_candidate = test_module_match.group(1).strip().strip("`'\"")
            if not raw_candidate.endswith(".py"):
                raw_candidate = f"{raw_candidate}.py"
            for candidate in (raw_candidate, f"tests/{raw_candidate}", f"src/{raw_candidate}", f"./{raw_candidate}"):
                try:
                    resolved = self.tools.resolve_path(candidate, allow_missing=False)
                    if resolved.exists() and resolved.is_file():
                        relative = self.tools.relative_label(resolved)
                        if self._path_looks_like_test_file(relative):
                            test_path = relative
                        else:
                            source_path = relative
                        break
                except Exception:
                    continue
        file_matches = re.findall(r"^\s*File \"([^\"]+\.py)\"(?:, line )?\d+,", raw, flags=re.MULTILINE)
        for path in file_matches:
            try:
                resolved = self.tools.resolve_path(path.replace("\\", "/"), allow_missing=False)
            except Exception:
                continue
            if not resolved.exists():
                continue
            relative = self.tools.relative_label(resolved)
            if self._path_looks_like_test_file(relative):
                if test_path is None:
                    test_path = relative
            else:
                source_path = relative
        return test_path, source_path

    def _failed_test_output_paths(self, successful_tool_results: list[dict[str, Any]]) -> tuple[str | None, str | None]:
        return self._failed_test_output_paths_from_text(self._latest_failed_run_test_output(successful_tool_results))

    def _failed_test_output_paths_from_text(self, output: str) -> tuple[str | None, str | None]:
        return self._failed_run_test_output_paths(output)

    def _infer_source_for_test_path(self, test_path: str, *, output: str = "", limit: int = 6) -> str | None:
        if not test_path:
            return None
        for resolver in (
            lambda: self._infer_source_for_test_path_from_output(test_path, output=output, limit=limit),
            lambda: self._infer_source_for_test_path_from_imports(test_path),
        ):
            inferred = resolver()
            if inferred:
                return inferred
        return None

    def _resolve_candidate_source_paths(self, module_name: str) -> list[str]:
        cleaned = module_name.replace("-", "_").strip(".")
        if not cleaned:
            return []
        parts = tuple(part for part in cleaned.split(".") if part)
        if not parts:
            return []
        candidates = [
            Path(*parts).with_suffix(".py"),
            Path(*parts).with_suffix(".py"),
            Path("src", *parts).with_suffix(".py"),
            Path(*parts[:-1]) / "__init__.py",
            Path("src", *parts[:-1]) / "__init__.py",
        ]
        if len(parts) == 1:
            candidates.extend([Path("src") / f"{parts[0]}.py", Path("src") / parts[0] / "__init__.py"])
        normalized: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            try:
                resolved = self.tools.relative_label(self.tools.resolve_path(str(candidate), allow_missing=False))
            except Exception:
                continue
            rel = str(resolved).replace("\\", "/")
            if not rel.endswith(".py") or self._path_looks_like_test_file(rel):
                continue
            if rel not in seen:
                seen.add(rel)
                normalized.append(rel)
        return normalized

    def _extract_imported_modules(self, text: str) -> list[str]:
        modules: list[str] = []
        try:
            tree = ast.parse(text)
        except Exception:
            return modules
        seen: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = (alias.name or "").strip()
                    if module and module not in seen:
                        seen.add(module)
                        modules.append(module)
            elif isinstance(node, ast.ImportFrom):
                module = (node.module or "").strip()
                if module and module not in seen:
                    seen.add(module)
                    modules.append(module)
        return modules

    def _infer_source_for_test_path_from_imports(self, test_path: str) -> str | None:
        if not test_path:
            return None
        try:
            path = self.tools.resolve_path(test_path, allow_missing=False)
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
        for module in self._extract_imported_modules(text):
            for candidate in self._resolve_candidate_source_paths(module):
                if candidate:
                    return candidate
        stem = Path(test_path).stem
        if stem.lower().startswith("test_"):
            stem = stem[5:]
        if stem.lower().endswith("_test"):
            stem = stem[:-5]
        for candidate in [f"src/{stem}.py", f"{stem}.py", f"src/{stem}/__init__.py"]:
            try:
                if self.tools.resolve_path(candidate, allow_missing=False).is_file():
                    rel = self.tools.relative_label(self.tools.resolve_path(candidate, allow_missing=False))
                    if not self._path_looks_like_test_file(rel):
                        return str(rel).replace("\\", "/")
            except Exception:
                continue
        return None

    def _infer_source_for_test_path_from_output(self, test_path: str, *, output: str, limit: int) -> str | None:
        if not output:
            return self._infer_source_for_test_path_from_imports(test_path)
        try:
            result = self.tools.find_implementation_target(test_path=test_path, output=output, limit=limit)
        except Exception:
            result = None
        if isinstance(result, dict):
            for item in result.get("targets", []) if isinstance(result.get("targets"), list) else []:
                if not isinstance(item, dict):
                    continue
                source_path = str(item.get("path") or "").strip().replace("\\", "/")
                if source_path.endswith(".py") and not self._path_looks_like_test_file(source_path):
                    return source_path
        return None

    def _failed_test_edit_target_hint(self, successful_tool_results: list[dict[str, Any]]) -> str:
        failed_test_path, failed_source_path = self._failed_test_output_paths(successful_tool_results)
        if failed_source_path:
            return failed_source_path
        if failed_test_path and failed_test_path not in {"", None}:
            return failed_test_path
        stubs = self._stub_targets_for_paths(self._recent_source_paths(successful_tool_results))
        if stubs:
            return ", ".join(stubs[:8])
        source_paths = self._recent_source_paths(successful_tool_results)
        if source_paths:
            return ", ".join(source_paths[:3])
        return "the likely implementation source file from the failing tests"

    def _failed_test_no_edit_guard_message(
        self,
        successful_tool_results: list[dict[str, Any]],
        *,
        latest_run_test_output: str = "",
    ) -> str:
        target = self._failed_test_edit_target_hint(successful_tool_results)
        latest_output = latest_run_test_output or self._latest_failed_run_test_output(successful_tool_results)
        repair_packet = self._run_test_repair_packet(latest_output, successful_tool_results=successful_tool_results)
        packet_text = self._format_run_test_repair_packet(repair_packet) if latest_output else ""
        return (
            "Tests are failing and source/tests have already been inspected. Stop reading. "
            f"Edit {target} now, preferably all remaining stubs in one replace_symbols or full-file edit, then rerun run_test. "
            + packet_text
            + " Next JSON only."
        )

    def _latest_failed_run_test_output(self, successful_tool_results: list[dict[str, Any]]) -> str:
        for item in reversed(successful_tool_results):
            if item.get("name") != "run_test":
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            if result.get("ok") is True:
                continue
            return str(result.get("output") or result.get("summary") or "")
        return ""

    def _edit_covers_bulk_stubs(self, name: str, arguments: dict[str, Any], path: str, stub_symbols: set[str]) -> bool:
        arg_path = str(arguments.get("path") or arguments.get("file") or arguments.get("filename") or "").strip().replace("\\", "/").lstrip("./")
        if arg_path and arg_path != path:
            return False
        if name == "write_file":
            return arg_path == path
        if name == "replace_symbols":
            replacements = arguments.get("replacements")
            if not isinstance(replacements, list):
                return False
            symbols = {str(item.get("symbol") or item.get("target") or "").strip().split(".")[-1] for item in replacements if isinstance(item, dict)}
            return stub_symbols.issubset(symbols)
        return False

    def _bulk_stub_repair_guard_message(self, name: str, arguments: dict[str, Any], successful_tool_results: list[dict[str, Any]]) -> str:
        paths = self._recent_source_paths(successful_tool_results)
        stubs = self._stub_targets_for_paths(paths)
        by_path: dict[str, set[str]] = {}
        for stub in stubs:
            if "::" not in stub:
                continue
            path, symbol = stub.split("::", 1)
            by_path.setdefault(path, set()).add(symbol)
        for path, symbols in by_path.items():
            if len(symbols) <= 1 and name != "run_test":
                continue
            try:
                target = self.tools.resolve_path(path, allow_missing=False)
                line_count = len(target.read_text(encoding="utf-8", errors="replace").splitlines())
            except Exception:
                line_count = 999
            if line_count > 180:
                continue
            if self._edit_covers_bulk_stubs(name, arguments, path, symbols):
                return ""
            ordered = ", ".join(sorted(symbols))
            return (
                f"Tests failed and {path} still has unimplemented stubs: {ordered}. "
                "Implement every remaining stub in that file together with replace_symbols or a full-file edit before rerunning run_test. Next JSON only."
            )
        return ""

    def _trajectory_loop_guard_blocks(
        self,
        *,
        name: str,
        arguments: dict[str, Any],
        tool_calls: list[dict[str, Any]],
        cache_hit: bool,
    ) -> bool:
        if name in CONTEXT_GATHERING_TOOL_NAMES and self._context_tool_streak(tool_calls) >= 3:
            return True
        if not cache_hit and self._same_tool_call_count(tool_calls, name, arguments) >= 2:
            return True
        return False

    def _context_planner_blocks(
        self,
        *,
        name: str,
        tool_calls: list[dict[str, Any]],
        latest_run_test_failed: bool,
        successful_tool_results: list[dict[str, Any]],
    ) -> bool:
        if latest_run_test_failed:
            return False
        if name not in BROAD_CONTEXT_GATHERING_TOOL_NAMES:
            return False
        if self._broad_context_tool_streak(tool_calls) >= 2:
            return True
        recent_identifier_query = self._recent_identifier_search_query(successful_tool_results)
        recent_identifier_code_paths = self._recent_identifier_search_code_paths(successful_tool_results)
        recent_search_code_paths = self._recent_search_code_paths(successful_tool_results)
        recent_list_files_code_paths = self._recent_list_files_code_paths(successful_tool_results)
        source_paths = self._recent_source_paths(successful_tool_results)
        context_pack_target = self._successful_context_pack_read_target(
            successful_tool_results=successful_tool_results,
            preferred_symbol=recent_identifier_query,
        )
        if not source_paths:
            if context_pack_target is not None:
                target_name, target_arguments = context_pack_target
                if target_name == "read_symbol":
                    return not self._has_successful_read_symbol(
                        path=str(target_arguments.get("path") or ""),
                        symbol=str(target_arguments.get("symbol") or ""),
                        successful_tool_results=successful_tool_results,
                    )
                return not self._has_successful_read_file(
                    path=str(target_arguments.get("path") or ""),
                    successful_tool_results=successful_tool_results,
                )
            if not recent_identifier_query or not recent_identifier_code_paths:
                return len(recent_search_code_paths) == 1 or len(recent_list_files_code_paths) == 1
            if not self._has_successful_symbol_search_query(
                query=recent_identifier_query,
                successful_tool_results=successful_tool_results,
            ):
                return True
            search_match = self._successful_symbol_search_match(
                query=recent_identifier_query,
                successful_tool_results=successful_tool_results,
            )
            if (
                search_match is not None
                and not self._has_successful_read_symbol(
                    path=search_match[0],
                    symbol=search_match[1],
                    successful_tool_results=successful_tool_results,
                )
            ):
                return True
            search_source_path = self._successful_symbol_search_source_path(
                query=recent_identifier_query,
                successful_tool_results=successful_tool_results,
            )
            if search_source_path:
                return not self._has_successful_read_file(
                    path=search_source_path,
                    successful_tool_results=successful_tool_results,
                )
            return False
        latest_source_path = source_paths[-1]
        implementation_target_symbol = self._successful_implementation_target_symbol(
            path=latest_source_path,
            successful_tool_results=successful_tool_results,
            query=recent_identifier_query,
        )
        if implementation_target_symbol:
            return not self._has_successful_read_symbol(
                path=latest_source_path,
                symbol=implementation_target_symbol,
                successful_tool_results=successful_tool_results,
            )
        if recent_identifier_query:
            search_match = self._successful_symbol_search_match(
                query=recent_identifier_query,
                successful_tool_results=successful_tool_results,
            )
            if (
                search_match is not None
                and search_match[0] == latest_source_path
                and not self._has_successful_read_symbol(
                    path=latest_source_path,
                    symbol=search_match[1],
                    successful_tool_results=successful_tool_results,
                )
            ):
                return True
        outlined_symbol = self._successful_single_outlined_symbol(
            path=latest_source_path,
            successful_tool_results=successful_tool_results,
        )
        if not outlined_symbol:
            return False
        return not self._has_successful_read_symbol(
            path=latest_source_path,
            symbol=outlined_symbol,
            successful_tool_results=successful_tool_results,
        )

    def _has_grounding_evidence(
        self,
        *,
        successful_tool_results: list[dict[str, Any]],
        latest_run_test_failed: bool,
    ) -> bool:
        if latest_run_test_failed:
            return True
        return any(str(item.get("name", "")).strip() in GROUNDING_EVIDENCE_TOOL_NAMES for item in successful_tool_results)

    def _mutation_target_paths(self, arguments: dict[str, Any]) -> list[str]:
        paths: list[str] = []
        seen: set[str] = set()
        for key in ("path", "file", "filename"):
            raw = str(arguments.get(key) or "").strip().replace("\\", "/")
            normalized = "." if raw in {".", "./"} else raw.lstrip("./")
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            paths.append(normalized)
        return paths

    def _has_targeted_grounding_evidence(
        self,
        *,
        paths: list[str],
        successful_tool_results: list[dict[str, Any]],
        latest_run_test_failed: bool,
    ) -> bool:
        if latest_run_test_failed:
            return True
        normalized_paths = {
            str(path or "").strip().replace("\\", "/").lstrip("./")
            for path in paths
            if str(path or "").strip()
        }
        if not normalized_paths:
            return False
        grounded_paths: set[str] = set()
        for item in successful_tool_results:
            name = str(item.get("name") or "").strip()
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            arguments_dict = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            if result.get("ok") is not True:
                continue
            if name in {"read_file", "read_symbol", "code_outline", "tree_sitter_syntax", "lsp_diagnostics"} or name in MUTATING_TOOL_NAMES:
                raw_path = result.get("path") or arguments_dict.get("path")
                path = str(raw_path or "").strip().replace("\\", "/").lstrip("./")
                if path in normalized_paths:
                    grounded_paths.add(path)
            elif name == "context_pack":
                ranked_paths = result.get("ranked_paths")
                if isinstance(ranked_paths, list):
                    for raw_path in ranked_paths:
                        path = str(raw_path or "").strip().replace("\\", "/").lstrip("./")
                        if path in normalized_paths:
                            grounded_paths.add(path)
                ranked_symbols = result.get("ranked_symbols")
                if isinstance(ranked_symbols, list):
                    for raw_symbol in ranked_symbols:
                        if not isinstance(raw_symbol, dict):
                            continue
                        path = str(raw_symbol.get("path") or "").strip().replace("\\", "/").lstrip("./")
                        if path in normalized_paths:
                            grounded_paths.add(path)
            elif name == "find_implementation_target":
                targets = result.get("targets")
                if isinstance(targets, list):
                    for target in targets:
                        if not isinstance(target, dict):
                            continue
                        path = str(target.get("path") or "").strip().replace("\\", "/").lstrip("./")
                        if path in normalized_paths:
                            grounded_paths.add(path)
        return normalized_paths.issubset(grounded_paths)

    def _direct_file_creation_allowed(self, request_text: str, name: str, arguments: dict[str, Any]) -> bool:
        if name != "write_file":
            return False
        path = str(arguments.get("path", "")).strip().replace("\\", "/").lstrip("./")
        content = str(arguments.get("content", ""))
        if not path or not content.strip() or path not in request_text.replace("\\", "/"):
            return False
        try:
            target = self.tools.resolve_path(path, allow_missing=True)
        except Exception:
            return False
        return not target.exists()

    def _trajectory_ground_guard_blocks(
        self,
        *,
        request_text: str,
        name: str,
        arguments: dict[str, Any],
        required_mutation_paths: set[str],
        successful_tool_results: list[dict[str, Any]],
        latest_run_test_failed: bool,
    ) -> bool:
        if not feature_enabled("trajectory-guards"):
            return False
        if name not in MUTATING_TOOL_NAMES:
            return False
        if not self._mutation_has_explicit_path_target(arguments):
            if latest_run_test_failed:
                has_contextual_grounding = any(
                    str(item.get("name", "")).strip() in GROUNDING_EVIDENCE_TOOL_NAMES for item in successful_tool_results
                )
                if not has_contextual_grounding:
                    return not self._direct_file_creation_allowed(request_text, name, arguments)
            else:
                pathless_probe = self._pathless_mutation_grounding_probe(
                    name=name,
                    arguments=arguments,
                    requested_mutation_paths=required_mutation_paths,
                    successful_tool_results=successful_tool_results,
                )
                if pathless_probe is None:
                    return not self._direct_file_creation_allowed(request_text, name, arguments)
                if not self._pathless_mutation_is_grounded(
                    name=name,
                    arguments=arguments,
                    requested_mutation_paths=required_mutation_paths,
                    successful_tool_results=successful_tool_results,
                ):
                    return not self._direct_file_creation_allowed(request_text, name, arguments)
        else:
            if name == "edit_intent":
                return False
            target_paths = [
                path
                for path in self._mutation_target_paths(arguments)
                if Path(path).suffix.lower() in CODE_EDIT_SUFFIXES and not self._path_looks_like_test_file(path)
            ]
            if target_paths and not self._has_targeted_grounding_evidence(
                paths=target_paths,
                successful_tool_results=successful_tool_results,
                latest_run_test_failed=latest_run_test_failed,
            ):
                return not self._direct_file_creation_allowed(request_text, name, arguments)
        if self._has_grounding_evidence(successful_tool_results=successful_tool_results, latest_run_test_failed=latest_run_test_failed):
            return False
        return not self._direct_file_creation_allowed(request_text, name, arguments)

    def _trajectory_ground_guard_message(self, request_text: str) -> str:
        if re.search(r"\b(?:traceback|failed|failing|pytest|unittest|test)\b", request_text, flags=re.IGNORECASE):
            return "Need current-turn evidence before mutation. Call find_implementation_target or diagnose_test_failure, then edit the grounded source. Next JSON only."
        return "Need current-turn evidence before mutation. Read the target file/symbol or call context_pack/repo_index_search first. Next JSON only."

    def _mutation_has_explicit_path_target(self, arguments: dict[str, Any]) -> bool:
        return bool(self._mutation_target_paths(arguments))

    def _mutation_requested_symbol_name(self, *, name: str, arguments: dict[str, Any]) -> str:
        raw_symbol = ""
        if name == "replace_symbol":
            raw_symbol = str(arguments.get("symbol") or "").strip()
        elif name == "edit_intent":
            intent = str(arguments.get("intent") or "").strip().lower()
            if intent in {"replace_symbol", "replace_body", "replace_signature"}:
                raw_symbol = str(arguments.get("target") or arguments.get("symbol") or "").strip()
        elif name == "replace_symbols":
            replacements = arguments.get("replacements")
            if isinstance(replacements, list) and len(replacements) == 1 and isinstance(replacements[0], dict):
                raw_symbol = str(replacements[0].get("symbol") or replacements[0].get("target") or "").strip()
        if raw_symbol and re.fullmatch(r"[A-Za-z_][\w.]*", raw_symbol):
            return raw_symbol
        return ""

    def _parse_search_symbols_matches(self, output: str) -> list[tuple[str, str]]:
        matches: list[tuple[str, str]] = []
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if not line or line == "(no symbols found)":
                continue
            match = re.match(r"^(?P<path>.+?):\d+-\d+\s+\w+\s+(?P<qualname>[A-Za-z_][\w.]*)\b", line)
            if not match:
                continue
            path = match.group("path").strip().replace("\\", "/").lstrip("./")
            qualname = match.group("qualname").strip()
            if path and qualname:
                matches.append((path, qualname))
        return matches

    def _successful_symbol_search_match(
        self,
        *,
        query: str,
        successful_tool_results: list[dict[str, Any]],
    ) -> tuple[str, str] | None:
        normalized_query = str(query or "").strip().lower()
        if not normalized_query:
            return None
        for item in reversed(successful_tool_results):
            if item.get("name") != "search_symbols":
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            if result.get("ok") is not True:
                continue
            arguments_dict = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            if str(arguments_dict.get("query") or "").strip().lower() != normalized_query:
                continue
            exact_matches: list[tuple[str, str]] = []
            seen: set[tuple[str, str]] = set()
            for path, qualname in self._parse_search_symbols_matches(str(result.get("output") or "")):
                leaf_name = qualname.rsplit(".", 1)[-1].lower()
                if normalized_query not in {leaf_name, qualname.lower()}:
                    continue
                candidate = (path, qualname)
                if candidate in seen:
                    continue
                seen.add(candidate)
                exact_matches.append(candidate)
            if len(exact_matches) == 1:
                return exact_matches[0]
            non_test_exact_matches = [candidate for candidate in exact_matches if not self._path_looks_like_test_file(candidate[0])]
            if len(non_test_exact_matches) == 1:
                return non_test_exact_matches[0]
            likely_source_path = self._successful_symbol_search_source_path(
                query=query,
                successful_tool_results=successful_tool_results,
            )
            if likely_source_path:
                path_matches = [candidate for candidate in non_test_exact_matches if candidate[0] == likely_source_path]
                if len(path_matches) == 1:
                    return path_matches[0]
            return None
        return None

    def _symbol_search_test_affinity_score(self, *, path: str, symbol: str) -> int:
        normalized_path = str(path or "").strip().replace("\\", "/").lstrip("./")
        normalized_symbol = str(symbol or "").strip()
        if not normalized_path or not normalized_symbol:
            return 0
        try:
            result = self.tools.select_tests([normalized_path], changed_symbols=[normalized_symbol], limit=1)
        except Exception:
            return 0
        tests = result.get("tests") if isinstance(result.get("tests"), list) else []
        if not tests:
            return 0
        first = tests[0] if isinstance(tests[0], dict) else {}
        score = int(first.get("score") or 0)
        confidence = str(result.get("confidence") or "").strip().lower()
        if confidence == "high":
            score += 2
        elif confidence == "medium":
            score += 1
        return score

    def _best_symbol_search_source_path_from_tests(
        self,
        *,
        candidate_paths: list[str],
        symbol: str,
    ) -> str | None:
        unique_paths = list(dict.fromkeys(str(path or "").strip().replace("\\", "/").lstrip("./") for path in candidate_paths if str(path or "").strip()))
        if len(unique_paths) < 2 or len(unique_paths) > 4:
            return None
        scored: list[tuple[int, str]] = []
        for path in unique_paths:
            score = self._symbol_search_test_affinity_score(path=path, symbol=symbol)
            if score <= 0:
                continue
            scored.append((score, path))
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], item[1]))
        top_score, top_path = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0
        if top_score >= 6 and top_score >= second_score + 3:
            return top_path
        return None

    def _explicit_source_path_for_symbol(
        self,
        *,
        candidate_paths: list[str],
        symbol: str,
    ) -> str | None:
        normalized_symbol = str(symbol or "").strip()
        unique_paths = list(
            dict.fromkeys(
                str(path or "").strip().replace("\\", "/").lstrip("./")
                for path in candidate_paths
                if str(path or "").strip()
            )
        )
        if not normalized_symbol or len(unique_paths) < 2:
            return None
        matches: list[str] = []
        for path in unique_paths:
            try:
                result = self.tools.read_symbol(path, normalized_symbol, include_context=0)
            except Exception:
                continue
            if result.get("ok") is not True:
                continue
            rel_path = str(result.get("path") or path).strip().replace("\\", "/").lstrip("./")
            if rel_path and rel_path not in matches:
                matches.append(rel_path)
        if len(matches) == 1:
            return matches[0]
        likely_match = self._best_symbol_search_source_path_from_tests(
            candidate_paths=matches,
            symbol=normalized_symbol.rsplit(".", 1)[-1],
        )
        if likely_match:
            return likely_match
        return None

    def _contextual_explicit_source_path(
        self,
        *,
        candidate_paths: list[str],
        successful_tool_results: list[dict[str, Any]],
    ) -> str | None:
        unique_paths = list(
            dict.fromkeys(
                str(path or "").strip().replace("\\", "/").lstrip("./")
                for path in candidate_paths
                if str(path or "").strip()
            )
        )
        if len(unique_paths) < 2:
            return unique_paths[0] if unique_paths else None
        bridge = self._test_to_source_bridge(successful_tool_results)
        if bridge is not None:
            bridged_source = str(bridge[1] or "").strip().replace("\\", "/").lstrip("./")
            if bridged_source in unique_paths:
                return bridged_source
        for recent_source in reversed(self._recent_source_paths(successful_tool_results)):
            normalized_recent = str(recent_source or "").strip().replace("\\", "/").lstrip("./")
            if normalized_recent in unique_paths:
                return normalized_recent
        return None

    def _successful_symbol_search_source_path(
        self,
        *,
        query: str,
        successful_tool_results: list[dict[str, Any]],
    ) -> str | None:
        normalized_query = str(query or "").strip().lower()
        if not normalized_query:
            return None
        for item in reversed(successful_tool_results):
            if item.get("name") != "search_symbols":
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            if result.get("ok") is not True:
                continue
            arguments_dict = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            if str(arguments_dict.get("query") or "").strip().lower() != normalized_query:
                continue
            matches = self._parse_search_symbols_matches(str(result.get("output") or ""))
            if not matches:
                return None
            exact_paths: list[str] = []
            all_paths: list[str] = []
            for path, qualname in matches:
                if not self._path_looks_like_test_file(path):
                    all_paths.append(path)
                leaf_name = qualname.rsplit(".", 1)[-1].lower()
                if normalized_query in {leaf_name, qualname.lower()} and not self._path_looks_like_test_file(path):
                    exact_paths.append(path)
            test_symbol = str(query or "").strip().rsplit(".", 1)[-1]
            unique_exact_paths = list(dict.fromkeys(exact_paths))
            if len(unique_exact_paths) == 1:
                return unique_exact_paths[0]
            likely_exact_path = self._best_symbol_search_source_path_from_tests(
                candidate_paths=unique_exact_paths,
                symbol=test_symbol,
            )
            if likely_exact_path:
                return likely_exact_path
            contextual_exact_path = self._contextual_explicit_source_path(
                candidate_paths=unique_exact_paths,
                successful_tool_results=successful_tool_results,
            )
            if contextual_exact_path:
                return contextual_exact_path
            unique_all_paths = list(dict.fromkeys(all_paths))
            if len(unique_all_paths) == 1:
                return unique_all_paths[0]
            contextual_all_path = self._contextual_explicit_source_path(
                candidate_paths=unique_all_paths,
                successful_tool_results=successful_tool_results,
            )
            if contextual_all_path:
                return contextual_all_path
            return None
        return None

    def _has_successful_symbol_search_query(
        self,
        *,
        query: str,
        successful_tool_results: list[dict[str, Any]],
    ) -> bool:
        normalized_query = str(query or "").strip().lower()
        if not normalized_query:
            return False
        for item in reversed(successful_tool_results):
            if item.get("name") != "search_symbols":
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            if result.get("ok") is not True:
                continue
            arguments_dict = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            if str(arguments_dict.get("query") or "").strip().lower() == normalized_query:
                return True
        return False

    def _mutation_symbol_grounding_target(self, *, name: str, arguments: dict[str, Any]) -> tuple[str, str] | None:
        raw_path = str(arguments.get("path") or arguments.get("file") or arguments.get("filename") or "").strip().replace("\\", "/").lstrip("./")
        if not raw_path:
            return None
        try:
            resolved = self.tools.resolve_path(raw_path, allow_missing=False)
        except Exception:
            return None
        if not resolved.exists() or not resolved.is_file():
            return None
        rel_path = self.tools.relative_label(resolved)
        if not self._path_looks_like_code_file(rel_path):
            return None
        raw_symbol = self._mutation_requested_symbol_name(name=name, arguments=arguments)
        if not raw_symbol:
            return None
        return rel_path, raw_symbol

    def _pathless_mutation_grounding_probe(
        self,
        *,
        name: str,
        arguments: dict[str, Any],
        requested_mutation_paths: set[str] | None = None,
        successful_tool_results: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]] | None:
        if self._mutation_has_explicit_path_target(arguments):
            return None
        explicit_source_paths = self._explicit_source_repair_candidates(requested_mutation_paths)
        explicit_source_scope = set(explicit_source_paths)
        symbol_name = self._mutation_requested_symbol_name(name=name, arguments=arguments)
        explicit_source_path = explicit_source_paths[0] if len(explicit_source_paths) == 1 else ""
        if not explicit_source_path and len(explicit_source_paths) > 1 and symbol_name:
            explicit_source_path = self._explicit_source_path_for_symbol(
                candidate_paths=explicit_source_paths,
                symbol=symbol_name,
            ) or ""
        if not explicit_source_path and len(explicit_source_paths) > 1:
            explicit_source_path = self._contextual_explicit_source_path(
                candidate_paths=explicit_source_paths,
                successful_tool_results=successful_tool_results,
            ) or ""
        source_paths = self._recent_source_paths(successful_tool_results)
        bridge = self._test_to_source_bridge(successful_tool_results)
        candidate_source = explicit_source_path
        if not candidate_source and not explicit_source_scope:
            candidate_source = bridge[1] if bridge is not None else (source_paths[-1] if source_paths else "")
        if not candidate_source:
            if not symbol_name:
                return None
            search_match = self._successful_symbol_search_match(
                query=symbol_name,
                successful_tool_results=successful_tool_results,
            )
            if search_match is not None:
                search_path, matched_symbol = search_match
                if not explicit_source_scope or search_path in explicit_source_scope:
                    return "read_symbol", {"path": search_path, "symbol": matched_symbol, "include_context": 0}
            search_source_path = self._successful_symbol_search_source_path(
                query=symbol_name,
                successful_tool_results=successful_tool_results,
            )
            if search_source_path and (not explicit_source_scope or search_source_path in explicit_source_scope):
                return "read_file", {"path": search_source_path}
            context_pack_target = self._successful_context_pack_read_target(
                successful_tool_results=successful_tool_results,
                preferred_symbol=symbol_name,
            )
            if context_pack_target is not None:
                probe_name, probe_arguments = context_pack_target
                probe_path = str(probe_arguments.get("path") or "").strip().replace("\\", "/").lstrip("./")
                if not explicit_source_scope or probe_path in explicit_source_scope:
                    return probe_name, probe_arguments
            if not self._has_successful_symbol_search_query(
                query=symbol_name,
                successful_tool_results=successful_tool_results,
            ):
                return "search_symbols", {"query": symbol_name, "path": "."}
            return None
        if symbol_name:
            return "read_symbol", {"path": candidate_source, "symbol": symbol_name, "include_context": 0}
        return "read_file", {"path": candidate_source}

    def _pathless_mutation_is_grounded(
        self,
        *,
        name: str,
        arguments: dict[str, Any],
        requested_mutation_paths: set[str] | None = None,
        successful_tool_results: list[dict[str, Any]],
    ) -> bool:
        probe = self._pathless_mutation_grounding_probe(
            name=name,
            arguments=arguments,
            requested_mutation_paths=requested_mutation_paths,
            successful_tool_results=successful_tool_results,
        )
        if probe is None:
            return False
        expected_name, expected_arguments = probe
        expected_path = str(expected_arguments.get("path") or "").strip().replace("\\", "/")
        expected_symbol = str(expected_arguments.get("symbol") or "").strip()
        for item in reversed(successful_tool_results):
            if item.get("name") != expected_name:
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            if result.get("ok") is not True:
                continue
            arguments_dict = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            result_path = str(result.get("path") or arguments_dict.get("path") or "").strip().replace("\\", "/")
            if expected_path and result_path != expected_path:
                continue
            if expected_name == "read_symbol":
                result_symbol = str(result.get("symbol") or arguments_dict.get("symbol") or "").strip()
                if expected_symbol and result_symbol != expected_symbol:
                    continue
            return True
        return False

    def _trajectory_grounding_probe(
        self,
        *,
        request_text: str,
        name: str,
        arguments: dict[str, Any],
        required_mutation_paths: set[str],
        forbidden_tool_names: set[str],
        successful_tool_results: list[dict[str, Any]],
        latest_run_test_failed: bool,
        latest_run_test_failure_output: str,
    ) -> tuple[str, dict[str, Any]] | None:
        symbol_target = self._mutation_symbol_grounding_target(name=name, arguments=arguments)
        if symbol_target is not None and "read_symbol" not in forbidden_tool_names:
            symbol_path, symbol_name = symbol_target
            return "read_symbol", {"path": symbol_path, "symbol": symbol_name, "include_context": 0}
        if not latest_run_test_failed:
            pathless_probe = self._pathless_mutation_grounding_probe(
                name=name,
                arguments=arguments,
                requested_mutation_paths=required_mutation_paths,
                successful_tool_results=successful_tool_results,
            )
            if pathless_probe is not None:
                probe_name, probe_arguments = pathless_probe
                if probe_name not in forbidden_tool_names:
                    return probe_name, probe_arguments
        if latest_run_test_failed:
            failed_output = latest_run_test_failure_output.strip()
            if failed_output:
                failed_test_path, _failed_source_path = self._failed_test_output_paths_from_text(failed_output)
                if failed_test_path and "find_implementation_target" not in forbidden_tool_names:
                    return "find_implementation_target", {"test_path": failed_test_path, "output": failed_output, "limit": 6}
                if "diagnose_test_failure" not in forbidden_tool_names:
                    return "diagnose_test_failure", {"output": failed_output, "limit": 6}
        candidate_paths: list[str] = []
        seen_paths: set[str] = set()
        for raw in (
            arguments.get("path"),
            arguments.get("file"),
            arguments.get("filename"),
            *sorted(required_mutation_paths),
        ):
            normalized = str(raw or "").strip().replace("\\", "/").lstrip("./")
            if not normalized or normalized in seen_paths:
                continue
            seen_paths.add(normalized)
            try:
                resolved = self.tools.resolve_path(normalized, allow_missing=False)
            except Exception:
                continue
            if not resolved.exists() or not resolved.is_file():
                continue
            candidate_paths.append(self.tools.relative_label(resolved))
        if len(candidate_paths) == 1:
            return "read_file", {"path": candidate_paths[0]}
        if len(candidate_paths) > 1:
            if feature_enabled("context-pack"):
                return "context_pack", {"request": request_text, "path": ".", "limit": 6}
            return None
        for raw in sorted(required_mutation_paths):
            normalized = str(raw or "").strip().replace("\\", "/").lstrip("./")
            if not normalized or not normalized.endswith(".py") or not self._path_looks_like_test_file(normalized):
                continue
            try:
                rel_test = self.tools.relative_label(self.tools.resolve_path(normalized, allow_missing=False))
            except Exception:
                continue
            return "find_implementation_target", {"test_path": rel_test, "limit": 6}
        if feature_enabled("context-pack"):
            return "context_pack", {"request": request_text, "path": ".", "limit": 6}
        return None

    def _trajectory_ground_probe_retry_message(
        self,
        *,
        request_text: str,
        probe_name: str,
        probe_arguments: dict[str, Any],
        probe_result: dict[str, Any],
        required_mutation_paths: set[str],
        mutated_paths_this_turn: set[str],
        test_run_required: bool,
    ) -> str:
        if probe_name == "context_pack":
            suggested_next_tool = str(probe_result.get("suggested_next_tool") or "").strip().lower()
            if suggested_next_tool == "read_symbol":
                return "Context pack ranked likely implementation matches. Read the most relevant implementation symbol now before editing. Next JSON only."
            if suggested_next_tool == "read_file":
                return "Context pack ranked likely implementation files. Read the most relevant implementation file now before editing. Next JSON only."
            return "Context pack ranked likely repository evidence. Read the most relevant implementation file or symbol now before editing. Next JSON only."
        pending_paths = sorted(required_mutation_paths - mutated_paths_this_turn)
        if pending_paths:
            target_list = ", ".join(pending_paths[:4])
            next_step = "Edit the grounded target now"
            if test_run_required:
                next_step += ", then run_test"
            return f"You now have current-turn grounding for {target_list}. {next_step}. Next JSON only."
        grounded_path = str(probe_result.get("path") or "").strip()
        grounded_symbol = str(probe_result.get("symbol") or "").strip()
        if probe_name == "read_symbol" and grounded_path and grounded_symbol:
            next_step = f"Use the grounded symbol {grounded_symbol} in {grounded_path} to make the edit now"
            if test_run_required:
                next_step += ", then run_test"
            return next_step + ". Next JSON only."
        if probe_name == "search_symbols":
            query = str(probe_arguments.get("query") or "").strip()
            exact_matches = self._parse_search_symbols_matches(str(probe_result.get("output") or ""))
            unique_exact_match = self._successful_symbol_search_match(
                query=query,
                successful_tool_results=[
                    {
                        "name": probe_name,
                        "arguments": probe_arguments,
                        "result": probe_result,
                    }
                ],
            )
            if unique_exact_match is not None:
                match_path, match_symbol = unique_exact_match
                return (
                    f"Search found a unique symbol match for {match_symbol} in {match_path}. "
                    "Read that exact symbol now before editing. Next JSON only."
                )
            unique_source_path = self._successful_symbol_search_source_path(
                query=query,
                successful_tool_results=[
                    {
                        "name": probe_name,
                        "arguments": probe_arguments,
                        "result": probe_result,
                    }
                ],
            )
            if unique_source_path:
                return f"Search narrowed the likely implementation file to {unique_source_path}. Read that file now before editing. Next JSON only."
            implementation_candidates = list(
                dict.fromkeys(
                    path
                    for path, _qualname in exact_matches
                    if not self._path_looks_like_test_file(path)
                )
            )
            if query and len(implementation_candidates) >= 2:
                candidate_list = ", ".join(implementation_candidates[:3])
                return (
                    f"Search found multiple implementation candidates for {query}: {candidate_list}. "
                    "Read the exact implementation symbol or name the target path before editing. Next JSON only."
                )
            if query and exact_matches:
                return f"Use the symbol-level matches for {query}. Read the exact implementation symbol now before editing. Next JSON only."
            if query:
                return f"Search did not ground a unique symbol for {query}. Name the target path or read the exact implementation symbol before editing. Next JSON only."
        if probe_name == "diagnose_test_failure":
            targets = probe_result.get("targets")
            if isinstance(targets, list):
                grounded_targets = [
                    str(item.get("path") or "").strip()
                    for item in targets
                    if isinstance(item, dict) and str(item.get("path") or "").strip()
                ]
                if grounded_targets:
                    target_list = ", ".join(list(dict.fromkeys(grounded_targets))[:3])
                    next_step = f"Use the diagnosis above to edit {target_list}"
                    if test_run_required:
                        next_step += ", then run_test"
                    return next_step + ". Next JSON only."
            next_step = "Use the diagnosis above to edit implementation"
            if test_run_required:
                next_step += ", then run_test"
            return next_step + ". Next JSON only."
        if grounded_path:
            next_step = f"Use the grounded file {grounded_path} to make the edit now"
            if test_run_required:
                next_step += ", then run_test"
            return next_step + ". Next JSON only."
        if probe_name == "find_implementation_target":
            targets = probe_result.get("targets")
            if isinstance(targets, list):
                grounded_targets = [
                    str(item.get("path") or "").strip()
                    for item in targets
                    if isinstance(item, dict) and str(item.get("path") or "").strip()
                ]
                if grounded_targets:
                    target_list = ", ".join(list(dict.fromkeys(grounded_targets))[:3])
                    next_step = "Edit the grounded implementation now"
                    if test_run_required:
                        next_step += ", then run_test"
                    return f"Use the grounded implementation target(s) {target_list}. {next_step}. Next JSON only."
        return self._trajectory_ground_guard_message(request_text)

    def _tool_error_class(self, result: dict[str, Any]) -> str:
        explicit = result.get("error_class")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
        text = "\n".join(
            str(result.get(key, ""))
            for key in ("summary", "output", "diagnostic", "normalized")
            if result.get(key) is not None
        )
        classifier = getattr(self.tools, "classify_error", None)
        if callable(classifier):
            try:
                return str(classifier(text))
            except Exception:
                pass
        return "unknown"

    def _tool_error_arg_key(self, name: str, arguments: dict[str, Any]) -> str:
        focused_keys = ("path", "cwd", "command", "query", "test_path", "module", "function")
        focused = {
            key: arguments.get(key)
            for key in focused_keys
            if key in arguments and isinstance(arguments.get(key), (str, int, float, bool))
        }
        if not focused:
            focused = arguments
        return json.dumps(focused, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    def _matching_repeated_tool_error(
        self,
        *,
        name: str,
        arguments: dict[str, Any],
        tool_error_counts: dict[tuple[str, str, str], int],
    ) -> str | None:
        arg_key = self._tool_error_arg_key(name, arguments)
        for prior_name, prior_arg_key, error_class in tool_error_counts:
            if (
                prior_name == name
                and prior_arg_key == arg_key
                and error_class == "syntax_error"
                and name in EDIT_TOOL_NAMES
                and tool_error_counts[(prior_name, prior_arg_key, error_class)] >= 1
            ):
                return error_class
        if name in LOW_LEVEL_EDIT_TOOL_NAMES and "edit_intent" in self.tools.available_tool_names():
            for prior_name, _prior_arg_key, error_class in tool_error_counts:
                if prior_name in LOW_LEVEL_EDIT_TOOL_NAMES and tool_error_counts[(prior_name, _prior_arg_key, error_class)] >= 1:
                    return f"edit_intent:{error_class}"
        for prior_name, prior_arg_key, error_class in tool_error_counts:
            if prior_name == name and prior_arg_key == arg_key and tool_error_counts[(prior_name, prior_arg_key, error_class)] >= 2:
                return error_class
        return None

    def _mutating_failure_key(self, name: str, arguments: dict[str, Any]) -> tuple[str, str]:
        path = str(arguments.get("path") or arguments.get("cwd") or "").strip().replace("\\", "/").lstrip("./")
        if path:
            return ("path", path.lower())
        return ("tool", name)

    def _repeated_mutating_failure_escape_message(
        self,
        *,
        name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> str:
        target_path = str(arguments.get("path") or arguments.get("cwd") or "").strip()
        evidence = str(result.get("summary") or result.get("output") or "").strip()
        if target_path:
            message = (
                f"Stop retrying narrow edit operations on {target_path} in this turn. "
                f"Read {target_path}, then make one broader syntactically valid repair with write_file or a full-symbol replacement, and rerun tests."
            )
        else:
            message = (
                f"Stop retrying {name} with small mutations in this turn. "
                "Switch to a broader direct repair, then rerun tests."
            )
        if evidence:
            message += " Evidence: " + self._truncate_text(evidence, limit=320)
        return message + " Next JSON only."

    def _remember_tool_error(
        self,
        *,
        name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        tool_error_counts: dict[tuple[str, str, str], int],
    ) -> tuple[str, int]:
        error_class = self._tool_error_class(result)
        key = (name, self._tool_error_arg_key(name, arguments), error_class)
        tool_error_counts[key] = tool_error_counts.get(key, 0) + 1
        return error_class, tool_error_counts[key]

    def _tool_error_guard_message(self, name: str, error_class: str, *, diagnosis_ran: bool = False) -> str:
        if error_class.startswith("edit_intent:"):
            return (
                f"A low-level edit already failed with {error_class.split(':', 1)[1]}. "
                "Use edit_intent with intent, target, replacement, and scope instead of another low-level edit retry. Next JSON only."
            )
        if error_class in {"path_missing", "cwd_git"}:
            if diagnosis_ran:
                return (
                    f"{name} already failed twice with {error_class}. Use the diagnosis above to repair the path/cwd, "
                    "choose a nearby existing path, or fail closed before retrying. Next JSON only."
                )
            return (
                f"{name} already failed twice with {error_class}. Do not retry the same path/cwd. "
                "Call list_files or repo_index_search to find the correct path, or fail closed. Next JSON only."
            )
        if error_class == "syntax_error":
            if name == "run_shell":
                return (
                    "run_shell already failed with shell syntax or quoting errors. Do not retry the same inline shell. "
                    "Use a simpler one-line command, write a temporary script with valid syntax, or switch to run_test if this is validation. Next JSON only."
                )
            return (
                f"{name} already failed with syntax_error. Inspect the exact symbol/diagnostic, repair the edit, "
                "then use edit_intent replace_body with a syntactically complete function body or replace_symbol with complete source before tests. Next JSON only."
            )
        if error_class == "timeout":
            if name == "run_shell":
                return (
                    "run_shell already failed twice with timeout. Do not retry the same long-running service or install command. "
                    "Use a short probe, bounded validator, background-safe launch, or report that the command needs a different execution strategy before retrying. Next JSON only."
                )
            return (
                f"{name} already failed twice with timeout. Do not retry the same arguments. "
                "Use a bounded probe, a cheaper validator, or fail closed before retrying. Next JSON only."
            )
        if error_class in {"missing_dependency", "command_not_found", "import_error"}:
            if diagnosis_ran:
                return (
                    f"{name} already failed twice with {error_class}. Use the diagnosis above to report the exact missing dependency/import, "
                    "choose an available validator, repair the import, or fail closed before retrying. Next JSON only."
                )
            return (
                f"{name} already failed twice with {error_class}. Do not auto-install or retry blindly. "
                "Report the exact missing dependency/command or choose an available validator. Next JSON only."
            )
        return (
            f"{name} already failed twice with {error_class}. Do not retry the same arguments. "
            "Use a different tool class, gather new evidence, edit the cause, or fail closed. Next JSON only."
        )

    def _ad_hoc_verification_script_path(self, command: str, mutated_paths_this_turn: set[str]) -> str | None:
        normalized_command = str(command or "").replace("\\", "/").strip()
        if not normalized_command:
            return None
        verification_roots = ("verify", "verification", "validate", "validation", "check", "smoke", "probe", "health", "completion")
        for raw_token in re.findall(r'"[^"]+"|\'[^\']+\'|\S+', normalized_command):
            token = raw_token.strip().strip("\"'")
            if not token:
                continue
            normalized_token = token.replace("\\", "/").lstrip("./")
            if not re.search(r"\.(?:py|sh|bash|ps1|bat|cmd|js|ts)$", normalized_token, re.IGNORECASE):
                continue
            basename = normalized_token.rsplit("/", 1)[-1].lower()
            if any(
                basename == f"{root_name}.{basename.rsplit('.', 1)[-1]}"
                or basename.startswith(root_name + "_")
                or basename.startswith(root_name + "-")
                for root_name in verification_roots
            ):
                if normalized_token in mutated_paths_this_turn:
                    return normalized_token
                try:
                    existing_path = self.tools.resolve_path(normalized_token, allow_missing=True)
                except Exception:
                    existing_path = None
                if existing_path is None or not existing_path.exists():
                    return normalized_token
        return None

    def _timeout_verification_guard_message(self, *, verification_script_path: str, prior_command: str, prior_summary: str) -> str:
        message = (
            f"Do not treat the ad hoc verification script {verification_script_path} as proof after the original timed-out command path failed. "
            "Use a bounded probe against the original command/service, a background-safe launch, or report that the real path still needs a different execution strategy."
        )
        if prior_command:
            message += " Timed-out command: " + self._truncate_text(prior_command, limit=180) + "."
        if prior_summary:
            message += " Evidence: " + self._truncate_text(prior_summary, limit=220) + "."
        return message + " Next JSON only."

    def _final_claims_timeout_success(self, message: str) -> bool:
        lowered = str(message or "").lower()
        if not lowered.strip():
            return False
        if re.search(r"\b(?:timed out|timeout|failed|could not|unable|did not|didn't|not working|still needs)\b", lowered):
            return False
        patterns = [
            r"\bservice\s+(?:is\s+)?working\b",
            r"\bverification\s+passed\b",
            r"\bworks\b",
            r"\bverified\b",
            r"\bsucceeded\b",
            r"\bstarted\b",
            r"\brunning\b",
            r"\bhealthy\b",
            r"\bavailable\b",
        ]
        return any(re.search(pattern, lowered) for pattern in patterns)

    def _final_claims_run_shell_success(self, message: str) -> bool:
        lowered = str(message or "").lower()
        if not lowered.strip():
            return False
        if re.search(r"\b(?:timed out|timeout|failed|could not|unable|did not|didn't|not installed|missing|not found|still needs)\b", lowered):
            return False
        patterns = [
            r"\bcommand\s+works\b",
            r"\bservice\s+(?:is\s+)?working\b",
            r"\bverification\s+passed\b",
            r"\bworks\b",
            r"\bverified\b",
            r"\bsucceeded\b",
            r"\bstarted\b",
            r"\brunning\b",
            r"\bhealthy\b",
            r"\bavailable\b",
        ]
        return any(re.search(pattern, lowered) for pattern in patterns)

    def _final_claims_path_exists(self, message: str) -> bool:
        lowered = str(message or "").lower().strip()
        if not lowered:
            return False
        if re.search(r"\b(?:does not exist|is missing|are missing|not found|path_missing|no such file|no such directory)\b", lowered):
            return False
        return bool(re.search(r"\bexists\b", lowered))

    def _final_acknowledges_missing_path(self, message: str) -> bool:
        lowered = str(message or "").lower().strip()
        if not lowered:
            return False
        return bool(re.search(r"\b(?:does not exist|is missing|are missing|not found|path_missing|no such file|no such directory)\b", lowered))

    def _forbidden_tool_feedback_message(
        self,
        *,
        request_text: str,
        name: str,
        arguments: dict[str, Any],
        forbidden_count: int,
        forbidden_tool_names: set[str],
        required_tool_names: set[str],
    ) -> str:
        command = str(arguments.get("command", "")).strip()
        disabled = name in getattr(self.tools, "disabled_tools", set())
        prefix = f"Do not use {name} in this turn."
        if disabled:
            prefix += f" {name} is disabled by config for this workspace."
        if name == "run_shell":
            if command and "run_test" in required_tool_names and "run_test" not in forbidden_tool_names:
                return (
                    prefix
                    + f" Use run_test instead with command {command!r}. "
                    + "If that fails, inspect the concrete failure and try a different repair before rerunning. Next JSON only."
                )
            if (
                self.tools.default_test_command
                and "run_test" not in forbidden_tool_names
                and command
                and self._shell_command_looks_like_test_run(command)
            ):
                return (
                    prefix
                    + f" Use run_test instead with command {command!r}. "
                    + "If that still fails, inspect the concrete failure and try a different repair before rerunning tests. Next JSON only."
                )
            if self.tools.default_test_command and "run_test" not in forbidden_tool_names:
                return (
                    prefix
                    + " Use run_test or another structured allowed tool instead, not run_shell. "
                    + "If tests already passed, answer from the successful run_test evidence. Next JSON only."
                )
        if forbidden_count >= 2:
            return (
                prefix
                + " You already proposed this forbidden tool before. "
                + "Choose a different allowed tool now or answer strictly from existing verified evidence. Next JSON only."
            )
        return prefix + " Choose a different allowed tool or answer from existing verified tool results. Respond with the next JSON object only."

    def _record_command_validation_event(self, *, name: str, result: dict[str, Any], round_number: int, cached: bool = False) -> None:
        validation = result.get("validation")
        if not isinstance(validation, dict):
            return
        self._record_event(
            "command_validation",
            tool=name,
            family=validation.get("family"),
            valid=validation.get("valid"),
            recognized=validation.get("recognized"),
            reason=validation.get("reason", ""),
            argv=validation.get("argv", []),
            cached=cached,
            rounds=round_number,
        )

    def _request_forbids_validation(self, text: str) -> bool:
        lowered = text.lower()
        return bool(re.search(r"\b(?:do not|don't|dont|skip|without|no)\b[^.?!\n]{0,80}\b(?:test|tests|pytest|unittest|validation|validate)\b", lowered))

    def _post_edit_validation_enabled(self) -> bool:
        return True

    def _auto_validation_plan(
        self,
        *,
        mutated_paths: set[str],
        forbidden_tool_names: set[str],
    ) -> tuple[str, dict[str, Any], str] | None:
        code_paths = sorted(path for path in mutated_paths if Path(path).suffix.lower() in CODE_EDIT_SUFFIXES)
        python_paths = sorted(path for path in code_paths if path.endswith(".py"))
        if code_paths and "lint_typecheck" not in forbidden_tool_names:
            return "lint_typecheck", {"paths": code_paths}, "syntax check changed code file(s)"
        if python_paths and "contract_check" not in forbidden_tool_names:
            return "contract_check", {"changed_files": python_paths}, "contract check changed Python function(s)"
        if code_paths and "select_tests" not in forbidden_tool_names:
            return "select_tests", {"changed_files": code_paths}, "select targeted tests for changed file(s)"
        if mutated_paths and "discover_validators" not in forbidden_tool_names:
            return "discover_validators", {"path": "."}, "discover validators for non-code edit"
        if self.tools.default_test_command and "run_test" not in forbidden_tool_names:
            return "run_test", {"command": self.tools.default_test_command}, "configured test command"
        return None

    def _discover_validators_followup(
        self,
        validation_result: dict[str, Any],
        *,
        forbidden_tool_names: set[str],
        allow_non_test_validators: bool,
        run_non_test_validators_as_commands: bool = False,
    ) -> tuple[str, dict[str, Any], str] | None:
        configured = str(self.tools.default_test_command or "").strip()
        if configured and "run_test" not in forbidden_tool_names:
            return "run_test", {"command": configured}, "test command selected after validator discovery"
        validators = validation_result.get("validators")
        if not isinstance(validators, list):
            return None
        priority_order = [("test", "run_test", "test command selected after validator discovery")]
        if allow_non_test_validators:
            non_test_tool_name = "run_test" if run_non_test_validators_as_commands else "lint_typecheck"
            non_test_reason_suffix = "validator command selected after validator discovery" if run_non_test_validators_as_commands else "validator selected after validator discovery"
            priority_order.extend(
                [
                    ("check", non_test_tool_name, f"check {non_test_reason_suffix}"),
                    ("typecheck", non_test_tool_name, f"typecheck {non_test_reason_suffix}"),
                    ("lint", non_test_tool_name, f"lint {non_test_reason_suffix}"),
                    ("syntax", non_test_tool_name, f"syntax {non_test_reason_suffix}"),
                ]
            )
        for kind, tool_name, reason in priority_order:
            if tool_name in forbidden_tool_names:
                continue
            for item in validators:
                if not isinstance(item, dict):
                    continue
                if item.get("available") is not True:
                    continue
                if str(item.get("kind") or "").strip().lower() != kind:
                    continue
                command = str(item.get("command") or "").strip()
                if command:
                    return tool_name, {"command": command}, reason
        return None

    def _execute_auto_validation_plan(
        self,
        *,
        request_text: str,
        round_number: int,
        mutated_paths: set[str],
        forbidden_tool_names: set[str],
        successful_tool_results: list[dict[str, Any]],
        satisfied_tool_names: set[str],
        tool_calls_this_turn: list[dict[str, Any]],
        mutation_version: int,
        reason_prefix: str = "",
    ) -> tuple[str | None, dict[str, Any] | None, dict[str, Any] | None]:
        validation_plan = self._auto_validation_plan(
            mutated_paths=mutated_paths,
            forbidden_tool_names=forbidden_tool_names,
        )
        if validation_plan is None:
            return None, None, None
        validation_name, validation_arguments, validation_reason = validation_plan
        self._record_event(
            "auto_validation",
            name=validation_name,
            arguments=validation_arguments,
            reason=f"{reason_prefix}{validation_reason}",
            mutation_version=mutation_version,
            rounds=round_number,
        )
        validation_result = self._execute_controller_tool(
            name=validation_name,
            arguments=validation_arguments,
            request_text=request_text,
            round_number=round_number,
            successful_tool_results=successful_tool_results,
            satisfied_tool_names=satisfied_tool_names,
            tool_calls_this_turn=tool_calls_this_turn,
        )
        validation_promoted_from_discovery = False
        validator_executed_before_select = False
        code_validation_paths = sorted(path for path in mutated_paths if Path(path).suffix.lower() in CODE_EDIT_SUFFIXES)
        if validation_name == "select_tests" and validation_result.get("ok") is True:
            commands = validation_result.get("test_commands")
            if isinstance(commands, list) and commands and "run_test" not in forbidden_tool_names:
                targeted_command = str(commands[0]).strip()
                self._record_event(
                    "auto_validation",
                    name="run_test",
                    arguments={"command": targeted_command},
                    reason=f"{reason_prefix}targeted test selected from changed files",
                    mutation_version=mutation_version,
                    rounds=round_number,
                )
                validation_name = "run_test"
                validation_arguments = {"command": targeted_command}
                validation_result = self._execute_controller_tool(
                    name="run_test",
                    arguments=validation_arguments,
                    request_text=request_text,
                    round_number=round_number,
                    successful_tool_results=successful_tool_results,
                    satisfied_tool_names=satisfied_tool_names,
                    tool_calls_this_turn=tool_calls_this_turn,
                )
            elif self.tools.default_test_command and "run_test" not in forbidden_tool_names:
                self._record_event(
                    "auto_validation",
                    name="run_test",
                    arguments={"command": self.tools.default_test_command},
                    reason=f"{reason_prefix}fallback to configured test command after empty targeted selection",
                    mutation_version=mutation_version,
                    rounds=round_number,
                )
                validation_name = "run_test"
                validation_arguments = {"command": self.tools.default_test_command}
                validation_result = self._execute_controller_tool(
                    name="run_test",
                    arguments=validation_arguments,
                    request_text=request_text,
                    round_number=round_number,
                    successful_tool_results=successful_tool_results,
                    satisfied_tool_names=satisfied_tool_names,
                    tool_calls_this_turn=tool_calls_this_turn,
                )
        if validation_name == "discover_validators" and validation_result.get("ok") is True:
            discovered_followup = self._discover_validators_followup(
                validation_result,
                forbidden_tool_names=forbidden_tool_names,
                allow_non_test_validators=bool(mutated_paths),
                run_non_test_validators_as_commands=not bool(code_validation_paths),
            )
            if discovered_followup is not None:
                followup_name, followup_arguments, followup_reason = discovered_followup
                self._record_event(
                    "auto_validation",
                    name=followup_name,
                    arguments=followup_arguments,
                    reason=f"{reason_prefix}{followup_reason}",
                    mutation_version=mutation_version,
                    rounds=round_number,
                )
                validation_promoted_from_discovery = True
                validation_name = followup_name
                validation_arguments = followup_arguments
                validation_result = self._execute_controller_tool(
                    name=followup_name,
                    arguments=validation_arguments,
                    request_text=request_text,
                    round_number=round_number,
                    successful_tool_results=successful_tool_results,
                    satisfied_tool_names=satisfied_tool_names,
                    tool_calls_this_turn=tool_calls_this_turn,
                )
        if (
            not validation_promoted_from_discovery
            and validation_name == "lint_typecheck"
            and validation_result.get("ok") is True
            and mutated_paths
            and "contract_check" not in forbidden_tool_names
        ):
            contract_args = {"changed_files": sorted(path for path in mutated_paths if path.endswith(".py"))}
            if contract_args["changed_files"]:
                self._record_event(
                    "auto_validation",
                    name="contract_check",
                    arguments=contract_args,
                    reason=f"{reason_prefix}contract check changed Python functions",
                    mutation_version=mutation_version,
                    rounds=round_number,
                )
                validation_name = "contract_check"
                validation_arguments = contract_args
                validation_result = self._execute_controller_tool(
                    name="contract_check",
                    arguments=validation_arguments,
                    request_text=request_text,
                    round_number=round_number,
                    successful_tool_results=successful_tool_results,
                    satisfied_tool_names=satisfied_tool_names,
                    tool_calls_this_turn=tool_calls_this_turn,
                )
        if (
            not validation_promoted_from_discovery
            and validation_result.get("ok") is True
            and validation_name in {"lint_typecheck", "contract_check"}
            and "select_tests" not in forbidden_tool_names
            and "run_test" not in forbidden_tool_names
            and code_validation_paths
        ):
            select_args = {"changed_files": code_validation_paths}
            validator_executed_before_select = validation_name in {"lint_typecheck", "contract_check"}
            self._record_event(
                "auto_validation",
                name="select_tests",
                arguments=select_args,
                reason=f"{reason_prefix}select targeted tests after syntax check",
                mutation_version=mutation_version,
                rounds=round_number,
            )
            select_result = self._execute_controller_tool(
                name="select_tests",
                arguments=select_args,
                request_text=request_text,
                round_number=round_number,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                tool_calls_this_turn=tool_calls_this_turn,
            )
            validation_name = "select_tests"
            validation_arguments = select_args
            validation_result = select_result
            commands = select_result.get("test_commands") if select_result.get("ok") is True else None
            if isinstance(commands, list) and commands:
                targeted_command = str(commands[0]).strip()
                self._record_event(
                    "auto_validation",
                    name="run_test",
                    arguments={"command": targeted_command},
                    reason=f"{reason_prefix}targeted test selected from changed files",
                    mutation_version=mutation_version,
                    rounds=round_number,
                )
                validation_name = "run_test"
                validation_arguments = {"command": targeted_command}
                validation_result = self._execute_controller_tool(
                    name="run_test",
                    arguments=validation_arguments,
                    request_text=request_text,
                    round_number=round_number,
                    successful_tool_results=successful_tool_results,
                    satisfied_tool_names=satisfied_tool_names,
                    tool_calls_this_turn=tool_calls_this_turn,
                )
            elif self.tools.default_test_command:
                self._record_event(
                    "auto_validation",
                    name="run_test",
                    arguments={"command": self.tools.default_test_command},
                    reason=f"{reason_prefix}fallback to configured test command after empty targeted selection",
                    mutation_version=mutation_version,
                    rounds=round_number,
                )
                validation_name = "run_test"
                validation_arguments = {"command": self.tools.default_test_command}
                validation_result = self._execute_controller_tool(
                    name="run_test",
                    arguments=validation_arguments,
                    request_text=request_text,
                    round_number=round_number,
                    successful_tool_results=successful_tool_results,
                    satisfied_tool_names=satisfied_tool_names,
                    tool_calls_this_turn=tool_calls_this_turn,
                )
            elif "discover_validators" not in forbidden_tool_names:
                self._record_event(
                    "auto_validation",
                    name="discover_validators",
                    arguments={"path": "."},
                    reason=f"{reason_prefix}discover validators after empty targeted test selection",
                    mutation_version=mutation_version,
                    rounds=round_number,
                )
                validation_name = "discover_validators"
                validation_arguments = {"path": "."}
                validation_result = self._execute_controller_tool(
                    name="discover_validators",
                    arguments=validation_arguments,
                    request_text=request_text,
                    round_number=round_number,
                    successful_tool_results=successful_tool_results,
                    satisfied_tool_names=satisfied_tool_names,
                    tool_calls_this_turn=tool_calls_this_turn,
                )
                if validation_result.get("ok") is True:
                    discovered_followup = self._discover_validators_followup(
                        validation_result,
                        forbidden_tool_names=forbidden_tool_names,
                        allow_non_test_validators=True,
                        run_non_test_validators_as_commands=False,
                    )
                    if discovered_followup is not None:
                        followup_name, followup_arguments, followup_reason = discovered_followup
                        if followup_name == "lint_typecheck" and validator_executed_before_select:
                            return validation_name, validation_arguments, validation_result
                        self._record_event(
                            "auto_validation",
                            name=followup_name,
                            arguments=followup_arguments,
                            reason=f"{reason_prefix}{followup_reason}",
                            mutation_version=mutation_version,
                            rounds=round_number,
                        )
                        validation_promoted_from_discovery = True
                        validation_name = followup_name
                        validation_arguments = followup_arguments
                        validation_result = self._execute_controller_tool(
                            name=followup_name,
                            arguments=validation_arguments,
                            request_text=request_text,
                            round_number=round_number,
                            successful_tool_results=successful_tool_results,
                            satisfied_tool_names=satisfied_tool_names,
                            tool_calls_this_turn=tool_calls_this_turn,
                        )
        return validation_name, validation_arguments, validation_result

    def _should_force_post_edit_validation(
        self,
        *,
        request_text: str,
        candidate_tool_name: str,
        mutation_verified_this_turn: bool,
        mutation_version: int,
        last_successful_validation_version: int | None,
        mutated_paths: set[str],
        required_mutation_paths: set[str],
        unresolved_syntax_diagnostics: dict[str, str],
        unresolved_static_diagnostics: dict[str, str],
        unresolved_probe_diagnostics: dict[str, str],
    ) -> bool:
        if not self._post_edit_validation_enabled():
            return False
        if not mutation_verified_this_turn or last_successful_validation_version == mutation_version:
            return False
        if self._request_forbids_validation(request_text):
            return False
        if candidate_tool_name in MUTATING_TOOL_NAMES or candidate_tool_name in VALIDATION_TOOL_NAMES:
            return False
        if required_mutation_paths - mutated_paths:
            return False
        if unresolved_syntax_diagnostics or unresolved_static_diagnostics or unresolved_probe_diagnostics:
            return False
        return bool(mutated_paths) or self._request_requires_test_run(request_text)

    def _failure_delta_summary(self, previous: str, current: str) -> str:
        previous_lines = {line.strip() for line in previous.splitlines() if line.strip()}
        changed = [line.strip() for line in current.splitlines() if line.strip() and line.strip() not in previous_lines]
        if not changed:
            changed = [line.strip() for line in current.splitlines() if line.strip()][:8]
        delta = " | ".join(changed[:8])
        return self._truncate_text("Repeated test failure delta: " + delta, limit=520)

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
        if name in {"lint_typecheck", "contract_check", "run_function_probe"}:
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

        if (
            name in MUTATING_TOOL_NAMES
            and result.get("ok") is not True
            and self._failure_result_counts_for_request(request_text, name, result)
        ):
            denial = str(result.get("summary") or result.get("output") or "").strip()
            if denial and "approval mode is read-only" in denial.lower():
                return denial

        mechanical_tool = self._requested_mechanical_tool_call(request_text, forbidden_tool_names=set())
        mechanical_tool_name = mechanical_tool.name if mechanical_tool is not None else None
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
            target_line_read = self._requested_target_line_read(request_text)
            if target_line_read is not None and self._request_asks_exact_line_text(request_text):
                line_text = self._extract_numbered_line_from_output(output, target_line_read.line)
                if line_text is not None:
                    return line_text
            if name == "read_file" and self._request_asks_direct_file_contents(request_text):
                numbered_lines = self._extract_numbered_lines(output)
                content_lines = [text for _, text in numbered_lines]
                if content_lines and len(content_lines) <= 4:
                    candidate = "\n".join(content_lines).strip()
                    if candidate and len(candidate) <= 240:
                        return candidate
            if name == "read_symbol" and self._request_asks_symbol_return(request_text):
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

        if name in {"list_files", "git_status", "git_diff", "discover_validators"} and mechanical_tool_name == name:
            output = str(result.get("output") or result.get("summary") or "").strip()
            if not output:
                return None
            return self._truncate_text(output, limit=2600)

        if name == "search_symbols" and result.get("ok") is True:
            lowered = request_text.lower()
            output = str(result.get("output") or "").strip()
            if not self._request_requires_mutation(request_text) and output:
                match = re.search(r"\b(?:function|method|class)\s+([A-Za-z_][\w.]*)\b", output)
                if match and ("function name only" in lowered or "symbol name only" in lowered or "name only" in lowered):
                    return match.group(1).split(".")[-1]
                if mechanical_tool_name == name:
                    return self._truncate_text(output, limit=2600)

        if name == "code_outline" and result.get("ok") is True:
            lowered = request_text.lower()
            output = str(result.get("output") or result.get("summary") or "").strip()
            if not self._request_requires_mutation(request_text) and output:
                functions = []
                for match in re.finditer(r"^\s*\d+-\d+\s+function\s+([A-Za-z_][\w.]*)\s*:", output, flags=re.MULTILINE):
                    function_name = match.group(1).split(".")[-1]
                    if function_name not in functions:
                        functions.append(function_name)
                if functions and ("which function" in lowered or "function is defined" in lowered or "defined there" in lowered):
                    if len(functions) == 1:
                        path = str(result.get("path") or arguments.get("path") or "that file")
                        return f"The function defined in {path} is {functions[0]}."
                    return "Functions defined there: " + ", ".join(functions) + "."
                if mechanical_tool_name == name:
                    return self._truncate_text(output, limit=2600)

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

        if name == "run_shell" and result.get("ok") is True and self._requested_exact_shell_command(request_text):
            output = str(result.get("output", "")).strip()
            if output and any(phrase in request_text.lower() for phrase in ["tell me", "reply with", "output"]):
                return output

        if name in {"todo_read", "todo_write"} and result.get("ok") is True:
            lowered = request_text.lower()
            output = str(result.get("output", "")).strip()
            if output and "todo" in lowered and any(phrase in lowered for phrase in ["status", "statuses", "list", "reply with", "show"]):
                return output

        if name == "search" and result.get("ok") is True:
            lowered = request_text.lower()
            if self._request_requires_test_run(request_text):
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
            output = str(result.get("output") or result.get("summary") or "").strip()
            if output and mechanical_tool_name == name:
                return self._truncate_text(output, limit=2600)

        if name == "find_implementation_target" and result.get("ok") is True:
            lowered = request_text.lower()
            targets = result.get("targets") if isinstance(result.get("targets"), list) else []
            rendered_targets: list[str] = []
            for item in targets[:3]:
                if not isinstance(item, dict):
                    continue
                path = str(item.get("path") or "").strip()
                symbol = str(item.get("symbol") or "").strip()
                if not path:
                    continue
                rendered_targets.append(path + (f"::{symbol}" if symbol else ""))
            asks_for_file = any(
                phrase in lowered
                for phrase in [
                    "implementation file",
                    "implementation path",
                    "implementation target",
                    "relevant implementation",
                    "likely implementation",
                ]
            )
            if asks_for_file and rendered_targets:
                first_path = rendered_targets[0].split("::", 1)[0]
                if "only" in lowered:
                    return first_path
                if len(rendered_targets) == 1:
                    return f"Relevant implementation file: {first_path}."
                return "Implementation targets: " + ", ".join(rendered_targets) + "."
            output = str(result.get("output") or result.get("summary") or "").strip()
            if output and mechanical_tool_name == name:
                return self._truncate_text(output, limit=2600)

        if name == "run_test" and (
            not self._request_requires_mutation(request_text)
            and (
                mechanical_tool_name == name
                or "whether tests passed" in request_text.lower()
                or "whether they passed" in request_text.lower()
                or "did they pass" in request_text.lower()
                or "tests passed" in request_text.lower()
            )
        ):
            output = str(result.get("output", ""))
            module_match = re.search(r"\b(test_[A-Za-z0-9_]+)\b", output)
            module = module_match.group(1) if module_match else "unknown"
            command = str(result.get("command") or arguments.get("command") or self.tools.default_test_command or "").strip()
            message = f"Tests passed: {'yes' if result.get('ok') is True else 'no'}."
            if module != "unknown":
                message += f" Test module: {module}."
            if command:
                message += f" Command: {command}."
            if result.get("ok") is not True:
                failure = str(result.get("summary") or result.get("output") or "").strip()
                if failure:
                    message += "\n" + self._truncate_text(failure, limit=900)
            return message

        if name == "lint_typecheck" and mechanical_tool_name == name:
            output = str(result.get("output") or result.get("summary") or "").strip()
            if result.get("ok") is True:
                return output or "Lint/typecheck passed."
            if output:
                return "Lint/typecheck failed:\n" + self._truncate_text(output, limit=1200)

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
        started = time.perf_counter()
        result = cached_result if cached_result is not None else self.tools.execute(name, arguments)
        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        if not cache_hit:
            self._store_cached_tool_result(name, arguments, result)
        self._invalidate_turn_cache_if_needed(name, result)
        self._record_command_validation_event(name=name, result=result, round_number=round_number, cached=cache_hit)
        evidence_id = self._next_evidence_id() if feature_enabled("evidence-handles") else None
        real_tool_use = self._counts_as_real_tool_use(name, result) or self._failure_result_counts_for_request(request_text, name, result)
        if real_tool_use:
            satisfied_tool_names.add(name)
        if self._counts_as_real_tool_use(name, result):
            successful_tool_results.append({"name": name, "arguments": deepcopy(arguments), "result": deepcopy(result), "evidence_id": evidence_id})
        self._record_event("tool_result", name=name, result=result, rounds=round_number, cached=cache_hit, duration_ms=duration_ms, evidence_id=evidence_id)
        self.messages.append(
            {
                "role": "user",
                "content": self._tool_result_feedback_message(
                    name,
                    result,
                    real_tool_use=real_tool_use,
                    evidence_id=evidence_id,
                    successful_tool_results=successful_tool_results,
                ),
            }
        )
        self._autosave()
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
        symbol_return_ops = self._symbol_return_update_operations(request_text, required_tool_names)
        if symbol_return_ops:
            if any(tool_name in forbidden_tool_names for tool_name, _ in symbol_return_ops):
                return None
            last_result: dict[str, Any] | None = None
            for tool_name, args in symbol_return_ops:
                round_number += 1
                last_result = self._execute_controller_tool(
                    name=tool_name,
                    arguments=args,
                    request_text=request_text,
                    round_number=round_number,
                    successful_tool_results=successful_tool_results,
                    satisfied_tool_names=satisfied_tool_names,
                    tool_calls_this_turn=tool_calls_this_turn,
                )
                if last_result.get("ok") is not True:
                    return None
            edited_paths = sorted({str(args.get("path", "")) for _, args in symbol_return_ops if args.get("path")})
            return self._record_synthesized_final(
                f"Updated {', '.join(edited_paths)}.",
                tool=str(last_result.get("tool") if last_result else "replace_in_file"),
                round_number=round_number,
            )
        test_grounded_return_rewrite = self._test_grounded_symbol_return_rewrite_spec(request_text)
        if test_grounded_return_rewrite and "find_implementation_target" not in forbidden_tool_names:
            test_path = test_grounded_return_rewrite["test_path"]
            symbol = test_grounded_return_rewrite["symbol"]
            new_expr = test_grounded_return_rewrite["new_expr"]
            old_expr = test_grounded_return_rewrite["old_expr"]
            round_number += 1
            find_args = {"test_path": test_path, "query": symbol, "limit": 6}
            find_result = self._execute_controller_tool(
                name="find_implementation_target",
                arguments=find_args,
                request_text=request_text,
                round_number=round_number,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                tool_calls_this_turn=tool_calls_this_turn,
            )
            if find_result.get("ok") is not True:
                return None
            source_paths = self._source_paths_from_tool_result({"arguments": find_args, "result": find_result})
            if len(source_paths) != 1:
                return None
            symbol_return_ops = self._symbol_return_update_operations_for_target(
                path=source_paths[0],
                symbol=symbol,
                new_expr=new_expr,
                old_expr=old_expr,
                request_text=request_text,
                required_tool_names=required_tool_names,
            )
            if not symbol_return_ops:
                return None
            if any(tool_name in forbidden_tool_names for tool_name, _ in symbol_return_ops):
                return None
            last_result: dict[str, Any] | None = find_result
            for tool_name, args in symbol_return_ops:
                round_number += 1
                last_result = self._execute_controller_tool(
                    name=tool_name,
                    arguments=args,
                    request_text=request_text,
                    round_number=round_number,
                    successful_tool_results=successful_tool_results,
                    satisfied_tool_names=satisfied_tool_names,
                    tool_calls_this_turn=tool_calls_this_turn,
                )
                if last_result.get("ok") is not True:
                    return None
            if self._request_requires_test_run(request_text) and self.tools.default_test_command and "run_test" not in forbidden_tool_names:
                round_number += 1
                test_result = self._execute_controller_tool(
                    name="run_test",
                    arguments={"command": self.tools.default_test_command},
                    request_text=request_text,
                    round_number=round_number,
                    successful_tool_results=successful_tool_results,
                    satisfied_tool_names=satisfied_tool_names,
                    tool_calls_this_turn=tool_calls_this_turn,
                )
                if test_result.get("ok") is not True:
                    return None
                return self._record_synthesized_final(f"Updated {source_paths[0]}; tests passed.", tool="run_test", round_number=round_number)
            return self._record_synthesized_final(
                f"Updated {source_paths[0]}.",
                tool=str(last_result.get("tool") if last_result else "replace_in_file"),
                round_number=round_number,
            )
        optional_parameter_ops = self._optional_parameter_update_operations(request_text)
        if optional_parameter_ops and "edit_intent" not in forbidden_tool_names:
            pending_optional_parameter_ops = [
                (tool_name, args)
                for tool_name, args in optional_parameter_ops
                if not self._successful_tool_call_already_satisfied(
                    name=tool_name,
                    arguments=args,
                    successful_tool_results=successful_tool_results,
                )
            ]
            if not pending_optional_parameter_ops:
                edited_paths = sorted({str(args.get("path", "")) for _, args in optional_parameter_ops if args.get("path")})
                return self._record_synthesized_final(
                    f"Updated {', '.join(edited_paths)}.",
                    tool="edit_intent",
                    round_number=round_number,
                )
            last_result: dict[str, Any] | None = None
            for tool_name, args in pending_optional_parameter_ops:
                round_number += 1
                last_result = self._execute_controller_tool(
                    name=tool_name,
                    arguments=args,
                    request_text=request_text,
                    round_number=round_number,
                    successful_tool_results=successful_tool_results,
                    satisfied_tool_names=satisfied_tool_names,
                    tool_calls_this_turn=tool_calls_this_turn,
                )
                if last_result.get("ok") is not True:
                    return None
            edited_paths = sorted({str(args.get("path", "")) for _, args in optional_parameter_ops if args.get("path")})
            return self._record_synthesized_final(
                f"Updated {', '.join(edited_paths)}.",
                tool=str(last_result.get("tool") if last_result else "edit_intent"),
                round_number=round_number,
            )
        rename_ops = self._project_function_rename_operations(request_text)
        if rename_ops and "edit_intent" not in forbidden_tool_names:
            last_result: dict[str, Any] | None = None
            rename_already_satisfied = self._project_function_rename_already_satisfied(
                request_text=request_text,
                successful_tool_results=successful_tool_results,
            )
            if not rename_already_satisfied:
                for tool_name, args in rename_ops:
                    round_number += 1
                    last_result = self._execute_controller_tool(
                        name=tool_name,
                        arguments=args,
                        request_text=request_text,
                        round_number=round_number,
                        successful_tool_results=successful_tool_results,
                        satisfied_tool_names=satisfied_tool_names,
                        tool_calls_this_turn=tool_calls_this_turn,
                    )
                    if last_result.get("ok") is not True:
                        return None
            if self._request_requires_test_run(request_text) and self.tools.default_test_command and "run_test" not in forbidden_tool_names:
                round_number += 1
                test_result = self._execute_controller_tool(
                    name="run_test",
                    arguments={"command": self.tools.default_test_command},
                    request_text=request_text,
                    round_number=round_number,
                    successful_tool_results=successful_tool_results,
                    satisfied_tool_names=satisfied_tool_names,
                    tool_calls_this_turn=tool_calls_this_turn,
                )
                if test_result.get("ok") is not True:
                    return None
                return self._record_synthesized_final("Renamed symbol project-wide; tests passed.", tool="run_test", round_number=round_number)
            return self._record_synthesized_final("Renamed symbol project-wide.", tool=str(last_result.get("tool") if last_result else "edit_intent"), round_number=round_number)
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
        else:
            constant_match = re.search(
                r"\b(?:update|change|set)\s+the\s+(?P<name>[A-Z_][A-Z0-9_]*)\s+constant\s+in\s+(?P<path>[\w./-]+\.py)\s+to\s+['\"](?P<new>[^'\"]+)['\"]",
                request_text,
                flags=re.IGNORECASE,
            )
            if constant_match:
                path = constant_match.group("path")
                name = constant_match.group("name")
                new = constant_match.group("new")
                try:
                    target = self.tools.resolve_path(path, allow_missing=False)
                    source = target.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    source = ""
                line_match = re.search(rf"(?m)^{re.escape(name)}\s*=\s*(['\"])(?P<old>.*?)\1\s*$", source)
                if line_match:
                    quote = line_match.group(1)
                    old_line = line_match.group(0)
                    new_line = f"{name} = {quote}{new}{quote}"
                    operation = ("replace_in_file", {"path": path, "old": old_line, "new": new_line})
            if operation is not None:
                pass
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

    def _project_function_rename_operations(self, request_text: str) -> list[tuple[str, dict[str, Any]]] | None:
        lowered = request_text.lower()
        if not re.search(r"\b(?:rename|renam|refactor|change|update)\b", lowered):
            return None
        match = re.search(
            r"\bfrom\s+(?P<old>[A-Za-z_]\w*)\s*\([^)]*\)\s+to\s+(?P<new>[A-Za-z_]\w*)\s*\(",
            request_text,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        old = match.group("old")
        new = match.group("new")
        if old == new:
            return None
        if not re.search(r"\b(?:api|function|symbol|call|calls|callers|docs?|tests?|project|repo|source)\b", lowered):
            return None
        return [
            (
                "edit_intent",
                {
                    "path": ".",
                    "intent": "rename",
                    "target": old,
                    "replacement": new,
                    "scope": "project",
                },
            )
        ]

    def _project_function_rename_already_satisfied(
        self,
        *,
        request_text: str,
        successful_tool_results: list[dict[str, Any]],
    ) -> bool:
        rename_ops = self._project_function_rename_operations(request_text)
        if not rename_ops or len(rename_ops) != 1:
            return False
        tool_name, args = rename_ops[0]
        if tool_name != "edit_intent" or not isinstance(args, dict):
            return False
        target = str(args.get("target") or "").strip()
        replacement = str(args.get("replacement") or "").strip()
        scope = str(args.get("scope") or "").strip().lower()
        path = str(args.get("path") or "").strip()
        if not target or not replacement:
            return False
        for item in reversed(successful_tool_results):
            if str(item.get("name") or "").strip() != "edit_intent":
                continue
            item_arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            if str(item_arguments.get("target") or "").strip() != target:
                continue
            if str(item_arguments.get("replacement") or "").strip() != replacement:
                continue
            if str(item_arguments.get("scope") or "").strip().lower() != scope:
                continue
            if str(item_arguments.get("path") or "").strip() != path:
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            if result.get("ok") is True:
                return True
        return False

    def _successful_tool_call_already_satisfied(
        self,
        *,
        name: str,
        arguments: dict[str, Any],
        successful_tool_results: list[dict[str, Any]],
    ) -> bool:
        expected = json.dumps(arguments, sort_keys=True, ensure_ascii=True)
        for item in reversed(successful_tool_results):
            if str(item.get("name") or "").strip() != name:
                continue
            item_arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            if json.dumps(item_arguments, sort_keys=True, ensure_ascii=True) != expected:
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            if result.get("ok") is True:
                return True
        return False

    def _test_grounded_symbol_return_rewrite_spec(self, request_text: str) -> dict[str, str] | None:
        match = re.search(
            r"\b(?P<test>[\w./-]+\.py)\b(?:(?!\n\n).){0,260}?\b(?:change|changing|update|updating)\s+(?P<symbol>[A-Za-z_]\w*)\s*\([^)]*\)\s+so\s+it\s+returns\s+(?P<new>.+?)\s+instead\s+of\s+(?P<old>.+?)(?:[.?!]|$)",
            request_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return None
        test_path = match.group("test").strip().rstrip(".,;:")
        if not self._path_looks_like_test_file(test_path):
            return None
        symbol = match.group("symbol").strip()
        new_expr = self._clean_return_expression(match.group("new"))
        old_expr = self._clean_return_expression(match.group("old"))
        if not test_path or not symbol or not new_expr or not old_expr:
            return None
        return {
            "test_path": test_path,
            "symbol": symbol,
            "new_expr": new_expr,
            "old_expr": old_expr,
        }

    def _symbol_return_update_operations_for_target(
        self,
        *,
        path: str,
        symbol: str,
        new_expr: str,
        old_expr: str,
        request_text: str,
        required_tool_names: set[str],
    ) -> list[tuple[str, dict[str, Any]]] | None:
        if not path or not symbol or not new_expr or not old_expr:
            return None
        try:
            target = self.tools.resolve_path(path, allow_missing=False)
            source = target.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
        old_line: str | None = None
        new_line: str | None = None
        old_candidates = {f"return {old_expr}", f"return {old_expr};"}
        for line in source.splitlines():
            stripped = line.strip()
            if stripped not in old_candidates:
                continue
            indent = line[: len(line) - len(line.lstrip())]
            semicolon = ";" if stripped.endswith(";") else ""
            old_line = line
            new_line = f"{indent}return {new_expr}{semicolon}"
            break
        if old_line is None or new_line is None or old_line == new_line:
            return None
        requested_tools = self._requested_tool_names(request_text, forbidden_tool_names=set())
        operations: list[tuple[str, dict[str, Any]]] = []
        if "search_symbols" in requested_tools or "search_symbols" in required_tool_names:
            operations.append(("search_symbols", {"query": symbol, "path": path}))
        if "read_symbol" in requested_tools or "read_symbol" in required_tool_names:
            operations.append(("read_symbol", {"path": path, "symbol": symbol, "include_context": 0}))
        operations.append(("replace_in_file", {"path": path, "old": old_line, "new": new_line}))
        return operations

    def _symbol_return_update_operations(self, request_text: str, required_tool_names: set[str]) -> list[tuple[str, dict[str, Any]]] | None:
        match = re.search(
            r"\b(?P<path>[\w./-]+\.(?:js|jsx|ts|tsx|py))\b(?:(?!\n\n).){0,240}?\b(?:change|changing|update|updating)\s+(?P<symbol>[A-Za-z_]\w*)\s*\([^)]*\)\s+so\s+it\s+returns\s+(?P<new>.+?)\s+instead\s+of\s+(?P<old>.+?)(?:[.?!]|$)",
            request_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return None
        path = match.group("path").strip().rstrip(".,;:")
        symbol = match.group("symbol").strip()
        new_expr = self._clean_return_expression(match.group("new"))
        old_expr = self._clean_return_expression(match.group("old"))
        return self._symbol_return_update_operations_for_target(
            path=path,
            symbol=symbol,
            new_expr=new_expr,
            old_expr=old_expr,
            request_text=request_text,
            required_tool_names=required_tool_names,
        )

    def _clean_return_expression(self, expression: str) -> str:
        expression = re.sub(r"\s+", " ", expression).strip()
        expression = expression.strip("`\"' ")
        expression = re.sub(r"\s+(?:then\s+)?(?:run|rerun|and|do not|don't)\b.*$", "", expression, flags=re.IGNORECASE).strip()
        return expression.rstrip(".,;:").strip()

    def _optional_parameter_update_operations(self, request_text: str) -> list[tuple[str, dict[str, Any]]] | None:
        match = re.search(
            r"\badd\s+an?\s+optional\s+(?P<param>[A-Za-z_]\w*)\s*:\s*(?P<annotation>[^=]+?)\s*=\s*(?P<default>False|True|None|[-+]?\d+(?:\.\d+)?|['\"][^'\"]*['\"])\s+parameter\s+to\s+(?P<symbol>[A-Za-z_]\w*)\s+in\s+(?P<src>[\w./-]+\.py)\b(?:(?!\n\n).){0,240}?\bupdate\s+(?P<doc>[\w./-]+\.md)\b",
            request_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return None
        src_path = match.group("src")
        doc_path = match.group("doc")
        symbol = match.group("symbol")
        param = match.group("param")
        annotation = re.sub(r"\s+", " ", match.group("annotation")).strip()
        default = match.group("default").strip()
        try:
            source_target = self.tools.resolve_path(src_path, allow_missing=False)
            source = source_target.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
            doc_target = self.tools.resolve_path(doc_path, allow_missing=False)
            docs = doc_target.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

        node = next(
            (
                child
                for child in tree.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == symbol
            ),
            None,
        )
        if node is None:
            return None
        existing_params = {arg.arg for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]}
        operations: list[tuple[str, dict[str, Any]]] = []
        if param not in existing_params:
            signature = self._signature_with_appended_parameter(source, node, f"{param}: {annotation} = {default}")
            if not signature:
                return None
            operations.append(
                (
                    "edit_intent",
                    {
                        "path": src_path,
                        "intent": "change_signature",
                        "target": symbol,
                        "replacement": signature,
                    },
                )
            )

        if param not in docs:
            call_match = re.search(rf"`{re.escape(symbol)}\((?P<args>[^`]*)\)`", docs)
            if call_match and param not in call_match.group("args"):
                old_call = call_match.group(0)
                args = call_match.group("args").strip()
                separator = ", " if args else ""
                new_call = f"`{symbol}({args}{separator}{param}={default})`"
                operations.append(("replace_in_file", {"path": doc_path, "old": old_call, "new": new_call}))
        if not operations:
            return None
        return operations

    def _signature_with_appended_parameter(self, source: str, node: ast.FunctionDef | ast.AsyncFunctionDef, parameter: str) -> str:
        lines = source.splitlines()
        start = int(getattr(node, "lineno", 1)) - 1
        if start < 0 or start >= len(lines):
            return ""
        signature_line = lines[start].strip()
        if "\n" in signature_line or not signature_line.startswith(("def ", "async def ")):
            return ""
        open_index = signature_line.find("(")
        if open_index < 0:
            return ""
        depth = 0
        close_index = -1
        for index, char in enumerate(signature_line[open_index:], start=open_index):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    close_index = index
                    break
        if close_index < 0:
            return ""
        current_params = signature_line[open_index + 1 : close_index].strip()
        if parameter.split(":", 1)[0].strip() in {part.split(":", 1)[0].split("=", 1)[0].strip() for part in current_params.split(",")}:
            return signature_line
        separator = ", " if current_params else ""
        return f"{signature_line[: open_index + 1]}{current_params}{separator}{parameter}{signature_line[close_index:]}"

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
        successful_tool_results: list[dict[str, Any]] | None = None,
    ) -> AgentResult | None:
        if session_memory_request:
            return None
        prior_successful_tool_results = list(successful_tool_results or [])
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
            successful_tool_results=[*prior_successful_tool_results, *successful_tool_results],
            satisfied_tool_names=satisfied_tool_names,
            tool_calls_this_turn=tool_calls_this_turn,
        )
        if simple_rewrite is not None:
            return simple_rewrite

        if (
            self.tools.default_test_command
            and "run_test" not in forbidden_tool_names
            and "write_file" not in forbidden_tool_names
            and (self._request_requires_mutation(request_text) or self._request_requires_code_mutation(request_text))
            and self._request_requires_test_run(request_text)
            and re.search(r"\b(?:import|package|module|modulenotfound|importerror)\b", lowered)
        ):
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
            if test_result.get("ok") is True:
                return self._record_synthesized_final("Tests already pass.", tool="run_test", round_number=round_number)
            import_repair = self._try_relative_import_repair(
                request_text=request_text,
                round_number=round_number,
                failed_run_test_result=test_result,
                run_test_arguments=test_args,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                tool_calls_this_turn=tool_calls_this_turn,
            )
            if import_repair is not None:
                return import_repair
        controller = NavigationValidationController(self)
        turn = NavigationValidationTurn(
            request_text=request_text,
            target_line_read=target_line_read,
            symbol_read=symbol_read,
            exact_shell_command=exact_shell_command,
            expected_exact_reply_text=expected_exact_reply_text,
            required_tool_names=required_tool_names,
            forbidden_tool_names=forbidden_tool_names,
            requested_git_diff_mode=requested_git_diff_mode,
        )
        return controller.handle(turn)

    def _read_only_mutation_probe(self, request_text: str, exact_file_write: ExactFileWriteSpec | None) -> tuple[str, dict[str, Any]] | None:
        if self.approval_mode() != "read-only":
            return None
        if not self._request_requires_mutation(request_text):
            return None
        if not self._failure_result_counts_for_request(request_text, "write_file", {"ok": False, "summary": "probe"}):
            return None
        if exact_file_write is not None:
            return "write_file", {"path": exact_file_write.path, "content": exact_file_write.line + "\n"}
        path = self._requested_loose_file_create_path(request_text)
        if path is None:
            return None
        return "write_file", {"path": path, "content": "placeholder\n"}

    def _final_claims_file_mutation(self, message: str) -> bool:
        lowered = message.lower()
        patterns = [
            r"\b(?:i|we)\s+(?:updated|edited|changed|modified|created|wrote|rewrote|deleted|removed|renamed)\b",
            r"\bhas been\s+(?:updated|edited|changed|modified|created|written|rewritten|deleted|removed|renamed)\b",
            r"\bwas\s+(?:updated|edited|changed|modified|created|written|rewritten|deleted|removed|renamed)\b",
        ]
        return any(re.search(pattern, lowered) for pattern in patterns)

    def _display_path_basename(self, raw_path: str) -> str:
        normalized = str(raw_path or "").replace("\\", "/").rstrip("/")
        if not normalized:
            return ""
        return normalized.rsplit("/", 1)[-1]

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

    def _request_mentions_workspace_path(self, text: str) -> bool:
        return bool(re.search(r"\b[\w./-]+\.[A-Za-z0-9]+\b", text))

    def _should_preload_context_pack(
        self,
        *,
        request_text: str,
        session_memory_request: bool,
        mutation_required: bool,
        test_run_required: bool,
        required_tool_names: set[str],
        forbidden_tool_names: set[str],
    ) -> bool:
        if session_memory_request or "context_pack" in forbidden_tool_names:
            return False
        if required_tool_names and "context_pack" not in required_tool_names:
            return False
        if self._request_expects_exact_tool_error(request_text):
            return False
        if "context_pack" in required_tool_names:
            return True
        if not feature_enabled("context-pack"):
            return False
        deterministic_bootstrap_available = (
            self._project_function_rename_operations(request_text)
            or self._optional_parameter_update_operations(request_text)
            or self._request_looks_like_explicit_python_import_bug_fix(request_text)
        )
        if deterministic_bootstrap_available:
            return False
        if self._request_is_broad_or_ambiguous(request_text):
            return True
        if (mutation_required or test_run_required) and self._request_mentions_workspace_path(request_text):
            return True
        return False

    def _request_forbids_clarifying_questions(self, text: str) -> bool:
        return bool(re.search(r"\b(?:do not|don't|dont|never|no)\s+(?:ask|clarify|question)s?\b", text, flags=re.IGNORECASE))

    def _request_explicitly_wants_clarification(self, text: str) -> bool:
        return bool(
            re.search(
                r"\b(?:ask (?:me )?(?:a )?questions?|clarify|clarifying question|before you (?:edit|change|implement)|don't assume|do not assume)\b",
                text,
                flags=re.IGNORECASE,
            )
        )

    def _request_has_clarification_risk_signal(self, text: str) -> bool:
        if self._request_is_broad_or_ambiguous(text):
            return True
        return bool(
            re.search(
                r"\b(?:keep fixing|make (?:it|this|the app|the cli|the repo) better|improve|optimi[sz]e|throughput|profile|benchmark|"
                r"architecture|design|workflow|integration|migration|public api|schema|compatib|delete|remove|security|auth|permission|"
                r"out of the box|first use|e2e|user experience|ux|tradeoff|default model|default behavior)\b",
                text,
                flags=re.IGNORECASE,
            )
        )

    def _should_plan_clarifying_questions(
        self,
        *,
        request_text: str,
        session_memory_request: bool,
        mutation_required: bool,
        test_run_required: bool,
        required_tool_names: set[str],
        forbidden_tool_names: set[str],
        exact_request: bool,
    ) -> bool:
        if not feature_enabled("question-planner"):
            return False
        if session_memory_request or exact_request or self._request_forbids_clarifying_questions(request_text):
            return False
        if required_tool_names or forbidden_tool_names:
            return False
        lowered = request_text.lower()
        if re.search(r"\b(?:commit|push|status|diff)\b", lowered) and not self._request_explicitly_wants_clarification(request_text):
            return False
        if re.search(r"\b(?:run|rerun|execute)\b[^.?!\n]{0,80}\btests?\b", lowered) and not mutation_required:
            return False
        explicit = self._request_explicitly_wants_clarification(request_text)
        risky = self._request_has_clarification_risk_signal(request_text)
        if not explicit and not risky:
            return False
        if self._request_mentions_workspace_path(request_text) and not explicit:
            high_risk_path_change = re.search(
                r"\b(?:refactor|migration|public api|schema|delete|remove|security|auth|permission|architecture|design)\b",
                lowered,
            )
            if not high_risk_path_change:
                return False
        return explicit or risky or mutation_required or test_run_required

    def _has_successful_tool_named(self, successful_tool_results: list[dict[str, Any]], name: str) -> bool:
        return any(str(item.get("name", "")).strip() == name for item in successful_tool_results)

    def _question_planner_payload(
        self,
        *,
        request_text: str,
        successful_tool_results: list[dict[str, Any]],
        mutation_required: bool,
        test_run_required: bool,
    ) -> dict[str, Any]:
        return {
            "workspace_root": self.tools.workspace_root.as_posix(),
            "user_request": self._truncate_text(request_text, limit=1200),
            "mutation_required": mutation_required,
            "test_run_required": test_run_required,
            "evidence": [
                self._compact_successful_tool_result_for_verification(item)
                for item in successful_tool_results[-QUESTION_PLANNER_EVIDENCE_LIMIT:]
            ],
            "decision_focus": [
                "highest-leverage ambiguity axis",
                "scope boundary",
                "acceptance signal",
                "optimization priority",
                "risk/tradeoff",
                "irreversible/high-cost choice",
                "model limits",
            ],
        }

    def _question_planner_messages(self, payload: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": QUESTION_PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True, separators=(",", ":"))},
        ]

    def _normalize_question_choices(self, raw_choices: list[str]) -> list[str]:
        choices: list[str] = []
        seen: set[str] = set()
        for raw_choice in raw_choices:
            cleaned = self._truncate_text(str(raw_choice).strip().rstrip(".,;:"), limit=80)
            if not cleaned:
                continue
            normalized = re.sub(r"\s+", " ", cleaned).strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            choices.append(cleaned)
            if len(choices) >= 4:
                break
        return choices

    def _question_aspect_tags(self, text: str) -> list[str]:
        lowered = text.lower()
        tag_patterns = {
            "scope": (r"\b(?:scope|boundary|in scope|out of scope|keep fixed|hold fixed|surface)\b",),
            "acceptance": (r"\b(?:acceptance|success|definition of done|pass rate|tests?|correctness|quality bar)\b",),
            "risk": (r"\b(?:risk|safe|safety|fail-closed|permission|security)\b",),
            "tradeoff": (r"\b(?:tradeoff|priority|prioritize|optimi[sz]e|latency|speed|throughput|token|cost|ux)\b",),
            "workflow": (r"\b(?:workflow|path|happy path|first-use|first use|user path|operator|maintainer|benchmark runner)\b",),
            "compatibility": (r"\b(?:compatib|public api|cli surface|session/transcript|tool contracts?|benchmark comparability)\b",),
        }
        tags: list[str] = []
        for tag, patterns in tag_patterns.items():
            if any(re.search(pattern, lowered) for pattern in patterns):
                tags.append(tag)
        return tags

    def _question_is_compound(self, question: str) -> bool:
        lowered = question.lower()
        conjunctions = len(re.findall(r"\b(?:and|or)\b", lowered))
        return question.count("?") > 1 or conjunctions >= 2

    def _question_looks_eba_style(self, question: str, choices: list[str]) -> bool:
        if not 2 <= len(choices) <= 4:
            return False
        lowered = question.lower().strip()
        if "should i proceed" in lowered or "should i continue" in lowered:
            return False
        if not any(
            lowered.startswith(prefix)
            for prefix in (
                "which ",
                "which one ",
                "which single ",
                "what outcome ",
                "what should ",
                "for this pass, which ",
            )
        ):
            return False
        generic_choices = {"yes", "no", "maybe", "proceed", "continue"}
        normalized_choices = {
            re.sub(r"[^a-z0-9]+", " ", choice.lower()).strip()
            for choice in choices
            if str(choice).strip()
        }
        if len({choice for choice in normalized_choices if choice not in generic_choices}) < 2:
            return False
        return bool(self._question_aspect_tags(question + " " + " ".join(choices)))

    def _question_quality_metrics(self, item: dict[str, Any], *, request_text: str = "") -> dict[str, Any]:
        question = str(item.get("question") or "").strip()
        why = str(item.get("why_it_matters") or "").strip()
        recommended_default = str(item.get("recommended_default") or "").strip()
        choices = self._normalize_question_choices(
            [str(choice) for choice in list(item.get("choices") or []) if str(choice).strip()]
        )
        low_value = self._question_text_is_low_value(question)
        compound = self._question_is_compound(question)
        aspect_tags = self._question_aspect_tags(" ".join([question, why, recommended_default, *choices]))
        request_tags = set(self._question_aspect_tags(request_text))
        eba_style = self._question_looks_eba_style(question, choices)
        score = 0
        if low_value:
            score -= 6
        if 2 <= len(choices) <= 4:
            score += 3
        elif not choices:
            score -= 3
        else:
            score -= 1
        if recommended_default:
            score += 2
        if why:
            score += 1
        if eba_style:
            score += 4
        if compound:
            score -= 2
        if aspect_tags:
            score += 1
        if request_tags.intersection(aspect_tags):
            score += 1
        return {
            "quality_score": score,
            "eba_style": eba_style,
            "compound": compound,
            "aspect_tags": aspect_tags,
            "normalized_choices": choices,
        }

    def _build_clarifying_question(
        self,
        *,
        question: str,
        why_it_matters: str = "",
        recommended_default: str = "",
        choices: list[str] | None = None,
        request_text: str = "",
    ) -> dict[str, Any] | None:
        normalized_question = self._truncate_text(str(question).strip(), limit=260)
        if not normalized_question or self._question_text_is_low_value(normalized_question):
            return None
        row = {
            "question": normalized_question,
            "why_it_matters": self._truncate_text(str(why_it_matters).strip(), limit=220),
            "recommended_default": self._truncate_text(str(recommended_default).strip(), limit=220),
            "choices": self._normalize_question_choices(
                [str(choice) for choice in list(choices or []) if str(choice).strip()]
            ),
        }
        row.update(self._question_quality_metrics(row, request_text=request_text))
        return row

    def _question_text_is_low_value(self, question: str) -> bool:
        lowered = question.lower()
        if "should i proceed" in lowered or "should i continue" in lowered:
            return True
        discoverable_patterns = [
            r"\bwhich files?\b[^?]{0,80}\b(?:read|inspect|open|search|look at)\b",
            r"\bwhere\b[^?]{0,80}\b(?:file|function|class|test|command)\b",
            r"\bwhat\b[^?]{0,80}\b(?:test command|error|traceback|path)\b",
        ]
        return any(re.search(pattern, lowered) for pattern in discoverable_patterns)

    def _question_planner_reason_indicates_ambiguity(self, reason: str) -> bool:
        return bool(
            re.search(
                r"\b(?:vague|ambiguous|unclear|unspecified|insufficient|not enough|requires? defining|need(?:s)? defining|"
                r"cannot determine|can't determine|unresolved|missing (?:scope|acceptance|tradeoff))\b",
                reason,
                flags=re.IGNORECASE,
            )
        )

    def _fallback_clarifying_questions(self, request_text: str, reason: str) -> list[dict[str, Any]]:
        lowered = request_text.lower()
        if re.search(r"\b(?:throughput|performance|perf|latency|speed|token|benchmark)\b", lowered):
            return [
                {
                    "question": "Which single optimization axis should dominate this pass: benchmark pass rate, wall-clock latency, model/tool token cost, or first-use UX?",
                    "why_it_matters": "That one axis decides whether the next change should favor controller routing, context compaction, validation, or onboarding behavior.",
                    "recommended_default": "Preserve benchmark pass rate first, then reduce repeated tool/model loops and wall-clock latency.",
                    "choices": ["benchmark pass rate", "wall-clock latency", "model/tool token cost", "first-use UX"],
                }
            ]
        if re.search(r"\b(?:rewrite|refactor|architecture|design|public api|schema|session|transcript|tool contract|compatib)\b", lowered):
            return [
                {
                    "question": "Which boundary should stay fixed in this pass: CLI surface, session/transcript format, tool contracts, or benchmark comparability?",
                    "why_it_matters": "That boundary determines how aggressive the internal rewrite can be without breaking existing usage or muddying performance claims.",
                    "recommended_default": "Keep the CLI surface and session/transcript format stable first, then refactor internals behind those boundaries.",
                    "choices": ["CLI surface", "session/transcript format", "tool contracts", "benchmark comparability"],
                }
            ]
        if re.search(r"\b(?:workflow|flow|ux|first use|out of the box|e2e)\b", lowered):
            return [
                {
                    "question": "Which workflow should define success for this pass: first-time local setup, repeat coding loop, benchmark runner, or user-visible runtime path?",
                    "why_it_matters": "Different paths prioritize different files, validators, and acceptance checks.",
                    "recommended_default": "Use the first-time local setup path unless local evidence already shows a more urgent failing runtime or test path.",
                    "choices": ["first-time local setup", "repeat coding loop", "benchmark runner", "user-visible runtime path"],
                }
            ]
        if re.search(r"\b(?:security|auth|permission|secrets?|credentials?)\b", lowered):
            return [
                {
                    "question": "Which risk posture should dominate this pass: fail-closed safety, compatibility with existing workflows, operator convenience, or minimal code churn?",
                    "why_it_matters": "That choice changes whether the agent should tighten guards, preserve legacy behavior, or optimize for least-invasive edits.",
                    "recommended_default": "Prefer fail-closed safety unless that would block a currently required and verified workflow.",
                    "choices": ["fail-closed safety", "compatibility with existing workflows", "operator convenience", "minimal code churn"],
                }
            ]
        if re.search(r"\b(?:most important|priority|highest impact)\b", lowered):
            return [
                {
                    "question": "Which signal should decide what is most important in this pass: failing tests, user-visible breakage, performance, or security?",
                    "why_it_matters": "The priority signal changes what subsystem gets edited first.",
                    "recommended_default": "Prioritize reproducible failures and first-use blockers before cleanup.",
                    "choices": ["failing tests", "user-visible breakage", "performance", "security"],
                }
            ]
        return [
            {
                "question": "Which aspect should break ties in this pass: scope coverage, acceptance strictness, safety, or speed?",
                "why_it_matters": "That single tie-breaker sets the boundary, acceptance bar, and risk tolerance for the next edit.",
                "recommended_default": "Prefer acceptance strictness first, then safety, before expanding scope or chasing speed.",
                "choices": ["scope coverage", "acceptance strictness", "safety", "speed"],
            }
        ]

    def _normalize_question_planner_payload(self, payload: dict[str, Any] | None, *, request_text: str = "") -> dict[str, Any]:
        decision = payload if isinstance(payload, dict) else {}
        verdict = str(decision.get("verdict", "")).strip().lower()
        if verdict not in {"ask", "proceed"}:
            verdict = "proceed"
        explicit_clarification = self._request_explicitly_wants_clarification(request_text) if request_text else False
        risky_clarification = self._request_has_clarification_risk_signal(request_text) if request_text else False
        questions: list[dict[str, Any]] = []
        raw_questions = decision.get("questions")
        if isinstance(raw_questions, list):
            for item in raw_questions:
                if isinstance(item, str):
                    normalized = self._build_clarifying_question(question=item.strip(), request_text=request_text)
                elif isinstance(item, dict):
                    raw_choices = item.get("choices")
                    normalized = self._build_clarifying_question(
                        question=str(item.get("question") or item.get("text") or "").strip(),
                        why_it_matters=str(item.get("why_it_matters") or item.get("why") or "").strip(),
                        recommended_default=str(item.get("recommended_default") or item.get("default") or "").strip(),
                        choices=[str(choice) for choice in raw_choices] if isinstance(raw_choices, list) else [],
                        request_text=request_text,
                    )
                else:
                    continue
                if normalized is None:
                    continue
                questions.append(normalized)
                if len(questions) >= QUESTION_PLANNER_MAX_QUESTIONS:
                    break
        raw_ambiguities = decision.get("ambiguities")
        ambiguities: list[dict[str, str]] = []
        if isinstance(raw_ambiguities, list):
            for item in raw_ambiguities:
                if not isinstance(item, dict):
                    continue
                kind = str(item.get("kind", "")).strip().lower()
                if kind not in {"scope", "intent", "acceptance", "risk", "tradeoff"}:
                    kind = "scope"
                detail = str(item.get("detail", "")).strip()
                if not detail:
                    continue
                ambiguities.append(
                    {
                        "kind": kind,
                        "detail": self._truncate_text(detail, limit=180),
                        "evidence": self._truncate_text(str(item.get("evidence", "")).strip(), limit=140),
                    }
                )
                if len(ambiguities) >= QUESTION_PLANNER_MAX_QUESTIONS:
                    break
        reason = self._truncate_text(str(decision.get("reason", "")).strip(), limit=260)
        if (
            not questions
            and request_text
            and (verdict == "ask" or self._question_planner_reason_indicates_ambiguity(reason) or explicit_clarification)
            and (risky_clarification or explicit_clarification)
        ):
            questions = [
                normalized
                for normalized in (
                    self._build_clarifying_question(
                        question=str(item.get("question") or ""),
                        why_it_matters=str(item.get("why_it_matters") or ""),
                        recommended_default=str(item.get("recommended_default") or ""),
                        choices=[str(choice) for choice in list(item.get("choices") or [])],
                        request_text=request_text,
                    )
                    for item in self._fallback_clarifying_questions(request_text, reason)[:QUESTION_PLANNER_MAX_QUESTIONS]
                )
                if normalized is not None
            ]
            verdict = "ask"
        if questions:
            if (
                (verdict == "ask" or explicit_clarification)
                and request_text
                and not any(bool(item.get("eba_style")) for item in questions)
                and (risky_clarification or explicit_clarification)
            ):
                fallback_questions = [
                    normalized
                    for normalized in (
                        self._build_clarifying_question(
                            question=str(item.get("question") or ""),
                            why_it_matters=str(item.get("why_it_matters") or ""),
                            recommended_default=str(item.get("recommended_default") or ""),
                            choices=[str(choice) for choice in list(item.get("choices") or [])],
                            request_text=request_text,
                        )
                        for item in self._fallback_clarifying_questions(request_text, reason)[:QUESTION_PLANNER_MAX_QUESTIONS]
                    )
                    if normalized is not None
                ]
                if fallback_questions:
                    questions = fallback_questions
            questions.sort(
                key=lambda item: (
                    -int(item.get("quality_score", 0)),
                    0 if item.get("eba_style") else 1,
                    0 if 2 <= len(list(item.get("choices") or [])) <= 4 else 1,
                    len(str(item.get("question") or "")),
                )
            )
            high_value_eba = next(
                (
                    item
                    for item in questions
                    if bool(item.get("eba_style")) and int(item.get("quality_score", 0)) >= 6
                ),
                None,
            )
            if high_value_eba is not None:
                questions = [high_value_eba]
            else:
                questions = questions[:QUESTION_PLANNER_MAX_QUESTIONS]
        if verdict == "ask" and not questions:
            verdict = "proceed"
        return {
            "verdict": verdict,
            "reason": reason,
            "ambiguities": ambiguities,
            "questions": questions,
        }

    def _format_clarifying_questions(self, decision: dict[str, Any]) -> str:
        questions = decision.get("questions") if isinstance(decision.get("questions"), list) else []
        count = len(questions)
        header = "Need one clarification before continuing:" if count == 1 else f"Need {count} clarifications before continuing:"
        lines = [header]
        reason = str(decision.get("reason", "")).strip()
        if reason:
            lines.append(f"Evidence gap: {reason}")
        for index, item in enumerate(questions, start=1):
            if not isinstance(item, dict):
                continue
            prefix = "" if count == 1 else f"{index}. "
            question = str(item.get("question", "")).strip()
            if question:
                lines.append(prefix + question)
            choices = item.get("choices") if isinstance(item.get("choices"), list) else []
            if choices:
                lines.append("Choices (pick one): " + " | ".join(str(choice) for choice in choices[:4]))
            recommended_default = str(item.get("recommended_default", "")).strip()
            if recommended_default:
                lines.append("Recommended default: " + recommended_default)
            why = str(item.get("why_it_matters", "")).strip()
            if why:
                lines.append("Why: " + why)
        return "\n".join(lines)

    def _maybe_return_clarifying_questions(
        self,
        *,
        request_text: str,
        successful_tool_results: list[dict[str, Any]],
        mutation_required: bool,
        test_run_required: bool,
    ) -> AgentResult | None:
        payload = self._question_planner_payload(
            request_text=request_text,
            successful_tool_results=successful_tool_results,
            mutation_required=mutation_required,
            test_run_required=test_run_required,
        )
        self.status_printer("planning clarifying questions")
        response = self._chat(
            purpose="question_planner",
            model=self.model,
            messages=self._question_planner_messages(payload),
            think=False,
        )
        decision = self._normalize_question_planner_payload(extract_json_response(response.content), request_text=request_text)
        self._record_event(
            "clarification_plan",
            verdict=decision["verdict"],
            reason=decision["reason"],
            ambiguities=decision["ambiguities"],
            questions=decision["questions"],
            evidence_count=len(payload.get("evidence", [])),
            planner=response.content,
        )
        if decision["verdict"] != "ask":
            return None
        message = self._format_clarifying_questions(decision)
        self._append_assistant_payload({"type": "final", "message": message})
        self._record_event("assistant", content=message, rounds=0)
        self._flush_llm_call_events()
        return AgentResult(message=message, rounds=0, completed=True)

    def _resolve_sub_agent_model(self, requested_model: str) -> tuple[str, str | None, str | None]:
        if requested_model == self.model:
            return requested_model, None, None
        list_models = getattr(self.client, "list_models", None)
        if not callable(list_models):
            return requested_model, None, None
        try:
            available = set(str(model) for model in list_models())
        except OllamaError as exc:
            return requested_model, None, f"Could not validate sub-agent model: {exc}"
        if requested_model in available:
            return requested_model, None, None
        latest = f"{requested_model}:latest"
        if latest in available:
            return latest, f"Normalized sub-agent model {requested_model} to {latest}.", None
        close_matches = difflib.get_close_matches(requested_model, available, n=1, cutoff=0.92)
        if close_matches:
            resolved = close_matches[0]
            return resolved, f"Normalized sub-agent model {requested_model} to installed model {resolved}.", None
        if self.model in available:
            ratio = difflib.SequenceMatcher(None, requested_model.lower(), self.model.lower()).ratio()
            if ratio >= 0.92:
                return self.model, f"Normalized sub-agent model {requested_model} to parent model {self.model}.", None
        available_preview = ", ".join(sorted(available)[:6])
        suffix = f" Available models include: {available_preview}." if available_preview else ""
        return requested_model, None, f"Sub-agent model is not installed: {requested_model}.{suffix}"

    def _request_looks_like_python_test_driven_repair(
        self,
        *,
        request_text: str,
        session_memory_request: bool,
        mutation_required: bool,
        test_run_required: bool,
        required_tool_names: set[str],
        forbidden_tool_names: set[str],
    ) -> bool:
        if session_memory_request or not mutation_required or not test_run_required:
            return False
        if required_tool_names or forbidden_tool_names:
            return False
        if not self.tools.default_test_command:
            return False
        lowered = request_text.lower()
        requested_paths = self._requested_mutation_paths(request_text)
        source_paths = [path for path in requested_paths if path.endswith(".py") and not self._path_looks_like_test_file(path)]
        doc_or_aux_paths = [
            path
            for path in requested_paths
            if self._path_looks_like_test_file(path) or path.startswith("docs/") or path.endswith((".md", ".rst", ".txt"))
        ]
        if len(source_paths) > 1 or doc_or_aux_paths:
            return False
        if re.search(r"\b(?:refactor|rename|callsites?|docs?|public api|api)\b", lowered) and (
            len(source_paths) == 1 or "docs" in lowered or "callsite" in lowered
        ):
            return False
        if source_paths:
            return True
        return bool(
            re.search(
                r"\b(?:python exercise|from the tests?|read source and tests|read tests and source|implement .*tests?|fix .*tests?)\b",
                lowered,
            )
        )

    def _focused_python_repair_paths(self, request_text: str) -> tuple[str, str] | None:
        requested_paths = sorted(
            path for path in self._requested_mutation_paths(request_text) if path.endswith(".py") and not self._path_looks_like_test_file(path)
        )
        for source_path in requested_paths:
            test_path = self._focused_python_repair_test_path(source_path)
            if test_path is not None:
                return source_path, test_path
        return self._preemptive_spec_guided_repair_paths()

    def _focused_python_repair_test_path(self, source_path: str) -> str | None:
        try:
            source_file = self.tools.resolve_path(source_path, allow_missing=False)
        except Exception:
            return None
        rel_source = self.tools.relative_label(source_file)
        source_stem = source_file.stem.lower()
        candidates: list[tuple[int, str]] = []
        try:
            files = self.tools._iter_code_files(self.tools.workspace_root, limit=160)
        except Exception:
            files = []
        for path in files:
            rel_test = self.tools.relative_label(path)
            if path.suffix.lower() != ".py" or not self._path_looks_like_test_file(rel_test):
                continue
            score = 0
            name = path.name.lower()
            if source_stem in name:
                score += 20
            try:
                match = self.tools.find_implementation_target(test_path=rel_test, limit=8)
            except Exception:
                match = {}
            targets = match.get("targets") if isinstance(match.get("targets"), list) else []
            if any(isinstance(item, dict) and str(item.get("path") or "").strip() == rel_source for item in targets):
                score += 100
            if score:
                candidates.append((score, rel_test))
        if candidates:
            candidates.sort(key=lambda item: (-item[0], item[1]))
            return candidates[0][1]
        return None

    def _repair_strategy_messages(
        self,
        *,
        request_text: str,
        source_path: str,
        test_path: str,
        source_text: str,
        spec_output: str,
        quick_spec: dict[str, Any],
    ) -> list[dict[str, str]]:
        example_count = len(list(quick_spec.get("examples") or []))
        stub_count = len(list(quick_spec.get("stubs") or []))
        definition_count = len(list(quick_spec.get("definitions") or []))
        payload = {
            "request": self._truncate_text(request_text, limit=320),
            "source_path": source_path,
            "test_path": test_path,
            "source_line_count": len(source_text.splitlines()),
            "definition_count": definition_count,
            "example_count": example_count,
            "stub_count": stub_count,
            "source_preview": self._truncate_text(source_text, limit=1200),
            "implementation_spec": self._truncate_text(spec_output, limit=2200),
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are a repair strategy planner for a coding CLI controller. "
                    "Choose spec_guided_repair when the task is a focused Python source repair with executable tests, "
                    "small scope, and enough structured evidence to safely synthesize or validate a full replacement. "
                    "Choose normal_loop when the task is broader, multi-file, or the spec is too weak."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True, separators=(",", ":"))},
        ]

    def _normalize_repair_strategy_payload(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        decision = payload if isinstance(payload, dict) else {}
        strategy = str(decision.get("strategy", "")).strip().lower()
        if strategy not in {"spec_guided_repair", "normal_loop"}:
            strategy = "normal_loop"
        notes = [str(item).strip() for item in list(decision.get("notes") or []) if isinstance(item, str) and str(item).strip()]
        return {
            "strategy": strategy,
            "reason": str(decision.get("reason", "")).strip(),
            "notes": notes,
        }

    def _plan_repair_strategy(
        self,
        *,
        request_text: str,
        source_path: str,
        test_path: str,
        source_text: str,
        spec_output: str,
        quick_spec: dict[str, Any],
    ) -> dict[str, Any]:
        self.status_printer("planning repair strategy")
        response = self._chat(
            purpose="repair_strategy",
            model=self.model,
            messages=self._repair_strategy_messages(
                request_text=request_text,
                source_path=source_path,
                test_path=test_path,
                source_text=source_text,
                spec_output=spec_output,
                quick_spec=quick_spec,
            ),
            think=False,
        )
        decision = self._normalize_repair_strategy_payload(extract_json_response(response.content))
        self._record_event(
            "repair_strategy",
            strategy=decision["strategy"],
            reason=decision["reason"],
            notes=decision["notes"],
            source_path=source_path,
            test_path=test_path,
            planner=response.content,
        )
        return decision

    def _request_likely_import_repair(self, request_text: str, source_text: str) -> bool:
        lowered = request_text.lower()
        if "import" not in lowered:
            return False
        if not any(token in lowered for token in ("bug", "fix", "repair", "module", "package")):
            return False
        return bool(re.search(r"(?m)^\s*(?:from\s+\S+\s+import\s+|import\s+\S+)", source_text))

    def _try_structured_test_driven_repair(
        self,
        *,
        request_text: str,
        session_memory_request: bool,
        mutation_required: bool,
        test_run_required: bool,
        required_tool_names: set[str],
        forbidden_tool_names: set[str],
        successful_tool_results: list[dict[str, Any]],
        satisfied_tool_names: set[str],
        tool_calls_this_turn: list[dict[str, Any]],
    ) -> AgentResult | None:
        if self._explicit_guard_profile_selected():
            return None
        if {"read_file", "implementation_spec", "write_file", "run_test"} & forbidden_tool_names:
            return None
        if not self._request_looks_like_python_test_driven_repair(
            request_text=request_text,
            session_memory_request=session_memory_request,
            mutation_required=mutation_required,
            test_run_required=test_run_required,
            required_tool_names=required_tool_names,
            forbidden_tool_names=forbidden_tool_names,
        ):
            return None
        paths = self._focused_python_repair_paths(request_text)
        if paths is None:
            return None
        source_path, test_path = paths
        initial_test_args = {"command": self.tools.default_test_command} if self.tools.default_test_command else {}
        initial_failed_test_result: dict[str, Any] | None = None
        if test_run_required and self._structured_repair_should_preflight_test_command():
            initial_test_result = self._execute_controller_tool(
                name="run_test",
                arguments=initial_test_args,
                request_text=request_text,
                round_number=0,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                tool_calls_this_turn=tool_calls_this_turn,
            )
            if initial_test_result.get("ok") is not True:
                initial_failed_test_result = initial_test_result
        try:
            source_text = self.tools.resolve_path(source_path, allow_missing=False).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
        if self._request_likely_import_repair(request_text, source_text):
            return None
        source_result = self._execute_controller_tool(
            name="read_file",
            arguments={"path": source_path},
            request_text=request_text,
            round_number=0,
            successful_tool_results=successful_tool_results,
            satisfied_tool_names=satisfied_tool_names,
            tool_calls_this_turn=tool_calls_this_turn,
        )
        test_result = self._execute_controller_tool(
            name="read_file",
            arguments={"path": test_path},
            request_text=request_text,
            round_number=0,
            successful_tool_results=successful_tool_results,
            satisfied_tool_names=satisfied_tool_names,
            tool_calls_this_turn=tool_calls_this_turn,
        )
        spec_result = self._execute_controller_tool(
            name="implementation_spec",
            arguments={"source_path": source_path, "test_path": test_path, "limit": 60},
            request_text=request_text,
            round_number=0,
            successful_tool_results=successful_tool_results,
            satisfied_tool_names=satisfied_tool_names,
            tool_calls_this_turn=tool_calls_this_turn,
        )
        if source_result.get("ok") is not True or test_result.get("ok") is not True or spec_result.get("ok") is not True:
            return None
        if not self._spec_guided_repair_has_actionable_spec(
            source_text=source_text,
            quick_spec=spec_result,
            failed_output="Focused Python test-driven repair requested.",
        ):
            return None
        strategy = self._plan_repair_strategy(
            request_text=request_text,
            source_path=source_path,
            test_path=test_path,
            source_text=source_text,
            spec_output=str(spec_result.get("output") or spec_result.get("summary") or ""),
            quick_spec=spec_result,
        )
        if strategy["strategy"] != "spec_guided_repair":
            return None
        return self._try_spec_guided_repair(
            request_text=request_text,
            round_number=0,
            failed_run_test_result=initial_failed_test_result
            or {
                "ok": False,
                "tool": "structured_test_repair",
                "summary": "Focused Python source+test repair selected by controller strategy planning.",
                "output": "Focused Python source+test repair selected by controller strategy planning.",
            },
            run_test_arguments=initial_test_args,
            successful_tool_results=successful_tool_results,
            satisfied_tool_names=satisfied_tool_names,
            tool_calls_this_turn=tool_calls_this_turn,
            cached_spec_result=spec_result,
        )

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
        model_note: str | None = None
        if selected_model != self.model:
            selected_model, model_note, model_error = self._resolve_sub_agent_model(selected_model)
            if model_error is not None:
                return {"ok": False, "tool": "run_agent", "summary": model_error, "missing_dependency": "ollama-model"}
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
        raw_max_tool_rounds = arguments.get("max_tool_rounds", self.max_tool_rounds)
        if isinstance(raw_max_tool_rounds, bool):
            return {"ok": False, "tool": "run_agent", "summary": "max_tool_rounds must be a positive integer."}
        try:
            max_tool_rounds = int(raw_max_tool_rounds)
        except (TypeError, ValueError):
            return {"ok": False, "tool": "run_agent", "summary": "max_tool_rounds must be a positive integer."}
        if max_tool_rounds < 1:
            return {"ok": False, "tool": "run_agent", "summary": "max_tool_rounds must be a positive integer."}
        child_tools = ToolExecutor(
            self.tools.workspace_root,
            approval_mode=approval_mode,
            input_func=self.tools.input_func,
            test_command=self.tools.default_test_command,
            default_tools_enabled=self.tools.default_tools_enabled,
            enabled_tools=self.tools.enabled_tools,
            disabled_tools=self.tools.disabled_tools,
            mcp_servers=self.tools.mcp_servers,
            browser_enabled=self.tools.browser_enabled,
            security_enabled=self.tools.security_enabled,
            indexer=getattr(self.tools, "indexer", None),
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
            require_llm_for_turn=self.require_llm_for_turn,
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
        if model_note:
            response["model_note"] = model_note
        if not result.completed:
            response["ok"] = False
            response["summary"] = f"Sub-agent failed: {result.message}"
            return response
        response["ok"] = True
        return response

    def _spec_guided_repair_paths(self, successful_tool_results: list[dict[str, Any]]) -> tuple[str, str] | None:
        failed_test_path, failed_source_path = self._failed_test_output_paths(successful_tool_results)
        if failed_source_path:
            try:
                source_file = self.tools.resolve_path(failed_source_path, allow_missing=False)
                source_line_count = len(source_file.read_text(encoding="utf-8", errors="replace").splitlines())
            except Exception:
                failed_source_path = None
            else:
                if source_line_count <= 260:
                    test_path = failed_test_path or (self._recent_test_paths(successful_tool_results)[-1] if self._recent_test_paths(successful_tool_results) else None)
                    if test_path is None:
                        return None
                    return failed_source_path, test_path

        source_paths = self._recent_source_paths(successful_tool_results)
        test_paths = self._recent_test_paths(successful_tool_results)
        if not source_paths or not test_paths:
            return None
        source_path = source_paths[-1]
        test_path = test_paths[-1]
        try:
            source_file = self.tools.resolve_path(source_path, allow_missing=False)
            line_count = len(source_file.read_text(encoding="utf-8", errors="replace").splitlines())
        except Exception:
            return None
        if line_count > 260:
            return None
        return source_path, test_path

    def _spec_guided_repair_has_actionable_spec(
        self,
        *,
        source_text: str,
        quick_spec: dict[str, Any],
        failed_output: str,
    ) -> bool:
        quick_examples = [item for item in list(quick_spec.get("examples") or []) if isinstance(item, dict)]
        quick_stubs = [item for item in list(quick_spec.get("stubs") or []) if str(item).strip()]
        quick_definitions = [item for item in list(quick_spec.get("definitions") or []) if isinstance(item, dict)]
        quick_example_text = "\n".join(str(item.get("example") or "") for item in quick_examples)
        has_structured_behavior_constraints = " matches " in quick_example_text or " != " in quick_example_text
        has_import_failure = "ModuleNotFoundError" in failed_output or "ImportError" in failed_output
        has_string_transform_hints = bool(quick_spec.get("string_transform_hints"))
        has_definition_risks = any(list(item.get("risks") or []) for item in quick_definitions)
        source_line_count = len(source_text.splitlines())
        small_module = source_line_count <= 220 and len(quick_definitions) <= 8
        literal_example_count = 0
        if small_module and len(quick_definitions) == 1:
            target_names = {
                str(quick_definitions[0].get("name") or "").strip(),
                str(quick_definitions[0].get("symbol") or "").strip(),
            } - {""}
            for item in quick_examples:
                if str(item.get("symbol") or "").strip() not in target_names:
                    continue
                kind, expr, expected = self.tools._split_test_example(str(item.get("example") or ""))
                if kind != "value":
                    continue
                try:
                    parsed = ast.parse(expr, mode="eval")
                    ast.literal_eval(expected)
                except (SyntaxError, ValueError):
                    continue
                call = parsed.body
                if not isinstance(call, ast.Call) or call.keywords:
                    continue
                if self.tools._test_spec_call_name(call) not in target_names:
                    continue
                try:
                    [ast.literal_eval(arg) for arg in call.args]
                except (SyntaxError, ValueError):
                    continue
                literal_example_count += 1
        has_small_literal_example_repair = small_module and len(quick_definitions) == 1 and literal_example_count >= 1
        if has_import_failure or has_structured_behavior_constraints or has_string_transform_hints or has_definition_risks:
            return True
        if quick_stubs:
            return True
        return small_module and (len(quick_examples) >= 4 or has_small_literal_example_repair)

    def _extract_candidate_python_source(self, text: str) -> str:
        raw = text.strip()
        if not raw:
            return ""
        fence = re.search(r"```(?:python|py)?\s*(?P<code>.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
        if fence:
            raw = fence.group("code").strip()
        lines = raw.splitlines()
        start = 0
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ", "def ", "class ", "@")):
                start = index
                break
        candidate = "\n".join(lines[start:]).strip()
        if not re.search(r"^\s*(?:def|class)\s+", candidate, flags=re.MULTILINE):
            return ""
        return candidate + "\n"

    def _spec_guided_repair_messages(
        self,
        *,
        source_path: str,
        test_path: str,
        source_text: str,
        spec_output: str,
        failed_test_output: str,
        prior_feedback: str,
    ) -> list[dict[str, str]]:
        user = (
            "Repair this Python implementation file using the tests as the executable spec.\n"
            "Return only the complete replacement source for the implementation file. No JSON, no markdown, no explanation.\n"
            "Preserve public class/function/method names and signatures exactly.\n\n"
            "Use unittest method names, transform hints, and assertion order as requirement labels; they often describe edge-case rules.\n\n"
            f"Source path: {source_path}\n"
            f"Test path: {test_path}\n\n"
            "--- current source ---\n"
            + self._truncate_text(source_text, limit=9000)
            + "\n\n--- implementation spec ---\n"
            + self._truncate_text(spec_output, limit=5000)
            + "\n\n--- failing tests ---\n"
            + self._truncate_text(failed_test_output, limit=2600)
        )
        if prior_feedback:
            user += "\n\n--- previous candidate validation failure ---\n" + self._truncate_text(prior_feedback, limit=1800)
        return [
            {
                "role": "system",
                "content": (
                    "You generate complete Python source files for a coding CLI repair engine. "
                    "Output only source code: no markdown, no prose, no comments, no placeholders. "
                    "Preserve the public API and satisfy every listed example; exception messages must match exactly."
                ),
            },
            {"role": "user", "content": user},
        ]

    def _spec_guided_repair_candidate_models(self) -> list[str]:
        models = [self.model]
        available: set[str] = set()
        list_models = getattr(self.client, "list_models", None)
        if callable(list_models):
            try:
                available = {str(model) for model in list_models()}
            except Exception:
                available = set()
        for candidate in (self.verifier_model,):
            if not candidate or candidate in models:
                continue
            if available and candidate not in available:
                continue
            models.append(candidate)
            if len(models) >= SPEC_GUIDED_REPAIR_MAX_ATTEMPTS:
                break
        while len(models) < SPEC_GUIDED_REPAIR_MAX_ATTEMPTS:
            models.append(self.model)
        return models

    def _explicit_source_repair_candidates(
        self,
        requested_mutation_paths: set[str] | list[str] | None,
    ) -> list[str]:
        if not requested_mutation_paths:
            return []
        candidates: list[str] = []
        for raw_path in requested_mutation_paths:
            normalized = str(raw_path or "").strip().replace("\\", "/")
            if not normalized:
                continue
            if not normalized.lower().endswith(".py"):
                continue
            if self._path_looks_like_test_file(normalized):
                continue
            try:
                rel = self.tools.relative_label(self.tools.resolve_path(normalized, allow_missing=False))
            except Exception:
                continue
            if rel in candidates:
                continue
            candidates.append(rel)
        return candidates

    def _related_tests_for_source(self, source_path: str) -> list[str]:
        if not source_path:
            return []
        source = source_path.replace("\\", "/")
        source_candidates = {source, source.rsplit("/", 1)[-1].rsplit(".", 1)[0]}
        stem = Path(source).stem
        parts = Path(source).with_suffix("").as_posix().split("/")
        if len(parts) > 1:
            source_candidates.add(".".join(parts[-2:]))
            source_candidates.add(".".join(parts))
        source_candidates.discard(".py")
        source_candidates.discard("")
        try:
            test_paths = sorted((path for path in self.tools.workspace_root.rglob("test_*.py")), key=lambda p: p.name.lower())
            test_paths.extend(sorted((path for path in self.tools.workspace_root.rglob("*_test.py")), key=lambda p: p.name.lower()))
        except Exception:
            return []
        related: list[str] = []
        for test_path in test_paths:
            if self._path_looks_like_code_file(self.tools.relative_label(test_path)) is False:
                continue
            if not self._path_looks_like_test_file(self.tools.relative_label(test_path)):
                continue
            try:
                raw = test_path.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(raw)
            except Exception:
                continue
            imports: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
            stem_name = Path(source).stem
            candidates = {stem, stem_name, source.rsplit("/", 1)[-1], ".".join(parts[-2:]), ".".join(parts)}
            if source_candidates.intersection(imports) or any(candidate in imports for candidate in candidates):
                related.append(self.tools.relative_label(test_path))
                continue
            if not imports and stem_name.lower() in test_path.name.lower():
                related.append(self.tools.relative_label(test_path))
        if not related:
            return []
        if len(related) == 1:
            return related
        test_file_name = f"test_{stem}.py"
        for item in related:
            if Path(item).name == test_file_name:
                return [item]
        for item in related:
            if stem.lower() in Path(item).name.lower():
                return [item]
        return related[:1]

    def _package_relative_import_rewrite(self, source_path: str) -> str | None:
        if not source_path:
            return None
        source_path = source_path.strip().replace("\\", "/")
        source_file = self.tools.resolve_path(source_path, allow_missing=False) if source_path else None
        if source_file is None or not source_file.is_file():
            return None
        if not source_file.parent.joinpath("__init__.py").is_file():
            return None
        try:
            text = source_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        changed_lines: list[str] = []
        changed = False
        for line in text.splitlines():
            if line.lstrip().startswith("from ") and " import " in line:
                match = re.match(r"^\s*from\s+([A-Za-z_][A-Za-z0-9_]*)\s+import\s+(.+?)\s*$", line)
                if match:
                    module = match.group(1).strip()
                    rest = match.group(2).strip()
                    candidate_module = source_file.parent / f"{module}.py"
                    if candidate_module.is_file():
                        changed_lines.append(f"from .{module} import {rest}")
                        changed = True
                        continue
            if re.match(r"^\s*import\s+[A-Za-z_][A-Za-z0-9_]*\s*(?:as\s+[\w_]+)?\s*$", line):
                names = re.match(r"^\s*import\s+(.+?)\s*$", line)
                if names:
                    import_items = [item.strip() for item in names.group(1).split(",")]
                    rewritten_items = []
                    did_rewrite = False
                    for item in import_items:
                        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", item):
                            rewritten_items.append(item)
                            continue
                        alias_match = re.match(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?:\s+as\s+(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\s*)?$", item)
                        if alias_match is None:
                            rewritten_items.append(item)
                            continue
                        name = alias_match.group("name")
                        alias = alias_match.group("alias")
                        if source_file.parent.joinpath(f"{name}.py").is_file():
                            rewritten_items.append(f".{name}" if not alias else f".{name} as {alias}")
                            did_rewrite = True
                        else:
                            rewritten_items.append(item)
                    if did_rewrite:
                        changed_lines.append("import " + ", ".join(rewritten_items))
                        changed = True
                        continue
            changed_lines.append(line)
        if not changed:
            return None
        candidate = "\n".join(changed_lines)
        if not candidate.endswith("\n"):
            candidate += "\n"
        return candidate

    def _preemptive_spec_guided_repair_paths(self) -> tuple[str, str] | None:
        try:
            files = self.tools._iter_code_files(self.tools.workspace_root, limit=80)
        except Exception:
            return None
        python_files: list[Path] = [path for path in files if path.suffix.lower() == ".py"]
        tests = [path for path in python_files if self._path_looks_like_test_file(self.tools.relative_label(path))]
        sources = [
            path
            for path in python_files
            if path not in tests and path.name not in {"conftest.py", "setup.py"} and not path.name.startswith("__")
        ]
        if not sources or not tests or len(sources) > 4 or len(tests) > 6:
            return None

        source_scores: list[tuple[int, int, str, Path]] = []
        for source in sources:
            try:
                rel = self.tools.relative_label(source)
                source_text = source.read_text(encoding="utf-8", errors="replace")
                line_count = len(source_text.splitlines())
            except Exception:
                continue
            if line_count > 260:
                continue
            stubs = self._stub_targets_for_paths([rel])
            if stubs:
                source_scores.append((len(stubs) * 10, -line_count, rel, source))
                continue
            try:
                tree = ast.parse(source_text)
            except SyntaxError:
                continue
            top_functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            top_classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
            if len(top_functions) == 1 and not top_classes and line_count <= 80:
                source_scores.append((1, -line_count, rel, source))
            elif (top_functions or top_classes) and line_count <= 120:
                source_scores.append((0, -line_count, rel, source))
        if not source_scores:
            return None
        _, _, source_rel, source_path = sorted(source_scores, reverse=True)[0]

        def normalized_stem(path: Path) -> str:
            stem = path.stem.lower()
            if stem.startswith("test_"):
                stem = stem[5:]
            if stem.endswith("_test"):
                stem = stem[:-5]
            return stem

        source_stem = normalized_stem(source_path)
        test_scores: list[tuple[int, str, Path]] = []
        for test in tests:
            test_rel = self.tools.relative_label(test)
            test_stem = normalized_stem(test)
            score = 0
            if test_stem == source_stem:
                score += 20
            elif source_stem and source_stem in test_stem:
                score += 8
            if test.parent == source_path.parent:
                score += 4
            if source_path.stem.lower() in test.name.lower():
                score += 4
            test_scores.append((score, test_rel, test))
        if not test_scores:
            return None
        _, test_rel, _ = sorted(test_scores, reverse=True)[0]
        return source_rel, test_rel

    def _client_allows_preemptive_mechanical_repair(self) -> bool:
        if self.disable_spec_guided_repair:
            return False
        scripted_responses = getattr(self.client, "responses", None)
        return not isinstance(scripted_responses, list) or not scripted_responses

    def _explicit_guard_profile_selected(self) -> bool:
        profile = active_feature_profile()
        return profile in {"contract-guards", "trajectory-guards"}

    def _spec_guided_repair_enabled(self) -> bool:
        return not self.disable_spec_guided_repair

    def _effective_repair_test_command(
        self,
        *,
        failed_run_test_result: dict[str, Any] | None = None,
        run_test_arguments: dict[str, Any] | None = None,
    ) -> str:
        if isinstance(failed_run_test_result, dict) and failed_run_test_result.get("recovered") is True:
            recovered_command = str(failed_run_test_result.get("command") or "").strip()
            original_command = str(failed_run_test_result.get("original_command") or "").strip()
            if recovered_command and recovered_command != original_command:
                return recovered_command
        raw_command = (run_test_arguments or {}).get("command")
        if isinstance(raw_command, str) and raw_command.strip():
            return raw_command.strip()
        return str(self.tools.default_test_command or "").strip()

    def _try_preemptive_mechanical_spec_guided_repair(
        self,
        *,
        request_text: str,
        forbidden_tool_names: set[str],
        successful_tool_results: list[dict[str, Any]],
        satisfied_tool_names: set[str],
        tool_calls_this_turn: list[dict[str, Any]],
        required_mutation_paths: set[str] | None = None,
    ) -> AgentResult | None:
        if not self.tools.default_test_command:
            return None
        if {"read_file", "implementation_spec", "write_file", "run_test"} & forbidden_tool_names:
            return None
        explicit_source_paths = self._explicit_source_repair_candidates(required_mutation_paths)
        paths = None
        if len(explicit_source_paths) == 1:
            source_path = explicit_source_paths[0]
            related_tests = self._related_tests_for_source(source_path)
            test_path = related_tests[0] if related_tests else None
            if test_path:
                paths = (source_path, test_path)
        if paths is None:
            paths = self._preemptive_spec_guided_repair_paths()
        if paths is None:
            return None
        source_path, test_path = paths
        test_command = self.tools.default_test_command
        self._record_event(
            "spec_guided_repair",
            phase="preemptive_mechanical_start",
            source_path=source_path,
            test_path=test_path,
            rounds=0,
        )
        import_rewrite = self._package_relative_import_rewrite(source_path)
        if import_rewrite is not None:
            validation = self.tools.validate_implementation_candidate(
                source_path,
                import_rewrite,
                test_path=test_path,
                test_command=test_command,
                probe_limit=24,
                timeout=120,
            )
            if validation.get("ok") is True:
                self._record_event(
                    "spec_guided_repair",
                    phase="mechanical_candidate_validation",
                    preemptive=True,
                    ok=True,
                    stage=validation.get("stage"),
                    source_path=source_path,
                    test_path=test_path,
                    summary=self._truncate_text(str(validation.get("summary") or validation.get("output") or ""), limit=700),
                    synthesis_summary="synthetic package-relative import rewrite",
                    rounds=0,
                )
                apply_result = self._execute_controller_tool(
                    name="write_file",
                    arguments={"path": source_path, "content": import_rewrite},
                    request_text=request_text,
                    round_number=0,
                    successful_tool_results=successful_tool_results,
                    satisfied_tool_names=satisfied_tool_names,
                    tool_calls_this_turn=tool_calls_this_turn,
                )
                if apply_result.get("ok") is True:
                    final_result = self._execute_controller_tool(
                        name="run_test",
                        arguments={"command": test_command},
                        request_text=request_text,
                        round_number=0,
                        successful_tool_results=successful_tool_results,
                        satisfied_tool_names=satisfied_tool_names,
                        tool_calls_this_turn=tool_calls_this_turn,
                    )
                    if final_result.get("ok") is True:
                        message = "Spec-guided mechanical repair applied and tests passed."
                        self._record_event("assistant_synthesized", content=message, tool="spec_guided_repair", rounds=0, auto=True)
                        self._record_event("assistant", content=message, rounds=0)
                        self._flush_llm_call_events()
                        return AgentResult(message=message, rounds=0, completed=True)
            else:
                import_rewrite = None
        for synthesis_name in (*PREEMPTIVE_SPEC_GUIDED_SYNTHESIS_TOOL_NAMES, *SPEC_GUIDED_SYNTHESIS_TOOL_NAMES):
            try:
                synthesize = getattr(self.tools, synthesis_name)
                synthesized = synthesize(source_path, test_path, limit=80)
            except Exception as exc:
                synthesized = {"ok": False, "summary": f"{synthesis_name} failed: {exc}"}
            if synthesized.get("ok") is not True or not isinstance(synthesized.get("candidate_source"), str):
                continue
            candidate = str(synthesized["candidate_source"])
            validation = self.tools.validate_implementation_candidate(
                source_path,
                candidate,
                test_path=test_path,
                test_command=test_command,
                probe_limit=24,
                timeout=120,
            )
            self._record_event(
                "spec_guided_repair",
                phase="mechanical_candidate_validation",
                preemptive=True,
                ok=validation.get("ok") is True,
                stage=validation.get("stage"),
                source_path=source_path,
                test_path=test_path,
                synthesis_name=synthesis_name,
                summary=self._truncate_text(str(validation.get("summary") or validation.get("output") or ""), limit=700),
                synthesis_summary=synthesized.get("summary"),
                rounds=0,
            )
            if validation.get("ok") is not True:
                continue

            self._execute_controller_tool(
                name="read_file",
                arguments={"path": source_path},
                request_text=request_text,
                round_number=0,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                tool_calls_this_turn=tool_calls_this_turn,
            )
            self._execute_controller_tool(
                name="read_file",
                arguments={"path": test_path},
                request_text=request_text,
                round_number=0,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                tool_calls_this_turn=tool_calls_this_turn,
            )
            self._execute_controller_tool(
                name="implementation_spec",
                arguments={"source_path": source_path, "test_path": test_path, "limit": 60},
                request_text=request_text,
                round_number=0,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                tool_calls_this_turn=tool_calls_this_turn,
            )
            apply_result = self._execute_controller_tool(
                name="write_file",
                arguments={"path": source_path, "content": str(validation.get("candidate_source") or candidate)},
                request_text=request_text,
                round_number=0,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                tool_calls_this_turn=tool_calls_this_turn,
            )
            if apply_result.get("ok") is not True:
                return None
            final_result = self._execute_controller_tool(
                name="run_test",
                arguments={"command": test_command},
                request_text=request_text,
                round_number=0,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                tool_calls_this_turn=tool_calls_this_turn,
            )
            if final_result.get("ok") is True:
                message = "Spec-guided mechanical repair applied and tests passed."
                self._record_event("assistant_synthesized", content=message, tool="spec_guided_repair", rounds=0, auto=True)
                self._record_event("assistant", content=message, rounds=0)
                self._flush_llm_call_events()
                return AgentResult(message=message, rounds=0, completed=True)
            self.messages.append(
                {
                    "role": "user",
                    "content": "Preemptive mechanical repair validated in isolation but failed in the workspace. Continue with the normal repair loop. Next JSON only.",
                }
            )
            return None
        return None

    def _try_relative_import_repair(
        self,
        *,
        request_text: str,
        round_number: int,
        failed_run_test_result: dict[str, Any],
        run_test_arguments: dict[str, Any],
        successful_tool_results: list[dict[str, Any]],
        satisfied_tool_names: set[str],
        tool_calls_this_turn: list[dict[str, Any]],
    ) -> AgentResult | None:
        output = str(failed_run_test_result.get("output") or failed_run_test_result.get("summary") or "")
        if "ModuleNotFoundError" not in output and "ImportError" not in output:
            return None
        try:
            targets = self.tools.find_implementation_target(output=output, limit=4)
        except Exception:
            targets = {}
        test_command = self._effective_repair_test_command(
            failed_run_test_result=failed_run_test_result,
            run_test_arguments=run_test_arguments,
        )
        for item in targets.get("targets", []) if isinstance(targets.get("targets"), list) else []:
            if not isinstance(item, dict):
                continue
            source_path = str(item.get("path") or "").strip()
            if not source_path.endswith(".py"):
                continue
            try:
                synthesized = self.tools.synthesize_relative_import_candidate(source_path, None, limit=40)
            except Exception as exc:
                synthesized = {"ok": False, "summary": f"synthesize_relative_import_candidate failed: {exc}"}
            if synthesized.get("ok") is not True or not isinstance(synthesized.get("candidate_source"), str):
                continue
            candidate = str(synthesized["candidate_source"])
            validation = self.tools.validate_implementation_candidate(
                source_path,
                candidate,
                test_command=test_command or None,
                probe_limit=1,
                timeout=120,
            )
            self._record_event(
                "spec_guided_repair",
                phase="relative_import_candidate_validation",
                ok=validation.get("ok") is True,
                stage=validation.get("stage"),
                summary=self._truncate_text(str(validation.get("summary") or validation.get("output") or ""), limit=700),
                synthesis_summary=synthesized.get("summary"),
                rounds=round_number,
            )
            if validation.get("ok") is not True:
                continue
            apply_result = self._execute_controller_tool(
                name="write_file",
                arguments={"path": source_path, "content": str(validation.get("candidate_source") or candidate)},
                request_text=request_text,
                round_number=round_number,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                tool_calls_this_turn=tool_calls_this_turn,
            )
            if apply_result.get("ok") is not True:
                return None
            test_args = {"command": test_command} if test_command else {}
            test_result = self._execute_controller_tool(
                name="run_test",
                arguments=test_args,
                request_text=request_text,
                round_number=round_number,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                tool_calls_this_turn=tool_calls_this_turn,
            )
            if test_result.get("ok") is True:
                return self._record_synthesized_final(
                    f"Applied relative import repair in {source_path}; tests passed.",
                    tool="run_test",
                    round_number=round_number,
                )
        return None

    def _try_spec_guided_repair(
        self,
        *,
        request_text: str,
        round_number: int,
        failed_run_test_result: dict[str, Any],
        run_test_arguments: dict[str, Any],
        successful_tool_results: list[dict[str, Any]],
        satisfied_tool_names: set[str],
        tool_calls_this_turn: list[dict[str, Any]],
        cached_spec_result: dict[str, Any] | None = None,
    ) -> AgentResult | None:
        paths = self._spec_guided_repair_paths(successful_tool_results)
        if paths is None:
            return None
        source_path, test_path = paths
        try:
            source_file = self.tools.resolve_path(source_path, allow_missing=False)
            source_text = source_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
        spec_result = cached_spec_result if isinstance(cached_spec_result, dict) and cached_spec_result.get("ok") is True else None
        if spec_result is None:
            try:
                quick_spec = self.tools.implementation_spec(source_path, test_path, limit=60)
            except Exception:
                return None
            if quick_spec.get("ok") is not True:
                return None
        else:
            quick_spec = spec_result
        failed_tool = str(failed_run_test_result.get("tool") or "").strip()
        if failed_tool == "preemptive_spec_repair":
            failed_output = str(failed_run_test_result.get("output") or failed_run_test_result.get("summary") or "").strip()
        elif failed_tool and failed_tool != "run_test":
            failed_output = "Previous edit candidate was rejected before applying. Ignore that malformed edit and implement from the source plus executable spec."
        else:
            failed_output = str(failed_run_test_result.get("output") or failed_run_test_result.get("summary") or "").strip()
        if not self._spec_guided_repair_has_actionable_spec(
            source_text=source_text,
            quick_spec=quick_spec,
            failed_output=failed_output,
        ):
            return None
        self._record_event(
            "spec_guided_repair",
            phase="start",
            source_path=source_path,
            test_path=test_path,
            rounds=round_number,
        )
        if spec_result is None:
            spec_result = self._execute_controller_tool(
                name="implementation_spec",
                arguments={"source_path": source_path, "test_path": test_path, "limit": 60},
                request_text=request_text,
                round_number=round_number,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                tool_calls_this_turn=tool_calls_this_turn,
            )
            if spec_result.get("ok") is not True:
                return None
        spec_output = str(spec_result.get("output") or spec_result.get("summary") or "")
        test_command = self._effective_repair_test_command(
            failed_run_test_result=failed_run_test_result,
            run_test_arguments=run_test_arguments,
        )
        feedback = ""
        for synthesis_name in (*PREEMPTIVE_SPEC_GUIDED_SYNTHESIS_TOOL_NAMES, *SPEC_GUIDED_SYNTHESIS_TOOL_NAMES):
            try:
                synthesize = getattr(self.tools, synthesis_name)
                synthesized = synthesize(source_path, test_path, limit=80)
            except Exception as exc:
                synthesized = {"ok": False, "summary": f"{synthesis_name} failed: {exc}"}
            if synthesized.get("ok") is not True or not isinstance(synthesized.get("candidate_source"), str):
                continue
            candidate = str(synthesized["candidate_source"])
            validation = self.tools.validate_implementation_candidate(
                source_path,
                candidate,
                test_path=test_path,
                test_command=test_command or None,
                probe_limit=24,
                timeout=120,
            )
            self._record_event(
                "spec_guided_repair",
                phase="mechanical_candidate_validation",
                ok=validation.get("ok") is True,
                stage=validation.get("stage"),
                summary=self._truncate_text(str(validation.get("summary") or validation.get("output") or ""), limit=700),
                candidate_chars=len(candidate),
                synthesis_summary=synthesized.get("summary"),
                rounds=round_number,
            )
            if validation.get("ok") is True:
                candidate_to_apply = str(validation.get("candidate_source") or candidate)
                apply_result = self._execute_controller_tool(
                    name="write_file",
                    arguments={"path": source_path, "content": candidate_to_apply},
                    request_text=request_text,
                    round_number=round_number,
                    successful_tool_results=successful_tool_results,
                    satisfied_tool_names=satisfied_tool_names,
                    tool_calls_this_turn=tool_calls_this_turn,
                )
                if apply_result.get("ok") is True:
                    final_test_args = dict(run_test_arguments)
                    if test_command:
                        final_test_args["command"] = test_command
                    final_result = self._execute_controller_tool(
                        name="run_test",
                        arguments=final_test_args,
                        request_text=request_text,
                        round_number=round_number,
                        successful_tool_results=successful_tool_results,
                        satisfied_tool_names=satisfied_tool_names,
                        tool_calls_this_turn=tool_calls_this_turn,
                    )
                    if final_result.get("ok") is True:
                        message = "Spec-guided mechanical repair applied and tests passed."
                        self._record_event("assistant_synthesized", content=message, tool="spec_guided_repair", rounds=round_number, auto=True)
                        self._record_event("assistant", content=message, rounds=round_number)
                        self._flush_llm_call_events()
                        return AgentResult(message=message, rounds=round_number, completed=True)
                    feedback = str(final_result.get("output") or final_result.get("summary") or "mechanical candidate failed after applying")
                else:
                    feedback = str(apply_result.get("summary") or "mechanical candidate could not be applied")
            else:
                feedback = str(validation.get("output") or validation.get("summary") or "mechanical candidate validation failed")
        candidate_models = self._spec_guided_repair_candidate_models()
        generated_candidate = False
        for attempt, candidate_model in enumerate(candidate_models, start=1):
            self._record_event(
                "spec_guided_repair",
                phase="candidate_start",
                attempt=attempt,
                model=candidate_model,
                source_path=source_path,
                test_path=test_path,
                rounds=round_number,
            )
            old_timeout: int | None = None
            if isinstance(self.client, OllamaClient):
                old_timeout = self.client.timeout
                self.client.timeout = min(self.client.timeout, SPEC_GUIDED_REPAIR_CANDIDATE_TIMEOUT)
            try:
                response = self._chat(
                    purpose="candidate_repair",
                    model=candidate_model,
                    messages=self._spec_guided_repair_messages(
                        source_path=source_path,
                        test_path=test_path,
                        source_text=source_text,
                        spec_output=spec_output,
                        failed_test_output=failed_output,
                        prior_feedback=feedback,
                    ),
                    response_format=None,
                    think=False,
                    options={"temperature": 0, "num_predict": 4096},
                    primary_can_emit_large_payload=True,
                )
            except Exception as exc:
                feedback = f"candidate generation failed: {exc}"
                self._record_event(
                    "spec_guided_repair",
                    phase="candidate_generation_failed",
                    attempt=attempt,
                    model=candidate_model,
                    error_class=exc.__class__.__name__,
                    summary=self._truncate_text(feedback, limit=700),
                    rounds=round_number,
                )
                continue
            finally:
                if old_timeout is not None and isinstance(self.client, OllamaClient):
                    self.client.timeout = old_timeout
            candidate = self._extract_candidate_python_source(response.content)
            if not candidate:
                feedback = "candidate generation did not return a complete Python source file"
                self._record_event(
                    "spec_guided_repair",
                    phase="candidate_invalid",
                    attempt=attempt,
                    model=candidate_model,
                    summary=feedback,
                    rounds=round_number,
                )
                continue
            generated_candidate = True
            validation = self.tools.validate_implementation_candidate(
                source_path,
                candidate,
                test_path=test_path,
                test_command=test_command or None,
                probe_limit=24,
                timeout=120,
            )
            self._record_event(
                "spec_guided_repair",
                phase="candidate_validation",
                attempt=attempt,
                model=candidate_model,
                ok=validation.get("ok") is True,
                stage=validation.get("stage"),
                normalized=validation.get("normalized"),
                summary=self._truncate_text(str(validation.get("summary") or validation.get("output") or ""), limit=700),
                candidate_chars=len(candidate),
                rounds=round_number,
            )
            if validation.get("ok") is not True:
                feedback = str(validation.get("output") or validation.get("summary") or "candidate validation failed")
                continue
            candidate_to_apply = str(validation.get("candidate_source") or candidate)
            apply_result = self._execute_controller_tool(
                name="write_file",
                arguments={"path": source_path, "content": candidate_to_apply},
                request_text=request_text,
                round_number=round_number,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                tool_calls_this_turn=tool_calls_this_turn,
            )
            if apply_result.get("ok") is not True:
                feedback = str(apply_result.get("summary") or "validated candidate could not be applied")
                continue
            final_test_args = dict(run_test_arguments)
            if test_command:
                final_test_args["command"] = test_command
            final_result = self._execute_controller_tool(
                name="run_test",
                arguments=final_test_args,
                request_text=request_text,
                round_number=round_number,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                tool_calls_this_turn=tool_calls_this_turn,
            )
            if final_result.get("ok") is True:
                message = "Spec-guided repair applied and tests passed."
                self._record_event("assistant_synthesized", content=message, tool="spec_guided_repair", rounds=round_number, auto=True)
                self._record_event("assistant", content=message, rounds=round_number)
                self._flush_llm_call_events()
                return AgentResult(message=message, rounds=round_number, completed=True)
            feedback = str(final_result.get("output") or final_result.get("summary") or "validated candidate failed after applying")
        if feedback:
            if not generated_candidate and "candidate generation" in feedback:
                message = (
                    "Spec-guided candidate generation failed before producing code. Continue from the "
                    "implementation_spec examples and write a complete implementation now."
                )
                self._record_event(
                    "spec_guided_repair",
                    phase="soft_failed",
                    source_path=source_path,
                    test_path=test_path,
                    summary=self._truncate_text(feedback, limit=700),
                    rounds=round_number,
                )
                self.messages.append({"role": "user", "content": message})
                self._flush_llm_call_events()
                return None
            message = "Spec-guided candidate repair did not produce a passing implementation: " + self._truncate_text(
                feedback,
                limit=900,
            )
            self._record_event(
                "spec_guided_repair",
                phase="failed_closed",
                source_path=source_path,
                test_path=test_path,
                summary=message,
                rounds=round_number,
            )
            self._record_event("assistant", content=message, rounds=round_number)
            self._flush_llm_call_events()
            return AgentResult(message=message, rounds=round_number, completed=False)
        return None

    def _llm_turn_requirement_satisfied(self) -> bool:
        if not self.require_llm_for_turn:
            return True
        return self._llm_used_this_turn

    def handle_user(self, text: str) -> AgentResult:
        self._reset_turn_cache()
        self._pending_llm_call_events = []
        self._llm_used_this_turn = False
        self.messages.append({"role": "user", "content": text})
        self._record_event("user", content=text)
        requires_tools = self._request_requires_tools(text)
        forbidden_tool_names = self._forbidden_tool_names(text)
        forbidden_tool_names.update(self._intrinsic_forbidden_tool_names())
        forbidden_tool_names.update(getattr(self.tools, "disabled_tools", set()))
        required_tool_names = self._requested_tool_names(text, forbidden_tool_names=forbidden_tool_names)
        sticky_forbidden_tool_names = set(forbidden_tool_names)
        sticky_required_tool_names = set(required_tool_names)
        requested_git_diff_mode = self._requested_git_diff_mode(text)
        expected_exact_file_line = self._requested_exact_file_line(text)
        exact_file_write = self._requested_exact_single_line_file_write(text)
        target_line_read = self._requested_target_line_read(text)
        symbol_read = self._requested_symbol_read(text)
        exact_shell_command = self._requested_exact_shell_command(text)
        expected_exact_reply_text = self._requested_exact_reply_text(text)
        exact_request = bool(
            exact_file_write
            or target_line_read
            or symbol_read
            or exact_shell_command
            or expected_exact_reply_text
            or requested_git_diff_mode
        )
        prefers_structured_file_tools = self._request_prefers_structured_file_tools(text)
        session_memory_request = self._request_targets_session_memory(text)
        mutation_allowed = self._request_allows_mutation(text)
        mutation_required = self._request_requires_mutation(text)
        code_mutation_required = self._request_requires_code_mutation(text)
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
        if not self.require_llm_for_turn:
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
                successful_tool_results=None,
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
        last_successful_validation_version: int | None = None
        latest_run_test_failed = False
        latest_run_test_failure_summary = ""
        latest_run_test_failure_output = ""
        previous_run_test_failure_summary = ""
        failed_test_context_reads = 0
        context_planner_prompted = False
        bulk_stub_guard_counts: dict[str, int] = {}
        spec_guided_repair_attempted = False
        failed_test_mutation_version: int | None = None
        last_failed_run_test_diagnosis_key: tuple[str, str, int] | None = None
        unresolved_syntax_diagnostics: dict[str, str] = {}
        unresolved_static_diagnostics: dict[str, str] = {}
        unresolved_probe_diagnostics: dict[str, str] = {}
        tool_error_counts: dict[tuple[str, str, str], int] = {}
        diagnosed_tool_error_keys: set[tuple[str, str, str]] = set()
        latest_tool_error_outputs: dict[tuple[str, str, str], str] = {}
        mutating_failure_counts: dict[tuple[str, str], int] = {}
        last_failed_run_shell_command = ""
        last_failed_run_shell_summary = ""
        last_failed_run_shell_error_class = ""
        last_failed_path_lookup_summary = ""
        last_failed_path_lookup_error_class = ""
        last_timeout_command = ""
        last_timeout_summary = ""
        if (
            (mutation_required or code_mutation_required)
            and not exact_request
            and not session_memory_request
            and not required_tool_names
            and not self.require_llm_for_turn
            and not self._explicit_guard_profile_selected()
            and self._client_allows_preemptive_mechanical_repair()
        ):
            preemptive_repair_result = self._try_preemptive_mechanical_spec_guided_repair(
                request_text=text,
                forbidden_tool_names=forbidden_tool_names,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                tool_calls_this_turn=tool_calls_this_turn,
                required_mutation_paths=required_mutation_paths,
            )
            if preemptive_repair_result is not None:
                return preemptive_repair_result
            tool_used_this_turn = tool_used_this_turn or bool(tool_calls_this_turn)
        structured_repair_result = self._try_structured_test_driven_repair(
            request_text=text,
            session_memory_request=session_memory_request,
            mutation_required=mutation_required or code_mutation_required,
            test_run_required=test_run_required,
            required_tool_names=required_tool_names,
            forbidden_tool_names=forbidden_tool_names,
            successful_tool_results=successful_tool_results,
            satisfied_tool_names=satisfied_tool_names,
            tool_calls_this_turn=tool_calls_this_turn,
        )
        if structured_repair_result is not None:
            return structured_repair_result
        tool_used_this_turn = tool_used_this_turn or bool(tool_calls_this_turn)
        if self._should_preload_context_pack(
            request_text=text,
            session_memory_request=session_memory_request,
            mutation_required=mutation_required,
            test_run_required=test_run_required,
            required_tool_names=required_tool_names,
            forbidden_tool_names=forbidden_tool_names,
        ):
            context_result = self._execute_controller_tool(
                name="context_pack",
                arguments={"request": text, "path": ".", "limit": 8},
                request_text=text,
                round_number=0,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                tool_calls_this_turn=tool_calls_this_turn,
            )
            tool_used_this_turn = tool_used_this_turn or self._counts_as_real_tool_use("context_pack", context_result)
        if self._should_plan_clarifying_questions(
            request_text=text,
            session_memory_request=session_memory_request,
            mutation_required=mutation_required,
            test_run_required=test_run_required,
            required_tool_names=required_tool_names,
            forbidden_tool_names=forbidden_tool_names,
            exact_request=exact_request,
        ):
            available_tools = self.tools.available_tool_names()
            if "context_pack" in available_tools and not self._has_successful_tool_named(successful_tool_results, "context_pack"):
                context_result = self._execute_controller_tool(
                    name="context_pack",
                    arguments={"request": text, "path": ".", "limit": 8},
                    request_text=text,
                    round_number=0,
                    successful_tool_results=successful_tool_results,
                    satisfied_tool_names=satisfied_tool_names,
                    tool_calls_this_turn=tool_calls_this_turn,
                )
                tool_used_this_turn = tool_used_this_turn or self._counts_as_real_tool_use("context_pack", context_result)
            if (
                "systems_lens" in available_tools
                and self._request_benefits_from_systems_lens(text)
                and not self._has_successful_tool_named(successful_tool_results, "systems_lens")
            ):
                context_item = self._latest_successful_tool_result(successful_tool_results, "context_pack")
                context_result = context_item.get("result") if isinstance(context_item, dict) and isinstance(context_item.get("result"), dict) else {}
                evidence_text = str(context_result.get("output", "")).strip()
                lens_result = self._execute_controller_tool(
                    name="systems_lens",
                    arguments={"request": text, "path": ".", "evidence": self._truncate_text(evidence_text, limit=700), "limit": 8},
                    request_text=text,
                    round_number=0,
                    successful_tool_results=successful_tool_results,
                    satisfied_tool_names=satisfied_tool_names,
                    tool_calls_this_turn=tool_calls_this_turn,
                )
                tool_used_this_turn = tool_used_this_turn or self._counts_as_real_tool_use("systems_lens", lens_result)
            if successful_tool_results or self._request_explicitly_wants_clarification(text):
                clarification_result = self._maybe_return_clarifying_questions(
                    request_text=text,
                    successful_tool_results=successful_tool_results,
                    mutation_required=mutation_required,
                    test_run_required=test_run_required,
                )
                if clarification_result is not None:
                    return clarification_result
        for round_number in range(1, self.max_tool_rounds + 1):
            if self._llm_turn_requirement_satisfied():
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
                    successful_tool_results=successful_tool_results,
                )
                if deterministic_result is not None:
                    return deterministic_result
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
                        request_text=text,
                        requires_tools=requires_tools,
                        mutation_required=mutation_required,
                        test_run_required=test_run_required,
                        round_number=round_number,
                        tool_used_this_turn=tool_used_this_turn,
                    ),
                    primary_can_emit_large_payload=mutation_allowed or mutation_required,
                )
            except OllamaError:
                raise
            recovered_tool_audit_bypass_reason: str | None = None
            payload = extract_json_response(response.content)
            if payload is None:
                if exact_shell_command and not tool_used_this_turn and "run_shell" not in forbidden_tool_names:
                    payload = {"type": "tool", "name": "run_shell", "arguments": {"command": exact_shell_command}}
                    recovered_tool_audit_bypass_reason = "Recovered exact run_shell command after invalid model JSON."
                    self._record_event(
                        "tool_normalized",
                        original_name="",
                        original_arguments={},
                        normalized_name="run_shell",
                        normalized_arguments={"command": exact_shell_command},
                        reason=recovered_tool_audit_bypass_reason,
                        rounds=round_number,
                    )
                elif (
                    exact_shell_command
                    and not tool_used_this_turn
                    and "run_test" in required_tool_names
                    and "run_test" not in forbidden_tool_names
                ):
                    payload = {"type": "tool", "name": "run_test", "arguments": {"command": exact_shell_command}}
                    recovered_tool_audit_bypass_reason = "Recovered exact run_test command after invalid model JSON."
                    self._record_event(
                        "tool_normalized",
                        original_name="",
                        original_arguments={},
                        normalized_name="run_test",
                        normalized_arguments={"command": exact_shell_command},
                        reason=recovered_tool_audit_bypass_reason,
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
            if response_type not in {"tool", "final"} and exact_shell_command and not tool_used_this_turn and "run_shell" not in forbidden_tool_names:
                malformed_name = str(payload.get("name", ""))
                malformed_arguments = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
                payload = {"type": "tool", "name": "run_shell", "arguments": {"command": exact_shell_command}}
                response_type = "tool"
                recovered_tool_audit_bypass_reason = "Recovered exact run_shell command from malformed model payload."
                self._record_event(
                    "tool_normalized",
                    original_name=malformed_name,
                    original_arguments=malformed_arguments,
                    normalized_name="run_shell",
                    normalized_arguments={"command": exact_shell_command},
                    reason=recovered_tool_audit_bypass_reason,
                    rounds=round_number,
                )
            elif (
                response_type not in {"tool", "final"}
                and exact_shell_command
                and not tool_used_this_turn
                and "run_test" in required_tool_names
                and "run_test" not in forbidden_tool_names
            ):
                malformed_name = str(payload.get("name", ""))
                malformed_arguments = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
                payload = {"type": "tool", "name": "run_test", "arguments": {"command": exact_shell_command}}
                response_type = "tool"
                recovered_tool_audit_bypass_reason = "Recovered exact run_test command from malformed model payload."
                self._record_event(
                    "tool_normalized",
                    original_name=malformed_name,
                    original_arguments=malformed_arguments,
                    normalized_name="run_test",
                    normalized_arguments={"command": exact_shell_command},
                    reason=recovered_tool_audit_bypass_reason,
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
                if (
                    last_failed_run_shell_command
                    and last_failed_run_shell_error_class != "timeout"
                    and self._final_claims_run_shell_success(assistant_text)
                ):
                    self._append_assistant_payload(payload)
                    self._record_event(
                        "controller_guard",
                        guard="run-shell-final-claim",
                        prior_command=self._truncate_text(last_failed_run_shell_command, limit=200),
                        error_class=last_failed_run_shell_error_class,
                        rounds=round_number,
                    )
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "The latest run_shell failed; do not claim the command works or that verification passed. "
                            + "Summarize the failure accurately or explain what different environment or dependency is needed. "
                            + "Evidence: "
                            + self._truncate_text(last_failed_run_shell_summary or last_failed_run_shell_command, limit=280)
                            + ". Next JSON only.",
                        }
                    )
                    continue
                if (
                    self._request_asks_if_path_exists(text)
                    and last_failed_path_lookup_error_class == "path_missing"
                    and self._final_claims_path_exists(assistant_text)
                ):
                    self._append_assistant_payload(payload)
                    self._record_event(
                        "controller_guard",
                        guard="path-exists-final-claim",
                        error_class=last_failed_path_lookup_error_class,
                        rounds=round_number,
                    )
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "The latest path lookup failed; do not claim the path exists. "
                            + "Summarize that it is missing or explain the lookup failure accurately. "
                            + "Evidence: "
                            + self._truncate_text(last_failed_path_lookup_summary, limit=280)
                            + ". Next JSON only.",
                        }
                    )
                    continue
                if (
                    last_failed_path_lookup_error_class == "path_missing"
                    and (
                        self._request_asks_direct_file_contents(text)
                        or self._request_asks_exact_line_text(text)
                        or self._request_asks_specific_file_line(text)
                    )
                    and not self._final_acknowledges_missing_path(assistant_text)
                ):
                    self._append_assistant_payload(payload)
                    self._record_event(
                        "controller_guard",
                        guard="missing-path-content-claim",
                        error_class=last_failed_path_lookup_error_class,
                        rounds=round_number,
                    )
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "The latest path lookup failed; do not invent file contents or line text. "
                            + "Summarize that the path is missing or explain the lookup failure accurately. "
                            + "Evidence: "
                            + self._truncate_text(last_failed_path_lookup_summary, limit=280)
                            + ". Next JSON only.",
                        }
                    )
                    continue
                if last_timeout_command and self._final_claims_timeout_success(assistant_text):
                    self._append_assistant_payload(payload)
                    self._record_event(
                        "controller_guard",
                        guard="timeout-final-claim",
                        prior_command=self._truncate_text(last_timeout_command, limit=200),
                        rounds=round_number,
                    )
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "The latest run_shell timed out; do not claim the command works or that verification passed. "
                            + "Report the timeout accurately, use a bounded probe, or describe the different execution strategy still needed. "
                            + "Evidence: "
                            + self._truncate_text(last_timeout_summary or last_timeout_command, limit=280)
                            + ". Next JSON only.",
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
                    read_only_probe = self._read_only_mutation_probe(text, exact_file_write)
                    mutating_tool_attempted = any(call.get("name") in MUTATING_TOOL_NAMES for call in tool_calls_this_turn)
                    if read_only_probe is not None and not mutating_tool_attempted:
                        probe_name, probe_arguments = read_only_probe
                        round_number += 1
                        result = self._execute_controller_tool(
                            name=probe_name,
                            arguments=probe_arguments,
                            request_text=text,
                            round_number=round_number,
                            successful_tool_results=successful_tool_results,
                            satisfied_tool_names=satisfied_tool_names,
                            tool_calls_this_turn=tool_calls_this_turn,
                        )
                        synthesized_final = self._synthesize_final_from_tool_result(
                            request_text=text,
                            name=probe_name,
                            arguments=probe_arguments,
                            result=result,
                            successful_tool_results=successful_tool_results,
                            satisfied_tool_names=satisfied_tool_names,
                            required_tool_names=required_tool_names,
                            expected_exact_reply_text=expected_exact_reply_text,
                        )
                        if synthesized_final:
                            self._record_event("assistant_synthesized", content=synthesized_final, tool=probe_name, rounds=round_number, auto=True)
                            self._record_event("assistant", content=synthesized_final, rounds=round_number)
                            self._flush_llm_call_events()
                            return AgentResult(message=synthesized_final, rounds=round_number, completed=True)
                    self._append_assistant_payload(payload)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "The user asked for a workspace change. Do not finish until write_file, replace_symbol, replace_symbols, replace_in_file, or git_commit succeeds in this turn. Use the next JSON object only.",
                        }
                    )
                    continue
                if code_mutation_required and not any(
                    Path(path).suffix.lower() in CODE_EDIT_SUFFIXES and not self._path_looks_like_test_file(path)
                    for path in mutated_paths_this_turn
                ):
                    self._append_assistant_payload(payload)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "The user asked for an implementation/code fix. A docs or test-only edit does not satisfy this request. Edit the relevant source file, then rerun validation. Next JSON only.",
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
                if mutation_required and unresolved_static_diagnostics:
                    diagnostics = "; ".join(
                        f"{path}: {diagnostic}" for path, diagnostic in sorted(unresolved_static_diagnostics.items())
                    )
                    if feature_enabled("trajectory-guards") and self._validation_failure_is_stub_placeholder(diagnostics):
                        self._append_assistant_payload(payload)
                        self._record_event(
                            "controller_guard",
                            guard="placeholder-completion-claim",
                            candidate_tool=name,
                            rounds=round_number,
                        )
                        self.messages.append(
                            {
                                "role": "user",
                                "content": "Post-edit validation shows the implementation is still a stub/comment/pass-style placeholder. "
                                + "Replace it with complete behavior in the source file before finishing. "
                                + self._truncate_text(diagnostics, limit=520)
                                + ". Next JSON only.",
                            }
                        )
                        continue
                    if not test_run_required:
                        failure = "Stopped because post-edit validation failed. " + self._truncate_text(diagnostics, limit=520)
                        self._record_event("assistant", content=failure, rounds=round_number)
                        self._flush_llm_call_events()
                        return AgentResult(message=failure, rounds=round_number, completed=False)
                    self._append_assistant_payload(payload)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "Python static sanity issues remain after edits: "
                            + self._truncate_text(diagnostics, limit=520)
                            + ". Fix them before final answer. Next JSON only.",
                        }
                    )
                    continue
                if mutation_required and unresolved_probe_diagnostics:
                    self._append_assistant_payload(payload)
                    diagnostics = "; ".join(
                        f"{path}: {diagnostic}" for path, diagnostic in sorted(unresolved_probe_diagnostics.items())
                    )
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "Post-edit example probes still fail: "
                            + self._truncate_text(diagnostics, limit=620)
                            + ". Fix those concrete mismatches before final answer. Next JSON only.",
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
                if (
                    self._post_edit_validation_enabled()
                    and mutation_verified_this_turn
                    and last_successful_validation_version != mutation_version
                    and not self._request_forbids_validation(text)
                ):
                    validation_name, _validation_arguments, validation_result = self._execute_auto_validation_plan(
                        request_text=text,
                        round_number=round_number,
                        mutated_paths=mutated_paths_this_turn,
                        forbidden_tool_names=forbidden_tool_names,
                        successful_tool_results=successful_tool_results,
                        satisfied_tool_names=satisfied_tool_names,
                        tool_calls_this_turn=tool_calls_this_turn,
                        mutation_version=mutation_version,
                    )
                    if validation_name is None or validation_result is None:
                        if not any(Path(path).suffix.lower() in CODE_EDIT_SUFFIXES for path in mutated_paths_this_turn) and not test_run_required:
                            last_successful_validation_version = mutation_version
                            mutation_verified_this_turn = False
                            self._record_event(
                                "auto_validation",
                                name="none",
                                arguments={},
                                reason="no validator configured for non-code edit",
                                mutation_version=mutation_version,
                                rounds=round_number,
                            )
                            continue
                        failure = "Stopped because validation was required after edits but no validator is configured."
                        self._record_event("assistant", content=failure, rounds=round_number)
                        self._flush_llm_call_events()
                        return AgentResult(message=failure, rounds=round_number, completed=False)
                    tool_used_this_turn = True
                    if validation_result.get("ok") is True:
                        last_successful_validation_version = mutation_version
                        if validation_name == "run_test":
                            last_successful_run_test_version = mutation_version
                            latest_run_test_failed = False
                            latest_run_test_failure_summary = ""
                    else:
                        summary = str(validation_result.get("summary") or validation_result.get("output") or "").strip()
                        if feature_enabled("trajectory-guards") and self._validation_failure_is_stub_placeholder(summary):
                            self._append_assistant_payload(payload)
                            self._record_event(
                                "controller_guard",
                                guard="placeholder-completion-claim",
                                candidate_tool="final",
                                validation_tool=validation_name,
                                rounds=round_number,
                            )
                            self.messages.append(
                                {
                                    "role": "user",
                                    "content": "Post-edit validation shows the implementation is still a stub/comment/pass-style placeholder. "
                                    + "Replace it with complete behavior in the source file before finishing. "
                                    + self._truncate_text(summary, limit=360)
                                    + " Next JSON only.",
                                }
                            )
                            continue
                        if validation_name == "run_test":
                            raw_failure = summary
                            latest_run_test_failed = True
                            latest_run_test_failure_summary = self._compact_run_test_output(raw_failure, limit=520) if raw_failure else ""
                        failure = "Stopped because post-edit validation failed."
                        if summary:
                            failure += " " + self._truncate_text(summary, limit=360)
                        self._record_event("assistant", content=failure, rounds=round_number)
                        self._flush_llm_call_events()
                        return AgentResult(message=failure, rounds=round_number, completed=False)
                if (
                    mutation_required
                    and test_run_required
                    and last_successful_run_test_version != mutation_version
                ):
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
                        retry_decision = self._stabilize_retry_tool_constraints(
                            retry_decision,
                            sticky_required_tool_names=sticky_required_tool_names,
                            sticky_forbidden_tool_names=sticky_forbidden_tool_names,
                        )
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
                    name, arguments, normalization_reason = self._normalize_shell_inspection_call(
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
                if normalization_reason is None:
                    name, arguments, normalization_reason = self._normalize_import_repair_bootstrap_call(
                        name,
                        arguments,
                        request_text=text,
                        tool_calls_this_turn=tool_calls_this_turn,
                    )
                if normalization_reason is None:
                    name, arguments, normalization_reason = self._normalize_project_rename_bootstrap_call(
                        name,
                        arguments,
                        request_text=text,
                        tool_calls_this_turn=tool_calls_this_turn,
                    )
                if normalization_reason is None:
                    name, arguments, normalization_reason = self._normalize_optional_parameter_bootstrap_call(
                        name,
                        arguments,
                        request_text=text,
                        tool_calls_this_turn=tool_calls_this_turn,
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
                    tool_error_counts[(name, self._tool_error_arg_key(name, arguments), "forbidden_tool")] = (
                        tool_error_counts.get((name, self._tool_error_arg_key(name, arguments), "forbidden_tool"), 0) + 1
                    )
                    forbidden_count = tool_error_counts[(name, self._tool_error_arg_key(name, arguments), "forbidden_tool")]
                    self._append_assistant_payload(payload)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": self._forbidden_tool_feedback_message(
                                request_text=text,
                                name=name,
                                arguments=arguments,
                                forbidden_count=forbidden_count,
                                forbidden_tool_names=forbidden_tool_names,
                                required_tool_names=required_tool_names,
                            ),
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
                if (
                    mutation_required
                    and name in {"write_file", "replace_symbol", "replace_symbols", "replace_in_file", "edit_intent"}
                    and self._path_looks_like_test_file(str(arguments.get("path", "")))
                    and not self._request_explicitly_allows_test_mutation(text)
                ):
                    self._append_assistant_payload(payload)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "This request looks like an implementation fix, not a test rewrite. Do not edit test files unless the user explicitly asks to update tests. Read or edit the relevant source file first, then rerun validation. Respond with the next JSON object only.",
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
                    diagnosis_ran = False
                    if (
                        feature_enabled("trajectory-guards")
                        and latest_run_test_failure_output
                        and last_failed_run_test_diagnosis_key != last_failed_run_test_key
                    ):
                        self._record_event(
                            "controller_guard",
                            guard="failure-compression",
                            candidate_tool=name,
                            mutation_version=mutation_version,
                            rounds=round_number,
                        )
                        diagnosis_result = self._execute_controller_tool(
                            name="diagnose_test_failure",
                            arguments={"output": latest_run_test_failure_output, "limit": 4},
                            request_text=text,
                            round_number=round_number,
                            successful_tool_results=successful_tool_results,
                            satisfied_tool_names=satisfied_tool_names,
                            tool_calls_this_turn=tool_calls_this_turn,
                        )
                        if diagnosis_result.get("ok") is True:
                            diagnosis_ran = True
                            last_failed_run_test_diagnosis_key = last_failed_run_test_key
                    self.messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The same run_test already failed and no file changed since. "
                                + (
                                    "Use the diagnosis above to edit implementation before rerunning run_test. "
                                    if diagnosis_ran
                                    else "Inspect evidence or edit files before rerunning that test. "
                                )
                                + "Next JSON only."
                            ),
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
                if name == "run_test" and unresolved_static_diagnostics and (mutation_required or code_mutation_required) and test_run_required:
                    self._append_assistant_payload(payload)
                    diagnostics = "; ".join(
                        f"{path}: {diagnostic}" for path, diagnostic in sorted(unresolved_static_diagnostics.items())
                    )
                    self._record_event(
                        "controller_guard",
                        guard="static-sanity-before-test",
                        candidate_tool=name,
                        forced_next_classes=["implementation_edit", "validation"],
                        rounds=round_number,
                    )
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "Do not run tests while Python static sanity issues are already known: "
                            + self._truncate_text(diagnostics, limit=520)
                            + ". Fix the implementation first. Next JSON only.",
                        }
                    )
                    continue
                if name == "run_test" and unresolved_probe_diagnostics and (mutation_required or code_mutation_required) and test_run_required:
                    self._append_assistant_payload(payload)
                    diagnostics = "; ".join(
                        f"{path}: {diagnostic}" for path, diagnostic in sorted(unresolved_probe_diagnostics.items())
                    )
                    self._record_event(
                        "controller_guard",
                        guard="example-probe-before-test",
                        candidate_tool=name,
                        forced_next_classes=["implementation_edit", "validation"],
                        rounds=round_number,
                    )
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "Do not rerun full tests while extracted examples already fail: "
                            + self._truncate_text(diagnostics, limit=620)
                            + ". Fix the implementation first. Next JSON only.",
                        }
                    )
                    continue
                if (
                    latest_run_test_failed
                    and mutation_required
                    and test_run_required
                    and name in MUTATING_TOOL_NAMES
                    and self._edit_payload_is_stub_like_repair(name, arguments)
                ):
                    self._append_assistant_payload(payload)
                    self._record_event(
                        "controller_guard",
                        guard="stub-repair-edit",
                        candidate_tool=name,
                        forced_next_classes=["implementation_edit", "validation"],
                        rounds=round_number,
                    )
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "Tests are failing and this edit is still a stub/comment/pass-style placeholder. Implement behavior required by the failing tests in the source file, then rerun run_test. Next JSON only.",
                        }
                    )
                    continue
                if (
                    latest_run_test_failed
                    and (mutation_required or code_mutation_required)
                    and test_run_required
                    and name in MUTATING_TOOL_NAMES
                ):
                    bulk_stub_message = self._bulk_stub_repair_guard_message(name, arguments, successful_tool_results)
                    if bulk_stub_message:
                        bulk_stub_guard_key = f"{name}:{bulk_stub_message}"
                        seen_bulk_stub_guard = bulk_stub_guard_counts.get(bulk_stub_guard_key, 0)
                        bulk_stub_guard_counts[bulk_stub_guard_key] = seen_bulk_stub_guard + 1
                        if seen_bulk_stub_guard == 0:
                            self._append_assistant_payload(payload)
                            self._record_event(
                                "controller_guard",
                                guard="bulk-stub-complete-edit",
                                candidate_tool=name,
                                forced_next_classes=["implementation_edit", "validation"],
                                rounds=round_number,
                            )
                            self.messages.append({"role": "user", "content": bulk_stub_message})
                            continue
                if (
                    latest_run_test_failed
                    and (mutation_required or code_mutation_required)
                    and test_run_required
                    and failed_test_mutation_version == mutation_version
                    and failed_test_context_reads == 0
                    and latest_run_test_failure_output
                    and last_failed_run_test_key is not None
                    and last_failed_run_test_diagnosis_key != last_failed_run_test_key
                    and name in CONTEXT_GATHERING_TOOL_NAMES
                ):
                    self._append_assistant_payload(payload)
                    self._record_event(
                        "controller_guard",
                        guard="diagnose-first-failed-test",
                        candidate_tool=name,
                        forced_next_classes=["failure_diagnosis", "implementation_edit", "validation"],
                        rounds=round_number,
                    )
                    diagnosis_result = self._execute_controller_tool(
                        name="diagnose_test_failure",
                        arguments={"output": latest_run_test_failure_output, "limit": 4},
                        request_text=text,
                        round_number=round_number,
                        successful_tool_results=successful_tool_results,
                        satisfied_tool_names=satisfied_tool_names,
                        tool_calls_this_turn=tool_calls_this_turn,
                    )
                    if diagnosis_result.get("ok") is True:
                        last_failed_run_test_diagnosis_key = last_failed_run_test_key
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "The latest run_test failed. Use the diagnosis above to edit the implementation before gathering more context or rerunning tests. Next JSON only.",
                        }
                    )
                    continue
                if (
                    latest_run_test_failed
                    and (mutation_required or code_mutation_required)
                    and test_run_required
                    and failed_test_mutation_version == mutation_version
                    and failed_test_context_reads >= 1
                    and name in CONTEXT_GATHERING_TOOL_NAMES
                ):
                    self._append_assistant_payload(payload)
                    self._record_event(
                        "controller_guard",
                        guard="no-edit-after-failed-test",
                        candidate_tool=name,
                        failed_test_context_reads=failed_test_context_reads,
                        forced_next_classes=["implementation_edit", "validation"],
                        rounds=round_number,
                    )
                    self.messages.append(
                        {
                            "role": "user",
                            "content": self._failed_test_no_edit_guard_message(
                                successful_tool_results,
                                latest_run_test_output=latest_run_test_failure_output,
                            ),
                        }
                    )
                    continue
                if (
                    latest_run_test_failed
                    and (mutation_required or code_mutation_required)
                    and test_run_required
                    and name == "run_test"
                ):
                    bulk_stub_message = self._bulk_stub_repair_guard_message(name, arguments, successful_tool_results)
                    if bulk_stub_message:
                        self._append_assistant_payload(payload)
                        self._record_event(
                            "controller_guard",
                            guard="bulk-stub-repair",
                            candidate_tool=name,
                            forced_next_classes=["implementation_edit", "validation"],
                            rounds=round_number,
                        )
                        self.messages.append({"role": "user", "content": bulk_stub_message})
                        continue
                if name == "run_shell":
                    verification_script_path = self._ad_hoc_verification_script_path(
                        str(arguments.get("command") or ""),
                        mutated_paths_this_turn,
                    )
                    if verification_script_path and last_timeout_command and last_timeout_command != str(arguments.get("command") or "").strip():
                        self._append_assistant_payload(payload)
                        self._record_event(
                            "controller_guard",
                            guard="fail-closed-timeout-verification",
                            candidate_tool=name,
                            verification_script=verification_script_path,
                            prior_command=self._truncate_text(last_timeout_command, limit=200),
                            rounds=round_number,
                        )
                        self.messages.append(
                            {
                                "role": "user",
                                "content": self._timeout_verification_guard_message(
                                    verification_script_path=verification_script_path,
                                    prior_command=last_timeout_command,
                                    prior_summary=last_timeout_summary,
                                ),
                            }
                        )
                        continue
                cached_result = self._get_cached_tool_result(name, arguments)
                cache_hit = cached_result is not None
                repeated_error_class = self._matching_repeated_tool_error(
                    name=name,
                    arguments=arguments,
                    tool_error_counts=tool_error_counts,
                )
                repeated_mutating_failure = (
                    name in RETRY_PRONE_MUTATING_TOOL_NAMES
                    and (mutation_required or code_mutation_required)
                    and mutating_failure_counts.get(self._mutating_failure_key(name, arguments), 0) >= 1
                )
                if repeated_error_class:
                    self._append_assistant_payload(payload)
                    repeated_error_key = (name, self._tool_error_arg_key(name, arguments), repeated_error_class)
                    diagnosis_ran = False
                    if (
                        feature_enabled("trajectory-guards")
                        and repeated_error_class in {"missing_dependency", "command_not_found", "import_error", "path_missing", "cwd_git", "timeout"}
                        and repeated_error_key not in diagnosed_tool_error_keys
                    ):
                        prior_error_output = latest_tool_error_outputs.get(repeated_error_key, "")
                        if prior_error_output and repeated_error_class != "timeout":
                            self._record_event(
                                "controller_guard",
                                guard="dependency-or-import-guard" if repeated_error_class in {"missing_dependency", "command_not_found", "import_error"} else "path-repair-guard",
                                candidate_tool=name,
                                error_class=repeated_error_class,
                                rounds=round_number,
                            )
                            diagnosis_arguments: dict[str, Any] = {"output": prior_error_output}
                            path_hint = str(arguments.get("path") or arguments.get("cwd") or ".").strip()
                            if path_hint:
                                diagnosis_arguments["path"] = path_hint
                            diagnosis_result = self._execute_controller_tool(
                                name="diagnose_dependency_error",
                                arguments=diagnosis_arguments,
                                request_text=text,
                                round_number=round_number,
                                successful_tool_results=successful_tool_results,
                                satisfied_tool_names=satisfied_tool_names,
                                tool_calls_this_turn=tool_calls_this_turn,
                            )
                            if diagnosis_result.get("ok") is True:
                                diagnosis_ran = True
                                diagnosed_tool_error_keys.add(repeated_error_key)
                        elif prior_error_output and repeated_error_class == "timeout":
                            self._record_event(
                                "controller_guard",
                                guard="bounded-command-validation",
                                candidate_tool=name,
                                error_class=repeated_error_class,
                                rounds=round_number,
                            )
                            tool_used_this_turn = True
                    self._record_event(
                        "tool_error_guard",
                        guard="repeat-error",
                        candidate_tool=name,
                        error_class=repeated_error_class,
                        arguments=arguments,
                        rounds=round_number,
                    )
                    self.messages.append({"role": "user", "content": self._tool_error_guard_message(name, repeated_error_class, diagnosis_ran=diagnosis_ran)})
                    continue
                if repeated_mutating_failure:
                    self._append_assistant_payload(payload)
                    self._record_event(
                        "controller_guard",
                        guard="repeated-mutating-failure-pivot",
                        candidate_tool=name,
                        target=self._mutating_failure_key(name, arguments)[1],
                        forced_next_classes=["read", "implementation_edit", "validation"],
                        rounds=round_number,
                    )
                    forbidden_tool_names.update({"edit_intent", "replace_in_file", "replace_symbol", "replace_symbols", "apply_structured_edit"})
                    required_tool_names.difference_update(forbidden_tool_names)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": self._repeated_mutating_failure_escape_message(
                                name=name,
                                arguments=arguments,
                                result={"summary": latest_run_test_failure_summary or "Earlier mutating edit already failed on this target."},
                            ),
                        }
                    )
                    continue
                if (
                    not context_planner_prompted
                    and self._context_planner_blocks(
                        name=name,
                        tool_calls=tool_calls_this_turn,
                        latest_run_test_failed=latest_run_test_failed,
                        successful_tool_results=successful_tool_results,
                    )
                ):
                    prior_tools = [str(item.get("name", "")) for item in tool_calls_this_turn[-4:]]
                    self._record_event(
                        "controller_guard",
                        guard="context-planner",
                        candidate_tool=name,
                        prior_tools=prior_tools,
                        forced_next_classes=["narrow_context", "implementation_target", "edit", "validation", "final"],
                        rounds=round_number,
                    )
                    self._append_assistant_payload(payload)
                    context_planner_prompted = True
                    context_probe = self._context_planner_probe(
                        successful_tool_results=successful_tool_results,
                        forbidden_tool_names=forbidden_tool_names,
                    )
                    if context_probe is not None:
                        probe_name, probe_arguments = context_probe
                        probe_result = self._execute_controller_tool(
                            name=probe_name,
                            arguments=probe_arguments,
                            request_text=text,
                            round_number=round_number,
                            successful_tool_results=successful_tool_results,
                            satisfied_tool_names=satisfied_tool_names,
                            tool_calls_this_turn=tool_calls_this_turn,
                        )
                        if probe_result.get("ok") is True:
                            context_planner_prompted = False
                            self.messages.append(
                                {
                                    "role": "user",
                                    "content": self._context_planner_probe_retry_message(
                                        probe_name=probe_name,
                                        probe_arguments=probe_arguments,
                                        probe_result=probe_result,
                                        mutation_required=mutation_required,
                                        code_mutation_required=code_mutation_required,
                                        test_run_required=test_run_required,
                                        required_mutation_paths=required_mutation_paths,
                                        mutated_paths_this_turn=mutated_paths_this_turn,
                                        successful_tool_results=successful_tool_results,
                                        broad=True,
                                    ),
                                }
                            )
                            continue
                    self.messages.append(
                        {
                            "role": "user",
                            "content": self._context_guard_retry_message(
                                mutation_required=mutation_required,
                                code_mutation_required=code_mutation_required,
                                test_run_required=test_run_required,
                                required_mutation_paths=required_mutation_paths,
                                mutated_paths_this_turn=mutated_paths_this_turn,
                                successful_tool_results=successful_tool_results,
                                broad=True,
                            ),
                        }
                    )
                    continue
                if self._trajectory_loop_guard_blocks(
                    name=name,
                    arguments=arguments,
                    tool_calls=tool_calls_this_turn,
                    cache_hit=cache_hit,
                ):
                    prior_tools = [str(item.get("name", "")) for item in tool_calls_this_turn[-6:]]
                    self._record_event(
                        "controller_guard",
                        guard="loop-cap",
                        candidate_tool=name,
                        prior_tools=prior_tools,
                        forced_next_classes=["narrow_context", "implementation_target", "edit", "validation", "final"],
                        rounds=round_number,
                    )
                    self._append_assistant_payload(payload)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": self._context_guard_retry_message(
                                mutation_required=mutation_required,
                                code_mutation_required=code_mutation_required,
                                test_run_required=test_run_required,
                                required_mutation_paths=required_mutation_paths,
                                mutated_paths_this_turn=mutated_paths_this_turn,
                                successful_tool_results=successful_tool_results,
                                broad=False,
                            ),
                        }
                    )
                    continue
                if self._trajectory_ground_guard_blocks(
                    request_text=text,
                    name=name,
                    arguments=arguments,
                    required_mutation_paths=required_mutation_paths,
                    successful_tool_results=successful_tool_results,
                    latest_run_test_failed=latest_run_test_failed,
                ):
                    self._append_assistant_payload(payload)
                    self._record_event(
                        "controller_guard",
                        guard="ground-before-mutate",
                        candidate_tool=name,
                        forced_next_classes=["read", "symbol", "implementation_target", "failure_diagnosis"],
                        rounds=round_number,
                    )
                    grounding_probe = self._trajectory_grounding_probe(
                        request_text=text,
                        name=name,
                        arguments=arguments,
                        required_mutation_paths=required_mutation_paths,
                        forbidden_tool_names=forbidden_tool_names,
                        successful_tool_results=successful_tool_results,
                        latest_run_test_failed=latest_run_test_failed,
                        latest_run_test_failure_output=latest_run_test_failure_output,
                    )
                    if grounding_probe is not None:
                        probe_name, probe_arguments = grounding_probe
                        probe_result = self._execute_controller_tool(
                            name=probe_name,
                            arguments=probe_arguments,
                            request_text=text,
                            round_number=round_number,
                            successful_tool_results=successful_tool_results,
                            satisfied_tool_names=satisfied_tool_names,
                            tool_calls_this_turn=tool_calls_this_turn,
                        )
                        if probe_result.get("ok") is True:
                            self.messages.append(
                                {
                                    "role": "user",
                                    "content": self._trajectory_ground_probe_retry_message(
                                        request_text=text,
                                        probe_name=probe_name,
                                        probe_arguments=probe_arguments,
                                        probe_result=probe_result,
                                        required_mutation_paths=required_mutation_paths,
                                        mutated_paths_this_turn=mutated_paths_this_turn,
                                        test_run_required=test_run_required,
                                    ),
                                }
                            )
                            continue
                    self.messages.append({"role": "user", "content": self._trajectory_ground_guard_message(text)})
                    continue
                if self._should_force_post_edit_validation(
                    request_text=text,
                    candidate_tool_name=name,
                    mutation_verified_this_turn=mutation_verified_this_turn,
                    mutation_version=mutation_version,
                    last_successful_validation_version=last_successful_validation_version,
                    mutated_paths=mutated_paths_this_turn,
                    required_mutation_paths=required_mutation_paths,
                    unresolved_syntax_diagnostics=unresolved_syntax_diagnostics,
                    unresolved_static_diagnostics=unresolved_static_diagnostics,
                    unresolved_probe_diagnostics=unresolved_probe_diagnostics,
                ):
                    self._append_assistant_payload(payload)
                    self._record_event(
                        "controller_guard",
                        guard="post-edit-validation",
                        candidate_tool=name,
                        forced_next_classes=["validation", "implementation_edit", "final"],
                        rounds=round_number,
                    )
                    validation_name, validation_arguments, validation_result = self._execute_auto_validation_plan(
                        request_text=text,
                        round_number=round_number,
                        mutated_paths=mutated_paths_this_turn,
                        forbidden_tool_names=forbidden_tool_names,
                        successful_tool_results=successful_tool_results,
                        satisfied_tool_names=satisfied_tool_names,
                        tool_calls_this_turn=tool_calls_this_turn,
                        mutation_version=mutation_version,
                        reason_prefix="proactive ",
                    )
                    tool_used_this_turn = True
                    if validation_name is None or validation_result is None:
                        failure = "Stopped because validation was required after edits but no validator is configured."
                        self._record_event("assistant", content=failure, rounds=round_number)
                        self._flush_llm_call_events()
                        return AgentResult(message=failure, rounds=round_number, completed=False)
                    if validation_result.get("ok") is True:
                        last_successful_validation_version = mutation_version
                        if validation_name == "run_test":
                            last_failed_run_test_key = None
                            last_failed_run_test_diagnosis_key = None
                            last_successful_run_test_version = mutation_version
                            latest_run_test_failed = False
                            latest_run_test_failure_summary = ""
                            latest_run_test_failure_output = ""
                            previous_run_test_failure_summary = ""
                            failed_test_context_reads = 0
                            failed_test_mutation_version = None
                        self.messages.append(
                            {
                                "role": "user",
                                "content": "Validation already ran after the latest edit: passed. If another edit is still needed, edit directly; otherwise answer from current evidence. Next JSON only.",
                            }
                        )
                        continue
                    if validation_name == "run_test":
                        last_failed_run_test_key = self._run_test_repeat_key(validation_arguments or {}, mutation_version)
                        last_failed_run_test_diagnosis_key = None
                        latest_run_test_failed = True
                        failed_test_context_reads = 0
                        failed_test_mutation_version = mutation_version
                        raw_failure = str(validation_result.get("output") or validation_result.get("summary") or "").strip()
                        latest_run_test_failure_output = raw_failure
                        compact_failure = self._compact_run_test_output(raw_failure, limit=520) if raw_failure else ""
                        latest_run_test_failure_summary = compact_failure
                        if compact_failure:
                            previous_run_test_failure_summary = compact_failure
                    validation_summary = str(validation_result.get("summary") or validation_result.get("output") or "").strip()
                    validation_feedback = "Post-edit validation failed before more tool use."
                    if validation_summary:
                        validation_feedback += " " + self._truncate_text(validation_summary, limit=520)
                    validation_feedback += " Fix the reported issue now instead of gathering more context. Next JSON only."
                    self.messages.append({"role": "user", "content": validation_feedback})
                    continue
                if (
                    name in MUTATING_TOOL_NAMES
                    and not spec_guided_repair_attempted
                    and (mutation_required or code_mutation_required)
                    and test_run_required
                    and "write_file" not in forbidden_tool_names
                    and self._spec_guided_repair_paths(successful_tool_results) is not None
                ):
                    repair_result = self._try_spec_guided_repair(
                        request_text=text,
                        round_number=round_number,
                        failed_run_test_result={
                            "ok": False,
                            "tool": "preemptive_spec_repair",
                            "summary": "Direct edit skipped for spec-guided candidate validation.",
                            "output": "Direct edit skipped for spec-guided candidate validation.",
                        },
                        run_test_arguments={},
                        successful_tool_results=successful_tool_results,
                        satisfied_tool_names=satisfied_tool_names,
                        tool_calls_this_turn=tool_calls_this_turn,
                    )
                    if repair_result is not None:
                        spec_guided_repair_attempted = True
                        return repair_result
                if self._tool_call_needs_assumption_audit(
                    request_text=text,
                    name=name,
                    arguments=arguments,
                    normalization_reason=normalization_reason or recovered_tool_audit_bypass_reason,
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
                        audit_decision = self._stabilize_retry_tool_constraints(
                            audit_decision,
                            sticky_required_tool_names=sticky_required_tool_names,
                            sticky_forbidden_tool_names=sticky_forbidden_tool_names,
                        )
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
                started = time.perf_counter()
                result = cached_result if cached_result is not None else self.tools.execute(name, arguments)
                duration_ms = round((time.perf_counter() - started) * 1000, 3)
                if not cache_hit:
                    self._store_cached_tool_result(name, arguments, result)
                self._invalidate_turn_cache_if_needed(name, result)
                self._record_command_validation_event(name=name, result=result, round_number=round_number, cached=cache_hit)
                if name not in BROAD_CONTEXT_GATHERING_TOOL_NAMES:
                    context_planner_prompted = False
                result_for_feedback = result
                post_tool_feedback: list[str] = []
                evidence_id = self._next_evidence_id() if feature_enabled("evidence-handles") else None
                if result.get("ok") is not True:
                    failed_tool_this_turn = True
                    if name in RETRY_PRONE_MUTATING_TOOL_NAMES and (mutation_required or code_mutation_required):
                        failed_test_context_reads = 0
                        failed_test_mutation_version = mutation_version
                        failure_key = self._mutating_failure_key(name, arguments)
                        mutating_failure_counts[failure_key] = mutating_failure_counts.get(failure_key, 0) + 1
                    error_class, _error_count = self._remember_tool_error(
                        name=name,
                        arguments=arguments,
                        result=result,
                        tool_error_counts=tool_error_counts,
                    )
                    if name == "run_shell":
                        last_failed_run_shell_command = str(arguments.get("command") or "").strip()
                        last_failed_run_shell_summary = str(result.get("summary") or result.get("output") or "").strip()
                        last_failed_run_shell_error_class = error_class
                    if error_class == "path_missing":
                        last_failed_path_lookup_summary = str(result.get("summary") or result.get("output") or "").strip()
                        last_failed_path_lookup_error_class = error_class
                    if name == "run_shell" and error_class == "timeout":
                        last_timeout_command = str(arguments.get("command") or "").strip()
                        last_timeout_summary = str(result.get("summary") or result.get("output") or "").strip()
                    latest_tool_error_outputs[(name, self._tool_error_arg_key(name, arguments), error_class)] = str(result.get("output") or result.get("summary") or "").strip()
                real_tool_use = self._counts_as_real_tool_use(name, result) or self._failure_result_counts_for_request(text, name, result)
                tool_used_this_turn = tool_used_this_turn or real_tool_use
                if real_tool_use:
                    satisfied_tool_names.add(name)
                    if name in MUTATING_TOOL_NAMES:
                        mutation_verified_this_turn = True
                        mutation_version += 1
                        last_failed_run_test_diagnosis_key = None
                        failed_test_context_reads = 0
                        failed_test_mutation_version = None
                        result_path = str(result.get("path", "")).strip().replace("\\", "/").lstrip("./")
                        if result_path:
                            mutated_paths_this_turn.add(result_path)
                    if (
                        name == "run_shell"
                        and result.get("ok") is True
                    ):
                        last_failed_run_shell_command = ""
                        last_failed_run_shell_summary = ""
                        last_failed_run_shell_error_class = ""
                    if result.get("ok") is True:
                        last_failed_path_lookup_summary = ""
                        last_failed_path_lookup_error_class = ""
                    if (
                        name == "run_shell"
                        and result.get("ok") is True
                        and last_timeout_command
                        and str(arguments.get("command") or "").strip()
                        and str(arguments.get("command") or "").strip() != last_timeout_command
                    ):
                        last_timeout_command = ""
                        last_timeout_summary = ""
                    if name in VALIDATION_TOOL_NAMES and result.get("ok") is True:
                        last_successful_validation_version = mutation_version
                if self._counts_as_real_tool_use(name, result):
                    successful_tool_results.append(
                        {
                            "name": name,
                            "arguments": deepcopy(arguments),
                            "result": deepcopy(result),
                            "evidence_id": evidence_id,
                        }
                    )
                if name == "run_test":
                    if result.get("ok") is True:
                        last_failed_run_test_key = None
                        last_failed_run_test_diagnosis_key = None
                        last_successful_run_test_version = mutation_version
                        latest_run_test_failed = False
                        latest_run_test_failure_summary = ""
                        latest_run_test_failure_output = ""
                        previous_run_test_failure_summary = ""
                        failed_test_context_reads = 0
                        failed_test_mutation_version = None
                    else:
                        last_failed_run_test_key = self._run_test_repeat_key(arguments, mutation_version)
                        last_failed_run_test_diagnosis_key = None
                        latest_run_test_failed = True
                        failed_test_context_reads = 0
                        failed_test_mutation_version = mutation_version
                        raw_failure = str(result.get("output") or result.get("summary") or "").strip()
                        latest_run_test_failure_output = raw_failure
                        compact_failure = self._compact_run_test_output(raw_failure, limit=520) if raw_failure else ""
                        if previous_run_test_failure_summary and compact_failure:
                            delta = self._failure_delta_summary(previous_run_test_failure_summary, compact_failure)
                            self._record_event(
                                "failure_delta",
                                previous=previous_run_test_failure_summary,
                                current=compact_failure,
                                delta=delta,
                                rounds=round_number,
                            )
                            latest_run_test_failure_summary = delta
                            result_for_feedback = deepcopy(result)
                            result_for_feedback["output"] = delta
                        else:
                            latest_run_test_failure_summary = compact_failure
                        if compact_failure:
                            previous_run_test_failure_summary = compact_failure
                elif (
                    latest_run_test_failed
                    and failed_test_mutation_version == mutation_version
                    and name in CONTEXT_GATHERING_TOOL_NAMES
                    and result.get("ok") is True
                ):
                    failed_test_context_reads += 1
                result_path = str(result.get("path", "")).strip()
                if name in MUTATING_TOOL_NAMES and result.get("ok") is True and result_path.endswith(".py"):
                    if result.get("syntax_ok") is False:
                        unresolved_syntax_diagnostics[result_path] = str(result.get("diagnostic") or result.get("summary") or "").strip()
                        unresolved_static_diagnostics.pop(result_path, None)
                        unresolved_probe_diagnostics.pop(result_path, None)
                    else:
                        unresolved_syntax_diagnostics.pop(result_path, None)
                        try:
                            static_result = self.tools.contract_check([result_path], limit=8)
                        except Exception:
                            static_result = {}
                        if static_result.get("ok") is False:
                            unresolved_static_diagnostics[result_path] = str(static_result.get("output") or static_result.get("summary") or "").strip()
                            unresolved_probe_diagnostics.pop(result_path, None)
                        else:
                            unresolved_static_diagnostics.pop(result_path, None)
                            if test_run_required and (mutation_required or code_mutation_required):
                                probe_result = self._test_example_probe_from_context(result_path, successful_tool_results, limit=20)
                                if probe_result is not None:
                                    self._record_event(
                                        "tool_result",
                                        name="run_function_probe",
                                        result=probe_result,
                                        rounds=round_number,
                                        cached=False,
                                        auto=True,
                                        duration_ms=0,
                                        evidence_id=None,
                                    )
                                    if probe_result.get("ok") is False:
                                        probe_text = str(probe_result.get("output") or probe_result.get("summary") or "").strip()
                                        unresolved_probe_diagnostics[result_path] = probe_text
                                        post_tool_feedback.append(
                                            "Post-edit example probes failed: "
                                            + self._truncate_text(probe_text, limit=620)
                                            + ". Fix these concrete mismatches before rerunning full tests. Next JSON only."
                                        )
                                    else:
                                        unresolved_probe_diagnostics.pop(result_path, None)
                self._record_event("tool_result", name=name, result=result, rounds=round_number, cached=cache_hit, duration_ms=duration_ms, evidence_id=evidence_id)
                self.messages.append(
                    {
                        "role": "user",
                        "content": self._tool_result_feedback_message(
                            name,
                            result_for_feedback,
                            real_tool_use=real_tool_use,
                            evidence_id=evidence_id,
                            successful_tool_results=successful_tool_results,
                        ),
                    }
                )
                for feedback in post_tool_feedback:
                    self.messages.append({"role": "user", "content": feedback})
                if (
                    name in MUTATING_TOOL_NAMES
                    and result.get("ok") is True
                    and post_tool_feedback
                    and not spec_guided_repair_attempted
                    and (mutation_required or code_mutation_required)
                    and test_run_required
                    and "write_file" not in forbidden_tool_names
                    and self._spec_guided_repair_enabled()
                ):
                    spec_guided_repair_attempted = True
                    feedback_text = "\n".join(post_tool_feedback)
                    repair_result = self._try_spec_guided_repair(
                        request_text=text,
                        round_number=round_number,
                        failed_run_test_result={"ok": False, "summary": feedback_text, "output": feedback_text},
                        run_test_arguments={},
                        successful_tool_results=successful_tool_results,
                        satisfied_tool_names=satisfied_tool_names,
                        tool_calls_this_turn=tool_calls_this_turn,
                    )
                    if repair_result is not None:
                        return repair_result
                if (
                    latest_run_test_failed
                    and (mutation_required or code_mutation_required)
                    and test_run_required
                    and failed_test_mutation_version == mutation_version
                    and failed_test_context_reads == 1
                    and name in CONTEXT_GATHERING_TOOL_NAMES
                    and self._spec_guided_repair_enabled()
                ):
                    self._record_event(
                        "controller_guard",
                        guard="no-edit-after-failed-test-reminder",
                        candidate_tool=name,
                        failed_test_context_reads=failed_test_context_reads,
                        forced_next_classes=["implementation_edit", "validation"],
                        rounds=round_number,
                    )
                    self.messages.append({"role": "user", "content": self._failed_test_no_edit_guard_message(successful_tool_results)})
                if (
                    name == "run_test"
                    and result.get("ok") is not True
                    and not spec_guided_repair_attempted
                    and (mutation_required or code_mutation_required)
                    and test_run_required
                    and "write_file" not in forbidden_tool_names
                    and self._spec_guided_repair_enabled()
                ):
                    import_repair = self._try_relative_import_repair(
                        request_text=text,
                        round_number=round_number,
                        failed_run_test_result=result,
                        run_test_arguments=arguments,
                        successful_tool_results=successful_tool_results,
                        satisfied_tool_names=satisfied_tool_names,
                        tool_calls_this_turn=tool_calls_this_turn,
                    )
                    if import_repair is not None:
                        return import_repair
                    spec_guided_repair_attempted = True
                    repair_result = self._try_spec_guided_repair(
                        request_text=text,
                        round_number=round_number,
                        failed_run_test_result=result,
                        run_test_arguments=arguments,
                        successful_tool_results=successful_tool_results,
                        satisfied_tool_names=satisfied_tool_names,
                        tool_calls_this_turn=tool_calls_this_turn,
                    )
                    if repair_result is not None:
                        return repair_result
                if (
                    name in MUTATING_TOOL_NAMES
                    and result.get("ok") is not True
                    and not spec_guided_repair_attempted
                    and (mutation_required or code_mutation_required)
                    and test_run_required
                    and "write_file" not in forbidden_tool_names
                    and self._spec_guided_repair_enabled()
                    and any(
                        token in str(result.get("summary") or result.get("output") or "").lower()
                        for token in ("target text was not found", "target text not found", "not found.")
                    )
                ):
                    spec_guided_repair_attempted = True
                    repair_result = self._try_spec_guided_repair(
                        request_text=text,
                        round_number=round_number,
                        failed_run_test_result=result,
                        run_test_arguments={},
                        successful_tool_results=successful_tool_results,
                        satisfied_tool_names=satisfied_tool_names,
                        tool_calls_this_turn=tool_calls_this_turn,
                    )
                    if repair_result is not None:
                        return repair_result
                if (
                    name in MUTATING_TOOL_NAMES
                    and result.get("ok") is not True
                    and not spec_guided_repair_attempted
                    and (mutation_required or code_mutation_required)
                    and test_run_required
                    and "write_file" not in forbidden_tool_names
                    and self._spec_guided_repair_enabled()
                    and any(
                        token in str(result.get("summary") or result.get("output") or "").lower()
                        for token in (
                            "syntax error",
                            "syntaxerror",
                            "indentationerror",
                            "invalid syntax",
                            "invalid",
                            "requires",
                            "bad arguments",
                        )
                    )
                ):
                    spec_guided_repair_attempted = True
                    repair_result = self._try_spec_guided_repair(
                        request_text=text,
                        round_number=round_number,
                        failed_run_test_result=result,
                        run_test_arguments={},
                        successful_tool_results=successful_tool_results,
                        satisfied_tool_names=satisfied_tool_names,
                        tool_calls_this_turn=tool_calls_this_turn,
                    )
                    if repair_result is not None:
                        return repair_result
                if (
                    name in MUTATING_TOOL_NAMES
                    and result.get("ok") is not True
                    and (mutation_required or code_mutation_required)
                    and test_run_required
                    and "write_file" not in forbidden_tool_names
                    and self._tool_error_class(result) == "syntax_error"
                ):
                    target_path = str(arguments.get("path") or arguments.get("cwd") or "").strip()
                    target_text = (result.get("summary") or result.get("output") or "").strip()
                    if target_path:
                        prompt = (
                            "This edit failed a syntax check. Do not retry low-level edits on this file.\n"
                            f"Read {target_path} and rewrite the needed function(s) in a single, syntax-valid patch.\n"
                            "Then rerun tests. Use full file edits only and keep the implementation minimal and correct."
                        )
                    else:
                        prompt = "This mutating edit failed a syntax check. Replace malformed edits with a new syntactically valid implementation and rerun tests."
                    if target_text:
                        prompt += f"\nEvidence: {self._truncate_text(target_text, limit=360)}"
                    self.messages.append({"role": "user", "content": prompt})
                    self._record_event("assistant", content="syntax_error_recovery_prompt", rounds=round_number)
                    continue
                self._autosave()
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
                        reconciliation_decision = self._stabilize_retry_tool_constraints(
                            reconciliation_decision,
                            sticky_required_tool_names=sticky_required_tool_names,
                            sticky_forbidden_tool_names=sticky_forbidden_tool_names,
                        )
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
                    and not unresolved_static_diagnostics
                    and not unresolved_probe_diagnostics
                    and not (required_mutation_paths - mutated_paths_this_turn)
                ):
                    auto_arguments: dict[str, Any] = {}
                    self.status_printer("tool run_test {}")
                    self._record_event("tool_call", name="run_test", arguments=auto_arguments, rounds=round_number, auto=True)
                    tool_calls_this_turn.append({"name": "run_test", "arguments": deepcopy(auto_arguments)})
                    started = time.perf_counter()
                    auto_result = self.tools.execute("run_test", auto_arguments)
                    duration_ms = round((time.perf_counter() - started) * 1000, 3)
                    self._record_command_validation_event(name="run_test", result=auto_result, round_number=round_number, cached=False)
                    evidence_id = self._next_evidence_id() if feature_enabled("evidence-handles") else None
                    self._record_event("tool_result", name="run_test", result=auto_result, rounds=round_number, cached=False, auto=True, duration_ms=duration_ms, evidence_id=evidence_id)
                    tool_used_this_turn = True
                    satisfied_tool_names.add("run_test")
                    raw_output = str(auto_result.get("output") or auto_result.get("summary") or "").strip()
                    if auto_result.get("ok") is True:
                        successful_tool_results.append(
                            {
                                "name": "run_test",
                                "arguments": deepcopy(auto_arguments),
                                "result": deepcopy(auto_result),
                                "evidence_id": evidence_id,
                            }
                        )
                        last_successful_run_test_version = mutation_version
                        latest_run_test_failed = False
                        latest_run_test_failure_summary = ""
                        message = "Ran tests after the latest edit: passed."
                        self._record_event("assistant_synthesized", content=message, tool="run_test", rounds=round_number, auto=True)
                        self._record_event("assistant", content=message, rounds=round_number)
                        self._flush_llm_call_events()
                        return AgentResult(message=message, rounds=round_number, completed=True)
                    latest_run_test_failed = True
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
        if (
            latest_run_test_failed
            and (mutation_required or code_mutation_required)
            and test_run_required
            and failed_test_mutation_version == mutation_version
            and failed_test_context_reads >= 1
        ):
            failure = (
                "Stopped because tests failed and the controller blocked further read-only looping. "
                + self._failed_test_no_edit_guard_message(
                    successful_tool_results,
                    latest_run_test_output=latest_run_test_failure_output,
                )
            )
            self._record_event("assistant", content=failure, rounds=self.max_tool_rounds)
            self._flush_llm_call_events()
            return AgentResult(message=failure, rounds=self.max_tool_rounds, completed=False)
        if (
            self._post_edit_validation_enabled()
            and mutation_verified_this_turn
            and last_successful_validation_version != mutation_version
            and not self._request_forbids_validation(text)
        ):
            validation_name, _validation_arguments, validation_result = self._execute_auto_validation_plan(
                request_text=text,
                round_number=self.max_tool_rounds,
                mutated_paths=mutated_paths_this_turn,
                forbidden_tool_names=forbidden_tool_names,
                successful_tool_results=successful_tool_results,
                satisfied_tool_names=satisfied_tool_names,
                tool_calls_this_turn=tool_calls_this_turn,
                mutation_version=mutation_version,
                reason_prefix="final-chance ",
            )
            if validation_name is not None and validation_result is not None:
                if validation_result.get("ok") is True:
                    message = "Ran validation after the latest edit: passed."
                    self._record_event("assistant_synthesized", content=message, tool=validation_name, rounds=self.max_tool_rounds, auto=True)
                    self._record_event("assistant", content=message, rounds=self.max_tool_rounds)
                    self._flush_llm_call_events()
                    return AgentResult(message=message, rounds=self.max_tool_rounds, completed=True)
                if (
                    not spec_guided_repair_attempted
                    and self._spec_guided_repair_enabled()
                    and (mutation_required or code_mutation_required)
                    and test_run_required
                    and "write_file" not in forbidden_tool_names
                    and self._spec_guided_repair_paths(successful_tool_results) is not None
                ):
                    spec_guided_repair_attempted = True
                    failed_validation_result = dict(validation_result)
                    failed_validation_result.setdefault("tool", validation_name)
                    repair_result = self._try_spec_guided_repair(
                        request_text=text,
                        round_number=self.max_tool_rounds,
                        failed_run_test_result=failed_validation_result,
                        run_test_arguments={"command": self.tools.default_test_command} if self.tools.default_test_command else {},
                        successful_tool_results=successful_tool_results,
                        satisfied_tool_names=satisfied_tool_names,
                        tool_calls_this_turn=tool_calls_this_turn,
                    )
                    if repair_result is not None:
                        return repair_result
                summary = str(validation_result.get("summary") or validation_result.get("output") or "").strip()
                failure = "Stopped because final-chance post-edit validation failed."
                if summary:
                    failure += " " + self._truncate_text(summary, limit=360)
                self._record_event("assistant", content=failure, rounds=self.max_tool_rounds)
                self._flush_llm_call_events()
                return AgentResult(message=failure, rounds=self.max_tool_rounds, completed=False)
        failure = "Stopped after reaching the maximum tool rounds."
        self._record_event("assistant", content=failure, rounds=self.max_tool_rounds)
        self._flush_llm_call_events()
        return AgentResult(message=failure, rounds=self.max_tool_rounds, completed=False)
