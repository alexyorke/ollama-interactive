from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import threading

from ollama_code.ollama_client import OllamaClient, OllamaError
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
- Question your assumptions before acting.
- Identify what you are assuming, then prove or disprove it with the available tools whenever a file read, search, git inspection, test run, or shell command can verify it.
- Do not guess about workspace contents, command output, repo state, or whether an edit worked when you can check instead.
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


KNOWN_TOOL_NAMES = {tool["name"] for tool in TOOL_DESCRIPTIONS}
APPROVAL_RANK = {"read-only": 0, "ask": 1, "auto": 2}


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
        self.events: list[dict[str, Any]] = []
        self._interrupt_event: threading.Event | None = None
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
        self._autosave()

    def set_model(self, model: str) -> None:
        self.model = model

    def set_approval_mode(self, mode: str) -> None:
        self.tools.set_approval_mode(mode)

    def approval_mode(self) -> str:
        return self.tools.approval_mode

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
        self.messages.append({"role": "user", "content": text})
        self._record_event("user", content=text)
        requires_tools = self._request_requires_tools(text)
        prefers_structured_file_tools = self._request_prefers_structured_file_tools(text)
        tool_used_this_turn = False
        for round_number in range(1, self.max_tool_rounds + 1):
            self.status_printer(f"thinking with {self.model} (round {round_number}/{self.max_tool_rounds})")
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=self.messages,
                    on_thinking=self.thinking_printer,
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
                self.messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=True)})
                self._record_event("assistant", content=assistant_text, rounds=round_number)
                return AgentResult(message=assistant_text, rounds=round_number, completed=True)
            if response_type == "tool":
                name = str(payload.get("name", "")).strip()
                arguments = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
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
                self.status_printer(f"tool {name} {json.dumps(arguments, ensure_ascii=True)}")
                self.messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=True)})
                self._record_event("tool_call", name=name, arguments=arguments, rounds=round_number)
                result = self.tools.execute(name, arguments)
                real_tool_use = self._counts_as_real_tool_use(name, result)
                tool_used_this_turn = tool_used_this_turn or real_tool_use
                self._record_event("tool_result", name=name, result=result, rounds=round_number)
                follow_up = "Respond with the next JSON object only."
                if not real_tool_use:
                    follow_up = (
                        "That tool call did not complete successfully, so it does not satisfy the tool-use requirement. "
                        "Fix the issue or use a different appropriate tool, then respond with the next JSON object only."
                    )
                self.messages.append(
                    {
                        "role": "user",
                        "content": "Tool result:\n"
                        + json.dumps(result, indent=2, ensure_ascii=True)
                        + "\n"
                        + follow_up,
                    }
                )
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
