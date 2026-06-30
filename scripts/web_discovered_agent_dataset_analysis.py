from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from huggingface_hub import HfApi, hf_hub_download
except ModuleNotFoundError:
    HfApi = None
    hf_hub_download = None

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    pq = None

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import trajectory_task_review as task_review


DEFAULT_CACHE_ROOT = Path("scratch") / "external" / "datasets" / "web-discovered"
DEFAULT_OUTPUT = Path("scratch") / "external" / "datasets" / "web-discovered-agent-datasets-analysis.json"
DEFAULT_MARKDOWN_OUTPUT = DEFAULT_OUTPUT.with_suffix(".md")

DATASET_SPECS: dict[str, dict[str, Any]] = {
    "personal-codex-dataclaw": {
        "repo_id": "peteromallet/my-personal-codex-data",
        "local_dir": "peteromallet__my-personal-codex-data",
        "kind": "real-user-coding-agent-sessions",
        "format": "dataclaw-session-jsonl",
    },
    "personal-claude-code-dataclaw": {
        "repo_id": "misterkerns/my-personal-claude-code-data",
        "local_dir": "misterkerns__my-personal-claude-code-data",
        "kind": "real-user-coding-agent-sessions",
        "format": "dataclaw-session-jsonl",
    },
    "pi-mono": {
        "repo_id": "badlogicgames/pi-mono",
        "local_dir": "badlogicgames__pi-mono",
        "kind": "real-user-coding-agent-sessions",
        "format": "pi-session-jsonl",
    },
    "pi-mono-sessions": {
        "repo_id": "thomasmustier/pi-mono-sessions",
        "local_dir": "thomasmustier__pi-mono-sessions",
        "kind": "real-user-coding-agent-sessions",
        "format": "pi-session-jsonl",
    },
    "championswimmer-pi-coding-sessions": {
        "repo_id": "championswimmer/pi-coding-sessions",
        "local_dir": "championswimmer__pi-coding-sessions",
        "kind": "real-user-coding-agent-sessions",
        "format": "pi-session-jsonl",
        "session_glob": "_upload_staging/*.jsonl",
        "extra_files": ["README.md"],
    },
    "formal-web-pi-coding-sessions": {
        "repo_id": "formal-web/pi-coding-sessions",
        "local_dir": "formal-web__pi-coding-sessions",
        "kind": "real-user-coding-agent-sessions",
        "format": "pi-session-jsonl",
        "session_glob": "*.jsonl",
    },
    "pi-nes-sessions": {
        "repo_id": "thomasmustier/pi-nes-sessions",
        "local_dir": "thomasmustier__pi-nes-sessions",
        "kind": "real-user-coding-agent-sessions",
        "format": "pi-session-jsonl",
    },
    "pi-for-excel-sessions": {
        "repo_id": "thomasmustier/pi-for-excel-sessions",
        "local_dir": "thomasmustier__pi-for-excel-sessions",
        "kind": "real-user-coding-agent-sessions",
        "format": "pi-session-jsonl",
    },
    "pi-extensions-sessions": {
        "repo_id": "thomasmustier/pi-extensions-sessions",
        "local_dir": "thomasmustier__pi-extensions-sessions",
        "kind": "real-user-coding-agent-sessions",
        "format": "pi-session-jsonl",
        "extra_files": ["README.md"],
    },
    "economist-tui-sessions": {
        "repo_id": "thomasmustier/economist-tui-sessions",
        "local_dir": "thomasmustier__economist-tui-sessions",
        "kind": "real-user-coding-agent-sessions",
        "format": "pi-session-jsonl",
        "extra_files": ["README.md"],
    },
    "julien-c-pi-sessions": {
        "repo_id": "julien-c/pi-sessions",
        "local_dir": "julien-c__pi-sessions",
        "kind": "real-user-coding-agent-sessions",
        "format": "pi-session-jsonl",
        "extra_files": ["README.md"],
    },
    "pi-sessions-viewer": {
        "repo_id": "aaaaliou/pi-sessions-viewer",
        "local_dir": "aaaaliou__pi-sessions-viewer",
        "kind": "real-user-coding-agent-sessions",
        "format": "pi-session-jsonl",
        "session_glob": "*.jsonl",
        "exclude_filename_substrings": ["manifest"],
        "extra_files": ["README.md", "manifest.jsonl"],
    },
    "gradio-pi-sessions": {
        "repo_id": "abidlabs/gradio-pi-sessions",
        "local_dir": "abidlabs__gradio-pi-sessions",
        "kind": "real-user-coding-agent-sessions",
        "format": "pi-session-jsonl",
        "session_glob": "sessions/gradio/*.jsonl",
        "extra_files": ["README.md"],
    },
    "0xkobolds": {
        "repo_id": "moikapy/0xKobolds",
        "local_dir": "moikapy__0xKobolds",
        "kind": "real-user-coding-agent-sessions",
        "format": "pi-session-jsonl",
        "session_glob": "_upload_staging/*.jsonl",
        "extra_files": ["README.md"],
    },
    "fable-5-traces": {
        "repo_id": "Glint-Research/Fable-5-traces",
        "local_dir": "Glint-Research__Fable-5-traces",
        "kind": "real-user-coding-agent-sessions",
        "format": "pi-session-jsonl",
        "session_glob": "pi-traces/*.jsonl",
        "extra_files": ["README.md"],
    },
    "fable-5-claude-code-traces": {
        "repo_id": "AlinCiocan/fable-5-claude-code-traces",
        "local_dir": "AlinCiocan__fable-5-claude-code-traces",
        "kind": "real-user-coding-agent-sessions",
        "format": "claude-code-session-jsonl",
        "session_glob": "*.jsonl",
        "extra_files": ["README.md"],
    },
    "merchantscroll-traces": {
        "repo_id": "vedalken/merchantscroll-traces",
        "local_dir": "vedalken__merchantscroll-traces",
        "kind": "real-user-coding-agent-sessions",
        "format": "nested-message-session-jsonl",
        "session_glob": "*.jsonl",
        "exclude_filename_substrings": ["__subagent__"],
        "extra_files": ["README.md"],
    },
    "share-codex": {
        "repo_id": "nmuendler/share-codex",
        "local_dir": "nmuendler__share-codex",
        "kind": "real-user-coding-agent-sessions",
        "format": "share-codex-jsonl",
    },
    "ranga-coding-sessions": {
        "repo_id": "RangaPrasath/coding-sessions",
        "local_dir": "RangaPrasath__coding-sessions",
        "kind": "real-user-coding-agent-sessions",
        "format": "message-bundle-jsonl",
        "session_glob": "sessions.jsonl",
        "extra_files": ["README.md", "manifest.json"],
    },
    "alexli-coding-sessions": {
        "repo_id": "AlexLi31415/coding-sessions",
        "local_dir": "AlexLi31415__coding-sessions",
        "kind": "real-user-coding-agent-sessions",
        "format": "message-bundle-jsonl",
        "session_glob": "sessions.jsonl",
        "extra_files": ["manifest.json"],
    },
    "codex-sessions": {
        "repo_id": "cfahlgren1/codex-sessions",
        "local_dir": "cfahlgren1__codex-sessions",
        "kind": "real-user-coding-agent-sessions",
        "format": "codex-session-jsonl",
        "session_glob": "sessions.jsonl",
        "extra_files": ["README.md"],
    },
    "agent-sessions-list": {
        "repo_id": "cfahlgren1/agent-sessions-list",
        "local_dir": "cfahlgren1__agent-sessions-list",
        "kind": "real-user-coding-agent-sessions",
        "format": "mixed-agent-session-jsonl",
        "extra_files": ["README.md"],
    },
    "ultralazr-claude-code-traces": {
        "repo_id": "ultralazr/claude-code-traces",
        "local_dir": "ultralazr__claude-code-traces",
        "kind": "real-user-coding-agent-sessions",
        "format": "claude-code-session-jsonl",
    },
    "trace-commons-agent-traces": {
        "repo_id": "trace-commons/agent-traces",
        "local_dir": "trace-commons__agent-traces",
        "kind": "real-user-coding-agent-sessions",
        "format": "claude-code-session-jsonl",
        "session_glob": "sessions/claude_code/*.jsonl",
        "extra_files": ["README.md"],
    },
    "prayagmatic-agent-traces": {
        "repo_id": "Prayagmatic/agent-traces",
        "local_dir": "Prayagmatic__agent-traces",
        "kind": "real-user-coding-agent-sessions",
        "format": "pi-session-jsonl",
        "session_glob": "**/*.jsonl",
        "exclude_filename_substrings": ["manifest"],
        "extra_files": ["README.md"],
    },
    "mimo-claude-code-traces-1k": {
        "repo_id": "choucsan/mimo-claude-code-traces-1k",
        "local_dir": "choucsan__mimo-claude-code-traces-1k",
        "kind": "real-user-coding-agent-sessions",
        "format": "claude-code-session-jsonl",
        "session_glob": "session/**/*.jsonl",
        "extra_files": ["README.md"],
    },
    "ml-intern-sessions": {
        "repo_id": "lewtun/ml-intern-sessions",
        "local_dir": "lewtun__ml-intern-sessions",
        "kind": "real-user-coding-agent-sessions",
        "format": "claude-code-session-jsonl",
        "session_glob": "sessions/**/*.jsonl",
        "extra_files": ["README.md"],
    },
    "add-sam-3-lite-text-agent-traces": {
        "repo_id": "nielsr/add-sam-3-lite-text-agent-traces",
        "local_dir": "nielsr__add-sam-3-lite-text-agent-traces",
        "kind": "real-user-coding-agent-sessions",
        "format": "raw-codex-session-jsonl",
        "session_glob": "traces/**/*.jsonl",
        "extra_files": ["README.md"],
    },
    "mishig-traces": {
        "repo_id": "mishig/traces",
        "local_dir": "mishig__traces",
        "kind": "real-user-coding-agent-sessions",
        "format": "nested-message-session-jsonl",
        "session_glob": "*.jsonl",
        "extra_files": ["README.md"],
    },
    "claudeset-community": {
        "repo_id": "lelouch0110/claudeset-community",
        "local_dir": "lelouch0110__claudeset-community",
        "kind": "real-user-coding-agent-sessions",
        "format": "claudeset-community-jsonl",
        "session_glob": "data/**/*.jsonl",
        "extra_files": ["README.md"],
    },
    "novita-agentic-code-dataset-22": {
        "repo_id": "novita/agentic_code_dataset_22",
        "local_dir": "novita__agentic_code_dataset_22",
        "kind": "real-user-coding-agent-sessions",
        "format": "openai-turn-bundle-json",
        "session_glob": "e22_sessions_openai.json",
        "extra_files": ["README.md"],
    },
    "agent-coding-traces-public": {
        "repo_id": "PotatoHD/agent-coding-traces-public",
        "local_dir": "PotatoHD__agent-coding-traces-public",
        "kind": "real-user-coding-agent-sessions",
        "format": "chatml-transcript-parquet",
        "session_glob": "data/train-*.parquet",
        "extra_files": ["README.md"],
    },
    "archit11-claude-code-traces": {
        "repo_id": "archit11/claude-code-traces",
        "local_dir": "archit11__claude-code-traces",
        "kind": "real-user-coding-agent-sessions",
        "format": "embedded-conversation-json-parquet",
        "session_glob": "data/train-*.parquet",
        "extra_files": ["README.md"],
    },
    "misc-merged-claude-code-traces-v1": {
        "repo_id": "nlile/misc-merged-claude-code-traces-v1",
        "local_dir": "nlile__misc-merged-claude-code-traces-v1",
        "kind": "real-user-coding-agent-sessions",
        "format": "merged-claude-code-parquet",
        "session_glob": "data/train-*.parquet",
        "extra_files": ["README.md"],
    },
}


def _require_hf_download_support() -> tuple[type[Any], Any]:
    if HfApi is None or hf_hub_download is None:
        raise RuntimeError(
            "huggingface_hub is required for web-discovered dataset downloads. "
            "Install it with `python -m pip install huggingface_hub`."
        )
    return HfApi, hf_hub_download


def _safe_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").lower()
            if item_type in {"text", "thinking", "reasoning"}:
                text = item.get("text") if item_type != "thinking" else item.get("thinking")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return ""


def _truncate(text: str, limit: int = 220) -> str:
    value = " ".join(text.split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _looks_like_env_context(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized.startswith("<environment_context>") and normalized.endswith("</environment_context>")


def _looks_like_user_instructions(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized.startswith("<user_instructions>") and normalized.endswith("</user_instructions>")


def _looks_like_agents_instructions_wrapper(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized.startswith("# agents.md instructions for ")


def _looks_like_local_command_wrapper(text: str) -> bool:
    normalized = text.strip().lower()
    return (
        normalized.startswith("<local-command-caveat>")
        or normalized.startswith("<command-name>")
        or normalized.startswith("<command-message>")
        or normalized.startswith("<command-args>")
        or normalized.startswith("<local-command-stdout>")
        or normalized.startswith("<local-command-stderr>")
        or normalized == "[request interrupted by user for tool use]"
    )


def _looks_like_skill_wrapper(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized.startswith("base directory for this skill:") or normalized.startswith("<skill name=")


def _looks_like_command_output_wrapper(text: str) -> bool:
    normalized = " ".join(str(text or "").strip().lower().split())
    return normalized.startswith("command:") and " output:" in normalized


def _inline_user_segments(text: str) -> list[str]:
    raw = str(text or "").replace("\r", "").strip()
    if not raw:
        return []
    if "USER:" not in raw:
        return [raw]
    parts = re.split(r"(?:(?<=\n)|^)\s*USER:\s*", raw)
    segments = [part.strip() for part in parts if part.strip()]
    return segments or [raw]


def _trim_embedded_transcript_tail(text: str) -> str:
    raw = str(text or "").replace("\r", "").strip()
    if not raw:
        return ""
    marker_pattern = re.compile(
        r"(?im)^\s*(?:ASSISTANT\s*\(|TOOL RESULT:|TOOL ERROR:|SYSTEM\s*:|SYSTEM\s*\()"
    )
    match = marker_pattern.search(raw)
    if match is None:
        return raw
    return raw[: match.start()].strip()


def _strip_ide_selection_wrapper(text: str) -> str:
    raw = str(text or "").replace("\r", "").strip()
    if not raw:
        return ""
    pattern = re.compile(r"(?is)^\s*<ide_selection>.*?</ide_selection>\s*")
    return pattern.sub("", raw, count=1).strip()


def _strip_system_reminder_wrapper(text: str) -> str:
    raw = str(text or "").replace("\r", "").strip()
    if not raw:
        return ""
    pattern = re.compile(r"(?is)^\s*<system-reminder>.*?</system-reminder>\s*")
    stripped = raw
    while True:
        updated = pattern.sub("", stripped, count=1).strip()
        if updated == stripped:
            return stripped
        stripped = updated


def _strip_file_wrapper(text: str) -> str:
    raw = str(text or "").replace("\r", "").strip()
    if not raw:
        return ""
    pattern = re.compile(r'(?is)^\s*<file\s+name="[^"]+">\s*')
    stripped = pattern.sub("", raw, count=1).strip()
    stripped = re.sub(r"(?is)\s*</file>\s*$", "", stripped, count=1).strip()
    return stripped


def _strip_pi_worker_wrapper(text: str) -> str:
    raw = str(text or "").replace("\r", "").strip()
    if not raw:
        return ""
    normalized = " ".join(raw.lower().split())
    if "complete your task autonomously." not in normalized or "set the tab title using set_tab_title" not in normalized:
        return raw
    start_match = re.search(r"(?im)^\s*(implement|review|fix|build|create|compare|research|investigate|analyze|read)\b", raw)
    if start_match is None:
        return raw
    trimmed = raw[start_match.start() :].strip()
    end_match = re.search(r"(?im)^\s*(commit:|important:|your final assistant message should)\b", trimmed)
    if end_match is not None:
        trimmed = trimmed[: end_match.start()].strip()
    return trimmed


def _looks_like_spawned_agent_wrapper(text: str) -> bool:
    normalized = " ".join(str(text or "").strip().lower().split())
    if not normalized:
        return False
    return (
        normalized.startswith("# researcher agent")
        or "you were spawned for a specific purpose" in normalized
        or "don't implement solutions or make architectural decisions" in normalized
        or "use the `claude` tool for all investigation work" in normalized
    )


def _strip_continued_conversation_wrapper(text: str) -> str:
    raw = str(text or "").replace("\r", "").strip()
    if not raw:
        return ""
    marker = "This session is being continued from a previous conversation that ran out of context. The conversation is summarized below:"
    marker_index = raw.find(marker)
    if marker_index != -1:
        raw = raw[marker_index + len(marker) :].strip()
    trailing_marker = "Please continue the conversation from where we left it off without asking the user any further questions."
    trailing_index = raw.find(trailing_marker)
    if trailing_index != -1:
        raw = raw[:trailing_index].strip()
    raw = raw.replace("<|im_end|>", " ").strip()
    return raw


def _strip_repo_setup_wrapper(text: str) -> str:
    raw = str(text or "").replace("\r", "").strip()
    if not raw:
        return ""
    marker = "TASK DESCRIPTION:"
    marker_index = raw.find(marker)
    if marker_index == -1:
        return raw.replace("<|im_end|>", " ").strip()
    raw = raw[marker_index + len(marker) :].strip()
    detail_marker = "DETAILED CONTEXT & HINTS:"
    detail_index = raw.find(detail_marker)
    if detail_index != -1:
        raw = raw[:detail_index].strip()
    raw = raw.replace("<|im_end|>", " ").strip()
    return raw


def _extract_substantive_prompt(messages: list[dict[str, Any]]) -> str:
    fallback = ""
    for message in messages:
        if not isinstance(message, dict) or str(message.get("role") or "").lower() != "user":
            continue
        content = _safe_text(message.get("content"))
        if not content:
            continue
        candidates = _inline_user_segments(content)
        substantive: list[str] = []
        for candidate in candidates:
            if (
                not candidate
                or _looks_like_env_context(candidate)
                or _looks_like_user_instructions(candidate)
                or _looks_like_agents_instructions_wrapper(candidate)
                or _looks_like_local_command_wrapper(candidate)
                or _looks_like_skill_wrapper(candidate)
                or _looks_like_command_output_wrapper(candidate)
            ):
                continue
            stripped = _strip_file_wrapper(candidate)
            stripped = _strip_ide_selection_wrapper(stripped)
            stripped = _strip_system_reminder_wrapper(stripped)
            stripped = _strip_pi_worker_wrapper(stripped)
            stripped = _strip_continued_conversation_wrapper(stripped)
            stripped = _strip_repo_setup_wrapper(stripped)
            if _looks_like_spawned_agent_wrapper(stripped):
                continue
            trimmed = _trim_embedded_transcript_tail(stripped)
            if trimmed:
                substantive.append(trimmed)
        if substantive:
            candidate = substantive[-1]
            if not fallback:
                fallback = candidate
            if not _looks_incomplete_or_junk_prompt(candidate):
                return candidate
    return fallback


def _looks_incomplete_or_junk_prompt(text: str) -> bool:
    normalized = " ".join(str(text or "").replace("<|im_end|>", " ").strip().lower().split())
    if not normalized:
        return True
    if normalized.startswith("[tool result]:") or normalized.startswith("<tool_result"):
        return True
    if normalized.startswith("[redacted_config_dump"):
        return True
    if normalized.startswith("<user_shell_command>") and "</user_shell_command>" in normalized:
        return True
    if re.fullmatch(r"\$[a-z_][a-z0-9_]*", normalized):
        return True
    if normalized.startswith("...[earlier truncated]...") or normalized.startswith("…[earlier truncated]…"):
        return True
    if re.fullmatch(r"/[a-z0-9][a-z0-9_-]*", normalized):
        return True
    if normalized in {"qefqq", "pi", "warmup"}:
        return True
    if len(normalized.split()) == 1 and normalized.isalpha() and len(normalized) <= 5:
        return True
    if normalized in {
        "can you try to understand why this is happening?",
        "can you try to understand why this is happening",
        "what's happening",
        "whats happening",
    }:
        return True
    if normalized.endswith(("i actually want", "can you", "please can you", "i want to")):
        return True
    return False


def _looks_like_coding_prompt(text: str) -> bool:
    normalized = " ".join(str(text or "").strip().lower().split())
    if not normalized:
        return False
    coding_markers = (
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".java",
        ".go",
        ".rs",
        ".cpp",
        ".c",
        ".json",
        ".yaml",
        ".yml",
        ".md",
        "code",
        "repo",
        "project",
        "app",
        "website",
        "landing page",
        "bug",
        "fix",
        "parser",
        "test",
        "build",
        "implement",
        "feature",
        "script",
        "cli",
        "api",
        "frontend",
        "backend",
        "component",
        "function",
        "class",
        "tool",
        "keyboard shortcuts",
        "captions",
    )
    return any(marker in normalized for marker in coding_markers)


def _looks_like_openai_turn_bundle_wrapper_prompt(text: str) -> bool:
    normalized = " ".join(str(text or "").strip().lower().split())
    return normalized in {"claude.md is ready, start", "claude.md ready, start"}


def _looks_like_openai_turn_bundle_wrapper_content(text: str) -> bool:
    normalized = " ".join(str(text or "").strip().lower().split())
    if not normalized:
        return True
    if normalized in {"i", "find", "mkdir", "ls", "cat"}:
        return True
    if normalized.startswith("<is_displaying_contents>"):
        return True
    if normalized.startswith("command:"):
        return True
    if normalized.startswith('"isnewtopic"') or "claude.md ready" in normalized:
        return True
    if normalized.startswith("i acknowledge the context"):
        return True
    if normalized.startswith("i've read the project instructions"):
        return True
    if normalized.startswith("i understand the context"):
        return True
    if normalized.startswith("i understand") or normalized.startswith("i'm ready to help you explore"):
        return True
    if normalized.startswith("i'm claude code") or normalized.startswith("let me check"):
        return True
    return False


_CHATML_BLOCK_RE = re.compile(
    r"(?s)<\|im_start\|>([^\n]+)\n(.*?)(?:(?:<\|im_end\|>\s*)?(?=<\|im_start\|>)|(?:<\|im_end\|>\s*)?$)"
)
_TOOL_USE_BLOCK_RE = re.compile(r"(?is)<tool_use\b[^>]*>(.*?)</tool_use>")


def _chatml_messages(text: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    raw = str(text or "").replace("\r", "").strip()
    if not raw:
        return messages
    for match in _CHATML_BLOCK_RE.finditer(raw):
        role = match.group(1).strip().lower()
        content = match.group(2).strip()
        if not content:
            continue
        if role in {"system", "user", "assistant", "tool"}:
            messages.append({"role": role, "content": content})
    return messages


def _extract_chatml_prompt(text: str) -> str:
    return _extract_substantive_prompt(_chatml_messages(text))


def _normalize_tool_name(name: str) -> str:
    value = str(name or "").strip()
    if not value:
        return ""
    parameter_index = value.lower().find("<parameter=")
    if parameter_index != -1:
        value = value[:parameter_index].strip()
    if "\n" in value:
        first_line = value.splitlines()[0].strip()
        if first_line:
            value = first_line
    return value.strip()


def _chatml_tool_names(content: str) -> list[str]:
    names: list[str] = []
    raw = str(content or "")
    if not raw:
        return names
    for match in _TOOL_USE_BLOCK_RE.finditer(raw):
        block = match.group(1).strip()
        if not block:
            continue
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        name = payload.get("name")
        if isinstance(name, str):
            normalized = _normalize_tool_name(name)
            if normalized:
                names.append(normalized)
    return names


def _looks_like_chatml_policy_helper_row(text: str) -> bool:
    raw = str(text or "").lower()
    return (
        "<policy_spec>" in raw
        and "command prefix" in raw
        and "only return the prefix" in raw
    )


def _embedded_conversation_payload(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_list_payload(text: Any) -> list[dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _extract_embedded_conversation_prompt(text: str) -> str:
    payload = _embedded_conversation_payload(text)
    request = payload.get("request")
    if not isinstance(request, dict):
        return ""
    raw_messages = request.get("messages")
    if not isinstance(raw_messages, list):
        return ""
    messages: list[dict[str, Any]] = []
    for message in raw_messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").lower()
        if role != "user":
            continue
        content = _safe_text(message.get("content"))
        if content:
            messages.append({"role": "user", "content": content})
    return _extract_substantive_prompt(messages)


def _embedded_conversation_model(text: str) -> str:
    payload = _embedded_conversation_payload(text)
    for key in ("request", "response"):
        section = payload.get(key)
        if not isinstance(section, dict):
            continue
        model = section.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()
    return ""


def _looks_like_embedded_meta_helper_row(text: str) -> bool:
    payload = _embedded_conversation_payload(text)
    request = payload.get("request")
    if not isinstance(request, dict):
        return False
    system = request.get("system")
    if not isinstance(system, list):
        return False
    system_text = _safe_text(system).lower()
    if not system_text:
        return False
    meta_markers = (
        "analyze if this message indicates a new conversation topic",
        "extract a 2-3 word title",
        "return exactly five filenames that are frequently modified",
        "return only the filenames' basenames",
    )
    return any(marker in system_text for marker in meta_markers)


def _looks_like_embedded_orientation_prompt(prompt: str) -> bool:
    normalized = " ".join(str(prompt or "").strip().lower().split())
    if not normalized:
        return False
    return normalized in {
        "checkout this direcotry",
        "checkout this directory",
        "check out this directory",
        "check out this directory and use todowrite",
        "deeply",
    }


def _merged_claude_code_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return _json_list_payload(row.get("messages_json"))


def _extract_merged_claude_code_prompt(row: dict[str, Any]) -> str:
    messages = _merged_claude_code_messages(row)
    user_messages: list[dict[str, Any]] = []
    for message in messages:
        if str(message.get("role") or "").lower() != "user":
            continue
        content = _safe_text(message.get("content"))
        if content:
            user_messages.append({"role": "user", "content": content})
    prompt = _extract_substantive_prompt(user_messages)
    if prompt:
        return prompt
    raw_user_prompt = row.get("user_prompt")
    fallback = _safe_text(raw_user_prompt)
    parsed_prompt_items = _json_list_payload(raw_user_prompt)
    if parsed_prompt_items:
        fallback = _safe_text(parsed_prompt_items)
    if fallback:
        prompt = _extract_substantive_prompt([{"role": "user", "content": fallback}])
        if prompt:
            return prompt
        cleaned = _strip_system_reminder_wrapper(_trim_embedded_transcript_tail(fallback))
        if cleaned and not _looks_like_command_output_wrapper(cleaned):
            return cleaned
    return ""


def _looks_like_merged_claude_meta_helper_row(row: dict[str, Any], prompt: str) -> bool:
    system_prompt = _safe_text(row.get("system_prompt")).lower()
    meta_markers = (
        "analyze if this message indicates a new conversation topic",
        "extract a 2-3 word title",
        "return exactly five filenames that are frequently modified",
        "return only the filenames' basenames",
    )
    if any(marker in system_prompt for marker in meta_markers):
        return True
    if _looks_like_chatml_policy_helper_row(prompt):
        return True
    return _looks_like_embedded_orientation_prompt(prompt)


def _merged_claude_code_tool_names(row: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for message in _merged_claude_code_messages(row):
        if str(message.get("role") or "").lower() != "assistant":
            continue
        names.extend(_tool_names_from_assistant_message(message))
    if names:
        return names
    return _chatml_tool_names(str(row.get("assistant_response") or ""))


def _merged_claude_code_message_count(row: dict[str, Any], prompt: str) -> int:
    messages = _merged_claude_code_messages(row)
    if messages:
        return len(messages)
    count = 1 if prompt else 0
    assistant_response = _safe_text(row.get("assistant_response"))
    if assistant_response:
        count += 1
    return count


def _extract_openai_turn_bundle_prompt(turns: list[dict[str, Any]]) -> str:
    session_user_messages: list[dict[str, Any]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        messages = turn.get("messages")
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, dict) or str(message.get("role") or "").lower() != "user":
                continue
            content = _safe_text(message.get("content"))
            if content:
                session_user_messages.append({"role": "user", "content": content})
    prompt = _extract_substantive_prompt(session_user_messages)
    if prompt and not _looks_like_openai_turn_bundle_wrapper_prompt(prompt):
        return prompt
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        content = str(turn.get("content") or "").strip()
        if not content or _looks_like_openai_turn_bundle_wrapper_content(content):
            continue
        return content
    return prompt


def _tool_names_from_assistant_message(message: dict[str, Any]) -> list[str]:
    names: list[str] = []
    top_level_name = message.get("toolName") or message.get("tool_name")
    if isinstance(top_level_name, str):
        normalized = _normalize_tool_name(top_level_name)
        if normalized:
            names.append(normalized)
    tool_calls = message.get("tool_calls") or message.get("toolCalls")
    if isinstance(tool_calls, list):
        for item in tool_calls:
            if not isinstance(item, dict):
                continue
            function = item.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                if isinstance(name, str):
                    normalized = _normalize_tool_name(name)
                    if normalized:
                        names.append(normalized)
            name = item.get("name")
            if isinstance(name, str):
                normalized = _normalize_tool_name(name)
                if normalized:
                    names.append(normalized)
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type") or "").lower()
            if part_type not in {"tool_use", "toolcall", "tool_call", "toolcallresult", "toolcallresult"}:
                continue
            name = part.get("name")
            if isinstance(name, str):
                normalized = _normalize_tool_name(name)
                if normalized:
                    names.append(normalized)
    return names


def _tool_names_from_dataclaw_message(message: dict[str, Any]) -> list[str]:
    names: list[str] = []
    tool_uses = message.get("tool_uses")
    if not isinstance(tool_uses, list):
        return names
    for item in tool_uses:
        if not isinstance(item, dict):
            continue
        name = item.get("tool")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def _claude_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").lower()
        if item_type in {"text", "thinking"}:
            value = item.get("text") if item_type == "text" else item.get("thinking")
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
    return "\n".join(parts).strip()


def _tool_names_from_claude_assistant_row(row: dict[str, Any]) -> list[str]:
    if str(row.get("type") or "").lower() != "assistant":
        return []
    return _tool_names_from_nested_assistant_message_row(row)


def _tool_names_from_nested_assistant_message_row(row: dict[str, Any]) -> list[str]:
    message = row.get("message")
    if not isinstance(message, dict):
        return []
    if str(message.get("role") or "").lower() != "assistant":
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    names: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").lower()
        if item_type not in {"tool_use", "tooluse", "tool_call", "toolcall"}:
            continue
        name = item.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def _codex_rows(dataset_dir: Path, max_rows: int | None) -> Iterable[dict[str, Any]]:
    path = dataset_dir / "sessions.jsonl"
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_rows is not None and count >= max_rows:
                break
            raw = json.loads(line)
            if isinstance(raw, dict):
                yield raw
                count += 1


def _raw_codex_session_files(dataset: str, dataset_dir: Path, max_files: int | None) -> list[Path]:
    session_glob = str(DATASET_SPECS.get(dataset, {}).get("session_glob") or "")
    if not session_glob:
        return []
    files = [path for path in dataset_dir.glob(session_glob) if path.is_file()]
    return sorted(files)[:max_files] if max_files is not None else sorted(files)


def _message_bundle_rows(dataset: str, dataset_dir: Path, max_rows: int | None) -> Iterable[dict[str, Any]]:
    session_glob = str(DATASET_SPECS.get(dataset, {}).get("session_glob") or "")
    if not session_glob:
        return
    path = dataset_dir / session_glob
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_rows is not None and count >= max_rows:
                break
            raw = json.loads(line)
            if isinstance(raw, dict):
                yield raw
                count += 1


def _codex_raw_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw_jsonl = row.get("raw_jsonl")
    if not isinstance(raw_jsonl, str) or not raw_jsonl.strip():
        return []
    events: list[dict[str, Any]] = []
    for line in raw_jsonl.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _codex_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").lower()
        if item_type != "input_text":
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts).strip()


def _codex_messages_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for event in events:
        event_type = str(event.get("type") or "").lower()
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        if event_type == "event_msg" and str(payload.get("type") or "").lower() == "user_message":
            message = payload.get("message")
            if isinstance(message, str) and message.strip():
                messages.append({"role": "user", "content": message.strip()})
            continue
        if event_type != "response_item" or str(payload.get("type") or "").lower() != "message":
            continue
        role = str(payload.get("role") or "").lower()
        if role not in {"user", "assistant"}:
            continue
        content = _codex_message_text(payload.get("content"))
        if content:
            messages.append({"role": role, "content": content})
    return messages


def _tool_names_from_codex_event(event: dict[str, Any]) -> list[str]:
    event_type = str(event.get("type") or "").lower()
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return []
    if event_type != "response_item":
        return []
    payload_type = str(payload.get("type") or "").lower()
    if payload_type == "function_call":
        name = payload.get("name")
        if isinstance(name, str) and name.strip():
            return [name.strip()]
        return []
    if payload_type == "web_search_call":
        return ["web_search"]
    return []


def _raw_codex_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                events.append(row)
    return events


def _share_codex_manifest_model_counts(dataset_dir: Path) -> Counter[str]:
    path = dataset_dir / "export_manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return Counter()
    stats = payload.get("dataset_statistics") if isinstance(payload, dict) else None
    rows = stats.get("model_data_volume") if isinstance(stats, dict) else None
    if not isinstance(rows, list):
        return Counter()
    counts: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        sessions = row.get("sessions")
        if not isinstance(name, str) or not name.strip():
            continue
        try:
            counts[name.strip()] += int(sessions or 0)
        except (TypeError, ValueError):
            continue
    return counts


def _share_codex_rows(dataset_dir: Path, max_rows: int | None) -> Iterable[dict[str, Any]]:
    path = dataset_dir / "train.jsonl"
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_rows is not None and count >= max_rows:
                break
            raw = json.loads(line)
            if isinstance(raw, dict):
                yield raw
                count += 1


def _openai_turn_bundle_rows(dataset: str, dataset_dir: Path, max_rows: int | None) -> Iterable[dict[str, Any]]:
    session_glob = str(DATASET_SPECS.get(dataset, {}).get("session_glob") or "")
    if not session_glob:
        return
    path = dataset_dir / session_glob
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    sessions = payload.get("sessions") if isinstance(payload, dict) else None
    if not isinstance(sessions, list):
        return
    count = 0
    for row in sessions:
        if max_rows is not None and count >= max_rows:
            break
        if isinstance(row, dict):
            yield row
            count += 1


def _dataclaw_session_files(dataset_dir: Path, max_files: int | None) -> list[Path]:
    paths = sorted(dataset_dir.rglob("*.jsonl"))
    if max_files is not None:
        return paths[:max_files]
    return paths


def _dataclaw_session_rows(dataset_dir: Path, max_files: int | None) -> Iterable[dict[str, Any]]:
    for path in _dataclaw_session_files(dataset_dir, max_files):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = json.loads(line)
                if isinstance(raw, dict):
                    yield raw


def _pi_mono_session_files(dataset_dir: Path, max_files: int | None) -> list[Path]:
    paths = sorted(
        path
        for path in dataset_dir.glob("*.jsonl")
        if path.name != "manifest.jsonl"
    )
    if max_files is not None:
        return paths[:max_files]
    return paths


def _pi_session_files(dataset: str, dataset_dir: Path, max_files: int | None) -> list[Path]:
    session_glob = str(DATASET_SPECS.get(dataset, {}).get("session_glob") or "")
    if session_glob:
        paths = sorted(
            path
            for path in dataset_dir.glob(session_glob)
            if path.is_file()
        )
    else:
        paths = _pi_mono_session_files(dataset_dir, None)
    if max_files is not None:
        return paths[:max_files]
    return paths


def _claude_code_session_files(dataset: str, dataset_dir: Path, max_files: int | None) -> list[Path]:
    session_glob = str(DATASET_SPECS.get(dataset, {}).get("session_glob") or "")
    excluded_substrings = tuple(str(item) for item in (DATASET_SPECS.get(dataset, {}).get("exclude_filename_substrings") or []))
    if session_glob:
        paths = sorted(
            path
            for path in dataset_dir.rglob("*.jsonl")
            if path.relative_to(dataset_dir).match(session_glob)
            and not any(part and part in path.name for part in excluded_substrings)
        )
    else:
        paths = sorted(
            path
            for path in dataset_dir.rglob("*.jsonl")
            if not any(part and part in path.name for part in excluded_substrings)
        )
    if max_files is not None:
        return paths[:max_files]
    return paths


def _agent_sessions_list_files(dataset_dir: Path, max_files: int | None) -> list[Path]:
    patterns = (
        "sessions/claude/*.jsonl",
        "sessions/codex/*.jsonl",
        "sessions/pi/*.jsonl",
        "droid/*.jsonl",
    )
    seen: set[Path] = set()
    paths: list[Path] = []
    for pattern in patterns:
        for path in sorted(dataset_dir.glob(pattern)):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            paths.append(path)
    if max_files is not None:
        return paths[:max_files]
    return paths


def _parquet_files(dataset: str, dataset_dir: Path, max_files: int | None) -> list[Path]:
    session_glob = str(DATASET_SPECS.get(dataset, {}).get("session_glob") or "*.parquet")
    paths = sorted(path for path in dataset_dir.glob(session_glob) if path.is_file())
    if max_files is not None:
        return paths[:max_files]
    return paths


def _iter_parquet_rows(paths: Iterable[Path], *, max_rows: int | None = None) -> Iterable[dict[str, Any]]:
    if pq is None:
        raise RuntimeError("web-discovered parquet dataset analysis requires optional dependency pyarrow")
    emitted = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=128):
            for row in batch.to_pylist():
                if not isinstance(row, dict):
                    continue
                yield row
                emitted += 1
                if max_rows is not None and emitted >= max_rows:
                    return


def _pi_mono_file_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            if isinstance(raw, dict):
                rows.append(raw)
    return rows


def _native_codex_file_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            if isinstance(raw, dict):
                events.append(raw)
    return events


def _summarize_native_codex_file(path: Path) -> dict[str, Any] | None:
    events = _native_codex_file_events(path)
    if not events:
        return None
    messages = _codex_messages_from_events(events)
    prompt = _extract_substantive_prompt(messages)
    if _looks_incomplete_or_junk_prompt(prompt):
        return None
    tool_names: list[str] = []
    for event in events:
        tool_names.extend(_tool_names_from_codex_event(event))
    return {
        "prompt": prompt,
        "message_count": len(messages),
        "tool_names": tool_names,
        "models": [],
    }


def _summarize_pi_file(path: Path) -> dict[str, Any] | None:
    rows = _pi_mono_file_rows(path)
    session_messages: list[dict[str, Any]] = []
    tool_names: list[str] = []
    models: list[str] = []
    for row in rows:
        row_type = str(row.get("type") or "").lower()
        if row_type == "model_change":
            model = row.get("modelId")
            if isinstance(model, str) and model.strip():
                models.append(model.strip())
        if row_type != "message":
            continue
        message = row.get("message")
        if not isinstance(message, dict):
            continue
        session_messages.append(message)
        if str(message.get("role") or "").lower() != "assistant":
            continue
        for part in message.get("content") or []:
            if not isinstance(part, dict):
                continue
            if str(part.get("type") or "").lower() != "toolcall":
                continue
            name = part.get("name")
            if isinstance(name, str) and name.strip():
                tool_names.append(name.strip())
    prompt = _extract_substantive_prompt(session_messages)
    if not tool_names and _looks_incomplete_or_junk_prompt(prompt):
        return None
    return {
        "prompt": prompt,
        "message_count": len(session_messages),
        "tool_names": tool_names,
        "models": models,
    }


def _summarize_claude_file(path: Path) -> dict[str, Any] | None:
    session_user_messages: list[dict[str, Any]] = []
    tool_names: list[str] = []
    models: list[str] = []
    message_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            row_type = str(row.get("type") or "").lower()
            if row_type in {"user", "assistant", "system"}:
                message_count += 1
            if row_type == "assistant":
                message = row.get("message")
                if isinstance(message, dict):
                    model = message.get("model")
                    if isinstance(model, str) and model.strip():
                        models.append(model.strip())
                tool_names.extend(_tool_names_from_claude_assistant_row(row))
            if row_type != "user":
                continue
            message = row.get("message")
            if not isinstance(message, dict):
                continue
            content = _claude_message_text(message.get("content"))
            if content:
                session_user_messages.append({"role": "user", "content": content})
    prompt = _extract_substantive_prompt(session_user_messages)
    if _looks_incomplete_or_junk_prompt(prompt):
        return None
    return {
        "prompt": prompt,
        "message_count": message_count,
        "tool_names": tool_names,
        "models": models,
    }


def _summarize_nested_message_file(path: Path) -> dict[str, Any] | None:
    session_user_messages: list[dict[str, Any]] = []
    tool_names: list[str] = []
    models: list[str] = []
    message_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            row_type = str(row.get("type") or "").lower()
            if row_type == "session":
                agent = row.get("agent")
                if isinstance(agent, str) and agent.strip():
                    models.append(agent.strip())
                continue
            if row_type != "message":
                continue
            message = row.get("message")
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").lower()
            if role in {"user", "assistant", "system", "tool"}:
                message_count += 1
            if role == "assistant":
                tool_names.extend(_tool_names_from_nested_assistant_message_row(row))
            if role != "user":
                continue
            content = _claude_message_text(message.get("content"))
            if content:
                session_user_messages.append({"role": "user", "content": content})
    prompt = _extract_substantive_prompt(session_user_messages)
    if _looks_incomplete_or_junk_prompt(prompt):
        return None
    if not tool_names and not _looks_like_coding_prompt(prompt):
        return None
    return {
        "prompt": prompt,
        "message_count": message_count,
        "tool_names": tool_names,
        "models": models,
    }


def _claudeset_session_rows(dataset_dir: Path, max_rows: int | None) -> Iterable[dict[str, Any]]:
    count = 0
    for path in _claude_code_session_files("claudeset-community", dataset_dir, None):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if max_rows is not None and count >= max_rows:
                    return
                raw = json.loads(line)
                if isinstance(raw, dict):
                    yield raw
                    count += 1


def _claudeset_tool_names(tool_calls: Any) -> list[str]:
    if not isinstance(tool_calls, list):
        return []
    names: list[str] = []
    for item in tool_calls:
        if not isinstance(item, dict):
            continue
        name = item.get("tool") or item.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def analyze_share_codex(dataset_dir: Path, *, max_rows: int | None, examples_per_family: int) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    message_total = 0
    tool_call_total = 0
    tool_output_total = 0
    rows = 0

    for row in _share_codex_rows(dataset_dir, max_rows):
        rows += 1
        messages = [item for item in row.get("messages") or [] if isinstance(item, dict)]
        prompt = _extract_substantive_prompt(messages)
        if prompt:
            family = task_review.classify_task_family(task="", prompt_preview=prompt)
            family_counts[family] += 1
            examples.setdefault(family, [])
            preview = _truncate(prompt)
            if preview not in examples[family] and len(examples[family]) < examples_per_family:
                examples[family].append(preview)
        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            model = metadata.get("model_name") or metadata.get("model")
            if isinstance(model, str) and model.strip():
                model_counts[model.strip()] += 1
            tool_call_total += int(metadata.get("tool_call_count") or 0)
            tool_output_total += int(metadata.get("tool_output_count") or 0)
        message_total += len(messages)
        for message in messages:
            for name in _tool_names_from_assistant_message(message):
                tool_counts[name] += 1

    if not model_counts:
        model_counts.update(_share_codex_manifest_model_counts(dataset_dir))

    return {
        "dataset": "share-codex",
        "repo_id": DATASET_SPECS["share-codex"]["repo_id"],
        "rows": rows,
        "task_kind_counts": family_counts.most_common(),
        "examples": examples,
        "avg_message_count": round(message_total / rows, 2) if rows else 0.0,
        "avg_tool_calls": round(tool_call_total / rows, 2) if rows else 0.0,
        "avg_tool_outputs": round(tool_output_total / rows, 2) if rows else 0.0,
        "tool_counts": tool_counts.most_common(12),
        "model_counts": model_counts.most_common(12),
        "insights": [
            "Codex-heavy real-user sessions skew toward frontend, feature, and bugfix work rather than benchmark-only repo repair."
            if rows
            else "No local rows available.",
            "Shell-heavy tool mix suggests repeated command execution is still a primary cost center before edits converge."
            if tool_counts
            else "Tool call mix unavailable.",
        ],
    }


def analyze_codex_sessions(
    dataset_dir: Path,
    *,
    max_rows: int | None,
    examples_per_family: int,
) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    message_total = 0
    tool_call_total = 0
    rows = 0

    for row in _codex_rows(dataset_dir, max_rows):
        rows += 1
        events = _codex_raw_events(row)
        messages = _codex_messages_from_events(events)
        prompt = _extract_substantive_prompt(messages)
        if prompt:
            family = task_review.classify_task_family(task="", prompt_preview=prompt)
            family_counts[family] += 1
            examples.setdefault(family, [])
            preview = _truncate(prompt)
            if preview not in examples[family] and len(examples[family]) < examples_per_family:
                examples[family].append(preview)
        message_total += len(messages)
        for event in events:
            for name in _tool_names_from_codex_event(event):
                tool_counts[name] += 1
                tool_call_total += 1

    return {
        "dataset": "codex-sessions",
        "repo_id": DATASET_SPECS["codex-sessions"]["repo_id"],
        "rows": rows,
        "task_kind_counts": family_counts.most_common(),
        "examples": examples,
        "avg_message_count": round(message_total / rows, 2) if rows else 0.0,
        "avg_tool_calls": round(tool_call_total / rows, 2) if rows else 0.0,
        "tool_counts": tool_counts.most_common(12),
        "model_counts": [],
        "insights": [
            "Embedded Codex raw sessions preserve commentary, user messages, and tool-call events in one row, so they expose overhead from search-heavy research turns cleanly."
            if rows
            else "No local rows available.",
            "Web search and shell call counts show where research and repo-orientation work are consuming turns before the agent reaches code changes."
            if tool_counts
            else "Tool call mix unavailable.",
        ],
    }


def analyze_message_bundle_dataset(
    dataset: str,
    dataset_dir: Path,
    *,
    max_rows: int | None,
    examples_per_family: int,
) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    message_total = 0
    tool_call_total = 0
    rows = 0

    for row in _message_bundle_rows(dataset, dataset_dir, max_rows):
        messages = [item for item in row.get("messages") or [] if isinstance(item, dict)]
        prompt = _extract_substantive_prompt(messages)
        if _looks_incomplete_or_junk_prompt(prompt):
            continue
        rows += 1
        source = row.get("source")
        if isinstance(source, str) and source.strip():
            source_counts[source.strip()] += 1
        message_total += len(messages)
        for message in messages:
            for name in _tool_names_from_assistant_message(message):
                tool_counts[name] += 1
                tool_call_total += 1
        family = task_review.classify_task_family(task="", prompt_preview=prompt)
        family_counts[family] += 1
        examples.setdefault(family, [])
        preview = _truncate(prompt)
        if preview not in examples[family] and len(examples[family]) < examples_per_family:
            examples[family].append(preview)

    return {
        "dataset": dataset,
        "repo_id": DATASET_SPECS[dataset]["repo_id"],
        "rows": rows,
        "task_kind_counts": family_counts.most_common(),
        "examples": examples,
        "avg_message_count": round(message_total / rows, 2) if rows else 0.0,
        "avg_tool_calls": round(tool_call_total / rows, 2) if rows else 0.0,
        "tool_counts": tool_counts.most_common(12),
        "model_counts": [],
        "source_counts": source_counts.most_common(12),
        "insights": [
            "Message-bundle Codex exports are useful because they preserve real prompt text and assistant tool calls without the larger event wrapper used by some raw session dumps."
            if rows
            else "No local rows available.",
            "These sessions still show prompt-wrapper noise, so extracting the substantive user request before classification is necessary for trustworthy task-family counts."
            if rows
            else "Prompt extraction evidence unavailable.",
        ],
    }


def analyze_raw_codex_session_dataset(
    dataset: str,
    dataset_dir: Path,
    *,
    max_files: int | None,
    examples_per_family: int,
) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    message_total = 0
    tool_call_total = 0
    files = 0

    for path in _raw_codex_session_files(dataset, dataset_dir, max_files):
        events = _raw_codex_events(path)
        messages = _codex_messages_from_events(events)
        prompt = _extract_substantive_prompt(messages)
        if _looks_incomplete_or_junk_prompt(prompt):
            continue
        files += 1
        message_total += len(messages)
        for event in events:
            for name in _tool_names_from_codex_event(event):
                tool_counts[name] += 1
                tool_call_total += 1
        if prompt:
            family = task_review.classify_task_family(task="", prompt_preview=prompt)
            family_counts[family] += 1
            examples.setdefault(family, [])
            preview = _truncate(prompt)
            if preview not in examples[family] and len(examples[family]) < examples_per_family:
                examples[family].append(preview)

    return {
        "dataset": dataset,
        "repo_id": DATASET_SPECS[dataset]["repo_id"],
        "rows": files,
        "task_kind_counts": family_counts.most_common(),
        "examples": examples,
        "avg_message_count": round(message_total / files, 2) if files else 0.0,
        "avg_tool_calls": round(tool_call_total / files, 2) if files else 0.0,
        "tool_counts": tool_counts.most_common(12),
        "model_counts": [],
        "insights": [
            "Raw Codex Desktop session exports preserve the actual developer and user turn structure, so they expose first-turn instruction bloat and tool-loop cost directly."
            if files
            else "No local files available.",
            "Executed function_call items are the reliable tool-usage signal in these raw sessions; developer wrappers should be filtered before task classification."
            if tool_call_total
            else "Tool-call evidence unavailable.",
        ],
    }


def analyze_dataclaw_dataset(
    dataset: str,
    dataset_dir: Path,
    *,
    max_files: int | None,
    examples_per_family: int,
) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    message_total = 0
    tool_use_total = 0
    input_token_total = 0
    output_token_total = 0
    rows = 0

    for row in _dataclaw_session_rows(dataset_dir, max_files):
        rows += 1
        messages = [item for item in row.get("messages") or [] if isinstance(item, dict)]
        prompt = _extract_substantive_prompt(messages)
        if prompt:
            family = task_review.classify_task_family(task="", prompt_preview=prompt)
            family_counts[family] += 1
            examples.setdefault(family, [])
            preview = _truncate(prompt)
            if preview not in examples[family] and len(examples[family]) < examples_per_family:
                examples[family].append(preview)
        model = row.get("model")
        if isinstance(model, str) and model.strip():
            model_counts[model.strip()] += 1
        stats = row.get("stats")
        if isinstance(stats, dict):
            tool_use_total += int(stats.get("tool_uses") or 0)
            input_token_total += int(stats.get("input_tokens") or 0)
            output_token_total += int(stats.get("output_tokens") or 0)
        message_total += len(messages)
        for message in messages:
            for name in _tool_names_from_dataclaw_message(message):
                tool_counts[name] += 1

    avg_tool_calls = round(tool_use_total / rows, 2) if rows else 0.0
    avg_input_tokens = round(input_token_total / rows, 2) if rows else 0.0
    avg_output_tokens = round(output_token_total / rows, 2) if rows else 0.0
    avg_input_tokens_per_tool = round(input_token_total / tool_use_total, 2) if tool_use_total else 0.0

    return {
        "dataset": dataset,
        "repo_id": DATASET_SPECS[dataset]["repo_id"],
        "rows": rows,
        "task_kind_counts": family_counts.most_common(),
        "examples": examples,
        "avg_message_count": round(message_total / rows, 2) if rows else 0.0,
        "avg_tool_calls": avg_tool_calls,
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
        "avg_input_tokens_per_tool": avg_input_tokens_per_tool,
        "tool_counts": tool_counts.most_common(12),
        "model_counts": model_counts.most_common(12),
        "insights": [
            "These real-user DataClaw exports expose direct token totals, so they are useful for measuring context blow-up per tool-heavy session."
            if rows
            else "No local rows available.",
            "Large input-token-per-tool ratios indicate that repeated long-context turns are still a bigger cost center than the raw number of tool calls."
            if tool_use_total
            else "Token-efficiency evidence unavailable.",
        ],
    }


def analyze_pi_dataset(
    dataset: str,
    dataset_dir: Path,
    *,
    max_files: int | None,
    examples_per_family: int,
) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    message_total = 0
    tool_call_total = 0
    files = 0

    for path in _pi_session_files(dataset, dataset_dir, max_files):
        rows = _pi_mono_file_rows(path)
        session_messages: list[dict[str, Any]] = []
        session_tool_calls = 0
        for row in rows:
            row_type = str(row.get("type") or "").lower()
            if row_type == "model_change":
                model = row.get("modelId")
                if isinstance(model, str) and model.strip():
                    model_counts[model.strip()] += 1
            if row_type != "message":
                continue
            message = row.get("message")
            if not isinstance(message, dict):
                continue
            session_messages.append(message)
            if str(message.get("role") or "").lower() != "assistant":
                continue
            for part in message.get("content") or []:
                if not isinstance(part, dict):
                    continue
                if str(part.get("type") or "").lower() != "toolcall":
                    continue
                name = part.get("name")
                if isinstance(name, str) and name.strip():
                    session_tool_calls += 1
                    tool_counts[name.strip()] += 1
        prompt = _extract_substantive_prompt(session_messages)
        if session_tool_calls == 0 and _looks_incomplete_or_junk_prompt(prompt):
            continue
        files += 1
        tool_call_total += session_tool_calls
        message_total += len(session_messages)
        if prompt:
            family = task_review.classify_task_family(task="", prompt_preview=prompt)
            family_counts[family] += 1
            examples.setdefault(family, [])
            preview = _truncate(prompt)
            if preview not in examples[family] and len(examples[family]) < examples_per_family:
                examples[family].append(preview)

    return {
        "dataset": dataset,
        "repo_id": DATASET_SPECS[dataset]["repo_id"],
        "rows": files,
        "task_kind_counts": family_counts.most_common(),
        "examples": examples,
        "avg_message_count": round(message_total / files, 2) if files else 0.0,
        "avg_tool_calls": round(tool_call_total / files, 2) if files else 0.0,
        "tool_counts": tool_counts.most_common(12),
        "model_counts": model_counts.most_common(12),
        "insights": [
            "Pi sessions show the same early broad-read habit seen in other coding-agent traces, but in a real monorepo workflow."
            if files
            else "No local files available.",
            "Initial turn discipline matters because some sessions spend the first expensive model turn on generic orientation before task-specific work begins."
            if files
            else "Turn-level evidence unavailable.",
        ],
    }


def analyze_claude_code_dataset(
    dataset: str,
    dataset_dir: Path,
    *,
    max_files: int | None,
    examples_per_family: int,
) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    message_total = 0
    tool_call_total = 0
    files = 0

    for path in _claude_code_session_files(dataset, dataset_dir, max_files):
        session_user_messages: list[dict[str, Any]] = []
        session_messages = 0
        session_tool_calls = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                row_type = str(row.get("type") or "").lower()
                if row_type in {"user", "assistant", "system"}:
                    session_messages += 1
                if row_type == "assistant":
                    message = row.get("message")
                    if isinstance(message, dict):
                        model = message.get("model")
                        if isinstance(model, str) and model.strip():
                            model_counts[model.strip()] += 1
                    for name in _tool_names_from_claude_assistant_row(row):
                        tool_counts[name] += 1
                        session_tool_calls += 1
                if row_type != "user":
                    continue
                message = row.get("message")
                if not isinstance(message, dict):
                    continue
                content = _claude_message_text(message.get("content"))
                if content:
                    session_user_messages.append({"role": "user", "content": content})
        prompt = _extract_substantive_prompt(session_user_messages)
        if _looks_incomplete_or_junk_prompt(prompt):
            continue
        files += 1
        message_total += session_messages
        tool_call_total += session_tool_calls
        if prompt:
            family = task_review.classify_task_family(task="", prompt_preview=prompt)
            family_counts[family] += 1
            examples.setdefault(family, [])
            preview = _truncate(prompt)
            if preview not in examples[family] and len(examples[family]) < examples_per_family:
                examples[family].append(preview)

    return {
        "dataset": dataset,
        "repo_id": DATASET_SPECS[dataset]["repo_id"],
        "rows": files,
        "task_kind_counts": family_counts.most_common(),
        "examples": examples,
        "avg_message_count": round(message_total / files, 2) if files else 0.0,
        "avg_tool_calls": round(tool_call_total / files, 2) if files else 0.0,
        "tool_counts": tool_counts.most_common(12),
        "model_counts": model_counts.most_common(12),
        "insights": [
            "Native Claude Code traces expose the same control chatter and broad first-turn orientation seen in Pi exports, but with tool names preserved directly from the assistant trace."
            if files
            else "No local files available.",
            "These traces are useful for spotting when large shell or read loops are replayed across many expensive assistant turns before validation."
            if files
            else "Turn-level evidence unavailable.",
        ],
    }


def analyze_nested_message_session_dataset(
    dataset: str,
    dataset_dir: Path,
    *,
    max_files: int | None,
    examples_per_family: int,
) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    message_total = 0
    tool_call_total = 0
    files = 0
    inspected_files = 0

    for path in _claude_code_session_files(dataset, dataset_dir, max_files):
        inspected_files += 1
        session_user_messages: list[dict[str, Any]] = []
        session_messages = 0
        session_tool_calls = 0
        session_agent = ""
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                row_type = str(row.get("type") or "").lower()
                if row_type == "session":
                    agent = row.get("agent")
                    if isinstance(agent, str) and agent.strip():
                        session_agent = agent.strip()
                    continue
                if row_type != "message":
                    continue
                message = row.get("message")
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role") or "").lower()
                if role in {"user", "assistant", "system", "tool"}:
                    session_messages += 1
                if role == "assistant":
                    for name in _tool_names_from_nested_assistant_message_row(row):
                        tool_counts[name] += 1
                        session_tool_calls += 1
                if role != "user":
                    continue
                content = _claude_message_text(message.get("content"))
                if content:
                    session_user_messages.append({"role": "user", "content": content})
        prompt = _extract_substantive_prompt(session_user_messages)
        if _looks_incomplete_or_junk_prompt(prompt):
            continue
        if session_tool_calls == 0 and not _looks_like_coding_prompt(prompt):
            continue
        files += 1
        message_total += session_messages
        tool_call_total += session_tool_calls
        if session_agent:
            model_counts[session_agent] += 1
        if prompt:
            family = task_review.classify_task_family(task="", prompt_preview=prompt)
            family_counts[family] += 1
            examples.setdefault(family, [])
            preview = _truncate(prompt)
            if preview not in examples[family] and len(examples[family]) < examples_per_family:
                examples[family].append(preview)

    return {
        "dataset": dataset,
        "repo_id": DATASET_SPECS[dataset]["repo_id"],
        "rows": files,
        "task_kind_counts": family_counts.most_common(),
        "examples": examples,
        "avg_message_count": round(message_total / files, 2) if files else 0.0,
        "avg_tool_calls": round(tool_call_total / files, 2) if files else 0.0,
        "tool_counts": tool_counts.most_common(12),
        "model_counts": model_counts.most_common(12),
        "insights": [
            "Cursor-style nested message traces expose the same real-user feature and bugfix requests, but preserve tool loops as plain JSONL."
            if files
            else "Local trace files were present but filtered out as non-coding or non-agent sessions."
            if inspected_files
            else "No local files available.",
            "Main-session traces should be reviewed separately from subagent traces because they reflect user-facing task flow more directly."
            if files
            else "The inspected local files did not contain coding-task evidence suitable for this report."
            if inspected_files
            else "Turn-level evidence unavailable.",
        ],
    }


def analyze_trace_commons_dataset(
    dataset_dir: Path,
    *,
    max_files: int | None,
    examples_per_family: int,
) -> dict[str, Any]:
    return analyze_claude_code_dataset(
        "trace-commons-agent-traces",
        dataset_dir,
        max_files=max_files,
        examples_per_family=examples_per_family,
    )


def analyze_pi_mono(dataset_dir: Path, *, max_files: int | None, examples_per_family: int) -> dict[str, Any]:
    return analyze_pi_dataset("pi-mono", dataset_dir, max_files=max_files, examples_per_family=examples_per_family)


def analyze_mixed_agent_session_dataset(
    dataset: str,
    dataset_dir: Path,
    *,
    max_files: int | None,
    examples_per_family: int,
) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    message_total = 0
    tool_call_total = 0
    files = 0

    for path in _agent_sessions_list_files(dataset_dir, max_files):
        relative = path.relative_to(dataset_dir).as_posix()
        summary: dict[str, Any] | None = None
        if relative.startswith("sessions/codex/"):
            summary = _summarize_native_codex_file(path)
        elif relative.startswith("sessions/pi/"):
            summary = _summarize_pi_file(path)
        elif relative.startswith("sessions/claude/"):
            summary = _summarize_claude_file(path)
        elif relative.startswith("droid/"):
            summary = _summarize_nested_message_file(path)
        if not summary:
            continue
        prompt = str(summary.get("prompt") or "")
        tool_names = [name for name in summary.get("tool_names", []) if isinstance(name, str) and name.strip()]
        if not tool_names and not _looks_like_coding_prompt(prompt):
            continue
        files += 1
        message_total += int(summary["message_count"])
        tool_call_total += len(tool_names)
        for name in tool_names:
            tool_counts[name] += 1
        for model in summary.get("models", []):
            if isinstance(model, str) and model.strip():
                model_counts[model.strip()] += 1
        if prompt:
            family = task_review.classify_task_family(task="", prompt_preview=prompt)
            family_counts[family] += 1
            examples.setdefault(family, [])
            preview = _truncate(prompt)
            if preview not in examples[family] and len(examples[family]) < examples_per_family:
                examples[family].append(preview)

    return {
        "dataset": dataset,
        "repo_id": DATASET_SPECS[dataset]["repo_id"],
        "rows": files,
        "task_kind_counts": family_counts.most_common(),
        "examples": examples,
        "avg_message_count": round(message_total / files, 2) if files else 0.0,
        "avg_tool_calls": round(tool_call_total / files, 2) if files else 0.0,
        "tool_counts": tool_counts.most_common(12),
        "model_counts": model_counts.most_common(12),
        "insights": [
            "Mixed public trace dumps are useful because they hold multiple coding-agent harnesses in one dataset, which makes control-turn and shell-loop waste easier to compare."
            if files
            else "No local files available.",
            "The same user-level task families show up across Codex, Claude Code, Pi, and Droid exports, which suggests many efficiency fixes should live in controller policy rather than tool-specific prompt text."
            if files
            else "Turn-level evidence unavailable.",
        ],
    }


def analyze_claudeset_community_dataset(
    dataset_dir: Path,
    *,
    max_rows: int | None,
    examples_per_family: int,
) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    message_total = 0
    tool_call_total = 0
    rows = 0

    for row in _claudeset_session_rows(dataset_dir, max_rows):
        turns = row.get("turns")
        if not isinstance(turns, list):
            continue
        session_user_messages: list[dict[str, Any]] = []
        session_messages = 0
        session_tool_calls = 0
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            if str(turn.get("type") or "").lower() != "exchange":
                continue
            user = turn.get("user")
            if isinstance(user, str) and user.strip():
                session_user_messages.append({"role": "user", "content": user.strip()})
                session_messages += 1
            assistant = turn.get("assistant")
            if not isinstance(assistant, dict):
                continue
            session_messages += 1
            for name in _claudeset_tool_names(assistant.get("tool_calls")):
                tool_counts[name] += 1
                session_tool_calls += 1
        prompt = _extract_substantive_prompt(session_user_messages)
        if _looks_incomplete_or_junk_prompt(prompt):
            continue
        rows += 1
        message_total += session_messages
        tool_call_total += session_tool_calls
        model = row.get("model")
        if isinstance(model, str) and model.strip():
            model_counts[model.strip()] += 1
        family = task_review.classify_task_family(task="", prompt_preview=prompt)
        family_counts[family] += 1
        examples.setdefault(family, [])
        preview = _truncate(prompt)
        if preview not in examples[family] and len(examples[family]) < examples_per_family:
            examples[family].append(preview)

    return {
        "dataset": "claudeset-community",
        "repo_id": DATASET_SPECS["claudeset-community"]["repo_id"],
        "rows": rows,
        "task_kind_counts": family_counts.most_common(),
        "examples": examples,
        "avg_message_count": round(message_total / rows, 2) if rows else 0.0,
        "avg_tool_calls": round(tool_call_total / rows, 2) if rows else 0.0,
        "tool_counts": tool_counts.most_common(12),
        "model_counts": model_counts.most_common(12),
        "insights": [
            "Claudeset community rows preserve full exchange turns with tool outputs, which makes them useful for comparing task shape and tool churn across many contributors."
            if rows
            else "No local files available.",
            "Compact summaries appear alongside exchanges, so analysis should ignore compacts and classify from the actual user turns."
            if rows
            else "Turn-level evidence unavailable.",
        ],
    }


def analyze_openai_turn_bundle_dataset(
    dataset: str,
    dataset_dir: Path,
    *,
    max_rows: int | None,
    examples_per_family: int,
) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    message_total = 0
    tool_call_total = 0
    input_token_total = 0
    output_token_total = 0
    rows = 0

    for row in _openai_turn_bundle_rows(dataset, dataset_dir, max_rows):
        turns = row.get("turns")
        if not isinstance(turns, list):
            continue
        session_models: set[str] = set()
        session_messages = 0
        session_tool_calls = 0
        session_input_tokens = 0
        session_output_tokens = 0
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            session_input_tokens += int(turn.get("input_tokens") or 0)
            session_output_tokens += int(turn.get("output_tokens") or 0)
            model = turn.get("model")
            if isinstance(model, str) and model.strip():
                session_models.add(model.strip())
            session_messages += 1
            messages = turn.get("messages")
            if not isinstance(messages, list):
                continue
            last_assistant_message: dict[str, Any] | None = None
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role") or "").lower()
                if role == "assistant":
                    last_assistant_message = message
            if last_assistant_message is not None:
                for name in _tool_names_from_assistant_message(last_assistant_message):
                    tool_counts[name] += 1
                    session_tool_calls += 1
        prompt = _extract_openai_turn_bundle_prompt(turns)
        if _looks_incomplete_or_junk_prompt(prompt):
            continue
        rows += 1
        message_total += session_messages
        tool_call_total += session_tool_calls
        input_token_total += session_input_tokens
        output_token_total += session_output_tokens
        for model in session_models:
            model_counts[model] += 1
        family = task_review.classify_task_family(task="", prompt_preview=prompt)
        family_counts[family] += 1
        examples.setdefault(family, [])
        preview = _truncate(prompt)
        if preview not in examples[family] and len(examples[family]) < examples_per_family:
            examples[family].append(preview)

    avg_tool_calls = round(tool_call_total / rows, 2) if rows else 0.0
    avg_input_tokens = round(input_token_total / rows, 2) if rows else 0.0
    avg_output_tokens = round(output_token_total / rows, 2) if rows else 0.0
    avg_input_tokens_per_tool = round(input_token_total / tool_call_total, 2) if tool_call_total else 0.0

    return {
        "dataset": dataset,
        "repo_id": DATASET_SPECS[dataset]["repo_id"],
        "rows": rows,
        "task_kind_counts": family_counts.most_common(),
        "examples": examples,
        "avg_message_count": round(message_total / rows, 2) if rows else 0.0,
        "avg_tool_calls": avg_tool_calls,
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
        "avg_input_tokens_per_tool": avg_input_tokens_per_tool,
        "tool_counts": tool_counts.most_common(12),
        "model_counts": model_counts.most_common(12),
        "insights": [
            "Bundled OpenAI-format session exports preserve per-turn token totals and assistant tool calls, so they are useful for measuring token cost across long real-user coding sessions."
            if rows
            else "No local rows available.",
            "High token totals combined with repeated exploration tool calls suggest the main waste source is still long-context replay around orientation and validation loops."
            if tool_call_total
            else "Token-efficiency evidence unavailable.",
        ],
    }


def analyze_chatml_transcript_parquet_dataset(
    dataset: str,
    dataset_dir: Path,
    *,
    max_rows: int | None,
    examples_per_family: int,
) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    message_total = 0
    tool_call_total = 0
    rows = 0

    for row in _iter_parquet_rows(_parquet_files(dataset, dataset_dir, None), max_rows=None):
        text = str(row.get("text") or "")
        if _looks_like_chatml_policy_helper_row(text):
            continue
        prompt = _extract_chatml_prompt(text)
        if _looks_incomplete_or_junk_prompt(prompt):
            continue
        if max_rows is not None and rows >= max_rows:
            break
        rows += 1
        source = row.get("source")
        if isinstance(source, str) and source.strip():
            model_counts[source.strip()] += 1
        messages = _chatml_messages(text)
        message_total += len(messages)
        for message in messages:
            if str(message.get("role") or "").lower() != "assistant":
                continue
            for name in _chatml_tool_names(str(message.get("content") or "")):
                tool_counts[name] += 1
                tool_call_total += 1
        family = task_review.classify_task_family(task="", prompt_preview=prompt)
        family_counts[family] += 1
        examples.setdefault(family, [])
        preview = _truncate(prompt)
        if preview not in examples[family] and len(examples[family]) < examples_per_family:
            examples[family].append(preview)

    return {
        "dataset": dataset,
        "repo_id": DATASET_SPECS[dataset]["repo_id"],
        "rows": rows,
        "task_kind_counts": family_counts.most_common(),
        "examples": examples,
        "avg_message_count": round(message_total / rows, 2) if rows else 0.0,
        "avg_tool_calls": round(tool_call_total / rows, 2) if rows else 0.0,
        "tool_counts": tool_counts.most_common(12),
        "model_counts": model_counts.most_common(12),
        "insights": [
            "Single-string ChatML transcript dumps are useful parser-hardening targets because wrapper-heavy exports can otherwise misstate the underlying user task."
            if rows
            else "No local rows available.",
            "Command-output wrapper skipping matters here because many rows include shell transcripts before the next real user request."
            if rows
            else "Prompt extraction evidence unavailable.",
        ],
    }


def analyze_embedded_conversation_json_parquet_dataset(
    dataset: str,
    dataset_dir: Path,
    *,
    max_rows: int | None,
    examples_per_family: int,
) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    message_total = 0
    rows = 0

    for row in _iter_parquet_rows(_parquet_files(dataset, dataset_dir, None), max_rows=None):
        payload_text = str(row.get("conversation_json") or "")
        if _looks_like_embedded_meta_helper_row(payload_text):
            continue
        prompt = _extract_embedded_conversation_prompt(payload_text)
        if _looks_incomplete_or_junk_prompt(prompt):
            continue
        if _looks_like_embedded_orientation_prompt(prompt):
            continue
        if max_rows is not None and rows >= max_rows:
            break
        rows += 1
        model = _embedded_conversation_model(payload_text)
        if model:
            model_counts[model] += 1
        payload = _embedded_conversation_payload(payload_text)
        request = payload.get("request")
        response = payload.get("response")
        request_messages = request.get("messages") if isinstance(request, dict) else []
        message_total += len(request_messages) if isinstance(request_messages, list) else 0
        if isinstance(response, dict):
            message_total += 1
        family = task_review.classify_task_family(task="", prompt_preview=prompt)
        family_counts[family] += 1
        examples.setdefault(family, [])
        preview = _truncate(prompt)
        if preview not in examples[family] and len(examples[family]) < examples_per_family:
            examples[family].append(preview)

    return {
        "dataset": dataset,
        "repo_id": DATASET_SPECS[dataset]["repo_id"],
        "rows": rows,
        "task_kind_counts": family_counts.most_common(),
        "examples": examples,
        "avg_message_count": round(message_total / rows, 2) if rows else 0.0,
        "avg_tool_calls": 0.0,
        "tool_counts": [],
        "model_counts": model_counts.most_common(12),
        "insights": [
            "Embedded request and response JSON rows preserve real Claude Code prompts even when the public export stores them as one serialized field."
            if rows
            else "No local rows available.",
            "Wrapper stripping still matters because these serialized requests often prepend system-reminder blocks ahead of the task-bearing user content."
            if rows
            else "Prompt extraction evidence unavailable.",
        ],
    }


def analyze_merged_claude_code_parquet_dataset(
    dataset: str,
    dataset_dir: Path,
    *,
    max_rows: int | None,
    examples_per_family: int,
) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    source_repo_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    message_total = 0
    tool_call_total = 0
    rows = 0

    for row in _iter_parquet_rows(_parquet_files(dataset, dataset_dir, None), max_rows=None):
        prompt = _extract_merged_claude_code_prompt(row)
        if not prompt or _looks_incomplete_or_junk_prompt(prompt):
            continue
        if _looks_like_merged_claude_meta_helper_row(row, prompt):
            continue
        if max_rows is not None and rows >= max_rows:
            break
        rows += 1
        model = _safe_text(row.get("model"))
        if model:
            model_counts[model] += 1
        source_repo = _safe_text(row.get("source_repo"))
        if source_repo:
            source_repo_counts[source_repo] += 1
        message_total += _merged_claude_code_message_count(row, prompt)
        for name in _merged_claude_code_tool_names(row):
            tool_counts[name] += 1
            tool_call_total += 1
        family = task_review.classify_task_family(task="", prompt_preview=prompt)
        family_counts[family] += 1
        examples.setdefault(family, [])
        preview = _truncate(prompt)
        if preview not in examples[family] and len(examples[family]) < examples_per_family:
            examples[family].append(preview)

    return {
        "dataset": dataset,
        "repo_id": DATASET_SPECS[dataset]["repo_id"],
        "rows": rows,
        "task_kind_counts": family_counts.most_common(),
        "examples": examples,
        "avg_message_count": round(message_total / rows, 2) if rows else 0.0,
        "avg_tool_calls": round(tool_call_total / rows, 2) if rows else 0.0,
        "tool_counts": tool_counts.most_common(12),
        "model_counts": model_counts.most_common(12),
        "source_repo_counts": source_repo_counts.most_common(12),
        "insights": [
            "The merged Claude Code parquet gives a broader real-user session pool, but it still needs helper-row filtering so prompt taxonomy and tool counts are not polluted by title or orientation tasks."
            if rows
            else "No local rows available.",
            "Assistant tool_use parts inside messages_json are the reliable tool-call signal here; tools_json is the available-tool catalog, not the executed call trace."
            if tool_call_total
            else "Tool-call evidence unavailable.",
        ],
    }


def _download_claudeset_community_dataset(dataset: str, dataset_dir: Path, *, max_files: int | None) -> list[str]:
    return _download_claude_code_dataset(dataset, dataset_dir, max_files=max_files)


def _download_share_codex(dataset: str, dataset_dir: Path) -> list[str]:
    _, download = _require_hf_download_support()
    files = ["README.md", "export_manifest.json", "train.jsonl"]
    for filename in files:
        download(
            repo_id=DATASET_SPECS[dataset]["repo_id"],
            repo_type="dataset",
            filename=filename,
            local_dir=str(dataset_dir),
        )
    return files


def _download_message_bundle_dataset(dataset: str, dataset_dir: Path) -> list[str]:
    _, download = _require_hf_download_support()
    files = list(DATASET_SPECS.get(dataset, {}).get("extra_files") or [])
    session_glob = str(DATASET_SPECS.get(dataset, {}).get("session_glob") or "")
    if session_glob:
        files.append(session_glob)
    for filename in files:
        download(
            repo_id=DATASET_SPECS[dataset]["repo_id"],
            repo_type="dataset",
            filename=filename,
            local_dir=str(dataset_dir),
        )
    return files


def _download_openai_turn_bundle_dataset(dataset: str, dataset_dir: Path) -> list[str]:
    _, download = _require_hf_download_support()
    files = list(DATASET_SPECS.get(dataset, {}).get("extra_files") or [])
    session_glob = str(DATASET_SPECS.get(dataset, {}).get("session_glob") or "")
    if session_glob:
        files.append(session_glob)
    for filename in files:
        download(
            repo_id=DATASET_SPECS[dataset]["repo_id"],
            repo_type="dataset",
            filename=filename,
            local_dir=str(dataset_dir),
        )
    return files


def _download_parquet_dataset(dataset: str, dataset_dir: Path) -> list[str]:
    api_class, download = _require_hf_download_support()
    api = api_class()
    repo_id = DATASET_SPECS[dataset]["repo_id"]
    info = api.dataset_info(repo_id)
    session_glob = str(DATASET_SPECS.get(dataset, {}).get("session_glob") or "")
    extra_files = list(DATASET_SPECS.get(dataset, {}).get("extra_files") or [])
    parquet_files = sorted(
        str(item.rfilename)
        for item in getattr(info, "siblings", [])
        if isinstance(getattr(item, "rfilename", None), str)
        and item.rfilename.endswith(".parquet")
        and (not session_glob or Path(item.rfilename).match(session_glob))
    )
    files = [*extra_files, *parquet_files]
    for filename in files:
        download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=str(dataset_dir),
        )
    return files


def _download_pi_dataset(dataset: str, dataset_dir: Path, *, max_files: int | None) -> list[str]:
    api_class, download = _require_hf_download_support()
    api = api_class()
    repo_id = DATASET_SPECS[dataset]["repo_id"]
    info = api.dataset_info(repo_id)
    session_glob = str(DATASET_SPECS.get(dataset, {}).get("session_glob") or "")
    extra_files = list(DATASET_SPECS.get(dataset, {}).get("extra_files") or [])
    session_files = sorted(
        str(item.rfilename)
        for item in getattr(info, "siblings", [])
        if isinstance(getattr(item, "rfilename", None), str)
        and item.rfilename.endswith(".jsonl")
        and item.rfilename != "manifest.jsonl"
        and (
            item.rfilename.count("/") == 0
            or (session_glob and Path(item.rfilename).match(session_glob))
        )
    )
    if max_files is not None:
        session_files = session_files[:max_files]
    files = [*extra_files]
    if any(str(item.rfilename) == "manifest.jsonl" for item in getattr(info, "siblings", [])):
        files.append("manifest.jsonl")
    files.extend(session_files)
    for filename in files:
        download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=str(dataset_dir),
        )
    return files


def _download_claude_code_dataset(dataset: str, dataset_dir: Path, *, max_files: int | None) -> list[str]:
    api_class, download = _require_hf_download_support()
    api = api_class()
    repo_id = DATASET_SPECS[dataset]["repo_id"]
    info = api.dataset_info(repo_id)
    session_glob = str(DATASET_SPECS.get(dataset, {}).get("session_glob") or "")
    excluded_substrings = tuple(str(item) for item in (DATASET_SPECS.get(dataset, {}).get("exclude_filename_substrings") or []))
    extra_files = list(DATASET_SPECS.get(dataset, {}).get("extra_files") or [])
    jsonl_files = sorted(
        str(item.rfilename)
        for item in getattr(info, "siblings", [])
        if isinstance(getattr(item, "rfilename", None), str)
        and item.rfilename.endswith(".jsonl")
        and (not session_glob or Path(item.rfilename).match(session_glob))
        and not any(part and part in Path(item.rfilename).name for part in excluded_substrings)
    )
    if max_files is not None:
        jsonl_files = jsonl_files[:max_files]
    files = [*extra_files]
    if "README.md" not in files:
        files.append("README.md")
    files.extend(jsonl_files)
    for filename in files:
        download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=str(dataset_dir),
        )
    return files


def _download_nested_message_session_dataset(dataset: str, dataset_dir: Path, *, max_files: int | None) -> list[str]:
    return _download_claude_code_dataset(dataset, dataset_dir, max_files=max_files)


def _download_mixed_agent_session_dataset(dataset: str, dataset_dir: Path, *, max_files: int | None) -> list[str]:
    api_class, download = _require_hf_download_support()
    api = api_class()
    repo_id = DATASET_SPECS[dataset]["repo_id"]
    info = api.dataset_info(repo_id)
    patterns = (
        "sessions/claude/*.jsonl",
        "sessions/codex/*.jsonl",
        "sessions/pi/*.jsonl",
        "droid/*.jsonl",
    )
    session_files = sorted(
        str(item.rfilename)
        for item in getattr(info, "siblings", [])
        if isinstance(getattr(item, "rfilename", None), str)
        and any(Path(item.rfilename).match(pattern) for pattern in patterns)
    )
    if max_files is not None:
        session_files = session_files[:max_files]
    files = [*list(DATASET_SPECS.get(dataset, {}).get("extra_files") or []), *session_files]
    for filename in files:
        download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=str(dataset_dir),
        )
    return files


def _download_dataclaw_dataset(dataset: str, dataset_dir: Path, *, max_files: int | None) -> list[str]:
    api_class, download = _require_hf_download_support()
    api = api_class()
    repo_id = DATASET_SPECS[dataset]["repo_id"]
    info = api.dataset_info(repo_id)
    jsonl_files = sorted(
        str(item.rfilename)
        for item in getattr(info, "siblings", [])
        if isinstance(getattr(item, "rfilename", None), str) and item.rfilename.endswith(".jsonl")
    )
    if max_files is not None:
        jsonl_files = jsonl_files[:max_files]
    other_files = sorted(
        str(item.rfilename)
        for item in getattr(info, "siblings", [])
        if isinstance(getattr(item, "rfilename", None), str)
        and item.rfilename in {"README.md", "metadata.json", ".dataclaw/manifest.json"}
    )
    files = [*other_files, *jsonl_files]
    for filename in files:
        download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=str(dataset_dir),
        )
    return files


def ensure_local_cache(cache_root: Path, *, max_pi_files: int | None, max_dataclaw_files: int | None) -> dict[str, Any]:
    cache_root.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, list[str]] = {}
    for dataset, spec in DATASET_SPECS.items():
        dataset_dir = cache_root / spec["local_dir"]
        dataset_format = str(spec.get("format") or "")
        if dataset_format == "share-codex-jsonl":
            downloaded[dataset] = _download_share_codex(dataset, dataset_dir)
            continue
        if dataset_format == "message-bundle-jsonl":
            downloaded[dataset] = _download_message_bundle_dataset(dataset, dataset_dir)
            continue
        if dataset_format == "codex-session-jsonl":
            downloaded[dataset] = _download_claude_code_dataset(dataset, dataset_dir, max_files=max_dataclaw_files)
            continue
        if dataset_format == "dataclaw-session-jsonl":
            downloaded[dataset] = _download_dataclaw_dataset(dataset, dataset_dir, max_files=max_dataclaw_files)
            continue
        if dataset_format == "pi-session-jsonl":
            downloaded[dataset] = _download_pi_dataset(dataset, dataset_dir, max_files=max_pi_files)
            continue
        if dataset_format == "claude-code-session-jsonl":
            downloaded[dataset] = _download_claude_code_dataset(dataset, dataset_dir, max_files=max_pi_files)
            continue
        if dataset_format == "raw-codex-session-jsonl":
            downloaded[dataset] = _download_claude_code_dataset(dataset, dataset_dir, max_files=max_pi_files)
            continue
        if dataset_format == "nested-message-session-jsonl":
            downloaded[dataset] = _download_nested_message_session_dataset(dataset, dataset_dir, max_files=max_pi_files)
            continue
        if dataset_format == "mixed-agent-session-jsonl":
            downloaded[dataset] = _download_mixed_agent_session_dataset(dataset, dataset_dir, max_files=max_pi_files)
            continue
        if dataset_format == "claudeset-community-jsonl":
            downloaded[dataset] = _download_claudeset_community_dataset(dataset, dataset_dir, max_files=max_dataclaw_files)
            continue
        if dataset_format == "openai-turn-bundle-json":
            downloaded[dataset] = _download_openai_turn_bundle_dataset(dataset, dataset_dir)
            continue
        if dataset_format in {
            "chatml-transcript-parquet",
            "embedded-conversation-json-parquet",
            "merged-claude-code-parquet",
        }:
            downloaded[dataset] = _download_parquet_dataset(dataset, dataset_dir)
            continue
    return downloaded


def build_analysis(
    cache_root: Path,
    *,
    max_share_codex_rows: int | None,
    max_pi_files: int | None,
    max_dataclaw_files: int | None,
    examples_per_family: int,
) -> dict[str, Any]:
    datasets: dict[str, dict[str, Any]] = {}
    for dataset, spec in DATASET_SPECS.items():
        dataset_dir = cache_root / spec["local_dir"]
        dataset_format = str(spec.get("format") or "")
        if dataset_format == "share-codex-jsonl":
            datasets["share_codex"] = analyze_share_codex(
                dataset_dir,
                max_rows=max_share_codex_rows,
                examples_per_family=examples_per_family,
            )
            continue
        if dataset_format == "message-bundle-jsonl":
            datasets[dataset.replace("-", "_")] = analyze_message_bundle_dataset(
                dataset,
                dataset_dir,
                max_rows=max_dataclaw_files,
                examples_per_family=examples_per_family,
            )
            continue
        if dataset_format == "codex-session-jsonl":
            datasets[dataset.replace("-", "_")] = analyze_codex_sessions(
                dataset_dir,
                max_rows=max_dataclaw_files,
                examples_per_family=examples_per_family,
            )
            continue
        if dataset_format == "dataclaw-session-jsonl":
            datasets[dataset.replace("-", "_")] = analyze_dataclaw_dataset(
                dataset,
                dataset_dir,
                max_files=max_dataclaw_files,
                examples_per_family=examples_per_family,
            )
            continue
        if dataset_format == "pi-session-jsonl":
            datasets[dataset.replace("-", "_")] = analyze_pi_dataset(
                dataset,
                dataset_dir,
                max_files=max_pi_files,
                examples_per_family=examples_per_family,
            )
            continue
        if dataset_format == "claude-code-session-jsonl":
            datasets[dataset.replace("-", "_")] = analyze_claude_code_dataset(
                dataset,
                dataset_dir,
                max_files=max_pi_files,
                examples_per_family=examples_per_family,
            )
            continue
        if dataset_format == "raw-codex-session-jsonl":
            datasets[dataset.replace("-", "_")] = analyze_raw_codex_session_dataset(
                dataset,
                dataset_dir,
                max_files=max_pi_files,
                examples_per_family=examples_per_family,
            )
            continue
        if dataset_format == "nested-message-session-jsonl":
            datasets[dataset.replace("-", "_")] = analyze_nested_message_session_dataset(
                dataset,
                dataset_dir,
                max_files=max_pi_files,
                examples_per_family=examples_per_family,
            )
            continue
        if dataset_format == "mixed-agent-session-jsonl":
            datasets[dataset.replace("-", "_")] = analyze_mixed_agent_session_dataset(
                dataset,
                dataset_dir,
                max_files=max_pi_files,
                examples_per_family=examples_per_family,
            )
            continue
        if dataset_format == "claudeset-community-jsonl":
            datasets[dataset.replace("-", "_")] = analyze_claudeset_community_dataset(
                dataset_dir,
                max_rows=max_dataclaw_files,
                examples_per_family=examples_per_family,
            )
            continue
        if dataset_format == "openai-turn-bundle-json":
            datasets[dataset.replace("-", "_")] = analyze_openai_turn_bundle_dataset(
                dataset,
                dataset_dir,
                max_rows=max_dataclaw_files,
                examples_per_family=examples_per_family,
            )
            continue
        if dataset_format == "chatml-transcript-parquet":
            datasets[dataset.replace("-", "_")] = analyze_chatml_transcript_parquet_dataset(
                dataset,
                dataset_dir,
                max_rows=max_dataclaw_files,
                examples_per_family=examples_per_family,
            )
            continue
        if dataset_format == "embedded-conversation-json-parquet":
            datasets[dataset.replace("-", "_")] = analyze_embedded_conversation_json_parquet_dataset(
                dataset,
                dataset_dir,
                max_rows=max_dataclaw_files,
                examples_per_family=examples_per_family,
            )
            continue
        if dataset_format == "merged-claude-code-parquet":
            datasets[dataset.replace("-", "_")] = analyze_merged_claude_code_parquet_dataset(
                dataset,
                dataset_dir,
                max_rows=max_dataclaw_files,
                examples_per_family=examples_per_family,
            )
            continue
    recommendations = [
        "Detect issue-review, PR-review, and audit-style prompts early, then summarize once and stop replaying large fetched context back into later turns.",
        "Compress shell and tool-result replay before the next model turn; the real-user corpora stay shell-heavy even across different repos and tasks.",
        "Route obvious control or housekeeping prompts such as `hello`, `continue`, `/load`, and dummy test loops through a much cheaper path than the full coding-agent loop.",
        "Bias earlier toward direct validation after edits instead of more narration or repo re-orientation.",
        "Trim first-turn orientation prompts when the user request already names a concrete file, bug, or feature.",
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cache_root": str(cache_root.resolve(strict=False)),
        "datasets": datasets,
        "top_recommendations": recommendations,
    }


def _format_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Web-Discovered Agent Dataset Analysis", ""]
    for name, summary in (payload.get("datasets") or {}).items():
        if not isinstance(summary, dict):
            continue
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- repo_id: `{summary.get('repo_id')}`")
        lines.append(f"- rows: `{summary.get('rows')}`")
        lines.append(f"- avg_message_count: `{summary.get('avg_message_count')}`")
        lines.append(f"- avg_tool_calls: `{summary.get('avg_tool_calls')}`")
        if "avg_input_tokens" in summary:
            lines.append(f"- avg_input_tokens: `{summary.get('avg_input_tokens')}`")
        if "avg_output_tokens" in summary:
            lines.append(f"- avg_output_tokens: `{summary.get('avg_output_tokens')}`")
        if "avg_input_tokens_per_tool" in summary:
            lines.append(f"- avg_input_tokens_per_tool: `{summary.get('avg_input_tokens_per_tool')}`")
        task_counts = summary.get("task_kind_counts") or []
        if task_counts:
            rendered = ", ".join(f"`{family}` ({count})" for family, count in task_counts[:8])
            lines.append(f"- task_kind_counts: {rendered}")
        tool_counts = summary.get("tool_counts") or []
        if tool_counts:
            rendered = ", ".join(f"`{tool}` ({count})" for tool, count in tool_counts[:8])
            lines.append(f"- tool_counts: {rendered}")
        insights = summary.get("insights") or []
        for insight in insights:
            lines.append(f"- {insight}")
        examples = summary.get("examples") or {}
        if isinstance(examples, dict):
            for family, values in examples.items():
                if not isinstance(values, list) or not values:
                    continue
                lines.append(f"- examples/{family}: " + " | ".join(values))
        lines.append("")
    recommendations = payload.get("top_recommendations") or []
    if recommendations:
        lines.append("## Recommendations")
        lines.append("")
        for item in recommendations:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and analyze newly discovered public coding-agent session datasets.")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT, help="Local cache root for downloaded web-discovered datasets.")
    parser.add_argument("--max-share-codex-rows", type=int, default=400, help="Maximum share-codex rows to analyze. Use 0 for all rows.")
    parser.add_argument("--max-pi-files", type=int, default=24, help="Maximum pi-mono session files to download and analyze. Use 0 for all local files.")
    parser.add_argument("--max-dataclaw-files", type=int, default=24, help="Maximum DataClaw shard files to download and analyze. Use 0 for all local files.")
    parser.add_argument("--examples-per-family", type=int, default=2, help="Prompt examples to keep per task family.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON output path.")
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MARKDOWN_OUTPUT, help="Markdown output path.")
    parser.add_argument("--skip-download", action="store_true", help="Analyze only already cached local files.")
    args = parser.parse_args(argv)

    max_share_codex_rows = None if args.max_share_codex_rows == 0 else args.max_share_codex_rows
    max_pi_files = None if args.max_pi_files == 0 else args.max_pi_files
    max_dataclaw_files = None if args.max_dataclaw_files == 0 else args.max_dataclaw_files
    downloaded: dict[str, Any] | None = None
    if not args.skip_download:
        downloaded = ensure_local_cache(
            args.cache_root,
            max_pi_files=max_pi_files,
            max_dataclaw_files=max_dataclaw_files,
        )
    payload = build_analysis(
        args.cache_root,
        max_share_codex_rows=max_share_codex_rows,
        max_pi_files=max_pi_files,
        max_dataclaw_files=max_dataclaw_files,
        examples_per_family=args.examples_per_family,
    )
    if downloaded is not None:
        payload["downloaded_files"] = downloaded
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_format_markdown(payload), encoding="utf-8")
    for name, summary in payload["datasets"].items():
        print(
            "[web-discovered-agent-datasets] "
            + f"dataset={name} rows={summary.get('rows', 0)} avg_tool_calls={summary.get('avg_tool_calls', 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
