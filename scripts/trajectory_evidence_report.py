from __future__ import annotations

import argparse
import ast
import json
import re
import shlex
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import trajectory_dataset_fetch, trajectory_error_profile, trajectory_profile


DEFAULT_DATA_ROOT = Path("scratch") / "external" / "datasets"
DEFAULT_JSON_OUTPUT = DEFAULT_DATA_ROOT / "trajectory-evidence-report.json"
DEFAULT_MARKDOWN_OUTPUT = DEFAULT_DATA_ROOT / "trajectory-evidence-report.md"
DEFAULT_REFERENCE_TRAJECTORY_PROFILE = DEFAULT_DATA_ROOT / "trajectory-profile-all-20260516.json"
DEFAULT_REFERENCE_ERROR_PROFILE = DEFAULT_DATA_ROOT / "trajectory-error-profile-full-20260516.json"

MESSAGE_THEME_META: dict[str, dict[str, str]] = {
    "large-system-prompt": {
        "category": "token_waste",
        "title": "Large system prompts",
        "description": "Very large system messages that are likely replayed every turn.",
    },
    "planning-monologue": {
        "category": "token_waste",
        "title": "Long planning monologues",
        "description": "Assistant planning messages that consume budget before a concrete tool step.",
    },
    "plan-plus-tool-call-blob": {
        "category": "token_waste",
        "title": "Long assistant messages before tool calls",
        "description": "Assistant messages that both narrate a long plan and invoke tools in the same turn.",
    },
    "large-context-blob": {
        "category": "token_waste",
        "title": "Large context blobs",
        "description": "Large read/search/tool-result messages that can be compacted before the next model turn.",
    },
    "large-failure-blob": {
        "category": "token_waste",
        "title": "Large failure blobs",
        "description": "Large error outputs such as tracebacks or failed test logs that should be summarized.",
    },
    "explicit-tool-request": {
        "category": "routing",
        "title": "Explicit tool requests",
        "description": "User messages that explicitly name a concrete tool and are strong candidates for deterministic routing.",
    },
}

ROW_PATTERN_META: dict[str, dict[str, str]] = {
    "context-loop-row": {
        "category": "stuckness",
        "title": "Context-only loops",
        "description": "Trajectories that keep reading/searching without progressing.",
    },
    "mechanical-turn-row": {
        "category": "routing",
        "title": "Mechanical tool turns",
        "description": "Rows that only need read/search/test/git routing and do not require broad model planning.",
    },
    "edit-without-context-row": {
        "category": "quality_risk",
        "title": "Edits without grounding",
        "description": "Rows that mutate before reading relevant source or failure evidence.",
    },
    "edit-without-test-row": {
        "category": "quality_risk",
        "title": "Edits without later validation",
        "description": "Rows that edit code but never run a later test in the same trajectory.",
    },
}

PRODUCT_FIXES: dict[str, dict[str, Any]] = {
    "mechanical-router": {
        "status": "partial",
        "summary": "Deterministic routing exists for read/search/symbol/test/git exact paths, including plain-language symbol return reads, direct implementation-target requests, and test-grounded return rewrites that resolve implementation targets before editing.",
        "references": [
            "ollama_code/controller/navigation_validation.py",
            "ollama_code/agent.py",
        ],
        "next_gap": "Mutation and failure-recovery routing still lives in the legacy agent monolith, and only part of the tool-selection surface is routed before the LLM loop.",
    },
    "context-planner": {
        "status": "partial",
        "summary": "Compact context tools exist, the agent can preload context_pack, structured context_pack ranking can auto-promote straight to read_file or read_symbol before another broad inspection, simple shell inspection can normalize to read_file, list_files, search, file_search, directory_search, or bounded read_file previews, repeated broad test inspection can auto-map tests to implementation targets, grounded implementation targets feed later source narrowing, grounded implementation-target symbols can auto-promote straight to read_symbol including when a recent identifier query disambiguates multi-symbol targets, repeated broad identifier inspection can auto-switch to repo-wide or source-scoped search_symbols before a source file is read, repeated broad repo search can auto-switch to code_outline when the latest search or list_files output narrows to exactly one non-test code file, repeated broad source inspection can auto-switch to search_symbols or code_outline, and grounded search-symbol or single-symbol outline results can auto-promote to read_symbol on later broad relapses.",
        "references": [
            "ollama_code/agent.py",
            "ollama_code/tools/__init__.py",
        ],
        "next_gap": "Only a subset of narrowing routes are proactive; broader inspection still relies on heuristics embedded in agent.py instead of a dedicated planner/controller.",
    },
    "loop-cap": {
        "status": "implemented",
        "summary": "trajectory-guards block repeated context-only loops and repeated tool failures.",
        "references": [
            "ollama_code/agent.py",
        ],
        "next_gap": "The loop policy is implemented, but it is not yet decomposed into its own controller module.",
    },
    "post-edit-validation": {
        "status": "partial",
        "summary": "After a successful edit, trajectory-guards proactively force lint_typecheck, contract_check, select_tests, and run_test for code changes, empty targeted-test selection can fall back through discover_validators before running repo tests, code edits can promote discovered check, typecheck, lint, or syntax validators when no runnable tests exist, and non-code edits can promote configured tests, discovered test commands, or discovered non-test validator commands before extra context gathering or a final answer.",
        "references": [
            "ollama_code/agent.py",
            "ollama_code/tools/__init__.py",
        ],
        "next_gap": "Non-code validation now promotes available discovered validator commands, but release gating still does not cover enough real doc/config-only edit scenarios.",
    },
    "failure-compression": {
        "status": "implemented",
        "summary": "Failed run_test output is compacted, repeated unchanged reruns auto-trigger diagnose_test_failure before another model retry, the first post-failure context-only relapse triggers diagnose_test_failure before more broad inspection, and proactive post-edit validation failures echo compact validator diagnostics before asking the model to repair.",
        "references": [
            "ollama_code/agent.py",
            "ollama_code/tools/__init__.py",
        ],
        "next_gap": "Other validator families still do not all route through the same failure-compression path.",
    },
    "ground-before-mutate": {
        "status": "partial",
        "summary": "Grounding guards exist, explicit mutation targets can auto-read the exact file or symbol before retry, failed tests without an explicit edit path can auto-route through find_implementation_target or diagnose_test_failure, pathless edits can auto-ground from a single explicitly named source path, from multiple explicitly named source paths when only one defines the requested symbol or one clearly wins focused test affinity, from otherwise ambiguous explicitly named source paths when current-turn source or test evidence points to one of those named files, can escalate unresolved multi-source ambiguity through context_pack instead of arbitrarily grounding the first file, can reuse structured context_pack ranking to ground directly to a ranked file or symbol before a new repo-wide symbol search, and no-context pathless symbol edits can auto-narrow through repo-wide search_symbols to either a unique symbol, a single non-test source file, a clearly test-affined implementation candidate, or a current-turn source/test-context candidate when the tied repo search results already include it.",
        "references": [
            "ollama_code/agent.py",
            "ollama_code/tools/__init__.py",
        ],
        "next_gap": "Multiple explicitly named source paths that still share the requested symbol after focused test affinity and without current-turn source or test evidence pointing to one named file, multiple non-test source candidates that stay tied even after focused test affinity and without current-turn source or test evidence pointing to one candidate, or repo-wide searches that still do not collapse to one likely source file, still need manual disambiguation before pathless mutation can be grounded.",
    },
    "diagnose-test-failure": {
        "status": "implemented",
        "summary": "Test failures are compacted and can be diagnosed with a dedicated tool.",
        "references": [
            "ollama_code/tools/__init__.py",
            "ollama_code/agent.py",
        ],
        "next_gap": "The diagnosis logic works, but benchmark gating does not enforce it as a release requirement yet.",
    },
    "syntax-repair-gate": {
        "status": "implemented",
        "summary": "Python writes surface syntax diagnostics immediately and the agent blocks unsafe follow-up claims.",
        "references": [
            "ollama_code/tools/__init__.py",
            "ollama_code/agent.py",
        ],
        "next_gap": "Structured-edit coverage is still language- and path-dependent.",
    },
    "validated-cli-preflight": {
        "status": "implemented",
        "summary": "Common shell/test/git command families are validated before execution, pre-execution Bash syntax rejections are classified as syntax errors, and repeated inline shell syntax failures are blocked with temp-script or simpler-command guidance.",
        "references": [
            "ollama_code/tools/__init__.py",
            "ollama_code/agent.py",
        ],
        "next_gap": "Proof is mostly local-script based; CI does not yet gate on these live paths.",
    },
    "dependency-or-import-guard": {
        "status": "implemented",
        "summary": "Dependency/import failures are classified, surfaced with targeted guidance, and repeated blind retries auto-trigger structured diagnosis.",
        "references": [
            "ollama_code/tools/__init__.py",
            "ollama_code/agent.py",
        ],
        "next_gap": "The first failure still relies on normal tool feedback; only repeated failures auto-trigger controller diagnosis.",
    },
    "bounded-command-validation": {
        "status": "implemented",
        "summary": "Command validation and timeout-aware failure handling exist for shell-heavy turns.",
        "references": [
            "ollama_code/tools/__init__.py",
            "ollama_code/agent.py",
        ],
        "next_gap": "Long-running real-model regression suites are still manual instead of CI-gated.",
    },
    "path-repair-guard": {
        "status": "implemented",
        "summary": "Workspace-escape and missing-path errors are blocked and suggested paths are surfaced.",
        "references": [
            "ollama_code/tools/__init__.py",
            "ollama_code/agent.py",
        ],
        "next_gap": "The broader UX still depends on the model following the guard hints rather than a dedicated path-repair controller.",
    },
    "fail-closed-permission": {
        "status": "implemented",
        "summary": "Permission failures fail closed instead of auto-retrying destructive paths.",
        "references": [
            "ollama_code/tools/__init__.py",
        ],
        "next_gap": "There is not yet a separate release report proving permission handling across live model runs.",
    },
    "structured-edit-preflight": {
        "status": "partial",
        "summary": "Structured edit tools and syntax checks exist.",
        "references": [
            "ollama_code/tools/__init__.py",
        ],
        "next_gap": "The edit layer is still monolithic and needs further extraction before it is easy to trust or sell.",
    },
}

ROW_PATTERN_FIX_MAP: dict[str, tuple[str, ...]] = {
    "mechanical-turn-row": ("mechanical-router",),
    "context-loop-row": ("loop-cap", "context-planner"),
    "edit-without-context-row": ("ground-before-mutate",),
    "edit-without-test-row": ("post-edit-validation",),
}

MESSAGE_THEME_FIX_MAP: dict[str, tuple[str, ...]] = {
    "large-failure-blob": ("failure-compression",),
}

SHELL_ERROR_FIX_MAP: dict[str, tuple[str, ...]] = {
    "bash_unexpected_token": ("validated-cli-preflight",),
    "bash_unexpected_eof": ("validated-cli-preflight",),
    "unmatched_quote": ("validated-cli-preflight",),
    "command_not_found": ("validated-cli-preflight",),
    "missing_file_or_dir": ("path-repair-guard",),
    "unrecognized_argument": ("validated-cli-preflight",),
    "missing_operand": ("validated-cli-preflight",),
    "permission_denied": ("fail-closed-permission",),
}


@dataclass(frozen=True)
class MessageRecord:
    dataset: str
    row_id: str
    message_index: int
    role: str
    kind: str
    name: str
    category: str
    content: str
    content_chars: int
    tool_call_names: tuple[str, ...] = ()
    tool_arguments: dict[str, Any] | None = None
    error_class: str | None = None
    shell_error: str | None = None


def _row_id(row: dict[str, Any], row_number: int) -> str:
    for key in ("instance_id", "trajectory_id", "traj_id", "trial_id", "trial_name", "task_name", "session_id", "source_id", "target", "repo"):
        value = row.get(key)
        if value:
            return str(value)
    return f"row-{row_number}"


def _normalize_tool_call_names(raw_calls: Any) -> tuple[str, ...]:
    return tuple(name for name, _arguments in _normalize_tool_calls(raw_calls))


def _normalize_tool_calls(raw_calls: Any) -> list[tuple[str, dict[str, Any]]]:
    normalized = trajectory_profile._deserialize_possible_json(raw_calls)
    calls: list[tuple[str, dict[str, Any]]] = []
    if not isinstance(normalized, list):
        return []
    for call in normalized:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else None
        candidate = ""
        raw_arguments: Any = {}
        if function is not None:
            candidate = str(function.get("name") or "")
            raw_arguments = function.get("arguments")
        if not candidate:
            candidate = str(call.get("name") or "")
            raw_arguments = call.get("arguments")
        candidate = trajectory_profile._normalize_tool_name(candidate)
        if candidate:
            parsed_arguments = trajectory_profile._deserialize_possible_json(raw_arguments)
            calls.append((candidate, parsed_arguments if isinstance(parsed_arguments, dict) else {}))
    return calls


def _make_message_record(
    *,
    dataset: str,
    row_id: str,
    message_index: int,
    role: str,
    kind: str,
    name: str,
    category: str,
    content: str,
    tool_call_names: tuple[str, ...] = (),
    tool_arguments: dict[str, Any] | None = None,
) -> MessageRecord:
    lowered = content.lower()
    looks_like_error = any(
        token in lowered
        for token in (
            "traceback",
            "assert",
            "error",
            "exception",
            "failed",
            "syntax",
            "timeout",
            "permission denied",
            "no such file",
            "command not found",
        )
    )
    error_class = trajectory_error_profile.classify_error(content) if looks_like_error else None
    shell_error = trajectory_error_profile.classify_shell_error(content) if looks_like_error else None
    return MessageRecord(
        dataset=dataset,
        row_id=row_id,
        message_index=message_index,
        role=role,
        kind=kind,
        name=name,
        category=category,
        content=content,
        content_chars=len(content),
        tool_call_names=tool_call_names,
        tool_arguments=tool_arguments,
        error_class=error_class,
        shell_error=shell_error,
    )


def _extract_openhands_messages(dataset: str, row: dict[str, Any], row_number: int) -> list[MessageRecord]:
    row_id = _row_id(row, row_number)
    records: list[MessageRecord] = []
    messages = trajectory_profile._openhands_messages(row)
    pending_tool_name = ""
    tool_names_by_id: dict[str, str] = {}
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").lower()
        content = trajectory_profile._content_text(message.get("content"))
        raw_display_name = str(message.get("name") or "")
        display_name = trajectory_profile._normalize_tool_name(raw_display_name)
        tool_calls = _normalize_tool_calls(
            message.get("tool_calls") if message.get("tool_calls") is not None else message.get("tool_calls_json")
        )
        tool_names_by_id.update(trajectory_profile._tool_call_name_by_id(message))
        tool_call_names = tuple(name for name, _arguments in tool_calls)
        inferred = ""
        if role in {"assistant", "ai"} and not tool_call_names:
            inferred = trajectory_profile._infer_tool_name_from_text(content) or ""
        message_tool_calls = tool_call_names or ((inferred,) if inferred else ())
        if role in {"assistant", "ai"}:
            pending_tool_name = message_tool_calls[0] if len(message_tool_calls) == 1 else ""
        elif role == "tool" and trajectory_profile._placeholder_tool_name(raw_display_name):
            tool_call_id = str(message.get("tool_call_id") or "")
            display_name = tool_names_by_id.get(tool_call_id) or pending_tool_name or display_name
        category_name = display_name or (message_tool_calls[0] if len(message_tool_calls) == 1 else "")
        category = trajectory_profile._tool_category(category_name, content)
        records.append(
            _make_message_record(
                dataset=dataset,
                row_id=row_id,
                message_index=index,
                role=role,
                kind="message",
                name=display_name if role == "tool" else "",
                category=category,
                content=content,
                tool_call_names=message_tool_calls,
            )
        )
        record_tool_calls = tool_calls if tool_calls else [(tool_name, {}) for tool_name in message_tool_calls]
        for tool_name, tool_arguments in record_tool_calls:
            records.append(
                _make_message_record(
                    dataset=dataset,
                    row_id=row_id,
                    message_index=index,
                    role=role,
                    kind="tool_call",
                    name=tool_name,
                    category=trajectory_profile._tool_category(tool_name, content),
                    content="",
                    tool_call_names=(tool_name,),
                    tool_arguments=tool_arguments,
                )
            )
        if role == "tool":
            pending_tool_name = ""
    return records


def _extract_swe_agent_messages(dataset: str, row: dict[str, Any], row_number: int) -> list[MessageRecord]:
    row_id = _row_id(row, row_number)
    records: list[MessageRecord] = []
    pending_observation_name = ""
    pending_observation_category = "other"
    for index, message in enumerate(list(row.get("trajectory") or [])):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").lower()
        content = trajectory_profile._content_text(message.get("text") or message.get("system_prompt"))
        inferred = trajectory_profile._infer_tool_name_from_text(content) if role in {"ai", "assistant"} else None
        tool_call_names = (inferred,) if inferred else ()
        message_name = ""
        message_category = trajectory_profile._tool_category(inferred or "", content)
        if role == "user" and pending_observation_name:
            message_name = pending_observation_name
            message_category = pending_observation_category
        records.append(
            _make_message_record(
                dataset=dataset,
                row_id=row_id,
                message_index=index,
                role=role,
                kind="message",
                name=message_name,
                category=message_category,
                content=content,
                tool_call_names=tool_call_names,
            )
        )
        if tool_call_names:
            tool_arguments = {"command": _inline_shell_command_from_content(content)} if inferred == "run_shell" else None
            records.append(
                _make_message_record(
                    dataset=dataset,
                    row_id=row_id,
                    message_index=index,
                    role=role,
                    kind="tool_call",
                    name=str(inferred),
                    category=trajectory_profile._tool_category(str(inferred), content),
                    content="",
                    tool_call_names=tool_call_names,
                    tool_arguments=tool_arguments,
                )
            )
            pending_observation_name = str(inferred or "")
            pending_observation_category = trajectory_profile._tool_category(str(inferred or ""), content)
        elif role == "user":
            pending_observation_name = ""
            pending_observation_category = "other"
    return records


def _extract_smith_messages(dataset: str, row: dict[str, Any], row_number: int) -> list[MessageRecord]:
    row_id = _row_id(row, row_number)
    records: list[MessageRecord] = []
    messages = trajectory_profile._deserialize_possible_json(row.get("messages"))
    if not isinstance(messages, list):
        return records
    pending_observation_name = ""
    pending_observation_category = "other"
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").lower()
        content = trajectory_profile._content_text(message.get("content"))
        display_name = trajectory_profile._normalize_tool_name(str(message.get("name") or ""))
        tool_calls = _normalize_tool_calls(message.get("tool_calls"))
        tool_call_names = tuple(name for name, _arguments in tool_calls)
        inferred = ""
        if role in {"assistant", "ai"} and not tool_call_names:
            inferred = trajectory_profile._infer_tool_name_from_text(content) or ""
        message_tool_calls = tool_call_names or ((inferred,) if inferred else ())
        category_name = display_name or (message_tool_calls[0] if len(message_tool_calls) == 1 else "")
        message_name = display_name if role == "tool" else ""
        if role == "user" and pending_observation_name:
            message_name = pending_observation_name
            category_name = pending_observation_name
        records.append(
            _make_message_record(
                dataset=dataset,
                row_id=row_id,
                message_index=index,
                role=role,
                kind="message",
                name=message_name,
                category=trajectory_profile._tool_category(category_name, content),
                content=content,
                tool_call_names=message_tool_calls,
            )
        )
        record_tool_calls = tool_calls
        if not record_tool_calls:
            record_tool_calls = []
            for tool_name in message_tool_calls:
                tool_arguments = {"command": _inline_shell_command_from_content(content)} if tool_name == "run_shell" else {}
                record_tool_calls.append((tool_name, tool_arguments))
        for tool_name, tool_arguments in record_tool_calls:
            records.append(
                _make_message_record(
                    dataset=dataset,
                    row_id=row_id,
                    message_index=index,
                    role=role,
                    kind="tool_call",
                    name=tool_name,
                    category=trajectory_profile._tool_category(tool_name, content),
                    content="",
                    tool_call_names=(tool_name,),
                    tool_arguments=tool_arguments,
                )
            )
        if role in {"assistant", "ai"} and len(message_tool_calls) == 1:
            pending_observation_name = message_tool_calls[0]
            pending_observation_category = trajectory_profile._tool_category(message_tool_calls[0], content)
        elif role == "user":
            pending_observation_name = ""
            pending_observation_category = "other"
    return records


def _extract_terminalbench_messages(dataset: str, row: dict[str, Any], row_number: int) -> list[MessageRecord]:
    row_id = _row_id(row, row_number)
    records: list[MessageRecord] = []
    steps = trajectory_profile._deserialize_possible_json(row.get("steps"))
    if not isinstance(steps, list):
        return records
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        role = str(step.get("src") or "unknown").lower()
        content = trajectory_profile._content_text(step.get("msg"))
        raw_tools = step.get("tools")
        tools = raw_tools if isinstance(raw_tools, list) else []
        normalized_tools: list[tuple[str, dict[str, Any]]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            name = trajectory_profile._normalize_terminalbench_tool_name(str(tool.get("fn") or ""))
            if not name:
                continue
            arguments = {"command": str(tool.get("cmd") or "").strip()} if str(tool.get("cmd") or "").strip() else {}
            normalized_tools.append((name, arguments))
        tool_call_names = tuple(name for name, _arguments in normalized_tools)
        inferred = ""
        if role in {"assistant", "agent"} and not tool_call_names:
            inferred = trajectory_profile._infer_tool_name_from_text(content) or ""
        message_tool_calls = tool_call_names or ((inferred,) if inferred else ())
        records.append(
            _make_message_record(
                dataset=dataset,
                row_id=row_id,
                message_index=index,
                role=role,
                kind="message",
                name="",
                category=trajectory_profile._tool_category(message_tool_calls[0] if len(message_tool_calls) == 1 else "", content),
                content=content,
                tool_call_names=message_tool_calls,
            )
        )
        record_tool_calls = normalized_tools if normalized_tools else [(tool_name, {"command": _inline_shell_command_from_content(content)} if tool_name == "run_shell" else {}) for tool_name in message_tool_calls]
        for tool_name, tool_arguments in record_tool_calls:
            records.append(
                _make_message_record(
                    dataset=dataset,
                    row_id=row_id,
                    message_index=index,
                    role=role,
                    kind="tool_call",
                    name=tool_name,
                    category=trajectory_profile._tool_category(tool_name, content),
                    content="",
                    tool_call_names=(tool_name,),
                    tool_arguments=tool_arguments,
                )
            )
        observation = trajectory_profile._content_text(step.get("obs"))
        if observation.strip():
            observation_name = message_tool_calls[0] if len(message_tool_calls) == 1 else ""
            observation_category = trajectory_profile._tool_category(observation_name, observation)
            records.append(
                _make_message_record(
                    dataset=dataset,
                    row_id=row_id,
                    message_index=index,
                    role="tool",
                    kind="message",
                    name=observation_name,
                    category=observation_category,
                    content=observation,
                )
            )
    return records


def extract_message_records(dataset: str, adapter: str, row: dict[str, Any], row_number: int) -> list[MessageRecord]:
    if adapter == "openhands":
        return _extract_openhands_messages(dataset, row, row_number)
    if adapter == "trace_commons":
        return _extract_openhands_messages(dataset, row, row_number)
    if adapter == "thoughtworks":
        effective_adapter = trajectory_profile._thoughtworks_row_adapter(row)
        if effective_adapter == "openhands":
            return _extract_openhands_messages(dataset, row, row_number)
        return _extract_smith_messages(
            dataset,
            {"messages": row.get("messages") or row.get("messages_json"), "traj_id": row.get("session_id") or row.get("source_id")},
            row_number,
        )
    if adapter == "swe_agent":
        return _extract_swe_agent_messages(dataset, row, row_number)
    if adapter == "smith":
        return _extract_smith_messages(dataset, row, row_number)
    if adapter == "terminalbench":
        return _extract_terminalbench_messages(dataset, row, row_number)
    return []


def _message_excerpt(text: str, *, limit: int = 220) -> str:
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - 16)] + "... truncated ..."


def _inline_shell_command_from_content(content: str) -> str:
    function_match = re.search(
        r"<function=(?:bash|run_shell|shell)>\s*<parameter=command>(.*?)</parameter>",
        content,
        re.IGNORECASE | re.DOTALL,
    )
    if function_match:
        return " ".join(function_match.group(1).split()).strip()
    fence_match = re.search(r"```(?:bash|sh|shell)?\s*(.*?)```", content, re.IGNORECASE | re.DOTALL)
    if fence_match:
        block = fence_match.group(1).strip()
        if block:
            first_line = next((line.strip() for line in block.splitlines() if line.strip()), "")
            return first_line.lstrip("$ ").strip()
    return ""


def _shell_command_from_record(record: MessageRecord) -> str:
    if record.name not in {"execute_bash", "run_shell", "bash", "powershell", "shell", "run_shell_command", "bash_command", "execute_ipython_cell"}:
        return ""
    arguments = record.tool_arguments if isinstance(record.tool_arguments, dict) else {}
    for key in ("command", "cmd"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            command = value.strip()
            focused = _focused_shell_segment(command)
            try:
                argv = shlex.split(focused, posix=True)
            except ValueError:
                argv = focused.split()
            if not argv:
                return ""
            family = Path(argv[0]).name.lower().rstrip(".,:;")
            if _is_pseudo_shell_family(family):
                return ""
            return command
    return ""


def _is_pseudo_shell_family(family: str) -> bool:
    return family.startswith("$") or family.startswith("#") or family in {
        "",
        "#",
        "{",
        "}",
        "cd",
        "edit",
        "submit",
        "scroll_down",
        "scroll_up",
        "create",
        "goto",
        "open",
        "c-c",
        "ctrl+c",
        "search_file",
        "search_dir",
        "find_file",
        "read_file",
        "list_files",
        "code_outline",
        "read_symbol",
        "search_symbols",
        "if",
        "then",
        "else",
        "elif",
        "fi",
        "for",
        "while",
        "do",
        "done",
        "case",
        "esac",
        "function",
        "q",
        "c-d",
        "c-z",
    }


def _is_wrapper_shell_family(family: str) -> bool:
    return family in {"echo", "printf", "sleep", "wait", "export", "source", "."}


def _is_setup_wrapper_segment(family: str, argv: list[str], *, has_more_segments: bool) -> bool:
    if family == "chmod" and has_more_segments:
        return True
    return _is_wrapper_shell_family(family)


def _strip_leading_env_assignments(argv: list[str]) -> list[str]:
    index = 0
    while index < len(argv):
        token = argv[index]
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", token):
            break
        index += 1
    return argv[index:]


def _split_shell_segments(command: str) -> list[str]:
    segments: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escape = False
    index = 0
    while index < len(command):
        char = command[index]
        if escape:
            current.append(char)
            escape = False
            index += 1
            continue
        if char == "\\":
            current.append(char)
            escape = True
            index += 1
            continue
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            current.append(char)
            quote = char
            index += 1
            continue
        if command.startswith("&&", index) or command.startswith("||", index):
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []
            index += 2
            continue
        if char == ";":
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []
            index += 1
            continue
        current.append(char)
        index += 1
    tail = "".join(current).strip()
    if tail:
        segments.append(tail)
    return segments


def _focused_shell_segment(command: str) -> str:
    unwrapped = _unwrap_shell_wrapper(command)
    if unwrapped != command:
        return _focused_shell_segment(unwrapped)
    command = re.sub(r"\\\s*\n\s*", " ", command)
    segments = _split_shell_segments(command)
    saw_only_pseudo = False
    for index, segment in enumerate(segments):
        try:
            argv = shlex.split(segment, posix=True)
        except ValueError:
            argv = segment.split()
        argv = _strip_leading_env_assignments(argv)
        if argv:
            family = Path(argv[0]).name.lower().rstrip(".,:;")
            has_more_segments = index < len(segments) - 1
            if not _is_pseudo_shell_family(family) and not _is_setup_wrapper_segment(family, argv, has_more_segments=has_more_segments):
                return segment
            if _is_pseudo_shell_family(family) or _is_setup_wrapper_segment(family, argv, has_more_segments=has_more_segments):
                saw_only_pseudo = True
        candidate_lines = [line.strip() for line in segment.splitlines() if line.strip()]
        for candidate in candidate_lines:
            if candidate.startswith("#"):
                saw_only_pseudo = True
                continue
            try:
                argv = shlex.split(candidate, posix=True)
            except ValueError:
                argv = candidate.split()
            argv = _strip_leading_env_assignments(argv)
            if not argv:
                continue
            family = Path(argv[0]).name.lower().rstrip(".,:;")
            has_more_segments = index < len(segments) - 1
            if _is_pseudo_shell_family(family):
                saw_only_pseudo = True
                continue
            if _is_setup_wrapper_segment(family, argv, has_more_segments=has_more_segments):
                saw_only_pseudo = True
                continue
            return candidate
    return "" if saw_only_pseudo else command.strip()


def _unwrap_shell_wrapper(command: str) -> str:
    text = command.strip()
    if not text.startswith("[") or not text.endswith("]"):
        return command
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return command
    if not isinstance(parsed, list) or not parsed:
        return command
    items = [str(item) for item in parsed if isinstance(item, str)]
    if not items:
        return command
    first = items[0].strip().lower()
    if first in {"apply_patch"}:
        if len(items) >= 2 and items[1].strip():
            return f"{first} {items[1].strip()}"
        return first
    if first not in {"bash", "sh", "/bin/bash", "/bin/sh"}:
        return command
    for item in reversed(items[1:]):
        candidate = item.strip()
        if candidate and not candidate.startswith("-"):
            return candidate
    return command


def _shell_command_family_and_shape(command: str) -> tuple[str, str]:
    focused = _focused_shell_segment(command)
    try:
        argv = shlex.split(focused, posix=True)
    except ValueError:
        argv = focused.split()
    argv = _strip_leading_env_assignments(argv)
    if not argv:
        return "", ""
    family = Path(argv[0]).name.lower().rstrip(".,:;")
    if _is_pseudo_shell_family(family):
        return "", ""
    shape_parts: list[str] = [family]
    for arg in argv[1:8]:
        lowered = arg.lower()
        if lowered.startswith("-"):
            shape_parts.append(lowered)
        elif arg in {"{}", ";"}:
            shape_parts.append(arg)
        elif "*" in arg or "?" in arg or "[" in arg:
            shape_parts.append("GLOB")
        elif "/" in arg or "\\" in arg or lowered in {".", ".."}:
            shape_parts.append("PATH")
        elif re.fullmatch(r"\d+", arg):
            shape_parts.append("N")
        else:
            shape_parts.append("ARG")
    return family, " ".join(shape_parts)


def _shell_command_intent(command: str) -> str:
    focused = _focused_shell_segment(command)
    try:
        argv = shlex.split(focused, posix=True)
    except ValueError:
        argv = focused.split()
    argv = _strip_leading_env_assignments(argv)
    if not argv:
        return "unknown"
    family = Path(argv[0]).name.lower().rstrip(".,:;")
    if _is_pseudo_shell_family(family):
        return "unknown"
    lowered = " ".join(argv).lower()
    full_lowered = command.lower()
    if "apt-get install" in full_lowered or "apt install" in full_lowered:
        return "dependency-install"
    if "yum install" in full_lowered or "dnf install" in full_lowered or "apk add" in full_lowered:
        return "dependency-install"
    if "brew install" in full_lowered or "pip install" in full_lowered or "npm install" in full_lowered:
        return "dependency-install"
    if family == "timeout" and len(argv) >= 3:
        wrapped_command = " ".join(argv[2:])
        wrapped_intent = _shell_command_intent(wrapped_command)
        if wrapped_intent != "unknown":
            return wrapped_intent
    if "dpkg --configure -a" in full_lowered:
        return "dependency-recovery"
    if family == "pkill":
        if any(
            marker in full_lowered
            for marker in (
                "--version",
                "command -v",
                " which ",
                "which ",
                "dpkg -l",
                " ls /usr/lib/",
                "/usr/bin/r ",
            )
        ):
            return "dependency-probe"
    if family in {"which", "command"}:
        return "dependency-probe"
    if "command -v" in full_lowered and "--version" in full_lowered:
        return "dependency-probe"
    if "dpkg -l" in full_lowered and "grep" in full_lowered:
        return "dependency-probe"
    if family in {"pip", "pip3"} and " list" in f" {lowered} ":
        return "dependency-probe"
    if family == "psql" and "--help" in argv:
        return "dependency-probe"
    if family == "man" and len(argv) >= 2:
        return "dependency-probe"
    if family in {"apt-get", "apt", "yum", "dnf", "apk", "brew", "pip", "pip3", "npm", "pnpm", "yarn"}:
        if any(token in argv for token in {"install", "add"}):
            return "dependency-install"
    if family.startswith("python"):
        if re.search(r"\b(?:pytest|unittest)\b", lowered):
            return "python-test-command"
        if len(argv) >= 3 and argv[1] == "-c":
            return "python-inline-probe"
        if len(argv) >= 2 and argv[1].endswith(".py"):
            return "python-script-run"
        return "python-other"
    if family in {"rscript", "r"}:
        if any(token in full_lowered for token in ("test()", "testthat", "devtools::test", "tinytest")):
            return "r-test-command"
        return "r-other"
    if family == "make" and any(token in {item.lower() for item in argv[1:]} for token in {"test", "tests", "unit", "check"}):
        return "test-command"
    if family in {"runtests.py", "runtests.sh", "run_tests.sh"}:
        return "test-command"
    if family in {"pytest"}:
        return "test-command"
    if family in {"curl", "wget"} and any(token in {item.lower() for item in argv} for token in {"sed", "head", "tail"}):
        return "file-inspection"
    if family in {"curl", "wget"} and "grep" in {item.lower() for item in argv}:
        return "text-search"
    if family == "curl" and not any(token in argv for token in {"-o", "-O", "--output", "--remote-name"}):
        return "file-inspection"
    if family == "wget" and not any(token in argv for token in {"-O", "--output-document"}):
        return "file-inspection"
    if family in {"grep", "rg", "ripgrep"}:
        return "text-search"
    if family == "ps":
        return "process-inspection"
    if family in {"kill", "killall", "pkill"}:
        return "process-control"
    if family == "find":
        if "-exec" in argv and "grep" in {item.lower() for item in argv}:
            return "find-grep-search"
        return "file-discovery"
    if family == "sed":
        if "-n" in argv:
            return "file-inspection"
        if "-i" in argv:
            return "file-mutation"
    if family == "awk":
        if any("/" in token or "\\" in token for token in argv[1:]):
            return "file-inspection"
    if family in {"ls", "dir", "cat", "type", "head", "tail", "wc", "nl", "pwd", "file", "stat"}:
        return "file-inspection"
    if family in {"cp", "mv", "rm", "mkdir", "chmod", "apply_patch"}:
        return "file-mutation"
    if family == "git":
        return "git-inspection"
    return "other-shell"


def _example_payload(record: MessageRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dataset": record.dataset,
        "row_id": record.row_id,
        "message_index": record.message_index,
        "role": record.role,
        "kind": record.kind,
        "chars": record.content_chars,
        "excerpt": _message_excerpt(record.content),
    }
    if record.name:
        payload["name"] = record.name
    if record.tool_call_names:
        payload["tool_calls"] = list(record.tool_call_names)
    return payload


def _maybe_store_example(store: dict[str, list[dict[str, Any]]], key: str, payload: dict[str, Any], *, limit: int = 3) -> None:
    rows = store.setdefault(key, [])
    identity = (
        payload.get("dataset"),
        payload.get("row_id"),
        payload.get("message_index"),
        payload.get("role"),
        payload.get("excerpt"),
    )
    if any(
        (
            existing.get("dataset"),
            existing.get("row_id"),
            existing.get("message_index"),
            existing.get("role"),
            existing.get("excerpt"),
        )
        == identity
        for existing in rows
    ):
        return
    if len(rows) < limit:
        rows.append(payload)


def classify_message_themes(record: MessageRecord) -> list[str]:
    themes: list[str] = []
    lowered = record.content.lower()
    if record.kind != "message":
        return themes
    if record.role == "system" and record.content_chars >= 1200:
        themes.append("large-system-prompt")
    if record.role in {"assistant", "ai"} and record.content_chars >= 500 and not record.tool_call_names:
        if any(token in lowered for token in ("phase", "let me", "i'll", "we need to", "first", "start by", "plan")):
            themes.append("planning-monologue")
    if record.role in {"assistant", "ai"} and record.content_chars >= 700 and record.tool_call_names:
        themes.append("plan-plus-tool-call-blob")
    if record.role in {"tool", "user"} and record.error_class and record.content_chars >= 400:
        themes.append("large-failure-blob")
    if record.role in {"tool", "user"} and not record.error_class and record.content_chars >= 900:
        themes.append("large-context-blob")
    if record.role == "user" and any(name in lowered for name in ("read_file", "search_symbols", "run_test", "git_status", "git_diff", "run_shell", "list_files", "code_outline")):
        if "use " in lowered or "then use " in lowered or "reply with" in lowered:
            themes.append("explicit-tool-request")
    return themes


def _is_result_like_message(record: MessageRecord) -> bool:
    return record.kind == "message" and (record.role == "tool" or bool(record.name))


def _iter_dataset_rows(data_root: Path, dataset: str, max_rows: int | None) -> tuple[str, Iterable[dict[str, Any]]]:
    spec = trajectory_profile.DATASET_SPECS[dataset]
    root = data_root / dataset
    paths: list[Path] = []
    for pattern in spec["paths"]:
        paths.extend(sorted(root.glob(pattern)))
    adapter = str(spec["adapter"])
    columns = ["trajectory", "messages", "messages_json", "instance_id", "trajectory_id", "repo"]
    if adapter == "swe_agent":
        columns = ["trajectory", "instance_id", "target"]
    elif adapter == "smith":
        columns = ["messages", "instance_id", "traj_id"]
    elif adapter == "trace_commons":
        columns = ["messages", "session_id", "harness", "prompt", "num_tool_calls"]
    elif adapter == "thoughtworks":
        columns = ["messages", "messages_json", "agent_framework", "source_dataset", "session_id", "source_id"]
    elif adapter == "terminalbench":
        columns = ["steps", "trial_id", "trial_name", "task_name", "agent", "model"]
    rows = trajectory_error_profile._iter_projected_parquet_rows(paths, columns=columns)
    return adapter, trajectory_profile._iter_rows_with_trajectory_content(adapter, rows, max_rows=max_rows)


def summarize_dataset(dataset: str, adapter: str, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows_profiled = 0
    messages_profiled = 0
    tool_call_events = 0
    message_tool_call_records = 0
    role_counts: Counter[str] = Counter()
    message_theme_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    shell_error_counts: Counter[str] = Counter()
    tool_category_counts: Counter[str] = Counter()
    shell_command_counts: Counter[str] = Counter()
    shell_command_shape_counts: Counter[str] = Counter()
    shell_command_intent_counts: Counter[str] = Counter()
    repeated_tool_counts: Counter[str] = Counter()
    row_pattern_counts: Counter[str] = Counter()
    message_theme_examples: dict[str, list[dict[str, Any]]] = {}
    shell_command_examples: dict[str, list[dict[str, Any]]] = {}
    shell_command_intent_examples: dict[str, list[dict[str, Any]]] = {}
    error_examples: dict[str, list[dict[str, Any]]] = {}
    row_pattern_examples: dict[str, list[dict[str, Any]]] = {}
    rows_with_edit = 0
    rows_with_test = 0

    for row_number, row in enumerate(rows):
        rows_profiled += 1
        row_id = _row_id(row, row_number)
        events = trajectory_profile._extract_events(adapter, row)
        metrics = trajectory_profile._trajectory_metrics(events)
        tool_call_events += int(metrics["tool_calls"])
        row_tool_names = [name for name in metrics["tool_names"] if name]
        tool_category_counts.update(metrics["categories"])
        for tool_name, run_length in metrics["repeated_tools"]:
            repeated_tool_counts[f"{tool_name} x{run_length}"] += 1
        if metrics["has_edit"]:
            rows_with_edit += 1
        if metrics["has_test"]:
            rows_with_test += 1
        if metrics["context_loop"]:
            row_pattern_counts["context-loop-row"] += 1
            _maybe_store_example(
                row_pattern_examples,
                "context-loop-row",
                {
                    "dataset": dataset,
                    "row_id": row_id,
                    "tool_names": row_tool_names[:8],
                },
            )
        if metrics["mechanical_turn_candidate"]:
            row_pattern_counts["mechanical-turn-row"] += 1
            _maybe_store_example(
                row_pattern_examples,
                "mechanical-turn-row",
                {
                    "dataset": dataset,
                    "row_id": row_id,
                    "tool_names": row_tool_names[:8],
                    "category": metrics["mechanical_turn_category"],
                },
            )
        if metrics["has_edit"] and not metrics["has_search_or_read_before_edit"]:
            row_pattern_counts["edit-without-context-row"] += 1
            _maybe_store_example(
                row_pattern_examples,
                "edit-without-context-row",
                {
                    "dataset": dataset,
                    "row_id": row_id,
                    "tool_names": row_tool_names[:8],
                    "first_edit_index": metrics["first_edit_index"],
                },
            )
        if metrics["has_edit"] and not metrics["has_test_after_edit"]:
            row_pattern_counts["edit-without-test-row"] += 1
            _maybe_store_example(
                row_pattern_examples,
                "edit-without-test-row",
                {
                    "dataset": dataset,
                    "row_id": row_id,
                    "tool_names": row_tool_names[:8],
                    "first_edit_index": metrics["first_edit_index"],
                },
            )

        for record in extract_message_records(dataset, adapter, row, row_number):
            if record.kind == "message":
                messages_profiled += 1
                role_counts[record.role] += 1
                for theme in classify_message_themes(record):
                    message_theme_counts[theme] += 1
                    _maybe_store_example(message_theme_examples, theme, _example_payload(record))
            else:
                message_tool_call_records += 1
                command = _shell_command_from_record(record)
                if command:
                    family, shape = _shell_command_family_and_shape(command)
                    intent = _shell_command_intent(command)
                    if family:
                        shell_command_counts[family] += 1
                        _maybe_store_example(shell_command_examples, family, {**_example_payload(record), "command": _message_excerpt(command, limit=220), "shape": shape})
                    if shape:
                        shell_command_shape_counts[shape] += 1
                    if intent:
                        shell_command_intent_counts[intent] += 1
                        _maybe_store_example(shell_command_intent_examples, intent, {**_example_payload(record), "command": _message_excerpt(command, limit=220), "shape": shape})
            if record.error_class and _is_result_like_message(record):
                error_counts[record.error_class] += 1
                _maybe_store_example(error_examples, record.error_class, _example_payload(record))
            if record.shell_error and _is_result_like_message(record):
                shell_error_counts[record.shell_error] += 1

    summary = {
        "dataset": dataset,
        "rows_profiled": rows_profiled,
        "messages_profiled": messages_profiled,
        "tool_call_events": tool_call_events,
        "message_tool_call_records": message_tool_call_records,
        "avg_tool_calls": round(tool_call_events / rows_profiled, 2) if rows_profiled else 0.0,
        "rows_with_edit_pct": round((rows_with_edit / rows_profiled) * 100, 2) if rows_profiled else 0.0,
        "rows_with_test_pct": round((rows_with_test / rows_profiled) * 100, 2) if rows_profiled else 0.0,
        "context_loop_rows_pct": round((row_pattern_counts.get("context-loop-row", 0) / rows_profiled) * 100, 2) if rows_profiled else 0.0,
        "edit_without_prior_context_pct": round((row_pattern_counts.get("edit-without-context-row", 0) / rows_with_edit) * 100, 2) if rows_with_edit else 0.0,
        "edit_without_later_test_pct": round((row_pattern_counts.get("edit-without-test-row", 0) / rows_with_edit) * 100, 2) if rows_with_edit else 0.0,
        "mechanical_turn_candidates_pct": round((row_pattern_counts.get("mechanical-turn-row", 0) / rows_profiled) * 100, 2) if rows_profiled else 0.0,
        "message_role_counts": dict(role_counts.most_common()),
        "message_theme_counts": dict(message_theme_counts.most_common()),
        "message_theme_examples": message_theme_examples,
        "error_counts": dict(error_counts.most_common()),
        "error_examples": error_examples,
        "shell_error_counts": dict(shell_error_counts.most_common()),
        "row_pattern_counts": dict(row_pattern_counts.most_common()),
        "row_pattern_examples": row_pattern_examples,
        "tool_category_counts": dict(tool_category_counts.most_common()),
        "shell_command_counts": dict(shell_command_counts.most_common(30)),
        "shell_command_shape_counts": dict(shell_command_shape_counts.most_common(30)),
        "shell_command_intent_counts": dict(shell_command_intent_counts.most_common(30)),
        "shell_command_examples": shell_command_examples,
        "shell_command_intent_examples": shell_command_intent_examples,
        "top_repeated_loops": [{"loop": loop, "count": count} for loop, count in repeated_tool_counts.most_common(20)],
        "recommendations": [],
    }
    summary["recommendations"] = [item.__dict__ for item in trajectory_profile._heuristic_recommendations(summary)]
    return summary


def _attach_manifest(summary: dict[str, Any], data_root: Path) -> None:
    dataset = str(summary.get("dataset") or "")
    if not dataset:
        return
    manifest = trajectory_dataset_fetch.read_dataset_manifest(data_root, dataset)
    if not manifest:
        return
    summary["source_manifest"] = {
        "repo_id": manifest.get("repo_id"),
        "requested_revision": manifest.get("requested_revision"),
        "resolved_revision": manifest.get("resolved_revision"),
        "downloaded_at": manifest.get("downloaded_at"),
        "file_count": manifest.get("file_count"),
        "manifest_path": trajectory_dataset_fetch.dataset_manifest_path(data_root, dataset).as_posix(),
    }


def _reference_row_pattern_counts(summary: dict[str, Any]) -> dict[str, int]:
    reference_trajectory = summary.get("reference_trajectory_metrics")
    if not isinstance(reference_trajectory, dict):
        return dict(summary.get("row_pattern_counts") or {})
    rows_profiled = int(reference_trajectory.get("rows_profiled") or 0)
    rows_with_edit = round(rows_profiled * (float(reference_trajectory.get("rows_with_edit_pct") or 0.0) / 100.0))
    return {
        "mechanical-turn-row": round(rows_profiled * (float(reference_trajectory.get("mechanical_turn_candidates_pct") or 0.0) / 100.0)),
        "context-loop-row": round(rows_profiled * (float(reference_trajectory.get("context_loop_rows_pct") or 0.0) / 100.0)),
        "edit-without-context-row": round(rows_with_edit * (float(reference_trajectory.get("edit_without_prior_context_pct") or 0.0) / 100.0)),
        "edit-without-test-row": round(rows_with_edit * (float(reference_trajectory.get("edit_without_later_test_pct") or 0.0) / 100.0)),
    }


def _fix_evidence_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    grouped_counts: Counter[str] = Counter()
    grouped_datasets: dict[str, set[str]] = defaultdict(set)
    dataset_name = str(summary.get("dataset") or "unknown")
    row_pattern_counts = _reference_row_pattern_counts(summary)
    for pattern_name, fix_ids in ROW_PATTERN_FIX_MAP.items():
        evidence_count = int(row_pattern_counts.get(pattern_name, 0))
        if evidence_count <= 0:
            continue
        for rec_id in fix_ids:
            grouped_counts[rec_id] += evidence_count
            grouped_datasets[rec_id].add(dataset_name)
    message_theme_counts = summary.get("message_theme_counts")
    if isinstance(message_theme_counts, dict):
        for theme_name, fix_ids in MESSAGE_THEME_FIX_MAP.items():
            evidence_count = int(message_theme_counts.get(theme_name, 0) or 0)
            if evidence_count <= 0:
                continue
            for rec_id in fix_ids:
                grouped_counts[rec_id] += evidence_count
                grouped_datasets[rec_id].add(dataset_name)

    trajectory_recommendations = summary.get("recommendations", [])
    reference_trajectory = summary.get("reference_trajectory_metrics")
    if isinstance(reference_trajectory, dict) and isinstance(reference_trajectory.get("recommendations"), list):
        trajectory_recommendations = reference_trajectory.get("recommendations", [])
    for raw in trajectory_recommendations:
        if not isinstance(raw, dict):
            continue
        rec_id = str(raw.get("id") or "")
        if not rec_id:
            continue
        if rec_id not in grouped_counts:
            grouped_counts[rec_id] += 1
        grouped_datasets[rec_id].add(dataset_name)
    error_counts: dict[str, Any] = dict(summary.get("error_counts") or {})
    reference_error = summary.get("reference_error_metrics")
    if isinstance(reference_error, dict) and isinstance(reference_error.get("error_counts"), dict):
        error_counts = dict(reference_error.get("error_counts") or {})
    for error_class, count in error_counts.items():
        rec_id = trajectory_error_profile.RECOMMENDATIONS.get(str(error_class))
        if not rec_id:
            continue
        grouped_counts[rec_id] += int(count)
        grouped_datasets[rec_id].add(dataset_name)
    shell_error_counts: dict[str, Any] = dict(summary.get("shell_error_counts") or {})
    if isinstance(reference_error, dict) and isinstance(reference_error.get("shell_error_counts"), dict):
        shell_error_counts = dict(reference_error.get("shell_error_counts") or {})
    for shell_error, count in shell_error_counts.items():
        for rec_id in SHELL_ERROR_FIX_MAP.get(str(shell_error), ()):
            grouped_counts[rec_id] += int(count)
            grouped_datasets[rec_id].add(dataset_name)
    rows: list[dict[str, Any]] = []
    for rec_id, evidence_count in grouped_counts.most_common():
        fix = PRODUCT_FIXES.get(rec_id, {})
        rows.append(
            {
                "id": rec_id,
                "evidence_count": evidence_count,
                "datasets": sorted(grouped_datasets[rec_id]),
                "status": str(fix.get("status") or "unknown"),
                "summary": str(fix.get("summary") or ""),
                "references": list(fix.get("references") or []),
                "next_gap": str(fix.get("next_gap") or ""),
            }
        )
    return rows


def _portfolio(payload: dict[str, Any]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    dataset_map: dict[str, set[str]] = defaultdict(set)
    for summary in payload.get("datasets", []):
        for row in _fix_evidence_rows(summary):
            counts[row["id"]] += int(row["evidence_count"])
            dataset_map[row["id"]].update(row["datasets"])
    rows: list[dict[str, Any]] = []
    for rec_id, evidence_count in counts.most_common():
        fix = PRODUCT_FIXES.get(rec_id, {})
        rows.append(
            {
                "id": rec_id,
                "evidence_count": evidence_count,
                "datasets": sorted(dataset_map[rec_id]),
                "status": str(fix.get("status") or "unknown"),
                "summary": str(fix.get("summary") or ""),
                "references": list(fix.get("references") or []),
                "next_gap": str(fix.get("next_gap") or ""),
            }
        )
    return rows


def _load_reference_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("datasets")
    if not isinstance(rows, list):
        return {}
    mapped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("dataset") or "")
        if name:
            mapped[name] = row
    return mapped


def _merge_reference_metrics(
    summary: dict[str, Any],
    *,
    reference_trajectory: dict[str, dict[str, Any]],
    reference_error: dict[str, dict[str, Any]],
) -> None:
    dataset = str(summary.get("dataset") or "")
    trajectory_row = reference_trajectory.get(dataset)
    if trajectory_row:
        summary["reference_trajectory_metrics"] = {
            key: trajectory_row.get(key)
            for key in (
                "rows_profiled",
                "avg_tool_calls",
                "rows_with_edit_pct",
                "rows_with_test_pct",
                "context_loop_rows_pct",
                "edit_without_prior_context_pct",
                "edit_without_later_test_pct",
                "mechanical_turn_candidates_pct",
                "top_repeated_loops",
                "recommendations",
            )
        }
    error_row = reference_error.get(dataset)
    if error_row:
        summary["reference_error_metrics"] = {
            key: error_row.get(key)
            for key in (
                "rows_profiled",
                "result_events",
                "error_counts",
                "top_shell_errors",
                "repeated_error_loops",
                "recommendations",
            )
        }


def build_report(
    data_root: Path,
    datasets: list[str],
    max_rows: int | None,
    *,
    reference_trajectory_path: Path | None = None,
    reference_error_path: Path | None = None,
) -> dict[str, Any]:
    reference_trajectory = _load_reference_map(reference_trajectory_path)
    reference_error = _load_reference_map(reference_error_path)
    summaries: list[dict[str, Any]] = []
    for dataset in datasets:
        adapter, rows = _iter_dataset_rows(data_root, dataset, max_rows)
        summary = summarize_dataset(dataset, adapter, rows)
        _attach_manifest(summary, data_root)
        _merge_reference_metrics(summary, reference_trajectory=reference_trajectory, reference_error=reference_error)
        summaries.append(summary)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": data_root.as_posix(),
        "max_rows_per_dataset": max_rows,
        "datasets": summaries,
        "reference_trajectory_profile": reference_trajectory_path.as_posix() if reference_trajectory_path and reference_trajectory_path.exists() else None,
        "reference_error_profile": reference_error_path.as_posix() if reference_error_path and reference_error_path.exists() else None,
    }
    payload["portfolio_fix_coverage"] = _portfolio(payload)
    return payload


def format_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Trajectory Evidence Report")
    lines.append("")
    lines.append("This report scans raw messages from the configured local trajectory datasets, then groups message-level token waste, failure themes, and row-level stuck patterns.")
    lines.append("")
    lines.append(f"Generated: `{payload.get('generated_at', '')}`")
    lines.append(f"Data root: `{payload.get('data_root', '')}`")
    lines.append(f"Max rows per dataset: `{payload.get('max_rows_per_dataset')}`")
    if payload.get("reference_trajectory_profile"):
        lines.append(f"Reference trajectory profile: `{payload.get('reference_trajectory_profile')}`")
    if payload.get("reference_error_profile"):
        lines.append(f"Reference error profile: `{payload.get('reference_error_profile')}`")
    if payload.get("max_rows_per_dataset") is not None:
        lines.append("Citation note: example citations come from the bounded sample above, while full-corpus percentages and error totals are merged from the reference profile JSON files when present.")
    lines.append("")
    lines.append("## Portfolio Fix Coverage")
    lines.append("")
    lines.append("| Fix | Evidence Count | Datasets | Status | References |")
    lines.append("|---|---:|---|---|---|")
    for row in payload.get("portfolio_fix_coverage", []):
        refs = ", ".join(row.get("references", []))
        datasets = ", ".join(row.get("datasets", []))
        lines.append(f"| `{row['id']}` | {row['evidence_count']} | {datasets} | `{row['status']}` | {refs} |")
    lines.append("")
    for summary in payload.get("datasets", []):
        lines.append(f"## {summary.get('dataset', 'unknown')}")
        lines.append("")
        lines.append(f"- Rows profiled: `{summary.get('rows_profiled', 0)}`")
        lines.append(f"- Messages profiled: `{summary.get('messages_profiled', 0)}`")
        lines.append(f"- Tool-call events: `{summary.get('tool_call_events', 0)}`")
        lines.append(f"- Message-extracted tool-call records: `{summary.get('message_tool_call_records', 0)}`")
        lines.append(f"- Avg tool calls: `{summary.get('avg_tool_calls', 0)}`")
        manifest = summary.get("source_manifest")
        if isinstance(manifest, dict):
            lines.append(
                "- Source manifest: "
                + f"`{manifest.get('repo_id', '')}` "
                + f"revision=`{manifest.get('resolved_revision') or manifest.get('requested_revision') or '-'}` "
                + f"files=`{manifest.get('file_count', 0)}`"
            )
        lines.append(f"- Context-loop rows: `{summary.get('context_loop_rows_pct', 0)}%`")
        lines.append(f"- Edit without prior context: `{summary.get('edit_without_prior_context_pct', 0)}%`")
        lines.append(f"- Edit without later test: `{summary.get('edit_without_later_test_pct', 0)}%`")
        lines.append(f"- Mechanical-turn candidates: `{summary.get('mechanical_turn_candidates_pct', 0)}%`")
        reference_trajectory = summary.get("reference_trajectory_metrics")
        if isinstance(reference_trajectory, dict):
            lines.append(
                "- Full-corpus reference:"
                + f" rows=`{reference_trajectory.get('rows_profiled')}`"
                + f", avg_tool_calls=`{reference_trajectory.get('avg_tool_calls')}`"
                + f", context_loops=`{reference_trajectory.get('context_loop_rows_pct')}%`"
                + f", edit_without_context=`{reference_trajectory.get('edit_without_prior_context_pct')}%`"
                + f", edit_without_test=`{reference_trajectory.get('edit_without_later_test_pct')}%`"
                + f", mechanical=`{reference_trajectory.get('mechanical_turn_candidates_pct')}%`"
            )
        reference_error = summary.get("reference_error_metrics")
        if isinstance(reference_error, dict):
            top_errors = dict(reference_error.get("error_counts") or {})
            if top_errors:
                ordered = sorted(top_errors.items(), key=lambda item: (-int(item[1]), item[0]))[:5]
                lines.append("- Full-corpus top errors: " + ", ".join(f"`{name}`={count}" for name, count in ordered))
        lines.append("")
        lines.append("### Message Themes")
        lines.append("")
        theme_counts = dict(summary.get("message_theme_counts") or {})
        if not theme_counts:
            lines.append("(none)")
        else:
            lines.append("| Theme | Count | Meaning |")
            lines.append("|---|---:|---|")
            for theme, count in sorted(theme_counts.items(), key=lambda item: (-int(item[1]), item[0]))[:10]:
                meta = MESSAGE_THEME_META.get(theme, {})
                lines.append(f"| `{theme}` | {count} | {meta.get('description', '')} |")
        lines.append("")
        lines.append("### Error Themes")
        lines.append("")
        error_counts = dict(summary.get("error_counts") or {})
        if not error_counts:
            lines.append("(none)")
        else:
            lines.append("| Error | Count | Fix |")
            lines.append("|---|---:|---|")
            for error_name, count in sorted(error_counts.items(), key=lambda item: (-int(item[1]), item[0]))[:10]:
                fix = trajectory_error_profile.RECOMMENDATIONS.get(error_name, "")
                lines.append(f"| `{error_name}` | {count} | `{fix}` |")
        lines.append("")
        lines.append("### Row Patterns")
        lines.append("")
        row_patterns = dict(summary.get("row_pattern_counts") or {})
        if not row_patterns:
            lines.append("(none)")
        else:
            lines.append("| Pattern | Count | Meaning |")
            lines.append("|---|---:|---|")
            for pattern_name, count in sorted(row_patterns.items(), key=lambda item: (-int(item[1]), item[0])):
                meta = ROW_PATTERN_META.get(pattern_name, {})
                lines.append(f"| `{pattern_name}` | {count} | {meta.get('description', '')} |")
        lines.append("")
        lines.append("### Shell Command Shapes")
        lines.append("")
        shell_commands = dict(summary.get("shell_command_counts") or {})
        shell_shapes = dict(summary.get("shell_command_shape_counts") or {})
        shell_intents = dict(summary.get("shell_command_intent_counts") or {})
        if not shell_commands and not shell_shapes and not shell_intents:
            lines.append("(none)")
        else:
            if shell_commands:
                ordered = sorted(shell_commands.items(), key=lambda item: (-int(item[1]), item[0]))[:10]
                lines.append("- Top families: " + ", ".join(f"`{name}`={count}" for name, count in ordered))
            if shell_intents:
                ordered = sorted(shell_intents.items(), key=lambda item: (-int(item[1]), item[0]))[:10]
                lines.append("- Top intents: " + ", ".join(f"`{name}`={count}" for name, count in ordered))
            if shell_shapes:
                lines.append("")
                lines.append("| Shape | Count |")
                lines.append("|---|---:|")
                for shape, count in sorted(shell_shapes.items(), key=lambda item: (-int(item[1]), item[0]))[:10]:
                    lines.append(f"| `{shape}` | {count} |")
        lines.append("")
        lines.append("### Example Citations")
        lines.append("")
        emitted = False
        for label, examples in list((summary.get("message_theme_examples") or {}).items())[:3]:
            if not examples:
                continue
            emitted = True
            lines.append(f"- `{label}`:")
            for example in examples:
                lines.append(
                    "  "
                    + f"{example.get('dataset')} / {example.get('row_id')} / msg {example.get('message_index')} / {example.get('role')}: "
                    + f"{example.get('excerpt')}"
                )
        for label, examples in list((summary.get("error_examples") or {}).items())[:3]:
            if not examples:
                continue
            emitted = True
            lines.append(f"- `error:{label}`:")
            for example in examples:
                lines.append(
                    "  "
                    + f"{example.get('dataset')} / {example.get('row_id')} / msg {example.get('message_index')} / {example.get('role')}: "
                    + f"{example.get('excerpt')}"
                )
        shell_intent_examples = dict(summary.get("shell_command_intent_examples") or {})
        ranked_shell_intents = sorted(
            shell_intent_examples.items(),
            key=lambda item: (-int(shell_intents.get(item[0], 0)), item[0]),
        )[:3]
        for label, examples in ranked_shell_intents:
            if not examples:
                continue
            emitted = True
            lines.append(f"- `shell-intent:{label}`:")
            for example in examples:
                lines.append(
                    "  "
                    + f"{example.get('dataset')} / {example.get('row_id')} / msg {example.get('message_index')} / {example.get('role')}: "
                    + f"{example.get('command')} [{example.get('shape')}]"
                )
        if not emitted:
            lines.append("(no examples)")
        lines.append("")
        lines.append("### Fix Coverage")
        lines.append("")
        fixes = _fix_evidence_rows(summary)
        if not fixes:
            lines.append("(none)")
        else:
            for row in fixes[:8]:
                lines.append(
                    f"- `{row['id']}` [{row['status']}]: {row['summary']} References: {', '.join(row['references'])}. Gap: {row['next_gap']}"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan every local trajectory message and emit grouped evidence tied to current product fixes.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(trajectory_profile.DATASET_SPECS.keys()),
        choices=sorted(trajectory_profile.DATASET_SPECS.keys()),
        help="Datasets to scan.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root directory containing downloaded trajectory datasets.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional per-dataset row cap for faster sampling.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_OUTPUT, help="Where to write raw JSON.")
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MARKDOWN_OUTPUT, help="Where to write the markdown report.")
    parser.add_argument(
        "--reference-trajectory-profile",
        type=Path,
        default=DEFAULT_REFERENCE_TRAJECTORY_PROFILE,
        help="Optional full-corpus trajectory-profile JSON to merge into the report.",
    )
    parser.add_argument(
        "--reference-error-profile",
        type=Path,
        default=DEFAULT_REFERENCE_ERROR_PROFILE,
        help="Optional full-corpus trajectory-error-profile JSON to merge into the report.",
    )
    args = parser.parse_args(argv)

    payload = build_report(
        args.data_root,
        args.datasets,
        args.max_rows,
        reference_trajectory_path=args.reference_trajectory_profile,
        reference_error_path=args.reference_error_profile,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown = format_markdown(payload)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")

    print(f"Wrote JSON: {args.output_json}")
    print(f"Wrote Markdown: {args.output_md}")
    for summary in payload.get("datasets", []):
        print(
            f"{summary.get('dataset')}: rows={summary.get('rows_profiled')} messages={summary.get('messages_profiled')} "
            f"context_loops={summary.get('context_loop_rows_pct')}% mechanical={summary.get('mechanical_turn_candidates_pct')}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
