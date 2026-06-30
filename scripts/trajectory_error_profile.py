from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    pq = None

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import trajectory_profile


DEFAULT_OUTPUT = Path("scratch") / "external" / "datasets" / "trajectory-error-profile.json"

ERROR_PATTERNS: dict[str, re.Pattern[str]] = {
    "missing_dependency": re.compile(r"ModuleNotFoundError:\s+No module named ['\"]([^'\"]+)['\"]|No module named ['\"]([^'\"]+)['\"]", re.I),
    "import_error": re.compile(r"ImportError|cannot import name", re.I),
    "syntax_error": re.compile(r"SyntaxError|IndentationError|unexpected EOF|invalid syntax", re.I),
    "test_assertion": re.compile(r"\bAssertionError\b|(?m:^\s*FAILED\s+\S+|^\s*FAIL:\s|^\s*E\s+assert\b)"),
    "invalid_args": re.compile(
        r"unrecognized arguments|invalid option|unknown option|error: argument|"
        r"missing required|the following arguments are required|no such command|too few arguments",
        re.I,
    ),
    "command_not_found": re.compile(r"command not found|not recognized as (?:an internal|the name)", re.I),
    "path_missing": re.compile(r"No such file or directory|cannot access|FileNotFoundError|Path does not exist|path .* does not exist", re.I),
    "cwd_git": re.compile(r"cd: .*No such file|outside workspace|escapes the workspace|not inside a git repository", re.I),
    "timeout": re.compile(r"\btimed out\b|\bdeadline exceeded\b|\bexceeded\b[^\n]{0,120}\btimeout\b", re.I),
    "permission": re.compile(r"Permission denied|access is denied|Operation not permitted", re.I),
    "patch_apply": re.compile(r"patch failed|does not apply|git apply.*failed|hunk FAILED|error: patch", re.I),
}

RECOMMENDATIONS = {
    "test_assertion": "diagnose-test-failure",
    "missing_dependency": "dependency-or-import-guard",
    "import_error": "dependency-or-import-guard",
    "syntax_error": "syntax-repair-gate",
    "invalid_args": "validated-cli-preflight",
    "command_not_found": "validated-cli-preflight",
    "path_missing": "path-repair-guard",
    "cwd_git": "path-repair-guard",
    "timeout": "bounded-command-validation",
    "permission": "fail-closed-permission",
    "patch_apply": "structured-edit-preflight",
}

ERROR_HINTS: dict[str, tuple[str, ...]] = {
    "missing_dependency": ("modulenotfounderror", "no module named"),
    "import_error": ("importerror", "cannot import name"),
    "syntax_error": ("syntaxerror", "indentationerror", "unexpected eof", "invalid syntax"),
    "test_assertion": ("assertionerror", "failed ", "fail:", "e assert"),
    "invalid_args": (
        "unrecognized arguments",
        "invalid option",
        "unknown option",
        "error: argument",
        "missing required",
        "the following arguments are required",
        "no such command",
        "too few arguments",
    ),
    "command_not_found": ("command not found", "not recognized as an internal", "not recognized as the name"),
    "path_missing": ("no such file or directory", "cannot access", "filenotfounderror", "path does not exist"),
    "cwd_git": ("cd:", "outside workspace", "escapes the workspace", "not inside a git repository"),
    "timeout": ("timed out", "deadline exceeded", "exceeded timeout"),
    "permission": ("permission denied", "access is denied", "operation not permitted"),
    "patch_apply": ("patch failed", "does not apply", "git apply", "hunk failed", "error: patch"),
}

SHELL_ERROR_PATTERNS: dict[str, re.Pattern[str]] = {
    "bash_unexpected_token": re.compile(r"syntax error near unexpected token [`'\"]?([^`'\"\n]+)", re.I),
    "bash_unexpected_eof": re.compile(r"syntax error: unexpected end of file|unexpected EOF while looking for matching", re.I),
    "unmatched_quote": re.compile(r"unterminated quoted string|No closing quotation|unexpected EOF while looking for matching [`'\"]", re.I),
    "command_not_found": re.compile(r"(?m)(?:^|\s)(?:bash: )?([A-Za-z0-9_.:/-]+): command not found|not recognized as (?:an internal|the name)", re.I),
    "missing_file_or_dir": re.compile(r"No such file or directory|cannot access|FileNotFoundError|Path does not exist", re.I),
    "unrecognized_argument": re.compile(r"unrecognized arguments?: ([^\n]+)|unknown option:?\s*([^\n]+)|invalid option --?\s*([^\n]+)", re.I),
    "missing_operand": re.compile(r"missing (?:file )?operand|missing required", re.I),
    "permission_denied": re.compile(r"Permission denied|access is denied|Operation not permitted", re.I),
}


def _agent_race_content_text(value: Any) -> str:
    if isinstance(value, list):
        text_parts: list[str] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text)
                continue
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                text_parts.append(content)
        if text_parts:
            return "\n".join(text_parts)
    return trajectory_profile._content_text(value)


def _extract_result_events(adapter: str, row: dict[str, Any]) -> list[trajectory_profile.Event]:
    events: list[trajectory_profile.Event] = []
    if adapter == "agent_race":
        raw_events = trajectory_profile._deserialize_possible_json(row.get("events"))
        if not isinstance(raw_events, list):
            return events
        tool_names_by_id: dict[str, str] = {}
        for raw_event in raw_events:
            if not isinstance(raw_event, dict):
                continue
            message = raw_event.get("message")
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").lower()
            content = message.get("content")
            if role == "assistant":
                items = content if isinstance(content, list) else []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    item_type = str(item.get("type") or "").lower()
                    if item_type not in {"tool_use", "toolcall"}:
                        continue
                    name = trajectory_profile._normalize_tool_name(str(item.get("name") or ""))
                    tool_id = str(item.get("id") or "")
                    if tool_id and name:
                        tool_names_by_id[tool_id] = name
                continue
            if role == "toolresult":
                raw_name = str(message.get("toolName") or "")
                name = trajectory_profile._normalize_tool_name(raw_name)
                tool_call_id = str(message.get("toolCallId") or "")
                if not name:
                    name = tool_names_by_id.get(tool_call_id, "tool")
                content_text = _agent_race_content_text(content)
                events.append(
                    trajectory_profile.Event(
                        role="tool",
                        kind="tool_result",
                        name=name,
                        category=trajectory_profile._tool_category(name, content_text),
                        content=content_text,
                    )
                )
                continue
            if role != "user" or not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                if str(item.get("type") or "").lower() != "tool_result":
                    continue
                tool_call_id = str(item.get("tool_use_id") or "")
                name = tool_names_by_id.get(tool_call_id, "tool")
                content_text = _agent_race_content_text(item.get("content"))
                events.append(
                    trajectory_profile.Event(
                        role="tool",
                        kind="tool_result",
                        name=name,
                        category=trajectory_profile._tool_category(name, content_text),
                        content=content_text,
                    )
                )
        return events
    if adapter == "openhands":
        messages = trajectory_profile._openhands_messages(row)
        pending_tool_name = ""
        tool_names_by_id: dict[str, str] = {}
        for message in messages if isinstance(messages, list) else []:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).lower()
            if role == "assistant":
                pending_tool_name = ""
                tool_names_by_id.update(trajectory_profile._tool_call_name_by_id(message))
                tool_calls = trajectory_profile._message_tool_calls(message)
                if isinstance(tool_calls, list) and len(tool_calls) == 1 and isinstance(tool_calls[0], dict):
                    function = tool_calls[0].get("function")
                    raw_name = ""
                    if isinstance(function, dict):
                        raw_name = str(function.get("name") or "")
                    if not raw_name:
                        raw_name = str(tool_calls[0].get("name") or "")
                    pending_tool_name = trajectory_profile._normalize_tool_name(raw_name)
                else:
                    assistant_content = trajectory_profile._openhands_message_content(message)
                    if not assistant_content.strip():
                        continue
                    inferred = trajectory_profile._infer_tool_name_from_text(assistant_content)
                    pending_tool_name = inferred or ""
                continue
            if role != "tool":
                continue
            raw_name = str(message.get("name") or "")
            name = trajectory_profile._normalize_tool_name(raw_name)
            if trajectory_profile._placeholder_tool_name(raw_name):
                tool_call_id = str(message.get("tool_call_id") or "")
                name = tool_names_by_id.get(tool_call_id) or pending_tool_name or name or "tool"
            content = str(message.get("content") or "")
            events.append(
                trajectory_profile.Event(
                    role="tool",
                    kind="tool_result",
                    name=name,
                    category=trajectory_profile._tool_category(name, content),
                    content=content,
                )
            )
            pending_tool_name = ""
        return events
    if adapter == "cc_bench":
        return _extract_result_events("openhands", row)
    if adapter == "trace_commons":
        if trajectory_profile._openhands_messages(row):
            return _extract_result_events("openhands", row)
        fallback_row = dict(row)
        fallback_row["messages"] = trajectory_profile._trace_commons_fallback_messages(row)
        return _extract_result_events("openhands", fallback_row)
    if adapter == "thoughtworks":
        effective_adapter = trajectory_profile._thoughtworks_row_adapter(row)
        if effective_adapter == "openhands":
            return _extract_result_events("openhands", row)
        return _extract_result_events(
            "smith",
            {"messages": trajectory_profile._first_non_empty_deserialized(row.get("messages"), row.get("messages_json"))},
        )
    if adapter == "swe_agent":
        previous_ai = ""
        for message in row.get("trajectory") or []:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").lower()
            text = str(message.get("text") or "")
            if role in {"ai", "assistant"}:
                previous_ai = text
                continue
            if role != "user" or not previous_ai:
                continue
            name = trajectory_profile._infer_tool_name_from_text(previous_ai) or "observation"
            events.append(
                trajectory_profile.Event(
                    role="user",
                    kind="observation",
                    name=name,
                    category=trajectory_profile._tool_category(name, text),
                    content=text,
                )
            )
            previous_ai = ""
        return events
    if adapter == "smith":
        messages = trajectory_profile._first_non_empty_deserialized(row.get("messages"), row.get("messages_json"))
        if not isinstance(messages, list):
            return events
        previous_ai = ""
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").lower()
            content = trajectory_profile._content_text(message.get("content"))
            if role == "assistant":
                previous_ai = content
                continue
            if role == "user" and previous_ai:
                name = trajectory_profile._infer_tool_name_from_text(previous_ai)
                if name and content.strip():
                    events.append(
                        trajectory_profile.Event(
                            role="user",
                            kind="observation",
                            name=name,
                            category=trajectory_profile._tool_category(name, content),
                            content=content,
                        )
                    )
                previous_ai = ""
                continue
            if role != "tool" and not message.get("tool_call_id"):
                previous_ai = ""
                continue
            name = str(message.get("name") or "tool")
            events.append(
                trajectory_profile.Event(
                    role=role or "tool",
                    kind="tool_result",
                    name=name,
                    category=trajectory_profile._tool_category(name, content),
                    content=content,
                )
            )
            previous_ai = ""
        return events
    if adapter == "terminalbench":
        steps = trajectory_profile._deserialize_possible_json(row.get("steps"))
        if not isinstance(steps, list):
            return events
        for step in steps:
            if not isinstance(step, dict):
                continue
            if str(step.get("src") or "").lower() != "agent":
                continue
            tools = step.get("tools")
            observation = trajectory_profile._content_text(step.get("obs"))
            if not isinstance(tools, list) or not observation.strip():
                continue
            first_tool = next((tool for tool in tools if isinstance(tool, dict) and str(tool.get("fn") or "").strip()), None)
            if first_tool is None:
                continue
            name = trajectory_profile._normalize_terminalbench_tool_name(str(first_tool.get("fn") or "tool"))
            events.append(
                trajectory_profile.Event(
                    role="tool",
                    kind="tool_result",
                    name=name,
                    category=trajectory_profile._tool_category(name, observation),
                    content=observation,
                )
            )
        return events
    return [event for event in trajectory_profile._extract_events(adapter, row) if event.kind in {"tool_result", "observation"}]


def classify_error(text: str) -> str | None:
    if len(text) > 12000:
        text = text[:6000] + "\n" + text[-6000:]
    lowered = text.lower()
    for name, pattern in ERROR_PATTERNS.items():
        hints = ERROR_HINTS.get(name, ())
        if hints and not any(hint in lowered for hint in hints):
            continue
        if pattern.search(text):
            return name
    return None


def classify_shell_error(text: str) -> str | None:
    if len(text) > 12000:
        text = text[:6000] + "\n" + text[-6000:]
    lowered = text.lower()
    shell_hints = (
        "syntax error",
        "unexpected eof",
        "command not found",
        "not recognized",
        "no such file",
        "cannot access",
        "unrecognized argument",
        "unknown option",
        "invalid option",
        "missing operand",
        "permission denied",
    )
    if not any(hint in lowered for hint in shell_hints):
        return None
    for name, pattern in SHELL_ERROR_PATTERNS.items():
        if pattern.search(text):
            return name
    return "other_shell_error"


def allow_shell_error_for_event(shell_error: str, event: trajectory_profile.Event) -> bool:
    del shell_error
    shellish_names = {"tool", "observation", "execute_bash", "run_shell", "bash", "powershell", "shell", "command"}
    return event.category in {"shell", "test"} or event.name in shellish_names


def _extract_success_output_block(text: str) -> str:
    if "<returncode>0</returncode>" not in text:
        return text
    match = re.search(r"<output>\s*(.*?)\s*</output>", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1)
    return text


def _unwrap_line_numbered_preview(text: str) -> str:
    if "`cat -n`" not in text and "has been edited. Here's the result of running `cat -n`" not in text:
        return text
    lines = text.splitlines()
    numbered: list[str] = []
    for line in lines:
        match = re.match(r"^\s*\d+\s+(.*)$", line)
        if match:
            numbered.append(match.group(1))
    if len(numbered) >= 3:
        return "\n".join(numbered)
    return text


def _is_line_numbered_preview(text: str) -> bool:
    return _unwrap_line_numbered_preview(text) != text


def _looks_like_source_code_snippet(text: str) -> bool:
    lowered = text.lower()
    if any(marker in lowered for marker in ("traceback (most recent call last)", "failed ", "fail:", " e   assert", "syntax error:")):
        return False
    if re.search(r"(?mi)^\s*error:\s", text):
        return False
    code_markers = 0
    for marker in (
        "from ",
        "import ",
        "def ",
        "class ",
        "return ",
        "except ",
        "raise ",
        "if ",
        "elif ",
        "else:",
        "try:",
        "func ",
        "package ",
        "const ",
        "let ",
        "var ",
        "describe(",
        "it(",
        "it.each",
        ":=",
    ):
        if marker in lowered:
            code_markers += 1
    indented_lines = sum(1 for line in text.splitlines() if line.startswith(("    ", "\t", "  ")))
    assignment_lines = sum(
        1
        for line in text.splitlines()
        if re.match(r'^\s*[A-Za-z_][A-Za-z0-9_]*\s*(:=|=)\s*["\']?[^"\']+', line)
    )
    return (
        code_markers >= 3
        or (code_markers >= 2 and indented_lines >= 2)
        or (code_markers >= 2 and assignment_lines >= 2)
    )


def _looks_like_expected_error_example(event: trajectory_profile.Event, error_class: str) -> bool:
    lowered = event.content.lower()
    success_wrapper = "<returncode>0</returncode>" in lowered
    if error_class == "syntax_error":
        if "pass [" in lowered and "=> error: syntaxerror" in lowered:
            if "all smoke tests passed" in lowered or "pass [rt_sys/syntax]" in lowered:
                return True
    if error_class == "invalid_args":
        if "test cli:" in lowered and "should fail" in lowered:
            if "unrecognized option" in lowered or "unrecognized arguments" in lowered:
                return True
        if "expected error:" in lowered and (
            "unrecognized option" in lowered
            or "unrecognized arguments" in lowered
            or "invalid option" in lowered
        ):
            return True
        if "results:" in lowered and "passed" in lowered:
            if "pass [exit=1]" in lowered or "fail [exit=1]" in lowered:
                if "unknown option" in lowered or "unrecognized option" in lowered:
                    return True
        if success_wrapper and "expected error:" in lowered:
            return True
    if error_class in {"import_error", "missing_dependency", "test_assertion"}:
        if success_wrapper and "expected error:" in lowered:
            return True
        if success_wrapper and "failed during execution due to " in lowered:
            return True
    if error_class == "test_assertion":
        if success_wrapper and "type: <class 'assertionerror'>" in lowered:
            return True
    return False


def _looks_like_source_code_observation(event: trajectory_profile.Event) -> bool:
    extracted = _extract_success_output_block(event.content)
    if _is_line_numbered_preview(extracted):
        return True
    text = _unwrap_line_numbered_preview(extracted)
    if event.category in {"read", "edit"} or event.name in {"read_file", "str_replace_editor", "edit"}:
        return _looks_like_source_code_snippet(text)
    if event.category in {"shell", "test"} or event.name in {"bash", "execute_bash", "run_shell", "powershell", "shell", "command"}:
        return _looks_like_source_code_snippet(text)
    return False


def _allow_error_class_for_event(error_class: str, event: trajectory_profile.Event) -> bool:
    if _looks_like_expected_error_example(event, error_class):
        return False
    if _looks_like_source_code_observation(event) and error_class in {
        "missing_dependency",
        "import_error",
        "syntax_error",
        "test_assertion",
        "invalid_args",
        "command_not_found",
        "path_missing",
        "cwd_git",
        "timeout",
        "permission",
        "patch_apply",
    }:
        return False
    if error_class == "invalid_args":
        if allow_shell_error_for_event(error_class, event):
            return True
        return False
    if error_class != "timeout":
        return True
    lowered = event.content.lower()
    if "successfully killed shell" in lowered:
        return False
    if event.category != "read":
        return True
    strong_timeout_signals = (
        "timed out",
        "timeout after",
        "timeout expired",
        "operation timed out",
        "process timed out",
        "command timed out",
        "request timed out",
        "traceback",
        "exit code",
        "error:",
        "killed",
    )
    return any(signal in lowered for signal in strong_timeout_signals)


def _excerpt(text: str, pattern: re.Pattern[str], limit: int = 360) -> str:
    match = pattern.search(text)
    if not match:
        return text.replace("\n", " ")[:limit]
    start = max(0, match.start() - 120)
    end = min(len(text), match.end() + 220)
    return text[start:end].replace("\n", " ")[:limit]


def summarize_dataset(name: str, adapter: str, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows_profiled = 0
    result_events = 0
    error_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    repeated_errors: Counter[str] = Counter()
    shell_errors: Counter[str] = Counter()
    shell_tool_errors: Counter[str] = Counter()
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    shell_examples: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        rows_profiled += 1
        previous_key = ""
        run_count = 0
        for event in _extract_result_events(adapter, row):
            result_events += 1
            error_class = classify_error(event.content)
            shell_error = classify_shell_error(event.content)
            if shell_error and allow_shell_error_for_event(shell_error, event):
                shell_errors[shell_error] += 1
                shell_tool_errors[f"{event.name}:{shell_error}"] += 1
                if len(shell_examples[shell_error]) < 3:
                    pattern = SHELL_ERROR_PATTERNS.get(shell_error, re.compile(r".+", re.S))
                    shell_examples[shell_error].append(
                        {
                            "tool": event.name,
                            "category": event.category,
                            "excerpt": _excerpt(event.content, pattern),
                        }
                    )
            if not error_class:
                previous_key = ""
                run_count = 0
                continue
            if not _allow_error_class_for_event(error_class, event):
                previous_key = ""
                run_count = 0
                continue
            error_counts[error_class] += 1
            tool_counts[f"{event.name}:{error_class}"] += 1
            key = f"{event.name}:{error_class}"
            if key == previous_key:
                run_count += 1
            else:
                if previous_key and run_count >= 2:
                    repeated_errors[previous_key] += 1
                previous_key = key
                run_count = 1
            if len(examples[error_class]) < 3:
                examples[error_class].append(
                    {
                        "tool": event.name,
                        "category": event.category,
                        "excerpt": _excerpt(event.content, ERROR_PATTERNS[error_class]),
                    }
                )
        if previous_key and run_count >= 2:
            repeated_errors[previous_key] += 1

    recommendations = [
        {
            "id": RECOMMENDATIONS[name],
            "error_class": name,
            "count": count,
            "priority": "high" if count >= max(10, result_events * 0.05) else "medium",
        }
        for name, count in error_counts.most_common()
    ]
    return {
        "dataset": name,
        "rows_profiled": rows_profiled,
        "result_events": result_events,
        "error_counts": dict(error_counts.most_common()),
        "top_tool_errors": [{"tool_error": key, "count": count} for key, count in tool_counts.most_common(20)],
        "top_shell_errors": [{"shell_error": key, "count": count} for key, count in shell_errors.most_common(20)],
        "top_shell_tool_errors": [{"tool_error": key, "count": count} for key, count in shell_tool_errors.most_common(20)],
        "repeated_error_loops": [{"tool_error": key, "count": count} for key, count in repeated_errors.most_common(20)],
        "examples": dict(examples),
        "shell_examples": dict(shell_examples),
        "recommendations": recommendations,
    }


def _iter_projected_parquet_rows(paths: Iterable[Path], *, columns: list[str], max_rows: int | None = None) -> Iterable[dict[str, Any]]:
    if pq is None:
        raise RuntimeError("trajectory error profiling requires optional dependency pyarrow")
    emitted = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        available = [column for column in columns if column in parquet.schema_arrow.names]
        if not available:
            continue
        for batch in parquet.iter_batches(batch_size=64, columns=available):
            for row in batch.to_pylist():
                yield row
                emitted += 1
                if max_rows is not None and emitted >= max_rows:
                    return


def _iter_dataset_rows(data_root: Path, dataset: str, max_rows: int | None) -> tuple[str, Iterable[dict[str, Any]]]:
    spec = trajectory_profile.DATASET_SPECS[dataset]
    root = data_root / dataset
    paths: list[Path] = []
    for pattern in spec["paths"]:
        paths.extend(sorted(root.glob(pattern)))
    adapter = str(spec["adapter"])
    if adapter == "agent_race":
        rows = trajectory_profile._iter_agent_race_rows(paths, max_rows=max_rows)
        return adapter, trajectory_profile._iter_rows_with_trajectory_content(adapter, rows, max_rows=max_rows)
    if adapter == "smith":
        columns = ["messages"]
    elif adapter == "openhands":
        columns = ["trajectory", "messages", "messages_json"]
    elif adapter == "cc_bench":
        columns = ["trajectory", "task_id", "id", "task_category", "model_name"]
    elif adapter == "trace_commons":
        columns = ["messages", "messages_json", "trace", "session_id", "harness", "prompt", "num_tool_calls"]
    elif adapter == "thoughtworks":
        columns = ["messages", "messages_json", "agent_framework", "source_dataset", "session_id", "source_id"]
    elif adapter == "terminalbench":
        columns = ["steps"]
    else:
        columns = ["trajectory"]
    rows = _iter_projected_parquet_rows(paths, columns=columns)
    return adapter, trajectory_profile._iter_rows_with_trajectory_content(adapter, rows, max_rows=max_rows)


def build_profile(data_root: Path, datasets: list[str], max_rows: int | None) -> dict[str, Any]:
    effective_max_rows = None if max_rows == 0 else max_rows
    summaries: list[dict[str, Any]] = []
    portfolio: Counter[str] = Counter()
    for dataset in datasets:
        adapter, rows = _iter_dataset_rows(data_root, dataset, effective_max_rows)
        summary = summarize_dataset(dataset, adapter, rows)
        summaries.append(summary)
        for recommendation in summary["recommendations"]:
            portfolio[str(recommendation["id"])] += int(recommendation["count"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": data_root.as_posix(),
        "max_rows_per_dataset": effective_max_rows,
        "datasets": summaries,
        "portfolio_recommendations": [{"id": key, "count": value} for key, value in portfolio.most_common()],
    }


def format_summary(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    for dataset in payload.get("datasets", []):
        lines.append(
            f"[trajectory-error-profile] dataset={dataset['dataset']} rows={dataset['rows_profiled']} "
            f"results={dataset['result_events']} errors={sum(dataset.get('error_counts', {}).values())}"
        )
        top = ", ".join(f"{name}:{count}" for name, count in list(dataset.get("error_counts", {}).items())[:8])
        if top:
            lines.append(f"  top_errors {top}")
        top_tool = ", ".join(f"{item['tool_error']}:{item['count']}" for item in dataset.get("top_tool_errors", [])[:5])
        if top_tool:
            lines.append(f"  top_tool_errors {top_tool}")
        top_shell = ", ".join(f"{item['shell_error']}:{item['count']}" for item in dataset.get("top_shell_errors", [])[:6])
        if top_shell:
            lines.append(f"  top_shell_errors {top_shell}")
    if payload.get("portfolio_recommendations"):
        lines.append("[trajectory-error-profile] portfolio " + ", ".join(f"{item['id']}:{item['count']}" for item in payload["portfolio_recommendations"][:8]))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile tool/result error classes in coding-agent trajectories.")
    parser.add_argument("--data-root", type=Path, default=trajectory_profile.DEFAULT_DATA_ROOT)
    parser.add_argument("--datasets", nargs="*", default=list(trajectory_profile.DATASET_SPECS))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = build_profile(args.data_root, args.datasets, args.max_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(format_summary(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
