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
    "invalid_args": re.compile(r"unrecognized arguments|invalid option|unknown option|usage:|error: argument|missing required", re.I),
    "command_not_found": re.compile(r"command not found|not recognized as (?:an internal|the name)", re.I),
    "path_missing": re.compile(r"No such file or directory|cannot access|FileNotFoundError|Path does not exist|path .* does not exist", re.I),
    "cwd_git": re.compile(r"cd: .*No such file|outside workspace|escapes the workspace|not inside a git repository", re.I),
    "timeout": re.compile(r"\btimed out\b|\btimeout\b|\bexceeded\b[^\n]*\btimeout\b|\bkilled\b", re.I),
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
    "invalid_args": ("unrecognized arguments", "invalid option", "unknown option", "usage:", "error: argument", "missing required"),
    "command_not_found": ("command not found", "not recognized as an internal", "not recognized as the name"),
    "path_missing": ("no such file or directory", "cannot access", "filenotfounderror", "path does not exist"),
    "cwd_git": ("cd:", "outside workspace", "escapes the workspace", "not inside a git repository"),
    "timeout": ("timed out", "timeout", "exceeded", "killed"),
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


def _extract_result_events(adapter: str, row: dict[str, Any]) -> list[trajectory_profile.Event]:
    events: list[trajectory_profile.Event] = []
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
                elif trajectory_profile._content_text(message.get("content")).strip():
                    inferred = trajectory_profile._infer_tool_name_from_text(trajectory_profile._content_text(message.get("content")))
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
    if adapter == "trace_commons":
        return _extract_result_events("openhands", row)
    if adapter == "thoughtworks":
        effective_adapter = trajectory_profile._thoughtworks_row_adapter(row)
        if effective_adapter == "openhands":
            return _extract_result_events("openhands", row)
        return _extract_result_events("smith", {"messages": row.get("messages") or row.get("messages_json")})
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
        messages = trajectory_profile._deserialize_possible_json(row.get("messages"))
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
            name = trajectory_profile._normalize_tool_name(str(first_tool.get("fn") or "tool"))
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


def _allow_error_class_for_event(error_class: str, event: trajectory_profile.Event) -> bool:
    if error_class != "timeout":
        return True
    if event.category != "read":
        return True
    lowered = event.content.lower()
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
            if shell_error and (event.category in {"shell", "test"} or event.name in {"tool", "observation", "execute_bash", "run_shell", "bash", "powershell", "shell", "command"}):
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
    if adapter == "smith":
        columns = ["messages"]
    elif adapter == "openhands":
        columns = ["trajectory", "messages", "messages_json"]
    elif adapter == "trace_commons":
        columns = ["messages", "session_id", "harness", "prompt", "num_tool_calls"]
    elif adapter == "thoughtworks":
        columns = ["messages", "messages_json", "agent_framework", "source_dataset", "session_id", "source_id"]
    elif adapter == "terminalbench":
        columns = ["steps"]
    else:
        columns = ["trajectory"]
    rows = _iter_projected_parquet_rows(paths, columns=columns)
    return adapter, trajectory_profile._iter_rows_with_trajectory_content(adapter, rows, max_rows=max_rows)


def build_profile(data_root: Path, datasets: list[str], max_rows: int | None) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    portfolio: Counter[str] = Counter()
    for dataset in datasets:
        adapter, rows = _iter_dataset_rows(data_root, dataset, max_rows)
        summary = summarize_dataset(dataset, adapter, rows)
        summaries.append(summary)
        for recommendation in summary["recommendations"]:
            portfolio[str(recommendation["id"])] += int(recommendation["count"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": data_root.as_posix(),
        "max_rows_per_dataset": max_rows,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile tool/result error classes in coding-agent trajectories.")
    parser.add_argument("--data-root", type=Path, default=trajectory_profile.DEFAULT_DATA_ROOT)
    parser.add_argument("--datasets", nargs="*", default=list(trajectory_profile.DATASET_SPECS))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_profile(args.data_root, args.datasets, args.max_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(format_summary(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
