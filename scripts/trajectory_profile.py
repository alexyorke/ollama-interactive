from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


DEFAULT_DATA_ROOT = Path("scratch") / "external" / "datasets"

DATASET_SPECS = {
    "nebius-swe-agent-trajectories": {
        "paths": ["data/*.parquet"],
        "adapter": "swe_agent",
    },
    "nebius-swe-rebench-openhands-trajectories": {
        "paths": ["trajectories.parquet"],
        "adapter": "openhands",
    },
    "swe-smith-trajectories": {
        "paths": ["data/train-*.parquet", "data/tool-*.parquet", "data/xml-*.parquet", "data/ticks-*.parquet"],
        "adapter": "smith",
    },
}

READ_TOOLS = {
    "open",
    "read",
    "read_file",
    "view",
    "cat",
    "code_view",
    "file_read",
    "read_symbol",
    "code_outline",
}
SEARCH_TOOLS = {
    "search",
    "search_file",
    "search_dir",
    "search_symbols",
    "find_file",
    "grep",
    "glob",
    "list_files",
    "ls",
    "context_pack",
}
EDIT_TOOLS = {
    "edit",
    "edit_file",
    "replace",
    "replace_file",
    "replace_in_file",
    "replace_symbol",
    "replace_symbols",
    "write",
    "write_file",
    "apply_patch",
    "str_replace_editor",
    "insert",
}
TEST_TOOLS = {
    "run_test",
    "pytest",
    "unittest",
    "test",
}
SHELL_TOOLS = {
    "bash",
    "run_shell",
    "shell",
    "command",
    "execute",
    "execute_bash",
}
GIT_TOOLS = {
    "git",
    "git_status",
    "git_diff",
    "git_commit",
}
SUBMIT_TOOLS = {"submit", "final", "finish"}

COMMAND_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("read_file", re.compile(r"\b(?:read_file|open|cat)\b", re.IGNORECASE)),
    ("search", re.compile(r"\b(?:search_file|search_dir|search|grep|find_file|glob|list_files|ls)\b", re.IGNORECASE)),
    ("replace_symbols", re.compile(r"\b(?:replace_symbols|replace_symbol|edit_file|str_replace_editor|apply_patch)\b", re.IGNORECASE)),
    ("write_file", re.compile(r"\b(?:write_file|create file|overwrite file|insert)\b", re.IGNORECASE)),
    ("run_test", re.compile(r"\b(?:run_test|pytest|unittest|nose|tox)\b", re.IGNORECASE)),
    ("run_shell", re.compile(r"\b(?:bash|run_shell|python\s+-m|pip\s+install|make\s+test)\b", re.IGNORECASE)),
    ("git", re.compile(r"\bgit\s+(?:status|diff|commit|apply|checkout)\b", re.IGNORECASE)),
    ("submit", re.compile(r"\b(?:submit|finish)\b", re.IGNORECASE)),
]


@dataclass
class Event:
    role: str
    kind: str
    name: str
    category: str
    content: str


@dataclass(frozen=True)
class Recommendation:
    id: str
    priority: str
    title: str
    change_type: str
    trigger: str
    rationale: str
    expected_effect: str
    experiments: tuple[str, ...]


def _command_category_from_content(content: str) -> str | None:
    lowered = content.lower()
    if re.search(r"\b(?:pytest|unittest|nose2?|tox|nox|cargo test|go test|npm test|pnpm test|yarn test)\b", lowered):
        return "test"
    if re.search(r"\bgit\s+(?:status|diff|commit|apply|checkout|restore|add)\b", lowered):
        return "git"
    if re.search(r"\b(?:cat|sed\s+-n|head|tail)\b", lowered):
        return "read"
    if re.search(r"\b(?:grep|rg|find|ls|tree)\b", lowered):
        return "search"
    return None


def _tool_category(name: str, content: str = "") -> str:
    lowered = name.strip().lower()
    inferred = _command_category_from_content(content) if lowered in SHELL_TOOLS else None
    if inferred:
        return inferred
    if lowered in READ_TOOLS:
        return "read"
    if lowered in SEARCH_TOOLS:
        return "search"
    if lowered in EDIT_TOOLS:
        return "edit"
    if lowered in TEST_TOOLS:
        return "test"
    if lowered in SHELL_TOOLS:
        return "shell"
    if lowered in GIT_TOOLS:
        return "git"
    if lowered in SUBMIT_TOOLS:
        return "submit"
    return "other"


def _normalize_tool_name(name: str) -> str:
    lowered = name.strip().lower()
    return lowered.replace(" ", "_")


def _infer_tool_name_from_text(text: str) -> str | None:
    snippet = text.strip()
    if not snippet:
        return None
    first_line = snippet.splitlines()[0].strip()
    for name, pattern in COMMAND_PATTERNS:
        if pattern.search(first_line):
            return name
    for name, pattern in COMMAND_PATTERNS:
        if pattern.search(snippet[:800]):
            return name
    return None


def _content_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _deserialize_possible_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _extract_openhands_events(trajectory: list[dict[str, Any]]) -> list[Event]:
    events: list[Event] = []
    for message in trajectory:
        role = str(message.get("role") or "")
        content = _content_text(message.get("content"))
        if role == "assistant":
            tool_calls = _deserialize_possible_json(message.get("tool_calls"))
            if isinstance(tool_calls, list) and tool_calls:
                for call in tool_calls:
                    function = call.get("function") if isinstance(call, dict) else None
                    name = _normalize_tool_name(str(function.get("name") or call.get("name") or "")) if isinstance(function, dict) or isinstance(call, dict) else ""
                    if name:
                        payload = _content_text(call)
                        events.append(Event(role="assistant", kind="tool_call", name=name, category=_tool_category(name, payload), content=payload))
            elif content.strip():
                inferred = _infer_tool_name_from_text(content)
                if inferred:
                    events.append(Event(role="assistant", kind="tool_call", name=inferred, category=_tool_category(inferred, content), content=content))
        elif role == "tool":
            name = _normalize_tool_name(str(message.get("name") or "tool"))
            events.append(Event(role="tool", kind="tool_result", name=name, category=_tool_category(name, content), content=content))
        elif role == "user" and content.strip():
            inferred = _infer_tool_name_from_text(content)
            if inferred:
                events.append(Event(role="user", kind="observation", name=inferred, category=_tool_category(inferred, content), content=content))
    return events


def _extract_swe_agent_events(trajectory: list[dict[str, Any]]) -> list[Event]:
    events: list[Event] = []
    for message in trajectory:
        role = str(message.get("role") or "")
        content = _content_text(message.get("text") or message.get("system_prompt"))
        if role.lower() in {"assistant", "ai"}:
            inferred = _infer_tool_name_from_text(content)
            if inferred:
                events.append(Event(role=role, kind="tool_call", name=inferred, category=_tool_category(inferred, content), content=content))
        elif role.lower() == "user" and content.strip():
            inferred = _infer_tool_name_from_text(content)
            name = inferred or "observation"
            events.append(Event(role=role, kind="tool_result", name=name, category=_tool_category(name, content), content=content))
    return events


def _extract_smith_events(messages: Any) -> list[Event]:
    events: list[Event] = []
    normalized = _deserialize_possible_json(messages)
    if not isinstance(normalized, list):
        return events
    for message in normalized:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        content = _content_text(message.get("content"))
        if role == "assistant":
            tool_calls = _deserialize_possible_json(message.get("tool_calls"))
            if isinstance(tool_calls, list) and tool_calls:
                for call in tool_calls:
                    function = call.get("function") if isinstance(call, dict) else None
                    name = _normalize_tool_name(str(function.get("name") or call.get("name") or "")) if isinstance(function, dict) or isinstance(call, dict) else ""
                    if name:
                        payload = _content_text(call)
                        events.append(Event(role="assistant", kind="tool_call", name=name, category=_tool_category(name, payload), content=payload))
            else:
                inferred = _infer_tool_name_from_text(content)
                if inferred:
                    events.append(Event(role="assistant", kind="tool_call", name=inferred, category=_tool_category(inferred, content), content=content))
        elif role == "tool":
            name = _normalize_tool_name(str(message.get("name") or "tool"))
            events.append(Event(role="tool", kind="tool_result", name=name, category=_tool_category(name, content), content=content))
    return events


def _extract_events(adapter: str, row: dict[str, Any]) -> list[Event]:
    if adapter == "openhands":
        return _extract_openhands_events(list(row.get("trajectory") or []))
    if adapter == "swe_agent":
        return _extract_swe_agent_events(list(row.get("trajectory") or []))
    if adapter == "smith":
        return _extract_smith_events(row.get("messages"))
    return []


def _iter_parquet_rows(paths: Iterable[Path], max_rows: int | None = None) -> Iterable[dict[str, Any]]:
    emitted = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=128):
            for row in batch.to_pylist():
                yield row
                emitted += 1
                if max_rows is not None and emitted >= max_rows:
                    return


def _detect_repeated_tool_runs(names: list[str]) -> list[tuple[str, int]]:
    loops: list[tuple[str, int]] = []
    if not names:
        return loops
    current = names[0]
    count = 1
    for name in names[1:]:
        if name == current:
            count += 1
            continue
        if count >= 3:
            loops.append((current, count))
        current = name
        count = 1
    if count >= 3:
        loops.append((current, count))
    return loops


def _detect_context_loop(categories: list[str]) -> bool:
    run = 0
    for category in categories:
        if category in {"read", "search"}:
            run += 1
            if run >= 4:
                return True
        else:
            run = 0
    return False


def _trajectory_metrics(events: list[Event]) -> dict[str, Any]:
    tool_events = [event for event in events if event.kind == "tool_call"]
    tool_names = [event.name for event in tool_events if event.name]
    categories = [event.category for event in tool_events]
    first_edit = next((idx for idx, event in enumerate(tool_events) if event.category == "edit"), None)
    first_test_after_edit = None
    if first_edit is not None:
        for idx, event in enumerate(tool_events[first_edit + 1 :], start=first_edit + 1):
            if event.category == "test":
                first_test_after_edit = idx
                break
    return {
        "tool_calls": len(tool_events),
        "categories": categories,
        "tool_names": tool_names,
        "first_edit_index": first_edit,
        "has_edit": first_edit is not None,
        "has_test": any(event.category == "test" for event in tool_events),
        "has_search_or_read_before_edit": bool(first_edit is not None and any(cat in {"read", "search"} for cat in categories[:first_edit])),
        "has_test_after_edit": first_test_after_edit is not None,
        "context_loop": _detect_context_loop(categories),
        "repeated_tools": _detect_repeated_tool_runs(tool_names),
    }


def _mean(values: list[int]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return round((ordered[mid - 1] + ordered[mid]) / 2, 2)


def _threshold_label(summary: dict[str, Any], key: str) -> str:
    return f"{key}={summary.get(key)}"


def _summarize_dataset(name: str, adapter: str, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    total_rows = 0
    total_tool_calls = 0
    category_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    repeated_loops: Counter[str] = Counter()
    first_edit_indexes: list[int] = []
    tool_calls_before_edit: list[int] = []
    context_loop_rows = 0
    edit_without_prior_context = 0
    edit_without_later_test = 0
    rows_with_edit = 0
    rows_with_test = 0

    for row in rows:
        total_rows += 1
        metrics = _trajectory_metrics(_extract_events(adapter, row))
        total_tool_calls += metrics["tool_calls"]
        category_counts.update(metrics["categories"])
        tool_counts.update(metrics["tool_names"])
        if metrics["has_edit"]:
            rows_with_edit += 1
            first_edit_indexes.append(int(metrics["first_edit_index"]))
            tool_calls_before_edit.append(int(metrics["first_edit_index"]))
            if not metrics["has_search_or_read_before_edit"]:
                edit_without_prior_context += 1
            if not metrics["has_test_after_edit"]:
                edit_without_later_test += 1
        if metrics["has_test"]:
            rows_with_test += 1
        if metrics["context_loop"]:
            context_loop_rows += 1
        for tool_name, count in metrics["repeated_tools"]:
            repeated_loops[f"{tool_name} x{count}"] += 1

    avg_tool_calls = round(total_tool_calls / total_rows, 2) if total_rows else 0.0
    summary = {
        "dataset": name,
        "rows_profiled": total_rows,
        "avg_tool_calls": avg_tool_calls,
        "rows_with_edit_pct": round((rows_with_edit / total_rows) * 100, 2) if total_rows else 0.0,
        "rows_with_test_pct": round((rows_with_test / total_rows) * 100, 2) if total_rows else 0.0,
        "context_loop_rows_pct": round((context_loop_rows / total_rows) * 100, 2) if total_rows else 0.0,
        "edit_without_prior_context_pct": round((edit_without_prior_context / rows_with_edit) * 100, 2) if rows_with_edit else 0.0,
        "edit_without_later_test_pct": round((edit_without_later_test / rows_with_edit) * 100, 2) if rows_with_edit else 0.0,
        "avg_first_edit_index": _mean(first_edit_indexes),
        "median_first_edit_index": _median(first_edit_indexes),
        "avg_tool_calls_before_edit": _mean(tool_calls_before_edit),
        "tool_category_counts": dict(category_counts.most_common()),
        "top_tools": [{"name": name, "count": count} for name, count in tool_counts.most_common(20)],
        "top_repeated_loops": [{"loop": loop, "count": count} for loop, count in repeated_loops.most_common(20)],
    }
    summary["recommendations"] = [recommendation.__dict__ for recommendation in _heuristic_recommendations(summary)]
    return summary


def _portfolio_recommendations(datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Recommendation]] = defaultdict(list)
    dataset_map: dict[str, set[str]] = defaultdict(set)
    for dataset in datasets:
        dataset_name = str(dataset.get("dataset") or "unknown")
        for raw in dataset.get("recommendations", []):
            if not isinstance(raw, dict):
                continue
            recommendation = Recommendation(**raw)
            grouped[recommendation.id].append(recommendation)
            dataset_map[recommendation.id].add(dataset_name)
    ranked: list[dict[str, Any]] = []
    for rec_id, items in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        exemplar = items[0]
        ranked.append(
            {
                "id": rec_id,
                "priority": exemplar.priority,
                "title": exemplar.title,
                "change_type": exemplar.change_type,
                "seen_in_datasets": len(dataset_map[rec_id]),
                "datasets": sorted(dataset_map[rec_id]),
                "expected_effect": exemplar.expected_effect,
                "experiments": list(exemplar.experiments),
            }
        )
    return ranked


def _heuristic_recommendations(summary: dict[str, Any]) -> list[Recommendation]:
    recommendations: list[Recommendation] = []
    category_counts = summary.get("tool_category_counts", {})
    read_search = int(category_counts.get("read", 0)) + int(category_counts.get("search", 0))
    edits = int(category_counts.get("edit", 0))
    tests = int(category_counts.get("test", 0))
    if edits and read_search / max(edits, 1) >= 3:
        recommendations.append(
            Recommendation(
                id="context-planner",
                priority="high",
                title="Add compact context planning before broad inspection",
                change_type="controller",
                trigger=f"read_search_per_edit={round(read_search / max(edits, 1), 2)}",
                rationale="Trajectories spend too many steps reading and searching before they make a grounded edit.",
                expected_effect="Fewer repeated file reads, smaller prompts, and faster first edit.",
                experiments=(
                    "A/B a controller nudge that switches from repeated read/search to symbol-first context planning after two context-only turns.",
                    "Measure median tool calls before first edit and token deltas on local coding benchmarks.",
                ),
            )
        )
    if summary.get("context_loop_rows_pct", 0.0) >= 20:
        recommendations.append(
            Recommendation(
                id="loop-cap",
                priority="high",
                title="Force loop-breaking after repeated context-only actions",
                change_type="controller",
                trigger=_threshold_label(summary, "context_loop_rows_pct"),
                rationale="Many sessions keep gathering context without progressing to validation or mutation.",
                expected_effect="Lower loop rate and less token waste from replaying stale context.",
                experiments=(
                    "Inject a controller message after four consecutive read/search actions requiring narrower scope, validation, or mutation.",
                    "Compare fail_closed rate and average tool calls against baseline.",
                ),
            )
        )
    if summary.get("edit_without_later_test_pct", 0.0) >= 25:
        recommendations.append(
            Recommendation(
                id="post-edit-validation",
                priority="high",
                title="Require cheap validation after edits",
                change_type="controller",
                trigger=_threshold_label(summary, "edit_without_later_test_pct"),
                rationale="Edits are often left unvalidated, which increases wrong-final risk and extra repair turns.",
                expected_effect="More grounded finals and fewer broken edits carried forward.",
                experiments=(
                    "Auto-run run_function_probe or configured run_test after mutations when a cheap validator exists.",
                    "Track accuracy and prompt-token cost with mandatory validation versus baseline.",
                ),
            )
        )
    if summary.get("edit_without_prior_context_pct", 0.0) >= 15:
        recommendations.append(
            Recommendation(
                id="ground-before-mutate",
                priority="medium",
                title="Require grounded context before mutation",
                change_type="guard",
                trigger=_threshold_label(summary, "edit_without_prior_context_pct"),
                rationale="Some trajectories mutate files before reading the relevant implementation, symbol, or failure evidence.",
                expected_effect="Fewer wrong-file edits and better use of structured edit tools.",
                experiments=(
                    "Reject mutation tools until matching file or symbol evidence exists in the current turn.",
                    "Measure wrong-file edit rate and retries on multiturn refactor tasks.",
                ),
            )
        )
    if tests and edits and tests / max(edits, 1) >= 2.5:
        recommendations.append(
            Recommendation(
                id="failure-compression",
                priority="medium",
                title="Compress failed test output before another model turn",
                change_type="tooling",
                trigger=f"tests_per_edit={round(tests / max(edits, 1), 2)}",
                rationale="Validation happens often, but repeated failures likely resend large blobs without enough diagnosis.",
                expected_effect="Lower retry prompt cost and more directed follow-up edits.",
                experiments=(
                    "Insert diagnose_test_failure before the second failed run_test on the same mutation version.",
                    "Measure prompt-token reduction and recovery rate after first failed tests.",
                ),
            )
        )
    if not recommendations:
        recommendations.append(
            Recommendation(
                id="baseline-review",
                priority="low",
                title="No dominant inefficiency exceeded thresholds",
                change_type="analysis",
                trigger="no-thresholds-fired",
                rationale="This dataset slice does not have one overwhelming waste signature.",
                expected_effect="Use side-by-side comparison to prioritize the next optimization target.",
                experiments=(
                    "Increase max_rows and compare repeated-tool loops across datasets.",
                    "Correlate trajectory signals with live CLI token traces before changing defaults.",
                ),
            )
        )
    return recommendations


def _resolve_dataset_paths(root: Path, dataset_name: str) -> tuple[str, list[Path]]:
    spec = DATASET_SPECS[dataset_name]
    dataset_root = root / dataset_name
    paths: list[Path] = []
    for pattern in spec["paths"]:
        paths.extend(sorted(dataset_root.glob(pattern)))
    return spec["adapter"], paths


def _format_report(payload: dict[str, Any]) -> str:
    lines = []
    for dataset in payload["datasets"]:
        lines.append(f"[trajectory-profile] dataset={dataset['dataset']} rows={dataset['rows_profiled']} avg_tool_calls={dataset['avg_tool_calls']} context_loop_pct={dataset['context_loop_rows_pct']} edit_without_test_pct={dataset['edit_without_later_test_pct']}")
        top_tools = ", ".join(f"{item['name']}:{item['count']}" for item in dataset["top_tools"][:8])
        if top_tools:
            lines.append(f"  top_tools {top_tools}")
        for recommendation in dataset["recommendations"]:
            lines.append(f"  recommendation[{recommendation['priority']}] {recommendation['id']} {recommendation['title']}")
    if payload.get("portfolio_recommendations"):
        lines.append("[trajectory-profile] portfolio")
        for recommendation in payload["portfolio_recommendations"][:8]:
            lines.append(
                "  "
                + f"{recommendation['priority']} {recommendation['id']} "
                + f"datasets={recommendation['seen_in_datasets']} "
                + recommendation["title"]
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile coding-agent trajectory datasets for inefficient workflow patterns.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Root folder containing downloaded dataset directories.")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_SPECS), default=sorted(DATASET_SPECS), help="Dataset directories to profile.")
    parser.add_argument("--max-rows", type=int, default=5000, help="Maximum rows per dataset to scan.")
    parser.add_argument("--output", default="scratch/external/datasets/trajectory-profile.json", help="JSON output path.")
    args = parser.parse_args(argv)

    root = Path(args.data_root)
    output = Path(args.output)
    if not output.is_absolute():
        output = Path.cwd() / output
    summaries = []
    for dataset_name in args.datasets:
        adapter, paths = _resolve_dataset_paths(root, dataset_name)
        if not paths:
            summaries.append(
                {
                    "dataset": dataset_name,
                    "rows_profiled": 0,
                    "status": "missing",
                    "recommendations": [f"No Parquet files found under {root / dataset_name}."],
                }
            )
            continue
        summaries.append(_summarize_dataset(dataset_name, adapter, _iter_parquet_rows(paths, max_rows=args.max_rows)))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(root.resolve(strict=False)),
        "max_rows_per_dataset": args.max_rows,
        "datasets": summaries,
    }
    payload["portfolio_recommendations"] = _portfolio_recommendations(summaries)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(_format_report(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
