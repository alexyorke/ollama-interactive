from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import trajectory_profile


DEFAULT_DATA_ROOT = trajectory_profile.DEFAULT_DATA_ROOT
DEFAULT_OUTPUT_JSON = DEFAULT_DATA_ROOT / "trajectory-transcript-sampler.json"
DEFAULT_OUTPUT_MD = DEFAULT_DATA_ROOT / "trajectory-transcript-sampler.md"


def _truncate(text: str, limit: int = 240) -> str:
    value = " ".join(text.split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _looks_like_placeholder_prompt(text: str) -> bool:
    stripped = text.strip()
    if re.fullmatch(r"\$\d+", stripped):
        return True
    return False


def _sanitize_terminalbench_prompt(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    sanitized = re.sub(
        r"(?is)^\s*<environment_context>.*?</environment_context>\s*",
        "",
        stripped,
        count=1,
    ).strip()
    if sanitized:
        return sanitized
    collapsed = " ".join(stripped.split()).lower()
    if collapsed.startswith("<environment_context>") and collapsed.endswith("</environment_context>"):
        return ""
    return stripped


def _stringify_message_content(value: Any) -> str:
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").lower()
            if item_type == "text":
                item_text = item.get("text")
                if isinstance(item_text, str) and item_text.strip():
                    parts.append(item_text.strip())
            elif item_type == "thinking":
                item_text = item.get("thinking")
                if isinstance(item_text, str) and item_text.strip():
                    parts.append(item_text.strip())
            elif item_type == "tool_result":
                item_text = trajectory_profile._content_text(item.get("content")).strip()
                if item_text:
                    parts.append(item_text)
        if parts:
            return "\n".join(parts)
        return ""
    text = trajectory_profile._content_text(value).strip()
    if text:
        return text
    return ""


def _preview_message_text(value: Any) -> str:
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").lower()
            if item_type in {"text", "reasoning", "thinking"}:
                item_text = item.get("text") if item_type != "thinking" else item.get("thinking")
                if isinstance(item_text, str) and item_text.strip():
                    parts.append(item_text.strip())
            elif item_type == "tool_result":
                item_text = trajectory_profile._content_text(item.get("content")).strip()
                if item_text:
                    parts.append(item_text)
        return "\n".join(parts).strip()
    return trajectory_profile._content_text(value).strip()


def _find_openhands_message(row: dict[str, Any], role: str) -> str:
    def _message_preview(message: dict[str, Any]) -> str:
        for raw_value in (message.get("content"), message.get("reasoning_content")):
            text = _preview_message_text(raw_value).strip()
            if text:
                return text
        return ""

    for message in trajectory_profile._openhands_messages(row):
        if str(message.get("role") or "").lower() != role:
            continue
        text = _message_preview(message)
        if text:
            return text
    for message in _trace_fallback_messages(row):
        if str(message.get("role") or "").lower() != role:
            continue
        text = _message_preview(message)
        if text:
            return text
    return ""


def _trace_fallback_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw_trace = trajectory_profile._deserialize_possible_json(row.get("trace"))
    trace_items: list[Any]
    if isinstance(raw_trace, list):
        trace_items = raw_trace
    elif raw_trace is None:
        return []
    else:
        trace_items = [raw_trace]
    normalized: list[dict[str, Any]] = []
    for item in trace_items:
        candidate = trajectory_profile._deserialize_possible_json(item)
        if not isinstance(candidate, dict):
            continue
        nested_message = candidate.get("message")
        role = str(candidate.get("role") or "").lower()
        if isinstance(nested_message, dict) and role:
            message_payload = dict(nested_message)
            message_payload.setdefault("role", role)
            normalized.append(message_payload)
            continue
        structured_messages = candidate.get("messages")
        if not isinstance(structured_messages, list):
            continue
        for structured in structured_messages:
            if not isinstance(structured, dict):
                continue
            info = structured.get("info")
            parts = structured.get("parts")
            structured_role = str(info.get("role") or "").lower() if isinstance(info, dict) else ""
            if not structured_role or not isinstance(parts, list):
                continue
            content_parts: list[dict[str, Any]] = []
            for part in parts:
                if not isinstance(part, dict):
                    continue
                part_type = str(part.get("type") or "").lower()
                if part_type in {"text", "reasoning"}:
                    text = str(part.get("text") or "").strip()
                    if text:
                        content_parts.append({"type": "text", "text": text})
                elif part_type == "tool":
                    tool_name = str(part.get("tool") or "").strip()
                    if not tool_name:
                        continue
                    state = part.get("state")
                    input_payload = state.get("input") if isinstance(state, dict) else None
                    tool_call: dict[str, Any] = {"type": "tool_use", "name": tool_name}
                    if isinstance(input_payload, dict):
                        tool_call["input"] = input_payload
                    content_parts.append(tool_call)
            if content_parts:
                normalized.append({"role": structured_role, "content": content_parts})
    return normalized


def _trace_fallback_tool_preview(row: dict[str, Any], limit: int = 6) -> list[str]:
    names: list[str] = []
    for message in _trace_fallback_messages(row):
        if str(message.get("role") or "").lower() != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if str(part.get("type") or "").lower() != "tool_use":
                continue
            name = trajectory_profile._normalize_tool_name(str(part.get("name") or ""))
            if not name:
                continue
            names.append(name)
            if len(names) >= limit:
                return names
    return names


def _find_terminalbench_prompt(row: dict[str, Any]) -> str:
    steps = trajectory_profile._deserialize_possible_json(row.get("steps"))
    if not isinstance(steps, list):
        return ""
    for step in steps:
        if not isinstance(step, dict):
            continue
        if str(step.get("src") or "").lower() != "user":
            continue
        text = trajectory_profile._content_text(step.get("msg")).strip()
        if not text:
            continue
        sanitized = _sanitize_terminalbench_prompt(text)
        if sanitized and not _looks_like_placeholder_prompt(sanitized):
            return sanitized
    return ""


def _find_terminalbench_assistant(row: dict[str, Any]) -> str:
    steps = trajectory_profile._deserialize_possible_json(row.get("steps"))
    if not isinstance(steps, list):
        return ""
    for step in steps:
        if not isinstance(step, dict):
            continue
        if str(step.get("src") or "").lower() != "agent":
            continue
        text = trajectory_profile._content_text(step.get("msg")).strip()
        if text:
            return text
    return ""


def _find_agent_race_message(row: dict[str, Any], event_type: str) -> str:
    raw_events = trajectory_profile._deserialize_possible_json(row.get("events"))
    if not isinstance(raw_events, list):
        return ""
    for event in raw_events:
        if not isinstance(event, dict):
            continue
        message = event.get("message")
        if isinstance(message, dict):
            message_role = str(message.get("role") or "").lower()
            raw_event_type = str(event.get("type") or "").lower()
            if message_role != event_type and raw_event_type != event_type:
                continue
            text = _stringify_message_content(message.get("content"))
            if text.strip():
                return text.strip()
        elif str(event.get("type") or "").lower() != event_type:
            continue
        tool_result = event.get("toolUseResult")
        if isinstance(tool_result, dict):
            text = _stringify_message_content(tool_result.get("content"))
            if text.strip():
                return text.strip()
    return ""


def _task_label(row: dict[str, Any]) -> str:
    for key in ("task_category", "task_name"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _model_or_harness_label(row: dict[str, Any]) -> str:
    for key in ("model_name", "recorded_model", "model", "agent", "harness"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    source_file = row.get("source_file")
    if isinstance(source_file, str) and source_file.strip():
        return Path(source_file).stem
    return ""


def _prompt_preview(adapter: str, row: dict[str, Any]) -> str:
    direct_prompt = row.get("prompt")
    if isinstance(direct_prompt, str) and direct_prompt.strip():
        return _truncate(direct_prompt.strip())
    if adapter in {"openhands", "cc_bench", "trace_commons", "thoughtworks"}:
        text = _find_openhands_message(row, "user")
        if text:
            return _truncate(text)
    if adapter == "terminalbench":
        text = _find_terminalbench_prompt(row)
        if text:
            return _truncate(text)
    if adapter == "agent_race":
        text = _find_agent_race_message(row, "user")
        if text:
            return _truncate(text)
    task_name = row.get("task_name")
    if isinstance(task_name, str) and task_name.strip():
        return _truncate(task_name.strip())
    return ""


def _assistant_preview(adapter: str, row: dict[str, Any]) -> str:
    if adapter in {"openhands", "cc_bench", "trace_commons", "thoughtworks"}:
        text = _find_openhands_message(row, "assistant")
        if text:
            return _truncate(text)
    if adapter == "terminalbench":
        text = _find_terminalbench_assistant(row)
        if text:
            return _truncate(text)
    if adapter == "agent_race":
        text = _find_agent_race_message(row, "assistant")
        if text:
            return _truncate(text)
    return ""


def _tool_preview(adapter: str, row: dict[str, Any], limit: int = 6) -> list[str]:
    names: list[str] = []
    for event in trajectory_profile._extract_events(adapter, row):
        if event.kind != "tool_call":
            continue
        names.append(event.name)
        if len(names) >= limit:
            break
    if names:
        return names
    if adapter in {"trace_commons", "thoughtworks", "openhands", "cc_bench"}:
        return _trace_fallback_tool_preview(row, limit=limit)
    return names


def _sample_priority(
    adapter: str,
    *,
    prompt_preview: str,
    assistant_preview: str,
    tool_preview: list[str],
) -> tuple[int, int, int, int]:
    prompt = prompt_preview.strip()
    assistant = assistant_preview.strip()
    normalized_prompt = prompt.lower()
    warmup_penalty = 1 if adapter == "terminalbench" and normalized_prompt == "warmup" else 0
    return (
        -warmup_penalty,
        1 if prompt else 0,
        1 if assistant else 0,
        len(tool_preview),
    )


def _sample_diversity_key(sample: dict[str, Any]) -> tuple[str, str]:
    prompt_key = str(sample.get("prompt_preview") or "").strip().lower()
    assistant_key = str(sample.get("assistant_preview") or "").strip().lower()
    return prompt_key, assistant_key


def _sample_is_sparse(sample: dict[str, Any]) -> bool:
    return not (
        str(sample.get("prompt_preview") or "").strip()
        or str(sample.get("assistant_preview") or "").strip()
        or bool(sample.get("tool_preview"))
    )


def _iter_dataset_rows(data_root: Path, dataset: str, max_rows: int | None) -> tuple[str, Iterable[dict[str, Any]]]:
    adapter, paths = trajectory_profile._resolve_dataset_paths(data_root, dataset)
    if adapter == "trace_commons":
        def _iter_trace_commons_rows() -> Iterable[dict[str, Any]]:
            emitted = 0
            for row in trajectory_profile._iter_parquet_rows(paths):
                has_messages = trajectory_profile._row_has_trajectory_content(adapter, row)
                has_trace = bool(trajectory_profile._deserialize_possible_json(row.get("trace")))
                if not has_messages and not has_trace:
                    continue
                yield row
                emitted += 1
                if max_rows is not None and emitted >= max_rows:
                    return

        return adapter, _iter_trace_commons_rows()
    return trajectory_profile._iter_dataset_rows(data_root, dataset, max_rows)


def sample_dataset(
    data_root: Path,
    dataset: str,
    *,
    max_rows: int | None,
    samples_per_dataset: int,
) -> dict[str, Any]:
    adapter, rows = _iter_dataset_rows(data_root, dataset, max_rows)
    task_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    candidates: list[tuple[tuple[int, int, int, int], int, dict[str, Any]]] = []
    rows_scanned = 0
    for row in rows:
        rows_scanned += 1
        task = _task_label(row)
        if task:
            task_counts[task] += 1
        model_or_harness = _model_or_harness_label(row)
        if model_or_harness:
            model_counts[model_or_harness] += 1
        prompt_preview = _prompt_preview(adapter, row)
        assistant_preview = _assistant_preview(adapter, row)
        tool_preview = _tool_preview(adapter, row)
        candidates.append(
            (
                _sample_priority(
                    adapter,
                    prompt_preview=prompt_preview,
                    assistant_preview=assistant_preview,
                    tool_preview=tool_preview,
                ),
                rows_scanned,
                {
                    "task": task,
                    "prompt_preview": prompt_preview,
                    "assistant_preview": assistant_preview,
                    "tool_preview": tool_preview,
                },
            )
        )
    sorted_candidates = sorted(candidates, key=lambda item: (item[0], -item[1]), reverse=True)
    samples: list[dict[str, Any]] = []
    seen_task_keys: set[str] = set()
    seen_shape_keys: set[tuple[str, str]] = set()
    deferred: list[dict[str, Any]] = []
    sparse: list[dict[str, Any]] = []
    for _, _, sample in sorted_candidates:
        if _sample_is_sparse(sample):
            sparse.append(sample)
            continue
        task_key = str(sample.get("task") or "").strip().lower()
        shape_key = _sample_diversity_key(sample)
        if shape_key != ("", "") and shape_key in seen_shape_keys:
            deferred.append(sample)
            continue
        if task_key and task_key in seen_task_keys:
            deferred.append(sample)
            continue
        if task_key:
            seen_task_keys.add(task_key)
        if shape_key != ("", ""):
            seen_shape_keys.add(shape_key)
        samples.append(sample)
        if len(samples) >= samples_per_dataset:
            break
    if len(samples) < samples_per_dataset:
        for sample in deferred:
            shape_key = _sample_diversity_key(sample)
            if shape_key != ("", "") and shape_key not in seen_shape_keys:
                samples.append(sample)
                seen_shape_keys.add(shape_key)
            if len(samples) >= samples_per_dataset:
                break
    if len(samples) < samples_per_dataset:
        for sample in deferred:
            shape_key = _sample_diversity_key(sample)
            if shape_key != ("", "") and shape_key not in seen_shape_keys:
                continue
            samples.append(sample)
            if len(samples) >= samples_per_dataset:
                break
    if len(samples) < samples_per_dataset:
        for sample in sparse:
            samples.append(sample)
            if len(samples) >= samples_per_dataset:
                break
    return {
        "dataset": dataset,
        "adapter": adapter,
        "rows_scanned": rows_scanned,
        "top_tasks": [{"name": name, "count": count} for name, count in task_counts.most_common(8)],
        "top_models_or_harnesses": [{"name": name, "count": count} for name, count in model_counts.most_common(8)],
        "samples": samples,
    }


def build_report(
    *,
    data_root: Path,
    datasets: list[str],
    max_rows: int | None,
    samples_per_dataset: int,
) -> dict[str, Any]:
    summaries = [
        sample_dataset(
            data_root,
            dataset,
            max_rows=max_rows,
            samples_per_dataset=samples_per_dataset,
        )
        for dataset in datasets
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root.resolve(strict=False)),
        "max_rows_per_dataset": max_rows,
        "samples_per_dataset": samples_per_dataset,
        "datasets": summaries,
    }


def _format_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Transcript Sampler", ""]
    for summary in payload.get("datasets", []):
        if not isinstance(summary, dict):
            continue
        lines.append(f"## {summary.get('dataset')}")
        lines.append("")
        lines.append(
            f"- adapter: `{summary.get('adapter')}`"
        )
        lines.append(
            f"- rows_scanned: `{summary.get('rows_scanned')}`"
        )
        top_tasks = summary.get("top_tasks") or []
        if top_tasks:
            lines.append(
                "- top_tasks: "
                + ", ".join(f"`{item['name']}` ({item['count']})" for item in top_tasks[:6] if isinstance(item, dict))
            )
        top_models = summary.get("top_models_or_harnesses") or []
        if top_models:
            lines.append(
                "- top_models_or_harnesses: "
                + ", ".join(f"`{item['name']}` ({item['count']})" for item in top_models[:6] if isinstance(item, dict))
            )
        lines.append("")
        for index, sample in enumerate(summary.get("samples", []), start=1):
            if not isinstance(sample, dict):
                continue
            lines.append(f"### Sample {index}")
            lines.append("")
            if sample.get("task"):
                lines.append(f"- task: `{sample['task']}`")
            lines.append(f"- prompt: {sample.get('prompt_preview') or '(none)'}")
            if sample.get("assistant_preview"):
                lines.append(f"- assistant: {sample['assistant_preview']}")
            tool_preview = sample.get("tool_preview") or []
            if tool_preview:
                lines.append("- tools: " + ", ".join(f"`{name}`" for name in tool_preview))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _resolve_output_paths(
    *,
    output: Path | None,
    output_json: Path | None,
    output_md: Path | None,
) -> tuple[Path, Path]:
    if output is None:
        return output_json or DEFAULT_OUTPUT_JSON, output_md or DEFAULT_OUTPUT_MD
    suffix = output.suffix.lower()
    if suffix == ".json":
        return output_json or output, output_md or output.with_suffix(".md")
    if suffix == ".md":
        return output_json or output.with_suffix(".json"), output_md or output
    return output_json or output.with_suffix(".json"), output_md or output.with_suffix(".md")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample representative coding-agent transcript prompts and first actions.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root directory containing local trajectory datasets.")
    parser.add_argument("--datasets", nargs="+", choices=sorted(trajectory_profile.DATASET_SPECS), default=sorted(trajectory_profile.DATASET_SPECS))
    parser.add_argument("--max-rows", type=int, default=100, help="Maximum rows to scan per dataset before building counters.")
    parser.add_argument("--samples-per-dataset", type=int, default=3, help="Number of sample rows to emit per dataset.")
    parser.add_argument("--output", type=Path, default=None, help="Base output path or explicit .json/.md path; writes both JSON and Markdown siblings.")
    parser.add_argument("--output-json", type=Path, default=None, help="JSON output path.")
    parser.add_argument("--output-md", type=Path, default=None, help="Markdown output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    output_json, output_md = _resolve_output_paths(
        output=args.output,
        output_json=args.output_json,
        output_md=args.output_md,
    )

    max_rows = None if args.max_rows == 0 else args.max_rows
    payload = build_report(
        data_root=args.data_root,
        datasets=list(args.datasets),
        max_rows=max_rows,
        samples_per_dataset=args.samples_per_dataset,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_format_markdown(payload), encoding="utf-8")
    for summary in payload["datasets"]:
        print(
            "[trajectory-transcript-sampler] "
            + f"dataset={summary['dataset']} "
            + f"rows={summary['rows_scanned']} "
            + f"samples={len(summary['samples'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
