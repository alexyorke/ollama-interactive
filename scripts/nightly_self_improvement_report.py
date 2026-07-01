from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import live_model_gate, trajectory_dataset_fetch, trajectory_profile


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_value(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_file_is_object(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict)


def _run_command(repo_root: Path, name: str, command: list[str], timeout: int, output_path: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, timeout=timeout, check=False)
        elapsed_s = round(time.perf_counter() - started, 3)
        output = "\n".join(part for part in (completed.stdout.strip(), completed.stderr.strip()) if part)
        return {
            "name": name,
            "command": command,
            "exit_code": completed.returncode,
            "elapsed_s": elapsed_s,
            "ok": completed.returncode == 0,
            "output_path": str(output_path) if output_path else None,
            "output_tail": output[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        elapsed_s = round(time.perf_counter() - started, 3)
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        output = "\n".join(part for part in (stdout.strip(), stderr.strip()) if part)
        return {
            "name": name,
            "command": command,
            "exit_code": None,
            "elapsed_s": elapsed_s,
            "ok": False,
            "timed_out": True,
            "output_path": str(output_path) if output_path else None,
            "output_tail": output[-4000:],
        }


def _run_command_with_env(
    repo_root: Path,
    name: str,
    command: list[str],
    timeout: int,
    output_path: Path | None = None,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    try:
        completed = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, timeout=timeout, check=False, env=env)
        elapsed_s = round(time.perf_counter() - started, 3)
        output = "\n".join(part for part in (completed.stdout.strip(), completed.stderr.strip()) if part)
        return {
            "name": name,
            "command": command,
            "exit_code": completed.returncode,
            "elapsed_s": elapsed_s,
            "ok": completed.returncode == 0,
            "output_path": str(output_path) if output_path else None,
            "output_tail": output[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        elapsed_s = round(time.perf_counter() - started, 3)
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        output = "\n".join(part for part in (stdout.strip(), stderr.strip()) if part)
        return {
            "name": name,
            "command": command,
            "exit_code": None,
            "elapsed_s": elapsed_s,
            "ok": False,
            "timed_out": True,
            "output_path": str(output_path) if output_path else None,
            "output_tail": output[-4000:],
        }


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary")
    return dict(summary) if isinstance(summary, dict) else {}


def _status_delta(current: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    current_summary = _summary(current)
    baseline_summary = _summary(baseline or {})
    keys = ("runs", "pass", "fail_closed", "fail", "skip", "total_llm_calls", "total_tokens", "median_total_tokens")
    delta: dict[str, Any] = {}
    for key in keys:
        current_value = current_summary.get(key)
        baseline_value = baseline_summary.get(key)
        if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
            delta[key] = current_value - baseline_value
    return delta


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _collect_tool_totals(*payloads: dict[str, Any]) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        for result in payload.get("results", []):
            if not isinstance(result, dict):
                continue
            profile = result.get("tool_profile")
            if not isinstance(profile, dict):
                continue
            by_tool = profile.get("by_tool")
            if not isinstance(by_tool, dict):
                continue
            for tool_name, item in by_tool.items():
                if not isinstance(item, dict):
                    continue
                bucket = totals.setdefault(
                    str(tool_name),
                    {"tool": str(tool_name), "calls": 0, "duration_ms": 0.0, "failed": 0, "cached": 0},
                )
                calls = _safe_int(item.get("calls", 0) or 0)
                duration_ms = _safe_float(item.get("duration_ms", 0.0) or 0.0)
                failed = _safe_int(item.get("failed", 0) or 0)
                cached = _safe_int(item.get("cached", 0) or 0)
                bucket["calls"] += calls or 0
                bucket["duration_ms"] += duration_ms or 0.0
                bucket["failed"] += failed or 0
                bucket["cached"] += cached or 0
    rows = [
        {**item, "duration_ms": round(float(item["duration_ms"]), 3)}
        for item in totals.values()
        if float(item["duration_ms"]) > 0 or int(item["calls"]) > 0
    ]
    return sorted(rows, key=lambda item: (float(item["duration_ms"]), int(item["calls"])), reverse=True)


def _collect_probe_slowest(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return []
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        elapsed_ms = _safe_float(row.get("elapsed_ms", 0.0) or 0.0)
        if elapsed_ms is None:
            continue
        normalized.append({**row, "elapsed_ms": elapsed_ms})
    return sorted(normalized, key=lambda row: float(row.get("elapsed_ms", 0.0) or 0.0), reverse=True)


def _question_quality_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return {"cases": 0, "passed": 0, "failed": 0}
    cases = _safe_int(summary.get("cases", 0) or 0)
    passed = _safe_int(summary.get("passed", 0) or 0)
    failed = _safe_int(summary.get("failed", 0) or 0)
    return {
        "cases": cases or 0,
        "passed": passed or 0,
        "failed": failed or 0,
    }


def _trajectory_profile_summary(payload: dict[str, Any], *, expected_datasets: list[str]) -> dict[str, Any]:
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        return {
            "available": False,
            "expected_datasets": expected_datasets,
            "rows_profiled": 0,
            "datasets_profiled": 0,
            "top_recommendations": [],
        }
    rows_profiled = 0
    for item in datasets:
        if not isinstance(item, dict):
            continue
        parsed = _safe_int(item.get("rows_profiled", 0) or 0)
        if parsed is None:
            continue
        rows_profiled += parsed
    portfolio = payload.get("portfolio_recommendations")
    top_recommendations = []
    if isinstance(portfolio, list):
        top_recommendations = [
            str(item.get("id"))
            for item in portfolio
            if isinstance(item, dict) and str(item.get("id") or "").strip()
        ][:5]
    return {
        "available": True,
        "expected_datasets": expected_datasets,
        "rows_profiled": rows_profiled,
        "datasets_profiled": len([item for item in datasets if isinstance(item, dict)]),
        "top_recommendations": top_recommendations,
    }


def _trajectory_error_summary(payload: dict[str, Any], *, expected_datasets: list[str]) -> dict[str, Any]:
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        return {"available": False, "expected_datasets": expected_datasets, "top_errors": []}
    counts: dict[str, int] = {}
    for item in datasets:
        if not isinstance(item, dict):
            continue
        error_counts = item.get("error_counts")
        if not isinstance(error_counts, dict):
            continue
        for name, value in error_counts.items():
            parsed = _safe_int(value or 0)
            if parsed is None:
                continue
            counts[str(name)] = counts.get(str(name), 0) + parsed
    top = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)[:8]
    return {
        "available": True,
        "expected_datasets": expected_datasets,
        "top_errors": [{"name": name, "count": count} for name, count in top],
    }


def _trajectory_evidence_summary(payload: dict[str, Any], *, expected_datasets: list[str]) -> dict[str, Any]:
    coverage = payload.get("portfolio_fix_coverage")
    if not isinstance(coverage, list):
        return {"available": False, "expected_datasets": expected_datasets, "top_fix_coverage": []}
    top_fix_coverage = []
    for item in coverage[:6]:
        if not isinstance(item, dict):
            continue
        evidence_count = _safe_int(item.get("evidence_count", 0) or 0)
        if evidence_count is None:
            continue
        top_fix_coverage.append(
            {
                "id": str(item.get("id") or ""),
                "evidence_count": evidence_count,
                "status": str(item.get("status") or ""),
            }
        )
    return {"available": True, "expected_datasets": expected_datasets, "top_fix_coverage": top_fix_coverage}


def _trajectory_catalog_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return {
            "entries": 0,
            "local_entries": 0,
            "analysis_ready_local_entries": 0,
            "public_missing_entries": 0,
            "gated_entries": 0,
            "high_priority_public_missing": [],
        }
    entries = _safe_int(summary.get("entries", 0) or 0)
    local_entries = _safe_int(summary.get("local_entries", 0) or 0)
    analysis_ready_local_entries = _safe_int(summary.get("analysis_ready_local_entries", 0) or 0)
    public_missing_entries = _safe_int(summary.get("public_missing_entries", 0) or 0)
    gated_entries = _safe_int(summary.get("gated_entries", 0) or 0)
    return {
        "entries": entries or 0,
        "local_entries": local_entries or 0,
        "analysis_ready_local_entries": analysis_ready_local_entries or 0,
        "public_missing_entries": public_missing_entries or 0,
        "gated_entries": gated_entries or 0,
        "high_priority_public_missing": list(summary.get("high_priority_public_missing") or []),
    }


def _trajectory_local_manifest_summary(data_root: Path, datasets: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        manifest = trajectory_dataset_fetch.read_dataset_manifest(data_root, dataset)
        if not manifest:
            continue
        rows.append(
            {
                "dataset": dataset,
                "repo_id": str(manifest.get("repo_id") or ""),
                "resolved_revision": str(manifest.get("resolved_revision") or ""),
                "downloaded_at": str(manifest.get("downloaded_at") or ""),
                "file_count": _safe_int(manifest.get("file_count", 0) or 0) or 0,
            }
        )
    return rows


def _latest_live_gate_summary_path(repo_root: Path) -> Path | None:
    candidates: list[tuple[float, Path]] = []
    scratch_root = repo_root / "scratch"
    if not scratch_root.exists():
        return None
    for path in scratch_root.glob("live-model-gate*/live-model-gate-summary.json"):
        payload = _load_json(path)
        if not live_model_gate.summary_contract_ok(payload):
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _live_model_gate_summary(payload: dict[str, Any], *, path: Path | None) -> dict[str, Any]:
    if not payload or not live_model_gate.summary_contract_ok(payload):
        return {
            "available": False,
            "path": str(path) if path else None,
            "ok": None,
            "benchmark_suite": None,
            "selected_default_model": None,
            "selection_reason": None,
            "models": [],
        }
    models = payload.get("models")
    model_rows = list(models) if isinstance(models, list) else []
    selected_default_model = payload.get("selected_default_model")
    selection_reason = payload.get("selection_reason")
    if not isinstance(selected_default_model, str) or not selected_default_model.strip():
        selected_default_model = None
    if not isinstance(selection_reason, str) or not selection_reason.strip():
        selection_reason = None
    return {
        "available": True,
        "path": str(path) if path else None,
        "ok": payload.get("ok"),
        "benchmark_suite": payload.get("benchmark_suite"),
        "selected_default_model": selected_default_model,
        "selection_reason": selection_reason,
        "models": model_rows,
    }


def _available_trajectory_datasets(data_root: Path) -> list[str]:
    available: list[str] = []
    for dataset_name in trajectory_profile.DATASET_SPECS:
        manifest = trajectory_dataset_fetch.read_dataset_manifest(data_root, dataset_name)
        if not isinstance(manifest, dict):
            continue
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            continue
        dataset_dir = data_root / dataset_name
        existing_files = [
            relative
            for relative in files
            if isinstance(relative, str) and relative.strip() and (dataset_dir / relative).is_file()
        ]
        if not existing_files:
            continue
        available.append(dataset_name)
    return available


def _run_dir_timestamp(path: Path) -> str | None:
    name = path.name.strip()
    if not re.fullmatch(r"\d{8}T\d{6}Z", name):
        legacy = re.fullmatch(r"(?P<day>\d{8})-(?P<clock>\d{6})", name)
        if legacy is None:
            return None
        return f"{legacy.group('day')}T{legacy.group('clock')}Z"
    return name


def _report_roots(repo_root: Path) -> tuple[Path, ...]:
    return (
        repo_root / "scratch" / "nightly-self-improvement",
        repo_root / ".ollama-code" / "self-improvement-runs",
    )


def _resolved_compare_path(repo_root: Path, current_run_dir: Path, requested: Path | None) -> Path | None:
    candidate = requested or _default_compare_path(repo_root, current_run_dir)
    if candidate is None:
        return None
    return candidate if _json_file_is_object(candidate) else None


def _default_compare_path(repo_root: Path, current_run_dir: Path) -> Path | None:
    timestamped: list[tuple[str, Path]] = []
    fallback: list[tuple[float, Path]] = []
    current_timestamp = _run_dir_timestamp(current_run_dir)
    for root in _report_roots(repo_root):
        if not root.exists():
            continue
        for path in root.glob("*/report.json"):
            if current_run_dir in path.parents:
                continue
            if not _json_file_is_object(path):
                continue
            run_dir = path.parent
            run_timestamp = _run_dir_timestamp(run_dir)
            if run_timestamp is not None:
                timestamped.append((run_timestamp, path))
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            fallback.append((mtime, path))
    if timestamped:
        if current_timestamp is not None:
            prior = [item for item in timestamped if item[0] < current_timestamp]
            if prior:
                prior.sort(key=lambda item: item[0], reverse=True)
                return prior[0][1]
            return None
        timestamped.sort(key=lambda item: item[0], reverse=True)
        return timestamped[0][1]
    if fallback:
        fallback.sort(key=lambda item: item[0], reverse=True)
        return fallback[0][1]
    return None


def _normalize_ollama_host(host: str) -> str:
    value = host.strip()
    if not value:
        return ""
    if not value.startswith(("http://", "https://")):
        value = f"http://{value}"
    return value.rstrip("/")


def _host_responds(host: str) -> bool:
    target = _normalize_ollama_host(host)
    if not target:
        return False
    request = urllib.request.Request(f"{target}/api/tags")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def _resolve_ollama_host() -> str | None:
    candidates = [
        os.environ.get("OLLAMA_HOST", ""),
        "http://127.0.0.1:11434",
        "http://[::1]:11434",
        "http://localhost:11434",
    ]
    seen: set[str] = set()
    for candidate in candidates:
        normalized = _normalize_ollama_host(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if _host_responds(normalized):
            return normalized
    return None


def _suggest_targets(metrics: dict[str, Any], commands: list[dict[str, Any]]) -> list[str]:
    suggestions: list[str] = []
    failed_commands = [item["name"] for item in commands if not item.get("ok")]
    if failed_commands:
        suggestions.append("Fix or shrink failing nightly command(s): " + ", ".join(failed_commands[:3]))
    question_quality = metrics.get("question_quality")
    question_quality_failed = _safe_int(question_quality.get("failed", 0) or 0) if isinstance(question_quality, dict) else None
    if question_quality_failed:
        suggestions.append("Improve clarification-question quality before expanding autonomy or rewrite scope.")
    for key in ("token_efficiency", "coding_benchmark"):
        section = metrics.get(key)
        if not isinstance(section, dict):
            continue
        summary = section.get("summary") if isinstance(section.get("summary"), dict) else {}
        summary_fail = _safe_int(summary.get("fail", 0) or 0) or 0
        summary_fail_closed = _safe_int(summary.get("fail_closed", 0) or 0) or 0
        if summary_fail or summary_fail_closed:
            suggestions.append(f"Inspect failed {key} cases before adding new features.")
        delta = section.get("delta") if isinstance(section.get("delta"), dict) else {}
        delta_total_tokens = _safe_float(delta.get("total_tokens", 0) or 0) or 0.0
        if delta_total_tokens > 0:
            suggestions.append(f"Reduce token growth in {key}; compare per-case prompt profiles.")
    slow_tools = metrics.get("slowest_tools")
    if isinstance(slow_tools, list) and slow_tools:
        first = slow_tools[0]
        if isinstance(first, dict) and first.get("tool"):
            suggestions.append(f"Profile deterministic tool path for {first['tool']}; it is the slowest recurring tool.")
    slow_probe_tools = metrics.get("slowest_probe_tools")
    if isinstance(slow_probe_tools, list) and slow_probe_tools:
        first = slow_probe_tools[0]
        if isinstance(first, dict) and first.get("name"):
            suggestions.append(f"Optimize no-LLM tool probe {first['name']} before model prompt changes.")
    probe = metrics.get("tool_speed_probe")
    if isinstance(probe, dict):
        slowest_probe = probe.get("slowest")
        if isinstance(slowest_probe, list) and slowest_probe:
            first = slowest_probe[0]
            if isinstance(first, dict) and first.get("name"):
                suggestions.append(f"Optimize no-LLM tool probe {first['name']} before model prompt changes.")
    sdk = metrics.get("python_sdk_search")
    if isinstance(sdk, dict):
        summary = sdk.get("summary") if isinstance(sdk.get("summary"), dict) else {}
        sdk_fail = _safe_int(summary.get("fail", 0) or 0) or 0
        if sdk_fail:
            suggestions.append("Improve Python SDK retrieval before relying on it for agent routing.")
    trajectory_profile_summary = metrics.get("trajectory_profile")
    if isinstance(trajectory_profile_summary, dict):
        top_recommendations = trajectory_profile_summary.get("top_recommendations")
        if isinstance(top_recommendations, list) and top_recommendations:
            suggestions.append("Continue trajectory-driven controller work around: " + ", ".join(str(item) for item in top_recommendations[:3]))
    trajectory_evidence = metrics.get("trajectory_evidence_report")
    if isinstance(trajectory_evidence, dict):
        top_fix_coverage = trajectory_evidence.get("top_fix_coverage")
        if isinstance(top_fix_coverage, list):
            partial = [str(item.get("id")) for item in top_fix_coverage if isinstance(item, dict) and str(item.get("status") or "") == "partial"]
            if partial:
                suggestions.append("Extract more monolith logic behind existing trajectory fixes: " + ", ".join(partial[:3]))
    dataset_catalog = metrics.get("trajectory_dataset_catalog")
    if isinstance(dataset_catalog, dict):
        missing = dataset_catalog.get("high_priority_public_missing")
        if isinstance(missing, list) and missing:
            first = missing[0] if isinstance(missing[0], dict) else {}
            repo_id = str(first.get("id") or "").strip()
            if repo_id:
                suggestions.append(f"Download or sample the next public trajectory candidate: {repo_id}.")
    return list(dict.fromkeys(suggestions))[:6]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = _repo_root()
    run_dir = args.output_dir or (repo_root / "scratch" / "nightly-self-improvement" / _timestamp())
    run_dir = run_dir.resolve(strict=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    compare_path = _resolved_compare_path(repo_root, run_dir, args.compare)
    available_trajectory_datasets = _available_trajectory_datasets(args.trajectory_data_root)
    resolved_ollama_host = None if args.skip_llm else _resolve_ollama_host()
    llm_skip_reason = ""
    trajectory_skip_reason = ""
    if args.skip_llm:
        llm_skip_reason = "LLM commands were skipped because --skip-llm was requested."
    elif not resolved_ollama_host:
        llm_skip_reason = "No reachable Ollama host found; skipped token-efficiency and coding-benchmark commands."
    if args.skip_trajectories:
        trajectory_skip_reason = "Trajectory commands were skipped because --skip-trajectories was requested."
    elif not available_trajectory_datasets:
        trajectory_skip_reason = f"No local trajectory datasets found under {args.trajectory_data_root}; skipped trajectory profile, error, and evidence commands."

    commands: list[dict[str, Any]] = []
    py = sys.executable

    tool_probe_path = run_dir / "tool_speed.json"
    commands.append(
        _run_command(
            repo_root,
            "tool_speed_probe",
            [py, "scripts/tool_speed_probe.py", "--generated-files", str(args.generated_files), "--output", str(tool_probe_path)],
            timeout=args.tool_timeout,
            output_path=tool_probe_path,
        )
    )

    sdk_eval_path = run_dir / "python_sdk_search.json"
    sdk_command = [
        py,
        "scripts/python_sdk_search_eval.py",
        "--workspace",
        str(repo_root),
        "--index-limit",
        str(args.sdk_index_limit),
        "--limit",
        "8",
        "--output",
        str(sdk_eval_path),
    ]
    if args.sdk_use_embeddings:
        sdk_command.append("--use-embeddings")
        if args.sdk_embedding_model:
            sdk_command.extend(["--embedding-model", args.sdk_embedding_model])
    if args.strict_accuracy:
        sdk_command.append("--strict-accuracy")
    commands.append(_run_command(repo_root, "python_sdk_search", sdk_command, timeout=args.tool_timeout, output_path=sdk_eval_path))

    question_quality_path = run_dir / "question-quality.json"
    commands.append(
        _run_command(
            repo_root,
            "question_quality",
            [
                py,
                "scripts/question_quality_eval.py",
                "--output-json",
                str(question_quality_path),
                "--output-md",
                str(run_dir / "question-quality.md"),
            ],
            timeout=args.tool_timeout,
            output_path=question_quality_path,
        )
    )

    trajectory_catalog_path = run_dir / "trajectory-dataset-catalog.json"
    commands.append(
        _run_command(
            repo_root,
            "trajectory_dataset_catalog",
            [
                py,
                "scripts/trajectory_dataset_catalog.py",
                "--data-root",
                str(args.trajectory_data_root),
                "--output",
                str(trajectory_catalog_path),
            ],
            timeout=args.tool_timeout,
            output_path=trajectory_catalog_path,
        )
    )

    trajectory_profile_path = run_dir / "trajectory-profile.json"
    trajectory_error_path = run_dir / "trajectory-error-profile.json"
    trajectory_evidence_path = run_dir / "trajectory-evidence-report.json"
    if not args.skip_trajectories and available_trajectory_datasets:
        max_rows_arg = str(args.trajectory_max_rows)
        commands.append(
            _run_command(
                repo_root,
                "trajectory_profile",
                [
                    py,
                    "scripts/trajectory_profile.py",
                    "--data-root",
                    str(args.trajectory_data_root),
                    "--datasets",
                    *available_trajectory_datasets,
                    "--max-rows",
                    max_rows_arg,
                    "--output",
                    str(trajectory_profile_path),
                ],
                timeout=args.tool_timeout,
                output_path=trajectory_profile_path,
            )
        )
        commands.append(
            _run_command(
                repo_root,
                "trajectory_error_profile",
                [
                    py,
                    "scripts/trajectory_error_profile.py",
                    "--data-root",
                    str(args.trajectory_data_root),
                    "--datasets",
                    *available_trajectory_datasets,
                    "--max-rows",
                    max_rows_arg,
                    "--output",
                    str(trajectory_error_path),
                ],
                timeout=args.tool_timeout,
                output_path=trajectory_error_path,
            )
        )
        commands.append(
            _run_command(
                repo_root,
                "trajectory_evidence_report",
                [
                    py,
                    "scripts/trajectory_evidence_report.py",
                    "--data-root",
                    str(args.trajectory_data_root),
                    "--datasets",
                    *available_trajectory_datasets,
                    "--max-rows",
                    max_rows_arg,
                    "--output-json",
                    str(trajectory_evidence_path),
                    "--output-md",
                    str(run_dir / "trajectory-evidence-report.md"),
                ],
                timeout=args.tool_timeout,
                output_path=trajectory_evidence_path,
            )
        )

    token_path = run_dir / "token_efficiency.json"
    benchmark_path = run_dir / "coding_benchmark.json"
    if not args.skip_llm:
        if resolved_ollama_host:
            llm_env = {"OLLAMA_HOST": resolved_ollama_host}
            token_command = [
                py,
                "scripts/token_efficiency_eval.py",
                "--models",
                *args.models,
                "--modes",
                "off",
                "--feature-profiles",
                "all",
                "--timeout",
                str(args.case_timeout),
                "--output",
                str(token_path),
            ]
            if args.strict_accuracy:
                token_command.append("--strict-accuracy")
            commands.append(
                _run_command_with_env(
                    repo_root,
                    "token_efficiency",
                    token_command,
                    timeout=args.command_timeout,
                    output_path=token_path,
                    env_overrides=llm_env,
                )
            )

            benchmark_command = [
                py,
                "scripts/coding_benchmark_eval.py",
                "--suite",
                args.suite,
                "--models",
                *args.models,
                "--modes",
                "off",
                "--reconcile-modes",
                "auto",
                "--feature-profiles",
                "all",
                "--jobs",
                "1",
                "--timeout",
                str(args.case_timeout),
                "--output",
                str(benchmark_path),
                "--require-llm-for-agent-benchmarks",
            ]
            if args.strict_accuracy:
                benchmark_command.append("--strict-accuracy")
            if args.strict_budget:
                benchmark_command.append("--strict-budget")
            commands.append(
                _run_command_with_env(
                    repo_root,
                    "coding_benchmark",
                    benchmark_command,
                    timeout=args.command_timeout,
                    output_path=benchmark_path,
                    env_overrides=llm_env,
                )
            )

    commands.append(_run_command(repo_root, "anti_cheat_scan", [py, "scripts/anti_cheat_scan.py"], timeout=args.tool_timeout))

    token_payload = _load_json(token_path)
    benchmark_payload = _load_json(benchmark_path)
    probe_payload = _load_json(tool_probe_path)
    sdk_payload = _load_json(sdk_eval_path)
    question_quality_payload = _load_json(question_quality_path)
    trajectory_catalog_payload = _load_json(trajectory_catalog_path)
    trajectory_profile_payload = _load_json(trajectory_profile_path)
    trajectory_error_payload = _load_json(trajectory_error_path)
    trajectory_evidence_payload = _load_json(trajectory_evidence_path)
    live_gate_summary_path = _latest_live_gate_summary_path(repo_root)
    live_gate_payload = _load_json(live_gate_summary_path) if live_gate_summary_path else {}
    baseline = _load_json(compare_path) if compare_path else {}
    baseline_metrics = baseline.get("metrics") if isinstance(baseline.get("metrics"), dict) else {}
    baseline_token = baseline_metrics.get("token_efficiency") if isinstance(baseline_metrics.get("token_efficiency"), dict) else {}
    baseline_benchmark = baseline_metrics.get("coding_benchmark") if isinstance(baseline_metrics.get("coding_benchmark"), dict) else {}
    slowest_probe_tools = _collect_probe_slowest(probe_payload)[:10]
    live_gate_metric = _live_model_gate_summary(live_gate_payload, path=live_gate_summary_path)

    metrics: dict[str, Any] = {
        "tool_speed_probe": {
            "generated_files": probe_payload.get("generated_files"),
            "slowest": slowest_probe_tools[:5],
        },
        "python_sdk_search": {
            "summary": _summary(sdk_payload),
            "refresh": sdk_payload.get("refresh", {}),
            "mode": sdk_payload.get("mode"),
        },
        "question_quality": _question_quality_summary(question_quality_payload),
        "trajectory_dataset_catalog": {
            **_trajectory_catalog_summary(trajectory_catalog_payload),
            "local_manifests": _trajectory_local_manifest_summary(args.trajectory_data_root, available_trajectory_datasets),
        },
        "trajectory_profile": _trajectory_profile_summary(
            trajectory_profile_payload, expected_datasets=available_trajectory_datasets
        ),
        "trajectory_error_profile": _trajectory_error_summary(
            trajectory_error_payload, expected_datasets=available_trajectory_datasets
        ),
        "trajectory_evidence_report": _trajectory_evidence_summary(
            trajectory_evidence_payload, expected_datasets=available_trajectory_datasets
        ),
        "live_model_gate": live_gate_metric,
        "token_efficiency": {
            "summary": _summary(token_payload),
            "delta": _status_delta(token_payload, baseline_token),
            "accuracy_regressions": token_payload.get("accuracy_regressions", []),
        },
        "coding_benchmark": {
            "summary": _summary(benchmark_payload),
            "delta": _status_delta(benchmark_payload, baseline_benchmark),
            "accuracy_regressions": benchmark_payload.get("accuracy_regressions", []),
            "budget_failures": benchmark_payload.get("budget_failures", []),
            "llm_bypass_failures": benchmark_payload.get("llm_bypass_failures", []),
        },
        "slowest_tools": _collect_tool_totals(token_payload, benchmark_payload)[:10],
        "slowest_probe_tools": slowest_probe_tools,
    }
    metrics["suggested_implementation_targets"] = _suggest_targets(metrics, commands)
    summary = {
        "selected_default_model": live_gate_metric.get("selected_default_model"),
        "live_gate": {
            "available": live_gate_metric.get("available", False),
            "ok": live_gate_metric.get("ok"),
            "benchmark_suite": live_gate_metric.get("benchmark_suite"),
            "selection_reason": live_gate_metric.get("selection_reason"),
            "path": live_gate_metric.get("path"),
        },
        "trajectory": {
            "available": bool(metrics["trajectory_profile"].get("available")),
            "datasets": available_trajectory_datasets,
            "data_root": str(args.trajectory_data_root.resolve(strict=False)),
        },
        "slowest_deterministic_probes": slowest_probe_tools[:5],
        "suggested_implementation_targets": metrics["suggested_implementation_targets"],
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo_root),
        "git": {
            "branch": _git_value(repo_root, "branch", "--show-current"),
            "head": _git_value(repo_root, "rev-parse", "HEAD"),
            "dirty": bool(_git_value(repo_root, "status", "--porcelain")),
        },
        "policy": {
            "auto_edit": False,
            "auto_merge": False,
            "auto_push": False,
            "purpose": "nightly profiling and benchmark report only",
        },
        "runtime": {
            "resolved_ollama_host": resolved_ollama_host,
            "llm_skip_reason": llm_skip_reason,
            "trajectory_skip_reason": trajectory_skip_reason,
            "trajectory_data_root": str(args.trajectory_data_root.resolve(strict=False)),
            "trajectory_profile_path": str(trajectory_profile_path) if not trajectory_skip_reason else None,
            "trajectory_error_profile_path": str(trajectory_error_path) if not trajectory_skip_reason else None,
            "trajectory_evidence_report_path": str(trajectory_evidence_path) if not trajectory_skip_reason else None,
            "live_gate_summary_path": str(live_gate_summary_path) if live_gate_summary_path else None,
        },
        "compare_path": str(compare_path) if compare_path else None,
        "run_dir": str(run_dir),
        "summary": summary,
        "commands": commands,
        "metrics": metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gated nightly self-improvement report. Produces JSON only; does not edit code.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for raw outputs and report.json.")
    parser.add_argument("--compare", type=Path, default=None, help="Prior report.json to compute coarse deltas.")
    parser.add_argument("--models", nargs="+", default=["gemma4:e4b"], help="Primary model(s) for LLM evaluations.")
    parser.add_argument("--suite", default="local-small", choices=["local-small", "local-full", "external-smoke"])
    parser.add_argument("--generated-files", type=int, default=2000, help="Synthetic generated-file count for no-LLM tool probe.")
    parser.add_argument("--case-timeout", type=int, default=600, help="Per-case timeout passed to eval scripts.")
    parser.add_argument("--command-timeout", type=int, default=7200, help="Wall-clock timeout for each LLM eval command.")
    parser.add_argument("--tool-timeout", type=int, default=600, help="Wall-clock timeout for non-LLM tooling commands.")
    parser.add_argument("--sdk-index-limit", type=int, default=5000, help="Python SDK API entries to index in the nightly retrieval eval.")
    parser.add_argument("--sdk-use-embeddings", action="store_true", help="Use Ollama embeddings for the Python SDK retrieval eval.")
    parser.add_argument("--sdk-embedding-model", default=None, help="Embedding model for --sdk-use-embeddings, e.g. nomic-embed-text.")
    parser.add_argument("--skip-llm", action="store_true", help="Run only no-LLM probes and anti-cheat.")
    parser.add_argument("--skip-trajectories", action="store_true", help="Skip local trajectory profile/error/evidence analysis.")
    parser.add_argument("--trajectory-data-root", type=Path, default=_repo_root() / "scratch" / "external" / "datasets", help="Root directory for local trajectory datasets.")
    parser.add_argument("--trajectory-max-rows", type=int, default=1000, help="Per-dataset row cap for recurring trajectory scans; use 0 for all rows.")
    parser.add_argument("--strict-accuracy", action="store_true", help="Propagate strict accuracy flags to eval scripts.")
    parser.add_argument("--strict-budget", action="store_true", help="Propagate strict budget checks to coding benchmarks.")
    args = parser.parse_args(argv)

    report = build_report(args)
    report_path = Path(report["run_dir"]) / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "ok": all(item.get("ok") for item in report["commands"])}, indent=2))
    return 0 if all(item.get("ok") for item in report["commands"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
