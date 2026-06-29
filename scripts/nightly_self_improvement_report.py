from __future__ import annotations

import argparse
import json
import os
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

from scripts import trajectory_profile


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
                bucket["calls"] += int(item.get("calls", 0) or 0)
                bucket["duration_ms"] += float(item.get("duration_ms", 0.0) or 0.0)
                bucket["failed"] += int(item.get("failed", 0) or 0)
                bucket["cached"] += int(item.get("cached", 0) or 0)
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
    normalized = [row for row in rows if isinstance(row, dict)]
    return sorted(normalized, key=lambda row: float(row.get("elapsed_ms", 0.0) or 0.0), reverse=True)


def _question_quality_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return {"cases": 0, "passed": 0, "failed": 0}
    return {
        "cases": int(summary.get("cases", 0) or 0),
        "passed": int(summary.get("passed", 0) or 0),
        "failed": int(summary.get("failed", 0) or 0),
    }


def _trajectory_profile_summary(payload: dict[str, Any], *, expected_datasets: list[str]) -> dict[str, Any]:
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        return {
            "available": bool(expected_datasets),
            "expected_datasets": expected_datasets,
            "rows_profiled": 0,
            "datasets_profiled": 0,
            "top_recommendations": [],
        }
    rows_profiled = sum(int(item.get("rows_profiled", 0) or 0) for item in datasets if isinstance(item, dict))
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
        return {"available": bool(expected_datasets), "expected_datasets": expected_datasets, "top_errors": []}
    counts: dict[str, int] = {}
    for item in datasets:
        if not isinstance(item, dict):
            continue
        error_counts = item.get("error_counts")
        if not isinstance(error_counts, dict):
            continue
        for name, value in error_counts.items():
            counts[str(name)] = counts.get(str(name), 0) + int(value or 0)
    top = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)[:8]
    return {
        "available": True,
        "expected_datasets": expected_datasets,
        "top_errors": [{"name": name, "count": count} for name, count in top],
    }


def _trajectory_evidence_summary(payload: dict[str, Any], *, expected_datasets: list[str]) -> dict[str, Any]:
    coverage = payload.get("portfolio_fix_coverage")
    if not isinstance(coverage, list):
        return {"available": bool(expected_datasets), "expected_datasets": expected_datasets, "top_fix_coverage": []}
    top_fix_coverage = []
    for item in coverage[:6]:
        if not isinstance(item, dict):
            continue
        top_fix_coverage.append(
            {
                "id": str(item.get("id") or ""),
                "evidence_count": int(item.get("evidence_count", 0) or 0),
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
    return {
        "entries": int(summary.get("entries", 0) or 0),
        "local_entries": int(summary.get("local_entries", 0) or 0),
        "analysis_ready_local_entries": int(summary.get("analysis_ready_local_entries", 0) or 0),
        "public_missing_entries": int(summary.get("public_missing_entries", 0) or 0),
        "gated_entries": int(summary.get("gated_entries", 0) or 0),
        "high_priority_public_missing": list(summary.get("high_priority_public_missing") or []),
    }


def _available_trajectory_datasets(data_root: Path) -> list[str]:
    available: list[str] = []
    for dataset_name in trajectory_profile.DATASET_SPECS:
        if (data_root / dataset_name).exists():
            available.append(dataset_name)
    return available


def _default_compare_path(repo_root: Path, current_run_dir: Path) -> Path | None:
    root = repo_root / "scratch" / "nightly-self-improvement"
    if not root.exists():
        return None
    candidates: list[Path] = []
    for path in root.glob("*/report.json"):
        if current_run_dir in path.parents:
            continue
        candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item.parent.name, str(item)), reverse=True)
    return candidates[0]


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
    if isinstance(question_quality, dict) and int(question_quality.get("failed", 0) or 0):
        suggestions.append("Improve clarification-question quality before expanding autonomy or rewrite scope.")
    for key in ("token_efficiency", "coding_benchmark"):
        section = metrics.get(key)
        if not isinstance(section, dict):
            continue
        summary = section.get("summary") if isinstance(section.get("summary"), dict) else {}
        if int(summary.get("fail", 0) or 0) or int(summary.get("fail_closed", 0) or 0):
            suggestions.append(f"Inspect failed {key} cases before adding new features.")
        delta = section.get("delta") if isinstance(section.get("delta"), dict) else {}
        if float(delta.get("total_tokens", 0) or 0) > 0:
            suggestions.append(f"Reduce token growth in {key}; compare per-case prompt profiles.")
    slow_tools = metrics.get("slowest_tools")
    if isinstance(slow_tools, list) and slow_tools:
        first = slow_tools[0]
        if isinstance(first, dict) and first.get("tool"):
            suggestions.append(f"Profile deterministic tool path for {first['tool']}; it is the slowest recurring tool.")
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
        if int(summary.get("fail", 0) or 0):
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
    compare_path = args.compare or _default_compare_path(repo_root, run_dir)
    available_trajectory_datasets = _available_trajectory_datasets(args.trajectory_data_root)
    resolved_ollama_host = None if args.skip_llm else _resolve_ollama_host()
    llm_skip_reason = ""
    if not args.skip_llm and not resolved_ollama_host:
        llm_skip_reason = "No reachable Ollama host found; skipped token-efficiency and coding-benchmark commands."

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
    baseline = _load_json(compare_path) if compare_path else {}
    baseline_metrics = baseline.get("metrics") if isinstance(baseline.get("metrics"), dict) else {}
    baseline_token = baseline_metrics.get("token_efficiency") if isinstance(baseline_metrics.get("token_efficiency"), dict) else {}
    baseline_benchmark = baseline_metrics.get("coding_benchmark") if isinstance(baseline_metrics.get("coding_benchmark"), dict) else {}

    metrics: dict[str, Any] = {
        "tool_speed_probe": {
            "generated_files": probe_payload.get("generated_files"),
            "slowest": _collect_probe_slowest(probe_payload)[:5],
        },
        "python_sdk_search": {
            "summary": _summary(sdk_payload),
            "refresh": sdk_payload.get("refresh", {}),
            "mode": sdk_payload.get("mode"),
        },
        "question_quality": _question_quality_summary(question_quality_payload),
        "trajectory_dataset_catalog": _trajectory_catalog_summary(trajectory_catalog_payload),
        "trajectory_profile": _trajectory_profile_summary(
            trajectory_profile_payload, expected_datasets=available_trajectory_datasets
        ),
        "trajectory_error_profile": _trajectory_error_summary(
            trajectory_error_payload, expected_datasets=available_trajectory_datasets
        ),
        "trajectory_evidence_report": _trajectory_evidence_summary(
            trajectory_evidence_payload, expected_datasets=available_trajectory_datasets
        ),
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
    }
    metrics["suggested_implementation_targets"] = _suggest_targets(metrics, commands)

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
        },
        "compare_path": str(compare_path) if compare_path else None,
        "run_dir": str(run_dir),
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
