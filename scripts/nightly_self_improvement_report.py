from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
                bucket = totals.setdefault(str(tool_name), {"tool": str(tool_name), "calls": 0, "duration_ms": 0.0, "failed": 0, "cached": 0})
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


def _suggest_targets(metrics: dict[str, Any], commands: list[dict[str, Any]]) -> list[str]:
    suggestions: list[str] = []
    failed_commands = [item["name"] for item in commands if not item.get("ok")]
    if failed_commands:
        suggestions.append("Fix or shrink failing nightly command(s): " + ", ".join(failed_commands[:3]))
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
    return list(dict.fromkeys(suggestions))[:6]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = _repo_root()
    run_dir = args.output_dir or (repo_root / "scratch" / "nightly-self-improvement" / _timestamp())
    run_dir = run_dir.resolve(strict=False)
    run_dir.mkdir(parents=True, exist_ok=True)

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

    token_path = run_dir / "token_efficiency.json"
    benchmark_path = run_dir / "coding_benchmark.json"
    if not args.skip_llm:
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
        commands.append(_run_command(repo_root, "token_efficiency", token_command, timeout=args.command_timeout, output_path=token_path))

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
        commands.append(_run_command(repo_root, "coding_benchmark", benchmark_command, timeout=args.command_timeout, output_path=benchmark_path))

    commands.append(_run_command(repo_root, "anti_cheat_scan", [py, "scripts/anti_cheat_scan.py"], timeout=args.tool_timeout))

    token_payload = _load_json(token_path)
    benchmark_payload = _load_json(benchmark_path)
    probe_payload = _load_json(tool_probe_path)
    sdk_payload = _load_json(sdk_eval_path)
    baseline = _load_json(args.compare) if args.compare else {}
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
