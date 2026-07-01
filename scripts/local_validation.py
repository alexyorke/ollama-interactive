from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "scratch" / "validation" / "local-validation-summary.json"

SMOKE_MODULES = (
    "tests.test_local_validation",
    "tests.test_live_model_gate",
    "tests.test_nightly_self_improvement_report",
    "tests.test_text_hygiene_scan",
    "tests.test_trajectory_dataset_catalog",
    "tests.test_trajectory_dataset_fetch",
    "tests.test_trajectory_profile",
    "tests.test_trajectory_error_profile",
    "tests.test_trajectory_evidence_report",
    "tests.test_web_discovered_agent_dataset_analysis",
)

AGENT_MODULES = (
    "tests.test_agent_grounding_path_repair",
    "tests.test_agent_post_edit_validation",
    "tests.test_agent_failure_compression",
    "tests.test_agent_shell_command_preflight",
    "tests.test_tools",
    "tests.test_coding_benchmark_eval",
)

MAX_AUTO_PYTEST_WORKERS = 16
LIVE_GATE_SUMMARY_NAME = "live-model-gate-summary.json"
LIVE_GATE_CLAIM_PATHS = (
    "README.md",
    "docs/token-efficiency.md",
    "TODO.md",
    "tests/test_live_model_gate.py",
)


def _module_to_path(module: str) -> str:
    return module.replace(".", "/") + ".py"


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _resolved_runner(requested: str) -> str:
    if requested == "auto":
        return "pytest" if _has_module("pytest") else "unittest"
    return requested


def _auto_pytest_jobs() -> str:
    cpu_count = max(1, int(os.cpu_count() or 1))
    if cpu_count <= 1:
        return "off"
    return str(min(cpu_count, MAX_AUTO_PYTEST_WORKERS))


def _resolved_jobs(jobs: str, *, runner: str) -> str:
    if runner != "pytest" or not _has_module("xdist"):
        return "off"
    value = jobs.strip().lower()
    if value in {"", "0", "1", "off"}:
        return "off"
    if value == "auto":
        return _auto_pytest_jobs()
    if value.isdigit() and int(value) > 1:
        return value
    raise ValueError(f"unsupported --jobs value: {jobs!r}")


def _pytest_worker_args(jobs: str) -> list[str]:
    resolved_jobs = _resolved_jobs(jobs, runner="pytest")
    return [] if resolved_jobs == "off" else ["-n", resolved_jobs]


def _pytest_command(modules: tuple[str, ...], *, jobs: str) -> list[str]:
    return [sys.executable, "-m", "pytest", "-q", *_pytest_worker_args(jobs), *(_module_to_path(module) for module in modules)]


def _pytest_target_command(*targets: str, jobs: str) -> list[str]:
    return [sys.executable, "-m", "pytest", "-q", *_pytest_worker_args(jobs), *targets]


def _all_pytest_targets(repo_root: Path = REPO_ROOT) -> tuple[str, ...]:
    tests_root = repo_root / "tests"
    return tuple(
        sorted(
            str(path.relative_to(repo_root)).replace("\\", "/")
            for path in tests_root.rglob("*.py")
            if path.name.startswith("test_") or path.name.endswith("_test.py")
        )
    )


def _remaining_pytest_targets(*excluded_modules: str, repo_root: Path = REPO_ROOT) -> tuple[str, ...]:
    excluded = {_module_to_path(module) for module in excluded_modules}
    return tuple(target for target in _all_pytest_targets(repo_root) if target not in excluded)


def _runtime_capabilities() -> dict[str, bool]:
    return {
        "pytest_available": _has_module("pytest"),
        "xdist_available": _has_module("xdist"),
    }


def _unittest_discover_command(*, quiet: bool) -> list[str]:
    return [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q" if quiet else "-v"]


def _unittest_target_count(command: list[str]) -> int:
    args = command[3:]
    if not args:
        return 0
    if args[0] == "discover":
        return 1
    trailing_flags = {"-q", "-v"}
    return len([arg for arg in args if arg not in trailing_flags and not arg.startswith("-")])


def _run(
    repo_root: Path,
    name: str,
    command: list[str],
    *,
    runner: str,
    resolved_jobs: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, check=False)
    elapsed_s = round(time.perf_counter() - started, 3)
    combined = "\n".join(part for part in (completed.stdout.strip(), completed.stderr.strip()) if part)
    target_args = []
    if runner == "pytest":
        target_args = command[6:] if resolved_jobs != "off" else command[4:]
    elif command[:3] == [sys.executable, "-m", "unittest"]:
        target_args = ["<discover>"] * _unittest_target_count(command)
    return {
        "name": name,
        "command": command,
        "runner": runner,
        "resolved_jobs": resolved_jobs,
        "target_count": len(target_args),
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "elapsed_s": elapsed_s,
        "output_tail": combined[-4000:],
    }


def _command_targets(command: list[str], *, runner: str, resolved_jobs: str) -> list[str]:
    if runner == "pytest":
        return command[6:] if resolved_jobs != "off" else command[4:]
    if command[:3] == [sys.executable, "-m", "unittest"]:
        args = command[3:]
        if not args or args[0] == "discover":
            return []
        trailing_flags = {"-q", "-v"}
        return [arg for arg in args if arg not in trailing_flags and not arg.startswith("-")]
    return []


def _timing_summary(command_rows: list[dict[str, Any]], *, elapsed_s: float) -> dict[str, Any]:
    total_elapsed = round(float(elapsed_s), 3)
    slowest = sorted(
        (
            {
                "name": str(row.get("name") or ""),
                "elapsed_s": round(float(row.get("elapsed_s", 0.0) or 0.0), 3),
                "target_count": int(row.get("target_count", 0) or 0),
                "ok": bool(row.get("ok")),
            }
            for row in command_rows
        ),
        key=lambda item: (-item["elapsed_s"], item["name"]),
    )
    return {
        "command_count": len(command_rows),
        "successful_commands": sum(1 for row in command_rows if row.get("ok")),
        "failed_commands": sum(1 for row in command_rows if not row.get("ok")),
        "total_elapsed_s": total_elapsed,
        "slowest_commands": slowest[:3],
    }


def _coverage_summary(
    tier: str,
    *,
    repo_root: Path,
    runner: str,
    jobs: str,
    resolved_jobs: str,
) -> dict[str, Any]:
    if runner != "pytest":
        return {
            "mode": "unittest",
            "requested_tier": tier,
            "note": "path-level coverage accounting is only available for pytest runs",
        }
    all_targets = list(_all_pytest_targets(repo_root))
    full_commands = _tier_commands("full", runner=runner, jobs=jobs)
    requested_commands = _tier_commands(tier, runner=runner, jobs=jobs)

    def flatten_target_lists(commands: list[tuple[str, list[str]]]) -> tuple[list[str], dict[str, list[str]]]:
        by_tier: dict[str, list[str]] = {}
        flat: list[str] = []
        for name, command in commands:
            targets = _command_targets(command, runner=runner, resolved_jobs=resolved_jobs)
            by_tier[name] = targets
            flat.extend(targets)
        return flat, by_tier

    requested_flat, requested_by_tier = flatten_target_lists(requested_commands)
    full_flat, full_by_tier = flatten_target_lists(full_commands)
    unique_full = set(full_flat)
    unique_requested = set(requested_flat)
    all_target_set = set(all_targets)
    duplicate_targets = sorted({target for target in full_flat if full_flat.count(target) > 1})
    uncovered_targets = sorted(all_target_set - unique_full)
    extra_targets = sorted(unique_full - all_target_set)
    return {
        "mode": "pytest_target_paths",
        "requested_tier": tier,
        "discovered_test_target_count": len(all_targets),
        "requested_target_count": len(requested_flat),
        "requested_unique_target_count": len(unique_requested),
        "full_plan_target_count": len(full_flat),
        "full_plan_unique_target_count": len(unique_full),
        "full_plan_duplicate_target_count": len(duplicate_targets),
        "full_plan_duplicate_targets_sample": duplicate_targets[:5],
        "full_plan_uncovered_target_count": len(uncovered_targets),
        "full_plan_uncovered_targets_sample": uncovered_targets[:5],
        "full_plan_extra_target_count": len(extra_targets),
        "full_plan_extra_targets_sample": extra_targets[:5],
        "full_plan_covers_all_discovered_targets": not duplicate_targets and not uncovered_targets and not extra_targets,
        "requested_targets_by_tier": requested_by_tier,
        "full_plan_targets_by_tier": full_by_tier,
    }


def _coverage_ok(summary: dict[str, Any]) -> bool:
    if summary.get("mode") != "pytest_target_paths":
        return True
    return bool(summary.get("full_plan_covers_all_discovered_targets"))


def _baseline_compare_payload(
    repo_root: Path,
    *,
    requested: bool,
    tier: str,
    validation_ok: bool,
    preferred_elapsed_s: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "requested": requested,
        "ran": False,
        "preferred_elapsed_s": round(preferred_elapsed_s, 3),
        "preferred_vs_unittest_ratio": None,
        "preferred_minus_unittest_s": None,
        "preferred_faster_than_unittest": None,
        "unittest_discover": None,
        "skipped_reason": None,
    }
    if not requested:
        return payload
    if tier != "full":
        payload["skipped_reason"] = "comparison only runs for the full tier"
        return payload
    if not validation_ok:
        payload["skipped_reason"] = "preferred validation failed"
        return payload
    row = _run(
        repo_root,
        "unittest-baseline",
        _unittest_discover_command(quiet=True),
        runner="unittest",
        resolved_jobs="off",
    )
    payload["ran"] = True
    payload["unittest_discover"] = row
    baseline_elapsed = float(row.get("elapsed_s", 0.0) or 0.0)
    if row.get("ok") and baseline_elapsed > 0:
        payload["preferred_vs_unittest_ratio"] = round(preferred_elapsed_s / baseline_elapsed, 2)
        payload["preferred_minus_unittest_s"] = round(preferred_elapsed_s - baseline_elapsed, 3)
        payload["preferred_faster_than_unittest"] = preferred_elapsed_s < baseline_elapsed
    return payload


def _latest_live_gate_summary_path(repo_root: Path) -> Path | None:
    canonical = repo_root / "scratch" / "live-model-gate" / LIVE_GATE_SUMMARY_NAME
    if canonical.exists():
        return canonical
    candidates = sorted((repo_root / "scratch").glob(f"live-model-gate-*/{LIVE_GATE_SUMMARY_NAME}"))
    return candidates[-1] if candidates else None


def _extract_release_token_claims(text: str, *, path: str) -> list[str] | None:
    if path in {"README.md", "TODO.md"}:
        match = re.search(r"fewest benchmark tokens \(`(\d+)` vs `(\d+)` and `(\d+)`\)", text)
        return list(match.groups()) if match else None
    if path == "docs/token-efficiency.md":
        found: list[str] = []
        for model in ("granite4.1:8b", "gemma4:e4b", "qwen3:8b"):
            match = re.search(rf"`{re.escape(model)}`.*?\(`(\d+)`\)", text, flags=re.DOTALL)
            if not match:
                return None
            found.append(match.group(1))
        return found
    if path == "tests/test_live_model_gate.py":
        found = re.findall(r'"benchmark_total_tokens": (\d+)', text)
        return found[:3] if len(found) >= 3 else None
    return None


def _live_gate_claim_consistency(repo_root: Path) -> dict[str, Any]:
    summary_path = _latest_live_gate_summary_path(repo_root)
    if summary_path is None:
        return {"ok": False, "available": False, "summary_path": None, "summary": "No live-gate summary artifact found."}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "available": False,
            "summary_path": str(summary_path),
            "summary": f"Could not read live-gate summary artifact: {exc}",
        }
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list) or len(models) < 3:
        return {
            "ok": False,
            "available": False,
            "summary_path": str(summary_path),
            "summary": "Live-gate summary artifact is missing expected model rows.",
        }
    expected_tokens = [str(item.get("benchmark_total_tokens")) for item in models[:3]]
    file_rows: list[dict[str, Any]] = []
    for rel in LIVE_GATE_CLAIM_PATHS:
        target = repo_root / rel
        try:
            text = target.read_text(encoding="utf-8")
        except OSError as exc:
            file_rows.append({"path": rel, "ok": False, "found_tokens": None, "summary": f"Could not read file: {exc}"})
            continue
        found_tokens = _extract_release_token_claims(text, path=rel)
        file_rows.append(
            {
                "path": rel,
                "ok": found_tokens == expected_tokens,
                "found_tokens": found_tokens,
                "summary": "matched" if found_tokens == expected_tokens else f"expected {expected_tokens}, found {found_tokens}",
            }
        )
    ok = all(bool(row.get("ok")) for row in file_rows)
    return {
        "ok": ok,
        "available": True,
        "summary_path": str(summary_path),
        "expected_tokens": expected_tokens,
        "files": file_rows,
        "summary": "All tracked live-gate token claims match the canonical summary." if ok else "Live-gate token claims drift from the canonical summary.",
    }


def _tier_commands(tier: str, *, runner: str, jobs: str) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    resolved_runner = _resolved_runner(runner)
    if tier in {"smoke", "full"}:
        if resolved_runner == "pytest":
            commands.append(("smoke", _pytest_command(SMOKE_MODULES, jobs=jobs)))
        else:
            commands.append(("smoke", [sys.executable, "-m", "unittest", *SMOKE_MODULES, "-q"]))
    if tier in {"agent", "full"}:
        if resolved_runner == "pytest":
            commands.append(("agent", _pytest_command(AGENT_MODULES, jobs=jobs)))
        else:
            commands.append(("agent", [sys.executable, "-m", "unittest", *AGENT_MODULES, "-q"]))
    if tier == "full":
        if resolved_runner == "pytest":
            remaining_targets = _remaining_pytest_targets(*SMOKE_MODULES, *AGENT_MODULES)
            if remaining_targets:
                commands.append(("full-remaining", _pytest_target_command(*remaining_targets, jobs=jobs)))
        else:
            commands.append(("full-discover", _unittest_discover_command(quiet=False)))
    return commands


def run_validation(
    tier: str,
    *,
    repo_root: Path,
    runner: str = "auto",
    jobs: str = "auto",
    compare_unittest_baseline: bool = False,
) -> dict[str, Any]:
    command_rows: list[dict[str, Any]] = []
    resolved_runner = _resolved_runner(runner)
    resolved_jobs = _resolved_jobs(jobs, runner=resolved_runner)
    planned_commands = _tier_commands(tier, runner=runner, jobs=jobs)
    planned_tiers = [name for name, _ in planned_commands]
    for name, command in planned_commands:
        row = _run(repo_root, name, command, runner=resolved_runner, resolved_jobs=resolved_jobs)
        command_rows.append(row)
        if not row["ok"]:
            break
    completed_tiers = [row["name"] for row in command_rows]
    remaining_tiers = planned_tiers[len(completed_tiers) :]
    elapsed_s = round(sum(float(row["elapsed_s"]) for row in command_rows), 3)
    command_ok = all(row["ok"] for row in command_rows)
    for row in command_rows:
        row_elapsed = round(float(row.get("elapsed_s", 0.0) or 0.0), 3)
        row["elapsed_share_pct"] = round((row_elapsed / elapsed_s) * 100, 1) if elapsed_s > 0 else 0.0
    coverage_summary = _coverage_summary(
        tier,
        repo_root=repo_root,
        runner=resolved_runner,
        jobs=jobs,
        resolved_jobs=resolved_jobs,
    )
    live_gate_claim_consistency = _live_gate_claim_consistency(repo_root)
    ok = command_ok and _coverage_ok(coverage_summary) and bool(live_gate_claim_consistency.get("ok"))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root.resolve(strict=False)),
        "requested_tier": tier,
        "requested_runner": runner,
        "resolved_runner": resolved_runner,
        "jobs": jobs,
        "resolved_jobs": resolved_jobs,
        **_runtime_capabilities(),
        "ok": ok,
        "command_ok": command_ok,
        "commands": command_rows,
        "planned_tiers": planned_tiers,
        "tiers_completed": completed_tiers,
        "remaining_tiers": remaining_tiers,
        "stopped_after_failure": bool(command_rows and not command_rows[-1]["ok"] and remaining_tiers),
        "elapsed_s": elapsed_s,
        "timing_summary": _timing_summary(command_rows, elapsed_s=elapsed_s),
        "coverage_summary": coverage_summary,
        "live_gate_claim_consistency": live_gate_claim_consistency,
        "baseline_compare": _baseline_compare_payload(
            repo_root,
            requested=compare_unittest_baseline,
            tier=tier,
            validation_ok=ok,
            preferred_elapsed_s=elapsed_s,
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the repo-owned local validation tiers and persist a JSON summary.")
    parser.add_argument("--tier", choices=["smoke", "agent", "full"], default="smoke")
    parser.add_argument("--runner", choices=["auto", "pytest", "unittest"], default="auto")
    parser.add_argument("--jobs", default="auto", help="Pytest xdist workers: auto, off, 1, or an integer > 1. auto uses a bounded worker count.")
    parser.add_argument(
        "--compare-unittest-baseline",
        action="store_true",
        help="When running the full tier, also time a raw unittest discover baseline for comparison.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON summary output path.")
    args = parser.parse_args(argv)

    if args.runner == "pytest" and not _has_module("pytest"):
        raise SystemExit("pytest runner requested, but pytest is not installed in this environment.")
    if args.jobs and args.runner != "unittest":
        _pytest_worker_args(args.jobs)

    payload = run_validation(
        args.tier,
        repo_root=REPO_ROOT,
        runner=args.runner,
        jobs=args.jobs,
        compare_unittest_baseline=args.compare_unittest_baseline,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[local-validation] wrote {args.output}")
    for row in payload["commands"]:
        print(
            "[local-validation]"
            + f" tier={row['name']}"
            + f" runner={row['runner']}"
            + f" jobs={row['resolved_jobs']}"
            + f" ok={row['ok']}"
            + f" elapsed_s={row['elapsed_s']}"
            + f" returncode={row['returncode']}"
        )
    timing_summary = payload.get("timing_summary") or {}
    slowest_commands = timing_summary.get("slowest_commands") or []
    if slowest_commands:
        slowest = slowest_commands[0]
        print(
            "[local-validation]"
            + f" slowest={slowest.get('name')}"
            + f" elapsed_s={slowest.get('elapsed_s')}"
            + f" share_pct={next((row.get('elapsed_share_pct') for row in payload['commands'] if row.get('name') == slowest.get('name')), None)}"
            + f" total_elapsed_s={timing_summary.get('total_elapsed_s')}"
        )
    coverage_summary = payload.get("coverage_summary") or {}
    if coverage_summary.get("mode") == "pytest_target_paths":
        print(
            "[local-validation]"
            + f" coverage tier={coverage_summary.get('requested_tier')}"
            + f" requested_unique={coverage_summary.get('requested_unique_target_count')}"
            + f" full_unique={coverage_summary.get('full_plan_unique_target_count')}"
            + f" discovered={coverage_summary.get('discovered_test_target_count')}"
            + f" full_plan_complete={coverage_summary.get('full_plan_covers_all_discovered_targets')}"
            + f" duplicates={coverage_summary.get('full_plan_duplicate_target_count')}"
            + f" uncovered={coverage_summary.get('full_plan_uncovered_target_count')}"
        )
    claim_consistency = payload.get("live_gate_claim_consistency") or {}
    if claim_consistency:
        print(
            "[local-validation]"
            + f" live_gate_claims_ok={claim_consistency.get('ok')}"
            + f" summary_path={claim_consistency.get('summary_path') or '-'}"
        )
    comparison = payload.get("baseline_compare") or {}
    if comparison.get("ran") and isinstance(comparison.get("unittest_discover"), dict):
        baseline = comparison["unittest_discover"]
        print(
            "[local-validation]"
            + f" baseline={baseline['name']}"
            + f" ok={baseline['ok']}"
            + f" elapsed_s={baseline['elapsed_s']}"
            + f" preferred_vs_unittest_ratio={comparison.get('preferred_vs_unittest_ratio')}"
            + f" preferred_minus_unittest_s={comparison.get('preferred_minus_unittest_s')}"
            + f" preferred_faster_than_unittest={comparison.get('preferred_faster_than_unittest')}"
        )
    elif comparison.get("requested") and comparison.get("skipped_reason"):
        print("[local-validation]" + f" baseline_skipped={comparison['skipped_reason']}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
