from __future__ import annotations

import argparse
import importlib.util
import json
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


def _module_to_path(module: str) -> str:
    return module.replace(".", "/") + ".py"


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _resolved_runner(requested: str) -> str:
    if requested == "auto":
        return "pytest" if _has_module("pytest") else "unittest"
    return requested


def _resolved_jobs(jobs: str, *, runner: str) -> str:
    if runner != "pytest" or not _has_module("xdist"):
        return "off"
    value = jobs.strip().lower()
    if value in {"", "0", "1", "off"}:
        return "off"
    if value == "auto":
        return "auto"
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


def _runtime_capabilities() -> dict[str, bool]:
    return {
        "pytest_available": _has_module("pytest"),
        "xdist_available": _has_module("xdist"),
    }


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
    target_args = command[6:] if runner == "pytest" and resolved_jobs != "off" else command[4:] if runner == "pytest" else command[3:-1] if command[:3] == [sys.executable, "-m", "unittest"] and command[-1] == "-q" else []
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
            commands.append(("full-suite", _pytest_target_command("tests", jobs=jobs)))
        else:
            commands.append(("full-discover", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]))
    return commands


def run_validation(tier: str, *, repo_root: Path, runner: str = "auto", jobs: str = "auto") -> dict[str, Any]:
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
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root.resolve(strict=False)),
        "requested_tier": tier,
        "requested_runner": runner,
        "resolved_runner": resolved_runner,
        "jobs": jobs,
        "resolved_jobs": resolved_jobs,
        **_runtime_capabilities(),
        "ok": all(row["ok"] for row in command_rows),
        "commands": command_rows,
        "planned_tiers": planned_tiers,
        "tiers_completed": completed_tiers,
        "remaining_tiers": remaining_tiers,
        "stopped_after_failure": bool(command_rows and not command_rows[-1]["ok"] and remaining_tiers),
        "elapsed_s": round(sum(float(row["elapsed_s"]) for row in command_rows), 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the repo-owned local validation tiers and persist a JSON summary.")
    parser.add_argument("--tier", choices=["smoke", "agent", "full"], default="smoke")
    parser.add_argument("--runner", choices=["auto", "pytest", "unittest"], default="auto")
    parser.add_argument("--jobs", default="auto", help="Pytest xdist workers: auto, off, 1, or an integer > 1.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON summary output path.")
    args = parser.parse_args(argv)

    if args.runner == "pytest" and not _has_module("pytest"):
        raise SystemExit("pytest runner requested, but pytest is not installed in this environment.")
    if args.jobs and args.runner != "unittest":
        _pytest_worker_args(args.jobs)

    payload = run_validation(args.tier, repo_root=REPO_ROOT, runner=args.runner, jobs=args.jobs)
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
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
