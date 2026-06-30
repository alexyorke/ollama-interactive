from __future__ import annotations

import argparse
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


def _run(repo_root: Path, name: str, command: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, check=False)
    elapsed_s = round(time.perf_counter() - started, 3)
    combined = "\n".join(part for part in (completed.stdout.strip(), completed.stderr.strip()) if part)
    return {
        "name": name,
        "command": command,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "elapsed_s": elapsed_s,
        "output_tail": combined[-4000:],
    }


def _tier_commands(tier: str) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    if tier in {"smoke", "full"}:
        commands.append(("smoke", [sys.executable, "-m", "unittest", *SMOKE_MODULES, "-q"]))
    if tier in {"agent", "full"}:
        commands.append(("agent", [sys.executable, "-m", "unittest", *AGENT_MODULES, "-q"]))
    if tier == "full":
        commands.append(("full-discover", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]))
    return commands


def run_validation(tier: str, *, repo_root: Path) -> dict[str, Any]:
    command_rows: list[dict[str, Any]] = []
    for name, command in _tier_commands(tier):
        row = _run(repo_root, name, command)
        command_rows.append(row)
        if not row["ok"]:
            break
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root.resolve(strict=False)),
        "requested_tier": tier,
        "ok": all(row["ok"] for row in command_rows),
        "commands": command_rows,
        "tiers_completed": [row["name"] for row in command_rows],
        "elapsed_s": round(sum(float(row["elapsed_s"]) for row in command_rows), 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the repo-owned local validation tiers and persist a JSON summary.")
    parser.add_argument("--tier", choices=["smoke", "agent", "full"], default="smoke")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON summary output path.")
    args = parser.parse_args(argv)

    payload = run_validation(args.tier, repo_root=REPO_ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[local-validation] wrote {args.output}")
    for row in payload["commands"]:
        print(
            "[local-validation]"
            + f" tier={row['name']}"
            + f" ok={row['ok']}"
            + f" elapsed_s={row['elapsed_s']}"
            + f" returncode={row['returncode']}"
        )
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
