from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import live_model_gate


DEFAULT_MAX_AGE_HOURS = 72.0
DEFAULT_REQUIRED_HARD_CASES = ("task_due_filter",)
REFRESH_COMMANDS = {
    "doctor": "python scripts/doctor_report.py",
    "local_validation": "python scripts/local_validation.py --tier agent",
    "live_model_gate": "python scripts/live_model_gate.py --models granite4.1:8b gemma4:e4b qwen3:8b --benchmark-suite local-small --benchmark-jobs 1 --continue-on-failure",
    "local_small": "python scripts/live_model_gate.py --models granite4.1:8b gemma4:e4b qwen3:8b --benchmark-suite local-small --benchmark-jobs 1 --continue-on-failure",
    "hard_cases": "python scripts/coding_benchmark_eval.py --suite local-full --models granite4.1:8b --modes off --cases task_due_filter --feature-profiles all --benchmark-classes agent controller --jobs 1 --strict-accuracy --strict-budget --require-llm-for-agent-benchmarks",
}


def _repo_root() -> Path:
    return REPO_ROOT


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable: {exc}"
    if not isinstance(payload, dict):
        return None, "not a JSON object"
    return payload, None


def _artifact_age_hours(path: Path, *, now: datetime | None = None) -> float | None:
    if not path.exists():
        return None
    timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return max(0.0, ((now or _now()) - timestamp).total_seconds() / 3600.0)


def _git_value(repo_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_head_commit(repo_root: Path) -> str | None:
    value = _git_value(repo_root, "rev-parse", "--short", "HEAD")
    return value or None


def _git_dirty(repo_root: Path) -> bool | None:
    value = _git_value(repo_root, "status", "--short")
    if value is None:
        return None
    return bool(value.strip())


def _check(
    name: str,
    ok: bool,
    summary: str,
    *,
    required: bool = True,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "ok": bool(ok),
        "required": bool(required),
        "summary": summary,
        "refresh_command": REFRESH_COMMANDS.get(name),
        "details": details or {},
    }


def _fresh_artifact_details(path: Path, *, max_age_hours: float, now: datetime | None = None) -> tuple[bool, dict[str, Any]]:
    age = _artifact_age_hours(path, now=now)
    details: dict[str, Any] = {"path": str(path), "max_age_hours": max_age_hours, "age_hours": age}
    if age is None:
        return False, details
    return age <= max_age_hours, details


def _check_git(repo_root: Path, *, allow_dirty: bool) -> dict[str, Any]:
    commit = _git_head_commit(repo_root)
    dirty = _git_dirty(repo_root)
    if commit is None or dirty is None:
        return _check("git", False, "Could not read git status.", details={"commit": commit, "dirty": dirty})
    if dirty and not allow_dirty:
        return _check("git", False, f"Worktree is dirty at {commit}.", details={"commit": commit, "dirty": dirty})
    return _check("git", True, f"Worktree {'dirty' if dirty else 'clean'} at {commit}.", details={"commit": commit, "dirty": dirty})


def _artifact_git_details(
    payload: dict[str, Any],
    *,
    current_commit: str | None,
    current_dirty: bool | None,
) -> dict[str, Any]:
    artifact_commit = payload.get("git_commit") if isinstance(payload.get("git_commit"), str) else None
    artifact_dirty = payload.get("git_dirty") if isinstance(payload.get("git_dirty"), bool) else None
    git_ok = artifact_commit == current_commit and artifact_dirty == current_dirty
    return {
        "git_commit": artifact_commit,
        "git_dirty": artifact_dirty,
        "current_commit": current_commit,
        "current_dirty": current_dirty,
        "git_metadata_ok": git_ok,
    }


def _check_doctor(
    path: Path | None,
    *,
    current_commit: str | None,
    current_dirty: bool | None,
    max_age_hours: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    if path is None:
        return _check("doctor", True, "No doctor artifact configured; skipped.", required=False)
    payload, error = _load_json(path)
    fresh, details = _fresh_artifact_details(path, max_age_hours=max_age_hours, now=now)
    details["error"] = error
    if payload:
        details["ok"] = payload.get("ok")
        details["status"] = payload.get("status")
    if error:
        return _check("doctor", False, f"Doctor artifact {error}.", details=details)
    assert payload is not None
    details.update(_artifact_git_details(payload, current_commit=current_commit, current_dirty=current_dirty))
    ok = fresh and bool(payload.get("ok", payload.get("status") in {"ok", "pass"})) and bool(details["git_metadata_ok"])
    return _check(
        "doctor",
        ok,
        "Doctor artifact is green." if ok else "Doctor artifact is stale, not green, or for a different git state.",
        details=details,
    )


def _check_local_validation(
    path: Path,
    *,
    current_commit: str | None,
    current_dirty: bool | None,
    max_age_hours: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    payload, error = _load_json(path)
    fresh, details = _fresh_artifact_details(path, max_age_hours=max_age_hours, now=now)
    details["error"] = error
    if error:
        return _check("local_validation", False, f"Local validation artifact {error}.", details=details)
    assert payload is not None
    command_ok = bool(payload.get("command_ok"))
    overall_ok = bool(payload.get("ok"))
    details.update(
        {
            "requested_tier": payload.get("requested_tier"),
            "resolved_runner": payload.get("resolved_runner"),
            "command_ok": command_ok,
            "ok": overall_ok,
        }
    )
    details.update(_artifact_git_details(payload, current_commit=current_commit, current_dirty=current_dirty))
    ok = fresh and command_ok and overall_ok and bool(details["git_metadata_ok"])
    if ok:
        summary = f"Local validation {payload.get('requested_tier') or 'unknown'} is green."
    elif command_ok and not overall_ok:
        summary = "Local validation commands passed, but release consistency checks failed."
    else:
        summary = "Local validation is missing, stale, failing, or for a different git state."
    return _check("local_validation", ok, summary, details=details)


def _check_live_gate(
    path: Path,
    *,
    current_commit: str | None,
    current_dirty: bool | None,
    max_age_hours: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    payload, error = _load_json(path)
    fresh, details = _fresh_artifact_details(path, max_age_hours=max_age_hours, now=now)
    details["error"] = error
    if error:
        return _check("live_model_gate", False, f"Live model gate artifact {error}.", details=details)
    assert payload is not None
    contract_ok = live_model_gate.summary_contract_ok(payload)
    gate_ok = bool(payload.get("ok"))
    git_ok = payload.get("git_commit") == current_commit and payload.get("git_dirty") == current_dirty
    models = payload.get("models") if isinstance(payload.get("models"), list) else []
    benchmark_rows_ok = bool(models) and all(bool(row.get("benchmark_ok")) for row in models if isinstance(row, dict))
    details.update(
        {
            "contract_ok": contract_ok,
            "ok": gate_ok,
            "git_commit": payload.get("git_commit"),
            "git_dirty": payload.get("git_dirty"),
            "current_commit": current_commit,
            "current_dirty": current_dirty,
            "git_metadata_ok": git_ok,
            "benchmark_suite": payload.get("benchmark_suite"),
            "selected_default_model": payload.get("selected_default_model"),
            "model_count": len(models),
        }
    )
    ok = fresh and contract_ok and gate_ok and git_ok and benchmark_rows_ok
    summary = "Live model gate is current and green." if ok else "Live model gate is missing, stale, failing, or for a different git state."
    return _check("live_model_gate", ok, summary, details=details)


def _check_local_small(
    path: Path,
    *,
    current_commit: str | None,
    current_dirty: bool | None,
    max_age_hours: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    payload, error = _load_json(path)
    fresh, details = _fresh_artifact_details(path, max_age_hours=max_age_hours, now=now)
    details["error"] = error
    if error:
        return _check("local_small", False, f"local-small artifact {error}.", details=details)
    assert payload is not None
    summary_payload = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    runs = int(summary_payload.get("runs") or 0)
    passes = int(summary_payload.get("pass") or 0)
    failures = list(payload.get("accuracy_regressions") or []) + list(payload.get("budget_failures") or []) + list(payload.get("llm_bypass_failures") or [])
    details.update({"runs": runs, "pass": passes, "failure_count": len(failures), "suite": payload.get("suite")})
    details.update(_artifact_git_details(payload, current_commit=current_commit, current_dirty=current_dirty))
    ok = fresh and runs > 0 and passes == runs and not failures and bool(details["git_metadata_ok"])
    return _check(
        "local_small",
        ok,
        f"local-small passed {passes}/{runs}." if ok else "local-small is missing, stale, not fully passing, or for a different git state.",
        details=details,
    )


def _check_hard_cases(
    path: Path,
    *,
    required_cases: tuple[str, ...],
    current_commit: str | None,
    current_dirty: bool | None,
    max_age_hours: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    payload, error = _load_json(path)
    fresh, details = _fresh_artifact_details(path, max_age_hours=max_age_hours, now=now)
    details["error"] = error
    if error:
        return _check("hard_cases", False, f"Hard-case artifact {error}.", details=details)
    assert payload is not None
    results = payload.get("results") if isinstance(payload.get("results"), list) else []
    by_case = {str(row.get("case") or ""): row for row in results if isinstance(row, dict)}
    case_statuses = {case: str(by_case.get(case, {}).get("status") or "missing") for case in required_cases}
    details.update({"required_cases": list(required_cases), "case_statuses": case_statuses, "suite": payload.get("suite")})
    details.update(_artifact_git_details(payload, current_commit=current_commit, current_dirty=current_dirty))
    ok = fresh and bool(required_cases) and all(status == "pass" for status in case_statuses.values()) and bool(details["git_metadata_ok"])
    passed = sum(1 for status in case_statuses.values() if status == "pass")
    summary = f"Hard cases passed {passed}/{len(required_cases)}." if ok else "Hard cases are missing, stale, failing, or for a different git state."
    return _check("hard_cases", ok, summary, details=details)


def build_report(
    repo_root: Path,
    *,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    allow_dirty: bool = False,
    doctor_json: Path | None = None,
    local_validation_json: Path | None = None,
    live_gate_json: Path | None = None,
    local_small_json: Path | None = None,
    hard_cases_json: Path | None = None,
    required_hard_cases: tuple[str, ...] = DEFAULT_REQUIRED_HARD_CASES,
    now: datetime | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    now = now or _now()
    local_validation_json = local_validation_json or repo_root / "scratch" / "validation" / "local-validation-summary.json"
    doctor_json = doctor_json or repo_root / "scratch" / "validation" / "doctor-report.json"
    live_gate_json = live_gate_json or repo_root / "scratch" / "live-model-gate" / "live-model-gate-summary.json"
    local_small_json = local_small_json or repo_root / "scratch" / "coding-benchmark" / "local-small.json"
    hard_cases_json = hard_cases_json or repo_root / "scratch" / "coding-benchmark" / "local-full.json"

    git_check = _check_git(repo_root, allow_dirty=allow_dirty)
    current_commit = git_check["details"].get("commit")
    current_dirty = git_check["details"].get("dirty")
    checks = [
        git_check,
        _check_doctor(doctor_json, current_commit=current_commit, current_dirty=current_dirty, max_age_hours=max_age_hours, now=now),
        _check_local_validation(local_validation_json, current_commit=current_commit, current_dirty=current_dirty, max_age_hours=max_age_hours, now=now),
        _check_live_gate(live_gate_json, current_commit=current_commit, current_dirty=current_dirty, max_age_hours=max_age_hours, now=now),
        _check_local_small(local_small_json, current_commit=current_commit, current_dirty=current_dirty, max_age_hours=max_age_hours, now=now),
        _check_hard_cases(hard_cases_json, required_cases=required_hard_cases, current_commit=current_commit, current_dirty=current_dirty, max_age_hours=max_age_hours, now=now),
    ]
    blocking = [check for check in checks if check["required"] and not check["ok"]]
    return {
        "generated_at": now.isoformat(),
        "repo_root": str(repo_root),
        "ok": not blocking,
        "max_age_hours": max_age_hours,
        "required_hard_cases": list(required_hard_cases),
        "blocking_checks": [check["name"] for check in blocking],
        "checks": checks,
    }


def render_text(payload: dict[str, Any]) -> str:
    status = "PASS" if payload.get("ok") else "FAIL"
    lines = [f"Product readiness: {status}", f"Repo: {payload.get('repo_root')}", ""]
    checks = payload.get("checks", [])
    for check in payload.get("checks", []):
        marker = "PASS" if check.get("ok") else ("WARN" if not check.get("required") else "FAIL")
        lines.append(f"- {marker} {check.get('name')}: {check.get('summary')}")
    blocking = payload.get("blocking_checks") or []
    if blocking:
        lines.append("")
        lines.append("Blocking checks: " + ", ".join(str(item) for item in blocking))
        refresh_commands = [
            (check.get("name"), check.get("refresh_command"))
            for check in checks
            if check.get("name") in blocking and check.get("refresh_command")
        ]
        if refresh_commands:
            lines.append("")
            lines.append("Refresh commands:")
            for name, command in refresh_commands:
                lines.append(f"- {name}: {command}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize existing artifacts into one product-readiness PASS/FAIL report.")
    parser.add_argument("--repo-root", type=Path, default=_repo_root())
    parser.add_argument("--output", type=Path, default=_repo_root() / "scratch" / "validation" / "product-readiness-report.json")
    parser.add_argument("--max-age-hours", type=float, default=DEFAULT_MAX_AGE_HOURS)
    parser.add_argument("--allow-dirty", action="store_true", help="Do not fail the git check for a dirty worktree.")
    parser.add_argument("--doctor-json", type=Path, default=None, help="JSON artifact from scripts/doctor_report.py.")
    parser.add_argument("--local-validation-json", type=Path, default=None)
    parser.add_argument("--live-gate-json", type=Path, default=None)
    parser.add_argument("--local-small-json", type=Path, default=None)
    parser.add_argument("--hard-cases-json", type=Path, default=None)
    parser.add_argument("--required-hard-cases", nargs="+", default=list(DEFAULT_REQUIRED_HARD_CASES))
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when required readiness checks fail.")
    args = parser.parse_args(argv)

    payload = build_report(
        args.repo_root,
        max_age_hours=args.max_age_hours,
        allow_dirty=args.allow_dirty,
        doctor_json=args.doctor_json,
        local_validation_json=args.local_validation_json,
        live_gate_json=args.live_gate_json,
        local_small_json=args.local_small_json,
        hard_cases_json=args.hard_cases_json,
        required_hard_cases=tuple(args.required_hard_cases),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    sys.stdout.write(render_text(payload))
    return 1 if args.strict and not payload.get("ok") else 0


if __name__ == "__main__":
    raise SystemExit(main())
