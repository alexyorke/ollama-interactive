from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "scratch" / "validation" / "doctor-report.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    return _git_value(repo_root, "rev-parse", "--short", "HEAD") or None


def _git_dirty(repo_root: Path) -> bool | None:
    value = _git_value(repo_root, "status", "--short")
    if value is None:
        return None
    return bool(value.strip())


def build_doctor_report(
    repo_root: Path,
    *,
    command: list[str] | None = None,
    timeout_s: float = 120.0,
    now_iso: str | None = None,
) -> dict[str, Any]:
    resolved_command = command or [sys.executable, "-m", "ollama_code", "--doctor", "--quiet"]
    started = datetime.now(timezone.utc)
    timed_out = False
    try:
        completed = subprocess.run(
            resolved_command,
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
        returncode = completed.returncode
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        stderr = "\n".join(part for part in (stderr, f"Doctor command timed out after {timeout_s} seconds.") if part)
    elapsed_s = (datetime.now(timezone.utc) - started).total_seconds()
    ok = returncode == 0 and not timed_out
    first_line = next((line.strip() for line in stdout.splitlines() if line.strip()), "")
    return {
        "generated_at": now_iso or _now_iso(),
        "repo_root": str(repo_root),
        "git_commit": _git_head_commit(repo_root),
        "git_dirty": _git_dirty(repo_root),
        "command": resolved_command,
        "timeout_s": timeout_s,
        "elapsed_s": elapsed_s,
        "returncode": returncode,
        "timed_out": timed_out,
        "ok": ok,
        "status": "pass" if ok else "fail",
        "summary": first_line or ("doctor passed" if ok else "doctor failed"),
        "stdout": stdout,
        "stderr": stderr,
    }


def write_report(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ollama-code --doctor and write a readiness JSON artifact.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cwd", type=Path, default=REPO_ROOT)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args(argv)

    payload = build_doctor_report(args.cwd.resolve(), timeout_s=args.timeout_s)
    write_report(payload, args.output)
    print(f"[doctor-report] wrote {args.output}")
    print(f"[doctor-report] ok={payload['ok']} returncode={payload['returncode']} elapsed_s={payload['elapsed_s']:.3f}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
