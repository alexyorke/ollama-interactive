from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    import coding_benchmark_eval as coding_bench
    import public_benchmark_eval as public_bench
except ModuleNotFoundError:  # Imported as scripts.anti_cheat_scan in unit tests.
    from scripts import coding_benchmark_eval as coding_bench
    from scripts import public_benchmark_eval as public_bench


def _public_task_ids_for_prompts() -> tuple[str, ...]:
    try:
        task_ids = public_bench.public_task_set("expanded")
    except Exception:
        task_ids = tuple(getattr(public_bench, "HARD_POLYGLOT_TASKS", ()))
    return tuple(task for task in task_ids if isinstance(task, str))


def _public_task_ids_for_runtime() -> tuple[str, ...]:
    try:
        task_ids = public_bench.public_task_set("hard")
    except Exception:
        task_ids = tuple(getattr(public_bench, "HARD_POLYGLOT_TASKS", ()))
    return tuple(task for task in task_ids if isinstance(task, str) and task != "transpose")


def _public_task_pattern(tasks: tuple[str, ...]) -> str:
    if not tasks:
        return r"(?i)\b(?:" + "transpose" + r")\b"
    escaped = "|".join(re.escape(task) for task in sorted(set(task.lower() for task in tasks)))
    return r"(?i)(?<![\\w-])(?:(?:" + escaped + r"))(?![\\w-])"


def _runtime_forbidden_patterns() -> dict[str, str]:
    return {
        "synthetic marker token": r"\b(?:BENCH|TOKEN|NEEDLE|EXACT)_[A-Z0-9_]+\b",
        "hard-coded public smoke task": _public_task_pattern(_public_task_ids_for_runtime()),
        "polyglot benchmark name": r"(?i)\bpolyglot-benchmark\b",
        "local benchmark case switch": r"(?i)\b(?:issue_fix_hidden_tests|multi_file_refactor|large_repo_symbol_nav|test_repair_task)\b",
    }


def _public_prompt_forbidden_patterns() -> dict[str, str]:
    return {
        "public smoke task": _public_task_pattern(_public_task_ids_for_prompts()),
        "task-specific solution hint": r"(?i)\b(?:foldr|pig latin|wordy|question)\b",
        "synthetic marker token": r"\b(?:BENCH|TOKEN|NEEDLE|EXACT)_[A-Z0-9_]+\b",
    }


RUNTIME_FORBIDDEN_PATTERNS: dict[str, str] = _runtime_forbidden_patterns()
PUBLIC_PROMPT_FORBIDDEN_PATTERNS: dict[str, str] = _public_prompt_forbidden_patterns()


def _refresh_patterns() -> None:
    RUNTIME_FORBIDDEN_PATTERNS.clear()
    RUNTIME_FORBIDDEN_PATTERNS.update(_runtime_forbidden_patterns())
    PUBLIC_PROMPT_FORBIDDEN_PATTERNS.clear()
    PUBLIC_PROMPT_FORBIDDEN_PATTERNS.update(_public_prompt_forbidden_patterns())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _runtime_files(repo_root: Path) -> list[Path]:
    return sorted((repo_root / "ollama_code").rglob("*.py"))


def _pattern_findings(text: str, patterns: dict[str, str]) -> list[str]:
    return [label for label, pattern in patterns.items() if re.search(pattern, text)]


def scan_runtime(repo_root: Path) -> list[dict[str, Any]]:
    _refresh_patterns()
    findings: list[dict[str, Any]] = []
    for path in _runtime_files(repo_root):
        text = path.read_text(encoding="utf-8", errors="replace")
        labels = _pattern_findings(text, RUNTIME_FORBIDDEN_PATTERNS)
        if labels:
            findings.append({"scope": "runtime", "path": str(path.relative_to(repo_root)), "findings": labels})
    return findings


def scan_benchmark_prompts() -> list[dict[str, Any]]:
    _refresh_patterns()
    findings: list[dict[str, Any]] = []
    for case in coding_bench.selected_cases("local-full"):
        labels = coding_bench.prompt_integrity_findings(case)
        if labels:
            findings.append({"scope": "coding_prompt", "case": case.name, "kind": case.benchmark_kind, "findings": labels})
    public_prompt = public_bench.public_task_prompt("Python")
    labels = _pattern_findings(public_prompt, PUBLIC_PROMPT_FORBIDDEN_PATTERNS)
    if labels:
        findings.append({"scope": "public_prompt", "findings": labels})
    return findings


def scan(repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or _repo_root()
    findings = scan_runtime(root) + scan_benchmark_prompts()
    return {"ok": not findings, "findings": findings}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan for benchmark/task-specific cheating in runtime code and prompts.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = parser.parse_args(argv)
    result = scan()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["ok"]:
        print("anti-cheat scan passed")
    else:
        print("anti-cheat scan failed")
        for item in result["findings"]:
            print(json.dumps(item, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
