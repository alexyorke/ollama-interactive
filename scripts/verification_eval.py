from __future__ import annotations

import argparse
import atexit
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    from e2e_suite import build_workspace, commit_all, installed_models, load_session, run_cli
    from workspace_temp import workspace_temp_dir
except ModuleNotFoundError:
    from scripts.e2e_suite import build_workspace, commit_all, installed_models, load_session, run_cli
    from scripts.workspace_temp import workspace_temp_dir


_LOADED_MODELS: set[str] = set()
FAIL_CLOSED_MESSAGES = {
    "Stopped because grounded final verification could not accept a final answer.",
    "Stopped because assumption audit could not approve a next tool step.",
}


def unload_model(model: str) -> None:
    subprocess.run(["ollama", "stop", model], capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)


def _cleanup_loaded_models() -> None:
    for model in sorted(_LOADED_MODELS):
        unload_model(model)


atexit.register(_cleanup_loaded_models)


def resolve_requested_models(requested: list[str], available: set[str]) -> list[str]:
    resolved: list[str] = []
    for model in requested:
        if model in available:
            resolved.append(model)
            continue
        latest = f"{model}:latest"
        if latest in available:
            resolved.append(latest)
    return resolved


def tool_results(session: dict[str, Any], tool_name: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for event in session.get("events", []):
        if event.get("type") == "tool_result" and event.get("name") == tool_name:
            result = event.get("result")
            if isinstance(result, dict):
                results.append(result)
    return results


def tool_calls(session: dict[str, Any]) -> list[str]:
    calls: list[str] = []
    for event in session.get("events", []):
        if event.get("type") == "tool_call" and isinstance(event.get("name"), str):
            calls.append(str(event["name"]))
    return calls


def verification_retry_count(session: dict[str, Any]) -> int:
    count = 0
    for event in session.get("events", []):
        if event.get("type") == "verification" and event.get("verdict") == "retry":
            count += 1
    return count


def assumption_audit_count(session: dict[str, Any]) -> int:
    count = 0
    for event in session.get("events", []):
        if event.get("type") == "assumption_audit":
            count += 1
    return count


def assumption_audit_retry_count(session: dict[str, Any]) -> int:
    count = 0
    for event in session.get("events", []):
        if event.get("type") == "assumption_audit" and event.get("verdict") == "retry":
            count += 1
    return count


def verification_rewrite_count(session: dict[str, Any]) -> int:
    count = 0
    for event in session.get("events", []):
        if event.get("type") == "verification_rewrite":
            count += 1
    return count


def final_assistant_message(session: dict[str, Any]) -> str:
    for event in reversed(session.get("events", [])):
        if event.get("type") == "assistant" and isinstance(event.get("content"), str):
            return str(event["content"]).strip()
    return ""


def is_fail_closed_message(message: str) -> bool:
    return message in FAIL_CLOSED_MESSAGES


@dataclass(frozen=True)
class EvalCase:
    name: str
    prompt: str
    acceptable_with_verification: set[str]
    prepare: Callable[[Path], None] | None = None


def prepare_git_diff_case(workspace: Path) -> None:
    commit_all(workspace, "checkpoint before verification eval")
    target = workspace / "src" / "app.py"
    target.write_text("def meaning() -> int:\n    return 99\n", encoding="utf-8")


def evaluate_case(
    repo_root: Path,
    workspace: Path,
    model: str,
    case: EvalCase,
    *,
    verification_enabled: bool,
    verifier_model: str | None,
    timeout: int,
) -> dict[str, Any]:
    if case.prepare is not None:
        case.prepare(workspace)
    session_file = workspace / "scratch" / f"{case.name}-{'on' if verification_enabled else 'off'}.json"
    extra_args = ["--debate", "on" if verification_enabled else "off"]
    if verifier_model:
        extra_args.extend(["--verifier-model", verifier_model])
    started = time.perf_counter()
    result = run_cli(
        repo_root,
        workspace,
        model,
        case.prompt,
        session_file=session_file,
        timeout=timeout,
        extra_args=extra_args,
    )
    elapsed = time.perf_counter() - started
    session = load_session(session_file)
    calls = tool_calls(session)
    assistant = final_assistant_message(session)

    if case.name == "tool_use_token":
        status = "pass"
        reads = tool_results(session, "read_file")
        if result.returncode != 0 or not any(item.get("ok") for item in reads) or "TOKEN_42" not in result.stdout:
            status = "fail"
    elif case.name == "git_diff_grounding":
        statuses = tool_results(session, "git_status")
        diffs = tool_results(session, "git_diff")
        if result.returncode != 0 or not any(item.get("ok") for item in statuses) or not any(item.get("ok") and "return 99" in str(item.get("output", "")) for item in diffs) or "read_file" in calls:
            status = "fail"
        elif "return 99" in result.stdout:
            status = "pass"
        elif is_fail_closed_message(assistant):
            status = "fail_closed"
        else:
            status = "fail"
    else:
        raise ValueError(f"Unknown eval case: {case.name}")

    return {
        "case": case.name,
        "model": model,
        "verifier_model": verifier_model,
        "verification": "on" if verification_enabled else "off",
        "status": status,
        "latency_s": round(elapsed, 2),
        "tool_calls": calls,
        "assumption_audits": assumption_audit_count(session),
        "assumption_audit_retries": assumption_audit_retry_count(session),
        "verification_retries": verification_retry_count(session),
        "verification_rewrites": verification_rewrite_count(session),
        "final": assistant,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run serial verification on/off evals against real Ollama Code models.")
    parser.add_argument("--models", nargs="+", default=["gemma4:e4b", "granite4.1:8b", "gemma3:4b", "qwen3:8b"], help="Models to test serially.")
    parser.add_argument("--modes", nargs="+", choices=["on", "off"], default=["off", "on"], help="Verification modes to run.")
    parser.add_argument("--verifier-model", default=None, help="Optional verifier/rewrite model override to use serially.")
    parser.add_argument("--timeout", type=int, default=600, help="Per-run timeout in seconds.")
    parser.add_argument("--strict-on", action="store_true", help="Exit non-zero if any verification-on run lands outside the acceptable status set for its case.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    available = set(installed_models())
    models = resolve_requested_models(args.models, available)
    if not models:
        print("No requested models are installed locally.")
        return 1

    cases = [
        EvalCase(
            name="tool_use_token",
            prompt="Use read_file on docs/guide.md and reply with the uppercase token only.",
            acceptable_with_verification={"pass"},
        ),
        EvalCase(
            name="git_diff_grounding",
            prompt="Use git_status on src/app.py. Then use git_diff on src/app.py for the working tree only; omit cached or set cached to false. Do not use read_file. Reply with src/app.py and whether the diff adds return 99.",
            acceptable_with_verification={"pass", "fail_closed"},
            prepare=prepare_git_diff_case,
        ),
    ]

    failures: list[dict[str, Any]] = []
    for model in models:
        _LOADED_MODELS.add(model)
        for mode in args.modes:
            verification_enabled = mode == "on"
            for case in cases:
                with workspace_temp_dir("ollama-code-verify-", repo_root) as tmp:
                    workspace = Path(tmp)
                    build_workspace(workspace)
                    outcome = evaluate_case(
                        repo_root,
                        workspace,
                        model,
                        case,
                        verification_enabled=verification_enabled,
                        verifier_model=args.verifier_model,
                        timeout=args.timeout,
                    )
                tools = ",".join(outcome["tool_calls"]) if outcome["tool_calls"] else "-"
                print(
                    "[verify-eval]"
                    f" model={outcome['model']}"
                    f" verifier_model={outcome['verifier_model'] or '-'}"
                    f" verification={outcome['verification']}"
                    f" case={outcome['case']}"
                    f" status={outcome['status']}"
                    f" latency_s={outcome['latency_s']}"
                    f" assumption_audits={outcome['assumption_audits']}"
                    f" assumption_audit_retries={outcome['assumption_audit_retries']}"
                    f" verifier_retries={outcome['verification_retries']}"
                    f" verifier_rewrites={outcome['verification_rewrites']}"
                    f" tools={tools}"
                    f" final={outcome['final']!r}"
                )
                if args.strict_on and verification_enabled and outcome["status"] not in case.acceptable_with_verification:
                    failures.append(outcome)
        unload_model(model)
        _LOADED_MODELS.discard(model)
    if failures:
        print(f"[verify-eval] strict verification failures: {len(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
