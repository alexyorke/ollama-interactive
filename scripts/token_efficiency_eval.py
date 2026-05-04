from __future__ import annotations

import argparse
import atexit
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from e2e_suite import build_workspace, commit_all, installed_models, load_session, run_cli
    from verification_eval import is_fail_closed_message
    from workspace_temp import workspace_temp_dir
except ModuleNotFoundError:
    from scripts.e2e_suite import build_workspace, commit_all, installed_models, load_session, run_cli
    from scripts.verification_eval import is_fail_closed_message
    from scripts.workspace_temp import workspace_temp_dir


_LOADED_MODELS: set[str] = set()


def unload_model(model: str) -> None:
    subprocess.run(["ollama", "stop", model], capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)


def _cleanup_loaded_models() -> None:
    for model in sorted(_LOADED_MODELS):
        unload_model(model)


atexit.register(_cleanup_loaded_models)


def resolve_requested_model(model: str, available: set[str]) -> str | None:
    if model in available:
        return model
    latest = f"{model}:latest"
    if latest in available:
        return latest
    return None


def tool_results(session: dict[str, Any], tool_name: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for event in session.get("events", []):
        if event.get("type") == "tool_result" and event.get("name") == tool_name:
            result = event.get("result")
            if isinstance(result, dict):
                results.append(result)
    return results


def tool_calls(session: dict[str, Any]) -> list[str]:
    return [
        str(event["name"])
        for event in session.get("events", [])
        if event.get("type") == "tool_call" and isinstance(event.get("name"), str)
    ]


def final_assistant_message(session: dict[str, Any]) -> str:
    for event in reversed(session.get("events", [])):
        if event.get("type") == "assistant" and isinstance(event.get("content"), str):
            return str(event["content"]).strip()
    return ""


def usage_totals(session: dict[str, Any]) -> dict[str, Any]:
    events = [event for event in session.get("events", []) if event.get("type") == "llm_call"]
    purposes: dict[str, dict[str, int]] = {}
    totals = {
        "llm_calls": len(events),
        "prompt_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_duration_ns": 0,
        "prompt_chars": 0,
        "response_chars": 0,
    }
    prompt_chars_by_role: dict[str, int] = {}
    top_prompt_messages: list[dict[str, Any]] = []
    for event in events:
        purpose = str(event.get("purpose", "unknown"))
        bucket = purposes.setdefault(purpose, {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        bucket["calls"] += 1
        for key in ("prompt_tokens", "output_tokens", "total_tokens"):
            value = event.get(key)
            if isinstance(value, int):
                totals[key] += value
                bucket[key] += value
        for key in ("total_duration_ns", "prompt_chars", "response_chars"):
            value = event.get(key)
            if isinstance(value, int):
                totals[key] += value
        role_chars = event.get("prompt_chars_by_role")
        if isinstance(role_chars, dict):
            for role, value in role_chars.items():
                if isinstance(role, str) and isinstance(value, int):
                    prompt_chars_by_role[role] = prompt_chars_by_role.get(role, 0) + value
        event_top = event.get("top_prompt_messages")
        if isinstance(event_top, list):
            for item in event_top:
                if not isinstance(item, dict):
                    continue
                chars = item.get("chars")
                if not isinstance(chars, int):
                    continue
                top_prompt_messages.append(
                    {
                        "purpose": purpose,
                        "role": str(item.get("role", "")),
                        "chars": chars,
                        "preview": str(item.get("preview", ""))[:80],
                    }
                )
    totals["purposes"] = purposes
    totals["prompt_chars_by_role"] = dict(sorted(prompt_chars_by_role.items()))
    totals["top_prompt_messages"] = sorted(top_prompt_messages, key=lambda item: int(item["chars"]), reverse=True)[:8]
    return totals


def prepare_git_diff_case(workspace: Path) -> None:
    commit_all(workspace, "checkpoint before token eval")
    (workspace / "src" / "app.py").write_text("def meaning() -> int:\n    return 99\n", encoding="utf-8")


def prepare_large_file_case(workspace: Path) -> None:
    lines = [f"line {index}: filler" for index in range(1, 501)]
    lines[249] = "line 250: NEEDLE_FAST_250"
    (workspace / "docs" / "large.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_code_symbol_case(workspace: Path) -> None:
    before = "\n\n".join(f"def helper_before_{index}():\n    return {index}" for index in range(220))
    after = "\n\n".join(f"def helper_after_{index}():\n    return {index}" for index in range(220))
    target = (
        "def calculate_discount(cart, percentage):\n"
        "    subtotal = sum(item['price'] for item in cart)\n"
        "    marker = 'TOKEN_SYMBOL_750'\n"
        "    discount = subtotal * percentage\n"
        "    return marker if discount >= 0 else 'BAD'\n"
    )
    (workspace / "src" / "large_pricing.py").write_text(f"{before}\n\n{target}\n\n{after}\n", encoding="utf-8")


def validate_token_read(stdout: str, session: dict[str, Any], workspace: Path) -> str:
    reads = tool_results(session, "read_file")
    return "pass" if any(item.get("ok") for item in reads) and "TOKEN_42" in stdout else "fail"


def validate_git_diff(stdout: str, session: dict[str, Any], workspace: Path) -> str:
    calls = tool_calls(session)
    statuses = tool_results(session, "git_status")
    diffs = tool_results(session, "git_diff")
    assistant = final_assistant_message(session)
    if "read_file" in calls:
        return "fail"
    if not any(item.get("ok") for item in statuses):
        return "fail"
    if not any(item.get("ok") and "return 99" in str(item.get("output", "")) for item in diffs):
        return "fail"
    if "return 99" in stdout:
        return "pass"
    return "fail_closed" if is_fail_closed_message(assistant) else "fail"


def validate_exact_literal(stdout: str, session: dict[str, Any], workspace: Path) -> str:
    target = workspace / "scratch" / "exact.txt"
    reads = tool_results(session, "read_file")
    if target.exists() and target.read_text(encoding="utf-8") == "EXACT_OK\n" and any(item.get("ok") for item in reads):
        return "pass" if "EXACT_OK" in stdout else "fail"
    return "fail"


def validate_path_escape(stdout: str, session: dict[str, Any], workspace: Path) -> str:
    reads = tool_results(session, "read_file")
    text = stdout + "\n" + final_assistant_message(session)
    return "pass" if any("escapes the workspace" in str(item.get("summary", "")) for item in reads) and "escapes the workspace" in text else "fail"


def validate_shell_failure(stdout: str, session: dict[str, Any], workspace: Path) -> str:
    shells = tool_results(session, "run_shell")
    if not any(item.get("exit_code") == 5 and item.get("output") == "boom" for item in shells):
        return "fail"
    lowered = stdout.lower()
    return "pass" if "5" in lowered and "boom" in lowered else "fail"


def validate_run_test(stdout: str, session: dict[str, Any], workspace: Path) -> str:
    runs = tool_results(session, "run_test")
    if any(item.get("ok") and "test_sample" in str(item.get("output", "")) and "OK" in str(item.get("output", "")) for item in runs):
        return "pass"
    return "fail"


def validate_repeated_read_cache(stdout: str, session: dict[str, Any], workspace: Path) -> str:
    reads = [event for event in session.get("events", []) if event.get("type") == "tool_result" and event.get("name") == "read_file"]
    if not reads:
        return "fail"
    calls = [
        event.get("arguments") if isinstance(event.get("arguments"), dict) else {}
        for event in session.get("events", [])
        if event.get("type") == "tool_call" and event.get("name") == "read_file"
    ]
    duplicate_call = any(calls[index] == calls[index - 1] for index in range(1, len(calls)))
    if duplicate_call and len(reads) >= 2 and not any(event.get("cached") is True for event in reads[1:]):
        return "fail"
    return "pass" if "TOKEN_42" in stdout else "fail"


def validate_large_file(stdout: str, session: dict[str, Any], workspace: Path) -> str:
    reads = tool_results(session, "read_file")
    targeted = False
    for event in session.get("events", []):
        if event.get("type") != "tool_call" or event.get("name") != "read_file":
            continue
        arguments = event.get("arguments") if isinstance(event.get("arguments"), dict) else {}
        path = str(arguments.get("path", "")).replace("\\", "/")
        if path.endswith("docs/large.md") and int(arguments.get("start", 1)) <= 250 <= int(arguments.get("end", 9999)):
            targeted = True
    return "pass" if targeted and any("NEEDLE_FAST_250" in str(item.get("output", "")) for item in reads) and "NEEDLE_FAST_250" in stdout else "fail"


def validate_code_symbol_navigation(stdout: str, session: dict[str, Any], workspace: Path) -> str:
    calls = tool_calls(session)
    symbol_reads = tool_results(session, "read_symbol")
    if "read_file" in calls:
        return "fail"
    if "search_symbols" not in calls or "read_symbol" not in calls:
        return "fail"
    if not any(item.get("ok") and "TOKEN_SYMBOL_750" in str(item.get("output", "")) for item in symbol_reads):
        return "fail"
    return "pass" if "TOKEN_SYMBOL_750" in stdout else "fail"


def validate_code_symbol_summary(stdout: str, session: dict[str, Any], workspace: Path) -> str:
    calls = tool_calls(session)
    symbol_reads = tool_results(session, "read_symbol")
    if "read_file" in calls:
        return "fail"
    if "search_symbols" not in calls or "read_symbol" not in calls:
        return "fail"
    if not any(item.get("ok") and "return 42" in str(item.get("output", "")) for item in symbol_reads):
        return "fail"
    return "pass" if "42" in stdout else "fail"


@dataclass(frozen=True)
class EvalCase:
    name: str
    prompt: str
    validate: Callable[[str, dict[str, Any], Path], str]
    prepare: Callable[[Path], None] | None = None
    extra_args: tuple[str, ...] = ()
    acceptable: tuple[str, ...] = ("pass",)
    requires_git: bool = False


CASES = [
    EvalCase(
        name="token_read",
        prompt="Use read_file on docs/guide.md and reply with the uppercase token only.",
        validate=validate_token_read,
    ),
    EvalCase(
        name="git_diff_grounding",
        prompt="Use git_status on src/app.py. Then use git_diff on src/app.py for the working tree only; omit cached or set cached to false. Do not use read_file. Reply with src/app.py and whether the diff adds return 99.",
        validate=validate_git_diff,
        prepare=prepare_git_diff_case,
        acceptable=("pass", "fail_closed"),
        requires_git=True,
    ),
    EvalCase(
        name="exact_literal_write_readback",
        prompt="Create scratch/exact.txt with exactly the single line EXACT_OK followed by a newline. Then use read_file to confirm it and reply with EXACT_OK only.",
        validate=validate_exact_literal,
    ),
    EvalCase(
        name="path_escape_error",
        prompt="Use read_file on ../outside.txt and tell me the exact tool error.",
        validate=validate_path_escape,
    ),
    EvalCase(
        name="shell_failure_exact_command",
        prompt="Use run_shell to execute exactly: python3 -c \"import sys; print('boom'); sys.exit(5)\". Then tell me the exit code and the printed word.",
        validate=validate_shell_failure,
    ),
    EvalCase(
        name="run_test_summary",
        prompt="Use run_test and tell me whether tests passed and which test module ran.",
        validate=validate_run_test,
        extra_args=("--test-cmd", "python3 -m unittest discover -s tests -v"),
    ),
    EvalCase(
        name="repeated_read_only_cache",
        prompt="Use read_file on docs/guide.md twice, then reply with the uppercase token only.",
        validate=validate_repeated_read_cache,
    ),
    EvalCase(
        name="forbidden_tool_constraint",
        prompt="Use git_status on src/app.py. Then use git_diff on src/app.py for the working tree only; omit cached or set cached to false. Do not use read_file. Reply with src/app.py and whether the diff adds return 99.",
        validate=validate_git_diff,
        prepare=prepare_git_diff_case,
        acceptable=("pass", "fail_closed"),
        requires_git=True,
    ),
    EvalCase(
        name="large_file_targeted_read",
        prompt="Use read_file on docs/large.md with the smallest useful line range around line 250, then reply with the exact marker token on that line only.",
        validate=validate_large_file,
        prepare=prepare_large_file_case,
    ),
    EvalCase(
        name="large_code_symbol_navigation",
        prompt="Use search_symbols to find calculate_discount in src/large_pricing.py. Then use read_symbol on the exact match. Do not use read_file. Reply with the uppercase TOKEN_SYMBOL marker from that symbol only.",
        validate=validate_code_symbol_navigation,
        prepare=prepare_code_symbol_case,
    ),
    EvalCase(
        name="code_symbol_summary",
        prompt="Use search_symbols to locate meaning in src/app.py, then use read_symbol on the exact match. Do not use read_file. Summarize what value it returns.",
        validate=validate_code_symbol_summary,
    ),
    EvalCase(
        name="verifier_rewrite_recovery",
        prompt="Use git_status on src/app.py. Then use git_diff on src/app.py for the working tree only; omit cached or set cached to false. Do not use read_file. Reply with src/app.py and whether the diff adds return 99.",
        validate=validate_git_diff,
        prepare=prepare_git_diff_case,
        acceptable=("pass", "fail_closed"),
        requires_git=True,
    ),
]


def evaluate_case(
    repo_root: Path,
    model: str,
    verifier_model: str | None,
    mode: str,
    case: EvalCase,
    timeout: int,
) -> dict[str, Any]:
    with workspace_temp_dir("ollama-code-token-", repo_root) as tmp:
        workspace = Path(tmp)
        git_available = build_workspace(workspace)
        if case.requires_git and not git_available:
            return {
                "model": model,
                "verifier_model": verifier_model,
                "debate": mode,
                "case": case.name,
                "status": "skipped",
                "acceptable": [*case.acceptable, "skipped"],
                "latency_s": 0.0,
                "returncode": 0,
                "usage": usage_totals({}),
                "tool_calls": [],
                "assumption_audits": 0,
                "assumption_retries": 0,
                "verifier_retries": 0,
                "stdout_tail": "skipped: git workspace unavailable",
                "stderr_tail": "",
            }
        if case.prepare is not None:
            case.prepare(workspace)
        session_file = workspace / "scratch" / f"{case.name}-{mode}.json"
        extra_args = ["--debate", mode, *case.extra_args]
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
        status = case.validate(result.stdout, session, workspace)
        if result.returncode != 0:
            status = "fail"
        usage = usage_totals(session)
        return {
            "case": case.name,
            "model": model,
            "verifier_model": verifier_model,
            "debate": mode,
            "status": status,
            "acceptable": list(case.acceptable),
            "latency_s": round(elapsed, 2),
            "usage": usage,
            "tool_calls": tool_calls(session),
            "assumption_audits": sum(1 for event in session.get("events", []) if event.get("type") == "assumption_audit"),
            "assumption_audit_retries": sum(1 for event in session.get("events", []) if event.get("type") == "assumption_audit" and event.get("verdict") == "retry"),
            "verification_retries": sum(1 for event in session.get("events", []) if event.get("type") == "verification" and event.get("verdict") == "retry"),
            "verification_rewrites": sum(1 for event in session.get("events", []) if event.get("type") == "verification_rewrite"),
            "final": final_assistant_message(session),
            "stdout_tail": result.stdout[-1000:],
            "stderr_tail": result.stderr[-1000:],
        }


def median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    passing_on = [
        int(item["usage"]["prompt_tokens"])
        for item in results
        if item.get("debate") == "on" and item.get("status") == "pass"
    ]
    return {
        "runs": len(results),
        "pass": sum(1 for item in results if item.get("status") == "pass"),
        "fail_closed": sum(1 for item in results if item.get("status") == "fail_closed"),
        "fail": sum(1 for item in results if item.get("status") == "fail"),
        "median_prompt_tokens_passing_debate_on": median(passing_on),
    }


def comparison_rows(current: list[dict[str, Any]], baseline: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if baseline is None:
        return []
    index = {
        (item.get("model"), item.get("verifier_model"), item.get("debate"), item.get("case")): item
        for item in baseline
    }
    rows: list[dict[str, Any]] = []
    for item in current:
        key = (item.get("model"), item.get("verifier_model"), item.get("debate"), item.get("case"))
        before = index.get(key)
        if before is None:
            continue
        before_prompt = int(before["usage"]["prompt_tokens"])
        after_prompt = int(item["usage"]["prompt_tokens"])
        delta_pct = 0.0 if before_prompt == 0 else round((after_prompt - before_prompt) * 100 / before_prompt, 1)
        rows.append(
            {
                "model": item.get("model"),
                "verifier_model": item.get("verifier_model"),
                "debate": item.get("debate"),
                "case": item.get("case"),
                "before_status": before.get("status"),
                "after_status": item.get("status"),
                "before_prompt_tokens": before_prompt,
                "after_prompt_tokens": after_prompt,
                "prompt_delta_pct": delta_pct,
                "before_llm_calls": before["usage"]["llm_calls"],
                "after_llm_calls": item["usage"]["llm_calls"],
                "before_latency_s": before.get("latency_s"),
                "after_latency_s": item.get("latency_s"),
            }
        )
    return rows


def print_table(results: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> None:
    for item in results:
        usage = item["usage"]
        print(
            "[token-eval]"
            f" model={item['model']}"
            f" verifier={item['verifier_model'] or '-'}"
            f" debate={item['debate']}"
            f" case={item['case']}"
            f" status={item['status']}"
            f" calls={usage['llm_calls']}"
            f" prompt={usage['prompt_tokens']}"
            f" output={usage['output_tokens']}"
            f" latency_s={item['latency_s']}"
            f" tools={','.join(item['tool_calls']) or '-'}"
        )
    if comparisons:
        print("[token-eval] comparison")
        for row in comparisons:
            print(
                "[token-eval]"
                f" model={row['model']}"
                f" verifier={row['verifier_model'] or '-'}"
                f" debate={row['debate']}"
                f" case={row['case']}"
                f" status={row['before_status']}->{row['after_status']}"
                f" prompt={row['before_prompt_tokens']}->{row['after_prompt_tokens']}"
                f" delta_pct={row['prompt_delta_pct']}"
                f" calls={row['before_llm_calls']}->{row['after_llm_calls']}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run serial token-efficiency evals against local Ollama models.")
    parser.add_argument("--models", nargs="+", default=["gemma3:4b", "qwen3:8b", "granite4.1:8b"], help="Primary models to run serially.")
    parser.add_argument("--verifier-pairs", nargs="*", default=["gemma3:4b=granite4.1:8b"], help="Optional primary=verifier entries, run debate-on only.")
    parser.add_argument("--modes", nargs="+", choices=["off", "on"], default=["off", "on"], help="Debate modes for primary-only models.")
    parser.add_argument("--cases", nargs="*", default=[case.name for case in CASES], help="Scenario names to run.")
    parser.add_argument("--output", required=True, help="Raw JSON output path.")
    parser.add_argument("--compare", default=None, help="Optional baseline JSON path.")
    parser.add_argument("--timeout", type=int, default=600, help="Per-case timeout in seconds.")
    parser.add_argument("--strict-accuracy", action="store_true", help="Exit non-zero for unacceptable statuses.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    available = set(installed_models())
    cases = [case for case in CASES if case.name in set(args.cases)]
    if not cases:
        raise SystemExit("No eval cases selected.")

    matrix: list[tuple[str, str | None, list[str]]] = []
    for requested in args.models:
        model = resolve_requested_model(requested, available)
        if model is not None:
            matrix.append((model, None, list(args.modes)))
    for pair in args.verifier_pairs:
        if "=" not in pair:
            raise SystemExit(f"Bad verifier pair {pair!r}; use primary=verifier.")
        primary_raw, verifier_raw = pair.split("=", 1)
        primary = resolve_requested_model(primary_raw, available)
        verifier = resolve_requested_model(verifier_raw, available)
        if primary is not None and verifier is not None:
            matrix.append((primary, verifier, ["on"]))
    if not matrix:
        raise SystemExit("No requested models are installed.")

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for model, verifier_model, modes in matrix:
        _LOADED_MODELS.add(model)
        if verifier_model:
            _LOADED_MODELS.add(verifier_model)
        for mode in modes:
            for case in cases:
                outcome = evaluate_case(repo_root, model, verifier_model, mode, case, args.timeout)
                results.append(outcome)
                acceptable_statuses = outcome.get("acceptable", case.acceptable)
                if args.strict_accuracy and outcome["status"] not in acceptable_statuses:
                    failures.append(outcome)
                print_table([outcome], [])
        unload_model(model)
        _LOADED_MODELS.discard(model)
        if verifier_model:
            unload_model(verifier_model)
            _LOADED_MODELS.discard(verifier_model)

    baseline_results = None
    if args.compare:
        baseline_payload = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        baseline_results = baseline_payload.get("results") if isinstance(baseline_payload.get("results"), list) else []
    comparisons = comparison_rows(results, baseline_results)
    accuracy_regressions = [
        row
        for row in comparisons
        if row.get("before_status") == "pass" and row.get("after_status") != "pass"
    ]
    if args.strict_accuracy:
        failures.extend(accuracy_regressions)
    if comparisons:
        print_table([], comparisons)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, text=True).strip(),
        "summary": summarize(results),
        "results": results,
        "comparisons": comparisons,
        "accuracy_regressions": accuracy_regressions,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if failures:
        print(f"[token-eval] strict accuracy failures: {len(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
