from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from coding_benchmark_eval import comparison_rows, failed_tools, tests_run, tool_calls, usage_totals
except ModuleNotFoundError:  # Imported as scripts.public_benchmark_eval in unit tests.
    from scripts.coding_benchmark_eval import comparison_rows, failed_tools, tests_run, tool_calls, usage_totals


POLYGLOT_REPO_URL = "https://github.com/Aider-AI/polyglot-benchmark.git"
DEFAULT_POLYGLOT_TASKS = ("list-ops", "pig-latin", "wordy")


def _run(command: list[str], cwd: Path, *, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)


def ensure_polyglot_repo(cache_dir: Path, *, repo_url: str = POLYGLOT_REPO_URL) -> Path:
    target = cache_dir / "polyglot-benchmark"
    if (target / ".git").exists():
        return target
    cache_dir.mkdir(parents=True, exist_ok=True)
    result = _run(["git", "clone", "--depth", "1", repo_url, str(target)], cache_dir, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "git clone failed")
    return target


def polyglot_task_path(repo_root: Path, task: str, *, language: str = "python") -> Path:
    path = repo_root / language / "exercises" / "practice" / task
    if not path.is_dir():
        raise ValueError(f"Unknown polyglot task: {language}/{task}")
    return path


def python_exercism_test_cmd() -> list[str]:
    return [sys.executable, "-m", "unittest", "discover", "-p", "*_test.py", "-v"]


def public_task_prompt(language: str) -> str:
    return (
        f"Implement this {language} Exercism exercise. Read tests and source, edit only implementation files, "
        "run tests with configured test command, and summarize concise."
    )


def load_session(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"events": []}
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_polyglot_python_task(
    *,
    project_root: Path,
    polyglot_root: Path,
    task: str,
    model: str,
    debate: str,
    timeout: int,
) -> dict[str, Any]:
    source = polyglot_task_path(polyglot_root, task)
    bench_root = project_root / "scratch" / "public-bench"
    bench_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"polyglot-{task}-", dir=bench_root) as tmp:
        workspace = Path(tmp)
        shutil.copytree(source, workspace, dirs_exist_ok=True)
        session_file = workspace / "scratch" / "session.json"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        test_cmd = python_exercism_test_cmd()
        started = time.perf_counter()
        initial = _run(test_cmd, workspace, timeout=120)
        cli_cmd = [
            sys.executable,
            "-m",
            "ollama_code.cli",
            "--cwd",
            str(workspace),
            "--model",
            model,
            "--approval",
            "auto",
            "--debate",
            debate,
            "--quiet",
            "--max-tool-rounds",
            "16",
            "--test-cmd",
            subprocess.list2cmdline(test_cmd),
            "--session-file",
            str(session_file),
            public_task_prompt("Python"),
        ]
        timed_out = False
        try:
            cli = _run(cli_cmd, project_root, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            cli = subprocess.CompletedProcess(cli_cmd, 124, exc.stdout or "", exc.stderr or str(exc))
        final_tests = _run(test_cmd, workspace, timeout=180)
        session = load_session(session_file)
        status = "pass" if cli.returncode == 0 and final_tests.returncode == 0 else "fail"
        if timed_out:
            status = "fail"
        return {
            "case": task,
            "suite": "aider-polyglot-python-smoke",
            "source": POLYGLOT_REPO_URL,
            "model": model,
            "verifier_model": None,
            "debate": debate,
            "status": status,
            "acceptable": ["pass"],
            "latency_s": round(time.perf_counter() - started, 2),
            "initial_tests_returncode": initial.returncode,
            "cli_returncode": cli.returncode,
            "final_tests_returncode": final_tests.returncode,
            "usage": usage_totals(session),
            "tool_calls": tool_calls(session),
            "failed_tools": failed_tools(session),
            "tests_run": tests_run(session),
            "stdout_tail": str(cli.stdout or "")[-1200:],
            "stderr_tail": str(cli.stderr or "")[-1200:],
            "test_tail": (final_tests.stdout + final_tests.stderr)[-1200:],
        }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "runs": len(results),
        "pass": sum(1 for item in results if item.get("status") == "pass"),
        "fail": sum(1 for item in results if item.get("status") != "pass"),
        "total_llm_calls": sum(int(item.get("usage", {}).get("llm_calls", 0)) for item in results),
        "total_tokens": sum(int(item.get("usage", {}).get("total_tokens", 0)) for item in results),
    }


def write_payload(
    output: Path,
    *,
    results: list[dict[str, Any]],
    comparisons: list[dict[str, Any]] | None = None,
    partial: bool = False,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite": "aider-polyglot-python-smoke",
        "partial": partial,
        "summary": summarize(results),
        "results": results,
        "comparisons": comparisons or [],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run serial public coding benchmark smoke tests.")
    parser.add_argument("--models", nargs="+", default=["granite4.1:8b"])
    parser.add_argument("--modes", nargs="+", choices=["off", "on"], default=["off"])
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_POLYGLOT_TASKS))
    parser.add_argument("--cache-dir", default="scratch/external")
    parser.add_argument("--output", default="scratch/public-bench/aider-polyglot-python-smoke.json")
    parser.add_argument("--compare", default=None)
    parser.add_argument("--timeout", type=int, default=480)
    parser.add_argument("--strict-accuracy", action="store_true")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    output = Path(args.output)
    if not output.is_absolute():
        output = project_root / output
    polyglot_root = ensure_polyglot_repo(project_root / args.cache_dir)
    results: list[dict[str, Any]] = []
    for model in args.models:
        for mode in args.modes:
            for task in args.tasks:
                outcome = evaluate_polyglot_python_task(
                    project_root=project_root,
                    polyglot_root=polyglot_root,
                    task=task,
                    model=model,
                    debate=mode,
                    timeout=args.timeout,
                )
                results.append(outcome)
                write_payload(output, results=results, partial=True)
                usage = outcome["usage"]
                print(
                    "[public-bench]"
                    f" model={model} debate={mode} case={task} status={outcome['status']}"
                    f" calls={usage['llm_calls']} tokens={usage['total_tokens']}"
                    f" latency_s={outcome['latency_s']}"
                )

    baseline_results = None
    if args.compare:
        baseline_payload = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        baseline_results = baseline_payload.get("results") if isinstance(baseline_payload.get("results"), list) else []
    comparisons = comparison_rows(results, baseline_results)
    if comparisons:
        for row in comparisons:
            print(
                "[public-bench] comparison"
                f" case={row['case']} status={row['before_status']}->{row['after_status']}"
                f" tokens={row['before_total_tokens']}->{row['after_total_tokens']}"
                f" delta_pct={row['total_token_delta_pct']}"
                f" calls={row['before_llm_calls']}->{row['after_llm_calls']}"
            )
    write_payload(output, results=results, comparisons=comparisons, partial=False)

    if args.strict_accuracy and any(item.get("status") != "pass" for item in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
