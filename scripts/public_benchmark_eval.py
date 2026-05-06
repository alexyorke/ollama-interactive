from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import time
import difflib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ollama_code.ollama_client import OllamaClient, OllamaError  # noqa: E402

try:
    from coding_benchmark_eval import comparison_rows, failed_tools, tests_run, tool_calls, usage_totals
    from workspace_temp import workspace_temp_dir
except ModuleNotFoundError:  # Imported as scripts.public_benchmark_eval in unit tests.
    from scripts.coding_benchmark_eval import comparison_rows, failed_tools, tests_run, tool_calls, usage_totals
    from scripts.workspace_temp import workspace_temp_dir


POLYGLOT_REPO_URL = "https://github.com/Aider-AI/polyglot-benchmark.git"
DEFAULT_POLYGLOT_TASKS = ("list-ops", "pig-latin", "wordy")
HARD_POLYGLOT_TASKS = (
    "list-ops",
    "pig-latin",
    "wordy",
    "phone-number",
    "grade-school",
    "variable-length-quantity",
    "robot-name",
    "simple-linked-list",
    "transpose",
    "scale-generator",
)


def _run(command: list[str], cwd: Path, *, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)


def warm_model(model: str, *, timeout: int = 120) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        client = OllamaClient(timeout=timeout)
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": "Reply with {}."}],
            response_format="json",
            think=False,
            options={"num_predict": 8},
        )
        return {"ok": True, "latency_s": round(time.perf_counter() - started, 2), "model": response.model}
    except (OllamaError, OSError, TimeoutError) as exc:
        return {"ok": False, "latency_s": round(time.perf_counter() - started, 2), "error": str(exc)[:300]}


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
        "do not edit tests, replace stubs with complete code, run tests with configured test command, "
        "keep editing until tests pass or tool rounds end, and summarize concise."
    )


def load_session(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"events": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _implementation_python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        name = path.name.lower()
        if name.startswith("test_") or name.endswith("_test.py") or "/tests/" in f"/{rel.lower()}":
            continue
        if "/scratch/" in f"/{rel.lower()}/" or "/.meta/" in f"/{rel.lower()}/":
            continue
        files.append(path)
    return files


def _detach_model_visible_meta(workspace: Path) -> Path | None:
    meta = workspace / ".meta"
    if not meta.exists():
        return None
    detached = workspace.parent / f"{workspace.name}-evaluator-meta"
    if detached.exists():
        _rmtree_force(detached)
    try:
        meta.rename(detached)
    except OSError:
        shutil.copytree(meta, detached)
        _rmtree_force(meta)
    return detached


def _rmtree_force(path: Path) -> None:
    def onexc(function: Any, failed_path: str, _excinfo: Any) -> None:
        try:
            os.chmod(failed_path, 0o700)
            function(failed_path)
        except OSError:
            raise

    shutil.rmtree(path, onexc=onexc)


def _evaluator_meta_snippets(meta_dir: Path | None, *, max_files: int = 3, max_chars: int = 1000) -> list[dict[str, str]]:
    if meta_dir is None or not meta_dir.exists():
        return []
    snippets: list[dict[str, str]] = []
    for path in sorted(meta_dir.rglob("*.py"))[:max_files]:
        try:
            rel = ".meta/" + path.relative_to(meta_dir).as_posix()
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        snippets.append({"path": rel, "content": text[:max_chars]})
    return snippets


def _final_source_snippets(workspace: Path, *, max_files: int = 6, max_chars: int = 1400) -> list[dict[str, str]]:
    snippets: list[dict[str, str]] = []
    for path in _implementation_python_files(workspace)[:max_files]:
        try:
            rel = path.relative_to(workspace).as_posix()
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        snippets.append({"path": rel, "content": text[:max_chars]})
    return snippets


def _changed_source_diffs(source: Path, workspace: Path, *, max_files: int = 8, max_chars: int = 1800) -> list[dict[str, str]]:
    diffs: list[dict[str, str]] = []
    for path in _implementation_python_files(workspace):
        rel = path.relative_to(workspace).as_posix()
        original_path = source / rel
        try:
            before = original_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            before = []
        try:
            after = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        if before == after:
            continue
        diff = "\n".join(
            difflib.unified_diff(
                before,
                after,
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                lineterm="",
                n=3,
            )
        )
        diffs.append({"path": rel, "diff": diff[:max_chars]})
        if len(diffs) >= max_files:
            break
    return diffs


def _source_has_stub(content: str) -> bool:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = [
            child
            for child in node.body
            if not (isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant) and isinstance(child.value.value, str))
        ]
        if not body or len(body) == 1 and isinstance(body[0], ast.Pass):
            return True
        if len(body) == 1 and isinstance(body[0], ast.Return):
            value = body[0].value
            if value is None or (isinstance(value, ast.Constant) and value.value is None):
                return True
    return False


def failure_classes(
    *,
    status: str,
    timed_out: bool,
    final_tests_returncode: int,
    calls: list[str],
    failures: list[dict[str, Any]],
    final_source_snippets: list[dict[str, Any]] | None = None,
    session_events: list[dict[str, Any]] | None = None,
) -> list[str]:
    if status == "pass" and final_tests_returncode == 0:
        return []
    classes: set[str] = set()
    mutating = {"write_file", "replace_symbol", "replace_symbols", "replace_in_file", "apply_structured_edit", "edit_intent"}
    if timed_out:
        classes.add("timeout")
        events = session_events or []
        has_started = any(item.get("type") == "llm_call_started" for item in events)
        has_finished = any(item.get("type") == "llm_call" for item in events)
        if not has_started:
            classes.add("subprocess_kill_before_agent")
        elif not has_finished:
            classes.add("active_generation_timeout")
        elif calls:
            classes.add("controller_loop_timeout")
        if status != "pass" and not any(call in mutating for call in calls):
            classes.add("timeout_before_edit")
    for item in failures:
        summary = str(item.get("summary") or "")
        name = str(item.get("name") or "")
        lowered = summary.lower()
        if "unknown tool" in lowered:
            classes.add("invalid_tool")
        if "bad arguments" in lowered or "invalid" in lowered or "path escapes workspace" in lowered:
            classes.add("invalid_args")
        if "syntaxerror" in summary or "indentationerror" in summary or "syntax error" in lowered:
            classes.add("syntax_edit")
            classes.add("syntax_rejected")
        if "static sanity" in lowered or "undefined local/global" in lowered or "still has stub body" in lowered or "shadowing method" in lowered:
            classes.add("static_sanity_failed")
        if name == "run_test":
            classes.add("tests_still_failing")
    if status != "pass" and not any(call in mutating for call in calls):
        classes.add("no_edit_attempted")
    if final_tests_returncode != 0:
        classes.add("tests_still_failing")
    if final_source_snippets and any(_source_has_stub(str(item.get("content") or "")) for item in final_source_snippets):
        classes.add("partial_stub_completion")
    return sorted(classes)


def evaluate_polyglot_python_task(
    *,
    project_root: Path,
    polyglot_root: Path,
    task: str,
    model: str,
    debate: str,
    reconcile: str,
    timeout: int,
    keep_workspaces_on_fail: bool = False,
) -> dict[str, Any]:
    source = polyglot_task_path(polyglot_root, task)
    bench_root = project_root / "scratch" / "public-bench"
    bench_root.mkdir(parents=True, exist_ok=True)
    keep_workspace = False
    with workspace_temp_dir(f"polyglot-{task}-", bench_root, keep=lambda: keep_workspace) as tmp:
        workspace = Path(tmp)
        shutil.copytree(source, workspace, dirs_exist_ok=True)
        evaluator_meta = _detach_model_visible_meta(workspace)
        session_file = workspace / "scratch" / "session.json"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        test_cmd = python_exercism_test_cmd()
        warmup = warm_model(model)
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
            "--reconcile",
            reconcile,
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
        keep_workspace = keep_workspaces_on_fail and status != "pass"
        calls = tool_calls(session)
        failures = failed_tools(session)
        final_tests_returncode = int(final_tests.returncode)
        final_source_snippets = _final_source_snippets(workspace)
        meta_snippets = _evaluator_meta_snippets(evaluator_meta)
        session_events = list(session.get("events") or []) + list(session.get("llm_telemetry_events") or [])
        if not keep_workspace and evaluator_meta is not None:
            shutil.rmtree(evaluator_meta, ignore_errors=True)
        return {
            "case": task,
            "suite": "aider-polyglot-python-smoke",
            "source": POLYGLOT_REPO_URL,
            "model": model,
            "verifier_model": None,
            "debate": debate,
            "reconcile": reconcile,
            "status": status,
            "acceptable": ["pass"],
            "latency_s": round(time.perf_counter() - started, 2),
            "initial_tests_returncode": initial.returncode,
            "cli_returncode": cli.returncode,
            "final_tests_returncode": final_tests_returncode,
            "usage": usage_totals(session),
            "tool_calls": calls,
            "failed_tools": failures,
            "failure_classes": failure_classes(
                status=status,
                timed_out=timed_out,
                final_tests_returncode=final_tests_returncode,
                calls=calls,
                failures=failures,
                final_source_snippets=final_source_snippets,
                session_events=session_events,
            ),
            "tests_run": tests_run(session),
            "model_warmup": warmup,
            "changed_source_diffs": _changed_source_diffs(source, workspace),
            "final_source_snippets": final_source_snippets,
            "evaluator_meta_snippets": meta_snippets,
            "evaluator_meta": str(evaluator_meta) if keep_workspace and evaluator_meta is not None else "",
            "workspace": str(workspace) if keep_workspace else "",
            "workspace_kept": keep_workspace,
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
    parser.add_argument("--reconcile-modes", nargs="+", choices=["off", "on", "auto"], default=["auto"])
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--task-set", choices=["smoke", "hard"], default="smoke", help="Use the default 3-task smoke set or 10-task hard public sample unless --tasks is provided.")
    parser.add_argument("--cache-dir", default="scratch/external")
    parser.add_argument("--output", default="scratch/public-bench/aider-polyglot-python-smoke.json")
    parser.add_argument("--compare", default=None)
    parser.add_argument("--timeout", type=int, default=480)
    parser.add_argument("--keep-workspaces-on-fail", action="store_true", help="Keep failed copied task workspaces under scratch/public-bench for inspection.")
    parser.add_argument("--strict-accuracy", action="store_true")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    output = Path(args.output)
    if not output.is_absolute():
        output = project_root / output
    polyglot_root = ensure_polyglot_repo(project_root / args.cache_dir)
    tasks = list(args.tasks) if args.tasks is not None else list(HARD_POLYGLOT_TASKS if args.task_set == "hard" else DEFAULT_POLYGLOT_TASKS)
    results: list[dict[str, Any]] = []
    for model in args.models:
        for mode in args.modes:
            for reconcile in args.reconcile_modes:
                for task in tasks:
                    outcome = evaluate_polyglot_python_task(
                        project_root=project_root,
                        polyglot_root=polyglot_root,
                        task=task,
                        model=model,
                        debate=mode,
                        reconcile=reconcile,
                        timeout=args.timeout,
                        keep_workspaces_on_fail=args.keep_workspaces_on_fail,
                    )
                    results.append(outcome)
                    write_payload(output, results=results, partial=True)
                    usage = outcome["usage"]
                    print(
                        "[public-bench]"
                        f" model={model} debate={mode} reconcile={reconcile} case={task} status={outcome['status']}"
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
                f" case={row['case']} reconcile={row.get('reconcile') or '-'} status={row['before_status']}->{row['after_status']}"
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
