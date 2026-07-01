from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from e2e_suite import installed_models, ollama_host
except ModuleNotFoundError:
    from scripts.e2e_suite import installed_models, ollama_host


DEFAULT_OUTPUT_DIR = Path("scratch") / "live-model-gate"
DEFAULT_MODELS = ("granite4.1:8b", "gemma4:e4b", "qwen3:8b")
DEFAULT_E2E_SCENARIOS = ("scenario_transcripted_tool_use", "scenario_run_test")
DEFAULT_BENCHMARK_FEATURE_PROFILES = ("all",)
DEFAULT_BENCHMARK_CLASSES = ("agent", "controller")
DEFAULT_BENCHMARK_MODES = ("off",)
DEFAULT_SELECTION_MODEL = "granite4.1:8b"
SUMMARY_FILENAME = "live-model-gate-summary.json"


@dataclass(frozen=True)
class GateStep:
    name: str
    model: str
    command: tuple[str, ...]
    timeout_s: int
    artifact: str | None = None


def _slug_model(model: str) -> str:
    return model.replace(":", "-").replace("/", "-").replace("\\", "-")


def _load_json_object(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _git_commit(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _git_dirty(repo_root: Path) -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return bool(completed.stdout.strip()) if completed.returncode == 0 else False


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _benchmark_median_latency_s(payload: dict[str, Any]) -> float | None:
    results = payload.get("results")
    if not isinstance(results, list):
        return None
    latencies = [
        parsed
        for parsed in (_safe_float(item.get("latency_s")) for item in results if isinstance(item, dict))
        if parsed is not None
    ]
    if not latencies:
        return None
    return round(float(statistics.median(latencies)), 3)


def _step_outcome_by_prefix(step_results: list[dict[str, Any]], prefix: str, model: str) -> dict[str, Any] | None:
    target = f"{prefix}:{model}"
    for row in step_results:
        if isinstance(row, dict) and str(row.get("name") or "") == target:
            return row
    return None


def build_model_rows(models: list[str], step_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        e2e = _step_outcome_by_prefix(step_results, "e2e", model)
        verification = _step_outcome_by_prefix(step_results, "verification", model)
        benchmark = _step_outcome_by_prefix(step_results, "benchmark", model)
        benchmark_artifact = Path(str(benchmark.get("artifact"))) if isinstance(benchmark, dict) and benchmark.get("artifact") else None
        benchmark_payload = _load_json_object(benchmark_artifact)
        benchmark_summary = benchmark_payload.get("summary") if isinstance(benchmark_payload.get("summary"), dict) else {}
        rows.append(
            {
                "model": model,
                "e2e_ok": e2e.get("ok") if isinstance(e2e, dict) else None,
                "verification_ok": verification.get("ok") if isinstance(verification, dict) else None,
                "benchmark_ok": benchmark.get("ok") if isinstance(benchmark, dict) else None,
                "benchmark_passes": _safe_int(benchmark_summary.get("pass")),
                "benchmark_runs": _safe_int(benchmark_summary.get("runs")),
                "benchmark_total_tokens": _safe_int(benchmark_summary.get("total_tokens")),
                "benchmark_total_llm_calls": _safe_int(benchmark_summary.get("total_llm_calls")),
                "benchmark_median_latency_s": _benchmark_median_latency_s(benchmark_payload),
                "benchmark_artifact": str(benchmark_artifact) if benchmark_artifact else None,
            }
        )
    return rows


def choose_default_model(model_rows: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    eligible = [
        row
        for row in model_rows
        if _safe_int(row.get("benchmark_passes")) is not None and _safe_int(row.get("benchmark_runs")) is not None
    ]
    if not eligible:
        return None, None

    def sort_key(row: dict[str, Any]) -> tuple[int, float, float, int]:
        passes = _safe_int(row.get("benchmark_passes"))
        tokens = _safe_int(row.get("benchmark_total_tokens"))
        latency = _safe_float(row.get("benchmark_median_latency_s"))
        return (
            -(passes if passes is not None else -1),
            float(tokens if tokens is not None else 10**18),
            float(latency if latency is not None else 10**18),
            0 if str(row.get("model") or "") == DEFAULT_SELECTION_MODEL else 1,
        )

    selected = min(eligible, key=sort_key)
    selected_model = str(selected.get("model") or "")
    top_passes = max(_safe_int(row.get("benchmark_passes")) or 0 for row in eligible)
    top_pass_rows = [row for row in eligible if (_safe_int(row.get("benchmark_passes")) or 0) == top_passes]
    selected_passes = _safe_int(selected.get("benchmark_passes")) or 0
    selected_runs = _safe_int(selected.get("benchmark_runs")) or 0
    if len(top_pass_rows) == 1:
        return selected_model, f"Selected {selected_model} because it had the highest benchmark pass count ({selected_passes}/{selected_runs})."

    token_candidates = [row for row in top_pass_rows if _safe_int(row.get("benchmark_total_tokens")) is not None]
    if token_candidates:
        lowest_tokens = min(_safe_int(row.get("benchmark_total_tokens")) or 0 for row in token_candidates)
        lowest_token_rows = [row for row in token_candidates if (_safe_int(row.get("benchmark_total_tokens")) or 0) == lowest_tokens]
        if len(lowest_token_rows) == 1:
            token_values = ", ".join(
                str(_safe_int(row.get("benchmark_total_tokens")) or 0)
                for row in top_pass_rows
            )
            return (
                selected_model,
                f"Selected {selected_model} because benchmark pass count tied at {selected_passes}/{selected_runs} and it used the fewest benchmark tokens ({token_values}).",
            )
    latency_candidates = [row for row in top_pass_rows if _safe_float(row.get("benchmark_median_latency_s")) is not None]
    if latency_candidates:
        lowest_latency = min(_safe_float(row.get("benchmark_median_latency_s")) or 0.0 for row in latency_candidates)
        lowest_latency_rows = [
            row for row in latency_candidates if (_safe_float(row.get("benchmark_median_latency_s")) or 0.0) == lowest_latency
        ]
        if len(lowest_latency_rows) == 1:
            return (
                selected_model,
                f"Selected {selected_model} because benchmark pass count and token totals tied, and it had the lowest benchmark median latency ({lowest_latency:.3f}s).",
            )
    if selected_model == DEFAULT_SELECTION_MODEL:
        return selected_model, f"Selected {selected_model} because benchmark pass count, total tokens, and median latency tied, so the configured Granite default won the tie-break."
    return selected_model, f"Selected {selected_model} as the best available benchmark tie-break winner among the requested models."


def summary_contract_ok(payload: dict[str, Any]) -> bool:
    benchmark_suite = payload.get("benchmark_suite")
    selected_default_model = payload.get("selected_default_model")
    selection_reason = payload.get("selection_reason")
    git_commit = payload.get("git_commit")
    git_dirty = payload.get("git_dirty")
    models = payload.get("models")
    ok = payload.get("ok")
    if not isinstance(benchmark_suite, str) or not benchmark_suite.strip():
        return False
    if not isinstance(git_commit, str) or not git_commit.strip():
        return False
    if not isinstance(git_dirty, bool):
        return False
    if not isinstance(models, list):
        return False
    if ok is True:
        if not isinstance(selected_default_model, str) or not selected_default_model.strip():
            return False
        if not isinstance(selection_reason, str) or not selection_reason.strip():
            return False
        model_rows = [row for row in models if isinstance(row, dict)]
        winner, _reason = choose_default_model(model_rows)
        if winner is None or winner != selected_default_model:
            return False
        return True
    if selected_default_model is not None and (not isinstance(selected_default_model, str) or not selected_default_model.strip()):
        return False
    if selection_reason is not None and (not isinstance(selection_reason, str) or not selection_reason.strip()):
        return False
    return True


def write_summary_artifacts(payload: dict[str, Any], output_dir: Path, *, repo_root: Path) -> list[Path]:
    summary_path = output_dir / SUMMARY_FILENAME
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2)
    summary_path.write_text(encoded, encoding="utf-8")
    written = [summary_path]
    if DEFAULT_OUTPUT_DIR.is_absolute():
        canonical_root = DEFAULT_OUTPUT_DIR
    else:
        scratch_root = (repo_root / "scratch").resolve(strict=False)
        output_root = output_dir.resolve(strict=False)
        if scratch_root not in {output_root, *output_root.parents}:
            return written
        canonical_root = repo_root / DEFAULT_OUTPUT_DIR
    canonical_path = canonical_root / SUMMARY_FILENAME
    if canonical_path.resolve(strict=False) != summary_path.resolve(strict=False):
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        canonical_path.write_text(encoded, encoding="utf-8")
        written.append(canonical_path)
    return written


def resolve_requested_models(requested: list[str], available: set[str]) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for model in requested:
        candidate: str | None = None
        if model in available:
            candidate = model
        else:
            latest = f"{model}:latest"
            if latest in available:
                candidate = latest
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        resolved.append(candidate)
    return resolved


def preflight(requested_models: list[str]) -> dict[str, Any]:
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        raise RuntimeError("ollama executable is not available on PATH")
    try:
        available = set(installed_models())
    except Exception as exc:
        raise RuntimeError(f"could not reach Ollama at {ollama_host()}: {exc}") from exc
    resolved = resolve_requested_models(requested_models, available)
    if not resolved:
        raise RuntimeError(
            "None of the requested models are installed on "
            + f"{ollama_host()}. Requested: {requested_models}. Available: {sorted(available)}"
        )
    return {
        "ollama_path": ollama_bin,
        "ollama_host": ollama_host(),
        "available_models": sorted(available),
        "resolved_models": resolved,
    }


def build_steps(
    repo_root: Path,
    output_dir: Path,
    models: list[str],
    *,
    run_e2e: bool,
    run_verification: bool,
    run_benchmarks: bool,
    e2e_scenarios: list[str],
    e2e_timeout: int,
    verification_timeout: int,
    benchmark_timeout: int,
    benchmark_suite: str,
    benchmark_modes: list[str],
    benchmark_feature_profiles: list[str],
    benchmark_classes: list[str],
    benchmark_jobs: int,
) -> list[GateStep]:
    steps: list[GateStep] = []
    scripts_root = repo_root / "scripts"
    for model in models:
        model_slug = _slug_model(model)
        if run_e2e:
            steps.append(
                GateStep(
                    name=f"e2e:{model}",
                    model=model,
                    command=(
                        sys.executable,
                        str(scripts_root / "e2e_suite.py"),
                        "--model",
                        model,
                        "--scenarios",
                        *e2e_scenarios,
                    ),
                    timeout_s=e2e_timeout,
                )
            )
        if run_verification:
            steps.append(
                GateStep(
                    name=f"verification:{model}",
                    model=model,
                    command=(
                        sys.executable,
                        str(scripts_root / "verification_eval.py"),
                        "--models",
                        model,
                        "--modes",
                        "on",
                        "--strict-on",
                        "--timeout",
                        str(verification_timeout),
                    ),
                    timeout_s=max(verification_timeout * 2, verification_timeout + 300),
                )
            )
        if run_benchmarks:
            benchmark_output = output_dir / f"coding-benchmark-{model_slug}.json"
            steps.append(
                GateStep(
                    name=f"benchmark:{model}",
                    model=model,
                    command=(
                        sys.executable,
                        str(scripts_root / "coding_benchmark_eval.py"),
                        "--suite",
                        benchmark_suite,
                        "--models",
                        model,
                        "--modes",
                        *benchmark_modes,
                        "--feature-profiles",
                        *benchmark_feature_profiles,
                        "--benchmark-classes",
                        *benchmark_classes,
                        "--jobs",
                        str(benchmark_jobs),
                        "--timeout",
                        str(benchmark_timeout),
                        "--strict-accuracy",
                        "--strict-budget",
                        "--require-llm-for-agent-benchmarks",
                        "--output",
                        str(benchmark_output),
                    ),
                    timeout_s=max(benchmark_timeout * 4, benchmark_timeout + 900),
                    artifact=str(benchmark_output),
                )
            )
    return steps


def run_step(step: GateStep, *, cwd: Path) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(
        list(step.command),
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=step.timeout_s,
        check=False,
    )
    elapsed = round(time.perf_counter() - started, 2)
    return {
        "name": step.name,
        "model": step.model,
        "command": list(step.command),
        "timeout_s": step.timeout_s,
        "artifact": step.artifact,
        "returncode": result.returncode,
        "ok": result.returncode == 0,
        "latency_s": elapsed,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def run_gate(
    repo_root: Path,
    output_dir: Path,
    *,
    requested_models: list[str],
    run_e2e: bool,
    run_verification: bool,
    run_benchmarks: bool,
    e2e_scenarios: list[str],
    e2e_timeout: int,
    verification_timeout: int,
    benchmark_timeout: int,
    benchmark_suite: str,
    benchmark_modes: list[str],
    benchmark_feature_profiles: list[str],
    benchmark_classes: list[str],
    benchmark_jobs: int,
    continue_on_failure: bool,
) -> dict[str, Any]:
    preflight_result = preflight(requested_models)
    resolved_models = list(preflight_result["resolved_models"])
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = build_steps(
        repo_root,
        output_dir,
        resolved_models,
        run_e2e=run_e2e,
        run_verification=run_verification,
        run_benchmarks=run_benchmarks,
        e2e_scenarios=e2e_scenarios,
        e2e_timeout=e2e_timeout,
        verification_timeout=verification_timeout,
        benchmark_timeout=benchmark_timeout,
        benchmark_suite=benchmark_suite,
        benchmark_modes=benchmark_modes,
        benchmark_feature_profiles=benchmark_feature_profiles,
        benchmark_classes=benchmark_classes,
        benchmark_jobs=benchmark_jobs,
    )
    started = time.perf_counter()
    step_results: list[dict[str, Any]] = []
    failed = False
    for step in steps:
        outcome = run_step(step, cwd=repo_root)
        step_results.append(outcome)
        if not outcome["ok"]:
            failed = True
            if not continue_on_failure:
                break
    elapsed = round(time.perf_counter() - started, 2)
    model_rows = build_model_rows(resolved_models, step_results)
    selected_default_model, selection_reason = choose_default_model(model_rows)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "git_dirty": _git_dirty(repo_root),
        "repo_root": str(repo_root.resolve(strict=False)),
        "output_dir": str(output_dir.resolve(strict=False)),
        "benchmark_suite": benchmark_suite,
        "preflight": preflight_result,
        "requested_models": requested_models,
        "resolved_models": resolved_models,
        "selected_default_model": selected_default_model,
        "selection_reason": selection_reason,
        "steps_requested": [step.name for step in steps],
        "steps_completed": [step["name"] for step in step_results],
        "elapsed_s": elapsed,
        "ok": not failed,
        "models": model_rows,
        "step_results": step_results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a release-style live-model gate against local Ollama models and persist a proof artifact.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), help="Installed local Ollama models to verify.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for gate summary JSON and benchmark outputs.")
    parser.add_argument("--skip-e2e", action="store_true", help="Skip the end-to-end Ollama CLI smoke scenarios.")
    parser.add_argument("--skip-verification", action="store_true", help="Skip the verification-on gate.")
    parser.add_argument("--skip-benchmarks", action="store_true", help="Skip the coding benchmark gate.")
    parser.add_argument("--continue-on-failure", action="store_true", help="Continue running later steps after a failure.")
    parser.add_argument("--e2e-scenarios", nargs="+", default=list(DEFAULT_E2E_SCENARIOS), help="Scenario names to pass to scripts/e2e_suite.py.")
    parser.add_argument("--e2e-timeout", type=int, default=600, help="Per e2e scenario-set timeout in seconds.")
    parser.add_argument("--verification-timeout", type=int, default=600, help="Per verification_eval case timeout in seconds.")
    parser.add_argument("--benchmark-suite", choices=["local-small", "local-full", "external-smoke"], default="local-small")
    parser.add_argument("--benchmark-modes", nargs="+", choices=["off", "on"], default=list(DEFAULT_BENCHMARK_MODES))
    parser.add_argument("--benchmark-feature-profiles", nargs="+", default=list(DEFAULT_BENCHMARK_FEATURE_PROFILES), help="Feature profiles for coding_benchmark_eval.")
    parser.add_argument("--benchmark-classes", nargs="+", choices=["agent", "controller"], default=list(DEFAULT_BENCHMARK_CLASSES))
    parser.add_argument("--benchmark-jobs", type=int, default=1, help="Parallel jobs for coding_benchmark_eval.")
    parser.add_argument("--benchmark-timeout", type=int, default=600, help="Per benchmark turn timeout in seconds.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    try:
        payload = run_gate(
            repo_root,
            args.output_dir,
            requested_models=list(args.models),
            run_e2e=not args.skip_e2e,
            run_verification=not args.skip_verification,
            run_benchmarks=not args.skip_benchmarks,
            e2e_scenarios=list(args.e2e_scenarios),
            e2e_timeout=args.e2e_timeout,
            verification_timeout=args.verification_timeout,
            benchmark_timeout=args.benchmark_timeout,
            benchmark_suite=args.benchmark_suite,
            benchmark_modes=list(args.benchmark_modes),
            benchmark_feature_profiles=list(args.benchmark_feature_profiles),
            benchmark_classes=list(args.benchmark_classes),
            benchmark_jobs=args.benchmark_jobs,
            continue_on_failure=args.continue_on_failure,
        )
    except Exception as exc:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(repo_root),
            "git_dirty": _git_dirty(repo_root),
            "repo_root": str(repo_root.resolve(strict=False)),
            "output_dir": str(args.output_dir.resolve(strict=False)),
            "benchmark_suite": args.benchmark_suite,
            "requested_models": list(args.models),
            "resolved_models": [],
            "selected_default_model": None,
            "selection_reason": None,
            "steps_requested": [],
            "steps_completed": [],
            "elapsed_s": 0.0,
            "ok": False,
            "preflight_error": str(exc),
            "models": [],
            "step_results": [],
        }
    written_paths = write_summary_artifacts(payload, args.output_dir, repo_root=repo_root)
    summary_path = written_paths[0]
    print(f"[live-model-gate] wrote {summary_path}")
    for mirror_path in written_paths[1:]:
        print(f"[live-model-gate] mirrored {mirror_path}")
    preflight_payload = payload.get("preflight") if isinstance(payload.get("preflight"), dict) else {}
    host = preflight_payload.get("ollama_host", ollama_host())
    print(
        "[live-model-gate]"
        + f" host={host}"
        + f" resolved_models={','.join(payload.get('resolved_models', []))}"
        + f" selected_default_model={payload.get('selected_default_model') or '-'}"
        + f" ok={payload['ok']}"
        + f" elapsed_s={payload['elapsed_s']}"
    )
    if payload.get("selection_reason"):
        print(f"[live-model-gate] selection_reason={payload['selection_reason']}")
    if payload.get("preflight_error"):
        print(f"[live-model-gate] preflight_error={payload['preflight_error']}")
    for row in payload["step_results"]:
        print(
            "[live-model-gate]"
            + f" step={row['name']}"
            + f" ok={row['ok']}"
            + f" latency_s={row['latency_s']}"
            + f" artifact={row['artifact'] or '-'}"
        )
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
