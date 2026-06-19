from __future__ import annotations

import argparse
import json
import shutil
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


@dataclass(frozen=True)
class GateStep:
    name: str
    model: str
    command: tuple[str, ...]
    timeout_s: int
    artifact: str | None = None


def _slug_model(model: str) -> str:
    return model.replace(":", "-").replace("/", "-").replace("\\", "-")


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
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root.resolve(strict=False)),
        "output_dir": str(output_dir.resolve(strict=False)),
        "preflight": preflight_result,
        "requested_models": requested_models,
        "resolved_models": resolved_models,
        "steps_requested": [step.name for step in steps],
        "steps_completed": [step["name"] for step in step_results],
        "elapsed_s": elapsed,
        "ok": not failed,
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
    summary_path = args.output_dir / "live-model-gate-summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
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
            "repo_root": str(repo_root.resolve(strict=False)),
            "output_dir": str(args.output_dir.resolve(strict=False)),
            "requested_models": list(args.models),
            "resolved_models": [],
            "steps_requested": [],
            "steps_completed": [],
            "elapsed_s": 0.0,
            "ok": False,
            "preflight_error": str(exc),
            "step_results": [],
        }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[live-model-gate] wrote {summary_path}")
    preflight_payload = payload.get("preflight") if isinstance(payload.get("preflight"), dict) else {}
    host = preflight_payload.get("ollama_host", ollama_host())
    print(
        "[live-model-gate]"
        + f" host={host}"
        + f" resolved_models={','.join(payload.get('resolved_models', []))}"
        + f" ok={payload['ok']}"
        + f" elapsed_s={payload['elapsed_s']}"
    )
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
