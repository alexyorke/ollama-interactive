from __future__ import annotations

import argparse
import fnmatch
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
from urllib.request import urlopen

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from coding_benchmark_eval import default_benchmark_jobs
except ModuleNotFoundError:  # Imported as scripts.terminal_bench_eval in unit tests.
    from scripts.coding_benchmark_eval import default_benchmark_jobs


DEFAULT_DATASET_SPEC = "terminal-bench-core==0.1.1"
DEFAULT_REGISTRY_URL = "https://raw.githubusercontent.com/laude-institute/terminal-bench/main/registry.json"
DEFAULT_OUTPUT_ROOT = ROOT / "scratch" / "terminal-bench"
DEFAULT_CACHE_ROOT = ROOT / "scratch" / "external"
DEFAULT_AGENT_IMPORT = "scripts.terminal_bench_agent:OllamaCodeTerminalBenchAgent"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11437"


@dataclass(frozen=True)
class DatasetRow:
    name: str
    version: str
    github_url: str
    dataset_path: str
    branch: str
    commit_hash: str
    task_id_subset: tuple[str, ...]
    description: str | None = None
    terminal_bench_version: str | None = None


@dataclass(frozen=True)
class TaskInfo:
    task_id: str
    difficulty: str
    category: str
    instruction: str


def parse_dataset_spec(value: str) -> tuple[str, str]:
    text = str(value or "").strip()
    if not text:
        raise ValueError("dataset spec must be non-empty")
    if "==" in text:
        name, version = text.split("==", 1)
        name = name.strip()
        version = version.strip()
        if not name or not version:
            raise ValueError(f"invalid dataset spec: {value!r}")
        return name, version
    return text, "head"


def _rmtree_force(path: Path) -> None:
    def onexc(function: Any, failed_path: str, _excinfo: Any) -> None:
        os.chmod(failed_path, 0o700)
        function(failed_path)

    shutil.rmtree(path, onexc=onexc)


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        command,
        cwd=cwd,
        env=merged_env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def load_registry_rows(*, registry_url: str, local_registry_path: Path | None = None) -> list[DatasetRow]:
    if local_registry_path is not None:
        payload = json.loads(local_registry_path.read_text(encoding="utf-8"))
    else:
        with urlopen(registry_url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    rows: list[DatasetRow] = []
    for item in payload:
        rows.append(
            DatasetRow(
                name=str(item["name"]),
                version=str(item["version"]),
                github_url=str(item["github_url"]),
                dataset_path=str(item["dataset_path"]),
                branch=str(item["branch"]),
                commit_hash=str(item["commit_hash"]),
                task_id_subset=tuple(item.get("task_id_subset") or ()),
                description=item.get("description"),
                terminal_bench_version=item.get("terminal_bench_version"),
            )
        )
    return rows


def get_dataset_row(
    dataset_spec: str,
    *,
    registry_url: str,
    local_registry_path: Path | None = None,
) -> DatasetRow:
    name, version = parse_dataset_spec(dataset_spec)
    for row in load_registry_rows(registry_url=registry_url, local_registry_path=local_registry_path):
        if row.name == name and row.version == version:
            return row
    raise ValueError(f"dataset not found in registry: {dataset_spec}")


def _checkout_commit(target: Path, *, commit_hash: str) -> None:
    if commit_hash == "head":
        return
    head = _run(["git", "rev-parse", "HEAD"], cwd=target, timeout=30)
    if head.returncode == 0 and head.stdout.strip() == commit_hash:
        return
    fetch = _run(["git", "fetch", "--depth", "1", "origin", commit_hash], cwd=target, timeout=300)
    if fetch.returncode != 0:
        raise RuntimeError(fetch.stderr.strip() or fetch.stdout.strip() or "git fetch failed")
    reset = _run(["git", "reset", "--hard", commit_hash], cwd=target, timeout=60)
    if reset.returncode != 0:
        raise RuntimeError(reset.stderr.strip() or reset.stdout.strip() or "git reset failed")


def ensure_dataset_checkout(
    row: DatasetRow,
    *,
    cache_root: Path,
    overwrite: bool = False,
) -> Path:
    target = cache_root / f"{row.name}-{row.version}"
    if overwrite and target.exists():
        _rmtree_force(target)
    if not (target / ".git").exists():
        if target.exists():
            _rmtree_force(target)
        cache_root.mkdir(parents=True, exist_ok=True)
        clone = _run(
            [
                "git",
                "-c",
                "core.autocrlf=false",
                "-c",
                "core.eol=lf",
                "clone",
                "--branch",
                row.branch,
                "--depth",
                "1",
                row.github_url,
                str(target),
            ],
            cwd=cache_root,
            timeout=600,
        )
        if clone.returncode != 0:
            raise RuntimeError(clone.stderr.strip() or clone.stdout.strip() or "git clone failed")
    _checkout_commit(target, commit_hash=row.commit_hash)
    dataset_rel = row.dataset_path.strip().lstrip("./")
    dataset_root = target / dataset_rel
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset path not found: {dataset_root}")
    normalize_task_checkout(dataset_root)
    return dataset_root


def normalize_task_checkout(dataset_root: Path) -> None:
    for path in dataset_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".sh", ".yaml", ".yml", ".dockerfile"} and path.name not in {
            "Dockerfile",
            "docker-compose.yaml",
            "docker-compose.yml",
        }:
            continue
        data = path.read_bytes()
        if b"\r\n" not in data:
            continue
        path.write_bytes(data.replace(b"\r\n", b"\n"))


def load_task_catalog(dataset_root: Path, *, task_id_subset: tuple[str, ...]) -> list[TaskInfo]:
    rows: list[TaskInfo] = []
    allowed = set(task_id_subset)
    for task_yaml in sorted(dataset_root.glob("*/task.yaml")):
        task_id = task_yaml.parent.name
        if allowed and task_id not in allowed:
            continue
        data = yaml.safe_load(task_yaml.read_text(encoding="utf-8")) or {}
        rows.append(
            TaskInfo(
                task_id=task_id,
                difficulty=str(data.get("difficulty") or "unknown"),
                category=str(data.get("category") or "unknown"),
                instruction=str(data.get("instruction") or ""),
            )
        )
    return rows


def select_tasks(
    catalog: list[TaskInfo],
    *,
    task_ids: list[str],
    difficulties: list[str],
    categories: list[str],
    limit: int | None,
) -> list[TaskInfo]:
    selected = list(catalog)
    if task_ids:
        selected = [
            row
            for row in selected
            if any(fnmatch.fnmatchcase(row.task_id, pattern) for pattern in task_ids)
        ]
    if difficulties:
        wanted = {item.lower() for item in difficulties}
        selected = [row for row in selected if row.difficulty.lower() in wanted]
    if categories:
        wanted = {item.lower() for item in categories}
        selected = [row for row in selected if row.category.lower() in wanted]
    if limit is not None:
        selected = selected[: max(0, limit)]
    return selected


def summarize_catalog(rows: list[TaskInfo]) -> str:
    if not rows:
        return "no tasks"
    difficulty_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    for row in rows:
        difficulty_counts[row.difficulty] = difficulty_counts.get(row.difficulty, 0) + 1
        category_counts[row.category] = category_counts.get(row.category, 0) + 1
    difficulty_summary = ", ".join(f"{key}={difficulty_counts[key]}" for key in sorted(difficulty_counts))
    category_summary = ", ".join(f"{key}={category_counts[key]}" for key in sorted(category_counts))
    return f"difficulty[{difficulty_summary}] category[{category_summary}]"


def build_tb_command(
    *,
    dataset_root: Path,
    output_dir: Path,
    selected_tasks: list[TaskInfo],
    model: str,
    jobs: int,
    agent_import_path: str,
    agent_kwargs: list[str],
    no_cleanup: bool,
    log_level: str,
) -> list[str]:
    command = [
        "tb",
        "run",
        "--dataset-path",
        str(dataset_root),
        "--agent-import-path",
        agent_import_path,
        "--model",
        model,
        "--output-path",
        str(output_dir),
        "--n-concurrent",
        str(max(1, jobs)),
        "--log-level",
        log_level,
    ]
    if no_cleanup:
        command.append("--no-cleanup")
    for row in selected_tasks:
        command.extend(["--task-id", row.task_id])
    for kwarg in agent_kwargs:
        command.extend(["--agent-kwarg", kwarg])
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official Terminal-Bench tasks through Ollama Code.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_SPEC, help="Dataset spec, e.g. terminal-bench-core==0.1.1")
    parser.add_argument("--registry-url", default=DEFAULT_REGISTRY_URL, help="Registry JSON URL")
    parser.add_argument("--local-registry-path", type=Path, help="Optional local registry JSON path")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT, help="Dataset checkout cache root")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Run output directory root")
    parser.add_argument("--task-id", action="append", default=[], help="Task id or glob. Repeatable.")
    parser.add_argument("--difficulty", action="append", default=[], help="Difficulty filter. Repeatable.")
    parser.add_argument("--category", action="append", default=[], help="Category filter. Repeatable.")
    parser.add_argument("--limit", type=int, help="Cap selected task count after filtering")
    parser.add_argument("--list-tasks", action="store_true", help="List matching tasks and exit")
    parser.add_argument("--overwrite-dataset", action="store_true", help="Force a fresh dataset checkout")
    parser.add_argument("--model", default="gemma4:e4b", help="Model name to pass to the agent")
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST, help="Host-side Ollama endpoint")
    parser.add_argument("--jobs", type=int, help="Concurrent Terminal-Bench trials")
    parser.add_argument("--max-tool-rounds", type=int, default=20, help="Max tool rounds per task")
    parser.add_argument("--approval", default="auto", help="CLI approval mode")
    parser.add_argument("--agent-import-path", default=DEFAULT_AGENT_IMPORT, help="Custom agent import path")
    parser.add_argument("--agent-kwarg", action="append", default=[], help="Extra agent kwarg key=value. Repeatable.")
    parser.add_argument("--cli-extra-arg", action="append", default=[], help="Extra raw CLI arg. Repeatable.")
    parser.add_argument("--no-cleanup", action="store_true", help="Keep run containers/images for inspection")
    parser.add_argument("--log-level", default="info", help="tb log level")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    row = get_dataset_row(
        args.dataset,
        registry_url=args.registry_url,
        local_registry_path=args.local_registry_path,
    )
    dataset_root = ensure_dataset_checkout(
        row,
        cache_root=args.cache_root,
        overwrite=args.overwrite_dataset,
    )
    catalog = load_task_catalog(dataset_root, task_id_subset=row.task_id_subset)
    selected = select_tasks(
        catalog,
        task_ids=args.task_id,
        difficulties=args.difficulty,
        categories=args.category,
        limit=args.limit,
    )
    if args.list_tasks:
        for task in selected:
            print(f"{task.task_id}\t{task.difficulty}\t{task.category}")
        print(f"\n{len(selected)} task(s) selected from {len(catalog)} total: {summarize_catalog(selected)}")
        return 0
    if not selected:
        print("No tasks matched the requested filters.", file=sys.stderr)
        return 2
    jobs = args.jobs or default_benchmark_jobs(args.ollama_host)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_root / f"{row.name}-{row.version}-{timestamp}"
    cli_extra_args = [str(item) for item in args.cli_extra_arg]
    agent_kwargs = [
        f"ollama_host={args.ollama_host!r}",
        f"approval={args.approval!r}",
        f"max_tool_rounds={max(1, int(args.max_tool_rounds))}",
        "quiet=True",
        "no_indexer=True",
        "require_llm_for_turn=True",
    ]
    if cli_extra_args:
        agent_kwargs.append(f"cli_extra_args={json.dumps(cli_extra_args)}")
    agent_kwargs.extend(args.agent_kwarg)
    command = build_tb_command(
        dataset_root=dataset_root,
        output_dir=output_dir,
        selected_tasks=selected,
        model=args.model,
        jobs=jobs,
        agent_import_path=args.agent_import_path,
        agent_kwargs=agent_kwargs,
        no_cleanup=args.no_cleanup,
        log_level=args.log_level,
    )
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    print(f"Dataset: {row.name}=={row.version}")
    print(f"Dataset path: {dataset_root}")
    print(f"Tasks: {', '.join(task.task_id for task in selected)}")
    print(f"Jobs: {jobs}")
    print(f"Output: {output_dir}")
    print(f"Catalog summary: {summarize_catalog(selected)}")

    result = _run(command, cwd=ROOT, env=env, timeout=None)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
