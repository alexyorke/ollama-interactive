from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ollama_code.tools import ToolExecutor


@dataclass(frozen=True)
class SdkCase:
    name: str
    query: str
    expected: tuple[str, ...]


CASES = (
    SdkCase("json_loads", "parse json string into Python object", ("json.loads",)),
    SdkCase("pathlib_glob", "find files recursively by wildcard pattern", ("pathlib.Path.rglob", "pathlib.Path.glob")),
    SdkCase("temporary_directory", "temporary directory context manager cleanup", ("tempfile.TemporaryDirectory",)),
    SdkCase("lru_cache", "memoize a function with least recently used cache", ("functools.lru_cache",)),
    SdkCase("counter", "count frequency of hashable items", ("collections.Counter",)),
    SdkCase("subprocess_run", "run a subprocess command and capture output", ("subprocess.run",)),
)


def _contains_expected(output: str, expected: tuple[str, ...]) -> bool:
    return any(item in output for item in expected)


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    workspace = Path(args.workspace).resolve(strict=False)
    tools = ToolExecutor(workspace, approval_mode="auto")
    started = time.perf_counter()
    refresh = tools.python_sdk_refresh(
        limit=args.index_limit,
    )
    refresh_elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    rows: list[dict[str, Any]] = []
    for case in CASES:
        case_started = time.perf_counter()
        result = tools.python_sdk_search(
            case.query,
            limit=args.limit,
            use_embeddings=args.use_embeddings,
            embedding_model=args.embedding_model,
            embedding_host=args.embedding_host,
            embedding_timeout=args.embedding_timeout,
        )
        elapsed_ms = round((time.perf_counter() - case_started) * 1000, 3)
        output = str(result.get("output") or "")
        rows.append(
            {
                "name": case.name,
                "query": case.query,
                "expected": list(case.expected),
                "status": "pass" if result.get("ok") is True and _contains_expected(output, case.expected) else "fail",
                "elapsed_ms": elapsed_ms,
                "count": result.get("count", 0),
                "embedding_error": result.get("embedding_error"),
                "top_output": "\n".join(output.splitlines()[:8]),
            }
        )
    passed = sum(1 for row in rows if row["status"] == "pass")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "python": sys.version.split()[0],
        "mode": "hybrid_embeddings" if args.use_embeddings else "lexical_fts",
        "embedding_model": args.embedding_model if args.use_embeddings else None,
        "refresh": {
            "ok": refresh.get("ok") is True,
            "items": refresh.get("items", 0),
            "embedded": refresh.get("embedded", 0),
            "cached_embeddings": refresh.get("cached_embeddings", 0),
            "elapsed_ms": refresh_elapsed_ms,
            "embedding_error": refresh.get("embedding_error"),
        },
        "summary": {
            "cases": len(rows),
            "pass": passed,
            "fail": len(rows) - passed,
            "median_case_elapsed_ms": _median([float(row["elapsed_ms"]) for row in rows]),
        },
        "results": rows,
    }


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate installed-Python SDK/API retrieval quality and speed.")
    parser.add_argument("--workspace", default=".", help="Workspace used for the .ollama-code index cache.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--index-limit", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--use-embeddings", action="store_true", help="Use cached/on-demand Ollama embeddings for hybrid candidate reranking.")
    parser.add_argument("--embedding-model", default=None, help="Ollama embedding model, e.g. nomic-embed-text.")
    parser.add_argument("--embedding-host", default=None)
    parser.add_argument("--embedding-timeout", type=int, default=120)
    parser.add_argument("--strict-accuracy", action="store_true")
    args = parser.parse_args(argv)
    payload = run_eval(args)
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.strict_accuracy and payload["summary"]["fail"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
