from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
import sys
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ollama_code.tools import ToolExecutor


def _write_fixture(root: Path, *, generated_files: int) -> None:
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir(parents=True)
    (root / "src" / "pricing.py").write_text(
        "def calculate_discount(cart):\n"
        "    subtotal = sum(cart)\n"
        "    return subtotal * 0.9\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_pricing.py").write_text(
        "from src.pricing import calculate_discount\n",
        encoding="utf-8",
    )
    generated_roots = [
        root / "scratch",
        root / "verify_scratch",
        root / "ollama-code-bench-fixture",
        root / "probe-fixture",
        root / "tmpabc123",
    ]
    for generated_root in generated_roots:
        generated_root.mkdir(parents=True, exist_ok=True)
    for index in range(generated_files):
        generated_root = generated_roots[index % len(generated_roots)]
        (generated_root / f"ignored_{index}.py").write_text(
            f"def ignored_target_{index}():\n    return {index}\n",
            encoding="utf-8",
        )


def _measure(label: str, action: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    result = action()
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    row = {
        "name": label,
        "elapsed_ms": elapsed_ms,
        "ok": result.get("ok") is True,
        "count": result.get("count") or result.get("files") or len(str(result.get("output", "")).splitlines()),
    }
    phase_timings = {
        key: float(result.get(key, 0.0) or 0.0)
        for key in ("scan_ms", "ruff_ms", "typecheck_ms", "shell_ms")
        if key in result
    }
    if phase_timings:
        row["phase_timings_ms"] = phase_timings
    if isinstance(result.get("validator_targets"), list) and result["validator_targets"]:
        row["validator_targets"] = list(result["validator_targets"])
    return row


def run_probe(generated_files: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ollama-code-speed-") as tmp:
        root = Path(tmp)
        _write_fixture(root, generated_files=generated_files)
        tools = ToolExecutor(root, approval_mode="auto")
        rows = [
            _measure("file_search", lambda: tools.file_search("pricing")),
            _measure("repo_index_search", lambda: tools.repo_index_search("calculate_discount")),
            _measure("fts_refresh", lambda: tools.fts_refresh()),
            _measure("context_pack", lambda: tools.context_pack("fix discount calculation")),
            _measure("discover_validators", lambda: tools.discover_validators()),
            _measure("tree_sitter_syntax", lambda: tools.tree_sitter_syntax("src")),
            _measure("ast_search", lambda: tools.ast_search("def $F($$$A): $$$B", "src/pricing.py", lang="python")),
            _measure("lint_typecheck", lambda: tools.lint_typecheck("src", timeout=120)),
        ]
    return {"generated_files": generated_files, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="No-LLM tool speed probe for broad repo operations.")
    parser.add_argument("--generated-files", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    payload = run_probe(max(0, args.generated_files))
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
