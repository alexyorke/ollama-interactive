from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _host(raw: str) -> str:
    if not raw.startswith(("http://", "https://")):
        raw = f"http://{raw}"
    return raw.rstrip("/")


def _chat(host: str, model: str, prompt: str, *, options: dict[str, int] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
    }
    if options:
        payload["options"] = options
    request = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        raw = json.loads(response.read().decode("utf-8"))
    wall_s = time.perf_counter() - started
    prompt_tokens = int(raw.get("prompt_eval_count") or 0)
    output_tokens = int(raw.get("eval_count") or 0)
    prompt_eval_ns = int(raw.get("prompt_eval_duration") or 0)
    eval_ns = int(raw.get("eval_duration") or 0)
    return {
        "options": options or {},
        "wall_s": round(wall_s, 3),
        "load_s": round(int(raw.get("load_duration") or 0) / 1e9, 3),
        "prompt_eval_s": round(prompt_eval_ns / 1e9, 3),
        "eval_s": round(eval_ns / 1e9, 3),
        "api_total_s": round(int(raw.get("total_duration") or 0) / 1e9, 3),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "prompt_tok_s": round(prompt_tokens / (prompt_eval_ns / 1e9), 2) if prompt_eval_ns else None,
        "output_tok_s": round(output_tokens / (eval_ns / 1e9), 2) if eval_ns else None,
        "done_reason": raw.get("done_reason"),
        "preview": str((raw.get("message") or {}).get("content", ""))[:120].replace("\n", " "),
    }


def _stop_model(model: str) -> None:
    subprocess.run(["ollama", "stop", model], capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)


def _ollama_ps() -> str:
    result = subprocess.run(["ollama", "ps"], capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    return (result.stdout or result.stderr or "").strip()


def _long_prompt(lines: int) -> str:
    body = "\n".join(f"line {index}: filler text for prefill measurement." for index in range(1, lines + 1))
    return body + "\n\nTask: reply with exactly one short sentence."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile Ollama local inference with raw API duration counters.")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--models", nargs="+", default=["gemma4:e4b", "granite4.1:8b", "gemma3:4b", "qwen3:8b"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--long-lines", type=int, default=800)
    parser.add_argument("--keep-loaded", action="store_true")
    args = parser.parse_args(argv)

    host = _host(args.host)
    prompts = {
        "short-default": ("Reply with exactly 80 lowercase words about software testing. No bullets.", {"num_predict": 96}),
        "short-ctx4096": (
            "Reply with exactly 80 lowercase words about software testing. No bullets.",
            {"num_ctx": 4096, "num_predict": 96},
        ),
        "long-ctx4096": (_long_prompt(args.long_lines), {"num_ctx": 4096, "num_predict": 48}),
    }
    results: list[dict[str, Any]] = []
    for model in args.models:
        if not args.keep_loaded:
            _stop_model(model)
        for case, (prompt, options) in prompts.items():
            outcome = _chat(host, model, prompt, options=options)
            row = {
                "model": model,
                "case": case,
                **outcome,
                "ollama_ps_after": _ollama_ps(),
            }
            results.append(row)
            print(
                f"[perf] model={model} case={case} wall={row['wall_s']}s "
                f"prompt={row['prompt_tok_s']} tok/s output={row['output_tok_s']} tok/s"
            )
        if not args.keep_loaded:
            _stop_model(model)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": host,
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
