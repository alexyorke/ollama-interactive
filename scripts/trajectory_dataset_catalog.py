from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = Path("scratch") / "external" / "datasets"
DEFAULT_OUTPUT = DEFAULT_DATA_ROOT / "trajectory-dataset-catalog.json"

HF_DATASET_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "repo_id": "SALT-NLP/SWE-chat",
        "slug": "swe-chat",
        "priority": "high",
        "kind": "real-user-coding-agent-sessions",
        "reason": "Real coding-agent interactions collected from open-source developers in the wild.",
        "adapter_hint": None,
        "path_globs": (),
    },
    {
        "repo_id": "trace-commons/agent-traces",
        "slug": "trace-commons-agent-traces",
        "priority": "high",
        "kind": "real-user-coding-agent-sessions",
        "reason": "Public donated coding-agent sessions from real open-source work, preserved with raw prompts, tool calls, and trace metadata.",
        "adapter_hint": "trace_commons",
        "path_globs": ("data/train-*.parquet",),
    },
    {
        "repo_id": "thoughtworks/agentic-coding-trajectories",
        "slug": "thoughtworks-agentic-coding-trajectories",
        "priority": "medium",
        "kind": "unified-agentic-coding-corpus",
        "reason": "Public 15k-session derivative corpus that unifies several coding-agent trajectory sources into one tokenized Parquet workload.",
        "adapter_hint": "thoughtworks",
        "path_globs": ("sessions.parquet",),
    },
    {
        "repo_id": "davanstrien/agent-race-traces",
        "slug": "agent-race-traces",
        "priority": "medium",
        "kind": "same-task-cross-harness-comparison",
        "reason": "Small public same-task corpus that holds the prompt constant across Claude Code, Pi, and ml-intern so harness-driven waste is easier to isolate.",
        "adapter_hint": "agent_race",
        "path_globs": ("*.jsonl",),
    },
    {
        "repo_id": "nebius/SWE-agent-trajectories",
        "slug": "nebius-swe-agent-trajectories",
        "priority": "high",
        "kind": "swe-agent-trajectories",
        "reason": "Large SWE-agent trajectory corpus already used for token-waste and loop analysis.",
        "adapter_hint": "swe_agent",
        "path_globs": ("data/*.parquet",),
    },
    {
        "repo_id": "nebius/SWE-rebench-openhands-trajectories",
        "slug": "nebius-swe-rebench-openhands-trajectories",
        "priority": "high",
        "kind": "openhands-trajectories",
        "reason": "Large OpenHands trajectory corpus with assistant/tool histories suited for controller and loop analysis.",
        "adapter_hint": "openhands",
        "path_globs": ("trajectories.parquet",),
    },
    {
        "repo_id": "SWE-bench/SWE-smith-trajectories",
        "slug": "swe-smith-trajectories",
        "priority": "high",
        "kind": "swe-smith-trajectories",
        "reason": "Large SWE-smith corpus that stresses edit and test-validation behavior.",
        "adapter_hint": "smith",
        "path_globs": ("data/train-*.parquet", "data/tool-*.parquet", "data/xml-*.parquet", "data/ticks-*.parquet"),
    },
    {
        "repo_id": "nvidia/SWE-Hero-openhands-trajectories",
        "slug": "nvidia-swe-hero-openhands-trajectories",
        "priority": "high",
        "kind": "openhands-trajectories",
        "reason": "Public 34k OpenHands trajectory set that appears adapter-compatible with the existing OpenHands parser.",
        "adapter_hint": "openhands",
        "path_globs": ("data/train-*.parquet",),
    },
    {
        "repo_id": "nvidia/SWE-Zero-openhands-trajectories",
        "slug": "nvidia-swe-zero-openhands-trajectories",
        "priority": "medium",
        "kind": "openhands-trajectories",
        "reason": "Public OpenHands-format synthetic SWE traces that add a large failure-heavy comparison set beyond SWE-Hero and Open-SWE.",
        "adapter_hint": "openhands",
        "path_globs": ("data/train-*.parquet",),
    },
    {
        "repo_id": "nvidia/Open-SWE-Traces",
        "slug": "open-swe-traces-openhands",
        "priority": "high",
        "kind": "openhands-trajectories",
        "reason": "Public OpenHands-format SWE traces with newer model families and raw trajectory lists under the trajectory field.",
        "adapter_hint": "openhands",
        "path_globs": ("data/*_openhands_trajectories/train-*.parquet",),
    },
    {
        "repo_id": "nvidia/Open-SWE-Traces",
        "slug": "open-swe-traces-sweagent",
        "priority": "high",
        "kind": "openhands-trajectories",
        "reason": "Public SWE-agent labeled traces that still serialize as tool-call chat trajectories and can be analyzed with the OpenHands message parser.",
        "adapter_hint": "openhands",
        "path_globs": ("data/*_sweagent_trajectories/train-*.parquet",),
    },
    {
        "repo_id": "togethercomputer/CoderForge-Preview-32B-SWE-Bench-Verified-Evaluation-trajectories",
        "slug": "coderforge-preview-swe-bench-verified-trajectories",
        "priority": "high",
        "kind": "openhands-trajectories",
        "reason": "Public CoderForge SWE-bench Verified evaluation traces with messages, patches, rewards, and test output for coding-agent failure analysis.",
        "adapter_hint": "openhands",
        "path_globs": ("trajectory/train-*.parquet",),
    },
    {
        "repo_id": "yoonholee/terminalbench-trajectories",
        "slug": "terminalbench-trajectories",
        "priority": "medium",
        "kind": "terminal-agent-trajectories",
        "reason": "Public terminal-agent trajectories that can reveal shell-loop and tool-budget patterns beyond SWE-style issue repair.",
        "adapter_hint": "terminalbench",
        "path_globs": ("data/train-*.parquet",),
    },
    {
        "repo_id": "zai-org/CC-Bench-trajectories",
        "slug": "cc-bench-trajectories",
        "priority": "medium",
        "kind": "interactive-coding-agent-trajectories",
        "reason": "Public interactive coding-agent traces with per-task token totals, tool-call counts, and mixed frontend, app-dev, data, and deployment workloads.",
        "adapter_hint": "cc_bench",
        "path_globs": ("train.parquet",),
    },
    {
        "repo_id": "Contextbench/Tracebench",
        "slug": "tracebench",
        "priority": "medium",
        "kind": "trajectory-artifacts",
        "reason": "Trace dataset with archived agent artifacts that could support future process-level waste analysis.",
        "adapter_hint": None,
        "path_globs": (),
    },
    {
        "repo_id": "NJU-LINK/CodeTraceBench",
        "slug": "codetracebench",
        "priority": "high",
        "kind": "trajectory-diagnosis-benchmark",
        "reason": "Human-verified incorrect-step benchmark for coding-agent trajectories, useful for premature-done, fake-validation, and timeout-loop analysis.",
        "adapter_hint": None,
        "path_globs": ("bench_manifest.parquet", "bench_manifest.verified.parquet"),
    },
    {
        "repo_id": "AlienKevin/SWE-ZERO-12M-trajectories",
        "slug": "swe-zero-12m-trajectories",
        "priority": "medium",
        "kind": "execution-free-agentic-coding-trajectories",
        "reason": "Large public execution-free coding-agent corpus that is useful for manual review of prompt, shell-command, and completion-claim efficiency, but not yet normalized into the local controller metrics pipeline.",
        "adapter_hint": None,
        "path_globs": ("data/train-*.parquet",),
    },
    {
        "repo_id": "peteromallet/my-personal-codex-data",
        "slug": "personal-codex-dataclaw",
        "priority": "medium",
        "kind": "real-user-coding-agent-sessions",
        "reason": "Public DataClaw export of real Codex CLI work with per-session token and tool-use totals.",
        "adapter_hint": None,
        "path_globs": ("codex/*.jsonl", ".dataclaw/manifest.json"),
    },
    {
        "repo_id": "misterkerns/my-personal-claude-code-data",
        "slug": "personal-claude-code-dataclaw",
        "priority": "medium",
        "kind": "real-user-coding-agent-sessions",
        "reason": "Public DataClaw export of real Claude Code work with per-session token and tool-use totals.",
        "adapter_hint": None,
        "path_globs": ("conversations.jsonl", "metadata.json"),
    },
    {
        "repo_id": "ultralazr/claude-code-traces",
        "slug": "ultralazr-claude-code-traces",
        "priority": "medium",
        "kind": "real-user-coding-agent-sessions",
        "reason": "Public redacted Claude Code session files exported in native JSONL trace format via cc-share-hf.",
        "adapter_hint": None,
        "path_globs": ("data/*.jsonl", "manifest.jsonl"),
    },
    {
        "repo_id": "Glint-Research/Fable-5-traces",
        "slug": "fable-5-traces",
        "priority": "medium",
        "kind": "agent-trace-corpus",
        "reason": "Public Pi-style converted coding-agent traces intended for inspection and reasoning or action distillation.",
        "adapter_hint": None,
        "path_globs": ("pi-traces/*.jsonl",),
    },
    {
        "repo_id": "badlogicgames/pi-mono",
        "slug": "pi-mono",
        "priority": "medium",
        "kind": "real-user-coding-agent-sessions",
        "reason": "Public redacted pi coding-agent sessions from real monorepo development, useful for first-turn orientation and shell-loop review.",
        "adapter_hint": None,
        "path_globs": ("*.jsonl", "manifest.jsonl"),
    },
    {
        "repo_id": "nmuendler/share-codex",
        "slug": "share-codex",
        "priority": "high",
        "kind": "real-user-coding-agent-sessions",
        "reason": "Public exported Codex and Claude Code sessions with prompts, tool calls, and tool outputs from local repo work.",
        "adapter_hint": None,
        "path_globs": ("train.jsonl", "export_manifest.json"),
    },
)

EXTERNAL_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "id": "agentlens/process-quality-paper",
        "slug": "agentlens-bench",
        "priority": "high",
        "kind": "process-level-trajectory-eval",
        "reason": "AgentLens describes process-level quality labels, waste signals, and divergence references for coding-agent trajectories, but the paper says the project site and release are still planned.",
        "source": "paper",
        "url": "https://arxiv.org/abs/2605.12925",
        "paper_url": "https://arxiv.org/abs/2605.12925",
    },
)


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"user-agent": "ollama-interactive/1.0"})


def _fetch_json(url: str, *, timeout: int = 30) -> Any:
    with urllib.request.urlopen(_request(url), timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _local_dir(data_root: Path, slug: str) -> str | None:
    path = data_root / slug
    if path.exists():
        return str(path.resolve())
    return None


def _hf_card_url(repo_id: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}"


def _sample_preview(repo_id: str) -> dict[str, Any]:
    splits_url = "https://datasets-server.huggingface.co/splits?" + urllib.parse.urlencode({"dataset": repo_id})
    splits_payload = _fetch_json(splits_url)
    raw_splits = splits_payload.get("splits") if isinstance(splits_payload, dict) else None
    if not isinstance(raw_splits, list) or not raw_splits:
        return {}
    first = raw_splits[0] if isinstance(raw_splits[0], dict) else {}
    config = str(first.get("config") or "default")
    split = str(first.get("split") or "train")
    preview: dict[str, Any] = {"config": config, "split": split}
    rows_url = "https://datasets-server.huggingface.co/first-rows?" + urllib.parse.urlencode(
        {"dataset": repo_id, "config": config, "split": split}
    )
    rows_payload = _fetch_json(rows_url)
    features = rows_payload.get("features") if isinstance(rows_payload, dict) else None
    if isinstance(features, list):
        preview["feature_names"] = [
            str(item.get("name"))
            for item in features
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        ]
    rows = rows_payload.get("rows") if isinstance(rows_payload, dict) else None
    if isinstance(rows, list) and rows:
        first_row = rows[0].get("row") if isinstance(rows[0], dict) else None
        if isinstance(first_row, dict):
            preview["row_keys"] = sorted(first_row.keys())
            preview["trajectory_like_fields"] = [
                key
                for key in sorted(first_row.keys())
                if any(token in key.lower() for token in ("trajectory", "messages", "steps", "tool", "patch"))
            ]
    return preview


def _build_hf_entry(spec: dict[str, Any], data_root: Path, *, include_preview: bool) -> dict[str, Any]:
    repo_id = str(spec["repo_id"])
    slug = str(spec["slug"])
    local_dir = _local_dir(data_root, slug)
    entry: dict[str, Any] = {
        "id": repo_id,
        "slug": slug,
        "source": "huggingface",
        "kind": str(spec["kind"]),
        "priority": str(spec["priority"]),
        "reason": str(spec["reason"]),
        "card_url": _hf_card_url(repo_id),
        "local_dir": local_dir,
        "adapter_hint": spec.get("adapter_hint"),
        "path_globs": list(spec.get("path_globs") or []),
        "analysis_ready": bool(local_dir and spec.get("adapter_hint") and spec.get("path_globs")),
    }
    try:
        payload = _fetch_json(f"https://huggingface.co/api/datasets/{repo_id}")
    except urllib.error.HTTPError as exc:
        status = "gated" if exc.code in {401, 403} else "unavailable"
        entry.update({"access_status": status, "error": f"HTTP {exc.code}: {exc.reason}"})
        return entry
    except Exception as exc:
        entry.update({"access_status": "unavailable", "error": str(exc)})
        return entry

    entry["access_status"] = "gated" if bool(payload.get("gated")) else "public"
    entry["downloads"] = int(payload.get("downloads", 0) or 0)
    entry["last_modified"] = str(payload.get("lastModified") or "")
    siblings = payload.get("siblings")
    if isinstance(siblings, list):
        entry["files_preview"] = [
            str(item.get("rfilename"))
            for item in siblings
            if isinstance(item, dict) and str(item.get("rfilename") or "").strip()
        ][:12]
    if include_preview and entry["access_status"] == "public":
        try:
            preview = _sample_preview(repo_id)
        except Exception as exc:
            preview = {"error": str(exc)}
        if preview:
            entry["remote_preview"] = preview
    return entry


def _build_external_entry(spec: dict[str, Any], data_root: Path) -> dict[str, Any]:
    local_dir = _local_dir(data_root, str(spec["slug"]))
    return {
        "id": str(spec["id"]),
        "slug": str(spec["slug"]),
        "source": str(spec["source"]),
        "kind": str(spec["kind"]),
        "priority": str(spec["priority"]),
        "reason": str(spec["reason"]),
        "url": str(spec["url"]),
        "paper_url": str(spec["paper_url"]),
        "local_dir": local_dir,
        "access_status": "metadata-only",
        "analysis_ready": False,
    }


def _summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    def _unique_ids(rows: list[dict[str, Any]]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in rows:
            value = str(item["id"])
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

    def _unique_high_priority_missing(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
        seen: set[str] = set()
        ordered: list[dict[str, str]] = []
        for item in rows:
            if str(item.get("priority") or "") != "high":
                continue
            value = str(item["id"])
            if value in seen:
                continue
            seen.add(value)
            ordered.append({"id": value, "reason": str(item.get("reason") or "")})
        return ordered

    local_entries = [item for item in entries if item.get("local_dir")]
    analysis_ready = [item for item in entries if item.get("analysis_ready")]
    public_missing = [
        item
        for item in entries
        if item.get("source") == "huggingface" and item.get("access_status") == "public" and not item.get("local_dir")
    ]
    gated = [item for item in entries if item.get("access_status") == "gated"]
    high_priority_public_missing = _unique_high_priority_missing(public_missing)
    return {
        "entries": len(entries),
        "local_entries": len(local_entries),
        "analysis_ready_local_entries": len(analysis_ready),
        "public_missing_entries": len(public_missing),
        "gated_entries": len(gated),
        "local_ids": _unique_ids(local_entries),
        "analysis_ready_local_ids": _unique_ids(analysis_ready),
        "public_missing_ids": _unique_ids(public_missing),
        "gated_ids": _unique_ids(gated),
        "high_priority_public_missing": high_priority_public_missing[:4],
    }


def build_catalog(data_root: Path, *, include_preview: bool = True) -> dict[str, Any]:
    entries = [_build_hf_entry(spec, data_root, include_preview=include_preview) for spec in HF_DATASET_CANDIDATES]
    entries.extend(_build_external_entry(spec, data_root) for spec in EXTERNAL_CANDIDATES)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root.resolve(strict=False)),
        "entries": entries,
        "summary": _summary(entries),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Catalog useful coding-agent trajectory datasets and probe local/remote availability.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root directory for local trajectory datasets.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON output path.")
    parser.add_argument("--skip-preview", action="store_true", help="Skip remote split/row preview calls.")
    args = parser.parse_args(argv)

    payload = build_catalog(args.data_root, include_preview=not args.skip_preview)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    print(
        "[trajectory-dataset-catalog] "
        + f"entries={summary.get('entries', 0)} "
        + f"local={summary.get('local_entries', 0)} "
        + f"analysis_ready_local={summary.get('analysis_ready_local_entries', 0)} "
        + f"public_missing={summary.get('public_missing_entries', 0)} "
        + f"gated={summary.get('gated_entries', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
