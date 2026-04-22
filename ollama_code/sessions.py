from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


SESSION_SUBDIR = Path(".ollama-code") / "sessions"


@dataclass
class SessionSummary:
    path: Path
    updated_at: datetime
    model: str
    approval_mode: str
    message_count: int
    summary: str


def default_session_dir(workspace_root: Path) -> Path:
    return workspace_root / SESSION_SUBDIR


def resolve_transcript_path(workspace_root: Path, raw_path: str | Path) -> Path:
    root = workspace_root.resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Transcript path escapes the workspace: {raw_path}")
    return resolved


def new_session_path(workspace_root: Path) -> Path:
    session_dir = default_session_dir(workspace_root)
    session_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return session_dir / f"{timestamp}-{uuid4().hex[:6]}.json"


def load_transcript_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid transcript JSON in {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Transcript root must be a JSON object: {path}")
    return payload


def _safe_session_paths(workspace_root: Path) -> list[Path]:
    session_dir = default_session_dir(workspace_root)
    if not session_dir.exists():
        return []
    candidates: list[Path] = []
    for path in session_dir.glob("*.json"):
        if not path.is_file():
            continue
        try:
            resolved = resolve_transcript_path(workspace_root, path)
        except ValueError:
            continue
        candidates.append(resolved)
    return sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)


def latest_session_path(workspace_root: Path) -> Path | None:
    candidates = _safe_session_paths(workspace_root)
    return candidates[0] if candidates else None


def _payload_summary(payload: dict[str, Any]) -> str:
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                content = str(message.get("content", "")).strip()
                if content:
                    single_line = " ".join(content.split())
                    return single_line[:77] + "..." if len(single_line) > 80 else single_line
    events = payload.get("events")
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict) and event.get("type") == "user":
                content = str(event.get("content", "")).strip()
                if content:
                    single_line = " ".join(content.split())
                    return single_line[:77] + "..." if len(single_line) > 80 else single_line
    return "(no summary)"


def list_sessions(workspace_root: Path, limit: int = 20) -> list[SessionSummary]:
    summaries: list[SessionSummary] = []
    for path in _safe_session_paths(workspace_root)[:limit]:
        try:
            payload = load_transcript_payload(path)
        except ValueError:
            continue
        messages = payload.get("messages")
        message_count = len(messages) if isinstance(messages, list) else 0
        summaries.append(
            SessionSummary(
                path=path.resolve(),
                updated_at=datetime.fromtimestamp(path.stat().st_mtime),
                model=str(payload.get("model", "")),
                approval_mode=str(payload.get("approval_mode", "")),
                message_count=message_count,
                summary=_payload_summary(payload),
            )
        )
    return summaries
