from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ollama_code.agent_parsing import _workspace_roots_match


SESSION_SUBDIR = Path(".ollama-code") / "sessions"
WINDOWS_DRIVE_PATH = re.compile(r"^(?P<drive>[A-Za-z]):(?:[\\/](?P<rest>.*))?$")
WSL_MOUNT_PATH = re.compile(r"^/mnt/(?P<drive>[A-Za-z])(?:/(?P<rest>.*))?$")


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


def _coerce_cross_platform_absolute_path(raw_path: str | Path) -> Path | None:
    text = str(raw_path).strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate
    normalized = text.replace("\\", "/")
    windows_match = WINDOWS_DRIVE_PATH.match(normalized)
    if windows_match:
        drive = windows_match.group("drive").lower()
        rest = (windows_match.group("rest") or "").strip("/")
        suffix = f"/{rest}" if rest else ""
        return Path(f"/mnt/{drive}{suffix}")
    wsl_match = WSL_MOUNT_PATH.match(normalized)
    if wsl_match:
        drive = wsl_match.group("drive").upper()
        rest = (wsl_match.group("rest") or "").strip("/")
        return Path(f"{drive}:/{rest}") if rest else Path(f"{drive}:/")
    return None


def resolve_transcript_path(workspace_root: Path, raw_path: str | Path) -> Path:
    root = workspace_root.resolve()
    candidate = _coerce_cross_platform_absolute_path(raw_path)
    if candidate is None:
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
        raw = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise ValueError(f"Transcript file not found: {path}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid transcript encoding in {path}; expected UTF-8") from exc
    except OSError as exc:
        raise ValueError(f"Unable to read transcript file: {path} ({exc})") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid transcript JSON in {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Transcript root must be a JSON object: {path}")
    return payload


def payload_can_restore_session(payload: dict[str, Any], workspace_root: Path) -> bool:
    if not _workspace_roots_match(payload.get("workspace_root"), workspace_root):
        return False
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    has_conversation_history = False
    for message in messages:
        if not isinstance(message, dict):
            return False
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            return False
        if role != "system":
            has_conversation_history = True
    return has_conversation_history


def write_transcript_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(raw_tmp_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _safe_session_paths(workspace_root: Path) -> list[Path]:
    session_dir = default_session_dir(workspace_root)
    if not session_dir.exists():
        return []
    candidates: list[tuple[Path, float]] = []
    for path in session_dir.glob("*.json"):
        try:
            if not path.is_file():
                continue
            resolved = resolve_transcript_path(workspace_root, path)
            modified_at = resolved.stat().st_mtime
        except (OSError, ValueError):
            continue
        candidates.append((resolved, modified_at))
    return [path for path, _ in sorted(candidates, key=lambda item: item[1], reverse=True)]


def latest_session_path(workspace_root: Path) -> Path | None:
    latest = latest_restorable_session(workspace_root)
    if latest is None:
        return None
    return latest[0]


def latest_restorable_session(workspace_root: Path) -> tuple[Path, dict[str, Any]] | None:
    for candidate in _safe_session_paths(workspace_root):
        try:
            payload = load_transcript_payload(candidate)
        except ValueError:
            continue
        if payload_can_restore_session(payload, workspace_root):
            return candidate, payload
    return None


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
    for path in _safe_session_paths(workspace_root):
        try:
            payload = load_transcript_payload(path)
        except ValueError:
            continue
        if not payload_can_restore_session(payload, workspace_root):
            continue
        messages = payload.get("messages")
        message_count = len(messages) if isinstance(messages, list) else 0
        try:
            updated_at = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            continue
        summaries.append(
            SessionSummary(
                path=path.resolve(),
                updated_at=updated_at,
                model=str(payload.get("model", "")),
                approval_mode=str(payload.get("approval_mode", "")),
                message_count=message_count,
                summary=_payload_summary(payload),
            )
        )
        if len(summaries) >= limit:
            break
    return summaries
