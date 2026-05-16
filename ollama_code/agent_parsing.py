from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ollama_code.agent_protocol import KNOWN_TOOL_NAMES


def extract_json_response(raw_text: str, *, _depth: int = 0) -> dict[str, Any] | None:
    candidate = raw_text.strip()
    if not candidate:
        return None
    candidate = re.sub(r"<think>.*?</think>", "", candidate, flags=re.DOTALL)
    candidate = candidate.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
        candidate = candidate.strip()
    repaired_candidate = _repair_truncated_json(candidate)
    if repaired_candidate is not None:
        candidate = repaired_candidate
    decoder = json.JSONDecoder()
    if _depth < 2:
        try:
            top_level, end_index = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            top_level = None
            end_index = -1
        if isinstance(top_level, str) and not candidate[end_index:].strip():
            return extract_json_response(top_level, _depth=_depth + 1)
    starts = [index for index, char in enumerate(candidate) if char == "{"]
    if candidate.startswith("{"):
        starts = [0] + [index for index in starts if index != 0]
    parsed_dicts: list[dict[str, Any]] = []
    agent_payloads: list[dict[str, Any]] = []
    top_level_dict: dict[str, Any] | None = None
    for start in starts:
        try:
            data, end_index = decoder.raw_decode(candidate[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            if start == 0 and not candidate[end_index:].strip():
                top_level_dict = data
            parsed_dicts.append(data)
            response_type = data.get("type")
            tool_name = data.get("name")
            if isinstance(response_type, str) and response_type.strip() in {"tool", "final", *KNOWN_TOOL_NAMES}:
                agent_payloads.append(data)
                continue
            if isinstance(tool_name, str) and tool_name.strip() in KNOWN_TOOL_NAMES and response_type in {None, "", "function", "tool_call"}:
                agent_payloads.append(data)
    if agent_payloads:
        return agent_payloads[-1]
    if top_level_dict is not None:
        return top_level_dict
    if parsed_dicts:
        return parsed_dicts[-1]
    return None


def _decode_json_like_string(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        text = value.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
        return text.replace("\\\\", "\\")


def _find_json_like_array_end(raw_text: str, start_index: int) -> int | None:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start_index, len(raw_text)):
        char = raw_text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "[":
            depth += 1
            continue
        if char == "]":
            depth -= 1
            if depth == 0:
                return index
    return None


def extract_json_like_fields(raw_text: str, *, scalar_keys: tuple[str, ...], array_keys: tuple[str, ...]) -> dict[str, Any]:
    candidate = re.sub(r"<think>.*?</think>", "", raw_text or "", flags=re.DOTALL).strip()
    if not candidate:
        return {}
    extracted: dict[str, Any] = {}
    for key in scalar_keys:
        match = re.search(rf'"?{re.escape(key)}"?\s*:\s*"((?:[^"\\]|\\.)*)"', candidate, flags=re.DOTALL)
        if match:
            extracted[key] = _decode_json_like_string(match.group(1))
    for key in array_keys:
        match = re.search(rf'"?{re.escape(key)}"?\s*:\s*\[', candidate, flags=re.DOTALL)
        if not match:
            continue
        start_index = match.end() - 1
        end_index = _find_json_like_array_end(candidate, start_index)
        if end_index is None:
            continue
        items = [
            _decode_json_like_string(group)
            for group in re.findall(r'"((?:[^"\\]|\\.)*)"', candidate[start_index + 1 : end_index], flags=re.DOTALL)
        ]
        extracted[key] = items
    return extracted


def _repair_truncated_json(candidate: str) -> str | None:
    if not candidate or candidate[0] not in "{[":
        return None
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in candidate:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            stack.append("}")
            continue
        if char == "[":
            stack.append("]")
            continue
        if char in "}]":
            if not stack or char != stack[-1]:
                return None
            stack.pop()
    if in_string or not stack:
        return None
    repaired = candidate + "".join(reversed(stack))
    try:
        _, end_index = json.JSONDecoder().raw_decode(repaired)
    except json.JSONDecodeError:
        return None
    if repaired[end_index:].strip():
        return None
    return repaired


def _portable_workspace_key(raw_path: str | Path) -> str | None:
    text = str(raw_path).strip()
    if not text:
        return None
    normalized = text.replace("\\", "/")
    drive_match = re.match(r"^(?P<drive>[A-Za-z]):(?:/(?P<rest>.*))?$", normalized)
    if drive_match:
        drive = drive_match.group("drive").lower()
        rest = (drive_match.group("rest") or "").strip("/")
        return f"{drive}:/{rest}" if rest else f"{drive}:/"
    wsl_match = re.match(r"^/mnt/(?P<drive>[A-Za-z])(?:/(?P<rest>.*))?$", normalized)
    if wsl_match:
        drive = wsl_match.group("drive").lower()
        rest = (wsl_match.group("rest") or "").strip("/")
        return f"{drive}:/{rest}" if rest else f"{drive}:/"
    try:
        return Path(text).resolve(strict=False).as_posix()
    except OSError:
        return None


def _workspace_roots_match(saved_root: object, current_root: Path) -> bool:
    if saved_root is None:
        return True
    saved_text = str(saved_root).strip()
    if not saved_text:
        return False
    current_root = current_root.resolve()
    try:
        saved_path = Path(saved_text).resolve(strict=False)
    except OSError:
        saved_path = None
    if saved_path == current_root:
        return True
    saved_key = _portable_workspace_key(saved_text)
    current_key = _portable_workspace_key(current_root)
    if saved_key is None or current_key is None:
        return False
    if re.match(r"^[A-Za-z]:/", saved_key) or re.match(r"^[A-Za-z]:/", current_key):
        return saved_key.casefold() == current_key.casefold()
    return saved_key == current_key
