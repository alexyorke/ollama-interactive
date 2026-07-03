from __future__ import annotations

from collections.abc import Callable
from typing import Any


ToolNamePredicate = Callable[[str], bool]


def normalize_payload(payload: dict[str, Any], *, is_supported_tool_name: ToolNamePredicate) -> dict[str, Any]:
    """Normalize model JSON payloads into the controller's tool/final shapes."""
    normalized = dict(payload)
    response_type = normalized.get("type")
    tool_name = normalized.get("name")
    if isinstance(response_type, str):
        normalized["type"] = response_type.strip()
    if isinstance(tool_name, str):
        normalized["name"] = tool_name.strip()
    response_type = normalized.get("type")
    tool_name = normalized.get("name")
    arguments = normalized.get("arguments")
    if isinstance(tool_name, str) and tool_name == "final" and response_type in {"tool", None, "", "function", "tool_call"}:
        message = normalized.get("message")
        if not isinstance(message, str) and isinstance(arguments, dict):
            arg_message = arguments.get("message")
            if isinstance(arg_message, str):
                message = arg_message
            else:
                arg_content = arguments.get("content")
                if isinstance(arg_content, str):
                    message = arg_content
        return {"type": "final", "message": str(message or "").strip()}
    if isinstance(response_type, str) and is_supported_tool_name(response_type):
        normalized["type"] = "tool"
        if not isinstance(tool_name, str) or not tool_name:
            normalized["name"] = response_type
        if not isinstance(arguments, dict):
            normalized["arguments"] = {}
        return normalized
    if isinstance(tool_name, str) and is_supported_tool_name(tool_name) and response_type in {None, "", "function", "tool_call"}:
        normalized["type"] = "tool"
        if not isinstance(arguments, dict):
            normalized["arguments"] = {}
        return normalized
    return normalized
