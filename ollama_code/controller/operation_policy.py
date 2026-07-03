from __future__ import annotations

import re
from typing import Any


ToolOperation = tuple[str, dict[str, Any]]


def project_function_rename_operations(request_text: str) -> list[ToolOperation] | None:
    lowered = request_text.lower()
    if not re.search(r"\b(?:rename|renam|refactor|change|update)\b", lowered):
        return None
    match = re.search(
        r"\bfrom\s+(?P<old>[A-Za-z_]\w*)\s*\([^)]*\)\s+to\s+(?P<new>[A-Za-z_]\w*)\s*\(",
        request_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    old = match.group("old")
    new = match.group("new")
    if old == new:
        return None
    if not re.search(r"\b(?:api|function|symbol|call|calls|callers|docs?|tests?|project|repo|source)\b", lowered):
        return None
    return [
        (
            "edit_intent",
            {
                "path": ".",
                "intent": "rename",
                "target": old,
                "replacement": new,
                "scope": "project",
            },
        )
    ]


def project_function_rename_already_satisfied(
    *,
    request_text: str,
    successful_tool_results: list[dict[str, Any]],
) -> bool:
    rename_ops = project_function_rename_operations(request_text)
    if not rename_ops or len(rename_ops) != 1:
        return False
    tool_name, args = rename_ops[0]
    if tool_name != "edit_intent" or not isinstance(args, dict):
        return False
    target = str(args.get("target") or "").strip()
    replacement = str(args.get("replacement") or "").strip()
    scope = str(args.get("scope") or "").strip().lower()
    path = str(args.get("path") or "").strip()
    if not target or not replacement:
        return False
    for item in reversed(successful_tool_results):
        if str(item.get("name") or "").strip() != "edit_intent":
            continue
        item_arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        if str(item_arguments.get("target") or "").strip() != target:
            continue
        if str(item_arguments.get("replacement") or "").strip() != replacement:
            continue
        if str(item_arguments.get("scope") or "").strip().lower() != scope:
            continue
        if str(item_arguments.get("path") or "").strip() != path:
            continue
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        if result.get("ok") is True:
            return True
    return False
