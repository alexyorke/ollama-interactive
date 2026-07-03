from __future__ import annotations

import re
from typing import Any


ToolOperation = tuple[str, dict[str, Any]]


def workflow_config_update_spec(request_text: str) -> dict[str, str] | None:
    lowered = request_text.lower()
    if "pull_request" not in lowered or "workflow" not in lowered:
        return None
    if not re.search(r"\b(?:change|update|set)\b", lowered):
        return None
    path_match = re.search(r"\b(?P<path>(?:\.github|github)/workflows/[\w.-]+\.ya?ml)\b", request_text, flags=re.IGNORECASE)
    command_match = re.search(
        r"\b(?:command|run(?:test)?|unittest)\b(?:(?!\n\n).){0,160}?\bto\s+`(?P<command>[^`]+)`",
        request_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not path_match or not command_match:
        return None
    path = path_match.group("path").strip().replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    if path.lower().startswith("github/workflows/"):
        path = "." + path
    if not path.lower().startswith(".github/workflows/"):
        return None
    new_command = command_match.group("command").strip()
    if not path or not new_command:
        return None
    return {"path": path, "new_command": new_command}


def workflow_config_update_operations_from_source(*, path: str, new_command: str, source: str) -> list[ToolOperation] | None:
    operations: list[ToolOperation] = [("read_file", {"path": path})]
    lines = source.splitlines()
    if "pull_request:" not in source:
        on_index = next((index for index, line in enumerate(lines) if line.strip() == "on:"), None)
        if on_index is None:
            return None
        insert_index = len(lines)
        for index in range(on_index + 1, len(lines)):
            if lines[index] and not lines[index].startswith((" ", "\t")):
                insert_index = index
                break
        old_block = "\n".join(lines[on_index:insert_index])
        new_block = old_block.rstrip("\n") + "\n  pull_request:"
        operations.append(("replace_in_file", {"path": path, "old": old_block, "new": new_block}))

    command_line = next(
        (
            line
            for line in lines
            if "python -m unittest" in line and new_command not in line and re.search(r"\b(?:run|command)\s*:", line)
        ),
        None,
    )
    if command_line is None:
        return None if len(operations) == 1 else operations
    indent = command_line[: len(command_line) - len(command_line.lstrip())]
    marker_match = re.match(r"(?P<prefix>\s*-\s*run:\s*|\s*run:\s*)", command_line)
    if not marker_match:
        return None
    new_line = f"{indent}{marker_match.group('prefix').strip()} {new_command}"
    if marker_match.group("prefix").lstrip().startswith("-"):
        new_line = f"{indent}- run: {new_command}"
    else:
        new_line = f"{indent}run: {new_command}"
    operations.append(("replace_in_file", {"path": path, "old": command_line, "new": new_line}))
    return operations if len(operations) > 1 else None


def workflow_config_update_operations(*, request_text: str, source: str) -> list[ToolOperation] | None:
    spec = workflow_config_update_spec(request_text)
    if not spec:
        return None
    return workflow_config_update_operations_from_source(path=spec["path"], new_command=spec["new_command"], source=source)


def clean_return_expression(expression: str) -> str:
    expression = re.sub(r"\s+", " ", expression).strip()
    expression = expression.strip("`\"' ")
    expression = re.sub(r"\s+(?:then\s+)?(?:run|rerun|and|do not|don't)\b.*$", "", expression, flags=re.IGNORECASE).strip()
    return expression.rstrip(".,;:`\"' ").strip()


def symbol_return_update_spec(request_text: str) -> dict[str, str] | None:
    patterns = [
        r"\b(?P<path>[\w./-]+\.(?:js|jsx|ts|tsx|py))\b(?:(?!\n\n).){0,240}?\b(?:change|changing|update|updating)\s+(?P<symbol>[A-Za-z_]\w*)\s*\([^)]*\)\s+so\s+it\s+returns\s+(?P<new>.+?)\s+instead\s+of\s+(?P<old>.+?)(?:[.?!]|$)",
        r"\b(?P<path>[\w./-]+\.(?:js|jsx|ts|tsx|py))\b(?:(?!\n\n).){0,240}?\b(?:so|make)\s+(?P<symbol>[A-Za-z_]\w*)\s*\([^)]*\)\s+returns\s+(?P<new>.+?)\s+instead\s+of\s+(?P<old>.+?)(?:[.?!]|$)",
    ]
    match = next((item for pattern in patterns if (item := re.search(pattern, request_text, flags=re.IGNORECASE | re.DOTALL))), None)
    if not match:
        return None
    path = str(match.group("path") or "").strip().rstrip(".,;:")
    symbol = match.group("symbol").strip()
    new_expr = clean_return_expression(match.group("new"))
    old_expr = clean_return_expression(match.group("old"))
    if not path or not symbol or not new_expr or not old_expr:
        return None
    return {"path": path, "symbol": symbol, "new_expr": new_expr, "old_expr": old_expr}


def symbol_return_update_operations_from_source(
    *,
    path: str,
    symbol: str,
    new_expr: str,
    old_expr: str,
    source: str,
    requested_tool_names: set[str],
    required_tool_names: set[str],
) -> list[ToolOperation] | None:
    if not path or not symbol or not new_expr or not old_expr:
        return None
    old_line: str | None = None
    new_line: str | None = None
    old_candidates = {f"return {old_expr}", f"return {old_expr};"}
    for line in source.splitlines():
        stripped = line.strip()
        if stripped not in old_candidates:
            continue
        indent = line[: len(line) - len(line.lstrip())]
        semicolon = ";" if stripped.endswith(";") else ""
        old_line = line
        new_line = f"{indent}return {new_expr}{semicolon}"
        break
    if old_line is None or new_line is None or old_line == new_line:
        return None
    operations: list[ToolOperation] = []
    if "search_symbols" in requested_tool_names or "search_symbols" in required_tool_names:
        operations.append(("search_symbols", {"query": symbol, "path": path}))
    if "read_symbol" in requested_tool_names or "read_symbol" in required_tool_names:
        operations.append(("read_symbol", {"path": path, "symbol": symbol, "include_context": 0}))
    operations.append(("replace_in_file", {"path": path, "old": old_line, "new": new_line}))
    return operations


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
