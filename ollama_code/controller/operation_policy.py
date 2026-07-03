from __future__ import annotations

import ast
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


def optional_parameter_update_spec(request_text: str) -> dict[str, str] | None:
    match = re.search(
        r"\badd\s+an?\s+optional\s+(?P<param>[A-Za-z_]\w*)\s*:\s*(?P<annotation>[^=]+?)\s*=\s*(?P<default>False|True|None|[-+]?\d+(?:\.\d+)?|['\"][^'\"]*['\"])\s+parameter\s+to\s+(?P<symbol>[A-Za-z_]\w*)\s+in\s+(?P<src>[\w./-]+\.py)\b(?:(?!\n\n).){0,240}?\bupdate\s+(?P<doc>[\w./-]+\.md)\b",
        request_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    return {
        "src_path": match.group("src"),
        "doc_path": match.group("doc"),
        "symbol": match.group("symbol"),
        "param": match.group("param"),
        "annotation": re.sub(r"\s+", " ", match.group("annotation")).strip(),
        "default": match.group("default").strip(),
    }


def signature_with_appended_parameter(source: str, node: ast.FunctionDef | ast.AsyncFunctionDef, parameter: str) -> str:
    lines = source.splitlines()
    start = int(getattr(node, "lineno", 1)) - 1
    if start < 0 or start >= len(lines):
        return ""
    signature_line = lines[start].strip()
    if "\n" in signature_line or not signature_line.startswith(("def ", "async def ")):
        return ""
    open_index = signature_line.find("(")
    if open_index < 0:
        return ""
    depth = 0
    close_index = -1
    for index, char in enumerate(signature_line[open_index:], start=open_index):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                close_index = index
                break
    if close_index < 0:
        return ""
    current_params = signature_line[open_index + 1 : close_index].strip()
    if parameter.split(":", 1)[0].strip() in {part.split(":", 1)[0].split("=", 1)[0].strip() for part in current_params.split(",")}:
        return signature_line
    separator = ", " if current_params else ""
    return f"{signature_line[: open_index + 1]}{current_params}{separator}{parameter}{signature_line[close_index:]}"


def optional_parameter_update_operations_from_source(
    *,
    src_path: str,
    doc_path: str,
    symbol: str,
    param: str,
    annotation: str,
    default: str,
    source: str,
    docs: str,
) -> list[ToolOperation] | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    node = next(
        (
            child
            for child in tree.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == symbol
        ),
        None,
    )
    if node is None:
        return None
    existing_params = {arg.arg for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]}
    operations: list[ToolOperation] = []
    if param not in existing_params:
        signature = signature_with_appended_parameter(source, node, f"{param}: {annotation} = {default}")
        if not signature:
            return None
        operations.append(
            (
                "edit_intent",
                {
                    "path": src_path,
                    "intent": "change_signature",
                    "target": symbol,
                    "replacement": signature,
                },
            )
        )

    if param not in docs:
        call_match = re.search(rf"`{re.escape(symbol)}\((?P<args>[^`]*)\)`", docs)
        if call_match and param not in call_match.group("args"):
            old_call = call_match.group(0)
            args = call_match.group("args").strip()
            separator = ", " if args else ""
            new_call = f"`{symbol}({args}{separator}{param}={default})`"
            operations.append(("replace_in_file", {"path": doc_path, "old": old_call, "new": new_call}))
    if not operations:
        return None
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
