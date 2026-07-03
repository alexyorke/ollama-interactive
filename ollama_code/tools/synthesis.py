from __future__ import annotations

import ast
import re
from typing import Any


def human_test_name(name: str) -> str:
    text = re.sub(r"^test_?", "", name.strip())
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or name


def node_expr(node: ast.AST, local_exprs: dict[str, str] | None = None) -> str:
    if local_exprs and isinstance(node, ast.Name) and node.id in local_exprs:
        return local_exprs[node.id]
    if local_exprs and isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self" and node.attr in local_exprs:
        return local_exprs[node.attr]
    if local_exprs and isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in local_exprs:
        return f"{local_exprs[node.value.id]}.{node.attr}"
    try:
        value = ast.literal_eval(node)
        return repr(value)
    except Exception:
        try:
            return ast.unparse(node)
        except Exception:
            return "?"


def test_spec_call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    try:
        return ast.unparse(func)
    except Exception:
        return "call"


def method_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute):
            return func.attr
        if isinstance(func, ast.Name):
            return func.id
    return ""


def call_expr(call: ast.Call, local_exprs: dict[str, str] | None = None) -> str:
    args = [node_expr(arg, local_exprs) for arg in call.args]
    args.extend(f"{kw.arg}={node_expr(kw.value, local_exprs)}" for kw in call.keywords if kw.arg)
    name = test_spec_call_name(call)
    if isinstance(call.func, ast.Attribute):
        receiver = node_expr(call.func.value, local_exprs)
        name = f"{receiver}.{call.func.attr}"
    return f"{name}({', '.join(args)})"


def test_spec_symbol_from_expr(expr: str, source_symbols: set[str], aliases: dict[str, str]) -> str:
    cleaned = expr.strip()
    parts = [part.strip() for part in cleaned.split(";") if part.strip()]
    if len(parts) > 1:
        last = parts[-1]
        receiver_attr = re.match(r"^(?P<receiver>[A-Za-z_][A-Za-z0-9_]*)\.(?P<attr>[A-Za-z_][A-Za-z0-9_]*)$", last)
        if receiver_attr:
            receiver = receiver_attr.group("receiver")
            for earlier in parts[:-1]:
                assignment = re.match(rf"^{re.escape(receiver)}\s*=\s*(?P<class>[A-Za-z_][A-Za-z0-9_]*)\(", earlier)
                if assignment and aliases.get(assignment.group("class"), assignment.group("class")) in source_symbols:
                    return aliases.get(assignment.group("class"), assignment.group("class"))
        len_call = re.match(r"^len\((?P<receiver>[A-Za-z_][A-Za-z0-9_]*)\)$", last)
        if len_call:
            receiver = len_call.group("receiver")
            for earlier in parts[:-1]:
                assignment = re.match(rf"^{re.escape(receiver)}\s*=\s*(?P<class>[A-Za-z_][A-Za-z0-9_]*)\(", earlier)
                if assignment and aliases.get(assignment.group("class"), assignment.group("class")) in source_symbols:
                    return "__len__"
        list_call = re.match(r"^list\((?P<inner>.+)\)$", last)
        if list_call:
            inner = list_call.group("inner").strip()
            method = re.match(r"^(?P<receiver>[A-Za-z_][A-Za-z0-9_]*)\.(?P<method>[A-Za-z_][A-Za-z0-9_]*)\(", inner)
            if method:
                return method.group("method")
            receiver = re.match(r"^(?P<receiver>[A-Za-z_][A-Za-z0-9_]*)$", inner)
            if receiver:
                receiver_name = receiver.group("receiver")
                for earlier in parts[:-1]:
                    assignment = re.match(rf"^{re.escape(receiver_name)}\s*=\s*(?P<class>[A-Za-z_][A-Za-z0-9_]*)\(", earlier)
                    if assignment and aliases.get(assignment.group("class"), assignment.group("class")) in source_symbols:
                        return "__iter__"
        receiver_call = re.match(r"^(?P<receiver>[A-Za-z_][A-Za-z0-9_]*)\.(?P<method>[A-Za-z_][A-Za-z0-9_]*)\(", last)
        if receiver_call:
            receiver = receiver_call.group("receiver")
            for earlier in parts[:-1]:
                assignment = re.match(rf"^{re.escape(receiver)}\s*=\s*(?P<class>[A-Za-z_][A-Za-z0-9_]*)\(", earlier)
                if assignment:
                    class_name = aliases.get(assignment.group("class"), assignment.group("class"))
                    if class_name in source_symbols:
                        return receiver_call.group("method")
        cleaned = last
    constructor = re.match(r"^(?P<class>[A-Za-z_][A-Za-z0-9_]*)\(", cleaned)
    if constructor:
        class_name = aliases.get(constructor.group("class"), constructor.group("class"))
        if class_name in source_symbols:
            method = re.match(r"^[A-Za-z_][A-Za-z0-9_]*\(.*\)\.(?P<method>[A-Za-z_][A-Za-z0-9_]*)\(", cleaned)
            if method:
                return method.group("method")
            return class_name
    call = re.match(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\(", cleaned)
    if call:
        name = aliases.get(call.group("name"), call.group("name"))
        if not source_symbols or name in source_symbols:
            return name
        len_constructor = re.match(r"^len\((?P<class>[A-Za-z_][A-Za-z0-9_]*)\(.*\)\)$", cleaned)
        if len_constructor and aliases.get(len_constructor.group("class"), len_constructor.group("class")) in source_symbols:
            return "__len__"
        list_method = re.match(r"^list\([A-Za-z_][A-Za-z0-9_]*\(.*\)\.(?P<method>[A-Za-z_][A-Za-z0-9_]*)\(.*\)\)$", cleaned)
        if list_method:
            return list_method.group("method")
        list_constructor = re.match(r"^list\((?P<class>[A-Za-z_][A-Za-z0-9_]*)\(.*\)\)$", cleaned)
        if list_constructor and aliases.get(list_constructor.group("class"), list_constructor.group("class")) in source_symbols:
            return "__iter__"
    method_call = re.match(r"^.+\.(?P<method>[A-Za-z_][A-Za-z0-9_]*)\(", cleaned)
    if method_call:
        method = method_call.group("method")
        if not source_symbols or method in source_symbols:
            return method
    attr = re.match(r"^(?P<class>[A-Za-z_][A-Za-z0-9_]*)\(.*\)\.[A-Za-z_][A-Za-z0-9_]*$", cleaned)
    if attr and aliases.get(attr.group("class"), attr.group("class")) in source_symbols:
        return aliases.get(attr.group("class"), attr.group("class"))
    return ""


def test_spec_iter_test_functions(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for item in getattr(tree, "body", []):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test"):
            functions.append(item)
        if isinstance(item, ast.ClassDef):
            for child in item.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test"):
                    functions.append(child)
    return functions


def test_spec_assignment_expr(stmt: ast.stmt, local_exprs: dict[str, str]) -> tuple[str, str] | None:
    if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
        return None
    target: ast.AST | None = None
    value: ast.AST | None = None
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        value = stmt.value
    elif isinstance(stmt, ast.AnnAssign):
        target = stmt.target
        value = stmt.value
    if not isinstance(target, ast.Name) or value is None:
        return None
    return target.id, node_expr(value, local_exprs)


def test_spec_assignment_value_node(stmt: ast.stmt) -> ast.AST | None:
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        return stmt.value
    if isinstance(stmt, ast.AnnAssign):
        return stmt.value
    return None


def test_spec_is_source_constructor(expr: str, source_symbols: set[str], aliases: dict[str, str]) -> bool:
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\(", expr)
    if not match:
        return False
    name = aliases.get(match.group(1), match.group(1))
    return not source_symbols or name in source_symbols


def test_spec_assigned_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        targets: list[ast.AST] = []
        if isinstance(child, ast.Assign):
            targets = list(child.targets)
        elif isinstance(child, ast.AnnAssign):
            targets = [child.target]
        elif isinstance(child, ast.AugAssign):
            targets = [child.target]
        elif isinstance(child, ast.For):
            targets = [child.target]
        for target in targets:
            for item in ast.walk(target):
                if isinstance(item, ast.Name):
                    names.add(item.id)
    return names


def test_spec_cli_result_access(node: ast.AST, local_exprs: dict[str, str]) -> tuple[str, str] | None:
    attrs: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        attrs.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name) or not attrs:
        return None
    command_expr = local_exprs.get(current.id, "")
    if not command_expr:
        return None
    command_lower = command_expr.lower()
    if not (
        "subprocess.run(" in command_lower
        or re.search(r"\b(?:_run|run_cli|cli|invoke|run_command)\s*\(", command_expr)
    ):
        return None
    attr_path = ".".join(reversed(attrs))
    if attr_path not in {"returncode", "stdout", "stderr"}:
        return None
    return command_expr, attr_path


def test_spec_call_may_mutate_state(call: ast.Call) -> bool:
    name = test_spec_call_name(call)
    return name in {
        "add",
        "add_student",
        "append",
        "clear",
        "delete",
        "discard",
        "insert",
        "next",
        "pop",
        "push",
        "remove",
        "reset",
        "set",
        "update",
    }


def select_test_spec_examples(examples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for item in examples:
        symbol = str(item.get("symbol") or "")
        if symbol not in grouped:
            grouped[symbol] = []
            order.append(symbol)
        grouped[symbol].append(item)
    if len(examples) <= limit:
        return examples
    selected: list[dict[str, Any]] = []
    per_symbol_limit = max(1, min(8, (limit + max(1, len(order)) - 1) // max(1, len(order))))
    cursors = {symbol: 0 for symbol in order}
    while len(selected) < limit:
        progressed = False
        for symbol in order:
            if len(selected) >= limit:
                break
            cursor = cursors[symbol]
            if cursor >= len(grouped[symbol]) or cursor >= per_symbol_limit:
                continue
            selected.append(grouped[symbol][cursor])
            cursors[symbol] += 1
            progressed = True
        if progressed:
            continue
        for symbol in order:
            while cursors[symbol] < len(grouped[symbol]) and len(selected) < limit:
                selected.append(grouped[symbol][cursors[symbol]])
                cursors[symbol] += 1
        break
    return selected


def split_test_example(example: str) -> tuple[str, str, str]:
    if " -> " in example:
        expr, expected = example.split(" -> ", 1)
        return "value", expr.strip(), expected.strip()
    match = re.match(r"^(?P<expr>.+?)\s+raises\s+(?P<raises>[A-Za-z_][\w.]*)(?:\((?P<message>.*)\))?$", example)
    if match:
        expected = str(match.group("raises") or "").strip()
        message = str(match.group("message") or "").strip()
        if message:
            expected += f"({message})"
        return "raises", str(match.group("expr") or "").strip(), expected
    return "", "", ""


def test_example_probe_expressions(examples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in examples:
        if not isinstance(item, dict):
            continue
        kind, expr, expected = split_test_example(str(item.get("example") or ""))
        if not kind or not expr or not expected:
            continue
        if re.search(r"\bself\.", expr) or re.search(r"\bself\.", expected):
            continue
        rows.append(
            {
                "kind": kind,
                "expr": expr,
                "expected": expected,
                "symbol": str(item.get("symbol") or ""),
                "line": int(item.get("line") or 1),
            }
        )
        if len(rows) >= max(1, int(limit)):
            break
    return rows
