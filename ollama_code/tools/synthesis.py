from __future__ import annotations

import ast
import re
from typing import Any, Callable


def human_test_name(name: str) -> str:
    text = re.sub(r"^test_?", "", name.strip())
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or name


def python_parse_text(content: str) -> str:
    return content[1:] if content.startswith("\ufeff") else content


def candidate_public_signature_map(source: str) -> dict[str, str]:
    try:
        tree = ast.parse(python_parse_text(source))
    except SyntaxError:
        return {}
    signatures: dict[str, str] = {}

    def arg_shape(arg: ast.arg, has_default: bool = False) -> str:
        return arg.arg + ("=*" if has_default else "")

    def function_shape(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        args = node.args
        defaults = [False] * (len(args.posonlyargs) + len(args.args) - len(args.defaults)) + [True] * len(args.defaults)
        positional = [arg_shape(arg, defaults[index]) for index, arg in enumerate([*args.posonlyargs, *args.args])]
        if args.vararg:
            positional.append("*" + args.vararg.arg)
        elif args.kwonlyargs:
            positional.append("*")
        positional.extend(arg_shape(arg, args.kw_defaults[index] is not None) for index, arg in enumerate(args.kwonlyargs))
        if args.kwarg:
            positional.append("**" + args.kwarg.arg)
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {node.name}({', '.join(positional)})"

    def visit(node: ast.AST, stack: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                qualname = ".".join([*stack, child.name])
                signatures[qualname] = f"class {child.name}"
                visit(child, [*stack, child.name])
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = ".".join([*stack, child.name])
                signatures[qualname] = function_shape(child)
                visit(child, [*stack, child.name])
            else:
                visit(child, stack)

    visit(tree, [])
    return signatures


def candidate_signature_diagnostics(original: str, candidate: str) -> list[str]:
    original_map = candidate_public_signature_map(original)
    candidate_map = candidate_public_signature_map(candidate)
    diagnostics: list[str] = []
    for symbol, signature in original_map.items():
        if symbol not in candidate_map:
            diagnostics.append(f"candidate removed public symbol {symbol}")
            continue
        if candidate_map[symbol] != signature:
            diagnostics.append(f"candidate changed signature for {symbol}: {signature} -> {candidate_map[symbol]}")
    return diagnostics


def candidate_signature_gate(
    signature_diagnostics: list[str],
    *,
    has_behavior_validation: bool,
) -> dict[str, Any]:
    removed_symbol_diagnostics = [item for item in signature_diagnostics if "removed public symbol" in item]
    signature_warnings = [item for item in signature_diagnostics if item not in removed_symbol_diagnostics]
    blocking_diagnostics = removed_symbol_diagnostics
    if not blocking_diagnostics and signature_warnings and not has_behavior_validation:
        blocking_diagnostics = signature_warnings
    return {
        "ok": not blocking_diagnostics,
        "blocking_diagnostics": blocking_diagnostics,
        "removed_symbol_diagnostics": removed_symbol_diagnostics,
        "signature_warnings": signature_warnings,
    }


def candidate_validation_failure_result(
    *,
    path: str,
    stage: str,
    summary: str,
    timing_fields: dict[str, float],
    diagnostics: list[str] | None = None,
    output: str | None = None,
    normalized: str | None = None,
    signature_warnings: list[str] | None = None,
    static: dict[str, Any] | None = None,
    probes: dict[str, Any] | None = None,
    test: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": False,
        "tool": "validate_implementation_candidate",
        "path": path,
        "stage": stage,
        "summary": summary,
        "output": output if output is not None else summary,
        **timing_fields,
    }
    if diagnostics is not None:
        result["diagnostics"] = diagnostics
    if normalized is not None:
        result["normalized"] = normalized
    if signature_warnings is not None:
        result["signature_warnings"] = signature_warnings
    if static is not None:
        result["static"] = static
    if probes is not None:
        result["probes"] = probes
    if test is not None:
        result["test"] = test
    return result


def candidate_validation_success_result(
    *,
    path: str,
    candidate_source: str,
    timing_fields: dict[str, float],
    normalized: str | None,
    signature_warnings: list[str],
) -> dict[str, Any]:
    summary = "candidate passed syntax, static sanity, example probes, and tests"
    return {
        "ok": True,
        "tool": "validate_implementation_candidate",
        "path": path,
        "stage": "passed",
        "candidate_source": candidate_source,
        "normalized": normalized,
        "signature_warnings": signature_warnings,
        "summary": summary,
        "output": summary,
        **timing_fields,
    }


def candidate_workspace_ignored_names(
    names: list[str],
    *,
    generated_dir_name: Callable[[str], bool],
) -> set[str]:
    skipped = {
        ".git",
        ".meta",
        ".ollama-code",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "scratch",
        "verify_scratch",
    }
    return {name for name in names if name in skipped or generated_dir_name(name)}


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


def test_spec_add_raw_example(
    examples: list[dict[str, Any]],
    *,
    expr: str,
    expected: str | None = None,
    raises: str | None = None,
    raises_message: str | None = None,
    line: int,
    test_name: str | None = None,
    source_symbols: set[str],
    aliases: dict[str, str],
) -> None:
    symbol = test_spec_symbol_from_expr(expr, source_symbols, aliases)
    if source_symbols and not symbol:
        return
    if expected is not None:
        text = f"{expr} -> {expected}"
    elif raises is not None:
        text = f"{expr} raises {raises}"
        if raises_message:
            text += f"({raises_message})"
    else:
        return
    item = {"symbol": symbol or "expression", "example": text, "line": line}
    if test_name:
        item["test_name"] = test_name
    if item not in examples:
        examples.append(item)


def test_spec_add_example(
    examples: list[dict[str, Any]],
    *,
    call: ast.Call,
    expected: str | None = None,
    raises: str | None = None,
    raises_message: str | None = None,
    expr_override: str | None = None,
    line: int,
    test_name: str | None = None,
    source_symbols: set[str],
    aliases: dict[str, str],
    local_exprs: dict[str, str] | None = None,
) -> None:
    symbol = test_spec_call_name(call)
    canonical = aliases.get(symbol, symbol)
    expr = expr_override or call_expr(call, local_exprs)
    if source_symbols and canonical not in source_symbols:
        derived_symbol = test_spec_symbol_from_expr(expr, source_symbols, aliases)
        if not derived_symbol:
            return
        canonical = derived_symbol
    if canonical != symbol and aliases.get(symbol) == canonical and expr.startswith(symbol + "("):
        expr = canonical + expr[len(symbol) :]
    if expected is not None:
        text = f"{expr} -> {expected}"
    elif raises is not None:
        text = f"{expr} raises {raises}"
        if raises_message:
            text += f"({raises_message})"
    else:
        return
    item = {"symbol": canonical, "example": text, "line": line}
    if test_name:
        item["test_name"] = test_name
    if item not in examples:
        examples.append(item)


def test_spec_source_symbols_from_text(source_text: str) -> set[str]:
    try:
        tree = ast.parse(python_parse_text(source_text))
    except Exception:
        return set()
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def test_spec_import_aliases(tree: ast.AST, source_path: str | None) -> dict[str, str]:
    if not source_path:
        return {}
    module_name = source_path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module != module_name:
            continue
        for alias in node.names:
            aliases[alias.asname or alias.name] = alias.name
    return aliases


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


def first_behavior_call(statements: list[ast.stmt]) -> ast.Call | None:
    for statement in statements:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call):
                continue
            name = method_name(node)
            if name.startswith("assert"):
                continue
            return node
    return None


def test_spec_receiver_root_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Name) and func.id in {"len", "list", "tuple", "set"} and call.args:
        first_arg = call.args[0]
        if isinstance(first_arg, ast.Name):
            return first_arg.id
        if isinstance(first_arg, ast.Call):
            return test_spec_receiver_root_name(first_arg)
        if isinstance(first_arg, ast.Attribute):
            return test_spec_receiver_root_name_from_node(first_arg)
    if isinstance(func, ast.Attribute):
        node: ast.AST = func.value
        while isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            node = node.func.value
        if isinstance(node, ast.Name):
            return node.id
    for child in ast.walk(call):
        if child is call or not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
            continue
        node = child.func.value
        while isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            node = node.func.value
        if isinstance(node, ast.Name):
            return node.id
    return ""


def test_spec_receiver_root_name_from_node(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        return test_spec_receiver_root_name(node)
    if isinstance(node, ast.Attribute):
        current: ast.AST = node.value
        while isinstance(current, ast.Attribute):
            current = current.value
        if isinstance(current, ast.Call):
            return test_spec_receiver_root_name(current)
        if isinstance(current, ast.Name):
            return current.id
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            receiver = test_spec_receiver_root_name(child)
            if receiver:
                return receiver
        if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
            return child.value.id
    return ""


def test_spec_behavior_expr(
    call: ast.Call,
    local_exprs: dict[str, str],
    object_history: dict[str, list[str]],
) -> str:
    receiver = test_spec_receiver_root_name(call)
    if receiver:
        history = object_history.get(receiver)
        if history:
            expr = call_expr(call, None)
            return "; ".join(history + [expr])
    return call_expr(call, local_exprs)


def test_spec_expr_with_history(node: ast.AST, local_exprs: dict[str, str], object_history: dict[str, list[str]]) -> str:
    receiver = test_spec_receiver_root_name_from_node(node)
    if receiver and receiver in object_history:
        return "; ".join(object_history[receiver] + [node_expr(node, None)])
    return node_expr(node, local_exprs)


def test_spec_expected_expr(node: ast.AST, local_exprs: dict[str, str], object_history: dict[str, list[str]]) -> str:
    receiver = test_spec_receiver_root_name_from_node(node)
    if receiver and receiver in object_history:
        return node_expr(node, None)
    return node_expr(node, local_exprs)


def test_spec_add_text_example(
    examples: list[dict[str, Any]],
    *,
    expr: str,
    text: str,
    line: int,
    test_name: str | None,
    source_symbols: set[str],
    aliases: dict[str, str],
) -> None:
    symbol = test_spec_symbol_from_expr(expr, source_symbols, aliases)
    if source_symbols and not symbol:
        return
    item = {"symbol": symbol or "expression", "example": text, "line": line}
    if test_name:
        item["test_name"] = test_name
    if item not in examples:
        examples.append(item)


def test_spec_add_cli_example(
    examples: list[dict[str, Any]],
    *,
    command_expr: str,
    assertion: str,
    line: int,
    test_name: str | None,
) -> None:
    item = {"symbol": "cli", "example": f"{command_expr} {assertion}", "line": line}
    if test_name:
        item["test_name"] = test_name
    if item not in examples:
        examples.append(item)


def test_spec_add_cli_assertion_examples(
    examples: list[dict[str, Any]],
    *,
    method_name_text: str,
    args: list[ast.AST],
    local_exprs: dict[str, str],
    line: int,
    test_name: str | None,
) -> bool:
    if method_name_text in {"assertEqual", "assertEquals"} and len(args) >= 2:
        for actual_node, expected_node in ((args[0], args[1]), (args[1], args[0])):
            access = test_spec_cli_result_access(actual_node, local_exprs)
            if access is None:
                continue
            command_expr, attr_path = access
            expected = node_expr(expected_node, local_exprs)
            test_spec_add_cli_example(
                examples,
                command_expr=command_expr,
                assertion=f"{attr_path} == {expected}",
                line=line,
                test_name=test_name,
            )
            return True
    if method_name_text in {"assertIn", "assertNotIn"} and len(args) >= 2:
        member = node_expr(args[0], local_exprs)
        access = test_spec_cli_result_access(args[1], local_exprs)
        if access is None:
            return False
        command_expr, attr_path = access
        relation = "contains" if method_name_text == "assertIn" else "does not contain"
        test_spec_add_cli_example(
            examples,
            command_expr=command_expr,
            assertion=f"{attr_path} {relation} {member}",
            line=line,
            test_name=test_name,
        )
        return True
    return False


def test_spec_record_side_effect_call(
    call: ast.Call,
    object_history: dict[str, list[str]],
) -> None:
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
        receiver = call.func.value.id
        if receiver in object_history:
            object_history[receiver].append(call_expr(call, None))


def assert_raises_expected_message(statements: list[ast.stmt], context_var: ast.expr | None, local_exprs: dict[str, str]) -> str | None:
    if not isinstance(context_var, ast.Name):
        return None
    pattern = f"{context_var.id}.exception.args[0]"
    for statement in statements:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call) or method_name(node) not in {"assertEqual", "assertEquals"} or len(node.args) < 2:
                continue
            left = node_expr(node.args[0], local_exprs)
            right = node_expr(node.args[1], local_exprs)
            if left == pattern:
                return right
            if right == pattern:
                return left
    return None


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
