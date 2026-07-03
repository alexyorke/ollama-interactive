from __future__ import annotations

import ast
import re
from typing import Any


def annotation_text(node: ast.AST | None) -> str:
    if node is None:
        return "Any"
    try:
        return ast.unparse(node)[:120]
    except Exception:
        return "Any"


def contract_args(args: ast.arguments) -> list[dict[str, Any]]:
    defaults_start = len(args.args) - len(args.defaults)
    rows: list[dict[str, Any]] = []
    all_positional = [*args.posonlyargs, *args.args]
    for index, arg in enumerate(all_positional):
        has_default = index >= defaults_start if arg in args.args else False
        rows.append({"name": arg.arg, "annotation": annotation_text(arg.annotation), "required": not has_default, "kind": "positional"})
    if args.vararg:
        rows.append({"name": "*" + args.vararg.arg, "annotation": annotation_text(args.vararg.annotation), "required": False, "kind": "vararg"})
    for index, arg in enumerate(args.kwonlyargs):
        rows.append({"name": arg.arg, "annotation": annotation_text(arg.annotation), "required": args.kw_defaults[index] is None, "kind": "kwonly"})
    if args.kwarg:
        rows.append({"name": "**" + args.kwarg.arg, "annotation": annotation_text(args.kwarg.annotation), "required": False, "kind": "kwarg"})
    return rows


def callable_arity(args: ast.arguments) -> dict[str, Any]:
    positional = [*args.posonlyargs, *args.args]
    required_positional = len(positional) - len(args.defaults)
    required_kwonly = sum(1 for default in args.kw_defaults if default is None)
    return {
        "min": required_positional + required_kwonly,
        "max": None if args.vararg else len(positional),
        "has_vararg": args.vararg is not None,
        "has_kwarg": args.kwarg is not None,
    }


def callable_arity_without_receiver(args: ast.arguments) -> dict[str, Any]:
    arity = callable_arity(args)
    positional = [*args.posonlyargs, *args.args]
    if positional and positional[0].arg in {"self", "cls"}:
        arity["min"] = max(0, int(arity.get("min", 0)) - 1)
        if isinstance(arity.get("max"), int):
            arity["max"] = max(0, int(arity["max"]) - 1)
    return arity


def decorator_leaf(decorator: ast.AST) -> str:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


def decorator_keyword_bool(decorator: ast.AST, name: str, default: bool) -> bool:
    if not isinstance(decorator, ast.Call):
        return default
    for keyword in decorator.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, bool):
            return bool(keyword.value.value)
    return default


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def dataclass_field_init_and_default(value: ast.AST | None) -> tuple[bool, bool]:
    if value is None:
        return True, False
    if not isinstance(value, ast.Call) or call_name(value.func) != "field":
        return True, True
    init = True
    has_default = False
    for keyword in value.keywords:
        if keyword.arg == "init" and isinstance(keyword.value, ast.Constant) and keyword.value.value is False:
            init = False
        elif keyword.arg in {"default", "default_factory"}:
            has_default = True
    return init, has_default


def dataclass_contract_args(node: ast.ClassDef) -> list[dict[str, Any]] | None:
    dataclass_decorator = next((decorator for decorator in node.decorator_list if decorator_leaf(decorator) == "dataclass"), None)
    if dataclass_decorator is None:
        return None
    if not decorator_keyword_bool(dataclass_decorator, "init", True):
        return []

    rows: list[dict[str, Any]] = []
    for child in node.body:
        if not isinstance(child, ast.AnnAssign) or not isinstance(child.target, ast.Name):
            continue
        name = child.target.id
        annotation = annotation_text(child.annotation)
        annotation_leaf = annotation.split(".")[-1]
        if annotation_leaf.startswith("ClassVar") or annotation_leaf == "KW_ONLY":
            continue
        init, has_default = dataclass_field_init_and_default(child.value)
        if not init:
            continue
        rows.append({"name": name, "annotation": annotation, "required": not has_default})
    return rows


def contract_signature(item: dict[str, Any]) -> str:
    args = []
    for arg in item.get("args", []):
        if not isinstance(arg, dict):
            continue
        text = str(arg.get("name", ""))
        annotation = str(arg.get("annotation", "Any"))
        if annotation and annotation != "Any":
            text += f": {annotation}"
        if not arg.get("required", True):
            text += "=?"
        args.append(text)
    returns = str(item.get("returns") or "Any")
    return f"{item.get('symbol')}({', '.join(args)})->{returns}"


def annotation_allows_none(annotation: str) -> bool:
    lowered = annotation.replace(" ", "").lower()
    return lowered in {"any", "none"} or "optional[" in lowered or "|none" in lowered or "none|" in lowered


def annotation_expected_shape(annotation: str) -> str | None:
    lowered = annotation.replace("typing.", "").lower()
    if lowered.startswith(("list", "sequence", "iterable")):
        return "list"
    if lowered.startswith("dict"):
        return "dict"
    if lowered.startswith("set"):
        return "set"
    if lowered.startswith("tuple"):
        return "tuple"
    if lowered in {"int", "str", "float", "bool"}:
        return lowered
    return None


def return_shape_compatible(expected: str, shapes: list[str]) -> bool:
    if not expected:
        return True
    if expected == "tuple":
        return any(shape.startswith("tuple") or shape == "call:tuple" for shape in shapes)
    if expected in {"list", "dict", "set"}:
        return expected in shapes or f"call:{expected}" in shapes
    if expected in {"int", "str", "float", "bool"}:
        container_shapes = {"list", "dict", "set"}
        if f"call:{expected}" in shapes:
            return True
        return not any(shape in container_shapes or shape.startswith("tuple") for shape in shapes)
    return True


def python_body_is_stub(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    statements = [
        child
        for child in node.body
        if not (isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant) and isinstance(child.value.value, str))
    ]
    if not statements:
        return True
    if len(statements) != 1:
        return False
    only = statements[0]
    if isinstance(only, ast.Pass):
        return True
    if isinstance(only, ast.Return):
        return only.value is None or (isinstance(only.value, ast.Constant) and only.value.value is None)
    if isinstance(only, ast.Expr) and isinstance(only.value, ast.Constant):
        value = only.value.value
        return value is Ellipsis or (
            isinstance(value, str) and re.search(r"\b(?:todo|stub|implement|your code)\b", value, flags=re.IGNORECASE) is not None
        )
    if isinstance(only, ast.Raise):
        raised = only.exc
        if isinstance(raised, ast.Call):
            raised = raised.func
        return isinstance(raised, ast.Name) and raised.id == "NotImplementedError"
    return False


def python_assigned_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            names.add(child.id)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(child.name)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                names.add((alias.asname or alias.name).split(".", 1)[0])
        elif isinstance(child, ast.ExceptHandler) and child.name:
            names.add(child.name)
    return names


def python_loaded_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[tuple[str, int]]:
    loaded: set[tuple[str, int]] = set()
    root = node

    class LoadVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, child: ast.FunctionDef) -> Any:
            if child is root:
                self.generic_visit(child)
            return None

        def visit_AsyncFunctionDef(self, child: ast.AsyncFunctionDef) -> Any:
            if child is root:
                self.generic_visit(child)
            return None

        def visit_ClassDef(self, child: ast.ClassDef) -> Any:
            return None

        def visit_Lambda(self, child: ast.Lambda) -> Any:
            return None

        def visit_Name(self, child: ast.Name) -> Any:
            if isinstance(child.ctx, ast.Load):
                loaded.add((child.id, int(getattr(child, "lineno", getattr(root, "lineno", 1)))))

    LoadVisitor().visit(node)
    return loaded


def python_function_param_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    args = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
    names = {arg.arg for arg in args}
    if node.args.vararg is not None:
        names.add(node.args.vararg.arg)
    if node.args.kwarg is not None:
        names.add(node.args.kwarg.arg)
    return names


def python_placeholder_text_diagnostics(rel: str, text: str, limit: int) -> list[str]:
    diagnostics: list[str] = []
    for index, line in enumerate(text.splitlines(), start=1):
        lowered = line.lower()
        stripped = line.strip()
        commentish = stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''")
        patterns = [
            (r"\bplaceholder implementation\b", "placeholder implementation text remains", True),
            (r"\bin a real scenario\b", "speculative placeholder text remains", True),
            (r"\bNote_(?:Interval_)?\b", "fake generated note placeholder remains", True),
            (r"\bTODO\b|\bstub\b|\byour code\b", "TODO/stub placeholder text remains", commentish),
        ]
        for pattern, message, enabled in patterns:
            if not enabled:
                continue
            if re.search(pattern, line, flags=re.IGNORECASE):
                diagnostics.append(f"{rel}:{index} {message}: {lowered.strip()[:100]}")
                break
        if len(diagnostics) >= max(1, int(limit)):
            break
    return diagnostics[: max(1, int(limit))]
