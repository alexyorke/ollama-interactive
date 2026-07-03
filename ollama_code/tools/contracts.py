from __future__ import annotations

import ast
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
