from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re
import textwrap
from typing import Any, Callable


def human_test_name(name: str) -> str:
    text = re.sub(r"^test_?", "", name.strip())
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or name


def python_parse_text(content: str) -> str:
    return content[1:] if content.startswith("\ufeff") else content


@dataclass(frozen=True)
class EditIntentRoutePlan:
    ok: bool
    path: str
    clean_intent: str
    old: str
    new: str
    clean_scope: str
    route: str = ""
    routed_tool: str = ""
    operation: dict[str, Any] | None = None
    replace_all: bool = False
    match_whole_word: bool = False
    summary: str = ""
    error_class: str | None = None


TEXT_REPLACE_INTENTS = {
    "replace",
    "replace_text",
    "text",
    "text_replace",
    "replace_in_file",
    "string_replace",
    "literal_replace",
    "update_text",
}

ADD_SYMBOL_INTENTS = {"add_function", "append_function", "create_function", "add_symbol", "append_symbol"}
SYMBOL_INTENTS = {"replace_symbol", "replace_function", "replace_class", "symbol"}
RENAME_INTENTS = {"rename", "rename_symbol", "rename_symbol_project", "update_callers", "refactor_rename"}
PROJECT_SCOPES = {"project", "repo", "repository", "all"}


def looks_like_symbol_name(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", value.strip()))


def looks_like_full_symbol_source(path_suffix: str, value: str) -> bool:
    stripped = value.lstrip()
    if path_suffix.lower() == ".py":
        return stripped.startswith(("def ", "async def ", "class "))
    return bool(re.match(r"(?:export\s+)?(?:async\s+)?(?:function|class)\s+\w+", stripped))


def single_python_replacement_symbol_name(value: str) -> str:
    try:
        tree = ast.parse(python_parse_text(value))
    except SyntaxError:
        return ""
    candidates = [
        child.name
        for child in tree.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    return candidates[0] if len(candidates) == 1 else ""


def looks_like_function_body_edit_intent(intent: str) -> bool:
    normalized = re.sub(r"[^a-z0-9_]+", "_", intent.lower())
    words = {word for word in normalized.split("_") if word}
    return bool(words & {"body", "implementation", "function", "method", "fix", "correct", "update"})


def normalize_python_symbol_target(path_suffix: str, intent: str, symbol: str) -> str:
    if path_suffix.lower() != ".py" or looks_like_symbol_name(symbol):
        return symbol
    clean_intent = str(intent or "").strip().lower().replace("-", "_")
    symbol_intents = {
        "replace_body",
        "replace_function_body",
        "function_body",
        "change_signature",
        "replace_signature",
        "signature",
        "replace_symbol",
        "replace_function",
        "replace_class",
        "symbol",
    }
    if clean_intent not in symbol_intents and not looks_like_function_body_edit_intent(clean_intent):
        return symbol
    match = re.match(r"\s*(?:async\s+def|def|class)?\s*([A-Za-z_][\w.]*)\s*(?:\(|:|$)", symbol)
    return match.group(1) if match else symbol


def edit_intent_route_plan(
    *,
    relative_path: str,
    path_suffix: str,
    intent: str,
    target: str | None,
    replacement: str | None,
    scope: str,
) -> EditIntentRoutePlan:
    clean_intent = str(intent or "").strip().lower().replace("-", "_")
    old = normalize_python_symbol_target(path_suffix, clean_intent, str(target or "").strip())
    new = "" if replacement is None else str(replacement)
    clean_scope = str(scope or "file").strip().lower()
    is_python = path_suffix.lower() == ".py"

    def plan(
        route: str,
        routed_tool: str,
        *,
        operation: dict[str, Any] | None = None,
        replace_all: bool = False,
        match_whole_word: bool = False,
    ) -> EditIntentRoutePlan:
        return EditIntentRoutePlan(
            ok=True,
            path=relative_path,
            clean_intent=clean_intent,
            old=old,
            new=new,
            clean_scope=clean_scope,
            route=route,
            routed_tool=routed_tool,
            operation=operation,
            replace_all=replace_all,
            match_whole_word=match_whole_word,
        )

    def invalid(summary: str) -> EditIntentRoutePlan:
        return EditIntentRoutePlan(
            ok=False,
            path=relative_path,
            clean_intent=clean_intent,
            old=old,
            new=new,
            clean_scope=clean_scope,
            summary=summary,
            error_class="invalid_args",
        )

    if not clean_intent:
        return invalid("edit_intent requires an intent.")
    if not old and clean_intent not in {"add_import", "add_import_if_missing", *ADD_SYMBOL_INTENTS}:
        return invalid("edit_intent requires target for this intent.")
    if replacement is None and clean_intent != "delete_symbol":
        return invalid("edit_intent requires replacement for this intent.")

    if clean_intent in RENAME_INTENTS and looks_like_symbol_name(old) and looks_like_symbol_name(new):
        if clean_scope in PROJECT_SCOPES:
            return plan(
                "project symbol rename",
                "apply_structured_edit",
                operation={"op": "rename_symbol_project", "path": ".", "old": old.rsplit(".", 1)[-1], "new": new.rsplit(".", 1)[-1]},
            )
        return plan("identifier text rename in file", "replace_in_file", replace_all=False, match_whole_word=True)

    if clean_intent in {"replace_body", "replace_function_body", "function_body"}:
        return plan(
            "replace Python function body",
            "apply_structured_edit",
            operation={"op": "replace_function_body", "path": relative_path, "symbol": old, "body": new},
        )

    if clean_intent in {"change_signature", "replace_signature", "signature"}:
        return plan(
            "change Python function signature",
            "apply_structured_edit",
            operation={"op": "change_signature", "path": relative_path, "symbol": old, "signature": new},
        )

    if clean_intent in {"add_import", "add_import_if_missing"}:
        statement = new.strip() or old
        return plan(
            "add import if missing",
            "apply_structured_edit",
            operation={"op": "add_import_if_missing", "path": relative_path, "statement": statement},
        )

    if clean_intent in ADD_SYMBOL_INTENTS:
        return plan("append Python symbol source", "append_symbol")

    if is_python and looks_like_symbol_name(old) and looks_like_function_body_edit_intent(clean_intent):
        replacement_name = single_python_replacement_symbol_name(new) if looks_like_full_symbol_source(path_suffix, new) else ""
        old_leaf = old.rsplit(".", 1)[-1]
        if replacement_name and replacement_name != old_leaf:
            return plan(
                "project symbol rename from replacement source",
                "apply_structured_edit",
                operation={"op": "rename_symbol_project", "path": ".", "old": old_leaf, "new": replacement_name},
            )
        if looks_like_full_symbol_source(path_suffix, new):
            return plan("replace symbol source", "replace_symbol")
        return plan(
            "replace Python function body",
            "apply_structured_edit",
            operation={"op": "replace_function_body", "path": relative_path, "symbol": old, "body": new},
        )

    if clean_intent in SYMBOL_INTENTS:
        replacement_name = single_python_replacement_symbol_name(new) if is_python and looks_like_full_symbol_source(path_suffix, new) else ""
        old_leaf = old.rsplit(".", 1)[-1]
        new_leaf = new.rsplit(".", 1)[-1]
        if looks_like_symbol_name(old_leaf) and looks_like_symbol_name(new_leaf):
            if clean_scope in PROJECT_SCOPES:
                return plan(
                    "project symbol rename",
                    "apply_structured_edit",
                    operation={"op": "rename_symbol_project", "path": ".", "old": old_leaf, "new": new_leaf},
                )
            return plan(
                "file symbol rename",
                "apply_structured_edit",
                operation={"op": "rename_symbol", "path": relative_path, "old": old_leaf, "new": new_leaf},
            )
        if replacement_name and replacement_name != old_leaf and looks_like_symbol_name(old_leaf):
            return plan(
                "project symbol rename from replacement source",
                "apply_structured_edit",
                operation={"op": "rename_symbol_project", "path": ".", "old": old_leaf, "new": replacement_name},
            )
        if looks_like_symbol_name(old) and looks_like_full_symbol_source(path_suffix, new):
            return plan("replace symbol source", "replace_symbol")
        if is_python and looks_like_symbol_name(old):
            return plan(
                "replace Python function body",
                "apply_structured_edit",
                operation={"op": "replace_function_body", "path": relative_path, "symbol": old, "body": new},
            )
        return plan("symbol-like request routed to text replace because target/replacement is not full symbol source", "replace_in_file")

    if clean_intent in TEXT_REPLACE_INTENTS:
        replace_all = clean_scope in PROJECT_SCOPES or clean_intent in {"rename", "rename_symbol", "refactor_rename"}
        match_whole_word = looks_like_symbol_name(old) and looks_like_symbol_name(new) and "(" not in old
        return plan("replace text in file", "replace_in_file", replace_all=replace_all, match_whole_word=match_whole_word)

    return invalid(
        f"Unknown edit_intent intent: {intent}. Use one of rename, replace_text, "
        "replace_symbol, replace_body, change_signature, add_import, or add_function."
    )


def normalize_python_signature_replacement(
    *,
    symbol: str,
    signature: str,
    expected_name: str,
) -> tuple[str, str]:
    clean = textwrap.dedent(signature).strip()
    if not clean:
        return "", "change_signature requires a non-empty signature."
    if "\n" in clean:
        lines = [line.rstrip() for line in clean.splitlines()]
        start_index = next(
            (
                index
                for index, line in enumerate(lines)
                if line.lstrip().startswith(("def ", "async def "))
            ),
            None,
        )
        if start_index is not None:
            collected: list[str] = []
            paren_balance = 0
            for line in lines[start_index:]:
                stripped = line.strip()
                if not stripped:
                    continue
                collected.append(stripped)
                paren_balance += stripped.count("(") - stripped.count(")")
                if stripped.endswith(":") and paren_balance <= 0:
                    break
            clean = " ".join(collected)
    if not clean.startswith(("def ", "async def ")):
        name = expected_name or symbol.split(".")[-1]
        bare_name = re.match(r"^(?P<prefix>async\s+)?(?P<name>[A-Za-z_]\w*)\s*\(", clean)
        if bare_name:
            prefix = "async def " if bare_name.group("prefix") else "def "
            clean = prefix + clean
        elif clean.startswith("("):
            clean = f"def {name}{clean}"
        else:
            clean = f"def {name}({clean})"
    if not clean.rstrip().endswith(":"):
        clean = clean.rstrip() + ":"
    try:
        tree = ast.parse(f"{clean}\n    pass\n")
    except SyntaxError as exc:
        return "", f"Invalid Python signature: {exc.msg}."
    candidates = [child for child in tree.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(candidates) != 1:
        return "", "change_signature requires one Python function signature."
    if candidates[0].name != expected_name:
        return "", f"Replacement signature defines {candidates[0].name!r}, but target symbol is {expected_name!r}."
    return clean, ""


def merge_from_import_statement(original: str, statement: str) -> str | None:
    match = re.fullmatch(r"from\s+(?P<module>[.\w]+)\s+import\s+(?P<names>.+)", statement.strip())
    if not match:
        return None
    module = match.group("module")
    requested_names = [name.strip() for name in match.group("names").split(",") if name.strip()]
    if not requested_names:
        return None
    lines = original.splitlines(keepends=True)
    existing_pattern = re.compile(rf"^(?P<prefix>\s*from\s+{re.escape(module)}\s+import\s+)(?P<names>.+?)(?P<newline>\r?\n?)$")
    for index, line in enumerate(lines):
        existing = existing_pattern.match(line)
        if not existing:
            continue
        existing_names = [name.strip() for name in existing.group("names").split(",") if name.strip()]
        existing_keys = {name.split(" as ", 1)[0].strip() for name in existing_names}
        missing = [name for name in requested_names if name.split(" as ", 1)[0].strip() not in existing_keys]
        if not missing:
            return original
        newline = existing.group("newline") or ("\n" if line.endswith("\n") else "")
        lines[index] = existing.group("prefix") + ", ".join([*existing_names, *missing]) + newline
        return "".join(lines)
    return None


def insert_single_import_statement(original: str, statement: str) -> str:
    lines = original.splitlines(keepends=True)
    insert_at = 0
    if lines and lines[0].startswith("#!"):
        insert_at = 1
    try:
        tree = ast.parse(python_parse_text(original))
        if (
            tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(getattr(tree.body[0], "value", None), ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        ):
            insert_at = max(insert_at, int(getattr(tree.body[0], "end_lineno", 1)))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                insert_at = max(insert_at, int(getattr(node, "end_lineno", getattr(node, "lineno", 1))))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                insert_at = max(insert_at, int(getattr(node, "end_lineno", getattr(node, "lineno", 1))))
            elif int(getattr(node, "lineno", 1)) > insert_at + 1:
                break
    except SyntaxError:
        pass
    if statement.strip() in {line.strip() for line in lines}:
        return original
    merged = merge_from_import_statement(original, statement)
    if merged is not None:
        return merged
    if not statement.endswith("\n"):
        statement += "\n"
    return "".join(lines[:insert_at]) + statement + "".join(lines[insert_at:])


def insert_import_statement(original: str, statement: str) -> str:
    updated = original
    statements = [line.strip() for line in statement.splitlines() if line.strip()]
    if not statements:
        return original
    for single_statement in statements:
        updated = insert_single_import_statement(updated, single_statement)
    return updated


def python_import_statement_is_safe(statement: str) -> bool:
    try:
        parsed_statement = ast.parse(statement)
    except SyntaxError:
        return False
    return bool(parsed_statement.body) and all(isinstance(node, (ast.Import, ast.ImportFrom)) for node in parsed_statement.body)


def render_symbol_matches(matches: list[dict[str, Any]], *, limit: int = 20) -> str:
    return "\n".join(
        f"{item['start']}-{item['end']} {item['kind']} {item['qualname']}"
        for item in matches[:limit]
    )


def delete_symbol_text_from_found(original: str, found: dict[str, Any]) -> str:
    lines = original.splitlines(keepends=True)
    start = int(found["start"])
    end = int(found["end"])
    while end < len(lines) and not lines[end].strip():
        end += 1
    return "".join(lines[: start - 1]) + "".join(lines[end:])


def append_moved_symbol_text(destination_original: str, moved_text: str) -> str:
    return destination_original.rstrip() + "\n\n" + moved_text.lstrip()


def python_parameter_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    args = node.args
    names = {arg.arg for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names


def python_parameter_sequence(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    args = node.args
    names = [arg.arg for arg in [*args.posonlyargs, *args.args]]
    if args.vararg:
        names.append("*" + args.vararg.arg)
    names.extend(arg.arg for arg in args.kwonlyargs)
    if args.kwarg:
        names.append("**" + args.kwarg.arg)
    return names


def shadowed_builtin_call_diagnostic(node: ast.FunctionDef | ast.AsyncFunctionDef, body: str) -> str:
    shadowable = {"list", "dict", "set", "tuple", "str", "int", "float", "bool", "sum", "map", "filter", "len", "min", "max"}
    shadowed = python_parameter_names(node) & shadowable
    if not shadowed:
        return ""
    indented = "\n".join("    " + line if line.strip() else "" for line in (body.splitlines() or ["pass"]))
    source = f"def __candidate__():\n{indented}\n"
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    for call in (child for child in ast.walk(tree) if isinstance(child, ast.Call)):
        if isinstance(call.func, ast.Name) and call.func.id in shadowed:
            return f"Replacement calls parameter {call.func.id!r} as a function; it shadows the Python builtin. Use a comprehension, rename the local parameter in a full function replacement, or call builtins.{call.func.id} explicitly."
    return ""


def unused_critical_parameter_diagnostic(node: ast.FunctionDef | ast.AsyncFunctionDef, body: str) -> str:
    critical = python_parameter_names(node) & {"initial", "default", "accumulator"}
    if not critical:
        return ""
    indented = "\n".join("    " + line if line.strip() else "" for line in (body.splitlines() or ["pass"]))
    try:
        tree = ast.parse(f"def __candidate__():\n{indented}\n")
    except SyntaxError:
        return ""
    used = {child.id for child in ast.walk(tree) if isinstance(child, ast.Name)}
    missing = sorted(critical - used)
    if not missing:
        return ""
    return "Replacement does not use required accumulator/default parameter(s): " + ", ".join(missing) + ". Include them in the implementation or replace the full function with a justified signature change."


def foldr_argument_order_diagnostic(node: ast.FunctionDef | ast.AsyncFunctionDef, body: str) -> str:
    if node.name.lower() != "foldr":
        return ""
    indented = "\n".join("    " + line if line.strip() else "" for line in (body.splitlines() or ["pass"]))
    try:
        tree = ast.parse(f"def __candidate__():\n{indented}\n")
    except SyntaxError:
        return ""
    for call in (child for child in ast.walk(tree) if isinstance(child, ast.Call)):
        if not isinstance(call.func, ast.Name) or call.func.id != "function" or len(call.args) < 2:
            continue
        first = ast.unparse(call.args[0]) if hasattr(ast, "unparse") else ""
        second = ast.unparse(call.args[1]) if hasattr(ast, "unparse") else ""
        if first in {"item", "el", "element"} and second in {"result", "acc", "accumulator"}:
            return "foldr reducer arguments look reversed. While traversing from the right, call the reducer with accumulator/result first and current element second."
        if first in {"item", "el", "element", "current"} and re.search(r"\b(?:foldr|folder|helper|recurse)\s*\(", second):
            return "foldr reducer arguments look reversed. While traversing from the right, call the reducer with accumulator/result first and current element second, e.g. function(foldr(function, rest, initial), current)."
        if (
            re.fullmatch(r"(?:list|items|values|seq|sequence)\s*\[\s*0\s*\]", first)
            and re.search(r"\bfoldr\s*\(", second)
        ):
            return "foldr reducer arguments look reversed. While traversing from the right, call the reducer with accumulator/result first and current element second, e.g. function(foldr(function, rest, initial), current)."
    return ""


def python_function_replacement_sanity_diagnostic(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    replacement: str,
) -> str:
    parsed_replacement = python_parse_text(replacement)
    try:
        tree = ast.parse(parsed_replacement)
    except SyntaxError:
        return ""
    candidates = [child for child in tree.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(candidates) != 1:
        return ""
    candidate = candidates[0]
    if candidate.name != node.name:
        return f"Replacement defines {candidate.name!r}, but target symbol is {node.name!r}. Use rename/change_signature for intentional API changes."
    existing_params = python_parameter_sequence(node)
    replacement_params = python_parameter_sequence(candidate)
    if replacement_params != existing_params:
        return (
            f"Replacement changes signature for {node.name} from ({', '.join(existing_params)}) "
            f"to ({', '.join(replacement_params)}). Use change_signature for intentional API changes; otherwise keep the existing parameter order."
        )
    if candidate.body:
        lines = parsed_replacement.splitlines()
        start = int(getattr(candidate.body[0], "lineno", 1))
        end = int(getattr(candidate.body[-1], "end_lineno", getattr(candidate.body[-1], "lineno", start)))
        body = textwrap.dedent("\n".join(lines[start - 1 : end]))
    else:
        body = ""
    return (
        shadowed_builtin_call_diagnostic(node, body)
        or unused_critical_parameter_diagnostic(node, body)
        or foldr_argument_order_diagnostic(node, body)
    )


def canonical_signature_order_replacement_if_safe(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    replacement: str,
    diagnostic: str,
) -> str:
    if "Replacement changes signature" not in diagnostic:
        return ""
    parsed_replacement = python_parse_text(replacement)
    try:
        tree = ast.parse(parsed_replacement)
    except SyntaxError:
        return ""
    candidates = [child for child in tree.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(candidates) != 1:
        return ""
    candidate = candidates[0]
    if candidate.name != node.name:
        return ""
    existing_params = python_parameter_sequence(node)
    replacement_params = python_parameter_sequence(candidate)
    existing_names = [entry for entry in existing_params if not entry.startswith("*")]
    replacement_names = [entry for entry in replacement_params if not entry.startswith("*")]

    if not existing_names or not replacement_names:
        return ""
    if len(existing_names) != len(replacement_names):
        return ""

    if sorted(existing_names) == sorted(replacement_names):
        if (
            replacement_names == existing_names
            or node.args.defaults
            or candidate.args.defaults
            or node.args.kw_defaults
            or candidate.args.kw_defaults
            or node.args.vararg
            or candidate.args.vararg
            or node.args.kwarg
            or candidate.args.kwarg
            or node.args.kwonlyargs
            or candidate.args.kwonlyargs
        ):
            return ""
        lines = parsed_replacement.splitlines()
        header_index = int(getattr(candidate, "lineno", 1)) - 1
        if header_index < 0 or header_index >= len(lines):
            return ""
        header = lines[header_index]
        match = re.match(rf"^(\s*(?:async\s+)?def\s+{re.escape(candidate.name)}\s*)\([^)]*\)(\s*(?:->\s*[^:]+)?\s*:\s*)$", header)
        if not match:
            return ""
        lines[header_index] = f"{match.group(1)}({', '.join(existing_names)}){match.group(2)}"
        return "\n".join(lines) + ("\n" if replacement.endswith(("\n", "\r")) else "")

    if (
        len(existing_params) != len(replacement_params)
        or node.args.defaults
        or candidate.args.defaults
        or node.args.kw_defaults
        or candidate.args.kw_defaults
        or node.args.vararg
        or candidate.args.vararg
        or node.args.kwarg
        or candidate.args.kwarg
        or node.args.kwonlyargs
        or candidate.args.kwonlyargs
        or any(name.startswith("*") for name in existing_params + replacement_params)
    ):
        return ""

    for child in ast.walk(candidate):
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)) and child.id in replacement_names:
            return ""

    class _CanonicalBodyRenamer(ast.NodeTransformer):
        def __init__(self, mapping: dict[str, str]):
            self._mapping = mapping
            self._depth = 0

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
            if self._depth == 0:
                self._depth = 1
                node.body = [self.visit(child) for child in node.body]
                self._depth = 0
                return node
            return node

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
            if self._depth == 0:
                self._depth = 1
                node.body = [self.visit(child) for child in node.body]
                self._depth = 0
                return node
            return node

        def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
            return node

        def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:
            return node

        def visit_Name(self, node: ast.Name) -> ast.Name:
            if self._depth == 1 and isinstance(node.ctx, ast.Load):
                mapped = self._mapping.get(node.id)
                if mapped is not None:
                    node = ast.copy_location(ast.Name(id=mapped, ctx=node.ctx), node)
            return node

    mapping = {replacement_name: existing_name for replacement_name, existing_name in zip(replacement_names, existing_names)}
    candidate_rebuilt = deepcopy(candidate)
    posonly_count = len(candidate.args.posonlyargs)

    for index, param_name in enumerate(existing_names):
        if index < posonly_count:
            if index >= len(candidate_rebuilt.args.posonlyargs):
                return ""
            candidate_rebuilt.args.posonlyargs[index].arg = param_name
        else:
            arg_index = index - posonly_count
            if arg_index >= len(candidate_rebuilt.args.args):
                return ""
            candidate_rebuilt.args.args[arg_index].arg = param_name

    candidate_rebuilt = _CanonicalBodyRenamer(mapping).visit(candidate_rebuilt)
    ast.fix_missing_locations(candidate_rebuilt)
    try:
        normalized = ast.unparse(candidate_rebuilt)
    except (SyntaxError, ValueError):
        return ""
    return normalized + "\n"


def canonical_foldr_replacement_if_safe(node: ast.FunctionDef | ast.AsyncFunctionDef, diagnostic: str) -> str:
    if "foldr reducer arguments look reversed" not in diagnostic:
        return ""
    if node.name.lower() != "foldr":
        return ""
    params = python_parameter_sequence(node)
    if params != ["function", "list", "initial"]:
        return ""
    return (
        "def foldr(function, list, initial):\n"
        "    accumulator = initial\n"
        "    for item in reversed(list):\n"
        "        accumulator = function(accumulator, item)\n"
        "    return accumulator\n"
    )


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


def candidate_validation_run_in_temp_workspace(
    *,
    tmp_base: Path,
    rel_source: str,
    candidate_source: str,
    test_path: str | None,
    test_command: str | None,
    selected_test_command: str | None,
    probe_limit: int,
    timeout: int,
    phase_timings_ms: dict[str, float],
    timing_fields: Callable[[], dict[str, float]],
    copy_workspace: Callable[[Path], None],
    create_tools: Callable[[Path, str | None], Any],
    cleanup: Callable[[Path], None],
    truncate_text: Callable[[str, int], str],
    timer: Callable[[], float],
    normalized: str | None,
    signature_warnings: list[str],
) -> dict[str, Any]:
    try:
        tmp_base.mkdir(parents=True, exist_ok=False)
        temp_root = tmp_base / "workspace"
        phase_started_at = timer()
        copy_workspace(temp_root)
        phase_timings_ms["copy_ms"] = round((timer() - phase_started_at) * 1000, 3)
        temp_source = temp_root / rel_source
        temp_source.parent.mkdir(parents=True, exist_ok=True)
        temp_source.write_text(candidate_source, encoding="utf-8")
        temp_tools = create_tools(temp_root, selected_test_command)
        phase_started_at = timer()
        static_result = temp_tools.contract_check([rel_source], limit=12)
        phase_timings_ms["static_ms"] = round((timer() - phase_started_at) * 1000, 3)
        if static_result.get("ok") is not True:
            summary = str(static_result.get("output") or static_result.get("summary") or "candidate static sanity failed")
            return candidate_validation_failure_result(
                path=rel_source,
                stage="static",
                summary=summary,
                output=summary,
                static=static_result,
                signature_warnings=signature_warnings,
                normalized=normalized,
                timing_fields=timing_fields(),
            )
        probe_result: dict[str, Any] | None = None
        probe_failure_summary = ""
        if test_path:
            phase_started_at = timer()
            probe_result = temp_tools.run_test_example_probes(
                rel_source,
                test_path,
                limit=max(1, min(int(probe_limit), 24)),
                timeout=min(max(1, int(timeout)), 60),
            )
            phase_timings_ms["probe_ms"] = round((timer() - phase_started_at) * 1000, 3)
            if probe_result.get("ok") is not True:
                probe_failure_summary = str(probe_result.get("output") or probe_result.get("summary") or "candidate example probes failed")
        run_args: dict[str, Any] = {"timeout": max(1, int(timeout))}
        if test_command:
            run_args["command"] = test_command
        phase_started_at = timer()
        test_result = temp_tools.run_test(**run_args)
        phase_timings_ms["test_ms"] = round((timer() - phase_started_at) * 1000, 3)
        if test_result.get("ok") is not True:
            summary = str(test_result.get("output") or test_result.get("summary") or "candidate tests failed")
            if probe_failure_summary:
                summary = f"example probe mismatches:\n{probe_failure_summary}\n\ntest output:\n{summary}"
            return candidate_validation_failure_result(
                path=rel_source,
                stage="tests",
                summary=truncate_text(summary, 520),
                output=truncate_text(summary, 1600),
                static=static_result,
                probes=probe_result,
                test=test_result,
                signature_warnings=signature_warnings,
                normalized=normalized,
                timing_fields=timing_fields(),
            )
    finally:
        cleanup(tmp_base)
    return candidate_validation_success_result(
        path=rel_source,
        candidate_source=candidate_source,
        normalized=normalized,
        signature_warnings=signature_warnings,
        timing_fields=timing_fields(),
    )


def function_probe_script(module: str, expressions: list[str], function: str | None = None) -> str:
    return (
        "import importlib,json,os,sys,traceback\n"
        "workspace=os.getcwd(); sys.path.insert(0, workspace); sys.path.insert(0, os.path.join(workspace, 'src'))\n"
        f"module_name={json.dumps(module)}\n"
        f"function_name={json.dumps(function or '')}\n"
        f"expressions={json.dumps(expressions)}\n"
        "rows=[]\n"
        "try:\n"
        "    mod=importlib.import_module(module_name)\n"
        "    ns={'module': mod}\n"
        "    if function_name:\n"
        "        ns['fn']=getattr(mod, function_name)\n"
        "    for expr in expressions:\n"
        "        try:\n"
        "            value=eval(expr, ns)\n"
        "            rows.append({'expression': expr, 'ok': True, 'repr': repr(value), 'type': type(value).__name__})\n"
        "        except Exception as exc:\n"
        "            rows.append({'expression': expr, 'ok': False, 'error': type(exc).__name__ + ': ' + str(exc)})\n"
        "except Exception as exc:\n"
        "    rows.append({'expression': '<import>', 'ok': False, 'error': type(exc).__name__ + ': ' + str(exc)})\n"
        "print(json.dumps(rows, ensure_ascii=False))\n"
    )


def function_probe_rendered_output(rows: list[Any], raw_output: str) -> str:
    rendered: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("ok"):
            rendered.append(f"{row.get('expression')}: {row.get('repr')} ({row.get('type')})")
        else:
            rendered.append(f"{row.get('expression')}: ERROR {row.get('error')}")
    return "\n".join(rendered) if rendered else raw_output


def function_probe_result(
    *,
    module: str,
    function: str | None,
    exit_code: int,
    rows: list[Any],
    raw_output: str,
) -> dict[str, Any]:
    return {
        "ok": exit_code == 0 and all(isinstance(row, dict) and row.get("ok") for row in rows),
        "tool": "run_function_probe",
        "module": module,
        "function": function or "",
        "exit_code": exit_code,
        "results": rows,
        "output": function_probe_rendered_output(rows, raw_output),
    }


def repair_common_python_join_typo(content: str) -> str:
    return re.sub(r"(?m)^(\s*return\s+)['\"]\.join\(", r'\1" ".join(', content)


def strip_python_rewrite_markers(content: str) -> str:
    lines = content.splitlines()
    if len(lines) < 3:
        return content
    first = lines[0].strip().lower()
    last = lines[-1].strip().lower()
    begin_rewrite = bool(re.fullmatch(r"[>=#/\-\s]*begin (?:rewrite|file|source|replacement)[<\-=#/\s]*", first))
    end_rewrite = bool(re.fullmatch(r"[>=#/\-\s]*end (?:rewrite|file|source|replacement)[<\-=#/\s]*", last))
    if begin_rewrite and end_rewrite:
        trailing_newline = "\n" if content.endswith(("\n", "\r\n")) else ""
        return "\n".join(lines[1:-1]) + trailing_newline
    return content


def strip_markdown_quote_prefixes(content: str) -> str:
    lines = content.splitlines(keepends=True)
    if not lines:
        return content
    prefixed = sum(1 for line in lines if re.match(r"^\s*>\s?", line))
    non_empty = sum(1 for line in lines if line.strip())
    first_non_empty = next((line for line in lines if line.strip()), "")
    if non_empty == 0 or (prefixed < max(1, non_empty // 2) and not re.match(r"^\s*>\s?", first_non_empty)):
        return content
    return "".join(re.sub(r"^\s*>\s?", "", line, count=1) for line in lines)


def normalize_python_write_content(
    content: str,
    *,
    is_python: bool,
    syntax_ok: Callable[[str], bool],
) -> tuple[str, str | None]:
    if not is_python:
        return content, None
    if syntax_ok(content):
        return content, None
    candidates: list[tuple[str, str]] = []
    fenced_match = re.match(r"^\s*```(?:python|py)?\s*\n(?P<body>.*?)(?:\n)?```\s*$", content, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        candidates.append((fenced_match.group("body"), "Stripped markdown code fence from Python file content before write."))
    rewrite_marker_stripped = strip_python_rewrite_markers(content)
    if rewrite_marker_stripped != content:
        candidates.append((rewrite_marker_stripped, "Stripped rewrite markers from Python file content before write."))
    quote_stripped = strip_markdown_quote_prefixes(content)
    if quote_stripped != content:
        candidates.append((quote_stripped, "Stripped markdown quote prefixes from Python file content before write."))
    if fenced_match:
        fenced_quote_stripped = strip_markdown_quote_prefixes(fenced_match.group("body"))
        if fenced_quote_stripped != fenced_match.group("body"):
            candidates.append((fenced_quote_stripped, "Stripped markdown code fence and quote prefixes from Python file content before write."))
    join_repaired = repair_common_python_join_typo(content)
    if join_repaired != content:
        candidates.append((join_repaired, "Repaired common Python join string typo before write."))
    for candidate, reason in list(candidates):
        if syntax_ok(candidate):
            return candidate, reason
    bases = [(content, "Auto-dedented Python file content before write."), *candidates]
    for candidate, reason in bases:
        dedented = textwrap.dedent(candidate)
        if dedented == candidate:
            continue
        if syntax_ok(dedented):
            if reason.startswith("Auto-dedented"):
                return dedented, reason
            return dedented, reason[:-1] + " and auto-dedented it."
    return content, None


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
