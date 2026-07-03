from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from ollama_code.agent_protocol import CODE_EDIT_SUFFIXES, ExactFileWriteSpec


def path_looks_like_code_file(path: str) -> bool:
    return Path(str(path or "").replace("\\", "/")).suffix.lower() in CODE_EDIT_SUFFIXES


def snippet_symbol_argument_looks_like_text(value: str) -> bool:
    snippet = str(value or "").strip()
    if not snippet:
        return False
    if "\n" in snippet:
        return True
    if re.match(r"^[A-Za-z_][\w.]*\s*(?:\(|$)", snippet):
        return False
    return bool(re.search(r"\b(?:return|raise|yield|if|else|for|while|with|try|except)\b|[=+\-*/%<>\[\]{}]", snippet))


def shell_looks_like_file_mutation(command: str) -> bool:
    lowered = str(command or "").lower()
    mutation_patterns = [
        r">>?",
        r"\btouch\b",
        r"\bmkdir\b",
        r"\bcp\b",
        r"\bmv\b",
        r"\brm\b",
        r"\bsed\s+-i\b",
        r"\btee\b",
        r"\bcat\s+>+\b",
    ]
    return any(re.search(pattern, lowered) for pattern in mutation_patterns)


def normalize_exact_literal_tool_call(
    name: str,
    arguments: dict[str, Any],
    *,
    exact_file_write: ExactFileWriteSpec | None,
) -> tuple[str, dict[str, Any], str | None]:
    if exact_file_write is None:
        return name, arguments, None
    if name == "write_file" or name == "replace_in_file":
        return (
            "write_file",
            {"path": exact_file_write.path, "content": exact_file_write.line + "\n"},
            "Normalized exact single-line file write to a deterministic write_file call.",
        )
    if name == "read_file":
        return (
            "read_file",
            {"path": exact_file_write.path, "start": 1, "end": 1},
            "Normalized exact single-line confirmation read to the requested file and line range.",
        )
    return name, arguments, None


def normalize_snippet_symbol_edit_call(
    name: str,
    arguments: dict[str, Any],
) -> tuple[str, dict[str, Any], str | None]:
    if name != "replace_symbol":
        return name, arguments, None
    path = str(arguments.get("path", "")).strip()
    symbol = arguments.get("symbol")
    content = arguments.get("content")
    if not path or not isinstance(symbol, str) or not isinstance(content, str):
        return name, arguments, None
    if not snippet_symbol_argument_looks_like_text(symbol):
        return name, arguments, None
    return (
        "replace_in_file",
        {"path": path, "old": symbol, "new": content, "all": False},
        "Normalized snippet-style replace_symbol call to replace_in_file.",
    )


def normalize_file_tool_alias_call(
    name: str,
    arguments: dict[str, Any],
) -> tuple[str, dict[str, Any], str | None]:
    lowered = str(name or "").strip().lower()
    if lowered in {"replace_body", "replace_function_body", "function_body", "replace_method_body"}:
        path = arguments.get("path") or arguments.get("file") or arguments.get("filename")
        target = arguments.get("target") or arguments.get("symbol") or arguments.get("name") or arguments.get("function") or arguments.get("method")
        replacement = (
            arguments.get("replacement")
            if "replacement" in arguments
            else arguments.get("body")
            if "body" in arguments
            else arguments.get("content")
            if "content" in arguments
            else arguments.get("new")
        )
        if isinstance(path, str) and path.strip() and isinstance(target, str) and target.strip() and isinstance(replacement, str):
            normalized: dict[str, Any] = {
                "path": path.strip(),
                "intent": "replace_body",
                "target": target.strip(),
                "replacement": replacement,
            }
            if "scope" in arguments:
                normalized["scope"] = arguments["scope"]
            if "apply" in arguments:
                normalized["apply"] = arguments["apply"]
            return "edit_intent", normalized, f"Normalized unsupported {name} alias to edit_intent."
    if lowered in {"edit_symbol", "fix_symbol", "update_symbol"}:
        path = arguments.get("path") or arguments.get("file") or arguments.get("filename")
        symbol = arguments.get("symbol") or arguments.get("name") or arguments.get("function") or arguments.get("target")
        content = arguments.get("content") or arguments.get("replacement") or arguments.get("new")
        if isinstance(path, str) and path.strip() and isinstance(symbol, str) and symbol.strip() and isinstance(content, str) and content.strip():
            return (
                "edit_intent",
                {
                    "path": path.strip(),
                    "intent": "replace_symbol",
                    "target": symbol.strip(),
                    "replacement": content,
                },
                f"Normalized unsupported {name} alias to edit_intent.",
            )
    if lowered in {"search_implementation_target", "search_implementation", "implementation_search"}:
        query = arguments.get("query") or arguments.get("symbol") or arguments.get("target")
        path = arguments.get("path") or arguments.get("root") or "."
        if isinstance(query, str) and query.strip():
            return (
                "repo_index_search",
                {
                    "query": query.strip(),
                    "path": str(path or "."),
                    "limit": int(arguments.get("limit", 10) or 10),
                },
                f"Normalized unsupported {name} alias to repo_index_search.",
            )
    if lowered in {"edit_implementation_target", "fix_implementation_target", "edit_target", "fix_target"}:
        path = arguments.get("path") or arguments.get("file") or arguments.get("filename")
        replacement = arguments.get("replacement") or arguments.get("content") or arguments.get("new")
        symbol = arguments.get("symbol") or arguments.get("name") or arguments.get("function") or arguments.get("target_symbol")
        target = symbol if isinstance(symbol, str) and symbol.strip() else arguments.get("target")
        if isinstance(path, str) and path.strip() and isinstance(replacement, str) and replacement.strip() and isinstance(target, str) and target.strip():
            normalized_target = target
            if path_looks_like_code_file(path):
                target_match = re.match(r"\s*(?:async\s+def|def|class)?\s*([A-Za-z_][\w.]*)", target)
                normalized_target = target_match.group(1) if target_match else target.strip()
            return (
                "edit_intent",
                {
                    "path": path.strip(),
                    "intent": "replace_symbol" if path_looks_like_code_file(path) else "replace_text",
                    "target": normalized_target,
                    "replacement": replacement,
                },
                f"Normalized unsupported {name} alias to edit_intent.",
            )
    if lowered not in {"edit_file", "modify_file", "update_file"}:
        return name, arguments, None
    path = arguments.get("path") or arguments.get("file") or arguments.get("filename")
    if not isinstance(path, str) or not path.strip():
        return name, arguments, None
    if isinstance(arguments.get("old"), str) and isinstance(arguments.get("new"), str):
        return (
            "replace_in_file",
            {
                "path": path.strip(),
                "old": arguments["old"],
                "new": arguments["new"],
                "replace_all": bool(arguments.get("replace_all", False)),
            },
            f"Normalized unsupported {name} alias to replace_in_file.",
        )
    symbol = arguments.get("symbol") or arguments.get("name") or arguments.get("function") or arguments.get("class")
    content = arguments.get("content")
    if isinstance(symbol, str) and isinstance(content, str):
        return (
            "replace_symbol",
            {"path": path.strip(), "symbol": symbol.strip(), "content": content},
            f"Normalized unsupported {name} alias to replace_symbol.",
        )
    replacements = arguments.get("replacements")
    if isinstance(replacements, list):
        return (
            "replace_symbols",
            {"path": path.strip(), "replacements": replacements},
            f"Normalized unsupported {name} alias to replace_symbols.",
        )
    if isinstance(content, str):
        return (
            "write_file",
            {"path": path.strip(), "content": content},
            f"Normalized unsupported {name} alias to write_file.",
        )
    return name, arguments, None


def decode_accidental_escaped_newlines(value: str) -> str:
    if "\n" in value or "\\n" not in value:
        return value
    if not re.search(r"\b(?:def|class|import|from|return|if|for|while|try|except|function|const|let|var)\b", value):
        return value
    return value.replace("\\r\\n", "\n").replace("\\n", "\n")


def normalize_edit_payload_aliases(
    name: str,
    arguments: dict[str, Any],
) -> tuple[str, dict[str, Any], str | None]:
    if name not in {"write_file", "replace_symbol", "replace_symbols", "replace_in_file"}:
        return name, arguments, None
    updated = deepcopy(arguments)
    changed_reasons: list[str] = []
    if name == "replace_in_file":
        path = updated.get("path") or updated.get("file") or updated.get("filename")
        if isinstance(path, str) and path.strip() and "path" not in updated:
            updated["path"] = path
            changed_reasons.append("normalized file/path alias")
        if "old" not in updated:
            for alias in ("old_text", "target", "find", "search", "before", "existing", "original", "from"):
                value = updated.get(alias)
                if isinstance(value, str):
                    updated["old"] = value
                    changed_reasons.append(f"normalized {alias} to old")
                    break
        if "new" not in updated:
            for alias in ("new_text", "replacement", "content", "replace_with", "after", "value", "to"):
                value = updated.get(alias)
                if isinstance(value, str):
                    updated["new"] = value
                    changed_reasons.append(f"normalized {alias} to new")
                    break
        if "replace_all" not in updated and "all" in updated:
            updated["replace_all"] = bool(updated.get("all"))
            changed_reasons.append("normalized all to replace_all")
        if "match_whole_word" not in updated:
            for alias in ("whole_word", "matchWholeWord"):
                if alias in updated:
                    updated["match_whole_word"] = bool(updated.get(alias))
                    changed_reasons.append(f"normalized {alias} to match_whole_word")
                    break
    for key in ("content", "new"):
        value = updated.get(key)
        if isinstance(value, str):
            decoded = decode_accidental_escaped_newlines(value)
            if decoded != value:
                updated[key] = decoded
                changed_reasons.append(f"decoded escaped newlines in {key}")
    if name == "replace_symbols":
        replacements = updated.get("replacements")
        if isinstance(replacements, list):
            normalized_replacements: list[Any] = []
            for item in replacements:
                if not isinstance(item, dict):
                    normalized_replacements.append(item)
                    continue
                normalized_item = dict(item)
                content = normalized_item.get("content")
                if isinstance(content, str):
                    decoded = decode_accidental_escaped_newlines(content)
                    if decoded != content:
                        normalized_item["content"] = decoded
                        changed_reasons.append("decoded escaped newlines in replacement content")
                normalized_replacements.append(normalized_item)
            updated["replacements"] = normalized_replacements
    if name == "replace_symbol":
        path = str(updated.get("path", "")).strip()
        symbol = updated.get("symbol")
        content = updated.get("content")
        replacements = updated.get("replacements")
        if path and isinstance(replacements, list) and len(replacements) == 1 and isinstance(replacements[0], dict):
            item = replacements[0]
            old = item.get("old")
            new = item.get("new")
            if isinstance(old, str) and isinstance(new, str):
                return (
                    "replace_in_file",
                    {"path": path, "old": old, "new": new, "replace_all": bool(item.get("replace_all", item.get("all", False)))},
                    "Normalized replace_symbol replacement-list payload to replace_in_file.",
                )
        if path and isinstance(symbol, str) and isinstance(content, str) and not path_looks_like_code_file(path):
            return (
                "replace_in_file",
                {"path": path, "old": symbol, "new": content, "replace_all": False},
                "Normalized replace_symbol text edit on a non-code file to replace_in_file.",
            )
    if changed_reasons:
        return name, updated, "Normalized edit payload: " + "; ".join(sorted(set(changed_reasons))) + "."
    return name, arguments, None
