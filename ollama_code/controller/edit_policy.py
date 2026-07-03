from __future__ import annotations

import re
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
