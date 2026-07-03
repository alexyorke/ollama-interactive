from __future__ import annotations

import re
from pathlib import Path

from ollama_code.agent_protocol import CODE_EDIT_SUFFIXES


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
