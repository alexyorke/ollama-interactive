from __future__ import annotations

import re
from collections.abc import Callable

from ollama_code.agent_protocol import SymbolReadSpec, TargetLineReadSpec


ToolNamePredicate = Callable[[str], bool]


def _strip_relative_prefix(path: str) -> str:
    normalized = str(path or "").strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _clean_match_group(match: re.Match[str], group: int | str = 1, *, default: str = "", strip_suffix: str = "") -> str:
    try:
        raw_value = match.group(group)
    except IndexError:
        raw_value = default
    value = str(raw_value or default).strip()
    if strip_suffix:
        value = value.rstrip(strip_suffix)
    return value


def _first_request_match(
    text: str,
    patterns: list[str],
    *,
    group: int | str = 1,
    flags: int = re.IGNORECASE,
    strip_suffix: str = "",
) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, str(text or ""), flags=flags)
        if not match:
            continue
        value = _clean_match_group(match, group, strip_suffix=strip_suffix)
        if value:
            return value
    return None


def path_looks_like_doc_target(path: str) -> bool:
    normalized = str(path or "").replace("\\", "/").lstrip("./").lower()
    if not normalized:
        return False
    if normalized == "readme.md" or normalized.endswith("/readme.md"):
        return True
    return normalized.startswith("docs/") or normalized.endswith((".md", ".rst", ".txt"))


def path_looks_like_test_file(path: str) -> bool:
    normalized = str(path or "").replace("\\", "/").lower()
    name = normalized.rsplit("/", 1)[-1]
    return bool(
        re.search(r"(^test_|_test\.|\.test\.|\.spec\.)", name)
        or "/tests/" in normalized
        or normalized.startswith("tests/")
    )


def request_is_continue_prompt(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
    return normalized in {
        "continue",
        "keep going",
        "go on",
        "resume",
        "try again",
        "fix it",
        "finish it",
    }


def tool_names_in_fragment(
    text: str,
    *,
    known_tool_names: set[str],
    is_supported_tool_name: ToolNamePredicate | None = None,
) -> set[str]:
    fragment = str(text or "")
    lowered = fragment.lower()
    supported = is_supported_tool_name or (lambda name: name in known_tool_names)
    matches: set[str] = set()
    for name in known_tool_names:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name.lower())}(?![A-Za-z0-9_])", lowered):
            matches.add(name)
    for match in re.findall(r"(?<![A-Za-z0-9_])(mcp\.[a-z0-9_-]+\.[a-z0-9_.-]+)(?![A-Za-z0-9_])", lowered):
        clean = match.rstrip(".,;:")
        if supported(clean):
            matches.add(clean)
    return matches


def forbidden_tool_names_from_request(
    text: str,
    *,
    known_tool_names: set[str],
    is_supported_tool_name: ToolNamePredicate | None = None,
) -> set[str]:
    lowered = str(text or "").lower()
    masked = re.sub(
        r"mcp\.[a-z0-9_-]+\.[a-z0-9_.-]+",
        lambda match: match.group(0).replace(".", "__mcpdot__"),
        lowered,
    )
    fragments = re.findall(r"\b(?:do not|don't|dont|never|avoid)\b[^.?!\n]{0,160}", masked)
    fragments.extend(re.findall(r"\bwithout(?: using)?\b[^.?!\n]{0,160}", masked))
    fragments.extend(re.findall(r"\bnot\s+(?:with|using|via)?\s*[^.?!\n]{0,80}", masked))
    forbidden: set[str] = set()
    for fragment in fragments:
        forbidden.update(
            tool_names_in_fragment(
                fragment.replace("__mcpdot__", "."),
                known_tool_names=known_tool_names,
                is_supported_tool_name=is_supported_tool_name,
            )
        )
    return forbidden


def requested_tool_names_from_request(
    text: str,
    *,
    known_tool_names: set[str],
    is_supported_tool_name: ToolNamePredicate | None = None,
    forbidden_tool_names: set[str] | None = None,
) -> set[str]:
    lowered = str(text or "").lower()
    fragments = re.findall(r"\b(?:use|call|run|invoke|start)\b[^.?!\n]{0,160}", lowered)
    requested: set[str] = set()
    for fragment in fragments:
        requested.update(
            tool_names_in_fragment(
                fragment,
                known_tool_names=known_tool_names,
                is_supported_tool_name=is_supported_tool_name,
            )
        )
    requested.update(
        tool_names_in_fragment(
            lowered,
            known_tool_names=known_tool_names,
            is_supported_tool_name=is_supported_tool_name,
        )
    )
    if forbidden_tool_names:
        requested.difference_update(forbidden_tool_names)
    return requested


def request_explicitly_requests_tool(
    text: str,
    name: str,
    *,
    known_tool_names: set[str],
    is_supported_tool_name: ToolNamePredicate | None = None,
) -> bool:
    requested = requested_tool_names_from_request(
        text,
        known_tool_names=known_tool_names,
        is_supported_tool_name=is_supported_tool_name,
        forbidden_tool_names=set(),
    )
    return name in requested


def request_requires_tools(text: str) -> bool:
    lowered = str(text or "").lower()
    tool_phrases = [
        "read file",
        "read the file",
        "search",
        "grep",
        "list files",
        "list the files",
        "workspace",
        "filesystem",
        "directory",
        "folder",
        "repo",
        "repository",
        "project",
        "create ",
        "write ",
        "replace ",
        "edit ",
        "update ",
        "run ",
        "execute ",
        "shell",
        "command",
        "test",
        "tests",
        "pytest",
        "unittest",
        "git",
        "checkout",
        "checked out",
        "merge",
        "rebase",
        "stash",
        "working tree",
        "staged",
        "unstaged",
        "commit ",
        "branch",
        "diff",
        "sub-agent",
        "subagent",
        "helper agent",
        "run_agent",
        "run_test",
        "code_outline",
        "read_symbol",
        "search_symbols",
    ]
    if any(phrase in lowered for phrase in tool_phrases):
        return True
    return bool(re.search(r"\b[\w./-]+\.[A-Za-z0-9]+\b", str(text or "")))


def request_prefers_structured_file_tools(text: str) -> bool:
    lowered = str(text or "").lower()
    if "run_shell" in lowered and "not run_shell" not in lowered:
        return False
    if "shell" in lowered and "not shell" not in lowered:
        return False
    if "command" in lowered and "run_test" not in lowered:
        return False
    file_verbs = ["create ", "write ", "replace ", "edit ", "update ", "rewrite ", "append "]
    has_file_target = bool(re.search(r"\b[\w./-]+\.[A-Za-z0-9]+\b", str(text or ""))) or "/" in str(text or "")
    return has_file_target and any(verb in lowered for verb in file_verbs)


def request_targets_session_memory(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(
        phrase in lowered
        for phrase in [
            "earlier in this session",
            "earlier in this conversation",
            "what token did i ask you to remember",
            "what did i ask you to remember",
            "remember earlier",
            "remember in this session",
        ]
    )


def request_allows_commit(text: str) -> bool:
    lowered = str(text or "").lower()
    return bool(re.search(r"\b(?:commit|git_commit)\b", lowered))


def request_asks_if_command_works(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(
        phrase in lowered
        for phrase in [
            "whether it works",
            "if it works",
            "whether the command works",
            "if the command works",
            "tell me whether it works",
            "tell me if it works",
        ]
    )


def request_asks_if_path_exists(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(
        phrase in lowered
        for phrase in [
            "whether it exists",
            "if it exists",
            "whether the file exists",
            "if the file exists",
            "whether the path exists",
            "if the path exists",
            "tell me whether it exists",
            "tell me if it exists",
        ]
    )


def request_expects_exact_tool_error(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(
        phrase in lowered
        for phrase in [
            "exact tool error",
            "tell me the exact tool error",
            "reply with the exact tool error",
            "what happened",
            "tell me what happened",
        ]
    )


def request_mentions_repeated_read(text: str) -> bool:
    lowered = str(text or "").lower()
    return "twice" in lowered or "two times" in lowered or "2 times" in lowered


def request_asks_token_only(text: str) -> bool:
    lowered = str(text or "").lower()
    return "token" in lowered and ("only" in lowered or "exact marker" in lowered or "exact token" in lowered)


def request_asks_exact_line_text(text: str) -> bool:
    lowered = str(text or "").lower()
    if "line" not in lowered or "exact" not in lowered:
        return False
    return any(phrase in lowered for phrase in ["text on line", "line text", "line only", "that line only"])


def request_asks_specific_file_line(text: str) -> bool:
    lowered = str(text or "").lower()
    if "line" not in lowered:
        return False
    if not re.search(r"\b[\w./-]+\.[A-Za-z0-9]+\b", str(text or "")):
        return False
    return bool(re.search(r"\bline\s+\d+\b", lowered))


def request_asks_symbol_return(text: str) -> bool:
    lowered = str(text or "").lower()
    return bool(
        re.search(
            r"\bwhat\s+does\b.*\breturn\b|\breturns?\s+what\b|\btell\s+me\s+what\b.*\breturns?\b|\bsummarize\b.*\breturns?\b|\breturn\s+value\b|\bvalue\s+it\s+returns?\b",
            lowered,
        )
    )


def requested_read_file_path(text: str) -> str | None:
    return _first_request_match(
        text,
        [
            r"\bread_file\s+on\s+(?P<path>[\w./\\:-]+)",
            r"\bread_file\s+(?P<path>[\w./\\:-]+)",
        ],
        group="path",
        strip_suffix=".,;:",
    )


def requested_natural_read_file_path(text: str) -> str | None:
    return _first_request_match(
        text,
        [
            r"\bwhat does\s+(?P<path>[\w./\\:-]+\.[A-Za-z0-9]+)\s+(?:say|contain)\b",
            r"\btell me what\s+(?P<path>[\w./\\:-]+\.[A-Za-z0-9]+)\s+(?:says|contains)\b",
        ],
        group="path",
        strip_suffix=".,;:",
    )


def request_asks_direct_file_contents(text: str, *, requested_file_path: str | None = None) -> bool:
    lowered = str(text or "").lower()
    if any(word in lowered for word in ["summarize", "summary", "explain", "why"]):
        return False
    return requested_file_path is not None


def requested_mutation_paths(text: str, *, mutation_required: bool) -> set[str]:
    if not mutation_required:
        return set()
    paths: set[str] = set()
    for raw_path in re.findall(r"(?<![\w./\\-])(?:\.?[\w.-]+[\\/])+[\w.-]+\.[A-Za-z0-9]+\b", str(text or "")):
        normalized = _strip_relative_prefix(raw_path.strip().strip("`'\"").rstrip(".,;:"))
        if normalized:
            paths.add(normalized)
    return paths


def requested_git_tool_path(text: str) -> str | None:
    return _first_request_match(
        text,
        [
            r"\bgit_status\s+on\s+(?P<path>[\w./\\:-]+)",
            r"\bgit_diff\s+on\s+(?P<path>[\w./\\:-]+)",
            r"\bgit\s+diff\s+(?P<path>[\w./\\:-]+)",
        ],
        group="path",
        strip_suffix=".,;:",
    )


def requested_list_files_path(text: str) -> str | None:
    lowered = str(text or "").lower().strip()
    if lowered in {"ls", "dir", "list files", "list the files", "show files", "show the files", "list_files", "use list_files"}:
        return "."
    path = _first_request_match(
        text,
        [
            r"\b(?:list|show)\s+(?:the\s+)?files\s+(?:in|under|for|from)\s+(?:the\s+)?(?P<path>[\w./\\:-]+)",
            r"\b(?:use\s+)?list_files\s+(?:on|in|under|for|from)\s+(?:the\s+)?(?P<path>[\w./\\:-]+)",
            r"\b(?:ls|dir)\s+(?P<path>[\w./\\:-]+)",
        ],
        group="path",
        strip_suffix=".,;:",
    )
    if path:
        return "." if path.lower() in {"the", "workspace", "repo", "repository", "project", "directory", "folder"} else path
    if re.search(r"\b(?:list|show)\s+(?:the\s+)?files\b|\blist_files\b", lowered):
        return "."
    return None


def requested_run_test_command(text: str) -> str | None:
    patterns = [
        r"\brun_test\s+to\s+execute\s+(?P<command>.+?)(?:\s+and\b|[.?!]\s|$)",
        r"\buse\s+run_test\s+to\s+execute\s+(?P<command>.+?)(?:\s+and\b|[.?!]\s|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, str(text or ""), flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        command = match.group("command").strip().strip("` ")
        if len(command) >= 2 and command[0] == command[-1] and command[0] in {"'", '"'}:
            command = command[1:-1].strip()
        if command:
            return command
    return None


def requested_local_search_spec(text: str) -> dict[str, object] | None:
    lowered = str(text or "").lower()
    if "web" in lowered and not any(term in lowered for term in ["workspace", "repo", "repository", "code", "files", "project"]):
        return None
    patterns = [
        r"\buse\s+search\s+to\s+find\s+(?P<query>.+?)(?:\s+and\b|[.?!]|$)",
        r"\b(?:search|grep|rg)\s+(?:for\s+)?(?P<query>.+?)(?:\s+in\s+(?P<path>[\w./\\:-]+))?(?:\s+and\b|[,.?!]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, str(text or ""), flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        query = match.group("query").strip().strip("`'\" ")
        query = re.sub(r"\s+in\s+(?:the\s+)?(?:repo|repository|workspace|project)\s*$", "", query, flags=re.IGNORECASE)
        path = _clean_match_group(match, "path", default=".", strip_suffix=".,;:")
        if query and len(query) <= 160:
            return {"query": query, "path": path or ".", "limit": 20}
    return None


def requested_code_outline_path(text: str) -> str | None:
    path_pattern = r"[\w./\\:-]+\.[A-Za-z0-9]+"
    return _first_request_match(
        text,
        [
            rf"\bcode_outline\b\s+(?:on|for|in)?\s*(?P<path>{path_pattern})",
            rf"\buse\s+code_outline\s+(?:on|for|in)\s+(?P<path>{path_pattern})",
            rf"\boutline\s+(?:the\s+)?code\s+(?:in|for)\s+(?P<path>{path_pattern})",
        ],
        group="path",
        strip_suffix=".,;:",
    )


def requested_find_implementation_target_spec(text: str) -> dict[str, object] | None:
    path_pattern = r"[\w./\\:-]+\.[A-Za-z0-9]+"
    path = _first_request_match(
        text,
        [
            rf"\bfind_implementation_target\b.*?\b(?:for|on|path|test_path)\s+(?P<path>{path_pattern})",
            rf"\b(?:identify|find|show)\s+(?:the\s+)?(?:relevant\s+|likely\s+)?implementation\s+(?:target|file|path)(?:s)?\s+(?:for|from)\s+(?P<path>{path_pattern})",
            rf"\bwhich\s+implementation\s+(?:file|path)\s+(?:corresponds\s+to|matches|goes\s+with|for)\s+(?P<path>{path_pattern})",
            rf"\bwhat\s+is\s+the\s+(?:relevant\s+|likely\s+)?implementation\s+(?:file|path)\s+for\s+(?P<path>{path_pattern})",
        ],
        group="path",
        flags=re.IGNORECASE | re.DOTALL,
        strip_suffix=".,;:",
    )
    return {"test_path": path} if path else None


def requested_search_symbols_spec(text: str) -> dict[str, object] | None:
    symbol_pattern = r"[A-Za-z_][\w.]*"
    path_pattern = r"[\w./\\:-]+\.[A-Za-z0-9]+|[\w./\\:-]+"
    patterns = [
        rf"\buse\s+search_symbols\s+to\s+(?:find|locate|search\s+for)\s+(?P<symbol>{symbol_pattern})\s+in\s+(?P<path>{path_pattern})",
        rf"\bsearch_symbols\b.*?\b(?:query|symbol)\s+(?P<symbol>{symbol_pattern}).*?\b(?:path|in|on)\s+(?P<path>{path_pattern})",
        rf"\b(?:find|locate)\s+(?P<symbol>{symbol_pattern})\s+in\s+(?P<path>{path_pattern})\s+using\s+search_symbols\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, str(text or ""), flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        symbol = _clean_match_group(match, "symbol", strip_suffix=".,;:")
        path = _clean_match_group(match, "path", strip_suffix=".,;:")
        if symbol and path:
            return {"query": symbol, "path": path}
    return None


def requested_target_line_read(text: str) -> TargetLineReadSpec | None:
    match = re.search(r"\bread_file\s+on\s+(?P<path>[\w./\\-]+).*?\bline\s+(?P<line>\d+)\b", str(text or ""), flags=re.IGNORECASE | re.DOTALL)
    if not match:
        match = re.search(r"\bread\s+(?P<path>[\w./\\-]+).*?\bline\s+(?P<line>\d+)\b", str(text or ""), flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    path = _clean_match_group(match, "path", strip_suffix=".,;:")
    try:
        line = int(match.group("line"))
    except ValueError:
        return None
    if not path or line < 1:
        return None
    return TargetLineReadSpec(path=path, start=max(1, line - 5), end=line + 5, line=line)


def requested_symbol_read(text: str) -> SymbolReadSpec | None:
    symbol_pattern = r"[A-Za-z_][\w.]*"
    path_pattern = r"[\w./\\:-]+\.[A-Za-z0-9]+"
    patterns = [
        rf"\b(?:function|method|class|symbol)\s+(?P<symbol>{symbol_pattern})\s+(?:in|from)\s+(?P<path>{path_pattern})",
        rf"\b(?:find|locate|search(?:_symbols)?(?:\s+for)?)\s+(?P<symbol>{symbol_pattern})\s+in\s+(?P<path>{path_pattern})",
        rf"\bread_symbol\b.*?\b(?:on|in)\s+(?P<path>{path_pattern}).*?\bsymbol\s+(?P<symbol>{symbol_pattern})",
    ]
    for pattern in patterns:
        match = re.search(pattern, str(text or ""), flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        path = _clean_match_group(match, "path", strip_suffix=".,;:")
        symbol = _clean_match_group(match, "symbol", strip_suffix=".,;:")
        if path and symbol:
            return SymbolReadSpec(path=path, symbol=symbol)
    return None


def request_mentions_workspace_path(text: str) -> bool:
    return bool(re.search(r"\b[\w./-]+\.[A-Za-z0-9]+\b", str(text or "")))


def request_is_broad_or_ambiguous(text: str) -> bool:
    lowered = str(text or "").lower()
    broad_phrases = [
        "inspect this repo",
        "inspect the repo",
        "summarize this repo",
        "summarize the project",
        "what does this project",
        "find bugs",
        "review the codebase",
        "search the codebase",
    ]
    if any(phrase in lowered for phrase in broad_phrases):
        return True
    return "repo" in lowered and not request_mentions_workspace_path(str(text or ""))


def request_benefits_from_systems_lens(text: str) -> bool:
    if request_is_broad_or_ambiguous(text):
        return True
    return bool(
        re.search(
            r"\b(?:debug|root cause|flaky|regression|perf|performance|slow|throughput|profile|benchmark|architecture|design|workflow|pipeline|integration|refactor|migration|system|systems)\b",
            str(text or ""),
            flags=re.IGNORECASE,
        )
    )


def request_benefits_from_todos(text: str, *, mutation_required: bool, test_run_required: bool) -> bool:
    lowered = str(text or "").lower()
    if re.search(r"\b(?:todo|to-do|checklist|task list|plan steps|track progress)\b", lowered):
        return True
    if request_is_broad_or_ambiguous(text) or test_run_required:
        return True
    if mutation_required and re.search(r"\b(?:implement|fix|refactor|debug|profile|migrate|integrate|keep fixing)\b", lowered):
        return True
    return False


def request_forbids_clarifying_questions(text: str) -> bool:
    return bool(re.search(r"\b(?:do not|don't|dont|never|no)\s+(?:ask|clarify|question)s?\b", str(text or ""), flags=re.IGNORECASE))


def request_explicitly_wants_clarification(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:ask (?:me )?(?:a )?questions?|clarify|clarifying question|before you (?:edit|change|implement)|don't assume|do not assume)\b",
            str(text or ""),
            flags=re.IGNORECASE,
        )
    )


def request_has_clarification_risk_signal(text: str) -> bool:
    if request_is_broad_or_ambiguous(text):
        return True
    return bool(
        re.search(
            r"\b(?:keep fixing|make (?:it|this|the app|the cli|the repo) better|improve|optimi[sz]e|throughput|profile|benchmark|"
            r"architecture|design|workflow|integration|migration|public api|schema|compatib|delete|remove|security|auth|permission|"
            r"out of the box|first use|e2e|user experience|ux|tradeoff|default model|default behavior)\b",
            str(text or ""),
            flags=re.IGNORECASE,
        )
    )


def request_looks_like_issue_report(text: str) -> bool:
    lowered = text.lower()
    has_code_context = bool(
        re.search(
            r"`[^`]+`|(?:^|\s)[A-Za-z_][\w./-]*\.(?:py|js|ts|tsx|jsx|java|go|rs|c|cc|cpp)\b|array\(\[|traceback|from [A-Za-z0-9_. ]+ import ",
            text,
        )
    )
    if not has_code_context:
        return False
    issue_patterns = [
        r"\b(?:bug|issue|regression)\b",
        r"\bdoes not\b[^.?!\n]{0,120}\b(?:correctly|properly|compute|return|handle|pass|work)\b",
        r"\breturns?\b[^.?!\n]{0,80}\bwrong\b",
        r"\b(?:incorrect|incorrectly|unexpected(?:ly)?)\b",
        r"\bfails?\b[^.?!\n]{0,120}\b(?:when|with|for|to|under|on)\b",
    ]
    return any(re.search(pattern, lowered) for pattern in issue_patterns)


def request_needs_exact_grounding(text: str) -> bool:
    lowered = text.lower()
    patterns = [
        r"\bexact(?:ly)?\b",
        r"\bline\s+\d+\b",
        r"\bfirst line\b",
        r"\bsingle line\b",
        r"\bwhat(?:'s| is)? .* say\b",
        r"\btoken\b",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def request_allows_mutation(text: str) -> bool:
    lowered = text.lower()
    mutation_phrases = [
        "create ",
        "write ",
        "replace ",
        "edit ",
        "update ",
        "rewrite ",
        "append ",
        "modify ",
        "change ",
        "delete ",
        "remove ",
        "rename ",
        "fix ",
        "implement ",
        "patch ",
        "refactor ",
        "make ",
        "add ",
        "commit ",
    ]
    return any(phrase in lowered for phrase in mutation_phrases) or request_looks_like_issue_report(text)


def request_requires_mutation(text: str) -> bool:
    lowered = text.lower()
    read_only_patterns = [
        r"\bdo not edit\b(?!\s+(?:tests?|test files?)\b)",
        r"\bdon't edit\b(?!\s+(?:tests?|test files?)\b)",
        r"\bdo not modify\b",
        r"\bdon't modify\b",
        r"\bdo not change\b(?!\s+(?:tests?|test files?)\b)",
        r"\bdon't change\b(?!\s+(?:tests?|test files?)\b)",
        r"\bwithout editing\b(?!\s+(?:tests?|test files?)\b)",
        r"\bwithout modifying\b",
        r"\bwithout changing\b(?!\s+(?:tests?|test files?)\b)",
        r"\bno changes\b",
        r"\bread-only\b",
        r"\binspect only\b",
        r"\bsummarize only\b",
    ]
    if any(re.search(pattern, lowered) for pattern in read_only_patterns):
        return False
    if re.search(r"\b(?:how|what|why)\s+(?:would|should|can)\b", lowered):
        return False
    if request_looks_like_issue_report(text):
        return True
    mutation_patterns = [
        r"\bimplement\b",
        r"\bfix\b",
        r"\bpatch\b",
        r"\brefactor\s+(?:[\w./-]+\.[A-Za-z0-9]+|[A-Za-z_][\w./-]*\s+to|code\s+to|module\s+to|tests?\s+to)",
        r"\bedit\b",
        r"\bupdate\b",
        r"\brewrite\b",
        r"\bmodify\b",
        r"\bchange\b",
        r"\bcreate\b",
        r"\bwrite\b",
        r"\badd\b",
        r"\bremove\b",
        r"\bdelete\b",
    ]
    return any(re.search(pattern, lowered) for pattern in mutation_patterns)


def request_forbids_tests(text: str) -> bool:
    lowered = text.lower()
    explicit_skip_patterns = [
        r"\b(?:do not|don't|dont|skip)\s+(?:run|rerun|execute)(?:\s+the)?\s+(?:(?:[\w-]+)\s+){0,4}(?:tests?|test suite)\b",
        r"\b(?:do not|don't|dont|skip)\s+(?:use\s+)?(?:pytest|unittest)\b",
        r"\bwithout\s+(?:running\s+)?(?:tests?|test suite)\b",
        r"\bwithout\s+(?:using\s+)?(?:pytest|unittest)\b",
        r"\bno tests?(?:\s+(?:needed|required|necessary))?\b",
    ]
    return any(re.search(pattern, lowered) for pattern in explicit_skip_patterns)


def request_requires_test_run(text: str) -> bool:
    lowered = text.lower()
    if request_forbids_tests(lowered):
        return False
    patterns = [
        r"\brun (?:the )?tests?\b",
        r"\brerun (?:the )?tests?\b",
        r"\bexecute (?:the )?tests?\b",
        r"\btest suite\b",
        r"\bkeep (?:the )?tests? (?:green|passing)\b",
        r"\btests? (?:stay|stays|remain|remains) (?:green|passing)\b",
        r"\bpytest\b",
        r"\bunittest\b",
        r"\brun_test\b",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def request_requires_code_mutation(text: str) -> bool:
    lowered = text.lower()
    if not request_requires_mutation(text):
        return False
    return bool(
        re.search(r"\b(?:fix|implement|patch|repair|refactor|change|update)\b", lowered)
        and (
            re.search(r"\b(?:implementation|source|code|bug|failing|failure|hidden tests?)\b", lowered)
            or re.search(r"\b(?:fix|patch|repair)\b.{0,120}\btests?\b", lowered)
        )
    )


def request_explicitly_allows_test_mutation(text: str) -> bool:
    lowered = text.lower()
    if any(
        phrase in lowered
        for phrase in [
            "update tests",
            "edit tests",
            "modify tests",
            "change tests",
            "rewrite tests",
            "add tests",
            "test file",
            "test files",
            "tests, and docs",
            "tests and docs",
        ]
    ):
        return True
    return bool(re.search(r"\btests?/[^\s,;:]+", lowered))


def request_forbids_test_mutation(text: str) -> bool:
    lowered = text.lower()
    if any(phrase in lowered for phrase in ["update tests", "edit tests", "modify tests", "change tests", "tests, and docs", "tests and docs"]):
        return False
    return any(
        phrase in lowered
        for phrase in [
            "edit only implementation",
            "only implementation files",
            "do not edit tests",
            "don't edit tests",
            "without editing tests",
            "leave tests unchanged",
        ]
    ) or bool(re.search(r"\b(?:fix|change|edit|update|implement|make)\b.{0,120}\bsrc/[^\s,;:]+", lowered))


def request_forbids_validation(text: str) -> bool:
    lowered = text.lower()
    return bool(
        re.search(
            r"\b(?:do not|don't|dont|skip|without|no)\b[^.?!;\n]{0,80}\b(?:validation|validate|validator|validators|lint|linter|typecheck|type\s+check|sanity\s+check)\b",
            lowered,
        )
    )


def validation_preferences(text: str) -> tuple[bool, bool]:
    forbid_validation = request_forbids_validation(text)
    forbid_tests = forbid_validation or request_forbids_tests(text)
    return not forbid_tests, not forbid_validation


def request_allows_any_validation(text: str) -> bool:
    allow_tests, allow_non_test_validation = validation_preferences(text)
    return allow_tests or allow_non_test_validation
