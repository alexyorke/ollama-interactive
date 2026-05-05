from __future__ import annotations

import ast
import configparser
import dis
import difflib
import hashlib
import importlib
import inspect
import io
import json
import os
import re
import signal
import shlex
import shutil
import sqlite3
import subprocess
import sys
import threading
import textwrap
import time
import tomllib
from pathlib import Path
from typing import Any, Callable, Iterable

from ollama_code.interrupts import OperationInterrupted

ApprovalMode = str
AgentRunner = Callable[[dict[str, Any]], dict[str, Any]]
WINDOWS_DRIVE_PATH = re.compile(r"^(?P<drive>[A-Za-z]):(?:[\\/](?P<rest>.*))?$")
WSL_MOUNT_PATH = re.compile(r"^/mnt/(?P<drive>[A-Za-z])(?:/(?P<rest>.*))?$")
CODE_FILE_SUFFIXES = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".kts",
}
REPO_INDEX_VERSION = 2
FILE_INDEX_VERSION = 1
FTS_INDEX_VERSION = 1
FTS_TEXT_SUFFIXES = CODE_FILE_SUFFIXES | {
    ".md",
    ".rst",
    ".txt",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".html",
    ".css",
    ".scss",
}
SKIP_CODE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".ollama-code",
    ".venv",
    "venv",
}
SKIP_GENERATED_DIRS = {
    "scratch",
    "verify_scratch",
    "htmlcov",
    ".tox",
    ".nox",
    ".cache",
}
SKIP_GENERATED_DIR_PREFIXES = ("ollama-code-", "probe-", "verify_")
SKIP_GENERATED_DIR_PATTERNS = (re.compile(r"^tmp[0-9a-z_-]+$", re.IGNORECASE),)
SKIP_WALK_DIRS = SKIP_CODE_DIRS | SKIP_GENERATED_DIRS
SKIP_WALK_GLOBS = (
    "!scratch/**",
    "!verify_scratch/**",
    "!ollama-code-*/**",
    "!probe-*/**",
    "!verify_*/**",
    "!tmp*/**",
)
DENY_MUTATION_DIRS = {".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ollama-code"}
ERROR_CLASS_PATTERNS: dict[str, re.Pattern[str]] = {
    "missing_dependency": re.compile(r"ModuleNotFoundError:\s+No module named ['\"]([^'\"]+)['\"]|No module named ['\"]([^'\"]+)['\"]", re.IGNORECASE),
    "import_error": re.compile(r"ImportError|cannot import name", re.IGNORECASE),
    "syntax_error": re.compile(r"SyntaxError|IndentationError|unexpected EOF|invalid syntax", re.IGNORECASE),
    "test_assertion": re.compile(r"AssertionError|FAILED\s+\S+|FAIL:|E\s+assert", re.IGNORECASE),
    "invalid_args": re.compile(r"unrecognized arguments|invalid option|unknown option|usage:|error: argument|missing required", re.IGNORECASE),
    "command_not_found": re.compile(r"command not found|not recognized as (?:an internal|the name)", re.IGNORECASE),
    "path_missing": re.compile(r"No such file or directory|cannot access|FileNotFoundError|Path does not exist|path .* does not exist", re.IGNORECASE),
    "cwd_git": re.compile(r"cd: .*No such file|outside workspace|escapes the workspace|not inside a git repository", re.IGNORECASE),
    "timeout": re.compile(r"timed out|Timeout|exceeded.*timeout|killed", re.IGNORECASE),
    "permission": re.compile(r"Permission denied|access is denied|Operation not permitted", re.IGNORECASE),
    "patch_apply": re.compile(r"patch failed|does not apply|git apply.*failed|hunk FAILED|error: patch", re.IGNORECASE),
}


TOOL_DESCRIPTIONS = [
    {
        "name": "list_files",
        "arguments": {"path": "relative path, default .", "max_depth": "int, default 4", "limit": "int, default 200"},
        "description": "List files and directories under the workspace.",
    },
    {
        "name": "read_file",
        "arguments": {"path": "relative path", "start": "1-based start line, default 1", "end": "inclusive end line, default 200"},
        "description": "Read a file with line numbers.",
    },
    {
        "name": "search",
        "arguments": {"query": "regex or plain text", "path": "relative path, default .", "limit": "int, default 100"},
        "description": "Search text in the workspace.",
    },
    {
        "name": "file_search",
        "arguments": {"query": "filename/path terms", "path": "relative path, default .", "limit": "int, default 100"},
        "description": "Search cached workspace file paths quickly without reading file contents.",
    },
    {
        "name": "fd_search",
        "arguments": {"query": "fd filename pattern", "path": "relative path, default .", "limit": "int, default 100", "kind": "any|file|dir, default any"},
        "description": "Use fd/fdfind for fast repo-local file discovery when installed.",
    },
    {
        "name": "file_index_refresh",
        "arguments": {"path": "relative path, default .", "limit": "int, default 50000"},
        "description": "Refresh the persistent workspace file path index for fast future file searches.",
    },
    {
        "name": "everything_search",
        "arguments": {"query": "Everything search query", "path": "relative path, default .", "limit": "int, default 100"},
        "description": "Use the native Everything CLI es.exe when installed, constrained to the workspace path.",
    },
    {
        "name": "search_symbols",
        "arguments": {"query": "symbol name regex/text", "path": "relative path, default .", "limit": "int, default 50"},
        "description": "Search code symbols by name without reading full files.",
    },
    {
        "name": "code_outline",
        "arguments": {"path": "file or directory, default .", "max_symbols": "int, default 120"},
        "description": "Show compact code symbols and line ranges without function bodies.",
    },
    {
        "name": "read_symbol",
        "arguments": {"path": "code file", "symbol": "name or qualified name", "include_context": "lines around symbol, default 2"},
        "description": "Read one code symbol body by AST/definition range.",
    },
    {
        "name": "inspect_library_source",
        "arguments": {"target": "Python import path like json.loads or json:loads", "context": "lines around source, default 3", "max_lines": "int, default 160", "include_disassembly": "bool, default false"},
        "description": "Inspect installed Python library source/signature/doc; falls back to bytecode or builtin diagnostics when source is unavailable.",
    },
    {
        "name": "repo_index_search",
        "arguments": {"query": "natural query or symbol", "path": "relative path, default .", "limit": "int, default 10"},
        "description": "Search code with compact ranked snippets instead of whole files.",
    },
    {
        "name": "fts_search",
        "arguments": {"query": "terms or phrase", "path": "relative path, default .", "limit": "int, default 20", "refresh": "bool, default false"},
        "description": "Search the local SQLite FTS5 cache across paths, symbols, headings, and snippets.",
    },
    {
        "name": "fts_refresh",
        "arguments": {"path": "relative path, default .", "limit": "int, default 2000"},
        "description": "Refresh the local SQLite FTS5 cache for fast lexical repository queries.",
    },
    {
        "name": "indexed_search",
        "arguments": {"query": "plain terms or regex-like text", "path": "relative path, default .", "limit": "int, default 100"},
        "description": "Search cached line snippets from the persistent repo index without scanning every file.",
    },
    {
        "name": "repo_index_refresh",
        "arguments": {"path": "relative path, default .", "limit": "int, default 1000"},
        "description": "Refresh the persistent repo index for faster future symbol and text searches.",
    },
    {
        "name": "semgrep_scan",
        "arguments": {"pattern": "Semgrep pattern", "path": "relative path, default .", "lang": "optional language", "limit": "int, default 50"},
        "description": "Run Semgrep structurally when semgrep is installed; useful for syntax-aware code search.",
    },
    {
        "name": "ast_search",
        "arguments": {"pattern": "ast-grep pattern", "path": "relative path, default .", "lang": "optional language", "limit": "int, default 50"},
        "description": "Run ast-grep structural search when ast-grep/sg is installed.",
    },
    {
        "name": "lsp_diagnostics",
        "arguments": {"path": "file or directory, default .", "lang": "optional language", "timeout": "seconds, default 60"},
        "description": "Run available language-server-style diagnostics without installing tools.",
    },
    {
        "name": "lsp_definition",
        "arguments": {"path": "file", "line": "1-based line", "column": "1-based column", "limit": "int, default 20"},
        "description": "Find likely definitions for the symbol at a location using available local indexes.",
    },
    {
        "name": "lsp_references",
        "arguments": {"path": "file", "line": "1-based line", "column": "1-based column", "limit": "int, default 40"},
        "description": "Find likely references for the symbol at a location using local search.",
    },
    {
        "name": "context_pack",
        "arguments": {"request": "user request or task text", "path": "relative path, default .", "limit": "int, default 8"},
        "description": "Build compact ranked evidence from repo index, tests, imports, symbols, and git state.",
    },
    {
        "name": "systems_lens",
        "arguments": {"request": "user request or task text", "path": "relative path, default .", "evidence": "optional current evidence", "limit": "int, default 8"},
        "description": "Frame complex work with boundary, state, scale, viewpoints, coupling, and validation checks.",
    },
    {
        "name": "find_implementation_target",
        "arguments": {"test_path": "optional test file", "path": "optional test/source path alias", "query": "optional symbol/query", "output": "optional failing test output/traceback", "limit": "int, default 12"},
        "description": "Map tests or tracebacks to likely implementation files and symbols.",
    },
    {
        "name": "diagnose_test_failure",
        "arguments": {"output": "run_test output", "path": "relative path, default .", "limit": "int, default 12"},
        "description": "Group failing tests by likely root cause with expected/actual and likely target files.",
    },
    {
        "name": "run_function_probe",
        "arguments": {"module": "python module", "expressions": "list of Python expressions", "function": "optional function name", "timeout": "seconds, default 30"},
        "description": "Run small Python expressions against a target module/function and return actual values/errors.",
    },
    {
        "name": "call_graph",
        "arguments": {"path": "file or directory", "symbol": "optional symbol", "limit": "int, default 40"},
        "description": "Show static Python callers, callees, imports, and affected tests.",
    },
    {
        "name": "contract_graph",
        "arguments": {"path": "file or directory, default .", "symbol": "optional symbol", "limit": "int, default 40"},
        "description": "Show compact function contracts, callers/callees, return-shape, and purity hints.",
    },
    {
        "name": "lint_typecheck",
        "arguments": {"paths": "path or list of paths, default .", "command": "optional lint/type command", "timeout": "seconds, default 120"},
        "description": "Run deterministic syntax/lint/type checks with exact file lines.",
    },
    {
        "name": "discover_validators",
        "arguments": {"path": "relative path, default .", "limit": "int, default 12"},
        "description": "Detect focused test/lint/typecheck commands for Python, JS/TS, Go, Rust, Java, and C/C++ projects.",
    },
    {
        "name": "diagnose_dependency_error",
        "arguments": {"output": "error output", "path": "relative path, default ."},
        "description": "Classify missing dependency/import/command/path failures and suggest safe next steps.",
    },
    {
        "name": "contract_check",
        "arguments": {"changed_files": "list of changed source files", "changed_symbols": "optional list of symbols", "limit": "int, default 80"},
        "description": "Check changed Python functions for arity, return-shape, async/yield, and annotation contract issues.",
    },
    {
        "name": "select_tests",
        "arguments": {"changed_files": "list of changed source files", "changed_symbols": "optional list of symbols", "limit": "int, default 8"},
        "description": "Select focused test commands for changed files using imports, names, and symbols.",
    },
    {
        "name": "replace_symbol",
        "arguments": {"path": "code file", "symbol": "name or qualified name", "content": "replacement symbol source"},
        "description": "Replace one function/class/method by symbol range. Python replacements must keep the full file syntactically valid.",
    },
    {
        "name": "replace_symbols",
        "arguments": {"path": "code file", "replacements": "list of {symbol, content} objects"},
        "description": "Replace multiple functions/classes/methods in one syntactically checked edit.",
    },
    {
        "name": "write_file",
        "arguments": {"path": "relative path", "content": "full new file content"},
        "description": "Create or fully replace a file.",
    },
    {
        "name": "replace_in_file",
        "arguments": {
            "path": "relative path",
            "old": "text to replace",
            "new": "replacement text",
            "replace_all": "bool, default false",
            "match_whole_word": "bool, default false",
        },
        "description": "Replace text inside an existing file. Prefer unique snippets; use match_whole_word for standalone words.",
    },
    {
        "name": "apply_structured_edit",
        "arguments": {"operation": "JSON operation: rename_symbol, replace_symbol, add_import, move_symbol, delete_symbol"},
        "description": "Apply a mechanical code edit from an intent JSON, then syntax-check and return diff.",
    },
    {
        "name": "edit_intent",
        "arguments": {"path": "relative path", "intent": "rename|replace_text|replace_symbol|replace_body|change_signature|add_import", "target": "old text/symbol", "replacement": "new text/source", "scope": "file|project, default file", "apply": "bool, default true"},
        "description": "Route an edit intent to the safest low-level edit tool and avoid common symbol/text mixups.",
    },
    {
        "name": "generate_tests_from_spec",
        "arguments": {"target_symbol": "symbol under test", "behavior": "expected behavior", "test_path": "optional output test path", "apply": "bool, default false"},
        "description": "Generate a compact pytest test patch from a spec; preview by default, apply only when requested.",
    },
    {
        "name": "browser_smoke",
        "arguments": {"url": "URL to open", "actions": "optional action list", "wait_for": "optional selector/text", "viewport": "desktop|mobile, default desktop", "screenshot": "bool, default false"},
        "description": "Run a Playwright smoke check for browser/UI tasks when Playwright is installed.",
    },
    {
        "name": "security_scan",
        "arguments": {"path": "relative path, default .", "scanners": "auto or scanner list", "limit": "int, default 80", "timeout": "seconds, default 120"},
        "description": "Run available local security/dependency scanners only when explicitly requested.",
    },
    {
        "name": "mcp_list_tools",
        "arguments": {"server": "optional configured MCP server name", "timeout": "seconds, default 15"},
        "description": "List tools exposed by configured MCP servers.",
    },
    {
        "name": "mcp_call",
        "arguments": {"server": "configured MCP server name", "tool": "tool name", "arguments": "tool arguments object", "timeout": "seconds, default 30"},
        "description": "Call a configured MCP server tool through a guarded stdio JSON-RPC request.",
    },
    {
        "name": "run_shell",
        "arguments": {"command": "shell command", "cwd": "relative path, default .", "timeout": "seconds, default 30"},
        "description": "Run a shell command inside the workspace.",
    },
    {
        "name": "run_test",
        "arguments": {
            "command": "optional test command override; defaults to the configured test command",
            "cwd": "relative path, default .",
            "timeout": "seconds, default 1200",
        },
        "description": "Run the project's test command inside the workspace.",
    },
    {
        "name": "git_status",
        "arguments": {"path": "optional relative path filter"},
        "description": "Show git status for the workspace or a specific path.",
    },
    {
        "name": "git_diff",
        "arguments": {"path": "optional relative path filter", "cached": "bool, default false", "context": "int, default 3"},
        "description": "Show a git diff for working tree or staged changes. Leave cached unset for working-tree changes; use cached=true only when the user explicitly asks for staged diff.",
    },
    {
        "name": "git_commit",
        "arguments": {"message": "commit message", "add_all": "bool, default true"},
        "description": "Create a git commit for the current workspace changes.",
    },
    {
        "name": "run_agent",
        "arguments": {
            "prompt": "subtask prompt for the child agent",
            "model": "optional model override",
            "approval_mode": "optional ask|auto|read-only",
            "max_tool_rounds": "optional int, default inherited",
        },
        "description": "Start a nested agent for a scoped subtask and return its final answer.",
    },
]


def format_tool_help(tool_names: Iterable[str] | None = None) -> str:
    lines: list[str] = []
    allowed = set(tool_names) if tool_names is not None else None
    for tool in TOOL_DESCRIPTIONS:
        if allowed is not None and tool["name"] not in allowed:
            continue
        arg_text = ", ".join(f"{key}: {value}" for key, value in tool["arguments"].items())
        lines.append(f"- {tool['name']}({arg_text}) -> {tool['description']}")
    return "\n".join(lines)


def format_compact_tool_help(tool_names: Iterable[str] | None = None) -> str:
    signatures = {
        "list_files": "list_files(path='.',depth=4,limit=200)",
        "read_file": "read_file(path,start=1,end=200)",
        "search": "search(query,path='.',limit=100)",
        "file_search": "file_search(query,path='.',limit=100)",
        "fd_search": "fd_search(query,path='.',limit=100,kind='any')",
        "file_index_refresh": "file_index_refresh(path='.',limit=50000)",
        "everything_search": "everything_search(query,path='.',limit=100)",
        "search_symbols": "search_symbols(query,path='.',limit=50)",
        "code_outline": "code_outline(path,max_symbols=120)",
        "read_symbol": "read_symbol(path,symbol,context=2)",
        "inspect_library_source": "inspect_library_source(target,context=3,max_lines=160,include_disassembly=false)",
        "repo_index_search": "repo_index_search(query,path='.',limit=10)",
        "fts_search": "fts_search(query,path='.',limit=20,refresh=false)",
        "fts_refresh": "fts_refresh(path='.',limit=2000)",
        "indexed_search": "indexed_search(query,path='.',limit=100)",
        "repo_index_refresh": "repo_index_refresh(path='.',limit=1000)",
        "semgrep_scan": "semgrep_scan(pattern,path='.',lang?,limit=50)",
        "ast_search": "ast_search(pattern,path='.',lang?,limit=50)",
        "lsp_diagnostics": "lsp_diagnostics(path='.',lang?,timeout=60)",
        "lsp_definition": "lsp_definition(path,line,column,limit=20)",
        "lsp_references": "lsp_references(path,line,column,limit=40)",
        "context_pack": "context_pack(request,path='.',limit=8)",
        "systems_lens": "systems_lens(request,path='.',evidence?,limit=8)",
        "find_implementation_target": "find_implementation_target(test_path?/path?,query?,output?,limit=12)",
        "diagnose_test_failure": "diagnose_test_failure(output,path='.',limit=12)",
        "run_function_probe": "run_function_probe(module,expressions,function?,timeout=30)",
        "call_graph": "call_graph(path,symbol?,limit=40)",
        "contract_graph": "contract_graph(path='.',symbol?,limit=40)",
        "lint_typecheck": "lint_typecheck(paths='.',command?,timeout=120)",
        "discover_validators": "discover_validators(path='.',limit=12)",
        "diagnose_dependency_error": "diagnose_dependency_error(output,path='.')",
        "contract_check": "contract_check(changed_files,changed_symbols?,limit=80)",
        "select_tests": "select_tests(changed_files,changed_symbols?,limit=8)",
        "replace_symbol": "replace_symbol(path,symbol,content)",
        "replace_symbols": 'replace_symbols(path,replacements=[{"symbol":"f","content":"def f():\\n    return ..."}])',
        "write_file": "write_file(path,content)",
        "replace_in_file": "replace_in_file(path,old,new,all=false,whole_word=false)",
        "apply_structured_edit": 'apply_structured_edit(operation={"op":"rename_symbol|replace_function_body|change_signature|...","path":"a.py"})',
        "edit_intent": "edit_intent(path,intent=rename|replace_text|replace_symbol|replace_body|change_signature|add_import,target?,replacement?,scope='file',apply=true)",
        "generate_tests_from_spec": "generate_tests_from_spec(target_symbol,behavior,test_path?,apply=false)",
        "browser_smoke": "browser_smoke(url,actions=[],wait_for?,viewport='desktop',screenshot=false)",
        "security_scan": "security_scan(path='.',scanners='auto',limit=80)",
        "mcp_list_tools": "mcp_list_tools(server?,timeout=15)",
        "mcp_call": "mcp_call(server,tool,arguments={},timeout=30)",
        "run_shell": "run_shell(command,cwd='.',timeout=30)",
        "run_test": "run_test(command?,cwd='.',timeout=1200)",
        "git_status": "git_status(path?)",
        "git_diff": "git_diff(path?,cached=false,context=3)",
        "git_commit": "git_commit(message,add_all=true)",
        "run_agent": "run_agent(prompt,model?,approval?,rounds?)",
    }
    allowed = set(tool_names) if tool_names is not None else None
    return "\n".join(signatures[tool["name"]] for tool in TOOL_DESCRIPTIONS if allowed is None or tool["name"] in allowed)


class ToolExecutor:
    def __init__(
        self,
        workspace_root: str | Path,
        *,
        approval_mode: ApprovalMode = "ask",
        input_func: Callable[[str], str] = input,
        agent_runner: AgentRunner | None = None,
        test_command: str | None = None,
        default_tools_enabled: bool = True,
        disabled_tools: Iterable[str] | None = None,
        mcp_servers: dict[str, Any] | None = None,
        browser_enabled: bool = True,
        security_enabled: bool = True,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.approval_mode = approval_mode
        self.input_func = input_func
        self.agent_runner = agent_runner
        self.default_test_command = test_command.strip() if isinstance(test_command, str) and test_command.strip() else None
        self.default_tools_enabled = bool(default_tools_enabled)
        self.disabled_tools = {str(name).strip() for name in (disabled_tools or []) if str(name).strip()}
        self.mcp_servers = dict(mcp_servers or {})
        self.browser_enabled = bool(browser_enabled)
        self.security_enabled = bool(security_enabled)
        self._interrupt_event: threading.Event | None = None
        self._initial_dirty_paths = self._git_dirty_paths()

    def set_approval_mode(self, mode: ApprovalMode) -> None:
        self.approval_mode = mode

    def set_interrupt_event(self, event: threading.Event | None) -> None:
        self._interrupt_event = event

    def set_test_command(self, command: str | None) -> None:
        self.default_test_command = command.strip() if isinstance(command, str) and command.strip() else None

    def _truncate_text(self, text: str, *, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 18)] + "... truncated ..."

    def _call_tool_handler(self, handler: Callable[..., dict[str, Any]], arguments: dict[str, Any]) -> dict[str, Any]:
        signature = inspect.signature(handler)
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            return handler(**arguments)
        accepted = {
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        }
        filtered = {key: value for key, value in arguments.items() if key in accepted}
        return handler(**filtered)

    def is_tool_enabled(self, name: str) -> bool:
        clean = str(name or "").strip()
        if not clean:
            return False
        if clean in self.disabled_tools:
            return False
        if clean.startswith("mcp.") and ("mcp_call" in self.disabled_tools or "mcp" in self.disabled_tools):
            return False
        return self.default_tools_enabled

    def available_tool_names(self) -> set[str]:
        if not self.default_tools_enabled:
            return set()
        return {tool["name"] for tool in TOOL_DESCRIPTIONS if self.is_tool_enabled(str(tool["name"]))}

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.is_tool_enabled(name):
            return {"ok": False, "tool": name, "summary": f"Tool disabled by config: {name}"}
        if name.startswith("mcp."):
            parts = name.split(".", 2)
            if len(parts) != 3:
                return {"ok": False, "tool": name, "summary": "MCP dynamic tool names must be mcp.<server>.<tool>."}
            return self.mcp_call(parts[1], parts[2], arguments)
        if name == "run_agent":
            if self.agent_runner is None:
                return {"ok": False, "tool": name, "summary": "Sub-agent support is not configured."}
            try:
                return self.agent_runner(arguments)
            except OperationInterrupted:
                return {"ok": False, "tool": name, "summary": "Interrupted by user.", "interrupted": True}
            except TypeError as exc:
                return {"ok": False, "tool": name, "summary": f"Bad arguments for {name}: {exc}"}
            except Exception as exc:  # pragma: no cover - defensive fallback
                return {"ok": False, "tool": name, "summary": f"{name} failed: {exc}"}
        handlers = {
            "list_files": self.list_files,
            "read_file": self.read_file,
            "search": self.search,
            "file_search": self.file_search,
            "fd_search": self.fd_search,
            "file_index_refresh": self.file_index_refresh,
            "everything_search": self.everything_search,
            "search_symbols": self.search_symbols,
            "code_outline": self.code_outline,
            "read_symbol": self.read_symbol,
            "inspect_library_source": self.inspect_library_source,
            "repo_index_search": self.repo_index_search,
            "fts_search": self.fts_search,
            "fts_refresh": self.fts_refresh,
            "indexed_search": self.indexed_search,
            "repo_index_refresh": self.repo_index_refresh,
            "semgrep_scan": self.semgrep_scan,
            "ast_search": self.ast_search,
            "lsp_diagnostics": self.lsp_diagnostics,
            "lsp_definition": self.lsp_definition,
            "lsp_references": self.lsp_references,
            "context_pack": self.context_pack,
            "systems_lens": self.systems_lens,
            "find_implementation_target": self.find_implementation_target,
            "diagnose_test_failure": self.diagnose_test_failure,
            "run_function_probe": self.run_function_probe,
            "call_graph": self.call_graph,
            "contract_graph": self.contract_graph,
            "lint_typecheck": self.lint_typecheck,
            "discover_validators": self.discover_validators,
            "diagnose_dependency_error": self.diagnose_dependency_error,
            "contract_check": self.contract_check,
            "select_tests": self.select_tests,
            "edit_intent": self.edit_intent,
            "replace_symbol": self.replace_symbol,
            "replace_symbols": self.replace_symbols,
            "apply_structured_edit": self.apply_structured_edit,
            "generate_tests_from_spec": self.generate_tests_from_spec,
            "browser_smoke": self.browser_smoke,
            "security_scan": self.security_scan,
            "mcp_list_tools": self.mcp_list_tools,
            "mcp_call": self.mcp_call,
            "write_file": self.write_file,
            "replace_in_file": self.replace_in_file,
            "run_shell": self.run_shell,
            "run_test": self.run_test,
            "git_status": self.git_status,
            "git_diff": self.git_diff,
            "git_commit": self.git_commit,
        }
        handler = handlers.get(name)
        if handler is None:
            return {"ok": False, "tool": name, "summary": f"Unknown tool: {name}"}
        try:
            return self._call_tool_handler(handler, arguments)
        except OperationInterrupted:
            return {"ok": False, "tool": name, "summary": "Interrupted by user.", "interrupted": True}
        except TypeError as exc:
            return {"ok": False, "tool": name, "summary": f"Bad arguments for {name}: {exc}"}
        except Exception as exc:  # pragma: no cover - defensive fallback
            return self._tool_exception_result(name, arguments, exc)

    def _tool_exception_result(self, name: str, arguments: dict[str, Any], exc: Exception) -> dict[str, Any]:
        summary = f"{name} failed: {exc}"
        error_class = self.classify_error(summary)
        result: dict[str, Any] = {"ok": False, "tool": name, "summary": summary, "error_class": error_class}
        raw_path = ""
        if isinstance(arguments, dict):
            for key in ("path", "cwd", "test_path"):
                value = arguments.get(key)
                if isinstance(value, str) and value.strip():
                    raw_path = value
                    break
        if error_class in {"path_missing", "cwd_git"} and raw_path:
            result["suggested_paths"] = self._nearest_existing_paths(raw_path)
        return result

    def resolve_path(self, raw_path: str | None, *, allow_missing: bool = True) -> Path:
        candidate = self.workspace_root if not raw_path else self._coerce_input_path(raw_path)
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        resolved = candidate.resolve(strict=False)
        if resolved != self.workspace_root and self.workspace_root not in resolved.parents:
            raise ValueError(f"Path escapes the workspace: {raw_path}")
        if not allow_missing and not resolved.exists():
            raise FileNotFoundError(f"Path does not exist: {raw_path}")
        return resolved

    def _coerce_input_path(self, raw_path: str | Path) -> Path:
        text = str(raw_path).strip()
        candidate = Path(text)
        if candidate.is_absolute():
            return candidate
        normalized = text.replace("\\", "/")
        windows_match = WINDOWS_DRIVE_PATH.match(normalized)
        if windows_match:
            drive = windows_match.group("drive").lower()
            rest = (windows_match.group("rest") or "").strip("/")
            suffix = f"/{rest}" if rest else ""
            return Path(f"/mnt/{drive}{suffix}")
        wsl_match = WSL_MOUNT_PATH.match(normalized)
        if wsl_match:
            drive = wsl_match.group("drive").upper()
            rest = (wsl_match.group("rest") or "").strip("/")
            return Path(f"{drive}:/{rest}") if rest else Path(f"{drive}:/")
        if os.name != "nt" and "\\" in text:
            return Path(normalized)
        return candidate

    def relative_label(self, path: Path) -> str:
        return path.resolve().relative_to(self.workspace_root).as_posix() or "."

    def _confirm(self, prompt: str) -> bool:
        while True:
            reply = self.input_func(f"{prompt} [y/N]: ").strip().lower()
            if reply in {"y", "yes"}:
                return True
            if reply in {"", "n", "no"}:
                return False

    def _diff_preview(self, relative_path: str, old_text: str, new_text: str) -> str:
        diff_lines = list(
            difflib.unified_diff(
                old_text.splitlines(),
                new_text.splitlines(),
                fromfile=f"a/{relative_path}",
                tofile=f"b/{relative_path}",
                lineterm="",
            )
        )
        if not diff_lines:
            return "(no diff)"
        preview = diff_lines[:200]
        if len(diff_lines) > 200:
            preview.append("... diff truncated ...")
        return "\n".join(preview)

    def _approve_mutation(self, action: str, preview: str) -> tuple[bool, str | None]:
        if self.approval_mode == "read-only":
            return False, "Mutation denied because approval mode is read-only."
        if self.approval_mode == "auto":
            return True, None
        approved = self._confirm(f"{action}\n{preview}")
        if not approved:
            return False, "User rejected the mutation."
        return True, None

    def _approve_shell(self, command: str, cwd: str) -> tuple[bool, str | None]:
        if self.approval_mode == "read-only":
            return False, "Shell command denied because approval mode is read-only."
        if self.approval_mode == "auto":
            return True, None
        approved = self._confirm(f"Run shell command in {cwd}? {command}")
        if not approved:
            return False, "User rejected the shell command."
        return True, None

    def _windows_powershell(self) -> str | None:
        return shutil.which("pwsh") or shutil.which("powershell")

    def _looks_like_powershell_command(self, command: str) -> bool:
        statement_start = r"(?:^|[;&]\s*)"
        patterns = [
            rf"{statement_start}\$[A-Za-z_][\w:]*\s*=",
            rf"{statement_start}\.[\\/].+\.ps1(?:\s|$)",
            rf"{statement_start}(?:Get|Set|New|Remove|Move|Copy|Join|Split|Resolve|Test|Write|Start|Stop|Select|Where|ForEach|Measure|Sort)-[A-Za-z]+\b",
            r"\|\s*(?:Where-Object|Select-Object|ForEach-Object|Sort-Object|Measure-Object)\b",
        ]
        return any(re.search(pattern, command, flags=re.IGNORECASE) for pattern in patterns)

    def _collect_process_output(self, completed: subprocess.CompletedProcess[str]) -> str:
        parts = []
        if completed.stdout.strip():
            parts.append(completed.stdout.strip())
        if completed.stderr.strip():
            parts.append(completed.stderr.strip())
        return "\n".join(parts) if parts else "(no output)"

    def _check_interrupted(self) -> None:
        if self._interrupt_event is not None and self._interrupt_event.is_set():
            raise OperationInterrupted("Interrupted by user.")

    def _terminate_process(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                check=False,
            )
            if process.poll() is None:
                process.kill()
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=1)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return

    def _run_process(
        self,
        args: str | list[str],
        *,
        cwd: Path,
        timeout: int,
        shell: bool = False,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        self._check_interrupted()
        process = subprocess.Popen(
            args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=shell,
            start_new_session=True,
            **kwargs,
        )
        deadline = time.monotonic() + timeout
        while True:
            self._check_interrupted()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._terminate_process(process)
                raise subprocess.TimeoutExpired(process.args, timeout)
            try:
                stdout, stderr = process.communicate(timeout=min(0.1, remaining))
                return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                if self._interrupt_event is not None and self._interrupt_event.is_set():
                    self._terminate_process(process)
                    raise OperationInterrupted("Interrupted by user.")
                continue

    def _run_git(self, args: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
        if shutil.which("git") is None:
            raise RuntimeError("git is not installed.")
        return self._run_process(
            ["git", *args],
            cwd=self.workspace_root,
            timeout=timeout,
        )

    def _generated_dir_name(self, name: str) -> bool:
        lowered = name.lower()
        if lowered in SKIP_WALK_DIRS:
            return True
        if lowered.startswith(SKIP_GENERATED_DIR_PREFIXES):
            return True
        return any(pattern.match(lowered) for pattern in SKIP_GENERATED_DIR_PATTERNS)

    def _path_has_skipped_part(self, path: Path) -> bool:
        try:
            parts = path.resolve(strict=False).relative_to(self.workspace_root).parts
        except ValueError:
            parts = path.parts
        return any(self._generated_dir_name(part) for part in parts)

    def _repo_files_from_git(self, base: Path, *, limit: int, suffixes: set[str] | None = None) -> list[Path] | None:
        try:
            top_level = self._run_git(["rev-parse", "--show-toplevel"], timeout=10)
        except OperationInterrupted:
            raise
        except Exception:
            return None
        if top_level.returncode != 0 or not top_level.stdout.strip():
            return None
        git_root = Path(top_level.stdout.strip()).resolve(strict=False)
        try:
            pathspec = base.resolve(strict=False).relative_to(git_root).as_posix()
        except ValueError:
            return None
        if not pathspec:
            pathspec = "."
        try:
            completed = self._run_git(["ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", pathspec], timeout=20)
        except OperationInterrupted:
            raise
        except Exception:
            return None
        if completed.returncode != 0:
            return None
        files: list[Path] = []
        seen: set[str] = set()
        for raw_path in completed.stdout.split("\0"):
            self._check_interrupted()
            if not raw_path:
                continue
            file_path = (git_root / raw_path).resolve(strict=False)
            if suffixes is not None and file_path.suffix.lower() not in suffixes:
                continue
            if base.is_file() and file_path != base:
                continue
            if not base.is_file() and base != file_path and base not in file_path.parents:
                continue
            if file_path != self.workspace_root and self.workspace_root not in file_path.parents:
                continue
            if self._path_has_skipped_part(file_path):
                continue
            if not file_path.is_file():
                continue
            rel_file = self.relative_label(file_path)
            if rel_file in seen:
                continue
            seen.add(rel_file)
            files.append(file_path)
            if len(files) >= limit:
                break
        if not files and base.exists():
            try:
                has_entries = base.is_file() or any(base.iterdir())
            except OSError:
                has_entries = False
            if has_entries:
                return None
        return sorted(files, key=self.relative_label)

    def _walk_repo_files(self, base: Path, *, limit: int, suffixes: set[str] | None = None) -> list[Path]:
        if base.is_file():
            if self._path_has_skipped_part(base):
                return []
            if suffixes is not None and base.suffix.lower() not in suffixes:
                return []
            return [base]
        files: list[Path] = []
        for root, dirs, names in os.walk(base):
            self._check_interrupted()
            dirs[:] = sorted(directory for directory in dirs if not self._generated_dir_name(directory))
            root_path = Path(root)
            for name in sorted(names):
                if len(files) >= limit:
                    return files
                file_path = root_path / name
                if self._path_has_skipped_part(file_path):
                    continue
                if suffixes is not None and file_path.suffix.lower() not in suffixes:
                    continue
                if file_path.is_file():
                    files.append(file_path)
        return files

    def _iter_repo_files(self, base: Path, *, limit: int = 50000, suffixes: set[str] | None = None) -> list[Path]:
        limit = max(1, int(limit))
        if base.is_file():
            return self._walk_repo_files(base, limit=limit, suffixes=suffixes)
        try:
            git_files = self._repo_files_from_git(base, limit=limit, suffixes=suffixes)
        except OperationInterrupted:
            raise
        if git_files is not None:
            return git_files
        return self._walk_repo_files(base, limit=limit, suffixes=suffixes)

    def _ensure_git_repo(self) -> tuple[bool, str | None]:
        probe = self._run_git(["rev-parse", "--is-inside-work-tree"])
        if probe.returncode == 0 and probe.stdout.strip() == "true":
            return True, None
        return False, self._collect_process_output(probe)

    def _git_dirty_paths(self) -> set[str]:
        ok, _ = self._ensure_git_repo()
        if not ok:
            return set()
        paths: set[str] = set()
        commands = [
            ["diff", "--name-only", "-z"],
            ["diff", "--cached", "--name-only", "-z"],
            ["ls-files", "--others", "--exclude-standard", "-z"],
        ]
        for args in commands:
            completed = self._run_git(args)
            if completed.returncode != 0:
                continue
            for raw_path in completed.stdout.split("\0"):
                if raw_path:
                    paths.add(raw_path)
        return paths

    def list_files(self, path: str = ".", max_depth: int = 4, limit: int = 200) -> dict[str, Any]:
        self._check_interrupted()
        base = self.resolve_path(path, allow_missing=False)
        if base.is_file():
            items = [self.relative_label(base)]
        else:
            items = []
            for root, dirs, files in os.walk(base):
                self._check_interrupted()
                root_path = Path(root)
                depth = len(root_path.relative_to(base).parts)
                dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not self._generated_dir_name(d))
                if depth >= max_depth:
                    dirs[:] = []
                for directory in dirs:
                    items.append(f"{self.relative_label(root_path / directory)}/")
                    if len(items) >= limit:
                        break
                if len(items) >= limit:
                    break
                for file_name in sorted(files):
                    if file_name.startswith("."):
                        continue
                    if self._path_has_skipped_part(root_path / file_name):
                        continue
                    items.append(self.relative_label(root_path / file_name))
                    if len(items) >= limit:
                        break
                if len(items) >= limit:
                    break
        return {
            "ok": True,
            "tool": "list_files",
            "path": self.relative_label(base),
            "count": len(items),
            "output": "\n".join(items) if items else "(empty)",
        }

    def read_file(self, path: str, start: int = 1, end: int = 200) -> dict[str, Any]:
        self._check_interrupted()
        target = self.resolve_path(path, allow_missing=False)
        if target.is_dir():
            return {"ok": False, "tool": "read_file", "summary": f"{path} is a directory."}
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        safe_start = max(1, start)
        safe_end = max(safe_start, end)
        selected = lines[safe_start - 1 : safe_end]
        rendered = [f"{index:4d} | {line}" for index, line in enumerate(selected, start=safe_start)]
        return {
            "ok": True,
            "tool": "read_file",
            "path": self.relative_label(target),
            "start": safe_start,
            "end": safe_start + len(selected) - 1 if selected else safe_start,
            "output": "\n".join(rendered) if rendered else "(no content in requested range)",
        }

    def search(self, query: str, path: str = ".", limit: int = 100) -> dict[str, Any]:
        self._check_interrupted()
        base = self.resolve_path(path, allow_missing=False)
        if shutil.which("rg"):
            command = ["rg", "-n", "--color", "never", "--no-ignore-parent", "--max-count", str(limit)]
            for directory in sorted(SKIP_CODE_DIRS):
                command.extend(["--glob", f"!{directory}/**"])
            for glob in SKIP_WALK_GLOBS:
                command.extend(["--glob", glob])
            command.extend([query, str(base)])
            result = self._run_process(command, cwd=self.workspace_root, timeout=30)
            if result.returncode not in {0, 1} and "regex parse error" in (result.stderr or ""):
                literal_command = list(command)
                literal_command.insert(1, "-F")
                result = self._run_process(literal_command, cwd=self.workspace_root, timeout=30)
            output = result.stdout.strip() or result.stderr.strip() or "(no matches)"
            if output != "(no matches)":
                output = "\n".join(output.splitlines()[: max(1, limit)])
            return {
                "ok": result.returncode in {0, 1},
                "tool": "search",
                "path": self.relative_label(base),
                "output": output,
            }
        matches: list[str] = []
        try:
            pattern = re.compile(query)
            matcher = lambda text: bool(pattern.search(text))
        except re.error:
            matcher = lambda text: query in text
        for file_path in self._iter_workspace_files(base, limit=50000):
            self._check_interrupted()
            for line_no, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                self._check_interrupted()
                if matcher(line):
                    matches.append(f"{self.relative_label(file_path)}:{line_no}:{line}")
                    if len(matches) >= limit:
                        break
            if len(matches) >= limit:
                break
        return {
            "ok": True,
            "tool": "search",
            "path": self.relative_label(base),
            "output": "\n".join(matches) if matches else "(no matches)",
        }

    def _file_index_cache_path(self) -> Path:
        return self.workspace_root / ".ollama-code" / "index" / "file_index.json"

    def _load_file_index(self) -> dict[str, Any]:
        cache_path = self._file_index_cache_path()
        if not cache_path.exists():
            return {"version": FILE_INDEX_VERSION, "files": {}}
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": FILE_INDEX_VERSION, "files": {}}
        if not isinstance(payload, dict) or payload.get("version") != FILE_INDEX_VERSION or not isinstance(payload.get("files"), dict):
            return {"version": FILE_INDEX_VERSION, "files": {}}
        return payload

    def _write_file_index(self, payload: dict[str, Any]) -> None:
        cache_path = self._file_index_cache_path()
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload["version"] = FILE_INDEX_VERSION
            cache_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        except OSError:
            return

    def _iter_workspace_files(self, base: Path, *, limit: int = 50000) -> list[Path]:
        return self._iter_repo_files(base, limit=limit)

    def _file_index_record(self, file_path: Path) -> dict[str, Any]:
        stat = file_path.stat()
        rel = self.relative_label(file_path)
        return {
            "path": rel,
            "name": file_path.name,
            "suffix": file_path.suffix.lower(),
            "mtime_ns": int(stat.st_mtime_ns),
            "size": int(stat.st_size),
            "terms": self._extract_index_terms(rel.replace("/", " "), limit=80),
        }

    def _indexed_file_records(self, base: Path, *, limit: int = 50000) -> list[dict[str, Any]]:
        payload = self._load_file_index()
        files_payload = payload.setdefault("files", {})
        assert isinstance(files_payload, dict)
        changed = False
        seen: set[str] = set()
        records: list[dict[str, Any]] = []
        for file_path in self._iter_workspace_files(base, limit=limit):
            self._check_interrupted()
            rel = self.relative_label(file_path)
            seen.add(rel)
            stat = file_path.stat()
            cached = files_payload.get(rel)
            if not (
                isinstance(cached, dict)
                and cached.get("mtime_ns") == int(stat.st_mtime_ns)
                and cached.get("size") == int(stat.st_size)
            ):
                cached = self._file_index_record(file_path)
                files_payload[rel] = cached
                changed = True
            records.append(cached)
        for rel in list(files_payload):
            path = self.workspace_root / rel
            if rel not in seen and (base == self.workspace_root or path == base or base in path.parents):
                files_payload.pop(rel, None)
                changed = True
        if changed:
            self._write_file_index(payload)
        return records

    def file_index_refresh(self, path: str = ".", limit: int = 50000) -> dict[str, Any]:
        self._check_interrupted()
        base = self.resolve_path(path, allow_missing=False)
        records = self._indexed_file_records(base, limit=max(1, int(limit)))
        return {
            "ok": True,
            "tool": "file_index_refresh",
            "path": self.relative_label(base),
            "files": len(records),
            "cache": self.relative_label(self._file_index_cache_path()),
            "output": f"Indexed {len(records)} file path(s).",
        }

    def file_search(self, query: str, path: str = ".", limit: int = 100) -> dict[str, Any]:
        self._check_interrupted()
        base = self.resolve_path(path, allow_missing=False)
        clean_query = str(query or "").strip()
        terms = self._extract_index_terms(clean_query, limit=20)
        if not terms:
            return {"ok": False, "tool": "file_search", "summary": "file_search requires a non-empty query."}
        query_lower = clean_query.lower()
        rows: list[tuple[int, str]] = []
        for record in self._indexed_file_records(base, limit=50000):
            rel = str(record.get("path", ""))
            rel_lower = rel.lower()
            name_lower = str(record.get("name", "")).lower()
            record_terms = set(str(term).lower() for term in record.get("terms", []) if isinstance(term, str))
            score = 0
            if query_lower and query_lower in rel_lower:
                score += 30
            if query_lower and query_lower in name_lower:
                score += 40
            for term in terms:
                if term in record_terms:
                    score += 10
                elif term in rel_lower:
                    score += 5
                if term in name_lower:
                    score += 8
            if score:
                rows.append((score, rel))
        ranked = [line for _, line in sorted(rows, key=lambda item: (-item[0], item[1]))[: max(1, int(limit))]]
        return {
            "ok": True,
            "tool": "file_search",
            "path": self.relative_label(base),
            "count": len(ranked),
            "output": "\n".join(ranked) if ranked else "(no file matches)",
        }

    def _fd_cli_path(self) -> str | None:
        configured = os.environ.get("FD_CLI", "").strip()
        candidates = [configured] if configured else []
        found = shutil.which("fd") or shutil.which("fd.exe") or shutil.which("fdfind")
        if found:
            candidates.append(found)
        if os.name == "nt":
            candidates.extend(
                [
                    str(Path.home() / "scoop" / "shims" / "fd.exe"),
                    r"C:\ProgramData\chocolatey\bin\fd.exe",
                ]
            )
        for candidate in candidates:
            if candidate and (Path(candidate).exists() or shutil.which(candidate)):
                return candidate
        return None

    def fd_search(self, query: str, path: str = ".", limit: int = 100, kind: str = "any") -> dict[str, Any]:
        self._check_interrupted()
        base = self.resolve_path(path, allow_missing=False)
        clean_query = str(query or "").strip()
        if not clean_query:
            return {"ok": False, "tool": "fd_search", "summary": "fd_search requires a non-empty query."}
        fd_path = self._fd_cli_path()
        if not fd_path:
            return self._missing_dependency_result("fd_search", "fd", "fd/fdfind is not installed or not on PATH. Install fd or use file_search.")
        clean_kind = str(kind or "any").strip().lower()
        command = [fd_path, "--color", "never", "--strip-cwd-prefix"]
        if clean_kind in {"file", "f"}:
            command.extend(["--type", "file"])
        elif clean_kind in {"dir", "directory", "d"}:
            command.extend(["--type", "directory"])
        command.extend([clean_query, str(base)])
        completed = self._run_process(command, cwd=self.workspace_root, timeout=30, shell=False)
        output = self._collect_process_output(completed)
        matches: list[str] = []
        base_resolved = base.resolve()
        for line in (completed.stdout or "").splitlines():
            raw = line.strip().strip('"')
            if not raw:
                continue
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = self.workspace_root / candidate
            try:
                resolved = candidate.resolve(strict=False)
            except OSError:
                continue
            if resolved != base_resolved and base_resolved not in resolved.parents:
                continue
            try:
                matches.append(self.relative_label(resolved))
            except ValueError:
                continue
            if len(matches) >= max(1, int(limit)):
                break
        return {
            "ok": completed.returncode in {0, 1},
            "tool": "fd_search",
            "path": self.relative_label(base),
            "count": len(matches),
            "output": "\n".join(matches) if matches else ("(no fd matches)" if completed.returncode in {0, 1} else output),
        }

    def _everything_cli_path(self) -> str | None:
        configured = os.environ.get("EVERYTHING_CLI", "").strip()
        candidates = [configured] if configured else []
        found = shutil.which("es.exe") or shutil.which("es")
        if found:
            candidates.append(found)
        if os.name == "nt":
            candidates.extend(
                [
                    r"C:\Program Files\Everything\es.exe",
                    r"C:\Program Files (x86)\Everything\es.exe",
                    str(Path.home() / "scoop" / "shims" / "es.exe"),
                    r"C:\ProgramData\chocolatey\bin\es.exe",
                ]
            )
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return candidate
        return None

    def everything_search(self, query: str, path: str = ".", limit: int = 100) -> dict[str, Any]:
        self._check_interrupted()
        base = self.resolve_path(path, allow_missing=False)
        clean_query = str(query or "").strip()
        if not clean_query:
            return {"ok": False, "tool": "everything_search", "summary": "everything_search requires a non-empty query."}
        es_path = self._everything_cli_path()
        if not es_path:
            return {
                "ok": False,
                "tool": "everything_search",
                "summary": "Everything CLI es.exe is not installed or not on PATH. Install es.exe or use file_search.",
            }
        command = [es_path, "-n", str(max(1, int(limit)) * 5), clean_query]
        completed = self._run_process(command, cwd=self.workspace_root, timeout=30, shell=False)
        output = self._collect_process_output(completed)
        matches: list[str] = []
        base_resolved = base.resolve()
        for line in (completed.stdout or "").splitlines():
            raw = line.strip().strip('"')
            if not raw:
                continue
            try:
                candidate = Path(raw).resolve()
            except OSError:
                continue
            if candidate == base_resolved or base_resolved in candidate.parents:
                try:
                    matches.append(self.relative_label(candidate))
                except ValueError:
                    continue
            if len(matches) >= max(1, int(limit)):
                break
        return {
            "ok": completed.returncode in {0, 1},
            "tool": "everything_search",
            "path": self.relative_label(base),
            "count": len(matches),
            "output": "\n".join(matches) if matches else ("(no Everything matches)" if completed.returncode in {0, 1} else output),
        }

    def _is_code_file(self, path: Path) -> bool:
        return path.suffix.lower() in CODE_FILE_SUFFIXES and not self._path_has_skipped_part(path)

    def _iter_code_files(self, base: Path, *, limit: int = 200) -> list[Path]:
        return self._iter_repo_files(base, limit=limit, suffixes=CODE_FILE_SUFFIXES)

    def _python_signature(self, lines: list[str], start: int) -> str:
        collected: list[str] = []
        paren_balance = 0
        for line in lines[start - 1 : min(len(lines), start + 6)]:
            stripped = line.strip()
            collected.append(stripped)
            paren_balance += stripped.count("(") - stripped.count(")")
            if stripped.endswith(":") and paren_balance <= 0:
                break
        signature = " ".join(collected)
        return signature[:180] + ("..." if len(signature) > 180 else "")

    def _python_symbols(self, target: Path, text: str) -> list[dict[str, Any]]:
        lines = text.splitlines()
        try:
            tree = ast.parse(self._python_parse_text(text))
        except SyntaxError:
            return []
        symbols: list[dict[str, Any]] = []

        def visit(node: ast.AST, stack: list[str]) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    kind = "class" if isinstance(child, ast.ClassDef) else ("method" if stack and stack[-1][:1].isupper() else "function")
                    qualname = ".".join([*stack, child.name])
                    symbols.append(
                        {
                            "name": child.name,
                            "qualname": qualname,
                            "kind": kind,
                            "start": int(getattr(child, "lineno", 1)),
                            "end": int(getattr(child, "end_lineno", getattr(child, "lineno", 1))),
                            "signature": self._python_signature(lines, int(getattr(child, "lineno", 1))),
                            "doc": (ast.get_docstring(child) or "").strip().splitlines()[0][:120] if ast.get_docstring(child) else "",
                        }
                    )
                    visit(child, [*stack, child.name])
                else:
                    visit(child, stack)
        visit(tree, [])
        return symbols

    def _generic_symbol_end(self, lines: list[str], start_index: int, indent: int) -> int:
        first = lines[start_index]
        if "{" in first:
            balance = 0
            for index in range(start_index, min(len(lines), start_index + 220)):
                balance += lines[index].count("{") - lines[index].count("}")
                if index > start_index and balance <= 0:
                    return index + 1
        for index in range(start_index + 1, min(len(lines), start_index + 220)):
            line = lines[index]
            if not line.strip():
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent and re.match(r"\s*(?:def|class|function|async function|export |const |let |var )", line):
                return index
        return min(len(lines), start_index + 80)

    def _generic_symbols(self, target: Path, text: str) -> list[dict[str, Any]]:
        patterns = [
            ("class", re.compile(r"^\s*(?:export\s+)?class\s+(?P<name>[A-Za-z_][\w]*)")),
            ("function", re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(?P<name>[A-Za-z_][\w]*)")),
            ("function", re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_][\w]*)\s*=\s*(?:async\s*)?\(")),
            ("function", re.compile(r"^\s*def\s+(?P<name>[A-Za-z_][\w]*)\s*\(")),
            ("class", re.compile(r"^\s*class\s+(?P<name>[A-Za-z_][\w]*)")),
        ]
        lines = text.splitlines()
        symbols: list[dict[str, Any]] = []
        for index, line in enumerate(lines):
            for kind, pattern in patterns:
                match = pattern.search(line)
                if not match:
                    continue
                indent = len(line) - len(line.lstrip())
                name = match.group("name")
                symbols.append(
                    {
                        "name": name,
                        "qualname": name,
                        "kind": kind,
                        "start": index + 1,
                        "end": self._generic_symbol_end(lines, index, indent),
                        "signature": line.strip()[:180],
                        "doc": "",
                    }
                )
                break
        return symbols

    def _code_symbols(self, target: Path) -> tuple[list[dict[str, Any]], str, str | None]:
        text = target.read_text(encoding="utf-8", errors="replace")
        if target.suffix.lower() == ".py":
            symbols = self._python_symbols(target, text)
            if symbols:
                return symbols, text, None
        symbols = self._generic_symbols(target, text)
        return symbols, text, None

    def search_symbols(self, query: str, path: str = ".", limit: int = 50) -> dict[str, Any]:
        self._check_interrupted()
        base = self.resolve_path(path, allow_missing=False)
        try:
            pattern = re.compile(query, flags=re.IGNORECASE)
            matcher = lambda text: bool(pattern.search(text))
        except re.error:
            lowered = query.lower()
            matcher = lambda text: lowered in text.lower()
        matches: list[str] = []
        for file_path in self._iter_code_files(base):
            symbols, _, _ = self._code_symbols(file_path)
            for symbol in symbols:
                haystack = " ".join(
                    str(symbol.get(key, ""))
                    for key in ("name", "qualname", "kind", "signature")
                )
                if not matcher(haystack):
                    continue
                matches.append(
                    f"{self.relative_label(file_path)}:{symbol['start']}-{symbol['end']} {symbol['kind']} {symbol['qualname']} {symbol['signature']}"
                )
                if len(matches) >= limit:
                    return {"ok": True, "tool": "search_symbols", "path": self.relative_label(base), "count": len(matches), "output": "\n".join(matches)}
        return {"ok": True, "tool": "search_symbols", "path": self.relative_label(base), "count": len(matches), "output": "\n".join(matches) if matches else "(no symbols found)"}

    def code_outline(self, path: str = ".", max_symbols: int = 120) -> dict[str, Any]:
        self._check_interrupted()
        base = self.resolve_path(path, allow_missing=False)
        files = self._iter_code_files(base, limit=80)
        output: list[str] = []
        count = 0
        for file_path in files:
            symbols, text, _ = self._code_symbols(file_path)
            imports: list[str] = []
            if file_path.suffix.lower() == ".py":
                for line in text.splitlines()[:80]:
                    stripped = line.strip()
                    if stripped.startswith(("import ", "from ")):
                        imports.append(stripped)
                    if len(imports) >= 8:
                        break
            output.append(f"{self.relative_label(file_path)}")
            if imports:
                output.append("  imports: " + "; ".join(imports))
            for symbol in symbols:
                output.append(
                    f"  {symbol['start']}-{symbol['end']} {symbol['kind']} {symbol['qualname']}: {symbol['signature']}"
                )
                count += 1
                if count >= max_symbols:
                    output.append("  ... symbols truncated ...")
                    return {"ok": True, "tool": "code_outline", "path": self.relative_label(base), "count": count, "output": "\n".join(output)}
            if not symbols:
                output.append("  (no symbols found)")
        return {"ok": True, "tool": "code_outline", "path": self.relative_label(base), "count": count, "output": "\n".join(output) if output else "(no code files found)"}

    def _symbol_matches(self, symbols: list[dict[str, Any]], symbol: str) -> list[dict[str, Any]]:
        needle = self._normalize_symbol_query(symbol)
        exact = [item for item in symbols if item["qualname"] == needle or item["name"] == needle]
        if exact:
            return exact
        lowered = needle.lower()
        return [item for item in symbols if lowered in str(item["qualname"]).lower() or lowered in str(item["name"]).lower()]

    def _normalize_symbol_query(self, symbol: str) -> str:
        needle = symbol.strip()
        needle = re.sub(r"^(?:async\s+def|def|class)\s+", "", needle)
        needle = needle.split(":", 1)[0].strip()
        match = re.match(r"(?P<name>[A-Za-z_][\w.]*)(?:\s*\(|$)", needle)
        return match.group("name") if match else needle

    def read_symbol(self, path: str, symbol: str, include_context: int = 2) -> dict[str, Any]:
        self._check_interrupted()
        target = self.resolve_path(path, allow_missing=False)
        if target.is_dir():
            return {"ok": False, "tool": "read_symbol", "summary": f"{path} is a directory."}
        if not self._is_code_file(target):
            return {"ok": False, "tool": "read_symbol", "summary": f"{path} is not a supported code file."}
        symbols, text, _ = self._code_symbols(target)
        exact = self._symbol_matches(symbols, symbol)
        if not exact:
            return {"ok": False, "tool": "read_symbol", "path": self.relative_label(target), "summary": f"Symbol not found: {symbol}"}
        if len(exact) > 1:
            matches = "\n".join(f"{item['start']}-{item['end']} {item['kind']} {item['qualname']}" for item in exact[:20])
            return {"ok": False, "tool": "read_symbol", "path": self.relative_label(target), "summary": f"Ambiguous symbol: {symbol}", "matches": matches}
        found = exact[0]
        lines = text.splitlines()
        context = max(0, int(include_context))
        start = max(1, int(found["start"]) - context)
        end = min(len(lines), int(found["end"]) + context)
        selected = lines[start - 1 : end]
        rendered = [f"{index:4d} | {line}" for index, line in enumerate(selected, start=start)]
        return {
            "ok": True,
            "tool": "read_symbol",
            "path": self.relative_label(target),
            "symbol": found["qualname"],
            "kind": found["kind"],
            "start": start,
            "end": end,
            "output": "\n".join(rendered) if rendered else "(empty symbol)",
        }

    def _safe_signature(self, obj: object) -> str:
        try:
            return str(inspect.signature(obj))
        except (TypeError, ValueError):
            return ""

    def _library_object_kind(self, obj: object) -> str:
        if inspect.ismodule(obj):
            return "module"
        if inspect.isclass(obj):
            return "class"
        if inspect.isfunction(obj):
            return "function"
        if inspect.ismethod(obj):
            return "method"
        if inspect.isbuiltin(obj):
            return "builtin"
        if inspect.ismethoddescriptor(obj):
            return "method_descriptor"
        if inspect.isroutine(obj):
            return "routine"
        return type(obj).__name__

    def _resolve_library_target(self, target: str) -> tuple[object | None, str, str, str | None]:
        cleaned = target.strip()
        if not cleaned:
            return None, "", "", "target must be a non-empty Python import path."
        if ":" in cleaned:
            module_name, qualname = cleaned.split(":", 1)
            module_name = module_name.strip()
            qualname = qualname.strip(". ")
            if not module_name:
                return None, "", "", f"Missing module in target: {target}"
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError as exc:
                return None, module_name, qualname, f"Could not import {module_name}: {exc}"
            except Exception as exc:
                return None, module_name, qualname, f"Importing {module_name} failed: {exc}"
            return module, module_name, qualname, None

        parts = [part for part in cleaned.split(".") if part]
        import_errors: list[str] = []
        for index in range(len(parts), 0, -1):
            module_name = ".".join(parts[:index])
            qualname = ".".join(parts[index:])
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError as exc:
                import_errors.append(f"{module_name}: {exc}")
                continue
            except Exception as exc:
                return None, module_name, qualname, f"Importing {module_name} failed: {exc}"
            return module, module_name, qualname, None
        detail = import_errors[0] if import_errors else cleaned
        return None, parts[0] if parts else cleaned, ".".join(parts[1:]), f"Could not import any module prefix for {cleaned}: {detail}"

    def _resolve_qualname(self, obj: object, qualname: str) -> tuple[object | None, str | None, str]:
        current = obj
        if not qualname:
            return current, None, ""
        resolved_parts: list[str] = []
        for part in qualname.split("."):
            if not part:
                continue
            if not hasattr(current, part):
                public = [name for name in dir(current) if not name.startswith("_")]
                close = difflib.get_close_matches(part, public, n=8)
                hint = f" Close matches: {', '.join(close)}." if close else ""
                return None, part, f"Attribute not found: {part}.{hint}"
            current = getattr(current, part)
            resolved_parts.append(part)
        return current, None, ".".join(resolved_parts)

    def _object_source_file(self, obj: object) -> str:
        try:
            source = inspect.getsourcefile(obj)
        except (TypeError, OSError):
            source = None
        if source:
            return source
        try:
            return inspect.getfile(obj)
        except (TypeError, OSError):
            return ""

    def _disassemble_object(self, obj: object, *, max_lines: int) -> str:
        try:
            buffer = io.StringIO()
            dis.dis(obj, file=buffer)
            lines = buffer.getvalue().splitlines()
        except (TypeError, AttributeError):
            return ""
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"... disassembly truncated at {max_lines} lines ..."]
        return "\n".join(lines)

    def inspect_library_source(
        self,
        target: str,
        context: int = 3,
        max_lines: int = 160,
        include_disassembly: bool = False,
    ) -> dict[str, Any]:
        self._check_interrupted()
        if not isinstance(target, str) or not target.strip():
            return {"ok": False, "tool": "inspect_library_source", "summary": "target must be a non-empty Python import path.", "error_class": "invalid_args"}
        context = max(0, min(50, int(context)))
        max_lines = max(20, min(400, int(max_lines)))
        root_obj, module_name, qualname, error = self._resolve_library_target(target)
        if error is not None or root_obj is None:
            missing = module_name.split(".", 1)[0] if module_name else target.split(".", 1)[0]
            return {
                "ok": False,
                "tool": "inspect_library_source",
                "target": target,
                "summary": error or f"Could not import target: {target}",
                "error_class": "missing_dependency",
                "missing_dependency": missing,
            }
        obj, missing_attr, resolved_qualname = self._resolve_qualname(root_obj, qualname)
        if obj is None:
            return {
                "ok": False,
                "tool": "inspect_library_source",
                "target": target,
                "module": module_name,
                "summary": f"{module_name}:{qualname} could not be resolved. {missing_attr or ''}".strip(),
                "output": resolved_qualname,
                "error_class": "invalid_args",
            }

        kind = self._library_object_kind(obj)
        signature = self._safe_signature(obj)
        doc = inspect.getdoc(obj) or ""
        doc_preview = "\n".join(doc.splitlines()[:8])
        source_file = self._object_source_file(obj)
        output: list[str] = [
            f"target: {target}",
            f"resolved: {module_name}{':' + resolved_qualname if resolved_qualname else ''}",
            f"kind: {kind}",
        ]
        if signature:
            output.append(f"signature: {signature}")
        if source_file:
            output.append(f"file: {source_file}")

        source_available = False
        line_start: int | None = None
        line_end: int | None = None
        try:
            source_lines, source_start = inspect.getsourcelines(obj)
            line_start = int(source_start)
            line_end = line_start + len(source_lines) - 1
            selected_start = line_start
            selected_lines = [line.rstrip("\n") for line in source_lines]
            if source_file and source_file.endswith(".py"):
                try:
                    file_lines = Path(source_file).read_text(encoding="utf-8", errors="replace").splitlines()
                    selected_start = max(1, line_start - context)
                    selected_end = min(len(file_lines), line_end + context)
                    selected_lines = file_lines[selected_start - 1 : selected_end]
                except OSError:
                    selected_start = line_start
            if len(selected_lines) > max_lines:
                selected_lines = selected_lines[:max_lines] + [f"... source truncated at {max_lines} lines ..."]
            output.append(f"lines: {line_start}-{line_end}")
            output.extend(f"{line_no:4d} | {line}" for line_no, line in enumerate(selected_lines, start=selected_start))
            source_available = True
        except (OSError, TypeError):
            output.append("source: unavailable")

        disassembly = self._disassemble_object(obj, max_lines=max_lines) if include_disassembly or not source_available else ""
        if disassembly:
            output.append("disassembly:")
            output.extend(disassembly.splitlines())
        if doc_preview:
            output.append("doc:")
            output.append(doc_preview)

        summary = "Source found." if source_available else "Source unavailable; returned signature/doc/disassembly diagnostics where possible."
        return {
            "ok": True,
            "tool": "inspect_library_source",
            "target": target,
            "module": module_name,
            "qualname": resolved_qualname,
            "kind": kind,
            "signature": signature,
            "file": source_file,
            "line_start": line_start,
            "line_end": line_end,
            "source_available": source_available,
            "summary": summary,
            "output": "\n".join(output),
        }

    def _repo_index_cache_path(self) -> Path:
        return self.workspace_root / ".ollama-code" / "index" / "repo_index.json"

    def _fts_cache_path(self) -> Path:
        return self.workspace_root / ".ollama-code" / "index" / "repo_fts.sqlite"

    def _connect_fts(self) -> sqlite3.Connection:
        cache_path = self._fts_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(cache_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS repo_fts USING fts5("
                "path UNINDEXED, path_text, symbols, headings, text, mtime_ns UNINDEXED, size UNINDEXED)"
            )
            conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('version', ?)", (str(FTS_INDEX_VERSION),))
            conn.commit()
        except sqlite3.OperationalError:
            conn.close()
            raise
        return conn

    def _iter_fts_files(self, base: Path, *, limit: int = 2000) -> list[Path]:
        files: list[Path] = []
        for file_path in self._iter_workspace_files(base, limit=limit * 3):
            self._check_interrupted()
            if len(files) >= limit:
                break
            if file_path.suffix.lower() not in FTS_TEXT_SUFFIXES:
                continue
            try:
                stat = file_path.stat()
            except OSError:
                continue
            if stat.st_size > 512_000:
                continue
            files.append(file_path)
        return files

    def _fts_headings(self, file_path: Path, text: str) -> str:
        headings: list[str] = []
        for line in text.splitlines()[:1000]:
            stripped = line.strip()
            if not stripped:
                continue
            if file_path.suffix.lower() in {".md", ".rst"} and stripped.startswith(("#", "##", "###")):
                headings.append(stripped.lstrip("#").strip())
            elif re.match(r"^\[[A-Za-z0-9_.-]+\]$", stripped):
                headings.append(stripped.strip("[]"))
            if len(headings) >= 80:
                break
        return " ".join(headings)

    def _fts_record(self, file_path: Path) -> tuple[str, str, str, str, str, int, int]:
        stat = file_path.stat()
        rel = self.relative_label(file_path)
        text = file_path.read_text(encoding="utf-8", errors="replace")
        if "\x00" in text[:4096]:
            text = ""
        symbols = ""
        if file_path.suffix.lower() in CODE_FILE_SUFFIXES:
            try:
                symbol_rows, _, _ = self._code_symbols(file_path)
                symbols = " ".join(
                    f"{item.get('kind', '')} {item.get('qualname', '')} {item.get('signature', '')} {item.get('doc', '')}"
                    for item in symbol_rows[:300]
                    if isinstance(item, dict)
                )
            except OSError:
                symbols = ""
        return (
            rel,
            rel.replace("/", " "),
            symbols,
            self._fts_headings(file_path, text),
            text[:200_000],
            int(stat.st_mtime_ns),
            int(stat.st_size),
        )

    def _fts_scope(self, base: Path) -> tuple[str, tuple[Any, ...]]:
        rel = self.relative_label(base)
        if rel == ".":
            return "", ()
        if base.is_file():
            return " AND path = ?", (rel,)
        return " AND (path = ? OR path LIKE ?)", (rel, f"{rel}/%")

    def _safe_fts_query(self, query: str) -> str | None:
        terms = self._extract_index_terms(query, limit=16)
        if not terms:
            return None
        return " OR ".join(f"{term}*" for term in terms)

    def fts_refresh(self, path: str = ".", limit: int = 2000) -> dict[str, Any]:
        self._check_interrupted()
        base = self.resolve_path(path, allow_missing=False)
        try:
            conn = self._connect_fts()
        except sqlite3.OperationalError as exc:
            return self._missing_dependency_result("fts_refresh", "sqlite-fts5", f"SQLite FTS5 is unavailable: {exc}")
        files = self._iter_fts_files(base, limit=max(1, int(limit)))
        scope_sql, scope_params = self._fts_scope(base)
        try:
            if scope_sql:
                conn.execute("DELETE FROM repo_fts WHERE " + scope_sql.removeprefix(" AND "), scope_params)
            else:
                conn.execute("DELETE FROM repo_fts")
            for file_path in files:
                conn.execute(
                    "INSERT INTO repo_fts(path,path_text,symbols,headings,text,mtime_ns,size) VALUES(?,?,?,?,?,?,?)",
                    self._fts_record(file_path),
                )
            conn.commit()
        finally:
            conn.close()
        return {
            "ok": True,
            "tool": "fts_refresh",
            "path": self.relative_label(base),
            "files": len(files),
            "cache": self.relative_label(self._fts_cache_path()),
            "output": f"Indexed {len(files)} file(s) into SQLite FTS.",
        }

    def fts_search(self, query: str, path: str = ".", limit: int = 20, refresh: bool = False) -> dict[str, Any]:
        self._check_interrupted()
        base = self.resolve_path(path, allow_missing=False)
        clean_query = str(query or "").strip()
        fts_query = self._safe_fts_query(clean_query)
        if not fts_query:
            return {"ok": False, "tool": "fts_search", "summary": "fts_search requires a non-empty query."}
        if refresh or not self._fts_cache_path().exists():
            refresh_result = self.fts_refresh(self.relative_label(base), limit=2000)
            if refresh_result.get("ok") is not True:
                return refresh_result
        try:
            conn = self._connect_fts()
        except sqlite3.OperationalError as exc:
            return self._missing_dependency_result("fts_search", "sqlite-fts5", f"SQLite FTS5 is unavailable: {exc}")
        scope_sql, scope_params = self._fts_scope(base)
        limit_value = max(1, int(limit))
        rows: list[str] = []
        try:
            query_sql = (
                "SELECT path, symbols, headings, snippet(repo_fts, 4, '[', ']', ' ... ', 18), bm25(repo_fts) "
                "FROM repo_fts WHERE repo_fts MATCH ?"
                + scope_sql
                + " ORDER BY 5 LIMIT ?"
            )
            records = conn.execute(query_sql, (fts_query, *scope_params, limit_value)).fetchall()
        except sqlite3.OperationalError:
            quoted = " OR ".join(f'"{term}"' for term in self._extract_index_terms(clean_query, limit=16))
            records = conn.execute(query_sql, (quoted, *scope_params, limit_value)).fetchall() if quoted else []
        finally:
            conn.close()
        for rel, symbols, headings, snippet, rank in records:
            context = str(snippet or "").strip().replace("\n", " ")
            extras = []
            if headings:
                extras.append("headings=" + str(headings)[:120])
            if symbols:
                extras.append("symbols=" + str(symbols)[:160])
            suffix = " " + " ".join(extras) if extras else ""
            rows.append(f"{rel}: {context[:240]}{suffix}".strip())
        return {
            "ok": True,
            "tool": "fts_search",
            "path": self.relative_label(base),
            "count": len(rows),
            "cache": self.relative_label(self._fts_cache_path()),
            "output": "\n".join(rows) if rows else "(no FTS matches)",
        }

    def _extract_index_terms(self, text: str, *, limit: int = 900) -> list[str]:
        terms = sorted(set(term.lower() for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}|[0-9]+", text)))
        return terms[:limit]

    def _python_imports_for_index(self, target: Path, text: str) -> list[str]:
        if target.suffix.lower() != ".py":
            return []
        try:
            tree = ast.parse(self._python_parse_text(text), filename=self.relative_label(target))
        except SyntaxError:
            return []
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.extend(f"{node.module}.{alias.name}" for alias in node.names)
        return sorted(set(imports))[:120]

    def _code_index_record(self, file_path: Path) -> dict[str, Any]:
        stat = file_path.stat()
        text = file_path.read_text(encoding="utf-8", errors="replace")
        symbols, _, _ = self._code_symbols(file_path)
        line_index: list[dict[str, Any]] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            terms = self._extract_index_terms(stripped, limit=40)
            if not terms:
                continue
            line_index.append({"line": line_no, "text": stripped[:220], "terms": terms})
            if len(line_index) >= 800:
                break
        return {
            "path": self.relative_label(file_path),
            "mtime_ns": int(stat.st_mtime_ns),
            "size": int(stat.st_size),
            "sha1": hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest(),
            "symbols": symbols[:300],
            "imports": self._python_imports_for_index(file_path, text),
            "terms": self._extract_index_terms(text),
            "line_index": line_index,
        }

    def _load_repo_index(self) -> dict[str, Any]:
        cache_path = self._repo_index_cache_path()
        if not cache_path.exists():
            return {"version": REPO_INDEX_VERSION, "files": {}}
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": REPO_INDEX_VERSION, "files": {}}
        if not isinstance(payload, dict) or payload.get("version") != REPO_INDEX_VERSION or not isinstance(payload.get("files"), dict):
            return {"version": REPO_INDEX_VERSION, "files": {}}
        return payload

    def _write_repo_index(self, payload: dict[str, Any]) -> None:
        cache_path = self._repo_index_cache_path()
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload["version"] = REPO_INDEX_VERSION
            cache_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        except OSError:
            return

    def _indexed_code_records(self, base: Path, *, limit: int = 1000) -> list[dict[str, Any]]:
        payload = self._load_repo_index()
        files_payload = payload.setdefault("files", {})
        assert isinstance(files_payload, dict)
        changed = False
        seen: set[str] = set()
        records: list[dict[str, Any]] = []
        for file_path in self._iter_code_files(base, limit=limit):
            self._check_interrupted()
            rel = self.relative_label(file_path)
            seen.add(rel)
            stat = file_path.stat()
            cached = files_payload.get(rel)
            if not (
                isinstance(cached, dict)
                and cached.get("mtime_ns") == int(stat.st_mtime_ns)
                and cached.get("size") == int(stat.st_size)
                and isinstance(cached.get("line_index"), list)
            ):
                cached = self._code_index_record(file_path)
                files_payload[rel] = cached
                changed = True
            records.append(cached)
        for rel in list(files_payload):
            path = self.workspace_root / rel
            if rel not in seen and (base == self.workspace_root or path == base or base in path.parents):
                files_payload.pop(rel, None)
                changed = True
        if changed:
            self._write_repo_index(payload)
        return records

    def _repo_index_score(self, record: dict[str, Any], terms: list[str]) -> tuple[int, list[str]]:
        score = 0
        rel = str(record.get("path", ""))
        path_text = rel.lower()
        record_terms = set(str(term).lower() for term in record.get("terms", []) if isinstance(term, str))
        imports = " ".join(str(item).lower() for item in record.get("imports", []) if isinstance(item, str))
        matched_symbols: list[str] = []
        for term in terms:
            if term in path_text:
                score += 8
            if term in record_terms:
                score += 2
            if term in imports:
                score += 3
        for symbol in record.get("symbols", []):
            if not isinstance(symbol, dict):
                continue
            haystack = f"{symbol.get('qualname', '')} {symbol.get('signature', '')} {symbol.get('doc', '')}".lower()
            hits = sum(1 for term in terms if term in haystack)
            if hits:
                score += 25 * hits
                matched_symbols.append(str(symbol.get("qualname", "")))
        return score, matched_symbols[:8]

    def _repo_index_search_records(self, terms: list[str], base: Path, *, limit: int, records: list[dict[str, Any]]) -> dict[str, Any]:
        record_by_path = {str(record.get("path")): record for record in records if isinstance(record, dict)}
        scored: list[dict[str, Any]] = []
        for record in records:
            score, matched_symbols = self._repo_index_score(record, terms)
            if score:
                scored.append({"score": score, "path": record.get("path"), "symbols": matched_symbols})
        ranked_records = sorted(scored, key=lambda item: (-int(item["score"]), str(item["path"])))[: max(1, int(limit))]
        output: list[str] = []
        for item in ranked_records:
            rel = str(item["path"])
            record = record_by_path.get(rel)
            if record is None:
                continue
            snippets: list[str] = []
            matched_symbol_names = set(item.get("symbols", []))
            for symbol in record.get("symbols", []):
                if not isinstance(symbol, dict):
                    continue
                haystack = f"{symbol.get('qualname', '')} {symbol.get('signature', '')}".lower()
                if symbol.get("qualname") in matched_symbol_names or any(term in haystack for term in terms):
                    snippets.append(f"{rel}:{symbol['start']}-{symbol['end']} {symbol['kind']} {symbol['qualname']}: {symbol['signature']}")
            for line_item in record.get("line_index", []):
                if not isinstance(line_item, dict):
                    continue
                line_terms = set(str(term).lower() for term in line_item.get("terms", []) if isinstance(term, str))
                text = str(line_item.get("text", ""))
                lowered = text.lower()
                hits = sum(1 for term in terms if term in line_terms or term in lowered)
                if hits and len(snippets) < 6:
                    snippets.append(f"{rel}:{line_item.get('line')}: {text[:180]}")
            output.append(f"{item['path']} score={item['score']}")
            output.extend(f"  {snippet}" for snippet in snippets[:6])
        return {
            "ok": True,
            "tool": "repo_index_search",
            "path": self.relative_label(base),
            "count": len(ranked_records),
            "output": "\n".join(output) if output else "(no ranked snippets)",
        }

    def repo_index_search(self, query: str, path: str = ".", limit: int = 10) -> dict[str, Any]:
        self._check_interrupted()
        base = self.resolve_path(path, allow_missing=False)
        terms = [term.lower() for term in re.findall(r"[A-Za-z_][\w.:-]*|\d+", query)]
        if not terms:
            return {"ok": False, "tool": "repo_index_search", "summary": "repo_index_search requires a non-empty query."}
        records = self._indexed_code_records(base, limit=1000)
        return self._repo_index_search_records(terms, base, limit=max(1, int(limit)), records=records)

    def indexed_search(self, query: str, path: str = ".", limit: int = 100) -> dict[str, Any]:
        self._check_interrupted()
        base = self.resolve_path(path, allow_missing=False)
        terms = [term.lower() for term in re.findall(r"[A-Za-z_][\w.:-]*|\d+", query)]
        if not terms:
            return {"ok": False, "tool": "indexed_search", "summary": "indexed_search requires a non-empty query."}
        limit_value = max(1, int(limit))
        rows: list[tuple[int, str]] = []
        for record in self._indexed_code_records(base, limit=1000):
            rel = str(record.get("path", ""))
            path_text = rel.lower()
            for item in record.get("line_index", []):
                if not isinstance(item, dict):
                    continue
                line_terms = set(str(term).lower() for term in item.get("terms", []) if isinstance(term, str))
                text = str(item.get("text", ""))
                lowered = text.lower()
                score = 0
                for term in terms:
                    if term in path_text:
                        score += 3
                    if term in line_terms:
                        score += 8
                    elif term in lowered:
                        score += 5
                if score:
                    rows.append((score, f"{rel}:{item.get('line')}: {text}"))
        ranked = [line for _, line in sorted(rows, key=lambda item: (-item[0], item[1]))[:limit_value]]
        return {
            "ok": True,
            "tool": "indexed_search",
            "path": self.relative_label(base),
            "count": len(ranked),
            "output": "\n".join(ranked) if ranked else "(no indexed matches)",
        }

    def repo_index_refresh(self, path: str = ".", limit: int = 1000) -> dict[str, Any]:
        self._check_interrupted()
        base = self.resolve_path(path, allow_missing=False)
        records = self._indexed_code_records(base, limit=max(1, int(limit)))
        symbol_count = sum(len(record.get("symbols", [])) for record in records if isinstance(record, dict))
        line_count = sum(len(record.get("line_index", [])) for record in records if isinstance(record, dict))
        return {
            "ok": True,
            "tool": "repo_index_refresh",
            "path": self.relative_label(base),
            "files": len(records),
            "symbols": symbol_count,
            "lines": line_count,
            "cache": self.relative_label(self._repo_index_cache_path()),
            "output": f"Indexed {len(records)} file(s), {symbol_count} symbol(s), {line_count} searchable line(s).",
        }

    def _semgrep_lang_for_path(self, base: Path) -> str | None:
        suffix = base.suffix.lower() if base.is_file() else ""
        if suffix == ".py":
            return "python"
        if suffix in {".js", ".jsx"}:
            return "javascript"
        if suffix in {".ts", ".tsx"}:
            return "typescript"
        if suffix == ".go":
            return "go"
        if suffix == ".rs":
            return "rust"
        if suffix == ".java":
            return "java"
        return None

    def semgrep_scan(self, pattern: str, path: str = ".", lang: str | None = None, limit: int = 50) -> dict[str, Any]:
        self._check_interrupted()
        semgrep = shutil.which("semgrep")
        if not semgrep:
            return self._missing_dependency_result("semgrep_scan", "semgrep", "semgrep is not installed. Install semgrep to use structural search.")
        base = self.resolve_path(path, allow_missing=False)
        clean_pattern = str(pattern or "").strip()
        if not clean_pattern:
            return {"ok": False, "tool": "semgrep_scan", "summary": "semgrep_scan requires a pattern."}
        clean_lang = (lang or self._semgrep_lang_for_path(base) or "python").strip().lower()
        allowed_langs = {"python", "javascript", "typescript", "go", "rust", "java", "c", "cpp", "csharp", "ruby", "php"}
        if clean_lang not in allowed_langs:
            return {"ok": False, "tool": "semgrep_scan", "summary": f"Unsupported semgrep language: {clean_lang}"}
        command = [semgrep, "--json", "--quiet", "-e", clean_pattern, "--lang", clean_lang, str(base)]
        completed = self._run_process(command, cwd=self.workspace_root, timeout=60, shell=False)
        output = self._collect_process_output(completed)
        try:
            payload = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError:
            payload = {}
        results = payload.get("results") if isinstance(payload, dict) else None
        rows: list[str] = []
        if isinstance(results, list):
            for item in results[: max(1, int(limit))]:
                if not isinstance(item, dict):
                    continue
                start = item.get("start") if isinstance(item.get("start"), dict) else {}
                extra = item.get("extra") if isinstance(item.get("extra"), dict) else {}
                item_path = str(item.get("path", ""))
                rel = item_path
                try:
                    rel = self.relative_label(Path(item_path))
                except Exception:
                    pass
                rows.append(f"{rel}:{start.get('line', '?')}: {str(extra.get('lines', '')).strip()[:220]}")
        return {
            "ok": completed.returncode in {0, 1},
            "tool": "semgrep_scan",
            "path": self.relative_label(base),
            "lang": clean_lang,
            "count": len(rows),
            "output": "\n".join(rows) if rows else ("(no semgrep matches)" if completed.returncode in {0, 1} else output),
        }

    def _missing_dependency_result(self, tool: str, dependency: str, summary: str) -> dict[str, Any]:
        return {
            "ok": False,
            "tool": tool,
            "summary": summary,
            "missing_dependency": dependency,
            "error_class": "missing_dependency",
        }

    def ast_search(self, pattern: str, path: str = ".", lang: str | None = None, limit: int = 50) -> dict[str, Any]:
        self._check_interrupted()
        executable = shutil.which("ast-grep") or shutil.which("sg")
        if not executable:
            return self._missing_dependency_result("ast_search", "ast-grep", "ast-grep is not installed. Install ast-grep or sg to use AST search.")
        base = self.resolve_path(path, allow_missing=False)
        clean_pattern = str(pattern or "").strip()
        if not clean_pattern:
            return {"ok": False, "tool": "ast_search", "summary": "ast_search requires a pattern."}
        clean_lang = (lang or self._semgrep_lang_for_path(base) or "").strip().lower()
        command = [executable, "--json", "-p", clean_pattern]
        if clean_lang:
            command.extend(["--lang", clean_lang])
        command.append(str(base))
        completed = self._run_process(command, cwd=self.workspace_root, timeout=60, shell=False)
        output = self._collect_process_output(completed)
        rows: list[str] = []
        try:
            payload = json.loads(completed.stdout or "[]")
        except json.JSONDecodeError:
            payload = None
        items = payload if isinstance(payload, list) else []
        for item in items[: max(1, int(limit))]:
            if not isinstance(item, dict):
                continue
            item_path = str(item.get("file") or item.get("path") or "")
            rel = item_path
            try:
                rel = self.relative_label(Path(item_path))
            except Exception:
                pass
            range_payload = item.get("range") if isinstance(item.get("range"), dict) else {}
            start = range_payload.get("start") if isinstance(range_payload.get("start"), dict) else {}
            line = int(start.get("line", 0)) + 1 if isinstance(start.get("line"), int) else "?"
            text = str(item.get("text") or item.get("lines") or "").strip().replace("\n", " ")
            rows.append(f"{rel}:{line}: {text[:220]}")
        return {
            "ok": completed.returncode in {0, 1},
            "tool": "ast_search",
            "path": self.relative_label(base),
            "lang": clean_lang or None,
            "count": len(rows),
            "output": "\n".join(rows) if rows else ("(no ast-grep matches)" if completed.returncode in {0, 1} else output),
        }

    def _language_for_path(self, path: Path) -> str | None:
        suffix = path.suffix.lower()
        if suffix == ".py" or (path / "pyproject.toml").exists():
            return "python"
        if suffix in {".js", ".jsx"} or (path / "package.json").exists():
            return "javascript"
        if suffix in {".ts", ".tsx"} or (path / "tsconfig.json").exists():
            return "typescript"
        if suffix == ".go" or (path / "go.mod").exists():
            return "go"
        if suffix == ".rs" or (path / "Cargo.toml").exists():
            return "rust"
        if suffix == ".java" or (path / "build.gradle").exists() or (path / "gradlew").exists():
            return "java"
        if suffix in {".c", ".cc", ".cpp", ".h", ".hpp"} or (path / "CMakeLists.txt").exists():
            return "cpp"
        return None

    def _lsp_server_for_lang(self, lang: str | None) -> tuple[str | None, str | None]:
        mapping = {
            "python": ("pyright", "pyright"),
            "javascript": ("typescript-language-server", "typescript-language-server"),
            "typescript": ("typescript-language-server", "typescript-language-server"),
            "go": ("gopls", "gopls"),
            "rust": ("rust-analyzer", "rust-analyzer"),
            "java": ("jdtls", "jdtls"),
            "cpp": ("clangd", "clangd"),
            "c": ("clangd", "clangd"),
        }
        key = (lang or "").strip().lower()
        name, executable = mapping.get(key, (None, None))
        if executable and not shutil.which(executable):
            return name, None
        return name, executable

    def lsp_diagnostics(self, path: str = ".", lang: str | None = None, timeout: int = 60) -> dict[str, Any]:
        self._check_interrupted()
        base = self.resolve_path(path, allow_missing=False)
        clean_lang = (lang or self._language_for_path(base) or "").strip().lower()
        server_name, server = self._lsp_server_for_lang(clean_lang)
        if server_name and server is None:
            return self._missing_dependency_result("lsp_diagnostics", server_name, f"{server_name} is not installed. Install it to use {clean_lang or 'language'} diagnostics.")
        if clean_lang == "python":
            result = self.lint_typecheck(self.relative_label(base), timeout=timeout)
            result["tool"] = "lsp_diagnostics"
            result["lang"] = clean_lang
            result["server"] = server_name
            return result
        validators = self.discover_validators(self.relative_label(base), limit=12)
        diagnostic_commands = [
            item
            for item in validators.get("validators", [])
            if isinstance(item, dict) and item.get("kind") in {"syntax", "typecheck", "lint", "check"} and item.get("available") is True
        ]
        if diagnostic_commands:
            command = str(diagnostic_commands[0].get("command") or "").strip()
            if command:
                result = self.run_shell(command, cwd=self.relative_label(base if base.is_dir() else base.parent), timeout=timeout)
                result["tool"] = "lsp_diagnostics"
                result["lang"] = clean_lang or None
                result["server"] = server_name
                return result
        return {
            "ok": True,
            "tool": "lsp_diagnostics",
            "path": self.relative_label(base),
            "lang": clean_lang or None,
            "server": server_name,
            "summary": "No available diagnostics command found.",
            "output": "(no diagnostics command found)",
        }

    def _identifier_at_location(self, path: Path, line: int, column: int) -> str | None:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line < 1 or line > len(lines):
            return None
        text = lines[line - 1]
        index = max(0, min(len(text), column - 1))
        left = index
        while left > 0 and re.match(r"\w", text[left - 1]):
            left -= 1
        right = index
        while right < len(text) and re.match(r"\w", text[right]):
            right += 1
        token = text[left:right].strip()
        return token if token else None

    def lsp_definition(self, path: str, line: int, column: int, limit: int = 20) -> dict[str, Any]:
        target = self.resolve_path(path, allow_missing=False)
        if target.is_dir():
            return {"ok": False, "tool": "lsp_definition", "summary": f"{path} is a directory."}
        clean_lang = self._language_for_path(target)
        server_name, server = self._lsp_server_for_lang(clean_lang)
        if server_name and server is None:
            return self._missing_dependency_result("lsp_definition", server_name, f"{server_name} is not installed. Install it to use {clean_lang or 'language'} definitions.")
        symbol = self._identifier_at_location(target, int(line), int(column))
        if not symbol:
            return {"ok": False, "tool": "lsp_definition", "path": self.relative_label(target), "summary": "No identifier found at location."}
        matches = self.search_symbols(symbol, path=".", limit=limit)
        return {
            "ok": matches.get("ok") is True,
            "tool": "lsp_definition",
            "path": self.relative_label(target),
            "server": server_name,
            "symbol": symbol,
            "count": matches.get("count", 0),
            "output": str(matches.get("output") or "(no definitions found)"),
        }

    def lsp_references(self, path: str, line: int, column: int, limit: int = 40) -> dict[str, Any]:
        target = self.resolve_path(path, allow_missing=False)
        if target.is_dir():
            return {"ok": False, "tool": "lsp_references", "summary": f"{path} is a directory."}
        clean_lang = self._language_for_path(target)
        server_name, server = self._lsp_server_for_lang(clean_lang)
        if server_name and server is None:
            return self._missing_dependency_result("lsp_references", server_name, f"{server_name} is not installed. Install it to use {clean_lang or 'language'} references.")
        symbol = self._identifier_at_location(target, int(line), int(column))
        if not symbol:
            return {"ok": False, "tool": "lsp_references", "path": self.relative_label(target), "summary": "No identifier found at location."}
        result = self.search(rf"\b{re.escape(symbol)}\b", path=".", limit=limit)
        result["tool"] = "lsp_references"
        result["server"] = server_name
        result["symbol"] = symbol
        return result

    def context_pack(self, request: str, path: str = ".", limit: int = 8) -> dict[str, Any]:
        self._check_interrupted()
        base = self.resolve_path(path, allow_missing=False)
        query = request.strip()
        if not query:
            return {"ok": False, "tool": "context_pack", "summary": "context_pack requires a non-empty request."}
        records = self._indexed_code_records(base, limit=1000)
        terms = [term.lower() for term in re.findall(r"[A-Za-z_][\w.:-]*|\d+", query)]
        ranked = self._repo_index_search_records(terms, base, limit=max(1, int(limit)), records=records) if terms else {"ok": False, "count": 0, "output": "(no ranked snippets)"}
        test_files: list[str] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            rel = str(record.get("path", ""))
            normalized_rel = rel.replace("\\", "/")
            if rel.endswith(".py") and ("test" in Path(rel).name.lower() or "/tests/" in f"/{normalized_rel}"):
                test_files.append(rel)
            if len(test_files) >= max(1, int(limit)):
                break
        git_summary = ""
        if (self.workspace_root / ".git").exists():
            status = self.git_status()
            if status.get("ok"):
                git_summary = self._truncate_text(str(status.get("output", "")).replace("\n", " | "), limit=220)
        suggested = "repo_index_search"
        ranked_output = str(ranked.get("output", ""))
        if re.search(r"\b(?:test|tests|pytest|unittest|failing|failure)\b", query, flags=re.IGNORECASE):
            suggested = "run_test"
        elif re.search(r"\b(?:fix|implement|refactor|change|edit|update)\b", query, flags=re.IGNORECASE):
            suggested = "read_symbol" if re.search(r"\b(?:function|method|class|symbol)\b", ranked_output) else "read_file"
        lines = [
            "context_pack:",
            f"request_terms={', '.join(self._extract_index_terms(query, limit=12))}",
            f"suggested_next_tool={suggested}",
        ]
        if test_files:
            lines.append("test_files=" + ", ".join(test_files))
        if git_summary:
            lines.append("git=" + git_summary)
        lines.append("ranked_evidence:")
        lines.append(self._truncate_text(ranked_output, limit=1300))
        return {
            "ok": True,
            "tool": "context_pack",
            "path": self.relative_label(base),
            "count": ranked.get("count", 0),
            "suggested_next_tool": suggested,
            "test_files": test_files,
            "output": "\n".join(lines),
        }

    def systems_lens(self, request: str, path: str = ".", evidence: str = "", limit: int = 8) -> dict[str, Any]:
        self._check_interrupted()
        query = str(request or "").strip()
        if not query:
            return {
                "ok": False,
                "tool": "systems_lens",
                "summary": "systems_lens requires a non-empty request.",
                "error_class": "invalid_args",
            }
        base = self.resolve_path(path, allow_missing=False)
        rel_path = self.relative_label(base)
        lowered = query.lower()
        try:
            limit_value = max(4, min(16, int(limit)))
        except (TypeError, ValueError):
            limit_value = 8

        concern_rules = [
            (
                r"\b(?:debug|bug|fail|failure|failing|error|exception|traceback|flaky|regression)\b",
                "debug: separate reproduced state, hidden inputs, validator signal, and likely changed boundary",
            ),
            (
                r"\b(?:perf|performance|slow|latency|throughput|profile|benchmark|token|speed)\b",
                "performance: measure before/after; split model time, tool time, IO, indexing, and retries",
            ),
            (
                r"\b(?:design|architecture|architect|system|workflow|pipeline|integration)\b",
                "design: compare component, workflow, dependency, and user-goal viewpoints",
            ),
            (
                r"\b(?:refactor|rename|migration|api|contract|signature|caller|callee)\b",
                "change: preserve observable contracts; check callers, callees, tests, and configs",
            ),
            (
                r"\b(?:security|secret|vulnerab|audit|permission|auth|token|credential)\b",
                "security: mark trust boundaries, data flow, secrets, dependency provenance, and write surfaces",
            ),
            (
                r"\b(?:ui|frontend|browser|page|localhost|url|screenshot)\b",
                "browser/UI: distinguish DOM state, rendered pixels, console/network errors, and user flow",
            ),
        ]
        concerns = [text for pattern, text in concern_rules if re.search(pattern, lowered)]
        if not concerns:
            concerns.append("general: start with the smallest useful subsystem and revise the boundary when evidence conflicts")
        concerns = concerns[:limit_value]

        core_questions = [
            "boundary: What exactly is inside the system, what is outside, and what excluded factor could still be causal?",
            "observer: Who is observing or measuring this, and would another role or metric show a different pattern?",
            "categories: What am I lumping together, and which distinction would change the conclusion?",
            "state: Which variables define the current condition, and which are merely symptoms?",
            "history: How did the system get here, and what earlier decision or hidden input could still be acting?",
            "feedback: What loop keeps regenerating the behavior, and what does it reinforce or dampen?",
            "delay: When should causes show up as effects, and are we measuring too early or too late?",
            "stocks_flows: What is accumulating or draining, and did inflow, outflow, or both change?",
            "coupling: Which parts affect each other unexpectedly, and where can changes propagate?",
            "incentives: Why is the behavior locally rational under each actor's information and constraints?",
            "model_limits: What does this explanation omit, and what observation would disconfirm it?",
            "intervention: After the fix, how will the system adapt and what second-order problem might appear?",
        ]
        question_rules = [
            (
                r"\b(?:debug|bug|fail|failure|failing|error|exception|traceback|flaky|regression)\b",
                [
                    "feedback: What loop keeps regenerating the failure after the first symptom appears?",
                    "coupling: Which dependency or interface is more connected than the code shape suggests?",
                    "decomposition: Is the failure in one component, or in interaction between components?",
                    "history: What recent change, delayed effect, or hidden input could explain the current state?",
                ],
            ),
            (
                r"\b(?:perf|performance|slow|latency|throughput|profile|benchmark|token|speed)\b",
                [
                    "stocks_flows: What is accumulating: calls, tokens, retries, subprocess startup, IO, cache misses, or waits?",
                    "delay: Are we measuring before warmup, cache fill, server startup, or delayed feedback completes?",
                    "measurement: Could the metric reward fewer calls while hiding longer wall-clock, or vice versa?",
                    "intervention: After optimization, what new bottleneck or behavior will appear?",
                ],
            ),
            (
                r"\b(?:design|architecture|architect|system|workflow|pipeline|integration)\b",
                [
                    "abstraction: Is this a code, architecture, product, workflow, dependency, or controller problem?",
                    "invariants: What behavior persists if the team, tool, model, metric, process, or module changes?",
                    "perspectives: What does engineering, tests, users, operations, and maintainers each reveal or hide?",
                    "local_rationality: Which locally reasonable behavior creates the global dysfunction?",
                ],
            ),
            (
                r"\b(?:refactor|rename|migration|api|contract|signature|caller|callee)\b",
                [
                    "invariants: Which external contracts must stay unchanged across the transformation?",
                    "second_order: What behavior changes after callers and tests adapt to this refactor?",
                    "burden_shift: Is this fix removing the cause, or adding a workaround that lowers pressure to fix it?",
                    "coupling: Where can this change propagate unexpectedly?",
                ],
            ),
            (
                r"\b(?:security|secret|vulnerab|audit|permission|auth|token|credential)\b",
                [
                    "trust_boundary: Where do data, credentials, permissions, and network effects cross boundaries?",
                    "observer: Could logs, scanners, or tests miss the behavior an attacker would exploit?",
                    "intervention: Who benefits, who pays, and how might actors adapt after the fix?",
                    "disconfirm: What evidence would show the issue is absent rather than merely unobserved?",
                ],
            ),
            (
                r"\b(?:ui|frontend|browser|page|localhost|url|screenshot)\b",
                [
                    "observer: Is the issue visible in DOM, rendered pixels, console, network, accessibility tree, or user flow?",
                    "delay: Does async loading, hydration, animation, caching, or debounce hide cause and effect?",
                    "state: Which route, viewport, auth, local storage, server state, and data response define this UI state?",
                    "measurement: Could screenshots pass while interaction or accessibility fails?",
                ],
            ),
        ]
        selected_questions = list(core_questions)
        for pattern, questions in question_rules:
            if re.search(pattern, lowered):
                selected_questions.extend(questions)
        selected_questions = list(dict.fromkeys(selected_questions))[:limit_value]

        next_tools = ["context_pack", "file_search/fts_search", "search_symbols/code_outline/read_symbol"]
        if any("debug:" in item for item in concerns):
            next_tools.extend(["discover_validators", "run_test", "diagnose_test_failure"])
        if any("performance:" in item for item in concerns):
            next_tools.extend(["discover_validators", "profile one focused command before/after"])
        if any("design:" in item for item in concerns):
            next_tools.extend(["repo_index_search", "contract_graph/call_graph"])
        if any("change:" in item for item in concerns):
            next_tools.extend(["contract_graph", "edit_intent", "select_tests"])
        if any("security:" in item for item in concerns):
            next_tools.append("security_scan when explicitly requested")
        if any("browser/UI:" in item for item in concerns):
            next_tools.append("browser_smoke for localhost/UI tasks")
        deduped_tools = list(dict.fromkeys(next_tools))[:limit_value]

        evidence_text = self._truncate_text(str(evidence or "").strip().replace("\r\n", "\n"), limit=500)
        lines = [
            "systems_lens:",
            f"boundary=Treat {rel_path} as the current system; dependencies, commands, tests, model behavior, and user workflow are environment until evidence shows coupling.",
            "state=Track inputs, outputs, config, git diff, failing traces, validators, and tool/model results as separate state dimensions.",
            "scale_time=Name whether the issue is line/symbol/file/repo/runtime/user-flow scale; measure current behavior before claiming improvement.",
            "viewpoints=user goal; implementation contract; tests/validators; runtime/dependencies; controller/tool loop.",
            "coupling=Assume independence is unproven. Use search, xref, validators, and traces to find strong connections.",
            "model_limit=Every search/read/graph is a projection. Avoid over-lumping the whole repo and over-splitting into line-by-line noise.",
            "stop_rule=Stop using a heuristic when it stops producing new evidence after a focused retry; switch representation/tool.",
            "concerns:",
        ]
        lines.extend(f"- {item}" for item in concerns)
        lines.append("questions:")
        lines.extend(f"- {item}" for item in selected_questions)
        lines.append("next_tools=" + ", ".join(deduped_tools))
        if evidence_text:
            lines.append("current_evidence=" + evidence_text)
        return {
            "ok": True,
            "tool": "systems_lens",
            "path": rel_path,
            "summary": "Framed task boundary, observer, categories, state, scale, feedback, coupling, model limits, and validation checks.",
            "concerns": concerns,
            "questions": selected_questions,
            "next_tools": deduped_tools,
            "output": "\n".join(lines),
        }

    def _resolve_python_module_file(self, module: str) -> Path | None:
        clean = module.strip(".")
        if not clean:
            return None
        rel = Path(*clean.split("."))
        candidates = [
            self.workspace_root / rel.with_suffix(".py"),
            self.workspace_root / rel / "__init__.py",
            self.workspace_root / "src" / rel.with_suffix(".py"),
            self.workspace_root / "src" / rel / "__init__.py",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file() and self.workspace_root in candidate.resolve().parents:
                return candidate.resolve()
        return None

    def _python_import_targets(self, target: Path) -> list[dict[str, Any]]:
        text = target.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(text), filename=self.relative_label(target))
        except SyntaxError:
            return []
        targets: list[dict[str, Any]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                module_file = self._resolve_python_module_file(node.module)
                for alias in node.names:
                    if module_file is not None:
                        targets.append(
                            {
                                "path": self.relative_label(module_file),
                                "symbol": alias.name,
                                "reason": f"{self.relative_label(target)} imports {alias.name} from {node.module}",
                            }
                        )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module_file = self._resolve_python_module_file(alias.name)
                    if module_file is not None:
                        targets.append(
                            {
                                "path": self.relative_label(module_file),
                                "symbol": alias.asname or alias.name.split(".")[-1],
                                "reason": f"{self.relative_label(target)} imports {alias.name}",
                            }
                        )
        return targets

    def _traceback_targets(self, output: str) -> list[dict[str, Any]]:
        targets: list[dict[str, Any]] = []
        for match in re.finditer(r'File "([^"]+)", line (\d+), in ([A-Za-z_<>][\w<>]*)', output):
            raw_path, line, symbol = match.groups()
            try:
                path = self.resolve_path(raw_path, allow_missing=False)
            except Exception:
                continue
            rel = self.relative_label(path)
            if any(part in {"tests", "test"} or part.startswith("test_") for part in Path(rel).parts):
                continue
            targets.append({"path": rel, "symbol": symbol, "line": int(line), "reason": "implementation frame in traceback"})
        return targets

    def _test_import_targets_from_output(self, output: str) -> list[dict[str, Any]]:
        targets: list[dict[str, Any]] = []
        seen_files: set[Path] = set()
        for raw_path in re.findall(r'File "([^"]+?\.py)"', output):
            try:
                path = self.resolve_path(raw_path, allow_missing=False)
            except Exception:
                continue
            rel = self.relative_label(path)
            name = path.name.lower()
            if not (name.startswith("test_") or name.endswith("_test.py") or "/tests/" in rel.replace("\\", "/").lower()):
                continue
            if path in seen_files:
                continue
            seen_files.add(path)
            targets.extend(self._python_import_targets(path))
        return targets

    def find_implementation_target(
        self,
        test_path: str | None = None,
        path: str | None = None,
        query: str | None = None,
        output: str | None = None,
        traceback: str | None = None,
        limit: int = 12,
    ) -> dict[str, Any]:
        self._check_interrupted()
        targets: list[dict[str, Any]] = []
        if test_path is None and isinstance(path, str) and path.strip():
            test_path = path
        if isinstance(test_path, str) and test_path.strip():
            try:
                direct = self.resolve_path(test_path, allow_missing=False)
                if direct.suffix.lower() in CODE_FILE_SUFFIXES and not self._path_looks_like_test(direct):
                    symbol = str(query or "").strip()
                    return {
                        "ok": True,
                        "tool": "find_implementation_target",
                        "count": 1,
                        "targets": [{"path": self.relative_label(direct), "symbol": symbol, "reason": "direct source path"}],
                        "output": f"{self.relative_label(direct)}" + (f" symbol={symbol}" if symbol else "") + " reason=direct source path",
                    }
            except Exception:
                pass
        raw_output = "\n".join(part for part in [output, traceback] if isinstance(part, str) and part.strip())
        if raw_output:
            targets.extend(self._traceback_targets(raw_output))
            targets.extend(self._test_import_targets_from_output(raw_output))
            for failed in re.findall(r"(?m)^FAILED\s+([^:\s]+)(?:::[^\s]+)?", raw_output):
                try:
                    test_file = self.resolve_path(failed, allow_missing=False)
                except Exception:
                    continue
                if test_file.suffix.lower() == ".py":
                    targets.extend(self._python_import_targets(test_file))
        if test_path:
            target = self.resolve_path(test_path, allow_missing=False)
            if target.suffix.lower() == ".py":
                targets.extend(self._python_import_targets(target))
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in targets:
            key = (str(item.get("path", "")), str(item.get("symbol", "")))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= max(1, int(limit)):
                break
        lines = []
        for item in deduped:
            suffix = f":{item['line']}" if "line" in item else ""
            symbol = f" symbol={item['symbol']}" if item.get("symbol") else ""
            lines.append(f"{item['path']}{suffix}{symbol} reason={item.get('reason', '')}")
        return {
            "ok": True,
            "tool": "find_implementation_target",
            "count": len(deduped),
            "targets": deduped,
            "output": "\n".join(lines) if lines else "(no likely implementation targets found)",
        }

    def diagnose_test_failure(
        self,
        output: str = "",
        run_test_output: str | None = None,
        path: str = ".",
        limit: int = 12,
    ) -> dict[str, Any]:
        self._check_interrupted()
        text = run_test_output if isinstance(run_test_output, str) and run_test_output.strip() else output
        if not isinstance(text, str) or not text.strip():
            return {"ok": False, "tool": "diagnose_test_failure", "summary": "diagnose_test_failure requires run_test output."}
        failures: list[dict[str, Any]] = []
        for match in re.finditer(r"(?m)^FAILED\s+([^\s]+?)(?:::([^\s]+))?\s*(?:-\s*(.*))?$", text):
            failures.append({"test": match.group(1), "case": match.group(2) or "", "message": (match.group(3) or "").strip()})
        for match in re.finditer(r"(?m)^(?:FAIL|ERROR):\s+([^\s]+)\s+\(([^)]+)\)", text):
            failures.append({"test": match.group(2), "case": match.group(1), "message": ""})
        assertions = re.findall(r"(?m)^\s*E\s+assert\s+(.+)$", text)
        unittest_assertions = re.findall(r"AssertionError:\s+([^\n]+)", text)
        exceptions = re.findall(r"(?m)^(?:E\s+)?([A-Za-z_][\w.]+(?:Error|Exception)):\s*(.+)$", text)
        error_class = self.classify_error(text)
        missing_dependency = self._missing_dependency_name(text)
        expected_actual: list[str] = []
        for assertion in assertions:
            if "==" in assertion:
                left, right = assertion.split("==", 1)
                expected_actual.append(f"actual={left.strip()[:120]} expected={right.strip()[:120]}")
            else:
                expected_actual.append(assertion.strip()[:180])
        for assertion in unittest_assertions:
            if "!=" in assertion:
                left, right = assertion.split("!=", 1)
                expected_actual.append(f"actual={left.strip()[:120]} expected={right.strip()[:120]}")
            else:
                expected_actual.append(assertion.strip()[:180])
        root_causes: list[str] = []
        if exceptions:
            for exc, message in exceptions[: max(1, int(limit))]:
                root_causes.append(f"{exc}: {message.strip()[:180]}")
        if expected_actual:
            root_causes.extend(f"assertion mismatch: {item}" for item in expected_actual[: max(1, int(limit))])
        if not root_causes and "ModuleNotFoundError" in text:
            root_causes.append("import/module discovery failure")
        if missing_dependency:
            root_causes.insert(0, f"missing dependency: {missing_dependency}")
        targets = self.find_implementation_target(output=text, limit=limit)
        next_tool = "read_file"
        if error_class == "syntax_error":
            next_tool = "lint_typecheck"
        elif error_class in {"test_assertion", "import_error"}:
            next_tool = "find_implementation_target"
        elif error_class == "missing_dependency":
            next_tool = "fail_closed"
        elif error_class in {"invalid_args", "command_not_found"}:
            next_tool = "run_test"
        lines: list[str] = []
        lines.append(f"class {error_class} next={next_tool}")
        if missing_dependency:
            lines.append(f"missing_dependency {missing_dependency}")
        for failure in failures[: max(1, int(limit))]:
            test_id = failure["test"] + (f"::{failure['case']}" if failure["case"] else "")
            lines.append(f"fail {test_id}: {failure['message'] or '(see traceback)'}")
        for cause in root_causes[: max(1, int(limit))]:
            lines.append(f"cause {cause}")
        if targets.get("targets"):
            lines.append("likely targets:")
            lines.extend(f"  {line}" for line in str(targets.get("output", "")).splitlines()[: max(1, int(limit))])
        return {
            "ok": True,
            "tool": "diagnose_test_failure",
            "path": self.relative_label(self.resolve_path(path, allow_missing=False)),
            "error_class": error_class,
            "missing_dependency": missing_dependency,
            "next_tool": next_tool,
            "failures": failures[: max(1, int(limit))],
            "root_causes": root_causes[: max(1, int(limit))],
            "targets": targets.get("targets", []),
            "output": "\n".join(lines) if lines else "(no structured failures found; inspect raw output)",
        }

    def run_function_probe(
        self,
        module: str,
        expressions: list[str] | str,
        function: str | None = None,
        timeout: int = 30,
    ) -> dict[str, Any]:
        if isinstance(expressions, str):
            probe_expressions = [expressions]
        elif isinstance(expressions, list):
            probe_expressions = [str(item) for item in expressions]
        else:
            return {"ok": False, "tool": "run_function_probe", "summary": "expressions must be a string or list of strings."}
        if not module.strip() or not probe_expressions:
            return {"ok": False, "tool": "run_function_probe", "summary": "module and expressions are required."}
        approved, reason = self._approve_shell(f"python function probe for {module}", ".")
        if not approved:
            return {"ok": False, "tool": "run_function_probe", "summary": reason}
        script = (
            "import importlib,json,os,sys,traceback\n"
            "workspace=os.getcwd(); sys.path.insert(0, workspace); sys.path.insert(0, os.path.join(workspace, 'src'))\n"
            f"module_name={json.dumps(module)}\n"
            f"function_name={json.dumps(function or '')}\n"
            f"expressions={json.dumps(probe_expressions)}\n"
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
        completed = self._run_process([sys.executable, "-c", script], cwd=self.workspace_root, timeout=max(1, int(timeout)))
        raw_output = self._collect_process_output(completed)
        rows: list[Any]
        try:
            rows = json.loads(completed.stdout.strip())
        except json.JSONDecodeError:
            rows = []
        rendered = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("ok"):
                rendered.append(f"{row.get('expression')}: {row.get('repr')} ({row.get('type')})")
            else:
                rendered.append(f"{row.get('expression')}: ERROR {row.get('error')}")
        return {
            "ok": completed.returncode == 0 and all(isinstance(row, dict) and row.get("ok") for row in rows),
            "tool": "run_function_probe",
            "module": module,
            "function": function or "",
            "exit_code": completed.returncode,
            "results": rows,
            "output": "\n".join(rendered) if rendered else raw_output,
        }

    def _annotation_text(self, node: ast.AST | None) -> str:
        if node is None:
            return "Any"
        try:
            return ast.unparse(node)[:120]
        except Exception:
            return "Any"

    def _python_function_contracts(self, base: Path, limit: int = 500) -> dict[str, Any]:
        definitions: dict[str, dict[str, Any]] = {}
        calls_by_def: dict[str, list[dict[str, Any]]] = {}
        imports_by_file: dict[str, list[str]] = {}
        tests_by_symbol: dict[str, list[str]] = {}
        module_aliases_by_file: dict[str, dict[str, str]] = {}

        for file_path in self._iter_code_files(base, limit=limit):
            if file_path.suffix.lower() != ".py":
                continue
            rel = self.relative_label(file_path)
            text = file_path.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(self._python_parse_text(text), filename=rel)
            except SyntaxError:
                continue
            imports: list[str] = []
            aliases: dict[str, str] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                        aliases[alias.asname or alias.name.split(".")[0]] = alias.name
                elif isinstance(node, ast.ImportFrom) and node.module:
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")
                        aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
            imports_by_file[rel] = sorted(set(imports))[:120]
            module_aliases_by_file[rel] = aliases

            class Visitor(ast.NodeVisitor):
                def __init__(self, outer: ToolExecutor) -> None:
                    self.outer = outer
                    self.stack: list[str] = []

                def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
                    self._visit_function(node, "function", is_async=False)

                def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
                    self._visit_function(node, "function", is_async=True)

                def visit_ClassDef(self, node: ast.ClassDef) -> Any:
                    previous = list(self.stack)
                    self.stack.append(node.name)
                    self.generic_visit(node)
                    self.stack = previous

                def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, kind: str, *, is_async: bool) -> None:
                    qualname = ".".join([*self.stack, node.name])
                    if self.stack and self.stack[-1][:1].isupper():
                        kind = "method"
                    key = f"{rel}:{qualname}"
                    args = self.outer._contract_args(node.args)
                    returns = self.outer._annotation_text(node.returns)
                    return_shapes = self.outer._return_shape_hints(node, args)
                    purity = self.outer._purity_hint(node, imports_by_file.get(rel, []))
                    definitions[key] = {
                        "path": rel,
                        "symbol": qualname,
                        "name": node.name,
                        "kind": kind,
                        "line": int(getattr(node, "lineno", 1)),
                        "end": int(getattr(node, "end_lineno", getattr(node, "lineno", 1))),
                        "args": args,
                        "arity": self.outer._callable_arity(node.args),
                        "returns": returns,
                        "return_shapes": sorted(return_shapes),
                        "purity": purity,
                        "is_async": is_async,
                        "has_yield": any(isinstance(child, (ast.Yield, ast.YieldFrom)) for child in ast.walk(node)),
                        "node": node,
                    }
                    calls_by_def[key] = self.outer._calls_in_node(node)
                    previous = list(self.stack)
                    self.stack.append(node.name)
                    self.generic_visit(node)
                    self.stack = previous

            Visitor(self).visit(tree)

            if self._path_looks_like_test(file_path):
                text_lower = text.lower()
                for key, item in definitions.items():
                    leaf = str(item.get("name", "")).lower()
                    if leaf and re.search(rf"\b{re.escape(leaf)}\b", text_lower):
                        tests_by_symbol.setdefault(str(item.get("name")), []).append(rel)

        callers_by_leaf: dict[str, list[dict[str, Any]]] = {}
        for caller_key, calls in calls_by_def.items():
            caller = definitions.get(caller_key)
            if not caller:
                continue
            for call in calls:
                callers_by_leaf.setdefault(str(call.get("name")), []).append(
                    {
                        "path": caller["path"],
                        "line": call.get("line", caller["line"]),
                        "symbol": caller["symbol"],
                        "args": call.get("args", 0),
                        "keywords": call.get("keywords", []),
                        "return_expectations": call.get("return_expectations", []),
                    }
                )

        return {
            "definitions": definitions,
            "calls_by_def": calls_by_def,
            "imports_by_file": imports_by_file,
            "aliases_by_file": module_aliases_by_file,
            "callers_by_leaf": callers_by_leaf,
            "tests_by_symbol": tests_by_symbol,
        }

    def _contract_args(self, args: ast.arguments) -> list[dict[str, Any]]:
        defaults_start = len(args.args) - len(args.defaults)
        rows: list[dict[str, Any]] = []
        all_positional = [*args.posonlyargs, *args.args]
        for index, arg in enumerate(all_positional):
            has_default = index >= defaults_start if arg in args.args else False
            rows.append({"name": arg.arg, "annotation": self._annotation_text(arg.annotation), "required": not has_default})
        if args.vararg:
            rows.append({"name": "*" + args.vararg.arg, "annotation": self._annotation_text(args.vararg.annotation), "required": False})
        for index, arg in enumerate(args.kwonlyargs):
            rows.append({"name": arg.arg, "annotation": self._annotation_text(arg.annotation), "required": args.kw_defaults[index] is None})
        if args.kwarg:
            rows.append({"name": "**" + args.kwarg.arg, "annotation": self._annotation_text(args.kwarg.annotation), "required": False})
        return rows

    def _callable_arity(self, args: ast.arguments) -> dict[str, Any]:
        positional = [*args.posonlyargs, *args.args]
        required_positional = len(positional) - len(args.defaults)
        required_kwonly = sum(1 for default in args.kw_defaults if default is None)
        return {
            "min": required_positional + required_kwonly,
            "max": None if args.vararg else len(positional),
            "has_vararg": args.vararg is not None,
            "has_kwarg": args.kwarg is not None,
        }

    def _call_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def _calls_in_node(self, node: ast.AST) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                name = self._call_name(child.func)
                if not name:
                    continue
                calls.append(
                    {
                        "name": name,
                        "args": len(child.args),
                        "keywords": [kw.arg for kw in child.keywords if kw.arg],
                        "line": int(getattr(child, "lineno", 1)),
                    }
                )
        for expectation in self._call_return_expectations(node):
            for call in calls:
                if call["name"] == expectation["name"] and call["line"] == expectation["call_line"]:
                    call.setdefault("return_expectations", []).append(
                        {"shape": expectation["shape"], "line": expectation["line"], "source": expectation["source"]}
                    )
        return calls

    def _call_return_expectations(self, node: ast.AST) -> list[dict[str, Any]]:
        expectations: list[dict[str, Any]] = []
        variable_sources: dict[str, list[dict[str, Any]]] = {}

        def call_source(value: ast.AST) -> dict[str, Any] | None:
            if isinstance(value, ast.Call):
                name = self._call_name(value.func)
                if name:
                    return {"name": name, "call_line": int(getattr(value, "lineno", 1))}
            if isinstance(value, ast.Name):
                sources = variable_sources.get(value.id)
                if sources:
                    return sources[-1]
            return None

        def add_expectation(value: ast.AST, shape: str, line: int, source: str) -> None:
            found = call_source(value)
            if found:
                expectations.append({**found, "shape": shape, "line": line, "source": source})

        class ExpectationVisitor(ast.NodeVisitor):
            def visit_Assign(self, assign: ast.Assign) -> Any:
                found = call_source(assign.value)
                if found:
                    for target in assign.targets:
                        if isinstance(target, ast.Name):
                            variable_sources.setdefault(target.id, []).append(found)
                self.generic_visit(assign)

            def visit_AnnAssign(self, assign: ast.AnnAssign) -> Any:
                if assign.value is not None:
                    found = call_source(assign.value)
                    if found and isinstance(assign.target, ast.Name):
                        variable_sources.setdefault(assign.target.id, []).append(found)
                self.generic_visit(assign)

            def visit_For(self, loop: ast.For) -> Any:
                add_expectation(loop.iter, "iterable", int(getattr(loop, "lineno", 1)), "for")
                self.generic_visit(loop)

            def visit_ListComp(self, comp: ast.ListComp) -> Any:
                for generator in comp.generators:
                    add_expectation(generator.iter, "iterable", int(getattr(generator, "lineno", 1)), "comprehension")
                self.generic_visit(comp)

            def visit_Subscript(self, subscript: ast.Subscript) -> Any:
                shape = "indexable"
                target_slice = subscript.slice
                if isinstance(target_slice, ast.Constant):
                    if isinstance(target_slice.value, str):
                        shape = "dict"
                    elif isinstance(target_slice.value, int):
                        shape = "sequence"
                add_expectation(subscript.value, shape, int(getattr(subscript, "lineno", 1)), "subscript")
                self.generic_visit(subscript)

            def visit_Call(self, call: ast.Call) -> Any:
                if isinstance(call.func, ast.Attribute):
                    attr = call.func.attr
                    if attr in {"items", "keys", "values", "get"}:
                        add_expectation(call.func.value, "dict", int(getattr(call, "lineno", 1)), f".{attr}()")
                    elif attr in {"append", "extend", "insert"}:
                        add_expectation(call.func.value, "list", int(getattr(call, "lineno", 1)), f".{attr}()")
                elif isinstance(call.func, ast.Name) and call.args:
                    if call.func.id in {"sum", "list", "tuple", "set", "sorted", "any", "all", "max", "min"}:
                        add_expectation(call.args[0], "iterable", int(getattr(call, "lineno", 1)), call.func.id)
                self.generic_visit(call)

        ExpectationVisitor().visit(node)
        return expectations

    def _return_shape_hints(self, node: ast.FunctionDef | ast.AsyncFunctionDef, args: list[dict[str, Any]] | None = None) -> set[str]:
        shapes: set[str] = set()
        arg_shapes: dict[str, str] = {}
        for arg in args or []:
            name = str(arg.get("name", "")).lstrip("*")
            shape = self._annotation_expected_shape(str(arg.get("annotation", "")))
            if name and shape:
                arg_shapes[name] = shape
        has_return = False
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                has_return = True
                value = child.value
                if value is None or isinstance(value, ast.Constant) and value.value is None:
                    shapes.add("none")
                elif isinstance(value, ast.Tuple):
                    shapes.add(f"tuple[{len(value.elts)}]")
                elif isinstance(value, ast.List):
                    shapes.add("list")
                elif isinstance(value, ast.Dict):
                    shapes.add("dict")
                elif isinstance(value, ast.Set):
                    shapes.add("set")
                elif isinstance(value, ast.Constant):
                    shapes.add(type(value.value).__name__)
                elif isinstance(value, ast.Call):
                    shapes.add(f"call:{self._call_name(value.func) or 'unknown'}")
                elif isinstance(value, ast.Name):
                    shapes.add(arg_shapes.get(value.id, f"name:{value.id}"))
                else:
                    shapes.add(type(value).__name__.lower())
        if not has_return:
            shapes.add("implicit_none")
        return shapes

    def _purity_hint(self, node: ast.FunctionDef | ast.AsyncFunctionDef, imports: list[str]) -> str:
        impure_imports = {"os", "subprocess", "requests", "httpx", "socket", "pathlib", "shutil"}
        mutating_methods = {"append", "extend", "insert", "remove", "pop", "clear", "update", "setdefault", "write", "writelines", "unlink", "mkdir", "rmdir", "rename", "replace"}
        for imported in imports:
            if imported.split(".")[0] in impure_imports:
                return "impure_hint"
        for child in ast.walk(node):
            if isinstance(child, (ast.Global, ast.Nonlocal, ast.Delete, ast.AugAssign)):
                return "impure_hint"
            if isinstance(child, ast.Assign) and any(isinstance(target, (ast.Attribute, ast.Subscript)) for target in child.targets):
                return "impure_hint"
            if isinstance(child, ast.Call):
                name = self._call_name(child.func)
                if name in mutating_methods or name in {"open", "print", "input"}:
                    return "impure_hint"
        return "pure_hint"

    def _contract_signature(self, item: dict[str, Any]) -> str:
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

    def contract_graph(self, path: str = ".", symbol: str | None = None, limit: int = 40) -> dict[str, Any]:
        self._check_interrupted()
        base = self.resolve_path(path, allow_missing=False)
        contracts = self._python_function_contracts(base)
        definitions = contracts["definitions"]
        target = (symbol or "").strip()
        if target:
            rows = [
                item for item in definitions.values()
                if item["symbol"] == target or str(item["symbol"]).endswith(f".{target}") or item["name"] == target
            ]
        else:
            rows = list(definitions.values())
        if not rows and base.suffix.lower() != ".py":
            fallback = self.call_graph(path=path, symbol=symbol, limit=limit)
            fallback["tool"] = "contract_graph"
            return fallback
        output: list[str] = []
        for item in rows[: max(1, int(limit))]:
            leaf = str(item["name"])
            key = f"{item['path']}:{item['symbol']}"
            callees = sorted({str(call.get("name")) for call in contracts["calls_by_def"].get(key, []) if call.get("name")})[:10]
            callers = contracts["callers_by_leaf"].get(leaf, [])[:8]
            tests = contracts["tests_by_symbol"].get(leaf, [])[:6]
            shapes = ",".join(item.get("return_shapes", [])) or "unknown"
            output.append(f"{item['path']}:{item['line']} {self._contract_signature(item)} shape={shapes} purity={item['purity']}")
            output.append(f"  callees: {', '.join(callees) if callees else '(none)'}")
            output.append("  callers: " + (", ".join(f"{caller['path']}:{caller['line']} {caller['symbol']}({caller['args']})" for caller in callers) if callers else "(none)"))
            output.append(f"  affected_tests: {', '.join(tests) if tests else '(none)'}")
        return {
            "ok": True,
            "tool": "contract_graph",
            "path": self.relative_label(base),
            "symbol": target,
            "count": len(rows),
            "output": "\n".join(output) if output else "(no Python contracts found)",
        }

    def _annotation_allows_none(self, annotation: str) -> bool:
        lowered = annotation.replace(" ", "").lower()
        return lowered in {"any", "none"} or "optional[" in lowered or "|none" in lowered or "none|" in lowered

    def _annotation_expected_shape(self, annotation: str) -> str | None:
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

    def _return_shape_compatible(self, expected: str, shapes: list[str]) -> bool:
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

    def _caller_return_expectation_compatible(self, expectation: str, annotation: str, shapes: list[str]) -> bool:
        annotation_shape = self._annotation_expected_shape(annotation)
        concrete_shapes = {shape for shape in shapes if not shape.startswith(("name:", "call:")) and shape not in {"binop", "unaryop", "boolop", "compare"}}
        if expectation == "dict":
            if annotation_shape:
                return annotation_shape == "dict"
            return "dict" in concrete_shapes or not concrete_shapes
        if expectation in {"sequence", "list"}:
            if annotation_shape:
                return annotation_shape in {"list", "tuple"}
            return bool({"list"} & concrete_shapes or any(shape.startswith("tuple") for shape in concrete_shapes) or not concrete_shapes)
        if expectation == "iterable":
            if annotation_shape:
                return annotation_shape in {"list", "tuple", "set", "dict"}
            scalar_shapes = {"int", "float", "bool", "none", "implicit_none"}
            return not bool(concrete_shapes & scalar_shapes)
        if expectation == "indexable":
            if annotation_shape:
                return annotation_shape in {"list", "tuple", "dict"}
            scalar_shapes = {"int", "float", "bool", "none", "implicit_none"}
            return not bool(concrete_shapes & scalar_shapes)
        return True

    def _value_shape_hint(self, value: ast.AST, local_shapes: dict[str, str]) -> str | None:
        if isinstance(value, ast.Name):
            return local_shapes.get(value.id)
        if isinstance(value, ast.List):
            return "list"
        if isinstance(value, ast.Dict):
            return "dict"
        if isinstance(value, ast.Set):
            return "set"
        if isinstance(value, ast.Tuple):
            return "tuple"
        if isinstance(value, ast.Constant):
            if isinstance(value.value, bool):
                return "bool"
            if isinstance(value.value, int):
                return "int"
            if isinstance(value.value, float):
                return "float"
            if isinstance(value.value, str):
                return "str"
            if value.value is None:
                return "none"
        if isinstance(value, ast.Call):
            name = self._call_name(value.func)
            if name in {"list", "dict", "set", "tuple", "str", "int", "float", "bool"}:
                return name
        return None

    def _library_method_receiver_contracts(self) -> dict[str, set[str]]:
        return {
            "append": {"list"},
            "extend": {"list"},
            "insert": {"list"},
            "sort": {"list"},
            "reverse": {"list"},
            "items": {"dict"},
            "keys": {"dict"},
            "values": {"dict"},
            "get": {"dict"},
            "setdefault": {"dict"},
            "add": {"set"},
            "discard": {"set"},
            "union": {"set"},
            "intersection": {"set"},
            "difference": {"set"},
            "lower": {"str"},
            "upper": {"str"},
            "strip": {"str"},
            "split": {"str"},
            "join": {"str"},
            "startswith": {"str"},
            "endswith": {"str"},
            "format": {"str"},
            "encode": {"str"},
            "bit_length": {"int"},
            "to_bytes": {"int"},
            "is_integer": {"float"},
        }

    def _library_method_contract_diagnostics(self, item: dict[str, Any], node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        local_shapes: dict[str, str] = {}
        for arg in item.get("args", []):
            if not isinstance(arg, dict):
                continue
            name = str(arg.get("name", "")).lstrip("*")
            shape = self._annotation_expected_shape(str(arg.get("annotation", "")))
            if name and shape:
                local_shapes[name] = shape
        diagnostics: list[str] = []
        method_contracts = self._library_method_receiver_contracts()

        class MethodVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, nested: ast.FunctionDef) -> Any:
                if nested is not node:
                    return None
                self.generic_visit(nested)
                return None

            def visit_AsyncFunctionDef(self, nested: ast.AsyncFunctionDef) -> Any:
                if nested is not node:
                    return None
                self.generic_visit(nested)
                return None

            def visit_Assign(self, assign: ast.Assign) -> Any:
                shape = self_outer._value_shape_hint(assign.value, local_shapes)
                if shape:
                    for target in assign.targets:
                        if isinstance(target, ast.Name):
                            local_shapes[target.id] = shape
                self.generic_visit(assign)

            def visit_AnnAssign(self, assign: ast.AnnAssign) -> Any:
                shape = self_outer._annotation_expected_shape(self_outer._annotation_text(assign.annotation))
                if not shape and assign.value is not None:
                    shape = self_outer._value_shape_hint(assign.value, local_shapes)
                if shape and isinstance(assign.target, ast.Name):
                    local_shapes[assign.target.id] = shape
                self.generic_visit(assign)

            def visit_Call(self, call: ast.Call) -> Any:
                if isinstance(call.func, ast.Attribute):
                    method = call.func.attr
                    expected = method_contracts.get(method)
                    receiver_shape = self_outer._value_shape_hint(call.func.value, local_shapes)
                    if expected and receiver_shape and receiver_shape not in expected:
                        expected_text = "/".join(sorted(expected))
                        diagnostics.append(
                            f"{item['path']}:{int(getattr(call, 'lineno', item['line']))} {item['symbol']} calls .{method}() "
                            f"on {receiver_shape}; expected {expected_text} receiver"
                        )
                self.generic_visit(call)

        self_outer = self
        MethodVisitor().visit(node)
        return diagnostics

    def contract_check(
        self,
        changed_files: list[str] | str,
        changed_symbols: list[str] | str | None = None,
        limit: int = 80,
    ) -> dict[str, Any]:
        raw_files = [changed_files] if isinstance(changed_files, str) else list(changed_files or [])
        raw_symbols = [changed_symbols] if isinstance(changed_symbols, str) else list(changed_symbols or [])
        selected_symbols = {str(symbol).split(".")[-1] for symbol in raw_symbols if str(symbol).strip()}
        python_files: list[Path] = []
        for raw_path in raw_files:
            try:
                path = self.resolve_path(str(raw_path), allow_missing=False)
            except Exception:
                continue
            if path.is_file() and path.suffix.lower() == ".py":
                python_files.append(path)
        if not python_files:
            return {"ok": True, "tool": "contract_check", "checked": [], "diagnostics": [], "output": "no changed Python source files"}
        contracts = self._python_function_contracts(self.workspace_root)
        definitions = contracts["definitions"]
        diagnostics: list[str] = []
        checked: list[str] = []
        changed_rels = {self.relative_label(path) for path in python_files}
        for key, item in definitions.items():
            if item["path"] not in changed_rels:
                continue
            if selected_symbols and item["name"] not in selected_symbols and item["symbol"] not in selected_symbols:
                continue
            checked.append(f"{item['path']}:{item['line']} {item['symbol']}")
            returns = str(item.get("returns") or "Any")
            shapes = list(item.get("return_shapes", []))
            if not self._annotation_allows_none(returns) and any(shape in {"none", "implicit_none"} for shape in shapes):
                diagnostics.append(f"{item['path']}:{item['line']} {item['symbol']} annotated -> {returns} but may return None")
            expected_shape = self._annotation_expected_shape(returns)
            if expected_shape and not self._return_shape_compatible(expected_shape, shapes):
                diagnostics.append(f"{item['path']}:{item['line']} {item['symbol']} annotated -> {returns} but returns shape {','.join(shapes)}")
            if item.get("is_async") and item.get("has_yield"):
                diagnostics.append(f"{item['path']}:{item['line']} {item['symbol']} is async and yields; verify async generator contract")
            node = item.get("node")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                diagnostics.extend(self._library_method_contract_diagnostics(item, node))
            arity = item.get("arity", {})
            min_args = int(arity.get("min", 0))
            max_args = arity.get("max")
            for caller in contracts["callers_by_leaf"].get(str(item["name"]), [])[: max(1, int(limit))]:
                arg_count = int(caller.get("args", 0))
                if arg_count < min_args or (isinstance(max_args, int) and arg_count > max_args):
                    diagnostics.append(
                        f"{caller['path']}:{caller['line']} {caller['symbol']} calls {item['name']} with {arg_count} positional args; expected {min_args}"
                        + (f"-{max_args}" if isinstance(max_args, int) and max_args != min_args else "")
                    )
                for expectation in caller.get("return_expectations", []) or []:
                    if not isinstance(expectation, dict):
                        continue
                    expected_shape = str(expectation.get("shape") or "")
                    if expected_shape and not self._caller_return_expectation_compatible(expected_shape, returns, shapes):
                        diagnostics.append(
                            f"{caller['path']}:{expectation.get('line', caller['line'])} {caller['symbol']} expects {item['name']} return as {expected_shape}; "
                            f"contract is -> {returns} shape {','.join(shapes)}"
                        )
        output = "\n".join(diagnostics[: max(1, int(limit))]) if diagnostics else f"contract ok: {len(checked)} Python function(s)"
        return {
            "ok": not diagnostics,
            "tool": "contract_check",
            "checked": checked,
            "diagnostics": diagnostics[: max(1, int(limit))],
            "output": output,
            "summary": output,
        }

    def call_graph(self, path: str = ".", symbol: str | None = None, limit: int = 40) -> dict[str, Any]:
        self._check_interrupted()
        base = self.resolve_path(path, allow_missing=False)
        definitions: dict[str, dict[str, Any]] = {}
        calls_by_def: dict[str, set[str]] = {}
        imports_by_file: dict[str, list[str]] = {}
        for file_path in self._iter_code_files(base, limit=500):
            if file_path.suffix.lower() != ".py":
                continue
            rel = self.relative_label(file_path)
            text = file_path.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(self._python_parse_text(text), filename=rel)
            except SyntaxError:
                continue
            imports_by_file[rel] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports_by_file[rel].extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports_by_file[rel].extend(f"{node.module}.{alias.name}" for alias in node.names)

            class Visitor(ast.NodeVisitor):
                def __init__(self) -> None:
                    self.stack: list[str] = []

                def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
                    self._visit_def(node, "function")

                def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
                    self._visit_def(node, "function")

                def visit_ClassDef(self, node: ast.ClassDef) -> Any:
                    self._visit_def(node, "class")

                def _visit_def(self, node: ast.AST, kind: str) -> None:
                    name = str(getattr(node, "name", ""))
                    qualname = ".".join([*self.stack, name])
                    key = f"{rel}:{qualname}"
                    definitions[key] = {"path": rel, "symbol": qualname, "kind": kind, "line": int(getattr(node, "lineno", 1))}
                    previous = list(self.stack)
                    self.stack.append(name)
                    calls: set[str] = set()
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            func = child.func
                            if isinstance(func, ast.Name):
                                calls.add(func.id)
                            elif isinstance(func, ast.Attribute):
                                calls.add(func.attr)
                    calls_by_def[key] = calls
                    self.generic_visit(node)
                    self.stack = previous

            Visitor().visit(tree)
        target_name = (symbol or "").strip()
        if target_name:
            matched_keys = [key for key, item in definitions.items() if item["symbol"] == target_name or item["symbol"].endswith(f".{target_name}") or key.endswith(f":{target_name}")]
        else:
            matched_keys = list(definitions)[: max(1, int(limit))]
        lines: list[str] = []
        for key in matched_keys[: max(1, int(limit))]:
            item = definitions[key]
            callees = sorted(calls_by_def.get(key, set()))[:12]
            target_leaf = str(item["symbol"]).split(".")[-1]
            callers = [
                definitions[other]
                for other, calls in calls_by_def.items()
                if other != key and target_leaf in calls
            ][:12]
            affected_tests: list[str] = []
            for file_path in imports_by_file:
                if "test" not in Path(file_path).name.lower() and "tests" not in Path(file_path).parts:
                    continue
                full_path = self.workspace_root / file_path
                if full_path.exists() and target_leaf in full_path.read_text(encoding="utf-8", errors="replace"):
                    affected_tests.append(file_path)
                    if len(affected_tests) >= 8:
                        break
            lines.append(f"{item['path']}:{item['line']} {item['kind']} {item['symbol']}")
            lines.append(f"  callees: {', '.join(callees) if callees else '(none)'}")
            lines.append("  callers: " + (", ".join(f"{caller['path']}:{caller['line']} {caller['symbol']}" for caller in callers) if callers else "(none)"))
            lines.append(f"  affected_tests: {', '.join(affected_tests) if affected_tests else '(none)'}")
        return {
            "ok": True,
            "tool": "call_graph",
            "path": self.relative_label(base),
            "symbol": target_name,
            "count": len(matched_keys),
            "output": "\n".join(lines) if lines else "(no call graph entries found)",
        }

    def _package_manager_for(self, base: Path) -> str:
        root = base if base.is_dir() else base.parent
        if (root / "pnpm-lock.yaml").exists():
            return "pnpm"
        if (root / "yarn.lock").exists():
            return "yarn"
        return "npm"

    def _project_root_for(self, base: Path) -> Path:
        start = base if base.is_dir() else base.parent
        markers = {"pyproject.toml", "pytest.ini", "package.json", "go.mod", "Cargo.toml", "build.gradle", "settings.gradle", "CMakeLists.txt"}
        for candidate in [start, *start.parents]:
            if candidate == self.workspace_root or self.workspace_root in candidate.parents:
                if any((candidate / marker).exists() for marker in markers):
                    return candidate
                if candidate == self.workspace_root:
                    break
        return start

    def _available_command(self, command: str) -> bool:
        try:
            argv = shlex.split(command, posix=os.name != "nt")
        except ValueError:
            return False
        if not argv:
            return False
        executable = argv[0]
        if executable.startswith(("./", ".\\")):
            return (self.workspace_root / executable).exists()
        return shutil.which(executable) is not None or Path(executable).exists()

    def _read_toml(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            return {}

    def _toml_tool_section(self, payload: dict[str, Any], name: str) -> bool:
        tool = payload.get("tool") if isinstance(payload, dict) else None
        return isinstance(tool, dict) and isinstance(tool.get(name), dict)

    def _ini_has_section(self, path: Path, prefixes: tuple[str, ...]) -> bool:
        if not path.exists():
            return False
        parser = configparser.ConfigParser()
        try:
            parser.read(path, encoding="utf-8")
        except configparser.Error:
            return False
        return any(section == prefix or section.startswith(prefix + ":") for section in parser.sections() for prefix in prefixes)

    def discover_validators(self, path: str = ".", limit: int = 12) -> dict[str, Any]:
        self._check_interrupted()
        base = self.resolve_path(path, allow_missing=False)
        root = self._project_root_for(base)
        validators: list[dict[str, Any]] = []

        def add(kind: str, lang: str, command: str, reason: str) -> None:
            validators.append({"kind": kind, "lang": lang, "command": command, "available": self._available_command(command), "reason": reason})

        pyproject = self._read_toml(root / "pyproject.toml")
        repo_files = self._iter_repo_files(root, limit=50000)
        suffixes = {file_path.suffix.lower() for file_path in repo_files}
        file_names = [file_path.name.lower() for file_path in repo_files]
        rel_paths = [self.relative_label(file_path).replace("\\", "/").lower() for file_path in repo_files]
        python_files = ".py" in suffixes
        python_tests = any(
            name.startswith("test_") or name.endswith("_test.py") or "/tests/" in f"/{rel}/"
            for name, rel in zip(file_names, rel_paths)
        ) or (root / "tests").exists()
        pytest_config = (
            (root / "pytest.ini").exists()
            or (root / "pytest.toml").exists()
            or self._toml_tool_section(pyproject, "pytest")
            or self._toml_tool_section(pyproject, "pytest.ini_options")
        )
        if python_files or pyproject or pytest_config:
            add("syntax", "python", f"{sys.executable} -m py_compile", "Python files found; lint_typecheck does exact syntax checks internally.")
            if pytest_config:
                add("collect", "python", f"{sys.executable} -m pytest --collect-only -q", "pytest config found.")
                add("test", "python", f"{sys.executable} -m pytest", "pytest config found.")
            if python_tests:
                add("test", "python", f"{sys.executable} -m unittest discover -s tests -v", "Python unittest discovery.")
            if (
                (root / "ruff.toml").exists()
                or (root / ".ruff.toml").exists()
                or self._toml_tool_section(pyproject, "ruff")
                or shutil.which("ruff")
            ):
                add("lint", "python", "ruff check .", "ruff config or executable found.")
            if (
                (root / "mypy.ini").exists()
                or (root / ".mypy.ini").exists()
                or self._ini_has_section(root / "setup.cfg", ("mypy",))
                or self._toml_tool_section(pyproject, "mypy")
            ):
                add("typecheck", "python", "mypy .", "mypy config found.")
            if (root / "pyrightconfig.json").exists() or self._toml_tool_section(pyproject, "pyright"):
                add("typecheck", "python", "pyright", "pyright config found.")
            if (root / "tox.ini").exists() or self._toml_tool_section(pyproject, "tox"):
                add("test", "python", "tox", "tox config found.")
            if (root / "noxfile.py").exists():
                add("test", "python", "nox", "noxfile.py found.")
        package_json = root / "package.json"
        if package_json.exists():
            manager = self._package_manager_for(root)
            try:
                package = json.loads(package_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                package = {}
            scripts = package.get("scripts") if isinstance(package, dict) else {}
            if isinstance(scripts, dict):
                for name, kind in (("typecheck", "typecheck"), ("lint", "lint"), ("test", "test")):
                    if name in scripts:
                        add(kind, "javascript", f"{manager} run {name}" if name != "test" else f"{manager} test", f"package.json script: {name}")
            if (root / "tsconfig.json").exists():
                add("typecheck", "typescript", "tsc --noEmit", "tsconfig.json found.")
        if (root / "go.mod").exists() or ".go" in suffixes:
            add("test", "go", "go test ./...", "Go module or source files found.")
        if (root / "Cargo.toml").exists() or ".rs" in suffixes:
            add("check", "rust", "cargo check", "Cargo project or Rust source files found.")
            add("test", "rust", "cargo test", "Cargo project or Rust source files found.")
        gradlew = "gradlew.bat" if os.name == "nt" else "./gradlew"
        if (root / "build.gradle").exists() or (root / "settings.gradle").exists() or (root / gradlew).exists() or ".java" in suffixes:
            command = gradlew + " test" if (root / gradlew).exists() else "gradle test"
            add("test", "java", command, "Gradle/Java project detected.")
        if (root / "CMakeLists.txt").exists() or ".cpp" in suffixes or ".c" in suffixes:
            if (root / "build").exists():
                add("test", "cpp", "ctest --test-dir build --output-on-failure", "CMake build directory found.")
            add("setup", "cpp", "cmake -S . -B build", "CMake project detected; creates/updates build dir if run.")
        selected = validators[: max(1, int(limit))]
        lines = [f"{item['kind']} {item['lang']}: {item['command']} available={item['available']} reason={item['reason']}" for item in selected]
        return {
            "ok": True,
            "tool": "discover_validators",
            "path": self.relative_label(root),
            "count": len(selected),
            "validators": selected,
            "output": "\n".join(lines) if lines else "(no validators discovered)",
        }

    def diagnose_dependency_error(self, output: str, path: str = ".") -> dict[str, Any]:
        text = str(output or "")
        base = self.resolve_path(path, allow_missing=True)
        root = self._project_root_for(base if base.exists() else self.workspace_root)
        error_class = self.classify_error(text)
        missing = self._missing_dependency_name(text)
        command_match = re.search(r"(?:command not found|not recognized as (?:an internal|the name)|executable not found):?\s*['\"]?([A-Za-z0-9_.-]+)", text, flags=re.IGNORECASE)
        command = command_match.group(1) if command_match else None
        import_match = re.search(r"cannot import name ['\"]?([^'\"\s]+)", text, flags=re.IGNORECASE)
        imported = import_match.group(1) if import_match else None
        suggested_paths: list[str] = []
        if error_class in {"path_missing", "cwd_git"}:
            path_match = re.search(r"(?:No such file or directory|FileNotFoundError|Path does not exist):?\s*['\"]?([^'\"\n]+)", text, flags=re.IGNORECASE)
            if path_match:
                suggested_paths = self._nearest_existing_paths(path_match.group(1))
        validators = self.discover_validators(self.relative_label(root), limit=6)
        managers = []
        if (root / "package.json").exists():
            managers.append(self._package_manager_for(root))
        if (root / "pyproject.toml").exists() or any(file_path.name.startswith("requirements") and file_path.suffix == ".txt" for file_path in self._iter_repo_files(root, limit=50000)):
            managers.append("pip")
        if (root / "Cargo.toml").exists():
            managers.append("cargo")
        if (root / "go.mod").exists():
            managers.append("go")
        facts = {
            "error_class": error_class,
            "missing_dependency": missing or command or imported,
            "missing_command": command,
            "missing_import": imported,
            "package_managers": sorted(set(managers)),
            "suggested_paths": suggested_paths,
            "validator_commands": [item.get("command") for item in validators.get("validators", []) if isinstance(item, dict)][:6],
        }
        lines = [f"{key}={value}" for key, value in facts.items() if value]
        return {
            "ok": True,
            "tool": "diagnose_dependency_error",
            **facts,
            "summary": f"Classified failure as {error_class}. Do not install automatically; report or use an available validator.",
            "output": "\n".join(lines) if lines else f"error_class={error_class}",
        }

    def lint_typecheck(self, paths: str | list[str] = ".", command: str | None = None, timeout: int = 120) -> dict[str, Any]:
        if isinstance(command, str) and command.strip():
            result = self.run_shell(command.strip(), timeout=timeout)
            result["tool"] = "lint_typecheck"
            return result
        raw_paths = paths if isinstance(paths, list) else [paths]
        checked: list[str] = []
        diagnostics: list[str] = []
        for raw_path in raw_paths:
            base = self.resolve_path(str(raw_path), allow_missing=False)
            files = self._iter_code_files(base, limit=50000)
            for file_path in files:
                rel = self.relative_label(file_path)
                checked.append(rel)
                text = file_path.read_text(encoding="utf-8", errors="replace")
                if file_path.suffix.lower() == ".py":
                    diagnostic = self._python_syntax_diagnostic(file_path, text)
                    if diagnostic:
                        diagnostics.append(diagnostic)
                elif file_path.suffix.lower() in {".js", ".jsx"} and shutil.which("node"):
                    completed = self._run_process(["node", "--check", str(file_path)], cwd=self.workspace_root, timeout=timeout, shell=False)
                    if completed.returncode != 0:
                        diagnostics.append(self._truncate_text(self._collect_process_output(completed), limit=500))
        return {
            "ok": not diagnostics,
            "tool": "lint_typecheck",
            "checked": checked,
            "diagnostics": diagnostics,
            "output": "\n".join(diagnostics) if diagnostics else f"syntax ok: {len(checked)} code file(s)",
        }

    def _test_module_for_path(self, test_path: Path) -> str:
        rel = self.relative_label(test_path)
        if rel.endswith(".py"):
            rel = rel[:-3]
        return rel.replace("/", ".").replace("\\", ".")

    def _source_modules_for_path(self, source_path: Path) -> set[str]:
        rel = self.relative_label(source_path)
        without_suffix = rel[:-3] if rel.endswith(".py") else rel
        modules = {without_suffix.replace("/", ".").replace("\\", ".")}
        if without_suffix.startswith("src/"):
            modules.add(without_suffix[4:].replace("/", ".").replace("\\", "."))
        return modules

    def _iter_python_test_files(self) -> list[Path]:
        candidates: list[Path] = []
        for file_path in self._iter_code_files(self.workspace_root, limit=50000):
            if file_path.suffix.lower() != ".py":
                continue
            rel = self.relative_label(file_path).replace("\\", "/")
            name = file_path.name.lower()
            if name.startswith("test_") or name.endswith("_test.py") or "/tests/" in rel:
                candidates.append(file_path)
        return candidates

    def _path_looks_like_test(self, path: Path) -> bool:
        rel = self.relative_label(path).replace("\\", "/").lower()
        name = path.name.lower()
        return name.startswith("test_") or name.endswith("_test.py") or "/tests/" in rel

    def _test_matches_source(self, test_path: Path, source_path: Path, symbols: set[str]) -> tuple[int, list[str]]:
        text = test_path.read_text(encoding="utf-8", errors="replace")
        score = 0
        reasons: list[str] = []
        imported_paths = {str(item.get("path", "")) for item in self._python_import_targets(test_path)}
        rel_source = self.relative_label(source_path)
        if rel_source in imported_paths:
            score += 6
            reasons.append(f"imports {rel_source}")
        source_modules = self._source_modules_for_path(source_path)
        for module in source_modules:
            if re.search(rf"\b(?:from|import)\s+{re.escape(module)}\b", text):
                score += 5
                reasons.append(f"imports module {module}")
                break
        source_stem = source_path.stem
        if source_stem and source_stem.lower() in test_path.stem.lower():
            score += 3
            reasons.append(f"filename matches {source_stem}")
        matched_symbols = sorted(symbol for symbol in symbols if symbol and re.search(rf"\b{re.escape(symbol)}\b", text))
        if matched_symbols:
            score += min(4, len(matched_symbols) * 2)
            reasons.append("mentions " + ", ".join(matched_symbols[:4]))
        return score, reasons

    def _targeted_unittest_command(self, test_path: Path, symbols: set[str]) -> str:
        test_dir = self.relative_label(test_path.parent)
        return f"{sys.executable} -m unittest discover -s {test_dir} -p {test_path.name}"

    def select_tests(
        self,
        changed_files: str | list[str],
        changed_symbols: str | list[str] | None = None,
        limit: int = 8,
    ) -> dict[str, Any]:
        self._check_interrupted()
        raw_files = [changed_files] if isinstance(changed_files, str) else list(changed_files or [])
        raw_symbols = [changed_symbols] if isinstance(changed_symbols, str) else list(changed_symbols or [])
        symbols = {str(symbol).strip() for symbol in raw_symbols if str(symbol).strip()}
        source_files: list[Path] = []
        non_python_files: list[Path] = []
        for raw_path in raw_files:
            try:
                path = self.resolve_path(str(raw_path), allow_missing=False)
            except Exception:
                continue
            if path.suffix.lower() == ".py" and path.is_file() and not self._path_looks_like_test(path):
                source_files.append(path)
            elif path.suffix.lower() in CODE_FILE_SUFFIXES and path.is_file():
                non_python_files.append(path)
        if not source_files:
            if non_python_files:
                root = self._project_root_for(non_python_files[0])
                validators = self.discover_validators(self.relative_label(root), limit=8)
                commands = [
                    str(item.get("command"))
                    for item in validators.get("validators", [])
                    if isinstance(item, dict) and item.get("kind") == "test" and item.get("command")
                ][: max(1, int(limit))]
                rows = [
                    {
                        "path": self.relative_label(path),
                        "command": commands[0] if commands else "",
                        "score": 1,
                        "reason": "language-level validator discovery",
                    }
                    for path in non_python_files[: max(1, int(limit))]
                ]
                return {
                    "ok": True,
                    "tool": "select_tests",
                    "confidence": "low" if not commands else "medium",
                    "changed_files": [self.relative_label(path) for path in non_python_files],
                    "changed_symbols": sorted(symbols),
                    "test_commands": commands,
                    "tests": rows,
                    "summary": "Selected language-level test commands from discover_validators." if commands else "No targeted tests found; use configured run_test.",
                    "output": "\n".join(commands) if commands else "(no targeted tests found)",
                }
            return {"ok": False, "tool": "select_tests", "summary": "No changed source files to map to tests."}
        ranked: list[tuple[int, Path, list[str]]] = []
        for test_path in self._iter_python_test_files():
            score = 0
            reasons: list[str] = []
            for source_path in source_files:
                item_score, item_reasons = self._test_matches_source(test_path, source_path, symbols)
                score += item_score
                reasons.extend(item_reasons)
            if score > 0:
                ranked.append((score, test_path, reasons))
        ranked.sort(key=lambda item: (-item[0], self.relative_label(item[1])))
        selected = ranked[: max(1, int(limit))]
        if not selected:
            return {
                "ok": True,
                "tool": "select_tests",
                "confidence": "low",
                "test_commands": [],
                "summary": "No targeted tests found; use configured run_test.",
                "output": "(no targeted tests found)",
            }
        commands: list[str] = []
        rows: list[dict[str, Any]] = []
        for score, test_path, reasons in selected:
            command = self._targeted_unittest_command(test_path, symbols)
            commands.append(command)
            rows.append(
                {
                    "path": self.relative_label(test_path),
                    "command": command,
                    "score": score,
                    "reason": "; ".join(dict.fromkeys(reasons)),
                }
            )
        confidence = "high" if selected[0][0] >= 6 else "medium"
        lines = [f"{row['path']} score={row['score']} command={row['command']} reason={row['reason']}" for row in rows]
        return {
            "ok": True,
            "tool": "select_tests",
            "confidence": confidence,
            "changed_files": [self.relative_label(path) for path in source_files],
            "changed_symbols": sorted(symbols),
            "test_commands": commands,
            "tests": rows,
            "output": "\n".join(lines),
        }

    def _first_non_empty_indent(self, content: str) -> str:
        for line in content.splitlines():
            if line.strip():
                return line[: len(line) - len(line.lstrip())]
        return ""

    def _align_symbol_replacement_indent(self, content: str, existing_first_line: str) -> str:
        target_indent = existing_first_line[: len(existing_first_line) - len(existing_first_line.lstrip())]
        if not target_indent:
            return content
        current_indent = self._first_non_empty_indent(content)
        if len(current_indent) >= len(target_indent):
            return content
        lines = content.splitlines(keepends=True)
        return "".join(target_indent + line if line.strip() else line for line in lines)

    def _python_replacement_kind_error(self, kind: str, replacement: str) -> str | None:
        stripped = replacement.lstrip("\ufeff").lstrip()
        if kind in {"function", "method"} and not stripped.startswith(("def ", "async def ")):
            return "Python function/method replacements must include full def source, not a call or expression."
        if kind == "class" and not stripped.startswith("class "):
            return "Python class replacements must include full class source."
        return None

    def replace_symbol(self, path: str, symbol: str, content: str) -> dict[str, Any]:
        self._check_interrupted()
        target = self.resolve_path(path, allow_missing=False)
        relative_path = self.relative_label(target)
        if target.is_dir():
            return {"ok": False, "tool": "replace_symbol", "summary": f"{path} is a directory."}
        if not self._is_code_file(target):
            return {"ok": False, "tool": "replace_symbol", "summary": f"{path} is not a supported code file."}
        denied = self._mutation_denied_path(target)
        if denied:
            return {"ok": False, "tool": "replace_symbol", "path": relative_path, "summary": denied}
        if self._contains_omitted_context_marker(content):
            return {
                "ok": False,
                "tool": "replace_symbol",
                "path": relative_path,
                "summary": "Refusing to use omitted-context marker as symbol replacement text. Read the file or reconstruct exact text instead.",
            }
        symbols, original, _ = self._code_symbols(target)
        matches = self._symbol_matches(symbols, symbol)
        if not matches:
            return {"ok": False, "tool": "replace_symbol", "path": relative_path, "summary": f"Symbol not found: {symbol}"}
        if len(matches) > 1:
            rendered = "\n".join(f"{item['start']}-{item['end']} {item['kind']} {item['qualname']}" for item in matches[:20])
            return {"ok": False, "tool": "replace_symbol", "path": relative_path, "summary": f"Ambiguous symbol: {symbol}", "matches": rendered}
        found = matches[0]
        lines = original.splitlines(keepends=True)
        start = int(found["start"])
        end = int(found["end"])
        replacement = self._strip_read_file_line_prefixes(content)
        normalization = None
        if target.suffix.lower() == ".py":
            replacement, normalization = self._normalize_python_write_content(target, replacement)
            kind_error = self._python_replacement_kind_error(str(found["kind"]), replacement)
            if kind_error:
                return {
                    "ok": False,
                    "tool": "replace_symbol",
                    "path": relative_path,
                    "symbol": found["qualname"],
                    "kind": found["kind"],
                    "summary": kind_error,
                }
            replacement = self._align_symbol_replacement_indent(replacement, lines[start - 1])
            if original.startswith("\ufeff") and start == 1 and not replacement.startswith("\ufeff"):
                replacement = "\ufeff" + replacement
        if replacement and not replacement.endswith(("\n", "\r")):
            replacement += "\n"
        updated = "".join(lines[: start - 1]) + replacement + "".join(lines[end:])
        diagnostic = self._python_syntax_diagnostic(target, updated)
        if diagnostic is not None:
            return {
                "ok": False,
                "tool": "replace_symbol",
                "path": relative_path,
                "symbol": found["qualname"],
                "start": start,
                "end": end,
                "syntax_ok": False,
                "diagnostic": diagnostic,
                "summary": f"Refusing replace_symbol because the updated file would be invalid. {diagnostic}",
            }
        preview = self._diff_preview(relative_path, original, updated)
        approved, reason = self._approve_mutation(f"Replace symbol {found['qualname']} in {relative_path}?", preview)
        if not approved:
            return {"ok": False, "tool": "replace_symbol", "path": relative_path, "symbol": found["qualname"], "summary": reason}
        target.write_text(updated, encoding="utf-8")
        result = {
            "ok": True,
            "tool": "replace_symbol",
            "path": relative_path,
            "symbol": found["qualname"],
            "kind": found["kind"],
            "start": start,
            "end": end,
            "syntax_ok": True if target.suffix.lower() == ".py" else None,
            "summary": f"Replaced {found['kind']} {found['qualname']} in {relative_path}.",
            "diff": preview,
        }
        if result["syntax_ok"] is None:
            result.pop("syntax_ok")
        if normalization:
            result["normalized"] = normalization
            result["summary"] += f" {normalization}"
        return result

    def replace_symbols(self, path: str, replacements: list[dict[str, Any]]) -> dict[str, Any]:
        self._check_interrupted()
        target = self.resolve_path(path, allow_missing=False)
        relative_path = self.relative_label(target)
        if target.is_dir():
            return {"ok": False, "tool": "replace_symbols", "summary": f"{path} is a directory."}
        if not self._is_code_file(target):
            return {"ok": False, "tool": "replace_symbols", "summary": f"{path} is not a supported code file."}
        denied = self._mutation_denied_path(target)
        if denied:
            return {"ok": False, "tool": "replace_symbols", "path": relative_path, "summary": denied}
        if not isinstance(replacements, list) or not replacements:
            return {"ok": False, "tool": "replace_symbols", "path": relative_path, "summary": "replace_symbols requires a non-empty replacements list."}
        symbols, original, _ = self._code_symbols(target)
        lines = original.splitlines(keepends=True)
        prepared: list[dict[str, Any]] = []
        seen_ranges: set[tuple[int, int]] = set()
        normalizations: list[str] = []
        for index, item in enumerate(replacements, start=1):
            if not isinstance(item, dict):
                return {"ok": False, "tool": "replace_symbols", "path": relative_path, "summary": f'Replacement {index} must be an object shaped {{"symbol":"name","content":"full def/class source"}}.'}
            symbol = item.get("symbol")
            content = item.get("content")
            if not isinstance(symbol, str) or not symbol.strip() or not isinstance(content, str):
                return {"ok": False, "tool": "replace_symbols", "path": relative_path, "summary": f'Replacement {index} requires string symbol and content fields, e.g. {{"symbol":"add","content":"def add(a, b):\\n    return a + b"}}.'}
            if self._contains_omitted_context_marker(content):
                return {
                    "ok": False,
                    "tool": "replace_symbols",
                    "path": relative_path,
                    "summary": "Refusing to use omitted-context marker as symbol replacement text. Read the file or reconstruct exact text instead.",
                }
            matches = self._symbol_matches(symbols, symbol)
            if not matches:
                return {"ok": False, "tool": "replace_symbols", "path": relative_path, "summary": f"Symbol not found: {symbol}"}
            if len(matches) > 1:
                rendered = "\n".join(f"{entry['start']}-{entry['end']} {entry['kind']} {entry['qualname']}" for entry in matches[:20])
                return {"ok": False, "tool": "replace_symbols", "path": relative_path, "summary": f"Ambiguous symbol: {symbol}", "matches": rendered}
            found = matches[0]
            start = int(found["start"])
            end = int(found["end"])
            line_range = (start, end)
            if line_range in seen_ranges:
                return {"ok": False, "tool": "replace_symbols", "path": relative_path, "summary": f"Duplicate replacement for symbol range {start}-{end}."}
            seen_ranges.add(line_range)
            replacement = self._strip_read_file_line_prefixes(content)
            normalization = None
            if target.suffix.lower() == ".py":
                replacement, normalization = self._normalize_python_write_content(target, replacement)
                kind_error = self._python_replacement_kind_error(str(found["kind"]), replacement)
                if kind_error:
                    return {
                        "ok": False,
                        "tool": "replace_symbols",
                        "path": relative_path,
                        "symbol": found["qualname"],
                        "kind": found["kind"],
                        "summary": kind_error,
                    }
                replacement = self._align_symbol_replacement_indent(replacement, lines[start - 1])
                if original.startswith("\ufeff") and start == 1 and not replacement.startswith("\ufeff"):
                    replacement = "\ufeff" + replacement
            if replacement and not replacement.endswith(("\n", "\r")):
                replacement += "\n"
            if normalization:
                normalizations.append(normalization)
            prepared.append({"found": found, "start": start, "end": end, "replacement": replacement})
        updated_lines = list(lines)
        for item in sorted(prepared, key=lambda entry: int(entry["start"]), reverse=True):
            updated_lines[int(item["start"]) - 1 : int(item["end"])] = str(item["replacement"]).splitlines(keepends=True)
        updated = "".join(updated_lines)
        diagnostic = self._python_syntax_diagnostic(target, updated)
        if diagnostic is not None:
            return {
                "ok": False,
                "tool": "replace_symbols",
                "path": relative_path,
                "count": len(prepared),
                "syntax_ok": False,
                "diagnostic": diagnostic,
                "summary": f"Refusing replace_symbols because the updated file would be invalid. {diagnostic}",
            }
        preview = self._diff_preview(relative_path, original, updated)
        names = [str(item["found"]["qualname"]) for item in prepared]
        approved, reason = self._approve_mutation(f"Replace {len(prepared)} symbols in {relative_path}?", preview)
        if not approved:
            return {"ok": False, "tool": "replace_symbols", "path": relative_path, "summary": reason}
        target.write_text(updated, encoding="utf-8")
        result = {
            "ok": True,
            "tool": "replace_symbols",
            "path": relative_path,
            "count": len(prepared),
            "symbols": names,
            "syntax_ok": True if target.suffix.lower() == ".py" else None,
            "summary": f"Replaced {len(prepared)} symbol(s) in {relative_path}: {', '.join(names[:8])}.",
            "diff": preview,
        }
        if result["syntax_ok"] is None:
            result.pop("syntax_ok")
        if normalizations:
            result["normalized"] = "; ".join(dict.fromkeys(normalizations))
            result["summary"] += f" {result['normalized']}"
        return result

    def _operation_payload(self, operation: dict[str, Any] | str) -> dict[str, Any] | None:
        if isinstance(operation, dict):
            return operation
        if isinstance(operation, str):
            try:
                parsed = json.loads(operation)
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, dict) else None
        return None

    def _insert_import_statement(self, original: str, statement: str) -> str:
        lines = original.splitlines(keepends=True)
        insert_at = 0
        if lines and lines[0].startswith("#!"):
            insert_at = 1
        try:
            tree = ast.parse(self._python_parse_text(original))
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
        if not statement.endswith("\n"):
            statement += "\n"
        return "".join(lines[:insert_at]) + statement + "".join(lines[insert_at:])

    def _delete_symbol_text(self, target: Path, symbol: str) -> tuple[str, str, dict[str, Any] | None]:
        symbols, original, _ = self._code_symbols(target)
        matches = self._symbol_matches(symbols, symbol)
        if not matches:
            return original, f"Symbol not found: {symbol}", None
        if len(matches) > 1:
            rendered = "\n".join(f"{item['start']}-{item['end']} {item['kind']} {item['qualname']}" for item in matches[:20])
            return original, f"Ambiguous symbol: {symbol}\n{rendered}", None
        found = matches[0]
        lines = original.splitlines(keepends=True)
        start = int(found["start"])
        end = int(found["end"])
        while end < len(lines) and not lines[end].strip():
            end += 1
        updated = "".join(lines[: start - 1]) + "".join(lines[end:])
        return updated, "", found

    def _matched_symbol(self, target: Path, symbol: str) -> tuple[dict[str, Any] | None, str]:
        symbols, _, _ = self._code_symbols(target)
        matches = self._symbol_matches(symbols, symbol)
        if not matches:
            return None, f"Symbol not found: {symbol}"
        if len(matches) > 1:
            rendered = "\n".join(f"{item['start']}-{item['end']} {item['kind']} {item['qualname']}" for item in matches[:20])
            return None, f"Ambiguous symbol: {symbol}\n{rendered}"
        return matches[0], ""

    def _python_function_node(self, target: Path, symbol: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        text = target.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(text))
        except SyntaxError:
            return None
        wanted = symbol.split(".")[-1]
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == wanted:
                return node
        return None

    def _apply_file_update(self, target: Path, original: str, updated: str, prompt: str, *, op: str) -> dict[str, Any]:
        relative_path = self.relative_label(target)
        denied = self._mutation_denied_path(target)
        if denied:
            return {"ok": False, "tool": "apply_structured_edit", "path": relative_path, "summary": denied}
        diagnostic = self._python_syntax_diagnostic(target, updated)
        if diagnostic:
            return {"ok": False, "tool": "apply_structured_edit", "path": relative_path, "op": op, "syntax_ok": False, "diagnostic": diagnostic, "summary": diagnostic}
        preview = self._diff_preview(relative_path, original, updated)
        approved, reason = self._approve_mutation(prompt, preview)
        if not approved:
            return {"ok": False, "tool": "apply_structured_edit", "path": relative_path, "op": op, "summary": reason}
        target.write_text(updated, encoding="utf-8")
        return {"ok": True, "tool": "apply_structured_edit", "path": relative_path, "op": op, "syntax_ok": True if target.suffix.lower() == ".py" else None, "summary": f"Applied {op} to {relative_path}.", "diff": preview}

    def _replace_python_function_body(self, target: Path, symbol: str, body: str) -> dict[str, Any]:
        relative_path = self.relative_label(target)
        if target.suffix.lower() != ".py":
            return {"ok": False, "tool": "apply_structured_edit", "path": relative_path, "summary": "replace_function_body supports Python files only."}
        node = self._python_function_node(target, symbol)
        if node is None or not node.body:
            return {"ok": False, "tool": "apply_structured_edit", "path": relative_path, "summary": f"Function not found: {symbol}"}
        original = target.read_text(encoding="utf-8", errors="replace")
        lines = original.splitlines(keepends=True)
        first_body = node.body[0]
        last_body = node.body[-1]
        start = int(getattr(first_body, "lineno", getattr(node, "lineno", 1)))
        end = int(getattr(last_body, "end_lineno", getattr(last_body, "lineno", start)))
        header_indent = lines[int(getattr(node, "lineno", 1)) - 1][: len(lines[int(getattr(node, "lineno", 1)) - 1]) - len(lines[int(getattr(node, "lineno", 1)) - 1].lstrip())]
        body_indent = header_indent + "    "
        normalized = textwrap.dedent(body).strip("\n")
        replacement_lines = ["pass"] if not normalized.strip() else normalized.splitlines()
        replacement = "".join(body_indent + line.rstrip() + "\n" if line.strip() else "\n" for line in replacement_lines)
        updated = "".join(lines[: start - 1]) + replacement + "".join(lines[end:])
        return self._apply_file_update(target, original, updated, f"Replace body of {symbol} in {relative_path}?", op="replace_function_body")

    def _change_python_signature(self, target: Path, symbol: str, signature: str) -> dict[str, Any]:
        relative_path = self.relative_label(target)
        if target.suffix.lower() != ".py":
            return {"ok": False, "tool": "apply_structured_edit", "path": relative_path, "summary": "change_signature supports Python files only."}
        found, error = self._matched_symbol(target, symbol)
        if found is None:
            return {"ok": False, "tool": "apply_structured_edit", "path": relative_path, "summary": error}
        original = target.read_text(encoding="utf-8", errors="replace")
        lines = original.splitlines(keepends=True)
        start = int(found["start"])
        current = lines[start - 1]
        indent = current[: len(current) - len(current.lstrip())]
        clean = signature.strip()
        if not clean.startswith(("def ", "async def ")):
            name = str(found.get("name") or symbol.split(".")[-1])
            clean = f"def {name}{clean if clean.startswith('(') else '(' + clean + ')'}:"
        if not clean.endswith("\n"):
            clean += "\n"
        lines[start - 1] = indent + clean.lstrip()
        updated = "".join(lines)
        return self._apply_file_update(target, original, updated, f"Change signature of {symbol} in {relative_path}?", op="change_signature")

    def _rename_symbol_project(self, base: Path, old: str, new: str) -> dict[str, Any]:
        if not old or not new or not re.match(r"^[A-Za-z_]\w*$", old) or not re.match(r"^[A-Za-z_]\w*$", new):
            return {"ok": False, "tool": "apply_structured_edit", "summary": "rename_symbol_project requires valid old/new identifiers."}
        updates: list[tuple[Path, str, str]] = []
        already_renamed_files: list[str] = []
        for file_path in self._iter_code_files(base, limit=1000):
            original = file_path.read_text(encoding="utf-8", errors="replace")
            updated, count = re.subn(rf"\b{re.escape(old)}\b", new, original)
            if count:
                diagnostic = self._python_syntax_diagnostic(file_path, updated)
                if diagnostic:
                    return {"ok": False, "tool": "apply_structured_edit", "path": self.relative_label(file_path), "syntax_ok": False, "diagnostic": diagnostic, "summary": diagnostic}
                updates.append((file_path, original, updated))
            elif re.search(rf"\b{re.escape(new)}\b", original):
                already_renamed_files.append(self.relative_label(file_path))
        if not updates:
            if already_renamed_files:
                return {
                    "ok": True,
                    "tool": "apply_structured_edit",
                    "path": self.relative_label(base),
                    "op": "rename_symbol_project",
                    "count": 0,
                    "summary": f"Identifier already renamed from {old} to {new} in {len(already_renamed_files)} file(s).",
                    "files": already_renamed_files[:20],
                }
            return {"ok": False, "tool": "apply_structured_edit", "summary": f"Identifier not found: {old}"}
        preview = "\n".join(self._diff_preview(self.relative_label(path), original, updated) for path, original, updated in updates[:12])
        approved, reason = self._approve_mutation(f"Rename {old} to {new} in {len(updates)} file(s)?", preview)
        if not approved:
            return {"ok": False, "tool": "apply_structured_edit", "summary": reason}
        for path, _, updated in updates:
            path.write_text(updated, encoding="utf-8")
        return {
            "ok": True,
            "tool": "apply_structured_edit",
            "path": self.relative_label(base),
            "op": "rename_symbol_project",
            "count": len(updates),
            "summary": f"Renamed {old} to {new} in {len(updates)} file(s).",
            "diff": preview,
        }

    def apply_structured_edit(self, operation: dict[str, Any] | str) -> dict[str, Any]:
        payload = self._operation_payload(operation)
        if payload is None:
            return {"ok": False, "tool": "apply_structured_edit", "summary": "operation must be a JSON object."}
        op = str(payload.get("op") or payload.get("operation") or "").strip().lower()
        if op == "replace_function_body":
            return self._replace_python_function_body(
                self.resolve_path(str(payload.get("path") or ""), allow_missing=False),
                str(payload.get("symbol") or payload.get("name") or ""),
                str(payload.get("body") or payload.get("content") or ""),
            )
        if op == "change_signature":
            return self._change_python_signature(
                self.resolve_path(str(payload.get("path") or ""), allow_missing=False),
                str(payload.get("symbol") or payload.get("name") or ""),
                str(payload.get("signature") or payload.get("new_signature") or ""),
            )
        if op in {"rename_symbol_project", "update_callers"}:
            return self._rename_symbol_project(
                self.resolve_path(str(payload.get("path") or "."), allow_missing=False),
                str(payload.get("old") or payload.get("symbol") or payload.get("from") or ""),
                str(payload.get("new") or payload.get("new_name") or payload.get("to") or ""),
            )
        if op == "add_import_if_missing":
            payload = {**payload, "op": "add_import"}
            op = "add_import"
        if op in {"replace_function", "replace_method", "replace_class", "replace_symbol"}:
            return self.replace_symbol(str(payload.get("path") or ""), str(payload.get("symbol") or payload.get("name") or ""), str(payload.get("content") or ""))
        if op == "rename_symbol":
            target = self.resolve_path(str(payload.get("path") or ""), allow_missing=False)
            relative_path = self.relative_label(target)
            old = str(payload.get("old") or payload.get("symbol") or "")
            new = str(payload.get("new") or payload.get("new_name") or "")
            if not old or not new or not re.match(r"^[A-Za-z_]\w*$", old) or not re.match(r"^[A-Za-z_]\w*$", new):
                return {"ok": False, "tool": "apply_structured_edit", "summary": "rename_symbol requires valid old/new identifiers."}
            denied = self._mutation_denied_path(target)
            if denied:
                return {"ok": False, "tool": "apply_structured_edit", "path": relative_path, "summary": denied}
            original = target.read_text(encoding="utf-8", errors="replace")
            updated, count = re.subn(rf"\b{re.escape(old)}\b", new, original)
            if count == 0:
                if re.search(rf"\b{re.escape(new)}\b", original):
                    return {"ok": True, "tool": "apply_structured_edit", "path": relative_path, "op": op, "count": 0, "summary": f"Identifier already renamed from {old} to {new} in {relative_path}."}
                return {"ok": False, "tool": "apply_structured_edit", "path": relative_path, "summary": f"Identifier not found: {old}"}
            diagnostic = self._python_syntax_diagnostic(target, updated)
            if diagnostic:
                return {"ok": False, "tool": "apply_structured_edit", "path": relative_path, "syntax_ok": False, "diagnostic": diagnostic, "summary": diagnostic}
            preview = self._diff_preview(relative_path, original, updated)
            approved, reason = self._approve_mutation(f"Rename {old} to {new} in {relative_path}?", preview)
            if not approved:
                return {"ok": False, "tool": "apply_structured_edit", "path": relative_path, "summary": reason}
            target.write_text(updated, encoding="utf-8")
            return {"ok": True, "tool": "apply_structured_edit", "path": relative_path, "op": op, "count": count, "summary": f"Renamed {count} occurrence(s) of {old} to {new}.", "diff": preview}
        if op == "add_import":
            target = self.resolve_path(str(payload.get("path") or ""), allow_missing=False)
            relative_path = self.relative_label(target)
            statement = str(payload.get("statement") or payload.get("import") or "").strip()
            if not statement:
                module = str(payload.get("module") or "").strip()
                name = str(payload.get("name") or "").strip()
                statement = f"from {module} import {name}" if module and name else ""
            if not statement.startswith(("import ", "from ")):
                return {"ok": False, "tool": "apply_structured_edit", "path": relative_path, "summary": "add_import requires an import/from statement."}
            original = target.read_text(encoding="utf-8", errors="replace")
            updated = self._insert_import_statement(original, statement)
            diagnostic = self._python_syntax_diagnostic(target, updated)
            if diagnostic:
                return {"ok": False, "tool": "apply_structured_edit", "path": relative_path, "syntax_ok": False, "diagnostic": diagnostic, "summary": diagnostic}
            preview = self._diff_preview(relative_path, original, updated)
            approved, reason = self._approve_mutation(f"Add import to {relative_path}?", preview)
            if not approved:
                return {"ok": False, "tool": "apply_structured_edit", "path": relative_path, "summary": reason}
            target.write_text(updated, encoding="utf-8")
            return {"ok": True, "tool": "apply_structured_edit", "path": relative_path, "op": op, "summary": f"Added import to {relative_path}.", "diff": preview}
        if op == "delete_symbol":
            target = self.resolve_path(str(payload.get("path") or ""), allow_missing=False)
            relative_path = self.relative_label(target)
            symbol = str(payload.get("symbol") or payload.get("name") or "")
            denied = self._mutation_denied_path(target)
            if denied:
                return {"ok": False, "tool": "apply_structured_edit", "path": relative_path, "summary": denied}
            original = target.read_text(encoding="utf-8", errors="replace")
            updated, error, found = self._delete_symbol_text(target, symbol)
            if found is None:
                return {"ok": False, "tool": "apply_structured_edit", "path": relative_path, "summary": error}
            diagnostic = self._python_syntax_diagnostic(target, updated)
            if diagnostic:
                return {"ok": False, "tool": "apply_structured_edit", "path": relative_path, "syntax_ok": False, "diagnostic": diagnostic, "summary": diagnostic}
            preview = self._diff_preview(relative_path, original, updated)
            approved, reason = self._approve_mutation(f"Delete {symbol} from {relative_path}?", preview)
            if not approved:
                return {"ok": False, "tool": "apply_structured_edit", "path": relative_path, "summary": reason}
            target.write_text(updated, encoding="utf-8")
            return {"ok": True, "tool": "apply_structured_edit", "path": relative_path, "op": op, "summary": f"Deleted {symbol} from {relative_path}.", "diff": preview}
        if op == "move_symbol":
            source = self.resolve_path(str(payload.get("from_path") or payload.get("path") or ""), allow_missing=False)
            destination = self.resolve_path(str(payload.get("to_path") or ""), allow_missing=True)
            source_rel = self.relative_label(source)
            dest_rel = self.relative_label(destination)
            symbol = str(payload.get("symbol") or payload.get("name") or "")
            source_original = source.read_text(encoding="utf-8", errors="replace")
            symbols, _, _ = self._code_symbols(source)
            matches = self._symbol_matches(symbols, symbol)
            if len(matches) != 1:
                return {"ok": False, "tool": "apply_structured_edit", "path": source_rel, "summary": f"move_symbol requires one exact symbol match for {symbol}."}
            found = matches[0]
            lines = source_original.splitlines(keepends=True)
            moved_text = "".join(lines[int(found["start"]) - 1 : int(found["end"])])
            source_updated, error, _ = self._delete_symbol_text(source, symbol)
            if error:
                return {"ok": False, "tool": "apply_structured_edit", "path": source_rel, "summary": error}
            dest_original = destination.read_text(encoding="utf-8", errors="replace") if destination.exists() else ""
            dest_updated = dest_original.rstrip() + "\n\n" + moved_text.lstrip()
            source_diag = self._python_syntax_diagnostic(source, source_updated)
            dest_diag = self._python_syntax_diagnostic(destination, dest_updated)
            if source_diag or dest_diag:
                return {"ok": False, "tool": "apply_structured_edit", "syntax_ok": False, "summary": source_diag or dest_diag}
            preview = self._diff_preview(source_rel, source_original, source_updated) + "\n" + self._diff_preview(dest_rel, dest_original, dest_updated)
            approved, reason = self._approve_mutation(f"Move {symbol} from {source_rel} to {dest_rel}?", preview)
            if not approved:
                return {"ok": False, "tool": "apply_structured_edit", "summary": reason}
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(source_updated, encoding="utf-8")
            destination.write_text(dest_updated, encoding="utf-8")
            return {"ok": True, "tool": "apply_structured_edit", "op": op, "path": source_rel, "to_path": dest_rel, "summary": f"Moved {symbol} to {dest_rel}.", "diff": preview}
        return {"ok": False, "tool": "apply_structured_edit", "summary": f"Unsupported structured edit op: {op or '(missing)'}"}

    def _wrap_routed_edit_result(self, routed_tool: str, route: str, result: dict[str, Any]) -> dict[str, Any]:
        wrapped = dict(result)
        wrapped["tool"] = "edit_intent"
        wrapped["routed_tool"] = routed_tool
        wrapped["route"] = route
        summary = str(wrapped.get("summary") or wrapped.get("output") or "").strip()
        wrapped["summary"] = f"{route}: {summary}" if summary else route
        return wrapped

    def _looks_like_symbol_name(self, value: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", value.strip()))

    def _looks_like_full_symbol_source(self, target: Path, value: str) -> bool:
        stripped = value.lstrip()
        if target.suffix.lower() == ".py":
            return stripped.startswith(("def ", "async def ", "class "))
        return bool(re.match(r"(?:export\s+)?(?:async\s+)?(?:function|class)\s+\w+", stripped))

    def _looks_like_function_body_edit_intent(self, intent: str) -> bool:
        normalized = re.sub(r"[^a-z0-9_]+", "_", intent.lower())
        words = {word for word in normalized.split("_") if word}
        return bool(words & {"body", "implementation", "function", "method", "fix", "correct", "update"})

    def edit_intent(
        self,
        path: str,
        intent: str,
        target: str | None = None,
        replacement: str | None = None,
        scope: str = "file",
        apply: bool = True,
    ) -> dict[str, Any]:
        self._check_interrupted()
        target_path = self.resolve_path(path, allow_missing=False)
        relative_path = self.relative_label(target_path)
        clean_intent = str(intent or "").strip().lower().replace("-", "_")
        old = str(target or "").strip()
        new = "" if replacement is None else str(replacement)
        clean_scope = str(scope or "file").strip().lower()
        if not clean_intent:
            return {"ok": False, "tool": "edit_intent", "path": relative_path, "summary": "edit_intent requires an intent."}
        if not old and clean_intent not in {"add_import", "add_import_if_missing"}:
            return {"ok": False, "tool": "edit_intent", "path": relative_path, "summary": "edit_intent requires target for this intent."}
        if replacement is None and clean_intent != "delete_symbol":
            return {"ok": False, "tool": "edit_intent", "path": relative_path, "summary": "edit_intent requires replacement for this intent."}
        route = ""
        routed_tool = ""
        operation: dict[str, Any] | None = None
        text_replace_intents = {
            "replace",
            "replace_text",
            "text",
            "text_replace",
            "replace_in_file",
            "string_replace",
            "literal_replace",
            "update_text",
        }
        if clean_intent in {"rename", "rename_symbol", "rename_symbol_project", "update_callers", "refactor_rename"} and self._looks_like_symbol_name(old) and self._looks_like_symbol_name(new):
            if clean_scope in {"project", "repo", "repository", "all"}:
                route = "project symbol rename"
                routed_tool = "apply_structured_edit"
                operation = {"op": "rename_symbol_project", "path": relative_path, "old": old, "new": new}
            else:
                route = "identifier text rename in file"
                routed_tool = "replace_in_file"
        elif clean_intent in {"replace_body", "replace_function_body", "function_body"}:
            route = "replace Python function body"
            routed_tool = "apply_structured_edit"
            operation = {"op": "replace_function_body", "path": relative_path, "symbol": old, "body": new}
        elif clean_intent in {"change_signature", "replace_signature", "signature"}:
            route = "change Python function signature"
            routed_tool = "apply_structured_edit"
            operation = {"op": "change_signature", "path": relative_path, "symbol": old, "signature": new}
        elif clean_intent in {"add_import", "add_import_if_missing"}:
            statement = new.strip() or old
            route = "add import if missing"
            routed_tool = "apply_structured_edit"
            operation = {"op": "add_import_if_missing", "path": relative_path, "statement": statement}
        elif (
            target_path.suffix.lower() == ".py"
            and self._looks_like_symbol_name(old)
            and self._looks_like_function_body_edit_intent(clean_intent)
        ):
            if self._looks_like_full_symbol_source(target_path, new):
                route = "replace symbol source"
                routed_tool = "replace_symbol"
            else:
                route = "replace Python function body"
                routed_tool = "apply_structured_edit"
                operation = {"op": "replace_function_body", "path": relative_path, "symbol": old, "body": new}
        elif clean_intent in {"replace_symbol", "replace_function", "replace_class", "symbol"}:
            if self._looks_like_symbol_name(old) and self._looks_like_full_symbol_source(target_path, new):
                route = "replace symbol source"
                routed_tool = "replace_symbol"
            elif target_path.suffix.lower() == ".py" and self._looks_like_symbol_name(old):
                route = "replace Python function body"
                routed_tool = "apply_structured_edit"
                operation = {"op": "replace_function_body", "path": relative_path, "symbol": old, "body": new}
            else:
                route = "symbol-like request routed to text replace because target/replacement is not full symbol source"
                routed_tool = "replace_in_file"
        elif clean_intent in text_replace_intents:
            route = "replace text in file"
            routed_tool = "replace_in_file"
        else:
            return {
                "ok": False,
                "tool": "edit_intent",
                "path": relative_path,
                "summary": (
                    f"Unknown edit_intent intent: {intent}. Use one of rename, replace_text, "
                    "replace_symbol, replace_body, change_signature, or add_import."
                ),
                "error_class": "invalid_args",
            }
        if not apply:
            return {
                "ok": True,
                "tool": "edit_intent",
                "path": relative_path,
                "route": route,
                "routed_tool": routed_tool,
                "output": f"{route} -> {routed_tool}",
            }
        if operation is not None:
            return self._wrap_routed_edit_result(routed_tool, route, self.apply_structured_edit(operation))
        if routed_tool == "replace_symbol":
            return self._wrap_routed_edit_result(routed_tool, route, self.replace_symbol(relative_path, old, new))
        replace_all = clean_scope in {"project", "repo", "repository", "all"} or clean_intent in {"rename", "rename_symbol", "refactor_rename"}
        match_whole_word = self._looks_like_symbol_name(old) and self._looks_like_symbol_name(new) and "(" not in old
        return self._wrap_routed_edit_result(
            "replace_in_file",
            route,
            self.replace_in_file(relative_path, old, new, replace_all=replace_all, match_whole_word=match_whole_word),
        )

    def generate_tests_from_spec(
        self,
        target_symbol: str,
        behavior: str,
        test_path: str | None = None,
        apply: bool = False,
    ) -> dict[str, Any]:
        symbol = target_symbol.strip()
        spec = behavior.strip()
        if not symbol or not spec:
            return {"ok": False, "tool": "generate_tests_from_spec", "summary": "target_symbol and behavior are required."}
        leaf = re.sub(r"\W+", "_", symbol.split(".")[-1]).strip("_").lower() or "target"
        rel_test_path = test_path or f"tests/test_{leaf}_spec.py"
        target = self.resolve_path(rel_test_path)
        test_name = f"test_{leaf}_matches_spec"
        module = ".".join(symbol.split(".")[:-1])
        imported = symbol.split(".")[-1]
        import_line = f"from {module} import {imported}\n\n" if module else ""
        content = (
            import_line
            + f"def {test_name}():\n"
            + f"    \"\"\"Spec: {spec[:180].replace(chr(10), ' ')}\"\"\"\n"
            + "    # Fill in concrete arrange/act/assert values from the spec before relying on this test.\n"
            + f"    raise AssertionError({json.dumps('TODO encode spec for ' + symbol + ': ' + spec[:220])})\n"
        )
        existing = target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""
        updated = existing.rstrip() + ("\n\n" if existing.strip() else "") + content
        preview = self._diff_preview(self.relative_label(target), existing, updated)
        if not apply:
            return {
                "ok": True,
                "tool": "generate_tests_from_spec",
                "path": self.relative_label(target),
                "applied": False,
                "summary": "Generated test patch preview only. Review and apply explicitly before using it as a gate.",
                "diff": preview,
            }
        approved, reason = self._approve_mutation(f"Apply generated tests to {self.relative_label(target)}?", preview)
        if not approved:
            return {"ok": False, "tool": "generate_tests_from_spec", "path": self.relative_label(target), "summary": reason}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(updated, encoding="utf-8")
        return {"ok": True, "tool": "generate_tests_from_spec", "path": self.relative_label(target), "applied": True, "summary": f"Wrote generated test scaffold to {self.relative_label(target)}.", "diff": preview}

    def git_status(self, path: str | None = None) -> dict[str, Any]:
        ok, error = self._ensure_git_repo()
        if not ok:
            return {"ok": False, "tool": "git_status", "summary": error or "Not inside a git repository."}
        command = ["status", "--short", "--branch", "--untracked-files=all"]
        relative_path = None
        if path:
            relative_path = self.relative_label(self.resolve_path(path, allow_missing=False))
            command.extend(["--", relative_path])
        result = self._run_git(command)
        output = result.stdout.strip() or "(clean)"
        if result.stderr.strip():
            output = f"{output}\n{result.stderr.strip()}"
        return {
            "ok": result.returncode == 0,
            "tool": "git_status",
            "path": relative_path or ".",
            "output": output,
        }

    def git_diff(self, path: str | None = None, cached: bool = False, context: int = 3) -> dict[str, Any]:
        ok, error = self._ensure_git_repo()
        if not ok:
            return {"ok": False, "tool": "git_diff", "summary": error or "Not inside a git repository."}
        command = ["diff", "--no-ext-diff", f"--unified={max(0, context)}"]
        if cached:
            command.append("--cached")
        relative_path = None
        if path:
            relative_path = self.relative_label(self.resolve_path(path, allow_missing=False))
            command.extend(["--", relative_path])
        result = self._run_git(command)
        output = self._collect_process_output(result)
        if output == "(no output)":
            output = "(no diff)"
        return {
            "ok": result.returncode == 0,
            "tool": "git_diff",
            "path": relative_path or ".",
            "cached": cached,
            "output": output,
        }

    def git_commit(self, message: str, add_all: bool = True) -> dict[str, Any]:
        commit_message = message.strip()
        if not commit_message:
            return {"ok": False, "tool": "git_commit", "summary": "git_commit requires a non-empty commit message."}
        ok, error = self._ensure_git_repo()
        if not ok:
            return {"ok": False, "tool": "git_commit", "summary": error or "Not inside a git repository."}
        status = self._run_git(["status", "--short", "--branch", "--untracked-files=all"])
        staged = self._run_git(["diff", "--cached", "--stat"])
        working = self._run_git(["diff", "--stat"])
        preview_parts = [f"Status:\n{status.stdout.strip() or '(clean)'}"]
        if staged.stdout.strip():
            preview_parts.append(f"Staged diff:\n{staged.stdout.strip()}")
        if add_all and working.stdout.strip():
            preview_parts.append(f"Working diff:\n{working.stdout.strip()}")
        preview = "\n\n".join(preview_parts)
        approved, reason = self._approve_mutation(f'Create git commit "{commit_message}"?', preview)
        if not approved:
            return {"ok": False, "tool": "git_commit", "summary": reason}
        if add_all:
            blocked_paths = sorted(self._git_dirty_paths() & self._initial_dirty_paths)
            if blocked_paths:
                preview_paths = ", ".join(blocked_paths[:5])
                if len(blocked_paths) > 5:
                    preview_paths = f"{preview_paths}, ..."
                return {
                    "ok": False,
                    "tool": "git_commit",
                    "summary": "git_commit(add_all=True) refused because the repo already had unrelated local changes when this tool session started: "
                    f"{preview_paths}",
                }
            added = self._run_git(["add", "-A"])
            if added.returncode != 0:
                return {"ok": False, "tool": "git_commit", "summary": self._collect_process_output(added)}
        has_changes = self._run_git(["diff", "--cached", "--quiet"])
        if has_changes.returncode == 0:
            return {"ok": False, "tool": "git_commit", "summary": "No staged changes to commit."}
        committed = self._run_git(["commit", "-m", commit_message])
        output = self._collect_process_output(committed)
        return {
            "ok": committed.returncode == 0,
            "tool": "git_commit",
            "summary": f'Created commit "{commit_message}".' if committed.returncode == 0 else output,
            "output": output,
        }

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        target = self.resolve_path(path)
        denied = self._mutation_denied_path(target)
        if denied:
            return {"ok": False, "tool": "write_file", "path": self.relative_label(target), "summary": denied}
        if self._contains_omitted_context_marker(content):
            return {
                "ok": False,
                "tool": "write_file",
                "path": self.relative_label(target),
                "summary": "Refusing to write omitted-context marker as file content. Reconstruct complete file content instead.",
            }
        content, normalization = self._normalize_python_write_content(target, content)
        existing = target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""
        relative_path = self.relative_label(target)
        dropped_symbols = self._python_dropped_top_level_symbols(target, existing, content)
        if dropped_symbols:
            preview_names = ", ".join(dropped_symbols[:6])
            if len(dropped_symbols) > 6:
                preview_names = f"{preview_names}, ..."
            return {
                "ok": False,
                "tool": "write_file",
                "path": relative_path,
                "summary": "Refusing to overwrite Python file with partial content that drops existing top-level symbols: "
                f"{preview_names}. Use replace_symbol/replace_symbols for targeted edits, delete_symbol for removals, or provide the complete file.",
            }
        preview = self._diff_preview(relative_path, existing, content)
        approved, reason = self._approve_mutation(f"Write {relative_path}?", preview)
        if not approved:
            return {"ok": False, "tool": "write_file", "path": relative_path, "summary": reason}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        result = {
            "ok": True,
            "tool": "write_file",
            "path": relative_path,
            "summary": f"Wrote {relative_path}.",
            "diff": preview,
        }
        if normalization:
            result["normalized"] = normalization
            result["summary"] += f" {normalization}"
        diagnostic = self._python_syntax_diagnostic(target, content)
        if diagnostic is not None:
            result["syntax_ok"] = False
            result["diagnostic"] = diagnostic
            result["summary"] += f" {diagnostic}"
        return result

    def _mutation_denied_path(self, target: Path) -> str | None:
        for part in target.relative_to(self.workspace_root).parts:
            if part in DENY_MUTATION_DIRS:
                return f"Refusing to mutate generated/cache path: {self.relative_label(target)}."
        return None

    def _contains_omitted_context_marker(self, content: str) -> bool:
        return bool(re.search(r"\[omitted \d+ chars from prior [A-Za-z_]+; do not copy\]", content))

    def _normalize_python_write_content(self, target: Path, content: str) -> tuple[str, str | None]:
        if target.suffix.lower() != ".py":
            return content, None
        if self._python_syntax_diagnostic(target, content) is None:
            return content, None
        candidates: list[tuple[str, str]] = []
        fenced_match = re.match(r"^\s*```(?:python|py)?\s*\n(?P<body>.*?)(?:\n)?```\s*$", content, flags=re.DOTALL | re.IGNORECASE)
        if fenced_match:
            candidates.append((fenced_match.group("body"), "Stripped markdown code fence from Python file content before write."))
        quote_stripped = self._strip_markdown_quote_prefixes(content)
        if quote_stripped != content:
            candidates.append((quote_stripped, "Stripped markdown quote prefixes from Python file content before write."))
        if fenced_match:
            fenced_quote_stripped = self._strip_markdown_quote_prefixes(fenced_match.group("body"))
            if fenced_quote_stripped != fenced_match.group("body"):
                candidates.append((fenced_quote_stripped, "Stripped markdown code fence and quote prefixes from Python file content before write."))
        join_repaired = self._repair_common_python_join_typo(content)
        if join_repaired != content:
            candidates.append((join_repaired, "Repaired common Python join string typo before write."))
        for candidate, reason in list(candidates):
            if self._python_syntax_diagnostic(target, candidate) is None:
                return candidate, reason
        bases = [(content, "Auto-dedented Python file content before write."), *candidates]
        for candidate, reason in bases:
            dedented = textwrap.dedent(candidate)
            if dedented == candidate:
                continue
            if self._python_syntax_diagnostic(target, dedented) is None:
                if reason.startswith("Auto-dedented"):
                    return dedented, reason
                return dedented, reason[:-1] + " and auto-dedented it."
        return content, None

    def _repair_common_python_join_typo(self, content: str) -> str:
        return re.sub(r"(?m)^(\s*return\s+)['\"]\.join\(", r'\1" ".join(', content)

    def _strip_markdown_quote_prefixes(self, content: str) -> str:
        lines = content.splitlines(keepends=True)
        if not lines:
            return content
        prefixed = sum(1 for line in lines if re.match(r"^\s*>\s?", line))
        non_empty = sum(1 for line in lines if line.strip())
        first_non_empty = next((line for line in lines if line.strip()), "")
        if non_empty == 0 or (prefixed < max(1, non_empty // 2) and not re.match(r"^\s*>\s?", first_non_empty)):
            return content
        return "".join(re.sub(r"^\s*>\s?", "", line, count=1) for line in lines)

    def _python_syntax_diagnostic(self, target: Path, content: str) -> str | None:
        if target.suffix.lower() != ".py":
            return None
        try:
            ast.parse(self._python_parse_text(content), filename=self.relative_label(target))
        except SyntaxError as exc:
            line = exc.lineno if exc.lineno is not None else "?"
            offset = exc.offset if exc.offset is not None else "?"
            near = f" near {exc.text.strip()[:80]!r}" if exc.text and exc.text.strip() else ""
            return f"Python syntax error: {exc.__class__.__name__}: {exc.msg} at {self.relative_label(target)}:{line}:{offset}{near}."
        return None

    def _python_parse_text(self, content: str) -> str:
        return content[1:] if content.startswith("\ufeff") else content

    def _python_top_level_symbol_names(self, target: Path, content: str) -> list[str]:
        if target.suffix.lower() != ".py":
            return []
        try:
            tree = ast.parse(self._python_parse_text(content), filename=self.relative_label(target))
        except SyntaxError:
            return []
        names: list[str] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(node.name)
        return names

    def _python_dropped_top_level_symbols(self, target: Path, existing: str, updated: str) -> list[str]:
        if target.suffix.lower() != ".py" or not existing.strip():
            return []
        existing_names = set(self._python_top_level_symbol_names(target, existing))
        updated_names = set(self._python_top_level_symbol_names(target, updated))
        if not existing_names or not updated_names:
            return []
        return sorted(existing_names - updated_names)

    def _looks_like_word_char(self, value: str) -> bool:
        return value.isalnum() or value == "_"

    def _ambiguous_short_replace(self, text: str, old: str) -> bool:
        needle = old.strip()
        if len(needle) > 3 or not needle.isidentifier():
            return False
        whole_word_matches = 0
        embedded_matches = 0
        start = 0
        while True:
            index = text.find(old, start)
            if index == -1:
                break
            before = text[index - 1] if index > 0 else ""
            after_index = index + len(old)
            after = text[after_index] if after_index < len(text) else ""
            is_whole_word = not self._looks_like_word_char(before) and not self._looks_like_word_char(after)
            if is_whole_word:
                whole_word_matches += 1
            else:
                embedded_matches += 1
            start = index + len(old)
        return whole_word_matches > 0 and embedded_matches > 0

    def _strip_read_file_line_prefixes(self, text: str) -> str:
        lines = text.splitlines()
        if not lines:
            return text
        converted: list[str] = []
        matched = 0
        for line in lines:
            match = re.match(r"^\s*\d+\s+\|\s?(.*)$", line)
            if match:
                converted.append(match.group(1))
                matched += 1
                continue
            if line.strip():
                return text
            converted.append("")
        if matched == 0:
            return text
        result = "\n".join(converted)
        if text.endswith("\n"):
            result += "\n"
        return result

    def _leading_whitespace_flexible_pattern(self, old: str) -> re.Pattern[str] | None:
        if not old.strip():
            return None
        lines = old.split("\n")
        if not any(re.match(r"^[ \t]+\S", line) for line in lines):
            return None
        parts: list[str] = []
        for index, line in enumerate(lines):
            match = re.match(r"^[ \t]+(\S.*)$", line)
            if match:
                prefix = r"(?m)^" if index == 0 else ""
                parts.append(prefix + r"[ \t]*" + re.escape(match.group(1)))
            else:
                parts.append(re.escape(line))
        return re.compile("\n".join(parts))

    def _identifier_call_rename_pattern(self, old: str, new: str) -> re.Pattern[str] | None:
        old_match = re.fullmatch(r"\s*(?P<old>[A-Za-z_]\w*)\s*\(\s*", old)
        new_match = re.fullmatch(r"\s*(?P<new>[A-Za-z_]\w*)\s*\(\s*", new)
        if not old_match or not new_match:
            return None
        old_name = old_match.group("old")
        if old_name == new_match.group("new"):
            return None
        return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(old_name)}\s*\(")

    def replace_in_file(
        self,
        path: str,
        old: str,
        new: str,
        replace_all: bool = False,
        match_whole_word: bool = False,
    ) -> dict[str, Any]:
        target = self.resolve_path(path, allow_missing=False)
        denied = self._mutation_denied_path(target)
        if denied:
            return {"ok": False, "tool": "replace_in_file", "path": self.relative_label(target), "summary": denied}
        if self._contains_omitted_context_marker(old) or self._contains_omitted_context_marker(new):
            return {
                "ok": False,
                "tool": "replace_in_file",
                "path": self.relative_label(target),
                "summary": "Refusing to use omitted-context marker as replacement text. Read the file or reconstruct exact text instead.",
            }
        original = target.read_text(encoding="utf-8", errors="replace")
        stripped_old = self._strip_read_file_line_prefixes(old)
        if stripped_old != old:
            old = stripped_old
            new = self._strip_read_file_line_prefixes(new)
        identifier_call_pattern = None if match_whole_word else self._identifier_call_rename_pattern(old, new)
        if identifier_call_pattern is not None:
            pattern = identifier_call_pattern
            count = len(pattern.findall(original))
        elif match_whole_word:
            pattern = re.compile(rf"\b{re.escape(old)}\b")
            count = len(pattern.findall(original))
        else:
            pattern = self._leading_whitespace_flexible_pattern(old)
            if pattern is not None:
                count = len(pattern.findall(original))
                if count == 0:
                    pattern = None
                    count = original.count(old)
            else:
                count = original.count(old)
        if count == 0:
            return {"ok": False, "tool": "replace_in_file", "path": self.relative_label(target), "summary": "Target text was not found."}
        if not replace_all and count != 1:
            return {
                "ok": False,
                "tool": "replace_in_file",
                "path": self.relative_label(target),
                "summary": f"Target text matched {count} times. Provide a more specific snippet or set replace_all.",
            }
        if replace_all and not match_whole_word and self._ambiguous_short_replace(original, old):
            return {
                "ok": False,
                "tool": "replace_in_file",
                "path": self.relative_label(target),
                "summary": "Short replacement is ambiguous because it appears both as a word and inside larger tokens. Use match_whole_word or a longer snippet.",
            }
        if match_whole_word:
            limit = 0 if replace_all else 1
            updated = pattern.sub(lambda _match: new, original, count=limit)
        elif pattern is not None:
            limit = 0 if replace_all else 1
            updated = pattern.sub(lambda _match: new, original, count=limit)
        else:
            updated = original.replace(old, new) if replace_all else original.replace(old, new, 1)
        relative_path = self.relative_label(target)
        preview = self._diff_preview(relative_path, original, updated)
        approved, reason = self._approve_mutation(f"Replace text in {relative_path}?", preview)
        if not approved:
            return {"ok": False, "tool": "replace_in_file", "path": relative_path, "summary": reason}
        target.write_text(updated, encoding="utf-8")
        replaced_count = count if replace_all else 1
        result = {
            "ok": True,
            "tool": "replace_in_file",
            "path": relative_path,
            "summary": f"Replaced {replaced_count} occurrence(s) in {relative_path}.",
            "diff": preview,
        }
        diagnostic = self._python_syntax_diagnostic(target, updated)
        if diagnostic is not None:
            result["syntax_ok"] = False
            result["diagnostic"] = diagnostic
            result["summary"] += f" {diagnostic}"
        return result

    def browser_smoke(
        self,
        url: str,
        actions: list[dict[str, Any]] | None = None,
        wait_for: str | None = None,
        viewport: str = "desktop",
        screenshot: bool = False,
    ) -> dict[str, Any]:
        if not self.browser_enabled:
            return {"ok": False, "tool": "browser_smoke", "summary": "browser_smoke is disabled by config."}
        clean_url = str(url or "").strip()
        if not clean_url:
            return {"ok": False, "tool": "browser_smoke", "summary": "browser_smoke requires a URL."}
        probe = subprocess.run([sys.executable, "-c", "import playwright.sync_api"], capture_output=True, text=True, timeout=10, check=False)
        if probe.returncode != 0:
            return self._missing_dependency_result("browser_smoke", "playwright", "Playwright is not installed. Install playwright and browsers to use browser_smoke.")
        artifact_path = None
        if screenshot:
            artifact_dir = self.workspace_root / ".ollama-code" / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha1(clean_url.encode("utf-8", errors="ignore")).hexdigest()[:10]
            artifact_path = artifact_dir / f"browser-{digest}-{int(time.time())}.png"
        payload = {
            "url": clean_url,
            "actions": actions or [],
            "wait_for": wait_for,
            "viewport": viewport,
            "screenshot": str(artifact_path) if artifact_path else None,
        }
        script = r'''
import json, sys
from playwright.sync_api import sync_playwright
payload = json.loads(sys.stdin.read())
events = {"console_errors": [], "page_errors": [], "failed_requests": []}
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    size = {"width": 390, "height": 844} if payload.get("viewport") == "mobile" else {"width": 1280, "height": 720}
    page = browser.new_page(viewport=size)
    page.on("console", lambda msg: events["console_errors"].append(msg.text) if msg.type == "error" else None)
    page.on("pageerror", lambda exc: events["page_errors"].append(str(exc)))
    page.on("requestfailed", lambda req: events["failed_requests"].append(req.url))
    page.goto(payload["url"], wait_until="networkidle", timeout=30000)
    for action in payload.get("actions") or []:
        kind = action.get("type") or action.get("action")
        selector = action.get("selector")
        if kind == "click" and selector:
            page.click(selector, timeout=10000)
        elif kind == "fill" and selector:
            page.fill(selector, str(action.get("text", "")), timeout=10000)
        elif kind == "press" and selector:
            page.press(selector, str(action.get("key", "Enter")), timeout=10000)
    wait_for = payload.get("wait_for")
    if wait_for:
        if wait_for.startswith("text="):
            page.get_by_text(wait_for[5:]).wait_for(timeout=10000)
        else:
            page.wait_for_selector(wait_for, timeout=10000)
    if payload.get("screenshot"):
        page.screenshot(path=payload["screenshot"], full_page=True)
    title = page.title()
    body = page.locator("body").inner_text(timeout=10000)[:1000] if page.locator("body").count() else ""
    browser.close()
print(json.dumps({"title": title, "body": body, **events}, ensure_ascii=True))
'''
        completed = subprocess.run([sys.executable, "-c", script], input=json.dumps(payload), capture_output=True, text=True, timeout=45, check=False)
        output = self._collect_process_output(completed)
        try:
            result_payload = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError:
            result_payload = {}
        problems = []
        for key in ("console_errors", "page_errors", "failed_requests"):
            values = result_payload.get(key)
            if isinstance(values, list) and values:
                problems.append(f"{key}={len(values)}")
        lines = [
            f"title={result_payload.get('title', '')}",
            self._truncate_text(str(result_payload.get("body", "")).replace("\n", " | "), limit=700),
        ]
        if artifact_path:
            lines.append(f"screenshot={self.relative_label(artifact_path)}")
        if problems:
            lines.append("problems=" + ", ".join(problems))
        return {
            "ok": completed.returncode == 0 and not problems,
            "tool": "browser_smoke",
            "url": clean_url,
            "screenshot": self.relative_label(artifact_path) if artifact_path else None,
            "summary": "Browser smoke passed." if completed.returncode == 0 and not problems else "Browser smoke found issues or failed.",
            "output": "\n".join(line for line in lines if line) if completed.returncode == 0 else output,
        }

    def security_scan(self, path: str = ".", scanners: str | list[str] = "auto", limit: int = 80, timeout: int = 120) -> dict[str, Any]:
        if not self.security_enabled:
            return {"ok": False, "tool": "security_scan", "summary": "security_scan is disabled by config."}
        base = self.resolve_path(path, allow_missing=False)
        requested = [scanners] if isinstance(scanners, str) else list(scanners or [])
        requested_names = {part.strip().lower() for item in requested for part in str(item).split(",") if part.strip()}
        available: list[tuple[str, list[str]]] = []
        if ("auto" in requested_names or "gitleaks" in requested_names) and shutil.which("gitleaks"):
            available.append(("gitleaks", ["gitleaks", "detect", "--source", str(base), "--redact", "--no-git"]))
        if ("auto" in requested_names or "pip-audit" in requested_names) and shutil.which("pip-audit") and ((base / "pyproject.toml").exists() if base.is_dir() else True):
            available.append(("pip-audit", ["pip-audit"]))
        if ("auto" in requested_names or "npm" in requested_names or "npm-audit" in requested_names) and shutil.which("npm") and ((base / "package.json").exists() if base.is_dir() else (base.parent / "package.json").exists()):
            available.append(("npm audit", ["npm", "audit", "--json"]))
        if ("auto" in requested_names or "osv-scanner" in requested_names) and shutil.which("osv-scanner"):
            available.append(("osv-scanner", ["osv-scanner", "--recursive", str(base)]))
        if ("auto" in requested_names or "trivy" in requested_names) and shutil.which("trivy"):
            available.append(("trivy", ["trivy", "fs", "--quiet", str(base)]))
        if not available:
            return self._missing_dependency_result("security_scan", "security scanner", "No supported security scanners found on PATH.")
        outputs: list[str] = []
        ok = True
        for name, command in available[: max(1, int(limit))]:
            completed = self._run_process(command, cwd=base if base.is_dir() else base.parent, timeout=timeout, shell=False)
            output = self._truncate_text(self._collect_process_output(completed), limit=1200)
            outputs.append(f"## {name} exit={completed.returncode}\n{output}")
            if completed.returncode not in {0}:
                ok = False
        return {
            "ok": ok,
            "tool": "security_scan",
            "path": self.relative_label(base),
            "count": len(outputs),
            "summary": "Security scan completed." if ok else "Security scan completed with findings or scanner errors.",
            "output": "\n\n".join(outputs),
        }

    def _mcp_server_payload(self, server: str) -> dict[str, Any] | None:
        payload = self.mcp_servers.get(server)
        return payload if isinstance(payload, dict) else None

    def _mcp_server_command(self, server: str) -> tuple[list[str] | None, dict[str, str], str | None]:
        payload = self._mcp_server_payload(server)
        if payload is None:
            return None, {}, f"MCP server is not configured: {server}"
        command = payload.get("command")
        if not isinstance(command, str) or not command.strip():
            return None, {}, f"MCP server {server} requires a command."
        args = payload.get("args") if isinstance(payload.get("args"), list) else []
        argv = [command.strip(), *(str(arg) for arg in args)]
        executable = shutil.which(argv[0]) if not Path(argv[0]).is_absolute() else argv[0]
        if not executable or not Path(executable).exists() and shutil.which(argv[0]) is None:
            return None, {}, f"executable not found: {argv[0]}"
        env_payload = payload.get("env") if isinstance(payload.get("env"), dict) else {}
        env = {str(key): str(value) for key, value in env_payload.items()}
        return argv, env, None

    def _mcp_request(self, server: str, requests: list[dict[str, Any]], timeout: int) -> tuple[bool, list[dict[str, Any]], str]:
        argv, extra_env, error = self._mcp_server_command(server)
        if error:
            return False, [], error
        assert argv is not None
        env = os.environ.copy()
        env.update(extra_env)
        input_text = "".join(json.dumps(request, separators=(",", ":")) + "\n" for request in requests)
        try:
            completed = subprocess.run(
                argv,
                cwd=self.workspace_root,
                input=input_text,
                text=True,
                capture_output=True,
                timeout=max(1, int(timeout)),
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, [], f"MCP server {server} timed out after {timeout}s."
        except Exception as exc:
            return False, [], str(exc)
        wanted_ids = {request.get("id") for request in requests if "id" in request}
        responses: list[dict[str, Any]] = []
        for line in completed.stdout.splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("id") in wanted_ids:
                responses.append(payload)
        if completed.returncode not in {0} and not responses:
            stderr = completed.stderr.strip()
            summary = stderr or completed.stdout.strip() or f"MCP server {server} exited with {completed.returncode}."
            return False, responses, self._truncate_text(summary, limit=800)
        return True, responses, completed.stderr.strip()

    def mcp_list_tools(self, server: str | None = None, timeout: int = 15) -> dict[str, Any]:
        servers = [server] if isinstance(server, str) and server.strip() else sorted(self.mcp_servers)
        if not servers:
            return {"ok": False, "tool": "mcp_list_tools", "summary": "No MCP servers configured.", "missing_dependency": "mcp.servers"}
        rows: list[str] = []
        for name in servers:
            ok, responses, error = self._mcp_request(
                name,
                [
                    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "ollama-code", "version": "0.1.0"}}},
                    {"jsonrpc": "2.0", "method": "notifications/initialized"},
                    {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
                ],
                timeout,
            )
            if not ok:
                rows.append(f"{name}: {error}")
                continue
            tools_payload = next((item.get("result", {}).get("tools") for item in responses if item.get("id") == 2), [])
            if isinstance(tools_payload, list):
                for item in tools_payload:
                    if isinstance(item, dict):
                        rows.append(f"mcp.{name}.{item.get('name')}: {str(item.get('description') or '')[:120]}")
            elif error:
                rows.append(f"{name}: {error[:200]}")
        return {"ok": True, "tool": "mcp_list_tools", "count": len(rows), "output": "\n".join(rows) if rows else "(no MCP tools returned)"}

    def mcp_call(self, server: str, tool: str, arguments: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            arguments = {}
        tool_name = str(tool or "").strip()
        server_name = str(server or "").strip()
        if not server_name or not tool_name:
            return {"ok": False, "tool": "mcp_call", "summary": "mcp_call requires server and tool."}
        approved, reason = self._approve_shell(f"mcp {server_name}.{tool_name}", ".")
        if not approved:
            return {"ok": False, "tool": "mcp_call", "summary": reason}
        ok, responses, error = self._mcp_request(
            server_name,
            [
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "ollama-code", "version": "0.1.0"}}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": tool_name, "arguments": arguments}},
            ],
            timeout,
        )
        if not ok:
            missing = "mcp server" if "not configured" in error else None
            result = {"ok": False, "tool": "mcp_call", "server": server_name, "mcp_tool": tool_name, "summary": error}
            if missing:
                result["missing_dependency"] = missing
            return result
        response = next((item for item in responses if item.get("id") == 2), {})
        if response.get("error"):
            return {"ok": False, "tool": "mcp_call", "server": server_name, "mcp_tool": tool_name, "summary": json.dumps(response.get("error"), ensure_ascii=True)[:800]}
        return {
            "ok": True,
            "tool": "mcp_call",
            "server": server_name,
            "mcp_tool": tool_name,
            "output": self._truncate_text(json.dumps(response.get("result", {}), ensure_ascii=True), limit=1600),
        }

    def classify_error(self, text: str) -> str:
        for name, pattern in ERROR_CLASS_PATTERNS.items():
            if pattern.search(text):
                return name
        return "unknown"

    def _missing_dependency_name(self, text: str) -> str | None:
        match = ERROR_CLASS_PATTERNS["missing_dependency"].search(text)
        if not match:
            return None
        return next((group for group in match.groups() if group), None)

    def _split_command_for_validation(self, command: str) -> tuple[list[str] | None, dict[str, Any] | None]:
        try:
            argv = shlex.split(command, posix=os.name != "nt")
        except ValueError as exc:
            return None, {
                "recognized": False,
                "valid": False,
                "family": "shell",
                "reason": f"Command rejected before execution: invalid quoting ({exc}).",
            }
        if not argv:
            return None, {
                "recognized": False,
                "valid": False,
                "family": "shell",
                "reason": "Command rejected before execution: empty command.",
            }
        return argv, None

    def _command_has_shell_chaining(self, command: str) -> bool:
        return bool(re.search(r"&&|\|\||[;|<>]", command))

    def _command_path_escapes(self, token: str, cwd: Path) -> bool:
        if not token or token.startswith("-"):
            return False
        if token in {".", ".."} or token.startswith("../") or token.startswith("..\\"):
            return True
        if any(ch in token for ch in "*?[]"):
            return False
        if re.match(r"^[A-Za-z]+://", token):
            return False
        candidate = self._coerce_input_path(token)
        if not candidate.is_absolute():
            candidate = cwd / candidate
        try:
            resolved = candidate.resolve(strict=False)
        except OSError:
            return False
        root = self.workspace_root.resolve(strict=False)
        return resolved != root and root not in resolved.parents

    def _validate_path_args(self, argv: list[str], cwd: Path, *, start: int = 1) -> str | None:
        for token in argv[start:]:
            if self._command_path_escapes(token, cwd):
                return f"path escapes workspace: {token}"
        return None

    def _command_family(self, argv: list[str]) -> str | None:
        executable = Path(str(argv[0]).strip().strip("'\"")).name.lower()
        if executable.endswith(".exe"):
            executable = executable[:-4]
        if executable in {"git", "pytest", "ruff", "mypy", "pyright", "tsc", "npm", "pnpm", "yarn", "go", "cargo", "gradle", "gradlew", "gradlew.bat", "cmake", "ctest", "node"}:
            return executable
        if executable in {"python", "python3", "py"}:
            if "-m" in argv:
                return "python"
            return "python_exec"
        return None

    def _unknown_command_error(self, argv: list[str]) -> str | None:
        executable = str(argv[0]).strip().strip("'\"")
        if not executable or "=" in executable and not executable.startswith(("./", "../", ".\\", "..\\")):
            return None
        shell_builtins = {
            "alias",
            "bg",
            "break",
            "cd",
            "command",
            "continue",
            "dirs",
            "echo",
            "eval",
            "exit",
            "export",
            "false",
            "fg",
            "hash",
            "help",
            "jobs",
            "popd",
            "printf",
            "pushd",
            "pwd",
            "read",
            "return",
            "set",
            "shift",
            "source",
            "test",
            "times",
            "trap",
            "true",
            "type",
            "ulimit",
            "umask",
            "unalias",
            "unset",
        }
        windows_builtins = {"assoc", "call", "cat", "clear", "cls", "copy", "cp", "del", "dir", "erase", "ls", "md", "mkdir", "move", "mv", "rd", "ren", "rm", "rmdir", "type"}
        name = Path(executable).name.lower()
        if name in shell_builtins or (os.name == "nt" and name in windows_builtins):
            return None
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", executable):
            return None
        if any(ch in executable for ch in "*?[]"):
            return None
        if shutil.which(executable):
            return None
        candidate = self._coerce_input_path(executable)
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        if candidate.exists():
            return None
        return f"executable not found: {executable}"

    def _validation_result(self, *, family: str, valid: bool, reason: str = "", argv: list[str] | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {"recognized": True, "valid": valid, "family": family}
        if argv is not None:
            result["argv"] = argv
        if reason:
            result["reason"] = reason
        return result

    def _needs_bash_syntax_check(self, command: str, family: str | None) -> bool:
        if family is not None:
            return False
        if os.name == "nt" and self._looks_like_powershell_command(command):
            return False
        return bool(
            "\n" in command
            or re.search(r"&&|\|\||[;|<>]|[(){}]", command)
            or re.match(r"\s*(?:if|then|else|elif|fi|for|while|case|function)\b", command)
        )

    def _bash_syntax_validation(self, command: str) -> dict[str, Any] | None:
        if os.name == "nt" and self._looks_like_powershell_command(command):
            return None
        bash = shutil.which("bash")
        if not bash:
            return None
        if Path(bash).name.lower() not in {"bash", "bash.exe"}:
            return None
        try:
            completed = subprocess.run([bash, "-n", "-c", command], capture_output=True, text=True, timeout=5, check=False)
        except (OperationInterrupted, subprocess.TimeoutExpired):
            raise
        except Exception:
            return None
        if completed.returncode == 0:
            return None
        output = self._collect_process_output(completed) or "bash syntax check failed"
        return {
            "recognized": True,
            "valid": False,
            "family": "bash",
            "reason": "bash -n rejected command syntax: " + self._truncate_text(output, limit=240),
            "argv": [bash, "-n", "-c", command],
        }

    def _validate_common_command(self, command: str, cwd: Path) -> tuple[list[str] | None, dict[str, Any]]:
        argv, split_error = self._split_command_for_validation(command)
        if split_error is not None:
            return None, split_error
        assert argv is not None
        argv = [str(argv[0]).strip().strip("'\""), *argv[1:]]
        if "-c" in argv:
            try:
                code_index = argv.index("-c") + 1
                argv[code_index] = str(argv[code_index]).strip().strip("'\"")
            except IndexError:
                pass
        family = self._command_family(argv)
        if self._needs_bash_syntax_check(command, family):
            syntax_error = self._bash_syntax_validation(command)
            if syntax_error is not None:
                return None, syntax_error
        if family is None:
            if os.name == "nt" and self._looks_like_powershell_command(command):
                return None, {"recognized": False, "valid": True, "family": "powershell"}
            executable_error = self._unknown_command_error(argv)
            if executable_error:
                return None, {"recognized": True, "valid": False, "family": "shell", "reason": executable_error}
            return None, {"recognized": False, "valid": True, "family": "unknown"}
        if family not in {"python", "python_exec"} and self._command_has_shell_chaining(command):
            return None, self._validation_result(family=family, valid=False, reason="shell chaining/redirection is not allowed for validated command families")
        executable = str(argv[0]).strip().strip("'\"")
        missing = None if os.path.isabs(executable) else shutil.which(executable)
        if missing is None and not Path(executable).exists():
            return None, self._validation_result(family=family, valid=False, reason=f"executable not found: {executable}")
        path_error = self._validate_path_args(argv, cwd)
        if path_error:
            return None, self._validation_result(family=family, valid=False, reason=path_error)
        if family == "git":
            subcommand = argv[1] if len(argv) > 1 else ""
            rejected = {"reset", "clean", "rebase", "push", "checkout", "restore", "switch"}
            allowed = {"status", "diff", "log", "show", "add", "commit", "branch", "rev-parse"}
            if subcommand in rejected or subcommand not in allowed:
                return None, self._validation_result(family=family, valid=False, reason=f"git {subcommand or '(missing)'} is not allowed through run_shell")
        elif family == "python":
            try:
                module = argv[argv.index("-m") + 1]
            except (ValueError, IndexError):
                return None, {"recognized": False, "valid": True, "family": "unknown"}
            if module not in {"unittest", "pytest", "pip"}:
                return None, {"recognized": False, "valid": True, "family": "unknown"}
            if module == "pip" and any(part in argv for part in {"install", "uninstall", "download"}):
                return None, self._validation_result(family=family, valid=False, reason="pip environment mutation is not allowed through run_shell")
        elif family == "python_exec":
            pass
        elif family in {"pytest", "ruff", "mypy", "pyright", "tsc"}:
            mutating_flags = {"--fix", "--write", "--watch"}
            if any(flag == token or token.startswith(flag + "=") for flag in mutating_flags for token in argv[1:]):
                return None, self._validation_result(family=family, valid=False, reason=f"{family} mutating/watch flags are not allowed through run_shell")
        elif family == "go":
            if tuple(argv[1:2]) != ("test",):
                return None, self._validation_result(family=family, valid=False, reason="go command is not an allowed validation command")
        elif family == "cargo":
            if tuple(argv[1:2]) not in {("test",), ("check",)}:
                return None, self._validation_result(family=family, valid=False, reason="cargo command is not an allowed validation command")
        elif family in {"gradle", "gradlew", "gradlew.bat"}:
            if "test" not in argv[1:]:
                return None, self._validation_result(family=family, valid=False, reason="gradle command is not an allowed validation command")
        elif family == "cmake":
            if "-S" not in argv or "-B" not in argv:
                return None, self._validation_result(family=family, valid=False, reason="cmake command must be a configure/check command")
        elif family == "ctest":
            pass
        elif family == "node":
            if "--check" not in argv[1:]:
                return None, self._validation_result(family=family, valid=False, reason="node command is not an allowed validation command")
        elif family in {"npm", "pnpm", "yarn"}:
            allowed_sequences = {("test",), ("run", "test"), ("run", "lint"), ("run", "typecheck")}
            sequence = tuple(argv[1:3]) if len(argv) > 2 and argv[1] == "run" else tuple(argv[1:2])
            if sequence not in allowed_sequences:
                return None, self._validation_result(family=family, valid=False, reason=f"{family} command is not an allowed validation command")
        return argv, self._validation_result(family=family, valid=True, argv=argv)

    def _nearest_existing_paths(self, raw_path: str, limit: int = 5) -> list[str]:
        needle = raw_path.replace("\\", "/").strip().strip("'\"")
        needle_name = Path(needle).name.lower()
        if not needle_name:
            return []
        candidates: list[tuple[int, str]] = []
        for path in self._iter_workspace_files(self.workspace_root, limit=50000):
            rel = self.relative_label(path)
            score = 0
            lowered = rel.lower()
            if needle.lower() in lowered:
                score += 4
            if needle_name and needle_name in Path(rel).name.lower():
                score += 6
            if score:
                candidates.append((score, rel))
            if len(candidates) > 200:
                break
        return [rel for _, rel in sorted(candidates, key=lambda item: (-item[0], item[1]))[:limit]]

    def run_shell(self, command: str, cwd: str = ".", timeout: int = 30) -> dict[str, Any]:
        working_dir = self.resolve_path(cwd, allow_missing=False)
        relative_cwd = self.relative_label(working_dir)
        argv, validation = self._validate_common_command(command, working_dir)
        if validation.get("valid") is False:
            reason = str(validation.get("reason") or "invalid command")
            error_class = "command_not_found" if "executable not found" in reason else "invalid_args"
            return {
                "ok": False,
                "tool": "run_shell",
                "cwd": relative_cwd,
                "summary": f"Command rejected before execution: {reason}",
                "error_class": error_class,
                "validation": validation,
            }
        approved, reason = self._approve_shell(command, relative_cwd)
        if not approved:
            return {"ok": False, "tool": "run_shell", "cwd": relative_cwd, "summary": reason}
        run_args: str | list[str] = command
        shell_kwargs: dict[str, Any] = {"shell": True}
        if validation.get("recognized") is True and argv:
            run_args = argv
            shell_kwargs = {"shell": False}
        if os.name == "nt":
            powershell = self._windows_powershell()
            if shell_kwargs.get("shell") is True and powershell and self._looks_like_powershell_command(command):
                run_args = [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command]
                shell_kwargs = {"shell": False}
        completed = self._run_process(run_args, cwd=working_dir, timeout=timeout, **shell_kwargs)
        output = self._collect_process_output(completed)
        result = {
            "ok": completed.returncode == 0,
            "tool": "run_shell",
            "cwd": relative_cwd,
            "exit_code": completed.returncode,
            "output": output,
        }
        if validation.get("recognized") is True:
            result["validation"] = validation
        if completed.returncode != 0:
            result["error_class"] = self.classify_error(output)
            if result["error_class"] in {"path_missing", "cwd_git"}:
                result["suggested_paths"] = self._nearest_existing_paths(command)
        return result

    def run_test(self, command: str | None = None, cwd: str = ".", timeout: int = 1200) -> dict[str, Any]:
        selected_command = command.strip() if isinstance(command, str) and command.strip() else self.default_test_command
        discovered = None
        if not selected_command:
            validators = self.discover_validators(cwd)
            candidates = [
                item
                for item in validators.get("validators", [])
                if isinstance(item, dict) and item.get("kind") == "test" and item.get("command") and item.get("available") is True
            ]
            preferred = [item for item in candidates if "unittest discover" in str(item.get("command", ""))]
            selected = (preferred or candidates)[:1]
            if selected:
                selected_command = str(selected[0]["command"])
                discovered = validators
            else:
                return {
                    "ok": False,
                    "tool": "run_test",
                    "summary": "No test command is configured and no runnable test validator was discovered. Set --test-cmd or pass a command to run_test.",
                    "validators": validators.get("validators", []),
                }
        normalized = None
        try:
            result = self.run_shell(selected_command, cwd=cwd, timeout=timeout)
        except ValueError as exc:
            if "Path escapes the workspace" not in str(exc) or str(cwd).strip() in {"", "."}:
                raise
            result = self.run_shell(selected_command, cwd=".", timeout=timeout)
            normalized = f"Ignored run_test cwd outside workspace: {cwd}"
        result["tool"] = "run_test"
        result["command"] = selected_command
        if discovered is not None:
            result["discovered"] = True
            result["validators"] = discovered.get("validators", [])
            prefix = f"Discovered test command: {selected_command}."
            if result.get("ok"):
                result["summary"] = prefix
            else:
                result["summary"] = f"{prefix} {result.get('output', '')[:180]}"
        if normalized:
            result["normalized"] = normalized
            result["summary"] = normalized if result.get("ok") else f"{normalized}. {result.get('output', '')[:180]}"
        return result
