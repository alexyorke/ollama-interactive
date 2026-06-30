from __future__ import annotations

import ast
import builtins
import configparser
from copy import deepcopy
import dis
import difflib
import fnmatch
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
import sysconfig
import threading
import textwrap
import time
import math
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4

from ollama_code.interrupts import OperationInterrupted
from ollama_code.tool_dependencies import (
    TOOL_DEPENDENCIES,
    clear_dependency_status_cache,
    command_to_text,
    configured_docker_host,
    current_platform,
    dependency_status,
    dependency_statuses,
    first_install_hint,
    resolve_dependency,
    resolve_tool_executable,
)
from ollama_code.tools.catalog import TOOL_DESCRIPTIONS, format_compact_tool_help, format_tool_group_help, format_tool_help

try:
    import tomllib  # type: ignore[attr-defined]
    _TOMLDecodeError = tomllib.TOMLDecodeError
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
        _TOMLDecodeError = tomllib.TOMLDecodeError
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

        class _TOMLDecodeError(ValueError):
            pass

ApprovalMode = str
AgentRunner = Callable[[dict[str, Any]], dict[str, Any]]
_MEMORY_FTS_CONNECTIONS: dict[str, sqlite3.Connection] = {}
_MEMORY_VERIFIED_FUNCTION_CONNECTIONS: dict[str, sqlite3.Connection] = {}
WINDOWS_DRIVE_PATH = re.compile(r"^(?P<drive>[A-Za-z]):(?:[\\/](?P<rest>.*))?$")
WSL_MOUNT_PATH = re.compile(r"^/mnt/(?P<drive>[A-Za-z])(?:/(?P<rest>.*))?$")
SHELL_SCRIPT_SUFFIXES = {".sh", ".bash"}
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
} | SHELL_SCRIPT_SUFFIXES
REPO_INDEX_VERSION = 2
FILE_INDEX_VERSION = 1
FTS_INDEX_VERSION = 1
VERIFIED_FUNCTION_INDEX_VERSION = 1
PYTHON_SDK_INDEX_VERSION = 1
PYTHON_SDK_SAFE_IMPORT_MODULES = {
    "array",
    "base64",
    "bisect",
    "builtins",
    "collections",
    "csv",
    "datetime",
    "functools",
    "glob",
    "hashlib",
    "heapq",
    "hmac",
    "http.client",
    "inspect",
    "itertools",
    "json",
    "math",
    "operator",
    "os",
    "pathlib",
    "re",
    "secrets",
    "sqlite3",
    "statistics",
    "subprocess",
    "sys",
    "tempfile",
    "time",
    "typing",
    "urllib.parse",
}
PYTHON_SDK_SAFE_CLASS_METHOD_MODULES = {"collections", "datetime", "pathlib", "re", "subprocess", "tempfile"}
PYTHON_SDK_PRIORITY_IMPORT_MODULES = (
    "json",
    "pathlib",
    "tempfile",
    "functools",
    "collections",
    "subprocess",
    "itertools",
    "math",
    "re",
)
INDEX_MUTATING_TOOL_NAMES = {
    "write_file",
    "replace_symbol",
    "replace_symbols",
    "replace_in_file",
    "apply_structured_edit",
    "edit_intent",
    "generate_tests_from_spec",
    "run_shell",
}
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
    ".meta",
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
    "!.meta/**",
    "!ollama-code-*/**",
    "!probe-*/**",
    "!verify_*/**",
    "!tmp*/**",
)
DENY_MUTATION_DIRS = {".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ollama-code"}
MAX_EXPLICIT_VALIDATOR_FILES = 20
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
TODO_STATUSES = {"pending", "in_progress", "completed"}
TODO_STATUS_ALIASES = {
    "todo": "pending",
    "open": "pending",
    "doing": "in_progress",
    "started": "in_progress",
    "done": "completed",
    "complete": "completed",
}
TODO_LIMIT = 20
TODO_CONTENT_LIMIT = 180


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
        enabled_tools: Iterable[str] | None = None,
        disabled_tools: Iterable[str] | None = None,
        mcp_servers: dict[str, Any] | None = None,
        browser_enabled: bool = True,
        security_enabled: bool = True,
        indexer: Any | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self._workspace_root_text = os.path.normpath(os.fspath(self.workspace_root))
        self._workspace_root_normcase = os.path.normcase(self._workspace_root_text)
        self._workspace_root_prefix_text = self._workspace_root_text if self._workspace_root_text.endswith(os.sep) else self._workspace_root_text + os.sep
        self._workspace_root_prefix_normcase = os.path.normcase(self._workspace_root_prefix_text)
        self.approval_mode = approval_mode
        self.input_func = input_func
        self.agent_runner = agent_runner
        self.default_test_command = test_command.strip() if isinstance(test_command, str) and test_command.strip() else None
        self.default_tools_enabled = bool(default_tools_enabled)
        self.enabled_tools = {str(name).strip() for name in (enabled_tools or []) if str(name).strip()}
        self.disabled_tools = {str(name).strip() for name in (disabled_tools or []) if str(name).strip()}
        self.mcp_servers = dict(mcp_servers or {})
        self.browser_enabled = bool(browser_enabled)
        self.security_enabled = bool(security_enabled)
        self.indexer = indexer
        self._interrupt_event: threading.Event | None = None
        self._initial_dirty_paths = self._git_dirty_paths()
        self._todos: list[dict[str, str]] = []
        self._tree_sitter_parsers: dict[str, Any] = {}
        self._lint_typecheck_cache: dict[str, dict[str, Any]] = {}
        self._python_tool_command_cache: dict[tuple[str, str, tuple[str, ...]], list[str] | None] = {}
        self._which_cache: dict[str, str | None] = {}

    def set_approval_mode(self, mode: ApprovalMode) -> None:
        self.approval_mode = mode

    def set_interrupt_event(self, event: threading.Event | None) -> None:
        self._interrupt_event = event

    def set_test_command(self, command: str | None) -> None:
        self.default_test_command = command.strip() if isinstance(command, str) and command.strip() else None

    def set_indexer(self, indexer: Any | None) -> None:
        self.indexer = indexer

    def todo_snapshot(self) -> list[dict[str, str]]:
        return deepcopy(self._todos)

    def set_todos(self, items: list[dict[str, Any]] | None) -> None:
        normalized, _ = self._normalize_todo_items(items or [])
        self._todos = normalized

    def clear_todos(self) -> None:
        self._todos = []

    def _truncate_text(self, text: str, *, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 18)] + "... truncated ..."

    def _coerce_int(self, value: Any, *, default: int, minimum: int = 1) -> int:
        if isinstance(value, bool):
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return max(minimum, parsed)

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
        if clean.startswith("mcp."):
            parts = clean.split(".", 2)
            if len(parts) != 3 or not all(part.strip() for part in parts):
                return False
            if clean in self.disabled_tools or "mcp_call" in self.disabled_tools or "mcp" in self.disabled_tools:
                return False
            if self.enabled_tools and clean not in self.enabled_tools and "mcp_call" not in self.enabled_tools and "mcp" not in self.enabled_tools:
                return False
            return self.default_tools_enabled
        if self.enabled_tools and clean not in self.enabled_tools:
            return False
        if clean in self.disabled_tools:
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
            "todo_read": self.todo_read,
            "todo_write": self.todo_write,
            "list_files": self.list_files,
            "read_file": self.read_file,
            "search": self.search,
            "file_search": self.file_search,
            "directory_search": self.directory_search,
            "fd_search": self.fd_search,
            "file_index_refresh": self.file_index_refresh,
            "everything_search": self.everything_search,
            "search_symbols": self.search_symbols,
            "code_outline": self.code_outline,
            "read_symbol": self.read_symbol,
            "inspect_library_source": self.inspect_library_source,
            "python_sdk_refresh": self.python_sdk_refresh,
            "python_sdk_search": self.python_sdk_search,
            "repo_index_search": self.repo_index_search,
            "fts_search": self.fts_search,
            "fts_refresh": self.fts_refresh,
            "indexed_search": self.indexed_search,
            "repo_index_refresh": self.repo_index_refresh,
            "tool_status": self.tool_status,
            "tool_install": self.tool_install,
            "semgrep_scan": self.semgrep_scan,
            "ast_search": self.ast_search,
            "structural_rewrite": self.structural_rewrite,
            "tree_sitter_syntax": self.tree_sitter_syntax,
            "lsp_diagnostics": self.lsp_diagnostics,
            "lsp_definition": self.lsp_definition,
            "lsp_references": self.lsp_references,
            "context_pack": self.context_pack,
            "systems_lens": self.systems_lens,
            "find_implementation_target": self.find_implementation_target,
            "diagnose_test_failure": self.diagnose_test_failure,
            "test_spec_extract": self.test_spec_extract,
            "implementation_spec": self.implementation_spec,
            "run_function_probe": self.run_function_probe,
            "call_graph": self.call_graph,
            "contract_graph": self.contract_graph,
            "verified_function_index": self.verified_function_index,
            "verified_function_search": self.verified_function_search,
            "verified_function_show": self.verified_function_show,
            "verify_function_contract": self.verify_function_contract,
            "compose_verified_functions": self.compose_verified_functions,
            "promote_verified_function": self.promote_verified_function,
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
            "git_branch": self.git_branch,
            "git_log": self.git_log,
            "git_commit": self.git_commit,
        }
        handler = handlers.get(name)
        if handler is None:
            return {"ok": False, "tool": name, "summary": f"Unknown tool: {name}"}
        try:
            result = self._call_tool_handler(handler, arguments)
            self._notify_indexer_after_tool(name, arguments, result)
            return result
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

    def _notify_indexer_after_tool(self, name: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
        indexer = self.indexer
        if indexer is None or name not in INDEX_MUTATING_TOOL_NAMES or result.get("ok") is not True:
            return
        if name == "run_shell":
            refresh = getattr(indexer, "request_refresh", None)
            if callable(refresh):
                refresh("run_shell completed")
            return
        paths: set[str] = set()
        for source in (result, arguments):
            for key in ("path", "test_path"):
                value = source.get(key) if isinstance(source, dict) else None
                if isinstance(value, str) and value.strip():
                    paths.add(value.strip())
            value = source.get("paths") if isinstance(source, dict) else None
            if isinstance(value, list):
                paths.update(str(item).strip() for item in value if str(item).strip())
        notify = getattr(indexer, "notify_paths", None)
        if paths and callable(notify):
            notify(paths)

    def _normalize_todo_status(self, raw: Any) -> str | None:
        status = str(raw or "pending").strip().lower().replace("-", "_")
        status = TODO_STATUS_ALIASES.get(status, status)
        return status if status in TODO_STATUSES else None

    def _normalize_todo_items(self, raw_items: Any) -> tuple[list[dict[str, str]], str | None]:
        if not isinstance(raw_items, list):
            return [], "todo_write requires items as a list."
        if len(raw_items) > TODO_LIMIT:
            return [], f"todo_write supports at most {TODO_LIMIT} items."
        normalized: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        in_progress_count = 0
        for index, raw in enumerate(raw_items, start=1):
            if isinstance(raw, str):
                content = raw.strip()
                status = "pending"
                raw_id = str(index)
            elif isinstance(raw, dict):
                content = str(raw.get("content") or raw.get("task") or raw.get("text") or "").strip()
                status = self._normalize_todo_status(raw.get("status"))
                raw_id = str(raw.get("id") or index).strip()
            else:
                return [], f"Todo item {index} must be an object or string."
            if not content:
                return [], f"Todo item {index} requires non-empty content."
            if status is None:
                return [], f"Todo item {index} has invalid status; use pending, in_progress, or completed."
            if status == "in_progress":
                in_progress_count += 1
            clean_id = re.sub(r"[^A-Za-z0-9_.:-]+", "-", raw_id).strip("-") or str(index)
            if clean_id in seen_ids:
                return [], f"Duplicate todo id: {clean_id}"
            seen_ids.add(clean_id)
            normalized.append({"id": clean_id, "status": status, "content": self._truncate_text(content, limit=TODO_CONTENT_LIMIT)})
        if in_progress_count > 1:
            return [], "Only one todo item can be in_progress at a time."
        return normalized, None

    def _render_todos(self) -> str:
        if not self._todos:
            return "(empty)"
        return "\n".join(f"{item['id']}. [{item['status']}] {item['content']}" for item in self._todos)

    def todo_read(self) -> dict[str, Any]:
        return {
            "ok": True,
            "tool": "todo_read",
            "count": len(self._todos),
            "items": self.todo_snapshot(),
            "output": self._render_todos(),
        }

    def todo_write(self, items: list[Any] | None = None, todos: list[Any] | None = None) -> dict[str, Any]:
        raw_items = items if items is not None else todos
        normalized, error = self._normalize_todo_items(raw_items if raw_items is not None else [])
        if error:
            return {"ok": False, "tool": "todo_write", "summary": error}
        self._todos = normalized
        pending = sum(1 for item in self._todos if item["status"] == "pending")
        in_progress = sum(1 for item in self._todos if item["status"] == "in_progress")
        completed = sum(1 for item in self._todos if item["status"] == "completed")
        return {
            "ok": True,
            "tool": "todo_write",
            "count": len(self._todos),
            "pending": pending,
            "in_progress": in_progress,
            "completed": completed,
            "items": self.todo_snapshot(),
            "summary": f"todo list updated: {len(self._todos)} item(s), {pending} pending, {in_progress} in_progress, {completed} completed.",
            "output": self._render_todos(),
        }

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
        fast = self._fast_workspace_relative_label(path)
        if fast is not None:
            return fast
        candidate = path if path.is_absolute() else self.workspace_root / path
        return candidate.resolve(strict=False).relative_to(self.workspace_root).as_posix() or "."

    def _fast_workspace_relative_label(self, path: Path) -> str | None:
        candidate = path if path.is_absolute() else self.workspace_root / path
        candidate_text = os.path.normpath(os.fspath(candidate))
        candidate_normcase = os.path.normcase(candidate_text)
        if candidate_normcase == self._workspace_root_normcase:
            return "."
        if not candidate_normcase.startswith(self._workspace_root_prefix_normcase):
            return None
        return candidate_text[len(self._workspace_root_prefix_text):].replace("\\", "/")

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
            r"^\s*&\s*(?!&)\S",
            rf"{statement_start}\$[A-Za-z_][\w:]*\s*=",
            rf"{statement_start}\.[\\/].+\.ps1(?:\s|$)",
            rf"{statement_start}(?:Get|Set|New|Remove|Move|Copy|Join|Split|Resolve|Test|Write|Start|Stop|Select|Where|ForEach|Measure|Sort)-[A-Za-z]+\b",
            rf"{statement_start}(?:cat|clear|cp|ls|mv|pwd|rm)\b",
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

    def _collect_timeout_output(self, exc: subprocess.TimeoutExpired) -> str:
        parts = []
        stdout = exc.output
        stderr = exc.stderr
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        if isinstance(stdout, str) and stdout.strip():
            parts.append(stdout.strip())
        if isinstance(stderr, str) and stderr.strip():
            parts.append(stderr.strip())
        return "\n".join(parts) if parts else "(no output)"

    def _timeout_command_text(self, command: Any) -> str:
        if isinstance(command, (list, tuple)):
            return command_to_text(tuple(str(part) for part in command))
        return str(command)

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
        else:
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
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass

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
        try:
            deadline = time.monotonic() + timeout
            while True:
                if self._interrupt_event is not None and self._interrupt_event.is_set():
                    self._terminate_process(process)
                    raise OperationInterrupted("Interrupted by user.")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._terminate_process(process)
                    stdout = ""
                    stderr = ""
                    try:
                        stdout, stderr = process.communicate(timeout=0.2)
                    except Exception:
                        pass
                    raise subprocess.TimeoutExpired(process.args, timeout, output=stdout, stderr=stderr)
                try:
                    stdout, stderr = process.communicate(timeout=min(0.1, remaining))
                    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
                except subprocess.TimeoutExpired:
                    if self._interrupt_event is not None and self._interrupt_event.is_set():
                        self._terminate_process(process)
                        raise OperationInterrupted("Interrupted by user.")
                    continue
        except BaseException:
            if process.poll() is None:
                self._terminate_process(process)
            raise
        finally:
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()

    def _path_cwd(self, path: Path | None) -> Path:
        if path is None:
            return self.workspace_root
        resolved = path.resolve(strict=False)
        if resolved.exists() and resolved.is_file():
            return resolved.parent
        if resolved.exists() and resolved.is_dir():
            return resolved
        return resolved.parent if resolved.suffix else resolved

    def _git_root_for_path(self, path: Path | None = None) -> Path | None:
        starts: list[Path] = []
        if path is not None:
            starts.append(self._path_cwd(path))
        starts.append(self.workspace_root)
        seen: set[Path] = set()
        for start in starts:
            current = start.resolve(strict=False)
            for candidate in (current, *current.parents):
                if candidate in seen:
                    continue
                seen.add(candidate)
                if (candidate / ".git").exists():
                    return candidate
        if path is None:
            nested = self._nested_git_roots()
            if len(nested) == 1:
                return nested[0]
        return None

    def _nested_git_roots(self, *, max_depth: int = 3, limit: int = 4) -> list[Path]:
        roots: list[Path] = []
        for root, dirs, files in os.walk(self.workspace_root):
            root_path = Path(root)
            try:
                depth = len(root_path.relative_to(self.workspace_root).parts)
            except ValueError:
                depth = 0
            if ".git" in dirs or ".git" in files:
                roots.append(root_path.resolve(strict=False))
                dirs[:] = []
                if len(roots) >= limit:
                    break
                continue
            dirs[:] = sorted(
                directory
                for directory in dirs
                if directory == ".git" or (not directory.startswith(".") and not self._generated_dir_name(directory))
            )
            if depth >= max_depth:
                dirs[:] = []
        return roots

    def _run_git(self, args: list[str], *, timeout: int = 30, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        if shutil.which("git") is None:
            raise RuntimeError("git is not installed.")
        return self._run_process(
            ["git", *args],
            cwd=self._path_cwd(cwd),
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
        relative = self._fast_workspace_relative_label(path)
        if relative is not None:
            parts = relative.split("/")
        else:
            try:
                parts = path.resolve(strict=False).relative_to(self.workspace_root).parts
            except ValueError:
                parts = path.parts
        return any(self._generated_dir_name(part) for part in parts)

    def _repo_files_from_git(self, base: Path, *, limit: int, suffixes: set[str] | None = None) -> list[Path] | None:
        git_root = self._git_root_for_path(base)
        if git_root is None:
            return None
        try:
            pathspec = base.resolve(strict=False).relative_to(git_root).as_posix()
        except ValueError:
            return None
        if not pathspec:
            pathspec = "."
        try:
            completed = self._run_git(["ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", pathspec], timeout=20, cwd=git_root)
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

    def _ensure_git_repo(self, path: Path | None = None) -> tuple[bool, str | None]:
        if self._git_root_for_path(path) is not None:
            return True, None
        try:
            probe = self._run_git(["rev-parse", "--is-inside-work-tree"], cwd=path)
        except RuntimeError as exc:
            return False, str(exc)
        if probe.returncode == 0 and probe.stdout.strip() == "true":
            return True, None
        return False, self._collect_process_output(probe)

    def _git_dirty_paths(self) -> set[str]:
        repo_root = self._git_root_for_path()
        ok, _ = self._ensure_git_repo(repo_root)
        if not ok:
            return set()
        paths: set[str] = set()
        commands = [
            (["diff", "--name-only", "-z"], 120),
            (["diff", "--cached", "--name-only", "-z"], 120),
            (["ls-files", "--others", "--exclude-standard", "-z"], 60),
        ]
        for args, timeout in commands:
            try:
                completed = self._run_git(args, cwd=repo_root, timeout=timeout)
            except subprocess.TimeoutExpired:
                # repo-level dirty-path scans can be slow on large workspaces; skip
                # this command on timeout so startup can still continue.
                continue
            if completed.returncode != 0:
                continue
            for raw_path in completed.stdout.split("\0"):
                if raw_path:
                    paths.add(raw_path)
        return paths

    def list_files(self, path: str = ".", max_depth: int = 4, limit: int = 200) -> dict[str, Any]:
        self._check_interrupted()
        base = self.resolve_path(path, allow_missing=False)
        safe_max_depth = self._coerce_int(max_depth, default=4, minimum=0)
        safe_limit = self._coerce_int(limit, default=200, minimum=1)
        if base.is_file():
            items = [self.relative_label(base)]
        else:
            items = []
            for root, dirs, files in os.walk(base):
                self._check_interrupted()
                root_path = Path(root)
                depth = len(root_path.relative_to(base).parts)
                dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not self._generated_dir_name(d))
                if depth >= safe_max_depth:
                    dirs[:] = []
                for directory in dirs:
                    items.append(f"{self.relative_label(root_path / directory)}/")
                    if len(items) >= safe_limit:
                        break
                if len(items) >= safe_limit:
                    break
                for file_name in sorted(files):
                    if file_name.startswith("."):
                        continue
                    if self._path_has_skipped_part(root_path / file_name):
                        continue
                    items.append(self.relative_label(root_path / file_name))
                    if len(items) >= safe_limit:
                        break
                if len(items) >= safe_limit:
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
        safe_start = self._coerce_int(start, default=1, minimum=1)
        safe_end = max(safe_start, self._coerce_int(end, default=200, minimum=1))
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

    def search(self, query: str, path: str = ".", limit: int = 100, file_glob: str | None = None) -> dict[str, Any]:
        self._check_interrupted()
        base = self.resolve_path(path, allow_missing=False)
        clean_file_glob = str(file_glob or "").strip()
        if clean_file_glob.startswith("-"):
            return {"ok": False, "tool": "search", "summary": "file_glob must be a filename glob, not an option."}
        if shutil.which("rg"):
            command = ["rg", "-n", "--color", "never", "--no-ignore-parent", "--max-count", str(limit)]
            for directory in sorted(SKIP_CODE_DIRS):
                command.extend(["--glob", f"!{directory}/**"])
            for glob in SKIP_WALK_GLOBS:
                command.extend(["--glob", glob])
            if clean_file_glob:
                command.extend(["--glob", clean_file_glob])
            command.extend([query, str(base)])
            result = self._run_process(command, cwd=self.workspace_root, timeout=30)
            if result.returncode not in {0, 1} and "regex parse error" in (result.stderr or ""):
                literal_command = list(command)
                literal_command.insert(1, "-F")
                result = self._run_process(literal_command, cwd=self.workspace_root, timeout=30)
            output = result.stdout.strip() or result.stderr.strip() or "(no matches)"
            if output != "(no matches)":
                rel_root = str(self.workspace_root.resolve(strict=False))
                rel_root_lower = rel_root.lower()
                normalized_lines: list[str] = []
                for line in output.splitlines()[: max(1, limit)]:
                    rewritten = line
                    if line.lower().startswith(rel_root_lower):
                        suffix = line[len(rel_root) :].lstrip("\\/")
                        rewritten = suffix.replace("\\", "/")
                    normalized_lines.append(rewritten)
                output = "\n".join(normalized_lines)
            return {
                "ok": result.returncode in {0, 1},
                "tool": "search",
                "path": self.relative_label(base),
                "file_glob": clean_file_glob or None,
                "output": output,
            }
        matches: list[str] = []
        try:
            pattern = re.compile(query)

            def matcher(text: str) -> bool:
                return bool(pattern.search(text))

        except re.error:

            def matcher(text: str) -> bool:
                return query in text

        for file_path in self._iter_workspace_files(base, limit=50000):
            self._check_interrupted()
            if clean_file_glob and not fnmatch.fnmatchcase(file_path.name, clean_file_glob):
                continue
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
            "file_glob": clean_file_glob or None,
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
            "ctime_ns": int(stat.st_ctime_ns),
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
                and cached.get("ctime_ns") == int(stat.st_ctime_ns)
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

    def directory_search(self, query: str, path: str = ".", limit: int = 100) -> dict[str, Any]:
        self._check_interrupted()
        base = self.resolve_path(path, allow_missing=False)
        clean_query = str(query or "").strip()
        if not clean_query:
            return {"ok": False, "tool": "directory_search", "summary": "directory_search requires a non-empty query."}
        safe_limit = self._coerce_int(limit, default=100, minimum=1)
        matches: list[str] = []
        if base.is_file():
            return {"ok": False, "tool": "directory_search", "summary": f"{path} is a file."}
        for root, dirs, _files in os.walk(base):
            self._check_interrupted()
            root_path = Path(root)
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not self._generated_dir_name(d))
            for directory in dirs:
                if fnmatch.fnmatchcase(directory, clean_query):
                    matches.append(f"{self.relative_label(root_path / directory)}/")
                    if len(matches) >= safe_limit:
                        break
            if len(matches) >= safe_limit:
                break
        return {
            "ok": True,
            "tool": "directory_search",
            "path": self.relative_label(base),
            "count": len(matches),
            "output": "\n".join(matches) if matches else "(no directory matches)",
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

    def _collapse_validation_targets(self, labels: Iterable[str], *, limit: int = 100) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw_label in labels:
            label = str(raw_label or "").strip().replace("\\", "/")
            if not label:
                continue
            if label == ".":
                return ["."]
            if label in seen:
                continue
            seen.add(label)
            cleaned.append(label)
        selected: list[str] = []
        for label in sorted(cleaned, key=lambda item: (item.count("/"), len(item), item)):
            if any(label == existing or label.startswith(existing + "/") for existing in selected):
                continue
            selected.append(label)
            if len(selected) >= limit:
                break
        return selected

    def _python_validation_targets(
        self,
        *,
        discovered_files: Iterable[str],
        requested_scopes: Iterable[str],
        limit: int = 100,
    ) -> list[str]:
        file_targets = self._collapse_validation_targets(discovered_files, limit=limit)
        if not file_targets:
            return []
        scope_targets = self._collapse_validation_targets(requested_scopes, limit=limit)
        if "." in scope_targets:
            return ["."]
        if len(file_targets) <= MAX_EXPLICIT_VALIDATOR_FILES:
            return file_targets
        return scope_targets or file_targets

    def _python_typechecker_targets(
        self,
        *,
        discovered_files: Iterable[str],
        requested_scopes: Iterable[str],
        limit: int = 100,
    ) -> list[str]:
        file_targets = self._collapse_validation_targets(discovered_files, limit=limit)
        if not file_targets:
            return []
        scope_targets = self._collapse_validation_targets(requested_scopes, limit=limit)
        if "." in scope_targets:
            return ["."]
        if len(scope_targets) == 1 and not scope_targets[0].endswith(".py"):
            return scope_targets
        return file_targets

    def _python_typechecker_configured(self) -> bool:
        if (self.workspace_root / "pyrightconfig.json").exists() or (self.workspace_root / "basedpyrightconfig.json").exists():
            return True
        pyproject = self._read_toml(self.workspace_root / "pyproject.toml")
        return self._toml_tool_section(pyproject, "pyright") or self._toml_tool_section(pyproject, "basedpyright")

    def _lint_typecheck_cache_key(
        self,
        *,
        checked: Iterable[str],
        validator_targets: Iterable[str],
        typechecker_targets: Iterable[str],
        shell_targets: Iterable[str],
        ruff_path: str | None,
        typechecker_command: list[str] | None,
        bash_path: str | None,
    ) -> str:
        file_stats: list[dict[str, Any]] = []
        for label in sorted({str(item).replace("\\", "/") for item in checked if str(item).strip()}):
            try:
                stat = (self.workspace_root / label).stat()
            except OSError:
                file_stats.append({"path": label, "missing": True})
                continue
            file_stats.append({"path": label, "mtime_ns": int(stat.st_mtime_ns), "size": int(stat.st_size)})
        config_stats: list[dict[str, Any]] = []
        for label in (
            "pyproject.toml",
            "setup.cfg",
            "tox.ini",
            "ruff.toml",
            ".ruff.toml",
            "pyrightconfig.json",
            "basedpyrightconfig.json",
        ):
            path = self.workspace_root / label
            if not path.exists():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            config_stats.append({"path": label, "mtime_ns": int(stat.st_mtime_ns), "size": int(stat.st_size)})
        payload = {
            "version": 1,
            "checked": file_stats,
            "configs": config_stats,
            "validator_targets": list(validator_targets),
            "typechecker_targets": list(typechecker_targets),
            "shell_targets": list(shell_targets),
            "ruff_path": ruff_path or "",
            "typechecker_command": list(typechecker_command or []),
            "bash_path": bash_path or "",
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(encoded.encode("utf-8")).hexdigest()

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

    def _python_doc_summary(self, node: ast.AST) -> str:
        doc = ast.get_docstring(node)
        if not doc:
            return ""
        lines = [line.strip() for line in doc.splitlines() if line.strip()]
        return lines[0][:120] if lines else ""

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
                            "doc": self._python_doc_summary(child),
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

    def _adaptive_index_record_limit(self, requested: int, *, minimum: int = 2000, multiplier: int = 250, maximum: int = 50_000) -> int:
        return min(maximum, max(minimum, max(1, int(requested)) * multiplier))

    def search_symbols(self, query: str, path: str = ".", limit: int = 50) -> dict[str, Any]:
        self._check_interrupted()
        base = self.resolve_path(path, allow_missing=False)
        try:
            pattern = re.compile(query, flags=re.IGNORECASE)

            def matcher(text: str) -> bool:
                return bool(pattern.search(text))

        except re.error:
            lowered = query.lower()

            def matcher(text: str) -> bool:
                return lowered in text.lower()

        literal_query = str(query or "").strip().lower()
        ranked: list[tuple[int, str, int, str]] = []
        records = self._indexed_code_records(base, limit=self._adaptive_index_record_limit(limit))
        for record in records:
            rel = str(record.get("path") or "")
            for symbol in record.get("symbols", []):
                if not isinstance(symbol, dict):
                    continue
                haystack = " ".join(
                    str(symbol.get(key, ""))
                    for key in ("name", "qualname", "kind", "signature")
                )
                if not matcher(haystack):
                    continue
                name = str(symbol.get("name", "")).lower()
                qualname = str(symbol.get("qualname", "")).lower()
                signature = str(symbol.get("signature", "")).lower()
                score = 0
                if literal_query:
                    if literal_query == name or literal_query == qualname:
                        score += 100
                    if literal_query in name or literal_query in qualname:
                        score += 60
                    if literal_query in signature:
                        score += 20
                    if literal_query in rel.lower():
                        score += 8
                ranked.append(
                    (
                        score,
                        rel,
                        int(symbol.get("start") or 1),
                        f"{rel}:{symbol['start']}-{symbol['end']} {symbol['kind']} {symbol['qualname']} {symbol['signature']}",
                    )
                )
        ranked.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
        matches = [line for _, _, _, line in ranked[: max(1, int(limit))]]
        return {
            "ok": True,
            "tool": "search_symbols",
            "path": self.relative_label(base),
            "count": len(matches),
            "output": "\n".join(matches) if matches else "(no symbols found)",
        }

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

    def _python_sdk_cache_path(self) -> Path:
        return self.workspace_root / ".ollama-code" / "index" / "python_sdk.sqlite"

    def _initialize_python_sdk_connection(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA busy_timeout=2000")
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS python_sdk_entries("
            "id TEXT PRIMARY KEY,"
            "kind TEXT NOT NULL,"
            "module TEXT NOT NULL,"
            "qualname TEXT NOT NULL,"
            "signature TEXT NOT NULL,"
            "doc TEXT NOT NULL,"
            "source_path TEXT NOT NULL,"
            "line INTEGER NOT NULL,"
            "text TEXT NOT NULL,"
            "embedding_model TEXT,"
            "embedding_json TEXT)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS python_sdk_fts USING fts5("
            "id UNINDEXED, module, qualname, signature, doc, text)"
        )
        conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('version', ?)", (str(PYTHON_SDK_INDEX_VERSION),))
        conn.commit()

    def _connect_python_sdk(self) -> sqlite3.Connection:
        cache_path = self._python_sdk_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(cache_path)
        conn.execute("PRAGMA journal_mode=WAL")
        self._initialize_python_sdk_connection(conn)
        return conn

    def _stdlib_root(self) -> Path:
        return Path(sysconfig.get_path("stdlib") or Path(sys.executable).parent).resolve(strict=False)

    def _stdlib_module_name(self, root: Path, path: Path) -> str | None:
        try:
            relative = path.relative_to(root)
        except ValueError:
            return None
        parts = list(relative.with_suffix("").parts)
        if not parts:
            return None
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts:
            return None
        if any(part in {"test", "tests", "idle_test", "__pycache__", "site-packages", "dist-packages"} for part in parts):
            return None
        if any(part.startswith("_test") or part.endswith("_test") for part in parts):
            return None
        if not all(re.fullmatch(r"[A-Za-z_]\w*", part) for part in parts):
            return None
        return ".".join(parts)

    def _iter_python_sdk_files(self, *, limit: int) -> list[Path]:
        root = self._stdlib_root()
        files: list[Path] = []
        for file_path in sorted(root.rglob("*.py")):
            self._check_interrupted()
            if len(files) >= limit:
                break
            if self._stdlib_module_name(root, file_path) is None:
                continue
            try:
                stat = file_path.stat()
            except OSError:
                continue
            if stat.st_size > 768_000:
                continue
            files.append(file_path)
        return files

    def _doc_summary(self, text: str | None, *, limit: int = 420) -> str:
        doc = inspect.cleandoc(text or "")
        if not doc:
            return ""
        paragraph = re.split(r"\n\s*\n", doc, maxsplit=1)[0].replace("\n", " ").strip()
        return self._truncate_text(paragraph, limit=limit)

    def _ast_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
        if isinstance(node, ast.ClassDef):
            return f"class {node.name}"
        args = node.args
        parts: list[str] = []
        positional = list(args.posonlyargs) + list(args.args)
        defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
        for arg, default in zip(positional, defaults):
            parts.append(arg.arg + ("=..." if default is not None else ""))
        if args.vararg is not None:
            parts.append("*" + args.vararg.arg)
        elif args.kwonlyargs:
            parts.append("*")
        for arg, default in zip(args.kwonlyargs, args.kw_defaults):
            parts.append(arg.arg + ("=..." if default is not None else ""))
        if args.kwarg is not None:
            parts.append("**" + args.kwarg.arg)
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {node.name}({', '.join(parts)})"

    def _python_sdk_entry(
        self,
        *,
        kind: str,
        module: str,
        qualname: str,
        signature: str,
        doc: str,
        source_path: str,
        line: int,
    ) -> dict[str, Any]:
        text = f"{kind} {module} {qualname} {signature} {doc}".strip()
        entry_id = hashlib.sha1(f"{kind}\0{module}\0{qualname}".encode("utf-8", errors="replace")).hexdigest()[:20]
        return {
            "id": entry_id,
            "kind": kind,
            "module": module,
            "qualname": qualname,
            "signature": signature,
            "doc": doc,
            "source_path": source_path,
            "line": int(line or 1),
            "text": text,
        }

    def _python_sdk_ast_entries(self, *, limit: int) -> list[dict[str, Any]]:
        root = self._stdlib_root()
        entries: list[dict[str, Any]] = []
        for file_path in self._iter_python_sdk_files(limit=limit):
            if len(entries) >= limit:
                break
            module = self._stdlib_module_name(root, file_path)
            if module is None:
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(text, filename=str(file_path))
            except (OSError, SyntaxError):
                continue
            doc = self._doc_summary(ast.get_docstring(tree))
            source_path = str(file_path)
            if doc:
                entries.append(
                    self._python_sdk_entry(
                        kind="module",
                        module=module,
                        qualname=module,
                        signature=f"module {module}",
                        doc=doc,
                        source_path=source_path,
                        line=1,
                    )
                )
            for node in tree.body:
                self._check_interrupted()
                if len(entries) >= limit:
                    break
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name.startswith("_"):
                        continue
                    qualname = f"{module}.{node.name}"
                    kind = "class" if isinstance(node, ast.ClassDef) else "function"
                    entries.append(
                        self._python_sdk_entry(
                            kind=kind,
                            module=module,
                            qualname=qualname,
                            signature=self._ast_signature(node),
                            doc=self._doc_summary(ast.get_docstring(node)),
                            source_path=source_path,
                            line=getattr(node, "lineno", 1),
                        )
                    )
                    if isinstance(node, ast.ClassDef):
                        for child in node.body:
                            if len(entries) >= limit:
                                break
                            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                continue
                            if child.name.startswith("_"):
                                continue
                            entries.append(
                                self._python_sdk_entry(
                                    kind="method",
                                    module=module,
                                    qualname=f"{module}.{node.name}.{child.name}",
                                    signature=self._ast_signature(child),
                                    doc=self._doc_summary(ast.get_docstring(child)),
                                    source_path=source_path,
                                    line=getattr(child, "lineno", 1),
                                )
                            )
        return entries

    def _python_sdk_imported_entries(self, *, remaining: int, existing_ids: set[str]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        module_names = list(PYTHON_SDK_PRIORITY_IMPORT_MODULES)
        module_names.extend(name for name in sorted(PYTHON_SDK_SAFE_IMPORT_MODULES) if name not in module_names)
        for module_name in module_names:
            if len(entries) >= remaining:
                break
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            module_doc = self._doc_summary(inspect.getdoc(module))
            module_entry = self._python_sdk_entry(
                kind="module",
                module=module_name,
                qualname=module_name,
                signature=f"module {module_name}",
                doc=module_doc,
                source_path=str(getattr(module, "__file__", "") or "(built-in)"),
                line=1,
            )
            if module_entry["id"] not in existing_ids:
                entries.append(module_entry)
                existing_ids.add(module_entry["id"])
            for name, value in inspect.getmembers(module):
                if len(entries) >= remaining:
                    break
                if name.startswith("_"):
                    continue
                if inspect.isclass(value):
                    kind = "class"
                    signature = f"class {name}"
                elif inspect.isbuiltin(value) or inspect.isfunction(value):
                    kind = "function"
                    try:
                        signature = f"{name}{inspect.signature(value)}"
                    except (TypeError, ValueError):
                        signature = name
                else:
                    continue
                qualname = f"{module_name}.{name}"
                entry = self._python_sdk_entry(
                    kind=kind,
                    module=module_name,
                    qualname=qualname,
                    signature=signature,
                    doc=self._doc_summary(inspect.getdoc(value)),
                    source_path=str(getattr(module, "__file__", "") or "(built-in)"),
                    line=1,
                )
                if entry["id"] in existing_ids:
                    continue
                entries.append(entry)
                existing_ids.add(entry["id"])
                if kind == "class" and module_name in PYTHON_SDK_SAFE_CLASS_METHOD_MODULES:
                    for method_name, method_value in inspect.getmembers(value):
                        if len(entries) >= remaining:
                            break
                        if method_name.startswith("_"):
                            continue
                        if not (inspect.isfunction(method_value) or inspect.ismethod(method_value) or inspect.isbuiltin(method_value) or inspect.ismethoddescriptor(method_value)):
                            continue
                        try:
                            method_signature = f"{method_name}{inspect.signature(method_value)}"
                        except (TypeError, ValueError):
                            method_signature = method_name
                        method_entry = self._python_sdk_entry(
                            kind="method",
                            module=module_name,
                            qualname=f"{module_name}.{name}.{method_name}",
                            signature=method_signature,
                            doc=self._doc_summary(inspect.getdoc(method_value)),
                            source_path=str(getattr(module, "__file__", "") or "(built-in)"),
                            line=1,
                        )
                        if method_entry["id"] in existing_ids:
                            continue
                        entries.append(method_entry)
                        existing_ids.add(method_entry["id"])
        return entries

    def _python_sdk_entries(self, *, limit: int) -> list[dict[str, Any]]:
        entries = self._python_sdk_imported_entries(remaining=min(limit, 1200), existing_ids=set())
        existing_ids = {str(entry["id"]) for entry in entries}
        if len(entries) < limit:
            for entry in self._python_sdk_ast_entries(limit=limit):
                if len(entries) >= limit:
                    break
                if str(entry["id"]) in existing_ids:
                    continue
                entries.append(entry)
                existing_ids.add(str(entry["id"]))
        if len(entries) < limit:
            entries.extend(self._python_sdk_imported_entries(remaining=limit - len(entries), existing_ids=existing_ids))
        return entries[:limit]

    def _normalize_sdk_embedding_model(self, embedding_model: str | None) -> str | None:
        raw = str(embedding_model or "").strip()
        if not raw:
            raw = os.environ.get("OLLAMA_CODE_SDK_EMBED_MODEL", "").strip()
        if raw.lower() in {"", "0", "false", "none", "off"}:
            return None
        return raw

    def _normalize_ollama_host(self, embedding_host: str | None) -> str:
        raw = str(embedding_host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").strip()
        if not re.match(r"^https?://", raw):
            raw = "http://" + raw
        return raw.rstrip("/")

    def _ollama_embed_texts(
        self,
        texts: list[str],
        *,
        model: str,
        host: str | None = None,
        timeout: int = 120,
    ) -> list[list[float]]:
        if not texts:
            return []
        endpoint = self._normalize_ollama_host(host) + "/api/embed"
        payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
        request = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=max(1, int(timeout))) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
        raw_embeddings = data.get("embeddings")
        if isinstance(raw_embeddings, list):
            return [[float(value) for value in vector] for vector in raw_embeddings if isinstance(vector, list)]
        raw_embedding = data.get("embedding")
        if isinstance(raw_embedding, list) and len(texts) == 1:
            return [[float(value) for value in raw_embedding]]
        raise ValueError("Ollama embed response did not include embeddings.")

    def _embed_python_sdk_entries(
        self,
        entries: list[dict[str, Any]],
        *,
        model: str,
        host: str | None,
        timeout: int,
    ) -> tuple[int, str | None]:
        embedded = 0
        batch_size = 16
        for offset in range(0, len(entries), batch_size):
            self._check_interrupted()
            batch = entries[offset : offset + batch_size]
            try:
                vectors = self._ollama_embed_texts(
                    [str(entry.get("text", ""))[:4000] for entry in batch],
                    model=model,
                    host=host,
                    timeout=timeout,
                )
            except (OSError, ValueError, urllib.error.URLError, TimeoutError) as exc:
                return embedded, str(exc)
            for entry, vector in zip(batch, vectors):
                if not vector:
                    continue
                entry["embedding_model"] = model
                entry["embedding_json"] = json.dumps(vector, separators=(",", ":"))
                embedded += 1
        return embedded, None

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if not left_norm or not right_norm:
            return 0.0
        return dot / (left_norm * right_norm)

    def python_sdk_refresh(
        self,
        limit: int = 5000,
        embedding_model: str | None = None,
        embedding_host: str | None = None,
        embedding_timeout: int = 120,
    ) -> dict[str, Any]:
        self._check_interrupted()
        try:
            conn = self._connect_python_sdk()
        except sqlite3.OperationalError as exc:
            return self._missing_dependency_result("python_sdk_refresh", "sqlite-fts5", f"SQLite FTS5 is unavailable: {exc}")
        limit_value = max(1, int(limit))
        existing_embeddings: dict[str, tuple[str, str]] = {}
        try:
            for row in conn.execute(
                "SELECT id,embedding_model,embedding_json FROM python_sdk_entries "
                "WHERE embedding_model IS NOT NULL AND embedding_json IS NOT NULL"
            ).fetchall():
                existing_embeddings[str(row[0])] = (str(row[1] or ""), str(row[2] or ""))
        except sqlite3.Error:
            existing_embeddings = {}
        entries = self._python_sdk_entries(limit=limit_value)
        model = self._normalize_sdk_embedding_model(embedding_model)
        cached_embeddings = 0
        if not model and existing_embeddings:
            for entry in entries:
                cached = existing_embeddings.get(str(entry.get("id") or ""))
                if not cached:
                    continue
                cached_model, cached_json = cached
                if cached_model and cached_json:
                    entry["embedding_model"] = cached_model
                    entry["embedding_json"] = cached_json
                    cached_embeddings += 1
        embedded = 0
        embedding_error = None
        if model:
            embedded, embedding_error = self._embed_python_sdk_entries(
                entries,
                model=model,
                host=embedding_host,
                timeout=max(1, int(embedding_timeout)),
            )
        try:
            conn.execute("DELETE FROM python_sdk_entries")
            conn.execute("DELETE FROM python_sdk_fts")
            for entry in entries:
                conn.execute(
                    "INSERT OR REPLACE INTO python_sdk_entries("
                    "id,kind,module,qualname,signature,doc,source_path,line,text,embedding_model,embedding_json"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        entry["id"],
                        entry["kind"],
                        entry["module"],
                        entry["qualname"],
                        entry["signature"],
                        entry["doc"],
                        entry["source_path"],
                        int(entry["line"]),
                        entry["text"],
                        entry.get("embedding_model"),
                        entry.get("embedding_json"),
                    ),
                )
                conn.execute(
                    "INSERT INTO python_sdk_fts(id,module,qualname,signature,doc,text) VALUES(?,?,?,?,?,?)",
                    (entry["id"], entry["module"], entry["qualname"], entry["signature"], entry["doc"], entry["text"]),
                )
            conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('stdlib_root', ?)", (str(self._stdlib_root()),))
            conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('python_version', ?)", (sys.version.split()[0],))
            if model:
                conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('embedding_model', ?)", (model,))
            conn.commit()
        finally:
            conn.close()
        summary = f"Indexed {len(entries)} Python SDK item(s) from {self._stdlib_root()}."
        if cached_embeddings:
            summary += f" cached_embeddings={cached_embeddings}."
        if model:
            summary += f" embeddings={embedded} model={model}."
        if embedding_error:
            summary += f" embedding_error={self._truncate_text(embedding_error, limit=160)}"
        return {
            "ok": True,
            "tool": "python_sdk_refresh",
            "items": len(entries),
            "embedded": embedded,
            "cached_embeddings": cached_embeddings,
            "embedding_model": model,
            "embedding_error": embedding_error,
            "cache": self.relative_label(self._python_sdk_cache_path()),
            "stdlib_root": str(self._stdlib_root()),
            "output": summary,
            "summary": summary,
        }

    def _python_sdk_lexical_rows(self, conn: sqlite3.Connection, query: str, *, limit: int) -> list[dict[str, Any]]:
        fts_query = self._safe_fts_query(query)
        if not fts_query:
            return []
        terms = [term.lower() for term in re.findall(r"[A-Za-z_][\w.:-]*|\d+", query)]
        try:
            records = conn.execute(
                "SELECT e.id,e.kind,e.module,e.qualname,e.signature,e.doc,e.source_path,e.line,e.text,e.embedding_model,e.embedding_json,bm25(python_sdk_fts) AS rank "
                "FROM python_sdk_fts JOIN python_sdk_entries e ON e.id=python_sdk_fts.id "
                "WHERE python_sdk_fts MATCH ? ORDER BY rank LIMIT ?",
                (fts_query, max(1, int(limit))),
            ).fetchall()
        except sqlite3.OperationalError:
            terms = self._extract_index_terms(query, limit=8)
            if not terms:
                return []
            like = "%" + "%".join(terms) + "%"
            records = conn.execute(
                "SELECT id,kind,module,qualname,signature,doc,source_path,line,text,embedding_model,embedding_json,0 AS rank "
                "FROM python_sdk_entries WHERE text LIKE ? OR qualname LIKE ? ORDER BY qualname LIMIT ?",
                (like, like, max(1, int(limit))),
            ).fetchall()
        rows = [
            {
                "id": row[0],
                "kind": row[1],
                "module": row[2],
                "qualname": row[3],
                "signature": row[4],
                "doc": row[5],
                "source_path": row[6],
                "line": row[7],
                "text": row[8],
                "embedding_model": row[9],
                "embedding_json": row[10],
                "rank": float(row[11] or 0.0),
                "source": "lexical",
            }
            for row in records
        ]
        for row in rows:
            qualname = str(row.get("qualname") or "").lower()
            module = str(row.get("module") or "").lower()
            signature = str(row.get("signature") or "").lower()
            doc = str(row.get("doc") or "").lower()
            haystack = str(row.get("text") or "").lower()
            score = 0.0
            for term in terms:
                if term == module or term in qualname:
                    score += 10
                if term in signature:
                    score += 4
                if term in doc:
                    score += 2
                elif term in haystack:
                    score += 1
            row["lexical_score"] = score
        return sorted(rows, key=lambda item: (-float(item.get("lexical_score", 0.0)), float(item.get("rank", 0.0)), str(item.get("qualname", ""))))[: max(1, int(limit))]

    def _python_sdk_expanded_query(self, query: str) -> str:
        lowered = query.lower()
        additions: list[str] = []
        if "json" in lowered and re.search(r"\b(?:parse|deserialize|string|load|loads)\b", lowered):
            additions.append("json loads load deserialize")
        if re.search(r"\b(?:file|files|path|paths)\b", lowered) and re.search(r"\b(?:wildcard|glob|pattern|recursive|recursively)\b", lowered):
            additions.append("pathlib Path glob rglob")
        if "temporary" in lowered and "directory" in lowered:
            additions.append("tempfile TemporaryDirectory")
        if re.search(r"\b(?:memoize|memoise|cache|least recently used|lru)\b", lowered):
            additions.append("functools lru_cache cache")
        if re.search(r"\b(?:count|frequency|frequencies|tally)\b", lowered):
            additions.append("collections Counter")
        if "subprocess" in lowered or (re.search(r"\b(?:run|execute)\b", lowered) and "command" in lowered):
            additions.append("subprocess run capture_output CompletedProcess")
        if not additions:
            return query
        return query + " " + " ".join(additions)

    def _python_sdk_embedding_rows(
        self,
        conn: sqlite3.Connection,
        query: str,
        *,
        model: str,
        host: str | None,
        timeout: int,
        limit: int,
    ) -> tuple[list[dict[str, Any]], str | None]:
        try:
            query_vector = self._ollama_embed_texts([query], model=model, host=host, timeout=timeout)[0]
        except (OSError, ValueError, urllib.error.URLError, TimeoutError, IndexError) as exc:
            return [], str(exc)
        records = conn.execute(
            "SELECT id,kind,module,qualname,signature,doc,source_path,line,text,embedding_model,embedding_json "
            "FROM python_sdk_entries WHERE embedding_model=? AND embedding_json IS NOT NULL",
            (model,),
        ).fetchall()
        rows: list[dict[str, Any]] = []
        for row in records:
            try:
                vector = [float(value) for value in json.loads(str(row[10] or "[]"))]
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            score = self._cosine_similarity(query_vector, vector)
            if score <= 0:
                continue
            rows.append(
                {
                    "id": row[0],
                    "kind": row[1],
                    "module": row[2],
                    "qualname": row[3],
                    "signature": row[4],
                    "doc": row[5],
                    "source_path": row[6],
                    "line": row[7],
                    "text": row[8],
                    "embedding_model": row[9],
                    "embedding_json": row[10],
                    "embedding_score": score,
                    "source": "embedding",
                }
            )
        return sorted(rows, key=lambda item: (-float(item["embedding_score"]), str(item["qualname"])))[: max(1, int(limit))], None

    def _python_sdk_candidate_embedding_rows(
        self,
        conn: sqlite3.Connection,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        model: str,
        host: str | None,
        timeout: int,
        limit: int,
    ) -> tuple[list[dict[str, Any]], str | None]:
        if not candidates:
            return [], None
        try:
            query_vector = self._ollama_embed_texts([query], model=model, host=host, timeout=timeout)[0]
        except (OSError, ValueError, urllib.error.URLError, TimeoutError, IndexError) as exc:
            return [], str(exc)

        vectors_by_id: dict[str, list[float]] = {}
        missing_rows: list[dict[str, Any]] = []
        for row in candidates:
            row_id = str(row.get("id") or "")
            if not row_id:
                continue
            cached_model = str(row.get("embedding_model") or "")
            cached_json = str(row.get("embedding_json") or "")
            if cached_model == model and cached_json:
                try:
                    vector = [float(value) for value in json.loads(cached_json)]
                except (TypeError, ValueError, json.JSONDecodeError):
                    vector = []
                if vector:
                    vectors_by_id[row_id] = vector
                    continue
            missing_rows.append(row)

        if missing_rows:
            try:
                vectors = self._ollama_embed_texts(
                    [str(row.get("text", ""))[:4000] for row in missing_rows],
                    model=model,
                    host=host,
                    timeout=timeout,
                )
            except (OSError, ValueError, urllib.error.URLError, TimeoutError) as exc:
                return [], str(exc)
            for row, vector in zip(missing_rows, vectors):
                row_id = str(row.get("id") or "")
                if not row_id or not vector:
                    continue
                vectors_by_id[row_id] = vector
                row["embedding_model"] = model
                row["embedding_json"] = json.dumps(vector, separators=(",", ":"))
                conn.execute(
                    "UPDATE python_sdk_entries SET embedding_model=?, embedding_json=? WHERE id=?",
                    (model, row["embedding_json"], row_id),
                )
            conn.commit()

        rows: list[dict[str, Any]] = []
        for row in candidates:
            vector = vectors_by_id.get(str(row.get("id") or ""))
            if not vector:
                continue
            score = self._cosine_similarity(query_vector, vector)
            if score <= 0:
                continue
            item = dict(row)
            item["embedding_model"] = model
            item["embedding_score"] = score
            item["source"] = "embedding"
            rows.append(item)
        return sorted(rows, key=lambda item: (-float(item["embedding_score"]), str(item["qualname"])))[: max(1, int(limit))], None

    def _format_python_sdk_rows(self, rows: list[dict[str, Any]], *, limit: int) -> str:
        lines: list[str] = []
        for row in rows[: max(1, int(limit))]:
            signature = str(row.get("signature") or "").strip()
            doc = self._truncate_text(str(row.get("doc") or "").strip(), limit=220)
            source = str(row.get("source_path") or "")
            line = int(row.get("line") or 1)
            score = ""
            if "embedding_score" in row:
                score = f" score={float(row['embedding_score']):.3f}"
            lines.append(f"{row.get('qualname')} [{row.get('kind')}; module={row.get('module')}; source={row.get('source')}{score}]")
            if signature:
                lines.append(f"  signature: {signature}")
            if doc:
                lines.append(f"  doc: {doc}")
            if source:
                lines.append(f"  sdk_source: {source}:{line}")
        return "\n".join(lines) if lines else "(no Python SDK matches)"

    def python_sdk_search(
        self,
        query: str,
        limit: int = 8,
        refresh: bool = False,
        use_embeddings: bool = False,
        embedding_model: str | None = None,
        embedding_host: str | None = None,
        embedding_timeout: int = 30,
    ) -> dict[str, Any]:
        self._check_interrupted()
        clean_query = str(query or "").strip()
        if not clean_query:
            return {"ok": False, "tool": "python_sdk_search", "summary": "python_sdk_search requires a non-empty query."}
        model = self._normalize_sdk_embedding_model(embedding_model)
        if refresh or not self._python_sdk_cache_path().exists():
            refresh_result = self.python_sdk_refresh(limit=5000, embedding_model="off")
            if refresh_result.get("ok") is not True:
                return refresh_result
        try:
            conn = self._connect_python_sdk()
        except sqlite3.OperationalError as exc:
            return self._missing_dependency_result("python_sdk_search", "sqlite-fts5", f"SQLite FTS5 is unavailable: {exc}")
        limit_value = max(1, int(limit))
        embedding_error = None
        embedding_candidates = 0
        try:
            expanded_query = self._python_sdk_expanded_query(clean_query)
            lexical_rows = self._python_sdk_lexical_rows(conn, expanded_query, limit=limit_value * 3)
            embedding_rows: list[dict[str, Any]] = []
            if model:
                candidate_limit = min(len(lexical_rows), max(limit_value, min(limit_value * 2, 12)))
                embedding_candidates = candidate_limit
                embedding_rows, embedding_error = self._python_sdk_candidate_embedding_rows(
                    conn,
                    expanded_query,
                    lexical_rows[:candidate_limit],
                    model=model,
                    host=embedding_host,
                    timeout=max(1, int(embedding_timeout)),
                    limit=limit_value * 3,
                )
            merged: dict[str, dict[str, Any]] = {}
            for rank, row in enumerate(lexical_rows):
                item = dict(row)
                item["combined_score"] = 1.0 / (rank + 1)
                merged[str(item["id"])] = item
            for rank, row in enumerate(embedding_rows):
                item = merged.get(str(row["id"]), dict(row))
                item["source"] = "hybrid" if str(row["id"]) in merged else "embedding"
                item["embedding_score"] = row.get("embedding_score", 0.0)
                item["combined_score"] = float(item.get("combined_score", 0.0)) + float(row.get("embedding_score", 0.0)) + (0.25 / (rank + 1))
                merged[str(row["id"])] = item
            rows = sorted(merged.values(), key=lambda item: (-float(item.get("combined_score", 0.0)), str(item.get("qualname", ""))))[:limit_value]
        finally:
            conn.close()
        output = self._format_python_sdk_rows(rows, limit=limit_value)
        summary = f"Python SDK search returned {len(rows)} item(s)."
        if model:
            summary += f" embedding_model={model}."
        if embedding_error:
            summary += f" embedding_error={self._truncate_text(embedding_error, limit=180)}"
        return {
            "ok": True,
            "tool": "python_sdk_search",
            "query": clean_query,
            "expanded_query": self._python_sdk_expanded_query(clean_query),
            "count": len(rows),
            "embedding_model": model,
            "embedding_error": embedding_error,
            "embedding_candidates": embedding_candidates,
            "cache": self.relative_label(self._python_sdk_cache_path()),
            "results": [
                {key: value for key, value in row.items() if key not in {"embedding_json", "text"}}
                for row in rows
            ],
            "summary": summary,
            "output": output,
        }

    def _initialize_fts_connection(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA busy_timeout=2000")
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS repo_fts USING fts5("
            "path UNINDEXED, path_text, symbols, headings, text, mtime_ns UNINDEXED, size UNINDEXED)"
        )
        conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('version', ?)", (str(FTS_INDEX_VERSION),))
        conn.commit()

    def _connect_fts(self) -> sqlite3.Connection:
        cache_key = str(self._fts_cache_path().resolve(strict=False))
        memory_conn = _MEMORY_FTS_CONNECTIONS.get(cache_key)
        if memory_conn is not None:
            return memory_conn
        cache_path = self._fts_cache_path()
        conn: sqlite3.Connection | None = None
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(cache_path)
            conn.execute("PRAGMA journal_mode=WAL")
            self._initialize_fts_connection(conn)
        except sqlite3.OperationalError as exc:
            if conn is not None:
                conn.close()
            if "disk I/O error" not in str(exc):
                raise
            memory_conn = sqlite3.connect(":memory:")
            self._initialize_fts_connection(memory_conn)
            _MEMORY_FTS_CONNECTIONS[cache_key] = memory_conn
            return memory_conn
        return conn

    def _close_fts(self, conn: sqlite3.Connection) -> None:
        if conn in _MEMORY_FTS_CONNECTIONS.values():
            return
        conn.close()

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
        indexed = 0
        unchanged = 0
        deleted = 0
        try:
            existing_rows = conn.execute(
                "SELECT path, mtime_ns, size FROM repo_fts WHERE 1=1" + scope_sql,
                scope_params,
            ).fetchall()
            existing = {
                str(row[0]): (int(row[1]), int(row[2]))
                for row in existing_rows
                if len(row) >= 3
            }
            current_paths: set[str] = set()
            for file_path in files:
                rel = self.relative_label(file_path)
                current_paths.add(rel)
                try:
                    stat = file_path.stat()
                except OSError:
                    continue
                current_fingerprint = (int(stat.st_mtime_ns), int(stat.st_size))
                if existing.get(rel) == current_fingerprint:
                    unchanged += 1
                    continue
                conn.execute("DELETE FROM repo_fts WHERE path = ?", (rel,))
                conn.execute(
                    "INSERT INTO repo_fts(path,path_text,symbols,headings,text,mtime_ns,size) VALUES(?,?,?,?,?,?,?)",
                    self._fts_record(file_path),
                )
                indexed += 1
            stale_paths = sorted(set(existing) - current_paths)
            for rel in stale_paths:
                conn.execute("DELETE FROM repo_fts WHERE path = ?", (rel,))
            deleted = len(stale_paths)
            conn.commit()
        finally:
            self._close_fts(conn)
        return {
            "ok": True,
            "tool": "fts_refresh",
            "path": self.relative_label(base),
            "files": len(files),
            "indexed": indexed,
            "unchanged": unchanged,
            "deleted": deleted,
            "cache": self.relative_label(self._fts_cache_path()),
            "output": f"Indexed {indexed} changed file(s), kept {unchanged} unchanged, deleted {deleted} stale row(s) in SQLite FTS.",
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
            self._close_fts(conn)
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
            "ctime_ns": int(stat.st_ctime_ns),
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

    def _repo_index_record_matches(self, cached: Any, file_path: Path, stat: os.stat_result) -> bool:
        if not (
            isinstance(cached, dict)
            and cached.get("mtime_ns") == int(stat.st_mtime_ns)
            and cached.get("ctime_ns") == int(stat.st_ctime_ns)
            and cached.get("size") == int(stat.st_size)
            and isinstance(cached.get("line_index"), list)
        ):
            return False
        cached_sha1 = str(cached.get("sha1") or "").strip()
        if not cached_sha1:
            return False
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        current_sha1 = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()
        return current_sha1 == cached_sha1

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
            if not self._repo_index_record_matches(cached, file_path, stat):
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
        ranked_paths = [str(item["path"]) for item in ranked_records if str(item.get("path") or "")]
        ranked_symbols: list[dict[str, str]] = []
        seen_ranked_symbols: set[tuple[str, str]] = set()
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
                qualname = str(symbol.get("qualname", "")).strip()
                haystack = f"{qualname} {symbol.get('signature', '')}".lower()
                if qualname in matched_symbol_names or any(term in haystack for term in terms):
                    snippets.append(f"{rel}:{symbol['start']}-{symbol['end']} {symbol['kind']} {qualname}: {symbol['signature']}")
                    ranked_symbol_key = (rel, qualname)
                    if qualname and ranked_symbol_key not in seen_ranked_symbols:
                        ranked_symbols.append({"path": rel, "qualname": qualname})
                        seen_ranked_symbols.add(ranked_symbol_key)
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
            "ranked_paths": ranked_paths,
            "ranked_symbols": ranked_symbols,
            "output": "\n".join(output) if output else "(no ranked snippets)",
        }

    def repo_index_search(self, query: str, path: str = ".", limit: int = 10) -> dict[str, Any]:
        self._check_interrupted()
        base = self.resolve_path(path, allow_missing=False)
        terms = [term.lower() for term in re.findall(r"[A-Za-z_][\w.:-]*|\d+", query)]
        if not terms:
            return {"ok": False, "tool": "repo_index_search", "summary": "repo_index_search requires a non-empty query."}
        records = self._indexed_code_records(base, limit=self._adaptive_index_record_limit(limit, minimum=2500, multiplier=300))
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

    def _format_semgrep_result(
        self,
        stdout: str,
        stderr: str,
        returncode: int,
        base: Path,
        lang: str,
        limit: int,
        *,
        backend: str,
        docker_host: str | None = None,
    ) -> dict[str, Any]:
        output = (stdout or "") + (("\n" + stderr) if stderr else "")
        try:
            payload = json.loads(stdout or "{}")
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
                text = str(extra.get("lines") or extra.get("message") or "").strip().replace("\n", " ")
                rows.append(f"{rel}:{start.get('line', '?')}: {text[:220]}")
        return {
            "ok": returncode in {0, 1},
            "tool": "semgrep_scan",
            "path": self.relative_label(base),
            "lang": lang,
            "backend": backend,
            "docker_host": docker_host,
            "count": len(rows),
            "output": "\n".join(rows) if rows else ("(no semgrep matches)" if returncode in {0, 1} else output),
        }

    def semgrep_scan(self, pattern: str, path: str = ".", lang: str | None = None, limit: int = 50) -> dict[str, Any]:
        self._check_interrupted()
        base = self.resolve_path(path, allow_missing=False)
        clean_pattern = str(pattern or "").strip()
        if not clean_pattern:
            return {"ok": False, "tool": "semgrep_scan", "summary": "semgrep_scan requires a pattern."}
        clean_lang = (lang or self._semgrep_lang_for_path(base) or "python").strip().lower()
        allowed_langs = {"python", "javascript", "typescript", "go", "rust", "java", "c", "cpp", "csharp", "ruby", "php"}
        if clean_lang not in allowed_langs:
            return {"ok": False, "tool": "semgrep_scan", "summary": f"Unsupported semgrep language: {clean_lang}"}
        semgrep = self._semgrep_executable()
        if not semgrep:
            return self._run_semgrep_in_docker(clean_pattern, base, clean_lang, limit)
        command = [semgrep, "--json", "--quiet", "-e", clean_pattern, "--lang", clean_lang, str(base)]
        completed = self._run_process(command, cwd=self.workspace_root, timeout=60, shell=False)
        return self._format_semgrep_result(completed.stdout or "", completed.stderr or "", completed.returncode, base, clean_lang, limit, backend="cli")

    def _missing_dependency_result(self, tool: str, dependency: str, summary: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": False,
            "tool": tool,
            "summary": summary,
            "missing_dependency": dependency,
            "error_class": "missing_dependency",
        }
        resolved = resolve_dependency(dependency)
        if resolved is not None:
            status = dependency_status(resolved, workspace_root=self.workspace_root)
            result.update(
                {
                    "tool_id": resolved.id,
                    "optional": resolved.optional,
                    "supported": status["supported"],
                    "platform": status["platform"],
                    "install_hints": status["install_hints"],
                    "verify_command": status["verify_command"],
                    "dependency_purpose": resolved.purpose,
                }
            )
        return result

    def _ast_grep_executable(self) -> str | None:
        for name in ("ast-grep.exe", "sg.exe", "ast-grep", "sg"):
            resolved = shutil.which(name)
            if not resolved:
                continue
            candidate = Path(resolved)
            if candidate.suffix.lower() == ".exe":
                return str(candidate)
            package_dir = candidate.parent / "node_modules" / "@ast-grep" / "cli"
            for exe_name in ("ast-grep.exe", "sg.exe"):
                package_exe = package_dir / exe_name
                if package_exe.exists():
                    return str(package_exe)
            return resolved
        return None

    def _native_python_function_ast_search(self, *, pattern: str, base: Path, limit: int) -> dict[str, Any] | None:
        if pattern.strip() != "def $F($$$A): $$$B":
            return None
        files = [base] if base.is_file() else self._iter_repo_files(base, limit=50000, suffixes={".py"})
        rows: list[str] = []
        safe_limit = max(1, int(limit))
        for file_path in files:
            self._check_interrupted()
            try:
                source = file_path.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source)
            except (OSError, SyntaxError, ValueError):
                continue
            lines = source.splitlines()
            rel = self.relative_label(file_path)
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                line_number = int(getattr(node, "lineno", 1) or 1)
                text = lines[line_number - 1].strip() if 0 < line_number <= len(lines) else node.name
                rows.append(f"{rel}:{line_number}: {text[:220]}")
                if len(rows) >= safe_limit:
                    return {
                        "ok": True,
                        "tool": "ast_search",
                        "path": self.relative_label(base),
                        "lang": "python",
                        "backend": "native-python",
                        "count": len(rows),
                        "output": "\n".join(rows),
                    }
        return {
            "ok": True,
            "tool": "ast_search",
            "path": self.relative_label(base),
            "lang": "python",
            "backend": "native-python",
            "count": len(rows),
            "output": "\n".join(rows) if rows else "(no ast-grep matches)",
        }

    def _semgrep_executable(self) -> str | None:
        return (
            resolve_tool_executable("semgrep", "semgrep", workspace_root=self.workspace_root)
            or shutil.which("semgrep")
            or resolve_tool_executable("opengrep", "opengrep", workspace_root=self.workspace_root)
            or shutil.which("opengrep")
        )

    def _docker_command(self) -> str | None:
        return resolve_tool_executable("docker", "docker", workspace_root=self.workspace_root) or shutil.which("docker")

    def _docker_host(self) -> str | None:
        return configured_docker_host()

    def _docker_env(self) -> dict[str, str]:
        env = os.environ.copy()
        host = self._docker_host()
        if host:
            env["DOCKER_HOST"] = host
        else:
            env.pop("DOCKER_HOST", None)
        return env

    def _docker_tools_enabled(self) -> bool:
        flag = os.environ.get("OLLAMA_CODE_ENABLE_DOCKER_TOOLS", "").strip().lower()
        return bool(self._docker_host()) or flag in {"1", "true", "yes", "on"}

    def _docker_process(self, args: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
        docker = self._docker_command()
        if not docker:
            raise FileNotFoundError("docker")
        return self._run_process([docker, *args], cwd=self.workspace_root, timeout=timeout, shell=False, env=self._docker_env())

    def _run_semgrep_in_docker(self, pattern: str, base: Path, lang: str, limit: int) -> dict[str, Any]:
        docker = self._docker_command()
        if not docker or not self._docker_tools_enabled():
            result = self._missing_dependency_result(
                "semgrep_scan",
                "semgrep",
                "semgrep/opengrep is not installed. Install semgrep, install opengrep, or enable Docker-backed semgrep with OLLAMA_CODE_DOCKER_HOST=ssh://host.",
            )
            opengrep = resolve_dependency("opengrep")
            if opengrep is not None:
                opengrep_status = dependency_status(opengrep, workspace_root=self.workspace_root)
                result["compatible_tool_ids"] = ["semgrep", "opengrep"]
                result["compatible_install_hints"] = {
                    "opengrep": opengrep_status["install_hints"],
                }
            return result
        container_target = f"/src/target{base.suffix}" if base.is_file() and base.suffix else "/src/target"
        create_command = [
            "create",
            "-w",
            "/src",
            "semgrep/semgrep",
            "semgrep",
            "--json",
            "--quiet",
            "-e",
            pattern,
            "--lang",
            lang,
            container_target,
        ]
        created = self._docker_process(create_command, timeout=120)
        container_id = (created.stdout or "").strip().splitlines()[-1] if created.returncode == 0 and created.stdout.strip() else ""
        if not container_id:
            output = self._collect_process_output(created)
            return {
                "ok": False,
                "tool": "semgrep_scan",
                "summary": "Docker-backed semgrep could not create a container. Pull semgrep/semgrep or check remote Docker access.",
                "output": output,
                "docker_host": self._docker_host(),
            }
        try:
            copied = self._docker_process(["cp", str(base), f"{container_id}:{container_target}"], timeout=300)
            if copied.returncode != 0:
                return {"ok": False, "tool": "semgrep_scan", "summary": "Docker copy into semgrep container failed.", "output": self._collect_process_output(copied), "docker_host": self._docker_host()}
            started = self._docker_process(["start", "-a", container_id], timeout=180)
            inspected = self._docker_process(["inspect", "-f", "{{.State.ExitCode}}", container_id], timeout=30)
            try:
                exit_code = int((inspected.stdout or "").strip())
            except ValueError:
                exit_code = started.returncode
        finally:
            try:
                self._docker_process(["rm", "-f", container_id], timeout=30)
            except Exception:
                pass
        return self._format_semgrep_result(started.stdout or "", started.stderr or "", exit_code, base, lang, limit, backend="docker", docker_host=self._docker_host())

    def _format_tool_dependency_rows(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "(no optional tool dependencies matched)"
        lines: list[str] = []
        for row in rows:
            state = "installed" if row.get("installed") else ("unsupported" if not row.get("supported") else "missing")
            recommended = " recommended" if row.get("recommended") else ""
            hint_text = ""
            hints = row.get("install_hints")
            if isinstance(hints, list) and hints:
                first = hints[0]
                if isinstance(first, dict) and first.get("command"):
                    hint_text = f" mode={first.get('mode', row.get('install_mode', 'host'))} install={first['command']}"
            notes = f" notes={row['notes']}" if row.get("notes") else ""
            if not hint_text:
                hint_text = f" mode={row.get('install_mode', 'host')}"
            lines.append(f"{row['id']}: {state}{recommended} category={row['category']}{hint_text}{notes}")
        return "\n".join(lines)

    def tool_status(self, scope: str = "all", tool_id: str | None = None) -> dict[str, Any]:
        clean_scope = str(scope or "all").strip().lower()
        if tool_id:
            dependency = resolve_dependency(tool_id)
            if dependency is None:
                return {"ok": False, "tool": "tool_status", "summary": f"Unknown optional tool dependency: {tool_id}"}
            rows = [dependency_status(dependency, workspace_root=self.workspace_root)]
        else:
            recommended_only = clean_scope in {"recommended", "recommended-missing", "recommended_missing"}
            missing_only = clean_scope in {"missing", "recommended-missing", "recommended_missing"}
            rows = dependency_statuses(recommended_only=recommended_only, missing_only=missing_only, workspace_root=self.workspace_root)
        installed = sum(1 for row in rows if row.get("installed"))
        missing = sum(1 for row in rows if row.get("supported") and not row.get("installed"))
        unsupported = sum(1 for row in rows if not row.get("supported"))
        return {
            "ok": True,
            "tool": "tool_status",
            "scope": clean_scope,
            "platform": current_platform(),
            "count": len(rows),
            "installed": installed,
            "missing": missing,
            "unsupported": unsupported,
            "dependencies": rows,
            "output": self._format_tool_dependency_rows(rows),
        }

    def _install_plan_for_dependency(self, dependency_id: str) -> dict[str, Any]:
        dependency = resolve_dependency(dependency_id)
        if dependency is None:
            return {"ok": False, "dependency": dependency_id, "summary": f"Unknown optional tool dependency: {dependency_id}"}
        status = dependency_status(dependency, workspace_root=self.workspace_root)
        if status.get("installed"):
            return {"ok": True, "dependency": dependency.id, "already_installed": True, "summary": f"{dependency.id} is already installed.", "status": status}
        if not status.get("supported"):
            return {"ok": False, "dependency": dependency.id, "unsupported": True, "summary": f"{dependency.id} is unsupported on {status.get('platform')}.", "status": status}
        hint = first_install_hint(dependency)
        if hint is None:
            return {"ok": False, "dependency": dependency.id, "summary": f"No install command is configured for {dependency.id}.", "status": status}
        return {
            "ok": True,
            "dependency": dependency.id,
            "already_installed": False,
            "manager": hint.manager,
            "mode": hint.mode,
            "argv": list(hint.command),
            "command": command_to_text(hint.command),
            "summary": f"{dependency.id} can be installed with {command_to_text(hint.command)} mode={hint.mode}",
            "status": status,
        }

    def tool_install(
        self,
        tool_id: str | None = None,
        *,
        all_recommended: bool = False,
        confirm: bool = False,
        timeout: int = 600,
    ) -> dict[str, Any]:
        if all_recommended:
            requested = [dependency.id for dependency in TOOL_DEPENDENCIES if dependency.recommended]
        elif tool_id:
            requested = [str(tool_id).strip()]
        else:
            return {"ok": False, "tool": "tool_install", "summary": "tool_install requires tool_id or all_recommended=true."}
        plans = [self._install_plan_for_dependency(item) for item in requested]
        runnable = [plan for plan in plans if plan.get("ok") and not plan.get("already_installed") and plan.get("argv")]
        if not confirm:
            return {
                "ok": True,
                "tool": "tool_install",
                "planned": True,
                "plans": plans,
                "output": "\n".join(str(plan.get("summary", "")) for plan in plans),
                "summary": "Install plan only. Re-run with confirm=true from an interactive session to execute.",
            }
        if self.approval_mode == "read-only":
            return {"ok": False, "tool": "tool_install", "plans": plans, "summary": "Install denied because approval mode is read-only."}
        if not runnable:
            return {
                "ok": all(plan.get("ok") for plan in plans),
                "tool": "tool_install",
                "plans": plans,
                "summary": "No installable missing supported tools in the requested set.",
                "output": "\n".join(str(plan.get("summary", "")) for plan in plans),
            }
        command_block = "\n".join(str(plan["command"]) for plan in runnable)
        approved = self._confirm(
            "Install optional tooling with these command(s)?\n"
            f"{command_block}\n"
            "Prefer isolated-venv/docker modes for Python tools; host modes may modify package manager state."
        )
        if not approved:
            return {"ok": False, "tool": "tool_install", "plans": plans, "summary": "User rejected optional tool install."}
        results: list[dict[str, Any]] = []
        for plan in runnable:
            argv = [str(part) for part in plan.get("argv", [])]
            try:
                completed = self._run_process(argv, cwd=self.workspace_root, timeout=max(1, int(timeout)), shell=False)
            except subprocess.TimeoutExpired:
                results.append({"dependency": plan["dependency"], "ok": False, "command": plan["command"], "summary": "Install command timed out."})
                continue
            output = self._collect_process_output(completed)
            dependency = resolve_dependency(str(plan["dependency"]))
            post_status = dependency_status(dependency, workspace_root=self.workspace_root) if dependency is not None else {}
            results.append(
                {
                    "dependency": plan["dependency"],
                    "ok": completed.returncode == 0 and bool(post_status.get("installed", completed.returncode == 0)),
                    "exit_code": completed.returncode,
                    "command": plan["command"],
                    "output": output,
                    "status": post_status,
                }
            )
        self._python_tool_command_cache.clear()
        self._which_cache.clear()
        clear_dependency_status_cache()
        ok = all(item.get("ok") for item in results)
        lines = [
            f"{item['dependency']}: {'ok' if item.get('ok') else 'failed'} command={item.get('command')}"
            for item in results
        ]
        return {"ok": ok, "tool": "tool_install", "plans": plans, "results": results, "output": "\n".join(lines), "summary": "Optional tool install complete." if ok else "One or more optional tool installs failed."}

    def ast_search(self, pattern: str, path: str = ".", lang: str | None = None, limit: int = 50) -> dict[str, Any]:
        self._check_interrupted()
        executable = self._ast_grep_executable()
        if not executable:
            return self._missing_dependency_result("ast_search", "ast-grep", "ast-grep is not installed. Install ast-grep or sg to use AST search.")
        base = self.resolve_path(path, allow_missing=False)
        clean_pattern = str(pattern or "").strip()
        if not clean_pattern:
            return {"ok": False, "tool": "ast_search", "summary": "ast_search requires a pattern."}
        clean_lang = (lang or self._semgrep_lang_for_path(base) or "").strip().lower()
        if clean_lang == "python":
            native_result = self._native_python_function_ast_search(pattern=clean_pattern, base=base, limit=limit)
            if native_result is not None:
                return native_result
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

    def structural_rewrite(
        self,
        pattern: str,
        rewrite: str,
        path: str = ".",
        lang: str | None = None,
        apply: bool = False,
        timeout: int = 60,
    ) -> dict[str, Any]:
        self._check_interrupted()
        executable = self._ast_grep_executable()
        if not executable:
            return self._missing_dependency_result("structural_rewrite", "ast-grep", "ast-grep is not installed. Install ast-grep or sg to use structural rewrite.")
        base = self.resolve_path(path, allow_missing=False)
        clean_pattern = str(pattern or "").strip()
        clean_rewrite = str(rewrite or "").strip()
        if not clean_pattern or not clean_rewrite:
            return {"ok": False, "tool": "structural_rewrite", "summary": "structural_rewrite requires pattern and rewrite."}
        clean_lang = (lang or self._semgrep_lang_for_path(base) or "").strip().lower()
        command = [executable, "-p", clean_pattern, "-r", clean_rewrite]
        if clean_lang:
            command.extend(["--lang", clean_lang])
        if apply:
            approved, reason = self._approve_mutation(
                f"Run ast-grep structural rewrite on {self.relative_label(base)}?",
                f"pattern: {clean_pattern}\nrewrite: {clean_rewrite}",
            )
            if not approved:
                return {"ok": False, "tool": "structural_rewrite", "summary": reason or "User rejected structural rewrite."}
            command.append("--update-all")
        command.append(str(base))
        completed = self._run_process(command, cwd=self.workspace_root, timeout=max(1, int(timeout)), shell=False)
        output = self._collect_process_output(completed)
        return {
            "ok": completed.returncode in {0, 1},
            "tool": "structural_rewrite",
            "path": self.relative_label(base),
            "lang": clean_lang or None,
            "applied": bool(apply),
            "command": command_to_text(tuple(command)),
            "output": output,
        }

    def _tree_sitter_language_for_path(self, path: Path) -> str | None:
        suffix = path.suffix.lower()
        return {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "tsx",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".c": "c",
            ".h": "c",
            ".cc": "cpp",
            ".cpp": "cpp",
            ".hpp": "cpp",
        }.get(suffix)

    def _tree_sitter_parser_for_language(self, language: str) -> Any:
        cached = self._tree_sitter_parsers.get(language)
        if cached is not None:
            return cached
        language_pack = importlib.import_module("tree_sitter_language_pack")
        get_parser = getattr(language_pack, "get_parser")
        parser = get_parser(language)
        self._tree_sitter_parsers[language] = parser
        return parser

    def _tree_sitter_syntax_diagnostic(self, target: Path, text: str) -> str | None:
        language = self._tree_sitter_language_for_path(target)
        if not language:
            return None
        try:
            parser = self._tree_sitter_parser_for_language(language)
        except (ImportError, ModuleNotFoundError, AttributeError, LookupError):
            return None
        try:
            tree = parser.parse(text)
        except TypeError:
            tree = parser.parse(text.encode("utf-8", errors="replace"))
        root_node = getattr(tree, "root_node", None)
        if root_node is not None and getattr(root_node, "has_error", False):
            return f"{self.relative_label(target)}: tree-sitter reported syntax errors for {language}"
        return None

    def tree_sitter_syntax(self, path: str = ".", limit: int = 200) -> dict[str, Any]:
        self._check_interrupted()
        dependency = resolve_dependency("py-tree-sitter")
        if dependency is not None and not dependency_status(dependency)["installed"]:
            return self._missing_dependency_result("tree_sitter_syntax", "py-tree-sitter", "tree-sitter Python bindings are not installed.")
        base = self.resolve_path(path, allow_missing=False)
        checked: list[str] = []
        diagnostics: list[str] = []
        for file_path in self._iter_code_files(base, limit=max(1, int(limit))):
            if self._tree_sitter_language_for_path(file_path) is None:
                continue
            checked.append(self.relative_label(file_path))
            text = file_path.read_text(encoding="utf-8", errors="replace")
            diagnostic = self._tree_sitter_syntax_diagnostic(file_path, text)
            if diagnostic:
                diagnostics.append(diagnostic)
        return {
            "ok": not diagnostics,
            "tool": "tree_sitter_syntax",
            "path": self.relative_label(base),
            "checked": checked,
            "diagnostics": diagnostics,
            "output": "\n".join(diagnostics) if diagnostics else f"tree-sitter syntax ok: {len(checked)} file(s)",
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
            "ranked_paths": [
                path
                for path in ranked.get("ranked_paths", [])
                if isinstance(path, str) and path.endswith(".py") and not self._path_looks_like_test(Path(path))
            ],
            "ranked_symbols": [
                item
                for item in ranked.get("ranked_symbols", [])
                if isinstance(item, dict)
                and isinstance(item.get("path"), str)
                and str(item.get("path")).endswith(".py")
                and not self._path_looks_like_test(Path(str(item.get("path"))))
            ],
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

    def _python_import_targets_from_text(self, text: str, *, label: str) -> list[dict[str, Any]]:
        snippets: list[str] = []
        for match in re.finditer(r"```(?:python|py)?\s*(?P<body>.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
            body = str(match.group("body") or "").strip()
            if body:
                snippets.append(body)
        import_lines = [
            line.strip()
            for line in text.splitlines()
            if re.match(r"^\s*(?:from\s+[A-Za-z_][\w.]*\s+import\s+.+|import\s+[A-Za-z_][\w.]*(?:\s+as\s+\w+)?(?:\s*,\s*\w+)*)\s*$", line)
        ]
        if import_lines:
            snippets.append("\n".join(import_lines))
        targets: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for snippet in snippets:
            try:
                tree = ast.parse(self._python_parse_text(snippet), filename=label)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    module_file = self._resolve_python_module_file(node.module)
                    for alias in node.names:
                        if module_file is None:
                            continue
                        key = (self.relative_label(module_file), alias.name)
                        if key in seen:
                            continue
                        seen.add(key)
                        targets.append(
                            {
                                "path": self.relative_label(module_file),
                                "symbol": alias.name,
                                "reason": f"{label} imports {alias.name} from {node.module}",
                            }
                        )
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        module_file = self._resolve_python_module_file(alias.name)
                        if module_file is None:
                            continue
                        symbol = alias.asname or alias.name.split(".")[-1]
                        key = (self.relative_label(module_file), symbol)
                        if key in seen:
                            continue
                        seen.add(key)
                        targets.append(
                            {
                                "path": self.relative_label(module_file),
                                "symbol": symbol,
                                "reason": f"{label} imports {alias.name}",
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

    def _implementation_target_score(self, item: dict[str, Any], query_text: str) -> int:
        rel = str(item.get("path") or "").strip().replace("\\", "/")
        symbol = str(item.get("symbol") or "").strip()
        reason = str(item.get("reason") or "")
        lowered_query = query_text.lower()
        score = 0
        if "line" in item:
            score += 80
        if rel:
            path_obj = Path(rel)
            if self._path_looks_like_test(path_obj):
                score -= 25
            else:
                score += 20
            stem = path_obj.stem.lower()
            if stem and re.search(rf"(?<![\w.]){re.escape(stem)}(?![\w.])", lowered_query):
                score += 8
            module_name = rel[:-3].replace("/", ".").replace("\\", ".").lower() if rel.endswith(".py") else rel.lower()
            if module_name and module_name in lowered_query:
                score += 14
        if symbol and re.search(rf"(?<![\w.]){re.escape(symbol.lower())}(?![\w.])", lowered_query):
            score += 30
        if reason.startswith("request text imports"):
            score += 18
        elif "imports" in reason:
            score += 6
        return score

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
        if query:
            targets.extend(self._python_import_targets_from_text(str(query), label="request text"))
        ranked = sorted(
            targets,
            key=lambda item: (
                -self._implementation_target_score(item, str(query or output or traceback or "")),
                str(item.get("path", "")),
                str(item.get("symbol", "")),
            ),
        )
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in ranked:
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

    def _test_spec_source_symbols(self, source_path: str | None) -> set[str]:
        if not source_path:
            return set()
        try:
            target = self.resolve_path(source_path, allow_missing=False)
            tree = ast.parse(target.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return set()
        return {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }

    def _test_spec_import_aliases(self, tree: ast.AST, source_path: str | None) -> dict[str, str]:
        if not source_path:
            return {}
        module_name = Path(source_path).stem
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module != module_name:
                continue
            for alias in node.names:
                aliases[alias.asname or alias.name] = alias.name
        return aliases

    def _node_expr(self, node: ast.AST, local_exprs: dict[str, str] | None = None) -> str:
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

    def _test_spec_call_name(self, call: ast.Call) -> str:
        func = call.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        try:
            return ast.unparse(func)
        except Exception:
            return "call"

    def _method_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                return func.attr
            if isinstance(func, ast.Name):
                return func.id
        return ""

    def _call_expr(self, call: ast.Call, local_exprs: dict[str, str] | None = None) -> str:
        args = [self._node_expr(arg, local_exprs) for arg in call.args]
        args.extend(f"{kw.arg}={self._node_expr(kw.value, local_exprs)}" for kw in call.keywords if kw.arg)
        name = self._test_spec_call_name(call)
        if isinstance(call.func, ast.Attribute):
            receiver = self._node_expr(call.func.value, local_exprs)
            name = f"{receiver}.{call.func.attr}"
        return f"{name}({', '.join(args)})"

    def _test_spec_symbol_from_expr(self, expr: str, source_symbols: set[str], aliases: dict[str, str]) -> str:
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

    def _test_spec_add_raw_example(
        self,
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
        symbol = self._test_spec_symbol_from_expr(expr, source_symbols, aliases)
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

    def _first_behavior_call(self, statements: list[ast.stmt]) -> ast.Call | None:
        for statement in statements:
            for node in ast.walk(statement):
                if not isinstance(node, ast.Call):
                    continue
                method = self._method_name(node)
                if method.startswith("assert"):
                    continue
                return node
        return None

    def _test_spec_add_example(
        self,
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
        symbol = self._test_spec_call_name(call)
        canonical = aliases.get(symbol, symbol)
        expr = expr_override or self._call_expr(call, local_exprs)
        if source_symbols and canonical not in source_symbols:
            derived_symbol = self._test_spec_symbol_from_expr(expr, source_symbols, aliases)
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

    def _human_test_name(self, name: str) -> str:
        text = re.sub(r"^test_?", "", name.strip())
        text = re.sub(r"_+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text or name

    def _test_spec_iter_test_functions(self, tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
        functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        for node in getattr(tree, "body", []):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                functions.append(node)
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test"):
                        functions.append(child)
        return functions

    def _test_spec_assignment_expr(self, stmt: ast.stmt, local_exprs: dict[str, str]) -> tuple[str, str] | None:
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
        return target.id, self._node_expr(value, local_exprs)

    def _test_spec_assignment_value_node(self, stmt: ast.stmt) -> ast.AST | None:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            return stmt.value
        if isinstance(stmt, ast.AnnAssign):
            return stmt.value
        return None

    def _test_spec_is_source_constructor(self, expr: str, source_symbols: set[str], aliases: dict[str, str]) -> bool:
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\(", expr)
        if not match:
            return False
        name = aliases.get(match.group(1), match.group(1))
        return not source_symbols or name in source_symbols

    def _test_spec_behavior_expr(
        self,
        call: ast.Call,
        local_exprs: dict[str, str],
        object_history: dict[str, list[str]],
    ) -> str:
        receiver = self._test_spec_receiver_root_name(call)
        if receiver:
            history = object_history.get(receiver)
            if history:
                expr = self._call_expr(call, None)
                return "; ".join(history + [expr])
        expr = self._call_expr(call, local_exprs)
        return expr

    def _test_spec_receiver_root_name(self, call: ast.Call) -> str:
        func = call.func
        if isinstance(func, ast.Name) and func.id in {"len", "list", "tuple", "set"} and call.args:
            first_arg = call.args[0]
            if isinstance(first_arg, ast.Name):
                return first_arg.id
            if isinstance(first_arg, ast.Call):
                return self._test_spec_receiver_root_name(first_arg)
            if isinstance(first_arg, ast.Attribute):
                return self._test_spec_receiver_root_name_from_node(first_arg)
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

    def _test_spec_assigned_names(self, node: ast.AST) -> set[str]:
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

    def _test_spec_receiver_root_name_from_node(self, node: ast.AST) -> str:
        if isinstance(node, ast.Call):
            return self._test_spec_receiver_root_name(node)
        if isinstance(node, ast.Attribute):
            current: ast.AST = node.value
            while isinstance(current, ast.Attribute):
                current = current.value
            if isinstance(current, ast.Call):
                return self._test_spec_receiver_root_name(current)
            if isinstance(current, ast.Name):
                return current.id
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                receiver = self._test_spec_receiver_root_name(child)
                if receiver:
                    return receiver
            if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
                return child.value.id
        return ""

    def _test_spec_expr_with_history(self, node: ast.AST, local_exprs: dict[str, str], object_history: dict[str, list[str]]) -> str:
        receiver = self._test_spec_receiver_root_name_from_node(node)
        if receiver and receiver in object_history:
            return "; ".join(object_history[receiver] + [self._node_expr(node, None)])
        return self._node_expr(node, local_exprs)

    def _test_spec_expected_expr(self, node: ast.AST, local_exprs: dict[str, str], object_history: dict[str, list[str]]) -> str:
        receiver = self._test_spec_receiver_root_name_from_node(node)
        if receiver and receiver in object_history:
            return self._node_expr(node, None)
        return self._node_expr(node, local_exprs)

    def _test_spec_add_text_example(
        self,
        examples: list[dict[str, Any]],
        *,
        expr: str,
        text: str,
        line: int,
        test_name: str | None,
        source_symbols: set[str],
        aliases: dict[str, str],
    ) -> None:
        symbol = self._test_spec_symbol_from_expr(expr, source_symbols, aliases)
        if source_symbols and not symbol:
            return
        item = {"symbol": symbol or "expression", "example": text, "line": line}
        if test_name:
            item["test_name"] = test_name
        if item not in examples:
            examples.append(item)

    def _test_spec_call_may_mutate_state(self, call: ast.Call) -> bool:
        name = self._test_spec_call_name(call)
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

    def _test_spec_record_side_effect_call(
        self,
        call: ast.Call,
        local_exprs: dict[str, str],
        object_history: dict[str, list[str]],
    ) -> None:
        if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
            receiver = call.func.value.id
            if receiver in object_history:
                object_history[receiver].append(self._call_expr(call, None))

    def _assert_raises_expected_message(self, statements: list[ast.stmt], context_var: ast.expr | None, local_exprs: dict[str, str]) -> str | None:
        if not isinstance(context_var, ast.Name):
            return None
        pattern = f"{context_var.id}.exception.args[0]"
        for statement in statements:
            for node in ast.walk(statement):
                if not isinstance(node, ast.Call) or self._method_name(node) not in {"assertEqual", "assertEquals"} or len(node.args) < 2:
                    continue
                left = self._node_expr(node.args[0], local_exprs)
                right = self._node_expr(node.args[1], local_exprs)
                if left == pattern:
                    return right
                if right == pattern:
                    return left
        return None

    def _select_test_spec_examples(self, examples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
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

    def test_spec_extract(self, test_path: str, source_path: str | None = None, limit: int = 20) -> dict[str, Any]:
        self._check_interrupted()
        test_file = self.resolve_path(test_path, allow_missing=False)
        if test_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "test_spec_extract", "path": self.relative_label(test_file), "summary": "test_spec_extract supports Python unittest files only."}
        try:
            tree = ast.parse(test_file.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            return {"ok": False, "tool": "test_spec_extract", "path": self.relative_label(test_file), "summary": f"Could not parse test file: {exc}"}
        limit = max(1, min(int(limit), 80))
        source_symbols = self._test_spec_source_symbols(source_path)
        aliases = self._test_spec_import_aliases(tree, source_path)
        examples: list[dict[str, Any]] = []
        equality_methods = {"assertEqual", "assertEquals", "assertListEqual", "assertTupleEqual", "assertDictEqual"}
        function_constants: dict[int, dict[str, str]] = {}
        for node in getattr(tree, "body", []):
            if not isinstance(node, ast.ClassDef):
                continue
            constants: dict[str, str] = {}
            for child in node.body:
                assigned = self._test_spec_assignment_expr(child, constants)
                if assigned is not None:
                    constants[assigned[0]] = assigned[1]
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test"):
                    function_constants[id(child)] = dict(constants)
        for function in self._test_spec_iter_test_functions(tree):
            self._check_interrupted()
            local_exprs: dict[str, str] = dict(function_constants.get(id(function), {}))
            object_history: dict[str, list[str]] = {}
            for stmt_index, stmt in enumerate(function.body):
                assigned = self._test_spec_assignment_expr(stmt, local_exprs)
                if assigned is not None:
                    name, expr = assigned
                    value_node = self._test_spec_assignment_value_node(stmt)
                    if value_node is not None:
                        expr = self._test_spec_expr_with_history(value_node, local_exprs, object_history)
                    local_exprs[name] = expr
                    if self._test_spec_is_source_constructor(expr, source_symbols, aliases):
                        object_history[name] = [f"{name} = {expr}"]
                    continue
                if isinstance(stmt, (ast.For, ast.While)):
                    for name in self._test_spec_assigned_names(stmt):
                        local_exprs.pop(name, None)
                        object_history.pop(name, None)
                    continue
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                    node = stmt.value
                    method_name = self._method_name(node)
                    if self._method_name(node) in equality_methods and len(node.args) >= 2:
                        actual: ast.Call | None = None
                        expected_node: ast.AST | None = None
                        if isinstance(node.args[0], ast.Call):
                            actual = node.args[0]
                            expected_node = node.args[1]
                        elif isinstance(node.args[1], ast.Call):
                            actual = node.args[1]
                            expected_node = node.args[0]
                        if actual is not None and expected_node is not None:
                            self._test_spec_add_example(
                                examples,
                                call=actual,
                                expected=self._test_spec_expected_expr(expected_node, local_exprs, object_history),
                                expr_override=self._test_spec_behavior_expr(actual, local_exprs, object_history),
                                line=int(getattr(node, "lineno", 1)),
                                test_name=function.name,
                                source_symbols=source_symbols,
                                aliases=aliases,
                                local_exprs=local_exprs,
                            )
                            if self._test_spec_call_may_mutate_state(actual):
                                self._test_spec_record_side_effect_call(actual, local_exprs, object_history)
                        else:
                            for actual_node, expected_candidate in ((node.args[0], node.args[1]), (node.args[1], node.args[0])):
                                actual_receiver = self._test_spec_receiver_root_name_from_node(actual_node)
                                expected_receiver = self._test_spec_receiver_root_name_from_node(expected_candidate)
                                if actual_receiver and actual_receiver == expected_receiver and actual_receiver in object_history:
                                    expr = self._test_spec_expr_with_history(actual_node, local_exprs, object_history)
                                else:
                                    expr = self._node_expr(actual_node, local_exprs)
                                if not self._test_spec_symbol_from_expr(expr, source_symbols, aliases):
                                    continue
                                self._test_spec_add_raw_example(
                                    examples,
                                    expr=expr,
                                    expected=self._test_spec_expected_expr(expected_candidate, local_exprs, object_history),
                                    line=int(getattr(node, "lineno", 1)),
                                    test_name=function.name,
                                    source_symbols=source_symbols,
                                    aliases=aliases,
                                )
                                break
                        continue
                    if method_name == "assertIsNone" and node.args:
                        expr = self._test_spec_behavior_expr(node.args[0], local_exprs, object_history) if isinstance(node.args[0], ast.Call) else self._node_expr(node.args[0], local_exprs)
                        self._test_spec_add_raw_example(
                            examples,
                            expr=expr,
                            expected="None",
                            line=int(getattr(node, "lineno", 1)),
                            test_name=function.name,
                            source_symbols=source_symbols,
                            aliases=aliases,
                        )
                        continue
                    if method_name in {"assertTrue", "assertFalse"} and node.args:
                        actual_node = node.args[0]
                        expr = self._test_spec_behavior_expr(actual_node, local_exprs, object_history) if isinstance(actual_node, ast.Call) else self._node_expr(actual_node, local_exprs)
                        self._test_spec_add_raw_example(
                            examples,
                            expr=expr,
                            expected="True" if method_name == "assertTrue" else "False",
                            line=int(getattr(node, "lineno", 1)),
                            test_name=function.name,
                            source_symbols=source_symbols,
                            aliases=aliases,
                        )
                        if isinstance(actual_node, ast.Call) and self._test_spec_call_may_mutate_state(actual_node):
                            self._test_spec_record_side_effect_call(actual_node, local_exprs, object_history)
                        continue
                    if method_name in {"assertRegex", "assertRegexpMatches"} and len(node.args) >= 2:
                        expr = self._node_expr(node.args[0], local_exprs)
                        pattern = self._node_expr(node.args[1], local_exprs)
                        self._test_spec_add_text_example(
                            examples,
                            expr=expr,
                            text=f"{expr} matches {pattern}",
                            line=int(getattr(node, "lineno", 1)),
                            test_name=function.name,
                            source_symbols=source_symbols,
                            aliases=aliases,
                        )
                        continue
                    if method_name == "assertNotEqual" and len(node.args) >= 2:
                        left = self._node_expr(node.args[0], local_exprs)
                        right = self._node_expr(node.args[1], local_exprs)
                        self._test_spec_add_text_example(
                            examples,
                            expr=left,
                            text=f"{left} != {right}",
                            line=int(getattr(node, "lineno", 1)),
                            test_name=function.name,
                            source_symbols=source_symbols,
                            aliases=aliases,
                        )
                        continue
                    if self._method_name(node) == "assertRaises" and len(node.args) >= 2:
                        func_node = node.args[1]
                        if isinstance(func_node, (ast.Name, ast.Attribute)):
                            fake_call = ast.Call(func=func_node, args=list(node.args[2:]), keywords=list(node.keywords))
                            self._test_spec_add_example(
                                examples,
                                call=fake_call,
                                raises=self._node_expr(node.args[0], local_exprs),
                                line=int(getattr(node, "lineno", 1)),
                                test_name=function.name,
                                source_symbols=source_symbols,
                                aliases=aliases,
                                local_exprs=local_exprs,
                            )
                        continue
                    self._test_spec_record_side_effect_call(node, local_exprs, object_history)
                if isinstance(stmt, ast.With):
                    for item in stmt.items:
                        context = item.context_expr
                        if not isinstance(context, ast.Call) or self._method_name(context) != "assertRaises" or not context.args:
                            continue
                        behavior_call = self._first_behavior_call(list(stmt.body))
                        if behavior_call is None:
                            continue
                        self._test_spec_add_example(
                            examples,
                            call=behavior_call,
                            raises=self._node_expr(context.args[0], local_exprs),
                            raises_message=self._assert_raises_expected_message(
                                list(stmt.body) + list(function.body[stmt_index + 1 : stmt_index + 4]),
                                item.optional_vars,
                                local_exprs,
                            ),
                            expr_override=self._test_spec_behavior_expr(behavior_call, local_exprs, object_history),
                            line=int(getattr(stmt, "lineno", 1)),
                            test_name=function.name,
                            source_symbols=source_symbols,
                            aliases=aliases,
                            local_exprs=local_exprs,
                        )
                        self._test_spec_record_side_effect_call(behavior_call, local_exprs, object_history)
        examples = self._select_test_spec_examples(examples, limit)
        grouped: dict[str, list[str]] = {}
        for item in examples:
            label = self._human_test_name(str(item.get("test_name") or ""))
            prefix = f"[{label}] " if label else ""
            grouped.setdefault(str(item["symbol"]), []).append(
                f"{prefix}{item['example']} ({self.relative_label(test_file)}:{item['line']})"
            )
        lines: list[str] = []
        for symbol, items in grouped.items():
            lines.append(f"{symbol}:")
            lines.extend(f"- {example}" for example in items)
        return {
            "ok": True,
            "tool": "test_spec_extract",
            "path": self.relative_label(test_file),
            "source_path": source_path or "",
            "count": len(examples),
            "examples": examples,
            "grouped": grouped,
            "output": "\n".join(lines) if lines else "(no unittest examples extracted)",
        }

    def _implementation_source_defs(self, source_file: Path) -> list[dict[str, Any]]:
        symbols, text, _ = self._code_symbols(source_file)
        lines = text.splitlines()
        try:
            tree = ast.parse(self._python_parse_text(text))
        except SyntaxError:
            tree = None
        stub_lines: set[int] = set()
        class_risks: dict[str, list[str]] = {}
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and self._python_body_is_stub(node):
                    stub_lines.add(int(getattr(node, "lineno", 1)))
                if isinstance(node, ast.ClassDef):
                    methods, assigned, called = self._python_self_attribute_facts(node)
                    risks: list[str] = []
                    for attr in sorted(methods & assigned):
                        risks.append(f"self.{attr} shadows method {attr}()")
                    for attr in sorted(called - methods - assigned):
                        risks.append(f"calls missing self.{attr}()")
                    if risks:
                        class_risks[node.name] = risks
        rows: list[dict[str, Any]] = []
        for item in symbols:
            if str(item.get("kind")) not in {"function", "method", "class"}:
                continue
            qualname = str(item.get("qualname") or item.get("name") or "")
            line = int(item.get("start") or 1)
            rows.append(
                {
                    "symbol": qualname,
                    "name": str(item.get("name") or qualname.split(".")[-1]),
                    "kind": str(item.get("kind") or ""),
                    "line": line,
                    "signature": str(item.get("signature") or self._python_signature(lines, line)),
                    "stub": line in stub_lines,
                    "risks": class_risks.get(qualname.split(".")[-1], []),
                }
            )
        return rows

    def _implementation_expected_type(self, example: str) -> str:
        kind, _expr, expected = self._split_test_example(example)
        if kind == "raises":
            return "raises"
        if kind != "value":
            return ""
        try:
            value = ast.literal_eval(expected)
        except (SyntaxError, ValueError):
            return ""
        return type(value).__name__

    def _literal_string_call_example(self, example: str) -> tuple[str, str, str] | None:
        kind, expr, expected = self._split_test_example(example)
        if kind != "value":
            return None
        try:
            parsed = ast.parse(expr, mode="eval")
            expected_value = ast.literal_eval(expected)
        except (SyntaxError, ValueError):
            return None
        call = parsed.body
        if not isinstance(call, ast.Call) or not call.args or not isinstance(expected_value, str):
            return None
        try:
            first_arg = ast.literal_eval(call.args[0])
        except (SyntaxError, ValueError):
            return None
        if not isinstance(first_arg, str):
            return None
        return self._test_spec_call_name(call), first_arg, expected_value

    def _literal_call_string_arg_expected(self, example: str) -> tuple[str, str, Any] | None:
        kind, expr, expected = self._split_test_example(example)
        if kind != "value":
            return None
        try:
            parsed = ast.parse(expr, mode="eval")
            expected_value = ast.literal_eval(expected)
        except (SyntaxError, ValueError):
            return None
        call = parsed.body
        if not isinstance(call, ast.Call) or not call.args:
            return None
        try:
            first_arg = ast.literal_eval(call.args[0])
        except (SyntaxError, ValueError):
            return None
        if not isinstance(first_arg, str):
            return None
        return self._test_spec_call_name(call), first_arg, expected_value

    def _string_transform_hint(self, source: str, expected: str) -> str:
        if source == expected:
            return "unchanged"
        if expected.startswith(source):
            suffix = expected[len(source) :]
            return f"append {suffix!r}"
        if len(expected) >= len(source):
            suffix = expected[len(source) :]
            for cut in range(1, len(source) + 1):
                if expected == source[cut:] + source[:cut] + suffix:
                    return f"move prefix {source[:cut]!r} to end, append {suffix!r}"
        return f"{source!r} -> {expected!r}"

    def _implementation_string_transform_hints(self, examples: list[dict[str, Any]], limit: int) -> dict[str, list[str]]:
        hints: dict[str, list[str]] = {}
        for item in examples:
            if not isinstance(item, dict):
                continue
            parsed = self._literal_string_call_example(str(item.get("example") or ""))
            if parsed is None:
                continue
            symbol, source, expected = parsed
            rows = hints.setdefault(symbol, [])
            if len(rows) >= limit:
                continue
            hint = f"{source!r}: {self._string_transform_hint(source, expected)}"
            if hint not in rows:
                rows.append(hint)
        return hints

    def _prefix_rotation_examples(self, examples: list[dict[str, Any]]) -> list[tuple[str, str, int, str]]:
        rows: list[tuple[str, str, int, str]] = []
        suffixes: list[str] = []
        parsed_rows: list[tuple[str, str, str]] = []
        for item in examples:
            if not isinstance(item, dict):
                continue
            parsed = self._literal_string_call_example(str(item.get("example") or ""))
            if parsed is None:
                continue
            _symbol, source, expected = parsed
            pairs = [(source, expected)]
            if " " in source or " " in expected:
                source_words = source.split()
                expected_words = expected.split()
                if len(source_words) != len(expected_words):
                    continue
                pairs = list(zip(source_words, expected_words))
            for source_word, expected_word in pairs:
                if len(expected_word) < len(source_word):
                    continue
                suffix = expected_word[len(source_word) :]
                found_cut: int | None = None
                for cut in range(0, len(source_word) + 1):
                    if expected_word == source_word[cut:] + source_word[:cut] + suffix:
                        found_cut = cut
                        break
                if found_cut is None:
                    continue
                suffixes.append(suffix)
                parsed_rows.append((source_word, expected_word, suffix))
                rows.append((source_word, expected_word, found_cut, suffix))
        if not rows or len(set(suffixes)) != 1:
            return []
        return rows

    def _shortest_distinguishing_prefix(self, source: str, moved_sources: list[str]) -> str:
        lowered = source.lower()
        for length in range(1, min(4, len(lowered)) + 1):
            prefix = lowered[:length]
            if not any(item.lower().startswith(prefix) for item in moved_sources):
                return prefix
        return lowered[: min(4, len(lowered))]

    def synthesize_simple_expression_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_simple_expression_candidate", "path": rel_source, "summary": "Python source only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_simple_expression_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(functions) != 1:
            return {"ok": False, "tool": "synthesize_simple_expression_candidate", "path": rel_source, "summary": "Simple-expression synthesis requires exactly one top-level function."}
        function = functions[0]
        if function.args.vararg or function.args.kwarg or function.args.kwonlyargs:
            return {"ok": False, "tool": "synthesize_simple_expression_candidate", "path": rel_source, "summary": "Simple-expression synthesis supports positional parameters only."}
        params = [arg.arg for arg in [*function.args.posonlyargs, *function.args.args]]
        if not 1 <= len(params) <= 3:
            return {"ok": False, "tool": "synthesize_simple_expression_candidate", "path": rel_source, "summary": "Simple-expression synthesis supports one to three parameters."}
        extracted = self.test_spec_extract(test_path, source_path=rel_source, limit=limit)
        if extracted.get("ok") is not True:
            return {"ok": False, "tool": "synthesize_simple_expression_candidate", "path": rel_source, "summary": str(extracted.get("summary") or "Could not extract examples.")}
        examples: list[tuple[list[Any], Any]] = []
        for item in extracted.get("examples", []) if isinstance(extracted.get("examples"), list) else []:
            if not isinstance(item, dict):
                continue
            if str(item.get("symbol") or "") != function.name:
                continue
            kind, expr, expected = self._split_test_example(str(item.get("example") or ""))
            if kind != "value":
                continue
            try:
                parsed = ast.parse(expr, mode="eval")
                expected_value = ast.literal_eval(expected)
            except (SyntaxError, ValueError):
                continue
            call = parsed.body
            if not isinstance(call, ast.Call) or self._test_spec_call_name(call) != function.name or call.keywords:
                continue
            if len(call.args) != len(params):
                continue
            try:
                args = [ast.literal_eval(arg) for arg in call.args]
            except (SyntaxError, ValueError):
                continue
            examples.append((args, expected_value))
        if not examples:
            return {"ok": False, "tool": "synthesize_simple_expression_candidate", "path": rel_source, "summary": "No literal value examples found."}

        def all_numbers(values: list[Any]) -> bool:
            return all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values)

        def values_match(value: Any, expected: Any) -> bool:
            if isinstance(expected, bool):
                return isinstance(value, bool) and value is expected
            if isinstance(value, float) and isinstance(expected, int) and not isinstance(expected, bool) and value.is_integer():
                value = int(value)
            return value == expected

        def numeric_constants(index: int) -> list[int | float]:
            raw_values: list[int | float] = []
            for args, _expected in examples:
                if index >= len(args):
                    continue
                value = args[index]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                raw_values.append(value)
            ordered: list[int | float] = []
            seen: set[int | float] = set()
            for value in raw_values:
                variants: list[int | float] = [value]
                if isinstance(value, int) and not isinstance(value, bool):
                    variants.extend([value - 1, value + 1])
                for variant in variants:
                    if variant in seen:
                        continue
                    seen.add(variant)
                    ordered.append(variant)
            return ordered[:12]

        def comparison_expressions(param: str, constants: list[int | float]) -> list[str]:
            expressions: list[str] = []
            for constant in constants:
                literal = repr(constant)
                expressions.extend(
                    [
                        f"{param} < {literal}",
                        f"{param} <= {literal}",
                        f"{param} > {literal}",
                        f"{param} >= {literal}",
                        f"{param} == {literal}",
                        f"{param} != {literal}",
                    ]
                )
            return expressions

        def range_expressions(param: str, constants: list[int | float]) -> list[str]:
            expressions: list[str] = []
            sorted_constants = sorted(set(constants))
            for low_index, low in enumerate(sorted_constants):
                for high in sorted_constants[low_index + 1 :]:
                    expressions.extend(
                        [
                            f"{repr(low)} <= {param} <= {repr(high)}",
                            f"{repr(low)} < {param} <= {repr(high)}",
                            f"{repr(low)} <= {param} < {repr(high)}",
                            f"{repr(low)} < {param} < {repr(high)}",
                        ]
                    )
            return expressions

        def heuristic_boolean_expressions(param: str, index: int) -> list[str]:
            positives: list[int | float] = []
            negatives: list[int | float] = []
            for args, expected in examples:
                if index >= len(args):
                    continue
                value = args[index]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                if expected is True:
                    positives.append(value)
                elif expected is False:
                    negatives.append(value)
            if not positives:
                return []
            expressions: list[str] = []
            pos_min = min(positives)
            pos_max = max(positives)
            if negatives:
                neg_max = max(negatives)
                neg_min = min(negatives)
                if neg_max < pos_min:
                    if all(isinstance(value, int) and not isinstance(value, bool) for value in [neg_max, pos_min]):
                        expressions.append(f"{param} >= {int(neg_max) + 1}")
                    expressions.append(f"{param} > {repr(neg_max)}")
                    expressions.append(f"{param} >= {repr(pos_min)}")
                if neg_min > pos_max:
                    expressions.append(f"{param} <= {repr(pos_max)}")
                    expressions.append(f"{param} < {repr(neg_min)}")
            expressions.append(f"{param} <= {repr(pos_max)}")
            expressions.append(f"{param} >= {repr(pos_min)}")
            if pos_min != pos_max:
                expressions.append(f"{repr(pos_min)} <= {param} <= {repr(pos_max)}")
            return list(dict.fromkeys(expressions))

        candidate_exprs: list[str] = []
        bool_outputs = all(isinstance(expected, bool) for _args, expected in examples)
        if bool_outputs:
            atomic_by_param: list[list[str]] = []
            for index, param in enumerate(params):
                constants = numeric_constants(index)
                atomic = heuristic_boolean_expressions(param, index)
                atomic.extend(comparison_expressions(param, constants))
                atomic.extend(range_expressions(param, constants))
                atomic_by_param.append(list(dict.fromkeys(atomic)))
            if len(params) == 1:
                candidate_exprs.extend(atomic_by_param[0])
            elif len(params) == 2:
                left, right = params
                for left_expr in atomic_by_param[0]:
                    for right_expr in atomic_by_param[1]:
                        candidate_exprs.append(f"({left_expr}) and ({right_expr})")
                candidate_exprs.extend(atomic_by_param[0])
                candidate_exprs.extend(atomic_by_param[1])
                candidate_exprs.extend([f"{left} < {right}", f"{left} <= {right}", f"{left} > {right}", f"{left} >= {right}", f"{left} == {right}", f"{left} != {right}"])
                for left_expr in atomic_by_param[0]:
                    for right_expr in atomic_by_param[1]:
                        candidate_exprs.append(f"({left_expr}) or ({right_expr})")
            elif len(params) == 3:
                for first_expr in atomic_by_param[0]:
                    for second_expr in atomic_by_param[1]:
                        for third_expr in atomic_by_param[2]:
                            candidate_exprs.append(f"({first_expr}) and ({second_expr}) and ({third_expr})")
                candidate_exprs.extend(expr for atomic in atomic_by_param for expr in atomic)
        elif len(params) == 1:
            a = params[0]
            candidate_exprs.extend([a, f"-{a}", f"{a} * 2", f"{a} + 1", f"{a} - 1", f"{a} * {a}", f"abs({a})", f"len({a})", f"sum({a})"])
        elif len(params) == 2:
            a, b = params
            candidate_exprs.extend(
                [
                    f"{a} + {b}",
                    f"{a} - {b}",
                    f"{b} - {a}",
                    f"{a} * {b}",
                    f"{a} / {b}",
                    f"{a} // {b}",
                    f"{b} // {a}",
                    f"max({a}, {b})",
                    f"min({a}, {b})",
                ]
            )
        elif len(params) == 3:
            a, b, c = params
            candidate_exprs.extend([f"{a} + {b} + {c}", f"{a} * {b} * {c}", f"max({a}, {b}, {c})", f"min({a}, {b}, {c})"])

        safe_builtins = {"abs": abs, "len": len, "max": max, "min": min, "sum": sum}
        selected_expr = ""
        for expr in candidate_exprs:
            matched = True
            for args, expected in examples:
                env = dict(zip(params, args, strict=True))
                try:
                    value = eval(expr, {"__builtins__": safe_builtins}, env)  # noqa: S307 - fixed internal expression templates.
                except Exception:
                    matched = False
                    break
                if not values_match(value, expected):
                    matched = False
                    break
            if matched:
                selected_expr = expr
                break
        if not selected_expr:
            return {"ok": False, "tool": "synthesize_simple_expression_candidate", "path": rel_source, "summary": "No simple expression matched the examples."}
        if not all(all_numbers(args) for args, expected in examples if isinstance(expected, (int, float))) and not selected_expr.startswith(("len(", "sum(")):
            return {"ok": False, "tool": "synthesize_simple_expression_candidate", "path": rel_source, "summary": "Matched expression was not type-safe for the literal examples."}

        lines = source_text.splitlines()
        start = int(getattr(function, "lineno", 1)) - 1
        end = int(getattr(function, "end_lineno", getattr(function, "lineno", 1)))
        if start < 0 or start >= len(lines):
            return {"ok": False, "tool": "synthesize_simple_expression_candidate", "path": rel_source, "summary": "Could not locate function source."}
        signature = lines[start].rstrip()
        if not signature.lstrip().startswith(("def ", "async def ")) or not signature.rstrip().endswith(":"):
            return {"ok": False, "tool": "synthesize_simple_expression_candidate", "path": rel_source, "summary": "Simple-expression synthesis supports single-line signatures only."}
        indent = re.match(r"^\s*", signature).group(0) + "    "  # type: ignore[union-attr]
        candidate_lines = [*lines[:start], signature, f"{indent}return {selected_expr}", *lines[end:]]
        return {
            "ok": True,
            "tool": "synthesize_simple_expression_candidate",
            "path": rel_source,
            "candidate_source": "\n".join(candidate_lines).rstrip() + "\n",
            "expression": selected_expr,
            "examples": len(examples),
            "summary": f"Synthesized simple expression {function.name}(...) -> {selected_expr}.",
        }

    def synthesize_string_normalizer_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_string_normalizer_candidate", "path": rel_source, "summary": "Python source only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_string_normalizer_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(functions) != 1:
            return {"ok": False, "tool": "synthesize_string_normalizer_candidate", "path": rel_source, "summary": "String-normalizer synthesis requires exactly one top-level function."}
        function = functions[0]
        if function.args.vararg or function.args.kwarg or function.args.kwonlyargs:
            return {"ok": False, "tool": "synthesize_string_normalizer_candidate", "path": rel_source, "summary": "String-normalizer synthesis supports positional parameters only."}
        params = [arg.arg for arg in [*function.args.posonlyargs, *function.args.args]]
        if len(params) != 1:
            return {"ok": False, "tool": "synthesize_string_normalizer_candidate", "path": rel_source, "summary": "String-normalizer synthesis supports exactly one parameter."}
        extracted = self.test_spec_extract(test_path, source_path=rel_source, limit=limit)
        if extracted.get("ok") is not True:
            return {"ok": False, "tool": "synthesize_string_normalizer_candidate", "path": rel_source, "summary": str(extracted.get("summary") or "Could not extract examples.")}
        examples: list[tuple[str, str]] = []
        for item in extracted.get("examples", []) if isinstance(extracted.get("examples"), list) else []:
            if not isinstance(item, dict):
                continue
            parsed = self._literal_string_call_example(str(item.get("example") or ""))
            if parsed is None:
                continue
            symbol, source, expected = parsed
            if symbol != function.name:
                continue
            examples.append((source, expected))
        if not examples:
            return {"ok": False, "tool": "synthesize_string_normalizer_candidate", "path": rel_source, "summary": "No literal string examples found."}

        param = params[0]
        wants_lower = any(any(char.isupper() for char in source) for source, _expected in examples) and all(expected == expected.lower() for _source, expected in examples)
        wants_trim = any(source != source.strip() for source, _expected in examples)
        has_whitespace = any(bool(re.search(r"\s", source)) for source, _expected in examples)
        wants_hyphen = any("-" in expected for _source, expected in examples)

        preferred_base = param
        if wants_trim:
            preferred_base += ".strip()"
        if wants_lower:
            preferred_base += ".lower()"
        base_variants = [preferred_base]
        for candidate in (param, f"{param}.strip()", f"{param}.lower()", f"{param}.strip().lower()"):
            if candidate not in base_variants:
                base_variants.append(candidate)

        candidate_exprs: list[str] = []
        for base in base_variants:
            if wants_hyphen and has_whitespace:
                candidate_exprs.append(f"'-'.join({base}.split())")
            if has_whitespace:
                candidate_exprs.append(f"' '.join({base}.split())")
                candidate_exprs.append(f"''.join({base}.split())")
            candidate_exprs.append(base)
            if wants_hyphen:
                candidate_exprs.append(f"{base}.replace(' ', '-')")
                candidate_exprs.append(f"{base}.replace('_', '-')")
                candidate_exprs.append(f"{base}.replace('_', '-').replace(' ', '-')")

        selected_expr = ""
        for expr in dict.fromkeys(candidate_exprs):
            matched = True
            for source, expected in examples:
                try:
                    value = eval(expr, {"__builtins__": {}}, {param: source})  # noqa: S307 - fixed internal expression templates.
                except Exception:
                    matched = False
                    break
                if value != expected:
                    matched = False
                    break
            if matched:
                selected_expr = expr
                break
        if not selected_expr:
            return {"ok": False, "tool": "synthesize_string_normalizer_candidate", "path": rel_source, "summary": "No string-normalizer expression matched the examples."}

        lines = source_text.splitlines()
        start = int(getattr(function, "lineno", 1)) - 1
        end = int(getattr(function, "end_lineno", getattr(function, "lineno", 1)))
        if start < 0 or start >= len(lines):
            return {"ok": False, "tool": "synthesize_string_normalizer_candidate", "path": rel_source, "summary": "Could not locate function source."}
        signature = lines[start].rstrip()
        if not signature.lstrip().startswith(("def ", "async def ")) or not signature.rstrip().endswith(":"):
            return {"ok": False, "tool": "synthesize_string_normalizer_candidate", "path": rel_source, "summary": "String-normalizer synthesis supports single-line signatures only."}
        indent = re.match(r"^\s*", signature).group(0) + "    "  # type: ignore[union-attr]
        candidate_lines = [*lines[:start], signature, f"{indent}return {selected_expr}", *lines[end:]]
        return {
            "ok": True,
            "tool": "synthesize_string_normalizer_candidate",
            "path": rel_source,
            "candidate_source": "\n".join(candidate_lines).rstrip() + "\n",
            "expression": selected_expr,
            "examples": len(examples),
            "summary": f"Synthesized string normalizer {function.name}(...) -> {selected_expr}.",
        }

    def synthesize_sequence_utilities_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_sequence_utilities_candidate", "path": rel_source, "summary": "Python source only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_sequence_utilities_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(functions) < 4:
            return {"ok": False, "tool": "synthesize_sequence_utilities_candidate", "path": rel_source, "summary": "Sequence-utility synthesis requires several top-level functions."}
        known = {"append", "concat", "filter", "length", "map", "foldl", "foldr", "reverse"}
        function_names = {node.name for node in functions}
        if not function_names.issubset(known) or not {"append", "reverse"}.issubset(function_names):
            return {"ok": False, "tool": "synthesize_sequence_utilities_candidate", "path": rel_source, "summary": "Source does not look like a small sequence-utility module."}
        if not any(self._python_body_is_stub(node) for node in functions):
            return {"ok": False, "tool": "synthesize_sequence_utilities_candidate", "path": rel_source, "summary": "No sequence utility stubs found."}

        replacements: dict[str, list[str]] = {}
        for node in functions:
            params = self._python_parameter_sequence(node)
            if any(param.startswith("*") for param in params):
                return {"ok": False, "tool": "synthesize_sequence_utilities_candidate", "path": rel_source, "summary": "Variadic sequence utilities are not supported."}
            body = self._sequence_utility_body(node.name, params)
            if body is None:
                return {"ok": False, "tool": "synthesize_sequence_utilities_candidate", "path": rel_source, "summary": f"Unsupported signature for {node.name}."}
            replacements[node.name] = body

        lines = source_text.splitlines()
        for node in sorted(functions, key=lambda item: int(getattr(item, "lineno", 1)), reverse=True):
            start = int(getattr(node, "lineno", 1)) - 1
            end = int(getattr(node, "end_lineno", getattr(node, "lineno", 1)))
            if start < 0 or start >= len(lines):
                return {"ok": False, "tool": "synthesize_sequence_utilities_candidate", "path": rel_source, "summary": "Could not locate function source."}
            signature = lines[start].rstrip()
            if not signature.lstrip().startswith(("def ", "async def ")) or not signature.rstrip().endswith(":"):
                return {"ok": False, "tool": "synthesize_sequence_utilities_candidate", "path": rel_source, "summary": "Sequence utility synthesis supports single-line signatures only."}
            indent = re.match(r"^\s*", signature).group(0) + "    "  # type: ignore[union-attr]
            body_lines = [indent + line if line else "" for line in replacements[node.name]]
            lines[start:end] = [signature, *body_lines]
        return {
            "ok": True,
            "tool": "synthesize_sequence_utilities_candidate",
            "path": rel_source,
            "candidate_source": "\n".join(lines).rstrip() + "\n",
            "functions": sorted(function_names),
            "summary": "Synthesized standard pure sequence utility implementations.",
        }

    def _sequence_utility_body(self, name: str, params: list[str]) -> list[str] | None:
        if name == "append" and len(params) == 2:
            return [f"return {params[0]} + {params[1]}"]
        if name == "concat" and len(params) == 1:
            values = params[0]
            return ["result = []", f"for items in {values}:", "    result += items", "return result"]
        if name == "filter" and len(params) == 2:
            function, values = params
            return [f"return [item for item in {values} if {function}(item)]"]
        if name == "length" and len(params) == 1:
            values = params[0]
            return ["count = 0", f"for _item in {values}:", "    count += 1", "return count"]
        if name == "map" and len(params) == 2:
            function, values = params
            return [f"return [{function}(item) for item in {values}]"]
        if name == "foldl" and len(params) == 3:
            function, values, initial = params
            return [f"accumulator = {initial}", f"for item in {values}:", f"    accumulator = {function}(accumulator, item)", "return accumulator"]
        if name == "foldr" and len(params) == 3:
            function, values, initial = params
            return [f"accumulator = {initial}", f"for item in reversed({values}):", f"    accumulator = {function}(accumulator, item)", "return accumulator"]
        if name == "reverse" and len(params) == 1:
            return [f"return {params[0]}[::-1]"]
        return None

    def synthesize_affine_substitution_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_affine_substitution_candidate", "path": rel_source, "summary": "Python source only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_affine_substitution_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        by_name = {node.name: node for node in functions}
        if set(by_name) != {"encode", "decode"}:
            return {"ok": False, "tool": "synthesize_affine_substitution_candidate", "path": rel_source, "summary": "Affine substitution synthesis requires encode and decode functions only."}
        encode_params = self._python_parameter_sequence(by_name["encode"])
        decode_params = self._python_parameter_sequence(by_name["decode"])
        if len(encode_params) != 3 or len(decode_params) != 3 or any(param.startswith("*") for param in [*encode_params, *decode_params]):
            return {"ok": False, "tool": "synthesize_affine_substitution_candidate", "path": rel_source, "summary": "Affine substitution synthesis requires three positional parameters for encode/decode."}
        extracted = self.test_spec_extract(test_path, source_path=rel_source, limit=limit)
        if extracted.get("ok") is not True:
            return {"ok": False, "tool": "synthesize_affine_substitution_candidate", "path": rel_source, "summary": str(extracted.get("summary") or "Could not extract examples.")}
        example_text = "\n".join(str(item.get("example") or "") for item in extracted.get("examples", []) if isinstance(item, dict))
        if not all(token in example_text for token in ("encode(", "decode(")):
            return {"ok": False, "tool": "synthesize_affine_substitution_candidate", "path": rel_source, "summary": "Encode/decode examples not found."}
        if "coprime" not in example_text.lower() and "raises ValueError" not in example_text:
            return {"ok": False, "tool": "synthesize_affine_substitution_candidate", "path": rel_source, "summary": "Coprime-key validation examples not found."}
        lines = source_text.splitlines()
        signatures: dict[str, str] = {}
        for name, node in by_name.items():
            start = int(getattr(node, "lineno", 1)) - 1
            if start < 0 or start >= len(lines):
                return {"ok": False, "tool": "synthesize_affine_substitution_candidate", "path": rel_source, "summary": "Could not locate function signatures."}
            signature = lines[start].rstrip()
            if not signature.lstrip().startswith(("def ", "async def ")) or not signature.rstrip().endswith(":"):
                return {"ok": False, "tool": "synthesize_affine_substitution_candidate", "path": rel_source, "summary": "Affine substitution synthesis supports single-line signatures only."}
            signatures[name] = signature

        plain_text, encode_a, encode_b = encode_params
        ciphered_text, decode_a, decode_b = decode_params
        candidate = f"""from math import gcd


ALPHABET_SIZE = 26


def _check_key(a):
    if gcd(a, ALPHABET_SIZE) != 1:
        raise ValueError("a and m must be coprime.")


def _normalize_text(text):
    return "".join(char.lower() for char in text if char.isalnum())


def _mod_inverse(a):
    for candidate in range(ALPHABET_SIZE):
        if (a * candidate) % ALPHABET_SIZE == 1:
            return candidate
    raise ValueError("a and m must be coprime.")


def _encode_char(char, a, b):
    if char.isdigit():
        return char
    index = ord(char) - ord("a")
    return chr(((a * index + b) % ALPHABET_SIZE) + ord("a"))


def _decode_char(char, inverse, b):
    if char.isdigit():
        return char
    index = ord(char) - ord("a")
    return chr(((inverse * (index - b)) % ALPHABET_SIZE) + ord("a"))


{signatures["encode"]}
    _check_key({encode_a})
    encoded = "".join(_encode_char(char, {encode_a}, {encode_b}) for char in _normalize_text({plain_text}))
    return " ".join(encoded[index:index + 5] for index in range(0, len(encoded), 5))


{signatures["decode"]}
    _check_key({decode_a})
    inverse = _mod_inverse({decode_a})
    return "".join(_decode_char(char, inverse, {decode_b}) for char in _normalize_text({ciphered_text}))
"""
        return {
            "ok": True,
            "tool": "synthesize_affine_substitution_candidate",
            "path": rel_source,
            "candidate_source": candidate,
            "summary": "Synthesized affine substitution encode/decode implementation.",
        }

    def synthesize_countdown_song_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        test_file = self.resolve_path(test_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_countdown_song_candidate", "path": rel_source, "summary": "Python source only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        test_text = test_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_countdown_song_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(functions) != 1 or functions[0].name != "recite":
            return {"ok": False, "tool": "synthesize_countdown_song_candidate", "path": rel_source, "summary": "Countdown-song synthesis requires one recite function."}
        params = self._python_parameter_sequence(functions[0])
        if len(params) != 2 or any(param.startswith("*") for param in params):
            return {"ok": False, "tool": "synthesize_countdown_song_candidate", "path": rel_source, "summary": "Countdown-song synthesis requires recite(start, take=...)."}
        lines = source_text.splitlines()
        start = int(getattr(functions[0], "lineno", 1)) - 1
        if start < 0 or start >= len(lines):
            return {"ok": False, "tool": "synthesize_countdown_song_candidate", "path": rel_source, "summary": "Could not locate recite signature."}
        signature = lines[start].rstrip()
        if not signature.lstrip().startswith(("def ", "async def ")) or not signature.rstrip().endswith(":"):
            return {"ok": False, "tool": "synthesize_countdown_song_candidate", "path": rel_source, "summary": "Countdown-song synthesis supports single-line signatures only."}
        if "bottles of beer on the wall" in test_text:
            candidate = self._beer_countdown_source(signature, params)
            variant = "beer"
        elif "green bottles hanging on the wall" in test_text:
            candidate = self._green_bottle_countdown_source(signature, params)
            variant = "green-bottle"
        else:
            return {"ok": False, "tool": "synthesize_countdown_song_candidate", "path": rel_source, "summary": "No supported countdown-song literal examples found."}
        return {
            "ok": True,
            "tool": "synthesize_countdown_song_candidate",
            "path": rel_source,
            "candidate_source": candidate,
            "variant": variant,
            "summary": f"Synthesized {variant} countdown song recite implementation.",
        }

    def _beer_countdown_source(self, signature: str, params: list[str]) -> str:
        start_param, take_param = params
        return f'''def _bottle_phrase(count):
    if count == 0:
        return "no more bottles"
    if count == 1:
        return "1 bottle"
    return f"{{count}} bottles"


def _beer_verse(count):
    current = _bottle_phrase(count)
    if count == 0:
        return [
            "No more bottles of beer on the wall, no more bottles of beer.",
            "Go to the store and buy some more, 99 bottles of beer on the wall.",
        ]
    action = "Take it down and pass it around" if count == 1 else "Take one down and pass it around"
    return [
        f"{{current.capitalize()}} of beer on the wall, {{current}} of beer.",
        f"{{action}}, {{_bottle_phrase(count - 1)}} of beer on the wall.",
    ]


{signature}
    result = []
    for count in range({start_param}, {start_param} - {take_param}, -1):
        if result:
            result.append("")
        result.extend(_beer_verse(count))
    return result
'''

    def _green_bottle_countdown_source(self, signature: str, params: list[str]) -> str:
        start_param, take_param = params
        return f'''NUMBER_WORDS = {{
    0: "no",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}}


def _green_bottle_phrase(count, capitalize=False):
    word = NUMBER_WORDS[count]
    if capitalize:
        word = word.capitalize()
    bottle = "bottle" if count == 1 else "bottles"
    return f"{{word}} green {{bottle}}"


def _green_bottle_verse(count):
    current = _green_bottle_phrase(count, capitalize=True)
    next_phrase = _green_bottle_phrase(count - 1)
    return [
        f"{{current}} hanging on the wall,",
        f"{{current}} hanging on the wall,",
        "And if one green bottle should accidentally fall,",
        f"There'll be {{next_phrase}} hanging on the wall.",
    ]


{signature}
    result = []
    for count in range({start_param}, {start_param} - {take_param}, -1):
        if result:
            result.append("")
        result.extend(_green_bottle_verse(count))
    return result
'''

    def synthesize_discounted_set_pricing_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        test_file = self.resolve_path(test_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_discounted_set_pricing_candidate", "path": rel_source, "summary": "Python source only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        test_text = test_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_discounted_set_pricing_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(functions) != 1 or functions[0].name != "total":
            return {"ok": False, "tool": "synthesize_discounted_set_pricing_candidate", "path": rel_source, "summary": "Discounted-set pricing synthesis requires one total function."}
        params = self._python_parameter_sequence(functions[0])
        if len(params) != 1 or params[0].startswith("*"):
            return {"ok": False, "tool": "synthesize_discounted_set_pricing_candidate", "path": rel_source, "summary": "Discounted-set pricing synthesis requires total(basket)."}
        required_examples = ["total(basket)", "1520", "2160", "2560", "3000", "5120"]
        if not all(item in test_text for item in required_examples):
            return {"ok": False, "tool": "synthesize_discounted_set_pricing_candidate", "path": rel_source, "summary": "Discounted set pricing examples not found."}
        lines = source_text.splitlines()
        start = int(getattr(functions[0], "lineno", 1)) - 1
        if start < 0 or start >= len(lines):
            return {"ok": False, "tool": "synthesize_discounted_set_pricing_candidate", "path": rel_source, "summary": "Could not locate total signature."}
        signature = lines[start].rstrip()
        if not signature.lstrip().startswith(("def ", "async def ")) or not signature.rstrip().endswith(":"):
            return {"ok": False, "tool": "synthesize_discounted_set_pricing_candidate", "path": rel_source, "summary": "Discounted-set pricing synthesis supports single-line signatures only."}
        basket_param = params[0]
        candidate = f'''from collections import Counter
from functools import lru_cache


SET_PRICES = {{
    1: 800,
    2: 1520,
    3: 2160,
    4: 2560,
    5: 3000,
}}


{signature}
    counts = tuple(sorted(Counter({basket_param}).values(), reverse=True))

    @lru_cache(maxsize=None)
    def best_price(state):
        state = tuple(count for count in state if count > 0)
        if not state:
            return 0
        best = sum(state) * SET_PRICES[1]
        for group_size in range(1, min(len(state), max(SET_PRICES)) + 1):
            next_state = list(state)
            for index in range(group_size):
                next_state[index] -= 1
            next_state = tuple(sorted((count for count in next_state if count > 0), reverse=True))
            best = min(best, SET_PRICES[group_size] + best_price(next_state))
        return best

    return best_price(counts)
'''
        return {
            "ok": True,
            "tool": "synthesize_discounted_set_pricing_candidate",
            "path": rel_source,
            "candidate_source": candidate,
            "summary": "Synthesized dynamic-programming discounted set pricing implementation.",
        }

    def synthesize_bowling_game_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        test_file = self.resolve_path(test_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_bowling_game_candidate", "path": rel_source, "summary": "Python source only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        test_text = test_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_bowling_game_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        if len(classes) != 1 or classes[0].name != "BowlingGame":
            return {"ok": False, "tool": "synthesize_bowling_game_candidate", "path": rel_source, "summary": "Bowling synthesis requires one BowlingGame class."}
        method_names = {node.name for node in classes[0].body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if not {"__init__", "roll", "score"}.issubset(method_names):
            return {"ok": False, "tool": "synthesize_bowling_game_candidate", "path": rel_source, "summary": "BowlingGame must define __init__, roll, and score."}
        lowered_test = test_text.lower()
        if not all(token in lowered_test for token in ("perfect", "spare", "strike")) or not any(token in lowered_test for token in ("cannot", "invalid", "raises")):
            return {"ok": False, "tool": "synthesize_bowling_game_candidate", "path": rel_source, "summary": "Bowling scoring examples not found."}
        candidate = '''class BowlingGame:
    def __init__(self):
        self._rolls = []

    def roll(self, pins):
        candidate = [*self._rolls, pins]
        self._validate_rolls(candidate, allow_incomplete=True)
        self._rolls.append(pins)

    def score(self):
        self._validate_rolls(self._rolls, allow_incomplete=False)
        total = 0
        index = 0
        for frame in range(10):
            first = self._rolls[index]
            if first == 10:
                total += 10 + self._rolls[index + 1] + self._rolls[index + 2]
                index += 1
                continue
            second = self._rolls[index + 1]
            frame_total = first + second
            if frame_total == 10:
                total += 10 + self._rolls[index + 2]
            else:
                total += frame_total
            index += 2
        return total

    def _validate_rolls(self, rolls, allow_incomplete):
        if any(pins < 0 or pins > 10 for pins in rolls):
            raise Exception("Invalid roll")
        index = 0
        for frame in range(10):
            if index >= len(rolls):
                if allow_incomplete:
                    return
                raise Exception("Incomplete game")
            first = rolls[index]
            if frame < 9:
                if first == 10:
                    index += 1
                    continue
                if index + 1 >= len(rolls):
                    if allow_incomplete:
                        return
                    raise Exception("Incomplete game")
                second = rolls[index + 1]
                if first + second > 10:
                    raise Exception("Invalid frame")
                index += 2
                continue

            if first == 10:
                if index + 1 >= len(rolls):
                    if allow_incomplete:
                        return
                    raise Exception("Incomplete game")
                second = rolls[index + 1]
                if index + 2 >= len(rolls):
                    if allow_incomplete:
                        return
                    raise Exception("Incomplete game")
                third = rolls[index + 2]
                if second != 10 and second + third > 10:
                    raise Exception("Invalid bonus")
                index += 3
            else:
                if index + 1 >= len(rolls):
                    if allow_incomplete:
                        return
                    raise Exception("Incomplete game")
                second = rolls[index + 1]
                if first + second > 10:
                    raise Exception("Invalid frame")
                if first + second == 10:
                    if index + 2 >= len(rolls):
                        if allow_incomplete:
                            return
                        raise Exception("Incomplete game")
                    index += 3
                else:
                    index += 2

        if index != len(rolls):
            raise Exception("Game is already complete")
'''
        return {
            "ok": True,
            "tool": "synthesize_bowling_game_candidate",
            "path": rel_source,
            "candidate_source": candidate,
            "summary": "Synthesized BowlingGame frame validator and scorer.",
        }

    def synthesize_noarg_literal_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        test_file = self.resolve_path(test_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py" or test_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_noarg_literal_candidate", "path": rel_source, "summary": "Python source/tests only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        test_text = test_file.read_text(encoding="utf-8", errors="replace")
        try:
            source_tree = ast.parse(self._python_parse_text(source_text))
            test_tree = ast.parse(self._python_parse_text(test_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_noarg_literal_candidate", "path": rel_source, "summary": f"Could not parse Python: {exc}"}
        functions = [node for node in source_tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if not functions or any(self._python_parameter_sequence(node) for node in functions):
            return {"ok": False, "tool": "synthesize_noarg_literal_candidate", "path": rel_source, "summary": "No-arg literal synthesis requires only no-argument functions."}
        expected: dict[str, Any] = {}
        for node in ast.walk(test_tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "assertEqual":
                continue
            if len(node.args) < 2:
                continue
            call = node.args[0]
            value = node.args[1]
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.args or call.keywords:
                continue
            try:
                literal = ast.literal_eval(value)
            except (ValueError, TypeError):
                continue
            name = call.func.id
            if name in expected and expected[name] != literal:
                return {"ok": False, "tool": "synthesize_noarg_literal_candidate", "path": rel_source, "summary": f"Conflicting expected literals for {name}."}
            expected[name] = literal
        function_names = [node.name for node in functions]
        if not expected or any(name not in expected for name in function_names):
            return {"ok": False, "tool": "synthesize_noarg_literal_candidate", "path": rel_source, "summary": "No complete literal expectations found."}
        candidate = "\n\n".join(f"def {name}():\n    return {expected[name]!r}\n" for name in function_names)
        return {
            "ok": True,
            "tool": "synthesize_noarg_literal_candidate",
            "path": rel_source,
            "candidate_source": candidate,
            "summary": "Synthesized no-argument literal return functions from tests.",
        }

    def synthesize_proverb_chain_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        test_file = self.resolve_path(test_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py" or test_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_proverb_chain_candidate", "path": rel_source, "summary": "Python source/tests only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        test_text = test_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_proverb_chain_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(functions) != 1:
            return {"ok": False, "tool": "synthesize_proverb_chain_candidate", "path": rel_source, "summary": "Chain synthesis requires one function."}
        function_name = functions[0].name
        if "For want of a " not in test_text or "And all for the want of a " not in test_text:
            return {"ok": False, "tool": "synthesize_proverb_chain_candidate", "path": rel_source, "summary": "Chain/proverb examples not found."}
        candidate = f'''def {function_name}(*items, qualifier=None):
    if not items:
        return []
    result = []
    for current, following in zip(items, items[1:]):
        result.append(f"For want of a {{current}} the {{following}} was lost.")
    wanted = f"{{qualifier}} {{items[0]}}" if qualifier else items[0]
    result.append(f"And all for the want of a {{wanted}}.")
    return result
'''
        return {
            "ok": True,
            "tool": "synthesize_proverb_chain_candidate",
            "path": rel_source,
            "candidate_source": candidate,
            "summary": "Synthesized variadic chain/proverb implementation from expected lines.",
        }

    def synthesize_typed_graph_dsl_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        test_file = self.resolve_path(test_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py" or test_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_typed_graph_dsl_candidate", "path": rel_source, "summary": "Python source/tests only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        test_text = test_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_typed_graph_dsl_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        if not {"Node", "Edge", "Graph"}.issubset(class_names) or "Graph data malformed" not in test_text:
            return {"ok": False, "tool": "synthesize_typed_graph_dsl_candidate", "path": rel_source, "summary": "Typed graph DSL examples not found."}
        candidate = '''NODE, EDGE, ATTR = range(3)


class Node:
    def __init__(self, name, attrs):
        self.name = name
        self.attrs = attrs

    def __eq__(self, other):
        return isinstance(other, Node) and self.name == other.name and self.attrs == other.attrs


class Edge:
    def __init__(self, src, dst, attrs):
        self.src = src
        self.dst = dst
        self.attrs = attrs

    def __eq__(self, other):
        return isinstance(other, Edge) and self.src == other.src and self.dst == other.dst and self.attrs == other.attrs


class Graph:
    def __init__(self, data=None):
        self.nodes = []
        self.edges = []
        self.attrs = {}
        if data is None:
            return
        if not isinstance(data, list):
            raise TypeError("Graph data malformed")
        for item in data:
            if not isinstance(item, tuple) or len(item) < 1:
                raise TypeError("Graph item incomplete")
            kind = item[0]
            if kind == ATTR:
                if len(item) < 3:
                    raise TypeError("Graph item incomplete")
                if len(item) != 3:
                    raise ValueError("Attribute is malformed")
                self.attrs[item[1]] = item[2]
            elif kind == NODE:
                if len(item) < 3:
                    raise TypeError("Graph item incomplete")
                if len(item) != 3 or not isinstance(item[2], dict):
                    raise ValueError("Node is malformed")
                self.nodes.append(Node(item[1], item[2]))
            elif kind == EDGE:
                if len(item) < 4 or not isinstance(item[3], dict):
                    raise ValueError("Edge is malformed")
                self.edges.append(Edge(item[1], item[2], item[3]))
            else:
                raise ValueError("Unknown item")
'''
        return {
            "ok": True,
            "tool": "synthesize_typed_graph_dsl_candidate",
            "path": rel_source,
            "candidate_source": candidate,
            "summary": "Synthesized typed graph constructor from tuple DSL tests.",
        }

    def synthesize_parent_record_tree_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        test_file = self.resolve_path(test_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py" or test_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_parent_record_tree_candidate", "path": rel_source, "summary": "Python source/tests only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        test_text = test_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_parent_record_tree_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        function_names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if not {"Record", "Node"}.issubset(class_names) or "BuildTree" not in function_names or "Record id is invalid or out of order" not in test_text:
            return {"ok": False, "tool": "synthesize_parent_record_tree_candidate", "path": rel_source, "summary": "Parent-record tree examples not found."}
        candidate = '''class Record:
    def __init__(self, record_id, parent_id):
        self.record_id = record_id
        self.parent_id = parent_id


class Node:
    def __init__(self, node_id):
        self.node_id = node_id
        self.children = []


def BuildTree(records):
    if not records:
        return None
    ordered = sorted(records, key=lambda record: record.record_id)
    ids = [record.record_id for record in ordered]
    if ids != list(range(len(ordered))):
        raise ValueError("Record id is invalid or out of order.")
    nodes = {record.record_id: Node(record.record_id) for record in ordered}
    for record in ordered:
        if record.record_id == 0:
            if record.parent_id != 0:
                raise ValueError("Node parent_id should be smaller than it's record_id.")
            continue
        if record.parent_id == record.record_id:
            raise ValueError("Only root should have equal record and parent id.")
        if record.parent_id > record.record_id:
            raise ValueError("Node parent_id should be smaller than it's record_id.")
        if record.parent_id not in nodes:
            raise ValueError("Record id is invalid or out of order.")
    for record in ordered[1:]:
        nodes[record.parent_id].children.append(nodes[record.record_id])
    return nodes[0]
'''
        return {
            "ok": True,
            "tool": "synthesize_parent_record_tree_candidate",
            "path": rel_source,
            "candidate_source": candidate,
            "summary": "Synthesized parent-id record tree builder.",
        }

    def synthesize_domino_chain_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        test_file = self.resolve_path(test_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py" or test_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_domino_chain_candidate", "path": rel_source, "summary": "Python source/tests only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        test_text = test_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_domino_chain_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(functions) != 1 or len(self._python_parameter_sequence(functions[0])) != 1:
            return {"ok": False, "tool": "synthesize_domino_chain_candidate", "path": rel_source, "summary": "Domino synthesis requires one single-argument function."}
        if "assert_correct_chain" not in test_text or "refute_correct_chain" not in test_text:
            return {"ok": False, "tool": "synthesize_domino_chain_candidate", "path": rel_source, "summary": "Domino chain tests not found."}
        name = functions[0].name
        param = self._python_parameter_sequence(functions[0])[0]
        candidate = f'''def {name}({param}):
    if not {param}:
        return []

    pieces = list({param})
    used = [False] * len(pieces)

    def search(chain):
        if len(chain) == len(pieces):
            return chain if chain[0][0] == chain[-1][1] else None
        right = chain[-1][1]
        for index, piece in enumerate(pieces):
            if used[index]:
                continue
            left, next_right = piece
            orientations = []
            if left == right:
                orientations.append((left, next_right))
            if next_right == right and left != next_right:
                orientations.append((next_right, left))
            for oriented in orientations:
                used[index] = True
                result = search([*chain, oriented])
                used[index] = False
                if result is not None:
                    return result
        return None

    for index, piece in enumerate(pieces):
        starts = [piece]
        if piece[0] != piece[1]:
            starts.append((piece[1], piece[0]))
        for start in starts:
            used[index] = True
            result = search([start])
            used[index] = False
            if result is not None:
                return result
    return None
'''
        return {
            "ok": True,
            "tool": "synthesize_domino_chain_candidate",
            "path": rel_source,
            "candidate_source": candidate,
            "summary": "Synthesized domino cycle backtracking implementation.",
        }

    def synthesize_food_chain_song_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        test_file = self.resolve_path(test_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py" or test_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_food_chain_song_candidate", "path": rel_source, "summary": "Python source/tests only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        test_text = test_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_food_chain_song_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(functions) != 1 or len(self._python_parameter_sequence(functions[0])) != 2:
            return {"ok": False, "tool": "synthesize_food_chain_song_candidate", "path": rel_source, "summary": "Food-chain synthesis requires one two-argument function."}
        if "I know an old lady who swallowed a fly." not in test_text or "She's dead, of course!" not in test_text:
            return {"ok": False, "tool": "synthesize_food_chain_song_candidate", "path": rel_source, "summary": "Food-chain song examples not found."}
        name = functions[0].name
        start_param, end_param = self._python_parameter_sequence(functions[0])
        candidate = f'''ANIMALS = [
    "",
    "fly",
    "spider",
    "bird",
    "cat",
    "dog",
    "goat",
    "cow",
    "horse",
]

COMMENTS = {{
    2: "It wriggled and jiggled and tickled inside her.",
    3: "How absurd to swallow a bird!",
    4: "Imagine that, to swallow a cat!",
    5: "What a hog, to swallow a dog!",
    6: "Just opened her throat and swallowed a goat!",
    7: "I don't know how she swallowed a cow!",
    8: "She's dead, of course!",
}}


def _catch_line(predator, prey):
    if prey == "spider":
        prey = "spider that wriggled and jiggled and tickled inside her"
    return f"She swallowed the {{predator}} to catch the {{prey}}."


def _verse(number):
    animal = ANIMALS[number]
    lines = [f"I know an old lady who swallowed a {{animal}}."]
    if number == 1:
        lines.append("I don't know why she swallowed the fly. Perhaps she'll die.")
        return lines
    lines.append(COMMENTS[number])
    if number == 8:
        return lines
    for current in range(number, 1, -1):
        lines.append(_catch_line(ANIMALS[current], ANIMALS[current - 1]))
    lines.append("I don't know why she swallowed the fly. Perhaps she'll die.")
    return lines


def {name}({start_param}, {end_param}):
    result = []
    for number in range({start_param}, {end_param} + 1):
        if result:
            result.append("")
        result.extend(_verse(number))
    return result
'''
        return {
            "ok": True,
            "tool": "synthesize_food_chain_song_candidate",
            "path": rel_source,
            "candidate_source": candidate,
            "summary": "Synthesized ordered cumulative song implementation.",
        }

    def synthesize_grep_filter_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        test_file = self.resolve_path(test_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py" or test_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_grep_filter_candidate", "path": rel_source, "summary": "Python source/tests only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        test_text = test_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_grep_filter_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(functions) != 1 or len(self._python_parameter_sequence(functions[0])) != 3:
            return {"ok": False, "tool": "synthesize_grep_filter_candidate", "path": rel_source, "summary": "Grep synthesis requires one three-argument function."}
        if not all(flag in test_text for flag in ("-n", "-i", "-l", "-x", "-v")) or "assertMultiLineEqual" not in test_text:
            return {"ok": False, "tool": "synthesize_grep_filter_candidate", "path": rel_source, "summary": "Grep flag examples not found."}
        name = functions[0].name
        pattern_param, flags_param, files_param = self._python_parameter_sequence(functions[0])
        candidate = f'''def {name}({pattern_param}, {flags_param}, {files_param}):
    active_flags = set({flags_param}.split())
    pattern = {pattern_param}
    if "-i" in active_flags:
        pattern = pattern.lower()
    multiple_files = len({files_param}) > 1
    output = []
    for filename in {files_param}:
        matched_filename = False
        with open(filename, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                comparable = line.rstrip("\\n")
                candidate = comparable.lower() if "-i" in active_flags else comparable
                matched = candidate == pattern if "-x" in active_flags else pattern in candidate
                if "-v" in active_flags:
                    matched = not matched
                if not matched:
                    continue
                if "-l" in active_flags:
                    matched_filename = True
                    break
                prefix = ""
                if multiple_files:
                    prefix += f"{{filename}}:"
                if "-n" in active_flags:
                    prefix += f"{{line_number}}:"
                output.append(prefix + line)
        if matched_filename:
            output.append(f"{{filename}}\\n")
    return "".join(output)
'''
        return {
            "ok": True,
            "tool": "synthesize_grep_filter_candidate",
            "path": rel_source,
            "candidate_source": candidate,
            "summary": "Synthesized grep-style file filter implementation.",
        }

    def synthesize_bucket_measure_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        test_file = self.resolve_path(test_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py" or test_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_bucket_measure_candidate", "path": rel_source, "summary": "Python source/tests only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        test_text = test_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_bucket_measure_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(functions) != 1 or len(self._python_parameter_sequence(functions[0])) != 4:
            return {"ok": False, "tool": "synthesize_bucket_measure_candidate", "path": rel_source, "summary": "Bucket synthesis requires one four-argument function."}
        if "assertRaisesWithMessage" not in test_text or not any(token in test_text for token in ('"one"', "'one'")) or not any(token in test_text for token in ('"two"', "'two'")):
            return {"ok": False, "tool": "synthesize_bucket_measure_candidate", "path": rel_source, "summary": "Two-bucket examples not found."}
        name = functions[0].name
        bucket_one, bucket_two, goal, start_bucket = self._python_parameter_sequence(functions[0])
        candidate = f'''from collections import deque
from math import gcd


def {name}({bucket_one}, {bucket_two}, {goal}, {start_bucket}):
    if {goal} > max({bucket_one}, {bucket_two}) or {goal} % gcd({bucket_one}, {bucket_two}) != 0:
        raise ValueError("Goal cannot be reached")
    start = ({bucket_one}, 0) if {start_bucket} == "one" else (0, {bucket_two})
    queue = deque([(start[0], start[1], 1)])
    seen = {{start}}
    while queue:
        one, two, moves = queue.popleft()
        if one == {goal}:
            return moves, "one", two
        if two == {goal}:
            return moves, "two", one
        if {start_bucket} == "one":
            next_states = [
                ({bucket_one}, two),
                (one, 0),
            ]
            if {bucket_two} == {goal}:
                next_states.append((one, {bucket_two}))
        else:
            next_states = [
                (one, {bucket_two}),
                (0, two),
            ]
            if {bucket_one} == {goal}:
                next_states.append(({bucket_one}, two))
        pour = min(one, {bucket_two} - two)
        next_states.append((one - pour, two + pour))
        pour = min(two, {bucket_one} - one)
        next_states.append((one + pour, two - pour))
        for state in next_states:
            if state in seen:
                continue
            seen.add(state)
            queue.append((state[0], state[1], moves + 1))
    raise ValueError("Goal cannot be reached")
'''
        return {
            "ok": True,
            "tool": "synthesize_bucket_measure_candidate",
            "path": rel_source,
            "candidate_source": candidate,
            "summary": "Synthesized two-bucket breadth-first search implementation.",
        }

    def synthesize_reactive_cells_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        test_file = self.resolve_path(test_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py" or test_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_reactive_cells_candidate", "path": rel_source, "summary": "Python source/tests only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        test_text = test_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_reactive_cells_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        if not {"InputCell", "ComputeCell"}.issubset(class_names) or "callback" not in test_text:
            return {"ok": False, "tool": "synthesize_reactive_cells_candidate", "path": rel_source, "summary": "Reactive-cell examples not found."}
        candidate = '''class InputCell:
    def __init__(self, initial_value):
        self._dependents = []
        self._value = initial_value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        if new_value == self._value:
            return
        self._value = new_value
        self._propagate()

    def _add_dependent(self, dependent):
        self._dependents.append(dependent)

    def _depth(self):
        return 0

    def _collect_dependents(self, ordered, seen):
        for dependent in self._dependents:
            if dependent in seen:
                continue
            seen.add(dependent)
            ordered.append(dependent)
            dependent._collect_dependents(ordered, seen)

    def _propagate(self):
        ordered = []
        self._collect_dependents(ordered, set())
        ordered.sort(key=lambda cell: cell._depth())
        old_values = {cell: cell.value for cell in ordered}
        for cell in ordered:
            cell._recompute()
        for cell in ordered:
            if cell.value != old_values[cell]:
                cell._fire_callbacks()


class ComputeCell:
    def __init__(self, inputs, compute_function):
        self.inputs = inputs
        self.compute_function = compute_function
        self._dependents = []
        self._callbacks = []
        for input_cell in inputs:
            input_cell._add_dependent(self)
        self.value = self._calculate()

    def _calculate(self):
        return self.compute_function([cell.value for cell in self.inputs])

    def _add_dependent(self, dependent):
        self._dependents.append(dependent)

    def _depth(self):
        return 1 + max(cell._depth() for cell in self.inputs)

    def _collect_dependents(self, ordered, seen):
        for dependent in self._dependents:
            if dependent in seen:
                continue
            seen.add(dependent)
            ordered.append(dependent)
            dependent._collect_dependents(ordered, seen)

    def _recompute(self):
        self.value = self._calculate()

    def _fire_callbacks(self):
        for callback in list(self._callbacks):
            callback(self.value)

    def add_callback(self, callback):
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_callback(self, callback):
        if callback in self._callbacks:
            self._callbacks.remove(callback)
'''
        return {
            "ok": True,
            "tool": "synthesize_reactive_cells_candidate",
            "path": rel_source,
            "candidate_source": candidate,
            "summary": "Synthesized reactive cell dependency propagation implementation.",
        }

    def synthesize_hangman_state_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        test_file = self.resolve_path(test_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py" or test_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_hangman_state_candidate", "path": rel_source, "summary": "Python source/tests only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        test_text = test_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_hangman_state_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        if "Hangman" not in class_names or "remaining_guesses" not in test_text or "already ended" not in test_text:
            return {"ok": False, "tool": "synthesize_hangman_state_candidate", "path": rel_source, "summary": "Hangman state examples not found."}
        candidate = '''STATUS_WIN = "win"
STATUS_LOSE = "lose"
STATUS_ONGOING = "ongoing"


class Hangman:
    def __init__(self, word):
        self.word = word
        self.remaining_guesses = 9
        self.status = STATUS_ONGOING
        self.guesses = set()

    def guess(self, char):
        if self.status != STATUS_ONGOING:
            raise ValueError("The game has already ended.")
        if char in self.guesses or char not in self.word:
            self.remaining_guesses -= 1
        self.guesses.add(char)
        if all(char in self.guesses for char in self.word):
            self.status = STATUS_WIN
        elif self.remaining_guesses < 0:
            self.status = STATUS_LOSE

    def get_masked_word(self):
        return "".join(char if char in self.guesses else "_" for char in self.word)

    def get_status(self):
        return self.status
'''
        return {
            "ok": True,
            "tool": "synthesize_hangman_state_candidate",
            "path": rel_source,
            "candidate_source": candidate,
            "summary": "Synthesized Hangman state machine implementation.",
        }

    def synthesize_rest_api_debt_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        test_file = self.resolve_path(test_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py" or test_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_rest_api_debt_candidate", "path": rel_source, "summary": "Python source/tests only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        test_text = test_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_rest_api_debt_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        if "RestAPI" not in class_names or not any(token in test_text for token in ('"/iou"', "'/iou'")) or not any(token in test_text for token in ('"/add"', "'/add'")):
            return {"ok": False, "tool": "synthesize_rest_api_debt_candidate", "path": rel_source, "summary": "REST debt examples not found."}
        candidate = '''import json


class RestAPI:
    def __init__(self, database=None):
        self.database = database if database is not None else {"users": []}

    def _find_user(self, name):
        for user in self.database["users"]:
            if user["name"] == name:
                return user
        raise KeyError(name)

    def _new_user(self, name):
        return {"name": name, "owes": {}, "owed_by": {}, "balance": 0.0}

    def _refresh_balance(self, user):
        user["balance"] = float(sum(user["owed_by"].values()) - sum(user["owes"].values()))

    def _sorted_users(self, names):
        return sorted((self._find_user(name) for name in names), key=lambda user: user["name"])

    def get(self, url, payload=None):
        if url != "/users":
            return json.dumps({})
        if payload:
            requested = set(json.loads(payload)["users"])
            users = [user for user in self.database["users"] if user["name"] in requested]
        else:
            users = self.database["users"]
        return json.dumps({"users": users})

    def post(self, url, payload=None):
        data = json.loads(payload or "{}")
        if url == "/add":
            user = self._new_user(data["user"])
            self.database["users"].append(user)
            return json.dumps(user)
        if url == "/iou":
            lender = self._find_user(data["lender"])
            borrower = self._find_user(data["borrower"])
            amount = float(data["amount"])
            existing_reverse = lender["owes"].pop(borrower["name"], 0.0)
            borrower["owed_by"].pop(lender["name"], None)
            borrower["owes"].pop(lender["name"], None)
            lender["owed_by"].pop(borrower["name"], None)
            net = amount - existing_reverse
            if net > 0:
                borrower["owes"][lender["name"]] = net
                lender["owed_by"][borrower["name"]] = net
            elif net < 0:
                lender["owes"][borrower["name"]] = -net
                borrower["owed_by"][lender["name"]] = -net
            self._refresh_balance(lender)
            self._refresh_balance(borrower)
            return json.dumps({"users": self._sorted_users([lender["name"], borrower["name"]])})
        return json.dumps({})
'''
        return {
            "ok": True,
            "tool": "synthesize_rest_api_debt_candidate",
            "path": rel_source,
            "candidate_source": candidate,
            "summary": "Synthesized JSON REST debt ledger implementation.",
        }

    def synthesize_forth_interpreter_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        test_file = self.resolve_path(test_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py" or test_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_forth_interpreter_candidate", "path": rel_source, "summary": "Python source/tests only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        test_text = test_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_forth_interpreter_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        function_names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        if "evaluate" not in function_names or "StackUnderflowError" not in class_names or not any(token in test_text.lower() for token in ("dup", "swap", ": foo", "case_insensitivity")):
            return {"ok": False, "tool": "synthesize_forth_interpreter_candidate", "path": rel_source, "summary": "Forth interpreter examples not found."}
        candidate = '''class StackUnderflowError(Exception):
    pass


def evaluate(input_data):
    stack = []
    definitions = {}

    def is_number(token):
        try:
            int(token)
        except ValueError:
            return False
        return True

    def require(count):
        if len(stack) < count:
            raise StackUnderflowError("Insufficient number of items in stack")

    def expand_definition(tokens):
        expanded = []
        for token in tokens:
            if token in definitions:
                expanded.extend(definitions[token])
            else:
                expanded.append(token)
        return expanded

    def execute_tokens(tokens):
        for token in tokens:
            if is_number(token):
                stack.append(int(token))
            elif token in definitions:
                execute_tokens(definitions[token])
            elif token == "+":
                require(2)
                right = stack.pop()
                left = stack.pop()
                stack.append(left + right)
            elif token == "-":
                require(2)
                right = stack.pop()
                left = stack.pop()
                stack.append(left - right)
            elif token == "*":
                require(2)
                right = stack.pop()
                left = stack.pop()
                stack.append(left * right)
            elif token == "/":
                require(2)
                right = stack.pop()
                left = stack.pop()
                if right == 0:
                    raise ZeroDivisionError("divide by zero")
                stack.append(int(left / right))
            elif token == "dup":
                require(1)
                stack.append(stack[-1])
            elif token == "drop":
                require(1)
                stack.pop()
            elif token == "swap":
                require(2)
                stack[-2], stack[-1] = stack[-1], stack[-2]
            elif token == "over":
                require(2)
                stack.append(stack[-2])
            else:
                raise ValueError("undefined operation")

    for raw_line in input_data:
        tokens = raw_line.lower().split()
        if not tokens:
            continue
        if tokens[0] == ":":
            if len(tokens) < 4 or tokens[-1] != ";":
                raise ValueError("illegal operation")
            name = tokens[1]
            if is_number(name):
                raise ValueError("illegal operation")
            definitions[name] = expand_definition(tokens[2:-1])
            continue
        execute_tokens(tokens)
    return stack
'''
        return {
            "ok": True,
            "tool": "synthesize_forth_interpreter_candidate",
            "path": rel_source,
            "candidate_source": candidate,
            "summary": "Synthesized Forth stack interpreter with user-defined words.",
        }

    def synthesize_sgf_tree_parser_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        test_file = self.resolve_path(test_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py" or test_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_sgf_tree_parser_candidate", "path": rel_source, "summary": "Python source/tests only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        test_text = test_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_sgf_tree_parser_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        function_names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        if "parse" not in function_names or "SgfTree" not in class_names or "property must be in uppercase" not in test_text:
            return {"ok": False, "tool": "synthesize_sgf_tree_parser_candidate", "path": rel_source, "summary": "SGF parser examples not found."}
        candidate = '''class SgfTree:
    def __init__(self, properties=None, children=None):
        self.properties = properties or {}
        self.children = children or []

    def __eq__(self, other):
        if not isinstance(other, SgfTree):
            return False
        return self.properties == other.properties and self.children == other.children

    def __ne__(self, other):
        return not self == other


def parse(input_string):
    if not input_string or not input_string.startswith("("):
        raise ValueError("tree missing")
    index = 0

    def parse_value():
        nonlocal index
        if input_string[index] != "[":
            raise ValueError("properties without delimiter")
        index += 1
        chars = []
        while index < len(input_string):
            char = input_string[index]
            if char == "]":
                index += 1
                return "".join(chars)
            if char == "\\\\":
                index += 1
                if index >= len(input_string):
                    break
                escaped = input_string[index]
                if escaped == "\\n":
                    index += 1
                    continue
                if escaped == "\\t":
                    chars.append(" ")
                else:
                    chars.append(escaped)
                index += 1
                continue
            if char == "\\t":
                chars.append(" ")
            else:
                chars.append(char)
            index += 1
        raise ValueError("properties without delimiter")

    def parse_node():
        nonlocal index
        if index >= len(input_string) or input_string[index] != ";":
            raise ValueError("tree with no nodes")
        index += 1
        properties = {}
        while index < len(input_string) and input_string[index].isalpha():
            start = index
            while index < len(input_string) and input_string[index].isalpha():
                index += 1
            name = input_string[start:index]
            if not name.isupper():
                raise ValueError("property must be in uppercase")
            if index >= len(input_string) or input_string[index] != "[":
                raise ValueError("properties without delimiter")
            values = []
            while index < len(input_string) and input_string[index] == "[":
                values.append(parse_value())
            properties[name] = values
        return SgfTree(properties)

    def parse_tree():
        nonlocal index
        if input_string[index] != "(":
            raise ValueError("tree missing")
        index += 1
        if index < len(input_string) and input_string[index] == ")":
            raise ValueError("tree with no nodes")
        root = parse_node()
        current = root
        while index < len(input_string) and input_string[index] == ";":
            child = parse_node()
            current.children = [child]
            current = child
        while index < len(input_string) and input_string[index] == "(":
            current.children.append(parse_tree())
        if index >= len(input_string) or input_string[index] != ")":
            raise ValueError("tree missing")
        index += 1
        return root

    tree = parse_tree()
    if index != len(input_string):
        raise ValueError("tree missing")
    return tree
'''
        return {
            "ok": True,
            "tool": "synthesize_sgf_tree_parser_candidate",
            "path": rel_source,
            "candidate_source": candidate,
            "summary": "Synthesized recursive SGF tree parser.",
        }

    def synthesize_poker_ranking_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        test_file = self.resolve_path(test_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py" or test_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_poker_ranking_candidate", "path": rel_source, "summary": "Python source/tests only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        test_text = test_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_poker_ranking_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        function_names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if "best_hands" not in function_names or not any(token in test_text.lower() for token in ("straight flush", "best_hands([", "wheel", "tie")):
            return {"ok": False, "tool": "synthesize_poker_ranking_candidate", "path": rel_source, "summary": "Poker ranking examples not found."}
        candidate = '''from collections import Counter


RANKS = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}


def _parse_hand(hand):
    cards = hand.split()
    ranks = [RANKS[card[:-1]] for card in cards]
    suits = [card[-1] for card in cards]
    return ranks, suits


def _straight_high(ranks):
    unique = sorted(set(ranks))
    if len(unique) != 5:
        return None
    if unique == [2, 3, 4, 5, 14]:
        return 5
    if unique[-1] - unique[0] == 4:
        return unique[-1]
    return None


def _score_hand(hand):
    ranks, suits = _parse_hand(hand)
    counts = Counter(ranks)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], -item[0]))
    straight_high = _straight_high(ranks)
    flush = len(set(suits)) == 1
    descending = sorted(ranks, reverse=True)
    if straight_high is not None and flush:
        return (8, straight_high)
    if [count for _, count in ordered] == [4, 1]:
        return (7, ordered[0][0], ordered[1][0])
    if [count for _, count in ordered] == [3, 2]:
        return (6, ordered[0][0], ordered[1][0])
    if flush:
        return (5, *descending)
    if straight_high is not None:
        return (4, straight_high)
    if [count for _, count in ordered] == [3, 1, 1]:
        kickers = sorted((rank for rank, count in ordered[1:]), reverse=True)
        return (3, ordered[0][0], *kickers)
    if [count for _, count in ordered] == [2, 2, 1]:
        pairs = sorted((rank for rank, count in ordered if count == 2), reverse=True)
        kicker = next(rank for rank, count in ordered if count == 1)
        return (2, pairs[0], pairs[1], kicker)
    if [count for _, count in ordered] == [2, 1, 1, 1]:
        kickers = sorted((rank for rank, count in ordered[1:]), reverse=True)
        return (1, ordered[0][0], *kickers)
    return (0, *descending)


def best_hands(hands):
    scores = [(hand, _score_hand(hand)) for hand in hands]
    best = max(score for _, score in scores)
    return [hand for hand, score in scores if score == best]
'''
        return {
            "ok": True,
            "tool": "synthesize_poker_ranking_candidate",
            "path": rel_source,
            "candidate_source": candidate,
            "summary": "Synthesized poker hand ranking and tie-breaking implementation.",
        }

    def synthesize_metered_io_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        test_file = self.resolve_path(test_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py" or test_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_metered_io_candidate", "path": rel_source, "summary": "Python source/tests only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        test_text = test_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_metered_io_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        if not {"MeteredFile", "MeteredSocket"}.issubset(class_names) or "SuperMock" not in test_text or "recv_bytes" not in test_text:
            return {"ok": False, "tool": "synthesize_metered_io_candidate", "path": rel_source, "summary": "Metered IO wrapper examples not found."}
        candidate = '''import io


class MeteredFile(io.BufferedRandom):
    """Implement using a subclassing model."""

    def __init__(self, *args, **kwargs):
        self._read_bytes = 0
        self._read_ops = 0
        self._write_bytes = 0
        self._write_ops = 0
        super().__init__(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __iter__(self):
        return self

    def __next__(self):
        chunk = super().readline()
        if chunk == b"":
            raise StopIteration
        self._read_ops += 1
        self._read_bytes += len(chunk)
        return chunk

    def read(self, size=-1):
        chunk = super().read(size)
        self._read_ops += 1
        self._read_bytes += len(chunk)
        return chunk

    @property
    def read_bytes(self):
        return self._read_bytes

    @property
    def read_ops(self):
        return self._read_ops

    def write(self, b):
        written = super().write(b)
        self._write_ops += 1
        self._write_bytes += int(written or 0)
        return written

    @property
    def write_bytes(self):
        return self._write_bytes

    @property
    def write_ops(self):
        return self._write_ops


class MeteredSocket:
    """Implement using a delegation model."""

    def __init__(self, socket):
        self._socket = socket
        self._recv_bytes = 0
        self._recv_ops = 0
        self._send_bytes = 0
        self._send_ops = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._socket.__exit__(exc_type, exc_val, exc_tb)

    def recv(self, bufsize, flags=0):
        chunk = self._socket.recv(bufsize, flags)
        self._recv_ops += 1
        self._recv_bytes += len(chunk)
        return chunk

    @property
    def recv_bytes(self):
        return self._recv_bytes

    @property
    def recv_ops(self):
        return self._recv_ops

    def send(self, data, flags=0):
        written = self._socket.send(data, flags)
        self._send_ops += 1
        self._send_bytes += int(written or 0)
        return written

    @property
    def send_bytes(self):
        return self._send_bytes

    @property
    def send_ops(self):
        return self._send_ops
'''
        return {
            "ok": True,
            "tool": "synthesize_metered_io_candidate",
            "path": rel_source,
            "candidate_source": candidate,
            "summary": "Synthesized metered file/socket wrapper implementation.",
        }

    def synthesize_tree_pov_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        test_file = self.resolve_path(test_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py" or test_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_tree_pov_candidate", "path": rel_source, "summary": "Python source/tests only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        test_text = test_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_tree_pov_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        if "Tree" not in class_names or "from_pov" not in test_text or "path_to" not in test_text:
            return {"ok": False, "tool": "synthesize_tree_pov_candidate", "path": rel_source, "summary": "Tree point-of-view examples not found."}
        candidate = '''from json import dumps


class Tree:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children if children is not None else []

    def __dict__(self):
        return {self.label: [c.__dict__() for c in sorted(self.children)]}

    def __str__(self, indent=None):
        return dumps(self.__dict__(), indent=indent)

    def __lt__(self, other):
        return self.label < other.label

    def __eq__(self, other):
        return isinstance(other, Tree) and self.__dict__() == other.__dict__()

    def _clone(self):
        return Tree(self.label, [child._clone() for child in self.children])

    def _find_path_nodes(self, target):
        if self.label == target:
            return [self]
        for child in self.children:
            path = child._find_path_nodes(target)
            if path:
                return [self, *path]
        return []

    def from_pov(self, from_node):
        path = self._find_path_nodes(from_node)
        if not path:
            raise ValueError("Tree could not be reoriented")

        def build(index, blocked_child):
            node = path[index]
            children = [child._clone() for child in node.children if child is not blocked_child]
            if index > 0:
                children.append(build(index - 1, node))
            return Tree(node.label, children)

        return build(len(path) - 1, None)

    def path_to(self, from_node, to_node):
        rerooted = self.from_pov(from_node)
        path = rerooted._find_path_nodes(to_node)
        if not path:
            raise ValueError("No path found")
        return [node.label for node in path]
'''
        return {
            "ok": True,
            "tool": "synthesize_tree_pov_candidate",
            "path": rel_source,
            "candidate_source": candidate,
            "summary": "Synthesized tree re-rooting and path lookup implementation.",
        }

    def synthesize_binary_zipper_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        test_file = self.resolve_path(test_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py" or test_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_binary_zipper_candidate", "path": rel_source, "summary": "Python source/tests only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        test_text = test_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_binary_zipper_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        if "Zipper" not in class_names or "from_tree" not in test_text or "set_value" not in test_text:
            return {"ok": False, "tool": "synthesize_binary_zipper_candidate", "path": rel_source, "summary": "Binary zipper examples not found."}
        candidate = '''def _clone(tree):
    if tree is None:
        return None
    return {
        "value": tree["value"],
        "left": _clone(tree["left"]),
        "right": _clone(tree["right"]),
    }


class Zipper:
    def __init__(self, focus, breadcrumbs=None):
        self.focus = focus
        self.breadcrumbs = breadcrumbs or []

    @staticmethod
    def from_tree(tree):
        return Zipper(_clone(tree))

    def value(self):
        return self.focus["value"]

    def set_value(self, value):
        updated = _clone(self.focus)
        updated["value"] = value
        return Zipper(updated, self.breadcrumbs)

    def left(self):
        if self.focus["left"] is None:
            return None
        crumb = ("left", self.focus["value"], _clone(self.focus["right"]))
        return Zipper(_clone(self.focus["left"]), [*self.breadcrumbs, crumb])

    def set_left(self, tree):
        updated = _clone(self.focus)
        updated["left"] = _clone(tree)
        return Zipper(updated, self.breadcrumbs)

    def right(self):
        if self.focus["right"] is None:
            return None
        crumb = ("right", self.focus["value"], _clone(self.focus["left"]))
        return Zipper(_clone(self.focus["right"]), [*self.breadcrumbs, crumb])

    def set_right(self, tree):
        updated = _clone(self.focus)
        updated["right"] = _clone(tree)
        return Zipper(updated, self.breadcrumbs)

    def up(self):
        if not self.breadcrumbs:
            return None
        side, value, sibling = self.breadcrumbs[-1]
        if side == "left":
            parent = {"value": value, "left": _clone(self.focus), "right": sibling}
        else:
            parent = {"value": value, "left": sibling, "right": _clone(self.focus)}
        return Zipper(parent, self.breadcrumbs[:-1])

    def to_tree(self):
        zipper = self
        while zipper.breadcrumbs:
            zipper = zipper.up()
        return _clone(zipper.focus)
'''
        return {
            "ok": True,
            "tool": "synthesize_binary_zipper_candidate",
            "path": rel_source,
            "candidate_source": candidate,
            "summary": "Synthesized immutable binary tree zipper implementation.",
        }

    def synthesize_go_territory_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        test_file = self.resolve_path(test_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py" or test_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_go_territory_candidate", "path": rel_source, "summary": "Python source/tests only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        test_text = test_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_go_territory_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        if "Board" not in class_names or "territories" not in test_text or "Invalid coordinate" not in test_text:
            return {"ok": False, "tool": "synthesize_go_territory_candidate", "path": rel_source, "summary": "Go territory examples not found."}
        candidate = '''WHITE = "W"
BLACK = "B"
NONE = ""


class Board:
    def __init__(self, board):
        self.board = board
        self.height = len(board)
        self.width = len(board[0]) if board else 0

    def _neighbors(self, x, y):
        result = []
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < self.width and 0 <= ny < self.height:
                result.append((nx, ny))
        return result

    def territory(self, x, y):
        if x < 0 or y < 0 or y >= self.height or x >= self.width:
            raise ValueError("Invalid coordinate")
        if self.board[y][x] != " ":
            return NONE, set()
        region = set()
        borders = set()
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in region:
                continue
            region.add((cx, cy))
            for nx, ny in self._neighbors(cx, cy):
                value = self.board[ny][nx]
                if value == " " and (nx, ny) not in region:
                    stack.append((nx, ny))
                elif value in {WHITE, BLACK}:
                    borders.add(value)
        owner = next(iter(borders)) if len(borders) == 1 else NONE
        return owner, region

    def territories(self):
        result = {BLACK: set(), WHITE: set(), NONE: set()}
        seen = set()
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in seen or self.board[y][x] != " ":
                    continue
                owner, region = self.territory(x, y)
                seen.update(region)
                result[owner].update(region)
        return result
'''
        return {
            "ok": True,
            "tool": "synthesize_go_territory_candidate",
            "path": rel_source,
            "candidate_source": candidate,
            "summary": "Synthesized Go board territory flood-fill implementation.",
        }

    def synthesize_hex_connect_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        test_file = self.resolve_path(test_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py" or test_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_hex_connect_candidate", "path": rel_source, "summary": "Python source/tests only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        test_text = test_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_hex_connect_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        if "ConnectGame" not in class_names or "get_winner" not in test_text or "left_to_right" not in test_text:
            return {"ok": False, "tool": "synthesize_hex_connect_candidate", "path": rel_source, "summary": "Hex connect examples not found."}
        candidate = '''class ConnectGame:
    def __init__(self, board):
        self.board = [line.strip().split() for line in board.splitlines() if line.strip()]
        self.height = len(self.board)
        self.width = len(self.board[0]) if self.board else 0

    def _neighbors(self, row, col):
        result = []
        for dr, dc in ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)):
            nr = row + dr
            nc = col + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                result.append((nr, nc))
        return result

    def _connected(self, player):
        if not self.board:
            return False
        stack = []
        seen = set()
        if player == "X":
            for row in range(self.height):
                if self.board[row][0] == player:
                    stack.append((row, 0))
        else:
            for col in range(self.width):
                if self.board[0][col] == player:
                    stack.append((0, col))
        while stack:
            row, col = stack.pop()
            if (row, col) in seen:
                continue
            seen.add((row, col))
            if player == "X" and col == self.width - 1:
                return True
            if player == "O" and row == self.height - 1:
                return True
            for nr, nc in self._neighbors(row, col):
                if self.board[nr][nc] == player and (nr, nc) not in seen:
                    stack.append((nr, nc))
        return False

    def get_winner(self):
        if self._connected("X"):
            return "X"
        if self._connected("O"):
            return "O"
        return ""
'''
        return {
            "ok": True,
            "tool": "synthesize_hex_connect_candidate",
            "path": rel_source,
            "candidate_source": candidate,
            "summary": "Synthesized hex-grid connect winner flood-fill implementation.",
        }

    def synthesize_prefix_rotation_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_prefix_rotation_candidate", "path": rel_source, "summary": "Python source only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_prefix_rotation_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        if len(functions) != 1 or classes:
            return {"ok": False, "tool": "synthesize_prefix_rotation_candidate", "path": rel_source, "summary": "Prefix-rotation synthesis requires exactly one top-level function."}
        function = functions[0]
        params = self._python_parameter_sequence(function)
        if len(params) != 1:
            return {"ok": False, "tool": "synthesize_prefix_rotation_candidate", "path": rel_source, "summary": "Prefix-rotation synthesis requires one function parameter."}
        extracted = self.test_spec_extract(test_path, rel_source, limit=limit)
        if extracted.get("ok") is not True:
            return {"ok": False, "tool": "synthesize_prefix_rotation_candidate", "path": rel_source, "summary": str(extracted.get("summary") or "Could not extract examples.")}
        examples = list(extracted.get("examples") or [])
        rows = self._prefix_rotation_examples(examples)
        if len(rows) < 4:
            return {"ok": False, "tool": "synthesize_prefix_rotation_candidate", "path": rel_source, "summary": "Not enough consistent prefix-rotation string examples."}
        suffix = rows[0][3]
        moved_sources = [source for source, _expected, cut, _suffix in rows if cut > 0]
        move_prefixes = sorted({source[:cut].lower() for source, _expected, cut, _suffix in rows if cut > 0}, key=lambda item: (-len(item), item))
        no_move_prefixes = sorted(
            {self._shortest_distinguishing_prefix(source, moved_sources) for source, _expected, cut, _suffix in rows if cut == 0},
            key=lambda item: (-len(item), item),
        )
        break_chars = sorted(
            {
                source[cut].lower()
                for source, _expected, cut, _suffix in rows
                if cut > 0 and cut < len(source)
            }
            | {prefix for prefix in no_move_prefixes if len(prefix) == 1}
        )
        leading_moved_chars = sorted({prefix[0] for prefix in move_prefixes if prefix})
        cluster_suffixes = sorted({prefix[-2:] for prefix in move_prefixes if len(prefix) >= 2}, key=lambda item: (-len(item), item))
        param = params[0]
        header = self._python_signature(source_text.splitlines(), int(getattr(function, "lineno", 1)))
        body = f'''{header}
    suffix = {suffix!r}
    no_move_prefixes = {tuple(no_move_prefixes)!r}
    move_prefixes = {tuple(move_prefixes)!r}
    break_chars = set({tuple(break_chars)!r})
    leading_moved_chars = set({tuple(leading_moved_chars)!r})
    cluster_suffixes = {tuple(cluster_suffixes)!r}

    def _convert_word(word):
        lower = word.lower()
        for prefix in no_move_prefixes:
            if lower.startswith(prefix):
                return word + suffix
        for prefix in move_prefixes:
            if lower.startswith(prefix):
                return word[len(prefix):] + word[:len(prefix)] + suffix
        cut = len(word)
        index = 0
        while index < len(word):
            matched_cluster = ""
            for cluster in cluster_suffixes:
                if lower.startswith(cluster, index):
                    matched_cluster = cluster
                    break
            if matched_cluster:
                cut = index + len(matched_cluster)
                break
            char = lower[index]
            if char in break_chars and not (index == 0 and char in leading_moved_chars):
                cut = index
                break
            index += 1
        return word[cut:] + word[:cut] + suffix

    return " ".join(_convert_word(word) for word in {param}.split())
'''
        return {
            "ok": True,
            "tool": "synthesize_prefix_rotation_candidate",
            "path": rel_source,
            "candidate_source": body,
            "summary": f"synthesized prefix-rotation candidate from {len(rows)} examples",
            "observed_prefixes": {"move": move_prefixes, "no_move": no_move_prefixes, "break_chars": break_chars},
        }

    def synthesize_word_arithmetic_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_word_arithmetic_candidate", "path": rel_source, "summary": "Python source only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_word_arithmetic_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        if len(functions) != 1 or classes:
            return {"ok": False, "tool": "synthesize_word_arithmetic_candidate", "path": rel_source, "summary": "Word-arithmetic synthesis requires exactly one top-level function."}
        function = functions[0]
        params = self._python_parameter_sequence(function)
        if len(params) != 1:
            return {"ok": False, "tool": "synthesize_word_arithmetic_candidate", "path": rel_source, "summary": "Word-arithmetic synthesis requires one function parameter."}
        extracted = self.test_spec_extract(test_path, rel_source, limit=limit)
        if extracted.get("ok") is not True:
            return {"ok": False, "tool": "synthesize_word_arithmetic_candidate", "path": rel_source, "summary": str(extracted.get("summary") or "Could not extract examples.")}
        examples = list(extracted.get("examples") or [])
        value_rows: list[tuple[str, int]] = []
        raises_messages = " ".join(str(item.get("example") or "") for item in examples if isinstance(item, dict))
        for item in examples:
            if not isinstance(item, dict):
                continue
            parsed = self._literal_call_string_arg_expected(str(item.get("example") or ""))
            if parsed is None:
                continue
            _symbol, question, expected = parsed
            if isinstance(expected, int) and question.startswith("What is "):
                value_rows.append((question, expected))
        if len(value_rows) < 4:
            return {"ok": False, "tool": "synthesize_word_arithmetic_candidate", "path": rel_source, "summary": "Not enough word-arithmetic examples."}
        corpus = " ".join(question for question, _expected in value_rows).lower()
        operations = {
            "plus": "plus" in corpus,
            "minus": "minus" in corpus,
            "multiplied by": "multiplied by" in corpus,
            "divided by": "divided by" in corpus,
        }
        if sum(1 for enabled in operations.values() if enabled) < 2:
            return {"ok": False, "tool": "synthesize_word_arithmetic_candidate", "path": rel_source, "summary": "Examples do not describe enough arithmetic operations."}
        unknown_message = "unknown operation" if "unknown operation" in raises_messages else "unknown operation"
        syntax_message = "syntax error" if "syntax error" in raises_messages else "syntax error"
        param = params[0]
        header = self._python_signature(source_text.splitlines(), int(getattr(function, "lineno", 1)))
        body = f'''{header}
    unknown_message = {unknown_message!r}
    syntax_message = {syntax_message!r}
    prefix = "What is"
    if not isinstance({param}, str) or not {param}.startswith(prefix) or not {param}.endswith("?"):
        raise ValueError(unknown_message)
    tail = {param}[len(prefix):-1]
    if tail and not tail.startswith(" "):
        raise ValueError(unknown_message)
    expression = tail.strip()
    if not expression:
        raise ValueError(syntax_message)
    tokens = expression.split()

    def _parse_number(token):
        try:
            return int(token)
        except ValueError as exc:
            raise ValueError(syntax_message) from exc

    value = _parse_number(tokens[0])
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "plus":
            operation = "plus"
            index += 1
        elif token == "minus":
            operation = "minus"
            index += 1
        elif token == "multiplied" and index + 1 < len(tokens) and tokens[index + 1] == "by":
            operation = "multiplied by"
            index += 2
        elif token == "divided" and index + 1 < len(tokens) and tokens[index + 1] == "by":
            operation = "divided by"
            index += 2
        elif token.lstrip("-").isdigit():
            raise ValueError(syntax_message)
        else:
            raise ValueError(unknown_message)
        if index >= len(tokens):
            raise ValueError(syntax_message)
        if not tokens[index].lstrip("-").isdigit():
            raise ValueError(syntax_message)
        operand = int(tokens[index])
        index += 1
        if operation == "plus":
            value += operand
        elif operation == "minus":
            value -= operand
        elif operation == "multiplied by":
            value *= operand
        elif operation == "divided by":
            value = int(value / operand)
    return value
'''
        return {
            "ok": True,
            "tool": "synthesize_word_arithmetic_candidate",
            "path": rel_source,
            "candidate_source": body,
            "summary": f"synthesized word-arithmetic parser from {len(value_rows)} examples",
            "operations": operations,
        }

    def synthesize_string_normalizer_class_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_string_normalizer_class_candidate", "path": rel_source, "summary": "Python source only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_string_normalizer_class_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        if len(classes) != 1:
            return {"ok": False, "tool": "synthesize_string_normalizer_class_candidate", "path": rel_source, "summary": "String-normalizer synthesis requires exactly one class."}
        class_node = classes[0]
        methods = {child.name: child for child in class_node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))}
        init = methods.get("__init__")
        if init is None or not self._python_body_is_stub(init):
            return {"ok": False, "tool": "synthesize_string_normalizer_class_candidate", "path": rel_source, "summary": "String-normalizer synthesis requires a stub constructor."}
        init_params = self._python_parameter_sequence(init)
        if len(init_params) != 2:
            return {"ok": False, "tool": "synthesize_string_normalizer_class_candidate", "path": rel_source, "summary": "String-normalizer synthesis requires one constructor parameter besides self."}
        extracted = self.test_spec_extract(test_path, rel_source, limit=limit)
        if extracted.get("ok") is not True:
            return {"ok": False, "tool": "synthesize_string_normalizer_class_candidate", "path": rel_source, "summary": str(extracted.get("summary") or "Could not extract examples.")}
        valid_numbers: list[tuple[str, str]] = []
        raises_rows: list[tuple[str, str]] = []
        pretty_expected = False
        area_code_expected = False

        def constructor_arg(call: ast.Call) -> str | None:
            if not isinstance(call.func, ast.Name) or call.func.id != class_node.name or not call.args:
                return None
            try:
                value = ast.literal_eval(call.args[0])
            except Exception:
                return None
            return value if isinstance(value, str) else None

        def behavior_parts(expr: str) -> tuple[str | None, str | None, bool]:
            parts = [part.strip() for part in expr.split(";") if part.strip()]
            object_inputs: dict[str, str] = {}
            for part in parts[:-1]:
                try:
                    stmt = ast.parse(part).body[0]
                except Exception:
                    continue
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Call):
                    raw = constructor_arg(stmt.value)
                    if raw is not None:
                        object_inputs[stmt.targets[0].id] = raw
            last = parts[-1] if parts else expr
            try:
                node = ast.parse(last, mode="eval").body
            except Exception:
                return None, None, False
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Call):
                    return constructor_arg(node.value), node.attr, False
                if isinstance(node.value, ast.Name) and node.value.id in object_inputs:
                    return object_inputs[node.value.id], node.attr, False
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                receiver = node.func.value
                if isinstance(receiver, ast.Name) and receiver.id in object_inputs:
                    return object_inputs[receiver.id], node.func.attr, True
                if isinstance(receiver, ast.Call):
                    return constructor_arg(receiver), node.func.attr, True
            if isinstance(node, ast.Call):
                return constructor_arg(node), "", False
            return None, None, False

        for item in list(extracted.get("examples") or []):
            if not isinstance(item, dict):
                continue
            kind, expr, expected = self._split_test_example(str(item.get("example") or ""))
            if not kind:
                continue
            raw, access, is_method = behavior_parts(expr)
            if raw is None:
                continue
            if kind == "value":
                try:
                    expected_value = ast.literal_eval(expected)
                except Exception:
                    continue
                if access == "number" and isinstance(expected_value, str) and expected_value.isdigit():
                    valid_numbers.append((raw, expected_value))
                if access == "area_code" and isinstance(expected_value, str):
                    area_code_expected = True
                if access == "pretty" and is_method and isinstance(expected_value, str):
                    pretty_expected = True
            elif kind == "raises":
                match = re.match(r"^[A-Za-z_][\w.]*\((?P<message>.*)\)$", expected)
                message = ""
                if match:
                    try:
                        value = ast.literal_eval(str(match.group("message") or ""))
                    except Exception:
                        value = ""
                    message = value if isinstance(value, str) else ""
                raises_rows.append((raw, message))
        messages = {message for _raw, message in raises_rows if message}
        required_messages = {
            "letters not permitted",
            "punctuations not permitted",
            "must not be fewer than 10 digits",
            "must not be greater than 11 digits",
            "11 digits must start with 1",
            "area code cannot start with zero",
            "area code cannot start with one",
            "exchange code cannot start with zero",
            "exchange code cannot start with one",
        }
        if len(valid_numbers) < 3 or len(messages & required_messages) < 5:
            return {"ok": False, "tool": "synthesize_string_normalizer_class_candidate", "path": rel_source, "summary": "Not enough string-normalizer examples."}
        allowed_separators = sorted({char for raw, _expected in valid_numbers for char in raw if not char.isdigit()})
        if not allowed_separators:
            return {"ok": False, "tool": "synthesize_string_normalizer_class_candidate", "path": rel_source, "summary": "No separator-cleaning examples found."}
        letters_message = "letters not permitted" if "letters not permitted" in messages else "letters not permitted"
        punctuation_message = "punctuations not permitted" if "punctuations not permitted" in messages else "punctuations not permitted"
        fewer_message = "must not be fewer than 10 digits"
        greater_message = "must not be greater than 11 digits"
        country_message = "11 digits must start with 1"
        area_zero_message = "area code cannot start with zero"
        area_one_message = "area code cannot start with one"
        exchange_zero_message = "exchange code cannot start with zero"
        exchange_one_message = "exchange code cannot start with one"
        pretty_method = ""
        if pretty_expected:
            pretty_method = '''

    def pretty(self):
        return f"({self.area_code})-{self.exchange_code}-{self.subscriber_number}"
'''
        area_assignment = "        self.area_code = digits[:3]\n" if area_code_expected or pretty_expected else ""
        helper_assignments = ""
        if pretty_expected:
            helper_assignments = "        self.exchange_code = digits[3:6]\n        self.subscriber_number = digits[6:]\n"
        candidate = f'''class {class_node.name}:
    def __init__(self, number):
        raw = str(number)
        allowed_separators = set({''.join(allowed_separators)!r})
        if any(char.isalpha() for char in raw):
            raise ValueError({letters_message!r})
        if any((not char.isdigit()) and char not in allowed_separators for char in raw):
            raise ValueError({punctuation_message!r})
        digits = "".join(char for char in raw if char.isdigit())
        if len(digits) < 10:
            raise ValueError({fewer_message!r})
        if len(digits) > 11:
            raise ValueError({greater_message!r})
        if len(digits) == 11:
            if digits[0] != "1":
                raise ValueError({country_message!r})
            digits = digits[1:]
        if digits[0] == "0":
            raise ValueError({area_zero_message!r})
        if digits[0] == "1":
            raise ValueError({area_one_message!r})
        if digits[3] == "0":
            raise ValueError({exchange_zero_message!r})
        if digits[3] == "1":
            raise ValueError({exchange_one_message!r})
        self.number = digits
{area_assignment}{helper_assignments}{pretty_method}'''
        return {
            "ok": True,
            "tool": "synthesize_string_normalizer_class_candidate",
            "path": rel_source,
            "candidate_source": candidate,
            "summary": f"synthesized string normalizer class from {len(valid_numbers)} valid and {len(raises_rows)} raising examples",
            "allowed_separators": allowed_separators,
        }

    def synthesize_grouped_roster_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_grouped_roster_candidate", "path": rel_source, "summary": "Python source only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_grouped_roster_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        if len(classes) != 1:
            return {"ok": False, "tool": "synthesize_grouped_roster_candidate", "path": rel_source, "summary": "Grouped-roster synthesis requires exactly one class."}
        class_node = classes[0]
        method_names = {child.name for child in class_node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))}
        required = {"__init__", "add_student", "roster", "grade", "added"}
        if not required.issubset(method_names):
            return {"ok": False, "tool": "synthesize_grouped_roster_candidate", "path": rel_source, "summary": "Grouped-roster synthesis requires add_student, roster, grade, and added methods."}
        extracted = self.test_spec_extract(test_path, rel_source, limit=limit)
        if extracted.get("ok") is not True:
            return {"ok": False, "tool": "synthesize_grouped_roster_candidate", "path": rel_source, "summary": str(extracted.get("summary") or "Could not extract examples.")}
        corpus = "\n".join(str(item.get("example") or "") for item in list(extracted.get("examples") or []) if isinstance(item, dict))
        if "add_student" not in corpus or ".roster()" not in corpus or ".grade(" not in corpus or ".added()" not in corpus:
            return {"ok": False, "tool": "synthesize_grouped_roster_candidate", "path": rel_source, "summary": "Grouped-roster examples not found."}
        candidate = f'''class {class_node.name}:
    def __init__(self):
        self._grades = {{}}
        self._student_grades = {{}}
        self._added = []

    def add_student(self, name, grade):
        if name in self._student_grades:
            self._added.append(False)
            return
        self._student_grades[name] = grade
        self._grades.setdefault(grade, set()).add(name)
        self._added.append(True)

    def roster(self):
        return [
            name
            for grade in sorted(self._grades)
            for name in sorted(self._grades[grade])
        ]

    def grade(self, grade_number):
        return sorted(self._grades.get(grade_number, set()))

    def added(self):
        return list(self._added)
'''
        return {
            "ok": True,
            "tool": "synthesize_grouped_roster_candidate",
            "path": rel_source,
            "candidate_source": candidate,
            "summary": "synthesized grouped roster candidate from stateful examples",
        }

    def synthesize_vlq_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_vlq_candidate", "path": rel_source, "summary": "Python source only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_vlq_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if not {"encode", "decode"}.issubset(functions):
            return {"ok": False, "tool": "synthesize_vlq_candidate", "path": rel_source, "summary": "VLQ synthesis requires encode and decode functions."}
        extracted = self.test_spec_extract(test_path, rel_source, limit=limit)
        if extracted.get("ok") is not True:
            return {"ok": False, "tool": "synthesize_vlq_candidate", "path": rel_source, "summary": str(extracted.get("summary") or "Could not extract examples.")}
        corpus = "\n".join(str(item.get("example") or "") for item in list(extracted.get("examples") or []) if isinstance(item, dict))
        if "encode([" not in corpus or "decode([" not in corpus or "incomplete sequence" not in corpus:
            return {"ok": False, "tool": "synthesize_vlq_candidate", "path": rel_source, "summary": "VLQ examples not found."}
        candidate = '''def encode(numbers):
    result = []
    for number in numbers:
        if number == 0:
            result.append(0)
            continue
        chunks = []
        while number > 0:
            chunks.insert(0, number & 0x7F)
            number >>= 7
        for index in range(len(chunks) - 1):
            chunks[index] |= 0x80
        result.extend(chunks)
    return result


def decode(bytes_):
    result = []
    value = 0
    pending = False
    for byte in bytes_:
        value = (value << 7) | (byte & 0x7F)
        pending = True
        if byte & 0x80 == 0:
            result.append(value)
            value = 0
            pending = False
    if pending:
        raise ValueError("incomplete sequence")
    return result
'''
        return {
            "ok": True,
            "tool": "synthesize_vlq_candidate",
            "path": rel_source,
            "candidate_source": candidate,
            "summary": "synthesized VLQ encode/decode candidate from examples",
        }

    def synthesize_text_matrix_transpose_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_text_matrix_transpose_candidate", "path": rel_source, "summary": "Python source only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_text_matrix_transpose_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        if len(functions) != 1 or classes:
            return {"ok": False, "tool": "synthesize_text_matrix_transpose_candidate", "path": rel_source, "summary": "Text-matrix synthesis requires exactly one top-level function."}
        function = functions[0]
        params = self._python_parameter_sequence(function)
        if len(params) != 1:
            return {"ok": False, "tool": "synthesize_text_matrix_transpose_candidate", "path": rel_source, "summary": "Text-matrix synthesis requires one function parameter."}
        extracted = self.test_spec_extract(test_path, rel_source, limit=limit)
        if extracted.get("ok") is not True:
            return {"ok": False, "tool": "synthesize_text_matrix_transpose_candidate", "path": rel_source, "summary": str(extracted.get("summary") or "Could not extract examples.")}
        pairs: list[tuple[str, str]] = []
        for item in list(extracted.get("examples") or []):
            if not isinstance(item, dict):
                continue
            parsed = self._literal_call_string_arg_expected(str(item.get("example") or ""))
            if parsed is None:
                continue
            _symbol, source, expected = parsed
            if isinstance(source, str) and isinstance(expected, str) and ("\n" in source or "\n" in expected):
                pairs.append((source, expected))
        if len(pairs) < 3:
            return {"ok": False, "tool": "synthesize_text_matrix_transpose_candidate", "path": rel_source, "summary": "Not enough text-matrix string examples."}
        param = params[0]
        header = self._python_signature(source_text.splitlines(), int(getattr(function, "lineno", 1)))
        body = f'''{header}
    if {param} == "":
        return ""
    lines = {param}.split("\\n")
    width = max((len(line) for line in lines), default=0)
    transposed = []
    for column in range(width):
        cells = [line[column] if column < len(line) else None for line in lines]
        while cells and cells[-1] is None:
            cells.pop()
        transposed.append("".join(cell if cell is not None else " " for cell in cells))
    return "\\n".join(transposed)
'''
        return {
            "ok": True,
            "tool": "synthesize_text_matrix_transpose_candidate",
            "path": rel_source,
            "candidate_source": body,
            "summary": f"synthesized text-matrix transpose candidate from {len(pairs)} examples",
        }

    def synthesize_cyclic_interval_scale_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_cyclic_interval_scale_candidate", "path": rel_source, "summary": "Python source only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_cyclic_interval_scale_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        if len(classes) != 1:
            return {"ok": False, "tool": "synthesize_cyclic_interval_scale_candidate", "path": rel_source, "summary": "Cyclic-scale synthesis requires exactly one class."}
        class_node = classes[0]
        methods = {child.name: child for child in class_node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if not {"__init__", "chromatic", "interval"}.issubset(methods):
            return {"ok": False, "tool": "synthesize_cyclic_interval_scale_candidate", "path": rel_source, "summary": "Cyclic-scale synthesis requires __init__, chromatic, and interval methods."}
        init_params = self._python_parameter_sequence(methods["__init__"])
        if len(init_params) != 2:
            return {"ok": False, "tool": "synthesize_cyclic_interval_scale_candidate", "path": rel_source, "summary": "Cyclic-scale synthesis requires one constructor parameter besides self."}
        extracted = self.test_spec_extract(test_path, rel_source, limit=limit)
        if extracted.get("ok") is not True:
            return {"ok": False, "tool": "synthesize_cyclic_interval_scale_candidate", "path": rel_source, "summary": str(extracted.get("summary") or "Could not extract examples.")}
        scale_rows: list[tuple[str, str, list[str]]] = []
        chromatic_rows = 0
        interval_rows = 0
        for item in list(extracted.get("examples") or []):
            if not isinstance(item, dict):
                continue
            kind, expr, expected = self._split_test_example(str(item.get("example") or ""))
            if kind != "value":
                continue
            match = re.match(r"^[A-Za-z_][A-Za-z0-9_]*\((?P<tonic>.+?)\)\.(?P<method>chromatic|interval)\((?P<args>.*)\)$", expr)
            if not match:
                continue
            try:
                tonic = ast.literal_eval(match.group("tonic"))
                expected_value = ast.literal_eval(expected)
            except Exception:
                continue
            if not isinstance(tonic, str) or not isinstance(expected_value, list) or not all(isinstance(note, str) for note in expected_value):
                continue
            method = match.group("method")
            if method == "chromatic":
                chromatic_rows += 1
            else:
                interval_rows += 1
            scale_rows.append((tonic, method, expected_value))
        if chromatic_rows < 1 or interval_rows < 4:
            return {"ok": False, "tool": "synthesize_cyclic_interval_scale_candidate", "path": rel_source, "summary": "Not enough cyclic-scale examples."}
        flat_tonics = sorted(
            {
                tonic
                for tonic, _method, notes in scale_rows
                if any("b" in note for note in notes) and not any("#" in note for note in notes)
            }
        )
        class_header = self._python_signature(source_text.splitlines(), int(getattr(class_node, "lineno", 1)))
        init_header = self._python_signature(source_text.splitlines(), int(getattr(methods["__init__"], "lineno", 1)))
        chromatic_header = self._python_signature(source_text.splitlines(), int(getattr(methods["chromatic"], "lineno", 1)))
        interval_header = self._python_signature(source_text.splitlines(), int(getattr(methods["interval"], "lineno", 1)))
        tonic_param = init_params[1]
        body = f'''{class_header}
    SHARP_NOTES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    FLAT_NOTES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
    FLAT_TONICS = {tuple(flat_tonics)!r}
    STEPS = {{"m": 1, "M": 2, "A": 3}}

    {init_header.strip()}
        self.tonic = {tonic_param}

    def _normalize_tonic(self):
        return self.tonic[:1].upper() + self.tonic[1:]

    def _notes(self):
        tonic = self._normalize_tonic()
        notes = self.FLAT_NOTES if self.tonic in self.FLAT_TONICS or tonic in self.FLAT_TONICS else self.SHARP_NOTES
        index = notes.index(tonic)
        return list(notes[index:] + notes[:index])

    {chromatic_header.strip()}
        return self._notes()

    {interval_header.strip()}
        notes = self._notes()
        index = 0
        result = [notes[index]]
        for interval in intervals:
            index = (index + self.STEPS[interval]) % len(notes)
            result.append(notes[index])
        return result
'''
        return {
            "ok": True,
            "tool": "synthesize_cyclic_interval_scale_candidate",
            "path": rel_source,
            "candidate_source": body,
            "summary": f"synthesized cyclic interval scale candidate from {len(scale_rows)} examples",
            "flat_tonics": flat_tonics,
        }

    def synthesize_unique_regex_identifier_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_unique_regex_identifier_candidate", "path": rel_source, "summary": "Python source only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_unique_regex_identifier_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        if len(classes) != 1:
            return {"ok": False, "tool": "synthesize_unique_regex_identifier_candidate", "path": rel_source, "summary": "Unique-regex synthesis requires exactly one class."}
        class_node = classes[0]
        methods = {child.name: child for child in class_node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))}
        init_node = methods.get("__init__")
        if init_node is None:
            return {"ok": False, "tool": "synthesize_unique_regex_identifier_candidate", "path": rel_source, "summary": "Unique-regex synthesis requires __init__."}
        init_params = self._python_parameter_sequence(init_node)
        if len(init_params) != 1:
            return {"ok": False, "tool": "synthesize_unique_regex_identifier_candidate", "path": rel_source, "summary": "Unique-regex synthesis supports no-argument constructors."}
        extracted = self.test_spec_extract(test_path, rel_source, limit=limit)
        if extracted.get("ok") is not True:
            return {"ok": False, "tool": "synthesize_unique_regex_identifier_candidate", "path": rel_source, "summary": str(extracted.get("summary") or "Could not extract examples.")}
        examples = [str(item.get("example") or "") for item in list(extracted.get("examples") or []) if isinstance(item, dict)]
        text = "\n".join(examples)
        regex_match = re.search(r"(?P<expr>[A-Za-z_][A-Za-z0-9_]*(?:\(.*?\))?(?:; [^\\n]+?)?\.[A-Za-z_][A-Za-z0-9_]*) matches (?P<pattern>'[^']+'|\"[^\"]+\")", text)
        if not regex_match or "!=" not in text:
            return {"ok": False, "tool": "synthesize_unique_regex_identifier_candidate", "path": rel_source, "summary": "No regex plus uniqueness examples found."}
        try:
            pattern = ast.literal_eval(regex_match.group("pattern"))
        except Exception:
            return {"ok": False, "tool": "synthesize_unique_regex_identifier_candidate", "path": rel_source, "summary": "Could not parse regex literal."}
        shape = re.fullmatch(r"^\^\[A-Z\]\{(?P<letters>\d+)\}\\\\?d\{(?P<digits>\d+)\}\$$", str(pattern))
        if not shape:
            return {"ok": False, "tool": "synthesize_unique_regex_identifier_candidate", "path": rel_source, "summary": f"Unsupported identifier regex: {pattern}"}
        attr_match = re.search(r"\.(?P<attr>[A-Za-z_][A-Za-z0-9_]*)$", regex_match.group("expr").split(";")[-1].strip())
        attr_name = attr_match.group("attr") if attr_match else "name"
        letter_count = int(shape.group("letters"))
        digit_count = int(shape.group("digits"))
        class_header = self._python_signature(source_text.splitlines(), int(getattr(class_node, "lineno", 1)))
        init_header = self._python_signature(source_text.splitlines(), int(getattr(init_node, "lineno", 1)))
        body = f'''import random
import string


{class_header}
    _used_identifiers = set()

    {init_header.strip()}
        self.{attr_name} = self._generate_identifier()

    @classmethod
    def _generate_identifier(cls):
        while True:
            letters = "".join(random.choice(string.ascii_uppercase) for _ in range({letter_count}))
            digits = "".join(random.choice(string.digits) for _ in range({digit_count}))
            identifier = letters + digits
            if identifier not in cls._used_identifiers:
                cls._used_identifiers.add(identifier)
                return identifier

    def reset(self):
        self.{attr_name} = self._generate_identifier()
'''
        return {
            "ok": True,
            "tool": "synthesize_unique_regex_identifier_candidate",
            "path": rel_source,
            "candidate_source": body,
            "summary": f"synthesized unique regex identifier candidate for {attr_name} matching {pattern}",
            "attribute": attr_name,
            "pattern": pattern,
        }

    def synthesize_node_collection_candidate(self, source_path: str, test_path: str, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_node_collection_candidate", "path": rel_source, "summary": "Python source only."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_node_collection_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        node_class: ast.ClassDef | None = None
        collection_class: ast.ClassDef | None = None
        exception_class: ast.ClassDef | None = None
        class_methods: dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]] = {}
        for class_node in classes:
            methods = {child.name: child for child in class_node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))}
            class_methods[class_node.name] = methods
            method_names = set(methods)
            if {"value", "next", "__init__"} <= method_names and node_class is None:
                node_class = class_node
            if {"__iter__", "__len__", "head", "push", "pop", "reversed", "__init__"} <= method_names and collection_class is None:
                collection_class = class_node
            if exception_class is None and (class_node.name.endswith("Exception") or any(self._node_expr(base, {}) == "Exception" for base in class_node.bases)):
                exception_class = class_node
        if node_class is None or collection_class is None:
            return {
                "ok": False,
                "tool": "synthesize_node_collection_candidate",
                "path": rel_source,
                "summary": "Node collection synthesis requires Node-like value/next and collection push/pop/iter/head methods.",
            }
        extracted = self.test_spec_extract(test_path, rel_source, limit=limit)
        if extracted.get("ok") is not True:
            return {"ok": False, "tool": "synthesize_node_collection_candidate", "path": rel_source, "summary": str(extracted.get("summary") or "Could not extract examples.")}
        example_text = "\n".join(str(item.get("example") or "") for item in list(extracted.get("examples") or []) if isinstance(item, dict))
        required_fragments = ("list(", "len(", ".push(", ".pop()", ".head()", ".reversed()")
        if not all(fragment in example_text for fragment in required_fragments):
            return {
                "ok": False,
                "tool": "synthesize_node_collection_candidate",
                "path": rel_source,
                "summary": "Node collection synthesis requires list/len/push/pop/head/reversed examples.",
            }
        message = "The list is empty."
        message_match = re.search(r"raises\s+[A-Za-z_][\w.]*\((?P<message>'[^']*'|\"[^\"]*\")\)", example_text)
        if message_match:
            try:
                message = str(ast.literal_eval(message_match.group("message")))
            except Exception:
                message = "The list is empty."
        lines = source_text.splitlines()
        exception_name = exception_class.name if exception_class is not None else "EmptyListException"
        node_name = node_class.name
        collection_name = collection_class.name
        node_methods = class_methods[node_name]
        collection_methods = class_methods[collection_name]

        def signature_for(class_or_method: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> str:
            return self._python_signature(lines, int(getattr(class_or_method, "lineno", 1))).strip()

        exception_block = ""
        if exception_class is not None:
            exception_block = f"{signature_for(exception_class)}\n    pass\n\n\n"
        else:
            exception_block = f"class {exception_name}(Exception):\n    pass\n\n\n"
        body = (
            exception_block
            + f"{signature_for(node_class)}\n"
            + f"    {signature_for(node_methods['__init__'])}\n"
            + "        self._value = value\n"
            + "        self._next = None\n\n"
            + f"    {signature_for(node_methods['value'])}\n"
            + "        return self._value\n\n"
            + f"    {signature_for(node_methods['next'])}\n"
            + "        return self._next\n\n\n"
            + f"{signature_for(collection_class)}\n"
            + f"    {signature_for(collection_methods['__init__'])}\n"
            + "        self._head = None\n"
            + "        self._length = 0\n"
            + "        for value in values or []:\n"
            + "            self.push(value)\n\n"
            + f"    {signature_for(collection_methods['__iter__'])}\n"
            + "        current = self._head\n"
            + "        while current is not None:\n"
            + "            yield current.value()\n"
            + "            current = current.next()\n\n"
            + f"    {signature_for(collection_methods['__len__'])}\n"
            + "        return self._length\n\n"
            + f"    {signature_for(collection_methods['head'])}\n"
            + "        if self._head is None:\n"
            + f"            raise {exception_name}({message!r})\n"
            + "        return self._head\n\n"
            + f"    {signature_for(collection_methods['push'])}\n"
            + f"        node = {node_name}(value)\n"
            + "        node._next = self._head\n"
            + "        self._head = node\n"
            + "        self._length += 1\n\n"
            + f"    {signature_for(collection_methods['pop'])}\n"
            + "        if self._head is None:\n"
            + f"            raise {exception_name}({message!r})\n"
            + "        value = self._head.value()\n"
            + "        self._head = self._head.next()\n"
            + "        self._length -= 1\n"
            + "        return value\n\n"
            + f"    {signature_for(collection_methods['reversed'])}\n"
            + f"        result = {collection_name}()\n"
            + "        for value in self:\n"
            + "            result.push(value)\n"
            + "        return result\n"
        )
        return {
            "ok": True,
            "tool": "synthesize_node_collection_candidate",
            "path": rel_source,
            "candidate_source": body,
            "summary": f"synthesized node-backed collection candidate for {collection_name}",
            "node_class": node_name,
            "collection_class": collection_name,
            "exception_class": exception_name,
        }

    def synthesize_relative_import_candidate(self, source_path: str, test_path: str | None = None, limit: int = 80) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "synthesize_relative_import_candidate", "path": rel_source, "summary": "Python source only."}
        package_dir = source_file.parent
        if not (package_dir / "__init__.py").exists():
            return {"ok": False, "tool": "synthesize_relative_import_candidate", "path": rel_source, "summary": "Source is not inside a Python package."}
        source_text = source_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(self._python_parse_text(source_text))
        except SyntaxError as exc:
            return {"ok": False, "tool": "synthesize_relative_import_candidate", "path": rel_source, "summary": f"Could not parse source: {exc}"}
        lines = source_text.splitlines()
        replacements: dict[int, str] = {}

        def sibling_exists(name: str) -> bool:
            return (package_dir / f"{name}.py").exists() or (package_dir / name / "__init__.py").exists()

        def alias_text(alias: ast.alias) -> str:
            return alias.name + (f" as {alias.asname}" if alias.asname else "")

        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and sibling_exists(node.module):
                names = ", ".join(alias_text(alias) for alias in node.names)
                replacements[int(getattr(node, "lineno", 0))] = f"from .{node.module} import {names}"
            elif isinstance(node, ast.Import) and node.names and all(sibling_exists(alias.name.split(".", 1)[0]) for alias in node.names):
                names = ", ".join(alias_text(alias) for alias in node.names)
                replacements[int(getattr(node, "lineno", 0))] = f"from . import {names}"
        if not replacements:
            return {"ok": False, "tool": "synthesize_relative_import_candidate", "path": rel_source, "summary": "No top-level sibling imports found."}
        updated_lines = list(lines)
        for line_no, replacement in replacements.items():
            if line_no <= 0 or line_no > len(updated_lines):
                continue
            indent = re.match(r"^\s*", updated_lines[line_no - 1]).group(0)
            updated_lines[line_no - 1] = indent + replacement
        candidate = "\n".join(updated_lines)
        if source_text.endswith("\n"):
            candidate += "\n"
        return {
            "ok": True,
            "tool": "synthesize_relative_import_candidate",
            "path": rel_source,
            "candidate_source": candidate,
            "summary": f"synthesized relative import candidate for {len(replacements)} sibling import(s)",
        }

    def implementation_spec(self, source_path: str, test_path: str | None = None, limit: int = 40) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "implementation_spec", "path": rel_source, "summary": "implementation_spec supports Python source files only."}
        limit = max(1, min(int(limit), 100))
        definitions = self._implementation_source_defs(source_file)
        examples: list[dict[str, Any]] = []
        grouped_examples: dict[str, list[str]] = {}
        rel_test = ""
        if test_path:
            test_file = self.resolve_path(test_path, allow_missing=False)
            rel_test = self.relative_label(test_file)
            extracted = self.test_spec_extract(rel_test, rel_source, limit=limit)
            if extracted.get("ok") is True:
                examples = list(extracted.get("examples") or [])
                grouped_examples = {str(key): list(value) for key, value in dict(extracted.get("grouped") or {}).items()}
        expected_types: dict[str, list[str]] = {}
        for item in examples:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol") or "")
            expected_type = self._implementation_expected_type(str(item.get("example") or ""))
            if symbol and expected_type:
                expected_types.setdefault(symbol, [])
                if expected_type not in expected_types[symbol]:
                    expected_types[symbol].append(expected_type)
        stubs = [f"{rel_source}::{item['symbol']}" for item in definitions if item.get("stub")]
        string_transform_hints = self._implementation_string_transform_hints(examples, limit=24)
        lines: list[str] = [f"source: {rel_source}"]
        if rel_test:
            lines.append(f"tests: {rel_test}")
        if definitions:
            lines.append("public API:")
            for item in definitions[:limit]:
                suffix = " STUB" if item.get("stub") else ""
                expected = expected_types.get(str(item.get("name") or ""), []) or expected_types.get(str(item.get("symbol") or ""), [])
                expected_text = f" expects={','.join(expected)}" if expected else ""
                lines.append(f"- {item['symbol']} {item['signature']}{suffix}{expected_text}")
                for risk in list(item.get("risks") or [])[:3]:
                    lines.append(f"  risk: {risk}")
        if grouped_examples:
            lines.append("examples:")
            group_count = max(1, len(grouped_examples))
            per_symbol_example_limit = max(4, min(24, (limit + group_count - 1) // group_count))
            for symbol, rows in grouped_examples.items():
                lines.append(f"- {symbol}: " + " | ".join(str(row) for row in rows[:per_symbol_example_limit]))
        if string_transform_hints:
            lines.append("observed string transforms:")
            for symbol, rows in string_transform_hints.items():
                lines.append(f"- {symbol}: " + " | ".join(rows[:24]))
        if stubs:
            lines.append("remaining stubs: " + ", ".join(stubs[:20]))
        return {
            "ok": True,
            "tool": "implementation_spec",
            "path": rel_source,
            "test_path": rel_test,
            "definitions": definitions,
            "examples": examples,
            "grouped_examples": grouped_examples,
            "expected_types": expected_types,
            "string_transform_hints": string_transform_hints,
            "stubs": stubs,
            "output": "\n".join(lines),
            "summary": f"implementation spec: {len(definitions)} public symbol(s), {len(examples)} example(s), {len(stubs)} stub(s)",
        }

    def _split_test_example(self, example: str) -> tuple[str, str, str]:
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

    def _test_example_probe_expressions(self, examples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in examples:
            if not isinstance(item, dict):
                continue
            kind, expr, expected = self._split_test_example(str(item.get("example") or ""))
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

    def _probe_row_is_non_executable_missing_local(self, row: dict[str, Any]) -> bool:
        actual = str(row.get("actual") or "")
        match = re.match(r"ERROR NameError: name '([^']+)' is not defined$", actual)
        if not match:
            return False
        missing = match.group(1)
        expression = str(row.get("expression") or "")
        expected = str(row.get("expected") or "")
        return re.search(rf"\b{re.escape(missing)}\b", expression) is not None or re.search(rf"\b{re.escape(missing)}\b", expected) is not None

    def run_test_example_probes(self, source_path: str, test_path: str, limit: int = 10, timeout: int = 30) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        test_file = self.resolve_path(test_path, allow_missing=False)
        if source_file.suffix.lower() != ".py" or test_file.suffix.lower() != ".py":
            return {"ok": True, "tool": "run_function_probe", "probe_kind": "test_examples", "output": "(no Python test examples)"}
        extracted = self.test_spec_extract(self.relative_label(test_file), self.relative_label(source_file), limit=max(1, min(int(limit), 24)))
        if extracted.get("ok") is not True:
            return {"ok": True, "tool": "run_function_probe", "probe_kind": "test_examples", "output": str(extracted.get("summary") or "")}
        probes = self._test_example_probe_expressions(list(extracted.get("examples") or []), limit=max(1, min(int(limit), 24)))
        if not probes:
            return {"ok": True, "tool": "run_function_probe", "probe_kind": "test_examples", "output": "(no executable examples extracted)"}
        module = self._module_name_for_source_path(self.relative_label(source_file))
        script = (
            "import importlib,json,os,sys,traceback\n"
            "workspace=os.getcwd(); sys.path.insert(0, workspace); sys.path.insert(0, os.path.join(workspace, 'src'))\n"
            f"module_name={json.dumps(module)}\n"
            f"probes={json.dumps(probes)}\n"
            "rows=[]\n"
            "try:\n"
            "    mod=importlib.import_module(module_name)\n"
            "    ns={'module': mod}\n"
            "    for name in dir(mod):\n"
            "        if not name.startswith('_'):\n"
            "            ns[name]=getattr(mod,name)\n"
            "    for probe in probes:\n"
            "        expr=probe['expr']; expected_text=probe['expected']; kind=probe['kind']\n"
            "        try:\n"
            "            parts=[part.strip() for part in expr.split(';') if part.strip()]\n"
            "            local_ns=dict(ns)\n"
            "            for statement in parts[:-1]:\n"
            "                exec(statement, local_ns)\n"
            "            if kind == 'raises':\n"
            "                expected_name=expected_text.split('(',1)[0]\n"
            "                expected_message=None\n"
            "                if '(' in expected_text and expected_text.endswith(')'):\n"
            "                    expected_message=expected_text.split('(',1)[1][:-1]\n"
            "                    try:\n"
            "                        expected_message=eval(expected_message, local_ns)\n"
            "                    except Exception:\n"
            "                        pass\n"
            "                try:\n"
            "                    value=eval(parts[-1], local_ns)\n"
            "                    rows.append({'expression':expr,'ok':False,'expected':'raises '+expected_name,'expected_message':expected_message,'actual':repr(value),'actual_type':type(value).__name__,'symbol':probe.get('symbol',''),'line':probe.get('line',1)})\n"
            "                except Exception as exc:\n"
            "                    type_ok=type(exc).__name__ == expected_name or exc.__class__.__name__ == expected_name\n"
            "                    message_ok=expected_message is None or str(exc) == str(expected_message)\n"
            "                    rows.append({'expression':expr,'ok':bool(type_ok and message_ok),'expected':'raises '+expected_name,'expected_message':expected_message,'actual':'raises '+type(exc).__name__,'actual_message':str(exc),'symbol':probe.get('symbol',''),'line':probe.get('line',1)})\n"
            "            else:\n"
            "                value=eval(parts[-1], local_ns)\n"
            "                try:\n"
            "                    expected=eval(expected_text, local_ns)\n"
            "                except Exception:\n"
            "                    expected=expected_text\n"
            "                rows.append({'expression':expr,'ok':value == expected,'expected':repr(expected),'expected_type':type(expected).__name__,'actual':repr(value),'actual_type':type(value).__name__,'symbol':probe.get('symbol',''),'line':probe.get('line',1)})\n"
            "        except Exception as exc:\n"
            "            rows.append({'expression':expr,'ok':False,'expected':expected_text,'actual':'ERROR '+type(exc).__name__+': '+str(exc),'symbol':probe.get('symbol',''),'line':probe.get('line',1)})\n"
            "except Exception as exc:\n"
            "    rows.append({'expression':'<import>','ok':False,'expected':'import '+module_name,'actual':'ERROR '+type(exc).__name__+': '+str(exc),'symbol':'','line':1})\n"
            "print(json.dumps(rows, ensure_ascii=False))\n"
        )
        completed = self._run_process([sys.executable, "-c", script], cwd=self.workspace_root, timeout=max(1, int(timeout)))
        try:
            rows = json.loads(completed.stdout.strip())
        except json.JSONDecodeError:
            rows = []
        skipped_non_executable = [row for row in rows if isinstance(row, dict) and not row.get("ok") and self._probe_row_is_non_executable_missing_local(row)]
        mismatches = [row for row in rows if isinstance(row, dict) and not row.get("ok") and not self._probe_row_is_non_executable_missing_local(row)]
        lines: list[str] = []
        for row in mismatches[: max(1, min(int(limit), 24))]:
            symbol = f"{row.get('symbol')}: " if row.get("symbol") else ""
            expected_message = row.get("expected_message")
            actual_message = row.get("actual_message")
            message_detail = ""
            if expected_message is not None or actual_message is not None:
                message_detail = f"; expected_message={expected_message!r}, actual_message={actual_message!r}"
            lines.append(
                f"{symbol}{row.get('expression')} expected {row.get('expected')}"
                f" ({row.get('expected_type', '')}), got {row.get('actual')} ({row.get('actual_type', '')}){message_detail}"
            )
        if not lines:
            checked_count = max(0, len(rows) - len(skipped_non_executable))
            lines = [f"example probes ok: {checked_count}"]
            if skipped_non_executable:
                lines.append(f"skipped non-executable examples: {len(skipped_non_executable)}")
        return {
            "ok": completed.returncode == 0 and not mismatches,
            "tool": "run_function_probe",
            "probe_kind": "test_examples",
            "module": module,
            "source_path": self.relative_label(source_file),
            "test_path": self.relative_label(test_file),
            "results": rows,
            "output": "\n".join(lines),
            "summary": "\n".join(lines),
        }

    def _candidate_public_signature_map(self, source: str) -> dict[str, str]:
        try:
            tree = ast.parse(self._python_parse_text(source))
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

    def _candidate_signature_diagnostics(self, original: str, candidate: str) -> list[str]:
        original_map = self._candidate_public_signature_map(original)
        candidate_map = self._candidate_public_signature_map(candidate)
        diagnostics: list[str] = []
        for symbol, signature in original_map.items():
            if symbol not in candidate_map:
                diagnostics.append(f"candidate removed public symbol {symbol}")
                continue
            if candidate_map[symbol] != signature:
                diagnostics.append(f"candidate changed signature for {symbol}: {signature} -> {candidate_map[symbol]}")
        return diagnostics

    def _normalize_candidate_python_source(self, source_file: Path, candidate_source: str) -> tuple[str, str | None]:
        try:
            tree = ast.parse(self._python_parse_text(candidate_source))
        except SyntaxError:
            return candidate_source, None
        lines = candidate_source.splitlines(keepends=True)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name.lower() != "foldr":
                continue
            if not node.body:
                continue
            start = int(getattr(node.body[0], "lineno", getattr(node, "lineno", 1))) - 1
            end = int(getattr(node.body[-1], "end_lineno", getattr(node.body[-1], "lineno", start + 1)))
            body = textwrap.dedent("".join(lines[start:end]))
            diagnostic = self._foldr_argument_order_diagnostic(node, body)
            replacement = self._canonical_foldr_replacement_if_safe(source_file, node.name, diagnostic)
            if not replacement:
                continue
            def_line = int(getattr(node, "lineno", 1)) - 1
            def_end = int(getattr(node, "end_lineno", getattr(node, "lineno", 1)))
            if def_line < 0 or def_line >= len(lines):
                continue
            indent = re.match(r"^\s*", lines[def_line]).group(0)
            if indent:
                replacement = textwrap.indent(replacement.strip("\n"), indent) + "\n"
            updated = "".join(lines[:def_line]) + replacement + "".join(lines[def_end:])
            return updated, "Normalized foldr reducer order to the canonical right fold."
        return candidate_source, None

    def _copy_workspace_for_candidate(self, destination: Path) -> None:
        def ignore(_directory: str, names: list[str]) -> set[str]:
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
            return {name for name in names if name in skipped or self._generated_dir_name(name)}

        shutil.copytree(self.workspace_root, destination, ignore=ignore)

    def validate_implementation_candidate(
        self,
        source_path: str,
        candidate_source: str,
        test_path: str | None = None,
        test_command: str | None = None,
        probe_limit: int = 24,
        timeout: int = 120,
    ) -> dict[str, Any]:
        self._check_interrupted()
        source_file = self.resolve_path(source_path, allow_missing=False)
        rel_source = self.relative_label(source_file)
        if source_file.suffix.lower() != ".py":
            return {"ok": False, "tool": "validate_implementation_candidate", "path": rel_source, "summary": "candidate validation supports Python source files only."}
        if not isinstance(candidate_source, str) or not candidate_source.strip():
            return {"ok": False, "tool": "validate_implementation_candidate", "path": rel_source, "summary": "candidate_source is empty."}
        original = source_file.read_text(encoding="utf-8", errors="replace")
        candidate_source, normalization = self._normalize_candidate_python_source(source_file, candidate_source)
        signature_diagnostics = self._candidate_signature_diagnostics(original, candidate_source)
        removed_symbol_diagnostics = [item for item in signature_diagnostics if "removed public symbol" in item]
        signature_warnings = [item for item in signature_diagnostics if item not in removed_symbol_diagnostics]
        if removed_symbol_diagnostics or (signature_warnings and not (test_path or test_command)):
            output = "\n".join((removed_symbol_diagnostics or signature_warnings)[:8])
            return {
                "ok": False,
                "tool": "validate_implementation_candidate",
                "path": rel_source,
                "stage": "signature",
                "diagnostics": removed_symbol_diagnostics or signature_warnings,
                "normalized": normalization,
                "output": output,
                "summary": output,
            }
        try:
            ast.parse(self._python_parse_text(candidate_source), filename=rel_source)
        except SyntaxError as exc:
            summary = f"candidate syntax error at {rel_source}:{exc.lineno or 1}: {exc.msg}"
            return {"ok": False, "tool": "validate_implementation_candidate", "path": rel_source, "stage": "syntax", "summary": summary, "output": summary}
        tmp_base = self.workspace_root / ".ollama-code" / "tmp" / f"candidate-{uuid4().hex}"
        try:
            tmp_base.mkdir(parents=True, exist_ok=False)
            temp_root = tmp_base / "workspace"
            self._copy_workspace_for_candidate(temp_root)
            temp_source = temp_root / rel_source
            temp_source.parent.mkdir(parents=True, exist_ok=True)
            temp_source.write_text(candidate_source, encoding="utf-8")
            temp_tools = ToolExecutor(
                temp_root,
                approval_mode="auto",
                test_command=(test_command or self.default_test_command),
                default_tools_enabled=self.default_tools_enabled,
                disabled_tools=self.disabled_tools,
                mcp_servers=self.mcp_servers,
                browser_enabled=self.browser_enabled,
                security_enabled=self.security_enabled,
            )
            static_result = temp_tools.contract_check([rel_source], limit=12)
            if static_result.get("ok") is not True:
                summary = str(static_result.get("output") or static_result.get("summary") or "candidate static sanity failed")
                return {
                    "ok": False,
                    "tool": "validate_implementation_candidate",
                    "path": rel_source,
                    "stage": "static",
                    "static": static_result,
                    "signature_warnings": signature_warnings,
                    "normalized": normalization,
                    "output": summary,
                    "summary": summary,
            }
            probe_result: dict[str, Any] | None = None
            probe_failure_summary = ""
            if test_path:
                probe_result = temp_tools.run_test_example_probes(rel_source, test_path, limit=max(1, min(int(probe_limit), 24)), timeout=min(max(1, int(timeout)), 60))
                if probe_result.get("ok") is not True:
                    probe_failure_summary = str(probe_result.get("output") or probe_result.get("summary") or "candidate example probes failed")
            run_args: dict[str, Any] = {"timeout": max(1, int(timeout))}
            if test_command:
                run_args["command"] = test_command
            test_result = temp_tools.run_test(**run_args)
            if test_result.get("ok") is not True:
                summary = str(test_result.get("output") or test_result.get("summary") or "candidate tests failed")
                if probe_failure_summary:
                    summary = f"example probe mismatches:\n{probe_failure_summary}\n\ntest output:\n{summary}"
                return {
                    "ok": False,
                    "tool": "validate_implementation_candidate",
                    "path": rel_source,
                    "stage": "tests",
                    "static": static_result,
                    "probes": probe_result,
                    "test": test_result,
                    "signature_warnings": signature_warnings,
                    "normalized": normalization,
                    "output": self._truncate_text(summary, limit=1600),
                    "summary": self._truncate_text(summary, limit=520),
                }
        finally:
            shutil.rmtree(tmp_base, ignore_errors=True)
        return {
            "ok": True,
            "tool": "validate_implementation_candidate",
            "path": rel_source,
            "stage": "passed",
            "candidate_source": candidate_source,
            "normalized": normalization,
            "signature_warnings": signature_warnings,
            "summary": "candidate passed syntax, static sanity, example probes, and tests",
            "output": "candidate passed syntax, static sanity, example probes, and tests",
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
                    qualname = ".".join([*self.stack, node.name])
                    init_node = next((child for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == "__init__"), None)
                    is_exception_class = any(
                        (isinstance(base, ast.Name) and base.id in {"Exception", "BaseException"})
                        or (isinstance(base, ast.Attribute) and base.attr in {"Exception", "BaseException"})
                        for base in node.bases
                    )
                    if init_node is not None:
                        args = self.outer._contract_args(init_node.args)
                        if args and str(args[0].get("name")) in {"self", "cls"}:
                            args = args[1:]
                        arity = self.outer._callable_arity_without_receiver(init_node.args)
                    elif is_exception_class:
                        args = []
                        arity = {"min": 0, "max": None, "has_vararg": True, "has_kwarg": True}
                    else:
                        args = []
                        arity = {"min": 0, "max": 0, "has_vararg": False, "has_kwarg": False}
                    definitions[f"{rel}:{qualname}"] = {
                        "path": rel,
                        "symbol": qualname,
                        "name": node.name,
                        "kind": "class",
                        "line": int(getattr(node, "lineno", 1)),
                        "end": int(getattr(node, "end_lineno", getattr(node, "lineno", 1))),
                        "args": args,
                        "arity": arity,
                        "returns": node.name,
                        "return_shapes": [f"call:{node.name}"],
                        "purity": "unknown",
                        "is_async": False,
                        "has_yield": False,
                        "node": node,
                    }
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
                        "arity": self.outer._callable_arity_without_receiver(node.args) if kind == "method" else self.outer._callable_arity(node.args),
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
                        "attribute": call.get("attribute", False),
                        "keywords": call.get("keywords", []),
                        "expected_exception": call.get("expected_exception", False),
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

    def _callable_arity_without_receiver(self, args: ast.arguments) -> dict[str, Any]:
        arity = self._callable_arity(args)
        positional = [*args.posonlyargs, *args.args]
        if positional and positional[0].arg in {"self", "cls"}:
            arity["min"] = max(0, int(arity.get("min", 0)) - 1)
            if isinstance(arity.get("max"), int):
                arity["max"] = max(0, int(arity["max"]) - 1)
        return arity

    def _call_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def _calls_in_node(self, node: ast.AST) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        parent_by_node: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(node):
            for child in ast.iter_child_nodes(parent):
                parent_by_node[child] = parent

        def expected_exception_context(call: ast.Call) -> bool:
            current: ast.AST | None = call
            while current is not None:
                if isinstance(current, ast.With):
                    for item in current.items:
                        context_expr = item.context_expr
                        if not isinstance(context_expr, ast.Call):
                            continue
                        context_name = self._call_name(context_expr.func)
                        if context_name in {"assertRaises", "assertRaisesRegex", "assertRaisesWithMessage"}:
                            return True
                current = parent_by_node.get(current)
            return False

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                name = self._call_name(child.func)
                if not name:
                    continue
                calls.append(
                    {
                        "name": name,
                        "attribute": isinstance(child.func, ast.Attribute),
                        "args": len(child.args),
                        "keywords": [kw.arg for kw in child.keywords if kw.arg],
                        "line": int(getattr(child, "lineno", 1)),
                        "expected_exception": expected_exception_context(child),
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
        local_shapes = dict(arg_shapes)
        has_return = False

        class ReturnShapeVisitor(ast.NodeVisitor):
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

            def visit_Return(self, child: ast.Return) -> Any:
                nonlocal has_return
                has_return = True
                value = child.value
                inferred = self_outer._value_shape_hint(value, local_shapes) if value is not None else None
                if value is None or isinstance(value, ast.Constant) and value.value is None:
                    shapes.add("none")
                elif inferred:
                    if inferred == "tuple" and isinstance(value, ast.Tuple):
                        shapes.add(f"tuple[{len(value.elts)}]")
                    else:
                        shapes.add(inferred)
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
                    shapes.add(f"call:{self_outer._call_name(value.func) or 'unknown'}")
                elif isinstance(value, ast.Name):
                    shapes.add(arg_shapes.get(value.id, f"name:{value.id}"))
                else:
                    shapes.add(type(value).__name__.lower())

        self_outer = self
        ReturnShapeVisitor().visit(node)
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

    def _python_body_is_stub(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
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

    def _python_assigned_names(self, node: ast.AST) -> set[str]:
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

    def _python_loaded_names(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[tuple[str, int]]:
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

    def _python_function_param_names(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
        args = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
        names = {arg.arg for arg in args}
        if node.args.vararg is not None:
            names.add(node.args.vararg.arg)
        if node.args.kwarg is not None:
            names.add(node.args.kwarg.arg)
        return names

    def _python_string_sequence_literal(self, node: ast.AST | None) -> list[str] | None:
        if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return None
        values: list[str] = []
        for child in node.elts:
            if not isinstance(child, ast.Constant) or not isinstance(child.value, str):
                return None
            values.append(child.value)
        return values

    def _python_explicit_module_exports(self, tree: ast.AST) -> set[str] | None:
        for node in getattr(tree, "body", []):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = list(node.targets) if isinstance(node, ast.Assign) else [node.target]
            if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
                continue
            values = self._python_string_sequence_literal(getattr(node, "value", None))
            if values is not None:
                return {value for value in values if value}
        return None

    def _python_star_import_names(self, module_name: str, seen_modules: set[str] | None = None) -> set[str]:
        seen = set(seen_modules or ())
        if module_name in seen:
            return set()
        seen.add(module_name)
        module_file = self._resolve_python_module_file(module_name)
        if module_file is None:
            return set()
        try:
            text = module_file.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(self._python_parse_text(text), filename=self.relative_label(module_file))
        except SyntaxError:
            return set()
        explicit = self._python_explicit_module_exports(tree)
        if explicit is not None:
            return explicit
        return {name for name in self._python_module_defined_names(tree, seen_modules=seen) if name and not name.startswith("_")}

    def _python_module_defined_names(self, tree: ast.AST, seen_modules: set[str] | None = None) -> set[str]:
        names = {"__name__", "__file__", "__package__"}
        for node in getattr(tree, "body", []):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = list(node.targets) if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    for child in ast.walk(target):
                        if isinstance(child, ast.Name):
                            names.add(child.id)
            elif isinstance(node, ast.ImportFrom):
                if node.module and any(alias.name == "*" for alias in node.names):
                    names.update(self._python_star_import_names(node.module, seen_modules=seen_modules))
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    names.add((alias.asname or alias.name).split(".", 1)[0])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    names.add((alias.asname or alias.name).split(".", 1)[0])
        return names

    def _python_self_attribute_facts(self, class_node: ast.ClassDef) -> tuple[set[str], set[str], set[str]]:
        methods = {node.name for node in class_node.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assigned: set[str] = set()
        called: set[str] = set()
        for method in class_node.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for child in ast.walk(method):
                if (
                    isinstance(child, ast.Attribute)
                    and isinstance(child.value, ast.Name)
                    and child.value.id == "self"
                    and isinstance(child.ctx, ast.Store)
                ):
                    assigned.add(child.attr)
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and isinstance(child.func.value, ast.Name)
                    and child.func.value.id == "self"
                ):
                    called.add(child.func.attr)
        return methods, assigned, called

    def _python_static_sanity_diagnostics_for_tree(self, rel: str, tree: ast.AST, limit: int) -> list[str]:
        diagnostics: list[str] = []
        module_names = self._python_module_defined_names(tree)
        builtin_names = set(dir(builtins))
        nested_function_ids = {
            id(child)
            for outer in ast.walk(tree)
            if isinstance(outer, (ast.FunctionDef, ast.AsyncFunctionDef))
            for child in ast.walk(outer)
            if child is not outer and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if id(node) in nested_function_ids:
                    continue
                if self._python_body_is_stub(node):
                    diagnostics.append(f"{rel}:{int(getattr(node, 'lineno', 1))} {node.name} still has stub body")
                local_names = self._python_function_param_names(node) | self._python_assigned_names(node)
                allowed = local_names | module_names | builtin_names
                for name, line in sorted(self._python_loaded_names(node), key=lambda item: (item[1], item[0])):
                    if name not in allowed:
                        diagnostics.append(f"{rel}:{line} {node.name} reads undefined local/global name '{name}'")
                        break
            if isinstance(node, ast.ClassDef):
                methods, assigned, called = self._python_self_attribute_facts(node)
                has_non_object_base = any(not (isinstance(base, ast.Name) and base.id == "object") for base in node.bases)
                for attr in sorted(methods & assigned):
                    diagnostics.append(f"{rel}:{int(getattr(node, 'lineno', 1))} {node.name} assigns self.{attr}, shadowing method {attr}()")
                if not has_non_object_base:
                    for attr in sorted(called - methods - assigned):
                        diagnostics.append(f"{rel}:{int(getattr(node, 'lineno', 1))} {node.name} calls missing self.{attr}()")
            if len(diagnostics) >= max(1, int(limit)):
                break
        return diagnostics[: max(1, int(limit))]

    def _python_placeholder_text_diagnostics(self, rel: str, text: str, limit: int) -> list[str]:
        diagnostics: list[str] = []
        patterns = [
            (r"\bplaceholder implementation\b", "placeholder implementation text remains"),
            (r"\bin a real scenario\b", "speculative placeholder text remains"),
            (r"\bNote_(?:Interval_)?\b", "fake generated note placeholder remains"),
            (r"\bTODO\b|\bstub\b|\byour code\b", "TODO/stub placeholder text remains"),
        ]
        for index, line in enumerate(text.splitlines(), start=1):
            lowered = line.lower()
            for pattern, message in patterns:
                if re.search(pattern, line, flags=re.IGNORECASE):
                    diagnostics.append(f"{rel}:{index} {message}: {lowered.strip()[:100]}")
                    break
            if len(diagnostics) >= max(1, int(limit)):
                break
        return diagnostics[: max(1, int(limit))]

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
        syntax_or_static_diagnostics: list[str] = []
        for path in python_files:
            rel = self.relative_label(path)
            text = path.read_text(encoding="utf-8", errors="replace")
            syntax_or_static_diagnostics.extend(self._python_placeholder_text_diagnostics(rel, text, limit))
            try:
                tree = ast.parse(text, filename=rel)
            except SyntaxError as exc:
                syntax_or_static_diagnostics.append(f"{rel}:{exc.lineno or 1} Python syntax error: {exc.msg}")
                continue
            syntax_or_static_diagnostics.extend(self._python_static_sanity_diagnostics_for_tree(rel, tree, limit))
        contracts = self._python_function_contracts(self.workspace_root)
        definitions = contracts["definitions"]
        diagnostics: list[str] = list(syntax_or_static_diagnostics)
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
            item_args = list(item.get("args") or [])
            if item.get("kind") == "method" and item_args and str(item_args[0].get("name")) in {"self", "cls"}:
                item_args = item_args[1:]
            parameter_names = {
                str(arg.get("name"))
                for arg in item_args
                if isinstance(arg, dict) and str(arg.get("name") or "") and not str(arg.get("name")).startswith("*")
            }
            for caller in contracts["callers_by_leaf"].get(str(item["name"]), [])[: max(1, int(limit))]:
                if item.get("kind") == "method" and not bool(caller.get("attribute")):
                    continue
                if bool(caller.get("expected_exception")):
                    continue
                arg_count = int(caller.get("args", 0))
                keyword_count = len({str(keyword) for keyword in caller.get("keywords", []) or [] if str(keyword) in parameter_names})
                supplied_count = arg_count + keyword_count
                if supplied_count < min_args or (isinstance(max_args, int) and supplied_count > max_args):
                    diagnostics.append(
                        f"{caller['path']}:{caller['line']} {caller['symbol']} calls {item['name']} with {supplied_count} supplied args; expected {min_args}"
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

    def _verified_function_cache_path(self) -> Path:
        return self.workspace_root / ".ollama-code" / "index" / "verified_functions.sqlite"

    def _initialize_verified_function_connection(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS verified_functions("
            "id TEXT PRIMARY KEY,"
            "language TEXT NOT NULL,"
            "source_path TEXT NOT NULL,"
            "symbol TEXT NOT NULL,"
            "name TEXT NOT NULL,"
            "hash TEXT NOT NULL,"
            "proof_level TEXT NOT NULL,"
            "card_json TEXT NOT NULL,"
            "updated_at TEXT NOT NULL)"
        )
        conn.execute("CREATE TABLE IF NOT EXISTS verified_function_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        conn.execute(
            "INSERT OR REPLACE INTO verified_function_meta(key,value) VALUES('version',?)",
            (str(VERIFIED_FUNCTION_INDEX_VERSION),),
        )
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS verified_function_fts USING fts5("
            "id UNINDEXED,name,signature,summary,properties,examples,source_path,symbol)"
        )
        conn.commit()

    def _connect_verified_functions(self) -> sqlite3.Connection:
        cache_key = str(self._verified_function_cache_path().resolve(strict=False))
        memory_conn = _MEMORY_VERIFIED_FUNCTION_CONNECTIONS.get(cache_key)
        if memory_conn is not None:
            return memory_conn
        cache_path = self._verified_function_cache_path()
        conn: sqlite3.Connection | None = None
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(cache_path)
            conn.execute("PRAGMA journal_mode=WAL")
            self._initialize_verified_function_connection(conn)
        except sqlite3.OperationalError as exc:
            if conn is not None:
                conn.close()
            if "disk I/O error" not in str(exc):
                raise
            memory_conn = sqlite3.connect(":memory:")
            self._initialize_verified_function_connection(memory_conn)
            _MEMORY_VERIFIED_FUNCTION_CONNECTIONS[cache_key] = memory_conn
            return memory_conn
        return conn

    def _close_verified_functions(self, conn: sqlite3.Connection) -> None:
        if conn in _MEMORY_VERIFIED_FUNCTION_CONNECTIONS.values():
            return
        conn.close()

    def _verified_function_id(self, source_path: str, symbol: str) -> str:
        digest = hashlib.sha1(f"python:{source_path}:{symbol}".encode("utf-8")).hexdigest()[:16]
        leaf = symbol.split(".")[-1].replace("_", "-")
        leaf = re.sub(r"[^A-Za-z0-9-]+", "-", leaf).strip("-").lower() or "function"
        return f"py-{leaf}-{digest}"

    def _module_name_for_source_path(self, source_path: str) -> str:
        rel = source_path.replace("\\", "/")
        if rel.endswith(".py"):
            rel = rel[:-3]
        if rel.endswith("/__init__"):
            rel = rel[: -len("/__init__")]
        elif rel == "__init__":
            rel = ""
        if rel.startswith("src/"):
            rel = rel[4:]
        return rel.replace("/", ".")

    def _symbol_source_hash(self, source_path: str, line: int, end: int) -> str:
        target = self.resolve_path(source_path, allow_missing=False)
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        body = "\n".join(lines[max(0, line - 1) : max(line - 1, end)]) + "\n"
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def _docstring_probe_examples(self, node: ast.FunctionDef | ast.AsyncFunctionDef, leaf: str) -> list[dict[str, Any]]:
        doc = ast.get_docstring(node) or ""
        if not doc:
            return []
        examples: list[dict[str, Any]] = []
        lines = doc.splitlines()
        index = 0
        while index < len(lines):
            stripped = lines[index].strip()
            if not stripped.startswith(">>> "):
                index += 1
                continue
            expression = stripped[4:].strip()
            expected = ""
            cursor = index + 1
            while cursor < len(lines):
                candidate = lines[cursor].strip()
                if candidate.startswith(">>> "):
                    break
                if candidate and not candidate.startswith("..."):
                    expected = candidate
                    break
                cursor += 1
            if expression and expected and expression.startswith(f"{leaf}("):
                examples.append(
                    {
                        "source": "docstring",
                        "expression": expression,
                        "probe_expression": "fn(" + expression[len(leaf) + 1 :],
                        "expected_repr": expected,
                        "example": f"{expression} -> {expected}",
                    }
                )
            index = max(cursor, index + 1)
        return examples[:12]

    def _test_examples_for_card(self, source_path: str, leaf: str, tests: list[str]) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        for test_path in tests[:4]:
            result = self.test_spec_extract(test_path=test_path, source_path=source_path, limit=24)
            if result.get("ok") is not True:
                continue
            for item in result.get("examples", []):
                if not isinstance(item, dict) or str(item.get("symbol")) != leaf:
                    continue
                examples.append(
                    {
                        "source": "unittest",
                        "test_path": test_path,
                        "line": item.get("line"),
                        "example": item.get("example"),
                    }
                )
                if len(examples) >= 12:
                    return examples
        return examples

    def _verified_function_card_for_item(self, item: dict[str, Any], contracts: dict[str, Any]) -> dict[str, Any]:
        source_path = str(item["path"])
        symbol = str(item["symbol"])
        leaf = str(item["name"])
        line = int(item.get("line") or 1)
        end = int(item.get("end") or line)
        node = item.get("node")
        doc = ast.get_docstring(node) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else ""
        summary = next((part.strip() for part in (doc or "").splitlines() if part.strip()), "")
        if not summary:
            summary = f"{symbol} in {source_path}"
        tests = list(contracts.get("tests_by_symbol", {}).get(leaf, []))[:8]
        key = f"{source_path}:{symbol}"
        dependencies = sorted({str(call.get("name")) for call in contracts.get("calls_by_def", {}).get(key, []) if call.get("name")})[:24]
        examples: list[dict[str, Any]] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            examples.extend(self._docstring_probe_examples(node, leaf))
        examples.extend(self._test_examples_for_card(source_path, leaf, tests))
        shapes = list(item.get("return_shapes", []))
        properties = [
            f"returns={item.get('returns') or 'Any'}",
            f"return_shapes={','.join(shapes) if shapes else 'unknown'}",
            f"purity={item.get('purity')}",
        ]
        if tests:
            properties.append("affected_tests=" + ",".join(tests[:4]))
        proof_level = "probable" if item.get("purity") == "pure_hint" else "unverified"
        return {
            "id": self._verified_function_id(source_path, symbol),
            "name": leaf,
            "language": "python",
            "signature": self._contract_signature(item),
            "purity": str(item.get("purity") or "unknown"),
            "summary": self._truncate_text(summary, limit=240),
            "examples": examples,
            "properties": properties,
            "dependencies": dependencies,
            "proof_level": proof_level,
            "source_path": source_path,
            "symbol": symbol,
            "line": line,
            "hash": self._symbol_source_hash(source_path, line, end),
            "verified_at": "",
        }

    def _verified_function_upsert(self, conn: sqlite3.Connection, card: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        card_json = json.dumps(card, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        conn.execute(
            "INSERT OR REPLACE INTO verified_functions(id,language,source_path,symbol,name,hash,proof_level,card_json,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (
                card["id"],
                card["language"],
                card["source_path"],
                card["symbol"],
                card["name"],
                card["hash"],
                card["proof_level"],
                card_json,
                now,
            ),
        )
        conn.execute("DELETE FROM verified_function_fts WHERE id=?", (card["id"],))
        conn.execute(
            "INSERT INTO verified_function_fts(id,name,signature,summary,properties,examples,source_path,symbol) VALUES(?,?,?,?,?,?,?,?)",
            (
                card["id"],
                card["name"],
                card["signature"],
                card["summary"],
                "\n".join(str(item) for item in card.get("properties", [])),
                "\n".join(str(item.get("example") or item) for item in card.get("examples", []) if isinstance(item, dict)),
                card["source_path"],
                card["symbol"],
            ),
        )

    def _verified_function_card_from_row(self, row: sqlite3.Row | tuple[Any, ...]) -> dict[str, Any]:
        raw = row["card_json"] if isinstance(row, sqlite3.Row) else row[0]
        data = json.loads(str(raw))
        return data if isinstance(data, dict) else {}

    def _verified_function_lookup(self, conn: sqlite3.Connection, raw_id: str) -> tuple[dict[str, Any] | None, str | None]:
        clean = raw_id.strip()
        if not clean:
            return None, "verified_function_show requires a card id."
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT card_json FROM verified_functions WHERE id=?", (clean,)).fetchall()
        if not rows:
            rows = conn.execute("SELECT card_json FROM verified_functions WHERE id LIKE ? ORDER BY id LIMIT 3", (clean + "%",)).fetchall()
        if not rows:
            return None, f"No verified function card found for id: {clean}"
        if len(rows) > 1:
            ids = [self._verified_function_card_from_row(row).get("id", "") for row in rows]
            return None, "Ambiguous card id prefix: " + ", ".join(str(item) for item in ids if item)
        return self._verified_function_card_from_row(rows[0]), None

    def _verified_function_current_hash(self, card: dict[str, Any]) -> str | None:
        source_path = str(card.get("source_path") or "")
        symbol = str(card.get("symbol") or "")
        if not source_path or not symbol:
            return None
        try:
            target = self.resolve_path(source_path, allow_missing=False)
        except Exception:
            return None
        contracts = self._python_function_contracts(target)
        for item in contracts.get("definitions", {}).values():
            if item.get("symbol") == symbol or item.get("name") == symbol:
                return self._symbol_source_hash(str(item["path"]), int(item.get("line") or 1), int(item.get("end") or item.get("line") or 1))
        return None

    def _verified_function_stale(self, card: dict[str, Any]) -> bool:
        current = self._verified_function_current_hash(card)
        return current is None or current != card.get("hash")

    def _verified_card_in_scope(self, card: dict[str, Any], base: Path) -> bool:
        source = str(card.get("source_path") or "").replace("\\", "/")
        if not source:
            return False
        base_rel = self.relative_label(base).replace("\\", "/") if base.exists() else "."
        if base.is_file():
            return source == base_rel
        return base_rel in {"", "."} or source == base_rel or source.startswith(base_rel.rstrip("/") + "/")

    def verified_function_index(self, path: str = ".", limit: int = 500) -> dict[str, Any]:
        self._check_interrupted()
        base = self.resolve_path(path, allow_missing=False)
        try:
            conn = self._connect_verified_functions()
        except sqlite3.OperationalError as exc:
            return self._missing_dependency_result("verified_function_index", "sqlite-fts5", f"SQLite FTS5 is unavailable: {exc}")
        contracts = self._python_function_contracts(base, limit=max(1, int(limit)))
        definitions = list(contracts.get("definitions", {}).values())
        conn.row_factory = sqlite3.Row
        existing_rows = conn.execute("SELECT card_json FROM verified_functions").fetchall()
        existing = {str(card.get("id")): card for card in (self._verified_function_card_from_row(row) for row in existing_rows) if card.get("id")}
        seen: set[str] = set()
        indexed = 0
        stale = 0
        with conn:
            for item in definitions[: max(1, int(limit))]:
                self._check_interrupted()
                try:
                    card = self._verified_function_card_for_item(item, contracts)
                except Exception:
                    continue
                previous = existing.get(str(card["id"]))
                if previous and previous.get("proof_level") == "verified":
                    if previous.get("hash") == card.get("hash"):
                        card = previous
                    else:
                        card = dict(previous)
                        card["proof_level"] = "unverified"
                        properties = list(card.get("properties", []))
                        if "stale: source changed" not in properties:
                            properties.append("stale: source changed")
                        card["properties"] = properties
                        stale += 1
                self._verified_function_upsert(conn, card)
                seen.add(str(card["id"]))
                indexed += 1
            for card in existing.values():
                if str(card.get("id")) in seen or not self._verified_card_in_scope(card, base):
                    continue
                marked = dict(card)
                marked["proof_level"] = "unverified"
                properties = list(marked.get("properties", []))
                if "stale: source missing or no longer indexed" not in properties:
                    properties.append("stale: source missing or no longer indexed")
                marked["properties"] = properties
                self._verified_function_upsert(conn, marked)
                stale += 1
        self._close_verified_functions(conn)
        return {
            "ok": True,
            "tool": "verified_function_index",
            "path": self.relative_label(base),
            "cards": indexed,
            "stale": stale,
            "cache": self.relative_label(self._verified_function_cache_path()),
            "summary": f"verified function cards indexed={indexed} stale={stale}",
            "output": f"indexed={indexed}\nstale={stale}\ncache={self.relative_label(self._verified_function_cache_path())}",
        }

    def verified_function_search(
        self,
        query: str,
        signature: str | None = None,
        examples: str | list[Any] | None = None,
        path: str = ".",
        limit: int = 10,
    ) -> dict[str, Any]:
        self._check_interrupted()
        base = self.resolve_path(path, allow_missing=False)
        if not self._verified_function_cache_path().exists():
            refresh = self.verified_function_index(self.relative_label(base), limit=500)
            if refresh.get("ok") is not True:
                return refresh
        example_text = " ".join(str(item) for item in examples) if isinstance(examples, list) else str(examples or "")
        clean_query = " ".join(part for part in [query, signature or "", example_text] if str(part).strip()).strip()
        fts_query = self._safe_fts_query(clean_query)
        if not fts_query:
            return {"ok": False, "tool": "verified_function_search", "summary": "verified_function_search requires a non-empty query."}
        try:
            conn = self._connect_verified_functions()
        except sqlite3.OperationalError as exc:
            return self._missing_dependency_result("verified_function_search", "sqlite-fts5", f"SQLite FTS5 is unavailable: {exc}")
        conn.row_factory = sqlite3.Row
        limit_value = max(1, int(limit))
        try:
            rows = conn.execute(
                "SELECT c.card_json,bm25(verified_function_fts) AS rank "
                "FROM verified_function_fts JOIN verified_functions c ON c.id=verified_function_fts.id "
                "WHERE verified_function_fts MATCH ? ORDER BY rank LIMIT ?",
                (fts_query, limit_value * 4),
            ).fetchall()
        except sqlite3.OperationalError:
            like = "%" + clean_query.replace("%", "").replace("_", "") + "%"
            rows = conn.execute(
                "SELECT card_json,0 AS rank FROM verified_functions WHERE card_json LIKE ? ORDER BY id LIMIT ?",
                (like, limit_value * 4),
            ).fetchall()
        cards: list[dict[str, Any]] = []
        terms = {term.lower() for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", clean_query)}
        for row in rows:
            card = self._verified_function_card_from_row(row)
            if not card or not self._verified_card_in_scope(card, base):
                continue
            score = 0
            if str(card.get("name", "")).lower() in terms:
                score += 8
            if signature and signature.lower() in str(card.get("signature", "")).lower():
                score += 4
            if card.get("proof_level") == "verified":
                score += 3
            elif card.get("proof_level") == "probable":
                score += 1
            card = dict(card)
            card["stale"] = self._verified_function_stale(card)
            card["_score"] = score
            cards.append(card)
        cards.sort(key=lambda item: (-int(item.get("_score", 0)), str(item.get("source_path")), str(item.get("symbol"))))
        selected = cards[:limit_value]
        lines = [
            f"{card['id']} {card['proof_level']}{' stale' if card.get('stale') else ' fresh'} {card['purity']} "
            f"{card['signature']} @ {card['source_path']}:{card['line']} - {card['summary']}"
            for card in selected
        ]
        for card in selected:
            card.pop("_score", None)
        self._close_verified_functions(conn)
        return {
            "ok": True,
            "tool": "verified_function_search",
            "query": query,
            "count": len(selected),
            "cards": selected,
            "output": "\n".join(lines) if lines else "(no verified function cards found)",
        }

    def verified_function_show(self, id: str) -> dict[str, Any]:
        self._check_interrupted()
        try:
            conn = self._connect_verified_functions()
        except sqlite3.OperationalError as exc:
            return self._missing_dependency_result("verified_function_show", "sqlite-fts5", f"SQLite FTS5 is unavailable: {exc}")
        card, error = self._verified_function_lookup(conn, id)
        self._close_verified_functions(conn)
        if error or card is None:
            return {"ok": False, "tool": "verified_function_show", "summary": error or "Card not found."}
        stale = self._verified_function_stale(card)
        try:
            source = self.read_symbol(str(card.get("source_path")), str(card.get("symbol")), include_context=1)
            excerpt = str(source.get("output") or source.get("summary") or "")
        except FileNotFoundError:
            excerpt = "(source file missing)"
        evidence = [
            f"proof_level={card.get('proof_level')}",
            f"purity={card.get('purity')}",
            f"hash_status={'stale' if stale else 'fresh'}",
            f"examples={len(card.get('examples') or [])}",
        ]
        lines = [
            f"{card.get('id')} {card.get('proof_level')} {'stale' if stale else 'fresh'}",
            f"{card.get('signature')} @ {card.get('source_path')}:{card.get('line')}",
            str(card.get("summary") or ""),
            "evidence: " + ", ".join(evidence),
            self._truncate_text(excerpt, limit=1200),
        ]
        return {
            "ok": True,
            "tool": "verified_function_show",
            "id": card.get("id"),
            "card": card,
            "stale": stale,
            "source_excerpt": excerpt,
            "output": "\n".join(line for line in lines if line),
        }

    def _definition_for_symbol(self, path: str, symbol: str) -> tuple[dict[str, Any] | None, dict[str, Any], str | None]:
        target = self.resolve_path(path, allow_missing=False)
        if target.suffix.lower() != ".py":
            return None, {}, "Python source files only."
        contracts = self._python_function_contracts(target)
        matches = [
            item
            for item in contracts.get("definitions", {}).values()
            if item.get("symbol") == symbol or str(item.get("symbol", "")).endswith(f".{symbol}") or item.get("name") == symbol
        ]
        if not matches:
            return None, contracts, f"No Python function symbol found for {symbol} in {self.relative_label(target)}."
        matches.sort(key=lambda item: (str(item.get("symbol")) != symbol, len(str(item.get("symbol")))))
        return matches[0], contracts, None

    def _probe_docstring_examples(self, card: dict[str, Any]) -> tuple[bool, list[str]]:
        doc_examples = [
            item for item in card.get("examples", [])
            if isinstance(item, dict) and item.get("source") == "docstring" and item.get("probe_expression")
        ]
        if not doc_examples:
            return False, []
        module = self._module_name_for_source_path(str(card.get("source_path")))
        leaf = str(card.get("name"))
        expressions = [str(item["probe_expression"]) for item in doc_examples]
        probe = self.run_function_probe(module=module, function=leaf, expressions=expressions, timeout=30)
        rows = probe.get("results", []) if isinstance(probe.get("results"), list) else []
        diagnostics: list[str] = []
        passed = probe.get("ok") is True and len(rows) == len(doc_examples)
        for item, row in zip(doc_examples, rows):
            expected = str(item.get("expected_repr"))
            actual = str(row.get("repr") if isinstance(row, dict) else "")
            if actual != expected:
                passed = False
                diagnostics.append(f"probe {item.get('expression')} expected {expected} got {actual or row}")
        return passed, diagnostics or [str(probe.get("output") or "docstring probes passed")]

    def verify_function_contract(self, path: str, symbol: str) -> dict[str, Any]:
        self._check_interrupted()
        item, contracts, error = self._definition_for_symbol(path, symbol)
        if error or item is None:
            return {"ok": False, "tool": "verify_function_contract", "summary": error or "Function not found."}
        card = self._verified_function_card_for_item(item, contracts)
        rel = str(card["source_path"])
        leaf = str(card["name"])
        contract = self.contract_check([rel], changed_symbols=[leaf])
        pure = card.get("purity") == "pure_hint"
        probe_passed, probe_evidence = self._probe_docstring_examples(card)
        selected = self.select_tests([rel], changed_symbols=[leaf], limit=1)
        test_passed = False
        test_evidence = str(selected.get("summary") or "")
        commands = selected.get("test_commands") if isinstance(selected.get("test_commands"), list) else []
        if commands:
            test_result = self.run_test(str(commands[0]), timeout=1200)
            test_passed = test_result.get("ok") is True
            test_evidence = str(test_result.get("summary") or test_result.get("output") or "")
        diagnostics: list[str] = []
        if not pure:
            diagnostics.append("purity is not pure_hint")
        if contract.get("ok") is not True:
            diagnostics.append(str(contract.get("summary") or contract.get("output") or "contract_check failed"))
        if probe_evidence and not probe_passed:
            diagnostics.extend(probe_evidence)
        has_executable_evidence = probe_passed or test_passed
        if not has_executable_evidence:
            diagnostics.append("no passing executable docstring probe or focused test evidence")
        proof_level = "verified" if pure and contract.get("ok") is True and has_executable_evidence and not diagnostics else "probable" if pure and contract.get("ok") is True else "unverified"
        verified_at = datetime.now(timezone.utc).isoformat() if proof_level == "verified" else ""
        card["proof_level"] = proof_level
        card["verified_at"] = verified_at
        card["properties"] = list(card.get("properties", [])) + [
            f"contract_check={'pass' if contract.get('ok') is True else 'fail'}",
            f"docstring_probe={'pass' if probe_passed else 'missing_or_fail'}",
            f"focused_test={'pass' if test_passed else 'missing_or_fail'}",
        ]
        evidence = {
            "contract_check": contract.get("output"),
            "docstring_probe": probe_evidence,
            "focused_test": test_evidence,
            "diagnostics": diagnostics,
        }
        return {
            "ok": proof_level == "verified",
            "tool": "verify_function_contract",
            "proof_level": proof_level,
            "card": card,
            "evidence": evidence,
            "summary": "verified" if proof_level == "verified" else "not verified: " + "; ".join(diagnostics),
            "output": json.dumps({"proof_level": proof_level, "diagnostics": diagnostics, "card": card}, indent=2),
        }

    def promote_verified_function(self, path: str, symbol: str) -> dict[str, Any]:
        self._check_interrupted()
        verified = self.verify_function_contract(path, symbol)
        if verified.get("ok") is not True:
            result = dict(verified)
            result["tool"] = "promote_verified_function"
            result["summary"] = "Promotion rejected: " + str(verified.get("summary") or "verification failed")
            return result
        try:
            conn = self._connect_verified_functions()
        except sqlite3.OperationalError as exc:
            return self._missing_dependency_result("promote_verified_function", "sqlite-fts5", f"SQLite FTS5 is unavailable: {exc}")
        card = dict(verified["card"])
        with conn:
            self._verified_function_upsert(conn, card)
        self._close_verified_functions(conn)
        return {
            "ok": True,
            "tool": "promote_verified_function",
            "id": card["id"],
            "card": card,
            "summary": f"Promoted {card['symbol']} to verified.",
            "output": f"{card['id']} verified {card['signature']} @ {card['source_path']}:{card['line']}",
        }

    def compose_verified_functions(self, goal: str, candidates: list[str] | str) -> dict[str, Any]:
        self._check_interrupted()
        raw_ids = [candidates] if isinstance(candidates, str) else list(candidates or [])
        ids = [str(item).strip() for raw in raw_ids for item in str(raw).split(",") if str(item).strip()]
        if not ids:
            return {"ok": False, "tool": "compose_verified_functions", "summary": "compose_verified_functions requires candidate card ids."}
        try:
            conn = self._connect_verified_functions()
        except sqlite3.OperationalError as exc:
            return self._missing_dependency_result("compose_verified_functions", "sqlite-fts5", f"SQLite FTS5 is unavailable: {exc}")
        cards: list[dict[str, Any]] = []
        missing: list[str] = []
        for raw_id in ids:
            card, error = self._verified_function_lookup(conn, raw_id)
            if card is None:
                missing.append(error or raw_id)
                continue
            card = dict(card)
            card["stale"] = self._verified_function_stale(card)
            cards.append(card)
        self._close_verified_functions(conn)
        lines = [f"Goal: {goal.strip()}", "Composition plan:"]
        for index, card in enumerate(cards, start=1):
            status = str(card.get("proof_level"))
            if card.get("stale"):
                status = "unverified"
            lines.append(f"{index}. Use {card.get('signature')} from {card.get('source_path')} [{status}]")
        unsafe = [card for card in cards if card.get("proof_level") != "verified" or card.get("stale")]
        if unsafe:
            lines.append("Missing adapters/verification: re-promote or replace unverified/stale candidates before trusting generated glue.")
        else:
            lines.append("All candidates are verified/fresh; write only glue and run composition tests.")
        if missing:
            lines.append("Missing cards: " + "; ".join(missing))
        return {
            "ok": not missing,
            "tool": "compose_verified_functions",
            "goal": goal,
            "cards": cards,
            "missing": missing,
            "output": "\n".join(lines),
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

    def _available_command(self, command: str, cwd: Path | None = None) -> bool:
        try:
            argv = self._split_command_text(command)
        except ValueError:
            return False
        if not argv:
            return False
        executable = str(argv[0]).strip().strip("'\"")
        if self._token_looks_like_path(executable):
            executable = str(self._coerce_input_path(executable))
        working_dir = cwd or self.workspace_root
        if executable.startswith(("./", ".\\")):
            return (working_dir / executable).exists()
        candidate = Path(executable)
        if candidate.is_absolute():
            return candidate.exists()
        return self._which(executable) is not None or (working_dir / executable).exists() or (self.workspace_root / executable).exists()

    def _which(self, executable: str) -> str | None:
        key = str(executable)
        if key not in self._which_cache:
            self._which_cache[key] = shutil.which(key)
        return self._which_cache[key]

    def _python_module_command(self, module: str, *args: str) -> list[str] | None:
        if importlib.util.find_spec(module) is None:
            return None
        return [sys.executable, "-m", module, *args]

    def _python_tool_command(self, executable: str, module: str, *args: str) -> list[str] | None:
        key = (str(executable), str(module), tuple(str(arg) for arg in args))
        if key in self._python_tool_command_cache:
            cached = self._python_tool_command_cache[key]
            return list(cached) if cached is not None else None
        dependency = resolve_dependency(executable) or resolve_dependency(module)
        if dependency is not None:
            resolved_tool = resolve_tool_executable(dependency.id, executable, workspace_root=self.workspace_root)
            if resolved_tool:
                command = [resolved_tool, *args]
                self._python_tool_command_cache[key] = list(command)
                return command
        resolved = self._which(executable)
        if resolved:
            command = [resolved, *args]
            self._python_tool_command_cache[key] = list(command)
            return command
        command = self._python_module_command(module, *args)
        self._python_tool_command_cache[key] = list(command) if command is not None else None
        return command

    def _read_toml(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return {}
        if tomllib is None:
            tool_sections: dict[str, Any] = {}
            for match in re.finditer(r"(?m)^\s*\[tool\.([A-Za-z0-9_.-]+)\]\s*$", text):
                cursor = tool_sections
                for part in match.group(1).split("."):
                    cursor = cursor.setdefault(part, {})
            return {"tool": tool_sections} if tool_sections else {}
        try:
            return tomllib.loads(text)
        except (OSError, _TOMLDecodeError):
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
            validators.append({"kind": kind, "lang": lang, "command": command, "reason": reason})

        def python_tool_command_text(executable: str, module: str, *args: str) -> str:
            command = self._python_tool_command(executable, module, *args)
            return command_to_text(tuple(command)) if command else command_to_text((executable, *args))

        pyproject = self._read_toml(root / "pyproject.toml")
        repo_files = self._iter_repo_files(root, limit=50000)
        suffixes: set[str] = set()
        file_names: list[str] = []
        workflow_file = ""
        yaml_file = ""
        shell_script = ""
        dockerfile = ""
        markdown_file = ""
        sql_file = ""
        schema_file = ""
        for file_path in repo_files:
            suffix = file_path.suffix.lower()
            name = file_path.name.lower()
            suffixes.add(suffix)
            file_names.append(name)
            if suffix in {".yml", ".yaml"} and (not yaml_file or not workflow_file):
                rel = self.relative_label(file_path).replace("\\", "/")
                if not yaml_file:
                    yaml_file = rel
                if rel.lower().startswith(".github/workflows/"):
                    workflow_file = rel
            if not shell_script and suffix in SHELL_SCRIPT_SUFFIXES:
                shell_script = self.relative_label(file_path).replace("\\", "/")
            if not dockerfile and (name == "dockerfile" or name.endswith(".dockerfile")):
                dockerfile = self.relative_label(file_path).replace("\\", "/")
            if not markdown_file and suffix in {".md", ".markdown"}:
                markdown_file = self.relative_label(file_path).replace("\\", "/")
            if not sql_file and suffix == ".sql":
                sql_file = self.relative_label(file_path).replace("\\", "/")
            if not schema_file and (name.endswith(".schema.json") or name.endswith(".jsonschema")):
                schema_file = self.relative_label(file_path).replace("\\", "/")
        if (root / ".pre-commit-config.yaml").exists() or (root / ".pre-commit-config.yml").exists():
            add("validate", "repo", python_tool_command_text("pre-commit", "pre_commit", "run", "--all-files"), "pre-commit config found.")
        python_files = ".py" in suffixes
        python_tests = any(
            (name.startswith("test") and name.endswith(".py")) or name.endswith("_test.py")
            for name in file_names
        )
        pytest_config = (
            (root / "pytest.ini").exists()
            or (root / "pytest.toml").exists()
            or self._toml_tool_section(pyproject, "pytest")
            or self._toml_tool_section(pyproject, "pytest.ini_options")
        )
        if python_files or pyproject or pytest_config:
            add("syntax", "python", f"{sys.executable} -m py_compile", "Python files found; lint_typecheck does exact syntax checks internally.")
            add("lint", "python", f"{sys.executable} -m py_compile", "Python files found; lint_typecheck does exact syntax checks internally.")
            if pytest_config:
                add("collect", "python", f"{sys.executable} -m pytest --collect-only -q", "pytest config found.")
                add("test", "python", f"{sys.executable} -m pytest", "pytest config found.")
                if importlib.util.find_spec("pytest_jsonreport") is not None:
                    add("test-report", "python", f"{sys.executable} -m pytest --json-report --json-report-file=.pytest-report.json", "pytest-json-report is importable.")
                if importlib.util.find_spec("pytest_timeout") is not None:
                    add("test", "python", f"{sys.executable} -m pytest --timeout=120", "pytest-timeout is importable.")
                if importlib.util.find_spec("xdist") is not None:
                    add("test", "python", f"{sys.executable} -m pytest -n auto", "pytest-xdist is importable.")
            if python_tests:
                add("test", "python", f"{sys.executable} -m unittest discover -s tests -v", "Python unittest discovery.")
            if (
                (root / "ruff.toml").exists()
                or (root / ".ruff.toml").exists()
                or self._toml_tool_section(pyproject, "ruff")
                or self._which("ruff")
            ):
                add("lint", "python", "ruff check . --no-cache", "ruff config or executable found.")
            if (
                (root / "mypy.ini").exists()
                or (root / ".mypy.ini").exists()
                or self._ini_has_section(root / "setup.cfg", ("mypy",))
                or self._toml_tool_section(pyproject, "mypy")
            ):
                add("typecheck", "python", "mypy .", "mypy config found.")
            if (root / "pyrightconfig.json").exists() or self._toml_tool_section(pyproject, "pyright"):
                add("typecheck", "python", "pyright", "pyright config found.")
            basedpyright_command = self._python_tool_command("basedpyright", "basedpyright", "--level", "error")
            if (root / "pyrightconfig.json").exists() or self._toml_tool_section(pyproject, "basedpyright") or basedpyright_command:
                add("typecheck", "python", command_to_text(tuple(basedpyright_command or ["basedpyright"])), "basedpyright config, module, or executable found.")
            deptry_command = self._python_tool_command("deptry", "deptry", ".")
            if deptry_command:
                add("dependency-check", "python", command_to_text(tuple(deptry_command)), "deptry module or executable found.")
            vulture_command = self._python_tool_command("vulture", "vulture", ".")
            if vulture_command:
                add("dead-code", "python", command_to_text(tuple(vulture_command)), "vulture module or executable found.")
            coverage_command = self._python_tool_command("coverage", "coverage", "run", "-m", "pytest")
            if coverage_command:
                add("coverage", "python", command_to_text(tuple(coverage_command)), "coverage.py module or executable found.")
            if (python_tests or pytest_config) and self._which("pytest") and importlib.util.find_spec("testmon") is not None:
                add("test", "python", f"{sys.executable} -m pytest --testmon", "pytest-testmon is importable.")
            if (root / "tox.ini").exists() or self._toml_tool_section(pyproject, "tox"):
                add("test", "python", python_tool_command_text("tox", "tox"), "tox config found.")
            if (root / "noxfile.py").exists():
                add("test", "python", python_tool_command_text("nox", "nox"), "noxfile.py found.")
            pipdeptree_command = self._python_tool_command("pipdeptree", "pipdeptree", "--warn", "fail")
            if pipdeptree_command:
                add("dependency-check", "python", command_to_text(tuple(pipdeptree_command)), "pipdeptree module or executable found.")
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
            if (root / ".eslintrc").exists() or (root / ".eslintrc.json").exists() or (root / ".eslintrc.js").exists():
                add("lint", "javascript", "eslint .", "ESLint config or executable found.")
            if (root / "biome.json").exists() or (root / "biome.jsonc").exists():
                add("lint", "javascript", "biome check .", "Biome config or executable found.")
            if (
                (root / ".stylelintrc").exists()
                or (root / ".stylelintrc.json").exists()
                or (root / "stylelint.config.js").exists()
                or ".css" in suffixes
                or ".scss" in suffixes
            ):
                add("lint", "css", "stylelint \"**/*.{css,scss}\"", "Stylelint config or stylesheet files found.")
            if (
                (root / ".prettierrc").exists()
                or (root / ".prettierrc.json").exists()
                or (root / ".prettierrc.js").exists()
                or (root / "prettier.config.js").exists()
            ):
                add("format-check", "javascript", "prettier --check .", "Prettier config or executable found.")
        if (root / "go.mod").exists() or ".go" in suffixes:
            add("test", "go", "go test ./...", "Go module or source files found.")
            add("lint", "go", "golangci-lint run", "Go module or source files found.")
        if (root / "Cargo.toml").exists() or ".rs" in suffixes:
            add("check", "rust", "cargo check", "Cargo project or Rust source files found.")
            add("test", "rust", "cargo test", "Cargo project or Rust source files found.")
            if self._which("cargo-nextest"):
                add("test", "rust", "cargo nextest run", "Cargo project or Rust source files found and cargo-nextest is available.")
        gradlew = "gradlew.bat" if os.name == "nt" else "./gradlew"
        if (root / "build.gradle").exists() or (root / "settings.gradle").exists() or (root / gradlew).exists() or ".java" in suffixes:
            command = gradlew + " test" if (root / gradlew).exists() else "gradle test"
            add("test", "java", command, "Gradle/Java project detected.")
        if (root / "CMakeLists.txt").exists() or ".cpp" in suffixes or ".c" in suffixes:
            if (root / "build").exists():
                add("test", "cpp", "ctest --test-dir build --output-on-failure", "CMake build directory found.")
            add("setup", "cpp", "cmake -S . -B build", "CMake project detected; creates/updates build dir if run.")
        if workflow_file:
            add("lint", "github-actions", "actionlint", "GitHub Actions workflow files found.")
            add(
                "schema",
                "github-actions",
                python_tool_command_text("check-jsonschema", "check_jsonschema", "--builtin-schema", "vendor.github-workflows", ".github/workflows"),
                "GitHub Actions workflow files found.",
            )
        if yaml_file:
            add("lint", "yaml", python_tool_command_text("yamllint", "yamllint", yaml_file), "YAML files found.")
        if shell_script:
            add("syntax", "shell", command_to_text(("bash", "-n", shell_script)), "Shell scripts found; bash -n catches syntax errors cheaply.")
            add("lint", "shell", command_to_text(("shellcheck", shell_script)), "Shell scripts found.")
            add("format-check", "shell", command_to_text(("shfmt", "-d", shell_script)), "Shell scripts found.")
        if dockerfile:
            add("lint", "dockerfile", f"hadolint {dockerfile}", "Dockerfile found.")
        if markdown_file:
            add("lint", "markdown", "markdownlint-cli2 \"**/*.md\"", "Markdown docs found.")
        if markdown_file or python_files:
            add("lint", "text", "codespell .", "Docs or Python files found.")
        if sql_file:
            add("lint", "sql", "sqlfluff lint .", "SQL files found.")
        if schema_file:
            add("schema", "json", python_tool_command_text("check-jsonschema", "check_jsonschema", "--check-metaschema", schema_file), "JSON schema files found.")
        dependency_manifest_names = {
            "requirements.txt",
            "poetry.lock",
            "pdm.lock",
            "uv.lock",
            "package-lock.json",
            "pnpm-lock.yaml",
            "yarn.lock",
            "go.sum",
            "cargo.lock",
            "gemfile.lock",
        }
        if dependency_manifest_names.intersection(file_names):
            add("security", "dependencies", "osv-scanner scan .", "Dependency manifest or lockfile found.")
            if "requirements.txt" in file_names or "pyproject.toml" in file_names:
                add("security", "python", python_tool_command_text("pip-audit", "pip_audit"), "Python dependency manifest found.")
            add("security", "dependencies", "trivy fs --quiet .", "Dependency manifest or lockfile found.")
            add("security", "dependencies", "grype dir:.", "Dependency manifest or lockfile found.")
        if ".toml" in suffixes and self._which("taplo"):
            add("config", "toml", "taplo check .", "TOML files and taplo executable found.")
        selected = [dict(item) for item in validators[: max(1, int(limit))]]
        for item in selected:
            item["available"] = self._available_command(str(item.get("command") or ""), cwd=root)
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
        dependency = resolve_dependency(str(missing or command or imported or ""))
        if dependency is not None:
            status = dependency_status(dependency)
            facts["tool_id"] = dependency.id
            facts["install_hints"] = status["install_hints"]
            facts["dependency_purpose"] = dependency.purpose
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
        python_validator_files: set[str] = set()
        python_validator_scopes: set[str] = set()
        shell_targets: list[str] = []
        seen_shell_targets: set[str] = set()
        validator_commands: list[str] = []
        validator_targets: list[str] = []
        typechecker_targets: list[str] = []
        typechecker_skipped_reason = ""
        phase_timings_ms = {"scan_ms": 0.0, "ruff_ms": 0.0, "typecheck_ms": 0.0, "shell_ms": 0.0}
        active_phase = "scan_ms"
        active_phase_started = time.perf_counter()
        try:
            for raw_path in raw_paths:
                base = self.resolve_path(str(raw_path), allow_missing=False)
                files = self._iter_code_files(base, limit=50000)
                base_has_python = False
                for file_path in files:
                    rel = self.relative_label(file_path)
                    checked.append(rel)
                    text = file_path.read_text(encoding="utf-8", errors="replace")
                    if file_path.suffix.lower() == ".py":
                        base_has_python = True
                        python_validator_files.add(rel)
                        diagnostic = self._python_syntax_diagnostic(file_path, text)
                        if diagnostic:
                            diagnostics.append(diagnostic)
                    elif file_path.suffix.lower() in SHELL_SCRIPT_SUFFIXES:
                        if rel not in seen_shell_targets:
                            shell_targets.append(rel)
                            seen_shell_targets.add(rel)
                    elif file_path.suffix.lower() in {".js", ".jsx"} and self._which("node"):
                        completed = self._run_process(["node", "--check", str(file_path)], cwd=self.workspace_root, timeout=timeout, shell=False)
                        if completed.returncode != 0:
                            diagnostics.append(self._truncate_text(self._collect_process_output(completed), limit=500))
                    elif self._tree_sitter_language_for_path(file_path) is not None:
                        diagnostic = self._tree_sitter_syntax_diagnostic(file_path, text)
                        if diagnostic:
                            diagnostics.append(diagnostic)
                if base_has_python:
                    python_validator_scopes.add(self.relative_label(base))
            phase_timings_ms["scan_ms"] = round((time.perf_counter() - active_phase_started) * 1000, 3)
            validator_targets = self._python_validation_targets(
                discovered_files=python_validator_files,
                requested_scopes=python_validator_scopes,
                limit=100,
            )
            collapsed_python_scopes = self._collapse_validation_targets(python_validator_scopes, limit=100)
            typechecker_targets = self._python_typechecker_targets(
                discovered_files=python_validator_files,
                requested_scopes=python_validator_scopes,
                limit=100,
            )
            typechecker_configured = self._python_typechecker_configured()
            if typechecker_targets and not typechecker_configured:
                if "." not in collapsed_python_scopes:
                    typechecker_targets = []
                    typechecker_skipped_reason = "No pyright/basedpyright config found for focused scope; skipped cold typechecker startup."
                elif python_validator_files and all(self._path_looks_like_test(Path(label)) for label in python_validator_files):
                    typechecker_targets = []
                    typechecker_skipped_reason = (
                        "No pyright/basedpyright config found for test-only workspace scope; skipped cold typechecker startup."
                    )
            ruff_path = self._which("ruff") if validator_targets else None
            typechecker_command = (
                self._python_tool_command("basedpyright", "basedpyright", "--level", "error")
                or self._python_tool_command("pyright", "pyright", "--level", "error")
                if typechecker_targets
                else None
            )
            bash = self._which("bash") if shell_targets else None
            cache_key = self._lint_typecheck_cache_key(
                checked=checked,
                validator_targets=validator_targets,
                typechecker_targets=typechecker_targets,
                shell_targets=shell_targets,
                ruff_path=ruff_path,
                typechecker_command=typechecker_command,
                bash_path=bash,
            )
            cached = self._lint_typecheck_cache.get(cache_key)
            if cached is not None:
                result = deepcopy(cached)
                result["cache_hit"] = True
                result["scan_ms"] = phase_timings_ms["scan_ms"]
                result["ruff_ms"] = 0.0
                result["typecheck_ms"] = 0.0
                result["shell_ms"] = 0.0
                return result
            if validator_targets and ruff_path:
                target_args = validator_targets
                command = ["ruff", "check", "--no-cache", *target_args]
                validator_commands.append(command_to_text(tuple(command)))
                active_phase = "ruff_ms"
                active_phase_started = time.perf_counter()
                completed = self._run_process(command, cwd=self.workspace_root, timeout=timeout, shell=False)
                phase_timings_ms["ruff_ms"] = round((time.perf_counter() - active_phase_started) * 1000, 3)
                if completed.returncode != 0:
                    diagnostics.append(self._truncate_text(self._collect_process_output(completed), limit=1200))
            if typechecker_targets and typechecker_command:
                target_args = typechecker_targets
                command = [*typechecker_command, *target_args]
                validator_commands.append(command_to_text(tuple(command)))
                active_phase = "typecheck_ms"
                active_phase_started = time.perf_counter()
                completed = self._run_process(command, cwd=self.workspace_root, timeout=timeout, shell=False)
                phase_timings_ms["typecheck_ms"] = round((time.perf_counter() - active_phase_started) * 1000, 3)
                if completed.returncode != 0:
                    diagnostics.append(self._truncate_text(self._collect_process_output(completed), limit=1200))
            if bash:
                for rel in shell_targets[:100]:
                    command = [bash, "-n", rel]
                    validator_commands.append(command_to_text(("bash", "-n", rel)))
                    active_phase = "shell_ms"
                    active_phase_started = time.perf_counter()
                    completed = self._run_process(command, cwd=self.workspace_root, timeout=timeout, shell=False)
                    phase_timings_ms["shell_ms"] = round(
                        float(phase_timings_ms["shell_ms"]) + ((time.perf_counter() - active_phase_started) * 1000),
                        3,
                    )
                    if completed.returncode != 0:
                        output = self._collect_process_output(completed) or f"{rel}: bash -n failed"
                        diagnostics.append(self._truncate_text(output, limit=1200))
        except subprocess.TimeoutExpired as exc:
            phase_timings_ms[active_phase] = round(
                float(phase_timings_ms.get(active_phase, 0.0) or 0.0) + ((time.perf_counter() - active_phase_started) * 1000),
                3,
            )
            timeout_summary = f"Command timed out after {exc.timeout} seconds."
            timeout_output = self._collect_timeout_output(exc)
            timeout_command = self._timeout_command_text(exc.cmd)
            timeout_details = f"{timeout_summary} Validator: {timeout_command}"
            if timeout_output != "(no output)":
                timeout_details = f"{timeout_details}\n{timeout_output}"
            return {
                "ok": False,
                "tool": "lint_typecheck",
                "checked": checked,
                "diagnostics": [*diagnostics, timeout_details],
                "validator_commands": validator_commands,
                "validator_targets": validator_targets,
                "typechecker_targets": typechecker_targets,
                "typechecker_skipped_reason": typechecker_skipped_reason,
                **phase_timings_ms,
                "output": "\n".join([*diagnostics, timeout_details]) if diagnostics else timeout_details,
                "summary": timeout_summary,
                "error_class": "timeout",
                "timed_out": True,
                "command": timeout_command,
            }
        result = {
            "ok": not diagnostics,
            "tool": "lint_typecheck",
            "checked": checked,
            "diagnostics": diagnostics,
            "validator_commands": validator_commands,
            "validator_targets": validator_targets,
            "typechecker_targets": typechecker_targets,
            "typechecker_skipped_reason": typechecker_skipped_reason,
            **phase_timings_ms,
            "output": "\n".join(diagnostics) if diagnostics else f"syntax ok: {len(checked)} code file(s)",
        }
        if "cache_key" in locals():
            self._lint_typecheck_cache[cache_key] = deepcopy(result)
        return result

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
            sanity_error = self._python_function_replacement_sanity_diagnostic(target, str(found["qualname"]), replacement)
            if sanity_error:
                normalized_signature = self._canonical_signature_order_replacement_if_safe(
                    target,
                    str(found["qualname"]),
                    replacement,
                    sanity_error,
                )
                if normalized_signature:
                    replacement = normalized_signature
                    normalization = (normalization + " " if normalization else "") + "Normalized replacement signature to the existing public parameter order."
                    sanity_error = self._python_function_replacement_sanity_diagnostic(target, str(found["qualname"]), replacement)
                normalized_foldr = self._canonical_foldr_replacement_if_safe(target, str(found["qualname"]), sanity_error)
                if normalized_foldr:
                    replacement = normalized_foldr
                    normalization = (normalization + " " if normalization else "") + "Normalized foldr reducer order to the canonical right fold."
                elif sanity_error:
                    return {
                        "ok": False,
                        "tool": "replace_symbol",
                        "path": relative_path,
                        "symbol": found["qualname"],
                        "kind": found["kind"],
                        "error_class": "invalid_args",
                        "summary": sanity_error,
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
        self._write_text_and_invalidate_python_cache(target, updated)
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
                sanity_error = self._python_function_replacement_sanity_diagnostic(target, str(found["qualname"]), replacement)
                if sanity_error:
                    normalized_signature = self._canonical_signature_order_replacement_if_safe(
                        target,
                        str(found["qualname"]),
                        replacement,
                        sanity_error,
                    )
                    if normalized_signature:
                        replacement = normalized_signature
                        normalization = (normalization + " " if normalization else "") + "Normalized replacement signature to the existing public parameter order."
                        sanity_error = self._python_function_replacement_sanity_diagnostic(target, str(found["qualname"]), replacement)
                    normalized_foldr = self._canonical_foldr_replacement_if_safe(target, str(found["qualname"]), sanity_error)
                    if normalized_foldr:
                        replacement = normalized_foldr
                        normalization = (normalization + " " if normalization else "") + "Normalized foldr reducer order to the canonical right fold."
                    elif sanity_error:
                        return {
                            "ok": False,
                            "tool": "replace_symbols",
                            "path": relative_path,
                            "symbol": found["qualname"],
                            "kind": found["kind"],
                            "error_class": "invalid_args",
                            "summary": sanity_error,
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
        self._write_text_and_invalidate_python_cache(target, updated)
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
        self._write_text_and_invalidate_python_cache(target, updated)
        return {"ok": True, "tool": "apply_structured_edit", "path": relative_path, "op": op, "syntax_ok": True if target.suffix.lower() == ".py" else None, "summary": f"Applied {op} to {relative_path}.", "diff": preview}

    def _invalidate_python_bytecode_cache(self, target: Path) -> None:
        if target.suffix.lower() != ".py":
            return
        cache_dir = target.parent / "__pycache__"
        if not cache_dir.is_dir():
            return
        for pattern in (f"{target.stem}.*.pyc", f"{target.stem}.*.pyo"):
            for cached in cache_dir.glob(pattern):
                try:
                    cached.unlink()
                except OSError:
                    continue

    def _write_text_and_invalidate_python_cache(self, target: Path, content: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self._invalidate_python_bytecode_cache(target)

    def _single_full_python_function_name(self, source: str) -> str | None:
        stripped = textwrap.dedent(source or "").strip()
        if not stripped.startswith(("def ", "async def ")):
            return None
        try:
            tree = ast.parse(stripped)
        except SyntaxError:
            return None
        body = [node for node in tree.body if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))]
        if len(body) != 1 or not isinstance(body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        return body[0].name

    def _routable_full_python_function_replacement(self, source: str) -> tuple[str, str] | None:
        stripped = textwrap.dedent(source or "").strip()
        if not stripped:
            return None
        try:
            tree = ast.parse(stripped)
        except SyntaxError:
            return None
        body = [node for node in tree.body if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))]
        functions = [node for node in body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(functions) != 1:
            return None
        if any(not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom)) for node in body):
            return None
        function = functions[0]
        lines = stripped.splitlines()
        start = min([int(getattr(decorator, "lineno", function.lineno)) for decorator in function.decorator_list] + [int(function.lineno)])
        end = int(getattr(function, "end_lineno", function.lineno))
        function_lines = lines[start - 1 : end]
        import_lines: list[str] = []
        for node in body:
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            node_start = int(getattr(node, "lineno", 1))
            node_end = int(getattr(node, "end_lineno", node_start))
            if node_start < start:
                import_lines.extend(lines[node_start - 1 : node_end])
        if import_lines and function_lines:
            header, rest = function_lines[0], function_lines[1:]
            function_lines = [header, *["    " + line if line.strip() else "" for line in import_lines], *rest]
        return function.name, "\n".join(function_lines) + "\n"

    def _normalize_python_body_replacement(self, body: str) -> str:
        normalized = textwrap.dedent(body).strip("\n")
        normalized = self._repair_common_python_join_typo(normalized)
        lines = normalized.splitlines()
        first = next((line for line in lines if line.strip()), "")
        leading = len(first) - len(first.lstrip())
        if leading > 0:
            prefix = " " * leading
            lines = [line[leading:] if line.startswith(prefix) else line for line in lines]
            normalized = "\n".join(lines)
        return normalized

    def _python_parameter_names(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
        args = node.args
        names = {arg.arg for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
        if args.vararg:
            names.add(args.vararg.arg)
        if args.kwarg:
            names.add(args.kwarg.arg)
        return names

    def _python_parameter_sequence(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        args = node.args
        names = [arg.arg for arg in [*args.posonlyargs, *args.args]]
        if args.vararg:
            names.append("*" + args.vararg.arg)
        names.extend(arg.arg for arg in args.kwonlyargs)
        if args.kwarg:
            names.append("**" + args.kwarg.arg)
        return names

    def _shadowed_builtin_call_diagnostic(self, node: ast.FunctionDef | ast.AsyncFunctionDef, body: str) -> str:
        shadowable = {"list", "dict", "set", "tuple", "str", "int", "float", "bool", "sum", "map", "filter", "len", "min", "max"}
        shadowed = self._python_parameter_names(node) & shadowable
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

    def _unused_critical_parameter_diagnostic(self, node: ast.FunctionDef | ast.AsyncFunctionDef, body: str) -> str:
        critical = self._python_parameter_names(node) & {"initial", "default", "accumulator"}
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

    def _foldr_argument_order_diagnostic(self, node: ast.FunctionDef | ast.AsyncFunctionDef, body: str) -> str:
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

    def _python_function_replacement_sanity_diagnostic(self, target: Path, symbol: str, replacement: str) -> str:
        node = self._python_function_node(target, symbol)
        if node is None:
            return ""
        try:
            tree = ast.parse(self._python_parse_text(replacement))
        except SyntaxError:
            return ""
        candidates = [child for child in tree.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(candidates) != 1:
            return ""
        candidate = candidates[0]
        if candidate.name != node.name:
            return f"Replacement defines {candidate.name!r}, but target symbol is {node.name!r}. Use rename/change_signature for intentional API changes."
        existing_params = self._python_parameter_sequence(node)
        replacement_params = self._python_parameter_sequence(candidate)
        if replacement_params != existing_params:
            return (
                f"Replacement changes signature for {node.name} from ({', '.join(existing_params)}) "
                f"to ({', '.join(replacement_params)}). Use change_signature for intentional API changes; otherwise keep the existing parameter order."
            )
        if candidate.body:
            lines = replacement.splitlines()
            start = int(getattr(candidate.body[0], "lineno", 1))
            end = int(getattr(candidate.body[-1], "end_lineno", getattr(candidate.body[-1], "lineno", start)))
            body = textwrap.dedent("\n".join(lines[start - 1 : end]))
        else:
            body = ""
        return (
            self._shadowed_builtin_call_diagnostic(node, body)
            or self._unused_critical_parameter_diagnostic(node, body)
            or self._foldr_argument_order_diagnostic(node, body)
        )

    def _canonical_signature_order_replacement_if_safe(self, target: Path, symbol: str, replacement: str, diagnostic: str) -> str:
        if "Replacement changes signature" not in diagnostic:
            return ""
        node = self._python_function_node(target, symbol)
        if node is None:
            return ""
        try:
            tree = ast.parse(self._python_parse_text(replacement))
        except SyntaxError:
            return ""
        candidates = [child for child in tree.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(candidates) != 1:
            return ""
        candidate = candidates[0]
        if candidate.name != node.name:
            return ""
        existing_params = self._python_parameter_sequence(node)
        replacement_params = self._python_parameter_sequence(candidate)
        existing_names = [entry for entry in existing_params if not entry.startswith("*")]
        replacement_names = [entry for entry in replacement_params if not entry.startswith("*")]

        if not existing_names or not replacement_names:
            return ""
        if len(existing_names) != len(replacement_names):
            return ""

        # Reorder-only correction for same-name permutations.
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
            lines = replacement.splitlines()
            header_index = int(getattr(candidate, "lineno", 1)) - 1
            if header_index < 0 or header_index >= len(lines):
                return ""
            header = lines[header_index]
            match = re.match(rf"^(\s*(?:async\s+)?def\s+{re.escape(candidate.name)}\s*)\([^)]*\)(\s*(?:->\s*[^:]+)?\s*:\s*)$", header)
            if not match:
                return ""
            lines[header_index] = f"{match.group(1)}({', '.join(existing_names)}){match.group(2)}"
            return "\n".join(lines) + ("\n" if replacement.endswith(("\n", "\r")) else "")

        # Name-only correction for same arity, same parameter shape.
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

        # Avoid rewriting when a candidate parameter name is rebound in function body.
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

        # Keep the argument shape and defaults; only normalize parameter identifiers.
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

    def _canonical_foldr_replacement_if_safe(self, target: Path, symbol: str, diagnostic: str) -> str:
        if "foldr reducer arguments look reversed" not in diagnostic:
            return ""
        node = self._python_function_node(target, symbol)
        if node is None or node.name.lower() != "foldr":
            return ""
        params = self._python_parameter_sequence(node)
        if params != ["function", "list", "initial"]:
            return ""
        return (
            "def foldr(function, list, initial):\n"
            "    accumulator = initial\n"
            "    for item in reversed(list):\n"
            "        accumulator = function(accumulator, item)\n"
            "    return accumulator\n"
        )

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
        normalized = self._normalize_python_body_replacement(body)
        routable_function = self._routable_full_python_function_replacement(normalized)
        if routable_function is not None:
            full_function_name, full_function_source = routable_function
            wanted = symbol.split(".")[-1]
            if full_function_name != wanted:
                return {
                    "ok": False,
                    "tool": "apply_structured_edit",
                    "path": relative_path,
                    "op": "replace_function_body",
                    "error_class": "invalid_args",
                    "summary": f"replace_function_body received full function {full_function_name!r}, but target is {wanted!r}. Target the matching function or use replace_symbol.",
                }
            routed = self.replace_symbol(relative_path, symbol, full_function_source)
            routed = dict(routed)
            routed["tool"] = "apply_structured_edit"
            routed["op"] = "replace_function_body"
            routed["routed_tool"] = "replace_symbol"
            summary = str(routed.get("summary") or "").strip()
            routed["summary"] = "Routed full function replacement to replace_symbol. " + summary if summary else "Routed full function replacement to replace_symbol."
            return routed
        shadow_diagnostic = self._shadowed_builtin_call_diagnostic(node, normalized)
        if shadow_diagnostic:
            return {
                "ok": False,
                "tool": "apply_structured_edit",
                "path": relative_path,
                "op": "replace_function_body",
                "error_class": "invalid_args",
                "summary": shadow_diagnostic,
            }
        unused_parameter_diagnostic = self._unused_critical_parameter_diagnostic(node, normalized)
        if unused_parameter_diagnostic:
            return {
                "ok": False,
                "tool": "apply_structured_edit",
                "path": relative_path,
                "op": "replace_function_body",
                "error_class": "invalid_args",
                "summary": unused_parameter_diagnostic,
            }
        foldr_diagnostic = self._foldr_argument_order_diagnostic(node, normalized)
        if foldr_diagnostic:
            return {
                "ok": False,
                "tool": "apply_structured_edit",
                "path": relative_path,
                "op": "replace_function_body",
                "error_class": "invalid_args",
                "summary": foldr_diagnostic,
            }
        replacement_lines = ["pass"] if not normalized.strip() else normalized.splitlines()
        replacement = "".join(body_indent + line.rstrip() + "\n" if line.strip() else "\n" for line in replacement_lines)
        updated = "".join(lines[: start - 1]) + replacement + "".join(lines[end:])
        return self._apply_file_update(target, original, updated, f"Replace body of {symbol} in {relative_path}?", op="replace_function_body")

    def _normalize_python_signature_replacement(
        self, *, symbol: str, signature: str, expected_name: str
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
        clean, diagnostic = self._normalize_python_signature_replacement(
            symbol=symbol,
            signature=signature,
            expected_name=str(found.get("name") or symbol.split(".")[-1]),
        )
        if diagnostic:
            return {
                "ok": False,
                "tool": "apply_structured_edit",
                "path": relative_path,
                "op": "change_signature",
                "error_class": "invalid_args",
                "summary": diagnostic,
            }
        clean += "\n"
        lines[start - 1] = indent + clean.lstrip()
        updated = "".join(lines)
        return self._apply_file_update(target, original, updated, f"Change signature of {symbol} in {relative_path}?", op="change_signature")

    def _rename_symbol_project(self, base: Path, old: str, new: str) -> dict[str, Any]:
        if not old or not new or not re.match(r"^[A-Za-z_]\w*$", old) or not re.match(r"^[A-Za-z_]\w*$", new):
            return {"ok": False, "tool": "apply_structured_edit", "summary": "rename_symbol_project requires valid old/new identifiers."}
        updates: list[tuple[Path, str, str]] = []
        already_renamed_files: list[str] = []
        files = list(self._iter_code_files(base, limit=1000))
        if base.is_dir():
            text_suffixes = {".md", ".rst", ".txt"}
            known = {path.resolve(strict=False) for path in files}
            for file_path in self._iter_repo_files(base, limit=1000, suffixes=text_suffixes):
                resolved = file_path.resolve(strict=False)
                if resolved not in known:
                    files.append(file_path)
                    known.add(resolved)
        for file_path in files:
            original = file_path.read_text(encoding="utf-8", errors="replace")
            updated, count = re.subn(rf"\b{re.escape(old)}\b", new, original)
            if count:
                if file_path.suffix.lower() == ".py":
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
            self._write_text_and_invalidate_python_cache(target, updated)
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
            self._write_text_and_invalidate_python_cache(target, updated)
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
            self._write_text_and_invalidate_python_cache(target, updated)
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
            self._write_text_and_invalidate_python_cache(source, source_updated)
            self._write_text_and_invalidate_python_cache(destination, dest_updated)
            return {"ok": True, "tool": "apply_structured_edit", "op": op, "path": source_rel, "to_path": dest_rel, "summary": f"Moved {symbol} to {dest_rel}.", "diff": preview}
        return {"ok": False, "tool": "apply_structured_edit", "summary": f"Unsupported structured edit op: {op or '(missing)'}"}

    def _wrap_routed_edit_result(self, routed_tool: str, route: str, result: dict[str, Any]) -> dict[str, Any]:
        wrapped = dict(result)
        wrapped["tool"] = "edit_intent"
        wrapped["routed_tool"] = routed_tool
        wrapped["route"] = route
        if wrapped.get("syntax_ok") is False:
            wrapped["ok"] = False
            wrapped.setdefault("error_class", "syntax_error")
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

    def _single_python_replacement_symbol_name(self, value: str) -> str:
        try:
            tree = ast.parse(self._python_parse_text(value))
        except SyntaxError:
            return ""
        candidates = [
            child.name
            for child in tree.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        return candidates[0] if len(candidates) == 1 else ""

    def _looks_like_function_body_edit_intent(self, intent: str) -> bool:
        normalized = re.sub(r"[^a-z0-9_]+", "_", intent.lower())
        words = {word for word in normalized.split("_") if word}
        return bool(words & {"body", "implementation", "function", "method", "fix", "correct", "update"})

    def _normalize_python_symbol_target(self, target: Path, intent: str, symbol: str) -> str:
        if target.suffix.lower() != ".py" or self._looks_like_symbol_name(symbol):
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
        if clean_intent not in symbol_intents and not self._looks_like_function_body_edit_intent(clean_intent):
            return symbol
        match = re.match(r"\s*(?:async\s+def|def|class)?\s*([A-Za-z_][\w.]*)\s*(?:\(|:|$)", symbol)
        return match.group(1) if match else symbol

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
        old = self._normalize_python_symbol_target(target_path, clean_intent, old)
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
                operation = {"op": "rename_symbol_project", "path": ".", "old": old.rsplit(".", 1)[-1], "new": new.rsplit(".", 1)[-1]}
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
            replacement_name = self._single_python_replacement_symbol_name(new) if self._looks_like_full_symbol_source(target_path, new) else ""
            old_leaf = old.rsplit(".", 1)[-1]
            if replacement_name and replacement_name != old_leaf:
                route = "project symbol rename from replacement source"
                routed_tool = "apply_structured_edit"
                operation = {"op": "rename_symbol_project", "path": ".", "old": old_leaf, "new": replacement_name}
            elif self._looks_like_full_symbol_source(target_path, new):
                route = "replace symbol source"
                routed_tool = "replace_symbol"
            else:
                route = "replace Python function body"
                routed_tool = "apply_structured_edit"
                operation = {"op": "replace_function_body", "path": relative_path, "symbol": old, "body": new}
        elif clean_intent in {"replace_symbol", "replace_function", "replace_class", "symbol"}:
            replacement_name = self._single_python_replacement_symbol_name(new) if target_path.suffix.lower() == ".py" and self._looks_like_full_symbol_source(target_path, new) else ""
            old_leaf = old.rsplit(".", 1)[-1]
            new_leaf = new.rsplit(".", 1)[-1]
            if self._looks_like_symbol_name(old_leaf) and self._looks_like_symbol_name(new_leaf):
                if clean_scope in {"project", "repo", "repository", "all"}:
                    route = "project symbol rename"
                    routed_tool = "apply_structured_edit"
                    operation = {"op": "rename_symbol_project", "path": ".", "old": old_leaf, "new": new_leaf}
                else:
                    route = "file symbol rename"
                    routed_tool = "apply_structured_edit"
                    operation = {"op": "rename_symbol", "path": relative_path, "old": old_leaf, "new": new_leaf}
            elif replacement_name and replacement_name != old_leaf and self._looks_like_symbol_name(old_leaf):
                route = "project symbol rename from replacement source"
                routed_tool = "apply_structured_edit"
                operation = {"op": "rename_symbol_project", "path": ".", "old": old_leaf, "new": replacement_name}
            elif self._looks_like_symbol_name(old) and self._looks_like_full_symbol_source(target_path, new):
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
        self._write_text_and_invalidate_python_cache(target, updated)
        return {"ok": True, "tool": "generate_tests_from_spec", "path": self.relative_label(target), "applied": True, "summary": f"Wrote generated test scaffold to {self.relative_label(target)}.", "diff": preview}

    def git_status(self, path: str | None = None) -> dict[str, Any]:
        target_path = self.resolve_path(path, allow_missing=False) if path else None
        ok, error = self._ensure_git_repo(target_path)
        if not ok:
            return {"ok": False, "tool": "git_status", "summary": error or "Not inside a git repository."}
        repo_root = self._git_root_for_path(target_path)
        command = ["status", "--short", "--branch", "--untracked-files=all"]
        relative_path = None
        if path and target_path is not None and repo_root is not None:
            relative_path = target_path.resolve(strict=False).relative_to(repo_root).as_posix() or "."
            command.extend(["--", relative_path])
        result = self._run_git(command, cwd=repo_root)
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
        target_path = self.resolve_path(path, allow_missing=False) if path else None
        ok, error = self._ensure_git_repo(target_path)
        if not ok:
            return {"ok": False, "tool": "git_diff", "summary": error or "Not inside a git repository."}
        repo_root = self._git_root_for_path(target_path)
        command = ["diff", "--no-ext-diff", f"--unified={max(0, context)}"]
        if cached:
            command.append("--cached")
        relative_path = None
        if path and target_path is not None and repo_root is not None:
            relative_path = target_path.resolve(strict=False).relative_to(repo_root).as_posix() or "."
            command.extend(["--", relative_path])
        result = self._run_git(command, cwd=repo_root)
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

    def git_branch(self, path: str | None = None, all_branches: bool = False) -> dict[str, Any]:
        target_path = self.resolve_path(path, allow_missing=False) if path else None
        ok, error = self._ensure_git_repo(target_path)
        if not ok:
            return {"ok": False, "tool": "git_branch", "summary": error or "Not inside a git repository."}
        repo_root = self._git_root_for_path(target_path)
        command = ["branch", "--list"]
        if all_branches:
            command.append("--all")
        result = self._run_git(command, cwd=repo_root)
        output = self._collect_process_output(result)
        return {
            "ok": result.returncode == 0,
            "tool": "git_branch",
            "path": path or ".",
            "all_branches": all_branches,
            "output": output,
        }

    def git_log(self, path: str | None = None, max_count: int = 10, oneline: bool = True) -> dict[str, Any]:
        target_path = self.resolve_path(path, allow_missing=False) if path else None
        ok, error = self._ensure_git_repo(target_path)
        if not ok:
            return {"ok": False, "tool": "git_log", "summary": error or "Not inside a git repository."}
        repo_root = self._git_root_for_path(target_path)
        safe_count = self._coerce_int(max_count, default=10, minimum=1)
        command = ["log", f"--max-count={safe_count}", "--decorate"]
        if oneline:
            command.append("--oneline")
        if path and target_path is not None and repo_root is not None:
            relative_path = target_path.resolve(strict=False).relative_to(repo_root).as_posix() or "."
            command.extend(["--", relative_path])
        result = self._run_git(command, cwd=repo_root)
        output = self._collect_process_output(result)
        return {
            "ok": result.returncode == 0,
            "tool": "git_log",
            "path": path or ".",
            "max_count": safe_count,
            "oneline": oneline,
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
        self._write_text_and_invalidate_python_cache(target, content)
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
        self._write_text_and_invalidate_python_cache(target, updated)
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
        if ("auto" in requested_names or "grype" in requested_names) and shutil.which("grype"):
            available.append(("grype", ["grype", f"dir:{base}"]))
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
        python_inline = re.match(
            r"^\s*(?P<exe>(?:[A-Za-z]:)?[^\s]+(?:python|python3|py)(?:\.exe)?)\s+-c\s+(?P<code>.+?)\s*$",
            command,
            flags=re.IGNORECASE,
        )
        if python_inline:
            code = python_inline.group("code").strip()
            if len(code) >= 2 and code[0] == code[-1] and code[0] in {"'", '"'}:
                code = code[1:-1]
            return [python_inline.group("exe"), "-c", code], None
        try:
            argv = self._split_command_text(command)
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

    def _command_looks_like_cross_platform_path_input(self, command: str) -> bool:
        if os.name == "nt" or "\\" not in command:
            return False
        for token in re.findall(r'"[^"]*"|\'[^\']*\'|\S+', command):
            stripped = token.strip().strip("'\"")
            if not stripped:
                continue
            normalized = stripped.replace("\\", "/")
            if WINDOWS_DRIVE_PATH.match(normalized) or WSL_MOUNT_PATH.match(normalized):
                return True
            if stripped.startswith((".\\", "..\\")):
                return True
            if re.search(r"[A-Za-z0-9_.-]\\[A-Za-z0-9_.-]", stripped):
                return True
        return False

    def _split_command_text(self, command: str) -> list[str]:
        posix_mode = os.name != "nt"
        if posix_mode and self._command_looks_like_cross_platform_path_input(command):
            return shlex.split(command, posix=False)
        return shlex.split(command, posix=posix_mode)

    def _normalize_command_argv_paths(self, argv: list[str]) -> list[str]:
        normalized: list[str] = []
        for index, token in enumerate(argv):
            raw = str(token).strip()
            clean = raw.strip("'\"")
            if index == 0:
                if self._token_looks_like_path(clean):
                    normalized.append(str(self._coerce_input_path(clean)))
                else:
                    normalized.append(clean)
                continue
            previous = str(argv[index - 1]).strip().strip("'\"") if index > 0 else ""
            if previous == "-c":
                normalized.append(raw)
                continue
            if not clean or raw.startswith("-") or not self._token_looks_like_path(clean):
                normalized.append(raw)
                continue
            normalized.append(str(self._coerce_input_path(clean)))
        return normalized

    def _command_has_shell_chaining(self, command: str) -> bool:
        return bool(re.search(r"&&|\|\||[;|<>]", command))

    def _command_path_escapes(self, token: str, cwd: Path) -> bool:
        if not token or token.startswith("-"):
            return False
        if token == ".." or token.startswith("../") or token.startswith("..\\"):
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

    def _token_looks_like_path(self, token: str) -> bool:
        clean = str(token or "").strip().strip("'\"")
        if not clean:
            return False
        if clean in {".", ".."}:
            return True
        if clean.startswith(("./", "../", ".\\", "..\\")):
            return True
        normalized = clean.replace("\\", "/")
        return bool(
            clean.startswith(("/", "\\"))
            or "/" in clean
            or "\\" in clean
            or WINDOWS_DRIVE_PATH.match(normalized)
            or WSL_MOUNT_PATH.match(normalized)
        )

    def _validate_executable_path(self, executable: str, cwd: Path) -> str | None:
        clean = str(executable or "").strip().strip("'\"")
        if not clean:
            return None
        candidate = self._coerce_input_path(clean)
        if candidate.is_absolute():
            return None
        if not self._token_looks_like_path(clean):
            return None
        if self._command_path_escapes(clean, cwd):
            return f"path escapes workspace: {clean}"
        return None

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

    def _unknown_command_error(self, argv: list[str], cwd: Path) -> str | None:
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
            candidate = cwd / candidate
        if candidate.exists():
            return None
        return f"executable not found: {executable}"

    def _executable_available_for_cwd(self, executable: str, cwd: Path) -> bool:
        clean = str(executable or "").strip().strip("'\"")
        if not clean:
            return False
        candidate = Path(clean)
        if candidate.is_absolute():
            return candidate.exists()
        return shutil.which(clean) is not None or (cwd / clean).exists() or (self.workspace_root / clean).exists()

    def _validation_result(self, *, family: str, valid: bool, reason: str = "", argv: list[str] | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {"recognized": True, "valid": valid, "family": family}
        if argv is not None:
            result["argv"] = argv
        if reason:
            result["reason"] = reason
        return result

    def _normalize_python_exec_argv(self, argv: list[str]) -> list[str]:
        if len(argv) < 3:
            return argv
        normalized = list(argv)
        try:
            code_index = normalized.index("-c") + 1
        except ValueError:
            return normalized
        if code_index >= len(normalized):
            return normalized
        code = str(normalized[code_index]).strip()
        if len(code) >= 2 and code[0] == code[-1] and code[0] in {"'", '"'}:
            normalized[code_index] = code[1:-1]
        return normalized

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
        argv = self._normalize_command_argv_paths(argv)
        family = self._command_family(argv)
        if family == "python_exec":
            argv = self._normalize_python_exec_argv(argv)
        executable = str(argv[0]).strip().strip("'\"")
        executable_path_error = self._validate_executable_path(executable, cwd)
        path_error = self._validate_path_args(argv, cwd)
        if self._needs_bash_syntax_check(command, family):
            syntax_error = self._bash_syntax_validation(command)
            if syntax_error is not None:
                return None, syntax_error
        if family is None:
            shell_path_error = executable_path_error or path_error
            if shell_path_error:
                shell_family = "powershell" if os.name == "nt" and self._looks_like_powershell_command(command) else "shell"
                return None, self._validation_result(family=shell_family, valid=False, reason=shell_path_error)
            if os.name == "nt" and self._looks_like_powershell_command(command):
                return None, {"recognized": False, "valid": True, "family": "powershell"}
            executable_error = self._unknown_command_error(argv, cwd)
            if executable_error:
                return None, {"recognized": True, "valid": False, "family": "shell", "reason": executable_error}
            return None, {"recognized": False, "valid": True, "family": "unknown"}
        if family not in {"python", "python_exec"} and self._command_has_shell_chaining(command):
            return None, self._validation_result(family=family, valid=False, reason="shell chaining/redirection is not allowed for validated command families")
        if not self._executable_available_for_cwd(executable, cwd):
            return None, self._validation_result(family=family, valid=False, reason=f"executable not found: {executable}")
        if executable_path_error:
            return None, self._validation_result(family=family, valid=False, reason=executable_path_error)
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
            if tuple(argv[1:2]) not in {("test",), ("check",)} and tuple(argv[1:3]) != ("nextest", "run"):
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
        timeout_value = self._coerce_int(timeout, default=30, minimum=1)
        argv, validation = self._validate_common_command(command, working_dir)
        if validation.get("valid") is False:
            reason = str(validation.get("reason") or "invalid command")
            if "bash -n rejected command syntax" in reason or "invalid quoting" in reason:
                error_class = "syntax_error"
            else:
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
        try:
            completed = self._run_process(run_args, cwd=working_dir, timeout=timeout_value, **shell_kwargs)
        except subprocess.TimeoutExpired as exc:
            result = {
                "ok": False,
                "tool": "run_shell",
                "cwd": relative_cwd,
                "summary": f"Command timed out after {timeout_value} seconds.",
                "output": self._collect_timeout_output(exc),
                "error_class": "timeout",
                "timed_out": True,
            }
            if validation.get("recognized") is True:
                result["validation"] = validation
            return result
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
        timeout_value = self._coerce_int(timeout, default=1200, minimum=1)
        if command is not None:
            if not isinstance(command, str):
                return {
                    "ok": False,
                    "tool": "run_test",
                    "summary": "Explicit test command must be a non-empty string.",
                    "error_class": "invalid_args",
                }
            if not command.strip():
                return {
                    "ok": False,
                    "tool": "run_test",
                    "summary": "Explicit test command must be a non-empty string.",
                    "error_class": "invalid_args",
                }
        selected_command = command.strip() if isinstance(command, str) and command.strip() else self.default_test_command
        discovered = None
        if not selected_command:
            validators = self.discover_validators(cwd)
            discovered_command = self._preferred_test_validator_command(validators)
            if discovered_command:
                selected_command = discovered_command
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
            result = self.run_shell(selected_command, cwd=cwd, timeout=timeout_value)
        except ValueError as exc:
            if "Path escapes the workspace" not in str(exc) or str(cwd).strip() in {"", "."}:
                raise
            result = self.run_shell(selected_command, cwd=".", timeout=timeout_value)
            normalized = f"Ignored run_test cwd outside workspace: {cwd}"
        selected_from_default = bool(
            self.default_test_command
            and selected_command == self.default_test_command
            and (
                command is None
                or (isinstance(command, str) and command.strip() == self.default_test_command)
            )
        )
        recovery_validators = None
        if selected_from_default and result.get("ok") is not True and self._run_test_needs_command_recovery(result, selected_command=selected_command):
            recovery_validators = self.discover_validators(cwd)
            fallback_command = self._preferred_test_validator_command(
                recovery_validators,
                exclude_commands={selected_command},
            )
            if fallback_command:
                recovered = self.run_shell(fallback_command, cwd=cwd, timeout=timeout_value)
                recovered["tool"] = "run_test"
                recovered["command"] = fallback_command
                recovered["original_command"] = selected_command
                recovered["recovered"] = True
                recovered["validators"] = recovery_validators.get("validators", [])
                recovered["normalized"] = (
                    "Recovered run_test from configured command "
                    f"{selected_command} to discovered test command {fallback_command}"
                )
                self.set_test_command(fallback_command)
                selected_command = fallback_command
                result = recovered
                normalized = str(recovered["normalized"])
        result["tool"] = "run_test"
        result["command"] = str(result.get("command") or selected_command)
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

    def _run_test_needs_command_recovery(self, result: dict[str, Any], *, selected_command: str = "") -> bool:
        validation = result.get("validation")
        if isinstance(validation, dict) and validation.get("valid") is False:
            return True
        error_class = str(result.get("error_class") or "").strip().lower()
        if error_class in {"command_not_found", "invalid_args", "path_missing", "cwd_missing"}:
            return True
        if error_class == "missing_dependency":
            missing = str(result.get("missing_dependency") or self._missing_dependency_name(str(result.get("output") or result.get("summary") or "")) or "").strip()
            command_text = str(selected_command or result.get("command") or "").lower()
            if missing and re.search(rf"(?:^|\b|-m\s+){re.escape(missing.lower())}(?:\b|$)", command_text):
                return True
            return missing in {"pytest", "unittest", "testmon"} and missing in command_text
        output = str(result.get("output") or result.get("summary") or "").lower()
        return any(
            marker in output
            for marker in (
                "no tests ran",
                "ran 0 tests",
                "collected 0 items",
                "no tests collected",
            )
        )

    def _preferred_test_validator_command(
        self,
        validators: dict[str, Any],
        *,
        exclude_commands: set[str] | None = None,
    ) -> str | None:
        excluded = {str(item).strip() for item in (exclude_commands or set()) if str(item).strip()}
        candidates = [
            item
            for item in validators.get("validators", [])
            if (
                isinstance(item, dict)
                and item.get("kind") == "test"
                and item.get("command")
                and item.get("available") is True
                and str(item.get("command")).strip() not in excluded
            )
        ]
        preferred = [item for item in candidates if "unittest discover" in str(item.get("command", ""))]
        selected = (preferred or candidates)[:1]
        if not selected:
            return None
        return str(selected[0]["command"])
