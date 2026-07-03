from __future__ import annotations

import re
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ollama_code.agent_protocol import TargetLineReadSpec


PathResolver = Callable[[str], Path]
PathLabeler = Callable[[Path], str]
UnittestCommandNormalizer = Callable[[str], str | None]
BarePythonTestFileDetector = Callable[[str], str | None]
ShellTestRunPredicate = Callable[[str], bool]
ToolArgumentsNormalizer = Callable[[str], dict[str, Any] | None]
ArgvArgumentsNormalizer = Callable[[list[str]], dict[str, Any] | None]
ArgvToolNormalizer = Callable[[list[str]], tuple[str, dict[str, Any]] | None]
LineCountProvider = Callable[[str], int | None]
ToolOperation = tuple[str, dict[str, Any]]


def normalize_target_line_read_call(
    name: str,
    arguments: dict[str, Any],
    *,
    target_line_read: TargetLineReadSpec | None,
) -> tuple[str, dict[str, Any], str | None]:
    if name != "read_file" or target_line_read is None:
        return name, arguments, None
    requested_path = str(arguments.get("path", "")).strip()
    if requested_path and requested_path.replace("\\", "/") != target_line_read.path.replace("\\", "/"):
        return name, arguments, None
    try:
        current_start = int(arguments.get("start", 1))
        current_end = int(arguments.get("end", 200))
    except (TypeError, ValueError):
        current_start = 1
        current_end = 200
    if current_start <= target_line_read.line <= current_end and (current_end - current_start) <= 40:
        return name, arguments, None
    return (
        "read_file",
        {"path": target_line_read.path, "start": target_line_read.start, "end": target_line_read.end},
        f"Normalized read_file to the requested line {target_line_read.line} with a small surrounding range.",
    )


def normalize_run_test_call(
    name: str,
    arguments: dict[str, Any],
    *,
    request_text: str,
    default_test_command: str | None,
    normalize_unittest_file_command: UnittestCommandNormalizer,
) -> tuple[str, dict[str, Any], str | None]:
    if name in {"test", "tests", "pytest", "unittest"} and default_test_command:
        normalized = dict(arguments)
        normalized["command"] = default_test_command
        return "run_test", normalized, f"Normalized {name} tool alias to the configured run_test command."
    if name != "run_test":
        return name, arguments, None
    command = str(arguments.get("command", "")).strip()
    normalized_unittest = normalize_unittest_file_command(command)
    if normalized_unittest:
        normalized = dict(arguments)
        normalized["command"] = normalized_unittest
        return "run_test", normalized, "Normalized unittest file path command to unittest discover."
    if not default_test_command:
        return name, arguments, None
    lowered_command = command.lower()
    lowered_request = request_text.lower()
    vague_command = lowered_command in {"", "test", "tests", "pytest", "unittest", "python -m unittest", "python3 -m unittest"}
    command_not_requested = bool(command) and lowered_command not in lowered_request
    if not vague_command and not command_not_requested:
        return name, arguments, None
    normalized = dict(arguments)
    normalized["command"] = default_test_command
    return "run_test", normalized, "Normalized vague run_test command to the configured test command."


def _copy_shell_command_context(arguments: dict[str, Any], *, command: str) -> dict[str, Any]:
    normalized: dict[str, Any] = {"command": command}
    if "cwd" in arguments:
        normalized["cwd"] = arguments["cwd"]
    if "timeout" in arguments:
        normalized["timeout"] = arguments["timeout"]
    return normalized


def normalize_shell_test_call(
    name: str,
    arguments: dict[str, Any],
    *,
    approval_mode: str,
    explicit_run_shell: bool,
    explicit_run_test: bool,
    exact_shell_command: str | None,
    default_test_command: str | None,
    bare_python_test_file_command: BarePythonTestFileDetector,
    shell_command_looks_like_test_run: ShellTestRunPredicate,
) -> tuple[str, dict[str, Any], str | None]:
    if name != "run_shell":
        return name, arguments, None
    if approval_mode == "read-only":
        return name, arguments, None
    command = str(arguments.get("command", "")).strip()
    if not command:
        return name, arguments, None
    if explicit_run_shell:
        return name, arguments, None
    if explicit_run_test:
        return (
            "run_test",
            _copy_shell_command_context(arguments, command=command),
            "Normalized run_shell to run_test because the request explicitly requires run_test.",
        )
    bare_test_path = bare_python_test_file_command(command)
    if bare_test_path is not None:
        quoted_test_path = f'"{bare_test_path}"' if re.search(r"\s", bare_test_path) else bare_test_path
        normalized = _copy_shell_command_context(arguments, command=default_test_command or f"python -m pytest {quoted_test_path}")
        reason = (
            "Normalized bare Python test-file shell command to the configured run_test command."
            if default_test_command
            else "Normalized bare Python test-file shell command to run_test with pytest."
        )
        return "run_test", normalized, reason
    if not shell_command_looks_like_test_run(command):
        return name, arguments, None
    if exact_shell_command and command == exact_shell_command:
        return name, arguments, None
    normalized = _copy_shell_command_context(arguments, command=default_test_command or command)
    reason = (
        "Normalized shell test command to the configured run_test command."
        if default_test_command
        else "Normalized shell test command to run_test with the original command."
    )
    return "run_test", normalized, reason


def normalize_shell_inspection_call(
    name: str,
    arguments: dict[str, Any],
    *,
    approval_mode: str,
    explicit_run_shell: bool,
    exact_shell_command: str | None,
    normalize_find_exec_grep_shell_command: ToolArgumentsNormalizer,
    normalize_grep_shell_inspection: ArgvArgumentsNormalizer,
    normalize_head_tail_shell_inspection: ArgvArgumentsNormalizer,
    normalize_find_shell_inspection: ArgvToolNormalizer,
) -> tuple[str, dict[str, Any], str | None]:
    if name != "run_shell":
        return name, arguments, None
    if approval_mode == "read-only":
        return name, arguments, None
    command = str(arguments.get("command", "")).strip()
    if not command or exact_shell_command:
        return name, arguments, None
    if explicit_run_shell:
        return name, arguments, None
    if str(arguments.get("cwd") or ".").strip() not in {"", "."}:
        return name, arguments, None
    normalized_find_exec = normalize_find_exec_grep_shell_command(command)
    if normalized_find_exec is not None:
        return "search", normalized_find_exec, "Normalized find-plus-grep inspection to search for cacheable structured context."
    if re.search(r"[|&;<>`$()\r\n]", command):
        return name, arguments, None
    path_token = r'(?:"([^"]+)"|\'([^\']+)\'|(\S+))'
    cat_match = re.fullmatch(rf"(?:cat|type)\s+(?:-n\s+)?{path_token}", command, flags=re.IGNORECASE)
    if cat_match:
        path = next(group for group in cat_match.groups() if group)
        if path.startswith("-"):
            return name, arguments, None
        return "read_file", {"path": path}, "Normalized shell file inspection to read_file for cacheable structured context."
    list_match = re.fullmatch(rf"(?:ls|dir)\s+{path_token}", command, flags=re.IGNORECASE)
    if list_match:
        path = next(group for group in list_match.groups() if group)
        if path.startswith("-"):
            return name, arguments, None
        return "list_files", {"path": path}, "Normalized shell directory inspection to list_files for cacheable structured context."
    try:
        argv = shlex.split(command, posix=True)
    except ValueError:
        return name, arguments, None
    normalized_search = normalize_grep_shell_inspection(argv)
    if normalized_search is not None:
        return "search", normalized_search, "Normalized shell text search to search for cacheable structured context."
    if argv and argv[0].lower() in {"head", "tail"}:
        normalized_read = normalize_head_tail_shell_inspection(argv)
        if normalized_read is not None:
            return "read_file", normalized_read, "Normalized shell file preview to read_file for bounded structured context."
    if argv and argv[0].lower() == "find":
        normalized_find = normalize_find_shell_inspection(argv)
        if normalized_find is not None:
            tool_name, tool_arguments = normalized_find
            return tool_name, tool_arguments, "Normalized simple shell discovery to structured search for cacheable context."
    return name, arguments, None


def _has_prior_non_context_tool_call(tool_calls_this_turn: list[dict[str, Any]]) -> bool:
    prior_tool_names = [str(item.get("name") or "").strip() for item in tool_calls_this_turn]
    return any(prior_name and prior_name != "context_pack" for prior_name in prior_tool_names)


def normalize_import_repair_bootstrap_call(
    name: str,
    arguments: dict[str, Any],
    *,
    tool_calls_this_turn: list[dict[str, Any]],
    request_looks_like_explicit_python_import_bug_fix: bool,
    default_test_command: str | None,
) -> tuple[str, dict[str, Any], str | None]:
    if name != "list_files":
        return name, arguments, None
    if _has_prior_non_context_tool_call(tool_calls_this_turn):
        return name, arguments, None
    if not request_looks_like_explicit_python_import_bug_fix:
        return name, arguments, None
    normalized: dict[str, Any] = {}
    if default_test_command:
        normalized["command"] = default_test_command
    return (
        "run_test",
        normalized,
        "Normalized initial list_files to run_test because the request already names a Python source path and needs concrete import/test failure evidence first.",
    )


def normalize_project_rename_bootstrap_call(
    name: str,
    arguments: dict[str, Any],
    *,
    tool_calls_this_turn: list[dict[str, Any]],
    requested_tool_names: set[str],
    rename_operations: list[ToolOperation] | None,
) -> tuple[str, dict[str, Any], str | None]:
    if name not in {"list_files", "edit_intent"}:
        return name, arguments, None
    if _has_prior_non_context_tool_call(tool_calls_this_turn):
        return name, arguments, None
    if name == "list_files" and "list_files" in requested_tool_names:
        return name, arguments, None
    if not rename_operations or len(rename_operations) != 1:
        return name, arguments, None
    tool_name, tool_arguments = rename_operations[0]
    if tool_name != "edit_intent" or not isinstance(tool_arguments, dict):
        return name, arguments, None
    if name == "edit_intent":
        target = str(arguments.get("target") or arguments.get("symbol") or "").strip()
        replacement = str(arguments.get("replacement") or arguments.get("new") or "").strip()
        if target != str(tool_arguments.get("target") or "").strip():
            return name, arguments, None
        if replacement != str(tool_arguments.get("replacement") or "").strip():
            return name, arguments, None
    return (
        tool_name,
        dict(tool_arguments),
        f"Normalized initial {name} to edit_intent because the request already specifies a grounded project rename operation.",
    )


def normalize_optional_parameter_bootstrap_call(
    name: str,
    arguments: dict[str, Any],
    *,
    tool_calls_this_turn: list[dict[str, Any]],
    requested_tool_names: set[str],
    optional_parameter_operations: list[ToolOperation] | None,
) -> tuple[str, dict[str, Any], str | None]:
    if name != "search_symbols":
        return name, arguments, None
    if _has_prior_non_context_tool_call(tool_calls_this_turn):
        return name, arguments, None
    if "search_symbols" in requested_tool_names or "read_symbol" in requested_tool_names:
        return name, arguments, None
    if not optional_parameter_operations:
        return name, arguments, None
    tool_name, tool_arguments = optional_parameter_operations[0]
    if tool_name != "edit_intent" or not isinstance(tool_arguments, dict):
        return name, arguments, None
    return (
        tool_name,
        dict(tool_arguments),
        "Normalized initial search_symbols to edit_intent because the request already specifies a grounded optional-parameter update.",
    )


def normalize_unittest_file_command(
    command: str,
    *,
    resolve_path: PathResolver,
    relative_label: PathLabeler,
) -> str | None:
    match = re.match(
        r"^(?P<prefix>(?:\"[^\"]+\"|'[^']+'|[^\s]+)\s+-m\s+unittest)\s+(?P<path>[^\s]+\.py)\s*$",
        command.strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    raw_path = match.group("path").strip("\"'")
    try:
        target = resolve_path(raw_path)
    except Exception:
        return None
    rel = relative_label(target).replace("\\", "/")
    if "/tests/" not in f"/{rel}" and not target.name.startswith("test_") and not target.name.endswith("_test.py"):
        return None
    test_dir = relative_label(target.parent)
    return f"{match.group('prefix')} discover -s {test_dir} -p {target.name}"


def normalize_head_tail_shell_inspection(
    argv: list[str],
    *,
    line_count_for_file: LineCountProvider,
) -> dict[str, Any] | None:
    if not argv:
        return None
    command = argv[0].lower()
    if command not in {"head", "tail"}:
        return None
    count = 10
    path: str | None = None
    index = 1
    if index < len(argv):
        token = argv[index]
        if token == "-n":
            if index + 2 >= len(argv):
                return None
            try:
                count = int(argv[index + 1])
            except ValueError:
                return None
            path = argv[index + 2]
            index += 3
        elif re.fullmatch(r"-\d+", token):
            count = int(token[1:])
            if index + 1 >= len(argv):
                return None
            path = argv[index + 1]
            index += 2
        elif token.startswith("-"):
            return None
        else:
            path = token
            index += 1
    if index != len(argv) or not path or path.startswith("-") or count <= 0:
        return None
    count = min(count, 200)
    if command == "head":
        return {"path": path, "start": 1, "end": count}
    line_count = line_count_for_file(path)
    if line_count is None:
        return None
    start = max(1, line_count - count + 1)
    return {"path": path, "start": start, "end": max(start, line_count)}


def normalize_find_shell_inspection(argv: list[str]) -> tuple[str, dict[str, Any]] | None:
    if len(argv) < 4 or argv[0].lower() != "find":
        return None
    path = argv[1]
    if path.startswith("-"):
        return None
    query: str | None = None
    target_tool = "file_search"
    index = 2
    while index < len(argv):
        token = argv[index]
        if token == "-name":
            if index + 1 >= len(argv) or query is not None:
                return None
            query = argv[index + 1]
            index += 2
            continue
        if token == "-type":
            if index + 1 >= len(argv):
                return None
            raw_kind = argv[index + 1].lower()
            if raw_kind in {"f", "file"}:
                target_tool = "file_search"
            elif raw_kind in {"d", "dir", "directory"}:
                target_tool = "directory_search"
            else:
                return None
            index += 2
            continue
        return None
    if not query or query.startswith("-"):
        return None
    clean_query = query.strip()
    if target_tool == "file_search":
        if clean_query.startswith("*") and clean_query.endswith("*") and len(clean_query) > 2:
            clean_query = clean_query.strip("*")
        elif clean_query.startswith("*.") and len(clean_query) > 2:
            clean_query = clean_query[1:]
    clean_query = clean_query.strip()
    if not clean_query:
        return None
    return target_tool, {"query": clean_query, "path": path, "limit": 100}
