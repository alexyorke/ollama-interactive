from __future__ import annotations

import ast
import difflib
import inspect
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import textwrap
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from ollama_code.interrupts import OperationInterrupted

ApprovalMode = str
AgentRunner = Callable[[dict[str, Any]], dict[str, Any]]
WINDOWS_DRIVE_PATH = re.compile(r"^(?P<drive>[A-Za-z]):(?:[\\/](?P<rest>.*))?$")
WSL_MOUNT_PATH = re.compile(r"^/mnt/(?P<drive>[A-Za-z])(?:/(?P<rest>.*))?$")
CODE_FILE_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".cs", ".rb", ".php"}
SKIP_CODE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "venv",
}
DENY_MUTATION_DIRS = {".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ollama-code"}


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
        "name": "repo_index_search",
        "arguments": {"query": "natural query or symbol", "path": "relative path, default .", "limit": "int, default 10"},
        "description": "Search code with compact ranked snippets instead of whole files.",
    },
    {
        "name": "find_implementation_target",
        "arguments": {"test_path": "optional test file", "output": "optional failing test output/traceback", "limit": "int, default 12"},
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
        "name": "lint_typecheck",
        "arguments": {"paths": "path or list of paths, default .", "command": "optional lint/type command", "timeout": "seconds, default 120"},
        "description": "Run deterministic syntax/lint/type checks with exact file lines.",
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
        "name": "generate_tests_from_spec",
        "arguments": {"target_symbol": "symbol under test", "behavior": "expected behavior", "test_path": "optional output test path", "apply": "bool, default false"},
        "description": "Generate a compact pytest test patch from a spec; preview by default, apply only when requested.",
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


def format_tool_help() -> str:
    lines: list[str] = []
    for tool in TOOL_DESCRIPTIONS:
        arg_text = ", ".join(f"{key}: {value}" for key, value in tool["arguments"].items())
        lines.append(f"- {tool['name']}({arg_text}) -> {tool['description']}")
    return "\n".join(lines)


def format_compact_tool_help(tool_names: Iterable[str] | None = None) -> str:
    signatures = {
        "list_files": "list_files(path='.',depth=4,limit=200)",
        "read_file": "read_file(path,start=1,end=200)",
        "search": "search(query,path='.',limit=100)",
        "search_symbols": "search_symbols(query,path='.',limit=50)",
        "code_outline": "code_outline(path,max_symbols=120)",
        "read_symbol": "read_symbol(path,symbol,context=2)",
        "repo_index_search": "repo_index_search(query,path='.',limit=10)",
        "find_implementation_target": "find_implementation_target(test_path?,output?,limit=12)",
        "diagnose_test_failure": "diagnose_test_failure(output,path='.',limit=12)",
        "run_function_probe": "run_function_probe(module,expressions,function?,timeout=30)",
        "call_graph": "call_graph(path,symbol?,limit=40)",
        "lint_typecheck": "lint_typecheck(paths='.',command?,timeout=120)",
        "replace_symbol": "replace_symbol(path,symbol,content)",
        "replace_symbols": 'replace_symbols(path,replacements=[{"symbol":"f","content":"def f():\\n    return ..."}])',
        "write_file": "write_file(path,content)",
        "replace_in_file": "replace_in_file(path,old,new,all=false,whole_word=false)",
        "apply_structured_edit": 'apply_structured_edit(operation={"op":"rename_symbol","path":"a.py","old":"x","new":"y"})',
        "generate_tests_from_spec": "generate_tests_from_spec(target_symbol,behavior,test_path?,apply=false)",
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
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.approval_mode = approval_mode
        self.input_func = input_func
        self.agent_runner = agent_runner
        self.default_test_command = test_command.strip() if isinstance(test_command, str) and test_command.strip() else None
        self._interrupt_event: threading.Event | None = None
        self._initial_dirty_paths = self._git_dirty_paths()

    def set_approval_mode(self, mode: ApprovalMode) -> None:
        self.approval_mode = mode

    def set_interrupt_event(self, event: threading.Event | None) -> None:
        self._interrupt_event = event

    def set_test_command(self, command: str | None) -> None:
        self.default_test_command = command.strip() if isinstance(command, str) and command.strip() else None

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

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
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
            "search_symbols": self.search_symbols,
            "code_outline": self.code_outline,
            "read_symbol": self.read_symbol,
            "repo_index_search": self.repo_index_search,
            "find_implementation_target": self.find_implementation_target,
            "diagnose_test_failure": self.diagnose_test_failure,
            "run_function_probe": self.run_function_probe,
            "call_graph": self.call_graph,
            "lint_typecheck": self.lint_typecheck,
            "replace_symbol": self.replace_symbol,
            "replace_symbols": self.replace_symbols,
            "apply_structured_edit": self.apply_structured_edit,
            "generate_tests_from_spec": self.generate_tests_from_spec,
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
            return {"ok": False, "tool": name, "summary": f"{name} failed: {exc}"}

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
                dirs[:] = sorted(d for d in dirs if not d.startswith("."))
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
            command.extend([query, str(base)])
            result = self._run_process(command, cwd=self.workspace_root, timeout=30)
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
        for file_path in sorted(base.rglob("*")):
            self._check_interrupted()
            if not file_path.is_file() or ".git" in file_path.parts:
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
            "output": "\n".join(matches) if matches else "(no matches)",
        }

    def _is_code_file(self, path: Path) -> bool:
        return path.suffix.lower() in CODE_FILE_SUFFIXES and not any(part in SKIP_CODE_DIRS for part in path.parts)

    def _iter_code_files(self, base: Path, *, limit: int = 200) -> list[Path]:
        if base.is_file():
            return [base] if self._is_code_file(base) else []
        files: list[Path] = []
        for root, dirs, names in os.walk(base):
            self._check_interrupted()
            dirs[:] = sorted(directory for directory in dirs if directory not in SKIP_CODE_DIRS)
            if len(files) >= limit:
                break
            root_path = Path(root)
            for name in sorted(names):
                if len(files) >= limit:
                    break
                file_path = root_path / name
                if not self._is_code_file(file_path):
                    continue
                files.append(file_path)
        return files

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
        needle = symbol.strip()
        exact = [item for item in symbols if item["qualname"] == needle or item["name"] == needle]
        if exact:
            return exact
        lowered = needle.lower()
        return [item for item in symbols if lowered in str(item["qualname"]).lower() or lowered in str(item["name"]).lower()]

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

    def repo_index_search(self, query: str, path: str = ".", limit: int = 10) -> dict[str, Any]:
        self._check_interrupted()
        base = self.resolve_path(path, allow_missing=False)
        terms = [term.lower() for term in re.findall(r"[A-Za-z_][\w.:-]*|\d+", query)]
        if not terms:
            return {"ok": False, "tool": "repo_index_search", "summary": "repo_index_search requires a non-empty query."}
        results: list[dict[str, Any]] = []
        for file_path in self._iter_code_files(base, limit=500):
            text = file_path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            symbols, _, _ = self._code_symbols(file_path)
            score = 0
            snippets: list[str] = []
            rel = self.relative_label(file_path)
            haystack_path = rel.lower()
            for term in terms:
                if term in haystack_path:
                    score += 4
            for symbol in symbols:
                haystack = f"{symbol.get('qualname', '')} {symbol.get('signature', '')}".lower()
                hits = sum(1 for term in terms if term in haystack)
                if hits:
                    score += 20 * hits
                    start = max(1, int(symbol["start"]) - 1)
                    end = min(len(lines), int(symbol["start"]) + 1)
                    snippets.append(f"{rel}:{symbol['start']}-{symbol['end']} {symbol['kind']} {symbol['qualname']}: {symbol['signature']}")
                    for line_no in range(start, end + 1):
                        snippets.append(f"{rel}:{line_no}: {lines[line_no - 1].strip()[:160]}")
            for line_no, line in enumerate(lines, start=1):
                lowered = line.lower()
                hits = sum(1 for term in terms if term in lowered)
                if hits:
                    score += hits
                    if len(snippets) < 6:
                        snippets.append(f"{rel}:{line_no}: {line.strip()[:180]}")
            if score:
                results.append({"score": score, "path": rel, "snippets": snippets[:6]})
        ranked = sorted(results, key=lambda item: (-int(item["score"]), str(item["path"])))[: max(1, int(limit))]
        output: list[str] = []
        for item in ranked:
            output.append(f"{item['path']} score={item['score']}")
            output.extend(f"  {snippet}" for snippet in item["snippets"])
        return {
            "ok": True,
            "tool": "repo_index_search",
            "path": self.relative_label(base),
            "count": len(ranked),
            "output": "\n".join(output) if output else "(no ranked snippets)",
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

    def find_implementation_target(
        self,
        test_path: str | None = None,
        output: str | None = None,
        traceback: str | None = None,
        limit: int = 12,
    ) -> dict[str, Any]:
        self._check_interrupted()
        targets: list[dict[str, Any]] = []
        raw_output = "\n".join(part for part in [output, traceback] if isinstance(part, str) and part.strip())
        if raw_output:
            targets.extend(self._traceback_targets(raw_output))
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
        assertions = re.findall(r"(?m)^\s*E\s+assert\s+(.+)$", text)
        exceptions = re.findall(r"(?m)^(?:E\s+)?([A-Za-z_][\w.]+(?:Error|Exception)):\s*(.+)$", text)
        expected_actual: list[str] = []
        for assertion in assertions:
            if "==" in assertion:
                left, right = assertion.split("==", 1)
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
        targets = self.find_implementation_target(output=text, limit=limit)
        lines: list[str] = []
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
            files = [base] if base.is_file() else list(base.rglob("*.py"))
            for file_path in files:
                if not file_path.is_file() or any(part in SKIP_CODE_DIRS for part in file_path.parts):
                    continue
                rel = self.relative_label(file_path)
                checked.append(rel)
                text = file_path.read_text(encoding="utf-8", errors="replace")
                diagnostic = self._python_syntax_diagnostic(file_path, text)
                if diagnostic:
                    diagnostics.append(diagnostic)
        return {
            "ok": not diagnostics,
            "tool": "lint_typecheck",
            "checked": checked,
            "diagnostics": diagnostics,
            "output": "\n".join(diagnostics) if diagnostics else f"syntax ok: {len(checked)} Python file(s)",
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

    def apply_structured_edit(self, operation: dict[str, Any] | str) -> dict[str, Any]:
        payload = self._operation_payload(operation)
        if payload is None:
            return {"ok": False, "tool": "apply_structured_edit", "summary": "operation must be a JSON object."}
        op = str(payload.get("op") or payload.get("operation") or "").strip().lower()
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
        if non_empty == 0 or prefixed < max(1, non_empty // 2):
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
        if match_whole_word:
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

    def run_shell(self, command: str, cwd: str = ".", timeout: int = 30) -> dict[str, Any]:
        working_dir = self.resolve_path(cwd, allow_missing=False)
        relative_cwd = self.relative_label(working_dir)
        approved, reason = self._approve_shell(command, relative_cwd)
        if not approved:
            return {"ok": False, "tool": "run_shell", "cwd": relative_cwd, "summary": reason}
        run_args: str | list[str] = command
        shell_kwargs: dict[str, Any] = {"shell": True}
        if os.name == "nt":
            powershell = self._windows_powershell()
            if powershell and self._looks_like_powershell_command(command):
                run_args = [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command]
                shell_kwargs = {"shell": False}
        completed = self._run_process(run_args, cwd=working_dir, timeout=timeout, **shell_kwargs)
        output = self._collect_process_output(completed)
        return {
            "ok": completed.returncode == 0,
            "tool": "run_shell",
            "cwd": relative_cwd,
            "exit_code": completed.returncode,
            "output": output,
        }

    def run_test(self, command: str | None = None, cwd: str = ".", timeout: int = 1200) -> dict[str, Any]:
        selected_command = command.strip() if isinstance(command, str) and command.strip() else self.default_test_command
        if not selected_command:
            return {
                "ok": False,
                "tool": "run_test",
                "summary": "No test command is configured. Set --test-cmd or pass a command to run_test.",
            }
        result = self.run_shell(selected_command, cwd=cwd, timeout=timeout)
        result["tool"] = "run_test"
        result["command"] = selected_command
        return result
