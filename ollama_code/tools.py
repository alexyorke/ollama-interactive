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
import threading
import time
from pathlib import Path
from typing import Any, Callable

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
        "arguments": {"path": "file or directory", "max_symbols": "int, default 120"},
        "description": "Show compact code symbols and line ranges without function bodies.",
    },
    {
        "name": "read_symbol",
        "arguments": {"path": "code file", "symbol": "name or qualified name", "include_context": "lines around symbol, default 2"},
        "description": "Read one code symbol body by AST/definition range.",
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


def format_compact_tool_help() -> str:
    signatures = {
        "list_files": "list_files(path='.',depth=4,limit=200)",
        "read_file": "read_file(path,start=1,end=200)",
        "search": "search(query,path='.',limit=100)",
        "search_symbols": "search_symbols(query,path='.',limit=50)",
        "code_outline": "code_outline(path,max_symbols=120)",
        "read_symbol": "read_symbol(path,symbol,context=2)",
        "write_file": "write_file(path,content)",
        "replace_in_file": "replace_in_file(path,old,new,all=false,whole_word=false)",
        "run_shell": "run_shell(command,cwd='.',timeout=30)",
        "run_test": "run_test(command?,cwd='.',timeout=1200)",
        "git_status": "git_status(path?)",
        "git_diff": "git_diff(path?,cached=false,context=3)",
        "git_commit": "git_commit(message,add_all=true)",
        "run_agent": "run_agent(prompt,model?,approval?,rounds?)",
    }
    return "\n".join(signatures[tool["name"]] for tool in TOOL_DESCRIPTIONS)


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
            command = ["rg", "-n", "--color", "never", "--max-count", str(limit), query, str(base)]
            result = self._run_process(command, cwd=self.workspace_root, timeout=30)
            output = result.stdout.strip() or result.stderr.strip() or "(no matches)"
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
            tree = ast.parse(text)
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

    def code_outline(self, path: str, max_symbols: int = 120) -> dict[str, Any]:
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

    def read_symbol(self, path: str, symbol: str, include_context: int = 2) -> dict[str, Any]:
        self._check_interrupted()
        target = self.resolve_path(path, allow_missing=False)
        if target.is_dir():
            return {"ok": False, "tool": "read_symbol", "summary": f"{path} is a directory."}
        if not self._is_code_file(target):
            return {"ok": False, "tool": "read_symbol", "summary": f"{path} is not a supported code file."}
        symbols, text, _ = self._code_symbols(target)
        needle = symbol.strip()
        exact = [
            item
            for item in symbols
            if item["qualname"] == needle or item["name"] == needle
        ]
        if not exact:
            lowered = needle.lower()
            exact = [item for item in symbols if lowered in str(item["qualname"]).lower() or lowered in str(item["name"]).lower()]
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
        existing = target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""
        relative_path = self.relative_label(target)
        preview = self._diff_preview(relative_path, existing, content)
        approved, reason = self._approve_mutation(f"Write {relative_path}?", preview)
        if not approved:
            return {"ok": False, "tool": "write_file", "path": relative_path, "summary": reason}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {
            "ok": True,
            "tool": "write_file",
            "path": relative_path,
            "summary": f"Wrote {relative_path}.",
            "diff": preview,
        }

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

    def replace_in_file(
        self,
        path: str,
        old: str,
        new: str,
        replace_all: bool = False,
        match_whole_word: bool = False,
    ) -> dict[str, Any]:
        target = self.resolve_path(path, allow_missing=False)
        original = target.read_text(encoding="utf-8", errors="replace")
        if match_whole_word:
            pattern = re.compile(rf"\b{re.escape(old)}\b")
            count = len(pattern.findall(original))
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
            updated = pattern.sub(new, original, count=limit)
        else:
            updated = original.replace(old, new) if replace_all else original.replace(old, new, 1)
        relative_path = self.relative_label(target)
        preview = self._diff_preview(relative_path, original, updated)
        approved, reason = self._approve_mutation(f"Replace text in {relative_path}?", preview)
        if not approved:
            return {"ok": False, "tool": "replace_in_file", "path": relative_path, "summary": reason}
        target.write_text(updated, encoding="utf-8")
        replaced_count = count if replace_all else 1
        return {
            "ok": True,
            "tool": "replace_in_file",
            "path": relative_path,
            "summary": f"Replaced {replaced_count} occurrence(s) in {relative_path}.",
            "diff": preview,
        }

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
