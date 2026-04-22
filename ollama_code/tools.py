from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

ApprovalMode = str
AgentRunner = Callable[[dict[str, Any]], dict[str, Any]]


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


class ToolExecutor:
    def __init__(
        self,
        workspace_root: str | Path,
        *,
        approval_mode: ApprovalMode = "ask",
        input_func: Callable[[str], str] = input,
        agent_runner: AgentRunner | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.approval_mode = approval_mode
        self.input_func = input_func
        self.agent_runner = agent_runner

    def set_approval_mode(self, mode: ApprovalMode) -> None:
        self.approval_mode = mode

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "run_agent":
            if self.agent_runner is None:
                return {"ok": False, "tool": name, "summary": "Sub-agent support is not configured."}
            try:
                return self.agent_runner(arguments)
            except TypeError as exc:
                return {"ok": False, "tool": name, "summary": f"Bad arguments for {name}: {exc}"}
            except Exception as exc:  # pragma: no cover - defensive fallback
                return {"ok": False, "tool": name, "summary": f"{name} failed: {exc}"}
        handlers = {
            "list_files": self.list_files,
            "read_file": self.read_file,
            "search": self.search,
            "write_file": self.write_file,
            "replace_in_file": self.replace_in_file,
            "run_shell": self.run_shell,
        }
        handler = handlers.get(name)
        if handler is None:
            return {"ok": False, "tool": name, "summary": f"Unknown tool: {name}"}
        try:
            return handler(**arguments)
        except TypeError as exc:
            return {"ok": False, "tool": name, "summary": f"Bad arguments for {name}: {exc}"}
        except Exception as exc:  # pragma: no cover - defensive fallback
            return {"ok": False, "tool": name, "summary": f"{name} failed: {exc}"}

    def resolve_path(self, raw_path: str | None, *, allow_missing: bool = True) -> Path:
        candidate = self.workspace_root if not raw_path else (self.workspace_root / raw_path)
        resolved = candidate.resolve(strict=False)
        if resolved != self.workspace_root and self.workspace_root not in resolved.parents:
            raise ValueError(f"Path escapes the workspace: {raw_path}")
        if not allow_missing and not resolved.exists():
            raise FileNotFoundError(f"Path does not exist: {raw_path}")
        return resolved

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

    def list_files(self, path: str = ".", max_depth: int = 4, limit: int = 200) -> dict[str, Any]:
        base = self.resolve_path(path, allow_missing=False)
        if base.is_file():
            items = [self.relative_label(base)]
        else:
            items = []
            for root, dirs, files in os.walk(base):
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
        base = self.resolve_path(path, allow_missing=False)
        if shutil.which("rg"):
            command = ["rg", "-n", "--color", "never", "--max-count", str(limit), query, str(base)]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
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
            if not file_path.is_file() or ".git" in file_path.parts:
                continue
            for line_no, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
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
        shell_kwargs: dict[str, Any] = {"shell": True}
        if os.name != "nt":
            shell_kwargs["executable"] = os.environ.get("SHELL", "/bin/sh")
        completed = subprocess.run(
            command,
            cwd=working_dir,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
            **shell_kwargs,
        )
        parts = []
        if completed.stdout.strip():
            parts.append(completed.stdout.strip())
        if completed.stderr.strip():
            parts.append(completed.stderr.strip())
        output = "\n".join(parts) if parts else "(no output)"
        return {
            "ok": completed.returncode == 0,
            "tool": "run_shell",
            "cwd": relative_cwd,
            "exit_code": completed.returncode,
            "output": output,
        }
