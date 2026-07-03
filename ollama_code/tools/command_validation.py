from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
from typing import Any


WINDOWS_DRIVE_PATH = re.compile(r"^(?P<drive>[A-Za-z]):(?:[\\/](?P<rest>.*))?$")
WSL_MOUNT_PATH = re.compile(r"^/mnt/(?P<drive>[A-Za-z])(?:/(?P<rest>.*))?$")


def command_looks_like_cross_platform_path_input(command: str, *, os_name: str = os.name) -> bool:
    if os_name == "nt" or "\\" not in command:
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


def split_command_text(command: str, *, os_name: str = os.name) -> list[str]:
    posix_mode = os_name != "nt"
    if posix_mode and command_looks_like_cross_platform_path_input(command, os_name=os_name):
        return shlex.split(command, posix=False)
    return shlex.split(command, posix=posix_mode)


def split_command_for_validation(command: str, *, os_name: str = os.name) -> tuple[list[str] | None, dict[str, Any] | None]:
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
        argv = split_command_text(command, os_name=os_name)
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


def command_has_shell_chaining(command: str) -> bool:
    return bool(re.search(r"&&|\|\||[;|<>]", command))


def token_looks_like_path(token: str) -> bool:
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


def command_family(argv: list[str]) -> str | None:
    executable = Path(str(argv[0]).strip().strip("'\"")).name.lower()
    if executable.endswith(".exe"):
        executable = executable[:-4]
    if executable in {
        "git",
        "pytest",
        "ruff",
        "mypy",
        "pyright",
        "tsc",
        "npm",
        "pnpm",
        "yarn",
        "go",
        "cargo",
        "gradle",
        "gradlew",
        "gradlew.bat",
        "cmake",
        "ctest",
        "node",
    }:
        return executable
    if executable in {"python", "python3", "py"}:
        if "-m" in argv:
            return "python"
        return "python_exec"
    return None


def validation_result(*, family: str, valid: bool, reason: str = "", argv: list[str] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"recognized": True, "valid": valid, "family": family}
    if argv is not None:
        result["argv"] = argv
    if reason:
        result["reason"] = reason
    return result
