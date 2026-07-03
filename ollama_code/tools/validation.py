from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def collapse_validation_targets(labels: Iterable[str], *, limit: int = 100) -> list[str]:
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


def python_validation_targets(
    *,
    discovered_files: Iterable[str],
    requested_scopes: Iterable[str],
    limit: int = 100,
    max_explicit_files: int = 20,
) -> list[str]:
    file_targets = collapse_validation_targets(discovered_files, limit=limit)
    if not file_targets:
        return []
    scope_targets = collapse_validation_targets(requested_scopes, limit=limit)
    if "." in scope_targets:
        return ["."]
    if len(file_targets) <= max_explicit_files:
        return file_targets
    return scope_targets or file_targets


def python_typechecker_targets(
    *,
    discovered_files: Iterable[str],
    requested_scopes: Iterable[str],
    limit: int = 100,
) -> list[str]:
    file_targets = collapse_validation_targets(discovered_files, limit=limit)
    if not file_targets:
        return []
    scope_targets = collapse_validation_targets(requested_scopes, limit=limit)
    if "." in scope_targets:
        return ["."]
    if len(scope_targets) == 1 and not scope_targets[0].endswith(".py"):
        return scope_targets
    return file_targets


def lint_typecheck_cache_key(
    *,
    workspace_root: Path,
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
            stat = (workspace_root / label).stat()
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
        path = workspace_root / label
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
