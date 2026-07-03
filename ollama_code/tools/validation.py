from __future__ import annotations

import configparser
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def toml_tool_section(payload: dict[str, Any], name: str) -> bool:
    tool = payload.get("tool") if isinstance(payload, dict) else None
    return isinstance(tool, dict) and isinstance(tool.get(name), dict)


def ini_has_section(path: Path, prefixes: tuple[str, ...]) -> bool:
    if not path.exists():
        return False
    parser = configparser.ConfigParser()
    try:
        parser.read(path, encoding="utf-8")
    except configparser.Error:
        return False
    return any(section == prefix or section.startswith(prefix + ":") for section in parser.sections() for prefix in prefixes)


def python_typechecker_configured(workspace_root: Path, pyproject: dict[str, Any]) -> bool:
    if (workspace_root / "pyrightconfig.json").exists() or (workspace_root / "basedpyrightconfig.json").exists():
        return True
    return toml_tool_section(pyproject, "pyright") or toml_tool_section(pyproject, "basedpyright")


def requested_validator_file_hints(requested_rel: str) -> dict[str, str]:
    hints = {
        "workflow_file": "",
        "yaml_file": "",
        "shell_script": "",
        "dockerfile": "",
        "markdown_file": "",
        "sql_file": "",
        "schema_file": "",
    }
    if not requested_rel:
        return hints
    requested_suffix = Path(requested_rel).suffix.lower()
    requested_name = Path(requested_rel).name.lower()
    if requested_suffix in {".yml", ".yaml"}:
        hints["yaml_file"] = requested_rel
        if requested_rel.lower().startswith(".github/workflows/"):
            hints["workflow_file"] = requested_rel
    if requested_suffix in {".sh", ".bash"}:
        hints["shell_script"] = requested_rel
    if requested_name == "dockerfile" or requested_name.endswith(".dockerfile"):
        hints["dockerfile"] = requested_rel
    if requested_suffix in {".md", ".markdown"}:
        hints["markdown_file"] = requested_rel
    if requested_suffix == ".sql":
        hints["sql_file"] = requested_rel
    if requested_name.endswith(".schema.json") or requested_name.endswith(".jsonschema"):
        hints["schema_file"] = requested_rel
    return hints


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
