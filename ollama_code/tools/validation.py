from __future__ import annotations

import configparser
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable


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


def python_module_label(label: str) -> str:
    rel = str(label or "").replace("\\", "/")
    if rel.endswith(".py"):
        rel = rel[:-3]
    return rel.replace("/", ".")


def source_module_labels(label: str) -> set[str]:
    rel = str(label or "").replace("\\", "/")
    without_suffix = rel[:-3] if rel.endswith(".py") else rel
    modules = {without_suffix.replace("/", ".")}
    if without_suffix.startswith("src/"):
        modules.add(without_suffix[4:].replace("/", "."))
    return modules


def path_label_looks_like_test(label: str, name: str) -> bool:
    rel = str(label or "").replace("\\", "/").lower()
    clean_name = str(name or "").lower()
    return clean_name.startswith("test_") or clean_name.endswith("_test.py") or "/tests/" in rel


def targeted_unittest_command(python_executable: str, test_dir: str, test_name: str) -> str:
    return f"{python_executable} -m unittest discover -s {test_dir} -p {test_name}"


def test_source_match_score(
    *,
    test_text: str,
    test_stem: str,
    imported_paths: set[str],
    rel_source: str,
    source_modules: set[str],
    source_stem: str,
    symbols: set[str],
) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    if rel_source in imported_paths:
        score += 6
        reasons.append(f"imports {rel_source}")
    for module in source_modules:
        if re.search(rf"\b(?:from|import)\s+{re.escape(module)}\b", test_text):
            score += 5
            reasons.append(f"imports module {module}")
            break
    if source_stem and source_stem.lower() in test_stem.lower():
        score += 3
        reasons.append(f"filename matches {source_stem}")
    matched_symbols = sorted(symbol for symbol in symbols if symbol and re.search(rf"\b{re.escape(symbol)}\b", test_text))
    if matched_symbols:
        score += min(4, len(matched_symbols) * 2)
        reasons.append("mentions " + ", ".join(matched_symbols[:4]))
    return score, reasons


def run_test_needs_command_recovery(
    result: dict[str, Any],
    *,
    selected_command: str = "",
    missing_dependency_name: str = "",
) -> bool:
    validation = result.get("validation")
    if isinstance(validation, dict) and validation.get("valid") is False:
        return True
    error_class = str(result.get("error_class") or "").strip().lower()
    if error_class in {"command_not_found", "invalid_args", "path_missing", "cwd_missing"}:
        return True
    if error_class == "missing_dependency":
        missing = str(result.get("missing_dependency") or missing_dependency_name or "").strip()
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


def preferred_test_validator_command(
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


def lint_typecheck_file_analysis(
    *,
    file_path: Path,
    rel: str,
    workspace_root: Path,
    cache: dict[str, dict[str, Any]],
    timeout: int,
    node_available: bool | None,
    run_process: Callable[..., Any],
    collect_process_output: Callable[[Any], str],
    truncate_text: Callable[[str], str],
    python_syntax_diagnostic: Callable[[Path, str], str | None],
    tree_sitter_language_for_path: Callable[[Path], Any],
    tree_sitter_syntax_diagnostic: Callable[[Path, str], str | None],
) -> dict[str, Any]:
    suffix = file_path.suffix.lower()
    stat = file_path.stat()
    signature = {"mtime_ns": int(stat.st_mtime_ns), "size": int(stat.st_size)}
    cached = cache.get(rel)
    if (
        isinstance(cached, dict)
        and cached.get("signature") == signature
        and cached.get("suffix") == suffix
        and cached.get("node_available") == node_available
    ):
        analysis = cached.get("analysis")
        if isinstance(analysis, dict):
            return dict(analysis)
    analysis: dict[str, Any] = {"suffix": suffix, "diagnostic": None}
    if suffix == ".py":
        text = file_path.read_text(encoding="utf-8", errors="replace")
        analysis["diagnostic"] = python_syntax_diagnostic(file_path, text)
    elif suffix in {".js", ".jsx"} and node_available:
        completed = run_process(["node", "--check", str(file_path)], cwd=workspace_root, timeout=timeout, shell=False)
        if completed.returncode != 0:
            analysis["diagnostic"] = truncate_text(collect_process_output(completed))
    elif tree_sitter_language_for_path(file_path) is not None:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        analysis["diagnostic"] = tree_sitter_syntax_diagnostic(file_path, text)
    cache[rel] = {
        "signature": signature,
        "suffix": suffix,
        "node_available": node_available,
        "analysis": dict(analysis),
    }
    return analysis
