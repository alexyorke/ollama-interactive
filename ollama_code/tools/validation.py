from __future__ import annotations

import configparser
import hashlib
import json
from pathlib import Path
import re
import subprocess
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


def iter_python_test_files(
    *,
    workspace_root: Path,
    iter_code_files: Callable[[Path], Iterable[Path]],
    relative_label: Callable[[Path], str],
) -> list[Path]:
    candidates: list[Path] = []
    for file_path in iter_code_files(workspace_root):
        if file_path.suffix.lower() != ".py":
            continue
        rel = relative_label(file_path).replace("\\", "/")
        if path_label_looks_like_test(rel, file_path.name):
            candidates.append(file_path)
    return candidates


def test_matches_source(
    *,
    test_path: Path,
    source_path: Path,
    symbols: set[str],
    relative_label: Callable[[Path], str],
    python_import_targets: Callable[[Path], Iterable[dict[str, Any]]],
) -> tuple[int, list[str]]:
    text = test_path.read_text(encoding="utf-8", errors="replace")
    imported_paths = {str(item.get("path", "")) for item in python_import_targets(test_path)}
    rel_source = relative_label(source_path)
    return test_source_match_score(
        test_text=text,
        test_stem=test_path.stem,
        imported_paths=imported_paths,
        rel_source=rel_source,
        source_modules=source_module_labels(rel_source),
        source_stem=source_path.stem,
        symbols=symbols,
    )


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


def runnable_test_validator_commands(validators: dict[str, Any], *, limit: int) -> list[str]:
    selected_limit = max(1, int(limit))
    return [
        str(item.get("command"))
        for item in validators.get("validators", [])
        if isinstance(item, dict) and item.get("kind") == "test" and item.get("command")
    ][:selected_limit]


def select_tests_language_validator_result(
    *,
    paths: list[Path],
    commands: list[str],
    symbols: set[str],
    relative_label: Callable[[Path], str],
    limit: int,
) -> dict[str, Any]:
    selected_limit = max(1, int(limit))
    rows = [
        {
            "path": relative_label(path),
            "command": commands[0] if commands else "",
            "score": 1,
            "reason": "language-level validator discovery",
        }
        for path in paths[:selected_limit]
    ]
    return {
        "ok": True,
        "tool": "select_tests",
        "confidence": "low" if not commands else "medium",
        "changed_files": [relative_label(path) for path in paths],
        "changed_symbols": sorted(symbols),
        "test_commands": commands,
        "tests": rows,
        "summary": "Selected language-level test commands from discover_validators." if commands else "No targeted tests found; use configured run_test.",
        "output": "\n".join(commands) if commands else "(no targeted tests found)",
    }


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


def lint_typecheck_target_plan(
    *,
    python_validator_files: Iterable[str],
    python_validator_scopes: Iterable[str],
    typechecker_configured: bool,
    path_looks_like_test: Callable[[str], bool],
    limit: int = 100,
) -> dict[str, Any]:
    file_set = {str(item).replace("\\", "/") for item in python_validator_files if str(item).strip()}
    scope_set = {str(item).replace("\\", "/") for item in python_validator_scopes if str(item).strip()}
    validator_targets = python_validation_targets(
        discovered_files=file_set,
        requested_scopes=scope_set,
        limit=limit,
    )
    collapsed_scopes = collapse_validation_targets(scope_set, limit=limit)
    typechecker_targets = python_typechecker_targets(
        discovered_files=file_set,
        requested_scopes=scope_set,
        limit=limit,
    )
    skipped_reason = ""
    if typechecker_targets and not typechecker_configured:
        if "." not in collapsed_scopes:
            typechecker_targets = []
            skipped_reason = "No pyright/basedpyright config found for focused scope; skipped cold typechecker startup."
        elif file_set and all(path_looks_like_test(label) for label in file_set):
            typechecker_targets = []
            skipped_reason = "No pyright/basedpyright config found for test-only workspace scope; skipped cold typechecker startup."
    return {
        "validator_targets": validator_targets,
        "typechecker_targets": typechecker_targets,
        "typechecker_skipped_reason": skipped_reason,
        "collapsed_python_scopes": collapsed_scopes,
    }


def lint_typecheck_scan_paths(
    *,
    raw_paths: Iterable[str],
    resolve_path: Callable[[str], Path],
    iter_code_files: Callable[[Path], Iterable[Path]],
    relative_label: Callable[[Path], str],
    file_analysis: Callable[[Path], dict[str, Any]],
    shell_script_suffixes: set[str],
    checked: list[str] | None = None,
    diagnostics: list[str] | None = None,
    python_validator_files: set[str] | None = None,
    python_validator_scopes: set[str] | None = None,
    shell_targets: list[str] | None = None,
) -> dict[str, Any]:
    checked = checked if checked is not None else []
    diagnostics = diagnostics if diagnostics is not None else []
    python_validator_files = python_validator_files if python_validator_files is not None else set()
    python_validator_scopes = python_validator_scopes if python_validator_scopes is not None else set()
    shell_targets = shell_targets if shell_targets is not None else []
    seen_shell_targets: set[str] = set(shell_targets)
    for raw_path in raw_paths:
        base = resolve_path(str(raw_path))
        files = iter_code_files(base)
        base_has_python = False
        for file_path in files:
            rel = relative_label(file_path)
            checked.append(rel)
            analysis = file_analysis(file_path)
            suffix = file_path.suffix.lower()
            diagnostic = analysis.get("diagnostic")
            if suffix == ".py":
                base_has_python = True
                python_validator_files.add(rel)
                if diagnostic:
                    diagnostics.append(str(diagnostic))
            elif suffix in shell_script_suffixes:
                if rel not in seen_shell_targets:
                    shell_targets.append(rel)
                    seen_shell_targets.add(rel)
            elif diagnostic:
                diagnostics.append(str(diagnostic))
        if base_has_python:
            python_validator_scopes.add(relative_label(base))
    return {
        "checked": checked,
        "diagnostics": diagnostics,
        "python_validator_files": python_validator_files,
        "python_validator_scopes": python_validator_scopes,
        "shell_targets": shell_targets,
    }


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


def lint_typecheck_run_validators(
    *,
    workspace_root: Path,
    timeout: int,
    checked: list[str],
    diagnostics: list[str],
    validator_commands: list[str],
    validator_targets: list[str],
    typechecker_targets: list[str],
    typechecker_skipped_reason: str,
    shell_targets: list[str],
    phase_timings_ms: dict[str, float],
    ruff_path: str | None,
    typechecker_command: list[str] | None,
    bash_path: str | None,
    run_process: Callable[..., Any],
    collect_process_output: Callable[[Any], str],
    collect_timeout_output: Callable[[subprocess.TimeoutExpired], str],
    timeout_command_text: Callable[[Any], str],
    command_to_text: Callable[[tuple[str, ...]], str],
    truncate_text: Callable[[str], str],
    timer: Callable[[], float],
) -> dict[str, Any]:
    active_phase = "scan_ms"
    active_phase_started = timer()
    try:
        if validator_targets and ruff_path:
            command = ["ruff", "check", "--no-cache", *validator_targets]
            validator_commands.append(command_to_text(tuple(command)))
            active_phase = "ruff_ms"
            active_phase_started = timer()
            completed = run_process(command, cwd=workspace_root, timeout=timeout, shell=False)
            phase_timings_ms["ruff_ms"] = round((timer() - active_phase_started) * 1000, 3)
            if completed.returncode != 0:
                diagnostics.append(truncate_text(collect_process_output(completed)))
        if typechecker_targets and typechecker_command:
            command = [*typechecker_command, *typechecker_targets]
            validator_commands.append(command_to_text(tuple(command)))
            active_phase = "typecheck_ms"
            active_phase_started = timer()
            completed = run_process(command, cwd=workspace_root, timeout=timeout, shell=False)
            phase_timings_ms["typecheck_ms"] = round((timer() - active_phase_started) * 1000, 3)
            if completed.returncode != 0:
                diagnostics.append(truncate_text(collect_process_output(completed)))
        if bash_path:
            for rel in shell_targets[:100]:
                command = [bash_path, "-n", rel]
                validator_commands.append(command_to_text(("bash", "-n", rel)))
                active_phase = "shell_ms"
                active_phase_started = timer()
                completed = run_process(command, cwd=workspace_root, timeout=timeout, shell=False)
                phase_timings_ms["shell_ms"] = round(
                    float(phase_timings_ms["shell_ms"]) + ((timer() - active_phase_started) * 1000),
                    3,
                )
                if completed.returncode != 0:
                    output = collect_process_output(completed) or f"{rel}: bash -n failed"
                    diagnostics.append(truncate_text(output))
    except subprocess.TimeoutExpired as exc:
        phase_timings_ms[active_phase] = round(
            float(phase_timings_ms.get(active_phase, 0.0) or 0.0) + ((timer() - active_phase_started) * 1000),
            3,
        )
        timeout_output = collect_timeout_output(exc)
        timeout_command = timeout_command_text(exc.cmd)
        return {
            "timed_out": True,
            "result": lint_typecheck_timeout_result(
                checked=checked,
                diagnostics=diagnostics,
                validator_commands=validator_commands,
                validator_targets=validator_targets,
                typechecker_targets=typechecker_targets,
                typechecker_skipped_reason=typechecker_skipped_reason,
                phase_timings_ms=phase_timings_ms,
                timeout_seconds=exc.timeout,
                timeout_output=timeout_output,
                timeout_command=timeout_command,
            ),
        }
    return {
        "timed_out": False,
        "diagnostics": diagnostics,
        "validator_commands": validator_commands,
        "phase_timings_ms": phase_timings_ms,
    }


def lint_typecheck_cache_hit_result(cached: dict[str, Any], *, scan_ms: float) -> dict[str, Any]:
    result = dict(cached)
    result["cache_hit"] = True
    result["scan_ms"] = scan_ms
    result["ruff_ms"] = 0.0
    result["typecheck_ms"] = 0.0
    result["shell_ms"] = 0.0
    return result


def lint_typecheck_timeout_result(
    *,
    checked: list[str],
    diagnostics: list[str],
    validator_commands: list[str],
    validator_targets: list[str],
    typechecker_targets: list[str],
    typechecker_skipped_reason: str,
    phase_timings_ms: dict[str, float],
    timeout_seconds: Any,
    timeout_output: str,
    timeout_command: str,
) -> dict[str, Any]:
    timeout_summary = f"Command timed out after {timeout_seconds} seconds."
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


def lint_typecheck_final_result(
    *,
    checked: list[str],
    diagnostics: list[str],
    validator_commands: list[str],
    validator_targets: list[str],
    typechecker_targets: list[str],
    typechecker_skipped_reason: str,
    phase_timings_ms: dict[str, float],
) -> dict[str, Any]:
    return {
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
