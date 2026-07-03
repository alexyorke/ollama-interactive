from __future__ import annotations

import ast
from dataclasses import dataclass
import re
import textwrap
from typing import Any, Callable


@dataclass(frozen=True)
class CliFeatureCapabilities:
    has_stats: bool = False
    has_priority_filter: bool = False
    has_json_flag: bool = False
    has_limit_flag: bool = False
    has_due_before: bool = False

    def any(self) -> bool:
        return bool(
            self.has_stats
            or self.has_priority_filter
            or self.has_json_flag
            or self.has_limit_flag
            or self.has_due_before
        )


def _request_is_cli_flag_bundle(request_text: str) -> bool:
    return bool(
        re.search(r"--[A-Za-z0-9][A-Za-z0-9-]*", request_text)
        and re.search(r"\b(?:cli|command|option|flag|parser|argparse)\b", request_text, flags=re.IGNORECASE)
    )


def typed_cli_flag_protocol_enabled(*, request_text: str, request_is_cli_flag_bundle: bool | None = None) -> bool:
    return bool(request_is_cli_flag_bundle if request_is_cli_flag_bundle is not None else _request_is_cli_flag_bundle(request_text))


def request_is_cli_flag_bundle(request_text: str) -> bool:
    return _request_is_cli_flag_bundle(request_text)


def request_looks_like_python_test_driven_repair(
    *,
    request_text: str,
    session_memory_request: bool,
    mutation_required: bool,
    test_run_required: bool,
    required_tool_names: set[str],
    forbidden_tool_names: set[str],
    default_test_command_configured: bool,
    requested_mutation_paths: set[str],
    path_looks_like_test_file: Callable[[str], bool],
) -> bool:
    if session_memory_request or not mutation_required or not test_run_required:
        return False
    if required_tool_names or forbidden_tool_names:
        return False
    if not default_test_command_configured:
        return False
    lowered = request_text.lower()
    source_paths = [
        path for path in requested_mutation_paths if path.endswith(".py") and not path_looks_like_test_file(path)
    ]
    doc_or_aux_paths = [
        path
        for path in requested_mutation_paths
        if path_looks_like_test_file(path) or path.startswith("docs/") or path.endswith((".md", ".rst", ".txt"))
    ]
    if len(source_paths) > 1 or doc_or_aux_paths:
        return False
    if re.search(r"\b(?:refactor|rename|callsites?|docs?|public api|api)\b", lowered) and (
        len(source_paths) == 1 or "docs" in lowered or "callsite" in lowered
    ):
        return False
    if source_paths:
        return True
    return bool(
        re.search(
            r"\b(?:python exercise|from the tests?|read source and tests|read tests and source|implement .*tests?|fix .*tests?)\b",
            lowered,
        )
    )


def spec_guided_repair_has_actionable_spec(
    *,
    source_text: str,
    quick_spec: dict[str, Any],
    failed_output: str,
    split_test_example: Callable[[str], tuple[str, str, str]],
    test_spec_call_name: Callable[[ast.Call], str],
) -> bool:
    quick_examples = [item for item in list(quick_spec.get("examples") or []) if isinstance(item, dict)]
    quick_stubs = [item for item in list(quick_spec.get("stubs") or []) if str(item).strip()]
    quick_definitions = [item for item in list(quick_spec.get("definitions") or []) if isinstance(item, dict)]
    quick_example_text = "\n".join(str(item.get("example") or "") for item in quick_examples)
    has_structured_behavior_constraints = " matches " in quick_example_text or " != " in quick_example_text
    has_import_failure = "ModuleNotFoundError" in failed_output or "ImportError" in failed_output
    has_string_transform_hints = bool(quick_spec.get("string_transform_hints"))
    has_definition_risks = any(list(item.get("risks") or []) for item in quick_definitions)
    source_line_count = len(source_text.splitlines())
    small_module = source_line_count <= 220 and len(quick_definitions) <= 8
    literal_example_count = 0
    if small_module and len(quick_definitions) == 1:
        target_names = {
            str(quick_definitions[0].get("name") or "").strip(),
            str(quick_definitions[0].get("symbol") or "").strip(),
        } - {""}
        for item in quick_examples:
            if str(item.get("symbol") or "").strip() not in target_names:
                continue
            kind, expr, expected = split_test_example(str(item.get("example") or ""))
            if kind != "value":
                continue
            try:
                parsed = ast.parse(expr, mode="eval")
                ast.literal_eval(expected)
            except (SyntaxError, ValueError):
                continue
            call = parsed.body
            if not isinstance(call, ast.Call) or call.keywords:
                continue
            if test_spec_call_name(call) not in target_names:
                continue
            try:
                [ast.literal_eval(arg) for arg in call.args]
            except (SyntaxError, ValueError):
                continue
            literal_example_count += 1
    has_small_literal_example_repair = small_module and len(quick_definitions) == 1 and literal_example_count >= 1
    if has_import_failure or has_structured_behavior_constraints or has_string_transform_hints or has_definition_risks:
        return True
    if quick_stubs:
        return True
    return small_module and (len(quick_examples) >= 4 or has_small_literal_example_repair)


def request_likely_import_repair(request_text: str, source_text: str) -> bool:
    lowered = request_text.lower()
    if "import" not in lowered:
        return False
    if not any(token in lowered for token in ("bug", "fix", "repair", "module", "package")):
        return False
    return bool(re.search(r"(?m)^\s*(?:from\s+\S+\s+import\s+|import\s+\S+)", source_text))


def spec_guided_repair_enabled(*, disable_spec_guided_repair: bool) -> bool:
    return not disable_spec_guided_repair


def client_allows_preemptive_mechanical_repair(
    *,
    disable_spec_guided_repair: bool,
    scripted_responses: object,
) -> bool:
    if disable_spec_guided_repair:
        return False
    return not isinstance(scripted_responses, list) or not scripted_responses


def effective_repair_test_command(
    *,
    failed_run_test_result: dict[str, Any] | None = None,
    run_test_arguments: dict[str, Any] | None = None,
    default_test_command: str = "",
) -> str:
    if isinstance(failed_run_test_result, dict) and failed_run_test_result.get("recovered") is True:
        recovered_command = str(failed_run_test_result.get("command") or "").strip()
        original_command = str(failed_run_test_result.get("original_command") or "").strip()
        if recovered_command and recovered_command != original_command:
            return recovered_command
    raw_command = (run_test_arguments or {}).get("command")
    if isinstance(raw_command, str) and raw_command.strip():
        return raw_command.strip()
    return str(default_test_command or "").strip()


def repair_state_spec_guided_paths(
    *,
    source_path: str,
    test_candidates: list[str],
    path_looks_like_test_file: Callable[[str], bool],
) -> tuple[str, str] | None:
    normalized_source = _normalize_repo_path(source_path)
    if not normalized_source or not normalized_source.endswith(".py") or path_looks_like_test_file(normalized_source):
        return None
    for test_path in test_candidates:
        normalized_test = _normalize_repo_path(test_path)
        if normalized_test.endswith(".py") and path_looks_like_test_file(normalized_test):
            return normalized_source, normalized_test
    return None


def focused_python_repair_test_score(
    *,
    source_stem: str,
    test_name: str,
    rel_source: str,
    implementation_targets: list[dict[str, Any]],
) -> int:
    score = 0
    if source_stem.lower() in test_name.lower():
        score += 20
    if any(isinstance(item, dict) and str(item.get("path") or "").strip() == rel_source for item in implementation_targets):
        score += 100
    return score


def select_focused_python_repair_test(candidates: list[tuple[int, str]]) -> str | None:
    if not candidates:
        return None
    ordered = sorted(candidates, key=lambda item: (-item[0], item[1]))
    return ordered[0][1]


def normalize_repair_strategy_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    decision = payload if isinstance(payload, dict) else {}
    strategy = str(decision.get("strategy", "")).strip().lower()
    if strategy not in {"spec_guided_repair", "normal_loop"}:
        strategy = "normal_loop"
    notes = [str(item).strip() for item in list(decision.get("notes") or []) if isinstance(item, str) and str(item).strip()]
    return {
        "strategy": strategy,
        "reason": str(decision.get("reason", "")).strip(),
        "notes": notes,
    }


def spec_guided_repair_candidate_models(
    *,
    primary_model: str,
    verifier_model: str | None,
    available_models: set[str],
    max_attempts: int,
) -> list[str]:
    models = [primary_model]
    for candidate in (verifier_model,):
        if not candidate or candidate in models:
            continue
        if available_models and candidate not in available_models:
            continue
        models.append(candidate)
        if len(models) >= max_attempts:
            break
    while len(models) < max_attempts:
        models.append(primary_model)
    return models


def extract_candidate_python_source(text: str) -> str:
    raw = text.strip()
    if not raw:
        return ""
    fence = re.search(r"```(?:python|py)?\s*(?P<code>.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        raw = fence.group("code").strip()
    lines = raw.splitlines()
    start = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ", "def ", "class ", "@")):
            start = index
            break
    candidate = "\n".join(lines[start:]).strip()
    if not re.search(r"^\s*(?:def|class)\s+", candidate, flags=re.MULTILINE):
        return ""
    return candidate + "\n"


def related_test_source_facts(source_path: str) -> dict[str, Any]:
    if not source_path:
        return {"source": "", "stem": "", "parts": [], "source_candidates": set(), "test_file_name": ""}
    source = source_path.replace("\\", "/")
    stem = source.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    without_suffix = source.rsplit(".", 1)[0]
    parts = without_suffix.split("/")
    source_candidates = {source, stem}
    if len(parts) > 1:
        source_candidates.add(".".join(parts[-2:]))
        source_candidates.add(".".join(parts))
    source_candidates.discard(".py")
    source_candidates.discard("")
    return {
        "source": source,
        "stem": stem,
        "parts": parts,
        "source_candidates": source_candidates,
        "test_file_name": f"test_{stem}.py",
    }


def related_test_matches_source(
    *,
    source_facts: dict[str, Any],
    imports: set[str],
    test_name: str,
) -> bool:
    stem = str(source_facts.get("stem") or "")
    parts = [str(item) for item in list(source_facts.get("parts") or [])]
    source = str(source_facts.get("source") or "")
    source_candidates = set(source_facts.get("source_candidates") or set())
    candidates = {stem, source.rsplit("/", 1)[-1], ".".join(parts[-2:]), ".".join(parts)}
    if source_candidates.intersection(imports) or any(candidate in imports for candidate in candidates):
        return True
    return not imports and stem.lower() in test_name.lower()


def select_related_tests_for_source(*, related: list[str], source_facts: dict[str, Any]) -> list[str]:
    if not related:
        return []
    if len(related) == 1:
        return related
    test_file_name = str(source_facts.get("test_file_name") or "")
    stem = str(source_facts.get("stem") or "")
    for item in related:
        if item.replace("\\", "/").rsplit("/", 1)[-1] == test_file_name:
            return [item]
    for item in related:
        if stem.lower() in item.replace("\\", "/").rsplit("/", 1)[-1].lower():
            return [item]
    return related[:1]


def package_relative_import_rewrite_source(
    *,
    source_text: str,
    local_module_exists: Callable[[str], bool],
) -> str | None:
    changed_lines: list[str] = []
    changed = False
    for line in source_text.splitlines():
        if line.lstrip().startswith("from ") and " import " in line:
            match = re.match(r"^\s*from\s+([A-Za-z_][A-Za-z0-9_]*)\s+import\s+(.+?)\s*$", line)
            if match:
                module = match.group(1).strip()
                rest = match.group(2).strip()
                if local_module_exists(module):
                    changed_lines.append(f"from .{module} import {rest}")
                    changed = True
                    continue
        if re.match(r"^\s*import\s+[A-Za-z_][A-Za-z0-9_]*\s*(?:as\s+[\w_]+)?\s*$", line):
            names = re.match(r"^\s*import\s+(.+?)\s*$", line)
            if names:
                import_items = [item.strip() for item in names.group(1).split(",")]
                rewritten_items = []
                did_rewrite = False
                for item in import_items:
                    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", item):
                        rewritten_items.append(item)
                        continue
                    alias_match = re.match(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?:\s+as\s+(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\s*)?$", item)
                    if alias_match is None:
                        rewritten_items.append(item)
                        continue
                    name = alias_match.group("name")
                    alias = alias_match.group("alias")
                    if local_module_exists(name):
                        rewritten_items.append(f".{name}" if not alias else f".{name} as {alias}")
                        did_rewrite = True
                    else:
                        rewritten_items.append(item)
                if did_rewrite:
                    changed_lines.append("import " + ", ".join(rewritten_items))
                    changed = True
                    continue
        changed_lines.append(line)
    if not changed:
        return None
    candidate = "\n".join(changed_lines)
    if not candidate.endswith("\n"):
        candidate += "\n"
    return candidate


def function_body_is_stub_like_python_repair(body: list[ast.stmt]) -> bool:
    statements = [
        node
        for node in body
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))
    ]
    if not statements:
        return True
    if len(statements) != 1:
        return False
    node = statements[0]
    if isinstance(node, ast.Pass):
        return True
    if isinstance(node, ast.Return):
        return node.value is None or (isinstance(node.value, ast.Constant) and node.value.value is None)
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
        value = node.value.value
        return value is Ellipsis or (
            isinstance(value, str)
            and re.search(r"\b(?:todo|stub|implement|your code)\b", value, flags=re.IGNORECASE) is not None
        )
    if isinstance(node, ast.Raise):
        raised = node.exc
        if isinstance(raised, ast.Call):
            raised = raised.func
        return isinstance(raised, ast.Name) and raised.id == "NotImplementedError"
    return False


def text_is_stub_like_python_repair(text: str) -> bool:
    stripped = textwrap.dedent(text or "").strip()
    if not stripped:
        return True
    meaningful_lines = [
        line.strip()
        for line in stripped.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not meaningful_lines:
        return True
    if all(line in {"pass", "...", "return None", "raise NotImplementedError", "raise NotImplementedError()"} for line in meaningful_lines):
        return True
    try:
        tree = ast.parse(stripped)
    except SyntaxError:
        return False
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return bool(functions) and all(function_body_is_stub_like_python_repair(list(node.body)) for node in functions)


def edit_payload_is_stub_like_repair(name: str, arguments: dict[str, Any]) -> bool:
    path = str(arguments.get("path") or arguments.get("file") or arguments.get("filename") or "").strip()
    if path and not path.replace("\\", "/").endswith(".py"):
        return False
    values: list[str] = []
    if name == "edit_intent":
        value = arguments.get("replacement")
        if isinstance(value, str):
            values.append(value)
    elif name == "replace_symbol":
        value = arguments.get("content")
        if isinstance(value, str):
            values.append(value)
    elif name == "replace_symbols":
        replacements = arguments.get("replacements")
        if isinstance(replacements, list):
            values.extend(str(item.get("content")) for item in replacements if isinstance(item, dict) and isinstance(item.get("content"), str))
    elif name == "write_file":
        value = arguments.get("content")
        if isinstance(value, str):
            values.append(value)
    elif name == "replace_in_file":
        value = arguments.get("new")
        if isinstance(value, str):
            values.append(value)
    return bool(values) and all(text_is_stub_like_python_repair(value) for value in values)


def mutation_payload_contains_omitted_context_marker(name: str, arguments: dict[str, Any]) -> bool:
    values: list[str] = []
    if name in {"write_file", "replace_symbol"}:
        value = arguments.get("content")
        if isinstance(value, str):
            values.append(value)
    elif name == "replace_symbols":
        replacements = arguments.get("replacements")
        if isinstance(replacements, list):
            values.extend(
                str(item.get("content"))
                for item in replacements
                if isinstance(item, dict) and isinstance(item.get("content"), str)
            )
    elif name == "replace_in_file":
        for key in ("old", "new"):
            value = arguments.get(key)
            if isinstance(value, str):
                values.append(value)
    elif name == "edit_intent":
        value = arguments.get("replacement")
        if isinstance(value, str):
            values.append(value)
    return any(re.search(r"\[omitted \d+ chars from prior [A-Za-z_]+; do not copy\]", value) for value in values)


def validation_failure_is_stub_placeholder(summary: str) -> bool:
    lowered = summary.lower()
    return "still has stub body" in lowered or "pass-style placeholder" in lowered or "stub/comment/pass-style placeholder" in lowered


def mechanical_obligation_repair_failed_for(
    *,
    source_path: str,
    test_path: str,
    events: list[dict[str, Any]],
) -> bool:
    normalized_source = _normalize_repo_path(source_path)
    normalized_test = _normalize_repo_path(test_path)
    for event in reversed(events):
        if event.get("type") != "spec_guided_repair":
            continue
        if event.get("phase") != "mechanical_obligation_verification":
            continue
        event_source = _normalize_repo_path(str(event.get("source_path") or ""))
        event_test = _normalize_repo_path(str(event.get("test_path") or ""))
        if event_source == normalized_source and event_test == normalized_test and event.get("ok") is False:
            return True
    return False


def cli_proof_commands(
    *,
    source_path: str,
    candidate_source: str,
    request_text: str = "",
    python_executable: str,
    shell_command: Callable[[list[str]], str],
) -> list[str]:
    return [
        shell_command([python_executable, *argv])
        for argv in cli_proof_command_argvs(source_path, candidate_source, request_text)
    ]


def cli_surface_source_eligible(*, source_text: str, function_names: set[str]) -> bool:
    return len(source_text.splitlines()) <= 260 and "argparse" in source_text and "main" in function_names


def cli_surface_repair_candidate_score(
    *,
    source_text: str,
    function_names: set[str],
    source_stem: str,
    test_path: str,
    test_text: str,
) -> int | None:
    if "subprocess" not in test_text and "_run(" not in test_text:
        return None
    score = 10
    if "TASKS" in source_text and {"list_tasks", "complete_task"}.issubset(function_names):
        score += 10
    if "@dataclass" in source_text and "--tag" in source_text:
        score += 8
    if source_stem.lower() in test_path.lower():
        score += 10
    return score


def select_cli_surface_repair_candidate(candidates: list[tuple[int, str, str]]) -> tuple[str, str] | None:
    if not candidates:
        return None
    _score, source_path, test_path = sorted(candidates, reverse=True)[0]
    return source_path, test_path


def preemptive_repair_source_score(
    *,
    line_count: int,
    stub_count: int,
    top_function_count: int,
    top_class_count: int,
    parse_ok: bool = True,
) -> tuple[int, int] | None:
    if line_count > 260:
        return None
    if stub_count:
        return (stub_count * 10, -line_count)
    if not parse_ok:
        return None
    if top_function_count == 1 and top_class_count == 0 and line_count <= 80:
        return (1, -line_count)
    if (top_function_count or top_class_count) and line_count <= 120:
        return (0, -line_count)
    return None


def normalized_test_or_source_stem(name: str) -> str:
    stem = str(name or "").strip().lower()
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    if stem.startswith("test_"):
        stem = stem[5:]
    if stem.endswith("_test"):
        stem = stem[:-5]
    return stem


def preemptive_repair_test_score(
    *,
    source_stem: str,
    source_file_stem: str,
    source_parent: str,
    test_stem: str,
    test_name: str,
    test_parent: str,
) -> int:
    score = 0
    normalized_source_stem = normalized_test_or_source_stem(source_stem)
    normalized_test_stem = normalized_test_or_source_stem(test_stem)
    if normalized_test_stem == normalized_source_stem:
        score += 20
    elif normalized_source_stem and normalized_source_stem in normalized_test_stem:
        score += 8
    if str(test_parent) == str(source_parent):
        score += 4
    if str(source_file_stem).lower() in str(test_name).lower():
        score += 4
    return score


def select_preemptive_repair_source(candidates: list[tuple[int, int, str]]) -> str | None:
    if not candidates:
        return None
    _score, _line_count, source_rel = sorted(candidates, reverse=True)[0]
    return source_rel


def select_preemptive_repair_test(candidates: list[tuple[int, str]]) -> str | None:
    if not candidates:
        return None
    _score, test_rel = sorted(candidates, reverse=True)[0]
    return test_rel


def _normalize_repo_path(path: str) -> str:
    return str(path or "").strip().replace("\\", "/").lstrip("./")


def merge_request_obligations(obligations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in obligations:
        if not isinstance(item, dict):
            continue
        obligation_id = str(item.get("id") or "").strip()
        if not obligation_id or obligation_id in seen:
            continue
        seen.add(obligation_id)
        merged.append(dict(item))
    return merged


def derive_request_obligations(
    *,
    request_text: str,
    required_tool_names: set[str],
    doc_targets: list[str] | tuple[str, ...],
    code_mutation_required: bool,
    test_run_required: bool,
) -> list[dict[str, Any]]:
    obligations: list[dict[str, Any]] = []
    lowered = request_text.lower()
    for tool_name in sorted(required_tool_names):
        obligations.append(
            {
                "id": f"tool:{tool_name}",
                "kind": "named_tool",
                "label": f"use {tool_name}",
                "token": tool_name,
            }
        )
    feature_delivery_requested = bool(
        code_mutation_required
        or re.search(
            r"\b(?:add|implement|create|change|modify|update|support|introduce)\b.*\b(?:subcommand|command|flag)\b",
            lowered,
        )
    )
    if feature_delivery_requested:
        obligations.append(
            {
                "id": "code-change",
                "kind": "code_change",
                "label": "implement the requested code change",
            }
        )
    if test_run_required:
        obligations.append(
            {
                "id": "tests-run",
                "kind": "test_run",
                "label": "run tests successfully after the latest edit",
            }
        )
    test_update_requested = bool(
        re.search(
            r"\b(?:add|write|create|update|extend|cover)\b[^.]{0,80}\btests?\b|"
            r"\btests?\b[^.]{0,80}\b(?:add|write|create|update|extend|cover)\b",
            lowered,
        )
    )
    if test_update_requested:
        obligations.append(
            {
                "id": "tests-update",
                "kind": "tests_update",
                "label": "add or update the requested tests",
            }
        )
    shell_proof_requested = bool(
        re.search(
            r"\bprove\b[^.]{0,120}\b(?:shell|command|cli)\b|"
            r"\b(?:shell|command|cli)\b[^.]{0,120}\bproof\b",
            lowered,
        )
    )
    if shell_proof_requested:
        obligations.append(
            {
                "id": "shell-proof",
                "kind": "shell_proof",
                "label": "prove the requested behavior with a shell command",
            }
        )
    docs_update_requested = bool(
        re.search(
            r"\b(?:update|edit|change|modify|revise|add|write|document|sync|mention|include)\b.*\b(?:readme|docs?|documentation)\b|"
            r"\b(?:readme|docs?|documentation)\b.*\b(?:update|edit|change|modify|revise|add|write|document|sync|mention|include)\b",
            lowered,
        )
    )
    if doc_targets or docs_update_requested:
        obligations.append(
            {
                "id": "docs-update",
                "kind": "docs_update",
                "label": "update the requested docs",
                "paths": list(doc_targets),
            }
        )
    if feature_delivery_requested:
        obligations.extend(_feature_token_obligations(request_text))
    return merge_request_obligations(obligations)


def _feature_token_obligations(request_text: str) -> list[dict[str, Any]]:
    obligations: list[dict[str, Any]] = []
    command_token_stopwords = {
        "a",
        "an",
        "the",
        "new",
        "existing",
        "current",
        "requested",
        "direct",
        "targeted",
        "shell",
        "cli",
    }
    function_tokens: set[str] = set()
    function_patterns = [
        r"\b(?:add|implement|create|introduce|support)\s+(?:an?\s+)?([A-Za-z_][A-Za-z0-9_]{1,80})\s*\(",
        r"\b(?:add|implement|create|introduce|support)\s+(?:an?\s+)?([A-Za-z_][A-Za-z0-9_]{1,80})\s+function\b",
        r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]{1,80})\s*\(",
    ]
    function_token_stopwords = {
        "a",
        "an",
        "the",
        "new",
        "function",
        "method",
        "command",
        "flag",
        "tests",
        "test",
    }
    for pattern in function_patterns:
        for match in re.finditer(pattern, request_text, flags=re.IGNORECASE):
            token = str(match.group(1)).strip()
            if token.lower() in function_token_stopwords:
                continue
            function_tokens.add(token)
    for token in sorted(function_tokens, key=str.lower):
        obligations.append(
            {
                "id": f"function:{token.lower()}",
                "kind": "feature_token",
                "label": f'prove the "{token}" function exists',
                "token": token,
                "feature_class": "function",
            }
        )
    for match in re.finditer(r"\b([A-Za-z][A-Za-z0-9_-]{1,40})\b\s+(?:subcommand|command)\b", request_text):
        token = str(match.group(1)).strip()
        if not token or token.lower() in command_token_stopwords:
            continue
        obligations.append(
            {
                "id": f"command:{token.lower()}",
                "kind": "feature_token",
                "label": f'prove the "{token}" command exists',
                "token": token,
                "feature_class": "command",
            }
        )
    for token in sorted(set(re.findall(r"(--[A-Za-z0-9][A-Za-z0-9-]*)", request_text))):
        obligations.append(
            {
                "id": f"flag:{token.lower()}",
                "kind": "feature_token",
                "label": f'prove the "{token}" flag exists',
                "token": token,
                "feature_class": "flag",
            }
        )
    return obligations


def request_obligation_proof_status(
    *,
    obligations: list[dict[str, Any]],
    successful_tool_results: list[dict[str, Any]],
    required_tool_names: set[str],
    mutated_paths: set[str],
    is_doc_path: Callable[[str], bool],
    is_test_path: Callable[[str], bool],
    test_ran: bool,
    shell_proof_ran: bool,
    truncate_text: Callable[[str, int], str],
) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    require_cli_behavior_proof = any(
        isinstance(item, dict)
        and str(item.get("kind") or "").strip() == "feature_token"
        and str(item.get("feature_class") or "").strip() == "flag"
        for item in obligations
    )
    successful_tool_names = {str(item.get("name", "")).strip() for item in successful_tool_results}
    code_mutated = any(not is_doc_path(path) and not is_test_path(path) for path in mutated_paths)
    docs_mutated = {path for path in mutated_paths if is_doc_path(path)}
    tests_mutated = {path for path in mutated_paths if is_test_path(path)}
    doc_reads: set[str] = set()
    for item in successful_tool_results:
        name = str(item.get("name", "")).strip()
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        path = _tool_result_path(result, arguments)
        if not path or name not in {"read_file", "read_symbol", "code_outline"}:
            continue
        if is_doc_path(path):
            doc_reads.add(path)
    for obligation in obligations:
        status = _obligation_status(
            obligation=obligation,
            successful_tool_results=successful_tool_results,
            successful_tool_names=successful_tool_names,
            mutated_paths=mutated_paths,
            code_mutated=code_mutated,
            docs_mutated=docs_mutated,
            tests_mutated=tests_mutated,
            doc_reads=doc_reads,
            test_ran=test_ran,
            shell_proof_ran=shell_proof_ran,
            require_cli_behavior_proof=require_cli_behavior_proof,
            is_doc_path=is_doc_path,
            is_test_path=is_test_path,
            truncate_text=truncate_text,
        )
        if status is not None:
            statuses.append(status)
    unresolved_required_tools = sorted(required_tool_names - successful_tool_names)
    for tool_name in unresolved_required_tools:
        statuses.append(
            {
                "id": f"tool:{tool_name}",
                "kind": "named_tool",
                "label": f"use {tool_name}",
                "status": "unproven",
                "evidence": "",
                "guidance": f"Use {tool_name} successfully before finishing.",
            }
        )
    return merge_request_obligations(statuses)


def _obligation_status(
    *,
    obligation: dict[str, Any],
    successful_tool_results: list[dict[str, Any]],
    successful_tool_names: set[str],
    mutated_paths: set[str],
    code_mutated: bool,
    docs_mutated: set[str],
    tests_mutated: set[str],
    doc_reads: set[str],
    test_ran: bool,
    shell_proof_ran: bool,
    require_cli_behavior_proof: bool,
    is_doc_path: Callable[[str], bool],
    is_test_path: Callable[[str], bool],
    truncate_text: Callable[[str, int], str],
) -> dict[str, Any] | None:
    kind = str(obligation.get("kind") or "").strip()
    label = str(obligation.get("label") or "").strip()
    token = str(obligation.get("token") or "").strip()
    status = {
        "id": str(obligation.get("id") or "").strip(),
        "kind": kind,
        "label": label,
        "status": "unproven",
        "evidence": "",
        "guidance": "",
    }
    if kind == "named_tool":
        if token in successful_tool_names:
            status["status"] = "proven"
            status["evidence"] = token
        else:
            status["guidance"] = f"Use {token} successfully before finishing."
    elif kind == "code_change":
        if code_mutated:
            status["status"] = "proven"
            evidence_paths = sorted(path for path in mutated_paths if not is_doc_path(path))[:3]
            status["evidence"] = ", ".join(evidence_paths)
        else:
            status["guidance"] = "The task still needs a real code change, not only docs or narration."
    elif kind == "test_run":
        if test_ran:
            status["status"] = "proven"
            status["evidence"] = "run_test"
        else:
            status["guidance"] = "Run tests after the latest edit before finishing."
    elif kind == "tests_update":
        if tests_mutated:
            status["status"] = "proven"
            status["evidence"] = ", ".join(sorted(tests_mutated)[:3])
        else:
            status["guidance"] = "The request asked to add or update tests; mutate a relevant test file before finishing."
    elif kind == "shell_proof":
        if shell_proof_ran:
            status["status"] = "proven"
            status["evidence"] = "run_shell"
        else:
            status["guidance"] = "The request asked for shell-command proof; run a direct shell command that demonstrates the requested behavior before finishing."
    elif kind == "docs_update":
        _set_docs_status(status=status, obligation=obligation, docs_mutated=docs_mutated, doc_reads=doc_reads)
    elif kind == "feature_token":
        _set_feature_token_status(
            status=status,
            obligation=obligation,
            successful_tool_results=successful_tool_results,
            require_cli_behavior_proof=require_cli_behavior_proof,
            is_doc_path=is_doc_path,
            is_test_path=is_test_path,
            truncate_text=truncate_text,
        )
    else:
        return None
    return status


def _set_docs_status(
    *,
    status: dict[str, Any],
    obligation: dict[str, Any],
    docs_mutated: set[str],
    doc_reads: set[str],
) -> None:
    requested_paths = [str(path).strip().replace("\\", "/") for path in list(obligation.get("paths") or []) if str(path).strip()]
    if requested_paths:
        matching = sorted(path for path in docs_mutated if path in requested_paths)
        if not matching:
            matching = sorted(path for path in doc_reads if path in requested_paths)
        if matching:
            status["status"] = "proven"
            status["evidence"] = ", ".join(matching[:3])
        else:
            status["guidance"] = "Update the requested docs file and verify it from current evidence."
    elif docs_mutated or doc_reads:
        status["status"] = "proven"
        status["evidence"] = ", ".join(sorted(docs_mutated or doc_reads)[:3])
    else:
        status["guidance"] = "The request asked for docs updates, but no docs file has been changed yet."


def _set_feature_token_status(
    *,
    status: dict[str, Any],
    obligation: dict[str, Any],
    successful_tool_results: list[dict[str, Any]],
    require_cli_behavior_proof: bool,
    is_doc_path: Callable[[str], bool],
    is_test_path: Callable[[str], bool],
    truncate_text: Callable[[str, int], str],
) -> None:
    token = str(obligation.get("token") or "").strip()
    token_lower = token.lower()
    source_evidence = ""
    behavior_evidence = ""
    for item in reversed(successful_tool_results):
        name = str(item.get("name", "")).strip()
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        path = _tool_result_path(result, arguments)
        if name not in {"read_file", "read_symbol", "code_outline", "run_shell", "run_test"}:
            continue
        haystack = _tool_result_haystack(result, arguments).lower()
        if name in {"read_file", "read_symbol", "code_outline"}:
            if path and (is_doc_path(path) or is_test_path(path)):
                continue
            if token_lower and token_lower in haystack and not source_evidence:
                source_evidence = path or name
            continue
        if token_lower and token_lower in haystack and not behavior_evidence:
            behavior_evidence = name
            command_text = str(arguments.get("command") or result.get("command") or "").strip()
            if command_text:
                behavior_evidence = truncate_text(command_text, 120)
    feature_class = str(obligation.get("feature_class") or "feature").strip()
    needs_behavior_proof = feature_class == "flag" or (feature_class == "command" and require_cli_behavior_proof)
    if source_evidence and (behavior_evidence or not needs_behavior_proof):
        status["status"] = "proven"
        status["evidence"] = source_evidence if not behavior_evidence else f"{source_evidence}; {behavior_evidence}"
    if status["status"] != "proven":
        if needs_behavior_proof:
            status["guidance"] = (
                f'The requested {feature_class} "{token}" still needs both implementation proof and behavior proof. '
                + "Read the relevant source file and run a direct command or targeted test that demonstrates it before finishing."
            )
        else:
            status["guidance"] = (
                f'The requested {feature_class} "{token}" is still unproven. '
                + "Read the relevant source file or run a direct command that demonstrates it before finishing."
            )


def _tool_result_path(result: dict[str, Any], arguments: dict[str, Any]) -> str:
    return str(result.get("path") or arguments.get("path") or "").strip().replace("\\", "/").lstrip("./")


def _tool_result_haystack(result: dict[str, Any], arguments: dict[str, Any]) -> str:
    samples = [
        str(result.get("output") or ""),
        str(result.get("summary") or ""),
        str(result.get("symbol") or arguments.get("symbol") or ""),
        str(arguments.get("command") or result.get("command") or ""),
    ]
    return "\n".join(sample for sample in samples if sample)


def cli_feature_capabilities(candidate_source: str, request_text: str = "") -> CliFeatureCapabilities:
    return CliFeatureCapabilities(
        has_stats=bool(re.search(r"add_parser\(\s*['\"]stats['\"]", candidate_source)),
        has_priority_filter="--priority" in candidate_source and bool(re.search(r"add_parser\(\s*['\"]list['\"]", candidate_source)),
        has_json_flag="--json" in candidate_source,
        has_limit_flag="--limit" in candidate_source and "--limit" in request_text,
        has_due_before="--due-before" in candidate_source,
    )


def cli_proof_command_argvs(source_path: str, candidate_source: str, request_text: str = "") -> list[list[str]]:
    caps = cli_feature_capabilities(candidate_source, request_text)
    commands: list[list[str]] = []
    if caps.has_stats:
        commands.append([source_path, "stats"])
    if caps.has_priority_filter:
        commands.append([source_path, "list", "--priority", "high"])
    if caps.has_due_before:
        commands.append([source_path, "list", "--due-before", "2026-07-06"])
        if caps.has_priority_filter:
            commands.append([source_path, "list", "--priority", "high", "--due-before", "2026-07-06"])
    if caps.has_json_flag:
        commands.append([source_path, "--json"])
        if "--tag" in candidate_source:
            commands.append([source_path, "--tag", "work", "--json"])
            if caps.has_limit_flag:
                commands.append([source_path, "--tag", "work", "--limit", "1", "--json"])
    elif caps.has_limit_flag:
        commands.append([source_path, "--limit", "1"])
    return commands


def cli_readme_additions(candidate_source: str, request_text: str, existing_readme: str) -> list[str]:
    caps = cli_feature_capabilities(candidate_source, request_text)
    if not caps.any():
        return []
    lowered = existing_readme.lower()
    additions: list[str] = []
    if caps.has_priority_filter and "--priority" not in lowered:
        additions.append("- `list --priority high` filters tasks by priority.")
    if caps.has_stats and "stats" not in lowered:
        additions.append("- `stats` prints counts by status and priority.")
    if caps.has_json_flag and "--json" not in lowered:
        additions.append("- `--json` prints the selected items as JSON objects.")
    if caps.has_limit_flag and "--limit" not in lowered:
        additions.append("- `--limit N` limits the selected items after filtering and works with `--json`.")
    if caps.has_due_before and "--due-before" not in lowered:
        additions.append("- `list --due-before YYYY-MM-DD` filters tasks by due date and can be combined with `--priority`.")
    return additions


def cli_readme_update_plan(
    *,
    request_text: str,
    candidate_source: str,
    existing_readme: str,
) -> dict[str, str]:
    if not re.search(r"\b(?:readme|docs?|documentation)\b", request_text, flags=re.IGNORECASE):
        return {"action": "skip"}
    capabilities = cli_feature_capabilities(candidate_source, request_text)
    if not capabilities.any():
        return {"action": "skip"}
    additions = cli_readme_additions(candidate_source, request_text, existing_readme)
    if not additions:
        return {"action": "read"}
    separator = "" if existing_readme.endswith("\n") else "\n"
    content = existing_readme + separator + "\nAdditional commands:\n" + "\n".join(additions) + "\n"
    return {"action": "write", "content": content}


def cli_test_additions(candidate_source: str, request_text: str, existing_test_text: str, helper_name: str) -> list[str]:
    wants_limit_tests = "--limit" in candidate_source and "--limit" in request_text
    wants_due_tests = "--due-before" in candidate_source and "--due-before" in request_text
    if "--json" not in candidate_source and not wants_limit_tests and not wants_due_tests:
        return []
    additions: list[str] = []
    if "--json" in candidate_source and "--json" not in existing_test_text:
        additions.append(
            "    def test_json_output(self) -> None:\n"
            f"        result = {helper_name}('--json')\n"
            "        self.assertEqual(result.returncode, 0)\n"
            "        self.assertIn('\"title\"', result.stdout)\n"
            "        self.assertIn('\"body\"', result.stdout)\n"
            "        self.assertIn('\"tags\"', result.stdout)\n"
            "\n"
        )
    if wants_limit_tests and "--limit" not in existing_test_text:
        additions.append(
            "    def test_limit_text_output(self) -> None:\n"
            f"        result = {helper_name}('--tag', 'work', '--limit', '1')\n"
            "        self.assertEqual(result.returncode, 0)\n"
            "        self.assertIn('ship-cli', result.stdout)\n"
            "        self.assertNotIn('fix-bug', result.stdout)\n"
            "\n"
            "    def test_limit_json_output(self) -> None:\n"
            f"        result = {helper_name}('--tag', 'work', '--limit', '1', '--json')\n"
            "        self.assertEqual(result.returncode, 0)\n"
            "        self.assertIn('\"title\": \"ship-cli\"', result.stdout)\n"
            "        self.assertNotIn('fix-bug', result.stdout)\n"
            "\n"
        )
    if wants_due_tests and "--due-before" not in existing_test_text:
        additions.append(
            "    def test_due_before_filter(self) -> None:\n"
            f"        result = {helper_name}('list', '--due-before', '2026-07-06')\n"
            "        self.assertEqual(result.returncode, 0)\n"
            "        self.assertIn('write-docs:todo:high:2026-07-01', result.stdout)\n"
            "        self.assertIn('ship-cli:done:low:2026-07-05', result.stdout)\n"
            "        self.assertNotIn('fix-bug', result.stdout)\n"
            "\n"
            "    def test_due_before_preserves_priority_filter(self) -> None:\n"
            f"        result = {helper_name}('list', '--priority', 'high', '--due-before', '2026-07-06')\n"
            "        self.assertEqual(result.returncode, 0)\n"
            "        self.assertIn('write-docs:todo:high:2026-07-01', result.stdout)\n"
            "        self.assertNotIn('ship-cli', result.stdout)\n"
            "        self.assertNotIn('fix-bug', result.stdout)\n"
            "\n"
            "    def test_due_before_rejects_invalid_date(self) -> None:\n"
            f"        result = {helper_name}('list', '--due-before', '2026-99-99')\n"
            "        self.assertNotEqual(result.returncode, 0)\n"
            "        self.assertIn('invalid', (result.stderr + result.stdout).lower())\n"
            "\n"
        )
    return additions


def cli_test_update_plan(
    *,
    candidate_source: str,
    request_text: str,
    existing_test_text: str,
) -> dict[str, str]:
    if "--json" not in candidate_source and "--limit" not in candidate_source and "--due-before" not in candidate_source:
        return {"action": "skip"}
    helper_match = re.search(r"(?m)^def\s+(?P<name>[_A-Za-z]\w*)\(\*args:\s*str\)", existing_test_text)
    if not helper_match:
        return {"action": "skip"}
    additions = cli_test_additions(candidate_source, request_text, existing_test_text, helper_match.group("name"))
    if not additions:
        return {"action": "skip"}
    insertion = "\n" + "\n".join(additions)
    marker = "\n\nif __name__ == '__main__':"
    if marker in existing_test_text:
        content = existing_test_text.replace(marker, insertion + marker, 1)
    else:
        content = existing_test_text.rstrip() + insertion
    return {"action": "write", "content": content}
