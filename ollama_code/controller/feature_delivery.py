from __future__ import annotations

import ast
from dataclasses import dataclass
import re
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
