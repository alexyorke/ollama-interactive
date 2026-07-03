from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


MUTATION_ACTIONS = {"implementation", "tests", "docs"}
PROOF_ACTIONS = {"validation", "behavior_proof", "final"}
VALIDATION_LOOP_TOOL_NAMES = {"select_tests", "run_test", "lint_typecheck", "contract_check", "run_function_probe"}


@dataclass(frozen=True)
class RequestedDeliverable:
    id: str
    kind: str
    label: str
    token: str = ""
    feature_class: str = ""
    status: str = "unproven"
    evidence: str = ""


@dataclass(frozen=True)
class GroundedTarget:
    path: str
    symbol: str = ""
    kind: str = "path"
    source: str = "evidence"


@dataclass(frozen=True)
class FailedAttempt:
    path: str
    symbol: str = ""
    tool_name: str = ""
    validator: str = ""
    diagnostic: str = ""
    granularity: str = ""


@dataclass(frozen=True)
class PatchPlan:
    strategy: str
    implementation_targets: list[str] = field(default_factory=list)
    test_targets: list[str] = field(default_factory=list)
    doc_targets: list[str] = field(default_factory=list)
    proof_commands: list[str] = field(default_factory=list)
    required_content: list[str] = field(default_factory=list)
    complete: bool = False
    missing: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ValidationPlan:
    required: list[str] = field(default_factory=list)
    blocked_until: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RepairProtocolState:
    requested_deliverables: list[RequestedDeliverable] = field(default_factory=list)
    grounded_targets: list[GroundedTarget] = field(default_factory=list)
    failed_attempts: list[FailedAttempt] = field(default_factory=list)
    allowed_next_actions: list[str] = field(default_factory=list)
    patch_plan: PatchPlan | None = None
    validation_plan: ValidationPlan = field(default_factory=ValidationPlan)
    proof_obligations: list[str] = field(default_factory=list)
    repair_strategy: str = "normal"

    def to_event_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.patch_plan is None:
            payload["patch_plan"] = None
        return payload


def deliverables_from_obligations(
    obligations: list[dict[str, Any]],
    statuses: list[dict[str, Any]] | None = None,
) -> list[RequestedDeliverable]:
    status_by_id = {
        str(item.get("id") or ""): item
        for item in list(statuses or [])
        if isinstance(item, dict) and str(item.get("id") or "")
    }
    deliverables: list[RequestedDeliverable] = []
    for item in obligations:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        if not item_id:
            continue
        status = status_by_id.get(item_id, {})
        deliverables.append(
            RequestedDeliverable(
                id=item_id,
                kind=str(item.get("kind") or "").strip(),
                label=str(item.get("label") or item_id).strip(),
                token=str(item.get("token") or "").strip(),
                feature_class=str(item.get("feature_class") or "").strip(),
                status=str(status.get("status") or "unproven").strip(),
                evidence=str(status.get("evidence") or "").strip(),
            )
        )
    return deliverables


def grounded_targets_from_tool_results(results: list[dict[str, Any]]) -> list[GroundedTarget]:
    targets: list[GroundedTarget] = []
    seen: set[tuple[str, str]] = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name not in {"read_file", "read_symbol", "code_outline"}:
            continue
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        args = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        if result.get("ok") is not True:
            continue
        path = str(result.get("path") or args.get("path") or "").strip().replace("\\", "/").lstrip("./")
        symbol = str(result.get("symbol") or args.get("symbol") or "").strip()
        if not path:
            continue
        key = (path, symbol)
        if key in seen:
            continue
        seen.add(key)
        targets.append(
            GroundedTarget(
                path=path,
                symbol=symbol,
                kind="symbol" if symbol else "path",
                source=name,
            )
        )
    return targets


def failed_attempts_from_recovery_states(states: list[dict[str, Any]]) -> list[FailedAttempt]:
    attempts: list[FailedAttempt] = []
    for item in states:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip().replace("\\", "/").lstrip("./")
        if not path:
            continue
        attempts.append(
            FailedAttempt(
                path=path,
                symbol=str(item.get("symbol") or "").strip(),
                tool_name=str(item.get("tool_name") or item.get("last_mutating_tool_family") or "").strip(),
                validator=str(item.get("validation_name") or "").strip(),
                diagnostic=str(item.get("diagnostic_excerpt") or item.get("diagnostic") or "").strip(),
                granularity=str(item.get("tool_granularity") or "").strip(),
            )
        )
    return attempts


def has_cli_feature(deliverables: list[RequestedDeliverable]) -> bool:
    return any(
        item.kind == "feature_token" and item.feature_class in {"command", "flag"}
        for item in deliverables
    )


def build_cli_patch_plan(
    *,
    deliverables: list[RequestedDeliverable],
    targets: list[GroundedTarget],
) -> PatchPlan:
    implementation_targets = [target.path for target in targets if target.path and not _is_docs_path(target.path) and not _is_test_path(target.path)]
    implementation_targets = list(dict.fromkeys(implementation_targets))
    doc_targets = ["README.md"] if any(item.kind == "docs_update" for item in deliverables) else []
    test_targets = ["tests/"] if any(item.kind in {"tests_update", "test_run"} for item in deliverables) else []
    proof_commands = []
    if any(item.kind == "shell_proof" for item in deliverables) or has_cli_feature(deliverables):
        proof_commands.append("run_shell direct CLI proof")

    required_content = ["parser change", "callable behavior"]
    if test_targets:
        required_content.append("tests")
    if doc_targets:
        required_content.append("docs")
    if proof_commands:
        required_content.append("behavior proof")

    missing: list[str] = []
    if not implementation_targets:
        missing.append("grounded implementation target")
    if any(item.kind == "tests_update" for item in deliverables) and not test_targets:
        missing.append("test target")
    if any(item.kind == "docs_update" for item in deliverables) and not doc_targets:
        missing.append("docs target")
    if has_cli_feature(deliverables) and not proof_commands:
        missing.append("CLI behavior proof")

    return PatchPlan(
        strategy="cli_patch_bundle",
        implementation_targets=implementation_targets,
        test_targets=test_targets,
        doc_targets=doc_targets,
        proof_commands=proof_commands,
        required_content=required_content,
        complete=not missing,
        missing=missing,
    )


def build_repair_protocol_state(
    *,
    obligations: list[dict[str, Any]],
    obligation_statuses: list[dict[str, Any]] | None,
    successful_tool_results: list[dict[str, Any]],
    recovery_states: list[dict[str, Any]],
) -> RepairProtocolState:
    deliverables = deliverables_from_obligations(obligations, obligation_statuses)
    targets = grounded_targets_from_tool_results(successful_tool_results)
    failed_attempts = failed_attempts_from_recovery_states(recovery_states)
    cli_feature = has_cli_feature(deliverables)
    patch_plan = build_cli_patch_plan(deliverables=deliverables, targets=targets) if cli_feature else None
    has_failed_attempt = bool(failed_attempts)
    proven = {item.id for item in deliverables if item.status == "proven"}
    unproven = [item for item in deliverables if item.id not in proven]

    allowed: list[str] = []
    blocked_until: list[str] = []
    if has_failed_attempt:
        allowed.extend(["grounding", "repair_plan", "implementation"])
        blocked_until.extend(["successful broader repair"])
    elif patch_plan is not None and not patch_plan.complete:
        allowed.append("grounding")
        blocked_until.extend(patch_plan.missing)
    else:
        if any(item.kind == "code_change" and item.status != "proven" for item in deliverables):
            allowed.append("implementation")
            blocked_until.append("source mutation")
        if any(item.kind == "tests_update" and item.status != "proven" for item in deliverables):
            allowed.append("tests")
            blocked_until.append("test mutation")
        if any(item.kind == "docs_update" and item.status != "proven" for item in deliverables):
            allowed.append("docs")
            blocked_until.append("docs mutation")
    if not blocked_until and any(item.kind in {"test_run", "shell_proof"} and item.status != "proven" for item in deliverables):
        allowed.extend(["validation", "behavior_proof"])
    elif not blocked_until and unproven:
        allowed.extend(["grounding", "behavior_proof"])
    elif not blocked_until:
        allowed.append("final")

    repair_strategy = "cli_patch_bundle" if cli_feature else ("repair_after_failure" if has_failed_attempt else "normal")
    return RepairProtocolState(
        requested_deliverables=deliverables,
        grounded_targets=targets,
        failed_attempts=failed_attempts,
        allowed_next_actions=list(dict.fromkeys(allowed)),
        patch_plan=patch_plan,
        validation_plan=ValidationPlan(required=["source_proof", "behavior_proof"] if cli_feature else [], blocked_until=blocked_until),
        proof_obligations=[item.label for item in unproven],
        repair_strategy=repair_strategy,
    )


def next_action_for_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    if tool_name in {"read_file", "read_symbol", "code_outline", "search", "search_symbols", "find_implementation_target"}:
        return "grounding"
    if tool_name in {"write_file", "replace_symbol", "replace_symbols", "replace_in_file", "edit_intent"}:
        path = str(arguments.get("path") or "").strip().replace("\\", "/").lstrip("./")
        if _is_docs_path(path):
            return "docs"
        if _is_test_path(path):
            return "tests"
        return "implementation"
    if tool_name in {"run_test", "lint_typecheck", "contract_check", "select_tests", "discover_validators"}:
        return "validation"
    if tool_name in {"run_shell", "run_function_probe"}:
        return "behavior_proof"
    if tool_name == "final":
        return "final"
    return "other"


def repair_decision_for_tool(
    state: RepairProtocolState,
    *,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    action = next_action_for_tool(tool_name, arguments)
    allowed = action in set(state.allowed_next_actions) or action == "grounding"
    violation = ""
    reason = ""
    if (
        state.repair_strategy == "cli_patch_bundle"
        and any(item.kind == "feature_token" and item.feature_class == "flag" for item in state.requested_deliverables)
        and action == "implementation"
        and _is_narrow_cli_mutation(tool_name, arguments)
    ):
        allowed = False
        violation = "narrow_cli_mutation"
        reason = (
            "CLI flag work must be repaired as a full command-surface bundle. "
            "Use replace_symbol, replace_symbols, or a grounded write_file so parser/signature/behavior stay consistent."
        )
    if not allowed:
        blocked = ", ".join(state.validation_plan.blocked_until[:3])
        reason = reason or f"Next action must satisfy typed protocol first: {blocked or ', '.join(state.allowed_next_actions)}."
    return {
        "action": action,
        "allowed": allowed,
        "reason": reason,
        "violation": violation,
        "allowed_next_actions": list(state.allowed_next_actions),
        "repair_strategy": state.repair_strategy,
    }


def merge_failed_edit_recovery(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in states:
        if not isinstance(item, dict):
            continue
        target_id = str(item.get("target_id") or "").strip()
        if not target_id or target_id in seen:
            continue
        seen.add(target_id)
        merged.append(dict(item))
    return merged


def mutation_edit_granularity(*, name: str, arguments: dict[str, Any]) -> str:
    if name == "write_file":
        return "broad_file"
    if name in {"replace_symbol", "replace_symbols"}:
        return "broad_symbol"
    if name == "replace_in_file":
        return "narrow"
    if name == "apply_structured_edit":
        operation = arguments.get("operation")
        op_name = ""
        if isinstance(operation, dict):
            op_name = str(operation.get("op") or "").strip().lower()
        if op_name in {"replace_symbol"}:
            return "broad_symbol"
        return "narrow"
    if name == "edit_intent":
        intent = str(arguments.get("intent") or "").strip().lower()
        if intent in {"replace_symbol"}:
            return "broad_symbol"
        if intent:
            return "narrow"
    return "other"


def recovery_target_from_mutation(
    *,
    symbol_target: tuple[str, str] | None,
    mutation_target_paths: list[str],
    result: dict[str, Any] | None,
    recent_source_paths: list[str],
    path_looks_like_test_file: Any,
) -> dict[str, str] | None:
    if symbol_target is not None:
        path, symbol = symbol_target
        normalized_path = _normalize_path(path)
        return {
            "target_id": f"symbol:{normalized_path.lower()}:{symbol}",
            "kind": "symbol",
            "path": normalized_path,
            "symbol": symbol,
        }
    for raw_path in mutation_target_paths:
        normalized = _normalize_path(str(raw_path or ""))
        if normalized and normalized.endswith(".py") and not path_looks_like_test_file(normalized):
            return {
                "target_id": f"path:{normalized.lower()}",
                "kind": "path",
                "path": normalized,
                "symbol": "",
            }
    result_dict = result if isinstance(result, dict) else {}
    result_path = _normalize_path(str(result_dict.get("path") or ""))
    if result_path and result_path.endswith(".py") and not path_looks_like_test_file(result_path):
        return {
            "target_id": f"path:{result_path.lower()}",
            "kind": "path",
            "path": result_path,
            "symbol": "",
        }
    if len(recent_source_paths) == 1:
        normalized = _normalize_path(recent_source_paths[0])
        return {
            "target_id": f"path:{normalized.lower()}",
            "kind": "path",
            "path": normalized,
            "symbol": "",
        }
    return None


def recovery_target_matches(state: dict[str, Any], target: dict[str, str]) -> bool:
    state_id = str(state.get("target_id") or "").strip()
    target_id = str(target.get("target_id") or "").strip()
    if state_id and target_id:
        return state_id == target_id
    state_path = str(state.get("path") or "").strip().lower()
    target_path = str(target.get("path") or "").strip().lower()
    if not state_path or not target_path or state_path != target_path:
        return False
    state_symbol = str(state.get("symbol") or "").strip()
    target_symbol = str(target.get("symbol") or "").strip()
    return not state_symbol or not target_symbol or state_symbol == target_symbol


def repair_spec_target_regrounded(state: dict[str, Any], events: list[dict[str, Any]]) -> bool:
    failure_event_index = int(state.get("failure_event_index", -1) or -1)
    target_path = _normalize_path(str(state.get("path") or ""))
    target_symbol = str(state.get("symbol") or "").strip()
    if not target_path:
        return False
    for index, event in enumerate(events):
        if index <= failure_event_index or event.get("type") != "tool_result":
            continue
        name = str(event.get("name") or "").strip()
        if name not in {"read_file", "read_symbol", "code_outline"}:
            continue
        result = event.get("result") if isinstance(event.get("result"), dict) else {}
        arguments = event.get("arguments") if isinstance(event.get("arguments"), dict) else {}
        if result.get("ok") is not True:
            continue
        event_path = _normalize_path(str(result.get("path") or arguments.get("path") or ""))
        if event_path != target_path:
            continue
        if name == "read_symbol" and target_symbol:
            event_symbol = str(result.get("symbol") or arguments.get("symbol") or "").strip()
            if event_symbol != target_symbol:
                continue
        return True
    return False


def repair_spec_behavior_regrounded(
    state: dict[str, Any],
    *,
    events: list[dict[str, Any]],
    behavior_paths: list[str] | tuple[str, ...],
) -> bool:
    failure_event_index = int(state.get("failure_event_index", -1) or -1)
    normalized_behavior_paths = {_normalize_path(str(path)) for path in behavior_paths if str(path).strip()}
    if not normalized_behavior_paths:
        return True
    for index, event in enumerate(events):
        if index <= failure_event_index or event.get("type") != "tool_result":
            continue
        name = str(event.get("name") or "").strip()
        if name not in {"read_file", "read_symbol", "code_outline"}:
            continue
        result = event.get("result") if isinstance(event.get("result"), dict) else {}
        arguments = event.get("arguments") if isinstance(event.get("arguments"), dict) else {}
        if result.get("ok") is not True:
            continue
        event_path = _normalize_path(str(result.get("path") or arguments.get("path") or ""))
        if event_path and event_path in normalized_behavior_paths:
            return True
    return False


def repair_spec_required_proof_items(state: dict[str, Any]) -> list[str]:
    items: list[str] = []
    raw_items = state.get("required_proof_items")
    if isinstance(raw_items, list):
        items.extend(str(item).strip() for item in raw_items if str(item).strip())
    if not items:
        for obligation in list(state.get("unresolved_obligations") or []):
            if not isinstance(obligation, dict):
                continue
            label = str(obligation.get("label") or "").strip()
            if label:
                items.append(label)
    return list(dict.fromkeys(items))


def repair_spec_behavior_paths(
    state: dict[str, Any],
    *,
    test_path_candidates: list[str] | tuple[str, ...] = (),
) -> list[str]:
    raw_paths = state.get("behavior_paths")
    if isinstance(raw_paths, list):
        normalized = [_normalize_path(str(item)) for item in raw_paths if str(item).strip()]
        if normalized:
            return list(dict.fromkeys(normalized))

    paths: list[str] = [_normalize_path(str(path)) for path in test_path_candidates if str(path).strip()]
    for obligation in list(state.get("unresolved_obligations") or []):
        if not isinstance(obligation, dict):
            continue
        if str(obligation.get("kind") or "").strip() != "docs_update":
            continue
        for raw_path in list(obligation.get("paths") or []):
            normalized = _normalize_path(str(raw_path))
            if normalized:
                paths.append(normalized)
    return sorted(dict.fromkeys(paths))


def repair_spec_strategy_class(
    *,
    target: dict[str, str],
    obligations: list[dict[str, Any]],
    file_repair_allowed: bool,
) -> str:
    if any(
        isinstance(item, dict)
        and str(item.get("kind") or "").strip() == "feature_token"
        and str(item.get("feature_class") or "").strip() in {"command", "flag"}
        for item in obligations
    ):
        return "cli_surface_repair"
    if str(target.get("symbol") or "").strip():
        return "symbol_rewrite"
    if file_repair_allowed:
        return "file_repair"
    return "cross_file_feature"


def repair_spec_complete_plan(
    state: dict[str, Any],
    *,
    required_proof_items: list[str] | None = None,
) -> str:
    strategy = str(state.get("repair_strategy") or "").strip() or "file_repair"
    path = str(state.get("path") or "").strip()
    obligations = list(required_proof_items or repair_spec_required_proof_items(state))
    obligations_text = ", ".join(obligations[:4]) if obligations else "the unresolved feature obligations"
    if strategy == "cli_surface_repair":
        return (
            f"Complete one command-surface repair in {path or 'the grounded CLI file'} so parser, behavior, docs, "
            f"and proof land together for {obligations_text}."
        )
    if strategy == "symbol_rewrite":
        return f"Complete one full-symbol repair that resolves {obligations_text} in the grounded source."
    if strategy == "cross_file_feature":
        return f"Complete one coordinated feature repair across the allowed files for {obligations_text}."
    return f"Complete one broader file repair in {path or 'the grounded source file'} for {obligations_text}."


def repair_spec_broad_repair_hint(
    state: dict[str, Any],
    *,
    file_repair_allowed: bool,
) -> str:
    strategy = str(state.get("repair_strategy") or "").strip()
    if strategy == "cli_surface_repair":
        path = str(state.get("path") or "").strip()
        if path and file_repair_allowed:
            return f"write_file on {path} so the CLI surface is repaired in one pass"
        return "one grounded whole-surface CLI repair"
    if str(state.get("symbol") or "").strip():
        return "a full-symbol replacement"
    if file_repair_allowed:
        return "write_file or a full-symbol replacement"
    return "a full-symbol replacement or another broader direct repair"


def repair_spec_mutation_decision(
    state: dict[str, Any],
    *,
    proposed_tool_name: str,
    proposed_paths: list[str],
    repair_granularity: str,
    file_repair_allowed: bool,
) -> dict[str, Any]:
    strategy = str(state.get("repair_strategy") or "").strip()
    target_path = _normalize_path(str(state.get("path") or ""))
    symbol = str(state.get("symbol") or "").strip()
    normalized_paths = [_normalize_path(path) for path in proposed_paths if str(path or "").strip()]
    if target_path and normalized_paths and any(path != target_path for path in normalized_paths):
        return {"allowed": False, "reason": "Repair this grounded target before mutating unrelated files."}
    if repair_granularity == "narrow":
        return {"allowed": False, "reason": "Do not make another small speculative edit on the same failed target."}
    if strategy == "cli_surface_repair":
        if proposed_tool_name == "write_file" and target_path and target_path in normalized_paths:
            if file_repair_allowed:
                return {"allowed": True, "reason": ""}
        if symbol and proposed_tool_name in {"replace_symbol", "replace_symbols"}:
            return {"allowed": True, "reason": ""}
        if not file_repair_allowed:
            return {
                "allowed": False,
                "reason": "Use a grounded symbol-level repair here because the CLI file is too large for a safe full rewrite.",
            }
        return {
            "allowed": False,
            "reason": "Use one broader direct repair on the grounded CLI surface before more validation.",
        }
    if repair_granularity == "broad_file" and not file_repair_allowed:
        return {
            "allowed": False,
            "reason": "Prefer a full-symbol replacement here; the grounded file is too large for a safe full-file rewrite fallback.",
        }
    return {"allowed": True, "reason": ""}


def repair_spec_blocks_validation_loop(*, tool_name: str, has_followup_mutation: bool) -> bool:
    if tool_name not in VALIDATION_LOOP_TOOL_NAMES:
        return False
    return not has_followup_mutation


def repair_spec_retry_message(
    state: dict[str, Any],
    *,
    need_reground: bool,
    need_behavior_reground: bool,
    behavior_paths: list[str] | tuple[str, ...],
    broad_repair_hint: str,
    complete_plan: str,
) -> str:
    path = str(state.get("path") or "").strip()
    symbol = str(state.get("symbol") or "").strip()
    diagnostic = str(state.get("diagnostic") or "").strip()
    validation_name = str(state.get("validation_name") or "").strip() or "validation"
    if symbol:
        target_label = f"{symbol} in {path}"
        reground_step = f"Re-ground {target_label} from current source with read_symbol before another mutation."
    else:
        target_label = path or "the current source target"
        reground_step = f"Re-ground {target_label} from current source with read_file before another mutation."
    behavior_step = ""
    if behavior_paths:
        behavior_targets = ", ".join(str(path).strip() for path in behavior_paths[:3] if str(path).strip())
        if behavior_targets:
            behavior_step = f" Re-read the failing behavior surface with read_file on {behavior_targets} before repairing."
    repair_step = (
        " Then make one broader repair with "
        + broad_repair_hint
        + ", rerun proof-producing validation, and only then finish."
    )
    if need_reground:
        message = f"Validation already failed after a prior edit on {target_label}. {reground_step}"
    else:
        message = (
            f"Validation already failed after a prior edit on {target_label}. "
            + "Do not make another small speculative edit on the same target."
        )
    if need_behavior_reground and behavior_step:
        message += behavior_step
    message += " " + complete_plan + repair_step
    if diagnostic:
        message += f" Last {validation_name}: {diagnostic}"
    return message


def failed_test_still_needs_repair(
    *,
    latest_run_test_failed: bool,
    failed_test_mutation_version: int | None,
    mutation_version: int,
) -> bool:
    return latest_run_test_failed and failed_test_mutation_version == mutation_version


def failed_test_repair_retry_message(summary: str, *, limit: int = 420) -> str:
    message = "The latest run_test failed after the current edit. Repair the implementation before rerunning validators or finishing."
    diagnostic = _truncate_text(str(summary or ""), limit=limit).strip()
    if diagnostic:
        message += " Evidence: " + diagnostic
    return message


def cli_patch_bundle_instruction(state: RepairProtocolState) -> str | None:
    plan = state.patch_plan
    if plan is None or plan.strategy != "cli_patch_bundle":
        return None
    if not plan.complete:
        missing = ", ".join(plan.missing[:4])
        return f"Typed repair protocol: ground the missing CLI patch-bundle inputs first: {missing}. Next JSON only."
    targets = ", ".join(plan.implementation_targets[:3]) or "the grounded CLI source"
    required = ", ".join(plan.required_content)
    return (
        f"Typed repair protocol: treat this as one CLI patch bundle for {targets}. "
        f"Plan and complete parser change, callable behavior, tests, docs, and direct CLI behavior proof together; "
        f"do not run validation or final until the bundle covers: {required}. Next JSON only."
    )


def _is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return normalized.startswith("tests/") or "/tests/" in normalized or normalized.startswith("test_") or normalized.endswith("_test.py")


def _is_docs_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return normalized.endswith(".md") or "readme" in normalized or normalized.startswith("docs/")


def _is_narrow_cli_mutation(tool_name: str, arguments: dict[str, Any]) -> bool:
    if tool_name == "replace_in_file":
        return True
    if tool_name == "edit_intent":
        return True
    return False


def _normalize_path(path: str) -> str:
    return str(path or "").strip().replace("\\", "/").lstrip("./")


def _truncate_text(text: str, *, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."
