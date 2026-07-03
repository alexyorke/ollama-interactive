from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


MUTATION_ACTIONS = {"implementation", "tests", "docs"}
PROOF_ACTIONS = {"validation", "behavior_proof", "final"}


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
