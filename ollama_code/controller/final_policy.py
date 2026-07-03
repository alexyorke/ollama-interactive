from __future__ import annotations

import re


SUCCESS_CLAIM_NEGATIONS = r"\b(?:timed out|timeout|failed|could not|unable|did not|didn't|not working|still needs)\b"
SHELL_SUCCESS_CLAIM_NEGATIONS = r"\b(?:timed out|timeout|failed|could not|unable|did not|didn't|not installed|missing|not found|still needs)\b"
GENERIC_SUCCESS_PATTERNS = (
    r"\bservice\s+(?:is\s+)?working\b",
    r"\bverification\s+passed\b",
    r"\bworks\b",
    r"\bverified\b",
    r"\bsucceeded\b",
    r"\bstarted\b",
    r"\brunning\b",
    r"\bhealthy\b",
    r"\bavailable\b",
)


def final_claims_timeout_success(message: str) -> bool:
    lowered = str(message or "").lower()
    if not lowered.strip():
        return False
    if re.search(SUCCESS_CLAIM_NEGATIONS, lowered):
        return False
    return any(re.search(pattern, lowered) for pattern in GENERIC_SUCCESS_PATTERNS)


def final_claims_run_shell_success(message: str) -> bool:
    lowered = str(message or "").lower()
    if not lowered.strip():
        return False
    if re.search(SHELL_SUCCESS_CLAIM_NEGATIONS, lowered):
        return False
    return any(re.search(pattern, lowered) for pattern in GENERIC_SUCCESS_PATTERNS)


def final_claims_path_exists(message: str) -> bool:
    lowered = str(message or "").lower().strip()
    if not lowered:
        return False
    if re.search(r"\b(?:does not exist|is missing|are missing|not found|path_missing|no such file|no such directory)\b", lowered):
        return False
    return bool(re.search(r"\bexists\b", lowered))


def final_acknowledges_missing_path(message: str) -> bool:
    lowered = str(message or "").lower().strip()
    if not lowered:
        return False
    return bool(re.search(r"\b(?:does not exist|is missing|are missing|not found|path_missing|no such file|no such directory)\b", lowered))


def final_claims_file_mutation(message: str) -> bool:
    lowered = str(message or "").lower()
    patterns = [
        r"\b(?:i|we)\s+(?:updated|edited|changed|modified|created|wrote|rewrote|deleted|removed|renamed)\b",
        r"\bhas been\s+(?:updated|edited|changed|modified|created|written|rewritten|deleted|removed|renamed)\b",
        r"\bwas\s+(?:updated|edited|changed|modified|created|written|rewritten|deleted|removed|renamed)\b",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def final_claims_test_success(message: str) -> bool:
    lowered = str(message or "").lower()
    patterns = [
        r"\btests?\s+(?:pass|passed|passing|succeed|succeeded|successful)\b",
        r"\btest suite\s+(?:passes|passed|succeeded|is successful)\b",
        r"\ball (?:provided )?tests?\s+(?:pass|passed|are passing)\b",
        r"\brun_test\s+(?:pass|passed|succeeded)\b",
        r"\bsuccessfully\s+(?:ran|executed).{0,40}\btests?\b",
        r"\btests?\s+(?:have been|were)\s+(?:executed|run)\s+successfully\b",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def final_requires_verification(
    *,
    has_request_obligations: bool,
    has_required_or_forbidden_tools: bool,
    mutation_verified_this_turn: bool,
    final_claims_mutation: bool,
    has_expected_exact_file_line: bool,
    tool_call_count: int,
    request_needs_exact_grounding: bool,
    tool_names: set[str],
    risky_verification_tool_names: set[str],
) -> bool:
    if has_request_obligations:
        return True
    if has_required_or_forbidden_tools:
        return True
    if mutation_verified_this_turn or final_claims_mutation:
        return True
    if has_expected_exact_file_line:
        return True
    if tool_call_count and request_needs_exact_grounding:
        return True
    if tool_call_count >= 2:
        return True
    return any(name in risky_verification_tool_names for name in tool_names)
