from __future__ import annotations

import re


def path_looks_like_doc_target(path: str) -> bool:
    normalized = str(path or "").replace("\\", "/").lstrip("./").lower()
    if not normalized:
        return False
    if normalized == "readme.md" or normalized.endswith("/readme.md"):
        return True
    return normalized.startswith("docs/") or normalized.endswith((".md", ".rst", ".txt"))


def path_looks_like_test_file(path: str) -> bool:
    normalized = str(path or "").replace("\\", "/").lower()
    name = normalized.rsplit("/", 1)[-1]
    return bool(
        re.search(r"(^test_|_test\.|\.test\.|\.spec\.)", name)
        or "/tests/" in normalized
        or normalized.startswith("tests/")
    )


def request_looks_like_issue_report(text: str) -> bool:
    lowered = text.lower()
    has_code_context = bool(
        re.search(
            r"`[^`]+`|(?:^|\s)[A-Za-z_][\w./-]*\.(?:py|js|ts|tsx|jsx|java|go|rs|c|cc|cpp)\b|array\(\[|traceback|from [A-Za-z0-9_. ]+ import ",
            text,
        )
    )
    if not has_code_context:
        return False
    issue_patterns = [
        r"\b(?:bug|issue|regression)\b",
        r"\bdoes not\b[^.?!\n]{0,120}\b(?:correctly|properly|compute|return|handle|pass|work)\b",
        r"\breturns?\b[^.?!\n]{0,80}\bwrong\b",
        r"\b(?:incorrect|incorrectly|unexpected(?:ly)?)\b",
        r"\bfails?\b[^.?!\n]{0,120}\b(?:when|with|for|to|under|on)\b",
    ]
    return any(re.search(pattern, lowered) for pattern in issue_patterns)


def request_needs_exact_grounding(text: str) -> bool:
    lowered = text.lower()
    patterns = [
        r"\bexact(?:ly)?\b",
        r"\bline\s+\d+\b",
        r"\bfirst line\b",
        r"\bsingle line\b",
        r"\bwhat(?:'s| is)? .* say\b",
        r"\btoken\b",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def request_allows_mutation(text: str) -> bool:
    lowered = text.lower()
    mutation_phrases = [
        "create ",
        "write ",
        "replace ",
        "edit ",
        "update ",
        "rewrite ",
        "append ",
        "modify ",
        "change ",
        "delete ",
        "remove ",
        "rename ",
        "fix ",
        "implement ",
        "patch ",
        "refactor ",
        "make ",
        "add ",
        "commit ",
    ]
    return any(phrase in lowered for phrase in mutation_phrases) or request_looks_like_issue_report(text)


def request_requires_mutation(text: str) -> bool:
    lowered = text.lower()
    read_only_patterns = [
        r"\bdo not edit\b(?!\s+(?:tests?|test files?)\b)",
        r"\bdon't edit\b(?!\s+(?:tests?|test files?)\b)",
        r"\bdo not modify\b",
        r"\bdon't modify\b",
        r"\bdo not change\b(?!\s+(?:tests?|test files?)\b)",
        r"\bdon't change\b(?!\s+(?:tests?|test files?)\b)",
        r"\bwithout editing\b(?!\s+(?:tests?|test files?)\b)",
        r"\bwithout modifying\b",
        r"\bwithout changing\b(?!\s+(?:tests?|test files?)\b)",
        r"\bno changes\b",
        r"\bread-only\b",
        r"\binspect only\b",
        r"\bsummarize only\b",
    ]
    if any(re.search(pattern, lowered) for pattern in read_only_patterns):
        return False
    if re.search(r"\b(?:how|what|why)\s+(?:would|should|can)\b", lowered):
        return False
    if request_looks_like_issue_report(text):
        return True
    mutation_patterns = [
        r"\bimplement\b",
        r"\bfix\b",
        r"\bpatch\b",
        r"\brefactor\s+(?:[\w./-]+\.[A-Za-z0-9]+|[A-Za-z_][\w./-]*\s+to|code\s+to|module\s+to|tests?\s+to)",
        r"\bedit\b",
        r"\bupdate\b",
        r"\brewrite\b",
        r"\bmodify\b",
        r"\bchange\b",
        r"\bcreate\b",
        r"\bwrite\b",
        r"\badd\b",
        r"\bremove\b",
        r"\bdelete\b",
    ]
    return any(re.search(pattern, lowered) for pattern in mutation_patterns)


def request_forbids_tests(text: str) -> bool:
    lowered = text.lower()
    explicit_skip_patterns = [
        r"\b(?:do not|don't|dont|skip)\s+(?:run|rerun|execute)(?:\s+the)?\s+(?:(?:[\w-]+)\s+){0,4}(?:tests?|test suite)\b",
        r"\b(?:do not|don't|dont|skip)\s+(?:use\s+)?(?:pytest|unittest)\b",
        r"\bwithout\s+(?:running\s+)?(?:tests?|test suite)\b",
        r"\bwithout\s+(?:using\s+)?(?:pytest|unittest)\b",
        r"\bno tests?(?:\s+(?:needed|required|necessary))?\b",
    ]
    return any(re.search(pattern, lowered) for pattern in explicit_skip_patterns)


def request_requires_test_run(text: str) -> bool:
    lowered = text.lower()
    if request_forbids_tests(lowered):
        return False
    patterns = [
        r"\brun (?:the )?tests?\b",
        r"\brerun (?:the )?tests?\b",
        r"\bexecute (?:the )?tests?\b",
        r"\btest suite\b",
        r"\bkeep (?:the )?tests? (?:green|passing)\b",
        r"\btests? (?:stay|stays|remain|remains) (?:green|passing)\b",
        r"\bpytest\b",
        r"\bunittest\b",
        r"\brun_test\b",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def request_requires_code_mutation(text: str) -> bool:
    lowered = text.lower()
    if not request_requires_mutation(text):
        return False
    return bool(
        re.search(r"\b(?:fix|implement|patch|repair|refactor|change|update)\b", lowered)
        and (
            re.search(r"\b(?:implementation|source|code|bug|failing|failure|hidden tests?)\b", lowered)
            or re.search(r"\b(?:fix|patch|repair)\b.{0,120}\btests?\b", lowered)
        )
    )


def request_explicitly_allows_test_mutation(text: str) -> bool:
    lowered = text.lower()
    if any(
        phrase in lowered
        for phrase in [
            "update tests",
            "edit tests",
            "modify tests",
            "change tests",
            "rewrite tests",
            "add tests",
            "test file",
            "test files",
            "tests, and docs",
            "tests and docs",
        ]
    ):
        return True
    return bool(re.search(r"\btests?/[^\s,;:]+", lowered))


def request_forbids_test_mutation(text: str) -> bool:
    lowered = text.lower()
    if any(phrase in lowered for phrase in ["update tests", "edit tests", "modify tests", "change tests", "tests, and docs", "tests and docs"]):
        return False
    return any(
        phrase in lowered
        for phrase in [
            "edit only implementation",
            "only implementation files",
            "do not edit tests",
            "don't edit tests",
            "without editing tests",
            "leave tests unchanged",
        ]
    ) or bool(re.search(r"\b(?:fix|change|edit|update|implement|make)\b.{0,120}\bsrc/[^\s,;:]+", lowered))


def request_forbids_validation(text: str) -> bool:
    lowered = text.lower()
    return bool(
        re.search(
            r"\b(?:do not|don't|dont|skip|without|no)\b[^.?!;\n]{0,80}\b(?:validation|validate|validator|validators|lint|linter|typecheck|type\s+check|sanity\s+check)\b",
            lowered,
        )
    )


def validation_preferences(text: str) -> tuple[bool, bool]:
    forbid_validation = request_forbids_validation(text)
    forbid_tests = forbid_validation or request_forbids_tests(text)
    return not forbid_tests, not forbid_validation


def request_allows_any_validation(text: str) -> bool:
    allow_tests, allow_non_test_validation = validation_preferences(text)
    return allow_tests or allow_non_test_validation
