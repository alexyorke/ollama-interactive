import unittest

from ollama_code.controller.request_policy import (
    request_allows_any_validation,
    request_allows_mutation,
    request_explicitly_allows_test_mutation,
    request_forbids_test_mutation,
    request_forbids_tests,
    request_forbids_validation,
    request_looks_like_issue_report,
    request_needs_exact_grounding,
    request_requires_code_mutation,
    request_requires_mutation,
    request_requires_test_run,
    validation_preferences,
)


class ControllerRequestPolicyTests(unittest.TestCase):
    def test_mutation_policy_respects_read_only_and_issue_report_shapes(self) -> None:
        self.assertFalse(request_requires_mutation("Inspect only; do not edit app.py."))
        self.assertFalse(request_requires_mutation("How should we refactor this later?"))
        self.assertTrue(request_requires_mutation("Fix `app.py`; it returns the wrong value."))
        self.assertTrue(request_looks_like_issue_report("`app.py` fails when input is empty."))
        self.assertTrue(request_allows_mutation("Please add a stats command."))

    def test_exact_grounding_policy_detects_exact_text_requests(self) -> None:
        self.assertTrue(request_needs_exact_grounding("Read docs/spec.md and reply with the token only."))
        self.assertTrue(request_needs_exact_grounding("What does line 12 say?"))
        self.assertTrue(request_needs_exact_grounding("Return exactly the first line."))
        self.assertFalse(request_needs_exact_grounding("Summarize docs/spec.md."))

    def test_code_mutation_policy_requires_source_or_bug_context(self) -> None:
        self.assertTrue(request_requires_code_mutation("Fix the failing tests by repairing source code."))
        self.assertTrue(request_requires_code_mutation("Patch the implementation bug in src/app.py."))
        self.assertFalse(request_requires_code_mutation("Update README with setup notes."))

    def test_test_run_policy_honors_explicit_skip(self) -> None:
        self.assertTrue(request_requires_test_run("Add the feature and keep tests green."))
        self.assertTrue(request_requires_test_run("Run pytest after the change."))
        self.assertFalse(request_requires_test_run("Add the feature without running tests."))
        self.assertTrue(request_forbids_tests("Do not run the unit tests."))

    def test_test_mutation_policy_distinguishes_allowed_and_forbidden_edits(self) -> None:
        self.assertTrue(request_explicitly_allows_test_mutation("Update tests and docs."))
        self.assertTrue(request_explicitly_allows_test_mutation("Edit tests/test_cli.py."))
        self.assertFalse(request_forbids_test_mutation("Update tests and docs."))
        self.assertTrue(request_forbids_test_mutation("Fix src/app.py but leave tests unchanged."))

    def test_validation_preferences_honor_validation_and_test_forbids(self) -> None:
        self.assertTrue(request_forbids_validation("Make the docs change without lint or validation."))
        self.assertEqual(validation_preferences("Make the docs change without lint or validation."), (False, False))
        self.assertEqual(validation_preferences("Make the change without running tests."), (False, True))
        self.assertFalse(request_allows_any_validation("Make the docs change without lint or validation."))


if __name__ == "__main__":
    unittest.main()
