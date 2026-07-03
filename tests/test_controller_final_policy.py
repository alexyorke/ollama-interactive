import unittest

from ollama_code.controller.final_policy import (
    final_acknowledges_missing_path,
    final_claims_file_mutation,
    final_claims_path_exists,
    final_claims_run_shell_success,
    final_claims_test_success,
    final_claims_timeout_success,
    final_requires_verification,
)


class ControllerFinalPolicyTests(unittest.TestCase):
    def test_success_claims_reject_failure_language(self) -> None:
        self.assertTrue(final_claims_timeout_success("The service is healthy and running."))
        self.assertFalse(final_claims_timeout_success("The service timed out and still needs work."))
        self.assertTrue(final_claims_run_shell_success("The command works and verification passed."))
        self.assertFalse(final_claims_run_shell_success("The command is missing and not found."))

    def test_path_claims_distinguish_exists_from_missing(self) -> None:
        self.assertTrue(final_claims_path_exists("The requested path exists."))
        self.assertFalse(final_claims_path_exists("The requested path does not exist."))
        self.assertTrue(final_acknowledges_missing_path("No such file or directory."))
        self.assertFalse(final_acknowledges_missing_path("The file exists."))

    def test_mutation_and_test_success_claims(self) -> None:
        self.assertTrue(final_claims_file_mutation("I updated README.md."))
        self.assertTrue(final_claims_file_mutation("The file was rewritten."))
        self.assertFalse(final_claims_file_mutation("I inspected README.md."))
        self.assertTrue(final_claims_test_success("All tests passed."))
        self.assertTrue(final_claims_test_success("Successfully ran the unit tests."))
        self.assertFalse(final_claims_test_success("Tests still need to be run."))

    def test_final_requires_verification_for_obligations_mutation_and_risky_tools(self) -> None:
        base = {
            "has_request_obligations": False,
            "has_required_or_forbidden_tools": False,
            "mutation_verified_this_turn": False,
            "final_claims_mutation": False,
            "has_expected_exact_file_line": False,
            "tool_call_count": 0,
            "request_needs_exact_grounding": False,
            "tool_names": set(),
            "risky_verification_tool_names": {"run_shell"},
        }

        self.assertFalse(final_requires_verification(**base))
        self.assertTrue(final_requires_verification(**{**base, "has_request_obligations": True}))
        self.assertTrue(final_requires_verification(**{**base, "final_claims_mutation": True}))
        self.assertTrue(final_requires_verification(**{**base, "tool_call_count": 2}))
        self.assertTrue(final_requires_verification(**{**base, "tool_call_count": 1, "request_needs_exact_grounding": True}))
        self.assertTrue(final_requires_verification(**{**base, "tool_names": {"run_shell"}}))


if __name__ == "__main__":
    unittest.main()
