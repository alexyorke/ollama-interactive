import unittest

from ollama_code.controller.request_policy import (
    forbidden_tool_names_from_request,
    path_looks_like_doc_target,
    path_looks_like_test_file,
    request_allows_any_validation,
    request_allows_commit,
    request_allows_mutation,
    request_asks_exact_line_text,
    request_asks_if_command_works,
    request_asks_if_path_exists,
    request_asks_specific_file_line,
    request_asks_symbol_return,
    request_asks_token_only,
    request_benefits_from_systems_lens,
    request_benefits_from_todos,
    request_explicitly_allows_test_mutation,
    request_explicitly_requests_tool,
    request_explicitly_wants_clarification,
    request_expects_exact_tool_error,
    request_forbids_clarifying_questions,
    request_forbids_test_mutation,
    request_forbids_tests,
    request_forbids_validation,
    request_has_clarification_risk_signal,
    request_is_broad_or_ambiguous,
    request_is_continue_prompt,
    request_looks_like_issue_report,
    request_mentions_repeated_read,
    request_mentions_workspace_path,
    request_needs_exact_grounding,
    request_prefers_structured_file_tools,
    request_requires_code_mutation,
    request_requires_mutation,
    request_requires_test_run,
    request_requires_tools,
    request_targets_session_memory,
    requested_tool_names_from_request,
    tool_names_in_fragment,
    validation_preferences,
)


class ControllerRequestPolicyTests(unittest.TestCase):
    def test_path_classification_identifies_docs_and_tests(self) -> None:
        self.assertTrue(path_looks_like_doc_target("README.md"))
        self.assertTrue(path_looks_like_doc_target("./docs/setup.rst"))
        self.assertTrue(path_looks_like_doc_target("notes.txt"))
        self.assertFalse(path_looks_like_doc_target("src/readme_parser.py"))
        self.assertTrue(path_looks_like_test_file("tests/test_cli.py"))
        self.assertTrue(path_looks_like_test_file("src/foo_test.py"))
        self.assertTrue(path_looks_like_test_file("src/foo.spec.ts"))
        self.assertFalse(path_looks_like_test_file("src/foo.py"))

    def test_continue_prompt_policy_matches_only_resume_phrases(self) -> None:
        self.assertTrue(request_is_continue_prompt(" Continue "))
        self.assertTrue(request_is_continue_prompt("keep   going"))
        self.assertTrue(request_is_continue_prompt("fix it"))
        self.assertFalse(request_is_continue_prompt("continue by editing README.md"))
        self.assertFalse(request_is_continue_prompt("fix item 3 in TODO.md"))

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

    def test_tool_name_policy_parses_static_and_dynamic_tools(self) -> None:
        known = {"read_file", "run_shell", "run_test"}
        is_supported = lambda name: name in known or name.startswith("mcp.demo.")

        self.assertEqual(
            tool_names_in_fragment(
                "Use read_file and mcp.demo.echo.",
                known_tool_names=known,
                is_supported_tool_name=is_supported,
            ),
            {"read_file", "mcp.demo.echo"},
        )

    def test_tool_request_policy_respects_forbidden_constraints(self) -> None:
        known = {"read_file", "run_shell", "run_test"}
        is_supported = lambda name: name in known or name.startswith("mcp.demo.")
        text = "Use mcp.demo.echo and run_test, but do not use run_shell or mcp.demo.delete."

        forbidden = forbidden_tool_names_from_request(
            text,
            known_tool_names=known,
            is_supported_tool_name=is_supported,
        )
        requested = requested_tool_names_from_request(
            text,
            known_tool_names=known,
            is_supported_tool_name=is_supported,
            forbidden_tool_names=forbidden,
        )

        self.assertEqual(forbidden, {"mcp.demo.delete", "run_shell"})
        self.assertEqual(requested, {"mcp.demo.echo", "run_test"})
        self.assertTrue(
            request_explicitly_requests_tool(
                text,
                "run_shell",
                known_tool_names=known,
                is_supported_tool_name=is_supported,
            )
        )

    def test_requires_tools_policy_matches_repo_and_file_oriented_requests(self) -> None:
        self.assertTrue(request_requires_tools("Help me merge my missing branch back."))
        self.assertTrue(request_requires_tools("Read src/app.py and explain the bug."))
        self.assertTrue(request_requires_tools("Run pytest and keep tests green."))
        self.assertFalse(request_requires_tools("What is a decorator in Python?"))

    def test_structured_file_tool_and_session_memory_policy(self) -> None:
        self.assertTrue(request_prefers_structured_file_tools("Update docs/setup.md with the new option."))
        self.assertFalse(request_prefers_structured_file_tools("Run this shell command against docs/setup.md."))
        self.assertFalse(request_prefers_structured_file_tools("Explain the design."))
        self.assertTrue(request_targets_session_memory("What token did I ask you to remember earlier in this session?"))
        self.assertFalse(request_targets_session_memory("Read the current README token section."))

    def test_exact_request_predicates_are_controller_owned(self) -> None:
        self.assertTrue(request_allows_commit("Commit the validated fix."))
        self.assertFalse(request_allows_commit("Summarize the diff."))
        self.assertTrue(request_asks_if_command_works("Run this and tell me whether the command works."))
        self.assertTrue(request_asks_if_path_exists("Check if the file exists."))
        self.assertTrue(request_expects_exact_tool_error("If it fails, tell me the exact tool error."))
        self.assertTrue(request_mentions_repeated_read("Read it twice before answering."))
        self.assertTrue(request_asks_token_only("Reply with the exact token only."))
        self.assertTrue(request_asks_exact_line_text("Return the exact line text on line 4."))
        self.assertTrue(request_asks_specific_file_line("Read src/app.py line 12."))
        self.assertTrue(request_asks_symbol_return("Tell me what parse_args returns."))
        self.assertFalse(request_asks_specific_file_line("Read line 12."))
        self.assertFalse(request_asks_exact_line_text("Summarize line 4."))

    def test_planning_lens_policy_identifies_broad_and_systems_work(self) -> None:
        self.assertTrue(request_is_broad_or_ambiguous("Review the codebase for bugs."))
        self.assertTrue(request_is_broad_or_ambiguous("Inspect this repo."))
        self.assertFalse(request_is_broad_or_ambiguous("Inspect src/app.py."))
        self.assertTrue(request_mentions_workspace_path("Inspect src/app.py."))
        self.assertTrue(request_benefits_from_systems_lens("Profile this pipeline regression."))
        self.assertTrue(request_benefits_from_todos("Implement the feature.", mutation_required=True, test_run_required=False))
        self.assertTrue(request_benefits_from_todos("Run tests and report.", mutation_required=False, test_run_required=True))
        self.assertFalse(request_benefits_from_todos("Explain Python decorators.", mutation_required=False, test_run_required=False))

    def test_clarification_policy_is_controller_owned(self) -> None:
        self.assertTrue(request_forbids_clarifying_questions("Do not ask questions; just inspect the repo."))
        self.assertTrue(request_explicitly_wants_clarification("Ask me a question before you edit."))
        self.assertTrue(request_explicitly_wants_clarification("Do not assume the API shape."))
        self.assertTrue(request_has_clarification_risk_signal("Improve the CLI user experience."))
        self.assertTrue(request_has_clarification_risk_signal("Find bugs in this repo."))
        self.assertFalse(request_has_clarification_risk_signal("Read README.md and summarize it."))


if __name__ == "__main__":
    unittest.main()
