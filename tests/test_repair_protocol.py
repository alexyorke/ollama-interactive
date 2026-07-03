from __future__ import annotations

import unittest

from ollama_code.controller.repair_protocol import (
    build_repair_protocol_state,
    cli_patch_bundle_instruction,
    repair_decision_for_tool,
    repair_spec_broad_repair_hint,
    repair_spec_complete_plan,
    repair_spec_mutation_decision,
    repair_spec_required_proof_items,
    repair_spec_strategy_class,
)


class RepairProtocolTests(unittest.TestCase):
    def test_cli_bundle_requires_grounded_implementation_before_validation(self) -> None:
        obligations = [
            {"id": "code-change", "kind": "code_change", "label": "implement the requested code change"},
            {"id": "flag:--due-before", "kind": "feature_token", "label": 'prove the "--due-before" flag exists', "token": "--due-before", "feature_class": "flag"},
            {"id": "tests-update", "kind": "tests_update", "label": "add or update the requested tests"},
            {"id": "docs-update", "kind": "docs_update", "label": "update the requested docs"},
            {"id": "shell-proof", "kind": "shell_proof", "label": "prove the requested behavior with a shell command"},
        ]

        state = build_repair_protocol_state(
            obligations=obligations,
            obligation_statuses=[],
            successful_tool_results=[],
            recovery_states=[],
        )

        self.assertEqual(state.repair_strategy, "cli_patch_bundle")
        self.assertEqual(state.allowed_next_actions, ["grounding"])
        self.assertIn("grounded implementation target", state.validation_plan.blocked_until)
        decision = repair_decision_for_tool(state, tool_name="run_test", arguments={"command": "python -m unittest"})
        self.assertFalse(decision["allowed"])
        self.assertEqual(decision["action"], "validation")

    def test_cli_bundle_allows_source_test_and_doc_mutations_before_proof(self) -> None:
        obligations = [
            {"id": "code-change", "kind": "code_change", "label": "implement the requested code change"},
            {"id": "tests-update", "kind": "tests_update", "label": "add or update the requested tests"},
            {"id": "docs-update", "kind": "docs_update", "label": "update the requested docs"},
            {"id": "flag:--priority", "kind": "feature_token", "label": 'prove the "--priority" flag exists', "token": "--priority", "feature_class": "flag"},
        ]
        statuses = [
            {"id": "code-change", "status": "unproven"},
            {"id": "tests-update", "status": "unproven"},
            {"id": "docs-update", "status": "unproven"},
            {"id": "flag:--priority", "status": "unproven"},
        ]
        results = [
            {"name": "read_file", "arguments": {"path": "task_cli.py"}, "result": {"ok": True, "path": "task_cli.py", "output": "--priority"}},
        ]

        state = build_repair_protocol_state(
            obligations=obligations,
            obligation_statuses=statuses,
            successful_tool_results=results,
            recovery_states=[],
        )

        self.assertEqual(state.patch_plan.strategy if state.patch_plan else "", "cli_patch_bundle")
        self.assertTrue(state.patch_plan.complete if state.patch_plan else False)
        self.assertEqual(state.allowed_next_actions, ["implementation", "tests", "docs"])
        self.assertTrue(repair_decision_for_tool(state, tool_name="write_file", arguments={"path": "task_cli.py"})["allowed"])
        self.assertTrue(repair_decision_for_tool(state, tool_name="write_file", arguments={"path": "README.md"})["allowed"])
        self.assertTrue(repair_decision_for_tool(state, tool_name="write_file", arguments={"path": "tests/test_task_cli.py"})["allowed"])
        self.assertFalse(repair_decision_for_tool(state, tool_name="run_shell", arguments={"command": "python task_cli.py list --priority high"})["allowed"])

    def test_cli_bundle_rejects_body_only_implementation_mutation(self) -> None:
        obligations = [
            {"id": "code-change", "kind": "code_change", "label": "implement the requested code change"},
            {"id": "flag:--due-before", "kind": "feature_token", "label": 'prove the "--due-before" flag exists', "token": "--due-before", "feature_class": "flag"},
        ]
        results = [
            {"name": "read_symbol", "arguments": {"path": "task_cli.py", "symbol": "list_tasks"}, "result": {"ok": True, "path": "task_cli.py", "symbol": "list_tasks"}},
        ]
        state = build_repair_protocol_state(
            obligations=obligations,
            obligation_statuses=[],
            successful_tool_results=results,
            recovery_states=[],
        )

        decision = repair_decision_for_tool(
            state,
            tool_name="edit_intent",
            arguments={"path": "task_cli.py", "intent": "replace_body", "target": "list_tasks", "replacement": "return []"},
        )

        self.assertFalse(decision["allowed"])
        self.assertEqual(decision["violation"], "narrow_cli_mutation")
        self.assertIn("full command-surface bundle", decision["reason"])

    def test_cli_patch_bundle_instruction_names_coupled_surfaces(self) -> None:
        obligations = [
            {"id": "code-change", "kind": "code_change", "label": "implement the requested code change"},
            {"id": "flag:--due-before", "kind": "feature_token", "label": 'prove the "--due-before" flag exists', "token": "--due-before", "feature_class": "flag"},
            {"id": "tests-update", "kind": "tests_update", "label": "add or update the requested tests"},
            {"id": "docs-update", "kind": "docs_update", "label": "update the requested docs"},
            {"id": "shell-proof", "kind": "shell_proof", "label": "prove the requested behavior with a shell command"},
        ]
        results = [
            {"name": "code_outline", "arguments": {"path": "task_cli.py"}, "result": {"ok": True, "path": "task_cli.py"}},
        ]

        state = build_repair_protocol_state(
            obligations=obligations,
            obligation_statuses=[],
            successful_tool_results=results,
            recovery_states=[],
        )
        instruction = cli_patch_bundle_instruction(state)

        self.assertIsNotNone(instruction)
        assert instruction is not None
        self.assertIn("parser change", instruction)
        self.assertIn("callable behavior", instruction)
        self.assertIn("tests", instruction)
        self.assertIn("docs", instruction)
        self.assertIn("direct CLI behavior proof", instruction)

    def test_repair_spec_policy_selects_cli_surface_for_flag_obligations(self) -> None:
        strategy = repair_spec_strategy_class(
            target={"path": "task_cli.py"},
            obligations=[
                {
                    "id": "flag:--due-before",
                    "kind": "feature_token",
                    "feature_class": "flag",
                    "label": 'prove "--due-before"',
                }
            ],
            file_repair_allowed=True,
        )

        self.assertEqual(strategy, "cli_surface_repair")

    def test_repair_spec_complete_plan_and_hint_are_policy_level(self) -> None:
        state = {
            "path": "task_cli.py",
            "repair_strategy": "cli_surface_repair",
            "required_proof_items": ["prove --due-before"],
        }

        self.assertEqual(repair_spec_required_proof_items(state), ["prove --due-before"])
        self.assertIn("parser, behavior, docs, and proof", repair_spec_complete_plan(state))
        self.assertEqual(
            repair_spec_broad_repair_hint(state, file_repair_allowed=True),
            "write_file on task_cli.py so the CLI surface is repaired in one pass",
        )

    def test_repair_spec_mutation_decision_rejects_narrow_or_unrelated_retry(self) -> None:
        state = {"path": "task_cli.py", "repair_strategy": "cli_surface_repair"}

        unrelated = repair_spec_mutation_decision(
            state,
            proposed_tool_name="write_file",
            proposed_paths=["other.py"],
            repair_granularity="broad_file",
            file_repair_allowed=True,
        )
        narrow = repair_spec_mutation_decision(
            state,
            proposed_tool_name="replace_in_file",
            proposed_paths=["task_cli.py"],
            repair_granularity="narrow",
            file_repair_allowed=True,
        )
        broad = repair_spec_mutation_decision(
            state,
            proposed_tool_name="write_file",
            proposed_paths=["task_cli.py"],
            repair_granularity="broad_file",
            file_repair_allowed=True,
        )

        self.assertFalse(unrelated["allowed"])
        self.assertIn("unrelated files", unrelated["reason"])
        self.assertFalse(narrow["allowed"])
        self.assertIn("small speculative edit", narrow["reason"])
        self.assertTrue(broad["allowed"])


if __name__ == "__main__":
    unittest.main()
