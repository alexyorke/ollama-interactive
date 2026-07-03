import ast
import unittest

from ollama_code.controller.feature_delivery import (
    cli_feature_capabilities,
    cli_proof_command_argvs,
    cli_readme_additions,
    cli_test_additions,
    derive_request_obligations,
    request_is_cli_flag_bundle,
    request_looks_like_python_test_driven_repair,
    request_obligation_proof_status,
    spec_guided_repair_has_actionable_spec,
    typed_cli_flag_protocol_enabled,
)


class ControllerFeatureDeliveryTests(unittest.TestCase):
    def test_typed_cli_flag_protocol_follows_detected_bundle(self) -> None:
        self.assertTrue(typed_cli_flag_protocol_enabled(request_text="add --due-before to the CLI", request_is_cli_flag_bundle=True))
        self.assertFalse(typed_cli_flag_protocol_enabled(request_text="add --due-before to the CLI", request_is_cli_flag_bundle=False))
        self.assertTrue(typed_cli_flag_protocol_enabled(request_text="add --due-before flag to argparse"))

    def test_request_is_cli_flag_bundle_requires_flag_and_cli_surface(self) -> None:
        self.assertTrue(request_is_cli_flag_bundle("Add a --due-before option to the command parser."))
        self.assertTrue(request_is_cli_flag_bundle("Support --priority in the CLI."))
        self.assertFalse(request_is_cli_flag_bundle("Mention --priority in README only."))
        self.assertFalse(request_is_cli_flag_bundle("Add a stats command without flags."))

    def test_python_test_driven_repair_classifier_accepts_focused_source_request(self) -> None:
        self.assertTrue(
            request_looks_like_python_test_driven_repair(
                request_text="Fix src/calculator.py from the tests and run tests.",
                session_memory_request=False,
                mutation_required=True,
                test_run_required=True,
                required_tool_names=set(),
                forbidden_tool_names=set(),
                default_test_command_configured=True,
                requested_mutation_paths={"src/calculator.py"},
                path_looks_like_test_file=lambda path: path.startswith("tests/") or path.endswith("_test.py"),
            )
        )

    def test_python_test_driven_repair_classifier_rejects_broad_docs_or_api_work(self) -> None:
        is_test_path = lambda path: path.startswith("tests/") or path.endswith("_test.py")

        self.assertFalse(
            request_looks_like_python_test_driven_repair(
                request_text="Refactor src/api.py and update README docs.",
                session_memory_request=False,
                mutation_required=True,
                test_run_required=True,
                required_tool_names=set(),
                forbidden_tool_names=set(),
                default_test_command_configured=True,
                requested_mutation_paths={"src/api.py", "README.md"},
                path_looks_like_test_file=is_test_path,
            )
        )
        self.assertFalse(
            request_looks_like_python_test_driven_repair(
                request_text="Rename the public API in src/api.py and all callsites.",
                session_memory_request=False,
                mutation_required=True,
                test_run_required=True,
                required_tool_names=set(),
                forbidden_tool_names=set(),
                default_test_command_configured=True,
                requested_mutation_paths={"src/api.py"},
                path_looks_like_test_file=is_test_path,
            )
        )

    def test_python_test_driven_repair_classifier_rejects_unsafe_runtime_context(self) -> None:
        base = {
            "request_text": "Implement the Python exercise from the tests.",
            "session_memory_request": False,
            "mutation_required": True,
            "test_run_required": True,
            "required_tool_names": set(),
            "forbidden_tool_names": set(),
            "default_test_command_configured": True,
            "requested_mutation_paths": set(),
            "path_looks_like_test_file": lambda path: path.startswith("tests/"),
        }

        self.assertFalse(request_looks_like_python_test_driven_repair(**{**base, "default_test_command_configured": False}))
        self.assertFalse(request_looks_like_python_test_driven_repair(**{**base, "required_tool_names": {"read_file"}}))
        self.assertFalse(request_looks_like_python_test_driven_repair(**{**base, "session_memory_request": True}))

    def test_spec_guided_repair_has_actionable_spec_accepts_explicit_failure_signals(self) -> None:
        weak_spec = {"examples": [], "definitions": []}

        self.assertTrue(
            spec_guided_repair_has_actionable_spec(
                source_text="def parse(value):\n    return value\n",
                quick_spec=weak_spec,
                failed_output="ModuleNotFoundError: No module named 'app'",
                split_test_example=lambda example: ("", "", ""),
                test_spec_call_name=lambda call: "",
            )
        )
        self.assertTrue(
            spec_guided_repair_has_actionable_spec(
                source_text="def parse(value):\n    return value\n",
                quick_spec={"examples": [{"example": "parse('x') != 'x'"}], "definitions": []},
                failed_output="",
                split_test_example=lambda example: ("", "", ""),
                test_spec_call_name=lambda call: "",
            )
        )

    def test_spec_guided_repair_has_actionable_spec_rejects_weak_large_spec(self) -> None:
        source_text = "\n".join(f"line_{index} = {index}" for index in range(230))

        self.assertFalse(
            spec_guided_repair_has_actionable_spec(
                source_text=source_text,
                quick_spec={"examples": [{"example": "parse('x') == 'x'"}], "definitions": [{"name": "parse"}]},
                failed_output="",
                split_test_example=lambda example: ("value", "parse('x')", "'x'"),
                test_spec_call_name=lambda call: "parse",
            )
        )

    def test_spec_guided_repair_has_actionable_spec_accepts_small_literal_example(self) -> None:
        def call_name(call: ast.Call) -> str:
            return call.func.id if isinstance(call.func, ast.Name) else ""

        self.assertTrue(
            spec_guided_repair_has_actionable_spec(
                source_text="def parse(value):\n    return value\n",
                quick_spec={
                    "examples": [{"symbol": "parse", "example": "parse('x') == 'X'"}],
                    "definitions": [{"name": "parse"}],
                },
                failed_output="",
                split_test_example=lambda example: ("value", "parse('x')", "'X'"),
                test_spec_call_name=call_name,
            )
        )

    def test_cli_feature_capabilities_detect_due_before_priority_and_limit(self) -> None:
        source = "parser.add_parser('list')\nlist_parser.add_argument('--priority')\nlist_parser.add_argument('--due-before')\n"

        caps = cli_feature_capabilities(source, "Add --due-before and preserve --limit behavior")

        self.assertTrue(caps.has_priority_filter)
        self.assertTrue(caps.has_due_before)
        self.assertFalse(caps.has_limit_flag)

    def test_cli_proof_commands_cover_due_before_and_combined_priority(self) -> None:
        source = "parser.add_parser('list')\nlist_parser.add_argument('--priority')\nlist_parser.add_argument('--due-before')\n"

        commands = cli_proof_command_argvs("task_cli.py", source, "Add --due-before.")

        self.assertEqual(
            commands,
            [
                ["task_cli.py", "list", "--priority", "high"],
                ["task_cli.py", "list", "--due-before", "2026-07-06"],
                ["task_cli.py", "list", "--priority", "high", "--due-before", "2026-07-06"],
            ],
        )

    def test_cli_readme_additions_skip_existing_content(self) -> None:
        source = "parser.add_parser('list')\nlist_parser.add_argument('--priority')\nlist_parser.add_argument('--due-before')\n"

        additions = cli_readme_additions(source, "Update README for --due-before.", "Use --priority already.\n")

        self.assertEqual(additions, ["- `list --due-before YYYY-MM-DD` filters tasks by due date and can be combined with `--priority`."])

    def test_cli_test_additions_create_due_before_regressions(self) -> None:
        source = "parser.add_parser('list')\nlist_parser.add_argument('--priority')\nlist_parser.add_argument('--due-before')\n"

        additions = cli_test_additions(source, "Add --due-before tests.", "def _run(*args: str): ...\n", "_run")

        joined = "\n".join(additions)
        self.assertIn("test_due_before_filter", joined)
        self.assertIn("--priority", joined)
        self.assertIn("2026-99-99", joined)

    def test_cli_test_additions_skip_existing_feature_tests(self) -> None:
        source = "parser.add_parser('list')\nlist_parser.add_argument('--due-before')\n"

        self.assertEqual(cli_test_additions(source, "Add --due-before tests.", "--due-before already covered", "_run"), [])

    def test_derive_request_obligations_extracts_feature_delivery_contract(self) -> None:
        obligations = derive_request_obligations(
            request_text=(
                "Add an export_ndjson(rows) function. Add tests, update README, "
                "run tests, and prove the behavior with a shell command."
            ),
            required_tool_names={"read_file"},
            doc_targets=["README.md"],
            code_mutation_required=True,
            test_run_required=True,
        )

        self.assertEqual(
            [item["id"] for item in obligations],
            [
                "tool:read_file",
                "code-change",
                "tests-run",
                "tests-update",
                "shell-proof",
                "docs-update",
                "function:export_ndjson",
            ],
        )

    def test_request_obligation_proof_status_requires_source_and_behavior_for_flags(self) -> None:
        obligations = [
            {
                "id": "flag:--due-before",
                "kind": "feature_token",
                "label": 'prove the "--due-before" flag exists',
                "token": "--due-before",
                "feature_class": "flag",
            }
        ]

        source_only = request_obligation_proof_status(
            obligations=obligations,
            successful_tool_results=[
                {
                    "name": "read_file",
                    "arguments": {"path": "task_cli.py"},
                    "result": {"ok": True, "path": "task_cli.py", "output": "parser.add_argument('--due-before')"},
                }
            ],
            required_tool_names=set(),
            mutated_paths=set(),
            is_doc_path=lambda path: path.endswith(".md"),
            is_test_path=lambda path: path.startswith("tests/"),
            test_ran=False,
            shell_proof_ran=False,
            truncate_text=lambda text, limit: text[:limit],
        )
        proven = request_obligation_proof_status(
            obligations=obligations,
            successful_tool_results=[
                {
                    "name": "read_file",
                    "arguments": {"path": "task_cli.py"},
                    "result": {"ok": True, "path": "task_cli.py", "output": "parser.add_argument('--due-before')"},
                },
                {
                    "name": "run_shell",
                    "arguments": {"command": "python task_cli.py list --due-before 2026-07-06"},
                    "result": {"ok": True, "command": "python task_cli.py list --due-before 2026-07-06"},
                },
            ],
            required_tool_names=set(),
            mutated_paths=set(),
            is_doc_path=lambda path: path.endswith(".md"),
            is_test_path=lambda path: path.startswith("tests/"),
            test_ran=False,
            shell_proof_ran=True,
            truncate_text=lambda text, limit: text[:limit],
        )

        self.assertEqual(source_only[0]["status"], "unproven")
        self.assertIn("both implementation proof and behavior proof", source_only[0]["guidance"])
        self.assertEqual(proven[0]["status"], "proven")
        self.assertIn("task_cli.py", proven[0]["evidence"])


if __name__ == "__main__":
    unittest.main()
