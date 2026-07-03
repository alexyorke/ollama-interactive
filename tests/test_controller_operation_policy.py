import unittest

from ollama_code.controller.operation_policy import (
    clean_return_expression,
    project_function_rename_already_satisfied,
    project_function_rename_operations,
    symbol_return_update_operations_from_source,
    symbol_return_update_spec,
    workflow_config_update_operations,
    workflow_config_update_operations_from_source,
    workflow_config_update_spec,
)


class ControllerOperationPolicyTests(unittest.TestCase):
    def test_clean_return_expression_removes_prompt_tail_and_punctuation(self) -> None:
        self.assertEqual(clean_return_expression(" `new_value`; then run tests"), "new_value")
        self.assertEqual(clean_return_expression('"result" and do not touch docs'), "result")
        self.assertEqual(clean_return_expression("foo + bar."), "foo + bar")

    def test_symbol_return_update_spec_parses_request(self) -> None:
        self.assertEqual(
            symbol_return_update_spec("In src/app.py change parse() so it returns `new_value` instead of `old_value`; then run tests."),
            {"path": "src/app.py", "symbol": "parse", "new_expr": "new_value", "old_expr": "old_value"},
        )
        self.assertEqual(
            symbol_return_update_spec("Update src/app.ts so compute() returns 2 instead of 1."),
            {"path": "src/app.ts", "symbol": "compute", "new_expr": "2", "old_expr": "1"},
        )
        self.assertIsNone(symbol_return_update_spec("Update src/app.py without return details."))

    def test_symbol_return_update_operations_replaces_matching_return(self) -> None:
        source = "def parse():\n    return old_value\n"
        self.assertEqual(
            symbol_return_update_operations_from_source(
                path="src/app.py",
                symbol="parse",
                new_expr="new_value",
                old_expr="old_value",
                source=source,
                requested_tool_names=set(),
                required_tool_names=set(),
            ),
            [("replace_in_file", {"path": "src/app.py", "old": "    return old_value", "new": "    return new_value"})],
        )

    def test_symbol_return_update_operations_adds_requested_grounding_tools(self) -> None:
        source = "export function compute() {\n  return 1;\n}\n"
        self.assertEqual(
            symbol_return_update_operations_from_source(
                path="src/app.ts",
                symbol="compute",
                new_expr="2",
                old_expr="1",
                source=source,
                requested_tool_names={"search_symbols"},
                required_tool_names={"read_symbol"},
            ),
            [
                ("search_symbols", {"query": "compute", "path": "src/app.ts"}),
                ("read_symbol", {"path": "src/app.ts", "symbol": "compute", "include_context": 0}),
                ("replace_in_file", {"path": "src/app.ts", "old": "  return 1;", "new": "  return 2;"}),
            ],
        )

    def test_symbol_return_update_operations_rejects_missing_or_noop_replacement(self) -> None:
        source = "def parse():\n    return old_value\n"
        self.assertIsNone(
            symbol_return_update_operations_from_source(
                path="src/app.py",
                symbol="parse",
                new_expr="new_value",
                old_expr="missing",
                source=source,
                requested_tool_names=set(),
                required_tool_names=set(),
            )
        )
        self.assertIsNone(
            symbol_return_update_operations_from_source(
                path="src/app.py",
                symbol="parse",
                new_expr="old_value",
                old_expr="old_value",
                source=source,
                requested_tool_names=set(),
                required_tool_names=set(),
            )
        )

    def test_workflow_config_update_spec_parses_path_and_command(self) -> None:
        self.assertEqual(
            workflow_config_update_spec(
                "Update github/workflows/ci.yml workflow to add pull_request and change its unittest command to `python -m unittest discover -s tests -v`."
            ),
            {
                "path": ".github/workflows/ci.yml",
                "new_command": "python -m unittest discover -s tests -v",
            },
        )
        self.assertIsNone(workflow_config_update_spec("Update .github/workflows/ci.yml to add pull_request."))
        self.assertIsNone(workflow_config_update_spec("Change .github/workflows/ci.yml command to `pytest` without workflow wording."))

    def test_workflow_config_update_operations_add_pull_request_and_replace_command(self) -> None:
        source = (
            "name: CI\n"
            "on:\n"
            "  push:\n"
            "    branches: [main]\n"
            "jobs:\n"
            "  test:\n"
            "    steps:\n"
            "      - run: python -m unittest\n"
        )
        self.assertEqual(
            workflow_config_update_operations_from_source(
                path=".github/workflows/ci.yml",
                new_command="python -m unittest discover -s tests -v",
                source=source,
            ),
            [
                ("read_file", {"path": ".github/workflows/ci.yml"}),
                (
                    "replace_in_file",
                    {
                        "path": ".github/workflows/ci.yml",
                        "old": "on:\n  push:\n    branches: [main]",
                        "new": "on:\n  push:\n    branches: [main]\n  pull_request:",
                    },
                ),
                (
                    "replace_in_file",
                    {
                        "path": ".github/workflows/ci.yml",
                        "old": "      - run: python -m unittest",
                        "new": "      - run: python -m unittest discover -s tests -v",
                    },
                ),
            ],
        )

    def test_workflow_config_update_operations_handles_existing_pull_request(self) -> None:
        source = (
            "name: CI\n"
            "on:\n"
            "  pull_request:\n"
            "jobs:\n"
            "  test:\n"
            "    steps:\n"
            "      run: python -m unittest\n"
        )
        self.assertEqual(
            workflow_config_update_operations_from_source(
                path=".github/workflows/ci.yml",
                new_command="python -m unittest discover -s tests -v",
                source=source,
            ),
            [
                ("read_file", {"path": ".github/workflows/ci.yml"}),
                (
                    "replace_in_file",
                    {
                        "path": ".github/workflows/ci.yml",
                        "old": "      run: python -m unittest",
                        "new": "      run: python -m unittest discover -s tests -v",
                    },
                ),
            ],
        )

    def test_workflow_config_update_operations_rejects_noop_or_missing_on_block(self) -> None:
        self.assertIsNone(
            workflow_config_update_operations_from_source(
                path=".github/workflows/ci.yml",
                new_command="python -m unittest discover -s tests -v",
                source="name: CI\njobs:\n  test:\n    steps:\n      - run: python -m unittest\n",
            )
        )
        self.assertIsNone(
            workflow_config_update_operations_from_source(
                path=".github/workflows/ci.yml",
                new_command="python -m unittest discover -s tests -v",
                source="name: CI\non:\n  pull_request:\njobs:\n  test:\n    steps:\n      - run: python -m unittest discover -s tests -v\n",
            )
        )

    def test_workflow_config_update_operations_combines_spec_and_source(self) -> None:
        request = (
            "Update .github/workflows/ci.yml workflow to add pull_request and change its unittest command "
            "to `python -m unittest discover -s tests -v`."
        )
        source = "name: CI\non:\n  push:\njobs:\n  test:\n    steps:\n      - run: python -m unittest\n"
        operations = workflow_config_update_operations(request_text=request, source=source)
        self.assertIsNotNone(operations)
        self.assertEqual(operations[0], ("read_file", {"path": ".github/workflows/ci.yml"}))

    def test_project_function_rename_operations_parse_grounded_request(self) -> None:
        self.assertEqual(
            project_function_rename_operations("Rename the project API from old_name(x) to new_name(x), including callers and tests."),
            [
                (
                    "edit_intent",
                    {
                        "path": ".",
                        "intent": "rename",
                        "target": "old_name",
                        "replacement": "new_name",
                        "scope": "project",
                    },
                )
            ],
        )

    def test_project_function_rename_operations_reject_weak_or_noop_requests(self) -> None:
        self.assertIsNone(project_function_rename_operations("Rename from old_name(x) to new_name(x)."))
        self.assertIsNone(project_function_rename_operations("Explain the API from old_name(x) to new_name(x)."))
        self.assertIsNone(project_function_rename_operations("Rename the project API from same(x) to same(x)."))
        self.assertIsNone(project_function_rename_operations("Rename this project symbol later."))

    def test_project_function_rename_already_satisfied_requires_matching_success(self) -> None:
        request = "Refactor the source function from old_name(x) to new_name(x) across callers."
        matching_result = {
            "name": "edit_intent",
            "arguments": {
                "path": ".",
                "intent": "rename",
                "target": "old_name",
                "replacement": "new_name",
                "scope": "project",
            },
            "result": {"ok": True},
        }
        self.assertTrue(project_function_rename_already_satisfied(request_text=request, successful_tool_results=[matching_result]))

    def test_project_function_rename_already_satisfied_rejects_mismatches(self) -> None:
        request = "Refactor the source function from old_name(x) to new_name(x) across callers."
        self.assertFalse(project_function_rename_already_satisfied(request_text=request, successful_tool_results=[]))
        self.assertFalse(
            project_function_rename_already_satisfied(
                request_text=request,
                successful_tool_results=[
                    {
                        "name": "edit_intent",
                        "arguments": {
                            "path": ".",
                            "intent": "rename",
                            "target": "old_name",
                            "replacement": "other_name",
                            "scope": "project",
                        },
                        "result": {"ok": True},
                    }
                ],
            )
        )
        self.assertFalse(
            project_function_rename_already_satisfied(
                request_text=request,
                successful_tool_results=[
                    {
                        "name": "edit_intent",
                        "arguments": {
                            "path": ".",
                            "intent": "rename",
                            "target": "old_name",
                            "replacement": "new_name",
                            "scope": "project",
                        },
                        "result": {"ok": False},
                    }
                ],
            )
        )


if __name__ == "__main__":
    unittest.main()
