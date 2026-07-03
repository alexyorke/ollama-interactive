import unittest

from ollama_code.controller.operation_policy import (
    project_function_rename_already_satisfied,
    project_function_rename_operations,
    workflow_config_update_operations,
    workflow_config_update_operations_from_source,
    workflow_config_update_spec,
)


class ControllerOperationPolicyTests(unittest.TestCase):
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
