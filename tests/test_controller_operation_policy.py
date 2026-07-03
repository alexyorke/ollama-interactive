import unittest

from ollama_code.controller.operation_policy import (
    project_function_rename_already_satisfied,
    project_function_rename_operations,
)


class ControllerOperationPolicyTests(unittest.TestCase):
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
