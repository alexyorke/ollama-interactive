import unittest

from scripts import trajectory_error_profile as error_profile


class TrajectoryErrorProfileTests(unittest.TestCase):
    def test_classifies_tool_result_errors_without_task_description_matches(self) -> None:
        rows = [
            {
                "trajectory": [
                    {"role": "user", "content": "Task mentions SyntaxError as text only."},
                    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "run_test", "arguments": "{}"}}]},
                    {"role": "tool", "name": "run_test", "content": "FAILED tests/test_app.py::test_x\nE   assert 1 == 2"},
                    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "run_shell", "arguments": "{}"}}]},
                    {"role": "tool", "name": "run_shell", "content": "usage: pytest [options]\nerror: unrecognized arguments: --wat"},
                ]
            }
        ]

        summary = error_profile.summarize_dataset("sample", "openhands", rows)

        self.assertEqual(summary["error_counts"]["test_assertion"], 1)
        self.assertEqual(summary["error_counts"]["invalid_args"], 1)
        self.assertEqual(summary["top_shell_errors"][0]["shell_error"], "unrecognized_argument")
        self.assertNotIn("syntax_error", summary["error_counts"])
        self.assertEqual(summary["examples"]["test_assertion"][0]["tool"], "run_test")

    def test_classifies_shell_syntax_and_command_errors(self) -> None:
        rows = [
            {
                "trajectory": [
                    {"role": "tool", "name": "run_shell", "content": "bash: -c: line 2: syntax error: unexpected end of file"},
                    {"role": "tool", "name": "execute_bash", "content": "bash: foozle: command not found"},
                    {"role": "tool", "name": "run_shell", "content": "syntax error near unexpected token `fi'"},
                ]
            }
        ]

        summary = error_profile.summarize_dataset("sample", "openhands", rows)
        shell_counts = {item["shell_error"]: item["count"] for item in summary["top_shell_errors"]}

        self.assertEqual(shell_counts["bash_unexpected_eof"], 1)
        self.assertEqual(shell_counts["command_not_found"], 1)
        self.assertEqual(shell_counts["bash_unexpected_token"], 1)

    def test_repeated_error_loop_counts_same_tool_error(self) -> None:
        rows = [
            {
                "trajectory": [
                    {"role": "tool", "name": "read_file", "content": "FileNotFoundError: missing.py"},
                    {"role": "tool", "name": "read_file", "content": "FileNotFoundError: missing.py"},
                ]
            }
        ]

        summary = error_profile.summarize_dataset("sample", "openhands", rows)

        self.assertEqual(summary["repeated_error_loops"][0]["tool_error"], "read_file:path_missing")


if __name__ == "__main__":
    unittest.main()
