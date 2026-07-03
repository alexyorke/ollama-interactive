import unittest

from ollama_code.controller.feature_delivery import (
    cli_feature_capabilities,
    cli_proof_command_argvs,
    cli_readme_additions,
    typed_cli_flag_protocol_enabled,
)


class ControllerFeatureDeliveryTests(unittest.TestCase):
    def test_typed_cli_flag_protocol_follows_detected_bundle(self) -> None:
        self.assertTrue(typed_cli_flag_protocol_enabled(request_text="add --due-before to the CLI", request_is_cli_flag_bundle=True))
        self.assertFalse(typed_cli_flag_protocol_enabled(request_text="add --due-before to the CLI", request_is_cli_flag_bundle=False))

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


if __name__ == "__main__":
    unittest.main()
