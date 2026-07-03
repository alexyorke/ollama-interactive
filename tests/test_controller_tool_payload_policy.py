import unittest

from ollama_code.controller.tool_payload_policy import normalize_payload


def _is_supported_tool_name(name: str) -> bool:
    return name in {"read_file", "run_test"} or name.startswith("mcp.demo.")


class ControllerToolPayloadPolicyTests(unittest.TestCase):
    def test_trims_type_and_name_without_other_changes(self) -> None:
        self.assertEqual(
            normalize_payload({"type": " unknown ", "name": " custom ", "arguments": "raw"}, is_supported_tool_name=_is_supported_tool_name),
            {"type": "unknown", "name": "custom", "arguments": "raw"},
        )

    def test_tool_name_in_type_field_becomes_tool_call(self) -> None:
        self.assertEqual(
            normalize_payload({"type": " run_test ", "arguments": None}, is_supported_tool_name=_is_supported_tool_name),
            {"type": "tool", "arguments": {}, "name": "run_test"},
        )

    def test_function_style_supported_name_becomes_tool_call(self) -> None:
        self.assertEqual(
            normalize_payload({"type": "function", "name": " read_file ", "arguments": None}, is_supported_tool_name=_is_supported_tool_name),
            {"type": "tool", "name": "read_file", "arguments": {}},
        )

    def test_final_tool_alias_uses_message_or_argument_content(self) -> None:
        self.assertEqual(
            normalize_payload({"type": "tool", "name": "final", "message": " done "}, is_supported_tool_name=_is_supported_tool_name),
            {"type": "final", "message": "done"},
        )
        self.assertEqual(
            normalize_payload({"type": "tool_call", "name": "final", "arguments": {"content": " finished "}}, is_supported_tool_name=_is_supported_tool_name),
            {"type": "final", "message": "finished"},
        )

    def test_dynamic_tool_name_predicate_is_used(self) -> None:
        self.assertEqual(
            normalize_payload({"type": "mcp.demo.search", "arguments": []}, is_supported_tool_name=_is_supported_tool_name),
            {"type": "tool", "arguments": {}, "name": "mcp.demo.search"},
        )

    def test_unsupported_payload_is_not_forced_into_tool_call(self) -> None:
        self.assertEqual(
            normalize_payload({"type": "function", "name": "missing_tool", "arguments": None}, is_supported_tool_name=_is_supported_tool_name),
            {"type": "function", "name": "missing_tool", "arguments": None},
        )


if __name__ == "__main__":
    unittest.main()
