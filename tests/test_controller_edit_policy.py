import unittest

from ollama_code.agent_protocol import ExactFileWriteSpec
from ollama_code.controller.edit_policy import (
    normalize_exact_literal_tool_call,
    normalize_snippet_symbol_edit_call,
    path_looks_like_code_file,
    shell_looks_like_file_mutation,
    snippet_symbol_argument_looks_like_text,
)


class ControllerEditPolicyTests(unittest.TestCase):
    def test_path_looks_like_code_file_uses_supported_edit_suffixes(self) -> None:
        self.assertTrue(path_looks_like_code_file("src/app.py"))
        self.assertTrue(path_looks_like_code_file("client\\app.tsx"))
        self.assertTrue(path_looks_like_code_file("scripts/run.sh"))
        self.assertFalse(path_looks_like_code_file("README.md"))
        self.assertFalse(path_looks_like_code_file("data/config.json"))

    def test_snippet_symbol_argument_distinguishes_code_text_from_symbols(self) -> None:
        self.assertFalse(snippet_symbol_argument_looks_like_text(""))
        self.assertFalse(snippet_symbol_argument_looks_like_text("parse_args"))
        self.assertFalse(snippet_symbol_argument_looks_like_text("Parser.parse"))
        self.assertTrue(snippet_symbol_argument_looks_like_text("return value + 1"))
        self.assertTrue(snippet_symbol_argument_looks_like_text("foo = bar"))
        self.assertTrue(snippet_symbol_argument_looks_like_text("if ready:\n    return True"))

    def test_shell_file_mutation_detection_flags_mutating_shell_shapes(self) -> None:
        self.assertTrue(shell_looks_like_file_mutation("echo hi > out.txt"))
        self.assertTrue(shell_looks_like_file_mutation("sed -i s/a/b/ file.txt"))
        self.assertTrue(shell_looks_like_file_mutation("touch created.txt"))
        self.assertFalse(shell_looks_like_file_mutation("grep TODO README.md"))
        self.assertFalse(shell_looks_like_file_mutation("python -m unittest"))

    def test_exact_literal_tool_call_normalization(self) -> None:
        spec = ExactFileWriteSpec(path="notes/token.txt", line="TOKEN_123")
        self.assertEqual(
            normalize_exact_literal_tool_call("replace_in_file", {"path": "wrong"}, exact_file_write=spec),
            (
                "write_file",
                {"path": "notes/token.txt", "content": "TOKEN_123\n"},
                "Normalized exact single-line file write to a deterministic write_file call.",
            ),
        )
        self.assertEqual(
            normalize_exact_literal_tool_call("read_file", {}, exact_file_write=spec),
            (
                "read_file",
                {"path": "notes/token.txt", "start": 1, "end": 1},
                "Normalized exact single-line confirmation read to the requested file and line range.",
            ),
        )
        self.assertEqual(normalize_exact_literal_tool_call("search", {"query": "TOKEN"}, exact_file_write=spec), ("search", {"query": "TOKEN"}, None))
        self.assertEqual(normalize_exact_literal_tool_call("write_file", {"path": "a"}, exact_file_write=None), ("write_file", {"path": "a"}, None))

    def test_snippet_symbol_edit_normalization(self) -> None:
        self.assertEqual(
            normalize_snippet_symbol_edit_call(
                "replace_symbol",
                {"path": "src/app.py", "symbol": "return old + 1", "content": "return new + 1"},
            ),
            (
                "replace_in_file",
                {"path": "src/app.py", "old": "return old + 1", "new": "return new + 1", "all": False},
                "Normalized snippet-style replace_symbol call to replace_in_file.",
            ),
        )
        self.assertEqual(
            normalize_snippet_symbol_edit_call("replace_symbol", {"path": "src/app.py", "symbol": "parse_args", "content": "body"}),
            ("replace_symbol", {"path": "src/app.py", "symbol": "parse_args", "content": "body"}, None),
        )
        self.assertEqual(normalize_snippet_symbol_edit_call("write_file", {}), ("write_file", {}, None))


if __name__ == "__main__":
    unittest.main()
