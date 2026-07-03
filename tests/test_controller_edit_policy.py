import unittest

from ollama_code.controller.edit_policy import (
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


if __name__ == "__main__":
    unittest.main()
