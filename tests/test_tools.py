from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ollama_code.tools import ToolExecutor


class ToolExecutorTests(unittest.TestCase):
    def test_list_and_read_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "alpha.txt").write_text("line1\nline2\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            listed = tools.list_files()
            read = tools.read_file("alpha.txt", start=2, end=2)

        self.assertTrue(listed["ok"])
        self.assertIn("alpha.txt", listed["output"])
        self.assertEqual(read["output"], "   2 | line2")

    def test_write_and_replace_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = ToolExecutor(root, approval_mode="auto")
            write = tools.write_file("nested/file.txt", "hello\n")
            replace = tools.replace_in_file("nested/file.txt", "hello", "goodbye")
            final_text = (root / "nested" / "file.txt").read_text(encoding="utf-8")

        self.assertTrue(write["ok"])
        self.assertTrue(replace["ok"])
        self.assertEqual(final_text, "goodbye\n")

    def test_replace_rejects_ambiguous_short_snippet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.txt").write_text("live smoke ok\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.replace_in_file("sample.txt", "ok", "passed", replace_all=True)
        self.assertFalse(result["ok"])
        self.assertIn("ambiguous", result["summary"])

    def test_replace_can_match_whole_word(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "sample.txt"
            sample.write_text("live smoke ok\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.replace_in_file("sample.txt", "ok", "passed", match_whole_word=True)
            final_text = sample.read_text(encoding="utf-8")
        self.assertTrue(result["ok"])
        self.assertEqual(final_text, "live smoke passed\n")

    def test_read_only_blocks_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="read-only")
            result = tools.write_file("blocked.txt", "data")
        self.assertFalse(result["ok"])
        self.assertIn("read-only", result["summary"])

    def test_path_escape_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.execute("read_file", {"path": "../escape.txt"})
        self.assertFalse(result["ok"])
        self.assertIn("escapes the workspace", result["summary"])

    def test_search_uses_python_fallback_when_rg_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "beta.txt").write_text("needle here\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            with patch.object(shutil, "which", return_value=None):
                result = tools.search("needle")
        self.assertTrue(result["ok"])
        self.assertIn("beta.txt:1:needle here", result["output"])

    def test_run_shell_returns_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            command = f'{sys.executable} -c "print(123)"'
            result = tools.run_shell(command)
        self.assertTrue(result["ok"])
        self.assertEqual(result["output"], "123")

    def test_read_only_blocks_shell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="read-only")
            result = tools.run_shell("echo denied")
        self.assertFalse(result["ok"])
        self.assertIn("read-only", result["summary"])
