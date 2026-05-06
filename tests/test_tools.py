from __future__ import annotations

import os
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from ollama_code.tools import ToolExecutor


class ToolExecutorTests(unittest.TestCase):
    def _workspace_scratch(self) -> Path:
        root = (Path.cwd() / "verify_scratch" / f"test-tools-{uuid4().hex}").resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return root

    def _cross_platform_workspace_alias(self, root: Path) -> str:
        resolved = root.resolve()
        if resolved.drive:
            drive = resolved.drive[0].lower()
            tail = "/".join(part for part in resolved.parts[1:] if part not in {"\\", "/"})
            return f"/mnt/{drive}/{tail}" if tail else f"/mnt/{drive}"
        match = re.match(r"^/mnt/(?P<drive>[A-Za-z])(?:/(?P<rest>.*))?$", resolved.as_posix())
        if not match:
            self.skipTest("workspace is not on a WSL-style /mnt/<drive> path")
        drive = match.group("drive").upper()
        rest = (match.group("rest") or "").strip("/")
        return f"{drive}:/{rest}" if rest else f"{drive}:/"

    def init_git_repo(self, root: Path) -> None:
        if shutil.which("git") is None:
            self.skipTest("git is not installed")
        try:
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            message = (exc.stderr or exc.stdout or str(exc)).strip()
            self.skipTest(f"git repo init is unavailable in this environment: {message}")

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

    def test_mutating_tools_notify_indexer(self) -> None:
        class FakeIndexer:
            def __init__(self) -> None:
                self.paths: list[str] = []

            def notify_paths(self, paths: object) -> None:
                self.paths.extend(str(path) for path in paths)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            indexer = FakeIndexer()
            tools = ToolExecutor(root, approval_mode="auto", indexer=indexer)
            result = tools.execute("write_file", {"path": "note.txt", "content": "hello\n"})

        self.assertTrue(result["ok"])
        self.assertIn("note.txt", indexer.paths)

    def test_write_file_reports_python_syntax_error_without_blocking_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.write_file("bad.py", "def f():\nreturn 1\n")
            final_text = (root / "bad.py").read_text(encoding="utf-8")

        self.assertTrue(result["ok"])
        self.assertFalse(result["syntax_ok"])
        self.assertIn("Python syntax error", result["summary"])
        self.assertIn("bad.py:2", result["diagnostic"])
        self.assertEqual(final_text, "def f():\nreturn 1\n")

    def test_write_file_auto_dedents_globally_indented_python(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.write_file("fixed.py", "    def f():\n        return 1\n")
            final_text = (root / "fixed.py").read_text(encoding="utf-8")

        self.assertTrue(result["ok"])
        self.assertNotIn("syntax_ok", result)
        self.assertIn("Auto-dedented", result["summary"])
        self.assertEqual(final_text, "def f():\n    return 1\n")

    def test_write_file_strips_markdown_quote_prefixes_for_python(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.write_file("quoted.py", "> def f():\n>     return 1\n")
            final_text = (root / "quoted.py").read_text(encoding="utf-8")

        self.assertTrue(result["ok"])
        self.assertIn("quote prefixes", result["summary"])
        self.assertEqual(final_text, "def f():\n    return 1\n")

    def test_write_file_strips_single_leading_markdown_quote_for_python(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.write_file("quoted.py", "> import re\n\ndef f():\n    return re.escape('x')\n")
            final_text = (root / "quoted.py").read_text(encoding="utf-8")

        self.assertTrue(result["ok"])
        self.assertIn("quote prefixes", result["summary"])
        self.assertEqual(final_text, "import re\n\ndef f():\n    return re.escape('x')\n")

    def test_write_file_repairs_common_join_string_typo_when_parseable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.write_file("joiner.py", "def f(items):\n    return '.join(items)\n")
            final_text = (root / "joiner.py").read_text(encoding="utf-8")

        self.assertTrue(result["ok"])
        self.assertIn("join string typo", result["summary"])
        self.assertEqual(final_text, 'def f(items):\n    return " ".join(items)\n')

    def test_write_file_rejects_partial_python_overwrite_that_drops_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "list_ops.py"
            target.write_text(
                "def append(a, b):\n    return a + b\n\n\ndef reverse(items):\n    return list(reversed(items))\n",
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.write_file("list_ops.py", "def append(a, b):\n    return a + b\n")

        self.assertFalse(result["ok"])
        self.assertIn("partial content", result["summary"])
        self.assertIn("reverse", result["summary"])

    def test_write_file_rejects_generated_cache_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.write_file("__pycache__/sample.pyc", "")

        self.assertFalse(result["ok"])
        self.assertIn("generated/cache", result["summary"])

    def test_write_file_rejects_omitted_context_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.write_file("sample.py", "[omitted 500 chars from prior content; do not copy]")

        self.assertFalse(result["ok"])
        self.assertIn("omitted-context marker", result["summary"])

    def test_replace_in_file_reports_python_syntax_error_without_blocking_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.py").write_text("def f():\n    return 1\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.replace_in_file("sample.py", "    return 1", "return 1")

        self.assertTrue(result["ok"])
        self.assertFalse(result["syntax_ok"])
        self.assertIn("Python syntax error", result["summary"])

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

    def test_replace_tolerates_leading_whitespace_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "sample.py"
            sample.write_text("def f():\n    return 'old'\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.replace_in_file("sample.py", "   return 'old'", "    return 'new'")
            final_text = sample.read_text(encoding="utf-8")
        self.assertTrue(result["ok"])
        self.assertEqual(final_text, "def f():\n    return 'new'\n")

    def test_replace_in_file_identifier_call_rename_ignores_embedded_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "test_pricing.py"
            sample.write_text(
                "def test_cart_total():\n"
                "    assert cart_total([1]) == 1\n"
                "    assert total([2]) == 2\n",
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.replace_in_file("test_pricing.py", "total(", "cart_total(")
            final_text = sample.read_text(encoding="utf-8")

        self.assertTrue(result["ok"], result)
        self.assertIn("def test_cart_total():", final_text)
        self.assertIn("assert cart_total([1]) == 1", final_text)
        self.assertIn("assert cart_total([2]) == 2", final_text)
        self.assertNotIn("cart_cart_total", final_text)

    def test_replace_strips_read_file_line_number_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "sample.py"
            sample.write_text("def f():\n    return 'old'\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.replace_in_file(
                "sample.py",
                "   1 | def f():\n   2 |     return 'old'",
                "   1 | def f():\n   2 |     return 'new'",
            )
            final_text = sample.read_text(encoding="utf-8")
        self.assertTrue(result["ok"])
        self.assertEqual(final_text, "def f():\n    return 'new'\n")

    def test_replace_regex_fallback_inserts_backslashes_literally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "sample.py"
            sample.write_text("def f(value):\n    return value.strip().lower()\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.replace_in_file(
                "sample.py",
                "   return value.strip().lower()",
                "    return re.sub(r'\\s+', '-', value.strip().lower())",
            )
            final_text = sample.read_text(encoding="utf-8")
        self.assertTrue(result["ok"])
        self.assertIn("r'\\s+'", final_text)

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

    def test_read_file_accepts_cross_platform_workspace_alias(self) -> None:
        root = Path.cwd().resolve()
        alias_root = self._cross_platform_workspace_alias(root)
        tools = ToolExecutor(root, approval_mode="auto")

        result = tools.read_file(f"{alias_root}/README.md", start=1, end=1)

        self.assertTrue(result["ok"])
        self.assertEqual(result["path"], "README.md")
        self.assertIn("Ollama", result["output"])

    @unittest.skipUnless(os.name != "nt", "POSIX only")
    def test_read_file_normalizes_backslash_relative_path_on_posix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "nested" / "file.txt"
            target.parent.mkdir(parents=True)
            target.write_text("hello\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")

            result = tools.read_file(r"nested\file.txt", start=1, end=1)

        self.assertTrue(result["ok"])
        self.assertEqual(result["path"], "nested/file.txt")
        self.assertIn("hello", result["output"])

    def test_read_file_accepts_numeric_string_line_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "sample.txt"
            target.write_text("one\ntwo\nthree\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")

            result = tools.execute("read_file", {"path": "sample.txt", "start": "2", "end": "2"})

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["start"], 2)
        self.assertEqual(result["end"], 2)
        self.assertIn("two", result["output"])
        self.assertNotIn("one", result["output"])

    def test_list_files_accepts_numeric_string_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nested").mkdir()
            (root / "nested" / "child.txt").write_text("x\n", encoding="utf-8")
            (root / "top.txt").write_text("x\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")

            result = tools.execute("list_files", {"max_depth": "0", "limit": "1"})

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["count"], 1)

    def test_search_uses_python_fallback_when_rg_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "beta.txt").write_text("needle here\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            with patch.object(shutil, "which", return_value=None):
                result = tools.search("needle")
        self.assertTrue(result["ok"])
        self.assertIn("beta.txt:1:needle here", result["output"])

    def test_search_limits_rg_output_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "many.txt").write_text("\n".join("needle" for _ in range(20)), encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.search("needle", limit=3)

        self.assertTrue(result["ok"])
        self.assertLessEqual(len(result["output"].splitlines()), 3)

    def test_search_falls_back_to_literal_when_rg_regex_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pricing.py").write_text("def total(prices):\n    return sum(prices)\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.search("total(", limit=5)

        self.assertTrue(result["ok"])
        self.assertIn("total(prices)", str(result["output"]))

    def test_code_outline_returns_python_symbols_without_bodies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "service.py").write_text(
                "import math\n\n"
                "class PricingService:\n"
                "    def calculate_total(self, subtotal, tax):\n"
                "        hidden_body_marker = subtotal + tax\n"
                "        return hidden_body_marker\n\n"
                "def helper(value):\n"
                "    return math.ceil(value)\n",
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.code_outline("src/service.py")

        self.assertTrue(result["ok"])
        self.assertIn("class PricingService", result["output"])
        self.assertIn("PricingService.calculate_total", result["output"])
        self.assertIn("function helper", result["output"])
        self.assertIn("imports: import math", result["output"])
        self.assertNotIn("hidden_body_marker", result["output"])

    def test_code_outline_defaults_to_workspace_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def meaning():\n    return 42\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.execute("code_outline", {})

        self.assertTrue(result["ok"])
        self.assertEqual(result["path"], ".")
        self.assertIn("meaning", result["output"])

    def test_search_symbols_and_read_symbol_target_large_python_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            filler = "\n\n".join(f"def filler_{index}():\n    return {index}" for index in range(120))
            target_body = (
                "def calculate_discount(cart, percentage):\n"
                "    subtotal = sum(item['price'] for item in cart)\n"
                "    discount = subtotal * percentage\n"
                "    return max(0, subtotal - discount)\n"
            )
            (root / "src" / "pricing.py").write_text(f"{filler}\n\n{target_body}\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")

            search = tools.search_symbols("calculate_discount", path="src")
            read = tools.read_symbol("src/pricing.py", "calculate_discount", include_context=0)

        self.assertTrue(search["ok"])
        self.assertIn("src/pricing.py", search["output"])
        self.assertIn("calculate_discount", search["output"])
        self.assertTrue(read["ok"])
        self.assertIn("subtotal = sum", read["output"])
        self.assertNotIn("filler_0", read["output"])
        self.assertLess(len(read["output"]), 400)

    def test_symbol_search_prunes_dependency_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "node_modules" / "pkg").mkdir(parents=True)
            (root / "src" / "app.py").write_text("def real_target():\n    return 1\n", encoding="utf-8")
            (root / "node_modules" / "pkg" / "bad.py").write_text("def real_target_dependency():\n    return 2\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.search_symbols("real_target")

        self.assertTrue(result["ok"])
        self.assertIn("src/app.py", result["output"])
        self.assertNotIn("node_modules", result["output"])

    def test_read_symbol_reports_ambiguous_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "models.py").write_text(
                "class Alpha:\n"
                "    def save(self):\n"
                "        return 'a'\n\n"
                "class Beta:\n"
                "    def save(self):\n"
                "        return 'b'\n",
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto")

            ambiguous = tools.read_symbol("models.py", "save")
            precise = tools.read_symbol("models.py", "Beta.save", include_context=0)

        self.assertFalse(ambiguous["ok"])
        self.assertIn("Ambiguous", ambiguous["summary"])
        self.assertIn("Alpha.save", ambiguous["matches"])
        self.assertTrue(precise["ok"])
        self.assertIn("return 'b'", precise["output"])

    def test_replace_symbol_replaces_python_function_without_old_text(self) -> None:
        root = self._workspace_scratch()
        sample = root / "sample.py"
        sample.write_text(
            "def keep():\n"
            "    return 'keep'\n\n"
            "def target(value):\n"
            "    return value + 1\n",
            encoding="utf-8",
        )
        tools = ToolExecutor(root, approval_mode="auto")
        result = tools.replace_symbol("sample.py", "target", "def target(value):\n    return value * 2")
        final_text = sample.read_text(encoding="utf-8")

        self.assertTrue(result["ok"])
        self.assertEqual(result["symbol"], "target")
        self.assertIn("return value * 2", final_text)
        self.assertIn("return 'keep'", final_text)

    def test_replace_symbol_accepts_signature_as_symbol_query(self) -> None:
        root = self._workspace_scratch()
        sample = root / "ops.py"
        sample.write_text("def map(f, items):\n    return []\n", encoding="utf-8")
        tools = ToolExecutor(root, approval_mode="auto")
        result = tools.replace_symbol("ops.py", "map(f, items):", "def map(f, items):\n    return [f(item) for item in items]\n")

        self.assertTrue(result["ok"], result)
        self.assertIn("return [f(item) for item in items]", sample.read_text(encoding="utf-8"))

    def test_replace_symbol_aligns_method_indent(self) -> None:
        root = self._workspace_scratch()
        sample = root / "sample.py"
        sample.write_text(
            "class Service:\n"
            "    def value(self):\n"
            "        return 1\n",
            encoding="utf-8",
        )
        tools = ToolExecutor(root, approval_mode="auto")
        result = tools.replace_symbol("sample.py", "Service.value", "def value(self):\n    return 2\n")
        final_text = sample.read_text(encoding="utf-8")

        self.assertTrue(result["ok"])
        self.assertIn("    def value(self):\n        return 2\n", final_text)

    def test_replace_symbol_rejects_python_syntax_error_without_writing(self) -> None:
        root = self._workspace_scratch()
        sample = root / "sample.py"
        original = "def target():\n    return 1\n"
        sample.write_text(original, encoding="utf-8")
        tools = ToolExecutor(root, approval_mode="auto")
        result = tools.replace_symbol("sample.py", "target", "def target(:\n    return 2\n")
        final_text = sample.read_text(encoding="utf-8")

        self.assertFalse(result["ok"])
        self.assertFalse(result["syntax_ok"])
        self.assertIn("Refusing replace_symbol", result["summary"])
        self.assertEqual(final_text, original)

    def test_replace_symbol_rejects_bare_expression_for_python_function(self) -> None:
        root = self._workspace_scratch()
        sample = root / "sample.py"
        original = "def target(value):\n    return value + 1\n"
        sample.write_text(original, encoding="utf-8")
        tools = ToolExecutor(root, approval_mode="auto")
        result = tools.replace_symbol("sample.py", "target(value)", "renamed_target(value)")
        final_text = sample.read_text(encoding="utf-8")

        self.assertFalse(result["ok"])
        self.assertIn("full def source", result["summary"])
        self.assertEqual(final_text, original)

    def test_replace_symbol_reports_ambiguous_names(self) -> None:
        root = self._workspace_scratch()
        (root / "models.py").write_text(
            "class Alpha:\n"
            "    def save(self):\n"
            "        return 'a'\n\n"
            "class Beta:\n"
            "    def save(self):\n"
            "        return 'b'\n",
            encoding="utf-8",
        )
        tools = ToolExecutor(root, approval_mode="auto")
        result = tools.replace_symbol("models.py", "save", "def save(self):\n    return 'x'\n")

        self.assertFalse(result["ok"])
        self.assertIn("Ambiguous", result["summary"])
        self.assertIn("Alpha.save", result["matches"])

    def test_replace_symbols_replaces_multiple_python_functions_atomically(self) -> None:
        root = self._workspace_scratch()
        sample = root / "ops.py"
        original = (
            "def add(a, b):\n"
            "    return None\n\n"
            "def multiply(a, b):\n"
            "    return None\n"
        )
        sample.write_text(original, encoding="utf-8")
        tools = ToolExecutor(root, approval_mode="auto")
        result = tools.replace_symbols(
            "ops.py",
            [
                {"symbol": "add", "content": "def add(a, b):\n    return a + b\n"},
                {"symbol": "multiply", "content": "def multiply(a, b):\n    return a * b\n"},
            ],
        )
        final_text = sample.read_text(encoding="utf-8")

        self.assertTrue(result["ok"])
        self.assertEqual(result["count"], 2)
        self.assertIn("return a + b", final_text)
        self.assertIn("return a * b", final_text)

    def test_replace_symbols_rejects_python_syntax_error_without_writing(self) -> None:
        root = self._workspace_scratch()
        sample = root / "ops.py"
        original = "def add(a, b):\n    return None\n"
        sample.write_text(original, encoding="utf-8")
        tools = ToolExecutor(root, approval_mode="auto")
        result = tools.replace_symbols(
            "ops.py",
            [{"symbol": "add", "content": "def add(a, b):\nreturn a + b\n"}],
        )
        final_text = sample.read_text(encoding="utf-8")

        self.assertFalse(result["ok"])
        self.assertFalse(result["syntax_ok"])
        self.assertEqual(final_text, original)

    def test_replace_symbols_accepts_python_file_with_utf8_bom(self) -> None:
        root = self._workspace_scratch()
        sample = root / "ops.py"
        sample.write_text("\ufeffdef add(a, b):\n    return None\n", encoding="utf-8")
        tools = ToolExecutor(root, approval_mode="auto")
        result = tools.replace_symbols(
            "ops.py",
            [{"symbol": "add", "content": "def add(a, b):\n    return a + b\n"}],
        )
        final_text = sample.read_text(encoding="utf-8")

        self.assertTrue(result["ok"])
        self.assertTrue(final_text.startswith("\ufeffdef add"))
        self.assertIn("return a + b", final_text)

    def test_execute_ignores_extra_tool_arguments_when_required_args_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pricing.py").write_text(
                "def calculate_discount():\n"
                "    return 'TOKEN_SYMBOL_750'\n",
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.execute(
                "read_symbol",
                {"path": "pricing.py", "symbol": "calculate_discount", "start": 1, "end": 20},
            )

        self.assertTrue(result["ok"])
        self.assertIn("TOKEN_SYMBOL_750", result["output"])

    def test_code_outline_supports_generic_javascript_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "web.js").write_text(
                "export class Widget {\n"
                "  render() { return '<div></div>'; }\n"
                "}\n\n"
                "export function hydrateWidget() {\n"
                "  return new Widget();\n"
                "}\n",
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto")
            outline = tools.code_outline("web.js")
            symbol = tools.read_symbol("web.js", "hydrateWidget", include_context=0)

        self.assertTrue(outline["ok"])
        self.assertIn("class Widget", outline["output"])
        self.assertIn("function hydrateWidget", outline["output"])
        self.assertTrue(symbol["ok"])
        self.assertIn("new Widget", symbol["output"])

    def test_repo_index_search_returns_ranked_compact_snippets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pricing.py").write_text(
                "def calculate_discount(cart):\n"
                "    return sum(cart) * 0.9\n",
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.repo_index_search("discount calculation")

        self.assertTrue(result["ok"])
        self.assertIn("pricing.py", result["output"])
        self.assertIn("calculate_discount", result["output"])
        self.assertNotIn("   1 |", result["output"])

    def test_fts_search_indexes_symbols_headings_and_scopes_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "docs").mkdir()
            (root / "src" / "pricing.py").write_text(
                "def calculate_discount(cart):\n"
                "    \"\"\"Seasonal discount calculation.\"\"\"\n"
                "    return sum(cart) * 0.9\n",
                encoding="utf-8",
            )
            (root / "docs" / "guide.md").write_text("# Seasonal Pricing\nUse discount calculation carefully.\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            refresh = tools.fts_refresh()
            all_result = tools.fts_search("seasonal discount")
            src_result = tools.fts_search("seasonal", path="src")

        self.assertTrue(refresh["ok"], refresh)
        self.assertTrue(all_result["ok"], all_result)
        self.assertIn("src/pricing.py", all_result["output"])
        self.assertIn("docs/guide.md", all_result["output"])
        self.assertIn("calculate_discount", all_result["output"])
        self.assertIn("src/pricing.py", src_result["output"])
        self.assertNotIn("docs/guide.md", src_result["output"])

    def test_fd_search_reports_missing_fd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            with patch.object(ToolExecutor, "_fd_cli_path", return_value=None):
                result = tools.fd_search("README")

        self.assertFalse(result["ok"])
        self.assertEqual(result["missing_dependency"], "fd")

    def test_context_pack_returns_ranked_evidence_and_writes_index_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "tests").mkdir()
            (root / "src" / "pricing.py").write_text("def calculate_discount(cart):\n    return sum(cart)\n", encoding="utf-8")
            (root / "tests" / "test_pricing.py").write_text("from src.pricing import calculate_discount\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.context_pack("fix discount calculation")

        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "context_pack")
        self.assertIn("suggested_next_tool", result["output"])
        self.assertIn("pricing.py", result["output"])
        self.assertIn("test_pricing.py", result["output"])

    def test_repo_index_search_invalidates_changed_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "ops.py"
            target.write_text("def first_name():\n    return 'old'\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            first = tools.repo_index_search("first_name")
            target.write_text("def second_name():\n    return 'new'\n", encoding="utf-8")
            second = tools.repo_index_search("second_name")

        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertIn("first_name", first["output"])
        self.assertIn("second_name", second["output"])

    def test_indexed_search_uses_cached_lines_and_invalidates_changed_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "ops.py"
            target.write_text("def first_name():\n    return 'alpha needle'\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            first = tools.indexed_search("alpha needle")
            target.write_text("def second_name():\n    return 'beta needle'\n", encoding="utf-8")
            second = tools.indexed_search("beta needle")

        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertIn("ops.py:2", first["output"])
        self.assertIn("alpha needle", first["output"])
        self.assertIn("beta needle", second["output"])

    def test_file_search_uses_cached_paths_and_invalidates_changed_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            first_path = root / "src" / "alpha_report.txt"
            second_path = root / "src" / "beta_report.txt"
            first_path.write_text("one", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            first = tools.file_search("alpha report")
            first_path.unlink()
            second_path.write_text("two", encoding="utf-8")
            second = tools.file_search("beta report")

        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertIn("src/alpha_report.txt", first["output"])
        self.assertIn("src/beta_report.txt", second["output"])
        self.assertNotIn("alpha_report", second["output"])

    def test_file_index_refresh_writes_file_path_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("hello", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.file_index_refresh()
            payload = json.loads((root / ".ollama-code" / "index" / "file_index.json").read_text(encoding="utf-8"))

        self.assertTrue(result["ok"])
        self.assertEqual(result["files"], 1)
        self.assertIn("README.md", payload["files"])

    def test_file_search_respects_path_scope_and_skips_generated_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "docs").mkdir()
            (root / "node_modules").mkdir()
            (root / ".ollama-code" / "index").mkdir(parents=True)
            (root / "src" / "target_config.py").write_text("x = 1\n", encoding="utf-8")
            (root / "docs" / "target_config.md").write_text("x\n", encoding="utf-8")
            (root / "node_modules" / "target_config.js").write_text("x\n", encoding="utf-8")
            (root / ".ollama-code" / "index" / "target_config.json").write_text("{}", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            src_result = tools.file_search("target config", path="src")
            root_result = tools.file_search("target config")

        self.assertTrue(src_result["ok"])
        self.assertEqual(src_result["output"], "src/target_config.py")
        self.assertIn("src/target_config.py", root_result["output"])
        self.assertIn("docs/target_config.md", root_result["output"])
        self.assertNotIn("node_modules", root_result["output"])
        self.assertNotIn(".ollama-code", root_result["output"])

    def test_broad_repo_tools_skip_generated_and_ignored_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "scratch").mkdir()
            (root / "verify_scratch").mkdir()
            (root / "ollama-code-bench-deadbeef").mkdir()
            (root / "probe-123").mkdir()
            (root / "tmpabc123").mkdir()
            (root / "src" / "app.py").write_text("def live_target():\n    return 'ok'\n", encoding="utf-8")
            for directory in ["scratch", "verify_scratch", "ollama-code-bench-deadbeef", "probe-123", "tmpabc123"]:
                (root / directory / "ignored_target.py").write_text("def ignored_target():\n    return 'skip'\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            real_which = shutil.which

            with patch("ollama_code.tools.shutil.which", side_effect=lambda name: None if name == "rg" else real_which(name)):
                search_result = tools.search("ignored_target")
            file_result = tools.file_search("ignored target")
            repo_result = tools.repo_index_search("ignored_target")
            refresh_result = tools.fts_refresh()
            fts_result = tools.fts_search("ignored_target")
            context_result = tools.context_pack("find ignored_target")
            validators = tools.discover_validators()
            live_result = tools.repo_index_search("live_target")

        for result in [search_result, file_result, repo_result, refresh_result, fts_result, context_result, validators]:
            self.assertTrue(result["ok"], result)
        self.assertNotIn("ignored_target", search_result["output"])
        self.assertNotIn("ignored_target", file_result["output"])
        self.assertNotIn("ignored_target", repo_result["output"])
        self.assertNotIn("ignored_target", fts_result["output"])
        self.assertNotIn("ignored_target.py", context_result["output"])
        self.assertNotIn("unittest discover", validators["output"])
        self.assertIn("live_target", live_result["output"])

    def test_file_search_prefers_exact_filename_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "alpha").mkdir()
            (root / "alpha" / "notes.py").write_text("x = 1\n", encoding="utf-8")
            (root / "alpha_notes.py").write_text("x = 2\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.file_search("alpha_notes.py", limit=2)

        self.assertTrue(result["ok"])
        self.assertEqual(result["output"].splitlines()[0], "alpha_notes.py")

    def test_everything_search_reports_missing_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = ToolExecutor(root, approval_mode="auto")
            with patch.object(tools, "_everything_cli_path", return_value=None):
                result = tools.everything_search("README")

        self.assertFalse(result["ok"])
        self.assertIn("es.exe", result["summary"])

    def test_everything_search_runs_cli_and_filters_to_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inside = root / "README.md"
            outside = Path(tmp).parent / f"outside-{uuid4().hex}.txt"
            inside.write_text("hello", encoding="utf-8")
            outside.write_text("skip", encoding="utf-8")
            self.addCleanup(lambda: outside.unlink(missing_ok=True))
            tools = ToolExecutor(root, approval_mode="auto")
            completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=f"{inside}\n{outside}\n", stderr="")
            with patch.object(tools, "_everything_cli_path", return_value="es.exe"):
                with patch.object(tools, "_run_process", return_value=completed) as run_process:
                    result = tools.everything_search("README", limit=10)

        self.assertTrue(result["ok"])
        self.assertIn("README.md", result["output"])
        self.assertNotIn("outside", result["output"])
        command = run_process.call_args.args[0]
        self.assertEqual(command[:3], ["es.exe", "-n", "50"])
        self.assertFalse(run_process.call_args.kwargs["shell"])

    def test_everything_search_uses_configured_cli_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_cli = root / "tools" / "es.exe"
            fake_cli.parent.mkdir()
            fake_cli.write_text("", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            with patch.dict("os.environ", {"EVERYTHING_CLI": str(fake_cli)}):
                with patch("ollama_code.tools.shutil.which", return_value=None):
                    discovered = tools._everything_cli_path()

        self.assertEqual(discovered, str(fake_cli))

    def test_everything_search_filters_to_requested_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "docs").mkdir()
            inside = root / "src" / "target.py"
            outside = root / "docs" / "target.py"
            inside.write_text("x\n", encoding="utf-8")
            outside.write_text("x\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=f"{inside}\n{outside}\n", stderr="")
            with patch.object(tools, "_everything_cli_path", return_value="es.exe"):
                with patch.object(tools, "_run_process", return_value=completed):
                    result = tools.everything_search("target.py", path="src", limit=10)

        self.assertTrue(result["ok"])
        self.assertEqual(result["output"], "src/target.py")

    def test_repo_index_refresh_writes_line_index_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ops.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.repo_index_refresh()
            payload = json.loads((root / ".ollama-code" / "index" / "repo_index.json").read_text(encoding="utf-8"))

        self.assertTrue(result["ok"])
        self.assertEqual(result["files"], 1)
        self.assertIn("line_index", payload["files"]["ops.py"])

    def test_semgrep_scan_reports_missing_semgrep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ops.py").write_text("eval('1')\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            with patch("ollama_code.tools.shutil.which", return_value=None):
                result = tools.semgrep_scan("eval(...)")

        self.assertFalse(result["ok"])
        self.assertIn("semgrep is not installed", result["summary"])

    def test_semgrep_scan_rejects_unsupported_language_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ops.py").write_text("eval('1')\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            with patch("ollama_code.tools.shutil.which", return_value="semgrep"):
                with patch.object(tools, "_run_process") as run_process:
                    result = tools.semgrep_scan("eval(...)", lang="madeuplang")

        self.assertFalse(result["ok"])
        self.assertIn("Unsupported semgrep language", result["summary"])
        run_process.assert_not_called()

    def test_semgrep_scan_runs_structural_search_and_compacts_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ops.py").write_text("eval('1')\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            payload = {
                "results": [
                    {
                        "path": str(root / "ops.py"),
                        "start": {"line": 1},
                        "extra": {"lines": "eval('1')"},
                    }
                ]
            }
            completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(payload), stderr="")
            with patch("ollama_code.tools.shutil.which", return_value="semgrep"):
                with patch.object(tools, "_run_process", return_value=completed) as run_process:
                    result = tools.semgrep_scan("eval(...)", lang="python")

        self.assertTrue(result["ok"])
        self.assertIn("ops.py:1", result["output"])
        command = run_process.call_args.args[0]
        self.assertIn("--json", command)
        self.assertIn("--lang", command)
        self.assertFalse(run_process.call_args.kwargs["shell"])

    def test_semgrep_scan_reports_cli_error_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ops.py").write_text("eval('1')\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            completed = subprocess.CompletedProcess(args=[], returncode=2, stdout="", stderr="semgrep parse error")
            with patch("ollama_code.tools.shutil.which", return_value="semgrep"):
                with patch.object(tools, "_run_process", return_value=completed):
                    result = tools.semgrep_scan("eval(...)", lang="python")

        self.assertFalse(result["ok"])
        self.assertIn("semgrep parse error", result["output"])

    def test_new_default_tools_are_registered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")

            for name in {
                "todo_read",
                "todo_write",
                "edit_intent",
                "fd_search",
                "fts_search",
                "fts_refresh",
                "ast_search",
                "lsp_diagnostics",
                "lsp_definition",
                "lsp_references",
                "inspect_library_source",
                "systems_lens",
                "discover_validators",
                "diagnose_dependency_error",
                "browser_smoke",
                "security_scan",
                "mcp_list_tools",
                "mcp_call",
            }:
                self.assertIn(name, tools.available_tool_names())

    def test_todo_tools_store_validate_and_render_session_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="read-only")

            written = tools.execute(
                "todo_write",
                {
                    "items": [
                        {"content": "Inspect failing test", "status": "completed", "id": "inspect"},
                        {"content": "Patch implementation", "status": "in_progress", "id": "patch"},
                        {"content": "Run focused tests", "status": "pending", "id": "test"},
                    ]
                },
            )
            read = tools.execute("todo_read", {})

        self.assertTrue(written["ok"], written)
        self.assertEqual(written["pending"], 1)
        self.assertEqual(written["in_progress"], 1)
        self.assertIn("patch. [in_progress] Patch implementation", read["output"])
        self.assertEqual(read["items"][0]["id"], "inspect")

    def test_todo_write_rejects_multiple_in_progress_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")

            result = tools.execute(
                "todo_write",
                {"items": [{"content": "one", "status": "in_progress"}, {"content": "two", "status": "doing"}]},
            )

        self.assertFalse(result["ok"])
        self.assertIn("one todo item", result["summary"].lower())

    def test_disabled_tool_is_hidden_and_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto", disabled_tools=["browser_smoke"])

            self.assertNotIn("browser_smoke", tools.available_tool_names())
            result = tools.execute("browser_smoke", {"url": "http://127.0.0.1"})

        self.assertFalse(result["ok"])
        self.assertIn("disabled", result["summary"])

    def test_systems_lens_frames_complex_task_without_grounding_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            tools = ToolExecutor(root, approval_mode="auto")

            result = tools.execute(
                "systems_lens",
                {
                    "request": "Debug a slow flaky integration pipeline and decide what to refactor",
                    "path": ".",
                    "evidence": "run_test timed out",
                    "limit": 12,
                },
            )

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["tool"], "systems_lens")
        output = result["output"]
        self.assertIn("boundary=", output)
        self.assertIn("state=", output)
        self.assertIn("scale_time=", output)
        self.assertIn("viewpoints=", output)
        self.assertIn("coupling=", output)
        self.assertIn("next_tools=", output)
        self.assertIn("questions:", output)
        self.assertIn("observer:", output)
        self.assertIn("categories:", output)
        self.assertIn("feedback:", output)
        self.assertIn("stocks_flows:", output)
        self.assertIn("delay:", output)
        self.assertIn("intervention:", output)
        self.assertIn("debug:", output)
        self.assertIn("performance:", output)
        self.assertIn("change:", output)
        self.assertIn("current_evidence=run_test timed out", output)
        self.assertIn("What exactly is inside the system", result["questions"][0])

    def test_inspect_library_source_reads_python_source_and_builtin_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            source = tools.execute("inspect_library_source", {"target": "json.loads", "max_lines": 40})
            builtin = tools.inspect_library_source("builtins.len")

        self.assertTrue(source["ok"], source)
        self.assertTrue(source["source_available"])
        self.assertIn("def loads", source["output"])
        self.assertIn("signature:", source["output"])
        self.assertTrue(builtin["ok"], builtin)
        self.assertFalse(builtin["source_available"])
        self.assertIn("source: unavailable", builtin["output"])
        self.assertIn("doc:", builtin["output"])

    def test_inspect_library_source_reports_missing_module(self) -> None:
        missing_name = f"missing_package_{uuid4().hex}"
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            result = tools.inspect_library_source(f"{missing_name}.helper")

        self.assertFalse(result["ok"])
        self.assertEqual(result["missing_dependency"], missing_name)
        self.assertEqual(result["error_class"], "missing_dependency")

    def test_ast_search_reports_missing_ast_grep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ops.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            with patch("ollama_code.tools.shutil.which", return_value=None):
                result = tools.ast_search("def $F($$$A): $$$B", lang="python")

        self.assertFalse(result["ok"])
        self.assertEqual(result["missing_dependency"], "ast-grep")

    def test_lsp_navigation_reports_missing_language_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ops.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            with patch("ollama_code.tools.shutil.which", return_value=None):
                definition = tools.lsp_definition("ops.py", 1, 5)
                references = tools.lsp_references("ops.py", 1, 5)

        self.assertFalse(definition["ok"])
        self.assertEqual(definition["missing_dependency"], "pyright")
        self.assertFalse(references["ok"])
        self.assertEqual(references["missing_dependency"], "pyright")

    def test_edit_intent_routes_bad_symbol_target_to_text_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "pricing.py"
            sample.write_text("def total(prices):\n    return sum(prices)\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.edit_intent("pricing.py", "replace_symbol", "return sum(prices)", "return 0")

            final_text = sample.read_text(encoding="utf-8")

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["routed_tool"], "replace_in_file")
        self.assertIn("return 0", final_text)

    def test_edit_intent_routes_freeform_function_fix_to_body_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "calculator.py"
            sample.write_text("def add(left: int, right: int) -> int:\n    return left - right\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.edit_intent(
                "calculator.py",
                "fix the implementation of the add function",
                "add",
                "return left + right",
            )

            final_text = sample.read_text(encoding="utf-8")

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["routed_tool"], "apply_structured_edit")
        self.assertIn("def add(left: int, right: int) -> int:\n    return left + right", final_text)

    def test_edit_intent_routes_fix_like_full_function_to_symbol_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "calculator.py"
            sample.write_text("def add(left: int, right: int) -> int:\n    return left - right\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.edit_intent(
                "calculator.py",
                "fix_function_body",
                "add",
                "def add(left: int, right: int) -> int:\n    return left + right\n",
            )

            final_text = sample.read_text(encoding="utf-8")

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["routed_tool"], "replace_symbol")
        self.assertIn("return left + right", final_text)
        self.assertNotIn("def def", final_text)

    def test_edit_intent_unknown_intent_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "calculator.py"
            original = "def add(left: int, right: int) -> int:\n    return left - right\n"
            sample.write_text(original, encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.edit_intent(
                "calculator.py",
                "remove_extra_def_keyword",
                "def def add(left, right):",
                "def add(left, right):",
            )

            final_text = sample.read_text(encoding="utf-8")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_class"], "invalid_args")
        self.assertEqual(final_text, original)

    def test_edit_intent_routes_project_rename_to_structured_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "tests").mkdir()
            (root / "src" / "pricing.py").write_text("def total(prices):\n    return sum(prices)\n", encoding="utf-8")
            (root / "tests" / "test_pricing.py").write_text("from src.pricing import total\n\nassert total([1]) == 1\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.edit_intent(".", "rename", "total", "cart_total", scope="project")

            source = (root / "src" / "pricing.py").read_text(encoding="utf-8")
            test = (root / "tests" / "test_pricing.py").read_text(encoding="utf-8")

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["routed_tool"], "apply_structured_edit")
        self.assertIn("def cart_total", source)
        self.assertIn("cart_total", test)

    def test_discover_validators_detects_polyglot_projects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "package.json").write_text(json.dumps({"scripts": {"test": "vitest", "lint": "eslint .", "typecheck": "tsc --noEmit"}}), encoding="utf-8")
            (root / "go.mod").write_text("module example.com/app\n", encoding="utf-8")
            (root / "Cargo.toml").write_text("[package]\nname='demo'\nversion='0.1.0'\nedition='2021'\n", encoding="utf-8")
            (root / "build.gradle").write_text("plugins { id 'java' }\n", encoding="utf-8")
            (root / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.discover_validators()

        output = result["output"]
        self.assertTrue(result["ok"])
        self.assertIn("npm test", output)
        self.assertIn("go test ./...", output)
        self.assertIn("cargo test", output)
        self.assertIn("gradle test", output)
        self.assertIn("cmake -S . -B build", output)

    def test_discover_validators_checks_local_wrappers_in_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            project = workspace / "service"
            project.mkdir()
            (project / "build.gradle").write_text("plugins { id 'java' }\n", encoding="utf-8")
            (project / "gradlew").write_text("#!/bin/sh\n", encoding="utf-8")
            (project / "gradlew.bat").write_text("@echo off\n", encoding="utf-8")
            tools = ToolExecutor(workspace, approval_mode="auto")

            result = tools.discover_validators("service", limit=20)

        wrapper_command = "gradlew.bat test" if os.name == "nt" else "./gradlew test"
        self.assertTrue(result["ok"], result)
        self.assertIn(f"{wrapper_command} available=True", result["output"])

    def test_discover_validators_detects_python_tooling_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "tests" / "test_app.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
            (root / "pyproject.toml").write_text(
                "[tool.pytest.ini_options]\naddopts='-q'\n\n[tool.ruff]\nline-length=100\n\n[tool.mypy]\npython_version='3.12'\n\n[tool.pyright]\ntypeCheckingMode='basic'\n\n[tool.tox]\nlegacy_tox_ini='[tox]\\nenvlist=py312'\n",
                encoding="utf-8",
            )
            (root / "noxfile.py").write_text("import nox\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.discover_validators(limit=20)

        output = result["output"]
        self.assertTrue(result["ok"])
        self.assertIn("pytest --collect-only", output)
        self.assertIn("pytest", output)
        self.assertIn("ruff check", output)
        self.assertIn("mypy .", output)
        self.assertIn("pyright", output)
        self.assertIn("tox", output)
        self.assertIn("nox", output)

    def test_select_tests_returns_language_level_validator_for_go(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "go.mod").write_text("module example.com/app\n", encoding="utf-8")
            (root / "calc.go").write_text("package app\nfunc Add(a int, b int) int { return a + b }\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.select_tests(["calc.go"])

        self.assertTrue(result["ok"])
        self.assertIn("go test ./...", result["test_commands"])

    def test_diagnose_dependency_error_reports_missing_module_and_manager(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.diagnose_dependency_error("ModuleNotFoundError: No module named 'requests'")

        self.assertTrue(result["ok"])
        self.assertEqual(result["missing_dependency"], "requests")
        self.assertIn("pip", result["package_managers"])

    def test_browser_and_security_missing_dependencies_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            with patch("ollama_code.tools.subprocess.run", return_value=subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="missing")):
                browser = tools.browser_smoke("http://127.0.0.1")
            with patch("ollama_code.tools.shutil.which", return_value=None):
                security = tools.security_scan()

        self.assertFalse(browser["ok"])
        self.assertEqual(browser["missing_dependency"], "playwright")
        self.assertFalse(security["ok"])
        self.assertIn("missing_dependency", security)

    def test_find_implementation_target_maps_test_imports_to_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "ops.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            (root / "tests" / "test_ops.py").write_text("from ops import add\n\ndef test_add():\n    assert add(1, 2) == 3\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.find_implementation_target(test_path="tests/test_ops.py")

        self.assertTrue(result["ok"])
        self.assertIn("ops.py", result["output"])
        self.assertIn("symbol=add", result["output"])

    def test_find_implementation_target_accepts_path_alias_for_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "ops.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.find_implementation_target(path="src/ops.py", query="add")

        self.assertTrue(result["ok"])
        self.assertIn("src/ops.py", result["output"])
        self.assertIn("symbol=add", result["output"])

    def test_diagnose_test_failure_groups_assertions_and_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "ops.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
            (root / "tests" / "test_ops.py").write_text("from ops import add\n", encoding="utf-8")
            output = (
                "FAILED tests/test_ops.py::test_add - AssertionError\n"
                "E       assert 1 == 3\n"
                '  File "ops.py", line 2, in add\n'
            )
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.diagnose_test_failure(output)

        self.assertTrue(result["ok"])
        self.assertIn("assertion mismatch", result["output"])
        self.assertIn("ops.py", result["output"])

    def test_diagnose_test_failure_maps_unittest_traceback_imports_to_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_file = root / "list_ops_test.py"
            (root / "list_ops.py").write_text("def foldr(function, items, initial):\n    return initial\n", encoding="utf-8")
            test_file.write_text(
                "from list_ops import foldr\n\n"
                "def test_foldr():\n"
                "    assert foldr(lambda acc, el: el + acc, ['e'], '!') == 'e!'\n",
                encoding="utf-8",
            )
            output = (
                "FAIL: test_foldr (list_ops_test.ListOpsTest.test_foldr)\n"
                "Traceback (most recent call last):\n"
                f'  File "{test_file}", line 4, in test_foldr\n'
                "AssertionError: '!e' != 'e!'\n"
            )
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.diagnose_test_failure(output)

        self.assertTrue(result["ok"])
        self.assertIn("actual='!e' expected='e!'", result["output"])
        self.assertIn("list_ops.py", result["output"])
        self.assertIn("symbol=foldr", result["output"])

    def test_run_function_probe_reports_actual_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ops.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.run_function_probe("ops", ["fn(2, 5)"], function="add")

        self.assertTrue(result["ok"])
        self.assertIn("fn(2, 5): 7", result["output"])

    def test_call_graph_finds_callers_and_callees(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text(
                "def helper(value):\n"
                "    return value + 1\n\n"
                "def target(value):\n"
                "    return helper(value)\n\n"
                "def caller():\n"
                "    return target(1)\n",
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.call_graph("app.py", "target")

        self.assertTrue(result["ok"])
        self.assertIn("callees: helper", result["output"])
        self.assertIn("caller", result["output"])

    def test_contract_graph_reports_contracts_and_purity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text(
                "import subprocess\n\n"
                "def helper(value: int) -> int:\n"
                "    return value + 1\n\n"
                "def target(value: int) -> int:\n"
                "    return helper(value)\n\n"
                "def caller() -> int:\n"
                "    return target(1)\n\n"
                "def shell() -> None:\n"
                "    subprocess.run(['echo', 'x'])\n",
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.contract_graph("app.py", "target")
            all_result = tools.contract_graph("app.py")

        self.assertTrue(result["ok"])
        self.assertIn("target(value: int)->int", result["output"])
        self.assertIn("callees: helper", result["output"])
        self.assertIn("caller", result["output"])
        self.assertIn("pure_hint", result["output"])
        self.assertIn("shell", all_result["output"])
        self.assertIn("impure_hint", all_result["output"])

    def test_contract_check_passes_compatible_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text(
                "def x() -> int:\n"
                "    return 1\n\n"
                "def y(value: int) -> int:\n"
                "    return value + 1\n\n"
                "def z() -> int:\n"
                "    return y(x())\n",
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.contract_check(["app.py"])

        self.assertTrue(result["ok"], result["output"])
        self.assertIn("contract ok", result["output"])

    def test_contract_check_flags_arity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text(
                "def total(items: list[int], tax: int) -> int:\n"
                "    return sum(items) + tax\n\n"
                "def checkout() -> int:\n"
                "    return total([1, 2])\n",
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.contract_check(["app.py"], changed_symbols=["total"])

        self.assertFalse(result["ok"])
        self.assertIn("calls total with 1 positional args", result["output"])

    def test_contract_check_flags_return_annotation_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text(
                "def values() -> list[int]:\n"
                "    return {'a': 1}\n\n"
                "def missing() -> int:\n"
                "    pass\n",
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.contract_check(["app.py"])

        self.assertFalse(result["ok"])
        self.assertIn("returns shape dict", result["output"])
        self.assertIn("may return None", result["output"])

    def test_contract_check_flags_caller_return_shape_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text(
                "def y() -> dict[str, int]:\n"
                "    return {'value': 1}\n\n"
                "def z() -> int:\n"
                "    return y()[0]\n",
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.contract_check(["app.py"], changed_symbols=["y"])

        self.assertFalse(result["ok"])
        self.assertIn("expects y return as sequence", result["output"])

    def test_contract_check_allows_matching_caller_return_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text(
                "def y() -> dict[str, int]:\n"
                "    return {'value': 1}\n\n"
                "def z() -> int:\n"
                "    return y()['value']\n",
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.contract_check(["app.py"], changed_symbols=["y"])

        self.assertTrue(result["ok"], result["output"])

    def test_contract_check_flags_builtin_method_receiver_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text(
                "def names(items: list[str]) -> list[str]:\n"
                "    return list(items.items())\n\n"
                "def update(payload: dict[str, int]) -> None:\n"
                "    payload.append(1)\n",
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.contract_check(["app.py"])

        self.assertFalse(result["ok"])
        self.assertIn("calls .items() on list; expected dict receiver", result["output"])
        self.assertIn("calls .append() on dict; expected list receiver", result["output"])

    def test_contract_check_allows_matching_builtin_method_receivers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text(
                "def names(items: dict[str, int], labels: list[str]) -> list[str]:\n"
                "    labels.append('x')\n"
                "    return list(items.keys())\n\n"
                "def normalize(value: str) -> str:\n"
                "    return value.strip().lower()\n",
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.contract_check(["app.py"])

        self.assertTrue(result["ok"], result["output"])

    def test_lint_typecheck_reports_python_syntax_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bad.py").write_text("def broken(:\n    pass\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.lint_typecheck("bad.py")

        self.assertFalse(result["ok"])
        self.assertIn("bad.py:1", result["output"])

    def test_select_tests_maps_python_source_to_importing_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "tests").mkdir()
            (root / "src" / "pricing.py").write_text("def cart_total(prices):\n    return sum(prices)\n", encoding="utf-8")
            (root / "tests" / "test_pricing.py").write_text(
                "from src.pricing import cart_total\n\n"
                "def test_cart_total():\n"
                "    assert cart_total([1, 2]) == 3\n",
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.select_tests(["src/pricing.py"], changed_symbols=["cart_total"])

        self.assertTrue(result["ok"])
        self.assertEqual(result["confidence"], "high")
        self.assertIn("unittest discover", result["test_commands"][0])
        self.assertIn("test_pricing.py", result["test_commands"][0])

    def test_apply_structured_edit_renames_symbol_mechanically(self) -> None:
        root = self._workspace_scratch()
        sample = root / "ops.py"
        sample.write_text("def old_name(value):\n    return value\n\nRESULT = old_name(1)\n", encoding="utf-8")
        tools = ToolExecutor(root, approval_mode="auto")
        result = tools.apply_structured_edit({"op": "rename_symbol", "path": "ops.py", "old": "old_name", "new": "new_name"})
        final_text = sample.read_text(encoding="utf-8")

        self.assertTrue(result["ok"])
        self.assertIn("def new_name", final_text)
        self.assertIn("RESULT = new_name(1)", final_text)

    def test_apply_structured_edit_rename_symbol_is_idempotent(self) -> None:
        root = self._workspace_scratch()
        sample = root / "ops.py"
        sample.write_text("def new_name(value):\n    return value\n\nRESULT = new_name(1)\n", encoding="utf-8")
        tools = ToolExecutor(root, approval_mode="auto")
        result = tools.apply_structured_edit({"op": "rename_symbol", "path": "ops.py", "old": "old_name", "new": "new_name"})

        self.assertTrue(result["ok"])
        self.assertEqual(result["count"], 0)
        self.assertIn("already renamed", result["summary"])

    def test_apply_structured_edit_adds_import(self) -> None:
        root = self._workspace_scratch()
        sample = root / "app.py"
        sample.write_text("def area(radius):\n    return math.pi * radius\n", encoding="utf-8")
        tools = ToolExecutor(root, approval_mode="auto")
        result = tools.apply_structured_edit({"op": "add_import", "path": "app.py", "statement": "import math"})
        final_text = sample.read_text(encoding="utf-8")

        self.assertTrue(result["ok"])
        self.assertTrue(final_text.startswith("import math\n"))

    def test_apply_structured_edit_replaces_function_body(self) -> None:
        root = self._workspace_scratch()
        sample = root / "ops.py"
        sample.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        tools = ToolExecutor(root, approval_mode="auto")
        result = tools.apply_structured_edit({"op": "replace_function_body", "path": "ops.py", "symbol": "add", "body": "return a + b"})
        final_text = sample.read_text(encoding="utf-8")

        self.assertTrue(result["ok"])
        self.assertIn("return a + b", final_text)
        self.assertNotIn("return a - b", final_text)

    def test_apply_structured_edit_changes_signature(self) -> None:
        root = self._workspace_scratch()
        sample = root / "ops.py"
        sample.write_text("def fetch_user(user_id):\n    return user_id\n", encoding="utf-8")
        tools = ToolExecutor(root, approval_mode="auto")
        result = tools.apply_structured_edit({"op": "change_signature", "path": "ops.py", "symbol": "fetch_user", "signature": "def fetch_user(user_id, include_orders=False):"})

        self.assertTrue(result["ok"])
        self.assertIn("def fetch_user(user_id, include_orders=False):", sample.read_text(encoding="utf-8"))

    def test_apply_structured_edit_renames_symbol_project(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "src" / "pricing.py").write_text("def total(items):\n    return sum(items)\n", encoding="utf-8")
        (root / "tests" / "test_pricing.py").write_text("from src.pricing import total\n\nassert total([1]) == 1\n", encoding="utf-8")
        tools = ToolExecutor(root, approval_mode="auto")
        result = tools.apply_structured_edit({"op": "rename_symbol_project", "path": ".", "old": "total", "new": "cart_total"})

        self.assertTrue(result["ok"])
        self.assertIn("def cart_total", (root / "src" / "pricing.py").read_text(encoding="utf-8"))
        self.assertIn("cart_total", (root / "tests" / "test_pricing.py").read_text(encoding="utf-8"))

    def test_apply_structured_edit_rename_symbol_project_is_idempotent(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "src" / "pricing.py").write_text("def cart_total(items):\n    return sum(items)\n", encoding="utf-8")
        (root / "tests" / "test_pricing.py").write_text("from src.pricing import cart_total\n\nassert cart_total([1]) == 1\n", encoding="utf-8")
        tools = ToolExecutor(root, approval_mode="auto")
        result = tools.apply_structured_edit({"op": "rename_symbol_project", "path": ".", "old": "total", "new": "cart_total"})

        self.assertTrue(result["ok"])
        self.assertEqual(result["count"], 0)
        self.assertIn("already renamed", result["summary"])

    def test_generate_tests_from_spec_previews_patch_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.generate_tests_from_spec("ops.add", "adds two positive integers")

        self.assertTrue(result["ok"])
        self.assertFalse(result["applied"])
        self.assertIn("tests/test_add_spec.py", result["path"])
        self.assertFalse((root / "tests" / "test_add_spec.py").exists())

    def test_run_shell_returns_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            command = f'{sys.executable} -c "print(123)"'
            result = tools.run_shell(command)
        self.assertTrue(result["ok"])
        self.assertEqual(result["output"], "123")

    def test_execute_reports_interrupt_for_run_shell(self) -> None:
        root = self._workspace_scratch()
        tools = ToolExecutor(root, approval_mode="auto")
        interrupted = threading.Event()
        tools.set_interrupt_event(interrupted)
        trigger = threading.Timer(0.2, interrupted.set)
        trigger.start()
        try:
            result = tools.execute("run_shell", {"command": f'"{sys.executable}" -c "import time; time.sleep(5)"', "timeout": 10})
        finally:
            trigger.cancel()

        self.assertFalse(result["ok"])
        self.assertTrue(result["interrupted"])
        self.assertEqual(result["summary"], "Interrupted by user.")

    def test_run_shell_ignores_posix_shell_env_override(self) -> None:
        tools = ToolExecutor(Path.cwd(), approval_mode="auto")
        completed = subprocess.CompletedProcess(args=["echo", "ok"], returncode=0, stdout="ok\n", stderr="")
        with patch("ollama_code.tools.os.name", "posix"):
            with patch.dict(os.environ, {"SHELL": "/usr/bin/fish"}, clear=False):
                with patch.object(ToolExecutor, "_run_process", return_value=completed) as run_mock:
                    result = tools.run_shell("echo ok")

        self.assertTrue(result["ok"])
        self.assertEqual(result["output"], "ok")
        kwargs = run_mock.call_args.kwargs
        self.assertTrue(kwargs["shell"])

    def test_run_shell_rejects_dangerous_validated_git_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            with patch("ollama_code.tools.shutil.which", return_value="git"):
                with patch.object(ToolExecutor, "_run_process") as run_mock:
                    result = tools.run_shell("git reset --hard")

        self.assertFalse(result["ok"])
        self.assertIn("Command rejected before execution", result["summary"])
        self.assertEqual(result["validation"]["family"], "git")
        run_mock.assert_not_called()

    def test_run_shell_rejects_unmatched_quotes_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            with patch.object(ToolExecutor, "_run_process") as run_mock:
                result = tools.run_shell('echo "unterminated')

        self.assertFalse(result["ok"])
        self.assertIn("invalid quoting", result["summary"])
        run_mock.assert_not_called()

    def test_run_shell_rejects_bash_syntax_error_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            syntax_result = subprocess.CompletedProcess(
                args=["bash", "-n", "-c", "if true; then echo hi"],
                returncode=2,
                stdout="",
                stderr="bash: -c: line 2: syntax error: unexpected end of file\n",
            )
            with patch("ollama_code.tools.shutil.which", return_value="bash"):
                with patch("ollama_code.tools.subprocess.run", return_value=syntax_result) as syntax_mock:
                    with patch.object(ToolExecutor, "_run_process") as run_mock:
                        result = tools.run_shell("if true; then echo hi")

        self.assertFalse(result["ok"])
        self.assertEqual(result["validation"]["family"], "bash")
        self.assertIn("bash -n rejected command syntax", result["summary"])
        self.assertEqual(syntax_mock.call_count, 1)
        run_mock.assert_not_called()

    def test_run_shell_bash_checks_then_runs_valid_unknown_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            run_result = subprocess.CompletedProcess(args="echo ok", returncode=0, stdout="ok\n", stderr="")
            with patch("ollama_code.tools.shutil.which", return_value="bash"):
                with patch("ollama_code.tools.subprocess.run") as syntax_mock:
                    with patch.object(ToolExecutor, "_run_process", return_value=run_result) as run_mock:
                        result = tools.run_shell("echo ok")

        self.assertTrue(result["ok"])
        self.assertEqual(result["output"], "ok")
        syntax_mock.assert_not_called()
        self.assertEqual(run_mock.call_count, 1)
        self.assertTrue(run_mock.call_args.kwargs["shell"])

    def test_run_shell_bash_checks_shell_metachar_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            syntax_result = subprocess.CompletedProcess(args=["bash", "-n", "-c", "echo ok | cat"], returncode=0, stdout="", stderr="")
            run_result = subprocess.CompletedProcess(args="echo ok | cat", returncode=0, stdout="ok\n", stderr="")
            with patch("ollama_code.tools.shutil.which", return_value="bash"):
                with patch("ollama_code.tools.subprocess.run", return_value=syntax_result) as syntax_mock:
                    with patch.object(ToolExecutor, "_run_process", return_value=run_result) as run_mock:
                        result = tools.run_shell("echo ok | cat")

        self.assertTrue(result["ok"])
        self.assertEqual(result["output"], "ok")
        self.assertEqual(syntax_mock.call_count, 1)
        self.assertEqual(run_mock.call_count, 1)
        self.assertTrue(run_mock.call_args.kwargs["shell"])

    def test_run_shell_rejects_missing_unknown_executable_after_bash_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")

            def fake_which(name: str) -> str | None:
                return "bash" if name == "bash" else None

            with patch("ollama_code.tools.shutil.which", side_effect=fake_which):
                with patch("ollama_code.tools.subprocess.run") as syntax_mock:
                    with patch.object(ToolExecutor, "_run_process") as run_mock:
                        result = tools.run_shell("foozle --help")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_class"], "command_not_found")
        self.assertIn("executable not found: foozle", result["summary"])
        syntax_mock.assert_not_called()
        run_mock.assert_not_called()

    def test_run_shell_rejects_path_escape_for_validated_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            with patch("ollama_code.tools.shutil.which", return_value="pytest"):
                with patch.object(ToolExecutor, "_run_process") as run_mock:
                    result = tools.run_shell("pytest ../outside")

        self.assertFalse(result["ok"])
        self.assertIn("path escapes workspace", result["summary"])
        run_mock.assert_not_called()

    def test_run_shell_validates_local_executable_relative_to_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "service"
            project.mkdir()
            (project / "gradlew.bat").write_text("@echo off\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            completed = subprocess.CompletedProcess(args=["gradlew.bat", "test"], returncode=0, stdout="ok\n", stderr="")

            with patch("ollama_code.tools.shutil.which", return_value=None):
                with patch.object(ToolExecutor, "_run_process", return_value=completed) as run_mock:
                    result = tools.run_shell("gradlew.bat test", cwd="service", timeout="7")

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["validation"]["family"], "gradlew.bat")
        self.assertEqual(run_mock.call_args.args[0], ["gradlew.bat", "test"])
        self.assertEqual(run_mock.call_args.kwargs["cwd"], project)
        self.assertEqual(run_mock.call_args.kwargs["timeout"], 7)

    def test_run_shell_runs_valid_common_command_without_shell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            completed = subprocess.CompletedProcess(args=["git", "status", "--short"], returncode=0, stdout="ok\n", stderr="")
            with patch("ollama_code.tools.shutil.which", return_value="git"):
                with patch.object(ToolExecutor, "_run_process", return_value=completed) as run_mock:
                    result = tools.run_shell("git status --short")

        self.assertTrue(result["ok"])
        self.assertEqual(result["validation"]["family"], "git")
        self.assertEqual(run_mock.call_args.args[0], ["git", "status", "--short"])
        self.assertFalse(run_mock.call_args.kwargs["shell"])

    @unittest.skipUnless(os.name == "nt", "Windows only")
    def test_run_shell_supports_powershell_cmdlets_on_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            result = tools.run_shell("Write-Output 123")
        self.assertTrue(result["ok"])
        self.assertEqual(result["output"], "123")

    @unittest.skipUnless(os.name == "nt", "Windows only")
    def test_run_shell_does_not_misclassify_dollar_sign_inside_quoted_argument(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            command = f'{sys.executable} -c "print(\'a $HOME b\')"'
            result = tools.run_shell(command)
        self.assertTrue(result["ok"])
        self.assertEqual(result["output"], "a $HOME b")

    def test_run_test_uses_configured_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = f'"{sys.executable}" -c "print(321)"'
            tools = ToolExecutor(root, approval_mode="auto", test_command=command)
            result = tools.run_test()
        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "run_test")
        self.assertEqual(result["output"], "321")

    def test_run_test_normalizes_escaped_cwd_to_workspace_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = f'"{sys.executable}" -c "print(789)"'
            tools = ToolExecutor(root, approval_mode="auto", test_command=command)
            result = tools.run_test(cwd=str(root.parent / "other-workspace"))

        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "run_test")
        self.assertEqual(result["output"], "789")
        self.assertIn("Ignored run_test cwd outside workspace", result["normalized"])

    def test_run_test_requires_configured_or_explicit_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            result = tools.run_test()
        self.assertFalse(result["ok"])
        self.assertIn("no runnable test validator", result["summary"])

    def test_run_test_discovers_unittest_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "tests" / "test_sample.py").write_text("import unittest\n\nclass SampleTests(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.run_test()
        self.assertTrue(result["ok"], result.get("output"))
        self.assertTrue(result["discovered"])
        self.assertIn("unittest discover", result["command"])

    def test_diagnose_test_failure_classifies_common_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            syntax = tools.diagnose_test_failure("SyntaxError: invalid syntax\n  File \"app.py\", line 1")
            missing = tools.diagnose_test_failure("ModuleNotFoundError: No module named 'requests_mock'")
            invalid = tools.diagnose_test_failure("usage: pytest [options]\nerror: unrecognized arguments: --wat")

        self.assertEqual(syntax["error_class"], "syntax_error")
        self.assertEqual(syntax["next_tool"], "lint_typecheck")
        self.assertEqual(missing["error_class"], "missing_dependency")
        self.assertEqual(missing["missing_dependency"], "requests_mock")
        self.assertEqual(missing["next_tool"], "fail_closed")
        self.assertEqual(invalid["error_class"], "invalid_args")

    def test_read_only_blocks_shell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="read-only")
            result = tools.run_shell("echo denied")
        self.assertFalse(result["ok"])
        self.assertIn("read-only", result["summary"])

    def test_git_status_and_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.init_git_repo(root)
            tracked = root / "tracked.txt"
            tracked.write_text("before\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "initial"], cwd=root, check=True, capture_output=True, text=True)
            tracked.write_text("before\nafter\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            status = tools.git_status()
            diff = tools.git_diff(path="tracked.txt")

        self.assertTrue(status["ok"])
        self.assertIn("tracked.txt", status["output"])
        self.assertTrue(diff["ok"])
        self.assertIn("+after", diff["output"])

    def test_git_commit_creates_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.init_git_repo(root)
            tracked = root / "tracked.txt"
            tracked.write_text("before\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "initial"], cwd=root, check=True, capture_output=True, text=True)
            tools = ToolExecutor(root, approval_mode="auto")
            tracked.write_text("before\nafter\n", encoding="utf-8")
            result = tools.git_commit("Update tracked file")
            subject = subprocess.run(
                ["git", "log", "-1", "--pretty=%s"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        self.assertTrue(result["ok"])
        self.assertEqual(subject, "Update tracked file")

    def test_git_commit_add_all_refuses_preexisting_dirty_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.init_git_repo(root)
            tracked = root / "tracked.txt"
            unrelated = root / "notes.txt"
            tracked.write_text("before\n", encoding="utf-8")
            unrelated.write_text("draft\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt", "notes.txt"], cwd=root, check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "initial"], cwd=root, check=True, capture_output=True, text=True)
            unrelated.write_text("draft\nlocal-only\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            tracked.write_text("before\nafter\n", encoding="utf-8")
            result = tools.git_commit("Update tracked file")
            subject = subprocess.run(
                ["git", "log", "-1", "--pretty=%s"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            staged = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        self.assertFalse(result["ok"])
        self.assertIn("already had unrelated local changes", result["summary"])
        self.assertIn("notes.txt", result["summary"])
        self.assertEqual(subject, "initial")
        self.assertEqual(staged, "")

    def test_read_only_blocks_git_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.init_git_repo(root)
            tracked = root / "tracked.txt"
            tracked.write_text("before\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "initial"], cwd=root, check=True, capture_output=True, text=True)
            tracked.write_text("before\nafter\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="read-only")
            result = tools.git_commit("Blocked commit")
        self.assertFalse(result["ok"])
        self.assertIn("read-only", result["summary"])
