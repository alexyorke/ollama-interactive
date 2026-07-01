from __future__ import annotations

from contextlib import contextmanager
import gc
import os
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from ollama_code.tools import ToolExecutor, format_compact_tool_help, format_tool_group_help


class ToolExecutorTests(unittest.TestCase):
    def _workspace_scratch(self) -> Path:
        root = (Path.cwd() / "verify_scratch" / f"test-tools-{uuid4().hex}").resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return root

    @contextmanager
    def _temp_tools(self, *, approval_mode: str = "auto", **kwargs: object):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            yield root, ToolExecutor(root, approval_mode=approval_mode, **kwargs)

    @contextmanager
    def _temp_files_tools(
        self,
        files: dict[str, str],
        *,
        approval_mode: str = "auto",
        **kwargs: object,
    ):
        with self._temp_tools(approval_mode=approval_mode, **kwargs) as (root, tools):
            for relative_path, content in files.items():
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            yield root, tools

    @contextmanager
    def _temp_python_tools(
        self,
        files: dict[str, str],
        *,
        approval_mode: str = "auto",
        test_command: str | None = None,
        test_discover_args: tuple[str, ...] = ("-p", "*_test.py", "-v"),
        **kwargs: object,
    ):
        resolved_test_command = test_command or subprocess.list2cmdline(
            [sys.executable, "-m", "unittest", "discover", *test_discover_args]
        )
        with self._temp_files_tools(
            files,
            approval_mode=approval_mode,
            test_command=resolved_test_command,
            **kwargs,
        ) as (root, tools):
            yield root, tools, resolved_test_command

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

    def init_git_repo_with_commit(self, root: Path, files: dict[str, str], *, message: str = "initial") -> None:
        self.init_git_repo(root)
        for relative_path, content in files.items():
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        subprocess.run(["git", "add", *sorted(files)], cwd=root, check=True, capture_output=True, text=True)
        subprocess.run(["git", "commit", "-m", message], cwd=root, check=True, capture_output=True, text=True)

    def test_list_and_read_file(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "alpha.txt").write_text("line1\nline2\n", encoding="utf-8")
            listed = tools.list_files()
            read = tools.read_file("alpha.txt", start=2, end=2)

        self.assertTrue(listed["ok"])
        self.assertIn("alpha.txt", listed["output"])
        self.assertEqual(read["output"], "   2 | line2")

    def test_tool_catalog_groups_model_facing_tools(self) -> None:
        compact = format_compact_tool_help({"read_file", "ast_search", "run_test"}, grouped=True)
        groups = format_tool_group_help({"read_file", "ast_search", "run_test"})

        self.assertIn("[navigation]", compact)
        self.assertIn("[structural]", compact)
        self.assertIn("[validation]", compact)
        self.assertIn("read_file(path,start=1,end=200)", compact)
        self.assertIn("navigation:", groups)
        self.assertIn("ast_search", groups)

    def test_non_git_tasks_work_when_git_is_missing(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "alpha.txt").write_text("line1\n", encoding="utf-8")
            with patch("ollama_code.tools.shutil.which", return_value=None):
                listed = tools.list_files()
                git_status = tools.git_status()

        self.assertTrue(listed["ok"])
        self.assertIn("alpha.txt", listed["output"])
        self.assertFalse(git_status["ok"])
        self.assertIn("git is not installed", git_status["summary"])

    def test_read_toml_extracts_tool_sections_when_no_toml_module(self) -> None:
        with self._temp_tools() as (root, tools):
            config = root / "pyproject.toml"
            config.write_text("[tool.example]\nname='demo'\n", encoding="utf-8")
            with patch("ollama_code.tools.tomllib", None):
                payload = tools._read_toml(config)

        self.assertEqual(payload, {"tool": {"example": {}}})

    def test_git_tools_can_target_nested_repo_path(self) -> None:
        root = self._workspace_scratch()
        repo = root / "nested-repo"
        repo.mkdir(parents=True, exist_ok=True)
        self.init_git_repo(repo)
        (repo / "note.txt").write_text("hello\n", encoding="utf-8")

        tools = ToolExecutor(root, approval_mode="auto")
        status = tools.git_status("nested-repo")

        self.assertTrue(status["ok"])
        self.assertIn("##", status["output"])

    def test_git_status_defaults_to_single_nested_repo(self) -> None:
        root = self._workspace_scratch()
        repo = root / "only-repo"
        repo.mkdir(parents=True, exist_ok=True)
        self.init_git_repo(repo)
        (repo / "note.txt").write_text("hello\n", encoding="utf-8")

        tools = ToolExecutor(root, approval_mode="auto")
        status = tools.git_status()

        self.assertTrue(status["ok"])
        self.assertIn("##", status["output"])

    def test_write_and_replace_file(self) -> None:
        with self._temp_tools() as (root, tools):
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

        indexer = FakeIndexer()
        with self._temp_tools(indexer=indexer) as (_root, tools):
            result = tools.execute("write_file", {"path": "note.txt", "content": "hello\n"})

        self.assertTrue(result["ok"])
        self.assertIn("note.txt", indexer.paths)

    def test_write_file_reports_python_syntax_error_without_blocking_write(self) -> None:
        with self._temp_tools() as (root, tools):
            result = tools.write_file("bad.py", "def f():\nreturn 1\n")
            final_text = (root / "bad.py").read_text(encoding="utf-8")

        self.assertTrue(result["ok"])
        self.assertFalse(result["syntax_ok"])
        self.assertIn("Python syntax error", result["summary"])
        self.assertIn("bad.py:2", result["diagnostic"])
        self.assertEqual(final_text, "def f():\nreturn 1\n")

    def test_write_file_auto_dedents_globally_indented_python(self) -> None:
        with self._temp_tools() as (root, tools):
            result = tools.write_file("fixed.py", "    def f():\n        return 1\n")
            final_text = (root / "fixed.py").read_text(encoding="utf-8")

        self.assertTrue(result["ok"])
        self.assertNotIn("syntax_ok", result)
        self.assertIn("Auto-dedented", result["summary"])
        self.assertEqual(final_text, "def f():\n    return 1\n")

    def test_write_file_strips_markdown_quote_prefixes_for_python(self) -> None:
        with self._temp_tools() as (root, tools):
            result = tools.write_file("quoted.py", "> def f():\n>     return 1\n")
            final_text = (root / "quoted.py").read_text(encoding="utf-8")

        self.assertTrue(result["ok"])
        self.assertIn("quote prefixes", result["summary"])
        self.assertEqual(final_text, "def f():\n    return 1\n")

    def test_write_file_strips_single_leading_markdown_quote_for_python(self) -> None:
        with self._temp_tools() as (root, tools):
            result = tools.write_file("quoted.py", "> import re\n\ndef f():\n    return re.escape('x')\n")
            final_text = (root / "quoted.py").read_text(encoding="utf-8")

        self.assertTrue(result["ok"])
        self.assertIn("quote prefixes", result["summary"])
        self.assertEqual(final_text, "import re\n\ndef f():\n    return re.escape('x')\n")

    def test_write_file_repairs_common_join_string_typo_when_parseable(self) -> None:
        with self._temp_tools() as (root, tools):
            result = tools.write_file("joiner.py", "def f(items):\n    return '.join(items)\n")
            final_text = (root / "joiner.py").read_text(encoding="utf-8")

        self.assertTrue(result["ok"])
        self.assertIn("join string typo", result["summary"])
        self.assertEqual(final_text, 'def f(items):\n    return " ".join(items)\n')

    def test_write_file_rejects_partial_python_overwrite_that_drops_symbols(self) -> None:
        with self._temp_tools() as (root, tools):
            target = root / "list_ops.py"
            target.write_text(
                "def append(a, b):\n    return a + b\n\n\ndef reverse(items):\n    return list(reversed(items))\n",
                encoding="utf-8",
            )
            result = tools.write_file("list_ops.py", "def append(a, b):\n    return a + b\n")

        self.assertFalse(result["ok"])
        self.assertIn("partial content", result["summary"])
        self.assertIn("reverse", result["summary"])

    def test_write_file_rejects_generated_cache_path(self) -> None:
        with self._temp_tools() as (_root, tools):
            result = tools.write_file("__pycache__/sample.pyc", "")

        self.assertFalse(result["ok"])
        self.assertIn("generated/cache", result["summary"])

    def test_write_file_rejects_omitted_context_marker(self) -> None:
        with self._temp_tools() as (_root, tools):
            result = tools.write_file("sample.py", "[omitted 500 chars from prior content; do not copy]")

        self.assertFalse(result["ok"])
        self.assertIn("omitted-context marker", result["summary"])

    def test_replace_in_file_reports_python_syntax_error_without_blocking_edit(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "sample.py").write_text("def f():\n    return 1\n", encoding="utf-8")
            result = tools.replace_in_file("sample.py", "    return 1", "return 1")

        self.assertTrue(result["ok"])
        self.assertFalse(result["syntax_ok"])
        self.assertIn("Python syntax error", result["summary"])

    def test_replace_rejects_ambiguous_short_snippet(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "sample.txt").write_text("live smoke ok\n", encoding="utf-8")
            result = tools.replace_in_file("sample.txt", "ok", "passed", replace_all=True)
        self.assertFalse(result["ok"])
        self.assertIn("ambiguous", result["summary"])

    def test_replace_can_match_whole_word(self) -> None:
        with self._temp_tools() as (root, tools):
            sample = root / "sample.txt"
            sample.write_text("live smoke ok\n", encoding="utf-8")
            result = tools.replace_in_file("sample.txt", "ok", "passed", match_whole_word=True)
            final_text = sample.read_text(encoding="utf-8")
        self.assertTrue(result["ok"])
        self.assertEqual(final_text, "live smoke passed\n")

    def test_replace_tolerates_leading_whitespace_mismatch(self) -> None:
        with self._temp_tools() as (root, tools):
            sample = root / "sample.py"
            sample.write_text("def f():\n    return 'old'\n", encoding="utf-8")
            result = tools.replace_in_file("sample.py", "   return 'old'", "    return 'new'")
            final_text = sample.read_text(encoding="utf-8")
        self.assertTrue(result["ok"])
        self.assertEqual(final_text, "def f():\n    return 'new'\n")

    def test_replace_in_file_identifier_call_rename_ignores_embedded_matches(self) -> None:
        with self._temp_tools() as (root, tools):
            sample = root / "test_pricing.py"
            sample.write_text(
                "def test_cart_total():\n"
                "    assert cart_total([1]) == 1\n"
                "    assert total([2]) == 2\n",
                encoding="utf-8",
            )
            result = tools.replace_in_file("test_pricing.py", "total(", "cart_total(")
            final_text = sample.read_text(encoding="utf-8")

        self.assertTrue(result["ok"], result)
        self.assertIn("def test_cart_total():", final_text)
        self.assertIn("assert cart_total([1]) == 1", final_text)
        self.assertIn("assert cart_total([2]) == 2", final_text)
        self.assertNotIn("cart_cart_total", final_text)

    def test_replace_strips_read_file_line_number_prefixes(self) -> None:
        with self._temp_tools() as (root, tools):
            sample = root / "sample.py"
            sample.write_text("def f():\n    return 'old'\n", encoding="utf-8")
            result = tools.replace_in_file(
                "sample.py",
                "   1 | def f():\n   2 |     return 'old'",
                "   1 | def f():\n   2 |     return 'new'",
            )
            final_text = sample.read_text(encoding="utf-8")
        self.assertTrue(result["ok"])
        self.assertEqual(final_text, "def f():\n    return 'new'\n")

    def test_replace_regex_fallback_inserts_backslashes_literally(self) -> None:
        with self._temp_tools() as (root, tools):
            sample = root / "sample.py"
            sample.write_text("def f(value):\n    return value.strip().lower()\n", encoding="utf-8")
            result = tools.replace_in_file(
                "sample.py",
                "   return value.strip().lower()",
                "    return re.sub(r'\\s+', '-', value.strip().lower())",
            )
            final_text = sample.read_text(encoding="utf-8")
        self.assertTrue(result["ok"])
        self.assertIn("r'\\s+'", final_text)

    def test_read_only_blocks_mutations(self) -> None:
        with self._temp_tools(approval_mode="read-only") as (_root, tools):
            result = tools.write_file("blocked.txt", "data")
        self.assertFalse(result["ok"])
        self.assertIn("read-only", result["summary"])

    def test_path_escape_is_blocked(self) -> None:
        with self._temp_tools() as (_root, tools):
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
        with self._temp_tools() as (root, tools):
            target = root / "nested" / "file.txt"
            target.parent.mkdir(parents=True)
            target.write_text("hello\n", encoding="utf-8")

            result = tools.read_file(r"nested\file.txt", start=1, end=1)

        self.assertTrue(result["ok"])
        self.assertEqual(result["path"], "nested/file.txt")
        self.assertIn("hello", result["output"])

    def test_read_file_accepts_numeric_string_line_bounds(self) -> None:
        with self._temp_tools() as (root, tools):
            target = root / "sample.txt"
            target.write_text("one\ntwo\nthree\n", encoding="utf-8")

            result = tools.execute("read_file", {"path": "sample.txt", "start": "2", "end": "2"})

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["start"], 2)
        self.assertEqual(result["end"], 2)
        self.assertIn("two", result["output"])
        self.assertNotIn("one", result["output"])

    def test_list_files_accepts_numeric_string_limits(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "nested").mkdir()
            (root / "nested" / "child.txt").write_text("x\n", encoding="utf-8")
            (root / "top.txt").write_text("x\n", encoding="utf-8")

            result = tools.execute("list_files", {"max_depth": "0", "limit": "1"})

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["count"], 1)

    def test_search_uses_python_fallback_when_rg_is_missing(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "beta.txt").write_text("needle here\n", encoding="utf-8")
            with patch.object(shutil, "which", return_value=None):
                result = tools.search("needle")
        self.assertTrue(result["ok"])
        self.assertIn("beta.txt:1:needle here", result["output"])

    def test_search_limits_rg_output_lines(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "many.txt").write_text("\n".join("needle" for _ in range(20)), encoding="utf-8")
            result = tools.search("needle", limit=3)

        self.assertTrue(result["ok"])
        self.assertLessEqual(len(result["output"].splitlines()), 3)

    def test_search_rg_output_is_workspace_relative(self) -> None:
        with self._temp_tools() as (root, tools):
            target = root / "src" / "app.py"
            target.parent.mkdir(parents=True)
            target.write_text("needle\n", encoding="utf-8")
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=f"{target}:1:needle\n",
                stderr="",
            )
            with patch.object(shutil, "which", return_value="rg"):
                with patch.object(tools, "_run_process", return_value=completed):
                    result = tools.search("needle", path="src")

        self.assertTrue(result["ok"])
        self.assertEqual(result["output"], "src/app.py:1:needle")

    def test_search_falls_back_to_literal_when_rg_regex_is_invalid(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "pricing.py").write_text("def total(prices):\n    return sum(prices)\n", encoding="utf-8")
            result = tools.search("total(", limit=5)

        self.assertTrue(result["ok"])
        self.assertIn("total(prices)", str(result["output"]))

    def test_search_accepts_filename_glob_filter(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("needle in python\n", encoding="utf-8")
            (root / "src" / "notes.md").write_text("needle in docs\n", encoding="utf-8")

            result = tools.search("needle", path="src", file_glob="*.py")

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["file_glob"], "*.py")
        self.assertIn("app.py", result["output"])
        self.assertNotIn("notes.md", result["output"])

    def test_code_outline_returns_python_symbols_without_bodies(self) -> None:
        with self._temp_tools() as (root, tools):
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
            result = tools.code_outline("src/service.py")

        self.assertTrue(result["ok"])
        self.assertIn("class PricingService", result["output"])
        self.assertIn("PricingService.calculate_total", result["output"])
        self.assertIn("function helper", result["output"])
        self.assertIn("imports: import math", result["output"])
        self.assertNotIn("hidden_body_marker", result["output"])

    def test_code_outline_defaults_to_workspace_root(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "app.py").write_text("def meaning():\n    return 42\n", encoding="utf-8")
            result = tools.execute("code_outline", {})

        self.assertTrue(result["ok"])
        self.assertEqual(result["path"], ".")
        self.assertIn("meaning", result["output"])

    def test_search_symbols_and_read_symbol_target_large_python_file(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "src").mkdir()
            filler = "\n\n".join(f"def filler_{index}():\n    return {index}" for index in range(120))
            target_body = (
                "def calculate_discount(cart, percentage):\n"
                "    subtotal = sum(item['price'] for item in cart)\n"
                "    discount = subtotal * percentage\n"
                "    return max(0, subtotal - discount)\n"
            )
            (root / "src" / "pricing.py").write_text(f"{filler}\n\n{target_body}\n", encoding="utf-8")

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
        with self._temp_tools() as (root, tools):
            (root / "src").mkdir()
            (root / "node_modules" / "pkg").mkdir(parents=True)
            (root / "src" / "app.py").write_text("def real_target():\n    return 1\n", encoding="utf-8")
            (root / "node_modules" / "pkg" / "bad.py").write_text("def real_target_dependency():\n    return 2\n", encoding="utf-8")
            result = tools.search_symbols("real_target")

        self.assertTrue(result["ok"])
        self.assertIn("src/app.py", result["output"])
        self.assertNotIn("node_modules", result["output"])

    def test_symbol_tools_tolerate_whitespace_only_python_docstrings(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "pkg").mkdir()
            (root / "pkg" / "mod.py").write_text(
                "def outer():\n"
                "    \"\"\"   \n"
                "    \"\"\"\n"
                "    return 1\n",
                encoding="utf-8",
            )

            search = tools.search_symbols("outer")
            repo = tools.repo_index_search("outer")
            outline = tools.code_outline("pkg/mod.py")

        self.assertTrue(search["ok"], search)
        self.assertIn("pkg/mod.py", search["output"])
        self.assertTrue(repo["ok"], repo)
        self.assertIn("outer", repo["output"])
        self.assertTrue(outline["ok"], outline)
        self.assertIn("outer", outline["output"])

    def test_symbol_tools_scan_beyond_first_two_hundred_code_files(self) -> None:
        with self._temp_tools() as (root, tools):
            src = root / "src"
            src.mkdir()
            for index in range(260):
                name = f"filler_{index:03d}"
                (src / f"{name}.py").write_text(
                    f"def {name}():\n"
                    f"    return {index}\n",
                    encoding="utf-8",
                )
            (src / "zz_target.py").write_text(
                "def separability_matrix():\n"
                "    return 'needle_symbol'\n",
                encoding="utf-8",
            )

            search = tools.search_symbols("separability_matrix")
            repo = tools.repo_index_search("separability_matrix")

        self.assertTrue(search["ok"], search)
        self.assertIn("src/zz_target.py", search["output"])
        self.assertTrue(repo["ok"], repo)
        self.assertIn("src/zz_target.py", repo["output"])

    def test_read_symbol_reports_ambiguous_names(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "models.py").write_text(
                "class Alpha:\n"
                "    def save(self):\n"
                "        return 'a'\n\n"
                "class Beta:\n"
                "    def save(self):\n"
                "        return 'b'\n",
                encoding="utf-8",
            )

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

    def test_replace_symbol_normalizes_parameter_names(self) -> None:
        root = self._workspace_scratch()
        sample = root / "ops.py"
        sample.write_text("def append(list1, list2):\n    pass\n", encoding="utf-8")
        tools = ToolExecutor(root, approval_mode="auto")
        result = tools.replace_symbol(
            "ops.py",
            "append",
            "def append(list, function):\n    return list + [function]\n",
        )

        final_text = sample.read_text(encoding="utf-8")

        self.assertTrue(result["ok"], result)
        self.assertIn("Normalized replacement signature", result["summary"])
        self.assertIn("def append(list1, list2):", final_text)
        self.assertIn("return list1 + [list2]", final_text)

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

    def test_replace_symbols_normalizes_permuted_existing_parameter_order(self) -> None:
        root = self._workspace_scratch()
        sample = root / "ops.py"
        sample.write_text("def foldl(function, list, initial):\n    pass\n", encoding="utf-8")
        tools = ToolExecutor(root, approval_mode="auto")
        result = tools.replace_symbols(
            "ops.py",
            [
                {
                    "symbol": "foldl",
                    "content": (
                        "def foldl(function, initial, list):\n"
                        "    accumulator = initial\n"
                        "    for item in list:\n"
                        "        accumulator = function(accumulator, item)\n"
                        "    return accumulator\n"
                    ),
                }
            ],
        )
        final_text = sample.read_text(encoding="utf-8")

        self.assertTrue(result["ok"], result)
        self.assertIn("Normalized replacement signature", result["summary"])
        self.assertIn("def foldl(function, list, initial):", final_text)
        self.assertIn("for item in list:", final_text)

    def test_replace_symbols_normalizes_parameter_names(self) -> None:
        root = self._workspace_scratch()
        sample = root / "ops.py"
        sample.write_text("def append(list1, list2):\n    pass\n", encoding="utf-8")
        tools = ToolExecutor(root, approval_mode="auto")
        result = tools.replace_symbols(
            "ops.py",
            [{"symbol": "append", "content": "def append(list, function):\n    return list + [function]\n"}],
        )
        final_text = sample.read_text(encoding="utf-8")

        self.assertTrue(result["ok"], result)
        self.assertIn("Normalized replacement signature", result["summary"])
        self.assertIn("def append(list1, list2):", final_text)
        self.assertIn("return list1 + [list2]", final_text)

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
        with self._temp_files_tools(
            {
                "pricing.py": (
                    "def calculate_discount():\n"
                    "    return 'TOKEN_SYMBOL_750'\n"
                )
            }
        ) as (_root, tools):
            result = tools.execute(
                "read_symbol",
                {"path": "pricing.py", "symbol": "calculate_discount", "start": 1, "end": 20},
            )

        self.assertTrue(result["ok"])
        self.assertIn("TOKEN_SYMBOL_750", result["output"])

    def test_code_outline_supports_generic_javascript_symbols(self) -> None:
        with self._temp_files_tools(
            {
                "web.js": (
                    "export class Widget {\n"
                    "  render() { return '<div></div>'; }\n"
                    "}\n\n"
                    "export function hydrateWidget() {\n"
                    "  return new Widget();\n"
                    "}\n"
                )
            }
        ) as (_root, tools):
            outline = tools.code_outline("web.js")
            symbol = tools.read_symbol("web.js", "hydrateWidget", include_context=0)

        self.assertTrue(outline["ok"])
        self.assertIn("class Widget", outline["output"])
        self.assertIn("function hydrateWidget", outline["output"])
        self.assertTrue(symbol["ok"])
        self.assertIn("new Widget", symbol["output"])

    def test_repo_index_search_returns_ranked_compact_snippets(self) -> None:
        with self._temp_files_tools(
            {
                "pricing.py": (
                    "def calculate_discount(cart):\n"
                    "    return sum(cart) * 0.9\n"
                )
            }
        ) as (_root, tools):
            result = tools.repo_index_search("discount calculation")

        self.assertTrue(result["ok"])
        self.assertIn("pricing.py", result["output"])
        self.assertIn("calculate_discount", result["output"])
        self.assertNotIn("   1 |", result["output"])

    def test_fts_search_indexes_symbols_headings_and_scopes_paths(self) -> None:
        with self._temp_files_tools(
            {
                "src/pricing.py": (
                    "def calculate_discount(cart):\n"
                    "    \"\"\"Seasonal discount calculation.\"\"\"\n"
                    "    return sum(cart) * 0.9\n"
                ),
                "docs/guide.md": "# Seasonal Pricing\nUse discount calculation carefully.\n",
            }
        ) as (_root, tools):
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

    def test_fts_refresh_reuses_unchanged_rows_and_deletes_stale_rows(self) -> None:
        with self._temp_files_tools({"notes.md": "# Alpha\nReusable content\n"}) as (root, tools):
            target = root / "notes.md"
            first = tools.fts_refresh()
            second = tools.fts_refresh()
            target.unlink()
            third = tools.fts_refresh()
            search = tools.fts_search("Reusable", refresh=False)

        self.assertTrue(first["ok"], first)
        self.assertEqual(first["indexed"], 1)
        self.assertEqual(first["unchanged"], 0)
        self.assertTrue(second["ok"], second)
        self.assertEqual(second["indexed"], 0)
        self.assertEqual(second["unchanged"], 1)
        self.assertTrue(third["ok"], third)
        self.assertEqual(third["deleted"], 1)
        self.assertTrue(search["ok"], search)
        self.assertEqual(search["output"], "(no FTS matches)")

    def test_fd_search_reports_missing_fd(self) -> None:
        with self._temp_tools() as (_root, tools):
            with patch.object(ToolExecutor, "_fd_cli_path", return_value=None):
                result = tools.fd_search("README")

        self.assertFalse(result["ok"])
        self.assertEqual(result["missing_dependency"], "fd")

    def test_context_pack_returns_ranked_evidence_and_writes_index_cache(self) -> None:
        with self._temp_files_tools(
            {
                "src/pricing.py": "def calculate_discount(cart):\n    return sum(cart)\n",
                "tests/test_pricing.py": "from src.pricing import calculate_discount\n",
            }
        ) as (_root, tools):
            result = tools.context_pack("fix discount calculation")

        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "context_pack")
        self.assertIn("suggested_next_tool", result["output"])
        self.assertIn("pricing.py", result["output"])
        self.assertIn("test_pricing.py", result["output"])
        self.assertEqual(result["ranked_paths"], ["src/pricing.py"])
        self.assertEqual(result["ranked_symbols"], [{"path": "src/pricing.py", "qualname": "calculate_discount"}])

    def test_repo_index_search_invalidates_changed_files(self) -> None:
        with self._temp_files_tools({"ops.py": "def first_name():\n    return 'old'\n"}) as (root, tools):
            target = root / "ops.py"
            first = tools.repo_index_search("first_name")
            target.write_text("def second_name():\n    return 'new'\n", encoding="utf-8")
            second = tools.repo_index_search("second_name")

        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertIn("first_name", first["output"])
        self.assertIn("second_name", second["output"])

    def test_repo_index_search_invalidates_same_size_changed_files(self) -> None:
        with self._temp_tools() as (root, tools):
            target = root / "ops.py"
            target.write_text("def first_name():\n    return 'old'\n", encoding="utf-8")
            first = tools.repo_index_search("first_name")
            target.write_text("def third_name():\n    return 'new'\n", encoding="utf-8")
            second = tools.repo_index_search("third_name")

        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertIn("first_name", first["output"])
        self.assertIn("third_name", second["output"])

    def test_indexed_search_uses_cached_lines_and_invalidates_changed_files(self) -> None:
        with self._temp_tools() as (root, tools):
            target = root / "ops.py"
            target.write_text("def first_name():\n    return 'alpha needle'\n", encoding="utf-8")
            first = tools.indexed_search("alpha needle")
            target.write_text("def second_name():\n    return 'beta needle'\n", encoding="utf-8")
            second = tools.indexed_search("beta needle")

        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertIn("ops.py:2", first["output"])
        self.assertIn("alpha needle", first["output"])
        self.assertIn("beta needle", second["output"])

    def test_file_search_uses_cached_paths_and_invalidates_changed_files(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "src").mkdir()
            first_path = root / "src" / "alpha_report.txt"
            second_path = root / "src" / "beta_report.txt"
            first_path.write_text("one", encoding="utf-8")
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
        with self._temp_tools() as (root, tools):
            (root / "README.md").write_text("hello", encoding="utf-8")
            result = tools.file_index_refresh()
            payload = json.loads((root / ".ollama-code" / "index" / "file_index.json").read_text(encoding="utf-8"))

        self.assertTrue(result["ok"])
        self.assertEqual(result["files"], 1)
        self.assertIn("README.md", payload["files"])

    def test_file_search_respects_path_scope_and_skips_generated_dirs(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "src").mkdir()
            (root / "docs").mkdir()
            (root / "node_modules").mkdir()
            (root / ".ollama-code" / "index").mkdir(parents=True)
            (root / "src" / "target_config.py").write_text("x = 1\n", encoding="utf-8")
            (root / "docs" / "target_config.md").write_text("x\n", encoding="utf-8")
            (root / "node_modules" / "target_config.js").write_text("x\n", encoding="utf-8")
            (root / ".ollama-code" / "index" / "target_config.json").write_text("{}", encoding="utf-8")
            src_result = tools.file_search("target config", path="src")
            root_result = tools.file_search("target config")

        self.assertTrue(src_result["ok"])
        self.assertEqual(src_result["output"], "src/target_config.py")
        self.assertIn("src/target_config.py", root_result["output"])
        self.assertIn("docs/target_config.md", root_result["output"])
        self.assertNotIn("node_modules", root_result["output"])
        self.assertNotIn(".ollama-code", root_result["output"])

    def test_broad_repo_tools_skip_generated_and_ignored_dirs(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "src").mkdir()
            (root / "scratch").mkdir()
            (root / "verify_scratch").mkdir()
            (root / "ollama-code-bench-deadbeef").mkdir()
            (root / "probe-123").mkdir()
            (root / "tmpabc123").mkdir()
            (root / "src" / "app.py").write_text("def live_target():\n    return 'ok'\n", encoding="utf-8")
            for directory in ["scratch", "verify_scratch", "ollama-code-bench-deadbeef", "probe-123", "tmpabc123"]:
                (root / directory / "ignored_target.py").write_text("def ignored_target():\n    return 'skip'\n", encoding="utf-8")
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
        with self._temp_tools() as (root, tools):
            (root / "alpha").mkdir()
            (root / "alpha" / "notes.py").write_text("x = 1\n", encoding="utf-8")
            (root / "alpha_notes.py").write_text("x = 2\n", encoding="utf-8")
            result = tools.file_search("alpha_notes.py", limit=2)

        self.assertTrue(result["ok"])
        self.assertEqual(result["output"].splitlines()[0], "alpha_notes.py")

    def test_directory_search_matches_directory_name_globs(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "tests" / "mysql_backend").mkdir(parents=True)
            (root / "tests" / "postgres_backend").mkdir(parents=True)
            result = tools.directory_search("*mysql*", path="tests")

        self.assertTrue(result["ok"])
        self.assertEqual(result["output"], "tests/mysql_backend/")

    def test_directory_search_skips_hidden_and_generated_dirs(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "src" / "target_cache").mkdir(parents=True)
            (root / ".hidden_target").mkdir()
            (root / "scratch" / "target_cache").mkdir(parents=True)
            result = tools.directory_search("*target*")

        self.assertTrue(result["ok"])
        self.assertIn("src/target_cache/", result["output"])
        self.assertNotIn(".hidden_target", result["output"])
        self.assertNotIn("scratch", result["output"])

    def test_everything_search_reports_missing_cli(self) -> None:
        with self._temp_tools() as (_root, tools):
            with patch.object(tools, "_everything_cli_path", return_value=None):
                result = tools.everything_search("README")

        self.assertFalse(result["ok"])
        self.assertIn("es.exe", result["summary"])

    def test_everything_search_runs_cli_and_filters_to_workspace(self) -> None:
        with self._temp_tools() as (root, tools):
            inside = root / "README.md"
            outside = root.parent / f"outside-{uuid4().hex}.txt"
            inside.write_text("hello", encoding="utf-8")
            outside.write_text("skip", encoding="utf-8")
            self.addCleanup(lambda: outside.unlink(missing_ok=True))
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
        with self._temp_tools() as (root, tools):
            fake_cli = root / "tools" / "es.exe"
            fake_cli.parent.mkdir()
            fake_cli.write_text("", encoding="utf-8")
            with patch.dict("os.environ", {"EVERYTHING_CLI": str(fake_cli)}):
                with patch("ollama_code.tools.shutil.which", return_value=None):
                    discovered = tools._everything_cli_path()

        self.assertEqual(discovered, str(fake_cli))

    def test_everything_search_filters_to_requested_subdirectory(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "src").mkdir()
            (root / "docs").mkdir()
            inside = root / "src" / "target.py"
            outside = root / "docs" / "target.py"
            inside.write_text("x\n", encoding="utf-8")
            outside.write_text("x\n", encoding="utf-8")
            completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=f"{inside}\n{outside}\n", stderr="")
            with patch.object(tools, "_everything_cli_path", return_value="es.exe"):
                with patch.object(tools, "_run_process", return_value=completed):
                    result = tools.everything_search("target.py", path="src", limit=10)

        self.assertTrue(result["ok"])
        self.assertEqual(result["output"], "src/target.py")

    def test_repo_index_refresh_writes_line_index_cache(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "ops.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
            result = tools.repo_index_refresh()
            payload = json.loads((root / ".ollama-code" / "index" / "repo_index.json").read_text(encoding="utf-8"))

        self.assertTrue(result["ok"])
        self.assertEqual(result["files"], 1)
        self.assertIn("line_index", payload["files"]["ops.py"])

    def test_semgrep_scan_reports_missing_semgrep(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "ops.py").write_text("eval('1')\n", encoding="utf-8")
            with patch("ollama_code.tools.resolve_tool_executable", return_value=None):
                with patch("ollama_code.tools.shutil.which", return_value=None):
                    result = tools.semgrep_scan("eval(...)")

        self.assertFalse(result["ok"])
        self.assertIn("semgrep/opengrep is not installed", result["summary"])
        self.assertEqual(result["compatible_tool_ids"], ["semgrep", "opengrep"])
        self.assertTrue(result["compatible_install_hints"]["opengrep"])

    def test_semgrep_scan_rejects_unsupported_language_before_execution(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "ops.py").write_text("eval('1')\n", encoding="utf-8")
            with patch("ollama_code.tools.shutil.which", return_value="semgrep"):
                with patch.object(tools, "_run_process") as run_process:
                    result = tools.semgrep_scan("eval(...)", lang="madeuplang")

        self.assertFalse(result["ok"])
        self.assertIn("Unsupported semgrep language", result["summary"])
        run_process.assert_not_called()

    def test_semgrep_scan_runs_structural_search_and_compacts_results(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "ops.py").write_text("eval('1')\n", encoding="utf-8")
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

    def test_semgrep_scan_uses_opengrep_when_semgrep_is_missing(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "ops.py").write_text("eval('1')\n", encoding="utf-8")
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

            def fake_resolve(tool_id: str, executable: str, **_kwargs: object) -> str | None:
                if tool_id == "opengrep" and executable == "opengrep":
                    return "opengrep"
                return None

            with patch("ollama_code.tools.resolve_tool_executable", side_effect=fake_resolve):
                with patch("ollama_code.tools.shutil.which", return_value=None):
                    with patch.object(tools, "_run_process", return_value=completed) as run_process:
                        result = tools.semgrep_scan("eval(...)", lang="python")

        self.assertTrue(result["ok"])
        self.assertIn("ops.py:1", result["output"])
        self.assertEqual(run_process.call_args.args[0][0], "opengrep")

    def test_semgrep_scan_can_run_through_remote_docker(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "ops.py").write_text("eval('1')\n", encoding="utf-8")
            payload = {
                "results": [
                    {
                        "path": "/src/target",
                        "start": {"line": 1},
                        "extra": {"lines": "eval('1')"},
                    }
                ]
            }
            responses = [
                subprocess.CompletedProcess(args=[], returncode=0, stdout="container123\n", stderr=""),
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
                subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(payload), stderr=""),
                subprocess.CompletedProcess(args=[], returncode=0, stdout="1\n", stderr=""),
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
            ]
            with patch.object(tools, "_semgrep_executable", return_value=None):
                with patch.object(tools, "_docker_command", return_value="docker"):
                    with patch.object(tools, "_run_process", side_effect=responses) as run_process:
                        with patch.dict(os.environ, {"OLLAMA_CODE_DOCKER_HOST": "ssh://car-detection-server"}):
                            result = tools.semgrep_scan("eval(...)", lang="python")

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["backend"], "docker")
        self.assertEqual(result["docker_host"], "ssh://car-detection-server")
        self.assertIn("/src/target:1", result["output"])
        create_call = run_process.call_args_list[0]
        copy_call = run_process.call_args_list[1]
        self.assertIn("create", create_call.args[0])
        self.assertIn("cp", copy_call.args[0])
        self.assertEqual(create_call.kwargs["env"]["DOCKER_HOST"], "ssh://car-detection-server")

    def test_docker_host_false_disables_docker_backed_semgrep(self) -> None:
        root = self._workspace_scratch()
        (root / "ops.py").write_text("eval('1')\n", encoding="utf-8")
        tools = ToolExecutor(root, approval_mode="auto")

        with patch.object(tools, "_semgrep_executable", return_value=None):
            with patch.object(tools, "_docker_command", return_value="docker"):
                with patch.dict(os.environ, {"OLLAMA_CODE_DOCKER_HOST": "false"}, clear=False):
                    result = tools.semgrep_scan("eval(...)", lang="python")

        self.assertFalse(result["ok"])
        self.assertEqual(result["missing_dependency"], "semgrep")
        self.assertIsNone(tools._docker_host())
        self.assertFalse(tools._docker_tools_enabled())

    def test_docker_env_scrubs_inherited_host_when_disabled(self) -> None:
        root = self._workspace_scratch()
        tools = ToolExecutor(root, approval_mode="auto")

        with patch.dict(os.environ, {"OLLAMA_CODE_DOCKER_HOST": "false", "DOCKER_HOST": "ssh://car-detection-server"}, clear=False):
            env = tools._docker_env()

        self.assertNotIn("DOCKER_HOST", env)

    def test_semgrep_scan_reports_cli_error_output(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "ops.py").write_text("eval('1')\n", encoding="utf-8")
            completed = subprocess.CompletedProcess(args=[], returncode=2, stdout="", stderr="semgrep parse error")
            with patch("ollama_code.tools.shutil.which", return_value="semgrep"):
                with patch.object(tools, "_run_process", return_value=completed):
                    result = tools.semgrep_scan("eval(...)", lang="python")

        self.assertFalse(result["ok"])
        self.assertIn("semgrep parse error", result["output"])

    def test_new_default_tools_are_registered(self) -> None:
        with self._temp_tools() as (_root, tools):
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
                "test_spec_extract",
                "verified_function_index",
                "verified_function_search",
                "verified_function_show",
                "verify_function_contract",
                "compose_verified_functions",
                "promote_verified_function",
                "browser_smoke",
                "security_scan",
                "mcp_list_tools",
                "mcp_call",
            }:
                self.assertIn(name, tools.available_tool_names())

    def test_todo_tools_store_validate_and_render_session_tasks(self) -> None:
        with self._temp_tools(approval_mode="read-only") as (_root, tools):
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
        with self._temp_tools() as (_root, tools):
            result = tools.execute(
                "todo_write",
                {"items": [{"content": "one", "status": "in_progress"}, {"content": "two", "status": "doing"}]},
            )

        self.assertFalse(result["ok"])
        self.assertIn("one todo item", result["summary"].lower())

    def test_disabled_tool_is_hidden_and_blocked(self) -> None:
        with self._temp_tools(disabled_tools=["browser_smoke"]) as (_root, tools):
            self.assertNotIn("browser_smoke", tools.available_tool_names())
            result = tools.execute("browser_smoke", {"url": "http://127.0.0.1"})

        self.assertFalse(result["ok"])
        self.assertIn("disabled", result["summary"])

    def test_enabled_tools_allowlist_hides_other_tools(self) -> None:
        with self._temp_tools(enabled_tools=["run_test", "read_file"]) as (_root, tools):
            self.assertEqual(tools.available_tool_names(), {"run_test", "read_file"})
            blocked = tools.execute("run_shell", {"command": "echo hi"})

        self.assertFalse(blocked["ok"])
        self.assertIn("disabled", blocked["summary"])

    def test_enabled_tools_allowlist_allows_dynamic_mcp_tool_names_via_mcp_call(self) -> None:
        with self._temp_tools(
            enabled_tools=["mcp_call"],
            mcp_servers={"demo": {"command": ["python", "-c", "pass"]}},
        ) as (_root, tools):
            with patch.object(tools, "_approve_shell", return_value=(True, "")):
                with patch.object(
                    tools,
                    "_mcp_request",
                    return_value=(True, [{"id": 2, "result": {"value": "ok"}}], ""),
                ) as request_mock:
                    result = tools.execute("mcp.demo.echo", {"text": "hi"})

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["tool"], "mcp_call")
        self.assertEqual(result["server"], "demo")
        self.assertEqual(result["mcp_tool"], "echo")
        self.assertIn('"value": "ok"', result["output"])
        self.assertEqual(request_mock.call_args.args[0], "demo")

    def test_systems_lens_frames_complex_task_without_grounding_edit(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "src").mkdir()

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
        with self._temp_tools() as (_root, tools):
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
        with self._temp_tools() as (_root, tools):
            result = tools.inspect_library_source(f"{missing_name}.helper")

        self.assertFalse(result["ok"])
        self.assertEqual(result["missing_dependency"], missing_name)
        self.assertEqual(result["error_class"], "missing_dependency")

    def test_python_sdk_search_finds_current_stdlib_api(self) -> None:
        with self._temp_tools() as (_root, tools):
            refresh = tools.python_sdk_refresh(limit=200)
            result = tools.python_sdk_search("parse json string loads", limit=5)

        self.assertTrue(refresh["ok"], refresh)
        self.assertTrue(result["ok"], result)
        self.assertIn("json.loads", result["output"])
        self.assertIn("Deserialize", result["output"])

    def test_python_sdk_refresh_skips_ast_scan_when_imported_entries_fill_limit(self) -> None:
        with self._temp_tools() as (_root, tools):
            entry = tools._python_sdk_entry(
                kind="function",
                module="json",
                qualname="json.loads",
                signature="loads(s)",
                doc="Deserialize JSON text.",
                source_path="(built-in)",
                line=1,
            )
            with patch.object(ToolExecutor, "_python_sdk_imported_entries", return_value=[entry]):
                with patch.object(ToolExecutor, "_python_sdk_ast_entries", side_effect=AssertionError("AST scan should be skipped")):
                    refresh = tools.python_sdk_refresh(limit=1)

        self.assertTrue(refresh["ok"], refresh)
        self.assertEqual(refresh["items"], 1)

    def test_python_sdk_search_can_rerank_with_cached_embeddings(self) -> None:
        entries = [
            {
                "id": "json",
                "kind": "function",
                "module": "json",
                "qualname": "json.dumps",
                "signature": "dumps(obj)",
                "doc": "Serialize an object to JSON text.",
                "source_path": "(built-in)",
                "line": 1,
                "text": "json dumps serialize object",
            },
            {
                "id": "pathlib",
                "kind": "method",
                "module": "pathlib",
                "qualname": "pathlib.Path.glob",
                "signature": "glob(pattern)",
                "doc": "Find files by wildcard pattern.",
                "source_path": "(built-in)",
                "line": 1,
                "text": "pathlib Path glob find files wildcard",
            },
        ]

        def fake_embed(texts: list[str], *, model: str, host: str | None = None, timeout: int = 120) -> list[list[float]]:
            vectors: list[list[float]] = []
            for text in texts:
                lowered = text.lower()
                vectors.append([0.0, 1.0] if "wildcard" in lowered or "files" in lowered else [1.0, 0.0])
            return vectors

        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            with patch.object(ToolExecutor, "_python_sdk_entries", return_value=entries):
                with patch.object(ToolExecutor, "_ollama_embed_texts", side_effect=fake_embed):
                    refresh = tools.python_sdk_refresh(limit=10, embedding_model="fake-embed")
                    result = tools.python_sdk_search("find files by wildcard", limit=2, use_embeddings=True, embedding_model="fake-embed")

        self.assertTrue(refresh["ok"], refresh)
        self.assertEqual(refresh["embedded"], 2)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["results"][0]["qualname"], "pathlib.Path.glob")
        self.assertIn("source=hybrid", result["output"])

    def test_python_sdk_search_reranks_candidates_with_on_demand_embeddings(self) -> None:
        entries = [
            {
                "id": "json",
                "kind": "function",
                "module": "json",
                "qualname": "json.dumps",
                "signature": "dumps(obj)",
                "doc": "Serialize an object to JSON text.",
                "source_path": "(built-in)",
                "line": 1,
                "text": "json dumps serialize object",
            },
            {
                "id": "pathlib",
                "kind": "method",
                "module": "pathlib",
                "qualname": "pathlib.Path.glob",
                "signature": "glob(pattern)",
                "doc": "Find files by wildcard pattern.",
                "source_path": "(built-in)",
                "line": 1,
                "text": "pathlib Path glob find files wildcard",
            },
        ]

        def fake_embed(texts: list[str], *, model: str, host: str | None = None, timeout: int = 120) -> list[list[float]]:
            vectors: list[list[float]] = []
            for text in texts:
                lowered = text.lower()
                vectors.append([0.0, 1.0] if "wildcard" in lowered or "files" in lowered else [1.0, 0.0])
            return vectors

        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            with patch.object(ToolExecutor, "_python_sdk_entries", return_value=entries):
                with patch.object(ToolExecutor, "_ollama_embed_texts", side_effect=fake_embed) as embed:
                    refresh = tools.python_sdk_refresh(limit=10)
                    result = tools.python_sdk_search("find files by wildcard", limit=2, use_embeddings=True, embedding_model="fake-embed")
                    refresh_again = tools.python_sdk_refresh(limit=10)
                    second = tools.python_sdk_search("find files by wildcard", limit=2, use_embeddings=True, embedding_model="fake-embed")

        self.assertTrue(refresh["ok"], refresh)
        self.assertEqual(refresh["embedded"], 0)
        self.assertEqual(refresh_again["cached_embeddings"], 1)
        self.assertTrue(result["ok"], result)
        self.assertIsNone(result["embedding_error"])
        self.assertEqual(result["results"][0]["qualname"], "pathlib.Path.glob")
        self.assertIn("source=hybrid", result["output"])
        self.assertTrue(second["ok"], second)
        self.assertLessEqual(embed.call_count, 4)

    def test_python_sdk_search_uses_env_embedding_model(self) -> None:
        entries = [
            {
                "id": "pathlib",
                "kind": "method",
                "module": "pathlib",
                "qualname": "pathlib.Path.glob",
                "signature": "glob(pattern)",
                "doc": "Find files by wildcard pattern.",
                "source_path": "(built-in)",
                "line": 1,
                "text": "pathlib Path glob find files wildcard",
            }
        ]

        def fake_embed(texts: list[str], *, model: str, host: str | None = None, timeout: int = 120) -> list[list[float]]:
            self.assertEqual(model, "fake-env-embed")
            return [[0.0, 1.0] for _ in texts]

        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            with patch.dict("os.environ", {"OLLAMA_CODE_SDK_EMBED_MODEL": "fake-env-embed"}):
                with patch.object(ToolExecutor, "_python_sdk_entries", return_value=entries):
                    with patch.object(ToolExecutor, "_ollama_embed_texts", side_effect=fake_embed) as embed:
                        result = tools.python_sdk_search("find files by wildcard", limit=1)

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["embedding_model"], "fake-env-embed")
        self.assertIsNone(result["embedding_error"])
        self.assertEqual(result["embedding_candidates"], 1)
        self.assertEqual(result["results"][0]["source"], "hybrid")
        self.assertEqual(embed.call_count, 2)

    def test_ast_search_reports_missing_ast_grep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ops.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            with patch("ollama_code.tools.shutil.which", return_value=None):
                result = tools.ast_search("def $F($$$A): $$$B", lang="python")

        self.assertFalse(result["ok"])
        self.assertEqual(result["missing_dependency"], "ast-grep")

    def test_ast_search_uses_native_python_function_fast_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ops.py").write_text(
                "def add(a, b):\n"
                "    return a + b\n\n"
                "async def fetch():\n"
                "    return 1\n",
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto")
            with patch.object(tools, "_ast_grep_executable", return_value="ast-grep"):
                with patch.object(tools, "_run_process", side_effect=AssertionError("ast-grep should not run")):
                    result = tools.ast_search("def $F($$$A): $$$B", "ops.py", lang="python")

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["backend"], "native-python")
        self.assertEqual(result["count"], 2)
        self.assertIn("ops.py:1: def add(a, b):", result["output"])
        self.assertIn("ops.py:4: async def fetch():", result["output"])

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

    def test_edit_intent_normalizes_signature_like_body_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "list_ops.py"
            sample.write_text("def append(list1, list2):\n    pass\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.edit_intent("list_ops.py", "replace_body", "append(list1, list2)", "return list1 + list2")

            final_text = sample.read_text(encoding="utf-8")

        self.assertTrue(result["ok"], result)
        self.assertIn("return list1 + list2", final_text)

    def test_edit_intent_normalizes_def_like_body_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "list_ops.py"
            sample.write_text("def append(list1, list2):\n    pass\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.edit_intent("list_ops.py", "replace_body", "def append(list1, list2):", "return list1 + list2")

            final_text = sample.read_text(encoding="utf-8")

        self.assertTrue(result["ok"], result)
        self.assertIn("return list1 + list2", final_text)

    def test_replace_body_full_matching_function_replaces_symbol_not_nested_def(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "pig_latin.py"
            sample.write_text("def translate(text):\n    return None\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.edit_intent(
                "pig_latin.py",
                "replace_body",
                "translate",
                "def translate(text):\n    return text + 'ay'\n",
            )

            final_text = sample.read_text(encoding="utf-8")

        self.assertTrue(result["ok"], result)
        self.assertIn("Routed full function replacement", result["summary"])
        self.assertEqual(final_text.count("def translate"), 1)
        self.assertIn("return text + 'ay'", final_text)

    def test_replace_body_routes_import_plus_matching_function_without_nested_def(self) -> None:
        with self._temp_tools() as (root, tools):
            sample = root / "wordy.py"
            sample.write_text("def answer(question):\n    return None\n", encoding="utf-8")
            result = tools.edit_intent(
                "wordy.py",
                "replace_body",
                "answer",
                "import re\n\n"
                "def answer(question):\n"
                "    match = re.search(r'-?\\d+', question)\n"
                "    return int(match.group(0)) if match else None\n",
            )

            final_text = sample.read_text(encoding="utf-8")

        self.assertTrue(result["ok"], result)
        self.assertEqual(final_text.count("def answer"), 1)
        self.assertIn("    import re\n", final_text)
        self.assertIn("return int(match.group(0))", final_text)

    def test_replace_body_full_mismatched_function_fails_closed(self) -> None:
        with self._temp_tools() as (root, tools):
            sample = root / "pig_latin.py"
            original = "def translate(text):\n    return None\n"
            sample.write_text(original, encoding="utf-8")
            result = tools.edit_intent(
                "pig_latin.py",
                "replace_body",
                "translate",
                "def encode(text):\n    return text + 'ay'\n",
            )

            final_text = sample.read_text(encoding="utf-8")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_class"], "invalid_args")
        self.assertEqual(final_text, original)

    def test_apply_structured_edit_replace_body_full_function_guard(self) -> None:
        with self._temp_tools() as (root, tools):
            sample = root / "app.py"
            original = "def add(left, right):\n    return None\n"
            sample.write_text(original, encoding="utf-8")
            ok = tools.apply_structured_edit(
                {"op": "replace_function_body", "path": "app.py", "symbol": "add", "body": "def add(left, right):\n    return left + right\n"}
            )
            bad = tools.apply_structured_edit(
                {"op": "replace_function_body", "path": "app.py", "symbol": "add", "body": "def subtract(left, right):\n    return left - right\n"}
            )

            final_text = sample.read_text(encoding="utf-8")

        self.assertTrue(ok["ok"], ok)
        self.assertEqual(ok["routed_tool"], "replace_symbol")
        self.assertFalse(bad["ok"])
        self.assertEqual(bad["error_class"], "invalid_args")
        self.assertEqual(final_text.count("def add"), 1)
        self.assertIn("return left + right", final_text)

    def test_edit_intent_repairs_common_join_typo_in_body_edit(self) -> None:
        with self._temp_tools() as (root, tools):
            sample = root / "joiner.py"
            sample.write_text("def join_items(items):\n    pass\n", encoding="utf-8")
            result = tools.edit_intent("joiner.py", "replace_body", "join_items", "return '.join(items)")

            final_text = sample.read_text(encoding="utf-8")

        self.assertTrue(result["ok"], result)
        self.assertIn('return " ".join(items)', final_text)

    def test_edit_intent_dedents_body_even_with_unindented_comments(self) -> None:
        with self._temp_tools() as (root, tools):
            sample = root / "ops.py"
            sample.write_text("def keep(values):\n    pass\n", encoding="utf-8")
            result = tools.edit_intent(
                "ops.py",
                "replace_body",
                "keep",
                "    return [value for value in values]\n\n# model note that should not break indentation",
            )

            final_text = sample.read_text(encoding="utf-8")

        self.assertTrue(result["ok"], result)
        self.assertIn("def keep(values):\n    return [value for value in values]\n\n    # model note", final_text)
        self.assertNotIn("\n        return [value", final_text)

    def test_edit_intent_rejects_calling_shadowed_builtin_parameter(self) -> None:
        with self._temp_tools() as (root, tools):
            sample = root / "ops.py"
            original = "def map_values(function, list):\n    pass\n"
            sample.write_text(original, encoding="utf-8")
            result = tools.edit_intent(
                "ops.py",
                "replace_body",
                "map_values",
                "return list(map(function, list))",
            )

            final_text = sample.read_text(encoding="utf-8")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_class"], "invalid_args")
        self.assertIn("shadows the Python builtin", result["summary"])
        self.assertEqual(final_text, original)

    def test_edit_intent_rejects_ignoring_initial_parameter(self) -> None:
        with self._temp_tools() as (root, tools):
            sample = root / "ops.py"
            original = "def foldr(function, list, initial):\n    pass\n"
            sample.write_text(original, encoding="utf-8")
            result = tools.edit_intent(
                "ops.py",
                "replace_body",
                "foldr",
                "result = None\nfor item in reversed(list):\n    result = function(item) if result is None else function(item, result)\nreturn result",
            )

            final_text = sample.read_text(encoding="utf-8")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_class"], "invalid_args")
        self.assertIn("initial", result["summary"])
        self.assertEqual(final_text, original)

    def test_edit_intent_rejects_reversed_foldr_reducer_arguments(self) -> None:
        with self._temp_tools() as (root, tools):
            sample = root / "ops.py"
            original = "def foldr(function, list, initial):\n    pass\n"
            sample.write_text(original, encoding="utf-8")
            result = tools.edit_intent(
                "ops.py",
                "replace_body",
                "foldr",
                "result = initial\nfor item in reversed(list):\n    result = function(item, result)\nreturn result",
            )

            final_text = sample.read_text(encoding="utf-8")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_class"], "invalid_args")
        self.assertIn("foldr reducer arguments look reversed", result["summary"])
        self.assertEqual(final_text, original)

    def test_replace_symbols_normalizes_reversed_recursive_foldr_reducer_arguments(self) -> None:
        with self._temp_tools() as (root, tools):
            sample = root / "ops.py"
            sample.write_text("def foldr(function, list, initial):\n    pass\n", encoding="utf-8")
            result = tools.replace_symbols(
                "ops.py",
                [
                    {
                        "symbol": "foldr",
                        "content": (
                            "def foldr(function, list, initial):\n"
                            "    if not list:\n"
                            "        return initial\n"
                            "    return function(list[0], foldr(function, list[1:], initial))\n"
                        ),
                    }
                ],
            )

            final_text = sample.read_text(encoding="utf-8")

        self.assertTrue(result["ok"], result)
        self.assertIn("Normalized foldr reducer order", result["summary"])
        self.assertIn("for item in reversed(list):", final_text)
        self.assertIn("accumulator = function(accumulator, item)", final_text)

    def test_replace_symbol_normalizes_helper_recursive_reversed_foldr_reducer_arguments(self) -> None:
        with self._temp_tools() as (root, tools):
            sample = root / "ops.py"
            sample.write_text("def foldr(function, list, initial):\n    pass\n", encoding="utf-8")
            result = tools.replace_symbol(
                "ops.py",
                "foldr",
                (
                    "def foldr(function, list, initial):\n"
                    "    def folder(item, rest):\n"
                    "        if not rest:\n"
                    "            return initial\n"
                    "        return function(item, folder(rest[0], rest[1:]))\n"
                    "    return folder(list[0], list[1:]) if list else initial\n"
                ),
            )

            final_text = sample.read_text(encoding="utf-8")

        self.assertTrue(result["ok"], result)
        self.assertIn("Normalized foldr reducer order", result["summary"])
        self.assertIn("for item in reversed(list):", final_text)
        self.assertIn("accumulator = function(accumulator, item)", final_text)

    def test_replace_symbol_rejects_accidental_signature_change(self) -> None:
        with self._temp_tools() as (root, tools):
            sample = root / "ops.py"
            original = "def foldl(function, list, initial):\n    pass\n"
            sample.write_text(original, encoding="utf-8")
            result = tools.replace_symbol(
                "ops.py",
                "foldl",
                (
                    "def foldl(function, values, initial, mode):\n"
                    "    accumulator = initial\n"
                    "    for item in values:\n"
                    "        accumulator = function(accumulator, item)\n"
                    "    return accumulator\n"
                ),
            )

            final_text = sample.read_text(encoding="utf-8")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_class"], "invalid_args")
        self.assertIn("changes signature", result["summary"])
        self.assertEqual(final_text, original)

    def test_edit_intent_routes_fix_like_full_function_to_symbol_replace(self) -> None:
        with self._temp_tools() as (root, tools):
            sample = root / "calculator.py"
            sample.write_text("def add(left: int, right: int) -> int:\n    return left - right\n", encoding="utf-8")
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

    def test_edit_intent_surfacing_syntax_errors_as_failed_tool(self) -> None:
        with self._temp_tools() as (root, tools):
            sample = root / "billing.py"
            sample.write_text('def invoice_label(user_id: int) -> str:\n    return f"{user_id}"\n', encoding="utf-8")
            result = tools.edit_intent(
                "billing.py",
                "replace_text",
                'def invoice_label(user_id: int) -> str:\n    return f"{user_id}"',
                'def invoice_label(user_id: int) -> str\n    return f"{user_id}"',
            )

            final_text = sample.read_text(encoding="utf-8")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_class"], "syntax_error")
        self.assertIn("Python syntax error", result["summary"])

    def test_edit_intent_routes_symbol_name_replacement_to_file_rename(self) -> None:
        with self._temp_tools() as (root, tools):
            sample = root / "pricing.py"
            sample.write_text("def total(prices):\n    return sum(prices)\n", encoding="utf-8")
            result = tools.edit_intent(
                "pricing.py",
                "replace_symbol",
                "total",
                "cart_total",
            )

            final_text = sample.read_text(encoding="utf-8")

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["routed_tool"], "apply_structured_edit")
        self.assertEqual(result["op"], "rename_symbol")
        self.assertIn("def cart_total(prices):", final_text)
        self.assertNotIn("def total(", final_text)

    def test_test_spec_extract_parses_unittest_examples_and_raises(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "wordy.py").write_text("def answer(question):\n    pass\n\n\ndef keep(function, values):\n    pass\n", encoding="utf-8")
            (root / "wordy_test.py").write_text(
                "import unittest\nfrom wordy import answer, keep as wordy_keep\n\n"
                "class WordyTest(unittest.TestCase):\n"
                "    def test_add(self):\n"
                "        self.assertEqual(answer('What is 1 plus 1?'), 2)\n"
                "    def test_alias(self):\n"
                "        self.assertEqual(wordy_keep(lambda value: value > 1, [1, 2]), [2])\n"
                "    def test_bad(self):\n"
                "        with self.assertRaises(ValueError):\n"
                "            answer('What is 52 cubed?')\n"
                "    def test_direct_raises(self):\n"
                "        self.assertRaises(ValueError, answer, 'What is?')\n",
                encoding="utf-8",
            )
            result = tools.test_spec_extract("wordy_test.py", source_path="wordy.py", limit=10)

        self.assertTrue(result["ok"], result)
        output = result["output"]
        self.assertIn("[add] answer('What is 1 plus 1?') -> 2", output)
        self.assertIn("answer('What is 1 plus 1?') -> 2", output)
        self.assertIn("keep(lambda value: value > 1, [1, 2]) -> [2]", output)
        self.assertIn("answer('What is 52 cubed?') raises ValueError", output)
        self.assertIn("answer('What is?') raises ValueError", output)
        self.assertEqual(result["examples"][0]["test_name"], "test_add")

    def test_test_spec_extract_resolves_local_expected_values_and_messages(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "transpose.py").write_text("def transpose(text):\n    pass\n", encoding="utf-8")
            (root / "transpose_test.py").write_text(
                "import unittest\nfrom transpose import transpose\n\n"
                "class TransposeTest(unittest.TestCase):\n"
                "    def test_row(self):\n"
                "        text = 'A1'\n"
                "        expected = 'A\\n1'\n"
                "        self.assertEqual(transpose(text), expected)\n"
                "    def test_bad(self):\n"
                "        with self.assertRaises(ValueError) as err:\n"
                "            transpose('bad')\n"
                "        self.assertEqual(err.exception.args[0], 'invalid input')\n",
                encoding="utf-8",
            )
            result = tools.test_spec_extract("transpose_test.py", source_path="transpose.py", limit=10)

        self.assertTrue(result["ok"], result)
        output = result["output"]
        self.assertIn("transpose('A1') -> 'A\\n1'", output)
        self.assertIn("transpose('bad') raises ValueError('invalid input')", output)

    def test_test_spec_extract_captures_stateful_object_scenarios(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "grade_school.py").write_text(
                "class School:\n"
                "    def add_student(self, name, grade):\n"
                "        pass\n"
                "    def roster(self):\n"
                "        pass\n",
                encoding="utf-8",
            )
            (root / "grade_school_test.py").write_text(
                "import unittest\nfrom grade_school import School\n\n"
                "class GradeSchoolTest(unittest.TestCase):\n"
                "    def test_student_is_added(self):\n"
                "        school = School()\n"
                "        school.add_student(name='Aimee', grade=2)\n"
                "        expected = ['Aimee']\n"
                "        self.assertEqual(school.roster(), expected)\n",
                encoding="utf-8",
            )
            result = tools.test_spec_extract("grade_school_test.py", source_path="grade_school.py", limit=10)

        self.assertTrue(result["ok"], result)
        self.assertIn("school = School(); school.add_student(name='Aimee', grade=2); school.roster() -> ['Aimee']", result["output"])

    def test_test_spec_extract_captures_assert_true_and_false_examples(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "retry.py").write_text("def should_retry(status_code, attempt):\n    pass\n", encoding="utf-8")
            (root / "retry_test.py").write_text(
                "import unittest\nfrom retry import should_retry\n\n"
                "class RetryPolicyTest(unittest.TestCase):\n"
                "    def test_retries_initial_attempts(self):\n"
                "        self.assertTrue(should_retry(503, 0))\n"
                "    def test_does_not_retry_non_server_error(self):\n"
                "        self.assertFalse(should_retry(404, 0))\n",
                encoding="utf-8",
            )
            result = tools.test_spec_extract("retry_test.py", source_path="retry.py", limit=10)

        self.assertTrue(result["ok"], result)
        self.assertIn("should_retry(503, 0) -> True", result["output"])
        self.assertIn("should_retry(404, 0) -> False", result["output"])

    def test_test_spec_extract_keeps_constructor_attribute_and_receiver_context(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "phone_number.py").write_text("class PhoneNumber:\n    def __init__(self, number):\n        pass\n", encoding="utf-8")
            (root / "phone_number_test.py").write_text(
                "import unittest\nfrom phone_number import PhoneNumber\n\n"
                "class PhoneNumberTest(unittest.TestCase):\n"
                "    def test_number_attribute(self):\n"
                "        number = PhoneNumber('(223) 456-7890').number\n"
                "        self.assertEqual(number, '2234567890')\n"
                "    def test_area_code_attribute(self):\n"
                "        number = PhoneNumber('2234567890')\n"
                "        self.assertEqual(number.area_code, '223')\n"
                "    def test_missing_method_required_by_tests(self):\n"
                "        number = PhoneNumber('2234567890')\n"
                "        self.assertEqual(number.pretty(), '(223)-456-7890')\n",
                encoding="utf-8",
            )
            (root / "scale_generator.py").write_text(
                "class Scale:\n"
                "    def __init__(self, tonic):\n"
                "        pass\n"
                "    def interval(self, intervals):\n"
                "        pass\n",
                encoding="utf-8",
            )
            (root / "scale_generator_test.py").write_text(
                "import unittest\nfrom scale_generator import Scale\n\n"
                "class ScaleTest(unittest.TestCase):\n"
                "    def test_major(self):\n"
                "        expected = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C']\n"
                "        self.assertEqual(Scale('C').interval('MMmMMMm'), expected)\n",
                encoding="utf-8",
            )
            phone = tools.test_spec_extract("phone_number_test.py", source_path="phone_number.py", limit=10)
            scale = tools.test_spec_extract("scale_generator_test.py", source_path="scale_generator.py", limit=10)

        self.assertTrue(phone["ok"], phone)
        self.assertIn("PhoneNumber('(223) 456-7890').number -> '2234567890'", phone["output"])
        self.assertIn("PhoneNumber('2234567890').area_code -> '223'", phone["output"])
        self.assertIn("number = PhoneNumber('2234567890'); number.pretty() -> '(223)-456-7890'", phone["output"])
        self.assertTrue(scale["ok"], scale)
        self.assertIn("Scale('C').interval('MMmMMMm') -> ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C']", scale["output"])

    def test_test_spec_extract_preserves_nested_receiver_and_repeated_mutation_history(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "simple_linked_list.py").write_text(
                "class LinkedList:\n"
                "    def __init__(self, values=None):\n"
                "        pass\n"
                "    def head(self):\n"
                "        pass\n"
                "    def pop(self):\n"
                "        pass\n",
                encoding="utf-8",
            )
            (root / "simple_linked_list_test.py").write_text(
                "import unittest\nfrom simple_linked_list import LinkedList\n\n"
                "class LinkedListTest(unittest.TestCase):\n"
                "    def test_nested_head_value(self):\n"
                "        sut = LinkedList([1])\n"
                "        self.assertEqual(sut.head().value(), 1)\n"
                "    def test_repeated_pop(self):\n"
                "        sut = LinkedList([1, 2])\n"
                "        sut.push(3)\n"
                "        self.assertEqual(len(sut), 3)\n"
                "        self.assertEqual(sut.pop(), 3)\n"
                "        self.assertEqual(sut.pop(), 2)\n"
                "        self.assertEqual(sut.pop(), 1)\n"
                "    def test_reversed_to_list(self):\n"
                "        sut = LinkedList([1, 2, 3])\n"
                "        self.assertEqual(list(sut.reversed()), [1, 2, 3])\n"
                "    def test_nested_none(self):\n"
                "        sut = LinkedList([1])\n"
                "        self.assertIsNone(sut.head().next())\n",
                encoding="utf-8",
            )
            result = tools.test_spec_extract("simple_linked_list_test.py", source_path="simple_linked_list.py", limit=10)

        self.assertTrue(result["ok"], result)
        self.assertIn("sut = LinkedList([1]); sut.head().value() -> 1", result["output"])
        self.assertIn("sut = LinkedList([1, 2]); sut.push(3); len(sut) -> 3", result["output"])
        self.assertIn("sut = LinkedList([1, 2]); sut.push(3); sut.pop() -> 3", result["output"])
        self.assertIn("sut = LinkedList([1, 2]); sut.push(3); sut.pop(); sut.pop() -> 2", result["output"])
        self.assertIn("sut = LinkedList([1, 2]); sut.push(3); sut.pop(); sut.pop(); sut.pop() -> 1", result["output"])
        self.assertIn("sut = LinkedList([1, 2, 3]); list(sut.reversed()) -> [1, 2, 3]", result["output"])
        self.assertNotIn("reversed(sut.reversed())", result["output"])
        self.assertIn("sut = LinkedList([1]); sut.head().next() -> None", result["output"])

    def test_test_spec_extract_captures_regex_not_equal_and_reset_history(self) -> None:
        with self._temp_tools() as (root, tools):
            (root / "robot_name.py").write_text(
                "class Robot:\n"
                "    def __init__(self):\n"
                "        self.name = None\n",
                encoding="utf-8",
            )
            (root / "robot_name_test.py").write_text(
                "import unittest\nfrom robot_name import Robot\n\n"
                "class RobotNameTest(unittest.TestCase):\n"
                "    name_re = r'^[A-Z]{2}\\d{3}$'\n"
                "    def test_has_name(self):\n"
                "        self.assertRegex(Robot().name, self.name_re)\n"
                "    def test_name_sticks(self):\n"
                "        robot = Robot()\n"
                "        robot.name\n"
                "        self.assertEqual(robot.name, robot.name)\n"
                "    def test_different_robots_have_different_names(self):\n"
                "        self.assertNotEqual(Robot().name, Robot().name)\n"
                "    def test_reset_name(self):\n"
                "        robot = Robot()\n"
                "        name = robot.name\n"
                "        robot.reset()\n"
                "        name2 = robot.name\n"
                "        self.assertNotEqual(name, name2)\n"
                "        self.assertRegex(name2, self.name_re)\n",
                encoding="utf-8",
            )
            result = tools.test_spec_extract("robot_name_test.py", source_path="robot_name.py", limit=20)

        self.assertTrue(result["ok"], result)
        output = result["output"]
        self.assertIn("Robot().name matches '^[A-Z]{2}\\\\d{3}$'", output)
        self.assertIn("robot = Robot(); robot.name -> robot.name", output)
        self.assertNotIn("Robot().name -> Robot().name", output)
        self.assertIn("Robot().name != Robot().name", output)
        self.assertIn("robot = Robot(); robot.reset(); robot.name matches '^[A-Z]{2}\\\\d{3}$'", output)
        self.assertIn("robot = Robot(); robot.name != robot = Robot(); robot.reset(); robot.name", output)

    def test_test_spec_extract_balances_examples_across_symbols(self) -> None:
        with self._temp_files_tools(
            {
                "list_ops.py": (
                    "def append(a, b):\n    pass\n"
                    "def foldr(function, values, initial):\n    pass\n"
                    "def reverse(values):\n    pass\n"
                ),
                "list_ops_test.py": (
                    "import unittest\nfrom list_ops import append, foldr, reverse\n\n"
                    "class ListOpsTest(unittest.TestCase):\n"
                    "    def test_append_one(self):\n"
                    "        self.assertEqual(append([], []), [])\n"
                    "    def test_append_two(self):\n"
                    "        self.assertEqual(append([1], [2]), [1, 2])\n"
                    "    def test_append_three(self):\n"
                    "        self.assertEqual(append([3], []), [3])\n"
                    "    def test_append_four(self):\n"
                    "        self.assertEqual(append([], [4]), [4])\n"
                    "    def test_foldr(self):\n"
                    "        self.assertEqual(foldr(lambda acc, el: el + acc, ['e'], '!'), 'e!')\n"
                    "    def test_reverse(self):\n"
                    "        self.assertEqual(reverse([1, 2]), [2, 1])\n"
                ),
            }
        ) as (_root, tools):
            result = tools.test_spec_extract("list_ops_test.py", source_path="list_ops.py", limit=4)

        self.assertTrue(result["ok"], result)
        self.assertIn("append:", result["output"])
        self.assertIn("foldr:", result["output"])
        self.assertIn("reverse:", result["output"])

    def test_implementation_spec_groups_signatures_stubs_examples_and_risks(self) -> None:
        with self._temp_files_tools(
            {
                "bag.py": (
                    "class Bag:\n"
                    "    def __init__(self):\n"
                    "        self.items = []\n"
                    "        self.size = 0\n"
                    "\n"
                    "    def size(self):\n"
                    "        pass\n"
                    "\n"
                    "def add(left, right):\n"
                    "    pass\n"
                ),
                "bag_test.py": (
                    "import unittest\nfrom bag import add, Bag\n\n"
                    "class BagTest(unittest.TestCase):\n"
                    "    def test_add(self):\n"
                    "        self.assertEqual(add(1, 2), 3)\n"
                    "    def test_bag_size(self):\n"
                    "        bag = Bag()\n"
                    "        self.assertEqual(bag.size(), 0)\n"
                ),
            }
        ) as (_root, tools):
            result = tools.implementation_spec("bag.py", "bag_test.py", limit=20)

        self.assertTrue(result["ok"], result)
        self.assertIn("Bag.size", result["output"])
        self.assertIn("add(1, 2) -> 3", result["output"])
        self.assertIn("self.size shadows method size()", result["output"])
        self.assertIn("bag.py::add", result["stubs"])

    def test_implementation_spec_does_not_hide_late_single_symbol_examples(self) -> None:
        tests = "import unittest\nfrom words import translate\n\nclass WordsTest(unittest.TestCase):\n"
        for index in range(12):
            tests += (
                f"    def test_case_{index}(self):\n"
                f"        self.assertEqual(translate('word{index}'), 'out{index}')\n"
            )
        with self._temp_files_tools(
            {
                "words.py": "def translate(text):\n    pass\n",
                "words_test.py": tests,
            }
        ) as (_root, tools):
            result = tools.implementation_spec("words.py", "words_test.py", limit=40)

        self.assertTrue(result["ok"], result)
        self.assertIn("translate('word0') -> 'out0'", result["output"])
        self.assertIn("translate('word11') -> 'out11'", result["output"])

    def test_implementation_spec_includes_string_transform_hints(self) -> None:
        with self._temp_files_tools(
            {
                "pig.py": "def translate(text):\n    pass\n",
                "pig_test.py": (
                    "import unittest\nfrom pig import translate\n\n"
                    "class PigTest(unittest.TestCase):\n"
                    "    def test_word_beginning_with_a_vowel(self):\n"
                    "        self.assertEqual(translate('apple'), 'appleay')\n"
                    "    def test_word_beginning_with_qu(self):\n"
                    "        self.assertEqual(translate('queen'), 'eenquay')\n"
                    "    def test_word_with_consonant_before_qu(self):\n"
                    "        self.assertEqual(translate('square'), 'aresquay')\n"
                ),
            }
        ) as (_root, tools):
            result = tools.implementation_spec("pig.py", "pig_test.py", limit=20)

        self.assertTrue(result["ok"], result)
        self.assertIn("observed string transforms:", result["output"])
        self.assertIn("'apple': append 'ay'", result["output"])
        self.assertIn("'queen': move prefix 'qu' to end, append 'ay'", result["output"])
        self.assertIn("'square': move prefix 'squ' to end, append 'ay'", result["output"])

    def test_synthesize_prefix_rotation_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "pig.py": "def translate(text):\n    pass\n",
                "pig_test.py": (
                    "import unittest\nfrom pig import translate\n\n"
                    "class PigTest(unittest.TestCase):\n"
                    "    def test_word_beginning_with_a_vowel(self):\n"
                    "        self.assertEqual(translate('apple'), 'appleay')\n"
                    "    def test_word_beginning_with_p(self):\n"
                    "        self.assertEqual(translate('pig'), 'igpay')\n"
                    "    def test_word_beginning_with_qu(self):\n"
                    "        self.assertEqual(translate('queen'), 'eenquay')\n"
                    "    def test_word_with_consonant_before_qu(self):\n"
                    "        self.assertEqual(translate('square'), 'aresquay')\n"
                    "    def test_word_beginning_with_xr(self):\n"
                    "        self.assertEqual(translate('xray'), 'xrayay')\n"
                    "    def test_y_after_consonant_cluster(self):\n"
                    "        self.assertEqual(translate('rhythm'), 'ythmrhay')\n"
                    "    def test_phrase(self):\n"
                    "        self.assertEqual(translate('quick fast run'), 'ickquay astfay unray')\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_prefix_rotation_candidate("pig.py", "pig_test.py")
            validation = tools.validate_implementation_candidate(
                "pig.py",
                str(synthesized.get("candidate_source") or ""),
                test_path="pig_test.py",
                test_command=command,
            )

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertTrue(validation["ok"], validation)
        self.assertIn("squ", synthesized["observed_prefixes"]["move"])

    def test_synthesize_word_arithmetic_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "wordy.py": "def answer(question):\n    pass\n",
                "wordy_test.py": (
                    "import unittest\nfrom wordy import answer\n\n"
                    "class WordyTest(unittest.TestCase):\n"
                    "    def test_just_a_number(self):\n"
                    "        self.assertEqual(answer('What is 5?'), 5)\n"
                    "    def test_addition(self):\n"
                    "        self.assertEqual(answer('What is 1 plus 1?'), 2)\n"
                    "    def test_subtraction(self):\n"
                    "        self.assertEqual(answer('What is 4 minus -12?'), 16)\n"
                    "    def test_multiplication(self):\n"
                    "        self.assertEqual(answer('What is -3 multiplied by 25?'), -75)\n"
                    "    def test_division(self):\n"
                    "        self.assertEqual(answer('What is 33 divided by -3?'), -11)\n"
                    "    def test_multiple_operations(self):\n"
                    "        self.assertEqual(answer('What is 17 minus 6 plus 3?'), 14)\n"
                    "    def test_unknown_operation(self):\n"
                    "        with self.assertRaises(ValueError) as err:\n"
                    "            answer('What is 52 cubed?')\n"
                    "        self.assertEqual(err.exception.args[0], 'unknown operation')\n"
                    "    def test_syntax_error(self):\n"
                    "        with self.assertRaises(ValueError) as err:\n"
                    "            answer('What is 1 plus?')\n"
                    "        self.assertEqual(err.exception.args[0], 'syntax error')\n"
                    "    def test_empty_question(self):\n"
                    "        with self.assertRaises(ValueError) as err:\n"
                    "            answer('What is?')\n"
                    "        self.assertEqual(err.exception.args[0], 'syntax error')\n"
                    "    def test_non_math_question(self):\n"
                    "        with self.assertRaises(ValueError) as err:\n"
                    "            answer('Who is the President of the United States?')\n"
                    "        self.assertEqual(err.exception.args[0], 'unknown operation')\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_word_arithmetic_candidate("wordy.py", "wordy_test.py")
            validation = tools.validate_implementation_candidate(
                "wordy.py",
                str(synthesized.get("candidate_source") or ""),
                test_path="wordy_test.py",
                test_command=command,
            )

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertTrue(validation["ok"], validation)
        self.assertTrue(synthesized["operations"]["multiplied by"])
        self.assertIn("unknown operation", synthesized["candidate_source"])

    def test_synthesize_simple_expression_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "calc.py": "def add(left: int, right: int) -> int:\n    return left - right\n",
                "calc_test.py": (
                    "import unittest\nfrom calc import add\n\n"
                    "class CalcTest(unittest.TestCase):\n"
                    "    def test_adds_values(self):\n"
                    "        self.assertEqual(add(2, 3), 5)\n"
                    "    def test_adds_negative_values(self):\n"
                    "        self.assertEqual(add(-2, 5), 3)\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_simple_expression_candidate("calc.py", "calc_test.py")
            validation = tools.validate_implementation_candidate(
                "calc.py",
                str(synthesized.get("candidate_source") or ""),
                test_path="calc_test.py",
                test_command=command,
            )

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertEqual(synthesized["expression"], "left + right")
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_simple_expression_candidate_handles_boolean_ranges(self) -> None:
        with self._temp_python_tools(
            {
                "retry.py": "def should_retry(status_code: int, attempt: int) -> bool:\n    return status_code >= 500 and attempt > 2\n",
                "client.py": (
                    "from retry import should_retry\n\n"
                    "def next_action(status_code: int, attempt: int) -> str:\n"
                    "    return 'retry' if should_retry(status_code, attempt) else 'stop'\n"
                ),
                "tests/test_retry_policy.py": (
                    "import unittest\nfrom retry import should_retry\nfrom client import next_action\n\n"
                    "class RetryPolicyTests(unittest.TestCase):\n"
                    "    def test_retries_initial_attempts(self):\n"
                    "        self.assertTrue(should_retry(503, 0))\n"
                    "    def test_does_not_retry_non_server_error(self):\n"
                    "        self.assertFalse(should_retry(404, 0))\n"
                    "    def test_retries_upper_server_range(self):\n"
                    "        self.assertTrue(should_retry(599, 2))\n"
                    "    def test_does_not_retry_client_error_boundary(self):\n"
                    "        self.assertFalse(should_retry(499, 0))\n"
                    "    def test_client_retry_action(self):\n"
                    "        self.assertEqual(next_action(503, 1), 'retry')\n"
                    "    def test_client_stop_after_budget(self):\n"
                    "        self.assertEqual(next_action(503, 3), 'stop')\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_simple_expression_candidate("retry.py", "tests/test_retry_policy.py")
            validation = tools.validate_implementation_candidate(
                "retry.py",
                str(synthesized.get("candidate_source") or ""),
                test_path="tests/test_retry_policy.py",
                test_command=command,
            )

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertTrue(validation["ok"], validation)
        self.assertIn("attempt", synthesized["expression"])
        self.assertIn("status_code", synthesized["expression"])

    def test_synthesize_string_normalizer_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "slug.py": "def slugify(value: str) -> str:\n    pass\n",
                "slug_test.py": (
                    "import unittest\nfrom slug import slugify\n\n"
                    "class SlugTest(unittest.TestCase):\n"
                    "    def test_spaces(self):\n"
                    "        self.assertEqual(slugify('Hello Local Model'), 'hello-local-model')\n"
                    "    def test_edges(self):\n"
                    "        self.assertEqual(slugify('  Mixed Case  '), 'mixed-case')\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_string_normalizer_candidate("slug.py", "slug_test.py")
            validation = tools.validate_implementation_candidate(
                "slug.py",
                str(synthesized.get("candidate_source") or ""),
                test_path="slug_test.py",
                test_command=command,
            )

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertIn("split()", synthesized["expression"])
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_sequence_utilities_candidate_from_standard_signatures(self) -> None:
        with self._temp_python_tools(
            {
                "seq.py": (
                    "def append(left, right):\n    pass\n\n"
                    "def concat(groups):\n    pass\n\n"
                    "def filter(function, values):\n    pass\n\n"
                    "def length(values):\n    pass\n\n"
                    "def map(function, values):\n    pass\n\n"
                    "def foldl(function, values, initial):\n    pass\n\n"
                    "def foldr(function, values, initial):\n    pass\n\n"
                    "def reverse(values):\n    pass\n"
                ),
                "seq_test.py": (
                    "import unittest\nfrom seq import append, concat, filter, foldl, foldr, length, map, reverse\n\n"
                    "class SequenceTest(unittest.TestCase):\n"
                    "    def test_append(self):\n"
                    "        self.assertEqual(append([1], [2]), [1, 2])\n"
                    "    def test_concat(self):\n"
                    "        self.assertEqual(concat([[1], [2, 3]]), [1, 2, 3])\n"
                    "    def test_filter(self):\n"
                    "        self.assertEqual(filter(lambda item: item % 2 == 1, [1, 2, 3]), [1, 3])\n"
                    "    def test_length(self):\n"
                    "        self.assertEqual(length(['a', 'b']), 2)\n"
                    "    def test_map(self):\n"
                    "        self.assertEqual(map(lambda item: item + 1, [1, 2]), [2, 3])\n"
                    "    def test_foldl(self):\n"
                    "        self.assertEqual(foldl(lambda acc, item: acc - item, [1, 2], 10), 7)\n"
                    "    def test_foldr(self):\n"
                    "        self.assertEqual(foldr(lambda acc, item: item + acc, ['e', 'x'], '!'), 'ex!')\n"
                    "    def test_reverse(self):\n"
                    "        self.assertEqual(reverse([1, 2, 3]), [3, 2, 1])\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_sequence_utilities_candidate("seq.py", "seq_test.py")
            validation = tools.validate_implementation_candidate(
                "seq.py",
                str(synthesized.get("candidate_source") or ""),
                test_path="seq_test.py",
                test_command=command,
            )

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertIn("def foldr(function, values, initial):", synthesized["candidate_source"])
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_affine_substitution_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "affine_cipher.py": "def encode(plain_text, a, b):\n    pass\n\n\ndef decode(ciphered_text, a, b):\n    pass\n",
                "affine_cipher_test.py": (
                    "import unittest\nfrom affine_cipher import decode, encode\n\n"
                    "class AffineCipherTest(unittest.TestCase):\n"
                    "    def test_encode_yes(self):\n"
                    "        self.assertEqual(encode('yes', 5, 7), 'xbt')\n"
                    "    def test_encode_omg(self):\n"
                    "        self.assertEqual(encode('OMG', 21, 3), 'lvz')\n"
                    "    def test_encode_numbers_and_groups(self):\n"
                    "        self.assertEqual(encode('Testing,1 2 3, testing.', 3, 4), 'jqgjc rw123 jqgjc rw')\n"
                    "    def test_encode_with_a_not_coprime_to_m(self):\n"
                    "        with self.assertRaises(ValueError) as err:\n"
                    "            encode('This is a test.', 6, 17)\n"
                    "        self.assertEqual(err.exception.args[0], 'a and m must be coprime.')\n"
                    "    def test_decode_exercism(self):\n"
                    "        self.assertEqual(decode('tytgn fjr', 3, 7), 'exercism')\n"
                    "    def test_decode_numbers(self):\n"
                    "        self.assertEqual(decode('odpoz ub123 odpoz ub', 25, 7), 'testing123testing')\n"
                    "    def test_decode_with_a_not_coprime_to_m(self):\n"
                    "        with self.assertRaises(ValueError) as err:\n"
                    "            decode('Test', 13, 5)\n"
                    "        self.assertEqual(err.exception.args[0], 'a and m must be coprime.')\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_affine_substitution_candidate("affine_cipher.py", "affine_cipher_test.py")
            validation = tools.validate_implementation_candidate(
                "affine_cipher.py",
                str(synthesized.get("candidate_source") or ""),
                test_path="affine_cipher_test.py",
                test_command=command,
            )

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertIn("ALPHABET_SIZE = 26", synthesized["candidate_source"])
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_countdown_song_candidate_for_beer_song(self) -> None:
        with self._temp_python_tools(
            {
                "beer_song.py": "def recite(start, take=1):\n    pass\n",
                "beer_song_test.py": (
                    "import unittest\nfrom beer_song import recite\n\n"
                    "class BeerSongTest(unittest.TestCase):\n"
                    "    def test_first_generic_verse(self):\n"
                    "        expected = ['99 bottles of beer on the wall, 99 bottles of beer.', 'Take one down and pass it around, 98 bottles of beer on the wall.']\n"
                    "        self.assertEqual(recite(start=99), expected)\n"
                    "    def test_verse_with_1_bottle(self):\n"
                    "        expected = ['1 bottle of beer on the wall, 1 bottle of beer.', 'Take it down and pass it around, no more bottles of beer on the wall.']\n"
                    "        self.assertEqual(recite(start=1), expected)\n"
                    "    def test_verse_with_0_bottles(self):\n"
                    "        expected = ['No more bottles of beer on the wall, no more bottles of beer.', 'Go to the store and buy some more, 99 bottles of beer on the wall.']\n"
                    "        self.assertEqual(recite(start=0), expected)\n"
                    "    def test_last_three_verses(self):\n"
                    "        expected = ['2 bottles of beer on the wall, 2 bottles of beer.', 'Take one down and pass it around, 1 bottle of beer on the wall.', '', '1 bottle of beer on the wall, 1 bottle of beer.', 'Take it down and pass it around, no more bottles of beer on the wall.', '', 'No more bottles of beer on the wall, no more bottles of beer.', 'Go to the store and buy some more, 99 bottles of beer on the wall.']\n"
                    "        self.assertEqual(recite(start=2, take=3), expected)\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_countdown_song_candidate("beer_song.py", "beer_song_test.py")
            validation = tools.validate_implementation_candidate(
                "beer_song.py",
                str(synthesized.get("candidate_source") or ""),
                test_path="beer_song_test.py",
                test_command=command,
            )

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertEqual(synthesized["variant"], "beer")
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_countdown_song_candidate_for_green_bottles(self) -> None:
        with self._temp_python_tools(
            {
                "bottle_song.py": "def recite(start, take=1):\n    pass\n",
                "bottle_song_test.py": (
                    "import unittest\nfrom bottle_song import recite\n\n"
                    "class BottleSongTest(unittest.TestCase):\n"
                    "    def test_first_generic_verse(self):\n"
                    "        expected = ['Ten green bottles hanging on the wall,', 'Ten green bottles hanging on the wall,', 'And if one green bottle should accidentally fall,', \"There'll be nine green bottles hanging on the wall.\"]\n"
                    "        self.assertEqual(recite(start=10), expected)\n"
                    "    def test_last_three_verses(self):\n"
                    "        expected = ['Three green bottles hanging on the wall,', 'Three green bottles hanging on the wall,', 'And if one green bottle should accidentally fall,', \"There'll be two green bottles hanging on the wall.\", '', 'Two green bottles hanging on the wall,', 'Two green bottles hanging on the wall,', 'And if one green bottle should accidentally fall,', \"There'll be one green bottle hanging on the wall.\", '', 'One green bottle hanging on the wall,', 'One green bottle hanging on the wall,', 'And if one green bottle should accidentally fall,', \"There'll be no green bottles hanging on the wall.\"]\n"
                    "        self.assertEqual(recite(start=3, take=3), expected)\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_countdown_song_candidate("bottle_song.py", "bottle_song_test.py")
            validation = tools.validate_implementation_candidate(
                "bottle_song.py",
                str(synthesized.get("candidate_source") or ""),
                test_path="bottle_song_test.py",
                test_command=command,
            )

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertEqual(synthesized["variant"], "green-bottle")
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_discounted_set_pricing_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "book_store.py": "def total(basket):\n    pass\n",
                "book_store_test.py": (
                    "import unittest\nfrom book_store import total\n\n"
                    "class BookStoreTest(unittest.TestCase):\n"
                    "    def test_only_a_single_book(self):\n"
                    "        basket = [1]\n"
                    "        self.assertEqual(total(basket), 800)\n"
                    "    def test_two_different_books(self):\n"
                    "        basket = [1, 2]\n"
                    "        self.assertEqual(total(basket), 1520)\n"
                    "    def test_three_different_books(self):\n"
                    "        basket = [1, 2, 3]\n"
                    "        self.assertEqual(total(basket), 2160)\n"
                    "    def test_four_different_books(self):\n"
                    "        basket = [1, 2, 3, 4]\n"
                    "        self.assertEqual(total(basket), 2560)\n"
                    "    def test_five_different_books(self):\n"
                    "        basket = [1, 2, 3, 4, 5]\n"
                    "        self.assertEqual(total(basket), 3000)\n"
                    "    def test_two_groups_of_four_is_cheaper_than_group_of_five_plus_group_of_three(self):\n"
                    "        basket = [1, 1, 2, 2, 3, 3, 4, 5]\n"
                    "        self.assertEqual(total(basket), 5120)\n"
                    "    def test_complex_grouping(self):\n"
                    "        basket = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]\n"
                    "        self.assertEqual(total(basket), 10000)\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_discounted_set_pricing_candidate("book_store.py", "book_store_test.py")
            validation = tools.validate_implementation_candidate(
                "book_store.py",
                str(synthesized.get("candidate_source") or ""),
                test_path="book_store_test.py",
                test_command=command,
            )

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertIn("best_price", synthesized["candidate_source"])
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_bowling_game_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "bowling.py": (
                    "class BowlingGame:\n"
                    "    def __init__(self):\n"
                    "        pass\n\n"
                    "    def roll(self, pins):\n"
                    "        pass\n\n"
                    "    def score(self):\n"
                    "        pass\n"
                ),
                "bowling_test.py": (
                    "import unittest\nfrom bowling import BowlingGame\n\n"
                    "class BowlingTest(unittest.TestCase):\n"
                    "    def roll_new_game(self, rolls):\n"
                    "        game = BowlingGame()\n"
                    "        for roll in rolls:\n"
                    "            game.roll(roll)\n"
                    "        return game\n"
                    "    def assertRaisesWithMessage(self, exception):\n"
                    "        return self.assertRaisesRegex(exception, r'.+')\n"
                    "    def test_all_zeros(self):\n"
                    "        self.assertEqual(self.roll_new_game([0] * 20).score(), 0)\n"
                    "    def test_spare_bonus(self):\n"
                    "        self.assertEqual(self.roll_new_game([6, 4, 3, 0] + [0] * 16).score(), 16)\n"
                    "    def test_strike_bonus(self):\n"
                    "        self.assertEqual(self.roll_new_game([10, 5, 3] + [0] * 16).score(), 26)\n"
                    "    def test_perfect_game(self):\n"
                    "        self.assertEqual(self.roll_new_game([10] * 12).score(), 300)\n"
                    "    def test_incomplete_game(self):\n"
                    "        with self.assertRaisesWithMessage(Exception):\n"
                    "            self.roll_new_game([0, 0]).score()\n"
                    "    def test_invalid_frame(self):\n"
                    "        game = self.roll_new_game([5])\n"
                    "        with self.assertRaisesWithMessage(Exception):\n"
                    "            game.roll(6)\n"
                    "    def test_invalid_last_strike_bonus(self):\n"
                    "        game = self.roll_new_game([0] * 18 + [10, 5])\n"
                    "        with self.assertRaisesWithMessage(Exception):\n"
                    "            game.roll(6)\n"
                    "    def test_cannot_roll_after_complete_game(self):\n"
                    "        game = self.roll_new_game([0] * 20)\n"
                    "        with self.assertRaisesWithMessage(Exception):\n"
                    "            game.roll(0)\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_bowling_game_candidate("bowling.py", "bowling_test.py")
            validation = tools.validate_implementation_candidate(
                "bowling.py",
                str(synthesized.get("candidate_source") or ""),
                test_path="bowling_test.py",
                test_command=command,
            )

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertIn("_validate_rolls", synthesized["candidate_source"])
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_noarg_literal_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "answers.py": "def drinks_water():\n    pass\n\ndef owns_zebra():\n    pass\n",
                "answers_test.py": (
                    "import unittest\nfrom answers import drinks_water, owns_zebra\n\n"
                    "class AnswerTest(unittest.TestCase):\n"
                    "    def test_water(self):\n"
                    "        self.assertEqual(drinks_water(), 'Norwegian')\n"
                    "    def test_zebra(self):\n"
                    "        self.assertEqual(owns_zebra(), 'Japanese')\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_noarg_literal_candidate("answers.py", "answers_test.py")
            validation = tools.validate_implementation_candidate("answers.py", str(synthesized.get("candidate_source") or ""), test_path="answers_test.py", test_command=command)

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertIn("return 'Norwegian'", synthesized["candidate_source"])
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_proverb_chain_candidate_allows_test_backed_signature_change(self) -> None:
        with self._temp_python_tools(
            {
                "proverb.py": "def proverb():\n    pass\n",
                "proverb_test.py": (
                    "import unittest\nfrom proverb import proverb\n\n"
                    "class ProverbTest(unittest.TestCase):\n"
                    "    def test_three(self):\n"
                    "        self.assertEqual(proverb('nail', 'shoe', 'horse', qualifier=None), [\n"
                    "            'For want of a nail the shoe was lost.',\n"
                    "            'For want of a shoe the horse was lost.',\n"
                    "            'And all for the want of a nail.',\n"
                    "        ])\n"
                    "    def test_qualifier(self):\n"
                    "        self.assertEqual(proverb('nail', qualifier='horseshoe'), ['And all for the want of a horseshoe nail.'])\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_proverb_chain_candidate("proverb.py", "proverb_test.py")
            validation = tools.validate_implementation_candidate("proverb.py", str(synthesized.get("candidate_source") or ""), test_path="proverb_test.py", test_command=command)

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertIn("candidate changed signature", "\n".join(validation.get("signature_warnings") or []))
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_typed_graph_dsl_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "dot_dsl.py": (
                    "NODE, EDGE, ATTR = range(3)\n\n"
                    "class Node:\n"
                    "    def __init__(self, name, attrs):\n"
                    "        self.name = name\n"
                    "        self.attrs = attrs\n"
                    "    def __eq__(self, other):\n"
                    "        return self.name == other.name and self.attrs == other.attrs\n\n"
                    "class Edge:\n"
                    "    def __init__(self, src, dst, attrs):\n"
                    "        self.src = src\n"
                    "        self.dst = dst\n"
                    "        self.attrs = attrs\n"
                    "    def __eq__(self, other):\n"
                    "        return self.src == other.src and self.dst == other.dst and self.attrs == other.attrs\n\n"
                    "class Graph:\n"
                    "    def __init__(self, data=None):\n"
                    "        pass\n"
                ),
                "dot_dsl_test.py": (
                    "import unittest\nfrom dot_dsl import Graph, Node, Edge, NODE, EDGE, ATTR\n\n"
                    "class GraphTest(unittest.TestCase):\n"
                    "    def test_items(self):\n"
                    "        g = Graph([(ATTR, 'title', 'T'), (NODE, 'a', {}), (EDGE, 'a', 'b', {'x': '1'})])\n"
                    "        self.assertEqual(g.attrs, {'title': 'T'})\n"
                    "        self.assertEqual(g.nodes, [Node('a', {})])\n"
                    "        self.assertEqual(g.edges, [Edge('a', 'b', {'x': '1'})])\n"
                    "    def test_malformed(self):\n"
                    "        with self.assertRaisesRegex(TypeError, 'Graph data malformed'):\n"
                    "            Graph(1)\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_typed_graph_dsl_candidate("dot_dsl.py", "dot_dsl_test.py")
            validation = tools.validate_implementation_candidate("dot_dsl.py", str(synthesized.get("candidate_source") or ""), test_path="dot_dsl_test.py", test_command=command)

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertIn("Graph data malformed", synthesized["candidate_source"])
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_parent_record_tree_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "tree_building.py": (
                    "class Record:\n"
                    "    def __init__(self, record_id, parent_id):\n"
                    "        self.record_id = record_id\n"
                    "        self.parent_id = parent_id\n\n"
                    "class Node:\n"
                    "    def __init__(self, node_id):\n"
                    "        self.node_id = node_id\n"
                    "        self.children = []\n\n"
                    "def BuildTree(records):\n"
                    "    return None\n"
                ),
                "tree_building_test.py": (
                    "import unittest\nfrom tree_building import Record, BuildTree\n\n"
                    "class TreeTest(unittest.TestCase):\n"
                    "    def test_tree(self):\n"
                    "        root = BuildTree([Record(2, 0), Record(1, 0), Record(0, 0)])\n"
                    "        self.assertEqual(root.node_id, 0)\n"
                    "        self.assertEqual([child.node_id for child in root.children], [1, 2])\n"
                    "    def test_invalid(self):\n"
                    "        with self.assertRaisesRegex(ValueError, 'Record id is invalid or out of order'):\n"
                    "            BuildTree([Record(1, 0)])\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_parent_record_tree_candidate("tree_building.py", "tree_building_test.py")
            validation = tools.validate_implementation_candidate("tree_building.py", str(synthesized.get("candidate_source") or ""), test_path="tree_building_test.py", test_command=command)

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertIn("Record id is invalid or out of order", synthesized["candidate_source"])
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_domino_chain_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "dominoes.py": "def can_chain(dominoes):\n    pass\n",
                "dominoes_test.py": (
                    "import unittest\nfrom dominoes import can_chain\n\n"
                    "class DominoTest(unittest.TestCase):\n"
                    "    def assert_correct_chain(self, input_dominoes, output_chain):\n"
                    "        self.assertIsNotNone(output_chain)\n"
                    "        self.assertEqual(sorted(tuple(sorted(d)) for d in input_dominoes), sorted(tuple(sorted(d)) for d in output_chain))\n"
                    "        if output_chain:\n"
                    "            self.assertEqual(output_chain[0][0], output_chain[-1][1])\n"
                    "            for left, right in zip(output_chain, output_chain[1:]):\n"
                    "                self.assertEqual(left[1], right[0])\n"
                    "    def refute_correct_chain(self, input_dominoes, output_chain):\n"
                    "        self.assertIsNone(output_chain)\n"
                    "    def test_chain(self):\n"
                    "        self.assert_correct_chain([(1, 2), (3, 1), (2, 3)], can_chain([(1, 2), (3, 1), (2, 3)]))\n"
                    "    def test_no_chain(self):\n"
                    "        self.refute_correct_chain([(1, 2)], can_chain([(1, 2)]))\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_domino_chain_candidate("dominoes.py", "dominoes_test.py")
            validation = tools.validate_implementation_candidate("dominoes.py", str(synthesized.get("candidate_source") or ""), test_path="dominoes_test.py", test_command=command)

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_food_chain_song_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "food_chain.py": "def recite(start, end):\n    pass\n",
                "food_chain_test.py": (
                    "import unittest\nfrom food_chain import recite\n\n"
                    "class FoodChainTest(unittest.TestCase):\n"
                    "    def test_fly(self):\n"
                    "        self.assertEqual(recite(1, 1), ['I know an old lady who swallowed a fly.', \"I don't know why she swallowed the fly. Perhaps she'll die.\"])\n"
                    "    def test_horse(self):\n"
                    "        self.assertEqual(recite(8, 8), ['I know an old lady who swallowed a horse.', \"She's dead, of course!\"])\n"
                    "    def test_range(self):\n"
                    "        self.assertIn('', recite(1, 3))\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_food_chain_song_candidate("food_chain.py", "food_chain_test.py")
            validation = tools.validate_implementation_candidate("food_chain.py", str(synthesized.get("candidate_source") or ""), test_path="food_chain_test.py", test_command=command)

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_grep_filter_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "grep.py": "def grep(pattern, flags, files):\n    pass\n",
                "grep_test.py": (
                    "import io\nimport unittest\nfrom unittest import mock\nfrom grep import grep\n\n"
                    "DATA = {'a.txt': 'Alpha\\nbeta\\n', 'b.txt': 'alpha\\n'}\n"
                    "def open_mock(name, *args, **kwargs):\n"
                    "    return io.StringIO(DATA[name])\n"
                    "@mock.patch('grep.open', side_effect=open_mock, create=True)\n"
                    "class GrepTest(unittest.TestCase):\n"
                    "    def test_flags(self, _open):\n"
                    "        self.assertMultiLineEqual(grep('ALPHA', '-i -n', ['a.txt']), '1:Alpha\\n')\n"
                    "        self.assertMultiLineEqual(grep('alpha', '-l', ['a.txt', 'b.txt']), 'b.txt\\n')\n"
                    "        self.assertMultiLineEqual(grep('beta', '-v -x', ['a.txt']), 'Alpha\\n')\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_grep_filter_candidate("grep.py", "grep_test.py")
            validation = tools.validate_implementation_candidate("grep.py", str(synthesized.get("candidate_source") or ""), test_path="grep_test.py", test_command=command)

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_bucket_measure_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "two_bucket.py": "def measure(bucket_one, bucket_two, goal, start_bucket):\n    pass\n",
                "two_bucket_test.py": (
                    "import unittest\nfrom two_bucket import measure\n\n"
                    "class BucketTest(unittest.TestCase):\n"
                    "    def assertRaisesWithMessage(self, exception):\n"
                    "        return self.assertRaisesRegex(exception, r'.+')\n"
                    "    def test_one(self):\n"
                    "        self.assertEqual(measure(3, 5, 1, 'one'), (4, 'one', 5))\n"
                    "    def test_start_two(self):\n"
                    "        self.assertEqual(measure(3, 5, 1, 'two'), (8, 'two', 3))\n"
                    "    def test_two(self):\n"
                    "        self.assertEqual(measure(1, 3, 3, 'two'), (1, 'two', 0))\n"
                    "    def test_goal_is_other_capacity(self):\n"
                    "        self.assertEqual(measure(2, 3, 3, 'one'), (2, 'two', 2))\n"
                    "    def test_impossible(self):\n"
                    "        with self.assertRaisesWithMessage(ValueError):\n"
                    "            measure(6, 15, 5, 'one')\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_bucket_measure_candidate("two_bucket.py", "two_bucket_test.py")
            validation = tools.validate_implementation_candidate("two_bucket.py", str(synthesized.get("candidate_source") or ""), test_path="two_bucket_test.py", test_command=command)

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_reactive_cells_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "react.py": (
                    "class InputCell:\n"
                    "    def __init__(self, initial_value):\n"
                    "        self.value = None\n\n"
                    "class ComputeCell:\n"
                    "    def __init__(self, inputs, compute_function):\n"
                    "        self.value = None\n"
                    "    def add_callback(self, callback):\n"
                    "        pass\n"
                    "    def remove_callback(self, callback):\n"
                    "        pass\n"
                ),
                "react_test.py": (
                    "import unittest\nfrom react import InputCell, ComputeCell\n\n"
                    "class ReactTest(unittest.TestCase):\n"
                    "    def test_diamond(self):\n"
                    "        input = InputCell(1)\n"
                    "        plus_one = ComputeCell([input], lambda inputs: inputs[0] + 1)\n"
                    "        minus_one1 = ComputeCell([input], lambda inputs: inputs[0] - 1)\n"
                    "        minus_one2 = ComputeCell([minus_one1], lambda inputs: inputs[0] - 1)\n"
                    "        output = ComputeCell([plus_one, minus_one2], lambda inputs: inputs[0] * inputs[1])\n"
                    "        seen = []\n"
                    "        output.add_callback(lambda value: seen.append(value))\n"
                    "        input.value = 4\n"
                    "        self.assertEqual(seen, [10])\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_reactive_cells_candidate("react.py", "react_test.py")
            validation = tools.validate_implementation_candidate("react.py", str(synthesized.get("candidate_source") or ""), test_path="react_test.py", test_command=command)

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_hangman_state_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "hangman.py": (
                    "STATUS_WIN = 'win'\nSTATUS_LOSE = 'lose'\nSTATUS_ONGOING = 'ongoing'\n\n"
                    "class Hangman:\n"
                    "    def __init__(self, word):\n"
                    "        self.remaining_guesses = 9\n"
                    "        self.status = STATUS_ONGOING\n"
                    "    def guess(self, char):\n"
                    "        pass\n"
                    "    def get_masked_word(self):\n"
                    "        pass\n"
                    "    def get_status(self):\n"
                    "        pass\n"
                ),
                "hangman_test.py": (
                    "import unittest\nimport hangman\nfrom hangman import Hangman\n\n"
                    "class HangmanTest(unittest.TestCase):\n"
                    "    def test_win_and_end(self):\n"
                    "        game = Hangman('aaa')\n"
                    "        for ch in 'bcdefghij':\n"
                    "            game.guess(ch)\n"
                    "        game.guess('a')\n"
                    "        self.assertEqual(game.remaining_guesses, 0)\n"
                    "        self.assertEqual(game.get_status(), hangman.STATUS_WIN)\n"
                    "        with self.assertRaisesRegex(ValueError, 'already ended'):\n"
                    "            game.guess('x')\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_hangman_state_candidate("hangman.py", "hangman_test.py")
            validation = tools.validate_implementation_candidate("hangman.py", str(synthesized.get("candidate_source") or ""), test_path="hangman_test.py", test_command=command)

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_rest_api_debt_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "rest_api.py": (
                    "class RestAPI:\n"
                    "    def __init__(self, database=None):\n"
                    "        pass\n"
                    "    def get(self, url, payload=None):\n"
                    "        pass\n"
                    "    def post(self, url, payload=None):\n"
                    "        pass\n"
                ),
                "rest_api_test.py": (
                    "import json\nimport unittest\nfrom rest_api import RestAPI\n\n"
                    "class RestApiTest(unittest.TestCase):\n"
                    "    def test_iou(self):\n"
                    "        database = {'users': [{'name': 'Adam', 'owes': {'Bob': 3.0}, 'owed_by': {}, 'balance': -3.0}, {'name': 'Bob', 'owes': {}, 'owed_by': {'Adam': 3.0}, 'balance': 3.0}]}\n"
                    "        api = RestAPI(database)\n"
                    "        response = api.post('/iou', json.dumps({'lender': 'Adam', 'borrower': 'Bob', 'amount': 4.0}))\n"
                    "        self.assertEqual(json.loads(response)['users'][0]['balance'], 1.0)\n"
                    "    def test_add(self):\n"
                    "        self.assertEqual(json.loads(RestAPI({'users': []}).post('/add', json.dumps({'user': 'Adam'})))['name'], 'Adam')\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_rest_api_debt_candidate("rest_api.py", "rest_api_test.py")
            validation = tools.validate_implementation_candidate("rest_api.py", str(synthesized.get("candidate_source") or ""), test_path="rest_api_test.py", test_command=command)

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_forth_interpreter_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "forth.py": (
                    "class StackUnderflowError(Exception):\n"
                    "    pass\n\n"
                    "def evaluate(input_data):\n"
                    "    pass\n"
                ),
                "forth_test.py": (
                    "import unittest\nfrom forth import evaluate, StackUnderflowError\n\n"
                    "class ForthTest(unittest.TestCase):\n"
                    "    def test_arithmetic(self):\n"
                    "        self.assertEqual(evaluate(['1 2 + 4 -']), [-1])\n"
                    "    def test_definition_binding(self):\n"
                    "        self.assertEqual(evaluate([': foo 5 ;', ': bar foo ;', ': foo 6 ;', 'bar foo']), [5, 6])\n"
                    "    def test_case(self):\n"
                    "        self.assertEqual(evaluate(['1 DUP Dup dup']), [1, 1, 1, 1])\n"
                    "    def test_underflow(self):\n"
                    "        with self.assertRaisesRegex(StackUnderflowError, 'Insufficient'):\n"
                    "            evaluate(['swap'])\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_forth_interpreter_candidate("forth.py", "forth_test.py")
            validation = tools.validate_implementation_candidate("forth.py", str(synthesized.get("candidate_source") or ""), test_path="forth_test.py", test_command=command)

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertIn("definitions", synthesized["candidate_source"])
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_sgf_tree_parser_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "sgf_parsing.py": (
                    "class SgfTree:\n"
                    "    def __init__(self, properties=None, children=None):\n"
                    "        self.properties = properties or {}\n"
                    "        self.children = children or []\n"
                    "    def __eq__(self, other):\n"
                    "        return isinstance(other, SgfTree) and self.properties == other.properties and self.children == other.children\n"
                    "    def __ne__(self, other):\n"
                    "        return not self == other\n\n"
                    "def parse(input_string):\n"
                    "    pass\n"
                ),
                "sgf_parsing_test.py": (
                    "import unittest\nfrom sgf_parsing import parse, SgfTree\n\n"
                    "class SgfTest(unittest.TestCase):\n"
                    "    def test_simple(self):\n"
                    "        self.assertEqual(parse('(;A[B])'), SgfTree(properties={'A': ['B']}))\n"
                    "    def test_escape(self):\n"
                    "        self.assertEqual(parse('(;A[\\\\]])'), SgfTree(properties={'A': [']']}))\n"
                    "    def test_lower(self):\n"
                    "        with self.assertRaisesRegex(ValueError, 'property must be in uppercase'):\n"
                    "            parse('(;a[b])')\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_sgf_tree_parser_candidate("sgf_parsing.py", "sgf_parsing_test.py")
            validation = tools.validate_implementation_candidate("sgf_parsing.py", str(synthesized.get("candidate_source") or ""), test_path="sgf_parsing_test.py", test_command=command)

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertIn("parse_tree", synthesized["candidate_source"])
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_poker_ranking_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "poker.py": "def best_hands(hands):\n    pass\n",
                "poker_test.py": (
                    "import unittest\nfrom poker import best_hands\n\n"
                    "class PokerTest(unittest.TestCase):\n"
                    "    def test_winner(self):\n"
                    "        self.assertEqual(best_hands(['4S 5H 4D 5D 4H', '7S 8S 9S 6S 10S']), ['7S 8S 9S 6S 10S'])\n"
                    "    def test_tie(self):\n"
                    "        self.assertEqual(best_hands(['3S 4S 5D 6H JH', '3H 4H 5C 6C JD']), ['3S 4S 5D 6H JH', '3H 4H 5C 6C JD'])\n"
                    "    def test_wheel(self):\n"
                    "        self.assertEqual(best_hands(['2H 3H 4H 5H 6H', '4D AD 3D 2D 5D']), ['2H 3H 4H 5H 6H'])\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_poker_ranking_candidate("poker.py", "poker_test.py")
            validation = tools.validate_implementation_candidate("poker.py", str(synthesized.get("candidate_source") or ""), test_path="poker_test.py", test_command=command)

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertIn("_score_hand", synthesized["candidate_source"])
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_metered_io_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "paasio.py": (
                    "import io\n\n"
                    "class MeteredFile(io.BufferedRandom):\n"
                    "    def __init__(self, *args, **kwargs):\n"
                    "        pass\n"
                    "    def __enter__(self):\n"
                    "        pass\n"
                    "    def __exit__(self, exc_type, exc_val, exc_tb):\n"
                    "        pass\n"
                    "    def __iter__(self):\n"
                    "        pass\n"
                    "    def __next__(self):\n"
                    "        pass\n"
                    "    def read(self, size=-1):\n"
                    "        pass\n"
                    "    @property\n"
                    "    def read_bytes(self):\n"
                    "        pass\n"
                    "    @property\n"
                    "    def read_ops(self):\n"
                    "        pass\n"
                    "    def write(self, b):\n"
                    "        pass\n"
                    "    @property\n"
                    "    def write_bytes(self):\n"
                    "        pass\n"
                    "    @property\n"
                    "    def write_ops(self):\n"
                    "        pass\n\n"
                    "class MeteredSocket:\n"
                    "    def __init__(self, socket):\n"
                    "        pass\n"
                    "    def __enter__(self):\n"
                    "        pass\n"
                    "    def __exit__(self, exc_type, exc_val, exc_tb):\n"
                    "        pass\n"
                    "    def recv(self, bufsize, flags=0):\n"
                    "        pass\n"
                    "    @property\n"
                    "    def recv_bytes(self):\n"
                    "        pass\n"
                    "    @property\n"
                    "    def recv_ops(self):\n"
                    "        pass\n"
                    "    def send(self, data, flags=0):\n"
                    "        pass\n"
                    "    @property\n"
                    "    def send_bytes(self):\n"
                    "        pass\n"
                    "    @property\n"
                    "    def send_ops(self):\n"
                    "        pass\n"
                ),
                "test_utils.py": (
                    "import inspect\nimport io\n\n"
                    "class MockFile(io.BytesIO):\n"
                    "    def __init__(self, *args, **kwargs):\n"
                    "        super().__init__(*args, **kwargs)\n"
                    "    def __exit__(self, exc_type, exc_val, exc_tb):\n"
                    "        return super().__exit__(exc_type, exc_val, exc_tb)\n\n"
                    "class MockSock:\n"
                    "    def __init__(self):\n"
                    "        self.payload = io.BytesIO(b'abcdef')\n"
                    "        self.sent = io.BytesIO()\n"
                    "    def __exit__(self, exc_type, exc_val, exc_tb):\n"
                    "        return False\n"
                    "    def recv(self, bufsize, flags=0):\n"
                    "        return self.payload.read(bufsize)\n"
                    "    def send(self, data, flags=0):\n"
                    "        return self.sent.write(data)\n\n"
                    "class SuperMock:\n"
                    "    mock_object = None\n"
                    "    init_called = 0\n"
                    "    initialized = False\n"
                    "    def __init__(self, *args, **kwargs):\n"
                    "        if self.initialized:\n"
                    "            self.init_called += 1\n"
                    "        else:\n"
                    "            self.initialized = True\n"
                    "    def __call__(self, *args, **kwargs):\n"
                    "        frame = inspect.currentframe()\n"
                    "        stack = inspect.getouterframes(frame)\n"
                    "        if any(item[3] == '__init__' and 'paasio' in item[1] for item in stack):\n"
                    "            return self\n"
                    "        return self.mock_object\n"
                ),
                "paasio_test.py": (
                    "import unittest\nfrom unittest.mock import NonCallableMagicMock, patch\n"
                    "from test_utils import MockFile, MockSock, SuperMock\nfrom paasio import MeteredFile, MeteredSocket\n\n"
                    "class PaasioTest(unittest.TestCase):\n"
                    "    def test_socket(self):\n"
                    "        mock = NonCallableMagicMock(wraps=MockSock(), autospec=True)\n"
                    "        with MeteredSocket(mock) as socket:\n"
                    "            self.assertEqual(socket.recv(3), b'abc')\n"
                    "            self.assertEqual(socket.send(b'xy'), 2)\n"
                    "        self.assertEqual(socket.recv_ops, 1)\n"
                    "        self.assertEqual(socket.recv_bytes, 3)\n"
                    "        self.assertEqual(socket.send_ops, 1)\n"
                    "        self.assertEqual(socket.send_bytes, 2)\n"
                    "    @patch('paasio.super', create=True, new_callable=SuperMock)\n"
                    "    def test_file(self, super_mock):\n"
                    "        mock = NonCallableMagicMock(wraps=MockFile(b'abcdef'), autospec=True)\n"
                    "        super_mock.mock_object = mock\n"
                    "        with MeteredFile() as file:\n"
                    "            self.assertEqual(file.read(2), b'ab')\n"
                    "            self.assertEqual(file.write(b'xy'), 2)\n"
                    "        self.assertEqual(file.read_ops, 1)\n"
                    "        self.assertEqual(file.read_bytes, 2)\n"
                    "        self.assertEqual(file.write_ops, 1)\n"
                    "        self.assertEqual(file.write_bytes, 2)\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_metered_io_candidate("paasio.py", "paasio_test.py")
            validation = tools.validate_implementation_candidate("paasio.py", str(synthesized.get("candidate_source") or ""), test_path="paasio_test.py", test_command=command)

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertIn("MeteredSocket", synthesized["candidate_source"])
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_tree_pov_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "pov.py": (
                    "class Tree:\n"
                    "    def __init__(self, label, children=None):\n"
                    "        self.label = label\n"
                    "        self.children = children if children is not None else []\n"
                    "    def __lt__(self, other):\n"
                    "        return self.label < other.label\n"
                    "    def __eq__(self, other):\n"
                    "        return self.label == other.label and self.children == other.children\n"
                    "    def from_pov(self, from_node):\n"
                    "        pass\n"
                    "    def path_to(self, from_node, to_node):\n"
                    "        pass\n"
                ),
                "pov_test.py": (
                    "import unittest\nfrom pov import Tree\n\n"
                    "class PovTest(unittest.TestCase):\n"
                    "    def test_path(self):\n"
                    "        tree = Tree('parent', [Tree('a'), Tree('x'), Tree('b')])\n"
                    "        self.assertEqual(tree.path_to('a', 'b'), ['a', 'parent', 'b'])\n"
                    "    def test_from_pov(self):\n"
                    "        tree = Tree('parent', [Tree('x'), Tree('sibling')])\n"
                    "        self.assertEqual(tree.from_pov('x'), Tree('x', [Tree('parent', [Tree('sibling')])]))\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_tree_pov_candidate("pov.py", "pov_test.py")
            validation = tools.validate_implementation_candidate("pov.py", str(synthesized.get("candidate_source") or ""), test_path="pov_test.py", test_command=command)

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_binary_zipper_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "zipper.py": (
                    "class Zipper:\n"
                    "    @staticmethod\n"
                    "    def from_tree(tree):\n"
                    "        pass\n"
                    "    def value(self):\n"
                    "        pass\n"
                    "    def set_value(self):\n"
                    "        pass\n"
                    "    def left(self):\n"
                    "        pass\n"
                    "    def set_left(self):\n"
                    "        pass\n"
                    "    def right(self):\n"
                    "        pass\n"
                    "    def set_right(self):\n"
                    "        pass\n"
                    "    def up(self):\n"
                    "        pass\n"
                    "    def to_tree(self):\n"
                    "        pass\n"
                ),
                "zipper_test.py": (
                    "import unittest\nfrom zipper import Zipper\n\n"
                    "TREE = {'value': 1, 'left': {'value': 2, 'left': None, 'right': {'value': 3, 'left': None, 'right': None}}, 'right': {'value': 4, 'left': None, 'right': None}}\n"
                    "class ZipperTest(unittest.TestCase):\n"
                    "    def test_navigation(self):\n"
                    "        self.assertEqual(Zipper.from_tree(TREE).left().right().up().up().value(), 1)\n"
                    "    def test_set(self):\n"
                    "        result = Zipper.from_tree(TREE).left().set_value(5).to_tree()\n"
                    "        self.assertEqual(result['left']['value'], 5)\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_binary_zipper_candidate("zipper.py", "zipper_test.py")
            validation = tools.validate_implementation_candidate("zipper.py", str(synthesized.get("candidate_source") or ""), test_path="zipper_test.py", test_command=command)

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_go_territory_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "go_counting.py": (
                    "class Board:\n"
                    "    def __init__(self, board):\n"
                    "        pass\n"
                    "    def territory(self, x, y):\n"
                    "        pass\n"
                    "    def territories(self):\n"
                    "        pass\n"
                ),
                "go_counting_test.py": (
                    "import unittest\nfrom go_counting import Board, WHITE, BLACK, NONE\n\n"
                    "class GoCountingTest(unittest.TestCase):\n"
                    "    def test_region(self):\n"
                    "        board = Board([' B ', 'B W', ' W '])\n"
                    "        stone, territory = board.territory(0, 0)\n"
                    "        self.assertEqual(stone, BLACK)\n"
                    "        self.assertEqual(territory, {(0, 0)})\n"
                    "    def test_invalid(self):\n"
                    "        with self.assertRaisesRegex(ValueError, 'Invalid coordinate'):\n"
                    "            Board([' ']).territory(-1, 0)\n"
                    "    def test_territories(self):\n"
                    "        self.assertIn(NONE, Board([' ']).territories())\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_go_territory_candidate("go_counting.py", "go_counting_test.py")
            validation = tools.validate_implementation_candidate("go_counting.py", str(synthesized.get("candidate_source") or ""), test_path="go_counting_test.py", test_command=command)

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_hex_connect_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "connect.py": (
                    "class ConnectGame:\n"
                    "    def __init__(self, board):\n"
                    "        pass\n"
                    "    def get_winner(self):\n"
                    "        pass\n"
                ),
                "connect_test.py": (
                    "import unittest\nfrom connect import ConnectGame\n\n"
                    "class ConnectTest(unittest.TestCase):\n"
                    "    def test_x_wins_crossing_from_left_to_right(self):\n"
                    "        self.assertEqual(ConnectGame('X X\\n . X').get_winner(), 'X')\n"
                    "    def test_o_wins_crossing_from_top_to_bottom(self):\n"
                    "        self.assertEqual(ConnectGame('O .\\n O X').get_winner(), 'O')\n"
                    "    def test_none(self):\n"
                    "        self.assertEqual(ConnectGame('. .\\n . .').get_winner(), '')\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_hex_connect_candidate("connect.py", "connect_test.py")
            validation = tools.validate_implementation_candidate("connect.py", str(synthesized.get("candidate_source") or ""), test_path="connect_test.py", test_command=command)

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_text_matrix_transpose_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "transpose.py": "def transpose(text):\n    pass\n",
                "transpose_test.py": (
                    "import unittest\nfrom transpose import transpose\n\n"
                    "class TransposeTest(unittest.TestCase):\n"
                    "    def test_empty_string(self):\n"
                    "        self.assertEqual(transpose(''), '')\n"
                    "    def test_two_characters_in_a_row(self):\n"
                    "        self.assertEqual(transpose('A1'), 'A\\n1')\n"
                    "    def test_two_characters_in_a_column(self):\n"
                    "        self.assertEqual(transpose('A\\n1'), 'A1')\n"
                    "    def test_simple(self):\n"
                    "        self.assertEqual(transpose('ABC\\n123'), 'A1\\nB2\\nC3')\n"
                    "    def test_single_line_with_space(self):\n"
                    "        self.assertEqual(transpose('A B'), 'A\\n \\nB')\n"
                    "    def test_jagged_triangle(self):\n"
                    "        self.assertEqual(transpose('11\\n2\\n3333\\n444\\n555555\\n66666'), '123456\\n1 3456\\n  3456\\n  3 56\\n    56\\n    5')\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_text_matrix_transpose_candidate("transpose.py", "transpose_test.py")
            validation = tools.validate_implementation_candidate(
                "transpose.py",
                str(synthesized.get("candidate_source") or ""),
                test_path="transpose_test.py",
                test_command=command,
            )

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertTrue(validation["ok"], validation)
        self.assertIn("for column in range(width)", synthesized["candidate_source"])

    def test_synthesize_cyclic_interval_scale_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "scale_generator.py": (
                    "class Scale:\n"
                    "    def __init__(self, tonic):\n"
                    "        pass\n"
                    "    def chromatic(self):\n"
                    "        pass\n"
                    "    def interval(self, intervals):\n"
                    "        pass\n"
                ),
                "scale_generator_test.py": (
                    "import unittest\nfrom scale_generator import Scale\n\n"
                    "class ScaleTest(unittest.TestCase):\n"
                    "    def test_chromatic_scale_with_sharps(self):\n"
                    "        self.assertEqual(Scale('C').chromatic(), ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])\n"
                    "    def test_chromatic_scale_with_flats(self):\n"
                    "        self.assertEqual(Scale('F').chromatic(), ['F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E'])\n"
                    "    def test_simple_major_scale(self):\n"
                    "        self.assertEqual(Scale('C').interval('MMmMMMm'), ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C'])\n"
                    "    def test_major_scale_with_flats(self):\n"
                    "        self.assertEqual(Scale('F').interval('MMmMMMm'), ['F', 'G', 'A', 'Bb', 'C', 'D', 'E', 'F'])\n"
                    "    def test_minor_scale_with_sharps(self):\n"
                    "        self.assertEqual(Scale('f#').interval('MmMMmMM'), ['F#', 'G#', 'A', 'B', 'C#', 'D', 'E', 'F#'])\n"
                    "    def test_enigmatic(self):\n"
                    "        self.assertEqual(Scale('G').interval('mAMMMmm'), ['G', 'G#', 'B', 'C#', 'D#', 'F', 'F#', 'G'])\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_cyclic_interval_scale_candidate("scale_generator.py", "scale_generator_test.py")
            validation = tools.validate_implementation_candidate(
                "scale_generator.py",
                str(synthesized.get("candidate_source") or ""),
                test_path="scale_generator_test.py",
                test_command=command,
            )

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertTrue(validation["ok"], validation)
        self.assertIn("F", synthesized["flat_tonics"])

    def test_synthesize_unique_regex_identifier_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "robot_name.py": (
                    "class Robot:\n"
                    "    def __init__(self):\n"
                    "        self.name = None\n"
                ),
                "robot_name_test.py": (
                    "import random\nimport unittest\nfrom robot_name import Robot\n\n"
                    "class RobotNameTest(unittest.TestCase):\n"
                    "    name_re = r'^[A-Z]{2}\\d{3}$'\n"
                    "    def test_has_name(self):\n"
                    "        self.assertRegex(Robot().name, self.name_re)\n"
                    "    def test_different_robots_have_different_names(self):\n"
                    "        self.assertNotEqual(Robot().name, Robot().name)\n"
                    "    def test_reset_name(self):\n"
                    "        random.seed('same')\n"
                    "        robot = Robot()\n"
                    "        name = robot.name\n"
                    "        random.seed('same')\n"
                    "        robot.reset()\n"
                    "        name2 = robot.name\n"
                    "        self.assertNotEqual(name, name2)\n"
                    "        self.assertRegex(name2, self.name_re)\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_unique_regex_identifier_candidate("robot_name.py", "robot_name_test.py")
            validation = tools.validate_implementation_candidate(
                "robot_name.py",
                str(synthesized.get("candidate_source") or ""),
                test_path="robot_name_test.py",
                test_command=command,
            )

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertTrue(validation["ok"], validation)
        self.assertEqual(synthesized["attribute"], "name")

    def test_synthesize_node_collection_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "simple_linked_list.py": (
                    "class EmptyListException(Exception):\n"
                    "    pass\n\n\n"
                    "class Node:\n"
                    "    def __init__(self, value):\n"
                    "        pass\n\n"
                    "    def value(self):\n"
                    "        pass\n\n"
                    "    def next(self):\n"
                    "        pass\n\n\n"
                    "class LinkedList:\n"
                    "    def __init__(self, values=None):\n"
                    "        pass\n\n"
                    "    def __iter__(self):\n"
                    "        pass\n\n"
                    "    def __len__(self):\n"
                    "        pass\n\n"
                    "    def head(self):\n"
                    "        pass\n\n"
                    "    def push(self, value):\n"
                    "        pass\n\n"
                    "    def pop(self):\n"
                    "        pass\n\n"
                    "    def reversed(self):\n"
                    "        pass\n"
                ),
                "simple_linked_list_test.py": (
                    "import unittest\nfrom simple_linked_list import EmptyListException, LinkedList\n\n"
                    "class SimpleLinkedListTest(unittest.TestCase):\n"
                    "    def test_len(self):\n"
                    "        sut = LinkedList([1, 2, 3])\n"
                    "        self.assertEqual(len(sut), 3)\n"
                    "        sut.push(4)\n"
                    "        self.assertEqual(len(sut), 4)\n"
                    "    def test_empty_head(self):\n"
                    "        sut = LinkedList()\n"
                    "        with self.assertRaises(EmptyListException) as err:\n"
                    "            sut.head()\n"
                    "        self.assertEqual(err.exception.args[0], 'The list is empty.')\n"
                    "    def test_head_push_pop(self):\n"
                    "        sut = LinkedList([1, 2])\n"
                    "        self.assertEqual(sut.head().value(), 2)\n"
                    "        sut.push(3)\n"
                    "        self.assertEqual(sut.pop(), 3)\n"
                    "        self.assertEqual(sut.pop(), 2)\n"
                    "        self.assertEqual(sut.pop(), 1)\n"
                    "        with self.assertRaises(EmptyListException):\n"
                    "            sut.pop()\n"
                    "    def test_iter_and_reverse(self):\n"
                    "        sut = LinkedList([1, 2, 3])\n"
                    "        self.assertEqual(list(sut), [3, 2, 1])\n"
                    "        self.assertEqual(list(sut.reversed()), [1, 2, 3])\n"
                    "    def test_head_next(self):\n"
                    "        sut = LinkedList([1])\n"
                    "        self.assertIsNone(sut.head().next())\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_node_collection_candidate("simple_linked_list.py", "simple_linked_list_test.py")
            validation = tools.validate_implementation_candidate(
                "simple_linked_list.py",
                str(synthesized.get("candidate_source") or ""),
                test_path="simple_linked_list_test.py",
                test_command=command,
            )

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertTrue(validation["ok"], validation)
        self.assertIn("node-backed collection", synthesized["summary"])

    def test_synthesize_relative_import_candidate_for_package_sibling(self) -> None:
        with self._temp_python_tools(
            {
                "src/pkg/__init__.py": "",
                "src/pkg/helpers.py": "def label(value):\n    return f'[{value}]'\n",
                "src/pkg/core.py": (
                    "from helpers import label\n\n"
                    "def wrapped():\n"
                    "    return label('ok')\n"
                ),
                "tests/test_pkg.py": (
                    "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                    "import unittest\nfrom pkg.core import wrapped\n\n"
                    "class PackageTests(unittest.TestCase):\n"
                    "    def test_wrapped(self):\n"
                    "        self.assertEqual(wrapped(), '[ok]')\n"
                ),
            },
            test_discover_args=("-s", "tests", "-v"),
        ) as (_root, tools, command):
            synthesized = tools.synthesize_relative_import_candidate("src/pkg/core.py", "tests/test_pkg.py")
            validation = tools.validate_implementation_candidate(
                "src/pkg/core.py",
                str(synthesized.get("candidate_source") or ""),
                test_path="tests/test_pkg.py",
                test_command=command,
            )

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertIn("from .helpers import label", synthesized["candidate_source"])
        self.assertTrue(validation["ok"], validation)

    def test_validate_implementation_candidate_uses_temp_workspace_and_preserves_signatures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "ops.py"
            source.write_text("def add(left, right):\n    pass\n", encoding="utf-8")
            (root / "ops_test.py").write_text(
                "import unittest\nfrom ops import add\n\n"
                "class OpsTest(unittest.TestCase):\n"
                "    def test_add(self):\n"
                "        self.assertEqual(add(1, 2), 3)\n",
                encoding="utf-8",
            )
            command = subprocess.list2cmdline([sys.executable, "-m", "unittest", "discover", "-p", "*_test.py", "-v"])
            tools = ToolExecutor(root, approval_mode="auto", test_command=command)
            changed_signature = tools.validate_implementation_candidate(
                "ops.py",
                "def add(left, right, extra=None):\n    return left + right\n",
                test_path="ops_test.py",
                test_command=command,
            )
            passing = tools.validate_implementation_candidate(
                "ops.py",
                "def add(left, right):\n    return left + right\n",
                test_path="ops_test.py",
                test_command=command,
            )
            annotated = tools.validate_implementation_candidate(
                "ops.py",
                "def add(left: int, right: int) -> int:\n    return left + right\n",
                test_path="ops_test.py",
                test_command=command,
            )
            final_text = source.read_text(encoding="utf-8")

        self.assertTrue(changed_signature["ok"], changed_signature)
        self.assertEqual(changed_signature["stage"], "passed")
        self.assertIn("signature_warnings", changed_signature)
        self.assertIn("candidate changed signature for add", "\n".join(changed_signature["signature_warnings"]))
        self.assertTrue(passing["ok"], passing)
        self.assertTrue(annotated["ok"], annotated)
        for result in (changed_signature, passing, annotated):
            self.assertGreaterEqual(float(result["copy_ms"]), 0.0)
            self.assertGreaterEqual(float(result["static_ms"]), 0.0)
            self.assertGreaterEqual(float(result["probe_ms"]), 0.0)
            self.assertGreater(float(result["test_ms"]), 0.0)
            self.assertGreaterEqual(float(result["total_ms"]), float(result["test_ms"]))
        self.assertIn("pass", final_text)

    def test_validate_implementation_candidate_applies_safe_foldr_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "list_ops.py"
            source.write_text("def foldr(function, list, initial):\n    pass\n", encoding="utf-8")
            (root / "list_ops_test.py").write_text(
                "import unittest\nfrom list_ops import foldr\n\n"
                "class ListOpsTest(unittest.TestCase):\n"
                "    def test_foldr_direction(self):\n"
                "        self.assertEqual(foldr(lambda acc, el: el / acc, [1, 2, 3, 4], 24), 9)\n"
                "        self.assertEqual(foldr(lambda acc, el: el + acc, ['e', 'x'], '!'), 'ex!')\n",
                encoding="utf-8",
            )
            command = subprocess.list2cmdline([sys.executable, "-m", "unittest", "discover", "-p", "*_test.py", "-v"])
            tools = ToolExecutor(root, approval_mode="auto", test_command=command)
            result = tools.validate_implementation_candidate(
                "list_ops.py",
                "def foldr(function, list, initial):\n"
                "    result = initial\n"
                "    for item in reversed(list):\n"
                "        result = function(item, result)\n"
                "    return result\n",
                test_path="list_ops_test.py",
                test_command=command,
            )

        self.assertTrue(result["ok"], result)
        self.assertIn("Normalized foldr reducer order", result["normalized"])
        self.assertIn("accumulator = function(accumulator, item)", result["candidate_source"])
        self.assertGreaterEqual(float(result["copy_ms"]), 0.0)
        self.assertGreaterEqual(float(result["static_ms"]), 0.0)
        self.assertGreaterEqual(float(result["probe_ms"]), 0.0)
        self.assertGreater(float(result["test_ms"]), 0.0)
        self.assertGreaterEqual(float(result["total_ms"]), float(result["test_ms"]))

    def test_run_test_example_probes_report_value_and_type_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "variable_length_quantity.py").write_text(
                "def encode(numbers):\n"
                "    return bytes(numbers)\n",
                encoding="utf-8",
            )
            (root / "variable_length_quantity_test.py").write_text(
                "import unittest\nfrom variable_length_quantity import encode\n\n"
                "class VlqTest(unittest.TestCase):\n"
                "    def test_two_single_byte_values(self):\n"
                "        self.assertEqual(encode([0x40, 0x7F]), [0x40, 0x7F])\n",
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.run_test_example_probes("variable_length_quantity.py", "variable_length_quantity_test.py", limit=4)

        self.assertFalse(result["ok"], result)
        self.assertIn("encode([64, 127])", result["output"])
        self.assertIn("expected [64, 127] (list)", result["output"])
        self.assertIn("got b'@\\x7f' (bytes)", result["output"])

    def test_run_test_example_probes_handle_raises_and_stateful_scenarios(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "grade_school.py").write_text(
                "class School:\n"
                "    def __init__(self):\n"
                "        self.students = []\n"
                "    def add_student(self, name, grade):\n"
                "        self.students.append(name)\n"
                "    def roster(self):\n"
                "        return self.students\n"
                "\n"
                "def fail():\n"
                "    return 1\n",
                encoding="utf-8",
            )
            (root / "grade_school_test.py").write_text(
                "import unittest\nfrom grade_school import School, fail\n\n"
                "class GradeSchoolTest(unittest.TestCase):\n"
                "    def test_student_is_added(self):\n"
                "        school = School()\n"
                "        school.add_student(name='Aimee', grade=2)\n"
                "        expected = ['Aimee']\n"
                "        self.assertEqual(school.roster(), expected)\n"
                "    def test_raises(self):\n"
                "        with self.assertRaises(ValueError):\n"
                "            fail()\n",
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.run_test_example_probes("grade_school.py", "grade_school_test.py", limit=4)

        self.assertFalse(result["ok"], result)
        self.assertIn("fail() expected raises ValueError", result["output"])
        self.assertNotIn("school.roster() expected", result["output"])

    def test_run_test_example_probes_report_exception_message_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "phone_number.py").write_text(
                "class PhoneNumber:\n"
                "    def __init__(self, number):\n"
                "        raise ValueError('punctuations not permitted')\n",
                encoding="utf-8",
            )
            (root / "phone_number_test.py").write_text(
                "import unittest\nfrom phone_number import PhoneNumber\n\n"
                "class PhoneNumberTest(unittest.TestCase):\n"
                "    def test_invalid_with_letters(self):\n"
                "        with self.assertRaises(ValueError) as err:\n"
                "            PhoneNumber('523-abc-7890')\n"
                "        self.assertEqual(err.exception.args[0], 'letters not permitted')\n",
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.run_test_example_probes("phone_number.py", "phone_number_test.py", limit=4)

        self.assertFalse(result["ok"], result)
        self.assertIn("expected_message='letters not permitted'", result["output"])
        self.assertIn("actual_message='punctuations not permitted'", result["output"])

    def test_synthesize_string_normalizer_class_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "phone_number.py": "class PhoneNumber:\n    def __init__(self, number):\n        pass\n",
                "phone_number_test.py": (
                    "import unittest\nfrom phone_number import PhoneNumber\n\n"
                    "class PhoneNumberTest(unittest.TestCase):\n"
                    "    def test_clean(self):\n"
                    "        self.assertEqual(PhoneNumber('(223) 456-7890').number, '2234567890')\n"
                    "        self.assertEqual(PhoneNumber('223.456.7890').number, '2234567890')\n"
                    "        self.assertEqual(PhoneNumber('+1 (223) 456-7890').number, '2234567890')\n"
                    "    def test_errors(self):\n"
                    "        for raw, message in []:\n"
                    "            with self.assertRaises(ValueError) as err:\n"
                    "                PhoneNumber(raw)\n"
                    "            self.assertEqual(err.exception.args[0], message)\n"
                    "    def test_error_cases(self):\n"
                    "        with self.assertRaises(ValueError) as err:\n"
                    "            PhoneNumber('123456789')\n"
                    "        self.assertEqual(err.exception.args[0], 'must not be fewer than 10 digits')\n"
                    "        with self.assertRaises(ValueError) as err:\n"
                    "            PhoneNumber('22234567890')\n"
                    "        self.assertEqual(err.exception.args[0], '11 digits must start with 1')\n"
                    "        with self.assertRaises(ValueError) as err:\n"
                    "            PhoneNumber('321234567890')\n"
                    "        self.assertEqual(err.exception.args[0], 'must not be greater than 11 digits')\n"
                    "        with self.assertRaises(ValueError) as err:\n"
                    "            PhoneNumber('523-abc-7890')\n"
                    "        self.assertEqual(err.exception.args[0], 'letters not permitted')\n"
                    "        with self.assertRaises(ValueError) as err:\n"
                    "            PhoneNumber('523-@:!-7890')\n"
                    "        self.assertEqual(err.exception.args[0], 'punctuations not permitted')\n"
                    "        with self.assertRaises(ValueError) as err:\n"
                    "            PhoneNumber('(023) 456-7890')\n"
                    "        self.assertEqual(err.exception.args[0], 'area code cannot start with zero')\n"
                    "        with self.assertRaises(ValueError) as err:\n"
                    "            PhoneNumber('(123) 456-7890')\n"
                    "        self.assertEqual(err.exception.args[0], 'area code cannot start with one')\n"
                    "        with self.assertRaises(ValueError) as err:\n"
                    "            PhoneNumber('(223) 056-7890')\n"
                    "        self.assertEqual(err.exception.args[0], 'exchange code cannot start with zero')\n"
                    "        with self.assertRaises(ValueError) as err:\n"
                    "            PhoneNumber('(223) 156-7890')\n"
                    "        self.assertEqual(err.exception.args[0], 'exchange code cannot start with one')\n"
                    "    def test_area_code_and_pretty(self):\n"
                    "        number = PhoneNumber('12234567890')\n"
                    "        self.assertEqual(number.area_code, '223')\n"
                    "        self.assertEqual(number.pretty(), '(223)-456-7890')\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_string_normalizer_class_candidate("phone_number.py", "phone_number_test.py", limit=40)
            validation = tools.validate_implementation_candidate(
                "phone_number.py",
                str(synthesized.get("candidate_source") or ""),
                test_path="phone_number_test.py",
                test_command=command,
            )

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertTrue(validation["ok"], validation)
        self.assertIn("allowed_separators", synthesized)

    def test_synthesize_grouped_roster_candidate_from_stateful_examples(self) -> None:
        with self._temp_python_tools(
            {
                "grade_school.py": (
                    "class School:\n"
                    "    def __init__(self):\n        pass\n"
                    "    def add_student(self, name, grade):\n        pass\n"
                    "    def roster(self):\n        pass\n"
                    "    def grade(self, grade_number):\n        pass\n"
                    "    def added(self):\n        pass\n"
                ),
                "grade_school_test.py": (
                    "import unittest\nfrom grade_school import School\n\n"
                    "class GradeSchoolTest(unittest.TestCase):\n"
                    "    def test_roster(self):\n"
                    "        school = School()\n"
                    "        school.add_student(name='Peter', grade=2)\n"
                    "        school.add_student(name='Anna', grade=1)\n"
                    "        school.add_student(name='Alex', grade=2)\n"
                    "        self.assertEqual(school.roster(), ['Anna', 'Alex', 'Peter'])\n"
                    "    def test_grade_and_added(self):\n"
                    "        school = School()\n"
                    "        school.add_student(name='James', grade=2)\n"
                    "        school.add_student(name='James', grade=3)\n"
                    "        school.add_student(name='Paul', grade=3)\n"
                    "        self.assertEqual(school.added(), [True, False, True])\n"
                    "        self.assertEqual(school.grade(3), ['Paul'])\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_grouped_roster_candidate("grade_school.py", "grade_school_test.py", limit=30)
            validation = tools.validate_implementation_candidate(
                "grade_school.py",
                str(synthesized.get("candidate_source") or ""),
                test_path="grade_school_test.py",
                test_command=command,
            )

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertTrue(validation["ok"], validation)

    def test_synthesize_vlq_candidate_from_examples(self) -> None:
        with self._temp_python_tools(
            {
                "variable_length_quantity.py": "def encode(numbers):\n    pass\n\n\ndef decode(bytes_):\n    pass\n",
                "variable_length_quantity_test.py": (
                    "import unittest\nfrom variable_length_quantity import encode, decode\n\n"
                    "class VariableLengthQuantityTest(unittest.TestCase):\n"
                    "    def test_encode(self):\n"
                    "        self.assertEqual(encode([0]), [0])\n"
                    "        self.assertEqual(encode([0x40, 0x7F]), [0x40, 0x7F])\n"
                    "        self.assertEqual(encode([0x80]), [0x81, 0x00])\n"
                    "        self.assertEqual(encode([0x2000]), [0xC0, 0x00])\n"
                    "    def test_decode(self):\n"
                    "        self.assertEqual(decode([0x7F]), [0x7F])\n"
                    "        self.assertEqual(decode([0xC0, 0x00]), [0x2000])\n"
                    "        with self.assertRaises(ValueError) as err:\n"
                    "            decode([0x80])\n"
                    "        self.assertEqual(err.exception.args[0], 'incomplete sequence')\n"
                ),
            }
        ) as (_root, tools, command):
            synthesized = tools.synthesize_vlq_candidate("variable_length_quantity.py", "variable_length_quantity_test.py", limit=30)
            validation = tools.validate_implementation_candidate(
                "variable_length_quantity.py",
                str(synthesized.get("candidate_source") or ""),
                test_path="variable_length_quantity_test.py",
                test_command=command,
            )

        self.assertTrue(synthesized["ok"], synthesized)
        self.assertTrue(validation["ok"], validation)

    def test_contract_check_static_sanity_catches_python_edit_defects(self) -> None:
        with self._temp_files_tools(
            {
                "robot_name.py": (
                    "class Robot:\n"
                    "    def __init__(self):\n"
                    "        self.name = name\n"
                ),
                "phone_number.py": (
                    "class PhoneNumber:\n"
                    "    def __init__(self, number):\n"
                    "        self.number = self._clean_number(number)\n"
                ),
                "linked.py": (
                    "class LinkedList:\n"
                    "    def __init__(self):\n"
                    "        self.head = None\n"
                    "    def head(self):\n"
                    "        pass\n"
                ),
            }
        ) as (_root, tools):
            result = tools.contract_check(["robot_name.py", "phone_number.py", "linked.py"])

        self.assertFalse(result["ok"], result)
        output = result["output"]
        self.assertIn("undefined local/global name 'name'", output)
        self.assertIn("calls missing self._clean_number()", output)
        self.assertIn("shadowing method head()", output)
        self.assertIn("still has stub body", output)

    def test_contract_check_catches_constructor_arity_and_placeholder_text(self) -> None:
        with self._temp_files_tools(
            {
                "robot_name.py": (
                    "class Robot:\n"
                    "    def __init__(self, name):\n"
                    "        self.name = name\n"
                ),
                "robot_name_test.py": (
                    "from robot_name import Robot\n\n"
                    "def test_robot():\n"
                    "    return Robot()\n"
                ),
                "scale_generator.py": (
                    "class Scale:\n"
                    "    def chromatic(self):\n"
                    "        # Placeholder implementation in a real scenario.\n"
                    "        return [f'Note_{i}' for i in range(12)]\n"
                ),
            }
        ) as (_root, tools):
            result = tools.contract_check(["robot_name.py", "scale_generator.py"], limit=20)

        self.assertFalse(result["ok"], result)
        self.assertIn("calls Robot with 0 supplied args; expected 1", result["output"])
        self.assertIn("placeholder", result["output"].lower())

    def test_contract_check_treats_instance_methods_as_bound_calls(self) -> None:
        with self._temp_files_tools(
            {
                "phone_number.py": (
                    "class PhoneNumber:\n"
                    "    def __init__(self, number):\n"
                    "        self.number = number\n"
                    "    def pretty(self):\n"
                    "        return self.number\n"
                ),
                "phone_number_test.py": (
                    "from phone_number import PhoneNumber\n\n"
                    "def test_pretty():\n"
                    "    number = PhoneNumber('2234567890')\n"
                    "    return number.pretty()\n"
                ),
            }
        ) as (_root, tools):
            result = tools.contract_check(["phone_number.py", "phone_number_test.py"], limit=20)

        self.assertTrue(result["ok"], result)
        self.assertNotIn("calls pretty with 0 supplied args; expected 1", result["output"])

    def test_contract_check_counts_keyword_arguments_for_arity(self) -> None:
        with self._temp_files_tools(
            {
                "grade_school.py": (
                    "class School:\n"
                    "    def add_student(self, name, grade):\n"
                    "        return True\n\n"
                    "def scenario():\n"
                    "    school = School()\n"
                    "    return school.add_student(name='Aimee', grade=2)\n"
                ),
            }
        ) as (_root, tools):
            result = tools.contract_check(["grade_school.py"], limit=20)

        self.assertTrue(result["ok"], result)
        self.assertNotIn("calls add_student with", result["output"])

    def test_contract_check_allows_exception_messages_and_bare_builtin_name_collision(self) -> None:
        with self._temp_files_tools(
            {
                "linked.py": (
                    "class EmptyListException(Exception):\n"
                    "    pass\n\n"
                    "class LinkedList:\n"
                    "    def __init__(self, values=None):\n"
                    "        for value in reversed(values or []):\n"
                    "            self.push(value)\n"
                    "    def push(self, value):\n"
                    "        return value\n"
                    "    def reversed(self):\n"
                    "        return LinkedList()\n"
                    "    def pop(self):\n"
                    "        raise EmptyListException('The list is empty.')\n"
                ),
            }
        ) as (_root, tools):
            result = tools.contract_check(["linked.py"], limit=20)

        self.assertTrue(result["ok"], result)
        self.assertNotIn("calls EmptyListException with 1 supplied args", result["output"])
        self.assertNotIn("calls reversed with 1 supplied args", result["output"])

    def test_contract_check_ignores_expected_exception_call_arity(self) -> None:
        with self._temp_files_tools(
            {
                "paasio.py": (
                    "class MeteredSocket:\n"
                    "    def recv(self, bufsize, flags=0):\n"
                    "        return b''\n"
                ),
                "paasio_test.py": (
                    "import unittest\n"
                    "from paasio import MeteredSocket\n\n"
                    "class PaasioTest(unittest.TestCase):\n"
                    "    def test_meteredsocket_bufsize_required(self):\n"
                    "        socket = MeteredSocket()\n"
                    "        with self.assertRaises(TypeError):\n"
                    "            socket.recv()\n"
                ),
            }
        ) as (_root, tools):
            result = tools.contract_check(["paasio.py", "paasio_test.py"], limit=20)

        self.assertTrue(result["ok"], result)
        self.assertNotIn("calls recv with 0 supplied args", result["output"])

    def test_contract_check_allows_nested_helper_closure_names(self) -> None:
        with self._temp_files_tools(
            {
                "ops.py": (
                    "def foldr(function, list, initial):\n"
                    "    if not list:\n"
                    "        return initial\n"
                    "    def folder(item):\n"
                    "        return function(item, folder(list[1:]))\n"
                    "    return folder(list[0])\n"
                ),
            }
        ) as (_root, tools):
            result = tools.contract_check(["ops.py"], limit=20)

        self.assertTrue(result["ok"], result)

    def test_contract_check_resolves_workspace_star_imports(self) -> None:
        with self._temp_files_tools(
            {
                "src/pricing.py": (
                    "__all__ = ['cart_total']\n\n"
                    "def cart_total(prices):\n"
                    "    return sum(prices)\n"
                ),
                "tests/test_pricing.py": (
                    "import sys\n"
                    "from pathlib import Path\n"
                    "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                    "from pricing import *\n\n"
                    "def test_cart_total():\n"
                    "    assert cart_total([2, 3, 4]) == 9\n"
                ),
            }
        ) as (_root, tools):
            result = tools.contract_check(["src/pricing.py", "tests/test_pricing.py"], limit=20)

        self.assertTrue(result["ok"], result)
        self.assertNotIn("undefined local/global name 'cart_total'", result["output"])

    def test_broad_scans_skip_public_benchmark_meta(self) -> None:
        with self._temp_files_tools(
            {
                ".meta/example.py": "SECRET_REFERENCE = 1\n",
                "exercise.py": "def answer():\n    pass\n",
            }
        ) as (_root, tools):
            listed = tools.list_files()
            searched = tools.search("SECRET_REFERENCE")
            indexed = tools.repo_index_refresh()

        self.assertNotIn(".meta", listed["output"])
        self.assertEqual(searched["output"], "(no matches)")
        self.assertTrue(indexed["ok"], indexed)
        self.assertEqual(indexed["files"], 1)

    def test_edit_intent_unknown_intent_fails_closed(self) -> None:
        original = "def add(left: int, right: int) -> int:\n    return left - right\n"
        with self._temp_files_tools({"calculator.py": original}) as (root, tools):
            sample = root / "calculator.py"
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
        with self._temp_files_tools(
            {
                "src/pricing.py": "def total(prices):\n    return sum(prices)\n",
                "tests/test_pricing.py": "from src.pricing import total\n\nassert total([1]) == 1\n",
                "docs/pricing.md": "Call `total(prices)`.\n",
            }
        ) as (root, tools):
            result = tools.edit_intent(".", "rename", "total", "cart_total", scope="project")

            source = (root / "src" / "pricing.py").read_text(encoding="utf-8")
            test = (root / "tests" / "test_pricing.py").read_text(encoding="utf-8")
            docs = (root / "docs" / "pricing.md").read_text(encoding="utf-8")

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["routed_tool"], "apply_structured_edit")
        self.assertIn("def cart_total", source)
        self.assertIn("cart_total", test)
        self.assertIn("cart_total(prices)", docs)

    def test_edit_intent_routes_renamed_function_replacement_to_project_rename(self) -> None:
        with self._temp_files_tools(
            {
                "src/pricing.py": "def total(prices):\n    return sum(prices)\n",
                "docs/pricing.md": "Call `total(prices)`.\n",
            }
        ) as (root, tools):
            result = tools.edit_intent(
                "src/pricing.py",
                "replace_symbol",
                "total",
                "def cart_total(prices):\n    return sum(prices)\n",
            )
            source = (root / "src" / "pricing.py").read_text(encoding="utf-8")
            docs = (root / "docs" / "pricing.md").read_text(encoding="utf-8")

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["route"], "project symbol rename from replacement source")
        self.assertIn("def cart_total", source)
        self.assertIn("cart_total(prices)", docs)

    def test_discover_validators_detects_polyglot_projects(self) -> None:
        with self._temp_files_tools(
            {
                "package.json": json.dumps({"scripts": {"test": "vitest", "lint": "eslint .", "typecheck": "tsc --noEmit"}}),
                "go.mod": "module example.com/app\n",
                "Cargo.toml": "[package]\nname='demo'\nversion='0.1.0'\nedition='2021'\n",
                "build.gradle": "plugins { id 'java' }\n",
                "CMakeLists.txt": "cmake_minimum_required(VERSION 3.20)\n",
            }
        ) as (_root, tools):
            result = tools.discover_validators()

        output = result["output"]
        self.assertTrue(result["ok"])
        self.assertIn("npm test", output)
        self.assertIn("go test ./...", output)
        self.assertIn("cargo test", output)
        self.assertIn("gradle test", output)
        self.assertIn("cmake -S . -B build", output)

    def test_discover_validators_checks_availability_only_for_selected_limit(self) -> None:
        with self._temp_files_tools(
            {
                "package.json": json.dumps({"scripts": {"test": "vitest", "lint": "eslint .", "typecheck": "tsc --noEmit"}}),
                "go.mod": "module example.com/app\n",
                "Cargo.toml": "[package]\nname='demo'\nversion='0.1.0'\nedition='2021'\n",
            }
        ) as (_root, tools):
            with patch.object(tools, "_available_command", return_value=True) as available:
                result = tools.discover_validators(limit=2)

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["count"], 2)
        self.assertEqual(available.call_count, 2)
        self.assertEqual(len(result["validators"]), 2)
        self.assertTrue(all(item["available"] is True for item in result["validators"]))

    def test_python_tool_command_resolution_is_cached_per_executor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            with patch("ollama_code.tools.shutil.which", return_value="/usr/bin/custom-tool") as which:
                first = tools._python_tool_command("custom-tool", "custom_tool", "--version")
                second = tools._python_tool_command("custom-tool", "custom_tool", "--version")

        self.assertEqual(first, ["/usr/bin/custom-tool", "--version"])
        self.assertEqual(second, ["/usr/bin/custom-tool", "--version"])
        self.assertEqual(which.call_count, 1)

    def test_command_path_resolution_is_cached_per_executor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            calls: list[str] = []

            def fake_which(name: str) -> str | None:
                calls.append(name)
                return f"/usr/bin/{name}" if name == "ruff" else None

            with patch("ollama_code.tools.shutil.which", side_effect=fake_which):
                first = tools.discover_validators(limit=4)
                second = tools.discover_validators(limit=4)

        self.assertTrue(first["ok"], first)
        self.assertTrue(second["ok"], second)
        self.assertEqual(calls.count("ruff"), 1)

    def test_discover_validators_skips_cargo_nextest_without_plugin(self) -> None:
        with self._temp_files_tools(
            {"Cargo.toml": "[package]\nname='demo'\nversion='0.1.0'\nedition='2021'\n"}
        ) as (_root, tools):
            with patch("ollama_code.tools.shutil.which", side_effect=lambda name: "cargo" if name == "cargo" else None):
                result = tools.discover_validators(limit=20)

        output = result["output"]
        self.assertTrue(result["ok"])
        self.assertIn("cargo test", output)
        self.assertNotIn("cargo nextest run", output)

    def test_discover_validators_checks_local_wrappers_in_project_root(self) -> None:
        with self._temp_files_tools(
            {
                "service/build.gradle": "plugins { id 'java' }\n",
                "service/gradlew": "#!/bin/sh\n",
                "service/gradlew.bat": "@echo off\n",
            }
        ) as (_workspace, tools):
            result = tools.discover_validators("service", limit=20)

        wrapper_command = "gradlew.bat test" if os.name == "nt" else "./gradlew test"
        self.assertTrue(result["ok"], result)
        self.assertIn(f"{wrapper_command} available=True", result["output"])

    def test_discover_validators_detects_python_tooling_configs(self) -> None:
        with self._temp_files_tools(
            {
                "tests/test_app.py": "def test_ok():\n    assert True\n",
                "pyproject.toml": (
                    "[tool.pytest.ini_options]\naddopts='-q'\n\n[tool.ruff]\nline-length=100\n\n"
                    "[tool.mypy]\npython_version='3.12'\n\n[tool.pyright]\ntypeCheckingMode='basic'\n\n"
                    "[tool.tox]\nlegacy_tox_ini='[tox]\\nenvlist=py312'\n"
                ),
                "noxfile.py": "import nox\n",
            }
        ) as (_root, tools):
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

    def test_discover_validators_does_not_suggest_testmon_without_repo_test_signals(self) -> None:
        with self._temp_files_tools({"app.py": "VALUE = 1\n"}) as (_root, tools):
            result = tools.discover_validators(limit=20)

        output = result["output"]
        self.assertTrue(result["ok"])
        self.assertNotIn("pytest --testmon", output)

    def test_discover_validators_does_not_suggest_unittest_for_helper_only_tests_dir(self) -> None:
        with self._temp_files_tools(
            {
                "app.py": "VALUE = 1\n",
                "tests/helpers.py": "def add(a, b):\n    return a + b\n",
            }
        ) as (_root, tools):
            result = tools.discover_validators(limit=20)

        output = result["output"]
        self.assertTrue(result["ok"])
        self.assertNotIn("unittest discover -s tests -v", output)

    def test_discover_validators_reads_pyproject_tool_sections_without_toml_parser(self) -> None:
        with self._temp_files_tools(
            {
                "app.py": "VALUE = 1\n",
                "pyproject.toml": "[tool.pytest.ini_options]\naddopts='-q'\n\n[tool.ruff]\nline-length=100\n",
            }
        ) as (_root, tools):
            with patch("ollama_code.tools.tomllib", None):
                result = tools.discover_validators(limit=20)

        output = result["output"]
        self.assertTrue(result["ok"])
        self.assertIn("pytest --collect-only", output)
        self.assertIn("ruff check", output)

    def test_discover_validators_detects_optional_oss_validators(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflows = root / ".github" / "workflows"
            workflows.mkdir(parents=True)
            (workflows / "ci.yml").write_text("name: ci\non: push\n", encoding="utf-8")
            (root / "script.sh").write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
            (root / "Dockerfile").write_text("FROM python:3.12\n", encoding="utf-8")
            (root / "requirements.txt").write_text("requests==2.32.3\n", encoding="utf-8")
            (root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "pyrightconfig.json").write_text("{}\n", encoding="utf-8")
            (root / "package.json").write_text(json.dumps({"scripts": {"test": "vitest"}}), encoding="utf-8")
            (root / "tsconfig.json").write_text("{}\n", encoding="utf-8")
            (root / ".eslintrc.json").write_text("{}\n", encoding="utf-8")
            (root / ".prettierrc").write_text("{}\n", encoding="utf-8")
            (root / "biome.json").write_text("{}\n", encoding="utf-8")
            (root / ".pre-commit-config.yaml").write_text("repos: []\n", encoding="utf-8")
            (root / "README.md").write_text("# Demo\n", encoding="utf-8")
            (root / "query.sql").write_text("select 1;\n", encoding="utf-8")
            (root / "schema.schema.json").write_text('{"$schema":"https://json-schema.org/draft/2020-12/schema"}\n', encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.discover_validators(limit=40)

        output = result["output"]
        self.assertTrue(result["ok"])
        self.assertIn("basedpyright", output)
        self.assertIn("actionlint", output)
        self.assertIn("bash -n script.sh", output)
        self.assertIn("shellcheck script.sh", output)
        self.assertIn("hadolint Dockerfile", output)
        self.assertIn("osv-scanner scan .", output)
        self.assertIn("tsc --noEmit", output)
        self.assertIn("eslint .", output)
        self.assertIn("prettier --check .", output)
        self.assertIn("biome check .", output)
        self.assertIn("run --all-files", output)
        self.assertIn("vendor.github-workflows", output)
        self.assertIn("--check-metaschema", output)
        self.assertIn("yamllint", output)
        self.assertIn("shfmt -d script.sh", output)
        self.assertIn("markdownlint-cli2", output)
        self.assertIn("codespell .", output)
        self.assertIn("sqlfluff lint .", output)
        self.assertIn("pip", output)
        self.assertIn("audit", output)
        self.assertIn("trivy fs", output)
        self.assertIn("grype dir:.", output)

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
        with self._temp_files_tools({"pyproject.toml": "[project]\nname='demo'\n"}) as (_root, tools):
            result = tools.diagnose_dependency_error("ModuleNotFoundError: No module named 'requests'")

        self.assertTrue(result["ok"])
        self.assertEqual(result["missing_dependency"], "requests")
        self.assertIn("pip", result["package_managers"])

    def test_browser_and_security_missing_dependencies_fail_closed(self) -> None:
        with self._temp_tools() as (_root, tools):
            with patch("ollama_code.tools.subprocess.run", return_value=subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="missing")):
                browser = tools.browser_smoke("http://127.0.0.1")
            with patch("ollama_code.tools.shutil.which", return_value=None):
                security = tools.security_scan()

        self.assertFalse(browser["ok"])
        self.assertEqual(browser["missing_dependency"], "playwright")
        self.assertFalse(security["ok"])
        self.assertIn("missing_dependency", security)

    def test_find_implementation_target_maps_test_imports_to_source(self) -> None:
        with self._temp_files_tools(
            {
                "ops.py": "def add(a, b):\n    return a + b\n",
                "tests/test_ops.py": "from ops import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
            }
        ) as (_root, tools):
            result = tools.find_implementation_target(test_path="tests/test_ops.py")

        self.assertTrue(result["ok"])
        self.assertIn("ops.py", result["output"])
        self.assertIn("symbol=add", result["output"])

    def test_find_implementation_target_accepts_path_alias_for_source(self) -> None:
        with self._temp_files_tools({"src/ops.py": "def add(a, b):\n    return a + b\n"}) as (_root, tools):
            result = tools.find_implementation_target(path="src/ops.py", query="add")

        self.assertTrue(result["ok"])
        self.assertIn("src/ops.py", result["output"])
        self.assertIn("symbol=add", result["output"])

    def test_diagnose_test_failure_groups_assertions_and_targets(self) -> None:
        with self._temp_files_tools(
            {
                "ops.py": "def add(a, b):\n    return a - b\n",
                "tests/test_ops.py": "from ops import add\n",
            }
        ) as (_root, tools):
            output = (
                "FAILED tests/test_ops.py::test_add - AssertionError\n"
                "E       assert 1 == 3\n"
                '  File "ops.py", line 2, in add\n'
            )
            result = tools.diagnose_test_failure(output)

        self.assertTrue(result["ok"])
        self.assertIn("assertion mismatch", result["output"])
        self.assertIn("ops.py", result["output"])

    def test_diagnose_test_failure_maps_unittest_traceback_imports_to_source(self) -> None:
        with self._temp_files_tools(
            {
                "list_ops.py": "def foldr(function, items, initial):\n    return initial\n",
                "list_ops_test.py": (
                    "from list_ops import foldr\n\n"
                    "def test_foldr():\n"
                    "    assert foldr(lambda acc, el: el + acc, ['e'], '!') == 'e!'\n"
                ),
            }
        ) as (root, tools):
            test_file = root / "list_ops_test.py"
            output = (
                "FAIL: test_foldr (list_ops_test.ListOpsTest.test_foldr)\n"
                "Traceback (most recent call last):\n"
                f'  File "{test_file}", line 4, in test_foldr\n'
                "AssertionError: '!e' != 'e!'\n"
            )
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
        with self._temp_files_tools(
            {
                "app.py": (
                    "def helper(value):\n"
                    "    return value + 1\n\n"
                    "def target(value):\n"
                    "    return helper(value)\n\n"
                    "def caller():\n"
                    "    return target(1)\n"
                )
            }
        ) as (_root, tools):
            result = tools.call_graph("app.py", "target")

        self.assertTrue(result["ok"])
        self.assertIn("callees: helper", result["output"])
        self.assertIn("caller", result["output"])

    def test_contract_graph_reports_contracts_and_purity(self) -> None:
        with self._temp_files_tools(
            {
                "app.py": (
                    "import subprocess\n\n"
                    "def helper(value: int) -> int:\n"
                    "    return value + 1\n\n"
                    "def target(value: int) -> int:\n"
                    "    return helper(value)\n\n"
                    "def caller() -> int:\n"
                    "    return target(1)\n\n"
                    "def shell() -> None:\n"
                    "    subprocess.run(['echo', 'x'])\n"
                )
            }
        ) as (_root, tools):
            result = tools.contract_graph("app.py", "target")
            all_result = tools.contract_graph("app.py")

        self.assertTrue(result["ok"])
        self.assertIn("target(value: int)->int", result["output"])
        self.assertIn("callees: helper", result["output"])
        self.assertIn("caller", result["output"])
        self.assertIn("pure_hint", result["output"])
        self.assertIn("shell", all_result["output"])
        self.assertIn("impure_hint", all_result["output"])

    def test_verified_function_index_search_and_show_cards(self) -> None:
        with self._temp_files_tools(
            {
                "utils.py": '''
def normalize_phone(value: str) -> str:
    """Normalize phone digits for lookup."""
    return "".join(ch for ch in value if ch.isdigit())

def noisy(value: str) -> str:
    print(value)
    return value
'''.lstrip()
            }
        ) as (_root, tools):
            indexed = tools.verified_function_index()
            search = tools.verified_function_search("phone lookup")
            card_id = search["cards"][0]["id"]
            shown = tools.verified_function_show(card_id)

        self.assertTrue(indexed["ok"])
        self.assertGreaterEqual(indexed["cards"], 2)
        self.assertTrue(search["ok"])
        self.assertIn("normalize_phone", search["output"])
        self.assertEqual(search["cards"][0]["proof_level"], "probable")
        self.assertTrue(shown["ok"])
        self.assertFalse(shown["stale"])
        self.assertIn("Normalize phone digits", shown["output"])

    def test_promote_verified_function_with_docstring_probe_and_stale_detection(self) -> None:
        with self._temp_files_tools(
            {
                "math_utils.py": '''
def double(value: int) -> int:
    """Double an integer.

    >>> double(3)
    6
    """
    return value * 2
'''.lstrip()
            }
        ) as (root, tools):
            target = root / "math_utils.py"
            promoted = tools.promote_verified_function("math_utils.py", "double")
            shown_fresh = tools.verified_function_show(promoted["id"])
            target.write_text(target.read_text(encoding="utf-8").replace("value * 2", "value * 3"), encoding="utf-8")
            shown_stale = tools.verified_function_show(promoted["id"])

        self.assertTrue(promoted["ok"], promoted.get("summary"))
        self.assertEqual(promoted["card"]["proof_level"], "verified")
        self.assertFalse(shown_fresh["stale"])
        self.assertTrue(shown_stale["stale"])

    def test_verify_function_contract_rejects_failing_docstring_probe(self) -> None:
        with self._temp_files_tools(
            {
                "math_utils.py": '''
def double(value: int) -> int:
    """Double an integer.

    >>> double(3)
    7
    """
    return value * 2
'''.lstrip()
            }
        ) as (_root, tools):
            result = tools.verify_function_contract("math_utils.py", "double")

        self.assertFalse(result["ok"])
        self.assertIn("expected 7 got 6", result["summary"])

    def test_promote_verified_function_with_focused_unittest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "math_utils.py").write_text(
                "def triple(value: int) -> int:\n"
                "    return value * 3\n",
                encoding="utf-8",
            )
            (root / "test_math_utils.py").write_text(
                "import unittest\n"
                "from math_utils import triple\n\n"
                "class MathTests(unittest.TestCase):\n"
                "    def test_triple(self):\n"
                "        self.assertEqual(triple(3), 9)\n",
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto")

            result = tools.promote_verified_function("math_utils.py", "triple")

        self.assertTrue(result["ok"], result.get("summary"))
        self.assertEqual(result["card"]["proof_level"], "verified")
        self.assertIn("focused_test=pass", result["card"]["properties"])

    def test_compose_verified_functions_labels_unverified_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "utils.py").write_text(
                "def strip_name(value: str) -> str:\n"
                "    return value.strip()\n",
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto")
            search = tools.verified_function_search("strip name")
            card_id = search["cards"][0]["id"]

            result = tools.compose_verified_functions("clean a customer name", [card_id])

        self.assertTrue(result["ok"])
        self.assertIn("strip_name", result["output"])
        self.assertIn("Missing adapters/verification", result["output"])

    def test_verify_function_contract_reports_missing_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "utils.py").write_text("def present():\n    return 1\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")

            result = tools.verify_function_contract("utils.py", "missing")

        self.assertFalse(result["ok"])
        self.assertIn("No Python function symbol", result["summary"])

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
        self.assertIn("calls total with 1 supplied args", result["output"])

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

    def test_contract_check_allows_returning_local_dict_variable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text(
                "def fetch_user(user_id: str) -> dict[str, str]:\n"
                "    user_data = {'id': user_id}\n"
                "    return user_data\n",
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.contract_check(["app.py"])

        self.assertTrue(result["ok"], result["output"])
        self.assertNotIn("returns shape name:user_data", result["output"])

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

    def test_lint_typecheck_runs_bash_n_for_shell_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "script.sh").write_text("if true; then\n  echo ok\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=2,
                stdout="",
                stderr="script.sh: line 3: syntax error: unexpected end of file",
            )
            with patch("ollama_code.tools.shutil.which", side_effect=lambda name: "bash" if name == "bash" else None):
                with patch.object(tools, "_run_process", return_value=completed) as run_process:
                    result = tools.lint_typecheck("script.sh")

        self.assertFalse(result["ok"])
        self.assertIn("bash -n script.sh", result["validator_commands"])
        self.assertIn("unexpected end of file", result["output"])
        self.assertIn("-n", run_process.call_args.args[0])

    def test_lint_typecheck_shell_only_target_skips_python_validators(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "script.sh").write_text("echo ok\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            calls: list[list[str]] = []

            def fake_run(command: list[str], cwd: Path, timeout: int, shell: bool) -> subprocess.CompletedProcess[str]:
                calls.append(list(command))
                return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

            with patch("ollama_code.tools.shutil.which", side_effect=lambda name: name if name in {"ruff", "bash"} else None):
                with patch.object(tools, "_python_tool_command", return_value=["basedpyright", "--level", "error"]):
                    with patch.object(tools, "_run_process", side_effect=fake_run):
                        result = tools.lint_typecheck("script.sh")

        self.assertTrue(result["ok"], result["output"])
        self.assertEqual(result["validator_commands"], ["bash -n script.sh"])
        self.assertEqual(calls, [["bash", "-n", "script.sh"]])

    def test_lint_typecheck_returns_structured_timeout_for_python_validator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pkg.py").write_text("print('ok')\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            timeout = subprocess.TimeoutExpired(["ruff", "check", "--no-cache", "pkg.py"], 1, output="partial stdout", stderr="partial stderr")
            with patch("ollama_code.tools.shutil.which", side_effect=lambda name: "ruff" if name == "ruff" else None):
                with patch.object(tools, "_run_process", side_effect=timeout):
                    result = tools.lint_typecheck("pkg.py", timeout=1)

        self.assertFalse(result["ok"])
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["error_class"], "timeout")
        self.assertEqual(result["summary"], "Command timed out after 1 seconds.")
        self.assertEqual(result["command"], "ruff check --no-cache pkg.py")
        self.assertIn("ruff check --no-cache pkg.py", result["validator_commands"])
        self.assertIn("partial stdout", result["output"])
        self.assertIn("partial stderr", result["output"])

    def test_lint_typecheck_returns_structured_timeout_for_shell_validator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "script.sh").write_text("echo ok\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            timeout = subprocess.TimeoutExpired(["bash", "-n", "script.sh"], 1, output="", stderr="shell partial stderr")
            with patch("ollama_code.tools.shutil.which", side_effect=lambda name: "bash" if name == "bash" else None):
                with patch.object(tools, "_python_tool_command", return_value=None):
                    with patch.object(tools, "_run_process", side_effect=timeout):
                        result = tools.lint_typecheck("script.sh", timeout=1)

        self.assertFalse(result["ok"])
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["error_class"], "timeout")
        self.assertEqual(result["summary"], "Command timed out after 1 seconds.")
        self.assertEqual(result["command"], "bash -n script.sh")
        self.assertIn("bash -n script.sh", result["validator_commands"])
        self.assertIn("shell partial stderr", result["output"])

    def test_lint_typecheck_collapses_validator_targets_to_requested_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "one.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "src" / "two.py").write_text("OTHER = 2\n", encoding="utf-8")
            (root / "pyrightconfig.json").write_text("{}\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            calls: list[list[str]] = []

            def fake_run(command: list[str], cwd: Path, timeout: int, shell: bool) -> subprocess.CompletedProcess[str]:
                calls.append(list(command))
                return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

            with patch("ollama_code.tools.shutil.which", side_effect=lambda name: "ruff" if name == "ruff" else None):
                with patch.object(tools, "_python_tool_command", return_value=["basedpyright", "--level", "error"]):
                    with patch.object(tools, "_run_process", side_effect=fake_run):
                        result = tools.lint_typecheck("src")

        self.assertTrue(result["ok"], result["output"])
        self.assertEqual(result["validator_targets"], ["src/one.py", "src/two.py"])
        self.assertEqual(result["typechecker_targets"], ["src"])
        self.assertEqual(
            result["validator_commands"],
            [
                "ruff check --no-cache src/one.py src/two.py",
                "basedpyright --level error src",
            ],
        )
        self.assertEqual(
            calls,
            [
                ["ruff", "check", "--no-cache", "src/one.py", "src/two.py"],
                ["basedpyright", "--level", "error", "src"],
            ],
        )
        self.assertGreaterEqual(float(result["scan_ms"]), 0.0)
        self.assertGreaterEqual(float(result["ruff_ms"]), 0.0)
        self.assertGreaterEqual(float(result["typecheck_ms"]), 0.0)
        self.assertEqual(float(result["shell_ms"]), 0.0)

    def test_lint_typecheck_preserves_workspace_scope_for_configured_full_repo_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "one.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "pyrightconfig.json").write_text("{}\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            calls: list[list[str]] = []

            def fake_run(command: list[str], cwd: Path, timeout: int, shell: bool) -> subprocess.CompletedProcess[str]:
                calls.append(list(command))
                return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

            with patch("ollama_code.tools.shutil.which", side_effect=lambda name: "ruff" if name == "ruff" else None):
                with patch.object(tools, "_python_tool_command", return_value=["basedpyright", "--level", "error"]):
                    with patch.object(tools, "_run_process", side_effect=fake_run):
                        result = tools.lint_typecheck(".")

        self.assertTrue(result["ok"], result["output"])
        self.assertEqual(result["validator_targets"], ["."])
        self.assertEqual(result["typechecker_targets"], ["."])
        self.assertEqual(
            calls,
            [["ruff", "check", "--no-cache", "."], ["basedpyright", "--level", "error", "."]],
        )

    def test_lint_typecheck_skips_typechecker_for_unconfigured_test_only_workspace_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "tests" / "test_sample.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            calls: list[list[str]] = []

            def fake_run(command: list[str], cwd: Path, timeout: int, shell: bool) -> subprocess.CompletedProcess[str]:
                calls.append(list(command))
                return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

            with patch("ollama_code.tools.shutil.which", side_effect=lambda name: "ruff" if name == "ruff" else None):
                with patch.object(tools, "_python_tool_command", return_value=["basedpyright", "--level", "error"]) as python_tool:
                    with patch.object(tools, "_run_process", side_effect=fake_run):
                        result = tools.lint_typecheck(".")

        self.assertTrue(result["ok"], result["output"])
        self.assertEqual(result["validator_targets"], ["."])
        self.assertEqual(result["typechecker_targets"], [])
        self.assertIn("test-only workspace scope", result["typechecker_skipped_reason"])
        self.assertEqual(calls, [["ruff", "check", "--no-cache", "."]])
        self.assertEqual(python_tool.call_count, 0)

    def test_lint_typecheck_falls_back_to_requested_scope_for_large_python_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            src.mkdir()
            (root / "pyrightconfig.json").write_text("{}\n", encoding="utf-8")
            for index in range(25):
                (src / f"module_{index}.py").write_text(f"VALUE_{index} = {index}\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            calls: list[list[str]] = []

            def fake_run(command: list[str], cwd: Path, timeout: int, shell: bool) -> subprocess.CompletedProcess[str]:
                calls.append(list(command))
                return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

            with patch("ollama_code.tools.shutil.which", side_effect=lambda name: "ruff" if name == "ruff" else None):
                with patch.object(tools, "_python_tool_command", return_value=["basedpyright", "--level", "error"]):
                    with patch.object(tools, "_run_process", side_effect=fake_run):
                        result = tools.lint_typecheck("src")

        self.assertTrue(result["ok"], result["output"])
        self.assertEqual(result["validator_targets"], ["src"])
        self.assertEqual(result["typechecker_targets"], ["src"])
        self.assertEqual(
            calls,
            [["ruff", "check", "--no-cache", "src"], ["basedpyright", "--level", "error", "src"]],
        )

    def test_lint_typecheck_caches_unchanged_validator_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "pyrightconfig.json").write_text("{}\n", encoding="utf-8")
            target = root / "src" / "one.py"
            target.write_text("VALUE = 1\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            calls: list[list[str]] = []

            def fake_run(command: list[str], cwd: Path, timeout: int, shell: bool) -> subprocess.CompletedProcess[str]:
                calls.append(list(command))
                return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

            with patch("ollama_code.tools.shutil.which", side_effect=lambda name: "ruff" if name == "ruff" else None):
                with patch.object(tools, "_python_tool_command", return_value=["basedpyright", "--level", "error"]):
                    with patch.object(tools, "_run_process", side_effect=fake_run):
                        first = tools.lint_typecheck("src")
                        second = tools.lint_typecheck("src")
                        target.write_text("VALUE = 2\n", encoding="utf-8")
                        third = tools.lint_typecheck("src")

        self.assertTrue(first["ok"], first["output"])
        self.assertTrue(second["ok"], second["output"])
        self.assertTrue(third["ok"], third["output"])
        self.assertIsNone(first.get("cache_hit"))
        self.assertTrue(second.get("cache_hit"))
        self.assertIsNone(third.get("cache_hit"))
        self.assertEqual(
            calls,
            [
                ["ruff", "check", "--no-cache", "src/one.py"],
                ["basedpyright", "--level", "error", "src"],
                ["ruff", "check", "--no-cache", "src/one.py"],
                ["basedpyright", "--level", "error", "src"],
            ],
        )

    def test_lint_typecheck_skips_typechecker_for_unconfigured_focused_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "one.py").write_text("VALUE = 1\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            calls: list[list[str]] = []

            def fake_run(command: list[str], cwd: Path, timeout: int, shell: bool) -> subprocess.CompletedProcess[str]:
                calls.append(list(command))
                return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

            with patch("ollama_code.tools.shutil.which", side_effect=lambda name: "ruff" if name == "ruff" else None):
                with patch.object(tools, "_python_tool_command", return_value=["basedpyright", "--level", "error"]) as python_tool:
                    with patch.object(tools, "_run_process", side_effect=fake_run):
                        result = tools.lint_typecheck("src")

        self.assertTrue(result["ok"], result["output"])
        self.assertEqual(result["validator_targets"], ["src/one.py"])
        self.assertEqual(result["typechecker_targets"], [])
        self.assertIn("No pyright/basedpyright config", result["typechecker_skipped_reason"])
        self.assertEqual(calls, [["ruff", "check", "--no-cache", "src/one.py"]])
        self.assertEqual(python_tool.call_count, 0)

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

    def test_apply_structured_edit_changes_signature_from_bare_callable_signature(self) -> None:
        root = self._workspace_scratch()
        sample = root / "ops.py"
        sample.write_text("def fetch_user(user_id: str) -> dict[str, str]:\n    return {'id': user_id}\n", encoding="utf-8")
        tools = ToolExecutor(root, approval_mode="auto")
        result = tools.apply_structured_edit(
            {
                "op": "change_signature",
                "path": "ops.py",
                "symbol": "fetch_user",
                "signature": "fetch_user(user_id: str, include_orders: bool = False) -> dict[str, str]:",
            }
        )

        self.assertTrue(result["ok"])
        self.assertIn(
            "def fetch_user(user_id: str, include_orders: bool = False) -> dict[str, str]:",
            sample.read_text(encoding="utf-8"),
        )

    def test_apply_structured_edit_changes_signature_from_full_function_input(self) -> None:
        root = self._workspace_scratch()
        sample = root / "ops.py"
        sample.write_text("def fetch_user(user_id: str) -> dict[str, str]:\n    return {'id': user_id}\n", encoding="utf-8")
        tools = ToolExecutor(root, approval_mode="auto")
        result = tools.apply_structured_edit(
            {
                "op": "change_signature",
                "path": "ops.py",
                "symbol": "fetch_user",
                "signature": "def fetch_user(user_id: str, include_orders: bool = False) -> dict[str, str]:\n    return {'id': user_id'}",
            }
        )
        final_text = sample.read_text(encoding="utf-8")

        self.assertTrue(result["ok"])
        self.assertIn("def fetch_user(user_id: str, include_orders: bool = False) -> dict[str, str]:", final_text)
        self.assertIn("return {'id': user_id}", final_text)

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

    def test_execute_interrupt_for_run_shell_closes_process_pipes(self) -> None:
        root = self._workspace_scratch()
        tools = ToolExecutor(root, approval_mode="auto")
        interrupted = threading.Event()
        tools.set_interrupt_event(interrupted)
        trigger = threading.Timer(0.2, interrupted.set)
        trigger.start()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ResourceWarning)
                result = tools.execute("run_shell", {"command": f'"{sys.executable}" -c "import time; time.sleep(5)"', "timeout": 10})
                gc.collect()
        finally:
            trigger.cancel()

        self.assertFalse(result["ok"])
        self.assertTrue(result["interrupted"])
        self.assertFalse(
            [warning for warning in caught if issubclass(warning.category, ResourceWarning)],
            [str(warning.message) for warning in caught],
        )

    def test_run_shell_timeout_returns_structured_result_with_partial_output(self) -> None:
        root = self._workspace_scratch()
        tools = ToolExecutor(root, approval_mode="auto")
        command = f'"{sys.executable}" -c "import sys,time; print(123); sys.stdout.flush(); time.sleep(2)"'
        timeout_error = subprocess.TimeoutExpired(cmd=command, timeout=1, output="123\n")

        with patch.object(ToolExecutor, "_run_process", side_effect=timeout_error):
            result = tools.run_shell(command, timeout=1)

        self.assertFalse(result["ok"])
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["error_class"], "timeout")
        self.assertEqual(result["cwd"], ".")
        self.assertEqual(result["output"], "123")
        self.assertEqual(result["summary"], "Command timed out after 1 seconds.")

    def test_run_shell_ignores_posix_shell_env_override(self) -> None:
        tools = ToolExecutor(Path.cwd(), approval_mode="auto")
        completed = subprocess.CompletedProcess(args=["echo", "ok"], returncode=0, stdout="ok\n", stderr="")
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
        self.assertEqual(result["error_class"], "syntax_error")
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

    def test_run_shell_rejects_path_escape_for_unknown_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root.parent / "outside.txt"
            outside.write_text("secret\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            with patch.object(ToolExecutor, "_run_process") as run_mock:
                result = tools.run_shell("cat ../outside.txt")

        self.assertFalse(result["ok"])
        self.assertIn("path escapes workspace", result["summary"])
        run_mock.assert_not_called()

    def test_run_shell_rejects_executable_path_escape_for_unknown_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside_name = "outside-tool.bat" if os.name == "nt" else "outside-tool.sh"
            outside = root.parent / outside_name
            outside.write_text("@echo off\r\necho hi\r\n" if os.name == "nt" else "#!/bin/sh\necho hi\n", encoding="utf-8")
            command = f"..\\{outside_name} --help" if os.name == "nt" else f"../{outside_name} --help"
            tools = ToolExecutor(root, approval_mode="auto")
            with patch.object(ToolExecutor, "_run_process") as run_mock:
                result = tools.run_shell(command)

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
    def test_run_shell_routes_powershell_aliases_on_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.txt").write_text("hello\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            completed = subprocess.CompletedProcess(
                args=["pwsh", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", "cat note.txt"],
                returncode=0,
                stdout="hello\n",
                stderr="",
            )

            with patch.object(ToolExecutor, "_windows_powershell", return_value="pwsh"):
                with patch.object(ToolExecutor, "_run_process", return_value=completed) as run_mock:
                    result = tools.run_shell("cat note.txt")

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["output"], "hello")
        self.assertEqual(
            run_mock.call_args.args[0],
            ["pwsh", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", "cat note.txt"],
        )
        self.assertFalse(run_mock.call_args.kwargs["shell"])

    @unittest.skipUnless(os.name == "nt", "Windows only")
    def test_run_shell_routes_powershell_call_operator_on_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            command = '& ".\\script with spaces.ps1" -Name demo'
            completed = subprocess.CompletedProcess(
                args=["pwsh", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
                returncode=0,
                stdout="ok\n",
                stderr="",
            )

            with patch.object(ToolExecutor, "_windows_powershell", return_value="pwsh"):
                with patch.object(ToolExecutor, "_run_process", return_value=completed) as run_mock:
                    result = tools.run_shell(command)

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["output"], "ok")
        self.assertEqual(
            run_mock.call_args.args[0],
            ["pwsh", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
        )
        self.assertFalse(run_mock.call_args.kwargs["shell"])

    @unittest.skipUnless(os.name == "nt", "Windows only")
    def test_run_shell_does_not_misclassify_dollar_sign_inside_quoted_argument(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            command = f'{sys.executable} -c "print(\'a $HOME b\')"'
            result = tools.run_shell(command)
        self.assertTrue(result["ok"])
        self.assertEqual(result["output"], "a $HOME b")

    @unittest.skipUnless(os.name == "nt", "Windows only")
    def test_validate_common_command_preserves_inline_python_path_literals_on_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = ToolExecutor(root, approval_mode="auto")
            command = subprocess.list2cmdline(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('scratch/artifact.txt').write_text('ok\\n')",
                ]
            )

            argv, validation = tools._validate_common_command(command, root)

        self.assertIsNotNone(argv)
        assert argv is not None
        self.assertEqual(argv[2], "from pathlib import Path; Path('scratch/artifact.txt').write_text('ok\\n')")
        self.assertTrue(validation["valid"])

    @unittest.skipUnless(os.name == "nt", "Windows only")
    def test_run_shell_preserves_inline_python_path_literals_on_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scratch").mkdir()
            tools = ToolExecutor(root, approval_mode="auto")
            command = subprocess.list2cmdline(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('scratch/artifact.txt').write_text('ok\\n')",
                ]
            )

            result = tools.run_shell(command)
            artifact_exists = (root / "scratch" / "artifact.txt").is_file()

        self.assertTrue(result["ok"], result.get("output") or result.get("summary"))
        self.assertTrue(artifact_exists)

    def test_run_test_uses_configured_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = f'"{sys.executable}" -c "print(321)"'
            tools = ToolExecutor(root, approval_mode="auto", test_command=command)
            result = tools.run_test()
        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "run_test")
        self.assertEqual(result["output"], "321")

    def test_run_test_rejects_explicit_whitespace_command(self) -> None:
        root = self._workspace_scratch()
        command = f'"{sys.executable}" -c "print(321)"'
        tools = ToolExecutor(root, approval_mode="auto", test_command=command)

        result = tools.run_test("   ")

        self.assertFalse(result["ok"])
        self.assertEqual(result["tool"], "run_test")
        self.assertEqual(result["error_class"], "invalid_args")
        self.assertEqual(result["summary"], "Explicit test command must be a non-empty string.")

    def test_run_test_rejects_non_string_explicit_command(self) -> None:
        root = self._workspace_scratch()
        command = f'"{sys.executable}" -c "print(321)"'
        tools = ToolExecutor(root, approval_mode="auto", test_command=command)

        result = tools.run_test(command=123)  # type: ignore[arg-type]

        self.assertFalse(result["ok"])
        self.assertEqual(result["tool"], "run_test")
        self.assertEqual(result["error_class"], "invalid_args")
        self.assertEqual(result["summary"], "Explicit test command must be a non-empty string.")

    @unittest.skipUnless(os.name != "nt", "POSIX only")
    def test_validate_common_command_normalizes_windows_style_paths_on_posix(self) -> None:
        root = self._workspace_scratch()
        tools = ToolExecutor(root, approval_mode="auto")

        with patch.object(ToolExecutor, "_executable_available_for_cwd", return_value=True):
            argv, validation = tools._validate_common_command(
                r"C:\Python312\python.exe -m unittest discover -s tests\unit -v",
                root,
            )

        self.assertEqual(
            argv,
            ["/mnt/c/Python312/python.exe", "-m", "unittest", "discover", "-s", "tests/unit", "-v"],
        )
        self.assertTrue(validation["valid"])

    def test_validate_common_command_allows_cargo_nextest_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = ToolExecutor(root, approval_mode="auto")

            with patch.object(ToolExecutor, "_executable_available_for_cwd", return_value=True):
                argv, validation = tools._validate_common_command("cargo nextest run", root)

        self.assertEqual(argv, ["cargo", "nextest", "run"])
        self.assertTrue(validation["valid"])

    @unittest.skipUnless(os.name != "nt", "POSIX only")
    def test_run_test_normalizes_backslash_relative_path_argument_on_posix(self) -> None:
        root = self._workspace_scratch()
        test_dir = root / "tests" / "unit"
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / "test_sample.py").write_text(
            "import unittest\n\nclass SampleTests(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n",
            encoding="utf-8",
        )
        tools = ToolExecutor(root, approval_mode="auto")

        result = tools.run_test(f"{sys.executable} -m unittest discover -s tests\\unit -v")

        self.assertTrue(result["ok"], result.get("summary") or result.get("output"))
        self.assertEqual(
            result["command"],
            f"{sys.executable} -m unittest discover -s tests\\unit -v",
        )
        self.assertIn("OK", result["output"])

    def test_run_test_timeout_returns_structured_result_with_command(self) -> None:
        root = self._workspace_scratch()
        command = f'"{sys.executable}" -c "import sys,time; print(321); sys.stdout.flush(); time.sleep(2)"'
        tools = ToolExecutor(root, approval_mode="auto", test_command=command)
        timeout_error = subprocess.TimeoutExpired(cmd=command, timeout=1, output="321\n")

        with patch.object(ToolExecutor, "_run_process", side_effect=timeout_error):
            result = tools.run_test(timeout=1)

        self.assertFalse(result["ok"])
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["tool"], "run_test")
        self.assertEqual(result["command"], command)
        self.assertEqual(result["output"], "321")
        self.assertEqual(result["summary"], "Command timed out after 1 seconds.")

    def test_run_test_reports_inline_python_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = subprocess.list2cmdline([sys.executable, "-c", "import sys; print('AssertionError: None != 3'); sys.exit(1)"])
            tools = ToolExecutor(root, approval_mode="auto", test_command=command)
            result = tools.run_test()
        self.assertFalse(result["ok"])
        self.assertEqual(result["tool"], "run_test")
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("AssertionError: None != 3", result["output"])

    def test_run_test_allows_unittest_discover_dot_start_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample_test.py").write_text(
                "import unittest\n\nclass SampleTests(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n",
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.run_test(f"{sys.executable} -m unittest discover -s . -p *_test.py -v")

        self.assertTrue(result["ok"], result.get("summary") or result.get("output"))
        self.assertEqual(result["tool"], "run_test")
        self.assertNotIn("path escapes workspace", str(result.get("summary", "")))

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

    def test_run_test_recovers_from_broken_configured_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "tests" / "test_sample.py").write_text(
                "import unittest\n\nclass SampleTests(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n",
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto", test_command="pytesst -q")
            result = tools.run_test()

        self.assertTrue(result["ok"], result.get("output"))
        self.assertTrue(result["recovered"])
        self.assertEqual(result["original_command"], "pytesst -q")
        self.assertIn("unittest discover", result["command"])
        self.assertEqual(tools.default_test_command, result["command"])

    def test_run_test_does_not_recover_from_normal_test_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("def add(left, right):\n    pass\n", encoding="utf-8")
            (root / "app_test.py").write_text(
                "import unittest\nfrom app import add\n\n"
                "class SampleTests(unittest.TestCase):\n"
                "    def test_add(self):\n"
                "        self.assertEqual(add(1, 2), 3)\n",
                encoding="utf-8",
            )
            command = subprocess.list2cmdline([sys.executable, "-m", "unittest", "discover", "-p", "*_test.py", "-v"])
            tools = ToolExecutor(root, approval_mode="auto", test_command=command)
            result = tools.run_test()

        self.assertFalse(result["ok"])
        self.assertNotIn("recovered", result)
        self.assertEqual(result["command"], command)
        self.assertIn("AssertionError", result["output"])
        self.assertEqual(tools.default_test_command, command)

    def test_run_test_does_not_recover_from_application_import_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "tests" / "test_app.py").write_text(
                "import unittest\nfrom missing_app_module import value\n\n"
                "class SampleTests(unittest.TestCase):\n"
                "    def test_value(self):\n"
                "        self.assertEqual(value(), 1)\n",
                encoding="utf-8",
            )
            command = subprocess.list2cmdline([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
            tools = ToolExecutor(root, approval_mode="auto", test_command=command)
            result = tools.run_test()

        self.assertFalse(result["ok"])
        self.assertNotIn("recovered", result)
        self.assertEqual(result["command"], command)
        self.assertIn("ModuleNotFoundError", result["output"])
        self.assertEqual(tools.default_test_command, command)

    def test_run_test_sees_immediate_same_size_python_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "tests").mkdir()
            (root / "src" / "balance.py").write_text(
                "def apply_credit(amount: int, credit: int) -> int:\n    return amount - credit\n",
                encoding="utf-8",
            )
            (root / "tests" / "test_balance_credit.py").write_text(
                "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\nfrom balance import *\nimport unittest\n\n"
                "class BalanceTests(unittest.TestCase):\n"
                "    def test_apply_credit(self):\n"
                "        self.assertEqual(apply_credit(10, 3), 13)\n",
                encoding="utf-8",
            )
            command = subprocess.list2cmdline([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
            tools = ToolExecutor(root, approval_mode="auto", test_command=command)

            first = tools.run_test()
            write_result = tools.write_file(
                "src/balance.py",
                "def apply_credit(amount: int, credit: int) -> int:\n    return amount + credit\n",
            )
            second = tools.run_test()

        self.assertFalse(first["ok"])
        self.assertTrue(write_result["ok"])
        self.assertTrue(second["ok"], second.get("output"))

    def test_run_test_does_not_override_explicit_bad_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "tests" / "test_sample.py").write_text(
                "import unittest\n\nclass SampleTests(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n",
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto")
            result = tools.run_test("pytesst -q")

        self.assertFalse(result["ok"])
        self.assertNotIn("recovered", result)
        self.assertEqual(result["command"], "pytesst -q")

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
            self.init_git_repo_with_commit(root, {"tracked.txt": "before\n"})
            tracked = root / "tracked.txt"
            tracked.write_text("before\nafter\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            status = tools.git_status()
            diff = tools.git_diff(path="tracked.txt")

        self.assertTrue(status["ok"])
        self.assertIn("tracked.txt", status["output"])
        self.assertTrue(diff["ok"])
        self.assertIn("+after", diff["output"])

    def test_git_branch_and_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.init_git_repo_with_commit(root, {"tracked.txt": "before\n"})
            tracked = root / "tracked.txt"
            subprocess.run(["git", "checkout", "-b", "feature"], cwd=root, check=True, capture_output=True, text=True)
            tracked.write_text("after\n", encoding="utf-8")
            subprocess.run(["git", "commit", "-am", "feature update"], cwd=root, check=True, capture_output=True, text=True)
            tools = ToolExecutor(root, approval_mode="auto")
            branches = tools.git_branch(all_branches=True)
            history = tools.git_log(max_count=5)

        self.assertTrue(branches["ok"])
        self.assertIn("feature", branches["output"])
        self.assertIn("master", branches["output"])
        self.assertTrue(history["ok"])
        self.assertIn("feature update", history["output"])

    def test_git_commit_creates_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.init_git_repo_with_commit(root, {"tracked.txt": "before\n"})
            tracked = root / "tracked.txt"
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
            self.init_git_repo_with_commit(root, {"tracked.txt": "before\n", "notes.txt": "draft\n"})
            tracked = root / "tracked.txt"
            unrelated = root / "notes.txt"
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
            self.init_git_repo_with_commit(root, {"tracked.txt": "before\n"})
            tracked = root / "tracked.txt"
            tracked.write_text("before\nafter\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="read-only")
            result = tools.git_commit("Blocked commit")
        self.assertFalse(result["ok"])
        self.assertIn("read-only", result["summary"])
