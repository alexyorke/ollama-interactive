from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
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
        subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, text=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True, capture_output=True, text=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True, capture_output=True, text=True)

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

    def test_run_test_requires_configured_or_explicit_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            result = tools.run_test()
        self.assertFalse(result["ok"])
        self.assertIn("No test command", result["summary"])

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
