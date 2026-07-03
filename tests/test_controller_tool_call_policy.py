import tempfile
import unittest
from pathlib import Path

from ollama_code.agent_protocol import TargetLineReadSpec
from ollama_code.controller.tool_call_policy import (
    normalize_find_shell_inspection,
    normalize_head_tail_shell_inspection,
    normalize_import_repair_bootstrap_call,
    normalize_optional_parameter_bootstrap_call,
    normalize_project_rename_bootstrap_call,
    normalize_run_test_call,
    normalize_shell_inspection_call,
    normalize_shell_test_call,
    normalize_target_line_read_call,
    normalize_unittest_file_command,
)


class ControllerToolCallPolicyTests(unittest.TestCase):
    def test_target_line_read_call_narrows_broad_read_file(self) -> None:
        spec = TargetLineReadSpec(path="src/app.py", start=7, end=17, line=12)

        self.assertEqual(
            normalize_target_line_read_call("read_file", {"path": "src/app.py", "start": 1, "end": 200}, target_line_read=spec),
            (
                "read_file",
                {"path": "src/app.py", "start": 7, "end": 17},
                "Normalized read_file to the requested line 12 with a small surrounding range.",
            ),
        )
        self.assertEqual(
            normalize_target_line_read_call("read_file", {"path": "src/app.py", "start": 10, "end": 15}, target_line_read=spec),
            ("read_file", {"path": "src/app.py", "start": 10, "end": 15}, None),
        )
        self.assertEqual(
            normalize_target_line_read_call("read_file", {"path": "other.py"}, target_line_read=spec),
            ("read_file", {"path": "other.py"}, None),
        )

    def test_run_test_call_normalizes_aliases_and_vague_commands(self) -> None:
        normalizer = lambda command: None

        self.assertEqual(
            normalize_run_test_call(
                "pytest",
                {"timeout": 10},
                request_text="Run tests.",
                default_test_command="python -m unittest discover -s tests -v",
                normalize_unittest_file_command=normalizer,
            ),
            (
                "run_test",
                {"timeout": 10, "command": "python -m unittest discover -s tests -v"},
                "Normalized pytest tool alias to the configured run_test command.",
            ),
        )
        self.assertEqual(
            normalize_run_test_call(
                "run_test",
                {"command": "pytest"},
                request_text="Run the test suite.",
                default_test_command="python -m unittest discover -s tests -v",
                normalize_unittest_file_command=normalizer,
            ),
            (
                "run_test",
                {"command": "python -m unittest discover -s tests -v"},
                "Normalized vague run_test command to the configured test command.",
            ),
        )
        self.assertEqual(
            normalize_run_test_call(
                "run_test",
                {"command": "python -m unittest tests.test_cli"},
                request_text="Run python -m unittest tests.test_cli.",
                default_test_command="python -m unittest discover -s tests -v",
                normalize_unittest_file_command=normalizer,
            ),
            ("run_test", {"command": "python -m unittest tests.test_cli"}, None),
        )

    def test_run_test_call_uses_unittest_file_command_normalizer(self) -> None:
        self.assertEqual(
            normalize_run_test_call(
                "run_test",
                {"command": "python -m unittest tests/test_cli.py"},
                request_text="Run tests.",
                default_test_command="python -m unittest discover -s tests -v",
                normalize_unittest_file_command=lambda command: "python -m unittest discover -s tests -p test_cli.py",
            ),
            (
                "run_test",
                {"command": "python -m unittest discover -s tests -p test_cli.py"},
                "Normalized unittest file path command to unittest discover.",
            ),
        )

    def test_shell_test_call_honors_explicit_tool_requests(self) -> None:
        self.assertEqual(
            normalize_shell_test_call(
                "run_shell",
                {"command": "pytest -q", "timeout": 30},
                approval_mode="default",
                explicit_run_shell=True,
                explicit_run_test=False,
                exact_shell_command=None,
                default_test_command="python -m unittest discover -s tests -v",
                bare_python_test_file_command=lambda command: None,
                shell_command_looks_like_test_run=lambda command: True,
            ),
            ("run_shell", {"command": "pytest -q", "timeout": 30}, None),
        )
        self.assertEqual(
            normalize_shell_test_call(
                "run_shell",
                {"command": "pytest -q", "cwd": "pkg"},
                approval_mode="default",
                explicit_run_shell=False,
                explicit_run_test=True,
                exact_shell_command=None,
                default_test_command="python -m unittest discover -s tests -v",
                bare_python_test_file_command=lambda command: None,
                shell_command_looks_like_test_run=lambda command: True,
            ),
            (
                "run_test",
                {"command": "pytest -q", "cwd": "pkg"},
                "Normalized run_shell to run_test because the request explicitly requires run_test.",
            ),
        )

    def test_shell_test_call_normalizes_test_like_shell_commands(self) -> None:
        self.assertEqual(
            normalize_shell_test_call(
                "run_shell",
                {"command": "pytest -q", "timeout": 30},
                approval_mode="default",
                explicit_run_shell=False,
                explicit_run_test=False,
                exact_shell_command=None,
                default_test_command="python -m unittest discover -s tests -v",
                bare_python_test_file_command=lambda command: None,
                shell_command_looks_like_test_run=lambda command: True,
            ),
            (
                "run_test",
                {"command": "python -m unittest discover -s tests -v", "timeout": 30},
                "Normalized shell test command to the configured run_test command.",
            ),
        )
        self.assertEqual(
            normalize_shell_test_call(
                "run_shell",
                {"command": "pytest -q"},
                approval_mode="default",
                explicit_run_shell=False,
                explicit_run_test=False,
                exact_shell_command=None,
                default_test_command=None,
                bare_python_test_file_command=lambda command: None,
                shell_command_looks_like_test_run=lambda command: True,
            ),
            ("run_test", {"command": "pytest -q"}, "Normalized shell test command to run_test with the original command."),
        )

    def test_shell_test_call_preserves_non_test_read_only_and_exact_commands(self) -> None:
        base_kwargs = {
            "explicit_run_shell": False,
            "explicit_run_test": False,
            "default_test_command": "python -m unittest discover -s tests -v",
            "bare_python_test_file_command": lambda command: None,
            "shell_command_looks_like_test_run": lambda command: command.startswith("pytest"),
        }
        self.assertEqual(
            normalize_shell_test_call(
                "run_shell",
                {"command": "pytest -q"},
                approval_mode="read-only",
                exact_shell_command=None,
                **base_kwargs,
            ),
            ("run_shell", {"command": "pytest -q"}, None),
        )
        self.assertEqual(
            normalize_shell_test_call(
                "run_shell",
                {"command": "pytest -q"},
                approval_mode="default",
                exact_shell_command="pytest -q",
                **base_kwargs,
            ),
            ("run_shell", {"command": "pytest -q"}, None),
        )
        self.assertEqual(
            normalize_shell_test_call(
                "run_shell",
                {"command": "python script.py"},
                approval_mode="default",
                exact_shell_command=None,
                **base_kwargs,
            ),
            ("run_shell", {"command": "python script.py"}, None),
        )

    def test_shell_test_call_normalizes_bare_python_test_file(self) -> None:
        self.assertEqual(
            normalize_shell_test_call(
                "run_shell",
                {"command": "tests/test_cli.py"},
                approval_mode="default",
                explicit_run_shell=False,
                explicit_run_test=False,
                exact_shell_command=None,
                default_test_command=None,
                bare_python_test_file_command=lambda command: "tests/test_cli.py",
                shell_command_looks_like_test_run=lambda command: False,
            ),
            (
                "run_test",
                {"command": "python -m pytest tests/test_cli.py"},
                "Normalized bare Python test-file shell command to run_test with pytest.",
            ),
        )
        self.assertEqual(
            normalize_shell_test_call(
                "run_shell",
                {"command": "tests/test_cli.py"},
                approval_mode="default",
                explicit_run_shell=False,
                explicit_run_test=False,
                exact_shell_command=None,
                default_test_command="python -m unittest discover -s tests -v",
                bare_python_test_file_command=lambda command: "tests/test_cli.py",
                shell_command_looks_like_test_run=lambda command: False,
            ),
            (
                "run_test",
                {"command": "python -m unittest discover -s tests -v"},
                "Normalized bare Python test-file shell command to the configured run_test command.",
            ),
        )

    def test_shell_inspection_call_normalizes_file_and_directory_reads(self) -> None:
        kwargs = {
            "approval_mode": "default",
            "explicit_run_shell": False,
            "exact_shell_command": None,
            "normalize_find_exec_grep_shell_command": lambda command: None,
            "normalize_grep_shell_inspection": lambda argv: None,
            "normalize_head_tail_shell_inspection": lambda argv: None,
            "normalize_find_shell_inspection": lambda argv: None,
        }
        self.assertEqual(
            normalize_shell_inspection_call("run_shell", {"command": "cat README.md"}, **kwargs),
            ("read_file", {"path": "README.md"}, "Normalized shell file inspection to read_file for cacheable structured context."),
        )
        self.assertEqual(
            normalize_shell_inspection_call("run_shell", {"command": "type -n docs/guide.md"}, **kwargs),
            ("read_file", {"path": "docs/guide.md"}, "Normalized shell file inspection to read_file for cacheable structured context."),
        )
        self.assertEqual(
            normalize_shell_inspection_call("run_shell", {"command": "ls src"}, **kwargs),
            ("list_files", {"path": "src"}, "Normalized shell directory inspection to list_files for cacheable structured context."),
        )
        self.assertEqual(
            normalize_shell_inspection_call("run_shell", {"command": "dir \"my docs\""}, **kwargs),
            ("list_files", {"path": "my docs"}, "Normalized shell directory inspection to list_files for cacheable structured context."),
        )

    def test_shell_inspection_call_preserves_explicit_or_unsafe_shell(self) -> None:
        kwargs = {
            "normalize_find_exec_grep_shell_command": lambda command: None,
            "normalize_grep_shell_inspection": lambda argv: None,
            "normalize_head_tail_shell_inspection": lambda argv: None,
            "normalize_find_shell_inspection": lambda argv: None,
        }
        self.assertEqual(
            normalize_shell_inspection_call(
                "run_shell",
                {"command": "cat README.md"},
                approval_mode="default",
                explicit_run_shell=True,
                exact_shell_command=None,
                **kwargs,
            ),
            ("run_shell", {"command": "cat README.md"}, None),
        )
        self.assertEqual(
            normalize_shell_inspection_call(
                "run_shell",
                {"command": "cat README.md", "cwd": "docs"},
                approval_mode="default",
                explicit_run_shell=False,
                exact_shell_command=None,
                **kwargs,
            ),
            ("run_shell", {"command": "cat README.md", "cwd": "docs"}, None),
        )
        self.assertEqual(
            normalize_shell_inspection_call(
                "run_shell",
                {"command": "cat README.md | head"},
                approval_mode="default",
                explicit_run_shell=False,
                exact_shell_command=None,
                **kwargs,
            ),
            ("run_shell", {"command": "cat README.md | head"}, None),
        )
        self.assertEqual(
            normalize_shell_inspection_call(
                "run_shell",
                {"command": "cat README.md"},
                approval_mode="read-only",
                explicit_run_shell=False,
                exact_shell_command=None,
                **kwargs,
            ),
            ("run_shell", {"command": "cat README.md"}, None),
        )

    def test_shell_inspection_call_uses_parser_callbacks(self) -> None:
        def normalize_grep(argv: list[str]) -> dict[str, object] | None:
            if argv[:2] == ["grep", "needle"]:
                return {"query": "needle", "path": ".", "limit": 20}
            return None

        def normalize_head_tail(argv: list[str]) -> dict[str, object] | None:
            if argv[:2] == ["head", "README.md"]:
                return {"path": "README.md", "start": 1, "end": 10}
            return None

        def normalize_find(argv: list[str]) -> tuple[str, dict[str, object]] | None:
            if argv[:2] == ["find", "."]:
                return "file_search", {"query": ".py", "path": ".", "limit": 100}
            return None

        base_kwargs = {
            "approval_mode": "default",
            "explicit_run_shell": False,
            "exact_shell_command": None,
            "normalize_find_exec_grep_shell_command": lambda command: {"query": "todo", "path": "."} if command == "find . -exec grep todo {} +" else None,
            "normalize_grep_shell_inspection": normalize_grep,
            "normalize_head_tail_shell_inspection": normalize_head_tail,
            "normalize_find_shell_inspection": normalize_find,
        }
        self.assertEqual(
            normalize_shell_inspection_call("run_shell", {"command": "find . -exec grep todo {} +"}, **base_kwargs),
            ("search", {"query": "todo", "path": "."}, "Normalized find-plus-grep inspection to search for cacheable structured context."),
        )
        self.assertEqual(
            normalize_shell_inspection_call("run_shell", {"command": "grep needle README.md"}, **base_kwargs),
            ("search", {"query": "needle", "path": ".", "limit": 20}, "Normalized shell text search to search for cacheable structured context."),
        )
        self.assertEqual(
            normalize_shell_inspection_call("run_shell", {"command": "head README.md"}, **base_kwargs),
            ("read_file", {"path": "README.md", "start": 1, "end": 10}, "Normalized shell file preview to read_file for bounded structured context."),
        )
        self.assertEqual(
            normalize_shell_inspection_call("run_shell", {"command": "find . -name '*.py'"}, **base_kwargs),
            ("file_search", {"query": ".py", "path": ".", "limit": 100}, "Normalized simple shell discovery to structured search for cacheable context."),
        )

    def test_head_tail_shell_inspection_normalizes_bounded_previews(self) -> None:
        line_counts = {"README.md": 50, "short.txt": 4}
        line_count_for_file = lambda path: line_counts.get(path)

        self.assertEqual(
            normalize_head_tail_shell_inspection(["head", "README.md"], line_count_for_file=line_count_for_file),
            {"path": "README.md", "start": 1, "end": 10},
        )
        self.assertEqual(
            normalize_head_tail_shell_inspection(["head", "-n", "5", "README.md"], line_count_for_file=line_count_for_file),
            {"path": "README.md", "start": 1, "end": 5},
        )
        self.assertEqual(
            normalize_head_tail_shell_inspection(["head", "-999", "README.md"], line_count_for_file=line_count_for_file),
            {"path": "README.md", "start": 1, "end": 200},
        )
        self.assertEqual(
            normalize_head_tail_shell_inspection(["tail", "-5", "README.md"], line_count_for_file=line_count_for_file),
            {"path": "README.md", "start": 46, "end": 50},
        )
        self.assertEqual(
            normalize_head_tail_shell_inspection(["tail", "-10", "short.txt"], line_count_for_file=line_count_for_file),
            {"path": "short.txt", "start": 1, "end": 4},
        )

    def test_head_tail_shell_inspection_rejects_invalid_or_unresolved_shapes(self) -> None:
        line_count_for_file = lambda path: None

        self.assertIsNone(normalize_head_tail_shell_inspection([], line_count_for_file=line_count_for_file))
        self.assertIsNone(normalize_head_tail_shell_inspection(["head", "-x", "README.md"], line_count_for_file=line_count_for_file))
        self.assertIsNone(normalize_head_tail_shell_inspection(["head", "-n", "bad", "README.md"], line_count_for_file=line_count_for_file))
        self.assertIsNone(normalize_head_tail_shell_inspection(["head", "README.md", "extra"], line_count_for_file=line_count_for_file))
        self.assertIsNone(normalize_head_tail_shell_inspection(["tail", "missing.txt"], line_count_for_file=line_count_for_file))

    def test_import_repair_bootstrap_routes_initial_listing_to_tests(self) -> None:
        self.assertEqual(
            normalize_import_repair_bootstrap_call(
                "list_files",
                {"path": "."},
                tool_calls_this_turn=[],
                request_looks_like_explicit_python_import_bug_fix=True,
                default_test_command="python -m unittest discover -s tests -v",
            ),
            (
                "run_test",
                {"command": "python -m unittest discover -s tests -v"},
                "Normalized initial list_files to run_test because the request already names a Python source path and needs concrete import/test failure evidence first.",
            ),
        )
        self.assertEqual(
            normalize_import_repair_bootstrap_call(
                "list_files",
                {"path": "."},
                tool_calls_this_turn=[{"name": "context_pack"}],
                request_looks_like_explicit_python_import_bug_fix=True,
                default_test_command=None,
            ),
            (
                "run_test",
                {},
                "Normalized initial list_files to run_test because the request already names a Python source path and needs concrete import/test failure evidence first.",
            ),
        )
        self.assertEqual(
            normalize_import_repair_bootstrap_call(
                "list_files",
                {"path": "."},
                tool_calls_this_turn=[{"name": "read_file"}],
                request_looks_like_explicit_python_import_bug_fix=True,
                default_test_command="python -m unittest",
            ),
            ("list_files", {"path": "."}, None),
        )

    def test_project_rename_bootstrap_routes_initial_tools_to_edit_intent(self) -> None:
        rename_ops = [
            (
                "edit_intent",
                {"path": ".", "intent": "rename", "target": "old_name", "replacement": "new_name", "scope": "project"},
            )
        ]
        self.assertEqual(
            normalize_project_rename_bootstrap_call(
                "list_files",
                {"path": "."},
                tool_calls_this_turn=[],
                requested_tool_names=set(),
                rename_operations=rename_ops,
            ),
            (
                "edit_intent",
                {"path": ".", "intent": "rename", "target": "old_name", "replacement": "new_name", "scope": "project"},
                "Normalized initial list_files to edit_intent because the request already specifies a grounded project rename operation.",
            ),
        )
        self.assertEqual(
            normalize_project_rename_bootstrap_call(
                "edit_intent",
                {"target": "old_name", "replacement": "new_name"},
                tool_calls_this_turn=[],
                requested_tool_names=set(),
                rename_operations=rename_ops,
            ),
            (
                "edit_intent",
                {"path": ".", "intent": "rename", "target": "old_name", "replacement": "new_name", "scope": "project"},
                "Normalized initial edit_intent to edit_intent because the request already specifies a grounded project rename operation.",
            ),
        )
        self.assertEqual(
            normalize_project_rename_bootstrap_call(
                "list_files",
                {"path": "."},
                tool_calls_this_turn=[],
                requested_tool_names={"list_files"},
                rename_operations=rename_ops,
            ),
            ("list_files", {"path": "."}, None),
        )
        self.assertEqual(
            normalize_project_rename_bootstrap_call(
                "edit_intent",
                {"target": "other", "replacement": "new_name"},
                tool_calls_this_turn=[],
                requested_tool_names=set(),
                rename_operations=rename_ops,
            ),
            ("edit_intent", {"target": "other", "replacement": "new_name"}, None),
        )

    def test_optional_parameter_bootstrap_routes_initial_search_to_edit_intent(self) -> None:
        operations = [
            (
                "edit_intent",
                {"path": "src/app.py", "intent": "change_signature", "target": "build", "replacement": "def build(flag: bool = False):"},
            )
        ]
        self.assertEqual(
            normalize_optional_parameter_bootstrap_call(
                "search_symbols",
                {"query": "build"},
                tool_calls_this_turn=[],
                requested_tool_names=set(),
                optional_parameter_operations=operations,
            ),
            (
                "edit_intent",
                {"path": "src/app.py", "intent": "change_signature", "target": "build", "replacement": "def build(flag: bool = False):"},
                "Normalized initial search_symbols to edit_intent because the request already specifies a grounded optional-parameter update.",
            ),
        )
        self.assertEqual(
            normalize_optional_parameter_bootstrap_call(
                "search_symbols",
                {"query": "build"},
                tool_calls_this_turn=[],
                requested_tool_names={"search_symbols"},
                optional_parameter_operations=operations,
            ),
            ("search_symbols", {"query": "build"}, None),
        )
        self.assertEqual(
            normalize_optional_parameter_bootstrap_call(
                "search_symbols",
                {"query": "build"},
                tool_calls_this_turn=[{"name": "read_file"}],
                requested_tool_names=set(),
                optional_parameter_operations=operations,
            ),
            ("search_symbols", {"query": "build"}, None),
        )

    def test_unittest_file_command_normalizes_test_file_paths_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_file = root / "tests" / "test_cli.py"
            source_file = root / "src" / "app.py"
            test_file.parent.mkdir()
            source_file.parent.mkdir()
            test_file.write_text("import unittest\n", encoding="utf-8")
            source_file.write_text("print('app')\n", encoding="utf-8")

            def resolve_path(raw_path: str) -> Path:
                candidate = root / raw_path.replace("/", "\\")
                if not candidate.exists():
                    raise FileNotFoundError(raw_path)
                return candidate

            def relative_label(path: Path) -> str:
                return path.relative_to(root).as_posix()

            self.assertEqual(
                normalize_unittest_file_command(
                    "python -m unittest tests/test_cli.py",
                    resolve_path=resolve_path,
                    relative_label=relative_label,
                ),
                "python -m unittest discover -s tests -p test_cli.py",
            )
            self.assertIsNone(
                normalize_unittest_file_command(
                    "python -m unittest src/app.py",
                    resolve_path=resolve_path,
                    relative_label=relative_label,
                )
            )

    def test_find_shell_inspection_normalizes_simple_discovery(self) -> None:
        self.assertEqual(
            normalize_find_shell_inspection(["find", ".", "-type", "f", "-name", "*.py"]),
            ("file_search", {"query": ".py", "path": ".", "limit": 100}),
        )
        self.assertEqual(
            normalize_find_shell_inspection(["find", "src", "-name", "*agent*", "-type", "d"]),
            ("directory_search", {"query": "*agent*", "path": "src", "limit": 100}),
        )
        self.assertIsNone(normalize_find_shell_inspection(["find", ".", "-maxdepth", "2", "-name", "*.py"]))


if __name__ == "__main__":
    unittest.main()
