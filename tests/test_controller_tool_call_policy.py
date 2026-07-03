import tempfile
import unittest
from pathlib import Path

from ollama_code.agent_protocol import TargetLineReadSpec
from ollama_code.controller.tool_call_policy import (
    normalize_find_shell_inspection,
    normalize_run_test_call,
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
