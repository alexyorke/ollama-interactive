from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

from ollama_code.tool_dependencies import configured_docker_host, dependency_status, resolve_dependency
from ollama_code.tools import ToolExecutor


class ToolDependencyTests(unittest.TestCase):
    def test_registry_resolves_requested_integrations(self) -> None:
        for tool_id in (
            "py-tree-sitter",
            "ast-grep",
            "ripgrep",
            "fd",
            "jq",
            "yq",
            "uv",
            "docker",
            "typescript",
            "eslint",
            "prettier",
            "biome",
            "stylelint",
            "markdownlint-cli2",
            "taplo",
            "ruff",
            "basedpyright",
            "pytest",
            "pytest-json-report",
            "pytest-timeout",
            "pytest-xdist",
            "pytest-cov",
            "pre-commit",
            "tox",
            "nox",
            "mypy",
            "pyright",
            "deptry",
            "pip-audit",
            "pipdeptree",
            "vulture",
            "hypothesis",
            "py-spy",
            "scalene",
            "hyperfine",
            "bubblewrap",
            "scip-python",
            "scip-typescript",
            "difftastic",
            "coverage.py",
            "pytest-testmon",
            "opa",
            "inspect-ai",
            "sqlite-vec",
            "actionlint",
            "shellcheck",
            "hadolint",
            "shfmt",
            "yamllint",
            "check-jsonschema",
            "codespell",
            "sqlfluff",
            "osv-scanner",
            "gitleaks",
            "trivy",
            "grype",
            "semgrep",
            "opengrep",
            "comby",
            "phoenix",
            "universal-ctags",
            "mergiraf",
            "gh",
            "golangci-lint",
            "cargo-nextest",
        ):
            self.assertIsNotNone(resolve_dependency(tool_id), tool_id)

    def test_python_module_dependency_requires_all_modules(self) -> None:
        dependency = resolve_dependency("py-tree-sitter")
        self.assertIsNotNone(dependency)

        def fake_find_spec(name: str) -> object | None:
            return object() if name == "tree_sitter" else None

        with patch("ollama_code.tool_dependencies.shutil.which", return_value=None):
            with patch("ollama_code.tool_dependencies.importlib.util.find_spec", side_effect=fake_find_spec):
                status = dependency_status(dependency)  # type: ignore[arg-type]

        self.assertFalse(status["installed"])
        self.assertEqual(status["found_modules"], ["tree_sitter"])
        self.assertIn("tree_sitter_language_pack", status["missing_modules"])

    def test_missing_dependency_result_includes_install_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ops.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            with patch("ollama_code.tools.shutil.which", return_value=None):
                result = tools.ast_search("def $F($$$A): $$$B", lang="python")

        self.assertFalse(result["ok"])
        self.assertEqual(result["missing_dependency"], "ast-grep")
        self.assertEqual(result["tool_id"], "ast-grep")
        self.assertTrue(result["install_hints"])
        self.assertIn("dependency_purpose", result)

    def test_tool_install_is_plan_only_without_confirm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            with patch("ollama_code.tool_dependencies.shutil.which", return_value=None):
                with patch("ollama_code.tool_dependencies.importlib.util.find_spec", return_value=None):
                    with patch("ollama_code.tool_dependencies._resolve_executable", return_value=None):
                        result = tools.tool_install("ruff")

        self.assertTrue(result["ok"])
        self.assertTrue(result["planned"])
        self.assertIn("ruff can be installed", result["output"])
        self.assertIn("mode=isolated-venv", result["output"])

    def test_conflict_prone_python_tools_prefer_isolated_install_modes(self) -> None:
        semgrep = resolve_dependency("semgrep")
        inspect_ai = resolve_dependency("inspect-ai")
        self.assertIsNotNone(semgrep)
        self.assertIsNotNone(inspect_ai)

        semgrep_hints = dependency_status(semgrep)["install_hints"]  # type: ignore[arg-type]
        inspect_hints = dependency_status(inspect_ai)["install_hints"]  # type: ignore[arg-type]

        self.assertEqual(semgrep_hints[0]["mode"], "isolated-venv")
        self.assertEqual(semgrep_hints[1]["mode"], "docker")
        self.assertEqual(inspect_hints[0]["mode"], "isolated-venv")

    def test_docker_host_prefers_docker_install_hint_when_available(self) -> None:
        semgrep = resolve_dependency("semgrep")
        self.assertIsNotNone(semgrep)

        with patch.dict(os.environ, {"OLLAMA_CODE_DOCKER_HOST": "ssh://car-detection-server"}):
            semgrep_hints = dependency_status(semgrep)["install_hints"]  # type: ignore[arg-type]

        self.assertEqual(semgrep_hints[0]["mode"], "docker")
        self.assertIn("docker pull", semgrep_hints[0]["command"])

    def test_configured_docker_host_ignores_disabled_sentinel_values(self) -> None:
        with patch.dict(os.environ, {"OLLAMA_CODE_DOCKER_HOST": " off "}, clear=False):
            self.assertIsNone(configured_docker_host())

    def test_dependency_status_detects_isolated_venv_executable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / ".ollama-code" / "tool-envs" / "semgrep" / ("Scripts" if os.name == "nt" else "bin")
            scripts.mkdir(parents=True)
            executable = scripts / ("semgrep.exe" if os.name == "nt" else "semgrep")
            executable.write_text("", encoding="utf-8")
            dependency = resolve_dependency("semgrep")

            with patch("ollama_code.tool_dependencies.shutil.which", return_value=None):
                with patch("ollama_code.tool_dependencies.importlib.util.find_spec", return_value=None):
                    status = dependency_status(dependency, workspace_root=root)  # type: ignore[arg-type]

        self.assertTrue(status["installed"])
        self.assertEqual(status["found_isolated_executables"], ["semgrep"])

    def test_tool_install_denies_read_only_even_when_confirmed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="read-only", input_func=lambda _prompt: "y")
            with patch("ollama_code.tool_dependencies.shutil.which", return_value=None):
                result = tools.tool_install("ruff", confirm=True)

        self.assertFalse(result["ok"])
        self.assertIn("read-only", result["summary"])

    def test_tree_sitter_syntax_reports_missing_python_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
            tools = ToolExecutor(root, approval_mode="auto")
            with patch("ollama_code.tool_dependencies.importlib.util.find_spec", return_value=None):
                result = tools.tree_sitter_syntax("app.py")

        self.assertFalse(result["ok"])
        self.assertEqual(result["missing_dependency"], "py-tree-sitter")
        self.assertEqual(result["tool_id"], "py-tree-sitter")

    def test_dependency_diagnosis_adds_install_hints_for_known_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            with patch("ollama_code.tool_dependencies.shutil.which", return_value=None):
                result = tools.diagnose_dependency_error("executable not found: pytest")

        self.assertTrue(result["ok"])
        self.assertEqual(result["missing_command"], "pytest")
        self.assertEqual(result["tool_id"], "pytest")
        self.assertTrue(result["install_hints"])


if __name__ == "__main__":
    unittest.main()
