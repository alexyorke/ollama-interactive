from __future__ import annotations

import importlib.util
import os
import platform
import shlex
import shutil
import site
import subprocess
import sys
import sysconfig
from urllib.parse import urlparse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InstallHint:
    manager: str
    command: tuple[str, ...]
    platforms: tuple[str, ...] = ("any",)
    note: str = ""
    mode: str = "host"


@dataclass(frozen=True)
class ToolDependency:
    id: str
    display_name: str
    category: str
    purpose: str
    executables: tuple[str, ...] = ()
    python_modules: tuple[str, ...] = ()
    recommended: bool = False
    optional: bool = True
    platforms: tuple[str, ...] = ("any",)
    install_hints: tuple[InstallHint, ...] = ()
    verify_command: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class DockerHostSetting:
    host: str | None
    source: str | None = None
    raw: str | None = None
    status: str = "absent"


_DOCKER_HOST_ENV_VARS = (
    "OLLAMA_CODE_DOCKER_HOST",
    "OLLAMA_CODE_REMOTE_DOCKER_HOST",
    "DOCKER_HOST",
)
_DOCKER_HOST_DISABLED_VALUES = frozenset({"0", "false", "none", "off"})


def _python_pip_command(*packages: str) -> tuple[str, ...]:
    return (sys.executable, "-m", "pip", "install", "--user", *packages)


def _isolated_pip_command(tool_id: str, *packages: str) -> tuple[str, ...]:
    return (sys.executable, "-m", "ollama_code.tool_dependencies", "install-venv", tool_id, *packages)


def _hint(manager: str, command: Iterable[str], *platforms: str, note: str = "", mode: str = "host") -> InstallHint:
    if mode == "host":
        mode = {
            "pip": "shared-python",
            "npm": "npm-global",
            "go": "go-bin",
            "cargo": "cargo-bin",
            "docker": "docker",
        }.get(manager, "host")
    return InstallHint(manager=manager, command=tuple(command), platforms=tuple(platforms or ("any",)), note=note, mode=mode)


TOOL_DEPENDENCIES: tuple[ToolDependency, ...] = (
    ToolDependency(
        id="py-tree-sitter",
        display_name="py-tree-sitter",
        category="syntax",
        purpose="Tree-sitter parser substrate for syntax-aware outlines, scopes, and patch validation.",
        python_modules=("tree_sitter", "tree_sitter_language_pack"),
        recommended=True,
        install_hints=(
            _hint("pip", _python_pip_command("tree-sitter", "tree-sitter-language-pack")),
        ),
        verify_command=(sys.executable, "-c", "import tree_sitter, tree_sitter_language_pack"),
    ),
    ToolDependency(
        id="ast-grep",
        display_name="ast-grep",
        category="syntax",
        purpose="Structural search and rewrite for AST-shaped code changes.",
        executables=("ast-grep", "sg"),
        recommended=True,
        install_hints=(
            _hint("npm", ("npm", "install", "-g", "@ast-grep/cli")),
            _hint("cargo", ("cargo", "install", "ast-grep")),
        ),
        verify_command=("ast-grep", "--version"),
    ),
    ToolDependency(
        id="ripgrep",
        display_name="ripgrep",
        category="search",
        purpose="Fast repo-local text search; reduces repeated slow shell grep/list loops.",
        executables=("rg", "ripgrep"),
        recommended=True,
        install_hints=(
            _hint("winget", ("winget", "install", "BurntSushi.ripgrep.MSVC"), "windows"),
            _hint("apt", ("sudo", "apt-get", "install", "-y", "ripgrep"), "linux", "wsl"),
            _hint("brew", ("brew", "install", "ripgrep"), "macos", "linux"),
            _hint("cargo", ("cargo", "install", "ripgrep")),
        ),
        verify_command=("rg", "--version"),
    ),
    ToolDependency(
        id="fd",
        display_name="fd",
        category="search",
        purpose="Fast file discovery; supports efficient path repair and avoids broad directory scans.",
        executables=("fd", "fdfind"),
        recommended=True,
        install_hints=(
            _hint("winget", ("winget", "install", "sharkdp.fd"), "windows"),
            _hint("apt", ("sudo", "apt-get", "install", "-y", "fd-find"), "linux", "wsl"),
            _hint("brew", ("brew", "install", "fd"), "macos", "linux"),
            _hint("cargo", ("cargo", "install", "fd-find")),
        ),
        verify_command=("fd", "--version"),
    ),
    ToolDependency(
        id="jq",
        display_name="jq",
        category="data",
        purpose="Deterministic JSON inspection for package manifests, lockfiles, traces, and benchmark outputs.",
        executables=("jq",),
        recommended=True,
        install_hints=(
            _hint("winget", ("winget", "install", "jqlang.jq"), "windows"),
            _hint("apt", ("sudo", "apt-get", "install", "-y", "jq"), "linux", "wsl"),
            _hint("brew", ("brew", "install", "jq"), "macos", "linux"),
        ),
        verify_command=("jq", "--version"),
    ),
    ToolDependency(
        id="yq",
        display_name="yq",
        category="data",
        purpose="Deterministic YAML/TOML/XML inspection for CI, Docker, Kubernetes, and config-heavy repos.",
        executables=("yq",),
        recommended=True,
        install_hints=(
            _hint("winget", ("winget", "install", "MikeFarah.yq"), "windows"),
            _hint("brew", ("brew", "install", "yq"), "macos", "linux"),
            _hint("go", ("go", "install", "github.com/mikefarah/yq/v4@latest")),
        ),
        verify_command=("yq", "--version"),
    ),
    ToolDependency(
        id="uv",
        display_name="uv",
        category="environment",
        purpose="Fast Python environment and dependency setup; useful for missing dependency and command-not-found failures.",
        executables=("uv",),
        recommended=True,
        install_hints=(
            _hint("pip", _python_pip_command("uv")),
            _hint("winget", ("winget", "install", "astral-sh.uv"), "windows"),
            _hint("brew", ("brew", "install", "uv"), "macos", "linux"),
        ),
        verify_command=("uv", "--version"),
    ),
    ToolDependency(
        id="docker",
        display_name="Docker CLI",
        category="container",
        purpose="Run heavy optional tooling in local or SSH-backed Docker containers without mutating the Python environment.",
        executables=("docker",),
        install_hints=(
            _hint("winget", ("winget", "install", "Docker.DockerDesktop"), "windows"),
            _hint("apt", ("sudo", "apt-get", "install", "-y", "docker.io"), "linux", "wsl"),
            _hint("brew", ("brew", "install", "--cask", "docker"), "macos"),
        ),
        verify_command=("docker", "version"),
        notes="Set OLLAMA_CODE_DOCKER_HOST=ssh://host to use a remote Docker daemon over SSH.",
    ),
    ToolDependency(
        id="typescript",
        display_name="TypeScript",
        category="javascript-validation",
        purpose="TypeScript compiler diagnostics for TS/JS repositories.",
        executables=("tsc",),
        install_hints=(
            _hint("npm", ("npm", "install", "-g", "typescript")),
        ),
        verify_command=("tsc", "--version"),
    ),
    ToolDependency(
        id="eslint",
        display_name="ESLint",
        category="javascript-validation",
        purpose="JavaScript/TypeScript lint diagnostics when target repos already use ESLint.",
        executables=("eslint",),
        install_hints=(
            _hint("npm", ("npm", "install", "-g", "eslint")),
        ),
        verify_command=("eslint", "--version"),
    ),
    ToolDependency(
        id="prettier",
        display_name="Prettier",
        category="formatter",
        purpose="Deterministic JS/TS/CSS/Markdown formatting after edits.",
        executables=("prettier",),
        install_hints=(
            _hint("npm", ("npm", "install", "-g", "prettier")),
        ),
        verify_command=("prettier", "--version"),
    ),
    ToolDependency(
        id="biome",
        display_name="Biome",
        category="javascript-validation",
        purpose="Fast JS/TS linting and formatting for repos that use Biome instead of ESLint/Prettier.",
        executables=("biome",),
        install_hints=(
            _hint("npm", ("npm", "install", "-g", "@biomejs/biome")),
        ),
        verify_command=("biome", "--version"),
    ),
    ToolDependency(
        id="stylelint",
        display_name="Stylelint",
        category="web-validation",
        purpose="CSS/SCSS lint diagnostics for frontend tasks.",
        executables=("stylelint",),
        install_hints=(
            _hint("npm", ("npm", "install", "-g", "stylelint")),
        ),
        verify_command=("stylelint", "--version"),
    ),
    ToolDependency(
        id="markdownlint-cli2",
        display_name="markdownlint-cli2",
        category="docs-validation",
        purpose="Markdown linting for README/docs edits that otherwise fail CI late.",
        executables=("markdownlint-cli2",),
        install_hints=(
            _hint("npm", ("npm", "install", "-g", "markdownlint-cli2")),
        ),
        verify_command=("markdownlint-cli2", "--version"),
    ),
    ToolDependency(
        id="taplo",
        display_name="Taplo",
        category="config-validation",
        purpose="TOML formatting and validation for pyproject/Cargo/config edits.",
        executables=("taplo",),
        install_hints=(
            _hint("cargo", ("cargo", "install", "taplo-cli", "--locked")),
            _hint("brew", ("brew", "install", "taplo"), "macos", "linux"),
        ),
        verify_command=("taplo", "--version"),
    ),
    ToolDependency(
        id="ruff",
        display_name="Ruff",
        category="python-validation",
        purpose="Fast Python linting and formatting diagnostics.",
        executables=("ruff",),
        python_modules=("ruff",),
        recommended=True,
        install_hints=(
            _hint("pip", _isolated_pip_command("ruff", "ruff"), mode="isolated-venv"),
            _hint("pip", _python_pip_command("ruff"), note="Fallback only; prefer isolated-venv to avoid shared Python changes."),
        ),
        verify_command=("ruff", "--version"),
    ),
    ToolDependency(
        id="basedpyright",
        display_name="basedpyright",
        category="python-validation",
        purpose="Python type checking and language intelligence.",
        executables=("basedpyright",),
        python_modules=("basedpyright",),
        recommended=True,
        install_hints=(
            _hint("pip", _isolated_pip_command("basedpyright", "basedpyright"), mode="isolated-venv"),
            _hint("pip", _python_pip_command("basedpyright"), note="Fallback only; prefer isolated-venv to avoid shared Python changes."),
        ),
        verify_command=("basedpyright", "--version"),
    ),
    ToolDependency(
        id="pytest",
        display_name="pytest",
        category="python-test",
        purpose="Standard Python test runner used by most real SWE benchmark repositories.",
        executables=("pytest",),
        python_modules=("pytest",),
        recommended=True,
        install_hints=(
            _hint("pip", _python_pip_command("pytest")),
        ),
        verify_command=(sys.executable, "-m", "pytest", "--version"),
    ),
    ToolDependency(
        id="pytest-json-report",
        display_name="pytest-json-report",
        category="python-test",
        purpose="Structured pytest failure summaries for smaller, more reliable repair prompts.",
        python_modules=("pytest_jsonreport",),
        recommended=True,
        install_hints=(
            _hint("pip", _python_pip_command("pytest-json-report")),
        ),
        verify_command=(sys.executable, "-c", "import pytest_jsonreport"),
    ),
    ToolDependency(
        id="pytest-timeout",
        display_name="pytest-timeout",
        category="python-test",
        purpose="Bound hanging tests so timeout failures fail fast with usable diagnostics.",
        python_modules=("pytest_timeout",),
        recommended=True,
        install_hints=(
            _hint("pip", _python_pip_command("pytest-timeout")),
        ),
        verify_command=(sys.executable, "-c", "import pytest_timeout"),
    ),
    ToolDependency(
        id="pytest-xdist",
        display_name="pytest-xdist",
        category="python-test",
        purpose="Parallel pytest execution for large suites once a repo is stable.",
        python_modules=("xdist",),
        install_hints=(
            _hint("pip", _python_pip_command("pytest-xdist")),
        ),
        verify_command=(sys.executable, "-c", "import xdist"),
    ),
    ToolDependency(
        id="pytest-cov",
        display_name="pytest-cov",
        category="test-selection",
        purpose="Coverage collection through pytest for changed-code impact analysis.",
        python_modules=("pytest_cov",),
        install_hints=(
            _hint("pip", _python_pip_command("pytest-cov")),
        ),
        verify_command=(sys.executable, "-c", "import pytest_cov"),
    ),
    ToolDependency(
        id="pre-commit",
        display_name="pre-commit",
        category="validation",
        purpose="Repo-native hook runner that executes the validators maintainers already chose.",
        executables=("pre-commit",),
        python_modules=("pre_commit",),
        install_hints=(
            _hint("pip", _python_pip_command("pre-commit")),
        ),
        verify_command=("pre-commit", "--version"),
    ),
    ToolDependency(
        id="tox",
        display_name="tox",
        category="python-test",
        purpose="Python test matrix runner for repos that encode validation in tox.ini or pyproject.",
        executables=("tox",),
        python_modules=("tox",),
        install_hints=(
            _hint("pip", _python_pip_command("tox")),
        ),
        verify_command=("tox", "--version"),
    ),
    ToolDependency(
        id="nox",
        display_name="nox",
        category="python-test",
        purpose="Python session runner for repos that encode validation in noxfile.py.",
        executables=("nox",),
        python_modules=("nox",),
        install_hints=(
            _hint("pip", _python_pip_command("nox")),
        ),
        verify_command=("nox", "--version"),
    ),
    ToolDependency(
        id="mypy",
        display_name="mypy",
        category="python-validation",
        purpose="Python static type checking when target repos already use mypy configs.",
        executables=("mypy",),
        python_modules=("mypy",),
        install_hints=(
            _hint("pip", _isolated_pip_command("mypy", "mypy"), mode="isolated-venv"),
            _hint("pip", _python_pip_command("mypy"), note="Fallback only; prefer isolated-venv to avoid shared Python changes."),
        ),
        verify_command=("mypy", "--version"),
    ),
    ToolDependency(
        id="pyright",
        display_name="Pyright",
        category="python-validation",
        purpose="Python type checking and language-server-compatible diagnostics.",
        executables=("pyright",),
        install_hints=(
            _hint("npm", ("npm", "install", "-g", "pyright")),
            _hint("pip", _python_pip_command("pyright")),
        ),
        verify_command=("pyright", "--version"),
    ),
    ToolDependency(
        id="deptry",
        display_name="deptry",
        category="python-validation",
        purpose="Detect missing, unused, and misplaced Python dependencies before retrying broken imports.",
        executables=("deptry",),
        python_modules=("deptry",),
        install_hints=(
            _hint("pip", _isolated_pip_command("deptry", "deptry"), mode="isolated-venv"),
            _hint("pip", _python_pip_command("deptry"), note="Fallback only; prefer isolated-venv to avoid shared Python changes."),
        ),
        verify_command=("deptry", "--version"),
    ),
    ToolDependency(
        id="pip-audit",
        display_name="pip-audit",
        category="security",
        purpose="Python dependency vulnerability scanning after requirements or pyproject changes.",
        executables=("pip-audit",),
        python_modules=("pip_audit",),
        install_hints=(
            _hint("pip", _isolated_pip_command("pip-audit", "pip-audit"), mode="isolated-venv"),
            _hint("pip", _python_pip_command("pip-audit"), note="Fallback only; prefer isolated-venv to avoid shared Python changes."),
        ),
        verify_command=("pip-audit", "--version"),
    ),
    ToolDependency(
        id="pipdeptree",
        display_name="pipdeptree",
        category="environment",
        purpose="Dependency graph inspection for import/version conflict diagnosis.",
        executables=("pipdeptree",),
        python_modules=("pipdeptree",),
        install_hints=(
            _hint("pip", _isolated_pip_command("pipdeptree", "pipdeptree"), mode="isolated-venv"),
            _hint("pip", _python_pip_command("pipdeptree"), note="Fallback only; prefer isolated-venv to avoid shared Python changes."),
        ),
        verify_command=("pipdeptree", "--version"),
    ),
    ToolDependency(
        id="vulture",
        display_name="Vulture",
        category="python-validation",
        purpose="Find dead Python code during cleanup/refactor tasks without model-heavy inspection.",
        executables=("vulture",),
        python_modules=("vulture",),
        install_hints=(
            _hint("pip", _isolated_pip_command("vulture", "vulture"), mode="isolated-venv"),
            _hint("pip", _python_pip_command("vulture"), note="Fallback only; prefer isolated-venv to avoid shared Python changes."),
        ),
        verify_command=("vulture", "--version"),
    ),
    ToolDependency(
        id="hypothesis",
        display_name="Hypothesis",
        category="python-test",
        purpose="Property-based test generation for edge cases after small deterministic fixes.",
        python_modules=("hypothesis",),
        install_hints=(
            _hint("pip", _python_pip_command("hypothesis")),
        ),
        verify_command=(sys.executable, "-c", "import hypothesis"),
    ),
    ToolDependency(
        id="py-spy",
        display_name="py-spy",
        category="profiling",
        purpose="Low-overhead Python profiler for timeout and performance failures.",
        executables=("py-spy",),
        install_hints=(
            _hint("pip", _isolated_pip_command("py-spy", "py-spy"), mode="isolated-venv"),
            _hint("pip", _python_pip_command("py-spy"), note="Fallback only; prefer isolated-venv to avoid shared Python changes."),
            _hint("cargo", ("cargo", "install", "py-spy")),
        ),
        verify_command=("py-spy", "--version"),
    ),
    ToolDependency(
        id="scalene",
        display_name="Scalene",
        category="profiling",
        purpose="Python CPU/GPU/memory profiler for slow benchmark and timeout investigations.",
        executables=("scalene",),
        python_modules=("scalene",),
        install_hints=(
            _hint("pip", _isolated_pip_command("scalene", "scalene"), mode="isolated-venv"),
            _hint("pip", _python_pip_command("scalene"), note="Fallback only; prefer isolated-venv to avoid shared Python changes."),
        ),
        verify_command=("scalene", "--version"),
    ),
    ToolDependency(
        id="hyperfine",
        display_name="hyperfine",
        category="profiling",
        purpose="Repeatable command timing for before/after performance claims.",
        executables=("hyperfine",),
        install_hints=(
            _hint("winget", ("winget", "install", "sharkdp.hyperfine"), "windows"),
            _hint("apt", ("sudo", "apt-get", "install", "-y", "hyperfine"), "linux", "wsl"),
            _hint("brew", ("brew", "install", "hyperfine"), "macos", "linux"),
            _hint("cargo", ("cargo", "install", "hyperfine")),
        ),
        verify_command=("hyperfine", "--version"),
    ),
    ToolDependency(
        id="bubblewrap",
        display_name="bubblewrap",
        category="sandbox",
        purpose="Linux/WSL command sandboxing with namespace isolation.",
        executables=("bwrap", "bubblewrap"),
        recommended=True,
        platforms=("linux", "wsl"),
        install_hints=(
            _hint("apt", ("sudo", "apt-get", "install", "-y", "bubblewrap"), "linux", "wsl"),
            _hint("dnf", ("sudo", "dnf", "install", "-y", "bubblewrap"), "linux"),
            _hint("pacman", ("sudo", "pacman", "-S", "bubblewrap"), "linux"),
        ),
        verify_command=("bwrap", "--version"),
        notes="Unsupported on native Windows; use WSL/Linux for this integration.",
    ),
    ToolDependency(
        id="difftastic",
        display_name="difftastic",
        category="review",
        purpose="Syntax-aware diff summaries for self-review after edits.",
        executables=("difft",),
        recommended=True,
        install_hints=(
            _hint("cargo", ("cargo", "install", "difftastic")),
            _hint("winget", ("winget", "install", "Wilfred.difftastic"), "windows"),
            _hint("brew", ("brew", "install", "difftastic"), "macos", "linux"),
        ),
        verify_command=("difft", "--version"),
    ),
    ToolDependency(
        id="actionlint",
        display_name="actionlint",
        category="validator",
        purpose="GitHub Actions workflow linting.",
        executables=("actionlint",),
        recommended=True,
        install_hints=(
            _hint("go", ("go", "install", "github.com/rhysd/actionlint/cmd/actionlint@latest")),
            _hint("brew", ("brew", "install", "actionlint"), "macos", "linux"),
        ),
        verify_command=("actionlint", "-version"),
    ),
    ToolDependency(
        id="shellcheck",
        display_name="ShellCheck",
        category="validator",
        purpose="Shell script static analysis.",
        executables=("shellcheck",),
        recommended=True,
        install_hints=(
            _hint("winget", ("winget", "install", "koalaman.shellcheck"), "windows"),
            _hint("apt", ("sudo", "apt-get", "install", "-y", "shellcheck"), "linux", "wsl"),
            _hint("brew", ("brew", "install", "shellcheck"), "macos", "linux"),
        ),
        verify_command=("shellcheck", "--version"),
    ),
    ToolDependency(
        id="hadolint",
        display_name="Hadolint",
        category="validator",
        purpose="Dockerfile linting.",
        executables=("hadolint",),
        recommended=True,
        install_hints=(
            _hint("winget", ("winget", "install", "hadolint.hadolint"), "windows"),
            _hint("brew", ("brew", "install", "hadolint"), "macos", "linux"),
        ),
        verify_command=("hadolint", "--version"),
    ),
    ToolDependency(
        id="shfmt",
        display_name="shfmt",
        category="validator",
        purpose="Shell formatting and parse checking before or after ShellCheck.",
        executables=("shfmt",),
        install_hints=(
            _hint("winget", ("winget", "install", "mvdan.shfmt"), "windows"),
            _hint("apt", ("sudo", "apt-get", "install", "-y", "shfmt"), "linux", "wsl"),
            _hint("brew", ("brew", "install", "shfmt"), "macos", "linux"),
            _hint("go", ("go", "install", "mvdan.cc/sh/v3/cmd/shfmt@latest")),
        ),
        verify_command=("shfmt", "--version"),
    ),
    ToolDependency(
        id="yamllint",
        display_name="yamllint",
        category="validator",
        purpose="YAML validation for CI/config files where syntax errors are common.",
        executables=("yamllint",),
        python_modules=("yamllint",),
        install_hints=(
            _hint("pip", _isolated_pip_command("yamllint", "yamllint"), mode="isolated-venv"),
            _hint("pip", _python_pip_command("yamllint"), note="Fallback only; prefer isolated-venv to avoid shared Python changes."),
        ),
        verify_command=("yamllint", "--version"),
    ),
    ToolDependency(
        id="check-jsonschema",
        display_name="check-jsonschema",
        category="validator",
        purpose="JSON/YAML schema checks for CI, config, and manifest files.",
        executables=("check-jsonschema",),
        python_modules=("check_jsonschema",),
        install_hints=(
            _hint("pip", _isolated_pip_command("check-jsonschema", "check-jsonschema"), mode="isolated-venv"),
            _hint("pip", _python_pip_command("check-jsonschema"), note="Fallback only; prefer isolated-venv to avoid shared Python changes."),
        ),
        verify_command=("check-jsonschema", "--version"),
    ),
    ToolDependency(
        id="codespell",
        display_name="codespell",
        category="docs-validation",
        purpose="Cheap typo detection in docs, identifiers, and user-facing strings.",
        executables=("codespell",),
        python_modules=("codespell_lib",),
        install_hints=(
            _hint("pip", _isolated_pip_command("codespell", "codespell"), mode="isolated-venv"),
            _hint("pip", _python_pip_command("codespell"), note="Fallback only; prefer isolated-venv to avoid shared Python changes."),
        ),
        verify_command=("codespell", "--version"),
    ),
    ToolDependency(
        id="sqlfluff",
        display_name="SQLFluff",
        category="sql-validation",
        purpose="SQL linting for migrations and query-heavy repositories.",
        executables=("sqlfluff",),
        python_modules=("sqlfluff",),
        install_hints=(
            _hint("pip", _isolated_pip_command("sqlfluff", "sqlfluff"), mode="isolated-venv"),
            _hint("pip", _python_pip_command("sqlfluff"), note="Fallback only; prefer isolated-venv to avoid shared Python changes."),
        ),
        verify_command=("sqlfluff", "--version"),
    ),
    ToolDependency(
        id="osv-scanner",
        display_name="OSV-Scanner",
        category="security",
        purpose="Dependency vulnerability scanning after manifest or lockfile edits.",
        executables=("osv-scanner",),
        recommended=True,
        install_hints=(
            _hint("go", ("go", "install", "github.com/google/osv-scanner/cmd/osv-scanner@latest")),
            _hint("brew", ("brew", "install", "osv-scanner"), "macos", "linux"),
        ),
        verify_command=("osv-scanner", "--version"),
    ),
    ToolDependency(
        id="gitleaks",
        display_name="Gitleaks",
        category="security",
        purpose="Secret scanning before commits or dependency/config changes.",
        executables=("gitleaks",),
        install_hints=(
            _hint("winget", ("winget", "install", "Gitleaks.Gitleaks"), "windows"),
            _hint("brew", ("brew", "install", "gitleaks"), "macos", "linux"),
            _hint("go", ("go", "install", "github.com/zricethezav/gitleaks/v8@latest")),
        ),
        verify_command=("gitleaks", "version"),
    ),
    ToolDependency(
        id="trivy",
        display_name="Trivy",
        category="security",
        purpose="Filesystem, container, and dependency vulnerability scanning.",
        executables=("trivy",),
        install_hints=(
            _hint("winget", ("winget", "install", "AquaSecurity.Trivy"), "windows"),
            _hint("brew", ("brew", "install", "trivy"), "macos", "linux"),
        ),
        verify_command=("trivy", "--version"),
    ),
    ToolDependency(
        id="grype",
        display_name="Grype",
        category="security",
        purpose="SBOM/dependency vulnerability scanning as another local fallback.",
        executables=("grype",),
        install_hints=(
            _hint("winget", ("winget", "install", "Anchore.Grype"), "windows"),
            _hint("brew", ("brew", "install", "grype"), "macos", "linux"),
        ),
        verify_command=("grype", "version"),
    ),
    ToolDependency(
        id="coverage.py",
        display_name="coverage.py",
        category="test-selection",
        purpose="Python coverage data and dynamic contexts for impact-aware test selection.",
        executables=("coverage",),
        python_modules=("coverage",),
        install_hints=(
            _hint("pip", _isolated_pip_command("coverage.py", "coverage"), mode="isolated-venv"),
            _hint("pip", _python_pip_command("coverage"), note="Fallback only; prefer isolated-venv to avoid shared Python changes."),
        ),
        verify_command=(sys.executable, "-m", "coverage", "--version"),
    ),
    ToolDependency(
        id="pytest-testmon",
        display_name="pytest-testmon",
        category="test-selection",
        purpose="Coverage-informed pytest test selection.",
        python_modules=("testmon",),
        install_hints=(
            _hint("pip", _python_pip_command("pytest-testmon")),
        ),
        verify_command=(sys.executable, "-m", "pytest", "--help"),
    ),
    ToolDependency(
        id="opa",
        display_name="Open Policy Agent",
        category="policy",
        purpose="Declarative allow/deny policy evaluation for tools and shell execution.",
        executables=("opa",),
        install_hints=(
            _hint("winget", ("winget", "install", "open-policy-agent.opa"), "windows"),
            _hint("brew", ("brew", "install", "opa"), "macos", "linux"),
        ),
        verify_command=("opa", "version"),
    ),
    ToolDependency(
        id="inspect-ai",
        display_name="Inspect AI",
        category="eval",
        purpose="Agent benchmark and trace harness for reproducible tool-loop evaluation.",
        executables=("inspect",),
        python_modules=("inspect_ai",),
        install_hints=(
            _hint("pip", _isolated_pip_command("inspect-ai", "inspect-ai"), mode="isolated-venv"),
            _hint("pip", _python_pip_command("inspect-ai"), note="Fallback only; known to conflict with some LiteLLM pins."),
        ),
        verify_command=("inspect", "--version"),
    ),
    ToolDependency(
        id="sqlite-vec",
        display_name="sqlite-vec",
        category="retrieval",
        purpose="SQLite-local vector search for docs/comments/issue-text reranking.",
        python_modules=("sqlite_vec",),
        install_hints=(
            _hint("pip", _python_pip_command("sqlite-vec")),
        ),
        verify_command=(sys.executable, "-c", "import sqlite_vec"),
    ),
    ToolDependency(
        id="scip-python",
        display_name="SCIP Python",
        category="xref",
        purpose="Python SCIP symbol index generation for durable definition/reference lookup.",
        executables=("scip-python",),
        install_hints=(
            _hint("npm", ("npm", "install", "-g", "@sourcegraph/scip-python")),
        ),
        verify_command=("scip-python", "--version"),
    ),
    ToolDependency(
        id="scip-typescript",
        display_name="SCIP TypeScript",
        category="xref",
        purpose="TypeScript SCIP symbol index generation for durable definition/reference lookup.",
        executables=("scip-typescript",),
        install_hints=(
            _hint("npm", ("npm", "install", "-g", "@sourcegraph/scip-typescript")),
        ),
        verify_command=("scip-typescript", "--version"),
    ),
    ToolDependency(
        id="gh",
        display_name="GitHub CLI",
        category="repo-context",
        purpose="Issue, PR, and CI context retrieval when a task references GitHub state.",
        executables=("gh",),
        install_hints=(
            _hint("winget", ("winget", "install", "GitHub.cli"), "windows"),
            _hint("brew", ("brew", "install", "gh"), "macos", "linux"),
        ),
        verify_command=("gh", "--version"),
    ),
    ToolDependency(
        id="golangci-lint",
        display_name="golangci-lint",
        category="go-validation",
        purpose="Fast aggregated Go linting for Go repositories.",
        executables=("golangci-lint",),
        install_hints=(
            _hint("go", ("go", "install", "github.com/golangci/golangci-lint/cmd/golangci-lint@latest")),
            _hint("brew", ("brew", "install", "golangci-lint"), "macos", "linux"),
        ),
        verify_command=("golangci-lint", "--version"),
    ),
    ToolDependency(
        id="cargo-nextest",
        display_name="cargo-nextest",
        category="rust-test",
        purpose="Faster structured Rust test execution when repos support it.",
        executables=("cargo-nextest",),
        install_hints=(
            _hint("cargo", ("cargo", "install", "cargo-nextest", "--locked")),
            _hint("brew", ("brew", "install", "cargo-nextest"), "macos", "linux"),
        ),
        verify_command=("cargo", "nextest", "--version"),
    ),
    ToolDependency(
        id="semgrep",
        display_name="Semgrep",
        category="security",
        purpose="Structural/security scanning with Semgrep-compatible rules.",
        executables=("semgrep",),
        python_modules=("semgrep",),
        install_hints=(
            _hint("pip", _isolated_pip_command("semgrep", "semgrep"), mode="isolated-venv"),
            _hint("docker", ("docker", "pull", "semgrep/semgrep"), mode="docker", note="Container image avoids Python dependency conflicts."),
            _hint("pip", _python_pip_command("semgrep"), note="Fallback only; known to conflict with some LiteLLM jsonschema pins."),
        ),
        verify_command=("semgrep", "--version"),
    ),
    ToolDependency(
        id="opengrep",
        display_name="Opengrep",
        category="security",
        purpose="Open Semgrep-compatible structural/security scanning.",
        executables=("opengrep",),
        install_hints=(
            _hint("pip", _isolated_pip_command("opengrep", "opengrep"), mode="isolated-venv"),
            _hint("pip", _python_pip_command("opengrep"), note="Fallback only; prefer isolated-venv to avoid shared Python changes."),
        ),
        verify_command=("opengrep", "--version"),
    ),
    ToolDependency(
        id="comby",
        display_name="Comby",
        category="rewrite",
        purpose="Template-based structural matching for languages/configs where AST support is weak.",
        executables=("comby",),
        install_hints=(
            _hint("brew", ("brew", "install", "comby"), "macos", "linux"),
            _hint("opam", ("opam", "install", "comby")),
        ),
        verify_command=("comby", "-version"),
    ),
    ToolDependency(
        id="phoenix",
        display_name="Phoenix",
        category="observability",
        purpose="Optional local trace viewer for OpenInference/OpenTelemetry-style evaluation traces.",
        python_modules=("phoenix",),
        install_hints=(
            _hint("pip", _isolated_pip_command("phoenix", "arize-phoenix"), mode="isolated-venv"),
            _hint("pip", _python_pip_command("arize-phoenix"), note="Fallback only; prefer isolated-venv to avoid shared Python changes."),
        ),
        verify_command=(sys.executable, "-c", "import phoenix"),
    ),
    ToolDependency(
        id="universal-ctags",
        display_name="Universal Ctags",
        category="xref",
        purpose="Broad-language fallback symbol tags when richer parsers are unavailable.",
        executables=("ctags",),
        install_hints=(
            _hint("winget", ("winget", "install", "UniversalCtags.Ctags"), "windows"),
            _hint("apt", ("sudo", "apt-get", "install", "-y", "universal-ctags"), "linux", "wsl"),
            _hint("brew", ("brew", "install", "universal-ctags"), "macos", "linux"),
        ),
        verify_command=("ctags", "--version"),
    ),
    ToolDependency(
        id="mergiraf",
        display_name="Mergiraf",
        category="merge",
        purpose="Syntax-aware merge helper for automated patch reconciliation.",
        executables=("mergiraf",),
        install_hints=(
            _hint("cargo", ("cargo", "install", "mergiraf")),
            _hint("brew", ("brew", "install", "mergiraf"), "macos", "linux"),
        ),
        verify_command=("mergiraf", "--version"),
    ),
)

TOOL_DEPENDENCY_BY_ID = {dependency.id: dependency for dependency in TOOL_DEPENDENCIES}
TOOL_DEPENDENCY_BY_NAME = {
    name.lower(): dependency
    for dependency in TOOL_DEPENDENCIES
    for name in (dependency.id, dependency.display_name, *dependency.executables, *dependency.python_modules)
}


def current_platform() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        release = platform.release().lower()
        if "microsoft" in release or "wsl" in release or os.environ.get("WSL_DISTRO_NAME"):
            return "wsl"
        return "linux"
    return sys.platform


def platform_supported(dependency: ToolDependency, platform_name: str | None = None) -> bool:
    platform_name = platform_name or current_platform()
    return "any" in dependency.platforms or platform_name in dependency.platforms


def command_to_text(command: tuple[str, ...]) -> str:
    if hasattr(shlex, "join"):
        return shlex.join(command)
    return " ".join(shlex.quote(part) for part in command)


def _executable_names(name: str) -> tuple[str, ...]:
    if os.name == "nt" and Path(name).suffix == "":
        return (name, f"{name}.exe", f"{name}.cmd", f"{name}.bat")
    return (name,)


def _tool_env_dir(workspace_root: str | Path | None, tool_id: str) -> Path | None:
    if workspace_root is None:
        return None
    return Path(workspace_root) / ".ollama-code" / "tool-envs" / tool_id


def _venv_bin_dir(env_dir: Path) -> Path:
    return env_dir / ("Scripts" if os.name == "nt" else "bin")


def _resolve_isolated_executable(workspace_root: str | Path | None, tool_id: str, name: str) -> str | None:
    env_dir = _tool_env_dir(workspace_root, tool_id)
    if env_dir is None:
        return None
    bin_dir = _venv_bin_dir(env_dir)
    for executable_name in _executable_names(name):
        candidate = bin_dir / executable_name
        if candidate.exists():
            return str(candidate)
    return None


def _resolve_executable(name: str, *, workspace_root: str | Path | None = None, tool_id: str | None = None) -> str | None:
    if tool_id:
        isolated = _resolve_isolated_executable(workspace_root, tool_id, name)
        if isolated:
            return isolated
    resolved = shutil.which(name)
    if resolved:
        return resolved
    candidate = Path(name)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)
    script_roots: set[Path] = set()
    for value in (sysconfig.get_path("scripts"), site.getuserbase()):
        if value:
            root = Path(value)
            script_roots.add(root)
            script_roots.add(root / "Scripts")
            script_roots.add(root / "bin")
            script_roots.add(root / f"Python{sys.version_info.major}{sys.version_info.minor}" / "Scripts")
            script_roots.add(root / f"Python{sys.version_info.major}{sys.version_info.minor}" / "bin")
    for root in script_roots:
        for executable_name in _executable_names(name):
            candidate = root / executable_name
            if candidate.exists():
                return str(candidate)
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            winget_packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
            if winget_packages.exists():
                for executable_name in _executable_names(name):
                    direct_matches = list(winget_packages.glob(f"*/*{executable_name}"))
                    recursive_matches = direct_matches or list(winget_packages.glob(f"*/**/{executable_name}"))
                    for candidate in recursive_matches:
                        if candidate.is_file():
                            return str(candidate)
    return None


def resolve_tool_executable(tool_id: str, executable: str, *, workspace_root: str | Path | None = None) -> str | None:
    return _resolve_executable(executable, workspace_root=workspace_root, tool_id=tool_id)


def install_hint_supported(hint: InstallHint, platform_name: str | None = None) -> bool:
    platform_name = platform_name or current_platform()
    return "any" in hint.platforms or platform_name in hint.platforms


def normalize_docker_host(value: str | None) -> str | None:
    if not value:
        return None
    clean = value.strip()
    if not clean:
        return None
    if clean.lower() in _DOCKER_HOST_DISABLED_VALUES:
        return None
    if "://" not in clean:
        clean = f"ssh://{clean}"
    parsed = urlparse(clean)
    if not parsed.scheme:
        return None
    if parsed.scheme.lower() in {"unix", "npipe"}:
        if not parsed.path:
            return None
    elif not parsed.netloc:
        return None
    return clean


def docker_host_kind(value: str | None) -> str | None:
    host = normalize_docker_host(value)
    if not host:
        return None
    scheme = urlparse(host).scheme.lower()
    if scheme in {"unix", "npipe"}:
        return "local"
    if scheme == "ssh":
        return "remote"
    return "custom"


def configured_docker_host_setting() -> DockerHostSetting:
    for name in _DOCKER_HOST_ENV_VARS:
        raw = os.environ.get(name)
        if raw is None:
            continue
        clean = raw.strip()
        if not clean:
            continue
        if clean.lower() in _DOCKER_HOST_DISABLED_VALUES:
            return DockerHostSetting(host=None, source=name, raw=raw, status="disabled")
        normalized = normalize_docker_host(raw)
        if normalized is not None:
            return DockerHostSetting(host=normalized, source=name, raw=raw, status="configured")
        return DockerHostSetting(host=None, source=name, raw=raw, status="invalid")
    return DockerHostSetting(host=None)


def configured_docker_host() -> str | None:
    return configured_docker_host_setting().host


def prefer_docker_tools() -> bool:
    docker_setting = configured_docker_host_setting()
    if docker_setting.host and docker_setting.source in {
        "OLLAMA_CODE_DOCKER_HOST",
        "OLLAMA_CODE_REMOTE_DOCKER_HOST",
    }:
        return True
    value = os.environ.get("OLLAMA_CODE_PREFER_DOCKER_TOOLS", "").strip().lower()
    return bool(value) and value not in _DOCKER_HOST_DISABLED_VALUES


def _ordered_install_hints(hints: Iterable[InstallHint], platform_name: str | None = None) -> list[InstallHint]:
    supported_hints = [hint for hint in hints if install_hint_supported(hint, platform_name)]
    if prefer_docker_tools():
        supported_hints.sort(key=lambda hint: 0 if hint.mode == "docker" else 1)
    return supported_hints


def install_hint_payload(hint: InstallHint, platform_name: str | None = None) -> dict[str, object]:
    manager_executable = hint.command[0] if hint.command else hint.manager
    return {
        "manager": hint.manager,
        "mode": hint.mode,
        "command": command_to_text(hint.command),
        "argv": list(hint.command),
        "platforms": list(hint.platforms),
        "manager_available": _resolve_executable(manager_executable) is not None if manager_executable else False,
        "supported": install_hint_supported(hint, platform_name),
        "note": hint.note,
    }


def resolve_dependency(name: str) -> ToolDependency | None:
    return TOOL_DEPENDENCY_BY_NAME.get(str(name or "").strip().lower())


def dependency_status(
    dependency: ToolDependency,
    platform_name: str | None = None,
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, object]:
    platform_name = platform_name or current_platform()
    supported = platform_supported(dependency, platform_name)
    found_executables = [
        name
        for name in dependency.executables
        if _resolve_executable(name, workspace_root=workspace_root, tool_id=dependency.id)
    ]
    found_isolated_executables = [
        name
        for name in dependency.executables
        if _resolve_isolated_executable(workspace_root, dependency.id, name)
    ]
    found_modules = [name for name in dependency.python_modules if importlib.util.find_spec(name) is not None]
    modules_installed = bool(dependency.python_modules) and len(found_modules) == len(dependency.python_modules)
    installed = bool(found_executables) or modules_installed
    missing_modules = [name for name in dependency.python_modules if name not in found_modules]
    ordered_hints = _ordered_install_hints(dependency.install_hints, platform_name)
    hints = [install_hint_payload(hint, platform_name) for hint in ordered_hints]
    return {
        "id": dependency.id,
        "display_name": dependency.display_name,
        "category": dependency.category,
        "purpose": dependency.purpose,
        "recommended": dependency.recommended,
        "optional": dependency.optional,
        "supported": supported,
        "platform": platform_name,
        "installed": installed,
        "install_mode": next((hint.mode for hint in ordered_hints), "host"),
        "found_executables": found_executables,
        "found_isolated_executables": found_isolated_executables,
        "found_modules": found_modules,
        "missing_executables": [name for name in dependency.executables if name not in found_executables],
        "missing_modules": missing_modules,
        "install_hints": hints,
        "verify_command": command_to_text(dependency.verify_command) if dependency.verify_command else "",
        "verify_argv": list(dependency.verify_command),
        "notes": dependency.notes,
    }


def dependency_statuses(
    *,
    recommended_only: bool = False,
    missing_only: bool = False,
    workspace_root: str | Path | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dependency in TOOL_DEPENDENCIES:
        if recommended_only and not dependency.recommended:
            continue
        status = dependency_status(dependency, workspace_root=workspace_root)
        if missing_only and status["installed"]:
            continue
        rows.append(status)
    return rows


def first_install_hint(dependency: ToolDependency, platform_name: str | None = None) -> InstallHint | None:
    supported_hints = _ordered_install_hints(dependency.install_hints, platform_name)
    if not supported_hints:
        return None
    available = [hint for hint in supported_hints if hint.command and _resolve_executable(hint.command[0])]
    return (available or supported_hints)[0]


def install_isolated_venv(tool_id: str, packages: Iterable[str], *, workspace_root: str | Path = ".") -> Path:
    clean_tool_id = str(tool_id or "").strip()
    package_list = [str(package).strip() for package in packages if str(package).strip()]
    if not clean_tool_id:
        raise ValueError("tool_id is required")
    if not package_list:
        raise ValueError("at least one package is required")
    env_dir = Path(workspace_root) / ".ollama-code" / "tool-envs" / clean_tool_id
    env_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "venv", str(env_dir)], check=True)
    python_exe = _venv_bin_dir(env_dir) / ("python.exe" if os.name == "nt" else "python")
    subprocess.run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(python_exe), "-m", "pip", "install", *package_list], check=True)
    return env_dir


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: python -m ollama_code.tool_dependencies install-venv <tool-id> <packages...>", file=sys.stderr)
        return 2
    command = args.pop(0)
    if command == "install-venv":
        if len(args) < 2:
            print("usage: install-venv <tool-id> <packages...>", file=sys.stderr)
            return 2
        tool_id, *packages = args
        env_dir = install_isolated_venv(tool_id, packages, workspace_root=Path.cwd())
        print(f"installed isolated tool env: {env_dir}")
        return 0
    print(f"unknown command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
