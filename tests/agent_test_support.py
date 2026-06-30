from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from ollama_code.features import ENV_OLLAMA_CODE_FEATURE_PROFILE


class AgentTestBase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._env_patcher = patch.dict(
            "os.environ",
            {
                ENV_OLLAMA_CODE_FEATURE_PROFILE: "",
                "OLLAMA_CODE_MODEL": "",
                "OLLAMA_CODE_REQUIRE_LLM_FOR_TURN": "",
            },
            clear=False,
        )
        self._env_patcher.start()
        self.addCleanup(self._env_patcher.stop)

    def _workspace_scratch(self) -> Path:
        root = (Path.cwd() / "verify_scratch" / f"test-agent-{uuid4().hex}").resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return root

    def _init_git_repo_or_skip(self, root: Path) -> None:
        if shutil.which("git") is None:
            self.skipTest("git is not installed")
        try:
            subprocess.run(["git", "init"], cwd=root, capture_output=True, text=True, check=True)
            subprocess.run(["git", "config", "user.name", "Tests"], cwd=root, capture_output=True, text=True, check=True)
            subprocess.run(["git", "config", "user.email", "tests@example.com"], cwd=root, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            message = (exc.stderr or exc.stdout or str(exc)).strip()
            self.skipTest(f"git repo init is unavailable in this environment: {message}")
