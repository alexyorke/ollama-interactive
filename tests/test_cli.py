from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ollama_code.cli import handle_meta_command


class DummyAgent:
    def __init__(self) -> None:
        self.model = "batiai/gemma4-26b:iq4"
        self.max_tool_rounds = 8
        self.max_agent_depth = 2
        self._approval = "ask"
        self._workspace = Path.cwd()
        self.saved_path: Path | None = None

    def workspace_root(self) -> Path:
        return self._workspace

    def approval_mode(self) -> str:
        return self._approval

    def set_model(self, model: str) -> None:
        self.model = model

    def set_approval_mode(self, mode: str) -> None:
        self._approval = mode

    def reset(self) -> None:
        pass

    def save_transcript(self, path: str | None = None) -> Path:
        self.saved_path = Path(path or "session.json").resolve()
        return self.saved_path

    def tool_help(self) -> str:
        return "list_files\nread_file"

    def list_models(self) -> list[str]:
        return ["batiai/gemma4-26b:iq4", "gemma4:latest"]


class CliCommandTests(unittest.TestCase):
    def test_model_command_updates_model(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        handled = handle_meta_command("/model gemma3:4b", agent, output.append)
        self.assertTrue(handled)
        self.assertEqual(agent.model, "gemma3:4b")

    def test_approval_command_updates_mode(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        handled = handle_meta_command("/approval auto", agent, output.append)
        self.assertTrue(handled)
        self.assertEqual(agent.approval_mode(), "auto")

    def test_save_command_uses_custom_path(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / "trace.json")
            handled = handle_meta_command(f"/save {target}", agent, output.append)
        self.assertTrue(handled)
        self.assertIsNotNone(agent.saved_path)
        self.assertTrue(str(agent.saved_path).endswith("trace.json"))

    def test_quit_command_returns_false(self) -> None:
        agent = DummyAgent()
        handled = handle_meta_command("/quit", agent, lambda _: None)
        self.assertFalse(handled)

    def test_models_command_lists_models(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        handled = handle_meta_command("/models", agent, output.append)
        self.assertTrue(handled)
        self.assertIn("batiai/gemma4-26b:iq4", output[0])
