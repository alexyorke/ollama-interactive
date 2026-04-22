from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ollama_code.cli import build_agent, build_parser, handle_meta_command


class DummyAgent:
    def __init__(self) -> None:
        self.model = "batiai/gemma4-26b:iq4"
        self.max_tool_rounds = 8
        self.max_agent_depth = 2
        self._approval = "ask"
        self._workspace = Path.cwd()
        self.saved_path: Path | None = None
        self.loaded_path: Path | None = None
        self._session_path = Path("session.json").resolve()
        self._test_command = "python -m unittest -v"

    def workspace_root(self) -> Path:
        return self._workspace

    def session_path(self) -> Path:
        return self._session_path

    def approval_mode(self) -> str:
        return self._approval

    def configured_test_command(self) -> str | None:
        return self._test_command

    def set_model(self, model: str) -> None:
        self.model = model

    def set_approval_mode(self, mode: str) -> None:
        self._approval = mode

    def reset(self) -> None:
        pass

    def save_transcript(self, path: str | None = None) -> Path:
        self.saved_path = Path(path or "session.json").resolve()
        return self.saved_path

    def list_sessions(self, limit: int = 10) -> list[object]:
        class Session:
            def __init__(self, path: Path, model: str, approval_mode: str, message_count: int, summary: str) -> None:
                self.path = path
                self.model = model
                self.approval_mode = approval_mode
                self.message_count = message_count
                self.summary = summary
                self.updated_at = __import__("datetime").datetime(2026, 4, 21, 12, 0, 0)

        return [Session(Path("trace.json"), self.model, self._approval, 3, "Inspect repo")]

    def load_session(self, path: str) -> Path:
        self.loaded_path = Path(path).resolve()
        self._session_path = self.loaded_path
        return self.loaded_path

    def git_status(self, path: str | None = None) -> dict[str, object]:
        return {"ok": True, "output": "## main\n M tracked.txt"}

    def git_diff(self, *, cached: bool = False, path: str | None = None) -> dict[str, object]:
        return {"ok": True, "output": "--- a/tracked.txt\n+++ b/tracked.txt\n+after"}

    def git_commit(self, message: str, *, add_all: bool = True) -> dict[str, object]:
        return {"ok": True, "output": f"[main abc123] {message}"}

    def run_test(self, command: str | None = None) -> dict[str, object]:
        selected = command or self._test_command
        return {"ok": True, "output": f"ran {selected}"}

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

    def test_sessions_command_lists_saved_sessions(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        handled = handle_meta_command("/sessions", agent, output.append)
        self.assertTrue(handled)
        self.assertIn("trace.json", output[0])

    def test_load_command_updates_session(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / "saved.json")
            handled = handle_meta_command(f"/load {target}", agent, output.append)
        self.assertTrue(handled)
        self.assertIsNotNone(agent.loaded_path)
        self.assertTrue(str(agent.loaded_path).endswith("saved.json"))

    def test_diff_command_prints_git_diff(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        handled = handle_meta_command("/diff --cached tracked.txt", agent, output.append)
        self.assertTrue(handled)
        self.assertIn("+after", output[0])

    def test_commit_command_prints_commit_output(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        handled = handle_meta_command('/commit "Save work"', agent, output.append)
        self.assertTrue(handled)
        self.assertIn("Save work", output[0])

    def test_test_command_prints_test_output(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        handled = handle_meta_command("/test pytest -q", agent, output.append)
        self.assertTrue(handled)
        self.assertIn("pytest -q", output[0])

    def test_build_agent_continue_restores_latest_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session_dir = root / ".ollama-code" / "sessions"
            session_dir.mkdir(parents=True)
            session = session_dir / "saved.json"
            session.write_text(
                '{"model":"gemma4:latest","approval_mode":"auto","workspace_root":"'
                + root.as_posix()
                + '","messages":[{"role":"system","content":"sys"},{"role":"user","content":"remember me"}],"events":[{"type":"user","content":"remember me"}]}',
                encoding="utf-8",
            )
            parser = build_parser()
            args = parser.parse_args(["--cwd", str(root), "--continue", "--quiet"])
            agent = build_agent(args)

        self.assertEqual(agent.model, "gemma4:latest")
        self.assertEqual(agent.approval_mode(), "auto")
        self.assertEqual(agent.messages[1]["content"], "remember me")
