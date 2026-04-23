from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from pathlib import Path, PureWindowsPath
from unittest.mock import patch
from uuid import uuid4

from ollama_code.cli import CliStatusRenderer, build_agent, build_parser, handle_meta_command


class DummyAgent:
    def __init__(self) -> None:
        self.model = "batiai/qwen3.6-35b:iq4"
        self.max_tool_rounds = 100
        self.max_agent_depth = 2
        self._approval = "ask"
        self._workspace = Path.cwd()
        self.saved_path: Path | None = None
        self.loaded_path: Path | PureWindowsPath | None = None
        self._session_path = Path("session.json").resolve()
        self._test_command = "python -m unittest -v"

    def workspace_root(self) -> Path:
        return self._workspace

    def session_path(self) -> Path | PureWindowsPath:
        return self._session_path

    def approval_mode(self) -> str:
        return self._approval

    def configured_test_command(self) -> str | None:
        return self._test_command

    def set_interrupt_event(self, event: object) -> None:
        return

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

    def load_session(self, path: str) -> Path | PureWindowsPath:
        if "\\" in path and len(path) >= 3 and path[1:3] == ":\\":
            self.loaded_path = PureWindowsPath(path)
        else:
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
        return ["batiai/qwen3.6-35b:iq4", "gemma4:latest"]


class CliCommandTests(unittest.TestCase):
    def _workspace_scratch(self) -> Path:
        root = (Path.cwd() / "verify_scratch" / f"test-cli-{uuid4().hex}").resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return root

    def test_model_command_updates_model(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        handled = handle_meta_command("/model gemma3:4b", agent, output.append)
        self.assertTrue(handled)
        self.assertEqual(agent.model, "gemma3:4b")

    def test_parser_defaults_max_tool_rounds_to_100(self) -> None:
        args = build_parser().parse_args([])
        self.assertIsNone(args.max_tool_rounds)

    def test_build_agent_uses_qwen_default_model(self) -> None:
        root = self._workspace_scratch()
        args = build_parser().parse_args(["--cwd", str(root), "--quiet"])

        agent = build_agent(args)

        self.assertEqual(agent.model, "batiai/qwen3.6-35b:iq4")
        self.assertEqual(agent.max_tool_rounds, 100)
        self.assertEqual(agent.max_agent_depth, 2)
        self.assertEqual(agent.approval_mode(), "ask")

    def test_status_renderer_shows_last_three_thinking_lines_and_clears_on_status(self) -> None:
        stream = io.StringIO()
        renderer = CliStatusRenderer(stream=stream, use_ansi=True)

        renderer.show_thinking("one\ntwo\nthree\nfour")
        renderer.status("tool read_file {}")

        output = stream.getvalue()
        self.assertNotIn("\x1b[90mone\x1b[0m", output)
        self.assertIn("\x1b[90mtwo\x1b[0m", output)
        self.assertIn("\x1b[90mthree\x1b[0m", output)
        self.assertIn("\x1b[90mfour\x1b[0m", output)
        self.assertIn("\x1b[1A", output)
        self.assertIn("[status] tool read_file {}", output)

    def test_build_agent_uses_config_file_for_runtime_defaults(self) -> None:
        root = self._workspace_scratch()
        config_dir = root / ".ollama-code"
        config_dir.mkdir(parents=True)
        (config_dir / "config.json").write_text(
            '{'
            '"host":"http://127.0.0.1:11435",'
            '"model":"gemma4:latest",'
            '"approval":"auto",'
            '"max_tool_rounds":12,'
            '"max_agent_depth":4,'
            '"timeout":45,'
            '"test_cmd":"python -m unittest -v"'
            '}',
            encoding="utf-8",
        )
        args = build_parser().parse_args(["--cwd", str(root), "--quiet"])

        agent = build_agent(args)

        self.assertEqual(agent.model, "gemma4:latest")
        self.assertEqual(agent.approval_mode(), "auto")
        self.assertEqual(agent.max_tool_rounds, 12)
        self.assertEqual(agent.max_agent_depth, 4)
        self.assertEqual(agent.configured_test_command(), "python -m unittest -v")

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
        self.assertIn("batiai/qwen3.6-35b:iq4", output[0])

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

    def test_load_command_preserves_windows_backslashes(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        with patch("ollama_code.cli.os.name", "nt"):
            handled = handle_meta_command(r"/load C:\Users\yorke\saved.json", agent, output.append)
        self.assertTrue(handled)
        self.assertIsNotNone(agent.loaded_path)
        self.assertIn(r"C:\Users\yorke\saved.json", str(agent.loaded_path))

    def test_diff_command_prints_git_diff(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        handled = handle_meta_command("/diff --cached tracked.txt", agent, output.append)
        self.assertTrue(handled)
        self.assertIn("+after", output[0])

    def test_test_command_preserves_windows_executable_path(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        command = r'"C:\Program Files\Python312\python.exe" -m unittest -q'
        with patch("ollama_code.cli.os.name", "nt"):
            handled = handle_meta_command(f"/test {command}", agent, output.append)
        self.assertTrue(handled)
        self.assertIn(command, output[0])

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
        root = self._workspace_scratch()
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

    def test_build_agent_resume_missing_session_raises_value_error(self) -> None:
        missing = f"missing-{uuid4().hex}.json"
        parser = build_parser()
        args = parser.parse_args(["--cwd", str(Path.cwd()), "--resume", missing, "--quiet"])

        with self.assertRaisesRegex(ValueError, "Transcript file not found"):
            build_agent(args)

    def test_build_agent_resume_does_not_rewrite_saved_transcript_on_startup(self) -> None:
        root = self._workspace_scratch()
        session = root / ".ollama-code" / "sessions" / "saved.json"
        session.parent.mkdir(parents=True)
        original = (
            '{'
            '"model":"saved-model",'
            '"approval_mode":"auto",'
            '"workspace_root":"'
            + root.as_posix()
            + '",'
            '"extra":"keep-me",'
            '"messages":[{"role":"system","content":"sys"},{"role":"user","content":"remember me"}],'
            '"events":[{"type":"user","content":"remember me"}]'
            '}'
        )
        session.write_text(original, encoding="utf-8")
        parser = build_parser()
        args = parser.parse_args(
            ["--cwd", str(root), "--resume", str(session), "--model", "override-model", "--approval", "ask", "--quiet"]
        )

        agent = build_agent(args)
        on_disk = session.read_text(encoding="utf-8")

        self.assertEqual(agent.model, "override-model")
        self.assertEqual(agent.approval_mode(), "ask")
        self.assertEqual(agent.messages[1]["content"], "remember me")
        self.assertEqual(on_disk, original)
