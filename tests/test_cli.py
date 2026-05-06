from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from pathlib import Path, PureWindowsPath
from unittest.mock import patch
from uuid import uuid4

from ollama_code.cli import CliStatusRenderer, build_agent, build_parser, doctor_report, ensure_runtime_default_model, handle_meta_command, main, startup_help_text
from ollama_code.config import DEFAULT_MODEL


class DummyAgent:
    def __init__(self) -> None:
        class ToolView:
            def available_tool_names(self) -> set[str]:
                return {
                    "verified_function_index",
                    "verified_function_search",
                    "verified_function_show",
                    "verify_function_contract",
                    "compose_verified_functions",
                    "promote_verified_function",
                }

        self.model = DEFAULT_MODEL
        self.max_tool_rounds = 100
        self.max_agent_depth = 2
        self._approval = "ask"
        self._debate_enabled = True
        self._reconcile_mode = "auto"
        self._verifier_model: str | None = None
        self._workspace = Path.cwd()
        self.saved_path: Path | None = None
        self.loaded_path: Path | PureWindowsPath | None = None
        self._session_path = Path("session.json").resolve()
        self._test_command = "python -m unittest -v"
        self._todos = "1. [pending] inspect\n2. [completed] setup"
        self.tools = ToolView()

    def workspace_root(self) -> Path:
        return self._workspace

    def session_path(self) -> Path | PureWindowsPath:
        return self._session_path

    def approval_mode(self) -> str:
        return self._approval

    def debate_mode(self) -> bool:
        return self._debate_enabled

    def reconcile_mode(self) -> str:
        return self._reconcile_mode

    def configured_test_command(self) -> str | None:
        return self._test_command

    def verifier_model_name(self) -> str | None:
        return self._verifier_model

    def set_interrupt_event(self, event: object) -> None:
        return

    def set_model(self, model: str) -> None:
        self.model = model

    def set_approval_mode(self, mode: str) -> None:
        self._approval = mode

    def set_debate_enabled(self, enabled: bool) -> None:
        self._debate_enabled = enabled

    def set_reconcile_mode(self, mode: str) -> None:
        self._reconcile_mode = mode

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

    def tool_help(self, *, compact: bool = False) -> str:
        return "list_files()\nread_file(path)" if compact else "list_files\nread_file"

    def todo_read(self) -> dict[str, object]:
        return {"ok": True, "output": self._todos}

    def todo_clear(self) -> dict[str, object]:
        self._todos = "(empty)"
        return {"ok": True, "output": self._todos}

    def list_models(self) -> list[str]:
        return [DEFAULT_MODEL, "gemma4:latest"]

    def index_status(self) -> dict[str, object]:
        return {
            "ok": True,
            "enabled": True,
            "running": False,
            "ready": True,
            "pending_paths": 0,
            "refresh_count": 1,
            "summary": "indexed files=2 code=1 fts=2",
            "cache_dir": str((Path.cwd() / ".ollama-code" / "index").resolve(strict=False)),
        }

    def refresh_index(self) -> dict[str, object]:
        return {"ok": True, "summary": "index refresh queued"}

    def start_indexer(self) -> bool:
        return True

    def stop_indexer(self) -> None:
        return


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

    def test_build_agent_uses_config_default_model(self) -> None:
        root = self._workspace_scratch()
        args = build_parser().parse_args(["--cwd", str(root), "--quiet"])

        agent = build_agent(args)

        self.assertEqual(agent.model, DEFAULT_MODEL)
        self.assertEqual(agent.max_tool_rounds, 100)
        self.assertEqual(agent.max_agent_depth, 2)
        self.assertEqual(agent.approval_mode(), "ask")
        self.assertTrue(agent.debate_mode())
        self.assertEqual(agent.reconcile_mode(), "auto")
        self.assertTrue(agent.index_status()["enabled"])
        for name in (
            "verified_function_index",
            "verified_function_search",
            "verified_function_show",
            "verify_function_contract",
            "compose_verified_functions",
            "promote_verified_function",
        ):
            self.assertIn(name, agent.tools.available_tool_names())

    def test_status_renderer_shows_last_three_thinking_lines_and_clears_on_status(self) -> None:
        stream = io.StringIO()
        renderer = CliStatusRenderer(stream=stream, use_ansi=True, update_interval=0.0)

        renderer.show_thinking("one\ntwo\nthree\nfour")
        renderer.status("tool read_file {}")

        output = stream.getvalue()
        self.assertNotIn("\x1b[90mone\x1b[0m", output)
        self.assertIn("\x1b[90mtwo\x1b[0m", output)
        self.assertIn("\x1b[90mthree\x1b[0m", output)
        self.assertIn("\x1b[90mfour\x1b[0m", output)
        self.assertIn("\x1b[1A", output)
        self.assertIn("[status] tool read_file {}", output)

    def test_runtime_default_fallback_prints_default_pull_hint(self) -> None:
        root = self._workspace_scratch()
        args = build_parser().parse_args(["--cwd", str(root)])
        agent = DummyAgent()
        agent.list_models = lambda: ["gemma3:4b"]  # type: ignore[method-assign]
        stream = io.StringIO()
        renderer = CliStatusRenderer(stream=stream, use_ansi=False, update_interval=0.0)

        ensure_runtime_default_model(agent, args, renderer)

        self.assertEqual(agent.model, "gemma3:4b")
        self.assertEqual(agent.model_source, "runtime fallback:gemma3:4b")
        self.assertIn(f"ollama pull {DEFAULT_MODEL}", stream.getvalue())

    def test_runtime_default_fallback_ignores_unpreferred_custom_model(self) -> None:
        root = self._workspace_scratch()
        args = build_parser().parse_args(["--cwd", str(root)])
        agent = DummyAgent()
        agent.list_models = lambda: ["hf.co/batiai/Granite-4.1-8B-GGUF:IQ4_XS"]  # type: ignore[method-assign]
        stream = io.StringIO()
        renderer = CliStatusRenderer(stream=stream, use_ansi=False, update_interval=0.0)

        ensure_runtime_default_model(agent, args, renderer)

        self.assertEqual(agent.model, DEFAULT_MODEL)
        self.assertIn("no preferred fallback model is available", stream.getvalue())

    def test_runtime_default_fallback_respects_quiet(self) -> None:
        root = self._workspace_scratch()
        args = build_parser().parse_args(["--cwd", str(root), "--quiet"])
        agent = DummyAgent()
        agent.list_models = lambda: ["gemma3:4b"]  # type: ignore[method-assign]
        stream = io.StringIO()
        renderer = CliStatusRenderer(stream=stream, use_ansi=False, update_interval=0.0)

        ensure_runtime_default_model(agent, args, renderer, quiet=True)

        self.assertEqual(agent.model, "gemma3:4b")
        self.assertEqual(stream.getvalue(), "")

    def test_startup_help_text_explains_basic_usage(self) -> None:
        agent = DummyAgent()

        text = startup_help_text(agent)

        self.assertIn("Ollama Code", text)
        self.assertIn("workspace:", text)
        self.assertIn(f"model: {DEFAULT_MODEL}", text)
        self.assertIn("Type a coding request", text)
        self.assertIn("/status", text)
        self.assertIn("/approval ask|auto|read-only", text)
        self.assertIn("/doctor", text)
        self.assertIn("/index", text)
        self.assertIn("/todos", text)
        self.assertIn("/tools", text)
        self.assertIn("verified-function cards are enabled by default", text)
        self.assertIn("Press Esc", text)

    def test_help_command_prints_multiline_command_reference(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        handled = handle_meta_command("/help", agent, output.append)

        self.assertTrue(handled)
        self.assertEqual(len(output), 1)
        self.assertIn("Slash commands:", output[0])
        self.assertIn("/model <name>", output[0])
        self.assertIn("/test [command]", output[0])
        self.assertIn("/doctor", output[0])
        self.assertIn("/index", output[0])
        self.assertIn("/todos", output[0])
        self.assertIn("fix failing tests", output[0])

    def test_doctor_report_checks_first_use_setup(self) -> None:
        agent = DummyAgent()
        agent.config_path = Path("settings.json").resolve()
        agent.model_source = "config:settings.json"

        with patch("ollama_code.cli.shutil.which", return_value=None):
            report, ok = doctor_report(agent)

        self.assertTrue(ok)
        self.assertIn("Ollama Code doctor", report)
        self.assertIn("workspace: ok", report)
        self.assertIn("config: ok", report)
        self.assertIn("model_source: config:settings.json", report)
        self.assertIn("ollama: ok", report)
        self.assertIn(f"model: ok {DEFAULT_MODEL}", report)
        self.assertIn("indexer: ok enabled", report)
        self.assertIn("verified functions: ok default-on", report)
        self.assertIn("console script: not on PATH", report)

    def test_doctor_command_prints_report(self) -> None:
        agent = DummyAgent()
        output: list[str] = []

        with patch("ollama_code.cli.shutil.which", return_value=None):
            handled = handle_meta_command("/doctor", agent, output.append)

        self.assertTrue(handled)
        self.assertEqual(len(output), 1)
        self.assertIn("Ollama Code doctor", output[0])

    def test_status_command_prints_config_and_model_source(self) -> None:
        agent = DummyAgent()
        agent.config_path = Path("settings.json").resolve()
        agent.model_source = "config:settings.json"
        output: list[str] = []

        handled = handle_meta_command("/status", agent, output.append)

        self.assertTrue(handled)
        self.assertIn("model_source=config:settings.json", output[0])
        self.assertIn("config=", output[0])

    def test_main_doctor_skips_runtime_model_resolution(self) -> None:
        agent = DummyAgent()
        stream = io.StringIO()

        with patch("ollama_code.cli.build_agent", return_value=agent):
            with patch("ollama_code.cli.ensure_runtime_default_model", side_effect=AssertionError("should not resolve model")):
                with patch("sys.stdout", stream):
                    exit_code = main(["--doctor", "--quiet"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Ollama Code doctor", stream.getvalue())

    def test_tools_command_is_compact_by_default_and_full_on_request(self) -> None:
        agent = DummyAgent()
        output: list[str] = []

        compact = handle_meta_command("/tools", agent, output.append)
        full = handle_meta_command("/tools full", agent, output.append)

        self.assertTrue(compact)
        self.assertTrue(full)
        self.assertEqual(output[0], "list_files()\nread_file(path)")
        self.assertEqual(output[1], "list_files\nread_file")

    def test_todos_command_shows_and_clears_todo_list(self) -> None:
        agent = DummyAgent()
        output: list[str] = []

        shown = handle_meta_command("/todos", agent, output.append)
        cleared = handle_meta_command("/todos clear", agent, output.append)

        self.assertTrue(shown)
        self.assertTrue(cleared)
        self.assertIn("[pending] inspect", output[0])
        self.assertEqual(output[1], "(empty)")

    def test_index_command_reports_and_controls_indexer(self) -> None:
        agent = DummyAgent()
        output: list[str] = []

        status = handle_meta_command("/index status", agent, output.append)
        refresh = handle_meta_command("/index refresh", agent, output.append)
        stop = handle_meta_command("/index stop", agent, output.append)
        start = handle_meta_command("/index start", agent, output.append)

        self.assertTrue(status)
        self.assertTrue(refresh)
        self.assertTrue(stop)
        self.assertTrue(start)
        self.assertIn("indexer enabled=yes", output[0])
        self.assertEqual(output[1], "index refresh queued")
        self.assertEqual(output[2], "indexer stopped")
        self.assertEqual(output[3], "indexer started")

    def test_status_renderer_skips_redundant_thinking_redraws(self) -> None:
        stream = io.StringIO()
        renderer = CliStatusRenderer(stream=stream, use_ansi=True, update_interval=0.0)

        renderer.show_thinking("one\ntwo\nthree")
        renderer.show_thinking("one\ntwo\nthree")

        output = stream.getvalue()
        self.assertEqual(output.count("\x1b[90mone\x1b[0m\n"), 1)
        self.assertEqual(output.count("\x1b[90mtwo\x1b[0m\n"), 1)
        self.assertEqual(output.count("\x1b[90mthree\x1b[0m\n"), 1)

    def test_status_renderer_rewrites_only_last_line_when_tail_prefix_matches(self) -> None:
        stream = io.StringIO()
        renderer = CliStatusRenderer(stream=stream, use_ansi=True, update_interval=0.0)

        renderer.show_thinking("one\ntwo\nthree")
        renderer.show_thinking("one\ntwo\nthree more")

        output = stream.getvalue()
        self.assertEqual(output.count("\x1b[90mone\x1b[0m\n"), 1)
        self.assertEqual(output.count("\x1b[90mtwo\x1b[0m\n"), 1)
        self.assertIn("\x1b[1A\r\x1b[2K\x1b[90mthree more\x1b[0m\n", output)
        self.assertNotIn("\x1b[M", output)

    def test_build_agent_uses_config_file_for_runtime_defaults(self) -> None:
        root = self._workspace_scratch()
        config_dir = root / ".ollama-code"
        config_dir.mkdir(parents=True)
        (config_dir / "config.json").write_text(
            '{'
            '"host":"http://127.0.0.1:11435",'
            '"model":"gemma4:latest",'
            '"approval":"auto",'
            '"debate":false,'
            '"reconcile":"on",'
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
        self.assertFalse(agent.debate_mode())
        self.assertEqual(agent.reconcile_mode(), "on")

    def test_approval_command_updates_mode(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        handled = handle_meta_command("/approval auto", agent, output.append)
        self.assertTrue(handled)
        self.assertEqual(agent.approval_mode(), "auto")

    def test_debate_command_updates_mode(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        handled = handle_meta_command("/debate off", agent, output.append)
        self.assertTrue(handled)
        self.assertFalse(agent.debate_mode())
        self.assertIn("debate set to off", output[0])

    def test_reconcile_command_updates_mode(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        handled = handle_meta_command("/reconcile on", agent, output.append)
        self.assertTrue(handled)
        self.assertEqual(agent.reconcile_mode(), "on")
        self.assertIn("reconcile set to on", output[0])

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
        self.assertIn(DEFAULT_MODEL, output[0])

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
