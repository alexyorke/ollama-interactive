from __future__ import annotations

import io
import os
import shutil
import tempfile
import unittest
from pathlib import Path, PureWindowsPath
from unittest.mock import patch
from uuid import uuid4

from ollama_code.agent import TRANSCRIPT_DIAGNOSTIC_COLLECTION_LIMIT, TRANSCRIPT_DIAGNOSTIC_DICT_MARKER_KEY
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
                    "python_sdk_search",
                    "python_sdk_refresh",
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
        self.test_requests: list[str | None] = []
        self.todo_requests: list[str] = []
        self.index_requests: list[str] = []
        self._todos = "1. [pending] inspect\n2. [completed] setup"
        self.diff_request: dict[str, object] | None = None
        self.install_requests: list[dict[str, object]] = []
        self.commit_requests: list[str] = []
        self.reset_requests = 0
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
        self.reset_requests += 1

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
        self.diff_request = {"cached": cached, "path": path}
        return {"ok": True, "output": "--- a/tracked.txt\n+++ b/tracked.txt\n+after"}

    def git_commit(self, message: str, *, add_all: bool = True) -> dict[str, object]:
        self.commit_requests.append(message)
        return {"ok": True, "output": f"[main abc123] {message}"}

    def run_test(self, command: str | None = None) -> dict[str, object]:
        self.test_requests.append(command)
        selected = command or self._test_command
        return {"ok": True, "output": f"ran {selected}"}

    def tool_help(self, *, compact: bool = False) -> str:
        return "list_files()\nread_file(path)" if compact else "list_files\nread_file"

    def tool_group_help(self) -> str:
        return "navigation: tools=list_files, read_file"

    def tool_dependency_status(self, scope: str = "all", tool_id: str | None = None) -> dict[str, object]:
        return {"ok": True, "output": f"tool dependency status scope={scope} tool_id={tool_id or '-'}"}

    def tool_dependency_install(self, tool_id: str | None = None, *, all_recommended: bool = False, confirm: bool = False) -> dict[str, object]:
        self.install_requests.append({"tool_id": tool_id, "all_recommended": all_recommended, "confirm": confirm})
        target = "--recommended" if all_recommended else tool_id
        return {"ok": True, "output": f"install target={target} confirm={confirm}"}

    def todo_read(self) -> dict[str, object]:
        self.todo_requests.append("read")
        return {"ok": True, "output": self._todos}

    def todo_clear(self) -> dict[str, object]:
        self.todo_requests.append("clear")
        self._todos = "(empty)"
        return {"ok": True, "output": self._todos}

    def list_models(self) -> list[str]:
        return [DEFAULT_MODEL, "gemma4:latest"]

    def index_status(self) -> dict[str, object]:
        self.index_requests.append("status")
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
        self.index_requests.append("refresh")
        return {"ok": True, "summary": "index refresh queued"}

    def start_indexer(self) -> bool:
        self.index_requests.append("start")
        return True

    def stop_indexer(self) -> None:
        self.index_requests.append("stop")
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

    def test_model_command_rejects_extra_args(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        handled = handle_meta_command("/model gemma3:4b trailing", agent, output.append)
        self.assertTrue(handled)
        self.assertEqual(agent.model, DEFAULT_MODEL)
        self.assertEqual(output, ["Usage: /model <name>"])

    def test_parser_defaults_max_tool_rounds_to_100(self) -> None:
        args = build_parser().parse_args([])
        self.assertIsNone(args.max_tool_rounds)
        self.assertFalse(args.disable_spec_guided_repair)
        self.assertFalse(args.require_llm_for_turn)

    def test_build_agent_rejects_non_positive_runtime_limits(self) -> None:
        root = self._workspace_scratch()
        parser = build_parser()
        cases = [
            (["--cwd", str(root), "--quiet", "--max-tool-rounds", "0"], "--max-tool-rounds must be a positive integer."),
            (["--cwd", str(root), "--quiet", "--max-agent-depth", "-1"], "--max-agent-depth must be a positive integer."),
            (["--cwd", str(root), "--quiet", "--timeout", "0"], "--timeout must be a positive integer."),
        ]

        for argv, message in cases:
            with self.subTest(argv=argv):
                args = parser.parse_args(argv)
                with self.assertRaisesRegex(ValueError, message):
                    build_agent(args)

    def test_build_agent_rejects_empty_session_path_flags(self) -> None:
        root = self._workspace_scratch()
        parser = build_parser()
        cases = [
            (["--cwd", str(root), "--quiet", "--resume", ""], "--resume must be a non-empty path."),
            (["--cwd", str(root), "--quiet", "--resume", "   "], "--resume must be a non-empty path."),
            (["--cwd", str(root), "--quiet", "--session-file", ""], "--session-file must be a non-empty path."),
            (["--cwd", str(root), "--quiet", "--session-file", "   "], "--session-file must be a non-empty path."),
        ]

        for argv, message in cases:
            with self.subTest(argv=argv):
                args = parser.parse_args(argv)
                with self.assertRaisesRegex(ValueError, message):
                    build_agent(args)

    def test_build_agent_rejects_empty_startup_path_flags(self) -> None:
        root = self._workspace_scratch()
        parser = build_parser()
        cases = [
            (["--cwd", "", "--quiet"], "--cwd must be a non-empty path."),
            (["--cwd", "   ", "--quiet"], "--cwd must be a non-empty path."),
            (["--cwd", str(root), "--quiet", "--config", ""], "--config must be a non-empty path."),
            (["--cwd", str(root), "--quiet", "--config", "   "], "--config must be a non-empty path."),
        ]

        for argv, message in cases:
            with self.subTest(argv=argv):
                args = parser.parse_args(argv)
                with self.assertRaisesRegex(ValueError, message):
                    build_agent(args)

    def test_build_agent_rejects_empty_startup_text_flags(self) -> None:
        root = self._workspace_scratch()
        parser = build_parser()
        cases = [
            (["--cwd", str(root), "--quiet", "--model", ""], "--model must be a non-empty value."),
            (["--cwd", str(root), "--quiet", "--model", "   "], "--model must be a non-empty value."),
            (["--cwd", str(root), "--quiet", "--host", ""], "--host must be a non-empty value."),
            (["--cwd", str(root), "--quiet", "--host", "   "], "--host must be a non-empty value."),
            (["--cwd", str(root), "--quiet", "--verifier-model", ""], "--verifier-model must be a non-empty value."),
            (["--cwd", str(root), "--quiet", "--verifier-model", "   "], "--verifier-model must be a non-empty value."),
            (["--cwd", str(root), "--quiet", "--test-cmd", ""], "--test-cmd must be a non-empty value."),
            (["--cwd", str(root), "--quiet", "--test-cmd", "   "], "--test-cmd must be a non-empty value."),
        ]

        for argv, message in cases:
            with self.subTest(argv=argv):
                args = parser.parse_args(argv)
                with self.assertRaisesRegex(ValueError, message):
                    build_agent(args)

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
            "python_sdk_search",
            "python_sdk_refresh",
        ):
            self.assertIn(name, agent.tools.available_tool_names())

    @unittest.skipUnless(os.name != "nt", "POSIX only")
    def test_build_agent_reads_windows_style_cwd_on_posix(self) -> None:
        root = self._workspace_scratch()
        config = root / ".ollama-code" / "config.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text('{"model":"cwd-alias-model"}', encoding="utf-8")
        root_path = root.as_posix()
        if not root_path.startswith("/mnt/") or len(root_path) < 8 or root_path[6] != "/":
            self.skipTest("requires a /mnt/<drive> workspace path")
        windows_style = f"{root_path[5].upper()}:{root_path[6:]}"
        args = build_parser().parse_args(["--cwd", windows_style, "--quiet"])

        agent = build_agent(args)

        self.assertEqual(agent.workspace_root().resolve(strict=False), root.resolve(strict=False))
        self.assertEqual(agent.model, "cwd-alias-model")

    @unittest.skipUnless(os.name != "nt", "POSIX only")
    def test_build_agent_reads_backslash_relative_cwd_on_posix(self) -> None:
        root = self._workspace_scratch()
        config = root / ".ollama-code" / "config.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text('{"model":"cwd-relative-model"}', encoding="utf-8")
        relative = root.relative_to(Path.cwd()).as_posix().replace("/", "\\")
        args = build_parser().parse_args(["--cwd", relative, "--quiet"])

        agent = build_agent(args)

        self.assertEqual(agent.workspace_root().resolve(strict=False), root.resolve(strict=False))
        self.assertEqual(agent.model, "cwd-relative-model")

    @unittest.skipUnless(os.name == "nt", "Windows only")
    def test_build_agent_reads_wsl_alias_cwd_on_windows(self) -> None:
        root = self._workspace_scratch()
        config = root / ".ollama-code" / "config.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text('{"model":"cwd-alias-model"}', encoding="utf-8")
        alias = f"/mnt/{root.drive[:1].lower()}{root.as_posix()[2:]}"
        args = build_parser().parse_args(["--cwd", alias, "--quiet"])

        agent = build_agent(args)

        self.assertEqual(agent.workspace_root().resolve(strict=False), root.resolve(strict=False))
        self.assertEqual(agent.model, "cwd-alias-model")

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

    def test_build_agent_respects_disable_spec_guided_repair_env(self) -> None:
        root = self._workspace_scratch()
        with patch.dict("os.environ", {"OLLAMA_CODE_DISABLE_SPEC_GUIDED_REPAIR": "1"}):
            args = build_parser().parse_args(["--cwd", str(root), "--quiet"])
            agent = build_agent(args)
        self.assertTrue(agent.disable_spec_guided_repair)

    def test_build_agent_respects_require_llm_for_turn_flag(self) -> None:
        root = self._workspace_scratch()
        args = build_parser().parse_args(["--cwd", str(root), "--quiet", "--require-llm-for-turn"])
        agent = build_agent(args)
        self.assertTrue(agent.require_llm_for_turn)

    def test_build_agent_respects_require_llm_for_turn_env(self) -> None:
        root = self._workspace_scratch()
        with patch.dict("os.environ", {"OLLAMA_CODE_REQUIRE_LLM_FOR_TURN": "1"}):
            args = build_parser().parse_args(["--cwd", str(root), "--quiet"])
            agent = build_agent(args)
        self.assertTrue(agent.require_llm_for_turn)

    def test_runtime_default_fallback_prints_default_pull_hint(self) -> None:
        root = self._workspace_scratch()
        args = build_parser().parse_args(["--cwd", str(root)])
        agent = DummyAgent()
        agent.list_models = lambda: ["gemma3:4b"]  # type: ignore[method-assign]
        stream = io.StringIO()
        renderer = CliStatusRenderer(stream=stream, use_ansi=False, update_interval=0.0)

        ensure_runtime_default_model(agent, args, renderer, allow_model_fallback=True)

        self.assertEqual(agent.model, "gemma3:4b")
        self.assertEqual(agent.model_source, "runtime fallback:gemma3:4b")
        self.assertIn(f"ollama pull {DEFAULT_MODEL}", stream.getvalue())

    def test_runtime_default_fallback_can_repair_stale_configured_model(self) -> None:
        root = self._workspace_scratch()
        args = build_parser().parse_args(["--cwd", str(root)])
        agent = DummyAgent()
        agent.model = "stale-model"
        agent.list_models = lambda: ["gemma3:4b"]  # type: ignore[method-assign]
        stream = io.StringIO()
        renderer = CliStatusRenderer(stream=stream, use_ansi=False, update_interval=0.0)

        ensure_runtime_default_model(agent, args, renderer, allow_model_fallback=True)

        self.assertEqual(agent.model, "gemma3:4b")
        self.assertIn("default model", stream.getvalue())

    def test_runtime_default_keeps_explicit_model_when_not_fallback_enabled(self) -> None:
        root = self._workspace_scratch()
        args = build_parser().parse_args(["--cwd", str(root)])
        agent = DummyAgent()
        agent.model = "stale-explicit"
        agent.list_models = lambda: ["gemma3:4b"]  # type: ignore[method-assign]
        stream = io.StringIO()
        renderer = CliStatusRenderer(stream=stream, use_ansi=False, update_interval=0.0)

        ensure_runtime_default_model(agent, args, renderer, allow_model_fallback=False)

        self.assertEqual(agent.model, "stale-explicit")
        self.assertIn("configured model stale-explicit is not installed", stream.getvalue())

    def test_runtime_default_fallback_ignores_unpreferred_custom_model(self) -> None:
        root = self._workspace_scratch()
        args = build_parser().parse_args(["--cwd", str(root)])
        agent = DummyAgent()
        agent.list_models = lambda: ["hf.co/batiai/Granite-4.1-8B-GGUF:IQ4_XS"]  # type: ignore[method-assign]
        stream = io.StringIO()
        renderer = CliStatusRenderer(stream=stream, use_ansi=False, update_interval=0.0)

        ensure_runtime_default_model(agent, args, renderer, allow_model_fallback=True)

        self.assertEqual(agent.model, DEFAULT_MODEL)
        self.assertIn("no preferred fallback model is available", stream.getvalue())

    def test_runtime_default_fallback_respects_quiet(self) -> None:
        root = self._workspace_scratch()
        args = build_parser().parse_args(["--cwd", str(root), "--quiet"])
        agent = DummyAgent()
        agent.list_models = lambda: ["gemma3:4b"]  # type: ignore[method-assign]
        stream = io.StringIO()
        renderer = CliStatusRenderer(stream=stream, use_ansi=False, update_interval=0.0)

        ensure_runtime_default_model(agent, args, renderer, quiet=True, allow_model_fallback=True)

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
        self.assertIn("python sdk embeddings:", report)
        self.assertIn("console script: not on PATH", report)

    def test_doctor_report_shows_configured_sdk_embedding_model(self) -> None:
        agent = DummyAgent()

        with patch("ollama_code.cli.shutil.which", return_value=None):
            with patch.dict("os.environ", {"OLLAMA_CODE_SDK_EMBED_MODEL": "qwen3-embedding:8b"}):
                report, ok = doctor_report(agent)

        self.assertTrue(ok)
        self.assertIn("python sdk embeddings: ok on-demand candidate rerank via qwen3-embedding:8b", report)

    def test_doctor_command_prints_report(self) -> None:
        agent = DummyAgent()
        output: list[str] = []

        with patch("ollama_code.cli.shutil.which", return_value=None):
            handled = handle_meta_command("/doctor", agent, output.append)

        self.assertTrue(handled)
        self.assertEqual(len(output), 1)
        self.assertIn("Ollama Code doctor", output[0])

    def test_doctor_command_rejects_extra_args(self) -> None:
        agent = DummyAgent()
        output: list[str] = []

        handled = handle_meta_command('/doctor "   "', agent, output.append)

        self.assertTrue(handled)
        self.assertEqual(output, ["Usage: /doctor"])

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
        groups = handle_meta_command("/tools groups", agent, output.append)

        self.assertTrue(compact)
        self.assertTrue(full)
        self.assertTrue(groups)
        self.assertEqual(output[0], "list_files()\nread_file(path)")
        self.assertEqual(output[1], "list_files\nread_file")
        self.assertEqual(output[2], "navigation: tools=list_files, read_file")

    def test_tools_command_reports_and_installs_optional_dependencies(self) -> None:
        agent = DummyAgent()
        output: list[str] = []

        missing = handle_meta_command("/tools missing", agent, output.append)
        install = handle_meta_command("/tools install ruff", agent, output.append)
        recommended = handle_meta_command("/tools install --recommended", agent, output.append)

        self.assertTrue(missing)
        self.assertTrue(install)
        self.assertTrue(recommended)
        self.assertIn("scope=missing", output[0])
        self.assertIn("install target=ruff confirm=True", output[1])
        self.assertIn("install target=--recommended confirm=True", output[2])
        self.assertEqual(
            agent.install_requests,
            [
                {"tool_id": "ruff", "all_recommended": False, "confirm": True},
                {"tool_id": None, "all_recommended": True, "confirm": True},
            ],
        )

    def test_tools_install_command_rejects_extra_args(self) -> None:
        agent = DummyAgent()
        output: list[str] = []

        handled = handle_meta_command("/tools install ruff extra", agent, output.append)

        self.assertTrue(handled)
        self.assertEqual(output, ["Usage: /tools install <tool-id>|--recommended"])
        self.assertEqual(agent.install_requests, [])

    def test_tools_install_command_rejects_empty_target(self) -> None:
        agent = DummyAgent()
        output: list[str] = []

        handled = handle_meta_command('/tools install ""', agent, output.append)

        self.assertTrue(handled)
        self.assertEqual(output, ["Usage: /tools install <tool-id>|--recommended"])
        self.assertEqual(agent.install_requests, [])

    def test_tools_install_command_rejects_whitespace_only_target(self) -> None:
        agent = DummyAgent()
        output: list[str] = []

        handled = handle_meta_command('/tools install "   "', agent, output.append)

        self.assertTrue(handled)
        self.assertEqual(output, ["Usage: /tools install <tool-id>|--recommended"])
        self.assertEqual(agent.install_requests, [])

    def test_todos_command_shows_and_clears_todo_list(self) -> None:
        agent = DummyAgent()
        output: list[str] = []

        shown = handle_meta_command("/todos", agent, output.append)
        cleared = handle_meta_command("/todos clear", agent, output.append)

        self.assertTrue(shown)
        self.assertTrue(cleared)
        self.assertIn("[pending] inspect", output[0])
        self.assertEqual(output[1], "(empty)")

    def test_todos_command_rejects_explicit_empty_subcommand(self) -> None:
        agent = DummyAgent()
        output: list[str] = []

        handled = handle_meta_command('/todos ""', agent, output.append)

        self.assertTrue(handled)
        self.assertEqual(output, ["Usage: /todos [clear]"])
        self.assertEqual(agent.todo_requests, [])

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

    def test_index_command_rejects_whitespace_only_quoted_subcommand(self) -> None:
        agent = DummyAgent()
        output: list[str] = []

        handled = handle_meta_command('/index "   "', agent, output.append)

        self.assertTrue(handled)
        self.assertEqual(output, ["Usage: /index status|refresh|stop|start"])
        self.assertEqual(agent.index_requests, [])

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

    def test_quit_command_rejects_extra_args(self) -> None:
        for command in ('/quit "   "', "/quit later", "/exit now"):
            with self.subTest(command=command):
                agent = DummyAgent()
                output: list[str] = []

                handled = handle_meta_command(command, agent, output.append)

                self.assertTrue(handled)
                self.assertEqual(output, ["Usage: /quit"])

    def test_reset_command_rejects_extra_args(self) -> None:
        for command in ('/reset "   "', "/reset later"):
            with self.subTest(command=command):
                agent = DummyAgent()
                output: list[str] = []

                handled = handle_meta_command(command, agent, output.append)

                self.assertTrue(handled)
                self.assertEqual(output, ["Usage: /reset"])
                self.assertEqual(agent.reset_requests, 0)

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

    def test_sessions_command_rejects_zero_limit(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        handled = handle_meta_command("/sessions 0", agent, output.append)
        self.assertTrue(handled)
        self.assertEqual(output, ["Usage: /sessions [limit]"])

    def test_sessions_command_rejects_negative_limit(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        handled = handle_meta_command("/sessions -5", agent, output.append)
        self.assertTrue(handled)
        self.assertEqual(output, ["Usage: /sessions [limit]"])

    def test_load_command_updates_session(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / "saved.json")
            handled = handle_meta_command(f"/load {target}", agent, output.append)
        self.assertTrue(handled)
        self.assertIsNotNone(agent.loaded_path)
        self.assertTrue(str(agent.loaded_path).endswith("saved.json"))

    def test_save_command_rejects_extra_path_args(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        handled = handle_meta_command("/save first.json second.json", agent, output.append)
        self.assertTrue(handled)
        self.assertIsNone(agent.saved_path)
        self.assertEqual(output, ["Usage: /save [path]"])

    def test_save_command_rejects_explicit_empty_path(self) -> None:
        agent = DummyAgent()
        output: list[str] = []

        handled = handle_meta_command('/save ""', agent, output.append)

        self.assertTrue(handled)
        self.assertIsNone(agent.saved_path)
        self.assertEqual(output, ["Usage: /save [path]"])

    def test_save_command_rejects_whitespace_only_quoted_path(self) -> None:
        agent = DummyAgent()
        output: list[str] = []

        handled = handle_meta_command('/save "   "', agent, output.append)

        self.assertTrue(handled)
        self.assertIsNone(agent.saved_path)
        self.assertEqual(output, ["Usage: /save [path]"])

    def test_load_command_rejects_extra_path_args(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        handled = handle_meta_command("/load saved.json trailing", agent, output.append)
        self.assertTrue(handled)
        self.assertIsNone(agent.loaded_path)
        self.assertEqual(output, ["Usage: /load <path>"])

    def test_load_command_rejects_explicit_empty_path(self) -> None:
        agent = DummyAgent()
        output: list[str] = []

        handled = handle_meta_command('/load ""', agent, output.append)

        self.assertTrue(handled)
        self.assertIsNone(agent.loaded_path)
        self.assertEqual(output, ["Usage: /load <path>"])

    def test_load_command_rejects_whitespace_only_quoted_path(self) -> None:
        agent = DummyAgent()
        output: list[str] = []

        handled = handle_meta_command('/load "   "', agent, output.append)

        self.assertTrue(handled)
        self.assertIsNone(agent.loaded_path)
        self.assertEqual(output, ["Usage: /load <path>"])

    def test_load_command_preserves_windows_backslashes(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        with patch("ollama_code.cli.os.name", "nt"):
            handled = handle_meta_command(r"/load C:\Users\yorke\saved.json", agent, output.append)
        self.assertTrue(handled)
        self.assertIsNotNone(agent.loaded_path)
        self.assertIn(r"C:\Users\yorke\saved.json", str(agent.loaded_path))

    @unittest.skipUnless(os.name != "nt", "POSIX only")
    def test_save_command_keeps_posix_escaped_spaces(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        handled = handle_meta_command(r"/save dir\ with\ spaces/trace.json", agent, output.append)
        self.assertTrue(handled)
        self.assertIsNotNone(agent.saved_path)
        self.assertTrue(str(agent.saved_path).endswith(str(Path("dir with spaces") / "trace.json")))

    @unittest.skipUnless(os.name != "nt", "POSIX only")
    def test_load_command_keeps_posix_escaped_spaces(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        handled = handle_meta_command(r"/load dir\ with\ spaces/saved.json", agent, output.append)
        self.assertTrue(handled)
        self.assertIsNotNone(agent.loaded_path)
        self.assertTrue(str(agent.loaded_path).endswith(str(Path("dir with spaces") / "saved.json")))

    def test_diff_command_prints_git_diff(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        handled = handle_meta_command("/diff --cached tracked.txt", agent, output.append)
        self.assertTrue(handled)
        self.assertIn("+after", output[0])
        self.assertEqual(agent.diff_request, {"cached": True, "path": "tracked.txt"})

    @unittest.skipUnless(os.name != "nt", "POSIX only")
    def test_diff_command_preserves_windows_backslashes_on_posix(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        handled = handle_meta_command(r"/diff C:\Users\yorke\ws\ollama-interactive\tracked.txt", agent, output.append)
        self.assertTrue(handled)
        self.assertEqual(agent.diff_request, {"cached": False, "path": r"C:\Users\yorke\ws\ollama-interactive\tracked.txt"})

    @unittest.skipUnless(os.name != "nt", "POSIX only")
    def test_diff_command_keeps_posix_escaped_spaces(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        handled = handle_meta_command(r"/diff dir\ with\ spaces/file.txt", agent, output.append)
        self.assertTrue(handled)
        self.assertEqual(agent.diff_request, {"cached": False, "path": "dir with spaces/file.txt"})

    def test_diff_command_rejects_explicit_empty_path(self) -> None:
        agent = DummyAgent()
        output: list[str] = []

        handled = handle_meta_command('/diff ""', agent, output.append)

        self.assertTrue(handled)
        self.assertEqual(output, ["Usage: /diff [--cached] [path]"])
        self.assertIsNone(agent.diff_request)

    def test_diff_command_rejects_whitespace_only_quoted_path_with_cached_flag(self) -> None:
        agent = DummyAgent()
        output: list[str] = []

        handled = handle_meta_command('/diff --cached "   "', agent, output.append)

        self.assertTrue(handled)
        self.assertEqual(output, ["Usage: /diff [--cached] [path]"])
        self.assertIsNone(agent.diff_request)

    def test_test_command_preserves_windows_executable_path(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        command = r'"C:\Program Files\Python312\python.exe" -m unittest -q'
        with patch("ollama_code.cli.os.name", "nt"):
            handled = handle_meta_command(f"/test {command}", agent, output.append)
        self.assertTrue(handled)
        self.assertIn(command, output[0])

    def test_test_command_strips_wrapping_quotes(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        handled = handle_meta_command('/test "pytest -q"', agent, output.append)
        self.assertTrue(handled)
        self.assertIn("pytest -q", output[0])
        self.assertNotIn('"pytest -q"', output[0])

    def test_test_command_rejects_explicit_empty_command(self) -> None:
        agent = DummyAgent()
        output: list[str] = []

        handled = handle_meta_command('/test ""', agent, output.append)

        self.assertTrue(handled)
        self.assertEqual(output, ["Usage: /test [command]"])
        self.assertEqual(agent.test_requests, [])

    def test_commit_command_prints_commit_output(self) -> None:
        agent = DummyAgent()
        output: list[str] = []
        handled = handle_meta_command('/commit "Save work"', agent, output.append)
        self.assertTrue(handled)
        self.assertIn("Save work", output[0])

    def test_commit_command_rejects_whitespace_only_quoted_message(self) -> None:
        agent = DummyAgent()
        output: list[str] = []

        handled = handle_meta_command('/commit "   "', agent, output.append)

        self.assertTrue(handled)
        self.assertEqual(output, ["Usage: /commit <message>"])
        self.assertEqual(agent.commit_requests, [])

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

    def test_build_agent_continue_skips_newer_invalid_session(self) -> None:
        root = self._workspace_scratch()
        session_dir = root / ".ollama-code" / "sessions"
        session_dir.mkdir(parents=True)
        older = session_dir / "older.json"
        newer = session_dir / "newer.json"
        older.write_text(
            '{"model":"gemma4:latest","approval_mode":"auto","workspace_root":"'
            + root.as_posix()
            + '","messages":[{"role":"system","content":"sys"},{"role":"user","content":"remember me"}],"events":[{"type":"user","content":"remember me"}]}',
            encoding="utf-8",
        )
        newer.write_text("{not json", encoding="utf-8")
        older_ts = newer.stat().st_mtime - 10
        os.utime(older, (older_ts, older_ts))
        parser = build_parser()
        args = parser.parse_args(["--cwd", str(root), "--continue", "--quiet"])

        agent = build_agent(args)

        self.assertEqual(agent.model, "gemma4:latest")
        self.assertEqual(agent.approval_mode(), "auto")
        self.assertEqual(agent.messages[1]["content"], "remember me")

    def test_build_agent_continue_skips_newer_unreadable_session(self) -> None:
        root = self._workspace_scratch()
        session_dir = root / ".ollama-code" / "sessions"
        session_dir.mkdir(parents=True)
        older = session_dir / "older.json"
        newer = session_dir / "newer.json"
        older.write_text(
            '{"model":"gemma4:latest","approval_mode":"auto","workspace_root":"'
            + root.as_posix()
            + '","messages":[{"role":"system","content":"sys"},{"role":"user","content":"remember me"}],"events":[{"type":"user","content":"remember me"}]}',
            encoding="utf-8",
        )
        newer.write_text(
            '{"model":"ignored","approval_mode":"ask","workspace_root":"'
            + root.as_posix()
            + '","messages":[{"role":"system","content":"sys"},{"role":"user","content":"blocked"}],"events":[{"type":"user","content":"blocked"}]}',
            encoding="utf-8",
        )
        older_ts = newer.stat().st_mtime - 10
        os.utime(older, (older_ts, older_ts))
        parser = build_parser()
        args = parser.parse_args(["--cwd", str(root), "--continue", "--quiet"])
        original_read_text = Path.read_text

        def denied_read_text(target: Path, *args: object, **kwargs: object) -> str:
            if target.resolve(strict=False) == newer.resolve(strict=False):
                raise PermissionError("denied")
            return original_read_text(target, *args, **kwargs)

        with patch.object(Path, "read_text", autospec=True, side_effect=denied_read_text):
            agent = build_agent(args)

        self.assertEqual(agent.model, "gemma4:latest")
        self.assertEqual(agent.approval_mode(), "auto")
        self.assertEqual(agent.messages[1]["content"], "remember me")

    def test_build_agent_continue_skips_newer_session_missing_message_history(self) -> None:
        root = self._workspace_scratch()
        session_dir = root / ".ollama-code" / "sessions"
        session_dir.mkdir(parents=True)
        older = session_dir / "older.json"
        newer = session_dir / "newer.json"
        older.write_text(
            '{"model":"gemma4:latest","approval_mode":"auto","workspace_root":"'
            + root.as_posix()
            + '","messages":[{"role":"system","content":"sys"},{"role":"user","content":"remember me"}],"events":[{"type":"user","content":"remember me"}]}',
            encoding="utf-8",
        )
        newer.write_text(
            '{"model":"ignored","approval_mode":"auto","workspace_root":"'
            + root.as_posix()
            + '","messages":[],"events":[]}',
            encoding="utf-8",
        )
        older_ts = newer.stat().st_mtime - 10
        os.utime(older, (older_ts, older_ts))
        parser = build_parser()
        args = parser.parse_args(["--cwd", str(root), "--continue", "--quiet"])

        agent = build_agent(args)

        self.assertEqual(agent.model, "gemma4:latest")
        self.assertEqual(agent.approval_mode(), "auto")
        self.assertEqual(agent.messages[1]["content"], "remember me")

    def test_build_agent_continue_skips_newer_system_only_session(self) -> None:
        root = self._workspace_scratch()
        session_dir = root / ".ollama-code" / "sessions"
        session_dir.mkdir(parents=True)
        older = session_dir / "older.json"
        newer = session_dir / "newer.json"
        older.write_text(
            '{"model":"gemma4:latest","approval_mode":"auto","workspace_root":"'
            + root.as_posix()
            + '","messages":[{"role":"system","content":"sys"},{"role":"user","content":"remember me"}],"events":[{"type":"user","content":"remember me"}]}',
            encoding="utf-8",
        )
        newer.write_text(
            '{"model":"blank-model","approval_mode":"read-only","workspace_root":"'
            + root.as_posix()
            + '","messages":[{"role":"system","content":"sys"}],"events":[]}',
            encoding="utf-8",
        )
        older_ts = newer.stat().st_mtime - 10
        os.utime(older, (older_ts, older_ts))
        parser = build_parser()
        args = parser.parse_args(["--cwd", str(root), "--continue", "--quiet"])

        agent = build_agent(args)

        self.assertEqual(agent.model, "gemma4:latest")
        self.assertEqual(agent.approval_mode(), "auto")
        self.assertEqual(agent.messages[1]["content"], "remember me")

    def test_build_agent_continue_uses_loaded_latest_session_payload_without_second_reload(self) -> None:
        root = self._workspace_scratch()
        session_dir = root / ".ollama-code" / "sessions"
        session_dir.mkdir(parents=True)
        older = session_dir / "older.json"
        newer = session_dir / "newer.json"
        older.write_text(
            '{"model":"older-model","approval_mode":"auto","workspace_root":"'
            + root.as_posix()
            + '","messages":[{"role":"system","content":"sys"},{"role":"user","content":"older"}],"events":[{"type":"user","content":"older"}]}',
            encoding="utf-8",
        )
        newer.write_text(
            '{"model":"newer-model","approval_mode":"ask","workspace_root":"'
            + root.as_posix()
            + '","messages":[{"role":"system","content":"sys"},{"role":"user","content":"newer"}],"events":[{"type":"user","content":"newer"}]}',
            encoding="utf-8",
        )
        older_ts = newer.stat().st_mtime - 10
        os.utime(older, (older_ts, older_ts))
        parser = build_parser()
        args = parser.parse_args(["--cwd", str(root), "--continue", "--quiet"])
        original_read_text = Path.read_text
        newer_reads = 0

        def flaky_read_text(target: Path, *args: object, **kwargs: object) -> str:
            nonlocal newer_reads
            if target.resolve(strict=False) == newer.resolve(strict=False):
                newer_reads += 1
                if newer_reads >= 2:
                    raise FileNotFoundError("vanished after discovery")
            return original_read_text(target, *args, **kwargs)

        with patch.object(Path, "read_text", autospec=True, side_effect=flaky_read_text):
            agent = build_agent(args)

        self.assertEqual(agent.model, "newer-model")
        self.assertEqual(agent.approval_mode(), "ask")
        self.assertEqual(agent.messages[1]["content"], "newer")
        self.assertEqual(newer_reads, 1)

    def test_build_agent_continue_normalizes_oversized_diagnostic_payloads(self) -> None:
        root = self._workspace_scratch()
        session_dir = root / ".ollama-code" / "sessions"
        session_dir.mkdir(parents=True)
        session = session_dir / "saved.json"
        session.write_text(
            '{"model":"gemma4:latest","approval_mode":"auto","workspace_root":"'
            + root.as_posix()
            + '","messages":[{"role":"system","content":"sys"},{"role":"user","content":"remember me"}],"events":[{"type":"tool_result","payload":"'
            + ("x" * 12000)
            + '"}],"llm_telemetry_events":[{"type":"llm_call","preview":"'
            + ("y" * 9000)
            + '"}]}',
            encoding="utf-8",
        )
        parser = build_parser()
        args = parser.parse_args(["--cwd", str(root), "--continue", "--quiet"])

        agent = build_agent(args)

        self.assertIn("truncated", agent.events[0]["payload"])
        self.assertLess(len(agent.events[0]["payload"]), 5000)
        self.assertIn("truncated", agent.llm_telemetry_events[0]["preview"])
        self.assertLess(len(agent.llm_telemetry_events[0]["preview"]), 5000)

    def test_build_agent_continue_truncates_large_nested_diagnostic_collections(self) -> None:
        root = self._workspace_scratch()
        session_dir = root / ".ollama-code" / "sessions"
        session_dir.mkdir(parents=True)
        session = session_dir / "saved.json"
        rows = ",".join('{"index":%d}' % index for index in range(TRANSCRIPT_DIAGNOSTIC_COLLECTION_LIMIT + 6))
        mapping = ",".join('"key_%d":%d' % (index, index) for index in range(TRANSCRIPT_DIAGNOSTIC_COLLECTION_LIMIT + 4))
        session.write_text(
            '{"model":"gemma4:latest","approval_mode":"auto","workspace_root":"'
            + root.as_posix()
            + '","messages":[{"role":"system","content":"sys"},{"role":"user","content":"remember me"}],"events":[{"type":"tool_result","result":{"rows":['
            + rows
            + '],"mapping":{'
            + mapping
            + '}}}]}',
            encoding="utf-8",
        )
        parser = build_parser()
        args = parser.parse_args(["--cwd", str(root), "--continue", "--quiet"])

        agent = build_agent(args)
        result = agent.events[0]["result"]

        self.assertEqual(len(result["rows"]), TRANSCRIPT_DIAGNOSTIC_COLLECTION_LIMIT + 1)
        self.assertEqual(result["rows"][-1], "[truncated 6 items for transcript]")
        self.assertEqual(result["mapping"][TRANSCRIPT_DIAGNOSTIC_DICT_MARKER_KEY], "[truncated 4 entries for transcript]")

    def test_build_agent_resume_missing_session_raises_value_error(self) -> None:
        missing = f"missing-{uuid4().hex}.json"
        parser = build_parser()
        args = parser.parse_args(["--cwd", str(Path.cwd()), "--resume", missing, "--quiet"])

        with self.assertRaisesRegex(ValueError, "Transcript file not found"):
            build_agent(args)

    def test_build_agent_resume_invalid_utf8_session_raises_value_error(self) -> None:
        root = self._workspace_scratch()
        session = root / ".ollama-code" / "sessions" / "invalid-utf8.json"
        session.parent.mkdir(parents=True, exist_ok=True)
        session.write_bytes(b"\xff\xfe\x80")
        parser = build_parser()
        args = parser.parse_args(["--cwd", str(root), "--resume", str(session), "--quiet"])

        with self.assertRaisesRegex(ValueError, "Invalid transcript encoding"):
            build_agent(args)

    def test_build_agent_resume_unsupported_message_role_raises_value_error(self) -> None:
        root = self._workspace_scratch()
        session = root / ".ollama-code" / "sessions" / "bad-role.json"
        session.parent.mkdir(parents=True, exist_ok=True)
        session.write_text(
            '{"model":"fake-model","approval_mode":"auto","workspace_root":"'
            + root.as_posix()
            + '","messages":[{"role":"system","content":"sys"},{"role":"tool","content":"bad role"}],"events":[]}',
            encoding="utf-8",
        )
        parser = build_parser()
        args = parser.parse_args(["--cwd", str(root), "--resume", str(session), "--quiet"])

        with self.assertRaisesRegex(ValueError, "Saved session contains a malformed message"):
            build_agent(args)

    def test_build_agent_resume_accepts_utf8_bom_session(self) -> None:
        root = self._workspace_scratch()
        session = root / ".ollama-code" / "sessions" / "bom.json"
        session.parent.mkdir(parents=True, exist_ok=True)
        session.write_bytes(
            b"\xef\xbb\xbf"
            + b'{"model":"bom-model","approval_mode":"auto","workspace_root":"'
            + root.as_posix().encode("utf-8")
            + b'","messages":[{"role":"system","content":"sys"},{"role":"user","content":"remember me"}],"events":[{"type":"user","content":"remember me"}]}',
        )
        parser = build_parser()
        args = parser.parse_args(["--cwd", str(root), "--resume", str(session), "--quiet"])

        agent = build_agent(args)

        self.assertEqual(agent.model, "bom-model")
        self.assertEqual(agent.messages[1]["content"], "remember me")

    @unittest.skipUnless(os.name != "nt", "POSIX only")
    def test_build_agent_resume_accepts_backslash_relative_path_on_posix(self) -> None:
        root = self._workspace_scratch()
        session = root / ".ollama-code" / "sessions" / "saved.json"
        session.parent.mkdir(parents=True, exist_ok=True)
        session.write_text(
            '{"model":"saved-model","approval_mode":"auto","workspace_root":"'
            + root.as_posix()
            + '","messages":[{"role":"system","content":"sys"},{"role":"user","content":"remember me"}],"events":[{"type":"user","content":"remember me"}]}',
            encoding="utf-8",
        )
        parser = build_parser()
        args = parser.parse_args(["--cwd", str(root), "--resume", r".ollama-code\sessions\saved.json", "--quiet"])

        agent = build_agent(args)

        self.assertEqual(agent.model, "saved-model")
        self.assertEqual(agent.messages[1]["content"], "remember me")

    def test_build_agent_resume_unreadable_session_raises_value_error(self) -> None:
        root = self._workspace_scratch()
        session = root / ".ollama-code" / "sessions" / "denied.json"
        session.parent.mkdir(parents=True, exist_ok=True)
        session.write_text(
            '{"model":"fake-model","approval_mode":"auto","workspace_root":"'
            + root.as_posix()
            + '","messages":[{"role":"system","content":"sys"},{"role":"user","content":"remember me"}],"events":[{"type":"user","content":"remember me"}]}',
            encoding="utf-8",
        )
        parser = build_parser()
        args = parser.parse_args(["--cwd", str(root), "--resume", str(session), "--quiet"])
        original_read_text = Path.read_text

        def denied_read_text(target: Path, *args: object, **kwargs: object) -> str:
            if target.resolve(strict=False) == session.resolve(strict=False):
                raise PermissionError("denied")
            return original_read_text(target, *args, **kwargs)

        with patch.object(Path, "read_text", autospec=True, side_effect=denied_read_text):
            with self.assertRaisesRegex(ValueError, "Unable to read transcript file"):
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
