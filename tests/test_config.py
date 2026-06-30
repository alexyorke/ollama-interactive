from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from ollama_code.cli import build_agent, build_parser
from ollama_code.config import DEFAULT_MODEL, load_config


class ConfigTests(unittest.TestCase):
    def _workspace_scratch(self) -> Path:
        root = (Path.cwd() / "verify_scratch" / f"test-config-{uuid4().hex}").resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return root

    def test_build_agent_reads_workspace_config_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".ollama-code").mkdir(parents=True)
            (root / ".ollama-code" / "config.json").write_text(
                json.dumps({"host": "http://127.0.0.1:11435", "model": "config-model", "verifier_model": "config-verifier", "reconcile": "on"}),
                encoding="utf-8",
            )
            parser = build_parser()
            args = parser.parse_args(["--cwd", str(root), "--quiet"])
            with patch.dict(os.environ, {"OLLAMA_HOST": "", "OLLAMA_CODE_MODEL": "", "OLLAMA_CODE_VERIFIER_MODEL": "", "OLLAMA_CODE_RECONCILE": ""}, clear=False):
                agent = build_agent(args)

        self.assertEqual(agent.client.host, "http://127.0.0.1:11435")
        self.assertEqual(agent.model, "config-model")
        self.assertIn(".ollama-code/config.json", agent.model_source.replace("\\", "/"))
        self.assertEqual(agent.verifier_model_name(), "config-verifier")
        self.assertEqual(agent.reconcile_mode(), "on")

    def test_build_agent_reads_explicit_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "settings.json"
            config.write_text(
                json.dumps({"ollama": {"host": "http://127.0.0.1:23456", "model": "nested-model"}}),
                encoding="utf-8",
            )
            parser = build_parser()
            args = parser.parse_args(["--cwd", str(root), "--config", "settings.json", "--quiet"])
            with patch.dict(os.environ, {"OLLAMA_HOST": "", "OLLAMA_CODE_MODEL": "", "OLLAMA_CODE_VERIFIER_MODEL": "", "OLLAMA_CODE_RECONCILE": ""}, clear=False):
                agent = build_agent(args)

        self.assertEqual(agent.client.host, "http://127.0.0.1:23456")
        self.assertEqual(agent.model, "nested-model")

    @unittest.skipUnless(os.name != "nt", "POSIX only")
    def test_build_agent_reads_backslash_relative_config_path_on_posix(self) -> None:
        root = self._workspace_scratch()
        config = root / ".ollama-code" / "config.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(
            json.dumps({"ollama": {"host": "http://127.0.0.1:23456", "model": "backslash-model"}}),
            encoding="utf-8",
        )
        parser = build_parser()
        args = parser.parse_args(["--cwd", str(root), "--config", r".ollama-code\config.json", "--quiet"])
        with patch.dict(os.environ, {"OLLAMA_HOST": "", "OLLAMA_CODE_MODEL": "", "OLLAMA_CODE_VERIFIER_MODEL": "", "OLLAMA_CODE_RECONCILE": ""}, clear=False):
            agent = build_agent(args)

        self.assertEqual(agent.client.host, "http://127.0.0.1:23456")
        self.assertEqual(agent.model, "backslash-model")

    @unittest.skipUnless(os.name != "nt", "POSIX only")
    def test_load_config_reads_windows_style_absolute_path_on_posix(self) -> None:
        root = self._workspace_scratch()
        config = root / ".ollama-code" / "config.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(json.dumps({"model": "alias-model"}), encoding="utf-8")
        config_path = config.as_posix()
        if not config_path.startswith("/mnt/") or len(config_path) < 8 or config_path[6] != "/":
            self.skipTest("requires a /mnt/<drive> workspace path")
        windows_style = f"{config_path[5].upper()}:{config_path[6:]}"

        loaded = load_config(root, windows_style)

        self.assertEqual(loaded.model, "alias-model")

    @unittest.skipUnless(os.name == "nt", "Windows only")
    def test_load_config_reads_wsl_alias_path_on_windows(self) -> None:
        root = self._workspace_scratch()
        config = root / ".ollama-code" / "config.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(json.dumps({"model": "alias-model"}), encoding="utf-8")
        alias = f"/mnt/{root.drive[:1].lower()}{config.as_posix()[2:]}"

        loaded = load_config(root, alias)

        self.assertEqual(loaded.model, "alias-model")

    def test_load_config_reports_invalid_utf8(self) -> None:
        root = self._workspace_scratch()
        config = root / ".ollama-code" / "config.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_bytes(b"\xff\xfe\x80")

        with self.assertRaisesRegex(ValueError, "Invalid config encoding"):
            load_config(root)

    def test_load_config_reports_unreadable_file(self) -> None:
        root = self._workspace_scratch()
        config = root / ".ollama-code" / "config.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(json.dumps({"model": "blocked-model"}), encoding="utf-8")
        original_read_text = Path.read_text

        def denied_read_text(target: Path, *args: object, **kwargs: object) -> str:
            if target.resolve(strict=False) == config.resolve(strict=False):
                raise PermissionError("denied")
            return original_read_text(target, *args, **kwargs)

        with patch.object(Path, "read_text", autospec=True, side_effect=denied_read_text):
            with self.assertRaisesRegex(ValueError, "Unable to read config file"):
                load_config(root)

    def test_load_config_accepts_utf8_bom(self) -> None:
        root = self._workspace_scratch()
        config = root / ".ollama-code" / "config.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_bytes(
            b"\xef\xbb\xbf" + json.dumps({"model": "bom-model", "host": "http://127.0.0.1:11435"}).encode("utf-8")
        )

        loaded = load_config(root)

        self.assertEqual(loaded.model, "bom-model")
        self.assertEqual(loaded.host, "http://127.0.0.1:11435")

    def test_continue_session_model_overrides_config_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".ollama-code"
            session_dir = config_dir / "sessions"
            session_dir.mkdir(parents=True)
            (config_dir / "config.json").write_text(
                json.dumps({"host": "http://127.0.0.1:11435", "model": "config-model"}),
                encoding="utf-8",
            )
            (session_dir / "saved.json").write_text(
                json.dumps(
                    {
                        "model": "session-model",
                        "verifier_model": "session-verifier",
                        "reconcile_mode": "on",
                        "approval_mode": "auto",
                        "workspace_root": root.as_posix(),
                        "messages": [
                            {"role": "system", "content": "sys"},
                            {"role": "user", "content": "remember me"},
                        ],
                        "events": [{"type": "user", "content": "remember me"}],
                    }
                ),
                encoding="utf-8",
            )
            parser = build_parser()
            args = parser.parse_args(["--cwd", str(root), "--continue", "--quiet"])
            with patch.dict(os.environ, {"OLLAMA_HOST": "", "OLLAMA_CODE_MODEL": "", "OLLAMA_CODE_VERIFIER_MODEL": "", "OLLAMA_CODE_RECONCILE": ""}, clear=False):
                agent = build_agent(args)

        self.assertEqual(agent.client.host, "http://127.0.0.1:11435")
        self.assertEqual(agent.model, "session-model")
        self.assertEqual(agent.verifier_model_name(), "session-verifier")
        self.assertEqual(agent.reconcile_mode(), "on")

    def test_env_overrides_workspace_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".ollama-code").mkdir(parents=True)
            (root / ".ollama-code" / "config.json").write_text(
                json.dumps({"host": "http://127.0.0.1:11435", "model": "config-model"}),
                encoding="utf-8",
            )
            parser = build_parser()
            args = parser.parse_args(["--cwd", str(root), "--quiet"])
            with patch.dict(os.environ, {"OLLAMA_HOST": "http://127.0.0.1:34567", "OLLAMA_CODE_MODEL": "env-model", "OLLAMA_CODE_VERIFIER_MODEL": "env-verifier", "OLLAMA_CODE_RECONCILE": "off"}, clear=False):
                agent = build_agent(args)

        self.assertEqual(agent.client.host, "http://127.0.0.1:34567")
        self.assertEqual(agent.model, "env-model")
        self.assertEqual(agent.model_source, "OLLAMA_CODE_MODEL")
        self.assertEqual(agent.verifier_model_name(), "env-verifier")
        self.assertEqual(agent.reconcile_mode(), "off")

    def test_build_agent_reads_bom_prefixed_workspace_config(self) -> None:
        root = self._workspace_scratch()
        config = root / ".ollama-code" / "config.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_bytes(
            b"\xef\xbb\xbf"
            + json.dumps({"host": "http://127.0.0.1:11435", "model": "bom-config-model"}).encode("utf-8")
        )
        parser = build_parser()
        args = parser.parse_args(["--cwd", str(root), "--quiet"])
        with patch.dict(os.environ, {"OLLAMA_HOST": "", "OLLAMA_CODE_MODEL": "", "OLLAMA_CODE_VERIFIER_MODEL": "", "OLLAMA_CODE_RECONCILE": ""}, clear=False):
            agent = build_agent(args)

        self.assertEqual(agent.client.host, "http://127.0.0.1:11435")
        self.assertEqual(agent.model, "bom-config-model")

    def test_missing_explicit_config_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parser = build_parser()
            args = parser.parse_args(["--cwd", str(root), "--config", "missing.json", "--quiet"])
            with self.assertRaisesRegex(ValueError, "Config file not found"):
                build_agent(args)

    def test_build_agent_invalid_utf8_workspace_config_raises_value_error(self) -> None:
        root = self._workspace_scratch()
        config = root / ".ollama-code" / "config.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_bytes(b"\xff\xfe\x80")
        parser = build_parser()
        args = parser.parse_args(["--cwd", str(root), "--quiet"])

        with self.assertRaisesRegex(ValueError, "Invalid config encoding"):
            build_agent(args)

    def test_build_agent_unreadable_workspace_config_raises_value_error(self) -> None:
        root = self._workspace_scratch()
        config = root / ".ollama-code" / "config.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(json.dumps({"model": "blocked-model"}), encoding="utf-8")
        parser = build_parser()
        args = parser.parse_args(["--cwd", str(root), "--quiet"])
        original_read_text = Path.read_text

        def denied_read_text(target: Path, *args: object, **kwargs: object) -> str:
            if target.resolve(strict=False) == config.resolve(strict=False):
                raise PermissionError("denied")
            return original_read_text(target, *args, **kwargs)

        with patch.object(Path, "read_text", autospec=True, side_effect=denied_read_text):
            with self.assertRaisesRegex(ValueError, "Unable to read config file"):
                build_agent(args)

    def test_integer_config_rejects_json_boolean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / ".ollama-code" / "config.json"
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps({"timeout": True}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, 'Config value "timeout" must be a positive integer'):
                load_config(root)

    def test_build_agent_reads_tooling_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".ollama-code").mkdir(parents=True)
            (root / ".ollama-code" / "config.json").write_text(
                json.dumps(
                    {
                        "tools": {"default_enabled": True, "enabled": ["run_test", "read_file"], "disabled": ["browser_smoke"]},
                        "mcp": {"servers": {"demo": {"command": "demo-mcp"}}},
                        "browser": {"enabled": False},
                        "security": {"enabled": False},
                        "indexer": {"enabled": False, "watch": False, "poll_interval_ms": 2500},
                    }
                ),
                encoding="utf-8",
            )
            parser = build_parser()
            args = parser.parse_args(["--cwd", str(root), "--quiet"])
            agent = build_agent(args)

        self.assertNotIn("browser_smoke", agent.tools.available_tool_names())
        self.assertEqual(agent.tools.enabled_tools, {"run_test", "read_file"})
        self.assertEqual(agent.tools.mcp_servers["demo"]["command"], "demo-mcp")
        self.assertFalse(agent.tools.browser_enabled)
        self.assertFalse(agent.tools.security_enabled)
        self.assertFalse(agent.index_status()["enabled"])
        self.assertFalse(agent.index_status()["watch"])
        self.assertEqual(agent.index_status()["poll_interval_ms"], 2500)


class _FakeOllamaHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, object]] = []
    tag_requests = 0
    available_models = [{"name": "config-model"}]

    def do_GET(self) -> None:
        if self.path != "/api/tags":
            self.send_error(404)
            return
        type(self).tag_requests += 1
        body = json.dumps({"models": type(self).available_models}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        type(self).requests.append(payload)
        messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
        system_prompt = ""
        if messages and isinstance(messages[0], dict):
            system_prompt = str(messages[0].get("content", ""))
        if system_prompt.startswith("You are a grounded final verifier"):
            response = {
                "model": payload.get("model", ""),
                "message": {"content": '{"verdict":"accept"}'},
            }
        else:
            response = {
                "model": payload.get("model", ""),
                "message": {"content": '{"type":"final","message":"config smoke ok"}'},
            }
        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def _start_test_server_thread(server: ThreadingHTTPServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    return thread


class ConfigCliSmokeTests(unittest.TestCase):
    def test_one_shot_cli_uses_workspace_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".ollama-code"
            config_dir.mkdir(parents=True)
            _FakeOllamaHandler.requests = []
            _FakeOllamaHandler.tag_requests = 0
            _FakeOllamaHandler.available_models = [{"name": "config-model"}]
            server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOllamaHandler)
            thread = _start_test_server_thread(server)
            try:
                (config_dir / "config.json").write_text(
                    json.dumps({"host": f"http://127.0.0.1:{server.server_port}", "model": "config-model"}),
                    encoding="utf-8",
                )
                result = subprocess.run(
                    [sys.executable, "-m", "ollama_code", "--cwd", str(root), "--quiet", "Reply with ok."],
                    cwd=Path(__file__).resolve().parents[1],
                    capture_output=True,
                    text=True,
                    check=False,
                    env={key: value for key, value in os.environ.items() if key not in {"OLLAMA_HOST", "OLLAMA_CODE_MODEL", "OLLAMA_CODE_VERIFIER_MODEL"}},
                    timeout=30,
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "config smoke ok")
        self.assertEqual(len(_FakeOllamaHandler.requests), 1)
        # model resolution now validates the configured model against local tags.
        self.assertEqual(_FakeOllamaHandler.tag_requests, 1)
        self.assertEqual(_FakeOllamaHandler.requests[0]["model"], "config-model")

    def test_one_shot_cli_falls_back_to_installed_runtime_default_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".ollama-code"
            config_dir.mkdir(parents=True)
            _FakeOllamaHandler.requests = []
            _FakeOllamaHandler.tag_requests = 0
            _FakeOllamaHandler.available_models = [{"name": "gemma3:4b"}]
            server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOllamaHandler)
            thread = _start_test_server_thread(server)
            try:
                (config_dir / "config.json").write_text(
                    json.dumps({"host": f"http://127.0.0.1:{server.server_port}"}),
                    encoding="utf-8",
                )
                result = subprocess.run(
                    [sys.executable, "-m", "ollama_code", "--cwd", str(root), "--quiet", "Reply with ok."],
                    cwd=Path(__file__).resolve().parents[1],
                    capture_output=True,
                    text=True,
                    check=False,
                    env={key: value for key, value in os.environ.items() if key not in {"OLLAMA_HOST", "OLLAMA_CODE_MODEL", "OLLAMA_CODE_VERIFIER_MODEL"}},
                    timeout=30,
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "config smoke ok")
        self.assertEqual(_FakeOllamaHandler.tag_requests, 1)
        self.assertEqual(len(_FakeOllamaHandler.requests), 1)
        self.assertEqual(_FakeOllamaHandler.requests[0]["model"], "gemma3:4b")

    def test_one_shot_cli_falls_back_when_configured_model_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".ollama-code"
            config_dir.mkdir(parents=True)
            _FakeOllamaHandler.requests = []
            _FakeOllamaHandler.tag_requests = 0
            _FakeOllamaHandler.available_models = [{"name": "gemma3:4b"}]
            server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOllamaHandler)
            thread = _start_test_server_thread(server)
            try:
                (config_dir / "config.json").write_text(
                    json.dumps({"host": f"http://127.0.0.1:{server.server_port}", "model": "stale-config-model"}),
                    encoding="utf-8",
                )
                result = subprocess.run(
                    [sys.executable, "-m", "ollama_code", "--cwd", str(root), "--quiet", "Reply with ok."],
                    cwd=Path(__file__).resolve().parents[1],
                    capture_output=True,
                    text=True,
                    check=False,
                    env={key: value for key, value in os.environ.items() if key not in {"OLLAMA_HOST", "OLLAMA_CODE_MODEL", "OLLAMA_CODE_VERIFIER_MODEL"}},
                    timeout=30,
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "config smoke ok")
        self.assertEqual(_FakeOllamaHandler.tag_requests, 1)
        self.assertEqual(len(_FakeOllamaHandler.requests), 1)
        self.assertEqual(_FakeOllamaHandler.requests[0]["model"], "gemma3:4b")

    def test_one_shot_cli_does_not_fallback_to_unpreferred_custom_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".ollama-code"
            config_dir.mkdir(parents=True)
            _FakeOllamaHandler.requests = []
            _FakeOllamaHandler.tag_requests = 0
            _FakeOllamaHandler.available_models = [{"name": "hf.co/batiai/Granite-4.1-8B-GGUF:IQ4_XS"}]
            server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOllamaHandler)
            thread = _start_test_server_thread(server)
            try:
                (config_dir / "config.json").write_text(
                    json.dumps({"host": f"http://127.0.0.1:{server.server_port}"}),
                    encoding="utf-8",
                )
                result = subprocess.run(
                    [sys.executable, "-m", "ollama_code", "--cwd", str(root), "--quiet", "Reply with ok."],
                    cwd=Path(__file__).resolve().parents[1],
                    capture_output=True,
                    text=True,
                    check=False,
                    env={key: value for key, value in os.environ.items() if key not in {"OLLAMA_HOST", "OLLAMA_CODE_MODEL", "OLLAMA_CODE_VERIFIER_MODEL"}},
                    timeout=30,
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(_FakeOllamaHandler.tag_requests, 1)
        self.assertEqual(len(_FakeOllamaHandler.requests), 1)
        self.assertEqual(_FakeOllamaHandler.requests[0]["model"], DEFAULT_MODEL)
