from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from ollama_code.cli import build_agent, build_parser


class ConfigTests(unittest.TestCase):
    def test_build_agent_reads_workspace_config_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".ollama-code").mkdir(parents=True)
            (root / ".ollama-code" / "config.json").write_text(
                json.dumps({"host": "http://127.0.0.1:11435", "model": "config-model", "verifier_model": "config-verifier"}),
                encoding="utf-8",
            )
            parser = build_parser()
            args = parser.parse_args(["--cwd", str(root), "--quiet"])
            with patch.dict(os.environ, {"OLLAMA_HOST": "", "OLLAMA_CODE_MODEL": "", "OLLAMA_CODE_VERIFIER_MODEL": ""}, clear=False):
                agent = build_agent(args)

        self.assertEqual(agent.client.host, "http://127.0.0.1:11435")
        self.assertEqual(agent.model, "config-model")
        self.assertEqual(agent.verifier_model_name(), "config-verifier")

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
            with patch.dict(os.environ, {"OLLAMA_HOST": "", "OLLAMA_CODE_MODEL": "", "OLLAMA_CODE_VERIFIER_MODEL": ""}, clear=False):
                agent = build_agent(args)

        self.assertEqual(agent.client.host, "http://127.0.0.1:23456")
        self.assertEqual(agent.model, "nested-model")

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
            with patch.dict(os.environ, {"OLLAMA_HOST": "", "OLLAMA_CODE_MODEL": "", "OLLAMA_CODE_VERIFIER_MODEL": ""}, clear=False):
                agent = build_agent(args)

        self.assertEqual(agent.client.host, "http://127.0.0.1:11435")
        self.assertEqual(agent.model, "session-model")
        self.assertEqual(agent.verifier_model_name(), "session-verifier")

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
            with patch.dict(os.environ, {"OLLAMA_HOST": "http://127.0.0.1:34567", "OLLAMA_CODE_MODEL": "env-model", "OLLAMA_CODE_VERIFIER_MODEL": "env-verifier"}, clear=False):
                agent = build_agent(args)

        self.assertEqual(agent.client.host, "http://127.0.0.1:34567")
        self.assertEqual(agent.model, "env-model")
        self.assertEqual(agent.verifier_model_name(), "env-verifier")

    def test_missing_explicit_config_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parser = build_parser()
            args = parser.parse_args(["--cwd", str(root), "--config", "missing.json", "--quiet"])
            with self.assertRaisesRegex(ValueError, "Config file not found"):
                build_agent(args)


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
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
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
        self.assertEqual(_FakeOllamaHandler.tag_requests, 0)
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
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
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
