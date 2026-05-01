from __future__ import annotations

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ollama_code.interrupts import OperationInterrupted
from ollama_code.ollama_client import OllamaClient, OllamaError


class _MalformedResponseHandler(BaseHTTPRequestHandler):
    response_body = b"{}"
    response_status = 200
    response_delay = 0.0
    response_sequence: list[tuple[int, bytes]] = []
    last_request_body = b""
    last_request_bodies: list[bytes] = []

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        type(self).last_request_body = self.rfile.read(length)
        type(self).last_request_bodies.append(type(self).last_request_body)
        status = type(self).response_status
        body = type(self).response_body
        if type(self).response_sequence:
            status, body = type(self).response_sequence.pop(0)
        if type(self).response_delay:
            time.sleep(type(self).response_delay)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/api/tags":
            self.send_error(404)
            return
        if type(self).response_delay:
            time.sleep(type(self).response_delay)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(type(self).response_body)

    def log_message(self, format: str, *args: object) -> None:
        return


class OllamaClientTests(unittest.TestCase):
    def _with_server(self, body: bytes, *, delay: float = 0.0) -> tuple[OllamaClient, ThreadingHTTPServer, threading.Thread]:
        _MalformedResponseHandler.response_body = body
        _MalformedResponseHandler.response_status = 200
        _MalformedResponseHandler.response_delay = delay
        _MalformedResponseHandler.response_sequence = []
        _MalformedResponseHandler.last_request_body = b""
        _MalformedResponseHandler.last_request_bodies = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), _MalformedResponseHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = OllamaClient(host=f"http://127.0.0.1:{server.server_port}", timeout=5)
        return client, server, thread

    def test_chat_reports_invalid_json_response(self) -> None:
        client, server, thread = self._with_server(b"not json")
        try:
            with self.assertRaisesRegex(OllamaError, r"invalid JSON from /api/chat"):
                client.chat(model="fake-model", messages=[{"role": "user", "content": "hi"}])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_list_models_reports_invalid_json_response(self) -> None:
        client, server, thread = self._with_server(b"{bad")
        try:
            with self.assertRaisesRegex(OllamaError, r"invalid JSON from /api/tags"):
                client.list_models()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_chat_can_be_interrupted(self) -> None:
        client, server, thread = self._with_server(b'{"message":{"content":"ok"}}', delay=1.0)
        interrupted = threading.Event()
        client.set_interrupt_event(interrupted)
        trigger = threading.Timer(0.1, interrupted.set)
        trigger.start()
        try:
            with self.assertRaisesRegex(OperationInterrupted, "Interrupted while waiting for Ollama"):
                client.chat(model="fake-model", messages=[{"role": "user", "content": "hi"}])
        finally:
            trigger.cancel()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_chat_enables_thinking_by_default(self) -> None:
        client, server, thread = self._with_server(b'{"message":{"content":"ok"}}')
        try:
            response = client.chat(model="fake-model", messages=[{"role": "user", "content": "hi"}])
            self.assertEqual(response.content, "ok")
            payload = json.loads(_MalformedResponseHandler.last_request_body.decode("utf-8"))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.assertTrue(payload["think"])

    def test_chat_retries_without_thinking_when_model_rejects_it(self) -> None:
        client, server, thread = self._with_server(b'{"message":{"content":"ok"}}')
        _MalformedResponseHandler.response_sequence = [
            (400, b'{"error":"\\"fake-model\\" does not support thinking"}'),
            (200, b'{"message":{"content":"ok"}}'),
        ]
        try:
            response = client.chat(model="fake-model", messages=[{"role": "user", "content": "hi"}])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        first_payload = json.loads(_MalformedResponseHandler.last_request_bodies[0].decode("utf-8"))
        second_payload = json.loads(_MalformedResponseHandler.last_request_bodies[1].decode("utf-8"))
        self.assertEqual(response.content, "ok")
        self.assertTrue(first_payload["think"])
        self.assertFalse(second_payload["think"])

    def test_chat_allows_explicit_think_disable(self) -> None:
        client, server, thread = self._with_server(b'{"message":{"content":"ok"}}')
        try:
            response = client.chat(model="fake-model", messages=[{"role": "user", "content": "hi"}], think=False)
            payload = json.loads(_MalformedResponseHandler.last_request_body.decode("utf-8"))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.assertEqual(response.content, "ok")
        self.assertFalse(payload["think"])

    def test_chat_streams_thinking_updates(self) -> None:
        body = (
            b'{"message":{"content":"","thinking":"alpha"},"done":false}\n'
            b'{"message":{"content":"ok","thinking":" beta"},"done":true}\n'
        )
        client, server, thread = self._with_server(body)
        updates: list[str] = []
        try:
            response = client.chat(model="fake-model", messages=[{"role": "user", "content": "hi"}], on_thinking=updates.append)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.assertEqual(updates[-1], "alpha beta")
        self.assertEqual(response.thinking, "alpha beta")
        self.assertEqual(response.content, "ok")

    def test_chat_stream_deduplicates_cumulative_thinking_chunks(self) -> None:
        body = (
            b'{"message":{"content":"","thinking":"Let me inspect files"},"done":false}\n'
            b'{"message":{"content":"","thinking":"Let me inspect files and check config"},"done":false}\n'
            b'{"message":{"content":"done","thinking":"Let me inspect files and check config"},"done":true}\n'
        )
        client, server, thread = self._with_server(body)
        updates: list[str] = []
        try:
            response = client.chat(model="fake-model", messages=[{"role": "user", "content": "hi"}], on_thinking=updates.append)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.assertEqual(updates[-1], "Let me inspect files and check config")
        self.assertEqual(response.thinking, "Let me inspect files and check config")
        self.assertEqual(response.content, "done")

    def test_chat_stream_deduplicates_overlapping_content_chunks(self) -> None:
        body = (
            b'{"message":{"content":"Hel","thinking":""},"done":false}\n'
            b'{"message":{"content":"Hello","thinking":""},"done":false}\n'
            b'{"message":{"content":"Hello there","thinking":""},"done":true}\n'
        )
        client, server, thread = self._with_server(body)
        try:
            response = client.chat(model="fake-model", messages=[{"role": "user", "content": "hi"}], on_thinking=lambda _: None)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.assertEqual(response.content, "Hello there")
