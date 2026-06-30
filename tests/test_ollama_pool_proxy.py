from __future__ import annotations

import unittest
from io import BytesIO
from unittest.mock import Mock, patch

from scripts import ollama_pool_proxy as proxy


class OllamaPoolProxyTests(unittest.TestCase):
    def test_parse_args_accepts_argv(self) -> None:
        args = proxy.parse_args(
            [
                "--listen-host",
                "0.0.0.0",
                "--listen-port",
                "12000",
                "--backends",
                "http://127.0.0.1:11435",
                "http://127.0.0.1:11436",
            ]
        )

        self.assertEqual(args.listen_host, "0.0.0.0")
        self.assertEqual(args.listen_port, 12000)
        self.assertEqual(args.backends, ["http://127.0.0.1:11435", "http://127.0.0.1:11436"])

    def test_main_accepts_argv_and_closes_server_on_keyboard_interrupt(self) -> None:
        fake_server = Mock()
        fake_server.serve_forever.side_effect = KeyboardInterrupt

        with patch.object(proxy, "ThreadingHTTPServer", return_value=fake_server) as server_ctor:
            exit_code = proxy.main(["--backends", "http://127.0.0.1:11435"])

        self.assertEqual(exit_code, 0)
        server_ctor.assert_called_once()
        fake_server.serve_forever.assert_called_once_with()
        fake_server.server_close.assert_called_once_with()

    def test_backend_pool_preserves_backend_path_prefix(self) -> None:
        pool = proxy.BackendPool.from_urls(["http://127.0.0.1:11435/ollama"])

        backend = pool.backends[0]
        self.assertEqual(backend.base_url, "http://127.0.0.1:11435/ollama")
        self.assertEqual(backend.path_prefix, "/ollama")

    def test_forward_to_backend_prepends_backend_path_prefix(self) -> None:
        backend = proxy.Backend(
            base_url="http://127.0.0.1:11435/ollama",
            scheme="http",
            host="127.0.0.1",
            port=11435,
            path_prefix="/ollama",
        )
        fake_response = Mock()
        fake_response.status = 200
        fake_response.reason = "OK"
        fake_response.getheaders.return_value = [("Content-Type", "application/json")]
        fake_response.read.side_effect = [b"{}", b""]
        fake_connection = Mock()
        fake_connection.getresponse.return_value = fake_response

        handler = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        handler.command = "GET"
        handler.path = "/api/tags"
        handler.headers = {"Content-Length": "0"}
        handler.rfile = BytesIO(b"")
        handler.wfile = Mock()
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()

        with patch("scripts.ollama_pool_proxy.http.client.HTTPConnection", return_value=fake_connection):
            handler._forward_to_backend(backend)

        fake_connection.request.assert_called_once_with(
            "GET",
            "/ollama/api/tags",
            body=None,
            headers={"Content-Length": "0"},
        )
        handler.send_response.assert_called_once_with(200, "OK")
        handler.end_headers.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
