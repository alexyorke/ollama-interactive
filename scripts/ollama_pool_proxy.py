from __future__ import annotations

import argparse
import http.client
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import count
from typing import ClassVar
from urllib.parse import urlsplit


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


@dataclass
class Backend:
    base_url: str
    scheme: str
    host: str
    port: int
    path_prefix: str = ""
    inflight: int = 0


@dataclass
class BackendPool:
    backends: list[Backend]
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _sequence: count = field(default_factory=count)

    @classmethod
    def from_urls(cls, raw_urls: list[str]) -> "BackendPool":
        backends: list[Backend] = []
        for raw in raw_urls:
            parts = urlsplit(raw)
            if parts.scheme not in {"http", "https"}:
                raise ValueError(f"Unsupported backend scheme: {raw}")
            if not parts.hostname or parts.port is None:
                raise ValueError(f"Backend must include host and port: {raw}")
            path_prefix = (parts.path or "").rstrip("/")
            base_url = f"{parts.scheme}://{parts.hostname}:{parts.port}{path_prefix}"
            backends.append(
                Backend(
                    base_url=base_url,
                    scheme=parts.scheme,
                    host=parts.hostname,
                    port=parts.port,
                    path_prefix=path_prefix,
                )
            )
        if not backends:
            raise ValueError("At least one backend is required.")
        return cls(backends)

    def acquire(self) -> Backend:
        with self._lock:
            min_inflight = min(backend.inflight for backend in self.backends)
            candidates = [backend for backend in self.backends if backend.inflight == min_inflight]
            index = next(self._sequence) % len(candidates)
            backend = candidates[index]
            backend.inflight += 1
            return backend

    def release(self, backend: Backend) -> None:
        with self._lock:
            backend.inflight = max(0, backend.inflight - 1)


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    pool: ClassVar[BackendPool]

    def do_GET(self) -> None:  # noqa: N802
        self._forward()

    def do_POST(self) -> None:  # noqa: N802
        self._forward()

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _forward(self) -> None:
        backend = self.pool.acquire()
        try:
            self._forward_to_backend(backend)
        finally:
            self.pool.release(backend)

    def _forward_to_backend(self, backend: Backend) -> None:
        connection_cls = http.client.HTTPSConnection if backend.scheme == "https" else http.client.HTTPConnection
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(content_length) if content_length > 0 else None
        upstream_headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
        }
        connection = connection_cls(backend.host, backend.port, timeout=900)
        upstream_path = self.path
        if backend.path_prefix:
            upstream_path = f"{backend.path_prefix}{self.path}" if self.path.startswith("/") else f"{backend.path_prefix}/{self.path}"
        try:
            connection.request(self.command, upstream_path, body=body, headers=upstream_headers)
            response = connection.getresponse()
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() in HOP_BY_HOP_HEADERS:
                    continue
                self.send_header(key, value)
            self.end_headers()
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        finally:
            connection.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Least-connections proxy for multiple Ollama backends.")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=11437)
    parser.add_argument(
        "--backends",
        nargs="+",
        required=True,
        help="Backend base URLs, for example http://127.0.0.1:11435 http://127.0.0.1:11436",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ProxyHandler.pool = BackendPool.from_urls(args.backends)
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), ProxyHandler)
    print(
        f"listening on http://{args.listen_host}:{args.listen_port} -> "
        + ", ".join(backend.base_url for backend in ProxyHandler.pool.backends),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
