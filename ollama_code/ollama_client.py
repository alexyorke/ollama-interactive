from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ollama_code.config import DEFAULT_OLLAMA_HOST, ENV_OLLAMA_HOST
from ollama_code.interrupts import OperationInterrupted


class OllamaError(RuntimeError):
    pass


@dataclass
class ChatResponse:
    content: str
    model: str
    raw: dict[str, Any]
    thinking: str = ""


class OllamaClient:
    def __init__(self, host: str | None = None, timeout: int = 300) -> None:
        normalized_host = host or os.environ.get(ENV_OLLAMA_HOST) or DEFAULT_OLLAMA_HOST
        if not normalized_host.startswith(("http://", "https://")):
            normalized_host = f"http://{normalized_host}"
        self.host = normalized_host.rstrip("/")
        self.timeout = timeout
        self._interrupt_event: threading.Event | None = None

    def set_interrupt_event(self, event: threading.Event | None) -> None:
        self._interrupt_event = event

    def _request_json(self, request: Request) -> dict[str, Any]:
        if self._interrupt_event is not None and self._interrupt_event.is_set():
            raise OperationInterrupted("Interrupted while waiting for Ollama.")

        result: dict[str, Any] = {}
        error: dict[str, BaseException] = {}
        done = threading.Event()

        def worker() -> None:
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    result["value"] = json.loads(response.read().decode("utf-8"))
            except BaseException as exc:
                error["value"] = exc
            finally:
                done.set()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while not done.wait(0.05):
            if self._interrupt_event is not None and self._interrupt_event.is_set():
                raise OperationInterrupted("Interrupted while waiting for Ollama.")
        if "value" in error:
            raise error["value"]
        return result["value"]

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: str | dict[str, Any] | None = "json",
        on_thinking: Callable[[str], None] | None = None,
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": on_thinking is not None,
            "think": True,
        }
        if response_format is not None:
            payload["format"] = response_format
        request = Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            if on_thinking is None:
                raw = self._request_json(request)
                message = raw.get("message") or {}
                return ChatResponse(
                    content=str(message.get("content", "")),
                    thinking=str(message.get("thinking", "")),
                    model=str(raw.get("model") or model),
                    raw=raw,
                )
            raw = self._request_chat_stream(request, model=model, on_thinking=on_thinking)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama HTTP error {exc.code}: {body}") from exc
        except URLError as exc:
            raise OllamaError(f"Could not reach Ollama at {self.host}: {exc}") from exc
        except TimeoutError as exc:
            raise OllamaError(f"Ollama timed out after {self.timeout} seconds.") from exc
        except json.JSONDecodeError as exc:
            raise OllamaError("Ollama returned invalid JSON from /api/chat.") from exc
        message = raw.get("message") or {}
        return ChatResponse(
            content=str(message.get("content", "")),
            thinking=str(message.get("thinking", "")),
            model=str(raw.get("model") or model),
            raw=raw,
        )

    def _request_chat_stream(
        self,
        request: Request,
        *,
        model: str,
        on_thinking: Callable[[str], None],
    ) -> dict[str, Any]:
        if self._interrupt_event is not None and self._interrupt_event.is_set():
            raise OperationInterrupted("Interrupted while waiting for Ollama.")

        result: dict[str, Any] = {}
        error: dict[str, BaseException] = {}
        done = threading.Event()

        def merge_stream_text(existing: str, chunk: str) -> str:
            if not chunk:
                return existing
            if not existing:
                return chunk
            if chunk.startswith(existing):
                return chunk
            if existing.endswith(chunk):
                return existing
            overlap = min(len(existing), len(chunk))
            while overlap > 0:
                if existing[-overlap:] == chunk[:overlap]:
                    return existing + chunk[overlap:]
                overlap -= 1
            return existing + chunk

        def worker() -> None:
            try:
                thinking_text = ""
                content_text = ""
                final_chunk: dict[str, Any] | None = None
                with urlopen(request, timeout=self.timeout) as response:
                    for raw_line in response:
                        if self._interrupt_event is not None and self._interrupt_event.is_set():
                            raise OperationInterrupted("Interrupted while waiting for Ollama.")
                        line = raw_line.decode("utf-8").strip()
                        if not line:
                            continue
                        chunk = json.loads(line)
                        if not isinstance(chunk, dict):
                            continue
                        final_chunk = chunk
                        message = chunk.get("message") if isinstance(chunk.get("message"), dict) else {}
                        thinking_delta = str(message.get("thinking", ""))
                        if thinking_delta:
                            thinking_text = merge_stream_text(thinking_text, thinking_delta)
                            on_thinking(thinking_text)
                        content_delta = str(message.get("content", ""))
                        if content_delta:
                            content_text = merge_stream_text(content_text, content_delta)
                merged = final_chunk or {}
                merged_message = dict(merged.get("message") or {})
                merged_message["content"] = content_text
                merged_message["thinking"] = thinking_text
                merged["message"] = merged_message
                merged["model"] = str(merged.get("model") or model)
                result["value"] = merged
            except BaseException as exc:
                error["value"] = exc
            finally:
                done.set()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while not done.wait(0.05):
            if self._interrupt_event is not None and self._interrupt_event.is_set():
                raise OperationInterrupted("Interrupted while waiting for Ollama.")
        if "value" in error:
            raise error["value"]
        return result["value"]

    def list_models(self) -> list[str]:
        request = Request(f"{self.host}/api/tags")
        try:
            raw = self._request_json(request)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama HTTP error {exc.code}: {body}") from exc
        except URLError as exc:
            raise OllamaError(f"Could not reach Ollama at {self.host}: {exc}") from exc
        except TimeoutError as exc:
            raise OllamaError(f"Ollama timed out after {self.timeout} seconds.") from exc
        except json.JSONDecodeError as exc:
            raise OllamaError("Ollama returned invalid JSON from /api/tags.") from exc
        models = raw.get("models") if isinstance(raw, dict) else None
        if not isinstance(models, list):
            return []
        names = [str(item.get("name")) for item in models if isinstance(item, dict) and item.get("name")]
        return sorted(names)
