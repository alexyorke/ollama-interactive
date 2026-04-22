from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaError(RuntimeError):
    pass


@dataclass
class ChatResponse:
    content: str
    model: str
    raw: dict[str, Any]


class OllamaClient:
    def __init__(self, host: str | None = None, timeout: int = 300) -> None:
        normalized_host = host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
        if not normalized_host.startswith(("http://", "https://")):
            normalized_host = f"http://{normalized_host}"
        self.host = normalized_host.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: str | dict[str, Any] | None = "json",
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
        }
        if response_format is not None:
            payload["format"] = response_format
        request = Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama HTTP error {exc.code}: {body}") from exc
        except URLError as exc:
            raise OllamaError(f"Could not reach Ollama at {self.host}: {exc}") from exc
        except TimeoutError as exc:
            raise OllamaError(f"Ollama timed out after {self.timeout} seconds.") from exc
        message = raw.get("message") or {}
        return ChatResponse(
            content=str(message.get("content", "")),
            model=str(raw.get("model") or model),
            raw=raw,
        )

    def list_models(self) -> list[str]:
        request = Request(f"{self.host}/api/tags")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama HTTP error {exc.code}: {body}") from exc
        except URLError as exc:
            raise OllamaError(f"Could not reach Ollama at {self.host}: {exc}") from exc
        models = raw.get("models") if isinstance(raw, dict) else None
        if not isinstance(models, list):
            return []
        names = [str(item.get("name")) for item in models if isinstance(item, dict) and item.get("name")]
        return sorted(names)
