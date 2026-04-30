from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(".ollama-code") / "config.json"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "batiai/qwen3.6-35b:iq4"
DEFAULT_APPROVAL_MODE = "ask"
DEFAULT_MAX_TOOL_ROUNDS = 100
DEFAULT_MAX_AGENT_DEPTH = 2
DEFAULT_TIMEOUT = 300
DEFAULT_DEBATE_ENABLED = True
ENV_OLLAMA_HOST = "OLLAMA_HOST"
ENV_OLLAMA_CODE_MODEL = "OLLAMA_CODE_MODEL"
ENV_OLLAMA_CODE_TEST_CMD = "OLLAMA_CODE_TEST_CMD"
ENV_OLLAMA_CODE_DEBATE = "OLLAMA_CODE_DEBATE"


@dataclass(frozen=True)
class CliConfig:
    host: str | None = None
    model: str | None = None
    approval: str | None = None
    max_tool_rounds: int | None = None
    max_agent_depth: int | None = None
    timeout: int | None = None
    test_cmd: str | None = None
    debate: bool | None = None
    path: Path | None = None


def resolve_config_path(workspace_root: Path, raw_path: str | Path | None = None) -> Path:
    if raw_path is None:
        return (workspace_root / DEFAULT_CONFIG_PATH).resolve(strict=False)
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = workspace_root / candidate
    return candidate.resolve(strict=False)


def _config_value(payload: dict[str, Any], key: str, path: Path) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f'Config value "{key}" must be a string in {path}')
    stripped = value.strip()
    return stripped or None


def _int_config_value(payload: dict[str, Any], key: str, path: Path) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or value < 1:
        raise ValueError(f'Config value "{key}" must be a positive integer in {path}')
    return value


def _approval_config_value(payload: dict[str, Any], path: Path) -> str | None:
    value = _config_value(payload, "approval", path)
    if value is None:
        return None
    if value not in {"ask", "auto", "read-only"}:
        raise ValueError(f'Config value "approval" must be ask, auto, or read-only in {path}')
    return value


def _bool_config_value(payload: dict[str, Any], key: str, path: Path) -> bool | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f'Config value "{key}" must be a boolean in {path}')
    return value


def load_config(workspace_root: Path, raw_path: str | Path | None = None) -> CliConfig:
    path = resolve_config_path(workspace_root, raw_path)
    explicit_path = raw_path is not None
    if not path.exists():
        if explicit_path:
            raise ValueError(f"Config file not found: {path}")
        return CliConfig()
    if not path.is_file():
        raise ValueError(f"Config path is not a file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid config JSON in {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a JSON object: {path}")
    ollama_payload = payload.get("ollama")
    if ollama_payload is None:
        data = payload
    else:
        if not isinstance(ollama_payload, dict):
            raise ValueError(f'Config key "ollama" must be a JSON object in {path}')
        data = ollama_payload
    return CliConfig(
        host=_config_value(data, "host", path),
        model=_config_value(data, "model", path),
        approval=_approval_config_value(data, path),
        max_tool_rounds=_int_config_value(data, "max_tool_rounds", path),
        max_agent_depth=_int_config_value(data, "max_agent_depth", path),
        timeout=_int_config_value(data, "timeout", path),
        test_cmd=_config_value(data, "test_cmd", path),
        debate=_bool_config_value(data, "debate", path),
        path=path,
    )
