from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(".ollama-code") / "config.json"


@dataclass(frozen=True)
class CliConfig:
    host: str | None = None
    model: str | None = None
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
        path=path,
    )
