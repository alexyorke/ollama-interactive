from __future__ import annotations


def resolve_requested_model(model: str, available: set[str]) -> str | None:
    if model in available:
        return model
    latest = f"{model}:latest"
    if latest in available:
        return latest
    return None
