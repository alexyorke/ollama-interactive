from __future__ import annotations


def resolve_requested_model(model: str, available: set[str]) -> str | None:
    if model in available:
        return model
    latest = f"{model}:latest"
    if latest in available:
        return latest
    return None


def resolve_requested_models(requested: list[str], available: set[str]) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for model in requested:
        candidate = resolve_requested_model(model, available)
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        resolved.append(candidate)
    return resolved
