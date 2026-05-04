from __future__ import annotations

import os
from typing import Any


ENV_OLLAMA_CODE_FEATURE_PROFILE = "OLLAMA_CODE_FEATURE_PROFILE"

FEATURE_NAMES = {
    "schema",
    "context-pack",
    "evidence-handles",
    "num-predict-caps",
    "structured-edits",
    "trajectory-guards",
    "contract-guards",
}

PROFILE_FEATURES: dict[str, set[str]] = {
    "baseline": set(),
    "schema": {"schema"},
    "context-pack": {"context-pack"},
    "evidence-handles": {"evidence-handles"},
    "num-predict-caps": {"num-predict-caps"},
    "structured-edits": {"structured-edits"},
    "trajectory-guards": {"trajectory-guards"},
    "contract-guards": {"contract-guards"},
    "all": set(FEATURE_NAMES),
}


def active_feature_profile() -> str:
    raw = os.environ.get(ENV_OLLAMA_CODE_FEATURE_PROFILE, "baseline").strip().lower()
    return raw or "baseline"


def active_features(profile: str | None = None) -> set[str]:
    selected = active_feature_profile() if profile is None else profile.strip().lower()
    if selected in PROFILE_FEATURES:
        return set(PROFILE_FEATURES[selected])
    features: set[str] = set()
    for part in selected.split(","):
        name = part.strip().lower()
        if name in FEATURE_NAMES:
            features.add(name)
    return features


def feature_enabled(name: str) -> bool:
    return name in active_features()


PRIMARY_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["tool", "final"]},
        "name": {"type": "string"},
        "arguments": {"type": "object"},
        "message": {"type": "string"},
    },
    "required": ["type"],
    "additionalProperties": True,
}

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["accept", "retry"]},
        "reason": {"type": "string"},
        "required_tools": {"type": "array", "items": {"type": "string"}},
        "forbidden_tools": {"type": "array", "items": {"type": "string"}},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "validation_steps": {"type": "array", "items": {"type": "string"}},
        "repair_plan": {"type": "array", "items": {"type": "string"}},
        "claim_checks": {"type": "array", "items": {"type": "object"}},
        "rewrite_guidance": {"type": "array", "items": {"type": "string"}},
        "rewrite_from_evidence": {"type": "boolean"},
    },
    "required": ["verdict"],
    "additionalProperties": True,
}

REWRITER_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["final"]},
        "message": {"type": "string"},
    },
    "required": ["type", "message"],
    "additionalProperties": True,
}


def response_format_for_purpose(purpose: str, fallback: str | dict[str, Any] | None) -> str | dict[str, Any] | None:
    if not feature_enabled("schema") or fallback != "json":
        return fallback
    if purpose == "primary":
        return PRIMARY_RESPONSE_SCHEMA
    if purpose in {"verification", "final_verifier", "assumption_audit", "artifact_reconciliation", "reconciliation"}:
        return VERDICT_SCHEMA
    if purpose in {"verification_rewrite", "final_rewrite"}:
        return REWRITER_RESPONSE_SCHEMA
    return fallback


def options_for_purpose(purpose: str, *, primary_can_emit_large_payload: bool = False) -> dict[str, Any]:
    if not feature_enabled("num-predict-caps"):
        return {}
    if purpose == "primary":
        return {} if primary_can_emit_large_payload else {"num_predict": 256}
    if purpose in {"verification", "final_verifier", "assumption_audit", "artifact_reconciliation", "reconciliation"}:
        return {"temperature": 0, "num_predict": 192}
    if purpose in {"verification_rewrite", "final_rewrite"}:
        return {"temperature": 0, "num_predict": 256}
    return {}
