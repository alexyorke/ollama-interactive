from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_output_paths(
    *,
    output: Path | None,
    output_json: Path | None,
    output_md: Path | None,
    default_json: Path,
    default_md: Path,
) -> tuple[Path, Path]:
    if output is None:
        return output_json or default_json, output_md or default_md
    suffix = output.suffix.lower()
    if suffix == ".json":
        return output_json or output, output_md or output.with_suffix(".md")
    if suffix == ".md":
        return output_json or output.with_suffix(".json"), output_md or output
    return output_json or output.with_suffix(".json"), output_md or output.with_suffix(".md")


def write_report_outputs(
    payload: dict[str, Any],
    markdown: str,
    *,
    output_json: Path,
    output_md: Path,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")
