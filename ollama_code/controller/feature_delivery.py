from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class CliFeatureCapabilities:
    has_stats: bool = False
    has_priority_filter: bool = False
    has_json_flag: bool = False
    has_limit_flag: bool = False
    has_due_before: bool = False

    def any(self) -> bool:
        return bool(
            self.has_stats
            or self.has_priority_filter
            or self.has_json_flag
            or self.has_limit_flag
            or self.has_due_before
        )


def typed_cli_flag_protocol_enabled(*, request_text: str, request_is_cli_flag_bundle: bool) -> bool:
    return bool(request_is_cli_flag_bundle)


def cli_feature_capabilities(candidate_source: str, request_text: str = "") -> CliFeatureCapabilities:
    return CliFeatureCapabilities(
        has_stats=bool(re.search(r"add_parser\(\s*['\"]stats['\"]", candidate_source)),
        has_priority_filter="--priority" in candidate_source and bool(re.search(r"add_parser\(\s*['\"]list['\"]", candidate_source)),
        has_json_flag="--json" in candidate_source,
        has_limit_flag="--limit" in candidate_source and "--limit" in request_text,
        has_due_before="--due-before" in candidate_source,
    )


def cli_proof_command_argvs(source_path: str, candidate_source: str, request_text: str = "") -> list[list[str]]:
    caps = cli_feature_capabilities(candidate_source, request_text)
    commands: list[list[str]] = []
    if caps.has_stats:
        commands.append([source_path, "stats"])
    if caps.has_priority_filter:
        commands.append([source_path, "list", "--priority", "high"])
    if caps.has_due_before:
        commands.append([source_path, "list", "--due-before", "2026-07-06"])
        if caps.has_priority_filter:
            commands.append([source_path, "list", "--priority", "high", "--due-before", "2026-07-06"])
    if caps.has_json_flag:
        commands.append([source_path, "--json"])
        if "--tag" in candidate_source:
            commands.append([source_path, "--tag", "work", "--json"])
            if caps.has_limit_flag:
                commands.append([source_path, "--tag", "work", "--limit", "1", "--json"])
    elif caps.has_limit_flag:
        commands.append([source_path, "--limit", "1"])
    return commands


def cli_readme_additions(candidate_source: str, request_text: str, existing_readme: str) -> list[str]:
    caps = cli_feature_capabilities(candidate_source, request_text)
    if not caps.any():
        return []
    lowered = existing_readme.lower()
    additions: list[str] = []
    if caps.has_priority_filter and "--priority" not in lowered:
        additions.append("- `list --priority high` filters tasks by priority.")
    if caps.has_stats and "stats" not in lowered:
        additions.append("- `stats` prints counts by status and priority.")
    if caps.has_json_flag and "--json" not in lowered:
        additions.append("- `--json` prints the selected items as JSON objects.")
    if caps.has_limit_flag and "--limit" not in lowered:
        additions.append("- `--limit N` limits the selected items after filtering and works with `--json`.")
    if caps.has_due_before and "--due-before" not in lowered:
        additions.append("- `list --due-before YYYY-MM-DD` filters tasks by due date and can be combined with `--priority`.")
    return additions
