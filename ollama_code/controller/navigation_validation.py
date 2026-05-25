from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from ollama_code.agent_protocol import AgentResult, SymbolReadSpec, TargetLineReadSpec

from .state import ControllerTurnState


@dataclass(frozen=True)
class NavigationValidationTurn:
    request_text: str
    target_line_read: TargetLineReadSpec | None
    symbol_read: SymbolReadSpec | None
    exact_shell_command: str | None
    expected_exact_reply_text: str | None
    required_tool_names: set[str]
    forbidden_tool_names: set[str]
    requested_git_diff_mode: str | None


class NavigationValidationController:
    def __init__(self, agent: Any) -> None:
        self.agent = agent

    def handle(self, turn: NavigationValidationTurn, state: ControllerTurnState | None = None) -> AgentResult | None:
        state = state or ControllerTurnState()
        request_text = turn.request_text
        required_tool_names = turn.required_tool_names
        forbidden_tool_names = turn.forbidden_tool_names
        expected_exact_reply_text = turn.expected_exact_reply_text
        lowered = request_text.lower()

        if turn.exact_shell_command and "run_shell" not in forbidden_tool_names:
            result = state.execute(
                self.agent,
                name="run_shell",
                arguments={"command": turn.exact_shell_command},
                request_text=request_text,
            )
            message = self._synthesize(
                turn=turn,
                state=state,
                name="run_shell",
                arguments={"command": turn.exact_shell_command},
                result=result,
            )
            if message:
                return state.finish(self.agent, message, tool="run_shell")
            return None

        if turn.exact_shell_command and "run_test" in required_tool_names and "run_test" not in forbidden_tool_names:
            result = state.execute(
                self.agent,
                name="run_test",
                arguments={"command": turn.exact_shell_command},
                request_text=request_text,
            )
            message = self._synthesize(
                turn=turn,
                state=state,
                name="run_test",
                arguments={"command": turn.exact_shell_command},
                result=result,
            )
            if message:
                return state.finish(self.agent, message, tool="run_test")
            return None

        if "run_test" in required_tool_names and self.agent.tools.default_test_command and "run_test" not in forbidden_tool_names:
            arguments = {"command": self.agent.tools.default_test_command}
            result = state.execute(self.agent, name="run_test", arguments=arguments, request_text=request_text)
            message = self._synthesize(
                turn=turn,
                state=state,
                name="run_test",
                arguments=arguments,
                result=result,
            )
            if message:
                return state.finish(self.agent, message, tool="run_test")
            return None

        if {"git_status", "git_diff"}.issubset(required_tool_names) and "git_status" not in forbidden_tool_names and "git_diff" not in forbidden_tool_names:
            value_match = re.search(r"\breturn\s+([A-Za-z0-9_]+)\b", request_text)
            path = self.agent._requested_git_tool_path(request_text)
            if value_match and path:
                status_args = {"path": path}
                state.execute(self.agent, name="git_status", arguments=status_args, request_text=request_text)
                diff_args: dict[str, Any] = {"path": path}
                if turn.requested_git_diff_mode == "staged":
                    diff_args["cached"] = True
                result = state.execute(self.agent, name="git_diff", arguments=diff_args, request_text=request_text)
                message = self._synthesize(
                    turn=turn,
                    state=state,
                    name="git_diff",
                    arguments=diff_args,
                    result=result,
                )
                if message:
                    return state.finish(self.agent, message, tool="git_diff")
            return None

        context_followup_sequence = self.agent._requested_context_followup_mechanical_sequence(
            request_text,
            forbidden_tool_names=forbidden_tool_names,
        )
        if context_followup_sequence:
            messages: list[str] = []
            final_tool = context_followup_sequence[-1][1].name
            for index, (fragment, spec) in enumerate(context_followup_sequence):
                result = state.execute(
                    self.agent,
                    name=spec.name,
                    arguments=spec.arguments,
                    request_text=fragment,
                )
                message = self._synthesize(
                    turn=NavigationValidationTurn(
                        request_text=fragment,
                        target_line_read=turn.target_line_read,
                        symbol_read=turn.symbol_read,
                        exact_shell_command=turn.exact_shell_command,
                        expected_exact_reply_text=expected_exact_reply_text,
                        required_tool_names={spec.name},
                        forbidden_tool_names=forbidden_tool_names,
                        requested_git_diff_mode=turn.requested_git_diff_mode,
                    ),
                    state=state,
                    name=spec.name,
                    arguments=spec.arguments,
                    result=result,
                )
                if spec.name == "search":
                    message = self._match_path_message(result)
                if message and (index == len(context_followup_sequence) - 1 or spec.name in {"list_files", "search", "discover_validators"}):
                    messages.append(message)
            if messages:
                return state.finish(self.agent, "\n".join(messages), tool=final_tool)
            return None

        mechanical_tool = self.agent._requested_mechanical_tool_call(request_text, forbidden_tool_names=forbidden_tool_names)
        if mechanical_tool is not None:
            result = state.execute(
                self.agent,
                name=mechanical_tool.name,
                arguments=mechanical_tool.arguments,
                request_text=request_text,
            )
            message = self._synthesize(
                turn=turn,
                state=state,
                name=mechanical_tool.name,
                arguments=mechanical_tool.arguments,
                result=result,
            )
            if self.agent._should_chain_run_test_after_mechanical(
                request_text=request_text,
                mechanical_tool_name=mechanical_tool.name,
                forbidden_tool_names=forbidden_tool_names,
            ):
                context_message = message
                test_args = {"command": self.agent.tools.default_test_command} if self.agent.tools.default_test_command else {}
                test_result = state.execute(
                    self.agent,
                    name="run_test",
                    arguments=test_args,
                    request_text=request_text,
                )
                test_message = self._synthesize(
                    turn=turn,
                    state=state,
                    name="run_test",
                    arguments=test_args,
                    result=test_result,
                )
                if context_message and test_message:
                    return state.finish(self.agent, context_message + "\n" + test_message, tool="run_test")
                if test_message:
                    return state.finish(self.agent, test_message, tool="run_test")
            if message:
                return state.finish(self.agent, message, tool=mechanical_tool.name)
            return None

        if (
            turn.symbol_read is not None
            and "search_symbols" not in forbidden_tool_names
            and "read_symbol" not in forbidden_tool_names
            and ("token" in lowered or "marker" in lowered or ("return" in lowered and "value" in lowered))
        ):
            search_args = {"query": turn.symbol_read.symbol, "path": turn.symbol_read.path}
            search_result = state.execute(
                self.agent,
                name="search_symbols",
                arguments=search_args,
                request_text=request_text,
            )
            if search_result.get("ok") is not True:
                return None
            read_args = {"path": turn.symbol_read.path, "symbol": turn.symbol_read.symbol, "include_context": 0}
            result = state.execute(
                self.agent,
                name="read_symbol",
                arguments=read_args,
                request_text=request_text,
            )
            message = self._synthesize(
                turn=turn,
                state=state,
                name="read_symbol",
                arguments=read_args,
                result=result,
            )
            if message:
                return state.finish(self.agent, message, tool="read_symbol")
            return None

        natural_read_path = self.agent._requested_natural_read_file_path(request_text)
        read_path = turn.target_line_read.path if turn.target_line_read is not None else (self.agent._requested_read_file_path(request_text) or natural_read_path)
        exact_line_read_requested = turn.target_line_read is not None and self.agent._request_asks_exact_line_text(request_text)
        direct_file_contents_requested = natural_read_path is not None and self.agent._request_asks_direct_file_contents(request_text)
        if read_path and "read_file" not in forbidden_tool_names and (
            self.agent._request_asks_token_only(request_text)
            or self.agent._request_expects_exact_tool_error(request_text)
            or exact_line_read_requested
            or direct_file_contents_requested
        ):
            if turn.target_line_read is not None:
                read_args = {"path": turn.target_line_read.path, "start": turn.target_line_read.start, "end": turn.target_line_read.end}
            else:
                read_args = {"path": read_path}
            read_count = 2 if self.agent._request_mentions_repeated_read(request_text) else 1
            result: dict[str, Any] = {}
            for _ in range(read_count):
                result = state.execute(
                    self.agent,
                    name="read_file",
                    arguments=read_args,
                    request_text=request_text,
                )
            message = self._synthesize(
                turn=turn,
                state=state,
                name="read_file",
                arguments=read_args,
                result=result,
            )
            if message:
                return state.finish(self.agent, message, tool="read_file")

        if lowered.startswith("use read_file") and self.agent._request_expects_exact_tool_error(request_text) and read_path and "read_file" not in forbidden_tool_names:
            result = state.execute(
                self.agent,
                name="read_file",
                arguments={"path": read_path},
                request_text=request_text,
            )
            message = str(result.get("summary") or result.get("output") or "").strip()
            if message:
                return state.finish(self.agent, message, tool="read_file")
        return None

    def _synthesize(
        self,
        *,
        turn: NavigationValidationTurn,
        state: ControllerTurnState,
        name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> str | None:
        return self.agent._synthesize_final_from_tool_result(
            request_text=turn.request_text,
            name=name,
            arguments=arguments,
            result=result,
            successful_tool_results=state.successful_tool_results,
            satisfied_tool_names=state.satisfied_tool_names,
            required_tool_names=turn.required_tool_names,
            expected_exact_reply_text=turn.expected_exact_reply_text,
        )

    def _match_path_message(self, result: dict[str, Any]) -> str | None:
        output = str(result.get("output", ""))
        for line in output.splitlines():
            match = re.match(r"(?P<path>.+):\d+:", line)
            if not match:
                continue
            raw_path = match.group("path").strip()
            try:
                path = Path(raw_path)
                label = self.agent.tools.relative_label(path) if path.is_absolute() else raw_path.replace("\\", "/")
            except (OSError, ValueError):
                label = raw_path.replace("\\", "/")
            return f"{label} contains the match."
        return None
