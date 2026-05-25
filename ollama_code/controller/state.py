from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ollama_code.agent_protocol import AgentResult


@dataclass
class ControllerTurnState:
    round_number: int = 0
    successful_tool_results: list[dict[str, Any]] = field(default_factory=list)
    tool_calls_this_turn: list[dict[str, Any]] = field(default_factory=list)
    satisfied_tool_names: set[str] = field(default_factory=set)

    def execute(self, agent: Any, *, name: str, arguments: dict[str, Any], request_text: str) -> dict[str, Any]:
        self.round_number += 1
        return agent._execute_controller_tool(
            name=name,
            arguments=arguments,
            request_text=request_text,
            round_number=self.round_number,
            successful_tool_results=self.successful_tool_results,
            satisfied_tool_names=self.satisfied_tool_names,
            tool_calls_this_turn=self.tool_calls_this_turn,
        )

    def finish(self, agent: Any, message: str, *, tool: str) -> AgentResult:
        return agent._record_synthesized_final(message, tool=tool, round_number=self.round_number)
