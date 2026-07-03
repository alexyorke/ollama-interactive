from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from ollama_code.features import ENV_OLLAMA_CODE_FEATURE_PROFILE
from ollama_code.ollama_client import ChatResponse
from ollama_code.tools import ToolExecutor


class FakeClient:
    def __init__(
        self,
        responses: list[str],
        *,
        script_verification: bool = False,
        script_assumption_audit: bool = False,
        script_final_rewrite: bool = False,
        script_reconciliation: bool = False,
        script_question_planner: bool = False,
        models: list[str] | None = None,
    ) -> None:
        self.responses = list(responses)
        self.script_verification = script_verification
        self.script_assumption_audit = script_assumption_audit
        self.script_final_rewrite = script_final_rewrite
        self.script_reconciliation = script_reconciliation
        self.script_question_planner = script_question_planner
        self.models = list(models) if models is not None else ["fake-model"]
        self.calls: list[dict[str, object]] = []
        self.interrupt_events: list[object | None] = []

    def set_interrupt_event(self, event: object | None) -> None:
        self.interrupt_events.append(event)

    def list_models(self) -> list[str]:
        return list(self.models)

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: str = "json",
        on_thinking: object | None = None,
        think: bool | None = None,
        options: dict[str, object] | None = None,
    ) -> ChatResponse:
        system_prompt = messages[0]["content"] if messages else ""
        self.calls.append(
            {
                "model": model,
                "messages": list(messages),
                "response_format": response_format,
                "on_thinking": on_thinking,
                "think": think,
                "options": options,
            }
        )
        if isinstance(system_prompt, str):
            if system_prompt.startswith("You are a grounded final verifier") and not self.script_verification:
                return ChatResponse(content='{"verdict":"accept"}', model=model, raw={})
            if system_prompt.startswith("You are a tool-step assumption auditor") and not self.script_assumption_audit:
                return ChatResponse(
                    content='{"verdict":"accept","reason":"","assumptions":["tool is needed"],"validation_steps":["run the tool to gather evidence"],"required_tools":[],"forbidden_tools":[]}',
                    model=model,
                    raw={},
                )
            if system_prompt.startswith("You are an evidence-backed final rewriter") and not self.script_final_rewrite:
                import json

                payload = json.loads(messages[1]["content"])
                return ChatResponse(
                    content=json.dumps({"type": "final", "message": payload.get("candidate_final_answer", "")}),
                    model=model,
                    raw={},
                )
            if system_prompt.startswith("You are an artifact reconciliation critic") and not self.script_reconciliation:
                return ChatResponse(content='{"verdict":"accept","reason":"","repair_plan":[],"required_tools":[],"forbidden_tools":[]}', model=model, raw={})
            if system_prompt.startswith("You are a clarification planner") and not self.script_question_planner:
                return ChatResponse(content='{"verdict":"proceed","reason":"No implementation-changing ambiguity found.","ambiguities":[],"questions":[]}', model=model, raw={})
        return ChatResponse(content=self.responses.pop(0), model=model, raw={})


class CountingToolExecutor(ToolExecutor):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.execute_counts: dict[str, int] = {}

    def execute(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        self.execute_counts[name] = self.execute_counts.get(name, 0) + 1
        return super().execute(name, arguments)


class RawImplementationSpecCountingToolExecutor(CountingToolExecutor):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.raw_implementation_spec_calls = 0

    def implementation_spec(self, source_path: str, test_path: str | None = None, limit: int = 40) -> dict[str, object]:
        self.raw_implementation_spec_calls += 1
        return super().implementation_spec(source_path, test_path, limit)


class EmptySelectTestsToolExecutor(CountingToolExecutor):
    def execute(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        self.execute_counts[name] = self.execute_counts.get(name, 0) + 1
        if name == "select_tests":
            return {
                "ok": True,
                "tool": "select_tests",
                "summary": "No targeted tests found.",
                "test_commands": [],
            }
        return ToolExecutor.execute(self, name, arguments)


class EmptySelectTestsLintFallbackToolExecutor(EmptySelectTestsToolExecutor):
    def __init__(self, *args: object, fallback_command: str, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.fallback_command = fallback_command

    def execute(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        self.execute_counts[name] = self.execute_counts.get(name, 0) + 1
        if name == "select_tests":
            return {
                "ok": True,
                "tool": "select_tests",
                "summary": "No targeted tests found.",
                "test_commands": [],
            }
        if name == "discover_validators":
            return {
                "ok": True,
                "tool": "discover_validators",
                "validators": [
                    {
                        "kind": "lint",
                        "lang": "custom",
                        "command": self.fallback_command,
                        "available": True,
                        "reason": "custom lint fallback",
                    }
                ],
                "summary": "Discovered lint fallback command.",
                "output": f"lint custom: {self.fallback_command} available=True reason=custom lint fallback",
            }
        return ToolExecutor.execute(self, name, arguments)


class WorkflowValidatorToolExecutor(CountingToolExecutor):
    def __init__(self, *args: object, workflow_command: str, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.workflow_command = workflow_command

    def execute(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        self.execute_counts[name] = self.execute_counts.get(name, 0) + 1
        if name == "discover_validators":
            return {
                "ok": True,
                "tool": "discover_validators",
                "validators": [
                    {
                        "kind": "test",
                        "lang": "python",
                        "command": str(self.default_test_command or ""),
                        "available": True,
                        "reason": "configured tests",
                    },
                    {
                        "kind": "lint",
                        "lang": "github-actions",
                        "command": self.workflow_command,
                        "available": True,
                        "reason": "GitHub Actions workflow files found.",
                    },
                ],
                "summary": "Discovered workflow validator.",
                "output": f"lint github-actions: {self.workflow_command} available=True reason=GitHub Actions workflow files found.",
            }
        return ToolExecutor.execute(self, name, arguments)


class AgentTestBase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._env_patcher = patch.dict(
            "os.environ",
            {
                ENV_OLLAMA_CODE_FEATURE_PROFILE: "",
                "OLLAMA_CODE_MODEL": "",
                "OLLAMA_CODE_REQUIRE_LLM_FOR_TURN": "",
            },
            clear=False,
        )
        self._env_patcher.start()
        self.addCleanup(self._env_patcher.stop)

    def _workspace_scratch(self) -> Path:
        root = (Path.cwd() / "verify_scratch" / f"test-agent-{uuid4().hex}").resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return root

    def _init_git_repo_or_skip(self, root: Path) -> None:
        if shutil.which("git") is None:
            self.skipTest("git is not installed")
        try:
            subprocess.run(["git", "init"], cwd=root, capture_output=True, text=True, check=True)
            subprocess.run(["git", "config", "user.name", "Tests"], cwd=root, capture_output=True, text=True, check=True)
            subprocess.run(["git", "config", "user.email", "tests@example.com"], cwd=root, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            message = (exc.stderr or exc.stdout or str(exc)).strip()
            self.skipTest(f"git repo init is unavailable in this environment: {message}")
