from __future__ import annotations

import json
import os
import shlex
from pathlib import Path, PurePosixPath
import shutil
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from terminal_bench.agents.base_agent import AgentResult, BaseAgent
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.terminal.docker_compose_manager import DockerComposeManager
from terminal_bench.terminal.models import TerminalCommand
from terminal_bench.terminal.tmux_session import TmuxSession


DEFAULT_CONTAINER_OLLAMA_HOST = "http://host.docker.internal:11437"


def _container_log_path(session_name: str) -> PurePosixPath:
    return PurePosixPath(DockerComposeManager.CONTAINER_SESSION_LOGS_PATH) / f"{session_name}.log"


def _container_recording_path(session_name: str) -> PurePosixPath:
    return PurePosixPath(DockerComposeManager.CONTAINER_SESSION_LOGS_PATH) / f"{session_name}.cast"


# Terminal-Bench currently builds in-container Linux paths with pathlib.Path.
# On Windows hosts that becomes backslash-prefixed paths like "\tmp", which
# break docker archive and tmux bootstrap calls. Patch those to POSIX paths
# from the adapter module so the fix stays local to this repo.
DockerComposeManager.CONTAINER_TEST_DIR = PurePosixPath("/tests")
TmuxSession._GET_ASCIINEMA_TIMESTAMP_SCRIPT_CONTAINER_PATH = PurePosixPath("/tmp/get-asciinema-timestamp.sh")
TmuxSession.logging_path = property(lambda self: _container_log_path(self._session_name))
TmuxSession._recording_path = property(
    lambda self: None if self._disable_recording else _container_recording_path(self._session_name)
)


def _normalize_cli_extra_args(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return shlex.split(stripped)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [str(parsed)]
    return [str(value)]


def _extract_usage(payload: dict[str, Any]) -> tuple[int, int]:
    prompt_tokens = 0
    output_tokens = 0
    for event in payload.get("events", []):
        if not isinstance(event, dict) or event.get("type") != "llm_call":
            continue
        prompt_tokens += int(event.get("prompt_tokens") or 0)
        output_tokens += int(event.get("output_tokens") or 0)
    return prompt_tokens, output_tokens


class OllamaCodeTerminalBenchAgent(BaseAgent):
    CONTAINER_SOURCE_ROOT = PurePosixPath("/tmp/ollama-code-src")
    CONTAINER_SESSION_FILE = PurePosixPath("/app/.ollama-code/terminal-bench-session.json")
    CONTAINER_EXIT_FILE = PurePosixPath("/app/.ollama-code/terminal-bench-exit.txt")

    @staticmethod
    def name() -> str:
        return "ollama-code-terminal-bench"

    def __init__(
        self,
        model_name: str | None = None,
        ollama_host: str | None = None,
        approval: str = "auto",
        max_tool_rounds: int = 20,
        quiet: bool = True,
        no_indexer: bool = True,
        require_llm_for_turn: bool = True,
        cli_extra_args: list[str] | str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model_name = self._normalize_model_name(model_name or "gemma4:e4b")
        self._ollama_host = self._container_ollama_host(ollama_host or os.environ.get("OLLAMA_HOST") or DEFAULT_CONTAINER_OLLAMA_HOST)
        self._approval = str(approval or "auto")
        self._max_tool_rounds = max(1, int(max_tool_rounds))
        self._quiet = bool(quiet)
        self._no_indexer = bool(no_indexer)
        self._require_llm_for_turn = bool(require_llm_for_turn)
        self._cli_extra_args = _normalize_cli_extra_args(cli_extra_args)

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        with TemporaryDirectory(prefix="ollama-code-tb-") as temp_root:
            staged_root = self._stage_repo_package(Path(temp_root))
            self._copy_repo_package(session, staged_root)
            command = self._agent_command(self._render_instruction(instruction))
            try:
                session.send_command(command)
            except TimeoutError:
                self._write_logs(session, logging_dir=logging_dir)
                return AgentResult(failure_mode=FailureMode.AGENT_TIMEOUT)

            payload = self._read_session_payload(session)
            prompt_tokens, output_tokens = _extract_usage(payload)
            exit_code = self._read_exit_code(session)
            self._write_logs(session, payload=payload, logging_dir=logging_dir)
            return AgentResult(
                total_input_tokens=prompt_tokens,
                total_output_tokens=output_tokens,
                failure_mode=FailureMode.NONE if exit_code == 0 else FailureMode.UNKNOWN_AGENT_ERROR,
            )

    def _stage_repo_package(self, temp_root: Path) -> Path:
        repo_root = Path(__file__).resolve().parents[1]
        source_root = repo_root / "ollama_code"
        staged_root = temp_root / "ollama_code"
        for source_path in source_root.rglob("*"):
            rel = source_path.relative_to(source_root)
            if "__pycache__" in rel.parts:
                continue
            if source_path.is_dir():
                continue
            target_path = staged_root / rel
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
        return temp_root

    def _copy_repo_package(self, session: TmuxSession, staged_root: Path) -> None:
        session.copy_to_container(
            paths=[staged_root],
            container_dir=str(self.CONTAINER_SOURCE_ROOT),
        )

    def _agent_command(self, instruction: str) -> TerminalCommand:
        cli_parts = [
            "__PYTHON_BIN__",
            "-m",
            "ollama_code.cli",
            "--cwd",
            "/app",
            "--model",
            self._model_name,
            "--approval",
            self._approval,
            "--max-tool-rounds",
            str(self._max_tool_rounds),
            "--session-file",
            str(self.CONTAINER_SESSION_FILE),
            *self._cli_extra_args,
        ]
        if self._quiet:
            cli_parts.append("--quiet")
        if self._no_indexer:
            cli_parts.append("--no-indexer")
        if self._require_llm_for_turn:
            cli_parts.append("--require-llm-for-turn")
        cli_parts.append(instruction)
        cli_command = " ".join(
            '"$PYTHON_BIN"' if part == "__PYTHON_BIN__" else shlex.quote(part)
            for part in cli_parts
        )
        script = (
            "set -e\n"
            "PYTHON_BIN=\"$(command -v python3 || command -v python)\"\n"
            "if [ -z \"$PYTHON_BIN\" ]; then\n"
            f"  printf '%s' 127 > {shlex.quote(str(self.CONTAINER_EXIT_FILE))}\n"
            "  exit 127\n"
            "fi\n"
            f"export PYTHONPATH={shlex.quote(str(self.CONTAINER_SOURCE_ROOT))}${{PYTHONPATH:+:$PYTHONPATH}}\n"
            f"export OLLAMA_HOST={shlex.quote(self._ollama_host)}\n"
            "cd /app\n"
            f"mkdir -p {shlex.quote(str(self.CONTAINER_SESSION_FILE.parent))}\n"
            "set +e\n"
            f"{cli_command}\n"
            "status=$?\n"
            "set -e\n"
            f"printf '%s' \"$status\" > {shlex.quote(str(self.CONTAINER_EXIT_FILE))}\n"
            "exit \"$status\"\n"
        )
        return TerminalCommand(
            command=f"sh -lc {shlex.quote(script)}",
            min_timeout_sec=0.0,
            max_timeout_sec=float("inf"),
            block=True,
            append_enter=True,
        )

    def _read_session_payload(self, session: TmuxSession) -> dict[str, Any]:
        result = session.container.exec_run(
            [
                "sh",
                "-lc",
                f"test -f {shlex.quote(str(self.CONTAINER_SESSION_FILE))} && cat {shlex.quote(str(self.CONTAINER_SESSION_FILE))}",
            ]
        )
        if result.exit_code != 0 or not result.output:
            return {"events": []}
        try:
            return json.loads(result.output.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            return {"events": []}

    def _read_exit_code(self, session: TmuxSession) -> int:
        result = session.container.exec_run(
            [
                "sh",
                "-lc",
                f"test -f {shlex.quote(str(self.CONTAINER_EXIT_FILE))} && cat {shlex.quote(str(self.CONTAINER_EXIT_FILE))}",
            ]
        )
        if result.exit_code != 0 or not result.output:
            return 1
        try:
            return int(result.output.decode("utf-8", errors="replace").strip() or "1")
        except ValueError:
            return 1

    def _write_logs(
        self,
        session: TmuxSession,
        *,
        payload: dict[str, Any] | None = None,
        logging_dir: Path | None = None,
    ) -> None:
        if logging_dir is None:
            return
        logging_dir.mkdir(parents=True, exist_ok=True)
        pane = session.capture_pane(capture_entire=True)
        (logging_dir / "terminal-pane.txt").write_text(pane, encoding="utf-8")
        if payload is not None:
            (logging_dir / "ollama-code-session.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _normalize_model_name(self, value: str) -> str:
        text = str(value).strip()
        if not text:
            return "gemma4:e4b"
        return text.split("/", 1)[-1]

    def _container_ollama_host(self, host: str) -> str:
        text = str(host).strip()
        if not text:
            return DEFAULT_CONTAINER_OLLAMA_HOST
        if not text.startswith(("http://", "https://")):
            text = f"http://{text}"
        parsed = urlsplit(text)
        hostname = parsed.hostname or ""
        if hostname not in {"127.0.0.1", "localhost"}:
            return text.rstrip("/")
        port = f":{parsed.port}" if parsed.port else ""
        netloc = f"host.docker.internal{port}"
        return urlunsplit((parsed.scheme or "http", netloc, parsed.path, parsed.query, parsed.fragment)).rstrip("/")
