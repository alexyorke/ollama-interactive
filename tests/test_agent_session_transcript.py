from __future__ import annotations

from datetime import datetime, timezone
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ollama_code.agent import (
    OllamaCodeAgent,
    TRANSCRIPT_DIAGNOSTIC_COLLECTION_LIMIT,
    TRANSCRIPT_DIAGNOSTIC_CYCLE_MARKER,
    TRANSCRIPT_DIAGNOSTIC_DICT_MARKER_KEY,
    TRANSCRIPT_DIAGNOSTIC_DEPTH_LIMIT,
    TRANSCRIPT_DIAGNOSTIC_DEPTH_MARKER,
    _workspace_roots_match,
)
from ollama_code.tools import ToolExecutor
from tests.agent_test_support import AgentTestBase, FakeClient


class AgentSessionTranscriptTests(AgentTestBase):
    def _cwd_agent(self, client: FakeClient | None = None, **kwargs: object) -> OllamaCodeAgent:
        resolved_client = client if client is not None else FakeClient([])
        return OllamaCodeAgent(
            client=resolved_client,
            tools=ToolExecutor(Path.cwd(), approval_mode="auto"),
            model="fake-model",
            **kwargs,
        )

    def _assert_single_turn_done_payload(self, payload: str) -> None:
        client = FakeClient([payload])
        agent = self._cwd_agent(client, debate_enabled=False)
        result = agent.handle_user("Say done.")
        self.assertEqual(result.message, "done")
        self.assertEqual(result.rounds, 1)
        self.assertEqual(len(client.calls), 1)

    def test_agent_retries_after_empty_model_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient(
                [
                    "",
                    '{"type":"final","message":"done"}',
                ]
            )
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
            result = agent.handle_user("Say done.")

        self.assertEqual(result.message, "done")
        self.assertEqual(len(client.calls), 2)

    def test_agent_retries_after_non_json_model_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient(
                [
                    "</html>",
                    '{"type":"final","message":"done"}',
                ]
            )
            tools = ToolExecutor(Path(tmp), approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False)
            result = agent.handle_user("Say done.")

        self.assertEqual(result.message, "done")
        self.assertEqual(len(client.calls), 2)

    def test_agent_prefers_last_agent_payload_in_mixed_model_output(self) -> None:
        self._assert_single_turn_done_payload('Context {"foo":1}\n{"type":"final","message":"done"}')

    def test_agent_accepts_singleton_array_wrapped_final_payload(self) -> None:
        self._assert_single_turn_done_payload('[{"type":"final","message":"done"}]')

    def test_relative_transcript_paths_use_workspace_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", session_file="scratch/session.json", debate_enabled=False)
            saved = agent.save_transcript("scratch/manual.json")

        self.assertEqual(agent.session_file, (root / "scratch" / "session.json").resolve())
        self.assertEqual(saved, (root / "scratch" / "manual.json").resolve())

    def test_autosave_batches_tool_call_and_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "note.txt").write_text("hello\n", encoding="utf-8")
            session = root / ".ollama-code" / "sessions" / "batched.json"
            client = FakeClient(
                [
                    '{"type":"tool","name":"read_file","arguments":{"path":"note.txt"}}',
                    '{"type":"final","message":"done"}',
                ]
            )
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", session_file=session, debate_enabled=False)
            original_save = agent.save_transcript
            save_count = 0

            def counted_save(path: str | Path | None = None) -> Path:
                nonlocal save_count
                save_count += 1
                return original_save(path)

            agent.save_transcript = counted_save  # type: ignore[method-assign]
            result = agent.handle_user("Read note.txt then say done.")
            payload = json.loads(session.read_text(encoding="utf-8"))

        self.assertEqual(result.message, "done")
        self.assertLessEqual(save_count, 4)
        self.assertTrue(any(event.get("type") == "tool_call" and event.get("name") == "read_file" for event in payload["events"]))
        self.assertTrue(any(event.get("type") == "tool_result" and event.get("name") == "read_file" for event in payload["events"]))
        self.assertTrue(any(event.get("type") == "assistant" and event.get("content") == "done" for event in payload["events"]))

    def test_save_transcript_preserves_existing_file_when_atomic_replace_fails(self) -> None:
        root = self._workspace_scratch()
        session = root / ".ollama-code" / "sessions" / "saved.json"
        session.parent.mkdir(parents=True, exist_ok=True)
        original = '{"keep":"original"}'
        session.write_text(original, encoding="utf-8")
        client = FakeClient([])
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", session_file=session, debate_enabled=False)

        with patch("ollama_code.sessions.os.replace", side_effect=OSError("replace failed")):
            with self.assertRaisesRegex(OSError, "replace failed"):
                agent.save_transcript()

        self.assertEqual(session.read_text(encoding="utf-8"), original)
        self.assertEqual(list(session.parent.glob(f".{session.name}.*.tmp")), [])

    def test_save_transcript_normalizes_and_compacts_diagnostic_payloads(self) -> None:
        root = self._workspace_scratch()
        session = root / ".ollama-code" / "sessions" / "saved.json"
        client = FakeClient([])
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", session_file=session, debate_enabled=False)
        full_message = "remember " + ("m" * 5000)
        agent.messages.append({"role": "user", "content": full_message})
        agent.events.append(
            {
                "type": "tool_result",
                "name": "run_test",
                "result": {
                    "ok": False,
                    "output": "x" * 12000,
                    "path": root / "nested" / "file.txt",
                    "payload": b"\xff\x00",
                    "when": datetime(2026, 5, 18, tzinfo=timezone.utc),
                    "nan": float("nan"),
                    "items": {"beta", "alpha"},
                },
            }
        )
        agent.llm_telemetry_events.append({"type": "llm_call", "preview": "y" * 9000})

        saved = agent.save_transcript()
        payload = json.loads(saved.read_text(encoding="utf-8"))
        result = payload["events"][0]["result"]

        self.assertEqual(payload["messages"][-1]["content"], full_message)
        self.assertEqual(result["path"], (root / "nested" / "file.txt").as_posix())
        self.assertEqual(result["payload"], "b'\\xff\\x00'")
        self.assertEqual(result["when"], "2026-05-18T00:00:00+00:00")
        self.assertEqual(result["nan"], "nan")
        self.assertEqual(result["items"], ["alpha", "beta"])
        self.assertIn("truncated", result["output"])
        self.assertLess(len(result["output"]), 5000)
        self.assertIn("truncated", payload["llm_telemetry_events"][0]["preview"])
        self.assertLess(len(payload["llm_telemetry_events"][0]["preview"]), 5000)

    def test_save_transcript_truncates_deeply_nested_diagnostic_payloads(self) -> None:
        root = self._workspace_scratch()
        session = root / ".ollama-code" / "sessions" / "saved.json"
        agent = OllamaCodeAgent(
            client=FakeClient([]),
            tools=ToolExecutor(root, approval_mode="auto"),
            model="fake-model",
            session_file=session,
            debate_enabled=False,
        )
        nested: dict[str, object] = {}
        current = nested
        for index in range(TRANSCRIPT_DIAGNOSTIC_DEPTH_LIMIT + 10):
            child: dict[str, object] = {"index": index}
            current["next"] = child
            current = child
        agent.events.append({"type": "tool_result", "name": "mcp_call", "result": nested})

        saved = agent.save_transcript()
        payload = json.loads(saved.read_text(encoding="utf-8"))
        cursor = payload["events"][0]["result"]
        for _ in range(TRANSCRIPT_DIAGNOSTIC_DEPTH_LIMIT + 2):
            if isinstance(cursor, str):
                break
            cursor = cursor["next"]

        self.assertEqual(cursor, TRANSCRIPT_DIAGNOSTIC_DEPTH_MARKER)

    def test_save_transcript_truncates_large_nested_diagnostic_collections(self) -> None:
        root = self._workspace_scratch()
        session = root / ".ollama-code" / "sessions" / "saved.json"
        agent = OllamaCodeAgent(
            client=FakeClient([]),
            tools=ToolExecutor(root, approval_mode="auto"),
            model="fake-model",
            session_file=session,
            debate_enabled=False,
        )
        nested_rows = [{"index": index} for index in range(TRANSCRIPT_DIAGNOSTIC_COLLECTION_LIMIT + 5)]
        nested_mapping = {f"key_{index}": index for index in range(TRANSCRIPT_DIAGNOSTIC_COLLECTION_LIMIT + 3)}
        agent.events.append({"type": "tool_result", "name": "mcp_call", "result": {"rows": nested_rows, "mapping": nested_mapping}})

        saved = agent.save_transcript()
        payload = json.loads(saved.read_text(encoding="utf-8"))
        result = payload["events"][0]["result"]

        self.assertEqual(len(result["rows"]), TRANSCRIPT_DIAGNOSTIC_COLLECTION_LIMIT + 1)
        self.assertEqual(result["rows"][-1], "[truncated 5 items for transcript]")
        self.assertEqual(result["rows"][0], {"index": 0})
        self.assertEqual(len(result["mapping"]), TRANSCRIPT_DIAGNOSTIC_COLLECTION_LIMIT + 1)
        self.assertEqual(result["mapping"][TRANSCRIPT_DIAGNOSTIC_DICT_MARKER_KEY], "[truncated 3 entries for transcript]")
        self.assertEqual(result["mapping"]["key_0"], 0)

    def test_save_transcript_replaces_circular_diagnostic_references(self) -> None:
        root = self._workspace_scratch()
        session = root / ".ollama-code" / "sessions" / "saved.json"
        agent = OllamaCodeAgent(
            client=FakeClient([]),
            tools=ToolExecutor(root, approval_mode="auto"),
            model="fake-model",
            session_file=session,
            debate_enabled=False,
        )
        nested: dict[str, object] = {"label": "root"}
        nested["self"] = nested
        agent.events.append({"type": "tool_result", "name": "mcp_call", "result": nested})

        saved = agent.save_transcript()
        payload = json.loads(saved.read_text(encoding="utf-8"))

        self.assertEqual(payload["events"][0]["result"]["self"], TRANSCRIPT_DIAGNOSTIC_CYCLE_MARKER)

    def test_agent_can_load_saved_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            session = root / ".ollama-code" / "sessions" / "saved.json"
            session.parent.mkdir(parents=True)
            session.write_text(
                '{"model":"fake-model","approval_mode":"auto","workspace_root":"'
                + root.as_posix()
                + '","messages":[{"role":"system","content":"sys"},{"role":"user","content":"remember TOKEN_42"}],"events":[{"type":"user","content":"remember TOKEN_42"}]}',
                encoding="utf-8",
            )
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="ask")
            agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", session_file="scratch/current.json", debate_enabled=False)
            loaded = agent.load_session(session)

        self.assertEqual(loaded, session.resolve())
        self.assertEqual(agent.session_path(), session.resolve())
        self.assertEqual(agent.model, "fake-model")
        self.assertEqual(agent.approval_mode(), "auto")
        self.assertEqual(agent.messages[1]["content"], "remember TOKEN_42")
        self.assertEqual(agent.events[0]["content"], "remember TOKEN_42")

    def test_restore_transcript_normalizes_oversized_diagnostic_payloads(self) -> None:
        root = self._workspace_scratch()
        agent = OllamaCodeAgent(
            client=FakeClient([]),
            tools=ToolExecutor(root, approval_mode="auto"),
            model="fake-model",
            debate_enabled=False,
        )
        payload = {
            "model": "fake-model",
            "approval_mode": "auto",
            "workspace_root": root.as_posix(),
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "remember me"},
            ],
            "events": [{"type": "tool_result", "payload": "x" * 12000}],
            "llm_telemetry_events": [{"type": "llm_call", "preview": "y" * 9000}],
        }

        agent.restore_transcript(payload)

        self.assertIn("truncated", agent.events[0]["payload"])
        self.assertLess(len(agent.events[0]["payload"]), 5000)
        self.assertIn("truncated", agent.llm_telemetry_events[0]["preview"])
        self.assertLess(len(agent.llm_telemetry_events[0]["preview"]), 5000)

    def test_restore_transcript_rejects_unsupported_message_role(self) -> None:
        root = self._workspace_scratch()
        agent = OllamaCodeAgent(
            client=FakeClient([]),
            tools=ToolExecutor(root, approval_mode="auto"),
            model="fake-model",
            debate_enabled=False,
        )
        payload = {
            "model": "fake-model",
            "approval_mode": "auto",
            "workspace_root": root.as_posix(),
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "tool", "content": "bad role"},
            ],
            "events": [],
        }

        with self.assertRaisesRegex(ValueError, "Saved session contains a malformed message"):
            agent.restore_transcript(payload)

    def test_restore_transcript_truncates_large_nested_diagnostic_collections(self) -> None:
        root = self._workspace_scratch()
        agent = OllamaCodeAgent(
            client=FakeClient([]),
            tools=ToolExecutor(root, approval_mode="auto"),
            model="fake-model",
            debate_enabled=False,
        )
        payload = {
            "model": "fake-model",
            "approval_mode": "auto",
            "workspace_root": root.as_posix(),
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "remember me"},
            ],
            "events": [
                {
                    "type": "tool_result",
                    "result": {
                        "rows": [{"index": index} for index in range(TRANSCRIPT_DIAGNOSTIC_COLLECTION_LIMIT + 2)],
                        "mapping": {f"key_{index}": index for index in range(TRANSCRIPT_DIAGNOSTIC_COLLECTION_LIMIT + 4)},
                    },
                }
            ],
        }

        agent.restore_transcript(payload)
        result = agent.events[0]["result"]

        self.assertEqual(len(result["rows"]), TRANSCRIPT_DIAGNOSTIC_COLLECTION_LIMIT + 1)
        self.assertEqual(result["rows"][-1], "[truncated 2 items for transcript]")
        self.assertEqual(result["mapping"][TRANSCRIPT_DIAGNOSTIC_DICT_MARKER_KEY], "[truncated 4 entries for transcript]")

    def test_todos_are_saved_and_restored_with_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            session = root / "scratch" / "session.json"
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", session_file=session, debate_enabled=False)
            tools.execute(
                "todo_write",
                {"items": [{"content": "Inspect source", "status": "completed"}, {"content": "Run tests", "status": "pending"}]},
            )
            saved = agent.save_transcript()
            payload = json.loads(saved.read_text(encoding="utf-8"))

            restored_tools = ToolExecutor(root, approval_mode="auto")
            restored = OllamaCodeAgent(client=FakeClient([]), tools=restored_tools, model="fake-model", debate_enabled=False)
            restored.restore_transcript(payload)
            read = restored.todo_read()

        self.assertEqual(payload["todos"][0]["content"], "Inspect source")
        self.assertIn("[pending] Run tests", str(read["output"]))

    def test_agent_load_session_restores_runtime_settings_without_rewriting_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            session = root / ".ollama-code" / "sessions" / "saved.json"
            session.parent.mkdir(parents=True)
            original = (
                '{"model":"saved-model","approval_mode":"read-only","reconcile_mode":"on","workspace_root":"'
                + root.as_posix()
                + '","messages":[{"role":"system","content":"sys"},{"role":"user","content":"restore me"}],"events":[{"type":"user","content":"restore me"}]}'
            )
            session.write_text(original, encoding="utf-8")
            client = FakeClient([])
            tools = ToolExecutor(root, approval_mode="auto")
            agent = OllamaCodeAgent(client=client, tools=tools, model="current-model", session_file="scratch/current.json", debate_enabled=False)

            loaded = agent.load_session(session)
            on_disk = session.read_text(encoding="utf-8")

        self.assertEqual(loaded, session.resolve())
        self.assertEqual(agent.model, "saved-model")
        self.assertEqual(agent.approval_mode(), "read-only")
        self.assertEqual(agent.reconcile_mode(), "on")
        self.assertEqual(on_disk, original)

    def test_workspace_roots_match_accepts_wsl_alias_for_windows_path(self) -> None:
        if Path.cwd().drive:
            saved_root = "/mnt/c/Users/yorke/OneDrive/Desktop/ollama-interactive"
            current_root = Path("C:/Users/yorke/OneDrive/Desktop/ollama-interactive")
        else:
            saved_root = "C:/Users/yorke/OneDrive/Desktop/ollama-interactive"
            current_root = Path("/mnt/c/Users/yorke/OneDrive/Desktop/ollama-interactive")
        self.assertTrue(_workspace_roots_match(saved_root, current_root))

    def test_workspace_roots_match_rejects_different_workspace(self) -> None:
        current_root = Path(__file__).resolve().parents[1]
        self.assertFalse(_workspace_roots_match(str(current_root.parent / "other-workspace"), current_root))


if __name__ == "__main__":
    unittest.main()
