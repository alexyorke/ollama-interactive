from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from ollama_code.sessions import list_sessions, latest_session_path, load_transcript_payload, resolve_transcript_path


class SessionPathTests(unittest.TestCase):
    def _cross_platform_workspace_pair(self) -> tuple[Path, str]:
        if Path.cwd().drive:
            return (
                Path("C:/Users/yorke/OneDrive/Desktop/ollama-interactive").resolve(strict=False),
                "/mnt/c/Users/yorke/OneDrive/Desktop/ollama-interactive",
            )
        return (
            Path("/mnt/c/Users/yorke/OneDrive/Desktop/ollama-interactive"),
            "C:/Users/yorke/OneDrive/Desktop/ollama-interactive",
        )

    def _write_session(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"model":"fake-model","approval_mode":"auto","messages":[{"role":"user","content":"'
            + content
            + '"}],"events":[]}',
            encoding="utf-8",
        )

    def _create_symlink_or_skip(self, link: Path, target: Path) -> None:
        try:
            link.symlink_to(target)
        except (NotImplementedError, OSError) as exc:
            self.skipTest(f"file symlinks are not available in this environment: {exc}")

    def test_resolve_transcript_path_keeps_paths_in_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            resolved = resolve_transcript_path(root, "scratch/session.json")

        self.assertEqual(resolved, (root / "scratch" / "session.json").resolve())

    def test_resolve_transcript_path_blocks_relative_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with self.assertRaisesRegex(ValueError, "escapes the workspace"):
                resolve_transcript_path(root, "../outside.json")

    def test_resolve_transcript_path_blocks_absolute_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            outside = root.parent / "outside.json"
            with self.assertRaisesRegex(ValueError, "escapes the workspace"):
                resolve_transcript_path(root, outside)

    def test_resolve_transcript_path_accepts_cross_platform_workspace_alias(self) -> None:
        root, alias_root = self._cross_platform_workspace_pair()
        resolved = resolve_transcript_path(root, f"{alias_root}/.ollama-code/sessions/saved.json")
        self.assertEqual(resolved, (root / ".ollama-code" / "sessions" / "saved.json").resolve(strict=False))

    def test_resolve_transcript_path_blocks_cross_platform_alias_escape(self) -> None:
        root, alias_root = self._cross_platform_workspace_pair()
        escaped_alias = alias_root.rsplit("/", 1)[0] + "/other-workspace/saved.json"
        with self.assertRaisesRegex(ValueError, "escapes the workspace"):
            resolve_transcript_path(root, escaped_alias)

    def test_load_transcript_payload_reports_missing_file(self) -> None:
        missing = Path.cwd() / ".ollama-code" / "sessions" / f"missing-{uuid4().hex}.json"
        with self.assertRaisesRegex(ValueError, "Transcript file not found"):
            load_transcript_payload(missing)

    def test_load_transcript_payload_reports_invalid_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            session = root / ".ollama-code" / "sessions" / "invalid-utf8.json"
            session.parent.mkdir(parents=True, exist_ok=True)
            session.write_bytes(b"\xff\xfe\x80")

            with self.assertRaisesRegex(ValueError, "Invalid transcript encoding"):
                load_transcript_payload(session)

    def test_load_transcript_payload_accepts_utf8_bom(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            session = root / ".ollama-code" / "sessions" / "bom.json"
            session.parent.mkdir(parents=True, exist_ok=True)
            session.write_bytes(
                b"\xef\xbb\xbf"
                + b'{"model":"fake-model","approval_mode":"auto","workspace_root":"'
                + root.as_posix().encode("utf-8")
                + b'","messages":[{"role":"system","content":"sys"}],"events":[]}',
            )

            payload = load_transcript_payload(session)

        self.assertEqual(payload["model"], "fake-model")

    def test_latest_session_path_ignores_symlink_that_resolves_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            session_dir = root / ".ollama-code" / "sessions"
            kept = session_dir / "kept.json"
            escaped = root.parent / "outside-session.json"
            self._write_session(kept, "keep me")
            self._write_session(escaped, "SECRET_TOKEN_99")
            self._create_symlink_or_skip(session_dir / "escaped.json", escaped)
            latest = latest_session_path(root)

        self.assertEqual(latest, kept.resolve())

    def test_latest_session_path_skips_newer_invalid_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            session_dir = root / ".ollama-code" / "sessions"
            older = session_dir / "older.json"
            newer = session_dir / "newer.json"
            self._write_session(older, "resume me")
            newer.parent.mkdir(parents=True, exist_ok=True)
            newer.write_text("{not json", encoding="utf-8")
            older_ts = newer.stat().st_mtime - 10
            os.utime(older, (older_ts, older_ts))
            latest = latest_session_path(root)

        self.assertEqual(latest, older.resolve())

    def test_list_sessions_ignores_symlink_that_resolves_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            session_dir = root / ".ollama-code" / "sessions"
            kept = session_dir / "kept.json"
            escaped = root.parent / "outside-session.json"
            self._write_session(kept, "keep me")
            self._write_session(escaped, "SECRET_TOKEN_99")
            self._create_symlink_or_skip(session_dir / "escaped.json", escaped)
            sessions = list_sessions(root)

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].path, kept.resolve())
        self.assertEqual(sessions[0].summary, "keep me")
