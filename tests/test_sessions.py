from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from ollama_code.sessions import latest_restorable_session, list_sessions, latest_session_path, load_transcript_payload, resolve_transcript_path


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

    @unittest.skipUnless(os.name != "nt", "POSIX only")
    def test_resolve_transcript_path_normalizes_backslash_relative_path_on_posix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            resolved = resolve_transcript_path(root, r".ollama-code\sessions\saved.json")

        self.assertEqual(resolved, (root / ".ollama-code" / "sessions" / "saved.json").resolve())

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

    def test_load_transcript_payload_reports_unreadable_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            session = root / ".ollama-code" / "sessions" / "denied.json"
            session.parent.mkdir(parents=True, exist_ok=True)
            self._write_session(session, "blocked")
            original_read_text = Path.read_text

            def denied_read_text(target: Path, *args: object, **kwargs: object) -> str:
                if target.resolve(strict=False) == session.resolve(strict=False):
                    raise PermissionError("denied")
                return original_read_text(target, *args, **kwargs)

            with patch.object(Path, "read_text", autospec=True, side_effect=denied_read_text):
                with self.assertRaisesRegex(ValueError, "Unable to read transcript file"):
                    load_transcript_payload(session)

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

    def test_latest_session_path_skips_newer_unreadable_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            session_dir = root / ".ollama-code" / "sessions"
            older = session_dir / "older.json"
            newer = session_dir / "newer.json"
            self._write_session(older, "resume me")
            self._write_session(newer, "blocked")
            older_ts = newer.stat().st_mtime - 10
            os.utime(older, (older_ts, older_ts))
            original_read_text = Path.read_text

            def denied_read_text(target: Path, *args: object, **kwargs: object) -> str:
                if target.resolve(strict=False) == newer.resolve(strict=False):
                    raise PermissionError("denied")
                return original_read_text(target, *args, **kwargs)

            with patch.object(Path, "read_text", autospec=True, side_effect=denied_read_text):
                latest = latest_session_path(root)

        self.assertEqual(latest, older.resolve())

    def test_latest_session_path_skips_newer_session_missing_message_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            session_dir = root / ".ollama-code" / "sessions"
            older = session_dir / "older.json"
            newer = session_dir / "newer.json"
            self._write_session(older, "resume me")
            newer.parent.mkdir(parents=True, exist_ok=True)
            newer.write_text(
                '{"model":"bad-model","approval_mode":"auto","workspace_root":"'
                + root.as_posix()
                + '","messages":[],"events":[]}',
                encoding="utf-8",
            )
            older_ts = newer.stat().st_mtime - 10
            os.utime(older, (older_ts, older_ts))

            latest = latest_session_path(root)

        self.assertEqual(latest, older.resolve())

    def test_latest_session_path_skips_newer_system_only_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            session_dir = root / ".ollama-code" / "sessions"
            older = session_dir / "older.json"
            newer = session_dir / "newer.json"
            self._write_session(older, "resume me")
            newer.parent.mkdir(parents=True, exist_ok=True)
            newer.write_text(
                '{"model":"blank-model","approval_mode":"auto","workspace_root":"'
                + root.as_posix()
                + '","messages":[{"role":"system","content":"sys"}],"events":[]}',
                encoding="utf-8",
            )
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

    def test_list_sessions_skips_unreadable_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            session_dir = root / ".ollama-code" / "sessions"
            kept = session_dir / "kept.json"
            denied = session_dir / "denied.json"
            self._write_session(kept, "keep me")
            self._write_session(denied, "blocked")
            original_read_text = Path.read_text

            def denied_read_text(target: Path, *args: object, **kwargs: object) -> str:
                if target.resolve(strict=False) == denied.resolve(strict=False):
                    raise PermissionError("denied")
                return original_read_text(target, *args, **kwargs)

            with patch.object(Path, "read_text", autospec=True, side_effect=denied_read_text):
                sessions = list_sessions(root)

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].path, kept.resolve())

    def test_list_sessions_skips_system_only_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            session_dir = root / ".ollama-code" / "sessions"
            kept = session_dir / "kept.json"
            blank = session_dir / "blank.json"
            self._write_session(kept, "keep me")
            blank.write_text(
                '{"model":"blank-model","approval_mode":"auto","workspace_root":"'
                + root.as_posix()
                + '","messages":[{"role":"system","content":"sys"}],"events":[]}',
                encoding="utf-8",
            )

            sessions = list_sessions(root)

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].path, kept.resolve())

    def test_list_sessions_skips_session_missing_message_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            session_dir = root / ".ollama-code" / "sessions"
            kept = session_dir / "kept.json"
            invalid = session_dir / "invalid.json"
            self._write_session(kept, "keep me")
            invalid.write_text(
                '{"model":"bad-model","approval_mode":"auto","workspace_root":"'
                + root.as_posix()
                + '","messages":[],"events":[]}',
                encoding="utf-8",
            )

            sessions = list_sessions(root)

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].path, kept.resolve())
        self.assertEqual(sessions[0].summary, "keep me")

    def test_list_sessions_limit_counts_valid_sessions_after_skipping_newer_invalid_ones(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            session_dir = root / ".ollama-code" / "sessions"
            valid = session_dir / "valid.json"
            newer_invalid = session_dir / "newer-invalid.json"
            newest_invalid = session_dir / "newest-invalid.json"
            self._write_session(valid, "keep me")
            newer_invalid.parent.mkdir(parents=True, exist_ok=True)
            newer_invalid.write_text("{not json", encoding="utf-8")
            newest_invalid.write_text("{still not json", encoding="utf-8")
            valid_ts = newer_invalid.stat().st_mtime - 10
            os.utime(valid, (valid_ts, valid_ts))
            newer_invalid_ts = newest_invalid.stat().st_mtime - 5
            os.utime(newer_invalid, (newer_invalid_ts, newer_invalid_ts))

            sessions = list_sessions(root, limit=1)

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].path, valid.resolve())
        self.assertEqual(sessions[0].summary, "keep me")

    def test_list_sessions_summary_uses_latest_user_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            session = root / ".ollama-code" / "sessions" / "thread.json"
            session.parent.mkdir(parents=True, exist_ok=True)
            session.write_text(
                '{"model":"fake-model","approval_mode":"auto","workspace_root":"'
                + root.as_posix()
                + '","messages":[{"role":"system","content":"sys"},{"role":"user","content":"first prompt"},{"role":"assistant","content":"working"},{"role":"user","content":"latest issue to resume"}],"events":[]}',
                encoding="utf-8",
            )

            sessions = list_sessions(root)

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].summary, "latest issue to resume")

    def test_list_sessions_summary_falls_back_to_latest_user_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            session = root / ".ollama-code" / "sessions" / "events-only.json"
            session.parent.mkdir(parents=True, exist_ok=True)
            session.write_text(
                '{"model":"fake-model","approval_mode":"auto","workspace_root":"'
                + root.as_posix()
                + '","messages":[{"role":"system","content":"sys"},{"role":"assistant","content":"still here"}],"events":[{"type":"user","content":"first event"},{"type":"user","content":"latest event"}]}',
                encoding="utf-8",
            )

            sessions = list_sessions(root)

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].summary, "latest event")

    def test_latest_session_path_skips_session_that_disappears_during_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            session_dir = root / ".ollama-code" / "sessions"
            kept = session_dir / "kept.json"
            vanished = session_dir / "vanished.json"
            self._write_session(kept, "keep me")
            self._write_session(vanished, "gone soon")
            older_ts = vanished.stat().st_mtime - 10
            os.utime(kept, (older_ts, older_ts))
            original_is_file = Path.is_file
            original_stat = Path.stat
            vanished_key = str(vanished).lower()

            def patched_is_file(target: Path) -> bool:
                if str(target).lower() == vanished_key:
                    return True
                return original_is_file(target)

            def patched_stat(target: Path, *args: object, **kwargs: object) -> os.stat_result:
                if str(target).lower() == vanished_key:
                    raise FileNotFoundError("vanished")
                return original_stat(target, *args, **kwargs)

            with patch.object(Path, "is_file", autospec=True, side_effect=patched_is_file):
                with patch.object(Path, "stat", autospec=True, side_effect=patched_stat):
                    latest = latest_session_path(root)

        self.assertEqual(latest, kept.resolve())

    def test_latest_restorable_session_returns_payload_for_newest_valid_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            session_dir = root / ".ollama-code" / "sessions"
            older = session_dir / "older.json"
            newer = session_dir / "newer.json"
            self._write_session(older, "older message")
            self._write_session(newer, "newer message")
            older_ts = newer.stat().st_mtime - 10
            os.utime(older, (older_ts, older_ts))

            latest = latest_restorable_session(root)

        assert latest is not None
        path, payload = latest
        self.assertEqual(path, newer.resolve())
        self.assertEqual(payload["messages"][0]["content"], "newer message")
