from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from ollama_code.tools import CODE_FILE_SUFFIXES, FTS_TEXT_SUFFIXES, ToolExecutor


StatusPrinter = Callable[[str], None]


class BackgroundIndexer:
    """Keep the repo-local search indexes warm without involving the model."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        enabled: bool = True,
        watch: bool = True,
        poll_interval_ms: int = 5000,
        max_files: int = 50000,
        code_limit: int = 1000,
        fts_limit: int = 2000,
        status_printer: StatusPrinter | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.enabled = bool(enabled)
        self.watch = bool(watch)
        self.poll_interval = max(0.25, float(max(1, int(poll_interval_ms))) / 1000.0)
        self.max_files = max(1, int(max_files))
        self.code_limit = max(1, int(code_limit))
        self.fts_limit = max(1, int(fts_limit))
        self.status_printer = status_printer or (lambda _message: None)
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._pending_paths: set[str] = set()
        self._full_refresh_requested = False
        self._snapshot: dict[str, tuple[int, int]] = {}
        self._running = False
        self._ready = False
        self._last_error: str | None = None
        self._last_summary = "indexer not started"
        self._last_indexed_at: float | None = None
        self._refresh_count = 0

    def start(self) -> bool:
        if not self.enabled:
            with self._lock:
                self._last_summary = "indexer disabled"
            return False
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            self._stop_event.clear()
            self._wake_event.clear()
            self._running = True
            self._last_summary = "indexer starting"
            self._thread = threading.Thread(target=self._run, name="ollama-code-indexer", daemon=True)
            self._thread.start()
            return True

    def stop(self, *, timeout: float = 2.0) -> None:
        self._stop_event.set()
        self._wake_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        with self._lock:
            self._running = False

    def request_refresh(self, reason: str | None = None) -> None:
        with self._lock:
            self._full_refresh_requested = True
            if reason:
                self._last_summary = f"refresh requested: {reason}"
        self._wake_event.set()

    def notify_paths(self, paths: Iterable[str | Path]) -> None:
        normalized = {self._normalize_path(path) for path in paths}
        normalized.discard(None)
        if not normalized:
            return
        with self._lock:
            self._pending_paths.update(str(path) for path in normalized)
            self._last_summary = f"queued {len(normalized)} changed path(s)"
        self._wake_event.set()

    def refresh_now(self) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "summary": "indexer disabled"}
        tools = self._new_tools()
        return self._refresh_full(tools)

    def status(self) -> dict[str, Any]:
        with self._lock:
            thread_alive = self._thread is not None and self._thread.is_alive()
            return {
                "enabled": self.enabled,
                "watch": self.watch,
                "running": thread_alive,
                "ready": self._ready,
                "poll_interval_ms": int(self.poll_interval * 1000),
                "pending_paths": len(self._pending_paths),
                "refresh_count": self._refresh_count,
                "last_indexed_at": self._last_indexed_at,
                "last_error": self._last_error,
                "summary": self._last_summary,
                "cache_dir": str((self.workspace_root / ".ollama-code" / "index").resolve(strict=False)),
            }

    def _run(self) -> None:
        tools = self._new_tools()
        with self._lock:
            self._running = True
            self._last_summary = "initial index refresh running"
        self._safe_call(lambda: self._refresh_full(tools))
        self._snapshot = self._collect_snapshot(tools)
        while not self._stop_event.is_set():
            self._wake_event.wait(self.poll_interval)
            self._wake_event.clear()
            if self._stop_event.is_set():
                break
            pending, full = self._drain_pending()
            if self.watch:
                pending.update(self._scan_changed_paths(tools))
            if full or len(pending) > 200:
                self._safe_call(lambda: self._refresh_full(tools))
                self._snapshot = self._collect_snapshot(tools)
            elif pending:
                self._safe_call(lambda: self._refresh_paths(tools, pending))
        with self._lock:
            self._running = False
            if self._last_summary != "indexer disabled":
                self._last_summary = "indexer stopped"

    def _new_tools(self) -> ToolExecutor:
        return ToolExecutor(self.workspace_root, approval_mode="read-only")

    def _drain_pending(self) -> tuple[set[str], bool]:
        with self._lock:
            pending = set(self._pending_paths)
            self._pending_paths.clear()
            full = self._full_refresh_requested
            self._full_refresh_requested = False
        return pending, full

    def _safe_call(self, callback: Callable[[], dict[str, Any] | None]) -> None:
        try:
            result = callback()
        except Exception as exc:  # pragma: no cover - defensive background guard
            with self._lock:
                self._last_error = str(exc)
                self._last_summary = f"index refresh failed: {exc}"
            return
        if isinstance(result, dict) and result.get("ok") is False:
            with self._lock:
                self._last_error = str(result.get("summary") or result)
                self._last_summary = self._last_error

    def _refresh_full(self, tools: ToolExecutor) -> dict[str, Any]:
        file_result = tools.file_index_refresh(".", limit=self.max_files)
        repo_result = tools.repo_index_refresh(".", limit=self.code_limit)
        fts_result = tools.fts_refresh(".", limit=self.fts_limit)
        ok = all(result.get("ok") is True for result in (file_result, repo_result, fts_result))
        summary = (
            f"indexed files={file_result.get('files', 0)} "
            f"code={repo_result.get('files', 0)} "
            f"fts={fts_result.get('files', 0)}"
        )
        with self._lock:
            self._ready = ok
            self._last_error = None if ok else str(fts_result.get("summary") or repo_result.get("summary") or file_result.get("summary"))
            self._last_summary = summary if ok else self._last_error or summary
            self._last_indexed_at = time.time()
            self._refresh_count += 1
        return {"ok": ok, "summary": summary, "file_index": file_result, "repo_index": repo_result, "fts": fts_result}

    def _refresh_paths(self, tools: ToolExecutor, paths: Iterable[str]) -> dict[str, Any]:
        updated = 0
        needs_full = False
        for rel in sorted(set(paths)):
            target = self.workspace_root / rel
            if not target.exists():
                needs_full = True
                continue
            if target.is_dir():
                needs_full = True
                continue
            tools.file_index_refresh(rel, limit=1)
            suffix = target.suffix.lower()
            if suffix in CODE_FILE_SUFFIXES:
                tools.repo_index_refresh(rel, limit=1)
            if suffix in FTS_TEXT_SUFFIXES:
                tools.fts_refresh(rel, limit=1)
            updated += 1
        if needs_full:
            return self._refresh_full(tools)
        with self._lock:
            self._ready = True
            self._last_error = None
            self._last_summary = f"indexed {updated} changed path(s)"
            self._last_indexed_at = time.time()
            self._refresh_count += 1
        return {"ok": True, "summary": f"indexed {updated} changed path(s)", "files": updated}

    def _collect_snapshot(self, tools: ToolExecutor) -> dict[str, tuple[int, int]]:
        snapshot: dict[str, tuple[int, int]] = {}
        for file_path in tools._iter_workspace_files(self.workspace_root, limit=self.max_files):
            try:
                stat = file_path.stat()
            except OSError:
                continue
            snapshot[tools.relative_label(file_path)] = (int(stat.st_mtime_ns), int(stat.st_size))
        return snapshot

    def _scan_changed_paths(self, tools: ToolExecutor) -> set[str]:
        current = self._collect_snapshot(tools)
        changed = {
            rel
            for rel, state in current.items()
            if self._snapshot.get(rel) != state
        }
        changed.update(rel for rel in self._snapshot if rel not in current)
        self._snapshot = current
        return changed

    def _normalize_path(self, path: str | Path) -> str | None:
        try:
            candidate = Path(path)
            if not candidate.is_absolute():
                candidate = self.workspace_root / candidate
            resolved = candidate.resolve(strict=False)
            if resolved != self.workspace_root and self.workspace_root not in resolved.parents:
                return None
            return resolved.relative_to(self.workspace_root).as_posix()
        except (OSError, ValueError):
            return None
