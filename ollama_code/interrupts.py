from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
from collections.abc import Iterator
from typing import Callable


class OperationInterrupted(RuntimeError):
    pass


class InterruptController:
    def __init__(self, writer: Callable[[str], None] | None = None) -> None:
        self.writer = writer or (lambda _: None)

    @contextlib.contextmanager
    def watch(self) -> Iterator[threading.Event]:
        interrupted = threading.Event()
        stop_event = threading.Event()
        watcher = self._build_watcher(interrupted, stop_event)
        if watcher is None:
            yield interrupted
            return
        watcher.start()
        try:
            yield interrupted
        finally:
            stop_event.set()
            watcher.join(timeout=1)

    def _build_watcher(self, interrupted: threading.Event, stop_event: threading.Event) -> threading.Thread | None:
        if not sys.stdin or not sys.stdin.isatty():
            return None
        if os.name == "nt":
            return threading.Thread(target=self._watch_windows, args=(interrupted, stop_event), daemon=True)
        return threading.Thread(target=self._watch_posix, args=(interrupted, stop_event), daemon=True)

    def _watch_windows(self, interrupted: threading.Event, stop_event: threading.Event) -> None:
        import msvcrt

        while not stop_event.is_set() and not interrupted.is_set():
            while msvcrt.kbhit():
                if msvcrt.getwch() == "\x1b":
                    interrupted.set()
                    self.writer("[status] interrupt requested")
                    return
            time.sleep(0.05)

    def _watch_posix(self, interrupted: threading.Event, stop_event: threading.Event) -> None:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        original = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not stop_event.is_set() and not interrupted.is_set():
                ready, _, _ = select.select([fd], [], [], 0.05)
                if not ready:
                    continue
                if os.read(fd, 1) == b"\x1b":
                    interrupted.set()
                    self.writer("[status] interrupt requested")
                    return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, original)
