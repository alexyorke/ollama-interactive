from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from uuid import uuid4


_TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "scratch" / "test-temp"


class _WorkspaceTemporaryDirectory:
    def __init__(
        self,
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | None = None,
        ignore_cleanup_errors: bool = False,
        *,
        delete: bool = True,
    ) -> None:
        base = _TEST_TEMP_ROOT
        base.mkdir(parents=True, exist_ok=True)
        name = f"{prefix or 'tmp'}{uuid4().hex}{suffix or ''}"
        self.name = str((base / name).resolve())
        Path(self.name).mkdir(parents=True, exist_ok=False)
        self._ignore_cleanup_errors = ignore_cleanup_errors
        self._delete = delete

    def cleanup(self) -> None:
        if self._delete:
            shutil.rmtree(self.name, ignore_errors=True)

    def __enter__(self) -> str:
        return self.name

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.cleanup()


tempfile.TemporaryDirectory = _WorkspaceTemporaryDirectory
