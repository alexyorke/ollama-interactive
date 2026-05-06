from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
import shutil
from pathlib import Path
from uuid import uuid4


@contextmanager
def workspace_temp_dir(prefix: str, root: Path, keep: bool | Callable[[], bool] = False):
    """Create a temporary workspace with inherited permissions.

    Windows sandboxed runs can reject files under stdlib tempfile dirs because
    mkdtemp applies restrictive permissions. Plain mkdir inherits the workspace
    ACLs and still keeps cleanup best-effort.
    """
    root.mkdir(parents=True, exist_ok=True)
    path = (root / f"{prefix}{uuid4().hex}").resolve()
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield str(path)
    finally:
        should_keep = keep() if callable(keep) else bool(keep)
        if not should_keep:
            shutil.rmtree(path, ignore_errors=True)
