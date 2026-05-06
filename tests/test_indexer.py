from __future__ import annotations

import shutil
import tempfile
import time
import unittest
from pathlib import Path
from uuid import uuid4

from ollama_code.indexer import BackgroundIndexer
from ollama_code.tools import ToolExecutor


class BackgroundIndexerTests(unittest.TestCase):
    def _workspace_scratch(self) -> Path:
        root = (Path.cwd() / "verify_scratch" / f"test-indexer-{uuid4().hex}").resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return root

    def _wait_for(self, predicate: object, *, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        callback = predicate
        while time.monotonic() < deadline:
            if callable(callback) and callback():
                return
            time.sleep(0.05)
        self.fail("condition did not become true before timeout")

    def test_refresh_now_warms_all_search_indexes(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "src" / "app.py").write_text("def alpha_token():\n    return 'warm'\n", encoding="utf-8")
        (root / "README.md").write_text("# Alpha Token\n", encoding="utf-8")

        indexer = BackgroundIndexer(root, enabled=True, watch=False)
        result = indexer.refresh_now()
        tools = ToolExecutor(root, approval_mode="auto")

        self.assertTrue(result["ok"])
        self.assertIn("src/app.py", tools.file_search("app")["output"])
        self.assertIn("alpha_token", tools.repo_index_search("alpha_token")["output"])
        self.assertIn("README.md", tools.fts_search("Alpha Token")["output"])
        self.assertIn("alpha_token", tools.verified_function_search("alpha token")["output"])

    def test_background_indexer_updates_changed_path_after_notification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "app.py"
            target.write_text("def alpha_token():\n    return 'one'\n", encoding="utf-8")
            indexer = BackgroundIndexer(root, enabled=True, watch=False, poll_interval_ms=100)
            try:
                self.assertTrue(indexer.start())
                self._wait_for(lambda: bool(indexer.status()["ready"]))
                target.write_text("def beta_token():\n    return 'two'\n", encoding="utf-8")
                indexer.notify_paths(["app.py"])
                tools = ToolExecutor(root, approval_mode="auto")
                self._wait_for(lambda: "beta_token" in str(tools.fts_search("beta_token").get("output", "")))
                self.assertIn("beta_token", tools.repo_index_search("beta_token")["output"])
                self.assertIn("beta_token", tools.verified_function_search("beta_token")["output"])
            finally:
                indexer.stop()

    def test_changed_path_refresh_marks_deleted_verified_function_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "math_utils.py"
            target.write_text(
                'def double(value: int) -> int:\n'
                '    """Double a value.\n\n'
                '    >>> double(3)\n'
                '    6\n'
                '    """\n'
                '    return value * 2\n',
                encoding="utf-8",
            )
            tools = ToolExecutor(root, approval_mode="auto")
            promoted = tools.promote_verified_function("math_utils.py", "double")
            self.assertTrue(promoted["ok"], promoted.get("summary"))

            target.unlink()
            indexer = BackgroundIndexer(root, enabled=True, watch=False)
            result = indexer._refresh_paths(tools, ["math_utils.py"])
            shown = tools.verified_function_show(promoted["id"])

        self.assertTrue(result["ok"], result.get("summary"))
        self.assertTrue(shown["stale"])
        self.assertEqual(shown["card"]["proof_level"], "unverified")


if __name__ == "__main__":
    unittest.main()
