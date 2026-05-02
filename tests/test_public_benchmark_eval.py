from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import public_benchmark_eval as public_bench


class PublicBenchmarkEvalTests(unittest.TestCase):
    def test_python_exercism_test_cmd_uses_exercism_pattern(self) -> None:
        command = public_bench.python_exercism_test_cmd()

        self.assertIn("unittest", command)
        self.assertIn("*_test.py", command)

    def test_polyglot_task_path_validates_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = root / "python" / "exercises" / "practice" / "wordy"
            task.mkdir(parents=True)

            self.assertEqual(public_bench.polyglot_task_path(root, "wordy"), task)
            with self.assertRaises(ValueError):
                public_bench.polyglot_task_path(root, "missing")

    def test_summarize_counts_runs_tokens_and_failures(self) -> None:
        results = [
            {"status": "pass", "usage": {"llm_calls": 2, "total_tokens": 10}},
            {"status": "fail", "usage": {"llm_calls": 3, "total_tokens": 15}},
        ]

        summary = public_bench.summarize(results)

        self.assertEqual(summary["runs"], 2)
        self.assertEqual(summary["pass"], 1)
        self.assertEqual(summary["fail"], 1)
        self.assertEqual(summary["total_llm_calls"], 5)
        self.assertEqual(summary["total_tokens"], 25)

    def test_write_payload_marks_partial_and_comparisons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"

            public_bench.write_payload(
                output,
                results=[{"status": "pass", "usage": {"llm_calls": 1, "total_tokens": 2}}],
                comparisons=[{"case": "wordy"}],
                partial=True,
            )

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(payload["partial"])
            self.assertEqual(payload["summary"]["total_tokens"], 2)
            self.assertEqual(payload["comparisons"], [{"case": "wordy"}])


if __name__ == "__main__":
    unittest.main()
