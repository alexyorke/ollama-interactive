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

    def test_public_task_prompt_requires_complete_implementation_loop(self) -> None:
        prompt = public_bench.public_task_prompt("Python")

        self.assertIn("edit only implementation files", prompt)
        self.assertIn("do not edit tests", prompt)
        self.assertIn("keep editing until tests pass", prompt)

    def test_public_task_prompt_is_task_agnostic(self) -> None:
        prompt = public_bench.public_task_prompt("Python").lower()

        for task_name in public_bench.DEFAULT_POLYGLOT_TASKS:
            self.assertNotIn(task_name, prompt)
        for solution_hint in ("foldr", "pig latin", "question", "return 99", "token_"):
            self.assertNotIn(solution_hint, prompt)

    def test_polyglot_task_path_validates_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = root / "python" / "exercises" / "practice" / "wordy"
            task.mkdir(parents=True)

            self.assertEqual(public_bench.polyglot_task_path(root, "wordy"), task)
        with self.assertRaises(ValueError):
            public_bench.polyglot_task_path(root, "missing")

    def test_failure_classes_identifies_public_failure_shapes(self) -> None:
        classes = public_bench.failure_classes(
            status="fail",
            timed_out=False,
            final_tests_returncode=1,
            calls=["read_file", "run_test"],
            failures=[
                {"name": "replace_body", "summary": "Unknown tool: replace_body"},
                {"name": "run_test", "summary": "Command rejected before execution: path escapes workspace: ."},
                {"name": "edit_intent", "summary": "Python syntax error: SyntaxError: unterminated string literal"},
            ],
        )

        self.assertIn("invalid_tool", classes)
        self.assertIn("invalid_args", classes)
        self.assertIn("syntax_edit", classes)
        self.assertIn("no_edit_attempted", classes)
        self.assertIn("tests_still_failing", classes)

    def test_source_snapshots_exclude_tests_and_report_diffs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            workspace = root / "workspace"
            source.mkdir()
            workspace.mkdir()
            (source / "exercise.py").write_text("def answer():\n    pass\n", encoding="utf-8")
            (workspace / "exercise.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
            (workspace / "exercise_test.py").write_text("assert answer() == 42\n", encoding="utf-8")

            diffs = public_bench._changed_source_diffs(source, workspace)
            snippets = public_bench._final_source_snippets(workspace)

        self.assertEqual(diffs[0]["path"], "exercise.py")
        self.assertIn("return 42", diffs[0]["diff"])
        self.assertEqual([item["path"] for item in snippets], ["exercise.py"])

    def test_public_benchmark_detaches_meta_from_model_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            (workspace / ".meta").mkdir()
            (workspace / ".meta" / "example.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
            (workspace / "exercise.py").write_text("def answer():\n    pass\n", encoding="utf-8")

            detached = public_bench._detach_model_visible_meta(workspace)
            snippets = public_bench._evaluator_meta_snippets(detached)
            visible = public_bench._final_source_snippets(workspace)

        self.assertIsNotNone(detached)
        self.assertFalse((workspace / ".meta").exists())
        self.assertEqual(snippets[0]["path"], ".meta/example.py")
        self.assertEqual([item["path"] for item in visible], ["exercise.py"])

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
