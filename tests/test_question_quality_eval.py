import json
import tempfile
import unittest
from pathlib import Path

from scripts import question_quality_eval as question_eval


class QuestionQualityEvalTests(unittest.TestCase):
    def test_match_answer_to_choices_returns_unique_choice(self) -> None:
        matches = question_eval.match_answer_to_choices(
            "Optimize model/tool token cost first.",
            ["benchmark pass rate", "wall-clock latency", "model/tool token cost", "first-use UX"],
        )

        self.assertEqual(matches, ["model/tool token cost"])

    def test_default_cases_cover_both_asking_and_restraint(self) -> None:
        cases = question_eval.default_cases()

        self.assertTrue(any(case.expected_verdict == "ask" for case in cases))
        self.assertTrue(any(case.expected_verdict == "proceed" for case in cases))

    def test_default_question_quality_cases_pass(self) -> None:
        payload = question_eval.evaluate_cases()

        self.assertEqual(payload["summary"]["failed"], 0)
        self.assertGreaterEqual(payload["summary"]["passed"], 1)
        by_name = {case["name"]: case for case in payload["cases"]}
        self.assertTrue(by_name["broad_efficiency_rewrite_fallback"]["questions"][0]["eba_style"])
        self.assertEqual(by_name["focused_exact_edit_proceeds"]["actual_verdict"], "proceed")

    def test_main_writes_eval_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            output_json = output_dir / "clarification-question-eval.json"
            output_md = output_dir / "clarification-question-eval.md"

            exit_code = question_eval.main(["--output-json", str(output_json), "--output-md", str(output_md)])

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_json.exists())
            self.assertTrue(output_md.exists())
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["failed"], 0)
            self.assertIn("Clarification Question Quality Eval", output_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
