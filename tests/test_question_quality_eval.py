import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from scripts import question_quality_eval as question_eval


def assert_legacy_output_contract(
    testcase: unittest.TestCase,
    module: Any,
    legacy_output: Path,
    *,
    parser: bool = False,
    resolve: bool = False,
) -> None:
    if not parser and not resolve:
        raise ValueError("at least one legacy output contract check must be requested")

    if parser:
        args = module._build_parser().parse_args(["--output", str(legacy_output)])
        testcase.assertEqual(args.output, legacy_output)

    if resolve:
        output_json, output_md = module._resolve_output_paths(
            output=legacy_output,
            output_json=None,
            output_md=None,
        )
        testcase.assertEqual(output_json, legacy_output)
        testcase.assertEqual(output_md, legacy_output.with_suffix(".md"))


class QuestionQualityEvalTests(unittest.TestCase):
    def test_match_answer_to_choices_prefers_exact_match_over_overlapping_subset(self) -> None:
        matches = question_eval.match_answer_to_choices(
            "benchmark pass rate",
            ["benchmark pass rate", "pass rate", "wall-clock latency"],
        )

        self.assertEqual(matches, ["benchmark pass rate"])

    def test_build_parser_accepts_legacy_output_flag(self) -> None:
        assert_legacy_output_contract(
            self,
            question_eval,
            Path("scratch/question-eval.json"),
            parser=True,
        )

    def test_resolve_output_paths_from_legacy_output_json_path(self) -> None:
        assert_legacy_output_contract(
            self,
            question_eval,
            Path("scratch/question-eval.json"),
            resolve=True,
        )

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
