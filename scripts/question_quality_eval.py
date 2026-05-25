from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ollama_code.agent import OllamaCodeAgent
from ollama_code.tools import ToolExecutor


DEFAULT_OUTPUT_DIR = Path("scratch") / "question-quality"
DEFAULT_JSON_OUTPUT = DEFAULT_OUTPUT_DIR / "clarification-question-eval.json"
DEFAULT_MARKDOWN_OUTPUT = DEFAULT_OUTPUT_DIR / "clarification-question-eval.md"


class NullClient:
    def set_interrupt_event(self, event: object | None) -> None:
        return None

    def list_models(self) -> list[str]:
        return ["eval-model"]

    def chat(self, **_: Any) -> Any:
        raise RuntimeError("question_quality_eval does not call the live model")


@dataclass(frozen=True)
class ClarificationEvalCase:
    name: str
    request_text: str
    planner_payload: dict[str, Any]
    expected_verdict: str = "ask"
    expected_aspects: tuple[str, ...] = ()
    pretend_answers: tuple[str, ...] = ()
    min_quality_score: int = 6


def _build_agent(workspace: Path) -> OllamaCodeAgent:
    tools = ToolExecutor(workspace, approval_mode="auto")
    return OllamaCodeAgent(client=NullClient(), tools=tools, model="eval-model", debate_enabled=False)


def _normalize_text(text: str) -> str:
    return "".join(char.lower() if char.isalnum() else " " for char in text).strip()


def match_answer_to_choices(answer: str, choices: list[str]) -> list[str]:
    normalized_answer = _normalize_text(answer)
    answer_tokens = {token for token in normalized_answer.split() if token}
    matches: list[str] = []
    for choice in choices:
        normalized_choice = _normalize_text(choice)
        choice_tokens = {token for token in normalized_choice.split() if token}
        if not normalized_choice or not choice_tokens:
            continue
        if (
            normalized_answer == normalized_choice
            or normalized_answer in normalized_choice
            or normalized_choice in normalized_answer
            or choice_tokens.issubset(answer_tokens)
        ):
            matches.append(choice)
    return matches


def default_cases() -> tuple[ClarificationEvalCase, ...]:
    return (
        ClarificationEvalCase(
            name="broad_efficiency_rewrite_fallback",
            request_text=(
                "Rewrite large chunks of the application so small local models are much more efficient, "
                "and make the benchmark and UX story strong enough to sell."
            ),
            planner_payload={
                "verdict": "proceed",
                "reason": "The request is broad and leaves the optimization priority unspecified.",
                "ambiguities": [],
                "questions": [],
            },
            expected_aspects=("tradeoff", "acceptance"),
            pretend_answers=("Prioritize benchmark pass rate first.", "Optimize model/tool token cost first."),
        ),
        ClarificationEvalCase(
            name="architecture_boundary_fallback",
            request_text="Refactor the architecture heavily, but do not break the surfaces that matter most.",
            planner_payload={
                "verdict": "ask",
                "reason": "The rewrite boundary is ambiguous and compatibility surface is unspecified.",
                "ambiguities": [],
                "questions": [],
            },
            expected_aspects=("scope", "compatibility"),
            pretend_answers=("Keep the CLI surface fixed.", "Tool contracts matter most."),
        ),
        ClarificationEvalCase(
            name="checkout_path_direct_payload",
            request_text="Improve the checkout workflow and fix the most important issue.",
            planner_payload={
                "verdict": "ask",
                "reason": "The request names a checkout workflow, but evidence does not identify which user path defines success.",
                "ambiguities": [
                    {
                        "kind": "acceptance",
                        "detail": "Guest checkout and subscription renewal could require different edits.",
                        "evidence": "context_pack found checkout.py only.",
                    }
                ],
                "questions": [
                    {
                        "question": "Which checkout path should define the fix: guest checkout or subscription renewal?",
                        "why_it_matters": "The target behavior and acceptance tests differ by path.",
                        "recommended_default": "Use guest checkout because it is the broadest first-use path.",
                        "choices": ["guest checkout", "subscription renewal"],
                    }
                ],
            },
            expected_aspects=("workflow", "acceptance"),
            pretend_answers=("Use guest checkout.", "Subscription renewal is the real blocker."),
        ),
        ClarificationEvalCase(
            name="low_value_question_replaced",
            request_text="Refactor the architecture to improve efficiency, but keep the important external behavior stable.",
            planner_payload={
                "verdict": "ask",
                "reason": "The request is broad and asks for a rewrite.",
                "ambiguities": [],
                "questions": [
                    {
                        "question": "Should I proceed?",
                        "why_it_matters": "I need permission first.",
                        "recommended_default": "",
                        "choices": ["yes", "no"],
                    }
                ],
            },
            expected_aspects=("scope", "compatibility"),
            pretend_answers=("Keep benchmark comparability fixed.",),
        ),
        ClarificationEvalCase(
            name="focused_exact_edit_proceeds",
            request_text="Fix app.py by changing old to new.",
            planner_payload={
                "verdict": "proceed",
                "reason": "The request already specifies the file, edit target, and acceptance direction.",
                "ambiguities": [],
                "questions": [],
            },
            expected_verdict="proceed",
            min_quality_score=0,
        ),
    )


def evaluate_case(agent: OllamaCodeAgent, case: ClarificationEvalCase) -> dict[str, Any]:
    decision = agent._normalize_question_planner_payload(case.planner_payload, request_text=case.request_text)
    questions = decision.get("questions") if isinstance(decision.get("questions"), list) else []
    evaluated_questions: list[dict[str, Any]] = []
    for item in questions:
        if not isinstance(item, dict):
            continue
        metrics = agent._question_quality_metrics(item, request_text=case.request_text)
        choices = list(metrics.get("normalized_choices") or item.get("choices") or [])
        answer_checks = []
        for answer in case.pretend_answers:
            matches = match_answer_to_choices(answer, choices)
            answer_checks.append(
                {
                    "answer": answer,
                    "matches": matches,
                    "unique_match": len(matches) == 1,
                }
            )
        matched_aspects = [aspect for aspect in case.expected_aspects if aspect in set(metrics.get("aspect_tags") or [])]
        evaluated_questions.append(
            {
                "question": str(item.get("question") or ""),
                "why_it_matters": str(item.get("why_it_matters") or ""),
                "recommended_default": str(item.get("recommended_default") or ""),
                "choices": choices,
                "quality_score": int(metrics.get("quality_score", 0)),
                "eba_style": bool(metrics.get("eba_style")),
                "aspect_tags": list(metrics.get("aspect_tags") or []),
                "pretend_answer_checks": answer_checks,
                "matched_expected_aspects": matched_aspects,
            }
        )
    overall_pass = decision.get("verdict") == case.expected_verdict
    if case.expected_verdict == "ask":
        overall_pass = overall_pass and bool(evaluated_questions)
        if evaluated_questions:
            first = evaluated_questions[0]
            overall_pass = (
                overall_pass
                and first["quality_score"] >= case.min_quality_score
                and bool(first["eba_style"])
                and 2 <= len(first["choices"]) <= 4
                and bool(first["recommended_default"])
            )
            if case.expected_aspects:
                overall_pass = overall_pass and bool(first["matched_expected_aspects"])
            if case.pretend_answers:
                overall_pass = overall_pass and all(check["unique_match"] for check in first["pretend_answer_checks"])
    return {
        "name": case.name,
        "request_text": case.request_text,
        "expected_verdict": case.expected_verdict,
        "actual_verdict": decision.get("verdict"),
        "reason": decision.get("reason", ""),
        "overall_pass": overall_pass,
        "questions": evaluated_questions,
        "formatted_message": agent._format_clarifying_questions(decision) if decision.get("verdict") == "ask" else "",
    }


def evaluate_cases(cases: tuple[ClarificationEvalCase, ...] | None = None) -> dict[str, Any]:
    selected_cases = default_cases() if cases is None else cases
    with tempfile.TemporaryDirectory(prefix="ollama-code-question-eval-") as tmp:
        agent = _build_agent(Path(tmp))
        results = [evaluate_case(agent, case) for case in selected_cases]
    passed = sum(1 for item in results if item["overall_pass"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases": results,
        "summary": {
            "cases": len(results),
            "passed": passed,
            "failed": len(results) - passed,
        },
    }


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Clarification Question Quality Eval",
        "",
        f"Generated: `{payload.get('generated_at', '')}`",
        "",
        "This eval scores clarification questions for elimination-by-aspect quality using synthetic broad requests and pretend human answers.",
        "",
    ]
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    lines.append(
        f"Summary: cases=`{summary.get('cases', 0)}` passed=`{summary.get('passed', 0)}` failed=`{summary.get('failed', 0)}`"
    )
    lines.append("")
    for case in payload.get("cases", []):
        lines.append(f"## {case.get('name', 'unknown')}")
        lines.append("")
        lines.append(f"- Overall pass: `{case.get('overall_pass')}`")
        lines.append(f"- Expected verdict: `{case.get('expected_verdict')}`")
        lines.append(f"- Actual verdict: `{case.get('actual_verdict')}`")
        lines.append(f"- Request: {case.get('request_text')}")
        if case.get("reason"):
            lines.append(f"- Reason: {case.get('reason')}")
        questions = case.get("questions") if isinstance(case.get("questions"), list) else []
        if not questions:
            lines.append("- Questions: `(none)`")
            lines.append("")
            continue
        first = questions[0]
        lines.append(f"- Top question: {first.get('question')}")
        lines.append(f"- Quality score: `{first.get('quality_score')}`")
        lines.append(f"- EBA style: `{first.get('eba_style')}`")
        lines.append(f"- Aspect tags: `{', '.join(first.get('aspect_tags') or [])}`")
        lines.append(f"- Choices: `{ ' | '.join(first.get('choices') or []) }`")
        lines.append(f"- Recommended default: {first.get('recommended_default')}")
        checks = first.get("pretend_answer_checks") if isinstance(first.get("pretend_answer_checks"), list) else []
        for check in checks:
            lines.append(
                f"- Pretend answer: `{check.get('answer')}` -> `{', '.join(check.get('matches') or []) or 'no match'}`"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate clarification question quality for EBA-style prompts using synthetic cases.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    args = parser.parse_args(argv)

    payload = evaluate_cases()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown = format_markdown(payload)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(f"Wrote JSON: {args.output_json}")
    print(f"Wrote Markdown: {args.output_md}")
    print(
        f"clarification-question-eval: cases={payload['summary']['cases']} passed={payload['summary']['passed']} failed={payload['summary']['failed']}"
    )
    return 0 if int(payload["summary"]["failed"]) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
