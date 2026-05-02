# Token Efficiency Report

Baseline was measured from clean repo commit `8273c36` with token-metering-only code added before behavior optimizations. Optimized results are from the current post-optimization worktree. Raw JSON lives in ignored `scratch/token-efficiency/`.

Metrics come from Ollama `/api/chat` response fields: `prompt_eval_count`, `eval_count`, `total_duration`, `load_duration`, `prompt_eval_duration`, and `eval_duration`.

## Summary

| Metric | Baseline | Optimized |
|---|---:|---:|
| Serial runs | 70 | 70 |
| Passing runs | 43 | 70 |
| Fail-closed runs | 1 | 0 |
| Failing runs | 26 | 0 |
| Median prompt tokens, passing debate-on runs | 4,333 | 1,364.5 |
| Median prompt-token change, matched passing debate-on runs | - | -74.9% |
| Accuracy regressions | - | 0 |

Second pass adds controller-direct execution for explicit low-ambiguity requests and primary-context compaction for normal model turns. Compared with the first optimized run, the fixed 70-run explicit-tool corpus stayed 70/70 passing and dropped to `0` LLM calls, `0` prompt tokens, and `0` output tokens. These are not model shortcuts for broad coding work; they only fire when the controller can deterministically prove the required tool path and final answer.

## Second-Pass Direct Routing

| Model | Verifier | Debate | First optimized prompt | Second-pass prompt | LLM calls before -> after | Pass before -> after |
|---|---|---|---:|---:|---:|---:|
| `gemma3:4b` | - | off | 12,555 | 0 | 15 -> 0 | 10 -> 10 |
| `gemma3:4b` | - | on | 19,516 | 0 | 25 -> 0 | 10 -> 10 |
| `qwen3:8b` | - | off | 12,104 | 0 | 16 -> 0 | 10 -> 10 |
| `qwen3:8b` | - | on | 16,378 | 0 | 24 -> 0 | 10 -> 10 |
| `granite4.1:8b` | - | off | 11,713 | 0 | 16 -> 0 | 10 -> 10 |
| `granite4.1:8b` | - | on | 14,567 | 0 | 22 -> 0 | 10 -> 10 |
| `gemma3:4b` | `granite4.1:8b` | on | 18,315 | 0 | 24 -> 0 | 10 -> 10 |

Added direct paths:

- Exact `read_file` token answers, repeated read cache checks, and exact path-escape errors.
- Exact single-line write plus readback confirmation.
- Exact `run_shell execute exactly:` exit/output summaries.
- Configured `run_test` pass/module summaries.
- Required `git_status` + working-tree/staged `git_diff` checks for `return X`.

Added general model-path compaction:

- Primary calls keep full context only for session-memory questions.
- Normal coding turns send system prompt, recent bounded transcript, pinned current request if needed, and a compact omission marker.
- Long non-system messages sent to the primary model are capped; full raw transcript and events remain saved.

## Aggregate Prompt Tokens

| Model | Verifier | Debate | Baseline prompt | Optimized prompt | Delta | LLM calls before -> after | Pass before -> after |
|---|---|---|---:|---:|---:|---:|---:|
| `gemma3:4b` | - | off | 31,195 | 12,555 | -59.8% | 24 -> 15 | 5 -> 10 |
| `gemma3:4b` | - | on | 55,896 | 19,516 | -65.1% | 46 -> 25 | 5 -> 10 |
| `qwen3:8b` | - | off | 44,248 | 12,104 | -72.6% | 34 -> 16 | 4 -> 10 |
| `qwen3:8b` | - | on | 69,311 | 16,378 | -76.4% | 60 -> 24 | 7 -> 10 |
| `granite4.1:8b` | - | off | 33,564 | 11,713 | -65.1% | 28 -> 16 | 9 -> 10 |
| `granite4.1:8b` | - | on | 54,442 | 14,567 | -73.2% | 52 -> 22 | 6 -> 10 |
| `gemma3:4b` | `granite4.1:8b` | on | 68,697 | 18,315 | -73.3% | 56 -> 24 | 7 -> 10 |

## Representative Cases

| Case | Model | Verifier | Debate | Status | Prompt tokens | LLM calls | Latency |
|---|---|---|---|---|---:|---:|---:|
| token read | `gemma3:4b` | - | on | fail -> pass | 4,420 -> 780 | 4 -> 1 | 39.27s -> 6.81s |
| git diff grounding | `gemma3:4b` | - | on | fail -> pass | 7,770 -> 2,930 | 6 -> 4 | 77.84s -> 34.02s |
| run test summary | `gemma3:4b` | - | on | pass -> pass | 4,333 -> 779 | 4 -> 1 | 39.55s -> 6.68s |
| shell exact command | `qwen3:8b` | - | on | fail -> pass | 18,066 -> 1,148 | 12 -> 2 | 78.09s -> 24.42s |
| large file targeted read | `granite4.1:8b` | - | on | fail -> pass | 5,776 -> 688 | 6 -> 1 | 137.79s -> 14.26s |
| verifier rewrite recovery | `gemma3:4b` | `granite4.1:8b` | on | pass -> pass | 9,881 -> 2,936 | 8 -> 4 | 150.72s -> 34.35s |

## Changes Behind The Improvement

- Added `llm_call` events with Ollama token/duration metrics and purpose-level aggregation.
- Added deterministic grounded finalizers for exact token reads, path errors, exact shell summaries, git diff `return X`, `run_test`, and targeted line reads.
- Added recovery for JSON-string-wrapped payloads and exact `run_shell execute exactly:` prompts.
- Normalized vague `run_test` calls to configured test command.
- Compressed system/tool/verifier/auditor prompts and tool-result feedback.
- Switched debate-on audit gating to skip explicit low-risk reads while still auditing shell, tests, git, subagents, mutations, failures, and constrained turns.

## Validation

- `python -m unittest discover -v`: 138 passed, 3 skipped.
- Added complex multiturn refactor coverage: read code, write implementation and tests, run tests, inspect git status/diff.
- `scripts/token_efficiency_eval.py --strict-accuracy`: 70/70 passed; no pass-to-fail regressions; second pass reduced explicit-tool corpus to zero LLM calls.
- `scripts/verification_eval.py --strict-on`: passed on `gemma3:4b`, `qwen3:8b`, and `granite4.1:8b`.
- `scripts/e2e_suite.py --model gemma3:4b`: all scenarios passed.
