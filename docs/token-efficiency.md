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

Latest compression pass reduced repeated model context without changing public CLI behavior:

- Static primary, verifier, auditor, and rewriter prompts were shortened while preserving the same JSON contracts.
- Verification/audit payloads now send fewer recent messages, fewer evidence rows, and shorter claim/tool-result snippets.
- Explicit low-risk validation tools (`run_test`, `git_status`, `git_diff`) and explicit `run_agent` delegation skip assumption-audit calls unless there is a retry, forbidden-tool constraint, or unsafe context.
- Failed-test turns now let read-only inspection proceed without an extra auditor call; repeated tests or ungrounded mutation stay guarded.
- The benchmark runner writes partial raw JSON after each case, so long local model runs preserve token profiles even if interrupted.

Current measured regression check on the strongest installed local coding model:

| Case | Model | Debate | Status | Tokens before | Tokens after | Delta | Calls before -> after |
|---|---|---|---|---:|---:|---:|---:|
| `multi_file_refactor` | `batiai/qwen3.6-35b:iq4` | on | pass -> pass | 12,630 | 10,625 | -15.9% | 10 -> 10 |

Additional live smoke:

| Model | Suite slice | Status |
|---|---|---|
| `gemma3:4b` | 18 deterministic/token-efficiency cases | 18/18 pass, 0 LLM calls |
| `granite4.1:8b` | 8 local coding smoke cases | 8/8 pass |
| `qwen3:8b` | 8 local coding smoke cases | 7/8 pass; debate-on `instructed_edit` timed out, so this model is not recommended for complex edit loops |

Public benchmark smoke from [Aider-AI/polyglot-benchmark](https://github.com/Aider-AI/polyglot-benchmark), Python Exercism tasks `list-ops`, `pig-latin`, `wordy`, `granite4.1:8b`, debate off:

| Pass | Tokens | LLM calls | Note |
|---:|---:|---:|---|
| 0/3 | 39,161 | 28 | Baseline after correcting Exercism test discovery to `*_test.py` |
| 0/3 | 27,168 | 22 | After shell-test normalization and prompt trim |

The model still fails these public tasks, so this is not an accuracy claim. It is useful token profiling: normalizing accidental `run_shell "python -m unittest ..."` calls to configured `run_test` cut one public-smoke run by 30.6% without a pass-to-fail regression. Single-task reruns are noisy with 8B local models, so compare full JSON runs rather than one task.

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

## Code-Aware Navigation

Added code navigation tools so broad code work can avoid `grep` plus full-file reads:

- `search_symbols(query, path='.', limit=50)` finds definitions by symbol name.
- `code_outline(path, max_symbols=120)` returns imports plus symbol names and line ranges, not bodies.
- `read_symbol(path, symbol, include_context=2)` reads one function/class/method body by AST range for Python, with generic regex fallback for JS/TS-like files.
- Symbol scans prune dependency/cache folders such as `node_modules`, `.git`, `dist`, `build`, `.venv`, and `__pycache__`.
- Extra hallucinated tool arguments are ignored when required arguments are present; this fixed a real `gemma3:4b` loop where `read_symbol` was called repeatedly with unused `start/end`.
- Exact uppercase token answers from `read_symbol` are synthesized by the controller, avoiding a final model call after the relevant symbol body is already proven.
- Explicit low-risk symbol tools skip assumption auditing even when the prompt forbids unrelated tools such as `read_file`; deterministic forbidden-tool enforcement still runs first.

Large code symbol eval uses a synthetic `src/large_pricing.py` with 440 filler functions around the target function. Prompt requires `search_symbols` then `read_symbol`, forbids `read_file`, and asks for a marker inside the target symbol only.

| Model | Debate | Status | LLM calls | Prompt tokens | Output tokens | Latency |
|---|---|---|---:|---:|---:|---:|
| `gemma3:4b` | off | pass | 2 | 1,902 | 83 | 15.67s |
| `gemma3:4b` | on | pass | 2 | 1,898 | 83 | 13.23s |
| `qwen3:8b` | off | pass | 2 | 1,668 | 63 | 21.93s |
| `qwen3:8b` | on | pass | 2 | 1,686 | 113 | 26.22s |
| `granite4.1:8b` | off | pass | 2 | 1,622 | 63 | 27.88s |
| `granite4.1:8b` | on | pass | 2 | 1,624 | 58 | 20.98s |

Regression caught during development: before extra-argument recovery and `read_symbol` synthesis, `gemma3:4b` debate-off failed after 12 LLM calls and 18,379 prompt tokens on this same symbol task. After the fix it passes in 2 calls and 1,902 prompt tokens.

## Profiling Pass

Added per-call prompt profiling to every `llm_call` event:

- `prompt_chars_by_role` shows how much input came from system, user, and assistant messages.
- `top_prompt_messages` records the largest prompt messages by purpose/role/char count.
- `scripts/token_efficiency_eval.py` now aggregates those fields into raw JSON so token waste can be tied to prompt components, not only total Ollama token counts.

Profiling a realistic symbol-summary task found the main waste pattern: models often looped after `search_symbols` instead of calling `read_symbol`, repeatedly resending the system prompt and growing transcript. Example before the fix:

| Model | Debate | Status before | Calls before | Prompt tokens before | Dominant prompt component |
|---|---|---|---:|---:|---|
| `gemma3:4b` | off | fail | 12 | 11,421 | system prompt repeated 12x |
| `gemma3:4b` | on | fail | 16 | 14,926 | auditor payloads plus system prompt |
| `qwen3:8b` | off | fail | 12 | 10,118 | system prompt repeated 12x |
| `qwen3:8b` | on | fail | 12 | 10,151 | system prompt repeated 12x |
| `granite4.1:8b` | off | pass | 9 | 6,666 | repeated search attempts |
| `granite4.1:8b` | on | fail | 13 | 11,532 | auditor payloads plus system prompt |

Fixes from that profile:

- Compacted primary system prompt/tool schema; current full system prompt with tool list is about 1,464 chars for a benchmark workspace path.
- Shortened tool-result feedback from verbose prose to compact `Tool result:` plus `Next JSON only.`
- Added deterministic symbol return-value synthesis for prompts like “find function X in file Y and summarize what value it returns.”
- Broadened deterministic symbol extraction to `find`, `locate`, `function`, `method`, `class`, and `symbol` wording.

After the fix, the same symbol-summary task passes on all tested models and debate modes with `0` LLM calls, `0` prompt tokens, and `0` output tokens.

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
- Added symbol-aware navigation and exact-token synthesis from symbol bodies.
- Added recovery for JSON-string-wrapped payloads and exact `run_shell execute exactly:` prompts.
- Added recovery for extra hallucinated tool arguments when required arguments are valid.
- Normalized vague `run_test` calls to configured test command.
- Compressed system/tool/verifier/auditor prompts and tool-result feedback.
- Switched debate-on audit gating to skip explicit low-risk reads and symbol tools while still auditing shell, tests, git, subagents, mutations, failures, and ambiguous turns.

## Validation

- `python -m unittest discover -v`: 169 passed, 3 skipped.
- Added complex multiturn refactor coverage: read code, write implementation and tests, run tests, inspect git status/diff.
- Added large-code symbol navigation coverage: find symbol, read exact symbol, avoid full-file read, synthesize exact token.
- Added prompt profile coverage: per-call prompt chars by role and largest prompt messages are recorded in eval JSON.
- `scripts/token_efficiency_eval.py --strict-accuracy`: 84/84 passed; no pass-to-fail regressions; explicit-tool and symbol-summary corpora are zero LLM calls.
- `scripts/verification_eval.py --strict-on`: passed on `gemma3:4b`, `qwen3:8b`, and `granite4.1:8b`.
- `scripts/e2e_suite.py --model gemma3:4b`: all scenarios passed.
