# Coding Benchmarks

`scripts/coding_benchmark_eval.py` measures coding-task accuracy and token cost from saved Ollama Code sessions. It supports parallel case execution and writes raw JSON under ignored `scratch/` by default.

For the current pooled two-GPU local proxy (`OLLAMA_HOST=http://127.0.0.1:11437`), the measured sweet spot is `12` jobs. Throughput improved through `12` workers, then flattened; `16` was effectively no faster and had worse latency.

Start that pooled proxy with:

```bash
python scripts/ollama_pool_proxy.py --listen-port 11437 --backends http://127.0.0.1:11435 http://127.0.0.1:11436
```

## Why Local First

The default suite is local and deterministic because it can run often without Docker images, large datasets, or leaderboard assumptions.

External benchmark inspiration:

- [SWE-bench](https://github.com/SWE-bench/SWE-bench): real issue-to-patch tasks with tests and Docker harnesses. Use sampled local smoke only unless you run official settings.
- [OpenAI SWE-bench Verified note](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/): Verified has contamination and test-quality caveats for current frontier coding claims.
- [Terminal-Bench](https://github.com/harbor-framework/terminal-bench) and [Terminal-Bench 2.0](https://arxiv.org/abs/2601.11868): terminal task folders with instructions, tests, and oracle solutions.
- [Aider Polyglot leaderboard docs](https://aider.chat/docs/leaderboards/): useful comparison for edit skill across languages, but not a direct measure of this CLI controller.
- [RepoBench](https://arxiv.org/abs/2306.03091): repo-level retrieval/completion motivation for symbol-aware context tools.
- [Edit, But Verify](https://arxiv.org/abs/2604.05100): edit benchmarks need test-backed validation, not only textual issue completion.

## Suites

- `local-small`: 8 fast tasks for regular local checks.
- `local-full`: current 33-task superset that includes `local-small` plus larger symbol navigation, multi-turn editing, refactors, validator-recovery, path/shell/git regressions, and token traps.
- `external-smoke`: preflight checks for optional external harnesses; not CI-blocking and not leaderboard-comparable.
- `scripts/public_benchmark_eval.py`: public Aider Polyglot Python smoke. It clones [Aider-AI/polyglot-benchmark](https://github.com/Aider-AI/polyglot-benchmark) under ignored `scratch/external/`, runs selected Exercism Python tasks, and records status/tokens/tool calls.

## Benchmark Classes

- `agent`: tools are allowed, but the benchmark must use the LLM at least once. This is the main coding benchmark class.
- `controller`: zero-LLM fast paths are allowed. These cases measure routing/tool correctness and should not be mixed into agent-model accuracy claims.

## Metrics

Each result records:

- `status`: `pass`, `fail`, or `fail_closed`
- LLM calls and Ollama token metrics: prompt, output, total
- latency, tool-call sequence, failed tools, changed files, tests run
- assumption-audit count/retries, verifier retries/rewrites
- prompt profile: chars by role and largest prompt messages
- feature profile: `baseline`, `schema`, `context-pack`, `evidence-handles`, `num-predict-caps`, `structured-edits`, `trajectory-guards`, `contract-guards`, or `all`
- benchmark class summary: agent vs controller runs, passes, LLM calls, and token totals are reported separately

## Commands

Run the frequent local suite:

```bash
python scripts/coding_benchmark_eval.py --suite local-small --models granite4.1:8b gemma4:e4b qwen3:8b --modes off on --benchmark-classes agent controller --jobs 12 --strict-accuracy --strict-budget
```

Start with the fast repo-owned readiness tier before broader benchmark work:

```bash
python scripts/local_validation.py --tier smoke
```

For `pytest` runs, the summary JSON includes a `coverage_summary` block that proves whether the repo-owned `smoke` plus `agent` plus `full-remaining` partition still covers the discovered test files exactly once.

Run the full local validation stack before merging larger controller changes:

```bash
python scripts/local_validation.py --tier full
```

When `pytest` and `xdist` are available, the local validation script resolves `--jobs auto` to a bounded worker count instead of delegating to unrestricted `xdist auto`. If `coverage_summary.full_plan_covers_all_discovered_targets` is false, `scripts/local_validation.py` exits nonzero even when the subprocesses themselves passed.

Run A/B feature profiles without changing prompts:

```bash
python scripts/coding_benchmark_eval.py --suite local-small --models granite4.1:8b --modes off --reconcile-modes auto --feature-profiles baseline trajectory-guards contract-guards all --benchmark-classes agent --jobs 12 --compare scratch/coding-benchmark/baseline.json --strict-accuracy --strict-budget
```

Run agent benchmarks only and require a real model call:

```bash
python scripts/coding_benchmark_eval.py --suite local-small --models granite4.1:8b --modes off --feature-profiles all --benchmark-classes agent --jobs 12 --require-llm-for-agent-benchmarks
```

Run controller-only routing checks without mixing them into agent numbers:

```bash
python scripts/coding_benchmark_eval.py --suite local-full --models granite4.1:8b --modes off --benchmark-classes controller --jobs 12 --strict-accuracy --strict-budget
```

Run the fuller local suite for deeper regression checks:

```bash
python scripts/coding_benchmark_eval.py --suite local-full --models granite4.1:8b --modes off on --benchmark-classes agent controller --jobs 12 --strict-accuracy --strict-budget
```

Compare against a previous run:

```bash
python scripts/coding_benchmark_eval.py --suite local-small --benchmark-classes agent --jobs 12 --output scratch/coding-benchmark/after.json --compare scratch/coding-benchmark/before.json --strict-accuracy --strict-budget
```

Check optional external harness availability:

```bash
python scripts/coding_benchmark_eval.py --suite external-smoke --output scratch/coding-benchmark/external-smoke.json
```

Run public Aider Polyglot Python smoke:

```bash
python scripts/public_benchmark_eval.py --models granite4.1:8b --modes off --jobs 12 --tasks list-ops pig-latin wordy --output scratch/public-bench/aider-polyglot-python-smoke.json
```

Compare public smoke before/after:

```bash
python scripts/public_benchmark_eval.py --models granite4.1:8b --modes off --jobs 12 --tasks list-ops pig-latin wordy --compare scratch/public-bench/baseline.json --output scratch/public-bench/after.json
```

Raw JSON stays ignored under `scratch/`. Track only concise summaries when results are worth preserving.

## Live Model Gate

Use `scripts/live_model_gate.py` when you need one release-style artifact that proves the stack hit a real local Ollama model rather than only `FakeClient` or mocked HTTP tests.

Default gate steps:

- `scripts/e2e_suite.py` on explicit live-model scenarios with `--require-llm-for-turn`
- `scripts/verification_eval.py --strict-on`
- `scripts/coding_benchmark_eval.py` with `--require-llm-for-agent-benchmarks` on the runtime default feature profile `all`

Frequent local release gate:

```bash
python scripts/live_model_gate.py --models granite4.1:8b --benchmark-suite local-small --benchmark-jobs 1
```

Fast proof that skips the longer benchmark step:

```bash
python scripts/live_model_gate.py --models granite4.1:8b --skip-benchmarks
```

Artifacts land under:

- `scratch/live-model-gate/live-model-gate-summary.json`
- `scratch/live-model-gate/coding-benchmark-<model>.json`

The summary JSON records the detected `OLLAMA_HOST`, installed models, resolved models, exact commands, return codes, durations, and command output tails for each gate step.
It also records the canonical provenance and release-selection fields: `git_commit`, `git_dirty`, `benchmark_suite`, `selected_default_model`, `selection_reason`, and per-model gate rows with benchmark pass/token/latency totals. The fixed-path summary is now a canonical mirror of the latest live-gate run for a specific source state, even when the full run artifacts are written under a timestamped scratch directory.

If you want to A/B a narrower profile such as `trajectory-guards`, pass it explicitly with `--benchmark-feature-profiles trajectory-guards`. Keep that separate from release gating, because the gate should reflect the actual shipped/default runtime profile.

For GitHub Actions, the repository now includes `.github/workflows/live-model-gate.yml`. It is intentionally manual and expects a `self-hosted` runner labeled `ollama`, because the default hosted runners do not provide local Ollama models.

## Clarification Quality

Use `scripts/question_quality_eval.py` to score whether the agent asks high-leverage elimination-by-aspect clarification questions when broad requests are genuinely ambiguous, while still proceeding on focused exact-edit requests.

Run it with:

```bash
python scripts/question_quality_eval.py
```

Run the live Ollama-backed clarification scenario with:

```bash
python scripts/e2e_suite.py --model granite4.1:8b --scenarios scenario_clarifying_question_eba
```

Artifacts land under:

- `scratch/question-quality/clarification-question-eval.json`
- `scratch/question-quality/clarification-question-eval.md`

The synthetic eval uses broad prompts plus pretend human answers to check question quality separately from coding accuracy. It verifies:

- EBA-style question shape: one highest-leverage axis, 2-4 mutually exclusive choices, recommended default
- question/choice alignment: pretend answers map cleanly to one choice
- restraint: focused exact-edit requests stay on `proceed` instead of asking unnecessary questions

## Anti-Cheat Rules

- `coding_accuracy` prompts must not include synthetic marker tokens, exact answer literals, forced tool clauses, or public task-specific answers.
- Runtime code under `ollama_code/` must not special-case public smoke task names such as `list-ops`, `pig-latin`, or `wordy`.
- Synthetic exact-answer cases stay marked as `tool_contract`, not coding accuracy.
- Agent benchmark claims should cite only `agent` class results, not controller routing checks.
- Run `python scripts/anti_cheat_scan.py` before reporting benchmark gains; it checks runtime code for task names/synthetic answer markers and benchmark prompts for leaked answers.
- Public benchmark results are local smoke only unless official harness/settings are used.
