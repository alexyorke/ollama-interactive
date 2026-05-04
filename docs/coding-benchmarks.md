# Coding Benchmarks

`scripts/coding_benchmark_eval.py` measures coding-task accuracy and token cost from saved Ollama Code sessions. It is serial-only and writes raw JSON under ignored `scratch/` by default.

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
- `local-full`: 20 tasks including larger symbol navigation, multi-turn editing, refactors, path/shell/git regressions, and token traps.
- `external-smoke`: preflight checks for optional external harnesses; not CI-blocking and not leaderboard-comparable.
- `scripts/public_benchmark_eval.py`: serial public Aider Polyglot Python smoke. It clones [Aider-AI/polyglot-benchmark](https://github.com/Aider-AI/polyglot-benchmark) under ignored `scratch/external/`, runs selected Exercism Python tasks, and records status/tokens/tool calls.

## Metrics

Each result records:

- `status`: `pass`, `fail`, or `fail_closed`
- LLM calls and Ollama token metrics: prompt, output, total
- latency, tool-call sequence, failed tools, changed files, tests run
- assumption-audit count/retries, verifier retries/rewrites
- prompt profile: chars by role and largest prompt messages
- feature profile: `baseline`, `schema`, `context-pack`, `evidence-handles`, `num-predict-caps`, `structured-edits`, `trajectory-guards`, `contract-guards`, or `all`

## Commands

Run the frequent local suite:

```bash
python scripts/coding_benchmark_eval.py --suite local-small --models gemma3:4b qwen3:8b granite4.1:8b --modes off on --strict-accuracy --strict-budget
```

Run A/B feature profiles without changing prompts:

```bash
python scripts/coding_benchmark_eval.py --suite local-small --models granite4.1:8b --modes off --reconcile-modes auto --feature-profiles baseline trajectory-guards contract-guards all --compare scratch/coding-benchmark/baseline.json --strict-accuracy --strict-budget
```

Run the fuller local suite for deeper regression checks:

```bash
python scripts/coding_benchmark_eval.py --suite local-full --models gemma3:4b --modes off on --strict-accuracy --strict-budget
```

Compare against a previous run:

```bash
python scripts/coding_benchmark_eval.py --suite local-small --output scratch/coding-benchmark/after.json --compare scratch/coding-benchmark/before.json --strict-accuracy --strict-budget
```

Check optional external harness availability:

```bash
python scripts/coding_benchmark_eval.py --suite external-smoke --output scratch/coding-benchmark/external-smoke.json
```

Run public Aider Polyglot Python smoke:

```bash
python scripts/public_benchmark_eval.py --models granite4.1:8b --modes off --tasks list-ops pig-latin wordy --output scratch/public-bench/aider-polyglot-python-smoke.json
```

Compare public smoke before/after:

```bash
python scripts/public_benchmark_eval.py --models granite4.1:8b --modes off --tasks list-ops pig-latin wordy --compare scratch/public-bench/baseline.json --output scratch/public-bench/after.json
```

Raw JSON stays ignored under `scratch/`. Track only concise summaries when results are worth preserving.

## Anti-Cheat Rules

- `coding_accuracy` prompts must not include synthetic marker tokens, exact answer literals, forced tool clauses, or public task-specific answers.
- Runtime code under `ollama_code/` must not special-case public smoke task names such as `list-ops`, `pig-latin`, or `wordy`.
- Synthetic exact-answer cases stay marked as `tool_contract`, not coding accuracy.
- Public benchmark results are local smoke only unless official harness/settings are used.
