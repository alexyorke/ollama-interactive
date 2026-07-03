# Product Strategy

## North Star

Ollama Code should be a reliable local coding agent for ordinary repository work. It is not primarily a benchmark lab, transcript research project, or optional-tool showcase. Those systems are useful only when they improve the user-visible coding loop.

The product is ready for a user when this path works:

1. Install and run `ollama-code --doctor`.
2. Ask for a normal feature, bug fix, refactor, or docs/test update.
3. The agent grounds the requested files and symbols before mutation.
4. The agent edits code, tests, and docs when requested.
5. The agent runs proof-producing validation.
6. The agent either reports the proven result or stops with explicit unresolved obligations.

## Readiness Contract

Release-readiness is blocked by these failures:

- First-use setup is broken or `--doctor` cannot explain missing requirements.
- The selected default model is not backed by a current live-model gate.
- `local_validation --tier smoke` or controller-focused `agent` validation is failing.
- A product-critical feature-delivery hard case such as `task_due_filter` fails.
- The agent claims success without source proof and behavior proof for requested command or flag work.
- The latest readiness artifacts are stale or for a different git state.

These are non-blocking unless they explain a current readiness failure:

- More public transcript discovery.
- Optional tool expansion beyond the currently useful local set.
- Token micro-optimization after deterministic probe cost is already low.
- Broad benchmark expansion without a failing product-critical workflow.

## Architecture Direction

New controller behavior should move into typed state and policy modules instead of growing the monolithic `agent.py` controller. The model may propose content, but deterministic state should decide the next allowed action.

The target controller shape is:

- requested deliverables,
- grounded targets,
- patch bundle,
- failed attempts,
- allowed next actions,
- validation plan,
- proof obligations,
- fail-closed state.

`agent.py` remains the adapter during migration. Legacy guards can stay as compatibility shims, but new fixes should be expressed as typed state transitions, patch-plan validation, or proof-obligation rules.

## Measurement Direction

Use `scripts/product_readiness_report.py` as the single current-state summary. It does not run expensive work; it reads existing artifacts and fails if they are missing, stale, dirty, or not green.

Preferred readiness flow:

```bash
python -m ollama_code --doctor
python scripts/local_validation.py --tier smoke
python scripts/local_validation.py --tier agent
python scripts/coding_benchmark_eval.py --suite local-full --models granite4.1:8b --modes off --cases task_due_filter --feature-profiles all --benchmark-classes agent controller --jobs 1 --strict-accuracy --strict-budget --require-llm-for-agent-benchmarks
python scripts/product_readiness_report.py --strict
```

Run full local and live-model gates before release-style claims, not during every small controller iteration.
