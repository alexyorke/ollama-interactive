# TODO

Current roadmap: make Ollama Code a reliable local coding agent for ordinary repository work. The canonical direction is `docs/product-strategy.md`.

## Current Blocker

- Keep `task_due_filter` green as the product-critical feature-delivery hard case for CLI flag work, tests, docs, parser errors, priority preservation, and shell proof.
- Treat `scripts/product_readiness_report.py --strict` as the current-state release summary once the required artifacts have been refreshed for the current git commit.

## Next Controller Tranche

- Move feature-delivery policy out of ad hoc `agent.py` guards and into typed controller state: deliverables, grounded targets, patch bundle, allowed next actions, validation plan, proof obligations, and fail-closed state.
- Keep `agent.py` as the adapter while behavior is proven.
- Add or move tests by behavior family instead of adding more cases to the omnibus `tests/test_agent.py`.

## Readiness Gates

- `python -m ollama_code --doctor`
- `python scripts/local_validation.py --tier smoke`
- `python scripts/local_validation.py --tier agent`
- `python scripts/coding_benchmark_eval.py --suite local-full --models granite4.1:8b --modes off --cases task_due_filter --feature-profiles all --benchmark-classes agent controller --jobs 1 --strict-accuracy --strict-budget --require-llm-for-agent-benchmarks`
- `python scripts/product_readiness_report.py --strict`

## Deferred Backlog

- More transcript or dataset discovery.
- Optional-tool expansion beyond the currently useful local set.
- Token micro-optimization after deterministic probe cost is already low.
- Broad `ToolExecutor` splitting before controller policy ownership is stable.
