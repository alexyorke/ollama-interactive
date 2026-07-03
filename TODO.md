# TODO

Current roadmap: keep Ollama Code ready as a reliable local coding agent for ordinary repository work. The canonical direction is `docs/product-strategy.md`.

## Current Status

- No active product-readiness blocker is expected when `scripts/product_readiness_report.py --strict` is green.
- Keep `task_due_filter` green as the product-critical feature-delivery hard case for CLI flag work, tests, docs, parser errors, priority preservation, and shell proof.
- Treat `scripts/product_readiness_report.py --strict` as the current-state release summary after refreshing doctor, validation, live-gate, and benchmark artifacts.

## Completed Controller Tranche

- Feature-delivery obligation derivation and proof-status policy now live in controller modules instead of ad hoc `agent.py` guards.
- Typed repair-protocol state now lives under `ollama_code.controller`, with the old top-level import kept as a compatibility shim.
- Failed-edit repair-spec policy decisions for strategy selection, proof items, broad repair hints, and retry mutation allowance now live in controller modules.
- Focused controller tests cover feature-delivery policy directly instead of relying only on the omnibus `tests/test_agent.py`.
- `ToolExecutor` contract and synthesis helper policy has started moving into focused modules:
  `ollama_code/tools/contracts.py`, `ollama_code/tools/synthesis.py`, `ollama_code/tools/validation.py`, and
  `ollama_code/tools/command_validation.py`.

## Next ToolExecutor Tranche

- Do not keep extracting tiny wrappers just to reduce line count.
- Next meaningful split is the remaining `lint_typecheck` subprocess/cache orchestration.
- File-level lint analysis, scan-state collection, and target planning already live in `ollama_code/tools/validation.py` with direct tests.
- Treat the remaining subprocess runner as a separate measured tranche because it couples cache hits, timeout behavior, ruff, basedpyright/pyright, bash, command rendering, and result assembly.
- Preserve the current compatibility wrappers until the extracted validator implementation has direct focused tests.

## Readiness Gates

- `python -m ollama_code --doctor`
- `python scripts/doctor_report.py`
- `python scripts/local_validation.py --tier smoke`
- `python scripts/local_validation.py --tier agent`
- `python scripts/coding_benchmark_eval.py --suite local-full --models granite4.1:8b --modes off --cases task_due_filter --feature-profiles all --benchmark-classes agent controller --jobs 1 --strict-accuracy --strict-budget --require-llm-for-agent-benchmarks`
- `python scripts/product_readiness_report.py --strict`

## Deferred Backlog

- More transcript or dataset discovery.
- Optional-tool expansion beyond the currently useful local set.
- Token micro-optimization after deterministic probe cost is already low.
- Broad `ToolExecutor` splitting beyond validator execution, contracts, and synthesis helpers unless a measured product-readiness gap justifies it.
