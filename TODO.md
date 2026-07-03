# TODO

Current roadmap: keep Ollama Code ready as a reliable local coding agent for ordinary repository work. The canonical direction is `docs/product-strategy.md`.

## Current Status

- No active product-readiness blocker is expected when `scripts/product_readiness_report.py --strict` is green.
- Keep `task_due_filter` green as the product-critical feature-delivery hard case for CLI flag work, tests, docs, parser errors, priority preservation, and shell proof.
- Treat `scripts/product_readiness_report.py --strict` as the current-state release summary after refreshing doctor, validation, live-gate, and benchmark artifacts.
- When readiness artifacts are stale, `scripts/product_readiness_report.py --strict` prints the exact refresh commands for the blocking checks.
- `local_small` readiness is proven from the selected model benchmark artifact referenced by the live-gate summary when that artifact exists.

## Completed Controller Tranche

- Feature-delivery obligation derivation and proof-status policy now live in controller modules instead of ad hoc `agent.py` guards.
- CLI flag-bundle request classification now lives in `ollama_code.controller.feature_delivery`; `agent.py` only delegates to it.
- Mutation, test-run, test-edit, and validation request-intent policy now lives in `ollama_code.controller.request_policy`; `agent.py` keeps compatibility adapters.
- Exact-grounding request detection now lives in `ollama_code.controller.request_policy` and feeds final verification through the agent adapter.
- Continue/resume prompt classification now lives in `ollama_code.controller.request_policy`, preserving sticky obligation handling through an adapter.
- Doc/test path classification now lives in `ollama_code.controller.request_policy` and is reused by obligation and proof adapters.
- Tool-request parsing, forbidden-tool constraints, dynamic MCP tool names, and generic requires-tools detection now live in `ollama_code.controller.request_policy`.
- Structured file-tool preference, session-memory detection, commit allowance, exact-output, and tool-error request predicates now live in `ollama_code.controller.request_policy`.
- Broad-request, systems-lens, TODO-benefit, workspace-path, and clarification-risk classifiers now live in `ollama_code.controller.request_policy`.
- Mechanical request parsers for read/list/search/test/git/outline/symbol/line targets now live in `ollama_code.controller.request_policy`.
- Exact-literal request parsers for single-line writes, exact replies, loose file creation, and exact shell commands now live in `ollama_code.controller.request_policy`.
- Shell/git request policy for git diff mode, test-command detection, grep normalization, and find-exec-grep normalization now lives in `ollama_code.controller.request_policy`.
- Edit/shell classification policy for code-file paths, snippet-shaped symbol edits, and shell file-mutation detection now lives in `ollama_code.controller.edit_policy`.
- Exact-literal and snippet-style edit normalization now live in `ollama_code.controller.edit_policy`.
- Unsupported edit/file tool alias normalization now lives in `ollama_code.controller.edit_policy`.
- Edit payload alias normalization and escaped-newline repair now live in `ollama_code.controller.edit_policy`.
- Tool payload normalization for model-emitted final/tool shapes now lives in `ollama_code.controller.tool_payload_policy`; `agent.py` supplies the runtime supported-tool predicate.
- Target-line read, run-test, unittest-file, and simple find-shell call normalization now live in `ollama_code.controller.tool_call_policy`; `agent.py` supplies filesystem and default-test-command context.
- Shell test-command routing from `run_shell` to `run_test` now lives in `ollama_code.controller.tool_call_policy`; `agent.py` supplies request, approval, and workspace context.
- Shell inspection routing from `run_shell` to structured read/list/search tools now lives in `ollama_code.controller.tool_call_policy`; `agent.py` supplies request, approval, and parser callbacks.
- Head/tail shell preview parsing now lives in `ollama_code.controller.tool_call_policy`; `agent.py` only supplies file line counts for tail ranges.
- Import-repair, project-rename, and optional-parameter bootstrap routing now live in `ollama_code.controller.tool_call_policy`; `agent.py` supplies parsed request facts and operation candidates.
- Project function-rename operation planning and already-satisfied detection now live in `ollama_code.controller.operation_policy`; `agent.py` only delegates.
- Workflow config update request parsing and source-to-operation planning now live in `ollama_code.controller.operation_policy`; `agent.py` only loads workflow text.
- Symbol return-update request parsing, expression cleanup, and source-to-operation planning now live in `ollama_code.controller.operation_policy`; `agent.py` only loads source and passes tool requirements.
- Optional-parameter request parsing, signature shaping, and docs/source operation planning now live in `ollama_code.controller.operation_policy`; `agent.py` only loads source and docs.
- Test-grounded symbol-return parsing and exact successful-tool-call detection now live in `ollama_code.controller.operation_policy`; `agent.py` supplies only the test-path predicate.
- Focused Python test-driven repair request classification now lives in `ollama_code.controller.feature_delivery`; `agent.py` supplies only runtime path/default-test context.
- Spec-guided repair eligibility policy now lives in `ollama_code.controller.feature_delivery`; `agent.py` supplies only test-example parser callbacks.
- Import-repair exclusion policy for structured test-driven repair now lives in `ollama_code.controller.feature_delivery`.
- Mechanical obligation repair failure detection now lives in `ollama_code.controller.feature_delivery`; `agent.py` supplies only recorded events.
- Final-claim detection and final-verification requirement policy now live in `ollama_code.controller.final_policy`; `agent.py` supplies runtime context.
- Typed repair-protocol state now lives under `ollama_code.controller`, with the old top-level import kept as a compatibility shim.
- Failed-edit repair-spec policy decisions for strategy selection, proof items, broad repair hints, and retry mutation allowance now live in controller modules.
- Repair-spec behavior-surface path policy now lives in `ollama_code.controller.repair_protocol`; `agent.py` only supplies workspace-discovered test candidates.
- Repair-spec validation-loop blocking, retry guidance, and failed-test repair retry policy now live in `ollama_code.controller.repair_protocol`.
- Focused controller tests cover feature-delivery policy directly instead of relying only on the omnibus `tests/test_agent.py`.
- `local_validation` now reports and gates focused agent test ownership so extracted behavior modules cannot depend on the legacy omnibus `tests/test_agent.py`.
- `ToolExecutor` contract and synthesis helper policy has started moving into focused modules:
  `ollama_code/tools/contracts.py`, `ollama_code/tools/synthesis.py`, `ollama_code/tools/validation.py`, and
  `ollama_code/tools/command_validation.py`.

## Completed ToolExecutor Tranche

- `lint_typecheck` file-level analysis, scan-state collection, target planning, subprocess runner, timeout shaping, cache-hit shaping, and final result shaping now live in `ollama_code/tools/validation.py` with direct tests.
- `ToolExecutor.lint_typecheck` is now mostly adapter code for path resolution, local tool discovery, cache storage, and callback injection.
- Candidate validation signature-gate policy, result shaping, workspace-copy ignore policy, and temp-workspace validation orchestration now live in `ollama_code/tools/synthesis.py` with direct tests.
- Function-probe script generation and result shaping now live in `ollama_code/tools/synthesis.py` with direct tests.
- Python write-content repair for generated code fences, quote prefixes, rewrite markers, common join typos, and auto-dedent now lives in `ollama_code/tools/synthesis.py` with direct tests.
- Python generated-function replacement policy for parameter diagnostics, shadowed-builtin checks, critical-parameter checks, foldr argument repair, and safe canonical-signature normalization now lives in `ollama_code/tools/synthesis.py` with direct tests.
- `edit_intent` symbol-routing policy for symbol target normalization, full-function routing, project rename routing, and text fallback now lives in `ollama_code/tools/synthesis.py` with direct tests.
- `change_signature` signature normalization for bare, callable, full-function, multiline, and invalid replacement inputs now lives in `ollama_code/tools/synthesis.py` with direct tests.
- `add_import` insertion policy for from-import merging, header-aware placement, multiline requests, and executable-payload rejection now lives in `ollama_code/tools/synthesis.py` with direct tests.
- Symbol deletion and move shaping for ambiguous-match rendering, matched-symbol text removal, and destination append formatting now lives in `ollama_code/tools/synthesis.py` with direct tests.
- Project rename identifier validation, per-file replacement detection, already-done detection, and result shaping now lives in `ollama_code/tools/synthesis.py` with direct tests.

## Next ToolExecutor Tranche

- Do not keep extracting tiny wrappers just to reduce line count.
- Next meaningful split is only the remaining structured-edit/apply wrappers if a measured gap appears; otherwise prefer readiness-artifact freshness or validator discovery work over more ToolExecutor splitting.
- Preserve compatibility wrappers until extracted implementations have direct focused tests.

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
