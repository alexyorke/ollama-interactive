# Direction Reset: Reliable Local Coding Agent

## Current Inventory

Runtime entrypoints are simple: `C:\Users\yorke\ws\ollama-interactive\pyproject.toml`, `C:\Users\yorke\ws\ollama-interactive\ollama_code\cli.py`, Docker, and Compose expose one local CLI with optional tools and Granite as default.

Main controller complexity is concentrated in `C:\Users\yorke\ws\ollama-interactive\ollama_code\agent.py`: about 17.5k lines, 448 methods, and a 3.7k-line `handle_user` method.

Tool execution complexity is concentrated in `C:\Users\yorke\ws\ollama-interactive\ollama_code\tools\__init__.py`: about 14.8k lines, 444 methods, many optional tools, validators, edit tools, and benchmark-oriented synthesis helpers.

Newer direction already exists but is incomplete: `C:\Users\yorke\ws\ollama-interactive\ollama_code\repair_protocol.py` and `C:\Users\yorke\ws\ollama-interactive\ollama_code\controller\` define typed state and navigation helpers, but most product-critical decisions still live in the monolith.

Tests are broad but structurally stale: `C:\Users\yorke\ws\ollama-interactive\tests\test_agent.py` is about 13.8k lines with 433 tests; extracted focused modules mostly import test dictionaries from that omnibus file.

Measurement is rich but fragmented: `local_validation`, `live_model_gate`, `nightly_self_improvement_report`, coding benchmarks, trajectory scripts, and docs all describe readiness differently.

Current dirty work is scoped to the measured `task_due_filter` gap in `agent.py`, `tools/__init__.py`, `tests/test_agent_typed_repair_protocol.py`, and `tests/test_tools.py`.

## Final Endpoint

The product should be a reliable local coding agent for ordinary repo work, not a research benchmark lab or optional-tool showcase.

"Ready to use" means a user can install it, run `--doctor`, ask for a normal feature/fix, get grounded code edits, get requested tests/docs/proof, and receive an honest stop instead of false success when obligations remain.

Controller architecture should become typed and policy-driven: task obligations, patch bundle, allowed next actions, validation plan, proof obligations, and fail-closed state are deterministic objects, while the model proposes content inside those bounds.

Measurement should collapse into one release-style readiness story: setup health, focused unit gates, live model gate, targeted hard feature-delivery gates, and dirty git state.

Research assets, optional integrations, transcript mining, and token micro-optimization stay useful, but only as supporting evidence after the product-readiness contract is green.

## Implementation Plan

Tranche 0: finish and commit the current measured gap before strategy refactors. Fix the temp-directory assertion bug in `tests/test_agent_typed_repair_protocol.py`, rerun focused due-filter tests, rerun the live `task_due_filter` benchmark, run anti-cheat, then commit the dirty due-filter work.

Tranche 1: add `docs/product-strategy.md` as the canonical strategy document. It should define the north star, readiness contract, blocked/non-blocked work rules, current hard gates, and explicit backlog demotions.

Tranche 2: replace `TODO.md` with a short strategy-aligned roadmap. It should list current blocker, current readiness artifact, next controller tranche, and deferred research/tooling backlog, not historical measurement prose.

Tranche 3: add `scripts/product_readiness_report.py`. It should read existing JSON artifacts and produce one PASS/FAIL summary for git dirty state, doctor/setup signal when available, local validation, live model gate, local-small, targeted hard cases, and stale/missing artifacts.

Tranche 4: update README and benchmark docs to point users and contributors to the same readiness path: `--doctor`, `local_validation --tier smoke`, `local_validation --tier agent` for controller work, targeted hard-case benchmark, then product readiness report.

Tranche 5: begin architecture migration only after the readiness report exists. Move feature-delivery policy from `agent.py` into typed controller modules, leaving `agent.py` as adapter/fallback; do not add more ad hoc guards unless they are regression shims for a measured failure.

Tranche 6: split tests by real behavior ownership. Move focused test bodies out of `tests/test_agent.py` into the extracted modules, keep the omnibus file for legacy broad behavior, and keep all tests rather than deleting coverage.

Tranche 7: split `ToolExecutor` by capability family only after controller direction is stable. Start with validators and synthesis helpers because they are the most entangled with feature-delivery reliability.

## Acceptance Tests

Current gap gate passes: targeted due-filter unit tests and live `task_due_filter` on `granite4.1:8b`.

Readiness report has unit tests for green artifacts, dirty git state, missing live gate, failing hard case, and stale artifact timestamps.

`python scripts/local_validation.py --tier smoke` stays green after strategy/report changes.

`python scripts/local_validation.py --tier agent` stays green after controller/test extraction changes.

`python scripts/anti_cheat_scan.py` stays green before reporting benchmark improvements.

No public CLI breaking changes are introduced.

## Assumptions

Preserve all current features and tests.

Do not rewrite existing commits.

Keep `granite4.1:8b` as default unless a fresh live gate changes the documented selection rule.

Do not add new datasets, models, optional tools, or broad heuristics until the product-readiness contract is explicit and green.
