# Trajectory Profiling

`scripts/trajectory_profile.py` profiles downloaded coding-agent trajectory datasets to find workflow patterns that waste tokens and turns.

Supported local datasets under ignored `scratch/external/datasets/`:

- `agent-race-traces`
- `trace-commons-agent-traces`
- `thoughtworks-agentic-coding-trajectories`
- `nebius-swe-agent-trajectories`
- `nebius-swe-rebench-openhands-trajectories`
- `swe-smith-trajectories`
- `nvidia-swe-hero-openhands-trajectories`
- `nvidia-swe-zero-openhands-trajectories`
- `open-swe-traces-openhands`
- `open-swe-traces-sweagent`
- `coderforge-preview-swe-bench-verified-trajectories`
- `cc-bench-trajectories`
- `terminalbench-trajectories`

Catalog-only or manual-review corpora:

- `codetracebench`
- `swe-zero-12m-trajectories`
- `tracebench`
- `swe-chat` (gated)

## Dataset Catalog

Use the catalog script to keep track of local trajectory corpora, public high-value candidates, and gated datasets that may still matter for future analysis:

```bash
python scripts/trajectory_dataset_catalog.py
```

It writes:

- `scratch/external/datasets/trajectory-dataset-catalog.json`

The catalog probes:

- local availability under `scratch/external/datasets/`
- Hugging Face dataset metadata such as gated/public status and file previews
- a small remote schema/row preview for public datasets when available
- high-priority public candidates that are not yet downloaded locally

## Usage

Fetch one supported public dataset locally:

```bash
python scripts/trajectory_dataset_fetch.py --datasets nebius-swe-agent-trajectories
```

Fetch the CoderForge SWE-bench Verified evaluation traces:

```bash
python scripts/trajectory_dataset_fetch.py --datasets coderforge-preview-swe-bench-verified-trajectories
```

Fetch the full supported public set in the current priority order:

```bash
python scripts/trajectory_dataset_fetch.py
```

The fetcher downloads only the allowed Parquet globs for each supported dataset and writes a small manifest at `scratch/external/datasets/<dataset>/.ollama-interactive-manifest.json`. Treat these local files and the generated JSON under `scratch/` as regenerated evidence, not versioned source artifacts.

The current analysis-ready public set includes Agent Race, Trace Commons, Thoughtworks, SWE-agent, OpenHands, SWE-smith, SWE-Hero, SWE-Zero, Open-SWE, CoderForge SWE-bench Verified, CC-Bench, and TerminalBench. TerminalBench uses its own `steps` schema, but the local profile, error, and evidence scripts normalize its tool aliases into the same category model used for the chat-style corpora.

As of June 29, 2026, the repo also has a bounded local download of one additional public corpus that is useful for manual diagnosis even though it is not in the normalized profile/error/evidence pipeline:

- `NJU-LINK/CodeTraceBench`: a human-verified incorrect-step benchmark with preserved assistant actions and observations, useful for studying premature completion claims, fake verification, install thrash, and timeout loops.
- `AlienKevin/SWE-ZERO-12M-trajectories`: a very large execution-free corpus with chat messages and shell-command-only repair attempts, useful for manual review of command discipline and completion-claim efficiency before adding any new adapter.
- `badlogicgames/pi-mono`: public redacted pi coding-agent sessions from real monorepo work, useful for first-turn orientation, tool-choice, and shell-loop review.
- `thomasmustier/pi-mono-sessions`: another public pi-share-hf export from real Pi development work, useful for checking whether the same shell and orientation costs repeat across a different repo and user.
- `thomasmustier/pi-nes-sessions`: public Pi extension sessions with audit, UI, docs, testing, and bugfix work, useful for seeing how coding agents behave on more creative product tasks instead of only issue repair.
- `nmuendler/share-codex`: public exported Codex and Claude Code sessions with prompts, tool calls, and tool outputs from local repo work.
- `peteromallet/my-personal-codex-data`: public DataClaw export of real Codex CLI work with per-session token and tool-use totals.
- `misterkerns/my-personal-claude-code-data`: public DataClaw export of real Claude Code work with per-session token and tool-use totals.
- `ultralazr/claude-code-traces`: public redacted Claude Code session files in native JSONL trace format from `cc-share-hf`.
- `Glint-Research/Fable-5-traces`: public Pi-style converted coding-agent traces intended for manual inspection and distillation work.
- `AlinCiocan/fable-5-claude-code-traces`: public native Claude Code JSONL traces with extra attachment rows that are ignored by the bounded analyzer.
- `vedalken/merchantscroll-traces`: public Cursor JSONL traces with nested `message.role` rows and separate subagent files; the bounded analyzer keeps only main-session files.

Use the normalized set for controller metrics and the catalog-only sets for manual review or future bounded adapters.

To regenerate the bounded analysis for the web-discovered real-user corpora:

```bash
python scripts/web_discovered_agent_dataset_analysis.py
```

Default outputs:

- `scratch/external/datasets/web-discovered-agent-datasets-analysis.json`
- `scratch/external/datasets/web-discovered-agent-datasets-analysis.md`

The current bounded run includes `share-codex`, `pi-mono`, `pi-mono-sessions`, `championswimmer/pi-coding-sessions`, `formal-web/pi-coding-sessions`, `pi-nes-sessions`, `pi-for-excel-sessions`, `pi-extensions-sessions`, `economist-tui-sessions`, `fable-5-traces`, `fable-5-claude-code-traces`, `merchantscroll-traces`, `RangaPrasath/coding-sessions`, `AlexLi31415/coding-sessions`, `ultralazr/claude-code-traces`, `peteromallet/my-personal-codex-data`, `misterkerns/my-personal-claude-code-data`, and `nlile/misc-merged-claude-code-traces-v1`. The latest local sample still shows shell-heavy behavior across the Pi, Claude Code, Cursor, and share-codex sets, but the newer Pi-family exports make the scale easier to see: `championswimmer/pi-coding-sessions` averages `51.33` tool calls per session, `formal-web/pi-coding-sessions` averages `105.08`, `pi-for-excel-sessions` averages `381.08`, and `0xKobolds` averages `187.83`, all dominated by repeated `bash`, `read`, and `edit` actions before convergence. The newer DataClaw exports add a stronger token-efficiency signal: the average session reaches about `157.0M` input tokens with `766.19` tool uses in the personal Codex dump and about `27.5M` input tokens with `146.46` tool uses in the personal Claude Code dump, which works out to roughly `204.8k` to `187.8k` input tokens per tool use on average. The bounded Cursor sample also averages `135.83` tool calls per session, while the native Fable Claude Code sample averages `60.0`. The new bounded `nlile/misc-merged-claude-code-traces-v1` slice now comes through the same analyzer path and shows `54.17` tool calls per accepted row over the first `500` rows, dominated by `Read`, `Grep`, `Bash`, `TodoWrite`, and `Edit` once helper rows and command-output wrappers are filtered. That is direct evidence that repeated long-context turns around tool loops are a larger cost center than the raw number of tools alone.

The analyzer now also supports raw Codex Desktop `.codex/sessions` JSONL exports directly. Bounded local samples such as `nielsr/add-sam-3-lite-text-agent-traces` keep the original developer wrapper plus executed `function_call` events, which makes first-turn instruction bloat and actual tool-loop cost measurable without converting the trace into an intermediate schema first.

The Pi exports still add two concrete efficiency signals that were easy to miss in benchmark-derived corpora: some sessions are just lightweight control chatter such as `hello`, `continue`, or `/load`, and many review or audit requests explicitly ask the model to read large PRs, issues, or architecture context "in full". Those should go through cheaper control handling and summarize-once context compression rather than the same full coding-agent loop used for code-edit tasks.

Recent local CoderForge artifacts:

- `scratch/external/datasets/trajectory-dataset-fetch-coderforge.json`
- `scratch/validation/trajectory-profile-coderforge-500.json`
- `scratch/validation/trajectory-error-profile-coderforge-500.json`
- `scratch/validation/trajectory-evidence-coderforge-500.json`
- `scratch/validation/trajectory-evidence-coderforge-500.md`

The June 29, 2026 CoderForge sample profiled 500 rows at 70.63 average tool calls with 44.4% context-loop rows and 4.8% edit-without-later-test rows. The matching 500-row message evidence sample scanned 71,630 messages and found 14,554 large context blobs, 2,633 large failure blobs, and 1,905 plan-plus-tool-call blobs. It now records shell command families, shapes, and intent directly from tool-call arguments, and the Markdown report cites shell-intent examples so the aggregated counts can be audited against concrete commands. The current top families are `python=9964`, `grep=4749`, `find=1719`, and `git=496`. Intent counts are required because shape-only evidence collapses Python tests, inline probes, and reproduction scripts into similar command families. The next bounded controller tranche should therefore focus on measured context-loop command intents before adding unrelated optional tools or speculative benchmark-specific logic.

Profile a manageable sample from every supported dataset:

```bash
python scripts/trajectory_profile.py --max-rows 2000 --output scratch/external/datasets/trajectory-profile.json
```

Profile one dataset only:

```bash
python scripts/trajectory_profile.py --datasets nebius-swe-rebench-openhands-trajectories --max-rows 5000
```

## Output

The script writes raw JSON and prints a concise summary. Per dataset it reports:

- average tool calls per trajectory
- tool-category mix (`read`, `search`, `edit`, `test`, `shell`, `git`, `other`)
- rows with long read/search loops before progress
- rows with edits but no later test
- rows that edit before enough context
- average and median tool steps before first edit
- top tools and repeated-tool loop signatures
- structured recommendations for controller/tooling changes
- a merged portfolio recommendation list across all profiled datasets

## Why it exists

This profiler is meant to answer questions like:

- Do agents burn too many steps on repeated reads and searches before editing?
- Do they mutate files before gathering enough context?
- How often do they fail to validate edits with tests?
- Which tools dominate trajectories, and which loops should be short-circuited or made deterministic?

Those outputs are intended to drive generic CLI improvements such as:

- symbol-first navigation
- compact context planning
- post-edit validation gates
- failure compression
- loop caps and controller intervention

## Recommendation Layer

Each recommendation now includes:

- `id`: stable key for grouping across datasets
- `priority`: `high`, `medium`, or `low`
- `change_type`: `controller`, `guard`, `tooling`, or `analysis`
- `trigger`: the metric threshold that fired
- `rationale`: why the profiler thinks this change matters
- `expected_effect`: what should improve if the change works
- `experiments`: concrete A/B or benchmark ideas to validate it

The JSON also includes `portfolio_recommendations`, which merges repeated recommendation IDs across datasets so the most general improvements rise to the top.

The first controller implementation is the `trajectory-guards` feature profile. It uses generic signals only: repeated context tools, missing grounding before mutation, missing validation after edits, and failed-test output. After the June 29, 2026 first-failure diagnosis pass, a failed `run_test` followed by another context-only action now triggers `diagnose_test_failure` before more broad reads/searches, so large failure blobs are compressed before the next repair decision instead of waiting for a repeated identical test command. The same pass also lets non-code edits promote available discovered validator commands before final answers, instead of stopping at validator discovery when no test command exists. Requests that say no tests are needed now still keep non-test validation enabled unless validation itself is explicitly disabled.

The follow-up context-loop pass adds a conservative `list_files` narrowing route: if a successful file listing exposes exactly one non-test code file and the model asks for another broad context tool, the controller runs `code_outline` on that file first. This targets repository-tree dump loops found in CoderForge traces without auto-picking a file when multiple implementation candidates are present.

The same shell-heavy context pass normalizes simple inspection commands from model-proposed `run_shell` calls into structured tools: `cat`/`type` becomes `read_file`, `ls`/`dir` becomes `list_files`, two-argument `grep`/`rg` and the measured `grep -n`/`--line-number` shape become `search`, simple `head`/`tail` file previews become bounded `read_file` calls, file-only `find <path> -name <pattern> [-type f]` becomes `file_search`, directory-only `find <path> -name <pattern> -type d` becomes `directory_search`, and the measured `find <path> -name <glob> [-type f] -exec grep -l <query> {} ;` shape becomes `search` with the filename glob preserved. The route applies only when the user did not explicitly request `run_shell` and the command has no unsupported metacharacters or flags. Piped discovery, broad `-exec`, and non-line-number/non-`-l` grep flags remain shell. That makes common CoderForge-style Bash inspection loops cacheable and eligible for the existing context planner without pretending complex shell pipelines are structured searches.

The measured Python command shapes are handled more conservatively. Pytest/unittest-like shell commands now normalize to `run_test` even when no default test command is configured, preserving the original command and cwd/timeout. Exact user requests for `run_shell` still bypass this route. Arbitrary `python script.py` and `python -c ...` snippets remain shell because the CoderForge evidence mixes reproducible checks with exploratory probes.

The same controller tranche now echoes compact validator diagnostics in proactive post-edit validation failure prompts. That keeps lint or contract failures in the immediate repair instruction instead of forcing the model to rediscover the actionable error from prior tool output.

The ground-before-mutate guard now requires explicit path edits to be grounded by evidence for the target path itself. An unrelated prior read, search, or tool-status result is no longer enough to edit `app.py`; the controller first auto-reads or otherwise grounds `app.py`, then retries the mutation. This closes a CoderForge-style edit-without-relevant-context gap without blocking direct creation of new user-named files.

## Error Profile

`scripts/trajectory_error_profile.py` scans only tool results and observations where the dataset exposes them. It avoids task descriptions so benchmark text does not inflate error counts.

```bash
python scripts/trajectory_error_profile.py --output scratch/external/datasets/trajectory-error-profile.json
```

It classifies result failures into generic buckets:

- `test_assertion`
- `missing_dependency`
- `import_error`
- `syntax_error`
- `invalid_args`
- `command_not_found`
- `path_missing`
- `cwd_git`
- `timeout`
- `permission`
- `patch_apply`

Mapped prevention policies:

- `diagnose_test_failure` now emits `error_class`, `missing_dependency`, `next_tool`, expected/actual hints, and likely target files.
- `run_shell` preflights common command families (`git`, Python test runners, `pytest`, `npm`/`pnpm`/`yarn` test commands, `ruff`, `mypy`, `pyright`, `tsc`) and rejects dangerous or malformed recognized commands before execution.
- `run_shell` runs `bash -n -c <command>` when bash is available and the command is not clearly PowerShell, rejecting shell syntax errors before execution.
- Unknown commands that pass shell syntax checking are still rejected before execution when the first executable cannot be resolved and is not a shell/cmd builtin.
- Valid recognized commands run as argv with `shell=False`; unknown commands keep the legacy shell path.
- `trajectory-guards` blocks a third identical tool/error-class failure and forces path discovery, syntax repair, dependency fail-closed, or another non-repeating next step.
- `trajectory-guards` also diagnoses the first failed test before allowing more context-only inspection when an implementation/test task is still unresolved.
- `trajectory-guards` promotes available discovered non-test validator commands after non-code edits when no configured test command is available, and a no-tests request only suppresses test commands rather than all validation.
- `trajectory-guards` promotes a single non-test code file from `list_files` to `code_outline` before another broad context step.
- proactive post-edit validation failure prompts include the compact validator diagnostic before forcing repair.
- explicit path mutations require target-path grounding rather than unrelated grounding evidence.
- Path and cwd failures include nearest existing path suggestions when available.

Latest full local run over downloaded supported datasets:

```text
nebius-swe-agent-trajectories: rows=80036 results=2035588 errors=572350
  test_assertion=166074 syntax_error=146301 command_not_found=72121 invalid_args=46324 path_missing=43823 import_error=43224 timeout=35951 missing_dependency=18135
nebius-swe-rebench-openhands-trajectories: rows=67074 results=4249707 errors=373045
  test_assertion=170744 timeout=74787 invalid_args=34645 missing_dependency=30681 import_error=28785 syntax_error=15757 path_missing=15201 command_not_found=1564
swe-smith-trajectories: rows=102078 results=269242 errors=21980
  test_assertion=9044 import_error=4593 timeout=2341 syntax_error=2141 path_missing=1838 invalid_args=1720 missing_dependency=176 command_not_found=81
portfolio: diagnose-test-failure=345862 syntax-repair-gate=164199 validated-cli-preflight=156455 dependency-or-import-guard=125594 bounded-command-validation=113079 path-repair-guard=60869
```

These counts are directional because public trajectory formats differ, but the top fixes are stable across datasets: better test-failure diagnosis, syntax repair gates, validated CLI execution, dependency/import fail-closed behavior, bounded command handling, and path repair.

Shell-specific full-run counts:

```text
nebius-swe-agent-trajectories:
  other_shell_error=236418 command_not_found=32827 missing_file_or_dir=18475 unrecognized_argument=7232 bash_unexpected_token=6281 bash_unexpected_eof=915
nebius-swe-rebench-openhands-trajectories:
  missing_file_or_dir=12592 bash_unexpected_token=6413 other_shell_error=4689 unrecognized_argument=1811 command_not_found=1187 permission_denied=386
swe-smith-trajectories:
  missing_file_or_dir=1323 other_shell_error=738 unrecognized_argument=287 command_not_found=77 bash_unexpected_token=27 permission_denied=25
```

Mapped shell prevention:

- `bash_unexpected_token`, `bash_unexpected_eof`, and `unmatched_quote`: reject with `bash -n`/quote parsing before execution.
- `command_not_found`: reject missing unknown first executables before execution.
- `missing_file_or_dir`: return nearest path suggestions and block repeated same-path failures under `trajectory-guards`.
- `unrecognized_argument`: keep family-specific argv allowlists for common validators and test runners.
- `permission_denied`: fail closed instead of retrying or changing permissions automatically.
- common observed mistake: Python code pasted directly into bash (`def ...`, `print(module.attr)`) is caught by `bash -n`; the model should use `python -c`, `run_function_probe`, or `run_test` instead.

## Message-Level Evidence Report

`scripts/trajectory_evidence_report.py` scans every raw message in the locally downloaded trajectory datasets, not only extracted tool events. It produces:

- raw JSON with per-dataset message counts, message-theme counts, error-theme counts, row-pattern counts, and example citations
- a Markdown report that maps evidence themes back to current Ollama Interactive fix coverage and remaining gaps

Use it when you need a sale-grade evidence chain for claims such as:

- where tokens are wasted in real coding sessions
- which failure classes dominate across datasets
- which fixes are already implemented, partially implemented, or still missing in the current product

Run it over the full local corpora:

```bash
python scripts/trajectory_evidence_report.py
```

Sample only a bounded subset while iterating on the report:

```bash
python scripts/trajectory_evidence_report.py --max-rows 500
```

Default outputs:

- `scratch/external/datasets/trajectory-evidence-report.json`
- `scratch/external/datasets/trajectory-evidence-report.md`

The report intentionally keeps example citations short and references them by dataset, row id, and message index so findings can be traced back to the original local corpus without copying large raw transcripts into version control.

When `--max-rows` is set, the citation layer is sampled from that bounded slice only. If the default reference JSONs are present, the report also merges full-corpus trajectory/error aggregates so you can cite portfolio-scale counts without rescanning every raw message on each iteration.

## Task Review

`scripts/trajectory_task_review.py` is the focused qualitative layer for "what are people actually trying to do, and where do coding agents waste budget while doing it?" It reuses the local transcript readers, classifies sampled prompts into task families, and joins that with measured per-dataset workflow signals such as context loops and edit-without-test rates.

```bash
python scripts/trajectory_task_review.py
```

It writes:

- `scratch/external/datasets/trajectory-task-review.json`
- `scratch/external/datasets/trajectory-task-review.md`

Use it when you want one bounded artifact that combines:

- task-family mix such as bugfix, refactor, docs, setup-build, frontend, or planning
- representative prompt examples from each family
- measured workflow warnings such as repeated context loops or missing post-edit validation
- ordered controller recommendations that stay grounded in the local transcript mix instead of one benchmark family

Fresh June 29, 2026 bounded reruns that are useful for agent-product tuning:

- `scratch/external/datasets/trajectory-task-review-agentmix-20260629.json`
- `scratch/external/datasets/trajectory-task-review-agentmix-20260629.md`
- `scratch/external/datasets/trajectory-profile-agentmix-20260629.json`
- `scratch/external/datasets/trajectory-evidence-agentmix-20260629.json`
- `scratch/external/datasets/trajectory-evidence-agentmix-20260629.md`
- `scratch/external/datasets/trajectory-evidence-swe-zero-20260629.json`
- `scratch/external/datasets/trajectory-evidence-swe-zero-20260629.md`

That bounded agent-mix pass says the interactive and real-user slices are not just benchmark bugfixes:

- `trace-commons-agent-traces` includes open-ended product work such as project scaffolding, styling, docs, setup, systems work, and data-analysis requests, but still averages `136.73` tool calls with `73.33%` context-loop rows.
- `cc-bench-trajectories` is dominated by `application_development`, `frontend_development`, and `ui_optimization`, and still shows `47.81` average tool calls, `73.0%` context-loop rows, and `91.92%` edit-without-later-test.
- `agent-race-traces` is tiny but useful for cross-harness comparison on the same task; the current local sample shows all four rows editing without prior grounding or later tests, so it is better for harness-discipline checks than broad controller tuning.
- `terminalbench-trajectories` stays shell-heavy and under-validated: `28.52` average tool calls, `53.12%` edit-without-later-test, and command mixes dominated by `run_shell`, `execute_bash`, and `str_replace_editor`.
- `thoughtworks-agentic-coding-trajectories` remains mostly issue-repair, but the bounded rerun still shows `25.0%` context-loop rows and `91.19%` edit-without-later-test, so it reinforces the same controller priorities.

The newly added `nvidia-swe-zero-openhands-trajectories` set is useful as a failure-heavy synthetic comparison slice rather than a real-user one. In the current 120-row evidence sample it shows `100.0%` edit-without-context, `99.17%` edit-without-later-test, `1881` large context blobs, and shell behavior dominated by repeated `grep` and `find ... -exec grep` search patterns. That makes it a good stress test for grounding, post-edit validation, and shell-search compression, but not a substitute for `trace-commons` or `cc-bench` when deciding what real developers ask coding agents to do.

Validation after implementing the generic guards:

```text
python -m unittest discover -v
  301 passed, 3 skipped
scripts/token_efficiency_eval.py --strict-accuracy
  all default cases passed
scripts/verification_eval.py --strict-on
  all strict-on cases passed
scripts/e2e_suite.py --model granite4.1:8b
  all scenarios passed
scripts/coding_benchmark_eval.py --suite local-small --models granite4.1:8b --feature-profiles trajectory-guards --strict-accuracy
  8/8 passed
```

On the local-small Granite run, `trajectory-guards` reduced the live model-token total on key non-zero-LLM cases versus the same-session baseline run: `issue_fix_hidden_tests` 6044 -> 4625 tokens, `multi_file_refactor` 18859 fail-closed -> 11458 pass, and `multi_turn_session_task` 4824 -> 4809 tokens. That older `all`-profile `multi_file_refactor` instability is no longer the current baseline: later June 19, 2026 targeted reruns passed three consecutive times, and the June 20, 2026 serial live gate kept `multi_file_refactor` green on `granite4.1:8b`, `gemma4:e4b`, and `qwen3:8b`. Keep rechecking that case after controller changes, but it is no longer a reason to hold back the current default profile.
